"""Teachers: NVIDIA's released ASR checkpoints, run locally through ``transformers`` (no NeMo).

Two jobs, both from NVIDIA's own playbook:

* **pseudo-labeling** (the Granary pattern): label a large pile of audio with a strong model, then
  distill that into a small student;
* **a real baseline** next to our students.

::

    t = Teacher.from_name("parakeet-tdt-0.6b-v3", device="mps")
    t.transcribe([audio_16k])                     # -> ["He hoped there would be stew ..."]
    t.transcribe([audio_16k], timestamps=True)    # -> [{"text", "words": [{"word", "start", "end"}], "tokens"}]
    pseudo_label("in.jsonl", "out.jsonl", t)      # NeMo manifest: audio_filepath, duration, text

    python -m audioforge.teachers list
    python -m audioforge.teachers transcribe a.flac b.wav --teacher parakeet-ctc-0.6b --timestamps
    python -m audioforge.teachers label <folder | in.jsonl> out.jsonl --teacher parakeet-tdt-0.6b-v3 \
        --wav-dir data/labeled_wav

The transformers feature extractors import ``librosa`` for a single function, the Slaney mel filterbank.
``_patch_librosa`` supplies a numpy copy of it (it matches transformers' own float64 filterbank to 3e-9),
so librosa and its numba/scipy/scikit-learn stack are not needed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import time
import types
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".opus")

# short name -> (HF repo, decoder, license). Other nvidia/* transformers ASR repos also work by full id.
TEACHERS = {
    "parakeet-ctc-0.6b": ("nvidia/parakeet-ctc-0.6b", "ctc", "cc-by-4.0"),
    "parakeet-ctc-1.1b": ("nvidia/parakeet-ctc-1.1b", "ctc", "cc-by-4.0"),
    "parakeet-tdt-0.6b-v3": ("nvidia/parakeet-tdt-0.6b-v3", "tdt", "cc-by-4.0"),
    "nemotron-speech-streaming-en-0.6b": ("nvidia/nemotron-speech-streaming-en-0.6b", "rnnt",
                                          "nvidia-open-model-license"),
}


# --------------------------------------------------------------------------- librosa-free mel
def _hz_to_mel(f):  # Slaney scale: linear below 1 kHz, log above (librosa htk=False)
    f = np.asanyarray(f, dtype=np.float64)
    f_sp, min_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    return np.where(f >= min_hz, min_hz / f_sp + np.log(np.maximum(f, 1e-10) / min_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    f_sp, min_hz, logstep = 200.0 / 3, 1000.0, np.log(6.4) / 27.0
    return np.where(m >= min_hz / f_sp, min_hz * np.exp(logstep * (m - min_hz / f_sp)), f_sp * m)


def slaney_mel(*, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=np.float32):
    """Drop-in for ``librosa.filters.mel`` (htk=False, norm="slaney"), the only librosa call transformers makes."""
    assert not htk and norm == "slaney", "only the Slaney filterbank used by Parakeet/Nemotron is implemented"
    fmax = sr / 2 if fmax is None else fmax
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff, ramps = np.diff(mel_f), np.subtract.outer(mel_f, np.fft.rfftfreq(n=n_fft, d=1.0 / sr))
    w = np.zeros((n_mels, 1 + n_fft // 2), dtype=dtype)
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w


_FEATURE_EXTRACTORS = [("parakeet", "feature_extraction_parakeet", "ParakeetFeatureExtractor"),
                       ("nemotron_asr_streaming", "feature_extraction_nemotron_asr_streaming",
                        "NemotronAsrStreamingFeatureExtractor")]


def _patch_librosa():
    """Without librosa, transformers exports dummy feature-extractor classes; swap the real ones back in."""
    import transformers
    from transformers.utils import is_librosa_available
    if is_librosa_available():
        return
    shim = types.SimpleNamespace(filters=types.SimpleNamespace(mel=slaney_mel))
    for pkg, mod_name, cls in _FEATURE_EXTRACTORS:
        try:
            mod = importlib.import_module(f"transformers.models.{pkg}.{mod_name}")
        except ImportError:
            continue
        if not hasattr(mod, "librosa"):
            mod.librosa = shim
        setattr(importlib.import_module(f"transformers.models.{pkg}"), cls, getattr(mod, cls))
        setattr(transformers, cls, getattr(mod, cls))


# --------------------------------------------------------------------------- audio / text
def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any libsndfile format (wav/flac/ogg/mp3) -> mono float32 at ``sample_rate``."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != sample_rate:
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x), sr, sample_rate).numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


def normalize_text(s: str) -> str:
    """WER normalization: lowercase, hyphens -> spaces, drop punctuation, keep in-word apostrophes."""
    s = re.sub(r"[^\w\s']", " ", s.lower().replace("-", " ")).replace("_", " ")
    return " ".join(w.strip("'") for w in s.split() if w.strip("'"))


def _words(tokens: list[dict]) -> list[dict]:
    """Token pieces (a leading space starts a word) -> word timestamps."""
    words = []
    for t in tokens:
        piece = t["token"]
        if not words or piece[:1].isspace():
            if piece.strip():
                words.append({"word": piece.strip(), "start": t["start"], "end": t["end"]})
        else:
            words[-1]["word"] += piece
            words[-1]["end"] = max(words[-1]["end"], t["end"])
    return words


# --------------------------------------------------------------------------- teacher
class Teacher:
    """A pretrained NVIDIA ASR model (CTC / TDT / RNNT) behind one ``transcribe`` call. Greedy decoding."""

    def __init__(self, name: str, model, processor, kind: str, device: torch.device, license: str = "?"):
        self.name, self.model, self.processor, self.kind = name, model, processor, kind
        self.device, self.license = device, license
        fe = processor.feature_extractor
        sub = getattr(getattr(model.config, "encoder_config", None), "subsampling_factor", 8)
        self.frame_sec = fe.hop_length / fe.sampling_rate * sub  # 0.08 s

    @classmethod
    def from_name(cls, name: str, device="auto", dtype: str = "float32") -> "Teacher":
        """``name``: a key of TEACHERS or a full ``nvidia/...`` repo id. ``device``: auto | mps | cuda | cpu."""
        _patch_librosa()
        from transformers import AutoConfig, AutoModelForCTC, AutoModelForRNNT, AutoModelForTDT, AutoProcessor

        from .train import pick_device
        repo, kind, lic = TEACHERS.get(name, (name, None, "?"))
        if kind is None:
            arch = (AutoConfig.from_pretrained(repo).architectures or [""])[0]
            kind = "ctc" if arch.endswith("CTC") else "tdt" if arch.endswith("TDT") else "rnnt"
        auto = {"ctc": AutoModelForCTC, "tdt": AutoModelForTDT, "rnnt": AutoModelForRNNT}[kind]
        device = pick_device(device) if isinstance(device, str) else device
        model = auto.from_pretrained(repo, dtype=getattr(torch, dtype)).to(device).eval()
        return cls(name, model, AutoProcessor.from_pretrained(repo), kind, device, lic)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def transcribe(self, audios, batch_size: int = 8, timestamps: bool = False) -> list:
        """audios: 1-D float arrays at 16 kHz. -> list[str], or with ``timestamps`` list of
        ``{"text", "words": [{"word", "start", "end"}], "tokens": [{"token", "start", "end"}]}`` (seconds).
        Utterances are batched by length so padding stays small; keep each one under ~5 min
        (full attention; the relative position table covers 5000 frames = 400 s)."""
        order = sorted(range(len(audios)), key=lambda i: len(audios[i]))
        out = [None] * len(audios)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            for i, r in zip(idx, self._run([np.asarray(audios[i], np.float32) for i in idx], timestamps)):
                out[i] = r
        return out

    def _run(self, batch, timestamps):
        inputs = self.processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        gen = self.model.generate(**inputs, return_dict_in_generate=True)
        if self.kind in ("tdt", "rnnt"):
            self._trim_to_encoder_length(gen, inputs["attention_mask"])
        texts = [t.strip() for t in self.processor.batch_decode(gen.sequences, skip_special_tokens=True)]
        if not timestamps:
            return texts
        if self.kind == "ctc":
            toks = [self._ctc_tokens(seq) for seq in gen.sequences.cpu()]
        else:  # TDT/RNNT: per-step durations -> token spans, same post-processing as NeMo
            _, toks = self.processor.decode(gen.sequences.cpu(), durations=gen.durations.cpu(),
                                            skip_special_tokens=True)
            toks = [[dict(k, start=round(k["start"], 3), end=round(k["end"], 3)) for k in ks] for ks in toks]
        return [{"text": t, "words": _words(k), "tokens": k} for t, k in zip(texts, toks)]

    def _trim_to_encoder_length(self, gen, attention_mask):
        """Blank out steps taken at or past each row's own last encoder frame. HF generate only pads finished
        rows when the generation config has an ``eos_token_id`` (TDT does, Nemotron RNNT doesn't), so a short
        clip batched with a long one otherwise keeps emitting tokens over the padding."""
        enc_len = self.model.encoder._get_subsampling_output_length(attention_mask.sum(-1)).to(gen.durations.device)
        start = gen.durations.cumsum(-1) - gen.durations  # encoder frame each step was taken at
        past = start >= enc_len[:, None]
        gen.sequences[past.to(gen.sequences.device)] = self.processor.tokenizer.pad_token_id
        gen.durations[past] = 0

    def _ctc_tokens(self, frame_ids: torch.Tensor) -> list[dict]:
        """Frame-level CTC argmax -> token spans (runs of one non-blank id)."""
        tok, blank = self.processor.tokenizer, self.model.config.pad_token_id
        spans, prev = [], blank
        for t, i in enumerate(frame_ids.tolist()):
            if i != blank and i == prev:
                spans[-1][2] = t + 1
            elif i != blank:
                spans.append([i, t, t + 1])
            prev = i
        pieces = tok.convert_ids_to_tokens([s[0] for s in spans])
        return [{"token": p.replace("▁", " "), "start": round(a * self.frame_sec, 3),
                 "end": round(b * self.frame_sec, 3)} for p, (_, a, b) in zip(pieces, spans)]


# --------------------------------------------------------------------------- pseudo-labeling
def manifest_from_folder(folder: str | Path, exts=AUDIO_EXTS) -> list[dict]:
    return [{"audio_filepath": str(p)} for p in sorted(Path(folder).rglob("*"))
            if p.suffix.lower() in exts and p.is_file()]


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _load_segment(e: dict, cache: dict) -> np.ndarray:
    """The audio of one manifest entry: NeMo ``offset`` / ``duration`` (seconds) select a segment of the file.
    ``cache`` keeps the last decoded file, so consecutive segments of one recording decode it once."""
    path = e["audio_filepath"]
    if cache.get("path") != path:
        cache.update(path=path, audio=load_audio(path))
    a = cache["audio"]
    off, dur = float(e.get("offset") or 0.0), e.get("duration")
    lo = int(round(off * SAMPLE_RATE))
    hi = len(a) if dur is None else int(round((off + float(dur)) * SAMPLE_RATE))
    if hi >= len(a) - SAMPLE_RATE // 1000:  # a (rounded) full-length duration keeps the whole tail
        hi = len(a)
    return a[lo:hi] if lo or hi < len(a) else a


def pseudo_label(manifest_in: str | Path, manifest_out: str | Path, teacher: Teacher, batch_size: int = 8,
                 wav_dir: str | Path | None = None, normalize: bool = False, timestamps: bool = False,
                 chunk: int = 64, log_every: int = 10) -> dict:
    """Label every entry of a NeMo JSONL manifest (or every audio file under a folder) with ``teacher``.

    Each output line keeps the input keys and sets ``audio_filepath``, ``duration`` and ``text`` (the
    teacher transcript; an existing ``text`` is kept as ``ref_text``), plus ``teacher`` and, with
    ``timestamps``, ``words``. ``normalize`` lowercases and strips punctuation (TDT-v3 and Nemotron emit
    punctuation + casing, the CTC models don't). ``read_manifest`` in audioforge.data falls back to a WAV-only
    stdlib reader without ``soundfile``, so pass ``wav_dir`` for FLAC/MP3 sources: each file is written there as 16 kHz mono WAV and
    ``audio_filepath`` points at the copy (a unique name that never overwrites a file already there).
    NeMo segment entries (``offset`` + ``duration``) are labeled from their segment only; with ``wav_dir``
    the copy holds just the segment and ``offset`` is dropped. Lines are written as they are labeled.
    """
    from .data import save_wav
    src = Path(manifest_in)
    entries = manifest_from_folder(src) if src.is_dir() else _read_jsonl(src)
    if wav_dir is not None:
        wav_dir = Path(wav_dir)
        wav_dir.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    # output WAV names, lowercased (APFS/NTFS are case-insensitive), including files already in wav_dir
    used = {p.stem.lower() for p in wav_dir.glob("*.wav")} if wav_dir is not None else set()
    t0, secs, cache = time.time(), 0.0, {}
    with open(manifest_out, "w") as f:
        for s in range(0, len(entries), chunk):
            group = entries[s: s + chunk]
            audios = [_load_segment(e, cache) for e in group]
            hyps = teacher.transcribe(audios, batch_size=batch_size, timestamps=timestamps)
            for e, a, h in zip(group, audios, hyps):
                out = dict(e)
                if "text" in e:
                    out["ref_text"] = e["text"]
                if wav_dir is not None:
                    stem = name = Path(e["audio_filepath"]).stem
                    k = 1
                    while name.lower() in used:  # repeated names across folders / an earlier run
                        name, k = f"{stem}_{k}", k + 1
                    used.add(name.lower())
                    out["audio_filepath"] = str(wav_dir / f"{name}.wav")
                    out.pop("offset", None)  # the copy holds just the segment
                    save_wav(out["audio_filepath"], a)
                text = h["text"] if timestamps else h
                out.update(duration=round(len(a) / SAMPLE_RATE, 3),
                           text=normalize_text(text) if normalize else text, teacher=teacher.name)
                if timestamps:
                    out["words"] = h["words"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                secs += len(a) / SAMPLE_RATE
            f.flush()
            if log_every and (s // chunk) % log_every == 0:
                log.info(f"  {min(s + chunk, len(entries))}/{len(entries)} files, {secs / 3600:.2f} h audio, "
                      f"RTFx {secs / (time.time() - t0):.1f}")
    wall = time.time() - t0
    return {"files": len(entries), "audio_hours": secs / 3600, "wall_sec": wall, "rtfx": secs / max(wall, 1e-9)}


# --------------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m audioforge.teachers", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("transcribe", "label"):
        s = sub.add_parser(name)
        s.add_argument("--teacher", default="parakeet-tdt-0.6b-v3")
        s.add_argument("--device", default="auto")
        s.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
        s.add_argument("--batch-size", type=int, default=8)
        s.add_argument("--timestamps", action="store_true")
    sub.choices["transcribe"].add_argument("audio", nargs="+")
    lab = sub.choices["label"]
    lab.add_argument("src", help="NeMo JSONL manifest or a folder of audio")
    lab.add_argument("out", help="output JSONL manifest")
    lab.add_argument("--wav-dir", default=None, help="write 16 kHz WAV copies here (needed for FLAC/MP3)")
    lab.add_argument("--normalize", action="store_true", help="lowercase + strip punctuation")
    a = p.parse_args(argv)
    if a.cmd == "list":
        for k, (repo, kind, lic) in TEACHERS.items():
            print(f"{k:<36} {repo:<44} {kind:<5} {lic}")
        return
    t = Teacher.from_name(a.teacher, a.device, a.dtype)
    print(f"{t.name}: {t.num_params / 1e6:.0f}M params, {t.kind}, {t.device}, {t.model.dtype}", flush=True)
    if a.cmd == "transcribe":
        for path, r in zip(a.audio, t.transcribe([load_audio(x) for x in a.audio], a.batch_size, a.timestamps)):
            print(path, "->", json.dumps(r, ensure_ascii=False) if a.timestamps else r)
    else:
        stats = pseudo_label(a.src, a.out, t, a.batch_size, a.wav_dir, a.normalize, a.timestamps)
        print(json.dumps(stats))


if __name__ == "__main__":
    main()
