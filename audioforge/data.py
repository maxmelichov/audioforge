"""Data: NeMo-style JSONL manifests + a synthetic "tone language" generator.

Manifest lines (same keys NeMo uses)::

    {"audio_filepath": "a.wav", "duration": 3.2, "text": "hello world",
     "source_lang": "en", "target_lang": "en", "task": "asr", "speaker": 12,
     "rttm_filepath": "a.rttm"}

The synthetic generator makes learnable audio for every head (ASR, AST
prompts, VAD, EOU, diarization, speaker ID, codec tokens) so recipes can be
verified end to end on a laptop in minutes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import warnings
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

FRAME_SEC = 0.08  # FastConformer output frame


# --------------------------------------------------------------------------- audio io
def _resample(x: np.ndarray, sr: int, sample_rate: int) -> np.ndarray:
    if sr == sample_rate:
        return x
    # linear interpolation; use a proper resampler for production data
    n = int(round(len(x) * sample_rate / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def load_wav(path: str, sample_rate: int = 16000, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Mono float32 at ``sample_rate``. ``offset``/``duration`` (seconds) select a segment, as in NeMo manifests."""
    try:  # every format (WAV PCM 8/16/24/32, float WAV, FLAC, OGG, ...) via libsndfile, like teachers.load_audio
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(1)
    else:  # stdlib fallback: integer PCM WAV only
        with wave.open(str(path)) as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        raw = raw[: len(raw) // (sw * ch) * sw * ch]  # drop a trailing partial frame (truncated file)
        if sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 2 ** 23
        else:
            x = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
            x = (x - 128) / 128 if sw == 1 else x / float(2 ** (8 * sw - 1))
        if ch > 1:
            x = x.reshape(-1, ch).mean(1)
    x = _resample(np.asarray(x, np.float32), sr, sample_rate)
    start = int(round((offset or 0.0) * sample_rate))
    if duration is not None and duration > 0:
        return x[start: start + int(round(duration * sample_rate))]
    return x[start:]


def save_wav(path: str, x: np.ndarray, sample_rate: int = 16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _limit_spks(y: np.ndarray, max_spks: int, src) -> np.ndarray:
    """Keep the first ``max_spks`` speaker columns (arrival order), warning if that drops speech."""
    if y.shape[1] > max_spks and y[:, max_spks:].any():
        lost = float(y[:, max_spks:].any(1).sum() - (y[:, max_spks:].any(1) & y[:, :max_spks].any(1)).sum())
        warnings.warn(f"{src}: {y.shape[1]} speakers but only {max_spks} slots; speakers after the "
                      f"{max_spks}th are dropped ({lost / max(1.0, float(y.any(1).sum())):.1%} of speech frames "
                      f"become silence)", UserWarning, stacklevel=2)
    y = y[:, :max_spks]
    return np.pad(y, ((0, 0), (0, max_spks - y.shape[1]))) if y.shape[1] < max_spks else y


def rttm_to_frames(path: str, n_frames: int, speakers: list[str] | None = None, frame_sec=FRAME_SEC,
                   max_spks: int | None = None, uniq_id: str | None = None, offset: float = 0.0) -> np.ndarray:
    """RTTM -> (n_frames, S) speaker activity, columns in arrival order.

    ``max_spks=None`` keeps every speaker (at least 4 columns); an int crops to that many (with a warning if
    speech is dropped). A shared RTTM (one file for many recordings) is split by its file-id field: only lines
    whose id equals ``uniq_id`` are used (a single-recording RTTM with a different id is used as is).
    ``offset`` (seconds) shifts times for a manifest segment that starts ``offset`` into the recording.
    """
    lines = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        p = line.split()
        if p and p[0] == "SPEAKER":
            if len(p) < 8:
                raise ValueError(f"{path}:{i}: malformed RTTM SPEAKER line (needs >= 8 fields): {line!r}")
            lines.append(p)
    if uniq_id is not None and lines:
        own = [p for p in lines if p[1] == uniq_id]
        ids = {p[1] for p in lines}
        if not own and len(ids) > 1:
            raise ValueError(f"{path}: no RTTM lines for file id {uniq_id!r} (file has ids {sorted(ids)[:5]}...)")
        lines = own or lines
    segs = [(float(p[3]) - offset, float(p[4]), p[7]) for p in lines]
    spk = speakers or sorted({s for _, _, s in segs}, key=lambda s: min(a for a, _, x in segs if x == s))
    y = np.zeros((n_frames, max(4, len(spk))), np.float32)
    for start, dur, s in segs:
        if s in spk:
            y[max(0, int(start / frame_sec)): max(0, int(math.ceil((start + dur) / frame_sec))), spk.index(s)] = 1
    return y if max_spks is None else _limit_spks(y, max_spks, path)


# --------------------------------------------------------------------------- prompts
def canary_prompt(source_lang="en", target_lang=None, task="asr", pnc=True) -> str:
    """Canary-style task prompt: <|src|><|task|><|tgt|><|pnc|>."""
    tgt = target_lang or source_lang
    return f"<|{source_lang}|><|{'transcribe' if task == 'asr' else 'translate'}|><|{tgt}|><|{'pnc' if pnc else 'nopnc'}|>"


def prompt_specials(langs) -> list[str]:
    s = ["<|transcribe|>", "<|translate|>", "<|pnc|>", "<|nopnc|>", "<|audio|>"]
    return s + [f"<|{l}|>" for l in langs]


# --------------------------------------------------------------------------- synthetic
class ToneLanguage:
    """Each character is a harmonic tone; speakers differ in pitch scale and timbre."""

    LEXICON = ["go", "stop", "left", "right", "yes", "no", "up", "down", "on", "off", "play", "call",
               "open", "red", "blue", "green", "one", "two", "six", "ten", "home", "note", "time", "set"]

    def __init__(self, sample_rate=16000, n_speakers=8, seed=0, char_ms=(70, 120), noise=(0.002, 0.02)):
        self.sr, self.char_ms, self.noise = sample_rate, char_ms, noise
        rng = np.random.default_rng(seed)
        self.chars = sorted(set("".join(self.LEXICON)))
        self.freq = {c: 220 * 2 ** (i / 4.5) for i, c in enumerate(self.chars)}  # distinct pitches
        self.speakers = [dict(scale=float(rng.uniform(0.93, 1.07)), harm=rng.dirichlet(np.ones(4)) + 0.05)
                         for _ in range(n_speakers)]

    def sentence(self, rng: random.Random, n=(2, 4)) -> str:
        return " ".join(rng.choice(self.LEXICON) for _ in range(rng.randint(*n)))

    def render(self, text: str, speaker: int, rng: random.Random):
        """-> audio (float32), per-sample activity mask."""
        spk, sr, pieces, act = self.speakers[speaker], self.sr, [], []
        for c in text:
            n = int(sr * rng.uniform(*self.char_ms) / 1000)
            if c == " ":
                pieces.append(np.zeros(n, np.float32))
                act.append(np.zeros(n, bool))
                continue
            t = np.arange(n) / sr
            f = self.freq[c] * spk["scale"]
            x = sum(h * np.sin(2 * np.pi * f * (k + 1) * t) for k, h in enumerate(spk["harm"]))
            env = np.minimum(1, np.minimum(t, t[::-1]) / 0.008)
            pieces.append((0.3 * x * env).astype(np.float32))
            act.append(np.ones(n, bool))
        return np.concatenate(pieces), np.concatenate(act)

    def frames(self, act: np.ndarray, n_frames: int) -> np.ndarray:
        hop = int(self.sr * FRAME_SEC)
        a = np.pad(act, (0, max(0, n_frames * hop - len(act))))[: n_frames * hop]
        return (a.reshape(n_frames, hop).mean(1) > 0.5).astype(np.float32)

    @staticmethod
    def n_frames(n_samples: int) -> int:
        n_mel = n_samples // 160 + 1
        for _ in range(3):
            n_mel = (n_mel + 1) // 2
        return n_mel

    def example(self, rng: random.Random, speaker=None, task="asr", lead=(0.1, 0.4), tail=(0.3, 0.6)) -> dict:
        speaker = rng.randrange(len(self.speakers)) if speaker is None else speaker
        text = self.sentence(rng)
        x, act = self.render(text, speaker, rng)
        a, b = (np.zeros(int(self.sr * rng.uniform(*r)), np.float32) for r in (lead, tail))
        x = np.concatenate([a, x, b])
        act = np.concatenate([np.zeros(len(a), bool), act, np.zeros(len(b), bool)])
        x = x + rng.uniform(*self.noise) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(x)).astype(np.float32)
        T = self.n_frames(len(x))
        vad = self.frames(act, T)
        eou = np.zeros(T, np.float32)
        last = int(np.nonzero(vad)[0].max()) if vad.any() else 0
        eou[last + 1: last + 3] = 1  # end-of-utterance fires right after speech ends
        tgt = text if task == "asr" else " ".join(text.split()[::-1])  # "translation" = reversed order
        return dict(audio=x, text=tgt, source_text=text, speaker=speaker, vad=vad, eou=eou,
                    prompt=canary_prompt("en", "en" if task == "asr" else "xx", task))

    def mixture(self, rng: random.Random, n_spk=2) -> dict:
        """Multi-speaker conversation with overlap; per-speaker transcripts and activity."""
        spks = rng.sample(range(len(self.speakers)), n_spk)
        parts, t0 = [], 0
        for s in spks:
            text = self.sentence(rng, (1, 3))
            x, act = self.render(text, s, rng)
            start = t0 + int(self.sr * rng.uniform(0.1, 0.3))
            parts.append((s, text, x, act, start))
            t0 = start + int(len(x) * rng.uniform(0.6, 1.0))  # overlap up to 40%
        N = max(st + len(x) for _, _, x, _, st in parts) + int(0.3 * self.sr)
        mix = np.zeros(N, np.float32)
        T = self.n_frames(N)
        spk_t = np.zeros((T, 4), np.float32)
        items = []
        for i, (s, text, x, act, st) in enumerate(parts):
            mix[st: st + len(x)] += x
            full = np.zeros(N, bool)
            full[st: st + len(x)] = act
            spk_t[:, i] = self.frames(full, T)
            items.append(dict(speaker=s, text=text))
        mix += 0.005 * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(N).astype(np.float32)
        return dict(audio=mix, spk_targets=spk_t, speakers=items)


def synthetic_dataset(kind: str, n: int, seed: int = 0, lang: ToneLanguage | None = None, codec=None) -> list[dict]:
    """kind: asr | multitask | ast | diar | speaker_attributed | enhance."""
    lang = lang or ToneLanguage(seed=0)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if kind in ("asr", "multitask"):
            out.append(lang.example(rng))
        elif kind == "ast":
            out.append(lang.example(rng, task=rng.choice(["asr", "ast"])))
        elif kind == "diar":
            out.append(lang.mixture(rng))
        elif kind == "speaker_attributed":
            m = lang.mixture(rng)
            i = rng.randrange(len(m["speakers"]))
            out.append(dict(audio=m["audio"], spk_targets=m["spk_targets"], spk_act=m["spk_targets"][:, i],
                            text=m["speakers"][i]["text"]))
        elif kind == "enhance":
            ex = lang.example(rng)
            clean = ex["audio"]
            noisy = clean + rng.uniform(0.05, 0.2) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(clean)).astype(np.float32)
            with torch.no_grad():
                codes = codec.encode(torch.from_numpy(clean)[None])[0]
            out.append(dict(audio=noisy, clean=clean, codes=codes.numpy()))
        else:
            raise ValueError(kind)
    return out


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _decode_row(m: dict, sample_rate: int, max_spks: int, any_prompt: bool) -> dict:
    """One manifest line -> one example: audio decoded now (on the fly), prompt / RTTM targets derived, and the
    ``labels_file`` npz sidecar written by write_manifest merged back in (frame labels, embeddings)."""
    seg = {"offset": m.get("offset") or 0.0, "duration": m.get("duration")} if "offset" in m else {}
    ex = dict(m, audio=load_wav(m["audio_filepath"], sample_rate, **seg))
    if any_prompt:
        ex["prompt"] = canary_prompt(m.get("source_lang", "en"), m.get("target_lang"), m.get("task", "asr"))
    if "rttm_filepath" in m:
        n = ToneLanguage.n_frames(len(ex["audio"]))
        full = rttm_to_frames(m["rttm_filepath"], n, uniq_id=m.get("uniq_id") or Path(m["audio_filepath"]).stem,
                              offset=seg.get("offset", 0.0))
        ex["spk_targets"] = _limit_spks(full, max_spks, m["rttm_filepath"])
        ex["spk_targets_full"] = full
    if "labels_file" in m:
        with np.load(m["labels_file"], allow_pickle=False) as z:
            ex.update({k: z[k] for k in z.files})
        del ex["labels_file"]
    return ex


def read_manifest(path: str, sample_rate: int = 16000, max_spks: int = 4) -> list[dict]:
    """NeMo JSONL manifest -> examples (all decoded now; the trainer uses the lazy ManifestData instead).

    ``offset``/``duration`` select a segment (duration alone is treated as informational and the whole file is
    read). If any line has ``task``/``target_lang`` every line gets a Canary prompt (plain ASR lines get the
    default en/asr one). RTTM targets: ``spk_targets`` is cropped/padded to ``max_spks`` arrival-ordered slots
    for training; ``spk_targets_full`` keeps every speaker for evaluation.
    """
    return list(ManifestData(path, sample_rate, max_spks))


class ManifestData(Sequence):
    """The trainer's one data format: a JSONL manifest read lazily. Each line is
    ``{"audio_filepath", "duration", "text"?, <any label>?, "labels_file"?: <npz of array labels>}``; audio is
    decoded on access, so training starts immediately. Items equal read_manifest's. ``durations`` (seconds, from
    the manifest, no decoding) feed the duration-bucketing sampler (epoch_batches)."""

    def __init__(self, path=None, sample_rate: int = 16000, max_spks: int = 4, rows: list[dict] | None = None):
        self.rows = _read_jsonl(path) if rows is None else rows
        self.sample_rate, self.max_spks = sample_rate, max_spks
        self.any_prompt = any("task" in m or "target_lang" in m for m in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return _decode_row(self.rows[i], self.sample_rate, self.max_spks, self.any_prompt)

    @property
    def durations(self) -> list[float]:
        return [float(m.get("duration") or 0.0) for m in self.rows]


def row_key(m: dict) -> str:
    """Identity of a manifest row's audio (file + segment): the unit a split must never share between sides."""
    return f"{m['audio_filepath']}@{float(m.get('offset') or 0.0):.3f}+{m.get('duration')}"


def seeded_split(rows: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Deterministic train / val split of one manifest: a row goes to val iff sha1(seed:file) falls under
    ``val_fraction``. Keyed by audio file (not segment), so segments of one recording never straddle the split."""
    def u(m):
        return int(hashlib.sha1(f"{seed}:{m['audio_filepath']}".encode()).hexdigest()[:12], 16) / 16 ** 12
    return [m for m in rows if u(m) >= val_fraction], [m for m in rows if u(m) < val_fraction]


def check_disjoint(train_rows: list[dict], val_rows: list[dict], what: str = "manifest") -> int:
    """Warn (count) val rows whose audio segment is also a train row: an eval leak."""
    seen = {row_key(m) for m in train_rows}
    leak = sum(row_key(m) in seen for m in val_rows)
    if leak:
        log.warning(f"[data] {what}: {leak} of {len(val_rows)} val rows are also train rows (eval leak)")
    return leak


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def write_manifest(rows, out_dir, split: str, sample_rate: int = 16000) -> Path:
    """Adapter: any in-memory examples (a dataset source's rows) -> ``<out_dir>/<split>.jsonl`` in the trainer's
    manifest format. Audio is written as float32 WAV (lossless, so ManifestData gives back the same arrays);
    numeric array labels go to one npz per row (``labels_file``); text and scalars stay inline."""
    import soundfile as sf
    out_dir = Path(out_dir).resolve()
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)
    lines = []
    for i in range(len(rows)):
        ex = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in rows[i].items()}
        wav = out_dir / "audio" / f"{split}_{i:07d}.wav"
        audio = np.asarray(ex["audio"], np.float32)
        sf.write(str(wav), audio, sample_rate, subtype="FLOAT")
        m = {"audio_filepath": str(wav), "duration": round(len(audio) / sample_rate, 4)}
        arrays = {k: v for k, v in ex.items() if k != "audio" and isinstance(v, np.ndarray) and v.dtype.kind in "biuf"}
        for k, v in ex.items():
            if k in ("audio", "audio_filepath", "duration") or k in arrays:
                continue
            m["source_" + k if k in ("offset", "rttm_filepath", "labels_file") else k] = _jsonable(v)
        if arrays:
            npz = out_dir / "labels" / f"{split}_{i:07d}.npz"
            np.savez(npz, **arrays)
            m["labels_file"] = str(npz)
        lines.append(json.dumps(m))
    path = out_dir / f"{split}.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- sampler
def epoch_batches(n: int, bs: int, rng: random.Random, source_ids=None, batching: str = "source",
                  durations=None, bucket: int = 0, names=None) -> list[list[int]]:
    """Index batches for one epoch (full batches only), drawn from ``rng`` (the caller's seeded sampler RNG, so the
    order is a function of seed + epoch). ``source_ids`` with batching "source" gives single-source batches
    (sources smaller than bs contribute none). ``bucket`` k > 0 with ``durations``: each shuffled window of k*bs
    items is sorted by duration before slicing, then the batches are shuffled (less padding, same epoch)."""
    def cut(idx):
        if bucket and durations is not None:
            w = bs * int(bucket)
            idx = [j for s in range(0, len(idx), w) for j in sorted(idx[s: s + w], key=lambda j: durations[j])]
            out = [idx[i: i + bs] for i in range(0, len(idx) - bs + 1, bs)]
            rng.shuffle(out)
            return out
        return [idx[i: i + bs] for i in range(0, len(idx) - bs + 1, bs)]

    if source_ids is None or batching != "source":
        idx = list(range(n))
        rng.shuffle(idx)
        return cut(idx)
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(source_ids):
        groups.setdefault(s, []).append(i)
    out = []
    for s, g in sorted(groups.items()):
        if len(g) < bs:
            log.debug(f"[data] source {names[s] if names else s!r} has {len(g)} < batch_size items: never batched")
        rng.shuffle(g)
        out += cut(g)
    if not out:
        raise ValueError(f"no data source has >= trainer.batch_size={bs} items")
    rng.shuffle(out)
    return out


def shard_batches(batches: list, rank: int, world: int) -> list:
    """This process's share of an epoch under DDP: every world-th batch, equal count on every rank."""
    if world <= 1:
        return batches
    return batches[rank::world][: len(batches) // world]


# --------------------------------------------------------------------------- batching
class Collate:
    def __init__(self, tokenizer=None):
        self.tok = tokenizer

    def _ids(self, batch, key):
        ids = [self.tok.encode(ex[key]) for ex in batch]
        L = max(1, max(map(len, ids)))
        t = torch.zeros(len(ids), L, dtype=torch.long)
        for i, x in enumerate(ids):
            t[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        return t, torch.tensor([len(x) for x in ids])

    @staticmethod
    def _frames(batch, key):
        arrs = [np.asarray(ex[key]) for ex in batch]
        shape = np.max([a.shape for a in arrs], 0)  # pad time and any trailing dim (e.g. speaker columns)
        t = torch.zeros((len(arrs), *shape), dtype=torch.from_numpy(arrs[0]).dtype)
        for i, a in enumerate(arrs):
            t[(i, *(slice(0, d) for d in a.shape))] = torch.from_numpy(a)
        return t, torch.tensor([len(a) for a in arrs])

    def __call__(self, batch: list[dict]) -> dict:
        lens = torch.tensor([len(ex["audio"]) for ex in batch])
        audio = torch.zeros(len(batch), int(lens.max()))
        for i, ex in enumerate(batch):
            audio[i, : len(ex["audio"])] = torch.from_numpy(np.asarray(ex["audio"], np.float32))
        out = {"audio": audio, "audio_len": lens}
        keys = set().union(*(ex.keys() for ex in batch))
        frame_keys = ["vad", "eou", "spk_targets", "spk_act", "codes"]
        # any other numeric 1-D/2-D array is collated too (FrameHead with a custom key, spk_targets_full, ...)
        frame_keys += sorted(k for k in keys - set(frame_keys) - {"audio", "text", "source_text", "prompt", "speaker"}
                             if all(isinstance(ex.get(k), np.ndarray) and ex[k].ndim in (1, 2)
                                    and ex[k].dtype.kind in "biuf" for ex in batch if k in ex))
        used = frame_keys + ["speaker"] + (["text", "source_text", "prompt"] if self.tok is not None else [])
        for k in used:
            if k in keys and not all(k in ex for ex in batch):
                raise ValueError(f"Collate: key {k!r} present in only some batch items")
        if "text" in keys and self.tok is not None:
            out["text"], out["text_len"] = self._ids(batch, "text")
        if "source_text" in keys and self.tok is not None:  # transcript for auxiliary CTC in AST
            out["source_text"], out["source_text_len"] = self._ids(batch, "source_text")
        if "prompt" in keys and self.tok is not None:
            out["prompt"], out["prompt_len"] = self._ids(batch, "prompt")
        for k in frame_keys:
            if k in keys:
                out[k], out[k + "_len"] = self._frames(batch, k)
        if "speaker" in keys:
            out["speaker"] = torch.tensor([ex["speaker"] for ex in batch])
        return out


def segment_id(ex: dict) -> str:
    """Stable id of a single-speaker example across processes: LibriSpeech's ``id``, else
    ``meeting:start:speaker:n_samples`` (AMI / ICSI asr-mode segments). Used to key cached teacher embeddings."""
    if ex.get("id") is not None:
        return str(ex["id"])
    return f"{ex['meeting']}:{float(ex['start']):.3f}:{int(ex.get('speaker', -1))}:{len(ex['audio'])}"


def load_teacher_cache(path) -> dict[str, np.ndarray]:
    """Teacher-embedding cache archive (ids: str array, emb: (N, D) float32) -> {id: embedding}."""
    z = np.load(path, allow_pickle=False)
    ids, emb = z["ids"], z["emb"]
    if emb.ndim == 3 and "lens" in z:  # per-frame teacher posteriors (N, T_max, S) + lens (diarization distillation)
        return {str(i): e[: int(n)] for i, e, n in zip(ids.tolist(), emb, z["lens"])}
    return {str(i): e for i, e in zip(ids.tolist(), emb)}


def attach_teacher(data: list[dict], spec, split: str = "train") -> list[dict]:
    """``data.spk_teacher: <npz path>`` or ``{file, key: spk_teacher, missing: drop | error}``: attach a cached
    teacher embedding per example under ``key`` (a (D,) float32 array, collated to (B, D)). ``file`` may contain
    ``{split}``. Examples without a cached embedding are dropped (default) or raise. Returns a new list."""
    spec = {"file": spec} if isinstance(spec, (str, Path)) else dict(spec)
    key, missing = spec.get("key", "spk_teacher"), spec.get("missing", "drop")
    path = Path(str(spec["file"]).format(split=split))
    if not path.exists():
        raise FileNotFoundError(f"spk_teacher: {path} missing (cache the teacher embeddings first)")
    cache = load_teacher_cache(path)
    out, lost = [], 0
    for ex in data:
        e = cache.get(segment_id(ex))
        if e is None:
            lost += 1
            if missing == "error":
                raise KeyError(f"spk_teacher: no embedding for {segment_id(ex)} in {path}")
            continue
        out.append({**ex, key: np.asarray(e, np.float32)})
    log.info(f"[teacher] {path.name}: attached {key} to {len(out)} of {len(data)} examples"
          + (f" ({lost} without embedding dropped)" if lost else ""))
    return out


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
