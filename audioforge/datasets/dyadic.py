"""Dyadic (two-party) conversations as audioforge list-of-dict data, with the SAME label rules and example keys as
``ami.py`` (turn windows, floor definition, backchannels, hesitations; ``Dyadic`` subclasses ``ami.AMI``) plus the
agent-side streams a voice agent really has: the other party's activity and the moments it stops speaking.

Corpora (two channels, one party per channel; a prepare script downloads the slices):
  behavior_sd  Behavior-SD (Lee, Kim & Kim, NAACL 2025), CC BY 4.0 (code MIT), ungated. 108 K SYNTHETIC full-duplex
               dialogues (CosyVoice TTS, 52 voices, 2,164 h) as HF tar shards of <id>.flac (stereo 22.05 kHz, channel
               k = speaker k: ``channel_check``) + <id>.json. Labels: per-UTTERANCE start / end (exact: the TTS
               timeline), explicit backchannels (rendered on the LISTENER's channel, verified by energy) and
               interruption types; no word timings. Each utterance's interval is spread over its whitespace tokens by
               character length (``word_timing = interpolated``), so pauses inside one TTS utterance are invisible
               and the hesitation label is (almost) always 0. Speaker ids: 2000 + hash of the TTS voice id.
  dailytalk    kyutai/DailyTalkContiguous, CC BY-SA 4.0, ungated: the 2,541 ACTED DailyTalk dialogues (2 actors,
               ~21.7 h) as one stereo 44.1 kHz WAV per dialogue (one actor per channel) + word alignments. Every
               word carries the tag SPEAKER_MAIN, so the party of a word is the channel with more energy over the
               word (``word_timing = aligned``); alternating turns with ~1-2.5 s gaps, no overlap. Speaker ids:
               3000 + channel (which actor sits on which channel is not published).
  oto          otoSpeech-full-duplex-processed-141h, CC BY 4.0, gated. REAL remote two-party conversations (~18 min,
               44.1 kHz stereo FLAC in WebDataset tars, denoised, redaction intervals, no transcripts). The card
               prohibits attempts to identify / de-anonymise speakers, so NO speaker-identity supervision is derived
               from it: ``speaker`` = -1 and the profile ids are not read. Party activity comes from Silero VAD v5 on
               each channel (``silero_vad.get_speech_timestamps`` defaults, ``activity_source = silero``; ``vad_head``
               = our own VAD head on each channel), each VAD segment being one "word" with empty text
               (``word_timing = vad``). Redacted segments are excluded zones: windows overlapping one are dropped.

  turnbench    mundo-ai/turn-benchmark-dev (Dataset Public License v1.0: NON-COMMERCIAL, attribution, no voice
               cloning; evaluation only). 38 real two-party conversations (7.3 h, 48 kHz 24-bit FLAC per speaker in
               parquet), three annotator tracks per speaker. Words per party = the majority-consensus floor-claiming
               spans of the vendored MIT scorer's turn view (integrations/turnbench_scorer, gold.TURN_CANONICAL) plus
               its consensus Backchannel events, one token per span with its transcript (``word_timing =
               annotated``); the turn view's excluded intervals are zones. Speaker ids: 5000 + hash of the actor id.
               A TurnBench benchmark scores our systems with THEIR scorer; this corpus gives the same audio the
               eot-bench v2 treatment (--corpus turnbench --roles both).

Roles. In each conversation one party is the HUMAN (the user whose turn ends an agent must detect) and the other the
AGENT. ``agent_rule``: ``second`` (default) = the party whose first voice comes later; ``alternate`` = channel 1 in
even-numbered conversations, channel 0 otherwise; or a fixed channel (0 / 1). ``turn_examples(roles=("human",))``
keeps the human's turns only; without ``roles`` both parties' turns are returned as ``ami.AMI`` would.

Labels: ``ami.speaker_turns`` on the two parties' words, unchanged (runs at turn_gap 0.5 s, floor definition with
max_hold 2 s: consecutive runs of one party are one turn unless the pause is >= 2 s or the other party produces a
non-backchannel run inside it; backchannel <= 2 words and <= 1 s; hesitation >= 0.3 s). Rule differences from AMI
follow from the words, not the code: Behavior-SD's interpolated words make hesitations invisible and turn the
"<= 2 words" half of the backchannel rule into a text-length rule; oto's VAD segments have no words, so a backchannel
is "an isolated VAD segment <= 1 s" (DualTurn's rule) and a hesitation is a >= 0.3 s VAD gap inside a turn.

Example keys (``turn_examples``): AMI's (audio = mixed mono 16 kHz, spk_targets (T, 4) with column 0 = the primary
and column 1 = the other party, spk_act / primary_act, eot, hes, turn_end_frame, onset_frame, text, speaker,
meeting, start, onset_clipped, n_hesitations, dropped_speakers) plus
  corpus, party (channel of the primary), role ('human' | 'agent'), word_timing,
  agent_act (T,)         the other party's activity on the window grid (= spk_targets[:, 1]),
  agent_end (T,)         CAUSAL event stream: 1 on the frame at which the other party's activity goes 1 -> 0
                         (the frame after its last active frame; frame 0 if it was active in the frame before the
                         window). It depends only on agent_act up to that frame, i.e. on what a product knows the
                         moment its TTS playout ends. "First voice after the agent stops" enrollment is
                         first_voice_after(any_activity, agent_end).
  agent_end_frame        the last agent_end event at or before onset_frame (-1 = none in the window),
  agent_turn_end (T,)    LABEL (not causal): 1 at the floor-definition turn ends of the other party inside the window
                         (a subset of agent_end: a floor turn end is the end of its last word).
Frames: [s, e) covers 80 ms frames int(s / 0.08) .. ceil(e / 0.08) - 1 (``ami.frames``), T = ToneLanguage.n_frames.

Audio: the two channels are resampled to 16 kHz (torchaudio) and SUMMED (one microphone hearing both parties;
clipped to [-1, 1]) into <root>/cache/<id>.mix16k.npy, memory-mapped on first use; labels (words per channel,
duration, zones) are built once into <root>/cache/labels/<id>.json.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import math
import os
import random
import tarfile
from pathlib import Path

import numpy as np

from ..paths import DATA_ROOT
from . import ami
from .ami import FRAME_SEC, LABEL_DEFAULTS, SR, activity, frames, normalize

log = logging.getLogger(__name__)

CORPORA = ("behavior_sd", "dailytalk", "oto", "turnbench")
_DATA = DATA_ROOT

BSD_REPO = "yhytoto12/behavior-sd"
BSD_ROOT = _DATA / "behavior_sd"
BSD_DEFAULT_SHARD = "test/0001.tar"
BSD_LICENSE_NOTE = ("Behavior-SD: CC BY 4.0 (dataset card), code MIT; synthetic CosyVoice dialogues. "
                    "https://huggingface.co/datasets/yhytoto12/behavior-sd\n")
DT_REPO = "kyutai/DailyTalkContiguous"
DT_ROOT = _DATA / "dailytalk"
DT_LICENSE_NOTE = ("DailyTalkContiguous: CC BY-SA 4.0 (same as DailyTalk); acted dialogues, 2 actors. "
                   "https://huggingface.co/datasets/kyutai/DailyTalkContiguous\n")
TB_REPO = "mundo-ai/turn-benchmark-dev"
TB_ROOT = _DATA / "turnbench_dev"
TB_LICENSE_NOTE = ("TurnBench dev: Dataset Public License v1.0 (Mundo AI): non-commercial, attribution, no voice "
                   "cloning. Evaluation only. https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev\n")
OTO_REPO = "otoearth/otoSpeech-full-duplex-processed-141h"
OTO_ROOT = _DATA / "oto141"
OTO_LICENSE_NOTE = ("otoSpeech-full-duplex-processed-141h: CC BY 4.0; the card prohibits attempts to identify / "
                    "de-anonymise speakers: no speaker-identity training on it. "
                    "https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h\n")
ROOTS = {"behavior_sd": BSD_ROOT, "dailytalk": DT_ROOT, "oto": OTO_ROOT, "turnbench": TB_ROOT}
TB_SCORER = _DATA.parent / "integrations" / "turnbench_scorer"  # vendored MIT scorer (gold consensus)
SPEAKER_OFFSET = {"behavior_sd": 2000, "dailytalk": 3000, "oto": -1, "turnbench": 5000}
WORD_TIMING = {"behavior_sd": "interpolated", "dailytalk": "aligned+vad", "oto": "vad", "turnbench": "annotated"}
AGENT_RULES = ("second", "alternate", 0, 1)
ACTIVITY_SOURCES = ("silero", "vad_head")
VAD_HEAD_CKPT = _DATA.parent / "runs" / "stage1_heads_pretrained.afm"
SPLIT_MOD = 5  # conversation i (in id order) is 'dev' iff i % 5 == 0, else 'train'


def _root(root, corpus: str) -> Path:
    assert corpus in CORPORA, f"corpus must be one of {CORPORA}, got {corpus!r}"
    return Path(root) if root else ROOTS[corpus]


# --------------------------------------------------------------------------- audio
def resample16k(x: np.ndarray, sr: int) -> np.ndarray:
    """(n, C) float32 at ``sr`` -> (C, n16) float32 at 16 kHz (torchaudio, sinc interpolation)."""
    import torch
    import torchaudio
    x = np.asarray(x, np.float32).T
    if sr == SR:
        return np.ascontiguousarray(x)
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x)), sr, SR).numpy()


def mix_mono(ch16: np.ndarray) -> np.ndarray:
    """(2, n) -> (n,) mixed mono: the sum of the two channels, clipped (one microphone hearing both parties)."""
    return np.clip(ch16.sum(0), -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- corpus readers
def list_ids(corpus: str, root=None) -> list[str]:
    """Conversation ids of a downloaded slice, in corpus order (a fixed prefix of the corpus, never a selection)."""
    root = _root(root, corpus)
    if corpus == "behavior_sd":
        return sorted(p.stem for d in bsd_dirs(root) for p in d.glob("*.json") if p.stem.isdigit())
    if corpus == "dailytalk":
        sl = root / "slice.json"
        if sl.exists():
            return [Path(p).stem for p in json.loads(sl.read_text())["paths"]]
        return sorted((p.stem for p in (root / "data_stereo").glob("*.wav")), key=int)
    if corpus == "turnbench":
        return sorted(tb_index(root), key=int)
    return list(oto_index(root))


def bsd_dirs(root: Path) -> list[Path]:
    """Extracted Behavior-SD shard directories <root>/<split>/<shard>/ (never the cache)."""
    return sorted(d for d in Path(root).glob("*/*/") if d.is_dir() and d.parts[-2] != "cache")


def bsd_file(root: Path, cid: str, ext: str) -> Path:
    return next(d / f"{cid}.{ext}" for d in bsd_dirs(root) if (d / f"{cid}.json").exists())


def oto_index(root: Path) -> dict[str, dict]:
    """{session id: {tar, flac: [offset, size], json: [offset, size]}} of every WebDataset shard under
    <root>/data/train (cached in <root>/cache/oto_index.json; shards are read in place, never extracted)."""
    root = Path(root)
    cache = root / "cache" / "oto_index.json"
    tars = sorted((root / "data" / "train").glob("shard-*.tar"))
    idx = json.loads(cache.read_text()) if cache.exists() else {}
    seen = {v["tar"] for v in idx.values()}
    new = [t for t in tars if str(t.relative_to(root)) not in seen]
    for t in new:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if not m.isfile() or "." not in m.name:
                    continue
                k, ext = m.name.rsplit(".", 1)
                if ext in ("flac", "json"):
                    idx.setdefault(k, {"tar": str(t.relative_to(root))})[ext] = [m.offset_data, m.size]
    if new:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(idx, indent=0, sort_keys=True))
    return {k: v for k, v in sorted(idx.items(), key=lambda kv: (kv[1]["tar"], kv[0])) if "flac" in v and "json" in v}


def _oto_member(root: Path, ent: dict, ext: str) -> bytes:
    off, size = ent[ext]
    with open(Path(root) / ent["tar"], "rb") as f:
        f.seek(off)
        return f.read(size)


def tb_index(root: Path) -> dict[str, list]:
    """{conversation_id: [parquet file, row group]} of TurnBench dev (one row group per conversation)."""
    import pyarrow.parquet as pq
    root = Path(root)
    cache = root / "cache" / "tb_index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    idx = {}
    for f in sorted((root / "data").glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            for cid in pf.read_row_group(g, columns=["conversation_id"])["conversation_id"].to_pylist():
                idx[str(cid)] = [f.name, g]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx, indent=0))
    return idx


def tb_row(root: Path, cid: str, columns) -> dict:
    import pyarrow.parquet as pq
    f, g = tb_index(root)[cid]
    pf = pq.ParquetFile(Path(root) / "data" / f)
    t = pf.read_row_group(g, columns=list(columns))
    rows = t.to_pylist()
    return next(r for r in rows if str(r["conversation_id"]) == cid)


def tb_words(row: dict) -> tuple[dict[int, list], list, dict]:
    """TurnBench annotations -> ({channel: words}, zones, stats) through the vendored scorer's consensus: turn-view
    floor-claiming spans + label-view Backchannel events per speaker (one token per span, its transcript as text),
    turn-view excluded intervals as zones."""
    import sys
    if str(TB_SCORER) not in sys.path:
        sys.path.insert(0, str(TB_SCORER))
    from turnbench.data import ANNOTATORS, SPEAKERS, Conversation
    from turnbench.gold import TURN_CANONICAL, consensus_for_conversation
    ann = {(sp, an): [(e["start_s"], e["end_s"], e["label"], e["text"]) for e in row[f"speaker_{sp}_annotation_{an}"]]
           for sp in SPEAKERS for an in ANNOTATORS}
    conv = Conversation(conversation_id=str(row["conversation_id"]), duration_s=0.0, annotations=ann, audio_bytes={})
    turn_ev, turn_ex = consensus_for_conversation(conv, TURN_CANONICAL)
    lab_ev, _ = consensus_for_conversation(conv)
    text = {}
    for sp in SPEAKERS:  # transcript of annotator a's segment starting nearest to a consensus span
        for st, _en, _lab, tx in ann[(sp, "a")]:
            text.setdefault(sp, []).append((st, tx))
    words = {0: [], 1: []}
    for e in turn_ev:
        if e.label != "Turn":
            continue
        cands = [(abs(st - e.start), tx) for st, tx in text.get(e.speaker, [])]
        tx = min(cands)[1] if cands and min(cands)[0] <= 0.3 else ""
        words[e.speaker - 1].append((float(e.start), float(e.end), normalize(tx)))
    nb = 0
    for e in lab_ev:
        if e.label == "Backchannel":
            words[e.speaker - 1].append((float(e.start), float(e.end), ""))
            nb += 1
    for k in words:
        words[k].sort(key=lambda w: (w[0], w[1]))
    zones = [(float(z.start), float(z.end)) for z in turn_ex]
    return words, zones, {"turn_spans": sum(1 for e in turn_ev if e.label == "Turn"), "backchannels": nb,
                          "excluded_intervals": len(zones)}


def read_stereo(corpus: str, root: Path, cid: str) -> tuple[np.ndarray, int, dict]:
    """(x (n, 2) float32 at the native rate, sr, metadata json) of one conversation."""
    import soundfile as sf
    root = Path(root)
    if corpus == "turnbench":
        row = tb_row(root, cid, ["conversation_id", "speaker_1_audio", "speaker_2_audio", "metadata"]
                     + [f"speaker_{sp}_annotation_{an}" for sp in (1, 2) for an in "abc"])
        a, sr = sf.read(io.BytesIO(row["speaker_1_audio"]["bytes"]), dtype="float32", always_2d=False)
        b, sr2 = sf.read(io.BytesIO(row["speaker_2_audio"]["bytes"]), dtype="float32", always_2d=False)
        assert sr == sr2 and a.ndim == 1 and b.ndim == 1, (cid, sr, sr2, a.shape, b.shape)
        n = min(len(a), len(b))
        return np.stack([a[:n], b[:n]], 1), sr, row
    if corpus == "behavior_sd":
        js = bsd_file(root, cid, "json")
        x, sr = sf.read(str(bsd_file(root, cid, "flac")), dtype="float32", always_2d=True)
        meta = json.loads(js.read_text())
    elif corpus == "dailytalk":
        x, sr = sf.read(str(root / "data_stereo" / f"{cid}.wav"), dtype="float32", always_2d=True)
        meta = json.loads((root / "data_stereo" / f"{cid}.json").read_text())
    else:
        ent = oto_index(root)[cid]
        x, sr = sf.read(io.BytesIO(_oto_member(root, ent, "flac")), dtype="float32", always_2d=True)
        meta = json.loads(_oto_member(root, ent, "json"))
    assert x.shape[1] == 2, (corpus, cid, x.shape)
    return x, sr, meta


# --------------------------------------------------------------------------- words per party
def _tok_text(tok: str) -> str:
    """Normalized text of a transcript token; bracketed non-lexical tokens ('[laughter]') -> ''."""
    t = tok.strip()
    return "" if t.startswith("[") or t.startswith("(") else normalize(t)


def interpolate_words(text: str, start: float, end: float) -> list[tuple[float, float, str]]:
    """Spread [start, end) over the whitespace tokens of ``text`` by character length (ICSI's rule for edge-timed
    stretches): pseudo-words with exact utterance edges. Empty text -> one wordless interval."""
    toks = text.split() or [""]
    w = np.array([max(1, len(t)) for t in toks], np.float64)
    b = start + (end - start) * np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    return [(float(b[i]), float(b[i + 1]), _tok_text(t)) for i, t in enumerate(toks)]


def bsd_words(meta: dict) -> tuple[dict[int, list], dict]:
    """Behavior-SD json -> ({channel: words}, events): utterances on their speaker's channel, backchannels on the
    listener's channel (1 - speaker_idx). events = labelled backchannel / interruption times for statistics."""
    words = {0: [], 1: []}
    ev = {"backchannels": [], "interruptions": []}
    for u in meta["utterances"]:
        k = int(u["speaker_idx"])
        words[k] += interpolate_words(u.get("tts_text") or "", float(u["start_time"]), float(u["end_time"]))
        if u.get("uttr_type"):
            ev["interruptions"].append((k, float(u["start_time"]), str(u["uttr_type"])))
        for b in u.get("backchannels") or []:
            words[1 - k] += interpolate_words(b.get("tts_text") or "", float(b["start_time"]), float(b["end_time"]))
            ev["backchannels"].append((1 - k, float(b["start_time"]), float(b["end_time"])))
    for k in words:
        words[k].sort(key=lambda w: (w[0], w[1]))
    return words, ev


def energy_channel(x: np.ndarray, sr: int, s: float, e: float, pad: float = 0.02) -> tuple[int, float]:
    """(channel with more energy over [s, e), energy ratio loud / quiet) of a stereo (n, 2) array."""
    a, b = max(0, int((s - pad) * sr)), min(len(x), int(math.ceil((e + pad) * sr)))
    seg = x[a:b] if b > a else x[max(0, a - int(0.05 * sr)): a + int(0.05 * sr)]
    en = (seg.astype(np.float64) ** 2).mean(0) + 1e-12
    k = int(np.argmax(en))
    return k, float(en[k] / en[1 - k])


def dt_words(meta: dict, x: np.ndarray, sr: int, ch16: np.ndarray, activity_source: str = "silero",
             min_ratio: float = 2.0) -> tuple[dict[int, list], dict]:
    """DailyTalkContiguous: the json aligns ONE actor's words only (tag SPEAKER_MAIN; the other actor speaks in the
    gaps with no timestamps). The aligned actor's channel is the one with more energy over the majority of the words
    (a word with an energy ratio < ``min_ratio`` keeps the majority channel); the OTHER channel's activity comes
    from VAD (``vad_segments``), as for oto. Stats: the ambiguous-word share and the frame agreement between the VAD
    and the aligned words on the aligned channel (a check of the VAD-derived labels)."""
    votes = [energy_channel(x, sr, float(s), float(e)) for _, (s, e), _ in meta["alignments"]]
    clear = [k for k, r in votes if r >= min_ratio]
    main = int(round(np.mean(clear))) if clear else 0
    amb = sum(r < min_ratio for _, r in votes)
    words = {main: [(float(s), float(e), normalize(tok)) for tok, (s, e), _ in meta["alignments"]], 1 - main: []}
    segs = vad_segments(ch16, activity_source)
    words[1 - main] = segs[1 - main]
    n_f = int(len(ch16[0]) / SR / FRAME_SEC)
    a_w, a_v = frames(activity(words[main]), n_f), frames(activity(segs[main]), n_f)
    agree = float((a_w == a_v).mean()) if n_f else 1.0
    return words, {"aligned_channel": main, "ambiguous_words": int(amb), "n_words": len(votes),
                   "vad_vs_words_frame_agreement": round(agree, 4),
                   "vad_segments_other_channel": len(segs[1 - main])}


_SILERO = None


def silero_model():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad
        _SILERO = load_silero_vad(onnx=True)
    return _SILERO


def vad_segments(ch16: np.ndarray, source: str = "silero", model=None) -> list[list[tuple[float, float, str]]]:
    """Per channel: speech segments [(start, end, '')] of 16 kHz audio. ``silero``: silero_vad.get_speech_timestamps
    defaults (threshold 0.5, min speech 250 ms, min silence 100 ms, pad 30 ms). ``vad_head``: our VAD head
    (runs/stage1_heads_pretrained.afm) on the 80 ms grid, p > 0.5, gaps <= 1 frame bridged, segments >= 3 frames."""
    assert source in ACTIVITY_SOURCES, source
    out = []
    if source == "silero":
        import torch
        from silero_vad import get_speech_timestamps
        m = model or silero_model()
        for c in ch16:
            ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)), m,
                                       sampling_rate=SR, return_seconds=True)
            out.append([(float(t["start"]), float(t["end"]), "") for t in ts])
        return out
    import torch
    m = model or vad_head_model()
    name = next(k for k, v in m.head_cfg.items() if v["type"] == "frame" and v.get("key") == "vad")
    with torch.no_grad():
        for c in ch16:
            segs, step = [], SR * 60
            for s0 in range(0, len(c), step):  # 60 s pieces (the encoder is causal / chunked: no context loss)
                seg = torch.from_numpy(np.ascontiguousarray(c[s0: s0 + step], dtype=np.float32))[None]
                enc, elen, hidden = m.encode(seg, torch.tensor([seg.shape[1]]), return_hidden=True)
                p = torch.sigmoid(m.heads[name].decode(m.head_input(name, enc, hidden), elen, logits=True)
                                  if "logits" in m.heads[name].decode.__code__.co_varnames
                                  else m.heads[name].decode(m.head_input(name, enc, hidden), elen))
                on = (p[0, : int(elen[0])] > 0.5).numpy().astype(np.int8)
                d = np.diff(np.concatenate([[0], on, [0]]))
                for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                    t0, t1 = s0 / SR + a * FRAME_SEC, s0 / SR + b * FRAME_SEC
                    if segs and t0 - segs[-1][1] <= FRAME_SEC + 1e-6:
                        segs[-1] = (segs[-1][0], t1)
                    else:
                        segs.append((t0, t1))
            out.append([(a, b, "") for a, b in segs if b - a >= 3 * FRAME_SEC - 1e-6])
    return out


_VAD_HEAD = None


def vad_head_model():
    global _VAD_HEAD
    if _VAD_HEAD is None:
        from ..train import load_model
        _VAD_HEAD = load_model(str(VAD_HEAD_CKPT), "cpu")
    return _VAD_HEAD


def channel_check(corpus: str, root=None, ids=None, n: int = 20) -> dict:
    """Share of labelled intervals whose louder channel is the labelled party (Behavior-SD: utterances on channel
    speaker_idx, backchannels on the listener's channel)."""
    root = _root(root, corpus)
    ids = list(ids or list_ids(corpus, root))[:n]
    agree = tot = bc_agree = bc_tot = 0
    for cid in ids:
        x, sr, meta = read_stereo(corpus, root, cid)
        if corpus != "behavior_sd":
            continue
        for u in meta["utterances"]:
            k, _ = energy_channel(x, sr, u["start_time"], u["end_time"])
            agree += k == int(u["speaker_idx"])
            tot += 1
            for b in u.get("backchannels") or []:  # overlapped by the utterance: judge the listener's own channel
                lis, s0, e0 = 1 - int(u["speaker_idx"]), float(b["start_time"]), float(b["end_time"])
                inside = (x[int(s0 * sr): int(e0 * sr), lis].astype(np.float64) ** 2).mean() + 1e-12
                ctx = np.concatenate([x[max(0, int((s0 - 0.5) * sr)): int(s0 * sr), lis],
                                      x[int(e0 * sr): int((e0 + 0.5) * sr), lis]]).astype(np.float64)
                around = (ctx ** 2).mean() + 1e-12 if len(ctx) else 1e-12
                bc_agree += inside > 4 * around
                bc_tot += 1
    return dict(n_conversations=len(ids), utterances=tot, utterance_channel_agreement=round(agree / max(1, tot), 4),
                backchannels=bc_tot, backchannel_on_listener_channel=round(bc_agree / max(1, bc_tot), 4))


# --------------------------------------------------------------------------- agent-side streams
def agent_end_stream(act: np.ndarray, active_before: bool = False) -> np.ndarray:
    """(T,) float32 causal event stream of an activity mask: 1 at frame t iff the party was active at t - 1 (or, for
    t = 0, in the frame before the window) and is not active at t. ev[:t + 1] depends on act[:t + 1] only."""
    a = np.asarray(act) > 0.5
    prev = np.concatenate([[bool(active_before)], a[:-1]])
    return (prev & ~a).astype(np.float32)


def agent_end_before(ev: np.ndarray, frame: int) -> int:
    """The last event frame <= ``frame`` (-1 if none)."""
    nz = np.nonzero(np.asarray(ev)[: int(frame) + 1] > 0.5)[0]
    return int(nz[-1]) if len(nz) else -1


def first_voice_after(any_act: np.ndarray, agent_end: np.ndarray) -> int:
    """Causal 'first voice after the agent stops' arming frame: the first frame with any activity at or after the
    first agent_end event (the first active frame if there is no event); len(any_act) if none."""
    T = len(any_act)
    ev = np.nonzero(np.asarray(agent_end) > 0.5)[0]
    t0 = int(ev[0]) if len(ev) else 0
    nz = np.nonzero(np.asarray(any_act)[t0:] > 0.5)[0]
    return int(t0 + nz[0]) if len(nz) else T


# --------------------------------------------------------------------------- dataset
def split_ids(ids: list[str], split: str, mod: int = SPLIT_MOD, dev_res=(0,)) -> list[str]:
    """Deterministic conversation split of a slice: 'dev' = conversations whose position i in id order has
    i % mod in dev_res (default every 5th, i % 5 == 0), 'train' = the rest, 'all' = everything. (Speaker-disjoint
    splits are impossible for DailyTalk's 2 actors; oto's speaker profiles are not read, licence.)"""
    assert split in ("train", "dev", "all"), split
    if split == "all":
        return list(ids)
    dev_res = {int(r) for r in dev_res}
    return [c for i, c in enumerate(ids) if ((i % mod) in dev_res) == (split == "dev")]


def slice_ids(corpus: str, root=None, hours: float | None = None, ids=None) -> list[str]:
    """The first conversations of the slice whose durations (audio headers / labels cache) sum to <= ``hours``."""
    root = _root(root, corpus)
    ids = list(ids or list_ids(corpus, root))
    if not hours:
        return ids
    out, cum = [], 0.0
    for cid in ids:
        d = conversation_duration(corpus, root, cid)
        if cum + d > hours * 3600:
            break
        out.append(cid)
        cum += d
    return out


def conversation_duration(corpus: str, root: Path, cid: str) -> float:
    import soundfile as sf
    lab = Path(root) / "cache" / "labels" / f"{cid}.json"
    if lab.exists():
        return float(json.loads(lab.read_text())["duration"])
    if corpus == "behavior_sd":
        return float(sf.info(str(bsd_file(Path(root), cid, "flac"))).duration)
    if corpus == "dailytalk":
        return float(sf.info(str(Path(root) / "data_stereo" / f"{cid}.wav")).duration)
    if corpus == "turnbench":
        return float(json.loads((TB_SCORER / "turnbench" / "durations-dev.json").read_text())["durations"][cid])
    return float(json.loads(_oto_member(Path(root), oto_index(root)[cid], "json"))["duration"])


class Dyadic(ami.AMI):
    """Two-party conversations with ``ami.AMI``'s label rules, modes and statistics. Party names are '<id>:<channel>'.
    Audio is loaded lazily per conversation (``audio=True`` only enables it); labels are built on first use and
    cached (Silero over both channels for oto: ~1 min per 18 min conversation on 2 CPU threads)."""

    def __init__(self, ids: list[str], corpus: str, root=None, audio: bool = True, verbose: bool = True,
                 agent_rule="second", activity_source: str = "silero", cache_dtype: str = "float32", **labels):
        assert agent_rule in AGENT_RULES, agent_rule
        assert activity_source in ACTIVITY_SOURCES, activity_source
        self.corpus, self.root, self.meetings, self.verbose = corpus, _root(root, corpus), list(ids), verbose
        self.cfg = {**LABEL_DEFAULTS, **{k: v for k, v in labels.items() if v is not None}}
        self.agent_rule, self.activity_source, self.cache_dtype = agent_rule, activity_source, cache_dtype
        self.audio_enabled = audio
        self.word_timing = WORD_TIMING[corpus]
        self.words, self.durations, self.zones, self.meta, self.party_gid = {}, {}, {}, {}, {}
        for m in self.meetings:
            lab = self._labels(m)
            self.words[m] = {f"{m}:{c}": [tuple(w) for w in lab["words"][str(c)]] for c in (0, 1)}
            self.durations[m] = float(lab["duration"])
            self.zones[m] = [tuple(z) for z in lab.get("zones", [])]
            self.meta[m] = lab.get("meta", {})
            for c in (0, 1):
                self.party_gid[f"{m}:{c}"] = int(lab["speaker_ids"][c])
        self.turns = {m: ami.speaker_turns(self.words[m], **self.cfg) for m in self.meetings}
        self.acts = {m: {s: activity(w, self.cfg["act_bridge"]) for s, w in self.words[m].items()}
                     for m in self.meetings}
        self.roles = {m: self._roles(i, m) for i, m in enumerate(self.meetings)}
        self.speaker_ids = sorted(set(self.party_gid.values()))
        self.spk_index = {}
        self.ann = self.root / "cache" / "labels"
        self._audio: dict[str, np.ndarray] = {}
        self.filtered: dict[str, dict] = {}

    # ----------------------------------------------------------------- labels
    def _labels(self, m: str) -> dict:
        p = self.root / "cache" / "labels" / f"{m}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("activity_source", "silero") == self.activity_source or self.corpus in ("behavior_sd", "turnbench"):
                return d
            p = p.with_name(f"{m}.{self.activity_source}.json")
            if p.exists():
                return json.loads(p.read_text())
        d = build_labels(self.corpus, self.root, m, self.activity_source)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, p)
        if self.verbose:
            log.info(f"  labels {self.corpus}/{m}: {d['duration'] / 60:.1f} min, "
                  f"{[len(d['words'][c]) for c in ('0', '1')]} words/segments -> {p}")
        return d

    def _roles(self, i: int, m: str) -> dict:
        first = {c: (self.acts[m][f"{m}:{c}"][0][0] if self.acts[m][f"{m}:{c}"] else float("inf")) for c in (0, 1)}
        if self.agent_rule == "second":
            agent = 1 if first[1] > first[0] else 0
        elif self.agent_rule == "alternate":
            agent = 1 if i % 2 == 0 else 0
        else:
            agent = int(self.agent_rule)
        return {"agent": agent, "human": 1 - agent}

    def party(self, name: str) -> int:
        return int(name.rsplit(":", 1)[1])

    def role(self, name: str) -> str:
        m, c = name.rsplit(":", 1)
        return "agent" if self.roles[m]["agent"] == int(c) else "human"

    def gid(self, name: str) -> int:
        return self.party_gid.get(name, -1)

    # ----------------------------------------------------------------- audio
    def _load_audio(self, m: str) -> np.ndarray:
        npy = self.root / "cache" / f"{m}.mix16k.npy"
        if not npy.exists():
            if self.corpus == "turnbench":  # the official protocol also needs the channels: decode once
                mix = mix_mono(np.asarray(self.channels16k(m), np.float32)).astype(self.cache_dtype)
            else:
                x, sr, _ = read_stereo(self.corpus, self.root, m)
                mix = mix_mono(resample16k(x, sr)).astype(self.cache_dtype)
            npy.parent.mkdir(parents=True, exist_ok=True)
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, mix)
            os.replace(tmp, npy)
            if self.verbose:
                log.info(f"  cache {self.corpus}/{m}: {len(mix) / SR / 60:.1f} min -> {npy}")
        return np.load(npy, mmap_mode="c")

    def channels16k(self, m: str) -> np.ndarray:
        """(2, n) float16 memmap of the two channels at 16 kHz (cache/<id>.ch16k.npy; built on first use)."""
        npy = self.root / "cache" / f"{m}.ch16k.npy"
        if not npy.exists():
            x, sr, _ = read_stereo(self.corpus, self.root, m)
            ch = resample16k(x, sr).astype(np.float16)
            npy.parent.mkdir(parents=True, exist_ok=True)
            tmp = npy.with_suffix(".tmp.npy")
            np.save(tmp, ch)
            os.replace(tmp, npy)
        return np.load(npy, mmap_mode="c")

    def _clip(self, m, a, b):
        if m not in self._audio:
            if not self.audio_enabled:
                raise RuntimeError("Dyadic(audio=False): no audio access")
            self._audio[m] = self._load_audio(m)
        x = self._audio[m]
        return x[int(round(a * SR)): int(round(b * SR))]

    def duration(self, m: str) -> float:
        return self.durations[m]

    # ----------------------------------------------------------------- modes
    def _keep(self, ex: dict) -> str | None:
        a = float(ex["start"])
        b = a + len(ex["audio"]) / SR
        if any(za < b and zb > a for za, zb in self.zones[ex["meeting"]]):
            return "redacted"
        return None

    @contextlib.contextmanager
    def tag_primary(self):
        """While active, ``gid`` returns a running index and records the party name it was asked for: the AMI window
        builders (``AMI.turn_examples``, ``eval_stage1.turn_windows``) call it exactly once per example, in order, so
        the k-th example's primary party is ``names[k]`` (``resolve_primary`` restores the speaker id)."""
        names: list[str] = []
        orig = self.gid

        def tag(name: str) -> int:
            names.append(name)
            return len(names) - 1
        self.gid = tag
        try:
            yield names
        finally:
            self.gid = orig

    def resolve_primary(self, data: list[dict], names: list[str]) -> list[dict]:
        """After ``tag_primary``: set each example's real speaker id and attach the agent-side fields."""
        assert len(data) == len(names), (len(data), len(names))
        for ex, name in zip(data, names):
            ex["speaker"] = self.party_gid[name]
            self.add_agent_fields(ex, name)
        return data

    def add_agent_fields(self, ex: dict, prim: str) -> dict:
        """Attach corpus / party / role / word_timing and the agent-side streams to an AMI-style turn example whose
        primary party is ``prim`` ('<id>:<channel>')."""
        m, a, T = ex["meeting"], float(ex["start"]), len(ex["spk_act"])
        other = f"{m}:{1 - self.party(prim)}"
        oa = frames(self.acts[m][other], T, a)
        before = bool(frames(self.acts[m][other], 1, a - FRAME_SEC)[0]) if a >= FRAME_SEC else False
        ev = agent_end_stream(oa, before)
        ate = np.zeros(T, np.float32)
        for t in self.turns[m]:
            if t["speaker"] == other and not t["bc"]:
                f = int(math.ceil((t["end"] - a) / FRAME_SEC - 1e-9))
                if 0 <= f < T:
                    ate[f] = 1
        ex.update(corpus=self.corpus, party=self.party(prim), role=self.role(prim), word_timing=self.word_timing,
                  agent_act=oa, agent_end=ev, agent_end_frame=agent_end_before(ev, ex["onset_frame"]),
                  agent_turn_end=ate)
        return ex

    def turn_examples(self, *a, roles=None, **kw) -> list[dict]:
        """AMI.turn_examples + the agent-side fields; ``roles`` = ('human',) keeps only the turns of the human
        party; windows overlapping an excluded zone (oto redactions) are dropped."""
        with self.tag_primary() as names:
            data = super().turn_examples(*a, **kw)
        self.resolve_primary(data, names)
        out, drop = [], {"redacted": 0, "role": 0}
        for ex in data:
            why = self._keep(ex)
            if why:
                drop[why] += 1
                continue
            if roles and ex["role"] not in roles:
                drop["role"] += 1
                continue
            out.append(ex)
        self.filtered["turn"] = dict(total=len(data), **drop)
        return out

    def diar(self, *a, **kw) -> list[dict]:
        data = super().diar(*a, **kw)
        return [ex for ex in data if self._keep(ex) is None]

    def human_turn_examples(self, *a, **kw) -> list[dict]:
        return self.turn_examples(*a, roles=("human",), **kw)

    def stats(self, *a, **kw) -> dict:
        st = super().stats(*a, **kw)
        st.update(corpus=self.corpus, word_timing=self.word_timing, agent_rule=self.agent_rule,
                  activity_source="labels" if self.corpus in ("behavior_sd", "turnbench") else self.activity_source,
                  roles={m: r for m, r in self.roles.items()})
        return st


def build_labels(corpus: str, root: Path, cid: str, activity_source: str = "silero") -> dict:
    """Words per channel + duration + excluded zones + speaker ids of one conversation (the labels cache entry)."""
    if corpus == "turnbench":  # annotations only: no audio decode
        meta = tb_row(Path(root), cid, ["conversation_id", "metadata"]
                      + [f"speaker_{sp}_annotation_{an}" for sp in (1, 2) for an in "abc"])
        x, sr, dur = None, 48000, conversation_duration(corpus, root, cid)
    else:
        x, sr, meta = read_stereo(corpus, root, cid)
        dur = len(x) / sr
    zones, stats, info = [], {}, {}
    if corpus == "behavior_sd":
        words, ev = bsd_words(meta)
        stats = {"backchannels": len(ev["backchannels"]), "interruptions": len(ev["interruptions"])}
        spk = [SPEAKER_OFFSET[corpus] + int(hashlib.md5(v.encode()).hexdigest(), 16) % 100000
               for v in meta["tts_speaker_ids"]]
        info = {"tts_genders": meta.get("tts_genders"), "num_turns": meta.get("num_turns"),
                "behaviors": meta.get("behaviors"), "statistics": meta.get("statistics")}
    elif corpus == "dailytalk":
        words, stats = dt_words(meta, x, sr, resample16k(x, sr), activity_source)
        spk = [SPEAKER_OFFSET[corpus] + c for c in (0, 1)]
    elif corpus == "turnbench":
        words, zones, stats = tb_words(meta)
        spk = [SPEAKER_OFFSET[corpus] + int(hashlib.md5(str(meta["metadata"][f"speaker_{k}_actor_id"]).encode()).hexdigest(), 16) % 100000
               for k in (1, 2)]
        info = {"conversation_type": meta["metadata"].get("conversation_type"),
                "genders": [meta["metadata"].get(f"speaker_{k}_actor_gender") for k in (1, 2)]}
    else:
        segs = vad_segments(resample16k(x, sr), activity_source)
        words = {0: segs[0], 1: segs[1]}
        zones = [(float(z["start_sec"]), float(z["end_sec"])) for z in meta.get("redacted_segments") or []]
        spk = [-1, -1]  # licence: no speaker identification; the profile ids are not read
        stats = {"segments": [len(segs[0]), len(segs[1])], "speech_sec": [round(sum(e - s for s, e, _ in g), 1) for g in segs]}
        info = {"session_type": meta.get("session_type"), "channels_gender": [s.get("gender") for s in meta.get("speakers", [])]}
    return {"corpus": corpus, "id": cid, "duration": dur, "sr": sr, "words": {str(c): words[c] for c in (0, 1)},
            "zones": zones, "speaker_ids": spk,
            "activity_source": "labels" if corpus in ("behavior_sd", "turnbench") else activity_source,
            "stats": stats, "meta": info}


def attach_asr_text(data: list[dict], root: Path) -> list[dict]:
    """``asr_text: true`` (oto has no transcripts): each turn window's ``text`` = the words of its PRIMARY party's
    own-channel ASR decode (<root>/cache/asr/<id>.json) emitted inside the window,
    i.e. the AMI convention (the primary's words) with the frozen ASR's hypothesis as the transcript. A missing
    cache raises (run the script first)."""
    cache: dict[str, dict] = {}
    for ex in data:
        m = ex["meeting"]
        if m not in cache:
            p = Path(root) / "cache" / "asr" / f"{m}.json"
            if not p.exists():
                raise FileNotFoundError(f"asr_text: no transcript cache {p} (transcribe the corpus first)")
            cache[m] = json.loads(p.read_text())
        a = float(ex["start"])
        b = a + len(ex["audio"]) / SR
        ex["text"] = " ".join(w for t, w in cache[m][str(ex["party"])] if a <= t < b)
    return data


# --------------------------------------------------------------------------- recipe hook
def recipe_data(cfg: dict, split: str) -> list[dict]:
    """``data: {dyadic: {...}}`` for ``audioforge.train.load_data``.

    Keys: corpus (behavior_sd | dailytalk | oto), root, mode (turn | diar), hours (slice prefix), n_conv (first n
    conversations), ids ({train: [...], val: [...]} to override), split ({train: train, val: dev}: split_ids of the
    slice), split_mod / dev_res (split_ids: dev = positions i with i % split_mod in dev_res), roles (turn mode: [human] |
    null = both parties), agent_rule, activity_source (oto), label keys (turn_gap, hes_gap, turn_def, max_hold,
    bc_max_sec, bc_max_words, act_bridge), mode kwargs (window_sec, hop_sec, lead_sec, trail_sec, min_trail),
    n_train / n_val (seeded caps), seed, cache_dtype. With data.synthetic.n_<split> a synthetic stand-in is returned
    (hermetic smoke tests), as datasets/ami.py."""
    d = cfg["data"]
    dc = dict(d["dyadic"])
    mode = dc.get("mode", "turn")
    syn = d.get("synthetic") or {}
    if f"n_{split}" in syn:
        n, seed = syn[f"n_{split}"], 0 if split == "train" else 1
        if mode == "turn":
            from ..conversation import conversation_dataset
            return list(conversation_dataset(n, seed=seed))
        from ..data import synthetic_dataset
        return synthetic_dataset("diar", n, seed=seed)
    corpus, root = dc["corpus"], dc.get("root")
    key = "train" if split == "train" else "val"
    ids = (dc.get("ids") or {}).get(key)
    if not ids:
        sname = (dc.get("split") or {"train": "train", "val": "dev"})[key]
        pool = slice_ids(corpus, root, dc.get("hours"))
        if dc.get("n_conv"):  # the first n_conv conversations of the slice (instead of / within hours)
            pool = pool[: int(dc["n_conv"])]
        ids = split_ids(pool, sname, int(dc.get("split_mod", SPLIT_MOD)), dc.get("dev_res", (0,)))
    labels = {k: dc[k] for k in LABEL_DEFAULTS if k in dc}
    ds = Dyadic(ids, corpus, root, agent_rule=dc.get("agent_rule", "second"),
                activity_source=dc.get("activity_source", "silero"), cache_dtype=dc.get("cache_dtype", "float32"),
                verbose=False, **labels)
    if mode == "diar":
        data = ds.diar(dc.get("window_sec", 20.0), dc.get("hop_sec", 10.0))
    else:
        roles = dc.get("roles", ["human"])
        data = ds.turn_examples(dc.get("window_sec", 20.0), dc.get("lead_sec", 4.0), dc.get("trail_sec", 2.0),
                                dc.get("min_trail", 1.0), roles=tuple(roles) if roles else None)
    cap = dc.get(f"n_{split}")
    if cap and cap < len(data):
        idx = sorted(random.Random(dc.get("seed", 0)).sample(range(len(data)), cap))
        data = [data[i] for i in idx]
    if mode == "turn" and dc.get("asr_text"):
        attach_asr_text(data, ds.root)
    log.info(f"[dyadic:{corpus}] {mode}/{split}: {len(ids)} conversations -> {len(data)} examples "
          f"{json.dumps(ds.filtered.get('turn', {}))}")
    return data
