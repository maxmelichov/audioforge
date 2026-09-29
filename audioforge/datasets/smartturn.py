"""pipecat-ai smart-turn training data as audioforge examples (research/COMPLETENESS.md §1 for the licence check).

Only ``pipecat-ai/human_5_all`` is used: it is the one smart-turn dataset whose card carries a licence
(``license: bsd-2-clause``; 3 862 human English clips, 1 931 complete / 1 931 incomplete, one FLAC per clip, 312 MB).
The v3 / v3.1 / v3.2 train and test datasets have no licence field and no LICENSE file, so their audio is never
downloaded here; only their metadata columns (``id``, ``dataset``, ``language``) were read to align our evaluation split
with smart-turn's own test split (``v32_human5_ids.json``: the human_5 clips that sit in smart-turn-data-v3.2-test).

Each clip becomes::

    audio            float32 (S,) 16 kHz mono
    complete         bool  (endpoint_bool: the user finished speaking)
    utt_end_frame    int64 (1,)  exclusive 80 ms frame of the last speech frame + 1 (energy VAD, datasets/librispeech.py)
    complete_label   float32 (1,) the same label as an array (collated; ``complete`` itself is not)
    completeness     float32 (T,) 1 on frames >= utt_end_frame if complete else 0; 0 before the end
    completeness_w   float32 (T,) 1 on frames >= utt_end_frame, ``mid_weight`` before
    id               str (the clip's path inside the parquet)

Decoded audio is cached once as one float32 .npy under ``<root>/cache/`` and memory-mapped (as datasets/librispeech.py).

Split: ``eval`` = the clips whose id is a human_5 id of smart-turn-data-v3.2-test (so neither smart-turn v3.2 nor our
head trained on them), ``train`` = the rest; without the id file a seeded 15 % stratified split (``seed``).
"""
from __future__ import annotations

import io
import json
import logging
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..data import ToneLanguage
from ..paths import DATA_ROOT

log = logging.getLogger(__name__)

SR = 16000
DEFAULT_ROOT = DATA_ROOT / "smartturn"
PARQUET = "human_5_all/data/train-00000-of-00001.parquet"
IDS_JSON = "v32_human5_ids.json"
LICENCE = "bsd-2-clause"  # the dataset card's license field (huggingface.co/datasets/pipecat-ai/human_5_all)


def _root(root) -> Path:
    return Path(root) if root else DEFAULT_ROOT


# --------------------------------------------------------------------------- decode + cache
def decode_flac(data: bytes) -> np.ndarray:
    """FLAC / WAV bytes -> float32 (S,) at 16 kHz mono."""
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import torch
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return np.ascontiguousarray(x, np.float32)


def build_cache(root=None, parquet: str | Path | None = None, verbose: bool = True) -> Path:
    """Decode every clip of the parquet once -> <root>/cache/human_5_all.npy (+ .json meta). Returns the meta path."""
    import pyarrow.parquet as pq
    root = _root(root)
    pq_path = Path(parquet) if parquet else root / PARQUET
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta_p, wav_p = cache / "human_5_all.json", cache / "human_5_all.npy"
    if meta_p.exists() and wav_p.exists():
        return meta_p
    pf = pq.ParquetFile(pq_path)
    ids, labels, lens, chunks = [], [], [], []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg).to_pydict()
        for a, lab in zip(t["audio"], t["endpoint_bool"]):
            x = decode_flac(a["bytes"])
            ids.append(str(a.get("path") or len(ids)))
            labels.append(bool(lab))
            lens.append(len(x))
            chunks.append(x)
        if verbose:
            log.info(f"[smartturn] decoded row group {rg + 1}/{pf.num_row_groups}: {len(ids)} clips")
    wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    np.save(wav_p, wav)
    off = np.concatenate([[0], np.cumsum(lens)]).tolist()
    meta_p.write_text(json.dumps({"ids": ids, "complete": labels, "offsets": off, "sr": SR, "licence": LICENCE,
                                  "source": str(pq_path)}))
    return meta_p


# --------------------------------------------------------------------------- labels
def utterance_end_frame(x: np.ndarray, rel_db: float = 30.0) -> int:
    """Exclusive 80 ms frame of the last speech frame + 1 (energy VAD relative to the loudest frame); T if silent."""
    from .librispeech import energy_vad
    T = ToneLanguage.n_frames(len(x))
    v = energy_vad(x, T, rel_db)
    return int(np.nonzero(v)[0].max()) + 1 if v.any() else T


def frame_labels(T: int, end: int, complete: bool, mid_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """(completeness (T,), completeness_w (T,)): the label on frames >= end, 0 before; weights 1 / mid_weight."""
    end = int(min(max(end, 1), T))
    y = np.zeros(T, np.float32)
    w = np.full(T, float(mid_weight), np.float32)
    y[end:] = float(bool(complete))
    w[end:] = 1.0
    return y, w


def _basename(s: str) -> str:
    s = str(s).replace("\\", "/").split("/")[-1]
    return s.rsplit(".", 1)[0] if "." in s else s


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _uuid(s: str) -> str:
    """The clip's uuid (human_5_all paths are 'complete_<uuid>_normalized.flac'; v3.2 ids are the bare uuid);
    the basename without extension when there is none."""
    m = _UUID.search(str(s).lower())
    return m.group(0) if m else _basename(s)


def match_test_ids(ids: list[str], test_ids: list[str]) -> np.ndarray:
    """Mask of the clips whose uuid (else basename) equals that of a smart-turn v3.2-test human_5 id."""
    tb = {_uuid(t) for t in test_ids}
    return np.array([_uuid(s) in tb for s in ids], bool)


def split_indices(meta: dict, root=None, seed: int = 0, eval_frac: float = 0.15) -> dict:
    """{'train': [...], 'eval': [...], 'how': ...}: v3.2-test alignment when the id file matches, else seeded
    stratified split."""
    root = _root(root)
    ids, lab = meta["ids"], np.asarray(meta["complete"], bool)
    p = root / IDS_JSON
    if p.exists():
        j = json.loads(p.read_text())
        m = match_test_ids(ids, [x["id"] for x in j.get("test", [])])
        if m.sum() >= 50:
            return {"train": np.nonzero(~m)[0].tolist(), "eval": np.nonzero(m)[0].tolist(),
                    "how": f"smart-turn-data-v3.2-test human_5 ids ({int(m.sum())} matched of {len(j.get('test', []))})"}
    ev = []
    for c in (True, False):
        idx = np.nonzero(lab == c)[0].tolist()
        ev += random.Random(seed + int(c)).sample(idx, int(round(eval_frac * len(idx))))
    evs = set(ev)
    return {"train": [i for i in range(len(ids)) if i not in evs], "eval": sorted(ev),
            "how": f"seeded stratified {eval_frac:.0%} split (seed {seed})"}


# --------------------------------------------------------------------------- examples
class SmartTurnClips(Sequence):
    """Lazy list of examples over the memory-mapped cache."""

    def __init__(self, root=None, split: str = "train", mid_weight: float = 0.1, rel_db: float = 30.0, seed: int = 0,
                 n: int | None = None, verbose: bool = True):
        self.root = _root(root)
        meta_p = build_cache(self.root, verbose=verbose)
        self.meta = json.loads(meta_p.read_text())
        self.wav = np.load(meta_p.with_suffix(".npy"), mmap_mode="r")
        sp = split_indices(self.meta, self.root, seed)
        self.how = sp["how"]
        self.idx = sp["eval" if split in ("eval", "val", "test") else "train"]
        if n and n < len(self.idx):
            self.idx = sorted(random.Random(seed).sample(self.idx, n))
        self.mid_weight, self.rel_db = float(mid_weight), float(rel_db)
        self._ends: dict[int, int] = {}
        if verbose:
            lab = np.asarray(self.meta["complete"], bool)[self.idx]
            log.info(f"[smartturn] {split}: {len(self.idx)} clips ({int(lab.sum())} complete / {int((~lab).sum())} "
                  f"incomplete), split = {self.how}")

    def __len__(self):
        return len(self.idx)

    def audio(self, j: int) -> np.ndarray:
        a, b = self.meta["offsets"][j], self.meta["offsets"][j + 1]
        return np.array(self.wav[a:b], np.float32)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(len(self)))]
        j = self.idx[i]
        x = self.audio(j)
        if j not in self._ends:
            self._ends[j] = utterance_end_frame(x, self.rel_db)
        end, complete = self._ends[j], bool(self.meta["complete"][j])
        T = ToneLanguage.n_frames(len(x))
        y, w = frame_labels(T, end, complete, self.mid_weight)
        return dict(audio=x, complete=complete, utt_end_frame=np.array([end], np.int64), completeness=y,
                    completeness_w=w, complete_label=np.array([float(complete)], np.float32), id=self.meta["ids"][j])


def recipe_data(cfg: dict, split: str):
    """``data: {smartturn: {root, mid_weight, rel_db, seed, n_train, n_val}}`` for audioforge.train.load_data.
    With ``data.synthetic.n_<split>`` a synthetic stand-in (LibriSpeech-free smoke tests): tone-language utterances,
    label = 'ends with speech' (complete) vs 'cut 0.4 s before the end' (incomplete)."""
    d = cfg["data"]
    sc = dict(d.get("smartturn") or {})
    syn = d.get("synthetic") or {}
    if f"n_{split}" in syn:
        return synthetic_clips(int(syn[f"n_{split}"]), seed=0 if split == "train" else 1,
                               mid_weight=sc.get("mid_weight", 0.1))
    key = "train" if split == "train" else "val"
    return SmartTurnClips(sc.get("root"), "train" if key == "train" else "eval", sc.get("mid_weight", 0.1),
                          sc.get("rel_db", 30.0), sc.get("seed", 0), sc.get(f"n_{key}"))


def synthetic_clips(n: int, seed: int = 0, mid_weight: float = 0.1) -> list[dict]:
    lang, rng = ToneLanguage(), random.Random(seed)
    out = []
    for i in range(n):
        ex = lang.example(rng)
        x = np.asarray(ex["audio"], np.float32)
        complete = i % 2 == 0
        if not complete:
            x = x[: max(SR // 2, len(x) - int(0.4 * SR))]
        x = np.concatenate([x, np.zeros(int(0.3 * SR), np.float32)])
        T = ToneLanguage.n_frames(len(x))
        end = utterance_end_frame(x)
        y, w = frame_labels(T, end, complete, mid_weight)
        out.append(dict(audio=x, complete=complete, utt_end_frame=np.array([end], np.int64), completeness=y,
                        completeness_w=w, complete_label=np.array([float(complete)], np.float32), id=f"syn{i}"))
    return out
