"""LibriSpeech (OpenSLR SLR12) as audioforge list-of-dict data.

Every utterance becomes the dict ``audioforge.data.ToneLanguage.example`` produces::

    audio    float32 (S,) at 16 kHz          text     lowercase transcript
    speaker  int index into the split's sorted speaker list
    vad      (T,) float32 energy VAD per 80 ms encoder frame (1280 samples)
    eou      (T,) float32, 1 on the 2 frames right after the last speech frame
    with T == ToneLanguage.n_frames(len(audio))   (+ "id", "duration" for bookkeeping)

``mixtures()`` and ``speaker_attributed()`` build 2-speaker overlapped mixtures with the
same keys as ``ToneLanguage.mixture`` / ``synthetic_dataset("speaker_attributed")``.

Decoded audio is cached once per (split, duration window) as a single float32 .npy under
``<root>/cache/`` and memory-mapped, so a 30 h training set costs no RAM up front and a
second run starts in seconds.

Energy VAD: frame energy (dB) > utterance max frame energy - ``rel_db`` (30 dB), then 1-frame
(80 ms) gaps between speech frames are bridged and isolated 1-frame blips dropped. On LibriSpeech
the noise floor sits 38-55 dB under the loudest frame and voiced speech within 0-25 dB of it, so
30 dB leaves a margin on both sides (print masks with ``scripts/research/prepare_librispeech.py --show-vad``).

MPS note: every new batch shape costs MPS graph-cache and buffer memory that is never freed, so
train with ``audioforge.datasets.train_librispeech`` (``QuantizedCollate``: <= 4 batch shapes).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import ToneLanguage, synthetic_dataset
from ..paths import DATA_ROOT

log = logging.getLogger(__name__)

SR = 16000
FRAME = 1280  # samples per 80 ms encoder frame
URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
SPLITS = {  # name -> approximate archive size (for messages only)
    "dev-clean": "337M", "dev-other": "314M", "test-clean": "346M", "test-other": "328M",
    "train-clean-100": "6.3G", "train-clean-360": "23G", "train-other-500": "30G"}
DEFAULT_ROOT = DATA_ROOT / "librispeech"
VAD_PARAMS = "energy rel_db={:g} bridge=1 min_run=2"  # cached labels are rebuilt when this changes


def _root(root) -> Path:
    return Path(root) if root else DEFAULT_ROOT


# --------------------------------------------------------------------------- download / extract
MD5 = {"dev-clean": "42e2234ba48799c1f50f24a7926300a1", "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
       "test-clean": "32fa31d27d2e1cad72775fee3f4849a9", "test-other": "fb5a50374b501bb3bac4815ee91d3135",
       "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
       "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
       "train-other-500": "d1a0fd59409feb2c614ce4d30c387708"}  # openslr.org/resources/12/md5sum.txt


class _NoRange(RuntimeError):
    """The server answered a Range request with something other than 206 (not retried)."""


def _fetch_range(url: str, dst: Path, start: int, end: int, min_rate: float = 3e5, window: float = 20.0,
                 retries: int = 500):
    """Bytes [start, end] of ``url`` -> ``dst`` (resumes). openslr.org throttles long-lived
    connections to a trickle, so reconnect whenever the rate over ``window`` s drops below ``min_rate``."""
    need = end - start + 1
    for _ in range(retries):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= need:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "ab") as f:
                if r.status != 206:  # not a transient error: retrying would only repeat it (fail fast)
                    raise _NoRange(f"server ignored the Range header (HTTP {r.status})")
                t0, got0, got = time.time(), 0, 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if time.time() - t0 > window:
                        if (got - got0) / (time.time() - t0) < min_rate:
                            break  # throttled: reconnect
                        t0, got0 = time.time(), got
        except _NoRange:
            raise
        except Exception as e:  # timeouts, resets: resume
            log.info(f"  {dst.name}: {type(e).__name__}: {e} (retrying)")
            time.sleep(2)
    raise RuntimeError(f"{url} [{start}-{end}] failed after {retries} attempts")


def download(split: str, root=None, connections: int = 6) -> Path:
    """Resumable parallel-range download of ``<split>.tar.gz`` into ``root``; md5-verified."""
    import hashlib
    import threading
    root = _root(root)
    root.mkdir(parents=True, exist_ok=True)
    dst, url = root / f"{split}.tar.gz", URL.format(split=split)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
        total = int(r.headers["Content-Length"])
    have = dst.stat().st_size if dst.exists() else 0  # a partial single-stream download is kept
    if have < total:
        size = -(-(total - have) // connections)
        ranges = [(have + k * size, min(total, have + (k + 1) * size) - 1) for k in range(connections)]
        ranges = [r for r in ranges if r[0] <= r[1]]
        parts = [dst.with_name(f"{dst.name}.part{k}") for k in range(len(ranges))]
        log.info(f"  {split}: {total / 1e9:.2f} GB, fetching {(total - have) / 1e9:.2f} GB over "
              f"{len(ranges)} connections")
        ths = [threading.Thread(target=_fetch_range, args=(url, p, a, b)) for p, (a, b) in zip(parts, ranges)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        with open(dst, "ab") as f:
            for p, (a, b) in zip(parts, ranges):
                assert p.stat().st_size == b - a + 1, f"incomplete {p}"
                with open(p, "rb") as src:
                    while chunk := src.read(1 << 24):
                        f.write(chunk)
        for p in parts:
            p.unlink()
    if split in MD5:
        h = hashlib.md5()
        with open(dst, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        if h.hexdigest() != MD5[split]:
            raise RuntimeError(f"md5 mismatch for {dst}: {h.hexdigest()} != {MD5[split]} (delete it and retry)")
    return dst


def extract(split: str, root=None) -> Path:
    """Extract to ``<root>/LibriSpeech/<split>/`` (idempotent; a sentinel marks completion)."""
    root = _root(root)
    out = root / "LibriSpeech" / split
    done = root / "cache" / f"{split}.extracted"
    if done.exists() and out.exists():
        return out
    with tarfile.open(root / f"{split}.tar.gz") as tar:
        tar.extractall(root, filter="data")
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(time.strftime("%F %T"))
    return out


# --------------------------------------------------------------------------- index
def scan(split: str, root=None) -> list[dict]:
    """All utterances of a split: id, speaker_id, text (lowercase), n (samples), path (relative)."""
    import soundfile as sf
    root = _root(root)
    cache = root / "cache" / f"{split}.index.json"
    if cache.exists():
        return json.loads(cache.read_text())
    base = root / "LibriSpeech" / split
    if not base.exists():
        raise FileNotFoundError(f"{base} missing: run scripts/research/prepare_librispeech.py --splits {split}")
    utts = []
    for trans in sorted(base.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            info = sf.info(str(path))
            assert info.samplerate == SR and info.channels == 1, (path, info)
            utts.append(dict(id=uid, speaker_id=int(uid.split("-")[0]), text=text.strip().lower(),
                             n=int(info.frames), path=str(path.relative_to(root))))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(utts))
    return utts


# --------------------------------------------------------------------------- labels
def frame_db(x: np.ndarray, T: int) -> np.ndarray:
    """(T,) mean energy in dB per 80 ms frame; frame t covers samples [1280 t, 1280 (t+1)) (zero-padded)."""
    xp = np.zeros(T * FRAME, np.float64)
    m = min(len(x), T * FRAME)
    xp[:m] = x[:m]
    return 10 * np.log10((xp.reshape(T, FRAME) ** 2).mean(1) + 1e-10)


def energy_vad(x: np.ndarray, n_frames: int | None = None, rel_db: float = 30.0, bridge: int = 1,
               min_run: int = 2) -> np.ndarray:
    """(T,) float32 speech mask at 80 ms (``frame_db`` relative to the loudest frame)."""
    T = n_frames or ToneLanguage.n_frames(len(x))
    e = frame_db(x, T)
    v = e > e.max() - rel_db
    if bridge and v.any():  # bridge short gaps (<= bridge frames) between speech frames
        on = np.nonzero(v)[0]
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= bridge + 1:
                v[a + 1: b] = True
    if min_run > 1 and v.any():  # drop isolated blips (clicks, breaths) shorter than min_run frames
        edges = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            if b - a < min_run:
                v[a:b] = False
    return v.astype(np.float32)


def eou_targets(vad: np.ndarray, width: int = 2) -> np.ndarray:
    """1 on the ``width`` frames after the last speech frame (as ToneLanguage.example)."""
    eou = np.zeros(len(vad), np.float32)
    if vad.any():
        last = int(np.nonzero(vad)[0].max())
        eou[last + 1: last + 1 + width] = 1
    return eou


def _frames(act: np.ndarray, T: int) -> np.ndarray:
    a = np.pad(act, (0, max(0, T * FRAME - len(act))))[: T * FRAME]
    return (a.reshape(T, FRAME).mean(1) > 0.5).astype(np.float32)


# --------------------------------------------------------------------------- dataset
class LibriSpeech:
    """One split, filtered to ``min_sec <= duration <= max_sec`` (None = no cap).

    ``speaker`` indices run over *all* speakers of the split (sorted by LibriSpeech id), so
    they are stable under the duration filter and ``num_speakers == len(ds.speaker_ids)``.
    """

    def __init__(self, split: str, root=None, max_sec: float | None = 12.0, min_sec: float = 1.0,
                 cache: bool = True, rel_db: float = 30.0, verbose: bool = True):
        self.split, self.root, self.rel_db, self.verbose = split, _root(root), rel_db, verbose
        self.all = scan(split, self.root)
        self.speaker_ids = sorted({u["speaker_id"] for u in self.all})
        self.spk_index = {s: i for i, s in enumerate(self.speaker_ids)}
        hi = math.inf if max_sec is None else max_sec * SR
        self.utts = [u for u in self.all if min_sec * SR <= u["n"] <= hi]
        self.max_sec, self.min_sec = max_sec, min_sec
        self._mm = self._off = self._vad = self._voff = None
        if cache:
            self._load_cache()

    # ----------------------------------------------------------------- cache
    def _cache_stem(self) -> Path:
        cap = "all" if self.max_sec is None else f"{self.max_sec:g}"
        return self.root / "cache" / f"{self.split}.{self.min_sec:g}-{cap}s"

    def _load_cache(self):
        stem = self._cache_stem()
        wav, meta = Path(f"{stem}.f32.npy"), Path(f"{stem}.meta.npz")
        if not (wav.exists() and meta.exists()):
            self._build_cache(wav, meta)
        m = np.load(meta)
        ids = [u["id"] for u in self.utts]
        if list(m["ids"]) != ids or str(m["vad_params"]) != VAD_PARAMS.format(self.rel_db):
            self._build_cache(wav, meta)
            m = np.load(meta)
        self._off, self._vad, self._voff = m["off"], m["vad"], m["voff"]
        self._mm = np.load(wav, mmap_mode="c")  # copy-on-write: writable views, file untouched

    def _build_cache(self, wav: Path, meta: Path):
        import soundfile as sf
        wav.parent.mkdir(parents=True, exist_ok=True)
        n = np.array([u["n"] for u in self.utts], np.int64)
        off = np.concatenate([[0], np.cumsum(n)])
        T = np.array([ToneLanguage.n_frames(int(k)) for k in n], np.int64)
        voff = np.concatenate([[0], np.cumsum(T)])
        vad = np.zeros(int(voff[-1]), np.float32)
        tmp = wav.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(int(off[-1]),))
        t0 = time.time()
        for i, u in enumerate(self.utts):
            x, sr = sf.read(str(self.root / u["path"]), dtype="float32")
            assert sr == SR and len(x) == u["n"], u["id"]
            mm[off[i]: off[i + 1]] = x
            vad[voff[i]: voff[i + 1]] = energy_vad(x, int(T[i]), self.rel_db)
            if self.verbose and (i + 1) % 2000 == 0:
                log.info(f"  cache {self.split}: {i + 1}/{len(self.utts)} ({time.time() - t0:.0f}s)")
        mm.flush()
        del mm
        os.replace(tmp, wav)
        np.savez(meta, ids=np.array([u["id"] for u in self.utts]), off=off, vad=vad, voff=voff,
                 vad_params=np.array(VAD_PARAMS.format(self.rel_db)))
        if self.verbose:
            log.info(f"  cache {self.split}: {len(self.utts)} utts, {off[-1] / SR / 3600:.1f} h -> {wav} "
                  f"({time.time() - t0:.0f}s)")

    # ----------------------------------------------------------------- access
    def __len__(self):
        return len(self.utts)

    def audio(self, i: int) -> np.ndarray:
        if self._mm is not None:
            return self._mm[self._off[i]: self._off[i + 1]]
        import soundfile as sf
        return sf.read(str(self.root / self.utts[i]["path"]), dtype="float32")[0]

    def vad(self, i: int) -> np.ndarray:
        if self._vad is not None:
            return self._vad[self._voff[i]: self._voff[i + 1]]
        return energy_vad(self.audio(i), rel_db=self.rel_db)

    def speaker(self, i: int) -> int:
        return self.spk_index[self.utts[i]["speaker_id"]]

    def example(self, i: int) -> dict:
        u, x, vad = self.utts[i], self.audio(i), self.vad(i)
        assert len(vad) == ToneLanguage.n_frames(len(x))
        return dict(audio=x, text=u["text"], speaker=self.speaker(i), vad=vad, eou=eou_targets(vad),
                    id=u["id"], duration=len(x) / SR)

    def utterances(self, indices=None) -> list[dict]:
        return [self.example(i) for i in (range(len(self)) if indices is None else indices)]

    def mixtures(self, n: int, seed: int = 0, n_spk: int = 2, indices=None, max_sec: float | None = 8.0,
                 overlap=(0.6, 1.0), gain_db: float = 3.0, max_spks: int = 4) -> list[dict]:
        """Overlapped multi-speaker mixtures (ToneLanguage.mixture layout).

        Sources are trimmed to their VAD speech span; speaker k+1 starts 0.1-0.3 s after ``overlap`` x
        (length of speaker k) -> up to 40 % overlap. Each source is scaled to equal speech RMS +- ``gain_db``. ``spk_targets``
        (T, max_spks) holds each source's energy VAD, columns in arrival order."""
        rng = random.Random(seed)
        pool = list(range(len(self)) if indices is None else indices)
        if max_sec:
            pool = [i for i in pool if self.utts[i]["n"] <= max_sec * SR] or pool
        by_spk = defaultdict(list)
        for i in pool:
            by_spk[self.utts[i]["speaker_id"]].append(i)
        spks = sorted(by_spk)
        out = []
        for _ in range(n):
            parts, t0 = [], 0
            for s in rng.sample(spks, n_spk):
                i = rng.choice(by_spk[s])
                x, v = np.array(self.audio(i), np.float32), self.vad(i)
                on = np.nonzero(v > 0.5)[0]
                if len(on):  # trim to the speech span (+-1 frame) like ToneLanguage.render (no lead/tail)
                    a, b = max(0, on[0] - 1) * FRAME, min(len(v), on[-1] + 2) * FRAME
                    x, v = x[a: b], v[a // FRAME: b // FRAME]
                speech = np.repeat(v > 0.5, FRAME)[: len(x)]
                rms = float(np.sqrt(np.mean(x[speech] ** 2))) if speech.any() else float(np.sqrt(np.mean(x ** 2)))
                x *= 0.05 / max(rms, 1e-5) * 10 ** (rng.uniform(-gain_db, gain_db) / 20)
                start = t0 + int(SR * rng.uniform(0.1, 0.3))
                parts.append((i, x, speech, start))
                t0 = start + int(len(x) * rng.uniform(*overlap))
            N = max(st + len(x) for _, x, _, st in parts) + int(0.3 * SR)
            mix = np.zeros(N, np.float32)
            T = ToneLanguage.n_frames(N)
            spk_t = np.zeros((T, max_spks), np.float32)
            items = []
            for k, (i, x, speech, st) in enumerate(parts):
                mix[st: st + len(x)] += x
                act = np.zeros(N, bool)
                act[st: st + len(x)] = speech
                spk_t[:, k] = _frames(act, T)
                items.append(dict(speaker=self.speaker(i), text=self.utts[i]["text"], id=self.utts[i]["id"]))
            mix += 3e-4 * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(N).astype(np.float32)
            peak = float(np.abs(mix).max())
            if peak > 0.95:
                mix *= 0.95 / peak
            out.append(dict(audio=mix, spk_targets=spk_t, speakers=items))
        return out

    def speaker_attributed(self, n: int, seed: int = 0, **kw) -> list[dict]:
        """Mixture + one target speaker (as ``synthetic_dataset("speaker_attributed")``)."""
        rng = random.Random(seed + 7919)
        out = []
        for m in self.mixtures(n, seed, **kw):
            i = rng.randrange(len(m["speakers"]))
            out.append(dict(audio=m["audio"], spk_targets=m["spk_targets"], spk_act=m["spk_targets"][:, i],
                            text=m["speakers"][i]["text"]))
        return out

    def stats(self) -> dict:
        sec = sum(u["n"] for u in self.utts) / SR
        return dict(split=self.split, utts=len(self.utts), hours=round(sec / 3600, 2),
                    speakers=len({u["speaker_id"] for u in self.utts}), speakers_in_split=len(self.speaker_ids),
                    all_utts=len(self.all), all_hours=round(sum(u["n"] for u in self.all) / SR / 3600, 2),
                    window_sec=[self.min_sec, self.max_sec])


# --------------------------------------------------------------------------- batching
def arrange_for_trainer(items: list[dict], batch_size: int, epochs: int, seed: int = 0,
                        jitter_sec: float = 1.0, fit_seed: int = 0, sortagrad: bool = False) -> list[dict]:
    """Duration-bucketed batches through an *unmodified* ``Trainer.fit``.

    ``Trainer.fit`` shuffles ``range(len(train))`` with ``random.Random(0)`` and takes
    consecutive slices of ``batch_size``. We build a list of ``epochs * len(items)``
    references (no audio copies) placed so that fit's first pass over it yields, per virtual
    epoch, batches of similar duration (sort by length + U(0, jitter) noise, cut into batches,
    shuffle batch order). ``sortagrad``: the first virtual epoch runs shortest-first (Deep
    Speech 2's SortaGrad curriculum). If fit's shuffling ever changes, this degrades to plain
    random batches over repeated data (still correct, just more padding)."""
    rng = random.Random(seed)
    order = []
    for e in range(epochs):
        key = [len(ex["audio"]) + rng.random() * jitter_sec * SR for ex in items]
        idx = sorted(range(len(items)), key=key.__getitem__)
        batches = [idx[i: i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)]
        if not (sortagrad and e == 0):
            rng.shuffle(batches)
        order += [j for b in batches for j in b]
    perm = list(range(len(order)))
    random.Random(fit_seed).shuffle(perm)  # exactly what Trainer.fit does on its first pass
    out: list = [None] * len(order)
    for pos, j in enumerate(order):
        out[perm[pos]] = items[j]
    return out


def padding_efficiency(train: list[dict], batch_size: int, steps: int, fit_seed: int = 0) -> float:
    """Real / padded samples over fit's first ``steps`` batches (1.0 = no padding)."""
    idx = list(range(len(train)))
    random.Random(fit_seed).shuffle(idx)
    real = pad = 0
    for k in range(min(steps, len(idx) // batch_size)):
        lens = [len(train[j]["audio"]) for j in idx[k * batch_size: (k + 1) * batch_size]]
        real, pad = real + sum(lens), pad + max(lens) * len(lens)
    return real / max(1, pad)


class QuantizedCollate:
    """``audioforge.data.Collate`` + padding of every batch to a coarse shape grid.

    PyTorch's MPS backend compiles and caches one graph per op per input shape and never evicts
    them. With free-form batch shapes (every batch a new (T, U)) host memory grew 13-20 MB per step
    (~40 GB over a 2800-step run). Padding audio to whole ``audio_sec`` and token sequences to
    multiples of ``tokens`` bounds the shapes to ~12 x 8 combinations; the true lengths stay in
    ``*_len``, so losses and masks are unchanged. Frame targets are padded to match."""

    def __init__(self, tokenizer, audio_sec: float = 1.0, tokens: int = 16):
        from ..data import Collate
        self.base, self.qa, self.qu = Collate(tokenizer), int(audio_sec * SR), tokens

    @staticmethod
    def _pad(t, n):
        import torch.nn.functional as F
        return F.pad(t, (0, 0) * (t.dim() - 2) + (0, n - t.shape[1])) if n > t.shape[1] else t

    def __call__(self, batch: list[dict]) -> dict:
        out = self.base(batch)
        S = -(-out["audio"].shape[1] // self.qa) * self.qa
        out["audio"] = self._pad(out["audio"], S)
        T = ToneLanguage.n_frames(S)
        for k in ("vad", "eou", "spk_targets", "spk_act"):
            if k in out:
                out[k] = self._pad(out[k], T)
        for k in ("text", "source_text", "prompt"):
            if k in out:
                out[k] = self._pad(out[k], -(-out[k].shape[1] // self.qu) * self.qu)
        return out


# --------------------------------------------------------------------------- recipe hook
def recipe_data(cfg: dict, split: str) -> list[dict]:
    """``data: {librispeech: {...}}`` for ``audioforge.train.load_data``.

    Keys: root, train_split (train-clean-100), val_split (defaults to train_split -> a seeded
    held-out subset of it, excluded from training; ``n_val`` or ``val_frac``), max_sec (12),
    min_sec (1), n_train (optional cap), speaker_offset (train utterances: add to every speaker id), seed, sortagrad (first epoch shortest-first), mode (utterances | mixtures | speaker_attributed,
    with n_mixtures: {train, val}), bucket (true: duration-bucketed batches, see
    ``arrange_for_trainer``). Speaker labels of a *different* val split index that split's own
    speakers, so closed-set speaker accuracy is only meaningful with the held-out subset.

    If the recipe is overridden with ``data.synthetic.n_train/n_val`` (tests/test_recipes.py
    smoke test), a tiny synthetic stand-in is returned so the test stays hermetic."""
    d = cfg["data"]
    lc = dict(d["librispeech"])
    mode = lc.get("mode", "utterances")
    syn = d.get("synthetic") or {}
    if f"n_{split}" in syn:
        kind = syn.get("kind", {"utterances": "multitask", "mixtures": "diar"}.get(mode, mode))
        return synthetic_dataset(kind, syn[f"n_{split}"], seed=0 if split == "train" else 1)
    common = dict(root=lc.get("root"), max_sec=lc.get("max_sec", 12.0), min_sec=lc.get("min_sec", 1.0))
    seed = lc.get("seed", 0)
    tr_name = lc.get("train_split", "train-clean-100")
    va_name = lc.get("val_split", tr_name)
    tr = LibriSpeech(tr_name, **common)
    holdout: list[int] = []
    if va_name == tr_name:
        n_val = lc.get("n_val") or int(round(lc.get("val_frac", 0.1) * len(tr)))
        holdout = sorted(random.Random(seed).sample(range(len(tr)), n_val))
    if split == "train":
        held = set(holdout)
        idx = [i for i in range(len(tr)) if i not in held]
        if lc.get("n_train"):
            idx = sorted(random.Random(seed + 1).sample(idx, min(len(idx), lc["n_train"])))
        if mode != "utterances":
            n = lc.get("n_mixtures", {}).get("train", 5000)
            return getattr(tr, mode)(n, seed=seed, indices=idx)
        data = tr.utterances(idx)
        if lc.get("speaker_offset"):  # namespace the split's speaker ids (e.g. 1000+) so they can join AMI's 0-189
            for ex in data:
                ex["speaker"] = int(ex["speaker"]) + int(lc["speaker_offset"])
        if lc.get("bucket", True):
            t = cfg.get("trainer", {})
            bs, steps = t.get("batch_size", 16), t.get("max_steps", 1000)
            epochs = max(1, math.ceil(steps / max(1, len(data) // bs)))
            data = arrange_for_trainer(data, bs, epochs, seed=seed, sortagrad=lc.get("sortagrad", False))
            log.info(f"[librispeech] {tr_name}: {len(idx)} train utts x {epochs} bucketed epochs, "
                  f"padding efficiency {padding_efficiency(data, bs, steps):.2f}")
        return data
    ds = tr if va_name == tr_name else LibriSpeech(va_name, **common)
    if not holdout:
        holdout = sorted(random.Random(seed).sample(range(len(ds)), min(len(ds), lc.get("n_val", 200))))
    if mode != "utterances":
        return getattr(ds, mode)(lc.get("n_mixtures", {}).get("val", 200), seed=seed + 1, indices=holdout)
    return ds.utterances(holdout)
