"""External diarizer activity tracks for AMI / ICSI turn examples (research/STAGE1.md, "Turn head with an external
diarizer" -> fix (1): train the turn head's ``spk_act`` conditioning on real diarizer tracks).

Cache layout (written by scripts/research/make_sortformer_tracks.py --dataset ami|icsi; <root> = data/ami or data/icsi,
the dataset's own root, so ``cache_dir(split=..., dataset="icsi")`` = data/icsi/cache/sortformer/<split>):

    <root>/cache/sortformer/<split>/<key>.npy          offline track: one Sortformer head pass over the window
    <root>/cache/sortformer/<split>/<key>.stream.npy   streaming track (StreamingDiarizer, card low-latency config)
    <root>/cache/sortformer/<split>/manifest.json      checkpoint, window_sec, stream config, keys per source

Each track is the diarizer's full (T_d, S) sigmoid output (S = 4), on the same 80 ms grid as the example.
``key`` = ``example_key(ex)`` = meeting + window start (ms) + window length (samples): deterministic, and a
window built with other settings (e.g. another window_sec) simply has no track.

Window sets with a non-default trail (``turn_examples(trail_sec=...)`` != 2 s, e.g. the 6 s post-end trail of
research/recipes/stage1_turn_v3_trail6.yaml) live in their own directory ``<split>_trail<trail:g>`` (``split_dirname``; e.g.
data/ami/cache/sortformer/train_trail6/), so they never mix with (or overwrite) the default 2 s-trail caches:
``cache_dir(split="train", trail_sec=6.0)``; ``attach(..., trail_sec=6.0)``; ami.recipe_data passes the recipe's
``trail_sec``. An explicit ``directory`` / recipe ``ext_tracks.dir`` always wins.

``attach(examples, ...)`` adds to each turn example
    spk_act_ext      (T,) float32 = the track's column chosen by "enrollment by who is talking" (enroll_column:
                     largest overlap with the oracle primary on frames [onset, turn_end) - never after turn_end;
                     the rule of scripts/research/eval_stage1.py), cropped / edge-padded to the example's T;
    spk_targets_ext  (T, S) float32 = the whole track, cropped / edge-padded to T;
    spk_prim_ext     (S,) float32 = one-hot of the chosen column (TurnHead act_columns > 1 reads all S columns
                     plus which one is the primary).
Examples without a cached track get all three filled with -1 (``MISSING``) so every item of a batch has the keys
(data.Collate refuses partial keys); SpeechModel.conditioning_act never uses such an item's ext track.
``fallback`` (e.g. source="stream", fallback="offline"): items without a track of ``source`` take the fallback
source's track (prefer the streaming track when present - what the head sees at inference - else the offline one).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SOURCES = ("offline", "stream")
DATASETS = ("ami", "icsi")  # ICSI subclasses ami.AMI: same turn windows / keys, its own root (data/icsi)
MISSING = -1.0
DEFAULT_TRAIL_SEC = 2.0  # ami.AMI.turn_examples default: the caches data/<ds>/cache/sortformer/<split>/ hold these windows


def example_key(ex: dict) -> str:
    """Deterministic id of an AMI / ICSI window: <meeting>_<start ms>_<n samples> (meeting names never collide)."""
    return f"{ex['meeting']}_{int(round(float(ex['start']) * 1000)):08d}_{len(ex['audio'])}"


def dataset_root(root=None, dataset: str = "ami") -> Path:
    """``root`` if given, else the dataset's default root (data/ami, data/icsi)."""
    assert dataset in DATASETS, f"dataset must be one of {DATASETS}, got {dataset!r}"
    if dataset == "icsi":
        from .icsi import _root
    else:
        from .ami import _root
    return _root(root)


def split_dirname(split: str, trail_sec: float | None = None) -> str:
    """Cache sub-directory of a window set: ``split`` for the default 2 s trail (or None), else
    ``<split>_trail<trail_sec:g>`` (6.0 -> train_trail6, 3.5 -> train_trail3.5)."""
    if trail_sec is None or abs(float(trail_sec) - DEFAULT_TRAIL_SEC) < 1e-9:
        return split
    return f"{split}_trail{float(trail_sec):g}"


def cache_dir(root=None, split: str = "train", dataset: str = "ami", trail_sec: float | None = None) -> Path:
    return dataset_root(root, dataset) / "cache" / "sortformer" / split_dirname(split, trail_sec)


def track_path(d: Path, key: str, source: str = "offline") -> Path:
    assert source in SOURCES, f"track source must be one of {SOURCES}, got {source!r}"
    return Path(d) / (f"{key}.npy" if source == "offline" else f"{key}.stream.npy")


def onset_end(spk_act) -> tuple[int, int]:
    """First active frame and one past the last active frame of the oracle primary (heads.turn._onset_end)."""
    nz = np.nonzero(np.asarray(spk_act) > 0.5)[0]
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)


def enroll_column(p: np.ndarray, ref: np.ndarray, onset: int, end: int) -> int:
    """'Enrollment by who is talking' (identical to scripts/research/eval_stage1.py enroll_column): the diarizer
    column that best overlaps the oracle primary activity on frames [onset, end) only - never after the turn
    end. Hard overlap (p > 0.5 on primary frames), ties broken by the soft overlap."""
    T = min(len(p), len(ref), end)
    pp, rr = p[onset:T], ref[onset:T] > 0.5
    hard = ((pp > 0.5) & rr[:, None]).sum(0)
    soft = (pp * rr[:, None]).sum(0)
    return int(np.lexsort((-soft, -hard))[0])


def fit(a: np.ndarray, T: int) -> np.ndarray:
    """Crop / edge-pad a (T_d,) or (T_d, S) track to T frames (as scripts/research/eval_stage1.py _fit)."""
    a = np.asarray(a, np.float32)
    if len(a) >= T:
        return a[:T]
    pad = np.repeat(a[-1:], T - len(a), 0) if len(a) else np.zeros((T - len(a), *a.shape[1:]), np.float32)
    return np.concatenate([a, pad], 0)


def ext_fields(track: np.ndarray, ex: dict) -> dict:
    """{spk_act_ext (T,), spk_targets_ext (T, S), ext_column} for one example from its (T_d, S) track."""
    ref = np.asarray(ex["spk_act"])
    T = len(ref)
    on, en = onset_end(ref)
    c = enroll_column(track, ref, on, en)
    return {"spk_act_ext": fit(track[:, c], T), "spk_targets_ext": fit(track, T), "ext_column": c,
            "spk_prim_ext": np.eye(track.shape[1], dtype=np.float32)[c]}


def attach(examples: list[dict], root=None, split: str = "train", source: str = "offline",
           directory=None, num_spks: int = 4, require: bool = False, fallback: str | None = None,
           dataset: str = "ami", trail_sec: float | None = None) -> dict:
    """Add spk_act_ext / spk_targets_ext / spk_prim_ext to every example in place (see module docstring). Returns
    {found, missing} (+ {fallback: n} when a fallback source is given). ``require``: raise if any example has no
    track (of source or fallback). ``directory`` overrides the cache dir of (root, split, dataset, trail_sec)."""
    d = Path(directory) if directory else cache_dir(root, split, dataset, trail_sec)
    found = missing = fell = 0
    for ex in examples:
        key = example_key(ex)
        p = track_path(d, key, source)
        T = len(np.asarray(ex["spk_act"]))
        if not p.exists() and fallback and fallback != source and track_path(d, key, fallback).exists():
            p = track_path(d, key, fallback)
            fell += 1
        if p.exists():
            f = ext_fields(np.load(p), ex)
            ex["spk_act_ext"], ex["spk_targets_ext"] = f["spk_act_ext"], f["spk_targets_ext"]
            ex["spk_prim_ext"] = f["spk_prim_ext"]
            found += 1
        else:
            if require:
                raise FileNotFoundError(f"no {source} track for {key} in {d} (scripts/research/make_sortformer_tracks.py)")
            ex["spk_act_ext"] = np.full(T, MISSING, np.float32)
            ex["spk_targets_ext"] = np.full((T, num_spks), MISSING, np.float32)
            ex["spk_prim_ext"] = np.full(num_spks, MISSING, np.float32)
            missing += 1
    res = {"found": found, "missing": missing}
    if fallback:
        res["fallback"] = fell
    return res


def has_tracks(root=None, split: str = "train", source: str = "offline", directory=None,
               dataset: str = "ami", trail_sec: float | None = None) -> bool:
    d = Path(directory) if directory else cache_dir(root, split, dataset, trail_sec)
    return d.is_dir() and any(d.glob("*.stream.npy" if source == "stream" else "*[0-9].npy"))


def act_stats(examples: list[dict], key: str = "spk_act_ext") -> dict:
    """Primary-activity frame errors of the attached track vs the oracle spk_act (scripts/research/eval_stage1.py
    act_stats): miss / FA per primary speech frame, FA per non-primary frame, post-turn-end frames active."""
    miss = fa = sp = nonsp = post_on = post = n = 0
    for ex in examples:
        a = np.asarray(ex[key])
        if (a < 0).any():
            continue
        n += 1
        r = np.asarray(ex["spk_act"]) > 0.5
        h = a > 0.5
        _, e = onset_end(ex["spk_act"])
        miss += int((r & ~h).sum()); fa += int((~r & h).sum())
        sp += int(r.sum()); nonsp += int((~r).sum())
        post_on += int(h[e:].sum()); post += max(0, len(r) - e)
    return {"n": n, "miss": round(miss / max(1, sp), 4), "fa_per_speech": round(fa / max(1, sp), 4),
            "fa_per_nonspeech": round(fa / max(1, nonsp), 4), "post_end_active": round(post_on / max(1, post), 4)}


def manifest(root=None, split: str = "train", directory=None, dataset: str = "ami",
             trail_sec: float | None = None) -> dict:
    p = (Path(directory) if directory else cache_dir(root, split, dataset, trail_sec)) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":  # activity-vs-oracle statistics of a cached split
    import argparse
    ap = argparse.ArgumentParser(description="miss / FA of cached external tracks vs the oracle primary")
    ap.add_argument("--dataset", default="ami", choices=DATASETS)
    ap.add_argument("--split", default="train", choices=["train", "dev", "eval"])
    ap.add_argument("--track-source", default="offline", choices=SOURCES)
    ap.add_argument("--window-sec", type=float, default=None, help="default: from the manifest")
    ap.add_argument("--trail-sec", type=float, default=DEFAULT_TRAIL_SEC, help="window set (cache <split>_trail<x>)")
    a = ap.parse_args()
    if a.dataset == "icsi":
        from .icsi import DEFAULT_MEETINGS
        from .icsi import ICSI as DS
    else:
        from .ami import AMI as DS
        from .ami import DEFAULT_MEETINGS
    man = manifest(split=a.split, dataset=a.dataset, trail_sec=a.trail_sec)
    ws = a.window_sec or man.get("window_sec", 16.0 if a.split == "train" else 20.0)
    ds = DS(man.get("meetings") or DEFAULT_MEETINGS[a.split], verbose=False)
    exs = ds.turn_examples(ws, trail_sec=a.trail_sec)
    print(json.dumps({"dataset": a.dataset, "split": a.split, "source": a.track_source, "window_sec": ws,
                      "trail_sec": a.trail_sec, **attach(exs, split=a.split, source=a.track_source, dataset=a.dataset,
                                                         trail_sec=a.trail_sec), **act_stats(exs)}))
