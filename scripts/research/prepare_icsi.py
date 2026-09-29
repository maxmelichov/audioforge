#!/usr/bin/env python
"""Download, verify, convert and cache an ICSI Meeting Corpus subset (headset mix + NXT core annotations v1.0).

    .venv/bin/python scripts/research/prepare_icsi.py                         # default 12/2/3 meetings (~15.3 h, 1.8 GB wav)
    .venv/bin/python scripts/research/prepare_icsi.py --n-train 30            # more train meetings (list order after defaults)
    .venv/bin/python scripts/research/prepare_icsi.py --stats-only --all      # annotation statistics of all 75 meetings, no audio

Steps (all resumable; rerun after an interruption or a --max-minutes stop):
  1. ICSI_core_NXT.zip (19.5 MB, md5-checked) -> data/icsi/raw/ICSI/{Words,Segments,...}
  2. conversion to AMI-format timed words + untimed zones -> data/icsi/annotations/ (see datasets/icsi.py)
  3. <meeting>.interaction.wav (the distributed headset mix, 16 kHz mono; <= 4 parallel connections;
     byte count vs the served Content-Length, header, md5 pinned on first download) -> data/icsi/audio/
  4. float32 .npy audio cache per meeting -> data/icsi/cache/ (skip with --no-cache)
  5. label statistics per split (same as AMI's) -> data/icsi/stats.json (or --json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audioforge.datasets.icsi import (DEFAULT_ROOT, ICSI, LABEL_DEFAULTS, SPLITS, WAV_BYTES,  # noqa: E402
                                      download_annotations, download_audio, subset)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="every meeting of every split (8.3 GB of audio)")
    ap.add_argument("--connections", type=int, default=4, help="parallel downloads (capped at 4)")
    ap.add_argument("--max-minutes", type=float, default=4.0, help="stop downloading after this long (rerun resumes)")
    ap.add_argument("--no-cache", action="store_true", help="do not build the .npy audio cache")
    ap.add_argument("--stats-only", action="store_true", help="annotations + statistics, no audio")
    ap.add_argument("--json", default=None, help="write statistics here (default <root>/stats.json)")
    for k, v in LABEL_DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()
    download_annotations(root)
    n = {s: len(v) for s, v in SPLITS.items()} if a.all else {"train": a.n_train, "dev": a.n_dev, "eval": a.n_eval}
    sub = subset(n)
    for s, ms in sub.items():
        print(f"{s}: {len(ms)} meetings, {sum(WAV_BYTES[m] for m in ms) / 1e9:.2f} GB wav: {' '.join(ms)}", flush=True)
    if not a.stats_only:
        allm = [m for ms in sub.values() for m in ms]
        st = download_audio(allm, root, a.connections, a.max_minutes)
        bad = {m: v for m, v in st.items() if not v.startswith("ok")}
        print(f"audio: {len(st) - len(bad)}/{len(st)} verified ({time.time() - t0:.0f}s)", flush=True)
        if bad:
            print(f"incomplete: {bad} -> rerun to resume", flush=True)
            return 1
    labels = {k: getattr(a, k) for k in LABEL_DEFAULTS}
    out = {}
    for s, ms in sub.items():
        ds = ICSI(ms, root, audio=not (a.stats_only or a.no_cache), **labels)
        out[s] = ds.stats()
        out[s]["meeting_list"] = ms
        print(s, json.dumps({k: v for k, v in out[s].items() if k not in ("label_cfg", "meeting_list")}), flush=True)
    Path(a.json or root / "stats.json").write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
