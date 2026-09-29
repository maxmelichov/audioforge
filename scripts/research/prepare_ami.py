#!/usr/bin/env python
"""Download, verify and cache an AMI Meeting Corpus subset (headset mix + manual annotations v1.6.2).

    .venv/bin/python scripts/research/prepare_ami.py                          # default 12/4/4 meetings (~12 h)
    .venv/bin/python scripts/research/prepare_ami.py --n-train 40 --n-dev 8 --n-eval 8
    .venv/bin/python scripts/research/prepare_ami.py --stats-only --json research/ami_stats.json

Steps (all resumable; rerun after an interruption or a --max-minutes stop):
  1. split lists (BUT/pyannote AMI-diarization-setup, Full-corpus-ASR partition) -> data/ami/lists/
  2. ami_public_manual_1.6.2.zip (md5-checked) -> data/ami/annotations/{words,segments,corpusResources}
  3. <meeting>.Mix-Headset.wav (<= 4 parallel connections, size/header/md5 verified) -> data/ami/audio/
  4. float32 .npy audio cache per meeting -> data/ami/cache/ (skip with --no-cache)
  5. label statistics per split (turns, switch gaps vs hesitations, overlap)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audioforge.datasets.ami import (AMI, DEFAULT_ROOT, LABEL_DEFAULTS, download_annotations,  # noqa: E402
                                     download_audio, fetch_split_lists, subset)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=4)
    ap.add_argument("--connections", type=int, default=4, help="parallel downloads (capped at 4)")
    ap.add_argument("--max-minutes", type=float, default=4.5, help="stop downloading after this long (rerun resumes)")
    ap.add_argument("--no-cache", action="store_true", help="do not build the .npy audio cache")
    ap.add_argument("--stats-only", action="store_true", help="annotations + statistics, no audio")
    ap.add_argument("--json", default=None, help="write statistics to this file")
    for k, v in LABEL_DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()
    fetch_split_lists(root)
    download_annotations(root)
    sub = subset({"train": a.n_train, "dev": a.n_dev, "eval": a.n_eval}, root)
    for s, ms in sub.items():
        print(f"{s}: {' '.join(ms)}", flush=True)
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
        ds = AMI(ms, root, audio=not (a.stats_only or a.no_cache), **labels)
        out[s] = ds.stats()
        out[s]["meeting_list"] = ms
        print(s, json.dumps({k: v for k, v in out[s].items() if k not in ("label_cfg", "meeting_list")}), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
