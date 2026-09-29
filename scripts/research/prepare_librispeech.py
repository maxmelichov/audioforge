#!/usr/bin/env python
"""Download, verify, extract and cache LibriSpeech splits for audioforge.

    .venv/bin/python scripts/research/prepare_librispeech.py --splits dev-clean train-clean-100
    .venv/bin/python scripts/research/prepare_librispeech.py --splits dev-clean --show-vad 3

Per split: resumable parallel-range download from openslr.org (md5-checked), extraction to
data/librispeech/LibriSpeech/<split>/, a transcript/duration index, and a memory-mapped float32
audio cache + energy-VAD labels for utterances in [--min-sec, --max-sec] under data/librispeech/cache/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audioforge.datasets.librispeech import (DEFAULT_ROOT, SPLITS, LibriSpeech,  # noqa: E402
                                             download, extract, frame_db)


def show_vad(ds: LibriSpeech, n: int, seed: int = 1):
    for i in random.Random(seed).sample(range(len(ds)), n):
        e = ds.example(i)
        db = frame_db(np.asarray(e["audio"]), len(e["vad"]))
        rel = db - db.max()
        print(f"\n{e['id']} {e['duration']:.2f}s speaker={e['speaker']} \"{e['text'][:60]}...\"")
        print(" energy:", "".join("#" if r > -10 else "+" if r > -20 else "-" if r > -30 else "." for r in rel),
              " (# >-10, + >-20, - >-30, . <=-30 dB re max)")
        print(" vad   :", "".join("1" if v else "_" for v in e["vad"]))
        print(" eou   :", "".join("E" if v else " " for v in e["eou"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["dev-clean", "train-clean-100"], choices=sorted(SPLITS))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-sec", type=float, default=12.0, help="cache utterances up to this long (0 = all)")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--connections", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true", help="only download + extract + index")
    ap.add_argument("--delete-archive", action="store_true")
    ap.add_argument("--show-vad", type=int, default=0, help="print energy / VAD / EOU for N utterances")
    a = ap.parse_args(argv)
    root = Path(a.root)
    for split in a.splits:
        print(f"== {split} ({SPLITS[split]})", flush=True)
        if not ((root / "cache" / f"{split}.extracted").exists() and (root / "LibriSpeech" / split).exists()):
            download(split, root, a.connections)
            extract(split, root)
        if a.delete_archive and (root / f"{split}.tar.gz").exists():
            (root / f"{split}.tar.gz").unlink()
        ds = LibriSpeech(split, root, max_sec=a.max_sec or None, min_sec=a.min_sec, cache=not a.no_cache)
        print(json.dumps(ds.stats()), flush=True)
        if a.show_vad:
            show_vad(ds, a.show_vad)


if __name__ == "__main__":
    main()
