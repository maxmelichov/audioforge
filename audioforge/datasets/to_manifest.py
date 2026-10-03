"""Adapter: any recipe data source -> the trainer's one data format (a JSONL manifest, data.ManifestData).

    python -m audioforge.datasets.to_manifest RECIPE -o data/manifests/<name> [key=value ...]

Loads the recipe's ``data`` block with the trainer's own loader (librispeech / ami / icsi / dyadic / smartturn /
synthetic, or a ``mix`` flattened into one list) and writes ``<out>/train.jsonl`` and ``<out>/val.jsonl`` with
data.write_manifest: float32 WAV per row (lossless) plus an npz of its array labels. The recipe then trains from

    data: {manifest: {train: <out>/train.jsonl, val: <out>/val.jsonl}}

and the rows are identical to what the source produced (tests/test_train_loop.py checks the round trip).
A ``mix`` loses its per-source batching (``batching: source``) when flattened: convert each source on its own and
keep the mix for that.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..data import write_manifest

log = logging.getLogger(__name__)


def convert(cfg: dict, out_dir: str | Path, splits=("train", "val")) -> dict[str, Path]:
    from ..train import load_data
    sr = cfg.get("sample_rate", 16000)
    return {sp: write_manifest(load_data(cfg, sp), out_dir, sp, sr) for sp in splits}


def main(argv=None) -> int:
    from ..train import load_recipe
    ap = argparse.ArgumentParser(prog="python -m audioforge.datasets.to_manifest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("overrides", nargs="*", help="dotted.key=value recipe overrides")
    ap.add_argument("-o", "--out", required=True, help="output directory (manifests, audio/, labels/)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = convert(load_recipe(a.recipe, a.overrides), a.out, a.splits)
    for sp, p in paths.items():
        log.info(f"{sp}: {p} ({sum(1 for _ in open(p))} rows)")
    log.info("data: {manifest: {%s}}" % ", ".join(f"{sp}: {p}" for sp, p in paths.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
