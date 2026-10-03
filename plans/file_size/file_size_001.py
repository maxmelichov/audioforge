#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""List every tracked source/doc file over the AGENTS.md 700-line limit, grouped.

Usage: uv run plans/file_size/file_size_001.py [--limit 700] [--repo PATH]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXTS = {".py", ".md", ".js", ".ts", ".tsx", ".jsx", ".sh", ".toml", ".yaml", ".yml", ".html", ".css", ".rs", ".txt"}
GROUPS = [
    ("product package (audioforge/)", lambda p: p.startswith("audioforge/")),
    ("tests/", lambda p: p.startswith("tests/")),
    ("scripts/research/", lambda p: p.startswith("scripts/research/")),
    ("scripts/ (top)", lambda p: p.startswith("scripts/")),
    ("research/ docs", lambda p: p.startswith("research/")),
    ("demo/archive", lambda p: p.startswith("demo/archive/")),
    ("other", lambda p: True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=700)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args()
    repo = Path(a.repo)
    files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split("\n")
    over: dict[str, list[tuple[int, str]]] = {g: [] for g, _ in GROUPS}
    for f in filter(None, files):
        path = repo / f
        if Path(f).suffix not in EXTS or not path.is_file():
            continue
        try:
            n = sum(1 for _ in path.open("rb"))
        except OSError:
            continue
        if n > a.limit:
            group = next(g for g, test in GROUPS if test(f))
            over[group].append((n, f))
    total = 0
    for g, _ in GROUPS:
        rows = sorted(over[g], reverse=True)
        total += len(rows)
        print(f"## {g}: {len(rows)} file(s) over {a.limit}")
        for n, f in rows:
            print(f"  {n:6d}  {f}")
        print()
    print(f"TOTAL: {total} tracked file(s) over {a.limit} lines")


if __name__ == "__main__":
    main()
