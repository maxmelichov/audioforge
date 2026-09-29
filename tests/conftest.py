"""Shared pytest setup: make the research drivers importable.

The tests import the research scripts by module name (``import eval_stage1``) or by file path. The scripts live in
``scripts/research/`` (user-facing ones stay in ``scripts/``); both directories go on ``sys.path`` here so the tests
and the scripts' own sibling imports resolve without per-file path juggling.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT, ROOT / "scripts", ROOT / "scripts" / "research"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
