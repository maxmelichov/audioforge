# /// script
# requires-python = ">=3.10"
# dependencies = ["audioforge"]
#
# [tool.uv.sources]
# audioforge = { path = "../..", editable = true }
# ///
"""Regenerate the flag reference of docs/CONFIGURATION.md from the server's flag table.

    python scripts/dev/gen_config_doc.py           # rewrite the generated block in place
    python scripts/dev/gen_config_doc.py --check   # exit 1 if the page is stale (CI and tests/test_config_doc.py)

The block between the BEGIN / END markers is ``audioforge.server.cli.reference_markdown()``; edit the flags in
``audioforge/server/cli.py`` (``FLAGS``, ``ENV_VARS``), never the block itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs" / "CONFIGURATION.md"
BEGIN = "<!-- BEGIN GENERATED FLAGS: scripts/dev/gen_config_doc.py writes this block from audioforge/server/cli.py -->"
END = "<!-- END GENERATED FLAGS -->"


def render(text: str) -> str:
    from audioforge.server.cli import reference_markdown

    if BEGIN not in text or END not in text:
        raise SystemExit(f"{DOC}: markers not found")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return head + BEGIN + "\n\n" + reference_markdown() + "\n" + END + tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if the page is stale instead of writing it")
    a = ap.parse_args(argv)
    old = DOC.read_text()
    new = render(old)
    if a.check:
        if new != old:
            print(f"{DOC.relative_to(ROOT)} is stale: run python scripts/dev/gen_config_doc.py", file=sys.stderr)
            return 1
        return 0
    if new != old:
        DOC.write_text(new)
        print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
