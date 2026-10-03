#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Check the repository rules of AGENTS.md / docs/PROJECT.md mechanically.

    uv run plans/repo_rules/repo_rules_001.py [--base 4895faf] [--limit 700]
    chore rules

Checks (FAIL makes the exit code 1; REPORT only lists):
  1. root      FAIL   the root holds only the project files and top directories (tracked and untracked, not ignored)
  2. size      REPORT tracked text files over 700 lines; files named in plans/file_size/*.md are marked "known"
  3. pep723    FAIL   every scripts/*.py, scripts/dev/*.py and plans/*/*.py carries a `# /// script` block
  4. chore     FAIL   the chorefile is at the root (and `chore list` works when chore is installed)
  5. commits   FAIL   no Co-Authored-By or Claude-Session trailer in the commits after --base
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {"README.md", "LICENSE", "NOTICE", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md", "AGENTS.md",
              "CLAUDE.md", "pyproject.toml", "chorefile", ".gitignore", ".pre-commit-config.yaml"}
ROOT_DIRS = {".github", "assets", "audioforge", "demo", "docs", "examples", "integrations", "packages", "plans",
             "research", "runs", "scripts", "tests"}
TEXT_EXTS = {".py", ".md", ".js", ".ts", ".tsx", ".jsx", ".sh", ".toml", ".yaml", ".yml", ".html", ".css", ".rs", ".txt"}
PEP723 = re.compile(r"^# /// script$.*?^# ///$", re.M | re.S)
TRAILER = re.compile(r"^(Co-Authored-By|Claude-Session):", re.M | re.I)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def check_root() -> list[str]:
    tracked = {f.split("/", 1)[0] for f in git("ls-files").splitlines() if f}
    untracked = {f.split("/", 1)[0] for f in git("ls-files", "--others", "--exclude-standard").splitlines() if f}
    bad = []
    for name in sorted(tracked | untracked):
        ok = name in ROOT_FILES if (ROOT / name).is_file() or (ROOT / name).is_symlink() else name in ROOT_DIRS
        if not ok:
            bad.append(f"{name}{'' if name in tracked else ' (untracked)'}")
    return bad


def check_size(limit: int) -> list[str]:
    known = " ".join(p.read_text() for p in (ROOT / "plans" / "file_size").glob("*.md"))
    rows = []
    for f in filter(None, git("ls-files").splitlines()):
        p = ROOT / f
        if p.suffix not in TEXT_EXTS or not p.is_file() or p.is_symlink():
            continue
        n = sum(1 for _ in p.open("rb"))
        if n > limit:
            mark = "known" if f in known or (f.startswith("audioforge/") and f[len("audioforge/"):] in known) else "new"
            rows.append((n, f, mark))
    return [f"{n:6d}  {mark:5s}  {f}" for n, f, mark in sorted(rows, reverse=True)]


def check_pep723() -> tuple[list[str], int]:
    files = sorted({*ROOT.glob("scripts/*.py"), *ROOT.glob("scripts/dev/*.py"), *ROOT.glob("plans/*/*.py")})
    missing = [str(p.relative_to(ROOT)) for p in files if not PEP723.search(p.read_text())]
    return missing, len(files)


def check_chore() -> tuple[list[str], str]:
    if not (ROOT / "chorefile").is_file():
        return ["chorefile missing at the root"], ""
    exe = shutil.which("chore")
    if not exe:
        return [], "chore not installed: chorefile present, `chore list` not run"
    r = subprocess.run([exe, "list"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return [f"`chore list` failed: {r.stderr.strip()}"], ""
    n = sum(1 for line in r.stdout.splitlines() if line.startswith("  "))
    return [], f"`chore list` OK, {n} tasks"


def check_commits(base: str) -> tuple[list[str], int]:
    log = git("log", f"{base}..HEAD", "--format=%h %s%x00%B%x1e")
    entries = [e.strip() for e in log.split("\x1e") if e.strip()]
    bad = []
    for e in entries:
        head, body = e.split("\x00", 1)
        if TRAILER.search(body):
            bad.append(head)
    return bad, len(entries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="4895faf", help="check commit messages after this commit")
    ap.add_argument("--limit", type=int, default=700)
    a = ap.parse_args()
    failed = []

    bad = check_root()
    print(f"[{'FAIL' if bad else 'ok'}] root: {len(bad)} entr{'y' if len(bad) == 1 else 'ies'} outside the allowed set")
    for b in bad:
        print(f"    {b}")
    failed += ["root"] * bool(bad)

    rows = check_size(a.limit)
    new = sum(r.split()[1] == "new" for r in rows)
    print(f"[report] size: {len(rows)} tracked text file(s) over {a.limit} lines ({len(rows) - new} known in "
          f"plans/file_size/, {new} not in that plan)")
    for r in rows:
        print(f"    {r}")

    missing, n = check_pep723()
    print(f"[{'FAIL' if missing else 'ok'}] pep723: {n - len(missing)}/{n} scripts carry a `# /// script` block")
    for m in missing:
        print(f"    {m}")
    failed += ["pep723"] * bool(missing)

    bad, info = check_chore()
    print(f"[{'FAIL' if bad else 'ok'}] chore: {'; '.join(bad) or info}")
    failed += ["chore"] * bool(bad)

    bad, n = check_commits(a.base)
    print(f"[{'FAIL' if bad else 'ok'}] commits: {len(bad)}/{n} commit(s) after {a.base} carry a trailer")
    for b in bad:
        print(f"    {b}")
    failed += ["commits"] * bool(bad)

    print(f"\nRESULT: {'FAIL (' + ', '.join(failed) + ')' if failed else 'PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
