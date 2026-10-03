#!/bin/zsh
# Build the public snapshot of the repository and check it: the committed tree only (git archive HEAD, so nothing
# untracked or uncommitted leaks), minus the LID heads that are not cleared for redistribution (assets/lid_*.pt,
# docs/RELEASE_CHECKLIST.md §3), then a secrets scan and the full test suite inside the snapshot.
#
#   scripts/dev/build_public.sh [OUT_DIR]      default OUT_DIR: $TMPDIR/audioforge_pub; it must not exist yet
#
# Run it through scripts/dev/logged.sh (chore build-public does): the test suite takes minutes.
# The scan prints file names only, never the matched text, so a token never reaches a terminal or a log.
set -eu
setopt pipefail
ROOT=${0:A:h:h:h}
OUT=${1:-${TMPDIR:-/tmp}/audioforge_pub}
OUT=${OUT:A}
PY=${PY:-$ROOT/.venv/bin/python}
case "$OUT" in "$ROOT"|"$ROOT"/*) echo "build-public: OUT_DIR must be outside the repository ($OUT)" >&2; exit 2;; esac
if [ -e "$OUT" ]; then echo "build-public: $OUT exists; remove it or pass another OUT_DIR" >&2; exit 2; fi
mkdir -p "$OUT"
echo "== snapshot $(git -C "$ROOT" rev-parse --short HEAD) -> $OUT"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$OUT" --exclude 'assets/lid_*.pt'
if ls "$OUT"/assets/lid_*.pt >/dev/null 2>&1; then echo "build-public: assets/lid_*.pt still present" >&2; exit 1; fi
echo "== files: $(find "$OUT" -type f | wc -l | tr -d ' '), size: $(du -sh "$OUT" | cut -f1)"
echo "== secrets scan (file names only)"
hits=$(grep -rlIE 'hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' \
  "$OUT" --exclude-dir=raw || true)
if [ -n "$hits" ]; then echo "build-public: possible secrets in:" >&2; echo "$hits" >&2; exit 1; fi
echo "clean"
echo "== pytest in the snapshot ($PY)"
cd "$OUT"
"$PY" -m pytest -q
echo "== build-public OK: $OUT"
