#!/bin/zsh
# Run a command with every line of its output in a log under runs/logs/, readable from the tail at any moment
# (docs/PROJECT.md "Commands": tee to a log under runs/, never a buffering pipe such as `| tail`).
#
#   scripts/dev/logged.sh <name> <command...>      foreground; output on the terminal and in the log
#   scripts/dev/logged.sh -b <name> <command...>   background; prints the pid and the log path, returns at once
#
# The log is runs/logs/<name>_<UTC stamp>.log (git-ignored) and ends with an "== exit <code>" line, so a finished
# run is visible from the tail. The exit code is the command's, not tee's. Compose with the machine gate:
#   scripts/dev/logged.sh -b sweep scripts/dev/gate.sh .venv/bin/python scripts/research/<driver>.py ...
set -u
setopt pipefail
ROOT=${0:A:h:h:h}
bg=0
if [ "${1:-}" = "-b" ]; then bg=1; shift; fi
if [ $# -lt 2 ]; then echo "usage: scripts/dev/logged.sh [-b] <name> <command...>" >&2; exit 2; fi
name=$1; shift
mkdir -p "$ROOT/runs/logs"
LOG="$ROOT/runs/logs/${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
export PYTHONUNBUFFERED=1
if [ $bg -eq 1 ]; then
  LOG="$LOG" nohup zsh -c '"$@" >>"$LOG" 2>&1; echo "== exit $? $(date -u +%FT%TZ)" >>"$LOG"' logged "$@" </dev/null >/dev/null 2>&1 &
  echo "pid $! log ${LOG#$ROOT/}"
  echo "follow: tail -n 20 ${LOG#$ROOT/}"
  exit 0
fi
echo "log ${LOG#$ROOT/}"
"$@" 2>&1 | tee -a "$LOG"
rc=$?
echo "== exit $rc $(date -u +%FT%TZ)" >>"$LOG"
exit $rc
