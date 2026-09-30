#!/bin/zsh
# Drive a resumable bench stage under the machine rules: one process at a time, <= 10 min each, a new process only
# when the 1-min load is below $MAXLOAD (default 9), repeated until the stage reports nothing left.
#   scripts/archive/run_dyadic_bench.sh <script.py> <args...>
#   e.g. scripts/archive/run_dyadic_bench.sh bench_turn_dyadic.py --corpus oto --stage head --n-conv 12
set -u
script=$1; shift
MAXLOAD=${MAXLOAD:-9}
cd "$(dirname "$0")/../.."
for i in {1..60}; do
  while true; do
    load=$(sysctl -n vm.loadavg | awk '{print $2}')
    if (( $(echo "$load < $MAXLOAD" | bc -l) )); then break; fi
    sleep 30
  done
  out=$(PYTHONPATH=. .venv/bin/python scripts/research/$script --budget 480 "$@" 2>&1 | grep -v -i warning)
  echo "$out" | tail -2
  if echo "$out" | grep -q -E "0 left|'todo': 0|^0$|^wrote "; then echo "== $script $* done"; exit 0; fi
  if echo "$out" | grep -q -i "Traceback"; then echo "$out" | tail -25; exit 1; fi
done
