#!/bin/zsh
# Held-out scans of turn_clean.py (CPU policy replays, no torch work: one niced single-thread process), one call (< 10 min) at a time. Usage: turn_clean_scan.sh <sys> <tag:fam> ...
cd ${0:A:h:h:h}
sys=$1; shift
for tf in "$@"; do
  tag=${tf%%:*}; fam=${tf##*:}
  for i in $(seq 1 12); do
    out=$(PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 5 .venv/bin/python scripts/research/turn_clean.py scan --sys $sys --tag $tag --fam $fam --budget 480 2>&1 | tail -1)
    echo "$out"
    [[ "$out" == *"done "* ]] && break
    [[ "$out" == *"rules,"* ]] || { echo "error: $out"; exit 1; }
    n=${${out##*: }%%/*}; tot=${${out#*/}%% *}; [ "$n" = "$tot" ] && break
  done
done
