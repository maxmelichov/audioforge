#!/bin/zsh
# 115M features on the held-out scopes without a 115M cache (turn_clean.py x115), one gated call per batch of work.
cd ${0:A:h:h:h}
for sp in quiet dev devq odev odevq atdev; do
  for i in 1 2 3 4; do
    GATE_MAX_JOBS=${GATE_MAX_JOBS:-1} GATE_SWAP_MB=${GATE_SWAP_MB:-14000} PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/turn_clean.py x115 --split $sp --budget 480; rc=$?; [ $rc -eq 0 ] || [ $rc -eq 75 ] || exit 1
    left=$(ls /Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_clean/x115/$sp 2>/dev/null | grep -vc tmp)
    echo "== $sp files $left"
  done
done
