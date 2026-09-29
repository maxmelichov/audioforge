#!/usr/bin/env bash
# Sequential GPU chain: v3 turn ablation -> stage 1 (frozen NVIDIA encoder + our heads). One training at a time.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 OMP_NUM_THREADS=2
bash scripts/research/train_turn_ablation.sh
echo "=== stage1 $(date +%H:%M)"
while [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -ge 8 ]; do sleep 60; done
nice -n 5 .venv/bin/python -m audioforge.cli train research/recipes/stage1_heads_pretrained.yaml -o runs/stage1_heads_pretrained.afm > runs/stage1_heads_pretrained.log 2>&1
grep -E "^\[|init.from|trainer\]|wer_gate|^final|Error|Traceback" runs/stage1_heads_pretrained.log | cut -c1-220 | tail -20
