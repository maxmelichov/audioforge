#!/bin/zsh
# Evaluation fix wave, second batch (after queue_eval.sh): standard DER of the diarizer tracks, parameter counts of the
# current builds. Same rules as queue_eval.sh.
cd /Users/maxm/nvidia-audio-models
export PYTHONPATH=. GATE_MAX_JOBS=1 GATE_WAIT_MAX=7200 GATE_SWAP_MB=17000
G="scripts/dev/gate.sh .venv/bin/python"
L=runs/fixwave/queue_eval.log
idle() { while [ $(ps -axo %cpu,args | awk '/\.venv\/bin\/python .*(scripts\/research|sweep)/ && !/gate\.sh/ && $1>25 {n++} END {print n+0}') -gt 0 ]; do sleep 2; done; }  # no busy research job
until grep -q "QUEUE DONE" $L; do sleep 30; done
for st in "stdder:scripts/research/final_scoring.py stdder" "params:scripts/research/final_compare.py params"; do
  idle; echo "== $(date +%T) ${st%%:*}" >>$L
  eval "$G ${st#*:}" 2>&1 | tee -a runs/fixwave/${st%%:*}.log >>$L
done
echo "== $(date +%T) QUEUE2 DONE" >>$L
