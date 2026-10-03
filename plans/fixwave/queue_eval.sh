#!/bin/zsh
# Evaluation fix wave: the heavy jobs, one at a time through the gate (GATE_MAX_JOBS=1: wait for any other busy job).
# Log: runs/fixwave/queue_eval.log. Each call < 10 min (--budget 540), repeated until the stage reports completion.
cd /Users/maxm/nvidia-audio-models
# swap on this Mac stays ~22 GB from earlier runs while ~50 % of RAM is free: the swap limit is 17 GB (coordinator note),
# and instead every call first waits until no other research / sweep python job is running (one heavy job at a time)
export PYTHONPATH=. GATE_MAX_JOBS=1 GATE_WAIT_MAX=7200 GATE_SWAP_MB=17000
G="scripts/dev/gate.sh .venv/bin/python"
L=runs/fixwave/queue_eval.log
idle() { while [ $(ps -axo %cpu,args | awk '/\.venv\/bin\/python .*(scripts\/research|sweep)/ && !/gate\.sh/ && $1>25 {n++} END {print n+0}') -gt 0 ]; do sleep 2; done; }  # no busy research job
step() {  # step <name> <cmd...>: rerun until a call prints STAGE_COMPLETE (max 12 calls)
  local name=$1; shift
  for i in {1..12}; do
    idle; echo "== $(date +%T) $name call $i" >>$L
    eval "$G $*" 2>&1 | tee -a runs/fixwave/$name.log >>$L
    tail -5 runs/fixwave/$name.log | grep -q STAGE_COMPLETE && return 0
  done
  echo "== $name: not complete after 12 calls" >>$L
}
once() { local name=$1; shift; idle; echo "== $(date +%T) $name" >>$L; eval "$G $*" 2>&1 | tee -a runs/fixwave/$name.log >>$L; }
step meta scripts/research/final_scoring.py meta
for s in ami_eval icsi_eval live ls_clean ls_other fleurs_en; do
  step asr_b5_$s scripts/research/final_compare.py asr --system whisper_small_b5 --set $s --budget 540
done
for s in parakeet_tdt_mps whisper_small_b5 whisper_small_mps; do
  [ -s /Volumes/ExternalSSD/nvidia-audio-models/scratch/finallat/$s.jsonl ] && grep -q "$s: 56 turns" runs/fixwave/lat_$s.log 2>/dev/null && continue
  once lat_$s scripts/research/final_latency.py run --system $s
done
step noprint_115m scripts/research/final_compare.py eotdump --sys 115m --which calls --noprint --device mps --budget 540
step noprint_0p6b scripts/research/final_compare.py eotdump --sys 0p6b --which calls --noprint --device mps --budget 540
echo "== $(date +%T) QUEUE DONE" >>$L
