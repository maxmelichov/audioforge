#!/bin/zsh
# Evaluation fix wave, remaining jobs (replaces the tail of queue_eval.sh / queue_eval2.sh, which starved behind the
# other agents' back-to-back jobs). CPU jobs: GATE_MAX_JOBS=2 (may run next to one other job, niced, 2 threads);
# MPS jobs: GATE_MAX_JOBS=1 (one MPS job at a time across agents). Log: runs/fixwave/queue_eval.log.
cd /Users/maxm/nvidia-audio-models
export PYTHONPATH=. GATE_WAIT_MAX=7200 GATE_SWAP_MB=17000
L=runs/fixwave/queue_eval.log
run() {  # run <maxjobs> <name> <cmd...>
  local mj=$1 name=$2; shift 2
  echo "== $(date +%T) $name (gate max $mj)" >>$L
  GATE_MAX_JOBS=$mj scripts/dev/gate.sh .venv/bin/python "$@" 2>&1 | tee -a runs/fixwave/$name.log >>$L
}
run 2 lat_whisper_small_b5 scripts/research/final_latency.py run --system whisper_small_b5
run 2 stdder scripts/research/final_scoring.py stdder
run 2 params scripts/research/final_compare.py params
run 1 lat_whisper_small_mps scripts/research/final_latency.py run --system whisper_small_mps
for c in 115m 0p6b; do
  for i in {1..8}; do
    run 1 noprint_$c scripts/research/final_compare.py eotdump --sys $c --which calls --noprint --device mps --budget 540
    tail -3 runs/fixwave/noprint_$c.log | grep -q STAGE_COMPLETE && break
  done
done
echo "== $(date +%T) QUEUE3 DONE" >>$L
