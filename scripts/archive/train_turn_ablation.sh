#!/usr/bin/env bash
# Turn-taking ablation v3 (research/archive/TURN_ABLATION.md), strictly sequential, load-guarded (waits for load<8 before
# each run). The recipe now carries the v3 fixes (diar weight 0.5 / from_layers all / prefix_prob 0.5, text
# delay/noise/decoded, conditioning on diar output + noisy oracle). Only the two speaker arms are retrained; the
# acoustic baseline has no speaker or text path, so runs/turn_acoustic.afm (v2) is reused as is (not retrained).
# The eval benchmarks every speaker arm with both activity sources (oracle and own causal diarization).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 OMP_NUM_THREADS=2
wait_quiet() { while :; do l=$(sysctl -n vm.loadavg | awk '{print int($2)}'); [ "$l" -lt 8 ] && break; sleep 60; done; }
run() { name=$1; shift; wait_quiet; echo "=== $name $(date +%H:%M)"; nice -n 5 .venv/bin/python -m audioforge.cli train research/recipes/speaker_aware_turn.yaml trainer.device=mps "$@" -o runs/$name.afm > runs/$name.log 2>&1; grep -E "^\[|^final|Error" runs/$name.log | cut -c1-200; }
[ -f runs/turn_acoustic.afm ] || { echo "runs/turn_acoustic.afm missing (acoustic baseline is reused, not retrained)"; exit 1; }
# ablation overrides (the other arms, if ever needed):
#   acoustic  heads.turn.mode=none   heads.turn.condition_on_speaker=false heads.turn.use_text=false
#   text      heads.turn.mode=none   heads.turn.condition_on_speaker=false heads.turn.use_text=true
run turn_speaker_v3       heads.turn.mode=kernel heads.turn.condition_on_speaker=true heads.turn.use_text=false
run turn_speaker_text_v3  heads.turn.mode=kernel heads.turn.condition_on_speaker=true heads.turn.use_text=true
wait_quiet; echo "=== eval $(date +%H:%M)"
PYTHONPATH=. nice -n 5 .venv/bin/python -m audioforge.conversation runs/turn_acoustic.afm runs/turn_speaker_v3.afm runs/turn_speaker_text_v3.afm --n 1000 --out runs/archive/turn_ablation_v3.json > runs/archive/turn_ablation_v3.eval.log 2>&1
grep -vE "Warn|warn" runs/archive/turn_ablation_v3.eval.log | tail -30
for n in turn_speaker_v3 turn_speaker_text_v3; do echo "$n: $(grep '^final' runs/$n.log | grep -oE '"(der_diar|eot_turn_diar_act_miss|turn_align_greedy_frac|turn_decoded_frac)": [0-9.e-]+' | tr '\n' ' ')"; done
