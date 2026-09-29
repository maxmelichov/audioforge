#!/usr/bin/env bash
# MPS smoke test of scripts/research/single_model_distill on ~1 h of audio (research/SINGLE_MODEL.md part B): every stage
# once, and the WER gate forced to fire twice (--lr 5e-2 on the unfrozen blocks) -> KILLED, last passing state restored.
# Laptop only, through the machine gate, when no other MPS job runs. Work dir on the SSD.
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=${W:-/Volumes/ExternalSSD/nvidia-audio-models/scratch/single_distill_smoke}
P=scripts/research/single_model_distill/single_distill.py
G="scripts/dev/gate.sh env PYTHONPATH=. .venv/bin/python"
TEACHER=${TEACHER:-/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_speech_streaming_en_0.6b.afm}
mkdir -p "$W/eval"
# reuse the laptop's already-built 200-utterance eval sets when present (same seeds as evalsets builds them)
for f in ami icsi libri; do
  [ -f "$W/eval/$f.npz" ] || { [ -f /Volumes/ExternalSSD/nvidia-audio-models/scratch/distill_smoke/eval/$f.npz ] && \
    cp /Volumes/ExternalSSD/nvidia-audio-models/scratch/distill_smoke/eval/${f}.npz /Volumes/ExternalSSD/nvidia-audio-models/scratch/distill_smoke/eval/${f}_refs.json "$W/eval/"; } || true
done
T0=$(date +%s)
step() { echo "=== $1 ($(( $(date +%s) - T0 )) s)"; }
step manifest;  $G $P manifest --work "$W" --n-ami 2 --n-icsi 2 --n-libri 300 --max-hours 1 --threads 2
step evalsets;  $G $P evalsets --work "$W" --threads 2
step teacher;   $G $P teacher --work "$W" --teacher "$TEACHER" --device mps --batch 8 --threads 2 --budget 300
COMMON="--work $W --device mps --threads 2 --batch 4 --gate-n 20 --log-every 5 --ckpt-every 10"
step "train s1 (heads only)"; $G $P train $COMMON --tag s1 --steps 30 --warmup 5 --gate-every 15
step "train s1 resume (no-op: done)"; $G $P train $COMMON --tag s1 --steps 30 --warmup 5 --gate-every 15
step "train s3 (unfreeze 5 + 0.6B KD)"; $G $P train $COMMON --tag s3 --init "$W/s1/good.pt" --unfreeze 5 --asr-kd \
  --steps 20 --warmup 5 --gate-every 10 --p-asr 0.6
step "gate test (lr 5e-2: must fire twice and kill)"; $G $P train $COMMON --tag gatetest --init "$W/s1/good.pt" \
  --unfreeze 5 --p-asr 1.0 --p-tsvad 0 --p-turn 0 --lr 5e-2 --warmup 1 --steps 40 --gate-every 5 --gate-libri 0.0 --gate-ami 0.0 || true
step eval;      $G $P eval --work "$W" --tag s1 --device mps --threads 2 --n-eval 32 --n-wer 20 \
  --out "$W/single_distill_smoke.json"
step done
