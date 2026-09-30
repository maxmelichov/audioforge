#!/bin/bash
# One-command rebuild of the v5 LinkedIn film (16:9 and 1:1). Machine rules: model-loading steps go through
# scripts/dev/gate.sh; the TTS on MPS runs alone; every render call stops after --budget s (exit 3 = run again);
# ffmpeg -threads 2; frames and renders on the SSD ($DEMO_OUT).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=. DEMO_OUT=${DEMO_OUT:-/Volumes/ExternalSSD/nvidia-audio-models/demo_out}
V=/Volumes/afdev/venvs/video/bin/python
TTS=/Volumes/afdev/venvs/tts/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5

# 1. numbers and recorded data (no model is loaded)
.venv/bin/python demo/archive/extract_numbers.py
scripts/dev/gate.sh env HF_HUB_OFFLINE=1 .venv/bin/python demo/archive/v5/examples.py
.venv/bin/python demo/archive/v5/export.py
python3 demo/v5/shots/export_b.py

# 2. narration (Ryan, lively, 1.0x) and the demo agent's line (Aiden); both cached per line
for k in 1 2 3; do
  scripts/dev/gate.sh $TTS demo/archive/tts.py synth --script demo/archive/v5/script.md --out narration_v5 --speaker Ryan --instruct lively --tempo 1.0 --max-spc 0.1 --budget 480
  [ $? -ne 3 ] && break
done
scripts/dev/gate.sh $TTS demo/archive/tts.py synth --script demo/archive/v5/agent_voice.md --out agent_voice_v5 --speaker Aiden \
  --instruct "Neutral, friendly voice assistant on a phone call, even pace, clear." --tempo 1.0 --max-spc 0.2 --budget 300
HF_HUB_OFFLINE=1 scripts/dev/gate.sh .venv/bin/python demo/archive/v5/check_voice.py $DEMO_OUT/narration_v5.json

# 3. frames (resumable), mux, QA, per size
for size in 1920x1080 1080x1080; do
  for i in 1 2 3 4 5; do
    $V demo/archive/v5/film.py --cut full --size $size --frames --budget 540 && break
  done
  $V demo/archive/v5/film.py --cut full --size $size --mux
  $V demo/archive/v5/qa.py --cut full --size $size
done
