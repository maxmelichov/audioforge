#!/bin/zsh
# speech_clean test pass (once): candidate builds, final_compare vad on the shipped and the new served model, scoring.
set -u
cd ${0:A:h:h:h}
export PYTHONPATH=. GATE_MAX_JOBS=1 GATE_SWAP_MB=17000 GATE_WAIT_MAX=14400
G="scripts/dev/gate.sh .venv/bin/python"
SC=/Volumes/ExternalSSD/nvidia-audio-models/scratch/speech_clean
FX=/Volumes/ExternalSSD/nvidia-audio-models/scratch/fixall
C6=/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_0p6b
step() { echo "== STEP $*"; "$@" || { echo "== FAIL $*"; exit 1; }; }
step zsh -c "$G scripts/research/speech_clean.py build --core 115m --tag clean_mix26"
step zsh -c "$G scripts/research/speech_clean.py build --core 0.6b --tag clean_mix"
A115=$(python3 -c "import json;print(json.load(open('runs/fixall.json'))['speech_clean']['115m']['candidate']['afm'])")
A06=$(python3 -c "import json;print(json.load(open('runs/fixall.json'))['speech_clean']['0p6b']['candidate']['afm'])")
for i in 1 2 3; do step zsh -c "FINAL_COMPARE_W=$SC/fc_115m_shipped FINAL_115M_AFM=$FX/stage1_served_v4.afm $G scripts/research/final_compare.py vad --system core_115m"; done
for i in 1 2 3; do step zsh -c "FINAL_COMPARE_W=$SC/fc_115m_new FINAL_115M_AFM=$A115 $G scripts/research/final_compare.py vad --system core_115m"; done
for i in 1 2 3; do step zsh -c "FINAL_COMPARE_W=$SC/fc_0p6b_shipped FINAL_0P6B_AFM=$C6/served_0p6b_v0.3.afm $G scripts/research/final_compare.py vad --system core_0p6b"; done
for i in 1 2 3; do step zsh -c "FINAL_COMPARE_W=$SC/fc_0p6b_new FINAL_0P6B_AFM=$A06 $G scripts/research/final_compare.py vad --system core_0p6b"; done
step zsh -c "$G scripts/research/speech_clean.py score --core 115m"
step zsh -c "$G scripts/research/speech_clean.py score --core 0.6b"
echo "== SPEECH TEST DONE"
