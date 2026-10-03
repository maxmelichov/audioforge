#!/bin/zsh
# After x115: every 115M classifier's p on the held-out scopes (turn_clean.py p115), gated MPS calls < 10 min each.
cd ${0:A:h:h:h}
while pgrep -f turn_clean_x115.sh >/dev/null; do sleep 20; done
for i in $(seq 1 14); do
  GATE_MAX_JOBS=1 GATE_SWAP_MB=17000 GATE_WAIT_MAX=10800 PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/turn_clean.py p115 --budget 200 || exit 1
  n=$(ls /Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_clean/p115/ | grep -c '^c\|^a_')
  echo "== p115 files $n"; [ "$n" -ge 12 ] && break
done
