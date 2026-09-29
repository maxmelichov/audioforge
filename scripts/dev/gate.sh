#!/bin/zsh
# Machine-wide job gate for heavy Python work on this laptop. Usage: scripts/dev/gate.sh <command...>
# Waits until fewer than MAX_JOBS busy project-python processes are running, swap is under SWAP_MB (default 14 GB; baseline apps use ~9),
# and the internal disk has at least MIN_DISK_GB free; then runs the command with 2 threads.
# Every benchmark / training / eval / server launch by an agent MUST go through this wrapper.
set -u
MAX_JOBS=${GATE_MAX_JOBS:-3}; SWAP_MB=${GATE_SWAP_MB:-14000}; MIN_DISK_GB=${GATE_MIN_DISK_GB:-10}; WAIT_MAX=${GATE_WAIT_MAX:-2700}
waited=0
while true; do
  jobs=$(ps -axo %cpu,args | awk '/\.venv\/bin\/python/ && !/gate\.sh/ && $1>25 {n++} END {print n+0}')
  swap=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
  disk=$(df -g / | awk 'NR==2{print $4}')
  if [ "$jobs" -lt "$MAX_JOBS" ] && [ "$swap" -lt "$SWAP_MB" ] && [ "$disk" -ge "$MIN_DISK_GB" ]; then break; fi
  if [ "$waited" -ge "$WAIT_MAX" ]; then echo "gate: gave up after ${WAIT_MAX}s (jobs=$jobs swap=${swap}M disk=${disk}G)" >&2; exit 75; fi
  [ $((waited % 300)) -eq 0 ] && echo "gate: waiting (jobs=$jobs/$MAX_JOBS swap=${swap}M disk=${disk}G)" >&2
  sleep 20; waited=$((waited+20))
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} TOKENIZERS_PARALLELISM=false
exec nice -n 5 "$@"
