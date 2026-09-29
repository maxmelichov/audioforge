#!/usr/bin/env bash
# Train every recipe at laptop scale and save models + logs under runs/.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for r in research/recipes/*.yaml; do
  n=$(basename "$r" .yaml)
  # recipes with a dedicated trainer or real-data downloads are not run here
  if grep -q "^# train_with:" "$r" || ! grep -q "synthetic" "$r"; then echo "=== $n (skipped: see recipe header)"; continue; fi
  echo "=== $n"
  $PY -m audioforge.cli train "$r" -o "runs/$n.afm" "$@" > "runs/$n.log" 2>&1
  grep -E "^\[|^final" "runs/$n.log"
done
