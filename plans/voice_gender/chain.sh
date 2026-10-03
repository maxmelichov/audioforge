#!/bin/zsh
# voice_gender: the heavy steps one after another, each through the machine gate (research/VOICE_GENDER.md)
set -e
cd ${0:A:h:h:h}
while pgrep -f "voice_gender.py feats" >/dev/null; do sleep 10; done
G=scripts/dev/gate.sh; P=".venv/bin/python scripts/research/voice_gender.py"
for c in 115m 0p6b; do $G ${=P} sweep --core $c --sizes 16,32,64,128,256 --budget 60 --evals 5; done
for c in 115m 0p6b; do $G ${=P} test --core $c; done
for c in 115m 0p6b; do $G ${=P} latency --core $c --device mps; done
