#!/bin/zsh
# research/IMPROVE_115M.md A.2 / A.3: the whole TS-VAD turn-bench chain, resumable stage by stage (re-run this script; finished
# stages return in seconds). One heavy job at a time, every launch through scripts/dev/gate.sh, <= 540 s per call; all
# intermediate files under $AUDIOFORGE_SCRATCH (default: the external SSD). Stages up to report_tsvad_only were completed
# on 2026-09-28; the Sortformer-track rebuild (tracks_ami onwards, ~5 h at 7 windows/min) gives the paired CIs vs the
# Sortformer-column systems and was stopped at 62 / 974 AMI windows.
set -u; setopt shwordsplit
R=/Users/maxm/nvidia-audio-models; SCR=/Volumes/ExternalSSD/nvidia-audio-models/scratch
S=${AUDIOFORGE_SCRATCH:-/Volumes/ExternalSSD/nvidia-audio-models/scratch}/logs; mkdir -p $S
PY=$R/.venv/bin/python; G=$R/scripts/dev/gate.sh; RS=$R/scripts/research
cd $R
B="--budget 540"
run_stage() {  # name max_repeats cmd... ; repeat until a call finishes in < 150 s (nothing left) or max reached
  local name=$1 max=$2; shift 2; local i=1
  while [ $i -le $max ]; do
    local t0=$(date +%s)
    echo "=== $name #$i $(date +%H:%M:%S)" >> $S/chain.log
    $G "$@" >> $S/chain_$name.log 2>&1; local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "    $name #$i rc=$rc ${dt}s" >> $S/chain.log
    [ $rc -ne 0 ] && [ $rc -ne 75 ] && { echo "    $name failed rc=$rc" >> $S/chain.log; return $rc; }
    [ $dt -lt 150 ] && return 0
    i=$((i+1))
  done
}
run_stage winfeats 6 $PY $RS/tsvad.py winfeats --corpora ami,icsi $B
run_stage vprints_ami 2 $PY $RS/tsvad.py vprints --corpora ami
run_stage vprints_icsi 2 $PY $RS/tsvad.py vprints --corpora icsi
run_stage bind_tsvad 6 $PY $RS/tsvad.py bind --corpora ami,icsi --bindings tsvad_spk $B
run_stage scores_tsvad 8 $PY $RS/tsvad.py scores --corpora ami,icsi --bindings tsvad_spk --device cpu $B
run_stage vad_ami 4 $PY $RS/bench_turn_baselines.py --stage vad --work $SCR/baselines_turn/work $B
run_stage vad_icsi 4 $PY $RS/bench_turn_icsi.py --stage vad $B
run_stage report_tsvad_only 1 $PY $RS/tsvad.py report --corpora ami,icsi --bindings tsvad_spk --out-tag tsvad_only
cp $SCR/tsvad/report.json $S/report_tsvad_only.json 2>/dev/null
run_stage fleurs30 4 $PY $RS/hybrid_asr.py run --set fleurs --n-lang 30 $B
run_stage fleurs2s30 4 $PY $RS/hybrid_asr.py run --set fleurs2s --n-lang 30 $B
run_stage tracks_ami 40 $PY $RS/eval_stage1.py --bench v2 --v2-stage tracks --v2-work $SCR/trail6/work --v2-budget 540 --ckpt runs/stage1_turn_v3_trail6.afm
run_stage tracks_icsi 40 $PY $RS/bench_turn_icsi.py --stage tracks $B
run_stage scores_ami_ref 6 $PY $RS/eval_stage1.py --bench v2 --v2-stage scores --v2-work $SCR/trail6/work --v2-budget 540 --ckpt runs/stage1_turn_v3_trail6.afm --v2-tag trail6 --v2-bindings oracle,causal_dominant
run_stage scores_icsi_ref 6 $PY $RS/bench_turn_icsi.py --stage scores $B
run_stage tnemb_ami 8 $PY $RS/eval_stage1.py --bench v2 --v2-stage embed --v2-embedder titanet --v2-work $SCR/primary/work --v2-tracks-dir $SCR/trail6/work/tracks --v2-budget 540 --ckpt runs/stage1_turn_v3_trail6.afm
run_stage tnemb_icsi 8 $PY $RS/tsvad.py tnemb $B
run_stage bind_vp 6 $PY $RS/tsvad.py bind --corpora ami,icsi --bindings vp_spk,vp_titanet,tsvad_spk $B
run_stage scores_vp 8 $PY $RS/tsvad.py scores --corpora ami,icsi --bindings tsvad_spk,vp_spk,vp_titanet --device cpu $B
run_stage report_full 1 $PY $RS/tsvad.py report --corpora ami,icsi --bindings tsvad_spk,vp_spk,vp_titanet --out-tag full
echo "CHAIN DONE $(date +%H:%M:%S)" >> $S/chain.log
