#!/bin/bash
cd /Users/maxm/nvidia-audio-models
export PYTHONPATH=. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
O=/Volumes/ExternalSSD/nvidia-audio-models/demo_out
scripts/dev/gate.sh /Volumes/afdev/venvs/tts/bin/python demo/tts.py synth --script demo/script_v4.md --out narration_v4_Ryan --speaker Ryan --instruct lively --tempo 1.0 --budget 480 > $O/tts_v4_Ryan_final.log 2>&1; echo "tts rc=$?"
cp $O/narration_v4_Ryan.json $O/narration_v4.json; cp $O/narration_v4_Ryan.wav $O/narration_v4.wav
for size in 1920x1080 1080x1080; do
  name=showcase_v4.mp4; [ "$size" = 1080x1080 ] && name=showcase_v4_square.mp4
  for i in 0 1 2; do scripts/dev/gate.sh .venv/bin/python demo/render_v4.py --size $size --out $O/$name --seg $i --nseg 3 2>&1 | grep -E "Error|Trace"; done
  scripts/dev/gate.sh .venv/bin/python demo/render_v4.py --size $size --out $O/$name --mux --nseg 3 2>&1 | grep -E "wrote|Error|Trace"
done
for f in showcase_v4.mp4 showcase_v4_square.mp4; do printf "%-24s " $f; ffprobe -v error -show_entries format=duration:stream=codec_name,width,height -of csv=p=0 $O/$f | tr '\n' ' '; ls -la $O/$f | awk '{printf "%.1f MB\n", $5/1e6}'; done
