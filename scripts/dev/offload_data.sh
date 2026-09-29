#!/bin/zsh
# Move large, low-frequency data dirs to the external SSD and leave symlinks behind. Idempotent; resumable.
set -u
SRC=/Users/maxm/nvidia-audio-models/data
DST=/Volumes/ExternalSSD/nvidia-audio-models/data
LOG=/Volumes/ExternalSSD/nvidia-audio-models/logs/offload.log
for d in librispeech oto141 nemo behavior_sd dailytalk lid smartturn; do
  [ -L "$SRC/$d" ] && { echo "$d already symlink" >>$LOG; continue; }
  [ -d /Volumes/ExternalSSD/nvidia-audio-models ] || { echo "SSD gone, abort" >>$LOG; exit 1; }
  echo "== $d start $(date)" >>$LOG
  nice -n 15 rsync -rtl --modify-window=2 --no-perms --no-owner --no-group --inplace "$SRC/$d/" "$DST/$d/" >>$LOG 2>&1 || { echo "$d rsync FAILED" >>$LOG; continue; }
  left=$(rsync -rtln --modify-window=2 --no-perms --no-owner --no-group --out-format='%n' "$SRC/$d/" "$DST/$d/" | grep -v '/$' | wc -l | tr -d ' ')
  if [ "$left" != "0" ]; then echo "$d verify FAILED ($left files differ), keeping original" >>$LOG; continue; fi
  if lsof +D "$SRC/$d" 2>/dev/null | grep -q python; then echo "$d in use, skipping swap" >>$LOG; continue; fi
  mv "$SRC/$d" "$SRC/.$d.__old" && ln -s "$DST/$d" "$SRC/$d" && rm -rf "$SRC/.$d.__old"
  echo "== $d swapped $(date) free=$(df -g / | awk 'NR==2{print $4}')G" >>$LOG
done
echo "ALL DONE $(date)" >>$LOG
