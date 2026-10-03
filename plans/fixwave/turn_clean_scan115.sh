#!/bin/zsh
# The 115M held-out scans in three niced single-thread CPU chains; each tag starts once its p115 file exists.
cd ${0:A:h:h:h}
P=/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_clean/p115
chain() {
  for tf in "$@"; do
    tag=${tf%%:*}
    [ "$tag" = head ] || while [ ! -f $P/$tag.npz ]; do sleep 20; done
    plans/fixwave/turn_clean_scan.sh 115m $tf
  done
}
chain head:conv c1:conv c1:asst a_notext:conv a_notext:asst a_nocompl:conv a_nocompl:asst c4:conv c4:asst a_big:conv a_big:asst c5s1:asst &
chain c2:conv c2:asst a_nost:conv a_nost:asst c3:conv c3:asst a_pool:conv a_pool:asst c5:conv c5:asst c23:asst &
chain c5s1:conv c23:conv &
wait
