# Improving the served 115M front end without touching the encoder

2026-09-28. Three head-only additions to the served model (`runs/stage1_served.afm`: NVIDIA's frozen 115M
cache-aware streaming FastConformer, our VAD / speaker / turn heads, Sortformer v2 as the diarizer), each measured
against the shipped configuration on the same benches. Part A: a target-speaker VAD head as the turn head's
speaker-activity input (replacing the diarizer track). Part B: meeting-ASR decoder adaptation, head only (section B).
Part C-lite: the dual-lookahead second ASR pass (`research/HYBRID_ASR.md`, summarised in section C).

## Part A. TS-VAD head on block 4 as the turn head's primary-activity input

**Why.** The speaker-aware turn head (`research/EOT_BENCH_V2.md`) is fed the *primary speaker's* activity. In the
served product that activity is a Sortformer v2 column bound to the user (causal-dominant rule, or a voice print via
`--enroll`), and the label-free binding is where most of the gap to oracle enrollment is lost (61.9 % vs 28.7 % missed
turn ends at 6 s). A personal-VAD head that reads the encoder the ASR pass already runs, conditioned on a voice print,
would give the turn head the user's activity directly: no diarizer column to choose, no column-lag, and no diarizer at
all when the product only needs "is the user talking".

**Head** (`audioforge/heads/tsvad.py`, 260k parameters): reads block 4 of the served encoder (the relational speaker
head's tap, `research/SPK_HEAD.md`), per-frame projection to 128, FiLM from the 192-d enrollment embedding, a
frame-level cosine pre-net, a causal GRU, two sigmoid outputs [P(target), P(other)] so overlap is represented; a
learned null vector makes it a plain VAD without enrollment (PVAD 2.0's enrollment-less training). Streaming `step`
is bit-equal to the offline forward (`tests/test_tsvad.py`, 7 tests).

**Training** (`scripts/tsvad.py trainfeats / enroll / train`): cached block-4 features of 16 s crops (hop 12 s) of
the 12 AMI + 12 ICSI train meetings (the library's default train subsets), per (meeting, speaker) enrollment pools of
32 single-speaker clips of U(1, 5) s embedded by the served block-4 speaker head (and TitaNet-L for the `--emb titanet`
variant). Per example: target = a speaker active in the crop (p 0.8) or any enrollable speaker, enrollment = one of
that speaker's clips that does not overlap the crop (+-1 s), dropped with p 0.15 (then target = any speech). AdamW
2e-3 one-cycle, batch 64, 3000 steps, AMI/ICSI balanced, CPU 2 threads, ~3.3 min. The encoder never runs in the loop
and no served tensor changes.

**Selection** (held-out AMI EN2003a + ICSI Bsr001, 3000 steps unless noted; frame F1 on the target column):
default 0.772 (recall on overlap frames 0.61); hidden 192: 0.776; 6000 steps: 0.772; +embedding noise 0.05: 0.766;
no pre-net: 0.707; TitaNet-L prints: 0.689 (the head is trained on the served speaker head's embedding space; the
TitaNet space needs its own head). The default (128, pre-net, served-head prints) was kept and retrained on all 24
meetings -> `runs/tsvad_spk.pt`.

### A.1 Frame-level target activity on the eot-bench v2 windows

Every window of eot-bench v2 (AMI dev 974, ICSI held-out 1312, the 6 s-extended windows), every labelled speaker with a
voice print as the target. The **voice print** is the target speaker's single-speaker speech *elsewhere in the same
meeting* (segments overlapping the window excluded): 5 s (and 1.5 s), embedded by the served block-4 speaker head
(and by TitaNet-L for the Sortformer follower). The Sortformer comparison uses the *same* print to bind a column
(voice following with the served speaker head or TitaNet-L on the column embeddings, `enrollment.VoiceFollower`);
the oracle column is chosen from the labels (upper bound). Frame classes from the labels: single = only the target
speaks, overlap = target + someone else, silence = nobody.

**AMI dev (974 windows), target = primary speaker (column 0), 2947 targets.** F1 / miss / FA on all frames, then F1 on single-speaker and overlap frames (miss on overlap frames) and FA on silence.

| track | F1 all | miss | FA | F1 single | F1 overlap | miss overlap | FA silence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS-VAD head, 5 s print | 0.743 | 0.265 | 0.154 | 0.786 | 0.696 | 0.380 | 0.080 |
| TS-VAD head, 1.5 s print | 0.672 | 0.357 | 0.170 | 0.694 | 0.666 | 0.422 | 0.067 |
| TS-VAD head, no enrollment (plain VAD) | 0.610 | 0.129 | 0.622 | 0.583 | 0.778 | 0.112 | 0.187 |
| Sortformer column bound by the same 5 s print (speaker head) | 0.629 | 0.321 | 0.302 | 0.599 | 0.727 | 0.295 | 0.036 |
| Sortformer column bound by the same 5 s print (TitaNet-L) | 0.657 | 0.321 | 0.245 | 0.634 | 0.733 | 0.301 | 0.033 |
| Sortformer column, 1.5 s print (speaker head) | 0.622 | 0.329 | 0.306 | 0.592 | 0.721 | 0.303 | 0.036 |
| Sortformer oracle column (label-chosen, upper bound) | 0.799 | 0.166 | 0.159 | 0.796 | 0.828 | 0.171 | 0.030 |

**AMI dev (974 windows), target = every labelled speaker, 2947 targets.** F1 / miss / FA on all frames, then F1 on single-speaker and overlap frames (miss on overlap frames) and FA on silence.

| track | F1 all | miss | FA | F1 single | F1 overlap | miss overlap | FA silence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS-VAD head, 5 s print | 0.694 | 0.295 | 0.156 | 0.726 | 0.674 | 0.390 | 0.063 |
| TS-VAD head, 1.5 s print | 0.636 | 0.365 | 0.174 | 0.646 | 0.649 | 0.426 | 0.057 |
| TS-VAD head, no enrollment (plain VAD) | 0.533 | 0.127 | 0.670 | 0.477 | 0.743 | 0.104 | 0.196 |
| Sortformer column bound by the same 5 s print (speaker head) | 0.545 | 0.351 | 0.351 | 0.489 | 0.678 | 0.317 | 0.037 |
| Sortformer column bound by the same 5 s print (TitaNet-L) | 0.570 | 0.350 | 0.302 | 0.519 | 0.688 | 0.317 | 0.034 |
| Sortformer column, 1.5 s print (speaker head) | 0.540 | 0.359 | 0.352 | 0.482 | 0.677 | 0.320 | 0.036 |
| Sortformer oracle column (label-chosen, upper bound) | 0.761 | 0.173 | 0.166 | 0.747 | 0.803 | 0.179 | 0.029 |

**ICSI held-out (1312 windows), target = primary speaker (column 0), 3669 targets.** F1 / miss / FA on all frames, then F1 on single-speaker and overlap frames (miss on overlap frames) and FA on silence.

| track | F1 all | miss | FA | F1 single | F1 overlap | miss overlap | FA silence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS-VAD head, 5 s print | 0.882 | 0.142 | 0.058 | 0.917 | 0.787 | 0.320 | 0.061 |
| TS-VAD head, 1.5 s print | 0.826 | 0.226 | 0.065 | 0.855 | 0.761 | 0.343 | 0.057 |
| TS-VAD head, no enrollment (plain VAD) | 0.652 | 0.043 | 0.640 | 0.651 | 0.835 | 0.013 | 0.161 |
| Sortformer column bound by the same 5 s print (speaker head) | 0.690 | 0.137 | 0.418 | 0.741 | 0.838 | 0.128 | 0.362 |
| Sortformer column bound by the same 5 s print (TitaNet-L) | 0.685 | 0.160 | 0.400 | 0.737 | 0.829 | 0.149 | 0.352 |
| Sortformer column, 1.5 s print (speaker head) | 0.682 | 0.143 | 0.430 | 0.732 | 0.833 | 0.130 | 0.364 |
| Sortformer oracle column (label-chosen, upper bound) | 0.794 | 0.033 | 0.306 | 0.859 | 0.892 | 0.045 | 0.329 |

**ICSI held-out (1312 windows), target = every labelled speaker, 3669 targets.** F1 / miss / FA on all frames, then F1 on single-speaker and overlap frames (miss on overlap frames) and FA on silence.

| track | F1 all | miss | FA | F1 single | F1 overlap | miss overlap | FA silence |
|---|---:|---:|---:|---:|---:|---:|---:|
| TS-VAD head, 5 s print | 0.855 | 0.181 | 0.043 | 0.896 | 0.747 | 0.370 | 0.043 |
| TS-VAD head, 1.5 s print | 0.800 | 0.252 | 0.054 | 0.834 | 0.723 | 0.387 | 0.040 |
| TS-VAD head, no enrollment (plain VAD) | 0.540 | 0.046 | 0.695 | 0.520 | 0.789 | 0.013 | 0.166 |
| Sortformer column bound by the same 5 s print (speaker head) | 0.564 | 0.186 | 0.477 | 0.592 | 0.765 | 0.170 | 0.361 |
| Sortformer column bound by the same 5 s print (TitaNet-L) | 0.556 | 0.212 | 0.466 | 0.582 | 0.757 | 0.189 | 0.352 |
| Sortformer column, 1.5 s print (speaker head) | 0.556 | 0.194 | 0.483 | 0.581 | 0.760 | 0.172 | 0.361 |
| Sortformer oracle column (label-chosen, upper bound) | 0.737 | 0.050 | 0.277 | 0.790 | 0.853 | 0.077 | 0.284 |

**Reading.** With a 5 s print the TS-VAD track is the better primary-activity track on both corpora: F1 0.743 vs
0.629 / 0.657 (AMI, speaker-head / TitaNet following) and **0.882 vs 0.690 / 0.685 on held-out ICSI**, where it also
beats the *oracle* Sortformer column (0.794) because that column carries Sortformer's own false alarms on silence
(FA 0.33 on ICSI silence frames vs 0.06). The split says where each is strong: on **single-speaker** frames TS-VAD is
far ahead (0.786 vs 0.599, 0.917 vs 0.741); on **overlap** frames the diarizer column is better (0.727 vs 0.696 on AMI,
0.838 vs 0.787 on ICSI): the head misses 32-38 % of the target's overlapped frames against 13-30 % for the column.
Its silence FA (0.06-0.08) is above the Sortformer column's on AMI (0.03) and far below it on ICSI (0.36). A 1.5 s print
costs 5-7 F1 points; without a print the head is a plain VAD (FA 0.62-0.64 on non-target speech), as trained.

### A.1b Nemotron-3 arm, 2026-09-29

The A.1 frame evaluation with NVIDIA's newer diarizer, **Nemotron-3-Diarization** (`runs/nemo_nemotron3_diar.afm`),
in place of Sortformer v2, at the **served settings** (`audioforge.hub.diarizer_defaults` + `serve.Engine.make_diarizer`):
all 8 columns, max pooling of the 10 ms sub-frames, window mode with `--diar-left 1`, the served preset
`low_latency_032` (0.32 s input buffer). Tracks are built on the same eot-bench v2 extended windows by the
`eval_stage1.v2_tracks` protocol (window audio + (C + R + 1) frames of real right padding, cropped) and cached with
read-back checks under `scratch/tsvad/<corpus>/n3/` (tracks + TitaNet-L look-back column embeddings, 632 MB).
Binding is the Sortformer arm's: `vp_follow` on the same 5 s print (served speaker head's `ColumnEmbedder`, or
TitaNet-L via `v2_embed_titanet`; margin 0.1, hold 6); the oracle column is `enroll_column` on the labels. CIs:
1000 window resamples (seed 0), 95 % percentile; the difference is paired (same resampled windows) against the
TS-VAD 5 s row, which the new stage recomputes and matches the stored row exactly. Targets = those with a 5 s print
(AMI 974 primaries, ICSI 1286). Stages: `tsvad.py n3tracks / n3tnemb / n3check / framearm / framearmreport`;
rows in `runs/improve_115m.json["frame"][corpus][group]["nemotron3_*"]`, protocol in `["frame_nemotron3_protocol"]`.

**Target = primary speaker (column 0).** F1 [95 % CI], precision, miss, FA on all frames; F1 single / overlap and FA
on silence; paired F1 difference vs the TS-VAD head [CI]. The Sortformer rows are the stored A.1 values (1.04 s card
config; their tracks were lost with the scratch, so they carry no CIs).

**AMI dev (974 windows). Caveat: Nemotron-3's model card lists AMI train + dev in its training data, so its AMI dev
rows are not held out** (they favour Nemotron-3; the TS-VAD head and Sortformer v2 never saw AMI dev).

| track | F1 all [CI] | precision | miss | FA | F1 single | F1 overlap | FA silence | F1 - TS-VAD [CI] |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TS-VAD head, 5 s print | **0.743 [0.726, 0.759]** | 0.750 | 0.265 | 0.154 | 0.786 | 0.696 | 0.080 | - |
| Nemotron-3 column, 5 s print (speaker head) | 0.618 [0.599, 0.636] | 0.599 | 0.361 | 0.270 | 0.588 | 0.707 | 0.023 | -0.125 [-0.138, -0.111] |
| Nemotron-3 column, 5 s print (TitaNet-L) | 0.648 [0.630, 0.665] | 0.657 | 0.360 | 0.211 | 0.625 | 0.719 | 0.022 | -0.095 [-0.108, -0.081] |
| Nemotron-3 oracle column (upper bound) | 0.796 [0.783, 0.809] | 0.805 | 0.213 | 0.120 | 0.793 | 0.822 | 0.027 | +0.053 [+0.039, +0.068] |
| Sortformer v2 column, 5 s print (speaker head / TitaNet-L) | 0.629 / 0.657 | 0.586 / 0.636 | 0.321 / 0.321 | 0.302 / 0.245 | 0.599 / 0.634 | 0.727 / 0.733 | 0.036 / 0.033 | - |
| Sortformer v2 oracle column | 0.799 | 0.768 | 0.166 | 0.159 | 0.796 | 0.828 | 0.030 | - |

**ICSI held-out (1312 windows, 1286 primaries with a 5 s print).**

| track | F1 all [CI] | precision | miss | FA | F1 single | F1 overlap | FA silence | F1 - TS-VAD [CI] |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TS-VAD head, 5 s print | **0.882 [0.875, 0.889]** | 0.907 | 0.142 | 0.058 | 0.917 | 0.787 | 0.061 | - |
| Nemotron-3 column, 5 s print (speaker head) | 0.698 [0.684, 0.712] | 0.585 | 0.136 | 0.402 | 0.749 | 0.839 | 0.349 | -0.184 [-0.197, -0.172] |
| Nemotron-3 column, 5 s print (TitaNet-L) | 0.696 [0.683, 0.711] | 0.594 | 0.159 | 0.376 | 0.747 | 0.828 | 0.336 | -0.186 [-0.198, -0.173] |
| Nemotron-3 oracle column (upper bound) | 0.828 [0.817, 0.839] | 0.719 | 0.024 | 0.249 | 0.893 | 0.925 | 0.309 | -0.054 [-0.064, -0.044] |
| Sortformer v2 column, 5 s print (speaker head / TitaNet-L) | 0.690 / 0.685 | 0.575 / 0.579 | 0.137 / 0.160 | 0.418 / 0.400 | 0.741 / 0.737 | 0.838 / 0.829 | 0.362 / 0.352 | - |
| Sortformer v2 oracle column | 0.794 | 0.674 | 0.033 | 0.306 | 0.859 | 0.892 | 0.329 | - |

Every labelled speaker as the target (all_speakers, same file): AMI TS-VAD 0.694 vs Nemotron-3 0.533 / 0.564
(speaker head / TitaNet; paired -0.161 / -0.131), oracle 0.770; ICSI 0.855 vs 0.573 / 0.567 (-0.282 / -0.288),
oracle 0.780.

**Verdict.** The newer diarizer does not change the A.1 conclusion. Bound to the same 5 s print, the Nemotron-3
column tracks the user **no better than the Sortformer v2 column** (AMI 0.618 / 0.648 vs 0.629 / 0.657; ICSI 0.698 /
0.696 vs 0.690 / 0.685), and the TS-VAD head stays ahead by 9.5-12.5 F1 points on AMI and 18.4-18.6 on ICSI
(paired CIs well clear of zero). The AMI result holds even though AMI dev is in Nemotron-3's training data.
Nemotron-3's better diarization shows in the **oracle column** instead: 0.828 vs 0.794 on ICSI (miss 0.024), and
precision 0.805 vs 0.768 on AMI. On AMI the oracle column is above the TS-VAD head (+0.053), as Sortformer's was.
On ICSI it is still below (-0.054) because it carries the diarizer's false alarms on silence (FA 0.31 vs 0.06).
So the gap is in the **binding** (which column is the user, frame by frame), not in the column quality. The binding
loses 15-18 points on AMI and 13 on ICSI against the oracle column, about what it loses with Sortformer.
The 8 columns do not matter here: the windows have at most 4 labelled speakers, and Nemotron-3 activates a 5th
column in 3.1 % of ICSI windows and never on AMI. Latency is not like-for-like and favours Sortformer: Nemotron-3
runs at the served 0.32 s, the stored Sortformer rows at 1.04 s. The overlap split is unchanged: the diarizer column
is better on overlap frames (0.71-0.84 vs 0.70 / 0.79) and far worse on single-speaker frames. A diarizer-overlap
fallback for the TS-VAD track is still the open idea from A.1, whichever diarizer is used.

### A.2 The TS-VAD track as the turn head's input: eot-bench v2, 6 s horizon

`scripts/research/tsvad.py bind / scores / report`. Every window's primary gets its 5 s voice print (single-speaker speech
elsewhere in the meeting, as in A.1; ICSI: 2 % of primaries have none and get the enrollment-less head), the head's
[P(target), P(other)] is fed to the **served trail6 turn head** as the primary-activity track (columns 0/1, the other
two zero), the timeout arm runs on P(target), and the systems are scored exactly as `research/EOT_BENCH_V2.md` /
`BASELINES.md`: cross-fitted <= 5 % per-turn false cut (leave-meetings-out halves), 1000-resample CIs, open / taken
strata (`runs/improve_115m.json["turn_bench"]`). Emission for the TS-VAD rows is the 160 ms encoder chunk (no
Sortformer C + R buffer). The Sortformer-column rows are the ones already in the file from the 2026-09-27 run (AMI) and
`runs/baselines_turn_icsi.json` (ICSI, `crossfit_icsi`); their per-turn outcomes were lost with the scratch, so the
paired CIs against them are pending the track rebuild (A.3).

**AMI dev, 974 turns (open 236 / taken 738).** Missed turn ends % [CI] (P50 latency ms, per-turn FC %); open / taken miss %.

| speaker-activity input -> system | all | open | taken |
|---|---|---|---|
| Silero timeout (Pipecat default) | 72.7 [69.8, 75.4] (inf, 4.8) | 26.7 | 86.8 |
| timeout on the primary's *labels* (oracle activity) | 6.2 [4.7, 7.8] (1520, 4.3) | 0.9 | 7.9 |
| Sortformer causal-dominant column -> timeout / head / **hybrid (shipped)** / hybrid_dyn (shipped) | 74.8 / 66.5 / **61.9 [58.6, 65.3]** / 56.4 [53.4, 59.5] | 46.1 / 57.3 / 45.9 / 25.1 | 83.6 / 69.4 / 67.0 / 66.0 |
| Sortformer oracle column (label-chosen) -> timeout / head / hybrid / hybrid_dyn | 32.0 / 30.3 / 28.7 [26.2, 31.1] / 26.0 [23.3, 28.8] | 15.2 / 21.7 / 17.1 / 6.6 | 37.2 / 33.0 / 32.2 / 31.9 |
| **TS-VAD track (5 s print)** -> timeout | 64.3 [61.0, 67.4] (inf, 4.6) | 58.0 | 66.2 |
| **TS-VAD track** -> turn head | 40.0 [37.0, 43.4] (4240, 5.0) | 39.7 | 40.1 |
| **TS-VAD track** -> hybrid (head OR timeout) | **39.3 [36.2, 42.8]** (4160, 5.2) | 37.9 | 39.8 |
| **TS-VAD track** -> hybrid_dyn (head, Silero timeout shortened by the head) | **34.2 [31.1, 37.5]** (3200, 5.5) | **17.9** | 39.3 |
| TS-VAD track -> hybrid_dyn at the *shipped* (theta, offset) point (fitted for the Sortformer-fed head) | 41.0 [37.9, 44.3] (4080, 3.7) | 22.7 | 46.8 |

**ICSI held-out, 1312 turns (open 216 / taken 1096).**

| speaker-activity input -> system | all | open | taken |
|---|---|---|---|
| Silero timeout | 84.4 [82.3, 86.4] (inf, 8.4) | 37.4 | 92.7 |
| timeout on the primary's labels | 10.0 [8.3, 11.7] (1600, 5.1) | 2.1 | 11.5 |
| Sortformer causal-dominant column -> timeout / head / **hybrid (shipped)** | 85.3 / 69.1 / **68.5 [65.8, 71.2]** | 60.9 / 65.2 / 55.2 | 90.1 / 69.8 / 71.1 |
| head + Silero timeout (BASELINES_ICSI candidate) | 65.2 [62.4, 68.0] | 36.3 | 70.6 |
| **TS-VAD track** -> timeout | 52.5 [49.8, 55.3] (inf, 5.3) | 54.2 | 52.2 |
| **TS-VAD track** -> turn head | 20.5 [18.4, 22.9] (1920, 5.9) | 21.5 | 20.3 |
| **TS-VAD track** -> hybrid | **20.4 [18.2, 22.7]** (1920, 6.0) | 21.5 | 20.2 |
| **TS-VAD track** -> hybrid_dyn | **18.7 [16.6, 21.0]** (1920, 6.2) | **13.2** | 19.7 |
| TS-VAD track -> hybrid_dyn at the shipped point | 38.4 [35.8, 41.2] (3760, 1.2) | 29.7 | 40.1 |

**Reading.** Feeding the turn head the TS-VAD track instead of the label-free Sortformer column cuts the missed turn
ends at 6 s from **61.9 % to 39.3 %** on AMI dev and from **68.5 % to 20.4 %** on held-out ICSI (non-overlapping CIs
in both), at the same <= 5 % per-turn false-cut budget (the ICSI cross-fitted points land at 5.9-6.2 % out of fold).
On ICSI the TS-VAD-fed head is already better than the *oracle-column* hybrid on AMI (28.7 %), consistent with A.1
(F1 0.88 vs 0.79 for the oracle column there); on AMI it closes two thirds of the gap between the shipped hybrid and
the oracle column (61.9 -> 39.3 vs 28.7). The head's P50 latency on AMI stays long (4.2 s; the head was trained on
Sortformer tracks and is used untrained on this input) and the timeout arm alone is weak (P(target) drops late after
the turn); the `hybrid_dyn` fusion with Silero brings the open-floor misses to 17.9 % / 13.2 % and the median to
3.2 s / 1.9 s. The shipped `hybrid_dyn` operating point does not transfer (FC 3.7 % / 1.2 %: the theta was fitted on
a differently calibrated activity); a served TS-VAD path needs its own (theta, offset), which the cross-fit here
provides per fold. Costs: the head adds 0.26M parameters on block 4 of the ASR pass that already runs, no diarizer
call; the 5 s print is one speaker-head embedding at enrollment.

## Part B. Meeting-ASR decoder adaptation, head only (`research/recipes/asr_meeting_decoder_adapt.yaml`)

**What.** Only the RNNT prediction net + joint of the served model train (`init.train_only`, 5.33M of 119.65M
parameters); the NVIDIA encoder and the VAD / speaker / diarizer / turn / CTC heads are frozen, so every live decision
is unchanged and only the streaming transcript can move. Data: AMI train (12 meetings, 930 single-speaker segments
1-15 s) + ICSI train (12 meetings, 3249 segments) + a LibriSpeech train-clean-100 anchor (2800 utterances <= 12 s, weight
2 -> 5600 items of the 9779-item epoch) so read speech is not forgotten; AdamW 1e-4 one-cycle, warm-up 50, batch 6,
**500 steps** (380 s on MPS), `wer_gate` every 50 steps on LibriSpeech test-clean (first 50 utterances, stop if
> +0.3 points): it never fired (1.23 % before and after). Machine constraints kept: nothing was decoded anew (the
anchor reads the existing `train-clean-100.1-12s` cache on the SSD, 3.8 GB; the meeting audio caches were already
there), one gated process, 6.5 min. Verified after training: **748 of 759 tensors bit-identical** to
`runs/stage1_served.afm`, the 11 changed ones all under `heads.rnnt.pred` / `heads.rnnt.joint`.

**Result** (the section-2 protocol of `research/HYBRID_ASR.md`: `hybrid_asr.py lookahead --model ... --tag _adapt`,
[70,1], batch 4; `adapt_report`; `runs/hybrid_asr.json["adapt"]["_adapt"]`). WER % `normalize_text` (Whisper
`EnglishTextNormalizer` in brackets), paired 1000-resample deltas.

| set (n = 200) | served [70,1] | adapted decoder | adapted - served |
|---|---:|---:|---|
| AMI dev segments | 24.43 [22.04, 27.22] (20.63) | **22.50 [20.13, 25.25]** (19.29) | **-1.93 [-2.77, -1.24]** (-1.34 [-2.16, -0.63]) |
| ICSI dev segments | 27.29 [24.75, 30.24] (26.21) | 27.17 [24.55, 30.25] (26.48) | -0.12 [-0.78, +0.56] (+0.28 [-0.43, +0.97]) |
| LibriSpeech test-clean | 2.29 [1.78, 2.87] (2.27) | 2.33 [1.81, 2.94] (2.20) | +0.04 [-0.09, +0.18] (-0.06 [-0.22, +0.07]) |

The trainer's own AMI dev validation (64 segments) went 23.8 % (100 steps) -> 22.2 % (500); the frozen CTC head reads
25.9 % before and after, as it must.

**Reading.** A 6-minute head-only pass buys **-1.9 WER points on AMI** with the clean-speech WER held (+0.04, CI
through zero) and nothing else in the model touched: about the same size as the dual-lookahead gain (Part C, -1.4)
and additive with it in principle, but nowhere near the offline TDT v3 final (-14.7). ICSI does not move although
ICSI train is half the meeting data: the ICSI dev segments (Bmr021 / Bns001) share the room and style of the training
meetings, so the gain on AMI is probably vocabulary / language-model adaptation of the prediction net to the AMI
scenario meetings (shared topic and names across the AMI splits) rather than acoustic robustness, which the frozen
encoder cannot provide. Worth shipping as the served RNNT head only if the product's meetings look like AMI; the
honest general-purpose transcript improvement remains `--final-asr tdt_v3`. Not done: more steps / a second chunk
(the recipe's `init.from` chaining), a decoder-only LM-style adaptation on transcripts alone, or an unfrozen last
encoder block (which would change the live heads and was out of scope by design).

## Part C-lite. Dual lookahead: a second text-only pass of the same encoder at [70,13]

Measured in the issues pass (`research/HYBRID_ASR.md` section 3, `runs/hybrid_asr.json["english"]`): the served
model's WER at att_context [70,0] / **[70,1] (served)** / [70,13] (`serve --asr-lookahead 13`, 1.04 s lookahead) on
AMI-200 / ICSI-200 / LibriSpeech-200 is 25.7 / **24.4** / 23.0, 29.9 / **27.3** / 24.8 and 2.48 / **2.29** / 1.92 %;
paired [70,13] - [70,1] = **-1.41 [-2.48, -0.32]**, **-2.50 [-3.80, -1.32]**, **-0.37 [-0.64, -0.11]** WER points.
A real but small gain for ~10-20 % more encoder time; the offline TDT v3 final (9.7 % on AMI-200) is what moves the
meeting transcript. No live decision changes (`tests/test_hybrid_asr.py::test_lookahead_heads_bit_identical_and_text_partition`).
