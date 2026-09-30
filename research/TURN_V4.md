# Turn head v4: a faster end-of-turn decision (2026-09-30)

**Verdict: not shipped.** No retrained turn head, with any rule we scored, beats today's calls p50 (956 ms) by 150 ms
while keeping false interruptions and misses no worse on both corpora.
- The best new head (h2) with its best rule under the goal gives calls 989 ms, 15.6 % FI, 7.3 % missed, and AMI
  1167 ms, 10.5 % FI, 33.5 % missed.
- So it trades 4.6 points fewer call interruptions and 159 ms faster AMI answers for 33 ms slower calls.
- `served_heads_v0.2.pt`, the rule and the defaults are unchanged.

Why the head cannot deliver the speed:
- **The served VAD head keeps saying "speech" for 240-320 ms (p50) after the real end of a call turn.** Silero agrees
  with the labels, so this is a VAD tail, not a labelling problem.
- **Mid-turn pauses are long.** 33 % of call turns have a pause of 400 ms or more.
- **So a silence wait alone cannot go below ~1 s at ≤ 20 % FI.** 65 of the 101 answered call ends go to the 640 ms
  fallback (1056 ms median). The 36 head-path answers take 663 ms.
- **What would change that:** a head that tells a turn end from a pause within 200-300 ms of silence. With these inputs
  (pass-2 acoustics, the streaming ASR's text state, the TS-VAD track), none of the heads trained here can do that.

Code: `scripts/research/turn_v4.py`. Numbers: `runs/turn_v4.json`. The eval labels: `runs/turn_v4_labels_eval.json`.
Scoring is `scripts/research/eot_latency.py` unchanged in its metric. It gained three environment switches
(`EOT_DUMP`, `EOT_LABELS`, `EOT_OUT`, plus `EOT_CLIP_FRAMES`) and two optional `sim_room` paths (dyn-wait, early),
all off by default. `check` still gives 0 mismatches on all 232 sessions.

## Final table

"Original" = the published reference ends. "Audible end" = the corrected ends (next section). Each cell: EOT total
p50 / p95 ms, false interruptions % / missed %.

| row | labels | calls | AMI |
|---|---|---|---|
| (a) today's head + shipped rule | original | 956 / 1919, 20.2 / 7.3 | 1326 / 3758, 10.5 / 33.5 |
| (a) today's head + shipped rule | audible end | 951 / 1916, 19.3 / 6.4 | 1326 / 3758, 10.5 / 33.5 |
| (b) new head h2 + shipped rule | original | 996 / 1933, 13.8 / 7.3 | 1327 / 3806, 9.0 / 37.0 |
| (b) new head h2 + shipped rule | audible end | 996 / 1933, 13.8 / 7.3 | 1327 / 3806, 9.0 / 37.0 |
| (c) h2 + its best rule under the goal¹ | original | 989 / 1919, 15.6 / 7.3 | 1167 / 3727, 10.5 / 33.5 |
| (c) h2 + its best rule under the goal | audible end | none meets the goal (calls missed stays 7.3 > 6.4) | |
| Pipecat smart-turn v3.2 + Silero (defaults) | original / audible end | 237 / 3217, 35.8 / 24.8 (both) | 385 / 3992, 27.5 / 44.5 (audible end: 27.0 / 44.5) |
| LiveKit EnglishModel + Silero (defaults) | original / audible end | 567 / 3127, 26.6 / 22.9 (audible end: 566 / 3124, 25.7 / 22.0) | 1890 / 4295, 12.5 / 67.5 (both) |

¹ The rule: VAD < 0.4 for ≥ 240 ms and p ≥ 0.95, OR 640 ms, others path at 640 ms. It does not cut the demo clip under
any of the six deliveries.

**The goal**: FI and missed no worse than row (a) under each label set: 20.2 / 7.3 and 10.5 / 33.5 (original); 19.3 /
6.4 and 10.5 / 33.5 (audible end). Each candidate was scored on 12 421 rules:
- the vad_head family (VAD 0.3-0.5, 80-480 ms, θ 0.3-0.999, fallback 320-800 ms, others path 640-960 ms);
- the dyn-wait family (the wait is clamp(cap − a·p, floor, cap) on our VAD, cap 480-960 ms, floor 80-240 ms, with or
  without a head path);
- the early family (p ≥ θ_e after 80-160 ms of VAD < 0.5 / 0.6 / 0.7 / 0.8 or no VAD condition, before the VAD
  silence starts).

The harness's own `sweep` (13 210 rules) on the h2 dump finds no rule under its print-fix goal either.
- **Dyn-wait:** its best rule under the goal is slower than the plain rules for every head. For h2 that is 1005 ms
  calls (20.2 / 6.4) with a 240-800 ms wait. It is on no Pareto front.
- **Early path:** its best is 996 ms (h2, h5). It lowers FI and never lowers the p50.

**"p ≥ 0.95 within 200 / 300 / 500 ms of true ends"** (original labels; audible-end labels change today's head by at
most 1 point):

| head | calls | AMI |
|---|---|---|
| today | 48.6 / 50.5 / 52.3 % | 19.5 / 22.5 / 25.0 % |
| h2 | 1.8 / 1.8 / 2.8 % | 5.0 / 5.5 / 7.5 % |
| h5 (low-LR fine-tune + today's objective) | 0.0 / 0.0 / 0.9 % | 2.0 / 3.0 / 4.5 % |

This table measures calibration more than speed.
- Today's head is high almost everywhere on a user channel. The median p is 0.76 on frames inside a turn, 0.98 in
  pauses and 0.94 at ends.
- The new heads rank ends above in-turn frames better. The frame AUC (ends vs in-turn frames) on calls is 0.66 for h2 /
  h4 / h5 against 0.59 for today's head. On AMI it is 0.70 against 0.65.
- They are also no longer sure at ends.
- The rule scan already covers any monotone recalibration (θ from 0.3 up). No threshold turns the better ranking into
  a faster rule at the same FI.

**Smart-turn's own test clips** (the 399 human_5_all clips in smart-turn-data-v3.2-test; `runs/smartturn_audit.json`
protocol; smart-turn v3.2 scores 97.0 %). "Complete" means p ≥ θ on the frame 160 ms after the clip's last VAD speech
frame (the rule's first chance):

| head | θ 0.5 | θ picked on the smart-turn train clips |
|---|---|---|
| today | 63.2 % | 61.9 % (θ 0.39) |
| h2 | 79.4 % | 80.2 % (θ 0.15) |
| h1 | 76.7 % | 84.0 % (θ 0.09) |
| h4 | 77.9 % | 83.2 % (θ 0.13) |

The new heads trained on the smart-turn train split, so they gain here, but they stay far below smart-turn.

**Demo clip:** the shipped defaults are unchanged, so `served_check` still applies (no cut under any of the six
deliveries). The (c) rule was checked on the h2 frames of all six deliveries and does not cut. The clip frames were
recomputed offline; the offline path is checked against the served session (next sections).

## 1. Labels: the audible end

**Definition:** the end of the last 80 ms frame with served VAD > 0.5, or of the last 32 ms Silero chunk > 0.5, that
overlaps the labelled turn [s, e]. It takes the later of the two and is clipped to e, so an end only moves earlier.
The original labels are kept. Stage `labels_eval` writes `runs/turn_v4_labels_eval.json`; the audit is in
`runs/turn_v4.json` "labels_eval".

| set | ends | moved | moved > 120 ms | largest move |
|---|---|---|---|---|
| calls (TurnBench + oto user channels) | 109 | 13 | 6 (5.5 %) | 746 ms |
| of which TurnBench | 56 | 12 | 6 (10.7 %) | 746 ms |
| of which oto | 53 | 1 | 0 | 66 ms |
| AMI eval | 200 | 5 | 1 (0.5 %) | 560 ms |
| training clips (oto / AMI / ICSI train; served VAD only) | 859 sampled | – | 0 / 0.5 / 0 % | – |

- **Why this is far below the audit's 27.5 %.** The audit measured against Pipecat's VAD, which is Silero at 0.7.
  With Silero alone at 0.7, 50 % of TurnBench ends are more than 120 ms after its last speech. At 0.5, plus our VAD,
  the soft tails count as speech.
- **The labels are not late.** Silero > 0.5 continues past the label end by 0 ms at p50 on calls (> 100 ms at 5 % of
  TurnBench and 0 % of oto ends).
- **Our VAD head is late.** The served VAD head stays > 0.5 for 320 ms (TurnBench) and 240 ms (oto) at p50 after the
  label end. That tail is the structural cost described at the top.
- **Under audible-end labels, today's rule barely moves:** 951 ms, 19.3 / 6.4 % on calls, and AMI unchanged. The ~1 s
  is not a labelling artefact.
- **Priority check (before training):** we re-scored today's head under the audible-end labels (harness sweep plus a
  4 860-rule vad_head scan) and asked for the fastest rule with FI and missed no worse than the shipped rule's on both
  corpora. Only the shipped rule qualifies, so no interim default shipped.
  - With calls missed ≤ 6.4 %, the fastest calls p50 at each calls-FI cap is 907 ms (≤ 19.3 %, but AMI FI 13 %), 803
    ms (≤ 25 %), 716 ms (≤ 30 %) and 586 ms (≤ 35 %, AMI FI 16 %).

## 2. Data

All local; nothing was downloaded. The printed licences:

| source | used for | amount | note |
|---|---|---|---|
| otoSpeech-141h (CC BY 4.0) | training | 188 conversations × 2 channels, user channel alone, 2 974 windows of 40 s (≈ 33 h) | the 16 eval conversations excluded by id; labels = the Dyadic Silero turns; print = 5 s of the same channel where the other party is silent ± 0.2 s; no speaker-identity supervision |
| AMI train (CC BY 4.0) | training | the 12 train meetings on disk (not 136), 900 eot-bench v2 windows | AMI dev = eval only |
| ICSI train | training | the 12 train meetings on disk, 885 windows | |
| smart-turn human_5_all (BSD-2-Clause) | training | 3 463 train clips + 1.2 s of −70 dBFS noise | its 399 v3.2-test clips held out for the accuracy check |
| smart-turn-data v3 / v3.1 / v3.2 | skipped | – | their cards carry no licence (research/archive/COMPLETENESS.md §1) |
| TurnBench dev | **eval only** | – | its licence (Mundo AI DPL v1.0) says evaluation only, so no training and no selection split |

The brief's 206 train meetings are not on this machine (AMI 12 + ICSI 12 local, about 18 h). I did not download
AMI / ICSI (about 19 GB).

## 3. Features and training

**Features.** Stage `feats`, 0.1-0.3 s per clip on MPS, 9 000 clips in about 35 min. It computes the served
single-mode turn-head inputs offline:
- pass 1 (masked [70,1]) → VAD head and block 4;
- the served TS-VAD track (`tsvad_stream.track_probs`: stored print from frame 0, anchored adaptation on, the
  print-fix protocol);
- the RNNT greedy decode frame by frame (the text state);
- pass 2 = the encoder conditioned on P(user) through the speaker kernels at layers 1 / 3, read at the top.

**Checked:**
- The served head on these inputs reproduces the dumped served p to 5e-5 (`verify`, calls and AMI sessions).
- It reproduces all 232 eval sessions to ≤ 7e-3. No p ≥ 0.99 crossing changes; the largest differences are three AMI
  IB4002 clips and tb_114.

**Training** (`train`, heads only). The served turn head is warm-started from the shipped weights. Everything else is
frozen and absent from the graph, so VAD, TS-VAD, speaker head and ASR outputs are bit-identical.
- **The speaker kernels are frozen too.** Training them means backprop through all 17 encoder blocks on every step,
  and it would have invalidated the pass-2 cache.
- **Objective (early confidence):**
  - positives: the last 2 speech frames (weights 0.3 / 0.6) and the first 6 silence frames after an audible end
    (weights 1 / 2 / 3 / 3 / 2 / 1.5: sure by +160-320 ms);
  - later frames: weight 0.3 until the next onset;
  - in-turn frames: 0 at weight 1;
  - pauses ≥ 160 ms inside turns, and the silence after an incomplete smart-turn clip: 0 at weight W_PAUSE (the hard
    negatives).
- Batches are drawn by source: oto 0.35-0.5, meetings 0.3-0.4, smart-turn the rest.
- The checkpoint is picked on a held-out 8 % of training conversations / meetings / clips, by end-vs-pause AUC. It
  never touches the eval sets.

| head | recipe | val AUC (start → best) | calls p50 at goal (no cut) |
|---|---|---|---|
| h1 | W_PAUSE 3, tail 0.3, lr 5e-4, 3 000 steps, 100 oto conversations | 0.59 → 0.80 | 946 ms (18.3 / 6.4; AMI 1246, 10.5 / 33.0)² |
| **h2** | W_PAUSE 1.5, tail 1.0, lr 3e-4, 2 000 steps | 0.59 → 0.79 | 989 ms (15.6 / 7.3; AMI 1167, 10.5 / 33.5) |
| h4 | as h2, 188 oto conversations, oto 0.5 of batches, 4 000 steps | 0.58 → 0.68 (larger, harder val split) | 1056 ms |
| h5 | as h4, lr 1e-4 + today's objective as an auxiliary (BCE, pos_weight 2) | 0.58 → 0.63 | 996 ms (14.7 / 7.3) |

² h1's last checkpoint; its best-BCE checkpoint gives 986 ms.

- The brief's (a)-(c) rows use h2, the first balanced recipe. h1 is 10 ms faster but was not the pre-set pick, and
  both are far from the 806 ms bar.
- h5 reaches 909 ms, 18.3 / 6.4 % (AMI 1326, 10.5 / 33.5) with VAD < 0.4 for ≥ 240 ms and p ≥ 0.8, but that rule
  cuts the demo clip.
- Per the coordinator's gate, "stop a candidate whose p ≥ 0.9 within 300 ms is below the shipped head's": every
  candidate fails it (h2 5.5 %, h5 0.9 % against 57.8 %). No further candidates were run.
- **Cross-fit.** The rule was selected on the eval sessions, as the shipped rule was. The selected rows already fail
  the bar in-sample, and a cross-fit (choose on half, score on the other half) can only make them worse, so it was not
  run.

**Not done:** the [70,0] / 80 ms-chunk variant (coordinator's item 2). At best it removes 80 ms of buffering, which is
below the 150 ms bar on its own. The 160 ms head gave no latency gain to add to it. It would also cost ASR WER
(research/LATENCY_BUDGET.md).

## What would move it

- **The VAD tail.** The served VAD head's 240-320 ms tail after call ends is the largest single term. A VAD head
  trained with the audible end as the speech end, or read with an onset / offset asymmetry, would cut the fallback path
  by up to about 250 ms without the turn head.
  - Raising the VAD threshold in the rule does not do this: at 0.6-0.9 the pauses grow as much as the ends shrink
    (best 1021 ms at ≤ 20.2 % FI).
- **Evidence for the head.** Prosody (pitch, energy) and a stronger text signal (the 0.6B's tokens or the per-turn TDT
  v3 text) are what smart-turn and LiveKit use. Our head sees only pass-2 acoustics and a 115M streaming ASR's text
  state.
- **More in-domain data.** The 206 meetings, and training on TurnBench if its licence allowed it (it does not).

## Reproduce

```bash
P=scripts/research/turn_v4.py; G="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python"
PYTHONPATH=. .venv/bin/python $P labels_eval
$G $P manifest && $G $P feats --budget 540 --batch 16     # repeat until 0 left
$G $P evalfeats && PYTHONPATH=. .venv/bin/python $P evaldump --tag served   # checks the offline path
$G $P train --tag h2 --w-pause 1.5 --tail-w 1.0 --lr 3e-4 --steps 2000
PYTHONPATH=. .venv/bin/python $P evaldump --head $W/h2/head.pt --tag h2
EOT_DUMP=$W/dump_h2 EOT_CLIP_FRAMES=$W/dump_h2_clip_frames.json $G $P scan --goal '{"calls_fi":20.2,"calls_miss":7.3,"ami_fi":10.5,"ami_miss":33.5}' --out scan.json
EOT_DUMP=$W/dump_h2 EOT_LABELS=runs/turn_v4_labels_eval.json EOT_OUT=... $G scripts/research/eot_latency.py sweep
$G $P stest --heads h2=$W/h2/head.pt
```

`W=/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v4`. It holds the feature cache (1.9 GB), the heads, the
dumps and every scan json.
