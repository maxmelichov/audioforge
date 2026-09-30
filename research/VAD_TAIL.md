# The served VAD head's tail after the end of speech (2026-09-30)

**Verdict: the tail is real, and it comes from the VAD head, not the labels. Removing it would not make end-of-turn
faster at equal false interruptions, so no VAD head was retrained.** `served_heads_v0.2.pt` is unchanged. What
shipped instead is a named choice of trade-off, `--turn-preset balanced | fast` (the table below). `balanced` stays
the default.

**Plain conclusion.** Getting below ~900 ms at ≤ 20 % false interruptions on these recordings needs a turn model that
tells a pause from an end within 200-300 ms of silence. That means text or prosody evidence and more conversational
data. A rule or a VAD change will not do it.

| preset / system | calls EOT p50 / p95 | calls FI / missed | AMI EOT p50 / p95 | AMI FI / missed |
|---|---|---|---|---|
| **`balanced`** (default): VAD < 0.4 ≥ 160 ms & p ≥ 0.99, OR 640 ms; others 960,640 | 956 / 1919 ms | 20.2 / 7.3 % | 1326 / 3758 ms | 10.5 / 33.5 % |
| **`fast`**: VAD < 0.6 ≥ 480 ms & p ≥ 0.99, OR 720 ms; others 640,640 | 886 / 1434 ms | 24.8 / 3.7 % | 1086 / 3886 ms | 16.0 / 30.5 % |
| LiveKit EnglishModel + Silero (defaults) | 567 / 3127 ms | 26.6 / 22.9 % | 1890 / 4295 ms | 12.5 / 67.5 % |
| Pipecat smart-turn v3.2 + Silero (defaults) | 237 / 3217 ms | 35.8 / 24.8 % | 385 / 3992 ms | 27.5 / 44.5 % |

What `fast` changes on calls:
- p50 is only ~70 ms faster. The session-bootstrap 90 % interval is −106 to −5 ms.
- p95 is ~490 ms faster.
- It misses half as many ends.
- It costs 4.6 points of false interruptions (still under LiveKit's).

Neither preset cuts the bundled clip under the six deliveries: the served engine was checked, and it equals the
offline rule. Details are in research/EOT_LATENCY.md "Turn presets".

Code: `scripts/research/vad_tail.py` (stages `tail`, `bounds`, `fast_scan`, `presets`). Numbers: `runs/vad_tail.json`.
The goal is today's shipped rule: calls FI ≤ 20.2 %, missed ≤ 7.3 %; AMI FI ≤ 10.5 %, missed ≤ 33.5 %.

## 1. The tail, measured

**Definition.** Tail = the end of the detector's on-run that contains the reference end, minus the reference end. The
run is followed forward. If the detector is already off at the end, the tail is the end of its last on-step before
it, which is negative.
- The served VAD is read on the label grid [0.08 v, 0.08 (v + 1)).
- Silero is read on its 32 ms chunks.
- Energy is read on 10 ms frames above max(floor + 12 dB, p95 − 40 dB).

All from the eot_latency dump (`scratch/tswer_fix/eot_dump`, the served session).

**Calls, 109 ends, p50 in ms (p25 / p75), against the labelled end:**

| threshold | served VAD | Silero |
|---|---|---|
| 0.3 | 447 (280 / 630) | −46 |
| 0.4 | 320 (180 / 498) | −56 |
| 0.5 | 212 (100 / 390) | −64 |
| 0.6 | 140 (20 / 305) | −66 |
| 0.7 | 65 (−30 / 160) | −70 |

- By corpus at 0.5: TurnBench 294 ms, oto 190 ms.
- Against Silero's own end on the same run: VAD0.5 is +352 ms, VAD0.4 +448 ms.
- Against the audible-end labels (`runs/turn_v4_labels_eval.json`) the numbers are the same within 0-10 ms.
- AMI turns are confounded by the other speakers, since the VAD is any-speaker: VAD0.5 − Silero0.5 is +384 ms p50.

**What the tail region holds (calls, label end to the VAD0.5 fall):**

| content | ends | tail p50 |
|---|---|---|
| nothing: no Silero speech, energy at the floor for most of it | 55 | 220 ms |
| energy only: breath, noise, echo (all TurnBench) | 26 | 300 ms |
| Silero speech: a soft trailing syllable | 9 | 616 ms |
| no tail | 19 | −74 ms |

So most of the tail is the head holding on over silence.

**Labels and alignment.**
- **Label window.** A frame is labelled speech if any part of its 80 ms window is speech (`data.rttm_to_frames`). An
  ideal head therefore shows 0-80 ms of "tail" on this grid, 40 ms at p50. That is the whole structural share.
- **Receptive field of block 4.** Measured by perturbing the input: a frame pair (one 160 ms chunk at `[70, 1]`) sees
  audio up to its chunk's end + 16 ms. So there is no structural lookahead delay. Odd frames are labelled up to 64 ms
  past their evidence, which, if anything, pushes the head to fall early.
- **Against its own training labels** (AMI dev VAD set, `runs/vad_single` scores, label ends followed by ≥ 400 ms of
  silence): the head's tail is 320 ms p50 at 0.5 and 400 ms at 0.4. It is ≥ 160 ms at 77 % of those ends.
- **Breakdown of the 212-320 ms:** about 40 ms is the frame grid. The rest (~170-280 ms) is the head.

## 2. Why removing it does not help

**The tail is a uniform hangover.** Inside call turns, at pauses ≥ 160 ms by Silero, the VAD0.4 fall comes 368 ms
(p50) after the pause starts. At turn ends it comes 416 ms after Silero's end. It fully bridges 67 % of those pauses
(250 pauses). A causal VAD cannot tell, at the moment speech stops, whether the user is pausing or done. Any VAD
that falls faster at ends also falls faster in pauses. The rule then has to wait longer to hold false interruptions,
which gives the time back.

**Scored with the harness's simulator and scorer** (`bounds`; the served dump; the goal above):

| variant | rules | meet the goal | fastest calls p50 at FI ≤ 20.2 % / missed ≤ 7.3 % |
|---|---|---|---|
| served VAD, shipped family | 2 368 | 1 (the shipped rule) | 956 ms |
| (a) no retrain: silence clock timed from the VAD's fall (below 0.3-0.5, from the last frame ≥ 0.5-0.9) | 3 375 | 0 | 935 ms, with AMI 12.0 / 35.5 % |
| (a) no retrain: a constant shift of the silence clock | = shorter k / fallback, already in the family | – | – |
| oracle tail-free VAD (Silero on the frame grid), + 0 / 80 / 160 / 240 / 320 ms hangover, fallback up to 1.28 s | 1 584 each | 0 | 1176-1216 ms |
| oracle tail-free VAD on the head path only, served VAD for the fallback | 360 | 0 | none (FI ≥ 29 %) |

- The oracle is an upper bound for any retrained tail-free VAD head (option (b)). It is worse than the served VAD at
  every setting.
- Retraining was therefore stopped after the block-4 feature cache of the eval sessions. That cache is 238 files in
  `scratch/vad_tail/b4`, and the stage is `b4feats`.
- Because nothing changed in the served VAD, TS-VAD arming, the turn hint and LID pooling are unchanged.

## 3. The presets

`fast` was picked (`fast_scan`, 6 966 vad_head rules) as the fastest calls p50 under calls FI ≤ 30 %, missed ≤ 7.3 %,
AMI FI ≤ 16 %, with no clip cut. How the pick went:
- The fastest rules under that bar (776-814 ms) cut the clip under 4-6 of the six deliveries.
- The fastest with no cut is 870 ms at 29.4 / 6.4 %.
- The coordinator chose the runner-up (886 ms at 24.8 / 3.7 %) for its lower FI and misses, and a p95 of 1434 ms.

Checks:
- `check`: the policy built from `TURN_PRESETS["fast"]` equals `sim_room(FAST)` on all 519 `turn_end`s.
- `served_check --preset fast`: no cut on the real engine, served equals offline, and the clip's end is answered in
  816 ms, against 976 ms for `balanced`.

## Reproduce

```bash
P=scripts/research/vad_tail.py; export PYTHONPATH=.
.venv/bin/python $P tail          # section 1 (numpy; reads the dump, the audio, runs/vad_single eval scores)
.venv/bin/python $P bounds        # section 2
.venv/bin/python $P fast_scan     # the fast preset's selection
.venv/bin/python $P presets       # both presets, both label sets, bootstrap
.venv/bin/python scripts/research/eot_latency.py check
scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py served_check --preset fast
```
