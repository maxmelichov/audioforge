# Why end of turn is worse on the 0.6B core, and the heads-only fix

2026-10-01. Script: `scripts/research/core_0p6b_turn.py` (imports `core_0p6b_heads.py`; reuses its caches).
Numbers: `runs/core_0p6b_turn.json`. Context: research/CORE_0P6B.md (§end of turn), LAYER_SWEEP_0P6B.md,
TURN_V5.md. Mac only, MPS, one job at a time through `scripts/dev/gate.sh`. The encoder is frozen throughout.

## Verdict

**Why it was worse: two heads, not the encoder.**
- **The VAD head** is a recurrent GRU trained on AMI meetings only. It keeps the VAD up for 1-2 s after the user stops
  (eval calls: 7.3 % of ends take ≥ 1 s to go quiet, against 0.9 % for the 115M) and flickers in the gaps, so the
  640 ms fallback never fires before the other party answers.
  - Putting the 115M's VAD into the 0.6B session cuts `fast`'s calls misses from 11.9 to 7.3 % and its false
    interruptions from 33.9 to 27.5 %.
  - A stateless head on the same 0.6B block, trained on the same AMI data, has no tail (p90 400 vs 1120 ms for a GRU
    on that block). The tail comes from the head and the data, not the features.
- **The v5 classifier** is confident at real ends (AUC 1.000 at +240 ms on the assistant clips). But the assistant
  trigger asks it at every quiet frame, and inside utterances it answers > 0.9 at pauses: 42 of its 53 false fires
  happen before the speaker has finished (115M: 9 of 12). Its training data never had those frames.
  - Putting the 115M's v5 p into the 0.6B session lifts the assistant row from 78.2 to 90.2 % (shared constants).
  - With both heads swapped, the 0.6B session scores like the 115M (fast 556 / 25.7 / 5.5; assistant 93.0 / 310 / 5.4).

**What ships: `served_heads_0p6b_v0.2.pt`.** The v0.1 heads, bit-identical, with every preset's constants re-picked on
held-out data (balanced now uses the v5 classifier). Against v0.1 on the evaluation sets:
- assistant 91.2 → 93.0 % accuracy and 7.1 → 4.5 % false fires at the same 374 ms;
- fast calls 530 / 33.9 / 11.9 → 487 / 26.6 / 11.0;
- balanced calls 920 / 20.2 / 11.0 → 725 / 19.3 / 10.1;
- the served session equals the offline rule on 32 / 32 sessions per preset.

**The bars are not all met.** Met: assistant accuracy and false fires, the fast p50, the balanced p50 and FI, the AMI
trade-off, served == offline, cost. Not met: the assistant p50 (374 vs 292 ms), fast FI (26.6 vs 24.8 %, CI includes
the bar), and the calls misses (fast 11.0, balanced 10.1 vs 5.5 / 7.3 %).

**The full fix was built but does not ship.** The new VAD + the retrained v5 + a block-12 per-frame turn head (`Q`)
does fix both causes on the evaluation sets: calls misses 4.6 %; assistant 95.2 % / 342 ms / 4.9 %. But it raises
balanced false interruptions on the TurnBench calls to 41 %. No held-out data looks like those quiet, noisy channels,
so nothing could select a safe decider for them.

The bars are also partly an artefact: with constants picked on held-out data by the same procedure, the 115M itself
scores fast 589 / 30.3 / 8.3 and assistant 95.0 % / 354 ms on these sets. Its shipped constants were tuned on them.

## 1. Diagnosis

### 1.0 Tools built for this

- **Evaluation cache** (`evcache`): the 232 calls / AMI sessions and the 399 assistant clips through the 0.6B's
  masked [70,1] forward (blocks 4 / 8 / 10 / 12 / 14 / 16 / 20 / 24, greedy RNNT tokens). 1.5 GB, 2 min on MPS.
  `evcheck`: from this cache, the shipped VAD head and the shipped v5 classifier reproduce the served dumps on the
  dumped frames to |Δ| ≤ 3.2e-4 (VAD) and ≤ 4.8e-4 (v5 p), with the same frame count in every session. So a head
  swapped in offline gives what the served session would give, frame for frame.
- **Held-out end-of-turn set** (`hoprep`, `ho115`, `hop6`): the turn_v4 / v5 held-out split (same seed and groups
  as every v5 model's validation; never trained on by either core's shipped heads; never an evaluation set):
  - 224 oto user-channel windows (540 turns): the calls-like scope;
  - 76 AMI windows (ES2015c): the meeting scope;
  - 279 smart-turn clips + 3 s silence, no print (st3): the assistant-like scope.
  Both cores' shipped heads are run on it, and every preset is replayed with the served policy twin. **All choices
  below are made on this set; the evaluation sets only verify.**

### 1.1 Which signal makes the 0.6B worse (cross-core swap, `swap`)

Both cores' served sessions run on the same 80 ms frame clock: the decision-ready time `t` of every frame is identical
in all 232 dumps (`lag.clock_equal`). So one core's per-frame signal can be put into the other core's session and
re-scored with the same rule. Shared preset constants; calls p50 ms / FI % / missed %, AMI the same, assistant
accuracy % / p50 ms / false fires %:

| 0.6B session with … | balanced calls | fast calls | fast AMI | assistant (smart-turn 399) |
|---|---|---|---|---|
| its own signals (as shipped) | 920 / 20.2 / **11.0** | 530 / **33.9** / **11.9** | 1017 / 19.5 / 24.5 | **78.2** / 230 / **23.2** |
| the 115M's **VAD** | 944 / 16.5 / 8.3 | 518 / 27.5 / 7.3 | 1177 / 13.5 / 28.0 | 79.9 / 230 / 20.1 |
| the 115M's **v5 p** | 920 / 20.2 / 11.0 | 535 / 33.0 / 8.3 | 1017 / 17.5 / 25.5 | 90.2 / 310 / 10.3 |
| the 115M's per-frame turn p | 905 / 25.7 / 10.1 | – | – | – |
| the 115M's TS-VAD track | 920 / 19.3 / 11.0 | 530 / 33.0 / 11.9 | 1017 / 20.0 / 27.0 | 78.2 / 230 / 23.2 |
| the 115M's VAD **and** v5 p | 944 / 16.5 / 8.3 | 556 / 25.7 / 5.5 | 1257 / 11.0 / 32.0 | 93.0 / 310 / 5.4 |
| (the 115M itself) | 956 / 20.2 / 7.3 | 547 / 24.8 / 5.5 | 1247 / 11.5 / 34.0 | 92.2 / 292 / 5.4 |

Two signals account for the whole gap; the TS-VAD track and the encoder clock do not:
- **The VAD head** drives the calls / AMI rows (fast FI 33.9 → 27.5 %, missed 11.9 → 7.3 %; balanced missed
  11.0 → 8.3 %).
- **The v5 classifier's calibration** drives the assistant row (78.2 → 90.2 % with the shared constants).
- With both swapped the 0.6B session lands on the 115M's numbers (fast 556 / 25.7 / 5.5; assistant 93.0 / 310 / 5.4).

### 1.2 The VAD: a long tail after the user stops, from the GRU head and AMI-only training (`lag`, `vadlag`)

**Frame alignment is not the cause.** The onset lag (labelled turn start → VAD > 0.5) is 80 ms p50 for both cores,
and the clocks are identical. The streaming forward was already shown equal to the masked offline forward
(CORE_0P6B.md `verify`), and the evaluation cache reproduces the served VAD on every frame (§1.0). The 0.6B frame at
time t does not see later audio than the 115M's.

**The lag is a tail, not a shift.** After the 109 labelled call ends (frames from the end until VAD < 0.4):

| VAD | p50 | p90 | ≥ 1 s | mean VAD 0.48 / 0.96 / 1.28 s after the end |
|---|---:|---:|---:|---|
| 115M served (stateless, block 4) | 320 ms | 640 ms | 0.9 % | 0.39 / 0.17 / 0.15 |
| 0.6B served (GRU-64, mix of 6 blocks, AMI only) | 240 ms | 880 ms | **7.3 %** | 0.38 / 0.28 / 0.25 |

Why the 0.6B misses 12 call ends (balanced; fast the same 12), classified on the replay (`lag.miss_*`):
- 9 of 12 are **flicker**: in the gap before the user's next turn the VAD never stays under 0.4 for the 640 ms
  fallback (the 115M: 7 of 8). Example tb_112 at 26.5 s: the audio is at −77 dBFS for 1.7 s, yet the 0.6B VAD reads
  0.5-0.8 for the first 1.3 s.
- 1 is a short turn the VAD never heard (no onset, so the rule never armed): oto_0be3 at 9.3 s, a loud turn
  (−16 dBFS) where the 0.6B VAD reads 0.0 and the 115M's 0.9.
- 2 are the decider (quiet long enough, p under θ).

**Head or features?** Heads trained on the same cached 0.6B block 12, same recipe, scored on the oto held-out
conversations (word-level activity labels, 1460 ends) and after the evaluation call ends:

| head (block 12) | data | oto held-out AUC | lag p50 / p90, ≥ 1 s | eval calls lag p50 / p90, ≥ 1 s |
|---|---|---:|---|---|
| 0.6B shipped (GRU-64, 6-block mix) | AMI | 0.9856 | 240 / 720, 2.1 % | 240 / 880, 7.3 % |
| 115M shipped (stateless, its block 4) | its recipe | 0.9891 | 240 / 560, 0.1 % | 320 / 640, 0.9 % |
| GRU-64 (`gru12`) | AMI | 0.9826 | 400 / 720, 2.9 % | 480 / 1120, **11.0 %** |
| stateless MLP (`mlp12`) | AMI | 0.9860 | 160 / 560, 0.6 % | **80 / 400, 0.0 %** |
| GRU-64 + oto (`gru12o`) | AMI + oto | 0.9942 | 80 / 160, 0 % | 80 / 480, 0.9 % |
| stateless MLP + oto (`mlp12o`) | AMI + oto | 0.9943 | 80 / 160, 0 % | 80 / 320, 0 % |

- With the same features and data, **the recurrent head makes the tail**: the GRU keeps the VAD up for 1-2 s after
  speech. A stateless head on the same block drops faster than even the 115M's.
- **AMI-only training is the second half.** Adding in-domain two-party conversation (1200 oto user-channel windows
  from the turn cache, word-activity labels) removes the tail for every head type.
- **The 0.6B features are not the problem**: a stateless head on block 12 has a shorter tail than the 115M's VAD.

### 1.3 The v5 classifier: confident at the end, but asked inside utterances

On the evaluation dumps (diagnostic only), at the frame 240 ms after each assistant clip's speech end the 0.6B v5
separates complete from incomplete clips better than the 115M's (AUC 1.000 vs 0.996; complete clips with p > 0.9:
98.3 vs 75.4 %; incomplete > 0.9: 0.0 vs 0.4 %). On call ends vs in-turn quiet frames it ranks about the same
(AUC 0.689 vs 0.706).

So "calibrated differently" is not a weak end-of-turn decision. Where the 0.6B's false fires happen (assistant
preset with the shared constants, incomplete clips, `ff` replay):

| | fires on an incomplete clip | inside the utterance (before the speech end) | after the cut |
|---|---:|---:|---:|
| 115M | 12 / 201 | 9 | 3 |
| 0.6B | 53 / 201 | **42** | 11 |

- The assistant trigger asks the classifier at every quiet frame: energy within 6 dB of the floor, or VAD < 0.4.
- Inside an utterance, the 0.6B classifier answers > 0.9 at pauses where the 115M's does not.
- The training data has no such frames for smart-turn clips. The st3 samples start after the utterance's last VAD
  frame. Mid-utterance pauses appear only in the conversational clips (pauses ≥ 160 ms, dips).
- The 0.6B's own re-tuned constants (320 ms quiet, p > 0.99) hide this at a cost of 80 ms.

### 1.4 The held-out set does not see the TurnBench part of the gap

The calls scope is 16 TurnBench + 16 oto conversations. Split (shipped heads, shared constants; p50 / FI / missed):

| | TurnBench (56 ends) fast | oto (53 ends) fast | TurnBench balanced | oto balanced |
|---|---|---|---|---|
| 115M | 607 / 19.6 / 10.7 | 526 / 30.2 / 0.0 | 1019 / 12.5 / 14.3 | 926 / 28.3 / 0.0 |
| 0.6B | 665 / 28.6 / **21.4** | 400 / 39.6 / 1.9 | 1001 / 16.1 / **19.6** | 905 / 24.5 / 1.9 |

- The 0.6B's extra misses are TurnBench sessions. Their user channels are quiet: speech at −41.8 dBFS median against
  −22.3 for oto. They also have a real noise floor (non-turn frames −70 dBFS; oto channels are digital zero between
  turns) and weak crosstalk (user-channel / mix energy correlation 0.22 in the gaps).
- The held-out oto windows are digital-zero between turns. On them the shipped 0.6B already ties the 115M with the
  shared constants (fast 631 / 29.4 / 7.0 vs 640 / 28.9 / 6.1).
- TurnBench is evaluation-only, so a fourth held-out scope was added (`quiet`): the 224 held-out oto windows made to
  look like those channels. Per clip:
  - speech scaled to a level drawn from −48…−34 dBFS;
  - a coloured noise floor drawn from −76…−58 dBFS;
  - on half of the clips, the other party at −35 dB under the user;
  - the print gets the same gain and floor.
  Both cores were re-run on it. It reproduces part of the effect (the shipped 0.6B VAD: held-out AUC 0.986 → 0.975,
  gap flicker 1.8 → 5.3 %) but not the TurnBench tail (≥ 1 s: 1.2 % vs 7.3 %).
- The levels come from aggregate statistics of the evaluation audio, not from any result on it.

## 2. The fix, step by step (selected on held-out only)

### 2.1 VAD candidates (`vadtrain`)

Every head reads cached 0.6B frames. Training:
- AMI train windows (1200, held-out meetings removed), optionally the shipped recipe's 2 SpecAugment re-encoded views
  (SA);
- 15 % room-tone crops;
- feature dropout, noise and time masks;
- optionally 1200 oto user-channel train windows ("oto") and their quiet-channel variants ("quiet oto", stage
  `quiet --which train`);
- AdamW, 3000 steps, early stopping on the held-out mean AUC.

The held-out mean AUC (selection) covers the AMI held-out meetings, the ICSI train windows, the oto held-out
conversations and their quiet variants, each with room-tone negatives. AMI dev / ICSI dev are the published VAD
rows: the shipped 0.6B scores 0.9507 / 0.9715 and 0.9055 / 0.9245. The last two columns are the oto held-out clips
(clean / quiet).

| tag | head | data | selection AUC | AMI dev F1 / AUC | ICSI dev F1 / AUC | room-tone p95 | oto AUC | lag p90 ms |
|---|---|---|---:|---|---|---:|---|---|
| `mlp12` | mlp b12 | AMI | 0.981 | 0.9383 / 0.9699 | 0.9179 / 0.9311 | 0.039 | 0.986 / – | 560 / – |
| `gru12` | gru64 b12 | AMI | 0.980 | 0.9479 / 0.9699 | 0.8972 / 0.9285 | 0.223 | 0.983 / – | 720 / – |
| `mlp12o` | mlp b12 | AMI + oto | 0.984 | 0.9500 / 0.9698 | 0.9045 / 0.9316 | 0.191 | 0.994 / – | 160 / – |
| `gru12o` | gru64 b12 | AMI + oto | 0.983 | 0.9494 / 0.9699 | 0.8963 / 0.9296 | 0.252 | 0.994 / – | 160 / – |
| `gru12o_ow3` | gru64, offset-weighted ×3 | AMI + oto | 0.984 | 0.9422 / 0.9702 | 0.9131 / 0.9322 | 0.228 | 0.994 / – | 80 / – |
| `gru16o` | gru16 (shorter memory) | AMI + oto | 0.983 | 0.9418 / 0.9678 | 0.9051 / 0.9242 | 0.211 | 0.994 / – | 160 / – |
| `gru12o_c96` | gru64, 96-frame crops | AMI + oto | 0.983 | 0.9444 / 0.9686 | 0.9025 / 0.9271 | 0.017 | 0.994 / – | 80 / – |
| `conv12o` | causal conv (3 frames) | AMI + oto | 0.982 | 0.9504 / 0.9707 | 0.8982 / 0.9212 | 0.195 | 0.994 / – | 80 / – |
| `mlpmixo` | mlp, mix of b8 / b12 / b24 | AMI + oto | 0.983 | 0.9505 / 0.9698 | 0.9013 / 0.9209 | 0.151 | 0.995 / – | 80 / – |
| `mlp12oq` | mlp b12 | AMI + oto + quiet oto | 0.984 | 0.9520 / 0.9701 | 0.9026 / 0.9131 | 0.145 | 0.994 / 0.990 | 80 / 80 |
| `mlpmixoq` | mlp, mix of b8 / b12 / b24 | AMI + oto + quiet oto | 0.984 | 0.9512 / 0.9700 | 0.9030 / 0.9165 | 0.152 | 0.995 / 0.990 | 80 / 80 |
| `gru12oq_c96` | gru64, 96-frame crops | AMI + oto + quiet oto | 0.984 | 0.9487 / 0.9711 | 0.8969 / 0.9208 | 0.044 | 0.994 / 0.988 | 80 / 160 |
| **`mlp12oq_sa`** | **mlp b12** | **AMI + SA + oto + quiet oto** | **0.986** | 0.9492 / **0.9725** | **0.9091 / 0.9325** | 0.092 | **0.995 / 0.990** | 80 / 80 |

Block 12 stands in for the sweep's block 14. They are 0.0024 apart on the sweep's mean AUC (within its noise), and
block 12 is the one cached for every turn clip, so the v5 classifier and the turn rows can be retrained and replayed
on the same VAD.

**Chosen: `mlp12oq_sa`.** It has the best selection AUC and the best quiet-channel AUC. Against the shipped GRU it is
equal or better on every VAD row: AMI AUC +0.001, ICSI F1 +0.004 / AUC +0.008, room-tone p95 0.09 vs 0.21. The one
exception is AMI F1 at 0.5: −0.0015 (paired CI in §3). It is the 115M's head type: stateless, one block, 65.7 K
parameters, no recurrent state per session.

### 2.2 The v5 classifier (`teacher`, `segtrain`)

All variants keep the TURN_V5.md architecture and the c5 recipe: 2 × 256 Transformer on ≤ 8 s of block 12, plus
VAD / P(user) / P(other) and the RNNT text; 3000 steps, lr 1e-4, batch 256. Each variant changes one thing on top:
- `vadall`: the VAD channel and event frames come from the chosen VAD, so the classifier is trained on what it will
  read when served.
- `--kd 1`: distillation from the 115M's shipped v5. `teacher` gives c5's calibrated p on every frame of every turn
  clip from the 115M's own cached inputs; 98 % of clips have them. The loss adds BCE(main logit, teacher p).
- `--st3dips 0.5`: new negatives inside smart-turn utterances. These are the frames where the assistant trigger asks
  (energy within 6 dB of the clip's floor, or VAD < 0.4, between the first and last VAD-on frame), label 0, drawn as
  their own kind at batch weight 0.5. This is §1.3's missing case.

Held-out assistant scope (279 st3 clips; the 115M with its shipped constants: 91.8 % / 350 ms / 3.5 % false fires).
Each cell is the best rule of 576 (k 2-5 frames, VAD < 0.4 / 0.5, energy quiet 4 / 6 dB / off, p 0.8-0.99, fallback
37 / 40 / 43 frames):

| classifier | VAD channel | best accuracy (p50, FF) | best with FF ≤ 3.5 % and p50 ≤ 350 ms |
|---|---|---|---|
| s12 (shipped) | shipped GRU | – | none |
| kd1 | `mlp12oq_sa` | 93.5 % (521 ms, 8.5 %) | none |
| **kd1st** | `mlp12oq_sa` | 96.1 % (521, 7.8 %) | **96.1 % / 281 ms / 3.5 %** (energy quiet 4 dB, 160 ms, VAD < 0.5, p > 0.8, fallback 3440 ms) |

Two findings came out of this table:
- **The in-utterance negatives remove the in-utterance false fires.** On held-out incomplete clips at k 2 / p 0.98,
  fires before the speech end go from 12 (s12) and 2 (kd1) to **0** (kd1st; the 115M: 0).
- **The 2960 ms fallback was tuned on the 115M's VAD tail.** 10 of kd1st's 11 remaining false fires were that
  fallback. A VAD with no hangover goes quiet up to ~0.3 s before the audible end, so the timer lands inside the
  incomplete clips' 3 s window; the 115M's VAD tail added those ~0.3 s. A 3440 ms timer (43 frames) restores the same
  margin. This is a per-model constant (`cfg["turn_presets"]`).

### 2.3 The per-frame turn head of `balanced` (`turntrain`)

Same recipe as the shipped `turn_f1` (served TurnHead without speaker kernels, TS-VAD columns, RNNT text, the served
objective, 3000 steps). The only change is the block it reads: block 12 instead of the top block 24. Held-out (turn_v4
split, 754 ends / 1106 pauses; checkpoint at the best end-vs-pause AUC):

| head | end-vs-pause AUC | reach ≤ 240 ms at p ≥ 0.9 | pause false alarms at p ≥ 0.9 | reach at 10 % pause FA |
|---|---:|---:|---:|---:|
| `turn_f1` (block 24, shipped) | 0.710 | 24.1 % | 18.7 % | 22.4 % |
| **`b12` (block 12)** | **0.812** | **34.7 %** | **15.7 %** | **41.0 %** |

- The top block is the ASR's output layer, which has lost most of the prosody and identity the turn decision needs
  (LAYER_SWEEP_0P6B.md: identity is gone by block 24).
- Block 12 ranks ends above pauses far better, and reaches a confident p sooner with fewer false alarms.
- With it, `balanced` meets the 115M's held-out numbers with a head rule at θ 0.9. θ 0.9 needed a per-preset
  `theta` in `cfg["turn_presets"]` (serve change, `Session._vad_head_theta`; the client's `eot_threshold` still
  wins; models without it keep 0.99).

### 2.4 The full fix on held-out, then once on the evaluation sets: the TurnBench calls do not follow

**Selected on held-out (`Q`).** VAD `mlp12oq_sa`, v5 `kd1stq` (kd1st + the 1200 quiet-channel oto windows), and the
block-12 per-frame head. Each preset's constants were scanned (`hoscan`: 576 / 720 / 912 rules) and picked against
the 115M's shipped presets replayed on the same held-out clips (`ho_bars`; held-out p50 / FI / missed, assistant
accuracy / p50 / FF):

| preset | 115M on held-out (calls · quiet · AMI) | `Q` pick on held-out | rule |
|---|---|---|---|
| balanced | 900 / 28.5 / 6.5 · 750 / 39.3 / 7.2 · 1230 / 18.4 / 27.6 | 651 / 28.1 / 4.8 · 661 / 35.9 / 4.6 · 1161 / 13.2 / 27.6 | head, VAD < 0.4 ≥ 240 ms & p ≥ 0.9, OR 960 ms |
| fast | 640 / 28.9 / 6.1 · 805 / 32.2 / 7.8 | 656 / 28.3 / 5.9 · 591 / 31.9 / 5.2 | v5 at ≥ 320 ms of VAD < 0.6, p > 0.7, OR 960 ms |
| assistant | 91.8 / 350 / 3.5 | 96.8 / 361 / 3.5 (one 40 ms step over) | v5 at ≥ 160 ms energy-or-VAD quiet, p > 0.9, OR 3440 ms |

**Verified once on the evaluation sets** (95 % session bootstrap CIs):

| preset | 0.6B v0.1 | `Q` | 115M |
|---|---|---|---|
| balanced, calls | 920 / 20.2 / 11.0 | 520 [410, 740] / **41.3 [31.9, 50.9]** / 4.6 [0.0, 11.2] | 956 / 20.2 / 7.3 |
| fast, calls | 530 / 33.9 / 11.9 | 637 [473, 796] / 32.1 [22.3, 42.0] / **4.6 [0.9, 9.8]** | 547 / 24.8 / 5.5 |
| assistant | 91.2 / 374 / 7.1 | **95.2 [93.0, 97.2] / 342 [310, 374] / 4.9 [2.3, 8.1]** | 92.2 / 292 / 5.4 |

- The VAD and classifier fixes do what the diagnosis said they would. Calls misses fall to 4.6 % in both
  conversational presets (v0.1: 11.0 / 11.9 %). The assistant preset beats v0.1 on all three numbers and the 115M on
  accuracy and false fires.
- **But the false interruptions on the TurnBench calls do not follow the held-out set.** Balanced FI is 48.2 % on
  the 16 TurnBench sessions and 34.0 % on the 16 oto ones. Of the TurnBench interruptions, 22 come from the head path
  (the block-12 head at θ 0.9 inside quiet, noisy pauses); of fast's, 11 come from the 960 ms fallback.
- A VAD without a hangover exposes every pause of those quiet, noisy channels to the timers. The 115M's VAD, by
  flickering on the TurnBench noise floor, both protects those pauses (fewer FI) and misses ends (its 10.7 % misses
  there).
- The quiet-channel held-out scope did not predict this: the 115M scores worse than `Q` on it (balanced FI 39.3 vs
  35.9 %). TurnBench is evaluation-only, so no TurnBench-like audio could be added to training or selection.
- `Q` regresses the 0.6B's default preset (balanced FI 20.2 → 41.3 %), so it is **not shipped**.

**The bars come from tuning on the evaluation sets.** The same held-out procedure applied to the 115M itself (its
own heads, constants picked on the same held-out clips, `hoscan --vad 115m`) gives on the evaluation sets:

| preset | 115M, shipped constants (tuned on the evaluation sets) | 115M, constants picked on held-out |
|---|---|---|
| balanced, calls | 956 / 20.2 / 7.3 | 956 / 20.2 / 7.3 (held-out picks the shipped rule) |
| fast, calls | 547 / 24.8 / 5.5 | 589 [540, 663] / 30.3 [21.8, 40.0] / 8.3 [3.2, 14.4] |
| assistant | 92.2 / 292 / 5.4 | 95.0 [93.0, 97.0] / 354 [322, 370] / 3.1 [1.1, 5.7] |

So under the rule this task imposes (held-out only), the 115M does not reach its own fast and assistant bars either.
Its published `fast` and `assistant` constants were chosen on these evaluation sets (`core_0p6b_heads.stage_preset_scan`
and TURN_V5.md say so).

### 2.5 What ships: v0.2 = the v0.1 heads with per-model constants re-picked on held-out

Since `Q` cannot ship, the constants of the shipped v0.1 heads were re-picked on the same held-out set with a
**non-regression** criterion: for each preset, the fastest rule (assistant: the most accurate) whose held-out FI and
missed (FF and p50) are no worse than v0.1's own constants on the same clips, on every held-out scope (`nonreg`). The
picks:

| preset | v0.1 constants | v0.2 constants (held-out pick) | held-out v0.1 → v0.2 |
|---|---|---|---|
| balanced | head: VAD < 0.4 ≥ 160 ms & p ≥ 0.99, OR 640 ms | **v5 classifier**: at ≥ 320 ms of VAD < 0.4, p > 0.6, re-asked; OR 720 ms | calls 931 / 22.0 / 8.5 → 721 / 21.7 / 8.0; quiet 891 / 28.1 / 8.0 → 961 / 24.1 / 8.0; AMI FI 13.2 → 10.5 |
| fast | v5 at ≥ 80 ms of VAD < 0.6, p > 0.7, OR 640 ms | v5 at ≥ 80 ms of VAD < 0.4, p > 0.5, OR 720 ms | calls 631 / 29.4 / 7.0 → 466 / 29.1 / 5.9 |
| assistant | v5 at ≥ 320 ms energy-or-VAD quiet, p > 0.99, OR 2960 ms | v5 at ≥ 160 ms of VAD < 0.5, p > 0.98, OR **3440 ms** | 91.4 / 441 / 7.1 → 95.7 / 441 / 2.8 |

These constants needed two serve changes, each a per-model option (models without it are unchanged):
- a preset without a `turn_model` (balanced) can get the v5 decider from `cfg["turn_presets"]` (`serve.model_presets`,
  `V5_TURN_MODEL`);
- a preset can carry its own head `theta` (used by `Q`'s balanced; not by v0.2).

The heads are bit-identical to v0.1 (state hash `564a419c…`); only `cfg["turn_presets"]` differs.

## 3. Result on the evaluation sets (verification; nothing below was used to choose)

Same harnesses and audio as the 115M rows (core_0p6b_heads `_score_sets` protocol: served chunk compute added,
eot_latency / eot_assistant scorers). 95 % session bootstrap CIs, 1000 resamples (`evverify`). The v0.2 rows are the
v0.1 sessions' own frames under v0.2's rules: the heads are bit-identical. The served v0.2 engine gives the same
turn_end times as that replay on 32 of 32 sessions per preset (`servedcheck`: 16 assistant clips + 16 calls each).

| row | bar (115M) | 0.6B before (v0.1) | **0.6B after (v0.2, ships)** | 0.6B `Q` (not shipped) |
|---|---|---|---|---|
| assistant (399): accuracy / p50 / false fires | 92.2 % / 292 ms / 5.4 % | 91.2 [88.5, 94.0] / 374 [348, 405] / 7.1 [4.0, 10.4] | **93.0 [90.5, 95.5] / 374 [373, 405] / 4.5 [1.9, 7.2]** | 95.2 [93.0, 97.2] / 342 [310, 374] / 4.9 [2.3, 8.1] |
| fast, calls (109): p50 / FI / missed | 547 / 24.8 / 5.5 | 530 [352, 753] / 33.9 [23.4, 44.2] / 11.9 [5.4, 19.8] | **487 [374, 616] / 26.6 [18.1, 34.8] / 11.0 [5.3, 17.8]** | 637 [473, 796] / 32.1 [22.3, 42.0] / 4.6 [0.9, 9.8] |
| balanced, calls: p50 / FI / missed | 956 / 20.2 / 7.3 | 920 [878, 1003] / 20.2 [13.0, 28.6] / 11.0 [5.6, 17.5] | **725 [616, 845] / 19.3 [12.1, 26.4] / 10.1 [4.9, 16.0]** | 520 [410, 740] / 41.3 [31.9, 50.9] / 4.6 [0.0, 11.2] |
| balanced, AMI (200): p50 / FI / missed | 1326 / 10.5 / 33.5 | 1176 [1017, 1337] / 13.0 [8.5, 18.0] / 28.0 [22.0, 34.5] | **1177 [1018, 1337] / 11.0 [6.5, 15.5] / 33.0 [26.5, 40.0]** | 1097 / 22.0 / 25.5 |
| served == offline (per preset) | 16 / 16 | 16 / 16 | **32 / 32** (16 clips + 16 calls) | – |
| engine cost per 160 ms chunk, MPS p50 (balanced / fast / assistant) | ≤ 45 ms | 39.4 / 44.5 / 44.4 | **43.4 / 43.4 / 44.8** | – |

Against the bars:
- **assistant:** accuracy met (93.0 ≥ 92.2) and false fires met (4.5 ≤ 5.4). **p50 not met** (374 vs 292 ms).
- **fast:** p50 met (487 ≤ 547). **FI not met** (26.6 vs 24.8; the CI contains the bar). **Missed not met**
  (11.0 vs 5.5).
- **balanced, calls:** p50 met (725 ≤ 956) and FI met (19.3 ≤ 20.2). **Missed not met** (10.1 vs 7.3).
- **balanced, AMI:** on the 115M's point of the FI–missed trade-off (11.0 / 33.0 vs 10.5 / 33.5: one window apart on
  each side), and 150 ms faster.
- **served == offline:** met.
- **cost:** met.

Against v0.1, every calls row and the assistant row improve or hold. The balanced AMI row moves along the trade-off:
FI 13.0 → 11.0, missed 28.0 → 33.0. Most differences are inside the CIs (109 call ends, 16 + 16 sessions).

**Other 0.6B rows are unchanged.** v0.2 changes only `cfg["turn_presets"]`, so WER, tWER, VAD, tracking, speaker and
LID are the v0.1 numbers.

## 4. Why the bars are not met, and what did not ship

1. **The calls misses (fast 11.0 %, balanced 10.1 %) are the VAD.** §1.2 proves it; `Q`'s VAD halves them (4.6 %).
   It cannot ship with a turn decider that was selectable on held-out data: without the shipped GRU's hangover, the
   TurnBench pauses reach the timers (§2.4).
2. **The assistant p50 needs `Q`'s classifier** (in-utterance negatives, distillation) **and its VAD**: 342 ms with
   both; v0.2's re-tuned constants alone stay at 374 ms. These are tied to the same VAD.
3. **The bars themselves** are the 115M's constants tuned on these evaluation sets. With constants picked on held-out
   data, the 115M scores fast 589 / 30.3 / 8.3 and assistant 95.0 % / 354 ms / 3.1 % (§2.4).

Built, measured, kept for a follow-up (TF = `scratch/core_0p6b/turnfix`):
- VAD `mlp12oq_sa`: stateless MLP on block 12, trained on AMI + SA views + oto + quiet oto. Against the shipped GRU:
  - AMI dev F1 0.9492 vs 0.9507 (Δ −0.0015 [−0.0041, +0.0012]), AUC 0.9725 vs 0.9715 (Δ +0.0010 [−0.0015, +0.0037]);
  - ICSI dev F1 0.9091 vs 0.9055 (Δ +0.0036 [+0.0014, +0.0059]), AUC 0.9325 vs 0.9245 (Δ +0.008 [+0.003, +0.013]);
  - eval-call offset lag p90 96 vs 880 ms.
- v5 `kd1stq` (`TF/seg/kd1stq`), `kd1st`, `kd1`, `skd1st`; the per-frame turn head `turn_b12` (held-out AUC 0.812 vs
  0.710).
- Serve support for both is in: a per-preset `theta`, and a v5 decider for balanced.

What the next step needs: audio like the TurnBench user channels (quiet speech, a real noise floor, crosstalk) for
training and held-out selection. The coloured-noise simulation (`quiet`) was not enough. With such data, `Q`'s heads
are the starting point.

Not tried:
- **v5 on block 10, or a block mix.** Blocks other than 8 / 12 / 24 are not cached for the 20 k turn clips. In the
  sweep, block 10 ranked below block 12 on held-out end-vs-pause AUC (0.755 vs 0.761), so the held-out rule would not
  pick it.
- **Richer v5 inputs** (energy, token timing beyond the text features). The 115M's v5 has the same three tracks + text;
  TURN_V5.md's prosody ablation was negative.

## Reproduce

```bash
P=scripts/research/core_0p6b_turn.py; G="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python"
$G $P evcache --which calls; $G $P evcache --which asst; $G $P evcheck --n 40      # evaluation cache (eval only)
$G $P lag; $G $P swap                                                              # diagnosis 1.1-1.2 (dumps)
$G $P hoprep; $G $P ho115; $G $P quiet --which val; $G $P hoprep; $G $P ho115      # held-out set (+ quiet scope)
$G $P quiet --which train --n 1200                                                 # x3 (resumable)
$G $P vadtrain --tag mlp12oq_sa --arch mlp --vblocks 12 --oto 1200 --otoq 1200 --sa  # (+ the other tags in 2.1)
$G $P vadlag --tags served,115m,mlp12,gru12,mlp12o,gru12o,mlp12oq_sa; $G $P vadcmp --vad mlp12oq_sa
$G $P vadall --vad mlp12oq_sa; $G $P teacher                                       # x3 each (resumable)
$G $P segtrain --tag kd1stq --vad mlp12oq_sa --block 12 --kd 1 --st3dips 0.5 --quiet-train   # x4 (resumable)
$G $P turntrain --tag b12 --block 12 --lr 1e-3
$G $P hop6 --vad mlp12oq_sa --seg kd1stq --turn b12; $G $P hoscan --vad mlp12oq_sa --seg kd1stq --turn b12
$G $P hop6 --vad served --seg s12 --turn f1; $G $P hoscan --vad served --seg s12 --turn f1; $G $P hoscan --vad 115m
$G $P evp6 --vad mlp12oq_sa --seg kd1stq --turn b12; $G $P evverify --tags ... --rules-json ...   # verification
$G $P build --tag v0.2 --ship --version 0.2 --presets-json '<v0.2 constants>'
$G $P servedcheck --tag v0.2 --n 16 --fam assistant; ... fast; ... balanced; $G $P cost --tags v0.1,v0.2 --fam balanced,fast,assistant
```
