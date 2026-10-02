# Real two-party channels for the 0.6B turn heads (running log)

Started 2026-10-02 12:00. Brief (the user): "The turn head: find the data and train, and we're finished." The open
head is the 0.6B core's turn taking. CORE_0P6B_TURN.md §1.4 / §2.4 and FIXALL.md §2 showed that the better `Q` heads
(stateless VAD `mlp12oq_sa` + v5 classifier `kd1stq`) fix the missed call ends and reach 95 % / 342 ms on the
assistant clips, but double the false interruptions on the quiet, noisy TurnBench user channels. No training or
held-out audio looked like those channels, and the simulated quiet-channel scope did not predict it.

Script: `scripts/research/turn_data.py` (data, held-out scope, the test of the data); training and building reuse
`scripts/research/core_0p6b_turn.py` (new options `--ihm*`, `--base`). Numbers: `runs/turn_data.json`. Scratch:
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/turndata/`. Mac only, MPS, one heavy job at a time
(`scripts/dev/gate.sh`). TurnBench stays evaluation-only; selection on held-out only.

## 1. The data

### 1.1 What was searched, and the licences

Searched: the Hugging Face Hub, OpenSLR, the corpora's own sites (checked 2026-10-02). Only sources whose licence
allows training a commercial model were downloaded.

| source | licence (official source) | commercial training | per-speaker channel | real noise floor / crosstalk | timings | used |
|---|---|---|---|---|---|---|
| **otoSpeech-full-duplex-280h** (raw) | CC BY 4.0 (HF card; terms: do not identify speakers, respect redactions) | yes | yes (48 kHz stereo, one party per channel) | yes: raw, not denoised (levels and floors of the never-touched part: §1.3) | none: derived (§1.3) | **yes** |
| **AMI individual headsets** (IHM) | CC BY 4.0 (groups.inf.ed.ac.uk/ami/corpus/license.shtml) | yes | yes (one headset per participant, 4 per meeting) | yes: real room, strong crosstalk from the other participants | manual word timings | **yes** |
| otoSpeech-full-duplex-processed-141h | CC BY 4.0 | yes | yes | no: denoised, digital zero between turns | none (Silero) | already in training |
| otoSpeech-full-duplex-turn-104h | oto-speech-license-v1.0 | **no** (restricts commercial use) | yes | – | yes | no |
| AppTek Call-Center Dialogues | CC BY-SA 4.0, but the card lists training as out-of-scope use | unclear | yes (16 kHz per speaker) | VoIP, home noise | no times per channel | no (stated intent) |
| CHiME-6 (OpenSLR 150) | CC BY-SA 4.0 | yes | in-ear mic per participant | yes (dinner parties) | utterances | not needed (97 GB, 4 speakers) |
| ICSI close-talk | CC BY 4.0 | yes | yes | yes | words | not needed (AMI IHM covers the case) |
| NOTSOFAR-1 | CC BY 4.0 | yes | close-talk in train | yes | words | not needed |
| AliMeeting (OpenSLR 119) | CC BY-SA 4.0 | yes | headset | yes | segments | no (Mandarin) |
| EdAcc | CC BY-SA 4.0 | yes | **no** (mono mix of the Zoom call) | yes | segments | no (not a user channel) |
| Earnings-21 / 22 | transcripts CC BY-SA 4.0; **audio licence unstated** | uncertain | no (mono mix) | phone lines | partial | no |
| VoxConverse | CC BY 4.0 "for research purposes"; video copyright stays with the owners | risky | no (mono) | broadcast | RTTM | no |
| CANDOR | BetterUp registration terms (the CC BY-NC figure is the paper's licence) | no / unknown | yes | yes | yes | no |
| Seamless Interaction, SpokenWOZ, MSDWild, MagicData samples | CC BY-NC 4.0 / research-only | no | – | – | – | no |
| Fisher / Switchboard / CallHome | LDC (paid for-profit licence) | no | yes (8 kHz) | yes | yes | no |
| People's Speech | CC BY / CC BY-SA | yes | no (single-speaker segments) | – | – | no (not conversational) |
| NVIDIA Granary | CC BY 4.0 labels, source-licensed audio | yes | no | – | – | no (no conversational part) |
| DailyTalk | CC BY-SA 4.0 (built from DailyDialog scripts, CC BY-NC-SA) | ambiguous | yes | no (studio, acted) | words | no |
| Behavior-SD | CC BY 4.0 | yes | yes | no (TTS) | utterances | no |
| MUSAN (OpenSLR 17) | CC BY 4.0 | yes | – | noise | – | not needed (see §1.4) |
| OpenSLR 26 / 28 RIRs | Apache 2.0 (28 bundles RWCP / REVERB / AIR RIRs with their own terms) | yes | – | – | – | not needed |
| DEMAND | CC BY 4.0 (Zenodo) | yes | – | noise | – | not needed |
| WHAM! noise | CC BY-NC 4.0 | **no** | – | – | – | no |

(log continues below)

### 1.2 Splits: train, held-out and never-touched are disjoint recordings

Standing rule (2026-10-02): no number in this file, FINAL_COMPARE.md, numbers_final.json or the final report comes
from audio that any head or preset was trained or selected on. Each new source was split once, before use, by
recording (meeting / session / call), into train / held-out (selection) / never touched (public rows only):

| source | train | held-out (selection) | never touched (reported) |
|---|---|---|---|
| AMI individual headsets | 45 AMI train-list meetings (every 3rd; TS3011b / ES2015c excluded) | the 18 AMI dev meetings | the 16 AMI test meetings (the AMI test turn row's meetings: both evaluation-only) |
| otoSpeech-280h raw | 64 sessions (not an evaluation conversation, not a turn_v4 / v5 held-out conversation) | 32 sessions not in the local 141 h slice | 32 more such sessions |
| AppTek call-center | none (the card excludes training) | 48 calls: en-US_General, en-GB, en-US_Southern, en-CA | 48 calls: en-AU, en-IE, en-ZA, en-IN |

The disjointness check (`turn_data.py splits`, by recording id) is printed in §1.3.

### 1.3 The test of the data, on the never-touched parts

**Disjointness, checked by recording id** (`turn_data.py splits` → `runs/turn_data.json` `splits`, ids in
`scratch/turndata/splits.json`). Sets: train_new 109 recordings (45 AMI meetings + 64 oto sessions), heldout_new 98
(18 AMI meetings + 32 oto sessions + 48 AppTek calls), never_new 96 (16 AMI meetings + 32 oto sessions + 48 AppTek
calls), train_shipped 379 (what the shipped heads trained on: turn_v4 / v5 oto conversations, AMI / ICSI train
meetings), heldout_old 16 (the turn_v4 / v5 held-out conversations and meetings), eval 32 (16 oto evaluation
conversations + 16 AMI test meetings; TurnBench dev is a corpus of its own that nothing trains on). Smart-turn: train
3463 / test 399 clips, overlap 0.

| intersection | recordings |
|---|---:|
| train_new ∩ heldout_new | 0 |
| train_new ∩ never_new | 0 |
| heldout_new ∩ never_new | 0 |
| train_new ∩ eval | 0 |
| heldout_new ∩ eval | 0 |
| heldout_new ∩ train_shipped | 0 |
| never_new ∩ train_shipped | 0 |
| never_new ∩ heldout_old | 0 |
| train_new ∩ heldout_old | 0 |
| heldout_new ∩ heldout_old | 0 |
| train_shipped ∩ eval | 0 |
| never_new ∩ eval (allowed: the 16 AMI test meetings, evaluation-only in both) | 16 |

**The test.** A held-out scope is useful only if it predicts what TurnBench showed: the shipped v0.2 (its heads and
held-out constants) against `Q` (its heads and its held-out rules, CORE_0P6B_TURN §2.4), the two conversational
presets. The decision was made on the held-out parts before any training; the numbers below are the same test
re-run on the never-touched parts (the held-out parts gave the same pattern). The target is the TurnBench half of the
evaluation calls (16 sessions, 56 ends; **not a public test split**; re-scored from the cached evaluation signals of
CORE_0P6B_TURN, no new evaluation run):

| preset (p50 ms / FI % / missed %) | v0.2 on TurnBench | Q on TurnBench | Q − v0.2 (FI, missed) |
|---|---|---|---|
| balanced | 892 / 14.3 / 17.9 | 478 / 48.2 / 8.9 | **+33.9**, −9.0 |
| fast | 641 / 21.4 / 19.6 | 668 / 30.4 / 8.9 | **+9.0**, −10.7 |

A scope passes when Q − v0.2 on FI is positive with a paired 95 % CI above zero for both presets, the balanced gap is
large (≥ 20 points), and Q misses fewer ends. Never-touched parts (paired window bootstrap, 1000 resamples; FI and
missed in percentage points):

| never-touched scope | windows / user turns | balanced Q − v0.2: FI, missed | fast Q − v0.2: FI, missed | passes? |
|---|---|---|---|---|
| AMI headsets, test meetings | 359 / 843 | +13.6 [+10.9, +16.2], −7.7 | **−4.8** [−7.3, −2.6], +0.1 | no |
| the same at TurnBench levels | 353 / 827 | +15.0 [+12.5, +17.6], −8.1 | **−2.6** [−5.1, −0.2], −0.8 | no |
| otoSpeech-280h raw, 32 sessions | 378 / 958 | +7.6 [+5.3, +10.0], −4.4 | **−2.6** [−4.8, −0.2], −1.7 | no |
| the same at TurnBench levels (real-floor windows) | 206 / 527 | +7.2 [+3.9, +10.5], −1.7 | **−3.0** [−5.7, −0.4], −0.1 | no |
| AppTek calls, 48 (en-AU / IE / ZA / IN) | 380 / 736 | +5.6 [+3.5, +7.6], −1.5 | **−3.0** [−4.7, −1.4], −0.8 | no |

**No source passes.** On every licensed real recording, at native level or scaled to TurnBench's levels, `Q`
interrupts more than v0.2 on `balanced` (by 6-15 points, not 34) and *less* on `fast` (TurnBench: 9 points more).

How the levels were varied: `level` scales a whole channel (speech, floor and crosstalk alike) so the user's speech
is drawn from −48…−34 dBFS; only windows with a real floor (median gap frame > −85 dBFS, i.e. not gated to digital
zero) are used. Nothing is synthesised. Labels: AMI = the manual words of all four participants through the floor
rule (turn_gap 0.5 s, max_hold 2 s, backchannels ≤ 1 s); oto 280 h = Silero VAD v5 on each raw channel (the 141 h
corpus's rule) minus the other party's bleed (a segment that overlaps the other party for ≥ 80 % of its length while
the other channel is ≥ 15 dB louder), then the floor rule; AppTek = the reference segments (one token each) through
the floor rule. Each window is 40 s around a user turn end; the TS-VAD print is 5 s of the user's own speech with the
other party silent, outside the window, same channel.

Channel statistics (speech = power mean over the user's speech; floor = median frame outside it; crosstalk =
correlation of user-channel and two-party-mix frame energy outside the user's speech):

| | speech dBFS p10 / p50 / p90 | floor dBFS p10 / p50 / p90 | crosstalk corr p50 |
|---|---|---|---|
| TurnBench user channels (not public test; audio statistics only) | −43.9 / −35.6 / −28.2 | −77.6 / −70.2 / −53.1 | 0.21 |
| AMI headsets, test meetings (never touched) | −47.1 / −37.5 / −22.0 | −80.4 / −71.8 / −43.3 | **0.76** |
| otoSpeech-280h raw (never touched) | −26.9 / −19.1 / −12.3 | −94.1 / −82.1 / −62.0 | 0.11 |
| AppTek calls (never touched) | −30.9 / −23.3 / −20.3 | digital zero on 60 % of gap frames (VoIP silence suppression) | 0.07 |

**Why (diagnosis on the never-touched parts and the cached TurnBench signals; nothing chosen on them).** False
interruptions by the path that fired:

| % of user turns | v0.2 balanced | v0.2 fast | Q balanced | Q fast |
|---|---|---|---|---|
| TurnBench (56) | model 5.4, fallback 7.1, others 1.8 | model 16.1, fallback 3.6, others 1.8 | **head 41.1**, fallback 5.4, others 1.8 | model 10.7, **fallback 17.9**, others 1.8 |
| AMI headsets at TurnBench levels (827) | model 10.5, fallback 5.3 | model 22.1, fallback 3.4 | head 29.5, fallback 1.5 | model 16.9, fallback 6.2 |
| oto 280 h at TurnBench levels (527) | model 6.8, fallback 15.6 | model 13.9, fallback 13.7 | head 23.7, fallback 5.9 | model 11.4, fallback 13.1 |
| AppTek calls (736) | model 10.2, fallback 5.3 | model 15.9, fallback 4.6 | head 18.3, fallback 2.7 | model 12.2, fallback 5.2 |

Share of in-turn quiet frames (energy 20 dB under the user's speech level, inside a labelled turn) with VAD < 0.4,
i.e. the time a silence clock can count:

| | v0.2 GRU VAD | Q stateless VAD |
|---|---:|---:|
| TurnBench | 0.20 | 0.55 |
| oto evaluation half | 0.27 | 0.71 |
| AMI headsets (test) | 0.18 | 0.29 |
| oto 280 h (never touched) | 0.38 | 0.69 |
| AppTek (never touched) | 0.33 | 0.58 |

- The **classifier path** (v5 at a quiet frame) fires at a similar rate on TurnBench and on the scopes (v0.2 fast:
  16 % of TurnBench turns, 14-22 % on the scopes; Q fast 11 % against 11-17 %).
- The **timer and head paths** do not transfer. On TurnBench, v0.2's GRU VAD stays at or above 0.4 through 80 % of
  the in-turn quiet frames, so its fallback rarely fires there (7 % / 4 % of turns), while Q's stateless VAD leaves
  the timers running (Q fast's 960 ms fallback: 18 %; Q balanced's block-12 head: 41 %). On the oto channels v0.2's
  own VAD is quiet in more of those frames, so its fallback fires (14-16 %) and Q looks no worse; in the AMI headsets
  the other participants' crosstalk keeps both VADs up.
- So what TurnBench adds is an interaction between its noise floor and the shipped GRU VAD (which stays up on it).
  None of the licensed recordings reproduces it, at native or TurnBench levels.
- Not tried: the OcularAI hi-fi two-speaker sample (CC BY 4.0) is gated with manual approval and not accessible to
  this account; CHiME-6 (97 GB, four-party dinner parties with in-ear microphones, i.e. the AMI headset case with more
  crosstalk) did not fit the day.

**Consequence for the design (§2).** A decider whose false interruptions come from the classifier path is predictable
from these scopes. One that relies on the timers running on a stateless VAD, or on the per-frame head, is not. And a
change to the conversational presets cannot be verified on held-out audio to keep v0.2's TurnBench false
interruptions, so those presets change only if a rule is no worse than v0.2 on every held-out scope.

## 2. Training with the new data, and what the selection rule allows

### 2.1 What was trained (teachers as before)

Training windows added (all from the train parts of §1.2; 3 349 windows of 40 s, ~37 h):
- AMI individual headsets, 45 train meetings: 703 windows; and a level-varied copy (whole channel scaled so the
  speech sits at −48…−34 dBFS): 696.
- otoSpeech-280h raw, 64 train sessions: 768 windows; level-varied copy of the windows with a real floor: 414; and a
  telephone-chain copy (G.711 μ-law 8 kHz, or 8 kHz Opus at 12 / 16 kbit/s, `voip` mode, through ffmpeg): 768.

Heads (block 12 of the frozen 0.6B encoder, the CORE_0P6B_TURN recipes, one thing changed: the data):
- **turn VAD `mlp12r`**: stateless MLP (65.7 K parameters), `mlp12oq_sa`'s recipe (AMI + SpecAugment views + oto +
  quiet oto + room tone) plus the real-channel windows at 25 % of the crops. Labels on the real channels: 1 = the
  user's words / VAD segments, 0 = nobody, and **no loss** where only the other party speaks (their crosstalk on the
  user's microphone is neither the user's speech nor silence). Early stopping on the held-out mean AUC (the old
  held-out sets + the AMI-dev and oto-280h held-out windows).
- **v5 classifier, two variants**, the c5 recipe with the 115M's v5 as the distillation teacher (`teach115`: the 115M's
  shipped c5 on its own served inputs of every new window), the in-utterance smart-turn negatives, the quiet-channel
  oto windows, and the real-channel windows:
  - `skd1stqr`: reads the served GRU VAD as its VAD channel (what `heads.vad` gives when served);
  - `kd1stqr`: reads the new stateless turn VAD as its VAD channel (`kd1stq` + the new data).
- The two-threshold silence clock (`reset_thr` 0.25) is in the rule grid. A short VAD hangover of H frames is the same
  rule as k + H and fallback + H (both clocks count from the last speech frame), so it is covered by the k and
  fallback values of the grid.
- New in the policy: the classifier's clock may run on the stateless turn VAD while arming, the gate, the fallback
  and the head path keep the GRU VAD (`VadHeadPolicy(vad_m=…)`, served as a preset's `model_clock: "turn_vad"`).
  §1.3 is the reason: the classifier path transfers from the scopes to TurnBench, the timers on a stateless VAD do
  not.

### 2.2 The selection rule (fixed before any candidate was scored)

All on held-out audio (core_0p6b_turn's held-out set: oto calls, quiet-channel oto, AMI, smart-turn st3; plus the
five real-channel held-out scopes: AMI dev headsets at native and TurnBench levels, oto-280h held-out at native and
TurnBench levels, AppTek held-out calls). Baseline = the served v0.2 / v0.3 turn rules on the same clips.
- `balanced` / `fast`: a rule ships only if its false interruptions AND missed ends are no worse than the baseline's on
  **every** scope (8 conversational scopes); among those, the fastest (calls + quiet-calls p50). If none passes, the
  preset keeps its v0.2 rule and classifier (bit-identical turn ends).
- `assistant`: among rules with held-out accuracy ≥ and false fires ≤ the baseline's, the fastest p50 (ties: more
  accurate, then fewer false fires).

### 2.3 Held-out results (selection only; no numbers from the selection audio are reported, per the standing rule)

Candidates scanned on every held-out scope (`turn_data.py hscan`: every rule of the k / VAD threshold / p /
fallback / two-threshold grid, per family; the v5 classifier variants × the VAD each clock reads). Only the outcome of
the selection rule is given here (the standing rule keeps selection-audio numbers out of this file; they are in
`runs/turn_data.json` `pick` for the record):

| candidate (tag) | VAD of arming / fallback | classifier's clock | classifier | balanced: a rule no worse than v0.2 on all 8 scopes? | fast: same? | assistant: accuracy ≥ and false fires ≤ v0.2? |
|---|---|---|---|---|---|---|
| `hs12` | served GRU | stateless (`mlp12oq_sa`) | shipped `s12` | none | none | not scanned |
| `hkq` | served GRU | stateless (`mlp12oq_sa`) | `kd1stq` (Q's) | none | none | not scanned |
| `hsk` | served GRU | GRU or stateless | `skd1st` (no new data) | none | none | yes |
| `hsr` | served GRU | GRU | **`skd1stqr`** (new data) | none | none | yes |
| `hkr` | served GRU | GRU or stateless (`mlp12r`) | **`kd1stqr`** (new data, reads `mlp12r`) | none | none | **yes: picked** |
| `hkq2` | stateless `mlp12r` (Q-style) | same | **`kd1stqr`** | none | none | yes |

- **balanced / fast: nothing passes.** Every rule that removes missed ends on the calls-like scopes adds false
  interruptions on at least one other scope, and every rule that keeps all eight scopes' false interruptions misses
  more ends somewhere. Retraining with the real channels did not change this. So `balanced` and `fast` keep v0.2's
  rules and classifier: their turn ends are bit-identical to v0.3 by construction.
- **assistant: picked `hkr`**: the `kd1stqr` classifier (reads the new stateless turn VAD `mlp12r` as its VAD
  channel), asked at 160 ms of energy-quiet audio (6 dB over the floor) or VAD < 0.4 on the GRU clock, P(complete)
  > 0.9, re-asked, fallback 3440 ms. All four candidates reach the same fastest held-out p50; `hkr` is the most
  accurate with the fewest false fires there, so the tie-breaks of §2.2 pick it. The second clock (`model_clock`) is
  not used by the pick.
- The served candidate equals its offline rule twin on held-out audio: 16 / 16 AppTek held-out windows per preset
  (`turn_data.py hocheck`).

## 3. Verified once on the evaluation sets, and what ships

Shipped: **`assets/served_heads_0p6b_v0.4.pt`** (`served_0p6b_v0.4.afm`, hub `HEADS_0P6B` 0.4, the default; 213
tensors, 27.0 MB, state hash `d6a7fafa…`, sha256 `2062496c…`). It is v0.3 plus two heads, both read by the
`assistant` preset only:
- `turn_vad` (`mlp12r`): the stateless block-12 VAD, 65.7 K parameters, the VAD channel of the classifier below;
- `turn_seg_a` (`kd1stqr`): the v5 classifier trained with the real-channel windows (2.59 M parameters).

The presets as the served model resolves them (`cfg["turn_presets"]`; a preset's `turn_model.head` names its own
classifier):

| preset | v0.3 | v0.4 |
|---|---|---|
| balanced | v5 `turn_seg` at ≥ 320 ms of VAD < 0.4, p > 0.6, re-asked; OR 720 ms | unchanged |
| fast | v5 `turn_seg` at ≥ 80 ms of VAD < 0.4, p > 0.5, re-asked; OR 720 ms | unchanged |
| assistant | v5 `turn_seg` at ≥ 160 ms of VAD < 0.5, p > 0.98, re-asked; OR 3440 ms | **v5 `turn_seg_a`** at ≥ 160 ms of energy-quiet (6 dB over the floor) or VAD < 0.4, p > 0.9, re-asked; OR 3440 ms |

The evaluation sets were touched once for this build (`final_compare.py eotdump` with `FINAL_0P6B_AFM` = v0.4, the
served `--mode single` engine on MPS; `turn_data.py evalv04` scores both builds with the served policy twin, 1000-
resample session bootstrap). One note on that run: the first pass of the dumps did not record the second
classifier's p (a recorder bug, fixed in `core_0p6b_heads._record` and the dump stages); nothing was scored from it,
and the dumps were made again on the same audio with the same engine.

**Public test rows** (smart-turn v3.2 test, 399 clips; AMI test turns, 200; 95 % CIs):

| row | 0.6B v0.3 (before) | **0.6B v0.4 (after, ships)** | goal |
|---|---|---|---|
| assistant on smart-turn test: accuracy | 93.0 % [90.5, 95.5] | **96.5 % [94.5, 98.2]** | ≥ 93 %: **met** |
| assistant: p50 / p95 ms | 379 [376, 408] / 702 | **351 [320, 383]** / 763 | ≤ 350 ms: **not met by 1 ms** (CI contains the bar) |
| assistant: false fires | 4.5 % [1.9, 7.2] | **3.1 % [0.9, 5.8]** | ≤ 5 %: **met** |
| assistant: missed (complete clips) | 9.7 % | 4.0 % | – |
| assistant on AMI test: p50 / interrupt / missed | 1619 / 4.0 / 53.0 | 1497 [1258, 1579] / 22.5 [17.0, 28.5] / 34.5 [28.0, 41.5] | – (the preset waits less, so it interrupts meetings more) |
| balanced on AMI test: p50 / interrupt / missed | 1497 / 10.0 [6.0, 14.5] / 33.5 [27.0, 40.0] | 1498 / 10.0 / 33.5 (same decisions) | no worse than v0.2: **met** (identical) |
| fast on AMI test | 1337 / 16.0 / 27.5 | 1338 / 16.0 / 27.5 (same decisions) | – |

**Calls (TurnBench dev + 16 oto conversations, 109 ends; not a public test split):**

| row | v0.3 | v0.4 | goal |
|---|---|---|---|
| balanced: p50 / FI / missed | 730 / 19.3 [12.1, 26.4] / 10.1 [4.9, 16.0] | 729 / 19.3 / 10.1 (same decisions) | missed ≤ 7.5 % at FI ≤ 20 %: FI kept, **missed not met** |
| fast | 494 / 26.6 / 11.0 | 492 / 26.6 / 11.0 (same decisions) | – |
| assistant | 3749 / 1.8 / 43.1 | 1866 / 9.2 [3.7, 16.0] / 35.8 [27.4, 44.1] | – |

`balanced` and `fast` make exactly the same decisions as v0.3: identical decision-time p50 / p95 on every set
(calls 686 / 1990 ms and 448 / 1930 ms, AMI 1456 / 4032 and 1296 / 4032); their totals differ by the measured chunk
compute of the two dump runs (median 41.7 vs 42.3 ms).

**Never-touched real-channel scopes** (unseen by any head or preset; the `assistant` preset, offline twin; paired
window bootstrap v0.4 − v0.3; `balanced` / `fast` are unchanged by construction):

| scope (user turns) | v0.3 p50 / interrupt / missed | v0.4 p50 / interrupt / missed | Δ interrupt, Δ missed (pp) |
|---|---|---|---|
| AMI headsets, test meetings (843) | 4081 / 0.4 / 73.2 | 891 / 18.7 / 45.1 | +18.3 [+16.2, +21.0], −28.1 [−31.4, −24.7] |
| the same at TurnBench levels (827) | 4091 / 0.6 / 74.6 | 811 / 19.3 / 46.1 | +18.7 [+16.2, +21.6], −28.5 [−31.7, −25.4] |
| otoSpeech-280h raw (958) | 3721 / 1.8 / 54.6 | 3581 / 4.8 / 47.8 | +3.0 [+1.9, +4.3], −6.8 [−8.7, −5.2] |
| the same at TurnBench levels (527) | 3671 / 1.1 / 53.1 | 3571 / 3.8 / 46.1 | +2.7 [+1.1, +4.1], −7.0 [−9.5, −4.8] |
| AppTek calls (736) | 3781 / 2.7 / 27.2 | 891 / 8.0 / 18.9 | +5.3 [+3.6, +7.0], −8.3 [−10.5, −6.1] |

The `assistant` preset is for speech aimed at an assistant; on human-to-human conversation it now answers sooner and
misses fewer ends, at the cost of more interruptions (most on the crosstalk-heavy AMI headsets).

**Served == offline:** 32 / 32 sessions per preset on the evaluation protocol (`core_0p6b_turn servedcheck --tag
v0.4 --n 16`: 16 assistant clips + 16 calls each), and 16 / 16 held-out AppTek windows per preset; `balanced` / `fast`
turn ends identical to v0.3 on 16 / 16 held-out windows each (`turn_data.py same`).

### 3.1 Latency gate (as in FIXALL.md)

Back to back on this Mac, v0.3 then v0.4, `core_0p6b_turn cost` (the mps_115m engine protocol: full single-mode
engine, bundled 16 s call with the 0.6B print, best of 3), per preset; streams = `final_compare cost --sub streams`.

| gate | v0.3 → v0.4 | pass? |
|---|---|---|
| engine ms per 160 ms chunk, MPS p50 (p95): balanced | 43.11 (48.97) → 43.50 (49.40): +0.39 | yes (≤ +1) |
| fast | 43.34 (49.17) → 43.76 (49.37): +0.42 | yes |
| assistant | 44.21 (50.44) → 44.79 (52.51): +0.58 | yes |
| CPU 2 threads p50, balanced (three back-to-back pairs) | 97.98 → 99.22 (+1.24); 97.98 → 98.27 (+0.29); 98.47 → 98.67 (+0.20); mean +0.58 | yes (the first pair is the run-to-run spread) |
| CPU p50, fast / assistant | 98.11 → 98.43 (+0.32) / 99.77 → 99.86 (+0.09) | yes |
| real-time streams, MPS (p95 of the summed block compute) | 3 (136.7 ms) → 3 (134.1 ms); 4 not real time for either | yes |
| end-of-turn decisions, balanced / fast | identical (decision times above; 16 / 16 held-out windows) | yes |
| end-of-turn latency, assistant | 379 → 351 ms p50 (decision + compute) | yes (faster) |
| other rows (WER, speech detector, TS-VAD, LID, speaker) | unchanged: those heads are bit-identical to v0.3 | yes |

Where the cost comes from: `turn_vad` (a 65.7 K MLP on block 12) runs on every chunk, and the `assistant` preset feeds
a second classifier stream (its classifier runs only when asked, instead of `turn_seg`). Peak RSS +21 MB.

## 4. In plain words

- **The data.** Real two-party channels with a real noise floor, level variation and crosstalk exist under licences
  that allow commercial training: the AMI individual headsets (CC BY 4.0) and the raw otoSpeech-280h channels
  (CC BY 4.0); the AppTek call-center calls (CC BY-SA 4.0) may be used for evaluation only. Used: 45 + 18 + 16 AMI
  meetings (train / held-out / never touched; 103.6 / 38.7 / 35.4 h of headset channels), 64 + 32 + 32 oto sessions
  (16.4 / 7.8 / 8.3 h of two-channel sessions), 48 + 48 AppTek calls; 37 h of 40 s training windows (with
  level-varied and telephone-codec copies), and 4-5 h of windows per held-out and never-touched scope.
- **Did the new held-out scope predict TurnBench? No.** On every source, at native level and scaled to TurnBench's
  levels, `Q` interrupts more than v0.2 on `balanced` (by 6-15 points instead of 34) and less on `fast` (TurnBench:
  9 points more). The reason is an interaction between TurnBench's noise floor and the shipped GRU VAD, which stays
  up there and so protects its pauses; none of the licensed recordings reproduces it. So no change to `balanced` or
  `fast` can be verified to keep v0.2's false interruptions on calls, and none passed the held-out rule (no worse on
  all eight scopes). They stay as they are; the calls-misses goal (≤ 7.5 %) is not met (10.1 %).
- **What did ship.** The `assistant` preset gets its own classifier trained with the new data: on the smart-turn
  test set 93.0 → 96.5 % accuracy, 4.5 → 3.1 % false fires, 379 → 351 ms (the ≤ 350 ms goal missed by 1 ms, inside the
  CI). Cost: +0.4-0.6 ms per chunk.

## Reproduce

```bash
TD=scripts/research/turn_data.py; T6=scripts/research/core_0p6b_turn.py; G="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python"
# data (downloads: data/ami_ihm/fetch.sh MEETINGS..., otoSpeech-280h shards 0-23, AppTek 8 locales x 12 calls)
$G $TD man --split dev; $G $TD man --split train --per 4; $G $TD man --split test
$G $TD oman --split dev|train|test (--shard k/n for the Silero label cache); $G $TD aman --split atdev|attest --per 4
$G $TD level --src-split odev --split odevq   # + devq, otestq, testq, trainq, otrainq;  $G $TD codec --src-split otrain --split otrainc
$G $TD splits; $G $TD stats --split test,otest,attest --n 0
$G $TD feats --split S (every split); $G $TD p6 --split S --vad served --seg s12 --turn f1; ... --vad mlp12oq_sa --seg kd1stq --turn b12
$G $TD test --split S; $G $TD diag --split ...                                 # the test of the data
# training (teachers as before) and selection
$G $TD teach115 --split S (train splits)
$G $T6 vadtrain --tag mlp12r --arch mlp --vblocks 12 --oto 1200 --otoq 1200 --sa --ihm 1200 --ihm-splits train,otrain,trainq,otrainq,otrainc --ihm-share 0.25 --ihm-val dev,odev
$G $T6 vadall --vad mlp12r
$G $T6 segtrain --tag kd1stqr --vad mlp12r --block 12 --kd 1 --st3dips 0.5 --quiet-train --ihm-train train,otrain,trainq,otrainq,otrainc  # x4
$G $T6 segtrain --tag skd1stqr --vad served ... (same)                          # x4
$G $T6 hop6 --vad mlp12r --seg kd1stqr --turn f1; $G $TD p6 --split S --vad mlp12r --seg kd1stqr --turn f1   (held-out splits)
$G $TD hscan --tag hkr --base served__s12__f1 --clock mlp12r__kd1stqr__f1 --seg mlp12r__kd1stqr__f1 --both-clocks --fam assistant,balanced,fast
$G $TD pick --tags hs12,hkq,hsk,hsr,hkr,hkq2 --fam assistant,balanced,fast
# build, check, verify once
$G $T6 build --tag v0.4 --ship --version 0.4 --base .../served_0p6b_v0.3.afm --turn-vad mlp12r --seg-a kd1stqr --seg-a-vad-input turn_vad --presets-json '<v0.4 presets>'
$G $TD hocheck --afm .../served_0p6b_v0.4.afm --split atdev --n 16; $G $TD same --afm .../served_0p6b_v0.4.afm --split atdev --n 16 --fam balanced,fast
EOT_AMI_SPLIT=eval FINAL_0P6B_AFM=.../served_0p6b_v0.4.afm FINAL_COMPARE_W=.../turndata/fc_v04 $G scripts/research/final_compare.py eotdump --sys 0p6b --which asst|calls
EOT_AMI_SPLIT=eval $G $TD evalv04; $G $T6 servedcheck --tag v0.4 --n 16 --fam assistant|balanced|fast
$G $T6 cost --tags v0.3,v0.4 --fam balanced,fast,assistant --device mps|cpu
```
