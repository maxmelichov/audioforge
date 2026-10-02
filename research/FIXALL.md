# Fix all: every row where audioforge loses, attacked head by head (running log)

Started 2026-10-01 21:00. Brief: the user, after reading research/FINAL_COMPARE.md: "Fix all please". Every row where
audioforge loses to the best model in FINAL_COMPARE is attacked on both cores (115M heads v0.3, English 0.6B heads
v0.2). The NVIDIA encoders stay frozen. Script: `scripts/research/fixall.py`. Numbers: `runs/fixall.json`. Scratch:
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/fixall/`. Mac only, MPS, one heavy job at a time through
`scripts/dev/gate.sh`.

Rules kept throughout:
- Selection on held-out data only. FINAL_COMPARE's evaluation sets are touched once per shipped candidate, through
  `scripts/research/final_compare.py` on the same audio.
- A head ships only if it beats the served head on held-out data and does not regress any other served row.
- Teachers: the best model of each row, run on our own training audio, gives soft targets next to the hard labels
  (user instruction "use the best model as teacher"). Teacher outputs are cached once under `scratch/fixall/teachers/`.

## Scoreboard

| row | grey bar (best other model) | before (115M / 0.6B) | after (115M / 0.6B) | shipped? |
|---|---|---|---|---|
| speech detector, ICSI F1 | 0.935 (TEN VAD) | 0.898 / 0.906 | **0.947 / 0.951** | yes (115M heads v0.4, 0.6B v0.3) |
| speech detector, AMI F1 (must stay) | ≥ 0.951 | 0.9511 / 0.9507 | 0.9505 / 0.9500 (paired Δ within CI) | yes (tie, see step 1) |
| turn, 0.6B assistant p50 / acc / FF | ≤ 300 ms, ≥ 93 %, ≤ 5 % | 379 ms / 93.0 % / 4.5 % | unchanged (held-out: a better rule exists only at the same p50) | no |
| turn, 0.6B balanced calls missed at ≤ 20 % FI | ≤ 7.3 % | 10.1 % at 19.3 % | unchanged (no candidate is no-worse than v0.2 on held-out at its speed) | no |
| turn, 0.6B fast calls p50 / FI / missed | ≤ 547 / 25 / 5.5 | 494 / 26.6 / 11.0 | unchanged (0 of 2160 rules no-worse than v0.2 on held-out) | no |
| turn, 115M fast / assistant, constants picked on held-out | (honesty check) | shipped (tuned on eval): 547 / 24.8 / 5.5; 92.7 % / 299 ms / 5.4 % | held-out-picked: 589 / 30.3 / 8.3; 95.0 % / 354 ms / 3.1 % | report only |
| language ID, 2 s / full | 95.9 / 99.5 % (Whisper large-v3) | 91.0 / 97.8 · 87.6 / 95.5 | | |
| speaker EER, AMI / ICSI | 12.0 / 2.1 % (TitaNet-L) | 19.8 / 7.0 · 13.6 / 3.6 | | |
| words, ICSI WER | 10.4 % (Parakeet-TDT v3) | 26.2 / 13.6 | | |

## Data and teacher licences

| item | licence | use here |
|---|---|---|
| AMI meeting corpus (train meetings) | CC BY 4.0 | VAD training windows (as before) |
| ICSI meeting corpus (12 cached train meetings; dev Bmr021 / Bns001 and eval never used) | CC BY 4.0 (`data/icsi/annotations/LICENCE.txt`) | new: VAD training windows; held-out selection on Bro026 / Bmr022 |
| otoSpeech full-duplex 141 h (user channels) | CC BY 4.0; the card forbids speaker identification | VAD / turn training (no speaker identity, as the rules require) |
| TEN VAD 1.0.6.8 (teacher) | Apache-2.0 with additional conditions | soft VAD targets on our training audio |
| pyannote segmentation-3.0 (teacher) | MIT | soft VAD targets (offline, 10 s windows; fine for targets) |
| smart-turn v3.2 (teacher, already in the v5 recipe) | BSD-2-Clause | turn classifier targets |
| Whisper large-v3 (teacher) | MIT | language posteriors |
| AmberNet (teacher) | CC BY 4.0 | language posteriors |
| TitaNet-L (teacher) | CC BY 4.0 | speaker embeddings |
| WeSpeaker ResNet34 via pyannote (teacher) | MIT | speaker embeddings |
| Parakeet-TDT 0.6B v3 | CC BY 4.0 | reference transcripts for decoder-side selection only (held-out audio) |

## Log

### 1. Speech detector (shipped: 115M heads v0.4, 0.6B heads v0.3)

**Data.** Window sets shared by both cores (`vman`, `VSet`):
- `ami1200`: the shipped recipe's 1200 seeded AMI train diar windows (20 s); held out for selection: TS3011b, ES2015c.
- `icsi600`: 600 seeded ICSI diar windows of the 12 cached train meetings (~10 h on disk). Held out for selection:
  Bro026, Bmr022 (110 windows). The FINAL_COMPARE ICSI rows use the dev meetings Bmr021 / Bns001, never touched here.
- `oto_tr` / `otoq_tr`: 1200 oto user-channel turn clips (40 s) of the train conversations, clean and as quiet noisy
  channels (CORE_0P6B_TURN's `quiet_audio`); `oto_va` / `otoq_va`: the 224 held-out conversations' clips.
- `room_tr` / `room_ho`: the noise-only negatives of the shipped recipe.
- Teachers (`vteach`, cached in `scratch/fixall/teachers/vad_*.npz`): TEN VAD (16 ms hop) and pyannote
  segmentation-3.0 (offline 10 s windows) on every window, max-pooled to the 80 ms grid exactly as FINAL_COMPARE
  scores them. 25 min on this Mac for 5 528 windows.
- Features (`vfeat`): the frozen encoders' masked [70,1] forward (= 160 ms streaming), 115M blocks 2-6, 0.6B blocks
  8/10/12/14/16, plus one SpecAugment view of the AMI and ICSI windows (the shipped recipe's augmentation).

**Heads.** Stateless, the served FrameHead shape (Linear(D, 64)-SiLU-Linear(64, 1)), optional learned block mix;
crops of 32 frames, shares AMI 0.35 / ICSI 0.25 / oto 0.15 / quiet oto 0.10 / room tone 0.15; feature dropout,
noise, time masks; 3000 steps; early stopping on the held-out mean AUC (`sel`: AMI-held, ICSI-held, oto-held,
quiet-oto-held, each with held-out room tone as extra negatives). Then a logit shift calibrated on held-out (the
shift maximising mean F1 at 0.5 over AMI-held and ICSI-held), folded into the output bias (`--calib`).

**Held-out selection (115M; served head first; F1 at 0.5 / AUC; nothing here touched the evaluation sets):**

| head | AMI-held F1 / AUC | ICSI-held F1 / AUC | oto-held F1 | quiet-oto F1 | room p95 | sel |
|---|---|---|---|---|---|---|
| served v0.3 `vad` (block 4, AMI only) | 0.966 / 0.989 | 0.903 / 0.982 | 0.924 | 0.902 | 0.394 | 0.984 |
| b4, teacher mean(TEN, pyannote), λ 1 | 0.962 / 0.988 | 0.948 / 0.989 | 0.963 | 0.953 | 0.201 | 0.989 |
| b4, hard labels only | 0.960 / 0.987 | 0.955 / 0.990 | 0.964 | 0.958 | 0.098 | 0.990 |
| b4, teacher pyannote | 0.963 / 0.988 | 0.934 / 0.989 | 0.962 | 0.948 | 0.175 | 0.989 |
| b4, teacher TEN | 0.956 / 0.987 | 0.953 / 0.989 | 0.963 | 0.953 | 0.222 | 0.989 |
| b4, teacher pyannote λ 0.3 | 0.962 / 0.988 | 0.952 / 0.990 | 0.964 | 0.956 | 0.132 | 0.990 |
| mix 2-6, teacher mean | 0.962 / 0.989 | 0.948 / 0.990 | 0.964 | 0.953 | 0.209 | 0.990 |
| mix 2-6, hard only, + SpecAugment view | 0.959 / 0.988 | 0.957 / 0.991 | 0.963 | 0.957 | 0.097 | **0.9904** |
| same, hidden 256 | 0.959 / 0.988 | 0.957 / 0.991 | 0.964 | 0.958 | 0.079 | 0.9906 |
| **chosen: mix 2-6, hard, SA view, calibrated (+0.7 logit)** | 0.964 / 0.988 | 0.955 / 0.991 | 0.963 | 0.957 | 0.177 | 0.9904 |

The served head shape was kept (hidden 256 is +0.0002 `sel`, under the noise). **The teachers did not help on
held-out**: every teacher variant is at or below hard labels alone on `sel` and on ICSI-held F1. TEN VAD's posterior
on AMI disagrees with the AMI "any part of the frame" labels (AMI-held F1 0.956 with the TEN target), and
pyannote's costs ICSI. The ICSI gain comes from the ICSI train meetings themselves (a probe with only 115 ICSI
windows moved ICSI-held F1 0.896 → 0.949).

0.6B (served first): served v0.2 GRU 0.966 / 0.989 · 0.909 / 0.982 · oto 0.922 · quiet 0.901 · room 0.215 · sel 0.983.
Block 12, hard, SA, calibrated: 0.965 / 0.989 · 0.956 / 0.991 · 0.964 · 0.957 · 0.175 · 0.9911. **Chosen, mix of
blocks 8-16, hard, SA, calibrated (+0.4):** 0.963 / 0.989 · 0.959 / 0.992 · 0.966 · 0.957 · 0.123 · **0.9915**. (The
other queued 0.6B variants were stopped: each took ~17 min of memory-mapped reads and the 115M sweep had already
answered the teacher and block-mix questions.)

**How it is served.** The new head is `heads.speech`. It is the speech probability the client sees (the frame
message's `vad` field, `final_compare`'s VAD rows). The turn rules, the TS-VAD arming, the LID gating and the v5
classifier keep reading the old `heads.vad`, so the turn rows cannot change. This was chosen for two reasons:
- the 115M's turn constants and its v5 classifier were tuned with the old head's hangover;
- CORE_0P6B_TURN.md showed that a VAD without that hangover exposes the TurnBench pauses to the timers.
Step 2 tries the new VAD as the 0.6B's turn trigger separately. Serving change: `ASRStream` adds `speech` to its
frame records when the model has the head (`tests/test_fixall.py`: the stream equals the masked forward, and `vad` is
unchanged). Served check (`servedeq`, bundled call, every preset, served v0.3 / v0.2 vs v0.4 / v0.3): identical
turn_end events on both cores; the client's `vad` differs on 183 / 201 (115M) and 200 / 200 (0.6B) frames.

**Evaluation (touched once, `final_compare.py vad`, same 64 + 64 windows; 95 % window bootstrap):**

| | AMI F1 | AMI AUC | AMI miss @7.5 % FA | ICSI F1 | ICSI AUC | ICSI miss | onset lag p50 / p90 AMI |
|---|---|---|---|---|---|---|---|
| TEN VAD (bar) | 0.925 | 0.948 | 13.6 % | 0.935 [0.929, 0.942] | 0.952 | 14.0 % | 0 / 80 ms |
| 115M before (v0.3) | 0.9511 | 0.972 | 10.6 % | 0.898 | 0.931 | 19.8 % | 80 / 160 ms |
| **115M after (v0.4)** | 0.9505 [0.939, 0.960] | 0.970 | 12.1 % | **0.947** [0.942, 0.952] | 0.964 | 10.7 % | 0 / 160 ms |
| 0.6B before (v0.2) | 0.9507 | 0.972 | 10.7 % | 0.906 | 0.925 | 22.4 % | 80 / 160 ms |
| **0.6B after (v0.3)** | 0.9500 [0.938, 0.959] | 0.969 | 12.5 % | **0.951** [0.946, 0.955] | 0.966 | 10.2 % | 80 / 160 ms |

- ICSI: both cores now beat TEN VAD. The paired F1 difference against TEN VAD is +0.009 to +0.015 (115M) and
  +0.011 to +0.019 (0.6B). The gain over the old heads is +0.045 to +0.055 (115M) and +0.040 to +0.050 (0.6B).
- AMI: paired F1 against the old head is −0.0006 [−0.0033, +0.0019] (115M) and −0.0007 [−0.0048, +0.0035] (0.6B).
  AUC is −0.002 / −0.003 and miss at 7.5 % false alarms +1.5 / +1.8 points.
  - So the "AMI stays ≥ 0.951" bar is a statistical tie: 0.9505 and 0.9500 at four decimals, not a strict pass.
  - It shipped because the held-out AMI difference was equally small (−0.002 F1) and the held-out ICSI / phone-channel
    gains were large. The held-out data predicted this outcome; nothing was re-tuned after the evaluation.
- Held-out room tone p95 (false speech on noise only): 115M 0.39 → 0.18, 0.6B 0.21 → 0.12.
- ICSI is no longer an unseen corpus for us: other ICSI meetings are in training.
- Turn rows: unchanged by construction (served check above). Cost: one more 33 K / 66 K-parameter frame head on
  blocks the encoder already computes (not re-measured separately; it is re-measured with the final cost row).

Shipped: `assets/served_heads_v0.4.pt` (115M, hub `HEADS` 0.4 default; `stage1_served_v4.afm`) and
`assets/served_heads_0p6b_v0.3.pt` (hub `HEADS_0P6B` 0.3 default; `served_0p6b_v0.3.afm`), both rebuilt from the NVIDIA
`.nemo` + heads file with a matching tensor hash; FINAL_COMPARE VAD rows, `runs/final_compare.json`, `numbers_final`
and `compare_vad.png` updated.

### 2. Turn taking (0.6B; nothing shipped; the 115M's constants checked on held-out)

**What was tried.** Each candidate was scored on the held-out end-of-turn set of CORE_0P6B_TURN.md (oto calls,
quiet-channel oto, AMI, smart-turn st3 clips; never an evaluation set):
- **Two thresholds on the silence clock** (`VadHeadPolicy(reset_thr=…)`, served as a preset's `reset_thr`; default off).
  A frame with VAD between `reset_thr` and the speech threshold does not arm a turn, but it restarts both silence
  clocks. So a pause the VAD is unsure about, as on a noisy channel, is not counted as silence. Grid: reset_thr ∈
  {off, 0.15, 0.25} on top of core_0p6b_turn's grids: 2 736 balanced, 2 160 fast and 1 728 assistant rules per
  candidate.
  - A short hangover (counting a frame as speech for H more frames) is the same rule as k + H / fallback + H, which
    are already in the grid, so it was not added separately.
- **Candidates:**
  - `Q`: VAD mlp12oq_sa + v5 kd1stq + block-12 turn head (CORE_0P6B_TURN).
  - `fx_b12`: the same with step 1's recipe as the VAD (block 12, AMI + ICSI + oto, hard labels, calibrated; block 12
    because only blocks 8 / 12 / 24 are cached for the turn clips).
  - `served`: the v0.2 heads themselves, to see whether the new rule alone helps.
- **Selection rule (new): no worse than the served v0.2 on every held-out scope.** For fast / balanced that means FI
  and missed on calls, quiet calls and AMI; for assistant, accuracy, false fires and p50. Among those rules, the
  fastest wins (assistant: the most accurate). CORE_0P6B_TURN picked against the 115M's held-out numbers instead.
  That let `Q` through with held-out calls FI 28.1 %, where v0.2 has 21.7 %, and Q then interrupted 41 % of
  TurnBench turns. This rule would have rejected Q.

**Held-out result** (calls p50 / FI / missed · quiet · AMI; assistant accuracy / p50 / FF):

| preset | v0.2 served (bar) | Q, best no-worse rule | fx_b12, best no-worse rule | v0.2 heads + reset_thr |
|---|---|---|---|---|
| balanced | 721 / 21.7 / 8.0 · 961 / 24.1 / 8.0 · 1401 / 10.5 / 40.8 | 21 of 2 736 pass; fastest 1041 / 21.3 / 7.8 · 1031 / 24.1 / 6.7 (320 ms slower) | 4 pass; fastest 1051 / 20.7 / 7.2 (330 ms slower) | the shipped rule is the only best one |
| fast | 466 / 29.1 / 5.9 · 891 / 28.7 / 7.8 · 1161 / 13.2 / 31.6 | **0 of 2 160 pass** | **0 pass** | the shipped rule |
| assistant | 95.7 % / 441 / 2.8 % | 180 pass; best 97.8 % / 441 / 1.4 % (reset_thr 0.15) | 560 pass; best 98.2 % / 441 / 1.4 % (reset_thr 0.25) | the shipped rule |

Why nothing shipped:
- **balanced / fast.** The stateless VADs (Q's and step 1's) remove the calls misses, as CORE_0P6B_TURN showed. But
  at v0.2's speed no rule keeps the false interruptions at v0.2's level on all three held-out scopes. The rules that
  do keep them are 300+ ms slower. So the calls-misses bar (≤ 7.3 % at ≤ 20 % FI) and the fast bar cannot be reached
  by a candidate that passes the no-regression rule.
- **assistant.** Q / fx_b12 with reset_thr beat v0.2 on held-out accuracy and false fires at the same held-out p50
  (441 ms). No rule reaches below 361 ms on the held-out st3 clips; the 115M's own held-out p50 is 350 ms. So the
  ≤ 300 ms bar is not reachable under held-out selection. Shipping the accuracy / false-fire gain would also need a
  per-preset VAD and v5 classifier (assistant on the new heads, balanced / fast on v0.2's), because the same heads
  lose on fast / balanced. That serving change was not made for a row whose accuracy and false fires already meet
  the bar. Not run on the evaluation sets (nothing to ship).
- **The two-threshold rule** is in the served policy (off by default; `tests/test_fixall.py`). It is picked on
  held-out for the new heads' assistant preset, and not for v0.2.

**The 115M's fast / assistant constants, picked on held-out only** (CORE_0P6B_TURN.md §2.4, same procedure, reused
from `runs/core_0p6b_turn.json` evverify `115m_heldout_tuned`; evaluation sets, 95 % session CIs):

| preset | shipped constants (tuned on the evaluation sets) | constants picked on held-out |
|---|---|---|
| fast, calls p50 / FI / missed | 547 / 24.8 / 5.5 | 589 [540, 663] / 30.3 [21.8, 40.0] / 8.3 [3.2, 14.4] |
| assistant, accuracy / p50 / FF | 92.7 % / 299 ms / 5.4 % | 95.0 [93.0, 97.0] % / 354 [322, 370] ms / 3.1 [1.1, 5.7] % |
| balanced | 956 / 20.2 / 7.3 | identical (held-out picks the shipped rule) |

The FINAL_COMPARE grey bars that come from the 115M's fast and assistant rows were set with constants tuned on the
evaluation sets. The honest held-out numbers are the right-hand column. The shipped 115M constants stay as they are,
since changing them now would be a choice made on evaluation data either way.
