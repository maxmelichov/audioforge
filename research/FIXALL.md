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

Grey bars and "before" values are FINAL_COMPARE's first pass (AMI / ICSI **dev** splits). "After" values are on the
public **test** splits of the 2026-10-02 audit (research/FINAL_COMPARE.md), next to the same baseline re-run on the
same test audio (in brackets).

| row | grey bar (first pass, dev) | before (115M / 0.6B, dev) | after (115M / 0.6B, **test**) [best baseline on test] | shipped? |
|---|---|---|---|---|
| speech detector, ICSI F1 | 0.935 (TEN VAD) | 0.898 / 0.906 | **0.938 / 0.940** [Silero 0.922, TEN 0.920] | yes: `speech` head, 115M heads v0.4, 0.6B v0.3 |
| speech detector, AMI F1 | ≥ 0.951 (must stay) | 0.951 / 0.951 | **0.959 / 0.957** [MarbleNet 0.941; pyannote 0.975 offline] | yes (dev tie, test +0.002 / −0.001 vs the old heads) |
| turn, 0.6B assistant p50 / acc / FF | ≤ 300 ms, ≥ 93 %, ≤ 5 % | 379 ms / 93.0 % / 4.5 % | unchanged: 379 / 93.0 / 4.5 (smart-turn test) | no (held-out: only equal-p50 gains) |
| turn, 0.6B balanced calls missed at ≤ 20 % FI | ≤ 7.3 % | 10.1 % at 19.3 % | unchanged (calls have no labelled test split) | no |
| turn, 0.6B fast calls p50 / FI / missed | ≤ 547 / 25 / 5.5 | 494 / 26.6 / 11.0 | unchanged | no |
| turn, 115M without the second encoder pass | −12 ms / chunk | – | 0 of 2160 held-out rules keep the AMI misses | no |
| language ID, 2 s / full | 95.9 / 99.5 % (Whisper large-v3) | 91.0 / 97.8 · 87.6 / 95.5 | **92.4 / 98.2 · 92.7 / 98.6** [Whisper large-v3 95.9 / 99.5] | yes: LID head v2 (both cores), bar not reached |
| speaker EER, AMI / ICSI | 12.0 / 2.1 % (TitaNet-L, dev) | 19.8 / 7.0 · 13.6 / 3.6 | test: 5.0 / 2.5 · 3.8 / 1.9 [TitaNet-L 1.9 / 1.2; WeSpeaker 2.0 / 0.9] (heads unchanged) | no (clean held-out probe: no TitaNet parity) |
| words, ICSI WER | 10.4 % (Parakeet-TDT v3, dev) | 26.2 / 13.6 | test: 18.4 / 10.3 [TDT 7.6]; 115M `--beam 8` 18.0 | `--beam K` option (115M); bar not reached |
| words, AMI WER | – | 20.6 / 10.4 | test: 16.1 / **7.9** [TDT 8.3]; 115M `--beam 8` 14.6 | option |

## Head architectures: which proven component each candidate is lifted from

Every candidate is a documented part of a published model placed on our frozen features. None is a new architecture.
Survey: research/HEAD_ARCHITECTURES.md (§1 VAD, §2 speaker / TS-VAD, §3 LID, §4 end of turn). ms = change in served
engine compute per 160 ms chunk (MPS p50, full single-mode engine, back to back with the served heads).

| head / candidate | architecture lifted from (paper / repo, which part) | params | ms / chunk | shipped? |
|---|---|---|---|---|
| speech detector `speech` (115M blocks 2-6, 0.6B blocks 8-16) | MarbleNet frame-VAD's decoder (Jia et al. 2021; NeMo `ConvASRDecoder`: a 1×1 conv = a per-frame classifier; here 2 layers, Linear-SiLU-Linear) on a learned softmax layer mix (ELMo / SUPERB weighted sum, Peters et al. 2018, Yang et al. 2021) | 33 K / 66 K | +0.39 / −0.07 | yes |
| two-threshold silence clock (`reset_thr`) | Silero VAD's `VADIterator` hysteresis (speech above `threshold`, silence only below `threshold − 0.15`) and WebRTC-style hangover, on our VAD posterior | 0 | 0 | no (not picked for the shipped heads) |
| turn classifier v5 (`turn_seg`; Q / fx_b12 candidates) | smart-turn v3's design (Pipecat): one classifier over the last 8 s of audio features with attention pooling and a linear output (TURN_V5.md) | 2.46 M | unchanged | unchanged |
| speaker head (probe) | TitaNet-L's / ECAPA-TDNN's attentive statistics pooling (Koluguri et al. 2022; Desplanques et al. 2020) + x-vector style projection with BN (one layer, or two with `hidden`) and AAM-softmax (Deng et al. 2019) | 0.5 M / 1.0 M | – | no |
| LID head (hidden 1024) | AmberNet's decoder (Jia et al. 2023: attentive statistics pooling + linear classifier, from TitaNet), causal running sums, MLP classifier | 2.37 M / 2.89 M | −0.03 / −0.20 | see step 3 |
| `--beam 8` | ALSD / modified beam search (Saon et al. 2020; icefall `modified_beam_search`): ≤ 3 expansions per frame, prefix merging, adaptive pruning | 0 | +1.5 (CPU) | option |
| TS-VAD (unchanged) | TS-VAD's speaker-conditioned frame tracker (Medennikov et al. 2020), causal | – | – | unchanged |

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

**Latency gate (added 2026-10-02 at the user's request; applied to this ship retroactively).** All numbers were
measured back to back on the same Mac, old heads then new, with the protocol of FINAL_COMPARE's cost row
(`final_compare.py cost`): the full single-mode engine, the bundled call, best of 3. The streams column uses K
interleaved sessions, real time while p95 of the summed block compute stays under 160 ms.

| gate | 115M v0.3 → v0.4 | 0.6B v0.2 → v0.3 | pass? |
|---|---|---|---|
| engine ms per 160 ms chunk, MPS p50 (p95) | 28.23 (29.74) → 28.62 (30.66): +0.39 | 42.96 (47.87) → 42.89 (48.95): −0.07 | yes (≤ +1 ms) |
| engine ms per chunk, CPU 2 threads p50 (p95) | 30.34 (32.70) → 30.35 (33.00): +0.01 | 98.69 (106.52) → 97.08 (105.06): −1.6 | yes |
| real-time streams, MPS / CPU | FINAL_COMPARE 4 / 4 → 5 / 4 (5 MPS streams: p95 151 ms) | 3 / 1 → 3 / 1 | yes (none dropped) |
| end-of-turn decisions (bundled call, every preset) | identical turn_end times | identical | yes |
| end-of-turn latency | = decision time + chunk compute: changes by the compute above (< 0.4 ms) | idem | yes |
| speech-detector onset lag p50 / p90, AMI dev | 80 / 160 → 0 / 160 ms | 80 / 160 → 80 / 160 | yes |
| onset lag, ICSI dev | 0 / 80 → 0 / 80 | 0 / 80 → 0 / 80 | yes |
| streaming word latency p50 / p95 (12 AMI test windows, MPS) | 273 / 570 → 274 / 571 ms (CI [250, 291]) | 290 / 590 → 291 / 591 ms (CI [272, 311]) | yes, within noise: the same words come out; the +1 ms is the run-to-run spread of the block compute (0.6B block p50 39.5 → 41.0 ms in this run, against −0.07 ms in the back-to-back engine run) |

So the block mix costs nothing measurable: the encoder computes every block anyway, and the head is a 33 K / 66 K
parameter MLP on a weighted sum of 5 block outputs. No head had to be replaced.

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

**115M without its second encoder pass (user request, speed).** The 115M's `balanced` preset reads the per-frame turn head, which
runs on a speaker-conditioned second encoder pass (~12 ms per chunk). The v5 classifier reads only first-pass
signals. If it could decide `balanced`, the second pass could go. All 2 160 model-mode rules (reset_thr included) were
scanned on the held-out set (`tscan --tags 115m`, `tpick115`) against the shipped rule (held-out calls 900 ms / 28.5 % /
6.5 %, quiet 750 / 39.3 / 7.2, AMI 1230 / 18.4 / 27.6). The criterion was no worse FI and missed on every scope and no
slower p50.
- **0 rules pass, and none even without the speed condition.** The closest keeps calls and quiet calls (690 ms / 28.4 % /
  6.5 %; 750 / 39.3 / 7.2) and is faster everywhere. But it misses one more of the 76 held-out AMI turns (missed 28.9 vs
  27.6 %).
- So the second pass stays. Under the no-regression rule the per-frame speaker-conditioned head still earns its 12 ms on
  meetings.

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

### 4. Speaker head (probe; nothing shipped, so voice prints and TS-VAD are unchanged)

**Teachers.** TitaNet-L (spk_frame's cached crop and whole-item embeddings) and, new, WeSpeaker ResNet34 (pyannote,
the best within-meeting EER in FINAL_COMPARE) on the same 11 133 items and 27 227 crops of spk_frame's training sources
(LibriSpeech train-clean-100's 251 speakers, AMI train, ICSI train). `steach`, 6.5 min on MPS,
`scratch/fixall/teachers/spk_wespeaker_*.npz`. No further licensed multi-speaker data with speaker labels is on disk:
oto may not be used for identity, and DailyTalk / Behavior-SD have two voices per conversation.

**Clean held-out set.** The shipped heads were trained on every AMI / ICSI train meeting, so they cannot be compared
fairly on any held-out train meeting. The probe holds out AMI TS3011b / ES2015c (142 items) and ICSI Bro026 / Bmr022
(572 items) from training. Within-meeting EER is measured on those items for the candidates and for the teachers
(`strain`).

| embedder (held-out meetings) | AMI EER % | ICSI EER % |
|---|---|---|
| TitaNet-L | 11.5 | 1.9 |
| WeSpeaker ResNet34 | 10.9 | 1.6 |
| 0.6B head, joint TitaNet + WeSpeaker, shipped projection (block 5) | 17.9 | 1.9 |
| 0.6B head, joint, 2-layer projection (hidden 512) | 17.6 | 1.9 |
| 115M head, joint, 2-layer projection (block 4) | 22.1 | 2.2 |
| *shipped heads (these meetings were in their training data, so not comparable)* | *6.1 (0.6B) / 15.3 (115M)* | *1.2 / 1.4* |

On meetings no head has seen, the AMI gap to TitaNet-L is 6-10 points, which is the same picture as FINAL_COMPARE's
dev meetings (13.6 vs 12.0 there). ICSI is a tie. A bigger projection and the WeSpeaker target each move held-out
EER by less than half a point. The shipped heads' low numbers on these meetings show what in-domain speaker
training buys: the heads memorise the training meetings' voices. So the bar (beat TitaNet-L on both corpora) is not
reachable with block-4 / block-5 features of the frozen encoders and the licensed speakers on disk.

Not done, because nothing ships: the TS-VAD retrain on new prints, the print version tag and the re-enrol message.
The SpeakerHead `hidden` option (two-layer projection) is in the code, off by default.

### 5. Words: RNNT beam search (decoding only; the encoders stay frozen)

**What.** `RNNTHead.beam_search` / `BeamTransducerStream` (audioforge/heads/asr.py): a transducer beam search with up
to 3 non-blank expansions per 80 ms frame, batched over the live hypotheses, merging identical token sequences
(log-sum-exp). Expansions that cannot beat the beam-th hypothesis already finished on the frame are pruned. This is
the adaptive-expansion idea of ALSD / modified beam search (Saon et al. 2020; icefall `modified_beam_search`), lifted
as a decoding component; nothing is trained.

**Held-out selection** (`wdec` / `wscore`; single-speaker segments of the held-out AMI / ICSI train meetings TS3011b,
ES2015c, Bro026, Bmr022; Whisper normaliser; 95 % item bootstrap; decoder ms per 80 ms frame on CPU):

| core | set | greedy WER | beam 4 | beam 8 | decoder ms / frame (greedy / beam 4 / beam 8) |
|---|---|---|---|---|---|
| 115M | ICSI held-out (200) | 18.37 | 17.38 (−0.99 [−1.52, −0.49]) | **17.11 (−1.26 [−1.91, −0.61])** | 0.11 / 0.67 / 0.87 |
| 115M | AMI held-out (142) | 16.48 | 14.58 (−1.90 [−3.08, −0.83]) | **14.20 (−2.28 [−3.67, −1.16])** | 0.09 / 0.59 / 0.80 |
| 0.6B | ICSI held-out | 9.03 | 9.60 (+0.57 [−0.36, +1.48]) | 9.22 (+0.19 [−0.51, +0.97]) | 0.25 / 1.51 / 2.04 |
| 0.6B | AMI held-out | 7.66 | 8.24 (+0.58 [−0.21, +1.50]) | 7.74 (+0.08 [−0.63, +0.79]) | 0.21 / 1.38 / 1.89 |

The 0.6B's greedy decode is already as good as the beam: the bigger transducer's posteriors are sharp. The changes
it makes are as often wrong as right ("waiting" → "weighting"). So the 0.6B vs Parakeet-TDT gap on ICSI cannot be
closed by decoding. **Selected: beam 8 for the 115M only.**

**Serving (`--beam K`, off by default).** The beam runs next to the greedy decoder over the same encoder frames.
Finals take the beam's best hypothesis for the segment; partials, the turn heads' token inputs and everything else
keep the greedy tokens, so turn taking cannot change (`ASRStream.beam_cut`; `tests/test_fixall.py`: the streamed beam
equals the offline beam on each segment). The beam runs on a CPU copy of the transducer head. On the Apple GPU its
many small calls cost +14.5 ms per chunk (beamcost, first build), on CPU +1.5 ms.

| served cost per 160 ms chunk (bundled call, `beamcost`) | greedy | `--beam 8` | turn ends |
|---|---|---|---|
| 115M CPU 2 threads p50 / p95 | 31.1 / 33.6 ms | 32.6 / 35.1 ms (+1.5) | identical |
| 115M MPS p50 / p95, beam on the GPU (first build) | 28.7 / 37.2 | 43.2 / 53.6 (+14.5) | identical |
| 115M MPS p50 / p95, beam on a CPU copy (shipped) | 28.6 / 31.5 | 30.8 / 34.9 (+2.2) | identical |

It costs +2.2 ms per chunk on MPS and +1.5 ms on CPU. On MPS that is more than the 2 ms the latency gate allows for a
default, so it stays an option and greedy stays the default.

**Test splits (once, `final_compare.py asr --system core_115m_beam8`, same audio as every system):**

| set | 115M greedy | 115M `--beam 8` | paired Δ [95 % CI] |
|---|---|---|---|
| LibriSpeech test-clean (300) | 2.38 | 2.23 | −0.15 |
| LibriSpeech test-other (300) | 6.82 | 6.26 | −0.56 |
| AMI test (200) | 16.11 | 14.58 | **−1.53 [−2.56, −0.66]** |
| ICSI test (200) | 18.44 | 18.03 | −0.41 [−1.15, +0.35] |
| live calls, every word (32) | 20.34 | 18.61 | **−1.73 [−2.77, −0.92]** |
| live calls, user channel (16) | 15.35 | 12.89 | **−2.46 [−4.57, −0.98]** |

### 3. Language ID (selection on FLEURS dev; test once per shipped head)

**Teacher.** Whisper large-v3's language-token posterior (restricted to the 17 languages, as FINAL_COMPARE scores
it). It was run on the first 2 s from the speech onset of every FLEURS `train` row and of 800 rows per language of
`trainx`: 17 850 rows, 75 min on MPS (`lteach`). On those training rows its 2 s accuracy is 95.1 % (train) and
93.7 % (trainx). The full-row view was dropped for time (0.25 s per row on MPS). AmberNet's per-window logits
(lid_fix) stay the second teacher.

**Candidates** (lid_fix's recipe: CE + 0.8 × T² KL at T 2, class-balanced batches, 8000 steps, train + trainx + extra
English; dev picks the step; the 0.6B's `trainx` frame cache was built for this pass: 42 913 rows, blocks 16 / 20,
`lfeat6`, 62 min). FLEURS dev accuracy at 2 s / full; engine ms per 160 ms chunk on MPS with the head attached (the
full single-mode engine, bundled call):

| core | head | teacher | params | dev 2 s | dev full | engine ms / chunk (MPS p50) |
|---|---|---|---|---|---|---|
| 115M | shipped `lid_distill.pt` (hidden 512) | AmberNet | 0.92 M | 91.2 | 97.6 | 28.83 |
| 115M | hidden 512, re-trained (control) | AmberNet | 0.92 M | 91.0 | 97.2 | 28.67 |
| 115M | hidden 512 | ½ Whisper + ½ AmberNet | 0.92 M | 89.5 | 97.2 | – |
| 115M | hidden 1024 | ½ Whisper + ½ AmberNet | 2.37 M | 91.3 | 97.6 | – |
| 115M | **hidden 1024** | AmberNet | 2.37 M | **93.1** | **98.2** | 28.80 |
| 0.6B | shipped `lid_0p6b.pt` (hidden 512, no trainx) | AmberNet | 1.19 M | 88.2 | 95.2 | 43.36 |
| 0.6B | hidden 512, + trainx | AmberNet | 1.19 M | 92.6 | 97.7 | 43.10 |
| 0.6B | **hidden 1024, + trainx** | AmberNet | 2.89 M | **94.1** | 97.6 | 43.16 |

- **The Whisper teacher hurt.** With the same head and data, mixing Whisper's posterior into the KL target cost 1.5-1.8
  points at 2 s on dev. Whisper's 2 s posteriors are sharp and are wrong on 5-6 % of the training rows, so the
  target becomes confidently wrong. AmberNet's softer window-level targets transfer better. The user asked for the
  best model as teacher; on held-out data it is not the better teacher here, so the shipped candidates use AmberNet.
- **Capacity helps.** Hidden 1024 (2.4-2.9 M parameters, inside the brief's 2-4 M) adds 2.1 (115M) and 1.5 (0.6B) points
  at 2 s over hidden 512 with the same data. For the 0.6B, the `trainx` rows add 4.4 points.
- **Latency gate.** The bigger head costs nothing measurable (±0.2 ms per chunk, run-to-run spread). It pools per frame
  and classifies once per frame on 2 × hidden features.
- Chosen on dev: 115M hidden 1024 (AmberNet), 0.6B hidden 1024 + trainx (AmberNet).

**FLEURS-17 test (once, `final_compare.py lid`, the served clip path, VAD-gated), accuracy % [95 % CI]:**

| head | 2 s | full clip |
|---|---|---|
| 115M `lid_distill.pt` (shipped before; re-run through the same path) | 90.9 | 97.8 |
| **115M `lid_115m_v2.pt`** | **92.4** [91.5, 93.5] | **98.2** [97.7, 98.8] |
| 0.6B `lid_0p6b.pt` (shipped before) | 87.6 | 95.5 |
| **0.6B `lid_0p6b_v2.pt`** | **92.7** [91.7, 93.7] | **98.6** [98.1, 99.0] |
| Whisper large-v3 (bar) | 95.9 | 99.5 |

The decision from 1 s, asked for speed: on FLEURS dev the new heads are right on 79.9 % (115M) / 82.9 % (0.6B) of clips
after 1 s, 93.1 / 94.1 % after 2 s, 96.1 / 95.9 % after 3 s. 1 s is not accurate enough for an announcement, so the
served rule is unchanged: announce at confidence 0.9 after ≥ 1 s of speech, or the top language after 3 s
(`LangDecider`). FINAL_COMPARE's test protocol has no 1 s condition.

**Shipped:** `assets/lid_115m_v2.pt` (hub component `lid`) and `assets/lid_0p6b_v2.pt` (`lid_0p6b`). Both are pinned by
size and sha256. They ship in `assets/` and are not yet attached to a GitHub release. The old files stay for
reproduction. The latency gate passes (±0.2 ms per chunk). The bar (95.9 % at 2 s) is not reached: the frozen
encoders' features leave a 3-point gap at 2 s.

### Test-split audit (user instructions of 2026-10-02: "use the right test sets", "only SOTA test results")

research/FINAL_COMPARE.md was re-measured on public test splits only, with every baseline re-run on the same audio:
- LibriSpeech test-clean and test-other: 300 random utterances each.
- AMI test meetings (IS1009b, ES2004b, TS3003b, EN2002a) and ICSI test meetings (Bmr013, Bmr018, Bro021), for words,
  speech detection, speaker EER and target-speaker rows.
- AMI test turns: the eot-bench v2 cut. Every turn system was re-run on them: both cores, Pipecat, LiveKit,
  Parakeet-EOU.
- smart-turn v3.2 test and FLEURS test were already test splits.
- Two-party calls have no labelled public test split: TurnBench's test labels are withheld, which we checked on the
  Hub (`mundo-ai/turn-benchmark-test`: audio public, annotations blank). Their rows moved out of the headline.

The table "which audio each head saw: train / selection / test" is in FINAL_COMPARE.md; the rows that moved are in
its "What changed with the test splits" section. The first pass's dev-split tables are in the appendix below.

Code added for this (`final_compare.py`): eval-split sets (`ami_eval`, `icsi_eval`, `ls_clean`, `ls_other`), an AMI
eval-turn switch (`EOT_AMI_SPLIT=eval` in eot_latency / eval_stage1 / core_0p6b_heads), AMI test windows for the
target-speaker pipeline (`tsvad.bench_windows('ami_eval')`), and `fixall.py twsub / frsub / frami` for the ICSI-test
re-pooling and AMI-test pooling.

## Appendix: FINAL_COMPARE first pass on dev splits (superseded, kept for the record)

These were the headline tables of research/FINAL_COMPARE.md before the test-split audit of 2026-10-02 (AMI / ICSI **dev** meetings for words, speech detection, turn AMI rows and speaker EER; the speech-detector rows already show the v0.4 / v0.3 heads). They are not test-split numbers.

#### In plain words

- **Words.** The 0.6B core makes about as few mistakes as the best open models. On meetings and live calls it ties
  NVIDIA's offline Parakeet-TDT v3. It beats Whisper large-v3 and large-v3-turbo on AMI meetings and on the user's own
  channel of live calls. It is a little worse than Parakeet-TDT on ICSI meetings. It does this while streaming: a word
  shows up about 0.3 s after it is said. The Whisper models and Parakeet-TDT only transcribe after the speaker stops.
  The small 115M core makes about twice as many mistakes on meetings as all of them.
- **Speech detection.** Both cores beat the VADs that ship in voice agents today (Silero, TEN VAD, NVIDIA MarbleNet) on
  AMI meetings and, since the 2026-10-01 speech head (research/FIXALL.md), on ICSI meetings too (0.947 / 0.951 vs TEN
  VAD 0.935). pyannote's segmentation model is best on AMI. It reads 10 s of audio ahead, so it cannot run live.
- **Turn taking.** On speech aimed at an assistant, our `assistant` preset is right on 93 % of clips with either core.
  It answers in 0.30 s (115M) or 0.38 s (0.6B). Pipecat's smart-turn stack is right on 70 % and LiveKit's on 73 %. NVIDIA's
  Parakeet-EOU is right on 48 %. On human two-party calls, our default preset interrupts on 20 % of turns and misses
  7 %. Pipecat interrupts on 36 % and misses 25 %; LiveKit interrupts on 27 % and misses 23 %. Pipecat answers
  faster (0.24 s, against 0.96 s for the 115M and 0.73 s for the 0.6B).
- **Following your voice.** With a 5 s voice print, our tracker keeps your words and drops other people's much better
  than an open diarizer bound to the same print. On ICSI meetings, your-words-only WER is 32 % (0.6B) and 37 % (115M).
  It is 65 % for NVIDIA Nemotron-3 diarization and 75 % for pyannote 3.1. As plain speaker embedders, though, TitaNet-L
  and pyannote's WeSpeaker beat our speaker heads.
- **Cost.** The 115M runs a whole session in 28 ms per 160 ms of audio on the Mac GPU, 4 streams live, 1.2 GB. The
  0.6B needs 44 ms, 3 streams, 4.8 GB.
- **Language.** Whisper's own language detection is better than our small heads: 96 % vs 91 % (115M) and 88 % (0.6B)
  after 2 s of speech.

#### Words: WER %, Whisper English normalizer, [95 % CI]

Same audio and the same references for every system. The bootstrap is over utterances; on the live sessions it is
over clips. Lower is better.

| system | LibriSpeech test-clean (200) | AMI dev (200 segments) | ICSI dev (200 segments) | live calls, every word (32 sessions) | live calls, user channel (16) | mode |
|---|---|---|---|---|---|---|
| **audioforge 0.6B** | 2.07 [1.54, 2.71] | **10.39** [8.61, 12.29] | 13.62 [11.62, 15.49] | 11.55 [8.96, 14.93] | **5.70** [4.39, 7.18] | streaming, 160 ms |
| audioforge 115M | 2.27 [1.75, 2.85] | 20.63 [17.94, 23.48] | 26.21 [23.40, 29.17] | 20.34 [16.67, 25.08] | 15.35 [12.56, 19.18] | streaming, 160 ms |
| NVIDIA Parakeet-TDT 0.6B v3 | 1.88 [1.36, 2.50] | **9.50** [7.88, 11.29] | **10.37** [8.79, 11.97] | **11.04** [8.30, 14.67] | 7.68 [5.23, 11.67] | offline |
| Whisper large-v3 | 1.41 [0.95, 1.97] | 12.36 [10.52, 14.26] | 12.79 [11.11, 14.76] | 12.69 [9.79, 15.93] | 8.45 [6.00, 11.25] | offline |
| Whisper large-v3-turbo | **1.28** [0.89, 1.80] | 12.33 [10.33, 14.20] | 13.86 [12.13, 15.88] | 12.91 [10.02, 16.33] | 8.38 [6.63, 10.34] | offline |
| Whisper small (LiveKit default STT) | 2.29 [1.69, 2.96] | 14.49 [12.36, 16.88] | 18.17 [15.99, 20.69] | 14.67 [11.60, 18.18] | 10.14 [8.25, 12.25] | offline |
| Nemotron 3.5 ASR 0.6B (multilingual), en-US prompt | 2.59 [1.94, 3.30] | 14.78 [12.85, 17.20] | 20.27 [17.80, 23.00] | 16.99 * | 11.17 * | streaming, 160 ms |
| Deepgram Nova-3 | not measured (paid API) | | | | | |

\* Repo normaliser, from `runs/core_3p5.json`. The other cells of that row also come from that file: the same 200-item
sets and sessions, reused.

Paired difference against the 0.6B, percentage points [95 % CI] (positive = the baseline makes more errors):

| baseline − 0.6B | LibriSpeech | AMI | ICSI | live, every word |
|---|---|---|---|---|
| Parakeet-TDT v3 | −0.19 [−0.62, +0.28] | −0.89 [−2.38, +0.40] | **−3.25 [−4.46, −2.05]** | −0.51 [−1.50, +0.69] |
| Whisper large-v3 | **−0.66 [−1.13, −0.25]** | **+1.97 [+0.35, +3.55]** | −0.83 [−2.40, +0.76] | +1.14 [−0.46, +2.89] |
| Whisper large-v3-turbo | **−0.79 [−1.28, −0.32]** | +1.94 [−0.00, +3.74] | +0.24 [−1.38, +2.16] | +1.36 [−0.12, +2.82] |
| Whisper small | +0.22 [−0.29, +0.70] | **+4.10 [+2.33, +5.89]** | **+4.55 [+2.92, +6.38]** | **+3.12 [+1.25, +4.92]** |
| audioforge 115M | +0.20 [−0.25, +0.57] | **+10.24 [+8.29, +12.34]** | **+12.59 [+10.37, +15.01]** | **+8.79 [+7.13, +10.93]** |

With the repo's own normaliser (`teachers.normalize_text`, used in earlier tables), Whisper scores much worse on the
meeting sets: AMI 19.1 / 19.4 / 21.2 % for large-v3 / turbo / small, against 11.2 % (0.6B) and 9.7 % (TDT). This is
because it counts spelling differences that Whisper's normaliser removes. The table above uses Whisper's normaliser,
which is the fair choice when Whisper is in the comparison. Both are in the json.

**Streaming word latency (ours only).** This is the time from the end of a word to the word first appearing in the
live transcript. It was measured through the served engine on MPS, on 12 two-minute AMI test windows
(`stt_latency.py` protocol):
- 115M: p50 273 ms [250, 290], p95 570 ms.
- 0.6B: p50 290 ms [270, 309], p95 590 ms.

The 115M's published 441 ms was measured on CPU before the chunk-trigger fix (d832fca, −70 ms). Its block compute was
32 ms there, against 29 ms on MPS here. The baselines above are offline and have no streaming latency. Whisper small
in the LiveKit default waits for the end of the utterance.

#### Speech detection

AMI dev (64 × 20 s) and ICSI dev (64 × 20 s), 80 ms frames, label = anyone speaking. Baselines are max-pooled onto the
80 ms grid. The bootstrap is over windows. "Miss" = speech frames missed at the threshold where 7.5 % of non-speech
frames are false alarms. "Onset lag" = labelled speech onset (after ≥ 400 ms of silence) to the first frame > 0.5, on
the frame grid. It does not include each model's own look-ahead.

| system | AMI F1 @0.5 | AMI ROC-AUC | AMI miss @7.5 % FA | ICSI F1 | ICSI AUC | ICSI miss | onset lag p50 / p90 (AMI) | live-capable |
|---|---|---|---|---|---|---|---|---|
| audioforge 115M speech head (v0.4) | 0.951 [0.939, 0.960] | 0.970 [0.960, 0.978] | 12.1 % [8.4, 14.8] | **0.947** [0.942, 0.952] | 0.964 [0.959, 0.969] | 10.7 % | 0 / 160 ms | yes (80 ms right context) |
| audioforge 0.6B speech head (v0.3) | 0.950 [0.938, 0.959] | 0.969 [0.958, 0.979] | 12.5 % [8.0, 16.9] | **0.951** [0.946, 0.955] | **0.966** [0.962, 0.970] | **10.2 %** | 80 / 160 ms | yes (80 ms right context) |
| *before (FINAL_COMPARE pass): 115M heads v0.3* | *0.951* | *0.972* | *10.6 %* | *0.898* | *0.931* | *19.8 %* | *80 / 160 ms* | |
| *before: 0.6B heads v0.2* | *0.951* | *0.972* | *10.7 %* | *0.906* | *0.925* | *22.4 %* | *80 / 160 ms* | |
| Silero VAD v5 (Pipecat / LiveKit) | 0.915 [0.900, 0.926] | 0.956 [0.941, 0.969] | 13.8 % [8.9, 21.8] | 0.929 [0.922, 0.935] | 0.934 [0.925, 0.942] | 19.9 % | 80 / 160 ms | yes |
| TEN VAD | 0.925 [0.912, 0.935] | 0.948 [0.934, 0.960] | 13.6 % [11.1, 19.2] | 0.935 [0.929, 0.942] | 0.952 [0.945, 0.957] | 14.0 % | 0 / 80 ms | yes |
| NVIDIA MarbleNet v2 frame VAD | 0.937 [0.919, 0.951] | 0.959 [0.946, 0.971] | 12.4 % [8.5, 16.2] | 0.929 [0.923, 0.935] | 0.940 [0.932, 0.948] | 19.8 % | 0 / 0 ms | offline here (whole-window conv) |
| pyannote segmentation-3.0 | **0.963** [0.952, 0.972] | **0.984** [0.978, 0.988] | **4.6 %** [3.2, 7.3] | 0.899 [0.891, 0.907] | 0.904 [0.876, 0.926] | 28.6 % | 0 / 0 ms | no (10 s windows, sees the future) |

Updated 2026-10-01 (research/FIXALL.md step 1): the rows are the new `speech` head of each core (served heads 115M
v0.4 / 0.6B v0.3), a stateless head trained on AMI + 10 h of ICSI train meetings + oto user channels. ICSI dev
meetings (Bmr021, Bns001) were never in training or selection, but ICSI is no longer an unseen corpus for us. Paired
against TEN VAD on ICSI: F1 +0.009 to +0.015 (115M), +0.011 to +0.019 (0.6B). Against the old heads on AMI: F1 −0.0006
[−0.0033, +0.0019] (115M) and −0.0007 [−0.0048, +0.0035] (0.6B), AUC −0.002 / −0.003, miss +1.5 / +1.8 points: within
the CIs, but a small step back. Paired, ours vs Silero on AMI: F1 +0.027 to +0.047, AUC +0.006 to +0.024 (95 % CI,
both cores). The turn rules keep reading the old `vad` head, so the turn rows below are unchanged.

#### Turn taking

The data:
- **Assistant clips**: smart-turn v3.2 test, 399 clips (175 complete, 224 cut mid-utterance), each padded with 3 s of
  silence.
- **Calls**: 32 two-party user channels, 109 turn ends.
- **AMI**: 200 meeting turns.

Latency is from the audible end of speech to the turn end, plus the measured compute. "False fire" = a turn end within
3 s of an incomplete clip's cut. "Interrupt" = a turn end inside the user's turn. "Missed" = no turn end before the
next speaker or within 6 s. The bootstrap is over clips / sessions (1000 resamples).

| system | assistant: accuracy | assistant: p50 / p95 ms | assistant: false fire on incomplete | assistant: missed | calls: p50 / p95 ms | calls: interrupt | calls: missed | AMI: p50 ms | AMI: interrupt | AMI: missed |
|---|---|---|---|---|---|---|---|---|---|---|
| **audioforge 115M `assistant`** | **92.7 %** [90.2, 95.0] | 299 / 710 | **5.4 %** | 5.1 % | 3232 / 4801 | 5.5 % | 33.9 % | 1647 | 7.0 % | 52.5 % |
| audioforge 115M `balanced` (default) | 43.9 % | 1226 / 1587 | 100 % | 0.0 % | 955 / 1918 | 20.2 % [13.1, 29.2] | **7.3 %** [3.5, 12.0] | 1326 | 10.5 % | **33.5 %** |
| audioforge 115M `fast` | 41.6 % | 461 / 698 | 100 % | 4.0 % | 547 / 1861 | 24.8 % | 5.5 % | 1248 | 11.5 % | 34.0 % |
| **audioforge 0.6B `assistant`** | **93.0 %** [90.5, 95.5] | 379 / 702 | **4.5 %** | 9.7 % | 3749 / 4755 | 1.8 % | 43.1 % | 1981 | 1.0 % | 64.5 % |
| audioforge 0.6B `balanced` (default) | 42.6 % | 634 / 794 | 100 % | 2.9 % | 730 / 2036 | 19.3 % [12.1, 26.4] | 10.1 % [4.9, 16.0] | 1180 | 11.0 % | 33.0 % |
| audioforge 0.6B `fast` | 40.1 % | 409 / 602 | 100 % | 8.0 % | 494 / 1977 | 26.6 % | 11.0 % | 1102 | 17.0 % | **27.0 %** |
| Pipecat smart-turn v3.2 + Silero (defaults) | 69.7 % [65.2, 74.2] | **211** / 242 | 40.2 % | 13.1 % | **237** / 3217 | 35.8 % [25.7, 46.6] | 24.8 % [16.1, 33.6] | 385 | 27.5 % | 44.5 % |
| LiveKit Agents 1.8 EnglishModel + Silero | 72.7 % [67.9, 76.9] | 547 / 3016 | 44.6 % | 4.6 % | 567 / 3127 | 26.6 % [20.2, 34.0] | 22.9 % [15.1, 30.2] | 1890 | 12.5 % | 67.5 % |
| NVIDIA Parakeet-Realtime-EOU 120M | 48.4 % [43.6, 53.1] | 462 / 1210 | 91.5 % | 0.6 % | 1277 / 5362 | 7.3 % | 30.3 % | 2251 | **1.5 %** | 77.5 % |
| *smart-turn v3.2 classifier alone (upper bound: one call per whole clip, no timing)* | *97.0 %* [95.2, 98.5] | – | *2.7 %* | *3.4 %* | – | – | – | – | – | – |

The presets trade one domain for another:
- `assistant` waits up to 3.4 s for a classifier "complete". That suits assistant speech, but it misses a third of
  human call turns.
- `balanced` (the default) is the calls preset.

No single setting of any system wins every column. The CIs of every cell are in the json.

#### Speaker tracking ("your words only")

The windows are the eot-bench v2 windows: AMI dev, 974 windows; ICSI held out, 1312 windows and 1286 target units.
- The target is the main speaker of each window, with a 5 s voice print cut from elsewhere in the same meeting.
- **tWER** = WER of the target's words only. Streaming words are kept where the system's target mask is on, with the
  same keep rule for every arm.
- **Target DER** = (missed target frames + false target frames) / target frames, at 80 ms, with no collar.
- The open diarizers' columns are bound to the same print by `tsvad.vp_follow`. The better of two binders (served
  speaker head, TitaNet-L) is shown.

| system | AMI tWER % [meeting CI] | ICSI tWER % [meeting CI] | AMI target DER % | ICSI target DER % | AMI tracking F1 | ICSI tracking F1 |
|---|---|---|---|---|---|---|
| audioforge 115M (115M words) | **62.1** [47.7, 75.7] | 37.1 [33.7, 41.3] | **50.9** | 23.0 | 0.743 [0.726, 0.759] | 0.882 [0.875, 0.889] |
| audioforge 0.6B (0.6B words) | 63.2 [47.7, 78.1] | **31.7** [28.5, 34.8] | 51.8 | **21.1** | **0.760** | **0.894** |
| Nemotron-3-Diarization + print, 115M words | 74.7 [59.5, 84.8] | 65.1 [56.9, 73.3] | 69.5 | 74.9 | 0.648 | 0.698 |
| Nemotron-3-Diarization + print, 0.6B words | 71.6 [55.5, 81.8] | 60.0 [49.6, 69.5] | (same track) | | | |
| pyannote speaker-diarization-3.1 + print, 115M words | 85.3 [69.6, 98.3] | 74.8 [65.1, 84.1] | 77.8 | 89.2 | 0.690 | 0.670 |
| pyannote speaker-diarization-3.1 + print, 0.6B words | 87.0 [69.8, 97.0] | 70.0 [58.4, 80.4] | (same track) | | | |
| no filter (115M / 0.6B words) | 106.5 / 117.8 | 100.8 / 99.0 | – | – | – | – |
| oracle filter (115M / 0.6B words) | 47.4 / 34.3 | 31.3 / 24.6 | – | – | – | – |

- The audioforge, Nemotron-3-with-115M-words, no-filter and oracle rows are reused unchanged from `runs/tswer.json`,
  `runs/core_0p6b.json` (tswer, tsvad_frame) and `runs/improve_115m.json`. They come from the same windows, words and
  keep rule.
- The pyannote rows and the Nemotron-3-with-0.6B-words row were measured in this pass. Two consistency checks: this
  pass recomputes the Nemotron-3 115M-word arm as exactly 80.89 / 74.66 (AMI) and 66.77 / 65.09 (ICSI), and the 115M
  TS-VAD frame row as 0.7428 / 50.93 %.
- pyannote runs offline on the whole 20 s window, which helps it.
- pyannote's DER is high because its column, even with the oracle column picked, keeps much non-target speech: oracle
  column DER 69 % (AMI) / 83 % (ICSI).
- Nemotron-3 has AMI in its training data.
- The 32 live sessions: our tWER is in `runs/tswer_live.json`. No diarizer was run on them (see "Not measured").

**Within-meeting speaker EER** (`spk_head.py` dev segments, n = 200 per corpus, segment bootstrap):

| embedder | AMI dev EER % | ICSI dev EER % |
|---|---|---|
| audioforge 115M speaker head (block 4) | 19.8 [15.3, 23.9] | 7.0 [4.3, 10.0] |
| audioforge 0.6B speaker head (block 5) | 13.6 [9.8, 17.3] | 3.6 [1.9, 5.4] |
| NVIDIA TitaNet-L (the teacher) | 12.0 [8.1, 16.2] | 2.1 [1.1, 3.0] |
| pyannote WeSpeaker ResNet34 (in pyannote 3.1) | **11.9** [8.4, 15.8] | **1.0** [0.5, 1.6] |

The dedicated embedders are better voice matchers than our heads. Our tracker still wins on tWER and DER, because the
TS-VAD head is trained to track the print frame by frame, not only to embed.

#### Language ID: FLEURS-17 test (2550 clips), accuracy % [95 % CI]

| system | 2 s from speech onset | full clip |
|---|---|---|
| Whisper large-v3 | **95.9** [95.2, 96.7] | 99.5 [99.2, 99.7] |
| Whisper large-v3-turbo | 95.2 [94.3, 96.0] | 99.3 [99.0, 99.7] |
| NVIDIA AmberNet (the teacher; reused, `data/lid/preds/ambernet`) | 95.1 [94.3, 95.9] | **99.5** [99.3, 99.8] |
| Whisper small | 90.6 [89.4, 91.7] | 99.1 [98.7, 99.4] |
| audioforge 115M head (reused, `lid_distill_vadgated`) | 91.0 [89.9, 92.0] | 97.8 [97.2, 98.4] |
| audioforge 0.6B head | 87.6 [86.3, 88.8] | 95.5 [94.6, 96.2] |

Every system's posterior is restricted to the 17 test languages (the `lid.py` protocol). Both of our heads are
VAD-gated, as served. The 0.6B head has less training data than the 115M head: FLEURS `trainx` was not cached for it.

#### Cost on this Mac (Apple M5)

Full single-mode engine: ASR, VAD, turn, TS-VAD and LID. The bundled 16 s call, best of 3. Streams = interleaved
sessions on 40 s AMI windows, real time while p95 of the summed block compute stays under 160 ms (`mps_115m.py`
protocols).

| core | ms per 160 ms chunk, MPS (p50 / p95) | CPU 2 threads | real-time streams, MPS | real-time streams, CPU | peak RSS |
|---|---|---|---|---|---|
| audioforge 115M | 27.7 / 29.5 | 31.0 / 33.1 | 4 | 4 | 1.2 GB (+0.5 GB MPS) |
| audioforge 0.6B | 43.7 / 49.3 | 99.4 / 107.1 | 3 | 1 | 4.8 GB (+3.3 GB MPS) |

**Model sizes** (parameters counted from the weight files):

| model | params |
|---|---|
| audioforge 115M (whole served model) | 122.1 M |
| audioforge 0.6B (whole served model) | 622.5 M |
| Whisper small / large-v3-turbo / large-v3 | 241.7 M / 808.9 M / 1543.5 M |
| Parakeet-TDT 0.6B v3 | 627.0 M |
| Parakeet-Realtime-EOU | 114.9 M |
| smart-turn v3.2 (ONNX) | 8.0 M |
| LiveKit turn detector (largest ONNX in the cache) | 65.5 M |
| Silero VAD v5 (TorchScript, its 8 kHz and 16 kHz branches together) | 0.46 M |
| MarbleNet v2 frame VAD | 0.09 M |
| pyannote segmentation-3.0 / WeSpeaker ResNet34 | 1.47 M / 6.63 M |
| TEN VAD | parameter count not exposed (compiled library, 0.7 MB) |
| Nemotron-3-Diarization | 99.2 M |
| TitaNet-L / AmberNet | 25.3 M / 28.9 M |

The baselines' cost per stream was not measured. Their RTFx in this pass is in the json, and it mixes devices: Whisper
large on MPS, the rest on CPU.

