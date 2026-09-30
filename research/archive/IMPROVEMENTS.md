# Improvements to what is not yet the best in the served model (2026-09-28)

Owner: the improvements research agent. Request: "start looking for improvements" / "can we improve what is not the
best in the model". Production target: GPU deployment but lightweight real-time first; a **known target speaker**
(the product holds the user's voice print, or takes the first voice after its own TTS ends); the 115M streaming core
at 80 ms frames for the live heads; per-turn Parakeet-TDT v3 for the transcript. Machine rules: every model launch
through `scripts/dev/gate.sh` (2 threads), one heavy job at a time, CPU calls < 10 min (resumable), MPS training one
at a time and < 20 min per call, large files on the SSD with a read-back check.

Starting point (research/IMPROVE_115M.md A.2): with a 5 s voice print, feeding the TS-VAD target-speaker track into
the served turn head cuts missed turn ends at 6 s from 61.9 % to 39.3 % on AMI dev and from 68.5 % to 20.4 % on
held-out ICSI (hybrid_dyn 34.2 / 18.7 %), but the head's P50 latency stays ~4 s on AMI and the path was not served.

## Pre-registered bars (written before any of the results below)

| # | experiment | success bar | kill criterion |
|---|---|---|---|
| 1 | Serve the TS-VAD path (`serve --turn-input tsvad`, voice print explicit or after_agent_arm), live on the 37 E2E clips (69 sessions) through the Pipecat adapter, vs the stored C / D records | pooled over the 69 sessions: **missed at 3 s below C's** and **cut-ins per clip <= D's** (paired clip bootstrap) | the served path does not reproduce the offline track / head (max abs difference > 0.01), or misses at 3 s >= C's with CI excluding a gain on every set |
| 2 | Turn-head latency on the TS-VAD track: retrain heads only (frozen encoder) on TS-VAD tracks with the predictive multi-horizon target (DYADIC section 8) | **P50 <= 1.5 s at the same FC (<= 5 % per-turn, cross-fitted) on AMI dev and ICSI held-out**, and TurnBench dev recall >= 0.85 at FP <= 0.10 | P50 not below 3.0 s on AMI at matched FC after two head variants, or misses at 6 s worse than the untrained head's 39.3 / 20.4 % with CI excluding 0 |
| 2b | (user request) `hybrid_fast`: at the first Silero / VAD silence onset + a minimum of 160 / 240 / 320 / 480 ms, fire if the turn head's p >= tau_fast (swept so realised per-turn FC <= 5 %), else fall back to hybrid_dyn; eot-bench v2 AMI dev + ICSI (frozen thresholds) and TurnBench dev, Sortformer and TS-VAD inputs, vs hybrid_dyn, the 1 s timeout and smart-turn | pin in serve.py (`--turn-policy hybrid_fast --fast-min-ms --fast-threshold`, with a test) **only if a point beats hybrid_dyn's P50 by >= 300 ms at equal or lower FC** | no point beats hybrid_dyn's P50 by 300 ms at matched FC |
| 3 | Speaker head: frame-level TitaNet distillation with more identities (LibriSpeech train-clean-100 speakers) + the relational loss | **within-meeting EER < 12 % at n = 200 on AMI dev, ICSI dev <= 5 %** | AMI within-meeting EER (n = 200) not below the shipped 19.8 % by >= 2 points |
| 4 | Enrolment quality: print length 1.5 / 3 / 5 / 10 s, after_agent_arm prints taken from live audio, print refresh | **<= 2 points of misses (6 s) lost vs the 5 s offline print** for the recommended live rule | every live rule loses > 5 points |
| 6 | (user request, 2026-09-28, runs after 1 and before 3-5) nemotron-speech-streaming-en-0.6b as the GPU streaming core | **adopted only if EVERY category improves over the 115M core** (user decision): WER on AMI-200 / ICSI-200 / LibriSpeech-200 at the same lookahead ([70,1] and [70,13]); VAD F1 (AMI, ICSI); within-meeting speaker EER (AMI n = 200, ICSI dev); eot-bench missed ends at matched FC (AMI, ICSI; all and floor-open) with the same turn rules; TurnBench recall at FP <= 0.10; first-partial latency; heads trained the same way on both encoders; one table with the paired delta and CI per category. A category counts as improved when its paired delta favours the 0.6B with a CI excluding 0; equal within noise is "not shown to improve" | any category worse beyond noise -> "not adopted" (the write-up names it and why); measured in order of cost, so the first failed category ends the programme. RTF and RSS on CPU 2 threads and MPS are reported either way |
| 7 | (user request, queued after 6 and 2b) distil nemotron-speech-streaming-en-0.6b into the 115M core: teacher = the 0.6B at [70,1] (+ [70,13] if cheap) over AMI + ICSI train + a LibriSpeech train-clean-100 anchor; student = the full 115M encoder + RNNT; sequence-level KD (teacher transcripts as targets) + frame-level KD (encoder-output regression through a small projector; RNNT/CTC posteriors where aligned), layer-wise LR decay, KL anchor on LibriSpeech, WER gate as kill switch; teacher outputs cached once to the SSD (<= 100 GB); designed for one A100 / H100 day, only a <= 20 min MPS smoke test here; control = continued training without the teacher at the same budget | **AMI-200 streaming WER <= 17 %** ([70,1]) and **LibriSpeech-200 <= 2.59 %** (+0.3 of 2.29), VAD / turn heads re-checked on the new encoder and not worse beyond their CIs | the WER gate fires (LibriSpeech + > 0.3) at every checkpoint, or AMI not below the continued-training control by > 1.5 points |
| 5 | (only if 1-4 are done) meeting-ASR partials from the decoder-adapted heads (runs/asr_meeting_decoder_adapt.afm) + the dual-lookahead pass on them | AMI streaming WER below the served 24.4 % with LibriSpeech not worse (CI through 0) and no live decision changed | any live head output changes, or LibriSpeech WER + > 0.2 points |

## Summary (2026-09-28)

| # | verdict | headline |
|---|---|---|
| 1 | bar not met; shipped behind flags | `serve --turn-input tsvad [--diar-off]` reproduces the offline TS-VAD path exactly; live (69 Pipecat sessions) it misses +10.8 [+5.6, +15.8] points more than C at 3 s, equals D on misses and cut-ins at 0.43x the server compute (RTF 0.34 vs 0.80, no diarizer) |
| 2 | killed | a lightweight TS-VAD-fed predictive head (no second encoder pass) misses 10-14 points more than the served head; ICSI median 400 ms earlier |
| 2b | no | `hybrid_fast` (head read at silence onset + 160-480 ms) moves P50 by <= 80 ms at matched FC; not pinned |
| 2c | product knob | at 10 % FC the TS-VAD hybrid_dyn misses 22 % (AMI) / 13 % (ICSI) with the median ~1 s / 0.5 s earlier than at 5 % |
| 3 | partial | crop-level TitaNet distillation: within-meeting EER AMI 19.8 -> 17.1 %, ICSI 7.0 -> 4.0 % (ICSI bar met, AMI bar of 12 % not) |
| 6 | not adopted | 0.6B core: meeting WER -13 points, VAD better, ICSI turn better, but AMI eot-bench (hybrid_dyn) +2.8 [+0.1, +5.3] worse and five categories only equal |
| 7 | package delivered, GPU run pending | distillation of the 0.6B into the 115M (AMI + ICSI + LibriSpeech anchor), smoke-tested; ~6 A100 hours |
| 4 | bar not met; live rules killed (2026-09-29, research/SINGLE_MODEL.md A2) | stored prints: 3 s +13.2 / +1.2, 10 s −6.5 / +0.9 points vs 5 s (AMI / ICSI, hybrid_dyn, 6 s); every live print after agent_end loses 16-55 points |
| 5 | not run | meeting-partials experiment deprioritised for 6 / 7 |

## 1. Serving the TS-VAD turn path (`serve --turn-input tsvad`)

**Hypothesis.** The offline A.2 gain (missed turn ends at 6 s 61.9 -> 39.3 % AMI, 68.5 -> 20.4 % ICSI) survives
serving: a live session that holds the user's voice print answers fewer user turns late than the product default C,
without more cut-ins than the no-cut-in rule D.

**What was built** (behind flags; the default protocol is unchanged):
- `audioforge/tsvad_stream.py`: `TSVADTrack` runs the TS-VAD head (`runs/tsvad_spk.pt`, 0.26 M) on block 4 of the ASR
  pass the server already makes (the speaker head's tap: no extra encoder pass) and keeps the voice print; the print
  is either stored (`{"type": "enroll", "embedding": [192 floats]}`, computed by `tsvad_stream.voiceprint` from a few
  seconds of the user's audio), or taken live from the next `--tsvad-print-s` seconds of served-VAD speech after
  `agent_end` (`--enroll after_agent_arm`) / after `enroll`, or from the session's first speech (`dominant`); optional
  refresh (`--tsvad-refresh-s`). Before a print exists the head runs with its learned no-enrolment vector (a plain
  VAD, as trained). Each new print is announced with `{"type": "voiceprint", "t", "seconds", "source"}`.
- `serve.py --turn-input tsvad`: the turn head is fed exactly the benchmark's input (primary activity P(target),
  columns [P(target), P(other), 0, 0], primary 0) on the ASR chunk clock, with the speaker-conditioned second encoder
  pass as for `diar`; `hybrid_dyn` uses its own refitted point `TSVAD_DYN` = (theta 0.99748, offset 3.22 frames), the
  mean of the two AMI-dev cross-fit fold points of A.2 (the shipped point was fitted for the Sortformer-fed head and
  does not transfer). TitaNet is never loaded in this mode.
- `serve.py --diar-off` (with tsvad only): the session's "diarizer" columns become [P(target), P(other), 0, 0]
  (`tsvad_stream.TSVADColumns`), so `frame.speakers` / `primary` and the timeout policies read the enrolled user and
  **no diarizer pass runs**. Measured on the first live sessions with the diarizer on, Sortformer was 2/3 of the
  server's compute (diar 15.7 s of 24.3 s processing for 27 s of audio) and pushed the server to RTF 0.86-0.95 under
  the shared machine's load, with backlogs up to 1.6 s; the turn path does not need it once the user is enrolled.
- Pipecat adapter: `AudioforgeSTTService.enroll(embedding=...)`, `hub.voiceprints`.
- Tests: `tests/test_tsvad_stream.py` (8: track == offline decode with a print, arming takes exactly the next N speech
  frames and swaps the print mid-stream with the GRU state kept, refresh, explicit / armed sessions, hybrid_dyn point,
  diar-off columns, protocol validation).

**Served path == offline benchmark** (`scripts/research/tsvad_serve_check.py`, 12 random AMI dev eot-bench windows,
1830 frames, audio streamed in 20 ms blocks through `serve.Session` with the benchmark's stored 5 s print): the live
TS-VAD track differs from the offline `bind_tsvad_spk` track by at most 5.3e-4 (mean 2e-5), the live turn-head
posterior from the offline `scores_tsvad_spk` by at most 4.0e-3 (mean 2.1e-5), and the frames at or above 0.997 are
the same (179 live, 179 offline). The kill criterion "served path does not reproduce the offline track" is not met:
the offline A.2 numbers are the served path's numbers at the frame level.

**Live setup** (`scripts/research/e2e_tsvad.py`). E2E_FINAL's scratch was lost in a reboot, so the 37 clips were rebuilt
from `runs/e2e_final.json`: the 32 TurnBench / oto clips with `e2e_final.two_party_clip` on the same conversations
(every start and duration matches to 10 ms), the 5 AMI windows from the eot-bench v2 AMI dev windows with the same
meeting and start (every scored turn end matches E2E_FINAL's). System **T** = `--turn-input tsvad --diar-off --enroll
explicit`, turn policy `hybrid_dyn` at `TSVAD_DYN`, voice print = 5 s of the user's single-speaker speech from
elsewhere in the same conversation (outside the clip +-2 s; TurnBench / oto: the user's own channel with the other
party silent; AMI: the primary's asr-mode segments), embedded by the served block-4 speaker head, sent once at
connect. Pipecat 1.12 through the committed adapter, E2E_FINAL's pads and scorer; C and D are E2E_FINAL's stored
Pipecat per-clip records (same clips, same protocol and scorer; run 2026-09-27 on the same machine, not re-run).
Deviation: E2E_FINAL's quiet check (other processes < 150 % of a core) was relaxed to 400 % (of 10 cores) because other agents kept
the laptop above it; each session's load and server RTF are recorded.

**Results** (`runs/e2e_tsvad.json`; 69 sessions per system, 223 scored user turn ends; T and Th run 2026-09-28 with
server RTF 0.24-0.50, median 0.34 / 0.29, no session flagged; C / D from E2E_FINAL at RTF 0.80). Pooled over the 69
sessions, paired clip bootstrap (2000 resamples):

| comparison | missed at 3 s | missed at 6 s | cut-ins per session |
|---|---|---|---|
| T (tsvad, hybrid_dyn) vs **C** | 45.3 vs 34.5 %: **+10.8 [+5.6, +15.8]** | 38.6 vs 29.2 %: +9.4 [+4.1, +14.9] | 0.52 vs 0.93: **-0.41 [-0.67, -0.12]** |
| T vs **D** | 45.3 vs 44.0 %: +1.3 [-3.6, +6.2] | 38.6 vs 37.7 %: +0.9 [-3.1, +5.1] | 0.52 vs 0.52: 0.00 [-0.17, +0.19] |
| Th (tsvad, head OR 1 s timeout on P(target); post hoc) vs C | 46.2 vs 34.5 %: +11.7 [+5.5, +19.0] | 35.0 vs 29.2 %: +5.8 [+0.9, +11.9] | 0.71 vs 0.93: -0.22 [-0.45, +0.01] |
| Th vs D | 46.2 vs 44.0 %: +2.2 [-4.4, +8.9] | 35.0 vs 37.7 %: -2.7 [-7.8, +2.3] | 0.71 vs 0.52: +0.19 [-0.06, +0.42] |

Per set (Pipecat; dead air median ms / missed 3 s / cut-ins per min):

| set | C | D | T | Th |
|---|---|---|---|---|
| AMI mono (5) | 1683 / 20 % / 1.36 | 2561 / 20 % / 0.00 | 2221 / 20 % / 0.00 | 1763 / 0 % / 0.68 |
| TurnBench mono (16) | 1346 / 68 % / 1.15 | 1696 / 70 % / 0.49 | 1948 / 80 % / 0.74 | 1916 / 75 % / 0.82 |
| TurnBench user (16) | 1353 / 23 % / 1.32 | 2311 / 38 % / 0.74 | 1986 / 41 % / 0.99 | 1866 / 48 % / 0.58 |
| oto mono (16) | 1372 / 42 % / 1.26 | 2252 / 57 % / 0.79 | 2001 / 49 % / 0.79 | 1421 / 45 % / 1.34 |
| oto user (16) | 1393 / 6 % / 1.26 | 2343 / 13 % / 0.87 | 2006 / 11 % / 0.39 | 1801 / 19 % / 1.10 |

Compute (server, CPU 2 threads, same clips): T RTF median 0.34 (max 0.50), Th 0.29 (0.34), peak RSS 1.6 GB, vs 0.80 /
3.6 GB for C / D with Sortformer in E2E_FINAL (the diarizer model is still loaded in T but never run; not loading it
would save its memory too).

**Verdict: bar not met.** Misses at 3 s are **10.8 points worse than C** (CI excludes 0); the cut-in half of the bar
holds (T equal to D). The kill criterion is not triggered (the served path reproduces the offline benchmark, and the
oto mono CI still admits a gain), so the path stays in the server **behind flags** (`--turn-input tsvad [--diar-off]`),
not as a default. What the run does show:
- **T is D at 0.43x the compute.** Against D (the no-cut-in rule it replaces) T has the same misses (+1.3 [-3.6, +6.2])
  and the same cut-ins (0.00), with lower median dead air on four of five sets (-250 to -340 ms), no diarizer pass and
  RTF 0.34 instead of 0.80. For a product that knows its user's voice this is the cheaper way to run D's rule.
- **Why it does not beat C here.** The offline gain (A.2) was measured on AMI / ICSI meetings, the domain the TS-VAD
  head and the turn head were trained on. 64 of the 69 sessions are two-party telephone-style calls (TurnBench, oto),
  where C's 1 s timeout is already strong on the user channel (6-23 % missed) and every head-gated rule pays a
  second of extra dead air (DYADIC section 7, E2E_FINAL D vs C). On the 5 AMI windows T matches C's misses with no
  cut-ins, and Th answers all five within 3 s (0 vs 20 %) at C's dead air (1763 vs 1683 ms), but n = 5.
- The latency problem is the head, not the track: T's decisions come a median ~2 s after the true end. This is
  experiment 2's target.

## 2. A turn head trained on TS-VAD tracks, with the predictive multi-horizon target

**Hypothesis.** The served turn head reads the TS-VAD track "untrained" (it was trained on Sortformer tracks, whose
calibration and 240 ms lag differ), which is why its P50 stays ~4.2 s on AMI (A.2). A head trained on TS-VAD tracks
with DYADIC section 8's predictive target (the user's future activity in disjoint bins, which was the faster decision
signal on TurnBench) should fire within 1.5 s at the same false-cut rate.

**Design choice.** Rather than fine-tuning the served kernel head (which needs a second, speaker-conditioned encoder
pass per chunk, RTF ~0.15), the new head is `audioforge.heads.uc_turn.UCTurnHead` (from the user-channel draft,
0.25 M parameters): it reads the **top layer of the ASR pass itself** (no second encoder pass), the TS-VAD track
[P(target), P(other)] as its two activity columns, their causal duration counters and the causal standardised log-RMS,
and predicts P(target active) in bins (0-240], (240-400], (400-640], (640-1040], (1040-2000] ms, P(target quiet for
1.04 / 2 s) and the others' bins. If it meets the bar it also removes the second encoder pass from the served turn path.

**Setup** (`scripts/research/tsvad_turn.py`). Training crops = the TS-VAD head's own cached 16 s crops (hop 12 s) of
the 12 AMI + 12 ICSI train meetings; top-layer features by continuing encoder blocks 5-17 from the cached block-4
features (same [70, 1] chunked mask). The TS-VAD tracks on training crops must look held-out, so they come from two
**cross-fitted TS-VAD heads** (each trained on the other half of the 24 meetings with the Part A recipe) with a random
voice print of the target from elsewhere in the meeting (the enrolment pools, clips overlapping the crop +-1 s
excluded); 10 % of items use the no-print track (target = any speech). Targets from the word-level speaker labels.
AdamW 2e-3 one-cycle, batch 32, 3000 steps, MPS. Evaluation = A.2's protocol exactly: eot-bench v2 AMI dev (974) and
ICSI held-out (1312), the stored 5 s-print TS-VAD tracks, emission at the 160 ms chunk, cross-fitted <= 5 % per-turn
false cuts (leave-meetings-out halves), 1000 bootstraps, paired against the served head on the same track. Decision
scores tried (pre-declared): P(quiet 2 s), P(quiet 1.04 s), 1 - max(P(bin1), P(bin2)) (the predictive trigger's
condition), 1 - max(bins 1-4); each alone and in `hybrid_dyn` with Silero.

**Results** (`runs/tsvad_turn.json`, 6 s horizon, cross-fitted <= 5 % per-turn FC; paired deltas vs the served head
on the same TS-VAD track).

| system (TS-VAD 5 s-print track) | AMI miss % | AMI FC % | AMI P50 ms | AMI open miss % | ICSI miss % | ICSI FC % | ICSI P50 ms | ICSI open miss % |
|---|---|---|---|---|---|---|---|---|
| served kernel head (A.2) | 40.0 | 5.0 | 4240 | 39.7 | 20.5 | 5.9 | 1920 | 21.5 |
| served head, hybrid_dyn (A.2) | **34.2** | 5.5 | 3200 | **17.9** | **18.7** | 6.2 | 1920 | **13.2** |
| (a) future-activity bins only: best score (1 - max(bin1, bin2)) | 77.2 | 3.5 | inf | 72.1 | 69.7 | 5.0 | inf | 84.5 |
| (a) hybrid_dyn, best score | 60.6 | 8.0 | inf | 28.2 | 62.9 | 6.5 | inf | 37.0 |
| (b) + end-of-turn output: P(eot) | 55.3 | 5.1 | inf | 61.3 | 34.0 | 4.8 | **1520** | 49.5 |
| (b) hybrid_dyn on P(eot) | 44.1 | 5.8 | 2960 | 24.7 | 32.2 | 5.0 | **1520** | 37.9 |

Paired, (b) hybrid_dyn - served hybrid_dyn: AMI misses **+9.9 [+6.3, +13.2]** points (P50 -240 [-800, +720] ms),
ICSI misses **+13.5 [+10.7, +16.3]** points (P50 **-400 [-480, -240] ms**).

**Verdict: killed** (pre-registered: misses worse than the untrained served head on the TS-VAD track with the CI
excluding 0, after two variants; P50 on AMI not below 3.0 s for the head alone). What was learned:
- **The predictive target alone is the wrong signal for eot-bench's ends.** A future-silence predictor legitimately
  predicts silence inside AMI's long within-turn pauses, which the benchmark counts as false cuts; at <= 5 % FC it
  must stay so conservative that 70-80 % of ends are missed. The end-of-turn label (variant b) recovers most of it.
- **The served kernel head's inputs carry the rest.** Without the speaker-conditioned second encoder pass and the
  decoded-text features, a head on the plain top layer loses 10-14 points of misses; it answers 400 ms earlier on
  ICSI at the same FC, but on fewer turns. The lightweight path (no second encoder pass, saving ~0.15 RTF) is not
  worth that on these benches.
- Not tried (next, if latency stays the target): the served head fine-tuned on TS-VAD tracks (keeps its inputs,
  needs the conditioned encoder pass per training window: ~1 CPU-h of features for AMI/ICSI train), and the
  predictive bins as a *gate* on the served head rather than as the decision.

## 2b. `hybrid_fast`: a fast path at the first silence onset (user request)

**Question.** smart-turn decides ~225 ms after speech stops (200 ms Silero stop + inference); hybrid_dyn cannot fire
before 7 frames (560 ms) of silence even at p -> 1. Does reading the head once at silence onset + m (160 / 240 / 320 /
480 ms) and firing on p >= tau_fast buy latency at matched FC?

**Setup** (`scripts/research/hybrid_fast.py`, `runs/hybrid_fast.json`). eot-bench v2, TS-VAD 5 s-print input with
the served head's stored scores; (theta, offset, tau_fast) chosen jointly at <= 5 % per-turn FC by the A.2 cross-fit
(theta in hybrid_dyn's grid, offset 1-6 frames, tau in 30 quantiles of the fast arm + never), ICSI also with the
AMI-fitted point frozen. The Sortformer-input rows need the Sortformer-track scores, which were lost with the scratch
(their rebuild is the ~5 h `tsvad_chain.sh` job): not run. TurnBench: not run (the served head's TurnBench scores on
the TS-VAD input do not exist yet).

| system (TS-VAD input) | AMI miss % | FC % | P10 / P50 ms | open miss % | ICSI miss % | FC % | P10 / P50 ms | ICSI frozen AMI point: miss / FC / P50 |
|---|---|---|---|---|---|---|---|---|
| hybrid_dyn (refit on this grid) | 34.7 | 5.3 | 1680 / 3280 | 19.1 | 21.5 | 3.7 | 1040 / 2080 | 32.9 / 1.7 / 3120 |
| hybrid_fast 160 ms | 33.9 | 6.4 | 1600 / 3200 | 18.2 | 21.5 | 4.6 | 1040 / 2080 | 32.8 / 1.6 / 3120 |
| hybrid_fast 240 ms | 33.8 | 6.2 | 1600 / 3200 | 18.2 | 20.8 | 6.3 | 1040 / 2080 | 33.0 / 1.6 / 3120 |
| hybrid_fast 320 ms | 34.1 | 5.9 | 1600 / 3200 | 18.2 | 20.8 | 6.2 | 1040 / 2080 | 33.0 / 1.6 / 3120 |
| hybrid_fast 480 ms | 34.3 | 5.7 | 1600 / 3200 | 18.2 | 21.0 | 6.2 | 1032 / 2080 | 32.7 / 1.6 / 3120 |

**Verdict: no.** The cross-fit selects tau_fast at 0.98-0.996, where the head's posterior 160-480 ms into a silence
rarely clears it; the fast arm moves P50 by -80 ms on AMI and 0 on ICSI and adds 0.4-2.6 points of FC. No point
beats hybrid_dyn's P50 by >= 300 ms at equal or lower FC, so nothing was pinned in serve.py. The obstacle is the head's
calibration at silence onset (p is not yet high 200 ms into a pause; it rises over the next seconds), not the rule:
a fast path needs a head trained to be confident early (the predictive target of experiment 2 was that attempt).
Recommended defaults stay: two-party product `timeout` 1000 ms on the user's own channel (C), rooms `hybrid_dyn`
(with the TS-VAD input when the user is enrolled: T, experiment 1).

## 2c. Latency / FC frontier (product knob)

`scripts/research/eot_frontier.py`, `runs/eot_frontier.json`; stored score arrays only (no model runs). eot-bench v2,
6 s horizon, the TS-VAD 5 s-print track feeding the served head; per FC budget the point with the lowest P50 whose
realised per-turn FC over all turns is within the budget (in-sample: it shows the trade-off; the cross-fitted rows
elsewhere in this file are the held-out numbers). "> 6 s" = P50 falls on a miss. The Sortformer-track head scores
were lost with the scratch (`tsvad_chain.sh` rebuilds them), so that input is missing here.

Miss at 6 s % / open-floor miss % / P10 / P50 ms:

| FC budget | head (threshold) | hybrid_dyn (threshold x offset) | timeout on P(target) | Silero timeout (any speaker) |
|---|---|---|---|---|
| **AMI dev** | | | | |
| 3 % | 47.1 / 48.3 / 2080 / 5520 | 45.0 / 25.9 / 2160 / 4960 | 68.5 / 61.4 / 4320 / > 6 s (4480 ms) | 74.9 / 34.3 / 2160 / > 6 s (2160 ms) |
| 5 % | 39.2 / 37.7 / 1680 / 4320 | 34.9 / 18.2 / 1760 / 3280 | 63.1 / 55.5 / 3760 / > 6 s (4000 ms) | 72.0 / 27.1 / 1920 / > 6 s (1920 ms) |
| 7.5 % | 27.9 / 23.7 / 1200 / 2800 | 25.8 / 14.0 / 1200 / 2560 | 60.7 / 53.8 / 3520 / > 6 s (3840 ms) | 68.9 / 19.5 / 1680 / > 6 s (1680 ms) |
| 10 % | 24.0 / 21.2 / 1040 / 2480 | **22.4 / 12.7 / 1040 / 2320** | 56.3 / 51.3 / 3280 / > 6 s (3600 ms) | 66.7 / 17.0 / 1520 / > 6 s (1520 ms) |
| 15 % | 18.2 / 17.4 / 720 / 2000 | 16.7 / 8.1 / 800 / 1840 | 49.4 / 39.0 / 2720 / > 6 s (3120 ms) | 61.9 / 8.9 / 1280 / > 6 s (1280 ms) |
| **ICSI held-out** | | | | |
| 3 % | 24.5 / 27.8 / 1200 / 2240 | 23.5 / 14.8 / 1280 / 2320 | 57.9 / 60.7 / 3920 / > 6 s (4000 ms) | 88.6 / 51.8 / 4480 / > 6 s (2080 ms) |
| 5 % | 21.5 / 23.2 / 960 / 2000 | 20.4 / 10.2 / 1120 / 2080 | 53.1 / 56.5 / 3440 / > 6 s (3520 ms) | 84.2 / 37.5 / 2080 / > 6 s (1760 ms) |
| 7.5 % | 16.5 / 17.1 / 800 / 1680 | 16.2 / 10.7 / 880 / 1760 | 44.0 / 43.5 / 2720 / 4240 (2800 ms) | 81.2 / 32.4 / 1760 / > 6 s (1520 ms) |
| 10 % | 14.2 / 13.9 / 640 / 1520 | **13.3 / 7.4 / 720 / 1600** | 32.9 / 27.3 / 2000 / 2560 (2160 ms) | 78.5 / 26.9 / 1520 / > 6 s (1360 ms) |
| 15 % | 11.2 / 9.7 / 560 / 1360 | 10.6 / 6.5 / 560 / 1360 | 23.8 / 14.8 / 1520 / 1840 (1680 ms) | 72.3 / 13.9 / 1200 / > 6 s (1120 ms) |

(timeout rows: the selected timeout_ms in brackets.) **What 10 % FC buys:** a deployment that tolerates one false
cut in ten turns (instead of one in twenty) gets, with the enrolled user's TS-VAD track, **12 fewer missed turn ends on
AMI (34.9 -> 22.4 %) and 7 fewer on ICSI (20.4 -> 13.3 %), with the median answer ~1 s earlier on AMI (3280 -> 2320 ms)
and ~0.5 s earlier on ICSI**; the head-gated rules dominate both timeouts at every budget (a plain timeout on the same
track still misses 33-56 % at 10 % FC).

## 3. Speaker head: crop-level TitaNet distillation with every LibriSpeech-100 identity + the relational loss

**Hypothesis.** The shipped block-4 relational head (AMI within-meeting EER 19.8 % at n = 200, ICSI 7.0 %) was trained
on whole segments with 5000 LibriSpeech utterances in batches of 6. Distilling TitaNet at the crop level (1.5-4 s,
what live binding and voice prints see), with all 251 LibriSpeech train-clean-100 speakers and 128-item batches for
the relational loss, should close more of the gap to TitaNet (12.0 %).

**Setup** (`scripts/research/spk_frame.py`, `runs/spk_frame.json`). Block-4 features of AMI train (930 segments, 45
speakers), ICSI train (3249, 33 speakers) and LibriSpeech train-clean-100 1-16 s (6954 utterances, 251 speakers),
cached once; per segment 3 random crops of 1.5-4 s (27 227 crops); TitaNet-L embedded every crop and every whole
segment (the exact audio). Head = the served SpeakerHead shape (attentive-stats pooling, 192-d); loss = 0.5 cosine +
1.0 relational MSE of the within-batch cosine matrices (student vs TitaNet) + AAM on AMI items (190 classes); batch
128, 70 % crops / 30 % whole segments, sources balanced, AdamW 5e-4 one-cycle, 4000 steps (130 s on MPS). Variants:
warm start from the served head (`c115_warm`) and fresh init (`c115_fresh`). Evaluation = SPK_HEAD.md's protocol
(same dev segments and trials, cosine EER), plus a paired segment bootstrap (500) of the within-meeting EER at n = 200.

| head (block 4, 115M) | AMI n=64 all / within | AMI n=200 all / within | ICSI n=64 within | ICSI n=200 all / within |
|---|---|---|---|---|
| shipped relational (SPK_HEAD.md) | 8.5 / 14.4 | 11.3 / 19.8 | 5.2 | 6.4 / 7.0 |
| crop-level distillation, warm start | 7.9 / 12.3 | 10.5 / **17.1** | 2.6 | 4.5 / **4.4** |
| crop-level distillation, fresh init | 7.9 / 11.5 | 10.7 / **17.4** | 2.5 | 4.2 / **4.0** |
| TitaNet-L (the teacher, BASELINES.md) | 6.6 / 8.2 | 10.9 / 12.0 | 1.0 | 2.2 / 2.1 |

Warm vs fresh (paired, within n = 200): AMI -0.3 [-1.5, +1.4], ICSI +0.5 [-0.1, +1.2] points: the same.

**Verdict: partial.** ICSI dev meets its bar (4.0-4.4 % <= 5 %), AMI does not (17.1 % vs the 12 % bar); the kill line
(AMI not 2 points below 19.8 %) is not crossed (-2.7). The head is a better voice-print embedder than the shipped one
on both corpora at no extra cost (same 0.5 M head on the same tap), but the AMI same-room gap to TitaNet stays ~5
points, as SPK_HEAD.md predicted for any head on the frozen block-4 features. Not transplanted into the served model
yet: the TS-VAD head (experiment 1) was trained on the shipped head's embedding space, so a new speaker head needs the
TS-VAD head retrained on its prints first.

## 6. nemotron-speech-streaming-en-0.6b as the GPU streaming core

**Hypothesis.** On GPU the 0.6B's 5x CPU cost stops mattering, and its card's meeting WER (14.7 % AMI at 160 ms)
would fix the 22-24 % streaming partials. Adoption rule (the user's): every category must improve, none may be worse
beyond noise (table in the pre-registration).

**Setup.** `scripts/research/core_0p6b.py`: re-import of `data/nemo/nemotron-speech-streaming-en-0.6b.nemo` (618.5 M
parameters, 651 of 653 tensors loaded, 264 zero biases, as in ENC_0P6B.md) to
`/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_speech_streaming_en_0.6b.afm` (2.19 GB, 915 tensors read
back identical). WER: `hybrid_asr.py lookahead` (masked offline forward = cache-aware streaming, greedy RNNT, batch 4),
the AMI-200 / ICSI-200 / LibriSpeech-200 sets, `normalize_text` (Whisper normaliser alongside), paired utterance
bootstrap (1000). VAD: the served VAD head's recipe trained identically on each core (softmax mix over all blocks +
Linear-SiLU-Linear 64, 3000 steps on the 300 seeded AMI train diar windows), scored on the BASELINES 64 x 20 s AMI dev
and ICSI dev windows, paired window bootstrap of F1. First partial: each AMI-200 / ICSI-200 segment streamed through
`StreamingSession` at [70,1] in 160 ms pieces, audio time at which the first token exists. Cost: `StreamingSession`
over 60 s of AMI at 160 ms chunks and batch-4 offline decoding, CPU 2 threads and MPS (GPU proxy), per-process RSS.

**Results so far** (`runs/hybrid_asr.json["core_0p6b"]`).

| category | 115M core | 0.6B core | 0.6B - 115M [95 % CI] | verdict |
|---|---:|---:|---|---|
| WER AMI-200 [70,1] | 24.43 % | **11.16 %** | **-13.27 [-15.48, -11.23]** | better |
| WER AMI-200 [70,13] | 23.02 % | **9.81 %** | **-13.21 [-15.48, -11.19]** | better |
| WER ICSI-200 [70,1] | 27.29 % | **14.35 %** | **-12.94 [-15.11, -10.90]** | better |
| WER ICSI-200 [70,13] | 24.78 % | **12.39 %** | **-12.39 [-14.95, -10.13]** | better |
| WER LibriSpeech-200 [70,1] | 2.29 % | 2.20 % | -0.09 [-0.46, +0.28] | equal (not shown to improve) |
| WER LibriSpeech-200 [70,13] | 1.92 % | 2.03 % | +0.11 [-0.23, +0.47] | equal (not shown to improve) |
| VAD F1 AMI dev (same head recipe) | 0.9329 | **0.9398** | **+0.0069 [+0.0018, +0.0120]** | better |
| VAD F1 ICSI dev | 0.9039 | **0.9079** | **+0.0040 [+0.0004, +0.0071]** | better |
| first partial, AMI-200 (median, audio time) | 1.12 s | 1.12 s | 0.00 [-0.16, +0.16] (mean 1.07 vs 1.15 s) | equal |
| first partial, ICSI-200 | 1.12 s | 1.12 s | 0.00 [0.00, +0.16] (mean 1.10 vs 1.09 s) | equal |
| speaker EER within meeting, AMI n = 200 (identical crop-level recipe, fresh heads; 115M block 4, 0.6B block 11) | 17.4 % | 16.2 % | -1.2 [-4.1, +1.9] | equal |
| speaker EER within meeting, ICSI dev n = 200 | 4.0 % | 2.9 % | -1.1 [-2.7, +0.7] | equal |
| eot-bench miss 6 s, AMI all, head alone (identical head recipe, TS-VAD track) | 55.3 % (FC 5.1) | 56.5 % (FC 4.5) | +1.1 [-1.5, +4.0] | equal |
| eot-bench miss 6 s, AMI open, head alone | 61.3 % | 57.6 % | -3.8 [-10.3, +2.6] | equal |
| eot-bench miss 6 s, AMI all, hybrid_dyn | 44.1 % (FC 5.8, P50 2960) | 46.9 % (FC 4.9, P50 3920) | **+2.8 [+0.1, +5.3]** | **worse** |
| eot-bench miss 6 s, AMI open, hybrid_dyn | 24.7 % | 23.4 % | -1.2 [-6.3, +3.8] | equal |
| eot-bench miss 6 s, ICSI all, head alone | 34.0 % (P50 1520) | **29.0 %** (P50 1360) | **-4.9 [-7.1, -2.7]** | better |
| eot-bench miss 6 s, ICSI open, head alone | 49.5 % | **36.3 %** | **-13.2 [-19.5, -6.9]** | better |
| eot-bench miss 6 s, ICSI all, hybrid_dyn | 32.2 % | **26.9 %** | **-5.3 [-7.4, -3.1]** | better |
| eot-bench miss 6 s, ICSI open, hybrid_dyn | 37.9 % | **25.8 %** | **-12.1 [-17.8, -6.9]** | better |
| TurnBench recall at FP <= 0.10 | | | | not measured (the verdict below cannot change) |

Whisper-normalised WER tells the same story (AMI [70,1] 20.6 -> 10.4 %, -10.2 [-12.4, -8.3]). The 0.6B at [70,1] is
already at the per-turn TDT v3 final's level on AMI (9.7 %, HYBRID_ASR.md). The served 115M VAD head (trained on more
windows) scores 0.949 / 0.900; the rows above are the identical-recipe comparison the rule asks for.

Cost (60 s of AMI, 160 ms chunks, `StreamingSession`, no server fast paths; per-process RSS after load and run):

| | CPU 2 threads: ms per 160 ms chunk (p50 / p95), RTF | CPU RSS | MPS: ms per chunk, RTF | MPS memory | offline batch-4 RTF CPU / MPS |
|---|---|---|---|---|---|
| 115M | 83.6 / 88.1, 0.52 | 1.33 GB | 12.8, 0.08 | 1.0 GB RSS + 1.5 GB MPS | 0.009 / 0.006 |
| 0.6B | 266.1 / 274.3, **1.68** (not real time) | 3.40 GB | 37.8, **0.24** | 0.9 GB RSS + 3.2 GB MPS | 0.023 / 0.012 |

Turn rows: the experiment-2 variant-(b) head (UCTurnHead with the end-of-turn output, top layer + the TS-VAD track,
3000 steps, identical recipe, data and tracks) trained once on each core's top layer; eot-bench v2 cross-fitted at
<= 5 % per-turn FC, paired window bootstrap (`runs/tsvad_turn_cores.json`). Both are weaker than the served kernel
head (section 2); the comparison is between the cores under one recipe, as the rule asks.

**Verdict: not adopted** under the user's rule. Worse beyond noise: **eot-bench missed ends on AMI with hybrid_dyn
(+2.8 [+0.1, +5.3] points)**, with a later median answer (3920 vs 2960 ms). Equal within noise (not improved):
LibriSpeech WER at [70,1] and [70,13], first-partial latency on AMI and ICSI, within-meeting speaker EER on AMI and
ICSI, and the AMI eot-bench rows for the head alone and the open-floor ends. Better: meeting WER (AMI and ICSI, both
lookaheads, 12-13 points), VAD F1 (AMI and ICSI), and every ICSI eot-bench row (-5 points all, -12 to -13 open).
TurnBench was not run: the AMI turn row already fixes the verdict. Why: the 0.6B's gain is in what is said, not in
when a turn ends; the turn head on its top layer is as good as the 115M's on AMI and a little worse with the Silero
arm, and better on held-out ICSI. Cost is visible either way: 3.2x the CPU per chunk (not real time on 2 threads),
real time on MPS at RTF 0.24 (3.2 GB of GPU memory).

What it means for the product: the recommended use of the 0.6B is as a **transcript** model (streaming partials
at ~11 % AMI WER, the TDT v3 final's level, at 160 ms latency on a GPU) next to the 115M core for the live heads,
not as the core; or its accuracy moved into the 115M by distillation (experiment 7).

## 7. Distilling the 0.6B into the 115M core (package delivered; GPU run pending)

**Hypothesis.** The 0.6B streaming model is 13 points better on meetings at the same 160 ms latency (section 6) but
3x the compute; training the 115M encoder + RNNT towards it (teacher transcripts as targets + encoder-output
regression) on the AMI + ICSI train meetings with a LibriSpeech anchor, could bring AMI streaming WER from 24.4 % to
<= 17 % at the 115M's cost, keeping every other head.

**Delivered** (`scripts/research/distill_0p6b_to_115m/`, committed): README.md (exact steps, per-stage GPU hours),
setup.sh (venv + CUDA torch + repo extras, teacher download / import, AMI 136 + ICSI 70 train meetings, LibriSpeech
train-clean-100 + test-clean), distill.py (stages manifest / evalsets / teacher / train /
eval / heads; KD losses, layer-wise LR decay 0.9, KL anchor on LibriSpeech, WER gate every 500 steps that restores
the last gate-passing state and stops, resumable checkpoints every 500 steps, controls `--control` (no teacher, same
budget)). Estimated cost on one A100: ~6 GPU hours in total (teacher 0.4-0.8 h, KD 2.5 h, control 2.0 h, eval
0.5 h), ~60 GB disk.

**Smoke test (laptop, MPS, ~10 min in total).** Manifest capped at 1.0 h (AMI 0.23 h, ICSI 0.42 h, LibriSpeech 0.36 h);
eval sets rebuilt from the data root are **identical** to HYBRID_ASR.md's (all 200 references of AMI / ICSI /
LibriSpeech match); teacher cache 55 s for the hour (91 MB fp16 at [70,1]; transcripts sensible, e.g. reference
"and wasn't really satisfied by what i saw" / teacher "... satisfied with what i saw"); KD run 80 steps at lr 2e-5
(all loss terms active: RNNT on references and teacher text, encoder cosine 1.00 -> 0.97, KL anchor on LibriSpeech
batches; gate 2.26 -> 2.30 -> 2.26 %, passes), control run 40 steps (no teacher terms), eval and heads stages end to
end (served VAD head on the student: AMI 0.949 / ICSI 0.900, as served). **WER gate check:** at lr 1e-2 the gate
WER jumps 2.26 -> 100 % at step 20, `GATE FIRED`, the step-0 state is restored and saved (a first version restored
the last periodic checkpoint, which had not passed the gate; fixed to keep a separate gate-passing `good.pt`); at
lr 2e-4 with a 20-utterance gate it fired on a 0.5-point wobble, so the package gates on 100 utterances.

**Corpus.** AMI + ICSI train plus the LibriSpeech train-clean-100 anchor only (~170 h; the user withdrew YODAS v3,
which had been added and then removed; nothing from it was downloaded).

## Appendix 0. The earlier scout ranking (superseded by the plan above)


Owner: the improvement-scout agent. Constraints (from the user): the 115M streaming encoder stays the always-on core
(no bigger always-on model); the product knows its target speaker, usually on its own audio channel, and knows its own
TTS timeline; accuracy should be strong everywhere. Machine rules: CPU 2 threads, < 10 min per process; one MPS
training process on the machine; >= 10 GB free disk (17 GB free at the start of this work).

Ranking = expected movement of a scorecard row per GPU-hour, discounted by the chance the row moves and by overlap with
other agents' work. Every item names the row, the control, the kill criterion and the cost.

### Ranked list

| # | candidate | scorecard row it moves | control | kill criterion | cost | status |
|---|---|---|---|---|---|---|
| 1 | **User-channel predictive turn head (UC head), live.** A new causal head on the frozen encoder's top layer computed on the *user's own channel* (the pass the ASR already makes: zero extra encoder cost), plus two activity columns (user Silero, agent activity from the TTS state / other channel), duration and energy inputs; multi-horizon future-activity targets for both parties (VAP-style bins, independent and joint). Trained on otoSpeech train (100 conversations, per-channel audio) from cached encoder features, so training is CPU minutes, not GPU hours. Decision = predictive trigger (optionally OR the user-channel Silero timeout). Served as `serve.py --turn-input two_channel`. | TurnBench dev (official scorer) recall / fp / P10 / P50; product dead air and cut-ins on the user-channel clips (E2E_FINAL oto / TurnBench rows) | head (c) predictive OR Silero on held-out TurnBench halves: 0.862 / 0.092 / P50 1137 ms (alone 0.804 / 0.091 / 961 ms); Dp offline dead air 786 / 1080 ms median with 21 / 19 cut-ins (runs/e2e_final.json) | not better than (c) at matched protocol: held-out-halves P50 at fp <= 0.10 with recall >= 0.83 must be < 1037 ms (100 ms better than 1137) or recall + 0.02 at equal P50; live: not better than Dp offline on dead air at <= Dp's cut-ins. Target: P50 < 900 ms, fp <= 0.10, recall >= 0.83 | ~1 CPU-h feature caching (chunked), ~10-20 CPU-min per training variant, 0 GPU-h; serve + tests ~0.5 day | **running first** |
| 2 | Prosody / hazard inputs for earlier firing (pitch, energy slope, hazard-shaped waits) | TurnBench P10 / P50 | #1's best head | no held-out P50 gain >= 50 ms at matched fp | folded into #1 as input ablations (energy on/off) because features are cached; pitch only if energy helps on TurnBench | folded into #1 |
| 3 | Speaker head: more identities (LibriSpeech train-clean-360/other speakers) with the relational TitaNet loss | Speaker EER within meeting (AMI 14.4 %, ICSI 5.2 %) | shipped block-4 relational head | AMI within-meeting EER not below 12 % (n = 200) | TitaNet teacher caching ~1-2 CPU-h, training ~30-40 min MPS | queued; lower product value when the user has their own channel. SPK_HEAD.md's AAM control with +251 LibriSpeech ids did not help, so the prior for "more ids" alone is weak |
| 4 | Enrollment quality (clean-span, VAD-masked, >= 3 s voice prints) | enrollment agreement on AMI | after_agent_arm binding | agreement <= 85 % | ~0.5 day CPU | **skipped: covered** by the TS-VAD agent (OUTSIDE.md 6.2 / scripts/tsvad.py, whose own kill criterion is agreement >= 85 % with >= 3 s enrollment); also near-moot when the user has their own channel |
| 5 | Meeting ASR: LoRA / adapter on the top encoder blocks merged into the weights, AMI + ICSI with a LibriSpeech anchor, WER gate | ASR AMI dev WER 20.63 % | the frozen served encoder | AMI WER gain < 1.5 pts, or LibriSpeech WER + > 0.2 pts, or any runtime change | several MPS hours (encoder backward), plus a WER gate pass | **held**: IMPROVE_115M Part B belongs to the fix agent; research/archive/ISSUES_FIXED.md and research/IMPROVE_115M.md do not exist yet, so coverage cannot be confirmed; not started to avoid two agents training the same thing on the one GPU |
| 6 | Calibrated eot field (isotonic on dyadic data) for the served head | none directly (client-side thresholds) | raw posterior | no change in any policy's held-out numbers | CPU minutes | backlog; DYADIC §7 found calibration moves P50 by at most ~0.24 s and destabilises fp across halves |

Why #1 is first: it is the only candidate that targets the product's actual input (own channel + known TTS state), it
attacks the largest measured product gap (dead air 1.25-2.3 s live vs Dp's 0.79-1.08 s offline headroom), and with
cached features it costs no GPU time. #5 has the largest scorecard gap (AMI WER 20.6 vs 14.1 %) but is owned elsewhere
and is the most expensive and riskiest (encoder fine-tuning has tripped the WER gate every time on this machine).

### A0.1 User-channel predictive turn head

Not run to completion: a reboot killed it after 2 of 38 TurnBench feature files. Its code (scripts/research/uc_turn_head.py, scripts/research/uc_stream.py, scripts/research/uc_turn.py, tests/test_uc_turn.py, tests/test_uc_stream.py; tests pass) is kept: experiment 2 reuses `UCTurnHead` as the TS-VAD-fed predictive head.
