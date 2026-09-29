# Single-model mode: everything from the one 115M checkpoint (known user)

2026-09-29. Request: "Can we do single model?" Product decision given during the work: "I don't need general
diarization, I need target diarization." So the question here is whether the served 115M model alone can run a voice
agent for one known user: no NVIDIA diarizer, and no second ASR model. Part A makes that a documented preset
(`audioforge-serve --mode single`, docs/CONFIGURATION.md §13) and measures it. It also runs two experiments: A1, a
faster turn rule on the target track, and A2, voice-sample quality. Part B prepares the GPU run that would make the one
model better at exactly this job (`scripts/research/single_model_distill/`).

Numbers: `runs/single_model.json` (`table`, `a1`, `a2`), `runs/tsvad_enroll.json`, and the stored runs each row names.
Machine rules held: every model run went through `scripts/dev/gate.sh` (2 threads), one heavy job at a time, calls
under 10 minutes, and scratch on the SSD.

**Update (2026-09-29, user decision: "I need target diarization").** `--mode single` is now the default of
`audioforge-serve`, `audioforge.load()` and `audioforge-bench`. The two-model stack is `--mode room` (general
diarization with Nemotron-3). `audioforge-download` fetches the two head files by default. The `voiceprint` message
returns the print for storage.

One finding from the new quickstart output: on the bundled clean LibriSpeech clip, with speaker A's 10 s stored print,
the TS-VAD head also takes speaker B for the user (P(user) 0.8-1.0 on B's words), so both finals read `speaker=0`.
The head was trained on 24 meetings and has not seen clean read speech of different speakers. Part B's LibriSpeech
mixtures target exactly this.

The quickstart therefore now uses an in-domain clip: `examples/audio/two_party_call_16s.wav`, 16 s of an otoSpeech
call (CC BY 4.0), with the user's print from 10 s of the user elsewhere in the call. On the default server the
user's question ends in a `turn_end` 1.4 s after its last word with `speaker=0`, and the other party's answer comes
out as `speaker=1`. The clip was picked by a search over the 16 otoSpeech E2E clips for 14-20 s excerpts that hold a
complete user turn and other-party speech: 20 candidates, 4 correct on every final. So it is a demonstration, not a
measurement; the table above is the measurement. The LibriSpeech clip stays for room mode.

## What `--mode single` is

`audioforge-serve --mode single` (config file: `mode: single`) adds `--turn-input tsvad --diar-off --lid head --enroll
after_agent_arm --dyn-wait-ms 2000,960` and the TS-VAD head file. It loads **no diarizer, no TitaNet, no AmberNet
and no final-ASR worker**. It refuses `--diar`, `--diarizer`, `--final-asr`, `--lid ambernet` and
`--diar-embed titanet`. Everything runs from `stage1_served.afm`: the streaming RNNT (transcript), the VAD head, the
turn head (second, speaker-conditioned pass), and the speaker head on block 4. Two small head files come on top: the
TS-VAD head on block 4 (0.26 M, `tsvad_spk.pt`) and the distilled LID head on blocks 8-12 (0.92 M, `lid_distill.pt`).

The user's voice print is 192 numbers from the served speaker head. It comes from one of two places:
- an `enroll` message carrying an embedding, which is now accepted under every `--enroll` mode when the TS-VAD path
  is on;
- live, the first `--tsvad-print-s` (5) seconds of speech after `agent_end`.

The TS-VAD track [P(user), P(other), 0, 0] is what `frame.speakers` carries and what the turn head reads. The primary
is always column 0. The one other model file loaded is Silero VAD (2.3 MB ONNX, MIT), and only when present: it is
the silence arm of `hybrid_dyn`, the rule A1 recommends.

Code changes, all behind the preset or the flag; the default mode is unchanged:
- `Engine` runs without a diarizer object when `--diar-off` has no `--diar`.
- The launcher preset and the matching `audioforge.load(mode="single")` and `audioforge-bench --mode single`.
- `--dyn-wait-ms CAP,FLOOR` (hybrid_dyn's silence wait).
- The column binder is skipped under `--diar-off` (column 0 is the user by construction).
- `ready.diar_config` reads `"off"` under `--diar-off`.
- Tests: `tests/test_single_mode.py` (preset flags, config-file key, refusals, no second model loaded, the bundled
  clip streamed end to end over the socket on tiny models with a stored print and with the live print).

## The table

"Single" is `--mode single`. "Default" is the product default: Nemotron-3-Diarization + the 115M streaming model,
`timeout` 1000 ms (E2E system CN), optionally `--final-asr tdt_v3`. "Pipecat" / "LiveKit" are their default local
stacks as measured in E2E_FINAL: A = Silero + smart-turn v3 + Whisper small; B = Silero + LiveKit EOU + Whisper small.
Lower is better except F1 and accuracy.

| row (data, n) | single (`--mode single`) | product default (Nemotron-3 + TDT v3) | Pipecat default | LiveKit default | source |
|---|---|---|---|---|---|
| turn-end misses, AMI dev offline (974 ends, 6 s, ≤ 5 % per-turn false cut-offs, cross-fitted) | **34.2 %** [31.1, 37.5] (hybrid_dyn; FC 5.5 %); 39.3 % hybrid; 40.0 % head alone. Needs a 5 s stored print | 74.8 % timeout / 61.9 % shipped hybrid, on the Sortformer v2 column (Nemotron-3 not scored offline) | 72.6 % (smart-turn + Silero timeout) | 74.2 % (EOU + timeout) | `runs/improve_115m.json`, `runs/baselines_turn.json` |
| turn-end misses, held-out ICSI offline (1312 ends, same protocol) | **18.7 %** [16.6, 21.0] (FC 6.2 %); 20.4 % hybrid | 85.3 % timeout / 68.5 % hybrid (Sortformer v2 column) | 84.4 % (Silero timeout; smart-turn not run on ICSI) | not measured | `runs/improve_115m.json`, `runs/baselines_turn_icsi.json` |
| live, 37 clips / 69 sessions (Pipecat): missed within 3 s / 6 s | **S (A1 rule): 37.7 % / 30.9 %**; T (served hybrid_dyn point): 45.3 % / 38.6 % | 34.1 % / 28.3 % (CN) | 58.7 % / 35.4 % | 41.3 % / 34.1 % | `scratch/e2e_tsvad/runs`, `runs/e2e_final.json` |
| live cut-ins (false cut-offs) per session | **S: 0.67**; T: 0.52 | 1.07 | 1.99 | 1.12 | same |
| live median dead air (answered ends) | S: 1388 ms; T: 2001 ms | 1292 ms | 1830 ms | 1350 ms | same |
| first words (first text after the user's first onset, median; the 64 two-party sessions) | S: 1081 ms; T: 1076 ms | 1075 ms | 2812 ms | 5120 ms | same |
| streaming WER, AMI-200 / LibriSpeech-200 | 24.4 % / 2.29 % (the same model: identical transcript) | the same streaming; final per turn with TDT v3: **9.7 % / 2.03 %** | Whisper small 21.2 % / 2.42 % | Whisper small 21.2 % / 2.42 % | `runs/final_asr.json`, `runs/hybrid_asr.json` |
| WER on the live clips (TurnBench + AMI refs) | S: 23.2 %; T: 23.2 % | 24.1 % streaming (TDT v3 finals not run live) | 23.5 % | 19.3 % | `runs/e2e_final.json`, stored records |
| LID, FLEURS-17 test (2550): 2 s / full utterance | **91.0 % / 97.8 %** (head, VAD-gated, the served path) | off by default; opt-in AmberNet 95.1 % / 99.5 % | not part of the stack (Whisper small forced to en) | same | `runs/lid.json` |
| speaker tracking F1 of the user's track (primary = target, 5 s print, all frames): AMI / ICSI | **0.743 / 0.882** | Sortformer v2 column bound by the same print: 0.629 / 0.690 (TitaNet binding 0.657 / 0.685); oracle column 0.799 / 0.794; Nemotron-3 not scored this way | no speaker tracking | no speaker tracking | `runs/improve_115m.json` frame (IMPROVE_115M A.1) |
| server RTF, CPU 2 threads, live (median over sessions) | **0.33 (max 0.34)** | 0.635 (+ TDT v3 worker: RTF 0.065 per turn in its own process) | 0.43 CPU s per audio s | 0.28 | live records, `runs/e2e_final.json` |
| server peak RSS | **1156 MB** | 1485 MB (+ ~2.5 GB TDT v3 worker) | 2372 MB (driver) | 2802 MB (driver) | same |

Reading the table:
- **In meetings, with a stored print, single mode is the best turn detector measured.** It is 34 vs 62-75 % missed on
  AMI and 19 vs 68-85 % on ICSI at the same false-cut budget. The reason is that it tracks the user, not "the most
  active column". The tracking F1 row says the same thing: 0.74 / 0.88 against 0.63-0.69 for a diarizer column bound
  by the same print.
- **On two-party calls it answers later than the default.** That was T's result (45 vs 34 % missed within 3 s), and
  A1 below narrows it.
- **It is cheaper.** No diarizer pass and no diarizer in memory (see the RTF and RSS rows).
- **The transcript is the streaming one.** 24.4 % WER on AMI, against 9.7 % with the default's per-turn TDT v3 pass.
- **LID** is 3.9 points behind AmberNet at 2 s. It runs off the same encoder pass at 1/100 of AmberNet's cost.

**What single mode cannot do:**
- It cannot label everyone in a room. It knows "the user" and "someone else", not who spoke when among other people,
  and not how many there are. Use the default mode with `--diar-labels registry` for that.
- It cannot give a meeting-grade final transcript. Use `--final-asr tdt_v3` in the default mode, at the cost of a
  2.5 GB second model.
- It does not recover well from a bad voice print. A2 shows that the live print after `agent_end` costs 16-41 points
  of misses in meetings.

## A1. A faster turn rule on the target track (two-party calls)

**Question.** Live T (hybrid_dyn at the served point on the TS-VAD track) missed 45 % of turns within 3 s against 34 %
for the default. Is there a rule on the same track that matches the default's 3 s misses without more cut-ins?

**Method** (`scripts/research/single_model.py replay / score`). This is the exact server session in-process
(`serve.Session`, the single-mode engine, stored 5 s print sent at connect) over the same 37 clips and pads as the
live test (E2E_FINAL's clips, rebuilt by `e2e_tsvad.py`). One run per variant covers the 32 two-party user-channel
sessions (TurnBench user 16, otoSpeech user 16). The server's `turn_end` audio times plus Pipecat's measured delivery
offset (76 ms) stand in for the response moments. They are scored with E2E_FINAL's scorer: miss = no response within
3 s of a user turn end; cut-in = a response while the user keeps talking. The replay matches the live server's
`turn_end` times (tb_20 / tb_21 / tb_22: identical, one end 160 ms apart). Offline scores are **not** identical to
live Pipecat scores: offline T misses 20.2 % vs 26.6 % live and cuts in 0.81 vs 0.53 per session, because Pipecat's
aggregator drops or merges some decisions. So the product default was **replayed the same way** (offline CN,
Nemotron-3, E2E_FINAL's flags) and variants are compared against it, offline to offline, with paired clip bootstraps.

Variants: `hybrid_dyn` waits clamp(T0 − A·p, 2, T0) + 3.2 frames of Silero silence, where p is the head's posterior.
The served rule is T0 75 and A 55: 6.3 s at p = 0, 1.9 s at p = 1.

| variant (32 two-party user sessions, 109 ends, offline) | missed 3 s | missed 6 s | cut-ins / session | dead air median | Δ misses 3 s vs default [CI] | Δ cut-ins vs default [CI] |
|---|---|---|---|---|---|---|
| **default CN** (Nemotron-3, timeout 1 s), replayed | **5.5 %** | 5.5 % | **1.19** | 1206 ms | | |
| T: hybrid_dyn, served wait (the live T) | 20.2 % | 15.6 % | 0.81 | 1984 ms | +14.7 [+7.3, +21.8] | −0.38 [−0.88, +0.19] |
| plain 1 s timeout on P(user) < 0.5 | 25.7 % | 20.2 % | 0.69 | 1738 ms | +20.2 [+10.0, +33.0] | −0.50 [−0.81, −0.16] |
| 0.8 s timeout on P(user) | 19.3 % | 17.4 % | 0.97 | 1432 ms | +13.8 [+3.6, +25.9] | −0.22 [−0.59, +0.16] |
| head OR 0.8 s timeout on P(user) | 18.4 % | 16.5 % | 1.28 | 1418 ms | +12.8 [+2.8, +24.6] | +0.09 [−0.44, +0.75] |
| hybrid_dyn, cap 40 frames, slope kept (p = 1 wait at the 2-frame floor) | 7.3 % | 6.4 % | 3.31 | 573 ms | +1.8 [−4.7, +8.9] | +2.13 [+1.41, +2.94] |
| hybrid_dyn, cap 25, slope kept | 4.6 % | 4.6 % | 3.53 | 547 ms | −0.9 [−6.8, +5.3] | +2.34 [+1.63, +3.13] |
| hybrid_dyn, cap 3.2 s, p = 1 wait 1.6 s | 14.7 % | 11.9 % | 0.75 | 1967 ms | +9.2 [+2.8, +15.6] | −0.44 [−0.91, +0.13] |
| hybrid_dyn, cap 2.4 s, p = 1 wait 1.6 s | 13.8 % | 11.0 % | 0.78 | 1962 ms | +8.3 [+1.9, +14.8] | −0.41 [−0.88, +0.13] |
| **hybrid_dyn, cap 2.0 s, p = 1 wait 0.96 s (`--dyn-wait-ms 2000,960`)** | **10.1 %** | 8.3 % | **1.06** | 1362 ms | **+4.6 [0.0, +9.3]** | **−0.13 [−0.53, +0.34]** |
| hybrid_dyn, cap 1.6 s, p = 1 wait 0.64 s | 10.1 % | 8.3 % | 1.44 | 1062 ms | +4.6 [−0.9, +10.9] | +0.25 [−0.16, +0.72] |

**Pick: `--dyn-wait-ms 2000,960`.** This is hybrid_dyn with a 2.0 s silence cap at p = 0 and 0.96 s at p = 1. No
variant fully matches the default's 3 s misses without more cut-ins. This one comes closest: +4.6 points (CI from 0.0
to +9.3), no more cut-ins (−0.13, CI through 0), 156 ms more dead air, and 10 points fewer misses than T. The two
rules that match the misses (caps 25 / 40 with the slope kept) triple the cut-ins. The plain timeouts on P(user) are
worst on misses, because the TS-VAD track lingers above 0.5 after the user stops (it misses less speech than it
releases). `--mode single` now sets this wait.

**Live** (`e2e_tsvad.py queue --system S`: `--mode single`'s exact server flags, no diarizer loaded, the LID head,
hybrid_dyn with `--dyn-wait-ms 2000,960`, the stored 5 s print sent at connect; Pipecat, all 37 clips / 69 sessions,
the same pads and scorer as T / C / CN):

| scope (Pipecat, live) | single S: missed 3 s / 6 s | cut-ins / session | dead air median | default CN | S − CN, paired [95 % CI]: misses 3 s / cut-ins / dead air | S − T (the earlier single-mode rule) |
|---|---|---|---|---|---|---|
| all 69 sessions (223 ends) | **37.7 % / 30.9 %** | **0.67** | 1388 ms | 34.1 % / 28.3 %, 1.07, 1292 ms | +3.6 [0.0, +7.6] / **−0.41 [−0.67, −0.17]** / +96 ms [+61, +146] | −7.6 [−11.9, −3.5] / +0.14 [−0.04, +0.33] / −613 ms |
| two-party, user channel (32) | 17.4 % / 11.9 % | 0.69 | 1382 ms | 12.8 % / 10.1 %, 1.03, 1272 ms | +4.6 [+1.7, +8.9] / −0.34 [−0.59, −0.09] / +110 ms | −9.2 [−16.2, −2.7] / +0.16 / −619 ms |
| two-party, mono mix (32) | 58.7 % / 51.4 % | 0.72 | 1406 ms | 55.1 % / 47.7 %, 1.22, 1311 ms | +3.7 [−2.8, +10.8] / −0.50 [−0.94, −0.06] / +95 ms | −6.4 [−11.8, −1.8] / +0.13 / −559 ms |
| AMI meeting windows (5) | 20 % / 0 % | 0.20 | 1682 ms | 40 % / 0 %, 0.40, 1903 ms | −20 [−60, 0] / −0.2 [−0.6, 0] / −221 ms | 0 / +0.2 / −539 ms |

Server (69 sessions, CPU 2 threads, no diarizer loaded): RTF median 0.33 (max 0.34) against 0.635 for CN, and peak
RSS 1156 MB against 1485 MB (CN) or 1616 MB (T, which still loaded Sortformer). First words 1081 ms (CN 1075). WER on
the live clips 23.2 % (CN 24.1 %: the same streaming model; the difference is segmentation).

**Verdict.** The pre-registered aim was to match the default's 3 s misses without more cut-ins. It is **almost met
live**: +3.6 points of 3 s misses with the CI reaching 0 (+0.0 to +7.6). Cut-ins are **38 % fewer** (−0.41 per
session, the CI excludes 0), dead air is +96 ms, and the server compute is half. Against the earlier single-mode rule
T it is better on every axis except cut-ins (+0.14, CI through 0): 7.6 points fewer misses and 0.6 s less dead air.
On the user channel of two-party calls the gap to the default is +4.6 points (CI excludes 0), so the default still
answers slightly more turns in time there. In meetings (5 AMI windows, too few to conclude) S is not behind.

## A2. How good must the voice sample be? (IMPROVEMENTS.md experiment 4)

**Pre-registered** (IMPROVEMENTS.md, written 2026-09-28 before any of this). Bar: ≤ 2 points of misses (6 s) lost
against the 5 s offline print for the recommended live rule. Kill: every live rule loses > 5 points.

**Method** (`scripts/research/tsvad_enroll.py tracks / frame`; `tsvad.py scores / report`). The eot-bench v2 windows
are used, AMI dev 974 and ICSI held-out 1312, with the primary as the target. Tracks come from the served TS-VAD head
with these prints:
- vp1.5 / 3 / 10: 1.5 / 3 / 10 s of the primary's single-speaker speech from elsewhere in the meeting (5 s = A.2's
  print);
- arm1.5 / 5: live, the first 1.5 / 5 s of speech after the last other-speaker end before the primary's onset (the
  `agent_end` stand-in; speech = the head's own no-print output), as `--enroll after_agent_arm` takes it;
- mem5: armed the same way at the primary's previous turn and kept for the session.

The served turn head is scored on each track, cross-fitted at ≤ 5 % per-turn false cut-offs, 6 s horizon, with paired
bootstraps against the 5 s print.

| print | tracking F1 AMI / ICSI | misses AMI, hybrid_dyn (Δ vs 5 s [CI]) | misses ICSI, hybrid_dyn (Δ vs 5 s [CI]) |
|---|---|---|---|
| stored 1.5 s | 0.672 / 0.822 | 56.3 % (+22.1 [+19.2, +25.0]) | 38.7 % (+20.0 [+17.6, +22.4]) |
| stored 3 s | 0.720 / 0.872 | 47.5 % (+13.2 [+10.8, +15.8]) | 19.9 % (+1.2 [−0.1, +2.4]) |
| **stored 5 s** (A.2) | **0.743 / 0.873** | **34.2 %** | **18.7 %** |
| stored 10 s | 0.754 / 0.873 | **27.8 % (−6.5 [−8.5, −4.4])** | 19.6 % (+0.9 [−0.1, +1.9]) |
| live, 1.5 s after agent_end | 0.637 / 0.729 | 52.6 % (+18.4 [+15.3, +21.3]) | 41.4 % (+22.7 [+19.8, +25.5]) |
| live, 5 s after agent_end | 0.634 / 0.695 | 50.5 % (+16.3 [+12.4, +19.8]) | 59.5 % (+40.8 [+37.6, +43.8]) |
| session memory (previous turn's live 5 s) | 0.555 / 0.670 | 66.8 % (+32.6 [+29.4, +35.8]) | 73.5 % (+54.8 [+51.8, +57.8]) |

The ICSI 5 s F1 here is 0.873 against 0.882 in IMPROVE_115M A.1: the same print and head, but this driver's frame
accounting; the AMI value reproduces exactly (0.743). The turn rows use A.2's stored tracks, so they are unaffected.

**Verdict: the bar is not met, and the kill criterion is met for the live rules.** Every live print loses 16-55
points. Stored prints are what works:
- **5 s is the minimum.** 3 s loses 13 points on AMI and 1 on ICSI; 1.5 s loses 20-22.
- **10 s helps in meetings.** −6.5 on AMI, equal on ICSI.

Why the live prints fail: in meetings the first seconds of speech after an agent end often contain someone else, or
overlap. The print then describes a mixture. That gives a low miss rate but a false-alarm rate of 0.47-0.54, so the
track fires on everyone. Session memory is worse, because the armed print from an earlier turn is contaminated too.

Product guidance:
- Store the user's print (≥ 5 s, better 10 s, of clean speech: an onboarding prompt) and send it with
  `{"type": "enroll", "embedding": [...]}`.
- Treat the live print after `agent_end` as a fallback. On a clean two-party user channel the live print is likely
  much better than these meeting numbers, but that was not measured separately here.
- Not done: gating the live print on "only one voice" (the P(other) output or the VAD head), and refreshing a stored
  print.

## Part B. The true single model: preparation (no GPU run)

`scripts/research/single_model_distill/` (README, `preflight.sh`, `setup.sh`, `single_distill.py`, `smoke.sh`), for
the user's two RTX 5090s. It is refocused per the product decision: **no diarization teacher**, and target-speaker +
turn first.

- **S1 (heads only).** The TS-VAD head and the turn head (with its speaker kernels) are trained on more speakers and
  data: 206 AMI + ICSI train meetings (vs 24), plus simulated overlap mixtures of LibriSpeech-100's 251 speakers, with
  voice prints of 1.5-10 s. The turn head reads the TS-VAD track exactly as `--mode single` serves it.
- **S2 (`--unfreeze 5`).** Adds encoder blocks 13-17, the VAD head and the RNNT under a WER gate on LibriSpeech-200
  and AMI-200. The gate restores the last passing state and halves the LR; a second firing kills the run.
- **S3 (optional, `--asr-kd`).** Adds nemotron-speech-streaming-en-0.6b's transcripts and top-layer targets. This
  reuses `distill_0p6b_to_115m`'s teacher cache.

Blocks 1-4 and the speaker head never train, so stored voice prints stay valid.

Pre-registered bars (README):
- tracking F1 ≥ 0.80 on AMI and ≥ 0.90 on ICSI (today 0.743 / 0.882);
- turn misses at matched false cut-offs below today's single mode, with a CI excluding 0 (34.2 / 18.7 %);
- LibriSpeech-200 ≤ 2.59 %; AMI-200 ≤ 24.4 + 1.0 %;
- VAD F1 ≥ 0.94 when unfrozen; LID 2 s ≥ 90 % if block 12 is touched.

Kill if the WER gate fires twice, or if S1 lifts AMI F1 by less than 0.02. Estimate: **~9.6 GPU-hours, ~7-8 h wall on
two 5090s** (README table: S1 1.5 GPU-h, S1 control 1.5, S2 2.5, S3 2.8 + teacher 0.4, evals ~0.9).

**Smoke test (2026-09-29, laptop MPS, through the gate, no other MPS job; `smoke.sh`, log
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/single_distill_smoke/smoke.log`): passed, 226 s wall, every stage
run.**
- manifest: 2 AMI + 2 ICSI train meetings, 112 crops, 112 turn windows, 12 LibriSpeech speakers, ASR manifest
  1.0 h.
- evalsets.
- The 0.6B teacher cache on MPS (3 shards, 56 s).
- S1 heads only: 30 steps × 4 at ~0.5 s per step on MPS. Gates pass (the ASR is frozen). Resume is a no-op when
  done. The export asserts 730 other tensors bit-identical.
- S3 `--unfreeze 5 --asr-kd`: all four loss families ran (tsvad, VAD, RNNT + KL anchor, 0.6B sequence + encoder
  cosine). The first gate fired at the smoke's learning rate (LibriSpeech 0.7 → 1.7 %, AMI 15.9 → 25.1 % on 20
  utterances each). It restored the step-0 state, halved the LR, and then passed twice (0.5 % / 15.6-16.6 %). The
  export asserts 529 other tensors bit-identical.
- The forced gate test (`--lr 5e-2`, zero tolerance): GATE FIRED 1x (restore, LR × 0.5), then GATE FIRED 2x →
  KILLED at the last passing state (step 0).
- eval: tracking F1 and WER, served vs student, on 32 windows / 20 utterances (plumbing only; not a result).

The smoke numbers prove the plumbing, not the method. Nothing here was trained long enough to say whether the bars
can be met.
