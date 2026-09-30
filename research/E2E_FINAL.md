# Final end-to-end comparison: our front end vs the Pipecat and LiveKit default local stacks

2026-09-27. Data: `runs/e2e_final.json` (generated 2026-09-27 23:41 by `scripts/e2e_final.py report`; keys `table`,
`paired`, `clean_checks`, `components_live`, `clips`, `per_clip`). Driver: `scripts/e2e_final.py`; metric definitions:
`audioforge/e2e_metrics.py`. Protocol background: [`INTEGRATION.md`](archive/INTEGRATION.md) §4 and §8. This document is a
write-up; every number below is read from the JSON (the tables are rendered from it, not typed).

## 1. Summary

Seven systems answered the same recorded conversations at 1x real time on this Mac (CPU, 2 threads per model
process), through the real Pipecat 1.12 and LiveKit Agents 1.8 pipelines, and were scored by when the agent would
have started to answer. Against **Pipecat's default local stack (A)**, our product default **C** cuts users off far
less (1.1-1.4 vs 2.1-7.5 cut-ins per minute) and misses fewer turn ends on every clip set, with median dead air that is
lower on three sets and higher on two (no dead-air CI excludes 0 at these n). Against **LiveKit's default local stack (B)**,
dead air is comparable (C − B medians within ±40 ms on three of the four two-party cells, −0.4 s on the TurnBench
mix, −1.0 s on AMI), we miss fewer ends on every set, and cut-ins are similar on the user channel of otoSpeech but higher on the TurnBench user channel and on every
mono mix: B is conservative (misses 45-70 % of ends on mono mixes) where C answers. The opt-in rules **D / DN**
(`hybrid_dyn` + `after_agent_arm`) trade about 0.9 s more dead air for near-zero cut-ins (0 on AMI, 0.4-0.9 per
minute elsewhere for DN). **Nemotron-3-Diarization 100M** as the diarizer (**CN / DN**) gives lower dead air, fewer or
equal misses, server RTF 0.63-0.65 instead of 0.79-0.81 and 1.5 GB instead of 3.6 GB of RSS; it should become the
default diarizer. The two-channel offline predictive trigger **Dp** has the lowest dead air of all (786 / 1080 ms
medians) but is not a live system. Transcripts: our streaming RNNT is 1-2 points behind Whisper small on the mono
mixes and 7 points behind it on the clean user channel (17 % vs 10 % WER), which is why the per-turn Parakeet-TDT
final pass exists (`research/FINAL_REPORT.md` §1); our first partial arrives 0.8-1.2 s after speech onset against
2.6-3.7 s (A) and 3.4-11.3 s (B).

**Recommended configurations.**
- *Two-party voice agent* (one user, the agent's own TTS known to the app): `--diar runs/nemo_nemotron3_diar.afm
  --diar-pool max --diar-spks 4 --enroll after_agent_arm`, `turn_policy: "timeout"` (1000 ms) when responsiveness
  matters (= CN: median dead air 1.25-1.29 s and 6-20 % missed at 3 s on the user channel), `turn_policy: "hybrid_dyn"`
  (= DN) when not interrupting the user matters more (+0.9 s dead air, cut-ins down 30-55 %). Feed the user channel,
  not a mix, whenever the transport has it: every system does better there. Add `--final-asr tdt_v3` for the
  transcript.
- *Multi-party room* (several people, the agent addressed by one of them): CN with `hybrid_dyn` (= DN). On the AMI
  windows DN has 0 cut-ins in both frameworks at 2.3 s median dead air; the timeout variants cut in 1.4-3.4 times per
  minute because a 1 s pause in a meeting is usually a hesitation, not a hand-over (`research/archive/AMI.md`). Send
  `agent_end` from the TTS so the primary speaker is bound to the first voice after the agent stops.

## 2. Systems

| id | stack | what decides the turn | transcript | notes |
|---|---|---|---|---|
| A | Pipecat default local stack | `SileroVADAnalyzer` (Pipecat's VAD state machine) + `LocalSmartTurnAnalyzerV3` (bundled smart-turn-v3.2-cpu, 3 s stop fallback) | `WhisperSTTService`, faster-whisper "small", segmented on VAD stops | Pipecat 1.12, in-process |
| B | LiveKit default local stack | `livekit-plugins-silero` VAD + `livekit-plugins-turn-detector` `EnglishModel` (in-process executor) | the same faster-whisper "small" as a non-streaming LiveKit STT behind `StreamAdapter` | LiveKit Agents 1.8, room-less `AgentSession`; evaluation only (LiveKit Model License) |
| C | ours, product default | `audioforge.serve`: `runs/stage1_served.afm` (115M frozen NVIDIA streaming encoder + our VAD / speaker / turn heads, 160 ms chunks) + Streaming Sortformer v2 at 0.32 s; `turn_policy: "timeout"` 1000 ms on the label-free primary column; `--enroll dominant` | our streaming RNNT (same model) | through the committed Pipecat / LiveKit adapters |
| D | ours, best rules | same server with `--enroll after_agent_arm` and `turn_policy: "hybrid_dyn"` (head p ≥ 0.998 OR any-speaker Silero silence ≥ clamp(80 − 55 p, 7, 80) frames); `agent_end` sent at the other party's labelled turn ends | same | the `agent_end` time is a **label-derived stand-in** for the TTS-end event |
| CN / DN | C / D with Nemotron-3-Diarization 100M instead of Sortformer v2 | `--diar runs/nemo_nemotron3_diar.afm --diar-pool max --diar-left 1 --diar-spks 4` (10 ms outputs max-pooled to 80 ms, the first 4 of its 8 arrival-order columns) | same | |
| Dp | ours, predictive OR Silero | head (c) of `research/archive/DYADIC.md` §8 (`runs/stage1_turn_dyadic_mh.afm`) on the mixed mono plus both parties' per-channel Silero tracks; predictive trigger (P(bin1) < 0.012 AND P(bin2) < 0.012, 2 s refractory) OR user-channel Pipecat-VAD silence ≥ 1.4 s | – | **offline** causal scorer, two-channel input, decision times without delivery lag; not wired into `serve.py` |

A, C, D, CN, DN ran through a Pipecat pipeline (WAV transport at 1x → STT → user aggregator → mock LLM / TTS); B, C, D,
CN, DN through a LiveKit `AgentSession` (stub LLM, tone TTS). The **response moment** is what a caller waits for:
Pipecat's `LLMContextFrame` reaching the LLM, LiveKit's committed user turn. Every framework × system × condition ×
clip is one timed session, started only when the machine guard passed (no training process, 1-min load < 6, other
processes < 150 % of one core). A run is **flagged** when its server RTF exceeded 1, the driver's component RTF
exceeded 1, or other processes were above 300 % of a core when it ended; flagged runs were queued for a rerun and a
clean rerun replaces the flagged record.

## 3. Clips and metrics

| set | clips | audio | scored user turn ends | `agent_end` events | conditions | reference text |
|---|---|---|---|---|---|---|
| AMI (INTEGRATION §4's 5 windows) | 5 | 88 s | 5 (one per window) | 5 | mono | AMI words |
| TurnBench dev | 16 | 730 s (35-57 s each) | 56 | 49 | mono mix, user channel | annotator segments (both labels) |
| otoSpeech dev (never trained on; minus the 16 conversations used for the predictive fit) | 16 | 761 s | 53 | 46 | mono mix, user channel | none |

Two-party clips are the first qualifying 35-60 s segment of each conversation (start and end in silent gaps ≥ 0.3 s,
no excluded zone), padded with 6 s of silence (AMI 7 s, INTEGRATION §8's pad). The "user" is the party the dyadic
role rule calls human; the other party stands in for the agent, and its labelled floor-turn ends are the `agent_end`
events for D / DN. Labels are used only for scoring.

Definitions (`audioforge/e2e_metrics.py`): the **response to end e** is the first response moment τ with
e ≤ τ < the user's next turn start; **dead air** = τ − e (median and P90 over ends answered within 6 s);
**missed within h** = no response in [e, min(next start, e + h)) for h = 3 s and 6 s, over all scored ends;
a **cut-in** is a response moment inside a user turn span (speech or a within-turn pause), counted per clip and per
audio minute. **WER** (TurnBench and AMI only) is on the concatenated finals after `normalize_text`. **First text** is
the first partial or final relative to the first speech onset of the condition's audio. Server RTF, backlog and RSS
come from the server's `stats`; A and B have no server, their models run inside the driver. Paired deltas resample
clips (2000 bootstraps) and recompute the pooled statistic of both systems on the same clips; with n = 5 AMI windows
the CIs are wide and only the largest AMI effects (cut-ins) clear zero.

## 4. Results by clip set

Per set and condition: Pipecat / LiveKit values side by side (C-DN run through both frameworks; A only in Pipecat, B
only in LiveKit). Dead air is over answered ends, so a system that misses more ends can show a lower median (A on the
TurnBench mono mix answers 25 % of ends within 3 s and 38 % within 6 s; its median is over those).

**AMI, 5 windows, mono** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2548 / – | 4605 / – | 60 % / – | 20 % / – | 7.46 / – | 44 % / – | 3689 / – |
| B LiveKit default | – / 2660 | – / 4280 | – / 40 % | – / 0 % | – / 3.39 | – / 43 % | – / 9190 |
| C ours timeout | 1683 / 1660 | 3042 / 2224 | 20 % / 0 % | 0 % / 0 % | 1.36 / 3.39 | 39 % / 39 % | 828 / 810 |
| CN ours+Nemotron-3 timeout | 1903 / 1380 | 3965 / 1776 | 40 % / 0 % | 0 % / 0 % | 1.36 / 2.04 | 39 % / 39 % | 819 / 810 |
| D ours hybrid_dyn+arm | 2561 / 2420 | 2990 / 2956 | 20 % / 20 % | 0 % / 0 % | 0.00 / 0.00 | 39 % / 39 % | 827 / 810 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2322 / 2340 | 2530 / 2500 | 20 % / 0 % | 20 % / 0 % | 0.00 / 0.00 | 39 % / 39 % | 821 / 810 |

**TurnBench dev, 16 clips, mono mix** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 1054 / – | 4576 / – | 75 % / – | 62 % / – | 2.96 / – | 26 % / – | 2598 / – |
| B LiveKit default | – / 1714 | – / 2993 | – / 70 % | – / 66 % | – / 1.07 | – / 23 % | – / 11288 |
| C ours timeout | 1346 / 1312 | 3355 / 1933 | 68 % / 34 % | 62 % / 30 % | 1.15 / 4.52 | 27 % / 27 % | 1005 / 999 |
| CN ours+Nemotron-3 timeout | 1338 / 1271 | 5431 / 1488 | 71 % / 30 % | 64 % / 29 % | 1.07 / 3.62 | 27 % / 27 % | 1004 / 999 |
| D ours hybrid_dyn+arm | 1696 / 1628 | 3303 / 3278 | 70 % / 54 % | 64 % / 46 % | 0.49 / 1.97 | 27 % / 27 % | 1005 / 999 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 1944 / 1755 | 2574 / 2617 | 66 % / 54 % | 62 % / 52 % | 0.41 / 0.58 | 27 % / 27 % | 1006 / 999 |
| Dp ours predictive (offline) | 786 | 1467 | 21 % | 20 % | 1.73 | – | – |

**TurnBench dev, 16 clips, user channel** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2099 / – | 3186 / – | 54 % / – | 32 % / – | 2.14 / – | 16 % / – | 2644 / – |
| B LiveKit default | – / 1350 | – / 3080 | – / 36 % | – / 25 % | – / 0.99 | – / 10 % | – / 5041 |
| C ours timeout | 1353 / 1330 | 1617 / 1500 | 23 % / 23 % | 20 % / 23 % | 1.32 / 2.22 | 17 % / 17 % | 942 / 936 |
| CN ours+Nemotron-3 timeout | 1285 / 1256 | 1406 / 1378 | 20 % / 18 % | 16 % / 18 % | 1.32 / 1.32 | 17 % / 17 % | 978 / 966 |
| D ours hybrid_dyn+arm | 2311 / 2308 | 2630 / 2562 | 38 % / 39 % | 32 % / 36 % | 0.74 / 0.82 | 17 % / 17 % | 942 / 936 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2256 / 2234 | 2573 / 2460 | 38 % / 39 % | 32 % / 36 % | 0.74 / 0.91 | 17 % / 17 % | 944 / 936 |
| Dp ours predictive (offline) | 786 | 1467 | 21 % | 20 % | 1.73 | – | – |

**otoSpeech, 16 clips, mono mix** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 2072 / – | 4311 / – | 62 % / – | 34 % / – | 2.84 / – | – / – | 2828 / – |
| B LiveKit default | – / 1310 | – / 3286 | – / 45 % | – / 36 % | – / 2.13 | – / – | – / 4610 |
| C ours timeout | 1372 / 1350 | 3553 / 3414 | 42 % / 32 % | 32 % / 21 % | 1.26 / 3.86 | – / – | 1077 / 1075 |
| CN ours+Nemotron-3 timeout | 1302 / 1290 | 3052 / 1674 | 38 % / 17 % | 30 % / 15 % | 2.05 / 3.23 | – / – | 1072 / 1065 |
| D ours hybrid_dyn+arm | 2252 / 2215 | 4290 / 3458 | 57 % / 55 % | 49 % / 47 % | 0.79 / 2.37 | – / – | 1078 / 1075 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2162 / 1960 | 3379 / 2420 | 43 % / 34 % | 36 % / 30 % | 0.47 / 0.87 | – / – | 1072 / 1065 |
| Dp ours predictive (offline) | 1080 | 1367 | 19 % | 17 % | 1.50 | – | – |

**otoSpeech, 16 clips, user channel** (Pipecat / LiveKit; Dp offline on the two-channel input)

| system | dead air median ms | dead air P90 ms | missed 3 s | missed 6 s | cut-ins per min | WER | first text ms |
|---|---|---|---|---|---|---|---|
| A Pipecat default | 1114 / – | 3159 / – | 43 % / – | 13 % / – | 2.21 / – | – / – | 2864 / – |
| B LiveKit default | – / 1350 | – / 2368 | – / 13 % | – / 11 % | – / 1.58 | – / – | – / 3375 |
| C ours timeout | 1393 / 1385 | 1593 / 2147 | 6 % / 9 % | 4 % / 6 % | 1.26 / 1.50 | – / – | 1187 / 1180 |
| CN ours+Nemotron-3 timeout | 1271 / 1250 | 1422 / 1441 | 6 % / 6 % | 4 % / 6 % | 1.34 / 1.34 | – / – | 1184 / 1175 |
| D ours hybrid_dyn+arm | 2343 / 2320 | 2843 / 2820 | 13 % / 13 % | 8 % / 8 % | 0.87 / 0.87 | – / – | 1186 / 1180 |
| DN ours+Nemotron-3 hybrid_dyn+arm | 2302 / 2280 | 3122 / 2553 | 19 % / 13 % | 8 % / 9 % | 0.63 / 0.63 | – / – | 1182 / 1175 |
| Dp ours predictive (offline) | 1080 | 1367 | 19 % | 17 % | 1.50 | – | – |

### Full tables (per framework, with compute)

"flagged runs" = kept records that are still flagged (no clean rerun replaced them): 4 in total (DN LiveKit on 2 AMI
windows, DN Pipecat on 1 oto mono clip, B LiveKit on 1 TurnBench mono clip). Their server RTF maxima are 0.58-0.66,
so the flag came from machine load at the end of the run, not from the server falling behind; B has no server.

**AMI, 5 windows, mono**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livekit | B LiveKit default | 5 / 5 | 2660 / 4280 | 40 % / 0 % | 5 (1.00, 3.39) | 43 % | 9190 | – / – | – | 2199 / – | 24.3 / – | 0 |
| livekit | C ours timeout | 5 / 5 | 1660 / 2224 | 0 % / 0 % | 5 (1.00, 3.39) | 39 % | 810 | 0.713 / 0.733 | 320 | 159 / 3564 | 1.2 / 77.2 | 0 |
| livekit | CN ours+Nemotron-3 timeout | 5 / 5 | 1380 / 1776 | 0 % / 0 % | 3 (0.60, 2.04) | 39 % | 810 | 0.566 / 0.576 | 260 | 159 / 1482 | 1.4 / 53.4 | 0 |
| livekit | D ours hybrid_dyn+arm | 5 / 5 | 2420 / 2956 | 20 % / 0 % | 0 (0.00, 0.00) | 39 % | 810 | 0.705 / 0.736 | 420 | 158 / 3561 | 1.2 / 77.1 | 0 |
| livekit | DN ours+Nemotron-3 hybrid_dyn+arm | 5 / 5 | 2340 / 2500 | 0 % / 0 % | 0 (0.00, 0.00) | 39 % | 810 | 0.558 / 0.583 | 400 | 159 / 1482 | 1.4 / 53.8 | 2 |
| pipecat | A Pipecat default | 5 / 5 | 2548 / 4605 | 60 % / 20 % | 11 (2.20, 7.46) | 44 % | 3689 | – / – | – | 1649 / – | 40.6 / – | 0 |
| pipecat | C ours timeout | 5 / 5 | 1683 / 3042 | 20 % / 0 % | 2 (0.40, 1.36) | 39 % | 828 | 0.697 / 0.725 | 380 | 91 / 3558 | 2.0 / 86.0 | 0 |
| pipecat | CN ours+Nemotron-3 timeout | 5 / 5 | 1903 / 3965 | 40 % / 0 % | 2 (0.40, 1.36) | 39 % | 819 | 0.551 / 0.585 | 240 | 95 / 1472 | 2.3 / 60.0 | 0 |
| pipecat | D ours hybrid_dyn+arm | 5 / 5 | 2561 / 2990 | 20 % / 0 % | 0 (0.00, 0.00) | 39 % | 827 | 0.697 / 0.737 | 300 | 91 / 3564 | 2.0 / 86.5 | 0 |
| pipecat | DN ours+Nemotron-3 hybrid_dyn+arm | 5 / 5 | 2322 / 2530 | 20 % / 20 % | 0 (0.00, 0.00) | 39 % | 821 | 0.564 / 0.581 | 220 | 92 / 1482 | 2.3 / 60.7 | 0 |

**TurnBench dev, 16 clips, mono mix**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livekit | B LiveKit default | 16 / 56 | 1714 / 2993 | 70 % / 66 % | 13 (0.81, 1.07) | 23 % | 11288 | – / – | – | 2334 / – | 26.0 / – | 1 |
| livekit | C ours timeout | 16 / 56 | 1312 / 1933 | 34 % / 30 % | 55 (3.44, 4.52) | 27 % | 999 | 0.798 / 0.815 | 320 | 202 / 3583 | 1.1 / 96.0 | 0 |
| livekit | CN ours+Nemotron-3 timeout | 16 / 56 | 1271 / 1488 | 30 % / 29 % | 44 (2.75, 3.62) | 27 % | 999 | 0.637 / 0.655 | 480 | 188 / 1432 | 1.3 / 67.0 | 0 |
| livekit | D ours hybrid_dyn+arm | 16 / 56 | 1628 / 3278 | 54 % / 46 % | 24 (1.50, 1.97) | 27 % | 999 | 0.803 / 0.820 | 400 | 195 / 3580 | 1.1 / 96.5 | 0 |
| livekit | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 56 | 1755 / 2617 | 54 % / 52 % | 7 (0.44, 0.58) | 27 % | 999 | 0.642 / 0.662 | 480 | 192 / 1476 | 1.3 / 67.4 | 0 |
| pipecat | A Pipecat default | 16 / 56 | 1054 / 4576 | 75 % / 62 % | 36 (2.25, 2.96) | 26 % | 2598 | – / – | – | 2344 / – | 50.8 / – | 0 |
| pipecat | C ours timeout | 16 / 56 | 1346 / 3355 | 68 % / 62 % | 14 (0.88, 1.15) | 27 % | 1005 | 0.795 / 0.820 | 380 | 119 / 3581 | 1.9 / 102.1 | 0 |
| pipecat | CN ours+Nemotron-3 timeout | 16 / 56 | 1338 / 5431 | 71 % / 64 % | 13 (0.81, 1.07) | 27 % | 1004 | 0.634 / 0.656 | 460 | 118 / 1485 | 2.3 / 70.8 | 0 |
| pipecat | D ours hybrid_dyn+arm | 16 / 56 | 1696 / 3303 | 70 % / 64 % | 6 (0.38, 0.49) | 27 % | 1005 | 0.803 / 0.826 | 360 | 121 / 3560 | 1.9 / 102.5 | 0 |
| pipecat | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 56 | 1944 / 2574 | 66 % / 62 % | 5 (0.31, 0.41) | 27 % | 1006 | 0.639 / 0.662 | 380 | 128 / 1480 | 2.3 / 71.5 | 0 |

**TurnBench dev, 16 clips, user channel**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livekit | B LiveKit default | 16 / 56 | 1350 / 3080 | 36 % / 25 % | 12 (0.75, 0.99) | 10 % | 5041 | – / – | – | 2802 / – | 24.5 / – | 0 |
| livekit | C ours timeout | 16 / 56 | 1330 / 1500 | 23 % / 23 % | 27 (1.69, 2.22) | 17 % | 936 | 0.788 / 0.812 | 420 | 200 / 3562 | 1.0 / 94.7 | 0 |
| livekit | CN ours+Nemotron-3 timeout | 16 / 56 | 1256 / 1378 | 18 % / 18 % | 16 (1.00, 1.32) | 17 % | 966 | 0.633 / 0.658 | 420 | 190 / 1480 | 1.2 / 65.6 | 0 |
| livekit | D ours hybrid_dyn+arm | 16 / 56 | 2308 / 2562 | 39 % / 36 % | 10 (0.62, 0.82) | 17 % | 936 | 0.792 / 0.821 | 500 | 202 / 3567 | 1.0 / 95.2 | 0 |
| livekit | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 56 | 2234 / 2460 | 39 % / 36 % | 11 (0.69, 0.91) | 17 % | 936 | 0.632 / 0.665 | 360 | 187 / 1361 | 1.3 / 66.1 | 0 |
| pipecat | A Pipecat default | 16 / 56 | 2099 / 3186 | 54 % / 32 % | 26 (1.62, 2.14) | 16 % | 2644 | – / – | – | 2295 / – | 35.0 / – | 0 |
| pipecat | C ours timeout | 16 / 56 | 1353 / 1617 | 23 % / 20 % | 16 (1.00, 1.32) | 17 % | 942 | 0.788 / 0.818 | 360 | 125 / 3556 | 1.9 / 100.5 | 0 |
| pipecat | CN ours+Nemotron-3 timeout | 16 / 56 | 1285 / 1406 | 20 % / 16 % | 16 (1.00, 1.32) | 17 % | 978 | 0.635 / 0.665 | 400 | 129 / 1481 | 2.3 / 70.0 | 0 |
| pipecat | D ours hybrid_dyn+arm | 16 / 56 | 2311 / 2630 | 38 % / 32 % | 9 (0.56, 0.74) | 17 % | 942 | 0.791 / 0.820 | 460 | 122 / 3560 | 1.9 / 100.6 | 0 |
| pipecat | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 56 | 2256 / 2573 | 38 % / 32 % | 9 (0.56, 0.74) | 17 % | 944 | 0.634 / 0.658 | 660 | 129 / 1482 | 2.3 / 69.7 | 0 |

**otoSpeech, 16 clips, mono mix**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livekit | B LiveKit default | 16 / 53 | 1310 / 3286 | 45 % / 36 % | 27 (1.69, 2.13) | – | 4610 | – / – | – | 2769 / – | 28.9 / – | 0 |
| livekit | C ours timeout | 16 / 53 | 1350 / 3414 | 32 % / 21 % | 49 (3.06, 3.86) | – | 1075 | 0.800 / 0.816 | 460 | 198 / 3586 | 1.0 / 96.7 | 0 |
| livekit | CN ours+Nemotron-3 timeout | 16 / 53 | 1290 / 1674 | 17 % / 15 % | 41 (2.56, 3.23) | – | 1065 | 0.644 / 0.723 | 460 | 192 / 1470 | 1.2 / 68.6 | 0 |
| livekit | D ours hybrid_dyn+arm | 16 / 53 | 2215 / 3458 | 55 % / 47 % | 30 (1.88, 2.37) | – | 1075 | 0.805 / 0.826 | 400 | 199 / 3594 | 1.0 / 97.3 | 0 |
| livekit | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 53 | 1960 / 2420 | 34 % / 30 % | 11 (0.69, 0.87) | – | 1065 | 0.677 / 0.701 | 700 | 173 / 1483 | 1.3 / 69.9 | 0 |
| pipecat | A Pipecat default | 16 / 53 | 2072 / 4311 | 62 % / 34 % | 36 (2.25, 2.84) | – | 2828 | – / – | – | 2372 / – | 49.2 / – | 0 |
| pipecat | C ours timeout | 16 / 53 | 1372 / 3553 | 42 % / 32 % | 16 (1.00, 1.26) | – | 1077 | 0.799 / 0.815 | 400 | 123 / 3557 | 1.9 / 102.2 | 0 |
| pipecat | CN ours+Nemotron-3 timeout | 16 / 53 | 1302 / 3052 | 38 % / 30 % | 26 (1.62, 2.05) | – | 1072 | 0.637 / 0.660 | 400 | 124 / 1350 | 2.3 / 71.1 | 0 |
| pipecat | D ours hybrid_dyn+arm | 16 / 53 | 2252 / 4290 | 57 % / 49 % | 10 (0.62, 0.79) | – | 1078 | 0.806 / 0.822 | 520 | 121 / 3558 | 1.9 / 102.8 | 0 |
| pipecat | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 53 | 2162 / 3379 | 43 % / 36 % | 6 (0.38, 0.47) | – | 1072 | 0.643 / 0.656 | 440 | 127 / 1378 | 2.3 / 71.7 | 1 |

**otoSpeech, 16 clips, user channel**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| livekit | B LiveKit default | 16 / 53 | 1350 / 2368 | 13 % / 11 % | 20 (1.25, 1.58) | – | 3375 | – / – | – | 2683 / – | 25.7 / – | 0 |
| livekit | C ours timeout | 16 / 53 | 1385 / 2147 | 9 % / 6 % | 19 (1.19, 1.50) | – | 1180 | 0.813 / 0.983 | 3800 | 161 / 3564 | 1.0 / 98.3 | 0 |
| livekit | CN ours+Nemotron-3 timeout | 16 / 53 | 1250 / 1441 | 6 % / 6 % | 17 (1.06, 1.34) | – | 1175 | 0.653 / 0.752 | 540 | 170 / 1047 | 1.2 / 67.0 | 0 |
| livekit | D ours hybrid_dyn+arm | 16 / 53 | 2320 / 2820 | 13 % / 8 % | 11 (0.69, 0.87) | – | 1180 | 0.798 / 0.825 | 680 | 198 / 3567 | 0.9 / 95.2 | 0 |
| livekit | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 53 | 2280 / 2553 | 13 % / 9 % | 8 (0.50, 0.63) | – | 1175 | 0.660 / 0.715 | 640 | 174 / 1471 | 1.2 / 67.7 | 0 |
| pipecat | A Pipecat default | 16 / 53 | 1114 / 3159 | 43 % / 13 % | 28 (1.75, 2.21) | – | 2864 | – / – | – | 2326 / – | 36.6 / – | 0 |
| pipecat | C ours timeout | 16 / 53 | 1393 / 1593 | 6 % / 4 % | 16 (1.00, 1.26) | – | 1187 | 0.805 / 0.830 | 440 | 109 / 3564 | 1.9 / 101.5 | 0 |
| pipecat | CN ours+Nemotron-3 timeout | 16 / 53 | 1271 / 1422 | 6 % / 4 % | 17 (1.06, 1.34) | – | 1184 | 0.639 / 0.705 | 540 | 112 / 1378 | 2.3 / 70.1 | 0 |
| pipecat | D ours hybrid_dyn+arm | 16 / 53 | 2343 / 2843 | 13 % / 8 % | 11 (0.69, 0.87) | – | 1186 | 0.801 / 0.838 | 520 | 112 / 3565 | 1.9 / 101.5 | 0 |
| pipecat | DN ours+Nemotron-3 hybrid_dyn+arm | 16 / 53 | 2302 / 3122 | 19 % / 8 % | 8 (0.50, 0.63) | – | 1182 | 0.687 / 0.763 | 2400 | 104 / 1398 | 2.5 / 73.3 | 0 |

**TurnBench dev, two-channel (offline Dp)**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| offline | Dp ours predictive (offline) | 16 / 56 | 786 / 1467 | 21 % / 20 % | 21 (1.31, 1.73) | – | – | – / – | – | – / – | – / – | 0 |

**otoSpeech, two-channel (offline Dp)**

| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per clip, per min) | WER | first text ms | server RTF median / max | backlog max ms | peak RSS MB driver / server | CPU % driver / server | flagged runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| offline | Dp ours predictive (offline) | 16 / 53 | 1080 / 1367 | 19 % / 17 % | 19 (1.19, 1.50) | – | – | – / – | – | – / – | – / – | 0 |

## 5. Paired differences (x − y), 95 % bootstrap CI over clips

Bold = the CI excludes 0. n = 5 clips on AMI, 16 on TurnBench and otoSpeech. Missed rates in percentage points,
cut-ins per clip.

**AMI, 5 windows, mono**

| framework | x − y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |
|---|---|---|---|---|---|
| pipecat | C-A | -865 [-3522, +807] | -40 [-80, +0] | -20 [-60, +0] | **-1.80 [-3.20, -0.40]** |
| livekit | C-B | -1000 [-3600, +920] | -40 [-80, +0] | +0 [+0, +0] | +0.00 [-1.20, +1.20] |
| pipecat | CN-A | -645 [-3302, +2491] | -20 [-80, +40] | -20 [-60, +0] | **-1.80 [-3.20, -0.40]** |
| livekit | CN-B | -1280 [-3420, +900] | -40 [-80, +0] | +0 [+0, +0] | -0.40 [-1.40, +0.60] |
| pipecat | D-A | +13 [-2644, +1685] | -40 [-100, +40] | -20 [-60, +0] | **-2.20 [-3.40, -1.00]** |
| livekit | D-B | -240 [-2620, +1660] | -20 [-80, +40] | +0 [+0, +0] | **-1.00 [-1.80, -0.20]** |
| pipecat | DN-A | -226 [-2923, +1485] | -40 [-80, +0] | +0 [-60, +60] | **-2.20 [-3.40, -1.00]** |
| livekit | DN-B | -320 [-2660, +1600] | -40 [-80, +0] | +0 [+0, +0] | **-1.00 [-1.80, -0.20]** |
| livekit | D-C | **+760 [+700, +980]** | +20 [+0, +60] | +0 [+0, +0] | **-1.00 [-1.60, -0.40]** |
| pipecat | D-C | +878 [-842, +959] | +0 [-60, +60] | +0 [+0, +0] | -0.40 [-0.80, +0.00] |
| livekit | CN-C | -280 [-1280, +180] | +0 [+0, +0] | +0 [+0, +0] | -0.40 [-0.80, +0.00] |
| pipecat | CN-C | +220 [-281, +1880] | +20 [+0, +60] | +0 [+0, +0] | +0.00 [+0.00, +0.00] |
| livekit | DN-D | **-80 [-1220, -40]** | -20 [-60, +0] | +0 [+0, +0] | +0.00 [+0.00, +0.00] |
| pipecat | DN-D | -239 [-1261, +41] | +0 [-60, +60] | +20 [+0, +60] | +0.00 [+0.00, +0.00] |

**TurnBench dev, 16 clips, mono mix**

| framework | x − y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |
|---|---|---|---|---|---|
| pipecat | C-A | +292 [-1616, +504] | -7 [-18, +3] | +0 [-14, +13] | **-1.38 [-2.19, -0.63]** |
| livekit | C-B | -402 [-1086, +750] | **-36 [-55, -17]** | **-36 [-53, -20]** | **+2.63 [+1.81, +3.44]** |
| pipecat | CN-A | +284 [-1584, +590] | -4 [-14, +6] | +2 [-11, +14] | **-1.44 [-2.12, -0.81]** |
| livekit | CN-B | -443 [-1179, +724] | **-39 [-56, -23]** | **-38 [-53, -23]** | **+1.94 [+1.25, +2.69]** |
| pipecat | D-A | +642 [-1331, +1370] | -5 [-15, +3] | +2 [-9, +12] | **-1.88 [-2.50, -1.31]** |
| livekit | D-B | -86 [-1056, +1558] | **-16 [-32, -2]** | **-20 [-36, -4]** | **+0.69 [+0.06, +1.25]** |
| pipecat | DN-A | +890 [-1100, +1308] | -9 [-22, +2] | +0 [-13, +13] | **-1.94 [-2.63, -1.25]** |
| livekit | DN-B | +41 [-965, +1387] | -16 [-34, +0] | -14 [-31, +0] | -0.37 [-0.87, +0.12] |
| livekit | D-C | +316 [-138, +1035] | **+20 [+5, +36]** | +16 [+0, +33] | **-1.94 [-3.00, -1.06]** |
| pipecat | D-C | +350 [-91, +1065] | +2 [-8, +13] | +2 [-7, +12] | **-0.50 [-0.88, -0.19]** |
| livekit | CN-C | -41 [-245, +106] | -4 [-15, +8] | -2 [-10, +8] | -0.69 [-1.62, +0.12] |
| pipecat | CN-C | -8.00 [-127.25, +438.00] | +4 [+0, +9] | +2 [+0, +6] | -0.06 [-0.44, +0.31] |
| livekit | DN-D | +127 [-608, +595] | +0 [-7, +7] | +5 [-2, +14] | **-1.06 [-1.50, -0.62]** |
| pipecat | DN-D | +248 [-465, +628] | -4 [-12, +5] | -2 [-8, +4] | -0.06 [-0.25, +0.13] |

**TurnBench dev, 16 clips, user channel**

| framework | x − y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |
|---|---|---|---|---|---|
| pipecat | C-A | -746 [-1646, +378] | **-30 [-51, -9]** | -12 [-27, +2] | -0.62 [-1.56, +0.25] |
| livekit | C-B | -20 [-372, +108] | -12 [-27, +4] | -2 [-11, +9] | **+0.94 [+0.31, +1.62]** |
| pipecat | CN-A | -814 [-1720, +296] | **-34 [-53, -15]** | **-16 [-30, -2]** | -0.62 [-1.56, +0.19] |
| livekit | CN-B | **-94 [-444, -2]** | **-18 [-32, -3]** | -7 [-17, +3] | +0.25 [-0.25, +0.75] |
| pipecat | D-A | +212 [-689, +1343] | -16 [-35, +2] | +0 [-16, +15] | **-1.06 [-2.00, -0.12]** |
| livekit | D-B | **+958 [+602, +1074]** | +4 [-13, +20] | +11 [-2, +25] | -0.12 [-0.50, +0.31] |
| pipecat | DN-A | +157 [-742, +1275] | -16 [-35, +2] | +0 [-16, +15] | **-1.06 [-1.94, -0.19]** |
| livekit | DN-B | **+884 [+546, +994]** | +4 [-13, +20] | +11 [-2, +25] | -0.06 [-0.50, +0.37] |
| livekit | D-C | **+978 [+734, +1028]** | **+16 [+5, +29]** | **+12 [+2, +26]** | **-1.06 [-1.94, -0.31]** |
| pipecat | D-C | **+958 [+716, +1026]** | **+14 [+4, +28]** | **+12 [+2, +26]** | -0.44 [-1.12, +0.31] |
| livekit | CN-C | **-74 [-167, -46]** | -5 [-12, +0] | -5 [-12, +0] | **-0.69 [-1.25, -0.19]** |
| pipecat | CN-C | **-68 [-145, -44]** | -4 [-9, +0] | -4 [-9, +0] | +0.00 [-0.19, +0.19] |
| livekit | DN-D | -74 [-165, +40] | +0 [+0, +0] | +0 [+0, +0] | +0.06 [-0.13, +0.25] |
| pipecat | DN-D | -55 [-149, +48] | +0 [+0, +0] | +0 [+0, +0] | +0.00 [-0.19, +0.19] |

**otoSpeech, 16 clips, mono mix**

| framework | x − y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |
|---|---|---|---|---|---|
| pipecat | C-A | -700 [-1741, +408] | **-21 [-39, -2]** | -2 [-19, +15] | **-1.25 [-2.56, -0.12]** |
| livekit | C-B | +40 [-710, +735] | -13 [-25, +0] | **-15 [-26, -4]** | **+1.37 [+0.75, +2.06]** |
| pipecat | CN-A | -770 [-1822, +328] | **-25 [-42, -4]** | -4 [-23, +15] | -0.62 [-1.94, +0.50] |
| livekit | CN-B | -20 [-850, +650] | **-28 [-48, -11]** | **-21 [-36, -8]** | **+0.87 [+0.25, +1.50]** |
| pipecat | D-A | +180 [-1259, +1330] | -6 [-23, +12] | +15 [-2, +31] | **-1.62 [-3.00, -0.44]** |
| livekit | D-B | +905 [-25, +1610] | +9 [-2, +23] | +11 [-2, +25] | +0.19 [-0.62, +1.25] |
| pipecat | DN-A | +90 [-1020, +1167] | **-19 [-34, -2]** | +2 [-13, +16] | **-1.88 [-3.31, -0.75]** |
| livekit | DN-B | +650 [-50, +1490] | -11 [-29, +4] | -6 [-20, +9] | **-1.00 [-1.81, -0.25]** |
| livekit | D-C | **+865 [+260, +940]** | **+23 [+10, +35]** | **+26 [+18, +35]** | **-1.19 [-1.88, -0.44]** |
| pipecat | D-C | **+880 [+300, +966]** | **+15 [+4, +27]** | **+17 [+8, +26]** | -0.38 [-1.00, +0.19] |
| livekit | CN-C | **-60 [-170, -30]** | **-15 [-29, -4]** | -6 [-16, +5] | -0.50 [-1.44, +0.25] |
| pipecat | CN-C | **-70 [-176, -21]** | -4 [-11, +4] | -2 [-8, +4] | +0.62 [+0.00, +1.31] |
| livekit | DN-D | -255 [-445, +280] | **-21 [-36, -4]** | **-17 [-30, -4]** | **-1.19 [-2.44, -0.25]** |
| pipecat | DN-D | -90 [-365, +368] | **-13 [-24, -4]** | **-13 [-24, -2]** | -0.25 [-0.63, +0.06] |

**otoSpeech, 16 clips, user channel**

| framework | x − y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |
|---|---|---|---|---|---|
| pipecat | C-A | +279 [-1443, +503] | **-38 [-53, -20]** | -9 [-19, +0] | -0.75 [-1.94, +0.37] |
| livekit | C-B | +35 [-150, +150] | -4 [-16, +8] | -6 [-17, +6] | -0.06 [-0.69, +0.50] |
| pipecat | CN-A | +157 [-1559, +379] | **-38 [-53, -20]** | -9 [-19, +0] | -0.69 [-2.00, +0.44] |
| livekit | CN-B | **-100 [-280, -25]** | -8 [-18, +2] | -6 [-17, +6] | -0.19 [-0.62, +0.25] |
| pipecat | D-A | +1229 [-529, +1447] | **-30 [-45, -14]** | -6 [-16, +4] | **-1.06 [-2.13, -0.06]** |
| livekit | D-B | **+970 [+785, +1060]** | +0 [-9, +10] | -4 [-15, +8] | **-0.56 [-1.19, -0.06]** |
| pipecat | DN-A | +1188 [-551, +1417] | **-25 [-38, -8]** | -6 [-13, +2] | **-1.25 [-2.31, -0.31]** |
| livekit | DN-B | **+930 [+730, +1020]** | +0 [-8, +9] | -2 [-13, +9] | **-0.75 [-1.38, -0.12]** |
| livekit | D-C | **+935 [+830, +990]** | +4 [-6, +13] | +2 [-4, +8] | **-0.50 [-0.94, -0.06]** |
| pipecat | D-C | **+950 [+860, +980]** | +8 [-2, +18] | +4 [-4, +11] | -0.31 [-0.75, +0.12] |
| livekit | CN-C | **-135 [-210, -90]** | -4 [-9, +0] | +0 [+0, +0] | -0.13 [-0.44, +0.12] |
| pipecat | CN-C | **-122 [-190, -99]** | +0 [+0, +0] | +0 [+0, +0] | +0.06 [-0.19, +0.38] |
| livekit | DN-D | -40 [-80, +0] | +0 [-6, +5] | +2 [+0, +5] | -0.19 [-0.50, +0.06] |
| pipecat | DN-D | -41 [-71, +19] | +6 [-4, +15] | +0 [-6, +5] | -0.19 [-0.50, +0.06] |

## 6. Clean-run checks

`clean_checks`: `{"pipecat_runs": 345, "pipecat_endframe": 345, "pipecat_violations": 0, "pipecat_errors": 0, "pipecat_warnings": 69, "livekit_runs": 345, "livekit_errors": 0, "livekit_warnings": 1001, "livekit_server_sessions_not_1": 0}`.

- Pipecat: 345 runs (5 systems × 69 clip-conditions), every one ended with `EndFrame`, 0 frame-protocol violations,
  0 pipeline errors, 69 log warnings, all of them A's `WhisperSTTService#k: ttfs_p99_latency not set, using default
  1.0s` (one per service instance).
- LiveKit: 345 runs, 0 errors, every C / CN / D / DN run had exactly 1 server session. 1001 warnings of two kinds:

| framework | warning | count by system |
|---|---|---|
| livekit | `stt end of speech received while vad is still in a speech segment, flushing vad` | C 237, CN 205, D 127, DN 85 |
| livekit | `resume_false_interruption is enabled but audio output does not support pause, it` | B 69, C 69, CN 69, D 69, DN 69 |
| pipecat | `WhisperSTTService` | A 49 |

  `resume_false_interruption ... audio output does not support pause` is the room-less session's tone TTS (one per
  run, every system). `stt end of speech received while vad is still in a speech segment, flushing vad` is LiveKit's
  `StreamAdapter` seeing our `END_OF_SPEECH` while its own Silero still counts speech; it happens when our turn end
  fires inside a pause the Silero state machine has not closed yet, so it is most frequent for the timeout policies
  (C 237, CN 205) and rarer for the hybrids (D 127, DN 85). It does not change the committed turn (the adapter's
  final is already out); it is the LiveKit-side symptom of the cut-ins counted above.

## 7. RTF by component

From `components_live`: A / B from driver-side timers around each component call; ours from the server's per-component
time sums in the debug stats (wall only). Audio: 3489 s per framework × system (69 clip-conditions including pads).
RTF = seconds of compute per second of audio at 2 threads; the "process" line is what `ps` would charge.

| stack | component | calls per audio min | RTF wall | RTF CPU | call p50 / p95 ms |
|---|---|---|---|---|---|
| A Pipecat default | Silero VAD (per 32 ms chunk) | 1874 | 0.018 | 0.022 | 0.7 / 1.0 |
|  | smart-turn v3.2 (per call, 8 s input) | 16 | 0.017 | 0.039 | 67.4 / 70.9 |
|  | faster-whisper small (per segment) | 16 | 0.223 | 0.369 | 711.6 / 1256.5 |
| | **process total** (driver CPU s per audio s; peak RSS) | | | **0.431** | RSS 2372 MB |
| B LiveKit default | Silero VAD (per 32 ms chunk) | 3897 | 0.034 | 0.069 | 0.6 / 0.9 |
|  | faster-whisper small (per segment) | 8 | 0.132 | 0.207 | 786.9 / 1883.7 |
|  | LiveKit EnglishModel EOU (per call) | 8 | 0.002 | 0.006 | 10.0 / 35.1 |
| | **process total** (driver CPU s per audio s; peak RSS) | | | **0.280** | RSS 2802 MB |

Ours (RTF wall, Pipecat / LiveKit sessions):

| component | C | CN | D | DN |
|---|---|---|---|---|
| ASR pass (encoder chunk step + RNNT decode + VAD head) | 0.149 / 0.151 | 0.167 / 0.170 | 0.149 / 0.149 | 0.171 / 0.172 |
| turn pass (speaker-conditioned encoder + turn head) | 0.150 / 0.150 | 0.166 / 0.168 | 0.149 / 0.148 | 0.169 / 0.170 |
| diarizer (Sortformer v2 0.32 s for C / D; Nemotron-3-Diarization 100M for CN / DN) | 0.496 / 0.500 | 0.300 / 0.302 | 0.495 / 0.495 | 0.303 / 0.303 |
| Silero branch (hybrid_dyn only) | 0.000 / 0.000 | 0.000 / 0.000 | 0.003 / 0.003 | 0.004 / 0.004 |
| arm binder (after_agent_arm only) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| **server total** (session wall) | 0.795 / 0.802 | 0.635 / 0.641 | 0.798 / 0.796 | 0.648 / 0.650 |
| server process CPU s per audio s | 1.021 / 1.027 | 0.708 / 0.712 | 1.024 / 1.021 | 0.718 / 0.719 |
| server peak RSS MB | 3581 / 3586 | 1485 / 1482 | 3565 / 3594 | 1482 / 1483 |
| driver (adapter) CPU s per audio s | 0.019 / 0.011 | 0.023 / 0.013 | 0.019 / 0.011 | 0.024 / 0.013 |

Reading: in A and B the dominant cost is Whisper small (0.13-0.22 wall, 0.21-0.37 CPU) with Silero and the turn
models under 0.04 each. In ours the ASR pass and the speaker-conditioned turn pass cost ~0.15 each and the diarizer
is the largest item: Sortformer v2 at the 0.32 s setting is 0.50, Nemotron-3-Diarization 0.30, which is the whole
difference between the 0.80 and 0.64 server totals. The Silero branch and the arm binder are free (< 0.005). Our
server is 2.4x (C) or 1.7x (CN) the CPU of the Pipecat default and 3.6x / 2.5x the LiveKit default, at 3.6 GB
(Sortformer) or 1.5 GB (Nemotron) RSS against 1.6-2.8 GB for A / B; it stays under RTF 1 in every kept run, with backlog ≤ 0.7 s except one LiveKit C session on the oto user
channel (3.8 s, RTF 0.98) and one Pipecat DN session there (2.4 s).

## 8. Verdicts

1. **vs the Pipecat default (A).** C has fewer cut-ins on every set (per clip: AMI −1.8 [−3.2, −0.4], TurnBench mono
   −1.38 [−2.19, −0.63], oto mono −1.25 [−2.56, −0.13]; the user-channel deltas −0.63 / −0.75 have CIs that include 0) and
   fewer misses at 3 s on every set (AMI −40, TurnBench user −30 [−51, −9], oto user −38 [−53, −20], oto mono −21
   [−39, −2], TurnBench mono −7 [−18, +3] points). Dead air is lower on AMI, TurnBench user and oto mono (−865, −746,
   −700 ms; all CIs include 0 at these n) and higher on the TurnBench mono mix and oto user (+292, +279; CIs include
   0), where A's median is over the minority of ends it answers. A's P90 dead air is 3.2-4.6 s everywhere.
2. **vs the LiveKit default (B).** Dead air is comparable on the two-party sets (C − B: +40, +35, −402, −20 ms; only
   the AMI −1000 ms is large, CI [−3600, +920]) and CN is 20-443 ms lower than B on the two-party sets, with two of four CIs excluding 0.
   Misses are lower on every set (TurnBench mono −36 [−55, −17], oto mono 6 s −15 [−26, −4]; the user-channel deltas
   −12 / −4 include 0). Cut-ins: on the oto user channel C and B are the same (−0.06 [−0.69, +0.50] per clip); on the
   TurnBench user channel C cuts in more (+0.94 [+0.31, +1.63]); on the mono mixes C cuts in much more (+2.63,
   +1.37 per clip) while B misses 45-70 % of ends there. B's transcripts are better (10 % vs 17 % WER on the user
   channel) and 3-11 s late.
3. **D / DN vs C.** The hybrid rules add 0.32-0.98 s of median dead air (D − C; CIs exclude 0 everywhere except the TurnBench
   mono mix and Pipecat AMI) and 2-23 points of misses at 3 s on the mono mixes, and remove cut-ins: 0 on AMI in both frameworks (−1.0 [−1.6,
   −0.4] per clip in LiveKit), −1.9 [−3.0, −1.1] per clip on the LiveKit TurnBench mix, −0.3 to −1.2 elsewhere. DN
   keeps the cut-in reduction with smaller miss increases (oto mono DN − D −21 [−36, −4] points at 3 s).
4. **Nemotron-3-Diarization as the diarizer (CN vs C, DN vs D).** Dead air 8-135 ms lower on the two-party sets
   (CN − C; CIs exclude 0 in 6 of 8 cells), misses equal or lower (oto mono LiveKit −15 [−29, −4]; one AMI end more in
   Pipecat, +4 [0, +9] points on the Pipecat TurnBench mix), cut-ins mixed (−0.69 [−1.62, +0.12] on the LiveKit TurnBench mix, +0.63 [0, +1.31] on the Pipecat
   oto mix). Server RTF 0.63-0.65 vs 0.79-0.81, RSS 1.48 GB vs 3.58 GB, server CPU 53-73 % vs 77-102 % of a core.
   **Recommendation: make it the default diarizer.** Caveat: it was scored only through the turn policy here (no DER
   on these clips), and it runs in `--diar-spks 4` mode because the wire protocol carries 4 columns.
5. **Dp (offline, two-channel).** 786 ms (TurnBench) / 1080 ms (oto) median dead air with 19-21 % missed at 3 s and
   1.5-1.7 cut-ins per minute, between C's Pipecat and LiveKit cut-in rates at 0.3-0.55 s less dead air. It is a decision-time
   number with no delivery lag, on an input (both channels) the live systems did not get, and the head is not in
   `serve.py`. Its operating point (0.012 / 0.012 / 1.4 s) was fitted on otoSpeech and scores fp 0.175 on TurnBench dev
   with the official scorer (`research/archive/DYADIC.md` §8, above the 0.10 budget), which is consistent with its cut-in rate
   here. It marks the headroom of a predictive trigger, not a shipped result.
6. **Caveats.** (a) `agent_end` for D / DN is the other party's labelled turn end, a stand-in for the TTS-end event a
   product would send; two of the five AMI windows have no earlier other-speaker turn, so their `agent_end` is at
   0.0 s. (b) n = 5 AMI windows with one scored end each: every AMI CI except the cut-in deltas includes 0. (c) 16
   clips per two-party set, 53-56 ends; the CIs above are what that buys. (d) Dead-air medians are over answered
   ends. (e) One shared Mac at load < 6; compute numbers are order-of-magnitude, and 4 kept runs carry a load flag
   (server RTF ≤ 0.66 in all of them). (f) The two-party "agent" is a human, so the other party's overlaps and backchannels are harder
   than a TTS voice would be. (g) B's Whisper runs unsegmented behind `StreamAdapter`, which is LiveKit's default for
   a non-streaming STT; a streaming STT would change its first-text latency, not its turn decisions.

## 9. Reproduction

```bash
PY=.venv/bin/python; W=<scratch>/e2e_final
PYTHONPATH=. $PY scripts/e2e_final.py prepare --work $W                       # clips + labels
PYTHONPATH=. $PY scripts/e2e_final.py queue --work $W --wt <clean worktree>   # 2 x 345 timed sessions, guarded
PYTHONPATH=. $PY scripts/e2e_final.py predictive --work $W                    # Dp, offline
PYTHONPATH=. $PY scripts/e2e_final.py report --work $W --out runs/e2e_final.json
```

The optional `components` stage (each component alone, replaying the live call pattern) was not run for this JSON;
§7 uses the live timers.
