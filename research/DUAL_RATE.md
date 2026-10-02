# Dual rate: heads every 80 ms, the final transcript from 1.12 s chunks

2026-10-02. Question from the user: "Can the ASR look at long timings of 1.12 s (whatever is the max for those
models) and the rest of the network look every 80 ms?"

**Short answer: yes, it is built and measured (`audioforge-serve --final-chunk-ms 1120`).** The 160 ms pass keeps
every head, partial and turn decision exactly as today. A second pass of the same frozen encoder at 1.12 s chunks
writes the final transcript. It arrives 9-48 ms (p50-p95) after the turn end; on the 0.6B on CPU it takes ~110 ms. Turn ends and the fast final are
byte-identical to single rate. On meetings the gain is mixed:
- 115M: ICSI test −3.5 WER points; AMI test −0.6, which is not significant.
- 0.6B: ICSI test −1.9; **AMI test −0.14, under the 0.5-point bar and not significant.**

The cost is bursty. A slow chunk runs once per 1.12 s and makes that one 160 ms block 20-100 ms longer. That adds
~20-27 ms at p95 to turn-end delivery (~93 ms for the 0.6B on CPU). It also lowers the number of real-time streams when all sessions start
together.

All numbers here come from the public test splits of research/FINAL_COMPARE.md: LibriSpeech test-clean / test-other
300, AMI test 200, ICSI test 200 and the 32 live sessions, plus AMI test windows for the stream count. The two rules
of the feature (flush, final cut) were picked on held-out AMI / ICSI **train** meetings before the test runs. Code:
`audioforge/server/streams.py` (`LookaheadStream.flush_view`), `audioforge/serve.py` (`--final-chunk-ms`), adapters
`final_text`. Measurements: `scripts/research/dual_rate.py` → `runs/dual_rate.json`. Tests:
`tests/test_dual_rate.py`.

## The chunk sizes NVIDIA trained

Both encoders were trained by NVIDIA with att_context [70,0] / [70,1] / [70,6] / [70,13]. Those are 80 / 160 / 560 /
1120 ms chunks with 0 / 80 / 480 / 1040 ms of lookahead (research/archive/ENC_0P6B.md, the NeMo configs). The brief
said "[70,16] / [70,33]", and research/LATENCY_BUDGET.md said the same. That was a typo: [70,33] would be 2.72 s, a
size the models never saw. LATENCY_BUDGET.md is corrected. 1120 ms is the largest trained chunk for both cores.

## What was trained, selected and tested on what

| part | trained on | selected / tuned on | tested on (every number below) |
|---|---|---|---|
| both encoders + RNNT (115M streaming hybrid, nemotron-speech-streaming-en-0.6b) | NVIDIA's data, not by us | the four chunk sizes are NVIDIA's training choice; nothing tuned by us | LibriSpeech test-clean / test-other (300 each, seed 0), AMI test meetings IS1009b / ES2004b / TS3003b / EN2002a (200 segments), ICSI test meetings Bmr013 / Bmr018 / Bro021 (200 segments), 32 live sessions |
| the served heads (VAD, speech, turn, TS-VAD, LID) | unchanged from research/FINAL_COMPARE.md (its train / selection table) | unchanged | the latency gate checks that they give identical outputs with and without the slow pass, on the 32 live sessions |
| dual-rate rules: flush at turn end (on / off), where a slow final ends (speech cut / turn cut) | nothing is trained | **held-out AMI train meetings TS3011b / ES2015c and ICSI train meetings Bro026 / Bmr022**: 32 streams of six single-speaker segments each with 2 s gaps. These meetings are held out of every head (research/FIXALL.md). No test audio was used | the 32 live sessions (one run per setting, after the choice) |
| cost: ms per chunk, delivery delay, memory | – | – | the 32 live sessions (served engine); real-time streams on the AMI test windows (mps_115m.py protocol) |

## 1. WER at every trained chunk (masked offline forward = cache-aware streaming)

`dual_rate.py wer`: the served `.afm` (115M `stage1_served_v3`, 0.6B `served_0p6b_v0.3`; the 0.6B's RNNT tensors are
the same in every build), `transcribe(att_context_size=[70, R])`, greedy, CPU 2 threads. The masked forward equals
chunk-by-chunk streaming (`tests/test_streaming.py`). The served dual-rate engine reproduces these numbers (section
3). Whisper English normaliser, 1000-resample bootstrap over utterances (clips for the live sessions), paired against
the 160 ms row. Bold Δ = the CI excludes 0. The 160 ms rows equal FINAL_COMPARE's core rows hypothesis for
hypothesis (300 / 300 / 200 / 200 / 32 identical).

**115M core** (WER %, [95 % CI]; Δ = paired difference against 160 ms, points [95 % CI])

| test set | 80 ms [70,0] | **160 ms [70,1] (fast pass)** | 560 ms [70,6] | 1120 ms [70,13] |
|---|---|---|---|---|
| LibriSpeech test-clean (300) | 2.83 [2.23, 3.49]<br>**Δ +0.45 [+0.15, +0.77]** | **2.38 [1.86, 2.96]** | 2.21 [1.70, 2.72]<br>Δ -0.17 [-0.43, +0.04] | 2.04 [1.57, 2.55]<br>**Δ -0.34 [-0.60, -0.11]** |
| LibriSpeech test-other (300) | 7.10 [6.17, 8.05]<br>Δ +0.28 [-0.12, +0.68] | **6.82 [5.81, 7.81]** | 5.67 [4.78, 6.59]<br>**Δ -1.15 [-1.61, -0.73]** | 5.39 [4.51, 6.31]<br>**Δ -1.43 [-1.95, -0.93]** |
| AMI test (200) | 16.93 [15.18, 18.79]<br>**Δ +0.82 [+0.17, +1.49]** | **16.11 [14.26, 18.02]** | 16.28 [14.36, 18.42]<br>Δ +0.17 [-0.79, +1.26] | 15.52 [13.76, 17.49]<br>Δ -0.59 [-1.49, +0.30] |
| ICSI test (200) | 20.76 [18.29, 23.36]<br>**Δ +2.32 [+1.34, +3.46]** | **18.44 [16.34, 20.76]** | 15.60 [13.57, 17.93]<br>**Δ -2.84 [-3.82, -1.84]** | 14.93 [12.93, 17.19]<br>**Δ -3.51 [-4.57, -2.53]** |
| live calls (32) | 21.62 [17.53, 27.18]<br>**Δ +1.28 [+0.32, +2.93]** | **20.34 [16.67, 25.08]** | 19.66 [15.99, 24.41]<br>Δ -0.68 [-1.92, +0.60] | 18.69 [15.03, 23.69]<br>Δ -1.65 [-3.44, +0.19] |

**0.6B core** (WER %, [95 % CI]; Δ = paired difference against 160 ms, points [95 % CI])

| test set | 80 ms [70,0] | **160 ms [70,1] (fast pass)** | 560 ms [70,6] | 1120 ms [70,13] |
|---|---|---|---|---|
| LibriSpeech test-clean (300) | 2.76 [2.11, 3.48]<br>Δ +0.07 [-0.11, +0.25] | **2.69 [2.05, 3.37]** | 2.42 [1.83, 3.10]<br>**Δ -0.27 [-0.52, -0.05]** | 2.35 [1.77, 3.00]<br>**Δ -0.34 [-0.57, -0.12]** |
| LibriSpeech test-other (300) | 6.11 [5.26, 7.03]<br>**Δ +0.40 [+0.05, +0.76]** | **5.71 [4.90, 6.64]** | 5.17 [4.33, 6.14]<br>**Δ -0.54 [-0.92, -0.16]** | 4.89 [4.10, 5.75]<br>**Δ -0.82 [-1.24, -0.39]** |
| AMI test (200) | 8.45 [7.30, 9.74]<br>**Δ +0.58 [+0.12, +1.07]** | **7.87 [6.80, 9.07]** | 7.84 [6.62, 9.18]<br>Δ -0.03 [-0.81, +0.90] | 7.73 [6.56, 8.90]<br>Δ -0.14 [-0.76, +0.55] |
| ICSI test (200) | 11.62 [10.09, 13.22]<br>Δ +1.37 [-0.21, +2.54] | **10.25 [8.56, 12.49]** | 9.00 [7.66, 10.56]<br>**Δ -1.25 [-2.57, -0.26]** | 8.37 [7.09, 9.82]<br>**Δ -1.88 [-3.61, -0.59]** |
| live calls (32) | 12.48 [9.31, 16.35]<br>Δ +0.93 [+0.00, +2.04] | **11.55 [8.96, 14.93]** | 10.90 [8.12, 14.48]<br>**Δ -0.65 [-1.20, -0.09]** | 10.74 [8.19, 14.08]<br>**Δ -0.81 [-1.24, -0.35]** |

How to read this:
- **The meeting gain, as the brief asked for it (bar: 0.5 WER points):**
  - **0.6B on AMI test: −0.14 points, under the bar and not significant.**
  - 0.6B on ICSI test: −1.88 points (significant).
  - 115M on AMI test: −0.59 points, just over the bar but not significant (CI [−1.49, +0.30]).
  - 115M on ICSI test: −3.51 points (significant).
  - The user decides whether to ship it for AMI-like audio.
- On read speech (test-other) and on the live calls, 1120 ms is a certain gain for the 0.6B (−0.8 points on
  both). For the 115M it is −1.4 points on test-other, and −1.7 on the live calls with a CI that just touches 0.
- 560 ms gets about two thirds of the 1120 ms gain on most sets. On AMI it gets none.
- 80 ms ([70,0]) is worse everywhere, as research/LATENCY_BUDGET.md found on dev sets.
- Nothing here closes the gap to the offline models on the live calls. Parakeet-TDT v3 scores 11.04 % there and the
  0.6B at 1120 ms scores 10.74 %, the same within noise.

## 2. The rules, picked on held-out train meetings

Two choices were left open:
- **Flush.** At a turn end, the slow chunk holding the turn's last frames is usually not complete. The server can
  wait for it (up to 1.12 s of audio), or encode the audio it has as a partial chunk on a throw-away copy of the
  slow pass.
- **Final cut.** The slow final can end where the fast final ends (the frames available at the decision, "turn"), or
  3 frames past the last VAD speech frame ("speech", the rule `--asr-lookahead` already used).

`dual_rate.py select` ran each candidate through the served single-mode engine at 1120 ms. The audio was the 32
held-out streams (AMI TS3011b / ES2015c, ICSI Bro026 / Bmr022 train meetings; ~1340 s; ~200 turn ends). The rule
was decided before any test run: lowest delay, among candidates whose WER is not significantly worse than the best
(paired CI includes 0).

| core, device | candidate rule | slow-final WER % [CI] | Δ vs the chosen rule (paired) | slow-final delay p50 / p95 ms | fast-final WER % |
|---|---|---|---|---|---|
| 115M cpu | **cut speech, flush on (chosen)** | 12.31 [9.89, 15.21] | – | 19.4 / 24.4 (compute) | 13.15 |
| 115M cpu | cut speech, flush off | 12.28 [9.86, 15.17] | -0.03 [-0.11, +0.00] | 160 / 800 (audio wait) | 13.15 |
| 115M cpu | cut turn, flush on | 12.31 [9.89, 15.21] | +0.00 [+0.00, +0.00] | 22.9 / 27.1 (compute) | 13.15 |
| 115M cpu | cut turn, flush off | 12.28 [9.86, 15.17] | -0.03 [-0.11, +0.00] | 480 / 960 (audio wait) | 13.15 |
| 0.6B mps | **cut speech, flush on (chosen)** | 6.51 [5.02, 8.05] | – | 0.0 / 43.3 (compute) | 7.14 |
| 0.6B mps | cut speech, flush off | 6.40 [4.93, 7.94] | -0.11 [-0.24, +0.00] | 0 / 640 (audio wait) | 7.14 |
| 0.6B mps | cut turn, flush on | 6.51 [4.95, 8.05] | +0.00 [-0.19, +0.15] | 35.4 / 42.5 (compute) | 7.14 |
| 0.6B mps | cut turn, flush off | 6.40 [4.93, 7.94] | -0.11 [-0.24, +0.00] | 480 / 960 (audio wait) | 7.14 |

What the selection runs show:
- The flush costs no measurable WER (≤ 0.11 points, CIs include 0). It saves 0.5-1 s of waiting.
- The two cuts give the same text. The speech cut often finds its frames already decoded, so no flush is needed
  (0.6B median 0 ms). It is the default (`Engine(final_cut="speech")`).

On the held-out audio the slow pass removes 0.6-0.8 WER points against the fast final.

## 3. The served engine on the 32 live sessions

`dual_rate.py live`. The served single-mode engine (`audioforge.load`, as `audioforge-serve --mode single`, default
preset) runs on each session in 20 ms blocks, in process. The session defers the slow flush exactly as the WebSocket
handler does: `turn_end` + `final_fast` are complete when `process()` returns, then `flush_slow()` makes the slow
final. Speech cut, flush on, unless marked otherwise. 87 turn ends (115M) / 83 (0.6B).

### Latency gate (turn ends and fast finals identical to single rate)

| core, device, slow chunk | sessions with identical `turn_end` messages | sessions with identical fast finals | served WER, fast finals | served WER, slow finals | offline WER at that chunk (section 1) |
|---|---|---|---|---|---|
| 115M cpu, 1120 | 32 / 32 | 32 / 32 | 20.34 | **18.69** | 18.69 |
| 115M cpu, 560 | 32 / 32 | 32 / 32 | 20.34 | 19.66 | 19.66 |
| 115M mps, 1120 | 32 / 32 | 32 / 32 | 20.34 | 18.69 | 18.69 |
| 0.6B mps, 1120 | 32 / 32 * | 32 / 32 | 11.55 | **10.71** | 10.74 |
| 0.6B mps, 560 | 32 / 32 * | 32 / 32 | 11.55 | 10.90 | 10.90 |
| 0.6B cpu, 1120 | 32 / 32 * | 32 / 32 | 11.55 | 10.71 | 10.74 |

\* Byte-identical except `model_ms`, the measured compute time of the v5 turn classifier. It changes from run to run
anyway (7-10 of 32 sessions match on it). Decision, p, silence, path and time are identical in every message.

The gate passes on both cores and both devices. The fast final is byte for byte today's final. The served slow
finals reproduce the offline long-context WER. The 0.6B's 10.71 against 10.74 is the flushed partial chunks at turn
ends, inside noise.

### Delay of the slow final after the turn end

The flush runs after the `turn_end` batch is sent, so the slow final follows by its compute.

| core, device, slow chunk | slow final after the turn_end batch, p50 / p95 / max ms | without the flush (`--final-flush off`): audio-time wait p50 / p95 ms |
|---|---|---|
| 115M cpu, 1120 | 16.8 / 24.6 / 26.2 | 160 / 960 |
| 115M mps, 1120 | 8.6 / 22.3 / 427 (one MPS stall) | – |
| 115M cpu, 560 | 0.0 / 21.5 / 21.9 | – |
| 0.6B mps, 1120 | 35.6 / 48.2 / 70.9 | 160 / 640 |
| 0.6B mps, 560 | 0.0 / 53.6 / 69.8 | – |
| 0.6B cpu, 1120 | 105.4 / 113.1 / 129.7 | – |

### Compute, delivery and memory (32 live sessions)

"Delivery" replays each session in real time: one 20 ms block arrives every 20 ms and waits for the previous one on
one compute thread, as the server's executor does. The delay is from a block's arrival to its messages being ready.
Paired: the same turns with and without the slow pass.

| core, device | slow chunk | ms per 160 ms chunk p50 / p95 / max | RTF | turn_end delivery p50 / p95 ms | all blocks delivery p95 / p99 ms | peak RSS MB (MPS driver MB) |
|---|---|---|---|---|---|---|
| 115M cpu | – (160) | 30.5 / 32.8 / 174 | 0.194 | 30.0 / 31.5 | 30.2 / 31.9 | 1146 |
| 115M cpu | 1120 | 31.0 / 53.8 / 284 | 0.217 | 30.2 / 51.0 | 31.2 / 52.5 | 1146 |
| 115M cpu | 560 | 31.1 / 50.6 / 249 | 0.230 | 30.1 / 47.7 | 30.5 / 48.4 | 1146 |
| 115M mps | – (160) | 33.0 / 35.9 / 325 | 0.215 | 27.5 / 29.1 | 28.5 / 30.3 | 1151 (572) |
| 115M mps | 1120 | 36.5 / 52.2 / 332 | 0.251 | 26.0 / 37.9 | 28.0 / 41.6 | 1160 (599) |
| 0.6B mps | – (160) | 45.2 / 53.1 / 487 | 0.299 | 47.0 / 52.8 | 41.6 / 47.4 | 4890 (3469) |
| 0.6B mps | 1120 | 51.0 / 89.5 / 427 | 0.366 | 47.8 / 79.5 | 47.9 / 78.7 | 4840 (3911) |
| 0.6B mps | 560 | 52.9 / 88.9 / 403 | 0.393 | 47.7 / 79.5 | 47.5 / 75.5 | 4847 (3894) |
| 0.6B cpu | – (160) | 96.4 / 106.6 / 437 | 0.617 | 100.2 / 106.4 | 97.5 / 108.6 | 4875 |
| 0.6B cpu | 1120 | 97.2 / 200.2 / 570 | 0.711 | 100.9 / 199.2 | 160.6 / 203.6 | 4857 |

### Real-time streams (AMI test windows)

This is mps_115m.py's protocol: K sessions fed 160 ms block by block on one thread, real time while the p95 of the
summed block compute stays under 160 ms. "Together" starts every session at the same moment, so their slow chunks
fall in the same block. "Staggered" offsets session k by k·7/K blocks, as sessions that start at different times
would be.

| core, device | 160 ms (single rate): together / staggered | 1120 ms dual rate: together / staggered |
|---|---|---|
| 115M cpu | 4 / 4 | 2 / **4** |
| 115M mps | 4 / 5 | 3 / 4 |
| 0.6B mps | 3 / 3 | 1 / 2 |
| 0.6B cpu | 1 / 1 | **0 / 0** |

Reading the cost:
- On average the slow pass is cheap: +0.02-0.07 RTF.
- But it comes as one burst per 1.12 s. A 14-frame chunk through the whole encoder adds ~20 ms (115M) or ~40 ms
  (0.6B mps) to one block in seven, and ~100 ms on the 0.6B on CPU.
- That burst is what moves the p95s:
  - turn_end delivery: +19.5 ms (115M cpu), +8.8 ms (115M mps), +26.7 ms (0.6B mps), +92.8 ms (0.6B cpu);
  - chunk compute p95.
- Medians do not move.
- The stream count drops only when sessions start together. Staggered, the 115M keeps 4 CPU streams.
- The 0.6B on CPU has no real-time stream at 1120 ms under the p95 rule: p95 199 ms. Its median block is still
  96 ms, so it keeps up on average with ~100 ms of jitter. It was already at one stream at 160 ms.
- Memory: RSS unchanged. MPS driver +27 MB (115M) / +440 MB (0.6B).

## 4. Recommendation

- **Ship it as an option, off by default.** It does what the user asked: the heads and turn decisions stay on the
  160 ms clock, unchanged byte for byte, and the transcript comes from 1.12 s chunks. Clients get both texts:
  `final_fast` at the turn end for the LLM, then the better `final` ~10-50 ms later to replace it. The Pipecat /
  LiveKit adapters forward the fast text by default (`final_text="fast"`), so reply latency is unchanged.
- **Where it pays:** ICSI-like far-field meetings (−3.5 / −1.9 points), read speech, and the live calls on the 0.6B
  (−0.8, significant).
- **Where it does not:** AMI test, where the gain is under or near the 0.5-point bar and not significant on either
  core.
- **The price** is a 20-40 ms bump to p95 turn-end delivery, from the slow chunk's burst; the median does not move.
  It also costs streams when many sessions start in lockstep.
- **Do not use it with the 0.6B on CPU:** +93 ms p95 on turn-end delivery, no real-time stream under the p95 rule.
- For transcript quality on meetings, `--final-asr tdt_v3` remains much stronger on the 115M (research/archive/HYBRID_ASR.md).
- Possible follow-up (not done): spread the slow chunk's encoder layers over the seven fast blocks, or run the slow
  pass on its own thread, to remove the burst.

## How it works (code)

- `Engine(final_chunk_ms=560|1120)` makes the session open a second `LookaheadStream` at [70,6] / [70,13] over the
  same audio. It is text only; no head reads it. The fast `ASRStream` is not touched, which is why the gate holds by
  construction. The test checks it anyway.
- At a cutting turn end the server sends `final_fast`: the old `final` dict with a new `type`.
- The slow final (`type: final`, `source: slow`, `pass: slow`, `start` / `end` / `latency_ms`) then covers the slow
  tokens up to 3 frames past the last VAD speech frame:
  - When those frames' slow chunk needs audio past the decision time, `LookaheadStream.flush_view(t_dec)` copies
    the pass's state and encodes the mel frames up to the decision as a partial attention chunk (`final=False`:
    complete frames only).
  - The copy is decoded and discarded; the real pass continues with full chunks.
  - A one-chunk snapshot lets the flush rewind when a client block reached past a chunk end. The flushed text then
    does not depend on how the client cut its audio (tested).
- Under load shedding level 2 the slow pass is dropped. Its finals then carry the fast text with `pass: fast`.
- In the WebSocket handler `turn_end` + `final_fast` go out first, and `flush_slow()` runs right after the send.

Tests (`tests/test_dual_rate.py`, tiny models with NeMo-aligned subsampling):
- every frame / partial / turn_end is identical to single rate, and `final_fast` equals today's `final`;
- with the flush off, the slow finals partition exactly the tokens of the encoder at the long chunk, which equals the
  offline masked forward;
- the flush leaves the stream unchanged and gives the same text for 20 ms and 250 ms client blocks;
- the deferred flush matches the in-batch flush;
- socket protocol, flag validation, both cut rules.

Adapters: `tests/test_pipecat_integration.py` / `tests/test_livekit_integration.py` `final_text` fast / slow.
