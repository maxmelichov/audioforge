# Live integration: streaming server, Pipecat, LiveKit (2026-09-26)

Sources: the server, Pipecat and LiveKit agents' reports, and the independent re-run in
[`research/archive/INTEGRATION_VERIFY.md`](INTEGRATION_VERIFY.md). Where the verifier's number differs from a report's, this
file uses the verifier's number and says so. All measurements are CPU only on this Mac (24 GB), shared with a GPU job
and other agents (load average 3–6 unless noted).

## 1. Executive summary

- **What runs end to end today:** `audioforge/serve.py`, a WebSocket server. It runs the frozen NVIDIA streaming ASR with our VAD and turn heads (`runs/stage1_heads_pretrained.afm`, 160 ms chunks) and NVIDIA Streaming Sortformer v2 (`runs/nemo_sortformer_v2.afm`; card "ultra low latency" 0.32 s setting by default since 2026-09-26, `--diar-config low_latency` for the 1.04 s setting the rest of this file measured) on one 80 ms clock. It emits `ready`/`frame`/`partial`/`turn_end`/`final`/`stats`. Two adapters drive it on real AMI meeting audio at 1× real time: a Pipecat 1.12 STT, VAD and turn analyzer, and a LiveKit Agents 1.8.3 STT, VAD and turn detector, tested inside a room-less `AgentSession`. Tests: 38+21+15 pass, and the real-model tests pass with `RUN_REAL=1` (verified).
- **Speed (verified, 2 threads, 1×):** RTF 0.45–0.52. It keeps up: frames arrive p50 129–136 ms (p95 up to 220 ms) after their audio, with no growth over the stream. The 6–86 ms figure is the structural availability, without compute. Chunk time p50/p95 is 38/195–250 ms. The first partial arrives about 55 ms after its audio is sent, which is 89–357 ms after the first word ends. Server peak RSS is 3.0–3.45 GB. It keeps up **only because of the fast-conv CPU path**: without it RTF is 1.13–1.21 and the server falls behind. The reported as-fast-as-possible RTF of 0.33/0.27 did **not** reproduce; the verifier measured 0.42–0.53.
- **Diarizer setting and lag (changed 2026-09-26): default `--diar-config low_latency_032`** (chunk 3 + right context 1 = 0.32 s, `AOSCConfig.preset`). A speaker column is final 80–240 ms (mean 160) after its frame ends, vs 560–960 ms (mean 760) for the previous `low_latency` (chunk 6 + right context 7 = 1.04 s), which stays available. Why: [`EOT_BENCH_V2.md`](EOT_BENCH_V2.md) §7 (974 AMI dev turns, leak-free cross-fit) finds the same miss rates at both settings (causal, 6 s: hybrid 63.7 vs 61.9 %, head 65.8 vs 66.5 %, timeout 70.1 vs 74.8 %, CIs overlapping) and 0.5–0.8 s lower P50 wherever systems fire (nominal emission delay 240 vs 840 ms), at DER 28.3 vs 26.3. The `ready` message adds `diar_config` and `column_lag_ms` (the structural mean); existing keys are unchanged, and both adapters ignore unknown `ready` keys.
- **Measured, both settings** (one process, CPU 2 threads, 1×, 2 AMI dev windows IS1008b_003 15.2 s / ES2011b_027 20 s, load ~1.5–2.4, `--debug-fields`):

  | `--diar-config` | RTF | chunk p50 / p95 ms | frame arrival lag p50 / p95 ms | speaker-column age on arrival p50 (p95) ms | max backlog ms |
  |---|---|---|---|---|---|
  | `low_latency_032` (default) | 0.49 / 0.53 | 80–89 / 148–163 | 126 / 210–229 | 210–227 (291) | 120–140 |
  | `low_latency` | 0.40 / 0.43 | 37 / 174–195 | 124–125 / 154–160 | 857 (1035) | 120 |

  The 0.32 s diarizer runs about 3× as often (RTF +0.1) and the server still keeps up at 1× on 2 threads (lag slope 2.4–2.6 ms/s, backlog ≤ 140 ms). The same 1 s timeout fires 240 ms earlier on these windows; transcripts are identical. The debug stat `diar_lag_ms_mean_measured` is computed from sample counts, not wall-clock time, so it always reads the structural mean (160 / 760) at 1× (D8).
- **Turn policy default (changed after verifier D1):** a **plain 1000 ms silence timeout on the diarizer's label-free primary column** (`turn_policy: "timeout"`; primary = the column with the most activity in the last 5 s, no oracle, no enrollment). It fires once the primary has been inactive for `timeout_ms`, whether or not another column is active. The old rule, which also waits for "nobody else active", is available as `turn_policy: "timeout_quiet"` (its `turn_end` is still tagged `timeout`; the protocol is otherwise unchanged). Why: in STAGE1 (n=200 AMI, misses at ≤ 5% false cutoffs) the plain timeout on the Sortformer primary misses 38.4%, while "primary silent AND nobody else" misses 66–68%, about the same as the served head (69%). The server previously shipped the second rule while citing the first rule's number. The head's probability is exposed only as `eot` and through the opt-in `head`/`both` policies. **Caveat:** STAGE1's 38.4% picks the column with the oracle primary, so the head-vs-timeout comparison for the shipped rule has to be made on the label-free primary.
- **eot-bench v2 (label-free enrollment)** raises every streaming system's miss rate substantially, the timeout's included. See [`research/archive/EOT_BENCH_V2.md`](EOT_BENCH_V2.md) (to be written).
- **Turn-taking quality, 0.32 s default (re-measured 2026-09-26, §4 "Default preset", `runs/integration_deadair_032.json`):** product-level dead air with the 1000 ms timeout has a median of **1.92 s in Pipecat and 1.68 s in LiveKit** (1.60–3.40 s and 1.44–2.47 s over the 5 windows). The decision itself comes a median 1.52 s after the true end. Run back to back under the same load on IS1008b_003 / ES2011b_027, the 0.32 s setting gives 1.60 / 1.66 s against **1.82 / 2.40 s** for the 1.04 s setting, with the same finals. The structural floor is about 1.2–1.4 s (1.0 s + 0.08–0.24 s + compute), against about 1.7 s for the 1.04 s setting, where clean ends gave 1.8–2.6 s. The timeout still ends turns during filled pauses ("um") or when the primary resumes within 1 s: 3 cut-ins over 5 windows in Pipecat, whose VAD gate and `resume_ms` drop some of them, and 5 in LiveKit `stt` mode.
- **`hybrid` at the product level does not beat `timeout` today:** with the served head (`stage1_heads_pretrained`, session input) `eot` never reaches 0.998: its maximum is 0.94–0.985 inside the windows and 0.983–0.995 in the padding. So `hybrid` at timeout_ms 1000 fires exactly when the timeout does, with the same dead air (medians 1.76 s Pipecat / 1.65 s LiveKit; the differences are delivery noise) and the same cut-ins (3 / 5). At timeout_ms 4000, the leak-free backstop, it is simply a 4 s timeout: **0 cut-ins, but a median dead air of 4.6 s (Pipecat) / 5.2 s (LiveKit)**. EOT_BENCH_V2 §7 chose θ≈0.998 for `stage1_turn_v3_trail6` on the bound diarizer track, which is a different head and input from what the server serves.
- **Shipped rules, §8 (2026-09-26):** `hybrid_dyn` / `hybrid_silero` (head OR any-speaker Silero silence) and `--enroll after_agent_arm` are in the server and both adapters. On the 5 windows they respond later than the 1000 ms timeout (median dead air +0.66 / +0.75 s for hybrid_dyn in Pipecat / LiveKit, +1.2 s for hybrid_silero) with 0 cut-ins vs 3 / 5; arming changes nothing measurable. Default unchanged (`timeout`).
- **Not tested:** a real LiveKit room (WebRTC/Opus, network), a real Pipecat transport, real microphones, echo of the agent's own voice and full duplex/barge-in, concurrent sessions under load, and real LLM/TTS.

## 2. Architecture

### Event protocol (`audioforge/serve.py`, fixed; `validate()` enforces exact keys)

Client → server: binary frames of int16 LE mono PCM, any frame size, 16 kHz unless configured. Optional first text frame
`{"type":"config","turn_policy":"timeout"|"timeout_quiet"|"head"|"both","timeout_ms":1000,"eot_threshold":0.98,"sample_rate":16000}`.
`{"type":"end"}` flushes. Other rates are resampled linearly.

| server → client | keys | when |
|---|---|---|
| `ready` | model, chunk_ms, frame_ms, diar_config, column_lag_ms | after connect (the last two added 2026-09-26: preset name, `+custom` with `--diar-set`; mean structural column lag in ms) |
| `frame` | t, vad, eot, speakers[4], primary | one per 80 ms frame, when its ASR frame is ready |
| `partial` | t, text | transcript changed (text since the last final) |
| `turn_end` | t, policy, p, silence_ms | a policy fired; `t` = the audio time at which the deciding data was complete |
| `final` | t, text, speaker | right after every turn_end of the cutting policy, and at `end` |
| `stats` | rtf, chunk_ms_p50, chunk_ms_p95, first_partial_ms, peak_rss_mb | after the end final; then the server closes |

`--debug-fields` adds `frame.spk_t`, `turn_end.frame_t` and a stats breakdown. All compute runs on one worker thread
shared by all sessions. Events and finals are identical across client frame sizes, thread counts and speeds (checked on
the real models): the final at a `turn_end` holds exactly the ASR tokens available at `t`.

### Clock alignment and lag

- Frame v covers [80v, 80(v+1)) ms, and the protocol `t` is its end. Both models produce the same number of frames.
- **ASR:** one encoder `stream_step` per 160 ms chunk. It feeds the RNNT (decoded frame by frame), the VAD head and the turn head's `step()`. Its two frames are available 6–86 ms after each frame ends (verified live arrival p50 about 130 ms including compute; the README's "within about 90 ms" is structural only).
- **Diarizer:** `StreamingDiarizer`, window mode, FIFO 188 / update 144 / cache 188 / left context 188, chunk C / right context R from `--diar-config`: `low_latency_032` (default) C 3 / R 1, `low_latency` C 6 / R 7. Column t is final at audio time ((t // C + 1)·C + R)·80 ms, i.e. **80–240 ms (mean 160) after the frame ends with the default, 560–960 ms (mean 760) with `low_latency`**, plus compute (per diarizer step p95 79–96 ms for C 3 at 2 threads; the C 6 steps are fewer).
- A `frame` message carries the newest diarizer column available when the ASR frame is ready, so `speakers`/`primary` describe older audio than `t`: measured column age on arrival p50 210–227 ms with the default, about 860 ms with `low_latency`. The column's own time (`spk_t`) is sent only in debug mode (D5). Until the first column exists (~0.3 s with the default, ~1 s with `low_latency`), `speakers` is `[0,0,0,0]` (D4).
- `StreamingRuntime` is not used: it implements only RoPE, not the imported encoder's rel_pos / xscaling / NeMo subsampling.
- **fast-conv:** both encoders' conformer convolutions are re-bound to an unfold/`F.linear` path. Tokens are identical and diarizer probabilities differ by ≤ 3e-7. It gives 2.5× end to end (verified); `--no-fast-conv` turns it off.

### Turn policies

- **`timeout` (default):** fires when the label-free primary column has been ≤ 0.5 for ≥ `timeout_ms` (1000 ms = 13 silent frames) and the primary spoke since the last timeout `turn_end`. Other columns are ignored. It runs on the diarizer's own frames as they are finalized, so `turn_end.t` ≥ last primary frame + 1.0 s + the column lag (0.08–0.24 s default, 0.56–0.96 s `low_latency`).
- **`timeout_quiet`:** the same, but it also waits until no other column is active on that frame. This was the default before D1; STAGE1's "primary silent AND nobody else" rule scored 66–68% misses at n=200. Its `turn_end` is tagged `timeout`, and it cuts the finals.
- **`head`:** fires when `eot` crosses `eot_threshold` (0.98) upwards; speech (VAD > 0.5) re-arms it. The head was trained with speaker-kernel conditioning. With `--turn-input session` (the default) it gets no speaker input, which is off its training distribution. `--turn-input diar` runs a second, speaker-conditioned encoder pass on the diarizer's clock (RTF 0.58–0.61, still keeps up).
- **`both`:** the plain timeout and the head both fire, each tagged with its policy. The timeout cuts the finals.

## 3. How to run

Port 8765 is held by Cursor on this Mac, so use a free port such as 8791. Put `TMPDIR` in a scratch directory.

```bash
# server (about 3.2 GB RSS; --debug-fields for lag diagnostics; --turn-input diar for the conditioned head)
PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_heads_pretrained.afm \
    --diar runs/nemo_sortformer_v2.afm --port 8791 --threads 2
# plain client: streams a file at real time, logs every event, optional scoring against a reference JSON
PYTHONPATH=. .venv/bin/python scripts/stream_client.py FILE.wav --url ws://127.0.0.1:8791 --speed 1 \
    --policy timeout --log ev.jsonl --summary s.json [--ref ref.json]
# Pipecat demo (pipecat-ai 1.12.0; mock LLM/TTS, no keys): prepare AMI windows, then run
PYTHONPATH=. .venv/bin/python examples/pipecat_local_demo.py prepare --n 5 --out DIR
PYTHONPATH=. .venv/bin/python examples/pipecat_local_demo.py run DIR/*.wav --url ws://127.0.0.1:8791 \
    --policy timeout head --out results.json
# LiveKit offline demo (no LiveKit server): plugin streams + a room-less AgentSession
PYTHONPATH=. .venv/bin/python examples/livekit_offline_demo.py DIR/*.wav --url ws://127.0.0.1:8791 \
    --agent-session both --out lk.json
# tests
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve.py tests/test_pipecat_integration.py tests/test_livekit_integration.py
RUN_REAL=1 LIVEKIT_DEMO_WAV=DIR/<win>.wav .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_livekit_integration.py
```

- **Pipecat wiring** (`audioforge/integrations/pipecat.py`): `AudioforgeSTTService` is a `WebsocketSTTService` (interim frames plus `TranscriptionFrame(finalized=True)`). `AudioforgeVADAnalyzer` goes in `LLMUserAggregatorParams(vad_analyzer=)`. `AudioforgeTurnAnalyzer` goes in `TurnAnalyzerUserTurnStopStrategy`. All three share one server session. The adapter has two defaults on: a gate that waits for VAD quiet before stopping the turn, and `resume_ms=240`, which drops a stale decision once speech resumes. Pipecat's volume gate stays off unless set explicitly. There is no file transport in Pipecat 1.12, so the demo uses its own `WavInputTransport`.
- **LiveKit wiring** (`audioforge/integrations/livekit.py`): `AudioforgeFrontend(url)` opens one server session and hands out `stt()`, `vad()` and `turn_detector()`.
  - `partial` → INTERIM and `final` → FINAL, with `speaker_id="S<k>"` and `metadata["audioforge"]`.
  - The cutting `turn_end` → END_OF_SPEECH.
  - The VAD gives one INFERENCE_DONE per frame, START after 160 ms and END after 560 ms, and hears any speaker.
  - **Recommended:** `turn_detection="stt"`, which commits exactly on the server's decision, 2 ms after it arrives (verified).
  - `AudioforgeTurnDetector` implements LiveKit's **private** `_StreamingTurnDetector`. The text-only `_TurnDetector` is not implemented, because it cannot carry an acoustic decision.

### LiveKit with a real room (not done here)

1. `brew install livekit && livekit-server --dev` (keys `devkey`/`secret`, `ws://127.0.0.1:7880`), or
   `docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev`.
2. Start the audioforge server as above, on port 8791.
3. `LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret AUDIOFORGE_URL=ws://127.0.0.1:8791 AUDIOFORGE_TURN=stt AUDIOFORGE_LOG=worker.jsonl PYTHONPATH=. .venv/bin/python examples/livekit_agent_worker.py dev`
   (`AUDIOFORGE_TURN=detector` selects the audio detector hook; stub LLM/TTS, one idle job process).
4. `LIVEKIT_URL=... LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret PYTHONPATH=. .venv/bin/python examples/livekit_publish_wav.py WIN.wav --room audioforge-test --log caller.jsonl`
   publishes the WAV as a microphone track at real time. The agent is auto-dispatched. The Agents Playground with a real microphone works too.
5. Compare `worker.jsonl` (turn commits) and the caller's `lk.transcription` times with the LiveKit table in §4.

Without a LiveKit server, the worker retries `Cannot connect to host 127.0.0.1:7880` and the publisher raises
`livekit.rtc.room.ConnectError ... Connection refused (os error 61)`. This is expected.

## 4. Measured tables

Definitions:
- **decision:** `turn_end.t` minus the true (forced-aligned) end of the primary's turn. It does not depend on machine load.
- **dead air:** arrival of the decision at the client, minus the true end.
- **cut-in:** a decision while the primary speaker keeps talking within the next 1 s.
- **WER:** all transcripts against all speakers' words in the window, through `normalize_text`.

The AMI windows are 5 dev turn windows from STAGE1's n=200 set.

### Server alone (server report, 1×; load 3–6; n = 3 AMI windows + 1 LibriSpeech utterance per row group)

| file (length) | threads | RTF | chunk ms p50/p95 | first partial ms (after send / after 1st word end) | max backlog ms |
|---|---|---|---|---|---|
| AMI IS1008b (15.2 s) | 2 | 0.433 | 33 / 222 | 50 / 350 | 160 |
| AMI ES2011b (20.0 s) | 2 | 0.479 | 35 / 260 | 51 / 81 | 200 |
| AMI TS3004b (16.6 s) | 2 | 0.447 | 35 / 223 | 50 / 830 | 180 |
| LibriSpeech (10.4 s) | 2 | 0.520 | 45 / 223 | 121 / 361 after speech onset | 180 |
| AMI IS1008b | 4 | 0.398 | 39 / 175 | 55 / 355 | 120 |
| AMI ES2011b | 4 | 0.422 | 35 / 202 | 78 / 108 | 140 |
| AMI TS3004b | 4 | 0.378 | 33 / 180 | 51 / 831 | 100 |
| LibriSpeech | 4 | 0.341 | 31 / 114 | 65 / 305 after speech onset | 80 |

Verifier re-run (2 threads, load 4.8–5.7; IS1008b, ES2011b, LibriSpeech 1188-133604-0003):
- RTF 0.453–0.519 (0.488/0.533 with `both`); chunk p50/p95 37.8–38.8 / 195–250 ms.
- Frame lag p50 129–136 ms, p95 167–220 ms, slope 2.9–4.2 ms/s (no growth); backlog max 180/200 ms.
- First partial 357/89 ms after the first word end, 54–59 ms after send.
- Peak RSS 3.01–3.45 GB (clients 0.2–0.3 GB).
- WER: LibriSpeech 0%, AMI 28.3% / 25.8%. The server report had LibriSpeech 3.6% on 1089-134686-0000 and AMI 26/28/58%.
- **As fast as possible:** 0.42–0.53, **not** the claimed 0.33/0.27. Speed 0 is no faster than 1× here.

Timeout decisions on the raw windows:
- IS1008b: +1.68 s late.
- ES2011b: cut at 16.88 s, during a 1.56 s "um", 0.93 s before the true end.
- TS3004b: about 0.9 s early (server report).

The head with `session` input never reaches 0.98 on the raw windows (max `eot` 0.84–0.96). With `--turn-input diar` it
fires 1.3–8.6 s early on all 3 windows (server report, n=3).

### Default preset `low_latency_032` (0.32 s): Pipecat and LiveKit dead air, re-run 2026-09-26

Setup: the same 5 windows, one server per session (`--threads 2 --debug-fields`, CPU, 1×), and `runs/integration_deadair_032.json`, which holds every per-window field and the commands.
- **Padding:** Pipecat 3.0 s and LiveKit 2.5 s, as before. The hybrid@4000 runs get 5.5 s so that the backstop can land.
- **Hybrid runs:** neither the demos' CLIs nor the adapters accept `hybrid` (their whitelist is timeout|head|both). A scratch driver called the unmodified `run_files` / `amain` after widening that whitelist at runtime, so no code changed. Hybrid uses θ 0.998.
- **Dead air:** Pipecat measures it at the `UserStoppedSpeakingFrame`. LiveKit measures it at the stream-mode STT END_OF_SPEECH, and the "commit" column is the `AgentSession(turn_detection="stt")` user-turn commit.
- **Cut-ins** are decisions while the primary speaks again within 1 s. For Pipecat they are counted after its VAD gate and `resume_ms=240`. For LiveKit every cutting `turn_end` counts, scored against the same `primary_intervals`.
- **LiveKit dead air** takes the first decision at or after the labelled end, even when the primary resumes within 1 s (IS1008b_014: 1.44 s, which is also a cut-in). Pipecat counts that decision as an interruption instead (3.40 s).

Dead air ms / cut-ins (LiveKit: STT dead air / cut-ins / AgentSession commit ms). The decision (`turn_end.t` − true end) is identical for `timeout` and `hybrid` 1000 on every window.

| window | decision ms | Pipecat timeout | Pipecat hybrid 1000 | Pipecat hybrid 4000 | LiveKit timeout | LiveKit hybrid 1000 | LiveKit hybrid 4000 | WER |
|---|---|---|---|---|---|---|---|---|
| ES2011b_027 | 1520 | 1923 / 0 | 1761 / 0 | 4643 / 0 | 1684 / 1 / 1674 | 1690 / 1 / 1653 | 4536 / 0 / 4563 | 0.258 |
| IB4002_119 | 1520 | 1761 / 1 | 1722 / 1 | 4642 / 0 | 1706 / 0 / 1844 | 1650 / 0 / 1661 | 4540 / 0 / 4706 | 0.433 |
| IS1008b_003 | 1440 | 1603 / 0 | 1621 / 0 | 4482 / 0 | 1578 / 1 / 1602 | 1584 / 1 / 1588 | 5163 / 0 / 4453 | 0.283 |
| IS1008b_014 | 3200 (LK 1280) | 3402 / 1 | 3382 / 1 | 7020 / 0 | 1437 / 1 / 1438 | 1411 / 1 / 1422 | 9324 / 0 / 10509 | 0.357 |
| TS3004b_064 | 2320 | 2502 / 1 | 2603 / 1 | 5382 / 0 | 2473 / 2 / 2493 | 2449 / 2 / 2449 | 7387 / 0 / 5358 | 0.582 |
| **median / total cut-ins** | 1520 | **1923 / 3** | **1761 / 3** | **4643 / 0** | **1684 / 5 / 1674** | **1650 / 5 / 1653** | **5163 / 0 / 4706** | 0.357 |

- **Sanity:** all 25 Pipecat runs (15 + 6 + 4) ended with `EndFrame`, with 0 frame-rule violations, 0 warnings and 0 pipeline errors. In all 25 LiveKit stream runs there was 1 server session, #END_OF_SPEECH equalled #cutting `turn_end`, and every END_OF_SPEECH came right after a FINAL. Transcripts and WER are identical across policies and presets.
- **Server, 0.32 s:** RTF 0.65–0.80 in Pipecat and 0.52–0.92 in LiveKit at load 2.8–5.9, with the other agents' CPU jobs running. Frame lag p50 is 130–166 ms and chunk p95 204–239 ms. It keeps up, with backlog ≤ 540 ms, except in the two slowest hybrid@4000 LiveKit runs (backlog 1.65 / 2.3 s, RTF 0.83 / 0.92), which inflate their dead air.

**Paired with the 1.04 s setting (`--diar-config low_latency`).** Both presets were run back to back on IS1008b_003 / ES2011b_027 with `timeout` 1000 at load 3.7–4.5:

| window | preset | decision ms | Pipecat dead air / cut-ins | LiveKit STT dead air / cut-ins | LiveKit commit ms |
|---|---|---|---|---|---|
| IS1008b_003 | 0.32 s | 1440 | 1602 / 0 | 1613 / 1 | 1587 |
| IS1008b_003 | 1.04 s | 1680 | 1821 / 0 | 1812 / 1 | 1818 |
| ES2011b_027 | 0.32 s | 1520 | 1661 / 0 | 1663 / 1 | 1720 |
| ES2011b_027 | 1.04 s | 2240 | 2402 / 1 | 2400 / 1 | 2396 |

- The 0.32 s setting saves 0.2–0.7 s of dead air per window. On ES2011b the 1.04 s setting's post-end decision comes 720 ms later. Server RTF is 0.60–0.70 for 0.32 s and 0.39–0.41 for 1.04 s; backlog is ≤ 620 and ≤ 260 ms.
- The first 1.04 s session, all three policies, ran at load 6–9, where the server fell behind (RTF 0.88–1.05, backlog up to 3 s). Its decisions match the table: hybrid 1000 equals the timeout, and hybrid 4000 is +4.56 / +5.12 s. Its dead air is load-inflated and is kept only in the JSON.
- The 4.64 / 4.88 s decision on IS1008b_003 is a LiveKit cut-in at both presets. The older LiveKit table below shows 0 early cuts there.

### 4b. Served `stage1_turn_v3_trail6` head with diarizer input (`--turn-input diar`), 0.32 s preset, 2026-09-26

Question: §4 found `hybrid` ≡ `timeout` because the served head (`stage1_heads_pretrained`, session input) never reaches θ 0.998. The leak-free win in [`EOT_BENCH_V2.md`](EOT_BENCH_V2.md) §7 used a different head and input: `runs/stage1_turn_v3_trail6.afm` fed the bound diarizer track. This re-runs the same 5-window Pipecat + LiveKit measurement with that head served, `--asr runs/stage1_turn_v3_trail6.afm --diar runs/nemo_sortformer_v2.afm --turn-input diar --threads 2 --debug-fields` (default `low_latency_032`). Everything else as §4: same WAVs/refs, pads (Pipecat 3.0 s, LiveKit 2.5 s, 5.5 s for the 4000 ms backstop), scratch driver for `hybrid`, definitions. Data: `runs/archive/integration_deadair_032_trail6.json`.

- **Checkpoint check.** `stage1_turn_v3_trail6.afm` holds heads `rnnt, ctc, vad, eou, spk, diar, turn`. Its encoder (657 tensors), RNNT (11) and VAD (4) weights are bit-identical to `stage1_heads_pretrained.afm`; only `heads.turn` differs (13 vs 10 tensors, `needs_cols=True`, so `--turn-input auto` also selects `diar`). Transcripts and WER are therefore identical to §4 on every window and policy (0.258 / 0.433 / 0.283 / 0.357 / 0.582).
- **Does the head reach its threshold with diar input?** Yes, unlike the session-input head: per window max `eot` 0.9977 / 0.9917 / 0.9915 / 0.9983 / 0.9967 (ES2011b / IB4002 / IS1008b_003 / IS1008b_014 / TS3004b; stream_client traces, `eotmax6.json`), every maximum after the true end. θ 0.98 is crossed in all 5 windows, θ 0.998 in 4 of 5 (ES2011b +2.72 s, IS1008b_003 +3.85 s, IS1008b_014 +2.0 s, TS3004b +2.32 s after the true end; IB4002 peaks at 0.9975), **but every 0.998 crossing comes 0.7–2.4 s after the 1000 ms timeout's decision (+1.28 to +2.32 s), and 3 of the 4 lie in the appended silence.** Inside the primary's own speech before the end the head reaches 0.9969 (ES2011b) and 0.9933 (TS3004b), so no θ separates trailing silence from speech on these 5 windows.
- **θ chosen from the served distribution: 0.99.** It crosses in the trailing silence of all 5 windows (+0.08 to +2.0 s) and inside speech in ES2011b (−1.44 s) and TS3004b (−2.96 s), the two windows where the 1000 ms timeout itself cuts in; θ 0.995 crosses post-end in only 3/5 with the same two in-speech crossings. 0.99 is picked in-sample on n = 5 and is not a held-out operating point.
- **Server, diar input:** the second speaker-conditioned encoder pass costs RTF +0.1–0.15. Pipecat sessions (load 3.2–5.1): RTF 0.65–0.91, backlog ≤ 1.02 s, frame lag p50 130–170 ms, peak RSS 3.3–3.4 GB. LiveKit timeout/head/hybrid-1000 sessions (load 3.7–5.7): RTF 0.72–1.09, backlog ≤ 0.96 s except one run (IS1008b_014 hybrid 0.99/1000: 2.84 s). **The LiveKit 4000 ms sessions ran at load 5.6–10 (RTF 0.86–2.3, backlog up to 11 s) and a re-run at load 8–10 was worse (RTF 1.0–2.4): their dead air and commit times are load-inflated and marked \*; their decisions (`turn_end.t` − true end) are identical between run and re-run and are valid.**

**Pipecat** (pipecat-ai 1.12; dead air ms / cut-ins; decision ms in parentheses where it differs from the timeout's; "dropped" = the head fired but the adapter's 240 ms resume rule discarded it because someone spoke on, and it did not re-fire in the pad):

| window | timeout 1000 | head 0.98 | hybrid 0.998 / 1000 | hybrid 0.998 / 4000 | hybrid 0.99 / 1000 | hybrid 0.99 / 4000 |
|---|---|---|---|---|---|---|
| ES2011b_027 (dec 1520) | 1702 / 0 | dropped (fired −1.36 s, in speech) | 1681 / 0 | 2942 / 0 (2720, head) | **1022 / 0 (566, head)** | 1022 / 1 (566; extra head firing −1.12 s) |
| IB4002_119 (1520) | 1701 / 1 | 1461 / 0 (1046) | 1682 / 1 | 5702 / 0 (4400, timeout) | 1700 / 1 | 2440 / 0 (2000, head) |
| IS1008b_003 (1440) | 1601 / 0 | 1862 / 0 (1446) | 1602 / 0 | 4522 / 0 (3846, head) | 1622 / 0 | 2361 / 0 (2160, head) |
| IS1008b_014 (3200) | 4281 / 1 | dropped (fired +0.57 s) | 3421 / 1 | 6562 / 0 (6320) | 3422 / 1 (head +0.8 s dropped) | 6500 / 0 (6320) |
| TS3004b_064 (2320) | 2522 / 2 | dropped (fired −2.71, +0.17 s) | 2521 / 1 | 2521 / 0 (2320, head 0.9981) | 2521 / 0 (head +0.4 s dropped) | 5541 / 0 (5200) |
| **median / total cut-ins** | **1702 / 4** | 1662 (2 of 5 delivered) / 0 | **1682 / 3** | **4522 / 0** | **1700 / 2** | **2440 / 1** |

**LiveKit** (livekit-agents 1.8.3, room-less; STT dead air ms / cut-ins / `AgentSession(turn_detection="stt")` commit ms; decision in parentheses where it differs from the timeout's; \* = load-inflated, see above):

| window | timeout 1000 | head 0.98 | hybrid 0.998 / 1000 | hybrid 0.998 / 4000 | hybrid 0.99 / 1000 | hybrid 0.99 / 4000 |
|---|---|---|---|---|---|---|
| ES2011b_027 (1520) | 1697 / 1 / 1701 | none after the end (fired −1.36 s, cut-in; commit 7505 = end flush) | 1695 / 1 / 1714 | 6473\* / 0 / 5345\* (2720) | **995 / 1 / 1005 (566)** | 1689\* / 1 / 707 (566) |
| IB4002_119 (1520) | 1688 / 0 / 1687 | 1475 / 0 / 1476 (1046) | 2241 / 0 / 2370 | 5483\* / 0 / 5051\* (4400) | 1823 / 0 / 1928 | 5967\* / 0 / 8313\* (2000) |
| IS1008b_003 (1440) | 1604 / 1 / 1598 | 1867 / 0 / 1866 (1446) | 1591 / 1 / 2061 | 4495\* / 0 / 4523\* (3846) | 1620 / 1 / 1747 | 6257\* / 0 / 7830\* (2160) |
| IS1008b_014 (1280) | 1481 / 1 / 1565 | 999 / 0 / 1007 (566) | 2323 / 1 / 1509 | 3124\* / 0 / 4375\* (2006) | 4088\* / 1 / 1058 (800; RTF 1.09) | 21893\* / 0 / 10504\* (800) |
| TS3004b_064 (2320) | 2529 / 2 / 2757 | 613 / 0 / 595 (166; plus an early cut at −2.71 s) | 2529 / 2 / 2570 | 3189\* / 0 / 3435\* (2320) | 1588 / 2 / 956 (400) | 4967\* / 1 / 3759\* (400) |
| **median / total cut-ins / median commit** | **1688 / 5 / 1687** | 1237 (4 of 5) / 1 / 1476 | **2241 / 5 / 2061** | 4495\* / 0 / 4523\* | **1620 / 5 / 1058** | 5967\* / 2 / 7830\* |

- **Sanity:** all 30 Pipecat runs ended with `EndFrame`, 0 frame-rule violations, 0 warnings, 0 pipeline errors. All LiveKit runs had 1 server session and every END_OF_SPEECH followed a FINAL; in the `head` runs 3 windows show one more END_OF_SPEECH than cutting `turn_end`s (the end-of-stream flush final), which the §4 sanity rule counts as a mismatch; the hybrid and timeout runs all pass it.
- **hybrid 0.998 / 1000 ≡ timeout** again, now for a different reason: the head does reach 0.998, but only after the timeout has already fired. Its dead air matches the timeout within delivery noise (Pipecat 1602–3421 vs 1601–4281; the IS1008b_014 timeout run carried an 820 ms backlog) and its cut-ins are the timeout's.
- **hybrid 0.998 / 4000** is the head on 4 windows and the 4 s timeout on IB4002: 0 cut-ins, but +0.2 to +4.0 s more dead air than the timeout per window (Pipecat median 4522 vs 1702). Only TS3004b is not slower (head 0.9981 at +2.32 s = the timeout's own decision time).
- **hybrid 0.99 / 1000** beats the timeout's decision on ES2011b (+566 vs +1520 ms: dead air 1022 vs 1702 Pipecat, 995 vs 1697 LiveKit) and, in LiveKit, on TS3004b (+400) and IS1008b_014 (+800); Pipecat drops those two because other speakers talk within 240 ms. LiveKit's commit median falls 1687 → 1058 ms, its STT dead-air median 1688 → 1620, with the same 5 cut-ins; Pipecat's median is unchanged (1700 vs 1702). It also fires inside the primary's speech (ES2011b −1.12/−1.36 s, TS3004b −2.48 s), on the windows where the timeout already cuts in. With the 4000 ms backstop it is 0.5–3 s slower than the timeout on 4 of 5 windows.
- **head 0.98 alone** (the demos' `head` policy): decisions at +0.17 to +1.45 s on 4 windows plus pre-end firings on ES2011b (−1.36 s) and TS3004b (−2.71 s). LiveKit turns these into a 1237 ms median with 1 cut-in and 1 early cut and one window without a post-end decision; Pipecat delivers a turn end on only 2 of 5 windows because its resume rule discards a decision followed by any speech and the head never re-fires within the pad.

**Verdict.** Serving trail6 with the diarizer input does not give a product-level dead-air or cut-in improvement over the 1000 ms timeout on these 5 windows. The head now reaches θ 0.998, so `hybrid` is no longer degenerate, but every 0.998 crossing lands 0.7–2.4 s after the timeout's decision, so `hybrid 0.998/1000` fires when the timeout fires, and the 4 s backstop trades the timeout's 3–5 cut-ins for +0.6 to +2.9 s median dead air. A lower θ (0.99, chosen in-sample) buys 0.4–1.1 s on 1–3 windows, mostly ones where Pipecat's own resume gate then swallows the decision, and adds in-speech firings. The §7 bench gain (−6.4 to −12.8 miss points at a 6 s horizon) is a miss-rate gain on ends the timeout never catches, at P50 = inf; on ends the timeout does catch within 1.5 s, the head is not earlier. Product-level improvement over `timeout` on these windows would need a head that crosses inside the first second of trailing silence, or an adapter rule that does not drop a head decision when another speaker continues.

### Pipecat, 1.04 s setting (pipecat-ai 1.12.0; 1×, 3 s silence after each WAV; load 3.6–6, server RTF 0.53–0.73; n = 5 windows × 2 policies)

| window (length, speakers) | timeout dead-air ms (server / transport / Pipecat) | cut-ins | head dead-air ms | cut-ins | WER |
|---|---|---|---|---|---|
| ES2011b_027 (20.0 s, 1) | 2503 (2240/247/16) | 1 | 4163 | 0 | 0.258 |
| IB4002_119 (16.6 s, 4) | 1983 (1760/211/12) | 1 | 2022 | 0 | 0.433 |
| IS1008b_003 (15.2 s, 2) | 1903 (1680/211/12) | 0 | 2463 | 0 | 0.283 |
| IS1008b_014 (20.0 s, 3) | 3943 (3680/243/20) | 0 | 3661 | 0 | 0.386 / 0.357 |
| TS3004b_064 (16.6 s, 2) | 2122 (1840/263/19) | 0 | 2803 | 0 | 0.582 |

- **Timeout:** mean 2491 ms, median 2122, 2 cut-ins. **Head:** mean 3022 ms, median 2803, 0 cut-ins. The head's 0 is misleading: 4 of its 5 firings fell in the appended silence, and without that padding it mostly does not fire.
- Pipecat's own share is 1–20 ms. All 10 runs ended cleanly with `EndFrame`, with 0 warnings and 0 ordering violations.
- Verifier (IS1008b_003, ES2011b_027): timeout 1862 / 2441 ms (1 cut-in on ES2011b); head 2461 / 4142 ms, both in the padding. This matches the table.

### LiveKit, 1.04 s setting (livekit-agents 1.8.3, rtc 1.1.18; room-less; 1×, 2.5 s silence; load 4.7–5.4, server RTF 0.79–0.85; n = 5 windows)

| window (true end) | 1st INTERIM after 1st word end | STT END: early cuts / decision / dead air | AgentSession commit after true end: `stt` / detector (early) | WER |
|---|---|---|---|---|
| IS1008b_003 (13.28 s) | 355 ms | 0 / 1680 / 2902 ms | 2413 / 2040 ms (0/0) | 28% |
| ES2011b_027 (18.0 s) | 93 ms | 1 / 2240 / 2536 ms | 2550 / 1944 ms (1/0) | 26% |
| TS3004b_064 (15.52 s) | 853 ms | 1 / 1840 / 3541 ms | 2304 / 2304 ms (1/0) | 58% |
| IB4002_119 (14.64 s) | 1360 ms | 0 / 1760 / 2207 ms | 2133 / 2616 ms (0/0) | 43% |
| IS1008b_014 (18.0 s) | −774 ms* | 2 / 3680 / 4087 ms | 4759 / 4444 ms (2/0) | 39% |

\* The ASR emitted "effort" before the forced alignment's end of that 1.55 s word.

- Dead air includes 0.3–1.7 s of delivery lag from the loaded machine. An earlier run at load about 9 hit RTF about 1.1.
- **Verifier** (IS1008b_003, ES2011b_027): decisions are identical (+1680/+2240); first interim 346/76 ms; STT dead air 1806/2448 ms and `stt` commit 1814/2592 ms at server RTF 0.38.
- The report's "detector mode: 0 early commits" **did not reproduce**: ES2011b committed early at 17.58 s (true end 18.0 s). It depends on timing (D3).

## 5. Defects found by the verifier and their status

The status is as of this writing: none of these is fixed in code. `_turn_end` still cuts at a raw token index.

| # | defect | impact | status |
|---|---|---|---|
| D1 | Default policy justified with STAGE1's 38% (oracle column choice, no "nobody else" condition). The shipped label-free rule corresponds to 66–68% misses, about the same as the head's 69% | The product default is not validated | **Fixed: default is now the plain label-free timeout (timeout_quiet keeps the old rule); score it at n=200.** Next: score the server's exact rule at n=200 |
| D2 | Finals can split a word (`'...every' \| 'one ...'`): `_turn_end` cuts at `tok_at[f-1]`, not at a word start | Wrong text into a downstream LLM | **Open**; fix in `serve.py` before an LLM is attached |
| D3 | LiveKit detector mode can commit early (ES2011b, 17.58 s) | "0 early" claim not robust | **Open.** Docs corrected here; use `stt` mode |
| D4 | `speakers` is `[0,0,0,0]` for the first ~1 s, the same as "nobody talking" | Client misreads the startup | **Open** (needs `null` or a "not yet known" field) |
| D5 | `speakers` is 480–960 ms stale; `spk_t` only with `--debug-fields` | Clients cannot align speakers to audio | **Open** (promote `spk_t` to the protocol) |
| D6 | `stats`: empty session `rtf` 65291; `peak_rss_mb` is the process lifetime peak; `first_partial_ms` is only server lag | Misleading telemetry | **Open** |
| D7 | Bad `turn_policy` silently ignored, `timeout_ms:-5` clamped to 80, logged only on the server; empty finals at end | Silent misconfiguration | **Open** (echo the effective config in `ready`, send an `error` message) |
| D8 | `diar_lag_ms_mean_measured` computed from sample counts (always 760); real lag p50 about 860 ms | Misleading debug stat | **Open** (rename or measure wall clock) |
| D9 | Overstated claims: as-fast-as-possible RTF 0.33/0.27 (really 0.42–0.53); "ASR frames within about 90 ms" (really p50 about 130); "2 real-model tests" (really 1) | Docs | **Corrected in this file.** The README still says "about 90 ms" |

Fixed during the agents' own testing:
- Pipecat: the VAD-quiet gate; `resume_ms` stale-decision drop; the volume gate staying off; the loguru handler error.
- LiveKit: the detector ignores a stale `turn_end` that speech followed; `speech_end_time` subtracts the diarizer lag.

Known framework quirks:
- Pipecat sends a frame after `EndFrame` if a turn is open at close (seen only with the fake server).
- LiveKit's `AgentSession` pushes 2 s of silence into the STT on close, which inflates backlog stats.
- LiveKit warns "stt end of speech received while vad is still in a speech segment" when others talk. This is expected.

Environment note: installing `pipecat-ai` downgraded onnxruntime 1.30.0→1.24.4, protobuf 7.36.2→6.33.6 and soundfile
0.14.0→0.13.1 in the shared venv. The existing tests still pass.

## 6. Product readiness

**What a developer gets today:**
- One WebSocket server with a fixed, validated JSON protocol, giving streaming English ASR partials and finals, per-80 ms VAD, 4-speaker activity and turn-end events. It runs on CPU at about 0.5 RTF on 2 threads, in about 3.2 GB of RAM.
- Drop-in adapters for Pipecat 1.12 (STT, VAD, turn analyzer) and LiveKit Agents 1.8 (STT, VAD, turn detector; `turn_detection="stt"`), each on one shared session.
- Reproducible demos on real meeting audio, and 73 unit tests plus real-model tests.

It is ready for a **live integration test** in a real LiveKit room or Pipecat transport. It is not ready to be the product's turn-taker.

**Known limits:**
- **ASR:** WER on meetings is about 25–58% per window against all speakers (typical about 26–28%); clean read speech is 0–4%.
- **Diarization:** DER 0.20 is the *offline* Sortformer figure. The low-latency streaming setting the server ships is **0.252** pooled on AMI dev windows (research/archive/SORTFORMER_IMPORT.md), and columns arrive about 0.86 s late.
- **Turn-taking is timeout-grade:**
  - dead air median 1.9 s (Pipecat) / 1.7 s (LiveKit) after a clean end with the 0.32 s default (floor about 1.2–1.4 s; 1.8–2.6 s at the older 1.04 s setting);
  - cuts on filled pauses;
  - accuracy of the shipped label-free rule unmeasured at n=200 (D1);
  - the learned head is not better and fires mostly in trailing silence.
- **Serving:** one worker thread shared by all sessions (concurrency not load-tested); keeping up at 1× depends on fast-conv and on the machine not being overloaded (load about 9 → RTF about 1.1); linear resampling.
- **Untested:** real microphones, echo and full duplex/barge-in, WebRTC/Opus, real LLM/TTS latency.

**Next 5 engineering items, in priority order:**
1. **Score the shipped rule at n=200** (label-free primary over 5 s, "primary silent & nobody else", 1000 ms) with `scripts/eval_stage1.py`. Sweep `timeout_ms`, and compare against the head and the head with `--turn-input diar`. Set the default and the docs from that result (D1).
2. **Fix word-split finals** in `_turn_end` by cutting at the last word-start piece (D2). Add a regression test.
3. **Protocol hygiene:** a `null`/unknown state for `speakers` before the first column (D4), `spk_t` in the normal protocol (D5), the effective config echoed in `ready` plus an `error` message for rejected config (D7), and correct `stats` (D6, D8).
4. **Real-room test:** `livekit-server --dev` plus a worker plus a real microphone and the Agents Playground, then a Pipecat WebRTC transport. Measure echo and barge-in with TTS playing, and the Opus/48 kHz path.
5. **Cut dead air and harden serving:** move fast-conv into `fastconformer.py`; try a shorter diarizer right context or ASR-VAD-gated early decisions to get under 1 s after the true end; load-test N concurrent sessions and add a worker pool.

## 7. Primary-speaker enrollment: `--enroll after_agent | explicit` (2026-09-26)

Added after research/archive/EOT_BENCH_V2.md §9. Off by default (`--enroll dominant` = the 5 s dominant column, protocol
unchanged). The default's numbers in §4 do not change.

**Why.** The label-free binding loses ~33-37 points of turn-end misses to the oracle binding, and §8/§9 show the
loss is *which* speaker is the user, not how the column is followed. A voice agent has one identity signal the
benchmark's rules lack: it knows when its own TTS ended. `after_agent` binds the primary to the first speaker after
that moment and then follows that voice with TitaNet-L across diarizer column swaps.

**Server.** `python -m audioforge.serve ... --enroll after_agent [--titanet data/nemo/speakerverification_en_titanet_large.nemo] [--enroll-stride 5]`
- Client -> server: `{"type": "agent_end"}` (after_agent) / `{"type": "enroll"}` (explicit). The message is applied at
  the audio position received so far: from that diarizer frame on, the first column active (p > 0.5) for >= 3
  consecutive frames is the user; its first 19 active frames (1.5 s) are the enrollment; then every 5 frames each
  column with >= 8 active frames in the last 2 s is embedded over that speech and the follower's hysteresis
  (margin 0.1 cosine, hold 6 frames = 2 updates) may move the binding (`enrollment.VoiceFollower`, the same code as
  the benchmark). Each `agent_end` re-arms (one enrollment per user turn). The other mode's trigger is logged and
  ignored. Until a column is chosen the primary is the dominant column.
- Server -> client: `frame.primary` / `final.speaker` are the bound column like any primary; a new
  `{"type": "enrolled", "t", "column"}` message marks the enrollment (t = the decision time of the 19th frame);
  `ready` and `stats` carry `enroll` (mode), `enrolled`, `primary_column` (all three or none; `validate` enforces it).
  Debug fields: `ready.enroll_rss_mb`, `ready.enroll_stride`, `stats.enroll_ms_p50 / _mean / _max / enroll_n_embed`.
- Cost (this Mac, 2 threads, `--diar-config low_latency_032`, 3 AMI dev turn windows, load 3-6): TitaNet-L
  adds +91 (psutil, on top of the loaded ASR + diarizer; +201 in a fresh process) MB RSS; the binder costs 1.3-13 ms per diarizer frame on average (1-21 embeddings per
  window: one TitaNet pass of 150-300 ms per active column every 400 ms once enrolled), RTF 0.87-1.16 (chunk p50 103-140 ms, p95 264-554 ms) vs
  0.60-0.84 (p50 85-96 ms, p95 187-251 ms): at 1x on this Mac the enrollment mode is at the edge of keeping up on 2 threads; `--enroll-stride 10` or a third thread is the remedy without it. `--enroll-stride 10` halves the following cost at the price of a slower switch.

**Pipecat** (`audioforge/integrations/pipecat.py`): `AudioforgeSTTService(enroll="after_agent")` sends `agent_end`
whenever the pipeline's output transport reports `BotStoppedSpeakingFrame` (Pipecat pushes it upstream through every
processor, so the STT service receives it in `process_frame`); `enroll="explicit"` sends `enroll` on
`await stt.enroll()`; the `enrolled` event sets `hub.enrolled_column`. A `ready.enroll` that differs from the
service's mode is logged as a warning. **LiveKit** (`audioforge/integrations/livekit.py`): not wired; the hook is the
agent session's `agent_state_changed` ("speaking" -> "listening") or the TTS node's completion, sending the same
JSON text frame.

**What the benchmark says about it** (EOT_BENCH_V2.md §9, AMI dev, the agent turn = the previous other speaker's
turn, its end from the labels as a stand-in for the TTS end): the agent-end binding is the first label-free rule that beats the shipped dominant rule's benchmark analogue (causal_dominant) with the CI excluding 0 at the 6 s horizon: hybrid 51.6 vs 61.9 % misses (−10.3 [−13.5, −6.9]; floor-open −11.9 [−18.1, −5.8]) with TitaNet following, 56.0 (−5.9 [−8.6, −3.5]) with the causal rule after the choice and no embedding at all. At the 2 s horizon the TitaNet-followed row is *worse* than causal_dominant (+8.5 [+6.2, +10.9]) while the causal-followed one is still better (−3.9 [−6.6, −1.4]): the voice follower catches the same turns 2-6 s late. So: arm at the agent end in any case; enable the TitaNet following only where the 6 s miss rate matters more than dead air. An arming-only server mode (choice, then the dominant rule) is the next item.

Tests: `tests/test_serve.py` (`VoiceBinder` selection, enrollment and swap following on a scripted two-speaker
scene; a Session with `agent_end` emitting `enrolled`; socket `agent_end` message; the default protocol unchanged),
`tests/test_pipecat_integration.py` (`BotStoppedSpeakingFrame` -> `agent_end`, `enroll()`),
`tests/test_enrollment.py` (backend causality, grid semantics, after_prev_end selection and fallbacks).

## 8. Shipped rules (2026-09-26)

Question: do today's confirmed rules make the served product respond earlier than the 1000 ms timeout, or cut in less, and by how much? The rules are research/archive/BASELINES.md "ICSI held-out confirmation" and "Dynamic timeout: ICSI held-out confirmation", and the arming rule is research/archive/EOT_BENCH_V2.md §9.

**What was shipped (code, tests):**
- `runs/stage1_served.afm` = `stage1_turn_v3_trail6` with `heads.spk` transplanted from `stage1_spk_relational` (`scripts/research/make_served_model.py`; the other 747 tensors are bit-identical; LibriSpeech-100 WER 2.26 % unchanged).
- `serve.py` turn policies:
  - `hybrid_dyn` (primary candidate): head p ≥ 0.998283 OR any-speaker Silero v5 silence ≥ clamp(80 − 55 p, 7, 80) frames.
  - `hybrid_silero` (fallback): head p ≥ 0.99828 OR silence ≥ 2.64 s (`--silero-timeout-ms`).
  - The Silero silence path is Silero v5 → Pipecat VAD state machine (conf 0.7, start/stop 0.2 s, 32 ms chunks); re-armed by speech; one `turn_end` per turn.
- `serve.py --enroll after_agent_arm`: on `agent_end`, the first column with ≥ 3 consecutive active frames is bound, then followed by causal_dominant. No TitaNet.
- Adapter hooks: Pipecat `enroll="after_agent_arm"` sends `agent_end` on `BotStoppedSpeakingFrame`; LiveKit `AudioforgeFrontend.agent_end()` / `attach(session)`.
- Silero costs 0.11 ms per 32 ms chunk in the server (RTF +0.003) and about +24 MB RSS. It is loaded only when a session uses these policies.
- Tests: `tests/test_serve_shipped.py`, plus additions in `tests/test_pipecat_integration.py` and `tests/test_livekit_integration.py`.

**Setup.** Same 5 AMI windows, labels and definitions as §4.
- Server: `--asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm --turn-input diar --threads 2 --debug-fields`, 0.32 s diarizer. One server per plan (at most two policy lines of 5 windows), run from a clean git worktree of commit `4e90fd5`. Sessions started at 1-min load 1.7–5.7.
- Pads: 7.0 s for all six policies in both demos, because the dyn wait reaches 6.4 s at p = 0.
- **`agent_end` stand-in:** it is injected at the previous other speaker's turn end from the labels (`enrollment.agent_end_frame`). This stands in for a TTS-end event; nothing else uses labels. Values: IB4002_119 4.32 s, IS1008b_003 3.12 s, TS3004b_064 2.08 s. ES2011b_027 and IS1008b_014 have no earlier other-speaker turn in the window, so `agent_end` is sent at 0.0 s, before the first audio.
- Pipecat: the WAV transport pushes a `BotStoppedSpeakingFrame` after the 20 ms frame that reaches that time; the frame is dropped after the STT. LiveKit: `fe.agent_end()` in both stream and AgentSession modes.
- Data: `runs/integration_deadair_shipped.json`.

**Pipecat** (dead air ms / cut-ins, decision ms = `turn_end.t` − true end in parentheses; \* = load-inflated delivery, see below):

| window | timeout 1000 | hybrid_dyn | hybrid_dyn + arm | hybrid_silero | hybrid_silero + arm | timeout 1000 + arm |
|---|---|---|---|---|---|---|
| ES2011b_027 | 1701 / 0 (1520) | 2662 / 0 (2480) | 2640 / 0 (2480) | 2941 / 0 (2896) | 2941 / 0 (2896) | 1681 / 0 (1520) |
| IB4002_119 | 1702 / 1 (1520) | 2421 / 0 (2240) | 2421 / 0 (2240) | 2941 / 0 (2736) | 2921 / 0 (2736) | 1682 / 1 (1520) |
| IS1008b_003 | 1602 / 0 (1440) | 2341 / 0 (2160) | 2341 / 0 (2160) | 2861 / 0 (2720) | 2840 / 0 (2720) | 1602 / 0 (1440) |
| IS1008b_014 | 3402 / 1 (3200) | 3682\* / 0 (2240) | 2563 / 0 (2240) | 2562 / 0 (2240) | 2562 / 0 (2240) | 3401 / 1 (3200) |
| TS3004b_064 | 2501 / 1 (2320) | 2981 / 0 (2800) | 3201 / 0 (3040) | 3001 / 0 (2800) | 3201 / 0 (3040) | 2501 / 1 (2320) |
| **median / total** | **1702 / 3** (1520) | **2662 / 0** (2240) | **2563 / 0** (2240) | **2941 / 0** (2736) | **2921 / 0** (2736) | **1682 / 3** (1520) |

**LiveKit** (STT dead air ms / cut-ins / AgentSession commit ms, decision in parentheses):

| window | timeout 1000 | hybrid_dyn | hybrid_dyn + arm | hybrid_silero | hybrid_silero + arm | timeout 1000 + arm |
|---|---|---|---|---|---|---|
| ES2011b_027 | 1676 / 1 / 1683 (1520) | 2636 / 0 / 2641 (2480) | 2636 / 0 / 2637 (2480) | 2925 / 0 / 2926 (2896) | 2924 / 0 / 2925 (2896) | 1681 / 1 / 1675 (1520) |
| IB4002_119 | 1707 / 0 / 1671 (1520) | 2414 / 0 / 2414 (2240) | 2412 / 0 / 2413 (2240) | 2921 / 0 / 2920 (2736) | 2918 / 0 / 2915 (2736) | 1667 / 0 / 1671 (1520) |
| IS1008b_003 | 1581 / 1 / 1582 (1440) | 2333 / 0 / 2332 (2160) | 2295 / 0 / 2330 (2160) | 2835 / 0 / 2836 (2720) | 2835 / 0 / 2835 (2720) | 1582 / 1 / 1595 (1440) |
| IS1008b_014 | 1459 / 1 / 1462 (1280) | 2425 / 0 / 2422 (2240) | 2419 / 0 / 2431 (2240) | 2453 / 0 / 2429 (2240) | 2417 / 0 / 2423 (2240) | 1457 / 1 / 1458 (1280) |
| TS3004b_064 | 2496 / 2 / 2497 (2320) | 2988 / 0 / 2978 (2800) | 3192 / 0 / 3193 (3040) | 2975 / 0 / 2975 (2800) | 3194 / 0 / 3193 (3040) | 2493 / 1 / 2493 (2320) |
| **median / total** | **1676 / 5 / 1671** (1520) | **2425 / 0 / 2422** (2240) | **2419 / 0 / 2431** (2240) | **2921 / 0 / 2920** (2736) | **2918 / 0 / 2915** (2736) | **1667 / 4 / 1671** (1520) |

**Paired deltas, per window** (the decision does not depend on load; dead air is paired against the §4 / §4b timeout rows):

| policy | decision − control decision, ms (LiveKit, 5 windows; median; Pipecat identical except IS1008b_014: −960) | Pipecat dead air − §4 timeout, median (vs §4b) | LiveKit dead air − §4 timeout, median (vs §4b) |
|---|---|---|---|
| timeout 1000 (control, this run) | 0 on all | −1 (−1) | +3 (−22) |
| hybrid_dyn | +960, +720, +720, +960, +480; **+720** | **+660** (+720) | **+755** (+729) |
| hybrid_dyn + arm | +960, +720, +720, +960, +720; **+720** | +699 (+720) | +719 (+724) |
| hybrid_silero | +1376, +1216, +1280, +960, +480; **+1216** | +1018 (+1239) | +1215 (+1228) |
| hybrid_silero + arm | +1376, +1216, +1280, +960, +720; **+1216** | +1018 (+1220) | +1212 (+1227) |
| timeout 1000 + arm | 0 on all | −1 (−21) | +4 (−22) |

On IS1008b_014 the timeout's first post-end decision (+1.28 s) is a cut-in: the primary speaks again within 1 s. LiveKit counts that decision as dead air (1459). Pipecat drops it and waits for the next one (+3.20 s), so that window's hybrid decision is +0.96 s later in LiveKit and −0.96 s earlier in Pipecat.

- **The control reproduces §4 / §4b.** Timeout decisions are identical on every window (+1520 / +1520 / +1440 / +3200 (Pipecat) or +1280 (LK) / +2320). Dead air is within delivery noise, except §4's ES2011b Pipecat run (1923 vs 1701). Cut-ins are 3 (Pipecat) and 5 (LiveKit), the same as §4.
- **Where the hybrids fire.** Every hybrid run has exactly one decision per window, and all are after the true end.
  - On 3 windows it is the Silero branch: the dyn wait with p ≈ 0.992–0.998 is 2.03–2.11 s of silence; the fixed branch waits 2.67–2.69 s.
  - On IS1008b_014 and TS3004b_064 it is the head path. The head crosses 0.99828 at +2.24 s / +2.80 s, where both hybrids agree.
  - The head never crosses inside the first second of trailing silence (§4b's finding stands), so the hybrids cannot beat a 1.0 s timeout on these windows. The dyn branch at p ≈ 0.99 is itself a 2.0 s timeout on any-speaker silence.
- **Cut-ins.** The hybrids have none: 0 of 5 windows in both demos, against the timeout's 3 (Pipecat) and 5 (LiveKit). The timeout's cut-ins are firings on the primary's 1–1.6 s pauses: IB4002 +2.96 s, IS1008b_014 +7.76 s and +19.28 s, TS3004b +3.44 s and +14.0 s. The any-speaker 2–2.7 s silence and θ 0.998 wait them out. This matches the rules' purpose (ICSI FC 1.8 / 2.4 %), not their dead air.
- **Arming (`after_agent_arm`).**
  - It bound column 0 / 0 / 1 / 2 / 1. Timeout decisions are unchanged on all windows.
  - With the timeout, it removes one early timeout firing on TS3004b (+3.44 s, before the agent's turn had ended in the labels), so LiveKit cut-ins go 5 → 4.
  - With the hybrids, arming makes TS3004b 240 ms later (+3040 vs +2800). The head is conditioned on the bound column (1), whose posterior crosses one chunk later.
  - Elsewhere it is neutral. On 5 windows arming neither helps nor hurts dead air measurably. The §9 bench gain (−5.9 miss points at 6 s) is about turns the timeout misses, which these 5 windows barely contain.
- **Server and clean-run checks.**
  - Server RTF 0.68–0.79 and peak RSS 3.55–3.58 GB for every run. Backlog ≤ 760 ms in the stream sessions. AgentSession sessions show 1.8–2.0 s, which is LiveKit pushing 2 s of silence on close (§5 quirk).
  - **No session exceeded RTF 1.** The one flagged run is Pipecat hybrid_dyn IS1008b_014 (RTF 0.99, 1.74 s backlog): its 3682 ms dead air is delivery-inflated. Its decision (+2240) equals the arm rerun's (2563 ms dead air).
  - WER is identical across all 60 runs (0.258 / 0.433 / 0.283 / 0.357 / 0.582).
  - All 30 Pipecat runs ended with `EndFrame`, with 0 violations, 0 warnings and 0 pipeline errors.
  - LiveKit: every run had 1 server session and every END_OF_SPEECH followed a FINAL. In the four hybrid runs, IS1008b_014 has one more END_OF_SPEECH than cutting `turn_end`s: the end-of-stream flush final, as in §4b.

**Verdict.** No rule wins on both axes, so the README default stays `timeout` 1000.
- The shipped rules make the product respond *later*, not earlier, than the 1000 ms timeout on these windows:
  - `hybrid_dyn`: median dead air +0.66 s (Pipecat 2662 vs 1702) and +0.75 s (LiveKit 2425 vs 1676).
  - `hybrid_silero`: +1.2 s (2941 / 2921).
- In exchange they cut in less: **0 cut-ins vs 3 (Pipecat) / 5 (LiveKit)**.
- `after_agent_arm` changes nothing measurable at this n: timeout identical with one fewer LiveKit cut-in, hybrids +240 ms on one window.
- If the product's priority is not interrupting the user, `hybrid_dyn` is the better of the two new rules. It costs about 0.7 s of dead air and has no cut-ins on 5 of 5 windows; it is the one to offer opt-in (`turn_policy: "hybrid_dyn"`).
- If the priority is responsiveness, the timeout remains.
- Caveats: n = 5 windows, one per turn; the benchmark gains these rules were chosen for are miss-rate gains on ends the timeout never catches, which a 5-window dead-air test cannot show; the `agent_end` time is a label-derived stand-in.
