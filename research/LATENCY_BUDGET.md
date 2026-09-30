# Latency budget of the live path (single mode), excluding the turn rule's own wait

2026-09-30. Complaint: about 1.3-1.4 s from "the user stops talking" to `turn_end`. The turn rule's silence wait (the
960 ms Silero floor of `hybrid_dyn`, `research/EOT_LATENCY.md`) is owned elsewhere and is not changed here. This note
covers every other delay between a microphone sample and the client receiving `turn_end`, partials and the final.

How it was measured. `scripts/research/latency_budget.py` (numbers in `runs/latency_budget.json`):
- `ws` starts `audioforge-serve --mode single` as a child process (2 threads, `stage1_served_v2.afm`, heads from `assets/`, Silero). It streams the bundled clip `examples/audio/two_party_call_16s.wav` with its stored voice print at 1x, in 20 ms blocks, over a real WebSocket, 3-4 repetitions. Each message is timed against the moment its audio was sent.
- "before" is the commit before this change (370e936, run from a git worktree). "after" is this change.
- `pipecat` runs the same server through `examples/pipecat_local_demo.run_pipeline` (Pipecat 1.12, policy `hybrid_dyn`, a mock LLM).
- `compute` runs the same session in process, with a timer on each part of a chunk.
- `wer` is LibriSpeech-200 through the streaming session.
- Another agent's 2-thread job ran on the same laptop for most runs, so live compute numbers vary by about ±10 ms.

The clip has one labelled user turn end, at 10.8 s. `turn_end` comes with `silence_ms` 1360 and head p 0.943, the same
before and after.

## Budget: user's turn end (10.8 s) to the client holding `turn_end` + `final`

| stage | ms before | ms after | how measured |
|---|---:|---:|---|
| Turn rule: Silero silence wait, from the last Silero speech chunk to the end of the deciding frame (not in scope) | 1360 | 1360 | `turn_end.silence_ms` (same frame, same p) |
| minus the offset between the label and Silero: the last speech chunk ends 80 ms before the labelled 10.8 s | -80 | -80 | deciding frame end 12.08 s - 1.36 s = 10.72 s |
| **ASR chunk trigger**: the encoder chunk waited for a whole 16-mel window, but its frames need only 9 mels | **70** | **0** | `turn_end.t` 12.166 → 12.096 s, identical in all runs; code: see "Fix 1" |
| Encoder lookahead, att_context [70,1]: frame 2j waits for 2j+1. Here the deciding frame is even (+80). With the STFT half window (+16) this is the head posterior's ready time past the frame end | 16 | 16 | `Session._asr_ready_t`; the Silero chunk that holds the frame end (32 ms grid) is also ready at +16 here |
| = decision audio time after the labelled end (`turn_end.t` - 10.8 s) | **1366** | **1296** | `ws`, 4/4 runs identical |
| Client block: the decision sample waits for its 20 ms block to be sent | 14 | 4 | pacing (mean 10, max 20 for any sample) |
| Server compute of that 160 ms chunk (2 threads). Encoder 17.5 ms + speaker-conditioned turn pass 12.6 + RNNT, VAD, TS-VAD, LID, turn steps, Silero 1.7 | ~31 | ~31 | `compute` in process: chunk-block p50 31 ms (p95 34) |
| Socket, JSON, event loop, plus load from the neighbour job | ~22 | ~6 | `ws` deliver_ms (receive - send of the decision block) p50 54 → 41; per frame p50 49 → 36 |
| = **client receives `turn_end`, after the user stopped** | **1434** | **1341** | `ws` recv_after_end_ms p50 (n = 4 each) |
| `final` after its `turn_end` | 0 | 0 | same message batch (`final_after_turn_end_ms` 0.0); the text holds every word, since the RNNT has emitted all of them long before the decision |
| Pipecat adapter: the turn analyzer can say COMPLETE only in `append_audio`, on the next 20 ms input frame | 7 | 12 | `pipecat` adapter_ms p50 (max 20; noise between runs) |
| = **Pipecat bot would start answering (mock LLM context), after the user stopped** | **1441** | **1361** | `pipecat` response_after_end_ms p50 (n = 3 each) |
| Pipecat response after `turn_end.t` (the 76 ms `delivery_offset_s` of runs/single_model.json) | 75 | 65 | = block + compute + socket + adapter |

Other delays on the same path, measured or checked in code:

| stage | ms before | ms after | how measured |
|---|---:|---:|---|
| `frame` messages: receive - the frame's label time (v+1)·80 ms (VAD, speakers, eot to the client) | 119 (p50) | 53 (p50) | `ws` frame_after_label_ms |
| `partial` receive - the audio time its chunk's input was complete (compute + socket) | 52 (p50) | 35 (p50) | `ws`. Word-level partial latency (441 ms p50 in METRICS.md) is mostly RNNT emission lag, 5 frames = 400 ms median (research/TSWER.md). That lag is model-intrinsic and not changed. The chunk-trigger fix takes 70 ms off every word. `scripts/research/stt_latency.py` feeds 160 ms blocks, and those blocks hid a further ~150 ms before the fix (chunk j was ready at 160j+166 ms = after block j+1, where it is now ready inside block j), so re-run it for the new number |
| Pass 2 (speaker-conditioned turn head) lag behind pass 1 | 0 frames | 0 frames | `run_turn_on_diar(avail = asr.n_frames)` under `--turn-input tsvad` runs in the same `process()` call. The TS-VAD columns of the chunk exist before the turn pass. A voice print applies from the next frame on, which costs nothing |
| Silero 512-sample sub-chunking vs the 80 ms frame end | 0 or 16 (mean 8) | 0 or 16 | `SileroSilence._cj`. This is part of the turn rule's input; left as is |
| Executor | 1 thread | 1 thread | one session sees no queueing; N sessions queue behind each other's ~31 ms chunks (METRICS.md "CPU streams") |
| Hot-path logging, validation, per-message JSON | 0 | 0 | `validate` only with `--debug-fields`; `engine.event` only on errors; ~5 small `json.dumps` per 160 ms |
| Server read loop | ≤ 0 | ≤ 0 | each 20 ms block is processed as soon as it arrives (`n_in >= 320`); no sleeps or polls on the path (only the 300 s idle timer) |
| LiveKit adapter | 0 | 0 | END_OF_SPEECH is emitted when the turn's `final` arrives (same batch as `turn_end`). The documented `endpointing.min_delay: 0.0` matters: LiveKit's default min_delay (0.5 s) would add 500 ms |
| Pipecat VAD gate (`wait_for_silence`) | 0 | 0 | Pipecat's VAD (stop_secs 0.2) stopped ~1 s before the decision; it only blocks when someone speaks |

## Fix 1: the chunk trigger (every frame 70 ms earlier, same outputs)

The served encoder uses NeMo's causal subsampling (`pre_encode.nemo_causal`), so encoder frame v ends at mel frame 8v.
The attention chunk of frames [2j, 2j+1] is therefore complete once mel frame 16j+8 exists. `ASRStream.feed_frames`
(and `StreamingSession.feed`) waited for a whole 16-mel window, up to mel 16j+15. `_stream_step_aligned` already
carried the extra 7 mels over, and its docstring says so: "the first call yields R+1 frames from 8R+1 mels". The
trigger ignored that and kept every frame, VAD value, token, TS-VAD column and turn-head posterior back by 7 mel
frames = 70 ms.

Now the first chunk runs after `chunk_lead = 8R+1` mel frames, then every `(R+1)·8`. `frame_ready_samples(v)` and
`frames_ready(samples)` replace the old formulas in `Session._asr_ready_t` / `_asr_frames_at` and
`LookaheadStream.ready_t`, so `turn_end.t` and the final's cut follow the new clock. Encoders without NeMo
alignment keep the whole window.

Tests: `tests/test_latency_budget.py`.
- Each frame appears in the first 20 ms block that completes its input.
- Heads, tokens and TS-VAD columns match the old trigger (atol 1e-5).
- Every head decision time is the same or earlier.
- The text equals `StreamingSession.feed` in one go.

On LibriSpeech-200 the streaming WER at [70,1] is 2.29 %, exactly the offline masked number in HYBRID_ASR.md.

For the turn rule's owner: decision times of head-bound firings move 70 ms earlier. When the deciding frame is odd,
the Silero chunk usually binds instead. Fired frames and p values do not change. Two things in `eot_latency.py`
assume the old clock: its simulator (`asr_clock`, `chunk_mel`) and its `STRUCT_OURS = 240 ms`. `turn_end.t` already
includes the buffering, which is now (8R+1-1)·10 + 16 ms past the chunk start, not 16 mels.

## Fix 2 (option, not the default): `--asr-chunk-ms 80` = att_context [70,0]

`audioforge-serve --asr-chunk-ms 80` (Engine `asr_chunk_ms`) runs the one ASR pass at [70,0]: 80 ms chunks, no
lookahead. The NVIDIA model was trained for {[70,0], [70,1], [70,16], [70,33]}. The pass covers the transcript and
the VAD, turn and TS-VAD heads. The TS-VAD column grid follows it.

| setting | chunk / lookahead | frame ready after its label end (even / odd frame) | LibriSpeech-200 WER, streaming (this run) | AMI-200 / ICSI-200 WER (HYBRID_ASR.md §3, masked offline) | ASR-only compute (enc + RNNT) | full single-mode session, 2 threads | clip `turn_end` (ws / in process) |
|---|---|---|---:|---:|---|---|---|
| **[70,1] (default)** | 160 ms / 80 ms on even frames | +16 / -64 ms | **2.29** | 24.43 / 27.29 | 17.7 ms per 160 ms chunk (RTF 0.12) | 31 ms per chunk, RTF 0.20 | 12.096 s (p 0.943) / 12.096 s |
| [70,0] | 80 ms / 0 | -64 / -64 ms | 2.48 (**+0.19**, 95 % CI [+0.02, +0.39]) | 25.68 / 29.87 (**+1.25 / +2.58**, CIs exclude 0) | 13.0 ms per 80 ms chunk (RTF 0.17) | 22 ms per chunk, RTF 0.29 | 12.256 s (p 0.739) / 12.160 s |

Reading:
- **What [70,0] buys, now that Fix 1 is in:** frames and partials of even frames arrive 80 ms sooner (40 ms on
  average). `turn_end` under `hybrid_dyn` gains at most 16 ms, because the Silero chunk at the frame end then binds.
  WER on LibriSpeech is within the 0.3-point bar.
- **Why the default stays [70,1]:**
  - Meeting WER rises by 1.3 to 2.6 points.
  - The heads were trained on [70,1] features. On the clip, the turn head's posterior at the deciding frame drops
    from 0.94 to 0.74, so `turn_end` comes 64-160 ms *later*.
  - Compute stays real time (RTF 0.29 in process, 0.39 live with the neighbour job), so compute is not the reason.
- A switch would need the heads (VAD, turn, TS-VAD) fine-tuned at [70,0] first.

## What is left, and where

- The turn rule (1360 ms silence on this clip) is almost all of the remaining time. It is outside this note.
- Compute, ~31 ms per chunk: 40 % of it is the second, speaker-conditioned encoder pass for the turn head. It
  cannot move off the path, because `hybrid_dyn` needs the head's p on the deciding frame. A GPU or a smaller turn
  pass would shrink it.
- The Pipecat adapter's wait for the next 20 ms input frame (mean ~10 ms) comes from Pipecat's analyzer API.
  `TurnAnalyzerUserTurnStopStrategy` sets "turn complete" only from `append_audio` or a VAD stop.
- Encoder lookahead: 80 ms on even frames. Removing it needs heads retrained at [70,0] (above).
