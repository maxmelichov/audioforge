# Bulletproofing the served front end

What a real deployment throws at `python -m audioforge.serve`, the stream client and the two adapters, and what
happens now. Every row: what could break, how it was reproduced, the fix, the test that pins it. All tests are in
`tests/test_bulletproof.py` (tiny random models, 48 tests as collected by pytest, ~1 min); the adapter rows also touch
`tests/test_pipecat_integration.py` / `tests/test_livekit_integration.py`; two shipped-behaviour tests moved in
`tests/test_serve_shipped.py`. **No default policy's measured behaviour changed**: every fix is either a refusal, a
report, a bound that a normal session never reaches, an opt-in flag, or a degradation that only engages when the
compute cannot keep up (section 4).

## 0. The one new protocol element

`{"type": "error", "code", "detail", "fatal"}` (`audioforge.serve.error_msg`, codes in `ERROR_CODES`). `fatal:
true` is followed by the close (1008 for the client's fault / limits, 1011 for a server failure, 1001 on shutdown,
1009 by the library for an oversized frame); `fatal: false` reports a refused message or a degradation and the
session goes on. A degradation is reported once per code per session, counted per session in `stats.degraded`
and server-wide in `GET /health` `counters`. Under load level 2 the per-frame `frame` messages of a block arrive
as one `{"type": "frames", "items": [...]}`. Old clients that ignore unknown types keep working; the three shipped
clients (`scripts/stream_client.py`, `audioforge/integrations/pipecat.py`, `audioforge/integrations/livekit.py`)
and the `audioforge-client` package handle both.

`GET /health` (also `/healthz`, `/stats`) on the WebSocket port answers plain HTTP with liveness, `sessions_active`
/ `sessions_total`, `backlog_ms_total` / `backlog_ms_max`, `load` (sum of the sessions' recent RTF), `shed_level`,
RSS, the final-ASR worker's state, Silero's state, the limits and every counter. `--log-json` makes every log line
one JSON object (`ts`, `event`, `peer`, fields).

## 1. Audio

| What could break | Reproduced by | Fix | Test |
|---|---|---|---|
| NaN / Inf samples poison the encoder caches for the rest of the session (every later frame NaN) | 3 non-finite samples in a 2 s stream; every later `vad` was NaN | `Session._sanitize`: non-finite -> 0, counted, reported once as `nan_input`; the float32 wire path repairs in `decode_pcm` and counts too. The stream after the bad block equals a clean stream fed the repaired audio (bitwise) | `test_nan_inf_input_is_repaired_reported_and_does_not_poison_the_stream` |
| Absurd amplitudes (1e30, float32 in int16 range) overflow the power spectrum -> NaN | `x * 1e30`, `x * 32768` | clipped to +-64 (`MAX_ABS_SAMPLE`), reported once as `clipped_input`; DC offset, -50 dBFS and hard-clipped audio pass untouched with no error | `test_extreme_amplitudes_stay_finite_and_bounded[huge,dc,quiet,clipped,int16_range_floats]` |
| Zero-length blocks, 1-sample blocks, a stream with no audio at all | fed each | `process([])` returns `[]`; 1-sample blocks with empty blocks in between give the same decisions as 20 ms blocks; `finish()` without audio returns an empty final + stats with `rtf 0.0`; a second `finish()` is a no-op | `test_zero_length_frames_one_sample_blocks_and_empty_streams` |
| Decisions depend on how the client blocks the audio (1 sample, 7 ms, 3 s) | compared block sizes | none needed: frames, turn_ends and finals are bitwise identical across block sizes on the scripted and the real streaming diarizer (frames carry the *latest* diarizer row at emission, so a 3 s block reports its frames with its last row: compared by audio time) | `test_odd_chunk_sizes_give_identical_decisions[1,112,48000]`, `test_same_audio_same_events_across_runs_bitwise` |
| Minutes of silence or stationary noise: the transducer keeps emitting tokens (hallucination), the segment text and the partial grow without a turn_end | a transducer that emits one token per frame on 120 s of silence / noise: 1501 characters of final text | `--asr-vad-gate P [--asr-vad-hangover-ms 1200]` (opt-in, off by default so measured WER is unchanged): no transducer step on frames whose served VAD <= P once the hangover has passed; the encoder and heads still run every frame. With speech the gate never engages (identical text) | `test_silence_and_noise_minutes_bounded_text_and_no_turns` |
| A segment no policy ever cuts (head never fires, timeout disabled) grows for the whole session | head policy with p ~ 0 on a talky model | `MAX_SEGMENT_S` = 300 s: the segment is cut with a final (no turn_end), reported once as `segment_cap`; nothing lost or duplicated | `test_segment_cap_cuts_an_endless_segment_with_a_final` |
| Wrong sample rate / channel count / sample format: 8 kHz phone audio, 48 kHz WebRTC, float32, stereo, odd frame sizes | streamed each | `config.sample_rate` 8000..192000 (1 Hz was a 16000x amplifier; refused), `config.format` int16 / float32, `config.channels` 1..8 (interleaved, averaged to mono), all before the first audio; `decode_pcm` carries partial sample frames across WebSocket frames. All paths give the audio's frame count; float32 stereo equals int16 mono within quantization | `test_resampling_paths_8k_and_48k_and_int16_float32_stereo_payloads`, `test_decode_pcm_formats_and_carry` |
| 1-hour sessions: per-frame lists (`rows`, `prims`, timing samples), the diarizer's `probs` accumulator and the lookahead buffers grow without bound; the stats percentiles get slower | 30 min through one session, RSS and per-block cost sampled | `_Ring` (last 2048 frames, absolute indexing) for rows / primaries / VAD; `deque(maxlen=STAT_KEEP)` for every timing series with running maxima; `StreamingDiarizer(keep_probs=False)` in the server; `--max-session-s` (default 4 h) flushes (final + stats) and closes with `session_limit`. 30 min: state bounded, RSS flat after warm-up, last-5-min block cost <= first-5-min. (The first soak showed a 5x block-cost growth; a profile put 2.0 of 2.9 s in the *test double* `EnergyDiarizer.feed`, which concatenates every sample ever fed; the soak now uses a bounded copy, `BoundedEnergyDiarizer`, pinned equal to the shared one.) | `test_long_session_flat_memory_bounded_state_no_slowdown`, `test_bounded_energy_diarizer_equals_the_shared_double`, `test_server_diarizer_keeps_no_per_frame_probabilities`, `test_idle_timeout_and_session_limit_close_with_errors` |
| Back-to-back speakers and a Sortformer column swap (speaker B continues on A's column) | scripted diarizer A -> B -> B-on-column-0 | none needed: every burst ends in a turn_end + final, the primary follows the active column, the primary rule re-binds once the new column dominates its window | `test_back_to_back_speakers_and_column_swap` |
| Coughs / laughs (60 ms bursts every 2 s) fire turns or crash a policy | 6 bursts on timeout / head / both / hybrid | none needed: at most one turn_end per burst (two on `both`), every turn_end has its final, outputs finite; a head that never says end-of-turn fires nothing | `test_bursts_fire_at_most_one_turn_each_and_never_crash` |

## 2. Protocol and connection handling

| What could break | Reproduced by | Fix | Test |
|---|---|---|---|
| Malformed text frames: non-JSON, JSON that is not an object, no / non-string `type`, unknown type, control bytes; bad config values (`timeout_ms: "abc"`, `eot_threshold: null`, `sample_rate: 1`, `channels: 99`, `turn_policy: "x"`); a 64 KB+ text frame | sent all eleven in a row, then audio | every one answers a non-fatal `error` (`bad_json`, `bad_message`, `unknown_type`, `bad_config` listing every bad field, `message_too_large`, `unsupported`); `SessionConfig.update` never raises (ignore or clamp + report); the session then processes the audio normally; nothing a client sends can raise out of `handle` | `test_malformed_and_unknown_messages_get_structured_errors_and_the_session_lives` |
| A config after the first audio changes `sample_rate` / `format` / `channels` / a Silero policy; controls arrive out of order or twice | late config + duplicate `agent_end` before any speech + re-arm during speech | refused with `bad_config` naming each field; the session's policy is fixed at the first audio frame. **Bug found**: a late `turn_policy` was accepted silently when the audio had arrived but the session was not created yet (the session is created lazily on the compute loop); the session is now created before a late config is judged. Duplicate arms are harmless | `test_config_after_audio_and_duplicate_out_of_order_controls` |
| Binary frames cut at odd byte offsets (1, 5, 1003 bytes), an empty binary frame, a 4 MB+ frame | sent each | odd bytes are carried to the next frame: decisions identical to a clean stream; the empty frame is ignored; a frame beyond `max_size` closes that connection with 1009 (library) and the server serves the next one | `test_odd_byte_binary_frames_and_huge_binary_frame` |
| Rapid connect / disconnect, a disconnect mid-turn, a connect that never sends | 3 x (connect, 1 s audio, close), then a full session; a silent connect | the connection registry (`Engine.conns`) is always cleaned in `finally`; `sessions_active` returns to 0; the next session is complete | `test_client_disconnect_mid_turn_then_reconnect`, `test_ready_handshake_is_immediate_and_a_silent_connect_is_clean` |
| Audio faster than real time / clock drift: a client dumps 40 s at once at a model 1.5x slower than real time; the inbox grows without bound and the turn-end latency with it | `SlowEngine` (diarizer 1.5 s per audio second), 40 s flood | **backpressure**: the reader stops reading once `MAX_INBOX_S` (30 s) is queued (TCP does the rest; peak inbox 30.5 s); **load shedding** (section 4) drained the 40 s in 3.3 s wall while a second short session and `/health` were served in between (one 0.5 s round of compute at a time) | `test_fast_client_backpressure_slow_client_and_fairness` |
| Multiple concurrent sessions cross-talk (shared models, one compute thread) | 12 sessions at once, each its own audio | none needed: every session's frames, turn_ends and finals equal its solo run bitwise (the compute thread, interleaving and block sizes change nothing; the solo reference must see the same int16 quantization) | `test_concurrent_sessions_are_isolated_fair_and_deterministic` |
| No way to see the server's state from outside; log lines unparseable | - | `GET /health` and `--log-json` (section 0) | `test_health_endpoint_and_structured_json_logs` |
| The server stops with sessions open: clients see a bare TCP reset | stop mid-session | every live client gets `server_shutdown` (fatal) and close 1001 first | `test_server_shutdown_mid_session_tells_the_client` |
| A client connects and sends nothing forever (a leaked connection per crashed client) | connect, wait | `--idle-timeout-s` (default 300 s): `idle_timeout` (fatal), close 1008 | `test_idle_timeout_and_session_limit_close_with_errors` |
| A model call raises inside a session (a diarizer exception) and takes the process or the loop down | a diarizer that raises on frame 6 | the compute step is wrapped: `processing_failed` (fatal, with the exception), close 1011, counted; the reader task and the registry are cleaned; the next session is served | `test_processing_failure_is_reported_closes_the_session_and_spares_the_server` |
| `NaN` reaches the wire (`json.dumps` writes `NaN`, which is not JSON: the client's parser dies) | - | `validate` rejects any non-finite float and `send` uses `allow_nan=False`; speaker probabilities are clamped to [0, 1] | `test_error_message_shape_and_validate` |

## 3. Models and state

| What could break | Reproduced by | Fix | Test |
|---|---|---|---|
| Enrollment (`--enroll after_agent*`) with minutes of silence, or two speakers starting together | scripted embedder | never enrolls on silence with bounded histories; overlap picks a deterministic column; no crash | `test_enrollment_with_no_speech_and_on_overlap` |
| A non-finite head output (turn posterior NaN) sticks in the recurrent state; `hybrid_dyn` never fires | turn-head bias = NaN | frames with non-finite `vad` / `eot` are zeroed and `ASRStream.reset_state()` drops the encoder / decoder / turn-GRU state (frame counters and buffers kept), reported once as `nan_state_reset`; the Silero path of `hybrid_dyn` still ends the turn at p = 0 | `test_hybrid_dyn_with_nan_head_posterior_still_ends_turns` |
| Diarizer rows with NaN | scripted | zeroed + `nan_state_reset`; probabilities stay in [0, 1] | `test_diarizer_nan_rows_are_zeroed_and_reported` |
| Silero VAD file missing or onnxruntime broken: `--silero` at startup, or the first `hybrid_silero` / `hybrid_dyn` session, died with a traceback | `/nonexistent.onnx` | at startup: one clear line, exit 2; at session start: `silero_unavailable` (non-fatal) and the session runs `hybrid` at the Silero policy's head threshold (head OR diarizer timeout) instead of dying; `/health` shows `silero` | `test_silero_missing_at_startup_is_a_clear_error_and_sessions_fall_back`, `test_serve_shipped.py::test_silero_not_loaded_unless_a_silero_policy_is_used` |
| The `--final-asr` worker process dies (pipe closed), fails every job while alive, or hangs: the client never gets that source's final and `end` waited forever for it | fake worker in three modes | `Engine.submit_final` never raises (a dead worker -> failed Future); `FINAL_ASR_TIMEOUT_S` (60 s) on every delivery; the job completes with the **streaming final's text under the offline source** (`final_asr_failed`, non-fatal, once); the worker is restarted in the background (dead or hung: at once; failing while alive: after `FINAL_ASR_FAILS_BEFORE_RESTART` = 2 consecutive failures; at most one restart per `FINAL_ASR_RESTART_S`); `FinalASRWorker.alive` / `restart()` added; `/health` shows `final_asr.alive` / `restarting` | `test_final_asr_worker_crash_falls_back_to_the_streaming_final_and_restarts[dead,failing,hanging]` |
| The `--asr-lookahead` second pass raises, or is too slow under load | a pass that raises; a level-2 block | dropped for the rest of the session (`lookahead_dropped`, once); pending lookahead finals carry the streaming text with `latency_ms 0`; every final still arrives | `test_lookahead_pass_dropped_under_load_or_failure_falls_back_to_single_pass` |
| `--device mps/cuda` raised; a missing model / Silero / TitaNet / LID / final-ASR file was found at the first frame | - | `--device` other than cpu falls back with a warning (counted `device_fallback`); `_startup_checks` names every missing file and exits 2 before loading anything; a load failure prints one clear line before the traceback | `test_device_fallback_and_startup_checks` |

## 4. Watchdog: load shedding

The compute loop processes a connection's backlog in rounds of at most 0.5 s of audio (fairness across sessions;
`/health` and other sessions run in between). Each round the session gets its own backlog and the watchdog's global
level (`Engine.global_shed`): level 1 at 1.5 s of backlog skips the diarizer for the block (`StreamingDiarizer.feed
(skip=True)` emits zeros without running the encoder; the served VAD stands in as column 0, so the timeout policy
keeps working); level 2 at 4 s also drops partials, batches the block's frames into one `frames` message and drops
the lookahead pass for good. Reported as `overloaded` at most once per 5 s, counted as `shed_diar_frames`,
`shed_partials`, `frames_batched`.

**A level is entered only while the session's recent RTF (last 4 s of audio) is >= `SHED_RTF` = 0.9**, i.e. the
compute really cannot keep up; a burst that a faster-than-real-time server drains by itself is never shed, so an
unpaced eval driver and a paced client measure the same thing (this is the rule that keeps measured behaviour
unchanged). Once entered, a level is kept until the backlog falls back under its threshold. The global level
follows the same rule with `Engine.global_load()` (sum of recent RTFs over the live sessions) against the one
compute thread.

| What could break | Reproduced by | Fix | Test |
|---|---|---|---|
| Shedding levels do the wrong thing or report every block | backlog 0 -> 1.5 s -> 4 s | levels 0 / 1 / 2 as designed, one notice, VAD stand-in in column 0, every frame delivered (batched or not), decisions still produced | `test_shedding_levels_skip_diarizer_then_partials_and_report_once` |
| Level 2 drops audio, not just messages | shed vs plain run | identical frames (vad) and finals; partials dropped | `test_shedding_keeps_decisions_identical_when_diarizer_is_a_noop` |
| Shedding changes results for a client that merely sends faster than real time | 40 s "queued" behind every block on a fast server | nothing shed (`recent_rtf` < 0.9), decisions equal the paced run bitwise; the global watchdog needs load too | `test_burst_backlog_on_a_fast_server_is_not_shed` |
| A degradation code the server can emit is not in `ERROR_CODES` (so `validate` / clients reject it) | source scan | every `_notice` / `error_msg` / `send_error` / `_SessionFailed` code is in `ERROR_CODES` | `test_stats_and_health_counters_cover_every_degradation` |

## 5. Clients and adapters

| What could break | Reproduced by | Fix | Test |
|---|---|---|---|
| `scripts/stream_client.py`: the server closes the session (fatal error) while the client is still sending -> `ConnectionClosed` traceback, no summary | server with a diarizer that raises | the sender stops when the receiver ends, send errors are caught, non-JSON / malformed server frames are skipped; the summary carries `errors`, `fatal_error`, `complete` | `test_stream_client_survives_a_fatal_server_error_and_reports_it` |
| Pipecat `AudioforgeSTTService` logs "unknown server message type 'error'" and never surfaces a fatal error | fed `error` messages | non-fatal: warning + `hub.errors`; fatal: `push_error` to the pipeline | `test_pipecat_adapter_accepts_error_messages` |
| LiveKit `_Link` ditto | same over a live socket | `errors` list; a fatal error sets `_Link.error` so the session ends with a reason | `test_livekit_link_records_errors_and_fatal_ends_the_session` |
| `audioforge-client` (`packages/audioforge-client`) | - | `ServerError(code, fatal)` and `frames` batches are parsed by the package (maintained separately) | `packages/audioforge-client/tests/test_protocol.py` |

## 6. Still fragile (known, not fixed here)

- **One compute thread** (`Engine.executor`): a session's 0.5 s round is the unit of fairness; a pathological block
  (e.g. the first block after a 30 s silence with `--enroll`) delays every other session by its duration. Shedding
  reacts within one round.
- **Shedding is per-session RTF**: with N sessions each individually under 0.9 RTF but together over 1.0, only the
  global rule (backlog per connection + `global_load`) engages, and it needs the per-connection backlog thresholds
  to be crossed first.
- **The diarizer skip path** is a hard degradation: speaker columns other than 0 read 0 while shed, so speaker
  changes are invisible to the primary rule during overload (the timeout policy on the VAD column still ends turns).
  In `causal` mode the encoder still runs for its own cache; only the head is skipped.
- **`/health` shares the event loop** with the sessions; it is served between rounds, so under a saturated loop it
  is delayed by up to one round.
- **Silero fallback changes the policy**: `hybrid_silero` / `hybrid_dyn` without Silero run as `hybrid`, which is
  measured to end turns later on long pauses (research/E2E_FINAL.md). It is reported; a deployment that wants the
  Silero policies must ship the file (`--silero` makes it fatal at startup).
- **Soak length**: the in-repo soak is 30 min on tiny models; a 1-hour run with the shipped 115M model + Sortformer
  on this machine was not repeated in this pass (the earlier real-model runs in research/E2E_FINAL.md are <= 10 min
  per clip). The bounds are structural (rings / deques / no per-frame lists), so the risk is in torch allocator
  fragmentation, which `peak_rss_mb` in `stats` will show.
- **`ASRStream.tokens` / `tok_at` still grow** (one int per emitted token / per frame, absolute frame indexing:
  ~45 k entries, under 1 MB, per hour at 80 ms frames); `--max-session-s` bounds them. Compacting them needs an
  offset in every `tok_at` reader and was not worth the risk in this pass.
- **`MAX_INBOX_S` backpressure relies on TCP**: the websockets library may buffer up to its own `max_queue`
  frames beyond the 30 s; with 20 ms frames that is under a second, with 1 s frames it is `max_queue` seconds.
- **`frames` batches** are new: a third-party client that only knows `frame` loses per-frame messages while a
  session is at level 2 (it still gets turn_ends, finals and the `overloaded` error).
