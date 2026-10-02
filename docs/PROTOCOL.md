# audioforge wire protocol

This is the WebSocket protocol of the streaming speech server `audioforge.serve`
(`audioforge/serve.py`). The module docstring of `serve.py` is the normative specification; this page restates it
with every field in one place. The server's `validate()` function checks outgoing messages. In `--debug-fields`
mode the server runs it on every message before sending; the test suite runs it on every message in the tests.
Without `--debug-fields`, every message carries exactly the keys listed here for the flags in use.

Server flags are documented in [CONFIGURATION.md](CONFIGURATION.md).

## Contents

1. [Overview](#1-overview)
2. [Connection lifecycle](#2-connection-lifecycle)
3. [Time semantics](#3-time-semantics)
4. [Client to server messages](#4-client-to-server-messages)
5. [Server to client messages](#5-server-to-client-messages)
6. [Keys added by `--debug-fields`](#6-keys-added-by---debug-fields)
7. [Message order](#7-message-order)
8. [Minimal Python client](#8-minimal-python-client)
9. [Example session transcript (illustrative)](#9-example-session-transcript-illustrative)
10. [Errors, limits and degraded operation](#10-errors-limits-and-degraded-operation)
11. [Health endpoint](#11-health-endpoint)
12. [Known protocol limitations](#12-known-protocol-limitations)

## 1. Overview

- Transport: WebSocket, `ws://<host>:<port>` (default `ws://127.0.0.1:8765`). Any path is accepted for the
  WebSocket handshake. A plain HTTP `GET /health` (also `/healthz`, `/stats`) on the same port returns server
  health as JSON ([§11](#11-health-endpoint)).
- One connection is one session. Sessions share nothing except the loaded models. All compute runs on one worker
  thread shared by every session.
- Client to server: binary frames of PCM audio, plus JSON text frames (`config`, `end`, `agent_end`, `enroll`).
- Server to client: JSON text frames only. Each frame holds one JSON object with a `"type"` key. Problems are
  reported as structured `error` messages ([§5.9](#59-error)).
- Maximum incoming message size: 4 MiB per WebSocket message (`max_size=2**22`); text messages over 64 KiB are
  refused with an `error`. The server sends no WebSocket pings (`ping_interval=None`).
- Two models run side by side on one 80 ms frame clock. The ASR front end (frozen NVIDIA cache-aware FastConformer,
  160 ms chunks, plus our VAD, turn and speaker heads) produces the transcript, `vad` and `eot`. NVIDIA Streaming
  Sortformer v2 (or Nemotron-3-Diarization, see CONFIGURATION.md) produces the four speaker-activity columns.

## 2. Connection lifecycle

```
client                                   server
  | ---- WebSocket connect ------------->  |
  | <--- ready --------------------------  |   sent immediately, before any client message
  | ---- config (optional, text) ------->  |   send it first, before any audio
  | ---- binary PCM, any frame size ---->  |
  | <--- frame / partial / turn_end /      |   streamed while audio arrives
  |      final / enrolled / language ----  |
  | ---- agent_end / enroll (text) ----->  |   only with --enroll (optional, any time)
  | ---- more binary PCM --------------->  |
  | ---- {"type": "end"} --------------->  |   flush
  | <--- remaining frames, turn_ends ----  |
  | <--- final (end of stream) ----------  |
  | <--- final (source tdt_v3), if any --  |   --final-asr only; the stats wait for them
  | <--- stats --------------------------  |
  | <--- close --------------------------  |   the server closes the socket
```

1. **Connect.** The server sends `ready` at once.
2. **Configure (optional).** Send one `config` text message before the first audio frame. `sample_rate`,
   `format` and `channels` are honoured only before the first binary frame. `turn_policy` values `hybrid_silero` and `hybrid_dyn`, and `turn_preset`, are honoured
   only if the session has not started yet: the Silero chunk grid starts with the stream. The server creates the
   session at the first audio block or at the first `agent_end` / `enroll`. The other fields (`turn_policy` other
   than the two Silero policies, `timeout_ms`, `eot_threshold`) may be changed later; a later `config` is applied to
   the running session from the next processed block on. Fields that are missing keep their current value. If any
   field was ignored or clamped, the server answers with one non-fatal `error` message with code `bad_config`
   listing the problems.
3. **Stream audio.** Binary frames of little-endian PCM: signed 16-bit by default, or 32-bit float with
   `config.format: "float32"`; mono by default, or interleaved channels (averaged to mono) with `config.channels`.
   The rate is 16 kHz unless `config.sample_rate` says otherwise (other rates are resampled linearly to 16 kHz). Any
   frame size is accepted: trailing bytes that do not complete a sample frame are carried over to the next binary
   message. The server processes what it has received once at
   least 20 ms (320 samples at 16 kHz) is queued, in blocks of at most 0.5 s. Events and finals do not depend on how
   the client splits its audio (see [time semantics](#3-time-semantics)). If more than 30 s of audio is queued
   unprocessed for a connection, the server stops reading from it until the queue drains (TCP backpressure).
4. **End.** Send `{"type": "end"}`. The server stops reading, processes the queued audio, flushes both models,
   sends the remaining `frame` / `turn_end` / `final` messages, then one end-of-stream `final`, then (with
   `--final-asr`) the pending offline finals, then `stats`, and closes the connection. Messages sent after `end`
   are not read.
5. **Disconnect without `end`.** The session is dropped. No `final` or `stats` is sent.
6. **Server-side close.** The server also ends a connection itself, always after a fatal `error` message: no
   message for `--idle-timeout-s` (default 300 s; code `idle_timeout`), more than `--max-session-s` of audio (default
   4 h; code `session_limit`, followed by the normal flush: end-of-stream `final` and `stats`), a processing
   failure (`processing_failed`), an internal error (`internal_error`), or server shutdown (`server_shutdown`). See
   [§10](#10-errors-limits-and-degraded-operation) for the WebSocket close codes.

Text frames that are not valid JSON (`bad_json`), JSON values that are not objects with a string `type`
(`bad_message`), objects with an unknown `type` (`unknown_type`), and an `agent_end` or `enroll` that is not the
trigger of the server's `--enroll` mode (`unsupported`) are answered with a non-fatal `error` message and otherwise
ignored; the session continues.

## 3. Time semantics

All times are **audio time in seconds**, counted from the first sample the client sent (after resampling to
16 kHz), rounded to 3 decimals. They never depend on wall-clock time, server load or client frame size.

- **Frames.** Frame `v` covers audio `[0.08 v, 0.08 (v + 1))` s. Its `t` is the **end** of the frame,
  `(v + 1) * 0.08`. Both models emit exactly one frame per 80 ms of audio.
- **ASR availability.** The ASR runs in 160 ms chunks (two frames). A chunk's two frames are available 6 ms after
  the chunk's audio has arrived (half of the STFT window), i.e. 6 to 86 ms after each frame ends, plus compute.
  A `frame` message is sent when its ASR frame is ready.
- **Diarizer column lag.** The diarizer finalizes column `t` only when its chunk of C frames plus R right-context
  frames has arrived: at audio time `((t // C + 1) * C + R) * 0.08` s. That is 80 to 240 ms (mean 160 ms) after the
  frame ends for the default `--diar-config low_latency_032` (C 3, R 1), and 560 to 960 ms (mean 760 ms) for
  `low_latency` (C 6, R 7), plus the diarizer's compute time. `ready.column_lag_ms` carries the structural mean
  (160.0 or 760.0 for the two presets; compute is excluded).
- **`speakers` and `primary` in a `frame` are older than its `t`.** A `frame` carries the newest diarizer column
  available when its ASR frame is ready, so `speakers` / `primary` describe audio about `column_lag_ms` plus compute
  before `t`. Measured column age on arrival, CPU, 2 threads, 1x, 2 AMI dev windows: p50 210-227 ms (p95 291 ms)
  with `low_latency_032` and 857 ms (p95 1035 ms) with `low_latency`
  ([EARLY_RESULTS](../research/EARLY_RESULTS.md#live-streaming-server-2026-09-26),
  [INTEGRATION.md §1](../research/archive/INTEGRATION.md#1-executive-summary)). The column's own frame end is sent only as
  the debug field `spk_t`. Until the first column exists (about 0.3 s with the default preset, about 1 s with
  `low_latency`), `speakers` is `[0, 0, 0, 0]`, the same as "nobody talking".
- **Decision time.** `turn_end.t` and the `t` of the `final` that follows it are the **decision time**: the audio
  time at which all data behind the decision was complete.
  - `timeout` / `timeout_quiet` path: the finalization time of the deciding diarizer column (formula above).
  - `head` path: the end of the ASR chunk holding the deciding frame; with `--turn-input diar`, the later of that
    and the diarizer column's finalization time.
  - Silero path (`hybrid_silero`, `hybrid_dyn`): the completion time of the 32 ms Silero chunk holding the frame
    end (so `t` need not be a multiple of 0.08); for `hybrid_dyn`, the later of that and the head frame's time.
  - `hybrid*`: the earlier of the candidate paths' decision times.
- **Finals are cut at the decision time.** The `final` at a `turn_end` holds exactly the tokens of the ASR frames
  available at `turn_end.t`. Tokens decoded later belong to the next segment. Because of this, events and finals
  are identical for any client frame size, thread count and streaming speed (tested with frame sizes 320 and 333
  samples, and on the real models).
- **`partial.t`** is the end of the latest ASR frame decoded so far.
- **End-of-stream `final.t`** is the total audio length received (not rounded to a frame boundary).
- **`enrolled.t`** is the finalization time of the diarizer frame on which the binding was made (after_agent /
  explicit: the 19th active frame of the chosen column; after_agent_arm: the 3rd consecutive active frame).
- **`language.t`** is the end of the frame that decided the announcement.
- **Offline finals** (`source` `tdt_v3` or a model name) keep the `t` of the streaming final they complete;
  `start` / `end` give the transcribed audio span in seconds.

## 4. Client to server messages

### 4.1 Binary audio

| property | value |
|---|---|
| encoding | `int16` (default): signed 16-bit little-endian (`<i2`), scaled by 1/32768. `float32` (`config.format`): 32-bit float little-endian (`<f4`); non-finite samples are replaced by 0 and values outside [-1, 1] are clipped |
| channels | 1 (default), or 1-8 interleaved channels (`config.channels`), averaged to mono |
| sample rate | 16000 Hz unless `config.sample_rate` (8000-192000) is set before the first audio frame; other rates are resampled linearly |
| frame size | any; bytes that do not complete a sample frame (sample width x channels) are kept for the next message |
| maximum message | 4 MiB |

### 4.2 `config` (optional, text)

Send before the first audio frame. Unknown keys are ignored. Invalid values are ignored or clamped; the server then
sends one non-fatal `error` message with code `bad_config` whose `detail` lists every ignored or clamped field. The
config message never ends the session.

| field | type | default | allowed values | effect |
|---|---|---|---|---|
| `type` | string | | `"config"` | |
| `turn_policy` | string | the server's `--turn-policy`: `"vad_head"` in single mode (the default), `"timeout"` in room mode | `timeout`, `timeout_quiet`, `timeout_any`, `head`, `both`, `hybrid`, `hybrid_silero`, `hybrid_dyn`, `vad_head` | end-of-turn rule; see [CONFIGURATION.md §4](CONFIGURATION.md#4-turn-policies-configturn_policy). An unknown value is ignored (the current policy stays). `hybrid_silero` / `hybrid_dyn` / `timeout_any` must be set before the session starts. `timeout_any` (multi-party rooms, [CONFIGURATION.md §7.4](CONFIGURATION.md#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)) ends *every* speaker's turn: `timeout_ms` of nobody talking, or a speaker change (`turn_end.policy: "change"`) |
| `turn_preset` | string | the server's `--turn-preset` (`"balanced"`) | `balanced`, `fast`; must be set before the session starts | `vad_head`'s constants as one named trade-off: `balanced` = VAD < 0.4 for ≥ 160 ms AND p ≥ 0.99, OR 640 ms; `fast` = VAD < 0.6 for ≥ 480 ms AND p ≥ 0.99, OR 720 ms (others path 640,640). `fast` is ~70 ms faster at p50 and ~490 ms faster at p95 on two-party calls with half the missed ends, for 4.6 points more false interruptions ([CONFIGURATION.md §4](CONFIGURATION.md#4-turn-policies-configturn_policy)). An unknown value, or a change after the session started, is ignored with `bad_config` |
| `timeout_ms` | integer (number accepted) | `1000` | clamped to [80, 4840] | primary-column silence that ends a turn for `timeout`, `timeout_quiet`, `both`, `hybrid`. Upper bound = 5 s primary window minus 160 ms. Not used by `hybrid_silero` / `hybrid_dyn` |
| `eot_threshold` | number | policy-dependent: `0.99828` for `hybrid_silero`, `0.998283` for `hybrid_dyn`, `0.99` for `vad_head`, else `0.98` | clamped to [0, 1] | turn-head threshold for `head`, `both`, `hybrid*`. Setting it overrides the Silero policies' frozen thresholds |
| `sample_rate` | integer | `16000` | 8000-192000; only before the first audio frame | input sample rate of the binary frames; other values are ignored |
| `format` | string | `"int16"` | `int16`, `float32`; only before the first audio frame | sample format of the binary frames |
| `channels` | integer | `1` | 1-8; only before the first audio frame | interleaved channels in the binary frames, averaged to mono |

The policy-dependent threshold applies only when `eot_threshold` is absent. Sending `"eot_threshold": 0.98` together
with `"turn_policy": "hybrid_dyn"` replaces the measured operating point 0.998283 with 0.98.

### 4.3 `end` (text)

```json
{"type": "end"}
```

Flushes the session. The server answers with the remaining messages, the end-of-stream `final`, `stats`, and closes.

### 4.4 `agent_end` (text; `--enroll after_agent` or `after_agent_arm`)

```json
{"type": "agent_end"}
```

Send it when the agent's own TTS playback has finished. It is applied at the audio position received so far, in
order with the audio: from diarizer frame `floor(samples_received / 1280)` on (16 kHz samples), the first
diarizer column active (p > 0.5) for at least 3 consecutive frames becomes the user. Every `agent_end` re-arms the
choice.

- `after_agent`: the chosen column's first 19 active frames (1.5 s) are the TitaNet-L voice enrollment; afterwards
  the primary follows that voice across column swaps.
- `after_agent_arm`: after the choice the primary follows the causal_dominant rule seeded at that column (no
  embedding).

Integrations send it automatically: Pipecat `AudioforgeSTTService(enroll="after_agent" | "after_agent_arm")` sends it on
`BotStoppedSpeakingFrame`; LiveKit `AudioforgeFrontend.agent_end()` (or `attach(session)`) sends it.

### 4.5 `enroll` (text; `--enroll explicit`)

```json
{"type": "enroll"}
```

The same as `agent_end` for the `explicit` mode (for example after a "say something" prompt): the next speaker
active for 3 consecutive frames is chosen and enrolled over 1.5 s of speech. One enrollment per message.

With `--turn-input tsvad`, and so in single-model mode (`audioforge-serve`'s default since 2026-09-29), `enroll` can
carry a stored voice print. **This is how a single-model session learns who the user is: send it right after the
config.**

```json
{"type": "enroll", "embedding": [0.012, -0.034, "... 192 numbers"]}
```

The print is 192 finite numbers from the served speaker head: `audioforge.voiceprint(audio)` over at least 5 s of the
user's clean speech, 10 s for meetings, or the `embedding` of an earlier `voiceprint` message. Live grabs are not
enough: research/SINGLE_MODEL.md A2 measured 16-55 more points of missed turn ends. The print is taken at once, under **every** `--enroll` mode, and announced with
a `voiceprint` message (`source: "explicit"`). Without `--turn-input tsvad` the embedding is refused with a
`bad_message` error. Without an embedding, the TS-VAD path takes the print live: the first `--tsvad-print-s` seconds
of speech after the mode's trigger (`agent_end` for `after_agent_arm`).

## 5. Server to client messages

Types used in the tables: `number` = JSON number (integer or float); `int` = JSON integer; `string`; `bool`;
`null`; `list`; `object`. "Always" means present in every message of that type.

### 5.1 `ready`

Sent once, immediately after the connection opens.

| field | type | meaning | when present |
|---|---|---|---|
| `model` | string | `<asr file stem>+<diarizer file stem>`, e.g. `stage1_served+nemo_sortformer_v2`; the ASR stem alone when no diarizer is loaded (`--mode single`) | always |
| `chunk_ms` | number | ASR chunk length in ms (160) | always |
| `frame_ms` | number | frame length in ms (80) | always |
| `diar_config` | string | diarizer preset name (`low_latency_032` or `low_latency`), with `+custom` appended when `--diar-set` changed any field; `off` with `--diar-off` | always (added 2026-09-26) |
| `column_lag_ms` | number | mean structural diarizer finalization lag after the frame end, compute excluded (160.0 / 760.0 for the presets) | always (added 2026-09-26) |
| `enroll` | string | the `--enroll` mode | `--enroll` other than `dominant` |
| `enrolled` | bool | always `false` in `ready` | with `enroll` |
| `primary_column` | int or null | always `null` in `ready` | with `enroll` |
| `final_asr` | string | comma-separated final sources besides `stream`, e.g. `tdt_v3`, `lookahead`, `slow`, or `lookahead,tdt_v3` | `--final-asr`, `--asr-lookahead` and/or `--final-chunk-ms` |
| `final_chunk_ms` | number | chunk of the slow pass that writes the `final` text (560 or 1120) | `--final-chunk-ms` 560 / 1120 only |

### 5.2 `frame`

One per 80 ms frame of audio, in frame order, sent when the frame's ASR output is ready.

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | end of the frame, s | always |
| `vad` | number in [0, 1] | our VAD head's speech probability (any speaker) | always |
| `eot` | number in [0, 1] or null | the turn head's latest end-of-turn probability; `null` until the head has produced its first value (with `--turn-input diar` the head runs on the diarizer's clock, so the first values arrive later) or when the model has no turn head | always |
| `speakers` | list of 4 to 8 numbers in [0, 1] | activity probability of the diarizer's columns (arrival-order slots, not identities), from the newest finalized column; all zeros before the first column. 4 with Sortformer v2 or `--diar-spks 4`; 8 with an uncut Nemotron-3-Diarization (since 2026-09-28; the length is fixed per server) | always |
| `primary` | int (< the number of columns) or null | the primary-speaker column at that diarizer frame (see [CONFIGURATION.md §6](CONFIGURATION.md#6-primary-speaker-enrollment---enroll)); `null` when no column was active (p > 0.5) in the last 5 s and no column is bound. With `turn_policy: timeout_any` the current speaker's column, `null` while the floor is open | always |

Under load shedding level 2 ([§10](#10-errors-limits-and-degraded-operation)) the frames of one processed block are
sent as one batch message instead: `{"type": "frames", "items": [<frame object without "type">, ...]}`. Clients
should accept both forms (the LiveKit adapter does).

### 5.3 `partial`

Sent after a processed block when the text of the current segment changed.

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | end of the latest decoded ASR frame, s | always |
| `text` | string | transcript since the last cutting `final` (the whole segment so far, not a delta). After a cut no partial is sent until new text appears | always |

### 5.4 `turn_end`

Sent when a turn policy fires.

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | decision time, s (see [time semantics](#3-time-semantics)) | always |
| `policy` | string | `timeout`, `head`, `hybrid`, `hybrid_silero`, `hybrid_dyn` or `vad_head`. `timeout_quiet` fires are tagged `timeout`; with `both` each path is tagged with its own name. `timeout_any` fires are tagged `timeout` (nobody talking for `timeout_ms`) or `change` (another speaker took over; `silence_ms` = the previous speaker's silence) | always |
| `p` | number or null | the turn head's probability. `head` path: the value that crossed the threshold. `timeout` under `timeout` / `timeout_quiet` / `both`: `null`. Under `hybrid*`: the head value used by the firing path, or for a timeout / Silero firing the latest head value ready by `t` (`null` without a turn head) | always |
| `silence_ms` | int | the firing path's silence in ms. timeout: silence of the diarizer primary column; head: frames since the VAD head last exceeded 0.5, times 80; Silero paths: Silero any-speaker silence; `vad_head`: the served VAD head's silence (head path / fallback) or the user's TS-VAD silence (others path) | always |
| `hinted_at` | number or null | the `t` of the `turn_end_hint` this decision confirms ([§5.11](#511-turn_end_hint-and-turn_end_hint_cancel)); `null` when no hint was outstanding | with turn hints on (the default; not with `--turn-hint-off`). Added 2026-09-30 |
| `path` | string | which `vad_head` path fired: `head` (turn head p >= theta after K ms of silence), `fallback` (the silence timeout), `others` (another speaker has the floor) or `model` (`--turn-model smartturn`: smart-turn said complete; `p` is then smart-turn's P(complete)) | `policy: "vad_head"`. Added 2026-09-30 |
| `model_ms` | number | wall time of the smart-turn call that decided (features + ONNX, ms); the reply can start `model_ms` after the audio time `t` at the earliest | `path: "model"` (`--turn-model smartturn`). Added 2026-09-30 |

Under `vad_head` the server's energy gate (`--energy-gate`, on by default; CONFIGURATION.md §4) sends no `turn_end`
and no `turn_end_hint` before 160 ms of speech onsets in the session, and arms each user turn only on an onset (VAD >
0.5 AND frame energy > the session's noise floor + 6 dB): leading room tone no longer ends an empty turn.

Which fires cut a `final` (the "cutting policy"): `timeout` fires for `timeout`, `timeout_quiet` and `both`; `head`
fires for `head`; the single merged fire for `hybrid`, `hybrid_silero`, `hybrid_dyn`, `vad_head`. Under `both`, `head` fires
are reported without a `final`.

### 5.5 `final`

Sent right after every cutting `turn_end` (same `t`), and once at end of stream.

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | decision time of the `turn_end` it follows; for the end-of-stream final, the total audio length | always |
| `text` | string | the segment's text: tokens from the previous cut to the ASR frames available at `t`. May be `""` | always |
| `speaker` | int or null | default (`--diar-labels column`): the primary column at the decision (the timeout's firing column, else the current primary), 0-3 (0-7 with 8 columns). With `--diar-labels registry`: a stable per-session speaker id (0, 1, 2, ... in order of first appearance, keyed by voice, so the same person keeps the id across column permutations and re-entries; may exceed the column count); `null` when the turn had too little speech to identify ([CONFIGURATION.md §7.4](CONFIGURATION.md#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)) | always |
| `speaker_conf` | number in [0, 1] or null | `registry`: cosine of the turn's voice to the assigned speaker (a new speaker: how far the closest known one was below the join threshold, 1 for the first); `null` when `speaker` fell back to the column's last id or is `null` | only with `--diar-labels registry` or `--shed-diar hold`; then on every `stream` final together with `diar_shed` |
| `diar_shed` | bool | true when the diarizer did not run on part of this turn (load shedding): the speaker came from held columns / the voice registry, not from a live diarizer frame | with `speaker_conf` |
| `source` | string | `stream` for the streaming model's final; `lookahead` for the `--asr-lookahead` pass; `slow` for the `--final-chunk-ms` pass; `tdt_v3` (or the `.nemo` file's stem when `--final-asr` is a path) for the offline pass | only when `--final-asr`, `--asr-lookahead` or `--final-chunk-ms` is on; then on every final |
| `pass` | string | `slow`: the text is the slow pass's; `fast`: the slow pass was dropped (load shedding level 2 or an error) and the text is the fast pass's | `--final-chunk-ms` finals only |
| `start` | number or null | start of the transcribed span, s. tdt_v3: turn onset (first frame with VAD > 0.5 after the previous cut) minus 0.3 s, never before the previous span's end. lookahead: the lookahead segment's first frame | finals whose `source` is not `stream` |
| `end` | number or null | end of the transcribed span, s. tdt_v3: last VAD speech frame plus 0.5 s, capped at `t`. lookahead: the frame where the lookahead segment was cut | finals whose `source` is not `stream` |
| `latency_ms` | number | tdt_v3: wall time from submitting the job (right after the `turn_end` is computed) to sending this message. lookahead: **audio-time** wait past the decision time (0 when the policy already waited long enough). slow: the turn_end flush's compute in ms (with `--final-flush off`: the audio-time wait for the slow chunk) | finals whose `source` is not `stream` |

Details of the extra finals:

- **`tdt_v3` (`--final-asr`).** For each cut segment the server takes the turn's audio (at most the last 60 s) and
  transcribes it once with NVIDIA Parakeet-TDT 0.6B v3 in a separate worker. The message arrives asynchronously,
  after the streaming `final` of the same turn and possibly after later `frame` messages, with the same `t` and
  `speaker`. A segment with no VAD speech gets an empty tdt_v3 final at once, with `start` / `end` `null` and
  `latency_ms` 0.0. If the offline pass fails (worker crash or error), the server sends a non-fatal
  `final_asr_failed` `error` (once per session) and completes the turn with a final under the offline `source` that
  carries the **streaming** text, so a client waiting for that source still gets the turn. A dead worker is restarted
  in the background, at most once per 60 s. The `stats` message is sent only after every pending offline final.
- **`slow` (`--final-chunk-ms 560|1120`, dual rate, research/DUAL_RATE.md).** The heads, partials and turn
  decisions keep the 160 ms pass. At a cutting `turn_end` the server sends that pass's final at once as a
  **`final_fast`** message (exactly the fields the plain `final` has without the flag, so a client can hand it to
  the LLM with no added delay), then a `final` with `source: "slow"` and `pass` holding the text of a second,
  text-only pass of the same frozen encoder at 560 ms (`[70,6]`) or 1120 ms (`[70,13]`) chunks over the same
  frames (`start` / `end` = the span, the same cut as `final_fast`). The slow pass's chunk holding the turn's last
  frames is normally not complete at the decision; the server encodes the audio up to the decision time as a partial
  chunk (a throw-away copy of the pass; the pass itself is unchanged) and sends the slow final right after the
  `turn_end` batch, without waiting for the rest of the chunk. A client that wants the better text replaces the
  `final_fast` text with the `final` of the same `t`.
- **`lookahead` (`--asr-lookahead R`).** A second, text-only pass of the same ASR model with R frames of right
  context. Its final for a segment is sent once that pass has decoded 3 frames past the segment's last VAD speech
  frame (capped at the streaming cut). Partials, heads and events are unchanged by it. If the pass is dropped
  (under load shedding level 2 or after an error), a `lookahead_dropped` `error` is sent once and the remaining
  lookahead finals carry the streaming text with `latency_ms` 0.

### 5.6 `enrolled` (`--enroll` other than `dominant`)

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | finalization time of the diarizer frame on which the binding completed | always |
| `column` | int | the bound diarizer column | always |

Sent once per arming (per `agent_end` / `enroll`). `after_agent` / `explicit`: when the 1.5 s voice enrollment
completes. `after_agent_arm`: when the column is chosen. The bound column then appears as `frame.primary` and
`final.speaker`.

### 5.7 `language` (`--lid`)

| field | type | meaning | when present |
|---|---|---|---|
| `t` | number | end of the frame that decided it | always |
| `language` | string | language code, e.g. `en`, `he`, `ar` | always |
| `confidence` | number in [0, 1] | posterior of that language | always |

Sent when the top language's posterior first reaches `--lid-threshold` after at least `--lid-min-ms` of pooled
speech, and again whenever the confident top language changes.

### 5.8 `stats`

The last message of a session, after the end-of-stream `final` (and after every pending offline final).

| field | type | meaning | when present |
|---|---|---|---|
| `rtf` | number | processing time / audio time for this session; 0.0 for a session without audio | always |
| `speakers_seen` | int | the session's speaker-count estimate: the number of ids in the voice registry | only with `--diar-labels registry` |
| `chunk_ms_p50` | number | median processing time per 160 ms of audio, ms | always |
| `chunk_ms_p95` | number | 95th percentile of the same, ms | always |
| `first_partial_ms` | number or null | ms from the arrival of the block that produced the first non-empty partial to that partial being computed (server lag only, not user-perceived latency); `null` if no text | always |
| `peak_rss_mb` | number | peak RSS of the server **process** (lifetime peak, not per session), MB | always |
| `enroll` | string | `--enroll` mode | `--enroll` other than `dominant` |
| `enrolled` | bool | a binding has completed since the last arming | with `enroll` |
| `primary_column` | int or null | the currently bound column | with `enroll` |
| `final_asr` | string | the final sources, as in `ready.final_asr` | `--final-asr` and/or `--asr-lookahead` |
| `final_latency_ms` | object | per source: `{"p50", "p95", "max", "n"}` of that source's `latency_ms` values | with `final_asr` |
| `final_asr_rss_mb` | number or null | peak RSS of the offline final-ASR worker process, MB; `null` in `--final-asr-worker thread` mode or with only `--asr-lookahead` | with `final_asr` |
| `lang` | string or null | the last announced language, `null` if none | `--lid` |
| `degraded` | object | counters by code of every degradation or repair in this session (for example `{"overloaded": 1, "shed_diar_frames": 120}`) | only when the session degraded |
| `turn_model` | object | `{"model": "smartturn", "calls", "complete", "ms_p50", "ms_p95"}`: the session's smart-turn calls, how many said complete, and their compute (ms) | `--turn-model smartturn` |
| `turn_hints` | object | `{"sent", "confirmed", "cancelled", "open", "lead_ms_p50"}`: the session's `turn_end_hint`s and how they resolved (`open`: 1 if one was outstanding at the end; `lead_ms_p50`: median `turn_end.t - hint t` of the confirmed ones, `null` if none) | with turn hints on (the default) and a turn head |

The enrollment keys (`enroll`, `enrolled`, `primary_column`) come together or not at all, and so do the final-ASR
keys; `validate()` enforces this.

### 5.9 `error`

Added 2026-09-28. Clients that do not know the type may ignore it; the other messages are unchanged by it.

| field | type | meaning | when present |
|---|---|---|---|
| `code` | string | one of the codes in [§10](#10-errors-limits-and-degraded-operation) | always |
| `detail` | string | human-readable explanation, at most 500 characters | always |
| `fatal` | bool | `true`: the server closes the connection right after this message. `false`: a degradation or a refused message is reported and the session goes on | always |

Degradation notices (`fatal: false` codes raised inside a session) are sent once per code per session; every
occurrence is counted in `stats.degraded`.


### 5.10 `voiceprint` (`--turn-input tsvad`, single-model mode)

```json
{"type": "voiceprint", "t": 0.0, "seconds": 0.0, "source": "explicit", "embedding": [0.0213, -0.0871, "..."]}
```

Sent whenever the session starts following a new voice print. `t` is the audio time it applies from. `seconds` is
how much live speech it was taken from (0 for a stored print). `source` is `explicit` (an `enroll` with an
embedding), `arm` (the speech after `agent_end` / `enroll`) or `refresh` (`--tsvad-refresh-s`). `embedding` is the
print itself (192 numbers, added 2026-09-29): store it and send it back with `enroll` next session.

### 5.11 `turn_end_hint` and `turn_end_hint_cancel`

Added 2026-09-30; on by default, off with `--turn-hint-off`; the threshold is `--turn-hint-p` (default 0.8). Needs a
turn head. Clients that do not know these types may ignore them: `turn_end` itself is unchanged.

```json
{"type": "turn_end_hint", "t": 11.616, "p": 0.87774, "kind": "hint", "text": "so um what value guides your life what values do you live by"}
{"type": "turn_end_hint_cancel", "t": 9.216}
```

An early, cheap guess that the user's turn is ending, for a voice agent to **start preparing its reply** (LLM, and
maybe TTS) before the real `turn_end` and to **throw that work away** if the user goes on: what LiveKit Agents calls
preemptive generation and Pipecat an eager end of turn.

| field | type | meaning |
|---|---|---|
| `t` | number | decision time, s: the end of the ASR chunk holding the deciding frame (the head path's clock, [§3](#3-time-semantics)) |
| `p` | number in [0, 1] | the turn head's probability on that frame |
| `kind` | string | always `hint` |
| `text` | string | the current segment's text through the ASR frames available at `t`: what the `final` would hold if the turn were cut at `t` |

Rules, per session (audioforge/server/turn_hint.py):

- A hint is sent on the first frame with at least 80 ms of served-VAD silence (VAD < 0.4, the `vad_head` silence)
  and turn-head p >= `--turn-hint-p`, provided the user spoke after the last `turn_end` and no hint is outstanding:
  at most one per user turn. The shipped `vad_head` rule needs 160 ms of silence and p >= 0.99, or 640 ms of silence.
- It is **confirmed** by the next `turn_end`, which carries its `t` as `hinted_at`, or **withdrawn** by a
  `turn_end_hint_cancel` (`t` = the decision time of the frame that showed it) when the user resumes: served VAD >
  0.5 on 2 frames in a row. After a cancel the next silence can hint again.
- A hint that would come at the same decision time as the `turn_end` is not sent (the `turn_end` already says it).
- Each outcome is logged (`[turn_hint] confirmed ...` / `cancelled ...`) and counted in `stats.turn_hints`.

Client contract: do not **speak** on a hint. Prepare, then release the prepared reply at the `turn_end` if its
`hinted_at` matches (and the `final` text equals the hint's `text`), discard it on `turn_end_hint_cancel`. False
interruptions are then exactly those of `turn_end`; a cancelled hint costs one discarded LLM call. The Pipecat
adapter (`AudioforgeSTTService(turn_hints=True)` + `AudioforgeEagerTurnStopStrategy`) and the LiveKit adapter
(`AudioforgeFrontend(turn_hints=True)`, hint -> `PREFLIGHT_TRANSCRIPT`) do this; both are opt-in.

Measured (scripts/research/turn_hint.py -> runs/turn_hint.json: the server's hint tracker and `vad_head` replayed on
the stored per-frame dumps of research/EOT_LATENCY.md, decision clock, compute excluded; calls = 109 TurnBench +
one-to-one user turns, AMI = 200 dev turns; `turn_end` unchanged: 20.2 % / 10.5 % false interruptions, 7.3 % /
33.5 % missed). Hint latency = the confirmed hint of each hinted end minus the reference end; precision = hints
inside a reference turn window that came at or after its end (the user did not resume in that turn); recall = ends
whose answering `turn_end` carried such a hint; response = when the reply could start if the LLM + TTS need `prep`
ms: `max(hint + prep, turn_end)` vs `turn_end + prep`, minus the end (p50 / p95, ms):

| H | corpus | hint p50 / p95 | precision | confirmed | recall | cancelled / turn | prep 300: at turn_end -> at hint | prep 600: at turn_end -> at hint |
|---|---|---|---|---|---|---|---|---|
| 0.7 | calls | 446 / 1377 | 57.6 % | 66.9 % | 68.3 % | 0.46 | 1226 / 2188 -> 962 / 1918 | 1526 / 2488 -> 1152 / 2038 |
| 0.7 | AMI | 656 / 3356 | 74.4 % | 57.3 % | 46.6 % | 0.35 | 1596 / 4028 -> 1516 / 4004 | 1896 / 4328 -> 1816 / 4200 |
| **0.8** | calls | 446 / 1095 | 58.6 % | 67.1 % | 64.4 % | 0.42 | 1226 / 2188 -> 976 / 1918 | 1526 / 2488 -> 1162 / 2198 |
| **0.8** | AMI | 736 / 3376 | 75.7 % | 59.2 % | 44.4 % | 0.31 | 1596 / 4028 -> 1516 / 4004 | 1896 / 4328 -> 1816 / 4200 |
| 0.9 | calls | 446 / 1243 | 57.3 % | 67.2 % | 59.4 % | 0.39 | 1226 / 2188 -> 1026 / 2130 | 1526 / 2488 -> 1188 / 2328 |
| 0.9 | AMI | 856 / 3376 | 75.8 % | 62.1 % | 39.1 % | 0.25 | 1596 / 4028 -> 1516 / 4004 | 1896 / 4328 -> 1816 / 4200 |
| 0.95 | calls | 456 / 1227 | 58.6 % | 69.4 % | 49.5 % | 0.31 | 1226 / 2188 -> 1032 / 2132 | 1526 / 2488 -> 1256 / 2430 |
| 0.95 | AMI | 816 / 3456 | 80.6 % | 66.0 % | 29.3 % | 0.18 | 1596 / 4028 -> 1596 / 4028 | 1896 / 4328 -> 1896 / 4328 |

Mean saving at H 0.8: 178 / 284 ms (calls, prep 300 / 600), 120 / 181 ms (AMI). Live on the bundled clip
(examples/audio/two_party_call_16s.wav, user turn ends 10.8 s, single mode, real time): hints at 7.78 and 8.74 s
(pauses inside the turn) were cancelled at 8.10 / 9.22 s; the hint at 11.616 s (p 0.878, text equal to the final)
arrived 34 ms after its `t` and was confirmed by the `turn_end` at 11.776 s (`hinted_at` 11.616), a 160 ms lead.
Through examples/pipecat_local_demo.py with a mock LLM (`--llm-ms`), the reply started 1342 -> 1173 ms (300 ms prep)
and 1642 -> 1477 ms (600 ms prep) after the true end with `--turn-hints`, no reply before the `turn_end`, two
speculative inferences discarded.

## 6. Keys added by `--debug-fields`

With `--debug-fields` the server adds these diagnostic keys (serve.py `DEBUG_KEYS`) and validates every outgoing
message. They are not part of the stable protocol.

| message | key | meaning |
|---|---|---|
| `frame` | `spk_t` | end time (s) of the diarizer column carried in `speakers` / `primary`; `null` before the first column |
| `turn_end` | `frame_t` | end time (s) of the frame that fired (compare with `t`, the decision time) |
| `ready` | `diar_lag_ms` | [min, max, mean] structural diarizer lag after the frame end, ms |
| `ready` | `turn_input` | `session` or `diar` (the resolved `--turn-input`) |
| `ready` | `threads` | `--threads` |
| `ready` | `enroll_rss_mb`, `enroll_stride` | RSS added by loading TitaNet-L, and `--enroll-stride` (with `--enroll` other than `dominant`) |
| `stats` | `audio_s`, `proc_s`, `n_chunks` | audio seconds, processing seconds, number of 160 ms chunks |
| `stats` | `asr_ms_p50`, `asr_ms_p95` | per processed block: ASR front end (plus lookahead pass) time, ms |
| `stats` | `diar_ms_p50`, `diar_ms_p95` | per processed block: diarizer time, ms |
| `stats` | `turn_ms_p50` | mean turn-head time per ASR frame, ms |
| `stats` | `backlog_ms_max`, `backlog_ms_end` | queued unprocessed audio when a block was taken, max and last, ms |
| `stats` | `send_lag_ms_p50`, `send_lag_ms_p95`, `send_lag_ms_max` | wall ms from a block's last sample arriving to its messages being ready |
| `stats` | `diar_lag_ms_mean_measured` | mean audio-time lag of diarizer columns. Computed from sample counts, so at 1x it always reads the structural mean; it is not a wall-clock measurement ([INTEGRATION.md §5 D8](../research/archive/INTEGRATION.md#5-defects-found-by-the-verifier-and-their-status)) |
| `stats` | `turn_input` | as in `ready` |
| `stats` | `enroll_ms_p50`, `enroll_ms_max`, `enroll_ms_mean`, `enroll_n_embed` | binder time per diarizer frame and number of TitaNet embeddings (with `--enroll` other than `dominant`) |
| `stats` | `silero_ms_p50`, `silero_ms_p95`, `silero_ms_mean`, `silero_chunks` | Silero VAD time per 32 ms chunk and chunk count (`hybrid_silero` / `hybrid_dyn` sessions) |
| `stats` | `lid_ms_mean` | LID time per ASR frame (`--lid`) |
| `stats` | `lookahead_ms_mean` | lookahead-pass time per 160 ms chunk (`--asr-lookahead`) |

## 7. Message order

Within one processed block of audio the server sends, in this order:

1. `error` notices raised while processing the block (degradations; non-fatal);
2. `frame` messages for the ASR frames completed by the block (or one `frames` batch under load shedding level 2);
3. `enrolled` messages;
4. `language` messages;
5. for each turn event in decision-time order: `turn_end`, then its `final` when the policy cuts (with
   `--final-asr` / `--asr-lookahead` that `final` has `source: "stream"`; with `--final-chunk-ms` it is a
   `final_fast`, and the `source: "slow"` finals follow as a separate send right after the block), interleaved in the same order with
   `turn_end_hint` / `turn_end_hint_cancel` (a `turn_end` before a hint of the same decision time);
6. a `segment_cap` `error` and a `final` without `turn_end`, if the segment has been open for 300 s;
7. `final` messages with `source: "lookahead"` whose frames are now decoded;
8. `partial`, if the segment text changed (not under load shedding level 2).

`error` messages about client messages (`bad_json`, `bad_config`, ...) are sent as soon as the message is read, in
between blocks.

At `end`: the flush block as above, then the end-of-stream `final` (and its lookahead final), then with `--final-asr`
the outstanding offline finals, then `stats`. Offline (`tdt_v3`) finals are otherwise delivered asynchronously
whenever the worker finishes, interleaved with the stream. The `frames` batch form is used only under load
shedding level 2.

## 8. Minimal Python client

Requires `pip install websockets` (the server uses the `websockets.asyncio` API). It streams a 16 kHz mono 16-bit WAV
at real time in 20 ms frames and prints every message except `frame`.

```python
import asyncio
import json
import sys
import wave

import websockets  # >= 13 (websockets.asyncio)
from websockets.asyncio.client import connect

URL = "ws://127.0.0.1:8765"
FRAME_S = 0.02  # 20 ms per binary message


async def main(path: str) -> None:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need mono 16-bit PCM"
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())  # little-endian int16 bytes
    step = int(rate * FRAME_S) * 2  # bytes per 20 ms

    async with connect(URL, max_size=2 ** 22, ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        print("ready", ready)
        await ws.send(json.dumps({"type": "config", "turn_policy": "timeout",
                                  "timeout_ms": 1000, "sample_rate": rate}))

        async def receive() -> None:
            async for raw in ws:  # ends when the server closes after "stats"
                msg = json.loads(raw)
                if msg["type"] != "frame":
                    print(msg)

        receiver = asyncio.create_task(receive())
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for i in range(0, len(pcm), step):
            await ws.send(pcm[i:i + step])
            delay = t0 + (i // step + 1) * FRAME_S - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)  # real-time pacing; drop it to stream as fast as possible
        await ws.send(json.dumps({"type": "end"}))
        await receiver


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
```

`scripts/stream_client.py` is a fuller client (event log with wall-clock arrival times, lag summary, optional WER
against a reference). `audioforge/integrations/pipecat.py` and `audioforge/integrations/livekit.py` are framework
adapters; both send `config` right after `ready` and ignore unknown `ready` keys.

## 9. Example session transcript (illustrative)

The transcript below is **illustrative**: it was constructed by hand from the message shapes in `serve.py` and the
tests (`tests/test_serve.py`, `tests/test_serve_shipped.py`), not recorded from a run. The numbers are plausible, not
measured. Setup: default server flags (`--diar-config low_latency_032`, `--enroll dominant`, no `--final-asr`), a
3.2 s clip with one speaker saying "hello there" between about 0.4 s and 1.3 s. Most `frame` messages are omitted
(one is sent every 80 ms: 40 in total here).

Client to server:

```json
{"type": "config", "turn_policy": "timeout", "timeout_ms": 1000, "sample_rate": 16000}
```
followed by 160 binary frames of 640 bytes (20 ms each), then:
```json
{"type": "end"}
```

Server to client, in order:

```json
{"type": "ready", "model": "stage1_served+nemo_sortformer_v2", "chunk_ms": 160, "frame_ms": 80, "diar_config": "low_latency_032", "column_lag_ms": 160.0}
{"type": "frame", "t": 0.08, "vad": 0.0213, "eot": null, "speakers": [0.0, 0.0, 0.0, 0.0], "primary": null}
{"type": "frame", "t": 0.16, "vad": 0.0188, "eot": null, "speakers": [0.0, 0.0, 0.0, 0.0], "primary": null}
{"type": "frame", "t": 0.64, "vad": 0.9731, "eot": 0.01822, "speakers": [0.9104, 0.0021, 0.0005, 0.0003], "primary": 0}
{"type": "partial", "t": 0.8, "text": "hello"}
{"type": "partial", "t": 1.28, "text": "hello there"}
{"type": "frame", "t": 2.4, "vad": 0.0102, "eot": 0.93517, "speakers": [0.0154, 0.0011, 0.0004, 0.0002], "primary": 0}
{"type": "turn_end", "t": 2.48, "policy": "timeout", "p": null, "silence_ms": 1040}
{"type": "final", "t": 2.48, "text": "hello there", "speaker": 0}
{"type": "frame", "t": 3.2, "vad": 0.0097, "eot": 0.97104, "speakers": [0.0101, 0.0009, 0.0003, 0.0002], "primary": 0}
{"type": "final", "t": 3.2, "text": "", "speaker": 0}
{"type": "stats", "rtf": 0.4912, "chunk_ms_p50": 81.3, "chunk_ms_p95": 152.7, "first_partial_ms": 63.4, "peak_rss_mb": 3563.2}
```

Then the server closes the connection. How the `turn_end` time follows from the rules: the primary column 0 is last
active on frame 15 (1.20-1.28 s); 13 silent frames (1040 ms >= `timeout_ms` 1000) are complete at frame 28; with
C 3 / R 1 that column is finalized at ((28 // 3 + 1) * 3 + 1) * 0.08 = 2.48 s, which is `turn_end.t`.

With `--final-asr tdt_v3` the same session would add `"source": "stream"` to both finals, `"final_asr": "tdt_v3"` to
`ready`, and a later message such as

```json
{"type": "final", "t": 2.48, "text": "Hello there.", "speaker": 0, "source": "tdt_v3", "start": 0.1, "end": 1.78, "latency_ms": 412.5}
```

plus `final_asr`, `final_latency_ms` and `final_asr_rss_mb` in `stats`. With `--enroll after_agent_arm` and an
`agent_end` sent at 0.2 s, an `{"type": "enrolled", "t": 0.8, "column": 0}` message would appear and `ready` /
`stats` would carry `enroll`, `enrolled`, `primary_column`. (All values in this section are illustrative.)

## 10. Errors, limits and degraded operation

Every `error` message has a `code` from this list (serve.py `ERROR_CODES`). The code decides whether the session
continues.

| code | fatal | cause |
|---|---|---|
| `bad_json` | no | a text frame is not valid JSON |
| `bad_message` | no | a JSON text frame is not an object with a string `type` |
| `unknown_type` | no | a text frame's `type` is not `config`, `end`, `agent_end` or `enroll` |
| `bad_config` | no | a `config` field was ignored or clamped (the detail lists them), or a Silero policy was requested after the first audio |
| `message_too_large` | no | a text frame over 64 KiB (ignored) |
| `unsupported` | no | `agent_end` / `enroll` that is not the server's `--enroll` trigger |
| `silero_unavailable` | no | `hybrid_silero` / `hybrid_dyn` requested but the Silero ONNX is missing or failed to load: the session runs `hybrid` (head OR diarizer-primary timeout) with the Silero policy's head threshold |
| `final_asr_failed` | no | an offline final pass failed; that turn's offline final carries the streaming text |
| `lookahead_dropped` | no | the `--asr-lookahead` pass was dropped (load or error); lookahead finals carry the streaming text from then on |
| `nan_input` | no | non-finite input samples were replaced by 0 |
| `clipped_input` | no | input samples beyond ±64 (after decoding) were clipped |
| `nan_state_reset` | no | a model produced non-finite outputs; the values were zeroed and, for the ASR heads, the streaming state was reset |
| `overloaded` | no | load shedding started or rose (at most one notice per 5 s per session); see below |
| `segment_cap` | no | a segment was open for 300 s without a `turn_end`; it was cut with a `final` (no `turn_end`) |
| `session_limit` | yes | the session reached `--max-session-s` of audio; the server then flushes (end-of-stream `final`, `stats`) and closes with code 1008 |
| `idle_timeout` | yes | no message from the client for `--idle-timeout-s`; closed with code 1008 |
| `processing_failed` | yes | the models raised an exception on this session's audio; closed with code 1011 |
| `internal_error` | yes | any other server-side exception; closed with code 1011 |
| `server_shutdown` | yes | the server is stopping; closed with code 1001 |

A normal end (after `stats`) closes with code 1000.

**Load shedding.** When the queued, unprocessed audio of a connection reaches 1500 ms (or the total over all
connections reaches 1500 ms times the number of connections), the server skips the diarizer for the blocks it
processes: level 1. The diarizer columns of those frames are replaced by the served VAD in column 0 and 0 in the
other columns, so `speakers` / `primary` then describe "anyone speaking", not a speaker. At 4000 ms it also drops
`partial` messages, sends frames as one `frames` batch per block, and drops the `--asr-lookahead` pass for the rest
of the session: level 2. Turn events and streaming finals continue at every level. The skipped work is counted in
`stats.degraded` (`shed_diar_frames`, `shed_partials`, `frames_batched`).

## 11. Health endpoint

`GET http://<host>:<port>/health` (also `/healthz` and `/stats`) returns one JSON object over plain HTTP:

| key | meaning |
|---|---|
| `ok` | `true` while the server answers |
| `model` | as `ready.model` |
| `uptime_s` | seconds since the engine started |
| `sessions_active`, `sessions_total` | live connections, connections since start |
| `backlog_ms_total`, `backlog_ms_max` | queued unprocessed audio over all connections, and the largest per connection, ms |
| `rss_mb`, `peak_rss_mb` | current and peak RSS of the server process, MB |
| `threads` | `--threads` |
| `final_asr` | `null`, or `{"source", "alive", "restarting"}` of the offline final-ASR worker |
| `silero` | `"loaded"`, `"not loaded"`, or the reason it cannot be loaded |
| `limits` | `max_session_s`, `idle_timeout_s`, `max_inbox_s` (30), `shed_diar_ms` (1500), `shed_partial_ms` (4000) |
| `counters` | server-wide counts of every degradation, refused message and error code |

## 12. Known protocol limitations

From the verifier's defect list ([INTEGRATION.md §5](../research/archive/INTEGRATION.md#5-defects-found-by-the-verifier-and-their-status)),
with their state in the current code:

- **D2** Finals are cut at a token index, not a word start, so a word can be split between two finals.
- **D4** `speakers` is `[0, 0, 0, 0]` before the first diarizer column, indistinguishable from "nobody talking".
- **D5** `speakers` / `primary` are older than `t`; the column time `spk_t` is sent only with `--debug-fields`.
- **D6** `stats.peak_rss_mb` is the process lifetime peak, and `first_partial_ms` is server lag only. (An empty
  session now reports `rtf` 0.0.)
- **D7** Rejected or clamped `config` fields are now reported with a `bad_config` error, but the effective
  configuration is still not echoed in `ready`.
- `speakers` columns are the diarizer's arrival-order slots. A column index is not a stable speaker identity across
  long pauses or sessions; use `--enroll` to bind "the user".
- The wire protocol carries exactly 4 speaker columns (`validate()` rejects any other count). An 8-column diarizer
  (Nemotron-3-Diarization) must be served with `--diar-spks 4`.
