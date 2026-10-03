# Server internals

Design notes for the WebSocket server: the frame clock and diarizer lag, how each turn policy decides,
primary-speaker binding, the turn head's inputs, language ID, the hybrid final ASR and the dual-lookahead pass. This
was the module docstring of `audioforge/serve.py` until the server was split (2026-09-29); the text is unchanged.
Users need [`CONFIGURATION.md`](CONFIGURATION.md) (flags) and [`PROTOCOL.md`](PROTOCOL.md) (messages); the one-page
overview is [`ARCHITECTURE.md`](ARCHITECTURE.md).

Module map: `audioforge/serve.py` holds `Engine` (the loaded models and the state shared by every connection:
diarizer factory, Silero, the final-ASR worker and its restarts, load shedding, health), `Session` (one connection's
streaming state: `process(samples)` -> messages, `finish()`) and the WebSocket loop (`handle`, `serve`, `/health`).
It is built from `audioforge/server/`: `constants.py` (clock, policy constants, limits), `protocol.py` (schema,
`validate`, `SessionConfig`, PCM decoding), `policies.py` (turn policies), `binding.py` (primary-speaker binding),
`streams.py` (ASR passes, CPU fast path, resampler), `cli.py` (flags), `util.py`.

Two models run side by side on one 80 ms frame clock:

* **ASR front end** (`--asr`): frozen NVIDIA cache-aware FastConformer at [70, 1] = 160 ms chunks, driven by
  `model.StreamingSession` (`ASRStream` below). Per 160 ms chunk the encoder runs once (`stream_step`);
  its output feeds the RNNT greedy decoder (frame by frame, so the text state at frame v holds exactly the
  tokens emitted at frames <= v), the VAD head (layer mix of the same chunk's per-layer outputs) and the turn
  head's GRU (`TurnHead.step`). `StreamingRuntime` is not used: its fixed-shape encoder implements RoPE
  only, not the rel_pos / xscaling / NeMo-aligned subsampling of the imported NVIDIA encoder.
* **Diarizer** (`--diar`): NVIDIA Streaming Sortformer v2, its own 128-mel front end and non-causal encoder,
  `streaming_diar.StreamingDiarizer` in window mode with a model-card setting chosen by `--diar-config`
  (`AOSCConfig.preset` / `streaming_diar.SORTFORMER_PRESETS`; FIFO 188, update 144, cache 188, encoder left
  context 188 frames in both):
  - `low_latency_032` (default since 2026-09-26): card "ultra low latency", chunk 3 + right context 1 = 0.32 s.
    research/archive/EOT_BENCH_V2.md section 7 (a selection experiment on n = 974 AMI dev turns, leak-free cross-fit,
    not a test result): the same miss rates as the 1.04 s setting for the hybrid, head and timeout rules (CIs overlap)
    with 0.5-0.8 s lower P50 wherever the systems fire (nominal emission delay 240 vs 840 ms), at DER 28.3 vs 26.3.
  - `low_latency`: card "low latency", chunk 6 + right context 7 = 1.04 s, as scored in research/archive/STAGE1.md
    (n=200) and research/archive/SORTFORMER_IMPORT.md.
  Its 4 columns are the speakers. Measured on this Mac (CPU, 2 threads, 1x, 2 AMI dev windows; research/EARLY_RESULTS.md "Live
  streaming server"): RTF 0.49-0.53 (0.32 s) vs 0.40-0.43 (1.04 s), both keep up (max backlog 120-140 ms);
  speaker-column age on arrival p50 210-227 ms vs 857 ms.

Frame clock and lag. Both models emit `n_frames(len(audio))` frames; frame v covers [80 v, 80 (v+1)) ms and the
protocol's `t` is its end, (v+1) * 0.08 s. The two ASR frames of a chunk are available 6 ms after the chunk's
audio (the STFT's half window), i.e. 6-86 ms after each frame ends. The diarizer finalizes frame t only when its
C-frame chunk plus R right-context frames have arrived: at audio time ((t // C + 1) * C + R) * 80 ms, i.e.
**80-240 ms (mean 160 ms) after the frame ends for low_latency_032 (C 3, R 1), 560-960 ms (mean 760 ms) for
low_latency (C 6, R 7), plus the diarizer step's compute time** (`diar_lag_ms`; the mean is the ready message's
`column_lag_ms`). A "frame" message is sent when the ASR frame is ready and carries the most recent diarizer
column available at that moment (so `speakers` / `primary` describe audio `column_lag_ms` + compute older
than `t`); the timeout policy runs on the diarizer's own frames as they are finalized.

Event times. `turn_end.t` (and its `final.t`) is the *decision time*: the audio time at which the data behind
the decision was complete (timeout: the deciding diarizer column's finalization time; head: its encoder chunk's,
or the later of that and the diarizer column's with `--turn-input diar`). The final at a turn_end holds exactly
the tokens of the ASR frames available at that time; later tokens open the next segment. Events and finals are
therefore independent of how the client batches its audio. `partial.t` = end of the latest ASR frame.

Turn policy. **The default policy is the plain silence timeout** on the diarizer's label-free primary track; the
head's probability is exposed as the `eot` field and as an optional policy.

* primary = the diarizer column with the most activity (sum of probabilities) over the last 5 s; `null` when no
  column was active (p > 0.5) in that window. No oracle, no enrollment.
* `timeout` (default): turn_end when the primary column has been inactive (p <= 0.5) for >= timeout_ms and the
  primary spoke since the last timeout turn_end. Other columns are ignored (someone else talking does not delay it).
* `timeout_quiet`: the same, but it additionally waits until no other column is active on that frame (the
  previous default; STAGE1's "primary silent AND nobody else" rule). Its turn_end is tagged `timeout`.
* `head`: turn_end when the turn head's probability crosses eot_threshold upwards; re-armed by speech (VAD > 0.5).
* `both`: plain timeout and head both fire, each tagged with its policy. Finals are cut by the timeout policy for
  `timeout`/`timeout_quiet`/`both`, by the head for `head` and by the hybrid event for `hybrid`.
* `hybrid`: plain timeout (timeout_ms) OR head >= eot_threshold: one turn_end per turn, tagged `hybrid`, at the
  earlier of the two paths' decision times (the later path's firing for the same turn is dropped); `p` = the head's
  probability at that time, `silence_ms` = the firing path's silence (timeout: diarizer primary; head: VAD).
  Measured (research/archive/TURN_ERRORS.md section 8, runs/turn_v3_hybrid_n200.json; 200 AMI dev turns, turn head v3,
  joint (θ, k) sweep, <= 5 % per-turn FC; oracle-overlap enrollment unless noted): oracle primary activity
  1.6 % miss at P50 560 ms (θ 0.947, k 24 frames = 1920 ms) vs head 5.8 % / 560 ms, timeout 6.3 % / 1440 ms;
  Sortformer streaming track 37.4 % / 2640 ms (θ 0.997, k 21) vs timeout 38.4 % / 2640 ms, head 62.6 % / inf;
  label-free causal_dominant enrollment 86.8 % / inf (k 26) vs timeout 88.4 %. On the streaming track the head adds
  about 1 point, so `timeout` stays the default; `hybrid` pays off only with a clean speaker track. In config
  terms k frames = timeout_ms 80 k (k 21 -> 1680). The head there is fed the diarizer track (`--turn-input diar`).
* `hybrid_silero` / `hybrid_dyn` (added 2026-09-26; research/archive/BASELINES.md "ICSI held-out confirmation" and "Dynamic
  timeout: ICSI held-out confirmation"; measured live in research/archive/INTEGRATION.md section 8): the head path as
  `hybrid` (threshold 0.99828 / 0.998283 unless the config sets `eot_threshold`) OR an **any-speaker Silero VAD v5
  silence** in place of the diarizer-primary timeout: Silero on 32 ms chunks -> Pipecat's VAD state machine
  (confidence 0.7, start / stop 0.2 s; `baselines.turn.PipecatVADState`) -> silence since the end of the last
  speech chunk on the 80 ms grid (`SileroSilence`, = `baselines.turn.silence_frames`). `hybrid_silero` fires at
  silence >= `--silero-timeout-ms` (2640 = the AMI-fitted k 33); `hybrid_dyn` (the primary shipping candidate)
  fires at silence - clamp(75 - 55 p, 2, 75) > 4.957 frames, i.e. silence >= clamp(80 - 55 p, 7, 80) frames (6.4 s
  at p = 0, 2.0 s at p = 1), p = the head's posterior on the same frame (decided when both are ready). One firing
  per silence run (re-armed by a Silero speech chunk); one turn_end per turn as `hybrid`, tagged with the policy.
  `timeout_ms` is not used by them. The Silero ONNX (`--silero`, default data/silero/silero_vad_v5.onnx, the
  benchmark's file) is loaded only when a session asks for such a policy (or at start when `--silero` is given):
  one shared ORT session (+~10 MB), ~0.1 ms per 32 ms chunk on this Mac (RTF +0.003). Set the policy in the first
  config message (the Silero chunk grid starts with the stream).

Why the plain timeout (research/archive/STAGE1.md, n=200 AMI dev turns, misses at <= 5 % false cutoffs;
research/archive/INTEGRATION_VERIFY.md D1): the plain silence timeout on the Sortformer streaming primary misses 38.4 %,
while "primary silent AND nobody else active" misses 66-68 %, no better than the served head (69 %). The previous
server implemented the second rule while citing the first rule's number. Caveat: STAGE1's 38.4 % picks the column
with the oracle primary's activity; with deployable, label-free enrollment (as served here) eot-bench v2
(research/archive/EOT_BENCH_V2.md) finds every streaming system's misses substantially higher, the timeout's included.
The head-vs-timeout ranking for the exact served rule (label-free 5 s primary) is the eot-bench v2 result, not
STAGE1's.

Primary-speaker enrollment (`--enroll`, default `dominant` = the 5 s dominant column above, unchanged).
research/archive/EOT_BENCH_V2.md section 9: the loss of the label-free binding is *which* speaker is the user, not how the
column is followed, and a voice agent has one identity signal the benchmark's rules lack: it knows when it stopped
speaking. Two opt-in modes bind the primary by voice with the ported NVIDIA TitaNet-L (`enrollment.TitaNetEmbedder`,
loaded only then, +~200 MB RSS; `--titanet` = its .nemo):
* `after_agent`: the client sends {"type": "agent_end"} when its TTS finished. From that audio time the first
  diarizer column active for >= 3 consecutive frames is the user; its first 19 active frames (1.5 s) are the
  enrollment; from then on the primary is the column whose last 2 s of active speech sounds most like the enrollment
  (`enrollment.VoiceFollower` hysteresis: margin 0.1 cosine, hold 6 frames), re-embedded every 5 frames
  (`--enroll-stride`, 400 ms; one TitaNet pass per active column, ~50-150 ms each on 2 threads). Every agent_end
  re-arms (a new enrollment for the next user turn).
* `explicit`: the same, started by {"type": "enroll"} (a "say something" prompt); one enrollment per message.
* `after_agent_arm` (research/archive/EOT_BENCH_V2.md section 9 "after_prev_end_causal"; no TitaNet, no embedding cost):
  on agent_end the first column active for >= 3 consecutive frames is chosen; from then on the primary follows the
  causal_dominant rule seeded at that column (`CausalDominant`: re-binds only after > 25 silent frames of the bound
  column, to the column with the most active frames over the last 25). Each agent_end re-arms; the binding holds
  until the next choice. `enrolled` = a column has been chosen since the last agent_end.
Until a column is chosen the primary is the dominant column (the default rule). The chosen column is reported in
`frame.primary` like any primary; a `{"type": "enrolled", "t", "column"}` message marks the enrollment; `ready`
and `stats` carry `enroll` (mode), `enrolled` and `primary_column` when the option is on (absent otherwise, so
the default protocol is unchanged). The benchmark's numbers for these rules (AMI dev, agent turn = the previous other
speaker's turn) are in EOT_BENCH_V2.md section 9.

Turn head input (`--turn-input`). The stage-1 head is `mode: kernel` (condition_on_speaker): it was trained on
the encoder *steered by the primary's activity* through speaker kernels, which needs the diarizer's column
before the encoder pass. `session` (default for kernel heads): the head reads the session's own encoder frames
(no second encoder pass) - no speaker information reaches it, which is off its training distribution.
`diar`: the head runs on the diarizer's clock, frame v once column v exists; for kernel heads a second,
speaker-conditioned cache-aware encoder stream is run over the same mel chunks (the StreamingSpeakerASR
pattern: one extra encoder pass per chunk); concat / v3 heads get spk_act / cols / prim from the diarizer.

`tsvad` (research/archive/IMPROVEMENTS.md section 1): the turn head is fed the enrolled user's activity from the TS-VAD head
(`--tsvad`, default runs/tsvad_spk.pt; research/IMPROVE_115M.md Part A) on block 4 of the ASR pass, [P(target),
P(other)] as columns 0 / 1 (primary 0), on the ASR chunk clock (decision time = the chunk's, no diarizer lag); the
speaker-conditioned second encoder pass is as with `diar`. The voice print (`audioforge.tsvad_stream`): with
`--enroll explicit` a stored print {"type": "enroll", "embedding": [192 floats]} (`tsvad_stream.voiceprint`) or,
without an embedding, the next `--tsvad-print-s` s of served-VAD speech; `after_agent_arm`: the first
`--tsvad-print-s` s of speech after each agent_end; `dominant`: the session's first speech. Before a print the head
is a plain VAD (its no-enrolment vector). Each new print sends {"type": "voiceprint", "t", "seconds", "source"}.
`hybrid_dyn` uses TSVAD_DYN's (theta, offset) unless the config sets eot_threshold. The diarizer still runs (frame
speakers / primary, the timeout policies) unless `--diar-off`: then the session's columns are [P(target), P(other),
0, 0] (finalized with their ASR chunk) and no diarizer pass runs; TitaNet is never loaded in this mode.

Spoken language ID (`--lid`, off by default; research/archive/LID.md). A LID head file (`audioforge.lid.save_head`,
e.g. runs/lid_head.pt: heads.audio.LanguageHead trained head-only on the frozen served encoder) is attached to the ASR
model at load. Per 160 ms chunk it reads the same chunk's per-layer encoder outputs (no extra encoder pass) and
updates a running attentive-stats posterior over its languages (`audioforge.lid.LIDStream`): frames with VAD <= 0.5
are not pooled, and the pooled sums decay with a 30 s half-life of speech so a long session can follow a language
change. A `{"type": "language", "t", "language", "confidence"}` message is sent when the top language's posterior
first reaches `--lid-threshold` after at least `--lid-min-ms` of pooled speech, and again whenever the confident
top language changes (`t` = end of the frame that decided it); `stats` then carries `lang` (the last announced
language or null). Without `--lid` no message or key changes. `--lid ambernet` (or a langid_ambernet .nemo path)
uses the dedicated NVIDIA AmberNet instead (`audioforge.lid.AmberNetLIDStream`, +29 M parameters): the audio of the
VAD-speech frames is kept and AmberNet re-classifies the last 8 s of speech at 1, 1.5, 2, 3, 5 and 8 s of pooled
speech and then every 4 s (16-81 ms per call on 2 threads for 1-8 s of speech, more under load: too much for every
chunk), restricted to `--lid-langs`; same message and rule. research/archive/LID.md measures both (AmberNet is the more accurate backend there).

Hybrid final ASR (`--final-asr tdt_v3`, off by default; research/archive/HYBRID_ASR.md). The streaming model keeps every
live decision (partials, VAD, turn_end, speakers, enrollment); a second, offline model (NVIDIA Parakeet-TDT 0.6B v3,
`audioforge.final_asr.FinalASRWorker`, loaded only with the flag, by default in its own process with its own torch
threads so it never runs on the streaming worker or holds its GIL) transcribes each finished turn once. At every
cutting turn_end (and at stream end) the server takes the turn's audio: from the first frame of the segment whose
served VAD > 0.5 (the speech onset) minus `FINAL_PRE_ROLL_S` = 0.3 s (never before the previous turn's end), to
the decision time, trimmed to `FINAL_POST_ROLL_S` = 0.5 s after the last VAD speech frame (at most
`FINAL_MAX_TURN_S` = 60 s of audio is kept). The streaming final is sent as before plus `"source": "stream"`;
when the offline pass finishes, a second final follows with `"source": "tdt_v3"`, the same `t` (decision time),
`speaker`, the transcribed span `start` / `end` (s) and `latency_ms` (wall time from the job's submission,
right after the turn_end is computed, to this message). A segment without VAD speech gets an empty tdt_v3 final at
once (no pass). The stats message waits for pending passes and carries `final_asr` (the source name),
`final_latency_ms` {p50, p95, max, n} and `final_asr_rss_mb` (worker peak RSS; null in thread mode); `ready`
carries `final_asr`. Without the flag no message or key changes (`validate`).

Dual lookahead (`--asr-lookahead R`, e.g. 13; off by default; research/archive/HYBRID_ASR.md). A second, text-only
cache-aware streaming pass of the *same* ASR model at att_context [70, R] (R = 13: 1.12 s chunks, 1.04 s lookahead;
`LookaheadStream`) runs next to the served [70, 1] pass inside `Session.process`; every head (VAD, turn, speakers,
LID) and the partials still come from the [70, 1] pass, unchanged. At each cutting turn_end the segment's lookahead
final is due once the lookahead pass has decoded `LOOKAHEAD_MARGIN_FRAMES` = 3 frames past the segment's last
served-VAD speech frame (capped at the streaming final's cut frame): it holds the lookahead tokens of frames
[previous lookahead cut, that frame) and is sent as `"source": "lookahead"` with the turn's `t`, `start` /
`end` (s, its frame span) and `latency_ms` = the *audio-time* wait past the decision time (0 when the rule
already waited that long, as a >= 1 s silence timeout does). Keys as for `--final-asr` (both flags may be combined:
`ready.final_asr` lists the sources, `stats.final_latency_ms` is keyed by source).

Protocol (fixed). Client -> server: binary = int16 LE PCM mono (16 kHz unless configured), any frame size; text
{"type":"end"} flushes; optional first text {"type":"config","turn_policy":"timeout"|"timeout_quiet"|"head"|"both"|"hybrid",
"timeout_ms":1000,"eot_threshold":0.98,"sample_rate":16000} (turn_policy also "hybrid_silero"|"hybrid_dyn");
with --enroll: {"type":"agent_end"} (after_agent, after_agent_arm) /
{"type":"enroll"} (explicit) arm the voice enrollment at the audio position received so far. Server -> client JSON text frames:
ready (model, chunk_ms, frame_ms, diar_config = the preset name, "+custom" with --diar-set overrides,
column_lag_ms = mean diarizer finalization lag after the frame end, compute excluded; the last two added
2026-09-26), frame (one per 80 ms frame), partial (on transcript change; text since the last final), turn_end, final
(at every turn_end of the cutting policy and at end), enrolled (--enroll only), stats (after the end final); the
server then closes.
CPU fast path: both encoders' conformer convolutions are re-bound to an equivalent unfold / F.linear forward
(`fast_conv`; identical tokens, diarizer probabilities within 3e-7), ~3x on the ASR and ~2x on the diarizer on
this Mac; `--no-fast-conv` disables it. `--debug-fields` adds diagnostics (frame: spk_t; turn_end: frame_t; stats: backlog/lag/cost breakdown);
without it every message has exactly the protocol's keys (`validate`).
