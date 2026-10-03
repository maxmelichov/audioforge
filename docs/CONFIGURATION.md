# audioforge server configuration

How to configure `audioforge-serve`: how to set options, the flag reference, the presets, the turn policies and the
performance options. The wire protocol is in [PROTOCOL.md](PROTOCOL.md). Per-session options (`turn_policy`,
`timeout_ms`, `eot_threshold`, `sample_rate`) are not flags. The client sends them in its first `config` message.

Numbers on this page give their scope (dataset, n, setting). Where nothing was measured, the page says so. Most
measurements ran on one shared Apple-silicon Mac, CPU only, at 2 torch threads. Treat compute numbers as
order-of-magnitude figures for other machines. Full results are in [RESULTS.md](RESULTS.md).

## Contents

1. [Launching](#1-launching)
2. [Flag reference](#2-flag-reference)
3. [Models, threads and speed](#3-models-threads-and-speed)
4. [Turn policies (`config.turn_policy`)](#4-turn-policies-configturn_policy)
5. [Turn head input (`--turn-input`)](#5-turn-head-input---turn-input)
6. [Primary-speaker enrollment (`--enroll`)](#6-primary-speaker-enrollment---enroll)
7. [Diarizer](#7-diarizer)
8. [Final ASR and dual lookahead](#8-final-asr-and-dual-lookahead)
9. [Spoken language ID (`--lid`)](#9-spoken-language-id---lid)
10. [Diagnostics](#10-diagnostics)
11. [Presets](#11-presets)
12. [Config file and environment](#12-config-file-and-environment)
13. [Single-model mode (`--mode single`)](#13-single-model-mode---mode-single)

Turn-taking terms used below:

- **EOT latency:** time from the true end of the user's turn to the decision reaching the agent framework (Pipecat
  or LiveKit).
- **False interruption:** a decision while the user keeps talking.
- **Missed:** a turn end with no decision within the horizon.
- **Miss at ≤ 5 % false cutoffs (FC):** an offline benchmark score. The threshold is fixed so that at most 5 % of
  turns get a decision before their true end. Lower is better. **Floor-open** ends are those where nobody else speaks
  within 1.04 s, i.e. where an agent should reply.

## 1. Launching

```bash
audioforge-download                 # fetch + verify the served ASR, the TS-VAD and LID heads
audioforge-serve                    # single-model mode (the default, §13): 127.0.0.1:8765, 2 threads
audioforge-download --diarizer nemotron3 && audioforge-serve --mode room   # room mode: + NVIDIA Nemotron-3-Diarization
```

There are two modes:

- **`single`, the default.** Everything runs from the one 115M checkpoint, for a known user. The client sends the
  user's stored voice print (≥ 5 s of clean speech, 10 s for meetings) as `{"type": "enroll", "embedding": [...]}`
  ([§13](#13-single-model-mode---mode-single)).
- **`room`.** General diarization with NVIDIA Nemotron-3-Diarization (or Streaming Sortformer v2) next to the 115M
  model, to label everyone in a room. Optionally add `--final-asr tdt_v3`.

Without `--mode`, any of `--diarizer`, `--diar` or `--final-asr` selects room mode. The launcher prints one line when
it does that.

`audioforge-serve` is a thin launcher on top of `python -m audioforge.serve`. It adds `--models-dir`, `--mode` and
`--diarizer`. `audioforge-serve --help` lists the everyday flags, `--help-advanced` all of them. Any flag can also
come from a config file ([§12](#12-config-file-and-environment)).

In room mode the launcher adds defaults that you can override:

- `--diarizer nemotron3` adds `--diar-pool max --diar-left 1` ([§7.3](#73-nemotron-3-diarization-as-the-diarizer)).
- Either diarizer adds `--shed-diar hold` ([§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)).

When `--asr` / `--diar` are not given, the launcher fills them from the models directory: `stage1_served.afm`, and
`nemo_nemotron3_diar.afm` if present, else `nemo_sortformer_v2.afm`. It also fills in the optional model files that
some flags need: TitaNet for `--enroll after_agent|explicit`, AmberNet for `--lid ambernet`, Parakeet-TDT v3 for
`--final-asr tdt_v3`, and Silero (room mode). If a needed model is missing, it exits with the `audioforge-download`
command to run. Every other flag is passed to `audioforge.serve` unchanged.

The module can also run directly, with explicit model paths:

```bash
python -m audioforge.serve --asr models/stage1_served.afm --diar models/nemo_sortformer_v2.afm [flags]
```

**Served model.** `audioforge-download` builds `stage1_served_v4.afm` from `assets/served_heads_v0.4.pt`. It adds a
`speech` detector head (the client's per-frame speech probability) and the turn classifier that `--turn-preset fast`
and `assistant` use. `--heads-version 0.3 | 0.2 | 0.1` rebuilds older head sets. Several measurements on this page
were taken on the v0.1 file (`stage1_served.afm`). Files and heads are described in
[ARCHITECTURE.md](ARCHITECTURE.md#models-and-files).

The default port is 8765 in the server, `scripts/stream_client.py` and both adapters.

## 2. Flag reference

Generated from the flag table in `audioforge/server/cli.py` (`python scripts/dev/gen_config_doc.py`; CI fails when
this block is stale). "advanced" flags are hidden from `--help` and listed by `--help-advanced`. The last column links
the section that explains each group.

<!-- BEGIN GENERATED FLAGS: scripts/dev/gen_config_doc.py writes this block from audioforge/server/cli.py -->

**models**

| flag | default | meaning | more |
|---|---|---|---|
| `--models-dir DIR` | `$AUDIOFORGE_HOME`, else `<repo>/models`, else `~/.cache/audioforge` | where audioforge-download put the models (`audioforge-serve` only) | [§1](#1-launching) |
| `--mode {single,room}` | `single` (`room` when `--diarizer`, `--diar` or `--final-asr` is given) | preset. `single` (the default): everything from the one 115M checkpoint, for a known user: adds `--turn-input tsvad --diar-off --lid head --enroll after_agent_arm --turn-policy vad_head --dyn-wait-ms 2000,960`, loads no diarizer, no final-ASR worker and no Silero (the default turn rule `vad_head` reads the model's own VAD, turn and TS-VAD heads; `hybrid_dyn` needs `--silero`), and refuses `--diar`, `--diarizer`, `--final-asr`, `--lid ambernet`, `--diar-embed titanet`; the voice print comes from an `enroll` message with an embedding (store >= 5 s of clean speech, 10 s for meetings), else live after `agent_end`. `room`: general diarization with NVIDIA Nemotron-3-Diarization (or `--diarizer sortformer`) next to the 115M model ([§7](#7-diarizer)). Flags you pass yourself win (`audioforge-serve` only) | [§13](#13-single-model-mode---mode-single) |
| `--core {115m,0.6b}` | `115m` | which streaming core the launcher loads from the models directory: `115m` (NVIDIA FastConformer 114M + our heads, real time on 2 CPU threads) or `0.6b` (NVIDIA nemotron-speech-streaming-en-0.6b, NVIDIA Open Model License, with every head retrained on it: `served_0p6b_v0.4.afm`, `tsvad_0p6b.pt`, `lid_0p6b_v2.pt`; about half the meeting WER, but ~96 ms of CPU per 160 ms chunk on 2 threads: one real-time stream per process against the 115M's 4, 3 on an Apple GPU with `--device mps`, 5 GB RSS). Voice prints belong to one core: re-enroll after switching. Install with `audioforge-download --core 0.6b`; docs/RESULTS.md has the numbers side by side (`audioforge-serve` only) | [§1](#1-launching) |
| `--diarizer {nemotron3,sortformer}` | room mode: `nemotron3` if downloaded, else `sortformer` | room mode: which downloaded diarizer to pass as `--diar`; `nemotron3` also adds `--diar-pool max --diar-left 1`. For either diarizer the launcher adds `--shed-diar hold`; flags you pass yourself win (`audioforge-serve` only) | [§7.3](#73-nemotron-3-diarization-as-the-diarizer) |
| `--asr PATH` | required (`audioforge-serve`: from the models directory) | ASR + heads `.afm` (`stage1_served_v4.afm` = `stage1_served_v3.afm` + the speech-detector head; v3 = the block-4 VAD build `stage1_served_v2.afm` + the turn head v5 classifier; `stage1_served.afm` = the measured v1) | [§3](#3-models-threads-and-speed) |
| `--diar PATH` | required unless `--diar-off` (`audioforge-serve`: from the models directory; none with `--mode single`) | diarizer `.afm`: the Streaming Sortformer v2 or the Nemotron-3-Diarization import | [§7](#7-diarizer) |

**server**

| flag | default | meaning | more |
|---|---|---|---|
| `--host HOST` | `127.0.0.1` | listen address; 0.0.0.0 for all interfaces |  |
| `--port N` | `8765` | listen port |  |
| `--threads N` | `2` | torch threads of the shared compute worker | [§3](#3-models-threads-and-speed) |
| `--config FILE` | `$AUDIOFORGE_CONFIG`, else none | YAML or JSON file with any of these options by long name; command-line flags win over it | [§12](#12-config-file-and-environment) |

**turn detection and speakers**

| flag | default | meaning | more |
|---|---|---|---|
| `--enroll {dominant,after_agent,after_agent_arm,explicit}` | `dominant` | primary-speaker binding: `dominant` (the 5 s dominant column), `after_agent` (TitaNet voice following from the client's `agent_end`), `after_agent_arm` (the next active column after `agent_end`, no TitaNet), `explicit` (the utterance after an `enroll` message) | [§6](#6-primary-speaker-enrollment---enroll) |
| `--diar-labels {column,registry}` | `column` | what `final.speaker` is: the primary column (legacy) or a stable voice-keyed id from a per-session speaker registry (adds `final.speaker_conf`, `final.diar_shed`, `stats.speakers_seen`) | [§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any) |
| `--turn-input {auto,session,diar,tsvad}` | `auto` | what the turn head reads: the session's encoder frames, a speaker-conditioned pass on the diarizer's clock (`diar`), or the TS-VAD track of the enrolled user's voice print (`tsvad`) (advanced) | [§5](#5-turn-head-input---turn-input) |
| `--tsvad PATH` | `runs/tsvad_spk.pt` | --turn-input tsvad: TS-VAD head file (advanced) | [§5](#5-turn-head-input---turn-input) |
| `--tsvad-print-s S` | `5.0` | --turn-input tsvad: seconds of speech in the voice print when armed (advanced) | [§5](#5-turn-head-input---turn-input) |
| `--tsvad-refresh-s S` | `0.0` | --turn-input tsvad: re-embed the print every S s of target speech (0 = never) (advanced) | [§5](#5-turn-head-input---turn-input) |
| `--titanet PATH` | `$AUDIOFORGE_DATA/nemo/speakerverification_en_titanet_large.nemo` | TitaNet-L .nemo for --enroll after_agent / explicit (advanced) | [§6](#6-primary-speaker-enrollment---enroll) |
| `--enroll-stride N` | `5` | voice modes: re-embed the columns every N frames (5 = 400 ms) (advanced) | [§6](#6-primary-speaker-enrollment---enroll) |
| `--silero PATH` | `$AUDIOFORGE_DATA/silero/silero_vad_v5.onnx` | Silero VAD v5 ONNX for `hybrid_silero` / `hybrid_dyn`; loaded at the first session that uses such a policy, or at start when the flag is given (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--silero-timeout-ms MS` | `2640` | hybrid_silero: any-speaker Silero silence that ends the turn (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--dyn-wait-ms CAP,FLOOR` | the served rule: 6000,1600 | `hybrid_dyn`: the Silero-silence wait at head posterior 0 (CAP) and 1 (FLOOR), linear in between (plus the rule's offset); `--mode single` sets `2000,960` (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--turn-policy {timeout,timeout_quiet,timeout_any,head,both,hybrid,hybrid_silero,hybrid_dyn,vad_head}` | `timeout` | the `turn_policy` of a session whose `config` does not name one (a client's `config` still wins); `--mode single` sets `vad_head` (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--vad-wait-ms K,FALLBACK` | `160,640` | `vad_head` (no Silero): the served VAD head's silence (VAD < 0.4) that the head path needs (K, with the turn head p >= theta, default 0.99) and the silence that ends the turn on its own (FALLBACK, 0 = none) (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--turn-preset {balanced,fast,steady,assistant}` | `balanced` | `vad_head`'s constants as one named trade-off (a client's `config.turn_preset` wins for its session). `balanced` = VAD < 0.4 for >= 160 ms AND p >= 0.99, OR 640 ms, others path 960,640. `fast` (turn head v5, served heads v0.3 and v0.4) = the v5 segment classifier asked after 80 ms of VAD < 0.6 and at every further quiet frame ends the turn at P(complete) > 0.7, OR 640 ms of VAD < 0.4, others path 960,640: on two-party calls (not a test split) 547 vs 955 ms p50 at 24.8 vs 20.2 % false interruptions and 5.5 vs 7.3 % missed (AMI test 1246 vs 1527 ms, 20.5 vs 15.5 % false interruptions, 37.5 vs 36.0 % missed). `steady` = the fast rule before v5 (VAD < 0.6 for >= 480 ms AND p >= 0.99, OR 720 ms, others path 640,640): 886 ms p50 but the best p95 (1434 ms) and misses (3.7 %). `assistant` = v5 asked after 240 ms of energy-or-VAD quiet, P(complete) > 0.9, OR 2960 ms of VAD silence: for speech directed at the agent, not for human conversation. `--vad-wait-ms` / `--others-wait-ms` override the preset's values | [§4](#4-turn-policies-configturn_policy) |
| `--others-wait-ms USER_SIL,HOLD` | `960,640` | `vad_head` with an enrolled TS-VAD track (`--turn-input tsvad`): the turn also ends when the user's own silence (P(user) < 0.5) reaches USER_SIL while P(other) >= 0.9 has held for HOLD, i.e. another speaker has the floor, without waiting for the room to go quiet (0 = off) (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--turn-hint-p P` | `0.8` | the early end-of-turn hint (`turn_end_hint`, docs/PROTOCOL.md 5.11): sent once per user turn on the first frame with 80 ms of served-VAD silence (VAD < 0.4) and turn-head p >= P, then confirmed by the next `turn_end` (`hinted_at`) or withdrawn by `turn_end_hint_cancel` when the user resumes (VAD > 0.5 on 2 frames); a voice agent starts preparing its reply on it (Pipecat eager end of turn, LiveKit preemptive generation). `turn_end` itself is unchanged (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--turn-hint-off` | off | no turn_end_hint / turn_end_hint_cancel events (and no turn_end.hinted_at) (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--energy-gate {on,off}` | `on` | `vad_head`'s energy gate: a per-session noise floor (10th percentile of the 80 ms frame log energies of the last 3 s); a user turn is armed only by an onset frame (VAD > 0.5 AND energy > floor + 6 dB), and no `turn_end` / `turn_end_hint` is sent before 160 ms of onset frames in the session. It removes the turn_ends the served VAD head (~0.55 on a fresh session's first frames, ~0.66 on room tone) fired before the user spoke: 137 -> 0 of 399 assistant clips; calls / AMI / the bundled clip unchanged. `off` = the VAD-only rule (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--energy-quiet-db X` | off | with `--energy-gate on`: a frame whose energy is below the noise floor + X dB is silence whatever the VAD says, so the silence starts at the audible end instead of at the end of the VAD head's tail. Off by default: on the assistant set it answers 400-600 ms sooner (X 6: 611 vs 1218 ms p50), but on two-party calls and AMI mid-turn pauses are room tone too, and false interruptions rise at every X in 3..12 dB (calls 20.2 -> 25-60 %, AMI 10.5 -> 11-40 %) even with the fallback re-tuned (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--turn-model {head,smartturn}` | `head` | `head` = the turn head's p (the shipped rule). `smartturn` = Pipecat's smart-turn v3.2 ONNX (pipecat-ai/smart-turn, BSD-2-Clause; a second model, 8.7 MB, ~22 ms per call on 2 CPU threads) asked once per silence run after 160 ms of VAD silence, on the turn's last <= 8 s with 0.5 s of pre-speech audio, prepared exactly as Pipecat's LocalSmartTurnAnalyzerV3; complete -> `turn_end` (`path: model`), incomplete -> wait for the next silence run or the preset's 3 s fallback. A bridge on assistant-directed speech while the native classifier (turn head v5) is trained (on human-to-human calls it misses more turn ends than the default) (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--smartturn-trigger {vad,energy}` | `vad` | `vad`: after 160 ms of VAD silence (VAD < the preset's threshold), fallback 3 s (the accurate setting). `energy`: two clocks, the classifier after 240 ms of energy-or-VAD quiet (energy < noise floor + 6 dB) ending the turn at P(complete) > 0.97, the fallback timer on the VAD's own silence at the preset's 640 / 720 ms (the fast setting; every mid-utterance stop longer than the timer still ends the turn). Numbers: docs/RESULTS.md (advanced) | [§4](#4-turn-policies-configturn_policy) |
| `--smartturn-onnx PATH` | the smart-turn-v3.2-cpu.onnx bundled in the installed `pipecat-ai` package | the smart-turn ONNX for `--turn-model smartturn` (advanced) | [§4](#4-turn-policies-configturn_policy) |

**transcripts**

| flag | default | meaning | more |
|---|---|---|---|
| `--final-asr SPEC` | off | offline per-turn final ASR: `tdt_v3` (NVIDIA Parakeet-TDT 0.6B v3) or a `.nemo` path; adds finals with `source: tdt_v3` | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-worker {process,thread}` | `process` | --final-asr: child process or thread (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-threads N` | `2` | --final-asr: torch threads of the offline model (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-device DEV` | `cpu` | --final-asr: cpu &#124; mps &#124; cuda (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--asr-lookahead R` | off | second, text-only ASR pass at attention context [70, R], e.g. 13 (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-chunk-ms {160,560,1120}` | 160 (single rate) | dual rate: the heads, partials and turn decisions keep the 160 ms pass; a second, text-only pass of the same frozen encoder at 560 ms ([70,6]) or 1120 ms ([70,13]) chunks writes the `final` text (`source: slow`). The 160 ms text is sent at once at the turn end as `final_fast`, so a client can answer from it and replace it. 160 = single rate | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-flush {on,off}` | `on` | --final-chunk-ms: at a turn end, encode the audio since the last slow chunk as a partial chunk so the slow `final` is sent at once (`on`); `off` waits for the whole slow chunk (up to 1.12 s of audio) (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--asr-chunk-ms {80,160}` | the model's (160) | streaming chunk of the one ASR pass (transcript, VAD, turn and TS-VAD heads): 160 = attention context [70,1] (the model's default) or 80 = [70,0], no lookahead (up to 80 ms earlier frames; +0.19 WER on LibriSpeech, +1.3 on AMI; the heads were trained at [70,1]) (advanced) |  |
| `--asr-vad-gate P` | off | do not decode transducer tokens on frames whose served VAD <= P once `--asr-vad-hangover-ms` of such frames have passed (bounds hallucinated text on long non-speech) (advanced) |  |
| `--asr-vad-hangover-ms MS` | `1200.0` | --asr-vad-gate: decoding continues this long after speech (advanced) |  |
| `--beam K` | 0 (greedy) | finals take the best hypothesis of an RNNT beam search of width K (up to 3 tokens per frame) run next to the greedy decoder; partials and the turn heads keep the greedy tokens. 115M: -1.3 / -2.3 WER points on held-out ICSI / AMI at K = 8, about 1 ms per 80 ms frame on CPU; the 0.6B does not gain (advanced) |  |

**language ID**

| flag | default | meaning | more |
|---|---|---|---|
| `--lid head&#124;PATH&#124;ambernet` | off | spoken language ID: `head` (the shipped distilled head on the shared encoder, `lid_115m_v2.pt`), a head file (`audioforge.lid.save_head`), or `ambernet` / an AmberNet `.nemo`; emits `language` messages and `stats.lang` (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-threshold P` | `0.9` | --lid: posterior needed to announce a language (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-min-ms MS` | `1000.0` | --lid: pooled speech needed before the first announcement (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-max-ms MS` | 3000 with `--lid head`, else off (0 = off) | --lid head/file: announce the top language after this much speech anyway (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-langs CODES` | 17 FLEURS languages | --lid ambernet: comma-separated language codes to choose from (advanced) | [§9](#9-spoken-language-id---lid) |
| `--voice-gender head&#124;PATH` | off | optional perceived voice-gender probabilities (female / male voice) from a small head on the speaker head's encoder tap: `head` (the core's shipped file, `voice_gender_115m.pt` / `voice_gender_0p6b.pt`) or a head file; adds `final.voice_gender` (that segment's speech) and `stats.voice_gender` (the session's). A perceived vocal characteristic, not a person's gender identity; it can be wrong for any individual (advanced) | [§9](#9-spoken-language-id---lid) |

**diarizer tuning**

| flag | default | meaning | more |
|---|---|---|---|
| `--diar-config {low_latency,low_latency_032}` | `low_latency_032` | Sortformer setting: 0.32 s (low_latency_032) or 1.04 s (advanced) | [§7.1](#71---diar-config-and---diar-set) |
| `--diar-set KEY=VALUE` | none | override one diarizer streaming field, KEY=VALUE (repeatable) (advanced) | [§7.1](#71---diar-config-and---diar-set) |
| `--diar-left N` | `188` | diarizer encoder left context in window mode (frames) (advanced) | [§7.2](#72---diar-left) |
| `--diar-pool {mean,max}` | the import's (`mean`) | Nemotron-3 only: 10 ms -> 80 ms pooling of its outputs (advanced) | [§7.3](#73-nemotron-3-diarization-as-the-diarizer) |
| `--diar-spks K` | all 8 | Nemotron-3 only: keep the first K of its 8 columns (advanced) | [§7.3](#73-nemotron-3-diarization-as-the-diarizer) |
| `--diar-embed {spk,titanet}` | `spk` | --diar-labels registry: turn embedding (spk = served head) (advanced) | [§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any) |
| `--diar-reg-thr X` | per embedder (spk 0.55, titanet 0.40) | --diar-labels registry: cosine at which a turn joins a known speaker (advanced) | [§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any) |
| `--shed-diar {vad,hold}` | `vad` (`audioforge-serve` adds `hold`) | under load: vad (VAD in column 0) &#124; hold (last column, half cadence) (advanced) | [§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any) |
| `--diar-off` | off | with `--turn-input tsvad`: run no diarizer (speakers = [P(user), P(other), 0, 0]); without `--diar` the diarizer is not even loaded (advanced) | [§5](#5-turn-head-input---turn-input) |

**speed**

| flag | default | meaning | more |
|---|---|---|---|
| `--perf SPEC` | `default` | CPU inference fast paths of `audioforge.perf`: `default` = the exact set (same outputs), `none`, `all` (adds float-rounding ones), or a list such as `default,-linear_t` (advanced) | [§3](#3-models-threads-and-speed) |
| `--device DEV` | `cpu` | cpu, or mps / cuda / cuda:N (opt-in GPU; others fall back to cpu) (advanced) | [§3](#3-models-threads-and-speed) |
| `--no-fast-conv` | off | keep PyTorch's Conv1d path in the conformer convolutions (slower) (advanced) | [§3](#3-models-threads-and-speed) |
| `--no-warmup` | off | skip the 2 s warm-up session at start (advanced) | [§3](#3-models-threads-and-speed) |

**limits, logging and diagnostics**

| flag | default | meaning | more |
|---|---|---|---|
| `--log-json` | off | one JSON object per log event instead of text lines |  |
| `--max-session-s S` | `14400.0` | audio per connection before the server flushes and closes it (advanced) |  |
| `--idle-timeout-s S` | `300.0` | close a connection that sends nothing for this long (advanced) |  |
| `--debug-fields` | off | add diagnostic keys to the messages and validate every message (advanced) | [§10](#10-diagnostics) |

**environment**

| variable | meaning |
|---|---|
| `AUDIOFORGE_HOME` | where `audioforge-download` writes the models and `audioforge-serve` finds them (default `<repo>/models` in a checkout, else `~/.cache/audioforge`) |
| `AUDIOFORGE_DATA` | data root: datasets, raw NVIDIA `.nemo` files (`nemo/`), Silero (`silero/`); default `<repo>/data` |
| `AUDIOFORGE_CONFIG` | config file used when `--config` is not given |
| `AUDIOFORGE_ACCEPT_LICENSES` | `1` accepts the model licences in `audioforge-download` without asking |

<!-- END GENERATED FLAGS -->

## 3. Models, threads and speed

**`--asr`** is the ASR front end: NVIDIA `stt_en_fastconformer_hybrid_large_streaming_multi` (frozen, cache-aware,
[70, 1] attention context = 160 ms chunks) plus this project's VAD, turn, speaker and EOU heads. The streaming RNNT
produces the partials and finals. The VAD head gives `frame.vad`, and the turn head gives `frame.eot`.

**`--threads`** sets the torch threads of the single worker thread that runs every session's compute. Measured on the
server alone (room mode, 1.04 s diarizer, older head set, 3 AMI windows + 1 LibriSpeech utterance per row):

| threads | RTF | chunk ms p50 / p95 | max backlog ms |
|---|---|---|---|
| 2 | 0.433-0.520 | 33-45 / 222-260 | 160-200 |
| 4 | 0.341-0.422 | 31-39 / 114-202 | 80-140 |

Room mode with the 0.32 s Sortformer and `--turn-input diar` ran at RTF 0.795 / 0.802 (Pipecat / LiveKit sessions) on
2 threads: ASR pass 0.149-0.151, turn pass 0.150, diarizer 0.496-0.500. Peak RSS was 3581 / 3586 MB. Concurrent
sessions share the one worker. Concurrency has not been load-tested.

**`--perf`.** The exact fast paths of `audioforge.perf` are on by default: column-major `nn.Linear` weights for the
small GEMMs of a 160 ms step (3-4x faster on Accelerate), a cached relative-position projection per attention layer,
one shared subsampling for the ASR and turn passes, and a cached RNNT prediction projection. Protocol messages match
`--perf none` (2 of 1112 `eot` values differ in the 5th decimal on 5 AMI windows). `--perf none` restores the older
code path.

**`--no-fast-conv`.** By default the conformer convolutions of both encoders use an equivalent unfold / `F.linear` CPU
path. Tokens stay identical, and diarizer probabilities differ by at most 3e-7. This path lets the server keep up: RTF
0.45-0.52 with it, 1.13-1.21 without it (2 threads). Use the flag only for debugging.

**`--device`.** The streaming server is CPU-only. `--final-asr-device` can put the offline final-ASR model on `mps`
or `cuda`.

**`--no-warmup`.** By default one throwaway session runs 2 s of low-level noise through the models at start, so the
first real session does not pay one-time initialization.

## 4. Turn policies (`config.turn_policy`)

A session picks its end-of-turn rule in its first `config` message. The default is `timeout` (room mode) or `vad_head`
(single mode). **Primary** means the diarizer column with the most activity over the last 5 s, or the bound column
under `--enroll` ([§6](#6-primary-speaker-enrollment---enroll)). A column is active on a frame when p > 0.5.

| policy | rule | config keys used | finals cut by |
|---|---|---|---|
| `timeout` (default) | `turn_end` when the primary column has been inactive for ≥ `timeout_ms` and has spoken since the last timeout `turn_end`. Other columns are ignored | `timeout_ms` (1000) | the timeout |
| `timeout_quiet` | the same, but also waits until no other column is active on that frame. Its `turn_end` is tagged `timeout` | `timeout_ms` | the timeout |
| `timeout_any` | ends every speaker's turn, for transcribing rooms ([§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)) | `timeout_ms` | the timeout or a speaker change |
| `head` | `turn_end` when the turn head's probability crosses `eot_threshold` upwards. Re-armed by speech (VAD > 0.5); fires at most once per speech segment | `eot_threshold` (0.98) | the head |
| `both` | the `timeout` rule and the `head` rule both fire, each tagged with its own name | `timeout_ms`, `eot_threshold` | the timeout |
| `hybrid` | primary-silence timeout OR head ≥ `eot_threshold`. One `turn_end` per turn, at the earlier of the two, tagged `hybrid` | `timeout_ms`, `eot_threshold` (0.98) | the hybrid event |
| `hybrid_silero` | head ≥ θ OR **any-speaker Silero VAD v5 silence** ≥ `--silero-timeout-ms` (2640 ms). Silero runs on 32 ms chunks through Pipecat's VAD state machine (confidence 0.7, start / stop 0.2 s). One firing per silence run | `eot_threshold` (default θ 0.99828); `timeout_ms` is not used | the hybrid event |
| `hybrid_dyn` | head ≥ θ OR Silero silence ≥ clamp(80 − 55 p, 7, 80) frames, where p is the head's posterior on the same frame: 6.4 s at p = 0, 4.24 s at p = 0.5, 2.48 s at p = 0.9, 2.0 s at p = 1 | `eot_threshold` (default θ 0.998283); `timeout_ms` is not used | the hybrid event |
| `vad_head` | **`--mode single`'s default**, no Silero. Fires when the served VAD head stays below 0.4 for ≥ K frames AND the turn head p ≥ θ, OR that silence reaches FALLBACK (`--vad-wait-ms K,FALLBACK`, default `160,640`; [presets](#turn-presets---turn-preset-configturn_preset) change it). With an enrolled TS-VAD track it also fires when the user is silent (P(user) < 0.5) for USER_SIL while someone else holds the floor (P(other) ≥ 0.9) for HOLD (`--others-wait-ms USER_SIL,HOLD`, default `960,640`, 0 = off). One `turn_end` per user turn | `eot_threshold` (default θ 0.99) | the vad_head event |

Notes:

- `timeout_ms` is clamped to [80, 4840]. With the default 1000 ms, 13 silent frames are needed. The decision also
  waits for the diarizer column to be finalized, so `turn_end.t` is at least the last primary frame + 1.0 s + the
  column lag (0.08-0.24 s with the default diarizer setting).
- `hybrid_silero` / `hybrid_dyn` must be in the first `config`, before any audio. They load the Silero ONNX once per
  server (one shared session, about 0.11 ms per 32 ms chunk, about +24 MB RSS). They need `onnxruntime`
  (`audioforge[serve]`).
- Setting `eot_threshold` in the config replaces the frozen θ of `hybrid_silero` / `hybrid_dyn`.
- `--silero-timeout-ms` is a server-wide flag, not a per-session config key.

### Turn presets (`--turn-preset`, `config.turn_preset`)

`vad_head` comes in four presets, each a named trade-off. `balanced` is the default. `fast`, `steady` and
`assistant` are opt-in, per server (`--turn-preset fast`) or per session (`"turn_preset": "fast"` in the first
`config`). `--vad-wait-ms` / `--others-wait-ms` still override a preset's waits. The exact rules are in the
`--turn-preset` flag above.

Measured with the 115M core and heads v0.4. AMI test = 200 turns of the AMI test meetings, with the user's 5 s print.
Calls = 16 TurnBench dev clips + 16 oto conversations, user channel, 109 turn ends. Calls are **not a test split**
(TurnBench publishes no labelled test split).

| preset / system | calls EOT p50 / p95 | calls false int. / missed | AMI test EOT p50 / p95 | AMI test false int. / missed |
|---|---|---|---|---|
| **`balanced`** (default) | **955 / 1918 ms** | **20.2 % / 7.3 %** | 1527 / 4439 ms | 15.5 % / 36.0 % |
| **`fast`** | **547 / 1861 ms** | **24.8 % / 5.5 %** | 1246 / 3967 ms | 20.5 % / 37.5 % |
| `assistant` | not for calls | | 1327 / 4774 ms | 23.5 % / 40.0 % |
| LiveKit turn detector (EnglishModel) + Silero, LiveKit defaults | 567 / 3127 ms | 26.6 % / 22.9 % | 745 / 4640 ms | 13.0 % / 72.0 % |
| Pipecat smart-turn v3.2 + Silero, Pipecat defaults | 237 / 3217 ms (bimodal p50) | 35.8 % / 24.8 % | 752 / 4419 ms | 39.5 % / 53.0 % |

- **`fast`** answers about 400 ms sooner at p50 on calls (547 vs 955 ms) and 280 ms sooner on AMI test. It misses
  fewer turn ends on calls (5.5 vs 7.3 %). It costs 4.6 more points of false interruptions on calls and 5.0 more on
  AMI test. Pick it when a late answer hurts more than an occasional interruption.
- **`assistant`** is for speech directed at an agent.

### Energy gate and `--turn-model smartturn` (`vad_head`)

**Energy gate** (`--energy-gate on`, the default). The served VAD head reads about 0.55 on a fresh session's first
frames and about 0.66 on room tone, so the plain rule could end an empty turn before the user spoke (137 of 399
assistant clips). The gate fixes this (0 of 399):

- It keeps a per-session noise floor: the 10th percentile of the 80 ms frame log energies of the last 3 s.
- A user turn is armed only by an onset: VAD > 0.5 AND energy > floor + 6 dB.
- No `turn_end` or `turn_end_hint` is sent before 160 ms of onsets in the session.

`--energy-quiet-db X` also counts frames below floor + X as silence. Assistant speech then answers in 611 instead of
1218 ms at X = 6. On human-to-human calls it raises false interruptions (48.6 % at X = 6), so it is off by default.

**`--turn-model smartturn`** (opt-in). This runs Pipecat's smart-turn v3.2 classifier (a second model: 8.7 MB ONNX,
BSD-2-Clause, the copy bundled with `pipecat-ai` or `--smartturn-onnx`) inside `vad_head`, in place of the turn
head's p. It is asked once per silence run, with input prepared as Pipecat's LocalSmartTurnAnalyzerV3 does. Compute:
about 20 ms per call on 2 CPU threads, about 1.1 calls per assistant turn. `turn_end.path` is `"model"`,
`turn_end.model_ms` its compute, `stats.turn_model` the counters.

| setting | assistant EOT p50 / p95 | false fires on incomplete | accuracy | calls p50 / FI / missed | AMI dev p50 / FI / missed |
|---|---|---|---|---|---|
| default (`--turn-model head`, energy gate) | 1240 / 1628 ms | 100 % | 43.9 % | 956 ms / 20.2 % / 7.3 % | 1326 ms / 10.5 % / 33.5 % |
| `--turn-model smartturn` (`--smartturn-trigger vad`: 160 ms of VAD silence, 3 s fallback) | 770 / 1004 ms | 5.8 % | 95.7 % | 629 ms / 19.3 % / 23.9 % | 1087 ms / 13.0 % / 36.5 % |
| `--turn-model smartturn --smartturn-trigger energy` (240 ms energy-or-VAD quiet, P > 0.97, the preset's VAD fallback) | 290 / 1267 ms | 100 % | 37.6 % | 961 ms / 24.8 % / 8.3 % | 1247 ms / 18.0 % / 33.5 % |
| Pipecat smart-turn v3.2 + Silero (defaults), for reference | 211 / 242 ms | 40.2 % | 69.7 % | 237 ms / 35.8 % / 24.8 % | 385 ms / 27.5 % / 44.5 % |

This table is a selection experiment on dev data, not a test result. Use smart-turn for assistant-directed speech,
where it tells a finished request from a mid-sentence stop. Do not use it for human-to-human calls: its 3 s fallback
answers "incomplete" turn ends late, so it misses about 3x more turn ends.

### 4.1 Measured tradeoffs

Room-mode policies, measured on AMI **dev** (selection experiments, not test results). Benchmark: miss at ≤ 5 % FC,
6 s horizon, label-free binding, n = 974, on 1.04 s / 0.32 s Sortformer tracks unless noted. Live: 5 AMI windows
through Pipecat / LiveKit, 0.32 s diarizer.

| policy | benchmark miss | live EOT latency (median), false interruptions |
|---|---|---|
| `timeout` | 74.8 % / 70.1 % | 1923 / 1684 ms, 3 / 5 |
| `timeout_quiet` | 66-68 %, against 38.4 % for `timeout` on the same track (n = 200) | not measured |
| `head` | 66.5 % / 65.8 % | Pipecat: a turn end on only 2 of 5 windows. LiveKit: 1237 ms on 4 of 5, 1 |
| `both` | not measured | not measured |
| `hybrid` | 61.9 % / 63.7 % | θ 0.998 / 1000 ms: same as `timeout`. θ 0.998 / 4000 ms: 4522 ms (Pipecat), 0 |
| `hybrid_silero` | held-out ICSI (n = 1312): 81.7 %, floor-open 62.2 % | 2941 / 2921 ms, 0 / 0 |
| `hybrid_dyn` | held-out ICSI: 79.5 %, floor-open 50.5 %. AMI dev: 58.2 % | 2662 / 2425 ms, 0 / 0 |

Reading:

- No rule wins on both axes. The hybrids respond later than the 1000 ms timeout (+0.66 to +1.2 s) and cut in less.
  Choose `timeout` when responsiveness matters most, and `hybrid_dyn` when not interrupting the user matters most.
- The Silero branch is speaker-unaware. It cannot tell a hand-over from a pause where nobody else speaks.
- `hybrid` pays off only with a clean speaker track. On the streaming Sortformer track the head adds about 1 point.
- Picking the primary speaker without labels is the largest loss. With oracle binding the same benchmark gives
  timeout 32.0 %, head 30.3 %, hybrid 28.7 %. `--enroll` addresses this ([§6](#6-primary-speaker-enrollment---enroll)).
- The timeout still ends turns during filled pauses ("um") or when the user resumes within 1 s.

## 5. Turn head input (`--turn-input`)

| value | behaviour | cost |
|---|---|---|
| `auto` (default) | `diar` if the model's turn head reads diarizer activity or columns, else `session`. The served head reads columns, so `auto` selects `diar` | |
| `session` | the head reads the session's own encoder frames. No speaker information reaches it. Refused at start-up for heads that read the diarizer's activity, including the served one | none extra |
| `tsvad` | the head reads the TS-VAD track of the enrolled user's voice print ([P(user), P(other), 0, 0], primary 0) from a 0.26 M head on block 4 of the ASR pass, on the 160 ms chunk clock (no diarizer lag). With `--diar-off` the track also replaces the diarizer's columns. `--mode single` sets both ([§13](#13-single-model-mode---mode-single)) | the turn pass (RTF about 0.15) as for `diar`; the TS-VAD head is negligible |
| `diar` | the head runs on the diarizer's clock: frame v once diarizer column v exists. Speaker-kernel heads get a second, speaker-conditioned encoder pass over the same mel chunks | turn pass RTF 0.150 |

With `diar`, `eot` and head decisions also carry the diarizer column lag.

## 6. Primary-speaker enrollment (`--enroll`)

Which diarizer column is "the user" decides what `timeout`, `timeout_quiet`, `both` and `hybrid` listen to. It also
decides what a `--turn-input diar` head is conditioned on and what `final.speaker` reports. The Silero branch of
`hybrid_silero` / `hybrid_dyn` does not depend on it. The head branch does.

| mode | client message | rule | cost |
|---|---|---|---|
| `dominant` (default) | none | the column with the most activity over the last 5 s; `null` if none was active | none |
| `after_agent_arm` | `{"type": "agent_end"}` when the agent's TTS ends | from that audio position, the first column active for ≥ 3 consecutive frames is bound. It re-binds only after > 25 silent frames of the bound column. Each `agent_end` re-arms. No TitaNet | none measurable |
| `after_agent` | `{"type": "agent_end"}` | the same choice. The column's first 19 active frames (1.5 s) become a TitaNet-L enrollment, and the primary then follows that voice (re-embedded every `--enroll-stride` frames; hysteresis margin 0.1 cosine, hold 6 frames) | TitaNet-L +91 MB RSS. RTF 0.87-1.16 vs 0.60-0.84 without it (2 threads): at the edge of keeping up |
| `explicit` | `{"type": "enroll"}` (e.g. after a "say something" prompt) | as `after_agent`, started by `enroll`; one enrollment per message | as `after_agent` |

Until a column is chosen, every mode uses the `dominant` rule. The other modes add `enroll` / `enrolled` /
`primary_column` to `ready` and `stats` and send an `enrolled` message
([PROTOCOL.md §5.6](PROTOCOL.md#56-enrolled---enroll-other-than-dominant)). `--titanet` sets the TitaNet-L `.nemo`
(the launcher fills it from the models directory). `--enroll-stride` (default 5 frames = 400 ms) sets how often the
voice modes re-embed. `--enroll-stride 10` halves that cost at the price of slower switching.

Measured effect (offline benchmark, AMI dev n = 974, miss at ≤ 5 % FC, 1.04 s Sortformer tracks; the "agent end" is
the previous speaker's labelled turn end):

| binding (server mode) | hybrid, 6 s | hybrid, 2 s | timeout, 6 s |
|---|---|---|---|
| label-free dominant (`dominant`) | 61.9 % | 79.1 % | 74.8 % |
| `after_agent_arm` | 56.0 % | 75.3 % | 67.6 % |
| `after_agent` | 51.6 % | 87.6 % | 53.7 % |
| `explicit` (enrollment on the true user's first utterance) | 47.1 % | 81.0 % | 52.9 % |
| oracle column (upper bound) | 28.7 % | 53.1 % | 32.0 % |

- `after_agent_arm` is better at both horizons. Use it whenever the app knows its TTS end.
- `after_agent` catches the same turns 2-6 s late. Enable it only where the 6 s miss rate matters more than latency.
- Send `agent_end` exactly when the TTS playback ends. If it is sent while the agent is still audible, the agent's
  own column can be chosen (tested in `tests/test_serve.py`).

## 7. Diarizer

### 7.1 `--diar-config` and `--diar-set`

Two NVIDIA model-card settings of Streaming Sortformer v2. Both use FIFO 188, speaker-cache update period 144,
speaker cache 188 and encoder left context 188 frames:

| preset | card name | chunk / right context (frames) | input buffer | column finalized after frame end | `ready.column_lag_ms` |
|---|---|---|---|---|---|
| `low_latency_032` (default) | ultra low latency | 3 / 1 | 0.32 s | 80-240 ms (mean 160) + compute | 160.0 |
| `low_latency` | low latency | 6 / 7 | 1.04 s | 560-960 ms (mean 760) + compute | 760.0 |

| measured (AMI dev) | `low_latency_032` | `low_latency` |
|---|---|---|
| turn-end miss, 6 s: hybrid / head / timeout | 63.7 / 65.8 / 70.1 % | 61.9 / 66.5 / 74.8 % |
| DER (miss / FA / confusion), 6 s windows | 28.3 (17.6 / 5.0 / 5.7) | 26.3 (16.4 / 5.0 / 4.9) |
| server RTF, 2 threads | 0.49 / 0.53 | 0.40 / 0.43 |
| speaker-column age on arrival, p50 (p95) | 210-227 (291) ms | 857 (1035) ms |

The 0.32 s setting is a latency gain, not an accuracy gain. Miss rates stay within the 1.04 s CIs, DER is 2 points
worse, and it costs about +0.1 RTF. Transcripts are identical at both settings.

**`--diar-set KEY=VALUE`** (repeatable) overrides one streaming field on top of the preset. Keys are fields of
`streaming_diar.AOSCConfig`: `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_len`,
`spkcache_update_period`, `sil_frames`, `strong_boost_rate`, `threshold`. A value containing `.` is parsed as a
float, anything else as an integer. Any override appends `+custom` to `ready.diar_config`, and `column_lag_ms`
follows the new chunk / right context. No custom setting has been measured.

### 7.2 `--diar-left`

Encoder left context of the diarizer, in frames (default 188, the value both presets were measured with). A shorter
left context is cheaper. For Sortformer v2 its effect on accuracy was not measured. Nemotron-3-Diarization's encoder
is frame-local, so 1 is the recommended value there.

### 7.3 Nemotron-3-Diarization as the diarizer

`--diar` also accepts the Nemotron-3-Diarization import (`nemo_nemotron3_diar.afm`: 99.2 M parameters, OpenMDW-1.1
licence, 8 output columns at 10 ms). Two flags exist only for it:

- `--diar-pool {mean,max}`: pooling of its 10 ms outputs to the 80 ms frame. The import's default is `mean`. `max`
  marks a frame active if any 10 ms sub-frame is active.
- `--diar-spks K`: keep the first K of its 8 arrival-order columns (dropped, not merged). `frame.speakers` carries
  as many columns as the diarizer has, so the cut is not needed.

Measured with `--diar-pool max --diar-left 1 --diar-spks 4`, live, 2 threads, Pipecat / LiveKit:

| | Sortformer v2 (0.32 s) | Nemotron-3, max pool |
|---|---|---|
| server RTF | 0.795 / 0.802 | 0.635 / 0.641 |
| diarizer RTF | 0.496 / 0.500 | 0.300 / 0.302 |
| server peak RSS | 3581 / 3586 MB | 1485 / 1482 MB |
| EOT latency with `timeout`, two-party calls | | 8-135 ms lower |
| streaming DER, 1.04 s buffer, AMI dev 64 × 20 s | 0.252 | 0.241 |
| primary-track miss (offline, 200 turn windows) | 0.129 | 0.192 |

Caveats: its model card lists AMI train + dev in its training data, and it misses more speech on this project's
labels. The server applies the Sortformer v2 presets (`--diar-config`) to it. Its own card profile has not been
measured inside the server.

### 7.4 Multi-speaker rooms: `--diar-labels`, `--shed-diar`, `timeout_any`

For rooms with many speakers. The defaults keep the single-user behaviour.

- **`--diar-labels registry`** labels each final by voice. The column that dominates the turn's own span picks the
  speaker's frames, and the served speaker head embeds them. A per-session registry returns the closest known
  speaker when the cosine is at least the threshold, else a new id. Ids are stable across column permutations and are
  not limited to the column count. `stats.speakers_seen` is the speaker-count estimate and `final.speaker_conf` the
  match cosine. `--diar-embed titanet` uses TitaNet-L per turn instead (+91 MB).
- **`turn_policy: "timeout_any"`** (session config, before the first audio) ends every speaker's turn: after
  `timeout_ms` of nobody talking, or at a speaker change once the previous speaker has been silent for 240 ms and the
  new one active for 240 ms (`turn_end.policy: "change"`). Overlap and short backchannels do not cut. Keep `timeout`
  for a voice agent that must follow one user.
- **`--shed-diar hold`** changes load shedding. Without it, shedding replaces the diarizer by the VAD in column 0, so
  every turn becomes speaker 0. With `hold`, the last stable column stays active while the VAD hears speech, the
  diarizer runs on every other block, and affected finals carry `diar_shed: true`.
- Nemotron-3 no longer runs with `--diar-spks 4`, so speakers 5-8 are visible in `frame.speakers`.

## 8. Final ASR and dual lookahead

The streaming model keeps every live decision (partials, VAD, `turn_end`, speakers, enrollment). Two opt-in extras
send additional finals for each cut turn, marked by `final.source` ([PROTOCOL.md §5.5](PROTOCOL.md#55-final)).

**`--final-asr tdt_v3`** transcribes each finished turn once with NVIDIA Parakeet-TDT 0.6B v3 (CC-BY-4.0; a 2.5 GB
download: `audioforge-download --with tdt_v3`). A `.nemo` path loadable by `audioforge.nemo_import` also works; the
finals then carry that file's stem as `source`. The turn's audio runs from 0.3 s before the VAD's speech onset to
0.5 s after the last VAD speech frame, at most 60 s.

| flag | default | meaning |
|---|---|---|
| `--final-asr-worker` | `process` | `process`: a child process with its own threads and GIL, so the offline decode never blocks streaming. `thread`: a worker thread in the server process, sharing the GIL and torch pool |
| `--final-asr-threads` | `2` | torch threads of the offline model |
| `--final-asr-device` | `cpu` | `cpu`, `mps` or `cuda` |

Jobs run one at a time, in order. `stats` waits for every pending offline final and reports `final_latency_ms` and
`final_asr_rss_mb`.

**`--asr-lookahead R`** runs a second, text-only pass of the same ASR model at attention context [70, R]. With
R = 13 that is 1.12 s chunks and 1.04 s of lookahead. Its finals (`source: "lookahead"`) are due 3 frames after the
segment's last VAD speech frame. Heads, events and partials are unchanged. It costs one more encoder pass per chunk.
Both flags can be combined.

Measured (Whisper normalizer):

- AMI test, 200 single-speaker segments: streaming 115M 16.1 % WER, `--final-asr tdt_v3` 8.3 %.
- TDT v3 cost: RTF 0.065 per turn on 2 CPU threads and about 2.5 GB RSS in the worker.
- LibriSpeech test-clean (first 200 utterances): `--asr-lookahead 13` 1.92 % against 2.29 % for streaming. Its AMI
  WER was not measured.
- TDT v3 covers 25 European languages. No Hebrew, Arabic, Turkish, Persian, Hindi, Chinese or Japanese.

## 9. Spoken language ID (`--lid`)

Off by default. With `--lid`, the server sends `language` messages and adds `stats.lang`. It announces the top
language once its posterior reaches `--lid-threshold` (default 0.9) after at least `--lid-min-ms` (default 1000) of
pooled speech, and again whenever the confident top language changes. With `--lid-max-ms MS` (default 3000 for
`--lid head`, off otherwise, `0` = off) it announces the top language anyway once that much speech is pooled.

| backend | how | FLEURS test (17 languages, n = 2550): 2 s / full | accented English (EdAcc): classified "en" | cost |
|---|---|---|---|---|
| **`--lid head`** | a 2.37 M-parameter head (0.6B core: 2.89 M) on the shared encoder pass, distilled from AmberNet. No extra encoder pass | **92.4 % / 98.2 %** (0.6B: 92.7 / 98.6 %) | **65.8 %** | 0.21 ms per 160 ms chunk |
| a head file (`--lid PATH.pt`) | an older 0.40 M-parameter head on the encoder's layer outputs | 75.0 % / 89.3 % | 18.7 % | 0.16 ms per chunk |
| `--lid ambernet` (or an AmberNet `.nemo`) | NVIDIA AmberNet (28.9 M parameters, NGC Terms of Use). Re-classifies the last 8 s of speech on a schedule, restricted to `--lid-langs` | 95.1 % / 99.5 % | 67.7 % | 16-81 ms per call; RTF about 0.02-0.03 |

- `--lid head` is the no-second-model option, at about 1 / 100 of AmberNet's cost. Its weakest languages are
  Ukrainian (58 % at 2 s, mostly called Russian) and German (70 %). It is trained on AmberNet's outputs (NGC Terms
  of Use): check those terms before redistributing it.
- For mostly non-native English speakers, restrict the label set and raise `--lid-threshold`. Both backends call
  about a third of accented English segments another language.
- `--lid-langs` (AmberNet only) defaults to `en, he, ar, ru, es, fr, de, pt, it, nl, pl, uk, tr, fa, hi, zh, ja`.
  The head's label set is the same 17, fixed by the head file.

**Perceived voice gender (`--voice-gender head|PATH`, off by default).** A small head on the speaker head's encoder
tap adds `final.voice_gender` (that segment) and `stats.voice_gender` (the session): probabilities for female voice /
male voice. `head` picks the core's shipped file (`audioforge-download --with voice_gender` / `voice_gender_0p6b`).
It is a perceived vocal characteristic, not a person's gender identity, and it can be wrong for any individual. Do
not use it to make decisions about people.

## 10. Diagnostics

`--debug-fields` adds diagnostic keys to `frame` (`spk_t`), `turn_end` (`frame_t`), `ready` and `stats` (timing,
backlog, lag and cost breakdown). The full list is in [PROTOCOL.md §6](PROTOCOL.md#6-keys-added-by---debug-fields).
It also runs `validate()` on every outgoing message. The debug stat `diar_lag_ms_mean_measured` is computed from
sample counts and always reads the structural mean at 1x.

The server prints one log line per connection event: config applied, ignored config fields, enrollment armed,
disconnects, and the final `stats`.

## 11. Presets

**The default is single-model mode** (`audioforge-serve` with no mode flag, [§13](#13-single-model-mode---mode-single)).
The presets below are **room mode**. Each passes `--diarizer nemotron3`, which selects room mode by itself. They
assume `audioforge-download --diarizer nemotron3` (plus any optional models named) has been run. Numbers are live,
CPU, 2 threads, Pipecat / LiveKit.

### Two-party voice agent

One user talking to the agent; the application knows when its TTS stops.

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --enroll after_agent_arm
# client: {"type":"config","turn_policy":"timeout","timeout_ms":1000}; send {"type":"agent_end"} when TTS playback ends
# optional transcript pass (AMI test WER 8.3 % vs 16.1 % streaming, see §8): add --final-asr tdt_v3
```

User channel, median EOT latency 1285 / 1256 ms and 20 % / 18 % missed within 3 s (TurnBench dev, 16 clips), 1271 /
1250 ms and 6 % / 6 % (otoSpeech dev, 16 clips), 1.32-1.34 false interruptions per minute. Server RTF 0.635 / 0.641.
For fewer interruptions use `"turn_policy": "hybrid_dyn"`: about +0.9 s of latency, interruptions down 30-55 %. Feed
the user's own channel rather than a mix when you have it.

### Multi-party room

Several people in the room; the agent is addressed by one of them.

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --enroll after_agent_arm
# client: {"type":"config","turn_policy":"hybrid_dyn"} (first message, before audio); send agent_end at TTS end
```

On 5 AMI meeting windows: 0.00 false interruptions per minute in both frameworks, at median EOT latency 2322 / 2340
ms. The timeout variants interrupted 1.36-3.39 times per minute there. Small sample: n = 5 windows.

### Lowest latency

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --threads 4
# client: {"type":"config","turn_policy":"timeout","timeout_ms":1000}
# do not add --final-asr / --asr-lookahead (their finals arrive after the streaming final)
```

The 1000 ms `timeout` on the 0.32 s diarizer setting is the lowest-latency room configuration measured. The floor is
about 1.2-1.4 s after a clean end (1.0 s + 0.08-0.24 s + compute). A lower `timeout_ms` lowers the floor by the same
amount, but has not been measured. `--enroll after_agent_arm` is free and can be added.

### Most accurate

```bash
audioforge-serve --diar-config low_latency --enroll after_agent --threads 4 \
    --final-asr tdt_v3 --asr-lookahead 13
# client: {"type":"config","turn_policy":"hybrid_dyn"}; send agent_end at TTS end
```

- `hybrid_dyn`: lowest all-ends miss on held-out ICSI and no false interruptions on 5 AMI windows ([§4.1](#41-measured-tradeoffs)).
- `after_agent`: lowest 6 s miss of the deployable bindings, but later at 2 s ([§6](#6-primary-speaker-enrollment---enroll)).
  It needs `--threads 4` to keep up.
- `low_latency` diarizer: 2 DER points better than 0.32 s, about 0.6 s more column lag ([§7.1](#71---diar-config-and---diar-set)).
- `--final-asr tdt_v3` for the transcript ([§8](#8-final-asr-and-dual-lookahead)).

This preset has not been run end to end.

## 12. Config file and environment

Every flag can also be set in a YAML or JSON file, by its long name with dashes or underscores. Command-line flags
win over the file; the file wins over the built-in defaults. Unknown keys and bad values stop the server at start
with one line naming the key.

```yaml
# serve.yaml: audioforge-serve --config serve.yaml   (or AUDIOFORGE_CONFIG=serve.yaml audioforge-serve)
diarizer: nemotron3        # audioforge-serve only
port: 8765
threads: 2
enroll: after_agent_arm
diar-labels: registry
final-asr: tdt_v3
log-json: true
```

`python -m audioforge.serve --config FILE` reads the same file and ignores the launcher-only keys (`models-dir`,
`diarizer`); there `asr` and `diar` may come from the file too. The environment variables are listed at the end of
the [flag reference](#2-flag-reference).

## 13. Single-model mode (`--mode single`)

The default mode, for a voice agent that talks to **one known user**. Everything runs from the one 115M checkpoint,
with no NVIDIA diarizer and no second ASR model. The two-model stack is `--mode room` ([§1](#1-launching)).

```bash
audioforge-serve                                # = --mode single (or mode: single in the --config file)
# client: {"type":"config"} (the turn rule defaults to vad_head; a config turn_policy wins)
#         then {"type":"enroll","embedding":[192 floats]} (a stored print), or {"type":"agent_end"} at each TTS end
```

What the preset sets (flags you pass yourself win):

| option | value | why |
|---|---|---|
| `--turn-input` | `tsvad` | the turn head reads the TS-VAD track of the user's voice print ([§5](#5-turn-head-input---turn-input)) |
| `--diar-off` | on, and no `--diar` | the track [P(user), P(other), 0, 0] is `frame.speakers`; the primary is always column 0 |
| `--lid` | `head` | language ID from the head on the same encoder pass ([§9](#9-spoken-language-id---lid)) |
| `--enroll` | `after_agent_arm` | without a stored print, the print is the first `--tsvad-print-s` (5) s of speech after `agent_end` |
| `--turn-policy` | `vad_head` | the model's own VAD and turn heads plus the user's TS-VAD track, no Silero ([presets](#turn-presets---turn-preset-configturn_preset)) |
| `--dyn-wait-ms` | `2000,960` | for a client that asks for `hybrid_dyn` (then pass `--silero`, or `audioforge-download --with silero`) |
| `--tsvad` | `tsvad_spk.pt` | from the models directory |

No Silero is loaded in single mode. A session that asks for `hybrid_silero` / `hybrid_dyn` without `--silero` runs
`hybrid` with a `silero_unavailable` notice.

Refused with a one-line error, because each would load a second model: `--diar`, `--diarizer`, `--final-asr`,
`--lid ambernet`, `--diar-embed titanet`. An `enroll` message carrying an `embedding` sets the print under every
`--enroll` mode whenever the TS-VAD path is on. The same preset is available in-process as
`audioforge.load(mode="single")` (then `session.enroll(embedding)` / `session.agent_end()`) and as
`audioforge-bench --mode single`.

**The voice print is required.** Store at least 5 s of the user's clean speech (10 s for meetings) and send it at
every connect. The print is 192 numbers from the served speaker head. Two ways to make one:

- `audioforge.voiceprint(audio)` in Python;
- a server started with `--enroll explicit`: `{"type": "enroll"}` without an embedding takes the next 5 s of speech,
  and the `voiceprint` message returns the 192 numbers to store.

Print length matters (offline benchmark, misses at 6 s, ≤ 5 % FC): a stored 5 s print gives 34.2 % misses on AMI and
18.7 % on ICSI, a 10 s print 27.8 % and 19.6 %. A 3 s print loses 13 points on AMI, a 1.5 s print 20-22. A print taken
live after `agent_end` loses 16-41 points in meetings.

`final.speaker` is 0 for the user and 1 for someone else, by the TS-VAD column that dominates the turn.

Measured on 32 two-party calls, user channel, through Pipecat, with the earlier turn rule `hybrid_dyn 2000,960`:

| | single mode | room mode | LiveKit default | Pipecat default |
|---|---|---|---|---|
| EOT latency p50 / p95 | 1382 / 3400 ms | 1272 / 1631 ms | 1350 / 3096 ms | 1675 / 3197 ms |
| false interruptions | 17.4 % | 26.6 % | 23.9 % | 30.3 % |
| user turns answered | 88.1 % | 89.9 % | 82.6 % | 78.9 % |

- **Partial latency** (word end to word shown): 441 / 732 ms p50 / p95 on CPU.
- **Compute:** 53 ms per 160 ms chunk on 2 CPU threads (3 real-time sessions per process), about half of room mode.
  Peak RSS 1156 MB, against 1485 MB for room mode.
- **WER** (streaming, Whisper normalizer): 2.38 % LibriSpeech test-clean, 6.82 % test-other, 16.1 % AMI test.

**Not for:**

- labelling several people in a room (use `--mode room --diar-labels registry`);
- a meeting-grade final transcript (use `--mode room --final-asr tdt_v3`).

The TS-VAD and LID head files are fetched by `audioforge-download`, which shows their sha256-pinned sources. In a
checkout they come from `assets/`.
