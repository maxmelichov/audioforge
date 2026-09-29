# audioforge server configuration

Every flag of `audioforge.serve`, what it does, what it costs, and what was measured about it. The wire protocol is
in [PROTOCOL.md](PROTOCOL.md). Per-session options (`turn_policy`, `timeout_ms`, `eot_threshold`, `sample_rate`)
are not flags: the client sends them in its first `config` message.

Rules for the numbers on this page: each one is copied from the linked research document, with its scope (dataset,
n, operating point, machine). Where nothing was measured, the page says so. Most measurements ran on one shared
Apple-silicon Mac, CPU only, at 2 torch threads. Treat the compute numbers as order-of-magnitude figures for other
machines.

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

How to read the turn-taking numbers:

- **Miss at ≤ 5 % false cutoffs (FC).** From eot-bench v2 and STAGE1. Each system's threshold is fixed so that at most 5 % of turns
  get a firing before their true end (a false cutoff). A miss is a turn end with no firing within the horizon: 2 s
  or 6 s after the end. Lower is better. In eot-bench v2 the threshold is cross-fitted: it is chosen on half of the
  meetings and applied to the other half.
- **floor-open** ends are the ones where nobody else speaks within 1.04 s after the end, i.e. where an agent should
  reply. **taken** ends are followed by another speaker or overlap.
- **Binding** says which diarizer column counts as "the user". `oracle` uses the labels (an upper bound).
  `causal_dominant` is label-free and deployable; the server's default 5 s dominant rule is the same kind of rule,
  not the identical rule.
- **Dead air / cut-ins (product tests).** Dead air is the time from the true end of the user's turn to the moment
  the agent framework (Pipecat or LiveKit) receives the decision. A cut-in is a decision while the user keeps
  talking. These were measured live through the integrations.
- The benchmark fits its own timeout length (for example k 35-39 frames, about 3 s, for the causal timeout at
  6 s). The server's default timeout is 1000 ms. Benchmark rows are therefore not the served default operating point
  unless the text says so. The product tests use the served values.

## 1. Launching

Installed entry point (from `pip install "audioforge[serve]"` or a source checkout):

```bash
audioforge-download                 # fetch + verify the served ASR, the TS-VAD and LID heads, Silero
audioforge-serve                    # single-model mode (the default, §13): 127.0.0.1:8765, 2 threads
audioforge-download --diarizer nemotron3 && audioforge-serve --mode room   # room mode: + NVIDIA Nemotron-3-Diarization
```

**Two modes (since 2026-09-29).**
- **`single`, the default.** Everything from the one 115M checkpoint for a known user, whose stored voice print
  (≥ 5 s of clean speech, 10 s for meetings) the client sends as `{"type": "enroll", "embedding": [...]}`
  ([§13](#13-single-model-mode---mode-single)).
- **`room`.** General diarization with NVIDIA Nemotron-3-Diarization (or Streaming Sortformer v2) next to the 115M
  model, for labelling everyone in a room, optionally with `--final-asr tdt_v3`. This was the default until
  2026-09-29.

Without `--mode`, `--diarizer`, `--diar` or `--final-asr` select room mode, so older command lines keep working.
The launcher prints one line when it does that.

`audioforge-serve` is a thin launcher (`audioforge/launch.py`) on top of `python -m audioforge.serve`: it adds
`--models-dir`, `--mode` and `--diarizer` (see the flag reference below). The rest of this section describes room
mode. `audioforge-serve --help` lists the everyday
flags, `--help-advanced` all of them; any flag can also come from a config file ([§12](#12-config-file-and-environment)).
With `--diarizer nemotron3` the launcher adds `--diar-pool max --diar-left 1` ([§7.3](#73-nemotron-3-diarization-as-the-diarizer));
until 2026-09-28 it also added `--diar-spks 4`, which hid speakers 5-8. For either diarizer it adds `--shed-diar hold`
([§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)); anything you pass yourself wins.

When `--asr` / `--diar` are not given, the launcher fills them from the models directory
(`stage1_served.afm`, and `nemo_nemotron3_diar.afm` if present, else `nemo_sortformer_v2.afm`). In a source checkout it falls back
to the legacy `runs/` paths. It also fills in the optional model files that some flags need when they are in the
models directory: TitaNet for `--enroll after_agent|explicit`, AmberNet for `--lid ambernet`, Parakeet-TDT v3 for
`--final-asr tdt_v3`, and Silero. If a needed model is missing, it exits with the `audioforge-download` command to
run. Every other flag is passed to `audioforge.serve` unchanged.

The module can also be run directly, with explicit model paths:

```bash
python -m audioforge.serve --asr models/stage1_served.afm --diar models/nemo_sortformer_v2.afm [flags]
# source checkout without installing:
PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_served_v2.afm --diar runs/nemo_sortformer_v2.afm
```

**`stage1_served_v2.afm` is what ships** (built by `audioforge-download` from `assets/served_heads_v0.2.pt`). It is
`stage1_served.afm` with the VAD head reading block 4 only; every other tensor is identical (research/VAD_SINGLE.md:
AMI VAD F1 0.951 vs 0.949). On the quickstart clip every partial, `turn_end` and final is identical between the two;
the LID announcement comes 160 ms later, because LID pools the VAD's speech frames. The numbers on this page were
measured on `stage1_served.afm` (`audioforge-download --heads-version 0.1` rebuilds it).

`stage1_served.afm` is the measured model: `stage1_turn_v3_trail6` with `heads.spk` transplanted from
`stage1_spk_relational`; the other 747 tensors are bit-identical
([INTEGRATION.md §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26)). Older documents and the `--asr` help
text use `runs/stage1_heads_pretrained.afm`. That is the previous model: its turn head reads session frames and
never reaches θ 0.998 ([INTEGRATION.md §1](../research/INTEGRATION.md#1-executive-summary)).

The default port is 8765 in the server, `scripts/stream_client.py` and both adapters. Some research documents use
`--port 8791`, because port 8765 was taken on the measurement machine; that is an override, not a default.

## 2. Flag reference

Generated from the flag table in `audioforge/server/cli.py` (`python scripts/dev/gen_config_doc.py`; CI fails when
this block is stale). "advanced" flags are hidden from `--help` and listed by `--help-advanced`. The sections
linked in the last column explain each group and what was measured about it.

<!-- BEGIN GENERATED FLAGS: scripts/dev/gen_config_doc.py writes this block from audioforge/server/cli.py -->

**models**

| flag | default | meaning | more |
|---|---|---|---|
| `--models-dir DIR` | `$AUDIOFORGE_HOME`, else `<repo>/models`, else `~/.cache/audioforge` | where audioforge-download put the models (`audioforge-serve` only) | [§1](#1-launching) |
| `--mode {single,room}` | `single` (`room` when `--diarizer`, `--diar` or `--final-asr` is given) | preset. `single` (the default): everything from the one 115M checkpoint, for a known user: adds `--turn-input tsvad --diar-off --lid head --enroll after_agent_arm --dyn-wait-ms 2000,960`, loads no diarizer and no final-ASR worker (Silero VAD, 2 MB, when present, for `hybrid_dyn`), and refuses `--diar`, `--diarizer`, `--final-asr`, `--lid ambernet`, `--diar-embed titanet`; the voice print comes from an `enroll` message with an embedding (store >= 5 s of clean speech, 10 s for meetings), else live after `agent_end`. `room`: general diarization with NVIDIA Nemotron-3-Diarization (or `--diarizer sortformer`) next to the 115M model ([§7](#7-diarizer)). Flags you pass yourself win (`audioforge-serve` only) | [§13](#13-single-model-mode---mode-single) |
| `--diarizer {nemotron3,sortformer}` | room mode: `nemotron3` if downloaded, else `sortformer` | room mode: which downloaded diarizer to pass as `--diar`; `nemotron3` also adds `--diar-pool max --diar-left 1`. For either diarizer the launcher adds `--shed-diar hold`; flags you pass yourself win (`audioforge-serve` only) | [§7.3](#73-nemotron-3-diarization-as-the-diarizer) |
| `--asr PATH` | required (`audioforge-serve`: from the models directory) | ASR + heads `.afm` (`stage1_served_v2.afm`, the block-4 VAD build; `stage1_served.afm` = the measured v1) | [§3](#3-models-threads-and-speed) |
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
| `--dyn-wait-ms CAP,FLOOR` | the served rule: 6000,1600 | `hybrid_dyn`: the Silero-silence wait at head posterior 0 (CAP) and 1 (FLOOR), linear in between (plus the rule's offset); `--mode single` sets `2000,960` (research/SINGLE_MODEL.md A1) (advanced) | [§4](#4-turn-policies-configturn_policy) |

**transcripts**

| flag | default | meaning | more |
|---|---|---|---|
| `--final-asr SPEC` | off | offline per-turn final ASR: `tdt_v3` (NVIDIA Parakeet-TDT 0.6B v3) or a `.nemo` path; adds finals with `source: tdt_v3` | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-worker {process,thread}` | `process` | --final-asr: child process or thread (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-threads N` | `2` | --final-asr: torch threads of the offline model (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--final-asr-device DEV` | `cpu` | --final-asr: cpu &#124; mps &#124; cuda (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--asr-lookahead R` | off | second, text-only ASR pass at attention context [70, R], e.g. 13 (advanced) | [§8](#8-final-asr-and-dual-lookahead) |
| `--asr-vad-gate P` | off | do not decode transducer tokens on frames whose served VAD <= P once `--asr-vad-hangover-ms` of such frames have passed (bounds hallucinated text on long non-speech; research/BULLETPROOF.md) (advanced) |  |
| `--asr-vad-hangover-ms MS` | `1200.0` | --asr-vad-gate: decoding continues this long after speech (advanced) |  |

**language ID**

| flag | default | meaning | more |
|---|---|---|---|
| `--lid head&#124;PATH&#124;ambernet` | off | spoken language ID: `head` (the shipped distilled head on the shared encoder, `lid_distill.pt`), a head file (`audioforge.lid.save_head`), or `ambernet` / an AmberNet `.nemo`; emits `language` messages and `stats.lang` (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-threshold P` | `0.9` | --lid: posterior needed to announce a language (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-min-ms MS` | `1000.0` | --lid: pooled speech needed before the first announcement (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-max-ms MS` | 3000 with `--lid head`, else off (0 = off) | --lid head/file: announce the top language after this much speech anyway (advanced) | [§9](#9-spoken-language-id---lid) |
| `--lid-langs CODES` | the 17 languages of research/LID.md | --lid ambernet: comma-separated language codes to choose from (advanced) | [§9](#9-spoken-language-id---lid) |

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
| `--perf SPEC` | `default` | CPU inference fast paths of `audioforge.perf`: `default` = the exact set (same outputs), `none`, `all` (adds float-rounding ones), or a list such as `default,-linear_t` (research/PERFORMANCE.md) (advanced) | [§3](#3-models-threads-and-speed) |
| `--device DEV` | `cpu` | cpu, or cuda / cuda:N (opt-in; others fall back to cpu) (advanced) | [§3](#3-models-threads-and-speed) |
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

**`--threads`** sets the torch threads of the single worker thread that runs every session's compute. Measured
(server alone, 1.04 s diarizer setting, `stage1_heads_pretrained`, 1x, load 3-6, n = 3 AMI windows + 1 LibriSpeech
utterance per thread count; [INTEGRATION.md §4](../research/INTEGRATION.md#4-measured-tables)):

| threads | RTF | chunk ms p50 / p95 | max backlog ms |
|---|---|---|---|
| 2 | 0.433-0.520 | 33-45 / 222-260 | 160-200 |
| 4 | 0.341-0.422 | 31-39 / 114-202 | 80-140 |

With the current served stack (`stage1_served`, 0.32 s Sortformer, `--turn-input diar`), the server's RTF was
0.795 / 0.802 (Pipecat / LiveKit sessions) at 2 threads: ASR pass 0.149-0.151, turn pass 0.150, diarizer
0.496-0.500. Peak RSS was 3581 / 3586 MB. Scope: 69 clip-conditions, 3489 s of audio per framework
([E2E_FINAL.md §7](../research/E2E_FINAL.md#7-rtf-by-component)). Concurrent sessions share the one worker, and
concurrency has not been load-tested ([INTEGRATION.md §6](../research/INTEGRATION.md#6-product-readiness)).

**`--perf`** (2026-09-28). The exact fast paths of `audioforge.perf` are on by default: column-major storage of the
encoder / diarizer `nn.Linear` weights (Accelerate runs the 2-row GEMMs of a 160 ms step 3-4x faster in that
layout), a cache of the relative-position projection per attention layer, one shared subsampling for the ASR and
the speaker-conditioned turn pass, and a cached RNNT prediction projection between emitted tokens. Protocol messages
are identical to `--perf none` (2 of 1112 `eot` values differ in the 5th decimal on the 5 AMI windows, everything
else bit-identical); the per-component costs before and after are in
[PERFORMANCE.md §2-3](../research/PERFORMANCE.md). `--perf none` restores the 2026-09-27 code path.

**`--no-fast-conv`.** By default the conformer convolutions of both encoders are re-bound to an equivalent
unfold / `F.linear` CPU path. Tokens stay identical, and diarizer probabilities differ by at most 3e-7. This
path is what lets the server keep up: RTF 0.45-0.52 with it, 1.13-1.21 without it (the server then falls behind);
2.5x end to end, verified, 2 threads, 1x
([INTEGRATION.md §1](../research/INTEGRATION.md#1-executive-summary)). Use the flag only for debugging.

**`--device`.** `cpu` (the default, and what every number in `research/` was measured on). `cuda` / `cuda:N`
(opt-in) runs the same streaming code with the models on an NVIDIA GPU: fast-conv (a CPU path) is off, and the
per-frame decoding stays in Python. On the bundled clips it emits the same turn_ends and finals as `cpu`, in single
and room mode. On an RTX 5090 it cut the 160 ms block compute from 77 to 20 ms (p50) in single mode and the RTF from
1.02 to 0.16 in room mode ([GPU_RUN_2026-09-29.md](../research/GPU_RUN_2026-09-29.md)). Other devices (`mps`, or
`cuda` without a visible GPU) fall back to `cpu`. `--final-asr-device` can put the offline final-ASR model on `mps`
or `cuda`.

**`--no-warmup`.** By default one throwaway session runs 2 s of low-level noise through both models at start, so
the first real session does not pay one-time initialization. The effect of skipping it was not measured.

## 4. Turn policies (`config.turn_policy`)

A session picks its end-of-turn rule in its first `config` message. The default is `timeout`. **Primary** means
the diarizer column with the most activity (sum of probabilities) over the last 5 s, or the bound column under
`--enroll` ([§6](#6-primary-speaker-enrollment---enroll)). A column is active on a frame when p > 0.5.

| policy | rule | config keys used | finals cut by |
|---|---|---|---|
| `timeout` (default) | `turn_end` when the primary column has been inactive for ≥ `timeout_ms` and has spoken since the last timeout `turn_end`. Other columns are ignored | `timeout_ms` (1000) | the timeout |
| `timeout_quiet` | the same, but also waits until no other column is active on that frame. Its `turn_end` is tagged `timeout` | `timeout_ms` | the timeout |
| `head` | `turn_end` when the turn head's probability crosses `eot_threshold` upwards. Re-armed by speech (VAD > 0.5); fires at most once per speech segment | `eot_threshold` (0.98) | the head |
| `both` | the `timeout` rule and the `head` rule both fire, each tagged with its own name | `timeout_ms`, `eot_threshold` | the timeout |
| `hybrid` | primary-silence timeout OR head ≥ `eot_threshold`. One `turn_end` per turn, at the earlier of the two decision times, tagged `hybrid`; the later path's firing for the same turn is dropped | `timeout_ms`, `eot_threshold` (0.98) | the hybrid event |
| `hybrid_silero` | head ≥ θ OR **any-speaker Silero VAD v5 silence** ≥ `--silero-timeout-ms` (2640 ms). Silero runs on 32 ms chunks through Pipecat's VAD state machine (confidence 0.7, start / stop 0.2 s). One firing per silence run | `eot_threshold` (default θ 0.99828); `timeout_ms` is not used | the hybrid event |
| `hybrid_dyn` | head ≥ θ OR Silero silence ≥ clamp(80 − 55 p, 7, 80) frames, where p is the head's posterior on the same frame: 6.4 s at p = 0, 4.24 s at p = 0.5, 2.48 s at p = 0.9, 2.0 s at p = 1 | `eot_threshold` (default θ 0.998283); `timeout_ms` is not used | the hybrid event |

Notes:

- `timeout_ms` is clamped to [80, 4840]. With the default 1000 ms, 13 silent frames are needed. The decision also
  waits for the diarizer column to be finalized, so `turn_end.t` is at least the last primary frame + 1.0 s + the
  column lag (0.08-0.24 s with the default diarizer setting) ([INTEGRATION.md §2](../research/INTEGRATION.md#2-architecture)).
- `hybrid_silero` / `hybrid_dyn` must be in the first `config`, before any audio. They load the Silero ONNX once per
  server: one shared session, about 0.11 ms per 32 ms chunk (RTF +0.003) and about +24 MB RSS
  ([INTEGRATION.md §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26)). They need `onnxruntime`
  (`audioforge[serve]`).
- Setting `eot_threshold` in the config replaces the frozen θ of `hybrid_silero` / `hybrid_dyn`.
- `--silero-timeout-ms` is a server-wide flag, not a per-session config key.

### 4.1 Measured tradeoffs

Benchmark rows use the served turn head (`stage1_turn_v3_trail6`, identical in `stage1_served`) unless noted.
Product rows use the served model and the values in the row.

| policy | benchmark: miss at ≤ 5 % FC | product (live, Pipecat / LiveKit) | sources |
|---|---|---|---|
| `timeout` | eot-bench v2, AMI dev n = 974, 6 s horizon, causal binding: **74.8 % [72.0, 77.5]** (floor-open 46.1 %) on 1.04 s Sortformer tracks; 70.1 % [67.2, 72.9] (open 36.5 %, held-out FC 8.4 %) on 0.32 s tracks. 2 s horizon: 92.5 %. STAGE1, AMI dev n = 200, column chosen with the oracle primary (optimistic): 38.4 % at k 21 frames (P50 2640 ms) | 1000 ms, 5 AMI windows, 0.32 s diarizer: median dead air **1923 ms / 1684 ms**, **3 / 5 cut-ins**. E2E, TurnBench dev user channel (16 clips): 1353 / 1330 ms, 1.32 / 2.22 cut-ins per min, 23 % / 23 % missed within 3 s | [EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26), [STAGE1](../research/STAGE1.md#consolidated-turn-taking-results-2026-09-26-ami-dev-n200-misses-at-5-per-turn-false-cutoffs), [INTEGRATION §4](../research/INTEGRATION.md#default-preset-low_latency_032-032-s-pipecat-and-livekit-dead-air-re-run-2026-09-26), [E2E_FINAL §4](../research/E2E_FINAL.md#4-results-by-clip-set) |
| `timeout_quiet` | STAGE1, AMI dev n = 200, Sortformer streaming (1.04 s): "primary silent & nobody else" misses **66-68 %**, against 38.4 % for the plain timeout on the same track | not measured | [STAGE1 turn head v3](../research/STAGE1.md#turn-head-v3-2026-09-26-all-four-diarizer-columns--silence-counters--future-activity-aux-trained-on-streaming-tracks) |
| `head` | eot-bench v2, n = 974, 6 s, causal: **66.5 % [63.4, 69.7]** (open 57.3 %) on 1.04 s tracks; 65.8 % [62.6, 68.7] (open 55.0 %) on 0.32 s tracks. 2 s horizon: 79.8 % | θ 0.98, `--turn-input diar`, 5 AMI windows: Pipecat delivered a turn end on only 2 of 5 windows (median 1662 ms); LiveKit median 1237 ms on 4 of 5, 1 cut-in, plus firings inside the user's speech | [EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26), [INTEGRATION §4b](../research/INTEGRATION.md#4b-served-stage1_turn_v3_trail6-head-with-diarizer-input---turn-input-diar-032-s-preset-2026-09-26) |
| `both` | not measured as a rule (it is `timeout` for the finals plus extra `head` events) | not measured | |
| `hybrid` | eot-bench v2, n = 974, 6 s, causal: **61.9 % [58.7, 65.0]** (open 45.9 %) on 1.04 s tracks, at the fitted (θ ≈ 0.998, k 49-52 frames ≈ 4 s); 63.7 % [60.5, 67.0] on 0.32 s tracks. With oracle speaker activity (n = 200): 1.6 % miss at P50 560 ms | θ 0.998 / 1000 ms fires exactly when the timeout fires: same dead air and the same 3 / 5 cut-ins. θ 0.998 / 4000 ms: 0 cut-ins, median dead air 4522 ms (Pipecat) | [EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26), [TURN_ERRORS §8](../research/TURN_ERRORS.md#8-hybrid-decision-rule-joint-θ-k-sweep-2026-09-26), [INTEGRATION §4b](../research/INTEGRATION.md#4b-served-stage1_turn_v3_trail6-head-with-diarizer-input---turn-input-diar-032-s-preset-2026-09-26) |
| `hybrid_silero` | held-out ICSI, n = 1312, 6 s, frozen AMI point (θ 0.998285, k 33 = 2.64 s): **81.7 % [79.5, 83.9]**, floor-open **62.2 %**, FC 2.4 % per turn / 0.6 % per pause. The hybrid on the same set: 82.1 % / open 71.5 % (open Δ −9.3 [−14.7, −4.1]). AMI dev (exploratory, rule picked after seeing the table): 59.6 % [56.3, 62.6], open 34.3 %, held-out FC 6.6 % | 5 AMI windows: median dead air **2941 / 2921 ms** against the timeout's 1702 / 1676 ms (about +1.2 s), **0 / 0 cut-ins** against 3 / 5 | [BASELINES ICSI](../research/BASELINES.md#turn-detection-icsi-held-out-confirmation), [BASELINES turn detection](../research/BASELINES.md#turn-detection), [INTEGRATION §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26) |
| `hybrid_dyn` | held-out ICSI, n = 1312, 6 s, frozen AMI point: **79.5 % [77.2, 81.7]**, floor-open **50.5 % [44.5, 56.7]**, FC 1.8 % / 0.5 %; against `hybrid_silero` −2.2 [−3.2, −1.3] all, −11.7 [−17.0, −7.0] open. It is the best all-ends row on ICSI at any frozen point. AMI dev: 58.2 % (open 31.5 %) at 6.1 % held-out FC | 5 AMI windows: median dead air **2662 / 2425 ms** (+0.66 / +0.75 s over the timeout), **0 / 0 cut-ins**. E2E with `after_agent_arm` (system D): 0.00 cut-ins per min on AMI in both frameworks; 0.32-0.98 s more median dead air than `timeout` | [BASELINES dynamic timeout](../research/BASELINES.md#dynamic-timeout-icsi-held-out-confirmation), [INTEGRATION §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26), [E2E_FINAL §8](../research/E2E_FINAL.md#8-verdicts) |

Reading:

- **Why `timeout` is the default.** No rule measured so far wins on both axes. The shipped hybrids make the product
  respond later than the 1000 ms timeout (+0.66 to +1.2 s median dead air on 5 AMI windows) and cut in less (0 vs
  3 / 5). Their benchmark gains are on turn ends that the timeout never catches
  ([INTEGRATION.md §8 verdict](../research/INTEGRATION.md#8-shipped-rules-2026-09-26)). Choose `timeout` when
  responsiveness matters most, and `hybrid_dyn` when not interrupting the user matters most.
- The Silero branch is speaker-unaware. On a floor-open end, silence from everyone is the right signal. At a pause
  where nobody else speaks, the branch sees the same silence. This is why `hybrid_silero` costs +1.4 points on taken ends on
  ICSI ([BASELINES](../research/BASELINES.md#turn-detection-icsi-held-out-confirmation)).
- `hybrid` pays off only with a clean speaker track. On the Sortformer streaming track the head adds about 1 point
  over the timeout ([TURN_ERRORS §8](../research/TURN_ERRORS.md#8-hybrid-decision-rule-joint-θ-k-sweep-2026-09-26)).
- Label-free primary selection is the dominant loss for every speaker-track rule: with oracle binding the same
  benchmark gives timeout 32.0 %, head 30.3 %, hybrid 28.7 % at 6 s
  ([EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26)).
  `--enroll after_agent_arm` / `after_agent` address this ([§6](#6-primary-speaker-enrollment---enroll)).
- The timeout still ends turns during filled pauses ("um") or when the user resumes within 1 s. On AMI a 1 s pause is
  usually a hesitation, not a hand-over ([INTEGRATION.md §1](../research/INTEGRATION.md#1-executive-summary)).

## 5. Turn head input (`--turn-input`)

| value | behaviour | cost |
|---|---|---|
| `auto` (default) | `diar` if the model's turn head reads diarizer activity or columns, else `session`. For `stage1_served.afm` (and `stage1_turn_v3_trail6.afm`) the head has `needs_cols`, so `auto` selects `diar` | |
| `session` | the head reads the session's own encoder frames (no second encoder pass). No speaker information reaches it. The stage-1 kernel head was trained with speaker conditioning, so this input is off its training distribution. It is refused (start-up error) for heads that read the diarizer's activity themselves, including the served one | none extra |
| `tsvad` | the head reads the TS-VAD track of the enrolled user's voice print ([P(user), P(other), 0, 0], primary 0) from a 0.26 M head on block 4 of the ASR pass, on the 160 ms chunk clock (no diarizer lag). With `--diar-off` the track also replaces the diarizer's columns; `--mode single` sets both ([§13](#13-single-model-mode---mode-single)) | the turn pass (RTF about 0.15) as for `diar`; the TS-VAD head is negligible |
| `diar` | the head runs on the diarizer's clock: frame v once diarizer column v exists. For speaker-kernel heads a second, speaker-conditioned encoder pass runs over the same mel chunks; concat / v3 heads receive the activity, columns and primary from the diarizer | turn pass RTF 0.150 with `stage1_served` ([E2E_FINAL §7](../research/E2E_FINAL.md#7-rtf-by-component)); with the older `stage1_heads_pretrained`, total RTF 0.58-0.61 vs about 0.5 ([EARLY_RESULTS](../research/EARLY_RESULTS.md#live-streaming-server-2026-09-26)) |

With `diar`, `eot` and head decisions also carry the diarizer column lag: the head's decision time is the later of
its ASR chunk and the diarizer column's finalization.

## 6. Primary-speaker enrollment (`--enroll`)

Which diarizer column is "the user" decides what `timeout`, `timeout_quiet`, `both` and `hybrid` listen to. It also
decides what a `--turn-input diar` head is conditioned on and what `final.speaker` reports. The Silero branch of
`hybrid_silero` / `hybrid_dyn` does not depend on it. The head branch does.

| mode | client message | rule | cost |
|---|---|---|---|
| `dominant` (default) | none | the column with the most activity over the last 5 s; `null` if none was active | none |
| `after_agent_arm` | `{"type": "agent_end"}` when the agent's TTS ends | from that audio position, the first column active for ≥ 3 consecutive frames is bound. Afterwards the causal_dominant rule runs, seeded at that column: it re-binds only after > 25 silent frames of the bound column, to the column with the most active frames over the last 25. Each `agent_end` re-arms. No TitaNet | none measurable: arm binder RTF 0.000 ([E2E_FINAL §7](../research/E2E_FINAL.md#7-rtf-by-component)) |
| `after_agent` | `{"type": "agent_end"}` | the same choice. The chosen column's first 19 active frames (1.5 s) become a TitaNet-L enrollment. Afterwards the primary follows that voice: every `--enroll-stride` frames, each column with ≥ 8 active frames in the last 2 s is embedded. Hysteresis: margin 0.1 cosine, hold 6 frames | TitaNet-L +91 MB RSS on top of the loaded server (+201 MB in a fresh process). Binder 1.3-13 ms per diarizer frame. RTF 0.87-1.16 vs 0.60-0.84 without it (3 AMI dev windows, 2 threads, 0.32 s diarizer, load 3-6): at the edge of keeping up on 2 threads ([INTEGRATION §7](../research/INTEGRATION.md#7-primary-speaker-enrollment---enroll-after_agent--explicit-2026-09-26)) |
| `explicit` | `{"type": "enroll"}` (e.g. after a "say something" prompt) | as `after_agent`, started by `enroll`; one enrollment per message | as `after_agent` |

Until a column is chosen, every mode uses the `dominant` rule. The modes other than `dominant` add
`enroll` / `enrolled` / `primary_column` to `ready` and `stats` and send an `enrolled` message
([PROTOCOL.md §5.6](PROTOCOL.md#56-enrolled---enroll-other-than-dominant)). `--titanet` sets the TitaNet-L `.nemo`
(default `data/nemo/speakerverification_en_titanet_large.nemo`; the launcher fills it from the models directory).
`--enroll-stride` (default 5 frames = 400 ms) sets how often the voice modes re-embed; `--enroll-stride 10` halves
the following cost at the price of slower switching, and a third thread is the other remedy (not measured
numerically; [INTEGRATION §7](../research/INTEGRATION.md#7-primary-speaker-enrollment---enroll-after_agent--explicit-2026-09-26)).

Measured effect (eot-bench v2, AMI dev n = 974, head trail6, 1.04 s Sortformer tracks, cross-fitted ≤ 5 % FC; "agent
end" = the previous other speaker's labelled turn end, a stand-in for the TTS end;
[EOT_BENCH_V2 §9](../research/EOT_BENCH_V2.md#9-who-is-the-primary-speaker-titanet-l-following-an-agent-end-identity-rule-and-enrollment-length-2026-09-26)):

| binding (server mode) | hybrid, 6 s | Δ vs causal_dominant, 6 s | hybrid, 2 s | Δ, 2 s | timeout, 6 s |
|---|---|---|---|---|---|
| causal_dominant (the analogue of `dominant`) | 61.9 % [58.7, 65.0] | ref | 79.1 % | ref | 74.8 % |
| after_prev_end_causal (`after_agent_arm`) | 56.0 % [52.5, 59.1] | −5.9 [−8.6, −3.5] | 75.3 % | −3.9 [−6.6, −1.4] | 67.6 % |
| after_prev_end_titanet (`after_agent`) | 51.6 % [48.5, 54.7] | −10.3 [−13.5, −6.9] | 87.6 % | +8.5 [+6.2, +10.9] | 53.7 % |
| voice_explicit_titanet (`explicit`; enrollment timed on the true primary's first utterance) | 47.1 % [43.8, 50.4] | −14.8 [−17.6, −11.6] | 81.0 % | +1.8 [−0.5, +4.0] | 52.9 % |
| oracle column (upper bound) | 28.7 % [25.8, 31.6] | | 53.1 % | | 32.0 % |

- `after_agent_arm` is the only new binding that is also better at the 2 s horizon. TitaNet following
  (`after_agent`) catches the same turns 2-6 s late: it is a loss for dead air and a gain for the 6 s miss rate. The
  benchmark's verdict: arm at the agent end in every case, and enable TitaNet following only where the 6 s miss rate
  matters more than dead air.
- Product test (5 AMI windows, [INTEGRATION §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26)):
  `after_agent_arm` changed nothing measurable. Timeout decisions were unchanged, there was one fewer LiveKit cut-in
  (5 to 4), and the hybrids were 240 ms later on one window.
- Send `agent_end` exactly when the TTS playback ends. If it is sent while the agent is still audible, the agent's
  own column can be chosen (tested in `tests/test_serve.py`).

## 7. Diarizer

### 7.1 `--diar-config` and `--diar-set`

Two NVIDIA model-card settings of Streaming Sortformer v2 (`streaming_diar.SORTFORMER_PRESETS`). Both use FIFO 188,
speaker-cache update period 144, speaker cache 188 and encoder left context 188 frames:

| preset | card name | chunk / right context (frames) | input buffer | column finalized after frame end | `ready.column_lag_ms` |
|---|---|---|---|---|---|
| `low_latency_032` (default since 2026-09-26) | ultra low latency | 3 / 1 | 0.32 s | 80-240 ms (mean 160) + compute | 160.0 |
| `low_latency` | low latency | 6 / 7 | 1.04 s | 560-960 ms (mean 760) + compute | 760.0 |

Measured:

| | `low_latency_032` | `low_latency` | scope, source |
|---|---|---|---|
| turn-end miss, 6 s, causal: hybrid / head / timeout | 63.7 / 65.8 / 70.1 % | 61.9 / 66.5 / 74.8 % | eot-bench v2, AMI dev n = 974; CIs overlap; the timeout's apparent gain at 0.32 s comes with held-out FC 8.4 % ([EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26)) |
| P50 at 6 s where systems fire (oracle timeout / head, oracle) | 2720 / 2320 ms | 3200 / 2960 ms | same |
| nominal emission delay | 240 ms | 840 ms | same |
| DER (miss / FA / confusion), 6 s windows, 4 columns pooled | 28.3 (17.6 / 5.0 / 5.7) | 26.3 (16.4 / 5.0 / 4.9) | same |
| server RTF, 2 threads, 1x | 0.49 / 0.53 | 0.40 / 0.43 | 2 AMI dev windows, load 1.5-2.4 ([EARLY_RESULTS](../research/EARLY_RESULTS.md#live-streaming-server-2026-09-26)) |
| speaker-column age on arrival, p50 (p95) | 210-227 (291) ms | 857 (1035) ms | same |
| max backlog | 120-140 ms | 120 ms | same |
| product dead air, `timeout` 1000, back to back | IS1008b_003 1602 ms, ES2011b_027 1661 ms (Pipecat) | 1821 ms, 2402 ms | 2 AMI windows, load 3.7-4.5 ([INTEGRATION §4](../research/INTEGRATION.md#default-preset-low_latency_032-032-s-pipecat-and-livekit-dead-air-re-run-2026-09-26)) |

The 0.32 s setting is a latency gain, not an accuracy gain. Miss rates stay within the 1.04 s CIs, DER is 2 points
worse, and it costs about +0.1 RTF (the diarizer runs about 3x as often). Transcripts are identical at both settings.
`low_latency` is the setting that STAGE1, TURN_ERRORS and EOT_BENCH_V2 §9 scored; `low_latency_032` is what the
product tests (INTEGRATION §4 and §8, E2E_FINAL) ran.

**`--diar-set KEY=VALUE`** (repeatable) overrides one streaming field on top of the preset. Keys are fields of
`streaming_diar.AOSCConfig`: `chunk_len`, `chunk_right_context`, `fifo_len`, `spkcache_len`,
`spkcache_update_period`, `sil_frames`, `strong_boost_rate`, `threshold`. A value containing `.` is parsed as a
float, anything else as an integer. Any override appends `+custom` to `ready.diar_config`, and `column_lag_ms`
follows the new chunk / right context. No custom setting has been measured.

### 7.2 `--diar-left`

Encoder left context of the diarizer in window mode, in frames (default 188, the value both presets were measured
with). Sortformer's encoder is non-causal, so each step re-encodes [left context, chunk, right context]. A shorter
left context is cheaper; for Sortformer v2 its effect on accuracy was not measured. For Nemotron-3-Diarization the
encoder is frame-local, so a larger left context only costs time, and 1 is the recommended value
([SORTFORMER_IMPORT.md](../research/SORTFORMER_IMPORT.md#patch-intent-files-owned-by-other-agents)).

### 7.3 Nemotron-3-Diarization as the diarizer

`--diar` also accepts the Nemotron-3-Diarization import (`nemo_nemotron3_diar.afm`: 99.2 M parameters, OpenMDW-1.1
licence, 8 output columns at 10 ms). Two flags exist only for it:

- `--diar-pool {mean,max}`: pooling of its 10 ms outputs to the 80 ms frame. The import's default is `mean`. `max`
  marks a frame active if any 10 ms sub-frame is active, which matches the word-level labels used here.
- `--diar-spks K`: keep the first K of its 8 arrival-order columns (dropped, not merged). `frame.speakers` carries
  as many columns as the diarizer has (4-8, PROTOCOL.md §5.2), so the cut is no longer needed; `--diar-spks 4`
  restores the E2E-measured configuration.

The measured configuration is `--diar nemo_nemotron3_diar.afm --diar-pool max --diar-left 1 --diar-spks 4` with the
default `--diar-config` (E2E systems CN / DN, [E2E_FINAL §2](../research/E2E_FINAL.md#2-systems)):

| | Sortformer v2 (0.32 s) | Nemotron-3, max pool | scope, source |
|---|---|---|---|
| server RTF (session wall) | 0.795 / 0.802 | 0.635 / 0.641 | E2E, 2 threads, Pipecat / LiveKit ([§7](../research/E2E_FINAL.md#7-rtf-by-component)) |
| diarizer RTF | 0.496 / 0.500 | 0.300 / 0.302 | same |
| server peak RSS | 3581 / 3586 MB | 1485 / 1482 MB | same |
| product dead air with `timeout` (CN − C) | | 8-135 ms lower on the two-party sets; CIs exclude 0 in 6 of 8 cells | [E2E_FINAL §8](../research/E2E_FINAL.md#8-verdicts) |
| streaming DER, 1.04 s buffer, AMI dev 64 × 20 s | 0.252 | 0.241 | [SORTFORMER_IMPORT](../research/SORTFORMER_IMPORT.md#comparison-ami-dev-headset-mix-word-level-labels-80-ms-frames-threshold-05-no-collar) |
| primary-track miss (offline, 200 turn windows) | 0.129 | 0.192 | same |

E2E_FINAL recommends Nemotron-3 as the default diarizer. Its caveats: it was scored there only through the turn
policy (no DER on those clips), and it runs in `--diar-spks 4` mode. Its model card lists AMI train + dev in its
training data, and it misses more speech on this project's labels. The server applies the Sortformer v2 presets
(`--diar-config`) to it. Its own card profile (chunk 9, right context 4, FIFO 264, cache 264, update 222) has not
been measured inside the server. The launcher's `--diarizer nemotron3` adds `--diar-pool max --diar-left 1` (`audioforge/launch.py`); it added
`--diar-spks 4` as well until 2026-09-28 ([§7.4](#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any)).

### 7.4 Multi-speaker rooms: `--diar-labels`, `--shed-diar`, `timeout_any`

Added 2026-09-28 after the report "many speakers in the room doesn't work" ([research/DIARIZATION_FIX.md](../research/DIARIZATION_FIX.md)). Three
independent causes, three flag-gated fixes; the defaults keep the previous behaviour:

- **`final.speaker` was the 5 s dominant column, not the turn's speaker**, and a turn only ended when that column fell
  silent, so other people's turns were glued to the primary's segment (per-final purity 0.69-0.85 on 4-6 speaker
  clips). `--diar-labels registry` labels each final by voice: the column that dominates the *turn's own span* picks
  the speaker's frames, the served speaker head (`heads.spk`, block-4 frames the ASR pass already produced) embeds
  them, and a per-session registry (`audioforge.speaker_registry.SpeakerRegistry`) returns the id of the closest
  known speaker when the cosine is at least the threshold, else a new id. Ids are stable across the diarizer's column
  permutations and re-entries and are not limited to the column count; `stats.speakers_seen` is the speaker-count
  estimate; `final.speaker_conf` is the match cosine. `--diar-embed titanet` uses TitaNet-L per turn instead (one
  embedding per final, not per frame; +91 MB). Session config `turn_policy: "timeout_any"` (before the first audio)
  ends every speaker's turn: `timeout_ms` of nobody talking, or a speaker change once the previous speaker has been
  silent for 240 ms and the new one active for 240 ms (`turn_end.policy: "change"`); overlap and short backchannels
  do not cut. Keep the default `timeout` for a voice agent that must follow one user.
- **`--diar-spks 4` hid speakers 5-8** of Nemotron-3-Diarization (the launcher added it; the cut columns are dropped,
  not merged). The launcher no longer adds it; `frame.speakers` then carries 8 columns (PROTOCOL.md §5.2).
- **Load shedding relabelled every turn to speaker 0** (level 1 replaced the diarizer by the served VAD in column 0).
  `--shed-diar hold` keeps the last stable column active while the VAD hears speech (silence stays silence, so the
  timeout keeps firing), runs the diarizer on every other block at level 1 (half its cost) instead of not at all,
  and flags the affected finals with `diar_shed: true`; with `registry` the voice still identifies the speaker.

Measurements (before / after, with and without shedding, server RTF) are in DIARIZATION_FIX.md sections 3-4.

## 8. Final ASR and dual lookahead

The streaming model keeps every live decision (partials, VAD, `turn_end`, speakers, enrollment). Two opt-in extras
send additional finals for each cut turn. They are distinguished by `final.source`
([PROTOCOL.md §5.5](PROTOCOL.md#55-final)).

**`--final-asr tdt_v3`** transcribes each finished turn once with NVIDIA Parakeet-TDT 0.6B v3 (CC-BY-4.0; a
2.5 GB download: `audioforge-download --with tdt_v3`). A `.nemo` path loadable by `audioforge.nemo_import` also
works. The finals then carry that file's stem as `source`. The turn's audio runs from 0.3 s before the served VAD's
speech onset (never before the previous turn's end) to 0.5 s after the last VAD speech frame, at most 60 s.

| flag | default | meaning |
|---|---|---|
| `--final-asr-worker` | `process` | `process`: a spawned child process with its own torch threads and its own GIL, so the offline decode never blocks the streaming loop. `thread`: one worker thread in the server process, which shares the GIL and the torch pool with the streaming worker |
| `--final-asr-threads` | `2` | torch threads of the offline model |
| `--final-asr-device` | `cpu` | `cpu`, `mps` or `cuda` |

Jobs run one at a time, in submission order. `stats` waits for every pending offline final and reports
`final_latency_ms` and `final_asr_rss_mb`. Tests check that frame messages keep flowing (gaps < 0.6 s) while a
1.2 s offline pass runs, in both worker modes (`tests/test_hybrid_asr.py`).

**`--asr-lookahead R`** runs a second, text-only cache-aware pass of the same ASR model at attention context
[70, R]. With R = 13 that is 1.12 s chunks and 1.04 s of lookahead. Its finals (`source: "lookahead"`) are due 3
frames after the segment's last VAD speech frame; their `latency_ms` is the audio-time wait past the decision time.
Heads, events and partials are unchanged: `tests/test_hybrid_asr.py` checks that they are identical to the
single-pass server. The pass costs one more encoder pass per chunk. Both flags can be combined.

Measured accuracy and latency (`runs/hybrid_asr.json`, `runs/final_asr.json`; [FINAL_REPORT §1](../research/FINAL_REPORT.md#1-asr)
and the `research/HYBRID_ASR.md` draft): on AMI dev, 200 single-speaker segments, `normalize_text` WER, the streaming
model alone has 24.4 % (20.6 % with the Whisper normaliser); Parakeet-TDT 0.6B v3 run offline on each turn, i.e.
what `--final-asr tdt_v3` produces, has 9.72 % [8.22, 11.54] (9.50 % Whisper-normalised), a paired difference of
−14.70 [−17.02, −12.71] points, at RTF 0.065 per turn on 2 CPU threads (batch 1, machine under load) and about
2.5 GB of RSS in the worker. On LibriSpeech test-clean (first 200 utterances) the [70,13] lookahead pass of the same
115M model is 1.92 % against 2.29 % for the [70,1] streaming pass; its AMI WER was not measured. On the TurnBench
user channel the streaming model has 17 % WER against 10 % for Whisper small ([E2E_FINAL §1](../research/E2E_FINAL.md#1-summary)).
TDT v3 limits (its card): 25 European languages, no Hebrew, Arabic, Turkish, Persian, Hindi, Chinese or Japanese.

## 9. Spoken language ID (`--lid`)

Off by default. With `--lid`, the server sends `language` messages and adds `stats.lang`. It announces the top
language once its posterior reaches `--lid-threshold` (default 0.9) after at least `--lid-min-ms` (default 1000) of
pooled speech, and again whenever the confident top language changes. With `--lid-max-ms MS` (default 3000 for
`--lid head`, off otherwise, `0` = off) it announces the top language anyway once that much speech has been pooled
without a confident call.

| backend | how | accuracy (FLEURS test, 17 languages, n = 2550): 2 s / full utterance | streaming rule, test split, threshold 0.9: first announcement correct / at p50 / flips per utterance | accented English (EdAcc, n = 257): share classified "en", full | cost |
|---|---|---|---|---|---|
| **`--lid head`** (`lid_distill.pt` in the models directory, else `runs/`) | a 0.92 M-parameter head on blocks 8-12 of the shared encoder pass, distilled from AmberNet on all of FLEURS-17 train plus accented English (AMI / ICSI / LibriSpeech); VAD-gated pooling, 30 s half-life, no extra encoder pass | **91.2 % / 97.7 %** | 91.8 % / 2.40 s / 0.05 (rule "0.9 or after 3 s"); 94.2 % / 2.32 s / 0.04 with `--lid-max-ms 0` (1.9 % never announced) | **65.8 %** | 0.21 ms per 160 ms chunk (RTF 0.0013) |
| head file, e.g. `--lid runs/lid_aug.pt` | a 0.40 M-parameter head on the served encoder's per-layer outputs. It pools VAD speech frames only, with a 30 s half-life, and needs no extra encoder pass | 75.0 % / 89.3 % (`lid_aug`); 72.6 % / 87.7 % (`lid_head`) | 78.7 % / 2.40 s / 0.17 (`lid_aug`) | 18.7 % (`lid_aug`), 31.1 % (`lid_head`) | 0.16 ms per 160 ms chunk (RTF 0.001) |
| `--lid ambernet` (or an AmberNet `.nemo`) | NVIDIA AmberNet (28.9 M parameters, NGC Terms of Use). It re-classifies the last 8 s of VAD speech at 1, 1.5, 2, 3, 5 and 8 s of pooled speech, then every 4 s, restricted to `--lid-langs` | 95.1 % / 99.5 % | 96.8 % / 2.16 s / 0.03 | 67.7 % | 16-81 ms per call for 1-8 s of speech; RTF about 0.02-0.03 at the served schedule |

Source: [LID.md fix pass](../research/LID.md#fix-pass-2026-09-29) for `--lid head` (it misses the pre-registered
bar of 92 % / 98 % on FLEURS by under a point and meets the EdAcc and cost bars, so LID stays off by default);
[LID.md §3](../research/LID.md#3-head-vs-dedicated-models-fleurs-test-17-languages-n--2550),
[§4](../research/LID.md#4-streaming-the-servers-announcement-rule), [§5](../research/LID.md#5-cost-cpu-2-threads-this-mac),
[verdict](../research/LID.md#verdict-and-recommendation). The recommendation there: use `--lid ambernet` with
`--lid-langs` restricted to the languages the product supports. For mostly non-native English speakers, restrict the
label set and raise `--lid-threshold`, because AmberNet calls about a third of accented English segments another
language. Since the fix pass, `--lid head` is the no-second-model option: 3.9 points behind AmberNet at 2 s, 1.8 on
full utterances, level on accented English, at 1 / 100 of its cost. Its weakest languages are Ukrainian (58 % at 2 s,
mostly called Russian) and German (70 %). It is trained on AmberNet's outputs (NGC Terms of Use): check those terms
before redistributing it.

`--lid-langs` (AmberNet only) defaults to the 17 languages
`en, he, ar, ru, es, fr, de, pt, it, nl, pl, uk, tr, fa, hi, zh, ja`; the head's label set is the same 17 and fixed
by the head file.

## 10. Diagnostics

`--debug-fields` adds diagnostic keys to `frame` (`spk_t`), `turn_end` (`frame_t`), `ready` and `stats` (timing,
backlog, lag and cost breakdown). The full list is in
[PROTOCOL.md §6](PROTOCOL.md#6-keys-added-by---debug-fields). It also runs `validate()` on every outgoing message.
The debug stat `diar_lag_ms_mean_measured` is computed from sample counts and always reads the structural mean at 1x
([INTEGRATION.md §5 D8](../research/INTEGRATION.md#5-defects-found-by-the-verifier-and-their-status)).

The server prints one log line per connection event to stdout: config applied, warnings about ignored config fields,
enrollment armed, disconnects, and the final `stats`.

## 11. Presets

**The default preset is single-model mode** (`audioforge-serve` with no mode flag, [§13](#13-single-model-mode---mode-single)).
The presets below are **room mode** (the two-model stack). Each one passes `--diarizer nemotron3`, which selects
room mode by itself; `--mode room` makes it explicit.

Each preset states what it was measured as. "Pipecat / LiveKit" values are the two frameworks from
[E2E_FINAL](../research/E2E_FINAL.md#4-results-by-clip-set) (live, 1x, CPU, 2 threads per model process). The paths
assume `audioforge-download --diarizer nemotron3` (plus the optional models named) has been run. With a source
checkout, replace `audioforge-serve` with
`PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_served.afm` and give `--diar` as a `runs/` path.

### Two-party voice agent

One user talking to the agent; the application knows when its TTS stops.

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --enroll after_agent_arm
# client: {"type":"config","turn_policy":"timeout","timeout_ms":1000}; send {"type":"agent_end"} when TTS playback ends
# optional transcript pass (AMI WER 9.7 % vs 24.4 % streaming, see §8): add --final-asr tdt_v3
```

Why: this is E2E system **CN**, the configuration E2E_FINAL recommends for this case. On the user channel it
measured median dead air 1285 / 1256 ms and 20 % / 18 % of ends missed within 3 s (TurnBench dev, 16 clips, 56 ends),
and 1271 / 1250 ms and 6 % / 6 % (otoSpeech dev, 16 clips, 53 ends), with 1.32-1.34 cut-ins per minute. Server RTF
0.635 / 0.641 at 1485 / 1482 MB peak RSS
([E2E_FINAL §1](../research/E2E_FINAL.md#1-summary), [§4](../research/E2E_FINAL.md#4-results-by-clip-set),
[§7](../research/E2E_FINAL.md#7-rtf-by-component)). If not interrupting the user matters more than responsiveness,
use `"turn_policy": "hybrid_dyn"` (system DN): about +0.9 s of dead air, with cut-ins down 30-55 %. Feed the user's
own channel rather than a mix when the transport has it: every system did better there. The `agent_end` in E2E was a
label-derived stand-in for the TTS-end event.

### Multi-party room

Several people in the room; the agent is addressed by one of them.

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --enroll after_agent_arm
# client: {"type":"config","turn_policy":"hybrid_dyn"} (first message, before audio); send agent_end at TTS end
```

Why: this is E2E system **DN**. On the 5 AMI meeting windows it had 0.00 cut-ins per minute in both frameworks, at
median dead air 2322 / 2340 ms. The timeout variants cut in 1.36-3.39 times per minute there, because a 1 s pause
in a meeting is usually a hesitation ([E2E_FINAL §1](../research/E2E_FINAL.md#1-summary),
[§4](../research/E2E_FINAL.md#4-results-by-clip-set)). The rule itself (`hybrid_dyn`) is the best all-ends turn rule
on held-out ICSI meetings: 79.5 % miss, floor-open 50.5 %, at 1.8 % FC, n = 1312
([BASELINES](../research/BASELINES.md#dynamic-timeout-icsi-held-out-confirmation)). Caveat: n = 5 AMI windows with
one scored end each. Every AMI CI except the cut-in deltas includes 0.

### Lowest latency

```bash
audioforge-serve --diarizer nemotron3 --diar-pool max --diar-left 1 --threads 4
# client: {"type":"config","turn_policy":"timeout","timeout_ms":1000}
# do not add --final-asr / --asr-lookahead (their finals arrive after the streaming final)
```

Why: the 1000 ms `timeout` on the 0.32 s diarizer setting (the default `--diar-config`) is the lowest-dead-air live
configuration measured. Nemotron-3 lowered dead air by another 8-135 ms on the two-party sets and cut server RTF from
about 0.80 to about 0.64 ([E2E_FINAL §8](../research/E2E_FINAL.md#8-verdicts)). The 0.32 s setting saved 0.2-0.7 s of
dead air per window against 1.04 s
([INTEGRATION §4](../research/INTEGRATION.md#default-preset-low_latency_032-032-s-pipecat-and-livekit-dead-air-re-run-2026-09-26)).
The structural floor is about 1.2-1.4 s after a clean end (1.0 s + 0.08-0.24 s + compute;
[INTEGRATION §1](../research/INTEGRATION.md#1-executive-summary)). `--threads 4` lowered RTF from 0.433-0.520 to
0.341-0.422 in the server-alone test ([§3](#3-models-threads-and-speed)); it was not part of the E2E runs. A
`timeout_ms` below 1000 lowers the floor by the same amount, but it has not been measured, and the 1000 ms timeout
already cuts in on filled pauses. `--enroll after_agent_arm` is free and can be added.

### Most accurate

```bash
audioforge-serve --diar-config low_latency --enroll after_agent --threads 4 \
    --final-asr tdt_v3 --asr-lookahead 13
# client: {"type":"config","turn_policy":"hybrid_dyn"}; send agent_end at TTS end
```

Why, component by component:

- **Turn rule:** `hybrid_dyn` has the lowest miss rate on all ends of any frozen rule on held-out ICSI (79.5 %,
  floor-open 50.5 %, FC 1.8 %; [BASELINES](../research/BASELINES.md#dynamic-timeout-icsi-held-out-confirmation)) and
  0 cut-ins on 5 AMI windows ([INTEGRATION §8](../research/INTEGRATION.md#8-shipped-rules-2026-09-26)).
- **Enrollment:** `after_agent` (TitaNet following) has the lowest 6 s miss of the deployable bindings (hybrid
  51.6 % vs 61.9 % for causal_dominant). It is **worse at the 2 s horizon** (+8.5 points), i.e. it catches ends
  later ([EOT_BENCH_V2 §9](../research/EOT_BENCH_V2.md#9-who-is-the-primary-speaker-titanet-l-following-an-agent-end-identity-rule-and-enrollment-length-2026-09-26)).
  It was at the edge of keeping up on 2 threads (RTF 0.87-1.16), hence `--threads 4`; the 4-thread cost with
  enrollment was not measured. Use `after_agent_arm` if 2 s responsiveness matters.
- **Diarizer:** Sortformer v2 at `low_latency` (1.04 s) is the setting all EOT_BENCH_V2 §9 enrollment numbers were
  measured on, and it has 2 DER points less than the 0.32 s setting (26.3 vs 28.3). Miss rates were within CIs at both
  settings ([EOT_BENCH_V2 §7](../research/EOT_BENCH_V2.md#7-hybrid-head-or-timeout-under-leak-free-scoring-and-the-032-s-diarizer-setting-2026-09-26)).
  It adds about 0.6 s of column lag. Nemotron-3 with max pooling had a lower streaming DER at the same 1.04 s buffer
  (0.241 vs 0.252, [SORTFORMER_IMPORT](../research/SORTFORMER_IMPORT.md#comparison-ami-dev-headset-mix-word-level-labels-80-ms-frames-threshold-05-no-collar)),
  but it has not been measured with enrollment or inside the server at that setting.
- **Transcript:** `--final-asr tdt_v3` and `--asr-lookahead 13` exist for transcript accuracy, but no committed
  document measures them yet ([§8](#8-final-asr-and-dual-lookahead)). This whole preset has not been run end to end.

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

The default mode since 2026-09-29, for a voice agent that talks to **one known user**. Everything runs from the one
115M checkpoint, with no NVIDIA diarizer and no second ASR model. The earlier two-model default is `--mode room`
([§1](#1-launching)).

```bash
audioforge-serve                                # = --mode single (or mode: single in the --config file)
# client: {"type":"config","turn_policy":"hybrid_dyn"}
#         then {"type":"enroll","embedding":[192 floats]} (a stored print), or {"type":"agent_end"} at each TTS end
```

What the preset sets (flags you pass yourself win):

| option | value | why |
|---|---|---|
| `--turn-input` | `tsvad` | the turn head reads the TS-VAD track of the user's voice print ([§5](#5-turn-head-input---turn-input)) |
| `--diar-off` | on, and no `--diar` | the track [P(user), P(other), 0, 0] is `frame.speakers`; the primary is always column 0; no diarizer is loaded |
| `--lid` | `head` | language ID from the distilled head on the same encoder pass ([§9](#9-spoken-language-id---lid)) |
| `--enroll` | `after_agent_arm` | without a stored print, the print is the first `--tsvad-print-s` (5) s of speech after `agent_end` |
| `--dyn-wait-ms` | `2000,960` | `hybrid_dyn` waits 2.0 s of silence at head p = 0 and 0.96 s at p = 1 (research/SINGLE_MODEL.md A1) |
| `--tsvad` | `tsvad_spk.pt` | from the models directory, else `runs/` |
| `--silero` | when present | the silence arm of `hybrid_dyn` (Silero VAD v5, 2.3 MB ONNX) |

Refused with a one-line error, because each would load a second model: `--diar`, `--diarizer`, `--final-asr`,
`--lid ambernet`, `--diar-embed titanet`. An `enroll` message carrying an `embedding` sets the print under every
`--enroll` mode whenever the TS-VAD path is on. The same preset is available in-process as `audioforge.load(mode="single")`
(then `session.enroll(embedding)` / `session.agent_end()`) and as `audioforge-bench --mode single`.

The voice print is 192 numbers from the served speaker head: `audioforge.tsvad_stream.voiceprint(model, audio)` on the
user's clean speech. What was measured about its length (research/SINGLE_MODEL.md A2, eot-bench v2, misses at 6 s,
≤ 5 % false cut-offs):
- A stored 5 s print gives 34.2 % misses on AMI and 18.7 % on ICSI. A 10 s print gives 27.8 % on AMI and 19.6 % on
  ICSI.
- A 3 s print loses 13 points on AMI and 1.2 on ICSI. A 1.5 s print loses 20-22.
- A print taken live after `agent_end` loses 16-41 points in meetings. Store the user's print (≥ 5 s) when you can.

Measured against room mode, the previous product default (the full table with sources is in research/SINGLE_MODEL.md):
- **Turn ends in meetings, stored print** (offline, ≤ 5 % false cut-offs, 6 s): 34.2 % missed on AMI vs 61.9-74.8 %,
  and 18.7 % on ICSI vs 68.5-85.3 %. This is the best turn detector measured on either corpus.
- **Live through Pipecat** (37 clips, 69 sessions, the preset's exact flags, `hybrid_dyn`, stored 5 s print):
  - 37.7 % of user turns not answered within 3 s, against 34.1 % for room mode (+3.6, CI +0.0 to +7.6).
  - 0.67 cut-ins per session against 1.07 (−38 %, CI excludes 0).
  - Median dead air 1388 ms against 1292 ms.
  - On the user channel of two-party calls the gap is +4.6 points (CI excludes 0).
  - The earlier rule (`hybrid_dyn` at the served wait) missed 45.3 % (research/SINGLE_MODEL.md A1).
- **Cost:** server RTF 0.33 (max 0.34) against 0.635, and peak RSS 1156 MB against 1485 MB. There is no diarizer
  pass and no diarizer in memory.
- **Transcript:** the streaming one (AMI-200 24.4 %, LibriSpeech-200 2.29 %). The default can add Parakeet-TDT v3
  per turn (9.7 % on AMI) as a second model.
- **LID:** 91.0 % at 2 s and 97.8 % on full utterances on FLEURS-17, against AmberNet's 95.1 % / 99.5 %.

**The voice sample is a requirement, not an option.** Store at least 5 s of the user's clean speech (10 s for
meetings) as a print and send it at every connect. Two ways to make one:
- `audioforge.voiceprint(audio)` in Python;
- a server started with `--enroll explicit`: `{"type": "enroll"}` without an embedding takes the next 5 s of speech,
  and the `voiceprint` message returns the 192 numbers to store.

Live grabs after `agent_end` are the fallback only (+16 to +55 points of misses in meetings, A2). `final.speaker` is
0 for the user and 1 for someone else, by the TS-VAD column that dominates the turn.

**Not for:**
- labelling several people in a room (single mode knows "the user" and "someone else" only; use `--mode room
  --diar-labels registry`);
- a meeting-grade final transcript (use `--mode room --final-asr tdt_v3`).

The two head files (`tsvad_spk.pt`, `lid_distill.pt`) are fetched by `audioforge-download`, which shows their
sha256-pinned sources. In a checkout they come from `assets/`.
