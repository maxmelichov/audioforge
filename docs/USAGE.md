# Using audioforge

How to run the server, pick a model size and a turn preset, give it the user's voice, and plug it into Pipecat,
LiveKit or your own code. Every flag is in [`CONFIGURATION.md`](CONFIGURATION.md); the messages on the wire are in
[`PROTOCOL.md`](PROTOCOL.md); the numbers behind each choice are in [`RESULTS.md`](RESULTS.md).

## Install and start

```bash
pip install -e ".[serve]"            # the server; ".[pipecat]" / ".[livekit]" add the adapters
audioforge-download                  # the default 115M core and its heads into ./models (asks before each licence)
audioforge-serve                     # single mode on ws://127.0.0.1:8765
```

Then, in a second terminal, stream the bundled 16 s two-party call
([`examples/audio/two_party_call_16s.wav`](../examples/audio), otoSpeech, CC BY 4.0) with the user's stored print
(`two_party_call_16s.voiceprint.json`):

```bash
python examples/quickstart_client.py        # or: ... my.wav --voiceprint me.json
```

Output on an Apple M5 laptop (CPU, 2 threads; intermediate partials omitted):

```
  0.00s  voiceprint source=explicit seconds=0.0
  8.16s  partial   so um what value guides your life
 11.36s  partial   so um what value guides your life what values do you live by
 11.78s  turn_end  policy=vad_head silence_ms=640
 11.78s  final     speaker=0 'so um what value guides your life what values do you live by'
 15.20s  partial   value guys my life
 15.30s  turn_end  policy=vad_head silence_ms=160
 15.30s  final     speaker=1 'value guys my life'
```

The user's question is one turn despite a half-second pause after "life"; the other party's answer comes out as
`speaker=1`.

`audioforge-download --list` shows every component; `--yes` (or `AUDIOFORGE_ACCEPT_LICENSES=1`) accepts the licences
non-interactively. What each download is, its pinned source and licence: [ARCHITECTURE.md "Models and files"](ARCHITECTURE.md#models-and-files).

## The flags that matter

| flag | what it does |
|---|---|
| `--device cpu\|mps\|cuda` | where the model runs (default `cpu`; `mps` is the Apple GPU) |
| `--core 115m\|0.6b` | the model size ([below](#the-two-cores)) |
| `--turn-preset balanced\|fast\|steady\|assistant` | how eagerly a turn is ended ([below](#turn-presets)) |
| `--enroll explicit` | take the user's voice print from the next 5 s of speech on request ([below](#the-users-voice-print)) |
| `--final-chunk-ms 1120` | a second, more accurate pass writes each turn's final text ([below](#dual-rate-final-text)) |
| `--mode room` | label everyone in the room with NVIDIA's diarizer ([below](#single-mode-and-room-mode)) |
| `--lid`, `--voice-gender head` | language ID and perceived voice gender ([below](#optional-heads-language-and-voice-gender)) |
| `--host`, `--port` | listen address (default `127.0.0.1:8765`) |
| `--config serve.yaml` | the same flags from a file ([CONFIGURATION.md §12](CONFIGURATION.md#12-config-file-and-environment)) |

`audioforge-serve --help` fits one screen; `--help-advanced` lists the rest.

## Single mode and room mode

**Single mode** (`audioforge-serve`, the default) is the product: one model, no Silero, no diarizer. It follows one
known user, from their voice print, and tells each moment and each turn apart as the user (`speaker=0`) or someone
else (`speaker=1`). Turn ends come from the model's own heads (the `vad_head` rule).

**Room mode** (`audioforge-download --diarizer nemotron3`, then `audioforge-serve --mode room`) is opt-in: NVIDIA
Nemotron-3-Diarization labels everyone in the room, and `--final-asr tdt_v3` rewrites each turn with Parakeet-TDT v3
(`audioforge-download --with tdt_v3`, 2.5 GB). See [CONFIGURATION.md §7](CONFIGURATION.md#7-diarizer) and
[§8](CONFIGURATION.md#8-final-asr-and-dual-lookahead).

## The user's voice print

Single mode needs the user's voice print, taken from **at least 5 s of their clean speech, alone (10 s for
meetings)**. Store it per user and send it right after the session config:

```json
{"type": "enroll", "embedding": [0.0213, -0.0871, "... 192 numbers"]}
```

Make one with `audioforge.voiceprint(audio)`, or start the server with `--enroll explicit` and send `{"type":
"enroll"}` to take the next 5 s of speech. Without a stored print the server grabs one live after `agent_end`,
which tracks the user worse.

From the command line: `python examples/quickstart_client.py me_5s.wav --save-voiceprint me.json --enroll-live`
records a print from a clip, and `--voiceprint me.json` uses it. A voice print belongs to one core: re-enroll after
switching cores. Details: [CONFIGURATION.md §6](CONFIGURATION.md#6-primary-speaker-enrollment---enroll) and
[§13](CONFIGURATION.md#13-single-model-mode---mode-single), [PROTOCOL.md §4.5](PROTOCOL.md#45-enroll-text---enroll-explicit).

## Turn presets

`--turn-preset` (per server) or `"turn_preset"` in the session's first `config` message (per session):

| preset | for | trade-off |
|---|---|---|
| `balanced` (default) | conversations and meetings | waits through pauses; the fewest interruptions on meetings |
| `fast` | phone calls | answers about 400 ms sooner on calls, at a few more interruptions |
| `steady` | calls where a missed turn end hurts most | the earlier fast rule: the steadiest timing (best p95) and the fewest missed turn ends on calls |
| `assistant` | speech directed at an agent | right about "done" most often on assistant speech; misses many meeting turns |

The measured rows for each preset and the exact rules: [CONFIGURATION.md "Turn presets"](CONFIGURATION.md#turn-presets---turn-preset-configturn_preset)
and [§4](CONFIGURATION.md#4-turn-policies-configturn_policy).

## The two cores

The default **115M** core (NVIDIA `stt_en_fastconformer_hybrid_large_streaming_multi`, CC-BY-4.0) runs well on a
CPU. The optional **0.6B** core is NVIDIA's 618M streaming model
([`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), NVIDIA Open
Model License), with every head retrained on it (speech detector, VAD, speaker, TS-VAD, turn classifiers, LID):

```bash
audioforge-download --core 0.6b          # NVIDIA nemotron-speech-streaming-en-0.6b + its heads (2.5 GB download)
audioforge-serve --core 0.6b --device mps
```

Same server, same protocol. What you gain: transcripts at about half the error, a better voice print and tracker, a
more accurate agent preset and fewer interruptions on meetings. What it costs: 3× the CPU per chunk, a quarter of the
CPU streams, 4× the memory, and a less permissive licence. Use a GPU for it. The numbers side by side:
[RESULTS.md "The two cores side by side"](RESULTS.md#the-two-cores-side-by-side).

## Dual-rate final text

**Dual rate** (`audioforge-serve --final-chunk-ms 1120`, opt-in): the heads and turn decisions stay on the 160 ms
pass, unchanged byte for byte, and a second pass of the same encoder at NVIDIA's largest trained chunk (1.12 s) writes
the final transcript. The 160 ms text still arrives at the turn end as `final_fast` (what the Pipecat / LiveKit
adapters forward by default); the slow `final` follows 9-48 ms later on the GPU (~105 ms for the 0.6B on CPU, where it
is not recommended). Test sets: ICSI test −3.5 (115M) / −1.9 (0.6B) WER points, live calls −1.7 / −0.8, AMI test
−0.6 / −0.1 (not significant). Cost: +9-27 ms at p95 on turn-end delivery from the slow chunk's burst (+93 ms for the
0.6B on CPU), and fewer real-time streams when sessions start together (same-run comparison): 115M on the GPU 4 → 3,
0.6B on the GPU 3 → 1.

To have the adapters forward the slow text instead, pass `final_text="slow"` to `AudioforgeSTTService` (Pipecat) or
`AudioforgeFrontend` (LiveKit).

## Optional heads: language and voice gender

- **Language ID** (`--lid head`, on in single mode when the head file is present): a `language` message once the
  language is clear, and `stats.lang`. `--lid ambernet` uses NVIDIA AmberNet instead (`audioforge-download --with
  ambernet`); `--lid-langs` restricts the labels. See [CONFIGURATION.md §9](CONFIGURATION.md#9-spoken-language-id---lid).
- **Perceived voice gender** (`--voice-gender head`, off by default; `audioforge-download --with voice_gender`, or
  `voice_gender_0p6b` for the 0.6B): adds `final.voice_gender` and `stats.voice_gender`, female / male voice
  probabilities. It is a perceived vocal characteristic estimated from audio, not anyone's gender identity, it can be
  wrong for any individual, and it must not be used to make decisions about people.

## Pipecat

`pip install -e ".[pipecat]"`. One server session per pipeline, shared by the STT, the VAD and the turn analyzer
through an `AudioforgeHub`:

```python
from audioforge.integrations.pipecat import (AudioforgeHub, AudioforgeSTTService, AudioforgeTurnAnalyzer,
                                              AudioforgeVADAnalyzer)

hub = AudioforgeHub()
stt = AudioforgeSTTService(url="ws://127.0.0.1:8765", hub=hub)
user = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(
    vad_analyzer=AudioforgeVADAnalyzer(hub),
    user_turn_strategies=UserTurnStrategies(
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=AudioforgeTurnAnalyzer(hub, policy="vad_head"))])))
Pipeline([transport.input(), stt, user, llm, tts, transport.output()])
```

Early end-of-turn hints (`turn_hints=True` with `AudioforgeEagerTurnStopStrategy`), enrolment after the bot speaks
(`enroll="after_agent"`) and the other options are in the module docstring of `audioforge/integrations/pipecat.py`.

## LiveKit Agents

`pip install -e ".[livekit]"`. One `AudioforgeFrontend` gives the STT, VAD and turn detector of one server session:

```python
from audioforge.integrations.livekit import AudioforgeFrontend

fe = AudioforgeFrontend("ws://127.0.0.1:8765", turn_policy="vad_head", turn_hints=True)
session = AgentSession(stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
                       turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.0},
                                      "interruption": {"mode": "vad"},
                                      "preemptive_generation": {"enabled": True}})
```

`fe.agent_end()` (or `fe.attach(session)`) sends `agent_end` when the agent stops speaking; `fe.enroll()` for
`--enroll explicit`. A runnable worker: [`examples/livekit_agent_worker.py`](../examples/livekit_agent_worker.py).

## Python API

No server, no socket ([`examples/python_api.py`](../examples/python_api.py)):

```python
import audioforge
fe = audioforge.load()                      # device="mps" / "cuda", core="0.6b", mode="room" also work
s = fe.session(sample_rate=16000)
s.enroll(voiceprint)                        # 192 numbers, or fe.voiceprint(>= 5 s of the user's clean speech)
events = s.feed(pcm) + s.end()              # the protocol's messages as dicts
```

`feed` takes int16 or float audio of any block size and returns the events it completed; `end` flushes the last
`final` and `stats`. `s.agent_end()` marks the end of the agent's own speech. `audioforge.load` takes the server flags
as keyword arguments with underscores (`enroll="explicit"`, `final_chunk_ms=1120`, ...).

## Any other language

Open a WebSocket, send an optional `{"type": "config"}`, then 16 kHz int16 mono PCM as binary messages, then
`{"type": "end"}`. The server sends `ready`, `frame` (every 80 ms), `partial`, `turn_end`, `final`, `stats`.
Everything: [`PROTOCOL.md`](PROTOCOL.md). A torch-free Python client (only `websockets`) is in
[`packages/audioforge-client`](../packages/audioforge-client); it is not on PyPI, install it straight from GitHub:

```bash
pip install "audioforge-client @ git+https://github.com/maxmelichov/audioforge#subdirectory=packages/audioforge-client"
# or from a clone: pip install ./packages/audioforge-client
```

The reference client is [`scripts/stream_client.py`](../scripts/stream_client.py).
