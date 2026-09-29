# audioforge

A streaming speech front end for voice agents that talk to **one known user**. A single frozen 115M NVIDIA
cache-aware FastConformer encoder, with small heads of ours, does all of it every 80 ms over one WebSocket:
- live transcripts and voice activity;
- target diarization: is the user talking, or someone else? It follows the user's stored voice print;
- when the user's turn is over;
- the spoken language.

No second model is loaded. It is plain PyTorch, no NeMo, and two CPU threads run a session at RTF 0.33. Adapters for
Pipecat and LiveKit are included. **Room mode** (`--mode room`) keeps the earlier two-model stack: NVIDIA's
Nemotron-3-Diarization labels everyone in a room, and an optional Parakeet-TDT v3 pass rewrites each finished turn.

Measured against the dedicated models and the default local stacks of Pipecat and LiveKit
([scorecard](#results), [`research/SINGLE_MODEL.md`](research/SINGLE_MODEL.md)):
- **Meetings, with the user's stored voice print:** the fewest missed turn ends of every turn detector we tried (AMI
  34 % vs 62-75 %, ICSI 19 % vs 68-85 %).
- **Live through Pipecat:** 38 % fewer cut-ins than the room stack, at half its CPU. It answers 3.6 points fewer
  turns within 3 s (37.7 vs 34.1 % missed). Pipecat's own default misses 58.7 %.
- **Worse than the dedicated models** at speaker verification, general diarization (room mode ships NVIDIA's) and
  language ID (3.9 points behind AmberNet at 2 s). Streaming meeting ASR is also worse; room mode's per-turn second
  pass fixes that (9.7 % vs 24.4 % WER).

The whole study is in [`research/FINAL_REPORT.md`](research/FINAL_REPORT.md).

## What you get

Per connection (or per `audioforge.Session` in Python), from int16 / float32 PCM at any sample rate:

| message | when | content |
|---|---|---|
| `frame` | every 80 ms | VAD probability, end-of-turn probability, speaker columns: [P(user), P(someone else), 0, 0] (room mode: 4-8 diarizer columns), the primary speaker |
| `partial` | when the text changes | the streaming transcript of the current turn |
| `turn_end` | the user's turn is over | from the policy you choose: `hybrid_dyn` (recommended), `timeout`, `head`, `timeout_any`, ... |
| `final` | one per turn | the turn's text and speaker (0 = the user, 1 = someone else; room mode: diarizer column or voice id) |
| `voiceprint` | when a print is set | the user's voice print (192 numbers) the session now follows; store it for the next session |
| `language` | when confident | the spoken language (17 languages) |
| `stats` | at the end | real-time factor, latency percentiles, memory |

The wire protocol is in [`docs/PROTOCOL.md`](docs/PROTOCOL.md). Every option is in
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## The user's voice sample

Single-model mode follows the user by their voice print: 192 numbers from the served model's speaker head. **Store
one per user** and send it right after the session config:

```json
{"type": "enroll", "embedding": [0.0213, -0.0871, "... 192 numbers"]}
```

- **Take it from at least 5 s of the user's clean speech, alone.** Use 10 s for meetings: it cut missed turn ends from
  34 % to 28 % on AMI. 3 s loses 13 points on AMI; 1.5 s loses 20-22.
- **How to make one:** `audioforge.voiceprint(audio)` in Python, or an onboarding prompt on a server started with
  `--enroll explicit`. There, `{"type": "enroll"}` without an embedding takes the next 5 s of speech, and the server
  returns the print in a `voiceprint` message for you to store (`examples/quickstart_client.py --save-voiceprint`).
- **Live grabs are not enough.** Without a stored print the server takes the first 5 s of speech after each
  `agent_end`. In meetings that print often catches someone else, and misses rose by 16-55 points
  ([`research/SINGLE_MODEL.md`](research/SINGLE_MODEL.md) A2).

## Quickstart

Python 3.10+. A CPU is enough. The model weights are NVIDIA's and Silero's and keep their own licences:
`audioforge-download` shows each licence and asks before fetching ([`NOTICE`](NOTICE),
[`docs/MODELS.md`](docs/MODELS.md)).

```bash
git clone https://github.com/maxmelichov/audioforge && cd audioforge
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[serve]"            # CPU-only torch on Linux: add --extra-index-url https://download.pytorch.org/whl/cpu
audioforge-download                  # the 115M model + the TS-VAD head + Silero, sha256-checked, into ./models
audioforge-serve                     # single-model mode, ws://127.0.0.1:8765, 2 threads
```

In a second terminal, stream the bundled two-party call like a microphone (20 ms blocks at 1x). The clip is 16 s of
a real conversation from otoSpeech (CC BY 4.0): the user's channel and the other party's channel mixed to mono. The
client first sends the user's stored voice print (`examples/audio/two_party_call_16s.voiceprint.json`, made from
10 s of the user elsewhere in the same call):

```bash
python examples/quickstart_client.py          # or: python examples/quickstart_client.py my.wav --voiceprint me.json
```

Output on an Apple M5 laptop (CPU, 2 threads, 2026-09-29, the default server on the shipped `stage1_served_v2.afm`
built by `audioforge-download`; intermediate partials omitted):

```
{'type': 'ready', 'model': 'stage1_served_v2', 'chunk_ms': 160, 'frame_ms': 80, 'diar_config': 'off', 'column_lag_ms': 40.0, 'enroll': 'after_agent_arm', 'enrolled': False, 'primary_column': None}
  0.00s  voiceprint source=explicit seconds=0.0
  5.12s  partial   so
{'type': 'language', 't': 6.96, 'language': 'en', 'confidence': 0.6991}
  8.16s  partial   so um what value guides your life
 11.36s  partial   so um what value guides your life what values do you live by
 12.17s  turn_end  policy=hybrid_dyn silence_ms=1360
 12.17s  final     speaker=0 'so um what value guides your life what values do you live by'
 13.92s  partial   value
 14.57s  turn_end  policy=hybrid_dyn silence_ms=0
 14.57s  final     speaker=1 'value'
 15.20s  partial   guys my life
 16.00s  final     speaker=1 'guys my life'
stats {'rtf': 0.3357, 'chunk_ms_p50': 53.69, 'chunk_ms_p95': 57.46, 'first_partial_ms': 46.0, 'peak_rss_mb': 1139.8, 'enroll': 'after_agent_arm', 'enrolled': True, 'primary_column': None, 'lang': 'en'}
```

- **The user's question** (speech 3.9-10.8 s by the clip's labels) ends in a `turn_end` 1.4 s after its last word.
  Its final is `speaker=0`, the user.
- **The other party's answer** (12.9-15.0 s) comes out as `speaker=1`: someone else. The server also ended that
  turn at 14.57 s, on the pause after their first word ("value" for "values"). The segmentation is
  speaker-independent, and in a deployment the other party is usually your agent's own TTS.
- **The last final** is flushed by `end`.

otoSpeech has no transcripts; the reference in `examples/audio/two_party_call_16s.json` holds the labelled speech
intervals.

The target-speaker head was trained on conversational speech (AMI and ICSI meetings), and this clip is in that
domain. On clean read speech of two different speakers, such as the LibriSpeech clip kept for room mode, it can take
the other speaker for the user. Training on read-speech mixtures is planned for the GPU run in
`scripts/research/single_model_distill/`.

For room mode: `audioforge-download --diarizer nemotron3`, then `audioforge-serve --mode room` and
`python examples/quickstart_client.py examples/audio/two_speakers_10s.wav --voiceprint ''`. The room-mode clip is
two LibriSpeech speakers, with no print needed.

Speed without a socket: `audioforge-bench [my.wav]` runs the same engine in-process and prints RTF, per-block
latency and the events.

## Python API

The same engine in-process ([`examples/python_api.py`](examples/python_api.py)):

```python
import json

import soundfile as sf

import audioforge

fe = audioforge.load()                                   # single-model mode: the models audioforge-download installed
user = json.load(open("examples/audio/two_party_call_16s.voiceprint.json"))  # or fe.voiceprint(>= 5 s of clean speech)
pcm, sr = sf.read("examples/audio/two_party_call_16s.wav", dtype="int16")
s = fe.session(turn_policy="hybrid_dyn", sample_rate=sr)
s.enroll(user)
blocks = [pcm[i:i + sr // 50] for i in range(0, len(pcm), sr // 50)]   # 20 ms, like a microphone
for events in [s.feed(b) for b in blocks] + [s.end()]:
    for ev in events:
        if ev["type"] in ("turn_end", "final"):
            print(f'{ev["t"]:5.2f}s {ev["type"]:8s} speaker={ev.get("speaker")} {ev.get("text", ev.get("policy"))}')
        elif ev["type"] == "stats":
            print(f'RTF {ev["rtf"]}, peak RSS {ev["peak_rss_mb"]} MB')
```

```
12.17s turn_end speaker=None hybrid_dyn
12.17s final    speaker=0 so um what value guides your life bled what values do you live by
14.57s turn_end speaker=None hybrid_dyn
14.57s final    speaker=1 value
16.00s final    speaker=1 guys my life
RTF 0.1955, peak RSS 1128.7 MB
```

(The engine also prints one `[serve] loaded ...` log line first, under the `audioforge` logger.) `s.feed(pcm)`
takes float or int16 arrays or int16 bytes and returns the protocol's messages as dicts; `frame` messages are
included with `fe.session(frames=True)`. `s.agent_end()` tells the session your agent stopped talking (for
`enroll="after_agent_arm"`; single mode uses it when no print was stored). `audioforge.load(mode="room",
diarizer="nemotron3")` loads room mode, and any server option works as a keyword (`final_asr="tdt_v3"`,
`diar_labels="registry"`, ...).

## Pipecat and LiveKit

Start `audioforge-serve`, then use it as the VAD, STT and turn detector of your agent.

**Pipecat** (`pip install -e ".[pipecat]"`; full demo: [`examples/pipecat_local_demo.py`](examples/pipecat_local_demo.py)):

```python
from audioforge.integrations.pipecat import (AudioforgeHub, AudioforgeSTTService, AudioforgeTurnAnalyzer,
                                             AudioforgeVADAnalyzer)

hub = AudioforgeHub()
stt = AudioforgeSTTService(url="ws://127.0.0.1:8765", hub=hub, turn_policy="hybrid_dyn")
await stt.enroll(embedding=user_voiceprint)            # the user's stored print (192 numbers), sent at connect
user = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(
    vad_analyzer=AudioforgeVADAnalyzer(hub),
    user_turn_strategies=UserTurnStrategies(
        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=AudioforgeTurnAnalyzer(hub, policy="hybrid_dyn"))])))
pipeline = Pipeline([transport.input(), stt, user, llm, tts, transport.output()])
```

**LiveKit Agents** (`pip install -e ".[livekit]"`; worker: [`examples/livekit_agent_worker.py`](examples/livekit_agent_worker.py)):

```python
from audioforge.integrations.livekit import AudioforgeFrontend

fe = AudioforgeFrontend("ws://127.0.0.1:8765")        # one server session per agent session
fe.enroll(embedding=user_voiceprint)                   # the user's stored print, sent right after the config
session = AgentSession(
    stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
    turn_handling={"turn_detection": "stt",            # the server's turn_end commits the user turn
                   "endpointing": {"min_delay": 0.0},  # the server already waited timeout_ms
                   "interruption": {"mode": "vad"}})
fe.attach(session)                                     # sends agent_end when the agent stops speaking
```

Any other language can speak the protocol directly; [`packages/audioforge-client`](packages/audioforge-client) is a
torch-free Python client.

## Choosing a configuration

| you have | run | send in the session config |
|---|---|---|
| **one known user** (default) | `audioforge-serve` | `turn_policy: "hybrid_dyn"`; `{"type": "enroll", "embedding": [...]}` with the user's stored print; `agent_end` when your TTS stops |
| a room with several people to label | `audioforge-serve --mode room --diar-labels registry` | `turn_policy: "timeout_any"` or `"hybrid_dyn"` |
| a meeting-grade transcript | `audioforge-download --diarizer nemotron3 --with tdt_v3`, then `audioforge-serve --mode room --final-asr tdt_v3` | |

`audioforge-serve --help` lists the everyday flags, `--help-advanced` all of them. Any flag can also come from a
YAML / JSON file (`--config serve.yaml`). `AUDIOFORGE_HOME` sets where models go and `AUDIOFORGE_DATA` sets the
data root ([`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) §11 has the measured presets, §12 the config file, §13
single-model mode).

## Single-model mode and room mode

**Single-model mode** is the default (`audioforge-serve`, config key `mode: single`). Everything runs from the one
115M checkpoint: no NVIDIA diarizer and no second ASR model is loaded.
- The user's voice print conditions a target-speaker head on block 4 of the same encoder.
- The VAD head reads block 4 too (research/VAD_SINGLE.md).
- That head's track [P(user), P(someone else)] is what `frame.speakers` carries and what the turn head reads.
- Language ID comes from a distilled head on the same encoder pass.

The preset is `--turn-input tsvad --diar-off --lid head --enroll after_agent_arm --dyn-wait-ms 2000,960`. Options
that would load a second model are refused.

The measured table is in [`research/SINGLE_MODEL.md`](research/SINGLE_MODEL.md). In short, against room mode:
- **In meetings, with a stored print,** it misses fewer turn ends than any configuration we measured (offline, at
  ≤ 5 % false cut-offs): AMI 34 % vs 62-75 %, ICSI 19 % vs 68-85 %.
- **Live through Pipecat** (37 clips, 69 sessions): 37.7 % of user turns not answered within 3 s vs 34.1 % for room
  mode (CI +0.0 to +7.6 points), 38 % fewer cut-ins (0.67 vs 1.07 per session), and 96 ms more dead air.
- **Cost:** half the server CPU (RTF 0.33 vs 0.64), and 1.16 GB peak RSS vs 1.49 GB.
- **Voice print:** it needs a stored print ([above](#the-users-voice-sample)).

What it cannot do:
- label other people in a room (it only knows "the user" and "someone else");
- give a meeting-grade final transcript (streaming AMI WER 24.4 % vs 9.7 % with room mode's TDT v3 pass).

**Room mode** is `audioforge-serve --mode room`: general diarization with NVIDIA Nemotron-3-Diarization (or
`--diarizer sortformer`) next to the 115M model. Use it for those two jobs, with `--diar-labels registry` for stable
per-person ids and `--final-asr tdt_v3` for the transcript. Without `--mode`, passing `--diarizer`, `--diar` or
`--final-asr` selects room mode.

## Results

Final scorecard, measured by this repository's code on the same data as each comparison system (2026-09-28). "Ours"
is the served model at 160 ms chunks on a 2-thread laptop CPU. FC = false cutoff (ending the turn while the user is
mid-turn); "open" = the floor-open ends an agent should answer. All 26 rows, with CIs and caveats, are in the
[final report's scorecard](research/FINAL_REPORT.md#scorecard); each number traces to the JSON named.

| task (data) | ours | strongest system measured | verdict | source |
|---|---|---|---|---|
| ASR, LibriSpeech test-clean (200 utt.) | 2.29 % WER streaming | Whisper small 2.42 %; Whisper large-v3-turbo 1.47 % | **equal** to Whisper small at 12x less CPU | `runs/final_asr.json` |
| ASR, AMI meetings (200 segments) | 24.4 % streaming; **9.7 %** with TDT v3 per turn | parakeet-ctc 19.3 %; Whisper turbo 19.6 % | streaming **worse**; with the per-turn pass **better** | `runs/final_asr.json`, `runs/hybrid_asr.json` |
| VAD, AMI (64 x 20 s) | F1 0.949 | MarbleNet v2 0.937; Silero v5 0.915 | **better** than every streaming VAD | `runs/baselines_sd.json` |
| end of turn, AMI all ends (n = 974, <= 5 % FC, 6 s) | 61.9 % missed (hybrid) | Silero timeout 72.7 %; Parakeet-Realtime-EOU 70.1 %; smart-turn + Silero 72.6 % | **better** than every dedicated detector | `runs/baselines_turn.json` |
| end of turn, AMI floor-open ends (236) | 31.5 % (head OR dyn) | Silero timeout 26.7 % | **equal / worse**: a silence timeout is as good here | `runs/baselines_turn.json` |
| end of turn, held-out ICSI (n = 1312) | 79.5 % all / 50.5 % open | Silero timeout 87.4 % / 45.5 % | **better** on all ends, **equal** on open | `runs/baselines_turn_icsi*.json` |
| end of turn, TurnBench two-party calls | recall 0.862 at FP 0.092, P50 1137 ms | VAP 0.841 / 0.045 / 463 ms | **worse** than VAP on latency and FP | `runs/turnbench_dev.json` |
| product vs Pipecat default (37 clips) | 1.1-1.4 cut-ins / min | 2.1-7.5 cut-ins / min | **better**: far fewer cut-ins and misses, similar dead air | `runs/e2e_final.json` |
| product vs LiveKit default | dead air 1.31-1.66 s | 1.31-2.66 s | **equal** dead air, fewer misses | `runs/e2e_final.json` |
| speaker verification, AMI (n = 200) | 19.8 % EER | TitaNet-L 12.0 % | **worse** | `runs/spk_head.json` |
| diarization, AMI (64 windows) | 0.394 DER (our head) | Sortformer v2 0.201; Nemotron-3 0.232 | **worse**: the product ships NVIDIA's diarizer | `runs/baselines_sd.json` |
| language ID, FLEURS 17 languages (2 s / full); accented English (EdAcc) | 91.2 % / 97.7 %; 65.8 % (distilled head, `--lid head`, RTF 0.0013) | AmberNet 95.1 % / 99.5 %; 67.7 % | **worse** by 3.9 / 1.8 points, **equal** on accented English; off by default, `--lid ambernet` wires AmberNet in | `runs/lid.json` |
| compute, live, per audio second | server RTF 0.64 (Nemotron-3), 1.5 GB | Pipecat default 0.43 CPU s; LiveKit 0.28 | **worse**: 1.7-3.6x the CPU; keeps up at 1x | `runs/e2e_final.json` |
| end of turn with the user's voice print, AMI / ICSI | 39.3 % / 20.4 % missed | shipped hybrid 61.9 % / 68.5 % | **better** in rooms; the default single-model mode: live within +3.6 points of room mode with 38 % fewer cut-ins | `runs/improve_115m.json`, `runs/single_model.json` |

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the model and the server on one page |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | every flag, the config file, what each option costs and what was measured (flag table generated from the code) |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | the WebSocket protocol, message by message |
| [`docs/MODELS.md`](docs/MODELS.md) | the checkpoints `audioforge-download` fetches, pinned revisions, licences |
| [`docs/SERVER_INTERNALS.md`](docs/SERVER_INTERNALS.md) | design notes: frame clock, diarizer lag, policies, enrollment, final ASR |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | tests, lint, the machine-safety gate for heavy jobs, where research code lives |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed |
| [`demo/README.md`](demo/README.md) | the showcase video, built from real served-model events |

## Repository layout

```
audioforge/            the package: pip install -e ".[serve]"
  api.py               audioforge.load() / Session.feed(): the in-process API
  serve.py, server/    the WebSocket server (engine, sessions, connection loop; protocol, policies, flags)
  launch.py, hub.py    audioforge-serve and audioforge-download (pinned revisions, sha256, licences)
  bench.py             audioforge-bench
  integrations/        Pipecat and LiveKit adapters
  model.py, modules/   SpeechModel: one FastConformer encoder, N heads, cache-aware streaming
  heads/, losses/      ASR (CTC / RNNT / TDT / AED), VAD, turn, speaker, Sortformer heads; exact losses
  nemo_import.py       NVIDIA .nemo checkpoints -> audioforge, without NeMo
  streaming_diar.py    NVIDIA Streaming Sortformer v2 / Nemotron-3-Diarization, streaming
  train.py, cli.py     recipes -> trainer -> .afm; the `audioforge` training / catalog CLI
  datasets/, baselines/  AMI, ICSI, LibriSpeech, dyadic corpora; the dedicated models we compare against
assets/                the trained heads (19.6 MB); audioforge-download merges them with NVIDIA's weights
examples/              quickstart client, Python API, Pipecat / LiveKit demos, the bundled 10 s clip
packages/              audioforge-client: torch-free client for the protocol
integrations/          old import names of the adapters; turnbench_scorer/ (vendored, MIT) for the benchmarks
docs/                  user documentation (above)
scripts/               stream_client.py (latency measurement client); dev/ (gate, doc generator); research/
research/              the lab notes (start at research/README.md), training recipes in research/recipes/
runs/*.json            every measured number (checkpoints are not tracked)
tests/                 pytest (fast CPU subset in tests/fast_ci.txt, run by CI)
```

## The research

This repository started as an analysis of all 149 NVIDIA audio models on Hugging Face and the design they share
([`research/ANALYSIS.md`](research/ANALYSIS.md)). It then rebuilt that design in plain PyTorch, from the front end and
FastConformer to swappable heads, cache-aware streaming and tokens as the interface, with recipes that reproduce
NVIDIA's models. Finally it asked whether one frozen streaming encoder can serve a voice agent's whole speech front
end. Read in this order:

1. [`research/FINAL_REPORT.md`](research/FINAL_REPORT.md) (also as [HTML](research/FINAL_REPORT.html)): the
   scorecard, per-task tables, the recommended stack (§8), negative results, limitations;
2. [`research/E2E_FINAL.md`](research/E2E_FINAL.md): the product comparison with Pipecat and LiveKit;
3. [`research/VERIFICATION_2026-09-28.md`](research/VERIFICATION_2026-09-28.md): every number in the report read
   back from its JSON;
4. [`research/README.md`](research/README.md): the index of every note, with its key number. The early synthetic
   results and the corrections log are in [`research/EARLY_RESULTS.md`](research/EARLY_RESULTS.md).

To cite this work, use [`CITATION.cff`](CITATION.cff) (GitHub shows it as "Cite this repository").

## License

The code is Apache-2.0 ([`LICENSE`](LICENSE)). The model weights are not covered by it: NVIDIA's checkpoints keep
their own licences (CC-BY-4.0, OpenMDW-1.1 or the NVIDIA Open Model License, per model), as do Silero (MIT) and the
vendored TurnBench scorer (MIT). [`NOTICE`](NOTICE) lists each one.
