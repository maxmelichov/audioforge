# audioforge

A real-time speech front end for voice agents: one frozen NVIDIA streaming speech model with small heads on top
gives live words, voice activity, "is it the user?" and "is the turn over?" over one WebSocket.

![architecture](docs/images/architecture_v10.png)

## What you get

- **Live words** while the user talks, and one final transcript per turn.
- **Voice activity** every 80 ms from the model itself, with no separate VAD.
- **The user, not the room:** each moment and each turn is marked as the user or someone else, from a 5 s voice print.
- **Turn ends** that wait through mid-sentence pauses, with presets for calls, meetings and speech to an assistant.
- **Optional extras:** spoken language ID and perceived voice gender. Plain PyTorch, on CPU, Apple GPU or CUDA.

## Quickstart

Python 3.10+. The download command shows each model's licence and asks before fetching it.

```bash
git clone https://github.com/maxmelichov/audioforge && cd audioforge
pip install -e ".[serve]"
audioforge-download          # the default 115M model and its heads, checked, into ./models
audioforge-serve             # ws://127.0.0.1:8765   (add --device mps or --device cuda for a GPU)
```

In a second terminal, stream the bundled 16 s two-party call (from otoSpeech, CC BY 4.0) with the user's stored
voice print:

```bash
python examples/quickstart_client.py        # or: python examples/quickstart_client.py my.wav --voiceprint me.json
```

```
  8.16s  partial   so um what value guides your life
 11.36s  partial   so um what value guides your life what values do you live by
 11.78s  turn_end  policy=vad_head silence_ms=640
 11.78s  final     speaker=0 'so um what value guides your life what values do you live by'
 15.30s  turn_end  policy=vad_head silence_ms=160
 15.30s  final     speaker=1 'value guys my life'
```

The user's question stays one turn despite a pause after "life"; the other person's answer comes out as `speaker=1`.

**In Python, without a server** (the full script is [`examples/python_api.py`](examples/python_api.py)):

```python
import json
import soundfile as sf
import audioforge

fe = audioforge.load()                                      # device="mps" / "cuda" also work
pcm, sr = sf.read("examples/audio/two_party_call_16s.wav", dtype="int16")
s = fe.session(sample_rate=sr)
s.enroll(json.load(open("examples/audio/two_party_call_16s.voiceprint.json")))   # or fe.voiceprint(5 s of the user)
for ev in s.feed(pcm) + s.end():                            # the same messages the server sends, as dicts
    if ev["type"] == "final":
        print(ev["speaker"], ev["text"])
```

**In a voice-agent framework:**

- Pipecat: `pip install -e ".[pipecat]"`, then `AudioforgeSTTService`, `AudioforgeVADAnalyzer` and
  `AudioforgeTurnAnalyzer` from `audioforge.integrations.pipecat`.
- LiveKit Agents: `pip install -e ".[livekit]"`, then `AudioforgeFrontend` from `audioforge.integrations.livekit`
  (`fe.stt()`, `fe.vad()`, `fe.turn_detector()`).
- Anything else: the WebSocket protocol, [`docs/PROTOCOL.md`](docs/PROTOCOL.md). A torch-free Python client
  (only `websockets`) installs straight from GitHub:
  `pip install "audioforge-client @ git+https://github.com/maxmelichov/audioforge#subdirectory=packages/audioforge-client"`.

Wiring examples, the larger 0.6B model, turn presets and voice prints: [`docs/USAGE.md`](docs/USAGE.md).

## Results

Public test sets only, every other system re-run on the same audio. Ours is the larger 0.6B model on every line
except speech detection (the default 115M; the 0.6B scores 0.957).

| | audioforge | other open models |
|---|---|---|
| Word errors on meeting audio (AMI test, WER) | **7.9 %** | Parakeet-TDT v3 8.3 %, Whisper large-v3 10.7 % |
| Final text ready after the user stops (Mac GPU, median) | **49 ms** | Parakeet-TDT v3 232 ms, Whisper large-v3-turbo 282 ms |
| Turn ends on speech to an assistant: right / cuts the user off | **97.7 % / 0.9 %** | LiveKit 85.2 % / 22.3 %, Pipecat 75.9 % / 29.0 % |
| Turn ends missed on meetings (AMI test; ours knows the user's voice) | **33.5 %** | Pipecat 53.0 %, LiveKit 72.0 % |
| Speech detection on meetings (AMI test, F1) | **0.959** | MarbleNet v2 0.941, Silero v5 0.901 |
| Errors on the user's own words in a meeting (AMI test) | **47.1 %** | Nemotron-3 diarizer 64.4 %, pyannote 3.1 77.8 % |

Where it loses: Pipecat answers an assistant sooner (211 ms against 351 ms), offline Parakeet-TDT v3 makes fewer
word errors on ICSI meetings and LibriSpeech, and Whisper large-v3 identifies languages better. The charts, the
caveats and every loss: [`docs/RESULTS.md`](docs/RESULTS.md).

## Documentation

- [`docs/USAGE.md`](docs/USAGE.md): running the server, the two model sizes, voice prints, Pipecat and LiveKit.
- [`docs/TRAINING.md`](docs/TRAINING.md): training or retraining the heads.
- [`docs/RESULTS.md`](docs/RESULTS.md): the comparisons, with charts.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the model, the server, the model files and their licences.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md): the WebSocket messages.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): every server flag.

## Licence

- Code and the trained heads: Apache-2.0 ([`LICENSE`](LICENSE)).
- The NVIDIA models keep their own licences (CC-BY-4.0 for the default 115M, with attribution to NVIDIA; NVIDIA Open
  Model License for the 0.6B); `audioforge-download` fetches them from NVIDIA.
- The language-ID head was trained on NVIDIA AmberNet outputs (NGC Terms of Use); treat it as under those terms
  until its redistribution is confirmed. It is optional.
- Full list: [`NOTICE`](NOTICE) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#models-and-files).
