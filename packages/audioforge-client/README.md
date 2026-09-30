# audioforge-client

WebSocket client and message framing for the **audioforge** streaming speech server
(`audioforge/serve.py` in [maxmelichov/audioforge](https://github.com/maxmelichov/audioforge)):
streaming ASR + VAD + speaker activity + end-of-turn events on one 80 ms clock.

This package contains **no model code and no torch dependency** (only `websockets`). It is the piece a
voice-agent framework plugin needs:

- `audioforge_client.protocol`: the message types (`Ready`, `FrameEvent`, `Partial`, `TurnEnd`, `Final`,
  `Enrolled`, `LanguageEvent`, `Stats`), `parse_message()` (expands `frames` batches), `config_message()`,
  the policy / enrollment constants and the two server rules a client must know: `cut_policy()` (which
  `turn_end` tag cuts the finals) and `use_final()` (which finals are the transcript on a `--final-asr` server).
- `audioforge_client.AudioforgeSession`: one asyncio session (connect, `send_audio`, `agent_end` / `enroll`,
  `end`, async-iterate the events).
- `audioforge_client.testing.FakeAudioforgeServer`: a deterministic, energy-scripted stand-in for the server
  that speaks the same protocol, for unit tests of clients and plugins.

```python
import asyncio
from audioforge_client import AudioforgeSession, Final, TurnEnd
from audioforge_client.testing import FakeAudioforgeServer, speech_pcm

async def main():
    async with FakeAudioforgeServer(timeout_ms=480) as srv:      # or a real server: "ws://127.0.0.1:8765"
        async with AudioforgeSession(srv.url, turn_policy="timeout", timeout_ms=480) as s:
            await s.send_audio(speech_pcm(1.0, 1.0))               # int16 mono PCM, any chunk size
            await s.end()                                          # flush: last final + stats, server closes
            async for ev in s:
                if isinstance(ev, (TurnEnd, Final)):
                    print(ev)

asyncio.run(main())
```

## Running the server

```bash
git clone https://github.com/maxmelichov/audioforge && cd audioforge
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy pyyaml sentencepiece websockets
# models: see research/archive/NEMO_IMPORT.md and research/archive/SORTFORMER_IMPORT.md (NVIDIA FastConformer streaming encoder,
# Streaming Sortformer v2, both CC-BY-4.0) and scripts/research/make_served_model.py for runs/stage1_served.afm
.venv/bin/python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm --port 8765
```

Options a client can ask for per session (the `config` message): `turn_policy` (`timeout` default,
`hybrid_dyn`, `hybrid_silero`, `head`, `both`, `hybrid`, `timeout_quiet`), `timeout_ms`, `eot_threshold`,
`sample_rate`. Options set on the server: `--enroll after_agent_arm` (armed by the client's `agent_end`),
`--final-asr tdt_v3` (a second, offline final per turn), `--lid ambernet` (`language` events).

## Protocol

See the docstring of `audioforge_client.protocol` and `audioforge/serve.py`. `t` is audio time in seconds
since the session start; `turn_end.t` is the *decision* time, not the last speech sample.

## License

Apache-2.0 (this client). The server and its models have their own licences (see the main repository).
