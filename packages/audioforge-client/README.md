# audioforge-client

WebSocket client and message framing for the **audioforge** streaming speech server
(`audioforge-serve` in [maxmelichov/audioforge](https://github.com/maxmelichov/audioforge)): streaming ASR,
speech detection, end-of-turn hints and decisions and the user's voice print on one 80 ms clock.

This package contains **no model code and no torch dependency** (only `websockets`). It is the piece a
voice-agent framework plugin needs:

- `audioforge_client.protocol`: the message types (`Ready`, `FrameEvent`, `Partial`, `TurnHint`,
  `TurnHintCancel`, `TurnEnd`, `Final` (also `final_fast`), `Enrolled`, `Voiceprint`, `LanguageEvent`,
  `Stats`, `ServerError`), `parse_message()` (expands `frames` batches), `config_message()`,
  `control_message()`, the policy / preset / enrollment constants and the two server rules a client must know:
  `cut_policy()` (which `turn_end` tag cuts the finals) and `use_final()` (which finals are the transcript when
  the server sends a second pass: `--final-chunk-ms`, `--final-asr`).
- `audioforge_client.AudioforgeSession`: one asyncio session (connect, `send_audio`, `agent_end` / `enroll`
  (optionally with a stored voice print), `end`, async-iterate the events).
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
pip install -e ".[serve]"
audioforge-download               # NVIDIA 115M streaming FastConformer + the audioforge heads
audioforge-serve --port 8765      # single-model mode (the default); --core 0.6b for the larger English core
```

Options a client can ask for per session (the `config` message): `turn_policy` (omitted = the server's default,
`vad_head` in single-model mode), `turn_preset` (`balanced` default, `fast`, `steady`, `assistant`),
`timeout_ms`, `eot_threshold`, `sample_rate`. Turn hints (`turn_end_hint`) are on by default on the server;
`--final-chunk-ms 1120` adds a slower, better final per turn after the immediate `final_fast`.

## Protocol

See `docs/PROTOCOL.md` in the server repository and the docstring of `audioforge_client.protocol`. `t` is audio time in seconds
since the session start; `turn_end.t` is the *decision* time, not the last speech sample.

## License

Apache-2.0 (this client). The server and its models have their own licences (see the main repository).
