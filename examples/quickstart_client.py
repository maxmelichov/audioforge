"""Minimal audioforge client: stream a WAV to a running server in real time and print the events.

    audioforge-serve &                                   # after audioforge-download (single-model mode)
    python examples/quickstart_client.py                 # the bundled two-party call + the user's stored print
    python examples/quickstart_client.py examples/audio/two_speakers_10s.wav --voiceprint ''   # room mode's clip
    python examples/quickstart_client.py my.wav --voiceprint me.json    # your stored print (see --save-voiceprint)
    python examples/quickstart_client.py me_5s.wav --save-voiceprint me.json --enroll-live   # make one

Single-model mode (the server's default) follows one known user: send their voice print right after the config,
{"type": "enroll", "embedding": [192 numbers]}, taken from >= 5 s of their clean speech (10 s for meetings). Get one
with audioforge.voiceprint(audio) in Python, or from a server's "voiceprint" message (--save-voiceprint).

Protocol (docs/PROTOCOL.md): send an optional {"type": "config"} text message, then int16 little-endian mono PCM as
binary messages (any size; 16 kHz unless the config says otherwise), then {"type": "end"}. The server sends JSON:
ready, frame (every 80 ms; not printed here), partial, turn_end, final, stats, and closes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from websockets.exceptions import InvalidHandshake, InvalidMessage

HERE = Path(__file__).resolve().parent
SR, BLOCK_S = 16000, 0.02  # 20 ms blocks, like a microphone


def read_wav(path: str) -> np.ndarray:
    """16 kHz mono float32 (uses audioforge's loader, which resamples)."""
    from audioforge.data import load_wav
    return load_wav(path, SR).astype(np.float32)


async def main(a) -> None:
    from websockets.asyncio.client import connect
    audio = read_wav(a.audio)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    n = int(SR * BLOCK_S)
    async with connect(a.url, max_size=2 ** 22) as ws:
        print(json.loads(await ws.recv()))  # ready
        cfg = {"type": "config", "sample_rate": SR}
        if a.policy:  # else the server's default: vad_head in single mode
            cfg["turn_policy"] = a.policy
        await ws.send(json.dumps(cfg))
        if a.voiceprint and not a.enroll_live:  # the user's stored print: the server follows this voice
            await ws.send(json.dumps({"type": "enroll", "embedding": json.loads(Path(a.voiceprint).read_text())}))
        elif a.enroll_live:  # take the print from the first seconds of speech that follow (a prompted, clean sample)
            await ws.send(json.dumps({"type": "enroll"}))

        async def sender():
            t0 = time.perf_counter()
            for i, k in enumerate(range(0, len(pcm), n)):
                await ws.send(pcm[k:k + n].tobytes())
                if a.realtime:  # pace like a live microphone
                    await asyncio.sleep(max(0.0, t0 + (i + 1) * BLOCK_S - time.perf_counter()))
            await ws.send(json.dumps({"type": "end"}))

        task = asyncio.create_task(sender())
        async for raw in ws:
            m = json.loads(raw)
            typ = m["type"]
            if typ == "partial":
                print(f"{m['t']:6.2f}s  partial   {m['text']}")
            elif typ == "turn_end":
                hint = f" hinted_at={m['hinted_at']}" if m.get("hinted_at") is not None else ""
                print(f"{m['t']:6.2f}s  turn_end  policy={m['policy']} silence_ms={m.get('silence_ms')}{hint}")
            elif typ == "turn_end_hint":  # early: start preparing the reply, speak it only at the turn_end
                print(f"{m['t']:6.2f}s  hint      p={m['p']:.3f} {m['text']!r}")
            elif typ == "turn_end_hint_cancel":  # the user went on: drop what the hint started
                print(f"{m['t']:6.2f}s  hint cancelled")
            elif typ == "final":
                if m["text"]:  # the end-of-stream flush can close a turn with no words
                    print(f"{m['t']:6.2f}s  final     speaker={m.get('speaker')} {m['text']!r}")
            elif typ == "voiceprint":
                print(f"{m['t']:6.2f}s  voiceprint source={m['source']} seconds={m['seconds']}")
                if a.save_voiceprint:
                    Path(a.save_voiceprint).write_text(json.dumps(m["embedding"]))
                    print(f"         saved to {a.save_voiceprint}")
            elif typ == "stats":
                print("stats", {k: v for k, v in m.items() if k != "type"})
            elif typ != "frame":
                print(m)
        await task


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("audio", nargs="?", default=str(HERE / "audio" / "two_party_call_16s.wav"))
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--policy", default=None, help="turn_policy (default: the server's; single mode: vad_head)")
    ap.add_argument("--voiceprint", default=str(HERE / "audio" / "two_party_call_16s.voiceprint.json"),
                    help="JSON list of 192 numbers: the user's stored voice print ('' = none)")
    ap.add_argument("--enroll-live", action="store_true", help="no stored print: take it from the next speech "
                    "(needs --enroll explicit on the server; single mode arms after agent_end)")
    ap.add_argument("--save-voiceprint", default=None, help="write the print from the server's voiceprint message here")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false", help="send as fast as possible")
    a = ap.parse_args()
    try:
        asyncio.run(main(a))
    except (OSError, InvalidHandshake, InvalidMessage) as e:
        raise SystemExit(f"could not talk to an audioforge server at {a.url} ({type(e).__name__}). Is "
                         "`uv run audioforge-serve` running? If another program holds the port, start the server "
                         "with --port 8766 and pass --url ws://127.0.0.1:8766 here.") from None
