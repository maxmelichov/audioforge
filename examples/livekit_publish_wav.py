"""Caller side of a LiveKit room test: publish a WAV as a participant's microphone track at real time and print the
agent's transcriptions (the ``lk.transcription`` text streams AgentSession publishes) with audio-relative times.

    LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
    PYTHONPATH=. .venv/bin/python examples/livekit_publish_wav.py WIN.wav [--room audioforge-test] [--wait 3]

Not run in this repo's tests (needs a LiveKit server).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import api, rtc  # noqa: E402

SR = 16000


async def main(a):
    from audioforge.data import load_wav
    url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    token = (api.AccessToken(os.environ.get("LIVEKIT_API_KEY", "devkey"), os.environ.get("LIVEKIT_API_SECRET", "secret"))
             .with_identity(a.identity).with_grants(api.VideoGrants(room_join=True, room=a.room)).to_jwt())
    room = rtc.Room()
    t_audio0 = None
    log = open(a.log, "w") if a.log else None

    def rel():
        return None if t_audio0 is None else round(time.time() - t_audio0, 3)

    async def on_text(reader, identity):
        info = reader.info
        text = await reader.read_all()
        rec = {"wall": time.time(), "audio_t": rel(), "from": identity, "attrs": dict(info.attributes or {}),
               "text": text}
        print(json.dumps(rec), flush=True)
        if log:
            log.write(json.dumps(rec) + "\n")

    room.register_text_stream_handler("lk.transcription",
                                      lambda reader, identity: asyncio.ensure_future(on_text(reader, identity)))
    await room.connect(url, token)
    print(f"connected to {url} room {a.room} as {a.identity}", flush=True)
    src = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", src)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(a.wait)  # let the agent join and subscribe
    x = load_wav(a.wav, SR)
    pcm = np.concatenate([(np.clip(x, -1, 1) * 32767).astype(np.int16), np.zeros(int(a.tail * SR), np.int16)])
    n = SR // 100  # 10 ms frames; capture_frame blocks when the source queue is full -> real-time pacing
    t_audio0 = time.time()
    print(json.dumps({"audio_start_wall": t_audio0}), flush=True)
    for i in range(0, len(pcm), n):
        chunk = pcm[i:i + n]
        await src.capture_frame(rtc.AudioFrame(chunk.tobytes(), SR, 1, len(chunk)))
    await src.wait_for_playout()
    await asyncio.sleep(a.linger)
    await room.disconnect()
    if log:
        log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wav")
    ap.add_argument("--room", default="audioforge-test")
    ap.add_argument("--identity", default="caller")
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to wait for the agent before streaming")
    ap.add_argument("--tail", type=float, default=2.5, help="seconds of silence after the file")
    ap.add_argument("--linger", type=float, default=3.0)
    ap.add_argument("--log", help="JSONL of received transcriptions")
    asyncio.run(main(ap.parse_args()))
