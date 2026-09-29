"""Stream one WAV into a running audioforge.serve and record every server event to JSON (for demo/render.py).

    PYTHONPATH=. .venv/bin/python demo/record_events.py clip.wav --out demo/events/clip.json \
        [--url ws://127.0.0.1:8765] [--policy hybrid_dyn] [--agent-end-s 3.12,9.7] [--ref-text "..."]

Like scripts/stream_client.py (20 ms frames paced at 1x, every message stamped with its arrival time relative to the
start of the stream) but it also accepts the hybrid_* policies, sends {"type": "agent_end"} at --agent-end-s (the
--enroll after_agent_arm trigger: the moment the agent's own TTS would have finished) and writes one JSON with the
audio path, the clip metadata and the event list. Nothing is simulated: every event comes from the server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRAME_S = 0.02


async def run(a) -> dict:
    from websockets.asyncio.client import connect
    from audioforge.data import load_wav
    x = load_wav(a.audio, 16000)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    n = int(16000 * FRAME_S)
    frames = [pcm[i: i + n].tobytes() for i in range(0, len(pcm), n)]
    events: list[dict] = []
    t_start = None
    done = asyncio.Event()

    def stamp(msg):
        rel = round((time.perf_counter() - t_start) * 1000, 1) if t_start else None
        events.append({"rel_ms": rel, "msg": msg})
        if msg.get("type") != "frame":
            print(f"+{rel or 0:8.1f}ms {msg.get('type'):8s} {json.dumps({k: v for k, v in msg.items() if k != 'type'})[:160]}",
                  flush=True)

    async with connect(a.url, max_size=2 ** 22, ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        stamp(ready)
        cfg = {"type": "config", "turn_policy": a.policy, "timeout_ms": a.timeout_ms,
               "eot_threshold": a.eot_threshold, "sample_rate": 16000}
        await ws.send(json.dumps(cfg))
        if a.enroll_embedding is not None:  # served TS-VAD path: a stored 5 s voice print (scratch/e2e_tsvad/prints.json)
            await ws.send(json.dumps({"type": "enroll", "embedding": a.enroll_embedding}))
            events.append({"rel_ms": 0.0, "msg": {"type": "client_enroll", "seconds": a.print_s}})

        async def receiver():
            try:
                async for m in ws:
                    msg = json.loads(m)
                    stamp(msg)
                    if msg.get("type") == "stats":
                        break
            finally:
                done.set()

        rtask = asyncio.create_task(receiver())
        t_start = time.perf_counter()
        agent_ends = {int(round(t / FRAME_S)): t for t in a.agent_end_s}
        for i, fr in enumerate(frames):
            due = t_start + (i + 1) * FRAME_S / a.speed
            d = due - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            await ws.send(fr)
            if i + 1 in agent_ends:
                await ws.send(json.dumps({"type": "agent_end"}))
                events.append({"rel_ms": round((time.perf_counter() - t_start) * 1000, 1),
                               "msg": {"type": "client_agent_end", "t": agent_ends[i + 1]}})
        await ws.send(json.dumps({"type": "end"}))
        await done.wait()
        await rtask
    return {"audio": str(Path(a.audio).resolve()), "duration_s": round(len(x) / 16000, 3), "policy": a.policy,
            "agent_end_s": a.agent_end_s, "ref_text": a.ref_text, "label": a.label, "speed": a.speed,
            "recorded": time.strftime("%Y-%m-%d %H:%M"), "events": events}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--policy", default="hybrid_dyn")
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--eot-threshold", type=float, default=0.98)
    ap.add_argument("--agent-end-s", default="", help="comma-separated audio times (s) at which to send agent_end")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--prints-json", default=None, help="voice-print store {clip: {'5.0': {'embedding': [...]}}}")
    ap.add_argument("--print-key", default=None, help="clip name in --prints-json")
    ap.add_argument("--print-s", default="5.0")
    a = ap.parse_args(argv)
    a.enroll_embedding = None
    if a.prints_json:
        a.enroll_embedding = json.loads(Path(a.prints_json).read_text())[a.print_key][a.print_s]["embedding"]
    a.agent_end_s = [float(t) for t in a.agent_end_s.split(",") if t.strip()]
    rec = asyncio.run(run(a))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    n = {}
    for e in rec["events"]:
        n[e["msg"]["type"]] = n.get(e["msg"]["type"], 0) + 1
    print("wrote", a.out, n)


if __name__ == "__main__":
    main()
