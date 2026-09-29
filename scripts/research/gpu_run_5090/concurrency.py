"""Concurrent live sessions against one running audioforge-serve: N clients stream the bundled two-party call at 1x
(20 ms int16 blocks, the stored print sent at connect, turn policy hybrid_dyn), start times staggered over one
block. Per session: frame delivery latency (arrival wall time - the frame's t, its audio end time, p50 / p95 / max), the
server's stats (rtf, chunk_ms_p95) and the finals. A session "keeps up" when its p95 frame latency stays below
500 ms and does not grow over the clip.

    PYTHONPATH=. python scripts/research/gpu_run_5090/concurrency.py --url ws://127.0.0.1:8771 --n 1 2 4 8 --out conc_cpu.json

research/GPU_RUN_2026-09-29.md "Concurrent sessions" (one audioforge-serve per device, started separately).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SR, BLOCK = 16000, 320


async def one(url, pcm, emb, delay):
    from websockets.asyncio.client import connect
    await asyncio.sleep(delay)
    lat, finals, stats = [], [], None
    async with connect(url, max_size=2 ** 22) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "config", "turn_policy": "hybrid_dyn", "sample_rate": SR}))
        await ws.send(json.dumps({"type": "enroll", "embedding": emb}))
        t0 = time.perf_counter()

        async def sender():
            for i, k in enumerate(range(0, len(pcm), BLOCK)):
                await ws.send(pcm[k:k + BLOCK].tobytes())
                await asyncio.sleep(max(0.0, t0 + (i + 1) * BLOCK / SR - time.perf_counter()))
            await ws.send(json.dumps({"type": "end"}))

        task = asyncio.create_task(sender())
        async for raw in ws:
            m = json.loads(raw)
            now = time.perf_counter() - t0
            if m["type"] in ("frame", "frames"):
                for f in (m["items"] if m["type"] == "frames" else [m]):
                    lat.append((f["t"], (now - f["t"]) * 1000))  # t = end of the frame (docs/PROTOCOL.md)
            elif m["type"] == "final":
                finals.append((round(m["t"], 2), m.get("speaker"), m["text"]))
            elif m["type"] == "stats":
                stats = {k: m.get(k) for k in ("rtf", "chunk_ms_p50", "chunk_ms_p95", "peak_rss_mb")}
        await task
    ls = [v for _, v in lat]
    q = len(ls) // 4
    return {"lat_ms_p50": round(float(np.percentile(ls, 50)), 1), "lat_ms_p95": round(float(np.percentile(ls, 95)), 1),
            "lat_ms_max": round(max(ls), 1),
            "lat_growth_ms": round(float(np.mean(ls[-q:]) - np.mean(ls[:q])), 1) if q else None,
            "server": stats, "finals": finals}


async def run(url, n, pcm, emb):
    res = await asyncio.gather(*[one(url, pcm, emb, i * BLOCK / SR / max(n, 1)) for i in range(n)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    pcm, sr = sf.read(str(ROOT / "examples/audio/two_party_call_16s.wav"), dtype="int16")
    assert sr == SR
    emb = json.loads((ROOT / "examples/audio/two_party_call_16s.voiceprint.json").read_text())
    out = {}
    for n in a.n:
        res = asyncio.run(run(a.url, n, pcm, emb))
        solo = res[0]["finals"]
        agg = {"n": n, "lat_ms_p95_max": max(r["lat_ms_p95"] for r in res),
               "lat_ms_p50_median": float(np.median([r["lat_ms_p50"] for r in res])),
               "lat_growth_ms_max": max(r["lat_growth_ms"] or 0 for r in res),
               "server_rtf_max": max((r["server"] or {}).get("rtf") or 0 for r in res),
               "server_chunk_ms_p95_max": max((r["server"] or {}).get("chunk_ms_p95") or 0 for r in res),
               "same_finals_text": all([f[2] for f in r["finals"]] == [f[2] for f in solo] for r in res)}
        print(json.dumps(agg), flush=True)
        out[str(n)] = {"summary": agg, "sessions": res}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
