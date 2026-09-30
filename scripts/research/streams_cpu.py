"""How many real-time single-mode streams one server process holds on this CPU (2 threads) -> runs/streams_cpu.json.

K sessions of `audioforge.load()` (--mode single, the served code) are fed interleaved, 160 ms block by block, on one
thread, as the server's single executor runs them; each stream is a different 60 s AMI test Mix-Headset window. The K
streams are real time when the compute for one 160 ms block of every stream, summed (aggregate compute per 160 ms of
wall clock), stays below 160 ms at p95. Research/GPU_RUN_2026-09-29.md measured the same thing live on the 5090 box
(CUDA: 4 streams; its x86 CPU at 2 threads: 1). Through scripts/dev/gate.sh.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SR, BLOCK = 16000, 2560
WINDOWS = [("IS1009b", 600), ("ES2004b", 600), ("TS3003b", 600), ("EN2002a", 600), ("IS1009b", 1200)]


def main():
    import audioforge
    from audioforge.data import load_wav
    fe = audioforge.load(threads=2)
    clips = []
    for m, st in WINDOWS:
        x = load_wav(str(ROOT / "data" / "ami" / "audio" / f"{m}.Mix-Headset.wav"), SR).astype(np.float32)
        clips.append(x[st * SR:(st + 60) * SR])
    out = {"setup": "Apple M5 laptop CPU, 2 threads, audioforge.load() (--mode single), 160 ms blocks, 60 s AMI windows",
           "criterion": "real time if p95 of the summed per-block compute of all K streams < 160 ms", "k": {}}
    for K in (1, 2, 3, 4):
        ss = [fe.session("hybrid_dyn") for _ in range(K)]
        agg = []
        n = min(len(c) for c in clips[:K]) // BLOCK
        for i in range(n):
            t0 = time.perf_counter()
            for s, c in zip(ss, clips[:K]):
                s.feed(c[i * BLOCK:(i + 1) * BLOCK])
            agg.append((time.perf_counter() - t0) * 1000)
        for s in ss:
            s.end()
        a = np.array(agg[5:])
        out["k"][str(K)] = {"agg_block_ms_p50": round(float(np.median(a)), 1),
                            "agg_block_ms_p95": round(float(np.percentile(a, 95)), 1),
                            "per_stream_block_ms_p50": round(float(np.median(a)) / K, 1),
                            "real_time": bool(np.percentile(a, 95) < 160)}
        print(K, out["k"][str(K)], flush=True)
    out["streams_real_time"] = max([int(k) for k, v in out["k"].items() if v["real_time"]], default=0)
    (ROOT / "runs" / "streams_cpu.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
