"""Collect the measurements of research/GPU_RUN_2026-09-29.md into runs/gpu_run_2026-09-29.json.

    PYTHONPATH=. python scripts/research/gpu_run_5090/build_results.py --raw <gpu_run dir> --live <live_table.py output>

<gpu_run dir> holds the raw outputs named in the note: audioforge-bench --json files (bench_{single,room}_{cpu,cuda}*.json),
cuda_stream_probe.py outputs (probe_*.json), concurrency.py outputs (conc_{cpu,cuda}.json), the LiveKit room caller
logs (lk_room/caller_*.jsonl) and the room-mode final-ASR log (room_final_asr.txt).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _probe(p: Path) -> dict:
    t = p.read_text()
    d = json.loads(t[t.index("{\n"): t.rindex("}") + 1])
    return {k: d[k] for k in ("dev", "rtf_all", "rtf", "block_compute_ms", "turn_ends", "finals")}


def _bench(p: Path) -> dict:
    d = json.loads(p.read_text())
    return {k: d.get(k) for k in ("device", "threads", "policy", "load_s", "peak_rss_mb", "rtf_all", "rtf", "block_compute_ms",
                                  "turn_ends", "finals")}


def _lk(p: Path) -> dict:
    fin = [json.loads(line) for line in p.read_text().splitlines() if '"lk.transcription_final": "true"' in line]
    return [{"audio_t": r["audio_t"], "from": "agent" if r["from"].startswith("agent") else r["from"], "text": r["text"]}
            for r in fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--live", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "gpu_run_2026-09-29.json"))
    a = ap.parse_args()
    raw = Path(a.raw)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "note": "research/GPU_RUN_2026-09-29.md"}
    res["probe_single_mode"] = {
        "what": "cuda_stream_probe.py: bundled 16 s two-party clip, single mode, hybrid_dyn, best of 3 after warm-up",
        "idle_gpu0_19h55": {"cpu_fastconv_2threads": _probe(raw / "probe_cpu.json"),
                            "cuda_tf32_default": _probe(raw / "probe_cuda.json")},
        "gpu1_shared_22h17": {f"cuda_tf32_off_{i}": _probe(raw / f"probe_cuda_idle_{i}.json") for i in (1, 2)}
        | {f"cuda_tf32_on_{i}": _probe(raw / f"probe_cuda_idle_tf32_{i}.json") for i in (1, 2)}
        | {f"cpu_{i}": _probe(raw / f"probe_cpu_now_{i}.json") for i in (1, 2)}}
    res["bench"] = {"what": "audioforge-bench --repeat 3 --policy hybrid_dyn --threads 2 on the bundled clip",
                    **{f"{m}_{d}": _bench(raw / f"bench_{m}_{d}_gpu1.json") for m in ("single", "room") for d in ("cpu", "cuda")},
                    "room_cpu_idle_19h58": _bench(raw / "bench_room_cpu.json"),
                    "room_cuda_idle_19h58": _bench(raw / "bench_room_cuda.json")}
    res["concurrency"] = {"what": "concurrency.py: N live clients at 1x against one audioforge-serve (single mode)",
                          **{d: {n: v["summary"] for n, v in json.loads((raw / f"conc_{d}.json").read_text()).items()}
                             for d in ("cpu", "cuda")}}
    lk = raw / "lk_room"
    res["livekit_room"] = {
        "what": "livekit-server --dev (docker, v1.13.7) + examples/livekit_agent_worker.py + livekit_publish_wav.py, "
                "bundled two-party clip, audioforge-serve single mode",
        "cpu": [_lk(lk / "caller.jsonl"), _lk(lk / "caller_cpu_2.jsonl"), _lk(lk / "caller_cpu_3.jsonl")],
        "cuda_tf32_default": [_lk(lk / "caller_cuda.jsonl"), _lk(lk / "caller_cuda_2.jsonl"), _lk(lk / "caller_cuda_3.jsonl")],
        "cuda_tf32_off": [_lk(lk / f"caller_cuda_{i}.jsonl") for i in (4, 5, 6)]}
    res["room_final_asr_log"] = (raw / "room_final_asr.txt").read_text().splitlines()
    if a.live:
        res["live_69"] = json.loads(Path(a.live).read_text())
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
