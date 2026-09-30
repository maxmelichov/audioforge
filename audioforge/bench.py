"""``audioforge-bench``: run the served stack in-process on a WAV file and report speed and events (no server).

    audioforge-bench                                  # the bundled examples/audio/two_party_call_16s.wav
    audioforge-bench my.wav --threads 2 --policy hybrid_dyn --repeat 3 --json out.json

The audio is fed in 160 ms blocks as fast as the models allow (no real-time pacing), through exactly the code the
WebSocket server runs (``serve.Engine`` / ``serve.Session``). Reported: RTF (compute / audio duration), per-block
compute p50 / p95 / max in ms, peak RSS, the turn_end events and the final transcript. Model paths resolve as in
``audioforge-serve`` (``--asr`` / ``--diar`` override). For live latency over a socket use
``scripts/stream_client.py`` against a running server.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import hub
from .metrics import peak_rss_mb

__all__ = ["DEFAULT_WAV", "main", "run_once"]

DEFAULT_WAV = hub.ROOT / "examples" / "audio" / "two_party_call_16s.wav"  # a real two-party call (single mode)


def run_once(engine, audio: np.ndarray, policy: str, block_ms: int = 160) -> dict:
    from .metrics import pct
    from .serve import SR, Session, SessionConfig
    s = Session(engine, SessionConfig(turn_policy=policy))
    n = SR * block_ms // 1000
    msgs, costs = [], []
    t_all = time.perf_counter()
    for i in range(0, len(audio), n):
        t0 = time.perf_counter()
        msgs += s.process(audio[i:i + n])
        costs.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    msgs += s.finish()
    costs.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - t_all
    dur = len(audio) / SR
    finals = [m for m in msgs if m["type"] == "final" and m.get("source", "stream") == "stream"]
    return {"audio_s": round(dur, 2), "compute_s": round(wall, 3), "rtf": round(wall / dur, 3),
            "block_ms": block_ms, "block_compute_ms": {"p50": pct(costs, 50, nd=1), "p95": pct(costs, 95, nd=1),
                                                       "max": round(max(costs), 1)},
            "turn_ends": [{k: m.get(k) for k in ("t", "policy", "p", "silence_ms")}
                          for m in msgs if m["type"] == "turn_end"],
            "finals": [{"t": m["t"], "speaker": m.get("speaker"), "text": m["text"]} for m in finals],
            "transcript": " ".join(m["text"] for m in finals if m["text"]).strip()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="audioforge-bench", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("audio", nargs="?", default=str(DEFAULT_WAV), help="WAV/FLAC file (any rate; resampled to 16 kHz)")
    ap.add_argument("--asr", default=None)
    ap.add_argument("--diar", default=None)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--diarizer", choices=sorted(hub.DIARIZERS), default=None, help="room mode's diarizer (implies room)")
    ap.add_argument("--mode", choices=["single", "room"], default=None,
                    help="single (default, as audioforge-serve: one 115M model, no diarizer) | room (+ a diarizer); "
                         "--diarizer / --diar select room")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--policy", default="timeout", help="turn policy (see docs/CONFIGURATION.md)")
    ap.add_argument("--diar-config", default=None, help="low_latency_032 (default) | low_latency")
    ap.add_argument("--repeat", type=int, default=1, help="runs after the warm-up; the best RTF is reported")
    ap.add_argument("--no-fast-conv", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (default) | mps | cuda / cuda:N (as audioforge-serve --device)")
    ap.add_argument("--json", default=None, help="write the report here")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(a.threads)
    from .data import load_wav
    from .serve import DEFAULT_DIAR_CONFIG, Engine

    def need(key):
        p = hub.find_model(key, a.models_dir)
        if p is None:
            sys.exit(f"audioforge-bench: model '{key}' not found; run audioforge-download")
        return str(p)

    asr = a.asr or need("asr")
    mode = a.mode or ("room" if a.diarizer or a.diar else "single")
    if mode == "single":  # the one model: TS-VAD columns, no diarizer loaded, LID head (as audioforge-serve)
        from .launch import TSVAD_FILE, find_head
        from .server.cli import MODES
        diar, product = None, {**MODES["single"], "tsvad": str(find_head(TSVAD_FILE, a.models_dir))}
    else:
        dz = a.diarizer or ("nemotron3" if hub.find_model("nemotron3", a.models_dir)
                            or not hub.find_model("sortformer", a.models_dir) else "sortformer")
        diar = a.diar or need(dz)
        product = hub.diarizer_defaults(dz) if a.diar is None else {}  # as audioforge-serve applies them
    t0 = time.perf_counter()
    eng = Engine.load(asr, diar, a.device, threads=a.threads, fast=not a.no_fast_conv,
                      diar_config=a.diar_config or DEFAULT_DIAR_CONFIG, **product)
    load_s = time.perf_counter() - t0
    audio = load_wav(a.audio, 16000).astype(np.float32)
    eng.warmup()
    runs = [run_once(eng, audio, a.policy) for _ in range(max(1, a.repeat))]
    best = min(runs, key=lambda r: r["rtf"])
    rep = {"audio": str(a.audio), "asr": asr, "diar": diar, "device": a.device, "threads": a.threads, "policy": a.policy,
           "diar_config": eng.diar_config, "load_s": round(load_s, 1), "peak_rss_mb": peak_rss_mb(),
           "rtf_all": [r["rtf"] for r in runs], **best}
    print(f"audio {rep['audio_s']} s | load {rep['load_s']} s | RTF {rep['rtf']} (runs {rep['rtf_all']}) | "
          f"160 ms block compute p50 {best['block_compute_ms']['p50']} / p95 {best['block_compute_ms']['p95']} / "
          f"max {best['block_compute_ms']['max']} ms | peak RSS {rep['peak_rss_mb']} MB | {a.threads} threads")
    for e in best["turn_ends"]:
        print(f"  turn_end t={e['t']:.2f}s policy={e['policy']} p={e['p']} silence_ms={e['silence_ms']}")
    for f in best["finals"]:
        print(f"  final    t={f['t']:.2f}s speaker={f['speaker']} {f['text']!r}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
