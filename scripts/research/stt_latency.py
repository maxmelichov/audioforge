"""STT latency of the served streaming transcript (research/METRICS.md, runs/stt_latency.json).

Definition (industry "word emission latency"): for every reference word that the final hypothesis gets right, the time
from the end of that word in the audio to the moment the word, in its final spelling, first appears in the streaming
transcript (a ``partial``, or the ``final`` if it never appeared in a partial). Lower is better.

Setup: ``audioforge.load()`` = ``--mode single`` (the default product), the exact server session code, CPU 2 threads,
fed in 160 ms blocks as a real-time client would send them. A word shown after block k is visible at
    audio clock (samples fed after block k) + the measured compute time of block k.
The stream is assumed paced in real time and never backlogged (compute per 160 ms block is well under 160 ms here,
checked and reported). Network time is not included.

Data: AMI test meetings with audio on disk (IS1009b, ES2004b, TS3003b, EN2002a; Mix-Headset), 120 s windows at
300 / 900 / 1500 s. Reference = every speaker's words in the window (AMI manual word times), sorted by start;
teachers.normalize_text on both sides; jiwer alignment; latency on the words aligned as correct ("hits").

Stages (each call < 10 min, resumable; through scripts/dev/gate.sh):
  run [--meetings ...]    stream the windows -> <scratch>/metrics/stt/<meeting>_<start>.json
  report                  pool -> runs/stt_latency.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SR, BLOCK_MS, WIN_S = 16000, 160, 120.0
STARTS = (300.0, 900.0, 1500.0)
MEETINGS = ("IS1009b", "ES2004b", "TS3003b", "EN2002a")
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/metrics/stt")
AMI = ROOT / "data" / "ami"


def norm_words(text: str) -> list[str]:
    from audioforge.teachers import normalize_text
    return normalize_text(text).split()


def ref_words(meeting: str, t0: float, t1: float) -> list[tuple[float, float, str]]:
    from audioforge.datasets.ami import meeting_words
    out = []
    for words in meeting_words(AMI / "annotations", meeting).values():
        for s, e, w in words:
            if t0 <= 0.5 * (s + e) < t1:
                for tok in norm_words(w):
                    out.append((s - t0, e - t0, tok))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def stream(fe, audio: np.ndarray) -> dict:
    """Feed 160 ms blocks; return hypothesis words with their first-visible times, block compute times."""
    s = fe.session("hybrid_dyn")
    n = SR * BLOCK_MS // 1000
    seg: list[list] = []          # current segment: [word, visible_t]
    hyp: list[tuple[str, float]] = []
    comp_ms = []

    def on_text(words, t_vis):
        for i, w in enumerate(words):
            if i >= len(seg) or seg[i][0] != w:
                del seg[i:]
                seg.append([w, t_vis])

    def handle(evs, t_vis):
        for ev in evs:
            if ev["type"] == "partial":
                on_text(norm_words(ev["text"]), t_vis)
            elif ev["type"] == "final":
                words = norm_words(ev.get("text", ""))
                on_text(words, t_vis)
                hyp.extend((w, t) for w, t in seg[:len(words)])
                seg.clear()

    fed = 0
    for i in range(0, len(audio), n):
        blk = audio[i:i + n]
        t = time.perf_counter()
        evs = s.feed(blk)
        dt = time.perf_counter() - t
        fed += len(blk)
        comp_ms.append(dt * 1000)
        handle(evs, fed / SR + dt)
    t = time.perf_counter()
    evs = s.end()
    dt = time.perf_counter() - t
    handle(evs, fed / SR + dt)
    stats = next((e for e in evs if e["type"] == "stats"), {})
    return {"hyp": hyp, "block_ms": comp_ms, "stats": {k: stats.get(k) for k in ("rtf", "chunk_ms_p50", "chunk_ms_p95")}}


def score(ref, hyp) -> dict:
    import jiwer
    r = [w for _, _, w in ref]
    h = [w for w, _ in hyp]
    out = jiwer.process_words(" ".join(r), " ".join(h))
    lat = []
    for ch in out.alignments[0]:
        if ch.type != "equal":
            continue
        for k in range(ch.ref_end_idx - ch.ref_start_idx):
            ri, hi = ch.ref_start_idx + k, ch.hyp_start_idx + k
            lat.append(hyp[hi][1] - ref[ri][1])
    return {"n_ref": len(r), "n_hyp": len(h), "hits": out.hits, "sub": out.substitutions, "del": out.deletions,
            "ins": out.insertions, "wer": out.wer, "latency_s": [round(x, 4) for x in lat]}


def cmd_run(a):
    import audioforge
    from audioforge.data import load_wav
    WORK.mkdir(parents=True, exist_ok=True)
    todo = [(m, st) for m in a.meetings for st in STARTS if not (WORK / f"{m}_{int(st)}.json").exists()]
    if not todo:
        print("all windows done"); return
    fe = audioforge.load(threads=2)
    t_start = time.time()
    cache = {}
    for m, st in todo:
        if time.time() - t_start > a.budget:
            print("budget reached; rerun to resume"); break
        if m not in cache:
            cache = {m: load_wav(str(AMI / "audio" / f"{m}.Mix-Headset.wav"), SR).astype(np.float32)}
        x = cache[m][int(st * SR):int((st + WIN_S) * SR)]
        ref = ref_words(m, st, st + WIN_S)
        r = stream(fe, x)
        sc = score(ref, r["hyp"])
        rec = {"meeting": m, "start": st, "dur": len(x) / SR, **sc, "block_ms_p50": float(np.median(r["block_ms"])),
               "block_ms_p90": float(np.percentile(r["block_ms"], 90)), "block_ms_max": float(np.max(r["block_ms"])),
               "stats": r["stats"]}
        (WORK / f"{m}_{int(st)}.json").write_text(json.dumps(rec))
        L = np.array(sc["latency_s"]) * 1000
        print(f"{m} {st:.0f}: WER {sc['wer']:.3f} hits {sc['hits']} latency p50 {np.median(L):.0f} p90 "
              f"{np.percentile(L, 90):.0f} ms; block p50 {rec['block_ms_p50']:.1f} ms", flush=True)


def cmd_report(a):
    recs = [json.loads(p.read_text()) for p in sorted(WORK.glob("*.json"))]
    L = np.concatenate([np.array(r["latency_s"]) for r in recs]) * 1000
    blk = [r["block_ms_p50"] for r in recs]
    blk90 = [r["block_ms_p90"] for r in recs]
    rng = np.random.default_rng(0)
    per = [np.array(r["latency_s"]) * 1000 for r in recs]
    boots = []
    for _ in range(1000):
        s = np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))])
        boots.append((np.median(s), np.percentile(s, 90), np.percentile(s, 95)))
    boots = np.array(boots)
    n_ref = sum(r["n_ref"] for r in recs)
    err = sum(r["sub"] + r["del"] + r["ins"] for r in recs)
    gpu_chunk_ms = 20.0  # research/GPU_RUN_2026-09-29.md (PR #1): 115M, one 160 ms chunk on the RTX 5090, CUDA
    med_cpu_block = float(np.median(blk))
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "definition": "per reference word correctly recognised: time the word (final spelling) first appears in the "
                      "streaming transcript minus the word's end time in the audio; audio clock + measured compute",
        "setup": "audioforge.load() (--mode single), CPU 2 threads (Apple laptop), 160 ms blocks, no network",
        "data": f"AMI test Mix-Headset, {len(recs)} windows x {WIN_S:.0f} s: "
                + ", ".join(sorted({r['meeting'] for r in recs})),
        "n_windows": len(recs), "n_ref_words": n_ref, "n_hits": int(len(L)),
        "wer_all_speech": round(err / n_ref, 4),
        "cpu": {"p50_ms": round(float(np.median(L))), "p90_ms": round(float(np.percentile(L, 90))),
                "p50_ci95": [round(float(x)) for x in np.percentile(boots[:, 0], [2.5, 97.5])],
                "p90_ci95": [round(float(x)) for x in np.percentile(boots[:, 1], [2.5, 97.5])],
                "p95_ms": round(float(np.percentile(L, 95))),
                "p95_ci95": [round(float(x)) for x in np.percentile(boots[:, 2], [2.5, 97.5])],
                "negative_share": round(float(np.mean(L < 0)), 3),
                "block_compute_ms_p50": round(med_cpu_block, 1), "block_compute_ms_p90": round(float(np.median(blk90)), 1),
                "block_compute_ms_max": round(max(r["block_ms_max"] for r in recs), 1)},
        "gpu_estimate": {"note": "CPU latency with the median CPU block compute replaced by the 5090's 20 ms/chunk "
                                 "(research/GPU_RUN_2026-09-29.md, PR #1); not measured end to end on the GPU",
                         "p50_ms": round(float(np.median(L)) - med_cpu_block + gpu_chunk_ms),
                         "p90_ms": round(float(np.percentile(L, 90)) - med_cpu_block + gpu_chunk_ms),
                         "p95_ms": round(float(np.percentile(L, 95)) - med_cpu_block + gpu_chunk_ms)},
        "windows": [{k: r[k] for k in ("meeting", "start", "n_ref", "hits", "wer", "block_ms_p50")}
                    | {"p50_ms": round(float(np.median(np.array(r["latency_s"]) * 1000))),
                       "p90_ms": round(float(np.percentile(np.array(r["latency_s"]) * 1000, 90)))} for r in recs],
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_windows", "n_hits", "wer_all_speech", "cpu", "gpu_estimate")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--meetings", nargs="+", default=list(MEETINGS))
    r.add_argument("--budget", type=float, default=480)
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(ROOT / "runs" / "stt_latency.json"))
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
