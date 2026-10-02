"""research/FINAL_COMPARE.md "Final transcript ready after you stop": ms from the end of a labelled user turn to that
turn's final transcript, for both audioforge cores (served engine, --final-chunk-ms 1120) and the offline baselines of
the words table, on the user channel of the live two-party sessions (FINAL_COMPARE live set; public test policy: not
used for any training or selection of the words path).

  run --system S    S: ours_115m_1120 / ours_0p6b_1120 (served single-mode engine on MPS, fed in 20 ms blocks; at
                    each labelled turn end the block is cut at the end sample and the clock covers the processing of
                    that last piece + the slow pass's flush (``LookaheadStream.flush_view``, a partial chunk up to the
                    turn end) + the token decode of the turn), or parakeet_tdt (CPU 2 threads), whisper_large /
                    whisper_turbo (transformers fp16 MPS, greedy), whisper_small (faster-whisper int8 CPU 2 threads):
                    the whole turn's audio is handed over at the turn end (an end-of-utterance STT, as in Pipecat /
                    LiveKit) and the clock runs to the returned text. Warm-up calls first (not timed), batch 1.
  report            -> runs/final_compare.json "final_latency" (p50 / p95 + 95 % bootstrap CI over turns, by length)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_latency.py run --system S
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
WORK = Path(os.environ.get("FINALLAT_W", SSD / "scratch" / "finallat"))
E2E = SSD / "scratch" / "e2e_tsvad"
OUT = ROOT / "runs" / "final_compare.json"
SR = 16000
AFM = {"ours_115m_1120": (ROOT / "runs" / "stage1_served_v4.afm", "115m"),
       "ours_0p6b_1120": (SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.3.afm", "0.6b")}
OFFLINE = {"parakeet_tdt": ("tdt_v3", "cpu"), "whisper_large": ("whisper_large_v3", "mps"),
           "whisper_turbo": ("whisper_turbo", "mps"), "whisper_small": ("whisper_small", "cpu")}
DEVICE = {"ours_115m_1120": "mps", "ours_0p6b_1120": "mps", **{k: v[1] for k, v in OFFLINE.items()}}
LABEL = {"ours_115m_1120": "audioforge 115M, --final-chunk-ms 1120 (served engine, MPS)",
         "ours_0p6b_1120": "audioforge 0.6B, --final-chunk-ms 1120 (served engine, MPS)",
         "parakeet_tdt": "NVIDIA Parakeet-TDT 0.6B v3, offline (audioforge.nemo_import, CPU 2 threads)",
         "whisper_large": "Whisper large-v3 (transformers fp16 MPS, greedy)",
         "whisper_turbo": "Whisper large-v3-turbo (transformers fp16 MPS, greedy)",
         "whisper_small": "Whisper small (faster-whisper int8, CPU 2 threads; LiveKit's default)"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sessions() -> list[dict]:
    """The user channel of the 16 TurnBench clips of the live set (final_compare.live_sessions, cond 'user'), with the
    labelled user turns (clips.json user_turns, s from the clip start)."""
    out = []
    for c in json.loads((E2E / "clips.json").read_text()):
        if c["set"] != "turnbench" or not c.get("text_user"):
            continue
        out.append({"clip": c["name"], "wav": str(E2E / "clips" / f"{c['name']}.user.wav"),
                    "turns": [list(map(float, t)) for t in c["user_turns"]]})
    return out


def load_wav(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(1)


def run_ours(a, S, f):
    import torch
    import audioforge
    torch.set_num_threads(2)
    afm, core = AFM[a.system]
    dev = DEVICE[a.system]
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    fe = audioforge.load(core=core, asr=str(afm), device=dev, threads=2, final_chunk_ms=1120, final_flush=True,
                         final_cut="speech")
    from audioforge.serve import FRAME_MS

    def one_session(x, turns, record):
        ss = fe.session()._s
        ss.defer_slow = True  # as the WebSocket handler: slow finals of the engine's own turn_ends flushed after
        tok = ss.e.asr.tokenizer
        rows = []
        ends = [round(e * SR) for _, e in turns]
        cuts = sorted(set(list(range(0, len(x), 320)) + ends + [len(x)]))
        with torch.inference_mode():
            for i, j in zip(cuts[:-1], cuts[1:]):
                is_end = j in ends
                t0 = time.perf_counter()
                ss.process(x[i:j])
                if is_end:
                    k = ends.index(j)
                    s_time = turns[k][0]
                    sync()
                    t1 = time.perf_counter()
                    view = ss.la.flush_view(j)
                    view = ss.la if view is None else view
                    sf_ = int(s_time * 1000 // FRAME_MS)
                    lo = view.tok_at[sf_ - 1] if 0 < sf_ <= len(view.tok_at) else 0
                    text = tok.decode(view.tokens[lo:])
                    sync()
                    t2 = time.perf_counter()
                    if record:
                        rows.append({"turn": k, "start": turns[k][0], "end": turns[k][1], "text": text,
                                     "ms": round((t2 - t0) * 1000, 3), "flush_ms": round((t2 - t1) * 1000, 3),
                                     "last_piece_ms": round((t1 - t0) * 1000, 3)})
                if ss.la_pending and ss.e.dual:
                    ss.flush_slow()
            ss.finish()
        return rows

    x0 = load_wav(S[0]["wav"])
    one_session(x0[: 12 * SR], [[1.0, 6.0], [7.0, 11.0]], False)  # warm-up (not timed, not recorded)
    for s in S:
        x = load_wav(s["wav"])
        for r in one_session(x, s["turns"], True):
            r["clip"] = s["clip"]
            f.write(json.dumps(r) + "\n")
            f.flush()
        log(f"{s['clip']}: {len(s['turns'])} turns")


def run_offline(a, S, f):
    import final_compare as FC
    system, dev = OFFLINE[a.system]
    fn = FC.make_asr(system, dev)
    x0 = load_wav(S[0]["wav"])
    for seg in (x0[: 3 * SR], x0[SR: 9 * SR], x0[: 35 * SR]):  # warm-up incl. the > 30 s path (not timed)
        fn(seg)
    for s in S:
        x = load_wav(s["wav"])
        for k, (st, en) in enumerate(s["turns"]):
            seg = np.ascontiguousarray(x[round(st * SR): round(en * SR)])
            t0 = time.perf_counter()
            text = fn(seg)
            dt = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"clip": s["clip"], "turn": k, "start": st, "end": en, "text": text,
                                "ms": round(dt, 3)}) + "\n")
            f.flush()
        log(f"{s['clip']}: {len(s['turns'])} turns")


def stage_run(a):
    S = sessions()
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"{a.system}.jsonl"
    t0 = time.time()
    with p.open("w") as f:
        (run_ours if a.system.startswith("ours") else run_offline)(a, S, f)
    log(f"{a.system}: {sum(len(s['turns']) for s in S)} turns in {time.time() - t0:.0f} s -> {p}")


def boot_ci(v, q, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    b = [np.percentile(v[rng.integers(0, len(v), len(v))], q) for _ in range(n)]
    return [round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)]


def stage_report(a):
    res = json.loads(OUT.read_text())
    o = {"what": "final transcript ready after the user stops talking: ms from the end of each labelled user turn "
                 "(clips.json user_turns) to that turn's final text, user channel of the 16 TurnBench clips of the "
                 "live set (the 'user' half of the 32 live sessions), batch 1, one stream, warm models",
         "method": __doc__.split("\n\n")[1].strip(),
         "bootstrap": "1000 resamples over turns, 95 % percentile interval", "systems": {}}
    for k in LABEL:
        p = WORK / f"{k}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln]
        v = np.array([r["ms"] for r in rows])
        L = np.array([r["end"] - r["start"] for r in rows])
        e = {"label": LABEL[k], "device": DEVICE[k], "n_turns": len(rows),
             "p50": round(float(np.percentile(v, 50)), 1), "p50_ci95": boot_ci(v, 50),
             "p95": round(float(np.percentile(v, 95)), 1), "p95_ci95": boot_ci(v, 95),
             "max": round(float(v.max()), 1), "by_length_p50": {}}
        for name, m in (("lt2s", L < 2), ("2to5s", (L >= 2) & (L <= 5)), ("gt5s", L > 5)):
            e["by_length_p50"][name] = {"n": int(m.sum()), "p50": round(float(np.percentile(v[m], 50)), 1)}
        if "flush_ms" in rows[0]:
            fl = np.array([r["flush_ms"] for r in rows])
            e["flush_only"] = {"p50": round(float(np.percentile(fl, 50)), 1), "p95": round(float(np.percentile(fl, 95)), 1)}
        o["systems"][k] = e
        L_all = L
    o["turn_length_s"] = {"n": int(len(L_all)), "p50": round(float(np.percentile(L_all, 50)), 2),
                          "p95": round(float(np.percentile(L_all, 95)), 2), "min": round(float(L_all.min()), 2),
                          "max": round(float(L_all.max()), 2), "lt2s": int((L_all < 2).sum()),
                          "2to5s": int(((L_all >= 2) & (L_all <= 5)).sum()), "gt5s": int((L_all > 5).sum())}
    res["final_latency"] = o
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"-> {OUT}")
    print(json.dumps(o["systems"], indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("run", "report"))
    p.add_argument("--system", default="", choices=("",) + tuple(LABEL))
    a = p.parse_args()
    (stage_run if a.stage == "run" else stage_report)(a)


if __name__ == "__main__":
    main()
