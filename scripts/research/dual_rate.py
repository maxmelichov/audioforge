"""research/DUAL_RATE.md: a dual-rate engine. The FAST pass (the served encoder at att [70,1] = 160 ms chunks) feeds
every head; a SLOW pass of the same frozen encoder at a longer trained chunk (att [70,6] = 560 ms or [70,13] =
1120 ms) produces the final transcript only (``audioforge-serve --final-chunk-ms``).

Both cores were trained by NVIDIA for att_context [70,0] / [70,1] / [70,6] / [70,13] = 80 / 160 / 560 / 1120 ms chunks
(research/archive/ENC_0P6B.md, NeMo configs). Stages (one process each, < 10 min, through scripts/dev/gate.sh):

  wer   --core 115m|0p6b --right R --set S   masked offline forward at [70,R] (== cache-aware streaming,
                                             tests/test_streaming.py), greedy RNNT, CPU 2 threads; S: ls_clean |
                                             ls_other | ami_eval | icsi_eval | live (the FINAL_COMPARE test sets)
  score                                      WER (Whisper English normaliser) + paired 95 % CIs vs the 160 ms row
  serve --core C --device D --chunk-ms F     the served single-mode engine over the 32 live sessions at 1x pacing
                                             (in process, 20 ms blocks): turn ends / fast finals vs single rate, the
                                             slow final's delay after turn_end, ms per 160 ms chunk, memory
  streams --core C --device D --chunk-ms F   real-time streams (mps_115m.py protocol)
  report                                     -> runs/dual_rate.json (+ numbers_final.json final/dualrate/*)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/dual_rate.py <stage> [...]
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
WORK = Path(os.environ.get("DUALRATE_W", SSD / "scratch" / "dualrate"))
OUT = ROOT / "runs" / "dual_rate.json"
SR = 16000
SETS = ("ls_clean", "ls_other", "ami_eval", "icsi_eval", "live")
RIGHTS = (0, 1, 6, 13)  # trained att_context right sizes: 80 / 160 / 560 / 1120 ms chunks
T_START = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def over(a) -> bool:
    return time.time() - T_START > a.budget


def jl_rows(p: Path) -> dict:
    return {r["i"]: r for r in map(json.loads, p.read_text().splitlines()) if r} if p.exists() else {}


def core_path(core: str) -> Path:
    import final_compare as FC
    return FC.AFM_115M if core == "115m" else Path(os.environ.get(
        "FINAL_0P6B_AFM", SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.3.afm"))


# =========================================================================== step 1: WER per chunk setting
def stage_wer(a):
    import torch
    import final_compare as FC
    torch.set_num_threads(2)
    xs, refs, _ = FC.asr_set(a.set)
    d = WORK / "wer"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{a.core}_r{a.right}_{a.set}.jsonl"
    done = jl_rows(p)
    todo = [i for i in range(len(refs)) if i not in done]
    log(f"wer {a.core} [70,{a.right}] {a.set}: {len(todo)} of {len(refs)} to do")
    if not todo:
        log("STAGE_COMPLETE")
        return
    from audioforge.train import load_model
    m = load_model(str(core_path(a.core)), "cpu").eval()
    att = [m.encoder.att_context_size[0], int(a.right)]

    def fn(x):
        with torch.inference_mode():
            return m.transcribe([np.asarray(x, np.float32)], head="rnnt", att_context_size=att)[0]
    fn(xs[todo[0]][: 3 * SR])  # warm-up
    n = 0
    with p.open("a") as f:
        for i in todo:
            if over(a):
                break
            t0 = time.perf_counter()
            h = fn(xs[i])
            f.write(json.dumps({"i": i, "hyp": h, "sec": round(time.perf_counter() - t0, 4),
                                "audio_sec": round(len(xs[i]) / SR, 3)}) + "\n")
            f.flush()
            n += 1
    log(f"wer {a.core} [70,{a.right}] {a.set}: {n} this call, {len(todo) - n} left")
    if n == len(todo):
        log("STAGE_COMPLETE")


def score_wer() -> dict:
    """Whisper EnglishTextNormalizer WER per core x chunk x set, 1000-resample bootstrap (utterances; clips for the
    live sessions), paired deltas against the 160 ms row on the same resamples (final_compare._boot_rate, seed 0)."""
    import final_compare as FC
    import hybrid_asr as H
    norms, _ = H._normalizers()
    fn = norms["whisper_norm"]
    res = {}
    for core in ("115m", "0p6b"):
        for s in SETS:
            hyp = {r: jl_rows(WORK / "wer" / f"{core}_r{r}_{s}.jsonl") for r in RIGHTS}
            _, refs, groups = FC.asr_set(s)
            n = len(refs)
            full = [r for r in RIGHTS if len(hyp[r]) >= n]
            if 1 not in full:
                continue
            rr = [fn(x) for x in refs]
            o, draws = {"n": n}, {}
            for r in full:
                e = np.array([H.edits_words(rr[i], fn(hyp[r][i]["hyp"])) for i in range(n)], np.float64)
                _, dr = FC._boot_rate(e, groups)
                draws[r] = dr
                sec = sum(x["sec"] for x in hyp[r].values())
                au = sum(x["audio_sec"] for x in hyp[r].values())
                o[f"r{r}"] = {"chunk_ms": (r + 1) * 80, "wer_pct": round(100 * e[:, 0].sum() / e[:, 1].sum(), 2),
                              "ci95": [round(100 * float(np.percentile(dr, q)), 2) for q in (2.5, 97.5)],
                              "errors": int(e[:, 0].sum()), "ref_words": int(e[:, 1].sum()),
                              "rtfx_cpu2": round(au / max(sec, 1e-9), 1)}
            for r in full:
                if r != 1:
                    o[f"r{r}"]["delta_vs_160_pp"] = round(o[f"r{r}"]["wer_pct"] - o["r1"]["wer_pct"], 2)
                    o[f"r{r}"]["delta_ci95"] = [round(100 * float(np.percentile(draws[r] - draws[1], q)), 2)
                                                for q in (2.5, 97.5)]
            # the 160 ms row must reproduce FINAL_COMPARE's core rows exactly (same model, same forward)
            fc = jl_rows(FC.WORK / "asr" / f"core_{core}_{s}.jsonl")
            if fc and len(fc) >= n and 1 in full:
                o["r1"]["same_hyps_as_final_compare"] = sum(fc[i]["hyp"] == hyp[1][i]["hyp"] for i in range(n))
            res[f"{core}|{s}"] = o
    return res


def stage_score(a):
    r = score_wer()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "wer_scores.json").write_text(json.dumps(r, indent=1))
    for k, o in r.items():
        row = "  ".join(f"{o[f'r{x}']['chunk_ms']}ms {o[f'r{x}']['wer_pct']:.2f}"
                        + (f" ({o[f'r{x}']['delta_vs_160_pp']:+.2f} {o[f'r{x}']['delta_ci95']})"
                           if f"r{x}" in o and "delta_vs_160_pp" in o[f"r{x}"] else "")
                        for x in RIGHTS if f"r{x}" in o)
        log(k, row, "same_as_FC:", o["r1"].get("same_hyps_as_final_compare"))


# =========================================================================== step 3: the served engine
os.environ.setdefault("FINAL_0P6B_AFM", str(SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.3.afm"))


def live_tag(a) -> str:
    return f"{a.core}_{a.device}_F{a.chunk_ms}" + ("" if a.flush == "on" else "_noflush")


def mem(device) -> dict:
    import resource
    import torch
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"peak_rss_mb": round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)}
    if device == "mps":
        out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20, 1)
    return out


def stage_live(a):
    """The served single-mode engine (audioforge.load, as audioforge-serve --mode single) on the 32 live sessions,
    fed in 20 ms blocks as fast as it runs (in process). Like the WebSocket handler, the session defers the slow
    flush: the batch with turn_end + final_fast is complete when process() returns; flush_slow() then computes the
    slow final (its wall time = the slow final's delay after the turn_end batch, plus a send)."""
    import torch
    import audioforge
    import final_compare as FC
    import soundfile as sf
    torch.set_num_threads(2)
    d = WORK / "live"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{live_tag(a)}.jsonl"
    S = FC.live_sessions()
    done = {r["key"] for r in map(json.loads, p.read_text().splitlines()) if r} if p.exists() else set()
    todo = [s for s in S if s["key"] not in done]
    log(f"live {live_tag(a)}: {len(todo)} of {len(S)} to do")
    if not todo:
        log("STAGE_COMPLETE")
        return
    core = "0.6b" if a.core == "0p6b" else "115m"
    fc = None if a.chunk_ms == 160 else a.chunk_ms
    t0 = time.perf_counter()
    fe = audioforge.load(core=core, asr=str(core_path(a.core)), device=a.device, threads=2, final_chunk_ms=fc,
                         final_flush=a.flush == "on", debug=True)
    load_s = time.perf_counter() - t0
    dev = a.device
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    n = 0
    for s in todo:
        if over(a):
            break
        x, sr = sf.read(s["wav"], dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        ss = fe.session()._s
        ss.defer_slow = True
        rec = {"key": s["key"], "audio_s": round(len(x) / SR, 3), "turn_ends": [], "finals": [], "slow": [],
               "batch_ms": [], "flush_wall_ms": [], "block_ms": []}
        with torch.inference_mode():
            for i in range(0, len(x), 320):
                tb = time.perf_counter()
                b = ss.process(x[i:i + 320])
                sync()
                bm = (time.perf_counter() - tb) * 1000
                rec["block_ms"].append(round(bm, 2))
                for m in b:
                    if m["type"] == "turn_end":
                        rec["turn_ends"].append(json.dumps(m, sort_keys=True))
                        rec["batch_ms"].append(round(bm, 2))
                    elif m["type"] in ("final", "final_fast"):
                        rec["finals"].append(json.dumps(dict(m, type="final"), sort_keys=True))
                if ss.la_pending and ss.e.dual:
                    tf = time.perf_counter()
                    sl = ss.flush_slow()
                    sync()
                    rec["flush_wall_ms"].append(round((time.perf_counter() - tf) * 1000, 2))
                    rec["slow"] += [dict(m, at_end=False) for m in sl]
            out = ss.finish()
        for m in out:
            if m["type"] == "turn_end":
                rec["turn_ends"].append(json.dumps(m, sort_keys=True))
            elif m["type"] == "final_fast" or (m["type"] == "final" and not ss.e.dual):
                rec["finals"].append(json.dumps(dict(m, type="final"), sort_keys=True))
            elif m["type"] == "final":
                rec["slow"].append(dict(m, at_end=True))
            elif m["type"] == "stats":
                rec["stats"] = m
        rec["chunk_ms"] = [round(c, 2) for c in ss.chunk_ms]
        rec["mem"] = mem(dev)
        rec["load_s"] = round(load_s, 1)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        log(f"{s['key']}: {len(rec['turn_ends'])} turn ends, {len(rec['slow'])} slow finals, "
            f"chunk p95 {np.percentile(rec['chunk_ms'], 95):.1f} ms")
    log(f"live {live_tag(a)}: {n} this call, {len(todo) - n} left")
    if n == len(todo):
        log("STAGE_COMPLETE")


def stage_cost(a):
    """final_compare.stage_cost (mps_115m.py protocols) with --final-chunk-ms: ms per 160 ms chunk of the full
    single-mode engine on the bundled clip (best of 3) or K interleaved real-time streams on AMI test windows."""
    import functools
    import audioforge
    import final_compare as FC
    FC.WORK = WORK / f"cost_F{a.chunk_ms}"
    fc = None if a.chunk_ms == 160 else a.chunk_ms
    audioforge.load = functools.partial(audioforge.load, final_chunk_ms=fc)
    a.sys = a.core
    FC.stage_cost(a)


def _pct(v, q):
    return round(float(np.percentile(v, q)), 1) if len(v) else None


def score_live() -> dict:
    """Per core / device: the latency gate (turn_end and fast-final messages byte-identical to single rate, per
    session), the slow final's delay after the turn_end batch (flush wall time; audio-time wait without the flush),
    block compute, memory, and the served WER of the concatenated finals (Whisper normaliser) vs the references."""
    import final_compare as FC
    import hybrid_asr as H
    norms, _ = H._normalizers()
    fn = norms["whisper_norm"]
    S = FC.live_sessions()
    refs = {s["key"]: fn(s["ref"]) for s in S}
    groups = [s["clip"] for s in S]
    res = {}
    for f in sorted((WORK / "live").glob("*.jsonl")):
        rows = {r["key"]: r for r in map(json.loads, f.read_text().splitlines()) if r}
        if len(rows) < len(S):
            continue
        tag = f.stem
        core, dev, F = tag.split("_")[:3]
        base = WORK / "live" / f"{core}_{dev}_F160.jsonl"
        o = {"n_sessions": len(rows), "audio_s": round(sum(r["audio_s"] for r in rows.values()), 1)}
        blocks = np.concatenate([r["block_ms"] for r in rows.values()])
        chunks = np.concatenate([r["chunk_ms"] for r in rows.values()])
        o["chunk_ms"] = {"p50": _pct(chunks, 50), "p95": _pct(chunks, 95), "p99": _pct(chunks, 99),
                         "max": round(float(chunks.max()), 1), "mean": round(float(chunks.mean()), 2)}
        o["block20_ms_max"] = round(float(blocks.max()), 1)
        o["rtf"] = round(float(blocks.sum()) / 1000 / o["audio_s"], 4)
        o["mem"] = {k: max(r["mem"].get(k, 0) for r in rows.values()) for k in rows[S[0]["key"]]["mem"]}
        o["n_turn_ends"] = sum(len(r["turn_ends"]) for r in rows.values())
        o["turn_end_batch_ms"] = {"p50": _pct([b for r in rows.values() for b in r["batch_ms"]], 50),
                                  "p95": _pct([b for r in rows.values() for b in r["batch_ms"]], 95)}
        if F != "F160" and base.exists():
            b = {r["key"]: r for r in map(json.loads, base.read_text().splitlines()) if r}
            o["gate"] = {"sessions_turn_ends_identical": sum(rows[k]["turn_ends"] == b[k]["turn_ends"] for k in rows),
                         "sessions_fast_finals_identical": sum(rows[k]["finals"] == b[k]["finals"] for k in rows),
                         "n": len(rows)}
            o["gate"]["pass"] = (o["gate"]["sessions_turn_ends_identical"] == len(rows)
                                 and o["gate"]["sessions_fast_finals_identical"] == len(rows))
            # the extra compute on the turn_end batch itself (paired: same turn, single vs dual rate)
            dd = [x - y for k in rows for x, y in zip(rows[k]["batch_ms"], b[k]["batch_ms"])]
            o["turn_end_batch_extra_ms"] = {"p50": _pct(dd, 50), "p95": _pct(dd, 95)}
        slow = [m for r in rows.values() for m in r["slow"] if not m["at_end"]]
        if slow:
            fl = [x for r in rows.values() for x in r["flush_wall_ms"]]
            lat = [m["latency_ms"] for m in slow]
            o["slow_final"] = {"n": len(slow), "pass_slow": sum(m["pass"] == "slow" for m in slow),
                               "latency_ms_field": {"p50": _pct(lat, 50), "p95": _pct(lat, 95),
                                                    "max": round(max(lat), 1)},
                               "flush_wall_ms": {"p50": _pct(fl, 50), "p95": _pct(fl, 95),
                                                 "max": round(max(fl), 1)} if fl else None}
            st = [r["stats"] for r in rows.values() if "stats" in r]
            o["slow_final"]["flush_miss"] = sum(x.get("final_flush_miss", 0) for x in st)
        # served WER: the session's finals (fast) and slow finals, concatenated, vs the reference
        for kind in ("fast", "slow"):
            E = []
            for s in S:
                r = rows[s["key"]]
                if kind == "fast":
                    hyp = " ".join(json.loads(m)["text"] for m in r["finals"])
                else:
                    if not r["slow"]:
                        E = None
                        break
                    hyp = " ".join(m["text"] for m in r["slow"])
                E.append(H.edits_words(refs[s["key"]], fn(hyp)))
            if E is None:
                continue
            E = np.array(E, np.float64)
            _, dr = FC._boot_rate(E, groups)
            o[f"served_wer_{kind}"] = {"wer_pct": round(100 * E[:, 0].sum() / E[:, 1].sum(), 2),
                                       "ci95": [round(100 * float(np.percentile(dr, q)), 2) for q in (2.5, 97.5)]}
        res[tag] = o
    return res


def score_cost() -> dict:
    out = {}
    for d in sorted(WORK.glob("cost_F*")):
        F = d.name.split("_F")[1]
        for core in ("115m", "0p6b"):
            for fn in ("engine", "streams_cpu", "streams_mps"):
                p = d / "cost" / core / f"{fn}.json"
                if p.exists():
                    v = json.loads(p.read_text())
                    o = out.setdefault(f"{core}|F{F}", {})
                    o[fn] = v
                    if fn.startswith("streams_"):  # the most interleaved sessions with block p95 < 160 ms
                        o[f"realtime_{fn}"] = max([int(k) for k, x in v.items() if x["real_time"]], default=0)
    return out


def stage_report(a):
    wer = json.loads((WORK / "wer_scores.json").read_text()) if (WORK / "wer_scores.json").exists() else score_wer()
    r = {"what": "research/DUAL_RATE.md: dual-rate engine (fast 160 ms pass for every head, slow 560 / 1120 ms pass "
                 "for the final text)", "wer": wer, "live": score_live(), "cost": score_cost()}
    OUT.write_text(json.dumps(r, indent=1))
    log(f"wrote {OUT}")
    print(json.dumps({"live": r["live"], "cost": r["cost"]}, indent=1)[:6000])


STAGES = {"wer": stage_wer, "score": stage_score, "live": stage_live, "cost": stage_cost, "report": stage_report}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--core", default="115m", choices=["115m", "0p6b"])
    p.add_argument("--right", type=int, default=1, choices=list(RIGHTS))
    p.add_argument("--set", default="ls_clean", choices=list(SETS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-ms", type=int, default=160)
    p.add_argument("--budget", type=float, default=540.0)
    p.add_argument("--flush", default="on", choices=["on", "off"])
    p.add_argument("--sub", default="engine", choices=["engine", "streams"])
    p.add_argument("--ks", default="1,2,3,4,5,6,7,8")
    p.add_argument("--seconds", type=float, default=60)
    a = p.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
