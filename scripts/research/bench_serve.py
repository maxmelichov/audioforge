"""Reproducible CPU benchmark of the served system (audioforge.serve Engine / Session, in process, no websocket).

    PYTHONPATH=. python scripts/research/bench_serve.py run --diar nemotron --clips e2e5 --out runs/perf/base_nemo.json \
        [--threads 2] [--policy timeout|hybrid_dyn] [--enroll dominant|after_agent_arm] [--streams K] \
        [--opt NAME ...] [--profile]
    PYTHONPATH=. python scripts/research/bench_serve.py compare runs/perf/base_nemo.json runs/perf/opt_nemo.json [--tol 1e-4]
    PYTHONPATH=. python scripts/research/bench_serve.py table runs/perf/*.json

What is measured (research/archive/PERFORMANCE.md):
* Every clip is streamed through a fresh ``Session`` in 20 ms blocks (the block size of a paced client and of the
  live runs in research/E2E_FINAL.md), as fast as the CPU allows (not paced). Per block: wall and process CPU time.
  RTF = processing wall time / audio time; ``cpu_rtf`` = process CPU time / audio time (both threads counted).
* Per-component wall time from the Session's own counters: the ASR pass (mel + encoder + RNNT + VAD / turn head on
  the session's frames), the speaker-conditioned turn pass (``asr.turn_ms``), the diarizer, Silero, the binder.
* ``chunk_ms`` p50 / p95: the server's own per-160-ms-of-audio processing time (``Session.chunk_ms``, the same
  number the live server reports as ``stats.chunk_ms_p95``).
* ``--streams K``: K sessions interleaved block by block on ONE worker (as the server's single executor thread),
  each clip starting at a different AMI offset. Latency per block is simulated in virtual time from the measured
  compute times: block i of stream s arrives at 20 ms * i + s * 20 ms / K, the worker takes blocks in arrival
  order, latency = finish - arrival. K streams are real time when the aggregate RTF (sum of compute / audio of one
  stream) stays < 1 and the latency does not grow with time (``lat_growth_ms`` = mean of the last 10 % of blocks -
  mean of the first 10 %).
* Messages: every protocol message except ``stats`` is kept per clip (``--dump``, default next to --out) so two
  runs can be compared for decision / transcript equality (``compare``).
The clips are fixed AMI Mix-Headset windows (the 5 AMI windows of research/E2E_FINAL.md plus longer ones for the
steady state, where the diarizer's speaker cache and FIFO are full: 30 s at the 0.32 s preset).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SR = 16000
BLOCK = 320  # 20 ms
AMI = ROOT / "data" / "ami" / "audio"
CLIP_SETS = {
    # research/E2E_FINAL.md's 5 AMI windows (runs/e2e_final.json "clips")
    "e2e5": [("ES2011b", 794.36, 20.0), ("IB4002", 22.17, 16.64), ("IS1008b", 557.64, 15.21),
             ("IS1008b", 1685.84, 20.0), ("TS3004b", 370.28, 16.59)],
    "long": [("ES2011b", 600.0, 120.0)],  # steady state: speaker cache + FIFO full after ~30 s
    "long2": [("ES2011b", 600.0, 120.0), ("IS1008b", 300.0, 120.0)],
    "smoke": [("ES2011b", 794.36, 6.0)],
}
STREAM_OFFSETS = [("ES2011b", 600.0), ("IS1008b", 300.0), ("TS3004b", 200.0), ("IB4002", 100.0), ("ES2012b", 400.0),
                  ("TS3009c", 500.0), ("EN2002a", 700.0), ("IS1009b", 800.0)]
NEMOTRON = dict(diar="runs/nemo_nemotron3_diar.afm", diar_pool="max", diar_left=1, diar_spks=4)
SORTFORMER = dict(diar="runs/nemo_sortformer_v2.afm")
ASR = "runs/stage1_served.afm"


def load_clip(conv: str, start: float, dur: float) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(AMI / f"{conv}.Mix-Headset.wav"), start=int(start * SR), stop=int((start + dur) * SR),
                    dtype="float32")
    assert sr == SR
    return x if x.ndim == 1 else x.mean(1)


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


def cur_rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2 ** 20, 1)
    except ImportError:
        return peak_rss_mb()


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def load_engine(a):
    import torch
    torch.set_num_threads(a.threads)
    import audioforge.serve as S
    d = NEMOTRON if a.diar == "nemotron" else SORTFORMER if a.diar == "sortformer" else dict(diar=a.diar)
    r0 = cur_rss_mb()
    t0 = time.perf_counter()
    eng = S.Engine.load(str(ROOT / a.asr), str(ROOT / d["diar"]), getattr(a, "device", "cpu"),
                        diar_pool=d.get("diar_pool"),
                        diar_spks=d.get("diar_spks"), threads=a.threads, diar_left=d.get("diar_left", S.DIAR_ENC_LEFT),
                        enroll=a.enroll, silero=str(ROOT / "data/silero/silero_vad_v5.onnx"),
                        preload_silero=a.policy in S.SILERO_POLICIES, perf="none")
    from audioforge import perf
    opts = perf.parse(",".join(["none"] + a.opt))
    info = perf.apply(eng.asr, eng.diar, opts)
    print(f"[bench] perf {opts} {info}", flush=True)
    load_s = time.perf_counter() - t0
    return S, eng, dict(load_s=round(load_s, 2), rss_after_load_mb=cur_rss_mb(), rss_before_load_mb=r0)


def strip(msgs):
    return [m for m in msgs if m.get("type") != "stats"]


def stream_clip(S, eng, x, a, arms=()):
    """One session over x in 20 ms blocks -> (messages, per-block (wall, cpu) ms, session)."""
    sess = S.Session(eng, S.SessionConfig(turn_policy=a.policy))
    msgs, blocks = [], []
    k = 0
    for i in range(0, len(x), BLOCK):
        while k < len(arms) and i / SR >= arms[k]:
            sess.arm_enrollment("agent_end")
            k += 1
        t0, c0 = time.perf_counter(), time.process_time()
        msgs += sess.process(x[i:i + BLOCK])
        blocks.append(((time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000))
    t0, c0 = time.perf_counter(), time.process_time()
    msgs += sess.finish()
    blocks.append(((time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000))
    return msgs, blocks, sess


def cmd_run(a):
    load0 = os.getloadavg()
    S, eng, meta = load_engine(a)
    if not a.no_warmup:
        eng.warmup()
    clips = CLIP_SETS[a.clips]
    res = dict(meta, argv=sys.argv[1:], diar=a.diar, threads=a.threads, policy=a.policy, enroll=a.enroll,
               opt=a.opt, clips=a.clips, load_start=[round(v, 2) for v in load0], per_clip={})
    dump = {}
    tot_audio = tot_wall = tot_cpu = 0.0
    comp = {"asr": 0.0, "turn": 0.0, "diar": 0.0, "silero": 0.0, "enroll": 0.0, "lookahead": 0.0}
    chunk_ms = []
    prof = None
    if a.profile:
        import cProfile
        prof = cProfile.Profile()
    if a.streams > 1:
        return cmd_streams(a, S, eng, res)
    for conv, st, dur in clips:
        x = load_clip(conv, st, dur)
        arms = [float(t) for t in np.arange(a.arm_every, dur, a.arm_every)] if a.enroll != "dominant" else []
        if prof:
            prof.enable()
        msgs, blocks, sess = stream_clip(S, eng, x, a, arms)
        if prof:
            prof.disable()
        name = f"{conv}_{st}"
        dump[name] = strip(msgs)
        w = sum(b[0] for b in blocks) / 1000
        c = sum(b[1] for b in blocks) / 1000
        au = len(x) / SR
        tot_audio, tot_wall, tot_cpu = tot_audio + au, tot_wall + w, tot_cpu + c
        cc = {"asr": sum(sess.asr_ms), "turn": sess.asr.turn_ms, "diar": sum(sess.diar_ms),
              "silero": sum(sess.sil.ms) if sess.sil is not None else 0.0, "enroll": sum(sess.enroll_ms),
              "lookahead": sess.la.ms if sess.la is not None else 0.0}
        for k_, v in cc.items():
            comp[k_] += v / 1000
        chunk_ms += sess.chunk_ms
        res["per_clip"][name] = dict(audio_s=round(au, 2), rtf=round(w / au, 4), cpu_rtf=round(c / au, 4),
                                     comp_rtf={k_: round(v / 1000 / au, 4) for k_, v in cc.items()},
                                     chunk_ms_p50=pct(sess.chunk_ms, 50), chunk_ms_p95=pct(sess.chunk_ms, 95),
                                     n_msgs=len(dump[name]))
        print(f"{name}: {au:.1f}s rtf {w / au:.3f} cpu {c / au:.3f} "
              + " ".join(f"{k_} {v / 1000 / au:.3f}" for k_, v in cc.items() if v) +
              f" chunk p95 {pct(sess.chunk_ms, 95)} ms", flush=True)
    res.update(audio_s=round(tot_audio, 2), rtf=round(tot_wall / tot_audio, 4), cpu_rtf=round(tot_cpu / tot_audio, 4),
               comp_rtf={k_: round(v / tot_audio, 4) for k_, v in comp.items()},
               chunk_ms_p50=pct(chunk_ms, 50), chunk_ms_p95=pct(chunk_ms, 95), chunk_ms_max=round(max(chunk_ms), 2),
               peak_rss_mb=peak_rss_mb(), rss_end_mb=cur_rss_mb(), load_end=[round(v, 2) for v in os.getloadavg()],
               swap_used_mb=_swap_mb())
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "per_clip"}), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    Path(a.dump or str(out).replace(".json", ".msgs.json")).write_text(json.dumps(dump))
    if prof:
        import pstats
        pstats.Stats(prof).sort_stats("cumulative").print_stats(45)
        pstats.Stats(prof).sort_stats("tottime").print_stats(30)


def cmd_streams(a, S, eng, res):
    """K interleaved sessions on one worker; virtual-time queueing latency from measured compute."""
    K, dur = a.streams, a.stream_seconds
    xs = [load_clip(c, st, dur) for c, st in (STREAM_OFFSETS * 4)[:K]]
    sessions = [S.Session(eng, S.SessionConfig(turn_policy=a.policy)) for _ in range(K)]
    n_blocks = min(len(x) for x in xs) // BLOCK
    batch = getattr(eng, "batch_process", None) if a.batch else None
    free = 0.0  # virtual time the worker becomes free (s)
    lat = [[] for _ in range(K)]
    comp_s = 0.0
    c0 = time.process_time()
    events = sorted(((i * BLOCK / SR + s * BLOCK / SR / K), s, i) for i in range(n_blocks) for s in range(K))
    j = 0
    while j < len(events):
        arr, s, i = events[j]
        start = max(free, arr)
        if batch is not None:  # every block that has arrived by the time the worker is free runs in one call
            grp = [events[j]]
            while j + len(grp) < len(events) and events[j + len(grp)][0] <= start and \
                    events[j + len(grp)][1] not in {g[1] for g in grp}:
                grp.append(events[j + len(grp)])
            t0 = time.perf_counter()
            batch([sessions[g[1]] for g in grp], [xs[g[1]][g[2] * BLOCK:(g[2] + 1) * BLOCK] for g in grp])
            dt = time.perf_counter() - t0
            free = start + dt
            for g in grp:
                lat[g[1]].append((free - g[0]) * 1000)
            j += len(grp)
        else:
            t0 = time.perf_counter()
            sessions[s].process(xs[s][i * BLOCK:(i + 1) * BLOCK])
            dt = time.perf_counter() - t0
            free = start + dt
            lat[s].append((free - arr) * 1000)
            j += 1
        comp_s += dt
    cpu = time.process_time() - c0
    audio = n_blocks * BLOCK / SR
    allat = [v for L in lat for v in L]
    n10 = max(1, len(lat[0]) // 10)
    growth = float(np.mean([np.mean(L[-n10:]) - np.mean(L[:n10]) for L in lat]))
    chunk = [v for s_ in sessions for v in s_.chunk_ms]
    res.update(streams=K, stream_seconds=dur, audio_s_per_stream=round(audio, 2), batch=bool(batch),
               agg_rtf=round(comp_s / audio, 4), per_stream_rtf=round(comp_s / audio / K, 4),
               agg_cpu_rtf=round(cpu / audio, 4), lat_ms_p50=pct(allat, 50), lat_ms_p95=pct(allat, 95),
               lat_ms_max=round(max(allat), 1), lat_growth_ms=round(growth, 1), realtime=bool(comp_s / audio < 1.0
                                                                                              and growth < 100),
               chunk_ms_p95=pct(chunk, 95), peak_rss_mb=peak_rss_mb(), rss_end_mb=cur_rss_mb(),
               load_end=[round(v, 2) for v in os.getloadavg()], swap_used_mb=_swap_mb())
    print(json.dumps(res), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ compare
def _cmp(a, b, tol, path, st):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            st["struct"].append(f"{path}: keys {sorted(set(a) ^ set(b))}")
            return
        for k in a:
            _cmp(a[k], b[k], tol, f"{path}.{k}", st)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            st["struct"].append(f"{path}: len {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, tol, f"{path}[{i}]", st)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(float(a) - float(b))
        key = path.rsplit(".", 1)[-1].split("[")[0]
        st["maxdiff"][key] = max(st["maxdiff"].get(key, 0.0), d)
        if d > tol:
            st["num"].append(f"{path}: {a} vs {b}")
    elif a != b:
        st["str"].append(f"{path}: {a!r} vs {b!r}")


def compare(pa: str, pb: str, tol: float = 1e-4) -> dict:
    A = json.loads(Path(pa.replace(".json", ".msgs.json") if not pa.endswith(".msgs.json") else pa).read_text())
    B = json.loads(Path(pb.replace(".json", ".msgs.json") if not pb.endswith(".msgs.json") else pb).read_text())
    st = {"struct": [], "num": [], "str": [], "maxdiff": {}}
    for clip in sorted(set(A) | set(B)):
        if clip not in A or clip not in B:
            st["struct"].append(f"{clip}: missing")
            continue
        _cmp(A[clip], B[clip], tol, clip, st)
    dec = {"turn_end": 0, "final": 0, "partial": 0, "primary": 0}
    for clip in set(A) & set(B):
        for kind in ("turn_end", "final", "partial"):
            ea = [m for m in A[clip] if m["type"] == kind]
            eb = [m for m in B[clip] if m["type"] == kind]
            dec[kind] += sum(x != y for x, y in zip(ea, eb)) + abs(len(ea) - len(eb))
        fa = [m["primary"] for m in A[clip] if m["type"] == "frame"]
        fb = [m["primary"] for m in B[clip] if m["type"] == "frame"]
        dec["primary"] += sum(x != y for x, y in zip(fa, fb)) + abs(len(fa) - len(fb))
    n_msgs = sum(len(v) for v in A.values())
    r = dict(n_msgs=n_msgs, identical=not (st["struct"] or st["num"] or st["str"]) and all(
        v == 0 for v in st["maxdiff"].values()), within_tol=not (st["struct"] or st["num"] or st["str"]),
             decision_diffs=dec, maxdiff={k: float(f"{v:.3g}") for k, v in st["maxdiff"].items()},
             n_struct=len(st["struct"]), n_num=len(st["num"]), n_str=len(st["str"]),
             examples=(st["struct"] + st["str"] + st["num"])[:8])
    return r


def cmd_compare(a):
    r = compare(a.a, a.b, a.tol)
    print(json.dumps(r, indent=1))
    return 0 if r["within_tol"] else 1


# ------------------------------------------------------------------------------------------------ components
class _Timer:
    """Wall-time accumulator installed around a bound method or module forward (nesting is the caller's business)."""

    def __init__(self):
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def wrap(self, obj, attr, name, split=None):
        fn = getattr(obj, attr)

        def w(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            key = name if split is None else split(name, a, k)
            dt = time.perf_counter() - t0
            self.t[key] = self.t.get(key, 0.0) + dt
            self.n[key] = self.n.get(key, 0) + 1
            return r
        setattr(obj, attr, w)
        return self

    def wrap_forward(self, module, name, split=None):
        """Module-level: every instance of ``module``'s class is timed (forward hooked on the instance)."""
        fn = module.forward

        def w(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            key = name if split is None else split(name, a, k)
            dt = time.perf_counter() - t0
            self.t[key] = self.t.get(key, 0.0) + dt
            self.n[key] = self.n.get(key, 0) + 1
            return r
        module.forward = w
        return self


def cmd_components(a):
    """Per-component wall time of the served path per 160 ms of audio: the leaf timers are installed around the
    encoder's subsampling / conformer layers (split by pass: ASR vs speaker-conditioned turn), the RNNT frame decode,
    the VAD and turn heads, the diarizer's encoder / Sortformer head / cache step, Silero and the mel front end. The
    speaker head (not on the per-chunk path: it embeds on demand) is timed separately on a 5 s window."""
    import torch
    load0 = os.getloadavg()
    S, eng, meta = load_engine(a)
    eng.warmup()
    T = _Timer()
    enc = eng.asr.encoder
    T.wrap(enc.pre_encode, "forward", "asr.subsample")
    T.wrap(enc, "_run_layers", "layers",
           split=lambda n, args, kw: "turn.layers" if (len(args) > 3 and args[3] is not None) or kw.get("spk_act") is not None else "asr.layers")
    T.wrap(enc, "stream_step", "encoder_step",
           split=lambda n, args, kw: "turn.encoder_step" if kw.get("spk_act") is not None else "asr.encoder_step")
    T.wrap(S.ASRStream, "_mel", "asr.mel")
    T.wrap(S.ASRStream, "_decode", "asr.rnnt_decode")
    T.wrap(S.ASRStream, "_turn_step", "turn.head")
    T.wrap(S.ASRStream, "feed_frames", "asr.pass_total")
    T.wrap(S.ASRStream, "run_turn_on_diar", "turn.pass_total")
    if eng.vad_name:
        T.wrap(eng.asr.heads[eng.vad_name], "forward", "asr.vad_head")
    from audioforge.streaming_diar import StreamingDiarizer, _MelStream
    T.wrap(StreamingDiarizer, "feed", "diar.total")
    T.wrap(StreamingDiarizer, "_frames", "diar.encode")
    T.wrap(_MelStream, "_mel", "diar.mel")
    T.wrap(eng.diar.heads[eng.diar_head], "forward_chunk", "diar.head")
    T.wrap(S.SileroSilence, "feed", "silero.total")
    T.wrap(S.Session, "process", "session.total")
    clips = CLIP_SETS[a.clips]
    tot_audio = 0.0
    chunk_ms, first_lag, first_onset = [], [], []
    for conv, st, dur in clips:
        x = load_clip(conv, st, dur)
        sess = S.Session(eng, S.SessionConfig(turn_policy=a.policy))
        msgs = []
        for i in range(0, len(x), BLOCK):
            msgs += sess.process(x[i:i + BLOCK], arrived=time.perf_counter())
        msgs += sess.finish()
        tot_audio += len(x) / SR
        chunk_ms += sess.chunk_ms
        if sess.first_partial_ms is not None:
            first_lag.append(sess.first_partial_ms)
        on = next((m["t"] for m in msgs if m["type"] == "frame" and m["vad"] > 0.5), None)
        fp = next((m["t"] for m in msgs if m["type"] == "partial" and m["text"]), None)
        if on is not None and fp is not None:
            first_onset.append(round((fp - on) * 1000, 1))
    per160 = {k: round(v / tot_audio * 160, 3) for k, v in T.t.items()}
    calls = {k: round(T.n[k] / (tot_audio / 0.16), 3) for k in T.t}
    leaves = ["asr.mel", "asr.subsample", "asr.layers", "asr.rnnt_decode", "asr.vad_head", "turn.layers", "turn.head",
              "diar.total", "silero.total"]
    other = per160["session.total"] - sum(per160.get(k, 0.0) for k in leaves)
    # speaker head: one embedding of the last 5 s (63 frames) from its layer tap, as an enrollment would run it
    spk = None
    spk_name = next((k for k, v in eng.asr.head_cfg.items() if v["type"] == "speaker"), None)
    if spk_name:
        h = eng.asr.heads[spk_name]
        e = torch.randn(1, 63, eng.asr.encoder.d_model)
        with torch.no_grad():
            h.embed(e, torch.tensor([63]))
            t0 = time.perf_counter()
            for _ in range(20):
                h.embed(e, torch.tensor([63]))
            spk = round((time.perf_counter() - t0) / 20 * 1000, 3)
    res = dict(meta, argv=sys.argv[1:], diar=a.diar, threads=a.threads, policy=a.policy, opt=a.opt, clips=a.clips,
               audio_s=round(tot_audio, 2), load_start=[round(v, 2) for v in load0],
               load_end=[round(v, 2) for v in os.getloadavg()], swap_used_mb=_swap_mb(),
               ms_per_160ms=per160, calls_per_160ms=calls, python_and_other_ms_per_160ms=round(other, 3),
               rtf=round(per160["session.total"] / 160, 4),
               chunk_ms_p50=pct(chunk_ms, 50), chunk_ms_p95=pct(chunk_ms, 95), chunk_ms_max=round(max(chunk_ms), 2),
               first_partial_server_lag_ms=first_lag, first_partial_after_onset_ms=first_onset,
               speaker_head_embed_5s_ms=spk, peak_rss_mb=peak_rss_mb(), rss_end_mb=cur_rss_mb())
    for k in sorted(per160):
        print(f"{k:22s} {per160[k]:8.3f} ms/160ms  ({calls[k]:.2f} calls)")
    print(f"{'python+other':22s} {other:8.3f} ms/160ms")
    print(json.dumps({k: v for k, v in res.items() if k not in ("ms_per_160ms", "calls_per_160ms")}))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def _swap_mb():
    try:
        import subprocess
        s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(s.split("used = ")[1].split("M")[0])
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ ab
def cmd_ab(a):
    """Interleaved A/B on one shared machine: two engines (A = --opt-a, B = --opt-b) in one process, every clip run
    A, B, A, B ... (--reps), so both see the same load; reports per-config RTF / component sums / chunk p95, the ratio
    B / A, and whether B's protocol messages equal A's (bit-identical decisions and probabilities)."""
    import torch
    torch.set_num_threads(a.threads)
    import audioforge.serve as S
    from audioforge import perf
    d = NEMOTRON if a.diar == "nemotron" else SORTFORMER if a.diar == "sortformer" else dict(diar=a.diar)
    engs, infos = [], []
    for spec in (a.opt_a, a.opt_b):
        eng = S.Engine.load(str(ROOT / a.asr), str(ROOT / d["diar"]), "cpu", diar_pool=d.get("diar_pool"),
                            diar_spks=d.get("diar_spks"), threads=a.threads, diar_left=d.get("diar_left", S.DIAR_ENC_LEFT),
                            enroll=a.enroll, silero=str(ROOT / "data/silero/silero_vad_v5.onnx"),
                            preload_silero=a.policy in S.SILERO_POLICIES, perf="none")
        infos.append(perf.apply(eng.asr, eng.diar, perf.parse(spec)))
        eng.warmup()
        engs.append(eng)
    print(f"[ab] A {a.opt_a} {infos[0]} | B {a.opt_b} {infos[1]}", flush=True)
    npar = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    e0 = engs[0]
    print(f"[ab] params: asr encoder {npar(e0.asr.encoder) / 1e6:.1f}M, diar encoder {npar(e0.diar.encoder) / 1e6:.1f}M, "
          f"diar head {npar(e0.diar.heads[e0.diar_head]) / 1e6:.1f}M", flush=True)
    clips = CLIP_SETS[a.clips]
    tot = [dict(wall=0.0, cpu=0.0, asr=0.0, turn=0.0, diar=0.0, sil=0.0, chunk=[]) for _ in range(2)]
    audio = 0.0
    msgs = [{}, {}]
    for conv, st, dur in clips:
        x = load_clip(conv, st, dur)
        audio += len(x) / SR * a.reps
        for r in range(a.reps):
            for i in (0, 1):
                perf.apply(engs[i].asr, engs[i].diar, perf.parse((a.opt_a, a.opt_b)[i]))  # class-level flags
                m, blocks, sess = stream_clip(S, engs[i], x, a)
                t = tot[i]
                t["wall"] += sum(b[0] for b in blocks) / 1000
                t["cpu"] += sum(b[1] for b in blocks) / 1000
                t["asr"] += sum(sess.asr_ms) / 1000
                t["turn"] += sess.asr.turn_ms / 1000
                t["diar"] += sum(sess.diar_ms) / 1000
                t["sil"] += (sum(sess.sil.ms) if sess.sil is not None else 0.0) / 1000
                t["chunk"] += sess.chunk_ms
                if r == 0:
                    msgs[i][f"{conv}_{st}"] = strip(m)
        print(f"{conv}_{st}: A rtf {tot[0]['wall'] / audio * 2:.3f} B {tot[1]['wall'] / audio * 2:.3f}", flush=True)
    audio_per = audio / 2 * 2 / a.reps / a.reps if False else audio  # audio streamed per config
    out = dict(argv=sys.argv[1:], diar=a.diar, threads=a.threads, policy=a.policy, clips=a.clips, reps=a.reps,
               opt_a=a.opt_a, opt_b=a.opt_b, perf_info=infos, audio_s_per_config=round(audio_per, 1),
               load_end=[round(v, 2) for v in os.getloadavg()], swap_used_mb=_swap_mb(), peak_rss_mb=peak_rss_mb())
    for i, k in enumerate("AB"):
        t = tot[i]
        out[k] = dict(rtf=round(t["wall"] / audio_per, 4), cpu_rtf=round(t["cpu"] / audio_per, 4),
                      asr_rtf=round(t["asr"] / audio_per, 4), turn_rtf=round(t["turn"] / audio_per, 4),
                      diar_rtf=round(t["diar"] / audio_per, 4), silero_rtf=round(t["sil"] / audio_per, 4),
                      chunk_ms_p50=pct(t["chunk"], 50), chunk_ms_p95=pct(t["chunk"], 95))
    out["ratio_B_over_A"] = {k: round(out["B"][k] / out["A"][k], 3) if out["A"][k] else None
                             for k in ("rtf", "cpu_rtf", "asr_rtf", "turn_rtf", "diar_rtf", "chunk_ms_p95")}
    pa, pb = Path(a.out).with_suffix(".A.msgs.json"), Path(a.out).with_suffix(".B.msgs.json")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pa.write_text(json.dumps(msgs[0]))
    pb.write_text(json.dumps(msgs[1]))
    out["compare"] = compare(str(pa), str(pb), a.tol)
    print(json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ audit
def _time_blocks(fn, x):
    """Feed x in 20 ms blocks to fn(block) -> (wall s, cpu s)."""
    t0, c0 = time.perf_counter(), time.process_time()
    for i in range(0, len(x), BLOCK):
        fn(x[i:i + BLOCK])
    return time.perf_counter() - t0, time.process_time() - c0


def cmd_audit(a):
    """Re-measure the figures research/archive/PERFORMANCE.md section 1 reconciles, under one stated condition: the served
    models, CPU, --threads, fixed AMI clips, one process, wall and process CPU time, load average recorded."""
    import torch
    torch.set_num_threads(a.threads)
    import audioforge.serve as S
    from audioforge.nemo_import import load_any
    from audioforge.streaming_diar import StreamingDiarizer
    from audioforge.train import load_model
    res = dict(threads=a.threads, load_start=[round(v, 2) for v in os.getloadavg()], rows={})
    clips = [load_clip(*c) for c in CLIP_SETS[a.clips]]
    longc = load_clip(*CLIP_SETS["long"][0])
    audio = sum(len(x) for x in clips) / SR

    def row(name, wall, cpu, au, **kw):
        res["rows"][name] = dict(rtf=round(wall / au, 4), cpu_rtf=round(cpu / au, 4), audio_s=round(au, 1),
                                 ms_per_160ms=round(wall / au * 160, 2), load=round(os.getloadavg()[0], 2), **kw)
        print(name, res["rows"][name], flush=True)

    asr = load_model(str(ROOT / a.asr), "cpu").eval()
    for fast in (False, True):
        if fast:
            S.fast_conv(asr)
        # streaming ASR pass alone (mel + encoder + RNNT + VAD head; no turn head, no diarizer), as ASRStream
        w = c = 0.0
        for x in clips:
            st = S.ASRStream(asr, "vad", None, "session")
            dw, dc = _time_blocks(lambda b: st.feed_frames(b), x)
            st.feed_frames(np.zeros(0, np.float32), final=True)
            w, c = w + dw, c + dc
        row(f"asr_stream_pass{'_fastconv' if fast else ''}", w, c, audio, what="ASRStream.feed_frames, 20 ms blocks")
    # offline masked forward over each whole clip (final_asr.json's m.transcribe), batch 1
    with torch.no_grad():
        t0, c0 = time.perf_counter(), time.process_time()
        for x in clips:
            asr.transcribe([x], head="rnnt")
        row("asr_offline_transcribe_b1", time.perf_counter() - t0, time.process_time() - c0, audio,
            what="SpeechModel.transcribe, whole clip, batch 1 (fast conv on)")
    for dname, d in (("sortformer", SORTFORMER), ("nemotron", NEMOTRON)):
        dm = load_any(str(ROOT / d["diar"]), "cpu").eval()
        S.fast_conv(dm)
        for h in dm.heads.values():
            if d.get("diar_pool") and hasattr(h, "pool"):
                h.pool = d["diar_pool"]
        head = next(k for k, v in dm.head_cfg.items() if v["type"] == "sortformer")
        mode = "causal" if dm.encoder.causal else "window"
        kw = dict(S.diar_preset(S.DEFAULT_DIAR_CONFIG))
        if mode == "window":
            kw["enc_left_context"] = d.get("diar_left", S.DIAR_ENC_LEFT)
        for tag, xs in (("e2e5", clips), ("long120", [longc])):
            w = c = 0.0
            for x in xs:
                sd = StreamingDiarizer(dm, head, mode=mode, **kw)
                dw, dc = _time_blocks(lambda b: sd.feed(b), x)
                w, c = w + dw, c + dc
            au = sum(len(x) for x in xs) / SR
            row(f"diar_{dname}_stream_{tag}", w, c, au, what=f"StreamingDiarizer {mode} 0.32 s preset, 20 ms blocks")
        with torch.no_grad():
            t0, c0 = time.perf_counter(), time.process_time()
            for x in clips:
                xt = torch.as_tensor(x)[None]
                enc, elen = dm.encode(xt, torch.tensor([xt.shape[1]]))
                dm.heads[head](enc, elen)
            row(f"diar_{dname}_offline_b1", time.perf_counter() - t0, time.process_time() - c0, audio,
                what="one offline pass per clip, batch 1")
        del dm
    res["load_end"] = [round(v, 2) for v in os.getloadavg()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def cmd_table(a):
    print("| run | diar | opt | audio s | RTF | CPU RTF | ASR | turn | diar | chunk p50 | chunk p95 | peak RSS MB | load |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.files:
        if p.endswith(".msgs.json"):
            continue
        r = json.loads(Path(p).read_text())
        if "streams" in r:
            print(f"| {Path(p).stem} | {r['diar']} | {','.join(r['opt']) or '-'} | {r['streams']} streams x "
                  f"{r['audio_s_per_stream']} | agg {r['agg_rtf']} | {r['agg_cpu_rtf']} | | | | lat p50 "
                  f"{r['lat_ms_p50']} | lat p95 {r['lat_ms_p95']} | {r['peak_rss_mb']} | {r['load_start'][0]} |")
            continue
        c = r["comp_rtf"]
        print(f"| {Path(p).stem} | {r['diar']} | {','.join(r['opt']) or '-'} | {r['audio_s']} | {r['rtf']} | "
              f"{r['cpu_rtf']} | {c['asr']} | {c['turn']} | {c['diar']} | {r['chunk_ms_p50']} | {r['chunk_ms_p95']} | "
              f"{r['peak_rss_mb']} | {r['load_start'][0]} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--asr", default=ASR)
    r.add_argument("--diar", default="nemotron", help="nemotron | sortformer | a .afm path")
    r.add_argument("--clips", default="e2e5", choices=list(CLIP_SETS))
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--policy", default="timeout")
    r.add_argument("--enroll", default="dominant")
    r.add_argument("--arm-every", type=float, default=8.0, help="--enroll after_agent_arm: agent_end every N s")
    r.add_argument("--opt", action="append", default=[], help="Engine perf option name[=value] (repeatable)")
    r.add_argument("--device", default="cpu", help="cpu | mps | cuda[:N] (serve --device; research/MPS_115M.md)")
    r.add_argument("--streams", type=int, default=1)
    r.add_argument("--stream-seconds", type=float, default=60.0)
    r.add_argument("--batch", action="store_true", help="--streams: batch the pending blocks of all sessions")
    r.add_argument("--no-warmup", action="store_true")
    r.add_argument("--profile", action="store_true")
    r.add_argument("--out", required=True)
    r.add_argument("--dump", default=None)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-4)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    b = sub.add_parser("ab", help="interleaved A/B of two --opt specs in one process (fair under load)")
    b.add_argument("--asr", default=ASR)
    b.add_argument("--diar", default="nemotron")
    b.add_argument("--clips", default="e2e5", choices=list(CLIP_SETS))
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--policy", default="timeout")
    b.add_argument("--enroll", default="dominant")
    b.add_argument("--opt-a", default="none")
    b.add_argument("--opt-b", default="default")
    b.add_argument("--reps", type=int, default=1)
    b.add_argument("--arm-every", type=float, default=8.0)
    b.add_argument("--tol", type=float, default=1e-4)
    b.add_argument("--out", required=True)
    co = sub.add_parser("components", help="per-component ms per 160 ms of audio (research/archive/PERFORMANCE.md section 2)")
    co.add_argument("--asr", default=ASR)
    co.add_argument("--diar", default="nemotron")
    co.add_argument("--clips", default="e2e5", choices=list(CLIP_SETS))
    co.add_argument("--threads", type=int, default=2)
    co.add_argument("--policy", default="hybrid_dyn")
    co.add_argument("--enroll", default="dominant")
    co.add_argument("--opt", action="append", default=[])
    co.add_argument("--out", required=True)
    au = sub.add_parser("audit", help="re-measure the component figures of research/archive/PERFORMANCE.md section 1")
    au.add_argument("--asr", default=ASR)
    au.add_argument("--clips", default="e2e5", choices=list(CLIP_SETS))
    au.add_argument("--threads", type=int, default=2)
    au.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "table": cmd_table, "audit": cmd_audit,
            "components": cmd_components, "ab": cmd_ab}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
