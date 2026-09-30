#!/usr/bin/env python
"""On-device streaming benchmark for the unified front-end (research/archive/ONDEVICE.md).

Full pipeline per chunk: incremental log-mel -> cache-aware FastConformer step -> TDT greedy
(+ CTC greedy, VAD + EOU frame heads, speaker layer-mix; speaker embedding pooled every 1 s over
the last 5 s) on CPU, fixed thread count. Random weights (speed only); the TDT joint's blank /
duration biases are set so the random model emits ~0.3 tokens per frame with duration 1 (realistic
BPE-1024 token rate; duration 1 means no TDT frame skipping, i.e. a conservative decode cost).

The joint's prediction projection is zeroed (same FLOPs) and the blank bias re-calibrated per backend so
every backend decodes the same token rate.

Every measurement runs in a fresh subprocess (clean peak RSS, no cross-talk):

    python scripts/archive/bench_ondevice.py sweep  --out DIR          # all sizes x contexts x backends x threads
    python scripts/archive/bench_ondevice.py one    --size M --att 70,1 --backend trace --threads 1
    python scripts/archive/bench_ondevice.py report --out DIR          # markdown table from DIR/results.jsonl
    python scripts/archive/bench_ondevice.py profile --size M --att 70,1  # torch.profiler breakdown
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from audioforge.model import SpeechModel  # noqa: E402
from audioforge.runtime import StreamingRuntime, export_chunk_onnx, time_stream  # noqa: E402
from audioforge.tokenizer import CharTokenizer  # noqa: E402

# name: (d_model, n_layers, n_heads). S/M/L are the parameter targets (30-40M / 100-120M / Parakeet-0.6B);
# "S512x12" and "M768x16" are the literal shapes from the task brief (they land at 80M / 227M).
SIZES = {
    "S": (256, 18, 4),
    "M": (512, 17, 8),       # = NVIDIA FastConformer-L streaming (streaming_multi 114M, EOU-120M)
    "L": (1024, 24, 8),      # = Parakeet / Nemotron 0.6B encoder
    "S512x12": (512, 12, 8),
    "M768x16": (768, 16, 8),
}
ATTS = {"70,1": [70, 1], "70,6": [70, 6]}
DEFAULT_OUT = Path(os.environ.get("ONDEVICE_OUT", "bench_ondevice_out"))


def tokenizer(vocab: int = 1024):
    return CharTokenizer([chr(0x4E00 + i) for i in range(vocab - 4)])


def prod_cfg(size: str, att) -> dict:
    d, n, h = SIZES[size]
    return {
        "preprocessor": {"n_mels": 80, "normalize": "fixed", "dither": 0.0},
        "encoder": dict(d_model=d, n_layers=n, n_heads=h, subsampling_channels=256, conv_kernel=9,
                        causal=True, att_context_size=list(att), att_context_sizes=[[70, 1], [70, 6]],
                        dropout=0.0),
        "heads": {
            "tdt": {"type": "tdt", "pred_hidden": 640, "joint_hidden": 640},
            "ctc": {"type": "ctc"},
            "vad": {"type": "frame", "key": "vad", "hidden": 128},
            "eou": {"type": "frame", "key": "eou", "hidden": 128},
            "spk": {"type": "speaker", "num_speakers": 1, "emb_dim": 192, "from_layers": "all"},
        },
        "decoding": {"primary": "tdt"},
    }


def bench_audio(seconds: float) -> np.ndarray:
    from audioforge.data import synthetic_dataset
    out, n, i = [], int(seconds * 16000), 0
    while sum(map(len, out)) < n:
        out += [e["audio"] for e in synthetic_dataset("asr", 32, seed=100 + i)]
        i += 1
    return np.concatenate(out)[:n].astype(np.float32)


@torch.no_grad()
def calibrate_tdt(m: SpeechModel, rate: float = 0.3):
    """Bias the random joint so greedy TDT emits ~``rate`` tokens per frame, always with duration 1."""
    h = m.heads["tdt"]
    # random weights make emission self-reinforcing through the LSTM (rate jumps between ~0 and ~1);
    # zeroing the joint's prediction projection (same FLOPs) makes the rate a smooth function of the bias
    h.joint.pred.weight.zero_()
    b = h.joint.out[-1].bias
    b[h.vocab_size + 1 + 1] += 30.0  # durations[1] == 1
    a = torch.from_numpy(bench_audio(8.0))[None]
    enc, _ = m.encode(a, torch.tensor([a.shape[1]]), [70, 1])
    base, lo, hi = float(b[h.blank]), -20.0, 20.0
    for _ in range(16):
        mid = (lo + hi) / 2
        b[h.blank] = base + mid
        r = len(h._greedy(enc[0])) / enc.shape[1]
        lo, hi = (lo, mid) if r < rate else (mid, hi)
    b[h.blank] = base + (lo + hi) / 2


@torch.no_grad()
def calibrate_runtime(rt: StreamingRuntime, rate: float = 0.3, seconds: float = 8.0):
    """Per-backend: re-set the blank bias on *this runtime's* encoder outputs so every backend decodes
    the same ~``rate`` tokens/frame (random weights + int8/bf16 noise otherwise change the decode load)."""
    fs, orig = [], rt._decode_primary
    rt._decode_primary = lambda f: fs.append(f.clone())
    audio = bench_audio(seconds)
    rt.reset()
    rt.feed(audio)
    rt._decode_primary = orig
    h = rt.head
    b = h.joint.out[-1].bias if not hasattr(h.joint.out[-1], "_packed_params") else None
    if b is None:
        return
    base, lo, hi, n = float(b[h.blank]), -20.0, 20.0, sum(f.shape[0] for f in fs)
    for _ in range(14):
        mid = (lo + hi) / 2
        b[h.blank] = base + mid
        rt.reset()
        for f in fs:
            rt._decode_primary(f)
        lo, hi = (lo, mid) if len(rt.tokens) / n < rate else (mid, hi)
    b[h.blank] = base + (lo + hi) / 2
    rt.reset()


def model_path(out: Path, size: str) -> Path:
    return out / "models" / f"{size}.pt"


def get_model(out: Path, size: str, att) -> tuple[SpeechModel, float]:
    """Build (and cache) the random model; returns (model, load_seconds) with load timed from disk."""
    p = model_path(out, size)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(0)
        m = SpeechModel(prod_cfg(size, [70, 1]), tokenizer()).eval()
        calibrate_tdt(m)
        torch.save(m.state_dict(), p)
        del m
    t0 = time.perf_counter()
    m = SpeechModel(prod_cfg(size, att), tokenizer())
    m.load_state_dict(torch.load(p, map_location="cpu", mmap=False))
    m.eval()
    return m, time.perf_counter() - t0


def onnx_path(out: Path, size: str, att_key: str, int8: bool = False) -> Path:
    return out / "onnx" / f"{size}_{att_key.replace(',', '_')}{'_int8' if int8 else ''}.onnx"


def ensure_onnx(out: Path, m: SpeechModel, size: str, att_key: str, int8: bool) -> Path:
    p = onnx_path(out, size, att_key, int8)
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    fp = onnx_path(out, size, att_key, False)
    if not fp.exists():
        export_chunk_onnx(m, fp, ATTS[att_key])
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp), str(p), weight_type=QuantType.QInt8)
    return p


def coreml_path(out: Path, size: str, att_key: str, fp16: bool = False) -> Path:
    return out / "coreml" / f"{size}_{att_key.replace(',', '_')}{'_fp16' if fp16 else ''}.mlpackage"


def ensure_coreml(out: Path, m: SpeechModel, size: str, att_key: str, fp16: bool = False) -> Path:
    p = coreml_path(out, size, att_key, fp16)
    if p.exists():
        return p
    import coremltools as ct
    from audioforge.runtime import ChunkEncoder, _example_inputs, streamable_heads
    heads = streamable_heads(m)
    ce = ChunkEncoder(m, ATTS[att_key], heads, "tdt").eval()

    class Rank1(torch.nn.Module):  # Core ML rejects rank-0 inputs: scalars come in as shape (1,)
        def __init__(self, ce):
            super().__init__()
            self.ce = ce

        def forward(self, mel, inv, offset, nvalid, kc, vc, cc):
            return self.ce(mel, inv[0].long(), offset[0].long(), nvalid[0].long(), kc, vc, cc)
    ex = _example_inputs(ce, m.preprocessor.n_mels)
    ex = tuple(e.to(torch.int32).reshape(1) if e.dtype == torch.int64 else e for e in ex)
    with torch.no_grad():
        tr = torch.jit.trace(Rank1(ce).eval(), ex, check_trace=False)
    names = ["mel", "inv", "offset", "nvalid", "kc", "vc", "cc"]
    mlm = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=e.shape, dtype=np.int32 if e.dtype == torch.int32
                                               else np.float32) for n, e in zip(names, ex)],
                     outputs=[ct.TensorType(name=n) for n in ce.output_names()],
                     convert_to="mlprogram",
                     compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
                     minimum_deployment_target=ct.target.macOS15)
    p.parent.mkdir(parents=True, exist_ok=True)
    mlm.save(str(p))
    return p


# ------------------------------------------------------------------------------------------ one run
def run_one(a) -> dict:
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    att = ATTS[a.att]
    backend, quant, dtype = a.backend, None, torch.float32
    rt_backend = backend
    if backend == "int8":
        rt_backend, quant = "eager", "int8"
    elif backend == "trace-int8":
        rt_backend, quant = "trace", "int8"
    elif backend in ("bf16", "fp16"):
        rt_backend, dtype = "eager", {"bf16": torch.bfloat16, "fp16": torch.float16}[backend]
    elif backend in ("onnx", "onnx-int8"):
        rt_backend = "onnx"
    m, load_s = get_model(out, a.size, att)
    n_params = m.num_params()
    kw = {}
    if backend.startswith("onnx"):
        kw["onnx_path"] = ensure_onnx(out, m, a.size, a.att, backend.endswith("int8"))
    if backend.startswith("coreml"):
        kw["coreml_path"] = ensure_coreml(out, m, a.size, a.att, fp16=backend == "coreml-ane")
        kw["coreml_units"] = {"coreml-cpu": "CPU_ONLY", "coreml-all": "ALL", "coreml-ane": "CPU_AND_NE"}[backend]
        rt_backend = "coreml"
    t0 = time.perf_counter()
    rt = StreamingRuntime(m, att_context_size=att, backend=rt_backend, quantize=quant, dtype=dtype,
                          threads=a.threads, copy_model=False, free_encoder=True, **kw)
    prep_s = time.perf_counter() - t0
    if quant or dtype != torch.float32:
        del m  # the runtime holds its own (quantized / cast) copy
    chunk = np.zeros(rt.chunk_mel * rt.hop + rt.n_fft, np.float32)  # enough samples for exactly one step
    t0 = time.perf_counter()
    rt.feed(chunk)  # cold first step (includes lazy JIT / compile / allocator)
    cold_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    rt.warmup(5)
    warm_s = time.perf_counter() - t0
    calibrate_runtime(rt)
    audio = bench_audio(a.seconds)
    reps = []
    for _ in range(a.repeats):
        r = time_stream(rt, audio)
        total = sum(r["step_ms"])  # includes the amortized speaker embedding
        reps.append(dict(ms_chunk=total / r["steps"], p90=float(np.percentile(r["step_ms"], 90)),
                         cpu_chunk=sum(r["cpu_ms"]) / r["steps"], user_chunk=r["user_ms"] / r["steps"],
                         sys_chunk=r["sys_ms"] / r["steps"], majflt=r["majflt"], minflt=r["minflt"],
                         spk_ms=float(np.mean(r["spk_ms"])) if r["spk_ms"] else 0.0,
                         tok_per_s=r["tokens"] / a.seconds, steps=r["steps"]))
    # one median for the headline and the per-field numbers: the repeat closest to the (possibly interpolated)
    # median ms/chunk (for an odd count the middle repeat, as before)
    warm_step = statistics.median(r["ms_chunk"] for r in reps)
    med = min(reps, key=lambda r: abs(r["ms_chunk"] - warm_step))
    ms_chunk = warm_step
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_mb = maxrss / 2 ** 20 if sys.platform == "darwin" else maxrss / 1024
    res = dict(size=a.size, params_m=round(n_params / 1e6, 2), att=a.att, backend=backend, threads=a.threads, seconds=a.seconds, repeats=a.repeats,
               chunk_ms=rt.chunk_ms, ms_chunk=round(ms_chunk, 3), ms_per_80ms=round(ms_chunk / rt.cs, 3),
               rtf=round(ms_chunk / rt.chunk_ms, 4), p90_ms=round(med["p90"], 3),
               cpu_ms_chunk=round(med["cpu_chunk"], 3), rtf_cpu=round(med["cpu_chunk"] / rt.chunk_ms, 4),
               user_ms_chunk=round(med["user_chunk"], 3), sys_ms_chunk=round(med["sys_chunk"], 3),
               rtf_user=round(med["user_chunk"] / rt.chunk_ms, 4), majflt=med["majflt"], minflt=med["minflt"],
               load_avg_1m=round(os.getloadavg()[0], 1),
               all_reps_ms=[round(r["ms_chunk"], 3) for r in reps], spk_embed_ms=round(med["spk_ms"], 3),
               tok_per_s=round(med["tok_per_s"], 2), load_s=round(load_s, 3), prep_s=round(prep_s, 3),
               warmup_s=round(warm_s, 3), cold_first_step_ms=round(cold_ms, 2),
               # first result: chunk buffering + mel window look-ahead (half n_fft) + compute of that chunk
               first_result_ms=round(rt.chunk_ms + 1000 * rt.half / 16000 + warm_step, 1),
               peak_rss_mb=round(maxrss_mb, 1))
    return res


# ------------------------------------------------------------------------------------------ sweep
def plan():
    """(size, ctx, backend, threads). Full backend list for S/M at [70,1]; fewer elsewhere (time on a shared box)."""
    runs = []
    full = ["eager", "trace", "compile", "int8", "trace-int8", "bf16", "fp16", "onnx", "onnx-int8",
            "coreml-cpu", "coreml-all", "coreml-ane"]
    core = ["eager", "trace", "int8", "onnx", "onnx-int8"]
    for size in ["S", "M", "L", "S512x12", "M768x16"]:
        for att in ATTS:
            if size in ("S", "M"):
                bes1 = full if att == "70,1" else core
                bes4 = ["eager", "onnx", "onnx-int8"]  # torch at 4 threads is 5-10x slower on the loaded box
            elif size == "L":
                bes1 = core if att == "70,1" else ["eager", "int8", "onnx-int8"]
                bes4 = ["onnx-int8"] if att == "70,1" else []
            else:
                bes1, bes4 = (["eager", "int8", "onnx-int8"] if att == "70,1" else []), []
            runs += [(size, att, be, 1) for be in bes1] + [(size, att, be, 4) for be in bes4]
    return runs


def sweep(a):
    """Round-robin: every config once per round, ``--rounds`` rounds, 1 x ``--seconds`` per run, so
    contention from other jobs on the shared machine is spread across configs; report takes medians."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_f = out / "results.jsonl"
    done = set()
    if res_f.exists():
        for line in res_f.read_text().splitlines():
            r = json.loads(line)
            done.add((r["size"], r["att"], r["backend"], r["threads"], r.get("round", 0)))
    for rnd in range(a.rounds):
        for size, att, be, th in plan():
            if a.only_size and size not in a.only_size.split(","):
                continue
            if a.only_backend and be not in a.only_backend.split(","):
                continue
            if (size, att, be, th, rnd) in done:
                continue
            if rnd >= a.rounds_large and (size in ("L", "M768x16", "S512x12") or th > 1):
                continue  # large / extra shapes: fewer rounds (each run is minutes on a shared box)
            if rnd > 0 and any(k[:4] == (size, att, be, th) and k[4] == -1 for k in done):
                continue  # failed in an earlier round: don't retry
            cmd = [sys.executable, __file__, "one", "--out", str(out), "--size", size, "--att", att,
                   "--backend", be, "--threads", str(th), "--seconds", str(a.seconds), "--repeats", "1"]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                ok, err = bool(line), (p.stderr.strip().splitlines() or ["?"])
            except subprocess.TimeoutExpired:
                ok, err = False, [f"timeout {a.timeout}s"]
            if ok:
                r = json.loads(line[-1])
                r["round"] = rnd
            else:
                r = dict(size=size, att=att, backend=be, threads=th, round=-1, error=" | ".join(err[-3:])[:800])
                done.add((size, att, be, th, -1))
            with res_f.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[r{rnd} {time.time() - t0:6.1f}s] {size:8s} {att} {be:10s} t{th}: "
                  f"{r.get('ms_chunk', 'ERR')} ms/chunk rtf={r.get('rtf', r.get('error', '')[-200:])}", flush=True)


def aggregate(out: Path) -> list[dict]:
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["size"], r["att"], r["backend"], r["threads"]), []).append(r)
    agg = []
    for (size, att, be, th), rs in groups.items():
        good = [r for r in rs if "error" not in r]
        if not good:
            agg.append(dict(size=size, att=att, backend=be, threads=th, error=rs[-1]["error"]))
            continue
        med = lambda k: statistics.median(r[k] for r in good)  # noqa: E731
        g0 = good[0]
        agg.append(dict(size=size, att=att, backend=be, threads=th, n=len(good), params_m=g0.get("params_m"),
                        chunk_ms=g0["chunk_ms"], ms_chunk=med("ms_chunk"),
                        ms_chunk_min=min(r["ms_chunk"] for r in good), ms_per_80ms=med("ms_per_80ms"),
                        rtf=med("rtf"), rtf_min=min(r["rtf"] for r in good), p90_ms=med("p90_ms"),
                        cpu_ms=med("cpu_ms_chunk"), cpu_ms_min=min(r["cpu_ms_chunk"] for r in good),
                        rtf_cpu=med("rtf_cpu"), load_avg=med("load_avg_1m"), user_ms=med("user_ms_chunk"),
                        user_ms_min=min(r["user_ms_chunk"] for r in good), rtf_user=med("rtf_user"),
                        sys_ms=med("sys_ms_chunk"), majflt=med("majflt"),
                        load_s=med("load_s"), prep_s=med("prep_s"), cold_ms=med("cold_first_step_ms"),
                        first_ms=med("first_result_ms"), rss=med("peak_rss_mb"), tok_per_s=med("tok_per_s"),
                        spk_ms=med("spk_embed_ms")))
    order = {k: i for i, k in enumerate(SIZES)}
    bo = {b: i for i, b in enumerate(["eager", "trace", "compile", "int8", "trace-int8", "bf16", "fp16", "onnx",
                                      "onnx-int8", "coreml-cpu", "coreml-all", "coreml-ane"])}
    agg.sort(key=lambda r: (order[r["size"]], r["att"], r["threads"], bo.get(r["backend"], 99)))
    return agg


def report(a):
    agg = aggregate(Path(a.out))
    print("| size | ctx (chunk) | backend | thr | wall ms/chunk med (min) | CPU ms/chunk med (min) | ms per 80 ms (CPU) |"
          " RTF wall | RTF CPU | p90 wall ms | load+prep s | cold 1st step ms | 1st result ms | peak RSS MB | n | load avg |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in agg:
        if "error" in r:
            print(f"| {r['size']} | [{r['att']}] | {r['backend']} | {r['threads']} | FAILED: {r['error'][-160:]} |"
                  + " |" * 11)
            continue
        cs = r["chunk_ms"] / 80
        print(f"| {r['size']} | [{r['att']}] ({r['chunk_ms']:.0f} ms) | {r['backend']} | {r['threads']} | "
              f"{r['ms_chunk']:.1f} ({r['ms_chunk_min']:.1f}) | {r['cpu_ms']:.1f} ({r['cpu_ms_min']:.1f}) | "
              f"{r['cpu_ms'] / cs:.1f} | {r['rtf']:.3f} | **{r['rtf_cpu']:.3f}** | {r['p90_ms']:.1f} | "
              f"{r['load_s'] + r['prep_s']:.2f} | {r['cold_ms']:.0f} | {r['first_ms']:.0f} | {r['rss']:.0f} | {r['n']} | "
              f"{r['load_avg']:.0f} |")
    if a.json:
        Path(a.json).write_text(json.dumps(agg, indent=1))


# ------------------------------------------------------------------------------------------ profile
def profile(a):
    """Where the time goes: coarse stages (any backend), per-module encoder split and torch.profiler ops (torch
    backends). Module timers wrap submodule calls (~1 us each), so the split is approximate."""
    from torch.profiler import ProfilerActivity, profile as tprof
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    att = ATTS[a.att]
    m, _ = get_model(out, a.size, att)
    kw, be, quant, dtype = {}, a.backend, None, torch.float32
    if be.startswith("onnx"):  # the same mapping as bench()
        kw["onnx_path"] = ensure_onnx(out, m, a.size, a.att, be.endswith("int8"))
        be = "onnx"
    elif be == "int8":
        be, quant = "eager", "int8"
    elif be == "trace-int8":
        be, quant = "trace", "int8"
    elif be in ("bf16", "fp16"):
        be, dtype = "eager", {"bf16": torch.bfloat16, "fp16": torch.float16}[be]
    elif be.startswith("coreml"):
        kw["coreml_path"] = ensure_coreml(out, m, a.size, a.att, fp16=be == "coreml-ane")
        kw["coreml_units"] = {"coreml-cpu": "CPU_ONLY", "coreml-all": "ALL", "coreml-ane": "CPU_AND_NE"}[be]
        be = "coreml"
    rt = StreamingRuntime(m, att_context_size=att, backend=be, quantize=quant, dtype=dtype, threads=a.threads,
                          copy_model=False, **kw).warmup(5)
    calibrate_runtime(rt)
    audio = bench_audio(a.seconds)
    push = int(round(rt.chunk_ms / 1000 * 16000))
    acc: dict = {}

    def timed(name, f):
        def g(*x, **k):
            t = time.perf_counter()
            r = f(*x, **k)
            acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
            return r
        return g

    def stream():
        rt.reset()
        t = time.perf_counter()
        for i in range(0, len(audio), push):
            rt.feed(audio[i:i + push])
        return time.perf_counter() - t

    # pass 1: coarse stages
    orig = (rt._mel, rt._fn, rt._decode_primary)
    rt._mel, rt._fn = timed("mel (STFT + filterbank)", rt._mel), timed("encoder step + head projections", rt._fn)
    rt._decode_primary = timed("TDT greedy (pred LSTM + joint)", rt._decode_primary)
    best = None
    for _ in range(3):  # keep the least-disturbed pass (shared machine)
        acc.clear()
        t = stream()
        if best is None or t < best[0]:
            best = (t, dict(acc))
    total, acc = best[0], best[1]
    n = rt.steps
    print(f"# {a.size} [{a.att}] {a.backend} threads={a.threads}: {n} chunks/pass, {1000 * total / n:.2f} ms/chunk")
    print("| stage | ms/chunk | share |\n|---|---|---|")
    for k, v in acc.items():
        print(f"| {k} | {1000 * v / n:.2f} | {100 * v / total:.0f}% |")
    rest = total - sum(acc.values())
    print(f"| other (frame heads, CTC greedy, buffers, Python) | {1000 * rest / n:.2f} | {100 * rest / total:.0f}% |")
    rt._mel, rt._fn, rt._decode_primary = orig
    if be in ("onnx", "coreml"):
        return  # no torch modules to split or profile
    if be != "eager":
        # trace/compile: the step function is not the eager module; the per-module split (which patches the eager
        # submodules) would silently profile the eager encoder instead, so go straight to the op-level table
        with tprof(activities=[ProfilerActivity.CPU]) as prof:
            stream()
        print("\n```")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
        print("```")
        return
    # pass 2: per-module split of the encoder step (eager only)
    acc = {}
    ce = rt.ce
    groups = {"pre-encode (subsampling convs + linear)": list(ce.pre_convs) + [ce.pre_out]}
    for l in ce.layers:
        groups.setdefault("feed-forward x2 (Linear 4x)", []).extend([l.ff1, l.ff2])
        groups.setdefault("attention qkv + out proj (Linear)", []).extend([l.qkv, l.proj])
        groups.setdefault("conv module pw1/pw2 (Linear) + depthwise", []).extend([l.pw1, l.pw2, l.dw])
        groups.setdefault("layer norms", []).extend([l.norm_att, l.norm_conv, l.norm_out, l.cnorm])
    groups["head projections"] = list(ce.heads.values())
    patched = []
    for gname, mods in groups.items():
        for mod in mods:
            patched.append((mod, mod.forward))
            mod.forward = timed(gname, mod.forward)
    rt._fn = timed("__enc_total", ce)
    total = stream()
    enc_total = acc.pop("__enc_total")
    print(f"\n| encoder step component ({1000 * enc_total / n:.2f} ms/chunk total) | ms/chunk | share of encoder |")
    print("|---|---|---|")
    for k, v in acc.items():
        print(f"| {k} | {1000 * v / n:.2f} | {100 * v / enc_total:.0f}% |")
    rest = enc_total - sum(acc.values())
    print(f"| inline ops (RoPE, SDPA, cache cat/stack, GLU/SiLU, residuals) | {1000 * rest / n:.2f} | "
          f"{100 * rest / enc_total:.0f}% |")
    for mod, f in patched:
        mod.forward = f
    rt._fn = orig[1]
    # pass 3: op-level table
    with tprof(activities=[ProfilerActivity.CPU]) as prof:
        stream()
    print("\n```")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
    print("```")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "one", "report", "profile"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--size", default="M")
    ap.add_argument("--att", default="70,1")
    ap.add_argument("--backend", default="eager")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=60.0)  # profile: use ~20
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--only-size", default="")
    ap.add_argument("--only-backend", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--rounds-large", type=int, default=1)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.cmd == "one":
        print(json.dumps(run_one(a)), flush=True)
        sys.stderr.flush()
        os._exit(0)  # skip interpreter teardown (ORT + torch thread pools can abort on exit)
    elif a.cmd == "sweep":
        sweep(a)
    elif a.cmd == "report":
        report(a)
    else:
        profile(a)


if __name__ == "__main__":
    main()
