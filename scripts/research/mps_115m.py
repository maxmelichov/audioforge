"""The served 115M streaming engine on the Mac GPU (MPS) vs the CPU (research/MPS_115M.md) -> runs/mps_115m.json.

Stages (each one process, < 10 min, resumable; run through scripts/dev/gate.sh, one torch/MPS job at a time):
  parity                 audioforge.load(device=cpu|mps) (--mode single) on examples/audio/two_party_call_16s.wav with its
                         stored voice print, 20 ms blocks: turn_end / final / voiceprint events CPU vs MPS; plus the
                         max |diff| of the streamed encoder output (ASR core at [70,1], 160 ms chunks) and of the
                         RNNT decode (token ids) on the clip
  core --device D        the ASR core as it streams words: model.StreamingSession (mel + cache-aware encoder at
                         [70,1] + greedy RNNT), batch 1, 160 ms chunks, 2 threads; per-chunk compute p50/p95 and RTF on
                         the first 20 LibriSpeech-200 utterances (warm-up, then best of 3 by RTF), then the WER on all
                         200 (normalize_text, the runs/hybrid_asr.json protocol, streamed, not masked offline).
                         Also the full single-mode engine per-chunk compute (Session.chunk_ms) on the clip.
  streams --device D     K single-mode sessions (audioforge.load) fed interleaved 160 ms block by block on one thread
                         (scripts/research/streams_cpu.py): real time while the p95 of the summed per-block compute of
                         all K streams < 160 ms
  report                 -> runs/mps_115m.json
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
sys.path.insert(0, str(ROOT))
SCRATCH = Path(os.environ.get("AUDIOFORGE_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch")) / "mps_115m"
FA_WORK = SCRATCH.parent / "final_asr"
OUT = ROOT / "runs" / "mps_115m.json"
ASR = ROOT / "runs" / "stage1_served_v2.afm"
CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
PRINT = ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json"
SR, CHUNK = 16000, 2560  # 160 ms
ATT = [70, 1]
AMI_WINDOWS = [("IS1009b", 600), ("ES2004b", 600), ("TS3003b", 600), ("EN2002a", 600), ("IS1009b", 1200),
               ("ES2004b", 1200), ("TS3003b", 1200), ("EN2002a", 1200)]


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)


def mem(device) -> dict:
    import torch
    out = {"peak_rss_mb": peak_rss_mb()}
    if device == "mps":
        out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20, 1)
        out["mps_current_mb"] = round(torch.mps.current_allocated_memory() / 2 ** 20, 1)
    return out


def sync(device):
    import torch
    if device == "mps":
        torch.mps.synchronize()


def save(stage: str, key: str, val):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = SCRATCH / f"{stage}.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1))


def load_clip():
    import soundfile as sf
    pcm, sr = sf.read(str(CLIP), dtype="int16")
    assert sr == SR
    return pcm


# --------------------------------------------------------------------------- parity
def events(fe, pcm, voiceprint=True):
    s = fe.session()
    if voiceprint:
        s.enroll(json.loads(PRINT.read_text()))
    out = []
    for i in range(0, len(pcm), SR // 50):
        out += s.feed(pcm[i:i + SR // 50])
    out += s.end()
    return [(m["type"], m["t"], m.get("speaker"), m.get("text"), m.get("policy")) for m in out
            if m["type"] in ("turn_end", "final", "voiceprint")]


def stream_core(m, x, att=ATT):
    """model.StreamingSession over x in 160 ms chunks -> (encoder output (T, D) float64 numpy, token ids)."""
    import torch
    from audioforge.model import StreamingSession
    s = StreamingSession(m, "rnnt", att)
    encs = []
    orig = m.encoder.stream_step

    def tap(*a, **k):
        r = orig(*a, **k)
        encs.append(r[0][0].detach().cpu().double().numpy())
        return r
    m.encoder.stream_step = tap
    try:
        with torch.inference_mode():
            for i in range(0, len(x), CHUNK):
                s.feed(x[i:i + CHUNK], final=i + CHUNK >= len(x))
    finally:
        m.encoder.stream_step = orig
    return np.concatenate(encs, 0), list(s.tokens)


def stage_parity(a):
    import torch
    import audioforge
    from audioforge.train import load_model
    torch.set_num_threads(2)
    pcm = load_clip()
    x = pcm.astype(np.float32) / 32768
    res = {}
    ev = {}
    for dev in ("cpu", "mps"):
        fe = audioforge.load(threads=2, device=dev)
        assert next(fe.engine.asr.parameters()).device.type == dev
        ev[dev] = events(fe, pcm)
        del fe
    res["events_equal"] = ev["cpu"] == ev["mps"]
    res["events_cpu"], res["events_mps"] = ev["cpu"], ev["mps"]
    core = {}
    for dev in ("cpu", "mps"):
        m = load_model(str(ASR), dev).eval()
        core[dev] = stream_core(m, x)
        del m
    e_c, e_m = core["cpu"][0], core["mps"][0]
    res["encoder_frames"] = int(e_c.shape[0])
    res["encoder_max_abs_diff"] = float(np.abs(e_c - e_m).max())
    res["encoder_rel_diff"] = float(np.linalg.norm(e_c - e_m) / np.linalg.norm(e_c))
    res["encoder_max_abs"] = float(np.abs(e_c).max())
    res["tokens_equal"] = core["cpu"][1] == core["mps"][1]
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("events_")}, indent=1))
    for d in ("cpu", "mps"):
        print(d, *ev[d], sep="\n  ")
    save("parity", "clip", res)
    print("done", flush=True)


# --------------------------------------------------------------------------- core
def load_libri():
    z = np.load(FA_WORK / "libri.npz")
    refs = json.loads((FA_WORK / "libri_refs.json").read_text())
    return [z[f"a{i}"] for i in range(len(refs))], refs


def time_stream(m, x, device):
    """One streamed utterance -> (per-160-ms-chunk compute ms list, total s, text)."""
    import torch
    from audioforge.model import StreamingSession
    s = StreamingSession(m, "rnnt", ATT)
    ms = []
    t_all = 0.0
    with torch.inference_mode():
        for i in range(0, len(x), CHUNK):
            sync(device)
            t0 = time.perf_counter()
            s.feed(x[i:i + CHUNK], final=i + CHUNK >= len(x))
            sync(device)
            dt = time.perf_counter() - t0
            ms.append(dt * 1000)
            t_all += dt
    return ms, t_all, s.text


def stage_core(a):
    import torch
    torch.set_num_threads(2)
    from audioforge.train import load_model
    from audioforge import perf
    dev = a.device
    t_start = time.time()
    m = load_model(str(ASR), dev).eval()
    if dev == "cpu" and not a.no_perf:  # the served CPU fast paths (exact); Engine applies the same by default
        from audioforge.server.streams import fast_conv
        fast_conv(m)
        perf.apply(m, None, perf.DEFAULT)
    elif not a.no_perf:
        perf.apply(m, None, perf.DEFAULT)
    audios, refs = load_libri()
    p = SCRATCH / f"core_{dev}.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    if "timing" not in st:
        sub = audios[:20]
        time_stream(m, sub[0], dev)  # warm-up
        runs = []
        for _ in range(3):
            chunks, tot = [], 0.0
            for x in sub:
                c, t, _ = time_stream(m, x, dev)
                chunks += c
                tot += t
            runs.append({"rtf": tot / (sum(len(x) for x in sub) / SR), "chunks": chunks})
        best = min(runs, key=lambda r: r["rtf"])
        c = np.array(best["chunks"])
        st["timing"] = {"utterances": len(sub), "audio_s": round(sum(len(x) for x in sub) / SR, 1),
                        "chunks": len(c), "chunk_ms_p50": round(float(np.percentile(c, 50)), 2),
                        "chunk_ms_p95": round(float(np.percentile(c, 95)), 2),
                        "chunk_ms_mean": round(float(c.mean()), 2),
                        "rtf": round(best["rtf"], 4), "rtf_all": [round(r["rtf"], 4) for r in runs], **mem(dev)}
        print("timing", st["timing"], flush=True)
        p.write_text(json.dumps(st, indent=1))
    hyps = st.setdefault("hyps", [])
    while len(hyps) < len(refs) and time.time() - t_start < a.budget:
        _, _, text = time_stream(m, audios[len(hyps)], dev)
        hyps.append(text)
        if len(hyps) % 20 == 0:
            p.write_text(json.dumps(st, indent=1))
            print(len(hyps), flush=True)
    st["mem_after_wer"] = mem(dev)
    p.write_text(json.dumps(st, indent=1))
    print("done" if len(hyps) >= len(refs) else f"left {len(refs) - len(hyps)}", flush=True)


def stage_engine(a):
    """Full single-mode engine per-chunk compute on the clip (Session.chunk_ms), warm-up then best of 3."""
    import torch
    import audioforge
    torch.set_num_threads(2)
    dev = a.device
    pcm = load_clip()
    fe = audioforge.load(threads=2, device=dev)
    events(fe, pcm)
    runs = []
    for _ in range(3):
        s = fe.session()
        s.enroll(json.loads(PRINT.read_text()))
        t0 = time.perf_counter()
        for i in range(0, len(pcm), CHUNK):
            s.feed(pcm[i:i + CHUNK])
        s.end()
        wall = time.perf_counter() - t0
        runs.append({"rtf": wall / (len(pcm) / SR), "chunk_ms": list(s._s.chunk_ms)})
    best = min(runs, key=lambda r: r["rtf"])
    c = np.array(best["chunk_ms"])
    r = {"chunk_ms_p50": round(float(np.percentile(c, 50)), 2), "chunk_ms_p95": round(float(np.percentile(c, 95)), 2),
         "rtf": round(best["rtf"], 4), "rtf_all": [round(x["rtf"], 4) for x in runs], **mem(dev)}
    print(r)
    save("engine", dev, r)
    print("done", flush=True)


# --------------------------------------------------------------------------- streams
def stage_streams(a):
    import torch
    import audioforge
    from audioforge.data import load_wav
    torch.set_num_threads(2)
    dev = a.device
    fe = audioforge.load(threads=2, device=dev)
    clips = []
    for mtg, st in AMI_WINDOWS[: max(a.ks)]:
        x = load_wav(str(ROOT / "data" / "ami" / "audio" / f"{mtg}.Mix-Headset.wav"), SR).astype(np.float32)
        clips.append(x[st * SR:(st + int(a.seconds)) * SR])
    p = SCRATCH / f"streams_{dev}.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    events(fe, load_clip())  # warm-up
    for K in a.ks:
        if str(K) in out:
            continue
        ss = [fe.session() for _ in range(K)]
        agg = []
        n = min(len(c) for c in clips[:K]) // CHUNK
        for i in range(n):
            t0 = time.perf_counter()
            for s, c in zip(ss, clips[:K]):
                s.feed(c[i * CHUNK:(i + 1) * CHUNK])
            agg.append((time.perf_counter() - t0) * 1000)
        for s in ss:
            s.end()
        x = np.array(agg[5:])
        out[str(K)] = {"agg_block_ms_p50": round(float(np.median(x)), 1),
                       "agg_block_ms_p95": round(float(np.percentile(x, 95)), 1),
                       "per_stream_block_ms_p50": round(float(np.median(x)) / K, 1),
                       "real_time": bool(np.percentile(x, 95) < 160), **mem(dev)}
        print(K, out[str(K)], flush=True)
        p.write_text(json.dumps(out, indent=1))
        if not out[str(K)]["real_time"]:
            break
    print("done", flush=True)


# --------------------------------------------------------------------------- report
def wer_of(hyps, refs):
    import jiwer
    from audioforge.teachers import normalize_text
    R = [normalize_text(r) for r in refs[: len(hyps)]]
    H = [normalize_text(h) for h in hyps]
    errs = words = 0
    for r, h in zip(R, H):
        o = jiwer.process_words(r, h)
        errs += o.substitutions + o.deletions + o.insertions
        words += len(r.split())
    return {"wer": round(errs / words, 4), "errors": errs, "ref_words": words, "n": len(hyps)}


def report(a):
    _, refs = load_libri()
    res = {"protocol": __doc__, "model": str(ASR.relative_to(ROOT)), "att_context": ATT, "chunk_ms": 160,
           "machine": "Apple M5 laptop, torch " + __import__("torch").__version__, "fp32": True}
    for f in ("parity", "engine"):
        p = SCRATCH / f"{f}.json"
        if p.exists():
            res[f] = json.loads(p.read_text())
    res["core"] = {}
    for dev in ("cpu", "mps"):
        p = SCRATCH / f"core_{dev}.json"
        if p.exists():
            st = json.loads(p.read_text())
            res["core"][dev] = {"timing": st.get("timing"), "wer_libri200_normalize_text": wer_of(st["hyps"], refs),
                                "mem_after_wer": st.get("mem_after_wer")}
    c, m = (SCRATCH / "core_cpu.json"), (SCRATCH / "core_mps.json")
    if c.exists() and m.exists():
        hc, hm = json.loads(c.read_text())["hyps"], json.loads(m.read_text())["hyps"]
        res["core"]["hyps_equal_cpu_mps"] = sum(x == y for x, y in zip(hc, hm))
        res["core"]["hyps_compared"] = min(len(hc), len(hm))
    res["streams"] = {}
    for dev in ("cpu", "mps"):
        p = SCRATCH / f"streams_{dev}.json"
        if p.exists():
            k = json.loads(p.read_text())
            res["streams"][dev] = {"k": k, "streams_real_time": max([int(x) for x, v in k.items() if v["real_time"]],
                                                                    default=0)}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "protocol"}, indent=1)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parity", "core", "engine", "streams", "report"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    a.ks = [int(k) for k in str(a.ks).split(",")]
    {"parity": stage_parity, "core": stage_core, "engine": stage_engine, "streams": stage_streams,
     "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
