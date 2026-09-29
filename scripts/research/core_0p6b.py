"""research/IMPROVEMENTS.md section 6: nemotron-speech-streaming-en-0.6b as the GPU streaming core (the user's request:
22.5 % AMI WER for the streaming partials is too high, production is GPU).

  import   re-import data/nemo/nemotron-speech-streaming-en-0.6b.nemo -> <ssd>/runs/nemo_nemotron_speech_streaming_en_0.6b.afm
           (read back, tensor count / non-zero check)
  wer      scripts/research/hybrid_asr.py lookahead on ami,icsi,libri (the AMI-200 / ICSI-200 / LibriSpeech-200
           protocols) at [70,1] and [70,13] for the 0.6B (--tag _0p6b) and the served 115M at [70,13] (its [70,1]
           hypotheses exist); run repeatedly under --budget
  rtf      streaming cost: StreamingSession at 160 ms chunks over 60 s of AMI (CPU 2 threads; MPS if --device mps),
           and the batched offline encode+decode RTF
  report   WER with 1000-resample CIs and paired deltas vs the 115M at the same context -> runs/hybrid_asr.json["core_0p6b"]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

NEMO = ROOT / "data/nemo/nemotron-speech-streaming-en-0.6b.nemo"
import os  # noqa: E402
AFM = Path(os.environ.get("AUDIOFORGE_0P6B_AFM",  # the laptop keeps it on the SSD; a GPU box sets runs/...
                          "/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_speech_streaming_en_0.6b.afm"))
TAG = "_0p6b"


def stage_import(a):
    import torch
    from audioforge.nemo_import import import_nemo
    from audioforge.train import load_model, save_model
    torch.set_num_threads(2)
    t0 = time.time()
    m = import_nemo(NEMO)
    info = dict(m.import_info)
    print(f"[import] {m.num_params() / 1e6:.1f}M params in {time.time() - t0:.1f}s: {info}", flush=True)
    AFM.parent.mkdir(parents=True, exist_ok=True)
    save_model(m, AFM)
    back = load_model(str(AFM), "cpu").state_dict()
    src = m.state_dict()
    assert set(back) == set(src) and all(torch.equal(back[k], src[k]) for k in src), "read-back mismatch"
    big = max(src, key=lambda k: src[k].numel())
    assert float(back[big].abs().sum()) > 0, "read-back zero-filled"
    print(f"[import] {AFM} ({AFM.stat().st_size / 2 ** 20:.0f} MB), {len(src)} tensors read back identical", flush=True)


def stage_wer(a):
    py = sys.executable
    hs = str(ROOT / "scripts" / "research" / "hybrid_asr.py")
    jobs = [["--model", str(AFM), "--tag", TAG, "--contexts", "1,13"], ["--contexts", "13"]]
    for extra in jobs:
        subprocess.run([py, hs, "lookahead", "--sets", "ami,icsi,libri", "--budget", str(a.budget / 2), *extra],
                       check=True, cwd=str(ROOT))


def stage_rtf(a):
    import torch
    from audioforge.datasets.ami import AMI
    from audioforge.model import StreamingSession
    from audioforge.train import load_model
    torch.set_num_threads(2)
    ds = AMI(["IS1008b"], verbose=False)
    x = np.asarray(ds._clip("IS1008b", 600.0, 660.0), np.float32)
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    import gc
    import psutil
    proc = psutil.Process()
    for name, path in (("0p6b", AFM), ("115m", ROOT / "runs/stage1_served.afm")):
        if a.only and name != a.only:
            continue
        gc.collect()
        r0 = proc.memory_info().rss / 2 ** 20
        m = load_model(str(path), a.device).eval()
        rec = {"rss_model_mb": round(proc.memory_info().rss / 2 ** 20 - r0, 1)}
        s = StreamingSession(m, "rnnt", [70, 1])
        with torch.inference_mode():
            s.feed(x[:16000 * 5])  # warm-up
            s = StreamingSession(m, "rnnt", [70, 1])
            t0 = time.perf_counter()
            per = []
            for i in range(0, len(x), 2560):
                t1 = time.perf_counter()
                s.feed(x[i:i + 2560])
                per.append((time.perf_counter() - t1) * 1000)
            dt = time.perf_counter() - t0
        rec["stream_rtf_160ms"] = round(dt / 60.0, 3)
        rec["ms_per_160ms_chunk_p50"] = round(float(np.median(per)), 1)
        rec["ms_per_160ms_chunk_p95"] = round(float(np.percentile(per, 95)), 1)
        with torch.inference_mode():
            xs = [x[i * 16000 * 12:(i + 1) * 16000 * 12] for i in range(4)]
            m.transcribe(xs[:1], head="rnnt", att_context_size=[70, 1])
            t0 = time.perf_counter()
            m.transcribe(xs, head="rnnt", att_context_size=[70, 1])
        rec["offline_rtf_batch4"] = round((time.perf_counter() - t0) / 48.0, 4)
        rec["rss_process_mb"] = round(proc.memory_info().rss / 2 ** 20, 1)
        if a.device == "mps":
            rec["mps_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20, 1)
        res.setdefault("rtf", {}).setdefault(a.device, {})[name] = rec
        print(name, a.device, rec, flush=True)
        del m
    OUTJ.write_text(json.dumps(res, indent=1))


def stage_partial(a):
    """First-partial latency, same protocol for both cores: each AMI-200 / ICSI-200 segment (single speaker, starts at
    the first word - 0.1 s) streamed through StreamingSession at [70,1] in 160 ms pieces; the audio time at the end of
    the piece after which the first token exists (algorithmic latency; the per-chunk compute is in ``rtf``)."""
    import torch
    import hybrid_asr as H
    from audioforge.model import StreamingSession
    from audioforge.train import load_model
    torch.set_num_threads(2)
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    for name, path in (("0p6b", AFM), ("115m", ROOT / "runs/stage1_served.afm")):
        if a.only and name != a.only:
            continue
        m = load_model(str(path), a.device).eval()
        for set_name in ("ami", "icsi"):
            audios, _ = H.load_fa_set(set_name)
            first = []
            for x in audios:
                s = StreamingSession(m, "rnnt", [70, 1])
                t = None
                with torch.inference_mode():
                    for i in range(0, len(x), 2560):
                        s.feed(x[i:i + 2560])
                        if s.tokens:
                            t = min(i + 2560, len(x)) / 16000
                            break
                first.append(t if t is not None else float("nan"))
            res.setdefault("first_token_s", {}).setdefault(set_name, {})[name] = first
            f = np.array(first)
            print(name, set_name, "median first token", round(float(np.nanmedian(f)), 3), "no token", int(np.isnan(f).sum()),
                  flush=True)
        del m
    OUTJ.write_text(json.dumps(res, indent=1))


VADW = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_0p6b/vad")
VAD_SETS = {"train": 300, "ami_dev": 64, "icsi_dev": 64}


def _model_path(name):
    return AFM if name == "0p6b" else ROOT / "runs/stage1_served.afm"


def stage_vadfeats(a):
    """All-block frame features (N, L, D) float16 of the VAD sets (vad_layers.load_set: AMI train diar windows, seeded
    cap 300; the BASELINES 64 x 20 s AMI dev windows; ICSI dev 64 windows) for one core -> VADW/<core>/<set>.f16."""
    import torch
    import vad_layers as V
    from audioforge.data import Collate
    from audioforge.train import load_model
    torch.set_num_threads(2)
    m = load_model(str(_model_path(a.only)), a.device).eval()
    L, D = len(m.encoder.layers), m.encoder.d_model
    t0 = time.time()
    for set_name, n in VAD_SETS.items():
        d = VADW / a.only
        d.mkdir(parents=True, exist_ok=True)
        meta_p = d / f"{set_name}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        val = V.load_set(set_name, n)
        Ts = [min(int(m.encoder.pre_encode.out_lengths(m.preprocessor.num_frames(torch.tensor(len(v["audio"]))))),
                  len(v["vad"])) for v in val]
        off = np.r_[0, np.cumsum(Ts)]
        N = int(off[-1])
        if meta.get("done") == len(val):
            continue
        X = np.memmap(d / f"{set_name}.f16", np.float16, "r+" if meta else "w+", shape=(N, L, D))
        lab = np.zeros(N, np.float32)
        win = np.zeros(N, np.int32)
        for w, v in enumerate(val):
            lab[off[w]: off[w + 1]] = v["vad"][: Ts[w]]
            win[off[w]: off[w + 1]] = w
        np.save(d / f"{set_name}_labels.npy", lab)
        np.save(d / f"{set_name}_win.npy", win)
        done = int(meta.get("done", 0))
        col = Collate(None)
        for i in range(done, len(val), 4):
            b = col([{"audio": v["audio"]} for v in val[i: i + 4]])
            with torch.inference_mode():
                enc, elen, hidden = m.encode(b["audio"].to(a.device), b["audio_len"].to(a.device), [70, 1],
                                             return_hidden=True)
            for j in range(len(elen)):
                w = i + j
                X[off[w]: off[w + 1]] = torch.stack([h[j, : Ts[w]] for h in hidden], 1).to(torch.float16).cpu().numpy()
            done = min(i + 4, len(val))
            X.flush()
            meta_p.write_text(json.dumps({"done": done, "n": len(val), "N": N, "L": L, "D": D}))
            if time.time() - t0 > a.budget:
                print(f"{a.only} {set_name}: {done}/{len(val)} (budget)", flush=True)
                return
        Xr = np.memmap(d / f"{set_name}.f16", np.float16, "r", shape=(N, L, D))
        assert np.any(Xr[-50:].astype(np.float32)), "read-back zero-filled"
        print(f"{a.only} {set_name}: {N} frames x {L} blocks done ({time.time() - t0:.0f}s)", flush=True)


def _vad_load(core, set_name):
    meta = json.loads((VADW / core / f"{set_name}.json").read_text())
    assert meta["done"] == meta["n"], (core, set_name, meta)
    X = np.memmap(VADW / core / f"{set_name}.f16", np.float16, "r", shape=(meta["N"], meta["L"], meta["D"]))
    return X, np.load(VADW / core / f"{set_name}_labels.npy"), np.load(VADW / core / f"{set_name}_win.npy")


def stage_vad(a):
    """The served VAD head's recipe on each core, identically: a learned softmax mix over ALL blocks + Linear(D,64)-
    SiLU-Linear(64,1) (heads.audio FrameHead shape), input dropout 0.2, Adam 1e-3, 3000 steps of 2048 frames, BCE, on
    the 300 AMI train windows; scored at 0.5 (sd.score_vad) on AMI dev / ICSI dev; paired window bootstrap of the F1
    difference (1000 resamples)."""
    import torch
    import vad_layers as V
    torch.set_num_threads(2)
    res = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    preds = {}
    for core in ("0p6b", "115m"):
        Xtr, ytr, _ = _vad_load(core, "train")
        N, L, D = Xtr.shape
        torch.manual_seed(0)
        mix = torch.nn.Parameter(torch.zeros(L))
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(D, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam([mix, *net.parameters()], lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = np.random.default_rng(0)
        t0 = time.time()
        for step in range(a.steps):
            idx = np.sort(g.integers(0, N, 2048))
            x = torch.from_numpy(np.asarray(Xtr[idx], np.float32))
            y = torch.from_numpy(ytr[idx])
            h = (x * mix.softmax(0)[None, :, None]).sum(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(h).squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        rec = {"train_sec": round(time.time() - t0, 1), "mix": [round(float(v), 3) for v in mix.detach().softmax(0)]}
        for set_name in ("ami_dev", "icsi_dev"):
            X, y, win = _vad_load(core, set_name)
            with torch.no_grad():
                p = np.concatenate([net((torch.from_numpy(np.asarray(X[i:i + 8192], np.float32))
                                         * mix.softmax(0)[None, :, None]).sum(1)).squeeze(-1).sigmoid().numpy()
                                    for i in range(0, len(X), 8192)])
            preds[(core, set_name)] = (p, y, win)
            rec[set_name] = V.vad_report(V.per_window(p, win), V.per_window(y, win))
            print(core, set_name, {k: rec[set_name][k] for k in ("f1", "auc") if k in rec[set_name]}, flush=True)
        res.setdefault("vad", {})[core] = rec

    def f1(p, y):
        d, t = p > 0.5, y > 0.5
        tp = float((d & t).sum())
        return 2 * tp / max(float(d.sum() + t.sum()), 1.0)
    for set_name in ("ami_dev", "icsi_dev"):
        pa, y, win = preds[("0p6b", set_name)]
        pb, yb, _ = preds[("115m", set_name)]
        W = int(win.max()) + 1
        A = [(pa[win == w], y[win == w]) for w in range(W)]
        Bw = [(pb[win == w], yb[win == w]) for w in range(W)]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(1000):
            k = rng.integers(0, W, W)
            fa = f1(np.concatenate([A[i][0] for i in k]), np.concatenate([A[i][1] for i in k]))
            fb = f1(np.concatenate([Bw[i][0] for i in k]), np.concatenate([Bw[i][1] for i in k]))
            bs.append(fa - fb)
        d0 = f1(pa, y) - f1(pb, yb)
        res.setdefault("vad", {}).setdefault("paired_f1", {})[set_name] = {
            "delta": round(d0, 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            "n_windows": W}
        print(set_name, "F1 0.6B - 115M", res["vad"]["paired_f1"][set_name], flush=True)
    OUTJ.write_text(json.dumps(res, indent=1))


OUTJ = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/hybrid_asr/core_0p6b_rtf.json")


def stage_report(a):
    import hybrid_asr as H
    norms, _ = H._normalizers()
    out = {"model": "nvidia/nemotron-speech-streaming-en-0.6b (NVIDIA Open Model License)", "afm": str(AFM),
           "protocol": "hybrid_asr.py lookahead: masked offline forward = cache-aware streaming at att_context, "
                       "greedy RNNT, batch 4; WER normalize_text (whisper_norm alongside); 1000 resamples",
           "sets": {}}
    for set_name in ("ami", "icsi", "libri"):
        _, refs = H.load_fa_set(set_name)
        rec = {"n": len(refs)}
        for r in (1, 13):
            base = H._rows(H.WORK / f"served_la{r}_{set_name}.jsonl")
            new = H._rows(H.WORK / f"served{TAG}_la{r}_{set_name}.jsonl")
            if len(base) < len(refs) or len(new) < len(refs):
                rec[f"la{r}"] = {"incomplete": f"115M {len(base)}, 0.6B {len(new)} of {len(refs)}"}
                continue
            rr = {}
            for nn in ("normalize_text", "whisper_norm"):
                fn = norms[nn]
                eb = np.array([H.edits_words(fn(t), fn(base[i]["hyp"])) for i, t in enumerate(refs)], float)
                en = np.array([H.edits_words(fn(t), fn(new[i]["hyp"])) for i, t in enumerate(refs)], float)
                rr[nn] = {"115m": H.rate_ci(eb), "0p6b": H.rate_ci(en), "0p6b - 115m": H.paired(en, eb)}
            sec = sum(new[i]["sec"] for i in range(len(refs)))
            aud = sum(new[i]["audio_sec"] for i in range(len(refs)))
            sec_b = sum(base[i]["sec"] for i in range(len(refs)))
            rr["offline_rtf_cpu_batch4"] = {"0p6b": round(sec / aud, 4), "115m": round(sec_b / aud, 4)}
            rec[f"la{r}"] = rr
        out["sets"][set_name] = rec
    if OUTJ.exists():
        extra = json.loads(OUTJ.read_text())
        out["rtf"] = extra.get("rtf")
        ft = {}
        for set_name, d in extra.get("first_token_s", {}).items():
            if "0p6b" in d and "115m" in d:
                A, B = np.array(d["0p6b"], float), np.array(d["115m"], float)
                ok = ~np.isnan(A) & ~np.isnan(B)
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(1000):
                    k = rng.integers(0, ok.sum(), ok.sum())
                    bs.append(float(np.median(A[ok][k]) - np.median(B[ok][k])))
                ft[set_name] = {"median_s": {"0p6b": round(float(np.median(A[ok])), 3), "115m": round(float(np.median(B[ok])), 3)},
                                "mean_s": {"0p6b": round(float(A[ok].mean()), 3), "115m": round(float(B[ok].mean()), 3)},
                                "median_delta_s": round(float(np.median(A[ok]) - np.median(B[ok])), 3),
                                "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                "n": int(ok.sum()), "no_token": {"0p6b": int(np.isnan(A).sum()), "115m": int(np.isnan(B).sum())}}
        out["first_partial"] = ft
        print("first partial", ft)
    res = H._json()
    res["core_0p6b"] = out
    H.OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for s_, rec in out["sets"].items():
        for r in (1, 13):
            v = rec.get(f"la{r}", {})
            if "normalize_text" in v:
                t = v["normalize_text"]
                print(f"{s_} [70,{r}]: 115M {t['115m']} 0.6B {t['0p6b']} delta {t['0p6b - 115m']}")
            else:
                print(s_, r, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--only", default=None, help="rtf: one model (0p6b | 115m), for a clean per-process RSS")
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
