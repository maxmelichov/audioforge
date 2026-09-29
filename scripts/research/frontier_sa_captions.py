"""FRONTIER pilot 1: speaker-attributed live captions from pieces we already serve (research/FRONTIER.md, Part 2).

Cascade: the served streaming RNNT (runs/stage1_served.afm, fed 160 ms blocks) emits words with emission times; an
NVIDIA streaming diarizer at the served setting (card "ultra low latency", chunk 3 + rc 1 = 0.32 s) gives per-slot
activity; each word gets the slot with the most activity over its estimated spoken interval
[t_first - LAG - 0.3 s, t_last - LAG + 0.08 s] (LAG = 0.24 s, set a priori from the 160 ms chunk + 80 ms right context,
not tuned). Scored with cpWER (concatenated minimum-permutation WER: Hungarian assignment of hypothesis slots to
reference speakers) against the AMI manual word references (mix-headset audio, words that *start* in the span).

Arms: sortformer (Streaming Sortformer v2, the benchmark diarizer), nemotron3 (Nemotron-3-Diarization, max pooling,
4 columns, left 1: the recommended product diarizer), oracle (AMI word activity: isolates ASR + the attribution rule),
agnostic (time-ordered reference vs all hypothesis words: the attribution-free floor, i.e. plain WER).
Scoring: one Hungarian permutation per meeting span ("session", what a live caption must get right) and per 60 s
window ("window", permutation re-chosen each minute); 95 % CIs by bootstrap over 60 s windows.

Resumable (each call stays under --budget seconds; units are cached on the SSD scratch):
  scripts/dev/gate.sh .venv/bin/python scripts/research/frontier_sa_captions.py run      # repeat until "all units done"
  scripts/dev/gate.sh .venv/bin/python scripts/research/frontier_sa_captions.py score    -> runs/frontier_sa_captions.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FRONTIER_WORK", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/frontier")) / "sa_captions"
SR, FRAME, BLOCK = 16000, 0.08, 2560
LAG = 0.24
MEETINGS = ["IS1008b", "ES2011b", "TS3004b", "IB4002"]   # AMI dev (one per site / scenario family)
DIARS = {"sortformer": ("runs/nemo_sortformer_v2.afm", None, 188),
         "nemotron3": ("runs/nemo_nemotron3_diar.afm", "max", 1)}


def span(ds, m, start, dur):
    return start, min(start + dur, ds.duration(m))


# ----------------------------------------------------------------------------- units
def unit_asr(m, audio):
    from audioforge.model import StreamingSession
    from audioforge.train import load_model
    asr = load_model(str(ROOT / "runs/stage1_served.afm"), "cpu").eval()
    from audioforge import perf  # exact CPU fast paths (bit-identical outputs, research/PERFORMANCE.md section 3)
    perf.linear_t(asr.encoder); perf.pos_cache(asr.encoder)
    StreamingSession.cache_joint_pred = True
    rt = StreamingSession(asr, head="rnnt")
    sp = rt.m.tokenizer.sp
    words, seen, t0 = [], 0, time.time()
    for i in range(0, len(audio), BLOCK):  # StreamingSession.feed runs under no_grad
        final = i + BLOCK >= len(audio)
        rt.feed(audio[i:i + BLOCK], final=final)
        t = min(len(audio), i + BLOCK) / SR
        for tok in rt.tokens[seen:]:
            piece = sp.id_to_piece(int(tok))
            if piece.startswith("▁") or not words:
                words.append([piece.lstrip("▁"), t, t])
            else:
                words[-1][0] += piece
                words[-1][2] = t
        seen = len(rt.tokens)
    return {"words": [(w, a, b) for w, a, b in words if w], "compute_s": time.time() - t0}


def unit_diar(name, audio, tail):
    from audioforge.nemo_import import load_any
    from audioforge.streaming_diar import SORTFORMER_PRESETS, StreamingDiarizer
    path, pool, left = DIARS[name]
    dm = load_any(str(ROOT / path), "cpu").eval()
    head = next(k for k, v in dm.head_cfg.items() if v["type"] == "sortformer")
    if pool:
        for h in dm.heads.values():
            if hasattr(h, "pool"):
                h.pool = pool
    from audioforge import perf
    perf.linear_t(dm); perf.pos_cache(dm)
    mode = "causal" if dm.encoder.causal else "window"
    kw = {"enc_left_context": left} if mode == "window" else {}
    t0 = time.time()
    sd = StreamingDiarizer(dm, head, mode=mode, **kw, **SORTFORMER_PRESETS["low_latency_032"])
    x = np.concatenate([audio, tail])
    for i in range(0, len(x), BLOCK):
        sd.feed(x[i:i + BLOCK], final=i + BLOCK >= len(x))
    T = int(np.ceil(len(audio) / 1280))
    return {"probs": sd.all_probs.float().numpy()[:T, :4], "compute_s": time.time() - t0, "mode": mode}


def run(a):
    from audioforge.datasets.ami import AMI
    WORK.mkdir(parents=True, exist_ok=True)
    ms = a.meetings.split(",") if a.meetings else MEETINGS
    ds = AMI(ms, verbose=False)
    t_start = time.time()
    # the product arm (asr + nemotron3) on every meeting first; Sortformer v2 afterwards as the budget allows
    units = [(m, u) for m in ms for u in ("asr", "nemotron3")] + [(m, u) for m in ms for u in a.late.split(",") if u]
    for m, u in units:
        out = WORK / f"{m}_{a.start:.0f}_{a.dur:.0f}_{u}.npz"
        if out.exists():
            continue
        if time.time() - t_start > a.budget:
            print("budget reached; re-run to continue", flush=True)
            return
        s0, s1 = span(ds, m, a.start, a.dur)
        x = ds._audio[m]
        audio = np.asarray(x[int(s0 * SR): int(s1 * SR)], np.float32)
        tail = np.asarray(x[int(s1 * SR): int(s1 * SR) + 5 * 1280], np.float32)  # rc lookahead past the span end
        r = unit_asr(m, audio) if u == "asr" else unit_diar(u, audio, tail)
        tmp = out.with_suffix(".tmp.npz")
        if u == "asr":
            np.savez(tmp, words=np.array(json.dumps(r["words"])), compute_s=r["compute_s"], dur=s1 - s0)
        else:
            np.savez(tmp, probs=r["probs"], compute_s=r["compute_s"], dur=s1 - s0)
        os.replace(tmp, out)
        print(f"{m} {u}: {s1 - s0:.0f}s audio, compute {r['compute_s']:.0f}s (RTF {r['compute_s'] / (s1 - s0):.3f})",
              flush=True)
    print("all units done", flush=True)


# ----------------------------------------------------------------------------- scoring
def cp_errors(refs: list[list[str]], hyps: list[list[str]]):
    n = max(len(refs), len(hyps))
    rs, hs = refs + [[]] * (n - len(refs)), hyps + [[]] * (n - len(hyps))
    C = np.array([[Levenshtein.distance(r, h) for h in hs] for r in rs])
    i, j = linear_sum_assignment(C)
    return int(C[i, j].sum()), dict(zip(i.tolist(), j.tolist()))


def attribute(words, act: np.ndarray, lag: float = LAG):
    """-> list of (word, slot, t_mid_est) with t in seconds from span start."""
    T, out = act.shape[0], []
    for w, a, b in words:
        f0 = max(0, int((a - lag - 0.3) / FRAME))
        f1 = min(T, max(f0 + 1, int((b - lag + 0.08) / FRAME) + 1))
        seg = act[f0:min(f1, T)].sum(0) if f0 < T else act[-1:].sum(0)
        out.append((w, int(seg.argmax()), max(0.0, (a + b) / 2 - lag - 0.15)))
    return out


def boot(err, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    err, n = np.asarray(err, float), np.asarray(n, float)
    idx = rng.integers(0, len(err), (B, len(err)))
    r = err[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def score(a):
    from audioforge.datasets.ami import AMI, activity, frames
    from audioforge.teachers import normalize_text as N
    ds = AMI(MEETINGS, verbose=False)
    # each meeting's span is read from its cached units (the budget forced shorter spans on the last two meetings)
    spans = {}
    for m in MEETINGS:
        f = sorted(WORK.glob(f"{m}_*_asr.npz"))
        if f:
            _, st, du, _ = f[0].stem.split("_")
            spans[m] = (float(st), float(du))
    have = lambda m, d: (WORK / f"{m}_{spans[m][0]:.0f}_{spans[m][1]:.0f}_{d}.npz").exists()
    arms = [d for d in DIARS if any(have(m, d) for m in spans)] + ["oracle"]
    res = {"config": {"meetings": MEETINGS, "spans": spans, "lag": LAG, "window_s": a.win,
                      "diar_setting": "low_latency_032 (0.32 s)", "asr": "runs/stage1_served.afm rnnt, 160 ms blocks"},
           "meetings": {}}
    win_rows = []  # per 60 s window: errors per arm (window permutation) + agnostic
    sess = {k: [0, 0] for k in arms + ["agnostic"]}
    sess_win = {k: [] for k in arms}   # session-permutation errors per window
    for m in [m for m in MEETINGS if m in spans and have(m, "nemotron3")]:
        st, du = spans[m]
        s0, s1 = span(ds, m, st, du)
        z = np.load(WORK / f"{m}_{st:.0f}_{du:.0f}_asr.npz")
        words = [(N(w), b0, b1) for w, b0, b1 in json.loads(str(z["words"])) if N(w)]
        words = [(tok, b0, b1) for w, b0, b1 in words for tok in w.split()]
        T = int(np.ceil((s1 - s0) / FRAME))
        spks = sorted(ds.words[m])
        refw = {s: [(st - s0, N(t)) for st, en, t in ds.words[m][s] if s0 <= st < s1 and N(t)] for s in spks}
        refw = {s: [(t, tok) for t, w in v for tok in w.split()] for s, v in refw.items() if v}
        spks = list(refw)
        acts = {"oracle": np.stack([frames(activity(ds.words[m][s]), T, s0) for s in spks], 1)}
        comp = {"asr": float(z["compute_s"])}
        for d in [d for d in DIARS if have(m, d)]:
            zd = np.load(WORK / f"{m}_{st:.0f}_{du:.0f}_{d}.npz")
            p = zd["probs"]
            acts[d] = np.pad(p, ((0, max(0, T - len(p))), (0, 0)))[:T]
            comp[d] = float(zd["compute_s"])
        row = {"span": [s0, s1], "n_ref_spk": len(spks), "ref_words": sum(map(len, refw.values())),
               "hyp_words": len(words), "rtf": {k: v / (s1 - s0) for k, v in comp.items()}}
        allref = [tok for _, tok in sorted((t, tok) for v in refw.values() for t, tok in v)]
        e = Levenshtein.distance(allref, [w for w, _, _ in words])
        row["agnostic_wer"] = e / len(allref)
        sess["agnostic"][0] += e; sess["agnostic"][1] += len(allref)
        marms = [k for k in arms if k in acts]
        att = {k: attribute(words, acts[k]) for k in marms}
        nw = int(np.ceil((s1 - s0) / a.win))
        for k in marms:
            S = acts[k].shape[1]
            hyps = [[w for w, s, _ in att[k] if s == j] for j in range(S)]
            err, perm = cp_errors([[tok for _, tok in refw[s]] for s in spks], hyps)
            row[k] = {"cpwer": err / row["ref_words"], "errors": err, "perm": {spks[i]: j for i, j in perm.items()
                                                                              if i < len(spks)}}
            sess[k][0] += err; sess[k][1] += row["ref_words"]
            # the same session permutation, errors counted per window (for CIs)
            for wi in range(nw):
                lo, hi = wi * a.win, (wi + 1) * a.win
                rr = [[tok for t, tok in refw[s] if lo <= t < hi] for s in spks]
                hh = [[w for w, s, t in att[k] if s == j and lo <= t < hi] for j in range(S)]
                n = max(len(rr), len(hh))
                rr, hh = rr + [[]] * (n - len(rr)), hh + [[]] * (n - len(hh))
                ew = sum(Levenshtein.distance(rr[i], hh[perm.get(i, i) if i in perm else i]) for i in range(len(rr)))
                sess_win[k].append((m, wi, ew, sum(map(len, rr))))
        for wi in range(nw):
            lo, hi = wi * a.win, (wi + 1) * a.win
            rr = [[tok for t, tok in refw[s] if lo <= t < hi] for s in spks]
            nref = sum(map(len, rr))
            if nref == 0:
                continue
            wr = {"meeting": m, "w": wi, "ref_words": nref}
            for k in marms:
                S = acts[k].shape[1]
                hh = [[w for w, s, t in att[k] if s == j and lo <= t < hi] for j in range(S)]
                wr[k] = cp_errors(rr, hh)[0]
            ar = [tok for _, tok in sorted((t, tok) for v in refw.values() for t, tok in v if lo <= t < hi)]
            wr["agnostic"] = Levenshtein.distance(ar, [w for w, _, t in words if lo <= t < hi])
            win_rows.append(wr)
        res["meetings"][m] = row
        print(m, {k: round(row[k]["cpwer"], 4) for k in marms}, "agnostic", round(row["agnostic_wer"], 4), flush=True)
    n = [r["ref_words"] for r in win_rows]
    res["n_windows"] = len(win_rows)
    res["session"] = {}
    for k in arms:
        ew = [e for _, _, e, _ in sess_win[k]]
        nn = [c for _, _, _, c in sess_win[k]]
        res["session"][k] = {"cpwer": sess[k][0] / sess[k][1], "errors": sess[k][0], "ref_words": sess[k][1],
                             "ci95_window_boot": boot(ew, nn),
                             "window_sum_check": sum(ew) / max(sum(nn), 1)}
    res["session"]["agnostic"] = {"wer": sess["agnostic"][0] / sess["agnostic"][1], "ref_words": sess["agnostic"][1],
                                  "ci95_window_boot": boot([r["agnostic"] for r in win_rows], n)}
    res["window"] = {}
    for k in arms + ["agnostic"]:
        rows = [r for r in win_rows if k in r]
        nk = [r["ref_words"] for r in rows]
        res["window"][k] = {"cpwer": sum(r[k] for r in rows) / sum(nk), "ci95": boot([r[k] for r in rows], nk),
                            "n_windows": len(rows), "ref_words": sum(nk)}
    # paired differences (window permutation), bootstrap over windows
    rng = np.random.default_rng(1)
    pd = {}
    for x, y in (("sortformer", "nemotron3"), ("nemotron3", "oracle"), ("oracle", "agnostic"), ("nemotron3", "agnostic")):
        rows = [r for r in win_rows if x in r and y in r]
        if not rows:
            continue
        idx = rng.integers(0, len(rows), (2000, len(rows)))
        na = np.array([r["ref_words"] for r in rows], float)
        d = np.array([r[x] - r[y] for r in rows], float)
        bs = d[idx].sum(1) / na[idx].sum(1)
        pd[f"{x}-{y}"] = {"delta": float(d.sum() / na.sum()), "n_windows": len(rows),
                          "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["paired_window"] = pd
    res["windows"] = win_rows
    out = ROOT / "runs/frontier_sa_captions.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("session", "window", "paired_window", "n_windows")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--dur", type=float, default=360.0)
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--meetings", default="")
    ap.add_argument("--late", default="sortformer", help="units run after asr + nemotron3 on every meeting")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
