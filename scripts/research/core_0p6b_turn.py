"""research/CORE_0P6B_TURN.md: why end of turn is worse on the 0.6B core and the heads-only fix (frozen encoder).
-> runs/core_0p6b_turn.json.

Builds on scripts/research/core_0p6b_heads.py (imported as C; its caches under C.W are reused, nothing there is
recomputed) and turn_v5.py (the v5 classifier, the run_policy twin of the served VadHeadPolicy).

Stages (each one process, < 10 min, resumable; run through scripts/dev/gate.sh, one torch/MPS job at a time):
  evcache --which calls|asst   the end-of-turn evaluation audio (232 calls / AMI sessions, 399 assistant clips) through
                               the 0.6B's masked [70,1] forward (= the served stream): blocks EV_BLOCKS + greedy RNNT
                               tokens + the served VAD head -> TF/ev/<key>.npz (evaluation only: never trained on)
  evcheck                      the cache reproduces the served dumps (VAD and v5 p on the dumped frames)
  lag                          VAD posteriors of both cores around the labelled turn ends (same frames)
  swap                         cross-core attribution on the dumps (the 115M VAD / v5 p / turn p in the 0.6B session)
  ...                          (further stages: see STAGES)
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

import core_0p6b_heads as C  # noqa: E402
import turn_v5 as _V5  # noqa: E402

# turn_v5's own (115M) globals, captured before core_0p6b_heads._v5_mod repoints the module at the 0.6B caches
_V5_ORIG = {k: getattr(_V5, k) for k in ("W", "BLK", "feat_path", "all_clips", "npz", "lm_frames")}


class v5_115m:
    """with v5_115m(): turn_v5 reads the 115M caches again (restored afterwards)."""

    def __enter__(self):
        self.saved = {k: getattr(_V5, k) for k in _V5_ORIG}
        for k, v in _V5_ORIG.items():
            setattr(_V5, k, v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(_V5, k, v)

W = C.W
TF = W / "turnfix"
EV = TF / "ev"
OUT = ROOT / "runs" / "core_0p6b_turn.json"
EV_BLOCKS = (4, 8, 10, 12, 14, 16, 20, 24)
SR = 16000
FRAME = 0.08
T_START = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val, sub=None):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    if sub is None:
        d[key] = val
    else:
        d.setdefault(key, {})[sub] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def load_json(key, default=None):
    return json.loads(OUT.read_text()).get(key, default) if OUT.exists() else default


def dumps(which="calls"):
    """The served 0.6B dumps (core_0p6b_heads eot_dump): {key: dump}."""
    d = C.EOT_DUMP if which == "calls" else C.ASST_DUMP
    return {x["key"]: x for x in (json.loads(p.read_text()) for p in sorted(d.glob("*.json")))}


def dumps_115m():
    """The 115M's shipped dumps as core_0p6b_heads.stage_eot_score reads them: 'p' = the served per-frame turn head,
    'p5' = the v5 c5 classifier; the assistant dump with its Silero conf."""
    import turn_v5 as V5
    import eot_assistant as EA
    with v5_115m():
        dump, ad, _ = V5.load_sets("c5")
    served = {d_["key"]: d_ for d_ in (json.loads(p_.read_text()) for p_ in sorted((V5.W4 / "dump_served").glob("*.json")))}
    for k, d in dump.items():
        d["head"]["p5"] = d["head"]["p"]
        d["head"]["p"] = served[k]["head"]["p"]
    sd = EA.load_dump()
    for k, d in ad.items():
        d["head"]["p5"] = d["head"]["p"]
        d["head"]["p"] = sd[k]["head"]["p"]
    return dump, ad


def dumps_0p6b():
    import eot_assistant as EA
    sd = EA.load_dump()
    d6, a6 = dumps("calls"), dumps("asst")
    for k, d in a6.items():  # the reference speech end comes from Silero on the audio (core-independent)
        d["conf"] = sd[k]["conf"]
    return d6, a6


# --------------------------------------------------------------------------- evaluation cache
def ev_items(which):
    import eot_latency as E
    import eot_assistant as EA
    if which == "calls":
        return [(s["key"], (lambda s=s: E.read_audio(s))) for s in E.sessions()]
    return [(c["key"], (lambda c=c: EA.audio(c))) for c in EA.clips()]


def stage_evcache(a):
    import torch
    from audioforge.heads.turn import greedy_decode_frames, token_counts
    torch.set_num_threads(2)
    EV.mkdir(parents=True, exist_ok=True)
    items = ev_items(a.which)
    todo = [it for it in items if not (EV / f"{it[0]}.npz").exists()]
    log(f"evcache {a.which}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    t0 = time.time()
    m = C.load_core(a.device, heads=("vad",))
    rnnt = __import__("copy").deepcopy(m.heads["rnnt"]).cpu().eval()
    w = m.layer_mix["vad"].detach().softmax(0).cpu().numpy()
    head = m.heads["vad"]
    need = sorted(set(EV_BLOCKS) | set(C.VAD_BLOCKS))
    done = 0
    for k, fn in todo:
        if time.time() - t0 > a.budget:
            break
        x = fn()
        outs, L, top = C.encode_blocks(m, [x], need, top=True)
        o, n = outs[0], L[0]
        hv = sum(w[j] * o[b].astype(np.float32) for j, b in enumerate(C.VAD_BLOCKS))
        with torch.no_grad():
            vad = torch.sigmoid(head(torch.from_numpy(hv).to(a.device)[None]))[0].float().cpu().numpy()
        hyp, fr = greedy_decode_frames(rnnt, top[0, :n].float().cpu())
        nn_ = token_counts(torch.tensor([fr], dtype=torch.long), torch.tensor([len(hyp)]), n)[0].numpy() \
            if hyp else np.zeros(n, np.int64)
        np.savez(EV / f"{k}.tmp.npz", vad=vad.astype(np.float16), y=np.asarray(hyp, np.int32),
                 n=nn_.astype(np.int32), **{f"b{b}": o[b] for b in EV_BLOCKS})
        (EV / f"{k}.tmp.npz").rename(EV / f"{k}.npz")
        done += 1
    log(f"evcache {a.which}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")


def ev_load(key, blocks=()):
    z = np.load(EV / f"{key}.npz")
    d = {k: z[k] for k in ("vad", "y", "n")}
    for b in blocks:
        d[f"b{b}"] = z[f"b{b}"]
    return d


def stage_evcheck(a):
    """The cache against the served dumps: the served VAD head on the cached blocks vs the dumped VAD, and the
    shipped v5 classifier (turn/s12) on the cached block 12 + (cached VAD, dumped P(user) / P(other), cached tokens)
    vs the dumped v5 p, on the dumped frames."""
    import torch
    import turn_v5 as V5
    torch.set_num_threads(2)
    model, ck = V5.load_seg(C.TURN / "s12" / "model.pt", a.device)
    res = {}
    for which in ("calls", "asst"):
        dd = dumps(which)
        keys = sorted(dd)[:: max(1, len(dd) // a.n)]
        dv, dp, dt = [], [], []
        for k in keys:
            if not (EV / f"{k}.npz").exists():
                continue
            e = ev_load(k, (12,))
            h = dd[k]["head"]
            v = np.asarray(h["v"])
            T = len(e["vad"])
            vv = e["vad"].astype(np.float32)
            dv.append(np.abs(vv[np.clip(v, 0, T - 1)] - np.asarray(h["vad"])).max())
            pu = np.zeros(T, np.float32)
            po = np.zeros(T, np.float32)
            pu[np.clip(v, 0, T - 1)] = h["pu"]
            po[np.clip(v, 0, T - 1)] = h["po"]
            ex = np.stack([vv, pu, po], 1)
            p = V5.seg_probs(model, e["b12"].astype(np.float32), ex, e["n"], e["y"], a.device)
            dp.append(np.abs(p[np.clip(v, 0, T - 1)] - np.asarray(h["p5"])).max())
            dt.append(abs(T - (int(v.max()) + 1)))
        res[which] = {"n": len(dv), "vad_max_abs": round(float(np.max(dv)), 5), "vad_p99_of_max": round(float(np.percentile(dv, 99)), 5),
                      "p5_max_abs": round(float(np.max(dp)), 5), "p5_median_of_max": round(float(np.median(dp)), 5),
                      "len_diff_max": int(np.max(dt))}
        log(which, res[which])
    save("evcheck", res)


# --------------------------------------------------------------------------- diagnosis 1: the VAD lag
def _lag_stats(vads: dict, sess, thr=0.4):
    """per scored end: frames from the end until the VAD first reads < thr; onset lag (frames from the labelled start
    until VAD > 0.5); mean VAD in 80 ms steps after the end."""
    off, on, after = [], [], []
    for s in sess:
        vad = vads[s["key"]]
        T = len(vad)
        for i, (s0, e1) in enumerate(s["user_turns"]):
            if not s["scored"][i]:
                continue
            fe = int(np.ceil(e1 / FRAME))
            j = fe
            while j < T and vad[j] >= thr:
                j += 1
            off.append((j - fe) * 80)
            fs = int(s0 / FRAME)
            k = max(0, fs - 10)
            while k < T and vad[k] <= 0.5:
                k += 1
            on.append((k - fs) * 80)
            seg = np.asarray(vad[fe: fe + 25], float)
            after.append(np.pad(seg, (0, 25 - len(seg)), constant_values=np.nan))
    off, on = np.asarray(off), np.asarray(on)
    A = np.nanmean(np.asarray(after), 0)
    return {"n": int(len(off)), "offset_lag_ms_p50": float(np.percentile(off, 50)),
            "offset_lag_ms_p75": float(np.percentile(off, 75)), "offset_lag_ms_p90": float(np.percentile(off, 90)),
            "offset_lag_ge_1s_pct": round(100 * float(np.mean(off >= 1000)), 1),
            "onset_lag_ms_p50": float(np.median(on)),
            "mean_vad_after_end_160ms_steps": [round(float(x), 3) for x in A[::2]]}


def stage_lag(a):
    """Diagnosis 1 on the served dumps (both cores' sessions share the frame clock: frame v ends at 80 (v + 1) ms and
    is ready at the same t): offset / onset lags of the served VAD around the labelled ends of the calls (109) and AMI
    (200), the frame clocks compared, and the misses of each preset classified by cause."""
    import eot_latency as E
    import turn_v5 as V5
    dump, _ = dumps_115m()
    d6, _ = dumps_0p6b()
    sess = E.sessions()
    res = {"clock_equal": all(dump[k]["head"]["t"] == d6[k]["head"]["t"] for k in d6)}
    for core, D in (("115m", dump), ("0p6b", d6)):
        vads = {k: np.asarray(d["head"]["vad"], float) for k, d in D.items()}
        for sc, sets in E.SCOPES.items():
            res[f"{core}_{sc}"] = _lag_stats(vads, [s for s in sess if s["set"] in sets])
            log(core, sc, res[f"{core}_{sc}"])
    # the misses of balanced / fast on calls, classified: no onset in the turn (VAD never > 0.5 => the rule never
    # armed), VAD flicker (no quiet run long enough for the fallback before the next onset), or the decider (quiet
    # runs exist but p stays under its threshold)
    calls = [s for s in sess if s["set"] != "ami"]
    for preset, r in (("balanced", V5.BAL), ("fast", V5.F5)):
        for core, D in (("115m", dump), ("0p6b", d6)):
            pk = "p" if r["mode"] == "head" else "p5"
            rows = []
            for s in calls:
                h = dict(D[s["key"]]["head"])
                h["p"] = h[pk]
                db = np.load(V5.ENERGY / f"{s['key']}.npy")
                v = np.asarray(h["v"])
                out = V5.run_policy({"head": h}, db[np.clip(v, 0, len(db) - 1)], r, {})
                T_ = np.array([t for t, _ in out])
                vad = np.asarray(h["vad"])
                for i, (s0, e1) in enumerate(s["user_turns"]):
                    if not s["scored"][i]:
                        continue
                    nx = s["next_onset"][i]
                    hz = min(e1 + E.HORIZON, nx) if nx else e1 + E.HORIZON
                    if ((T_ >= e1 - E.TOL) & (T_ < hz)).any():
                        continue
                    fs, fe, fn = int(s0 / FRAME), int(np.ceil(e1 / FRAME)), int(hz / FRAME)
                    run, best = 0, 0
                    for x in vad[fe:fn]:
                        run = run + 1 if x < r["vad_thr"] else 0
                        best = max(best, run)
                    spoke = bool((vad[fs:fe] > 0.5).any())
                    cause = "no_onset" if not spoke else ("flicker" if best < r["fb"] else "disarmed_or_decider")
                    rows.append({"key": s["key"], "end": round(e1, 2), "gap_s": round(hz - e1, 2),
                                 "max_quiet_run_ms": best * 80, "cause": cause})
            res[f"miss_{preset}_{core}"] = rows
            log(preset, core, len(rows), {c: sum(r_["cause"] == c for r_ in rows) for c in
                                          ("no_onset", "flicker", "disarmed_or_decider")})
    save("lag", res)


def stage_swap(a):
    """Cross-core attribution on the evaluation dumps (diagnosis only; the sessions share the frame clock): the 0.6B
    session with the 115M's VAD, v5 p, per-frame turn p, TS-VAD track, or pairs of them put in place of its own,
    scored with the shared presets (core_0p6b_heads._score_sets)."""
    dump, ad = dumps_115m()
    d6, a6 = dumps_0p6b()

    def mix(base, other, keys):
        out = {}
        for k, d in base.items():
            h = dict(d["head"])
            o = other[k]["head"]
            assert h["v"] == o["v"], k
            for kk in keys:
                h[kk] = o[kk]
            out[k] = {**d, "head": h}
        return out
    res = {}
    for name, keys in (("115m", None), ("0p6b", ()), ("vad", ("vad",)), ("p5", ("p5",)), ("p", ("p",)),
                       ("pu_po", ("pu", "po")), ("vad_p5", ("vad", "p5")), ("vad_p", ("vad", "p"))):
        if keys is None:
            r = C._score_sets(dump, ad, "p5")
        else:
            r = C._score_sets(mix(d6, dump, keys), mix(a6, ad, keys), "p5")
        res[name] = {pn: {"calls": [x["calls"][q] for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")],
                          "ami": [x["ami"][q] for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")],
                          "asst": [x["asst"][q] for q in ("accuracy_pct", "p50", "false_fire_pct")]} for pn, x in r.items()}
        log(name, res[name])
    save("swap", res)


# --------------------------------------------------------------------------- VAD candidates (diagnosis 1 a-c)
HEADS = TF / "heads"


def _turn_manifest():
    import turn_v4 as V4
    return json.loads((V4.WORK / "manifest.json").read_text())


def _val_groups():
    V5 = C._v5_mod()
    return V5.val_groups(V5.all_clips(with_cuts=False))


def _act_labels(c, T):
    """frame-centre (+-20 ms) labels of a turn clip from its activity intervals (the user's words, bridged)."""
    y = np.zeros(T, np.float32)
    cen = (np.arange(T) + 0.5) * FRAME
    for a0, b0 in c["act"]:
        y[(cen + 0.02 > a0) & (cen - 0.02 < b0)] = 1
    return y


def _oto_vad_set(blocks, which, n_max=None, seed=0, quiet=False):
    """(X (N, nb, D) fp16, y, win) of oto user-channel turn clips (turn_v4 manifest; the 16 evaluation conversations
    are not in it): which = 'train' (not the turn_v4/v5 held-out conversations) or 'val' (only those); quiet = their
    quiet-channel variants (stage quiet)."""
    import random
    vg = _val_groups()
    pre = "q_" if quiet else ""
    cl = [c for c in _turn_manifest() if c["src"] == "oto" and blk_path(pre + c["id"]).exists()
          and (((c["src"], c["meeting"]) in vg) == (which == "val"))]
    if n_max and len(cl) > n_max:
        cl = random.Random(seed).sample(cl, n_max)
    X, Y, Wn = [], [], []
    for i, c in enumerate(cl):
        z = np.load(blk_path(pre + c["id"]))
        T = len(np.load(inp_path(pre + c["id"]))["pu"])
        x = np.stack([z[f"b{b}"][:T] for b in blocks], 1)
        T = len(x)
        X.append(x)
        Y.append(_act_labels(c, T))
        Wn.append(np.full(T, i, np.int32))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Wn), cl


def _vad_arch(arch, d_in, nb):
    """mlp = the served FrameHead shape (per frame, stateless); gru64 = the shipped 0.6B FrameGRUHead shape; gru16 =
    a 16-unit GRU (shorter memory); conv = causal 3-frame conv (2 past frames of state). nb > 1: softmax block mix."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = arch
            h = {"gru16": 16, "gru32": 32}.get(arch, 64)
            self.inp = nn.Linear(d_in, h)
            if arch == "conv":
                self.conv = nn.Conv1d(h, h, 3)
            elif arch.startswith("gru"):
                self.rnn = nn.GRU(h, h, batch_first=True)
            self.out = nn.Linear(h, 1)
            self.mix = nn.Parameter(torch.zeros(nb)) if nb > 1 else None

        def forward(self, x):  # (B, T, nb, D) -> logits (B, T)
            x = (x * self.mix.softmax(0)[None, None, :, None]).sum(2) if self.mix is not None else x[:, :, 0]
            h = F.silu(self.inp(x))
            if self.arch == "conv":
                h = F.silu(self.conv(F.pad(h.transpose(1, 2), (2, 0))).transpose(1, 2))
            elif self.arch.startswith("gru"):
                h, _ = self.rnn(h)
            return self.out(h).squeeze(-1)
    return Net()


def _offset_weight(yy, w, offw, n=12):
    """frames in the n frames (~1 s) after a speech end (label 1 -> 0) and still non-speech weighted offw."""
    import torch
    if offw == 1.0:
        return w
    fall = (yy[:, :-1] > 0.5) & (yy[:, 1:] <= 0.5)
    m = torch.zeros_like(yy, dtype=torch.bool)
    for k in range(n):
        m[:, 1 + k:] |= fall[:, : fall.shape[1] - k]
    m &= yy <= 0.5
    return torch.where(m, w * offw, w)


def vad_predict(net, X, win, dev, bs_frames=200000):
    """per-window whole-sequence forward (a GRU starts from zero state at each window, as a session does)."""
    import torch
    net.eval()
    out = np.zeros(len(X), np.float32)
    idx = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
    ends = np.r_[idx[1:], len(win)]
    with torch.no_grad():
        for i0, i1 in zip(idx, ends):
            x = torch.from_numpy(np.asarray(X[i0:i1], np.float32))[None].to(dev)
            out[i0:i1] = torch.sigmoid(net(x))[0].float().cpu().numpy()
    return out


def offset_lag(p, y, win, thr=0.4, min_gap=6):
    """frames from each labelled speech end (followed by >= min_gap non-speech frames) until p < thr, in ms:
    p50 / p90 / share >= 1 s; and the share of non-speech frames in those gaps with p >= thr (flicker)."""
    lags, fl, n_gap = [], 0, 0
    for w in np.unique(win):
        ii = np.nonzero(win == w)[0]
        pp, yy = p[ii], y[ii] > 0.5
        T = len(yy)
        f = np.nonzero(yy[:-1] & ~yy[1:])[0] + 1
        for e in f:
            g = e
            while g < T and not yy[g]:
                g += 1
            if g - e < min_gap:
                continue
            j = e
            while j < g and pp[j] >= thr:
                j += 1
            lags.append((j - e) * 80)
            fl += int((pp[j:g] >= thr).sum())
            n_gap += g - j
    lags = np.asarray(lags)
    return {"n_ends": int(len(lags)), "lag_ms_p50": float(np.percentile(lags, 50)) if len(lags) else None,
            "lag_ms_p90": float(np.percentile(lags, 90)) if len(lags) else None,
            "lag_ge_1s_pct": round(100 * float(np.mean(lags >= 1000)), 2) if len(lags) else None,
            "flicker_pct_of_gap_frames": round(100 * fl / max(n_gap, 1), 2)}


def stage_vadtrain(a):
    """A VAD candidate on cached 0.6B blocks (--vblocks, 1-based; > 1 = softmax mix): --arch mlp|gru64|gru16|conv;
    data: the shipped recipe's AMI train windows (+ its 2 SpecAugment views with --sa) minus the held-out meetings,
    --oto N oto user-channel clips (train conversations only) at --oto-share of the crops, room tone at 15 %; crops of
    --crop frames; feature dropout / noise / 2 time masks; --offw: weight of the non-speech frames in the 1 s after a
    speech end (the turn rule's trigger frames). Early stopping on the held-out mean AUC (AMI held-out meetings, ICSI
    train windows, oto held-out conversations; each with room-tone validation negatives). Scored on AMI dev / ICSI dev
    (64 x 20 s, the published protocol), held-out room tone, and the offset lag / flicker on the oto held-out clips.
    -> TF/heads/vad_<tag>.pt, runs/core_0p6b_turn.json vadtrain.<tag>."""
    import copy
    import torch
    import torch.nn.functional as F
    import vad_layers as V
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    dev = a.device
    HEADS.mkdir(parents=True, exist_ok=True)
    blocks = [int(b) for b in a.vblocks.split(",")]
    nb, L = len(blocks), a.crop
    t0 = time.time()
    srcs = []
    for k, name in enumerate(["train"] + (["train1200_sa1", "train1200_sa2"] if a.sa else [])):
        X, y, win, meets = C._vad_src(name, blocks)
        X = np.asarray(X)
        idx = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
        ends = np.r_[idx[1:], len(win)]
        keep = np.array([meets[win[i]] not in C.VAD_HOLD for i in idx])
        srcs.append(("ami", X, y, idx[keep], ends[keep]))
        if k == 0:
            m_ = np.array([meets[w] not in C.VAD_HOLD for w in win])
            Xv1, yv1, winv1 = X[~m_], y[~m_], win[~m_]
    if a.oto:
        Xo, yo, wo, _ = _oto_vad_set(blocks, "train", a.oto, a.seed)
        idx = np.nonzero(np.r_[True, wo[1:] != wo[:-1]])[0]
        srcs.append(("oto", Xo, yo, idx, np.r_[idx[1:], len(wo)]))
    if a.otoq:
        Xo, yo, wo, _ = _oto_vad_set(blocks, "train", a.otoq, a.seed + 1, quiet=True)
        idx = np.nonzero(np.r_[True, wo[1:] != wo[:-1]])[0]
        srcs.append(("otoq", Xo, yo, idx, np.r_[idx[1:], len(wo)]))
    Xv2, yv2, winv2, _ = C._vad_src("icsival150_all", blocks)
    Xv2 = np.asarray(Xv2)
    Xv3, yv3, winv3, _ = _oto_vad_set(blocks, "val")
    Xv4, yv4, winv4, _ = _oto_vad_set(blocks, "val", quiet=True)
    Xr, yr, winr, _ = C._vad_src("room_train_all", blocks)
    Xr = np.asarray(Xr)
    ridx = np.nonzero(np.r_[True, winr[1:] != winr[:-1]])[0]
    rend = np.r_[ridx[1:], len(winr)]
    nrv = len(ridx) // 5
    room = ("room", Xr, yr, ridx[:-nrv], rend[:-nrv])
    rv0 = ridx[-nrv]
    Xrv, yrv, winrv = Xr[rv0:], yr[rv0:], winr[rv0:]
    log(f"data {time.time() - t0:.0f} s: " + ", ".join(f"{s_[0]} {len(s_[1])}" for s_ in srcs) +
        f"; val ami {len(Xv1)} icsi {len(Xv2)} oto {len(Xv3)} room {len(Xrv)}")
    conv = (a.oto > 0) + (a.otoq > 0)
    share = {"ami": (1 - a.oto_share) / (1 + 2 * a.sa) if conv else 1 / (1 + 2 * a.sa),
             "oto": a.oto_share / max(conv, 1), "otoq": a.oto_share / max(conv, 1)}
    sw = np.array([share[s_[0]] for s_ in srcs])

    def crops(src, n, g):
        _, Xs, ys, idx, ends = src
        w = g.integers(0, len(idx), n)
        s0 = idx[w] + (g.random(n) * np.maximum(1, ends[w] - idx[w] - L)).astype(int)
        s0 = np.maximum(idx[w], np.minimum(s0, ends[w] - L))
        xs, yl = [], []
        for q, e in zip(s0, ends[w]):
            xx, yy = Xs[q:min(q + L, e)], ys[q:min(q + L, e)]
            if len(xx) < L:  # a short window: pad with its last frame, no loss there (weight from the label copy)
                xx = np.concatenate([xx, np.repeat(xx[-1:], L - len(xx), 0)])
                yy = np.concatenate([yy, np.full(L - len(yy), -1.0, np.float32)])
            xs.append(xx)
            yl.append(yy)
        return np.stack(xs), np.stack(yl)

    fstd = float(np.asarray(srcs[0][1][:20000], np.float32).std())
    net = _vad_arch(a.arch, C.D, nb).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = np.random.default_rng(1000 + a.seed)

    def val_score():
        pr = vad_predict(net, Xrv, winrv, dev)
        aucs = []
        for Xs, ys, ws in ((Xv1, yv1, winv1), (Xv2, yv2, winv2), (Xv3, yv3, winv3), (Xv4, yv4, winv4)):
            p_ = vad_predict(net, Xs, ws, dev)
            aucs.append(roc_auc_score(np.r_[ys, yrv] > 0.5, np.r_[p_, pr]))
        net.train()
        return float(np.mean(aucs)), aucs
    B = 64
    nr = int(round(B * 0.15))
    best, best_state, bad, hist = -1.0, None, 0, []
    for step in range(1, a.steps + 1):
        which = g.choice(len(srcs), B - nr, p=sw / sw.sum())
        xs, ys = [], []
        for k in np.unique(which):
            x_, y_ = crops(srcs[k], int((which == k).sum()), g)
            xs.append(x_)
            ys.append(y_)
        x_, y_ = crops(room, nr, g)
        xs.append(x_)
        ys.append(y_)
        x = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(dev)
        yy = torch.from_numpy(np.concatenate(ys).astype(np.float32)).to(dev)
        w = (yy >= 0).float()
        yy = yy.clamp(min=0)
        x = F.dropout(x, 0.1)
        x = x + 0.1 * fstd * torch.randn_like(x)
        for _ in range(2):
            ln = torch.randint(0, 5, (len(x),), device=dev)
            st = (torch.rand(len(x), device=dev) * (L - 4)).long()
            tt = torch.arange(L, device=dev)[None]
            msk = (tt >= st[:, None]) & (tt < (st + ln)[:, None])
            x = x.masked_fill(msk[..., None, None], 0.0)
            if a.arch == "mlp":
                w = w.masked_fill(msk, 0.0)
        w = _offset_weight(yy, w, a.offw)
        z = net(x)
        loss = (F.binary_cross_entropy_with_logits(z, yy, reduction="none") * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            v, aucs = val_score()
            hist.append({"step": step, "loss": round(float(loss), 4), "val_auc": round(v, 4),
                         "aucs": [round(x_, 4) for x_ in aucs]})
            log(hist[-1])
            if v > best:
                best, bad, best_state = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if bad >= a.patience or time.time() - t0 > a.budget:
                break
    net.load_state_dict(best_state)
    r = {"val_auc": round(best, 4), "hist": hist, "args": {k: getattr(a, k) for k in
                                                           ("arch", "vblocks", "oto", "otoq", "oto_share", "offw", "crop", "sa", "steps", "seed")},
         "params": sum(p_.numel() for p_ in net.parameters())}
    for sn in ("ami_dev", "icsi_dev"):
        Xs, ys, ws, _ = C._vad_src(sn, blocks)
        p_ = vad_predict(net, Xs, ws, dev)
        q = V.vad_report(V.per_window(p_, ws), V.per_window(ys, ws))
        r[sn] = {"f1": round(q["f1"], 4), "auc": round(q["auc"], 4), **{k: v for k, v in q.items() if k.startswith("miss_at")}}
        np.save(HEADS / f"pred_{a.tag}_{sn}.npy", p_)
    Xh, yh, wh, _ = C._vad_src("room_held_all", blocks)
    ph = vad_predict(net, np.asarray(Xh), wh, dev)
    r["room_held"] = {"p95": round(float(np.percentile(ph, 95)), 4), "share_gt_0.5": round(float((ph > 0.5).mean()), 4)}
    po = vad_predict(net, Xv3, winv3, dev)
    r["oto_val"] = {"auc": round(float(roc_auc_score(yv3 > 0.5, po)), 4), **offset_lag(po, yv3, winv3)}
    po = vad_predict(net, Xv4, winv4, dev)
    r["oto_val_quiet"] = {"auc": round(float(roc_auc_score(yv4 > 0.5, po)), 4), **offset_lag(po, yv4, winv4)}
    log(a.tag, json.dumps({k: r[k] for k in ("val_auc", "ami_dev", "icsi_dev", "room_held", "oto_val", "oto_val_quiet")}))
    torch.save({"arch": a.arch, "blocks": blocks, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "eval": {k: r[k] for k in ("val_auc", "ami_dev", "icsi_dev", "room_held", "oto_val", "oto_val_quiet")}},
               HEADS / f"vad_{a.tag}.pt")
    save("vadtrain", r, sub=a.tag)


def load_vad(tag, dev="cpu"):
    """('served' = the shipped 0.6B GRU head on the mix of blocks 4..24) -> (fn(blocks dict) -> p (T,), blocks)."""
    import torch
    if tag == "served":
        ck = torch.load(C.HEADS_DIR / "vad.pt", map_location="cpu", weights_only=False)
        from audioforge.heads.audio import FrameGRUHead
        h = FrameGRUHead(C.D, hidden=ck["state_dict"]["inp.weight"].shape[0])
        h.load_state_dict(ck["state_dict"])
        h = h.to(dev).eval()
        w = ck["mix"].softmax(0).numpy()
        bl = list(C.VAD_BLOCKS)

        def fn(d):
            x = sum(float(w[j]) * d[f"b{b}"].astype(np.float32) for j, b in enumerate(bl))
            with torch.no_grad():
                return torch.sigmoid(h(torch.from_numpy(x).to(dev)[None]))[0].float().cpu().numpy()
        return fn, bl
    ck = torch.load(HEADS / f"vad_{tag}.pt", map_location="cpu", weights_only=False)
    net = _vad_arch(ck["arch"], C.D, len(ck["blocks"]))
    net.load_state_dict(ck["state_dict"])
    net = net.to(dev).eval()
    bl = ck["blocks"]

    def fn(d):
        x = np.stack([d[f"b{b}"] for b in bl], 1).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(x).to(dev)[None]))[0].float().cpu().numpy()
    return fn, bl


# --------------------------------------------------------------------------- held-out end-of-turn replay (selection set)
HO = TF / "ho"
COMP = {"115m": 0.030, "0p6b": 0.041}  # chunk compute added to every decision (CORE_0P6B.md: 115M CPU, 0.6B MPS)


def stage_hoprep(a):
    """The held-out end-of-turn set: the turn_v4/v5 held-out split (same seed and groups as every v5 model's
    validation; never trained on by the shipped heads of either core, never an evaluation set): oto user-channel
    windows (the calls-like scope), AMI windows (the meeting scope), smart-turn clips + 3 s silence without a print
    (st3, the assistant-like scope). Per clip: frame energies (dBFS per 80 ms frame) and the reference turns
    (oto / AMI: the labelled user turns and the user's next onset, as eot_latency's sessions; st3: the energy-VAD
    utterance end = the smart-turn label end, horizon = clip + 3 s). -> TF/ho/meta.json, TF/ho/db/<id>.npy."""
    import turn_v4 as V4
    import turn_v5 as V5
    from audioforge.datasets import smartturn as ST
    (HO / "db").mkdir(parents=True, exist_ok=True)
    V5p = C._v5_mod()
    man_all = V5p.all_clips(with_cuts=False)
    vg = V5p.val_groups(man_all)
    man = _turn_manifest()
    au = V4.ClipAudio()
    meta_st = json.loads(ST.build_cache(verbose=False).read_text())
    wav = np.load(ST.DEFAULT_ROOT / "cache" / "human_5_all.npy", mmap_mode="r")
    o = meta_st["offsets"]
    out = {}
    t0 = time.time()
    for c in man_all:
        g = c["group"] if "group" in c else (c["src"], c.get("meeting", c["id"]))
        if c["src"] not in ("oto", "ami", "icsi", "st3") or g not in vg or not (C.TURN_INP / f"{c['id']}.npz").exists():
            continue
        T = len(np.load(C.TURN_INP / f"{c['id']}.npz")["pu"])
        if c["src"] == "st3":
            u = np.asarray(wav[o[c["idx"]]: o[c["idx"] + 1]], np.float32)
            x = np.concatenate([u, np.zeros(int(3.0 * SR), np.float32)])
            clip_s = len(u) / SR
            e = min(ST.utterance_end_frame(u) * FRAME, clip_s)
            fdb = V5.frame_abs_db(u, max(1, len(u) // 1280))
            on = np.nonzero(fdb > fdb.max() - 30)[0]
            r = {"src": "st3", "complete": bool(c["complete"]), "clip_s": round(clip_s, 4),
                 "onset": round(float(on[0]) * FRAME, 3) if len(on) else 0.0,
                 "user_turns": [(0.0, round(float(e), 4))], "scored": [True], "next_onset": [round(clip_s + 3.0, 4)]}
        else:
            x = au(c, man)[0]
            turns = [(t[0], t[1]) for t in c["turns"]]
            r = {"src": c["src"], "user_turns": turns, "scored": [True] * len(turns),
                 "next_onset": [t[2] for t in c["turns"]]}
        r["T"] = T
        np.save(HO / "db" / f"{c['id']}.npy", V5.frame_abs_db(x, T))
        out[c["id"]] = r
    for c in _quiet_clips("val"):  # the quiet-channel variants of the held-out oto windows (stage quiet)
        qid = f"q_{c['id']}"
        if not inp_path(qid).exists():
            continue
        T = len(np.load(inp_path(qid))["pu"])
        x = quiet_audio(c, au, man)[0]
        np.save(HO / "db" / f"{qid}.npy", V5.frame_abs_db(x, T))
        turns = [(t[0], t[1]) for t in c["turns"]]
        out[qid] = {"src": "otoq", "user_turns": turns, "scored": [True] * len(turns),
                    "next_onset": [t[2] for t in c["turns"]], "T": T}
    (HO / "meta.json").write_text(json.dumps(out))
    log(f"hoprep: {len(out)} clips ({ {s_: sum(v['src'] == s_ for v in out.values()) for s_ in ('oto', 'otoq', 'ami', 'icsi', 'st3')} }), "
        f"{sum(len(v['user_turns']) for v in out.values() if v['src'] != 'st3')} conversational turns, {time.time() - t0:.0f} s")


def ho_meta():
    return json.loads((HO / "meta.json").read_text())


def stage_ho115(a):
    """The 115M on the held-out set with its shipped heads: the served VAD / TS-VAD / per-frame turn head (turn_v4
    cache, pass-2 'e') and the shipped v5 c5 classifier (turn_v5 cache, block 8) -> TF/ho/p_115m.npz."""
    import torch
    import turn_v4 as V4
    import turn_v5 as V5
    torch.set_num_threads(2)
    meta = ho_meta()
    V5W, V5BLK = _V5_ORIG["W"], _V5_ORIG["BLK"]
    model, ck = V5.load_seg(V5W / "c5" / "model.pt", a.device)
    head = V4.load_head(None, a.device)
    out = {}
    t0 = time.time()
    for cid, r in meta.items():
        if r["src"] == "otoq":
            z = np.load(TF / "quiet" / "i115" / f"{cid}.npz")
            d = {k: z[k] for k in z.files}
            T = len(d["pu"])
            x = d["b8"][:T]
        else:
            f = (V5W / "st3" / f"{cid}.npz") if r["src"] == "st3" else (V5.FEATS4 / f"{cid}.npz")
            d = _V5_ORIG["npz"](f)
            T = len(d["pu"])
            x = V5.load_block(d, V5BLK / f"{cid}.npz", "8")[:T]
        ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
        p5 = V5.seg_probs(model, x.astype(np.float32), ex, d["n"].astype(np.int32), d["y"].astype(np.int32), a.device)
        p = V4.head_probs(head, d, a.device)
        for k, v in (("vad", d["vad"]), ("pu", d["pu"]), ("po", d["po"]), ("p5", p5), ("p", p)):
            out[f"{cid}|{k}"] = np.asarray(v, np.float32)
    np.savez(HO / "p_115m.npz", **out)
    log(f"ho115: {len(meta)} clips in {time.time() - t0:.0f} s")


def _load_turn_head(tag, dev):
    """The 0.6B per-frame turn head: 'f1' = the shipped one (C.HEADS_DIR/turn_f1.pt, reads block 24), else
    TF/heads/turn_<tag>.pt -> (head bound to the 0.6B's RNNT prediction net, block it reads)."""
    import torch
    f = C.HEADS_DIR / "turn_f1.pt" if tag == "f1" else HEADS / f"turn_{tag}.pt"
    ck = torch.load(f, map_location="cpu", weights_only=False)
    m = C.load_core(dev, heads=("turn",), cfgs={"turn": C.TURN_CFG}, fresh=("turn",))
    m.heads["turn"].load_state_dict(ck["state_dict"])
    return m.heads["turn"].eval(), int(ck.get("block", 24))


def _load_seg(tag, dev):
    """'s12' = the shipped 0.6B v5 (C.TURN/s12), else TF/seg/<tag>/model.pt -> (model, block)."""
    import turn_v5 as V5
    f = C.TURN / tag / "model.pt" if tag == "s12" else TF / "seg" / tag / "model.pt"
    model, ck = V5.load_seg(f, dev)
    return model, str(ck["args"]["block"]) if "args" in ck else str(ck["block"])


def heads_on(d, vad_fn, seg, turn, dev, vad_blocks=None):
    """The 0.6B's per-frame turn signals of one clip from its cached blocks / inputs: d has b<k>, pu, po, y, n
    (and 'vad' when vad_fn is None) -> {vad, p5, p}."""
    import turn_v4 as V4
    import turn_v5 as V5
    vad = d["vad"].astype(np.float32) if vad_fn is None else vad_fn(d)
    T = len(d["pu"])
    vad = vad[:T]
    out = {"vad": vad}
    if seg is not None:
        model, block = seg
        x = np.concatenate([d[f"b{b}"][:T] for b in block.split("+")], 1).astype(np.float32)
        ex = np.stack([vad, d["pu"][:T], d["po"][:T]], 1).astype(np.float32)
        out["p5"] = V5.seg_probs(model, x, ex, d["n"][:T].astype(np.int32), d["y"].astype(np.int32), dev)
    if turn is not None:
        head, tb = turn
        f = {"e": d[f"b{tb}"][:T], "pu": d["pu"][:T], "po": d["po"][:T], "y": d["y"], "n": d["n"][:T]}
        out["p"] = V4.head_probs(head, f, dev)
    return out


def inp_path(cid):
    return (TF / "quiet" / "inp" if cid.startswith("q_") else C.TURN_INP) / f"{cid}.npz"


def blk_path(cid):
    return (TF / "quiet" / "blk" if cid.startswith("q_") else C.TURN_BLK) / f"{cid}.npz"


def combo_name(vad, seg, turn):
    return f"{vad}__{seg}__{turn}"


def stage_hop6(a):
    """A 0.6B candidate (--vad tag | served, --seg tag | s12, --turn tag | f1) on the held-out set ->
    TF/ho/p6_<combo>.npz (vad, p5, p, pu, po per clip)."""
    import torch
    torch.set_num_threads(2)
    meta = ho_meta()
    name = combo_name(a.vad, a.seg, a.turn)
    f = HO / f"p6_{name}.npz"
    if f.exists():
        log("exists", f)
        return
    vad_fn = None if a.vad == "served" else load_vad(a.vad, a.device)[0]
    seg = _load_seg(a.seg, a.device)
    turn = _load_turn_head(a.turn, a.device)
    out = {}
    t0 = time.time()
    for cid in meta:
        z = np.load(inp_path(cid))
        d = {k: z[k] for k in z.files}
        zb = np.load(blk_path(cid))
        d.update({k: zb[k] for k in zb.files})
        r = heads_on(d, vad_fn, seg, turn, a.device)
        for k in ("vad", "p5", "p"):
            out[f"{cid}|{k}"] = r[k].astype(np.float32)
        out[f"{cid}|pu"] = d["pu"].astype(np.float32)
        out[f"{cid}|po"] = d["po"].astype(np.float32)
    np.savez(f, **out)
    log(f"hop6 {name}: {len(meta)} clips in {time.time() - t0:.0f} s")


def load_ho(fname):
    z = np.load(HO / fname)
    out = {}
    for k in z.files:
        cid, kk = k.split("|")
        out.setdefault(cid, {})[kk] = z[k]
    return out


# --------------------------------------------------------------------------- scoring (held-out and evaluation)
def _sess_ho(meta):
    """eot_latency-style sessions of the held-out clips."""
    return [{"key": k, **v} for k, v in meta.items()]


def ho_score(sig, meta, rule, comp, gcache=None):
    """One rule on the held-out set: {'calls': oto pooled, 'meet': AMI pooled, 'asst': st3 accuracy / p50 / FF}
    with per-session details for the bootstrap."""
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    gcache = {} if gcache is None else gcache
    pk = "p" if rule["mode"] == "head" else "p5"
    per = {"calls": [], "quiet": [], "meet": []}
    times, comps, asess = {}, {}, []
    for k, m in meta.items():
        s = sig[k]
        T = min(len(s["vad"]), m["T"])
        h = {"v": list(range(T)), "t": [round((v + 1) * FRAME, 4) for v in range(T)], "p": s[pk][:T],
             "vad": s["vad"][:T], "pu": s["pu"][:T], "po": s["po"][:T]}
        db = np.load(HO / "db" / f"{k}.npy")[:T]
        tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, gcache.setdefault(k, {}))]
        ss = {"key": k, **m}
        if m["src"] == "st3":
            times[k], comps[k] = tt, comp
            asess.append(ss)
        else:
            per[{"oto": "calls", "otoq": "quiet"}.get(m["src"], "meet")].append(E.score_session(tt, ss, comp))
    asc = EA.score_system(asess, times, comps)
    return {"calls": E.pool(per["calls"]), "quiet": E.pool(per["quiet"]), "meet": E.pool(per["meet"]),
            "asst": {"accuracy_pct": asc["accuracy_pct"], "p50": asc["complete"]["eot_total_ms_p50"],
                     "false_fire_pct": asc["incomplete"]["false_fire_pct"]}}


def short(r):
    c, m, s = r["calls"], r.get("meet", r.get("ami")), r["asst"]
    q = r.get("quiet")
    qs = f"| quiet {q['eot_total_ms_p50']} {q['false_interruption_pct']} {q['missed_pct']} " if q and q.get("n_turns") else ""
    return (f"calls {c['eot_total_ms_p50']} {c['false_interruption_pct']} {c['missed_pct']} {qs}| meet {m['eot_total_ms_p50']} "
            f"{m['false_interruption_pct']} {m['missed_pct']} | asst {s['accuracy_pct']} {s['p50']} {s['false_fire_pct']}")



def stage_evp6(a):
    """A 0.6B candidate (--vad / --seg / --turn as hop6) on the evaluation cache (calls + AMI sessions with the
    dumped P(user) / P(other); assistant clips without a print) -> TF/evp6_<combo>.npz (per session: vad, p5, p on
    every frame). Evaluation only: never used to choose."""
    import torch
    torch.set_num_threads(2)
    name = combo_name(a.vad, a.seg, a.turn)
    f = TF / f"evp6_{name}.npz"
    if f.exists():
        log("exists", f)
        return
    vad_fn = None if a.vad == "served" else load_vad(a.vad, a.device)[0]
    seg = _load_seg(a.seg, a.device)
    turn = _load_turn_head(a.turn, a.device)
    out = {}
    t0 = time.time()
    for which in ("calls", "asst"):
        for k, dd in dumps(which).items():
            z = np.load(EV / f"{k}.npz")
            d = {kk: z[kk] for kk in z.files}
            T = len(d["vad"])
            h = dd["head"]
            v = np.clip(np.asarray(h["v"]), 0, T - 1)
            d["pu"], d["po"] = np.zeros(T, np.float32), np.zeros(T, np.float32)
            d["pu"][v], d["po"][v] = h["pu"], h["po"]
            r = heads_on(d, vad_fn, seg, turn, a.device)
            for kk in ("vad", "p5", "p"):
                out[f"{k}|{kk}"] = r[kk].astype(np.float32)
    np.savez(f, **out)
    log(f"evp6 {name}: {len(out) // 3} sessions in {time.time() - t0:.0f} s")


def ev_sets(name):
    """(calls/AMI dumps, assistant dumps) of the served 0.6B with a candidate's vad / p5 / p (TF/evp6_<name>.npz)
    on the dumped frames; name '115m' = the 115M's shipped dumps, 'dump' = the served 0.6B as dumped."""
    if name == "115m":
        return dumps_115m()
    d6, a6 = dumps_0p6b()
    if name == "dump":
        return d6, a6
    z = np.load(TF / f"evp6_{name}.npz")
    for D in (d6, a6):
        for k, d in D.items():
            h = dict(d["head"])
            v = np.asarray(h["v"])
            for kk in ("vad", "p5", "p"):
                x = z[f"{k}|{kk}"]
                h[kk] = x[np.clip(v, 0, len(x) - 1)].tolist()
            D[k] = {**d, "head": h}
    return d6, a6


def ev_score(D, A, rule, comp_scale=1.0, boot=0, seed=0):
    """One rule on the evaluation sets, the core_0p6b_heads._score_sets protocol (served chunk compute added; the
    eot_latency / eot_assistant scorers), with a session bootstrap (boot resamples) of every reported number."""
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    pk = "p" if rule["mode"] == "head" else "p5"
    sess = [s_ for s_ in E.sessions() if s_["key"] in D]
    asess = EA.sessions(A)
    per = {"calls": [], "ami": []}
    for ss in sess:
        d = D[ss["key"]]
        h = dict(d["head"])
        h["p"] = h[pk]
        comp = comp_scale * d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"{ss['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        per["calls" if ss["set"] in E.SCOPES["two_party_user"] else "ami"].append(E.score_session(tt, ss, comp))
    times, comps = {}, {}
    for s_ in asess:
        d = A[s_["key"]]
        h = dict(d["head"])
        h["p"] = h[pk]
        comp = comp_scale * d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"asst_{s_['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        times[s_["key"]] = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        comps[s_["key"]] = comp

    def asst(ss):
        sc = EA.score_system(ss, times, comps)
        return {"accuracy_pct": sc["accuracy_pct"], "p50": sc["complete"]["eot_total_ms_p50"],
                "p95": sc["complete"]["eot_total_ms_p95"], "false_fire_pct": sc["incomplete"]["false_fire_pct"],
                "missed_pct": sc["complete"]["missed_pct"]}
    res = {"calls": E.pool(per["calls"]), "ami": E.pool(per["ami"]), "asst": asst(asess)}
    if boot:
        rng = np.random.default_rng(seed)
        bc = {"calls": [], "ami": [], "asst": []}
        for _ in range(boot):
            for sc in ("calls", "ami"):
                P = per[sc]
                bc[sc].append(E.pool([P[i] for i in rng.integers(0, len(P), len(P))]))
            bc["asst"].append(asst([asess[i] for i in rng.integers(0, len(asess), len(asess))]))

        def ci(rows, q):
            x = np.array([r_[q] for r_ in rows if r_[q] is not None], float)
            return [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)] if len(x) else None
        for sc in ("calls", "ami"):
            res[sc]["ci95"] = {q: ci(bc[sc], q) for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")}
        res["asst"]["ci95"] = {q: ci(bc["asst"], q) for q in ("accuracy_pct", "p50", "false_fire_pct")}
    return res



def stage_vadlag(a):
    """Diagnosis 1 (features or head?): for VAD heads --tags (served = the shipped GRU; 115m = the 115M's served head,
    from its turn_v4 cache) the offset lag / flicker against the word-level activity labels of the oto held-out clips,
    and the lag after the labelled ends of the evaluation calls (eval cache; information only)."""
    import torch
    import eot_latency as E
    import turn_v5 as V5
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(2)
    vg = _val_groups()
    cl = [c for c in _turn_manifest() if c["src"] == "oto" and (c["src"], c["meeting"]) in vg
          and (C.TURN_BLK / f"{c['id']}.npz").exists()]
    sess = [s_ for s_ in E.sessions() if s_["set"] != "ami"]
    res = {}
    for tag in a.tags.split(","):
        P, Y, Wn = [], [], []
        fn = None if tag in ("served", "115m") else load_vad(tag, a.device)[0]
        for i, c in enumerate(cl):
            if tag == "115m":
                p = _V5_ORIG["npz"](V5.FEATS4 / f"{c['id']}.npz")["vad"].astype(np.float32)
            elif tag == "served":
                p = np.load(C.TURN_INP / f"{c['id']}.npz")["vad"].astype(np.float32)
            else:
                z = np.load(C.TURN_BLK / f"{c['id']}.npz")
                T = len(np.load(C.TURN_INP / f"{c['id']}.npz")["pu"])
                p = fn({k: z[k][:T] for k in z.files})
            P.append(p)
            Y.append(_act_labels(c, len(p)))
            Wn.append(np.full(len(p), i))
        P, Y, Wn = np.concatenate(P), np.concatenate(Y), np.concatenate(Wn)
        r = {"oto_val": {"auc": round(float(roc_auc_score(Y > 0.5, P)), 4), **offset_lag(P, Y, Wn)}}
        P, Y, Wn = [], [], []
        for i, c in enumerate(cl):  # the quiet-channel variants of the same clips
            qid = f"q_{c['id']}"
            if not inp_path(qid).exists():
                continue
            if tag == "115m":
                p = np.load(TF / "quiet" / "i115" / f"{qid}.npz")["vad"].astype(np.float32)
            elif tag == "served":
                p = np.load(inp_path(qid))["vad"].astype(np.float32)
            else:
                z = np.load(blk_path(qid))
                T = len(np.load(inp_path(qid))["pu"])
                p = fn({k: z[k][:T] for k in z.files})
            P.append(p)
            Y.append(_act_labels(c, len(p)))
            Wn.append(np.full(len(p), i))
        if P:
            P, Y, Wn = np.concatenate(P), np.concatenate(Y), np.concatenate(Wn)
            r["oto_val_quiet"] = {"auc": round(float(roc_auc_score(Y > 0.5, P)), 4), **offset_lag(P, Y, Wn)}
        if tag == "115m":
            dump, _ = dumps_115m()
            vads = {k: np.asarray(d["head"]["vad"]) for k, d in dump.items()}
        else:
            vads = {}
            for s_ in sess:
                z = np.load(EV / f"{s_['key']}.npz")
                vads[s_["key"]] = z["vad"].astype(np.float32) if tag == "served" else fn({k: z[k] for k in z.files})
        r["eval_calls"] = _lag_stats(vads, sess)
        res[tag] = r
        log(tag, r)
    d = load_json("vadlag", {})
    d.update(res)
    save("vadlag", d)



# --------------------------------------------------------------------------- the v5 classifier (diagnosis 2)
VADC = TF / "vadc"


def stage_vadall(a):
    """A VAD candidate (--vad) on every cached turn clip (TURN_BLK blocks) -> TF/vadc/<tag>/<id>.npy (fp16): the VAD
    channel and event frames the v5 classifier is retrained with when that VAD would ship. Resumable."""
    import torch
    torch.set_num_threads(2)
    od = VADC / a.vad
    od.mkdir(parents=True, exist_ok=True)
    fn, bl = load_vad(a.vad, a.device)
    ids = sorted(p_.name[:-4] for p_ in C.TURN_INP.glob("*.npz")) + sorted(p_.name[:-4] for p_ in (Q / "inp").glob("q_*.npz"))
    todo = [i for i in ids if not (od / f"{i}.npy").exists()]
    log(f"vadall {a.vad}: {len(todo)} of {len(ids)}")
    t0 = time.time()
    n = 0
    for cid in todo:
        if time.time() - t0 > a.budget:
            break
        T = len(np.load(inp_path(cid))["pu"])
        z = np.load(blk_path(cid))
        p = fn({f"b{b}": z[f"b{b}"][:T] for b in bl})
        np.save(od / f"{cid}.npy", p.astype(np.float16))
        n += 1
    log(f"vadall {a.vad}: {n} in {time.time() - t0:.0f} s; {len(todo) - n} left")


def _v5_vad(vad_tag):
    """turn_v5 on the 0.6B inputs (core_0p6b_heads._v5_mod) with the VAD channel of --vad (served = as cached)."""
    V5 = C._v5_mod()
    if not hasattr(V5, "_npz_orig"):
        V5._npz_orig = V5.npz
    if vad_tag == "served":
        V5.npz = V5._npz_orig
        return V5

    def npz(path, _o=V5._npz_orig, _t=vad_tag):
        d = _o(path)
        f = VADC / _t / (Path(path).name[:-4] + ".npy")
        if Path(path).parent in (C.TURN_INP, Q / "inp") and f.exists():
            d["vad"] = np.load(f)[: len(d["pu"])]
        return d
    V5.npz = npz
    return V5


T5 = {}  # the 115M turn_v5 cache paths (captured before core_0p6b_heads._v5_mod repoints the module)


def _t5_paths():
    if not T5:
        T5.update({"W": _V5_ORIG["W"], "BLK": _V5_ORIG["BLK"], "CUTS": _V5.CUTS, "FEATS4": _V5.FEATS4})
    return T5


def stage_teacher(a):
    """The 115M's shipped v5 (c5, block 8 of its own encoder, its own served inputs) on every frame of every turn
    clip -> TF/teacher/<id>.npy (fp16, calibrated p): the distillation target of the 0.6B's v5 (same audio, same 80 ms
    frame clock). Clips whose 115M inputs are missing get none. Resumable."""
    import torch
    torch.set_num_threads(2)
    P = _t5_paths()
    import turn_v5 as V5
    model, ck = V5.load_seg(P["W"] / "c5" / "model.pt", a.device)
    od = TF / "teacher"
    od.mkdir(parents=True, exist_ok=True)
    ids = sorted(p_.name[:-4] for p_ in C.TURN_INP.glob("*.npz"))
    todo = [i for i in ids if not (od / f"{i}.npy").exists() and not i.startswith("stt_")]
    log(f"teacher: {len(todo)} of {len(ids)}")
    t0 = time.time()
    n = miss = 0
    for cid in todo:
        if time.time() - t0 > a.budget:
            break
        f = (P["W"] / "st3" / f"{cid}.npz") if cid.startswith("st3_") else (
            P["CUTS"] / f"{cid}.npz" if (P["CUTS"] / f"{cid}.npz").exists() else P["FEATS4"] / f"{cid}.npz")
        b = P["BLK"] / f"{cid}.npz"
        if not f.exists() or not b.exists():
            np.save(od / f"{cid}.npy", np.zeros(0, np.float16))
            miss += 1
            continue
        d = _V5_ORIG["npz"](f)
        T = len(d["pu"])
        x = np.load(b)["b8"][:T].astype(np.float32)
        ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
        p = V5.seg_probs(model, x, ex, d["n"].astype(np.int32), d["y"].astype(np.int32), a.device, bs=512)
        np.save(od / f"{cid}.npy", p.astype(np.float16))
        n += 1
    log(f"teacher: {n} in {time.time() - t0:.0f} s ({miss} without 115M inputs); {len(todo) - n - miss} left")


def stage_segtrain(a):
    """The 0.6B v5 classifier, turn_v5's c5 recipe (2 x 256 Transformer on <= 8 s of one block + VAD / P(user) /
    P(other) + RNNT text; ends vs pauses / dips / cuts / smart-turn clips; smart-turn and LM-completeness auxiliaries;
    lr 1e-4, dropout 0.2, wd 0.05, batch 256), with: --vad (the VAD channel and event frames), --block, --steps, and
    --kd w: + w x BCE(main logit, the 115M v5's p at the same frame) (stage teacher). Checkpoint = best held-out
    end-vs-pause AUC (turn_v5's split); temperature calibration on held-out at the end.
    -> TF/seg/<tag>/model.pt. Resumable (--budget s per call)."""
    import torch
    import torch.nn.functional as F
    from audioforge.heads.turn_seg import SegTurn
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    V5 = _v5_vad(a.vad)
    if not hasattr(V5, "_samples_orig"):
        V5._samples_orig = V5.samples_of
    V5.samples_of = V5._samples_orig
    if a.st3dips:  # + the frames inside a smart-turn utterance where the served assistant trigger asks: energy within
        # 6 dB of the clip's quiet floor or VAD < 0.4, between the first and the last VAD-on frame: label 0 (kind 5,
        # drawn with batch weight --st3dips)
        from audioforge.datasets import smartturn as ST
        meta_st = json.loads(ST.build_cache(verbose=False).read_text())
        wav = np.load(ST.DEFAULT_ROOT / "cache" / "human_5_all.npy", mmap_mode="r")
        off = meta_st["offsets"]

        def samples_of(c, d, T, _o=V5._samples_orig):
            S_ = _o(c, d, T)
            if c["src"] != "st3":
                return S_
            vad = d["vad"].astype(np.float32)
            on = np.nonzero(vad > 0.5)[0]
            if len(on) < 2:
                return S_
            u = np.asarray(wav[off[c["idx"]]: off[c["idx"] + 1]], np.float32)
            db = V5.frame_abs_db(u, min(T, len(u) // 1280))
            fl = np.percentile(db, 10)
            for v in range(int(on[0]) + 3, min(int(on[-1]), len(db))):
                if db[v] < fl + 6.0 or vad[v] < 0.4:
                    S_.append((v, 0.0, 1.0, 5))
            return S_
        V5.samples_of = samples_of
    if not hasattr(V5, "_all_clips_q"):
        V5._all_clips_q, V5._feat_path_q, V5._blk_q = V5.all_clips, V5.feat_path, V5.BLK
    V5.all_clips, V5.feat_path, V5.BLK = V5._all_clips_q, V5._feat_path_q, V5._blk_q
    if a.quiet_train:  # + the quiet-channel variants of oto training windows (stage quiet --which train) as oto clips
        union = TF / "blk_union"
        if not (union / "done").exists():
            union.mkdir(exist_ok=True)
            for d_ in (C.TURN_BLK, Q / "blk"):
                for f in d_.glob("*.npz"):
                    if not (union / f.name).exists():
                        (union / f.name).symlink_to(f)
            (union / "done").write_text("1")
        man = {c["id"]: c for c in _turn_manifest()}
        qids = [p_.name[2:-4] for p_ in (Q / "inp").glob("q_*.npz")]
        qtrain = [dict(man[i], id=f"q_{i}") for i in qids if i in man and (Q / "blk" / f"q_{i}.npz").exists()]
        vg = _val_groups()
        qtrain = [c for c in qtrain if (c["src"], c["meeting"]) not in vg]

        def all_clips(with_cuts=True, _o=V5._all_clips_q):
            return _o(with_cuts) + qtrain

        def feat_path(c, _o=V5._feat_path_q):
            return Q / "inp" / f"{c['id']}.npz" if c["id"].startswith("q_") else _o(c)
        V5.all_clips, V5.feat_path, V5.BLK = all_clips, feat_path, union
        log(f"+ {len(qtrain)} quiet-channel oto training windows")
    od = TF / "seg" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    dev = a.device
    V5.USE_ST32 = False
    t0 = time.time()
    data = V5.SegData(str(a.block), False)
    S = data.S
    V5.LM_AB = V5.fit_lm_calib(S)
    kd = np.full(len(S), -1.0)
    if a.kd > 0:
        tc = {}
        for i, c in enumerate(data.clips):
            f = TF / "teacher" / f"{c['id']}.npy"
            tc[i] = np.load(f).astype(np.float32) if f.exists() else np.zeros(0, np.float32)
        for j, r in enumerate(S):
            t_ = tc[int(r[0])]
            if int(r[1]) < len(t_):
                kd[j] = t_[int(r[1])]
        log(f"kd targets on {(kd >= 0).mean() * 100:.1f} % of samples")
    S = np.concatenate([S, kd[:, None]], 1)
    tr, va = S[S[:, 7] == 0], S[S[:, 7] == 1]
    if len(va) > 30000:
        va = va[np.sort(np.random.default_rng(0).choice(len(va), 30000, replace=False))]
    D_ = data.clips[0]["x"].shape[1]
    model = SegTurn(d_in=D_, n_extra=3, d=256, n_layers=2, heads=4, ff=1024, win=V5.WIN, use_text=True, n_pros=0,
                    dropout=0.2, block=str(a.block)).to(dev)
    npar = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    step, hist, best = 0, [], -1.0
    last = od / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
    log(f"{a.tag}: {npar / 1e6:.2f}M params; train {len(tr)} val {len(va)}; data {time.time() - t0:.0f} s; step {step}")
    args = {"tag": a.tag, "block": str(a.block), "vad": a.vad, "kd": a.kd, "steps": a.steps, "lr": a.lr,
            "seed": a.seed, "pros": False, "st3dips": a.st3dips, "quiet_train": a.quiet_train}
    if step < a.steps:
        kinds = tr[:, 4].astype(int)
        kw = {0: 1.0, 1: 1.5, 2: 0.5, 3: 0.7, 4: 0.5, 5: a.st3dips}
        wts = np.array([kw[k] for k in kinds]) / np.bincount(kinds, minlength=6)[kinds]
        wts /= wts.sum()
        rng = np.random.default_rng(a.seed + step)
        model.train()
        while step < a.steps and time.time() - t0 < a.budget:
            lr = a.lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
            for g in opt.param_groups:
                g["lr"] = lr
            rows = tr[rng.choice(len(tr), 256, p=wts)]
            b = data.batch(rows, dev)
            drop = torch.rand(len(rows), device=dev) < 0.25
            b[2][drop, :, 1:3] = 0.0
            out = model(*b)
            lab = torch.tensor(rows[:, 2], dtype=torch.float32, device=dev)
            npos = lab.sum().clamp(min=1)
            pw = ((len(lab) - npos) / npos).clamp(0.1, 10.0)
            loss = F.binary_cross_entropy_with_logits(out["main"], lab, pos_weight=pw)
            stt = torch.tensor(rows[:, 5], dtype=torch.float32, device=dev)
            ms = stt >= 0
            if ms.any():
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(out["st"][ms], stt[ms])
            if "compl" in out and V5.LM_AB is not None:
                lm = torch.tensor(rows[:, 6], dtype=torch.float32, device=dev)
                ml = lm > -19.5
                if ml.any():
                    ct = torch.sigmoid(V5.LM_AB[0] * lm + V5.LM_AB[1])
                    loss = loss + 0.3 * F.binary_cross_entropy_with_logits(out["compl"][ml], ct[ml])
            if a.kd > 0:
                kt = torch.tensor(rows[:, 8], dtype=torch.float32, device=dev)
                mk = kt >= 0
                if mk.any():
                    loss = loss + a.kd * F.binary_cross_entropy_with_logits(out["main"][mk], kt[mk])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0 or step == a.steps:
                r, _ = V5.seg_eval(model, data, va[:, :8], dev)
                r = {"step": step, "loss": round(float(loss), 4), **r}
                hist.append(r)
                log(r)
                if r["auc"] > best:
                    best = r["auc"]
                    torch.save({"cfg": model.cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                "step": step, "val": r, "args": args, "lm_ab": V5.LM_AB}, od / "model.pt")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist,
                            "best": best}, last)
                (od / "history.json").write_text(json.dumps({"args": args, "params": npar, "hist": hist}, indent=1))
    log(f"{a.tag}: step {step} ({time.time() - t0:.0f} s)")
    if step >= a.steps and not (od / "calibrated").exists() and time.time() - t0 < a.budget + 120:
        V5.calibrate(od, data, va[:, :8], dev)
        (od / "calibrated").write_text("1")
        h = json.loads((od / "history.json").read_text())
        save("seg", {"args": args, "best": max(h["hist"], key=lambda r_: r_["auc"]), "last": h["hist"][-1],
                     "params": npar}, sub=a.tag)



# --------------------------------------------------------------------------- rule scan on the held-out set (preset constants)
def rule_grid(fam):
    import itertools
    G = []
    if fam in ("fast", "balanced"):
        for k, mvt, mp, fb in itertools.product((1, 2, 3, 4, 5, 6), (0.4, 0.5, 0.6),
                                                (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97), (8, 9, 10, 12)):
            if fb <= k:
                continue
            G.append({"mode": "model", "gate": True, "quiet_db": None, "k": k, "mvt": mvt, "mp": mp, "fb": fb,
                      "vad_thr": 0.4, "reask": True, "others": (12, 8)})
    if fam == "balanced":
        for k, th, fb, vt in itertools.product((1, 2, 3, 4), (0.9, 0.95, 0.97, 0.99), (6, 8, 10, 12), (0.4, 0.5, 0.6)):
            G.append({"mode": "head", "gate": True, "quiet_db": None, "k": k, "th": th, "fb": fb, "vad_thr": vt,
                      "others": (12, 8)})
    if fam == "assistant":
        # fallback 37-43 frames: a VAD without a hangover goes quiet up to ~0.3 s before the audible end, so the
        # 2960 ms timer (tuned on the 115M's VAD, whose tail adds ~0.3 s) fires inside the incomplete clips' 3 s window
        for k, mvt, mp, qd, fb in itertools.product((2, 3, 4, 5), (0.4, 0.5), (0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99),
                                                    (6.0, 4.0, None), (37, 40, 43)):
            G.append({"mode": "model", "gate": True, "quiet_db": qd, "mqo": qd is not None, "k": k, "mvt": mvt,
                      "mp": mp, "fb": fb, "vad_thr": 0.4, "reask": True, "others": (12, 8)})
    return G


def ho_bars():
    """The 115M's shipped presets on the held-out set (its own heads): the held-out bars a 0.6B rule must meet."""
    meta = ho_meta()
    sig = load_ho("p_115m.npz")
    rules = C._preset_rules()
    return {pn: ho_score(sig, meta, r, COMP["115m"]) for pn, r in rules.items()}


def pick(rows, fam, bars):
    """The held-out choice: fast / balanced = the fastest calls-scope p50 among rules with calls FI and missed no
    worse than the 115M's on the same clips (balanced also the meeting scope's FI and missed); assistant = the most
    accurate st3 rule with p50 and false fires no worse than the 115M's (ties: faster)."""
    b = bars[fam]
    if fam == "assistant":
        ok = [r for r in rows if r["asst"]["p50"] is not None and r["asst"]["p50"] <= b["asst"]["p50"]
              and r["asst"]["false_fire_pct"] <= b["asst"]["false_fire_pct"]]
        return (max(ok, key=lambda r: (r["asst"]["accuracy_pct"], -r["asst"]["p50"])) if ok else None), len(ok)
    c, bc = (lambda r: r["calls"]), b["calls"]
    ok = [r for r in rows if c(r)["eot_total_ms_p50"] is not None
          and c(r)["false_interruption_pct"] <= bc["false_interruption_pct"] and c(r)["missed_pct"] <= bc["missed_pct"]
          and r["quiet"]["false_interruption_pct"] <= b["quiet"]["false_interruption_pct"]
          and r["quiet"]["missed_pct"] <= b["quiet"]["missed_pct"]
          and (fam != "balanced" or (r["meet"]["false_interruption_pct"] <= b["meet"]["false_interruption_pct"]
                                     and r["meet"]["missed_pct"] <= b["meet"]["missed_pct"]))]
    return (min(ok, key=lambda r: (c(r)["eot_total_ms_p50"] + (r["quiet"]["eot_total_ms_p50"] or 9999),
                                   c(r)["false_interruption_pct"])) if ok else None), len(ok)


def stage_hoscan(a):
    """Every rule of --fam on the held-out set for one 0.6B candidate (hop6 output) -> TF/ho/scan_<combo>_<fam>.json
    and the held-out pick against the 115M's held-out numbers (runs/... hoscan.<combo>.<fam>)."""
    meta = ho_meta()
    name = "115m" if a.vad == "115m" else combo_name(a.vad, a.seg, a.turn)  # --vad 115m: the control (its own heads)
    sig = load_ho("p_115m.npz" if name == "115m" else f"p6_{name}.npz")
    comp = COMP["115m" if name == "115m" else "0p6b"]
    bars = load_json("ho_bars")
    if bars is None:
        bars = ho_bars()
        save("ho_bars", bars)
    t0 = time.time()
    rows, gc = [], {}
    for fam in a.fam.split(","):
        for r in rule_grid(fam):
            rows.append({"rule": r, **ho_score(sig, meta, r, comp, gc)})
        (HO / f"scan_{name}_{fam}.json").write_text(json.dumps(rows))
        pk, n_ok = pick(rows, fam, bars)
        log(fam, f"{len(rows)} rules, {n_ok} meet the 115M's held-out numbers ({time.time() - t0:.0f} s); 115M:",
            short(bars[fam]), "| pick:", short(pk) if pk else None, pk["rule"] if pk else "")
        d = load_json("hoscan", {})
        d.setdefault(name, {})[fam] = {"n_rules": len(rows), "n_ok": n_ok, "pick": pk}
        save("hoscan", d)
        rows = []



def stage_hoquick(a):
    """The shipped presets (shared constants) on the held-out set for candidates --tags (combo names; 115m = the 115M's
    own heads): the effect of a head swap before any constant is re-tuned -> runs/... hoquick.<combo>."""
    meta = ho_meta()
    rules = C._preset_rules()
    d = load_json("hoquick", {})
    for name in a.tags.split(","):
        sig = load_ho("p_115m.npz" if name == "115m" else f"p6_{name}.npz")
        comp = COMP["115m" if name == "115m" else "0p6b"]
        d[name] = {pn: ho_score(sig, meta, r, comp) for pn, r in rules.items()}
        for pn in rules:
            log(f"{name:28s} {pn:9s} {short(d[name][pn])}")
    save("hoquick", d)



# --------------------------------------------------------------------------- quiet-channel variants (TurnBench-like levels)
Q = TF / "quiet"
Q_SPEECH_DB = (-48.0, -34.0)  # active-speech level drawn per clip (TurnBench user channels: median -41.8 dBFS)
Q_FLOOR_DB = (-76.0, -58.0)  # noise floor (TurnBench non-turn frames: median -70.1, 10-90 % -77.6..-53.1 dBFS)
Q_BLEED_DB = -35.0  # the other party's channel, relative to the user's speech level, on half of the clips


def _colored(n, rng):
    w = rng.standard_normal(n).astype(np.float32)
    col = rng.choice(["white", "pink", "brown"], p=[0.3, 0.4, 0.3])
    if col != "white":
        f = np.fft.rfft(w)
        k = np.arange(len(f)) + 1.0
        f /= np.sqrt(k) if col == "pink" else k
        w = np.fft.irfft(f, n).astype(np.float32)
    return w / (np.sqrt(np.mean(w ** 2)) + 1e-12)


def quiet_audio(c, au, man):
    """An oto turn clip as a quiet, noisy user channel: the user's speech scaled to a level from Q_SPEECH_DB, a
    coloured noise floor from Q_FLOOR_DB, and on half of the clips the other party's channel at Q_BLEED_DB under the
    user's speech level (crosstalk). The print audio gets the same gain and floor. Deterministic per clip id."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(c["id"].encode()))
    x, pr = au(c, man)
    T = len(x) // 1280
    act = _act_labels(c, T) > 0.5
    xs = x[: T * 1280].reshape(T, 1280)[act]
    lvl = 10 * np.log10(np.mean(xs.astype(np.float64) ** 2) + 1e-12) if act.any() else -30.0
    tgt, fl = rng.uniform(*Q_SPEECH_DB), rng.uniform(*Q_FLOOR_DB)
    g = 10 ** ((tgt - lvl) / 20)
    y = x * g + _colored(len(x), rng) * 10 ** (fl / 20)
    if rng.random() < 0.5:
        d = au._d("oto", man)
        oth = np.asarray(d.channels16k(c["meeting"])[1 - c["ch"]][int(round(c["a"] * SR)): int(round(c["b"] * SR))],
                         np.float32)[: len(x)]
        y[: len(oth)] += oth * g * 10 ** (Q_BLEED_DB / 20)
    pq = pr * g + _colored(len(pr), rng) * 10 ** (fl / 20)
    return y.astype(np.float32), pq.astype(np.float32), {"speech_db": round(tgt, 1), "floor_db": round(fl, 1)}


def _quiet_clips(which, n=None, seed=0):
    import random
    vg = _val_groups()
    cl = [c for c in _turn_manifest() if c["src"] == "oto" and (((c["src"], c["meeting"]) in vg) == (which == "val"))]
    if n and len(cl) > n:
        cl = random.Random(seed).sample(cl, n)
    return cl


def stage_quiet(a):
    """Quiet-channel variants of oto turn clips (quiet_audio) through the 0.6B (C.TurnExtractor: served VAD, TS-VAD
    track from the quiet print, tokens, blocks 8 / 12 / 24) -> TF/quiet/inp|blk/q_<id>.npz; --which val also through
    the 115M (turn_v4.Extractor: its served VAD, TS-VAD, tokens, pass-2 top; + pass-1 block 8 for its v5)
    -> TF/quiet/i115/q_<id>.npz. val = the held-out conversations (a fourth held-out scope), train = --n training
    windows (VAD / v5 training data). Resumable."""
    import torch
    import turn_v4 as V4
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    for d_ in ("inp", "blk", "i115", "meta"):
        (Q / d_).mkdir(parents=True, exist_ok=True)
    man = _turn_manifest()
    cl = _quiet_clips(a.which, a.n if a.which == "train" else None, a.seed)
    todo = [c for c in cl if not (Q / "inp" / f"q_{c['id']}.npz").exists()
            or (a.which == "val" and not (Q / "i115" / f"q_{c['id']}.npz").exists())]
    log(f"quiet {a.which}: {len(todo)} of {len(cl)} to do")
    if not todo:
        return
    t0 = time.time()
    au = V4.ClipAudio()
    ex6 = C.TurnExtractor(a.device)
    ex1 = V4.Extractor(device=a.device) if a.which == "val" else None
    done = 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        aud = [quiet_audio(c, au, man) for c in cs]
        xs, prs = [q[0] for q in aud], [q[1] for q in aud]
        pr6 = ex6.prints(prs)
        outs = ex6(xs, pr6, (8, 12, 24))
        for c, (inp, blk), q in zip(cs, outs, aud):
            qid = f"q_{c['id']}"
            np.savez(Q / "blk" / f"{qid}.npz", **blk)
            np.savez(Q / "inp" / f"{qid}.npz", **inp)
            (Q / "meta" / f"{qid}.json").write_text(json.dumps(q[2]))
        if ex1 is not None:
            prints = [voiceprint(ex1.cpu, p) if len(p) >= 8000 else None for p in prs]
            o1 = ex1(xs, prints)
            with torch.no_grad():
                x_, xl = ex1.m._pad(xs)
                enc, elen, hid = ex1.m.encode(x_, xl, V4.ATT, return_hidden=True)
                b8 = hid[7].half().cpu().numpy()
            for j, (c, o) in enumerate(zip(cs, o1)):
                L = len(o["pu"])
                np.savez(Q / "i115" / f"q_{c['id']}.npz", **o, b8=b8[j, :L])
        done += len(cs)
    log(f"quiet {a.which}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")



# --------------------------------------------------------------------------- the served candidate (heads only)
def stage_build(a):
    """A served 0.6B candidate = served_0p6b_v0.1.afm (every NVIDIA tensor and the speaker head unchanged) with heads.vad
    = --vad (mlp -> type frame, gru64 -> frame_gru; from_layers = its blocks), heads.turn_seg = --seg, heads.turn =
    --turn (f1 = unchanged) and cfg turn_presets = --presets-json -> TF/cand_<tag>.afm; with --ship also
    assets/served_heads_0p6b_v<--version>.pt (hub.export_heads against the .nemo) + a rebuild check."""
    import copy
    import torch
    from audioforge import hub
    from audioforge.model import SpeechModel
    from audioforge.train import load_model, save_model
    torch.set_num_threads(2)
    base = load_model(str(C.SERVED_0P6B), "cpu")
    cfg = copy.deepcopy(base.cfg)
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    changed = []
    if a.vad != "served":
        ck = torch.load(HEADS / f"vad_{a.vad}.pt", map_location="cpu", weights_only=False)
        bl = [int(b) - 1 for b in ck["blocks"]]
        st = ck["state_dict"]
        for k in [k for k in sd if k.startswith("heads.vad.") or k == "layer_mix.vad"]:
            del sd[k]
        if ck["arch"] == "mlp":
            cfg["heads"]["vad"] = {"type": "frame", "key": "vad", "hidden": 64, "from_layers": bl, "weight": 0.5}
            sd.update({"heads.vad.net.0.weight": st["inp.weight"], "heads.vad.net.0.bias": st["inp.bias"],
                       "heads.vad.net.2.weight": st["out.weight"], "heads.vad.net.2.bias": st["out.bias"]})
        elif ck["arch"].startswith("gru"):
            h = st["inp.weight"].shape[0]
            cfg["heads"]["vad"] = {"type": "frame_gru", "key": "vad", "hidden": h, "from_layers": bl, "weight": 0.5}
            sd.update({f"heads.vad.{k}": v for k, v in st.items() if k != "mix"})
        else:
            raise ValueError(f"VAD arch {ck['arch']} has no served head type")
        if len(bl) > 1:
            sd["layer_mix.vad"] = st["mix"]
        changed.append(f"vad={a.vad}")
    if a.seg != "s12":
        ck = torch.load(TF / "seg" / a.seg / "model.pt", map_location="cpu", weights_only=False)
        cfg["heads"]["turn_seg"] = {"type": "turn_seg", "weight": 0.0, **ck["cfg"]}
        for k in [k for k in sd if k.startswith("heads.turn_seg.")]:
            del sd[k]
        sd.update({f"heads.turn_seg.{k}": v for k, v in ck["state_dict"].items()})
        changed.append(f"turn_seg={a.seg}")
    if a.turn != "f1":
        ck = torch.load(HEADS / f"turn_{a.turn}.pt", map_location="cpu", weights_only=False)
        if int(ck.get("block", 24)) != 24:
            cfg["heads"]["turn"]["from_layers"] = [int(ck["block"]) - 1]
        sd.update({f"heads.turn.{k}": v for k, v in ck["state_dict"].items()})
        changed.append(f"turn={a.turn}")
    cfg["turn_presets"] = json.loads(a.presets_json) if a.presets_json else cfg.get("turn_presets")
    cfg["name"] = f"served_0p6b_v{a.version}" if a.ship else f"served_0p6b_cand_{a.tag}"
    m = SpeechModel(cfg, base.tokenizer)
    m.load_state_dict(sd, strict=True)
    out = (C.W / f"{cfg['name']}.afm") if a.ship else (TF / f"cand_{a.tag}.afm")
    save_model(m.eval(), out)
    back = load_model(str(out), "cpu")
    assert hub.state_hash(back.state_dict()) == hub.state_hash(m.state_dict())
    keep = [k for k in base.state_dict() if not k.startswith(("heads.vad.", "heads.turn_seg.", "heads.turn.", "layer_mix.vad"))]
    bsd = back.state_dict()
    assert all(torch.equal(base.state_dict()[k], bsd[k]) for k in keep), "a frozen tensor changed"
    rec = {"afm": str(out), "changed": changed, "turn_presets": cfg["turn_presets"], "n_unchanged": len(keep)}
    if a.ship:
        heads = ROOT / "assets" / f"served_heads_0p6b_v{a.version}.pt"
        info = hub.export_heads(out, C.NEMO, heads, base_key="asr_0p6b")
        got = hub.build_served(C.NEMO, heads, TF / "rebuilt.afm")
        assert got == info["state_hash"], (got, info)
        (TF / "rebuilt.afm").unlink()
        rec.update({**info, "heads": str(heads), "size": heads.stat().st_size, "sha256": hub.sha256_file(heads)})
    log(rec)
    save("build", rec, sub=a.tag)



# --------------------------------------------------------------------------- the per-frame turn head (diagnosis 3)
def stage_turntrain(a):
    """The balanced preset's per-frame turn head (core_0p6b_heads.stage_turn_train's recipe: the served TurnHead without
    speaker kernels, TS-VAD columns, RNNT text, the served objective, turn_v4 mix, batch 16, AdamW --lr 1e-3, cosine,
    3000 steps) reading --block instead of the top block. -> TF/heads/turn_<tag>.pt (best held-out end-vs-pause AUC)."""
    import random
    import torch
    import torch.nn.functional as F
    import turn_v4 as V4
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    dev = a.device
    HEADS.mkdir(parents=True, exist_ok=True)
    od = TF / f"frame_{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    m = C.load_core(dev, heads=("turn",), cfgs={"turn": C.TURN_CFG}, fresh=("turn",))
    head = m.heads["turn"]
    for p_ in m.parameters():
        p_.requires_grad_(False)
    params = list(head.parameters())
    for p_ in params:
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    step, hist, best = 0, [], 1e9
    last = od / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
    if step >= a.steps:
        return
    blk = f"b{a.block}"
    man = _turn_manifest()
    groups = sorted({(c["src"], c.get("meeting", c["id"])) for c in man})
    val_g = set(random.Random(0).sample(groups, int(0.08 * len(groups))))
    tr, va = [], []
    t0 = time.time()
    for c in man:
        f, b = C.TURN_INP / f"{c['id']}.npz", C.TURN_BLK / f"{c['id']}.npz"
        if not f.exists() or not b.exists():
            continue
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        T = len(d["pu"])
        e = np.load(b)[blk]
        d["e"] = e[:T] if len(e) >= T else np.concatenate([e, np.repeat(e[-1:], T - len(e), 0)])
        d["tgt"], d["wt"], d["ends"], d["pauses"] = V4.targets(c, d["vad"].astype(np.float32), T)
        d["tgt0"], d["wt0"] = V4.targets.orig
        d["src"], d["id"] = c["src"], c["id"]
        (va if (c["src"], c.get("meeting", c["id"])) in val_g else tr).append(d)
    by = {}
    for d in tr:
        by.setdefault("meet" if d["src"] in ("ami", "icsi") else d["src"], []).append(d)
    mix = {"oto": 0.45, "meet": 0.35, "st": 0.2}
    log(f"turntrain {a.tag} (block {a.block}): train {len(tr)}, val {len(va)}, data {time.time() - t0:.0f} s; step {step}")
    rng = random.Random(a.seed + step)
    head.train()
    while step < a.steps and time.time() - t0 < a.budget:
        lr = a.lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(step / a.steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        src = rng.choices(list(mix), weights=list(mix.values()))[0]
        b = V4.collate(rng.sample(by[src], min(16, len(by[src]))), dev)
        z = V4.head_logits(head, b)
        l0 = F.binary_cross_entropy_with_logits(z, b[7], reduction="none", pos_weight=torch.tensor(2.0, device=dev))
        loss = (l0 * b[8]).sum() / b[8].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % 250 == 0 or step == a.steps:
            r = {"step": step, "loss": round(float(loss), 4), **V4.val_metrics(head, va, dev)}
            hist.append(r)
            log(r)
            if -r["auc"] < best:
                best = -r["auc"]
                torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "cfg": C.TURN_CFG,
                            "block": int(a.block), "step": step, "val": r}, HEADS / f"turn_{a.tag}.pt")
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist, "best": best},
                       last)
    log(f"turntrain {a.tag}: step {step} ({time.time() - t0:.0f} s)")
    if step >= a.steps:
        save("turn_frame", {"block": a.block, "hist": hist, "best": max(hist, key=lambda r: r["auc"])}, sub=a.tag)



def stage_evverify(a):
    """Verification on the evaluation sets (never used to choose): --tags = evp6 combos (or 'dump' = the served 0.6B as
    shipped, '115m'); --rules-json = {preset: rule} for the 0.6B candidates (default: the shared presets); 1000-session
    bootstrap CIs. -> runs/... evverify.<label>."""
    rules = json.loads(a.rules_json) if a.rules_json else C._preset_rules()
    for k, r in rules.items():
        if "others" in r:
            r["others"] = tuple(r["others"])
    res = load_json("evverify", {})
    for name in a.tags.split(","):
        D, A = ev_sets(name)
        rr = C._preset_rules() if (name == "115m" and not a.rules_json) else rules
        if name == "dump" and not a.rules_json:  # the shipped 0.6B: its own assistant constants (cfg turn_presets)
            rr = dict(C._preset_rules())
            rr["assistant"] = dict(rr["assistant"], k=4, mp=0.99)
        out = {pn: ev_score(D, A, r, boot=a.boot) for pn, r in rr.items()}
        lab = a.label or name
        res[lab] = {"combo": name, "rules": rr, "res": out}
        for pn, x in out.items():
            log(f"{lab:24s} {pn:9s} {short(x)}  CI calls {x['calls'].get('ci95')} asst {x['asst'].get('ci95')}")
    save("evverify", res)



# --------------------------------------------------------------------------- the served candidate end to end
def engine_cand(afm, device):
    """core_0p6b_heads.engine_0p6b on another .afm (the --mode single engine, explicit enrolment, Silero preloaded)."""
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    lid = ROOT / "assets" / "lid_0p6b.pt"
    opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ROOT / "assets" / "tsvad_0p6b.pt"),
            "lid": str(lid) if lid.exists() else None,
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True}
    return Engine.load(str(afm), None, device, threads=2, **opts)


def cand_afm(tag):
    return C.W / "served_0p6b_v0.2.afm" if tag == "v0.2" else TF / f"cand_{tag}.afm"


def stage_evdump(a):
    """core_0p6b_heads.stage_eot_dump on a candidate served model (--tag: TF/cand_<tag>.afm, or v0.2) -> TF/dump_<tag>/
    (calls + AMI) and TF/asst_<tag>/ (assistant clips): the served session's own per-frame signals and compute."""
    import torch
    import audioforge.serve as S
    import eot_latency as E
    import eot_assistant as EA
    torch.set_num_threads(2)
    prints = json.loads(C.PRINTS.read_text()) if a.which == "calls" else {}
    od = TF / (f"dump_{a.tag}" if a.which == "calls" else f"asst_{a.tag}")
    od.mkdir(parents=True, exist_ok=True)
    items = E.sessions() if a.which == "calls" else EA.clips()
    todo = [x for x in items if not (od / f"{x['key']}.json").exists()]
    log(f"evdump {a.tag} {a.which}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    eng = engine_cand(cand_afm(a.tag), a.device)
    eng.warmup()
    t0, n = time.time(), 0
    for it in todo:
        if time.time() - t0 > a.budget:
            break
        k = it["key"]
        if a.which == "calls":
            x = E.read_audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="hybrid_dyn", timeout_ms=1000))
            if it["embedding"] is not None:
                s.arm_enrollment("enroll", 0, embedding=prints[k])
            blk = SR * 160 // 1000
        else:
            x = EA.audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))
            blk = 320
        if s.asr.seg is None and eng.seg_name is not None:
            sh = eng.asr.heads[eng.seg_name]
            s.asr.attach_seg(sh, next(sh.parameters()).device)
        rec = []
        C._record(s, rec, enrolled_only=a.which == "asst")
        msgs = []
        for i in range(0, len(x), blk):
            msgs += s.process(x[i:i + blk])
        msgs += s.finish()
        cm = np.asarray(list(s.chunk_ms), float)
        d = {"key": k, "audio_s": round(len(x) / SR, 3),
             "head": {kk: [r[j] for r in rec] for j, kk in enumerate(("v", "t", "p", "vad", "pu", "po", "p5"))},
             "tok_at": [int(q) for q in s.asr.tok_at],
             "chunk_ms": {"p50": round(float(np.median(cm)), 2), "p95": round(float(np.percentile(cm, 95)), 2),
                          "mean": round(float(cm.mean()), 2), "n": int(len(cm))}, "device": a.device}
        if a.which == "calls":
            d.update({"set": it["set"], "cond": it["cond"], "has_print": it["embedding"] is not None})
        else:
            d.update({"complete": it["complete"], "clip_s": round((it["b"] - it["a"]) / SR, 4)})
        (od / f"{k}.json").write_text(json.dumps(d))
        n += 1
    log(f"evdump {a.tag} {a.which}: {n} this call ({time.time() - t0:.0f} s), {len(todo) - n} left")


def served_rules(afm):
    """The rule twins (turn_v5.run_policy) of the presets as a served model resolves them (cfg turn_presets merged)."""
    from audioforge.serve import model_presets, vad_head_params
    from audioforge.server.constants import POLICY_THETA
    from audioforge.train import load_model
    m = load_model(str(afm), "cpu")
    pr = model_presets(m)
    out = {}
    for name in ("balanced", "fast", "assistant"):
        k, fb, thr, others = vad_head_params(name, presets=pr)
        r = {"gate": True, "k": k, "fb": fb, "vad_thr": thr, "others": others if others else None,
             "rt": pr[name].get("reset_thr")}
        tm = pr[name].get("turn_model")
        if tm is None:
            r.update({"mode": "head", "quiet_db": None, "th": float(pr[name].get("theta", POLICY_THETA["vad_head"]))})
        else:
            r.update({"mode": "model", "quiet_db": tm.get("quiet_db"), "mqo": tm.get("quiet_db") is not None,
                      "mvt": tm["vad_thr"], "mp": tm["p"], "reask": bool(tm.get("reask"))})
        out[name] = r
    return out


def stage_evscore(a):
    """A candidate's served dumps (stage evdump) scored with the rules its served model resolves, with 1000-session
    bootstrap CIs -> runs/... evscore.<tag>."""
    import eot_assistant as EA
    sd = EA.load_dump()
    D = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted((TF / f"dump_{a.tag}").glob("*.json")))}
    A = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted((TF / f"asst_{a.tag}").glob("*.json")))}
    for k, d in A.items():
        d["conf"] = sd[k]["conf"]
    rules = served_rules(cand_afm(a.tag))
    out = {pn: ev_score(D, A, r, boot=a.boot) for pn, r in rules.items()}
    for pn, x in out.items():
        log(f"{a.tag} {pn:9s} {short(x)} | CI calls {x['calls']['ci95']} ami {x['ami']['ci95']} asst {x['asst']['ci95']}")
    cm = [d["chunk_ms"]["p50"] for d in D.values()]
    save("evscore", {"rules": rules, "res": out, "n": [len(D), len(A)],
                     "chunk_ms_p50_median": round(float(np.median(cm)), 2)}, sub=a.tag)


def stage_servedcheck(a):
    """core_0p6b_heads.stage_served_check on a candidate: the served session with --turn-preset assistant / fast /
    balanced on --n assistant clips and --n calls sessions; its turn_end times vs the offline twin of the preset as the
    engine resolves it (served_rules), on the session's own frames. -> runs/... served_check.<tag>."""
    import torch
    import audioforge.serve as S
    import eot_latency as E
    import eot_assistant as EA
    import turn_v5 as V5
    torch.set_num_threads(2)
    afm = cand_afm(a.tag)
    eng = engine_cand(afm, a.device)
    eng.warmup()
    prints = json.loads(C.PRINTS.read_text())
    rules = served_rules(afm)
    res = load_json("served_check", {}).get(a.tag, {})
    for preset in a.fam.split(","):
        if preset in res:
            continue
        same, n, cms = 0, 0, []
        items = [("a", c) for c in EA.clips()[: a.n]] + [("c", s_) for s_ in E.sessions()[: a.n]]
        for kind, it in items:
            x = EA.audio(it) if kind == "a" else E.read_audio(it)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
            if kind == "c" and it["embedding"] is not None:
                s.arm_enrollment("enroll", 0, embedding=prints[it["key"]])
            if s.asr.seg is None and eng.seg_name is not None:
                sh = eng.asr.heads[eng.seg_name]
                s.asr.attach_seg(sh, next(sh.parameters()).device)
            rec = []
            C._record(s, rec, enrolled_only=True)
            msgs = []
            for i in range(0, len(x), 320):
                msgs += s.process(x[i:i + 320])
            msgs += s.finish()
            h = {k: [r[j] for r in rec] for j, k in enumerate(("v", "t", "p", "vad", "pu", "po", "p5"))}
            if rules[preset]["mode"] == "model":
                h["p"] = h["p5"]
            db = V5.ready_db(x, max(h["v"]) + 1, h["v"], h["t"])
            r = dict(rules[preset])
            off = [round(t_, 3) for t_, _ in V5.run_policy({"head": h}, db[np.asarray(h["v"])], r, {})]
            served = [m_["t"] for m_ in msgs if m_["type"] == "turn_end"]
            same += served == off
            n += 1
            if kind == "c":
                cms += list(s.chunk_ms)
        res[preset] = f"{same}/{n}"
        res[f"{preset}_calls_chunk_ms_p50_{a.device}"] = round(float(np.median(cms)), 2)
        log(preset, res[preset])
        save("served_check", res, sub=a.tag)



def stage_vadcmp(a):
    """The VAD rows: a candidate (--vad) against the shipped 0.6B head (core_0p6b_heads vad2 gru64_sa predictions) on
    AMI dev / ICSI dev (64 x 20 s, the published windows and labels), F1 at 0.5 and AUC with a paired 1000-window
    bootstrap; + room-tone p95. -> runs/... vadcmp.<tag>."""
    from sklearn.metrics import roc_auc_score
    res = {}
    for sn in ("ami_dev", "icsi_dev"):
        _, y, win = C._vad_old(sn, (4,))
        pa = np.load(HEADS / f"pred_{a.vad}_{sn}.npy")
        pb = np.load(C.VADW / f"pred2_gru64_sa_{sn}.npy")
        assert len(pa) == len(pb) == len(y)
        W_ = int(win.max()) + 1
        rows = [np.nonzero(win == w)[0] for w in range(W_)]

        def f1(p_, t):
            d, tt = p_ > 0.5, t > 0.5
            return 2 * float((d & tt).sum()) / max(float(d.sum() + tt.sum()), 1)
        base = {"f1_new": f1(pa, y), "f1_shipped": f1(pb, y), "auc_new": roc_auc_score(y > 0.5, pa),
                "auc_shipped": roc_auc_score(y > 0.5, pb)}
        rng = np.random.default_rng(0)
        df, da = [], []
        for _ in range(1000):
            ii = np.concatenate([rows[k] for k in rng.integers(0, W_, W_)])
            df.append(f1(pa[ii], y[ii]) - f1(pb[ii], y[ii]))
            da.append(roc_auc_score(y[ii] > 0.5, pa[ii]) - roc_auc_score(y[ii] > 0.5, pb[ii]))
        res[sn] = {**{k: round(v, 4) for k, v in base.items()},
                   "f1_delta_ci95": [round(float(np.percentile(df, q)), 4) for q in (2.5, 97.5)],
                   "auc_delta_ci95": [round(float(np.percentile(da, q)), 4) for q in (2.5, 97.5)]}
        log(sn, res[sn])
    save("vadcmp", res, sub=a.vad)



def stage_cost(a):
    """Served-engine cost per 160 ms chunk on MPS / CPU (mps_115m.stage_engine's protocol: the bundled two-party clip,
    the 0.6B voice print, warm-up, best of 3 runs) for v0.1 and v0.2 back to back, per preset (--fam)."""
    import torch
    import audioforge.serve as S
    import mps_115m as M
    torch.set_num_threads(2)
    pcm = M.load_clip().astype(np.float32) / 32768.0
    pr = json.loads((C.W / "two_party_call_16s.voiceprint_0p6b.json").read_text())
    res = load_json("cost", {})
    for tag in a.tags.split(","):
        eng = engine_cand(C.SERVED_0P6B if tag == "v0.1" else cand_afm(tag), a.device)
        eng.warmup()
        for preset in a.fam.split(","):
            runs = []
            for _ in range(3):
                s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
                s.arm_enrollment("enroll", 0, embedding=pr)
                t0 = time.perf_counter()
                for i in range(0, len(pcm), M.CHUNK):
                    s.process(pcm[i:i + M.CHUNK])
                s.finish()
                runs.append({"rtf": (time.perf_counter() - t0) / (len(pcm) / SR), "chunk_ms": list(s.chunk_ms)})
            best = min(runs, key=lambda r: r["rtf"])
            c = np.array(best["chunk_ms"])
            r = {"chunk_ms_p50": round(float(np.median(c)), 2), "chunk_ms_p95": round(float(np.percentile(c, 95)), 2),
                 "rtf": round(best["rtf"], 4)}
            res.setdefault(f"{tag}_{a.device}", {})[preset] = r
            log(tag, a.device, preset, r)
    save("cost", res)


STAGES = {"evcache": stage_evcache, "evcheck": stage_evcheck, "lag": stage_lag, "swap": stage_swap, "vadtrain": stage_vadtrain, "hoprep": stage_hoprep, "ho115": stage_ho115, "hop6": stage_hop6, "evp6": stage_evp6, "vadlag": stage_vadlag, "vadall": stage_vadall, "teacher": stage_teacher, "segtrain": stage_segtrain, "hoscan": stage_hoscan, "hoquick": stage_hoquick, "quiet": stage_quiet, "build": stage_build, "turntrain": stage_turntrain, "evverify": stage_evverify, "evdump": stage_evdump, "evscore": stage_evscore, "servedcheck": stage_servedcheck, "vadcmp": stage_vadcmp, "cost": stage_cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--which", default="calls")
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vad", default="served")
    ap.add_argument("--tags", default="served")
    ap.add_argument("--block", default="12")
    ap.add_argument("--fam", default="fast,balanced,assistant")
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--st3dips", type=float, default=0.0)
    ap.add_argument("--quiet-train", action="store_true")
    ap.add_argument("--presets-json", default=None)
    ap.add_argument("--rules-json", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--label", default=None)
    ap.add_argument("--ship", action="store_true")
    ap.add_argument("--version", default="0.2")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seg", default="s12")
    ap.add_argument("--turn", default="f1")
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--vblocks", default="12")
    ap.add_argument("--oto", type=int, default=0)
    ap.add_argument("--oto-share", type=float, default=0.35)
    ap.add_argument("--otoq", type=int, default=0)
    ap.add_argument("--offw", type=float, default=1.0)
    ap.add_argument("--crop", type=int, default=32)
    ap.add_argument("--sa", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
