"""research/FIXALL.md: attack every FINAL_COMPARE row where audioforge loses to the best model, head by head, on both
cores (115M heads v0.3, English 0.6B heads v0.2), frozen NVIDIA encoders. -> runs/fixall.json.

Teachers ("use the best model as teacher"): the strongest system of each FINAL_COMPARE row run on OUR training audio
to make soft targets next to the hard labels (cached once under FX/teachers/). Selection on held-out data only; the
evaluation sets of FINAL_COMPARE are touched once per shipped candidate (through final_compare.py, same audio).

Stages (each one process, < 10 min per call, resumable; run through scripts/dev/gate.sh, one heavy job at a time):
  vman                         (--core 0.6b) the VAD window lists shared by both cores -> FX/vman.json
  vteach --set S               TEN VAD + pyannote segmentation-3.0 posteriors on the 80 ms grid of every window of S
  vfeat  --set S --core C      frozen-encoder blocks of S (masked [70,1] forward = 160 ms streaming) -> FX/feats/<core>/
  vtrain --core C --tag T      a VAD head on the cached blocks: hard labels + teacher soft targets; held-out selection
  ...                          (later stages: see STAGES)

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/fixall.py <stage> --core 115m|0.6b [...]
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

import core_0p6b_heads as C  # noqa: E402  (reads --core from sys.argv)

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
FX = SSD / "scratch" / "fixall"
TEACH = FX / "teachers"
OUT = ROOT / "runs" / "fixall.json"
SR, FRAME, HOP = 16000, 0.08, 1280
CORE = C.TAG  # "115m" | "0p6b"
FEATS = FX / "feats" / CORE
HEADS = FX / "heads"
T_START = time.time()
VAD_BLOCKS_CORE = {"115m": [2, 3, 4, 5, 6], "0p6b": [8, 10, 12, 14, 16]}
ICSI_HOLD = C.ICSI_HOLD  # ("Bro026", "Bmr022"): ICSI train meetings held out for selection
AMI_HOLD = C.VAD_HOLD  # ("TS3011b", "ES2015c")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val, sub=None):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if sub is None:
        res[key] = val
    else:
        res.setdefault(key, {})[sub] = val
    OUT.write_text(json.dumps(res, indent=1))


def load_json(key, default=None):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    return res.get(key, default)


def over(a, t0):
    return time.time() - t0 > a.budget


# =========================================================================== 1. speech detector (VAD)
# window sets: ami1200 (the shipped recipe's 1200 seeded AMI train diar windows), icsi600 (600 seeded ICSI train
# diar windows of the 12 cached train meetings; ICSI dev / eval meetings are never in it), oto_tr / otoq_tr (1200
# oto user-channel turn clips of the train conversations, clean / quiet-channel variant), oto_va / otoq_va (the
# turn_v4/v5 held-out conversations), room_tr / room_ho (noise-only negatives). Held-out selection scopes: AMI_HOLD
# windows of ami1200, ICSI_HOLD windows of icsi600, oto_va, otoq_va, room_ho.
VSETS = ("ami1200", "icsi600", "oto_tr", "otoq_tr", "oto_va", "otoq_va", "room_tr", "room_ho")


def stage_vman(a):
    """The oto clip lists exactly as core_0p6b_turn._oto_vad_set draws them (0.6B caches exist for them), so both
    cores and the teachers see the same clips -> FX/vman.json."""
    assert CORE == "0p6b", "vman reads the 0.6B turn caches: run with --core 0.6b"
    import core_0p6b_turn as T
    out = {}
    for name, which, n, seed, quiet in (("oto_tr", "train", 1200, 0, False), ("otoq_tr", "train", 1200, 1, True),
                                        ("oto_va", "val", None, 0, False), ("otoq_va", "val", None, 0, True)):
        import random
        vg = T._val_groups()
        pre = "q_" if quiet else ""
        cl = [c for c in T._turn_manifest() if c["src"] == "oto" and T.blk_path(pre + c["id"]).exists()
              and (((c["src"], c["meeting"]) in vg) == (which == "val"))]
        if n and len(cl) > n:
            cl = random.Random(seed).sample(cl, n)
        out[name] = [dict(c, T=int(len(np.load(T.inp_path(pre + c["id"]))["pu"])), quiet=quiet) for c in cl]
        log(name, len(out[name]))
    FX.mkdir(parents=True, exist_ok=True)
    (FX / "vman.json").write_text(json.dumps(out))


class VSet:
    """Items of a VAD window set: .ids, .meetings, .audio(i), .hard(i, T) ('any part of the 80 ms frame', the
    published label convention), .centre(i, T) (frame centre +-20 ms)."""

    def __init__(self, name):
        self.name = name
        if name == "ami1200":
            import vad_layers as V
            from audioforge.datasets.ami import AMI
            self.val = V.load_set("train", 1200)
            ds = AMI(sorted({v["meeting"] for v in self.val}), verbose=False)
            for v in self.val:
                v["acts"] = ds.acts[v["meeting"]]
        elif name == "icsi600":
            from audioforge.datasets.icsi import ICSI, recipe_data
            from audioforge.train import derive_labels
            self.val = [derive_labels(dict(v), ["vad"]) for v in recipe_data(
                {"data": {"icsi": {"mode": "diar", "n_train": 600, "seed": 0}}}, "train")]
            ds = ICSI(sorted({v["meeting"] for v in self.val}), verbose=False)
            for v in self.val:
                v["acts"] = ds.acts[v["meeting"]]
        elif name.startswith("room"):
            self.val = C.room_tone_items("train" if name == "room_tr" else "held")
        else:
            self.val = json.loads((FX / "vman.json").read_text())[name]
            self._au = None
        self.ids = [str(v.get("id", f"{name}_{i:05d}")) for i, v in enumerate(self.val)]
        self.meetings = [v.get("meeting", "room") for v in self.val]

    def audio(self, i):
        v = self.val[i]
        if "audio" in v:
            return np.asarray(v["audio"], np.float32)
        import core_0p6b_turn as T
        import turn_v4 as V4
        if self._au is None:
            self._au, self._man = V4.ClipAudio(), T._turn_manifest()
        return (T.quiet_audio(v, self._au, self._man)[0] if v["quiet"] else self._au(v, self._man)[0]).astype(np.float32)

    def hard(self, i, T):
        v = self.val[i]
        if "act" in v:  # oto: the user's word activity intervals, any overlap with the frame
            y = np.zeros(T, np.float32)
            t0 = np.arange(T) * FRAME
            for a0, b0 in v["act"]:
                y[(t0 + FRAME > a0) & (t0 < b0)] = 1
            return y
        y = np.asarray(v["vad"], np.float32)[:T]
        return np.concatenate([y, np.zeros(T - len(y), np.float32)])

    def centre(self, i, T):
        v = self.val[i]
        if "act" in v:
            import core_0p6b_turn as Tn
            return Tn._act_labels(v, T)
        c = C.centre_labels(v, T)
        return self.hard(i, T) if c is None else c


def stage_vteach(a):
    """Teacher posteriors on one window set (--set): TEN VAD (16 ms hop, the best ICSI VAD in FINAL_COMPARE) and
    pyannote segmentation-3.0 (speech = 1 - P(no speaker), 10 s windows: offline, which is fine for targets; the best
    AMI VAD), both max-pooled onto the 80 ms label grid exactly as final_compare scores them ->
    TEACH/vad_<set>.npz {<id>|ten, <id>|pya}. Resumable per 50 windows."""
    import torch
    from audioforge.baselines import sd
    torch.set_num_threads(2)
    TEACH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vs = VSet(a.set)
    part = TEACH / f"vad_{a.set}.part.npz"
    fin = TEACH / f"vad_{a.set}.npz"
    if fin.exists():
        log(f"vteach {a.set}: done STAGE_COMPLETE")
        return
    got = dict(np.load(part)) if part.exists() else {}
    os.environ["HF_HUB_OFFLINE"] = "1"
    from pyannote.audio import Inference, Model
    from ten_vad import TenVad
    pm = Model.from_pretrained("pyannote/segmentation-3.0")
    inf = Inference(pm, skip_conversion=True, device=torch.device(a.device),
                    pre_aggregation_hook=lambda s: 1.0 - np.exp(s[..., :1]))
    n_new = 0
    for i, cid in enumerate(vs.ids):
        if f"{cid}|ten" in got:
            continue
        x = vs.audio(i)
        T = len(x) // HOP + 2
        v = TenVad(256, 0.5)
        pcm = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
        p = np.asarray([v.process(pcm[j:j + 256])[0] for j in range(0, len(pcm) - 255, 256)])
        got[f"{cid}|ten"] = sd.pool_probs(p, 256 / SR, T).astype(np.float16)
        swf = inf({"waveform": torch.from_numpy(x)[None], "sample_rate": SR})
        sw = swf.sliding_window
        got[f"{cid}|pya"] = sd.pool_probs(np.asarray(swf.data)[:, 0], sw.step, T, offset=sw.start).astype(np.float16)
        n_new += 1
        if n_new % 50 == 0:
            np.savez(part, **got)
            log(f"vteach {a.set}: {len(got) // 2}/{len(vs.ids)} ({time.time() - t0:.0f} s)")
        if over(a, t0):
            break
    if len(got) // 2 == len(vs.ids):
        np.savez(fin, **got)
        part.unlink(missing_ok=True)
        log(f"vteach {a.set}: {len(vs.ids)} windows STAGE_COMPLETE")
    else:
        np.savez(part, **got)
        log(f"vteach {a.set}: {len(got) // 2}/{len(vs.ids)} (budget)")


def _feat_meta(name):
    p = FEATS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def stage_vfeat(a):
    """Frozen-encoder blocks VAD_BLOCKS_CORE[core] of a window set through the shipped core (masked [70,1] forward,
    = 160 ms streaming; --sa k: SpecAugment view k on the mel input, as the shipped VAD recipe) ->
    FEATS/<set>[_sa<k>].f16 memmap (N, nb, D) + _win.npy + .json {done, n, N, Ts, blocks}. Resumable."""
    import torch
    torch.set_num_threads(2)
    FEATS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    name = a.set + (f"_sa{a.sa}" if a.sa else "")
    meta = _feat_meta(name)
    if meta and meta["done"] == meta["n"]:
        log(f"vfeat {name}: done STAGE_COMPLETE")
        return
    vs = VSet(a.set)
    m = C.load_core(a.device, heads=())
    blocks = VAD_BLOCKS_CORE[CORE]
    if not meta:
        Ts = []
        for i, v in enumerate(vs.val):
            n = len(v["audio"]) if "audio" in v else int(round((v["b"] - v["a"]) * SR))
            Ts.append(int(m.encoder.pre_encode.out_lengths(m.preprocessor.num_frames(torch.tensor(n)))))
        meta = {"done": 0, "n": len(Ts), "N": int(sum(Ts)), "Ts": Ts, "blocks": blocks}
        np.save(FEATS / f"{name}_win.npy", np.repeat(np.arange(len(Ts), dtype=np.int32), Ts))
    off = np.r_[0, np.cumsum(meta["Ts"])]
    X = np.memmap(FEATS / f"{name}.f16", np.float16, "r+" if meta["done"] else "w+", shape=(meta["N"], len(blocks), C.D))
    sa = None
    for i in range(meta["done"], meta["n"], a.batch):
        if a.sa:
            from audioforge.features import SpecAugment
            torch.manual_seed(int(a.sa) * 1000 + i)
            sa = SpecAugment(2, 27, 10, 0.05).train()
        idx = list(range(i, min(i + a.batch, meta["n"])))
        auds = [vs.audio(j) for j in idx]
        outs, L, _ = C.encode_blocks(m, auds, blocks, sa=sa)
        for j, o in zip(idx, outs):
            T = meta["Ts"][j]
            blk = np.stack([o[b][:T] for b in blocks], 1)
            if len(blk) < T:
                blk = np.concatenate([blk, np.repeat(blk[-1:], T - len(blk), 0)])
            X[off[j]: off[j + 1]] = blk
        meta["done"] = idx[-1] + 1
        X.flush()
        (FEATS / f"{name}.json").write_text(json.dumps(meta))
        if over(a, t0):
            log(f"vfeat {name}: {meta['done']}/{meta['n']} (budget)")
            return
    log(f"vfeat {name}: {meta['n']} windows, {meta['N']} frames ({time.time() - t0:.0f} s) STAGE_COMPLETE")


def vdata(name, blocks, sa=0):
    """(X (N, nb, D) memmap or array, win, Ts, hard labels, centre labels, teacher ten, teacher pya, meetings) of a
    window set on this core."""
    fname = name + (f"_sa{sa}" if sa else "")
    meta = _feat_meta(fname)
    assert meta and meta["done"] == meta["n"], f"{fname}: features not cached ({CORE})"
    have = meta["blocks"]
    X = np.memmap(FEATS / f"{fname}.f16", np.float16, "r", shape=(meta["N"], len(have), C.D))
    sel = [have.index(b) for b in blocks]  # read lazily: X[i:j][:, sel]
    lab = FEATS / f"{name}_lab.npz"
    if not lab.exists():
        vs = VSet(name)
        Ts = meta["Ts"]
        z = np.load(TEACH / f"vad_{name}.npz")

        def tcrop(k, i, T):
            t = z[f"{vs.ids[i]}|{k}"].astype(np.float32)[:T]
            return np.concatenate([t, np.zeros(T - len(t), np.float32)])
        np.savez(lab, hard=np.concatenate([vs.hard(i, T) for i, T in enumerate(Ts)]),
                 centre=np.concatenate([vs.centre(i, T) for i, T in enumerate(Ts)]),
                 ten=np.concatenate([tcrop("ten", i, T) for i, T in enumerate(Ts)]),
                 pya=np.concatenate([tcrop("pya", i, T) for i, T in enumerate(Ts)]),
                 meetings=np.array(vs.meetings))
    L = np.load(lab)
    win = np.load(FEATS / f"{name}_win.npy")
    return {"X": X, "sel": sel, "win": win, "Ts": meta["Ts"], "hard": L["hard"], "centre": L["centre"], "ten": L["ten"],
            "pya": L["pya"], "meetings": list(L["meetings"])}


def _vnet(d_in, nb, hidden=64):
    """the served FrameHead shape (Linear(D, 64)-SiLU-Linear(64, 1), stateless) + a softmax block mix when nb > 1:
    maps 1:1 onto heads.vad.net.{0,2} (+ layer_mix.vad) of a served model."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.mix = nn.Parameter(torch.zeros(nb)) if nb > 1 else None

        def forward(self, x):  # (..., nb, D) -> logits (...)
            x = (x * self.mix.softmax(0)[:, None]).sum(-2) if self.mix is not None else x[..., 0, :]
            return self.net(x).squeeze(-1)
    return Net()


def _windows(win):
    idx = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
    return idx, np.r_[idx[1:], len(win)]


def vpredict(net, X, sel, dev, bs=65536):
    import torch
    net.eval()
    out = np.empty(len(X), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.from_numpy(np.asarray(X[i:i + bs][:, sel], np.float32)).to(dev)
            out[i:i + bs] = torch.sigmoid(net(x)).float().cpu().numpy()
    return out


def f1_at(p, y, th=0.5):
    d, t = p > th, y > 0.5
    tp, fp, fn = float((d & t).sum()), float((d & ~t).sum()), float((~d & t).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def vho_sets(blocks):
    """the held-out selection scopes on this core: ami_ho (AMI_HOLD windows of ami1200), icsi_ho (ICSI_HOLD windows
    of icsi600), oto_va, otoq_va, and room_ho (negatives) -> {name: (X, sel, rows or None, hard labels, win)}."""
    out = {}
    for name, src, hold in (("ami_ho", "ami1200", AMI_HOLD), ("icsi_ho", "icsi600", ICSI_HOLD), ("oto_va", "oto_va", None),
                            ("otoq_va", "otoq_va", None), ("room_ho", "room_ho", None)):
        d = vdata(src, blocks)
        if hold:
            m = np.array([d["meetings"][w] in hold for w in d["win"]])
            rows = np.nonzero(m)[0]
            X = np.asarray(d["X"][rows[0]:rows[-1] + 1])[rows - rows[0]]  # the held-out rows (small)
            out[name] = (X, d["sel"], d["hard"][rows], d["win"][rows])
        else:
            out[name] = (d["X"], d["sel"], d["hard"], d["win"])
    return out


def vho_score(pred, H):
    """held-out report: per scope AUC (with room_ho frames as extra negatives) and F1 at 0.5 (AMI / ICSI held-out,
    hard labels = the published convention); 'sel' = mean AUC of the four speech scopes (selection)."""
    from sklearn.metrics import roc_auc_score
    pr = pred["room_ho"]
    r = {}
    for k in ("ami_ho", "icsi_ho", "oto_va", "otoq_va"):
        y = H[k][2]
        r[k] = {"auc": round(float(roc_auc_score(np.r_[y, np.zeros(len(pr))] > 0.5, np.r_[pred[k], pr])), 4),
                "f1": round(f1_at(pred[k], y), 4)}
    r["room_p95"] = round(float(np.percentile(pr, 95)), 4)
    r["sel"] = round(float(np.mean([r[k]["auc"] for k in ("ami_ho", "icsi_ho", "oto_va", "otoq_va")])), 4)
    r["f1_mi"] = round(float(np.mean([r[k]["f1"] for k in ("ami_ho", "icsi_ho")])), 4)
    return r


def stage_vtrain(a):
    """A stateless VAD head (the served FrameHead shape; --vblocks, > 1 = softmax mix) on the cached blocks of this
    core. Crops of 32 frames from: ami1200 minus AMI_HOLD, icsi600 minus ICSI_HOLD, oto_tr, otoq_tr (+ the SpecAugment
    views with --sa), shares --shares; 15 % room-tone crops. Target: --lab hard|centre labels, plus --lam x BCE on the
    teacher posterior --teach ten|pya|mean|max|none (the soft target). Feature dropout 0.1, noise 0.1 std, 2 time
    masks. AdamW 1e-3, cosine, --steps; early stopping on the held-out mean AUC (vho_score 'sel'). Then the output
    bias is calibrated on held-out: the logit shift (-1.5..1.5) that maximises the mean F1 at 0.5 over ami_ho /
    icsi_ho, applied only with --calib. -> HEADS/vad_<core>_<tag>.pt, runs/fixall.json vtrain.<core>.<tag>."""
    import copy
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    dev = a.device
    HEADS.mkdir(parents=True, exist_ok=True)
    blocks = [int(b) for b in a.vblocks.split(",")]
    L = 32
    t0 = time.time()
    shares = {k: float(v) for k, v in (kv.split(":") for kv in a.shares.split(","))}
    srcs = []
    for name, hold in (("ami", AMI_HOLD), ("icsi", ICSI_HOLD), ("oto", None), ("otoq", None)):
        if shares.get(name, 0) <= 0:
            continue
        set_name = {"ami": "ami1200", "icsi": "icsi600", "oto": "oto_tr", "otoq": "otoq_tr"}[name]
        views = [0] + ([1] if a.sa and name in ("ami", "icsi") else [])
        for sa in views:
            d = vdata(set_name, blocks, sa)
            idx, ends = _windows(d["win"])
            keep = np.array([not hold or d["meetings"][d["win"][i]] not in hold for i in idx])
            y = d[a.lab]
            if a.teach == "none":
                s = None
            elif a.teach in ("ten", "pya"):
                s = d[a.teach]
            elif a.teach == "mean":
                s = 0.5 * (d["ten"] + d["pya"])
            else:
                s = np.maximum(d["ten"], d["pya"])
            srcs.append((name, d["X"], d["sel"], y, s, idx[keep], ends[keep], shares[name] / len(views)))
    dr = vdata("room_tr", blocks)
    ridx, rend = _windows(dr["win"])
    room = ("room", dr["X"], dr["sel"], dr["hard"], np.zeros_like(dr["hard"]), ridx, rend, 0.0)
    H = vho_sets(blocks)
    log(f"data {time.time() - t0:.0f} s: " + ", ".join(f"{s_[0]} {len(s_[5])} win" for s_ in srcs))
    sw = np.array([s_[7] for s_ in srcs])

    def crops(src, n, g):
        _, Xs, sel, ys, ss, idx, ends, _ = src
        w = g.integers(0, len(idx), n)
        s0 = idx[w] + (g.random(n) * np.maximum(1, ends[w] - idx[w] - L)).astype(int)
        s0 = np.maximum(idx[w], np.minimum(s0, ends[w] - L))
        xs, yl, sl = [], [], []
        for q, e in zip(s0, ends[w]):
            q1 = min(q + L, e)
            xx, yy = np.asarray(Xs[q:q1][:, sel]), ys[q:q1]
            tt = ss[q:q1] if ss is not None else yy
            if len(xx) < L:
                xx = np.concatenate([xx, np.repeat(xx[-1:], L - len(xx), 0)])
                yy = np.concatenate([yy, np.full(L - len(yy), -1.0, np.float32)])
                tt = np.concatenate([tt, np.zeros(L - len(tt), np.float32)])
            xs.append(xx)
            yl.append(yy)
            sl.append(tt)
        return np.stack(xs), np.stack(yl), np.stack(sl)

    fstd = float(np.asarray(srcs[0][1][:20000][:, srcs[0][2]], np.float32).std())
    net = _vnet(C.D, len(blocks), a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = np.random.default_rng(1000 + a.seed)

    def val():
        pred = {k: vpredict(net, v[0], v[1], dev) for k, v in H.items()}
        net.train()
        return vho_score(pred, H), pred
    B, nr = 64, 10
    best, best_state, bad, hist = -1.0, None, 0, []
    net.train()
    for step in range(1, a.steps + 1):
        which = g.choice(len(srcs), B - nr, p=sw / sw.sum())
        parts = [crops(srcs[k], int((which == k).sum()), g) for k in np.unique(which)] + [crops(room, nr, g)]
        x = torch.from_numpy(np.concatenate([p_[0] for p_ in parts]).astype(np.float32)).to(dev)
        yy = torch.from_numpy(np.concatenate([p_[1] for p_ in parts]).astype(np.float32)).to(dev)
        ss = torch.from_numpy(np.concatenate([p_[2] for p_ in parts]).astype(np.float32)).to(dev)
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
            w = w.masked_fill(msk, 0.0)
        z = net(x)
        loss = F.binary_cross_entropy_with_logits(z, yy, reduction="none")
        if a.teach != "none":
            loss = loss + a.lam * F.binary_cross_entropy_with_logits(z, ss.clamp(0, 1), reduction="none")
        loss = (loss * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            v, _ = val()
            hist.append({"step": step, "loss": round(float(loss), 4), **v})
            log(json.dumps(hist[-1]))
            if v["sel"] > best:
                best, bad, best_state = v["sel"], 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if bad >= a.patience or over(a, t0):
                break
    net.load_state_dict(best_state)
    v, pred = val()
    # bias calibration on held-out (AMI / ICSI held-out F1 at 0.5)
    from scipy.special import logit
    shifts = np.round(np.arange(-1.5, 1.51, 0.1), 2)
    f1s = []
    for b in shifts:
        f1s.append(np.mean([f1_at(1 / (1 + np.exp(-(logit(np.clip(pred[k], 1e-6, 1 - 1e-6)) + b))), H[k][2])
                            for k in ("ami_ho", "icsi_ho")]))
    b_best = float(shifts[int(np.argmax(f1s))])
    r = {"ho": v, "hist": hist, "calib_shift": b_best, "calib_f1_mi": round(float(max(f1s)), 4),
         "args": {k: getattr(a, k) for k in ("hidden", "vblocks", "shares", "lab", "teach", "lam", "sa", "steps", "seed", "calib")},
         "params": sum(p_.numel() for p_ in net.parameters()), "sec": round(time.time() - t0)}
    if a.calib and b_best != 0:
        with torch.no_grad():
            net.net[2].bias += b_best
        pred = {k: vpredict(net, vv[0], vv[1], dev) for k, vv in H.items()}
        r["ho_calib"] = vho_score(pred, H)
    log(a.tag, json.dumps({k: r[k] for k in ("ho", "calib_shift", "calib_f1_mi")} | ({"ho_calib": r["ho_calib"]} if "ho_calib" in r else {})))
    torch.save({"blocks": blocks, "hidden": a.hidden, "state_dict": {k: v_.cpu() for k, v_ in net.state_dict().items()}, "res": r},
               HEADS / f"vad_{CORE}_{a.tag}.pt")
    save(f"vtrain_{CORE}", r, sub=a.tag)


def served_afm():
    """the served model of this core (FINAL_COMPARE's builds): 115M stage1_served_v3.afm, 0.6B served_0p6b_v0.2.afm."""
    if CORE == "115m":
        return ROOT / "runs" / "stage1_served_v3.afm"
    return SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.2.afm"


def stage_vserved(a):
    """The served VAD head of this core (whole-window masked [70,1] forward, as final_compare's vad stage) on the
    held-out scopes, on the same frames as the cached features -> FX/vserved_<core>.npz {scope: p}."""
    import torch
    from audioforge.train import load_model
    torch.set_num_threads(2)
    f = FX / f"vserved_{CORE}.npz"
    m = load_model(str(served_afm()), a.device).eval()
    out = {}
    for name, src, hold in (("ami_ho", "ami1200", AMI_HOLD), ("icsi_ho", "icsi600", ICSI_HOLD), ("oto_va", "oto_va", None),
                            ("otoq_va", "otoq_va", None), ("room_ho", "room_ho", None)):
        vs = VSet(src)
        Ts = _feat_meta(src)["Ts"]
        P = []
        for i in range(len(vs.ids)):
            if hold and vs.meetings[i] not in hold:
                continue
            x = vs.audio(i)
            with torch.inference_mode():
                xx = torch.from_numpy(x)[None].to(a.device)
                enc, elen, hid = m.encode(xx, torch.tensor([xx.shape[1]], device=a.device), [70, 1], return_hidden=True)
                p = m.heads["vad"](m.head_input("vad", enc, hid)).sigmoid()[0, : int(elen[0])].float().cpu().numpy().reshape(-1)
            T = Ts[i]
            P.append(np.concatenate([p, np.zeros(max(0, T - len(p)), np.float32)])[:T])
        out[name] = np.concatenate(P)
        log(name, len(P), len(out[name]))
    np.savez(f, **out)
    blocks = [VAD_BLOCKS_CORE[CORE][0]]
    H = vho_sets(blocks)
    r = vho_score(out, H)
    log("served", json.dumps(r))
    save(f"vserved_{CORE}", r)


# =========================================================================== 5. words: RNNT beam search (decoding only)
WORDS = FX / "words"


def stage_wprep(a):
    """Held-out word sets for the beam selection (never FINAL_COMPARE's evaluation sets): single-speaker segments
    (1-15 s, the final_asr / hybrid_asr protocol) of the held-out ICSI train meetings ICSI_HOLD and AMI train
    meetings AMI_HOLD, random.Random(0).sample(200) -> WORDS/ho_icsi.npz | ho_ami.npz + _refs.json."""
    import random
    from audioforge.datasets.ami import AMI
    from audioforge.datasets.icsi import ICSI
    WORDS.mkdir(parents=True, exist_ok=True)
    for name, ds in (("ho_icsi", lambda: ICSI(list(ICSI_HOLD), verbose=False)), ("ho_ami", lambda: AMI(list(AMI_HOLD), verbose=False))):
        if (WORDS / f"{name}_refs.json").exists():
            continue
        segs = ds().asr(1.0, 15.0)
        idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
        np.savez(WORDS / f"{name}.npz", **{f"a{i}": np.asarray(segs[j]["audio"], np.float32) for i, j in enumerate(idx)})
        (WORDS / f"{name}_refs.json").write_text(json.dumps([segs[j]["text"] for j in idx]))
        log(f"wprep {name}: {len(idx)} of {len(segs)}")


def word_set(name):
    if name.startswith("ho_"):
        z = np.load(WORDS / f"{name}.npz")
        refs = json.loads((WORDS / f"{name}_refs.json").read_text())
        return [z[f"a{i}"] for i in range(len(refs))], refs
    import final_compare as FC
    xs, refs, _ = FC.asr_set(name)
    return xs, refs


def stage_wdec(a):
    """The served core's streaming RNNT (masked [70,1] forward, CPU, as final_compare's core rows) decoded greedy
    and with beam search (--beams k1,k2,..; --max-sym) on --set -> WORDS/<core>_<set>.jsonl rows {i, mode, hyp,
    dec_ms, frames}. Encoder outputs are computed once per item; decode time is the decoder alone."""
    import torch
    from audioforge.train import load_model
    torch.set_num_threads(2)
    WORDS.mkdir(parents=True, exist_ok=True)
    xs, refs = word_set(a.set)
    p = WORDS / f"{CORE}_{a.set}.jsonl"
    done = set()
    if p.exists():
        for ln in p.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["i"], r["mode"]))
    modes = ["greedy"] + [f"beam{k}_s{a.max_sym}" for k in a.beams.split(",") if k]
    m = load_model(str(served_afm()), "cpu").eval()
    head = m.heads["rnnt"]
    t0 = time.time()
    with p.open("a") as f:
        for i, x in enumerate(xs):
            todo = [md for md in modes if (i, md) not in done]
            if not todo:
                continue
            with torch.inference_mode():
                xx, ll = m._pad([np.asarray(x, np.float32)])
                enc, elen, hid = m.encode(xx, ll, [70, 1], return_hidden=True)
                e = m.head_input("rnnt", enc, hid)[0, : int(elen[0])]
                for md in todo:
                    t1 = time.perf_counter()
                    if md == "greedy":
                        ids = head._greedy(e)
                    else:
                        k = int(md[4:].split("_")[0])
                        ids = head.beam_search(e, beam=k, max_sym=a.max_sym)
                    dt = (time.perf_counter() - t1) * 1000
                    f.write(json.dumps({"i": i, "mode": md, "hyp": m.tokenizer.decode(ids), "dec_ms": round(dt, 2),
                                        "frames": int(elen[0])}) + "\n")
            f.flush()
            if over(a, t0):
                log(f"wdec {a.set}: budget at {i}/{len(xs)}")
                return
    log(f"wdec {CORE} {a.set}: STAGE_COMPLETE")


def stage_wscore(a):
    """WER (Whisper English normalizer, final_compare's primary) per mode with 95 % item-bootstrap CIs and the paired
    difference against greedy; decoder ms per 80 ms frame (p50 / p95 per item)."""
    import hybrid_asr as H
    norms, _ = H._normalizers()
    fn = norms["whisper_norm"]
    res = {}
    for set_name in a.set.split(","):
        _, refs = word_set(set_name)
        rows = {}
        for ln in (WORDS / f"{CORE}_{set_name}.jsonl").read_text().splitlines():
            r = json.loads(ln)
            rows.setdefault(r["mode"], {})[r["i"]] = r
        n = len(refs)
        rr = [fn(x) for x in refs]
        E, o = {}, {}
        rng = np.random.default_rng(0)
        bi = rng.integers(0, n, (1000, n))
        for md, R in rows.items():
            if len(R) < n:
                continue
            E[md] = np.array([H.edits_words(rr[i], fn(R[i]["hyp"])) for i in range(n)], np.float64)
            dr = E[md][bi].sum(1)
            dr = dr[:, 0] / dr[:, 1]
            per_f = np.array([R[i]["dec_ms"] / max(R[i]["frames"], 1) for i in range(n)])
            o[md] = {"wer_pct": round(100 * E[md][:, 0].sum() / E[md][:, 1].sum(), 2),
                     "ci95": [round(100 * float(np.percentile(dr, q)), 2) for q in (2.5, 97.5)],
                     "dec_ms_per_frame_mean": round(float(per_f.mean()), 3)}
        for md in E:
            if md != "greedy" and "greedy" in E:
                d = E[md][bi].sum(1)
                g = E["greedy"][bi].sum(1)
                dd = d[:, 0] / d[:, 1] - g[:, 0] / g[:, 1]
                o[md]["delta_vs_greedy_pp"] = round(o[md]["wer_pct"] - o["greedy"]["wer_pct"], 2)
                o[md]["delta_ci95"] = [round(100 * float(np.percentile(dd, q)), 2) for q in (2.5, 97.5)]
        res[set_name] = o
        log(CORE, set_name, json.dumps(o))
    save(f"words_{CORE}", res)


# =========================================================================== 3. language ID: Whisper large-v3 teacher
def lid_rows(set_):
    """(rows, audio_offset per row) of a lid_fix training set: the 'on' view starts at the speech onset - 0.1 s
    (lid_fix shards' audio_offset), capped at 20 s, as the cached head-training frames."""
    import lid_fix as LF
    rows = LF.set_rows(set_)
    off = {}
    for f in sorted(LF.shard_dir(set_, "on").glob("s[0-9]*.json")):
        m = json.loads(f.read_text())
        off.update(zip(m["ids"], m["audio_offset"]))
    return rows, off


def stage_lteach(a):
    """Whisper large-v3 language posteriors (the best LID row of FINAL_COMPARE; fp16 on MPS, the language-token
    distribution after <|startoftranscript|>, restricted to the 17 languages exactly as final_compare scores it) on
    a lid_fix training set (--set train|trainx|extra_en), two views per row: the first 2 s of the 'on' view and the
    whole 'on' view (<= 20 s) -> TEACH/lid_whisper_<set>.npz {ids, p2 (N, 17), pf (N, 17)}. Resumable."""
    import torch
    import lid as L
    import lid_data as LD
    from audioforge.baselines.lid import restrict
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    import final_compare as FC
    torch.set_num_threads(2)
    TEACH.mkdir(parents=True, exist_ok=True)
    fin = TEACH / f"lid_whisper_{a.set}.npz"
    part = TEACH / f"lid_whisper_{a.set}.part.npz"
    if fin.exists():
        log(f"lteach {a.set}: done STAGE_COMPLETE")
        return
    rows, off = lid_rows(a.set)
    rows = [r for r in rows if r["id"] in off]
    n = len(rows)
    P = dict(np.load(part)) if part.exists() else {"p2": np.zeros((n, 17), np.float32), "pf": np.zeros((n, 17), np.float32),
                                                   "n": np.array(0)}
    dev, dt = a.device, torch.float16
    proc = WhisperProcessor.from_pretrained(FC.snap("openai/whisper-large-v3"))
    model = WhisperForConditionalGeneration.from_pretrained(FC.snap("openai/whisper-large-v3"), dtype=dt).to(dev).eval()
    tok = proc.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    codes = list(LANGUAGES)
    ids = [tok.convert_tokens_to_ids(f"<|{c}|>") for c in codes]

    def probs(clips):
        feats = proc.feature_extractor([np.asarray(c, np.float32) for c in clips], sampling_rate=SR,
                                       return_tensors="pt").input_features.to(dev, dt)
        with torch.inference_mode():
            lg = model(input_features=feats, decoder_input_ids=torch.full((len(clips), 1), sot, device=dev)).logits[:, -1]
            pr = lg[:, ids].float().softmax(-1).cpu().numpy()
        return np.stack([restrict(dict(zip(codes, row.tolist())), L.CODES) for row in pr]).astype(np.float32)
    t0 = time.time()
    i = int(P["n"])
    bs = 16
    while i < n and not over(a, t0):
        rs = rows[i:i + bs]
        aud = []
        for r in rs:
            x = LD.load_audio(r)
            o = int(off[r["id"]])
            aud.append(np.asarray(x[o: o + 20 * SR], np.float32))
        P["p2"][i:i + len(rs)] = probs([x[: 2 * SR] for x in aud])
        P["pf"][i:i + len(rs)] = probs(aud)
        i += len(rs)
    if i < n:
        np.savez(part, p2=P["p2"], pf=P["pf"], n=np.array(i))
        log(f"lteach {a.set}: {i}/{n} ({time.time() - t0:.0f} s)")
        return
    y = np.array([L.CODES.index(r["lang"]) for r in rows])
    np.savez(fin, ids=np.array([r["id"] for r in rows]), p2=P["p2"], pf=P["pf"], y=y)
    part.unlink(missing_ok=True)
    log(f"lteach {a.set}: {n} rows; teacher acc 2s {np.mean(P['p2'].argmax(1) == y):.4f} full {np.mean(P['pf'].argmax(1) == y):.4f} STAGE_COMPLETE")


NEMO_BASE = {"115m": ROOT / "data/nemo/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
             "0p6b": ROOT / "data/nemo/nemotron-speech-streaming-en-0.6b.nemo"}


def build_cand(name, speech=None, base_afm=None, presets=None, extra=None):
    """A served candidate of this core: the served afm (or base_afm) + heads.speech = the VAD head
    HEADS/vad_<core>_<speech>.pt (type frame, its blocks as from_layers, a learned mix when > 1 block; the client's
    speech probability, research/FIXALL.md step 1) + optional cfg turn_presets; every other tensor unchanged
    (checked) -> FX/<name>.afm. extra(cfg, sd) may edit further heads."""
    import copy
    import torch
    from audioforge import hub
    from audioforge.model import SpeechModel
    from audioforge.train import load_model, save_model
    base = load_model(str(base_afm or served_afm()), "cpu")
    cfg = copy.deepcopy(base.cfg)
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    changed = []
    if speech:
        ck = torch.load(HEADS / f"vad_{CORE}_{speech}.pt", map_location="cpu", weights_only=False)
        bl = [int(b) - 1 for b in ck["blocks"]]
        st = ck["state_dict"]
        cfg["heads"]["speech"] = {"type": "frame", "key": "speech", "hidden": int(ck.get("hidden", 64)),
                                  "from_layers": bl, "weight": 0.0}
        sd = {k: v for k, v in sd.items() if not k.startswith("heads.speech.") and k != "layer_mix.speech"}
        sd.update({f"heads.speech.{k}": v for k, v in st.items() if k != "mix"})
        if len(bl) > 1:
            sd["layer_mix.speech"] = st["mix"]
        changed.append(f"speech={speech}")
    if presets is not None:
        cfg["turn_presets"] = presets
        changed.append("turn_presets")
    if extra is not None:
        changed += extra(cfg, sd)
    cfg["name"] = name
    m = SpeechModel(cfg, base.tokenizer)
    m.load_state_dict(sd, strict=True)
    out = FX / f"{name}.afm"
    save_model(m.eval(), out)
    back = load_model(str(out), "cpu")
    assert hub.state_hash(back.state_dict()) == hub.state_hash(m.state_dict())
    bsd, osd = back.state_dict(), base.state_dict()
    same = [k for k in osd if k in sd and torch.equal(osd[k], sd[k])]
    assert all(torch.equal(osd[k], bsd[k]) for k in same), "a kept tensor changed"
    log(f"build {out}: {changed}; {len(same)} of {len(osd)} base tensors identical")
    return out, changed


def stage_vbuild(a):
    """--tag NAME --speech TAG [--ship --version V]: build_cand; with --ship also the heads asset
    assets/served_heads[_0p6b]_v<V>.pt (hub.export_heads against the NVIDIA .nemo) + a rebuild check."""
    from audioforge import hub
    name = a.tag
    presets = json.loads(a.presets_json) if a.presets_json else None
    out, changed = build_cand(name, a.speech or None, presets=presets)
    rec = {"afm": str(out), "changed": changed}
    if a.ship:
        heads = ROOT / "assets" / (f"served_heads_v{a.version}.pt" if CORE == "115m" else f"served_heads_0p6b_v{a.version}.pt")
        info = hub.export_heads(out, NEMO_BASE[CORE], heads, base_key="asr" if CORE == "115m" else "asr_0p6b")
        got = hub.build_served(NEMO_BASE[CORE], heads, FX / "rebuilt.afm")
        assert got == info["state_hash"], (got, info)
        (FX / "rebuilt.afm").unlink()
        rec.update({**info, "heads": str(heads), "size": heads.stat().st_size, "sha256": hub.sha256_file(heads)})
    log(json.dumps(rec))
    save(f"build_{CORE}", rec, sub=name)


def _engine(afm, device):
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    if CORE == "115m":
        opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ROOT / "assets" / "tsvad_spk.pt")}
    else:
        lid = ROOT / "assets" / "lid_0p6b.pt"
        opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ROOT / "assets" / "tsvad_0p6b.pt"),
                "lid": str(lid) if lid.exists() else None}
    return Engine.load(str(afm), None, device, threads=2, **opts)


def stage_servedeq(a):
    """Served check of a build (--new AFM) against the served one (--old AFM, default served_afm()): the --mode single
    engine on the bundled call (examples/audio/two_party_call_16s.wav + its stored voice print), every preset
    (balanced / fast / assistant): identical turn_end events and identical per-frame turn inputs (heads.vad), and
    the client's frame 'vad' field (heads.speech when present) -> runs/fixall.json servedeq.<core>.<tag>."""
    import torch
    import audioforge.serve as S
    import eot_latency as E
    from audioforge.data import load_wav
    torch.set_num_threads(2)
    wav = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
    x = load_wav(str(wav), SR).astype(np.float32)
    vp = (ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json") if CORE == "115m" else \
        (SSD / "scratch" / "core_0p6b" / "two_party_call_16s.voiceprint_0p6b.json")
    emb = json.loads(vp.read_text())
    res = {}
    engs = {"old": _engine(a.old or served_afm(), a.device), "new": _engine(a.new, a.device)}
    for e in engs.values():
        e.warmup()
    for preset in ("balanced", "fast", "assistant"):
        out = {}
        for k, eng in engs.items():
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
            s.arm_enrollment("enroll", 0, embedding=emb)
            msgs = []
            for i in range(0, len(x), 320):
                msgs += s.process(x[i:i + 320])
            msgs += s.finish()
            fr = [m for m in msgs if m["type"] == "frame"] + [it for m in msgs if m["type"] == "frames" for it in m["items"]]
            out[k] = {"turn_ends": [round(m["t"], 3) for m in msgs if m["type"] == "turn_end"],
                      "vad_frames": [round(float(v), 5) for v in s.asr.vad_ring.d] if hasattr(s.asr, "vad_ring") else None,
                      "client_vad": [m["vad"] for m in fr]}
        same_turn = out["old"]["turn_ends"] == out["new"]["turn_ends"]
        res[preset] = {"turn_ends_old": out["old"]["turn_ends"], "turn_ends_new": out["new"]["turn_ends"],
                       "turn_ends_equal": same_turn,
                       "client_vad_changed_frames": int(sum(abs(p - q) > 1e-3 for p, q in zip(out["old"]["client_vad"], out["new"]["client_vad"]))),
                       "n_frames": len(out["new"]["client_vad"])}
        log(preset, json.dumps(res[preset]))
    save(f"servedeq_{CORE}", res, sub=a.tag)


# =========================================================================== 2. turn taking (0.6B): Q heads + two thresholds
TURN = FX / "turn"
RTS = (None, 0.15, 0.25)  # reset_thr options (two-threshold silence clock); None = one threshold


def _T():
    assert CORE == "0p6b", "turn stages: --core 0.6b"
    import core_0p6b_turn as T
    return T


def stage_texport(a):
    """A fixall VAD head (HEADS/vad_0p6b_<--speech>.pt, single block) as a core_0p6b_turn VAD candidate
    (TF/heads/vad_fx_<tag>.pt, arch mlp: inp / out), so its hop6 / evp6 / vadall / segtrain / build stages read it."""
    import torch
    T = _T()
    ck = torch.load(HEADS / f"vad_0p6b_{a.speech}.pt", map_location="cpu", weights_only=False)
    st = ck["state_dict"]
    assert "mix" not in st and len(ck["blocks"]) == 1, "turn VAD candidates read one cached block (8 / 12 / 24)"
    out = {"inp.weight": st["net.0.weight"], "inp.bias": st["net.0.bias"], "out.weight": st["net.2.weight"],
           "out.bias": st["net.2.bias"]}
    torch.save({"arch": "mlp", "blocks": ck["blocks"], "state_dict": out, "eval": ck["res"]["ho"]},
               T.HEADS / f"vad_fx_{a.tag}.pt")
    log(f"-> {T.HEADS / f'vad_fx_{a.tag}.pt'}")


def tgrid(fam):
    T = _T()
    G = []
    for r in T.rule_grid(fam):
        thr = min(r.get("vad_thr", 0.4), r.get("mvt", 1.0))
        for rt in RTS:
            if rt is None or rt < thr:
                G.append({**r, "rt": rt})
    return G


def stage_tscan(a):
    """Every rule of tgrid(--fam) (core_0p6b_turn's grid x reset_thr RTS) on the held-out end-of-turn set for one
    0.6B candidate (--tags combo, a TF/ho/p6_<combo>.npz) -> TURN/scan_<combo>_<fam>.json. Held-out only."""
    T = _T()
    TURN.mkdir(parents=True, exist_ok=True)
    meta = T.ho_meta()
    sig = T.load_ho(f"p6_{a.tags}.npz")
    t0 = time.time()
    for fam in a.fam.split(","):
        f = TURN / f"scan_{a.tags}_{fam}.json"
        rows = json.loads(f.read_text()) if f.exists() else []
        G = tgrid(fam)
        gc = {}
        for r in G[len(rows):]:
            rows.append({"rule": r, **T.ho_score(sig, meta, r, T.COMP["0p6b"], gc)})
            if over(a, t0):
                break
        f.write_text(json.dumps(rows))
        log(f"tscan {a.tags} {fam}: {len(rows)}/{len(G)} ({time.time() - t0:.0f} s)" + (" STAGE_COMPLETE" if len(rows) == len(G) else ""))
        if len(rows) < len(G):
            return


def _nonreg_ok(r, b, fam):
    """r no worse than b on every held-out scope (fast / balanced: calls, quiet, meet FI and missed; assistant: FF and p50)."""
    if fam == "assistant":
        return r["asst"]["p50"] is not None and r["asst"]["p50"] <= b["asst"]["p50"] and \
            r["asst"]["false_fire_pct"] <= b["asst"]["false_fire_pct"] and r["asst"]["accuracy_pct"] >= b["asst"]["accuracy_pct"]
    for sc in ("calls", "quiet", "meet"):
        if r[sc]["eot_total_ms_p50"] is None or r[sc]["false_interruption_pct"] > b[sc]["false_interruption_pct"] \
                or r[sc]["missed_pct"] > b[sc]["missed_pct"]:
            return False
    return True


def stage_tpick(a):
    """Held-out picks per preset for candidates --tags (comma list of combos): the fastest rule (assistant: the most
    accurate, ties faster) that is no worse than the SERVED 0.6B v0.2 (its own heads and rules on the same held-out
    clips) on every scope (_nonreg_ok); and, for reference, core_0p6b_turn's pick against the 115M's held-out
    numbers. -> runs/fixall.json tpick.<combo>.<fam>."""
    T = _T()
    meta = T.ho_meta()
    v02 = T.served_rules(SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.2.afm")
    sig02 = T.load_ho("p6_served__s12__f1.npz")
    base = {fam: T.ho_score(sig02, meta, v02[fam], T.COMP["0p6b"]) for fam in ("balanced", "fast", "assistant")}
    bars115 = T.load_json("ho_bars")
    res = load_json("tpick", {})
    res["v0.2_heldout"] = {"rules": v02, "res": base}
    for fam in base:
        log(f"v0.2 held-out {fam}: {T.short(base[fam])}")
    for combo in a.tags.split(","):
        for fam in a.fam.split(","):
            f = TURN / f"scan_{combo}_{fam}.json"
            if not f.exists():
                continue
            rows = json.loads(f.read_text())
            ok = [r for r in rows if _nonreg_ok(r, base[fam], fam)]
            if fam == "assistant":
                pk = max(ok, key=lambda r: (r["asst"]["accuracy_pct"], -r["asst"]["p50"])) if ok else None
            else:
                pk = min(ok, key=lambda r: (r["calls"]["eot_total_ms_p50"] + r["quiet"]["eot_total_ms_p50"],
                                            r["calls"]["false_interruption_pct"])) if ok else None
            p115, n115 = T.pick(rows, fam, bars115)
            res.setdefault(combo, {})[fam] = {"n_rules": len(rows), "n_nonreg_v02": len(ok), "pick_nonreg_v02": pk,
                                              "n_ok_115m_bars": n115, "pick_115m_bars": p115}
            log(f"{combo} {fam}: {len(ok)}/{len(rows)} no worse than v0.2 | pick {T.short(pk) if pk else None} "
                f"{pk['rule'] if pk else ''}")
    save("tpick", res)


def stage_ltrain(a):
    """The 115M LID head (lid_fix.stage_train's recipe: anytime CE + T^2 KL at the window's last frame, class-balanced
    batches, train + trainx + extra English, blocks --lblocks) with an ensemble teacher: per training window the
    KL target is --wmix x Whisper large-v3 (TEACH/lid_whisper_<set>.npz: the 2 s view for windows starting at the
    onset and <= 3 s, else the whole-row view) + (1 - --wmix) x AmberNet (lid_fix's per-window logits); windows
    without an AmberNet target (the utterance from the onset) get Whisper's whole-row view. Selection on FLEURS dev
    (lid_fix's step choice); the test / EdAcc caches are NOT read (dev stands in for them in the saved record) ->
    SSD runs/lid_fix/<tag>.pt, runs/fixall.json ltrain.<tag>."""
    import lid_fix as LF
    from scipy.special import softmax
    W_ = {}
    for set_ in ("train", "trainx"):
        f = TEACH / f"lid_whisper_{set_}.npz"
        if f.exists():
            z = np.load(f)
            W_.update({rid: (p2, pf) for rid, p2, pf in zip(z["ids"], z["p2"], z["pf"])})
    log(f"whisper teacher rows: {len(W_)}")
    orig_bw, orig_ev, orig_merge = LF.build_windows, LF.EvalCache, LF.merge_fix

    def bw(sets, k_rand, need_teacher):
        Wn, Tn = orig_bw(sets, k_rand, need_teacher)
        if a.wmix <= 0:
            return Wn, Tn
        sh = {s_: LF.Shards(s_, "on") for s_ in sets}
        n_w = 0
        for j, (s_, si, i, f0, f1, y) in enumerate(Wn):
            rid = sh[s_].metas[si]["ids"][i]
            if rid not in W_:
                continue
            p2, pf = W_[rid]
            onset = sh[s_].metas[si]["onset"][i]
            pw = p2 if (f0 <= onset + 1 and f1 - f0 <= 38) else pf
            pw = np.clip(pw.astype(np.float64), 1e-6, 1)
            if Tn[j] is None:
                ens = pw
            else:
                ens = a.wmix * pw + (1 - a.wmix) * softmax(Tn[j].astype(np.float64))
            Tn[j] = np.log(ens / ens.sum()).astype(np.float32)
            n_w += 1
        log(f"ensemble teacher on {n_w} of {len(Wn)} windows (wmix {a.wmix})")
        return Wn, Tn

    def ev(set_, layers):
        return orig_ev("dev" if set_ in ("test", "edacc") else set_, layers)

    def merge(key, rec):
        rec = dict(rec)
        rec["dev_as_cached_test"] = rec.pop("cached_test", None)  # the evaluation set was not read here
        save("ltrain", rec, sub=a.tag)
    LF.build_windows, LF.EvalCache, LF.merge_fix = bw, ev, merge
    la = argparse.Namespace(blocks=a.lblocks, device=a.device, tag=a.tag, steps=a.steps, batch_size=128, lr=2e-3,
                            wd=0.01, hidden=a.hidden, context=0, rnn=0, feat_noise=0.0, dropout=0.2, alpha=a.lam,
                            temp=2.0, trainx=True, extra_en=True, extra_frac=0.5, aug=False, aug_p=0.25, full_p=0.35,
                            crop=False, frame_drop=0.1, n_per_lang=0, k_rand=1, eval_every=250, seed=a.seed,
                            segment=a.budget)
    LF.stage_train(la)


STAGES = {"vman": stage_vman, "vteach": stage_vteach, "vfeat": stage_vfeat, "vtrain": stage_vtrain, "vserved": stage_vserved, "wprep": stage_wprep, "wdec": stage_wdec, "wscore": stage_wscore, "lteach": stage_lteach, "vbuild": stage_vbuild, "servedeq": stage_servedeq, "texport": stage_texport, "tscan": stage_tscan, "tpick": stage_tpick, "ltrain": stage_ltrain}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--core", default="0.6b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--set", default="ami1200")
    ap.add_argument("--sa", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vblocks", default="4")
    ap.add_argument("--shares", default="ami:0.35,icsi:0.25,oto:0.15,otoq:0.1")
    ap.add_argument("--lab", default="hard", choices=["hard", "centre"])
    ap.add_argument("--teach", default="mean", choices=["none", "ten", "pya", "mean", "max"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--speech", default="")
    ap.add_argument("--ship", action="store_true")
    ap.add_argument("--version", default="")
    ap.add_argument("--presets-json", default=None)
    ap.add_argument("--old", default="")
    ap.add_argument("--tags", default="")
    ap.add_argument("--wmix", type=float, default=0.5)
    ap.add_argument("--lblocks", default="8-12")
    ap.add_argument("--fam", default="balanced,fast,assistant")
    ap.add_argument("--new", default="")
    ap.add_argument("--beams", default="4,8")
    ap.add_argument("--max-sym", type=int, default=3)
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
