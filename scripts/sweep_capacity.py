# /// script
# requires-python = ">=3.10"
# dependencies = ["audioforge", "numpy>=1.24", "torch>=2.1"]
#
# [tool.uv.sources]
# audioforge = { path = "..", editable = true }
# ///
"""Capacity sweep for one head (docs/PROJECT.md "Sizing"): every candidate size trains for the same wall-clock
budget on cached frozen-encoder features, the held-out eval loss is reported against the parameter count, and the
knee is picked by a stated rule. The table goes to plans/sweeps/<head>_<YYYY-MM-DD>.md, the raw numbers to
runs/sweeps/<head>_<YYYY-MM-DD>.json.

    uv run scripts/sweep_capacity.py speech --core 115m --sizes 16,32,64,128,256 --budget 120
    uv run scripts/sweep_capacity.py speech --core 0p6b --grid hidden=32,64,128 depth=1,2 --budget 60

Long sweeps go to the background through the machine gate, with the log readable at any moment:

    scripts/dev/gate.sh uv run scripts/sweep_capacity.py speech --core 115m --sizes 16,32,64,128,256 --budget 120 \\
        2>&1 | tee runs/sweeps/speech_115m/log.txt &

Knee rule: L_min = the lowest best-eval-loss of the sweep; the pick is the candidate with the fewest parameters whose
best eval loss is <= L_min * (1 + --tol) (default tol 0.01, i.e. within 1 % of the best). A pick at the largest
candidate means the sweep did not bracket the knee (extend it upward); at the smallest, extend it downward.

Fairness: each candidate gets --budget seconds of training time (evaluation time is not counted), the learning rate
follows a cosine over that budget, the data order and initialisation share one seed, and the reported loss is the
best of the periodic evaluations (the same rule as a run's `best` checkpoint: it moves only when eval loss improves).

Heads (the registry below; add one by writing a Spec):
  speech   the client's speech detector: the served FrameHead (Linear(D, hidden)-SiLU-Linear(hidden, 1)) on a learned
           softmax mix of encoder blocks (115M blocks 2-6, 0.6B blocks 8-16, 1-based), trained on the cached features
           of the speech-detector feature step (SSD scratch/fixall/feats/<core>/) with that step's sources, shares and
           augmentation. Eval = held-out AMI meetings TS3011b / ES2015c, ICSI Bro026 / Bmr022, oto / quiet-oto
           held-out conversations and held-out room tone (never the evaluation sets). Size keys: hidden (the served
           FrameHead), depth (> 1 stacks extra Linear-SiLU layers: research only, the served FrameHead has one).
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRATCH = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- data
@dataclass
class Data:
    """Training sources (each: X (N, nb, D) fp16 in RAM, labels (N,), window starts / ends, share) and held-out eval
    scopes (name -> (X, labels)); every scope counts equally in the eval loss."""
    train: list
    eval: dict
    d_in: int
    nb: int
    desc: str
    fstd: float = 1.0


def _windows(win):
    idx = np.nonzero(np.r_[True, win[1:] != win[:-1]])[0]
    return idx, np.r_[idx[1:], len(win)]


def _fix_set(feats: Path, name: str, blocks: list[int]):
    meta = json.loads((feats / f"{name}.json").read_text())
    assert meta["done"] == meta["n"], f"{feats / name}: features not fully cached"
    N, nb = meta["N"], len(meta["blocks"])
    D = (feats / f"{name}.f16").stat().st_size // (2 * N * nb)  # encoder width: 512 (115M) / 1024 (0.6B)
    X = np.memmap(feats / f"{name}.f16", np.float16, "r", shape=(N, nb, D))
    sel = [meta["blocks"].index(b) for b in blocks]
    base = name.split("_sa")[0]
    lab = np.load(feats / f"{base}_lab.npz")
    win = np.load(feats / f"{name}_win.npy")
    return X, sel, lab["hard"].astype(np.float32), win, list(lab["meetings"])


def _take(X, sel, rows_lo_hi):
    """Read [lo, hi) row ranges of a memmap (sorted, contiguous reads) into one in-RAM fp16 array."""
    return np.concatenate([np.asarray(X[lo:hi])[:, sel] for lo, hi in rows_lo_hi]) if rows_lo_hi else None


def load_speech(a) -> Data:
    core = a.core
    feats = SCRATCH / "fixall" / "feats" / core
    blocks = [int(b) for b in (a.blocks or {"115m": "2,3,4,5,6", "0p6b": "8,10,12,14,16"}[core]).split(",")]
    ami_hold, icsi_hold = ("TS3011b", "ES2015c"), ("Bro026", "Bmr022")
    g = np.random.default_rng(a.seed)
    # FIXALL step 1 shares (AMI 0.35 / ICSI 0.25 / oto 0.15 / quiet oto 0.10 / room tone 0.15), SpecAugment view of
    # AMI and ICSI as half of their share
    plan = [("ami1200", ami_hold, 0.175), ("ami1200_sa1", ami_hold, 0.175), ("icsi600", icsi_hold, 0.125),
            ("icsi600_sa1", icsi_hold, 0.125), ("oto_tr", None, 0.15), ("otoq_tr", None, 0.10), ("room_tr", None, 0.15)]
    train, t0 = [], time.time()
    for name, hold, share in plan:
        X, sel, y, win, meets = _fix_set(feats, name, blocks)
        idx, ends = _windows(win)
        keep = [k for k in range(len(idx)) if not hold or meets[win[idx[k]]] not in hold]
        n_win = max(1, int(round(a.train_frames * share / max(1, np.mean(ends - idx)))))
        pick = np.sort(g.choice(keep, min(n_win, len(keep)), replace=False))
        rng = [(int(idx[k]), int(ends[k])) for k in pick]
        Xs = _take(X, sel, rng)
        ys = np.concatenate([y[lo:hi] for lo, hi in rng])
        lens = np.array([hi - lo for lo, hi in rng])
        st = np.r_[0, np.cumsum(lens)[:-1]]
        train.append((name, Xs, ys, st, st + lens, share))
        log(f"train {name}: {len(pick)} windows, {len(ys)} frames")
    ev = {}
    for scope, name, hold, cap in (("ami_ho", "ami1200", ami_hold, None), ("icsi_ho", "icsi600", icsi_hold, None),
                                   ("oto_va", "oto_va", None, a.eval_windows), ("otoq_va", "otoq_va", None, a.eval_windows),
                                   ("room_ho", "room_ho", None, None)):
        X, sel, y, win, meets = _fix_set(feats, name, blocks)
        idx, ends = _windows(win)
        ks = [k for k in range(len(idx)) if not hold or meets[win[idx[k]]] in hold]
        if cap and len(ks) > cap:
            ks = sorted(np.random.default_rng(12345).choice(ks, cap, replace=False).tolist())
        rng = [(int(idx[k]), int(ends[k])) for k in ks]
        ev[scope] = (_take(X, sel, rng), np.concatenate([y[lo:hi] for lo, hi in rng]))
        log(f"eval {scope}: {len(ks)} windows, {len(ev[scope][1])} frames")
    fstd = float(train[0][1][:20000].astype(np.float32).std())
    d_in = train[0][1].shape[-1]
    desc = (f"cached frozen-encoder features {feats} (FIXALL step 1), blocks {blocks} (1-based), "
            f"train {sum(len(t[2]) for t in train)} frames sampled with seed {a.seed}; eval scopes "
            + ", ".join(f"{k} {len(v[1])}" for k, v in ev.items()) + f" frames; loaded in {time.time() - t0:.0f} s")
    return Data(train, ev, d_in, len(blocks), desc, fstd)


# --------------------------------------------------------------------------- models
class Mixed(nn.Module):
    """A learned softmax mix over the cached blocks (the served model's layer_mix) in front of a per-frame head."""

    def __init__(self, head: nn.Module, nb: int):
        super().__init__()
        self.head = head
        self.mix = nn.Parameter(torch.zeros(nb)) if nb > 1 else None

    def forward(self, x):  # (..., nb, D) -> logits (...)
        x = (x * self.mix.softmax(0)[:, None]).sum(-2) if self.mix is not None else x[..., 0, :]
        return self.head(x)


def build_speech(size: dict, d_in: int, nb: int) -> nn.Module:
    hidden, depth = int(size.get("hidden", 64)), int(size.get("depth", 1))
    if depth == 1:
        from audioforge.heads.audio import FrameHead  # the served class itself
        return Mixed(FrameHead(d_in, key="speech", hidden=hidden), nb)
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    net = nn.Sequential(*layers, nn.Linear(d, 1))
    return Mixed(_Squeeze(net), nb)


class _Squeeze(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).squeeze(-1)


def frame_bce(z, y, w):
    return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum().clamp(min=1)


@dataclass
class Spec:
    load: Callable[[argparse.Namespace], Data]
    build: Callable[[dict, int, int], nn.Module]
    size_key: str                # what --sizes sets
    shipped: dict                # the shipped size per core (what the sweep is checked against)
    shipped_src: str
    crop: int = 32
    batch: int = 64
    lr: float = 1e-3
    extra: dict = field(default_factory=dict)


HEADS = {
    "speech": Spec(load_speech, build_speech, "hidden", {"115m": {"hidden": 64, "depth": 1}, "0p6b": {"hidden": 64, "depth": 1}},
                   "assets/served_heads_v0.4.pt / served_heads_0p6b_v0.4.pt heads.speech (type frame, hidden 64)"),
}


# --------------------------------------------------------------------------- train / eval
def auc(p, y):
    """ROC AUC by ranks (ties averaged)."""
    y = y > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(net, data: Data, dev, bs=32768):
    net.eval()
    out, preds = {}, {}
    for k, (X, y) in data.eval.items():
        zs = [net(torch.from_numpy(X[i:i + bs].astype(np.float32)).to(dev)).float().cpu() for i in range(0, len(X), bs)]
        z = torch.cat(zs)
        out[k] = float(F.binary_cross_entropy_with_logits(z, torch.from_numpy(y)))
        preds[k] = torch.sigmoid(z).numpy()
    net.train()
    loss = float(np.mean(list(out.values())))
    pr = preds["room_ho"]
    aucs = [auc(np.r_[preds[k], pr], np.r_[data.eval[k][1], np.zeros(len(pr), np.float32)])
            for k in data.eval if k != "room_ho"]
    return {"eval_loss": round(loss, 5), "per_scope": {k: round(v, 5) for k, v in out.items()},
            "sel_auc": round(float(np.mean(aucs)), 4), "room_p95": round(float(np.percentile(pr, 95)), 4)}


def crops(src, n, L, g):
    _, X, y, st, en, _ = src
    w = g.integers(0, len(st), n)
    s0 = st[w] + (g.random(n) * np.maximum(1, en[w] - st[w] - L)).astype(int)
    s0 = np.maximum(st[w], np.minimum(s0, en[w] - L))
    xs, ys = [], []
    for q, e in zip(s0, en[w]):
        q1 = min(q + L, e)
        xx, yy = X[q:q1], y[q:q1]
        if len(xx) < L:
            xx = np.concatenate([xx, np.repeat(xx[-1:], L - len(xx), 0)])
            yy = np.concatenate([yy, np.full(L - len(yy), -1.0, np.float32)])
        xs.append(xx)
        ys.append(yy)
    return np.stack(xs), np.stack(ys)


def train_one(spec: Spec, size: dict, data: Data, a, dev) -> dict:
    torch.manual_seed(a.seed)
    g = np.random.default_rng(1000 + a.seed)
    net = spec.build(size, data.d_in, data.nb).to(dev)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=spec.lr, weight_decay=1e-3)
    shares = np.array([s[5] for s in data.train])
    L, B = spec.crop, spec.batch
    n_evals = max(1, a.evals)
    next_eval, trained, step, hist = a.budget / n_evals, 0.0, 0, []
    best = None
    net.train()
    while True:
        t = time.perf_counter()
        for pg in opt.param_groups:  # cosine over the wall-clock budget
            pg["lr"] = spec.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, trained / a.budget)))
        which = g.choice(len(data.train), B, p=shares / shares.sum())
        parts = [crops(data.train[k], int((which == k).sum()), L, g) for k in np.unique(which)]
        x = torch.from_numpy(np.concatenate([p[0] for p in parts]).astype(np.float32)).to(dev)
        y = torch.from_numpy(np.concatenate([p[1] for p in parts])).to(dev)
        w = (y >= 0).float()
        y = y.clamp(min=0)
        # FIXALL step 1 augmentation: feature dropout 0.1, noise 0.1 std, 2 time masks of <= 4 frames
        x = F.dropout(x, 0.1)
        x = x + 0.1 * data.fstd * torch.randn_like(x)
        for _ in range(2):
            ln = torch.randint(0, 5, (len(x),), device=dev)
            s = (torch.rand(len(x), device=dev) * (L - 4)).long()
            tt = torch.arange(L, device=dev)[None]
            m = (tt >= s[:, None]) & (tt < (s + ln)[:, None])
            x = x.masked_fill(m[..., None, None], 0.0)
            w = w.masked_fill(m, 0.0)
        loss = frame_bce(net(x), y, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev == "mps":
            torch.mps.synchronize()
        trained += time.perf_counter() - t
        step += 1
        if trained >= next_eval or trained >= a.budget:
            r = evaluate(net, data, dev)
            hist.append({"t": round(trained, 1), "step": step, "train_loss": round(loss.item(), 5), **r})
            log(f"  {size} t {trained:.0f}/{a.budget:.0f} s step {step}: train {loss.item():.4f} eval "
                f"{r['eval_loss']:.5f} sel_auc {r['sel_auc']:.4f}")
            if best is None or r["eval_loss"] < best["eval_loss"]:
                best = hist[-1]
            next_eval += a.budget / n_evals
            if trained >= a.budget:
                break
    return {"size": size, "params": params, "steps": step, "best": best, "hist": hist}


def knee(rows: list[dict], tol: float) -> dict:
    lmin = min(r["best"]["eval_loss"] for r in rows)
    ok = [r for r in rows if r["best"]["eval_loss"] <= lmin * (1 + tol)]
    pick = min(ok, key=lambda r: r["params"])
    by_p = sorted(rows, key=lambda r: r["params"])
    edge = "largest" if pick is by_p[-1] and len(rows) > 1 else "smallest" if pick is by_p[0] and len(rows) > 1 else None
    return {"pick": pick["size"], "params": pick["params"], "l_min": lmin, "tol": tol, "edge": edge}


# --------------------------------------------------------------------------- report
def write_md(path: Path, head: str, spec: Spec, a, data: Data, rows, k, dev, shipped):
    fmt = lambda s: ", ".join(f"{kk} {vv}" for kk, vv in s.items())  # noqa: E731
    lines = [f"# Capacity sweep: {head}", "",
             f"- Date: {dt.date.today().isoformat()}",
             f"- Command: `uv run scripts/sweep_capacity.py {' '.join(sys.argv[1:])}`",
             f"- Budget: {a.budget:.0f} s of training per candidate (eval time excluded), device {dev}, seed {a.seed}, "
             f"{a.evals} evaluations per candidate, best eval loss reported",
             f"- Data: {data.desc}",
             "- Eval metric: held-out frame BCE (hard labels), the mean over the scopes; sel AUC = mean held-out AUC "
             "with room tone as extra negatives (FIXALL's selection number), for reference",
             f"- Knee rule: fewest parameters with best eval loss <= L_min x (1 + {a.tol}) (L_min = {k['l_min']:.5f})",
             f"- Shipped: {fmt(shipped)} ({spec.shipped_src})", "",
             "| size | params | steps in budget | best eval loss | at t (s) | sel AUC | room p95 | |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["params"]):
        tags = []
        if r["size"] == k["pick"]:
            tags.append("**knee**")
        if all(r["size"].get(kk) == vv for kk, vv in shipped.items() if kk in r["size"]):
            tags.append("shipped")
        b = r["best"]
        lines.append(f"| {fmt(r['size'])} | {r['params']:,} | {r['steps']} | {b['eval_loss']:.5f} | {b['t']} | "
                     f"{b['sel_auc']:.4f} | {b['room_p95']:.3f} | {' '.join(tags)} |")
    shipped_row = next((r for r in rows if all(r["size"].get(kk) == vv for kk, vv in shipped.items() if kk in r["size"])), None)
    lines += ["", f"**Pick:** {fmt(k['pick'])} ({k['params']:,} parameters)."]
    if k["edge"]:
        lines.append(f"The pick is the {k['edge']} candidate: the sweep does not bracket the knee; extend it "
                     f"{'upward' if k['edge'] == 'largest' else 'downward'} before trusting it.")
    if shipped_row is not None:
        d = shipped_row["best"]["eval_loss"] / k["l_min"] - 1
        at = shipped_row["size"] == k["pick"]
        lines.append(f"Shipped size: {'at the knee' if at else 'not at the knee'} (its eval loss is {100 * d:+.2f} % "
                     f"from L_min). A default moves only on a measured held-out gain re-verified on the test split.")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("head", choices=sorted(HEADS))
    p.add_argument("--core", default="115m", choices=["115m", "0p6b"])
    p.add_argument("--sizes", default="", help="comma list for the head's main size key (speech: hidden)")
    p.add_argument("--grid", nargs="*", default=[], help="key=v1,v2 ... ; the cross product is swept")
    p.add_argument("--budget", type=float, default=120.0, help="training seconds per candidate")
    p.add_argument("--evals", type=int, default=6, help="evaluations per candidate (best eval loss is kept)")
    p.add_argument("--tol", type=float, default=0.01, help="knee tolerance, relative to the best eval loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", default="", help="override the cached blocks the head reads (1-based)")
    p.add_argument("--train-frames", type=int, default=500_000, help="frames of training features held in RAM (fp16: 5 KB per frame on the 115M)")
    p.add_argument("--eval-windows", type=int, default=60, help="cap on held-out oto / quiet-oto windows")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--tag", default="", help="head label in the output names (default <head>_<core>)")
    p.add_argument("--out-root", default=str(ROOT), help="where plans/sweeps/ and runs/sweeps/ go (smoke tests)")
    a = p.parse_args()
    spec = HEADS[a.head]
    torch.set_num_threads(2)
    grid = {spec.size_key: [int(s) for s in a.sizes.split(",")]} if a.sizes else {}
    for kv in a.grid:
        kk, vs = kv.split("=")
        grid[kk] = [int(v) for v in vs.split(",")]
    if not grid:
        p.error("give --sizes or --grid")
    cands = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    head = a.tag or f"{a.head}_{a.core}"
    day = dt.date.today().isoformat()
    out_root = Path(a.out_root)
    out_json = out_root / "runs" / "sweeps" / f"{head}_{day}.json"
    out_md = out_root / "plans" / "sweeps" / f"{head}_{day}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    log(f"sweep {head}: {len(cands)} candidates x {a.budget:.0f} s on {a.device}: {cands}")
    data = spec.load(a)
    rows = []
    for c in cands:
        log(f"candidate {c}")
        rows.append(train_one(spec, c, data, a, a.device))
        res = {"head": head, "args": vars(a), "data": data.desc, "rows": rows}
        out_json.write_text(json.dumps(res, indent=1))  # after every candidate: a stopped sweep keeps its rows
    k = knee(rows, a.tol)
    shipped = spec.shipped[a.core]
    res = {"head": head, "date": day, "args": vars(a), "data": data.desc, "rows": rows, "knee": k, "shipped": shipped}
    out_json.write_text(json.dumps(res, indent=1))
    write_md(out_md, head, spec, a, data, rows, k, a.device, shipped)
    for r in sorted(rows, key=lambda r: r["params"]):
        log(f"{r['size']} params {r['params']:,} steps {r['steps']} best eval {r['best']['eval_loss']:.5f}")
    log(f"knee {k['pick']} ({k['params']:,} params) -> {out_md}, {out_json}")


if __name__ == "__main__":
    main()
