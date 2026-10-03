"""Training and evaluation stages of scripts/research/voice_gender.py (run that file, not this one): ``sweep`` (equal
wall-clock capacity sweep that also trains the shipped head), ``test`` (the public test tables) and ``latency``
(the served engine per 160 ms chunk with and without the head)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time

import numpy as np

V = None  # the voice_gender module (set by main)


# --------------------------------------------------------------------------- sweep (= training)
def sample(train, B, L, g):
    """Class-balanced batch: class 0 / 1 with equal odds, then the source (FLEURS / LibriSpeech) with equal odds, a
    random row of that class and a random crop of up to L frames. -> X (B, L, D), lens (B,), keep (B, L), y (B,)"""
    ys = g.integers(0, 2, B)
    X = np.zeros((B, L, train[0][0].X.shape[1]), np.float32)
    K = np.zeros((B, L), bool)
    lens = np.zeros(B, np.int64)
    for q, c in enumerate(ys):
        S, by = train[g.integers(0, len(train))]
        i = by[c][g.integers(0, len(by[c]))]
        x, v = S.row(i)
        n = len(x)
        s0 = g.integers(0, max(1, n - L + 1))
        x, v = x[s0:s0 + L], v[s0:s0 + L]
        X[q, :len(x)], K[q, :len(x)], lens[q] = x, v > 0.5, len(x)
    return X, lens, K, ys


def evaluate(head, held, dev):
    out = {}
    for name, S in held.items():
        p = V.posteriors(head, S, dev, at=(25,))
        out[f"{name}_2s"] = V.bal_ce(p[25], S.y)
        out[f"{name}_full"] = V.bal_ce(p["full"], S.y)
        out[f"{name}_bacc_2s"] = float(np.mean([((p[25] > 0.5) == c)[S.y == c].mean() for c in (0, 1)]))
    loss = float(np.mean([v for k, v in out.items() if "bacc" not in k]))
    return {"eval_loss": round(loss, 5), **{k: round(v, 4) for k, v in out.items()}}


def train_one(hidden, train, held, d_in, a, dev):
    import torch
    import torch.nn.functional as F

    from audioforge.heads.voice_gender import VoiceGenderHead
    torch.manual_seed(a.seed)
    g = np.random.default_rng(1000 + a.seed)
    head = VoiceGenderHead(d_in, hidden=hidden).to(dev)
    params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    fstd = float(train[0][0].X[:20000].astype(np.float32).std())
    next_eval, trained, step, hist, best, best_state = a.budget / a.evals, 0.0, 0, [], None, None
    while True:
        t = time.perf_counter()
        for pg in opt.param_groups:  # cosine over the wall-clock budget
            pg["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, trained / a.budget)))
        X, lens, K, y = sample(train, a.batch, a.crop, g)
        X = torch.from_numpy(X).to(dev)
        X = F.dropout(X, 0.1) + 0.1 * fstd * torch.randn_like(X)  # feature dropout + noise (FIXALL step 1's recipe)
        head.train()
        loss = head.loss(X, torch.from_numpy(lens).to(dev),
                         {"voice_gender": torch.from_numpy(y).to(dev), "keep": torch.from_numpy(K).to(dev)})
        opt.zero_grad()
        loss.backward()
        opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()
        trained += time.perf_counter() - t
        step += 1
        if trained >= next_eval or trained >= a.budget:
            r = evaluate(head, held, dev)
            hist.append({"t": round(trained, 1), "step": step, "train_loss": round(loss.item(), 5), **r})
            V.log(f"  hidden {hidden} t {trained:.0f}/{a.budget:.0f} s step {step}: train {loss.item():.4f} {r}")
            if best is None or r["eval_loss"] < best["eval_loss"]:
                best = hist[-1]
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            next_eval += a.budget / a.evals
            if trained >= a.budget:
                break
    return {"hidden": hidden, "params": params, "steps": step, "best": best, "hist": hist}, best_state


def stage_sweep(a):
    import torch
    dev = torch.device(a.device)
    train = []
    for name in ("fleurs_train", "ls_train"):
        S = V.Set(a.core, name, ram=True)
        train.append((S, {c: np.flatnonzero(S.y == c) for c in (0, 1)}))
        V.log(f"{name}: {S.n} rows, {len(S.X)} frames, {S.y.mean():.2f} male")
    held = {"fleurs_dev": V.Set(a.core, "fleurs_dev", ram=True), "ls_dev": V.Set(a.core, "ls_dev", ram=True)}
    d_in = train[0][0].X.shape[1]
    rows, states = [], {}
    for h in [int(s) for s in a.sizes.split(",")]:
        r, st = train_one(h, train, held, d_in, a, dev)
        rows.append(r)
        states[h] = st
    lmin = min(r["best"]["eval_loss"] for r in rows)
    pick = min((r for r in rows if r["best"]["eval_loss"] <= lmin * (1 + a.tol)), key=lambda r: r["params"])
    V.save(f"sweep_{a.core}", {"rows": rows, "pick": pick["hidden"], "l_min": lmin, "args": vars(a)})
    write_md(a, rows, pick, lmin)
    from audioforge.heads.voice_gender import LABELS
    from audioforge.voice_gender import save_head_state
    path = V.ROOT / "assets" / f"voice_gender_{a.core}.pt"
    save_head_state(states[pick["hidden"]], {"type": "voice_gender", "hidden": pick["hidden"], "labels": list(LABELS)},
                    [V.BLOCK[a.core] - 1], d_in, path,
                    meta={"recipe": "research/VOICE_GENDER.md", "sweep": f"plans/sweeps/voice_gender_{dt.date.today()}.md",
                          "core": a.core, "block": V.BLOCK[a.core], "best": pick["best"]})
    V.log(f"pick hidden {pick['hidden']} ({pick['params']} params) -> {path}")


def write_md(a, rows, pick, lmin):
    p = V.ROOT / "plans" / "sweeps" / f"voice_gender_{dt.date.today()}.md"
    head = (f"# Capacity sweep: voice_gender head (`hidden`), both cores\n\n- Date: {dt.date.today()}\n"
            "- Script: `scripts/research/voice_gender.py sweep` (the protocol of scripts/sweep_capacity.py: equal "
            "wall-clock budget per candidate, cosine LR over it, one seed, best held-out eval loss, knee = fewest "
            "parameters within tol of the best; that script has no registry entry for pooled classifier heads)\n"
            "- Eval loss: class-balanced cross-entropy of P(male), mean of FLEURS dev and LibriSpeech dev-clean "
            "(20 utterances x 40 held-out speakers), after 2 s of pooled speech and on the full clip\n"
            "- Train: FLEURS train (17 languages) + LibriSpeech train-clean-100 (251 speakers), class-balanced\n"
            "- Other sizes fixed: att_hidden 64 (placeholder: never swept)\n")
    sec = (f"\n## {a.core} (block {V.BLOCK[a.core]}; budget {a.budget:.0f} s per candidate on {a.device}, "
           f"batch {a.batch}, crop {a.crop} frames, lr {a.lr}, seed {a.seed}, {a.evals} evals)\n\n"
           "| hidden | params | steps in budget | best eval loss | at t (s) | FLEURS dev bal. acc 2 s | "
           "LS dev bal. acc 2 s | |\n|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r: r["params"]):
        b = r["best"]
        sec += (f"| {r['hidden']} | {r['params']:,} | {r['steps']} | {b['eval_loss']:.5f} | {b['t']} | "
                f"{100 * b['fleurs_dev_bacc_2s']:.1f} | {100 * b['ls_dev_bacc_2s']:.1f} | "
                f"{'**knee**' if r is pick else ''} |\n")
    by_p = sorted(rows, key=lambda r: r["params"])
    edge = (" The pick is the largest candidate: the sweep does not bracket the knee." if pick is by_p[-1] else
            " The pick is the smallest candidate: the knee may lie lower." if pick is by_p[0] else "")
    sec += (f"\n**Pick ({a.core}):** hidden {pick['hidden']} ({pick['params']:,} parameters); L_min {lmin:.5f}, "
            f"tol {a.tol}.{edge} Shipped as assets/voice_gender_{a.core}.pt (the pick's best-eval-loss state).\n")
    text = p.read_text() if p.exists() else head
    import re
    text = re.sub(rf"\n## {a.core} \(.*?(?=\n## |\Z)", "", text, flags=re.S)
    p.write_text(text.rstrip("\n") + "\n" + sec)


# --------------------------------------------------------------------------- test
def boot_ci(y, pred, unit, B=1000, seed=0):
    """95 % percentile bootstrap over ``unit`` (speakers, or rows when the data has no speaker id) of accuracy,
    balanced accuracy and per-class accuracy."""
    def stats(sel):
        yy, pp = y[sel], pred[sel]
        pc = [float((pp[yy == c] == c).mean()) if (yy == c).any() else np.nan for c in (0, 1)]
        return [float((pp == yy).mean()), float(np.nanmean(pc)), *pc]
    units = np.unique(unit)
    rows_of = {u: np.flatnonzero(unit == u) for u in units}
    g = np.random.default_rng(seed)
    bs = np.array([stats(np.concatenate([rows_of[u] for u in g.choice(units, len(units))])) for _ in range(B)])
    pt = stats(np.arange(len(y)))
    return [(round(100 * p, 1), round(100 * np.nanpercentile(bs[:, k], 2.5), 1),
             round(100 * np.nanpercentile(bs[:, k], 97.5), 1)) for k, p in enumerate(pt)]


def stage_test(a):
    import torch

    from audioforge.voice_gender import build_from_file
    dev = torch.device(a.device)
    head, blob = build_from_file(V.ROOT / "assets" / f"voice_gender_{a.core}.pt")
    head = head.to(dev)
    res = {}
    for name in ("fleurs_test", "ls_test"):
        S = V.Set(a.core, name, ram=True)
        p = V.posteriors(head, S, dev)
        subsets = {"FLEURS-17 test": np.ones(S.n, bool), "FLEURS English test": S.lang == "en"} \
            if name == "fleurs_test" else {"LibriSpeech test-clean": np.ones(S.n, bool)}
        for label, sel in subsets.items():
            unit = S.spk[sel]
            r = {"rows": int(sel.sum()), "speakers": int(len(np.unique(unit))) if name == "ls_test" else None,
                 "female_rows": int((S.y[sel] == 0).sum()), "male_rows": int((S.y[sel] == 1).sum()),
                 "unit": "speaker" if name == "ls_test" else "row"}
            for k, tag in ((13, "1s"), (25, "2s"), ("full", "full")):
                r[tag] = boot_ci(S.y[sel], (p[k][sel] > 0.5).astype(int), unit)
            res[label] = r
            V.log(label, json.dumps(r))
    V.save(f"test_{a.core}", res)


# --------------------------------------------------------------------------- latency
def stage_latency(a):
    """The --mode single engine on the bundled 16 s call, chunk compute p50 / p95 per 160 ms chunk with and without
    the voice-gender stream: one engine with the head attached; "off" sessions drop the stream (``asr.gender =
    None``), so both arms share the model, the process and the device state. 9 rounds, the order alternating per
    round; the best p50 and the median of the run p50s are compared, plus the stream's own time per chunk."""
    import torch

    import audioforge.serve as S
    from audioforge.data import load_wav
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    torch.set_num_threads(2)
    root = V.ROOT
    x = load_wav(str(root / "examples/audio/two_party_call_16s.wav"), V.SR).astype(np.float32)
    tsv = root / "assets" / ("tsvad_spk.pt" if a.core == "115m" else "tsvad_0p6b.pt")
    lid = root / "assets" / ("lid_115m_v2.pt" if a.core == "115m" else "lid_0p6b_v2.pt")
    opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(tsv), "lid": str(lid),
            "voice_gender": str(root / "assets" / f"voice_gender_{a.core}.pt")}
    eng = Engine.load(str(V.AFM[a.core]), None, a.device, threads=2, **opts)
    eng.warmup()
    best, p50s = {}, {"off": [], "on": []}
    for rnd in range(9):
        for k in (("off", "on") if rnd % 2 == 0 else ("on", "off")):
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))
            if k == "off":
                s.asr.gender = None
            msgs = []
            for i in range(0, len(x), 2560):
                msgs += s.process(x[i:i + 2560])
            msgs += s.finish()
            cm = np.array(list(s.chunk_ms))
            r = {"p50": round(float(np.percentile(cm, 50)), 2), "p95": round(float(np.percentile(cm, 95)), 2),
                 "gender_ms_per_chunk": (round(s.asr.gender.ms / max(1, len(cm)), 4) if k == "on" else None),
                 "finals_with_field": sum("voice_gender" in m for m in msgs if m["type"] == "final"),
                 "stats_field": next((m.get("voice_gender") for m in msgs if m["type"] == "stats"), None)}
            p50s[k].append(r["p50"])
            if k not in best or r["p50"] < best[k]["p50"]:
                best[k] = r
    best["delta_p50_ms"] = round(best["on"]["p50"] - best["off"]["p50"], 2)  # best run vs best run
    best["delta_median_p50_ms"] = round(float(np.median(p50s["on"]) - np.median(p50s["off"])), 2)
    best["p50_runs"] = p50s
    V.log(json.dumps(best))
    V.save(f"latency_{a.core}_{a.device}", best)


def main(mod):
    global V
    V = mod
    import torch
    p = argparse.ArgumentParser(description=V.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["check", "feats", "sweep", "test", "latency"])
    p.add_argument("--core", default="115m", choices=["115m", "0p6b"])
    p.add_argument("--sets", default="fleurs_dev,ls_dev,fleurs_train,fleurs_test,ls_test")
    p.add_argument("--sizes", default="32,64,128,256")
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--evals", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--crop", type=int, default=75, help="training crop, frames (75 = 6 s)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    {"check": V.stage_check, "feats": V.stage_feats, "sweep": stage_sweep, "test": stage_test,
     "latency": stage_latency}[a.stage](a)
