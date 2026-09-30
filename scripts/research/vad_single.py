"""Single-layer VAD head vs the served all-layer head (research/VAD_SINGLE.md).

research/archive/VAD_LAYERS.md found a block-4 probe within 0.001 F1 of the served VAD head (a 64-hidden frame head on a
softmax mix of all 17 blocks). This trains real heads with the served recipe (research/recipes/vad_single.yaml) on one
block and measures them against the served head, directly and through every consumer of the VAD in the server
(audioforge/server/streams.py: --asr-vad-gate, the LID head's VAD-gated pooling, the TS-VAD track's arm collection).

Stages (every model-loading call through scripts/dev/gate.sh; each call stops at --budget and is resumable):
  train    (MPS) heads.vad on block --layer (0-based) -> <SSD>/runs/vad_single/L<k>/{state.pt, head.pt}
  eval     (CPU) AMI dev diar 64 x 20 s (BASELINES VAD set) and ICSI dev diar 64 x 20 s: served head and the single-tap
           heads from ONE encoder pass per batch; F1 / acc / FPR / miss at 0.5 (sd.score_vad), AUC, miss and F1 at
           FPR 0.075, paired window-bootstrap CI of the F1 difference
  lid      (CPU / MPS) FLEURS-17 test 2 s and full: the served LID head (runs/lid_distill.pt) with VAD-gated pooling
           (scripts/research/lid.py eval_head's rule) under each VAD head; per-utterance paired bootstrap
  vadwin   (MPS / CPU) per-frame VAD of every head on the eot-bench v2 windows (input of tsvad)
  tsvad    (CPU) TS-VAD track with a live arm print (TSVADTrack.arm: the first 5 s of VAD speech after the last
           other-speaker end before the primary's onset) under each VAD head, on the eot-bench v2 windows; frame F1
           of P(target) vs the primary's labels; then tsvad.py scores --bindings tsvad_armvad_<h>,...
  tsvadturn  the served turn head's turn-end misses on each track (tsvad.score_systems, paired vs served VAD)
  stream   (CPU) streaming words with each VAD head: identical to the served model with --asr-vad-gate off; with the
           gate on (0.5, hangover 15) WER vs reference on the AMI dev windows
  compute  (CPU) ms per 160 ms chunk of the VAD read-out and of the streaming step, served vs single tap

  scripts/dev/gate.sh .venv/bin/python scripts/research/vad_single.py train --layer 3 --budget 1080   # repeat until done
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
OUT = SSD / "runs" / "vad_single"
SERVED = ROOT / "runs" / "stage1_served.afm"
RECIPE = ROOT / "research" / "recipes" / "vad_single.yaml"
JSON = ROOT / "runs" / "vad_single.json"
FPR_POINT = 0.075


def log(*a):
    print(*a, flush=True)


def jload(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def merge(key: str, rec):
    d = jload(JSON)
    d[key] = rec
    JSON.write_text(json.dumps(d, indent=1))


def save_checked(obj: dict, path: Path):
    """torch.save via a temp file + rename, then read back and compare every tensor (exFAT SSD)."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)

    def eq(a, b):
        if torch.is_tensor(a):
            return torch.equal(a.cpu(), b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        return a == b
    assert eq(obj, back), f"read-back mismatch for {path}"
    tmp.replace(path)


def head_cfg(layer: int) -> dict:
    return {"type": "frame", "key": "vad", "hidden": 64, "from_layers": [int(layer)], "weight": 0.5}


def load_head(layer: int):
    """-> FrameHead (eval, cpu) of the finished run on block ``layer`` (0-based)."""
    from audioforge.model import build_head
    blob = torch.load(OUT / f"L{layer}" / "head.pt", map_location="cpu", weights_only=False)
    h = build_head(dict(blob["cfg"]), 512)
    h.load_state_dict(blob["state_dict"])
    return h.eval()


def truncated(enc, k: int):
    from vad_layers import truncated_encoder
    return truncated_encoder(enc, k)


# --------------------------------------------------------------------------- train
def stage_train(a):
    from audioforge.data import Collate, to_device
    from audioforge.model import build_head
    from audioforge.train import Trainer, load_data, load_model, load_recipe
    torch.set_num_threads(2)
    k = int(a.layer)
    d = OUT / f"L{k}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "head.pt").exists():
        log(f"[train] L{k}: done ({d / 'head.pt'})")
        return
    cfg = load_recipe(RECIPE)
    t = cfg["trainer"]
    steps, bs, lr, warm = int(t["max_steps"]), int(t["batch_size"]), float(t["lr"]), int(t["warmup_steps"])
    dev = torch.device(a.device)
    t0 = time.time()
    train = load_data(cfg, "train")
    log(f"[train] data {len(train)} items {train.counts} in {time.time() - t0:.0f}s")
    # the Trainer's batch order (random.Random(0), single-source batches, epoch after epoch) up to max_steps
    rng, order, ns = random.Random(0), [], types.SimpleNamespace(bs=bs)
    while len(order) < steps:
        order += Trainer._batches(ns, train, rng)
    order = order[:steps]
    model = load_model(SERVED, "cpu")
    enc = truncated(model.encoder, k + 1).to(dev)
    pre, sa = model.preprocessor.to(dev), model.spec_augment
    sa = sa.to(dev) if sa is not None else None
    torch.manual_seed(1234 + k)
    head = build_head(head_cfg(k), 512).to(dev)
    for p in list(model.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
        0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))) * 0.95 + 0.05))
    step, hist = 0, []
    st = d / "state.pt"
    if st.exists():
        s = torch.load(st, map_location="cpu", weights_only=False)
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step, hist = int(s["step"]), s["hist"]
        log(f"[train] L{k}: resumed at step {step}")
    ctxs = [list(c) for c in t["att_context_sizes"]]
    col = Collate(None)
    enc.train(), pre.train(), head.train()
    if sa is not None:
        sa.train()
    w = float(head_cfg(k)["weight"])
    clip = float(t["grad_clip"])
    t1, run = time.time(), []

    def ckpt():
        save_checked({"head": {n: v.detach().cpu() for n, v in head.state_dict().items()}, "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "step": step, "hist": hist, "layer": k}, st)

    while step < steps:
        idx = order[step]
        items = [train[j] for j in idx]
        assert all("vad" in ex for ex in items), "every source carries vad (derive: [vad])"
        b = to_device(col([{"audio": ex["audio"], "vad": ex["vad"]} for ex in items]), dev)
        torch.manual_seed(100_000 * k + step)  # SpecAugment / dropout masks: reproducible across resumes
        att = random.Random(7 * steps + step).choice(ctxs)
        with torch.no_grad():
            feats, flen = pre(b["audio"], b["audio_len"])
            if sa is not None:
                feats = sa(feats, flen)
            _, elen, hidden = enc(feats, flen, att, return_hidden=True)
            x = hidden[k]
        loss = w * head.loss(x, elen, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), clip)
        opt.step()
        sched.step()
        step += 1
        run.append(float(loss.detach()) / w)
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss_vad": round(float(np.mean(run)), 4), "grad_norm": round(float(gn), 3),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t1, 1)}
            hist.append(rec)
            run = []
            log(" ".join(f"{kk}={v}" for kk, v in rec.items()))
        if step % int(t["checkpoint_every"]) == 0 and step < steps:
            ckpt()
        if time.time() - t0 > a.budget and step < steps:
            ckpt()
            log(f"[train] L{k}: budget reached at step {step}; re-run to resume")
            return
    blob = {"cfg": head_cfg(k), "state_dict": {n: v.detach().cpu() for n, v in head.state_dict().items()},
            "steps": steps, "recipe": str(RECIPE.relative_to(ROOT)), "hist": hist,
            "params": int(sum(p.numel() for p in head.parameters())), "device": str(dev),
            "sec_last_segment": round(time.time() - t0, 1)}
    save_checked(blob, d / "head.pt")
    ckpt()
    merge(f"train_L{k}", {kk: v for kk, v in blob.items() if kk != "state_dict"})
    log(f"[train] L{k}: wrote {d / 'head.pt'}")


# --------------------------------------------------------------------------- eval (VAD F1 on AMI / ICSI dev)
def roc_f1(scores, labels, fpr_point=FPR_POINT) -> dict:
    """F1 / miss / FPR at the threshold whose FPR is closest to fpr_point from below."""
    s, y = np.concatenate(scores).astype(np.float64), np.concatenate(labels).astype(bool)
    neg = np.sort(s[~y])[::-1]
    j = int(math.floor(fpr_point * len(neg)))
    thr = float(neg[j]) if j < len(neg) else float(neg[-1])
    pred = s > thr
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return {"thr": round(thr, 4), "fpr": round(fp / max((~y).sum(), 1), 4), "miss": round(1 - rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-12), 4)}


def f1_at(scores, labels, thr=0.5) -> float:
    s, y = np.concatenate(scores), np.concatenate(labels).astype(bool)
    p = s > thr
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def boot_diff(sa, sb, labels, n=1000, seed=0) -> list[float]:
    """95 % window-bootstrap CI of F1(a) - F1(b) at 0.5 (paired: the same resampled windows)."""
    rng = np.random.default_rng(seed)
    W = len(labels)
    out = []
    for _ in range(n):
        i = rng.integers(0, W, W)
        out.append(f1_at([sa[j] for j in i], [labels[j] for j in i]) - f1_at([sb[j] for j in i], [labels[j] for j in i]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


@torch.no_grad()
def stage_eval(a):
    from vad_layers import load_set, vad_report

    from audioforge.data import Collate
    from audioforge.train import load_model
    torch.set_num_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    heads = {f"L{k}": load_head(k) for k in layers}
    model = load_model(SERVED, "cpu")
    col = Collate(None)
    res = {}
    for name in ("ami_dev", "icsi_dev"):
        val = load_set(name)
        sc = {"served": [], **{h: [] for h in heads}}
        labels = []
        t0 = time.time()
        for i in range(0, len(val), 8):
            vb = val[i: i + 8]
            b = col([{"audio": v["audio"]} for v in vb])
            enc, elen, hid = model.encode(b["audio"], b["audio_len"], return_hidden=True)
            outs = {"served": model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()}
            for hn, h in heads.items():
                outs[hn] = h(hid[int(hn[1:])]).sigmoid()
            for j, v in enumerate(vb):
                T = min(int(elen[j]), len(v["vad"]))
                labels.append(np.asarray(v["vad"][:T]) > 0.5)
                for hn in sc:
                    sc[hn].append(outs[hn][j, :T].numpy())
        r = {"n_windows": len(val), "n_frames": int(sum(len(x) for x in labels)), "sec": round(time.time() - t0, 1)}
        for hn in sc:
            rr = vad_report(sc[hn], labels)
            rr["at_fpr0.075"] = roc_f1(sc[hn], labels)
            if hn != "served":
                rr["df1_vs_served"] = round(rr["f1"] - r["served"]["f1"], 4)
                rr["df1_ci95"] = boot_diff(sc[hn], sc["served"], labels)
            r[hn] = rr
            log(name, hn, json.dumps(rr))
        res[name] = r
        np.savez_compressed(OUT / f"eval_{name}.npz", labels=np.array(labels, dtype=object),
                            **{hn: np.array(v, dtype=object) for hn, v in sc.items()})
    merge("eval", res)


# --------------------------------------------------------------------------- LID with VAD-gated pooling
@torch.no_grad()
def stage_lid(a):
    """lid.py eval_head's VAD-gated rule (frames with VAD <= 0.5 not pooled; no speech frame -> pool all), FLEURS-17
    test 2 s / full, with the served VAD and each single-tap VAD from the same encoder pass. Resumable per 250 rows."""
    import lid as L

    from audioforge.lid import attach_head
    from audioforge.train import load_model
    torch.set_num_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    heads = {f"L{k}": load_head(k) for k in layers}
    dev = torch.device(a.device)
    model = load_model(SERVED, "cpu")
    name = attach_head(model, str(ROOT / "runs" / "lid_distill.pt"))
    model = model.to(dev).eval()
    heads = {hn: h.to(dev) for hn, h in heads.items()}
    lid = model.heads[name]
    ons = L.load_onsets()
    rows = L.rows_for("fleurs")
    part = OUT / "lid"
    part.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    CH = 250
    for cond in a.conds.split(","):
        order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
        for c0 in range(0, len(order), CH):
            f = part / f"{cond}_{c0:05d}.npz"
            if f.exists():
                continue
            if time.time() - t0 > a.budget:
                log(f"[lid] budget reached at {cond} {c0}; re-run")
                return
            ids = order[c0: c0 + CH]
            out = {"idx": np.array(ids), "P": [], **{f"G_{hn}": [] for hn in ["served", *heads]}}
            for s in range(0, len(ids), a.batch_size):
                idx = ids[s: s + a.batch_size]
                clips = [L.clip(L.load_audio(rows[i]), ons[rows[i]["id"]][0], cond) for i in idx]
                lens = torch.tensor([len(c) for c in clips])
                x = torch.zeros(len(clips), int(lens.max()))
                for q, c in enumerate(clips):
                    x[q, : len(c)] = torch.from_numpy(c)
                enc, elen, hid = model.encode(x.to(dev), lens.to(dev), return_hidden=True)
                e = model.head_input(name, enc, hid)
                out["P"].append(lid(e, elen).softmax(-1).float().cpu().numpy())
                w, wx, wxx = lid._terms(e)
                valid = (torch.arange(e.shape[1], device=dev)[None] < elen[:, None]).float()[..., None]
                vads = {"served": model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()}
                for hn, h in heads.items():
                    vads[hn] = h(hid[int(hn[1:])]).sigmoid()
                for hn, v in vads.items():
                    m = valid * (v > 0.5).float()[..., None]
                    m = torch.where(m.sum(1, keepdim=True) > 0, m, valid)
                    out[f"G_{hn}"].append(lid._classify((w * m).sum(1), (wx * m).sum(1), (wxx * m).sum(1))
                                          .softmax(-1).float().cpu().numpy())
            np.savez(f, **{kk: (np.concatenate(v) if isinstance(v, list) else v) for kk, v in out.items()})
            chk = np.load(f)
            assert np.array_equal(chk["idx"], np.array(ids))
            log(f"[lid] {cond} rows {c0}-{c0 + len(ids)} {time.time() - t0:.0f}s")
    y = np.array([L.CODES.index(r["lang"]) for r in rows])
    res = {"head": "runs/lid_distill.pt", "n": len(rows), "rule": "lid.py eval_head VAD-gated pooling (VAD > 0.5)"}
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(rows), (1000, len(rows)))
    for cond in a.conds.split(","):
        fs = sorted(part.glob(f"{cond}_*.npz"))
        if sum(len(np.load(f)["idx"]) for f in fs) != len(rows):
            log(f"[lid] {cond} incomplete")
            return
        acc = {}
        oks = {}
        for kk in ["P", "G_served", *[f"G_{hn}" for hn in heads]]:
            P = np.zeros((len(rows), len(L.CODES)), np.float32)
            for f in fs:
                z = np.load(f)
                P[z["idx"]] = z[kk]
            ok = P.argmax(-1) == y
            oks[kk] = ok
            acc[kk] = {"acc": round(float(ok.mean()), 4),
                       "ci95": [round(float(np.percentile(ok[B].mean(1), q)), 4) for q in (2.5, 97.5)]}
        for hn in heads:
            dlt = oks[f"G_{hn}"].astype(float) - oks["G_served"]
            acc[f"G_{hn}"]["d_vs_served"] = round(float(dlt.mean()), 4)
            acc[f"G_{hn}"]["d_ci95"] = [round(float(np.percentile(dlt[B].mean(1), q)), 4) for q in (2.5, 97.5)]
            acc[f"G_{hn}"]["n_changed"] = int((oks[f"G_{hn}"] != oks["G_served"]).sum())
        res[cond] = acc
        log(cond, json.dumps(acc))
    merge("lid", res)


# --------------------------------------------------------------------------- TS-VAD track with a VAD-armed print
def with_vad(model, layer: int):
    """``model`` with heads.vad replaced by the single-tap head on block ``layer`` (in place; config included)."""
    h = load_head(layer).to(next(model.parameters()).device)
    model.heads["vad"] = h
    model.head_cfg["vad"] = head_cfg(layer)
    model.cfg["heads"]["vad"] = head_cfg(layer)
    model.layer_tap["vad"] = [int(layer)]
    if "vad" in model.layer_mix:
        del model.layer_mix["vad"]
    return model


def vadwin_path(corpus, v) -> Path:
    import tsvad as T
    return OUT / "tsvad" / corpus / "vad" / f"{T.wkey(v)}.npy"


@torch.no_grad()
def stage_vadwin(a):
    """Per eot-bench v2 window (AMI dev 974, ICSI held-out 1312) the per-frame VAD of the served head and of each
    single-tap head, from one plain causal encode at [70, 1] (= cache-aware streaming) -> <OUT>/tsvad/<corpus>/vad."""
    import tsvad as T

    from audioforge.data import Collate
    from audioforge.train import load_model
    torch.set_num_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    dev = torch.device(a.device)
    heads = {k: load_head(k).to(dev) for k in layers}
    model = load_model(SERVED, "cpu").to(dev).eval()
    col = Collate(None)
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        (OUT / "tsvad" / corpus / "vad").mkdir(parents=True, exist_ok=True)
        todo = sorted([v for v in ext if not vadwin_path(corpus, v).exists()], key=lambda v: len(v["audio"]))
        done = 0
        for i in range(0, len(todo), 8):
            if time.time() - t0 > a.budget:
                break
            cc = todo[i: i + 8]
            b = col([{"audio": np.asarray(v["audio"], np.float32)} for v in cc])
            enc, elen, hid = model.encode(b["audio"].to(dev), b["audio_len"].to(dev), return_hidden=True)
            cols = [model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()]
            cols += [heads[k](hid[k]).sigmoid() for k in layers]
            P = torch.stack(cols, -1).float().cpu().numpy()
            for j, v in enumerate(cc):
                x = P[j, : int(elen[j])].astype(np.float16)
                q = vadwin_path(corpus, v)
                np.save(q, x)
                assert np.array_equal(np.load(q), x)
            done += len(cc)
        left = len(todo) - done
        log(f"[vadwin] {corpus}: {done} done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            return


def stage_tsvad(a):
    """TS-VAD track (runs/tsvad_spk.pt) with TSVADTrack's live arm: armed at the last other-speaker speech end before
    the primary's onset (tsvad_enroll.arm_frame), print = the first 5 s of frames the VAD calls speech (> 0.5) after it
    (serve --enroll after_agent_arm); plain-VAD mode before. One track per VAD head -> tsvad.py bind_tsvad_armvad_<h>
    (then tsvad.py scores --bindings ...); frame metrics of P(target) vs the primary's labels -> runs/vad_single.json."""
    import tsvad as T
    from tsvad_enroll import arm_frame

    from audioforge.tsvad_stream import TSVADTrack
    T.set_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    names = ["served", *[f"L{k}" for k in layers]]
    model = T.load_served("cpu")
    head, _ = T.load_head(ROOT / "runs" / "tsvad_spk.pt")
    spk = model.heads["spk"]
    t0 = time.time()
    res = jload(JSON).get("tsvad_frame", {})
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        for n in names:
            (T.WORK / corpus / f"bind_tsvad_armvad_{n}").mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not all(T.bind_path(corpus, f"tsvad_armvad_{n}", v).exists() for n in names)]
        for v in todo:
            if time.time() - t0 > a.budget:
                log(f"[tsvad] {corpus}: budget reached; re-run")
                return
            f = T.win_feats(corpus, v)
            vad = np.load(vadwin_path(corpus, v)).astype(np.float32)
            if len(vad) < len(f):
                vad = np.concatenate([vad, np.repeat(vad[-1:], len(f) - len(vad), 0)])
            arm = arm_frame(v)
            for ci, n in enumerate(names):
                tr = TSVADTrack(head, spk, print_s=5.0)
                tr.arm(arm)
                p = tr.feed(torch.as_tensor(f)[None], vad[: len(f), ci])
                q = T.bind_path(corpus, f"tsvad_armvad_{n}", v)
                np.save(q, p.astype(np.float32))
        r = {"n": len(ext)}
        for n in names:
            P, Y, got = [], [], 0
            for v in ext:
                p = np.load(T.bind_path(corpus, f"tsvad_armvad_{n}", v))
                y = np.asarray(v["spk_act"], np.float32)
                t = min(len(p), len(y))
                P.append(p[:t, 0])
                Y.append(y[:t])
            pr, rc, f1, fa = T.frame_prf(np.concatenate(P), np.concatenate(Y))
            r[n] = {"f1": round(f1, 4), "precision": round(pr, 4), "miss": round(1 - rc, 4), "fa": round(fa, 4)}
            # paired window bootstrap of the F1 difference vs the served-VAD track
            r[n]["_P"] = P
            r["_Y"] = Y
        rng = np.random.default_rng(0)
        B = rng.integers(0, len(ext), (500, len(ext)))
        for n in names[1:]:
            d = []
            for bi in B:
                yy = np.concatenate([r["_Y"][i] for i in bi])
                d.append(T.frame_prf(np.concatenate([r[n]["_P"][i] for i in bi]), yy)[2]
                         - T.frame_prf(np.concatenate([r["served"]["_P"][i] for i in bi]), yy)[2])
            r[n]["df1_vs_served"] = round(r[n]["f1"] - r["served"]["f1"], 4)
            r[n]["df1_ci95"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
            r[n]["windows_changed"] = int(sum(not np.array_equal(p, q) for p, q in zip(r[n]["_P"], r["served"]["_P"])))
        for n in names:
            r[n].pop("_P")
        r.pop("_Y")
        res[corpus] = r
        log(f"[tsvad] {corpus}: {json.dumps(r)} ({time.time() - t0:.0f}s)")
    merge("tsvad_frame", res)


def stage_tsvadturn(a):
    """eot-bench v2 turn-end misses of the served turn head (trail6) on each VAD-armed TS-VAD track (tsvad.py scores),
    head alone and hybrid_dyn with Silero, cross-fitted <= 5 % per-turn FC, 2 s / 6 s horizons, paired vs served VAD."""
    import bench_turn_icsi as M
    import tsvad as T
    layers = [int(x) for x in a.layers.split(",")]
    names = ["served", *[f"L{k}" for k in layers]]
    ec = lambda t: (t // T.CHUNK + 1) * T.CHUNK  # noqa: E731
    out = {}
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        convs = [dict(v) for v in ext]
        for v in convs:
            v.pop("audio", None)
        folds = T.folds_of(corpus, convs)
        Tl = [len(v["spk_act"]) for v in convs]
        sil = [M.silero_timeout_track(np.load(T.SILERO[corpus] / f"{T.wkey(v)}.npy"), t)[:t] for v, t in zip(ext, Tl)]
        singles, hybrids = {}, {}
        for n in names:
            b = f"tsvad_armvad_{n}"
            head = [np.load(T.score_path(corpus, b, v))[:t] for v, t in zip(ext, Tl)]
            singles[f"head_{n}"] = (head, ec)
            hybrids[f"hybrid_dyn_{n}"] = (head, ec, [T.dyn_track(s_, h) for s_, h in zip(sil, head)], ec)
        pairs = [(f"{k}_{n}", f"{k}_served") for n in names[1:] for k in ("head", "hybrid_dyn")]
        r = T.score_systems(convs, meta, folds, singles, hybrids, {}, {"2s": 25, "6s": 75}, a.n_boot, pairs)
        out[corpus] = r
        for name, rr in r["systems"].items():
            x = rr["6s"]
            log(f"  {corpus} {name:22s} miss6s {x['miss_rate'] * 100:5.1f} FC {x['fc_rate'] * 100:4.1f} "
                f"miss2s {rr['2s']['miss_rate'] * 100:5.1f}")
        for k, d in r["paired"].items():
            log(f"  {corpus} {k}: {json.dumps(d['all'])}")
    merge("tsvad_turn", out)


# --------------------------------------------------------------------------- streaming words
@torch.no_grad()
def stage_stream(a):
    """ASRStream (the server's ASR pass) over AMI dev diar windows, fed in 20 ms blocks: tokens with the served model
    and with each single-tap VAD, --asr-vad-gate off (must be identical) and on (0.5, hangover 15). Resumable per run."""
    from vad_layers import load_set

    from audioforge.server.streams import ASRStream
    from audioforge.train import load_model
    torch.set_num_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    val = load_set("ami_dev")[: a.n_stream]
    part = OUT / "stream.json"
    st = jload(part)
    t0 = time.time()
    for tag in ["served", *[f"L{k}" for k in layers]]:
        model = load_model(SERVED, "cpu").eval()
        if tag != "served":
            with_vad(model, int(tag[1:]))
        for gate in (None, 0.5):
            for wi, v in enumerate(val):
                key = f"{tag}|{gate}|{wi}"
                if key in st:
                    continue
                if time.time() - t0 > a.budget:
                    log("[stream] budget reached; re-run")
                    return
                s = ASRStream(model, turn=None, vad_gate=gate)
                x = np.asarray(v["audio"], np.float32)
                vads = []
                for i in range(0, len(x), 320):
                    vads += [r["vad"] for r in s.feed_frames(x[i: i + 320])]
                vads += [r["vad"] for r in s.feed_frames(np.zeros(0, np.float32), final=True)]
                st[key] = {"tokens": list(map(int, s.tokens)), "text": s.text, "n_gated": s.n_gated,
                           "vad_mean": round(float(np.mean(vads)), 4) if vads else None}
                part.write_text(json.dumps(st))
                log(f"[stream] {key} {len(s.tokens)} tokens gated {s.n_gated} ({time.time() - t0:.0f}s)")
    from audioforge.metrics import wer
    res = {"n_windows": len(val), "audio_s": round(sum(len(v["audio"]) for v in val) / 16000, 1)}
    ref_off = [st[f"served|None|{i}"]["text"] for i in range(len(val))]
    ref_on = [st[f"served|0.5|{i}"]["text"] for i in range(len(val))]
    for tag in [f"L{k}" for k in layers]:
        off_same = all(st[f"{tag}|None|{i}"]["tokens"] == st[f"served|None|{i}"]["tokens"] for i in range(len(val)))
        on = [st[f"{tag}|0.5|{i}"]["text"] for i in range(len(val))]
        res[tag] = {"gate_off_tokens_identical": off_same,
                    "gate_on_windows_identical_to_served_gate_on": int(sum(
                        st[f"{tag}|0.5|{i}"]["tokens"] == st[f"served|0.5|{i}"]["tokens"] for i in range(len(val)))),
                    "gate_on_wer_vs_served_gate_on": round(wer(ref_on, on), 4),
                    "gate_on_wer_vs_gate_off": round(wer(ref_off, on), 4),
                    "frames_gated": int(sum(st[f"{tag}|0.5|{i}"]["n_gated"] for i in range(len(val))))}
    res["served"] = {"gate_on_wer_vs_gate_off": round(wer(ref_off, ref_on), 4),
                     "frames_gated": int(sum(st[f"served|0.5|{i}"]["n_gated"] for i in range(len(val))))}
    log(json.dumps(res))
    merge("stream", res)


# --------------------------------------------------------------------------- compute
@torch.no_grad()
def stage_compute(a):
    """ms per 160 ms chunk of the VAD read-out (head_input + head + sigmoid, as ASRStream.feed_frames does) for the
    served all-layer head and a single tap, on real streaming chunks (hidden states from stream_step), CPU 2 threads."""
    from vad_layers import load_set

    from audioforge.train import load_model
    torch.set_num_threads(2)
    layers = [int(x) for x in a.layers.split(",")]
    served = load_model(SERVED, "cpu").eval()
    single = {k: with_vad(load_model(SERVED, "cpu").eval(), k) for k in layers}
    v = load_set("ami_dev")[47]
    from audioforge.model import StreamingSession
    s = StreamingSession(served)
    x = torch.as_tensor(np.asarray(v["audio"], np.float32))
    feats, flen = served.preprocessor(x[None], torch.tensor([len(x)]))
    st, chunks = None, []
    cm = served.encoder.stream_chunk_frames(s.att)
    for i in range(0, feats.shape[-1] - cm + 1, cm):
        enc, hid, st = served.encoder.stream_step(feats[..., i: i + cm], st, s.att, return_hidden=True)
        if enc.shape[1]:
            chunks.append((enc, hid))

    def readout(m):
        for enc, hid in chunks:
            m.heads["vad"](m.head_input("vad", enc, hid)).sigmoid()[0].tolist()

    res = {"chunks": len(chunks), "reps": a.reps}
    ms = {"served": [], **{f"L{k}": [] for k in layers}}
    for _ in range(a.reps):  # interleaved A/B
        for tag, m in [("served", served), *[(f"L{k}", single[k]) for k in layers]]:
            t0 = time.perf_counter()
            readout(m)
            ms[tag].append((time.perf_counter() - t0) / len(chunks) * 1000)
    for tag, xs in ms.items():
        res[tag] = {"vad_readout_ms_per_chunk_p50": round(float(np.median(xs)), 4),
                    "min": round(float(np.min(xs)), 4)}
    log(json.dumps(res))
    merge("compute_readout", res)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("train", "eval", "lid", "vadwin", "tsvad", "tsvadturn", "stream", "compute"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--layers", default="3,8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=540.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conds", default="2s,full")
    ap.add_argument("--corpora", default="ami,icsi")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-stream", type=int, default=8)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    if a.stage == "train" and a.device == "cpu" and "--device" not in sys.argv:
        a.device = "mps"
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
