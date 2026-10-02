"""Target-speaker VAD head on the frozen 115M streaming encoder (research/IMPROVE_115M.md, Part A).

The head (audioforge/heads/tsvad.py) reads block 4 of the served encoder (the speaker head's tap, att context [70, 1]
= 160 ms chunks) and an enrollment embedding (the served block-4 relational speaker head's voice print, or TitaNet-L's)
and outputs per-frame [P(target), P(other)]. Everything else stays frozen; the head is trained on cached block-4
features, so the encoder never runs in the training loop.

Stages (each one process <= 10 min, resumable; CPU at 2 threads unless noted):
  trainfeats  block-4 features of 16 s crops (hop 12 s) of the AMI train (12) and ICSI train (12) meetings + per-speaker
              frame activity -> data/cache/tsvad/feats/<corpus>_<meeting>.npz
  enroll      enrollment pools for training: per (meeting, speaker) N clips of U(1, 5) s single-speaker speech (the
              library's asr-mode segments), embedded by the served speaker head (and TitaNet-L with --titanet)
              -> data/cache/tsvad/enroll/<corpus>_<meeting>.npz
  train       (MPS) the head on the cached features -> runs/tsvad_<tag>.pt (+ log)
  winfeats    block-4 features of the eot-bench v2 extended windows (AMI dev 974, ICSI held-out 1312) -> <work>/<corpus>/feat
  vprints     per window and labelled speaker, the voice print = that speaker's single-speaker speech ELSEWHERE in the
              same meeting (segments overlapping the window excluded), 5 s and 1.5 s, speaker head (+ TitaNet)
  frame       frame metrics: TS-VAD vs the Sortformer column bound by the same voice print (+ oracle column)
  bind        per-frame Sortformer bindings by voice print (speaker-head / TitaNet following) for the turn bench
  scores      turn-head (served trail6) scores on the TS-VAD track and on the voice-print-bound Sortformer track
  report      eot-bench v2 tables (6 s block C, cross-fitted <= 5 % per-turn FC, paired bootstraps) -> runs/improve_115m.json
  n3tracks / n3tnemb / n3check / framearm / framearmreport
              the Nemotron-3 arm of `frame` (served settings; tracks + TitaNet column embeddings cached under
              <work>/<corpus>/n3, per-window counts, window bootstraps paired vs tsvad_spk_vp5p0)

  .venv/bin/python scripts/research/tsvad.py trainfeats --budget 540     # repeat until "0 left"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SERVED = ROOT / "runs" / "stage1_served.afm"
CACHE = ROOT / "data" / "cache" / "tsvad"
SCRATCH = Path(os.environ.get("AUDIOFORGE_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch"))  # machine rule: scratch on the SSD
WORK = SCRATCH / "tsvad"
SR = 16000
FS = 1280  # samples per 80 ms frame
TAP = 3  # 0-based block index = block 4 (the served speaker head's from_layers)
ATT = [70, 1]
CROP_SEC, HOP_SEC = 16.0, 12.0
N_ENROLL = 32
ENROLL_SEC = (1.0, 5.0)
AMI_TRAIN = None  # the library's default AMI train subset (12 meetings)
ICSI_TRAIN = None


def log(*a):
    print(*a, flush=True)


def set_threads(n: int = 2):
    torch.set_num_threads(n)


def load_served(device: str = "cpu", path=None):
    from audioforge.train import load_model
    m = load_model(str(path or SERVED), device).eval()
    assert m.layer_tap.get("spk") == [TAP], m.layer_tap
    return m


@torch.no_grad()
def block_feats(model, audios, layer: int = TAP, att=ATT, batch_size: int = 8) -> list[np.ndarray]:
    """Output of encoder block ``layer`` (0-based) for each audio, (T_i, D) float32: the plain causal encode at ``att``
    (the masked offline forward = chunk-by-chunk cache-aware streaming), only the first layer + 1 blocks computed."""
    model.eval()
    enc = model.encoder
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(audios), batch_size):
        x, lens = model._pad(audios[i: i + batch_size])
        feats, flen = model.preprocessor(x.to(dev), lens.to(dev))
        h, hl = enc.pre_encode(feats, flen)
        if enc.xscale is not None:
            h = h * enc.xscale
        from audioforge.modules.fastconformer import chunked_attention_mask
        T = h.shape[1]
        am = chunked_attention_mask(T, hl, list(att))
        pm = torch.arange(T, device=h.device)[None] >= hl[:, None]
        for li in range(layer + 1):
            h, _, _ = enc.layers[li](h, am, pm)
        out += [h[j, : int(hl[j])].float().cpu().numpy() for j in range(len(hl))]
    return out


def spk_embed(model, feats_list) -> np.ndarray:
    """Served speaker head embeddings (N, 192) of whole block-4 frame sequences."""
    from audioforge.enrollment import ColumnEmbedder
    emb = ColumnEmbedder(model.heads["spk"])
    out = []
    for f in feats_list:
        out.append(emb(f[None], np.ones((1, len(f)), bool))[0])
    return np.stack(out) if out else np.zeros((0, 192), np.float32)


def save_npz(path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **kw)
    tmp.replace(path)


# --------------------------------------------------------------------------- corpora
def train_sets():
    """[(corpus, dataset)] of the train meetings (audio loaded lazily by the library: mmap)."""
    from audioforge.datasets.ami import AMI
    from audioforge.datasets.ami import subset as ami_subset
    from audioforge.datasets.icsi import ICSI
    from audioforge.datasets.icsi import subset as icsi_subset
    return [("ami", AMI(ami_subset({"train": 12})["train"], verbose=False)),
            ("icsi", ICSI(icsi_subset({"train": 12})["train"], verbose=False))]


def untimed_overlap(ds, m, a, b) -> bool:
    zones = getattr(ds, "zones", {}).get(m, [])
    return any(za < b and zb > a for za, zb in zones)


def single_segments(ds, m) -> dict:
    """{speaker name: [(a, b), ...]} single-speaker stretches (the library's asr-mode segments: one speaker, nobody
    else within 0.2 s, 1-15 s; ICSI: untimed zones dropped), in meeting seconds (without the 0.1 s padding)."""
    from audioforge.datasets.ami import _runs
    import bisect
    out = {}
    guard, MAXR = 0.2, 600.0
    for s, ws in ds.words[m].items():
        oth = sorted(iv for o, ivs in ds.acts[m].items() if o != s for iv in ivs)
        ostart = [iv[0] for iv in oth]
        segs = []
        for run in _runs(ws, ds.cfg["turn_gap"]):
            chunks, cur = [], []
            for w in run:
                if cur and w[1] - cur[0][0] > 15.0:
                    chunks.append(cur)
                    cur = []
                cur.append(w)
            chunks.append(cur)
            for c in chunks:
                a, b = c[0][0], max(w[1] for w in c)
                if b - a < 1.0 or not any(w[2] for w in c):
                    continue
                lo, hi = bisect.bisect_left(ostart, a - guard - MAXR), bisect.bisect_left(ostart, b + guard)
                if any(oth[i][1] > a - guard for i in range(lo, hi)):
                    continue
                if untimed_overlap(ds, m, a - 0.1, b + 0.1):
                    continue
                segs.append((float(a), float(b)))
        if segs:
            out[s] = segs
    return out


def clip_from(segs, L: float, rng: random.Random, exclude=None):
    """A list of (a, b) intervals totalling L seconds from ``segs`` (random segment, random sub-window; shorter segments
    are concatenated in random order), avoiding any segment that overlaps ``exclude`` = (a, b). None if not enough."""
    pool = [s for s in segs if exclude is None or not (s[0] < exclude[1] and s[1] > exclude[0])]
    if not pool or sum(b - a for a, b in pool) < L:
        return None
    order = pool[:]
    rng.shuffle(order)
    first = next((s for s in order if s[1] - s[0] >= L), None)
    if first is not None:
        a = rng.uniform(first[0], first[1] - L)
        return [(a, a + L)]
    out, left = [], L
    for a, b in order:
        d = min(b - a, left)
        out.append((a, a + d))
        left -= d
        if left <= 1e-6:
            break
    return out


def clip_audio(ds, m, ivs) -> np.ndarray:
    return np.concatenate([np.asarray(ds._clip(m, a, b), np.float32) for a, b in ivs])


# --------------------------------------------------------------------------- stage: trainfeats
def stage_trainfeats(a):
    from audioforge.datasets.ami import frames
    set_threads(2)
    model = load_served()
    t0 = time.time()
    left = 0
    for corpus, ds in train_sets():
        for m in ds.meetings:
            out = CACHE / "feats" / f"{corpus}_{m}.npz"
            if out.exists():
                continue
            if time.time() - t0 > a.budget:
                left += 1
                continue
            dur = ds.duration(m)
            starts = [s for s in np.arange(0.0, max(dur - CROP_SEC, 0.0) + 1e-6, HOP_SEC)
                      if not untimed_overlap(ds, m, s, s + CROP_SEC)]
            spks = sorted(ds.acts[m])
            F, A = [], []
            for i in range(0, len(starts), 8):
                xs = [np.asarray(ds._clip(m, s, s + CROP_SEC), np.float32) for s in starts[i: i + 8]]
                fs = block_feats(model, xs)
                for s, f in zip(starts[i: i + 8], fs):
                    T = len(f)
                    F.append(f.astype(np.float16))
                    A.append(np.stack([frames(ds.acts[m][sp], T, s) for sp in spks], 1).astype(np.uint8))
            Tm = min(len(f) for f in F)
            save_npz(out, feats=np.stack([f[:Tm] for f in F]), act=np.stack([x[:Tm] for x in A]),
                     starts=np.array(starts), speakers=np.array(spks))
            log(f"  {corpus} {m}: {len(starts)} crops x {Tm} frames, {len(spks)} speakers ({time.time() - t0:.0f}s)")
    log(f"[trainfeats] {left} left ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage: enroll
def stage_enroll(a):
    set_threads(2)
    model = load_served()
    tn = None
    if a.titanet:
        from audioforge.enrollment import TitaNetEmbedder
        tn = TitaNetEmbedder()
    t0 = time.time()
    left = 0
    for corpus, ds in train_sets():
        for m in ds.meetings:
            out = CACHE / "enroll" / f"{corpus}_{m}.npz"
            if out.exists() and (tn is None or "titanet" in np.load(out).files):
                continue
            if time.time() - t0 > a.budget:
                left += 1
                continue
            rng = random.Random(f"{corpus}_{m}")
            segs = single_segments(ds, m)
            spk, ivs_all, auds = [], [], []
            for s in sorted(segs):
                for _ in range(N_ENROLL):
                    L = rng.uniform(*ENROLL_SEC)
                    ivs = clip_from(segs[s], L, rng)
                    if ivs is None:
                        ivs = clip_from(segs[s], sum(b - a_ for a_, b in segs[s]), rng)
                    spk.append(s)
                    ivs_all.append(ivs)
                    auds.append(clip_audio(ds, m, ivs))
            fs = block_feats(model, auds)
            E = spk_embed(model, fs)
            kw = {}
            if tn is not None:
                kw["titanet"] = tn.model.embed(auds, 8).float().cpu().numpy()
            iv = np.full((len(ivs_all), 8, 2), -1.0)
            for i, x in enumerate(ivs_all):
                iv[i, : len(x)] = np.array(x[:8])
            save_npz(out, spk=np.array(spk), emb=E.astype(np.float32), ivs=iv,
                     dur=np.array([len(x) / SR for x in auds]), **kw)
            log(f"  {corpus} {m}: {len(spk)} clips, {len(segs)} speakers ({time.time() - t0:.0f}s)")
    log(f"[enroll] {left} left ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage: train
VAL_MEETINGS = ("ami_EN2003a", "icsi_Bsr001")  # held out of the selection runs only (the final run uses all 24)


class TrainData:
    """All cached crops in memory; ``batch`` samples (feats, labels, enrollment, has) as in research/IMPROVE_115M.md:
    target = a speaker active in the crop (p ``p_active``) or any enrollable speaker of the meeting, enrollment = one of
    that speaker's clips that does not overlap the crop (+-1 s), dropped with p ``p_drop`` (then target = any speech,
    other = 0, the plain-VAD fallback)."""

    def __init__(self, names, emb_key: str = "emb", p_active: float = 0.8, p_drop: float = 0.15, seed: int = 0):
        self.F, self.A, self.S, self.meet = [], [], [], []
        self.pools = []
        for k, n in enumerate(names):
            z = np.load(CACHE / "feats" / f"{n}.npz")
            e = np.load(CACHE / "enroll" / f"{n}.npz")
            spks = [str(x) for x in z["speakers"]]
            espk = np.array([str(x) for x in e["spk"]])
            pool = {}
            for j, sp in enumerate(spks):
                idx = np.nonzero(espk == sp)[0]
                if len(idx):
                    pool[j] = (e[emb_key][idx].astype(np.float32), e["ivs"][idx])
            self.pools.append(pool)
            self.F.append(torch.from_numpy(z["feats"]))
            self.A.append(z["act"])
            self.S.append(z["starts"])
        self.n = [len(f) for f in self.F]
        self.src = np.concatenate([[k] * n for k, n in enumerate(self.n)])
        self.row = np.concatenate([np.arange(n) for n in self.n])
        corpus = np.array([names[k].split("_")[0] for k in self.src])
        w = np.where(corpus == "ami", 0.5 / max(1, (corpus == "ami").sum()), 0.5 / max(1, (corpus != "ami").sum()))
        self.w = w / w.sum()
        self.p_active, self.p_drop = p_active, p_drop
        self.rng = np.random.default_rng(seed)

    def one(self, i):
        k, r = self.src[i], self.row[i]
        act = self.A[k][r].astype(np.float32)  # (T, S)
        pool = self.pools[k]
        a0 = float(self.S[k][r])
        active = [j for j in pool if act[:, j].any()]
        cand = active if (active and self.rng.random() < self.p_active) else list(pool)
        j = int(self.rng.choice(cand))
        E, ivs = pool[j]
        lo, hi = a0 - 1.0, a0 + CROP_SEC + 1.0
        okc = [c for c in range(len(E)) if not any((iv[0] >= 0) and iv[0] < hi and iv[1] > lo for iv in ivs[c])]
        has = bool(okc) and self.rng.random() >= self.p_drop
        if has:
            e = E[int(self.rng.choice(okc))]
            y = np.stack([act[:, j], np.delete(act, j, 1).max(1) if act.shape[1] > 1 else np.zeros(len(act))], 1)
        else:
            e = np.zeros(E.shape[1], np.float32)
            y = np.stack([act.max(1), np.zeros(len(act))], 1)
        return self.F[k][r], y.astype(np.float32), e, has

    def batch(self, B: int):
        idx = self.rng.choice(len(self.src), B, p=self.w)
        xs = [self.one(i) for i in idx]
        return (torch.stack([x[0] for x in xs]).float(), torch.from_numpy(np.stack([x[1] for x in xs])),
                torch.from_numpy(np.stack([x[2] for x in xs])), torch.tensor([x[3] for x in xs]))


def frame_prf(p, y, thr=0.5):
    """(precision, recall, f1, fa rate) of binary frame decisions p > thr against labels y > 0.5 (flattened)."""
    d, t = np.asarray(p) > thr, np.asarray(y) > 0.5
    tp, fp, fn = float((d & t).sum()), float((d & ~t).sum()), float((~d & t).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-9), fp / max(float((~t).sum()), 1)


def stage_train(a):
    """Head-only training on the cached block-4 features (MPS). Every tensor of the served model is untouched by
    construction (the head is a separate module; the built .afm is checked tensor by tensor in `build`)."""
    from audioforge.heads.tsvad import TSVADHead
    set_threads(2)
    dev = a.device
    names = sorted(p.stem for p in (CACHE / "feats").glob("*.npz"))
    val = [n for n in names if n in VAL_MEETINGS] if a.holdout else []
    tr = [n for n in names if n not in val]
    torch.manual_seed(a.seed)
    data = TrainData(tr, "titanet" if a.emb == "titanet" else "emb", p_drop=a.p_drop, seed=a.seed)
    vdata = TrainData(val, "titanet" if a.emb == "titanet" else "emb", p_drop=0.0, seed=123) if val else None
    head = TSVADHead(512, emb_dim=192, hidden=a.hidden, prenet=not a.no_prenet).to(dev)
    nparam = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    lens = torch.full((a.batch,), 201, device=dev)
    hist = []
    t0 = time.time()
    vb = [vdata.batch(64) for _ in range(8)] if vdata else []
    for step in range(1, a.steps + 1):
        head.train()
        x, y, e, has = data.batch(a.batch)
        x, y, e, has = x.to(dev), y.to(dev), e.to(dev), has.to(dev)
        if a.emb_noise > 0:
            e = e + a.emb_noise * torch.randn_like(e)
        l = head.loss(x, lens[: len(x)], {"tsvad_targets": y, "tsvad_enroll": e, "tsvad_has": has})
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps:
            rec = {"step": step, "loss": round(float(l), 4), "sec": round(time.time() - t0, 1)}
            if vb:
                head.eval()
                P, Y = [], []
                with torch.no_grad():
                    for x, y, e, has in vb:
                        P.append(head.decode(x.to(dev), None, e.to(dev), has.to(dev)).cpu().numpy())
                        Y.append(y.numpy())
                P, Y = np.concatenate(P), np.concatenate(Y)
                rec["val_f1_target"] = round(frame_prf(P[..., 0], Y[..., 0])[2], 4)
                rec["val_f1_other"] = round(frame_prf(P[..., 1], Y[..., 1])[2], 4)
                ov = (Y[..., 0] > 0.5) & (Y[..., 1] > 0.5)
                rec["val_recall_target_overlap"] = round(float((P[..., 0][ov] > 0.5).mean()), 4) if ov.any() else None
            hist.append(rec)
            log(json.dumps(rec))
    out = ROOT / "runs" / f"tsvad_{a.tag}.pt"
    cfg = {"type": "tsvad", "from_layers": [TAP], "emb_dim": 192, "hidden": a.hidden, "prenet": not a.no_prenet,
           "enroll_embedder": a.emb, "weight": 0.0}
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "cfg": cfg, "params": nparam,
                "args": vars(a), "train": tr, "val": val, "history": hist}, out)
    log(f"[train] {out} ({nparam} params, {time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- eval windows
def bench_windows(corpus: str):
    """(ext windows, meta, ds) of eot-bench v2: AMI dev 974 (eval_stage1.v2_data) / ICSI held-out 1312
    (bench_turn_icsi.icsi_data), 6 s-extended windows."""
    if corpus == "ami":
        import eval_stage1 as E
        base, ext, meta, ds, _ = E.v2_data()
    elif corpus == "ami_eval":  # the same windows cut from the AMI test (eval) meetings (research/FIXALL.md test audit)
        import eval_stage1 as E
        base, ext, meta, ds, _ = E.v2_data_split("eval")
    else:
        import bench_turn_icsi as M
        base, ext, meta, ds, _ = M.icsi_data()
    return ext, meta, ds


def wkey(ex) -> str:
    from audioforge.datasets import ext_tracks as xt
    return xt.example_key(ex)


def window_speakers(ds, ex) -> list[str]:
    """Speaker names of the window's spk_targets columns (column 0 = the primary), as turn_windows built them."""
    from audioforge.datasets.ami import spk_matrix
    off = getattr(ds, "speaker_offset", 0)
    prim = ds.speaker_ids[int(ex["speaker"]) - off]
    a = float(ex["start"])
    b = a + len(ex["audio"]) / SR
    T = len(ex["spk_act"])
    _, order, _ = spk_matrix(ds.acts[ex["meeting"]], T, a, b, 4, first=prim)
    assert order[0] == prim
    return list(order)


def stage_winfeats(a):
    set_threads(2)
    model = load_served()
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        d = WORK / corpus / "feat"
        d.mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not (d / f"{wkey(v)}.npy").exists()]
        n = 0
        for i in range(0, len(todo), 8):
            if time.time() - t0 > a.budget:
                break
            cc = todo[i: i + 8]
            for v, f in zip(cc, block_feats(model, [np.asarray(v["audio"], np.float32) for v in cc])):
                np.save(d / f"{wkey(v)}.npy", f.astype(np.float16))
            n += len(cc)
        log(f"  {corpus} winfeats: {n} done, {len(todo) - n} left ({time.time() - t0:.0f}s)")


VP_LENS = (5.0, 1.5)
VP_BANK = 8


def stage_vprints(a):
    """Voice prints per (window, column speaker): a bank of VP_BANK clips per (meeting, speaker) and length (seeded,
    anywhere in the meeting's single-speaker speech); a window takes the first bank clip that does not overlap
    [window start - 2 s, window end + 2 s]. Embedded by the served speaker head (block-4 feats of the clip alone) and
    TitaNet-L."""
    set_threads(2)
    from audioforge.enrollment import TitaNetEmbedder
    model = load_served()
    tn = TitaNetEmbedder()
    t0 = time.time()
    for corpus in a.corpora.split(","):
        out = WORK / corpus / "vprints.npz"
        if out.exists():
            continue
        ext, meta, ds = bench_windows(corpus)
        bank = {}  # (m, s, L) -> [(ivs, emb_spk, emb_tn)]
        need = sorted({(v["meeting"], s) for v in ext for s in window_speakers(ds, v)})
        segs_m = {m: single_segments(ds, m) for m in sorted({m for m, _ in need})}
        for m, s in need:
            segs = segs_m[m].get(s, [])
            for L in VP_LENS:
                clips = []
                for i in range(VP_BANK):
                    ivs = clip_from(segs, L, random.Random(f"{m}_{s}_{L}_{i}"))
                    if ivs is not None:
                        clips.append(ivs)
                auds = [clip_audio(ds, m, iv) for iv in clips]
                es = spk_embed(model, block_feats(model, auds)) if auds else np.zeros((0, 192), np.float32)
                et = tn.model.embed(auds, 8).float().cpu().numpy() if auds else np.zeros((0, 192), np.float32)
                bank[(m, s, L)] = list(zip(clips, es, et))
        log(f"  {corpus}: bank of {len(need)} speakers built ({time.time() - t0:.0f}s)")
        keys, S = [], 4
        E = {f"{b}_{L}": np.zeros((len(ext), S, 192), np.float32) for b in ("spk", "tn") for L in VP_LENS}
        H = {L: np.zeros((len(ext), S), bool) for L in VP_LENS}
        IV = {L: np.full((len(ext), S, 8, 2), -1.0) for L in VP_LENS}
        for w, v in enumerate(ext):
            keys.append(wkey(v))
            a0 = float(v["start"])
            b0 = a0 + len(v["audio"]) / SR
            for c, s in enumerate(window_speakers(ds, v)):
                for L in VP_LENS:
                    for ivs, es, et in bank[(v["meeting"], s, L)]:
                        if any(x < b0 + 2 and y > a0 - 2 for x, y in ivs):
                            continue
                        E[f"spk_{L}"][w, c], E[f"tn_{L}"][w, c], H[L][w, c] = es, et, True
                        IV[L][w, c, : len(ivs)] = np.array(ivs[:8])
                        break
        save_npz(out, keys=np.array(keys), **{k.replace(".", "p"): x for k, x in E.items()},
                 **{f"has_{L}".replace(".", "p"): x for L, x in H.items()},
                 **{f"ivs_{L}".replace(".", "p"): x for L, x in IV.items()})
        log(f"  {corpus} vprints: {len(ext)} windows, primary has 5 s print in {H[5.0][:, 0].mean():.3f} "
            f"({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- tracks, bindings, TS-VAD outputs
TRACKS = {"ami": SCRATCH / "trail6" / "work" / "tracks", "icsi": SCRATCH / "icsi_turn" / "work" / "tracks"}
TITANET_EMB = {"ami": SCRATCH / "primary" / "work", "icsi": WORK / "icsi"}  # spkemb_titanet/<key>.npz (5-frame grid)


def sf_track(corpus, ex) -> np.ndarray:
    import eval_stage1 as E
    p = np.load(TRACKS[corpus] / f"{wkey(ex)}.stream_rc.npy")
    T = len(ex["spk_act"])
    return np.stack([E._fit(p[:, j], T) for j in range(p.shape[1])], 1)


def win_feats(corpus, ex) -> np.ndarray:
    f = np.load(WORK / corpus / "feat" / f"{wkey(ex)}.npy").astype(np.float32)
    T = len(ex["spk_act"])
    if len(f) < T:
        f = np.concatenate([f, np.repeat(f[-1:], T - len(f), 0)])
    return f[:T]


def load_vprints(corpus):
    z = np.load(WORK / corpus / "vprints.npz")
    return {str(k): i for i, k in enumerate(z["keys"])}, {k: z[k] for k in z.files}


def vp_follow(emb, ok, e_vp):
    """Per-frame Sortformer column bound by a voice print known from the start: unbound (-1) until some column has
    recent speech (``ok``), then the closest such column (cosine) and from there enrollment.VoiceFollower (margin 0.1,
    hold 6 frames: the rule the server's VoiceBinder runs). Causal (frame t reads emb / ok of frame t = speech <= t - 1)."""
    from audioforge.enrollment import follow_voice
    T = len(ok)
    init = np.full(T, -1, np.int64)
    first = np.nonzero(ok.any(1))[0]
    if not len(first):
        return init
    t0 = int(first[0])
    s = np.where(ok[t0], emb[t0] @ np.asarray(e_vp, np.float32), -np.inf)
    col, _ = follow_voice(emb, ok, e_vp, int(np.argmax(s)), t0, init)
    return col


def spk_column_embeddings(model, feats, p):
    from audioforge.enrollment import ColumnEmbedder, recent_embeddings
    return recent_embeddings(feats, p, ColumnEmbedder(model.heads["spk"]))


def load_head(path, device="cpu"):
    from audioforge.heads.tsvad import TSVADHead
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = {k: v for k, v in ck["cfg"].items() if k not in ("type", "from_layers", "weight", "enroll_embedder")}
    h = TSVADHead(512, **c)
    h.load_state_dict(ck["state_dict"])
    return h.eval().to(device), ck


@torch.no_grad()
def tsvad_probs(head, feats, e) -> np.ndarray:
    """(T, 2) [P(target), P(other)] for one window and one voice print (None = no enrollment)."""
    x = torch.as_tensor(feats)[None]
    ee = None if e is None else torch.as_tensor(np.asarray(e, np.float32))[None]
    return head.decode(x, None, ee).numpy()[0]


def frame_classes(y) -> np.ndarray:
    """0 silence, 1 single speaker, 2 overlap (labels, all columns)."""
    n = (np.asarray(y) > 0.5).sum(1)
    return np.minimum(n, 2)


class FrameStats:
    """Counts for target-activity decisions (p > 0.5) split by frame class."""

    def __init__(self):
        self.c = {}

    def add(self, name, pt, yt, cls):
        d, t = np.asarray(pt) > 0.5, np.asarray(yt) > 0.5
        for cname, m in (("all", np.ones(len(t), bool)), ("single", cls == 1), ("overlap", cls == 2), ("silence", cls == 0)):
            r = self.c.setdefault(name, {}).setdefault(cname, np.zeros(4))
            r += [(d & t & m).sum(), (d & ~t & m).sum(), (~d & t & m).sum(), (~t & m).sum()]

    def summary(self) -> dict:
        out = {}
        for name, byc in self.c.items():
            o = {}
            for cname, (tp, fp, fn, neg) in byc.items():
                pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                o[cname] = {"f1": round(2 * pr * rc / max(pr + rc, 1e-9), 4), "precision": round(pr, 4),
                            "miss": round(1 - rc, 4) if tp + fn else None, "fa": round(fp / max(neg, 1), 4),
                            "target_frames": int(tp + fn), "nontarget_frames": int(neg)}
            out[name] = o
        return out


def stage_frame(a):
    """Frame target-activity metrics on the eot-bench v2 extended windows, every labelled column speaker with a voice
    print as the target (column 0 = the turn's primary), TS-VAD vs Sortformer bound by the same print."""
    import eval_stage1 as E
    set_threads(2)
    model = load_served()
    heads = {h.split("=")[0]: load_head(ROOT / h.split("=")[1])[0] for h in a.heads.split(",")}
    res = {}
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        kidx, VP = load_vprints(corpus)
        st = {"primary": FrameStats(), "all_speakers": FrameStats()}
        n_t = 0
        for w, v in enumerate(ext):
            k = kidx[wkey(v)]
            f = win_feats(corpus, v)
            p = sf_track(corpus, v)
            y = np.asarray(v["spk_targets"], np.float32)
            T = len(y)
            cls = frame_classes(y)
            emb_ok = spk_column_embeddings(model, f, p)
            tn_ok = None
            tpath = TITANET_EMB[corpus] / "spkemb_titanet" / f"{wkey(v)}.npz"
            if tpath.exists():
                tn_ok = E.v2_load_titanet(TITANET_EMB[corpus], v, T)
            nspk = len(window_speakers(ds, v))
            for c in range(nspk):
                yt = y[:, c]
                if not yt.any():
                    continue
                groups = ["primary", "all_speakers"] if c == 0 else ["all_speakers"]
                sys_ = {}
                for L in VP_LENS:
                    Ls = str(L).replace(".", "p")
                    if not VP[f"has_{Ls}"][k, c]:
                        continue
                    for hn, h in heads.items():
                        ek = "tn" if "titanet" in hn else "spk"
                        sys_[f"tsvad_{hn}_vp{Ls}"] = tsvad_probs(h, f, VP[f"{ek}_{Ls}"][k, c])[:, 0]
                    col = vp_follow(emb_ok[0], emb_ok[1], VP[f"spk_{Ls}"][k, c])
                    sys_[f"sortformer_vp_spk_vp{Ls}"] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, 3)], 0.0)
                    if tn_ok is not None:
                        col = vp_follow(tn_ok[0], tn_ok[1], VP[f"tn_{Ls}"][k, c])
                        sys_[f"sortformer_vp_titanet_vp{Ls}"] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, 3)], 0.0)
                if not sys_:
                    continue
                for hn, h in heads.items():
                    sys_[f"tsvad_{hn}_noenroll"] = tsvad_probs(h, f, None)[:, 0]
                oc = E.enroll_column(p, yt, 0, T)
                sys_["sortformer_oracle_column"] = p[:, oc]
                for name, pt in sys_.items():
                    for g in groups:
                        st[g].add(name, pt, yt, cls)
                n_t += 1
            if time.time() - t0 > a.budget:
                raise SystemExit("frame: over budget; raise --budget or split corpora")
        res[corpus] = {g: s_.summary() for g, s_ in st.items()}
        res[corpus]["n_windows"], res[corpus]["n_targets"] = len(ext), n_t
        log(f"  {corpus}: {n_t} targets ({time.time() - t0:.0f}s)")
    out = WORK / f"frame_{a.out_tag}.json"
    out.write_text(json.dumps(res, indent=1))
    log(f"[frame] -> {out}")


# --------------------------------------------------------------------------- Nemotron-3 arm (IMPROVE_115M "Nemotron-3 arm")
# NVIDIA Nemotron-3-Diarization at the served settings (audioforge.hub.diarizer_defaults + serve.Engine.make_diarizer:
# all 8 columns, max pooling of the 10 ms sub-frames, window mode with --diar-left 1, the served diarizer preset
# low_latency_032 = 0.32 s input buffer), run on the same eot-bench v2 extended windows as the Sortformer tracks
# (eval_stage1.v2_tracks: window audio + (C + R + 1) frames of real right padding, cropped to the window's frames).
N3_AFM = ROOT / "runs" / "nemo_nemotron3_diar.afm"
N3_PRESET = "low_latency_032"
N3_LEFT = 1
N3_POOL = "max"
DIAR_ARMS = {  # arm -> (tracks dir, TitaNet column-embedding work dir) per corpus; files <key>.stream_rc.npy
    "nemotron3": lambda c: (WORK / c / "n3" / "tracks", WORK / c / "n3"),
}
FRAME_ARM_SYSTEMS = ("tsvad_spk_vp5p0", "{arm}_vp_spk_vp5p0", "{arm}_vp_titanet_vp5p0", "{arm}_oracle_column")
FRAME_CLASSES = ("all", "single", "overlap", "silence")


def load_nemotron3(device: str = "cpu"):
    from audioforge.nemo_import import load_any
    dm = load_any(str(N3_AFM), device)
    for h in dm.heads.values():
        if hasattr(h, "pool"):
            h.pool = N3_POOL
    dname = next(k for k, v in dm.head_cfg.items() if v["type"] == "sortformer")
    return dm.eval(), dname


def stage_n3tracks(a):
    """Nemotron-3 streaming column posteriors (T, 8) of every extended window -> <work>/<corpus>/n3/tracks/, float32,
    each file read back and compared after the write. Resumable; one call <= --budget s."""
    from audioforge.server.constants import diar_preset
    from audioforge.streaming_diar import StreamingDiarizer
    set_threads(2)
    cfg = diar_preset(N3_PRESET)
    C, R = cfg["chunk_len"], cfg["chunk_right_context"]
    dm, dname = load_nemotron3()
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        d = DIAR_ARMS["nemotron3"](corpus)[0]
        d.mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not (d / f"{wkey(v)}.stream_rc.npy").exists()]
        n = 0
        with torch.no_grad():
            for v in todo:
                if time.time() - t0 > a.budget:
                    break
                m, s0 = v["meeting"], float(v["start"])
                b = s0 + len(v["audio"]) / SR
                x = ds._clip(m, s0, min(ds.duration(m), b + (C + R + 1) * 0.08))
                sd = StreamingDiarizer(dm, diar_head=dname, mode="window", enc_left_context=N3_LEFT, **cfg)
                sd.feed(np.asarray(x, np.float32), final=True)
                p = sd.all_probs.float().cpu().numpy()[: len(v["spk_act"])].astype(np.float32)
                q = d / f"{wkey(v)}.stream_rc.npy"
                tmp = q.with_name(q.stem + ".tmp.npy")
                np.save(tmp, p)
                if not np.array_equal(np.load(tmp), p):
                    raise SystemExit(f"read-back mismatch {tmp}")
                tmp.replace(q)
                n += 1
        left = len(todo) - n
        log(f"  {corpus} n3tracks: {n} done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            break


def stage_n3tnemb(a):
    """TitaNet-L look-back column embeddings on the Nemotron-3 tracks (eval_stage1.v2_embed_titanet, the same backend
    and defaults as the Sortformer arm's TitaNet following) -> <work>/<corpus>/n3/spkemb_titanet/."""
    import eval_stage1 as E
    from audioforge.enrollment import TitaNetEmbedder
    set_threads(2)
    tn = TitaNetEmbedder()
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        tdir, twork = DIAR_ARMS["nemotron3"](corpus)
        r = E.v2_embed_titanet(tn, ext, twork, a.budget - (time.time() - t0), tracks_dir=tdir)
        log(f"  {corpus} n3tnemb: {r}")
        if r["todo"]:
            break


def stage_n3check(a):
    """Read back every cached Nemotron-3 track / TitaNet file: present, loadable, (T, 8), finite, in [0, 1]."""
    import eval_stage1 as E
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        tdir, twork = DIAR_ARMS["nemotron3"](corpus)
        bad, act = [], []
        for v in ext:
            T = len(v["spk_act"])
            try:
                p = np.load(tdir / f"{wkey(v)}.stream_rc.npy")
                e, ok = E.v2_load_titanet(twork, v, T)
                good = (p.shape == (T, 8) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
                        and e.shape[:2] == (T, 8) and np.isfinite(e).all())
                act.append((p > 0.5).sum(0))
            except Exception as ex_:  # noqa: BLE001
                good = False
                log(f"    {wkey(v)}: {ex_}")
            if not good:
                bad.append(wkey(v))
        act = np.array(act)
        log(f"  {corpus} n3check: {len(ext) - len(bad)}/{len(ext)} ok; windows using column c (>= 1 active frame): "
            f"{(act > 0).mean(0).round(3).tolist() if len(act) else []}")
        if bad:
            raise SystemExit(f"{corpus}: {len(bad)} bad files, e.g. {bad[:3]}")


def framecount_path(corpus, arm, ex) -> Path:
    return WORK / corpus / f"framecounts_{arm}" / f"{wkey(ex)}.npz"


def stage_framearm(a):
    """stage_frame for one diarizer arm (--arm), resumable: per window, the (system, group, class, [tp, fp, fn, neg])
    counts of the 5 s-print TS-VAD track and of the arm's column bound by the same print (speaker head / TitaNet-L,
    vp_follow: the Sortformer arm's rule) and its oracle column, the targets exactly as stage_frame selects them."""
    import eval_stage1 as E
    set_threads(2)
    arm = a.arm
    systems = [s.format(arm=arm) for s in FRAME_ARM_SYSTEMS]
    model = load_served()
    head = load_head(ROOT / "runs" / "tsvad_spk.pt")[0]
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        kidx, VP = load_vprints(corpus)
        tdir, twork = DIAR_ARMS[arm](corpus)
        todo = [v for v in ext if not framecount_path(corpus, arm, v).exists()]
        n = 0
        for v in todo:
            if time.time() - t0 > a.budget:
                break
            k = kidx[wkey(v)]
            f = win_feats(corpus, v)
            y = np.asarray(v["spk_targets"], np.float32)
            T = len(y)
            p = np.load(tdir / f"{wkey(v)}.stream_rc.npy")
            p = np.stack([E._fit(p[:, j], T) for j in range(p.shape[1])], 1)
            S = p.shape[1]
            cls = frame_classes(y)
            emb_ok = spk_column_embeddings(model, f, p)
            tn_ok = E.v2_load_titanet(twork, v, T)
            cnt = np.zeros((len(systems), 2, 4, 4), np.int64)  # system, group (primary, all_speakers), class, counts
            n_t = np.zeros(2, np.int64)
            for c in range(len(window_speakers(ds, v))):
                yt = y[:, c]
                if not yt.any() or not VP["has_5p0"][k, c]:
                    continue
                sys_ = {systems[0]: tsvad_probs(head, f, VP["spk_5p0"][k, c])[:, 0]}
                col = vp_follow(emb_ok[0], emb_ok[1], VP["spk_5p0"][k, c])
                sys_[systems[1]] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, S - 1)], 0.0)
                col = vp_follow(tn_ok[0], tn_ok[1], VP["tn_5p0"][k, c])
                sys_[systems[2]] = np.where(col >= 0, p[np.arange(T), np.clip(col, 0, S - 1)], 0.0)
                sys_[systems[3]] = p[:, E.enroll_column(p, yt, 0, T)]
                gs = [0, 1] if c == 0 else [1]
                t = yt > 0.5
                for i, name in enumerate(systems):
                    d = sys_[name] > 0.5
                    for j, m in enumerate((np.ones(T, bool), cls == 1, cls == 2, cls == 0)):
                        row = [(d & t & m).sum(), (d & ~t & m).sum(), (~d & t & m).sum(), (~t & m).sum()]
                        for g in gs:
                            cnt[i, g, j] += row
                n_t[gs] += 1
            q = framecount_path(corpus, arm, v)
            save_npz(q, counts=cnt, n_targets=n_t)
            z = np.load(q)
            if not np.array_equal(z["counts"], cnt):
                raise SystemExit(f"read-back mismatch {q}")
            n += 1
        left = len(todo) - n
        log(f"  {corpus} framearm {arm}: {n} done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            break


def _f1_counts(c):
    """c (..., 4) [tp, fp, fn, neg] -> f1, precision, miss, fa arrays."""
    c = np.asarray(c, np.float64)
    tp, fp, fn, neg = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    pr = tp / np.maximum(tp + fp, 1)
    rc = tp / np.maximum(tp + fn, 1)
    return 2 * pr * rc / np.maximum(pr + rc, 1e-9), pr, 1 - rc, fp / np.maximum(neg, 1)


def stage_framearmreport(a):
    """Aggregate the per-window counts (FrameStats.summary format), check that the recomputed TS-VAD row equals the
    stored one, 1000-resample window bootstraps (seed 0) of F1 / precision / miss / FA and the paired F1 difference
    vs tsvad_spk_vp5p0 (same resampled windows), and add the arm's rows to runs/improve_115m.json["frame"] under new
    keys (existing values untouched)."""
    arm = a.arm
    systems = [s.format(arm=arm) for s in FRAME_ARM_SYSTEMS]
    path = ROOT / "runs" / "improve_115m.json"
    res = json.loads(path.read_text())
    new = {}
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        Z = [np.load(framecount_path(corpus, arm, v)) for v in ext]
        C = np.stack([z["counts"] for z in Z])  # (W, sys, group, class, 4)
        ntg = np.stack([z["n_targets"] for z in Z]).sum(0)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(ext), (a.n_boot, len(ext)))
        new[corpus] = {}
        for g, gname in enumerate(("primary", "all_speakers")):
            stored = res["frame"][corpus][gname]["tsvad_spk_vp5p0"]
            tot = C[:, :, g].sum(0)  # (sys, class, 4)
            boot = np.stack([C[ii, :, g].sum(0) for ii in idx])  # (B, sys, class, 4)
            f1b, prb, msb, fab = _f1_counts(boot)
            q = lambda x: [round(float(np.quantile(x, 0.025)), 4), round(float(np.quantile(x, 0.975)), 4)]
            rows = {}
            for i, name in enumerate(systems):
                o = {}
                for j, cname in enumerate(FRAME_CLASSES):
                    tp, fp, fn, neg = tot[i, j]
                    f1, pr, ms, fa = (float(x) for x in _f1_counts(tot[i, j]))
                    o[cname] = {"f1": round(f1, 4), "precision": round(pr, 4), "miss": round(ms, 4) if tp + fn else None,
                                "fa": round(fa, 4), "target_frames": int(tp + fn), "nontarget_frames": int(neg),
                                "f1_ci": q(f1b[:, i, j]), "precision_ci": q(prb[:, i, j]),
                                "miss_ci": q(msb[:, i, j]) if tp + fn else None, "fa_ci": q(fab[:, i, j])}
                    if i:
                        dlt = f1b[:, i, j] - f1b[:, 0, j]
                        o[cname]["paired_f1_minus_tsvad_spk_vp5p0"] = round(f1 - float(_f1_counts(tot[0, j])[0]), 4)
                        o[cname]["paired_f1_minus_tsvad_spk_vp5p0_ci"] = q(dlt)
                rows[name] = o
            chk = {c_: {k_: rows[systems[0]][c_][k_] for k_ in stored[c_]} for c_ in FRAME_CLASSES}
            if chk != stored:
                raise SystemExit(f"{corpus}/{gname}: recomputed tsvad_spk_vp5p0 differs from the stored row:\n{chk}\n{stored}")
            log(f"  {corpus} {gname}: tsvad_spk_vp5p0 reproduced exactly; targets {int(ntg[g])}")
            new[corpus][gname] = rows
    for corpus, byg in new.items():
        for gname, rows in byg.items():
            ref = rows[systems[0]]
            for name in systems[1:]:
                assert name not in res["frame"][corpus][gname], f"{name} already in the file"
                res["frame"][corpus][gname][name] = rows[name]
            # CIs of the reference row live under a separate key so the stored row stays identical
            res["frame"][corpus].setdefault("tsvad_spk_vp5p0_ci", {})[gname] = {
                c_: {k_: ref[c_][k_] for k_ in ("f1_ci", "precision_ci", "miss_ci", "fa_ci")} for c_ in FRAME_CLASSES}
    res[f"frame_{arm}_protocol"] = {
        "date": "2026-09-29", "model": str(N3_AFM.relative_to(ROOT)), "columns": 8, "pool": N3_POOL,
        "mode": "window", "diar_left": N3_LEFT, "preset": N3_PRESET, "input_buffer_ms": 320,
        "tracks": "eval_stage1.v2_tracks protocol (window + (C+R+1) frames real right padding, cropped)",
        "binding": "vp_follow on the same 5 s print (speaker head: ColumnEmbedder look-back; TitaNet-L: v2_embed_titanet "
                   "look-back, 5-frame grid), margin 0.1 / hold 6; oracle = eval_stage1.enroll_column on the labels",
        "bootstrap": f"{a.n_boot} window resamples, seed 0, 95 % percentile; paired = same resampled windows",
        "cache": str(WORK / "<corpus>" / "n3"), "note_ami": "Nemotron-3's model card lists AMI train + dev in its "
        "training data: its AMI dev rows are not held out", "sortformer_rows": "1.04 s card low-latency tracks "
        "(2026-09-27/28 run), not recomputed"}
    path.write_text(json.dumps(res, indent=1))
    log(f"[framearmreport] {arm} rows -> {path}")


def stage_tnemb(a):
    """TitaNet-L look-back column embeddings of the ICSI windows (eval_stage1.v2_embed_titanet, the §9 backend the
    server's VoiceBinder uses); AMI reuses EOT_BENCH_V2 §9's stored ones."""
    import eval_stage1 as E
    from audioforge.enrollment import TitaNetEmbedder
    set_threads(2)
    ext, meta, ds = bench_windows("icsi")
    r = E.v2_embed_titanet(TitaNetEmbedder(), ext, WORK / "icsi", a.budget, tracks_dir=TRACKS["icsi"])
    log(f"[tnemb] {r}")


# --------------------------------------------------------------------------- stage: bind (turn-bench inputs)
def bind_path(corpus, name, ex) -> Path:
    return WORK / corpus / f"bind_{name}" / f"{wkey(ex)}.npy"


def stage_bind(a):
    """Per window, the primary's 5 s voice print (speech elsewhere in the meeting) binds (i) the Sortformer column by
    following with the served speaker head (``vp_spk``) and with TitaNet-L (``vp_titanet``, the server's VoiceBinder
    backend), stored as per-frame columns; (ii) the TS-VAD head's [P(target), P(other)] (``tsvad_<name>``). No print
    (ICSI: 2 % of primaries have < 5 s of clean speech elsewhere) -> causal_dominant binding / enrollment-less TS-VAD."""
    import eval_stage1 as E
    set_threads(2)
    model = load_served()
    heads = {h.split("=")[0]: load_head(ROOT / h.split("=")[1])[0] for h in a.heads.split(",")}
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        kidx, VP = load_vprints(corpus)
        names = [b for b in a.bindings.split(",") if b in ("vp_spk", "vp_titanet") or b in [f"tsvad_{h}" for h in heads]]
        for n in names:
            (WORK / corpus / f"bind_{n}").mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not all(bind_path(corpus, n, v).exists() for n in names)]
        done = 0
        for v in todo:
            if time.time() - t0 > a.budget:
                break
            k = kidx[wkey(v)]
            has = bool(VP["has_5p0"][k, 0])
            f = win_feats(corpus, v)
            p = sf_track(corpus, v) if any(n.startswith("vp_") for n in names) else None  # the Sortformer track
            T = len(f)
            if "vp_spk" in names and not bind_path(corpus, "vp_spk", v).exists():
                col = vp_follow(*spk_column_embeddings(model, f, p), VP["spk_5p0"][k, 0]) if has else E.enroll_causal_dominant(p)
                np.save(bind_path(corpus, "vp_spk", v), col.astype(np.int64))
            if "vp_titanet" in names and not bind_path(corpus, "vp_titanet", v).exists():
                tp = TITANET_EMB[corpus] / "spkemb_titanet" / f"{wkey(v)}.npz"
                if tp.exists():
                    col = vp_follow(*E.v2_load_titanet(TITANET_EMB[corpus], v, T), VP["tn_5p0"][k, 0]) if has \
                        else E.enroll_causal_dominant(p)
                    np.save(bind_path(corpus, "vp_titanet", v), col.astype(np.int64))
            for hn, h in heads.items():
                q = bind_path(corpus, f"tsvad_{hn}", v)
                if f"tsvad_{hn}" in names and not q.exists():
                    ek = "tn" if "titanet" in hn else "spk"
                    np.save(q, tsvad_probs(h, f, VP[f"{ek}_5p0"][k, 0] if has else None).astype(np.float32))
            done += 1
        left = len(todo) - done
        log(f"  {corpus} bind: {done} done, {left} left ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage: scores
def binding_inputs(corpus, name, ex):
    """(act (T,), cols (T, S), prim) fed to the turn head for a binding."""
    import eval_stage1 as E
    if name.startswith("tsvad_"):
        P = np.load(bind_path(corpus, name, ex))
        cols = np.zeros((len(P), 4), np.float32)
        cols[:, :2] = P
        return P[:, 0].copy(), cols, 0
    p = sf_track(corpus, ex)
    if name == "oracle":
        oc = E.enroll_column(p, ex["spk_act"], ex["onset_frame"], ex["turn_end_frame"])
        return p[:, oc].copy(), p, oc
    if name == "causal_dominant":
        act, cols = E.bound_track(p, E.enroll_causal_dominant(p))
        return act, cols, 0
    col = np.load(bind_path(corpus, name, ex))
    act, cols = E.bound_track(p, col)
    return act, cols, 0


def score_path(corpus, name, ex) -> Path:
    return WORK / corpus / f"scores_{name}" / f"{wkey(ex)}.npy"


def stage_scores(a):
    """Served turn head (trail6, bit-identical to runs/stage1_turn_v3_trail6.afm) scores per binding (resumable);
    --model scores another .afm's turn head instead (e.g. single_model_distill's student; use a distinct binding name)."""
    import eval_stage1 as E
    set_threads(2)
    model = load_served(a.device, a.model)
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        for name in a.bindings.split(","):
            d = WORK / corpus / f"scores_{name}"
            d.mkdir(parents=True, exist_ok=True)
            todo = [v for v in ext if not score_path(corpus, name, v).exists()
                    and (name in ("oracle", "causal_dominant") or bind_path(corpus, name, v).exists())]
            todo.sort(key=lambda v: len(v["audio"]))
            done = 0
            bs = 8 if a.batch == 64 else a.batch
            for i in range(0, len(todo), bs):
                if time.time() - t0 > a.budget:
                    break
                cc = todo[i: i + bs]
                xs = [binding_inputs(corpus, name, v) for v in cc]
                sc = E.turn_scores_given_act(model, "turn", cc, [x[0] for x in xs], bs, [x[1] for x in xs],
                                             [x[2] for x in xs])
                for v, sv in zip(cc, sc):
                    np.save(score_path(corpus, name, v), sv.astype(np.float32))
                done += len(cc)
            log(f"  {corpus} scores {name}: {done} done, {len(todo) - done} left ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage: report (eot-bench v2)
SILERO = {"ami": SCRATCH / "baselines_turn" / "work" / "silero", "icsi": SCRATCH / "icsi_turn" / "work" / "silero"}
AMI_SCORES = {"oracle": SCRATCH / "trail6" / "work" / "scores_oracle__trail6",
              "causal_dominant": SCRATCH / "trail6" / "work" / "scores_causal_dominant__trail6"}
ICSI_SCORES = {"causal_dominant": SCRATCH / "icsi_turn" / "work" / "scores_causal_dominant__trail6"}
DYN = (75, 55, 2)  # dyn(Silero | head): required = clamp(T0 - a p, Tmin, T0) frames (BASELINES.md, AMI fit)
DYN_FROZEN = (0.998283, 4.957)  # the shipped hybrid_dyn point (theta, offset), fitted on the Sortformer-fed head
C_SF, R_SF = 6, 7
CHUNK = 2


def stored_scores(corpus, name, ex):
    q = score_path(corpus, name, ex)
    if q.exists():
        return np.load(q)
    ref = (AMI_SCORES if corpus == "ami" else ICSI_SCORES).get(name)
    if ref is None:
        raise FileNotFoundError(f"no stored {name} scores for {corpus}")
    return np.load(ref / f"{wkey(ex)}.npy")


def folds_of(corpus, convs):
    import eval_stage1 as E
    import bench_turn_icsi as M
    f0 = E.V2_DEV_FOLDS[0] if corpus == "ami" else M.ICSI_FOLDS[0]
    return np.array([0 if v["meeting"] in f0 else 1 for v in convs])


def score_systems(convs, meta, folds, singles, hybrids, fixed_pts, horizons, n_boot, pairs):
    """bench_turn_baselines.score_block with explicit folds (+ fixed operating points): cross-fitted <= 5 % per-turn FC
    (leave-meetings-out halves), bootstrap CIs, strata, paired bootstraps (seeds 0 / 1)."""
    import eval_stage1 as E
    from audioforge.conversation import (bootstrap_ci, eot_outcomes, eot_outcomes_or, floor_stratum, outcome_metrics,
                                         pause_runs)
    n = len(convs)
    on = np.array([v["onset_frame"] for v in convs])
    en = np.array([v["turn_end_frame"] for v in convs])
    strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
    groups = {"open": strata == "open", "taken": strata != "open"}
    pauses = [pause_runs(v["hes"]) for v in convs]
    res = {"n": n, "strata_counts": {g: int(m.sum()) for g, m in groups.items()}, "systems": {}, "paired": {}}
    outs = {}

    def pack(fc, lat, pf, npause):
        pt = outcome_metrics(fc, lat, pf, npause)
        pt.update(bootstrap_ci(fc, lat, n_boot, 0))
        pt["strata"] = {g: {**outcome_metrics(fc[m], lat[m], pf[m], npause[m]), **bootstrap_ci(fc[m], lat[m], n_boot, 0)}
                        for g, m in groups.items()}
        return pt

    for name, spec in list(singles.items()) + list(hybrids.items()):
        r = {}
        for hz, L in horizons.items():
            if name in hybrids:
                ha, ea, tb, eb = spec
                za, zb = E._emit_transform(ha, en, ea), E._emit_transform(tb, en, eb)
                ga, gb = E.v2_hybrid_grid(za, on, en, pauses), E.v2_hybrid_grid(zb, on, en, pauses)
                oc = eot_outcomes_or(za, zb, on, en, ga, gb, post_end_frames=L, pauses=pauses)
                ths = oc["ths"]
                oc = dict(oc, ths=np.arange(oc["fc"].shape[1], dtype=np.float64))
                fc, lat, pf, th = E.v2_crossfit(oc, folds, 0.05, "turn", tie_miss=True)
                th = {f: [float(x) if np.isfinite(x) else None for x in ths[int(round(j))]] for f, j in th.items()}
            else:
                sc, emit = spec
                z = E._emit_transform(sc, en, emit)
                allv = np.unique(np.concatenate([np.asarray(x, np.float64) for x in z]))
                grid = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
                oc = eot_outcomes(z, on, en, grid, post_end_frames=L, pauses=pauses)
                fc, lat, pf, _ = E.v2_crossfit(oc, folds, 0.05, "turn")
                th = {f: float(oc["ths"][E._select(oc, folds != f, 0.05, "turn")]) for f in (0, 1)}
            pt = pack(fc, lat, pf, oc["npause"])
            pt["thresholds_by_fold"] = th
            r[hz] = pt
            outs[(name, hz)] = (fc, lat)
        res["systems"][name] = r
    for name, (sa, ea, ta, sb, eb, tb) in fixed_pts.items():
        r = {}
        for hz, L in horizons.items():
            za, zb = E._emit_transform(sa, en, ea), E._emit_transform(sb, en, eb)
            oc = eot_outcomes_or(za, zb, on, en, [ta], [tb], post_end_frames=L, pauses=pauses)
            fc, lat, pf = oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0]
            r[hz] = pack(fc, lat, pf, oc["npause"])
            outs[(name, hz)] = (fc, lat)
        res["systems"][name] = r
    idx = np.random.default_rng(0).integers(0, n, (n_boot, n))
    for a_, b_ in pairs:
        for hz in horizons:
            if (a_, hz) not in outs or (b_, hz) not in outs:
                continue
            A, Bb = outs[(a_, hz)], outs[(b_, hz)]
            d = {"all": E._paired(A, Bb, idx)}
            for g in ("open", "taken"):
                sub = np.nonzero(groups[g])[0]
                gi = np.random.default_rng(1).integers(0, len(sub), (n_boot, len(sub)))
                d[g] = E._paired((A[0][sub], A[1][sub]), (Bb[0][sub], Bb[1][sub]), gi)
            res["paired"][f"{a_} - {b_} | {hz}"] = d
    return res


def dyn_track(sil, head):
    T0, a_, tmin = DYN
    return np.asarray(sil, np.float64) - np.clip(T0 - a_ * np.asarray(head, np.float64), tmin, T0)


def stage_report(a):
    import eval_stage1 as E
    import bench_turn_icsi as M
    t0 = time.time()
    es, eh = E._stream_emit(C_SF, R_SF), E._stream_emit(C_SF, R_SF, CHUNK)
    ec = lambda t: (t // CHUNK + 1) * CHUNK  # noqa: E731 - TS-VAD track and head on it: the encoder's 160 ms chunk
    ef = lambda t: t + 1  # noqa: E731
    out = json.loads((WORK / "report.json").read_text()) if (WORK / "report.json").exists() else {}
    for corpus in a.corpora.split(","):
        ext, meta, ds = bench_windows(corpus)
        convs = [dict(v) for v in ext]
        for v in convs:
            v.pop("audio", None)
        folds = folds_of(corpus, convs)
        T = [len(v["spk_act"]) for v in convs]
        on = [int(v["onset_frame"]) for v in convs]
        sil = [M.silero_timeout_track(np.load(SILERO[corpus] / f"{wkey(v)}.npy"), t)[:t] for v, t in zip(ext, T)]
        singles, hybrids, fixed = {}, {}, {}
        singles["silero_timeout"] = (sil, ef)
        singles["timeout_primary_labels"] = ([E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)], ef)
        bnames = ["oracle", "causal_dominant"] + [b for b in a.bindings.split(",") if b]
        for b in bnames:
            try:
                head = [stored_scores(corpus, b, v)[:t] for v, t in zip(ext, T)]
            except FileNotFoundError:
                log(f"  {corpus}: no scores for {b}, skipped")
                continue
            acts = [binding_inputs(corpus, b, v)[0][:t] for v, t in zip(ext, T)]
            e_to, e_h = (ec, ec) if b.startswith("tsvad_") else (es, eh)
            to = [E.silence_scores(x, E.arm_frame(x)) for x in acts]
            singles[f"timeout_{b}"] = (to, e_to)
            singles[f"head_{b}"] = (head, e_h)
            hybrids[f"hybrid_{b}"] = (head, e_h, to, e_to)
            dy = [dyn_track(s_, h) for s_, h in zip(sil, head)]
            hybrids[f"hybrid_dyn_{b}"] = (head, e_h, dy, e_h)
            fixed[f"hybrid_dyn_frozen_{b}"] = (head, e_h, DYN_FROZEN[0], dy, e_h, DYN_FROZEN[1])
        pairs = []
        for kind in ("timeout", "head", "hybrid", "hybrid_dyn", "hybrid_dyn_frozen"):
            for t_ in [b for b in bnames if b.startswith("tsvad_")]:
                for ref in ("vp_spk", "vp_titanet", "causal_dominant", "oracle"):
                    pairs.append((f"{kind}_{t_}", f"{kind}_{ref}"))
                if t_ != "tsvad_spk" and "tsvad_spk" in bnames:  # other prints / heads vs the 5 s offline print (A.2)
                    pairs.append((f"{kind}_{t_}", f"{kind}_tsvad_spk"))
            pairs.append((f"{kind}_vp_titanet", f"{kind}_causal_dominant"))
            pairs.append((f"{kind}_vp_spk", f"{kind}_causal_dominant"))
        res = score_systems(convs, meta, folds, singles, hybrids, fixed, {"2s": 25, "6s": 75}, a.n_boot, pairs)
        res["fold_counts"] = [int((folds == f).sum()) for f in (0, 1)]
        out[corpus] = res
        log(f"  {corpus} report done ({time.time() - t0:.0f}s)")
        (WORK / "report.json").write_text(json.dumps(out, indent=1))
    log(f"[report] -> {WORK / 'report.json'}")


def fmt_row(r):
    return r["6s"]


# --------------------------------------------------------------------------- stage: compute
def stage_compute(a):
    """CPU cost (2 threads) of the turn path's speaker-activity input: streaming Sortformer v2 (the served diarizer,
    both presets) vs the TS-VAD head on the shared encoder's block 4 (+ the voice print's one-off cost). Audio: 60 s of
    AMI dev (IS1008b from 600 s) in 160 ms blocks, as the server feeds it."""
    import gc
    import psutil
    from audioforge.datasets.ami import AMI
    from audioforge.serve import diar_preset, fast_conv
    from audioforge.streaming_diar import StreamingDiarizer
    from audioforge.train import load_model
    set_threads(2)
    proc = psutil.Process()
    ds = AMI(["IS1008b"], verbose=False)
    x = np.asarray(ds._clip("IS1008b", 600.0, 660.0), np.float32)
    blk = 2560
    res = {"audio_sec": 60.0, "threads": 2, "block_ms": 160}
    gc.collect()
    r0 = proc.memory_info().rss / 2 ** 20
    dm = load_model(str(ROOT / "runs" / "nemo_sortformer_v2.afm"), "cpu").eval()
    fast_conv(dm)
    r1 = proc.memory_info().rss / 2 ** 20
    dname = next(k for k, v in dm.head_cfg.items() if v["type"] == "sortformer")
    res["sortformer"] = {"params_M": round(sum(p.numel() for p in dm.parameters()) / 1e6, 1),
                         "rss_load_mb": round(r1 - r0, 1)}
    for cfg in ("low_latency", "low_latency_032"):
        sd = StreamingDiarizer(dm, dname, mode="window", enc_left_context=188, **diar_preset(cfg))
        with torch.no_grad():
            sd.feed(x[:blk * 10], final=False)  # warm-up
            sd = StreamingDiarizer(dm, dname, mode="window", enc_left_context=188, **diar_preset(cfg))
            t = time.perf_counter()
            for i in range(0, len(x), blk):
                sd.feed(x[i: i + blk], final=i + blk >= len(x))
            dt = time.perf_counter() - t
        res["sortformer"][cfg] = {"rtf": round(dt / 60.0, 4), "ms_per_160ms_block": round(dt / (len(x) / blk) * 1000, 2)}
        log(f"  sortformer {cfg}: RTF {dt / 60:.4f}")
    res["sortformer"]["rss_after_run_mb"] = round(proc.memory_info().rss / 2 ** 20 - r0, 1)
    del dm, sd
    gc.collect()
    model = load_served()
    head, ck = load_head(ROOT / a.heads.split(",")[0].split("=")[1])
    f = block_feats(model, [x])[0]
    st = head.init_stream(np.random.randn(192).astype(np.float32))
    with torch.no_grad():
        t = time.perf_counter()
        for i in range(0, len(f), 2):
            head.step(torch.as_tensor(f[None, i: i + 2]), st)
        dt = time.perf_counter() - t
    res["tsvad"] = {"params": int(sum(p.numel() for p in head.parameters())),
                    "param_mb_fp32": round(sum(p.numel() for p in head.parameters()) * 4 / 2 ** 20, 2),
                    "rtf": round(dt / 60.0, 5), "ms_per_160ms_block": round(dt / (len(f) / 2) * 1000, 3),
                    "extra_encoder_cost": "none: reads block 4 of the ASR pass (hid[3]) that already runs"}
    aud5 = x[: 5 * SR]
    t = time.perf_counter()
    for _ in range(5):
        spk_embed(model, block_feats(model, [aud5]))
    res["voiceprint_spk_head_5s_ms"] = round((time.perf_counter() - t) / 5 * 1000, 1)
    t = time.perf_counter()
    for _ in range(5):
        spk_embed(model, [f[:62]])
    res["voiceprint_spk_head_from_stream_feats_ms"] = round((time.perf_counter() - t) / 5 * 1000, 2)
    out = WORK / "compute.json"
    out.write_text(json.dumps(res, indent=1))
    log(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--budget", type=float, default=520.0)
    ap.add_argument("--titanet", action="store_true")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--corpora", default="ami,icsi")
    ap.add_argument("--heads", default="spk=runs/tsvad_spk.pt", help="name=path[,name=path]; 'titanet' in the name = TitaNet prints")
    ap.add_argument("--out-tag", default="main")
    ap.add_argument("--model", default=None, help="scores: the .afm whose turn head scores (default the served model)")
    ap.add_argument("--bindings", default="tsvad_spk,vp_spk,vp_titanet")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="spk")
    ap.add_argument("--emb", default="spk", choices=("spk", "titanet"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64, help="train batch; scores: 8 is used when left at 64")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-prenet", action="store_true")
    ap.add_argument("--p-drop", type=float, default=0.15)
    ap.add_argument("--emb-noise", type=float, default=0.0)
    ap.add_argument("--holdout", action="store_true", help="hold out VAL_MEETINGS (selection runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="nemotron3", help="framearm / framearmreport: diarizer arm (DIAR_ARMS)")
    a, rest = ap.parse_known_args()
    fn = globals().get(f"stage_{a.stage}")
    if fn is None:
        raise SystemExit(f"unknown stage {a.stage}")
    fn(a) if not rest else fn(a, rest)


if __name__ == "__main__":
    main()
