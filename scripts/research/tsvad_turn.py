"""research/IMPROVEMENTS.md section 2: a predictive turn head trained on TS-VAD tracks (heads only, frozen encoder).

The head is ``uc_turn_head.UCTurnHead`` (scripts/research/uc_turn_head.py) (reused from the user-channel draft): per 80 ms frame it reads the
frozen encoder's TOP layer of the ASR pass (no speaker-conditioned second pass), the TS-VAD track [P(target),
P(other)] as its two activity columns, their duration counters and the causal standardised log-RMS, and predicts the
target's future activity in disjoint bins (0-240, 240-400, 400-640, 640-1040, 1040-2000 ms), P(target quiet for 1.04 /
2 s) and the others' bins (DYADIC section 8's multi-horizon target). Decision scores are read from those outputs.

Training data (AMI train 12 + ICSI train 12 meetings, the TS-VAD head's own 16 s crops, data/cache/tsvad/feats):
the TS-VAD tracks must look like held-out tracks, so they come from two CROSS-FITTED TS-VAD heads (each trained on
the other half of the 24 meetings, the recipe of scripts/research/tsvad.py train) with a random voice print of the
target from elsewhere in the meeting (the enrolment pools); 10 % of crops use the no-print track (target = any
speech), as served before a print exists.

Stages (CPU 2 threads unless noted; each call <= --budget s, resumable):
  xfit       two TS-VAD heads on meeting halves -> <scratch>/tsvad_turn/tsvad_xfit_{0,1}.pt
  trainfeats top-layer feats (encoder blocks 5-17 continued from the cached block-4 crops), raw log-RMS, and per
             speaker slot the held-out TS-VAD track -> <cache>/tsvad_turn/<corpus>_<meeting>.npz
  winfeats   the same for the eot-bench v2 windows (AMI dev 974 / ICSI held-out 1312; TS-VAD tracks = the stored
             bind_tsvad_spk of research/IMPROVE_115M.md A.2) -> <scratch>/tsvad_turn/<corpus>/{top,rms}/<key>.npy
  train      (MPS, <= 20 min per call, checkpointed) -> <scratch>/tsvad_turn/head_<tag>.pt
  scores     per eval window the head's outputs -> <scratch>/tsvad_turn/<corpus>/out_<tag>/<key>.npy
  report     eot-bench v2 (tsvad.score_systems: cross-fitted <= 5 % per-turn FC, 1000 bootstraps, paired vs the
             served head on the same TS-VAD track) -> runs/tsvad_turn.json
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
import tsvad as T  # noqa: E402

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
CACHE = SSD / "cache" / "tsvad_turn"
WORK = SSD / "scratch" / "tsvad_turn"
OUT = ROOT / "runs" / "tsvad_turn.json"
SR = 16000
FS = 1280
CROP_T = 201


def log(*a):
    print(*a, flush=True)


def names():
    return sorted(p.stem for p in (T.CACHE / "feats").glob("*.npz"))


def folds():
    """Two halves of the 24 train meetings, balanced by corpus (alternating in sorted order)."""
    ns = names()
    out = {0: [], 1: []}
    for c in ("ami", "icsi"):
        for i, n in enumerate([n for n in ns if n.startswith(c)]):
            out[i % 2].append(n)
    return out


def check_readback(path: Path, key: str):
    """exFAT caveat: a large file must read back non-zero."""
    z = np.load(path)
    x = z[key]
    if not np.any(x.reshape(-1)[: 100000].astype(np.float32)) or not np.any(x.reshape(-1)[-100000:].astype(np.float32)):
        raise IOError(f"{path}: {key} reads back zero-filled")


# --------------------------------------------------------------------------- xfit
def stage_xfit(a):
    """Train TS-VAD heads on each half (tsvad.stage_train's loop with an explicit meeting list), CPU."""
    from audioforge.heads.tsvad import TSVADHead
    T.set_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    fo = folds()
    for k in (0, 1):
        out = WORK / f"tsvad_xfit_{k}.pt"
        if out.exists():
            continue
        tr = fo[1 - k]  # head k is trained WITHOUT fold k's meetings -> used for fold k's tracks
        torch.manual_seed(k)
        data = T.TrainData(tr, "emb", p_drop=0.15, seed=k)
        head = TSVADHead(512, emb_dim=192, hidden=128, prenet=True)
        opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.steps, pct_start=0.05)
        lens = torch.full((64,), 201)
        t0 = time.time()
        for step in range(1, a.steps + 1):
            head.train()
            x, y, e, has = data.batch(64)
            l = head.loss(x, lens, {"tsvad_targets": y, "tsvad_enroll": e, "tsvad_has": has})
            opt.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 500 == 0:
                log(f"  xfit {k} step {step} loss {float(l):.4f} ({time.time() - t0:.0f}s)")
        cfg = {"type": "tsvad", "from_layers": [T.TAP], "emb_dim": 192, "hidden": 128, "prenet": True,
               "enroll_embedder": "spk", "weight": 0.0}
        torch.save({"state_dict": head.state_dict(), "cfg": cfg, "train": tr, "heldout": fo[k]}, out)
        log(f"[xfit] {out} ({time.time() - t0:.0f}s)")
        if time.time() - t0 > a.budget * 0.5:
            break


# --------------------------------------------------------------------------- features
@torch.no_grad()
def continue_top(model, b4: np.ndarray, lens=None, att=T.ATT) -> np.ndarray:
    """Encoder blocks 5..17 on block-4 outputs (B, T, D) -> the top layer (B, T, D) float32 (masked like forward)."""
    from audioforge.modules.fastconformer import chunked_attention_mask
    enc = model.encoder
    h = torch.as_tensor(np.asarray(b4, np.float32))
    B, Tn, _ = h.shape
    hl = torch.full((B,), Tn) if lens is None else torch.as_tensor(lens)
    am = chunked_attention_mask(Tn, hl, list(att))
    pm = torch.arange(Tn)[None] >= hl[:, None]
    for li in range(T.TAP + 1, len(enc.layers)):
        h, _, _ = enc.layers[li](h, am, pm)
    return h.masked_fill(pm[..., None], 0.0).numpy()


def log_rms(x: np.ndarray, Tn: int) -> np.ndarray:
    from uc_turn_head import frame_log_rms_np
    return frame_log_rms_np(np.asarray(x, np.float32), Tn).astype(np.float32)


def train_ds(corpus):
    for c, ds in T.train_sets():
        if c == corpus:
            return ds


def stage_trainfeats(a):
    T.set_threads(2)
    model = T.load_served()
    heads = {k: T.load_head(WORK / f"tsvad_xfit_{k}.pt")[0] for k in (0, 1)}
    fo = folds()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dss = {}
    left = 0
    for n in names():
        out = CACHE / f"{n}.npz"
        if out.exists():
            continue
        if time.time() - t0 > a.budget * 0.6:
            left += 1
            continue
        corpus, m = n.split("_", 1)
        if corpus not in dss:
            dss[corpus] = train_ds(corpus)
        ds = dss[corpus]
        z = np.load(T.CACHE / "feats" / f"{n}.npz")
        e = np.load(T.CACHE / "enroll" / f"{n}.npz")
        F, starts, spks = z["feats"], z["starts"], [str(x) for x in z["speakers"]]
        head = heads[0 if n in fo[0] else 1]
        espk = np.array([str(x) for x in e["spk"]])
        rng = np.random.default_rng(abs(hash(n)) % 2 ** 31)
        N, Tn, _ = F.shape
        top = np.zeros((N, Tn, 512), np.float16)
        rms = np.zeros((N, Tn), np.float32)
        trk = np.zeros((N, len(spks), Tn, 2), np.float16)
        has = np.zeros((N, len(spks)), bool)
        null = np.zeros((N, Tn, 2), np.float16)
        for i in range(0, N, 8):
            top[i: i + 8] = continue_top(model, F[i: i + 8]).astype(np.float16)
            for r in range(i, min(i + 8, N)):
                s0 = float(starts[r])
                rms[r] = log_rms(ds._clip(m, s0, s0 + T.CROP_SEC), Tn)
                x = F[r].astype(np.float32)
                null[r] = T.tsvad_probs(head, x, None).astype(np.float16)
                for j, sp in enumerate(spks):
                    idx = np.nonzero(espk == sp)[0]
                    ok = [c for c in idx if not any(iv[0] >= 0 and iv[0] < s0 + T.CROP_SEC + 1 and iv[1] > s0 - 1
                                                    for iv in e["ivs"][c])]
                    if not ok:
                        continue
                    c = int(rng.choice(ok))
                    trk[r, j] = T.tsvad_probs(head, x, e["emb"][c]).astype(np.float16)
                    has[r, j] = True
        T.save_npz(out, top=top, rms=rms, trk=trk, has=has, null=null, act=z["act"], starts=starts,
                   speakers=np.array(spks))
        check_readback(out, "top")
        log(f"  {n}: {N} crops ({time.time() - t0:.0f}s)")
    log(f"[trainfeats] {left} left ({time.time() - t0:.0f}s)")


CORE_0P6B = SSD / "runs" / "nemo_nemotron_speech_streaming_en_0.6b.afm"


def win_paths(corpus, key, core="115m"):
    top = "top" if core == "115m" else f"top_{core}"
    return WORK / corpus / top / f"{key}.npy", WORK / corpus / "rms" / f"{key}.npy"


def core_crop_path(n, core):
    return CACHE / (f"{n}.npz" if core == "115m" else f"{core}_{n}.npz")


@torch.no_grad()
def encode_top(model, audios, device):
    """Top layer at [70,1] (masked offline forward = cache-aware streaming) of raw audio, per item (T_i, D) float16."""
    x, lens = model._pad([np.asarray(z, np.float32) for z in audios])
    enc, elen = model.encode(x.to(device), lens.to(device), [70, 1])
    return [enc[j, : int(elen[j])].float().cpu().numpy().astype(np.float16) for j in range(len(audios))]


def stage_topfeats(a):
    """--core 0p6b: the 0.6B core's top layer on the same training crops (from audio) and eval windows; the TS-VAD
    tracks, labels and energy are shared with the 115M files (core-independent inputs)."""
    from audioforge.train import load_model
    torch.set_num_threads(2)
    assert a.core != "115m"
    m = load_model(str(CORE_0P6B), a.device).eval()
    t0 = time.time()
    dss = {}
    for n in names():
        out = core_crop_path(n, a.core)
        if out.exists():
            continue
        if time.time() - t0 > a.budget * 0.6:
            log("budget: rerun")
            return
        corpus, mt = n.split("_", 1)
        if corpus not in dss:
            dss[corpus] = train_ds(corpus)
        z = np.load(CACHE / f"{n}.npz")
        starts, Tn = z["starts"], z["top"].shape[1]
        tops = []
        for i in range(0, len(starts), 8):
            auds = [dss[corpus]._clip(mt, float(s0), float(s0) + T.CROP_SEC) for s0 in starts[i:i + 8]]
            for f in encode_top(m, auds, a.device):
                g = np.zeros((Tn, f.shape[1]), np.float16)
                g[: min(Tn, len(f))] = f[:Tn]
                tops.append(g)
        T.save_npz(out, top=np.stack(tops))
        check_readback(out, "top")
        log(f"  {a.core} {n}: {len(tops)} crops ({time.time() - t0:.0f}s)")
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        (WORK / corpus / f"top_{a.core}").mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not win_paths(corpus, T.wkey(v), a.core)[0].exists()]
        todo.sort(key=lambda v: len(v["audio"]))
        for i in range(0, len(todo), 8):
            if time.time() - t0 > a.budget:
                log(f"  {corpus}: {len(todo) - i} windows left (budget)")
                return
            cc = todo[i:i + 8]
            for v, f in zip(cc, encode_top(m, [v["audio"] for v in cc], a.device)):
                np.save(win_paths(corpus, T.wkey(v), a.core)[0], f)
        log(f"  {corpus} top_{a.core}: done ({time.time() - t0:.0f}s)")


def stage_winfeats(a):
    T.set_threads(2)
    model = T.load_served()
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        for sub in ("top", "rms"):
            (WORK / corpus / sub).mkdir(parents=True, exist_ok=True)
        todo = [v for v in ext if not win_paths(corpus, T.wkey(v))[0].exists()]
        todo.sort(key=lambda v: len(v["audio"]))
        n = 0
        for i in range(0, len(todo), 8):
            if time.time() - t0 > a.budget:
                break
            cc = todo[i: i + 8]
            fs = [np.load(T.WORK / corpus / "feat" / f"{T.wkey(v)}.npy").astype(np.float32) for v in cc]
            L = max(len(f) for f in fs)
            pad = np.zeros((len(fs), L, 512), np.float32)
            for j, f in enumerate(fs):
                pad[j, : len(f)] = f
            tops = continue_top(model, pad, [len(f) for f in fs])
            for j, (v, f) in enumerate(zip(cc, fs)):
                pt, pr = win_paths(corpus, T.wkey(v))
                np.save(pr, log_rms(v["audio"], len(f)))
                np.save(pt, tops[j, : len(f)].astype(np.float16))
            n += len(cc)
        log(f"  {corpus} winfeats: {n} done, {len(todo) - n} left ({time.time() - t0:.0f}s)")


def stage_eotlabels(a):
    """End-of-turn labels per crop and speaker slot, eot-bench's convention (AMI.turn_examples): for each non-
    backchannel turn of the slot's speaker, 0 on [turn start, turn end), 1 on [turn end, min(end + 2 s, the same
    speaker's next turn start)); -1 (no label) elsewhere -> <cache>/tsvad_turn/eot_<name>.npy (N, S, T) int8."""
    t0 = time.time()
    dss = {}
    for n in names():
        out = CACHE / f"eot_{n}.npy"
        if out.exists():
            continue
        corpus, m = n.split("_", 1)
        if corpus not in dss:
            dss[corpus] = train_ds(corpus)
        ds = dss[corpus]
        z = np.load(CACHE / f"{n}.npz")
        starts, spks, Tn = z["starts"], [str(x) for x in z["speakers"]], z["top"].shape[1]
        E = np.full((len(starts), len(spks), Tn), -1, np.int8)
        turns = [t for t in ds.turns[m] if not t["bc"]]
        by = {}
        for t in ds.turns[m]:
            by.setdefault(t["speaker"], []).append(t)
        nxt = {}
        for sp, ts in by.items():
            ts = sorted(ts, key=lambda t: t["start"])
            for j, t in enumerate(ts):
                nxt[(sp, t["start"])] = ts[j + 1]["start"] if j + 1 < len(ts) else 1e9
        for r, s0 in enumerate(starts):
            s0 = float(s0)
            for t in turns:
                if t["end"] + 2.0 < s0 or t["start"] > s0 + T.CROP_SEC or t["speaker"] not in spks:
                    continue
                j = spks.index(t["speaker"])
                f = lambda x: int(np.clip(round((x - s0) / 0.08), 0, Tn))  # noqa: E731
                E[r, j, f(t["start"]): f(t["end"])] = 0
                E[r, j, f(t["end"]): f(min(t["end"] + 2.0, nxt[(t["speaker"], t["start"])]))] = 1
        np.save(out, E)
        log(f"  eot {n}: {(E == 1).mean():.3f} pos / {(E == 0).mean():.3f} neg ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- train
class Crops:
    def __init__(self, ns, p_null=0.1, seed=0, core="115m"):
        self.Z = [dict(np.load(CACHE / f"{n}.npz")) for n in ns]
        if core != "115m":  # the other core's top layer, same crops / tracks / labels / energy
            for z, n in zip(self.Z, ns):
                z["top"] = np.load(core_crop_path(n, core))["top"]
        for z, n in zip(self.Z, ns):
            p = CACHE / f"eot_{n}.npy"
            z["eot"] = np.load(p) if p.exists() else None
        self.idx = [(k, r) for k, z in enumerate(self.Z) for r in range(len(z["top"]))]
        corpus = np.array([ns[k].split("_")[0] for k, _ in self.idx])
        w = np.where(corpus == "ami", 0.5 / max(1, (corpus == "ami").sum()), 0.5 / max(1, (corpus != "ami").sum()))
        self.w = w / w.sum()
        self.p_null = p_null
        self.rng = np.random.default_rng(seed)

    def one(self, i):
        k, r = self.idx[i]
        z = self.Z[k]
        act = z["act"][r].astype(np.float32)
        slots = [j for j in range(act.shape[1]) if z["has"][r, j]]
        active = [j for j in slots if act[:, j].any()]
        eot = np.full(len(act), -1, np.float32)
        if not slots or self.rng.random() < self.p_null:
            trk = z["null"][r].astype(np.float32)
            user, agent = act.max(1), np.zeros(len(act), np.float32)
        else:
            j = int(self.rng.choice(active if active and self.rng.random() < 0.9 else slots))
            trk = z["trk"][r, j].astype(np.float32)
            user = act[:, j]
            agent = np.delete(act, j, 1).max(1) if act.shape[1] > 1 else np.zeros(len(act), np.float32)
            if z["eot"] is not None:
                eot = z["eot"][r, j].astype(np.float32)
        return z["top"][r].astype(np.float32), trk, z["rms"][r], user, agent, eot

    def batch(self, B):
        xs = [self.one(i) for i in self.rng.choice(len(self.idx), B, p=self.w)]
        return [torch.from_numpy(np.stack([x[q] for x in xs])) for q in range(6)]


def head_cfg(a):
    return dict(hidden=a.hidden, layers=a.layers, energy=not a.no_energy, dropout=0.1,
                d_enc=512 if a.core == "115m" else 1024, eot=a.eot)


def stage_train(a):
    from uc_turn_head import UCTurnHead, save_head
    dev = a.device
    torch.set_num_threads(2)
    ck_path = WORK / f"head_{a.tag}.ckpt"
    out = WORK / f"head_{a.tag}.pt"
    if out.exists():
        log(f"{out} exists")
        return
    ns = names()
    val = [n for n in ns if n in T.VAL_MEETINGS] if a.holdout else []
    tr = [n for n in ns if n not in val]
    data = Crops(tr, a.p_null, a.seed, a.core)
    head = UCTurnHead(**head_cfg(a)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    step0, hist = 0, []
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step0, hist = ck["step"], ck["hist"]
        data.rng = np.random.default_rng(a.seed + step0)
        log(f"resumed at step {step0}")
    lens = torch.full((a.batch,), CROP_T, device=dev)
    t0 = time.time()
    for step in range(step0 + 1, a.steps + 1):
        head.train()
        enc, trk, rms, user, agent, eot = [x.to(dev) for x in data.batch(a.batch)]
        o, _ = head(enc, trk, rms, None)
        l, parts = head.loss(o, user, agent, lens[: len(enc)], eot=eot if a.eot else None)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            hist.append({"step": step, "loss": round(float(l), 4), **{k: round(v, 4) for k, v in parts.items()},
                         "sec": round(time.time() - t0, 1)})
            log(json.dumps(hist[-1]))
        if step % 500 == 0 or step == a.steps or time.time() - t0 > a.budget:
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "hist": hist}, ck_path)
            if time.time() - t0 > a.budget and step < a.steps:
                log(f"budget: checkpoint at step {step}")
                return
    save_head(head.cpu(), out, {"args": vars(a), "train": tr, "hist": hist})
    log(f"[train] {out}")


# --------------------------------------------------------------------------- scores / report
ENROLL_WORK = SSD / "scratch" / "tsvad_enroll"


def eval_track(corpus, v, track="tsvad_spk"):
    """The TS-VAD track fed at evaluation: the A.2 5 s print track (default) or an experiment-4 variant
    (scripts/research/tsvad_enroll.py -> <scratch>/tsvad_enroll/<corpus>/track_<name>/<key>.npy)."""
    if track == "tsvad_spk":
        return np.load(T.bind_path(corpus, "tsvad_spk", v)).astype(np.float32)
    return np.load(ENROLL_WORK / corpus / f"track_{track}" / f"{T.wkey(v)}.npy").astype(np.float32)


def out_dir(corpus, tag, track="tsvad_spk"):
    return WORK / corpus / (f"out_{tag}" if track == "tsvad_spk" else f"out_{tag}__{track}")


def stage_scores(a):
    from uc_turn_head import load_head
    torch.set_num_threads(2)
    head = load_head(WORK / f"head_{a.tag}.pt")
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        d = out_dir(corpus, a.tag, a.track)
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for v in ext:
            k = T.wkey(v)
            q = d / f"{k}.npy"
            if q.exists():
                continue
            if time.time() - t0 > a.budget:
                break
            pt, pr = win_paths(corpus, k, a.core)
            top = np.load(pt).astype(np.float32)
            trk = eval_track(corpus, v, a.track)[: len(top)]
            rms = np.load(pr)
            Tn = min(len(top), len(trk), len(rms))
            with torch.no_grad():
                o, _ = head(torch.from_numpy(top[None, :Tn]), torch.from_numpy(trk[None, :Tn]),
                            torch.from_numpy(rms[None, :Tn]), None)
                p = head.probs(o)
            cols = [p["user_bins"], p["user_quiet"], p["agent_bins"]] + ([p["eot"]] if "eot" in p else [])
            np.save(q, torch.cat(cols, 1).numpy().astype(np.float32))
            n += 1
        log(f"  {corpus} scores {a.tag}: {n} new ({time.time() - t0:.0f}s)")


def score_tracks(P: np.ndarray, kind: str) -> np.ndarray:
    """(T, 12) head outputs -> one eot score per frame (higher = more likely the end). Columns: user bins 0-4,
    quiet 1.04 s (5), quiet 2 s (6), agent bins 7-11."""
    if kind == "quiet2":
        return P[:, 6]
    if kind == "quiet1":
        return P[:, 5]
    if kind == "pred12":  # the predictive trigger's condition as a score: 1 - max(P(bin1), P(bin2))
        return 1.0 - np.maximum(P[:, 0], P[:, 1])
    if kind == "pred1234":
        return 1.0 - P[:, :4].max(1)
    if kind == "eot":
        return P[:, 12]
    if kind == "eot_x_quiet":  # the EOT posterior gated by the predicted silence
        return P[:, 12] * P[:, 6]
    raise ValueError(kind)


KINDS = ("quiet2", "quiet1", "pred12", "pred1234")
KINDS_EOT = KINDS + ("eot", "eot_x_quiet")


def stage_report(a):
    import eval_stage1 as E
    import bench_turn_icsi as M
    t0 = time.time()
    ec = lambda t: (t // T.CHUNK + 1) * T.CHUNK  # noqa: E731
    ef = lambda t: t + 1  # noqa: E731
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    tags = a.tags.split(",")
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        convs = [dict(v) for v in ext]
        for v in convs:
            v.pop("audio", None)
        folds_ = T.folds_of(corpus, convs)
        Tl = [len(v["spk_act"]) for v in convs]
        sil = [M.silero_timeout_track(np.load(T.SILERO[corpus] / f"{T.wkey(v)}.npy"), t)[:t] for v, t in zip(ext, Tl)]
        singles, hybrids = {"silero_timeout": (sil, ef)}, {}
        ref = [T.stored_scores(corpus, "tsvad_spk", v)[:t] for v, t in zip(ext, Tl)]
        singles["head_served"] = (ref, ec)
        hybrids["hybrid_dyn_served"] = (ref, ec, [T.dyn_track(s_, h) for s_, h in zip(sil, ref)], ec)

        def fit(x, t):
            x = np.asarray(x, np.float64)[:t]
            return np.concatenate([x, np.repeat(x[-1:], t - len(x))]) if len(x) < t else x
        for tag in tags:
            outs = [np.load(WORK / corpus / f"out_{tag}" / f"{T.wkey(v)}.npy") for v in ext]
            for kind in (KINDS_EOT if outs[0].shape[1] > 12 else KINDS):
                sc = [fit(score_tracks(P, kind), t) for P, t in zip(outs, Tl)]
                singles[f"head_{tag}_{kind}"] = (sc, ec)
                hybrids[f"hybrid_dyn_{tag}_{kind}"] = (sc, ec, [T.dyn_track(s_, h) for s_, h in zip(sil, sc)], ec)
        pairs = [(f"{k}_{tag}_{kind}", f"{k}_served") for tag in tags for kind in KINDS_EOT for k in ("head", "hybrid_dyn")
                 if f"{k}_{tag}_{kind}" in singles or f"{k}_{tag}_{kind}" in hybrids]
        for t2 in tags[1:]:  # the other tags vs the first one, same score (e.g. the 0.6B core vs the 115M core)
            pairs += [(f"{k}_{t2}_{kind}", f"{k}_{tags[0]}_{kind}") for kind in KINDS_EOT for k in ("head", "hybrid_dyn")
                      if all(x in singles or x in hybrids for x in (f"{k}_{t2}_{kind}", f"{k}_{tags[0]}_{kind}"))]
        r = T.score_systems(convs, meta, folds_, singles, hybrids, {}, {"2s": 25, "6s": 75}, a.n_boot, pairs)
        res[corpus] = r
        OUT.write_text(json.dumps(res, indent=1))
        log(f"  {corpus} report ({time.time() - t0:.0f}s)")
        for name, rr in r["systems"].items():
            x = rr["6s"]
            log(f"    {name:36s} miss {x['miss_rate'] * 100:5.1f} FC {x['fc_rate'] * 100:4.1f} P50 {x['p50_ms']} "
                f"open {x['strata']['open']['miss_rate'] * 100:5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--corpora", default="ami,icsi")
    p.add_argument("--tag", default="a")
    p.add_argument("--tags", default="a")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--eot", action="store_true", help="train: add the end-of-turn output and its label")
    p.add_argument("--p-null", type=float, default=0.1)
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--core", default="115m", choices=("115m", "0p6b"))
    p.add_argument("--track", default="tsvad_spk", help="scores: the eval TS-VAD track (experiment 4 variants)")
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
