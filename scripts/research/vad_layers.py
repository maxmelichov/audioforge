"""Frame VAD per encoder block, and what a "gated encoder" would save (research/archive/VAD_LAYERS.md).

Question 1  Is a probe on an early block (2-4) as good a VAD as the served head (stage1_heads_pretrained.afm, a
            64-hidden frame head on a learned mix of all 17 blocks)?  Logistic regression and a 64-hidden MLP on the
            512-d frame vector of every block 1..17 and of the head's mix, fitted on AMI train diar windows (300 x 20 s,
            seeded) and scored with the BASELINES.md VAD protocol (64 x 20 s AMI dev diar windows, any-speaker label on
            the 80 ms grid, pooled acc / recall / precision / F1 at 0.5, sd.score_vad) plus AUC and miss at FPR 0.075
            (the matched-FPR point vs Silero), and on ICSI dev diar windows as held-out.
Question 2  Per-chunk cost (2 threads, cache-aware streaming) of the front-end and of the encoder truncated after block
            k, and the cost / error of re-priming blocks k+1..17 after a gated stretch (8, 16, 36, 70, 140 frames).
Question 3  Non-speech fraction (primary speaker / anyone) on AMI dev and the Pipecat demo windows, and the expected
            saving of the gated design.

Features come from the offline forward with the [70, 1] chunked-limited mask, which is what cache-aware streaming
computes (max |delta| ~1e-5, asserted in `compute` on real windows and in tests/test_streaming.py).

Stages (CPU, 2 threads, each < 10 min, resumable; results merge into --json):
  features --set train|ami_dev|icsi_dev   per-block features -> float16 memmaps in --work
  probes [--layers 1,2,...]                fit + score, one block at a time (skips blocks already in the json)
  compute --window 20|60 [--k 2,3,4,8]    streaming costs, truncation check, re-prime error / time
  silence                                  non-speech fractions and expected savings
  report                                   markdown tables to stdout

  W=<scratch>/vad_layers
  for s in train ami_dev icsi_dev; do .venv/bin/python scripts/research/vad_layers.py features --set $s --work $W; done
  .venv/bin/python scripts/research/vad_layers.py probes --work $W          # ~25 s per block; rerun to resume
  .venv/bin/python scripts/research/vad_layers.py compute --window 20 --work $W
  .venv/bin/python scripts/research/vad_layers.py compute --window 60 --work $W
  .venv/bin/python scripts/research/vad_layers.py silence --work $W && .venv/bin/python scripts/research/vad_layers.py report
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from audioforge.baselines import sd  # noqa: E402
from audioforge.modules.fastconformer import FastConformerEncoder, StreamState, chunked_attention_mask  # noqa: E402

CKPT = "runs/stage1_heads_pretrained.afm"
FPR_POINT = 0.075  # BASELINES.md: the FPR Silero v5 reaches at 0.5; "ours misses 12.1 % there"
N_TRAIN = 1000  # AMI train diar windows for the probes (300 was data-limited: the mix probe fell 0.012 F1 short of the head)
PRIMES = (8, 16, 36, 70, 140, 280)  # re-prime lengths in encoder frames (chunk-aligned: 35 -> 36)
KS = (2, 3, 4, 8, 17)
SETS = {"train": "AMI train diar windows (12 meetings, 20 s / hop 10 s, seeded cap --n-train)",
        "ami_dev": "AMI dev diar 64 x 20 s (BASELINES.md VAD set)",
        "icsi_dev": "ICSI dev diar 64 x 20 s (held-out corpus)"}


# --------------------------------------------------------------------------- gated-encoder helpers (pure functions
# over a FastConformerEncoder; serving code is untouched)
def truncated_encoder(enc: FastConformerEncoder, k: int) -> FastConformerEncoder:
    """A view of ``enc`` that runs only blocks 1..k (shares the weights): ``stream_step`` / ``forward`` of the view
    return block k's output, and its ``StreamState`` holds caches for k layers only. k = 0 is the subsampling alone."""
    assert 0 <= k <= len(enc.layers), k
    sub = copy.copy(enc)
    sub._modules = dict(enc._modules)
    sub._parameters = dict(enc._parameters)
    sub._buffers = dict(enc._buffers)
    sub.layers = enc.layers[:k]
    return sub


class DeepRunner:
    """Blocks k+1..n of ``enc`` in streaming mode with their own caches, fed block k's chunk outputs.

    ``step(xk, gate)``: with ``gate`` (silence) the deep blocks are skipped and marked cold; on the next un-gated
    chunk they are re-primed over the last ``prime`` buffered block-k frames (chunk by chunk, fresh caches), then the
    chunk itself is run. Returns (top (1, T, D) or None, deep hidden list or None, re-prime seconds)."""

    def __init__(self, enc: FastConformerEncoder, k: int, att_context_size=None, prime: int = 70, batched: bool = True):
        self.enc, self.k, self.n = enc, k, len(enc.layers)
        self.att_ctx = list(att_context_size or enc.att_context_size)
        self.batched = batched  # re-prime in one masked forward over the history (same numbers as chunk by chunk)
        L, R = self.att_ctx
        self.cs = R + 1
        self.left = (L // self.cs) * self.cs if L >= 0 else 10 ** 9
        assert prime % self.cs == 0, f"prime {prime} must be a multiple of the {self.cs}-frame chunk"
        self.prime = prime
        self.buf = None  # recent block-k frames, whole chunks
        self.att = self.conv = None  # None = cold
        self.pos = 0

    def _reset(self):
        d = self.enc.d_model
        self.att = [None] * (self.n - self.k)
        self.conv = [torch.zeros(1, d, l.conv.kernel - 1) for l in self.enc.layers[self.k:]]

    def _run(self, x, hidden=None):
        for j, layer in enumerate(self.enc.layers[self.k:]):
            x, (kk, vv), self.conv[j] = layer(x, None, None, self.pos, self.att[j], self.conv[j])
            self.att[j] = (kk[:, :, -self.left:], vv[:, :, -self.left:]) if self.left > 0 else None
            if hidden is not None:
                hidden.append(x)
        return x

    def _prime_batched(self, hist):
        """Deep blocks over the whole history at once with the chunked-limited mask (= the streaming computation,
        tests/test_streaming.py), then the caches are the last ``left`` keys / values and conv inputs."""
        P = hist.shape[1]
        mask = chunked_attention_mask(P, torch.tensor([P]), self.att_ctx)
        x = hist
        for j, layer in enumerate(self.enc.layers[self.k:]):
            x, (kk, vv), self.conv[j] = layer(x, mask, None, 0, None, self.conv[j])
            self.att[j] = (kk[:, :, -self.left:], vv[:, :, -self.left:]) if self.left > 0 else None

    @torch.no_grad()
    def step(self, xk: torch.Tensor, gate: bool):
        keep = max(self.prime, self.left) + xk.shape[1]
        self.buf = xk if self.buf is None else torch.cat([self.buf, xk], 1)[:, -keep:]
        if gate:
            self.att = None
            self.pos += xk.shape[1]
            return None, None, 0.0
        sec = 0.0
        if self.att is None:
            t0 = time.perf_counter()
            self._reset()
            hist = self.buf[:, : -xk.shape[1]]
            hist = hist[:, hist.shape[1] - min(self.prime, hist.shape[1]):]
            if self.batched and hist.shape[1]:
                self._prime_batched(hist)
            else:
                self.pos -= hist.shape[1]
                for s in range(0, hist.shape[1], self.cs):
                    self._run(hist[:, s: s + self.cs])
                    self.pos += min(self.cs, hist.shape[1] - s)
            sec = time.perf_counter() - t0 if hist.shape[1] else 0.0  # the cold start (no history) is not a re-prime
        hidden = []
        top = self._run(xk, hidden)
        self.pos += xk.shape[1]
        return top, hidden, sec


class GatedEncoder:
    """Streaming encoder that skips blocks k+1..n while ``gate`` is set (silence) and re-primes them on resume."""

    def __init__(self, enc: FastConformerEncoder, k: int, att_context_size=None, prime: int = 70):
        self.att_ctx = list(att_context_size or enc.att_context_size)
        self.shallow = truncated_encoder(enc, k)
        self.deep = DeepRunner(enc, k, self.att_ctx, prime)
        self.state = StreamState()

    @torch.no_grad()
    def step(self, mel_chunk, gate: bool, final=None):
        """-> (enc (1, T, D) or None when gated, hidden list of all n blocks or None, re-prime seconds)."""
        xk, hid, self.state = self.shallow.stream_step(mel_chunk, self.state, self.att_ctx, final=final,
                                                       return_hidden=True)
        if xk.shape[1] == 0:
            return None, None, 0.0
        top, deep_hid, sec = self.deep.step(xk, gate)
        return (None, None, sec) if top is None else (top, hid + deep_hid, sec)


# --------------------------------------------------------------------------- metrics (BASELINES VAD protocol + ROC)
def roc_stats(scores: np.ndarray, labels: np.ndarray, fpr_point: float = FPR_POINT) -> dict:
    """AUC (trapezoid) and the miss rate / threshold where FPR = fpr_point (linear interpolation on the ROC)."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels, bool)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    P, N = int(y.sum()), int((~y).sum())
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last = np.r_[s[1:] != s[:-1], True]  # one point per distinct score
    tpr, fpr = np.r_[0, tp[last] / max(P, 1)], np.r_[0, fp[last] / max(N, 1)]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    i = int(np.searchsorted(fpr, fpr_point, side="right"))
    if i >= len(fpr):
        tpr_at, thr = 1.0, float(s.min())
    elif i == 0:
        tpr_at, thr = 0.0, float(s.max())
    else:
        w = (fpr_point - fpr[i - 1]) / max(fpr[i] - fpr[i - 1], 1e-12)
        tpr_at = float(tpr[i - 1] + w * (tpr[i] - tpr[i - 1]))
        thr = float(s[last][i - 1])
    return {"auc": round(auc, 4), f"miss_at_fpr{fpr_point}": round(1 - tpr_at, 4), f"thr_at_fpr{fpr_point}": round(thr, 4)}


def vad_report(scores: list[np.ndarray], labels: list[np.ndarray]) -> dict:
    """sd.score_vad at 0.5 (acc / recall / precision / F1 / FPR / miss) + AUC + miss at the matched FPR."""
    r = dict(sd.score_vad(scores, labels, [0.5])["0.5"])
    r.update(roc_stats(np.concatenate(scores), np.concatenate(labels)))
    r["n_frames"] = int(sum(len(l) for l in labels))
    return r


# --------------------------------------------------------------------------- data
def load_set(name: str, n_train: int = N_TRAIN) -> list[dict]:
    from audioforge.train import derive_labels
    if name.startswith("train"):
        from audioforge.datasets.ami import recipe_data
        val = recipe_data({"data": {"ami": {"mode": "diar", "n_train": n_train, "seed": 0}}}, "train")
    elif name == "ami_dev":
        from eval_stage1 import ami_dev
        val = ami_dev("diar", 64)
    elif name == "icsi_dev":
        from audioforge.datasets.icsi import recipe_data
        val = recipe_data({"data": {"icsi": {"mode": "diar", "val_split": "dev", "n_val": 64, "seed": 0}}}, "val")
    else:
        raise ValueError(name)
    return [derive_labels(dict(v), ["vad"]) for v in val]


def load_model(path=CKPT):
    from audioforge.train import load_model as _lm
    return _lm(path, "cpu")


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, res: dict):
    path.write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- stage: features
@torch.no_grad()
def stage_features(a):
    """Per-block frame features of --set: memmap (N_frames, n_layers + 1, D) float16 (last row = the served head's
    layer mix), labels, window ids, the served head's probabilities. Resumable per batch."""
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    from audioforge.data import Collate
    tag = f"train{a.n_train}" if a.set == "train" else a.set
    val = load_set(a.set, a.n_train)
    model = load_model(a.ckpt)
    n_layers, D = len(model.encoder.layers), model.encoder.d_model
    Ts = [min(int(model.encoder.pre_encode.out_lengths(model.preprocessor.num_frames(torch.tensor(len(v["audio"]))))),
              len(v["vad"])) for v in val]
    off = np.r_[0, np.cumsum(Ts)]
    N = int(off[-1])
    meta_p, feat_p = work / f"{tag}.json", work / f"{tag}.f16"
    meta = _json(meta_p)
    if meta.get("done") == len(val):
        print(f"[features] {tag}: already complete ({N} frames)")
        return
    mode = "r+" if feat_p.exists() and meta.get("n_frames") == N else "w+"
    X = np.memmap(feat_p, np.float16, mode, shape=(N, n_layers + 1, D))
    labels = np.zeros(N, np.float32)
    head_p = np.zeros(N, np.float32)
    win = np.zeros(N, np.int32)
    if mode == "r+":
        labels[:], head_p[:], win[:] = np.load(work / f"{tag}_labels.npy"), np.load(work / f"{tag}_head.npy"), \
            np.load(work / f"{tag}_win.npy")
    done = int(meta.get("done", 0)) if mode == "r+" else 0
    col = Collate(None)
    t0 = time.time()
    for i in range(done, len(val), a.batch_size):
        b = col([{"audio": v["audio"]} for v in val[i: i + a.batch_size]])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        mix = model.head_input("vad", enc, hidden)
        pv = model.heads["vad"].decode(mix, elen)
        for j in range(len(elen)):
            w = i + j
            T = Ts[w]
            assert int(elen[j]) >= T, (int(elen[j]), T)
            X[off[w]: off[w + 1], :n_layers] = torch.stack([h[j, :T] for h in hidden], 1).to(torch.float16).numpy()
            X[off[w]: off[w + 1], n_layers] = mix[j, :T].to(torch.float16).numpy()
            labels[off[w]: off[w + 1]] = val[w]["vad"][:T]
            head_p[off[w]: off[w + 1]] = pv[j, :T].numpy()
            win[off[w]: off[w + 1]] = w
        done = min(i + a.batch_size, len(val))
        X.flush()
        np.save(work / f"{tag}_labels.npy", labels)
        np.save(work / f"{tag}_head.npy", head_p)
        np.save(work / f"{tag}_win.npy", win)
        _save(meta_p, {"set": tag, "what": SETS[a.set], "n_windows": len(val), "n_frames": N, "done": done,
                       "n_layers": n_layers, "d_model": D, "Ts": Ts, "ckpt": a.ckpt,
                       "att_context_size": list(model.encoder.att_context_size),
                       "layer_mix_vad": [round(float(x), 4) for x in model.layer_weights("vad")],
                       "audio_sec": round(sum(len(v["audio"]) for v in val) / sd.SR, 1),
                       "speech_frac": round(float(labels[: off[done]].mean()), 4), "sec": round(time.time() - t0, 1)})
        print(f"[features] {tag}: {done}/{len(val)} windows, {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > a.budget:
            print("[features] budget reached, rerun to resume")
            return


def load_features(work: Path, name: str):
    meta = _json(work / f"{name}.json")
    assert meta.get("done") == meta.get("n_windows"), f"features for {name} incomplete: run features --set {name}"
    X = np.memmap(work / f"{name}.f16", np.float16, "r", shape=(meta["n_frames"], meta["n_layers"] + 1, meta["d_model"]))
    return meta, X, np.load(work / f"{name}_labels.npy"), np.load(work / f"{name}_win.npy"), np.load(work / f"{name}_head.npy")


def per_window(p: np.ndarray, win: np.ndarray) -> list[np.ndarray]:
    return [p[win == w] for w in range(int(win.max()) + 1)]


# --------------------------------------------------------------------------- stage: probes
def fit_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=1000, tol=1e-4)
    clf.fit(X, y)
    return lambda Z: clf.predict_proba(Z)[:, 1]


def fit_mlp(X: np.ndarray, y: np.ndarray, hidden: int = 64, steps: int = 1500, batch: int = 2048, lr: float = 1e-3,
            seed: int = 0, drop: float = 0.2, wd: float = 1e-3):
    """The served head's shape (Linear-SiLU-Linear, 64 hidden), Adam on mini-batches, BCE, input dropout (the head
    trained over the frozen encoder in train mode, i.e. with its dropout on) and weight decay; a few hundred steps."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(X.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    with torch.enable_grad():
        for _ in range(steps):
            idx = torch.randint(0, len(Xt), (batch,), generator=g)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]).squeeze(-1), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()

    @torch.no_grad()
    def predict(Z):
        return torch.cat([net(torch.from_numpy(Z[i: i + 8192])).squeeze(-1).sigmoid() for i in range(0, len(Z), 8192)]).numpy()
    return predict


def stage_probes(a):
    work, jp = Path(a.work), Path(a.json)
    torch.set_num_threads(a.threads)
    mtr, Xtr, ytr, _, _ = load_features(work, f"train{a.n_train}")
    dev = {n: load_features(work, n) for n in ("ami_dev", "icsi_dev")}
    n_layers = mtr["n_layers"]
    res = _json(jp)
    q1 = res.setdefault("q1_probes", {})
    q1["setup"] = {"train": dict(mtr, Ts=None), "ami_dev": dict(dev["ami_dev"][0], Ts=None),
                   "icsi_dev": dict(dev["icsi_dev"][0], Ts=None), "features": "offline forward, [70,1] mask (= streaming)",
                   "standardize": "per-block mean / std of the train frames",
                   "logreg": "sklearn LogisticRegression C=1 lbfgs", "mlp": "Dropout(0.2)-Linear(512,64)-SiLU-Linear(64,1), Adam 1e-3 cosine, "
                   "1500 steps x 2048 frames, wd 1e-3, seed 0", "metrics": "sd.score_vad @0.5 (BASELINES), AUC, miss @FPR 0.075"}
    # the served head itself, from the same pass (must reproduce BASELINES.md: F1 0.949 on AMI dev)
    q1["served_head"] = {n: vad_report(per_window(h, w), per_window(y > 0.5, w)) for n, (m, X, y, w, h) in dev.items()}
    print("[probes] served head:", {n: (r["f1"], r["auc"], r[f"miss_at_fpr{FPR_POINT}"]) for n, r in q1["served_head"].items()})
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(1, n_layers + 1)) + ["mix"]
    rows = q1.setdefault("rows", {})
    for lay in layers:
        key = "mix" if lay == "mix" else f"block{int(lay):02d}"
        if key in rows and not a.force:
            continue
        li = n_layers if lay == "mix" else int(lay) - 1
        t0 = time.time()
        X = np.asarray(Xtr[:, li, :], np.float32)
        mu, sdv = X.mean(0), X.std(0) + 1e-5
        X = (X - mu) / sdv
        y = ytr > 0.5
        row = {"block": lay, "train_frames": int(len(y))}
        for name, fit in (("logreg", fit_logreg), ("mlp64", fit_mlp)):
            t1 = time.time()
            pred = fit(X, y)
            row[name] = {"fit_sec": round(time.time() - t1, 1)}
            for n, (m, Xd, yd, wd, hd) in dev.items():
                p = pred((np.asarray(Xd[:, li, :], np.float32) - mu) / sdv)
                row[name][n] = vad_report(per_window(p, wd), per_window(yd > 0.5, wd))
        row["sec"] = round(time.time() - t0, 1)
        rows[key] = row
        _save(jp, res)
        print(f"[probes] {key}: logreg F1 {row['logreg']['ami_dev']['f1']:.4f} AUC {row['logreg']['ami_dev']['auc']:.4f} "
              f"| mlp64 F1 {row['mlp64']['ami_dev']['f1']:.4f} AUC {row['mlp64']['ami_dev']['auc']:.4f} "
              f"miss@fpr {row['mlp64']['ami_dev'][f'miss_at_fpr{FPR_POINT}']:.4f} | ICSI mlp F1 "
              f"{row['mlp64']['icsi_dev']['f1']:.4f} ({row['sec']:.0f}s)", flush=True)
        if time.time() - t0 > a.budget:
            break
    _save(jp, res)


# --------------------------------------------------------------------------- stage: compute
WIN_INDEX = 47  # AMI dev diar window 47 = IB4002 @ 60 s: 77.3 % speech (the set: 76.9 %), 5 speech resumes in 20 s


def window_audio(sec: int, index: int = WIN_INDEX):
    """Window ``index`` of the AMI dev diar set (20 s) or the 60 s of the same meeting from the same start; with its
    any-speaker label on the 80 ms grid."""
    from eval_stage1 import ami_dev
    from audioforge.train import derive_labels
    v = derive_labels(dict(ami_dev("diar", 64)[index]), ["vad"])
    if sec == 20:
        return v["audio"], v["vad"], v["meeting"], float(v["start"])
    from audioforge.data import ToneLanguage
    from audioforge.datasets.ami import AMI, spk_matrix
    ds = AMI([v["meeting"]], verbose=False)
    a = float(v["start"])
    x = ds._clip(v["meeting"], a, a + sec)
    T = ToneLanguage.n_frames(len(x))
    y, _, _ = spk_matrix(ds.acts[v["meeting"]], T, a, a + sec)
    return x, (y.max(1) > 0.5).astype(np.float32), v["meeting"], a


def _times(ts):
    ts = np.asarray(ts) * 1000
    return {"p50_ms": round(float(np.median(ts)), 2), "mean_ms": round(float(ts.mean()), 2),
            "p95_ms": round(float(np.percentile(ts, 95)), 2), "total_s": round(float(ts.sum() / 1000), 2), "n": int(len(ts))}


@torch.no_grad()
def stage_compute(a):
    work, jp = Path(a.work), Path(a.json)
    work.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    model = load_model(a.ckpt)
    enc, att = model.encoder, list(model.encoder.att_context_size)
    cs, f = att[1] + 1, enc.subsampling_factor
    chunk_mel = cs * f
    audio, label, meeting, start = window_audio(a.window, a.win_index)
    x = torch.from_numpy(np.asarray(audio, np.float32))[None]
    mel, mlen = model.preprocessor(x, torch.tensor([x.shape[1]]))
    n_chunks = mel.shape[-1] // chunk_mel
    mel = mel[..., : n_chunks * chunk_mel]
    res = _json(jp)
    q2 = res.setdefault("q2_compute", {})
    key = f"win{a.window}s"
    rec = q2.setdefault(key, {"meeting": meeting, "start": start, "win_index": a.win_index, "sec": a.window, "n_chunks": n_chunks,
                             "chunk_ms": cs * 80, "att_context_size": att, "threads": a.threads})
    rec["loadavg_start"] = os.getloadavg()
    # 1. ungated streaming == offline (existing test pattern), per-layer wall-time profile
    off, olen, off_hidden = enc(mel, torch.tensor([mel.shape[-1]]), att, return_hidden=True)
    cache = work / f"stream_{key}.pt"
    if cache.exists() and not a.force:
        st = torch.load(cache)
        hid_chunks, top_chunks, vad_ref = st["hid_chunks"], st["top_chunks"], st["vad_ref"]
        rec["per_layer_ms"] = st["per_layer_ms"]
    else:
        state, hid_chunks, top_chunks, vad_ref = StreamState(), [], [], []
        lt = [[] for _ in enc.layers]
        hooks = []
        for i, layer in enumerate(enc.layers):
            hooks.append(layer.register_forward_pre_hook(lambda m, inp, i=i: lt[i].append(time.perf_counter())))
            hooks.append(layer.register_forward_hook(lambda m, inp, out, i=i: lt[i].__setitem__(-1, time.perf_counter() - lt[i][-1])))
        for c in range(n_chunks):
            o, hid, state = enc.stream_step(mel[..., c * chunk_mel: (c + 1) * chunk_mel], state, att, final=False,
                                            return_hidden=True)
            hid_chunks.append([h.clone() for h in hid])
            top_chunks.append(o.clone())
            vad_ref.append(model.heads["vad"].decode(model.head_input("vad", o, hid), torch.tensor([o.shape[1]]))[0].clone())
        for h in hooks:
            h.remove()
        rec["per_layer_ms"] = [round(float(np.median(t)) * 1000, 3) for t in lt]
        torch.save({"hid_chunks": hid_chunks, "top_chunks": top_chunks, "vad_ref": vad_ref, "per_layer_ms": rec["per_layer_ms"]}, cache)
    stream = torch.cat(top_chunks, 1)
    T = min(stream.shape[1], int(olen[0]))
    d = float((stream[:, :T] - off[:, :T]).abs().max())
    rec["ungated_stream_vs_offline_max_abs"] = d
    # without ``final`` the aligned streamer holds back the trailing frame the offline pass computes over zero padding
    assert int(olen[0]) - 1 <= stream.shape[1] <= int(olen[0]) and d < 1e-3, (stream.shape, olen, d)
    print(f"[compute] {key}: {n_chunks} chunks, streaming == offline (max |d| {d:.1e})", flush=True)
    # 2. front-end: log-mel per chunk (StreamingSession._mel on the pre-emphasized signal) and subsampling (k = 0)
    from audioforge.model import StreamingSession
    s = StreamingSession(model)
    xt = x[0]
    s.sig = torch.cat([xt[:1], xt[1:] - model.preprocessor.preemph * xt[:-1]])
    ts = []
    for c in range(n_chunks):
        t0 = time.perf_counter()
        s._mel(c * chunk_mel, (c + 1) * chunk_mel)
        ts.append(time.perf_counter() - t0)
    rec["logmel_per_chunk"] = _times(ts)
    # 3. encoder truncated after block k, full streaming passes (k = 0: subsampling only)
    ks = [int(k) for k in a.k.split(",")] if a.k else list(KS)
    trunc = rec.setdefault("truncated", {})
    for k in [0] + [k for k in ks if k <= len(enc.layers)]:
        if str(k) in trunc and not a.force:
            continue
        sub = truncated_encoder(enc, k)
        state, ts, outs = StreamState(), [], []
        for c in range(n_chunks):
            t0 = time.perf_counter()
            o, state = sub.stream_step(mel[..., c * chunk_mel: (c + 1) * chunk_mel], state, att, final=False)
            ts.append(time.perf_counter() - t0)
            outs.append(o)
        o = torch.cat(outs, 1)
        ref = torch.cat([h[k - 1] for h in hid_chunks], 1) if k else None
        dk = float((o - ref).abs().max()) if k else None
        if k:
            assert dk < 1e-4, dk
        trunc[str(k)] = dict(_times(ts), max_abs_vs_full=dk, loadavg=os.getloadavg()[0])
        print(f"[compute] {key} k={k}: {trunc[str(k)]['p50_ms']} ms / chunk (max |d| vs full pass {dk})", flush=True)
        _save(jp, res)
    # 4. re-prime after gated stretches (oracle any-speaker label; gate = whole chunk non-speech)
    lab = label[: n_chunks * cs].reshape(n_chunks, cs).max(1) > 0.5
    gate = ~lab
    resumes = _runs_from_gate(gate)
    rec["gate"] = {"gated_chunks": int(gate.sum()), "gated_frac": round(float(gate.mean()), 4), "resumes": resumes,
                   "definition": "oracle any-speaker label, chunk gated iff both of its 80 ms frames are non-speech"}
    rp = rec.setdefault("reprime", {})
    for k in [k for k in ks if k < len(enc.layers)]:
        for P in ([int(x) for x in a.primes.split(",") if x] if a.primes is not None else PRIMES):
            kk = f"k{k}_P{P}"
            if kk in rp and not a.force:
                continue
            runner = DeepRunner(enc, k, att, P)
            d_top, d_vad, d_first, secs, first = [], [], [], [], True
            for c in range(n_chunks):
                top, dh, sec = runner.step(hid_chunks[c][k - 1], bool(gate[c]))
                if top is None:
                    first = True
                    continue
                if sec:
                    secs.append(sec)
                e = float((top - top_chunks[c]).abs().max())
                pv = model.heads["vad"].decode(model.head_input("vad", top, hid_chunks[c][:k] + dh), torch.tensor([cs]))[0]
                ev = float((pv - vad_ref[c]).abs().max())
                d_top.append(e)
                d_vad.append(ev)
                if first and c > 0:
                    d_first.append(e)
                first = False
            rp[kk] = {"k": k, "prime_frames": P, "max_abs_top": round(max(d_top), 6), "max_abs_top_first_chunk_after_resume":
                      round(max(d_first), 6) if d_first else None, "p50_abs_top_first_chunk": round(float(np.median(d_first)), 6) if d_first else None,
                      "max_abs_vad_prob": round(max(d_vad), 6), "reprime_ms_p50": round(float(np.median(secs)) * 1000, 2) if secs else 0.0,
                      "reprime_ms_mean": round(float(np.mean(secs)) * 1000, 2) if secs else 0.0, "n_reprimes": len(secs)}
            print(f"[compute] {key} {kk}: max|d top| {rp[kk]['max_abs_top']:.2e} (first chunk {rp[kk]['max_abs_top_first_chunk_after_resume']}) "
                  f"vad {rp[kk]['max_abs_vad_prob']:.2e}, re-prime {rp[kk]['reprime_ms_p50']} ms x {len(secs)}", flush=True)
            _save(jp, res)
    rec["loadavg_end"] = os.getloadavg()
    _save(jp, res)


# --------------------------------------------------------------------------- stage: silence
def _runs_from_gate(g: np.ndarray) -> int:
    return int(((~g[1:]) & g[:-1]).sum())


def gate_from_label(lab: np.ndarray, cs: int, hangover: int) -> np.ndarray:
    """Chunk gate from a frame label: a chunk is gated iff all its frames are non-speech and the last ``hangover``
    chunks were non-speech too (hangover = how long after speech the deep layers keep running)."""
    n = len(lab) // cs
    silent = ~(lab[: n * cs].reshape(n, cs).max(1) > 0.5)
    g = silent.copy()
    for h in range(1, hangover + 1):
        g[h:] &= silent[:-h]
        g[:h] = False
    return g


def stage_silence(a):
    jp = Path(a.json)
    res = _json(jp)
    from eval_stage1 import ami_dev
    from audioforge.datasets.ami import recipe_data
    from audioforge.train import derive_labels
    q3 = res.setdefault("q3_silence", {})
    sets = {}
    diar = [derive_labels(dict(v), ["vad"]) for v in ami_dev("diar", 64)]
    turn = ami_dev("turn", 64)
    val200 = recipe_data({"data": {"ami": {"mode": "turn", "val_split": "dev", "n_val": 200, "seed": 0}}}, "val")
    demo_idx = [3, 14, 27, 64, 119]  # the Pipecat / LiveKit demo windows of INTEGRATION.md (examples/pipecat_local_demo.py prepare)
    demo = [val200[i] for i in demo_idx]
    sets["ami_dev_diar64"] = {"anyone": [v["vad"] for v in diar]}
    sets["ami_dev_turn64"] = {"anyone": [np.asarray(v["spk_targets"]).max(1) for v in turn],
                              "primary": [np.asarray(v["spk_act"]) for v in turn]}
    sets["pipecat_demo_windows5"] = {"anyone": [np.asarray(v["spk_targets"]).max(1) for v in demo],
                                     "primary": [np.asarray(v["spk_act"]) for v in demo],
                                     "names": [f"ami_{v['meeting']}_{i:03d}" for i, v in zip(demo_idx, demo)]}
    # per-chunk costs from q2 (20 s window); the saving model: gated chunks cost c_k, others c_17, + re-prime per resume
    q2 = res.get("q2_compute", {}).get("win60s") or res.get("q2_compute", {}).get("win20s", {})
    tr = q2.get("truncated", {})
    c_full = tr.get("17", {}).get("p50_ms")
    out = {}
    for name, d in sets.items():
        o = {"n_windows": len(d["anyone"]), "frames": int(sum(len(l) for l in d["anyone"]))}
        if "names" in d:
            o["names"] = d["names"]
        for defn in ("anyone", "primary"):
            if defn not in d:
                continue
            labs = d[defn]
            fr = {"nonspeech_frac": round(1 - float(np.concatenate(labs).mean()), 4)}
            for hang in (0, 5):
                gates = [gate_from_label(l, 2, hang) for l in labs]
                nch = sum(len(g) for g in gates)
                fr[f"gated_frac_hang{hang}"] = round(sum(int(g.sum()) for g in gates) / max(nch, 1), 4)
                fr[f"resumes_per_min_hang{hang}"] = round(60 * sum(_runs_from_gate(g) for g in gates) / (nch * 0.16), 2)
                if c_full:
                    sav = {}
                    for k in (2, 3, 4, 8):
                        ck = tr.get(str(k), {}).get("p50_ms")
                        if ck is None:
                            continue
                        best = q2.get("reprime", {})
                        # cheapest P whose top-layer error is < 1e-3 (else the largest measured)
                        cands = sorted((r for r in best.values() if r["k"] == k), key=lambda r: r["prime_frames"])
                        ok = [r for r in cands if r["max_abs_top"] < 1e-3]
                        r = ok[0] if ok else (cands[-1] if cands else None)
                        rep = r["reprime_ms_mean"] if r else 0.0
                        cost = sum(int(g.sum()) * ck + int((~g).sum()) * c_full + _runs_from_gate(g) * rep for g in gates)
                        sav[f"k{k}"] = {"saving_frac": round(1 - cost / (nch * c_full), 4), "prime_frames": r["prime_frames"] if r else None,
                                        "reprime_ms": rep, "ck_ms": ck, "c_full_ms": c_full}
                    fr[f"saving_hang{hang}"] = sav
            o[defn] = fr
        out[name] = o
        print(f"[silence] {name}: " + ", ".join(f"{k}: nonspeech {v['nonspeech_frac']}" for k, v in o.items() if isinstance(v, dict)), flush=True)
    q3.update(out)
    q3["pipecat_log"] = ("no Pipecat event log is checked into the repo (the demos' ev.jsonl live in another agent's scratch); "
                         "the 5 demo windows' AMI labels are used instead")
    q3["model"] = ("cost per chunk = c_k (gated) or c_17 (un-gated) from q2 win60s p50 + one re-prime per resume "
                   "(mean re-prime ms of the cheapest P with top-layer error < 1e-3); hang = chunks (160 ms) the deep blocks keep "
                   "running after the last speech chunk; the gate is the oracle label, not a detector")
    _save(jp, res)


# --------------------------------------------------------------------------- stage: report
def stage_report(a):
    res = _json(Path(a.json))
    q1 = res.get("q1_probes", {})
    if q1:
        sh = q1["served_head"]
        mk = f"miss_at_fpr{FPR_POINT}"
        print("| input | logreg AMI dev F1 / AUC / miss@FPR.075 | mlp64 AMI dev F1 / AUC / miss@FPR.075 | mlp64 acc / rec / prec | "
              "mlp64 ICSI dev F1 / AUC / miss | logreg ICSI F1 | vs served (mlp64 F1 delta) |")
        print("|---|---|---|---|---|---|---|")
        f1s = sh["ami_dev"]["f1"]
        for key, r in sorted(q1.get("rows", {}).items(), key=lambda kv: (kv[0] == "mix", kv[0])):
            lr, ml = r["logreg"], r["mlp64"]
            d = ml["ami_dev"]["f1"] - f1s
            tag = "match" if abs(d) <= 0.005 else ("better" if d > 0 else "worse")
            print(f"| {key} | {lr['ami_dev']['f1']:.4f} / {lr['ami_dev']['auc']:.4f} / {lr['ami_dev'][mk]:.3f} | "
                  f"{ml['ami_dev']['f1']:.4f} / {ml['ami_dev']['auc']:.4f} / {ml['ami_dev'][mk]:.3f} | "
                  f"{ml['ami_dev']['acc']:.3f} / {ml['ami_dev']['recall']:.3f} / {ml['ami_dev']['precision']:.3f} | "
                  f"{ml['icsi_dev']['f1']:.4f} / {ml['icsi_dev']['auc']:.4f} / {ml['icsi_dev'][mk]:.3f} | {lr['icsi_dev']['f1']:.4f} | "
                  f"{d:+.4f} {tag} |")
        print(f"| served head | - | {sh['ami_dev']['f1']:.4f} / {sh['ami_dev']['auc']:.4f} / {sh['ami_dev'][mk]:.3f} | "
              f"{sh['ami_dev']['acc']:.3f} / {sh['ami_dev']['recall']:.3f} / {sh['ami_dev']['precision']:.3f} | "
              f"{sh['icsi_dev']['f1']:.4f} / {sh['icsi_dev']['auc']:.4f} / {sh['icsi_dev'][mk]:.3f} | - | ref |")
    for key, rec in res.get("q2_compute", {}).items():
        print(f"\n{key}: {rec['n_chunks']} chunks of {rec['chunk_ms']} ms, log-mel {rec['logmel_per_chunk']['p50_ms']} ms/chunk, "
              f"stream vs offline max |d| {rec['ungated_stream_vs_offline_max_abs']:.1e}, load {rec.get('loadavg_start')}")
        print("| k (blocks) | ms / chunk p50 | mean | fraction of full (p50) |")
        print("|---|---|---|---|")
        full = rec["truncated"]["17"]["p50_ms"] if "17" in rec["truncated"] else None
        for k, t in sorted(rec["truncated"].items(), key=lambda kv: int(kv[0])):
            print(f"| {k} | {t['p50_ms']} | {t['mean_ms']} | {t['p50_ms'] / full:.3f} |" if full else f"| {k} | {t['p50_ms']} | {t['mean_ms']} | - |")
        if rec.get("per_layer_ms"):
            print("per-layer ms (p50):", rec["per_layer_ms"])
        if rec.get("reprime"):
            print(f"gate: {rec['gate']}")
            print("| k | P frames | max |d| top (all) | max |d| top first chunk | max |d| VAD prob | re-prime ms p50 |")
            print("|---|---|---|---|---|---|")
            for kk, r in sorted(rec["reprime"].items(), key=lambda kv: (kv[1]["k"], kv[1]["prime_frames"])):
                print(f"| {r['k']} | {r['prime_frames']} | {r['max_abs_top']:.2e} | {r['max_abs_top_first_chunk_after_resume']} | "
                      f"{r['max_abs_vad_prob']:.2e} | {r['reprime_ms_p50']} |")
    q3 = res.get("q3_silence", {})
    for name, o in q3.items():
        if isinstance(o, dict):
            print(f"\n{name}: {json.dumps(o)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("features", "probes", "compute", "silence", "report"))
    ap.add_argument("--set", default="ami_dev", choices=tuple(SETS))
    ap.add_argument("--work", default=os.environ.get("VAD_LAYERS_WORK", "/private/tmp/claude-501/-Users-maxm/"
                                                     "54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad/vad_layers"))
    ap.add_argument("--json", default="runs/vad_layers.json")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layers", default=None, help="probes: comma list of blocks (1-based) and/or mix")
    ap.add_argument("--window", type=int, default=20, choices=(20, 60))
    ap.add_argument("--win-index", type=int, default=WIN_INDEX, help="compute: AMI dev diar window")
    ap.add_argument("--k", default=None, help="compute: comma list of truncation depths")
    ap.add_argument("--primes", default=None, help="compute: comma list of re-prime lengths (frames); '' = none")
    ap.add_argument("--n-train", type=int, default=N_TRAIN, help="features/probes: AMI train diar windows")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget", type=float, default=480.0, help="seconds per process before stopping (resumable)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"features": stage_features, "probes": stage_probes, "compute": stage_compute, "silence": stage_silence,
     "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
