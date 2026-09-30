"""FRONTIER pilot 2: does the frozen 12.5 Hz streaming ASR encoder predict 12.5 Hz codec tokens frame-synchronously?
(research/archive/FRONTIER.md, P2: the precondition for ASR-encoder-driven streaming speech-to-speech / voice conversion.)

The served FastConformer (runs/stage1_served.afm, 80 ms frames, [70,1] streaming mask) and Kyutai Mimi (24 kHz, 12.5 Hz,
codebook 0 = WavLM-distilled "semantic" RVQ level, 1..7 acoustic) share an 80 ms clock. We fit per-frame probes
(linear, and a 2-layer MLP on the best layer) from encoder layer L at frame t+lag to Mimi code k at frame t, on
LibriSpeech dev-clean with speaker-disjoint train / test, and decode Mimi with predicted codebook 0 to measure content
(WER of the served ASR on the resynthesis).
  scripts/dev/gate.sh .venv/bin/python scripts/research/frontier_codec_probe.py extract --budget 420   # repeat until done
  scripts/dev/gate.sh .venv/bin/python scripts/research/frontier_codec_probe.py probe   -> runs/frontier_codec_probe.json
Lag is chosen on 5 held-out *train* speakers (val), never on the test speakers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FRONTIER_WORK", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/frontier")) / "codec"
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # kyutai/mimi (385 MB) is read from the local HF cache only
LS = ROOT / "data/librispeech/LibriSpeech/dev-clean"
LAYERS = [1, 3, 5, 8, 11, 16]   # 0-based block outputs of the 17-block encoder
NQ = 8
MIMI = "kyutai/mimi"


def utt_list(n_train_spk=30, per_train=10, per_test=8, seed=0, n_val_spk=5):
    rng = random.Random(seed)
    spks = sorted(p.name for p in LS.iterdir() if p.is_dir())
    out = []
    for i, s in enumerate(spks):
        files = sorted(LS.glob(f"{s}/*/*.flac"))
        rng.shuffle(files)
        split = "train" if i < n_train_spk - n_val_spk else ("val" if i < n_train_spk else "test")
        for f in files[: per_test if split == "test" else per_train]:
            out.append((split, f))
    return out


def transcripts():
    t = {}
    for p in LS.glob("*/*/*.trans.txt"):
        for line in p.read_text().splitlines():
            k, v = line.split(" ", 1)
            t[k] = v.lower()
    return t


def load_mimi():
    from transformers import MimiModel  # read-only from the local HF cache snapshot (no download)
    snap = sorted((Path.home() / ".cache/huggingface/hub/models--kyutai--mimi/snapshots").iterdir())[-1]
    return MimiModel.from_pretrained(str(snap)).eval()


@torch.inference_mode()
def extract(budget: float):
    from audioforge.train import load_model
    import torchaudio.functional as AF
    WORK.mkdir(parents=True, exist_ok=True)
    m = load_model(str(ROOT / "runs/stage1_served.afm"), "cpu").eval()
    mimi = load_mimi()
    t0 = time.time()
    done = 0
    for split, f in utt_list():
        o = WORK / f"{f.stem}.npz"
        if o.exists():
            continue
        if time.time() - t0 > budget:
            print("budget reached; re-run to continue", flush=True)
            return
        x, sr = sf.read(f, dtype="float32")
        xa = torch.from_numpy(x)[None]
        _, elen, hidden = m.encode(xa, torch.tensor([len(x)]), return_hidden=True)
        feats = np.stack([hidden[l][0, : int(elen)].numpy() for l in LAYERS]).astype(np.float16)
        mel, _ = m.features(xa, torch.tensor([len(x)]))
        x24 = AF.resample(xa, 16000, 24000)
        codes = mimi.encode(x24[:, None], num_quantizers=NQ).audio_codes[0].numpy().astype(np.int16)  # (NQ, K)
        np.savez(o, feats=feats, mel=mel[0].numpy().astype(np.float16), codes=codes, split=split)
        done += 1
    print(f"extracted {done} in {time.time() - t0:.0f}s", flush=True)


def load_set(split, lag, layer_idx=None, mel=False):
    X, Y, keys = [], [], []
    for split_, f in utt_list():
        if split_ != split:
            continue
        z = np.load(WORK / f"{f.stem}.npz")
        c = z["codes"].astype(np.int64)  # (NQ, K)
        if mel:
            mm = z["mel"].astype(np.float32)  # (80, Tmel) -> stack 8 frames per 80 ms
            T8 = mm.shape[1] // 8
            feat = mm[:, : T8 * 8].reshape(80, T8, 8).transpose(1, 0, 2).reshape(T8, 640)
            # 5 consecutive 80 ms stacks ending one frame after the source frame (400 ms of context, like [left, 1])
            pad = np.pad(feat, ((3, 1), (0, 0)))
            feat = np.concatenate([pad[k: k + T8] for k in range(5)], 1)
        else:
            feat = z["feats"][layer_idx].astype(np.float32)
        K, T = c.shape[1], feat.shape[0]
        ts = np.arange(K)
        src = ts + lag
        ok = (src >= 0) & (src < T)
        X.append(feat[src[ok]]); Y.append(c[:, ts[ok]].T); keys.append((f.stem, int(ok.sum())))
    return torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y)), keys


def fit(Xtr, ytr, hidden=0, epochs=12, lr=2e-3, n_cls=2048, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    D = Xtr.shape[1]
    net = nn.Linear(D, n_cls) if not hidden else nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, n_cls))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    Xn = (Xtr - mu) / sd
    for ep in range(epochs):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), 512):
            j = perm[i:i + 512]
            loss = F.cross_entropy(net(Xn[j]), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return lambda X: net((X - mu) / sd)


def acc(pred_fn, X, y):
    with torch.no_grad():
        lo = pred_fn(X)
        top5 = lo.topk(5, -1).indices
        return float((lo.argmax(-1) == y).float().mean()), float((top5 == y[:, None]).any(-1).float().mean())


def probe():
    res = {"layers": LAYERS, "nq": NQ, "mimi": MIMI}
    # 1) lag search on layer 11 (chosen on the val speakers)
    lag_acc = {}
    for lag in (0, 2, 3, 4, 5, 6, 8):
        Xtr, Ytr, _ = load_set("train", lag, LAYERS.index(11))
        Xte, Yte, _ = load_set("val", lag, LAYERS.index(11))
        fn = fit(Xtr, Ytr[:, 0], epochs=4)
        lag_acc[lag] = acc(fn, Xte, Yte[:, 0])[0]
        print("lag", lag, lag_acc[lag], flush=True)
    lag = max(lag_acc, key=lag_acc.get)
    res["lag_search_cb0_top1"] = lag_acc
    res["lag"] = lag
    # 2) per-layer linear probes, cb0 and cb1; majority and mel controls
    rows = {}
    Xtr_all = {}
    for li, l in enumerate(LAYERS):
        Xtr, Ytr, _ = load_set("train", lag, li)
        Xte, Yte, _ = load_set("test", lag, li)
        Xtr_all[l] = (Xtr, Ytr, Xte, Yte)
        r = {}
        for q in (0, 1):
            fn = fit(Xtr, Ytr[:, q])
            r[f"cb{q}"] = acc(fn, Xte, Yte[:, q])
        rows[f"L{l + 1}"] = r
        print(f"L{l + 1}", r, flush=True)
    Xtr, Ytr, _ = load_set("train", lag, mel=True)
    Xte, Yte, _ = load_set("test", lag, mel=True)
    rows["mel_stack8x5"] = {f"cb{q}": acc(fit(Xtr, Ytr[:, q]), Xte, Yte[:, q]) for q in (0, 1)}
    print("mel", rows["mel_stack8x5"], flush=True)
    for q in (0, 1):
        maj = torch.bincount(Ytr[:, q], minlength=2048).argmax()
        rows.setdefault("majority", {})[f"cb{q}"] = float((Yte[:, q] == maj).float().mean())
    # persistence control: code(t) = true code(t-1) on test utterances (how slowly the codes change)
    same = tot = 0
    for split_, f in utt_list():
        if split_ == "test":
            c = np.load(WORK / f"{f.stem}.npz")["codes"].astype(np.int64)
            same += int((c[:, 1:] == c[:, :-1]).sum(1)[0]); tot += c.shape[1] - 1
    rows["persistence_cb0"] = same / tot
    res["n_frames"] = {"train": int(len(Xtr_all[LAYERS[0]][0])), "test": int(len(Xtr_all[LAYERS[0]][2]))}
    res["linear"] = rows
    best = max(LAYERS, key=lambda l: rows[f"L{l + 1}"]["cb0"][0])
    Xtr, Ytr, Xte, Yte = Xtr_all[best]
    mlp0 = fit(Xtr, Ytr[:, 0], hidden=1024, epochs=15)
    res["mlp_best_layer"] = {"layer": best + 1, "cb0": acc(mlp0, Xte, Yte[:, 0])}
    # concat of two layers (best + L4) for MLP
    print("mlp", res["mlp_best_layer"], flush=True)
    # 3) resynthesis: 24 test utterances, served-ASR WER of Mimi decodes
    res["resynth"] = resynth(mlp0, best, lag)
    Path(ROOT / "runs/frontier_codec_probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


@torch.inference_mode()
def resynth(pred_fn, best, lag, n=24):
    import torchaudio.functional as AF
    from audioforge.teachers import normalize_text as N
    from audioforge.train import load_model
    mimi = load_mimi()
    asr = load_model(str(ROOT / "runs/stage1_served.afm"), "cpu").eval()
    tr = transcripts()
    li = LAYERS.index(best)
    test = [f for s, f in utt_list() if s == "test"][:n]
    arms = {"orig": [], "true_q8": [], "true_q1": [], "pred_q1": [], "pred_cb0_true_ac7": []}
    refs = []
    for f in test:
        z = np.load(WORK / f"{f.stem}.npz")
        c = torch.from_numpy(z["codes"].astype(np.int64))[None]  # (1,NQ,K)
        feat = torch.from_numpy(z["feats"][li].astype(np.float32))
        K, T = c.shape[-1], feat.shape[0]
        src = (torch.arange(K) + lag).clamp(0, T - 1)
        p0 = pred_fn(feat[src]).argmax(-1)
        x, _ = sf.read(f, dtype="float32")
        refs.append(N(tr[f.stem]))
        dec = lambda codes: AF.resample(mimi.decode(codes).audio_values[0, 0], 24000, 16000).numpy()
        arms["orig"].append(x)
        arms["true_q8"].append(dec(c))
        arms["true_q1"].append(dec(c[:, :1]))
        arms["pred_q1"].append(dec(p0[None, None]))
        cc = c.clone(); cc[0, 0] = p0
        arms["pred_cb0_true_ac7"].append(dec(cc))
    out = {}
    for k, xs in arms.items():
        hyps = asr.transcribe(xs, head="rnnt")
        e = sum(_ed(r.split(), N(h).split()) for r, h in zip(refs, hyps))
        out[k] = {"wer": e / sum(len(r.split()) for r in refs), "n": len(refs)}
        print("resynth", k, out[k], flush=True)
    return out


def _ed(r, h):
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "probe"])
    ap.add_argument("--budget", type=float, default=520)
    a = ap.parse_args()
    extract(a.budget) if a.cmd == "extract" else probe()
