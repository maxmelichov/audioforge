# /// script
# requires-python = ">=3.10"
# dependencies = ["audioforge", "numpy>=1.24", "torch>=2.1", "soundfile"]
#
# [tool.uv.sources]
# audioforge = { path = "../..", editable = true }
# ///
"""Perceived voice-gender head (research/VOICE_GENDER.md): data checks, frozen-encoder features, the capacity sweep
(which is also the training run) and the test tables. Stages:

    check                  speaker-disjointness of every split, printed
    feats   --core C       the speaker tap (115M block 4, 0.6B block 5) + heads.vad per frame for FLEURS train / dev /
                           test, LibriSpeech dev-clean (20 utterances per speaker) and test-clean; LibriSpeech
                           train-clean-100 is reused from the speaker head's frame cache (cache/spk_frame)
    sweep   --core C       hidden sizes at an equal wall-clock budget; held-out eval loss vs parameters; the knee is
                           saved as assets/voice_gender_<core>.pt (its best-eval-loss state)
    test    --core C       FLEURS test (17 languages, English alone) and LibriSpeech test-clean at 1 s / 2 s of pooled
                           speech and the full clip, 95 % bootstrap CIs (speakers where the data has speaker ids)

Licences: FLEURS (CC BY 4.0), LibriSpeech (CC BY 4.0). Never read here: oto, TurnBench, AMI / ICSI speakers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
OUT = SSD / "scratch" / "voice_gender"
LS = ROOT / "data/librispeech/LibriSpeech"
JSON = ROOT / "runs/voice_gender/results.json"
SR = 16000
BLOCK = {"115m": 4, "0p6b": 5}  # the speaker head's tap (1-based; research/LAYER_SWEEP_*.md, plans/voice_gender_001)
AFM = {"115m": ROOT / "runs/stage1_served_v4.afm", "0p6b": ROOT / "runs/served_0p6b_v0.3.afm"}  # same frozen encoders
SPK_CACHE = {"115m": SSD / "cache/spk_frame/librispeech.npz", "0p6b": SSD / "cache/spk_frame/0p6b5_librispeech.npz"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    JSON.parent.mkdir(parents=True, exist_ok=True)
    r = json.loads(JSON.read_text()) if JSON.exists() else {}
    r[key] = val
    JSON.write_text(json.dumps(r, indent=1))


# --------------------------------------------------------------------------- rows
def ls_gender() -> dict[str, tuple[str, str]]:
    """LibriSpeech reader id -> (gender F/M, subset), from SPEAKERS.TXT."""
    out = {}
    for line in (LS / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        p = [x.strip() for x in line.split("|")]
        out[p[0]] = (p[1], p[2])
    return out


def ls_rows(subset: str, per_spk: int | None = None) -> list[dict]:
    g = ls_gender()
    rows = []
    for spk_dir in sorted((LS / subset).iterdir()):
        if not spk_dir.is_dir():
            continue
        fs = sorted(spk_dir.glob("*/*.flac"))
        if per_spk:
            fs = [fs[i] for i in np.random.default_rng(int(spk_dir.name)).choice(len(fs), min(per_spk, len(fs)),
                                                                                 replace=False)]
        for f in sorted(fs):
            rows.append({"id": f.stem, "path": str(f), "y": int(g[spk_dir.name][0] == "M"), "spk": f"ls{spk_dir.name}",
                         "lang": "en"})
    return rows


def fleurs_rows(split: str) -> list[dict]:
    rows = []
    for line in (ROOT / "data/lid/fleurs/manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] == split and r.get("gender") in ("FEMALE", "MALE"):
            # FLEURS has no speaker id: the bootstrap unit is the row (see stage_test)
            rows.append({"id": r["id"], "path": str(ROOT / r["path"]), "y": int(r["gender"] == "MALE"),
                         "spk": f"fl{r['id']}", "lang": r["lang"], "sent": r["sent_id"]})
    return rows


SETS = {"fleurs_train": lambda: fleurs_rows("train"), "fleurs_dev": lambda: fleurs_rows("dev"),
        "fleurs_test": lambda: fleurs_rows("test"), "ls_dev": lambda: ls_rows("dev-clean", 20),
        "ls_test": lambda: ls_rows("test-clean")}


# --------------------------------------------------------------------------- check
def stage_check(a):
    g = ls_gender()
    z = np.load(SPK_CACHE["115m"])
    tr_ls = {i.split("-")[0] for i in z["ids"].tolist()}
    sub = {s: {d.name for d in (LS / s).iterdir() if d.is_dir()} for s in ("train-clean-100", "dev-clean", "test-clean")}
    print(f"LibriSpeech train speakers (spk_frame cache): {len(tr_ls)}; all in train-clean-100 per SPEAKERS.TXT: "
          f"{all(g[s][1] == 'train-clean-100' for s in tr_ls)}; == the train-clean-100 directory: "
          f"{tr_ls == sub['train-clean-100']}")
    for s in ("dev-clean", "test-clean"):
        print(f"  {s}: {len(sub[s])} speakers ({sum(g[x][0] == 'F' for x in sub[s])} F); overlap with train: "
              f"{len(tr_ls & sub[s])}")
    print(f"  dev-clean vs test-clean overlap: {len(sub['dev-clean'] & sub['test-clean'])}")
    fl = {s: fleurs_rows(s) for s in ("train", "dev", "test")}
    for s1, s2 in (("train", "dev"), ("train", "test"), ("dev", "test")):
        ids = len({r["id"] for r in fl[s1]} & {r["id"] for r in fl[s2]})
        sents = len({(r["lang"], r["sent"]) for r in fl[s1]} & {(r["lang"], r["sent"]) for r in fl[s2]})
        print(f"FLEURS {s1} vs {s2}: shared row ids {ids}, shared (language, sentence) {sents}")
    print("FLEURS publishes no speaker id, so its speaker disjointness cannot be checked here; the splits are "
          "sentence-disjoint (above). Selection uses FLEURS dev + LibriSpeech dev-clean; test is touched once.")
    ok = not (tr_ls & sub["dev-clean"]) and not (tr_ls & sub["test-clean"]) and not (sub["dev-clean"] & sub["test-clean"])
    print("LibriSpeech speaker-disjoint:", ok)
    save("check", {"ls_disjoint": ok, "ls_train_speakers": len(tr_ls)})


# --------------------------------------------------------------------------- features
def stage_feats(a):
    import soundfile as sf
    import torch

    from audioforge.train import load_model
    torch.set_num_threads(2)
    dev = torch.device(a.device)
    d = OUT / a.core
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "ls_train.feats.npy").exists():  # the speaker head's cache, as a memory-mappable copy
        z = np.load(SPK_CACHE[a.core])
        g = ls_gender()
        ids = z["ids"].tolist()
        np.save(d / "ls_train.feats.npy", z["feats"])
        np.save(d / "ls_train.vad.npy", np.ones(len(z["feats"]), np.float16))
        np.savez(d / "ls_train.meta.npz", off=z["off"], y=np.array([int(g[i.split("-")[0]][0] == "M") for i in ids]),
                 spk=np.array([f"ls{i.split('-')[0]}" for i in ids]), ids=np.array(ids), lang=np.array(["en"] * len(ids)))
        log(f"ls_train: {len(ids)} rows reused from {SPK_CACHE[a.core].name}")
    model = load_model(AFM[a.core], "cpu").eval().to(dev)
    b = BLOCK[a.core]
    assert model.layer_tap.get("spk") == [b - 1], (model.layer_tap.get("spk"), b)  # the speaker head's own tap
    for name in a.sets.split(","):
        if (d / f"{name}.meta.npz").exists():
            continue
        rows = SETS[name]()
        cap = 20.0 if name == "fleurs_train" else None
        auds = []
        for r in rows:
            x, sr = sf.read(r["path"], dtype="float32")
            assert sr == SR
            auds.append(x[: int(cap * SR)] if cap else x)
        order = sorted(range(len(rows)), key=lambda i: len(auds[i]))
        feats, vads, t0, i = [None] * len(rows), [None] * len(rows), time.time(), 0
        while i < len(order):
            j = i
            while j < len(order) and (j - i + 1) * len(auds[order[j]]) <= 240 * SR:
                j += 1
            idx = order[i:max(j, i + 1)]
            lens = torch.tensor([len(auds[k]) for k in idx])
            x = torch.zeros(len(idx), int(lens.max()))
            for q, k in enumerate(idx):
                x[q, :lens[q]] = torch.from_numpy(auds[k])
            with torch.no_grad():
                enc, elen, hid = model.encode(x.to(dev), lens.to(dev), return_hidden=True)
                v = model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid().float().cpu().numpy()
                h = hid[b - 1].half().cpu().numpy()
            for q, k in enumerate(idx):
                n = int(elen[q])
                feats[k], vads[k] = h[q, :n], v[q, :n].astype(np.float16)
            i = max(j, i + 1)
        off = np.cumsum([0] + [len(f) for f in feats])
        np.save(d / f"{name}.feats.npy", np.concatenate(feats))
        np.save(d / f"{name}.vad.npy", np.concatenate(vads))
        np.savez(d / f"{name}.meta.npz", off=off, y=np.array([r["y"] for r in rows]), spk=np.array([r["spk"] for r in rows]),
                 ids=np.array([r["id"] for r in rows]), lang=np.array([r["lang"] for r in rows]))
        log(f"{a.core} {name}: {len(rows)} rows, {off[-1]} frames, {sum(map(len, auds)) / SR / 3600:.2f} h, "
            f"{time.time() - t0:.0f} s")


class Set:
    """One cached set: frames (memory-mapped), VAD per frame, row offsets, labels, speakers."""

    def __init__(self, core, name, ram=False):
        d = OUT / core
        self.X = np.load(d / f"{name}.feats.npy", mmap_mode=None if ram else "r")
        self.vad = np.load(d / f"{name}.vad.npy")
        m = np.load(d / f"{name}.meta.npz")
        self.off, self.y, self.spk, self.ids, self.lang = m["off"], m["y"], m["spk"], m["ids"], m["lang"]
        self.n = len(self.y)

    def row(self, i):
        return self.X[self.off[i]:self.off[i + 1]], self.vad[self.off[i]:self.off[i + 1]]


# --------------------------------------------------------------------------- eval helpers
def batch_rows(S: Set, idx, dev, vad_gate=0.5, max_frames=None):
    import torch
    rs = [S.row(i) for i in idx]
    if max_frames:
        rs = [(x[:max_frames], v[:max_frames]) for x, v in rs]
    T = max(len(x) for x, _ in rs)
    X = np.zeros((len(rs), T, S.X.shape[1]), np.float32)
    K = np.zeros((len(rs), T), bool)
    for q, (x, v) in enumerate(rs):
        X[q, :len(x)] = x
        K[q, :len(x)] = v > vad_gate if vad_gate is not None else True
    lens = torch.tensor([len(x) for x, _ in rs])
    return torch.from_numpy(X).to(dev), lens.to(dev), torch.from_numpy(K).to(dev)


def posteriors(head, S: Set, dev, at=(13, 25), bs=64):
    """P(male) per row after ``at`` pooled speech frames (1 s / 2 s at 80 ms; a row with less speech gives its
    end-of-row posterior) and after the full row (VAD-gated pooling, as served)."""
    import torch
    out = {k: np.zeros(S.n) for k in (*at, "full")}
    order = np.argsort(S.off[1:] - S.off[:-1])
    head.eval()
    with torch.no_grad():
        for s in range(0, S.n, bs):
            idx = order[s:s + bs]
            X, lens, K = batch_rows(S, idx, dev)
            z = head.running_logits(X, lens, K).softmax(-1)[..., 1].float()
            seen = (K & (torch.arange(X.shape[1], device=dev)[None] < lens[:, None])).cumsum(1)
            last = (lens - 1).clamp(min=0)
            out["full"][idx] = z.gather(1, last[:, None])[:, 0].cpu().numpy()
            for k in at:
                hit = (seen >= k).float()
                first = torch.where(hit.any(1), hit.argmax(1), last)
                out[k][idx] = z.gather(1, first[:, None])[:, 0].cpu().numpy()
    return out


def bal_ce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean([-np.log(np.where(y == c, p if c else 1 - p, 1))[y == c].mean() for c in (0, 1)]))


if __name__ == "__main__":
    from voice_gender_train import main
    main(sys.modules[__name__])
