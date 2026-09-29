"""research/IMPROVEMENTS.md section 3: the block-4 speaker head, crop-level ("frame-level") TitaNet distillation with
every LibriSpeech train-clean-100 speaker + the relational loss, trained on cached block-4 features.

What changes vs research/SPK_HEAD.md's shipped relational head (batch 6 whole segments, 5000 LibriSpeech utterances):
* the student embeds random 1.5-4 s crops of each segment (3 per segment, fixed seeds) and whole segments, each
  against TitaNet-L's embedding of exactly that audio (the teacher at the crop level: short, partial speech is what
  the live binder and the TS-VAD prints see);
* all 7727 cached LibriSpeech train-clean-100 1-12 s utterances (251 speakers) + AMI train (930) + ICSI train (3249);
* batch 128 (the relational loss compares within-batch cosine matrices: 128 x 128 pairs instead of 6 x 6);
* warm start from the served head (runs/stage1_served.afm heads.spk), AAM on AMI items kept.
The encoder never runs in the training loop (block-4 features cached once); only heads.spk changes.

Stages (CPU 2 threads unless noted; <= --budget s per call, resumable):
  feats    block-4 features per source + crop table -> <cache>/spk_frame/<source>.npz
  teacher  TitaNet-L on every crop and whole segment -> <cache>/spk_frame/<source>.teacher.npz
  train    (MPS <= 20 min per call, checkpointed) -> <scratch>/spk_frame/head_<tag>.pt
  build    the served model with heads.spk replaced -> <ssd runs>/stage1_served_spk_<tag>.afm (tensor check)
  eval     scripts/research/spk_head.py eval on it (AMI / ICSI dev, n = 64 / 200) -> runs/spk_frame.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
CACHE = SSD / "cache" / "spk_frame"
WORK = SSD / "scratch" / "spk_frame"
RUNS = SSD / "runs"
OUT = ROOT / "runs" / "spk_frame.json"
SERVED = ROOT / "runs" / "stage1_served.afm"
RECIPE = ROOT / "research" / "recipes" / "spk_relational_titanet.yaml"
SOURCES = {"ami_asr": {}, "icsi_asr": {}, "librispeech": {"n_train": 7727}}
# research/IMPROVEMENTS.md section 6: the same recipe on the 0.6B core (its best untrained speaker block, ENC_0P6B.md:
# block 11 of 24 within-meeting on AMI / ICSI) vs the 115M core's block 4 (SPK_HEAD.md sweep)
CORES = {"115m": (ROOT / "runs" / "stage1_served.afm", 3, 512),
         "0p6b": (Path("/Volumes/ExternalSSD/nvidia-audio-models/runs/nemo_nemotron_speech_streaming_en_0.6b.afm"), 10, 1024)}


def feat_path(name, core):
    return CACHE / (f"{name}.npz" if core == "115m" else f"{core}_{name}.npz")
N_CROPS, CROP_S = 3, (1.5, 4.0)
SR = 16000


def log(*a):
    print(*a, flush=True)


def load_source(name):
    import cache_titanet as CT
    from audioforge.train import load_data
    block, split, _ = CT.source_block(str(RECIPE), name)
    for k, v in SOURCES[name].items():
        for key in ("librispeech",):
            if key in block:
                block[key] = {**block[key], k: v}
    block.pop("drop_keys", None)
    return load_data({"data": block, "trainer": {}}, split)


def stage_feats(a):
    import tsvad as T
    from audioforge.data import segment_id
    from audioforge.train import load_model
    T.set_threads(2)
    CACHE.mkdir(parents=True, exist_ok=True)
    path, tap, _ = CORES[a.core]
    model = load_model(str(path), a.device).eval()
    t0 = time.time()
    for name in SOURCES:
        out = feat_path(name, a.core)
        if out.exists():
            continue
        part = out.with_name(out.stem + ".part.npz")
        data = load_source(name)
        prev = dict(np.load(part)) if part.exists() else None
        done = int(prev["n_done"]) if prev is not None else 0
        F = [prev["feats"]] if prev is not None else []
        lens = list(prev["lens"]) if prev is not None else []
        order = list(range(len(data)))
        for i in range(done, len(data), 8):
            if time.time() - t0 > a.budget:
                break
            fs = block_feats(model, [np.asarray(data[j]["audio"], np.float32) for j in order[i: i + 8]], tap, a.device)
            F.append(np.concatenate([f.astype(np.float16) for f in fs]))
            lens += [len(f) for f in fs]
            done = min(i + 8, len(data))
        feats = np.concatenate(F) if F else np.zeros((0, 512), np.float16)
        if done < len(data):
            T.save_npz(part, feats=feats, lens=np.array(lens), n_done=np.array(done))
            log(f"  {name}: {done}/{len(data)} ({time.time() - t0:.0f}s), resumable")
            return
        rng = np.random.default_rng(0)
        crops = []  # (item, frame0, frame1): 80 ms frames; audio = samples [1280 f0, 1280 f1)
        for i, L in enumerate(lens):
            for _ in range(N_CROPS):
                n = int(round(rng.uniform(*CROP_S) / 0.08))
                if n >= L:
                    continue
                f0 = int(rng.integers(0, L - n + 1))
                crops.append((i, f0, f0 + n))
        spk = np.array([int(d.get("speaker", -1)) if d.get("speaker") is not None else -1 for d in data])
        T.save_npz(out, feats=feats, lens=np.array(lens), off=np.concatenate([[0], np.cumsum(lens)]),
                   crops=np.array(crops), speaker=spk, ids=np.array([segment_id(d) for d in data]),
                   corpus=np.array([name] * len(data)))
        z = np.load(out)
        assert np.any(z["feats"][-1000:].astype(np.float32)), f"{out} reads back zero-filled"
        part.unlink(missing_ok=True)
        log(f"  {name}: {len(data)} items, {len(crops)} crops, {len(feats)} frames ({time.time() - t0:.0f}s)")


def block_feats(model, audios, layer, device="cpu", att=(70, 1)):
    """tsvad.block_feats for any core / device: the [70,1] chunked-mask forward up to block ``layer`` (0-based)."""
    import torch
    from audioforge.modules.fastconformer import chunked_attention_mask
    enc = model.encoder
    with torch.inference_mode():
        x, lens = model._pad(audios)
        feats, flen = model.preprocessor(x.to(device), lens.to(device))
        h, hl = enc.pre_encode(feats, flen)
        if enc.xscale is not None:
            h = h * enc.xscale
        T = h.shape[1]
        am = chunked_attention_mask(T, hl, list(att))
        pm = torch.arange(T, device=h.device)[None] >= hl[:, None]
        for li in range(layer + 1):
            h, _, _ = enc.layers[li](h, am, pm)
        return [h[j, : int(hl[j])].float().cpu().numpy() for j in range(len(hl))]


def stage_teacher(a):
    """TitaNet-L embeddings of each whole item and each crop (the crop's exact samples)."""
    import cache_titanet as CT
    from audioforge.nemo_import import import_titanet
    torch.set_num_threads(2)
    tn = import_titanet(str(CT.TITANET_LOCAL) if CT.TITANET_LOCAL.exists() else "nvidia/speakerverification_en_titanet_large")
    tn = tn.to(a.device).eval()
    t0 = time.time()
    for name in SOURCES:
        z = np.load(feat_path(name, "115m"))
        out = CACHE / f"{name}.teacher.npz"
        prev = dict(np.load(out)) if out.exists() else {}
        if prev and bool(prev.get("complete", False)):
            continue
        data = load_source(name)
        items = [(i, 0, int(z["lens"][i])) for i in range(len(data))]
        jobs = [("whole", k, it) for k, it in enumerate(items)] + [("crop", k, tuple(c)) for k, c in enumerate(z["crops"])]
        W = prev.get("whole", np.zeros((len(items), 192), np.float32))
        C = prev.get("crop", np.zeros((len(z["crops"]), 192), np.float32))
        todo = [j for j in jobs if not np.any((W if j[0] == "whole" else C)[j[1]])]
        todo.sort(key=lambda j: j[2][2] - j[2][1])
        n = 0
        for b in range(0, len(todo), 16):
            if time.time() - t0 > a.budget:
                break
            chunk = todo[b: b + 16]
            auds = []
            for kind, k, (i, f0, f1) in chunk:
                x = np.asarray(data[i]["audio"], np.float32)
                auds.append(x if kind == "whole" else x[f0 * 1280: f1 * 1280])
            with torch.no_grad():
                E = embed_dev(tn, auds, a.device)
            for (kind, k, _), e in zip(chunk, E):
                (W if kind == "whole" else C)[k] = e
            n += len(chunk)
        complete = n == len(todo)
        np.savez(out.with_suffix(".tmp.npz"), whole=W, crop=C, complete=np.array(complete))
        out.with_suffix(".tmp.npz").replace(out)
        log(f"  {name} teacher: {n} embedded, {len(todo) - n} left ({time.time() - t0:.0f}s)")
        if not complete:
            return


def embed_dev(tn, audios, device):
    """TitaNet.embed on any device (masked batching is exact)."""
    chunk = [torch.as_tensor(np.asarray(x, np.float32)) for x in audios]
    lens = torch.tensor([len(x) for x in chunk])
    x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
    return torch.nn.functional.normalize(tn(x.to(device), lens.to(device)), dim=-1).float().cpu().numpy()


class Items:
    """All sources' block-4 features in memory; ``batch`` draws items (crop with p 0.7, else the whole segment)."""

    def __init__(self, p_crop=0.7, seed=0, weights=(1.0, 1.0, 1.0), core="115m"):
        self.S = []
        for name in SOURCES:
            z = dict(np.load(feat_path(name, core)))
            t = np.load(CACHE / f"{name}.teacher.npz")
            z["tw"], z["tc"] = t["whole"], t["crop"]
            self.S.append(z)
        self.w = np.array(weights, np.float64) / sum(weights)
        self.p_crop = p_crop
        self.rng = np.random.default_rng(seed)

    def one(self, s):
        z = self.S[s]
        if self.rng.random() < self.p_crop and len(z["crops"]):
            k = int(self.rng.integers(len(z["crops"])))
            i, f0, f1 = z["crops"][k]
            t = z["tc"][k]
        else:
            i = int(self.rng.integers(len(z["lens"])))
            f0, f1 = 0, int(z["lens"][i])
            t = z["tw"][i]
        o = int(z["off"][i])
        x = z["feats"][o + f0: o + f1]
        if len(x) > 150:  # cap long whole segments at 12 s of frames (random window)
            q = int(self.rng.integers(0, len(x) - 150 + 1))
            x = x[q: q + 150]
        spk = int(z["speaker"][i]) if s == 0 else -1  # AAM only on AMI items (the served head's 190 classes)
        return x, t, spk

    def batch(self, B):
        src = self.rng.choice(len(self.S), B, p=self.w)
        xs = [self.one(int(s)) for s in src]
        L = max(len(x[0]) for x in xs)
        X = np.zeros((B, L, xs[0][0].shape[1]), np.float32)
        for j, (x, _, _) in enumerate(xs):
            X[j, : len(x)] = x
        return (torch.from_numpy(X), torch.tensor([len(x[0]) for x in xs]),
                torch.from_numpy(np.stack([x[1] for x in xs]).astype(np.float32)), torch.tensor([x[2] for x in xs]))


def served_head():
    from audioforge.train import load_model
    m = load_model(str(SERVED), "cpu")
    return m.heads["spk"], m.head_cfg["spk"]


def stage_train(a):
    dev = a.device
    torch.set_num_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    ck_path, out = WORK / f"head_{a.tag}.ckpt", WORK / f"head_{a.tag}.pt"
    if out.exists():
        log(f"{out} exists")
        return
    if a.init == "served":
        assert a.core == "115m", "--init served is the 115M head"
        head, cfg = served_head()
    else:  # fresh head, identical for both cores (the served head's config, the core's width)
        from audioforge.heads.audio import SpeakerHead
        torch.manual_seed(a.seed)
        head = SpeakerHead(CORES[a.core][2], num_speakers=190, emb_dim=192)
    head = head.to(dev)
    head.distill = {"target": "spk_teacher", "weight": a.cos_w, "relational_weight": a.rel_w}
    head.key = "spk_teacher"
    head.aam_weight = a.aam_w
    data = Items(a.p_crop, a.seed, core=a.core)
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
    t0 = time.time()
    for step in range(step0 + 1, a.steps + 1):
        head.train()
        x, ln, t, spk = data.batch(a.batch)
        x, ln, t, spk = x.to(dev), ln.to(dev), t.to(dev), spk.to(dev)
        e = head.embed(x, ln)
        b = {"spk_teacher": t}
        l = a.cos_w * head.distill_loss(e, b) + a.rel_w * head.relational_loss(e, b)
        m = spk >= 0
        if a.aam_w > 0 and m.sum() > 1:
            l = l + a.aam_w * head.aam_loss(e[m], spk[m])
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            with torch.no_grad():
                cos = float((e * torch.nn.functional.normalize(t, dim=-1)).sum(-1).mean())
            hist.append({"step": step, "loss": round(float(l), 4), "cos": round(cos, 4), "sec": round(time.time() - t0, 1)})
            log(json.dumps(hist[-1]))
        if step % 500 == 0 or step == a.steps or time.time() - t0 > a.budget:
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "hist": hist}, ck_path)
            if time.time() - t0 > a.budget and step < a.steps:
                log(f"budget: checkpoint at step {step}")
                return
    torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "args": vars(a), "hist": hist}, out)
    log(f"[train] {out}")


def stage_evalcore(a):
    """Within-meeting / all-pairs EER of a trained head on its core, the spk_head.py protocol (AMI dev and ICSI dev,
    n = 64 / 200, same trials), without building a model file: block-tap features of each segment, head.embed."""
    import torch
    import spk_head as SH
    from audioforge.train import load_model
    torch.set_num_threads(2)
    path, tap, D = CORES[a.core]
    m = load_model(str(path), a.device).eval()
    ck = torch.load(WORK / f"head_{a.tag}.pt", map_location="cpu", weights_only=False)
    from audioforge.heads.audio import SpeakerHead
    head = SpeakerHead(D, num_speakers=190, emb_dim=192)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec = {"core": a.core, "tap_block": tap + 1, "args": ck.get("args")}
    for corpus in ("ami", "icsi"):
        for n in (64, 200):
            val = SH.dev_segments(corpus, n)
            E = []
            for i in range(0, len(val), 8):
                fs = block_feats(m, [np.asarray(v["audio"], np.float32) for v in val[i:i + 8]], tap, a.device)
                with torch.no_grad():
                    for f in fs:
                        E.append(head.embed(torch.from_numpy(f)[None], torch.tensor([len(f)]))[0])
            r = SH.eer_block(torch.stack(E), val)
            rec[f"{corpus}_n{n}"] = r
            print(a.tag, corpus, n, r["eer"], r["eer_within_meeting"], flush=True)
            if n == 200:
                np.save(WORK / f"emb_{a.tag}_{corpus}200.npy", torch.stack(E).numpy())
    res.setdefault("variants", {})[a.tag] = rec
    OUT.write_text(json.dumps(res, indent=1))


def stage_paired(a):
    """Paired within-meeting EER differences at n = 200 (segment bootstrap, 500 resamples) from the stored
    embeddings of ``evalcore``: 0.6B vs 115M (both fresh, same recipe) and 115M warm vs fresh."""
    import spk_head as SH
    from audioforge.metrics import eer
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = {}
    pairs = [p.split(":") for p in a.pairs.split(",")]
    for corpus in ("ami", "icsi"):
        val = SH.dev_segments(corpus, 200)
        spk = np.array([v["speaker"] for v in val])
        meet = np.array([v["meeting"] for v in val])
        tags = sorted({t for p in pairs for t in p})
        E = {t: np.load(WORK / f"emb_{t}_{corpus}200.npy") for t in tags}

        def eer_w(Em, idx):
            X = Em[idx] / np.linalg.norm(Em[idx], axis=1, keepdims=True)
            iu = np.triu_indices(len(idx), 1)
            s = (X[iu[0]] * X[iu[1]]).sum(1)
            lab = spk[idx][iu[0]] == spk[idx][iu[1]]
            same = (meet[idx][iu[0]] == meet[idx][iu[1]]) & (idx[iu[0]] != idx[iu[1]])
            return float(eer(torch.tensor(s[same]), torch.tensor(lab[same].astype(np.int64))))
        n = len(val)
        rng = np.random.default_rng(0)
        base = {t: eer_w(E[t], np.arange(n)) for t in tags}
        for x, y in pairs:
            bs = []
            for _ in range(500):
                idx = rng.integers(0, n, n)
                bs.append(eer_w(E[x], idx) - eer_w(E[y], idx))
            out[f"{corpus}|{x}-{y}"] = {"a": round(base[x], 4), "b": round(base[y], 4),
                                        "delta": round(base[x] - base[y], 4),
                                        "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                                 round(float(np.percentile(bs, 97.5)), 4)]}
            print(corpus, x, y, out[f"{corpus}|{x}-{y}"], flush=True)
    res["paired_within_n200"] = out
    OUT.write_text(json.dumps(res, indent=1))


def stage_build(a):
    from audioforge.train import load_model, save_model
    m = load_model(str(SERVED), "cpu")
    src = {k: v.clone() for k, v in m.state_dict().items()}
    ck = torch.load(WORK / f"head_{a.tag}.pt", map_location="cpu", weights_only=False)
    m.heads["spk"].load_state_dict(ck["state_dict"])
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"stage1_served_spk_{a.tag}.afm"
    save_model(m, out)
    back = load_model(str(out), "cpu").state_dict()
    changed = [k for k in src if not torch.equal(src[k], back[k])]
    assert all(k.startswith("heads.spk.") for k in changed), changed[:5]
    log(f"[build] {out}: {len(changed)} tensors changed, all under heads.spk; {len(src) - len(changed)} identical")


def stage_eval(a):
    ck = RUNS / f"stage1_served_spk_{a.tag}.afm"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "research" / "spk_head.py"), "eval", "--ckpt", str(ck),
                    "--tag", a.tag, "--json", str(OUT)], check=True, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--tag", default="a")
    p.add_argument("--device", default="mps")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--cos-w", type=float, default=0.5)
    p.add_argument("--rel-w", type=float, default=1.0)
    p.add_argument("--aam-w", type=float, default=1.0)
    p.add_argument("--p-crop", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pairs", default="c0p6b_fresh:c115_fresh,c115_warm:c115_fresh")
    p.add_argument("--core", default="115m", choices=list(CORES))
    p.add_argument("--init", default="served", choices=("served", "fresh"))
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
