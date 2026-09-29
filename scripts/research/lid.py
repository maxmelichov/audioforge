"""Spoken language identification on the single-model front end vs dedicated LID models (research/LID.md).

Data: scripts/research/lid_data.py (FLEURS subset, 17 languages, speaker-disjoint train / dev / test). Every system sees the
same clips: for each test utterance, the first 1 / 2 / 3 / 5 s from the speech onset (Silero VAD v5 onset - 0.1 s)
and the full utterance, peak-normalised to -1 dBFS (lid_data.load_audio). Stages:

  onsets        Silero VAD speech onset / offset of every utterance -> data/lid/fleurs/onsets.json (CPU).
  probe_feats   frozen served encoder (runs/stage1_served.afm, [70, 1]) on the 5 s clip of each utterance: mean-pooled
                output of every block over the first 2 s and the whole 5 s -> data/lid/cache/probe_<split>_<lang>.npz
                (CPU, 2 threads; resumable per language).
  probe         per block and duration: standardise + multinomial logistic regression fitted on train, accuracy /
                macro-F1 on test -> runs/lid.json["probe"].
  train         LID head (heads.audio.LanguageHead) on the chosen block(s) of the frozen served model, head-only, MPS;
                asserts every other tensor is bit-identical to the init checkpoint; saves the head alone
                (audioforge.lid.save_head) -> runs/lid_head.pt, log in runs/lid.json["train"].
  eval_head     the head's running posterior at the end of each clip (encoder run on the clip itself) -> predictions
                data/lid/preds/<tag>.npz; plus the per-160 ms-chunk CPU cost of head + encoder (2 threads).
  baseline      --system ambernet | speechbrain | whisper_tiny | whisper_base: the same clips, posteriors restricted
                (renormalised) to the 17 languages -> data/lid/preds/<system>.npz (resumable per language), cost.
  decide        the server's announcement rule (audioforge.lid: LangDecider over the VAD-gated running posterior;
                --system ambernet adds the AmberNet backend) on whole dev / test utterances: first-announcement
                accuracy and time, flips -> runs/lid.json["decide"].
  cost          ms per call of each dedicated model on 1-8 s clips and the head's ms per 160 ms chunk, 2 threads.
  report        accuracy / macro-F1 per duration, hard-pair confusions, EdAcc (accented English) -> runs/lid.json
                ["systems"] / ["edacc"], markdown to stdout.
  (--set edacc on eval_head / baseline: the accented-English probe, scripts/research/lid_data.py edacc.)

  .venv/bin/python scripts/research/lid.py onsets
  .venv/bin/python scripts/research/lid.py probe_feats --split train      # re-run until "all languages done"
  .venv/bin/python scripts/research/lid.py probe
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/research/lid.py train --recipe research/recipes/lid_aug.yaml
  .venv/bin/python scripts/research/lid.py eval_head --head runs/lid_aug.pt [--set edacc]
  .venv/bin/python scripts/research/lid.py baseline --system ambernet [--set edacc]   # speechbrain, whisper_tiny, whisper_base
  .venv/bin/python scripts/research/lid.py decide --head runs/lid_aug.pt --system ambernet --n 20
  .venv/bin/python scripts/research/lid.py cost --head runs/lid_aug.pt
  .venv/bin/python scripts/research/lid.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from lid_data import LANGS, load_audio, read_edacc, read_manifest  # noqa: E402

SR = 16000
CODES = list(LANGS)  # class index = position (en he ar ru es fr de pt it nl pl uk tr fa hi zh ja)
DURS = (1.0, 2.0, 3.0, 5.0)
COND = [f"{d:g}s" for d in DURS] + ["full"]
ONSETS = ROOT / "data/lid/fleurs/onsets.json"
CACHE = ROOT / "data/lid/cache"
PREDS = ROOT / "data/lid/preds"
SERVED = ROOT / "runs/stage1_served.afm"
JSON = ROOT / "runs/lid.json"
PRE_ROLL = 0.1  # s before the Silero onset
HARD_PAIRS = [("he", "ar"), ("ar", "fa"), ("es", "pt"), ("es", "it"), ("pt", "it"), ("ru", "uk"), ("ru", "pl"),
              ("uk", "pl"), ("de", "nl"), ("en", "nl"), ("zh", "ja"), ("hi", "fa")]


def _json() -> dict:
    return json.loads(JSON.read_text()) if JSON.exists() else {}


def _merge(key: str, rec, sub: str | None = None):
    res = _json()
    if sub is None:
        res[key] = rec
    else:
        res.setdefault(key, {})[sub] = rec
    JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def load_onsets() -> dict:
    return json.loads(ONSETS.read_text()) if ONSETS.exists() else {}


def clip(audio: np.ndarray, onset: float, cond: str) -> np.ndarray:
    """The clip for condition ``cond`` ("1s" .. "5s" from the speech onset, or "full")."""
    if cond == "full":
        return audio
    s = int(max(0.0, onset - PRE_ROLL) * SR)
    return audio[s: s + int(float(cond[:-1]) * SR)]


def test_rows(langs=None, n: int | None = None) -> list[dict]:
    rows = read_manifest("test", langs)
    if n:
        out = []
        for lang in CODES:
            out += [r for r in rows if r["lang"] == lang][:n]
        rows = out
    return rows


def rows_for(set_: str) -> list[dict]:
    return test_rows() if set_ == "fleurs" else read_edacc()


def load_served(device="cpu"):
    from audioforge.train import load_model
    return load_model(SERVED, device)


# --------------------------------------------------------------------------- onsets
def stage_onsets(a):
    from audioforge.baselines.turn import CHUNK, SileroVAD
    torch.set_num_threads(a.threads)
    vad = SileroVAD(ROOT / "data/silero/silero_vad_v5.onnx")
    res = load_onsets()
    rows = [r for r in read_manifest() + read_edacc() if r["id"] not in res]
    t0 = time.time()
    for i, r in enumerate(rows):
        p = vad.probs(load_audio(r))
        on = p >= 0.5
        run = np.convolve(on.astype(int), np.ones(3, int), "valid") == 3  # 3 consecutive 32 ms chunks
        idx = np.nonzero(run)[0]
        res[r["id"]] = ([round(float(idx[0]) * CHUNK / SR, 3), round(float(idx[-1] + 3) * CHUNK / SR, 3)]
                        if len(idx) else [0.0, r["duration"]])
        if time.time() - t0 > a.budget:
            break
    ONSETS.write_text(json.dumps(res))
    left = len([r for r in read_manifest() + read_edacc() if r["id"] not in res])
    print(f"[onsets] {len(res)} done, {left} left, {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- probe
@torch.no_grad()
def encode_clips(model, clips: list[np.ndarray]):
    """-> (enc, elen, hidden list) of the padded batch (CPU)."""
    lens = torch.tensor([len(c) for c in clips])
    x = torch.zeros(len(clips), int(lens.max()))
    for i, c in enumerate(clips):
        x[i, : len(c)] = torch.from_numpy(c)
    return model.encode(x, lens, return_hidden=True)


def stage_probe_feats(a):
    torch.set_num_threads(a.threads)
    model = load_served()
    ons = load_onsets()
    t0 = time.time()
    for lang in CODES:
        out = CACHE / f"probe_{a.split}_{lang}.npz"
        if out.exists():
            continue
        if time.time() - t0 > a.budget:
            print("[probe_feats] budget reached; re-run", flush=True)
            return
        rows = read_manifest(a.split, [lang])
        feats = []  # (N, 2, L, D): mean over the first 2 s and the whole 5 s clip
        for i in range(0, len(rows), a.batch_size):
            b = rows[i: i + a.batch_size]
            clips = [clip(load_audio(r), ons[r["id"]][0], "5s") for r in b]
            enc, elen, hidden = encode_clips(model, clips)
            H = torch.stack(hidden, 1)  # (B, L, T, D)
            T = H.shape[2]
            t = torch.arange(T)
            for k, n in enumerate(elen.tolist()):
                n2 = min(n, int(round(2.0 / model.frame_sec)))
                feats.append(torch.stack([H[k, :, :n2].mean(1), H[k, :, :n].mean(1)]).half().numpy())
            del t
        np.savez(out, X=np.stack(feats), ids=np.array([r["id"] for r in rows]))
        print(f"[probe_feats] {a.split} {lang}: {len(rows)} utts, {time.time() - t0:.0f}s", flush=True)
    print("[probe_feats] all languages done", flush=True)


def _load_probe(split):
    X, y = [], []
    for i, lang in enumerate(CODES):
        d = np.load(CACHE / f"probe_{split}_{lang}.npz")
        X.append(d["X"].astype(np.float32))
        y += [i] * len(d["X"])
    return np.concatenate(X), np.array(y)


def metrics(y, pred, n_classes=len(CODES)) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    return {"acc": round(float((y == pred).mean()), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", labels=list(range(n_classes)))), 4),
            "n": int(len(y)), "confusion": cm.tolist()}


def stage_probe(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(a.threads)
    Xtr, ytr = _load_probe("train")
    Xte, yte = _load_probe("test")
    rec = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "C": a.C, "per_block": []}
    for li in range(Xtr.shape[2]):
        row = {"block": li + 1}
        for di, dur in enumerate(("2s", "5s")):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
            clf.fit(Xtr[:, di, li], ytr)
            m = metrics(yte, clf.predict(Xte[:, di, li]))
            row[dur] = {"acc": m["acc"], "macro_f1": m["macro_f1"]}
        rec["per_block"].append(row)
        print(f"block {li + 1:2d}: 2s acc {row['2s']['acc']:.3f} F1 {row['2s']['macro_f1']:.3f} | "
              f"5s acc {row['5s']['acc']:.3f} F1 {row['5s']['macro_f1']:.3f}", flush=True)
    best = max(rec["per_block"], key=lambda r: r["5s"]["acc"])
    rec["best_block_5s"] = best["block"]
    _merge("probe", rec)


# --------------------------------------------------------------------------- head training (MPS, head only)
HEAD_CFG = {"type": "language", "num_languages": len(CODES), "hidden": 256, "att_hidden": 128, "cls_hidden": 256,
            "dropout": 0.1, "min_frames": 6, "labels": CODES, "weight": 1.0}
TRAINED = ("heads.lid.", "layer_mix.lid")


def build_lid_model(layers, device, dropout=0.1):
    """The served model + an untrained heads.lid on encoder block(s) ``layers`` (1-based, as the probe table)."""
    import copy
    from audioforge.model import SpeechModel
    from audioforge.train import load_model
    served = load_model(SERVED, "cpu")
    cfg = copy.deepcopy(served.cfg)
    cfg["heads"]["lid"] = {**HEAD_CFG, "dropout": float(dropout), "from_layers": [int(k) - 1 for k in layers]}
    model = SpeechModel(cfg, served.tokenizer)
    missing, unexpected = model.load_state_dict(served.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(TRAINED) for k in missing), (missing[:3], unexpected[:3])
    for k, p in model.named_parameters():
        p.requires_grad = k.startswith(TRAINED)
    return model.to(device), served


def _crop_batch(rows, ons, rng, B, L):
    """B random crops of L s from train rows (a quarter start at the speech onset, as the evaluation clips)."""
    out = []
    for r in (rows[i] for i in rng.integers(0, len(rows), B)):
        a = load_audio(r)
        n = int(L * SR)
        if rng.random() < 0.25:
            s = int(max(0.0, ons[r["id"]][0] - PRE_ROLL) * SR)
        else:
            s = int(rng.integers(0, max(1, len(a) - n + 1)))
        c = a[s: s + n] * float(10 ** (rng.uniform(-6, 0) / 20))  # gain -6..0 dB below the -1 dBFS peak
        out.append((c.astype(np.float32), CODES.index(r["lang"])))
    return out


@torch.no_grad()
def _dev_feats(model, rows, ons, device, cond):
    """(list of per-utterance head inputs, labels) of the dev clips for ``cond`` (encoder once, kept on device)."""
    feats, ys = [], []
    for i in range(0, len(rows), 4):  # small batches: MPS memory (the watermark caps the process at ~10 GB)
        b = rows[i: i + 4]
        clips = [clip(load_audio(r), ons[r["id"]][0], cond)[: 20 * SR] for r in b]
        lens = torch.tensor([len(c) for c in clips])
        x = torch.zeros(len(b), int(lens.max()))
        for j, c in enumerate(clips):
            x[j, : len(c)] = torch.from_numpy(c)
        enc, elen, hid = model.encode(x.to(device), lens.to(device), return_hidden=True)
        e = model.head_input("lid", enc, hid)
        for j in range(len(b)):
            feats.append(e[j: j + 1, : int(elen[j])].detach())
            ys.append(CODES.index(b[j]["lang"]))
    return feats, torch.tensor(ys)


@torch.no_grad()
def _dev_acc(head, feats, ys):
    head.eval()
    pred = torch.stack([head(f, torch.tensor([f.shape[1]], device=f.device))[0].argmax() for f in feats]).cpu()
    head.train()
    return float((pred == ys).float().mean())


def stage_train(a):
    from audioforge.lid import save_head
    dev = torch.device(a.device)
    layers = [int(x) for x in a.layers.split(",")]
    state_path = CACHE / f"train_state_{a.tag}.pt"
    model, served = build_lid_model(layers, dev, a.dropout)
    model.eval()
    head = model.heads["lid"].train()
    if a.aug:  # input-side augmentation through the FROZEN encoder: SpecAugment on the mel + the encoder's own
        # dropout (model.train() only switches dropout / dither / SpecAugment on; no parameter or buffer changes)
        from audioforge.features import SpecAugment
        model.spec_augment = SpecAugment(freq_masks=2, freq_width=15, time_masks=4, time_width=0.05).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    st = {"step": 0, "log": [], "best": -1.0, "best_state": None, "sec": 0.0}
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict({**model.state_dict(), **st["trained"]})
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
    rows = read_manifest("train")
    ons = load_onsets()
    dev_rows = read_manifest("dev")
    dfe = {c: _dev_feats(model, dev_rows, ons, dev, c) for c in ("2s", "full")}
    rng = np.random.default_rng(1000 + st["step"])
    if dev.type == "mps":
        torch.mps.empty_cache()
    t0 = time.time()
    while st["step"] < a.steps and time.time() - t0 < a.segment:
        L = float(rng.choice(np.arange(1.0, 8.01, 0.5)))
        b = _crop_batch(rows, ons, rng, a.batch_size, L)
        lens = torch.tensor([len(c) for c, _ in b])
        x = torch.zeros(len(b), int(lens.max()))
        for j, (c, _) in enumerate(b):
            x[j, : len(c)] = torch.from_numpy(c)
        y = torch.tensor([t for _, t in b], device=dev)
        model.train(bool(a.aug))
        head.train()
        with torch.no_grad():
            enc, elen, hid = model.encode(x.to(dev), lens.to(dev), return_hidden=True)
        e = model.head_input("lid", enc, [h.detach() for h in hid])
        loss = head.loss(e, elen, {"lang": y})
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        st["step"] += 1
        if st["step"] % a.eval_every == 0 or st["step"] == a.steps:
            model.eval()
            acc = {c: round(_dev_acc(head, *dfe[c]), 4) for c in dfe}
            st["log"].append({"step": st["step"], "loss": round(float(loss), 4), "dev_acc": acc,
                              "sec": round(st["sec"] + time.time() - t0, 1)})
            print(st["log"][-1], flush=True)
            score = acc["2s"] + acc["full"]
            if score > st["best"]:
                st["best"] = score
                st["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                                    if k.startswith(TRAINED)}
                st["best_step"] = st["step"]
    st["sec"] += time.time() - t0
    st["trained"] = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(TRAINED)}
    st["opt"], st["sched"] = opt.state_dict(), sched.state_dict()
    torch.save(st, state_path)
    if st["step"] < a.steps:
        print(f"[train] segment done at step {st['step']}/{a.steps}; re-run to resume", flush=True)
        return
    # final: best dev checkpoint, frozen-tensor check against the served model, head file
    model.load_state_dict({**model.state_dict(), **st["best_state"]})
    model.spec_augment = None
    model = model.cpu().eval()
    src = served.state_dict()
    sd = model.state_dict()
    bad = [k for k in src if not torch.equal(src[k], sd[k])]
    assert not bad, f"tensors outside heads.lid changed: {bad[:5]}"
    extra = [k for k in sd if k not in src]
    assert all(k.startswith(TRAINED) for k in extra)
    out = Path(a.out)
    save_head(model, "lid", out, meta={"layers_1based": layers, "steps": a.steps, "best_step": st["best_step"],
                                        "train": "FLEURS 17 languages x 250 utts (scripts/research/lid_data.py)"})
    rec = {"tag": a.tag, "layers_1based": layers, "steps": a.steps, "batch_size": a.batch_size, "lr": a.lr,
           "aug": bool(a.aug), "dropout": a.dropout,
           "best_step": st["best_step"], "sec": round(st["sec"], 1), "device": a.device, "log": st["log"],
           "frozen_tensors_identical": len(src), "trained_tensors": extra,
           "head_params": int(sum(p.numel() for p in model.heads["lid"].parameters())), "out": str(out)}
    _merge("train", rec, a.tag)
    print(f"[train] {a.tag}: best step {st['best_step']}, {len(src)} frozen tensors identical, head "
          f"{rec['head_params']} params -> {out}", flush=True)
    print("[train] finished", flush=True)


# --------------------------------------------------------------------------- evaluation of the head (CPU)
def _pred_path(tag):
    PREDS.mkdir(parents=True, exist_ok=True)
    return PREDS / f"{tag}.npz"


@torch.no_grad()
def stage_eval_head(a):
    """Resumable per condition (data/lid/preds/<tag>/<cond>.npz); assembled into <tag>.npz / <tag>_vadgated.npz."""
    from audioforge.lid import attach_head
    torch.set_num_threads(a.threads)
    model = load_served()
    name = attach_head(model, a.head)
    head = model.heads[name]
    ons = load_onsets()
    rows = rows_for(a.set)
    tag = (a.tag or Path(a.head).stem) + ("_edacc" if a.set == "edacc" else "")
    part = PREDS / f"_{tag}"
    part.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ci, cond in enumerate(COND):
        f = part / f"{cond}.npz"
        if f.exists():
            continue
        if time.time() - t0 > a.budget:
            print("[eval_head] budget reached; re-run", flush=True)
            return
        P = np.zeros((len(rows), len(CODES)), np.float32)
        G = np.zeros_like(P)  # VAD-gated (the server rule: frames with VAD <= 0.5 not pooled)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
        for s in range(0, len(order), a.batch_size):
            idx = order[s: s + a.batch_size]
            clips = [clip(load_audio(rows[i]), ons[rows[i]["id"]][0], cond) for i in idx]
            enc, elen, hid = encode_clips(model, clips)
            e = model.head_input(name, enc, hid)
            P[idx] = head(e, elen).softmax(-1).numpy()
            vad = model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()
            w, wx, wxx = head._terms(e)
            valid = (torch.arange(e.shape[1])[None] < elen[:, None]).float()[..., None]
            m = valid * (vad > 0.5).float()[..., None]
            m = torch.where(m.sum(1, keepdim=True) > 0, m, valid)  # no speech frame: pool everything
            G[idx] = head._classify((w * m).sum(1), (wx * m).sum(1), (wxx * m).sum(1)).softmax(-1).numpy()
        np.savez(f, P=P, G=G)
        print(f"[eval_head] {tag} {cond} {time.time() - t0:.0f}s", flush=True)
    P = np.stack([np.load(part / f"{c}.npz")["P"] for c in COND], 1)
    G = np.stack([np.load(part / f"{c}.npz")["G"] for c in COND], 1)
    y = np.array([CODES.index(r["lang"]) for r in rows])
    ids = np.array([r["id"] for r in rows])
    np.savez(_pred_path(tag), P=P, y=y, ids=ids)
    np.savez(_pred_path(tag + "_vadgated"), P=G, y=y, ids=ids)
    rec = {"head": a.head, "n": len(rows)}
    if a.set == "fleurs":
        rec["cost"] = head_cost(model, name, a.threads)
        print(rec["cost"], flush=True)
    _merge("eval_head", rec, tag)
    print("[eval_head] all conditions done", flush=True)


@torch.no_grad()
def head_cost(model, name, threads, seconds=10.0, reps=3):
    """Per-160 ms-chunk CPU cost (ms) on ``threads`` threads: encoder stream_step (fast_conv, as served) and the LID
    head's streaming step on the same chunk's per-layer outputs; one 10 s FLEURS utterance, best of ``reps``."""
    from audioforge.serve import ASRStream, fast_conv
    torch.set_num_threads(threads)
    fast_conv(model)
    a = load_audio(test_rows(["en"])[0])[: int(seconds * SR)]
    best = None
    for _ in range(reps):
        s = ASRStream(model, vad="vad", turn=None)
        st = model.heads[name].init_stream(1)
        mel = s._mel(0, (len(a) - s.half) // s.hop + 1)
        enc_ms, lid_ms = [], []
        state = s.enc_state
        while mel.shape[-1] >= s.chunk_mel:
            chunk, mel = mel[..., : s.chunk_mel], mel[..., s.chunk_mel:]
            t0 = time.perf_counter()
            enc, hid, state = model.encoder.stream_step(chunk, state, s.att, return_hidden=True)
            t1 = time.perf_counter()
            model.heads[name].step(model.head_input(name, enc, hid), st).softmax(-1)
            t2 = time.perf_counter()
            enc_ms.append((t1 - t0) * 1000)
            lid_ms.append((t2 - t1) * 1000)
        r = {"chunks": len(enc_ms), "encoder_ms_p50": round(float(np.median(enc_ms)), 3),
             "lid_head_ms_p50": round(float(np.median(lid_ms)), 3), "lid_head_ms_mean": round(float(np.mean(lid_ms)), 3),
             "lid_rtf": round(float(np.mean(lid_ms)) / 160.0, 5), "threads": threads}
        best = r if best is None or r["lid_head_ms_p50"] < best["lid_head_ms_p50"] else best
    best["head_params"] = int(sum(p.numel() for p in model.heads[name].parameters()))
    return best


# --------------------------------------------------------------------------- the serving rule (audioforge.lid)
def _rule_stats(events_list, ys):
    first_ok, first_t, flips, announced = [], [], [], 0
    for ev, y in zip(events_list, ys):
        if ev:
            announced += 1
            first_ok.append(ev[0]["language"] == y)
            first_t.append(ev[0]["t"])
            flips.append(len(ev) - 1)
    return {"announced": round(announced / len(ys), 4),
            "first_acc": round(float(np.mean(first_ok)), 4) if first_ok else None,
            "first_t_p50": round(float(np.median(first_t)), 2) if first_t else None,
            "first_t_p90": round(float(np.percentile(first_t, 90)), 2) if first_t else None,
            "flips_per_utt": round(float(np.mean(flips)), 3) if flips else None, "n": len(ys)}


@torch.no_grad()
def stage_decide(a):
    """The server's announcement rule on whole utterances streamed from the file start: per threshold, the fraction
    of utterances with an announcement, the accuracy of the FIRST announcement, its time (s from file start) and the
    flips per utterance. Head backend: VAD-gated running posterior, 30 s half-life (audioforge.lid.LIDStream's
    numbers, computed offline = streaming). ``--system ambernet``: also the AmberNet backend
    (audioforge.lid.AmberNetLIDStream: re-run on the last 8 s of VAD speech every 480 ms) on the same utterances.
    ``--n``: utterances per language (default all). Thresholds are compared on dev; test is reported for all."""
    from audioforge.lid import AmberNetLIDStream, LangDecider, attach_head
    torch.set_num_threads(a.threads)
    model = load_served()
    name = attach_head(model, a.head)
    head = model.heads[name]
    fs = model.frame_sec
    decay = 0.5 ** (fs / 30.0)
    thresholds = (0.5, 0.7, 0.8, 0.9, 0.95)
    amber = _make_system("ambernet") if a.system == "ambernet" else None
    rec = {}
    for split in ("dev", "test"):
        rows = read_manifest(split)
        if a.n:
            rows = [r for l in CODES for r in [x for x in rows if x["lang"] == l][: a.n]]
        post, keeps, vads, audios, ys = [], [], [], [], []
        for i in range(0, len(rows), a.batch_size):
            b = rows[i: i + a.batch_size]
            au = [load_audio(r) for r in b]
            enc, elen, hid = encode_clips(model, au)
            x = model.head_input(name, enc, hid)
            vad = model.heads["vad"](model.head_input("vad", enc, hid)).sigmoid()
            for j, r in enumerate(b):
                n = int(elen[j])
                keep = vad[j: j + 1, :n] > 0.5
                st = head.init_stream(1)
                post.append(head.step(x[j: j + 1, :n], st, keep=keep, decay=decay).softmax(-1)[0].numpy())
                keeps.append(keep[0].numpy())
                vads.append(vad[j, :n].numpy())
                audios.append(au[j])
                ys.append(r["lang"])
        out = {"head": {}, "ambernet": {}}
        for th in thresholds:
            evs = []
            for P, K in zip(post, keeps):
                d = LangDecider(CODES, threshold=th, min_ms=a.min_ms, frame_ms=fs * 1000)
                evs.append([e for t in range(len(P)) if K[t] for e in [d.update(P[t], (t + 1) * fs)] if e])
            out["head"][str(th)] = _rule_stats(evs, ys)
            print(split, "head", th, out["head"][str(th)], flush=True)
        if amber is not None:
            for th in (0.8, 0.9):
                evs, ms, runs = [], 0.0, 0
                for au, vd in zip(audios, vads):
                    s_ = AmberNetLIDStream(amber, CODES, threshold=th, min_ms=a.min_ms)
                    s_.feed_audio(au)
                    evs.append(s_.feed(None, None, vd.tolist(), [(t + 1) * fs for t in range(len(vd))]))
                    ms, runs = ms + s_.ms, runs + s_.runs
                out["ambernet"][str(th)] = {**_rule_stats(evs, ys), "ms_per_run": round(ms / max(1, runs), 1),
                                            "runs_per_utt": round(runs / len(ys), 2)}
                print(split, "ambernet", th, out["ambernet"][str(th)], flush=True)
        rec[split] = out
    _merge("decide", {"head": a.head, "min_ms": a.min_ms, "half_life_s": 30.0, "n_per_lang": a.n, **rec},
           Path(a.head).stem + ("_vs_ambernet" if amber is not None else ""))


# --------------------------------------------------------------------------- dedicated baselines (CPU)
def _make_system(name):
    from audioforge.baselines import lid as B
    if name == "ambernet":
        from audioforge.nemo_import import import_ambernet
        return import_ambernet()
    if name == "speechbrain":
        return B.SpeechBrainLID(savedir=str(ROOT / "data/lid/models/speechbrain_voxlingua107_ecapa"))
    if name == "whisper_base":  # the converted weights are cached locally; no extra download
        return B.WhisperCT2LID("base", threads=2)
    if name.startswith("whisper_"):
        return B.WhisperLID(f"openai/whisper-{name.split('_', 1)[1]}")
    raise SystemExit(f"unknown system {name}")


def stage_baseline(a):
    from audioforge.baselines.lid import restrict
    torch.set_num_threads(a.threads)
    sysm = _make_system(a.system)
    ons = load_onsets()
    d = PREDS / a.system
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for lang in (CODES if a.set == "fleurs" else ["edacc"]):
        out = d / f"{lang}.npz"
        if out.exists():
            continue
        if time.time() - t0 > a.budget:
            print("[baseline] budget reached; re-run", flush=True)
            return
        rows = test_rows([lang]) if a.set == "fleurs" else read_edacc()
        P = np.zeros((len(rows), len(COND), len(CODES)), np.float32)
        sec = np.zeros((len(rows), len(COND)))
        for i, r in enumerate(rows):
            audio = load_audio(r)
            for ci, cond in enumerate(COND):
                c = clip(audio, ons[r["id"]][0], cond)
                t1 = time.perf_counter()
                P[i, ci] = restrict(sysm.probs(c), CODES)
                sec[i, ci] = time.perf_counter() - t1
        np.savez(out, P=P, sec=sec, ids=np.array([r["id"] for r in rows]))
        print(f"[baseline] {a.system} {lang}: acc full {np.mean(P[:, -1].argmax(-1) == CODES.index(rows[0]['lang'])):.3f}, "
              f"{time.time() - t0:.0f}s", flush=True)
    print("[baseline] all languages done", flush=True)


def stage_cost(a):
    """CPU cost on ``--threads`` threads: ms per call of each dedicated model on 1 / 2 / 3 / 5 / 8 s clips (median of 5
    after a warm-up), and the head's ms per 160 ms chunk (head_cost). The 1-min load average is recorded (the Mac is
    shared; costs under load are upper bounds)."""
    import os
    torch.set_num_threads(a.threads)
    rows = test_rows(["en"])
    audio = np.concatenate([load_audio(r) for r in rows[:3]])
    old = _json().get("cost", {})
    rec = {"threads": a.threads, "load_1min_start": round(os.getloadavg()[0], 2),
           "per_call_ms": old.get("per_call_ms", {}), "params": old.get("params", {})}
    for name in (a.rest or ["ambernet", "speechbrain", "whisper_tiny", "whisper_base"]):
        m = _make_system(name)
        m.probs(audio[: SR])
        r = {}
        for d in (1, 2, 3, 5, 8):
            x = audio[: d * SR]
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                m.probs(x)
                ts.append((time.perf_counter() - t0) * 1000)
            r[f"{d}s"] = round(float(np.median(ts)), 1)
        rec["per_call_ms"][name] = r
        mod = getattr(m, "model", m)
        try:
            rec["params"][name] = int(sum(p.numel() for p in mod.parameters()))
        except Exception:
            rec["params"][name] = None
        print(name, r, rec["params"][name], flush=True)
    from audioforge.lid import attach_head
    model = load_served()
    nm = attach_head(model, a.head)
    rec["head"] = head_cost(model, nm, a.threads, reps=5)
    rec["load_1min_end"] = round(os.getloadavg()[0], 2)
    print(rec["head"], flush=True)
    _merge("cost", rec)


def _load_preds(tag):
    f = PREDS / f"{tag}.npz"
    if f.exists():
        d = np.load(f)
        return d["P"], d["y"], None
    d = PREDS / tag
    if not d.is_dir() or not all((d / f"{l}.npz").exists() for l in CODES):
        return None
    P, y, sec = [], [], []
    for i, l in enumerate(CODES):
        z = np.load(d / f"{l}.npz")
        P.append(z["P"])
        sec.append(z["sec"])
        y += [i] * len(z["P"])
    return np.concatenate(P), np.array(y), np.concatenate(sec)


def stage_report(a):
    systems = {}
    tags = a.rest or [t for t in ["lid_head", "lid_head_vadgated", "lid_aug", "lid_aug_vadgated", "ambernet", "speechbrain", "whisper_tiny",
                                  "whisper_base"]]
    for tag in tags:
        got = _load_preds(tag)
        if got is None:
            continue
        P, y, sec = got
        rec = {}
        for ci, cond in enumerate(COND):
            m = metrics(y, P[:, ci].argmax(-1))
            cm = np.array(m["confusion"])
            pairs = {f"{p}->{q}": round(float(cm[CODES.index(p), CODES.index(q)] / max(1, cm[CODES.index(p)].sum())), 4)
                     for a_, b_ in HARD_PAIRS for p, q in ((a_, b_), (b_, a_))}
            per_lang = {l: round(float(cm[i, i] / max(1, cm[i].sum())), 4) for i, l in enumerate(CODES)}
            ok = (P[:, ci].argmax(-1) == y).astype(np.float64)
            bs = np.random.default_rng(0).integers(0, len(ok), (1000, len(ok)))
            ci95 = [round(float(np.percentile(ok[bs].mean(1), q)), 4) for q in (2.5, 97.5)]
            rec[cond] = {**m, "acc_ci95": ci95, "pairs": pairs, "per_lang_acc": per_lang}
            if sec is not None:
                rec[cond]["ms_per_call_mean"] = round(float(sec[:, ci].mean() * 1000), 2)
        systems[tag] = rec
    _merge("systems", systems)
    print("| system | " + " | ".join(f"{c} acc / F1" for c in COND) + " |")
    print("|---|" + "---|" * len(COND))
    for tag, rec in systems.items():
        print(f"| {tag} | " + " | ".join(f"{100 * rec[c]['acc']:.1f} / {100 * rec[c]['macro_f1']:.1f}" for c in COND) + " |")
    acc = {}
    for tag in tags:
        base = tag.replace("_vadgated", "")
        f = PREDS / f"{tag}_edacc.npz" if base.startswith("lid") else PREDS / tag / "edacc.npz"
        if not f.exists():
            continue
        P = np.load(f)["P"]
        er = read_edacc()
        l1 = np.array([r["l1"] for r in er])
        pred = P.argmax(-1)  # (N, cond)
        rec = {c: {"acc_en": round(float((pred[:, ci] == 0).mean()), 4),
                   "top_wrong": sorted(((CODES[k], int(((pred[:, ci] == k)).sum())) for k in range(1, len(CODES))
                                        if (pred[:, ci] == k).any()), key=lambda x: -x[1])[:4]}
               for ci, c in enumerate(COND)}
        rec["by_l1_full"] = {u: round(float((pred[l1 == u, -1] == 0).mean()), 3) for u in sorted(set(l1))}
        rec["n"] = int(len(P))
        acc[tag] = rec
    if acc:
        _merge("edacc", acc)
        print("\nEdAcc accented English (fraction classified en among the 17):")
        for tag, rec in acc.items():
            print(tag, " ".join(f"{c}:{100 * rec[c]['acc_en']:.1f}" for c in COND), rec["full"]["top_wrong"])
    print("\nhard pairs (full / 5s):")
    for tag, rec in systems.items():
        print(tag, {k: (rec["full"]["pairs"][k], rec["5s"]["pairs"][k]) for k in rec["full"]["pairs"]
                    if rec["full"]["pairs"][k] >= 0.02 or rec["5s"]["pairs"][k] >= 0.02})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage")
    ap.add_argument("--split", default="train")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--budget", type=float, default=480.0)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--recipe", default=None, help="train: research/recipes/lid_*.yaml (overrides the train arguments)")
    ap.add_argument("--layers", default="12", help="train: encoder block(s), 1-based, comma-separated")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="runs/lid_head.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--segment", type=float, default=480.0, help="train: seconds per process before checkpointing")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--aug", action="store_true", help="train: SpecAugment + encoder dropout on the frozen encoder")
    ap.add_argument("--dropout", type=float, default=0.1, help="train: head dropout")
    ap.add_argument("--head", default="runs/lid_head.pt")
    ap.add_argument("--system", default="ambernet")
    ap.add_argument("--min-ms", type=float, default=1000.0)
    ap.add_argument("--n", type=int, default=None, help="decide: utterances per language")
    ap.add_argument("--set", default="fleurs", choices=("fleurs", "edacc"))
    a, rest = ap.parse_known_args()
    fn = globals().get(f"stage_{a.stage}")
    if fn is None:
        raise SystemExit(f"unknown stage {a.stage}")
    a.rest = rest
    if a.stage == "train" and a.recipe:  # research/recipes/lid_*.yaml: the same arguments as a file
        import yaml
        r = yaml.safe_load(Path(a.recipe).read_text())
        assert r.get("init", "runs/stage1_served.afm") == "runs/stage1_served.afm"
        a.layers = ",".join(str(k) for k in r["layers"])
        a.steps, a.lr, a.dropout, a.aug, a.out = r["steps"], r["lr"], r["dropout"], bool(r["aug"]), r["out"]
        a.batch_size, a.tag = r["batch_size"], r["name"]
    if a.stage == "train":
        a.tag = a.tag or Path(a.out).stem
        if a.batch_size == 8:
            a.batch_size = 32
    fn(a)


if __name__ == "__main__":
    main()
