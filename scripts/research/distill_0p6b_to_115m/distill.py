"""Distil nemotron-speech-streaming-en-0.6b into the served 115M streaming core (research/archive/IMPROVEMENTS.md section 7).

Stages (each resumable; see README.md in this directory for the exact order, paths and GPU-hour estimates):

  manifest   the training item list: every single-speaker 1-15 s segment of the AMI / ICSI train meetings (the
             library's asr mode) + a LibriSpeech train-clean-100 anchor subset -> <work>/manifest.jsonl
  evalsets   AMI-200 / ICSI-200 / LibriSpeech-200 exactly as research/archive/HYBRID_ASR.md (seeded 200-segment samples of
             the AMI dev (4) / ICSI dev (2) meetings; the first 200 test-clean utterances) -> <work>/eval/*.npz
  teacher    the 0.6B at [70,1] (and [70,13] with --contexts 1,13) over every manifest item: greedy RNNT transcript
             (normalize_text'ed, the student's text target) + top encoder layer (T, 1024) fp16 -> <work>/teacher/
             shard_XXXX.npz (256 items each); ~16 GB per context for ~170 h
  train      the student: the whole 115M encoder + RNNT (pred + joint) of runs/stage1_served.afm; losses (per batch):
               meeting items   w_gt * RNNT(reference) + w_seq * RNNT(teacher transcript)
                               + w_enc * (1 - cos(P h_s, h_t)) on valid frames (P: Linear(512, 1024), trained)
               anchor items    RNNT(reference) + w_kl * KL(student RNNT joint || frozen initial student) (the
                               repo's R2 anchor, audioforge.losses.consistency.head_kl)
             layer-wise LR decay (block i of 17 at lr * decay^(17 - i), pre-encode at lr * decay^18, heads at lr),
             AdamW, warm-up + cosine, bf16 autocast on CUDA; WER gate every --gate-every steps on LibriSpeech
             test-clean (first --gate-n utterances, [70,1]): WER > baseline + --gate-max-delta -> the last good
             checkpoint is restored and the run stops ("gate_fired" in the log); checkpoints every --ckpt-every steps
             (<work>/<tag>/ckpt.pt, resumable), final <work>/<tag>/student.afm. --control: the same run without any
             teacher term (w_seq = w_enc = 0): continued training at the same budget.
  eval       AMI-200 / ICSI-200 / LibriSpeech-200 WER at [70,1] and [70,13] for the served model, the distilled
             student and the control (normalize_text, 1000-resample CIs, paired deltas) -> runs/distill_0p6b.json
  heads      the served VAD head on the new encoder (BASELINES 64 x 20 s AMI dev and ICSI dev windows, F1 at 0.5)
             vs the served model; the turn-head check command is in README.md

  python scripts/research/distill_0p6b_to_115m/distill.py <stage> --work <dir> [options]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
HERE = Path(__file__).resolve().parent
SR = 16000
SHARD = 256


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dev_of(a):
    if a.device != "auto":
        return a.device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------------------------------------------------------------- data
class Audio:
    """Audio of a manifest item (AMI / ICSI segment or LibriSpeech utterance); datasets opened lazily with exactly
    the meetings the manifest uses."""

    def __init__(self, items=None, work=None):
        self.work = Path(work) if work is not None else None
        self.meetings = {}
        for it in items or []:
            if it["src"] != "libri":
                self.meetings.setdefault(it["src"], set()).add(it["meeting"])
        self.ds = {}

    def _open(self, src):
        if src == "libri":
            from audioforge.datasets.librispeech import LibriSpeech
            return LibriSpeech("train-clean-100", max_sec=16.0, verbose=False)
        ms = sorted(self.meetings.get(src, []))
        if src == "ami":
            from audioforge.datasets.ami import AMI
            return AMI(ms, verbose=False)
        from audioforge.datasets.icsi import ICSI
        return ICSI(ms, verbose=False)

    def __call__(self, it: dict) -> np.ndarray:
        src = it["src"]
        if src not in self.ds:
            self.ds[src] = self._open(src)
        if src == "libri":
            return np.asarray(self.ds[src].audio(it["idx"]), np.float32)
        return np.asarray(self.ds[src]._clip(it["meeting"], it["a"], it["b"]), np.float32)


def stage_manifest(a):
    """AMI / ICSI train single-speaker segments (1-15 s) of the meetings prepared under the data root + the first
    --n-libri LibriSpeech train-clean-100 utterances (<= 16 s, seeded sample) -> manifest.jsonl."""
    from audioforge.datasets.ami import AMI, split_lists, subset
    from audioforge.datasets.icsi import SPLITS, ICSI, subset as isub
    from audioforge.datasets.librispeech import LibriSpeech
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    items = []
    ami_m = subset({"train": a.n_ami or len(split_lists()["train"])})["train"]  # the library's order: defaults first
    icsi_m = isub({"train": a.n_icsi or len(SPLITS["train"])})["train"]
    for src, ds in (("ami", AMI(ami_m, verbose=False)), ("icsi", ICSI(icsi_m, verbose=False))):
        for ex in ds.asr(1.0, 15.0):
            items.append({"src": src, "meeting": ex["meeting"], "a": round(ex["start"], 3),
                          "b": round(ex["start"] + ex["duration"], 3), "text": ex["text"], "dur": round(ex["duration"], 2)})
        log(f"{src}: {len(ds.meetings)} meetings, {sum(1 for i in items if i['src'] == src)} segments")
    ls = LibriSpeech("train-clean-100", max_sec=16.0, verbose=False)
    idx = sorted(random.Random(0).sample(range(len(ls)), min(a.n_libri, len(ls))))
    for i in idx:
        u = ls.utts[i]
        items.append({"src": "libri", "idx": i, "text": u["text"], "dur": round(u["n"] / SR, 2)})
    if a.max_hours:  # smoke tests: a seeded subset of at most this many hours, all sources kept
        random.Random(0).shuffle(items)
        out, tot = [], 0.0
        for it in items:
            if tot >= a.max_hours * 3600:
                break
            out.append(it)
            tot += it["dur"]
        items = out
    for k, it in enumerate(items):
        it["i"] = k
    with (work / "manifest.jsonl").open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    hrs = {s: round(sum(i["dur"] for i in items if i["src"] == s) / 3600, 2) for s in ("ami", "icsi", "libri")}
    log(f"manifest: {len(items)} items, hours {hrs}")


def load_manifest(work) -> list[dict]:
    return [json.loads(x) for x in (Path(work) / "manifest.jsonl").read_text().splitlines()]


def stage_evalsets(a):
    """HYBRID_ASR.md's three 200-item sets, rebuilt from the data root (same seeds / meeting lists)."""
    from audioforge.data import load_wav
    from audioforge.datasets.ami import AMI, subset
    from audioforge.datasets.icsi import ICSI, subset as isub
    d = Path(a.work) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    for name, ds in (("ami", lambda: AMI(subset({"dev": 4})["dev"], verbose=False)),
                     ("icsi", lambda: ICSI(isub()["dev"], verbose=False))):
        if (d / f"{name}.npz").exists():
            continue
        segs = ds().asr(1.0, 15.0)
        idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
        np.savez(d / f"{name}.npz", **{f"a{i}": np.asarray(segs[j]["audio"], np.float32) for i, j in enumerate(idx)})
        (d / f"{name}_refs.json").write_text(json.dumps([segs[j]["text"] for j in idx]))
        log(f"evalset {name}: 200 of {len(segs)}")
    if not (d / "libri.npz").exists():
        lines = [json.loads(x) for x in (HERE / "test-clean-first200.jsonl").read_text().splitlines() if x.strip()]
        np.savez(d / "libri.npz", **{f"a{i}": load_wav(str(ROOT / x["audio_filepath"]), SR) for i, x in enumerate(lines)})
        (d / "libri_refs.json").write_text(json.dumps([x["text"] for x in lines]))
        log("evalset libri: 200")


def load_eval(work, name, n=None):
    d = Path(work) / "eval"
    z = np.load(d / f"{name}.npz")
    refs = json.loads((d / f"{name}_refs.json").read_text())
    n = len(refs) if n is None else min(n, len(refs))
    return [z[f"a{i}"] for i in range(n)], refs[:n]


# --------------------------------------------------------------------------- teacher
@torch.no_grad()
def stage_teacher(a):
    from audioforge.teachers import normalize_text
    from audioforge.train import load_model
    dev = dev_of(a)
    torch.set_num_threads(a.threads)
    work = Path(a.work)
    items = load_manifest(work)
    out = work / "teacher"
    out.mkdir(parents=True, exist_ok=True)
    ctxs = [int(c) for c in a.contexts.split(",")]
    m = load_model(a.teacher, dev).eval()
    audio = Audio(items, work)
    t0 = time.time()
    n_sh = math.ceil(len(items) / SHARD)
    part, nparts = (int(x) for x in a.part.split("/"))  # --part k/N: this process caches shards s % N == k
    for s in range(n_sh):
        p = out / f"shard_{s:05d}.npz"
        if s % nparts != part or p.exists():
            continue
        if time.time() - t0 > a.budget:
            log(f"teacher: budget, {s}/{n_sh} shards")
            return
        its = items[s * SHARD:(s + 1) * SHARD]
        its_sorted = sorted(range(len(its)), key=lambda k: its[k]["dur"])
        rec = {}
        for c in ctxs:
            texts, encs = [None] * len(its), [None] * len(its)
            for b in range(0, len(its_sorted), a.batch):
                ks = its_sorted[b:b + a.batch]
                xs = [audio(its[k]) for k in ks]
                x, lens = m._pad(xs)
                ctx = torch.autocast("cuda", torch.bfloat16) if dev == "cuda" else _null()
                with ctx:
                    enc, elen = m.encode(x.to(dev), lens.to(dev), [70, c])
                hyps = m.heads["rnnt"].decode(enc.float(), elen)
                for j, k in enumerate(ks):
                    texts[k] = normalize_text(m.tokenizer.decode(hyps[j]) if not isinstance(hyps[j], str) else hyps[j])
                    encs[k] = enc[j, : int(elen[j])].float().cpu().numpy().astype(np.float16)
            off = np.r_[0, np.cumsum([len(e) for e in encs])]
            rec[f"enc{c}"] = np.concatenate(encs)
            rec[f"off{c}"] = off
            rec[f"text{c}"] = np.array(texts)
        np.savez(p.with_suffix(".tmp.npz"), ids=np.array([it["i"] for it in its]), **rec)
        p.with_suffix(".tmp.npz").replace(p)
        z = np.load(p)  # exFAT / network-disk caveat: the shard must read back non-zero
        e0 = z[f"enc{ctxs[0]}"]
        assert len(e0) == 0 or np.any(e0[-20:].astype(np.float32)), f"{p} reads back zero-filled"
        log(f"teacher shard {s + 1}/{n_sh} ({time.time() - t0:.0f}s)")
    log("teacher: done")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TeacherIndex:
    def __init__(self, work, ctx=1):
        self.shards = sorted((Path(work) / "teacher").glob("shard_*.npz"))
        self.where, self.ctx, self.cache = {}, ctx, {}
        for si, p in enumerate(self.shards):
            for k, i in enumerate(np.load(p)["ids"]):
                self.where[int(i)] = (si, k)

    def get(self, i):
        si, k = self.where[i]
        if si not in self.cache:
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[si] = dict(np.load(self.shards[si]))
        z = self.cache[si]
        o = z[f"off{self.ctx}"]
        return str(z[f"text{self.ctx}"][k]), z[f"enc{self.ctx}"][o[k]: o[k + 1]]


# --------------------------------------------------------------------------- train
def param_groups(model, proj, lr, decay):
    enc = model.encoder
    L = len(enc.layers)
    groups, seen = [], set()

    def add(params, lrx, name):
        ps = [p for p in params if p.requires_grad and id(p) not in seen]
        seen.update(id(p) for p in ps)
        if ps:
            groups.append({"params": ps, "lr": lrx, "base_lr": lrx, "name": name})
    for i, layer in enumerate(enc.layers):
        add(layer.parameters(), lr * decay ** (L - 1 - i), f"block{i + 1}")
    add(enc.parameters(), lr * decay ** L, "pre_encode")  # everything else in the encoder (subsampling, kernels)
    add(model.heads["rnnt"].parameters(), lr, "rnnt")
    add(proj.parameters(), lr, "projector")
    return groups


def gate_wer(model, work, n, dev):
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    xs, refs = load_eval(work, "libri", n)
    model.eval()
    hyps = []
    with torch.no_grad():
        for i in range(0, len(xs), 8):
            hyps += model.transcribe(xs[i:i + 8], head="rnnt", att_context_size=[70, 1])
    model.train()
    return float(wer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps]))


def stage_train(a):
    from audioforge.data import Collate
    from audioforge.losses.consistency import FrozenTeacher, head_kl
    from audioforge.train import load_model, save_model
    dev = dev_of(a)
    torch.set_num_threads(a.threads)
    work = Path(a.work)
    tag = a.tag + ("_control" if a.control else "")
    od = work / tag
    od.mkdir(parents=True, exist_ok=True)
    items = load_manifest(work)
    srcs = set(a.sources.split(","))
    meet = [it for it in items if it["src"] in ("ami", "icsi") and it["src"] in srcs]
    anch = [it for it in items if it["src"] == "libri"]
    tix = None if a.control else TeacherIndex(work, a.teacher_ctx)
    model = load_model(a.student, dev)
    for n_, p in model.named_parameters():  # encoder + RNNT only; every other head stays frozen and bit-identical
        p.requires_grad = n_.startswith("encoder.") or n_.startswith("heads.rnnt.")
    frozen = FrozenTeacher(model, ["rnnt"]).to(dev).eval()
    proj = torch.nn.Linear(model.encoder.d_model, 1024).to(dev)
    opt = torch.optim.AdamW(param_groups(model, proj, a.lr, a.lr_decay), weight_decay=a.wd)
    col = Collate(model.tokenizer)
    audio = Audio(items, work)
    rng = random.Random(a.seed)
    ck = od / "ckpt.pt"
    step, hist, best_gate = 0, [], None
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        proj.load_state_dict(c["proj"])
        opt.load_state_dict(c["opt"])
        step, hist, best_gate = c["step"], c["hist"], c.get("gate_base")
        rng = random.Random(a.seed + step)
        log(f"resumed {tag} at step {step}")
    model.train()
    if best_gate is None:
        best_gate = gate_wer(model, work, a.gate_n, dev)
        hist.append({"step": 0, "gate_wer": best_gate})
        log(f"gate baseline WER {best_gate:.4f}")
        _save(od / "good.pt", model, proj, opt, 0, hist, best_gate)  # the initial student is the first restore point
    t0 = time.time()
    amp = torch.autocast("cuda", torch.bfloat16) if dev == "cuda" else _null()

    def sched(s):
        if s < a.warmup:
            return s / max(1, a.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))))
    stopped = any(h.get("gate_fired") for h in hist)  # a resumed run whose gate already fired does not train on
    if stopped:
        log("the WER gate fired earlier in this run: not training further")
    while step < a.steps and not stopped:
        step += 1
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * sched(step)
        opt.zero_grad(set_to_none=True)
        for _micro in range(a.accum):  # gradient accumulation: an effective batch of batch x accum items
            is_anchor = bool(anch) and rng.random() < a.p_anchor
            pool = anch if is_anchor else meet
            its = rng.sample(pool, min(a.batch, len(pool)))
            exs = [{"audio": audio(it), "text": it["text"] or " "} for it in its]
            b = col(exs)
            x, xl = b["audio"].to(dev), b["audio_len"].to(dev)
            parts = {}
            with amp:
                enc, elen, hid = model.encode(x, xl, [70, 1], return_hidden=True)
                h = model.heads["rnnt"]
                bt = {"text": b["text"].to(dev), "text_len": b["text_len"].to(dev)}
                parts["rnnt_gt"] = h.loss(enc.float(), elen, bt)
                if is_anchor and a.w_kl > 0:
                    with torch.no_grad():
                        te, _, thid = frozen.encode(*frozen_inputs(frozen, x, xl))
                    parts["kl_anchor"] = head_kl(h, enc.float(), frozen.heads["rnnt"], te.float(), elen, bt["text"],
                                                 bt["text_len"])
                if not is_anchor and tix is not None:
                    tt, te = zip(*[tix.get(it["i"]) for it in its])
                    if a.w_seq > 0:
                        tb = col([{"audio": e["audio"], "text": t or " "} for e, t in zip(exs, tt)])
                        parts["rnnt_seq"] = h.loss(enc.float(), elen, {"text": tb["text"].to(dev),
                                                                       "text_len": tb["text_len"].to(dev)})
                    if a.w_enc > 0 and any(len(e) for e in te):
                        Tm = enc.shape[1]
                        tgt = torch.zeros(len(te), Tm, 1024, device=dev)
                        msk = torch.zeros(len(te), Tm, device=dev)
                        for j, e in enumerate(te):
                            n = min(len(e), int(elen[j]), Tm)
                            tgt[j, :n] = torch.from_numpy(e[:n].astype(np.float32)).to(dev)
                            msk[j, :n] = 1
                        cos = F.cosine_similarity(proj(enc.float()), tgt, dim=-1)
                        parts["enc_cos"] = ((1 - cos) * msk).sum() / msk.sum().clamp(min=1)
            w = {"rnnt_gt": a.w_gt if not is_anchor else 1.0, "rnnt_seq": a.w_seq, "enc_cos": a.w_enc, "kl_anchor": a.w_kl}
            loss = sum(w[k] * v for k, v in parts.items())
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        if not math.isfinite(float(loss)):
            log(f"non-finite loss at step {step}: stop")
            break
        if step % a.log_every == 0:
            rec = {"step": step, "pool": "anchor" if is_anchor else "meeting",
                   "loss": round(float(loss), 4),
                   **{k: round(float(v), 4) for k, v in parts.items()}, "sec": round(time.time() - t0, 1)}
            hist.append(rec)
            log(json.dumps(rec))
        gate_now = step % a.gate_every == 0
        if gate_now:
            gw = gate_wer(model, work, a.gate_n, dev)
            hist.append({"step": step, "gate_wer": gw, "gate_base": best_gate})
            log(f"gate step {step}: WER {gw:.4f} (baseline {best_gate:.4f}, max +{a.gate_max_delta})")
            if gw <= best_gate + a.gate_max_delta:  # this state passed the gate: it becomes the restore point
                _save(od / "good.pt", model, proj, opt, step, hist, best_gate)
            else:
                hist.append({"step": step, "gate_fired": True})
                log("GATE FIRED: restoring the last gate-passing state and stopping")
                good = od / "good.pt"
                if good.exists():
                    c = torch.load(good, map_location="cpu", weights_only=False)
                    model.load_state_dict(c["model"])
                    step = c["step"]
                else:
                    model = load_model(a.student, dev)
                    step = 0
                _save(ck, model, proj, opt, step, hist, best_gate)
                break
        if step % a.ckpt_every == 0 or step == a.steps:
            _save(ck, model, proj, opt, step, hist, best_gate)
        if time.time() - t0 > a.budget:
            _save(ck, model, proj, opt, step, hist, best_gate)
            log(f"budget: checkpoint at step {step} (resumable)")
            (od / "history.json").write_text(json.dumps(hist, indent=1))
            return
    save_model(model.cpu(), od / "student.afm")
    src = load_model(a.student, "cpu").state_dict()
    out = load_model(str(od / "student.afm"), "cpu").state_dict()
    bad = [k for k in src if not k.startswith(("encoder.", "heads.rnnt.")) and not torch.equal(src[k], out[k])]
    assert not bad, f"tensors outside the encoder / RNNT changed: {bad[:5]}"
    (od / "history.json").write_text(json.dumps(hist, indent=1))
    log(f"train {tag}: {od / 'student.afm'} (step {step}); other heads bit-identical")


def frozen_inputs(frozen, x, xl):
    feats, flen = frozen.preprocessor(x, xl)
    return feats, flen, [70, 1]


def _save(ck, model, proj, opt, step, hist, gate_base):
    tmp = ck.with_suffix(".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "proj": proj.state_dict(),
                "opt": opt.state_dict(), "step": step, "hist": hist, "gate_base": gate_base}, tmp)
    tmp.replace(ck)


# --------------------------------------------------------------------------- eval / heads
def stage_eval(a):
    import hybrid_asr as H
    from audioforge.train import load_model
    dev = dev_of(a)
    torch.set_num_threads(a.threads)
    work = Path(a.work)
    norms = {"normalize_text": __import__("audioforge.teachers", fromlist=["normalize_text"]).normalize_text}
    models = {"served": a.student}
    for tg in (a.tag, a.tag + "_control"):
        p = work / tg / "student.afm"
        if p.exists():
            models[tg] = str(p)
    hyps = {}
    for name, path in models.items():
        m = load_model(path, dev).eval()
        for s in ("ami", "icsi", "libri"):
            xs, refs = load_eval(work, s, a.n_eval)
            for c in (1, 13):
                hp = work / "eval" / f"hyp_{name}_{s}_la{c}.json"
                if hp.exists():
                    hyps[(name, s, c)] = json.loads(hp.read_text())
                    continue
                out = []
                with torch.no_grad():
                    for i in range(0, len(xs), 8):
                        out += m.transcribe(xs[i:i + 8], head="rnnt", att_context_size=[70, c])
                hp.write_text(json.dumps(out))
                hyps[(name, s, c)] = out
        del m
    res = {"protocol": "HYBRID_ASR.md sets, masked offline forward = cache-aware streaming, greedy RNNT, "
                       "normalize_text, 1000-resample CIs, paired utterance bootstrap", "models": models, "sets": {}}
    fn = norms["normalize_text"]
    for s in ("ami", "icsi", "libri"):
        _, refs = load_eval(work, s, a.n_eval)
        for c in (1, 13):
            E = {n: np.array([H.edits_words(fn(r), fn(hyps[(n, s, c)][i])) for i, r in enumerate(refs)], float)
                 for n in models}
            r = {n: H.rate_ci(e) for n, e in E.items()}
            for n in models:
                if n != "served":
                    r[f"{n} - served"] = H.paired(E[n], E["served"])
            res["sets"][f"{s}_la{c}"] = r
            log(s, c, {n: r[n]["wer"] for n in models})
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1))
    log(f"-> {out}")


def stage_heads(a):
    """The served VAD head on the distilled encoder vs the served model (BASELINES VAD protocol)."""
    import vad_layers as V
    from audioforge.data import Collate
    from audioforge.train import load_model
    dev = dev_of(a)
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    res = {}
    for name, path in (("served", a.student), (a.tag, str(Path(a.work) / a.tag / "student.afm"))):
        m = load_model(path, dev).eval()
        for s in ("ami_dev", "icsi_dev"):
            val = V.load_set(s, 64)
            P, Y = [], []
            col = Collate(None)
            with torch.no_grad():
                for i in range(0, len(val), 8):
                    b = col([{"audio": v["audio"]} for v in val[i:i + 8]])
                    enc, elen, hid = m.encode(b["audio"].to(dev), b["audio_len"].to(dev), [70, 1], return_hidden=True)
                    p = m.heads["vad"].decode(m.head_input("vad", enc, hid), elen).float().cpu().numpy()
                    for j, v in enumerate(val[i:i + 8]):
                        T = min(int(elen[j]), len(v["vad"]))
                        P.append(p[j, :T])
                        Y.append(np.asarray(v["vad"][:T]))
            res.setdefault(name, {})[s] = V.vad_report(P, Y)
            log(name, s, res[name][s]["f1"])
    out["heads_vad"] = res
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["manifest", "evalsets", "teacher", "train", "eval", "heads"])
    p.add_argument("--sources", default="ami,icsi", help="train: meeting pools used (e.g. ami only)")
    p.add_argument("--work", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--budget", type=float, default=1e9, help="seconds per call (resumable stages)")
    p.add_argument("--teacher", default=str(ROOT / "runs/nemo_nemotron_speech_streaming_en_0.6b.afm"))
    p.add_argument("--student", default=str(ROOT / "runs/stage1_served.afm"))
    p.add_argument("--contexts", default="1", help="teacher: att right contexts to cache (1 or 1,13)")
    p.add_argument("--n-ami", type=int, default=0, help="manifest: AMI train meetings (0 = all prepared)")
    p.add_argument("--n-icsi", type=int, default=0)
    p.add_argument("--n-libri", type=int, default=28000, help="manifest: LibriSpeech anchor utterances (~100 h)")
    p.add_argument("--max-hours", type=float, default=0.0, help="manifest: cap (smoke tests)")
    p.add_argument("--batch", type=int, default=32, help="items per micro-batch")
    p.add_argument("--accum", type=int, default=1, help="train: micro-batches per optimiser step")
    p.add_argument("--part", default="0/1", help="teacher: k/N, cache every N-th shard starting at k (one per GPU)")
    p.add_argument("--tag", default="kd")
    p.add_argument("--control", action="store_true", help="train: no teacher terms (continued training)")
    p.add_argument("--teacher-ctx", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-decay", type=float, default=0.9, help="layer-wise LR decay per block from the top")
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--w-gt", type=float, default=0.5)
    p.add_argument("--w-seq", type=float, default=0.5)
    p.add_argument("--w-enc", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=0.5)
    p.add_argument("--p-anchor", type=float, default=0.25)
    p.add_argument("--gate-every", type=int, default=500)
    p.add_argument("--gate-n", type=int, default=100)
    p.add_argument("--gate-max-delta", type=float, default=0.003)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "runs" / "distill_0p6b.json"))
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
