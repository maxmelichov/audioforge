"""The true single model: train the target-speaker (TS-VAD) head and the turn head of the served 115M model with
more speakers and data, optionally unfreezing the top encoder blocks under a WER gate, and optionally distilling
nemotron-speech-streaming-en-0.6b into the ASR (research/SINGLE_MODEL.md part B; README.md in this directory has the
exact commands, the pre-registered bars and the GPU-hour estimate per stage).

Product decision (2026-09-29): the product needs *target* diarization (is the known user talking, is the user's turn
over), not general diarization, so there is no diarization teacher. The served speaker head (block 4) stays frozen,
and so do blocks 1-4, so every stored voice print stays valid.

Stages (each resumable; one process per GPU, CUDA_VISIBLE_DEVICES pins it):

  manifest  AMI (all train meetings) + ICSI (all train meetings) 16 s crops (hop 12 s) with per-speaker activity,
            per (meeting, speaker) single-speaker enrollment pools, AMI / ICSI turn windows (the served turn head's
            training windows: primary speaker, end-of-turn labels), LibriSpeech train-clean-100 utterances by speaker
            (the simulated-overlap mixtures), and distill_0p6b_to_115m's ASR manifest (for the anchor / KD batches)
            -> <work>/{crops,turns}.pkl, <work>/libri_spk.json, <work>/manifest.jsonl
  evalsets  AMI-200 / ICSI-200 / LibriSpeech-200 (distill_0p6b_to_115m.evalsets: the laptop's seeded sets)
  teacher   (optional, --asr-kd) the 0.6B's [70,1] transcripts + top-layer outputs over the ASR manifest
            (distill_0p6b_to_115m.teacher; --part k/N splits it across GPUs)
  train     student = runs/stage1_served.afm + the TS-VAD head (init runs/tsvad_spk.pt). Trainable: the TS-VAD head,
            the turn head and its speaker kernels, and with --unfreeze N the top N encoder blocks, the VAD head and
            the RNNT head. Tasks per step, sampled with --p-tsvad / --p-turn / --p-asr:
              tsvad  a 16 s meeting crop (p --p-meeting) or a simulated LibriSpeech mixture of 2-3 speakers with
                     overlap; the target = a speaker in the crop (p 0.8) or one who is absent; the voice print = that
                     speaker's single-speaker speech elsewhere, U(--print-min, --print-max) s, embedded by the frozen
                     served speaker head; dropped with p 0.15 (then target = any speech). BCE on [P(target), P(other)]
              turn   an AMI / ICSI turn window; the primary's print -> the TS-VAD track (current head, detached)
                     -> the served turn head's external-track input (spk_act_ext / spk_targets_ext /
                     spk_prim_ext, exactly the served path's [P(user), P(other), 0, 0], primary 0); its own loss
              asr    (only with --unfreeze > 0 or --asr-kd) meeting segments: RNNT(reference) [+ with --asr-kd
                     0.5 RNNT(0.6B transcript) + (1 - cos(P h, h_0.6B))]; LibriSpeech anchor (25 %): RNNT(reference)
                     + 0.5 KL(student joint || frozen initial student). With --unfreeze the tsvad batches also train
                     the VAD head (label: any speaker active, weight --w-vad), which reads all 17 blocks
            AdamW, warm-up + cosine, layer-wise LR decay over the unfrozen blocks, bf16 autocast on CUDA, grad clip 1.
            WER gate every --gate-every steps: LibriSpeech (first --gate-n of the 200) and AMI (first --gate-n) at
            [70,1]; LibriSpeech > baseline + --gate-libri or AMI > baseline + --gate-ami -> the last passing state is
            restored and the LR halved; the second firing stops the run ("killed" in history.json). Checkpoints every
            --ckpt-every steps (<work>/<tag>/ckpt.pt, trainable tensors + optimiser; resumable). Output:
            <work>/<tag>/student.afm (served format) and <work>/<tag>/tsvad.pt (serve --tsvad format); tensors outside
            the trainable set are asserted bit-identical to the served model.
  eval      target-tracking F1 on the eot-bench v2 windows (AMI dev 974, ICSI held-out 1312; primary = target; 5 s
            print from elsewhere in the meeting, tsvad.py's bank rule and seeds) for the served head and the student;
            LibriSpeech-200 / AMI-200 / ICSI-200 WER (served vs student); VAD F1 (64 AMI dev windows)
            -> runs/single_distill.json. The turn-miss bar is scored on the laptop (README step 7: tsvad.py bind /
            scores --model / report, which need the Silero and Sortformer caches there).

  python scripts/research/single_model_distill/single_distill.py <stage> --work <dir> [options]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
HERE = Path(__file__).resolve().parent
SR, FS = 16000, 1280  # 80 ms frames
TAP = 3  # block 4: the served speaker head's and the TS-VAD head's input
CROP_S, HOP_S = 16.0, 12.0


def _load_distill():
    spec = importlib.util.spec_from_file_location("distill_0p6b", ROOT / "scripts/research/distill_0p6b_to_115m/distill.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _load_distill()  # Audio, evalsets, teacher cache, TeacherIndex, gate_wer, load_eval
log = D.log
dev_of = D.dev_of


# --------------------------------------------------------------------------- data sources
def _ds(src, meetings):
    if src == "ami":
        from audioforge.datasets.ami import AMI
        return AMI(sorted(meetings), verbose=False)
    from audioforge.datasets.icsi import ICSI
    return ICSI(sorted(meetings), verbose=False)


def _train_meetings(a):
    from audioforge.datasets.ami import split_lists, subset
    from audioforge.datasets.icsi import SPLITS
    from audioforge.datasets.icsi import subset as isub
    return {"ami": subset({"train": a.n_ami or len(split_lists()["train"])})["train"],
            "icsi": isub({"train": a.n_icsi or len(SPLITS["train"])})["train"]}


def stage_manifest(a):
    import tsvad as T

    from audioforge.datasets.librispeech import LibriSpeech
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    crops, turns, pools = [], [], {}
    for src, ms in _train_meetings(a).items():
        ds = _ds(src, ms)
        for m in ds.meetings:
            segs = T.single_segments(ds, m)
            pools[f"{src}:{m}"] = {s: v for s, v in segs.items()}
            dur = ds.duration(m)
            for s0 in np.arange(0.0, max(dur - CROP_S, 0.0) + 1e-6, HOP_S):
                crops.append({"src": src, "meeting": m, "a": round(float(s0), 3), "b": round(float(min(dur, s0 + CROP_S)), 3)})
            for ex in ds.turn_examples(meetings=[m]):
                spk = ds.speaker_ids[int(ex["speaker"]) - getattr(ds, "speaker_offset", 0)]
                a0 = float(ex["start"])
                turns.append({"src": src, "meeting": m, "a": round(a0, 3), "b": round(a0 + len(ex["audio"]) / SR, 3),
                              "speaker": spk, "text": ex["text"],
                              **{k: np.asarray(ex[k]).astype(np.float16 if np.asarray(ex[k]).dtype.kind == "f" else np.int32)
                                 for k in ("spk_targets", "spk_act", "eot", "hes")},
                              "turn_end_frame": int(ex["turn_end_frame"]), "onset_frame": int(ex["onset_frame"])})
        log(f"{src}: {len(ds.meetings)} meetings, {sum(c['src'] == src for c in crops)} crops, "
            f"{sum(t['src'] == src for t in turns)} turn windows")
    ls = LibriSpeech("train-clean-100", max_sec=16.0, verbose=False)
    by_spk: dict[int, list[int]] = {}
    for i, u in enumerate(ls.utts):
        by_spk.setdefault(int(u["speaker_id"]), []).append(i)
    if a.max_hours:  # smoke tests: seeded subsets
        rng.shuffle(crops)
        rng.shuffle(turns)
        k = max(8, int(a.max_hours * 3600 / CROP_S / 2))
        crops, turns = crops[:k], turns[:k]
        keep = sorted(by_spk)[: max(4, a.n_libri_spk_smoke)]
        by_spk = {s: by_spk[s][:20] for s in keep}
    (work / "crops.pkl").write_bytes(pickle.dumps({"crops": crops, "pools": pools}))
    (work / "turns.pkl").write_bytes(pickle.dumps(turns))
    (work / "libri_spk.json").write_text(json.dumps({str(k): v for k, v in by_spk.items()}))
    for p in ("crops.pkl", "turns.pkl"):  # read-back check (exFAT / network disks)
        assert len(pickle.loads((work / p).read_bytes())) > 0, p
    # the ASR manifest of distill_0p6b_to_115m (asr / anchor batches and the optional 0.6B teacher cache)
    if not (work / "manifest.jsonl").exists():
        D.stage_manifest(argparse.Namespace(work=a.work, n_ami=a.n_ami, n_icsi=a.n_icsi, n_libri=a.n_libri,
                                            max_hours=a.max_hours))
    log(f"manifest: {len(crops)} crops, {len(turns)} turn windows, {len(by_spk)} LibriSpeech speakers")


def stage_evalsets(a):
    D.stage_evalsets(a)


def stage_teacher(a):
    D.stage_teacher(a)


# --------------------------------------------------------------------------- batch builders
class Sources:
    """Audio and labels for the three tasks; datasets opened lazily (memory-mapped caches)."""

    def __init__(self, work, seed):
        work = Path(work)
        cp = pickle.loads((work / "crops.pkl").read_bytes())
        self.crops, self.pools = cp["crops"], cp["pools"]
        self.turns = pickle.loads((work / "turns.pkl").read_bytes())
        self.libri = {int(k): v for k, v in json.loads((work / "libri_spk.json").read_text()).items()}
        self.rng = random.Random(seed)
        self.ds, self.ls = {}, None
        self.meetings = {}
        for c in self.crops + self.turns:
            self.meetings.setdefault(c["src"], set()).add(c["meeting"])

    def dsx(self, src):
        if src not in self.ds:
            self.ds[src] = _ds(src, self.meetings[src])
        return self.ds[src]

    def libri_audio(self, i):
        if self.ls is None:
            from audioforge.datasets.librispeech import LibriSpeech
            self.ls = LibriSpeech("train-clean-100", max_sec=16.0, verbose=False)
        return np.asarray(self.ls.audio(i), np.float32)

    def print_audio(self, src, m, spk, excl, L):
        """U(L) s of ``spk``'s single-speaker speech in meeting m outside ``excl`` (+-1 s), else None."""
        import tsvad as T
        segs = self.pools.get(f"{src}:{m}", {}).get(spk, [])
        ivs = T.clip_from(segs, L, self.rng, exclude=(excl[0] - 1.0, excl[1] + 1.0))
        return None if ivs is None else T.clip_audio(self.dsx(src), m, ivs)

    # ---- tsvad examples: (audio, targets (T, 2), print audio or None)
    def meeting_crop(self, L):
        from audioforge.data import ToneLanguage
        from audioforge.datasets.ami import spk_matrix
        c = self.rng.choice(self.crops)
        ds = self.dsx(c["src"])
        x = np.asarray(ds._clip(c["meeting"], c["a"], c["b"]), np.float32)
        T_ = ToneLanguage.n_frames(len(x))
        y, order, _ = spk_matrix(ds.acts[c["meeting"]], T_, c["a"], c["b"], 8)
        present = [s for j, s in enumerate(order) if y[:, j].any()]
        enrollable = list(self.pools.get(f"{c['src']}:{c['meeting']}", {}))
        cand = present if (present and self.rng.random() < 0.8) else enrollable
        if not cand:
            return x, np.stack([np.zeros(T_), y.max(1)], 1).astype(np.float32), None
        spk = self.rng.choice(cand)
        tgt = y[:, order.index(spk)] if spk in order else np.zeros(T_, np.float32)
        oth = np.max([y[:, j] for j, s in enumerate(order) if s != spk] or [np.zeros(T_)], 0)
        return x, np.stack([tgt, oth], 1).astype(np.float32), self.print_audio(c["src"], c["meeting"], spk,
                                                                              (c["a"], c["b"]), L)

    def libri_mixture(self, L, dur_s=CROP_S):
        """2-3 LibriSpeech speakers placed at random offsets in dur_s s (overlap happens), random gains; frame activity
        by an energy VAD per source; the target's print = another utterance of the same speaker."""
        n = self.rng.choice((2, 2, 3))
        spks = self.rng.sample(sorted(self.libri), min(n, len(self.libri)))
        N = int(dur_s * SR)
        mix = np.zeros(N, np.float32)
        acts = []
        for s in spks:
            u = self.rng.choice(self.libri[s])
            x = self.libri_audio(u)[: N]
            off = self.rng.randint(0, max(0, N - len(x)))
            g = 10 ** (self.rng.uniform(-6, 6) / 20)
            mix[off: off + len(x)] += g * x
            fr = np.zeros(N // FS, np.float32)
            e = np.sqrt(np.convolve(x ** 2, np.ones(FS) / FS, mode="same")[::FS] + 1e-10)
            on = (e > 0.1 * e.max()).astype(np.float32)
            k0 = off // FS
            fr[k0: k0 + len(on)] = on[: len(fr) - k0]
            acts.append((s, u, fr))
        mix /= max(1.0, float(np.abs(mix).max()) / 0.9)
        tgt_i = self.rng.randrange(len(acts))
        s, u, tgt = acts[tgt_i]
        oth = np.max([f for j, (_, _, f) in enumerate(acts) if j != tgt_i], 0)
        others = [v for v in self.libri[s] if v != u]
        pr = None
        if others:
            p = self.libri_audio(self.rng.choice(others))
            pr = p[: int(L * SR)] if len(p) >= int(L * SR) else p
        from audioforge.data import ToneLanguage
        T_ = ToneLanguage.n_frames(N)
        y = np.zeros((T_, 2), np.float32)
        m = min(T_, len(tgt))
        y[:m, 0], y[:m, 1] = tgt[:m], oth[:m]
        return mix, y, pr

    def turn_window(self):
        t = self.rng.choice(self.turns)
        ds = self.dsx(t["src"])
        x = np.asarray(ds._clip(t["meeting"], t["a"], t["b"]), np.float32)
        ex = {"audio": x, "text": t["text"] or " "}
        for k in ("spk_targets", "spk_act", "eot", "hes"):
            ex[k] = np.asarray(t[k], np.float32)
        return ex, t


# --------------------------------------------------------------------------- model pieces
@torch.no_grad()
def block4(model, x, xl, att=(70, 1)):
    """Encoder block-4 output (B, T, D) and lengths: blocks 1-4 only (frozen), the masked causal forward."""
    from audioforge.modules.fastconformer import chunked_attention_mask
    enc = model.encoder
    feats, flen = model.preprocessor(x, xl)
    h, hl = enc.pre_encode(feats, flen)
    if enc.xscale is not None:
        h = h * enc.xscale
    Tn = h.shape[1]
    am = chunked_attention_mask(Tn, hl, list(att))
    pm = torch.arange(Tn, device=h.device)[None] >= hl[:, None]
    for li in range(TAP + 1):
        h, _, _ = enc.layers[li](h, am, pm)
    return h, hl


def train_mode(model, names):
    """model.train(), then every module without a trainable parameter back to eval: the frozen blocks 1-4, the speaker
    head (BatchNorm: prints must be the served ones) and the other frozen heads run exactly as served (no dropout)."""
    model.train()
    for mod in list(model.encoder.layers) + [model.encoder.pre_encode] + list(model.heads.values()):
        if not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


@torch.no_grad()
def voiceprints(model, prints, dev):
    """Print audios (None allowed) -> (E (B, 192), has (B,)) with the frozen served speaker head on block 4."""
    assert not model.heads["spk"].training and not model.encoder.layers[TAP].training, "frozen parts must be in eval"
    has = torch.tensor([p is not None and len(p) >= SR // 2 for p in prints], device=dev)
    E = torch.zeros(len(prints), 192, device=dev)
    idx = [i for i, ok in enumerate(has.tolist()) if ok]
    if idx:
        x, xl = model._pad([prints[i] for i in idx])
        h, hl = block4(model, x, xl)
        e = model.heads["spk"].embed(h.float(), hl)
        E[idx] = e.float()
    return E, has


@contextmanager
def only_heads(model, names):
    """Temporarily give every head outside ``names`` weight 0 (model.forward then skips its loss)."""
    old = {k: v.get("weight", 1.0) for k, v in model.head_cfg.items()}
    for k, v in model.head_cfg.items():
        v["weight"] = old[k] if k in names else 0.0
    try:
        yield
    finally:
        for k, v in model.head_cfg.items():
            v["weight"] = old[k]


def trainable_names(model, unfreeze: int, asr: bool) -> list[str]:
    L = len(model.encoder.layers)
    keep = ["heads.turn.", "encoder.speaker_kernels.", "heads.tsvad."]
    if unfreeze > 0:
        keep += [f"encoder.layers.{i}." for i in range(L - unfreeze, L)] + ["heads.vad.", "layer_mix.vad"]
    if asr or unfreeze > 0:
        keep += ["heads.rnnt."]
    return keep


def build_student(a, dev):
    from audioforge.model import build_head
    from audioforge.train import load_model
    model = load_model(a.student, dev)
    assert model.layer_tap.get("spk") == [TAP], "the served speaker head must read block 4"
    ck = torch.load(a.tsvad_init, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in ck["cfg"].items() if k not in ("enroll_embedder",)}
    cfg.update(type="tsvad", from_layers=[TAP], weight=0.0)
    head = build_head(cfg, model.encoder.d_model)
    head.load_state_dict(ck["state_dict"])
    model.heads["tsvad"] = head.to(dev)
    model.head_cfg["tsvad"] = cfg
    model.layer_tap["tsvad"] = [TAP]
    model.cfg["heads"]["tsvad"] = cfg
    # the turn head reads the external (TS-VAD) track on every item, as served; noise augmentation as trained
    model.cond = {**model.cond, "p_ext": 1.0, "p_diar": 0}
    model.cond_on = True
    return model, ck


def param_groups(model, names, lr, decay):
    L = len(model.encoder.layers)
    groups = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lrx = lr
        if n.startswith("encoder.layers."):
            i = int(n.split(".")[2])
            lrx = lr * decay ** (L - 1 - i)
        groups.append({"params": [p], "lr": lrx, "base_lr": lrx, "name": n})
    return groups


def gate_wer(model, work, n, dev):
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    out = {}
    model.eval()
    for s in ("libri", "ami"):
        xs, refs = D.load_eval(work, s, n)
        hyps = []
        with torch.no_grad():
            for i in range(0, len(xs), 8):
                hyps += model.transcribe(xs[i:i + 8], head="rnnt", att_context_size=[70, 1])
        out[s] = float(wer([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps]))
    return out


def _save_ck(path, model, names, opt, step, hist, base, lr_scale, fires, proj=None):
    tmp = path.with_suffix(".tmp")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(tuple(names))}
    torch.save({"trainable": sd, "opt": opt.state_dict(), "step": step, "hist": hist, "gate_base": base,
                "lr_scale": lr_scale, "fires": fires, "proj": None if proj is None else proj.state_dict()}, tmp)
    tmp.replace(path)
    chk = torch.load(path, map_location="cpu", weights_only=False)  # read-back check
    assert chk["step"] == step and len(chk["trainable"]) == len(sd), path


def stage_train(a):
    from audioforge.data import Collate
    from audioforge.losses.consistency import FrozenTeacher, head_kl
    dev = dev_of(a)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    work = Path(a.work)
    od = work / a.tag
    od.mkdir(parents=True, exist_ok=True)
    model, tck = build_student(a, dev)
    if a.init:  # continue from an earlier stage's student (its trainable tensors)
        c = torch.load(Path(a.init), map_location="cpu", weights_only=False)
        missing = model.load_state_dict(c["trainable"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys[:5]
        log(f"init from {a.init}: {len(c['trainable'])} tensors")
    use_asr = a.unfreeze > 0 or a.asr_kd
    names = trainable_names(model, a.unfreeze, a.asr_kd)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(tuple(names))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {n_tr / 1e6:.2f} M parameters ({', '.join(names)})")
    frozen = FrozenTeacher(model, ["rnnt"]).to(dev).eval() if use_asr else None
    proj = torch.nn.Linear(model.encoder.d_model, 1024).to(dev) if a.asr_kd else None
    groups = param_groups(model, names, a.lr, a.lr_decay)
    if proj is not None:
        groups.append({"params": list(proj.parameters()), "lr": a.lr, "base_lr": a.lr, "name": "projector"})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    src = Sources(work, a.seed)
    items = D.load_manifest(work) if use_asr else []
    meet = [it for it in items if it["src"] in ("ami", "icsi")]
    anch = [it for it in items if it["src"] == "libri"]
    audio = D.Audio(items, work) if use_asr else None
    tix = D.TeacherIndex(work, 1) if a.asr_kd else None
    col = Collate(model.tokenizer)
    rng = random.Random(a.seed)
    ck = od / "ckpt.pt"
    step, hist, base, lr_scale, fires = 0, [], None, 1.0, 0
    if ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["trainable"], strict=False)
        opt.load_state_dict(c["opt"])
        if proj is not None and c.get("proj") is not None:
            proj.load_state_dict(c["proj"])
        step, hist, base, lr_scale, fires = c["step"], c["hist"], c["gate_base"], c["lr_scale"], c["fires"]
        rng = random.Random(a.seed + step)
        src.rng = random.Random(a.seed + 7 * step)
        log(f"resumed {a.tag} at step {step} (gate fired {fires}x)")
    if base is None:
        base = gate_wer(model, work, a.gate_n, dev)
        hist.append({"step": 0, "gate": base})
        log(f"gate baseline {base}")
        _save_ck(od / "good.pt", model, names, opt, 0, hist, base, lr_scale, fires, proj)
    killed = any(h.get("killed") for h in hist)
    amp = torch.autocast("cuda", torch.bfloat16) if dev == "cuda" else D._null()
    train_mode(model, names)
    t0 = time.time()

    def sched(s):
        w = s / max(1, a.warmup) if s < a.warmup else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))))
        return w * lr_scale

    tasks = [("tsvad", a.p_tsvad), ("turn", a.p_turn)] + ([("asr", a.p_asr)] if use_asr else [])
    while step < a.steps and not killed:
        step += 1
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * sched(step)
        opt.zero_grad(set_to_none=True)
        task = rng.choices([t for t, _ in tasks], [w for _, w in tasks])[0]
        parts = {}
        for _micro in range(a.accum):
            with amp:
                if task == "tsvad":
                    exs = [src.meeting_crop(rng.uniform(a.print_min, a.print_max)) if rng.random() < a.p_meeting
                           else src.libri_mixture(rng.uniform(a.print_min, a.print_max)) for _ in range(a.batch)]
                    x, xl = model._pad([e[0] for e in exs])
                    E, has = voiceprints(model, [e[2] for e in exs], dev)
                    has = has & (torch.rand(len(exs), device=dev) >= a.p_noprint)
                    Y = torch.zeros(len(exs), max(len(e[1]) for e in exs), 2, device=dev)
                    for j, e in enumerate(exs):
                        Y[j, : len(e[1])] = torch.from_numpy(e[1]).to(dev)
                        if not bool(has[j]):  # no print: the head is a plain VAD (target = any speech)
                            Y[j, :, 0] = torch.clamp(Y[j, :, 0] + Y[j, :, 1], max=1.0)
                            Y[j, :, 1] = 0.0
                    vad_y = torch.zeros(Y.shape[:2], device=dev)
                    for j, e in enumerate(exs):
                        vad_y[j, : len(e[1])] = torch.from_numpy(np.clip(e[1].sum(1), 0, 1)).to(dev)
                    h4, elen = block4(model, x, xl)  # blocks 1-4 are frozen: the TS-VAD input never moves
                    loss = model.heads["tsvad"].loss(h4.float(), elen, {"tsvad_targets": Y, "tsvad_enroll": E,
                                                                        "tsvad_has": has})
                    parts["tsvad"] = loss
                    if a.unfreeze > 0 and a.w_vad > 0:  # the VAD head reads all blocks: keep it trained on labels
                        enc, elen2, hid = model.encode(x, xl, [70, 1], return_hidden=True)
                        parts["vad"] = model.heads["vad"].loss(model.head_input("vad", enc, hid), elen2,
                                                               {"vad": vad_y[:, : enc.shape[1]]})
                        loss = loss + a.w_vad * parts["vad"]
                elif task == "turn":
                    pairs = [src.turn_window() for _ in range(a.batch)]
                    exs = [p[0] for p in pairs]
                    x, xl = model._pad([e["audio"] for e in exs])
                    prints = [src.print_audio(t["src"], t["meeting"], t["speaker"], (t["a"], t["b"]),
                                              rng.uniform(a.print_min, a.print_max)) for _, t in pairs]
                    E, has = voiceprints(model, prints, dev)
                    with torch.no_grad():  # the served path: the current TS-VAD head's track, as the turn head sees it
                        th = model.heads["tsvad"]
                        was = th.training
                        th.eval()  # the track as served (no dropout)
                        h4, elen = block4(model, x, xl)
                        P = torch.sigmoid(th(h4.float(), elen, E, has)).float()
                        th.train(was)
                    for j, e in enumerate(exs):
                        T_ = len(e["spk_act"])
                        pj = P[j, :T_].cpu().numpy()
                        if len(pj) < T_:
                            pj = np.concatenate([pj, np.zeros((T_ - len(pj), 2), np.float32)])
                        cols = np.zeros((T_, 4), np.float32)
                        cols[:, :2] = pj
                        e["spk_act_ext"] = pj[:, 0].copy()
                        e["spk_targets_ext"] = cols
                        e["spk_prim_ext"] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in col(exs).items()}
                    with only_heads(model, ("turn",)):
                        out = model(b, att_context_size=[70, 1])
                    loss = out["loss_turn"]
                    parts["turn"] = loss
                else:  # asr
                    is_anchor = bool(anch) and rng.random() < a.p_anchor
                    pool = anch if is_anchor else meet
                    its = rng.sample(pool, min(a.batch, len(pool)))
                    exs = [{"audio": audio(it), "text": it["text"] or " "} for it in its]
                    b = col(exs)
                    x, xl = b["audio"].to(dev), b["audio_len"].to(dev)
                    enc, elen, hid = model.encode(x, xl, [70, 1], return_hidden=True)
                    h = model.heads["rnnt"]
                    bt = {"text": b["text"].to(dev), "text_len": b["text_len"].to(dev)}
                    parts["rnnt_gt"] = h.loss(enc.float(), elen, bt)
                    loss = parts["rnnt_gt"]
                    if is_anchor and a.w_kl > 0:
                        with torch.no_grad():
                            feats, flen = frozen.preprocessor(x, xl)
                            te, _, _ = frozen.encode(feats, flen, [70, 1])
                        parts["kl_anchor"] = head_kl(h, enc.float(), frozen.heads["rnnt"], te.float(), elen,
                                                     bt["text"], bt["text_len"])
                        loss = loss + a.w_kl * parts["kl_anchor"]
                    if not is_anchor and tix is not None:
                        tt, tenc = zip(*[tix.get(it["i"]) for it in its])
                        tb = col([{"audio": e["audio"], "text": t or " "} for e, t in zip(exs, tt)])
                        parts["rnnt_seq"] = h.loss(enc.float(), elen, {"text": tb["text"].to(dev),
                                                                       "text_len": tb["text_len"].to(dev)})
                        Tm = enc.shape[1]
                        tgt = torch.zeros(len(tenc), Tm, 1024, device=dev)
                        msk = torch.zeros(len(tenc), Tm, device=dev)
                        for j, e in enumerate(tenc):
                            n_ = min(len(e), int(elen[j]), Tm)
                            tgt[j, :n_] = torch.from_numpy(e[:n_].astype(np.float32)).to(dev)
                            msk[j, :n_] = 1
                        cos = F.cosine_similarity(proj(enc.float()), tgt, dim=-1)
                        parts["enc_cos"] = ((1 - cos) * msk).sum() / msk.sum().clamp(min=1)
                        loss = a.w_gt * parts["rnnt_gt"] + a.w_seq * parts["rnnt_seq"] + a.w_enc * parts["enc_cos"]
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        if not math.isfinite(float(loss.detach())):
            log(f"non-finite loss at step {step}: stop")
            break
        if step % a.log_every == 0:
            rec = {"step": step, "task": task, "loss": round(float(loss.detach()), 4),
                   **{k: round(float(v.detach()), 4) for k, v in parts.items()}, "lr_scale": lr_scale,
                   "sec": round(time.time() - t0, 1)}
            hist.append(rec)
            log(json.dumps(rec))
        if step % a.gate_every == 0 or step == a.steps:
            g = gate_wer(model, work, a.gate_n, dev)
            train_mode(model, names)
            ok = g["libri"] <= base["libri"] + a.gate_libri and g["ami"] <= base["ami"] + a.gate_ami
            hist.append({"step": step, "gate": g, "gate_base": base, "pass": ok})
            log(f"gate step {step}: {g} (baseline {base}, max +{a.gate_libri} / +{a.gate_ami}) {'pass' if ok else 'FAIL'}")
            if ok:
                _save_ck(od / "good.pt", model, names, opt, step, hist, base, lr_scale, fires, proj)
            else:
                fires += 1
                c = torch.load(od / "good.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(c["trainable"], strict=False)
                step = c["step"]
                if fires >= a.max_fires:
                    hist.append({"step": step, "gate_fired": True, "killed": True})
                    log(f"GATE FIRED ({fires}x): restored the last passing state (step {step}); KILLED")
                    killed = True
                else:
                    lr_scale *= 0.5
                    hist.append({"step": step, "gate_fired": True, "lr_scale": lr_scale})
                    log(f"GATE FIRED ({fires}x): restored the last passing state (step {step}); LR x0.5, continuing")
        if step % a.ckpt_every == 0 or killed or step == a.steps:
            _save_ck(ck, model, names, opt, step, hist, base, lr_scale, fires, proj)
        if time.time() - t0 > a.budget:
            _save_ck(ck, model, names, opt, step, hist, base, lr_scale, fires, proj)
            (od / "history.json").write_text(json.dumps(hist, indent=1))
            log(f"budget: checkpoint at step {step} (resumable)")
            return
    export(a, model, names, tck, od, hist)


def export(a, model, names, tck, od, hist):
    from audioforge.train import load_model, save_model
    head = model.heads["tsvad"]
    cfg = {k: v for k, v in model.head_cfg["tsvad"].items() if k not in ("type", "from_layers", "weight")}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "cfg": {"type": "tsvad", "from_layers": [TAP], "weight": 0.0, "enroll_embedder": "spk", **cfg},
                "params": sum(p.numel() for p in head.parameters()), "history": hist}, od / "tsvad.pt")
    # the served-format student: the model without the tsvad head (serve reads it from tsvad.pt)
    del model.heads["tsvad"], model.head_cfg["tsvad"], model.cfg["heads"]["tsvad"]
    model.layer_tap.pop("tsvad", None)
    save_model(model.cpu(), od / "student.afm")
    srcsd = load_model(a.student, "cpu").state_dict()
    out = load_model(str(od / "student.afm"), "cpu").state_dict()
    bad = [k for k in srcsd if not k.startswith(tuple(names)) and not torch.equal(srcsd[k], out[k])]
    assert not bad, f"tensors outside the trainable set changed: {bad[:5]}"
    (od / "history.json").write_text(json.dumps(hist, indent=1))
    log(f"train {a.tag}: {od / 'student.afm'} + {od / 'tsvad.pt'}; the other {len(srcsd) - sum(k.startswith(tuple(names)) for k in srcsd)} tensors bit-identical")


# --------------------------------------------------------------------------- eval
def _prints_for(ds, ext, model, L=5.0):
    """Per window, the primary's L s print from elsewhere in the meeting (tsvad.py stage_vprints' bank rule / seeds)."""
    import tsvad as T
    segs_m, out = {}, []
    for v in ext:
        m = v["meeting"]
        if m not in segs_m:
            segs_m[m] = T.single_segments(ds, m)
        s = T.window_speakers(ds, v)[0]
        a0, b0 = float(v["start"]), float(v["start"]) + len(v["audio"]) / SR
        aud = None
        for i in range(T.VP_BANK):
            ivs = T.clip_from(segs_m[m].get(s, []), L, random.Random(f"{m}_{s}_{L}_{i}"))
            if ivs is None or any(x < b0 + 2 and y > a0 - 2 for x, y in ivs):
                continue
            aud = T.clip_audio(ds, m, ivs)
            break
        out.append(aud)
    return out


def stage_eval(a):
    import tsvad as T

    from audioforge.train import load_model
    dev = dev_of(a)
    torch.set_num_threads(a.threads)
    work = Path(a.work)
    od = work / a.tag
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    rec = {"tag": a.tag, "student": str(od / "student.afm"), "tracking": {}, "wer": {}}
    served = load_model(a.student, dev).eval()
    heads = {"served": T.load_head(Path(a.tsvad_init), dev)[0], "student": T.load_head(od / "tsvad.pt", dev)[0]}
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        if a.n_eval:
            ext = ext[: a.n_eval]
        prints = _prints_for(ds, ext, served)
        P = {k: [] for k in heads}
        Y = []
        for i in range(0, len(ext), 8):
            chunk = ext[i:i + 8]
            x, xl = served._pad([v["audio"] for v in chunk])
            h4, hl = block4(served, x, xl)
            E, has = voiceprints(served, prints[i:i + 8], dev)
            for k, h in heads.items():
                with torch.no_grad():
                    p = torch.sigmoid(h(h4.float(), hl, E, has)).float().cpu().numpy()
                for j, v in enumerate(chunk):
                    t = min(int(hl[j]), len(v["spk_act"]))
                    P[k].append(p[j, :t, 0])
            for j, v in enumerate(chunk):
                Y.append(np.asarray(v["spk_act"][: min(int(hl[j]), len(v["spk_act"]))], np.float32))
        y = np.concatenate(Y)
        r = {"n_windows": len(ext), "print_found": float(np.mean([p is not None for p in prints]))}
        for k in heads:
            pr, rc, f1, fa = T.frame_prf(np.concatenate(P[k]), y)
            r[k] = {"f1": round(f1, 4), "miss": round(1 - rc, 4), "fa": round(fa, 4)}
        rec["tracking"][corpus] = r
        log(f"tracking {corpus}: {r}")
    student = load_model(str(od / "student.afm"), dev).eval()
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    for s in ("libri", "ami", "icsi"):
        xs, refs = D.load_eval(work, s, a.n_wer)
        r = {}
        for name, m in (("served", served), ("student", student)):
            hyps = []
            with torch.no_grad():
                for i in range(0, len(xs), 8):
                    hyps += m.transcribe(xs[i:i + 8], head="rnnt", att_context_size=[70, 1])
            r[name] = round(float(wer([normalize_text(q) for q in refs], [normalize_text(h) for h in hyps])), 4)
        rec["wer"][s] = r
        log(f"WER {s}: {r}")
    res.setdefault("eval", {})[a.tag] = rec
    Path(a.out).write_text(json.dumps(res, indent=1))
    log(f"-> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["manifest", "evalsets", "teacher", "train", "eval"])
    p.add_argument("--work", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--budget", type=float, default=1e9, help="seconds per call (resumable stages)")
    p.add_argument("--student", default=str(ROOT / "runs/stage1_served.afm"))
    p.add_argument("--tsvad-init", default=str(ROOT / "runs/tsvad_spk.pt"))
    p.add_argument("--init", default=None, help="train: start from <work>/<tag>/good.pt of an earlier stage")
    # manifest / teacher (distill_0p6b_to_115m's options)
    p.add_argument("--n-ami", type=int, default=0)
    p.add_argument("--n-icsi", type=int, default=0)
    p.add_argument("--n-libri", type=int, default=28000)
    p.add_argument("--n-libri-spk-smoke", type=int, default=12)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--teacher", default=str(ROOT / "runs/nemo_nemotron_speech_streaming_en_0.6b.afm"))
    p.add_argument("--contexts", default="1")
    p.add_argument("--part", default="0/1")
    # train
    p.add_argument("--tag", default="s1")
    p.add_argument("--unfreeze", type=int, default=0, help="top encoder blocks to train (0 = heads only; 5 keeps "
                   "every served tap below block 13 bit-identical)")
    p.add_argument("--asr-kd", action="store_true", help="add the 0.6B teacher terms to the asr batches")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", type=float, default=0.85)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--p-tsvad", type=float, default=0.5)
    p.add_argument("--p-turn", type=float, default=0.5)
    p.add_argument("--p-asr", type=float, default=0.3)
    p.add_argument("--p-meeting", type=float, default=0.6)
    p.add_argument("--p-noprint", type=float, default=0.15)
    p.add_argument("--p-anchor", type=float, default=0.25)
    p.add_argument("--print-min", type=float, default=1.5)
    p.add_argument("--print-max", type=float, default=10.0)
    p.add_argument("--w-gt", type=float, default=0.5)
    p.add_argument("--w-seq", type=float, default=0.5)
    p.add_argument("--w-enc", type=float, default=1.0)
    p.add_argument("--w-kl", type=float, default=0.5)
    p.add_argument("--w-vad", type=float, default=0.5, help="with --unfreeze: VAD head label loss on the tsvad crops")
    p.add_argument("--gate-every", type=int, default=1000)
    p.add_argument("--gate-n", type=int, default=200)
    p.add_argument("--gate-libri", type=float, default=0.003)
    p.add_argument("--gate-ami", type=float, default=0.01)
    p.add_argument("--max-fires", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # eval
    p.add_argument("--corpora", default="ami,icsi")
    p.add_argument("--n-eval", type=int, default=0, help="eval: first N eot-bench windows per corpus (0 = all)")
    p.add_argument("--n-wer", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "runs" / "single_distill.json"))
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
