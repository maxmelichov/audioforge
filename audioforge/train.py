"""Recipe-driven training: YAML -> tokenizer + SpeechModel -> Trainer -> .afm archive."""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import tarfile
import tempfile
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import Collate, attach_teacher, prompt_specials, read_manifest, synthetic_dataset, to_device
from .metrics import frame_der, wer
from .model import TEXT_HEADS, SpeechModel
from .tokenizer import load_tokenizer, train_tokenizer

log = logging.getLogger(__name__)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_recipe(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:  # dotted.key=value, value parsed as YAML
        k, v = ov.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(v)
    return cfg


def load_data(cfg: dict, split: str) -> list[dict]:
    d = cfg["data"]
    if "mix" in d:  # several sources (e.g. synthetic conversations + AMI turn + AMI diar): see load_mix
        return load_mix(cfg, split)
    data = _load_source(cfg, split)
    if d.get("spk_teacher"):  # cached teacher embeddings per example (speaker distillation, research/SPK_HEAD.md)
        data = attach_teacher(data, d["spk_teacher"], split)
    return data


def _load_source(cfg: dict, split: str) -> list[dict]:
    d = cfg["data"]
    if "librispeech" in d:  # real speech: audioforge/datasets/librispeech.py
        from .datasets.librispeech import recipe_data
        return recipe_data(cfg, split)
    if "ami" in d:  # AMI meetings (diarization / speaker-aware EOT): audioforge/datasets/ami.py
        from .datasets.ami import recipe_data as ami_data
        return ami_data(cfg, split)
    if "icsi" in d:  # ICSI meetings, AMI's label rules (speaker ids 1000+): audioforge/datasets/icsi.py
        from .datasets.icsi import recipe_data as icsi_data
        return icsi_data(cfg, split)
    if "dyadic" in d:  # two-party conversations (Behavior-SD / DailyTalk / otoSpeech): audioforge/datasets/dyadic.py
        from .datasets.dyadic import recipe_data as dyadic_data
        return dyadic_data(cfg, split)
    if "smartturn" in d:  # pipecat smart-turn clips (complete / incomplete utterances): audioforge/datasets/smartturn.py
        from .datasets.smartturn import recipe_data as smartturn_data
        return smartturn_data(cfg, split)
    if "manifest" in d:
        return read_manifest(d["manifest"][split], cfg.get("sample_rate", 16000))
    n, kind = d["synthetic"][f"n_{split}"], d["synthetic"]["kind"]
    if kind == "conversation":  # speaker-aware end-of-turn data (conversation.py)
        from .conversation import conversation_dataset
        return conversation_dataset(n, seed=0 if split == "train" else 1, preset=d["synthetic"].get("preset", "default"))
    return synthetic_dataset(kind, n, seed=0 if split == "train" else 1,
                             codec=build_codec(d) if kind == "enhance" else None)


MIX_META = ("weight", "name", "drop_keys")


def derive_labels(ex: dict, derive=()) -> dict:
    """Labels computed from others for items that lack them (``data.derive``):
    vad = any speaker active in ``spk_targets``; eou = the 2 frames after ``turn_end_frame`` (the primary
    speaker's turn end; same width as ToneLanguage / LibriSpeech eou targets)."""
    if "vad" in derive and "vad" not in ex and "spk_targets" in ex:
        ex["vad"] = (np.asarray(ex["spk_targets"]).max(1) > 0.5).astype(np.float32)
    if "eou" in derive and "eou" not in ex and "turn_end_frame" in ex:
        ref = next(ex[k] for k in ("spk_act", "eot", "spk_targets", "vad") if k in ex)
        eou = np.zeros(len(ref), np.float32)
        eou[int(ex["turn_end_frame"]): int(ex["turn_end_frame"]) + 2] = 1
        ex["eou"] = eou
    return ex


class MixedData(Sequence):
    """Train/val data drawn from several sources, items fetched lazily (a ConversationDataset stays lazy).

    Weight semantics (per source, deterministic): ``weight = k + f`` repeats every item k times and adds a
    seeded sample of round(f * n) distinct items, so 1.0 = as is, 2.0 = twice, 0.5 = a random half, 1.5 =
    all once + half again. The repetition is fixed for the whole run (not re-drawn per epoch).
    ``source_ids[i]`` is the source of item i; ``batching`` ("source" | "mixed") tells Trainer.fit how
    to batch (see Trainer._batches)."""

    def __init__(self, sources: list, names: list[str], weights: list[float], drop_keys: list, derive=(),
                 seed: int = 0, batching: str = "source"):
        self.sources, self.names, self.drop, self.derive, self.batching = sources, names, drop_keys, derive, batching
        assert batching in ("source", "mixed"), f"data.batching: source | mixed, got {batching!r}"
        self.index = []
        for s, (data, w) in enumerate(zip(sources, weights)):
            n, k = len(data), int(w)
            extra = sorted(random.Random(seed + s).sample(range(n), int(round((w - k) * n)))) if n else []
            self.index += [(s, i) for i in list(range(n)) * k + extra]
        self.source_ids = [s for s, _ in self.index]

    @property
    def counts(self) -> dict:
        return {nm: self.source_ids.count(s) for s, nm in enumerate(self.names)}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        s, j = self.index[i]
        ex = {k: v for k, v in self.sources[s][j].items() if k not in self.drop[s]}
        return derive_labels(ex, self.derive)


def load_mix(cfg: dict, split: str) -> MixedData:
    """``data: {mix: [<source>, ...], val: <source>, derive: [vad, eou], batching: source, standin: {...}}``

    A <source> is any single-source ``data`` block (``synthetic`` / ``ami`` / ``librispeech`` / ``manifest``,
    loaded by load_data exactly as a one-source recipe would) plus optional ``weight`` (default 1.0, see
    MixedData), ``name`` and ``drop_keys`` (e.g. [speaker] so synthetic speaker ids 0-7 do not collide with
    AMI's 190 global ids). Train = the weighted concatenation of every source's train split; val = the
    single ``val`` source's val split (default: the first source). ``standin: {n_train, n_val}`` merges
    ``synthetic: {n_train, n_val}`` into every source (hermetic smoke tests: AMI/LibriSpeech return their
    synthetic stand-ins). ``batching: source`` (default) makes every batch single-source, so a batch never
    mixes label sets; ``mixed`` batches across sources with MixCollate (only correct for heads that honour
    ``<key>_present``, see MixCollate)."""
    d = cfg["data"]
    stand = d.get("standin")

    def one(entry, sp):
        src = {k: v for k, v in entry.items() if k not in MIX_META}
        if stand:
            src["synthetic"] = {**(src.get("synthetic") or {}), **stand}
        return load_data({**cfg, "data": src}, sp)

    if split == "train":
        entries = d["mix"]
        names = [e.get("name") or f"{i}:{next(k for k in e if k not in MIX_META)}" for i, e in enumerate(entries)]
        data = MixedData([one(e, "train") for e in entries], names, [float(e.get("weight", 1.0)) for e in entries],
                         [set(e.get("drop_keys") or ()) for e in entries], d.get("derive") or (),
                         seed=cfg.get("seed", 0), batching=d.get("batching", "source"))
        log.info(f"[mix] train {len(data)} items: {data.counts}")
        return data
    entry = d.get("val") or d["mix"][0]
    return MixedData([one(entry, "val")], ["val"], [1.0], [set(entry.get("drop_keys") or ())],
                     d.get("derive") or (), batching="source")


class MixCollate:
    """Collate for batches that mix sources (``data.batching: mixed``). data.Collate refuses a label key that
    only some items carry; this fills the gaps with empty placeholders (a 0-frame array -> ``<key>_len`` 0,
    "" -> ``text_len`` 0, speaker id 0) and adds ``<key>_present`` (B,) bool for each such key.

    Heads that already honour it through ``<key>_len``: turn (masks by spk_act_len), codec_tokens (codes_len).
    Heads that would train on the placeholders and need a per-item mask before ``mixed`` is safe:
    frame (masks by enc_len only -> would learn all-zero targets), sortformer (same -> silence),
    speaker (id 0 -> wrong class), ctc/rnnt/tdt/aed (empty transcript -> blank-only targets)."""

    LABELS = ("text", "source_text", "prompt", "speaker")

    def __init__(self, base: Collate):
        self.base = base

    def __call__(self, batch: list[dict]) -> dict:
        keys = set().union(*(ex.keys() for ex in batch)) - {"audio"}
        fill = {}
        for k in keys:
            have = [ex[k] for ex in batch if k in ex]
            if len(have) == len(batch):
                continue
            v = have[0]
            if isinstance(v, np.ndarray) and v.ndim in (1, 2) and v.dtype.kind in "biuf":
                fill[k] = np.zeros((0, *v.shape[1:]), v.dtype)
            elif k in ("text", "source_text", "prompt"):
                fill[k] = ""
            elif k == "speaker":
                fill[k] = 0
        items = [{**{k: v for k, v in fill.items() if k not in ex}, **ex} for ex in batch]
        # keys that stay partial and are not labels (metadata such as meeting / events) are dropped
        items = [{k: v for k, v in ex.items() if k in fill or all(k in e for e in batch)} for ex in items]
        out = self.base(items)
        for k in fill:
            out[k + "_present"] = torch.tensor([k in ex for ex in batch])
        return out


def build_codec(data_cfg: dict):
    """Codec whose tokens are the targets. Loads trained weights if ``codec_path`` is given;
    otherwise a fixed-seed random codec (still a deterministic tokenizer of the signal)."""
    from .codec import AudioCodec, MelCodec
    cc = dict(data_cfg.get("codec", {}))
    cls = MelCodec if cc.pop("type", "wave") == "mel" else AudioCodec
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        codec = cls(**cc).eval()
    if data_cfg.get("codec_path"):
        codec.load_state_dict(torch.load(data_cfg["codec_path"], map_location="cpu"))
    return codec


def build_tokenizer(cfg: dict, train: list[dict]):
    tc = dict(cfg.get("tokenizer", {"kind": "char"}))
    kind = tc.pop("kind", "char")
    specials = tc.pop("specials", []) + prompt_specials(tc.pop("langs", ["en", "xx"]))
    texts = [ex[k] for ex in train for k in ("text", "source_text") if k in ex]
    return train_tokenizer(kind, texts or ["a"], specials=specials, **tc)


class Trainer:
    def __init__(self, model: SpeechModel, cfg: dict, device=None):
        self.model, self.cfg = model, cfg
        t = cfg.get("trainer", {})
        self.device = device or pick_device(t.get("device", "auto"))
        self.model.to(self.device)
        self.steps, self.bs = t.get("max_steps", 1000), t.get("batch_size", 16)
        self.clip, self.log_every = t.get("grad_clip", 1.0), t.get("log_every", 50)
        self.eval_every = t.get("eval_every", 0)
        lr, mult = t.get("lr", 1e-3), (cfg.get("init") or {}).get("pretrained_lr_mult")
        pre = getattr(model, "pretrained_keys", set())
        scope = (cfg.get("init") or {}).get("pretrained_scope")
        if scope:  # only these loaded tensors get lr*mult (e.g. stage 2: backbone + ASR; stage-1 heads at full lr)
            pre = {k for k in pre if any(k.startswith(p) for p in scope)}
        layers = (cfg.get("init") or {}).get("trainable_encoder_layers")
        if layers is not None:  # partial unfreeze: only these encoder blocks train (memory: no optimizer state
            keep = set(int(i) for i in layers)  # or stored activations below the first trainable block)
            for n, p in model.encoder.named_parameters():
                blk = n.split(".")[1] if n.startswith("layers.") else None
                if blk is None or int(blk) not in keep:
                    p.requires_grad_(False)
            log.info(f"[trainer] encoder blocks trainable: {sorted(keep)} of {len(model.encoder.layers)}")
        only = (cfg.get("init") or {}).get("train_only")
        if only:  # init.from: train ONLY parameters under these prefixes (loaded or new), freeze everything else
            only = [str(p) for p in only]
            pre = {k for k in pre if not any(k.startswith(p) for p in only)}
            n_on = 0
            for n, p in model.named_parameters():
                if not any(n.startswith(q) for q in only):
                    p.requires_grad_(False)
                else:
                    n_on += p.numel()
            log.info(f"[trainer] train_only {only}: {n_on / 1e6:.3f}M trainable params, the rest frozen")
        if mult is not None and pre:  # init.from: pretrained tensors at lr*mult (0 freezes them), new heads at lr
            groups = {"pretrained": [], "new": []}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    groups["pretrained" if n in pre else "new"].append(p)
            if mult == 0:
                for p in groups["pretrained"]:
                    p.requires_grad_(False)
                params = [{"params": groups["new"], "lr": lr}]
            else:
                params = [{"params": groups["pretrained"], "lr": lr * mult}, {"params": groups["new"], "lr": lr}]
            log.info(f"[trainer] {sum(p.numel() for p in groups['pretrained']) / 1e6:.1f}M pretrained params at "
                  f"lr x{mult}, {sum(p.numel() for p in groups['new']) / 1e6:.2f}M new params at lr")
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        rows = t.get("row_lr")  # {params: [names], rows: [ids], lr: x}: only these rows of these tensors train, at lr x
        if rows:  # (new vocabulary rows, research/YIELD_TOKENS.md: Adam moves every element by ~lr per step, so new
            named = dict(model.named_parameters())  # output rows starting near zero cannot grow at a gate-safe lr)
            idx = torch.as_tensor(sorted(int(i) for i in rows["rows"]))
            fast = []
            for name in rows["params"]:
                prm = named[name]
                mask = torch.zeros(prm.shape[0], dtype=prm.dtype)
                mask[idx] = 1
                prm.register_hook(lambda g, m=mask.view(-1, *([1] * (prm.dim() - 1))): g * m.to(g.device))
                fast.append(prm)
            fast_ids = {id(p) for p in fast}
            if isinstance(params[0], dict):
                for g in params:
                    g["params"] = [p for p in g["params"] if id(p) not in fast_ids]
            else:
                params = [{"params": [p for p in params if id(p) not in fast_ids], "lr": lr}]
            params.append({"params": fast, "lr": float(rows["lr"]), "weight_decay": 0.0})
            log.info(f"[trainer] row_lr: rows {rows['rows']} of {rows['params']} at lr {rows['lr']}; their other rows frozen")
        self.opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=t.get("weight_decay", 1e-3))
        warm = t.get("warmup_steps", max(1, self.steps // 10))
        # linear warmup + cosine decay (NeMo recipes use Noam/cosine with warmup)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / warm) * (
            0.5 * (1 + math.cos(math.pi * min(1.0, s / self.steps))) * 0.95 + 0.05))
        self.history: list[dict] = []
        self._setup_guards(t)

    # ------------------------------------------------------------------ R2 guards (KL anchor, MCR, WER gate)
    def _setup_guards(self, t: dict):
        """Optional ``trainer:`` keys (all off by default):
        anchor: {weight: 0.5, heads: [rnnt], symmetric: true, teacher: <.afm, default = weights at start>}
        mode_consistency: {weight: 0.3, contexts: [[-1,-1],[70,1]], heads: [rnnt], symmetric: true,
                           loss_weight: 1.0, detach_targets: true}
        wer_gate: {manifest: ..., every: 500, max_delta: 0.2, n: 50, head: <primary>, att_context_size: null}
        checkpoint_every: 0  (-> <out stem>.step<k>.afm; needs fit(out=...)); checkpoint_keep: k (keep the newest k)"""
        from .losses.consistency import TRANSDUCERS, FrozenTeacher
        m = self.model
        tx = [k for k, v in getattr(m, "head_cfg", {}).items() if v["type"] in TRANSDUCERS]

        def heads_of(c, what):
            hs = list(c.get("heads") or tx)
            bad = [h for h in hs if h not in tx]
            if not hs or bad:
                raise ValueError(f"trainer.{what}: needs transducer heads, got {hs} (model has {tx})")
            return hs

        self.anchor = dict(t.get("anchor") or {}) or None
        self.teacher = None
        if self.anchor:
            self.anchor.setdefault("weight", 0.5)
            self.anchor["heads"] = heads_of(self.anchor, "anchor")
            if self.anchor.get("teacher"):
                self.teacher = FrozenTeacher.from_afm(self.anchor["teacher"], self.anchor["heads"], self.device)
            else:
                if not getattr(m, "pretrained_keys", None):
                    log.info("[trainer] anchor: model has no init.from weights; anchoring to its initial state")
                self.teacher = FrozenTeacher(m, self.anchor["heads"]).to(self.device)
        self.mcr = dict(t.get("mode_consistency") or {}) or None
        if self.mcr:
            self.mcr.setdefault("weight", 0.3)
            self.mcr["contexts"] = [list(c) for c in self.mcr.get("contexts", [[-1, -1], [70, 1]])]
            assert len(self.mcr["contexts"]) == 2, "mode_consistency.contexts: exactly two [left, right] sizes"
            self.mcr["heads"] = heads_of(self.mcr, "mode_consistency")
        self.wer_gate = dict(t.get("wer_gate") or {}) or None
        if self.wer_gate:
            for k, v in {"every": 500, "max_delta": 0.2, "n": 50, "head": None}.items():
                self.wer_gate.setdefault(k, v)
        self.checkpoint_every = t.get("checkpoint_every", 0)
        self._gate_data, self.stopped, self.checkpoints = None, None, []

    def _head_in(self, name, enc, hidden):
        """What head ``name`` sees in SpeechModel.forward (layer mix + grad_scale)."""
        m = self.model
        e = m.head_input(name, enc, hidden)
        if "grad_scale" in m.head_cfg[name]:
            from .model import GradScale
            e = GradScale.apply(e, float(m.head_cfg[name]["grad_scale"]))
        return e

    def _consistency(self, batch: dict, enc: dict) -> dict:
        """KL terms for one step (weighted into ``loss`` by fit). ``enc``: SpeechModel.forward(return_enc=True)."""
        from .losses.consistency import head_kl
        m, out = self.model, {}
        if self.anchor:
            te, _, thid = self.teacher.encode(enc["feats"], enc["flen"], enc["att"])
            kls = []
            for name in self.anchor["heads"]:
                head = m.heads[name]
                if head.key not in batch:
                    continue
                with torch.no_grad():
                    t_in = self.teacher.head_input(name, te, thid)
                kls.append(head_kl(head, self._head_in(name, enc["enc"], enc["hidden"]), self.teacher.heads[name],
                                   t_in, enc["elen"], batch[head.key], batch[head.key + "_len"],
                                   symmetric=self.anchor.get("symmetric", True)))
            if kls:
                out["kl_anchor"] = sum(kls) / len(kls)
        if self.mcr:
            enc_b, elen_b, hid_b = m.encoder(enc["feats"], enc["flen"], self.mcr["contexts"][1], return_hidden=True)
            kls = []
            for name in self.mcr["heads"]:
                head, hc = m.heads[name], m.head_cfg[name]
                if head.key not in batch:
                    continue
                y, yl = batch[head.key], batch[head.key + "_len"]
                e_b = self._head_in(name, enc_b, hid_b)
                if self.mcr.get("loss_weight", 1.0):  # the second mode is trained too (paper: dual-mode loss)
                    out[f"loss_{name}_mode2"] = self.mcr.get("loss_weight", 1.0) * hc.get("weight", 1.0) * \
                        head.loss(e_b, elen_b, batch)
                kls.append(head_kl(head, self._head_in(name, enc["enc"], enc["hidden"]), head, e_b, elen_b, y, yl,
                                   symmetric=self.mcr.get("symmetric", True),
                                   detach_targets=self.mcr.get("detach_targets", True)))
            if kls:
                out["kl_mode"] = sum(kls) / len(kls)
        return out

    def _gate_wer(self) -> float:
        """WER of ``wer_gate.head`` on the first n utterances of the gate manifest (normalized text)."""
        from .data import load_wav
        g, m = self.wer_gate, self.model
        if self._gate_data is None:
            lines = [json.loads(x) for x in Path(g["manifest"]).read_text().splitlines() if x.strip()][: g["n"]]
            sr = m.preprocessor.sample_rate
            self._gate_data = [(load_wav(x["audio_filepath"], sr), x["text"]) for x in lines]
        try:
            from .teachers import normalize_text
        except Exception:  # teachers pulls optional deps; same rule inline
            import re

            def normalize_text(s):
                return " ".join(re.sub(r"[^\w\s']", " ", s.lower().replace("-", " ")).split())
        hyps = []
        for i in range(0, len(self._gate_data), 8):
            hyps += m.transcribe([a for a, _ in self._gate_data[i: i + 8]], head=g["head"],
                                 att_context_size=g.get("att_context_size"))
        m.train()
        return wer([normalize_text(r) for _, r in self._gate_data], [normalize_text(h) for h in hyps])

    def _checkpoint(self, step: int):
        path = str(Path(self.out).with_suffix("")) + f".step{step}.afm"
        save_model(self.model, path)
        self.checkpoints.append(path)
        log.info("%s %s", "checkpoint", path)
        keep = int(self.cfg.get("trainer", {}).get("checkpoint_keep", 0) or 0)
        if keep:  # frequent checkpoints (gate every 50 steps): keep only the newest ``keep`` files
            for old in self.checkpoints[:-keep]:
                Path(old).unlink(missing_ok=True)
            self.checkpoints = self.checkpoints[-keep:]

    def fit(self, train: list[dict], val: list[dict] | None = None, collate=None, out: str | None = None) -> list[dict]:
        """Train for max_steps. With ``wer_gate`` the run stops early (``self.stopped`` is set) when WER rises
        more than max_delta over the step-0 baseline; the model is then reloaded from the last good
        checkpoint, if one was written."""
        if len(train) < max(1, self.bs):
            raise ValueError(f"train has {len(train)} items but trainer.batch_size={self.bs}: no full batch "
                             "can be formed (lower trainer.batch_size or add data)")
        mixed = getattr(train, "batching", None) == "mixed"
        if mixed:
            log.info("[trainer] data.batching=mixed: frame/sortformer/speaker/text heads do not honour <key>_present "
                  "yet and train on placeholders for items without their labels (see MixCollate)")
        collate = collate or (MixCollate(Collate(self.model.tokenizer)) if mixed else Collate(self.model.tokenizer))
        rng = random.Random(0)
        self.out = out
        if self.checkpoint_every and not out:
            log.info("[trainer] checkpoint_every set but no output path: checkpoints disabled")
        guard = bool(self.anchor or self.mcr)
        fkw = {"return_enc": True, "att_context_size": self.mcr["contexts"][0] if self.mcr else None} if guard else {}
        gate_base = None
        if self.wer_gate:
            gate_base = self._gate_wer()
            self.history.append({"step": 0, "wer_gate": gate_base, "wer_gate_baseline": gate_base})
            log.info(f"wer_gate baseline={gate_base:.4f}")
        self.model.train()
        t0, step = time.time(), 0
        while step < self.steps:
            trained = False
            for b_idx in self._batches(train, rng):
                batch = to_device(collate([train[j] for j in b_idx]), self.device)
                o = self.model(batch, **fkw)
                if not (torch.is_tensor(o["loss"]) and o["loss"].requires_grad):
                    continue  # no head had labels in this batch (or all of them are frozen)
                trained = True
                if guard:
                    terms = self._consistency(batch, o.pop("enc"))
                    for k, v in terms.items():
                        w = 1.0 if k.startswith("loss_") else (self.anchor if k == "kl_anchor" else self.mcr)["weight"]
                        o["loss"] = o["loss"] + w * v
                    o.update(terms)
                self.opt.zero_grad(set_to_none=True)
                o["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt.step()
                self.sched.step()
                step += 1
                rec = {"step": step, **{k: float(v.detach()) for k, v in o.items()}, "grad_norm": float(gn),
                       "lr": self.sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                self.history.append(rec)
                if step % self.log_every == 0 or step == 1:
                    log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
                if val and self.eval_every and step % self.eval_every == 0:
                    log.info("%s %s", "eval", self.evaluate(val))
                    self.model.train()
                if self.wer_gate and step % self.wer_gate["every"] == 0:
                    w = self._gate_wer()
                    rec.update(wer_gate=w, wer_gate_baseline=gate_base)
                    log.info(f"wer_gate step={step} wer={w:.4f} baseline={gate_base:.4f}")
                    if w - gate_base > self.wer_gate["max_delta"]:
                        good = self.checkpoints[-1] if self.checkpoints else None
                        self.stopped = {"step": step, "wer": w, "baseline": gate_base, "restored_from": good}
                        log.info(f"[trainer] STOP: WER {w:.4f} > baseline {gate_base:.4f} + "
                              f"{self.wer_gate['max_delta']}; last good checkpoint: {good}")
                        if good:
                            self.model.load_state_dict(load_model(good, self.device).state_dict())
                        return self.history
                if self.checkpoint_every and out and step % self.checkpoint_every == 0:
                    self._checkpoint(step)
                if step >= self.steps:
                    break
            if not trained:
                raise ValueError("a full pass over train produced no trainable loss: no batch carries labels "
                                 "for a head with trainable parameters")
        return self.history

    def _batches(self, train, rng) -> list[list[int]]:
        """Index batches for one epoch (full batches only). A MixedData with batching "source" gets
        single-source batches (sources smaller than batch_size contribute none), in shuffled order."""
        n, bs = len(train), self.bs
        src = getattr(train, "source_ids", None)
        if src is None or getattr(train, "batching", "source") != "source":
            idx = list(range(n))
            rng.shuffle(idx)
            return [idx[i: i + bs] for i in range(0, n - bs + 1, bs)]
        groups: dict[int, list[int]] = {}
        for i, s in enumerate(src):
            groups.setdefault(s, []).append(i)
        out = []
        for s, g in sorted(groups.items()):
            if len(g) < bs and not getattr(self, "_warned_small", False):
                log.info(f"[trainer] source {train.names[s]!r} has {len(g)} < batch_size items: never batched")
            rng.shuffle(g)
            out += [g[i: i + bs] for i in range(0, len(g) - bs + 1, bs)]
        self._warned_small = True
        if not out:
            raise ValueError(f"no data source has >= trainer.batch_size={bs} items")
        rng.shuffle(out)
        return out

    @torch.no_grad()
    def evaluate(self, val: list[dict], n: int = 64, chunk: int = 16) -> dict:
        m, res = self.model, {}
        m.eval()
        val = list(val[:n])
        if not val:
            return res
        dev = next(m.parameters()).device
        for name, hc in m.head_cfg.items():
            ref_key = hc.get("key", "text")
            if hc["type"] in TEXT_HEADS and not hc.get("condition_on_speaker") and ref_key in val[0]:
                hyps = []
                for i in range(0, len(val), 16):
                    part = val[i: i + 16]
                    if hc["type"] == "aed":
                        for ex in part:
                            hyps += m.transcribe([ex["audio"]], head=name, prompt=ex.get("prompt"))
                    else:
                        hyps += m.transcribe([ex["audio"] for ex in part], head=name)
                res[f"wer_{name}"] = round(wer([ex[ref_key] for ex in val], hyps), 4)
        others = [k for k, hc in m.head_cfg.items() if hc["type"] not in TEXT_HEADS]
        if others:
            res.update(self._eval_non_text(val, others, dev, chunk))
        # speaker-kernel ASR: WER with the oracle activity of the target speaker (also for ASR-only recipes)
        cond = [k for k, hc in m.head_cfg.items() if hc.get("condition_on_speaker") and hc["type"] in TEXT_HEADS]
        if cond and "spk_act" in val[0] and "text" in val[0]:
            name = cond[0]
            hyps = [m.transcribe([ex["audio"]], head=name,
                                 spk_act=torch.as_tensor(ex["spk_act"], device=dev)[None])[0] for ex in val]
            res[f"wer_{name}_oracle_act"] = round(wer([ex["text"] for ex in val], hyps), 4)
        return res

    def _eval_non_text(self, val: list[dict], others: list[str], dev, chunk: int) -> dict:
        """Frame / sortformer / speaker / codec metrics, accumulated over val in chunks (bounded memory).
        Heads whose label key is absent are skipped (as in SpeechModel.forward)."""
        m, res, acc = self.model, {}, {}
        col = Collate(m.tokenizer)
        batched = [k for k in others if not hasattr(m.heads[k], "evaluate_model")]
        for i in range(0, len(val), chunk) if batched else ():
            batch = to_device(col(val[i: i + chunk]), dev)
            enc0, elen, hidden = m.encode(batch["audio"], batch["audio_len"], return_hidden=True)
            for name in batched:
                hc, head = m.head_cfg[name], m.heads[name]
                key = getattr(head, "key", None)
                if key and key not in batch:
                    continue
                s = acc.setdefault(name, {})
                add = lambda k, v: s.__setitem__(k, s.get(k, 0.0) + float(v))  # noqa: E731, B023 - used in this iteration
                enc = m.head_input(name, enc0, hidden)
                if hc["type"] == "sortformer":
                    pred = head.decode(enc, elen)
                    # labels have ToneLanguage.n_frames(len(audio)) frames; the encoder may have one more
                    # (subsampling_padding: nemo, causal) or fewer: score the common prefix, as _match_len does
                    # in training (tests/test_eval_frames.py)
                    n = torch.minimum(elen, batch[key + "_len"]).tolist()
                    for b in range(len(elen)):
                        add("der", frame_der(pred[b, : n[b]].cpu(), batch[key][b, : n[b]].cpu()))
                        add("n", 1)
                elif hc["type"] == "frame":
                    out, lab = head.decode(enc, elen), batch[key]
                    T = min(out.shape[1], lab.shape[1])
                    if head.num_classes > 1:  # softmax output: argmax class, per-class recall for c > 0
                        p, t = out[:, :T].argmax(-1), lab[:, :T].long()
                    else:
                        p, t = out[:, :T] > 0.5, lab[:, :T] > 0.5
                    ar = torch.arange(T, device=dev)[None]
                    valid = (ar < elen[:, None]) & (ar < batch[key + "_len"][:, None])
                    add("correct", ((p == t) & valid).sum())
                    add("total", valid.sum())
                    for c in (range(1, head.num_classes) if head.num_classes > 1 else [None]):
                        tc, pc = (t, p) if c is None else (t == c, p == c)
                        add(f"tp{c}", (pc & tc & valid).sum())
                        add(f"pos{c}", (tc & valid).sum())
                elif hc["type"] == "codec_tokens":
                    p = head.decode(enc, elen)
                    T = min(p.shape[1], batch["codes"].shape[1])
                    ar = torch.arange(T, device=dev)[None]
                    valid = (ar < batch["codes_len"][:, None]) & (ar < elen[:, None])
                    eq = (p[:, :T] == batch["codes"][:, :T])[valid]
                    add("correct", eq.sum())
                    add("total", eq.numel())
                elif hc["type"] == "speaker":
                    e = head.embed(enc, elen)
                    logits = e @ torch.nn.functional.normalize(head.W, dim=-1).T
                    add("correct", (logits.argmax(-1) == batch["speaker"]).sum())
                    add("total", len(elen))
        for name, s in acc.items():
            t = m.head_cfg[name]["type"]
            if t == "sortformer":
                res[f"der_{name}"] = round(s["der"] / s["n"], 4)
            elif t == "frame":
                res[f"acc_{name}"] = round(s["correct"] / max(1.0, s["total"]), 4)
                for k in [k for k in s if k.startswith("pos")]:
                    c = k[3:]
                    if s[k] > 0:
                        res[f"recall_{name}" + ("" if c == "None" else f"_{c}")] = round(s["tp" + c] / s[k], 4)
            elif t == "codec_tokens":
                res[f"token_acc_{name}"] = round(s["correct"] / max(1.0, s["total"]), 4)
            elif t == "speaker":
                res[f"spk_acc_{name}"] = round(s["correct"] / max(1.0, s["total"]), 4)
        for name in others:  # heads with their own benchmark (e.g. turn: eot-bench sweep)
            if hasattr(m.heads[name], "evaluate_model"):
                res.update(m.heads[name].evaluate_model(m, name, val))
        return res


# --------------------------------------------------------------------------- archive (.afm ~ .nemo)
def save_model(model: SpeechModel, path: str | Path):
    """Single-file archive: config.yaml + weights + tokenizer, like NeMo's .nemo tarball."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "config.yaml").write_text(yaml.safe_dump(model.cfg, sort_keys=False))
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, d / "model_weights.pt")
        if model.tokenizer is not None:
            model.tokenizer.save(d)
        with tarfile.open(path, "w:gz") as tar:
            for f in d.iterdir():
                tar.add(f, arcname=f.name)


def load_model(path: str | Path, device="cpu") -> SpeechModel:
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(path) as tar:
            tar.extractall(d, filter="data")
        d = Path(d)
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        tok = load_tokenizer(d) if (d / "tokenizer.json").exists() else None
        model = SpeechModel(cfg, tok)
        model.load_state_dict(torch.load(d / "model_weights.pt", map_location="cpu"))
    return model.to(device).eval()


def export_onnx(model: SpeechModel, path: str | Path, seconds: float = 4.0):
    """Export preprocessed-features -> encoder outputs (the part Riva/TensorRT deploys)."""

    class Enc(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, feats, lengths):
            return self.enc(feats, lengths, att_context_size=self.enc.att_context_size)

    m = Enc(model.encoder).eval().cpu()
    T = int(seconds * 100)
    x = torch.randn(1, model.preprocessor.n_mels, T)
    l = torch.tensor([T])
    torch.onnx.export(m, (x, l), str(path), input_names=["features", "lengths"],
                      output_names=["encoded", "encoded_lengths"],
                      dynamic_axes={"features": {0: "B", 2: "T"}, "lengths": {0: "B"},
                                    "encoded": {0: "B", 1: "T8"}, "encoded_lengths": {0: "B"}},
                      opset_version=17, dynamo=False)
    return path


def init_from_afm(cfg: dict, path: str) -> tuple[SpeechModel, dict]:
    """Start a recipe from a trained or ported .afm (e.g. runs/nemo_hybrid_streaming_multi.afm).

    Keeps the source's preprocessor, encoder, tokenizer and heads; the recipe's ``heads`` add new
    heads or adjust existing ones (merged per head, so ``rnnt: {weight: 0.5}`` keeps the pretrained
    RNNT config), and its ``encoder`` may override streaming/conditioning keys only
    (att_context_size(s), speaker_kernel_layers). Training sections come from the recipe.
    Loaded parameter names are recorded in ``model.pretrained_keys`` for per-group learning rates.
    """
    src = load_model(path)
    new_cfg = copy.deepcopy(src.cfg)
    heads = {k: dict(v) for k, v in src.cfg["heads"].items()}
    for k, v in cfg.get("heads", {}).items():
        heads[k] = {**heads.get(k, {}), **v}
    new_cfg["heads"] = heads
    for k in ("att_context_size", "att_context_sizes", "speaker_kernel_layers"):
        if k in cfg.get("encoder", {}):
            new_cfg["encoder"][k] = cfg["encoder"][k]
    for k in ("name", "data", "trainer", "spec_augment", "decoding", "seed", "init", "tokenizer"):
        if k in cfg:
            new_cfg[k] = cfg[k]
    model = SpeechModel(new_cfg, src.tokenizer)
    src_sd = src.state_dict()
    msd = model.state_dict()
    # a head whose from_layers changed (all -> [k]) no longer owns a layer_mix tensor: drop the source's
    # tensor; one whose mix changed size (all -> [1..9], or a different sub-mix) gets a fresh (uniform) mix
    dropped = [k for k in src_sd if k.startswith("layer_mix.")
               and (k not in msd or tuple(src_sd[k].shape) != tuple(msd[k].shape))]
    # a speaker head's AAM classifier re-sized by the recipe (num_speakers) is re-initialised, the rest of the head
    # kept; any other shape change still fails to load (the turn head opts in through its own init_partial)
    spk_cls = {f"heads.{k}.W" for k, v in heads.items() if v.get("type") == "speaker"}
    resized = [k for k in src_sd if k in spk_cls and k in msd and tuple(src_sd[k].shape) != tuple(msd[k].shape)]
    if dropped or resized:
        log.info(f"[init.from] dropping {dropped} (from_layers changed); re-initialising {resized} (shape changed)")
        src_sd = {k: v for k, v in src_sd.items() if k not in dropped and k not in resized}
    missing, unexpected = model.load_state_dict(src_sd, strict=False)
    assert not unexpected, f"init.from: unexpected keys {unexpected[:5]}"
    model.pretrained_keys = set(src_sd) & set(model.state_dict())
    new_heads = [k for k in heads if k not in src.cfg["heads"]]
    log.info(f"[init.from {path}] loaded {len(model.pretrained_keys)} tensors; new heads {new_heads}; "
          f"{len(missing)} new tensors")
    return model, new_cfg


def build_model(cfg: dict, train: list[dict]) -> tuple[SpeechModel, dict]:
    """Model + tokenizer for a recipe: from ``init.from`` if given, else fresh (tokenizer trained on data)."""
    init = cfg.get("init") or {}
    if init.get("from"):
        return init_from_afm(cfg, init["from"])
    tok = build_tokenizer(cfg, train) if any(v["type"] in TEXT_HEADS for v in cfg["heads"].values()) else None
    return SpeechModel(cfg, tok), cfg


def run_recipe(path: str, overrides: list[str] | None = None, out: str | None = None) -> tuple[SpeechModel, dict]:
    cfg = load_recipe(path, overrides)
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    train, val = load_data(cfg, "train"), load_data(cfg, "val")
    model, cfg = build_model(cfg, train)
    tok = model.tokenizer
    log.info(f"[{cfg.get('name')}] params={model.num_params() / 1e6:.2f}M vocab={tok.vocab_size if tok else '-'} "
          f"heads={list(model.heads)} frame={model.frame_sec * 1000:.0f}ms")
    tr = Trainer(model, cfg)
    tr.fit(train, val, out=out)
    if out:  # save first: an evaluation error must never throw away the trained weights
        save_model(model, out)
        log.info("%s %s", "saved", out)
    try:
        metrics = tr.evaluate(val)
    except Exception as e:  # noqa: BLE001 - reported, the run's weights are already on disk
        traceback.print_exc()
        metrics = {"eval_error": f"{type(e).__name__}: {e}"}
    log.info("%s %s", "final", json.dumps(metrics))
    return model.cpu(), metrics


# --------------------------------------------------------------------------- codec & speech-LLM
def train_codec(steps: int = 500, batch_size: int = 8, seconds: float = 1.28, lr: float = 1e-3,
                device=None, codec_cfg: dict | None = None, gan: bool = False, log_every: int = 50):
    """Train an FSQ codec on synthetic tone-language speech.

    type "wave" (default): waveform AudioCodec, multi-res mel + L1 (+ optional GAN). Needs long
    training (NVIDIA-class codecs train 100k+ steps). type "mel": MelCodec, L1 on log-mel; trains in
    minutes and gives informative 12.5 fps tokens."""
    from .codec import AudioCodec, MelCodec, MultiResMelLoss, PeriodDiscriminator
    from .data import ToneLanguage
    device = device or pick_device()
    cc = dict(codec_cfg or {})
    if cc.pop("type", "wave") == "mel":
        return _train_mel_codec(MelCodec(**cc).to(device), steps, batch_size, lr, device, log_every)
    codec = AudioCodec(**cc).to(device)
    mel_loss = MultiResMelLoss(codec.sample_rate).to(device)
    opt = torch.optim.AdamW(codec.parameters(), lr=lr, betas=(0.8, 0.99))
    disc = PeriodDiscriminator().to(device) if gan else None
    opt_d = torch.optim.AdamW(disc.parameters(), lr=lr, betas=(0.8, 0.99)) if gan else None
    lang, rng, n = ToneLanguage(), random.Random(0), int(seconds * codec.sample_rate)
    hist = []
    for step in range(1, steps + 1):
        xs = []
        for _ in range(batch_size):
            a = lang.example(rng)["audio"]
            a = np.pad(a, (0, max(0, n - len(a))))[:n]
            xs.append(torch.from_numpy(a))
        x = torch.stack(xs).to(device)
        y, _ = codec(x)
        loss = mel_loss(y, x) + F_l1(y, x)
        if gan:
            fake, ffeat = disc(y)
            _, rfeat = disc(x)
            loss = loss + sum(((1 - f) ** 2).mean() for f in fake) + sum(
                (a - b.detach()).abs().mean() for a, b in zip(ffeat, rfeat))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if gan:
            real, _ = disc(x)
            fake, _ = disc(y.detach())
            ld = sum(((1 - r) ** 2).mean() + (f ** 2).mean() for r, f in zip(real, fake))
            opt_d.zero_grad()
            ld.backward()
            opt_d.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            log.info(f"codec step={step} loss={hist[-1]:.4f}")
    return codec.cpu().eval(), hist


def _train_mel_codec(codec, steps, batch_size, lr, device, log_every):
    from .data import ToneLanguage
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    lang, rng, hist = ToneLanguage(), random.Random(0), []
    for step in range(1, steps + 1):
        exs = [lang.example(rng)["audio"] for _ in range(batch_size)]
        n = max(map(len, exs))
        x = torch.tensor(np.stack([np.pad(a, (0, n - len(a))) for a in exs]), device=device)
        m_hat, m, _ = codec(x)
        loss = F_l1(m_hat, m)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            log.info(f"melcodec step={step} l1={hist[-1]:.4f}")
    return codec.cpu().eval(), hist


def F_l1(a, b):
    return (a - b).abs().mean()


def train_speech_llm(steps: int = 300, batch_size: int = 16, lr: float = 1e-3, device=None,
                     encoder_cfg: dict | None = None, log_every: int = 50, encoder_from: str | None = None,
                     freeze_llm: bool = False):
    """SALM recipe on synthetic data: frozen causal LM (pre-trained here on the transcripts as
    'text-only LM pretraining'), then encoder + projector trained to make the LM transcribe.

    ``encoder_from``: a trained .afm whose encoder initializes the speech encoder, as Canary-Qwen
    starts from canary-1b-flash. From a random encoder the frozen LM gives too weak a signal.
    ``freeze_llm=False`` lets the tiny LM adapt: the stand-in for the LoRA adapters Canary-Qwen trains."""
    from .data import ToneLanguage
    from .speech_llm import SpeechLLM, TinyCausalLM
    device = device or pick_device()
    lang, rng = ToneLanguage(), random.Random(0)
    data = [lang.example(rng) for _ in range(1500)]
    tok = train_tokenizer("char", [ex["text"] for ex in data], specials=prompt_specials(["en"]))
    eos = tok.token_id("<eos>")
    llm = TinyCausalLM(tok.vocab_size, 128, 2, 4).to(device)
    # stage 0: text-only LM pretraining (stand-in for a real pretrained LLM such as Qwen3)
    o = torch.optim.AdamW(llm.parameters(), lr=2e-3)
    for _ in range(150):
        seqs = [tok.encode(ex["text"]) + [eos] for ex in rng.sample(data, 32)]
        L = max(map(len, seqs))
        ids = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=device)
        logits = llm(inputs_embeds=llm.emb(ids)).logits
        l = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], ignore_index=0)
        o.zero_grad(); l.backward(); o.step()
    enc = encoder_cfg or dict(d_model=144, n_layers=4, n_heads=4, subsampling_channels=64)
    pre = None
    if encoder_from:
        src = load_model(encoder_from)
        enc = {k: v for k, v in src.cfg["encoder"].items()}
        pre = dict(src.cfg.get("preprocessor", {}))
    model = SpeechLLM(enc, llm, tok, freeze_llm=freeze_llm, preprocessor=pre).to(device)
    if encoder_from:
        model.encoder.load_state_dict(src.encoder.state_dict())
    prompt = tok.encode("Transcribe: <|audio|>")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    col = Collate(None)
    hist = []
    for step in range(1, steps + 1):
        exs = rng.sample(data, batch_size)
        b = to_device(col(exs), device)
        loss = model(b["audio"], b["audio_len"], [prompt] * batch_size,
                     [tok.encode(ex["text"]) + [eos] for ex in exs])
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            log.info(f"salm step={step} loss={hist[-1]:.4f}")
    model.eval()
    val = [lang.example(random.Random(99 + i)) for i in range(16)]
    hyps = []
    for ex in val:
        a = torch.from_numpy(ex["audio"])[None].to(device)
        hyps.append(tok.decode(model.generate(a, torch.tensor([a.shape[1]], device=device), prompt, 40)))
    return model, hist, wer([ex["text"] for ex in val], hyps)
