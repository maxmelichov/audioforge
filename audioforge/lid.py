"""Spoken language ID on the shared encoder: head files, attaching a head to a served model, and the serving rule
(research/archive/LID.md; ``audioforge.serve --lid``).

A LID head file (``save_head``) is a small torch archive {cfg, state_dict, labels, from_layers, meta}: the head is
trained on a frozen served model, so only its own tensors are stored and it is attached at load time
(``attach_head``) to any model with the same encoder (the served runs/stage1_served.afm), without rewriting the
443 MB model archive.
"""
from __future__ import annotations

import numpy as np
import torch

LID_NAME = "lid"
HEAD_FILE = "lid_distill.pt"  # the served head of research/archive/LID.md "Fix pass, 2026-09-29" (``serve --lid head``)
HEAD_MAX_MS = 3000.0  # its pre-registered rule: announce at the threshold, or the top language after 3 s of speech


def resolve_head(value, models_dir=None) -> str:
    """``--lid head`` -> the shipped head file (<models dir>/lid_distill.pt, else <repo>/assets/, else <repo>/runs/); any
    other value (a head-file path) is returned unchanged."""
    if value != "head":
        return value
    from .hub import models_dir as _models_dir
    from .paths import ROOT
    cands = [_models_dir(models_dir) / HEAD_FILE, ROOT / "assets" / HEAD_FILE, ROOT / "runs" / HEAD_FILE]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"--lid head: {HEAD_FILE} not found in {' or '.join(str(c.parent) for c in cands)}")


def save_head(model, name: str, path, meta: dict | None = None):
    cfg = dict(model.head_cfg[name])
    blob = {"format": "audioforge-lid-head-v1", "name": name, "cfg": cfg,
            "state_dict": {k: v.detach().cpu() for k, v in model.heads[name].state_dict().items()},
            "labels": list(cfg.get("labels") or model.heads[name].labels),
            "from_layers": model.layer_tap.get(name),
            "layer_mix": (model.layer_mix[name].detach().cpu() if name in model.layer_mix else None),
            "encoder": {k: model.cfg["encoder"].get(k) for k in ("d_model", "n_layers", "att_context_size")},
            "meta": meta or {}}
    torch.save(blob, path)


def load_head(path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != "audioforge-lid-head-v1":
        raise ValueError(f"{path}: not a LID head file")
    return blob


def attach_head(model, path, name: str | None = None) -> str:
    """Add the head stored at ``path`` to ``model`` (heads / head_cfg / layer_tap / layer_mix); -> its name."""
    from .model import build_head
    blob = load_head(path)
    name = name or blob["name"]
    enc = blob["encoder"]
    assert enc["d_model"] == model.encoder.d_model and enc["n_layers"] == len(model.encoder.layers), \
        f"LID head trained on a different encoder ({enc})"
    cfg = dict(blob["cfg"])
    head = build_head(cfg, model.encoder.d_model)
    head.load_state_dict(blob["state_dict"], strict=True)
    dev = next(model.parameters()).device
    model.heads[name] = head.to(dev).eval()
    model.head_cfg[name] = cfg
    if blob["from_layers"] is not None:
        model.layer_tap[name] = list(blob["from_layers"])
    if blob["layer_mix"] is not None:
        model.layer_mix[name] = torch.nn.Parameter(blob["layer_mix"].to(dev), requires_grad=False)
    return name


class LangDecider:
    """Serving rule: announce the top language once it is confident, and again whenever the confident top changes.
    ``update(p, t)`` is called once per pooled (speech) frame with the running posterior p over ``labels``; it returns
    {"type": "language", "t", "language", "confidence"} when (a) at least ``min_ms`` of pooled audio has been seen,
    (b) max p >= ``threshold`` and (c) the top language differs from the last announced one; else None.
    ``max_ms`` (None = off): if nothing has been announced after ``max_ms`` of pooled audio, the top language is
    announced anyway (evidence threshold or timeout, research/archive/LID.md fix pass)."""

    def __init__(self, labels, threshold: float = 0.8, min_ms: float = 1000.0, frame_ms: float = 80.0,
                 max_ms: float | None = None):
        self.labels, self.threshold, self.min_ms, self.frame_ms = list(labels), float(threshold), float(min_ms), frame_ms
        self.max_ms = None if max_ms is None else float(max_ms)
        self.n, self.current, self.confidence = 0, None, None

    def update(self, p, t: float, speech_ms: float | None = None):
        """``speech_ms``: pooled audio so far, when the caller does not call once per frame (AmberNet backend)."""
        self.n += 1
        p = np.asarray(p, np.float64)
        k = int(p.argmax())
        seen = self.n * self.frame_ms if speech_ms is None else speech_ms
        forced = self.max_ms is not None and self.current is None and seen >= self.max_ms
        if seen < self.min_ms or (p[k] < self.threshold and not forced):
            return None
        self.confidence = round(float(p[k]), 4)
        if self.labels[k] == self.current:
            return None
        self.current = self.labels[k]
        return {"type": "language", "t": round(float(t), 3), "language": self.current, "confidence": self.confidence}


class LIDStream:
    """Per-session running posterior on the served encoder's per-layer chunk outputs (``hid`` of stream_step).
    ``feed(hid, vad, t_end)`` -> list of language events. Frames with VAD <= ``vad_gate`` are not pooled (the
    posterior holds through silence); ``half_life_s`` (of pooled speech) sets the exponential forgetting."""

    def __init__(self, model, name: str = LID_NAME, threshold: float = 0.8, min_ms: float = 1000.0,
                 vad_gate: float = 0.5, half_life_s: float | None = 30.0, max_ms: float | None = None):
        self.m, self.name = model, name
        self.head = model.heads[name]
        self.state = self.head.init_stream(1, device=next(model.parameters()).device)
        fs = model.frame_sec
        self.decay = 0.5 ** (fs / half_life_s) if half_life_s else 1.0
        self.vad_gate = vad_gate
        self.decider = LangDecider(self.head.labels, threshold, min_ms, fs * 1000, max_ms=max_ms)
        self.ms = 0.0

    @property
    def current(self):
        return self.decider.current

    @torch.no_grad()
    def feed(self, enc, hid, vad, frame_end_t) -> list[dict]:
        import time
        t0 = time.perf_counter()
        x = self.m.head_input(self.name, enc, hid)
        keep = (torch.tensor([[v > self.vad_gate for v in vad]], device=x.device) if self.vad_gate is not None
                else None)
        z = self.head.step(x, self.state, keep=keep, decay=self.decay).softmax(-1)[0]
        out = []
        for j in range(z.shape[0]):
            if keep is None or bool(keep[0, j]):
                ev = self.decider.update(z[j].cpu().numpy(), frame_end_t[j])
                if ev:
                    out.append(ev)
        self.ms += (time.perf_counter() - t0) * 1000
        return out


# the 17 FLEURS languages of research/archive/LID.md (default label set of the AmberNet backend)
DEFAULT_LANGS = ["en", "he", "ar", "ru", "es", "fr", "de", "pt", "it", "nl", "pl", "uk", "tr", "fa", "hi", "zh", "ja"]


class AmberNetLIDStream:
    """The dedicated-model backend (``serve --lid ambernet``): NVIDIA AmberNet (``nemo_import.import_ambernet``)
    re-run on the speech heard so far. The audio of every ASR frame with VAD > ``vad_gate`` is kept; AmberNet
    classifies the last ``window_s`` s of it when the pooled speech first reaches each of ``checkpoints_ms`` (those
    below ``min_ms`` are skipped) and then every ``period_ms`` of new speech; its posterior restricted (renormalised)
    to ``labels`` goes through the same ``LangDecider``. AmberNet costs 16-81 ms per call on 2 CPU threads (1-8 s)
    (research/archive/LID.md), so it is not run per 160 ms chunk: the default schedule is ~6 calls in the first 8 s of speech,
    then one per 4 s (RTF ~0.02-0.03). Same interface as ``LIDStream`` (``feed_audio`` is called by
    ``serve.ASRStream`` with the raw samples first)."""

    CHECKPOINTS_MS = (1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 8000.0)

    def __init__(self, model, labels=None, threshold: float = 0.9, min_ms: float = 1000.0, vad_gate: float = 0.5,
                 period_ms: float = 4000.0, window_s: float = 8.0, checkpoints_ms=CHECKPOINTS_MS,
                 frame_samples: int = 1280, sr: int = 16000):
        self.m, self.labels = model, list(labels or DEFAULT_LANGS)
        self.vad_gate, self.period_ms, self.window = vad_gate, float(period_ms), int(window_s * sr)
        self.checkpoints = sorted(float(c) for c in checkpoints_ms if c >= min_ms) or [float(min_ms)]
        self.fs, self.frame_ms = frame_samples, frame_samples * 1000.0 / sr
        self.decider = LangDecider(self.labels, threshold, min_ms, self.frame_ms)
        self.audio = np.zeros(0, np.float32)
        self.audio0 = 0  # absolute sample index of audio[0]
        self.speech: list[np.ndarray] = []
        self.n_speech, self.last_run = 0, 0
        self.ms, self.runs = 0.0, 0

    @property
    def current(self):
        return self.decider.current

    def feed_audio(self, x):
        self.audio = np.concatenate([self.audio, np.asarray(x, np.float32)])
        drop = max(0, len(self.audio) - 4 * self.window)
        if drop:
            self.audio, self.audio0 = self.audio[drop:], self.audio0 + drop

    def feed(self, enc, hid, vad, frame_end_t) -> list[dict]:
        import time

        from .baselines.lid import restrict
        out = []
        for v_end, p_vad in zip(frame_end_t, vad):
            if self.vad_gate is not None and p_vad <= self.vad_gate:
                continue
            v = int(round(v_end * 1000.0 / self.frame_ms)) - 1
            a, b = v * self.fs - self.audio0, (v + 1) * self.fs - self.audio0
            if a < 0 or b > len(self.audio):
                continue
            self.speech.append(self.audio[a:b])
            self.n_speech += 1
            keep = int(np.ceil(self.window / self.fs))
            if len(self.speech) > keep:
                self.speech = self.speech[-keep:]
            speech_ms = self.n_speech * self.frame_ms
            if speech_ms < self.decider.min_ms:
                continue
            due = (self.checkpoints and speech_ms >= self.checkpoints[0]) or (
                not self.checkpoints and (self.n_speech - self.last_run) * self.frame_ms >= self.period_ms)
            if not due:
                continue
            while self.checkpoints and speech_ms >= self.checkpoints[0]:
                self.checkpoints.pop(0)
            self.last_run = self.n_speech
            t0 = time.perf_counter()
            p = restrict(self.m.probs(np.concatenate(self.speech)[-self.window:]), self.labels)
            self.ms += (time.perf_counter() - t0) * 1000
            self.runs += 1
            ev = self.decider.update(p, v_end, speech_ms=speech_ms)
            if ev:
                out.append(ev)
        return out
