"""Optional perceived voice-gender head (``audioforge.serve --voice-gender``): head files,
attaching a head to a served model, and the per-session stream.

What it reports: probabilities for "female voice" / "male voice", the two classes the training data (FLEURS,
LibriSpeech) annotate. It is a perceived vocal characteristic estimated from audio, not a person's gender identity,
and it can be wrong for any individual. Off by default.

A head file is a small torch archive {format, name, cfg, state_dict, labels, from_layers, encoder, meta}, attached at
load time to a served model with the same encoder (like the LID head files, ``audioforge.lid``).
"""
from __future__ import annotations

import time

import torch

NAME = "voice_gender"
FORMAT = "audioforge-voice-gender-head-v1"
HEAD_FILES = {512: "voice_gender_115m.pt", 1024: "voice_gender_0p6b.pt"}  # by encoder width (115M / 0.6B core)


def resolve_head(value, d_model: int = 512, models_dir=None) -> str:
    """``head`` -> the shipped file for the core with encoder width ``d_model`` (<models dir>/, else <repo>/assets/);
    any other value (a head-file path) is returned unchanged."""
    if value != "head":
        return value
    from .hub import models_dir as _models_dir
    from .paths import ROOT
    f = HEAD_FILES[int(d_model)]
    cands = [_models_dir(models_dir) / f, ROOT / "assets" / f]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"--voice-gender head: {f} not found in {' or '.join(str(c.parent) for c in cands)}")


def save_head_state(state_dict, cfg: dict, from_layers, d_model: int, path, n_layers: int | None = None,
                    meta: dict | None = None):
    blob = {"format": FORMAT, "name": NAME, "cfg": dict(cfg), "state_dict": {k: v.cpu() for k, v in state_dict.items()},
            "labels": list(cfg["labels"]), "from_layers": list(from_layers),
            "encoder": {"d_model": int(d_model), "n_layers": n_layers}, "meta": meta or {}}
    torch.save(blob, path)


def load_head(path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT:
        raise ValueError(f"{path}: not a voice-gender head file")
    return blob


def build_from_file(path):
    """-> (VoiceGenderHead with the file's weights, eval mode, on CPU; the file's dict)."""
    from .model import build_head
    blob = load_head(path)
    head = build_head(blob["cfg"], blob["encoder"]["d_model"])
    head.load_state_dict(blob["state_dict"], strict=True)
    return head.eval(), blob


def attach_head(model, path, name: str = NAME) -> str:
    """Add the head stored at ``path`` to ``model`` (heads / head_cfg / layer_tap); -> its name."""
    head, blob = build_from_file(path)
    enc = blob["encoder"]
    assert enc["d_model"] == model.encoder.d_model, f"voice-gender head trained on a different encoder ({enc})"
    assert max(blob["from_layers"]) < len(model.encoder.layers), f"tap {blob['from_layers']} beyond the encoder"
    model.heads[name] = head.requires_grad_(False).to(next(model.parameters()).device)
    model.head_cfg[name] = dict(blob["cfg"])
    model.layer_tap[name] = list(blob["from_layers"])
    return name


class VoiceGenderStream:
    """Per-session posterior on the served encoder's per-layer chunk outputs (``hid`` of stream_step). Frames with
    VAD <= ``vad_gate`` are not pooled. Two readings: ``session()`` (every pooled frame of the session) and
    ``segment(f0, f1)`` (the pooled frames of ASR frames [f0, f1), e.g. one final's span; frames before f1 are then
    released). Each reading is {"female": p, "male": p, "speech_ms": pooled speech} or None without speech.

    Lazy by design (the latency gate): ``feed`` only keeps a reference to the chunk's tap frames and the VAD flags,
    so a 160 ms chunk costs no device work; the head runs once per reading, batched over the frames it covers
    (``segment`` once per final, ``session`` once per stats message). The session sums fold the frames in as they are
    read or released, so every frame is pooled exactly once. At most ``MAX_FRAMES`` frames are held."""

    MAX_FRAMES = 7500  # 10 min of 80 ms frames

    def __init__(self, model, name: str = NAME, vad_gate: float = 0.5, frame_ms: float = 80.0):
        self.m, self.name, self.head = model, name, model.heads[name]
        self.labels = list(self.head.labels)
        self.vad_gate, self.frame_ms = vad_gate, float(frame_ms)
        self.chunks: list = []  # (first ASR frame index, tap frames (1, n, D), keep flags [n])
        self.n_frames = 0  # ASR frames fed
        self.folded = 0  # frames [0, folded) are in the session sums
        self.sums, self.n_pooled = None, 0
        self.ms = 0.0

    @torch.no_grad()
    def feed(self, enc, hid, vad) -> None:
        t0 = time.perf_counter()
        x = self.m.head_input(self.name, enc, hid)
        self.chunks.append((self.n_frames, x, [v > self.vad_gate for v in vad]))
        self.n_frames += x.shape[1]
        if self.n_frames - self.chunks[0][0] > self.MAX_FRAMES:
            self._release(self.n_frames - self.MAX_FRAMES)
        self.ms += (time.perf_counter() - t0) * 1000

    def _sums(self, a: int, b: int):
        """Pooled (sum w, sum w h, sum w h^2) over the kept frames of [a, b) still held, and their count."""
        xs, ks = [], []
        for s, x, keep in self.chunks:
            lo, hi = max(a - s, 0), min(b - s, x.shape[1])
            if lo < hi:
                xs.append(x[:, lo:hi])
                ks += keep[lo:hi]
        n = sum(ks)
        if n == 0:
            return None, 0
        w, wx, wxx = self.head.terms(torch.cat(xs, 1))
        k = torch.tensor(ks, dtype=w.dtype, device=w.device)[None, :, None]
        return [(t * k).sum(1)[0] for t in (w, wx, wxx)], n

    def _fold(self, upto: int):
        if upto <= self.folded:
            return
        s, n = self._sums(self.folded, upto)
        if n:
            self.sums = s if self.sums is None else [a + b for a, b in zip(self.sums, s)]
            self.n_pooled += n
        self.folded = upto

    def _release(self, upto: int):
        """Fold frames < ``upto`` into the session sums and drop the chunks that end at or before it."""
        self._fold(upto)
        self.chunks = [c for c in self.chunks if c[0] + c[1].shape[1] > upto]

    def _read(self, sums, n):
        if n == 0:
            return None
        p = self.head.classify(*(s.reshape(1, -1) for s in sums)).softmax(-1)[0].tolist()
        return {**{lab: round(float(v), 4) for lab, v in zip(self.labels, p)}, "speech_ms": round(n * self.frame_ms)}

    @torch.no_grad()
    def session(self):
        t0 = time.perf_counter()
        self._fold(self.n_frames)
        out = self._read(self.sums, self.n_pooled)
        self.ms += (time.perf_counter() - t0) * 1000
        return out

    @torch.no_grad()
    def segment(self, f0: int, f1: int):
        t0 = time.perf_counter()
        out = self._read(*self._sums(f0, f1))
        self._release(f1)
        self.ms += (time.perf_counter() - t0) * 1000
        return out
