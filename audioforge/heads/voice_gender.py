"""Perceived voice gender (optional head, research/VOICE_GENDER.md): a running two-class posterior, female voice /
male voice, read from the speaker head's encoder tap (115M block 4, 0.6B block 5).

What it measures: a perceived vocal characteristic of the audio (the label the FLEURS and LibriSpeech annotators gave
the recording), not a person's gender identity. It can be wrong for any individual voice and must not be used to
decide anything about a person.

Architecture (lifted, not invented):
- attentive statistics pooling, the decoder of ECAPA-TDNN (Desplanques et al. 2020), TitaNet (Koluguri et al. 2022)
  and AmberNet (Jia et al. 2023): attention weights per frame and channel, then the weighted mean and standard
  deviation over time;
- a linear classifier on [mean, std] (AmberNet's decoder ends in one linear layer);
- the pooling is made causal the way this project's LID head is (heads/audio.py ``LanguageHead``): the weights depend
  on the frame alone, ``w_t = exp(5 tanh(a(h_t) / 5))`` (bounded, so running sums need no max-subtraction), so the
  posterior after frame t is a function of running sums and streams with constant state;
- a LayerNorm + linear projection to ``hidden`` channels in front of the pooling (the only size that is swept).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asr import Head

LABELS = ("female", "male")  # the datasets' two annotated classes (FLEURS gender field, LibriSpeech SPEAKERS.TXT)


def _pad_mask(x, lens):
    return torch.arange(x.shape[1], device=x.device)[None] < lens[:, None]


class VoiceGenderHead(Head):
    """Causal attentive-statistics pooling + linear classifier. ``forward`` / ``running_logits`` (offline) and
    ``step`` (streaming, carried sums) compute the same numbers. ``batch["voice_gender"]``: class index (0 female, 1
    male); optional ``batch["keep"]`` (B, T) bool: frames that are pooled (VAD-gated speech, as served).
    ``loss`` = cross-entropy of the running posterior averaged over the pooled frames from ``min_frames`` on (an
    anytime classifier) plus the cross-entropy after the last frame."""

    key = "voice_gender"
    A_MAX = 5.0

    def __init__(self, d_model: int, hidden: int = 64, att_hidden: int = 64, num_classes: int = 2,
                 labels: list | None = None, min_frames: int = 6, dropout: float = 0.1):
        super().__init__()
        # hidden: measured, plans/sweeps/voice_gender_2026-10-03.md (shipped: 32 on the 115M, 16 on the 0.6B, carried
        # in the head files' cfg; this default is not a measured size). att_hidden 64: placeholder: never swept
        self.labels = list(labels) if labels else list(LABELS)
        assert len(self.labels) == num_classes
        self.min_frames = int(min_frames)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.Dropout(dropout))
        self.att = nn.Sequential(nn.Linear(hidden, att_hidden), nn.Tanh(), nn.Linear(att_hidden, hidden))
        self.cls = nn.Linear(2 * hidden, num_classes)

    @property
    def hidden(self) -> int:
        return self.cls.in_features // 2

    def terms(self, x):
        """Per-frame pooling terms (w, w h, w h^2), each (B, T, hidden), float32."""
        h = self.proj(x).float()
        w = torch.exp(self.A_MAX * torch.tanh(self.att(h) / self.A_MAX))
        return w, w * h, w * h * h

    def classify(self, sw, swx, swxx):
        sw = sw.clamp(min=1e-6)
        mu = swx / sw
        sd = (swxx / sw - mu * mu).clamp(min=1e-6).sqrt()
        return self.cls(torch.cat([mu, sd], -1))

    def _mask(self, enc, enc_len, keep=None):
        m = _pad_mask(enc, enc_len)
        return (m & keep if keep is not None else m)[..., None].float()

    def running_logits(self, enc, enc_len, keep=None):
        """(B, T, C) logits of the posterior after each frame (frames not pooled hold the previous posterior)."""
        w, wx, wxx = self.terms(enc)
        m = self._mask(enc, enc_len, keep)
        return self.classify((w * m).cumsum(1), (wx * m).cumsum(1), (wxx * m).cumsum(1))

    def forward(self, enc, enc_len, keep=None):
        """(B, C) logits after the last frame (the whole-utterance decision)."""
        w, wx, wxx = self.terms(enc)
        m = self._mask(enc, enc_len, keep)
        return self.classify((w * m).sum(1), (wx * m).sum(1), (wxx * m).sum(1))

    def init_stream(self, batch: int = 1, device=None):
        z = torch.zeros(batch, self.hidden, device=device)
        return [z, z.clone(), z.clone()]

    def step(self, x, state, keep=None):
        """Streaming update with a chunk x (B, n, D); ``state`` (``init_stream``) is updated in place.
        ``keep`` (B, n) bool: frames with False are not pooled. -> (B, n, C) logits after each frame."""
        w, wx, wxx = self.terms(x)
        if keep is not None:
            k = keep.to(w.dtype)[..., None]
            w, wx, wxx = w * k, wx * k, wxx * k
        cw, cx, cxx = (state[0][:, None] + w.cumsum(1), state[1][:, None] + wx.cumsum(1),
                       state[2][:, None] + wxx.cumsum(1))
        state[0], state[1], state[2] = cw[:, -1], cx[:, -1], cxx[:, -1]
        return self.classify(cw, cx, cxx)

    def loss(self, enc, enc_len, batch):
        y = batch[self.key].long()
        keep = batch.get("keep")
        z = self.running_logits(enc, enc_len, keep)
        B, T, C = z.shape
        m = self._mask(enc, enc_len, keep)[..., 0]
        seen = m.cumsum(1)  # pooled frames so far
        use = (m > 0) & (seen >= self.min_frames)
        ce = F.cross_entropy(z.reshape(-1, C), y[:, None].expand(B, T).reshape(-1), reduction="none").view(B, T)
        anytime = (ce * use).sum(1) / use.sum(1).clamp(min=1)
        last = F.cross_entropy(self.forward(enc, enc_len, keep), y, reduction="none")
        return 0.5 * (anytime.mean() + last.mean())

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return self(enc, enc_len).softmax(-1)
