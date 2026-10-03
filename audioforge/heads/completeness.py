"""CompletenessHead: "is what the user has said so far a complete utterance?" on the shared encoder
(research/archive/COMPLETENESS.md; the smart-turn task, pipecat-ai/smart-turn, on our frozen FastConformer).

Two outputs from one small causal temporal model (Linear -> SiLU -> GRU over the encoder frames):

* frame logit ``z_t`` = P(the speech up to frame t is a complete utterance): strictly causal (unidirectional GRU),
  streams frame by frame (``init_stream`` / ``step``), and is what eot-bench v2 / the fusion read as a per-frame score;
* utterance logit = one decision per clip from the mean of the projected encoder frames over the last ``pool_frames``
  frames before the utterance end plus the GRU state at that frame (the smart-turn setting: one clip, one label).

Labels (datasets/smartturn.py): ``completeness`` (T,) = the clip's label on the frames at / after the utterance end
(the last speech frame + 1, energy VAD) and 0 before it, ``completeness_w`` (T,) per-frame weights (``mid_weight`` on
the pre-end frames: speech in progress is "not complete yet", a soft prior), ``utt_end_frame`` (1,) the end frame,
``complete_label`` (1,) the clip's label.
Loss = BCE(utterance logit, label) + ``frame_weight`` x weighted frame BCE. Balanced data, no pos_weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .asr import Head
from .audio import _match_len, _pad_mask


@dataclass
class CompletenessStreamState:
    h: torch.Tensor | None = None  # GRU hidden (1, B, hidden)
    buf: torch.Tensor | None = None  # (B, pool_frames, hidden) last projected inputs, oldest first
    n: int = 0  # frames fed so far


class CompletenessHead(Head):
    key = "completeness"

    # not shipped; hidden: placeholder: never swept
    def __init__(self, d_model: int, hidden: int = 96, pool_frames: int = 25, frame_weight: float = 0.5,
                 mid_weight: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.hidden, self.pool_frames = int(hidden), int(pool_frames)
        self.frame_weight, self.mid_weight = float(frame_weight), float(mid_weight)
        self.inp = nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, 1, batch_first=True)
        self.frame_out = nn.Linear(hidden, 1)
        self.utt_out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    # ------------------------------------------------------------------ core
    def project(self, enc):
        return self.inp(enc)

    def hidden_states(self, enc):
        x = self.project(enc)
        h, _ = self.rnn(x)
        return x, h

    def forward(self, enc, enc_len=None):
        """(B,T) frame logits; frame t depends on frames <= t only."""
        _, h = self.hidden_states(enc)
        return self.frame_out(h).squeeze(-1)

    def utterance_logit(self, x, h, end) -> torch.Tensor:
        """x, h (B,T,H) projected inputs / GRU states; end (B,) exclusive end frame (1..T) -> (B,) logits of
        'the utterance ending at ``end`` is complete', from the last ``pool_frames`` frames before it."""
        B, T, H = x.shape
        end = end.to(x.device).long().clamp(1, T)
        ar = torch.arange(T, device=x.device)[None]
        m = ((ar < end[:, None]) & (ar >= (end - self.pool_frames)[:, None])).to(x.dtype)  # B,T
        pooled = (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        last = h.gather(1, (end - 1)[:, None, None].expand(B, 1, H))[:, 0]
        return self.utt_out(torch.cat([pooled, last], -1)).squeeze(-1)

    def loss(self, enc, enc_len, batch):
        x, h = self.hidden_states(enc)
        z = self.frame_out(h).squeeze(-1).float()
        T = z.shape[1]
        tgt = _match_len(batch[self.key].float(), T)
        w = _match_len(batch["completeness_w"].float(), T) if "completeness_w" in batch else torch.ones_like(tgt)
        valid = _pad_mask(enc, enc_len)
        if self.key + "_len" in batch:
            valid = valid & (torch.arange(T, device=z.device)[None] < batch[self.key + "_len"][:, None])
        w = w * valid
        lf = F.binary_cross_entropy_with_logits(z, tgt, reduction="none")
        lf = (lf * w).sum() / w.sum().clamp(min=1e-6)
        end = batch["utt_end_frame"].reshape(-1) if "utt_end_frame" in batch else enc_len
        end = torch.minimum(end.to(z.device).long(), enc_len.to(z.device).long()).clamp(min=1)
        if "complete_label" in batch:  # the clip's label (datasets/smartturn.py)
            lab = batch["complete_label"].reshape(-1).float().to(z.device)
        else:  # the first post-end frame carries it (0 when the clip ends exactly at the utterance end)
            lab = tgt.gather(1, end.clamp(max=T - 1)[:, None])[:, 0]
        lu = F.binary_cross_entropy_with_logits(self.utterance_logit(x, h, end).float(), lab)
        return lu + self.frame_weight * lf

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        """(B,T) P(complete so far) per frame."""
        return self(enc, enc_len).sigmoid()

    @torch.no_grad()
    def utterance_prob(self, enc, enc_len, end=None) -> torch.Tensor:
        """(B,) P(complete) of the utterance ending at ``end`` (default: the clip end)."""
        x, h = self.hidden_states(enc)
        end = enc_len if end is None else end
        return self.utterance_logit(x, h, torch.minimum(end.reshape(-1).long(), enc_len.long())).sigmoid()

    # ------------------------------------------------------------------ streaming
    def init_stream(self, batch_size: int = 1) -> CompletenessStreamState:
        return CompletenessStreamState()

    @torch.no_grad()
    def step(self, enc_frames, state: CompletenessStreamState | None = None):
        """Strictly causal step over new (B,n,D) encoder frames -> (frame probs (B,n), utterance prob (B,) as if the
        utterance ended after these frames); ``state`` is updated in place. Equals the offline outputs frame for frame."""
        state = state if state is not None else CompletenessStreamState()
        B, n, _ = enc_frames.shape
        x = self.project(enc_frames)
        h, state.h = self.rnn(x, state.h)
        buf = x if state.buf is None else torch.cat([state.buf, x], 1)
        state.buf = buf[:, -self.pool_frames:]
        state.n += n
        pooled = state.buf.mean(1)
        utt = self.utt_out(torch.cat([pooled, h[:, -1]], -1)).squeeze(-1).sigmoid()
        return self.frame_out(h).squeeze(-1).sigmoid(), utt

    # ------------------------------------------------------------------ evaluation hook (Trainer.evaluate)
    @torch.no_grad()
    def evaluate_model(self, model, name: str, val: list[dict]) -> dict:
        if self.key not in val[0]:
            return {}
        from ..data import Collate, to_device
        dev = next(model.parameters()).device
        ps, pf, ys = [], [], []
        for i in range(0, len(val), 8):
            part = val[i: i + 8]
            b = to_device(Collate(model.tokenizer)(part), dev)
            enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
            e = model.head_input(name, enc, hidden)
            end = torch.minimum(b["utt_end_frame"].reshape(-1).long(), elen.long()).clamp(min=1)
            ps += self.utterance_prob(e, elen, end).float().cpu().tolist()
            z = self.decode(e, elen)
            pf += z.gather(1, (end - 1).clamp(max=z.shape[1] - 1)[:, None])[:, 0].float().cpu().tolist()
            ys += [bool(ex["complete"]) for ex in part]
        return {f"{name}_utt_acc": round(accuracy(ys, ps), 4), f"{name}_utt_auc": round(auc(ys, ps), 4),
                f"{name}_frame_acc": round(accuracy(ys, pf), 4), f"{name}_frame_auc": round(auc(ys, pf), 4),
                f"{name}_n": len(ys)}


def accuracy(y, p, thr: float = 0.5) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, np.float64)
    return float(np.mean((p >= thr) == y)) if len(y) else float("nan")


def auc(y, p) -> float:
    """ROC AUC by the rank statistic (ties count half)."""
    y, p = np.asarray(y, bool), np.asarray(p, np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
