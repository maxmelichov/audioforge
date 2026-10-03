"""Target-speaker VAD (TS-VAD / personal VAD) head on the shared encoder.

Per frame it outputs two independent probabilities: P(target speaking) and P(someone else speaking), so overlap
(both) is represented. The target is given by an enrollment embedding e (a unit-norm voice print, e.g. the block-4
relational speaker head's 192-d embedding of a few seconds of the user's speech, or TitaNet-L's); without one
(``e = None``) the head falls back to a learned "no enrollment" vector and was trained to behave as a plain VAD
there (target = any speech, other = 0: PVAD 2.0's enrollment-less training).

Architecture (FiLM on hidden features beats input concatenation; PVAD 2.0 / Bovbjerg 2025):

    x (B,T,D) encoder frames (one block, e.g. block 4 = the speaker head's tap)
    h = SiLU(LN(W x))                                    per-frame projection to ``hidden``
    h = h * (1 + gamma(e)) + beta(e)                     FiLM from the enrollment
    h = h + w_s * cos(q(x_t), k(e))                      optional score pre-net (frame-level similarity), ``prenet``
    h = GRU(h)                                           causal: frame t sees frames <= t only
    logits = W_o h   -> (B,T,2) = [target, other]

Streaming: ``init_stream`` / ``step`` run the same GRU frame block by frame block (bit-equal to ``forward`` on the
concatenation, tests/test_tsvad.py). Cost: one Linear + GRU-``hidden`` step per 80 ms frame; the FiLM vectors are
computed once per enrollment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asr import Head


class TSVADHead(Head):
    key = "tsvad_targets"  # (B,T,2) float labels [target, other]; batch["tsvad_enroll"] (B,E), batch["tsvad_has"] (B,)

    # emb_dim 192 = the speaker head's print size. hidden 128: measured on the 115M against 192
    # (scripts/sweep_capacity.py, 2026-09-28); on the 0.6B (assets/tsvad_0p6b.pt) placeholder: never swept.
    # prenet_dim 64: placeholder: never swept
    def __init__(self, d_model: int, emb_dim: int = 192, hidden: int = 128, prenet: bool = True, prenet_dim: int = 64,
                 dropout: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.emb_dim, self.hidden, self.use_prenet, self.pos_weight = int(emb_dim), int(hidden), bool(prenet), float(pos_weight)
        self.proj = nn.Linear(d_model, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.null = nn.Parameter(torch.zeros(emb_dim))  # the "no enrollment" vector (learned)
        self.film = nn.Linear(emb_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # starts as identity modulation
        if self.use_prenet:
            self.q = nn.Linear(d_model, prenet_dim, bias=False)
            self.k = nn.Linear(emb_dim, prenet_dim, bias=False)
            self.score_in = nn.Linear(1, hidden)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden, hidden, 1, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    # ------------------------------------------------------------------ enrollment
    def cond(self, e: torch.Tensor | None, B: int, device=None, has: torch.Tensor | None = None) -> torch.Tensor:
        """(B, E) conditioning vectors: the unit-norm enrollment where ``has`` (default: all when e is given), the
        learned null vector elsewhere."""
        null = self.null[None].expand(B, -1)
        if e is None:
            return null
        e = F.normalize(e.float().reshape(B, -1), dim=-1).to(null.dtype)
        if has is None:
            return e
        return torch.where(has.reshape(B, 1).bool(), e, null)

    def _pre(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(self.proj(x)))
        g, b = self.film(c).chunk(2, -1)
        h = h * (1 + g[:, None]) + b[:, None]
        if self.use_prenet:
            s = F.cosine_similarity(self.q(x), self.k(c)[:, None], dim=-1)  # (B,T)
            h = h + self.score_in(s[..., None])
        return self.drop(h)

    # ------------------------------------------------------------------ offline
    def forward(self, enc, enc_len=None, e=None, has=None):
        """enc (B,T,D), e (B,E) or None -> logits (B,T,2) [target, other]."""
        c = self.cond(e, enc.shape[0], enc.device, has)
        h, _ = self.rnn(self._pre(enc, c))
        return self.out(h)

    def loss(self, enc, enc_len, batch):
        y = batch[self.key].float()
        T = min(enc.shape[1], y.shape[1])
        logits = self.forward(enc[:, :T], None, batch.get("tsvad_enroll"), batch.get("tsvad_has"))
        y = y[:, :T]
        valid = (torch.arange(T, device=enc.device)[None] < enc_len[:, None]).float()[..., None]
        pw = torch.tensor([self.pos_weight, 1.0], device=enc.device)
        l = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none", pos_weight=pw)
        return (l * valid).sum() / (valid.sum() * 2).clamp(min=1)

    @torch.no_grad()
    def decode(self, enc, enc_len=None, e=None, has=None, **_):
        """-> probabilities (B,T,2) [P(target), P(other)]."""
        return torch.sigmoid(self.forward(enc, enc_len, e, has))

    # ------------------------------------------------------------------ streaming
    def init_stream(self, e=None, batch: int = 1, device=None) -> dict:
        """State for ``step``: the conditioning vector (computed once per enrollment) and the GRU state."""
        dev = device or self.null.device
        c = self.cond(None if e is None else torch.as_tensor(e, device=dev).reshape(batch, -1), batch, dev)
        return {"c": c, "h": None}

    @torch.no_grad()
    def step(self, frames: torch.Tensor, state: dict) -> torch.Tensor:
        """frames (B,t,D) -> P (B,t,2); updates ``state`` in place (causal, bit-equal to ``decode`` on the concatenation)."""
        y, state["h"] = self.rnn(self._pre(frames, state["c"]), state["h"])
        return torch.sigmoid(self.out(y))

    def set_enrollment(self, state: dict, e=None):
        """Swap the voice print mid-stream (the GRU state is kept)."""
        B = state["c"].shape[0]
        state["c"] = self.cond(None if e is None else torch.as_tensor(e, device=state["c"].device).reshape(B, -1),
                               B, state["c"].device)
        return state
