"""Segment end-of-turn classifier (turn head v5, research/TURN_V5.md), smart-turn style on our frozen encoder.

When the served VAD goes quiet, the classifier reads the last ``win`` encoder frames (<= 8 s at 80 ms) of one block
of the streaming encoder (already computed by the ASR pass: no extra encoder work), the served VAD / TS-VAD tracks of
those frames, optional prosody, and the RNNT tokens decoded so far, and returns P(the user's turn is complete).

  frames  -> Linear(d) + learned position-from-the-end -> n_layers bidirectional Transformer inside the window
          -> attention pooling (smart-turn v3's pool: Linear-Tanh-Linear weights over time)
  tokens  -> embedding + position-from-the-end -> text_layers Transformer -> attention pooling -> text vector
             (+ log frames since the last token, log tokens in the window)
  [audio, text] -> smart-turn v3's classifier MLP (d -> 256 -> LN -> GELU -> 64 -> GELU -> 1) = main logit;
  aux logits: smart-turn v3.2's P(complete) (distillation) from the joint vector, LM completeness from the text vector.

``SegTurnStream`` keeps the ring buffers for the served path (streams.py) and runs the classifier on a frame."""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class AttnPool(nn.Module):
    def __init__(self, d: int, h: int = 128):  # placeholder: never swept (h)
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x, mask):  # x (B,T,d), mask (B,T) True = valid
        a = self.w(x).squeeze(-1).masked_fill(~mask, -1e4)
        return (a.softmax(-1).unsqueeze(-1) * x).sum(1)


def _enc(d, heads, layers, ff, dropout):
    lay = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
    return nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)


class SegTurn(nn.Module):
    key = "turn_seg"  # a batch field no training set carries: the Trainer skips this head (it is trained offline)

    # n_layers 2: measured on the 115M (0 / 2 / 4, plans/sweeps/turn_seg_115m_2026-09-30.md); the 0.6B's turn_seg and
    # turn_seg_a copy it: placeholder: never swept there. d 256, heads 4, ff 1024, d_text 128, text_layers 2 and
    # the classifier MLP widths (256, 64): placeholder: never swept. d_in, vocab, win, max_tok are fixed by the
    # encoder, the tokenizer and the window, not sizes to sweep
    def __init__(self, d_in: int = 512, n_extra: int = 3, d: int = 256, n_layers: int = 2, heads: int = 4,
                 ff: int = 1024, win: int = 100, use_text: bool = True, vocab: int = 1024, d_text: int = 128,
                 text_layers: int = 2, max_tok: int = 48, dropout: float = 0.1, block: int = 17, n_pros: int = 0):
        super().__init__()
        self.cfg = dict(d_in=d_in, n_extra=n_extra, d=d, n_layers=n_layers, heads=heads, ff=ff, win=win,
                        use_text=use_text, vocab=vocab, d_text=d_text, text_layers=text_layers, max_tok=max_tok,
                        dropout=dropout, block=block, n_pros=n_pros)
        self.win, self.max_tok, self.use_text, self.block, self.n_pros = win, max_tok, use_text, block, n_pros
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.extra = nn.Linear(n_extra + n_pros, d)
        self.pos = nn.Embedding(win, d)
        self.enc = _enc(d, heads, n_layers, ff, dropout) if n_layers else None
        self.norm = nn.LayerNorm(d)
        self.pool = AttnPool(d)
        dj = d
        if use_text:
            self.tok = nn.Embedding(vocab + 1, d_text, padding_idx=vocab)
            self.tpos = nn.Embedding(max_tok, d_text)
            self.tenc = _enc(d_text, 4, text_layers, 4 * d_text, dropout)
            self.tnorm = nn.LayerNorm(d_text)
            self.tpool = AttnPool(d_text, 64)
            self.tfeat = nn.Linear(2, d_text)
            self.compl = nn.Linear(d_text, 1)
            dj += d_text
        self.cls = nn.Sequential(nn.Linear(dj, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.st = nn.Sequential(nn.Linear(dj, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("temp", torch.tensor([1.0, 0.0]))  # calibration: logit / T + b

    def forward(self, x, xm, extra, tok=None, tm=None, tfeat=None):
        """x (B,W,d_in) right-aligned window (last = the current frame), xm (B,W) valid, extra (B,W,n_extra+n_pros),
        tok (B,U) right-aligned token ids (pad = vocab), tm (B,U) valid, tfeat (B,2) -> dict of logits (B,)."""
        B, Wn, _ = x.shape
        pos = torch.arange(Wn - 1, -1, -1, device=x.device)
        h = self.inp(x) + self.extra(extra) + self.pos(pos)[None]
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=~xm)
        a = self.pool(self.norm(h), xm)
        out = {}
        if self.use_text:
            U = tok.shape[1]
            tp = torch.arange(U - 1, -1, -1, device=x.device).clamp(max=self.max_tok - 1)
            t = self.tok(tok) + self.tpos(tp)[None]
            any_t = tm.any(1)
            tm2 = tm.clone()
            tm2[~any_t, -1] = True  # an empty prefix reads the pad slot
            t = self.tenc(t, src_key_padding_mask=~tm2)
            tv = self.tpool(self.tnorm(t), tm2) * any_t[:, None] + self.tfeat(tfeat)
            out["compl"] = self.compl(tv).squeeze(-1)
            a = torch.cat([a, tv], -1)
        out["main"] = self.cls(a).squeeze(-1)
        out["st"] = self.st(a).squeeze(-1)
        return out

    def decode(self, enc, enc_len=None, **_):
        """Not a per-frame head of the encoder: it runs on the session's window (SegTurnStream.prob)."""
        return None

    def prob(self, *args):
        z = self.forward(*args)["main"]
        return (z / self.temp[0] + self.temp[1]).sigmoid()


def text_feats(n_now: int, n_win0: int, since_tok: int) -> list:
    return [math.log1p(max(0, since_tok)) / 5, math.log1p(max(0, n_now - n_win0)) / 4]


class SegTurnStream:
    """Served path. ``push`` one frame at a time in frame order (the block features of the streaming encoder, the
    served VAD, P(user) / P(other) (0 / 0 without an enrolled print), the tokens decoded at frames <= v, and the
    frame's prosody when the model reads it); ``prob(v, tokens)`` classifies the window ending at any recent frame v.
    The window, the tokens and the text features equal the offline ones (``scripts/research/turn_v5.py seg_probs``)."""

    KEEP = 256  # frames kept (>= win + the policy's lag behind the newest frame)

    def __init__(self, model: SegTurn, device="cpu"):
        self.m = model.eval()
        self.dev = device
        self.x, self.ex, self.ntok, self.lastv = {}, {}, {}, {}
        self.v = -1
        self.last_tok_v = -1
        self.n_prev = 0
        self.ms = 0.0
        self.calls = 0

    def push(self, feat, vad: float, pu: float, po: float, n_tokens: int, pros=None) -> int:
        self.v += 1
        v = self.v
        self.x[v] = np.asarray(feat, np.float32).reshape(-1)
        e = [vad, pu, po] + ([] if pros is None else list(np.asarray(pros, np.float32)))
        self.ex[v] = np.asarray(e, np.float32)
        if n_tokens > self.n_prev:
            self.last_tok_v = v
        self.n_prev = int(n_tokens)
        self.ntok[v] = int(n_tokens)
        self.lastv[v] = self.last_tok_v
        old = v - self.KEEP
        for d in (self.x, self.ex, self.ntok, self.lastv):
            d.pop(old, None)
        return v

    @torch.no_grad()
    def prob(self, v: int, tokens: list) -> float:
        """P(complete) of the window ending at frame v (<= the newest pushed frame); ``tokens`` = the decoded ids."""
        import time
        t0 = time.perf_counter()
        m = self.m
        W = m.win
        a0 = max(0, v - W + 1)
        x = torch.as_tensor(np.stack([self.x[u] for u in range(a0, v + 1)]), device=self.dev)[None]
        ex = torch.as_tensor(np.stack([self.ex[u] for u in range(a0, v + 1)]), device=self.dev)[None]
        xm = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.dev)
        args = [x, xm, ex]
        if m.use_text:
            n_now = self.ntok[v]
            toks = [int(t) for t in list(tokens[:n_now])[-m.max_tok:]]
            U = max(1, len(toks))
            tok = torch.full((1, U), m.cfg["vocab"], dtype=torch.long, device=self.dev)
            tm = torch.zeros(1, U, dtype=torch.bool, device=self.dev)
            if toks:
                tok[0, U - len(toks):] = torch.tensor(toks, device=self.dev)
                tm[0, U - len(toks):] = True
            n0 = self.ntok[v - W] if v - W >= 0 else 0
            lt = self.lastv[v]
            since = v - lt if lt >= 0 else v + 1
            tf = torch.tensor([text_feats(n_now, n0, since)], dtype=torch.float32, device=self.dev)
            args += [tok, tm, tf]
        p = float(m.prob(*args)[0])
        self.ms += (time.perf_counter() - t0) * 1000
        self.calls += 1
        return p


def load_seg_turn(path, device="cpu") -> SegTurn:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SegTurn(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()
