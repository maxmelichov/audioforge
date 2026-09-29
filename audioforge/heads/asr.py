"""ASR heads: CTC, RNNT, TDT and a Canary-style attention encoder-decoder (AED).

All heads share the interface
    loss(enc, enc_len, batch) -> scalar
    decode(enc, enc_len, **kw) -> list[list[int]]
so any combination can hang off one encoder (hybrid TDT+CTC, AED+CTC, ...).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..losses.transducer import _reduce, rnnt_loss, tdt_loss


class Head(nn.Module):
    key: str = "text"  # batch field the head is supervised with

    def loss(self, enc, enc_len, batch):  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- CTC
class CTCHead(Head):
    def __init__(self, d_model: int, vocab_size: int, key: str = "text"):
        super().__init__()
        self.vocab_size, self.blank, self.key = vocab_size, vocab_size, key
        self.proj = nn.Linear(d_model, vocab_size + 1)

    def forward(self, enc):
        return self.proj(enc).log_softmax(-1)

    def loss(self, enc, enc_len, batch):
        y, yl = batch[self.key], batch[self.key + "_len"]
        lp = self(enc).float().transpose(0, 1)
        return F.ctc_loss(lp, y, enc_len, yl, blank=self.blank, reduction="mean", zero_infinity=True)

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        best = self(enc).argmax(-1)
        out = []
        for b in range(best.shape[0]):
            seq, prev = [], -1
            for p in best[b, : int(enc_len[b])].tolist():
                if p != prev and p != self.blank:
                    seq.append(p)
                prev = p
            out.append(seq)
        return out


# --------------------------------------------------------------------------- transducers
class PredictionNet(nn.Module):
    """LSTM label predictor. Index ``blank`` doubles as the start-of-sequence symbol."""

    def __init__(self, vocab_size: int, hidden: int = 320, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blank = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)

    def forward(self, y, state=None):
        return self.lstm(self.embed(y), state)

    def prepend_sos(self, y):
        return F.pad(y, (1, 0), value=self.blank)


class Joint(nn.Module):
    def __init__(self, d_enc: int, d_pred: int, d_joint: int, n_out: int, dropout: float = 0.1):
        super().__init__()
        self.enc, self.pred = nn.Linear(d_enc, d_joint), nn.Linear(d_pred, d_joint)
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_joint, n_out))

    def forward(self, f, g):
        """f (B,T,De), g (B,U,Dp) -> (B,T,U,n_out)."""
        return self.out(self.enc(f)[:, :, None] + self.pred(g)[:, None])


class RNNTHead(Head):
    def __init__(self, d_model: int, vocab_size: int, pred_hidden: int = 320, pred_layers: int = 1,
                 joint_hidden: int = 320, durations: list[int] | None = None, max_symbols: int = 10,
                 sigma: float = 0.0, fused_batch_size: int = 0, key: str = "text"):
        super().__init__()
        self.vocab_size, self.blank, self.key = vocab_size, vocab_size, key
        self.durations = list(durations) if durations else None  # None -> plain RNNT
        self.max_symbols, self.sigma = max_symbols, sigma
        # >0: build the B x T x (U+1) x V joint only for sub-batches of this size, cropped to their own
        # max T/U, and recompute it in backward (activation checkpointing), like NeMo's fused batch step
        self.fused_batch_size = fused_batch_size
        self.pred = PredictionNet(vocab_size, pred_hidden, pred_layers)
        n_out = vocab_size + 1 + (len(self.durations) if self.durations else 0)
        self.joint = Joint(d_model, pred_hidden, joint_hidden, n_out)

    @property
    def is_tdt(self):
        return self.durations is not None

    def _nll(self, enc, g, y, enc_len, yl, reduction="none"):
        z = self.joint(enc, g)
        V1 = self.vocab_size + 1
        if self.is_tdt:
            return tdt_loss(z[..., :V1], z[..., V1:], y, enc_len, yl, self.blank, self.durations, self.sigma,
                            reduction=reduction)
        return rnnt_loss(z, y, enc_len, yl, self.blank, reduction=reduction)

    def loss(self, enc, enc_len, batch):
        y, yl = batch[self.key], batch[self.key + "_len"]
        g, _ = self.pred(self.pred.prepend_sos(y))
        if not self.fused_batch_size:
            return self._nll(enc, g, y, enc_len, yl, reduction="mean")
        nll = []
        for i in range(0, enc.shape[0], self.fused_batch_size):
            s = slice(i, i + self.fused_batch_size)
            T, U = int(enc_len[s].max()), int(yl[s].max())
            args = (enc[s, :T], g[s, :U + 1], y[s, :U], enc_len[s], yl[s])
            nll.append(checkpoint(self._nll, *args, use_reentrant=False) if torch.is_grad_enabled()
                       else self._nll(*args))
        return _reduce(torch.cat(nll), yl, "mean")

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return [self._greedy(enc[b, : int(enc_len[b])]) for b in range(enc.shape[0])]

    def _greedy(self, f):
        """Greedy transducer decoding; TDT skips ``duration`` frames per step."""
        V1, dev = self.vocab_size + 1, f.device
        hyp, state = [], None
        g, state = self.pred(torch.tensor([[self.blank]], device=dev), state)
        fe = self.joint.enc(f)
        t, T, emitted = 0, f.shape[0], 0
        while t < T:
            z = self.joint.out(fe[t][None, None] + self.joint.pred(g))[0, 0]
            k = int(z[:V1].argmax())
            if self.is_tdt:
                d = self.durations[int(z[V1:].argmax())]
                if k == self.blank and d == 0:
                    d = 1
            else:
                d = 1 if k == self.blank else 0
            if k != self.blank:
                hyp.append(k)
                g, state = self.pred(torch.tensor([[k]], device=dev), state)
                emitted += 1
            if d == 0 and emitted >= self.max_symbols:
                d = 1
            if d > 0:
                emitted = 0
            t += d
        return hyp

    def greedy_stream(self) -> GreedyTransducerStream:
        """Incremental greedy decoder over encoder frames fed in pieces (same rule as ``_greedy``)."""
        return GreedyTransducerStream(self)


class GreedyTransducerStream:
    """Greedy RNNT / TDT decoding over encoder frames that arrive in pieces. ``feed(f)`` (f: (T, D) encoder frames)
    returns the token ids emitted by this piece; ``tokens`` holds all of them. The prediction-net state and a TDT
    duration that jumps past the end of a piece (``skip``) carry over, so any split of the frames gives exactly the
    tokens of ``RNNTHead._greedy`` on the whole sequence (tests/test_hybrid_asr.py)."""

    def __init__(self, head: RNNTHead):
        self.h, self.tokens, self.skip, self.emitted, self.pred = head, [], 0, 0, None

    @torch.no_grad()
    def feed(self, f: torch.Tensor) -> list[int]:
        h, dev, V1 = self.h, f.device, self.h.vocab_size + 1
        if self.pred is None:
            self.pred = h.pred(torch.tensor([[h.blank]], device=dev), None)
        g, st = self.pred
        fe = h.joint.enc(f)
        new, t, emitted = [], self.skip, self.emitted
        while t < f.shape[0]:
            z = h.joint.out(fe[t][None, None] + h.joint.pred(g))[0, 0]
            k = int(z[:V1].argmax())
            if h.is_tdt:
                d = h.durations[int(z[V1:].argmax())]
                if k == h.blank and d == 0:
                    d = 1
            else:
                d = 1 if k == h.blank else 0
            if k != h.blank:
                new.append(k)
                g, st = h.pred(torch.tensor([[k]], device=dev), st)
                emitted += 1
            if d == 0 and emitted >= h.max_symbols:
                d = 1
            if d > 0:
                emitted = 0
            t += d
        self.skip, self.emitted, self.pred = t - f.shape[0], emitted, (g, st)
        self.tokens += new
        return new


# --------------------------------------------------------------------------- AED (Canary)
class AEDHead(Head):
    """Transformer decoder with cross-attention and task prompts (Canary-style).

    The batch provides ``prompt`` (task tokens, e.g. <|en|><|transcribe|><|pnc|>)
    and ``text``; the decoder is trained on prompt + text + <eos>, and loss is
    only computed on the text + <eos> part.
    """

    def __init__(self, d_model: int, vocab_size: int, bos_id: int, eos_id: int, pad_id: int,
                 d_dec: int = 256, n_layers: int = 4, n_heads: int = 4, max_len: int = 512,
                 dropout: float = 0.1, key: str = "text"):
        super().__init__()
        self.bos, self.eos, self.pad, self.key, self.max_len = bos_id, eos_id, pad_id, key, max_len
        self.enc_proj = nn.Linear(d_model, d_dec) if d_model != d_dec else nn.Identity()
        self.embed = nn.Embedding(vocab_size, d_dec)
        self.pos = nn.Embedding(max_len, d_dec)
        layer = nn.TransformerDecoderLayer(d_dec, n_heads, 4 * d_dec, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_dec)
        self.out = nn.Linear(d_dec, vocab_size)

    def _forward(self, tokens, mem, mem_pad):
        T = tokens.shape[1]
        h = self.embed(tokens) + self.pos(torch.arange(T, device=tokens.device))[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        h = self.dec(h, mem, tgt_mask=causal, tgt_key_padding_mask=tokens == self.pad,
                     memory_key_padding_mask=mem_pad, tgt_is_causal=True)
        return self.out(self.norm(h))

    def _mem(self, enc, enc_len):
        pad = torch.arange(enc.shape[1], device=enc.device)[None] >= enc_len[:, None]
        return self.enc_proj(enc), pad

    def loss(self, enc, enc_len, batch):
        mem, mem_pad = self._mem(enc, enc_len)
        y, yl = batch[self.key], batch[self.key + "_len"]
        B, dev = y.shape[0], y.device
        prompt = batch.get("prompt", torch.zeros(B, 0, dtype=torch.long, device=dev))
        pl = batch.get("prompt_len", torch.zeros(B, dtype=torch.long, device=dev))
        seqs, tgts = [], []
        for b in range(B):
            p = [self.bos] + prompt[b, : pl[b]].tolist()
            t = y[b, : yl[b]].tolist() + [self.eos]
            seqs.append(p + t[:-1])
            tgts.append([-100] * (len(p) - 1) + t)
        L = max(map(len, seqs))
        inp = torch.full((B, L), self.pad, device=dev)
        tgt = torch.full((B, L), -100, device=dev)
        for b in range(B):
            inp[b, : len(seqs[b])] = torch.tensor(seqs[b], device=dev)
            tgt[b, : len(tgts[b])] = torch.tensor(tgts[b], device=dev)
        logits = self._forward(inp, mem, mem_pad)
        return F.cross_entropy(logits.float().transpose(1, 2), tgt, ignore_index=-100)

    @torch.no_grad()
    def decode(self, enc, enc_len, prompt=None, max_new: int = 200, **_):
        mem, mem_pad = self._mem(enc, enc_len)
        out = []
        for b in range(enc.shape[0]):
            p = [self.bos] + (prompt[b] if prompt is not None else [])
            seq = torch.tensor([p], device=enc.device)
            for _ in range(min(max_new, self.max_len - len(p))):
                nxt = int(self._forward(seq, mem[b:b + 1], mem_pad[b:b + 1])[0, -1].argmax())
                if nxt == self.eos:
                    break
                seq = torch.cat([seq, torch.tensor([[nxt]], device=enc.device)], 1)
            out.append(seq[0, len(p):].tolist())
        return out
