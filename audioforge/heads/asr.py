"""ASR heads: CTC, RNNT, TDT and a Canary-style attention encoder-decoder (AED).

All heads share the interface
    loss(enc, enc_len, batch) -> scalar
    decode(enc, enc_len, **kw) -> list[list[int]]
so any combination can hang off one encoder (hybrid TDT+CTC, AED+CTC, ...).
"""
from __future__ import annotations

import math

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


class PromptedLinear(nn.Linear):
    """The encoder projection of a language-prompted transducer (nemotron-3.5-asr-streaming-0.6b, NeMo
    ``EncDecRNNTBPEModelWithPrompt``): every encoder frame is concatenated with a one-hot language prompt (K =
    ``num_prompts``), mapped back to d_enc by ``kernel`` (Linear(d_enc + K, hidden), ReLU, Linear(hidden, d_enc)),
    then projected as usual. The prompt only touches the RNNT path; the encoder blocks (what every audioforge head
    reads) do not depend on it. ``prompt_id`` is a runtime choice (not saved): ``set_prompt``."""

    def __init__(self, d_enc: int, d_joint: int, num_prompts: int, hidden: int, prompt_id: int = 0):
        super().__init__(d_enc, d_joint)
        self.num_prompts, self.prompt_id = num_prompts, int(prompt_id)
        self.kernel = nn.Sequential(nn.Linear(d_enc + num_prompts, hidden), nn.ReLU(), nn.Linear(hidden, d_enc))

    def forward(self, f):
        oh = f.new_zeros(*f.shape[:-1], self.num_prompts)
        oh[..., self.prompt_id] = 1.0
        return super().forward(self.kernel(torch.cat([f, oh], -1)))


class Joint(nn.Module):
    def __init__(self, d_enc: int, d_pred: int, d_joint: int, n_out: int, dropout: float = 0.1,
                 prompt: dict | None = None):
        super().__init__()
        self.enc = (PromptedLinear(d_enc, d_joint, int(prompt["num_prompts"]), int(prompt["hidden"]),
                                   int(prompt.get("dictionary", {}).get(prompt.get("default"), 0)))
                    if prompt else nn.Linear(d_enc, d_joint))
        self.pred = nn.Linear(d_pred, d_joint)
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_joint, n_out))

    def forward(self, f, g):
        """f (B,T,De), g (B,U,Dp) -> (B,T,U,n_out)."""
        return self.out(self.enc(f)[:, :, None] + self.pred(g)[:, None])


class RNNTHead(Head):
    # The shipped RNNT / CTC / prompt sizes come from the imported NVIDIA checkpoint's config (pred / joint 640 on both
    # cores) and are fixed by its weights; the defaults here (320) are for from-scratch configs: placeholder: never swept
    def __init__(self, d_model: int, vocab_size: int, pred_hidden: int = 320, pred_layers: int = 1,
                 joint_hidden: int = 320, durations: list[int] | None = None, max_symbols: int = 10,
                 sigma: float = 0.0, fused_batch_size: int = 0, key: str = "text", prompt: dict | None = None):
        super().__init__()
        self.vocab_size, self.blank, self.key = vocab_size, vocab_size, key
        self.prompt = dict(prompt) if prompt else None  # language-prompted joint (PromptedLinear)
        self.durations = list(durations) if durations else None  # None -> plain RNNT
        self.max_symbols, self.sigma = max_symbols, sigma
        # >0: build the B x T x (U+1) x V joint only for sub-batches of this size, cropped to their own
        # max T/U, and recompute it in backward (activation checkpointing), like NeMo's fused batch step
        self.fused_batch_size = fused_batch_size
        self.pred = PredictionNet(vocab_size, pred_hidden, pred_layers)
        n_out = vocab_size + 1 + (len(self.durations) if self.durations else 0)
        self.joint = Joint(d_model, pred_hidden, joint_hidden, n_out, prompt=prompt)

    def set_prompt(self, lang: str | int):
        """Language prompt of a prompted joint: a locale of the model's prompt dictionary ('en-US', 'de', 'auto')
        or a prompt index. 'auto' makes the model detect the language and emit a ``<xx-XX>`` tag token after the
        terminal punctuation."""
        if self.prompt is None:
            raise ValueError("this transducer has no language prompt")
        d = self.prompt.get("dictionary", {})
        if isinstance(lang, str) and lang not in d:
            raise ValueError(f"unknown language prompt {lang!r}: one of {', '.join(sorted(d))}")
        self.joint.enc.prompt_id = int(d[lang] if isinstance(lang, str) else lang)

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

    def beam_stream(self, beam: int = 4, max_sym: int = 3) -> BeamTransducerStream:
        """Incremental RNNT beam search over encoder frames fed in pieces (``BeamTransducerStream``)."""
        return BeamTransducerStream(self, beam, max_sym)

    @torch.no_grad()
    def beam_search(self, f, beam: int = 4, max_sym: int = 3) -> list[int]:
        """Beam search over one utterance's encoder frames f (T, D): ``BeamTransducerStream`` fed all frames."""
        st = BeamTransducerStream(self, beam, max_sym)
        st.feed(f)
        return st.tokens


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


class BeamTransducerStream:
    """RNNT beam search over encoder frames that arrive in pieces (plain RNNT only; research/FIXALL.md step 5).

    Per frame: up to ``max_sym`` non-blank expansions (batched over the live hypotheses, joint + prediction net),
    every hypothesis that emits blank moves to the next frame with its log-probability added; identical token
    sequences are merged (log-sum-exp); the ``beam`` best survive. Expansions that cannot beat the beam-th
    hypothesis already finished on this frame are pruned (adaptive-expansion pruning). ``tokens`` is the current best
    hypothesis (it may revise its last tokens as frames arrive); ``stable`` is the prefix shared by every hypothesis.
    Unlike greedy decoding, a hypothesis that emitted a token competes with its own blank-ending parent, so even
    beam 1 is not greedy."""

    def __init__(self, head: RNNTHead, beam: int = 4, max_sym: int = 3):
        assert not head.is_tdt, "beam search: plain RNNT heads only"
        self.h, self.beam, self.max_sym = head, int(beam), int(max_sym)
        self.hyps = None  # list of (tokens tuple, logp, joint-pred projection (Dj,), lstm state (h, c))

    def _pred(self, toks, states):
        h = self.h
        dev = h.joint.out[-1].weight.device
        if states[0] is None:
            st = None
        else:
            st = (torch.cat([s[0] for s in states], 1), torch.cat([s[1] for s in states], 1))
        g, (hh, cc) = h.pred(torch.tensor(toks, device=dev)[:, None], st)
        gp = h.joint.pred(g[:, 0])
        return gp, [(hh[:, i:i + 1], cc[:, i:i + 1]) for i in range(len(toks))]

    @torch.no_grad()
    def feed(self, f: torch.Tensor) -> list[int]:
        h, V = self.h, self.h.vocab_size
        if self.hyps is None:
            gp, st = self._pred([h.blank], [None])
            self.hyps = [((), 0.0, gp[0], st[0])]
        fe = h.joint.enc(f)
        for t in range(fe.shape[0]):
            A, B = self.hyps, {}
            for s in range(self.max_sym + 1):
                if not A:
                    break
                lp = h.joint.out(fe[t][None] + torch.stack([a[2] for a in A])).float().log_softmax(-1)
                lpb = (torch.tensor([a[1] for a in A], device=lp.device) + lp[:, V]).tolist()
                for a, sc in zip(A, lpb):
                    if a[0] in B:
                        o = B[a[0]]
                        B[a[0]] = (a[0], max(o[1], sc) + math.log1p(math.exp(-abs(o[1] - sc))), o[2], o[3])
                    else:
                        B[a[0]] = (a[0], sc, a[2], a[3])
                if s == self.max_sym:
                    break
                fin = sorted((b[1] for b in B.values()), reverse=True)
                thr = fin[self.beam - 1] if len(fin) >= self.beam else -float("inf")
                cand = torch.tensor([a[1] for a in A], device=lp.device)[:, None] + lp[:, :V]
                k = min(self.beam, cand.numel())
                vals, idx = cand.reshape(-1).topk(k)
                sel = [(int(i) // V, int(i) % V, float(v)) for v, i in zip(vals, idx) if float(v) > thr]
                if not sel:
                    break
                gp, st = self._pred([kk for _, kk, _ in sel], [A[i][3] for i, _, _ in sel])
                A = [(A[i][0] + (kk,), sc, gp[j], st[j]) for j, (i, kk, sc) in enumerate(sel)]
            self.hyps = sorted(B.values(), key=lambda b: -b[1])[: self.beam]
        return self.tokens

    @property
    def tokens(self) -> list[int]:
        return list(self.hyps[0][0]) if self.hyps else []

    @property
    def stable(self) -> list[int]:
        if not self.hyps:
            return []
        seqs = [hh[0] for hh in self.hyps]
        n = 0
        while all(len(q) > n for q in seqs) and len({q[n] for q in seqs}) == 1:
            n += 1
        return list(seqs[0][:n])


# --------------------------------------------------------------------------- AED (Canary)
class AEDHead(Head):
    """Transformer decoder with cross-attention and task prompts (Canary-style).

    The batch provides ``prompt`` (task tokens, e.g. <|en|><|transcribe|><|pnc|>)
    and ``text``; the decoder is trained on prompt + text + <eos>, and loss is
    only computed on the text + <eos> part.
    """

    def __init__(self, d_model: int, vocab_size: int, bos_id: int, eos_id: int, pad_id: int,
                 d_dec: int = 256, n_layers: int = 4, n_heads: int = 4, max_len: int = 512,  # placeholder: never swept
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
