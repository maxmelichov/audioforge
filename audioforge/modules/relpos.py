"""Transformer-XL relative-position self-attention, exactly as NeMo's ConformerEncoder.

Ports ``RelPositionalEncoding`` and ``RelPositionMultiHeadAttention`` from
nemo/collections/asr/parts/submodules/{multi_head_attention.py} so that pretrained
NeMo FastConformer checkpoints (``self_attention_model: rel_pos``) load weight-for-weight
(parameter names match NeMo's: linear_q/k/v/out, linear_pos, pos_bias_u/v).

    scores = ((q + u) k^T + rel_shift((q + v) P^T)) / sqrt(d_k),   P = linear_pos(pe)

``pe`` holds sinusoids for relative positions (L-1, ..., -(L-1)) with L = cache + T key frames.
Unlike NeMo (which caches the normalized layer input and re-projects it), the streaming cache
holds the projected keys/values; the result is identical because linear_k/linear_v are per frame.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sdpa(q, k, v, attn_mask=None, dropout_p: float = 0.0):
    """scaled_dot_product_attention with a manual fallback: the fused MPS kernel has no dropout."""
    if dropout_p > 0 and q.device.type == "mps":
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            s = s.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else s + attn_mask
        return torch.matmul(F.dropout(s.softmax(-1), dropout_p, training=True), v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


INF_VAL = 10000.0  # NeMo masks scores with -10000, not -inf


def rel_positional_encoding(length: int, d_model: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1, 2*length-1, d_model) sinusoids for positions length-1 .. -(length-1) (NeMo create_pe)."""
    pos = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / d_model))
    pe = torch.zeros(pos.shape[0], d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe[None].to(dtype)


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """(B, H, T1, 2*T2-1) -> (B, H, T1, 2*T2-1) with out[i, j] = x[i, j + T1 - 1 - i] (NeMo rel_shift)."""
    b, h, qlen, pos_len = x.shape
    x = F.pad(x, (1, 0)).view(b, h, -1, qlen)
    return x[:, :, 1:].reshape(b, h, qlen, pos_len)


class RelPositionMultiHeadAttention(nn.Module):
    """Same call signature as RoPEMultiHeadAttention: forward(x, mask, offset, cache) -> (out, (k, v)).

    ``offset`` is ignored: only the cache length matters for relative positions.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.d_model = n_heads, d_model // n_heads, d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.dk))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.dk))
        self.dropout = dropout

    pos_cache_size = 0  # > 0 (audioforge.perf.pos_cache): keep linear_pos(pe(L)) for that many L values (LRU)

    def _pos(self, L: int, x: torch.Tensor) -> torch.Tensor:
        """linear_pos(pe(L)) as (1, H, 2L-1, dk). In inference with ``pos_cache_size`` set, cached per L: it depends
        only on L and the weights (the key includes the weight's storage and version), and the cached tensor is the
        same computation, so outputs are bit-identical; streaming steps reuse one L (cache + chunk)."""
        cache = self.__dict__.get("_pos_cache") if self.pos_cache_size and not self.training \
            and not torch.is_grad_enabled() else None
        if cache is not None:
            w = self.linear_pos.weight
            # inference tensors (a model loaded or re-laid out under torch.inference_mode) have no version counter
            ver = 0 if w.is_inference() else w._version
            key = (L, x.device, x.dtype, w.data_ptr(), ver)
            p = cache.get(key)
            if p is not None:
                cache.move_to_end(key)
                return p
            # the cached tensor outlives this call: make it a regular tensor even inside inference_mode, so it can
            # be used by later calls in either mode
            with torch.inference_mode(False):
                pe = rel_positional_encoding(L, self.d_model, x.device, x.dtype)
                p = self.linear_pos(pe).view(1, -1, self.h, self.dk).transpose(1, 2)
            cache[key] = p
            while len(cache) > self.pos_cache_size:
                cache.popitem(last=False)
            return p
        pe = rel_positional_encoding(L, self.d_model, x.device, x.dtype)
        return self.linear_pos(pe).view(1, -1, self.h, self.dk).transpose(1, 2)

    def forward(self, x, mask=None, offset: int = 0, cache: tuple | None = None):
        """x (B,T,D); mask (B,T,S) bool True=attend (S = cache + T); cache=(k,v) (B,H,C,dk)."""
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.h, self.dk)
        k = self.linear_k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        L = k.shape[2]
        p = self._pos(L, x)  # 1,H,2L-1,dk
        q_u = (q + self.pos_bias_u).transpose(1, 2)  # B,H,T,dk
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        bd = rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))[..., :L] / math.sqrt(self.dk)
        if mask is not None:
            bd = bd.masked_fill(~mask[:, None], -INF_VAL)
        # SDPA adds bd to (q_u k^T)/sqrt(dk): the same scores as NeMo's (ac + bd) / sqrt(dk)
        out = sdpa(q_u, k, v, attn_mask=bd, dropout_p=self.dropout if self.training else 0.0)
        return self.linear_out(out.transpose(1, 2).reshape(B, T, -1)), (k, v)
