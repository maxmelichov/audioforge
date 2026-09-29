"""FastConformer encoder with cache-aware streaming.

This is the backbone NVIDIA reuses across Parakeet, Canary, Nemotron-ASR,
Sortformer, NEST, multitalker ASR, EOU and the VoiceChat speech encoder:

* 8x depthwise-striding conv subsampling  -> 80 ms frames (12.5 Hz) at 10 ms hop
* Macaron conformer blocks: 1/2 FF -> MHSA -> depthwise conv -> 1/2 FF -> LN
* ``att_context_size=[L, R]`` chunked-limited attention: the same weights run
  offline (full context) or streaming (chunks of R+1 frames, L frames of left
  context). Several sizes can be trained together ("unified"/"multi" models).
* In streaming mode every layer caches its past keys/values and the depthwise
  conv state, so each chunk is computed exactly once.

Deviation from NeMo (default): rotary position embeddings instead of Transformer-XL
relative positions (simpler caching, same asymptotics). The subsampling convs also
re-zero padded frames between stages (NeMo does not), so an utterance padded in a
batch encodes exactly as it does alone (tests/test_batch_invariance.py).

NeMo-compatible options (used by audioforge.nemo_import to load pretrained .nemo weights):
``pos_emb="rel_pos"`` (modules/relpos.py), ``xscaling=True`` (x * sqrt(d_model) after
subsampling), ``subsampling_activation="relu"`` and ``subsampling_padding="nemo"`` (NeMo's
CausalConv2D pads (2, 1) in time *and* frequency; output length floor(l/2)+1 per stage).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .relpos import RelPositionMultiHeadAttention, sdpa


# --------------------------------------------------------------------------- subsampling
class DWStridingSubsampling(nn.Module):
    """3 stride-2 conv stages (1 full conv + 2 depthwise-separable) = 8x in time and freq."""

    def __init__(self, feat_in: int, d_model: int, channels: int = 256, causal: bool = False,
                 activation: str = "silu", padding: str = "ours"):
        super().__init__()
        self.causal = causal
        self.act = {"silu": F.silu, "relu": F.relu}[activation]
        # "nemo" + causal: NeMo CausalConv2D, pad (left 2, right 1) on time and freq
        self.nemo_causal = padding == "nemo" and causal
        assert padding in ("ours", "nemo")
        C = channels
        self.convs = nn.ModuleList([
            nn.Conv2d(1, C, 3, stride=2),
            nn.Sequential(nn.Conv2d(C, C, 3, stride=2, groups=C), nn.Conv2d(C, C, 1)),
            nn.Sequential(nn.Conv2d(C, C, 3, stride=2, groups=C), nn.Conv2d(C, C, 1)),
        ])
        f = feat_in
        for _ in range(3):
            f = (f + (3 if self.nemo_causal else 2) - 3) // 2 + 1
        self.out = nn.Linear(C * f, d_model)
        self.factor = 8
        # streaming look-back in mel frames: the 3-stage receptive field is 15 frames, rounded up to the 8x factor
        self.lookback = 16

    def _pad(self, x):
        # time is dim 2, freq dim 3. causal -> pad only the past in time.
        if self.nemo_causal:
            return F.pad(x, (2, 1, 2, 1))
        return F.pad(x, (1, 1, 2, 0) if self.causal else (1, 1, 1, 1))

    def out_lengths(self, lengths):
        for _ in range(3):
            lengths = self._stage_len(lengths)
        return lengths

    def _stage_len(self, lengths):
        if self.nemo_causal:
            return torch.div(lengths, 2, rounding_mode="floor") + 1
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """x (B, F, T) -> (B, T/8, d_model)."""
        x = x.transpose(1, 2).unsqueeze(1)  # B,1,T,F
        for conv in self.convs:
            # zero frames past each item's length so padded items see the same zeros as unpadded ones
            x = x.masked_fill((torch.arange(x.shape[2], device=x.device) >= lengths[:, None])[:, None, :, None], 0.0)
            x = self.act(conv(self._pad(x)))
            lengths = self._stage_len(lengths)
        B, C, T, Fr = x.shape
        x = self.out(x.permute(0, 2, 1, 3).reshape(B, T, C * Fr))
        return x, lengths


# --------------------------------------------------------------------------- attention

def rotary(x: torch.Tensor, offset: int, base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to x (B, H, T, D) with absolute positions starting at ``offset``."""
    D = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    pos = torch.arange(offset, offset + x.shape[2], device=x.device, dtype=torch.float32)
    ang = pos[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x, mask=None, offset: int = 0, cache: tuple | None = None):
        """x (B,T,D); mask (B,T,S) bool True=attend; cache=(k,v) of past frames."""
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, offset), rotary(k, offset)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        out = sdpa(q, k, v, attn_mask=None if mask is None else mask[:, None],
                   dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, T, -1)), (k, v)


# --------------------------------------------------------------------------- conformer block
class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int = 9, causal: bool = False, norm: str = "layer"):
        super().__init__()
        self.kernel, self.causal = kernel, causal
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.norm_type = norm
        self.norm = nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x, pad_mask=None, cache: torch.Tensor | None = None):
        """x (B,T,D). pad_mask (B,T) True=padding. cache (B,D,k-1) past dw inputs."""
        x = F.glu(self.pw1(x.transpose(1, 2)), dim=1)
        if pad_mask is not None:  # after pw1+GLU (as NeMo): pw1's bias would make padding non-zero
            x = x.masked_fill(pad_mask[:, None], 0.0)
        if cache is not None:
            x = torch.cat([cache, x], dim=2)
            new_cache = x[:, :, -(self.kernel - 1):]
            y = self.dw(x)
        else:
            new_cache = None
            left = self.kernel - 1 if self.causal else (self.kernel - 1) // 2
            y = self.dw(F.pad(x, (left, self.kernel - 1 - left)))
        y = self.norm(y) if self.norm_type == "batch" else self.norm(y.transpose(1, 2)).transpose(1, 2)
        return self.pw2(F.silu(y)).transpose(1, 2), new_cache


class FeedForward(nn.Sequential):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * expansion), nn.SiLU(),
                         nn.Dropout(dropout), nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))


class FastConformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, conv_kernel=9, ff_expansion=4, dropout=0.1, causal=False,
                 conv_norm="layer", pos_emb="rope"):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        self.norm_att = nn.LayerNorm(d_model)
        att_cls = {"rope": RoPEMultiHeadAttention, "rel_pos": RelPositionMultiHeadAttention}[pos_emb]
        self.att = att_cls(d_model, n_heads, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, conv_kernel, causal, conv_norm)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, att_mask, pad_mask, offset=0, att_cache=None, conv_cache=None):
        x = x + 0.5 * self.ff1(x)
        a, kv = self.att(self.norm_att(x), att_mask, offset, att_cache)
        x = x + self.drop(a)
        c, conv_cache = self.conv(self.norm_conv(x), pad_mask, conv_cache)
        x = x + self.drop(c)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x), kv, conv_cache


class SpeakerKernel(nn.Module):
    """Multitalker-Parakeet style speaker kernel injection.

    Given per-frame target-speaker activity a in [0,1], add a speaker branch on
    active frames and a background branch elsewhere, so one shared encoder can
    be steered to a single speaker using only diarization output.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.spk = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.bg = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        for m in (self.spk[-1], self.bg[-1]):  # start as identity
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, act):
        a = act[..., None].to(x.dtype)
        return x + a * self.spk(x) + (1 - a) * self.bg(x)


# --------------------------------------------------------------------------- encoder
def chunked_attention_mask(T: int, lengths: torch.Tensor, att_context: list[int]) -> torch.Tensor:
    """(B,T,T) bool mask. att_context=[L,R] in frames; [-1,-1] = full context.

    As NeMo's ``chunked_limited`` style: chunks of R+1 frames, left context = L // (R+1) whole
    chunks (L is floored to a multiple of the chunk size; L=-1: all past chunks). R=-1 makes one
    chunk of the whole utterance, i.e. full context (L is then ignored).
    """
    dev = lengths.device
    valid = torch.arange(T, device=dev)[None] < lengths[:, None]  # B,T
    mask = valid[:, None, :] & valid[:, :, None]
    L, R = att_context
    if L < 0 and R < 0:
        return mask | ~valid[:, :, None]  # padded queries attend anywhere (avoids NaN rows)
    cs = R + 1 if R >= 0 else T
    idx = torch.arange(T, device=dev) // cs
    left_chunks = L // cs if L >= 0 else T
    diff = idx[:, None] - idx[None, :]  # query chunk - key chunk
    allowed = (diff >= 0) & (diff <= left_chunks)
    return (mask & allowed[None]) | ~valid[:, :, None]


@dataclass
class StreamState:
    offset: int = 0
    mel_cache: torch.Tensor | None = None
    att: list = field(default_factory=list)
    conv: list = field(default_factory=list)
    started: bool = False
    mel_start: int = 0  # NeMo-style aligned streaming: absolute index of mel_cache[..., 0]
    n_mel: int = 0  # mel frames received so far


class FastConformerEncoder(nn.Module):
    def __init__(self, feat_in=80, d_model=256, n_layers=8, n_heads=4, conv_kernel=9, ff_expansion=4,
                 subsampling_channels=256, dropout=0.1, causal=False, att_context_size=None,
                 att_context_sizes=None, conv_norm="layer", speaker_kernel_layers=(),
                 pos_emb="rope", xscaling=False, subsampling_activation="silu", subsampling_padding="ours"):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.xscale = math.sqrt(d_model) if xscaling else None
        self.pre_encode = DWStridingSubsampling(feat_in, d_model, subsampling_channels, causal,
                                                subsampling_activation, subsampling_padding)
        self.layers = nn.ModuleList([
            FastConformerLayer(d_model, n_heads, conv_kernel, ff_expansion, dropout, causal, conv_norm, pos_emb)
            for _ in range(n_layers)])
        # an explicit att_context_size is the inference default and wins over att_context_sizes[0]
        # (added to the training set if missing); with only att_context_sizes (nemo_import), use sizes[0]
        self.att_context_sizes = [list(c) for c in (att_context_sizes or [att_context_size or (-1, -1)])]
        if att_context_size is not None:
            self.att_context_size = list(att_context_size)
            if self.att_context_size not in self.att_context_sizes:
                self.att_context_sizes.append(self.att_context_size)
        else:
            self.att_context_size = self.att_context_sizes[0]
        self.speaker_kernels = nn.ModuleDict({str(i): SpeakerKernel(d_model) for i in speaker_kernel_layers})

    @property
    def subsampling_factor(self) -> int:
        return self.pre_encode.factor

    def forward(self, feats, lengths, att_context_size=None, spk_act=None, return_hidden=False):
        """feats (B,F,T) -> enc (B,T',D), lengths (B,). spk_act (B,T') optional."""
        x, lengths = self.pre_encode(feats, lengths)
        if self.xscale is not None:
            x = x * self.xscale
        if att_context_size is None:
            att_context_size = (random.choice(self.att_context_sizes) if self.training
                                else self.att_context_size)
        T = x.shape[1]
        if spk_act is not None:  # align diarization activity to encoder frames
            spk_act = F.pad(spk_act[:, :T], (0, max(0, T - spk_act.shape[1])))
        att_mask = chunked_attention_mask(T, lengths, att_context_size)
        pad_mask = torch.arange(T, device=x.device)[None] >= lengths[:, None]
        hidden = []
        for i, layer in enumerate(self.layers):
            if spk_act is not None and str(i) in self.speaker_kernels:
                x = self.speaker_kernels[str(i)](x, spk_act)
            x, _, _ = layer(x, att_mask, pad_mask)
            hidden.append(x)
        x = x.masked_fill(pad_mask[..., None], 0.0)
        return (x, lengths, hidden) if return_hidden else (x, lengths)

    # ----------------------------------------------------------------- streaming
    def stream_chunk_frames(self, att_context_size=None) -> int:
        """Number of *mel* frames per streaming step."""
        L, R = att_context_size or self.att_context_size
        return (R + 1) * self.subsampling_factor

    @torch.no_grad()
    def stream_step(self, mel_chunk: torch.Tensor, state: StreamState | None = None,
                    att_context_size=None, spk_act=None, final: bool | None = None, return_hidden: bool = False):
        """Process one chunk (1, F, (R+1)*8) of mel frames; returns (enc (1,R+1,D), state).

        ``return_hidden``: returns (enc, hidden, state), hidden = per-layer outputs for this chunk
        (list of (1,T,D), hidden[-1] is enc), as the offline ``forward(return_hidden=True)``.

        Requires ``causal=True``. Output equals the offline forward with the same
        ``att_context_size`` (verified in tests/test_streaming.py).
        """
        assert self.causal, "streaming requires a causal encoder (causal: true)"
        if self.pre_encode.nemo_causal:
            return self._stream_step_aligned(mel_chunk, state, att_context_size, spk_act, final, return_hidden)
        L, R = att_context_size or self.att_context_size
        cs = R + 1
        left = (L // cs) * cs if L >= 0 else 10 ** 9
        state = state or StreamState()
        if state.started:
            inp = torch.cat([state.mel_cache, mel_chunk], dim=-1)
            drop = state.mel_cache.shape[-1] // self.subsampling_factor
        else:
            inp, drop = mel_chunk, 0
        lb = self.pre_encode.lookback
        state.mel_cache = inp[..., -lb:] if inp.shape[-1] >= lb else inp
        lens = torch.tensor([inp.shape[-1]], device=inp.device)
        x, _ = self.pre_encode(inp, lens)
        x = x[:, drop:]
        if self.xscale is not None:
            x = x * self.xscale
        if not state.started:
            state.att = [None] * len(self.layers)
            state.conv = [torch.zeros(1, self.d_model, l.conv.kernel - 1, device=x.device, dtype=x.dtype)
                          for l in self.layers]
            state.started = True
        hidden = []
        x = self._run_layers(x, state, left, spk_act, hidden)
        return (x, hidden, state) if return_hidden else (x, state)

    def _run_layers(self, x, state, left, spk_act=None, hidden=None):
        for i, layer in enumerate(self.layers):
            if spk_act is not None and str(i) in self.speaker_kernels:
                x = self.speaker_kernels[str(i)](x, spk_act)
            x, kv, state.conv[i] = layer(x, None, None, state.offset, state.att[i], state.conv[i])
            k, v = kv
            state.att[i] = (k[:, :, -left:], v[:, :, -left:]) if left > 0 else None
            if hidden is not None:
                hidden.append(x)
        state.offset += x.shape[1]
        return x

    def _stream_step_aligned(self, mel_chunk, state, att_context_size, spk_act, final, return_hidden=False):
        """Streaming for NeMo-style causal subsampling, where encoder frame v ends at mel frame 8v.

        Emits every frame whose input is complete (8v <= mels received - 1); with ``final`` also the
        trailing frames the offline forward computes over zero padding. Keeps the mel frames from
        8 * (next_frame - 2) on (the subsampling's receptive field is 15 mel frames), so each call
        re-runs the subsampling on at most 16 + chunk frames. Frames are fed to the conformer layers
        in pieces aligned to attention chunks, so feeding (R+1)*8 mel frames per call reproduces the
        offline chunked-limited forward exactly (the first call yields R+1 frames from 8R+1 mels and
        carries 7 over). ``final`` defaults to "this chunk is shorter than (R+1)*8 frames".
        """
        L, R = att_context_size or self.att_context_size
        cs = R + 1
        left = (L // cs) * cs if L >= 0 else 10 ** 9
        f = self.subsampling_factor
        if final is None:
            final = mel_chunk.shape[-1] < cs * f
        state = state or StreamState()
        if not state.started:
            state.att = [None] * len(self.layers)
            state.conv = [torch.zeros(1, self.d_model, l.conv.kernel - 1, device=mel_chunk.device,
                                      dtype=mel_chunk.dtype) for l in self.layers]
            state.mel_cache, state.mel_start, state.n_mel = mel_chunk[..., :0], 0, 0
            state.started = True
        buf = torch.cat([state.mel_cache, mel_chunk], dim=-1)
        state.n_mel += mel_chunk.shape[-1]
        v0 = state.offset
        if final:
            v1 = int(self.pre_encode.out_lengths(torch.tensor(state.n_mel)))
        else:
            v1 = (state.n_mel - 1) // f + 1 if state.n_mel else 0
        outs, hids = [], []
        if v1 > v0:
            # audioforge.perf.share_subsample: a second stream over the same mel chunk at the same position (the
            # speaker-conditioned turn pass) reuses the unconditioned pass's subsampled frames (speaker kernels
            # act after the subsampling, so these frames are the same tensor either way)
            memo = getattr(self, "_sub_memo", None)
            key = (v0, v1, state.n_mel, state.mel_start, state.mel_cache.shape[-1]) if memo is not None else None
            x = memo.get(mel_chunk, key) if memo is not None and spk_act is not None else None
            if x is None:
                a = max(0, f * (v0 - 2))  # window start (absolute mel frame), multiple of 8
                w = buf[..., a - state.mel_start:]
                x, _ = self.pre_encode(w, torch.tensor([w.shape[-1]], device=w.device))
                x = x[:, v0 - a // f: v1 - a // f]
                if self.xscale is not None:
                    x = x * self.xscale
                if memo is not None and spk_act is None:
                    memo.put(mel_chunk, key, x)
            s = 0
            while s < x.shape[1]:  # split at attention-chunk boundaries
                e = min(x.shape[1], s + cs - (v0 + s) % cs)
                sa = None if spk_act is None else spk_act[:, s:e]
                hids.append([])
                outs.append(self._run_layers(x[:, s:e], state, left, sa, hids[-1]))
                s = e
            keep = max(0, f * (v1 - 2))
            state.mel_cache, state.mel_start = buf[..., keep - state.mel_start:], keep
        else:
            state.mel_cache = buf
        if not outs:
            empty = mel_chunk.new_zeros(1, 0, self.d_model)
            return (empty, [empty] * len(self.layers), state) if return_hidden else (empty, state)
        x = torch.cat(outs, 1)
        if not return_hidden:
            return x, state
        return x, [torch.cat([h[i] for h in hids], 1) for i in range(len(self.layers))], state
