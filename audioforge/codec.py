"""FSQ neural audio codec (NVIDIA Low-Frame-rate / NanoCodec pattern).

conv encoder (residual MRF-style blocks, strided) -> Finite Scalar Quantization
(K codebooks, levels e.g. [8,7,6,6] -> 2016 codes) -> HiFi-GAN-style causal
upsampling decoder. Trained with multi-resolution mel L1 + (optional) GAN.
With ``hop = 1280`` at 16 kHz the codec runs at 12.5 fps, frame-aligned with
the FastConformer encoder, which enables the CodecTokenHead recipes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LogMel, mel_filterbank


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., 2023). No codebook, no commitment loss."""

    def __init__(self, levels: list[int], num_codebooks: int):
        super().__init__()
        self.levels, self.K = list(levels), num_codebooks
        self.dim = len(levels)
        self.register_buffer("L", torch.tensor(levels, dtype=torch.float32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), 0)
        self.register_buffer("basis", basis, persistent=False)

    @property
    def codebook_size(self) -> int:
        return math.prod(self.levels)

    def _bound(self, z):
        half = (self.L - 1) / 2
        offset = torch.where(self.L % 2 == 0, 0.5, 0.0)
        r = offset / half  # L=2: r=1, atanh=inf would pin every value to one level with zero gradient
        shift = torch.atanh(torch.where(r < 1, r, torch.zeros_like(r)))
        return torch.tanh(z + shift) * half - offset

    def forward(self, z):
        """z (B, K*dim, T) -> quantized z (same shape), codes (B, T, K)."""
        B, _, T = z.shape
        z = z.view(B, self.K, self.dim, T).permute(0, 3, 1, 2)  # B,T,K,dim
        b = self._bound(z)
        q = b + (b.round() - b).detach()  # straight-through
        half = (self.L - 1) / 2
        offset = torch.where(self.L % 2 == 0, 0.5, 0.0)
        idx = (q.round() + offset + half).long()  # 0..L-1
        codes = (idx * self.basis).sum(-1)
        zq = (q / (half + offset)).permute(0, 2, 3, 1).reshape(B, self.K * self.dim, T)
        return zq, codes

    def decode_codes(self, codes):
        """codes (B,T,K) -> zq (B, K*dim, T)."""
        B, T, K = codes.shape
        idx = (codes[..., None] // self.basis) % self.L.long()
        half = (self.L - 1) / 2
        offset = torch.where(self.L % 2 == 0, 0.5, 0.0)
        q = idx.float() - half - offset
        return (q / (half + offset)).permute(0, 2, 3, 1).reshape(B, K * self.dim, T)


class ResBlock(nn.Module):
    def __init__(self, c, dilations=(1, 3, 9), causal=False):
        super().__init__()
        self.causal = causal
        self.convs = nn.ModuleList([nn.Conv1d(c, c, 7, dilation=d) for d in dilations])

    def forward(self, x):
        for conv in self.convs:
            p = 6 * conv.dilation[0]
            y = F.pad(F.leaky_relu(x, 0.1), (p, 0) if self.causal else (p // 2, p - p // 2))
            x = x + conv(y)
        return x


class ChannelNorm(nn.Module):
    """Per-frame LayerNorm over channels. Keeps latents in FSQ's tanh range; without it the
    unnormalized conv stack drifts to |z|~1e3, tanh saturates and every frame gets one code."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class AudioCodec(nn.Module):
    def __init__(self, sample_rate=16000, channels=32, strides=(4, 4, 8, 10), levels=(8, 7, 6, 6),
                 num_codebooks=8, causal_decoder=True):
        super().__init__()
        self.sample_rate, self.hop = sample_rate, math.prod(strides)
        self.fsq = FSQ(list(levels), num_codebooks)
        latent = num_codebooks * len(levels)
        enc, c = [nn.Conv1d(1, channels, 7, padding=3)], channels
        for s in strides:
            enc += [ResBlock(c), nn.LeakyReLU(0.1), nn.Conv1d(c, 2 * c, 2 * s, stride=s, padding=s // 2 + s % 2)]
            c *= 2
        enc += [nn.LeakyReLU(0.1), nn.Conv1d(c, latent, 3, padding=1), ChannelNorm(latent)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv1d(latent, c, 7, padding=3)]
        for s in reversed(strides):
            dec += [nn.LeakyReLU(0.1), nn.ConvTranspose1d(c, c // 2, 2 * s, stride=s, padding=s // 2 + s % 2,
                                                          output_padding=s % 2),  # odd s: exactly x s
                    ResBlock(c // 2, causal=causal_decoder)]
            c //= 2
        dec += [nn.LeakyReLU(0.1), nn.Conv1d(c, 1, 7, padding=3), nn.Tanh()]
        self.decoder = nn.Sequential(*dec)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    def encode(self, audio):
        """audio (B,S) -> codes (B, S/hop, K)."""
        S = audio.shape[-1]
        x = F.pad(audio, (0, (-S) % self.hop))[:, None]
        return self.fsq(self.encoder(x))[1]

    def decode(self, codes):
        return self.decoder(self.fsq.decode_codes(codes))[:, 0]

    def forward(self, audio):
        S = audio.shape[-1]
        x = F.pad(audio, (0, (-S) % self.hop))[:, None]
        zq, codes = self.fsq(self.encoder(x))
        return self.decoder(zq)[:, 0, :S], codes


class MultiResMelLoss(nn.Module):
    def __init__(self, sample_rate=16000, ffts=(256, 512, 1024, 2048), n_mels=(32, 64, 80, 128)):
        super().__init__()
        self.cfg = list(zip(ffts, n_mels))
        for i, (n, m) in enumerate(self.cfg):
            self.register_buffer(f"fb{i}", mel_filterbank(sample_rate, n, m), persistent=False)
            self.register_buffer(f"w{i}", torch.hann_window(n), persistent=False)

    def forward(self, y_hat, y):
        loss = 0.0
        for i, (n, _) in enumerate(self.cfg):
            fb, w = getattr(self, f"fb{i}"), getattr(self, f"w{i}")

            def mel(x, n=n, w=w, fb=fb):
                s = torch.stft(x, n, n // 4, n, w, return_complex=True).abs()
                return torch.log(fb @ s + 1e-5)
            loss = loss + F.l1_loss(mel(y_hat), mel(y))
        return loss / len(self.cfg)


class PeriodDiscriminator(nn.Module):
    """Small multi-period discriminator (HiFi-GAN) for optional adversarial training."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.periods = periods
        self.nets = nn.ModuleList([nn.ModuleList([
            nn.Conv2d(1, 16, (5, 1), (3, 1), (2, 0)), nn.Conv2d(16, 64, (5, 1), (3, 1), (2, 0)),
            nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0)), nn.Conv2d(128, 1, (3, 1), 1, (1, 0))]) for _ in periods])

    def forward(self, x):
        outs, feats = [], []
        for p, net in zip(self.periods, self.nets):
            y = F.pad(x[:, None], (0, (-x.shape[-1]) % p), mode="reflect")
            y = y.view(y.shape[0], 1, -1, p)
            for i, conv in enumerate(net):
                y = conv(y)
                if i < len(net) - 1:
                    y = F.leaky_relu(y, 0.1)
                    feats.append(y)
            outs.append(y.flatten(1))
        return outs, feats


class MelCodec(nn.Module):
    """FSQ over stacked log-mel frames (NVIDIA mel-codec pattern; render audio with a mel vocoder
    such as BigVGAN). ``stack=8`` x 10 ms = 80 ms -> 12.5 fps, frame-aligned with FastConformer."""

    def __init__(self, n_mels=80, stack=8, hidden=256, levels=(8, 5, 5, 4), num_codebooks=4, sample_rate=16000):
        super().__init__()
        self.pre = LogMel(sample_rate=sample_rate, n_mels=n_mels, normalize="fixed", dither=0.0)
        self.stack, self.n_mels, self.sample_rate = stack, n_mels, sample_rate
        self.hop = self.pre.hop * stack
        self.fsq = FSQ(list(levels), num_codebooks)
        latent = num_codebooks * len(levels)
        io = n_mels * stack
        self.encoder = nn.Sequential(nn.Conv1d(io, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(hidden, latent, 1), ChannelNorm(latent))
        self.decoder = nn.Sequential(nn.Conv1d(latent, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                                     nn.Conv1d(hidden, io, 1))

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    def mel(self, audio):
        m, _ = self.pre(audio, torch.full((audio.shape[0],), audio.shape[-1], device=audio.device))
        return F.pad(m, (0, (-m.shape[-1]) % self.stack))  # B,F,T (multiple of stack)

    def _stack(self, m):
        B, Fm, T = m.shape
        return m.view(B, Fm, T // self.stack, self.stack).permute(0, 1, 3, 2).reshape(B, Fm * self.stack, -1)

    def _unstack(self, s):
        B, _, T = s.shape
        return s.view(B, self.n_mels, self.stack, T).permute(0, 1, 3, 2).reshape(B, self.n_mels, T * self.stack)

    def encode(self, audio):
        return self.fsq(self.encoder(self._stack(self.mel(audio))))[1]

    def decode_mel(self, codes):
        return self._unstack(self.decoder(self.fsq.decode_codes(codes)))

    def forward(self, audio):
        m = self.mel(audio)
        zq, codes = self.fsq(self.encoder(self._stack(m)))
        return self._unstack(self.decoder(zq)), m, codes
