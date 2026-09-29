"""Audio front end shared by every NVIDIA speech model: 16 kHz log-mel + SpecAugment.

Defaults match the NeMo FastConformer preprocessor: 25 ms window, 10 ms hop,
512-point FFT, 80 (or 128) mel bands, log, per-feature normalization.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """Slaney-style mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    fmax = fmax or sr / 2

    def hz_to_mel(f):
        f = torch.as_tensor(f, dtype=torch.float64)
        lin = f / (200.0 / 3)
        log = 15.0 + torch.log(torch.clamp(f, min=1e-10) / 1000.0) / (math.log(6.4) / 27.0)
        return torch.where(f >= 1000.0, log, lin)

    def mel_to_hz(m):
        lin = m * (200.0 / 3)
        log = 1000.0 * torch.exp((math.log(6.4) / 27.0) * (m - 15.0))
        return torch.where(m >= 15.0, log, lin)

    mels = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=torch.float64)
    hz = mel_to_hz(mels)
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1, dtype=torch.float64)
    lower = (bins[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    upper = (hz[2:, None] - bins[None]) / (hz[2:, None] - hz[1:-1, None])
    fb = torch.clamp(torch.minimum(lower, upper), min=0.0)
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]  # slaney area normalization
    return fb.float()


class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80,
                 preemph=0.97, dither=1e-5, normalize="per_feature", norm_mean=-9.8, norm_std=3.2):
        super().__init__()
        self.sample_rate, self.n_fft, self.win, self.hop = sample_rate, n_fft, win_length, hop_length
        self.preemph, self.dither, self.normalize, self.n_mels = preemph, dither, normalize, n_mels
        # "fixed": global constants, frame-local -> streaming-safe (per_feature needs the whole utterance)
        self.norm_mean, self.norm_std = norm_mean, norm_std
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels), persistent=False)

    def num_frames(self, n_samples: torch.Tensor) -> torch.Tensor:
        return torch.div(n_samples, self.hop, rounding_mode="floor") + 1

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, lengths: torch.Tensor):
        """audio (B, S) float in [-1, 1] -> feats (B, n_mels, T), lengths (B,)."""
        x = audio.float()
        if self.training and self.dither > 0:
            x = x + self.dither * torch.randn_like(x)
        if self.preemph:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1)
        # as NeMo: re-zero padding, else pre-emphasis/dither leak into the last frames of short items
        x = x.masked_fill(torch.arange(x.shape[1], device=x.device)[None] >= lengths[:, None], 0.0)
        spec = torch.stft(x, self.n_fft, self.hop, self.win, self.window, center=True,
                          pad_mode="constant", return_complex=True)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.log(torch.matmul(self.fb, power) + 2 ** -24)
        flen = self.num_frames(lengths)
        mask = torch.arange(mel.shape[-1], device=mel.device)[None] < flen[:, None]
        if self.normalize == "per_feature":
            m = mask[:, None].float()
            n = m.sum(-1, keepdim=True).clamp(min=1)
            mean = (mel * m).sum(-1, keepdim=True) / n
            std = (((mel - mean) * m).pow(2).sum(-1, keepdim=True) / (n - 1).clamp(min=1)).sqrt() + 1e-5
            mel = (mel - mean) / std
        elif self.normalize == "fixed":
            mel = self.fixed_norm(mel)
        mel = mel.masked_fill(~mask[:, None], 0.0)
        return mel, flen

    def fixed_norm(self, mel):
        return (mel - self.norm_mean) / self.norm_std if self.normalize == "fixed" else mel


class SpecAugment(nn.Module):
    """Frequency and time masking (NeMo defaults: 2 x 27 freq, 10 x 5% time)."""

    def __init__(self, freq_masks=2, freq_width=27, time_masks=10, time_width=0.05):
        super().__init__()
        self.fm, self.fw, self.tm, self.tw = freq_masks, freq_width, time_masks, time_width

    @torch.no_grad()
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        B, F, T = x.shape
        for b in range(B):
            L = int(lengths[b])
            for _ in range(self.fm):
                w = int(torch.randint(0, self.fw + 1, ()))
                f0 = int(torch.randint(0, max(1, F - w), ()))
                x[b, f0:f0 + w] = 0.0
            tw = max(1, int(self.tw * L)) if isinstance(self.tw, float) else self.tw
            for _ in range(self.tm):
                w = int(torch.randint(0, tw + 1, ()))
                t0 = int(torch.randint(0, max(1, L - w), ()))
                x[b, :, t0:t0 + w] = 0.0
        return x
