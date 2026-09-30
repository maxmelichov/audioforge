"""Cheap causal prosody per 80 ms frame (turn head v5, research/TURN_V5.md): energy contour, F0 and a
final-lengthening proxy, from the raw 16 kHz samples only. ``Prosody`` is the streaming form (served path);
``prosody_frames`` runs the same object over a whole clip, so offline and served features are identical.

Per 10 ms sub-frame j (analysis window: the 512 samples ending at the sub-frame's end, causal):
  level  dB of the sub-frame relative to a peak follower (decays 2 dB/s);
  F0     normalised-difference (YIN) minimum over 60-400 Hz lags; voiced when the minimum < 0.3 and level > -35 dB.
Per 80 ms frame v (8 sub-frames), N_FEATS = 12 values:
  0-2  level mean / max / last (dB / 20, clipped to [-3, 0.25]);
  3    level change against the previous frame's mean (dB / 20);
  4    voiced fraction;
  5    mean log2 F0 of the voiced sub-frames minus the speaker's running mean (EMA over voiced sub-frames), octaves;
  6    the last voiced sub-frame's log2 F0 minus that mean (held through unvoiced stretches);
  7    F0 slope over the voiced sub-frames of the last 300 ms (octaves / s, / 4, clipped);
  8    length of the current voiced run, or of the last one when unvoiced (log(1 + sub-frames) / 4);
  9    that length / the running mean voiced-run length (final lengthening), log;
  10   time since the last voiced sub-frame (log(1 + sub-frames) / 5);
  11   time since the level was last above -30 dB (log(1 + sub-frames) / 5).
"""
from __future__ import annotations

import math

import numpy as np

SR, HOP, WIN, SUB = 16000, 160, 512, 8
LAG0, LAG1 = SR // 400, SR // 60
N_FEATS = 12


class Prosody:
    def __init__(self):
        self.buf = np.zeros(WIN, np.float32)  # the last WIN samples
        self.pend = np.zeros(0, np.float32)  # samples not yet in a full 10 ms sub-frame
        self.ref = -30.0
        self.f0_mean = None
        self.last_f0 = 0.0
        self.run = 0  # current voiced run (sub-frames)
        self.last_run = 0
        self.run_mean = 10.0
        self.since_v = 1000
        self.since_loud = 1000
        self.prev_mean = None
        self.hist = []  # (voiced, log2 f0) of the last 30 sub-frames
        self.sub = []  # per sub-frame (level, voiced, lf0) of the current frame

    def _subframe(self, x10: np.ndarray):
        self.buf = np.concatenate([self.buf[len(x10):], x10])
        e = float(np.mean(x10.astype(np.float64) ** 2))
        db = 10 * math.log10(e + 1e-10)
        self.ref = max(db, self.ref - 0.02)
        lvl = db - self.ref
        w = self.buf.astype(np.float64)
        w = w - w.mean()
        n = len(w)
        # difference function d(tau) = r(0) terms - 2 r(tau), via FFT autocorrelation
        F = np.fft.rfft(w, 2 * n)
        r = np.fft.irfft(F * np.conj(F))[: LAG1 + 1]
        cs = np.concatenate([[0.0], np.cumsum(w * w)])
        taus = np.arange(1, LAG1 + 1)
        d = cs[n - taus] + (cs[n] - cs[taus]) - 2 * r[1: LAG1 + 1]
        cm = d * taus / np.maximum(np.cumsum(d), 1e-12)
        k = int(np.argmin(cm[LAG0 - 1:])) + LAG0 - 1
        voiced = bool(cm[k] < 0.3 and lvl > -35 and db > -75)
        lf0 = math.log2(SR / (k + 1)) if voiced else 0.0
        if voiced:
            self.f0_mean = lf0 if self.f0_mean is None else 0.995 * self.f0_mean + 0.005 * lf0
            self.last_f0 = lf0
            self.run += 1
            self.since_v = 0
        else:
            if self.run:
                self.last_run = self.run
                self.run_mean = 0.95 * self.run_mean + 0.05 * self.run
            self.run = 0
            self.since_v += 1
        self.since_loud = 0 if lvl > -30 else self.since_loud + 1
        self.hist = (self.hist + [(voiced, lf0)])[-30:]
        self.sub.append((lvl, voiced, lf0))

    def _frame(self) -> np.ndarray:
        lv = np.array([s[0] for s in self.sub])
        vo = np.array([s[1] for s in self.sub], bool)
        lf = np.array([s[2] for s in self.sub])
        self.sub = []
        o = np.zeros(N_FEATS, np.float32)
        c = lambda x: float(np.clip(x / 20, -3, 0.25))  # noqa: E731
        o[0], o[1], o[2] = c(lv.mean()), c(lv.max()), c(lv[-1])
        o[3] = 0.0 if self.prev_mean is None else float(np.clip((lv.mean() - self.prev_mean) / 20, -2, 2))
        self.prev_mean = lv.mean()
        o[4] = vo.mean()
        m0 = self.f0_mean if self.f0_mean is not None else 0.0
        o[5] = float(lf[vo].mean() - m0) if vo.any() and self.f0_mean is not None else 0.0
        o[6] = float(self.last_f0 - m0) if self.f0_mean is not None else 0.0
        hv = [(i, f) for i, (v, f) in enumerate(self.hist) if v]
        if len(hv) >= 4:
            t = np.array([i for i, _ in hv]) * 0.01
            f = np.array([f for _, f in hv])
            o[7] = float(np.clip(np.polyfit(t, f, 1)[0] / 4, -2, 2))
        L = self.run if self.run else self.last_run
        o[8] = math.log1p(L) / 4
        o[9] = float(np.clip(math.log((L + 1) / (self.run_mean + 1)), -3, 3))
        o[10] = math.log1p(min(self.since_v, 1000)) / 5
        o[11] = math.log1p(min(self.since_loud, 1000)) / 5
        return o

    def feed(self, x: np.ndarray) -> list:
        """Raw samples -> one N_FEATS vector per completed 80 ms frame."""
        x = np.concatenate([self.pend, np.asarray(x, np.float32)])
        out, i = [], 0
        while i + HOP <= len(x):
            self._subframe(x[i: i + HOP])
            i += HOP
            if len(self.sub) == SUB:
                out.append(self._frame())
        self.pend = x[i:]
        return out


def prosody_frames(x: np.ndarray, T: int) -> np.ndarray:
    """(T, N_FEATS) for a clip whose encoder has T frames (frame v <- samples [1280 v, 1280 (v + 1)))."""
    x = np.asarray(x, np.float32)
    need = T * HOP * SUB
    x = np.pad(x, (0, max(0, need - len(x))))[:need]
    fr = Prosody().feed(x)
    return np.stack(fr) if fr else np.zeros((0, N_FEATS), np.float32)
