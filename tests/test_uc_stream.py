"""scripts/research/uc_stream.py: the live two-channel decisions equal the offline replica used on TurnBench
(scripts/research/uc_turn.py: one head pass over the whole channel, frame_track of the Silero chunk probabilities, rising edges
committed at the 160 ms chunk end, Pipecat-VAD silence timeout, OR with a 2 s refractory), whatever the feed sizes."""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uc_stream import CHUNK, FRAME_S, UCTurnStream  # noqa: E402
from uc_turn_head import UCTurnHead, frame_log_rms_np  # noqa: E402

from audioforge.baselines import turn as B  # noqa: E402


class FakeSilero:
    """Deterministic stand-in: probability = clipped chunk RMS / 0.05 (stateless)."""

    def new_state(self):
        return None

    def step(self, state, chunk):
        return float(min(1.0, np.sqrt(np.mean(np.asarray(chunk, np.float64) ** 2)) / 0.05))

    def probs(self, x):
        n = len(x) // CHUNK
        return np.array([self.step(None, x[j * CHUNK:(j + 1) * CHUNK]) for j in range(n)], np.float32)


def frame_track(p, T):  # bench_turnbench.frame_track
    ends = (np.arange(len(p)) + 1) * B.CHUNK_SEC
    f = np.minimum((ends / 0.08 - 1e-9).astype(np.int64), T - 1)
    s = np.bincount(f, weights=p, minlength=T)[:T]
    c = np.bincount(f, minlength=T)[:T]
    return np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)


def offline(head, x, enc, agent_frames, t1, t2, k_s, rule, theta):
    sil = FakeSilero()
    T = enc.shape[1]
    p = sil.probs(x)
    act = np.stack([frame_track(p, T), agent_frames], -1)
    with torch.no_grad():
        out, _ = head(enc, torch.from_numpy(act)[None], torch.from_numpy(frame_log_rms_np(x, T))[None])
        pr = head.probs(out)
    if rule == "predictive":
        cond = (pr["user_bins"][:, 0] < t1) & (pr["user_bins"][:, 1] < t2)
    else:
        cond = pr["user_quiet"][:, -1] > theta
    cond = cond.numpy()
    emit = ((np.arange(T) // 2 + 1) * 2) * FRAME_S
    rises = np.nonzero(cond & ~np.concatenate([[False], cond[:-1]]))[0]
    ev = [(round(float(emit[t]), 4), 0) for t in rises]
    if k_s is not None:
        sp = B.speech_chunks_pipecat(B.pipecat_vad(p))
        run, armed, need = 0, False, int(np.ceil(k_s / B.CHUNK_SEC - 1e-9))
        for j, s in enumerate(sp):
            if s:
                run, armed = 0, True
            else:
                run += 1
                if armed and run == need:
                    ev.append((round((j + 1) * B.CHUNK_SEC, 4), 1))
    out, last = [], -math.inf
    for t, s in sorted(ev):
        if t - last >= 2.0 and (not out or t > out[-1][0]):
            out.append((t, "head" if s == 0 else "silence"))
            last = t
    return out


def make_audio(rng, sec):
    x = np.zeros(int(sec * 16000), np.float32)
    t = 0.3
    while t < sec - 0.5:
        d = rng.uniform(0.3, 2.0)
        a, b = int(t * 16000), int(min(sec, t + d) * 16000)
        x[a:b] = rng.normal(0, 0.2, b - a)
        t += d + rng.choice([0.15, 0.4, 0.9, 1.8, 2.5])
    return x


def test_stream_equals_offline():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for rule, k_s, t1, theta in (("predictive", 0.6, 0.45, None), ("quiet", None, None, 0.5), ("predictive", None, 0.5, None)):
        head = UCTurnHead(hidden=24, layers=1, energy=True).eval()
        # bias the bins so both branches fire now and then on random features
        x = make_audio(rng, 30.0)
        T = len(x) // 1280
        enc = torch.randn(1, T, 512) * 0.5
        agent_frames = np.zeros(T, np.float32)
        changes = [(3.0, True), (4.2, False), (12.0, True), (13.1, False)]
        for t, s in changes:
            agent_frames[int(round(t / FRAME_S)):] = float(s)
        ref = offline(head, x, enc, agent_frames, t1, 0.5, k_s, rule, theta)
        st = UCTurnStream(head, FakeSilero(), t1=t1 or 0.0, t2=0.5, k_s=k_s, rule=rule, theta=theta or 0.5)
        for t, s in changes:
            st.set_agent(s, t - 1e-3)
        got, pos, v = [], 0, 0
        r = random.Random(1)
        while pos < len(x):
            n = r.choice([160, 512, 1000, 1280, 3000, 7000])
            st.feed_audio(x[pos: pos + n])
            pos += n
            ready = min(T, pos // 1280)
            if ready > v:
                st.feed_enc(enc[:, v:ready])
                v = ready
            got += [(t, s) for t, s, _ in st.poll()]
        st.end()
        got += [(t, s) for t, s, _ in st.poll()]
        assert len(ref) > 0
        assert got == ref, (rule, got[:8], ref[:8])
