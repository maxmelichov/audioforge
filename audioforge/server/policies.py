"""End-of-turn policies on the 80 ms frame clock: the silence timeouts on the diarizer's primary column, the turn
head's threshold crossing, and the any-speaker Silero silence used by ``hybrid_silero`` / ``hybrid_dyn``."""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .constants import ACT_THRESHOLD, FRAME_MS, FRAME_SAMPLES, PRIMARY_WINDOW_S, SR

__all__ = ["AnySpeakerTimeout", "HeadPolicy", "SileroSilence", "TimeoutPolicy", "VadHeadPolicy"]


# --------------------------------------------------------------------------- turn policies
class TimeoutPolicy:
    """Silence timeout on the diarizer's primary track (the product default, docs/SERVER_INTERNALS.md).

    ``require_quiet=False`` (policy ``timeout``, default): fires after timeout_ms of primary silence regardless of
    the other columns. ``require_quiet=True`` (policy ``timeout_quiet``): additionally waits for no other active column.
    ``update(row)`` once per finalized diarizer frame (row = the S column probabilities) -> None or
    {"silence_ms", "primary"} when the turn ends. ``primary`` is maintained for every frame."""

    def __init__(self, timeout_ms: int = 1000, num_spks: int = 4, window_s: float = PRIMARY_WINDOW_S,
                 threshold: float = ACT_THRESHOLD, frame_ms: int = FRAME_MS, require_quiet: bool = False):
        self.timeout_ms, self.S, self.thr, self.frame_ms = timeout_ms, num_spks, threshold, frame_ms
        self.require_quiet = require_quiet
        self.hist: deque = deque(maxlen=int(math.ceil(window_s * 1000 / frame_ms)))
        self.sums = np.zeros(num_spks)
        self.counts = np.zeros(num_spks, dtype=np.int64)
        self.spoke = np.zeros(num_spks, dtype=bool)  # active since the last turn_end
        self.primary: int | None = None
        self.silence_ms = 0
        self.n = 0

    def update(self, row, primary: int | None = None) -> dict | None:
        """``primary``: a column bound by the voice enrollment (--enroll); None = the dominant rule (default)."""
        row = np.asarray(row, dtype=np.float64)[: self.S]
        if len(self.hist) == self.hist.maxlen:
            old = self.hist[0]
            self.sums -= old
            self.counts -= old > self.thr
        self.hist.append(row)
        self.sums += row
        active = row > self.thr
        self.counts += active
        self.spoke |= active
        self.n += 1
        self.primary = int(primary) if primary is not None else (int(np.argmax(self.sums)) if self.counts.max() > 0
                                                                 else None)
        if self.primary is None:
            self.silence_ms = 0
            return None
        p, s = self.primary, 0
        for r in reversed(self.hist):
            if r[p] > self.thr:
                break
            s += 1
        self.silence_ms = s * self.frame_ms
        others = self.require_quiet and bool(np.delete(active, p).any())
        if self.spoke[p] and self.silence_ms >= self.timeout_ms and not others:
            self.spoke[:] = False
            return {"silence_ms": int(self.silence_ms), "primary": p}
        return None


class AnySpeakerTimeout:
    """turn_policy ``timeout_any`` (multi-party rooms; research/archive/DIARIZATION_FIX.md section 3): a turn belongs to
    whoever spoke last, not to the 5 s dominant column, so every speaker's turns end. Same ``update(row)`` /
    ``primary`` / ``silence_ms`` surface as TimeoutPolicy. Fires
      * ``timeout``: no column active for ``timeout_ms`` and someone spoke since the last firing;
      * ``change`` (``{"change": True}``): another column has been active for ``min_run`` consecutive frames while
        the current speaker's column has been inactive for ``change_gap`` frames (overlap and short backchannels do
        not cut: the previous speaker must have stopped). ``primary`` = the current speaker's column; the event's
        ``primary`` is the column whose turn ended."""

    def __init__(self, timeout_ms: int = 1000, num_spks: int = 4, threshold: float = ACT_THRESHOLD,
                 frame_ms: int = FRAME_MS, min_run: int = 3, change_gap: int = 3):
        self.timeout_ms, self.S, self.thr, self.frame_ms = timeout_ms, num_spks, threshold, frame_ms
        self.min_run, self.change_gap = min_run, change_gap
        self.primary: int | None = None
        self.silence_ms = 0
        self.spoke = False
        self.since_any = 0
        self.run = np.zeros(num_spks, dtype=np.int64)
        self.off = np.full(num_spks, 10 ** 6, dtype=np.int64)
        self.n = 0

    def update(self, row, primary: int | None = None) -> dict | None:
        """One finalized diarizer row -> None, or the turn-end event when any speaker's turn is over."""
        row = np.asarray(row, dtype=np.float64)[: self.S]
        active = row > self.thr
        self.run = np.where(active, self.run + 1, 0)
        self.off = np.where(active, 0, self.off + 1)
        self.n += 1
        ev = None
        if active.any():
            self.since_any = 0
            self.spoke = True
            cur = self.primary
            if cur is None:
                self.primary = int(np.argmax(np.where(active, row, -1.0)))
            else:
                cands = [c for c in range(self.S) if c != cur and self.run[c] >= self.min_run]
                if cands and self.off[cur] >= self.change_gap:
                    new = max(cands, key=lambda c: (self.run[c], row[c]))
                    ev = {"silence_ms": int(min(self.off[cur], 10 ** 6) * self.frame_ms), "primary": cur, "change": True}
                    self.primary = new
        else:
            self.since_any += 1
        self.silence_ms = self.since_any * self.frame_ms
        if ev is None and self.spoke and self.silence_ms >= self.timeout_ms:
            self.spoke = False
            ev = {"silence_ms": int(self.silence_ms), "primary": self.primary}
            self.primary = None  # the floor is open: the next speaker starts a fresh turn (no change cut)
        return ev


class HeadPolicy:
    """turn_end when the turn head's probability crosses ``threshold`` upwards (previous frame below it) and there
    was speech (VAD > 0.5) since the last firing - one firing per speech segment, never repeated while p stays high."""

    def __init__(self, threshold: float = 0.98, frame_ms: int = FRAME_MS):
        self.thr, self.frame_ms = threshold, frame_ms
        self.armed = False
        self.prev = 0.0
        self.silent = 0

    def update(self, p: float, vad: float) -> dict | None:
        """The turn head's probability and the VAD on one frame -> None, or the turn-end event when the probability
        crosses the threshold upwards (re-armed by speech)."""
        if vad > ACT_THRESHOLD:
            self.armed, self.silent = True, 0
        else:
            self.silent += 1
        fire = self.armed and p >= self.thr > self.prev
        self.prev = p
        if fire:
            self.armed = False
            return {"p": float(p), "silence_ms": int(self.silent * self.frame_ms)}
        return None


class VadHeadPolicy:
    """turn_policy ``vad_head`` (research/EOT_LATENCY.md; ``--mode single``'s default rule), no Silero. Per 80 ms frame
    v (fed in order) with the turn head's p, the served VAD and, when a TS-VAD track is enrolled, P(user) / P(other):

    - head path: the served VAD's silence (frames since the last frame with VAD >= ``vad_thr``) >= ``k_frames`` and
      p >= ``threshold``;
    - fallback: that silence >= ``fallback_frames`` (None = off);
    - others path (``others`` = (user silence frames, hold frames), None = off; needs P(user) / P(other)): at the frame
      where the user's own silence (frames since P(user) >= ``user_p``) reaches the user-silence frames, P(other) >=
      ``others_p`` has held for the last hold frames: another speaker has the floor, so the user's turn ended when the
      user went quiet (the any-speaker VAD would wait for the room to go quiet).

    One firing per user turn: the head path and the fallback re-arm on a VAD speech frame, the others path on a
    P(user) >= ``user_p`` frame, and a firing of either disarms both. The same rule as scripts/research/eot_latency.py
    ``sim_room`` (``fw`` 0)."""

    def __init__(self, threshold: float = 0.99, k_frames: int = 2, fallback_frames: int | None = 8,
                 vad_thr: float = 0.4, frame_ms: int = FRAME_MS, others: tuple[int, int] | None = (12, 8),
                 others_p: float = 0.9, user_p: float = 0.5):
        self.thr, self.k, self.fb, self.vad_thr, self.frame_ms = threshold, int(k_frames), fallback_frames, vad_thr, frame_ms
        self.others = None if not others else (int(others[0]), max(1, int(others[1])))
        self.others_p, self.user_p = float(others_p), float(user_p)
        self.v = 0  # next frame index
        self.last = -1  # last VAD speech frame
        self.fired = -2  # the VAD speech frame whose silence run already fired
        self.last_u = -1  # last user speech frame (P(user) >= user_p)
        self.fired_u = -2
        self.orun = 0  # frames in a row with P(other) >= others_p

    def update(self, p: float, vad: float, p_user: float | None = None, p_other: float | None = None) -> dict | None:
        """The turn head's p, the served VAD and (enrolled TS-VAD track) P(user), P(other) on the next frame -> None,
        or the turn-end event."""
        v, self.v = self.v, self.v + 1
        if vad >= self.vad_thr:
            self.last = v
        track = p_user is not None and p_other is not None
        if track:
            if p_user >= self.user_p:
                self.last_u = v
            self.orun = self.orun + 1 if p_other >= self.others_p else 0
        sil = v - self.last if self.last >= 0 else 0
        path = None
        if self.last >= 0 and self.last != self.fired and sil > 0:
            if sil >= self.k and p >= self.thr:
                path = "head"
            elif self.fb is not None and sil >= self.fb:
                path = "fallback"
        if path is None and track and self.others is not None and self.last_u >= 0 and self.last_u != self.fired_u \
                and self.orun >= self.others[1] and v - self.last_u == self.others[0]:
            path, sil = "others", v - self.last_u
        if path is None:
            return None
        self.fired, self.fired_u = self.last, self.last_u
        return {"p": float(p), "silence_ms": int(sil * self.frame_ms), "path": path}


class SileroSilence:
    """The any-speaker silence of the shipped rules, live: Silero VAD v5 (``baselines.turn.SileroStream``, 512-sample
    chunks) -> Pipecat's VAD state machine (``baselines.turn.PipecatVADState``; confidence 0.7, start / stop 0.2 s; a
    chunk is speech when STARTING or SPEAKING) -> on the 80 ms frame grid, the time from the end of the last speech
    chunk to the end of frame v, in frames (0 while the chunk holding the frame end is speech or before any speech):
    exactly ``baselines.turn.silence_frames`` on the same audio. ``feed(samples)`` -> [(v, silence frames, ready time s,
    index of the last speech chunk or -1)]; frame v is ready when chunk ceil(2.5 (v + 1)) - 1 (the chunk holding its
    end) is complete. A trailing partial chunk is never run (as the benchmark)."""

    def __init__(self, vad, confidence: float = 0.7, start_secs: float = 0.2, stop_secs: float = 0.2):
        from ..baselines.turn import CHUNK, PipecatVADState
        self.vad, self.state, self.C = vad, vad.new_state(), CHUNK
        self.sm = PipecatVADState(confidence, start_secs, stop_secs)
        self.buf = np.zeros(0, np.float32)
        self.n = 0  # chunks run
        self.v = 0  # next frame
        self.last_sp = -1
        self.ms: list[float] = []  # per-chunk cost (Silero + state machine)

    def _cj(self, v: int) -> int:
        return -(-FRAME_SAMPLES * (v + 1) // self.C) - 1

    def feed(self, x: np.ndarray) -> list[tuple[int, float, float, int]]:
        """Audio samples -> one ``(frame, silence in frames, decision time s, last speech chunk)`` per newly completed
        80 ms frame, from Silero on 32 ms chunks and Pipecat's VAD state machine."""
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        out, C = [], self.C
        while len(self.buf) >= C:
            t0 = time.perf_counter()
            self.sm.step(self.vad.step(self.state, self.buf[:C]))
            self.ms.append((time.perf_counter() - t0) * 1000)
            self.buf = self.buf[C:]
            j, self.n = self.n, self.n + 1
            speech = self.sm.speaking
            if speech:
                self.last_sp = j
            while self._cj(self.v) == j:
                fe = FRAME_SAMPLES * (self.v + 1)
                sil = 0.0 if speech or self.last_sp < 0 else round((fe - C * (self.last_sp + 1)) / FRAME_SAMPLES, 4)
                out.append((self.v, sil, (j + 1) * C / SR, self.last_sp))
                self.v += 1
        return out
