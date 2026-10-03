"""End-of-turn policies on the 80 ms frame clock: the silence timeouts on the diarizer's primary column, the turn
head's threshold crossing, and the any-speaker Silero silence used by ``hybrid_silero`` / ``hybrid_dyn``."""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .constants import ACT_THRESHOLD, FRAME_MS, FRAME_SAMPLES, PRIMARY_WINDOW_S, SR

__all__ = ["AnySpeakerTimeout", "EnergyGate", "HeadPolicy", "SileroSilence", "TimeoutPolicy", "VadHeadPolicy", "frame_db"]


# --------------------------------------------------------------------------- turn policies
class TimeoutPolicy:
    """Silence timeout on the diarizer's primary track (the product default).

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
    """turn_policy ``timeout_any`` (multi-party rooms): a turn belongs to
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


def frame_db(x) -> float:
    """Log energy of one frame of 16 kHz samples in dBFS (10 log10 of the mean square; -100 for an empty or
    digitally silent frame)."""
    x = np.asarray(x, np.float64)
    return float(10.0 * np.log10(np.mean(x * x) + 1e-10)) if len(x) else -100.0


class EnergyGate:
    """The energy-aware quiet gate of ``vad_head`` (``--quiet-gate energy``).

    The served VAD head reads about 0.66 on -50 dBFS room tone and about 0.55 on a fresh session's first frames, so
    on its own the rule's silence (VAD < thr) starts only at digital silence. Per session, one 80 ms frame at a time
    with its log energy e (``frame_db``) and the served VAD:

    - noise floor: the ``pct`` percentile of the frame energies of the last ``window_s`` (including this frame; the
      first frames use what there is, so the floor starts from the session's first ~300 ms). With ``admit="quiet"``
      a frame that is a speech onset (below) does not enter the window, so a long turn cannot lift the floor to
      speech level.
    - quiet: e < floor + ``quiet_db``. A quiet frame counts as silence whatever the VAD says (``vad_head``'s silence
      = VAD < thr OR quiet).
    - onset: VAD > ``onset_vad`` AND e > floor + ``onset_db``. Only an onset arms the rule for a new turn_end.
    - warm-up guard: ``warm`` once ``warmup_frames`` onset frames were seen in the session (default 4 = 320 ms of
      speech); before that no turn_end and no turn_end_hint.

    ``update(e, vad)`` -> (quiet, onset). ``floor`` / ``n_onset`` / ``warm`` describe the state after the call."""

    def __init__(self, quiet_db: float | None = 6.0, onset_db: float = 9.0, window_s: float = 3.0, pct: float = 10.0,
                 warmup_frames: int = 4, onset_vad: float = ACT_THRESHOLD, admit: str = "all",
                 abs_db: float | None = None, frame_ms: int = FRAME_MS):
        if admit not in ("all", "quiet"):
            raise ValueError(f"admit {admit!r}: 'all' or 'quiet'")
        self.quiet_db = None if quiet_db is None else float(quiet_db)
        self.abs_db = None if abs_db is None else float(abs_db)
        self.onset_db, self.pct = float(onset_db), float(pct)
        self.onset_vad, self.admit, self.warmup_frames = float(onset_vad), admit, max(0, int(warmup_frames))
        self.hist: deque = deque(maxlen=max(1, int(math.ceil(window_s * 1000 / frame_ms))))
        self.floor: float | None = None
        self.n_onset = 0
        self.last_quiet = False  # the last frame's quiet

    @property
    def warm(self) -> bool:
        return self.n_onset >= self.warmup_frames

    def update(self, e: float, vad: float) -> tuple[bool, bool]:
        """One frame's log energy (dBFS) and served VAD -> (quiet, onset)."""
        e = float(e) if math.isfinite(e) else -100.0
        cand = list(self.hist) + [e]
        floor = float(np.percentile(cand, self.pct))
        onset = vad > self.onset_vad and e > floor + self.onset_db
        if self.admit == "all" or not onset:
            self.hist.append(e)
        self.floor = floor
        self.n_onset += onset
        quiet = (self.quiet_db is not None and e < floor + self.quiet_db) or (self.abs_db is not None and e < self.abs_db)
        self.last_quiet = bool(quiet)
        return self.last_quiet, onset


class VadHeadPolicy:
    """turn_policy ``vad_head`` (``--mode single``'s default rule), no Silero. Per 80 ms frame
    v (fed in order) with the turn head's p, the served VAD and, when a TS-VAD track is enrolled, P(user) / P(other):

    - head path: the served VAD's silence (frames since the last speech frame, a frame with VAD >= ``vad_thr``) >=
      ``k_frames`` and p >= ``threshold``;
    - fallback: that silence >= ``fallback_frames`` (None = off);
    - others path (``others`` = (user silence frames, hold frames), None = off; needs P(user) / P(other)): at the frame
      where the user's own silence (frames since P(user) >= ``user_p``) reaches the user-silence frames, P(other) >=
      ``others_p`` has held for the last hold frames: another speaker has the floor, so the user's turn ended when the
      user went quiet (the any-speaker VAD would wait for the room to go quiet).

    One firing per user turn: the head path and the fallback re-arm on a VAD speech frame, the others path on a
    P(user) >= ``user_p`` frame, and a firing of either disarms both. The same rule as the EOT-latency
    simulation's ``sim_room`` (``fw`` 0).

    ``gate`` (an ``EnergyGate``; None = the VAD alone, as scored in the EOT-latency benchmark): when given and a frame's
    log energy is passed to ``update``, a frame is a speech frame only if VAD >= ``vad_thr`` AND it is not energy-quiet,
    the head path / fallback re-arm only on an energy onset, and nothing fires before the gate is warm.

    ``turn_model`` (``--turn-model smartturn``): a callable ``(v, onset_v) -> (p_complete, ms)`` that replaces the head
    path. It runs once per silence run, at the frame where the silence reaches ``k_frames``; p_complete > 0.5 fires
    (path "model"), otherwise the rule waits for the next silence run or the fallback. ``onset_v`` = the first speech
    frame of the turn (the first onset since the last firing)."""

    def __init__(self, threshold: float = 0.99, k_frames: int = 2, fallback_frames: int | None = 8,
                 vad_thr: float = 0.4, frame_ms: int = FRAME_MS, others: tuple[int, int] | None = (12, 8),
                 others_p: float = 0.9, user_p: float = 0.5, gate: EnergyGate | None = None, turn_model=None,
                 model_quiet_only: bool = False, model_vad_thr: float | None = None, model_p: float = 0.5,
                 model_reask: bool = False, reset_thr: float | None = None):
        self.thr, self.k, self.fb, self.vad_thr, self.frame_ms = threshold, int(k_frames), fallback_frames, vad_thr, frame_ms
        self.others = None if not others else (int(others[0]), max(1, int(others[1])))
        self.others_p, self.user_p = float(others_p), float(user_p)
        self.gate, self.turn_model = gate, turn_model
        # two clocks (turn_model): the classifier's trigger counts silence from the last frame with VAD >=
        # model_vad_thr (default vad_thr) that is not energy-quiet; with model_quiet_only the energy-quiet frames do
        # NOT count as silence for the head path / fallback (their timer stays on the VAD's own silence)
        self.model_quiet_only = bool(model_quiet_only)
        self.model_vad_thr = vad_thr if model_vad_thr is None else float(model_vad_thr)
        self.last_m = -1  # last speech frame on the classifier's clock
        self.model_p = float(model_p)  # the classifier's P(complete) that ends the turn (Pipecat: > 0.5)
        # model_reask (turn head v5): after an "incomplete" answer the classifier is asked again
        # at every further quiet frame of the same silence run (default: once per run, then the fallback)
        self.model_reask = bool(model_reask)
        # reset_thr (two thresholds): a frame with VAD >= reset_thr (below the speech
        # threshold) does not arm a turn but restarts both silence clocks, so a pause the VAD is unsure about (a noisy
        # channel) is not counted as silence; None = one threshold (the default)
        self.reset_thr = None if reset_thr is None else float(reset_thr)
        self.v = 0  # next frame index
        self.last = -1  # last VAD speech frame
        self.armed = False  # a speech frame (gated: an onset) since the last firing
        self.onset_v = -1  # first arming frame of the current turn
        self.last_u = -1  # last user speech frame (P(user) >= user_p)
        self.fired_u = -2
        self.orun = 0  # frames in a row with P(other) >= others_p
        self.model_calls: list[dict] = []  # turn_model: {"v", "p", "ms", "complete"} per call
        self._asked = -1  # turn_model: the speech frame whose silence run was already classified

    def update(self, p: float, vad: float, p_user: float | None = None, p_other: float | None = None,
               energy_db: float | None = None, vad_m: float | None = None) -> dict | None:
        """The turn head's p, the served VAD, (enrolled TS-VAD track) P(user), P(other) and (with a gate) the
        frame's log energy on the next frame -> None, or the turn-end event. ``vad_m``: a
        second, stateless VAD for the classifier's clock only (``turn_model``); arming, the gate, the fallback and the
        head path keep reading ``vad``. None = the classifier's clock reads ``vad`` too (the default)."""
        v, self.v = self.v, self.v + 1
        gated = self.gate is not None and energy_db is not None
        quiet = False
        if gated:
            quiet, onset = self.gate.update(energy_db, vad)
            speech = vad >= self.vad_thr and not (quiet and not self.model_quiet_only)
            arm = speech and onset
        else:
            speech = arm = vad >= self.vad_thr
        if speech:
            self.last = v
        vm = vad if vad_m is None else vad_m
        if vm >= self.model_vad_thr and not quiet:
            self.last_m = v
        if self.reset_thr is not None and not (quiet and not self.model_quiet_only):
            if vad >= self.reset_thr:
                self.last = v
            if vm >= self.reset_thr and not quiet:
                self.last_m = v
        if arm:
            if not self.armed:
                self.onset_v = v
            self.armed = True
        track = p_user is not None and p_other is not None
        if track:
            if p_user >= self.user_p:
                self.last_u = v
            self.orun = self.orun + 1 if p_other >= self.others_p else 0
        if gated and not self.gate.warm:
            return None
        sil = v - self.last if self.last >= 0 else 0
        path, pv = None, float(p)
        if self.armed and self.turn_model is not None and self.last_m >= 0:
            sil_m = v - self.last_m  # the classifier's clock: asked once per silence run on it
            if sil_m >= self.k and self._asked != self.last_m:
                self._asked = self.last_m
                pc, ms = self.turn_model(v, self.onset_v)
                self.model_calls.append({"v": v, "p": float(pc), "ms": float(ms), "complete": pc > self.model_p})
                if pc > self.model_p:
                    path, pv, sil = "model", float(pc), sil_m
                elif self.model_reask:
                    self._asked = -1
        if path is None and self.armed and self.last >= 0 and sil > 0:
            if self.turn_model is None and sil >= self.k and p >= self.thr:
                path = "head"
            elif self.fb is not None and sil >= self.fb:
                path = "fallback"
        if path is None and track and self.others is not None and self.last_u >= 0 and self.last_u != self.fired_u \
                and self.orun >= self.others[1] and v - self.last_u == self.others[0]:
            path, sil = "others", v - self.last_u
        if path is None:
            return None
        self.armed, self.fired_u = False, self.last_u
        return {"p": pv, "silence_ms": int(sil * self.frame_ms), "path": path}


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
