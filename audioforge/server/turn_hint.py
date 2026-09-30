"""The early end-of-turn hint (``turn_end_hint`` / ``turn_end_hint_cancel``, docs/PROTOCOL.md section 5.11).

The shipped ``vad_head`` rule waits for 160 ms of VAD silence AND p >= 0.99 (or 640 ms of silence) before its
``turn_end``. A voice agent can start preparing its reply earlier and throw the work away if the user goes on - what
LiveKit Agents calls preemptive generation and Pipecat an eager end of turn. The hint is that earlier, cheaper signal:

* ``TurnHintTracker`` (per 80 ms frame, fed the turn head's p and the served VAD): a hint candidate on the first frame
  of a silence run with VAD silence (frames since the last VAD >= ``vad_thr``) >= ``quiet_frames`` and p >= ``p``;
  then no further candidate until the user resumes, i.e. VAD > ``resume_thr`` on ``resume_frames`` frames in a row,
  which is a cancel candidate and re-arms the tracker.
* ``TurnHintLedger`` (per session, fed the candidates and the session's ``turn_end`` decisions in decision-time
  order): at most one hint outstanding. A hint candidate becomes a ``turn_end_hint`` message only if the user spoke
  after the last ``turn_end`` (a candidate from the silence that turn_end already closed is dropped) and no hint is
  outstanding; the next ``turn_end`` confirms it (``turn_end.hinted_at``), a cancel candidate withdraws it
  (``turn_end_hint_cancel``). Every outcome is counted (``stats.turn_hints``) and logged, so the hint's precision can
  be read off a live session.

Both are plain Python on the frame clock (no model), so scripts/research/turn_hint.py replays them on the stored
per-frame dumps exactly as the server runs them.
"""
from __future__ import annotations

import logging

__all__ = ["TURN_HINT_P", "TURN_HINT_QUIET_FRAMES", "TURN_HINT_RESUME", "TurnHintLedger", "TurnHintTracker", "replay"]

log = logging.getLogger("audioforge.serve")

TURN_HINT_P = 0.8  # --turn-hint-p: the turn head's posterior a hint needs
TURN_HINT_QUIET_FRAMES = 1  # 80 ms of served-VAD silence (VAD < 0.4, the vad_head silence threshold)
TURN_HINT_RESUME = (0.5, 2)  # the user resumed: served VAD > 0.5 on 2 frames in a row


class TurnHintTracker:
    """Hint / cancel candidates from the turn head's p and the served VAD, one frame at a time (module docstring)."""

    def __init__(self, p: float = TURN_HINT_P, quiet_frames: int = TURN_HINT_QUIET_FRAMES, vad_thr: float = 0.4,
                 resume_thr: float = TURN_HINT_RESUME[0], resume_frames: int = TURN_HINT_RESUME[1]):
        self.p, self.quiet, self.vad_thr = float(p), max(1, int(quiet_frames)), float(vad_thr)
        self.resume_thr, self.resume_frames = float(resume_thr), max(1, int(resume_frames))
        self.v = 0  # next frame index
        self.last = -1  # last frame with VAD >= vad_thr
        self.hinted = False  # a candidate fired and the user has not resumed since
        self.run = 0  # frames in a row with VAD > resume_thr

    def update(self, p: float, vad: float) -> list[dict]:
        """The next frame's turn-head p and served VAD -> [] or one candidate: {"kind": "hint", "v", "p",
        "speech_v"} (speech_v = the last speech frame) or {"kind": "cancel", "v"}."""
        v, self.v = self.v, self.v + 1
        if vad >= self.vad_thr:
            self.last = v
        self.run = self.run + 1 if vad > self.resume_thr else 0
        if self.hinted:
            if self.run >= self.resume_frames:
                self.hinted = False
                return [{"kind": "cancel", "v": v}]
            return []
        if self.last >= 0 and v - self.last >= self.quiet and p >= self.p:
            self.hinted = True
            return [{"kind": "hint", "v": v, "p": float(p), "speech_v": self.last}]
        return []


class TurnHintLedger:
    """Turns the tracker's candidates and the session's turn_end decisions into protocol messages (module
    docstring). Feed everything in decision-time order; at equal times the turn_end first."""

    def __init__(self):
        self.pending: dict | None = None  # the outstanding hint: {"t", "p", "v"}
        self.end_v = -1  # frame of the last turn_end
        self.counts = {"sent": 0, "confirmed": 0, "cancelled": 0}
        self.lead_ms: list[float] = []  # per confirmed hint: turn_end.t - hint.t

    def hint(self, t: float, cand: dict, text: str | None = None) -> dict | None:
        """A hint candidate decided at time t -> the ``turn_end_hint`` message, or None (dropped)."""
        if self.pending is not None or cand["speech_v"] <= self.end_v:
            return None
        self.pending = {"t": round(float(t), 3), "p": float(cand["p"]), "v": int(cand["v"])}
        self.counts["sent"] += 1
        msg = {"type": "turn_end_hint", "t": self.pending["t"], "p": round(float(cand["p"]), 5), "kind": "hint"}
        if text is not None:
            msg["text"] = text
        return msg

    def cancel(self, t: float) -> dict | None:
        """A cancel candidate (the user resumed) at time t -> the ``turn_end_hint_cancel`` message, or None."""
        if self.pending is None:
            return None
        h, self.pending = self.pending, None
        self.counts["cancelled"] += 1
        log.info(f"[turn_hint] cancelled: hint t={h['t']:.3f} p={h['p']:.3f}, user resumed at t={t:.3f}")
        return {"type": "turn_end_hint_cancel", "t": round(float(t), 3)}

    def turn_end(self, t: float, v: int) -> float | None:
        """A turn_end decided at time t on frame v -> ``hinted_at`` (the outstanding hint's t, now confirmed) or
        None."""
        self.end_v = max(self.end_v, int(v))
        if self.pending is None:
            return None
        h, self.pending = self.pending, None
        self.counts["confirmed"] += 1
        lead = (float(t) - h["t"]) * 1000
        self.lead_ms.append(lead)
        log.info(f"[turn_hint] confirmed: hint t={h['t']:.3f} p={h['p']:.3f}, turn_end t={t:.3f} "
                 f"({lead:.0f} ms later)")
        return h["t"]

    def stats(self) -> dict:
        """``stats.turn_hints``: hints sent / confirmed / cancelled / outstanding at the end, and the median lead of
        the confirmed ones over their turn_end (ms)."""
        lead = sorted(self.lead_ms)
        return {**self.counts, "open": int(self.pending is not None),
                "lead_ms_p50": round(lead[len(lead) // 2], 1) if lead else None}


def replay(t, p, vad, policy, p_user=None, p_other=None, tracker: TurnHintTracker | None = None) -> dict:
    """Offline replay of one session's per-frame decision times ``t``, turn-head ``p`` and served ``vad`` (plus the
    TS-VAD P(user) / P(other) for ``VadHeadPolicy``'s others path) through ``policy`` (``update(p, vad[, pu, po])``
    -> turn-end event or None) and the hint tracker / ledger, merged as ``Session.process`` merges them (decision-time
    order, a turn_end before a hint / cancel of the same time). -> {"turn_ends": [(t, v, hinted_at)],
    "hints": [(t, v, p, outcome)], "cancels": [t]}; outcome "confirmed" | "cancelled" | "open"."""
    tr = tracker or TurnHintTracker()
    led = TurnHintLedger()
    evs = []
    for v in range(len(p)):
        tv = float(t[v])
        if p_user is not None:
            ev = policy.update(float(p[v]), float(vad[v]), float(p_user[v]), float(p_other[v]))
        else:
            ev = policy.update(float(p[v]), float(vad[v]))
        if ev is not None:
            evs.append((tv, 0, v, None))
        evs += [(tv, 1, v, c) for c in tr.update(float(p[v]), float(vad[v]))]
    evs.sort(key=lambda e: e[:2])
    ends, hints, cancels = [], [], []
    for tv, kind, v, c in evs:
        if kind == 0:
            ha = led.turn_end(tv, v)
            ends.append((tv, v, ha))
            if ha is not None:
                hints[-1] = (*hints[-1][:3], "confirmed")
        elif c["kind"] == "cancel":
            if led.cancel(tv) is not None:
                cancels.append(tv)
                hints[-1] = (*hints[-1][:3], "cancelled")
        elif led.hint(tv, c) is not None:
            hints.append((round(tv, 3), v, c["p"], "open"))
    return {"turn_ends": ends, "hints": hints, "cancels": cancels}
