"""Stable speaker ids for the served finals (research/archive/DIARIZATION_FIX.md).

The diarizer's columns are arrival-order *slots*, not identities: the AOSC cache can permute them, a speaker who
re-enters after the cache dropped them gets a new slot, and a 4-column head cannot hold a fifth person. The
server's legacy rule attaches the 5 s dominant column to every final, and under load shedding replaces the
diarizer by the served VAD in column 0, so every turn becomes speaker 0. This module gives ``audioforge.serve``
(``--diar-labels registry``, ``--shed-diar hold``) the pieces that fix that:

* ``turn_column(rows, a, b)``: the column that dominates the *turn's own span* (not the last 5 s);
* ``SpeakerRegistry``: a per-session registry of unit-norm voice embeddings; ``assign(e)`` returns the id of the
  closest known speaker when its cosine is at least ``thr`` (and updates that centroid), else a new id. Ids are
  therefore keyed by voice and survive column permutations and re-entries; ``len(registry)`` is the session's
  speaker-count estimate;
* ``held_row(last_row, vad, primary, S)``: the diarizer row to report for a frame the diarizer did not run on:
  the last stable column stays active while the served VAD hears speech, and silence stays silence (so the
  primary's silence timeout keeps working); nothing is relabelled to column 0.

Thresholds: cosine on unit vectors. ``DEFAULT_THR`` per embedder was set on the 6-speaker LibriSpeech mix of
research/archive/DIARIZATION_FIX.md section 3 (same-speaker vs different-speaker turn cosines).
"""
from __future__ import annotations

import numpy as np

DEFAULT_THR = {"spk": 0.55, "titanet": 0.40}  # research/archive/DIARIZATION_FIX.md section 3.2: same-speaker p10 0.66 / 0.56,
# different-speaker p90 0.45 / 0.15 on the 6-speaker mix's reference turns
MIN_FRAMES = 8  # 0.64 s of a speaker's own frames before a turn is embedded (else: last id of its column / null)


def turn_column(rows, thr: float = 0.5):
    """(n, S) diarizer rows of a turn -> the dominant column (most active frames, ties by summed probability),
    or None when no column is active on any frame."""
    r = np.asarray(rows, np.float64)
    if r.ndim != 2 or len(r) == 0:
        return None
    act = (r > thr).sum(0)
    if act.max() == 0:
        return None
    score = act + r.sum(0) / (len(r) + 1.0)
    return int(score.argmax())


def held_row(last_row, vad: float, primary, S: int, thr: float = 0.5) -> np.ndarray:
    """The row for a frame the diarizer skipped: the last real row's most active column (else the current primary,
    else column 0) carries the served VAD; everything else is 0."""
    row = np.zeros(S, np.float64)
    last = None if last_row is None else np.asarray(last_row, np.float64)
    if last is not None and len(last) == S and last.max() > thr:
        col = int(last.argmax())
    elif primary is not None and 0 <= int(primary) < S:
        col = int(primary)
    else:
        col = 0
    row[col] = float(vad)
    return row


class SpeakerRegistry:
    """Voice-keyed speaker ids. ``assign(e)`` -> (id, confidence, is_new). Centroids are running means of the
    assigned embeddings (re-normalised, capped at ``cap`` samples so a speaker keeps adapting)."""

    def __init__(self, thr: float, cap: int = 20, max_speakers: int = 64):
        self.thr, self.cap, self.max_speakers = float(thr), int(cap), int(max_speakers)
        self.cent: list[np.ndarray] = []
        self.n: list[int] = []
        self.n_assign = 0

    def __len__(self) -> int:
        return len(self.cent)

    def scores(self, e: np.ndarray) -> np.ndarray:
        if not self.cent:
            return np.zeros(0)
        return np.stack(self.cent) @ np.asarray(e, np.float64)

    def assign(self, e, force_new: bool = False):
        e = np.asarray(e, np.float64).ravel()
        n = np.linalg.norm(e)
        if not np.isfinite(n) or n == 0:
            return None, None, False
        e = e / n
        s = self.scores(e)
        best = int(s.argmax()) if len(s) else -1
        self.n_assign += 1
        if best >= 0 and s[best] >= self.thr and not force_new:
            k = min(self.n[best], self.cap)
            c = self.cent[best] * k + e
            self.cent[best] = c / (np.linalg.norm(c) + 1e-12)
            self.n[best] += 1
            return best, float(round(s[best], 3)), False
        if len(self.cent) >= self.max_speakers:  # bounded: fall back to the closest known speaker
            if best < 0:
                return None, None, False
            return best, float(round(s[best], 3)), False
        self.cent.append(e)
        self.n.append(1)
        # confidence of a *new* speaker = how far the best known one is below the threshold (1 when nobody is known)
        conf = 1.0 if best < 0 else float(round(min(1.0, max(0.0, self.thr - s[best]) / max(self.thr, 1e-6)), 3))
        return len(self.cent) - 1, conf, True

    def state(self) -> dict:
        return {"speakers": len(self.cent), "assignments": self.n_assign, "per_speaker": list(self.n)}
