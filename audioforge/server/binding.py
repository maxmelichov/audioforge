"""Primary-speaker binding (``--enroll``): which diarizer column is the user. ``VoiceBinder`` follows a TitaNet
enrollment (after_agent / explicit), ``CausalDominant`` and ``ArmBinder`` bind without embeddings (after_agent_arm)."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from .constants import ACT_THRESHOLD, FRAME_SAMPLES

__all__ = ["ArmBinder", "CausalDominant", "VoiceBinder"]


class VoiceBinder:
    """--enroll after_agent | explicit: the primary column by voice (docs/SERVER_INTERNALS.md; research/EOT_BENCH_V2.md §9).

    ``arm(frame)``: look for the user from diarizer frame ``frame`` on (the audio position of the client's agent_end /
    enroll message). ``update(v, row, frame_audio)`` once per finalized diarizer frame v (its S probabilities and its
    80 ms of audio) -> the bound column, or None while nothing is bound (the caller falls back to its default rule):
      1. armed, no candidate: the first column active (p > thr) for ``min_run`` consecutive frames becomes the
         candidate and the bound column at once (the benchmark's after_prev_end choice);
      2. candidate, not enrolled: its active frames' audio is collected; at ``enroll_frames`` of them the
         enrollment embedding is taken (TitaNet over that speech) and the follower starts on the candidate;
      3. enrolled: every ``stride`` frames each column with >= ``min_frames`` active frames among the last ``win``
         frames is embedded over that speech (the same rule as enrollment.recent_embeddings_audio: frames
         [v - win, v - 1], all before v); the follower's hysteresis (enrollment.VoiceFollower, stepped every frame on
         the latest embeddings) may move the binding.
    A new arm() restarts at step 1 (re-enrollment for the next user turn). Every step reads audio and columns of
    frames <= v only."""

    def __init__(self, embedder, num_spks: int = 4, win: int = 25, min_frames: int = 8, enroll_frames: int = 19,
                 min_run: int = 3, stride: int = 5, margin: float = 0.10, hold: int = 6, thr: float = ACT_THRESHOLD,
                 frame_samples: int = FRAME_SAMPLES):
        from ..enrollment import VoiceFollower  # noqa: F401 - the follower class used below
        self.embedder, self.S = embedder, num_spks
        self.win, self.min_frames, self.enroll_frames, self.min_run = win, min_frames, enroll_frames, min_run
        self.stride, self.margin, self.hold, self.thr, self.fs = stride, margin, hold, thr, frame_samples
        self.on_hist: deque = deque(maxlen=win)  # active masks of the last `win` frames (< v)
        self.audio_hist: deque = deque(maxlen=win)  # their audio
        self.armed_from: int | None = None
        self.cand: int | None = None
        self.run = np.zeros(num_spks, np.int64)
        self.enroll_audio: list[np.ndarray] = []
        self.follower = None
        self.enrolled_at: int | None = None
        self.column: int | None = None
        self.last_emb: tuple | None = None
        self.n_arm = self.n_enrolled = self.n_embed = 0
        self.embed_ms: list[float] = []

    @property
    def enrolled(self) -> bool:
        """Whether an enrollment has completed since the last arming."""
        return self.follower is not None

    def arm(self, frame: int):
        """Arm the binder at audio frame ``frame`` (the client's agent_end / enroll message)."""
        self.armed_from, self.cand, self.follower, self.last_emb = max(0, int(frame)), None, None, None
        self.run[:] = 0
        self.enroll_audio = []
        self.column = None
        self.n_arm += 1

    def _embed(self, audio_frames) -> np.ndarray:
        a = np.concatenate(audio_frames)
        t0 = time.perf_counter()
        e = self.embedder.frames(a, [np.arange(len(audio_frames))])[0]
        self.embed_ms.append((time.perf_counter() - t0) * 1000)
        self.n_embed += 1
        return e

    def _recent(self):
        """(emb (S, E), ok (S,)) of the last `win` frames' active speech per column (frames < v)."""
        on = np.array(self.on_hist, bool)  # (n, S)
        cnt = on.sum(0)
        ok = cnt >= self.min_frames
        E = getattr(self.embedder, "dim", None) or (self.last_emb[0].shape[1] if self.last_emb else 192)
        emb = np.zeros((self.S, E), np.float32)
        cols = np.nonzero(ok)[0]
        if len(cols):
            aud = list(self.audio_hist)
            segs = [np.concatenate([aud[i] for i in np.nonzero(on[:, c])[0]]) for c in cols]
            t0 = time.perf_counter()
            es = self.embedder.frames(np.concatenate(segs), [np.arange(len(s_) // self.fs) + off // self.fs
                                                              for s_, off in zip(segs, np.cumsum([0] + [len(x) for x in segs[:-1]]))])
            self.embed_ms.append((time.perf_counter() - t0) * 1000)
            self.n_embed += len(cols)
            emb[cols] = es
        return emb, ok

    def update(self, v: int, row, frame_audio) -> int | None:
        """One diarizer frame -> the primary column under the voice binding."""
        from ..enrollment import VoiceFollower
        row = np.asarray(row, np.float64)[: self.S]
        active = row > self.thr
        fa = np.asarray(frame_audio, np.float32)
        if len(fa) != self.fs:  # a flush frame past the audio: pad / crop to one frame
            fa = np.concatenate([fa, np.zeros(max(0, self.fs - len(fa)), np.float32)])[: self.fs]
        if self.follower is not None:  # 3. follow (look-back = frames < v, i.e. the histories before this frame)
            if len(self.on_hist) and (v - self.enrolled_at) % self.stride == 0:
                self.last_emb = self._recent()
            if self.last_emb is not None:
                self.column = self.follower.step(*self.last_emb)
        elif self.armed_from is not None and v >= self.armed_from:
            if self.cand is None:  # 1. the first column active for min_run consecutive frames
                self.run = np.where(active, self.run + 1, 0)
                if (self.run >= self.min_run).any():
                    self.cand = int(np.argmax(self.run >= self.min_run))
                    self.column = self.cand
                    self.enroll_audio = list(self.audio_hist)[-(self.min_run - 1):] + [fa] if self.min_run > 1 else [fa]
            elif active[self.cand]:  # 2. collect the candidate's speech
                self.enroll_audio.append(fa)
            if self.cand is not None and self.follower is None and len(self.enroll_audio) >= self.enroll_frames:
                e = self._embed(self.enroll_audio[: self.enroll_frames])
                self.follower = VoiceFollower(e, self.cand, self.margin, self.hold)
                self.enrolled_at, self.n_enrolled = v, self.n_enrolled + 1
                self.enroll_audio = []
        self.on_hist.append(active)
        self.audio_hist.append(fa)
        return self.column



class CausalDominant:
    """Streaming ``enrollment.causal_dominant_from`` (= eval_stage1.enroll_causal_dominant): bind the column with the
    most active frames (then the largest probability sum) over the last ``k`` frames, keep it, and re-bind only after
    more than ``s`` silent frames of the bound column. ``step(row, force=c0)`` binds c0 on this frame (silence count
    reset) and the rule continues from there, as ``causal_dominant_from(p, force=(c0, t))``. Returns the column or -1."""

    def __init__(self, num_spks: int = 4, k: int = 25, s: int = 25, thr: float = ACT_THRESHOLD):
        self.S, self.s, self.thr = num_spks, s, thr
        self.hist: deque = deque(maxlen=k)
        self.c, self.silent = -1, 0

    def step(self, row, force: int | None = None) -> int:
        """One diarizer frame -> the bound column (re-binds after a long silence of the bound column)."""
        p = np.asarray(row, np.float64)[: self.S]
        on = p > self.thr
        self.hist.append((on, p))
        if force is not None:
            self.c, self.silent = int(force), 0
            return self.c
        hard = np.sum([h[0] for h in self.hist], 0)
        soft = np.sum([h[1] for h in self.hist], 0)
        if self.c < 0:
            if on.any():
                self.c, self.silent = int(np.lexsort((-soft, -hard))[0]), 0
        else:
            self.silent = 0 if on[self.c] else self.silent + 1
            if self.silent > self.s:
                new = int(np.lexsort((-soft, -hard))[0])
                if new != self.c and hard[new] > 0:
                    self.c, self.silent = new, 0
        return self.c


class ArmBinder:
    """--enroll after_agent_arm (research/EOT_BENCH_V2.md section 9 "after_prev_end_causal": the agent-end choice, then
    the causal_dominant rule; no TitaNet, no embedding cost). ``arm(frame)`` on the client's agent_end; from that
    diarizer frame on, the first column active (p > thr) for ``min_run`` consecutive frames is bound
    (``enrollment.after_prev_end_choice``, decided on the run's last frame); from then on the bound column follows
    ``CausalDominant`` seeded at the choice (re-binds only after > 25 silent frames of the bound column). Before the
    first choice ``update`` returns None and the server's default 5 s dominant rule applies; a later agent_end re-arms
    and the current binding is kept until the next choice. Same interface as ``VoiceBinder`` (``enrolled`` = a
    column has been chosen since the last arm)."""

    def __init__(self, num_spks: int = 4, min_run: int = 3, thr: float = ACT_THRESHOLD, k: int = 25, s: int = 25):
        self.S, self.min_run, self.thr = num_spks, min_run, thr
        self.cd = CausalDominant(num_spks, k, s, thr)
        self.run = np.zeros(num_spks, np.int64)
        self.armed_from: int | None = None
        self.chosen: int | None = None  # the column chosen since the last arm
        self.ever = False  # a column was chosen at least once (the causal_dominant binding is then in force)
        self.column: int | None = None
        self.n_arm = self.n_enrolled = self.n_embed = 0
        self.embed_ms: list[float] = []

    @property
    def enrolled(self) -> bool:
        """Whether a column has been chosen since the last agent_end."""
        return self.chosen is not None

    def arm(self, frame: int):
        """Arm at audio frame ``frame``: the next column active for 3 frames becomes the primary."""
        self.armed_from, self.chosen = max(0, int(frame)), None
        self.run[:] = 0
        self.n_arm += 1

    def update(self, v: int, row, frame_audio=None) -> int | None:
        """One diarizer frame -> the primary column under the arm-then-follow rule."""
        row = np.asarray(row, np.float64)[: self.S]
        force = None
        if self.armed_from is not None and self.chosen is None and v >= self.armed_from:
            self.run = np.where(row > self.thr, self.run + 1, 0)
            if (self.run >= self.min_run).any():
                force = self.chosen = int(np.argmax(self.run >= self.min_run))
                self.ever, self.n_enrolled = True, self.n_enrolled + 1
        c = self.cd.step(row, force)
        self.column = (c if c >= 0 else None) if self.ever else None
        return self.column
