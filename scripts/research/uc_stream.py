"""Live two-channel predictive end-of-turn for one session (research/archive/IMPROVEMENTS.md section 1).

``UCTurnStream`` combines, on the session's audio clock:

* the user channel's audio (``feed_audio``): Silero VAD v5 on 32 ms chunks (``baselines.turn.SileroStream``) ->
  soft per-80 ms-frame user activity (the mean of the chunk probabilities whose chunk ends inside the frame, exactly
  ``bench_turnbench.frame_track``) and Pipecat's VAD state machine (confidence 0.7, start / stop 0.2 s) -> the
  silence since the last speech chunk, plus the frame's raw log-RMS for the head's energy input;
* the agent's activity (``set_agent(active, t)``): from the client's TTS timeline ({"type": "agent_start"} /
  {"type": "agent_end"} in serve.py); frame v's agent column is the state at the frame's end;
* the encoder's top-layer frames of the same user channel (``feed_enc``): the pass the streaming ASR already makes.

Frame v is run through ``uc_turn_head.UCTurnHead`` once its encoder frame, its Silero chunks and its audio are all
in. Decisions (``poll`` returns them in time order):

* ``predictive``: rising edge of P(user active in bin 1) < t1 AND P(bin 2) < t2 (research/archive/DYADIC.md section 8), or
  with ``rule="quiet"`` the rising edge of P(user quiet for 2 s) > theta; committed at the end of the 160 ms encoder
  chunk holding the frame (``emit``);
* ``silence`` (optional, ``k_s``): the first 32 ms chunk of a silence run whose silence since the last Pipecat speech
  chunk reaches k_s (one per run, after the first speech);
* the OR of both with a 2 s refractory (``bench_turnbench_latency.merged``): an event is released once both
  branches' clocks have passed it, so the online events equal the offline replica's on the same inputs
  (tests/test_uc_stream.py).
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np
import torch

SR = 16000
FRAME_SAMPLES = 1280  # 80 ms
CHUNK = 512  # 32 ms Silero chunk
FRAME_S = 0.08
EMIT_FRAMES = 2  # 160 ms encoder chunks
REFRACTORY = 2.0


class UCTurnStream:
    def __init__(self, head, silero, t1: float = 0.01, t2: float = 0.01, k_s: float | None = None,
                 rule: str = "predictive", theta: float = 0.5, refractory: float = REFRACTORY,
                 confidence: float = 0.7, start_secs: float = 0.2, stop_secs: float = 0.2):
        from audioforge.baselines.turn import PipecatVADState
        assert rule in ("predictive", "quiet"), rule
        self.head, self.silero = head, silero
        self.t1, self.t2, self.k_s, self.rule, self.theta, self.refractory = t1, t2, k_s, rule, theta, refractory
        self.vstate = silero.new_state()
        self.sm = PipecatVADState(confidence, start_secs, stop_secs)
        self.hstate = head.init_state()
        self.buf = np.zeros(0, np.float32)  # Silero chunk buffer
        self.n_chunks = 0
        self.chunk_p: list[float] = []
        self.last_sp = -1  # last Pipecat speech chunk
        self.sil_armed_run = None  # silence run (last_sp) whose event has fired
        self.rms_buf = np.zeros(0, np.float32)
        self.rms: list[float] = []  # raw log-RMS per complete frame
        self.enc: deque = deque()  # pending encoder frames (1, 1, d)
        self.enc_seen = 0
        self.v = 0  # next frame to run through the head
        self.agent_t: list[float] = []  # agent state change times (sorted) and states
        self.agent_s: list[bool] = []
        self.prev_cond = False
        self.cands: list[tuple[float, str]] = []  # candidate events (time, source), not yet released
        self.last_event = -math.inf
        self.last_out = -math.inf
        self.probs: list[dict] = []  # per frame (for diagnostics / the frame message)
        self.sil_clock = 0.0  # audio time up to which the silence branch is complete
        self.head_clock = 0.0  # emit time up to which the head branch is complete
        self.ended = False

    # ----------------------------------------------------------------- inputs
    def set_agent(self, active: bool, t: float):
        """Agent (TTS) state change at audio time t (s)."""
        i = bisect.bisect_right(self.agent_t, t)
        self.agent_t.insert(i, float(t))
        self.agent_s.insert(i, bool(active))

    def agent_at(self, t: float) -> float:
        i = bisect.bisect_right(self.agent_t, t) - 1
        return float(self.agent_s[i]) if i >= 0 else 0.0

    def feed_audio(self, x: np.ndarray):
        x = np.asarray(x, np.float32)
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= CHUNK:
            p = self.silero.step(self.vstate, self.buf[:CHUNK])
            self.buf = self.buf[CHUNK:]
            j = self.n_chunks
            self.n_chunks += 1
            self.chunk_p.append(float(p))
            self.sm.step(float(p))
            tau = (j + 1) * CHUNK / SR
            if self.sm.speaking:
                self.last_sp = j
            elif self.k_s is not None and self.last_sp >= 0 and self.sil_armed_run != self.last_sp:
                s = tau - (self.last_sp + 1) * CHUNK / SR
                if s >= self.k_s - 1e-9:
                    self.sil_armed_run = self.last_sp
                    self.cands.append((round(tau, 4), "silence"))
            self.sil_clock = tau
        self.rms_buf = np.concatenate([self.rms_buf, x])
        while len(self.rms_buf) >= FRAME_SAMPLES:
            f = self.rms_buf[:FRAME_SAMPLES].astype(np.float64)
            self.rms_buf = self.rms_buf[FRAME_SAMPLES:]
            self.rms.append(float(np.log(max(math.sqrt(float((f * f).mean())), 1e-5))))

    def feed_enc(self, enc: torch.Tensor):
        """Encoder frames (1, n, d) of the user channel, in order."""
        for j in range(enc.shape[1]):
            self.enc.append(enc[:, j: j + 1])
        self.enc_seen += enc.shape[1]

    def end(self):
        """Stream end: the partial trailing chunk / frame are never run (as the benchmark); release everything."""
        self.ended = True

    # ----------------------------------------------------------------- frames
    def _frame_ready(self, v: int) -> bool:
        last_chunk = -(-FRAME_SAMPLES * (v + 1) // CHUNK) - 1  # the chunk holding the frame end
        return bool(self.enc) and len(self.rms) > v and self.n_chunks > last_chunk

    def _user_act(self, v: int) -> float:
        # bench_turnbench.frame_track: mean of the chunks whose END falls inside frame v
        lo = int(math.floor(v * FRAME_SAMPLES / CHUNK))  # candidates
        vals = []
        for j in range(max(0, lo - 1), min(self.n_chunks, lo + 4)):
            e = (j + 1) * CHUNK
            if int((e / SR) / FRAME_S - 1e-9) == v:
                vals.append(self.chunk_p[j])
        return float(np.mean(vals)) if vals else 0.0

    def _run_frames(self):
        while self._frame_ready(self.v):
            v = self.v
            e = self.enc.popleft()
            act = np.array([[self._user_act(v), self.agent_at((v + 1) * FRAME_S - 1e-6)]], np.float32)
            p, self.hstate = self.head.step(e, act, np.array([self.rms[v]], np.float32), self.hstate)
            rec = {k: val[0].tolist() for k, val in p.items()}
            self.probs.append(rec)
            if self.rule == "predictive":
                cond = rec["user_bins"][0] < self.t1 and rec["user_bins"][1] < self.t2
            else:
                cond = rec["user_quiet"][-1] > self.theta
            emit = ((v // EMIT_FRAMES + 1) * EMIT_FRAMES) * FRAME_S
            if cond and not self.prev_cond:
                self.cands.append((round(emit, 4), "head"))
            self.prev_cond = cond
            self.v += 1
            # the head branch is complete up to the emit time of the last fully processed chunk
            if (v + 1) % EMIT_FRAMES == 0:
                self.head_clock = emit

    def poll(self) -> list[tuple[float, str, dict]]:
        """Run every ready frame; release the candidates both clocks have passed. -> [(t, source, probs at t)]."""
        self._run_frames()
        clock = math.inf if self.ended else min(self.head_clock, self.sil_clock if self.k_s is not None else math.inf)
        self.cands.sort()
        out = []
        keep = []
        for t, src in self.cands:
            if t > clock:
                keep.append((t, src))
                continue
            if t - self.last_event >= self.refractory and t > self.last_out:
                self.last_event = self.last_out = t
                fi = min(len(self.probs) - 1, max(0, int(round(t / FRAME_S)) - 1))
                out.append((t, src, self.probs[fi] if self.probs else {}))
        self.cands = keep
        return out
