"""Local Pipecat voice-agent pipeline on real meeting audio: WAV (real time) -> audioforge STT/VAD/turn -> mock LLM+TTS.

No cloud, no API keys: the only network hop is the local audioforge server (``python -m audioforge.serve``).

    # 1. the server (another shell; any free port):
    PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_heads_pretrained.afm \
        --diar runs/nemo_sortformer_v2.afm --port 8765 --threads 2
    # 2. AMI dev turn windows (WAV + label JSON used ONLY for scoring):
    PYTHONPATH=. .venv/bin/python examples/pipecat_local_demo.py prepare --n 5 --out DIR
    # 3. the pipeline, one fresh pipeline + server session per (window, policy):
    PYTHONPATH=. .venv/bin/python examples/pipecat_local_demo.py run DIR/*.wav --url ws://127.0.0.1:8765 \
        --policy timeout head --out results.json

Pipeline: ``WavInputTransport`` (20 ms InputAudioRawFrames paced at real time, then ``--pad-s`` of silence, as a live
microphone keeps streaming) -> ``AudioforgeSTTService`` -> ``TranscriptTap`` -> ``LLMUserAggregator`` (VAD =
``AudioforgeVADAnalyzer``; stop strategy = ``TurnAnalyzerUserTurnStopStrategy(AudioforgeTurnAnalyzer(policy))``;
start strategies = Pipecat's defaults, VAD + transcription) -> ``MockLLMTTS``, which logs when it WOULD start
answering (every ``UserStoppedSpeakingFrame``; the ``LLMContextFrame`` carries the user text it would answer).

Scoring (after the pipeline has finished; the labels never reach the pipeline). Audio time of a wall-clock moment
= seconds since the transport started playback (frame i, holding audio up to (i+1)*20 ms, is pushed at t0+(i+1)*20 ms).
* dead-air: first decision at or after the true end of the primary's turn (window label ``turn_end_s``) minus that
  end - the silence a caller would sit through before the bot starts (plus LLM/TTS time, not modelled here).
* interruptions: decisions while the primary's audio still follows within the next 1 s (label activity).
* early / pre_onset: other decisions before the true end (pre_onset: before the primary's first word, i.e. ending
  another speaker's turn in the window's lead-in). ``answers`` classifies the LLMContextFrame moments the same way.
* WER: all TranscriptionFrames (finals, including the end flush) vs all speakers' words in the window, both through
  ``audioforge.teachers.normalize_text``.
* with ``--llm-ms MS`` a ``MockLLMService`` (a real ``pipecat`` ``LLMService``, so its ``SpeculationGate`` applies)
  answers every ``LLMContextFrame`` after MS ms (LLM + TTS time to first audio); ``response_after_end_ms`` = the first
  released ``LLMFullResponseStartFrame`` at or after the true end, minus that end: what the caller hears. With
  ``--turn-hints`` the STT forwards the server's ``turn_end_hint`` as Pipecat's eager end of turn and the stop
  strategy is ``AudioforgeEagerTurnStopStrategy``: the reply is prepared from the hint and released at the turn end.
* Pipecat semantics: loguru WARNING/ERROR records and Python warnings from pipecat during the run, plus ordering
  checks at the tap and the mock (StartFrame first; UserStarted/UserStopped and VADUserStarted/Stopped alternate;
  finals are finalized and non-empty; nothing after EndFrame).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.services.settings import LLMSettings  # noqa: E402
from pipecat.transports.base_input import BaseInputTransport  # noqa: E402
from pipecat.transports.base_transport import TransportParams  # noqa: E402
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy  # noqa: E402
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy  # noqa: E402
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402

from audioforge.integrations.pipecat import (  # noqa: E402
    ENROLL_MODES,
    POLICIES,
    AudioforgeEagerTurnStopStrategy,
    AudioforgeHub,
    AudioforgeSTTService,
    AudioforgeTurnAnalyzer,
    AudioforgeVADAnalyzer,
)

SR = 16000
CHUNK_S = 0.02
INTERRUPT_HORIZON_S = 1.0


# ------------------------------------------------------------------------------------------ pipeline pieces
class WavInputTransport(BaseInputTransport):
    """Input transport that plays a mono 16 kHz float array in real time as 20 ms InputAudioRawFrames, followed by
    ``pad_s`` seconds of silence, then calls ``on_done``. ``t0`` (perf_counter) = the moment audio time 0 started.
    ``agent_end_s``: right after the 20 ms frame that reaches this audio time, a ``BotStoppedSpeakingFrame`` is pushed
    downstream (marked ``synthetic_agent_end``): a stand-in for the output transport's TTS-end event, which the STT
    turns into the server's agent_end (``enroll="after_agent[_arm]"``); ``TranscriptTap`` drops it after the STT."""

    def __init__(self, audio: np.ndarray, *, pad_s: float = 3.0, speed: float = 1.0, on_done=None,
                 agent_end_s: float | None = None, **kw):
        super().__init__(TransportParams(audio_in_enabled=True, audio_in_sample_rate=SR, audio_in_passthrough=True),
                         **kw)
        x = np.concatenate([np.asarray(audio, np.float32), np.zeros(int(round(pad_s * SR)), np.float32)])
        self._pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        self.audio_s = len(audio) / SR
        self.total_s = len(x) / SR
        self.speed, self.on_done = speed, on_done
        self.t0: float | None = None
        self.t_end: float | None = None
        self.max_push_late_ms = 0.0
        self._task = None
        self.agent_end_s = agent_end_s
        self.agent_end_perf: float | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        if self._task is None:
            self._task = self.create_task(self._play(), "wav_play")

    async def _play(self):
        n = int(SR * CHUNK_S) * 2
        chunks = [self._pcm[i:i + n] for i in range(0, len(self._pcm), n)]
        self.t0 = t0 = time.perf_counter()
        for i, c in enumerate(chunks):
            due = t0 + (i + 1) * CHUNK_S / self.speed
            d = due - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            self.max_push_late_ms = max(self.max_push_late_ms, (time.perf_counter() - due) * 1000)
            await self.push_audio_frame(InputAudioRawFrame(audio=c, sample_rate=SR, num_channels=1))
            if self.agent_end_s is not None and self.agent_end_perf is None and (i + 1) * CHUNK_S >= self.agent_end_s - 1e-9:
                f = BotStoppedSpeakingFrame()
                f.synthetic_agent_end = True
                self.agent_end_perf = time.perf_counter()
                await self.push_frame(f)
        self.t_end = time.perf_counter()
        if self.on_done:
            self.on_done()

    async def stop(self, frame: EndFrame):
        await self._stop_play()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._stop_play()
        await super().cancel(frame)

    async def _stop_play(self):
        if self._task is not None:
            await self.cancel_task(self._task)
            self._task = None


class _OrderChecks:
    def __init__(self, where: str):
        self.where, self.violations, self.notes = where, [], []
        self.started = self.ended = False

    def see(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self.started = True
        elif not self.started and not isinstance(frame, (CancelFrame,)):
            self.violations.append(f"{self.where}: {frame.name} before StartFrame")
        if self.ended and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMContextFrame):  # LLMUserAggregator pushes EndFrame, then flushes an open turn
                self.notes.append(f"{self.where}: session-end LLMContextFrame after EndFrame (Pipecat aggregator)")
            else:
                self.violations.append(f"{self.where}: {frame.name} after EndFrame")
        if isinstance(frame, EndFrame):
            self.ended = True


class TranscriptTap(FrameProcessor):
    """Between the STT and the user aggregator: records transcription frames (downstream) and VAD frames (the
    aggregator broadcasts them upstream) and checks their ordering."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.finals: list[dict] = []
        self.interims = 0
        self.vad: list[tuple[str, float]] = []
        self.checks = _OrderChecks("tap")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if getattr(frame, "synthetic_agent_end", False):
            return  # the demo's stand-in TTS-end event has reached the STT; the rest of the pipeline never sees it
        self.checks.see(frame, direction)
        now = time.perf_counter()
        if isinstance(frame, TranscriptionFrame):
            if not frame.finalized or not frame.text.strip():
                self.checks.violations.append(f"tap: TranscriptionFrame finalized={frame.finalized} "
                                              f"text={frame.text!r}")
            self.finals.append({"perf": now, "text": frame.text, **frame.metadata.get("audioforge", {})})
        elif isinstance(frame, InterimTranscriptionFrame):
            self.interims += 1
        elif isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            kind = "start" if isinstance(frame, VADUserStartedSpeakingFrame) else "stop"
            if self.vad and self.vad[-1][0] == kind:
                self.checks.violations.append(f"tap: two VAD {kind} frames in a row")
            if not self.vad and kind == "stop":
                self.checks.violations.append("tap: VAD stop before any start")
            self.vad.append((kind, now))
        await self.push_frame(frame, direction)


class MockLLMService(LLMService):
    """Stands in for a streaming LLM + TTS as a real ``LLMService`` (so Pipecat's ``SpeculationGate`` holds a
    speculative reply until the turn is confirmed): every ``LLMContextFrame`` is answered ``prep_s`` seconds later
    (time to the first audio) with LLMFullResponseStart / LLMText / LLMFullResponseEnd. An eager-end-of-turn cancel
    or an interruption cancels the wait (``LLMService`` restarts its processing task)."""

    def __init__(self, prep_s: float, **kw):
        super().__init__(settings=LLMSettings(
            model="mock", system_instruction=None, temperature=None, max_tokens=None, top_p=None, top_k=None,
            frequency_penalty=None, presence_penalty=None, seed=None, filter_incomplete_user_turns=False,
            user_turn_completion_config=None), **kw)
        self.prep_s = float(prep_s)
        self.runs: list[tuple[float, bool, str]] = []  # (perf, speculative, user text) per inference started
        self.done: list[tuple[float, bool]] = []  # (perf, speculative) per reply produced (held or not)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        msgs = frame.context.messages
        text = msgs[-1].get("content", "") if msgs else ""
        self.runs.append((time.perf_counter(), bool(frame.speculation), str(text)))
        await asyncio.sleep(self.prep_s)
        self.done.append((time.perf_counter(), bool(frame.speculation)))
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(f"(reply to: {str(text)[:40]})"))
        await self.push_frame(LLMFullResponseEndFrame())


class MockLLMTTS(FrameProcessor):
    """Stands in for LLM + TTS: logs when the bot WOULD start answering and records every turn frame."""

    def __init__(self, clock, **kw):
        super().__init__(**kw)
        self.clock = clock  # perf_counter -> audio time (s), or None before playback
        self.starts: list[float] = []
        self.decisions: list[float] = []
        self.contexts: list[tuple[float, str]] = []
        self.responses: list[float] = []  # LLMFullResponseStartFrame arrivals (with --llm-ms: released replies)
        self.checks = _OrderChecks("mock")
        self._in_turn = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.checks.see(frame, direction)
        now = time.perf_counter()
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, UserStartedSpeakingFrame):
                if self._in_turn:
                    self.checks.violations.append("mock: UserStartedSpeakingFrame twice")
                self._in_turn = True
                self.starts.append(now)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                if not self._in_turn:
                    self.checks.violations.append("mock: UserStoppedSpeakingFrame without a start")
                self._in_turn = False
                self.decisions.append(now)
                logger.info(f"[mock-llm] audio t={self.clock(now):7.2f}s user stopped -> WOULD start answering now")
            elif isinstance(frame, LLMContextFrame):
                msgs = frame.context.messages
                text = msgs[-1].get("content", "") if msgs else ""
                self.contexts.append((now, text if isinstance(text, str) else str(text)))
                logger.info(f"[mock-llm] audio t={self.clock(now):7.2f}s would answer: {str(text)[:90]!r}")
            elif isinstance(frame, LLMFullResponseStartFrame):
                self.responses.append(now)
                logger.info(f"[mock-llm] audio t={self.clock(now):7.2f}s reply starts (released)")
        await self.push_frame(frame, direction)


# ------------------------------------------------------------------------------------------ one run
async def run_pipeline(audio: np.ndarray, *, url: str, policy: str, pad_s: float = 3.0, timeout_ms: int = 1000,
                       eot_threshold: float | None = None, speed: float = 1.0, max_wall_s: float | None = None,
                       wait_for_silence: bool = True, resume_ms: int = 240, enroll: str | None = None,
                       agent_end_s: float | None = None, turn_hints: bool = False, llm_ms: float | None = None,
                       voiceprint: list | None = None) -> dict:
    """Run the local pipeline once over ``audio`` (16 kHz float). Returns raw timings (perf_counter based).
    ``enroll`` / ``agent_end_s``: the STT's enrollment mode and the audio time of the stand-in TTS-end event.
    ``turn_hints``: Pipecat eager end of turn on the server's turn_end_hint; ``llm_ms``: a ``MockLLMService`` with
    that time to first audio (None = no LLM, the mock only logs); ``voiceprint``: a stored print sent as ``enroll``."""
    hub = AudioforgeHub()
    stt = AudioforgeSTTService(url=url, hub=hub, timeout_ms=timeout_ms, eot_threshold=eot_threshold, enroll=enroll,
                               turn_hints=turn_hints)
    if voiceprint is not None:
        await stt.enroll(voiceprint)  # queued: sent right after the config
    vad = AudioforgeVADAnalyzer(hub)
    turn = AudioforgeTurnAnalyzer(hub, policy=policy, wait_for_silence=wait_for_silence, resume_ms=resume_ms)
    stop = (AudioforgeEagerTurnStopStrategy(turn_analyzer=turn) if turn_hints
            else TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn))
    agg = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(
        vad_analyzer=vad,
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
            stop=[stop])))
    done = asyncio.Event()
    transport = WavInputTransport(audio, pad_s=pad_s, speed=speed, on_done=done.set, agent_end_s=agent_end_s)
    clock = (lambda p: (p - transport.t0) * speed if transport.t0 is not None else float("nan"))
    tap, mock = TranscriptTap(), MockLLMTTS(clock)
    stopped: list[tuple[float, str, str | None]] = []
    watchdog: list[float] = []

    @agg.event_handler("on_user_turn_stopped")
    async def _stopped(_agg, strategy, message):
        stopped.append((time.perf_counter(), type(strategy).__name__ if strategy else "watchdog/session-end",
                        getattr(message, "content", None)))

    @agg.event_handler("on_user_turn_stop_timeout")
    async def _stop_timeout(_agg):
        watchdog.append(time.perf_counter())

    llm = MockLLMService(llm_ms / 1000.0) if llm_ms is not None else None
    worker = PipelineWorker(Pipeline([transport, stt, tap, agg] + ([llm] if llm else []) + [mock]),
                            params=PipelineParams(audio_in_sample_rate=SR), enable_rtvi=False,
                            cancel_on_idle_timeout=False, idle_timeout_secs=None)
    errors: list[str] = []

    @worker.event_handler("on_pipeline_error")
    async def _err(_w, frame):
        errors.append(str(frame))

    finished: list[str] = []

    @worker.event_handler("on_pipeline_finished")
    async def _fin(_w, frame):
        finished.append(type(frame).__name__)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    end_perf: list[float] = []

    async def ender():
        await done.wait()
        end_perf.append(time.perf_counter())
        await worker.queue_frame(EndFrame())

    budget = max_wall_s or (len(audio) / SR + pad_s) / speed + 60
    await asyncio.wait_for(asyncio.gather(runner.run(), ender()), budget)
    return {"t0": transport.t0, "audio_s": transport.audio_s, "total_s": transport.total_s, "speed": speed,
            "stream_end_perf": end_perf[0] if end_perf else None, "max_push_late_ms": transport.max_push_late_ms,
            "finals": tap.finals, "interims": tap.interims, "vad": tap.vad, "starts": mock.starts,
            "decisions": mock.decisions, "contexts": mock.contexts, "stopped": stopped, "watchdog": watchdog,
            "server_turn_ends": [e.__dict__ for e in hub.turn_ends], "turn_received": len(turn.received),
            "turn_discarded": [e.__dict__ for e in turn.discarded], "server_stats": hub.stats,
            "server_ready": hub.ready, "server_finals": hub.finals, "n_frames": hub.n_frames,
            "frame_lag_ms": [round((p - hub.sent_perf(t)) * 1000, 1) for t, p in hub.frame_arrivals
                             if hub.sent_perf(t) is not None],
            "vad_lag_frames": vad.lag_frames, "violations": tap.checks.violations + mock.checks.violations,
            "order_notes": tap.checks.notes + mock.checks.notes,
            "pipeline_errors": errors, "finished": finished, "policy": policy,
            "agent_end_s": agent_end_s, "agent_end_audio_t": clock(transport.agent_end_perf)
            if transport.agent_end_perf is not None else None, "enroll_sent": hub.enroll_sent,
            "enrolled_column": hub.enrolled_column, "turn_hints": turn_hints, "llm_ms": llm_ms,
            "responses": mock.responses, "hints": [h.__dict__ for h in hub.hints],
            "llm_runs": llm.runs if llm else [], "llm_done": llm.done if llm else [],
            "speculated": getattr(stop, "speculated", []), "spec_kept": getattr(stop, "kept", []),
            "spec_discarded": getattr(stop, "discarded", [])}


# ------------------------------------------------------------------------------------------ scoring
def _active(intervals, a: float, b: float) -> bool:
    return any(e > a + 1e-9 and s < b for s, e in intervals)


def score(raw: dict, ref: dict) -> dict:
    """Labels are used here only, after the run."""
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    t0, speed = raw["t0"], raw["speed"]
    at = (lambda p: (p - t0) * speed)
    end_cut = raw["stream_end_perf"] or float("inf")
    prim, true_end, onset = ref["primary_intervals"], ref["turn_end_s"], ref["onset_s"]
    def classify(perfs):
        rows = []
        for tau in (at(p) for p in perfs if p <= end_cut):
            kind = ("interruption" if _active(prim, tau, tau + INTERRUPT_HORIZON_S)
                    else "after_end" if tau >= true_end else "pre_onset" if tau < onset else "early")
            rows.append({"t": round(tau, 3), "kind": kind})
        return rows

    rows = classify(raw["decisions"])  # UserStoppedSpeakingFrame at the mock = "the user stopped"
    answers = classify([p for p, _ in raw["contexts"]])  # LLMContextFrame = the bot WOULD start answering
    after = [r["t"] for r in rows if r["kind"] == "after_end"]
    dead_air = round((after[0] - true_end) * 1000) if after else None
    # the server event behind the dead-air decision: the latest turn_end that had arrived by then
    srv0, pip_over = None, None
    if after:
        p_dec = t0 + after[0] / speed
        before = [e for e in raw["server_turn_ends"] if e["perf"] <= p_dec + 1e-9]
        srv0 = before[-1] if before else None
        pip_over = round((p_dec - srv0["perf"]) * 1000) if srv0 else None
    hyp = " ".join(f["text"] for f in raw["finals"])
    lags = raw["frame_lag_ms"]
    responses = classify(raw.get("responses", []))  # released replies (with --llm-ms)
    resp_after = [r["t"] for r in responses if r["kind"] == "after_end"]
    hints = [{"t": h["t"], "p": round(h["p"], 3), "outcome": h["outcome"], "arrival_t": round(at(h["perf"]), 3),
              "text": h["text"]} for h in raw.get("hints", [])]
    return {
        "dead_air_ms": dead_air,
        "no_decision_within_ms": None if after else round((raw["total_s"] - true_end) * 1000),
        "interruptions": sum(r["kind"] == "interruption" for r in rows),
        "early": sum(r["kind"] == "early" for r in rows),
        "pre_onset": sum(r["kind"] == "pre_onset" for r in rows),
        "decisions": rows,
        "answers": answers,
        "answer_interruptions": sum(r["kind"] == "interruption" for r in answers),
        "server_turn_end_t_minus_true_end_ms": round((srv0["t"] - true_end) * 1000) if srv0 else None,
        "server_turn_end_arrival_lag_ms": round((at(srv0["perf"]) - srv0["t"]) * 1000) if srv0 else None,
        "pipecat_after_server_turn_end_ms": pip_over,
        "server_turn_ends": [{"t": e["t"], "policy": e["policy"], "p": e["p"], "silence_ms": e["silence_ms"],
                              "arrival_t": round(at(e["perf"]), 3)} for e in raw["server_turn_ends"]],
        "stopped_by": [s[1] for s in raw["stopped"] if s[0] <= end_cut],
        "watchdog_stops": sum(p <= end_cut for p in raw["watchdog"]),
        "transcript": hyp,
        "wer_all_speakers": (round(float(wer([normalize_text(ref["all_text"])], [normalize_text(hyp)])), 4)
                             if ref.get("all_text") else None),
        "response_after_end_ms": round((resp_after[0] - true_end) * 1000) if resp_after else None,
        "responses": responses,
        "response_interruptions": sum(r["kind"] in ("interruption", "early") for r in responses),
        "hints": hints,
        "speculations": len(raw.get("speculated", [])), "speculations_kept": len(raw.get("spec_kept", [])),
        "speculations_discarded": [d[2] for d in raw.get("spec_discarded", [])],
        "llm_runs": [{"t": round(at(p), 3), "speculative": sp} for p, sp, _ in raw.get("llm_runs", [])],
        "n_finals": len(raw["finals"]), "n_interims": raw["interims"],
        "final_speakers": [f.get("speaker") for f in raw["finals"]],
        "vad_events": [(k, round(at(p), 2)) for k, p in raw["vad"]],
        "frame_lag_ms_p50": round(float(np.percentile(lags, 50)), 1) if lags else None,
        "frame_lag_ms_p95": round(float(np.percentile(lags, 95)), 1) if lags else None,
        "vad_lag_frames_p50": float(np.median(raw["vad_lag_frames"])) if raw["vad_lag_frames"] else None,
        "max_push_late_ms": round(raw["max_push_late_ms"], 1),
        "server_stats": raw["server_stats"],
        "agent_end_s": raw.get("agent_end_s"), "agent_end_audio_t": raw.get("agent_end_audio_t"),
        "enroll_sent": raw.get("enroll_sent"), "enrolled_column": raw.get("enrolled_column"),
        "server_ready": raw.get("server_ready"),
    }


# ------------------------------------------------------------------------------------------ AMI windows
def prepare(n: int, out: Path) -> list[str]:
    """First window of >= 15 s per AMI dev meeting in STAGE1's seeded n=200 turn set, then further such windows in
    set order (at most 2 per meeting) until ``n``. Writes <name>.wav and <name>.ref.json (labels, scoring only)."""
    import soundfile as sf

    from audioforge.datasets.ami import AMI, DEFAULT_MEETINGS, recipe_data
    out.mkdir(parents=True, exist_ok=True)
    val = recipe_data({"data": {"ami": {"mode": "turn", "val_split": "dev", "n_val": 200, "seed": 0}}}, "val")
    ds = AMI(DEFAULT_MEETINGS["dev"], audio=False, verbose=False)
    long = [(i, v) for i, v in enumerate(val) if len(v["audio"]) / SR >= 15.0]
    picked, per = [], {}
    for i, v in long:  # first per meeting
        if v["meeting"] not in per:
            picked.append((i, v))
            per[v["meeting"]] = 1
    for i, v in long:  # then fill in set order
        if len(picked) >= n:
            break
        if (i, v) not in picked and per[v["meeting"]] < 2:
            picked.append((i, v))
            per[v["meeting"]] += 1
    names = []
    for i, v in sorted(picked[:n], key=lambda iv: iv[0]):
        a, b = v["start"], v["start"] + len(v["audio"]) / SR
        words = sorted((s - a, e - a, w) for spk, ws in ds.words[v["meeting"]].items() for s, e, w in ws
                       if e > a and s < b)
        y = np.asarray(v["spk_targets"])
        prim = y[:, 0] > 0.5
        edges = np.flatnonzero(np.diff(np.r_[0, prim.astype(int), 0]))
        intervals = [(round(s * 0.08, 3), round(e * 0.08, 3)) for s, e in zip(edges[::2], edges[1::2])]
        name = f"ami_{v['meeting']}_{i:03d}"
        sf.write(out / f"{name}.wav", np.asarray(v["audio"], np.float32), SR, subtype="PCM_16")
        ref = {"name": name, "index": i, "meeting": v["meeting"], "start": round(a, 3), "dur": round(b - a, 3),
               "onset_s": round(v["onset_frame"] * 0.08, 3), "turn_end_s": round(v["turn_end_frame"] * 0.08, 3),
               "primary_intervals": intervals, "primary_text": v["text"],
               "all_text": " ".join(w for _, _, w in words),
               "n_speakers": int((y.sum(0) > 0).sum()), "note": "labels for scoring only; never fed to the pipeline"}
        (out / f"{name}.ref.json").write_text(json.dumps(ref, indent=1))
        names.append(name)
        print(json.dumps({k: ref[k] for k in ("name", "dur", "onset_s", "turn_end_s", "n_speakers")}), flush=True)
    return names


# ------------------------------------------------------------------------------------------ CLI
class _LogCapture:
    def __init__(self):
        self.records: list[dict] = []

    def __call__(self, message):
        r = message.record
        self.records.append({"level": r["level"].name, "name": r["name"], "msg": r["message"][:300]})


_STDERR_SINK: list[int] = []


def _console(level: str):
    """Replace loguru's default stderr sink (and our own from an earlier call) with one at ``level``; other sinks
    (e.g. a caller's capture) are left alone."""
    for hid in _STDERR_SINK or [0]:
        try:
            logger.remove(hid)
        except ValueError:
            pass
    _STDERR_SINK[:] = [logger.add(sys.stderr, level=level)]


def run_files(a) -> list[dict]:
    import soundfile as sf
    _console(a.log_level)
    results = []
    for wav in a.wavs:
        wav = Path(wav)
        ref_path = wav.with_suffix(".ref.json")
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(1)
        assert sr == SR, f"{wav}: {sr} Hz (prepare writes 16 kHz)"
        vp = json.loads(Path(a.voiceprint).read_text()) if getattr(a, "voiceprint", None) else None
        agent_end = getattr(a, "agent_end", None) or {}
        ae = agent_end.get(wav.stem) if isinstance(agent_end, dict) else None
        for policy in a.policy:
            cap = _LogCapture()
            hid = logger.add(cap, level="WARNING")
            print(f"=== {wav.name} policy={policy} ({len(x) / SR:.1f}s + {a.pad_s}s pad) ===", flush=True)
            with warnings.catch_warnings(record=True) as wrec:
                warnings.simplefilter("always")
                raw = asyncio.run(run_pipeline(x, url=a.url, policy=policy, pad_s=a.pad_s, timeout_ms=a.timeout_ms,
                                               eot_threshold=a.eot_threshold, speed=a.speed,
                                               wait_for_silence=not a.no_wait_for_silence, resume_ms=a.resume_ms,
                                               enroll=getattr(a, "enroll", None), agent_end_s=ae,
                                               turn_hints=getattr(a, "turn_hints", False),
                                               llm_ms=getattr(a, "llm_ms", None), voiceprint=vp))
            logger.remove(hid)
            pywarn = [f"{w.category.__name__}: {str(w.message)[:200]} ({Path(w.filename).name})" for w in wrec
                      if "pipecat" in w.filename or "integrations" in w.filename]
            res = {"file": wav.name, "policy": policy, "audio_s": raw["audio_s"], "pad_s": a.pad_s,
                   "turn_hints": raw["turn_hints"], "llm_ms": raw["llm_ms"],
                   "wait_for_silence": not a.no_wait_for_silence, "resume_ms": a.resume_ms,
                   "pipeline_finished": raw["finished"], "pipeline_errors": raw["pipeline_errors"],
                   "violations": raw["violations"], "order_notes": raw["order_notes"], "log_warnings": cap.records,
                   "py_warnings": pywarn}
            if ref_path.exists():
                ref = json.loads(ref_path.read_text())
                res.update(ref={k: ref[k] for k in ("turn_end_s", "onset_s", "n_speakers", "dur")}, **score(raw, ref))
            if a.raw_dir:
                Path(a.raw_dir).mkdir(parents=True, exist_ok=True)
                (Path(a.raw_dir) / f"{wav.stem}_{policy}.raw.json").write_text(json.dumps(raw, default=str))
            print("RESULT " + json.dumps({k: res.get(k) for k in (
                "file", "policy", "dead_air_ms", "no_decision_within_ms", "interruptions", "early", "pre_onset",
                "answer_interruptions",
                "server_turn_end_t_minus_true_end_ms", "pipecat_after_server_turn_end_ms", "wer_all_speakers",
                "turn_hints", "llm_ms", "response_after_end_ms", "response_interruptions", "speculations",
                "speculations_kept", "speculations_discarded", "stopped_by", "violations")} | {"n_warn": len(cap.records), "n_pywarn": len(pywarn)}), flush=True)
            results.append(res)
            if a.out:
                Path(a.out).write_text(json.dumps(results, indent=1, default=str))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pipecat_local_demo", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="write AMI dev turn windows (WAV + scoring labels)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", required=True)
    r = sub.add_parser("run", help="run the pipeline on WAV files (labels next to them are used for scoring)")
    r.add_argument("wavs", nargs="+")
    r.add_argument("--url", default="ws://127.0.0.1:8765")
    r.add_argument("--policy", nargs="+", choices=list(POLICIES), default=["timeout"])
    r.add_argument("--pad-s", type=float, default=3.0, help="silence streamed after the file (a live mic continues)")
    r.add_argument("--timeout-ms", type=int, default=1000)
    r.add_argument("--eot-threshold", type=float, default=None,
                   help="head threshold (default: the server's for the policy, 0.98 / 0.99828 / 0.998283)")
    r.add_argument("--enroll", choices=list(ENROLL_MODES), default=None,
                   help="the server's --enroll mode; after_agent[_arm] sends agent_end at each --agent-end time")
    r.add_argument("--agent-end", type=lambda p: json.loads(Path(p).read_text()), default=None,
                   help="JSON {window stem: audio seconds} of the stand-in TTS-end event per window (null = none)")
    r.add_argument("--speed", type=float, default=1.0, help="1 = real time")
    r.add_argument("--no-wait-for-silence", action="store_true",
                   help="turn analyzer reports COMPLETE even while Pipecat's VAD hears speech")
    r.add_argument("--resume-ms", type=int, default=240,
                   help="drop a turn_end after this much server-VAD speech past its decision time (0 = never)")
    r.add_argument("--turn-hints", action="store_true",
                   help="Pipecat eager end of turn on the server's turn_end_hint (AudioforgeEagerTurnStopStrategy)")
    r.add_argument("--llm-ms", type=float, default=None,
                   help="add a MockLLMService answering after this many ms (LLM + TTS to first audio)")
    r.add_argument("--voiceprint", default=None, help="JSON list of 192 numbers sent as the user's stored print")
    r.add_argument("--out", help="results JSON")
    r.add_argument("--raw-dir", help="also dump raw per-run timings here")
    r.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        return prepare(a.n, Path(a.out))
    return run_files(a)


if __name__ == "__main__":
    main()
