"""Pipecat adapters for the audioforge streaming front end (``audioforge-serve``). Tested with pipecat-ai 1.12.0.

One WebSocket session per pipeline, shared by three adapters through an ``AudioforgeHub``:

* ``AudioforgeSTTService`` (``WebsocketSTTService``): streams every ``InputAudioRawFrame`` (int16 PCM) to the server;
  ``partial`` events become ``InterimTranscriptionFrame`` and ``final`` events ``TranscriptionFrame(finalized=True)``.
  ``frame.metadata["audioforge"]`` = {"t", "speaker", "kind"}: for finals ``speaker`` is the server's primary at the
  turn end; for interims it is the latest ``primary`` of the frame stream (the diarizer lags the audio by
  ``ready.column_lag_ms`` + compute: ~0.2 s with the server's default 0.32 s setting, ~0.86 s with ``low_latency``).
  ``EndFrame`` sends {"type": "end"} and waits up to ``end_timeout`` for the flush final and the stats message, so the
  last TranscriptionFrame is pushed before the EndFrame. ``supports_ttfs`` is False: the server decides the turn
  boundaries and sends each final together with its turn_end, so there is no separate speech-end-to-final wait.
* ``AudioforgeVADAnalyzer`` (``VADAnalyzer``): the confidence of each 80 ms step is the server's VAD probability
  (``frame.vad``), and Pipecat's own start/stop hysteresis (``VADParams.confidence / start_secs / stop_secs``) runs
  on top of it unchanged. The probability comes back over the socket, so the value used for the audio just delivered
  is the newest one the server has sent: about 1-2 frames (80-160 ms at 1x) older than that audio.
  ``VADParams.min_volume`` is 0 here unless set explicitly (the learned VAD decides; Pipecat's loudness gate would
  make the result depend on the recording level and adds a 0.4 s warm-up).
* ``AudioforgeTurnAnalyzer`` (``BaseTurnAnalyzer``; plug into ``TurnAnalyzerUserTurnStopStrategy``): end of turn from
  the server's ``turn_end`` events. ``policy``: ``timeout`` (default) | ``head`` | ``both`` (either event) |
  ``hybrid`` | ``hybrid_silero`` | ``hybrid_dyn`` (the server's head-OR-silence rules, one event per turn). It asks
  the server for the same policy (``config.turn_policy``), so the finals are cut where this analyzer ends turns.
  A received turn_end latches until Pipecat ends the turn (``clear()``). With ``wait_for_silence`` (default) the
  analyzer reports COMPLETE only while Pipecat's VAD says the user is not speaking: Pipecat starts LLM inference
  the moment an analyzer reports COMPLETE, but its ``UserTurnController`` refuses to end the turn while the VAD hears
  speech, so without the gate the bot would start answering while someone is still audible. The latch is dropped
  once the server's VAD has reported speech (p >= 0.5) on ``resume_ms`` (default 240 ms = 3 frames) of audio after
  the event's decision time ``t``: someone kept or started talking after the decision, so it is stale (the analogue
  of Pipecat discarding a pending end of turn when the user resumes). This rule runs on the server's frame clock,
  so it does not depend on socket timing, and it also catches speech that continues across the decision, which
  never produces a new VAD start edge in Pipecat.

Default turn policy: **timeout** - the server's plain silence timeout (1000 ms) on the diarizer's label-free primary
track; other speakers do not delay it. The server's older rule that also waits for "nobody else active" is its
``timeout_quiet`` policy (direct protocol only, not offered here): at n=200 AMI dev turns that rule misses 66-68 %
at <= 5 % false cutoffs vs 38.4 % for the plain timeout (research/STAGE1.md, oracle-enrolled column;
research/INTEGRATION_VERIFY.md D1), and the served head misses 69 %. With label-free enrollment every streaming
system misses more (research/EOT_BENCH_V2.md). The head's probability is exposed as an optional signal (``hub.eot``
on every frame event, ``policy="head"`` / ``"both"``) and is not the default.

Pipecat-side semantics to know (not changed here): ``UserTurnController`` never ends a turn while its VAD says the
user is speaking, and "the user" is whoever is audible on the input stream. On multi-party audio (a meeting replay)
someone else talking right after the server's turn_end therefore either holds the turn until the VAD stops (under
``resume_ms`` of speech after the decision) or makes the decision stale (more), and the turn then waits for the
server's next turn_end. Measured on AMI with examples/pipecat_local_demo.py.

Primary-speaker enrollment (server ``--enroll after_agent | after_agent_arm | explicit``, research/EOT_BENCH_V2.md
section 9; ``after_agent_arm`` = the same agent_end message, the server then follows the chosen column with its
causal_dominant rule instead of TitaNet):
``AudioforgeSTTService(enroll="after_agent")`` sends {"type": "agent_end"} to the server whenever Pipecat's output
transport reports ``BotStoppedSpeakingFrame`` (it pushes that frame upstream through the pipeline, so the STT
service sees it): the server then binds the primary to the first speaker after the bot's TTS and follows that voice.
``enroll="explicit"`` sends nothing by itself; call ``await stt.enroll()`` when the user was asked to speak. The
server's ``enrolled`` event sets ``hub.enrolled_column``. Default ``enroll=None``: nothing is sent (the server's
default binding).

Usage::

    hub = AudioforgeHub()
    stt = AudioforgeSTTService(url="ws://127.0.0.1:8765", hub=hub)
    user = LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(
        vad_analyzer=AudioforgeVADAnalyzer(hub),
        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=AudioforgeTurnAnalyzer(hub, policy="timeout"))])))
    Pipeline([transport.input(), stt, user, llm, tts, transport.output()])
"""
from __future__ import annotations

import asyncio
import bisect
import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from websockets.protocol import State

# hybrid* = the server's one-event-per-turn rules (research/INTEGRATION.md section 8): hybrid_dyn / hybrid_silero =
# head OR any-speaker Silero silence (dynamic / 2.64 s wait); their turn_end carries the policy name
POLICIES = ("timeout", "head", "both", "hybrid", "hybrid_silero", "hybrid_dyn")
ENROLL_MODES = ("after_agent", "after_agent_arm", "explicit")  # the server's --enroll modes (None = its default)
AGENT_END_MODES = ("after_agent", "after_agent_arm")  # modes armed by {"type": "agent_end"}
FRAME_MS = 80  # the server's frame clock (one "frame" event per 80 ms of audio)
NUM_SPKS = 4


@dataclass
class TurnEnd:
    """A server ``turn_end`` event plus its arrival time (``perf`` = time.perf_counter(), ``wall`` = time.time())."""

    t: float
    policy: str
    p: float | None
    silence_ms: int
    perf: float
    wall: float


class AudioforgeHub:
    """Per-session state shared by the STT service (which owns the socket) and the two analyzers.

    Plain attributes, written only from the event loop (the STT's receive task); the VAD analyzer reads ``vad`` from
    Pipecat's analyzer thread (a single float read). ``policy`` is the turn policy the STT asks the server for when
    the STT was not given one explicitly (the turn analyzer sets it)."""

    def __init__(self, policy: str | None = None):
        self.policy = policy
        self.final_source = "stream"  # set by AudioforgeSTTService(final_source=...)
        self._turn_listeners: list[Callable[[TurnEnd], None]] = []
        self._frame_listeners: list[Callable[[float, float], None]] = []
        self.reset()

    def reset(self):
        """Forget the session state (a new server session starts at audio time 0). Listeners are kept."""
        self.ready: dict | None = None
        self.stats: dict | None = None
        self.vad = 0.0
        self.eot: float | None = None
        self.speakers = [0.0] * NUM_SPKS
        self.primary: int | None = None
        self.frame_t = 0.0
        self.n_frames = 0
        self.frame_arrivals: list[tuple[float, float]] = []  # (t, perf) per frame event
        self.frame_vad: list[tuple[float, float]] = []  # (t, vad) per frame event
        self.turn_ends: list[TurnEnd] = []
        self.finals: list[dict] = []
        self.offline_final_t: set[float] = set()  # t of the offline finals received (server --final-asr)
        self.partials = 0
        self.enrolled_column: int | None = None  # the server's 'enrolled' event (--enroll)
        self.errors: list[dict] = []  # the server's 'error' messages (degradations; a fatal one ends the session)
        self.enroll_sent = 0  # agent_end / enroll messages sent
        self.voiceprints: list[dict] = []  # the server's 'voiceprint' events (--turn-input tsvad)
        self._sent = [0]  # cumulative samples sent, at the pipeline rate
        self._sent_perf = [0.0]
        self.sample_rate = 16000

    def add_turn_listener(self, fn: Callable[[TurnEnd], None]):
        self._turn_listeners.append(fn)

    def add_frame_listener(self, fn: Callable[[float, float], None]):
        """fn(t, vad) on every frame event."""
        self._frame_listeners.append(fn)

    # ------------------------------------------------------------------ fed by the STT service
    def note_sent(self, n_samples: int, sample_rate: int, perf: float | None = None):
        self.sample_rate = sample_rate
        self._sent.append(self._sent[-1] + n_samples)
        self._sent_perf.append(time.perf_counter() if perf is None else perf)

    def sent_perf(self, t: float) -> float | None:
        """perf_counter time at which the audio up to ``t`` seconds had been sent (None if not sent yet)."""
        i = bisect.bisect_left(self._sent, int(round(t * self.sample_rate)))
        return self._sent_perf[i] if 0 < i < len(self._sent) else None

    def on_frame(self, f: dict, perf: float | None = None):
        self.vad = float(f["vad"])
        self.eot = f.get("eot")
        self.speakers = list(f.get("speakers") or self.speakers)
        self.primary = f.get("primary")
        self.frame_t = float(f["t"])
        self.n_frames += 1
        self.frame_arrivals.append((self.frame_t, time.perf_counter() if perf is None else perf))
        self.frame_vad.append((self.frame_t, self.vad))
        for fn in list(self._frame_listeners):
            fn(self.frame_t, self.vad)

    def on_turn_end(self, m: dict, perf: float | None = None) -> TurnEnd:
        ev = TurnEnd(float(m["t"]), str(m["policy"]), m.get("p"), int(m.get("silence_ms", 0)),
                     time.perf_counter() if perf is None else perf, time.time())
        self.turn_ends.append(ev)
        for fn in list(self._turn_listeners):
            fn(ev)
        return ev


class AudioforgeSTTService(WebsocketSTTService):
    """Streaming STT on the audioforge server. See the module docstring for the frames it pushes."""

    def __init__(self, *, url: str = "ws://127.0.0.1:8765", hub: AudioforgeHub | None = None,
                 turn_policy: str | None = None, timeout_ms: int = 1000, eot_threshold: float | None = None,
                 end_timeout: float = 5.0, enroll: str | None = None, final_source: str = "stream", **kwargs):
        """
        Args:
            url: the audioforge server (``python -m audioforge.serve``).
            enroll: None (server default) | "after_agent" / "after_agent_arm" (send agent_end at every
                BotStoppedSpeakingFrame) | "explicit" (send enroll on ``await stt.enroll()``); the server must run
                with the same --enroll.
            hub: shared state for the analyzers (created if None; see ``create_vad_analyzer`` / ``create_turn_analyzer``).
            turn_policy: policy requested from the server; None = the turn analyzer's policy, else "timeout".
            timeout_ms / eot_threshold: the server's timeout-policy silence and head threshold (None = the server's
                default for the policy: 0.98, hybrid_silero 0.99828, hybrid_dyn 0.998283).
            end_timeout: seconds to wait for the flush final + stats after sending {"type": "end"} on EndFrame.
            final_source: with a server running --final-asr (research/HYBRID_ASR.md): "stream" (default) pushes
                the streaming finals as TranscriptionFrames and ignores the offline ones; "offline" pushes the offline
                model's finals instead, and the turn analyzer reports COMPLETE only once the turn's offline final
                (same t) has arrived, so the LLM sees the offline transcript. Finals without "source" (server without
                the flag) are always pushed.
        """
        kwargs.setdefault("settings", STTSettings(model="audioforge", language=None))
        super().__init__(**kwargs)
        if turn_policy is not None and turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}")
        if enroll is not None and enroll not in ENROLL_MODES:
            raise ValueError(f"enroll must be None or one of {ENROLL_MODES}")
        self._enroll = enroll
        self._url = url
        self.hub = hub or AudioforgeHub()
        if final_source not in ("stream", "offline"):
            raise ValueError("final_source must be stream|offline")
        self.hub.final_source = final_source
        self._turn_policy = turn_policy
        self._timeout_ms = int(timeout_ms)
        self._eot_threshold = None if eot_threshold is None else float(eot_threshold)
        self._end_timeout = float(end_timeout)
        self._receive_task: asyncio.Task | None = None
        self._stats_event: asyncio.Event | None = None
        self._pending_controls: list = []  # agent_end / enroll requested before the first connection

    # ------------------------------------------------------------------ convenience
    def create_vad_analyzer(self, **kw) -> "AudioforgeVADAnalyzer":
        return AudioforgeVADAnalyzer(self.hub, **kw)

    def create_turn_analyzer(self, **kw) -> "AudioforgeTurnAnalyzer":
        return AudioforgeTurnAnalyzer(self.hub, **kw)

    @property
    def turn_policy(self) -> str:
        return self._turn_policy or self.hub.policy or "timeout"

    @property
    def supports_ttfs(self) -> bool:
        return False  # server-defined turn boundaries: each final arrives with its turn_end

    def can_generate_metrics(self) -> bool:
        return False

    # ------------------------------------------------------------------ lifecycle
    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self.hub.policy and self._turn_policy and self.hub.policy != self._turn_policy \
                and self._turn_policy != "both":
            logger.warning(f"{self}: server turn_policy {self._turn_policy!r} will not emit the "
                           f"{self.hub.policy!r} turn_end events the turn analyzer waits for")
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._flush_and_wait()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def _flush_and_wait(self):
        """Send {"type": "end"} and wait for the server's flush final and stats (it then closes the socket)."""
        ws = self._websocket
        if ws is None or ws.state is not State.OPEN:
            return
        self._disconnecting = True  # the server closes after "stats": that close is not an error to reconnect on
        try:
            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(self._stats_event.wait(), self._end_timeout)
        except TimeoutError:
            logger.warning(f"{self}: no stats message within {self._end_timeout}s after end")
        except Exception as e:  # noqa: BLE001 - a failed flush must not block pipeline shutdown
            logger.warning(f"{self}: end flush failed: {e}")

    # ------------------------------------------------------------------ enrollment (server --enroll)
    async def _send_control(self, typ: str, extra: dict | None = None) -> bool:
        ws = self._websocket
        msg = {"type": typ, **(extra or {})}
        if ws is None or ws.state is not State.OPEN:  # not connected (yet): sent right after the next config
            self._pending_controls.append(msg if extra else typ)
            return True
        try:
            await ws.send(json.dumps(msg))
            self.hub.enroll_sent += 1
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: {typ} send failed: {e}")
            return False

    async def enroll(self, embedding=None) -> bool:
        """enroll="explicit": ask the server to enroll the primary on the next utterance (after prompting the user);
        with ``embedding`` (server --turn-input tsvad: a stored 192-d voice print, audioforge.tsvad_stream.voiceprint)
        the print is used at once."""
        if self._enroll != "explicit" and embedding is None:  # a stored print is taken under every mode (single mode)
            logger.warning(f"{self}: enroll() called with enroll={self._enroll!r}; nothing sent")
            return False
        extra = None if embedding is None else {"embedding": [float(x) for x in embedding]}
        return await self._send_control("enroll", extra)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # the bot's TTS ended (the output transport pushes BotStoppedSpeakingFrame upstream): with enroll="after_agent[_arm]"
        # the server now binds the primary to the next speaker
        if isinstance(frame, BotStoppedSpeakingFrame) and self._enroll in AGENT_END_MODES:
            await self._send_control("agent_end")
        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        ws = self._websocket
        if ws is not None and ws.state is State.OPEN:
            try:
                await ws.send(audio)
                self.hub.note_sent(len(audio) // 2, self.sample_rate)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{self}: send failed: {e}")
        yield None

    # ------------------------------------------------------------------ websocket
    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self.hub.reset()
            self._stats_event = asyncio.Event()
            self._websocket = await self._websocket_connect(self._url, max_size=2 ** 22, ping_interval=None)
            cfg = {"type": "config", "turn_policy": self.turn_policy, "timeout_ms": self._timeout_ms,
                   "sample_rate": self.sample_rate}
            if self._eot_threshold is not None:
                cfg["eot_threshold"] = self._eot_threshold
            await self._websocket.send(json.dumps(cfg))
            for m in self._pending_controls:
                await self._websocket.send(json.dumps(m if isinstance(m, dict) else {"type": m}))
                self.hub.enroll_sent += 1
            self._pending_controls = []
            logger.debug(f"{self}: connected to {self._url} (turn_policy {self.turn_policy})")
            await self._call_event_handler("on_connected")
        except Exception as e:  # noqa: BLE001
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to audioforge at {self._url}: {e}", exception=e)

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: error closing websocket: {e}")
        finally:
            if self._websocket is not None:
                self._websocket = None
                await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        async for raw in self._websocket:
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = json.loads(raw)
            except ValueError:
                logger.warning(f"{self}: non-JSON message ignored")
                continue
            await self.handle_server_message(msg)

    async def handle_server_message(self, msg: dict):
        """Map one server message to hub updates and Pipecat frames (public so tests can feed canned events)."""
        typ = msg.get("type")
        if typ == "frame":
            self.hub.on_frame(msg)
        elif typ == "frames":
            now = time.perf_counter()
            for it in msg.get("items", []):
                self.hub.on_frame(it, now)
        elif typ == "partial":
            self.hub.partials += 1
            text = (msg.get("text") or "").strip()
            if text:
                f = InterimTranscriptionFrame(text=text, user_id=self._user_id, timestamp=time_now_iso8601(),
                                              result=msg)
                f.metadata["audioforge"] = {"t": msg.get("t"), "speaker": self.hub.primary, "kind": "partial"}
                await self.push_frame(f)
        elif typ == "turn_end":
            self.hub.on_turn_end(msg)
        elif typ == "final":
            self.hub.finals.append(msg)
            src = msg.get("source")
            if src not in (None, "stream"):
                self.hub.offline_final_t.add(float(msg["t"]))
            use = src is None or (src == "stream") == (self.hub.final_source == "stream")
            text = (msg.get("text") or "").strip()
            if text and use:
                f = TranscriptionFrame(text=text, user_id=self._user_id, timestamp=time_now_iso8601(), result=msg,
                                       finalized=True)
                f.metadata["audioforge"] = {"t": msg.get("t"), "speaker": msg.get("speaker"), "kind": "final"}
                await self.push_frame(f)
        elif typ == "ready":
            self.hub.ready = msg
            if self._enroll and msg.get("enroll") != self._enroll:
                logger.warning(f"{self}: enroll={self._enroll!r} but the server runs --enroll "
                               f"{msg.get('enroll', 'dominant')!r}; its trigger messages will be ignored")
        elif typ == "enrolled":
            self.hub.enrolled_column = msg.get("column")
        elif typ == "voiceprint":  # server --turn-input tsvad: a new TS-VAD voice print is in use
            self.hub.voiceprints.append(msg)
        elif typ == "stats":
            self.hub.stats = msg
            if self._stats_event is not None:
                self._stats_event.set()
        elif typ == "error":  # structured degradation / failure report (audioforge.serve.error_msg)
            self.hub.errors.append(msg)
            text = f"{self}: audioforge server error {msg.get('code')}: {msg.get('detail')}"
            if msg.get("fatal"):
                logger.error(text)
                await self.push_error(error_msg=text)
            else:
                logger.warning(text)
        else:
            logger.warning(f"{self}: unknown server message type {typ!r}")


class AudioforgeVADAnalyzer(VADAnalyzer):
    """VAD confidence = the server's per-frame VAD probability; Pipecat's hysteresis on top (module docstring)."""

    def __init__(self, hub: AudioforgeHub, *, sample_rate: int | None = None, params: VADParams | None = None):
        if params is None:
            params = VADParams(min_volume=0.0)
        elif "min_volume" not in params.model_fields_set:  # keep the loudness gate off unless asked for
            params = params.model_copy(update={"min_volume": 0.0})
        super().__init__(sample_rate=sample_rate, params=params)
        self._hub = hub
        self.steps = 0
        self.lag_frames: list[int] = []  # per step: audio frames delivered - server frames received

    def num_frames_required(self) -> int:
        return int((self.sample_rate or 16000) * FRAME_MS / 1000)  # one server frame per analyzer step

    def voice_confidence(self, buffer: bytes) -> float:
        self.steps += 1
        self.lag_frames.append(self.steps - self._hub.n_frames)
        return float(self._hub.vad)


class AudioforgeTurnParams(BaseTurnParams):
    policy: str = "timeout"
    wait_for_silence: bool = True
    resume_ms: int = 240


class AudioforgeTurnAnalyzer(BaseTurnAnalyzer):
    """End of turn from the server's turn_end events (see the module docstring for the latch semantics)."""

    def __init__(self, hub: AudioforgeHub, *, policy: str = "timeout", wait_for_silence: bool = True,
                 resume_ms: int = 240, sample_rate: int | None = None):
        super().__init__(sample_rate=sample_rate)
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self._hub = hub
        if hub.policy is None:
            hub.policy = policy
        self._params = AudioforgeTurnParams(policy=policy, wait_for_silence=wait_for_silence, resume_ms=resume_ms)
        self._pending: TurnEnd | None = None
        self._resume_frames = 0
        self._speech = False
        self.received: list[TurnEnd] = []  # events of this analyzer's policy
        self.discarded: list[TurnEnd] = []  # latched events dropped because speech followed the decision
        hub.add_turn_listener(self._on_turn_end)
        hub.add_frame_listener(self._on_frame)

    def _on_turn_end(self, ev: TurnEnd):
        if self._params.policy in ("both", ev.policy):
            self.received.append(ev)
            self._pending = ev
            self._resume_frames = 0
            for t, vad in reversed(self._hub.frame_vad):  # frames of the same batch can precede the event
                if t <= ev.t + 1e-6:
                    break
                self._resume_frames += vad >= 0.5
            self._maybe_discard()

    def _on_frame(self, t: float, vad: float):
        if self._pending is not None and t > self._pending.t + 1e-6 and vad >= 0.5:
            self._resume_frames += 1
            self._maybe_discard()

    def _maybe_discard(self):
        if self._pending is not None and self._params.resume_ms > 0 \
                and self._resume_frames * FRAME_MS >= self._params.resume_ms:
            logger.debug(f"{self}: speech after turn_end t={self._pending.t}; dropping it")
            self.discarded.append(self._pending)
            self._pending = None

    @property
    def pending(self) -> TurnEnd | None:
        return self._pending

    @property
    def speech_triggered(self) -> bool:
        return self._speech

    @property
    def params(self) -> AudioforgeTurnParams:
        return self._params

    def _final_ready(self) -> bool:
        """final_source "offline" with a --final-asr server: the pending turn's offline final has arrived."""
        h = self._hub
        if h.final_source == "stream" or not (h.ready or {}).get("final_asr"):
            return True
        return self._pending is not None and any(abs(t - self._pending.t) < 1e-6 for t in h.offline_final_t)

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        if is_speech:
            self._speech = True
        if self._pending is None or (is_speech and self._params.wait_for_silence) or not self._final_ready():
            return EndOfTurnState.INCOMPLETE
        return EndOfTurnState.COMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Called by the strategy when Pipecat's VAD stops (and after a COMPLETE from ``append_audio``)."""
        ok = self._pending is not None and self._final_ready()
        return (EndOfTurnState.COMPLETE if ok else EndOfTurnState.INCOMPLETE), None

    def clear(self):
        self._pending = None
        self._speech = False
