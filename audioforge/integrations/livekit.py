"""LiveKit Agents plugin for the audioforge streaming front end (``audioforge-serve``).

    from audioforge.integrations.livekit import AudioforgeFrontend
    fe = AudioforgeFrontend("ws://127.0.0.1:8765")        # ONE server session per agent session
    session = AgentSession(
        stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
        turn_handling={"turn_detection": "stt",            # our turn_end -> STT END_OF_SPEECH commits the turn
                       "endpointing": {"min_delay": 0.0},  # the server already waited timeout_ms
                       "interruption": {"mode": "vad"}})   # barge-in from our VAD frames, no cloud model

Written against livekit-agents 1.8.3 / livekit (rtc) 1.1.18. What is mapped to what:

* ``AudioforgeSTT`` (``stt.STT``, streaming, interim results, diarization) - one ``RecognizeStream`` per
  LiveKit stream, over the server's WebSocket protocol (int16 16 kHz PCM in, JSON events out):
  partial -> INTERIM_TRANSCRIPT; final -> FINAL_TRANSCRIPT with ``speaker_id="S<k>"`` (diarizer column, arrival
  order) and ``metadata["audioforge"] = {"speaker", "primary", "t", "turn_end", "eot"}``; turn_end of the cutting
  policy -> END_OF_SPEECH right after its final, with ``speech_end_time`` = wall clock of the primary's last
  active frame (deciding frame end - silence_ms; the frame end is turn_end.t - ``diar_lag_s`` unless the server sends
  the debug field frame_t; ``diar_lag_s`` defaults to 0.76 s, the 1.04 s diarizer setting's mean lag: with the
  server's default ``--diar-config low_latency_032`` set it to the ``ready`` message's ``column_lag_ms`` / 1000 =
  0.16 s; unknown ``ready`` keys are ignored); START_OF_SPEECH on the first of: ``min_speech_ms`` of VAD >= 0.5 or a
  non-empty partial. ``stats`` -> RECOGNITION_USAGE. No confidence is produced (``confidence`` = 0).
* ``AudioforgeVAD`` (``vad.VAD``) - VADEvents from the server's ``frame`` messages (one per 80 ms): INFERENCE_DONE
  per frame (``probability`` = the VAD head, ``inference_duration`` = arrival lag of that frame),
  START_OF_SPEECH after ``min_speech_duration`` above ``activation_threshold``, END_OF_SPEECH after
  ``min_silence_duration`` below it (``frames`` = the speech audio). Any-speaker VAD: it does not know who speaks.
* Turn detection. The product default is the server's **plain silence timeout on the diarizer's label-free primary
  track** (other speakers do not delay it; research/archive/STAGE1.md n=200 AMI dev turns: 38.4 % misses for the plain
  timeout with an oracle-enrolled column vs 66-68 % for the server's older "nobody else active" rule, now its
  ``timeout_quiet`` policy, and 69 % for the served head; research/archive/INTEGRATION_VERIFY.md D1; label-free enrollment
  raises all of these, research/archive/EOT_BENCH_V2.md). Two hooks:
  (a) ``turn_detection="stt"`` + ``AudioforgeSTT``: LiveKit commits the user turn on our END_OF_SPEECH
  (recommended; exactly the server's decision). (b) ``AudioforgeTurnDetector`` implements the installed
  version's *audio* turn-detector protocol (``voice.turn._StreamingTurnDetector``: ``stream()`` ->
  ``push_audio`` / ``predict() -> Future[TurnDetectionEvent]``, the interface of ``inference.TurnDetector``).
  LiveKit then ends turns on VAD END_OF_SPEECH and uses our prediction only to choose ``min_delay`` (p >=
  unlikely_threshold) or ``max_delay``: policy "timeout" resolves p = 1.0 when the server's timeout turn_end
  arrives and 0.0 on resumed speech or after ``prediction_timeout``; policy "head" resolves the turn head's
  probability (the ``eot`` field). The text-based ``_TurnDetector.predict_end_of_turn(chat_ctx)`` interface (the
  livekit-plugins-turn-detector ``MultilingualModel``) cannot take an acoustic decision and is not implemented.
  ``_StreamingTurnDetector`` is a private (underscore) protocol and may change between LiveKit releases.

Server policies ``hybrid`` / ``hybrid_silero`` / ``hybrid_dyn`` (``turn_policy=``; the head OR an any-speaker Silero
silence, research/archive/INTEGRATION.md section 8) emit one turn_end per turn tagged with the policy name, which then cuts the
finals and maps to END_OF_SPEECH like the timeout's. ``eot_threshold=None`` (default) leaves the head threshold to the
server (0.98; the hybrid_silero / hybrid_dyn frozen points 0.99828 / 0.998283). Primary enrollment (server ``--enroll
after_agent | after_agent_arm``): ``fe.agent_end()`` when the agent's TTS finished (queued behind the audio already
sent, so the server arms at that audio position) or ``fe.attach(session)`` to send it on every ``agent_state_changed``
speaking -> listening; ``fe.enroll()`` for ``--enroll explicit``.

Early end-of-turn hints (server ``turn_end_hint``, docs/PROTOCOL.md 5.11; opt-in with ``turn_hints=True``):
LiveKit's preemptive generation. Each hint becomes a ``PREFLIGHT_TRANSCRIPT`` (the text a final would hold if the
turn ended at the hint), which LiveKit 1.8 answers with a preemptive reply (``AgentActivity.on_preemptive_generation``:
LLM, and TTS with ``preemptive_tts``, run but nothing is scheduled). The server's ``turn_end`` then commits the turn
exactly as before (FINAL_TRANSCRIPT + END_OF_SPEECH under ``turn_detection="stt"``), and LiveKit plays the preemptive
reply if the committed transcript equals the preflight one, else discards it and generates again. A
``turn_end_hint_cancel`` (the user resumed) has no LiveKit counterpart and is not forwarded: the stale preemptive
reply is superseded by the next preflight or discarded at the commit, because the transcript has grown (its LLM
tokens are spent). Keep preemptive generation on (LiveKit's default)::

    fe = AudioforgeFrontend("ws://127.0.0.1:8765", turn_policy="vad_head", turn_hints=True)
    session = AgentSession(stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
                           turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.0},
                                          "interruption": {"mode": "vad"},
                                          "preemptive_generation": {"enabled": True}})

With ``turn_hints=False`` (default) the hints are only recorded (``fe.hints``) and nothing changes.

Sharing. The STT, VAD and turn detector of one ``AudioforgeFrontend`` share one WebSocket session (one server
session = one ASR + one diarizer pass); LiveKit pushes the same audio into each of them, the first stream that
pushes audio feeds the server and the others only read events. Speaker identities (diarizer columns) persist for
the life of that session; a reconnect starts a new diarizer state. Standalone ``AudioforgeSTT(url)`` etc. create
their own frontend (then each opens its own server session - do not combine standalone STT + standalone VAD).
"""
from __future__ import annotations

import asyncio
import bisect
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np
from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents._exceptions import APIConnectionError
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer, aio

logger = logging.getLogger("livekit.plugins.audioforge")

SR = 16000
FRAME_S = 0.08
DEFAULT_URL = "ws://127.0.0.1:8765"
DIAR_LAG_MAX_S = 0.96  # server docstring: a diarizer column is final 560-960 ms after its frame ends (+ compute)
DIAR_LAG_MEAN_S = 0.76
# hybrid* = the server's one-event-per-turn rules (research/archive/INTEGRATION.md section 8): hybrid_dyn / hybrid_silero =
# head OR any-speaker Silero silence; their turn_end carries the policy name and cuts the finals
POLICIES = ("timeout", "head", "both", "hybrid", "hybrid_silero", "hybrid_dyn", "vad_head")
HYBRID_POLICIES = ("hybrid", "hybrid_silero", "hybrid_dyn", "vad_head")
DEFAULT_EOT_THRESHOLD = 0.98  # the server's head threshold for "head" / "both" / "hybrid"


@dataclass
class AudioforgeOptions:
    url: str = DEFAULT_URL
    turn_policy: str = "timeout"  # server-side policy; "timeout" = plain primary-silence timeout, the product default
    timeout_ms: int = 1000
    eot_threshold: float | None = None  # None = the server's default for the policy (0.98; 0.99828 / 0.998283)
    language: str = "en"
    connect_timeout: float = 10.0
    end_timeout: float = 10.0  # how long a closing session waits for the server's final + stats
    # turn_end.t is the decision time (the deciding diarizer frame's end + 560-960 ms); without the server's
    # --debug-fields "frame_t" the primary's last active frame is estimated as t - diar_lag_s - silence_ms
    diar_lag_s: float = DIAR_LAG_MEAN_S
    # server --final-asr (research/archive/HYBRID_ASR.md): which finals are the transcript. "stream" (default): the streaming
    # finals, offline ones ignored; "offline": the offline model's finals - the streaming final of a turn is held and
    # FINAL_TRANSCRIPT + END_OF_SPEECH are emitted when the turn's offline final (same t) arrives. Servers without
    # --final-asr send no "source" and every final is used, whatever this says.
    final_source: str = "stream"
    # the server's turn_end_hint -> PREFLIGHT_TRANSCRIPT (LiveKit preemptive generation); off = hints only recorded
    turn_hints: bool = False

    def __post_init__(self):
        if self.turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}, got {self.turn_policy!r}")
        if self.final_source not in ("stream", "offline"):
            raise ValueError(f"final_source must be stream|offline, got {self.final_source!r}")

    def use_final(self, msg: dict) -> bool:
        """Whether this final message is part of the transcript under ``final_source``."""
        src = msg.get("source")
        return src is None or (src == "stream") == (self.final_source == "stream")

    @property
    def cut_policy(self) -> str:
        """The policy whose turn_end cuts finals (server rule): head for "head", the policy itself for the hybrid
        rules, else timeout."""
        return self.turn_policy if self.turn_policy in ("head",) + HYBRID_POLICIES else "timeout"


# --------------------------------------------------------------------------- audio helpers
class _PcmConverter:
    """rtc.AudioFrame (any rate / channels, int16) -> int16 mono 16 kHz numpy (LiveKit's resampler, stateful)."""

    def __init__(self):
        self.rate: int | None = None
        self.resampler: rtc.AudioResampler | None = None

    def __call__(self, frame: rtc.AudioFrame) -> np.ndarray:
        x = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            x = x.reshape(-1, frame.num_channels).mean(1).astype(np.int16)
        if self.rate is None:
            self.rate = frame.sample_rate
            if self.rate != SR:
                self.resampler = rtc.AudioResampler(self.rate, SR, num_channels=1,
                                                    quality=rtc.AudioResamplerQuality.HIGH)
        elif frame.sample_rate != self.rate:
            raise ValueError(f"sample rate changed mid-stream ({self.rate} -> {frame.sample_rate})")
        if self.resampler is None:
            return x
        out = self.resampler.push(rtc.AudioFrame(x.tobytes(), self.rate, 1, len(x)))
        return _cat(out)

    def flush(self) -> np.ndarray:
        return _cat(self.resampler.flush()) if self.resampler is not None else np.zeros(0, np.int16)


def _cat(frames) -> np.ndarray:
    parts = [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    return np.concatenate(parts) if parts else np.zeros(0, np.int16)


# --------------------------------------------------------------------------- one server session
class _Link:
    """One WebSocket session to audioforge.serve: audio from a single feeder, events fanned out to subscribers.

    ``wall_of(t)`` maps the server's audio time t (seconds of audio received) to the wall-clock time at which
    that sample was pushed (= captured, for live audio)."""

    def __init__(self, opts: AudioforgeOptions):
        self.opts = opts
        self.id = uuid.uuid4().hex[:12]
        self.ws = None
        self.ready: dict | None = None
        self.stats: dict | None = None
        self.subs: list[asyncio.Queue] = []
        self.feeder: object | None = None
        self.samples = 0  # 16 kHz samples sent
        self._cum: list[int] = []
        self._wall: list[float] = []
        self._conv = _PcmConverter()
        self._outq: asyncio.Queue = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self.closed = asyncio.Event()
        self.error: BaseException | None = None
        self.errors: list[dict] = []  # the server's 'error' messages (audioforge.serve.error_msg)
        self.end_sent = False
        self.pending_controls: list = []  # "agent_end" / "enroll" or a full message dict (enroll with an embedding)
        self.controls_sent: list[tuple[str, float]] = []  # (type, audio seconds sent before it)

    async def open(self) -> None:
        from websockets.asyncio.client import connect
        t = self.opts.connect_timeout
        try:
            self.ws = await asyncio.wait_for(connect(self.opts.url, max_size=2 ** 22, ping_interval=None,
                                                     open_timeout=t), t)
            self.ready = json.loads(await asyncio.wait_for(self.ws.recv(), t))
        except Exception as e:  # OSError, TimeoutError, websockets errors, bad JSON
            if self.ws is not None:
                await self.ws.close()
            raise APIConnectionError(f"audioforge server {self.opts.url} unreachable: {type(e).__name__}: {e}") \
                from e
        if not isinstance(self.ready, dict) or self.ready.get("type") != "ready":
            await self.ws.close()
            raise APIConnectionError(f"audioforge server {self.opts.url}: expected a ready message, got {self.ready}")
        # the server's diarizer preset sets the column lag; prefer its own figure over the option default
        lag_ms = self.ready.get("column_lag_ms")
        self._diar_lag_s = float(lag_ms) / 1000.0 if isinstance(lag_ms, (int, float)) else float(self.opts.diar_lag_s)
        cfg = {"type": "config", "turn_policy": self.opts.turn_policy, "timeout_ms": int(self.opts.timeout_ms),
               "sample_rate": SR}
        if self.opts.eot_threshold is not None:
            cfg["eot_threshold"] = float(self.opts.eot_threshold)
        await self.ws.send(json.dumps(cfg))
        for c in self.pending_controls:  # agent_end / enroll requested before the session existed (audio position 0)
            await self.ws.send(json.dumps(c if isinstance(c, dict) else {"type": c}))
        self.pending_controls = []
        self._tasks = [asyncio.create_task(self._reader(), name="audioforge.reader"),
                       asyncio.create_task(self._writer(), name="audioforge.writer")]

    # ------------------------------------------------------------ subscribers
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        if self.closed.is_set():
            q.put_nowait(None)
        self.subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subs:
            self.subs.remove(q)

    # ------------------------------------------------------------ audio in
    def push(self, frame: rtc.AudioFrame, src: object) -> bool:
        """Send ``frame`` if ``src`` is (or becomes) the feeder. Returns whether it was sent."""
        if self.end_sent or self.closed.is_set():
            return False
        if self.feeder is None:
            self.feeder = src
        if src is not self.feeder:
            return False
        self._send_pcm(self._conv(frame))
        return True

    def _send_pcm(self, pcm: np.ndarray) -> None:
        if not len(pcm):
            return
        self.samples += len(pcm)
        self._cum.append(self.samples)
        self._wall.append(time.time())
        if len(self._cum) > 60000:  # keep ~20 min of 20 ms pushes
            del self._cum[:10000], self._wall[:10000]
        self._outq.put_nowait(pcm.astype("<i2").tobytes())

    def send_control(self, typ: str, extra: dict | None = None) -> bool:
        """Queue a control message (``agent_end`` / ``enroll``) behind the audio already pushed, so the server applies
        it at exactly this audio position. Returns whether it was queued."""
        if self.end_sent or self.closed.is_set():
            return False
        self._outq.put_nowait(json.dumps({"type": typ, **(extra or {})}))
        self.controls_sent.append((typ, self.audio_s))
        return True

    def finish_audio(self) -> None:
        """Send {"type": "end"}: the server flushes, emits the last final + stats and closes."""
        if self.end_sent or self.closed.is_set():
            return
        self._send_pcm(self._conv.flush())
        self.end_sent = True
        self._outq.put_nowait(json.dumps({"type": "end"}))

    def wall_of(self, t: float) -> float:
        k = t * SR
        if not self._cum:
            return time.time()
        i = bisect.bisect_left(self._cum, k)
        if i >= len(self._cum):
            return self._wall[-1] + (k - self._cum[-1]) / SR
        return self._wall[i] - (self._cum[i] - k) / SR

    @property
    def audio_s(self) -> float:
        return self.samples / SR

    # ------------------------------------------------------------ tasks
    async def _reader(self):
        try:
            async for raw in self.ws:
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                now = time.time()
                items = ([{**it, "type": "frame"} for it in msg.get("items", [])] if msg.get("type") == "frames"
                         else [msg])
                for m in items:
                    if m.get("type") == "stats":
                        self.stats = m
                    elif m.get("type") == "error":  # the server's structured degradation / failure report
                        self.errors.append(m)
                        if m.get("fatal"):
                            logger.error("audioforge session %s: fatal server error %s: %s", self.id, m.get("code"),
                                         m.get("detail"))
                            if self.error is None:
                                self.error = ConnectionError(f"audioforge server error {m.get('code')}: "
                                                             f"{m.get('detail')}")
                        else:
                            logger.warning("audioforge session %s: server %s: %s", self.id, m.get("code"),
                                           m.get("detail"))
                    for q in list(self.subs):
                        q.put_nowait((m, now))
        except Exception as e:  # ConnectionClosedError etc.
            if not (self.end_sent and self.stats is not None):
                self.error = e
        finally:
            if self.stats is None and self.error is None and not self.end_sent:
                self.error = ConnectionError("audioforge server closed the session")
            self.closed.set()
            for q in list(self.subs):
                q.put_nowait(None)

    async def _writer(self):
        try:
            while True:
                item = await self._outq.get()
                await self.ws.send(item)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self.error is None and not self.closed.is_set():
                self.error = e

    async def end(self, wait: float | None = None) -> dict | None:
        """Finish the audio, wait for the server's stats (or ``wait`` s), close. Returns the stats message."""
        self.finish_audio()
        try:
            await asyncio.wait_for(self.closed.wait(), self.opts.end_timeout if wait is None else wait)
        except asyncio.TimeoutError:
            logger.warning("audioforge session %s: no stats after end; closing", self.id)
        await self.aclose()
        return self.stats

    async def aclose(self):
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        await aio.cancel_and_wait(*self._tasks)
        self.closed.set()


class AudioforgeFrontend:
    """Factory for the STT / VAD / turn detector that share one audioforge server session (see module docstring).

    The session opens when the first stream starts and ends (``{"type": "end"}``) when the last one closes."""

    def __init__(self, url: str = DEFAULT_URL, *, turn_policy: str = "timeout", timeout_ms: int = 1000,
                 eot_threshold: float | None = None, language: str = "en", connect_timeout: float = 10.0,
                 end_timeout: float = 10.0, final_source: str = "stream", turn_hints: bool = False):
        self.opts = AudioforgeOptions(url=url, turn_policy=turn_policy, timeout_ms=timeout_ms,
                                      eot_threshold=eot_threshold, language=language,
                                      connect_timeout=connect_timeout, end_timeout=end_timeout,
                                      final_source=final_source, turn_hints=turn_hints)
        self._link: _Link | None = None
        self._owners: set = set()
        self._lock: asyncio.Lock | None = None
        self._bg: set[asyncio.Task] = set()
        self.sessions_opened = 0
        self.last_stats: dict | None = None
        self.turn_ends: list[dict] = []  # every server turn_end seen by an STT stream, + "arrival" wall time
        self.hints: list[dict] = []  # every turn_end_hint / turn_end_hint_cancel seen by an STT stream, + "arrival"
        self.controls: list[tuple[str, float]] = []  # agent_end / enroll sent: (type, audio seconds before it)
        self._pending_controls: list = []

    # factories
    def stt(self, **kw) -> "AudioforgeSTT":
        return AudioforgeSTT(frontend=self, **kw)

    def vad(self, **kw) -> "AudioforgeVAD":
        return AudioforgeVAD(frontend=self, **kw)

    def turn_detector(self, **kw) -> "AudioforgeTurnDetector":
        return AudioforgeTurnDetector(frontend=self, **kw)

    @property
    def link(self) -> _Link | None:
        return self._link

    # server --enroll triggers
    def agent_end(self) -> bool:
        """The agent's TTS finished: send {"type": "agent_end"} (server ``--enroll after_agent | after_agent_arm``),
        queued behind the audio already sent so the server arms at this audio position. Before the session exists it
        is sent right after the session's config (audio position 0). Synchronous: safe from event callbacks."""
        return self._control("agent_end")

    def enroll(self, embedding=None) -> bool:
        """Server ``--enroll explicit``: enroll the primary on the next utterance (after prompting the user). With
        ``embedding`` (the user's stored 192-number voice print; single-model mode, the server default) the print is
        used at once, under every ``--enroll`` mode."""
        extra = None if embedding is None else {"embedding": [float(x) for x in embedding]}
        return self._control("enroll", extra)

    def _control(self, typ: str, extra: dict | None = None) -> bool:
        lk = self._link
        if lk is not None and not lk.closed.is_set() and not lk.end_sent and lk.ws is not None:
            ok = lk.send_control(typ, extra)
            if ok:
                self.controls.append((typ, lk.audio_s))
            return ok
        self._pending_controls.append({"type": typ, **extra} if extra else typ)
        self.controls.append((typ, 0.0))
        return True

    def attach(self, session) -> None:
        """Send agent_end whenever the AgentSession's agent stops speaking (``agent_state_changed`` speaking -> any
        other state): the product hook for ``--enroll after_agent | after_agent_arm``."""

        @session.on("agent_state_changed")
        def _on_agent_state(ev):
            if getattr(ev, "old_state", None) == "speaking" and getattr(ev, "new_state", None) != "speaking":
                self.agent_end()

    async def _acquire(self, owner) -> tuple[_Link, asyncio.Queue]:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            lk = self._link
            if lk is None or lk.closed.is_set() or lk.end_sent:
                lk = _Link(self.opts)
                lk.pending_controls, self._pending_controls = self._pending_controls, []
                await lk.open()
                self._link, self._owners = lk, set()
                self.sessions_opened += 1
                logger.info("audioforge session %s opened (%s)", lk.id, lk.ready)
            self._owners.add(owner)
            return lk, lk.subscribe()

    def _release(self, owner, lk: _Link, q: asyncio.Queue | None) -> None:
        """Detach ``owner``; when it was the last one, end the session in the background (never blocks)."""
        if q is not None:
            lk.unsubscribe(q)
        if self._link is lk:
            self._owners.discard(owner)
            if self._owners:
                return
            self._link = None
        t = asyncio.get_running_loop().create_task(self._end(lk))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _end(self, lk: _Link):
        st = await lk.end()
        if st is not None:
            self.last_stats = st

    async def aclose(self):
        if self._link is not None:
            lk, self._link = self._link, None
            await self._end(lk)
        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)


# --------------------------------------------------------------------------- STT
class _SpeechMapper:
    """Server messages -> LiveKit SpeechEvents (pure state machine; unit-tested without a socket).

    ``offset`` = RecognizeStream.start_time_offset (seconds from LiveKit's audio-input start to this stream's start);
    SpeechData start/end times are ``offset + server audio time``."""

    def __init__(self, opts: AudioforgeOptions, min_speech_ms: int = 160, vad_threshold: float = 0.5):
        self.opts = opts
        self.min_frames = max(1, int(round(min_speech_ms / (FRAME_S * 1000))))
        self.vad_thr = vad_threshold
        self.in_speech = False
        self.vad_run = 0
        self.seg_start_t = 0.0
        self.primary: int | None = None
        self.eot: float | None = None
        self.pending_turn_end: dict | None = None
        self.other_turn_end: dict | None = None
        self.turn_end_by_t: dict[float, dict] = {}  # final_source "offline": the cutting turn_end of each held turn
        self.last_interim = ""
        self.request_id = ""
        self.hint: dict | None = None  # the outstanding turn_end_hint

    def _ev(self, typ, **kw) -> stt.SpeechEvent:
        return stt.SpeechEvent(type=typ, request_id=self.request_id, **kw)

    def _data(self, text, t_end, offset, meta=None, speaker=None) -> stt.SpeechData:
        return stt.SpeechData(language=self.opts.language, text=text, start_time=offset + self.seg_start_t,
                              end_time=offset + t_end, confidence=0.0,
                              speaker_id=f"S{speaker}" if speaker is not None else None,
                              metadata={"audioforge": meta} if meta is not None else None)

    def _start(self, onset_t: float, wall_of) -> stt.SpeechEvent:
        self.in_speech = True
        self.seg_start_t = max(0.0, onset_t)
        return self._ev(stt.SpeechEventType.START_OF_SPEECH, speech_start_time=wall_of(self.seg_start_t))

    def on_message(self, msg: dict, wall_of, offset: float = 0.0, audio_s: float = 0.0) -> list[stt.SpeechEvent]:
        typ, out = msg.get("type"), []
        if typ == "frame":
            self.primary, self.eot = msg.get("primary"), msg.get("eot")
            self.vad_run = self.vad_run + 1 if msg.get("vad", 0.0) >= self.vad_thr else 0
            if not self.in_speech and self.vad_run >= self.min_frames:
                out.append(self._start(msg["t"] - self.vad_run * FRAME_S, wall_of))
        elif typ == "partial":
            text = msg.get("text", "").strip()
            if text and not self.in_speech:
                out.append(self._start(msg["t"] - FRAME_S * 2, wall_of))
            if text and text != self.last_interim:
                self.last_interim = text
                out.append(self._ev(stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                    alternatives=[self._data(text, msg["t"], offset,
                                                             speaker=self.primary)]))
        elif typ == "turn_end_hint":
            text = (msg.get("text") or "").strip()
            self.hint = msg
            if self.opts.turn_hints and text:
                if not self.in_speech:
                    out.append(self._start(msg["t"] - FRAME_S, wall_of))
                meta = {"kind": "hint", "t": msg["t"], "p": msg.get("p"), "speaker": self.primary}
                out.append(self._ev(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                                    alternatives=[self._data(text, msg["t"], offset, meta, self.primary)]))
        elif typ == "turn_end_hint_cancel":
            self.hint = None  # no LiveKit counterpart: the preemptive reply is superseded or discarded at commit
        elif typ == "turn_end":
            self.hint = None
            if msg.get("policy") == self.opts.cut_policy:
                self.pending_turn_end = msg
                self.turn_end_by_t[msg["t"]] = msg
            else:  # "both": the non-cutting policy's decision is reported in the next final's metadata
                self.other_turn_end = msg
        elif typ == "final":
            if not self.opts.use_final(msg):  # --final-asr: the other source's final of this turn
                return out
            te = self.pending_turn_end
            if msg.get("source") not in (None, "stream"):  # an offline final: its own turn's decision
                te = self.turn_end_by_t.pop(msg["t"], te)
            text, spk = msg.get("text", "").strip(), msg.get("speaker")
            meta = {"speaker": spk, "primary": self.primary, "t": msg["t"], "turn_end": te, "eot": self.eot}
            if self.other_turn_end is not None:
                meta["other_turn_end"], self.other_turn_end = self.other_turn_end, None
            if text:
                if not self.in_speech:
                    out.append(self._start(msg["t"] - FRAME_S, wall_of))
                out.append(self._ev(stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[self._data(text, msg["t"], offset, meta, spk)]))
            if self.in_speech:
                last_speech_t = (te.get("frame_t", te["t"] - getattr(self, "_diar_lag_s", self.opts.diar_lag_s)) - te["silence_ms"] / 1000.0
                                 if te is not None else msg["t"])
                out.append(self._ev(stt.SpeechEventType.END_OF_SPEECH,
                                    speech_end_time=wall_of(max(self.seg_start_t, last_speech_t))))
            self.in_speech, self.pending_turn_end, self.vad_run, self.last_interim = False, None, 0, ""
        elif typ == "stats":
            out.append(self._ev(stt.SpeechEventType.RECOGNITION_USAGE,
                                recognition_usage=stt.RecognitionUsage(audio_duration=float(audio_s))))
        return out


class AudioforgeSTT(stt.STT):
    def __init__(self, url: str = DEFAULT_URL, *, frontend: AudioforgeFrontend | None = None,
                 min_speech_ms: int = 160, **frontend_kw):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True, diarization=True,
                                                          aligned_transcript=False, offline_recognize=True))
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kw)
        self._min_speech_ms = min_speech_ms

    @property
    def frontend(self) -> AudioforgeFrontend:
        return self._fe

    @property
    def model(self) -> str:
        lk = self._fe.link
        return (lk.ready or {}).get("model", "audioforge") if lk is not None else "audioforge"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "AudioforgeSpeechStream":
        return AudioforgeSpeechStream(stt=self, conn_options=conn_options)

    async def _recognize_impl(self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        """Batch recognition: a private server session, all audio at once, the finals joined."""
        from livekit.agents.utils import merge_frames
        frame = merge_frames(buffer) if isinstance(buffer, list) else buffer
        lk = _Link(self._fe.opts)
        await lk.open()
        q = lk.subscribe()
        lk.push(frame, self)
        lk.finish_audio()
        finals, spk = [], None
        while (item := await q.get()) is not None:
            m, _ = item
            if m["type"] == "final" and m["text"].strip() and self._fe.opts.use_final(m):
                finals.append(m["text"].strip())
                spk = m["speaker"] if spk is None else spk
            if m["type"] == "stats":
                break
        await lk.aclose()
        if lk.error is not None and lk.stats is None:
            raise APIConnectionError(f"audioforge recognize failed: {lk.error}")
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, request_id=lk.id, alternatives=[
            stt.SpeechData(language=self._fe.opts.language, text=" ".join(finals), start_time=0.0,
                           end_time=lk.audio_s, speaker_id=f"S{spk}" if spk is not None else None)])


class AudioforgeSpeechStream(stt.RecognizeStream):
    def __init__(self, *, stt: AudioforgeSTT, conn_options: APIConnectOptions):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SR)
        self._af = stt

    async def _run(self) -> None:
        fe = self._af._fe
        lk, q = await fe._acquire(self)
        mapper = _SpeechMapper(fe.opts, self._af._min_speech_ms)
        mapper.request_id = lk.id
        fwd = asyncio.create_task(self._forward(lk), name="audioforge.stt.forward")
        try:
            while (item := await q.get()) is not None:
                msg, arrived = item
                if msg.get("type") == "turn_end":
                    fe.turn_ends.append({**msg, "arrival": arrived})
                elif msg.get("type") in ("turn_end_hint", "turn_end_hint_cancel"):
                    fe.hints.append({**msg, "arrival": arrived})
                for ev in mapper.on_message(msg, lk.wall_of, self.start_time_offset, lk.audio_s):
                    self._event_ch.send_nowait(ev)
                if msg.get("type") == "stats":
                    break
            if lk.error is not None and lk.stats is None:
                raise APIConnectionError(f"audioforge session {lk.id} lost: {lk.error}", retryable=True)
        finally:
            await aio.cancel_and_wait(fwd)
            fe._release(self, lk, q)

    async def _forward(self, lk: _Link) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                continue  # the server has no segment flush: turn ends are decided server-side
            lk.push(item, self)
        if lk.feeder is self or lk.feeder is None:
            lk.finish_audio()  # input ended -> server flushes, sends the last final + stats


# --------------------------------------------------------------------------- VAD
class _VadMapper:
    """``frame`` messages -> VADEvents (silero-like hysteresis on the server's 80 ms VAD probabilities)."""

    def __init__(self, activation_threshold=0.5, min_speech_duration=0.16, min_silence_duration=0.56):
        self.thr, self.min_speech, self.min_silence = activation_threshold, min_speech_duration, min_silence_duration
        self.reset()

    def reset(self):
        self.speaking = False
        self.speech_run = 0.0  # consecutive speech (raw)
        self.silence_run = 0.0  # consecutive silence (raw)
        self.speech_dur = 0.0  # published speech segment duration
        self.onset_t = 0.0

    def on_frame(self, msg: dict, now: float, wall_of) -> list[tuple[vad.VADEventType, dict]]:
        p, t = float(msg.get("vad", 0.0)), float(msg["t"])
        lag = max(0.0, now - wall_of(t))
        out = []
        if p >= self.thr:
            self.speech_run += FRAME_S
            self.silence_run = 0.0
        else:
            self.silence_run += FRAME_S
            self.speech_run = 0.0
        if self.speaking:
            self.speech_dur += FRAME_S
            if self.silence_run >= self.min_silence - 1e-9:
                out.append((vad.VADEventType.END_OF_SPEECH,
                            dict(speech_duration=self.speech_dur - self.silence_run,
                                 silence_duration=self.silence_run, span=(self.onset_t, t - self.silence_run))))
                self.speaking, self.speech_dur = False, 0.0
        elif self.speech_run >= self.min_speech - 1e-9:
            self.speaking, self.speech_dur, self.onset_t = True, self.speech_run, t - self.speech_run
            out.append((vad.VADEventType.START_OF_SPEECH,
                        dict(speech_duration=self.speech_dur, silence_duration=0.0, span=(self.onset_t, t))))
        out.append((vad.VADEventType.INFERENCE_DONE,
                    dict(probability=p, speaking=self.speaking,
                         speech_duration=self.speech_dur if self.speaking else 0.0,
                         silence_duration=self.silence_run)))
        return [(typ, {**kw, "t": t, "lag": lag, "raw_speech": self.speech_run, "raw_silence": self.silence_run})
                for typ, kw in out]


class AudioforgeVAD(vad.VAD):
    def __init__(self, url: str = DEFAULT_URL, *, frontend: AudioforgeFrontend | None = None,
                 activation_threshold: float = 0.5, min_speech_duration: float = 0.16,
                 min_silence_duration: float = 0.56, keep_audio_s: float = 60.0, **frontend_kw):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=FRAME_S))
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kw)
        self.activation_threshold = activation_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration  # read by LiveKit's audio turn-detector check (>= 0.25 s)
        self.keep_audio_s = keep_audio_s

    @property
    def model(self) -> str:
        return "audioforge-vad-head"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(self) -> "AudioforgeVADStream":
        return AudioforgeVADStream(self)


class AudioforgeVADStream(vad.VADStream):
    def __init__(self, v: AudioforgeVAD):
        self._af = v
        self._mapper = _VadMapper(v.activation_threshold, v.min_speech_duration, v.min_silence_duration)
        self._audio: deque = deque()  # (start s, frame) of this stream's own input
        self._pushed_s = 0.0
        super().__init__(v)

    async def _main_task(self) -> None:
        fe = self._af._fe
        attempts = 0
        while True:
            lk, q = await fe._acquire(self)
            fwd = asyncio.create_task(self._forward(lk), name="audioforge.vad.forward")
            try:
                while (item := await q.get()) is not None:
                    msg, now = item
                    if msg.get("type") == "frame":
                        for typ, kw in self._mapper.on_frame(msg, now, lk.wall_of):
                            self._event_ch.send_nowait(self._event(typ, kw))
                    elif msg.get("type") == "stats":
                        return
            finally:
                await aio.cancel_and_wait(fwd)
                fe._release(self, lk, q)
            if self._input_ch.closed or lk.error is None or attempts >= 3:
                if lk.error is not None and not self._input_ch.closed:
                    raise APIConnectionError(f"audioforge session {lk.id} lost: {lk.error}")
                return
            attempts += 1
            logger.warning("audioforge VAD: session %s lost (%s); reconnecting", lk.id, lk.error)
            self._mapper.reset()
            await asyncio.sleep(0.5 * attempts)

    async def _forward(self, lk: _Link) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._mapper.reset()  # hard segment boundary (VADStream.flush contract)
                continue
            lk.push(item, self)
            self._audio.append((self._pushed_s, item))
            self._pushed_s += item.duration
            while self._audio and self._audio[0][0] < self._pushed_s - self._af.keep_audio_s:
                self._audio.popleft()
        if lk.feeder is self or lk.feeder is None:
            lk.finish_audio()

    def _frames(self, t0: float, t1: float) -> list[rtc.AudioFrame]:
        return [f for s, f in self._audio if s + f.duration > t0 and s < t1]

    def _event(self, typ: vad.VADEventType, kw: dict) -> vad.VADEvent:
        frames = self._frames(*kw["span"]) if "span" in kw else []
        return vad.VADEvent(type=typ, samples_index=int(round(kw["t"] * SR)), timestamp=kw["t"],
                            speech_duration=kw["speech_duration"], silence_duration=kw["silence_duration"],
                            frames=frames, probability=kw.get("probability", 0.0), inference_duration=kw["lag"],
                            speaking=kw.get("speaking", typ == vad.VADEventType.START_OF_SPEECH),
                            raw_accumulated_silence=kw["raw_silence"], raw_accumulated_speech=kw["raw_speech"])


# --------------------------------------------------------------------------- turn detection (audio protocol)
class AudioforgeTurnDetector:
    """``turn_detection=`` object for AgentSession implementing ``voice.turn._StreamingTurnDetector``.

    policy "timeout" (default, the product policy): p = 1.0 when the server's timeout turn_end arrives, 0.0 on
    resumed speech or after ``prediction_timeout`` (-> LiveKit waits ``max_delay``). policy "head": p = the turn
    head's ``eot`` for the first frame at/after the prediction request (opt-in; STAGE1 n=200: 69 % misses vs 38.4 %
    for the plain timeout, oracle-enrolled)."""

    def __init__(self, url: str = DEFAULT_URL, *, frontend: AudioforgeFrontend | None = None,
                 policy: str = "timeout", unlikely_threshold: float | None = None,
                 prediction_timeout: float | None = None, **frontend_kw):
        if policy not in ("timeout", "head"):
            raise ValueError("policy must be 'timeout' or 'head'")
        self._fe = frontend or AudioforgeFrontend(url, **frontend_kw)
        self.policy = policy
        if policy == "timeout" and self._fe.opts.turn_policy == "head":
            raise ValueError("policy 'timeout' needs the server's timeout turn_end: frontend turn_policy "
                             "'timeout' or 'both'")
        self._unlikely = (unlikely_threshold if unlikely_threshold is not None
                          else 0.5 if policy == "timeout" else (self._fe.opts.eot_threshold or DEFAULT_EOT_THRESHOLD))
        # the server decides >= timeout_ms after the primary's last active frame, plus the diarizer's lag + compute
        self._pred_timeout = (prediction_timeout if prediction_timeout is not None
                              else self._fe.opts.timeout_ms / 1000 + DIAR_LAG_MAX_S + 0.8)
        self.predictions: list[dict] = []  # every settled request of every stream: {"p", "why", "wall"}

    @property
    def model(self) -> str:
        return f"audioforge-{self.policy}"

    @property
    def provider(self) -> str:
        return "audioforge"

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "AudioforgeTurnStream":
        return AudioforgeTurnStream(self)


class AudioforgeTurnStream:
    """``voice.turn._StreamingTurnDetectorStream`` over the shared server session."""

    def __init__(self, det: AudioforgeTurnDetector):
        from livekit.agents.voice.turn import TurnDetectionEvent
        self._Ev = TurnDetectionEvent
        self._det = det
        self._fut: asyncio.Future | None = None
        self._deadline: asyncio.TimerHandle | None = None
        self._armed_t: float | None = None  # decision time of a timeout turn_end that arrived with no request pending
        self._last_voiced_t = -1.0  # latest server frame with VAD >= 0.5 (a decision is stale once speech follows it)
        self._pred_t = 0.0
        self._link: _Link | None = None
        self._q: asyncio.Queue | None = None
        self._early: list[rtc.AudioFrame] = []  # audio pushed before the session is up
        self._ended = False
        self.predictions: list[dict] = []  # log (for demos/tests)
        self._task = asyncio.create_task(self._run(), name="audioforge.turn")

    # protocol properties
    @property
    def model(self) -> str:
        return self._det.model

    @property
    def provider(self) -> str:
        return "audioforge"

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return self._det._pred_timeout

    async def unlikely_threshold(self, language) -> float | None:
        return self._det._unlikely

    async def backchannel_threshold(self, language) -> float | None:
        return None

    async def supports_language(self, language) -> bool:
        return True  # acoustic decision (the ASR itself is English-only)

    # requests
    def predict(self) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        self._settle(0.0, "superseded")
        fut = self._fut = loop.create_future()
        self._pred_t = self._link.audio_s if self._link is not None else 0.0
        if self._ended:
            self._settle(1.0, "closed")
        elif self.policy == "timeout" and self._armed_t is not None and self._last_voiced_t <= self._armed_t:
            self._settle(1.0, "turn_end_before_request", detection_delay=0.0)
        else:
            self._deadline = loop.call_later(max(0.05, self.prediction_timeout - 0.05), self._settle, 0.0,
                                             "deadline")
        return fut

    @property
    def policy(self) -> str:
        return self._det.policy

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        self._armed_t = None
        self._settle(0.0, "cancelled")

    def flush(self, reason: str | None = None) -> None:
        self._armed_t = None
        self._settle(0.0, f"flush:{reason}")

    def _settle(self, p: float, why: str, detection_delay: float | None = None) -> None:
        if self._deadline is not None:
            self._deadline.cancel()
            self._deadline = None
        fut, self._fut = self._fut, None
        if fut is not None and not fut.done():
            fut.set_result(self._Ev(type="eot_prediction", end_of_turn_probability=float(p),
                                    last_speaking_time=time.time(), detection_delay=detection_delay))
            rec = {"p": float(p), "why": why, "wall": time.time()}
            self.predictions.append(rec)
            self._det.predictions.append(rec)
            logger.debug("audioforge turn prediction p=%.3f (%s)", p, why)

    # audio
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._link is not None:
            self._link.push(frame, self)
        elif not self._ended:
            self._early.append(frame)
            if len(self._early) > 500:
                self._early.pop(0)

    def end_input(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._link is not None and self._link.feeder is self:
            self._link.finish_audio()
        self._settle(1.0, "closed")

    async def aclose(self) -> None:
        self.end_input()
        await aio.cancel_and_wait(self._task)

    async def _run(self):
        fe = self._det._fe
        try:
            lk, q = await fe._acquire(self)
        except Exception as e:
            logger.error("audioforge turn detector: no server session (%s); predictions fall back to the "
                         "deadline (p=0 -> max_delay)", e)
            return
        self._link, self._q = lk, q
        for f in self._early:
            lk.push(f, self)
        self._early.clear()
        try:
            while (item := await q.get()) is not None:
                msg, now = item
                typ = msg.get("type")
                if typ == "frame":
                    if msg.get("vad", 0.0) >= 0.5:
                        self._last_voiced_t = max(self._last_voiced_t, float(msg["t"]))
                    if self.policy == "head" and self._fut is not None and msg["t"] >= self._pred_t:
                        self._settle(msg.get("eot") or 0.0, "head", detection_delay=now - lk.wall_of(msg["t"]))
                elif typ == "turn_end" and msg.get("policy") == self.policy == "timeout":
                    t_dec = float(msg["t"])
                    if self._last_voiced_t > t_dec:
                        continue  # speech after the decision: it is about an earlier pause, not the current one
                    if self._fut is not None:
                        self._settle(1.0, "turn_end", detection_delay=now - lk.wall_of(t_dec))
                    else:
                        self._armed_t = t_dec
                elif typ == "stats":
                    break
        finally:
            fe._release(self, lk, q)
