"""One WebSocket session to an audioforge server, as an asyncio object.

    async with AudioforgeSession("ws://127.0.0.1:8765") as s:   # turn_policy: the server default
        print(s.ready)                          # the server's ready message
        await s.send_audio(pcm16_bytes)         # any chunk size, int16 mono at s.sample_rate
        ...
        await s.end()                           # flush: the server sends the last final + stats and closes
        async for ev in s:                      # Ready, FrameEvent, Partial, TurnEnd, Final, ..., Stats
            ...

Audio is sent as soon as :meth:`send_audio` is awaited (no internal buffering); control messages
(:meth:`agent_end`, :meth:`enroll`) are therefore applied by the server at exactly the audio position sent so
far. Receiving is a background task that feeds an unbounded queue, so a slow consumer never blocks the socket
(the server's events are small). ``audio_seconds_sent`` maps the server's ``t`` to what the client has sent.
"""

from __future__ import annotations

import asyncio
import bisect
import json
import time
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Literal

from . import protocol as P
from .protocol import Event, Ready, ServerError, Stats

__all__ = ["AudioforgeSession", "AudioforgeConnectionError", "AudioforgeClosedError"]


class AudioforgeConnectionError(ConnectionError):
    """The server could not be reached, or did not start the session with a ``ready`` message."""


class AudioforgeClosedError(ConnectionError):
    """The session was closed by the server before its ``stats`` message (the server died or dropped us)."""


class AudioforgeSession:
    """A client session (see the module docstring). Use it as an async context manager or call
    :meth:`connect` / :meth:`close` yourself."""

    def __init__(
        self,
        url: str = P.DEFAULT_URL,
        *,
        turn_policy: str | None = None,
        timeout_ms: int | None = 1000,
        eot_threshold: float | None = None,
        sample_rate: int = P.DEFAULT_SAMPLE_RATE,
        turn_preset: str | None = None,
        connect_timeout: float = 10.0,
        max_size: int = 2**22,
    ) -> None:
        self.url = url
        self.config = P.config_message(
            turn_policy=turn_policy,
            timeout_ms=timeout_ms,
            eot_threshold=eot_threshold,
            sample_rate=sample_rate,
            turn_preset=turn_preset,
        )
        self.sample_rate = int(sample_rate)
        self.connect_timeout = float(connect_timeout)
        self.max_size = max_size
        self.ready: Ready | None = None
        self.stats: Stats | None = None
        self.error: BaseException | None = None
        self.bytes_sent = 0
        self.end_sent = False
        self.controls_sent: list[tuple[str, float]] = []
        """``(kind, audio seconds sent before it)`` for every agent_end / enroll message."""
        self.closed = asyncio.Event()
        self._ws: Any = None
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._sent_bytes: list[int] = []  # cumulative bytes per send
        self._sent_perf: list[float] = []  # perf_counter per send

    # ------------------------------------------------------------------ lifecycle
    async def connect(self) -> Ready:
        """Open the socket, read ``ready``, send ``config``. Raises :class:`AudioforgeConnectionError`."""
        from websockets.asyncio.client import connect

        try:
            self._ws = await asyncio.wait_for(
                connect(
                    self.url,
                    max_size=self.max_size,
                    ping_interval=None,
                    open_timeout=self.connect_timeout,
                ),
                self.connect_timeout,
            )
            first = await asyncio.wait_for(self._ws.recv(), self.connect_timeout)
        except Exception as e:  # OSError, TimeoutError, websockets exceptions
            await self._close_ws()
            raise AudioforgeConnectionError(
                f"audioforge server {self.url} unreachable: {type(e).__name__}: {e}"
            ) from e
        events = P.parse_message(first)
        if not events or not isinstance(events[0], Ready):
            await self._close_ws()
            raise AudioforgeConnectionError(
                f"audioforge server {self.url}: expected a ready message, got {first!r}"
            )
        self.ready = events[0]
        await self._ws.send(json.dumps(self.config))
        self._reader = asyncio.create_task(self._read_loop(), name="audioforge.reader")
        return self.ready

    async def __aenter__(self) -> AudioforgeSession:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def is_open(self) -> bool:
        """True while audio / control messages can still be sent."""
        return self._ws is not None and not self.end_sent and not self.closed.is_set()

    async def close(self) -> None:
        """Close the socket without waiting for the server (use :meth:`end` for a clean flush)."""
        await self._close_ws()
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        self._mark_closed()

    # ------------------------------------------------------------------ sending
    @property
    def audio_seconds_sent(self) -> float:
        """Seconds of audio sent so far (the server's ``t`` of the last sample)."""
        return P.pcm16_duration_s(self.bytes_sent, self.sample_rate)

    async def send_audio(self, pcm16: bytes) -> bool:
        """Send int16 mono PCM. Returns False (and drops the audio) once the session is closed or ended."""
        if not self.is_open or not pcm16:
            return False
        try:
            await self._ws.send(pcm16)
        except Exception as e:
            self._fail(e)
            return False
        self.bytes_sent += len(pcm16)
        self._sent_bytes.append(self.bytes_sent)
        self._sent_perf.append(time.perf_counter())
        if len(self._sent_perf) > 120000:  # ~40 min of 20 ms pushes
            del self._sent_bytes[:20000], self._sent_perf[:20000]
        return True

    def sent_perf(self, t: float) -> float | None:
        """``time.perf_counter()`` at which the audio up to time ``t`` had been sent (None if not yet)."""
        need = int(round(t * self.sample_rate)) * 2
        i = bisect.bisect_left(self._sent_bytes, need)
        if i >= len(self._sent_perf):
            return None
        return self._sent_perf[i]

    async def _send_control(
        self, kind: Literal["agent_end", "enroll"], embedding: list[float] | None = None
    ) -> bool:
        if not self.is_open:
            return False
        try:
            await self._ws.send(json.dumps(P.control_message(kind, embedding)))
        except Exception as e:
            self._fail(e)
            return False
        self.controls_sent.append((kind, self.audio_seconds_sent))
        return True

    async def agent_end(self) -> bool:
        """The agent's TTS finished: arm ``--enroll after_agent | after_agent_arm`` at this audio position."""
        return await self._send_control("agent_end")

    async def enroll(self, embedding: list[float] | None = None) -> bool:
        """``--enroll explicit``: enroll the primary speaker on the next utterance. With ``embedding`` (a stored
        192-number voice print; single-model mode) the print is used at once, under every ``--enroll`` mode."""
        return await self._send_control("enroll", embedding)

    async def end(self, wait: float | None = 10.0) -> Stats | None:
        """Send ``{"type": "end"}`` and wait up to ``wait`` seconds for the server's ``stats``.

        Returns the stats (None on timeout or if the session was already closed). The socket is closed by the
        server; events already queued stay readable through the iterator.
        """
        if self.is_open:
            try:
                await self._ws.send(json.dumps({"type": "end"}))
            except Exception as e:
                self._fail(e)
            self.end_sent = True
        if wait is not None and wait > 0:
            try:
                await asyncio.wait_for(self.closed.wait(), wait)
            except asyncio.TimeoutError:
                pass
        return self.stats

    # ------------------------------------------------------------------ receiving
    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        ev = await self._queue.get()
        if ev is None:
            self._queue.put_nowait(None)  # keep the iterator terminated for later callers
            raise StopAsyncIteration
        return ev

    async def next_event(self, timeout: float | None = None) -> Event | None:
        """The next event, or None when the session is over (or ``timeout`` elapsed)."""
        try:
            ev = await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None
        if ev is None:
            self._queue.put_nowait(None)
        return ev

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                for ev in P.parse_message(raw):
                    if isinstance(ev, Stats):
                        self.stats = ev
                    elif isinstance(ev, Ready) and self.ready is None:
                        self.ready = ev
                    self._queue.put_nowait(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # ConnectionClosedError etc.
            if not (self.end_sent and self.stats is not None):
                self.error = e
        finally:
            if self.stats is None and self.error is None and not self.end_sent:
                self.error = AudioforgeClosedError(
                    "audioforge server closed the session before stats"
                )
            if self.error is not None:
                self._queue.put_nowait(ServerError(message=str(self.error)))
            self._mark_closed()

    # ------------------------------------------------------------------ internals
    def _fail(self, e: BaseException) -> None:
        if self.error is None and not self.closed.is_set():
            self.error = e

    def _mark_closed(self) -> None:
        if not self.closed.is_set():
            self.closed.set()
            self._queue.put_nowait(None)

    async def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
