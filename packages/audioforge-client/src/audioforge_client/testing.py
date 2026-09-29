"""A scripted stand-in for ``audioforge.serve`` for unit tests (no models, no numpy, deterministic).

    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        async with AudioforgeSession(srv.url) as s:
            await s.send_audio(speech_pcm(1.0, 1.0))
            await s.end()
            events = [ev async for ev in s]

It speaks the protocol of :mod:`audioforge_client.protocol` and reacts to the audio's energy:

* one ``frame`` per 80 ms: ``vad`` 0.95 on frames whose RMS is above ``threshold`` (else 0.02), ``speakers``
  column 0 active on speech, ``primary`` 0 once anyone spoke, ``eot`` rising with the silence (0.99 from the
  third silent frame on) so the head policy fires there;
* a new word ``w<n>`` every ``frames_per_word`` speech frames, sent as ``partial`` (text since the last final);
* the configured turn policy: ``timeout`` / ``timeout_quiet`` fire when the primary has been silent for
  ``timeout_ms`` (the config's, or the constructor's default), ``head`` when ``eot`` crosses the threshold,
  ``both`` sends both, ``hybrid*`` send one event tagged with the policy at the earlier of the two; a
  ``final`` follows every turn_end of the cutting policy and the ``end`` message; ``lag_frames`` delays the
  timeout decisions like the diarizer would;
* server options: ``final_asr`` (each final gets ``source: "stream"`` and an offline twin with
  ``source: "tdt_v3"``, upper-cased text, start / end / latency_ms), ``enroll`` (``agent_end`` / ``enroll``
  messages are logged and answered with ``enrolled`` on the next speech frame), ``language`` (a ``language``
  message after ``lid_frames`` speech frames), ``batch_frames`` (frames arrive as ``frames`` batches),
  ``drop_after_s`` (the socket is closed abruptly after that much audio, code 1011).

Everything the client sent is logged (``configs``, ``controls`` as ``(connection, kind, samples before it)``,
``audio_bytes``, ``ends``), so tests can assert on ordering.
"""

from __future__ import annotations

import asyncio
import json
import math
from array import array
from typing import Any

from .protocol import (
    DEFAULT_EOT_THRESHOLD,
    FRAME_MS,
    HYBRID_POLICIES,
    POLICIES,
    POLICY_EOT_THRESHOLD,
)

__all__ = ["FakeAudioforgeServer", "speech_pcm", "silence_pcm"]

SR = 16000
FRAME_SAMPLES = SR * FRAME_MS // 1000
STATS: dict[str, Any] = {
    "type": "stats",
    "rtf": 0.1,
    "chunk_ms_p50": 1.0,
    "chunk_ms_p95": 2.0,
    "first_partial_ms": 5.0,
    "peak_rss_mb": 10.0,
}


def speech_pcm(
    seconds_speech: float,
    seconds_silence: float = 0.0,
    *,
    amplitude: float = 0.1,
    hz: float = 220.0,
    sample_rate: int = SR,
) -> bytes:
    """A tone (RMS ``amplitude / sqrt(2)``) followed by digital silence, as int16 mono PCM bytes."""
    n = int(seconds_speech * sample_rate)
    a = array(
        "h",
        (int(32767 * amplitude * math.sin(2 * math.pi * hz * i / sample_rate)) for i in range(n)),
    )
    a.extend(array("h", bytes(2 * int(seconds_silence * sample_rate))))
    return a.tobytes()


def silence_pcm(seconds: float, sample_rate: int = SR) -> bytes:
    """Digital silence as int16 mono PCM bytes."""
    return bytes(2 * int(seconds * sample_rate))


def _rms(frame: "array[int]") -> float:
    return math.sqrt(sum(x * x for x in frame) / len(frame)) / 32768.0 if len(frame) else 0.0


class FakeAudioforgeServer:
    """See the module docstring."""

    def __init__(
        self,
        *,
        timeout_ms: int = 1000,
        threshold: float = 0.01,
        frames_per_word: int = 5,
        lag_frames: int = 0,
        head_silent_frames: int = 3,
        final_asr: bool = False,
        offline_delay_frames: int = 0,
        enroll: str | None = None,
        language: str | None = None,
        lid_frames: int = 12,
        batch_frames: bool = False,
        drop_after_s: float | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.default_timeout_ms = int(timeout_ms)
        self.threshold = threshold
        self.frames_per_word = frames_per_word
        self.lag_frames = lag_frames
        self.head_silent_frames = head_silent_frames
        self.final_asr = final_asr
        self.offline_delay_frames = offline_delay_frames
        self.enroll = enroll
        self.language = language
        self.lid_frames = lid_frames
        self.batch_frames = batch_frames
        self.drop_after_s = drop_after_s
        self.host = host
        self.port: int | None = None
        self.url = ""
        # logs
        self.connections = 0
        self.configs: list[dict[str, Any]] = []
        self.controls: list[tuple[int, str, int]] = []
        self.audio_bytes: list[int] = []
        self.ends = 0
        self.sent: list[list[dict[str, Any]]] = []  # per connection, every message sent
        self._server: Any = None

    # ------------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> FakeAudioforgeServer:
        from websockets.asyncio.server import serve

        self._server = await serve(self.handler, self.host, 0, ping_interval=None).__aenter__()
        self.port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://{self.host}:{self.port}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    # ------------------------------------------------------------------ the session
    async def handler(self, ws: Any) -> None:
        idx = self.connections
        self.connections += 1
        self.audio_bytes.append(0)
        log: list[dict[str, Any]] = []
        self.sent.append(log)

        async def send(m: dict[str, Any]) -> None:
            log.append(m)
            await ws.send(json.dumps(m))

        cfg: dict[str, Any] = {
            "turn_policy": "timeout",
            "timeout_ms": self.default_timeout_ms,
            "eot_threshold": None,
            "sample_rate": SR,
        }
        st: dict[str, Any] = dict(
            buf=array("h"),
            v=0,
            silent=0,
            spoke=False,
            words=0,
            seg_words=0,
            speech=0,
            seg_speech=0,
            head_armed=False,
            prev_eot=0.0,
            fired=set(),
            pending=[],
            offline=[],
            armed=False,
            enrolled=False,
            lang_sent=False,
            dropped=False,
            seg_start_t=0.0,
        )
        ready: dict[str, Any] = {
            "type": "ready",
            "model": "fake-audioforge",
            "chunk_ms": 160,
            "frame_ms": FRAME_MS,
            "diar_config": "fake",
            "column_lag_ms": 160,
        }
        if self.enroll:
            ready.update(enroll=self.enroll, enrolled=False, primary_column=None)
        if self.final_asr:
            ready["final_asr"] = "tdt_v3"
        await send(ready)

        def policy() -> str:
            return str(cfg["turn_policy"])

        def cut() -> str:
            p = policy()
            return p if p == "head" or p in HYBRID_POLICIES else "timeout"

        def theta() -> float:
            if cfg.get("eot_threshold") is not None:
                return float(cfg["eot_threshold"])
            return POLICY_EOT_THRESHOLD.get(policy(), DEFAULT_EOT_THRESHOLD)

        def seg_text() -> str:
            return " ".join(f"w{i}" for i in range(st["words"] - st["seg_words"], st["words"]))

        async def emit_final(t: float, offline_now: bool) -> None:
            text = seg_text()
            fin: dict[str, Any] = {
                "type": "final",
                "t": t,
                "text": text,
                "speaker": 0 if st["spoke_ever"] else None,
            }
            if self.final_asr:
                fin["source"] = "stream"
                off = {
                    **fin,
                    "source": "tdt_v3",
                    "text": text.upper(),
                    "start": st["seg_start_t"],
                    "end": t,
                    "latency_ms": 1.0,
                }
                if offline_now or self.offline_delay_frames <= 0:
                    await send(fin)
                    await send(off)
                else:
                    await send(fin)
                    st["offline"].append((st["v"] + self.offline_delay_frames, off))
            else:
                await send(fin)
            st["seg_words"] = 0
            st["seg_speech"] = 0

        async def emit_due(final: bool = False) -> None:
            due = [p for p in st["pending"] if p[0] <= st["v"] or final]
            st["pending"] = [p for p in st["pending"] if p not in due]
            for _, t, pol, sil, p in due:
                await send({"type": "turn_end", "t": t, "policy": pol, "p": p, "silence_ms": sil})
                if pol == cut():
                    await emit_final(t, offline_now=final)
            due_off = [o for o in st["offline"] if o[0] <= st["v"] or final]
            st["offline"] = [o for o in st["offline"] if o not in due_off]
            for _, off in due_off:
                await send(off)

        def fire(t: float, path: str, sil_ms: int, p: float | None, delay: int) -> None:
            pol = policy()
            if pol in HYBRID_POLICIES:
                if "hybrid" in st["fired"]:
                    return  # one event per turn
                st["fired"].add("hybrid")
                st["pending"].append((st["v"] + delay, t, pol, sil_ms, p))
            else:
                st["pending"].append(
                    (st["v"] + delay, t, "timeout" if path == "timeout" else "head", sil_ms, p)
                )

        st["spoke_ever"] = False
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    d = json.loads(msg)
                except ValueError:
                    continue
                kind = d.get("type")
                if kind == "config":
                    self.configs.append(d)
                    for k in ("turn_policy", "timeout_ms", "eot_threshold", "sample_rate"):
                        if k in d:
                            cfg[k] = d[k]
                    if cfg["turn_policy"] not in POLICIES:
                        cfg["turn_policy"] = "timeout"
                elif kind in ("agent_end", "enroll"):
                    self.controls.append((idx, kind, self.audio_bytes[idx] // 2))
                    if self.enroll:
                        st["armed"] = True
                elif kind == "end":
                    self.ends += 1
                    t = round(st["v"] * FRAME_MS / 1000, 3)
                    await emit_due(final=True)
                    await emit_final(t, offline_now=True)
                    stats = dict(STATS)
                    if self.enroll:
                        stats.update(
                            enroll=self.enroll,
                            enrolled=st["enrolled"],
                            primary_column=0 if st["enrolled"] else None,
                        )
                    if self.final_asr:
                        stats.update(
                            final_asr="tdt_v3",
                            final_latency_ms={"p50": 1.0, "p95": 1.0, "max": 1.0, "n": 1},
                            final_asr_rss_mb=1.0,
                        )
                    if self.language:
                        stats["lang"] = self.language if st["lang_sent"] else None
                    await send(stats)
                    await ws.close()
                    return
                continue
            # binary audio
            if st["dropped"]:
                continue
            self.audio_bytes[idx] += len(msg)
            st["buf"].frombytes(bytes(msg[: len(msg) - len(msg) % 2]))
            if (
                self.drop_after_s is not None
                and self.audio_bytes[idx] / 2 / SR >= self.drop_after_s
            ):
                st["dropped"] = True
                asyncio.ensure_future(ws.close(code=1011))
                continue
            frames: list[dict[str, Any]] = []
            others: list[dict[str, Any]] = []
            while len(st["buf"]) >= FRAME_SAMPLES:
                fr, st["buf"] = st["buf"][:FRAME_SAMPLES], st["buf"][FRAME_SAMPLES:]
                st["v"] += 1
                t = round(st["v"] * FRAME_MS / 1000, 3)
                speech = _rms(fr) > self.threshold
                if speech:
                    if st["seg_speech"] == 0:
                        st["seg_start_t"] = round(t - FRAME_MS / 1000, 3)
                    st["silent"] = 0
                    st["spoke"] = st["spoke_ever"] = st["head_armed"] = True
                    st["fired"] = set()
                    st["speech"] += 1
                    st["seg_speech"] += 1
                    if st["speech"] % self.frames_per_word == 0:
                        st["words"] += 1
                        st["seg_words"] += 1
                        others.append({"type": "partial", "t": t, "text": seg_text()})
                    if st["armed"] and not st["enrolled"]:
                        st["enrolled"] = True
                        others.append({"type": "enrolled", "t": t, "column": 0})
                    if self.language and not st["lang_sent"] and st["speech"] >= self.lid_frames:
                        st["lang_sent"] = True
                        others.append(
                            {
                                "type": "language",
                                "t": t,
                                "language": self.language,
                                "confidence": 0.97,
                            }
                        )
                else:
                    st["silent"] += 1
                eot = 0.05 if speech else min(0.995, 0.2 + 0.3 * st["silent"])
                if not speech and st["silent"] >= self.head_silent_frames:
                    eot = 0.999
                frames.append(
                    {
                        "type": "frame",
                        "t": t,
                        "vad": 0.95 if speech else 0.02,
                        "eot": eot,
                        "speakers": [0.9 if speech else 0.02, 0.02, 0.02, 0.02],
                        "primary": 0 if st["spoke_ever"] else None,
                    }
                )
                pol = policy()
                sil_ms = st["silent"] * FRAME_MS
                if st["spoke"] and pol != "head" and sil_ms >= int(cfg["timeout_ms"]):
                    st["spoke"] = False
                    fire(
                        t,
                        "timeout",
                        sil_ms,
                        eot if pol in HYBRID_POLICIES else None,
                        self.lag_frames,
                    )
                if (
                    pol != "timeout"
                    and pol != "timeout_quiet"
                    and st["head_armed"]
                    and eot >= theta() > st["prev_eot"]
                ):
                    st["head_armed"] = False
                    fire(t, "head", sil_ms, eot, 0)
                st["prev_eot"] = eot
            if self.batch_frames and frames:
                await send(
                    {
                        "type": "frames",
                        "items": [{k: v for k, v in f.items() if k != "type"} for f in frames],
                    }
                )
            else:
                for f in frames:
                    await send(f)
            for m in others:
                await send(m)
            await emit_due()
