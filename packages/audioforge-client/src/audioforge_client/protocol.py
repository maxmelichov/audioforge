"""The audioforge WebSocket protocol (``audioforge/serve.py``, docs/PROTOCOL.md), as typed messages.

Client -> server
    binary frames: int16 little-endian PCM, mono, ``sample_rate`` Hz (16 kHz unless configured), any size.
    ``{"type": "config", ...}`` optional, sent first: ``turn_policy`` (omitted = the server's own default,
        ``vad_head`` in single-model mode), ``turn_preset``, ``timeout_ms``, ``eot_threshold``, ``sample_rate``.
        ``sample_rate``, ``turn_preset`` and the Silero policies are only honoured before the first audio.
    ``{"type": "agent_end"}`` / ``{"type": "enroll"}``: arm the server's user enrollment at the audio position
        received so far. ``enroll`` may carry ``"embedding"``: a stored 192-number voice print (single-model mode),
        taken at once.
    ``{"type": "end"}``: flush; the server sends the last ``final`` and ``stats`` and closes.

Server -> client (JSON text frames, one message each; ``frame`` events may arrive batched as
``{"type": "frames", "items": [...]}``, which :func:`parse_message` expands):
    ``ready``      once, after the connection: model, chunk_ms, frame_ms, diar_config, column_lag_ms
                   (+ enroll / final_asr / final_chunk_ms keys when those server options are on).
    ``frame``      one per 80 ms of audio: t, vad (0..1), eot (turn head, may be null), speakers (activity
                   columns), primary (column index or null).
    ``partial``    the transcript since the last final, whenever it changes.
    ``turn_end_hint`` / ``turn_end_hint_cancel``
                   an early guess that the turn is ending (prepare a reply, do not speak) and its withdrawal.
    ``turn_end``   an end-of-turn decision: t (decision time), policy, p, silence_ms (+ hinted_at, path).
    ``final``      the turn's transcript, after every turn_end of the cutting policy and at the end of the stream.
                   With ``--final-asr`` / ``--asr-lookahead`` / ``--final-chunk-ms`` every final carries
                   ``source``; with ``--final-chunk-ms`` the streaming final is sent first as ``final_fast`` and
                   the ``final`` of the same ``t`` (``source: "slow"``) carries the slower, better text.
    ``enrolled``   (``--enroll``) the primary was bound to a diarizer column.
    ``voiceprint`` (single-model mode) the session follows a new voice print; store ``embedding`` for next time.
    ``language``   (``--lid``) the spoken language changed or was first decided.
    ``error``      a structured problem report: code, detail, fatal (fatal -> the server closes next).
    ``stats``      once, after the end final; the server then closes the socket.

``t`` is audio time in seconds since the start of the session (audio received, not wall clock).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Union

FRAME_MS = 80
"""The server's frame clock: one ``frame`` event per 80 ms of audio."""

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_URL = "ws://127.0.0.1:8765"
NUM_SPEAKERS = 4
"""Speaker-activity columns in every ``frame.speakers`` list."""

POLICIES: tuple[str, ...] = (
    "vad_head",
    "timeout",
    "timeout_quiet",
    "timeout_any",
    "head",
    "both",
    "hybrid",
    "hybrid_silero",
    "hybrid_dyn",
)
"""Server turn policies (the ``config.turn_policy`` values).

``vad_head``      the served VAD head quiet for K frames AND the turn head p >= ``eot_threshold``, OR a silence
                  fallback, OR (with the user's voice print) the user silent while someone else talks. One event
                  per turn, tagged ``vad_head``. The single-model server's default; ``turn_preset`` picks its
                  constants.
``timeout``       silence of the primary speaker >= ``timeout_ms``.
``timeout_quiet`` the same, but it also waits until no other speaker is active (tagged ``timeout``).
``timeout_any``   multi-party rooms: every speaker's turn ends (tagged ``timeout`` or ``change``).
``head``          the turn head's probability crosses ``eot_threshold``.
``both``          timeout and head events both fire, each tagged; the timeout cuts the finals.
``hybrid``        head OR primary-silence timeout, one event per turn, tagged ``hybrid``.
``hybrid_silero`` head OR an any-speaker Silero silence (fixed ``--silero-timeout-ms``).
``hybrid_dyn``    head OR an any-speaker Silero silence whose length shrinks as the head's probability grows.
"""

HYBRID_POLICIES: tuple[str, ...] = ("hybrid", "hybrid_silero", "hybrid_dyn", "vad_head")
"""Policies that emit exactly one ``turn_end`` per turn, tagged with their own name."""

TURN_PRESETS: tuple[str, ...] = ("balanced", "fast", "steady", "assistant")
"""``vad_head`` constant sets (``config.turn_preset``): ``balanced`` (server default) for conversations, ``fast``
answers sooner for more interruptions, ``assistant`` for speech directed at an agent."""

ENROLL_MODES: tuple[str, ...] = ("after_agent", "after_agent_arm", "explicit")
"""Client-armed enrollment modes (the server's ``--enroll`` values other than its default ``dominant``)."""

AGENT_END_MODES: tuple[str, ...] = ("after_agent", "after_agent_arm")
"""Enrollment modes armed by ``{"type": "agent_end"}``; ``explicit`` is armed by ``{"type": "enroll"}``."""

DEFAULT_EOT_THRESHOLD = 0.98
"""The server's head threshold for ``head`` / ``both`` / ``hybrid`` when the config sets none."""

POLICY_EOT_THRESHOLD: dict[str, float] = {
    "hybrid_silero": 0.99828,
    "hybrid_dyn": 0.998283,
    "vad_head": 0.99,
}
"""The server's policy-specific head thresholds when the config sets none."""

FINAL_SOURCES: tuple[str, ...] = ("stream", "offline")
"""Which finals a client treats as the transcript on a ``--final-asr`` server (see :func:`use_final`)."""

FINAL_TEXTS: tuple[str, ...] = ("fast", "slow")
"""Which final text a client uses on a ``--final-chunk-ms`` server (see :func:`use_final`)."""


def cut_policy(turn_policy: str) -> str:
    """The policy tag whose ``turn_end`` cuts the finals under ``turn_policy`` (the server's rule).

    ``head`` cuts for ``head``; each hybrid policy (including ``vad_head``) cuts for itself; every other policy's
    finals are cut by the plain timeout (``timeout_quiet`` events are tagged ``timeout``).
    """
    if turn_policy == "head" or turn_policy in HYBRID_POLICIES:
        return turn_policy
    return "timeout"


def final_source_of(msg: dict[str, Any]) -> str | None:
    """The source of a ``final`` / ``final_fast`` message: ``final_fast`` is the streaming final (``stream``)."""
    if msg.get("type") == "final_fast":
        return "stream"
    src = msg.get("source")
    return None if src is None else str(src)


def use_final(msg: dict[str, Any], final_source: str = "stream", final_text: str = "fast") -> bool:
    """Whether a ``final`` / ``final_fast`` message belongs to the transcript.

    Servers without a second final pass send no ``source``: every final is used.

    - ``final_text`` (``--final-chunk-ms`` servers): ``"fast"`` keeps the streaming ``final_fast`` (sent at the
      turn end, no added reply latency), ``"slow"`` keeps the slow pass's ``final`` (``source: "slow"``).
    - ``final_source`` (``--final-asr`` servers): ``"stream"`` keeps the streaming finals, ``"offline"`` the
      offline model's (``source`` = its name, e.g. ``tdt_v3``).
    """
    src = final_source_of(msg)
    if src is None:
        return True
    if final_text == "slow" or src == "slow":
        return src == "slow" and final_text == "slow"
    return bool((src == "stream") == (final_source == "stream"))


def config_message(
    *,
    turn_policy: str | None = None,
    timeout_ms: int | None = 1000,
    eot_threshold: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    turn_preset: str | None = None,
) -> dict[str, Any]:
    """Build the ``config`` message (validated). ``None`` leaves a field to the server's default."""
    cfg: dict[str, Any] = {"type": "config"}
    if turn_policy is not None:
        if turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}, got {turn_policy!r}")
        cfg["turn_policy"] = turn_policy
    if turn_preset is not None:
        if turn_preset not in TURN_PRESETS:
            raise ValueError(f"turn_preset must be one of {TURN_PRESETS}, got {turn_preset!r}")
        cfg["turn_preset"] = turn_preset
    if timeout_ms is not None:
        if int(timeout_ms) <= 0:
            raise ValueError("timeout_ms must be positive")
        cfg["timeout_ms"] = int(timeout_ms)
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    cfg["sample_rate"] = int(sample_rate)
    if eot_threshold is not None:
        if not 0.0 < float(eot_threshold) <= 1.0:
            raise ValueError("eot_threshold must be in (0, 1]")
        cfg["eot_threshold"] = float(eot_threshold)
    return cfg


def control_message(
    kind: Literal["agent_end", "enroll", "end"], embedding: list[float] | None = None
) -> dict[str, Any]:
    """Build a control message. ``embedding`` (``enroll`` only): a stored voice print, taken at once."""
    if kind not in ("agent_end", "enroll", "end"):
        raise ValueError(f"unknown control message {kind!r}")
    msg: dict[str, Any] = {"type": kind}
    if embedding is not None:
        if kind != "enroll":
            raise ValueError("only enroll carries an embedding")
        msg["embedding"] = [float(x) for x in embedding]
    return msg


# --------------------------------------------------------------------------- server messages
@dataclass(frozen=True)
class Ready:
    """The first message of a session."""

    model: str
    chunk_ms: float
    frame_ms: float
    diar_config: str | None = None
    column_lag_ms: float | None = None
    """Mean lag (ms, compute excluded) between the end of a frame and its diarizer column being final."""
    enroll: str | None = None
    final_asr: str | None = None
    """Comma-separated extra final sources (``tdt_v3``, ``lookahead``, ``slow``), None without them."""
    final_chunk_ms: float | None = None
    """Chunk of the slow final pass (``--final-chunk-ms``), None without it."""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["ready"] = "ready"


@dataclass(frozen=True)
class FrameEvent:
    """One 80 ms frame: the VAD and turn-head probabilities and the diarizer's speaker columns."""

    t: float
    vad: float
    eot: float | None
    speakers: tuple[float, ...]
    primary: int | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["frame"] = "frame"


@dataclass(frozen=True)
class Partial:
    """The transcript since the last final (sent whenever it changes)."""

    t: float
    text: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["partial"] = "partial"


@dataclass(frozen=True)
class TurnEnd:
    """An end-of-turn decision. ``t`` is the decision time (audio seconds), not the last speech sample.

    ``hinted_at`` is the ``t`` of the :class:`TurnHint` this decision confirms (None when no hint was out);
    ``path`` says which ``vad_head`` path fired (``head``, ``fallback``, ``others`` or ``model``).
    """

    t: float
    policy: str
    p: float | None
    silence_ms: int
    hinted_at: float | None = None
    path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["turn_end"] = "turn_end"


@dataclass(frozen=True)
class TurnHint:
    """``turn_end_hint``: the turn is probably ending. Prepare a reply on ``text``; speak only at the
    ``turn_end`` whose ``hinted_at`` equals ``t``, and drop the prepared reply on :class:`TurnHintCancel`."""

    t: float
    p: float
    text: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["turn_end_hint"] = "turn_end_hint"


@dataclass(frozen=True)
class TurnHintCancel:
    """``turn_end_hint_cancel``: the user resumed; the outstanding hint is withdrawn."""

    t: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["turn_end_hint_cancel"] = "turn_end_hint_cancel"


@dataclass(frozen=True)
class Final:
    """A turn's transcript. ``source`` is None on servers without ``--final-asr``."""

    t: float
    text: str
    speaker: int | None
    source: str | None = None
    start: float | None = None
    end: float | None = None
    latency_ms: float | None = None
    voice_gender: dict[str, float] | None = None
    """``--voice-gender``: perceived voice-gender probabilities of the segment's speech (may be wrong for
    anyone; not an identity)."""
    fast: bool = False
    """True for ``final_fast``: the streaming final a ``--final-chunk-ms`` server sends at the turn end."""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["final"] = "final"

    @property
    def stream_source(self) -> str | None:
        """The final's source, with ``final_fast`` counted as ``stream``."""
        return "stream" if self.fast else self.source

    @property
    def is_second_pass(self) -> bool:
        """True for a second pass's final of a turn (offline ``tdt_v3``, ``lookahead`` or ``slow``)."""
        return self.stream_source not in (None, "stream")

    @property
    def is_offline(self) -> bool:
        """Alias of :attr:`is_second_pass`."""
        return self.is_second_pass


@dataclass(frozen=True)
class Enrolled:
    """The primary speaker was bound to a diarizer column (``--enroll``)."""

    t: float
    column: int
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["enrolled"] = "enrolled"


@dataclass(frozen=True)
class Voiceprint:
    """The session follows a new voice print (single-model mode). Store ``embedding`` and send it back with
    ``enroll`` next session; ``source`` is ``explicit``, ``arm`` or ``refresh``."""

    t: float
    seconds: float
    source: str
    embedding: tuple[float, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["voiceprint"] = "voiceprint"


@dataclass(frozen=True)
class LanguageEvent:
    """The spoken language was decided or changed (``--lid``)."""

    t: float
    language: str
    confidence: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["language"] = "language"


@dataclass(frozen=True)
class Stats:
    """End-of-session statistics; the server closes the socket after it."""

    rtf: float
    chunk_ms_p50: float
    chunk_ms_p95: float
    first_partial_ms: float | None
    peak_rss_mb: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["stats"] = "stats"


@dataclass(frozen=True)
class ServerError:
    """The server's structured ``error`` message, or a message this client could not parse.

    Server errors carry ``code`` (``serve.py`` ``ERROR_CODES``, e.g. ``bad_config``, ``overloaded``,
    ``idle_timeout``), a human-readable ``detail`` and ``fatal``: after a fatal error the server closes the
    connection; a non-fatal one reports a degradation or a refused message and the session goes on.
    Client-side parse failures have ``code="client_parse"`` and ``fatal=False``; ``message`` always says
    what happened.
    """

    message: str
    code: str = "client_parse"
    detail: str = ""
    fatal: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["error"] = "error"


@dataclass(frozen=True)
class Unknown:
    """A message type this client does not know (newer server); ``raw`` holds it."""

    kind: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    type: Literal["unknown"] = "unknown"


Event = Union[
    Ready,
    FrameEvent,
    Partial,
    TurnHint,
    TurnHintCancel,
    TurnEnd,
    Final,
    Enrolled,
    Voiceprint,
    LanguageEvent,
    Stats,
    ServerError,
    Unknown,
]


def _num(d: dict[str, Any], key: str) -> float:
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key}: expected a number, got {v!r}")
    return float(v)


def _opt_num(d: dict[str, Any], key: str) -> float | None:
    v = d.get(key)
    return None if v is None else _num(d, key)


def _opt_int(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{key}: expected an int, got {v!r}")
    return v


def parse_dict(m: dict[str, Any]) -> Event:
    """Convert one decoded server message (not a ``frames`` batch) to an :data:`Event`.

    Raises ``ValueError`` for a malformed message of a known type.
    """
    typ = m.get("type")
    try:
        if typ == "frame":
            spk = m.get("speakers") or []
            return FrameEvent(
                t=_num(m, "t"),
                vad=_num(m, "vad"),
                eot=_opt_num(m, "eot"),
                speakers=tuple(float(x) for x in spk),
                primary=_opt_int(m, "primary"),
                raw=m,
            )
        if typ == "partial":
            return Partial(t=_num(m, "t"), text=str(m.get("text", "")), raw=m)
        if typ == "turn_end":
            return TurnEnd(
                t=_num(m, "t"),
                policy=str(m["policy"]),
                p=_opt_num(m, "p"),
                silence_ms=int(m.get("silence_ms", 0)),
                hinted_at=_opt_num(m, "hinted_at"),
                path=None if m.get("path") is None else str(m["path"]),
                raw=m,
            )
        if typ == "turn_end_hint":
            return TurnHint(
                t=_num(m, "t"),
                p=_opt_num(m, "p") or 0.0,
                text=str(m.get("text") or ""),
                raw=m,
            )
        if typ == "turn_end_hint_cancel":
            return TurnHintCancel(t=_num(m, "t"), raw=m)
        if typ in ("final", "final_fast"):
            src = m.get("source")
            vg = m.get("voice_gender")
            return Final(
                t=_num(m, "t"),
                text=str(m.get("text", "")),
                speaker=_opt_int(m, "speaker"),
                source=None if src is None else str(src),
                start=_opt_num(m, "start"),
                end=_opt_num(m, "end"),
                latency_ms=_opt_num(m, "latency_ms"),
                voice_gender=dict(vg) if isinstance(vg, dict) else None,
                fast=typ == "final_fast",
                raw=m,
            )
        if typ == "voiceprint":
            return Voiceprint(
                t=_num(m, "t"),
                seconds=_opt_num(m, "seconds") or 0.0,
                source=str(m.get("source", "")),
                embedding=tuple(float(x) for x in m.get("embedding") or ()),
                raw=m,
            )
        if typ == "ready":
            return Ready(
                model=str(m.get("model", "audioforge")),
                chunk_ms=_num(m, "chunk_ms") if "chunk_ms" in m else 160.0,
                frame_ms=_num(m, "frame_ms") if "frame_ms" in m else float(FRAME_MS),
                diar_config=None if m.get("diar_config") is None else str(m["diar_config"]),
                column_lag_ms=_opt_num(m, "column_lag_ms"),
                enroll=None if m.get("enroll") is None else str(m["enroll"]),
                final_asr=None if m.get("final_asr") is None else str(m["final_asr"]),
                final_chunk_ms=_opt_num(m, "final_chunk_ms"),
                raw=m,
            )
        if typ == "stats":
            return Stats(
                rtf=_num(m, "rtf") if "rtf" in m else 0.0,
                chunk_ms_p50=_num(m, "chunk_ms_p50") if "chunk_ms_p50" in m else 0.0,
                chunk_ms_p95=_num(m, "chunk_ms_p95") if "chunk_ms_p95" in m else 0.0,
                first_partial_ms=_opt_num(m, "first_partial_ms"),
                peak_rss_mb=_num(m, "peak_rss_mb") if "peak_rss_mb" in m else 0.0,
                raw=m,
            )
        if typ == "enrolled":
            return Enrolled(t=_num(m, "t"), column=int(m["column"]), raw=m)
        if typ == "language":
            return LanguageEvent(
                t=_num(m, "t"),
                language=str(m["language"]),
                confidence=_num(m, "confidence") if "confidence" in m else 1.0,
                raw=m,
            )
        if typ == "error":
            code = str(m.get("code") or "unknown")
            detail = str(m.get("detail") or m.get("message") or m.get("error") or "")
            return ServerError(
                message=f"{code}: {detail}" if detail else code,
                code=code,
                detail=detail,
                fatal=bool(m.get("fatal", False)),
                raw=m,
            )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed {typ!r} message: {e}") from e
    return Unknown(kind=str(typ), raw=m)


def parse_message(data: str | bytes) -> list[Event]:
    """Decode one WebSocket text frame into events (a ``frames`` batch becomes one FrameEvent per item).

    Non-JSON or non-object payloads yield a single :class:`ServerError` instead of raising, so a receive loop
    can log and continue.
    """
    try:
        m = json.loads(data)
    except (ValueError, TypeError) as e:
        return [ServerError(message=f"non-JSON message: {e}")]
    if not isinstance(m, dict):
        return [ServerError(message=f"unexpected payload {m!r}")]
    if m.get("type") == "frames":
        items = m.get("items")
        if not isinstance(items, list):
            return [ServerError(message="frames batch without items", raw=m)]
        out: list[Event] = []
        for it in items:
            try:
                out.append(parse_dict({**it, "type": "frame"}))
            except ValueError as e:
                out.append(ServerError(message=str(e), raw=it))
        return out
    try:
        return [parse_dict(m)]
    except ValueError as e:
        return [ServerError(message=str(e), raw=m)]


def pcm16_duration_s(n_bytes: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """Seconds of audio in ``n_bytes`` of int16 mono PCM."""
    return n_bytes / 2 / float(sample_rate)
