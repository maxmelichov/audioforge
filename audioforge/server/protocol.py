"""The wire protocol: message schema and ``validate``, the ``error`` message, the per-session ``config`` message
(``SessionConfig``) and PCM decoding. docs/PROTOCOL.md is the user-facing description."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    DEFAULT_THETA,
    ERROR_CODES,
    FRAME_MS,
    HYBRID_POLICIES,
    MAX_CHANNELS,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    PCM_FORMATS,
    POLICIES,
    POLICY_THETA,
    PRIMARY_WINDOW_S,
    SR,
    TURN_PRESETS,
)

__all__ = [
    "DEBUG_KEYS", "decode_pcm", "ENROLL_KEYS", "error_msg", "FINAL_ASR_EXTRA", "FINAL_ASR_KEYS", "LID_KEYS",
    "OPTIONAL_KEYS", "SCHEMA", "SessionConfig", "validate",
]


def error_msg(code: str, detail: str = "", fatal: bool = False) -> dict:
    """The structured ``error`` message: ``code`` (ERROR_CODES), a human-readable ``detail`` and ``fatal`` (the
    server closes the connection right after a fatal error; a non-fatal one reports a degradation and the session
    goes on). Added 2026-09-28; clients that do not know the type may ignore it."""
    assert code in ERROR_CODES, code
    return {"type": "error", "code": code, "detail": str(detail)[:500], "fatal": bool(fatal)}


# --------------------------------------------------------------------------- protocol
_NUM = (int, float)
SCHEMA = {  # type -> {key: allowed python types}; None allowed where listed
    "ready": {"model": (str,), "chunk_ms": _NUM, "frame_ms": _NUM, "diar_config": (str,), "column_lag_ms": _NUM},
    "frame": {"t": _NUM, "vad": _NUM, "eot": _NUM + (type(None),), "speakers": (list,), "primary": (int, type(None))},
    "partial": {"t": _NUM, "text": (str,)},
    "turn_end": {"t": _NUM, "policy": (str,), "p": _NUM + (type(None),), "silence_ms": (int,)},
    "final": {"t": _NUM, "text": (str,), "speaker": (int, type(None))},
    # --final-chunk-ms (dual rate) only: the fast pass's final, sent at the turn_end; the `final` that follows carries
    # the slow pass's text (source "slow", pass "slow" | "fast")
    "final_fast": {"t": _NUM, "text": (str,), "speaker": (int, type(None))},
    "stats": {"rtf": _NUM, "chunk_ms_p50": _NUM, "chunk_ms_p95": _NUM, "first_partial_ms": _NUM + (type(None),),
              "peak_rss_mb": _NUM},
    "enrolled": {"t": _NUM, "column": (int,)},  # --enroll only: the voice enrollment completed on this column
    "voiceprint": {"t": _NUM, "seconds": _NUM, "source": (str,), "embedding": (list,)},  # --turn-input tsvad only: a new TS-VAD print
    "language": {"t": _NUM, "language": (str,), "confidence": _NUM},  # --lid only
    "error": {"code": (str,), "detail": (str,), "fatal": (bool,)},  # protocol / processing errors and degradations
    # the early end-of-turn hint (audioforge.server.turn_hint; off with --turn-hint-off): at most one outstanding per
    # user turn, then confirmed by the next turn_end (turn_end.hinted_at) or withdrawn by turn_end_hint_cancel
    "turn_end_hint": {"t": _NUM, "p": _NUM, "kind": (str,), "text": (str,)},
    "turn_end_hint_cancel": {"t": _NUM},
}
LID_KEYS = {"stats": {"lang": (str, type(None))}}  # present only with --lid
OPTIONAL_KEYS = {"stats": {"degraded": (dict,), "speakers_seen": (int,),  # degraded: only when the session degraded
                           "turn_hints": (dict,), "turn_model": (dict,)},  # turn_hints / turn_end.hinted_at: with turn hints on (default)
                 # path: vad_head's deciding path (head | fallback | others | model); model_ms: --turn-model
                 # smartturn's compute for a "model" decision; stats.turn_model: that model's per-session counters
                 "turn_end": {"hinted_at": _NUM + (type(None),), "path": (str,), "model_ms": _NUM},
                 # speaker_conf / diar_shed: only with --diar-labels registry or --shed-diar hold (DIARIZATION_FIX.md)
                 "final": {"speaker_conf": _NUM + (type(None),), "diar_shed": (bool,), "pass": (str,)},
                 "final_fast": {"speaker_conf": _NUM + (type(None),), "diar_shed": (bool,)},
                 "ready": {"final_chunk_ms": _NUM}}  # pass / final_chunk_ms: --final-chunk-ms only
# present only with --final-asr: every final has "source"; the offline model's finals add start / end / latency_ms
FINAL_ASR_KEYS = {"ready": {"final_asr": (str,)},
                  "final": {"source": (str,)},
                  "stats": {"final_asr": (str,), "final_latency_ms": (dict,), "final_asr_rss_mb": _NUM + (type(None),)}}
FINAL_ASR_EXTRA = {"start": _NUM + (type(None),), "end": _NUM + (type(None),), "latency_ms": _NUM}
# present only with --enroll after_agent|explicit (the default protocol has exactly SCHEMA's keys)
ENROLL_KEYS = {"ready": {"enroll": (str,), "enrolled": (bool,), "primary_column": (int, type(None))},
               "stats": {"enroll": (str,), "enrolled": (bool,), "primary_column": (int, type(None))}}
DEBUG_KEYS = {"frame": {"spk_t"}, "turn_end": {"frame_t"}, "ready": {"diar_lag_ms", "turn_input", "threads",
                                                                      "enroll_rss_mb", "enroll_stride"},
              "stats": {"audio_s", "proc_s", "asr_ms_p50", "asr_ms_p95", "diar_ms_p50", "diar_ms_p95",
                        "turn_ms_p50", "backlog_ms_max", "backlog_ms_end", "send_lag_ms_p50", "send_lag_ms_p95",
                        "send_lag_ms_max", "diar_lag_ms_mean_measured", "n_chunks", "turn_input",
                        "enroll_ms_p50", "enroll_ms_max", "enroll_ms_mean", "enroll_n_embed",
                        "silero_ms_p50", "silero_ms_p95", "silero_ms_mean", "silero_chunks", "lid_ms_mean",
                        "lookahead_ms_mean", "final_flush_n", "final_flush_miss", "final_flush_ms_p50",
                        "final_flush_ms_p95"}}


def validate(msg: dict, debug: bool = False) -> None:
    """Raise ValueError unless ``msg`` has exactly the protocol's keys and types (plus DEBUG_KEYS if debug)."""
    typ = msg.get("type")
    if typ == "frames":
        if set(msg) != {"type", "items"} or not isinstance(msg["items"], list):
            raise ValueError(f"bad frames batch: {msg}")
        for it in msg["items"]:
            validate({**it, "type": "frame"}, debug)
        return
    if typ not in SCHEMA:
        raise ValueError(f"unknown message type {typ!r}")
    spec = SCHEMA[typ]
    enr = ENROLL_KEYS.get(typ, {})
    extra = set(msg) - set(spec) - {"type"}
    if "enroll" in msg:  # the enrollment keys come all together or not at all
        if set(enr) - set(msg):
            raise ValueError(f"{typ}: incomplete enrollment keys {sorted(set(msg) & set(enr))}")
        extra -= set(enr)
    lid = {k: v for k, v in LID_KEYS.get(typ, {}).items() if k in msg}
    lid.update({k: v for k, v in OPTIONAL_KEYS.get(typ, {}).items() if k in msg})
    extra -= set(lid)
    fa = {}
    if any(k in msg for k in FINAL_ASR_KEYS.get(typ, {})):  # --final-asr keys come all together or not at all
        fa = dict(FINAL_ASR_KEYS[typ])
        if typ == "final" and msg.get("source") != "stream":
            fa.update(FINAL_ASR_EXTRA)
        if set(fa) - set(msg):
            raise ValueError(f"{typ}: incomplete final-asr keys, missing {sorted(set(fa) - set(msg))}")
        extra -= set(fa)
    if extra - (DEBUG_KEYS.get(typ, set()) if debug else set()):
        raise ValueError(f"{typ}: unexpected keys {sorted(extra)}")
    for k, tys in (list(spec.items()) + [(k, v) for k, v in enr.items() if k in msg] + list(lid.items())
                     + list(fa.items())):
        if k not in msg:
            raise ValueError(f"{typ}: missing key {k!r}")
        v = msg[k]
        if (isinstance(v, bool) and bool not in tys) or not isinstance(v, tys):
            raise ValueError(f"{typ}.{k}: bad type {type(v).__name__} ({v!r})")
    if typ == "frame":
        if not 4 <= len(msg["speakers"]) <= 8 or not all(isinstance(p, _NUM) and 0 <= p <= 1 for p in msg["speakers"]):
            raise ValueError(f"frame.speakers must be 4-8 probabilities (4, or 8 for an uncut Nemotron-3): {msg['speakers']}")
        if not 0 <= msg["vad"] <= 1 or (msg["eot"] is not None and not 0 <= msg["eot"] <= 1):
            raise ValueError(f"frame probabilities out of range: {msg}")
        if msg["primary"] is not None and not 0 <= msg["primary"] < len(msg["speakers"]):
            raise ValueError(f"frame.primary out of range: {msg}")
    if typ == "turn_end_hint" and (msg["kind"] != "hint" or not 0 <= msg["p"] <= 1):
        raise ValueError(f"turn_end_hint needs kind 'hint' and 0 <= p <= 1: {msg}")
    if typ == "language" and not 0 <= msg["confidence"] <= 1:
        raise ValueError(f"language.confidence out of range: {msg}")
    if typ == "turn_end" and msg.get("path", "head") not in ("head", "fallback", "others", "model"):
        raise ValueError(f"turn_end.path must be head|fallback|others|model: {msg}")
    if typ == "turn_end" and msg["policy"] not in ("timeout", "change", "head") + HYBRID_POLICIES:
        raise ValueError(f"turn_end.policy must be timeout|change|head|{'|'.join(HYBRID_POLICIES)}: {msg}")
    if typ == "error" and msg["code"] not in ERROR_CODES:
        raise ValueError(f"error.code must be one of ERROR_CODES: {msg}")
    for k, v in msg.items():  # nothing non-finite reaches the wire (json.dumps would write NaN, which is not JSON)
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{typ}.{k} is not finite: {v!r}")


@dataclass
class SessionConfig:
    """Per-session options from the client's ``config`` message (docs/PROTOCOL.md)."""
    turn_policy: str = "timeout"  # plain primary-silence timeout; "timeout_quiet" = also wait for no other speaker
    timeout_ms: int = 1000
    eot_threshold: float | None = None  # None = the policy's default (POLICY_THETA, else 0.98)
    sample_rate: int = SR
    format: str = "int16"  # binary frames: int16 (default) | float32, little-endian
    channels: int = 1  # interleaved channels in the binary frames (averaged to mono)
    turn_preset: str | None = None  # vad_head's constants (TURN_PRESETS: balanced | fast); None = the server's --turn-preset

    @property
    def theta(self) -> float:
        """The head threshold in use: eot_threshold if set, else the policy's frozen point (hybrid_silero 0.99828,
        hybrid_dyn 0.998283), else 0.98."""
        return self.eot_threshold if self.eot_threshold is not None else POLICY_THETA.get(self.turn_policy,
                                                                                          DEFAULT_THETA)

    @property
    def bytes_per_frame(self) -> int:
        """Bytes per sample frame of the binary PCM (format width x channels)."""
        return (4 if self.format == "float32" else 2) * self.channels

    def update(self, d: dict, audio_started: bool = False) -> list[str]:
        """Apply a config message; returns warnings (never raises: a bad field is ignored or clamped and reported).
        sample_rate / format / channels only before the first audio."""
        warn = []

        def num(key, cast, lo, hi):
            v = d[key]
            if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                warn.append(f"{key} {v!r} ignored (not a number)")
                return None
            try:
                v = cast(float(v)) if cast is int else cast(v)
            except (TypeError, ValueError, OverflowError):
                warn.append(f"{key} {v!r} ignored (not a number)")
                return None
            if not math.isfinite(v):
                warn.append(f"{key} {v!r} ignored (not finite)")
                return None
            c = min(max(v, lo), hi)
            if c != v:
                warn.append(f"{key} {v} clamped to {c}")
            return c

        if "turn_policy" in d:
            if d["turn_policy"] in POLICIES:
                self.turn_policy = d["turn_policy"]
            else:
                warn.append(f"turn_policy {d['turn_policy']!r} ignored (one of {POLICIES})")
        if "turn_preset" in d:
            if d["turn_preset"] in TURN_PRESETS:
                self.turn_preset = d["turn_preset"]
            else:
                warn.append(f"turn_preset {d['turn_preset']!r} ignored (one of {tuple(TURN_PRESETS)})")
        if "timeout_ms" in d:  # the primary window is 5 s, so a longer silence cannot be measured
            v = num("timeout_ms", int, FRAME_MS, int(PRIMARY_WINDOW_S * 1000 - 2 * FRAME_MS))
            if v is not None:
                self.timeout_ms = int(v)
        if "eot_threshold" in d:
            v = num("eot_threshold", float, 0.0, 1.0)
            if v is not None:
                self.eot_threshold = float(v)
        for key in ("sample_rate", "format", "channels"):
            if key in d and audio_started:
                warn.append(f"{key} ignored after the first audio frame")
        if "sample_rate" in d and not audio_started:
            v = d["sample_rate"]
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) \
                and MIN_SAMPLE_RATE <= v <= MAX_SAMPLE_RATE
            if ok:
                self.sample_rate = int(v)
            else:
                warn.append(f"sample_rate {v!r} ignored (need {MIN_SAMPLE_RATE}..{MAX_SAMPLE_RATE})")
        if "format" in d and not audio_started:
            if d["format"] in PCM_FORMATS:
                self.format = d["format"]
            else:
                warn.append(f"format {d['format']!r} ignored (one of {PCM_FORMATS})")
        if "channels" in d and not audio_started:
            v = d["channels"]
            if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= MAX_CHANNELS:
                self.channels = v
            else:
                warn.append(f"channels {v!r} ignored (need 1..{MAX_CHANNELS})")
        return warn


def decode_pcm(raw: bytes, cfg: SessionConfig) -> tuple[np.ndarray, bytes, int]:
    """Binary frame bytes -> (mono float32 samples at cfg.sample_rate, trailing partial sample bytes, number of
    non-finite / out-of-range float32 samples repaired). int16 is scaled by 1/32768; float32 is clipped to [-1, 1]."""
    bpf = cfg.bytes_per_frame
    cut = len(raw) - len(raw) % bpf
    bad = 0
    if cfg.format == "float32":
        x = np.frombuffer(raw[:cut], dtype="<f4").astype(np.float32)
        fin = np.isfinite(x)
        if not fin.all():
            bad += int((~fin).sum())
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        big = np.abs(x) > 1.0
        if big.any():
            bad += int(big.sum())
            x = np.clip(x, -1.0, 1.0)
    else:
        x = np.frombuffer(raw[:cut], dtype="<i2").astype(np.float32) / 32768.0
    if cfg.channels > 1 and len(x):
        x = x.reshape(-1, cfg.channels).mean(1).astype(np.float32)
    return x, raw[cut:], bad
