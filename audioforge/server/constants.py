"""The server's fixed clock (16 kHz audio, 80 ms frames, 160 ms chunks), policy constants, diarizer presets and
robustness limits (research/archive/BULLETPROOF.md). Nothing here holds state."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paths import DATA_ROOT
from ..streaming_diar import SORTFORMER_PRESETS, AOSCConfig

__all__ = [
    "ACT_THRESHOLD", "CHUNK_MS", "CHUNK_SAMPLES", "DEFAULT_DIAR_CONFIG", "DEFAULT_IDLE_TIMEOUT_S",
    "DEFAULT_MAX_SESSION_S", "DEFAULT_SILERO", "DEFAULT_THETA", "DEFAULT_TSVAD", "DIAR_CONFIGS", "DIAR_ENC_LEFT",
    "DIAR_LABEL_MODES", "diar_lag_ms", "DIAR_LOW_LATENCY", "DIAR_LOW_LATENCY_032", "diar_preset", "DYN_A",
    "DYN_OFFSET", "DYN_T0", "DYN_TMIN", "ENROLL_MODES", "ENROLL_TRIGGER", "ERROR_CODES",
    "FINAL_ASR_FAILS_BEFORE_RESTART", "FINAL_ASR_RESTART_S", "FINAL_ASR_TIMEOUT_S", "FINAL_MAX_TURN_S",
    "FINAL_POST_ROLL_S", "FINAL_PRE_ROLL_S", "FRAME_MS", "FRAME_SAMPLES", "HYBRID_POLICIES",
    "LOOKAHEAD_MARGIN_FRAMES", "MAX_ABS_SAMPLE", "MAX_CHANNELS", "MAX_INBOX_S", "MAX_ROUND_SAMPLES",
    "MAX_SAMPLE_RATE", "MAX_SEGMENT_S", "MAX_TEXT_MESSAGE", "MIN_SAMPLE_RATE", "PCM_FORMATS", "POLICIES",
    "POLICY_THETA", "PRIMARY_WINDOW_S", "REGISTRY_AUDIO_S", "SHED_DIAR_MODES", "SHED_DIAR_MS", "SHED_NOTICE_S",
    "SHED_PARTIAL_MS", "SHED_RTF", "SHED_WINDOW_S", "SILERO_POLICIES", "SILERO_TIMEOUT_MS", "SR", "STAT_KEEP",
    "TSVAD_DYN", "VAD_HEAD_OTHERS_MS", "VAD_HEAD_OTHERS_P", "VAD_HEAD_SIL_THR", "VAD_HEAD_USER_P", "VAD_HEAD_WAIT_MS",
    "VOICE_MODES",
]

SR = 16000
FRAME_MS = 80
CHUNK_MS = 160
FRAME_SAMPLES = SR * FRAME_MS // 1000
CHUNK_SAMPLES = SR * CHUNK_MS // 1000
MAX_ROUND_SAMPLES = SR // 2  # a backlog is processed in blocks of at most 0.5 s
PRIMARY_WINDOW_S = 5.0
ACT_THRESHOLD = 0.5
# NVIDIA streaming Sortformer v2 card settings (streaming_diar.SORTFORMER_PRESETS, via AOSCConfig.preset).
# "low_latency" = card "low latency", 1.04 s (chunk 6, right context 7; the setting scored in research/archive/STAGE1.md);
# "low_latency_032" = card "ultra low latency", 0.32 s (chunk 3, right context 1), the server default since
# 2026-09-26: research/archive/EOT_BENCH_V2.md section 7 finds the same miss rates at 0.5-0.8 s less latency.
DIAR_CONFIGS = tuple(SORTFORMER_PRESETS)
DEFAULT_DIAR_CONFIG = "low_latency_032"


def diar_preset(name: str) -> dict:
    """The StreamingDiarizer keyword arguments of a SORTFORMER_PRESETS entry, read through AOSCConfig.preset."""
    if name not in SORTFORMER_PRESETS:
        raise ValueError(f"unknown diarizer config {name!r} (one of {DIAR_CONFIGS})")
    cfg = AOSCConfig.preset(name)
    return {k: getattr(cfg, k) for k in SORTFORMER_PRESETS[name]}


DIAR_LOW_LATENCY = diar_preset("low_latency")  # the 1.04 s setting (= eval_stage1.SORTFORMER_LOW_LATENCY)
DIAR_LOW_LATENCY_032 = diar_preset("low_latency_032")
DIAR_ENC_LEFT = 188
# Shipped rules (research/archive/BASELINES.md "ICSI held-out confirmation" and "Dynamic timeout: ICSI held-out confirmation"):
# the trail6 head OR an any-speaker Silero v5 silence (Pipecat VAD state machine, confidence 0.7, start / stop 0.2 s,
# 32 ms chunks). hybrid_silero: fixed silence >= --silero-timeout-ms (2640 = the AMI-fitted k 33, "> 32.8 frames");
# hybrid_dyn: silence - clamp(75 - 55 p, 2, 75) > 4.957 frames (= silence >= clamp(80 - 55 p, 7, 80) frames to 3 ms;
# 6.4 s at p = 0, 2.0 s at p = 1), p = the head's posterior on the same frame.
SILERO_POLICIES = ("hybrid_silero", "hybrid_dyn")
# vad_head (research/EOT_LATENCY.md; --mode single's default rule): no Silero; the served VAD head's silence
# (VAD < VAD_HEAD_SIL_THR) >= K frames AND the turn head p >= theta (0.99), OR that silence >= FALLBACK
# (--vad-wait-ms K,FALLBACK); and, with an enrolled TS-VAD track, the others path: when the user's own silence
# (P(user) < 0.5) reaches USER_SIL while P(other) >= VAD_HEAD_OTHERS_P has held for HOLD (another speaker has the
# floor), the turn is over without waiting for the room to go quiet (--others-wait-ms USER_SIL,HOLD; 0 = off).
# One firing per user turn.
VAD_HEAD_WAIT_MS = (160, 640)
VAD_HEAD_SIL_THR = 0.4
VAD_HEAD_OTHERS_MS = (960, 640)
VAD_HEAD_OTHERS_P = 0.9
VAD_HEAD_USER_P = 0.5
HYBRID_POLICIES = ("hybrid",) + SILERO_POLICIES + ("vad_head",)
POLICIES = ("timeout", "timeout_quiet", "timeout_any", "head", "both") + HYBRID_POLICIES
POLICY_THETA = {"hybrid_silero": 0.99828, "hybrid_dyn": 0.998283, "vad_head": 0.99}  # head threshold unless the config sets one
DEFAULT_THETA = 0.98
SILERO_TIMEOUT_MS = 2640
DYN_T0, DYN_A, DYN_TMIN, DYN_OFFSET = 75.0, 55.0, 2.0, 4.957  # frames (bench_turn_icsi_dyn frozen all-AMI point)
# --turn-input tsvad (research/archive/IMPROVEMENTS.md section 1): the hybrid_dyn (theta, offset) refitted for the turn head
# fed the TS-VAD track (mean of the two AMI-dev cross-fit fold points, research/IMPROVE_115M.md A.2); the shipped
# point above was fitted for the Sortformer-fed head and does not transfer (A.2 "at the shipped point" rows)
TSVAD_DYN = (0.99748, 3.22)
DEFAULT_TSVAD = Path(__file__).resolve().parents[2] / "runs" / "tsvad_spk.pt"
DEFAULT_SILERO = DATA_ROOT / "silero" / "silero_vad_v5.onnx"  # $AUDIOFORGE_DATA/silero
ENROLL_MODES = ("dominant", "after_agent", "after_agent_arm", "explicit")  # --enroll; dominant = the 5 s rule (default)
VOICE_MODES = ("after_agent", "explicit")  # the modes that load TitaNet
# --final-asr (research/archive/HYBRID_ASR.md): the turn span handed to the offline model
FINAL_PRE_ROLL_S = 0.3  # before the served VAD's speech onset
FINAL_POST_ROLL_S = 0.5  # after the last VAD speech frame (trailing silence beyond it is not transcribed)
FINAL_MAX_TURN_S = 60.0  # audio kept for one turn (older samples are dropped)
LOOKAHEAD_MARGIN_FRAMES = 3  # --asr-lookahead: frames decoded past the last VAD speech frame before the final is due
ENROLL_TRIGGER = {"after_agent": "agent_end", "after_agent_arm": "agent_end", "explicit": "enroll"}  # arming message
# --- robustness limits (research/archive/BULLETPROOF.md) ---
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 8000, 192000  # config.sample_rate outside this is refused (1 Hz would be a 16000x amplifier)
MAX_CHANNELS = 8  # config.channels: interleaved channels are averaged to mono
PCM_FORMATS = ("int16", "float32")  # config.format: the binary frames' sample format
MAX_TEXT_MESSAGE = 64 * 1024  # bytes; a larger text frame is refused with an error message
MAX_INBOX_S = 30.0  # audio a connection may have queued unprocessed; beyond it the reader stops reading (TCP backpressure)
MAX_ABS_SAMPLE = 64.0  # samples are clipped here (|x| >= ~1e19 would overflow the float32 power spectrum)
SHED_DIAR_MS = 1500.0  # backlog at which the diarizer is skipped (VAD-only speaker activity in column 0)
# research/archive/DIARIZATION_FIX.md: how finals are labelled and what a shed diarizer frame reports (audioforge.speaker_registry)
DIAR_LABEL_MODES = ("column", "registry")  # column = the primary column (legacy); registry = voice-keyed stable ids
SHED_DIAR_MODES = ("vad", "hold")  # vad = served VAD in column 0 while shed (legacy); hold = last stable column, half cadence
REGISTRY_AUDIO_S = 60.0  # --diar-embed titanet: seconds of audio kept for the per-turn embedding (= the final-ASR cap)
SHED_PARTIAL_MS = 4000.0  # backlog at which partials are dropped, frames batched and the lookahead pass dropped
SHED_NOTICE_S = 5.0  # at most one "overloaded" notice per session per this many seconds
SHED_RTF = 0.9  # shedding engages only while the recent processing rate is at least this (a backlog that a faster-
SHED_WINDOW_S = 4.0  # than-real-time server drains on its own is not overload); measured over this much recent audio
MAX_SEGMENT_S = 300.0  # a segment (text since the last final) is cut with a final at most this long
DEFAULT_MAX_SESSION_S = 4 * 3600.0  # --max-session-s: audio per connection before the server flushes and closes it
DEFAULT_IDLE_TIMEOUT_S = 300.0  # --idle-timeout-s: a connection that sends nothing for this long is closed
STAT_KEEP = 20000  # per-session timing samples kept for the stats percentiles (the most recent ones)
FINAL_ASR_RESTART_S = 60.0  # min seconds between restarts of a dead final-ASR worker
FINAL_ASR_TIMEOUT_S = 60.0  # an offline final not answered within this is a hang: streaming text, worker restarted
FINAL_ASR_FAILS_BEFORE_RESTART = 2  # consecutive failed jobs of a live worker before it is restarted
ERROR_CODES = ("bad_json", "bad_message", "unknown_type", "bad_config", "message_too_large", "unsupported",
               "processing_failed", "internal_error", "session_limit", "idle_timeout", "silero_unavailable",
               "final_asr_failed", "lookahead_dropped", "nan_input", "clipped_input", "nan_state_reset",
               "overloaded", "segment_cap", "server_shutdown", "registry_failed")


def diar_lag_ms(chunk_len: int, right_context: int) -> tuple[float, float, float]:
    """(min, max, mean) ms between the end of a frame and the audio time its diarizer column is finalized."""
    lags = [((t // chunk_len + 1) * chunk_len + right_context - (t + 1)) * FRAME_MS for t in range(chunk_len)]
    return float(min(lags)), float(max(lags)), float(np.mean(lags))
