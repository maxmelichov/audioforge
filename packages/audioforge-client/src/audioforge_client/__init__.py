"""audioforge-client: WebSocket client and message framing for the audioforge streaming speech server.

The server (``audioforge/serve.py`` in https://github.com/maxmelichov/audioforge) runs a streaming
ASR with VAD and turn heads plus a streaming speaker diarizer on one 80 ms clock and emits ``frame`` /
``partial`` / ``turn_end`` / ``final`` events over a WebSocket. This package has no model code and no torch
dependency: it is what a voice-agent framework plugin (Pipecat, LiveKit Agents, ...) needs to talk to it.
"""

from .protocol import (
    AGENT_END_MODES,
    DEFAULT_EOT_THRESHOLD,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_URL,
    ENROLL_MODES,
    FINAL_SOURCES,
    FRAME_MS,
    HYBRID_POLICIES,
    NUM_SPEAKERS,
    POLICIES,
    POLICY_EOT_THRESHOLD,
    Enrolled,
    Event,
    Final,
    FrameEvent,
    LanguageEvent,
    Partial,
    Ready,
    ServerError,
    Stats,
    TurnEnd,
    Unknown,
    config_message,
    control_message,
    cut_policy,
    parse_dict,
    parse_message,
    pcm16_duration_s,
    use_final,
)
from .session import AudioforgeClosedError, AudioforgeConnectionError, AudioforgeSession

__version__ = "0.1.0"

__all__ = [
    "AGENT_END_MODES",
    "DEFAULT_EOT_THRESHOLD",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_URL",
    "ENROLL_MODES",
    "FINAL_SOURCES",
    "FRAME_MS",
    "HYBRID_POLICIES",
    "NUM_SPEAKERS",
    "POLICIES",
    "POLICY_EOT_THRESHOLD",
    "AudioforgeClosedError",
    "AudioforgeConnectionError",
    "AudioforgeSession",
    "Enrolled",
    "Event",
    "Final",
    "FrameEvent",
    "LanguageEvent",
    "Partial",
    "Ready",
    "ServerError",
    "Stats",
    "TurnEnd",
    "Unknown",
    "__version__",
    "config_message",
    "control_message",
    "cut_policy",
    "parse_dict",
    "parse_message",
    "pcm16_duration_s",
    "use_final",
]
