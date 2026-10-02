"""Command line and config file of the server: ``python -m audioforge.serve`` and ``audioforge-serve``.

Every flag is declared once in ``FLAGS`` (group, one-line help, the longer text for docs/CONFIGURATION.md and the
section that explains it). ``build_parser`` turns the table into argparse; ``--help`` shows the everyday flags and
``--help-advanced`` all of them. A YAML or JSON file given with ``--config`` (or ``$AUDIOFORGE_CONFIG``) sets the same
options by their long name (``port: 9000``, ``final-asr: tdt_v3`` or ``final_asr: tdt_v3``); flags on the command
line win over the file. ``scripts/dev/gen_config_doc.py`` writes the flag reference of docs/CONFIGURATION.md from this
table and ``tests/test_config_doc.py`` fails when the page is stale.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_DIAR_CONFIG,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_MAX_SESSION_S,
    DIAR_CONFIGS,
    DIAR_ENC_LEFT,
    DIAR_LABEL_MODES,
    ENROLL_MODES,
    POLICIES,
    SHED_DIAR_MODES,
    SILERO_TIMEOUT_MS,
    TURN_PRESET_DEFAULT,
    TURN_PRESETS,
    VOICE_MODES,
)

__all__ = ["ENV_VARS", "FLAGS", "GROUPS", "MODES", "SINGLE_CONFLICTS", "Flag", "build_parser", "load_config", "load_engine", "main",
           "parse_args"]

DESCRIPTION = "Streaming speech front end: ASR + VAD + speaker activity + end-of-turn events over one WebSocket."


@dataclass(frozen=True)
class Flag:
    """One command-line option: argparse arguments plus what the docs need."""

    flags: tuple[str, ...]
    group: str
    help: str  # one line for --help
    kwargs: dict = field(default_factory=dict)  # argparse keyword arguments (type, default, choices, action, ...)
    advanced: bool = False  # hidden from --help, shown by --help-advanced
    doc: str = ""  # longer text for docs/CONFIGURATION.md (default: help)
    doc_default: str | None = None  # the default as the docs show it (default: from kwargs)
    section: str = ""  # anchor in docs/CONFIGURATION.md that explains the flag
    launcher: bool = False  # audioforge-serve only (not python -m audioforge.serve)

    @property
    def dest(self) -> str:
        return self.kwargs.get("dest") or self.flags[0].lstrip("-").replace("-", "_")


# group titles in --help order
GROUPS = {
    "models": "models",
    "server": "server",
    "turns": "turn detection and speakers",
    "transcripts": "transcripts",
    "lid": "language ID",
    "diarizer": "diarizer tuning",
    "speed": "speed",
    "limits": "limits, logging and diagnostics",
}

S3, S4, S5, S6 = "#3-models-threads-and-speed", "#4-turn-policies-configturn_policy", \
    "#5-turn-head-input---turn-input", "#6-primary-speaker-enrollment---enroll"
S71, S72, S73, S74 = "#71---diar-config-and---diar-set", "#72---diar-left", \
    "#73-nemotron-3-diarization-as-the-diarizer", "#74-multi-speaker-rooms---diar-labels---shed-diar-timeout_any"
S8, S9, S10, SCFG = "#8-final-asr-and-dual-lookahead", "#9-spoken-language-id---lid", "#10-diagnostics", \
    "#12-config-file-and-environment"
SSINGLE = "#13-single-model-mode---mode-single"

# --mode presets of audioforge-serve (and audioforge.load / audioforge-bench): the options each one sets unless given.
# single (the default since 2026-09-29) = everything from the one 115M checkpoint (research/SINGLE_MODEL.md): the TS-VAD track of the user's voice
# print feeds the turn head and stands in for the diarizer's columns (no diarizer is loaded), the distilled LID head
# on the shared encoder, no final-ASR worker. The print comes from an `enroll` message carrying an embedding, else
# live from the first --tsvad-print-s seconds of speech after `agent_end`.
# room = the two-model stack: general diarization with NVIDIA Nemotron-3-Diarization (or Sortformer v2) next to the
# 115M model, for labelling everyone in a room. Without --mode, a diarizer / --diar / --final-asr selects room.
MODES: dict[str, dict[str, Any]] = {
    "room": {},
    "single": {"turn_input": "tsvad", "diar_off": True, "lid": "head", "enroll": "after_agent_arm",
               "turn_policy": "vad_head", "dyn_wait_ms": "2000,960",
               "turn_preset": "balanced"},
}
# options that would load a second model; --mode single refuses them (name, why)
SINGLE_CONFLICTS = (("diar", "a diarizer model"), ("final_asr", "a final-ASR model (Parakeet-TDT v3)"),
                    ("titanet", "TitaNet-L"), ("diar_embed", "TitaNet-L for --diar-embed titanet"))

FLAGS: tuple[Flag, ...] = (
    # --- models
    Flag(("--models-dir",), "models", "where audioforge-download put the models",
         {"metavar": "DIR"}, launcher=True, doc_default="`$AUDIOFORGE_HOME`, else `<repo>/models`, else "
         "`~/.cache/audioforge`", section="#1-launching"),
    Flag(("--mode",), "models", "single (default: the one 115M model, known user) | room (+ NVIDIA diarizer)",
         {"choices": ["single", "room"], "default": None}, launcher=True,
         doc_default="`single` (`room` when `--diarizer`, `--diar` or `--final-asr` is given)",
         doc="preset. `single` (the default): everything from the one 115M checkpoint, for a known user: adds `--turn-input tsvad "
         "--diar-off --lid head --enroll after_agent_arm --turn-policy vad_head --dyn-wait-ms 2000,960`, loads no "
         "diarizer, no final-ASR worker and no Silero (the default turn rule `vad_head` reads the model's own VAD, turn "
         "and TS-VAD heads; `hybrid_dyn` needs `--silero`), and refuses "
         "`--diar`, `--diarizer`, `--final-asr`, `--lid ambernet`, `--diar-embed titanet`; the voice print comes from an `enroll` message "
         "with an embedding (store >= 5 s of clean speech, 10 s for meetings), else live after `agent_end`. `room`: "
         "general diarization with NVIDIA Nemotron-3-Diarization (or `--diarizer sortformer`) next to the 115M model "
         "([§7](#7-diarizer)). Flags you pass yourself win", section=SSINGLE),
    Flag(("--core",), "models", "streaming core: 115m (default) | 0.6b (nemotron-speech-streaming-en-0.6b; GPU)",
         {"choices": ["115m", "0.6b"], "default": None}, launcher=True, doc_default="`115m`",
         doc="which streaming core the launcher loads from the models directory: `115m` (NVIDIA FastConformer 114M + "
         "our heads, real time on 2 CPU threads) or `0.6b` (NVIDIA nemotron-speech-streaming-en-0.6b, NVIDIA Open Model "
         "License, with every head retrained on it: `served_0p6b_v0.3.afm`, `tsvad_0p6b.pt`, `lid_0p6b_v2.pt`; about half the "
         "meeting WER, but ~96 ms of CPU per 160 ms chunk on 2 threads: one real-time stream per process against the 115M's 4, "
         "3 on an Apple GPU with `--device mps`, 5 GB RSS). Voice prints "
         "belong to one core: re-enroll after switching. Install with `audioforge-download --core 0.6b`; "
         "research/CORE_0P6B.md has every number side by side", section="#1-launching"),
    Flag(("--diarizer",), "models", "room mode: which downloaded diarizer to run (implies --mode room)",
         {"choices": ["nemotron3", "sortformer"], "default": None}, launcher=True,
         doc_default="room mode: `nemotron3` if downloaded, else `sortformer`",
         doc="room mode: which downloaded diarizer to pass as `--diar`; `nemotron3` also adds `--diar-pool max --diar-left 1`. "
         "For either diarizer the launcher adds `--shed-diar hold`; flags you pass yourself win",
         section=S73),
    Flag(("--asr",), "models", "ASR + heads .afm (audioforge-serve: from the models directory)",
         {"metavar": "PATH"}, doc="ASR + heads `.afm` (`stage1_served_v4.afm` = `stage1_served_v3.afm` + the speech-detector head; v3 = the block-4 VAD build `stage1_served_v2.afm` + the turn head v5 classifier; `stage1_served.afm` = the measured v1)",
         doc_default="required (`audioforge-serve`: from the models directory)", section=S3),
    Flag(("--diar",), "models", "diarizer .afm (audioforge-serve: from the models directory)",
         {"metavar": "PATH"}, doc="diarizer `.afm`: the Streaming Sortformer v2 or the Nemotron-3-Diarization import",
         doc_default="required unless `--diar-off` (`audioforge-serve`: from the models directory; none with "
         "`--mode single`)", section="#7-diarizer"),
    # --- server
    Flag(("--host",), "server", "listen address; 0.0.0.0 for all interfaces", {"default": "127.0.0.1"}),
    Flag(("--port",), "server", "listen port", {"type": int, "default": 8765, "metavar": "N"}),
    Flag(("--threads",), "server", "torch threads of the shared compute worker",
         {"type": int, "default": 2, "metavar": "N"},
         section=S3),
    Flag(("--config",), "server", "YAML / JSON file with any of these options (flags win)",
         {"metavar": "FILE"}, doc="YAML or JSON file with any of these options by long name; command-line flags "
         "win over it", doc_default="`$AUDIOFORGE_CONFIG`, else none", section=SCFG),
    # --- turn detection and speakers
    Flag(("--enroll",), "turns", "user binding: " + " | ".join(ENROLL_MODES),
         {"choices": ENROLL_MODES, "default": "dominant", "metavar": "MODE"},
         doc="primary-speaker binding: `dominant` (the 5 s dominant column), `after_agent` (TitaNet voice following "
         "from the client's `agent_end`), `after_agent_arm` (the next active column after `agent_end`, no TitaNet), "
         "`explicit` (the utterance after an `enroll` message)", section=S6),
    Flag(("--diar-labels",), "turns", "final.speaker: primary column | stable voice ids",
         {"choices": DIAR_LABEL_MODES, "default": "column"},
         doc="what `final.speaker` is: the primary column (legacy) or a stable voice-keyed id from a per-session "
         "speaker registry (adds `final.speaker_conf`, `final.diar_shed`, `stats.speakers_seen`)", section=S74),
    Flag(("--turn-input",), "turns", "what the turn head reads: auto | session | diar | tsvad",
         {"choices": ["auto", "session", "diar", "tsvad"], "default": "auto"}, advanced=True,
         doc="what the turn head reads: the session's encoder frames, a speaker-conditioned pass on the diarizer's "
         "clock (`diar`), or the TS-VAD track of the enrolled user's voice print (`tsvad`)", section=S5),
    Flag(("--tsvad",), "turns", "--turn-input tsvad: TS-VAD head file", {"metavar": "PATH"}, advanced=True,
         doc_default="`runs/tsvad_spk.pt`", section=S5),
    Flag(("--tsvad-print-s",), "turns", "--turn-input tsvad: seconds of speech in the voice print when armed",
         {"type": float, "default": 5.0, "metavar": "S"}, advanced=True, section=S5),
    Flag(("--tsvad-refresh-s",), "turns", "--turn-input tsvad: re-embed the print every S s of target speech (0 = never)",
         {"type": float, "default": 0.0, "metavar": "S"}, advanced=True, section=S5),
    Flag(("--titanet",), "turns", "TitaNet-L .nemo for --enroll after_agent / explicit", {"metavar": "PATH"},
         advanced=True, doc_default="`$AUDIOFORGE_DATA/nemo/speakerverification_en_titanet_large.nemo`", section=S6),
    Flag(("--enroll-stride",), "turns", "voice modes: re-embed the columns every N frames (5 = 400 ms)",
         {"type": int, "default": 5, "metavar": "N"}, advanced=True, section=S6),
    Flag(("--silero",), "turns", "Silero VAD v5 ONNX for turn_policy hybrid_silero / hybrid_dyn",
         {"metavar": "PATH"}, advanced=True,
         doc="Silero VAD v5 ONNX for `hybrid_silero` / `hybrid_dyn`; loaded at the first session that uses such a "
         "policy, or at start when the flag is given", doc_default="`$AUDIOFORGE_DATA/silero/silero_vad_v5.onnx`",
         section=S4),
    Flag(("--silero-timeout-ms",), "turns", "hybrid_silero: any-speaker Silero silence that ends the turn",
         {"type": int, "default": SILERO_TIMEOUT_MS, "metavar": "MS"}, advanced=True, section=S4),
    Flag(("--dyn-wait-ms",), "turns", "hybrid_dyn: Silero-silence wait at head p = 0 and p = 1, e.g. 2000,960",
         {"metavar": "CAP,FLOOR"}, advanced=True,
         doc="`hybrid_dyn`: the Silero-silence wait at head posterior 0 (CAP) and 1 (FLOOR), linear in between (plus "
         "the rule's offset); `--mode single` sets `2000,960` (research/SINGLE_MODEL.md A1)",
         doc_default="the served rule: 6000,1600", section=S4),
    Flag(("--turn-policy",), "turns", "turn_policy of a session whose config names none (--mode single: vad_head)",
         {"choices": list(POLICIES), "default": "timeout", "metavar": "POLICY"}, advanced=True,
         doc="the `turn_policy` of a session whose `config` does not name one (a client's `config` still wins); "
         "`--mode single` sets `vad_head` (research/EOT_LATENCY.md)", section=S4),
    Flag(("--vad-wait-ms",), "turns", "vad_head: VAD-head silence for the head path and the fallback, e.g. 160,640",
         {"metavar": "K,FALLBACK"}, advanced=True,
         doc="`vad_head` (no Silero): the served VAD head's silence (VAD < 0.4) that the head path needs (K, with the turn "
         "head p >= theta, default 0.99) and the silence that ends the turn on its own (FALLBACK, 0 = none) "
         "(research/EOT_LATENCY.md)", doc_default="`160,640`", section=S4),
    Flag(("--turn-preset",), "turns", "vad_head's trade-off: balanced (default), fast, steady or assistant (see docs)",
         {"choices": list(TURN_PRESETS), "default": TURN_PRESET_DEFAULT, "metavar": "PRESET"},
         doc="`vad_head`'s constants as one named trade-off (a client's `config.turn_preset` wins for its session). "
         "`balanced` = VAD < 0.4 for >= 160 ms AND p >= 0.99, OR 640 ms, others path 960,640. `fast` (turn head v5, "
         "served heads v0.3) = the v5 segment classifier asked after 80 ms of VAD < 0.6 and at every further quiet "
         "frame ends the turn at P(complete) > 0.7, OR 640 ms of VAD < 0.4, others path 960,640: on two-party calls "
         "547 vs 956 ms p50 at 24.8 vs 20.2 % false interruptions and 5.5 vs 7.3 % missed (AMI 1247 vs 1326 ms, "
         "11.5 vs 10.5 % FI, 34.0 vs 33.5 % missed). `steady` = the fast rule before v5 (VAD < 0.6 for >= 480 ms AND "
         "p >= 0.99, OR 720 ms, others path 640,640): 886 ms p50 but the best p95 (1434 ms) and misses (3.7 %). "
         "`assistant` = v5 asked after 240 ms of energy-or-VAD quiet, P(complete) > 0.9, OR 2960 ms of VAD silence: "
         "for speech directed at the agent (smart-turn's 399 test clips: 92 % accuracy at 291 ms p50), not for human "
         "conversation. `--vad-wait-ms` / `--others-wait-ms` override the preset's values (research/TURN_V5.md, "
         "research/EOT_LATENCY.md \"Turn presets\")", doc_default="`balanced`", section=S4),
    Flag(("--others-wait-ms",), "turns", "vad_head: user's TS-VAD silence + P(other) hold of the others path, e.g. 960,640",
         {"metavar": "USER_SIL,HOLD"}, advanced=True,
         doc="`vad_head` with an enrolled TS-VAD track (`--turn-input tsvad`): the turn also ends when the user's own "
         "silence (P(user) < 0.5) reaches USER_SIL while P(other) >= 0.9 has held for HOLD, i.e. another speaker has the "
         "floor, without waiting for the room to go quiet (0 = off; research/EOT_LATENCY.md)", doc_default="`960,640`",
         section=S4),
    Flag(("--turn-hint-p",), "turns", "turn_end_hint: turn-head posterior (with 80 ms of VAD silence) for the early hint",
         {"type": float, "default": 0.8, "metavar": "P"}, advanced=True,
         doc="the early end-of-turn hint (`turn_end_hint`, docs/PROTOCOL.md 5.11): sent once per user turn on the first "
         "frame with 80 ms of served-VAD silence (VAD < 0.4) and turn-head p >= P, then confirmed by the next `turn_end` "
         "(`hinted_at`) or withdrawn by `turn_end_hint_cancel` when the user resumes (VAD > 0.5 on 2 frames); a voice "
         "agent starts preparing its reply on it (Pipecat eager end of turn, LiveKit preemptive generation). "
         "`turn_end` itself is unchanged", section=S4),
    Flag(("--turn-hint-off",), "turns", "no turn_end_hint / turn_end_hint_cancel events (and no turn_end.hinted_at)",
         {"action": "store_true"}, advanced=True, section=S4),
    Flag(("--energy-gate",), "turns", "vad_head: energy-aware onset arming + warm-up guard (on | off)",
         {"choices": ["on", "off"], "default": "on"}, advanced=True,
         doc="`vad_head`'s energy gate (research/EOT_ASSISTANT.md \"Energy gate\"): a per-session noise floor (10th "
         "percentile of the 80 ms frame log energies of the last 3 s); a user turn is armed only by an onset frame "
         "(VAD > 0.5 AND energy > floor + 6 dB), and no `turn_end` / `turn_end_hint` is sent before 160 ms of onset "
         "frames in the session. It removes the turn_ends the served VAD head (~0.55 on a fresh session's first "
         "frames, ~0.66 on room tone) fired before the user spoke: 137 -> 0 of 399 assistant clips; calls / AMI / the "
         "bundled clip unchanged. `off` = the VAD-only rule", doc_default="`on`", section=S4),
    Flag(("--energy-quiet-db",), "turns", "vad_head: also count frames below floor + X dB as silence (default off)",
         {"type": float, "default": None, "metavar": "X"}, advanced=True,
         doc="with `--energy-gate on`: a frame whose energy is below the noise floor + X dB is silence whatever the "
         "VAD says, so the silence starts at the audible end instead of at the end of the VAD head's tail. Off by "
         "default: on the assistant set it answers 400-600 ms sooner (X 6: 611 vs 1218 ms p50), but on two-party "
         "calls and AMI mid-turn pauses are room tone too, and false interruptions rise at every X in 3..12 dB (calls "
         "20.2 -> 25-60 %, AMI 10.5 -> 11-40 %) even with the fallback re-tuned (research/EOT_ASSISTANT.md)",
         doc_default="off", section=S4),
    Flag(("--turn-model",), "turns", "vad_head's end-of-turn classifier: head (default) | smartturn (opt-in bridge)",
         {"choices": ["head", "smartturn"], "default": "head"}, advanced=True,
         doc="`head` = the turn head's p (the shipped rule). `smartturn` = Pipecat's smart-turn v3.2 ONNX "
         "(pipecat-ai/smart-turn, BSD-2-Clause; a second model, 8.7 MB, ~22 ms per call on 2 CPU threads) asked once "
         "per silence run after 160 ms of VAD silence, on the turn's last <= 8 s with 0.5 s of pre-speech audio, "
         "prepared exactly as Pipecat's LocalSmartTurnAnalyzerV3; complete -> `turn_end` (`path: model`), "
         "incomplete -> wait for the next silence run or the preset's 3 s fallback. A bridge on assistant-directed "
         "speech while the native classifier (turn head v5) is trained; see research/EOT_ASSISTANT.md for its "
         "numbers on all three benchmarks (on human-to-human calls it misses more turn ends than the default)",
         doc_default="`head`", section=S4),
    Flag(("--smartturn-trigger",), "turns", "--turn-model smartturn: when it is asked: vad (default) | energy",
         {"choices": ["vad", "energy"], "default": "vad"}, advanced=True,
         doc="`vad`: after 160 ms of VAD silence (VAD < the preset's threshold), fallback 3 s (the accurate "
         "setting). `energy`: two clocks, the classifier after 240 ms of energy-or-VAD quiet (energy < noise floor + 6 "
         "dB) ending the turn at P(complete) > 0.97, the fallback timer on the VAD's own silence at the preset's "
         "640 / 720 ms (the fast setting; every mid-utterance stop longer than the timer still ends the turn). "
         "Numbers: research/EOT_ASSISTANT.md", doc_default="`vad`", section=S4),
    Flag(("--smartturn-onnx",), "turns", "--turn-model smartturn: the smart-turn v3.x ONNX file",
         {"metavar": "PATH"}, advanced=True,
         doc="the smart-turn ONNX for `--turn-model smartturn`", doc_default="the smart-turn-v3.2-cpu.onnx bundled "
         "in the installed `pipecat-ai` package", section=S4),
    # --- transcripts
    Flag(("--final-asr",), "transcripts", "re-transcribe each finished turn: tdt_v3 or a .nemo path",
         {"metavar": "SPEC"}, doc="offline per-turn final ASR: `tdt_v3` (NVIDIA Parakeet-TDT 0.6B v3) or a `.nemo` "
         "path; adds finals with `source: tdt_v3`", doc_default="off", section=S8),
    Flag(("--final-asr-worker",), "transcripts", "--final-asr: child process or thread",
         {"choices": ["process", "thread"], "default": "process"}, advanced=True, section=S8),
    Flag(("--final-asr-threads",), "transcripts", "--final-asr: torch threads of the offline model",
         {"type": int, "default": 2, "metavar": "N"}, advanced=True, section=S8),
    Flag(("--final-asr-device",), "transcripts", "--final-asr: cpu | mps | cuda", {"default": "cpu", "metavar": "DEV"},
         advanced=True, section=S8),
    Flag(("--asr-lookahead",), "transcripts", "second, text-only ASR pass at attention context [70, R], e.g. 13",
         {"type": int, "metavar": "R"}, advanced=True, doc_default="off", section=S8),
    Flag(("--asr-chunk-ms",), "transcripts", "streaming chunk of the ASR pass: 160 ([70,1]) or 80 ([70,0], no lookahead)",
         {"type": int, "choices": [80, 160], "metavar": "MS"}, advanced=True,
         doc="streaming chunk of the one ASR pass (transcript, VAD, turn and TS-VAD heads): 160 = attention context "
         "[70,1] (the model's default) or 80 = [70,0], no lookahead (up to 80 ms earlier frames; +0.19 WER on "
         "LibriSpeech, +1.3 on AMI; the heads were trained at [70,1]; research/LATENCY_BUDGET.md)",
         doc_default="the model's (160)"),
    Flag(("--asr-vad-gate",), "transcripts", "stop decoding tokens on long non-speech (served VAD <= this)",
         {"type": float, "metavar": "P"}, advanced=True,
         doc="do not decode transducer tokens on frames whose served VAD <= P once `--asr-vad-hangover-ms` of such "
         "frames have passed (bounds hallucinated text on long non-speech; research/archive/BULLETPROOF.md)",
         doc_default="off"),
    Flag(("--asr-vad-hangover-ms",), "transcripts", "--asr-vad-gate: decoding continues this long after speech",
         {"type": float, "default": 1200.0, "metavar": "MS"}, advanced=True),
    Flag(("--beam",), "transcripts", "RNNT beam search width for finals (0 = greedy)",
         {"type": int, "default": 0, "metavar": "K"}, advanced=True,
         doc="finals take the best hypothesis of an RNNT beam search of width K (up to 3 tokens per frame) run next "
         "to the greedy decoder; partials and the turn heads keep the greedy tokens. 115M: -1.3 / -2.3 WER points on "
         "held-out ICSI / AMI at K = 8, about 1 ms per 80 ms frame on CPU; the 0.6B does not gain (research/FIXALL.md)",
         doc_default="0 (greedy)"),
    # --- language ID
    Flag(("--lid",), "lid", "spoken language ID: head, a head file or ambernet", {"metavar": "head|PATH|ambernet"},
         advanced=True, doc="spoken language ID: `head` (the shipped distilled head on the shared encoder, "
         "`lid_115m_v2.pt`), a head file (`audioforge.lid.save_head`), or `ambernet` / an AmberNet `.nemo`; emits "
         "`language` messages and `stats.lang`", doc_default="off", section=S9),
    Flag(("--lid-threshold",), "lid", "--lid: posterior needed to announce a language",
         {"type": float, "default": 0.9, "metavar": "P"}, advanced=True, section=S9),
    Flag(("--lid-min-ms",), "lid", "--lid: pooled speech needed before the first announcement",
         {"type": float, "default": 1000.0, "metavar": "MS"}, advanced=True, section=S9),
    Flag(("--lid-max-ms",), "lid", "--lid head/file: announce the top language after this much speech anyway",
         {"type": float, "default": None, "metavar": "MS"}, advanced=True,
         doc_default="3000 with `--lid head`, else off (0 = off)", section=S9),
    Flag(("--lid-langs",), "lid", "--lid ambernet: comma-separated language codes to choose from",
         {"metavar": "CODES"}, advanced=True, doc_default="the 17 languages of research/archive/LID.md", section=S9),
    # --- diarizer tuning
    Flag(("--diar-config",), "diarizer", "Sortformer setting: 0.32 s (low_latency_032) or 1.04 s",
         {"choices": DIAR_CONFIGS, "default": DEFAULT_DIAR_CONFIG}, advanced=True, section=S71),
    Flag(("--diar-set",), "diarizer", "override one diarizer streaming field, KEY=VALUE (repeatable)",
         {"action": "append", "default": [], "metavar": "KEY=VALUE"}, advanced=True, doc_default="none",
         section=S71),
    Flag(("--diar-left",), "diarizer", "diarizer encoder left context in window mode (frames)",
         {"type": int, "default": DIAR_ENC_LEFT, "metavar": "N"}, advanced=True, section=S72),
    Flag(("--diar-pool",), "diarizer", "Nemotron-3 only: 10 ms -> 80 ms pooling of its outputs",
         {"choices": ["mean", "max"]}, advanced=True, doc_default="the import's (`mean`)", section=S73),
    Flag(("--diar-spks",), "diarizer", "Nemotron-3 only: keep the first K of its 8 columns",
         {"type": int, "metavar": "K"}, advanced=True, doc_default="all 8", section=S73),
    Flag(("--diar-embed",), "diarizer", "--diar-labels registry: turn embedding (spk = served head)",
         {"choices": ["spk", "titanet"], "default": "spk"}, advanced=True, section=S74),
    Flag(("--diar-reg-thr",), "diarizer", "--diar-labels registry: cosine at which a turn joins a known speaker",
         {"type": float, "metavar": "X"}, advanced=True, doc_default="per embedder (spk 0.55, titanet 0.40)",
         section=S74),
    Flag(("--shed-diar",), "diarizer", "under load: vad (VAD in column 0) | hold (last column, half cadence)",
         {"choices": SHED_DIAR_MODES, "default": "vad"}, advanced=True,
         doc_default="`vad` (`audioforge-serve` adds `hold`)", section=S74),
    Flag(("--diar-off",), "diarizer", "with --turn-input tsvad: run no diarizer (speakers = [P(user), P(other), 0, 0])",
         {"action": "store_true"}, advanced=True,
         doc="with `--turn-input tsvad`: run no diarizer (speakers = [P(user), P(other), 0, 0]); without `--diar` the "
         "diarizer is not even loaded", section=S5),
    # --- speed
    Flag(("--perf",), "speed", "CPU fast paths: default (exact) | none | all | list, e.g. default,-linear_t",
         {"default": "default", "metavar": "SPEC"}, advanced=True,
         doc="CPU inference fast paths of `audioforge.perf`: `default` = the exact set (same outputs), `none`, `all` "
         "(adds float-rounding ones), or a list such as `default,-linear_t` (research/archive/PERFORMANCE.md)", section=S3),
    Flag(("--device",), "speed", "cpu, or mps / cuda / cuda:N (opt-in GPU; others fall back to cpu)", {"default": "cpu", "metavar": "DEV"},
         advanced=True, section=S3),
    Flag(("--no-fast-conv",), "speed", "keep PyTorch's Conv1d path in the conformer convolutions (slower)",
         {"action": "store_true"}, advanced=True, section=S3),
    Flag(("--no-warmup",), "speed", "skip the 2 s warm-up session at start", {"action": "store_true"},
         advanced=True, section=S3),
    # --- limits, logging and diagnostics
    Flag(("--log-json",), "limits", "one JSON object per log event instead of text lines", {"action": "store_true"}),
    Flag(("--max-session-s",), "limits", "audio per connection before the server flushes and closes it",
         {"type": float, "default": DEFAULT_MAX_SESSION_S, "metavar": "S"}, advanced=True),
    Flag(("--idle-timeout-s",), "limits", "close a connection that sends nothing for this long",
         {"type": float, "default": DEFAULT_IDLE_TIMEOUT_S, "metavar": "S"}, advanced=True),
    Flag(("--debug-fields",), "limits", "add diagnostic keys to the messages and validate every message",
         {"action": "store_true"}, advanced=True, section=S10),
)

# environment variables the server and the downloader read (docs/CONFIGURATION.md section 12)
ENV_VARS = {
    "AUDIOFORGE_HOME": "where `audioforge-download` writes the models and `audioforge-serve` finds them (default "
                       "`<repo>/models` in a checkout, else `~/.cache/audioforge`)",
    "AUDIOFORGE_DATA": "data root: datasets, raw NVIDIA `.nemo` files (`nemo/`), Silero (`silero/`); default `<repo>/data`",
    "AUDIOFORGE_CONFIG": "config file used when `--config` is not given",
    "AUDIOFORGE_ACCEPT_LICENSES": "`1` accepts the model licences in `audioforge-download` without asking",
}

_BY_DEST = {f.dest: f for f in FLAGS}


class _Formatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=34, width=118)


def _default_text(f: Flag) -> str:
    if f.doc_default is not None:
        return f.doc_default
    k = f.kwargs
    if k.get("action") == "store_true":
        return "off"
    d = k.get("default")
    return "none" if d is None else f"`{d}`"


def build_parser(prog: str = "audioforge.serve", advanced: bool = False, launcher: bool = False
                 ) -> argparse.ArgumentParser:
    """The argparse parser: every flag of ``FLAGS`` (launcher flags only when ``launcher``); advanced flags are
    parsed either way but listed by ``--help`` only when ``advanced``. No flag is ``required`` here: ``--asr`` /
    ``--diar`` may come from the config file (checked in ``parse_args``)."""
    epilog = ("Everyday flags only; --help-advanced lists all of them. Reference: docs/CONFIGURATION.md."
              if not advanced else "Reference with measured costs: docs/CONFIGURATION.md.")
    ap = argparse.ArgumentParser(prog=prog, usage="%(prog)s [options]", description=DESCRIPTION, epilog=epilog,
                                 add_help=False, formatter_class=_Formatter)
    groups = {g: ap.add_argument_group(title) for g, title in GROUPS.items()}
    for f in FLAGS:
        if f.launcher and not launcher:
            continue
        kw = dict(f.kwargs)
        d = kw.get("default")
        text = f.help + (f" (default: {d})" if d not in (None, False, []) else "")
        kw["help"] = argparse.SUPPRESS if f.advanced and not advanced else text.replace("%", "%%")
        groups[f.group].add_argument(*f.flags, **kw)
    h = ap.add_argument_group("help")
    h.add_argument("-h", "--help", action="store_true", help="show the everyday flags and exit")
    h.add_argument("--help-advanced", action="store_true", help="show every flag and exit")
    return ap


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Read a YAML or JSON config file into ``{dest: value}`` (keys by long flag name, dashes or underscores).
    Unknown keys raise ``ValueError``; values are converted and checked like command-line values."""
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        raw = json.loads(text)
    else:
        import yaml
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of option: value")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        dest = str(key).lstrip("-").replace("-", "_")
        f = _BY_DEST.get(dest)
        if f is None or dest in ("config",):
            raise ValueError(f"{path}: unknown option {key!r} (see audioforge-serve --help-advanced)")
        out[dest] = _convert(f, value, str(path))
    return out


def _convert(f: Flag, value: Any, where: str) -> Any:
    k = f.kwargs
    name = f.flags[0]
    if k.get("action") == "store_true":
        if not isinstance(value, bool):
            raise ValueError(f"{where}: {name} must be true or false, got {value!r}")
        return value
    if k.get("action") == "append":
        return [str(v) for v in (value if isinstance(value, list) else [value])]
    if value is None:
        return None
    typ = k.get("type", str)
    try:
        v = typ(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: {name}: {e}") from e
    if "choices" in k and v not in k["choices"]:
        raise ValueError(f"{where}: {name} must be one of {list(k['choices'])}, got {value!r}")
    return v


def parse_args(argv: list[str] | None = None, prog: str = "audioforge.serve", launcher: bool = False
               ) -> argparse.Namespace:
    """Parse ``argv`` on top of the config file (``--config`` / ``$AUDIOFORGE_CONFIG``); prints help and exits for
    ``-h`` / ``--help-advanced``; exits 2 when ``--asr`` / ``--diar`` are missing (not for the launcher, which fills
    them in; ``--diar`` may be left out with ``--diar-off``)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.environ.get("AUDIOFORGE_CONFIG") or None)
    pre.add_argument("-h", "--help", action="store_true")
    pre.add_argument("--help-advanced", action="store_true")
    known, _ = pre.parse_known_args(argv)
    if known.help or known.help_advanced:
        build_parser(prog, advanced=known.help_advanced, launcher=launcher).print_help()
        raise SystemExit(0)
    ap = build_parser(prog, launcher=launcher)
    if known.config:
        try:
            cfg = load_config(known.config)
        except (OSError, ValueError) as e:
            ap.error(str(e))
        if not launcher:
            cfg = {k: v for k, v in cfg.items() if not _BY_DEST[k].launcher}
        ap.set_defaults(**cfg)
    a = ap.parse_args(argv)
    if not launcher:
        missing = [f for f in ("--asr", "--diar") if getattr(a, f[2:]) is None and not (f == "--diar" and a.diar_off)]
        if missing:
            ap.error(f"the following arguments are required: {', '.join(missing)} (or set them in --config)")
    return a


def _startup_checks(a) -> None:
    """Fail fast with one clear line for a missing model file (never at the first frame)."""
    from ..enrollment import TITANET_NEMO
    missing = [(f, p) for f, p in (("--asr", a.asr), ("--diar", a.diar)) if p is not None and not Path(p).exists()]
    if a.turn_input == "tsvad" and a.tsvad is not None and not Path(a.tsvad).exists():
        missing.append(("--tsvad", a.tsvad))
    if a.silero is not None and not Path(a.silero).exists():
        missing.append(("--silero", a.silero))
    if a.enroll in VOICE_MODES and a.turn_input != "tsvad" and not Path(a.titanet or TITANET_NEMO).exists():
        missing.append(("--titanet (TitaNet-L for --enroll %s)" % a.enroll, str(a.titanet or TITANET_NEMO)))
    if a.lid == "head":
        from ..lid import HEAD_FILE, resolve_head
        try:
            resolve_head("head")
        except FileNotFoundError:
            missing.append(("--lid head", f"{HEAD_FILE} (models directory or runs/)"))
    elif a.lid and a.lid != "ambernet" and not Path(a.lid).exists():
        missing.append(("--lid", a.lid))
    if a.final_asr and a.final_asr.endswith(".nemo") and not Path(a.final_asr).exists():
        missing.append(("--final-asr", a.final_asr))
    if missing:
        for f, p in missing:
            print(f"[serve] error: {f} file not found: {p}", file=sys.stderr, flush=True)
        raise SystemExit(2)


def load_engine(a):
    """The ``Engine`` for parsed arguments ``a`` (``parse_args``), exactly as ``main`` builds it."""
    import torch

    from ..serve import Engine  # the server module imports this one

    torch.set_num_threads(a.threads)
    dcfg = {}
    for kv in a.diar_set:
        k, v = kv.split("=", 1)
        dcfg[k] = float(v) if "." in v else int(v)
    return Engine.load(a.asr, a.diar, a.device, diar_pool=a.diar_pool, diar_spks=a.diar_spks, threads=a.threads,
                       turn_input=a.turn_input, diar_config=a.diar_config, diar_cfg=dcfg,
                       diar_left=a.diar_left, debug=a.debug_fields, fast=not a.no_fast_conv, enroll=a.enroll,
                       titanet=a.titanet, enroll_stride=a.enroll_stride, silero=a.silero,
                       silero_timeout_ms=a.silero_timeout_ms, preload_silero=a.silero is not None,
                       lid=a.lid, lid_threshold=a.lid_threshold, lid_min_ms=a.lid_min_ms,
                       lid_langs=a.lid_langs.split(",") if a.lid_langs else None, lid_max_ms=a.lid_max_ms,
                       final_asr=a.final_asr,
                       final_asr_worker=a.final_asr_worker, final_asr_threads=a.final_asr_threads,
                       final_asr_device=a.final_asr_device, asr_lookahead=a.asr_lookahead, asr_chunk_ms=a.asr_chunk_ms,
                       asr_vad_gate=a.asr_vad_gate, asr_vad_hangover_ms=a.asr_vad_hangover_ms, asr_beam=a.beam,
                       max_session_s=a.max_session_s, idle_timeout_s=a.idle_timeout_s, log_json=a.log_json,
                       perf=a.perf, tsvad=a.tsvad, tsvad_print_s=a.tsvad_print_s,
                       tsvad_refresh_s=a.tsvad_refresh_s, diar_off=a.diar_off, diar_labels=a.diar_labels,
                       diar_embed=a.diar_embed, shed_diar=a.shed_diar, registry_thr=a.diar_reg_thr,
                       dyn_wait_ms=a.dyn_wait_ms, vad_wait_ms=a.vad_wait_ms, others_wait_ms=a.others_wait_ms,
                       turn_policy=a.turn_policy, turn_hint_p=None if a.turn_hint_off else a.turn_hint_p,
                       turn_preset=a.turn_preset, energy_gate=a.energy_gate == "on",
                       energy_quiet_db=a.energy_quiet_db or None, turn_model=a.turn_model,
                       smartturn_onnx=a.smartturn_onnx, smartturn_trigger=a.smartturn_trigger)


def main(argv: list[str] | None = None) -> int:
    """``python -m audioforge.serve``: load the models and serve until interrupted."""
    from ..serve import serve  # the server module imports this one

    a = parse_args(argv)
    _startup_checks(a)
    try:
        eng = load_engine(a)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:  # noqa: BLE001 - one clear line first, then the traceback for the log
        print(f"[serve] fatal: could not load the models: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    if a.enroll in VOICE_MODES and eng.embedder is not None:
        print(f"[serve] enrollment {a.enroll}: TitaNet-L loaded (+{eng.enroll_rss_mb} MB RSS), stride "
              f"{a.enroll_stride} frames", flush=True)
    lo, hi, mean = eng.lag
    if eng.diar is None:
        print("[serve] single model: no diarizer loaded; speaker columns = the TS-VAD track of the user's voice print "
              "(ready with its 160 ms chunk)", flush=True)
    else:
        print(f"[serve] diarizer lag after frame end: {lo:.0f}-{hi:.0f} ms (mean {mean:.0f}) + compute", flush=True)
    if not a.no_warmup:
        t0 = time.time()
        eng.warmup()
        print(f"[serve] warmup {time.time() - t0:.1f}s", flush=True)
    try:
        asyncio.run(serve(eng, a.host, a.port))
    except KeyboardInterrupt:
        pass
    return 0


def reference_markdown() -> str:
    """The flag and environment tables of docs/CONFIGURATION.md, generated from ``FLAGS`` / ``ENV_VARS``."""
    out = []
    for g, title in GROUPS.items():
        fs = [f for f in FLAGS if f.group == g]
        if not fs:
            continue
        out += [f"**{title}**", "", "| flag | default | meaning | more |", "|---|---|---|---|"]
        for f in fs:
            k = f.kwargs
            if k.get("action") == "store_true":
                shown = f.flags[0]
            elif "choices" in k:
                shown = f"{f.flags[0]} {{{','.join(map(str, k['choices']))}}}"
            else:
                shown = f"{f.flags[0]} {k.get('metavar') or f.dest.upper()}"
            tags = []
            if f.launcher:
                tags.append("`audioforge-serve` only")
            if f.advanced:
                tags.append("advanced")
            meaning = (f.doc or f.help) + (f" ({', '.join(tags)})" if tags else "")
            more = f"[§{f.section.lstrip('#').split('-')[0]}]({f.section})" if f.section else ""
            more = more.replace("§71", "§7.1").replace("§72", "§7.2").replace("§73", "§7.3").replace("§74", "§7.4")
            out.append(f"| `{shown.replace('|', '&#124;')}` | {_default_text(f)} | {meaning.replace('|', '&#124;')} "
                       f"| {more} |")
        out.append("")
    out += ["**environment**", "", "| variable | meaning |", "|---|---|"]
    out += [f"| `{k}` | {v} |" for k, v in ENV_VARS.items()]
    return "\n".join(out) + "\n"
