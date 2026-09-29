"""Streaming WebSocket server: live ASR + VAD + speaker activity + end-of-turn events, one session per connection.

    audioforge-serve                                    # the downloaded models, ws://127.0.0.1:8765
    python -m audioforge.serve --asr models/stage1_served_v2.afm --diar models/nemo_sortformer_v2.afm [flags]

Two models run side by side on one 80 ms frame clock: the frozen NVIDIA cache-aware FastConformer with our heads
(ASR, VAD, turn, speaker; 160 ms chunks) and NVIDIA's streaming diarizer (Sortformer v2 or Nemotron-3). Each
connection gets a ``Session``; the client sends PCM and receives ``ready`` / ``frame`` / ``partial`` / ``turn_end`` /
``final`` / ``stats`` JSON (docs/PROTOCOL.md). Flags: docs/CONFIGURATION.md. Design notes (frame clock, policies,
enrollment, final ASR): docs/SERVER_INTERNALS.md.

In-process use without a socket (what ``audioforge-bench`` and ``audioforge.load`` do)::

    engine = Engine.load("models/stage1_served_v2.afm", "models/nemo_sortformer_v2.afm")
    session = Session(engine, SessionConfig(turn_policy="timeout"))
    messages = session.process(samples_16k_float32) + session.finish()

This module holds the engine, the sessions and the connection loop; the parts they are built from live in
``audioforge.server`` (``constants``, ``protocol``, ``policies``, ``binding``, ``streams``, ``cli``) and are re-exported
here, so ``audioforge.serve.X`` keeps working. Module-level limits (``SHED_RTF``, ``MAX_SEGMENT_S``, ``FINAL_ASR_*``)
are read at call time and may be overridden on this module.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .server.binding import ArmBinder, CausalDominant, VoiceBinder
from .server.cli import main
from .server.constants import (
    ACT_THRESHOLD,
    CHUNK_MS,
    CHUNK_SAMPLES,
    DEFAULT_DIAR_CONFIG,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_MAX_SESSION_S,
    DEFAULT_SILERO,
    DEFAULT_THETA,
    DEFAULT_TSVAD,
    DIAR_CONFIGS,
    DIAR_ENC_LEFT,
    DIAR_LABEL_MODES,
    DIAR_LOW_LATENCY,
    DIAR_LOW_LATENCY_032,
    DYN_A,
    DYN_OFFSET,
    DYN_T0,
    DYN_TMIN,
    ENROLL_MODES,
    ENROLL_TRIGGER,
    ERROR_CODES,
    FINAL_ASR_FAILS_BEFORE_RESTART,
    FINAL_ASR_RESTART_S,
    FINAL_ASR_TIMEOUT_S,
    FINAL_MAX_TURN_S,
    FINAL_POST_ROLL_S,
    FINAL_PRE_ROLL_S,
    FRAME_MS,
    FRAME_SAMPLES,
    HYBRID_POLICIES,
    LOOKAHEAD_MARGIN_FRAMES,
    MAX_ABS_SAMPLE,
    MAX_CHANNELS,
    MAX_INBOX_S,
    MAX_ROUND_SAMPLES,
    MAX_SAMPLE_RATE,
    MAX_SEGMENT_S,
    MAX_TEXT_MESSAGE,
    MIN_SAMPLE_RATE,
    PCM_FORMATS,
    POLICIES,
    POLICY_THETA,
    PRIMARY_WINDOW_S,
    REGISTRY_AUDIO_S,
    SHED_DIAR_MODES,
    SHED_DIAR_MS,
    SHED_NOTICE_S,
    SHED_PARTIAL_MS,
    SHED_RTF,
    SHED_WINDOW_S,
    SILERO_POLICIES,
    SILERO_TIMEOUT_MS,
    SR,
    STAT_KEEP,
    TSVAD_DYN,
    VOICE_MODES,
    diar_lag_ms,
    diar_preset,
)
from .server.policies import AnySpeakerTimeout, HeadPolicy, SileroSilence, TimeoutPolicy
from .server.protocol import (
    DEBUG_KEYS,
    ENROLL_KEYS,
    FINAL_ASR_EXTRA,
    FINAL_ASR_KEYS,
    LID_KEYS,
    OPTIONAL_KEYS,
    SCHEMA,
    SessionConfig,
    decode_pcm,
    error_msg,
    validate,
)
from .server.streams import ASRStream, LookaheadStream, Resampler, fast_conv
from .server.util import _pct, _Ring, peak_rss_mb, rss_mb

log = logging.getLogger(__name__)

__all__ = [
    "ACT_THRESHOLD",
    "AnySpeakerTimeout",
    "ArmBinder",
    "ASRStream",
    "CausalDominant",
    "CHUNK_MS",
    "CHUNK_SAMPLES",
    "DEBUG_KEYS",
    "decode_pcm",
    "DEFAULT_DIAR_CONFIG",
    "DEFAULT_IDLE_TIMEOUT_S",
    "DEFAULT_MAX_SESSION_S",
    "DEFAULT_SILERO",
    "DEFAULT_THETA",
    "DEFAULT_TSVAD",
    "DIAR_CONFIGS",
    "DIAR_ENC_LEFT",
    "DIAR_LABEL_MODES",
    "diar_lag_ms",
    "DIAR_LOW_LATENCY",
    "DIAR_LOW_LATENCY_032",
    "diar_preset",
    "DYN_A",
    "DYN_OFFSET",
    "DYN_T0",
    "DYN_TMIN",
    "Engine",
    "ENROLL_KEYS",
    "ENROLL_MODES",
    "ENROLL_TRIGGER",
    "ERROR_CODES",
    "error_msg",
    "fast_conv",
    "FINAL_ASR_EXTRA",
    "FINAL_ASR_FAILS_BEFORE_RESTART",
    "FINAL_ASR_KEYS",
    "FINAL_ASR_RESTART_S",
    "FINAL_ASR_TIMEOUT_S",
    "FINAL_MAX_TURN_S",
    "FINAL_POST_ROLL_S",
    "FINAL_PRE_ROLL_S",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "handle",
    "HeadPolicy",
    "HYBRID_POLICIES",
    "LID_KEYS",
    "LOOKAHEAD_MARGIN_FRAMES",
    "LookaheadStream",
    "main",
    "MAX_ABS_SAMPLE",
    "MAX_CHANNELS",
    "MAX_INBOX_S",
    "MAX_ROUND_SAMPLES",
    "MAX_SAMPLE_RATE",
    "MAX_SEGMENT_S",
    "MAX_TEXT_MESSAGE",
    "MIN_SAMPLE_RATE",
    "OPTIONAL_KEYS",
    "PCM_FORMATS",
    "peak_rss_mb",
    "POLICIES",
    "POLICY_THETA",
    "PRIMARY_WINDOW_S",
    "REGISTRY_AUDIO_S",
    "Resampler",
    "rss_mb",
    "SCHEMA",
    "serve",
    "Session",
    "SessionConfig",
    "SHED_DIAR_MODES",
    "SHED_DIAR_MS",
    "SHED_NOTICE_S",
    "SHED_PARTIAL_MS",
    "SHED_RTF",
    "SHED_WINDOW_S",
    "SILERO_POLICIES",
    "SILERO_TIMEOUT_MS",
    "SileroSilence",
    "SR",
    "STAT_KEEP",
    "TimeoutPolicy",
    "TSVAD_DYN",
    "validate",
    "VOICE_MODES",
    "VoiceBinder",
]


class Engine:
    """Both models, loaded once, shared by every session (compute is serialized on one worker thread)."""

    def __init__(self, asr_model, diar_model, name: str = "audioforge", threads: int | None = 2,
                 turn_input: str = "auto", diar_config: str = DEFAULT_DIAR_CONFIG, diar_cfg: dict | None = None,
                 diar_left: int = DIAR_ENC_LEFT,
                 debug: bool = False, fast: bool = True, enroll: str = "dominant", embedder=None,
                 titanet: str | None = None, enroll_stride: int = 5, silero: str | None = None,
                 silero_timeout_ms: int = SILERO_TIMEOUT_MS, preload_silero: bool = False,
                 lid: str | None = None, lid_threshold: float = 0.9, lid_min_ms: float = 1000.0,
                 lid_langs: list | None = None, lid_max_ms: float | None = None, final_asr=None, final_asr_worker: str = "process",
                 final_asr_threads: int | None = 2, final_asr_device: str = "cpu", asr_lookahead: int | None = None,
                 asr_vad_gate: float | None = None, asr_vad_hangover_ms: float = 1200.0,
                 max_session_s: float = DEFAULT_MAX_SESSION_S, idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S,
                 log_json: bool = False, perf: str | None = "default", tsvad: str | None = None,
                 tsvad_print_s: float = 5.0, tsvad_refresh_s: float = 0.0, diar_off: bool = False,
                 diar_labels: str = "column", diar_embed: str = "spk", shed_diar: str = "vad",
                 registry_thr: float | None = None, dyn_wait_ms: str | tuple | None = None):
        self.threads = threads
        self.asr_lookahead = int(asr_lookahead) if asr_lookahead else None
        self.asr_vad_gate = None if asr_vad_gate is None else float(asr_vad_gate)
        self.asr_vad_hangover = max(1, int(round(float(asr_vad_hangover_ms) / FRAME_MS)))
        self.max_session_s, self.idle_timeout_s = float(max_session_s), float(idle_timeout_s)
        self.log_json = bool(log_json)
        # robustness bookkeeping (research/BULLETPROOF.md): every degradation and refused message is counted here
        # (Engine.health()), and every live connection is registered so the watchdog can see the global backlog
        self.counters: dict[str, int] = {}
        self.counter_lock = threading.Lock()
        self.conns: set = set()
        self.sessions_total = 0
        self.t_start = time.time()
        self._fa_restart_t, self._fa_restarting, self._fa_fails = -math.inf, False, 0
        if threads:
            torch.set_num_threads(threads)
        if enroll not in ENROLL_MODES:
            raise ValueError(f"unknown --enroll {enroll!r} (one of {ENROLL_MODES})")
        if diar_model is None and not diar_off:  # --diar-off (single-model mode) is the only way to run without one
            raise ValueError("no diarizer model: pass --diar, or --turn-input tsvad --diar-off (--mode single)")
        self.enroll, self.enroll_stride, self.enroll_rss_mb = enroll, int(enroll_stride), 0.0
        self.embedder = embedder
        if enroll in VOICE_MODES and embedder is None and turn_input != "tsvad":  # TitaNet-L, only for a voice mode
            from .enrollment import TitaNetEmbedder
            r0 = rss_mb()
            self.embedder = TitaNetEmbedder(path=titanet)
            self.embedder.frames(np.zeros(FRAME_SAMPLES * 8, np.float32), [np.arange(8)])  # warm; counts its RSS
            self.enroll_rss_mb = round(rss_mb() - r0, 1)
        self.asr, self.name, self.debug = asr_model.eval(), name, debug
        self.diar = diar_model.eval() if diar_model is not None else None  # None: --diar-off, the diarizer never loaded
        self.lid_name, self.lid_threshold, self.lid_min_ms = None, float(lid_threshold), float(lid_min_ms)
        self.lid_model, self.lid_langs = None, list(lid_langs) if lid_langs else None
        self.lid_max_ms = float(lid_max_ms) if lid_max_ms else None  # 0 / None: no timeout
        if lid in ("ambernet",) or (lid and str(lid).endswith(".nemo")):  # the dedicated backend (research/LID.md)
            from .nemo_import import import_ambernet
            self.lid_model = import_ambernet(None if lid == "ambernet" else lid)
            self.lid_name = "ambernet"
        elif lid:  # spoken language ID head (research/LID.md), attached to the ASR model's encoder
            from .lid import HEAD_MAX_MS, attach_head, resolve_head
            if lid == "head" and lid_max_ms is None:  # the shipped head and its pre-registered rule (fix pass)
                self.lid_max_ms = HEAD_MAX_MS
            self.lid_name = attach_head(self.asr, resolve_head(lid))
        self.fast = fast
        if fast:  # CPU fast path for the conformer conv modules of both encoders (same outputs)
            fast_conv(self.asr)
            if self.diar is not None:
                fast_conv(self.diar)
        # --perf: the CPU inference fast paths of audioforge.perf (research/PERFORMANCE.md section 3); the default
        # set is exact (same outputs as without it), "none" turns them off
        from . import perf as _perf
        self.perf_opts = _perf.parse(perf)
        self.perf_info = _perf.apply(self.asr, self.diar, self.perf_opts)
        if diar_model is not None:
            self.diar_head = next(k for k, v in diar_model.head_cfg.items() if v["type"] == "sortformer")
            self.num_spks = diar_model.heads[self.diar_head].num_spks
        else:  # the TS-VAD columns [P(user), P(other), 0, 0] (tsvad_stream.TSVADColumns)
            self.diar_head, self.num_spks = None, 4
        self.turn_name = next((k for k, v in asr_model.head_cfg.items() if v["type"] == "turn"), None)
        self.vad_name = next((k for k, v in asr_model.head_cfg.items()
                              if v["type"] == "frame" and v.get("key") == "vad"), None)
        h = asr_model.heads[self.turn_name] if self.turn_name else None
        needs_diar = h is not None and (getattr(h, "needs_act", getattr(h, "concat", False))
                                        or getattr(h, "needs_cols", False))
        self.turn_input = ("diar" if needs_diar else "session") if turn_input == "auto" else turn_input
        if needs_diar and self.turn_input == "session":
            raise ValueError("this turn head reads the diarizer's activity itself: use --turn-input diar")
        # --turn-input tsvad: the TS-VAD head on block 4 of the ASR pass gives the turn head the enrolled user's
        # activity (audioforge.tsvad_stream); the diarizer still runs for frame.speakers / primary / timeout
        self.tsvad, self.dyn_offset = None, DYN_OFFSET
        # --dyn-wait-ms CAP,FLOOR: hybrid_dyn's Silero-silence wait at head p = 0 / p = 1 (default: the served rule,
        # DYN_T0 / DYN_T0 - DYN_A frames); research/SINGLE_MODEL.md A1 picked 2000,960 for --mode single
        self.dyn_t0 = self.dyn_a = None
        if dyn_wait_ms:
            cap, floor = (float(x) for x in (dyn_wait_ms.split(",") if isinstance(dyn_wait_ms, str) else dyn_wait_ms))
            if not 0 < floor <= cap:
                raise ValueError(f"--dyn-wait-ms {dyn_wait_ms!r}: needs 0 < FLOOR <= CAP")
            self.dyn_t0, self.dyn_a = cap / FRAME_MS, (cap - floor) / FRAME_MS
        self.tsvad_print_s, self.tsvad_refresh_s = float(tsvad_print_s), float(tsvad_refresh_s)
        if self.turn_input == "tsvad":
            if not (needs_diar or (h is not None and asr_model.head_cfg[self.turn_name].get("condition_on_speaker"))):
                raise ValueError("--turn-input tsvad needs a turn head that reads a speaker-activity track")
            if len(asr_model.layer_tap.get("spk", [])) != 1:
                raise ValueError("--turn-input tsvad needs a single-layer speaker head (served: heads.spk from_layers [3])")
            from .tsvad_stream import load_tsvad
            self.tsvad = tsvad if isinstance(tsvad, torch.nn.Module) else load_tsvad(str(tsvad or DEFAULT_TSVAD),
                                                                                         asr_model.encoder.d_model)
            self.dyn_offset = TSVAD_DYN[1]
        # --diar-labels / --shed-diar (research/DIARIZATION_FIX.md): stable voice-keyed ids on the finals and the
        # last-stable-column rule under load shedding; defaults keep the legacy behaviour
        if diar_labels not in DIAR_LABEL_MODES or shed_diar not in SHED_DIAR_MODES or diar_embed not in ("spk", "titanet"):
            raise ValueError(f"--diar-labels {diar_labels!r} / --shed-diar {shed_diar!r} / --diar-embed {diar_embed!r}")
        self.diar_labels, self.diar_embed, self.shed_diar = diar_labels, diar_embed, shed_diar
        from .speaker_registry import DEFAULT_THR
        self.registry_thr = float(registry_thr) if registry_thr is not None else DEFAULT_THR[diar_embed]
        self.registry_embedder = None
        if diar_labels == "registry":
            if diar_embed == "spk":
                if "spk" not in asr_model.heads:
                    raise ValueError("--diar-labels registry --diar-embed spk needs a speaker head ('spk') in --asr")
                from .enrollment import ColumnEmbedder
                self.registry_embedder = ColumnEmbedder(asr_model.heads["spk"])
            elif self.embedder is not None and hasattr(self.embedder, "frames"):
                self.registry_embedder = self.embedder  # TitaNet already loaded for --enroll
            else:
                from .enrollment import TitaNetEmbedder
                self.registry_embedder = TitaNetEmbedder(path=titanet)
        # --diar-off (only with tsvad): the session's "diarizer" columns are the TS-VAD track (tsvad_stream.TSVADColumns)
        self.diar_off = bool(diar_off)
        if self.diar_off and self.tsvad is None:
            raise ValueError("--diar-off needs --turn-input tsvad (the TS-VAD track replaces the diarizer's columns)")
        # diar_config names a preset; diar_cfg (e.g. --diar-set) overrides single AOSC fields on top of it
        base = diar_preset(diar_config)
        self.diar_cfg = {**base, **(diar_cfg or {})}
        self.diar_config = diar_config if self.diar_cfg == base else f"{diar_config}+custom"
        self.diar_left = diar_left
        self.diar_mode = "off" if diar_model is None else "causal" if diar_model.encoder.causal else "window"
        asr_frame_ms = asr_model.frame_sec * 1000
        assert abs(asr_frame_ms - FRAME_MS) < 1e-6 and (
            diar_model is None or abs(diar_model.frame_sec * 1000 - FRAME_MS) < 1e-6), \
            "both models must run on the 80 ms frame clock"
        att = asr_model.encoder.att_context_size
        self.chunk_ms = (att[1] + 1) * FRAME_MS
        self.lag = diar_lag_ms(2, 0) if diar_off else diar_lag_ms(self.diar_cfg["chunk_len"],
                                                                 self.diar_cfg["chunk_right_context"])
        # Silero VAD v5 for hybrid_silero / hybrid_dyn: loaded at the first session that asks for such a policy
        # (or at start with preload_silero); ``silero_model`` may be injected (tests)
        if silero is None and not Path(DEFAULT_SILERO).exists():  # then where audioforge-download put it (models dir)
            from . import hub
            silero = hub.find_model("silero")
        self.silero_path = str(silero or DEFAULT_SILERO)
        self.silero_timeout_ms = int(silero_timeout_ms)
        self.silero_model = None
        self.silero_load_s: float | None = None
        self.silero_error: str | None = None  # set when the file is missing / failed to load (sessions fall back)
        if not Path(self.silero_path).exists():
            self.silero_error = f"Silero VAD file not found: {self.silero_path}"
            if preload_silero:
                raise FileNotFoundError(f"--silero: {self.silero_error}")
            log.warning(f"[serve] warning: {self.silero_error}; turn_policy hybrid_silero / hybrid_dyn will fall back to "
                  f"hybrid")
        elif preload_silero:
            self.get_silero()
        self.executor = ThreadPoolExecutor(1, thread_name_prefix="audioforge-serve",
                                           initializer=(lambda: torch.set_num_threads(threads)) if threads else None)
        self.lock = threading.Lock()
        # --final-asr: a spec string (audioforge.final_asr.FinalASRWorker) or an injected object with
        # submit(audio) -> Future[FinalResult], .source, .rss_mb (tests); None = off (protocol unchanged)
        self.final_asr = None
        if isinstance(final_asr, str):
            from .final_asr import FinalASRWorker
            t0 = time.perf_counter()
            self.final_asr = FinalASRWorker(final_asr, mode=final_asr_worker, threads=final_asr_threads,
                                            device=final_asr_device)
            log.info(f"[serve] final ASR {final_asr} ({final_asr_worker}, {final_asr_device}, {final_asr_threads} threads) "
                  f"loaded in {time.perf_counter() - t0:.1f}s, worker RSS {self.final_asr.rss_mb} MB")
        elif final_asr is not None:
            self.final_asr = final_asr
        self.final_sources = (["lookahead"] if self.asr_lookahead else []) + (
            [self.final_asr.source] if self.final_asr is not None else [])

    def get_silero(self):
        """The shared Silero VAD v5 session (``baselines.turn.SileroStream``), loaded on first use. Raises
        (FileNotFoundError / the ORT error) when it cannot be loaded; ``silero_error`` then holds the reason."""
        if self.silero_model is None:
            from .baselines.turn import SileroStream
            if not Path(self.silero_path).exists():
                self.silero_error = f"Silero VAD file not found: {self.silero_path}"
                raise FileNotFoundError(self.silero_error)
            t0 = time.perf_counter()
            try:
                self.silero_model = SileroStream(self.silero_path)
            except Exception as e:  # noqa: BLE001 - onnxruntime missing / corrupt file: reported, sessions fall back
                self.silero_error = f"Silero VAD failed to load: {type(e).__name__}: {e}"
                raise
            self.silero_error = None
            self.silero_load_s = round(time.perf_counter() - t0, 3)
            log.info(f"[serve] Silero VAD loaded from {self.silero_path} in {self.silero_load_s}s")
        return self.silero_model

    # ---------------------------------------------------------------- robustness bookkeeping
    def count(self, code: str, n: int = 1) -> None:
        """Add ``n`` to the server-wide counter ``code`` (reported by ``health()`` and ``/health``)."""
        with self.counter_lock:
            self.counters[code] = self.counters.get(code, 0) + int(n)

    def event(self, event: str, **kw) -> None:
        """One structured log line: JSON with --log-json, else ``[serve] peer: event k=v ...``."""
        if self.log_json:
            log.info(json.dumps({"ts": round(time.time(), 3), "event": event, **kw}, default=str))
        else:
            peer = kw.pop("peer", None)
            tail = " ".join(f"{k}={json.dumps(v, default=str) if not isinstance(v, str) else v}" for k, v in kw.items())
            log.info(f"[serve] {peer + ': ' if peer else ''}{event}{' ' + tail if tail else ''}")

    def global_backlog_ms(self) -> float:
        """Audio queued but not yet processed over every live connection (ms)."""
        return sum(c.n_in for c in list(self.conns)) / SR * 1000

    def global_load(self) -> float:
        """The compute thread's recent load: the sum of every live session's recent_rtf() (>= SHED_RTF means the
        sessions together need more than the one thread can give in real time)."""
        return sum(c.session.recent_rtf() for c in list(self.conns) if c.session is not None)

    def global_shed(self) -> int:
        """The watchdog's shedding level for every session: the global backlog per connection crosses the same
        thresholds as a single session's, but only while the thread is loaded (global_load() >= SHED_RTF)."""
        n = max(1, len(self.conns))
        g = self.global_backlog_ms()
        want = 2 if g >= SHED_PARTIAL_MS * n else 1 if g >= SHED_DIAR_MS * n else 0
        return want if want and self.global_load() >= SHED_RTF else 0

    def health(self) -> dict:
        """The /health JSON: liveness, load, and the counters of every degradation / refused message."""
        with self.counter_lock:
            counters = dict(sorted(self.counters.items()))
        conns = list(self.conns)
        fa = self.final_asr
        return {"ok": True, "model": self.name, "uptime_s": round(time.time() - self.t_start, 1),
                "sessions_active": len(conns), "sessions_total": self.sessions_total,
                "backlog_ms_total": round(self.global_backlog_ms(), 1),
                "backlog_ms_max": round(max([c.n_in for c in conns], default=0) / SR * 1000, 1),
                "load": round(self.global_load(), 3), "shed_level": self.global_shed(),
                "rss_mb": round(rss_mb(), 1), "peak_rss_mb": peak_rss_mb(), "threads": self.threads,
                "final_asr": None if fa is None else {"source": getattr(fa, "source", None),
                                                      "alive": bool(getattr(fa, "alive", True)),
                                                      "restarting": self._fa_restarting},
                "silero": "loaded" if self.silero_model is not None else (self.silero_error or "not loaded"),
                "limits": {"max_session_s": self.max_session_s, "idle_timeout_s": self.idle_timeout_s,
                           "max_inbox_s": MAX_INBOX_S, "shed_diar_ms": SHED_DIAR_MS, "shed_partial_ms": SHED_PARTIAL_MS},
                "counters": counters}

    def submit_final(self, audio):
        """--final-asr: ``final_asr.submit`` that never raises: a dead / failing worker yields a failed Future (the
        caller falls back to the streaming final) and a restart is attempted in the background, at most once per
        FINAL_ASR_RESTART_S."""
        from concurrent.futures import Future
        fa = self.final_asr
        try:
            if not getattr(fa, "alive", True):
                raise RuntimeError("final-ASR worker is not alive")
            return fa.submit(audio)
        except Exception as e:  # noqa: BLE001
            self.count("final_asr_submit_failed")
            self._maybe_restart_final_asr(e)
            f: Future = Future()
            f.set_exception(e)
            return f

    def final_done(self, err=None, hung: bool = False) -> None:
        """Outcome of one offline final: a success resets the failure streak; a failure counts towards
        FINAL_ASR_FAILS_BEFORE_RESTART (a worker that is alive but fails every job is wedged); a hang (no answer
        within FINAL_ASR_TIMEOUT_S) or a dead worker restarts at once."""
        if err is None:
            self._fa_fails = 0
            return
        self._fa_fails += 1
        fa = self.final_asr
        if hung or not getattr(fa, "alive", True) or self._fa_fails >= FINAL_ASR_FAILS_BEFORE_RESTART:
            if self._maybe_restart_final_asr(err):
                self._fa_fails = 0

    def _maybe_restart_final_asr(self, err) -> bool:
        fa = self.final_asr
        now = time.time()
        if not hasattr(fa, "restart") or self._fa_restarting or now - self._fa_restart_t < FINAL_ASR_RESTART_S:
            return False
        self._fa_restart_t, self._fa_restarting = now, True
        self.event("final_asr_restart", reason=f"{type(err).__name__}: {err}")

        def run():
            try:
                fa.restart()
                self.count("final_asr_restarted")
                self.event("final_asr_restarted", rss_mb=getattr(fa, "rss_mb", None))
            except Exception as e2:  # noqa: BLE001
                self.count("final_asr_restart_failed")
                self.event("final_asr_restart_failed", error=f"{type(e2).__name__}: {e2}")
            finally:
                self._fa_restarting = False

        threading.Thread(target=run, name="final-asr-restart", daemon=True).start()
        return True

    @classmethod
    def load(cls, asr_path: str, diar_path: str | None, device: str = "cpu", diar_pool: str | None = None,
             diar_spks: int | None = None, **kw) -> "Engine":
        """Load the ASR + heads ``.afm`` and the diarizer (any device falls back to CPU) and build the engine; ``kw``
        are the server options (the ``audioforge.serve`` flags with underscores). ``diar_path`` None (only with
        ``diar_off``, ``--mode single``): no diarizer is loaded at all."""
        from .nemo_import import load_any
        from .train import load_model
        fallback = None
        if device != "cpu":  # the server's streaming path is CPU-only (fast-conv, per-frame decoding): degrade, not die
            fallback = device
            log.warning(f"[serve] warning: --device {device} is not supported by this server; falling back to cpu")
            device = "cpu"
        if kw.get("threads"):
            torch.set_num_threads(kw["threads"])
        t0 = time.time()
        asr = load_model(asr_path, device)
        # Sortformer .afm, or the arch-tagged Nemotron-3-Diarization import; none with --diar-off and no --diar
        diar = load_any(diar_path, device) if diar_path is not None else None
        if diar is None and not kw.get("diar_off"):
            raise ValueError("no --diar given: a diarizer is required unless --turn-input tsvad --diar-off")
        if diar is not None and diar_pool is not None:  # Nemotron-3: 10 ms -> 80 ms pooling (research/SORTFORMER_IMPORT.md: max for streaming)
            for h in diar.heads.values():
                if hasattr(h, "pool"):
                    h.pool = diar_pool
        if diar is not None and diar_spks is not None:  # keep the first k arrival-order columns (the wire protocol carries 4)
            for name, h in diar.heads.items():
                if diar.head_cfg[name]["type"] == "sortformer" and h.num_spks > diar_spks:
                    last = h.out[-1]
                    cut = torch.nn.Linear(last.in_features, diar_spks).to(last.weight.device)
                    cut.weight.data, cut.bias.data = last.weight.data[:diar_spks].clone(), last.bias.data[:diar_spks].clone()
                    h.out[-1], h.num_spks = cut, diar_spks
                    diar.head_cfg[name]["num_spks"] = diar_spks
        name = Path(asr_path).stem + (f"+{Path(diar_path).stem}" if diar_path is not None else "")
        eng = cls(asr, diar, name=name, **kw)
        if fallback:
            eng.count("device_fallback")
        log.info(f"[serve] loaded {eng.name} in {time.time() - t0:.1f}s (turn head {eng.turn_name}, "
              f"input {eng.turn_input}; diarizer {eng.diar_mode} {eng.diar_config} {eng.diar_cfg} left {eng.diar_left}; "
              f"perf {','.join(eng.perf_opts) or 'none'})")
        return eng

    def make_diarizer(self):
        """A fresh per-session diarizer: feed(samples, final) -> (T_new, S) probabilities; ``cfg`` = AOSCConfig."""
        from .streaming_diar import StreamingDiarizer
        kw = {"enc_left_context": self.diar_left} if self.diar_mode == "window" else {}
        return StreamingDiarizer(self.diar, self.diar_head, mode=self.diar_mode, keep_probs=False, **kw, **self.diar_cfg)

    def ready_msg(self) -> dict:
        """The ``ready`` message every connection receives first (docs/PROTOCOL.md)."""
        m = {"type": "ready", "model": self.name, "chunk_ms": self.chunk_ms, "frame_ms": FRAME_MS,
             "diar_config": "off" if self.diar_off else self.diar_config, "column_lag_ms": round(self.lag[2], 1)}
        if self.enroll != "dominant":
            m.update(enroll=self.enroll, enrolled=False, primary_column=None)
        if self.final_sources:
            m["final_asr"] = ",".join(self.final_sources)
        if self.debug:
            m.update(diar_lag_ms=list(self.lag), turn_input=self.turn_input, threads=self.threads)
            if self.enroll != "dominant":
                m.update(enroll_rss_mb=self.enroll_rss_mb, enroll_stride=self.enroll_stride)
        return m

    def warmup(self, seconds: float = 2.0):
        """Run ``seconds`` of noise through a throwaway session so the first real session starts warm."""
        s = Session(self, SessionConfig())
        rng = np.random.default_rng(0)
        s.process((rng.standard_normal(int(seconds * SR)) * 0.01).astype(np.float32))
        s.finish()


class Session:
    """One connection's streaming state. ``process(samples)`` -> protocol messages; ``finish()`` flushes."""

    def __init__(self, engine: Engine, cfg: SessionConfig | None = None):
        e = self.e = engine
        self.cfg = cfg or SessionConfig()
        self.notices: list[dict] = []  # error messages to send with the next block (degradations, repairs)
        self.degraded: dict[str, int] = {}  # per-session counters by error code (stats.degraded)
        if self.cfg.turn_policy in SILERO_POLICIES:  # a Silero policy without Silero: head OR the diarizer timeout
            try:
                e.get_silero()
            except Exception as ex:  # noqa: BLE001
                self.cfg.eot_threshold = self.cfg.theta  # keep the Silero policy's head threshold
                want, self.cfg.turn_policy = self.cfg.turn_policy, "hybrid"
                self._notice("silero_unavailable", f"turn_policy {want} needs Silero VAD ({ex}); using hybrid")
        lid = None
        if e.lid_model is not None:
            from .lid import AmberNetLIDStream
            lid = AmberNetLIDStream(e.lid_model, e.lid_langs, threshold=e.lid_threshold, min_ms=e.lid_min_ms)
        elif e.lid_name:
            from .lid import LIDStream
            lid = LIDStream(e.asr, e.lid_name, threshold=e.lid_threshold, min_ms=e.lid_min_ms, max_ms=e.lid_max_ms)
        self.asr = ASRStream(e.asr, e.vad_name, e.turn_name, e.turn_input, lid=lid, vad_gate=e.asr_vad_gate,
                             vad_hangover_frames=e.asr_vad_hangover)
        if e.diar_off:  # --diar-off: the TS-VAD track stands in for the diarizer's columns (no Sortformer pass)
            from .tsvad_stream import TSVADColumns
            self.diar = TSVADColumns(self.asr, e.num_spks)
        else:
            self.diar = e.make_diarizer()
        self.timeout = (AnySpeakerTimeout(self.cfg.timeout_ms, e.num_spks) if self.cfg.turn_policy == "timeout_any"
                        else TimeoutPolicy(self.cfg.timeout_ms, e.num_spks,
                                           require_quiet=self.cfg.turn_policy == "timeout_quiet"))
        self.head_pol = HeadPolicy(self.cfg.theta)
        self.resampler = Resampler(self.cfg.sample_rate) if self.cfg.sample_rate != SR else None
        self.S = e.num_spks
        self.rows = _Ring()  # finalized diarizer columns (the last 2048 = 164 s; len() counts all)
        self.prims = _Ring()  # primary at each diarizer frame
        self.vad_ring = _Ring()  # served VAD per ASR frame (the diarizer fallback under load shedding)
        self.shed_ring = _Ring()  # per diarizer frame: True when the diarizer did not run on it (its row is a stand-in)
        self._last_real_row = None  # the last row the diarizer produced (--shed-diar hold keeps its column)
        self._shed_tick = 0  # --shed-diar hold: level 1 runs the diarizer on every other block
        self.registry = None  # --diar-labels registry: audioforge.speaker_registry.SpeakerRegistry
        self.col_ids: dict[int, int] = {}  # column -> the registry id last attributed through it (fallback)
        if e.diar_labels == "registry":
            from .speaker_registry import SpeakerRegistry
            self.registry = SpeakerRegistry(e.registry_thr)
            self.asr.keep_spk = e.diar_embed == "spk"
        self._rbuf = np.zeros(0, np.float32)  # --diar-embed titanet: the last REGISTRY_AUDIO_S of audio
        self._rbuf0 = 0
        self.last_eot: float | None = None
        self.eot_hist: deque = deque(maxlen=256)  # (decision-ready time, head p) of recent head frames (hybrid: p)
        self.hyb_t = -math.inf  # decision time of the last hybrid turn_end
        self.seg_tok = 0  # token index where the current (not yet finalized) segment starts
        self.seg_frame0 = 0  # ASR frame where the current segment starts (MAX_SEGMENT_S cap)
        self.last_partial = ""
        self.samples = 0
        self.proc_s = 0.0
        self._rate: deque = deque()  # (samples, seconds) of the last SHED_WINDOW_S of audio -> recent_rtf()
        self.chunk_ms: deque = deque(maxlen=STAT_KEEP)
        self._acc = 0.0  # processing time accumulated for the current 160 ms of audio
        self._next_chunk = CHUNK_SAMPLES
        self.asr_ms: deque = deque(maxlen=STAT_KEEP)
        self.diar_ms: deque = deque(maxlen=STAT_KEEP)
        self.first_partial_ms: float | None = None
        self.backlog_ms: deque = deque(maxlen=STAT_KEEP)
        self.backlog_ms_max = 0.0
        self.send_lag_ms: deque = deque(maxlen=STAT_KEEP)
        self.send_lag_ms_max = 0.0
        self.diar_emit_lag: deque = deque(maxlen=STAT_KEEP)  # measured: audio time at emission - frame end (ms)
        self.finished = False
        self.shed = 0  # current load-shedding level (0 none, 1 diarizer skipped, 2 + partials / lookahead dropped)
        self.shed_max = 0
        self._shed_notice_t = -math.inf
        self.la_dropped = False  # --asr-lookahead: the second pass was dropped under load (finals fall back)
        # --enroll after_agent | explicit: the voice binder and the recent audio it reads per diarizer frame;
        # after_agent_arm: the agent-end choice + causal_dominant (no audio, no embedding)
        self.binder = (VoiceBinder(e.embedder, e.num_spks, stride=e.enroll_stride)
                       if e.enroll in VOICE_MODES and e.tsvad is None
                       else ArmBinder(e.num_spks) if e.enroll == "after_agent_arm" and not e.diar_off
                       else None)  # --diar-off: column 0 is the user's TS-VAD track, there is no column to bind
        self.tsvad = None  # --turn-input tsvad: the user's voice print and TS-VAD state (audioforge.tsvad_stream)
        if e.tsvad is not None:
            from .tsvad_stream import TSVADTrack
            self.tsvad = self.asr.tsvad = TSVADTrack(e.tsvad, e.asr.heads["spk"], e.tsvad_print_s, e.tsvad_refresh_s)
            if e.enroll == "dominant":  # no enrolment signal: the session's first voice is the user
                self.tsvad.arm(0)
            self._tsvad_theta()
        # hybrid_silero / hybrid_dyn: the Silero any-speaker silence and the head posterior per frame
        self.sil = SileroSilence(e.get_silero()) if self.cfg.turn_policy in SILERO_POLICIES else None
        self.sil_fired = -2  # last speech chunk of the silence run that already fired (one firing per run)
        self.enroll_ms: deque = deque(maxlen=STAT_KEEP)
        self.dyn_pending: deque = deque()  # silence frames waiting for the head's posterior on the same frame
        self.head_p: dict[int, tuple[float, float]] = {}  # frame v -> (decision-ready time, head p)
        self._abuf = np.zeros(0, np.float32)  # samples from self._abuf0 on (kept: the last 40 frames)
        self._abuf0 = 0
        # --final-asr: the turn audio and served VAD since the current segment, and the offline jobs to submit
        self.fa = e.final_asr is not None
        self.la = LookaheadStream(e.asr, e.asr_lookahead) if e.asr_lookahead else None
        self.la_pending: deque = deque()  # lookahead finals waiting for their frames: dicts (see _lookahead_due)
        self.la_seg = 0  # lookahead frame where the current lookahead segment starts
        self.final_jobs: list[dict] = []
        self.final_lat_ms: dict[str, deque] = {src: deque(maxlen=STAT_KEEP) for src in e.final_sources}
        self.final_compute_ms: deque = deque(maxlen=STAT_KEEP)
        self._fbuf, self._fbuf0 = np.zeros(0, np.float32), 0  # audio from absolute sample _fbuf0 on
        self._fvad: list[float] = []  # served VAD of ASR frames _fvad0, _fvad0 + 1, ...
        self._fvad0 = 0
        self._fseg0 = 0  # sample where the current segment starts (the previous cut's decision time)
        self._fprev_end = 0  # end sample of the previous transcribed span

    # ---------------------------------------------------------------- helpers
    def _notice(self, code: str, detail: str = "", fatal: bool = False, n: int = 1) -> None:
        """Count a degradation (session + engine) and queue one ``error`` message per code per session."""
        first = code not in self.degraded
        self.degraded[code] = self.degraded.get(code, 0) + n
        self.e.count(code, n)
        if first or fatal:
            self.notices.append(error_msg(code, detail, fatal))

    def take_notices(self) -> list[dict]:
        """The non-fatal ``error`` messages (degradations) queued since the last call."""
        out, self.notices = self.notices, []
        return out

    def apply_config(self, cfg: SessionConfig):
        """Apply a session ``config`` message received before the first audio."""
        if (cfg.turn_policy == "timeout_any") != isinstance(self.timeout, AnySpeakerTimeout):
            # the policy object is built with the session: switching to / from timeout_any needs the first config
            self._notice("bad_config", f"turn_policy {cfg.turn_policy} ignored after the first audio (send it in the "
                                       f"first config); keeping {self.cfg.turn_policy}")
            cfg.turn_policy = self.cfg.turn_policy
        if cfg.turn_policy in SILERO_POLICIES and self.sil is None:
            # the Silero chunk grid starts with the stream: these policies must be set before the first audio
            self._notice("bad_config", f"turn_policy {cfg.turn_policy} ignored after the first audio (send it in the "
                                       f"first config); keeping {self.cfg.turn_policy}")
            cfg.turn_policy = self.cfg.turn_policy
        self.cfg = cfg
        self._tsvad_theta()
        self.timeout.timeout_ms = cfg.timeout_ms
        if isinstance(self.timeout, TimeoutPolicy):
            self.timeout.require_quiet = cfg.turn_policy == "timeout_quiet"
        self.head_pol.thr = cfg.theta

    def _tsvad_theta(self):
        """--turn-input tsvad: the head threshold of hybrid_dyn / hybrid defaults to TSVAD_DYN's (a config
        eot_threshold wins)."""
        if self.tsvad is not None and self.cfg.turn_policy in ("hybrid_dyn", "hybrid") and self.cfg.eot_threshold is None:
            self.cfg.eot_threshold = TSVAD_DYN[0]
            if hasattr(self, "head_pol"):
                self.head_pol.thr = TSVAD_DYN[0]

    def _act_tsvad(self, v: int):
        """turn_input tsvad: (P(target), columns [P(target), P(other), 0, 0], primary 0) of ASR frame v, as the
        benchmark fed the turn head (scripts/research/tsvad.py binding_inputs)."""
        tp = self.asr.tsvad_p
        p = tp[v] if v < len(tp) else (tp[-1] if tp else np.zeros(2, np.float32))
        return float(p[0]), np.array([p[0], p[1], 0.0, 0.0], np.float32), 0

    @property
    def t(self) -> float:
        """Audio time processed so far, in seconds."""
        return round(self.samples / SR, 3)

    def _latest_row(self):
        return self.rows[-1] if self.rows else np.zeros(self.S)

    def _seg_text(self) -> str:
        tok = self.e.asr.tokenizer
        return tok.decode(self.asr.tokens[self.seg_tok:]) if tok is not None else ""

    def _cut(self, pol: str) -> bool:
        tp = self.cfg.turn_policy
        return pol == (tp if tp in ("head",) + HYBRID_POLICIES else "timeout")

    def _fires(self, pol: str) -> bool:
        """Whether candidate path ``pol`` (timeout | head) runs under the session's policy."""
        tp = "timeout" if self.cfg.turn_policy in ("timeout_quiet", "timeout_any") else self.cfg.turn_policy
        if tp in SILERO_POLICIES:  # head OR Silero: the Sortformer timeout path is replaced by the Silero one
            return pol == "head"
        return tp in (pol, "both", "hybrid")

    def _p_at(self, t: float) -> float | None:
        """The head's probability of the latest head frame whose decision data was complete by time t."""
        p = None
        for tr, q in self.eot_hist:
            if tr > t:
                break
            p = q
        return p

    def _hybrid(self, events: list) -> list:
        """turn_policy "hybrid" / "hybrid_silero" / "hybrid_dyn": the candidates (sorted by decision time; head plus
        timeout, silero or dyn) -> one event per turn tagged with the policy, the earlier of the paths. A candidate is
        kept only if its path heard speech after the previous such turn_end's decision time (last speech frame =
        firing frame - its silence), so the later path's firing for the same turn is dropped. p = the head's
        probability (head / dyn path: the one it used; timeout / silero path: the latest head frame ready by the
        decision time; None without a turn head); silence_ms = the firing path's own silence (timeout: diarizer
        primary; head: VAD head; silero / dyn: Silero any-speaker)."""
        out = []
        for t_dec, pol, ev, v in events:
            t_speech = (v - int(ev["silence_ms"]) // FRAME_MS + 1) * FRAME_MS / 1000
            if t_speech <= self.hyb_t:
                continue
            self.hyb_t = t_dec
            ev = dict(ev, p=ev["p"] if pol in ("head", "dyn") else self._p_at(t_dec))
            out.append((t_dec, self.cfg.turn_policy, ev, v))
        return out

    def _silero_events(self, sil_frames: list, final: bool) -> list:
        """hybrid_silero / hybrid_dyn Silero candidates from this block's silence frames (one firing per silence run,
        i.e. re-armed by a new speech chunk). hybrid_silero: silence >= silero_timeout_ms, decided when the frame's
        Silero chunk is complete. hybrid_dyn: silence - clamp(75 - 55 p, 2, 75) > 4.957 frames with p = the head's
        posterior on the same frame, decided when both are ready (frames wait in dyn_pending for the head)."""
        ev = []
        if self.cfg.turn_policy == "hybrid_silero":
            thr = self.e.silero_timeout_ms / FRAME_MS - 1e-6
            for v, sil, t_rdy, last_sp in sil_frames:
                if last_sp >= 0 and last_sp != self.sil_fired and sil >= thr:
                    self.sil_fired = last_sp
                    ev.append((t_rdy, "silero", {"silence_ms": int(round(sil * FRAME_MS)), "p": None}, v))
            return ev
        self.dyn_pending.extend(sil_frames)
        no_head = self.asr.turn is None
        while self.dyn_pending:
            v, sil, t_rdy, last_sp = self.dyn_pending[0]
            if no_head:
                t_h, p = t_rdy, 0.0
            elif v in self.head_p:
                t_h, p = self.head_p.pop(v)
            elif final:  # the head never produced this frame (flush past its last chunk)
                self.dyn_pending.popleft()
                continue
            else:
                break
            self.dyn_pending.popleft()
            t0 = DYN_T0 if self.e.dyn_t0 is None else self.e.dyn_t0
            req = min(max(t0 - (DYN_A if self.e.dyn_a is None else self.e.dyn_a) * p, DYN_TMIN), t0)
            if last_sp >= 0 and last_sp != self.sil_fired and sil - req > self.e.dyn_offset:
                self.sil_fired = last_sp
                ev.append((max(t_rdy, t_h), "dyn", {"silence_ms": int(round(sil * FRAME_MS)), "p": float(p)}, v))
        for v in [k for k in self.head_p if self.sil is not None and k < self.sil.v - 64]:
            del self.head_p[v]  # frames the Silero grid is long past (no pending entry will ask for them)
        return ev

    def arm_enrollment(self, kind: str, frame: int | None = None, embedding=None) -> bool:
        """Client message 'agent_end' / 'enroll' at audio frame ``frame`` (default: the audio received so far):
        arms the binder if it is this session's trigger. Returns whether it was applied. With --turn-input tsvad an
        'enroll' message that carries a stored voice print (``embedding``) is applied under every --enroll mode, so
        --mode single takes the print from the client or, without one, live after agent_end."""
        if self.tsvad is not None and kind == "enroll" and embedding is not None:
            self.tsvad.set_print(embedding)
            return True
        if ENROLL_TRIGGER.get(self.e.enroll) != kind or (self.binder is None and self.tsvad is None):
            return False
        f = self.samples // FRAME_SAMPLES if frame is None else int(frame)
        if self.tsvad is not None:  # the TS-VAD print: a stored embedding, else the next print_s s of speech
            if embedding is not None:
                self.tsvad.set_print(embedding)
            else:
                self.tsvad.arm(f)
        if self.binder is not None:
            self.binder.arm(f)
        return True

    def _frame_audio(self, v: int) -> np.ndarray:
        a, b = v * FRAME_SAMPLES - self._abuf0, (v + 1) * FRAME_SAMPLES - self._abuf0
        return self._abuf[max(0, a): max(0, b)]

    def _act(self, v: int):
        if v < len(self.rows):
            row, p = self.rows[v], self.prims[v]
        else:  # flush: frames past the diarizer's last column reuse it
            row, p = self._latest_row(), (self.prims[-1] if self.prims else None)
        return (float(row[p]) if p is not None else 0.0), row, p

    # decision times: the audio time at which the data behind a frame's decision is complete (independent of how
    # the client batched its audio, so events and finals are deterministic)
    def _asr_ready_t(self, v: int) -> float:
        a = self.asr
        need = ((v // a.cs + 1) * a.chunk_mel - 1) * a.hop + a.half  # samples for the chunk holding frame v
        return min(need, self.samples) / SR

    def _diar_ready_t(self, v: int) -> float:
        C, R = self.diar.cfg.chunk_len, self.diar.cfg.chunk_right_context
        return min(((v // C + 1) * C + R) * FRAME_SAMPLES, self.samples) / SR

    def _asr_frames_at(self, t: float) -> int:
        a = self.asr
        ready = (int(round(t * SR)) - a.half) // a.hop + 1
        n = max(ready, 0) // a.chunk_mel * a.cs
        return a.n_frames if t >= self.samples / SR and self.finished else min(n, a.n_frames)

    def _turn_end(self, pol: str, ev: dict, frame_v: int, t_dec: float, out: list):
        msg = {"type": "turn_end", "t": round(t_dec, 3), "policy": "change" if ev.get("change") else pol,
               "p": ev.get("p"), "silence_ms": int(ev["silence_ms"])}
        if self.e.debug:
            msg["frame_t"] = round((frame_v + 1) * FRAME_MS / 1000, 3)
        out.append(msg)
        if self._cut(pol):  # the final holds the tokens of the ASR frames available at the decision time
            self._emit_final(t_dec, self._asr_frames_at(t_dec), ev.get("primary", self.timeout.primary), out)

    def _attribute(self, f0: int, f: int, col_speaker):
        """``final.speaker`` (+ the optional ``speaker_conf`` / ``diar_shed`` keys) for the segment of frames [f0, f)
        (research/DIARIZATION_FIX.md). Legacy (``--diar-labels column --shed-diar vad``): the caller's primary column,
        no extra keys. ``hold``: the same column plus ``diar_shed``. ``registry``: the turn's dominant column picks
        the speaker's own frames, their voice embedding is matched against the session's SpeakerRegistry (stable id
        across column permutations and re-entries); with too few frames the column's last id, else ``null``."""
        e = self.e
        if e.diar_off:  # single-model mode: 0 = the enrolled user, 1 = someone else (the TS-VAD column that dominates)
            from .speaker_registry import turn_column
            lo, hi = max(f0, len(self.rows) - len(self.rows.d)), min(f, len(self.rows))
            rows = [self.rows[v][:2] for v in range(lo, hi)]
            col = turn_column(rows) if rows else None
            return (col_speaker if col is None else int(col)), {}
        if e.diar_labels != "registry" and e.shed_diar != "hold":
            return col_speaker, {}
        lo, hi = max(f0, len(self.rows) - len(self.rows.d)), min(f, len(self.rows))
        vs = range(lo, hi)
        held = [bool(self.shed_ring[v]) for v in vs] if hi > lo else []
        shed = any(held)
        if self.registry is None:
            return col_speaker, {"speaker_conf": None, "diar_shed": shed}
        from .speaker_registry import MIN_FRAMES, turn_column
        real = [v for v, h in zip(vs, held) if not h]
        col = turn_column([self.rows[v] for v in real]) if real else None
        idx = [v for v in real if self.rows[v][col] > ACT_THRESHOLD] if col is not None else []
        if len(idx) < MIN_FRAMES:  # too few own frames (overlap, or the diarizer was shed): the served VAD's frames
            vlo, vhi = max(f0, len(self.vad_ring) - len(self.vad_ring.d)), min(f, len(self.vad_ring))
            idx = [v for v in range(vlo, vhi) if self.vad_ring[v] > ACT_THRESHOLD]
        emb = None
        if len(idx) >= MIN_FRAMES:
            try:
                emb = self._embed(idx)
            except Exception as ex:  # noqa: BLE001 - a label must never take the session down
                self._notice("registry_failed", f"{type(ex).__name__}: {ex}")
        if emb is None:
            sid = self.col_ids.get(col) if col is not None else None
            return sid, {"speaker_conf": None, "diar_shed": shed}
        sid, conf, _ = self.registry.assign(emb)
        if col is not None and sid is not None:
            self.col_ids[col] = sid
        return sid, {"speaker_conf": conf, "diar_shed": shed}

    def _embed(self, idx: list[int]):
        """Unit-norm voice embedding of the frames ``idx``: the speaker head over its kept block-4 frames
        (``--diar-embed spk``, no extra model pass) or TitaNet-L over their audio (``titanet``)."""
        e = self.e
        if e.diar_embed == "spk":
            r = self.asr.spk_feats
            idx = [v for v in idx if v >= len(r) - len(r.d) and v < len(r)]
            if not idx:
                return None
            return e.registry_embedder(np.stack([r[v] for v in idx])[None], np.ones((1, len(idx)), bool))[0]
        v0 = self._rbuf0 // FRAME_SAMPLES
        local = [v - v0 for v in idx if v >= v0 and (v + 1) * FRAME_SAMPLES <= self._rbuf0 + len(self._rbuf)]
        if not local:
            return None
        return e.registry_embedder.frames(self._rbuf, [np.asarray(local)])[0]

    def _emit_final(self, t_dec: float, f: int, speaker, out: list):
        """Cut the current segment at ASR frame ``f`` (decision time ``t_dec``): the streaming final, then the
        lookahead / offline finals it is due (research/HYBRID_ASR.md)."""
        cut = max(self.seg_tok, self.asr.tok_at[f - 1] if f > 0 else 0)
        tok = self.e.asr.tokenizer
        text = tok.decode(self.asr.tokens[self.seg_tok:cut]) if tok is not None else ""
        speaker, extra = self._attribute(self.seg_frame0, f, speaker)
        fin = {"type": "final", "t": round(t_dec, 3), "text": text, "speaker": speaker, **extra}
        out.append(fin)
        if self.e.final_sources:
            fin["source"] = "stream"
        if self.la is not None:
            self._lookahead_due(f, t_dec, speaker, text)
        if self.fa:
            self._final_job(round(t_dec * SR), f, t_dec, speaker, out, text)
        self.seg_tok, self.seg_frame0 = cut, f
        self.last_partial = ""  # tokens after the cut (if any) start the next segment's partial

    def _final_job(self, end_lim: int, n_frames: int, t_dec: float, speaker, out: list, stream_text: str = ""):
        """--final-asr: queue the finished segment's turn audio (VAD onset - pre-roll .. last speech + post-roll,
        within [previous span end, decision time]) for the offline model; a segment without VAD speech gets an empty
        final at once. Called right after the segment's streaming final."""
        pre, post = int(FINAL_PRE_ROLL_S * SR), int(FINAL_POST_ROLL_S * SR)
        f0 = max(self._fseg0 // FRAME_SAMPLES, self._fvad0)
        if self.asr.vad_name is None:  # no VAD head: the whole segment
            start, end = max(self._fseg0, self._fbuf0), end_lim
        else:
            sp = [v for v in range(f0, min(n_frames, self._fvad0 + len(self._fvad)))
                  if self._fvad[v - self._fvad0] > ACT_THRESHOLD]
            start = max(sp[0] * FRAME_SAMPLES - pre, self._fprev_end, self._fbuf0) if sp else None
            end = min(end_lim, (sp[-1] + 1) * FRAME_SAMPLES + post) if sp else None
        src = self.e.final_asr.source
        job = {"type": "final", "t": round(t_dec, 3), "text": "", "speaker": speaker, "source": src,
               "start": None, "end": None, "latency_ms": 0.0}
        if start is not None and end is not None and end > start:
            job.update(start=round(start / SR, 3), end=round(end / SR, 3))
            self.final_jobs.append({"msg": job, "audio": self._fbuf[start - self._fbuf0: end - self._fbuf0].copy(),
                                    "stream_text": stream_text})  # the fallback text if the offline pass fails
            self._fprev_end = end
        else:
            out.append(job)
        self._fseg0 = end_lim
        keep = max(self._fbuf0, end_lim - FRAME_SAMPLES - pre)  # the next span starts at >= this sample
        self._fbuf, self._fbuf0 = self._fbuf[keep - self._fbuf0:], keep
        kv = max(self._fvad0, end_lim // FRAME_SAMPLES)
        self._fvad, self._fvad0 = self._fvad[kv - self._fvad0:], kv

    def _lookahead_due(self, n_frames: int, t_dec: float, speaker, stream_text: str = ""):
        """--asr-lookahead: queue the segment's lookahead final; due once the lookahead pass has decoded
        LOOKAHEAD_MARGIN_FRAMES past the segment's last VAD speech frame (capped at the streaming cut n_frames)."""
        f0 = max(self._fseg0 // FRAME_SAMPLES, self._fvad0)
        sp = [v for v in range(f0, min(n_frames, self._fvad0 + len(self._fvad)))
              if self._fvad[v - self._fvad0] > ACT_THRESHOLD] if self.asr.vad_name is not None else [n_frames - 1]
        need = min(n_frames, sp[-1] + 1 + LOOKAHEAD_MARGIN_FRAMES) if sp else min(n_frames, self.la.n_frames)
        self.la_pending.append({"t": round(t_dec, 3), "t_dec": t_dec, "speaker": speaker, "need": max(need, 0),
                                "stream_text": stream_text, "cut_frames": n_frames})
        if not self.fa:  # (with --final-asr, _final_job keeps the segment bookkeeping)
            self._fseg0 = round(t_dec * SR)
            kv = max(self._fvad0, self._fseg0 // FRAME_SAMPLES)
            self._fvad, self._fvad0 = self._fvad[kv - self._fvad0:], kv

    def _drop_lookahead(self, reason: str):
        """Fall back to the single streaming pass for the rest of the session (its pending finals carry the
        streaming text), reported once as a non-fatal ``lookahead_dropped`` error."""
        if not self.la_dropped:
            self.la_dropped = True
            self._notice("lookahead_dropped", f"lookahead pass dropped ({reason}); finals with source lookahead "
                                              f"now carry the streaming text")

    def _lookahead_emit(self, out: list):
        """Send the lookahead finals whose frames are decoded (FIFO), each cutting the lookahead tokens there."""
        la = self.la
        while self.la_pending and (la.n_frames >= self.la_pending[0]["need"] or self.finished or self.la_dropped):
            p = self.la_pending.popleft()
            if self.la_dropped:
                cut = max(self.la_seg, p["cut_frames"])
                text, lat = p["stream_text"], 0.0
            else:
                cut = min(max(p["need"], self.la_seg), la.n_frames)
                a = la.tok_at[self.la_seg - 1] if self.la_seg > 0 else 0
                b = la.tok_at[cut - 1] if cut > 0 else 0
                tok = self.e.asr.tokenizer
                text = tok.decode(la.tokens[a:b]) if tok is not None else ""
                ready = la.ready_t(cut - 1) if cut > 0 else 0.0
                lat = max(0.0, min(ready, self.samples / SR) - p["t_dec"]) * 1000
            out.append({"type": "final", "t": p["t"], "text": text, "speaker": p["speaker"], "source": "lookahead",
                        "start": round(self.la_seg * FRAME_MS / 1000, 3), "end": round(cut * FRAME_MS / 1000, 3),
                        "latency_ms": round(lat, 1)})
            self.final_lat_ms["lookahead"].append(lat)
            self.la_seg = cut

    def take_final_jobs(self) -> list[dict]:
        """--final-asr: the queued offline jobs ({"msg": the final to complete, "audio"}), oldest first."""
        jobs, self.final_jobs = self.final_jobs, []
        return jobs

    def run_final_jobs_sync(self) -> list[dict]:
        """Run the queued jobs on the engine's final-ASR worker and wait (offline drivers and tests; the WebSocket
        handler submits them asynchronously instead)."""
        out = []
        for j in self.take_final_jobs():
            t0 = time.perf_counter()
            r = self.e.final_asr.submit(j["audio"]).result()
            out.append(self.complete_final(j, r, (time.perf_counter() - t0) * 1000))
        return out

    def complete_final(self, job: dict, result, latency_ms: float) -> dict:
        """The offline final of ``job`` with the worker's ``result`` text and its latency."""
        m = dict(job["msg"], text=result.text, latency_ms=round(latency_ms, 1))
        self.final_lat_ms[job["msg"]["source"]].append(latency_ms)
        if getattr(result, "compute_ms", None) is not None:
            self.final_compute_ms.append(result.compute_ms)
        return m

    def fallback_final(self, job: dict, err, latency_ms: float) -> list[dict]:
        """--final-asr: the offline pass failed (worker crash / timeout): report it once and complete the job with
        the streaming final's text under the offline source, so a client waiting for that source is not left
        without the turn."""
        self._notice("final_asr_failed", f"{job['msg']['source']} failed ({type(err).__name__}: {err}); "
                                         f"falling back to the streaming final")
        m = dict(job["msg"], text=job.get("stream_text", ""), latency_ms=round(latency_ms, 1))
        return self.take_notices() + [m]

    # ---------------------------------------------------------------- processing
    def _sanitize(self, samples) -> np.ndarray:
        """Non-finite samples -> 0, |x| > MAX_ABS_SAMPLE clipped (both counted and reported once): a NaN in the
        input would otherwise poison the encoder caches for the rest of the session."""
        x = np.asarray(samples, np.float32)
        if x.ndim > 1:  # a (n, C) or (C, n) array from an offline driver: average to mono
            x = (x.mean(1) if x.shape[0] >= x.shape[-1] else x.mean(0)).astype(np.float32)
        if len(x):
            fin = np.isfinite(x)
            if not fin.all():
                bad = int((~fin).sum())
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                self._notice("nan_input", f"{bad} non-finite samples replaced by 0", n=bad)
            big = np.abs(x) > MAX_ABS_SAMPLE
            if big.any():
                x = np.clip(x, -MAX_ABS_SAMPLE, MAX_ABS_SAMPLE)
                self._notice("clipped_input", f"{int(big.sum())} samples beyond +-{MAX_ABS_SAMPLE:g} clipped",
                             n=int(big.sum()))
        return x

    def recent_rtf(self) -> float:
        """Processing seconds per audio second over the last SHED_WINDOW_S of audio (0 before any audio)."""
        n = sum(s for s, _ in self._rate)
        return sum(d for _, d in self._rate) / (n / SR) if n else 0.0

    def _shed_level(self, shed: int, backlog_ms: float | None) -> int:
        """The load-shedding level for this block: the caller's (global watchdog) or the session's own backlog, but
        a level is only *entered* while recent_rtf() >= SHED_RTF (a client that dumps a file at a faster-than-real-
        time server builds a backlog that drains by itself: nothing is shed, so unpaced runs equal paced ones); a
        level already entered is kept until the backlog falls back under its threshold."""
        want = int(shed or 0)
        if backlog_ms is not None:
            want = max(want, 2 if backlog_ms >= SHED_PARTIAL_MS else 1 if backlog_ms >= SHED_DIAR_MS else 0)
        rtf = self.recent_rtf()
        lvl = want if want <= self.shed or rtf >= SHED_RTF else self.shed
        if lvl > self.shed and time.perf_counter() - self._shed_notice_t >= SHED_NOTICE_S:
            self._shed_notice_t = time.perf_counter()
            what = "diarizer skipped (VAD-only speaker activity)" if lvl == 1 else \
                "diarizer skipped, partials dropped, frames batched" + (", lookahead dropped" if self.la else "")
            self.notices.append(error_msg("overloaded", f"backlog {backlog_ms or 0:.0f} ms at rtf {rtf:.2f}: level "
                                                        f"{lvl}, {what}"))
            self.e.count("overloaded")
        self.shed, self.shed_max = lvl, max(self.shed_max, lvl)
        return lvl

    @torch.no_grad()
    def process(self, samples: np.ndarray, final: bool = False, arrived: float | None = None,
                backlog_ms: float | None = None, shed: int = 0) -> list[dict]:
        """Run a block of 16 kHz float samples (any length) through both models -> messages, in order:
        error*, frame* | frames, (turn_end, final?)*, partial?. ``arrived``: perf_counter time the block's last
        sample arrived. ``backlog_ms`` / ``shed``: the connection's unprocessed audio and the watchdog's shedding
        level (research/BULLETPROOF.md section 4): level >= 1 skips the diarizer for this block (its columns are
        replaced by the served VAD in column 0), level >= 2 also drops the partial, batches the frames into one
        ``frames`` message and drops the lookahead pass for good."""
        t0 = time.perf_counter()
        x = self._sanitize(samples)
        lvl = self._shed_level(shed, backlog_ms)
        self.samples += len(x)
        if backlog_ms is not None:
            self.backlog_ms.append(backlog_ms)
            self.backlog_ms_max = max(self.backlog_ms_max, backlog_ms)
        if self.registry is not None and self.e.diar_embed == "titanet":  # per-turn TitaNet: keep the recent audio
            self._rbuf = np.concatenate([self._rbuf, x])
            drop = max(0, len(self._rbuf) - int(REGISTRY_AUDIO_S * SR)) // FRAME_SAMPLES * FRAME_SAMPLES
            self._rbuf, self._rbuf0 = self._rbuf[drop:], self._rbuf0 + drop
        if self.binder is not None:  # the binder reads each diarizer frame's audio: keep the last 40 frames
            self._abuf = np.concatenate([self._abuf, x])
            drop = max(0, len(self._abuf) - 40 * FRAME_SAMPLES)
            self._abuf, self._abuf0 = self._abuf[drop:], self._abuf0 + drop
        frames = self.asr.feed_frames(x, final)
        bad = [f for f in frames if not math.isfinite(f["vad"]) or (f["eot"] is not None and not math.isfinite(f["eot"]))]
        if bad:  # a non-finite head output: report, zero it, and restart the recurrent state
            for f in bad:
                f["vad"] = f["vad"] if math.isfinite(f["vad"]) else 0.0
                f["eot"] = None if f["eot"] is None else (f["eot"] if math.isfinite(f["eot"]) else 0.0)
            self.asr.reset_state()
            self._notice("nan_state_reset", f"{len(bad)} ASR frames with non-finite head outputs at t={self.t}; "
                                            f"encoder / decoder state reset", n=len(bad))
        for f in frames:
            self.vad_ring.append(f["vad"])
        if self.fa:  # the turn audio for the offline final pass
            self._fbuf = np.concatenate([self._fbuf, x])
            cap = int(FINAL_MAX_TURN_S * SR)
            if len(self._fbuf) > cap:
                self._fbuf0 += len(self._fbuf) - cap
                self._fbuf = self._fbuf[len(self._fbuf) - cap:]
        if self.fa or self.la is not None:  # the served VAD per ASR frame (turn onset / last speech)
            self._fvad += [fr["vad"] for fr in frames]
        if self.la is not None and not self.la_dropped:  # the text-only lookahead pass over the same audio
            if lvl >= 2:
                self._drop_lookahead(f"backlog {backlog_ms or 0:.0f} ms")
            else:
                tl = time.perf_counter()
                try:
                    self.la.feed(x, final)
                except Exception as ex:  # noqa: BLE001 - the second pass must never take the session down
                    self._drop_lookahead(f"{type(ex).__name__}: {ex}")
                self.la.ms += (time.perf_counter() - tl) * 1000
        t1 = time.perf_counter()
        n_before = int(getattr(self.diar, "done", len(self.rows)))
        diar_skipped = lvl >= 1
        if diar_skipped and lvl == 1 and self.e.shed_diar == "hold":  # half cadence instead of no diarizer
            self._shed_tick += 1
            diar_skipped = self._shed_tick % 2 == 1
        if diar_skipped:
            try:
                drows = self.diar.feed(x, final, skip=True)
            except TypeError:  # a diarizer without the skip path (test doubles): run it, keep the message shedding
                drows, diar_skipped = self.diar.feed(x, final), False
        else:
            drows = self.diar.feed(x, final)
        drows = drows.cpu().numpy()
        if diar_skipped and len(drows):  # shed: a stand-in row per frame the diarizer did not run on
            if self.e.shed_diar == "hold":  # the last stable column carries the served VAD (speaker_registry.held_row)
                from .speaker_registry import held_row
                for i in range(len(drows)):
                    v = n_before + i
                    drows[i] = held_row(self._last_real_row, self.vad_ring[v] if v < len(self.vad_ring) else 0.0,
                                        self.timeout.primary, drows.shape[1])
            else:  # legacy: the served VAD in column 0
                for i in range(len(drows)):
                    v = n_before + i
                    drows[i, 0] = self.vad_ring[v] if v < len(self.vad_ring) else 0.0
            self._count_only("shed_diar_frames", len(drows))
        elif len(drows):  # remember the last real row with an active column (the "last stable column" under hold)
            act = drows[np.asarray(drows).max(1) > ACT_THRESHOLD] if drows.ndim == 2 else drows[:0]
            if len(act):
                self._last_real_row = np.asarray(act[-1], np.float64).copy()
        if len(drows) and not np.isfinite(drows).all():
            n_bad = int((~np.isfinite(drows)).any(1).sum())
            drows = np.nan_to_num(drows, nan=0.0, posinf=0.0, neginf=0.0)
            self._notice("nan_state_reset", f"{n_bad} diarizer frames with non-finite probabilities zeroed", n=n_bad)
        t2 = time.perf_counter()
        self.asr_ms.append((t1 - t0) * 1000)
        self.diar_ms.append((t2 - t1) * 1000)
        events: list[tuple[float, str, dict, int]] = []
        enrolled_msgs = []
        for row in drows:
            v = len(self.rows)
            bound = None
            if self.binder is not None:
                tb = time.perf_counter()
                was = self.binder.enrolled
                bound = self.binder.update(v, row, self._frame_audio(v))
                self.enroll_ms.append((time.perf_counter() - tb) * 1000)
                if self.binder.enrolled and not was:
                    enrolled_msgs.append({"type": "enrolled", "t": round(self._diar_ready_t(v), 3),
                                          "column": int(bound)})
            if self.e.diar_off:  # the TS-VAD columns: the primary is the enrolled user (column 0)
                bound = 0
            ev = self.timeout.update(row, bound)
            self.rows.append(np.asarray(row, np.float64))
            self.shed_ring.append(bool(diar_skipped))
            self.prims.append(self.timeout.primary)
            if not final:
                self.diar_emit_lag.append(self.samples / SR * 1000 - (v + 1) * FRAME_MS)
            if ev is not None and self._fires("timeout"):
                events.append((self._diar_ready_t(v), "timeout", ev, v))
        if self.e.turn_input in ("diar", "tsvad") and self.asr.turn is not None:
            tsv = self.e.turn_input == "tsvad"  # the TS-VAD track is ready with its ASR chunk: no diarizer wait
            for v, p, vad in self.asr.run_turn_on_diar(self.asr.n_frames if tsv else len(self.rows),
                                                       self._act_tsvad if tsv else self._act, flush=final):
                if not math.isfinite(p):
                    p = 0.0
                    self._notice("nan_state_reset", "non-finite turn-head output on the diarizer clock zeroed")
                self.last_eot = p
                t_rdy = self._asr_ready_t(v) if tsv else max(self._diar_ready_t(v), self._asr_ready_t(v))
                self.eot_hist.append((t_rdy, p))
                if self.sil is not None:
                    self.head_p[v] = (t_rdy, p)
                ev = self.head_pol.update(p, vad)
                if ev is not None and self._fires("head"):
                    events.append((t_rdy, "head", ev, v))
        out = self.take_notices()
        row, spk_v = self._latest_row(), len(self.rows) - 1
        fr_msgs = []
        for f in frames:
            if f["eot"] is not None:
                self.last_eot = f["eot"]
                self.eot_hist.append((self._asr_ready_t(f["v"]), f["eot"]))
                if self.sil is not None:
                    self.head_p[f["v"]] = (self._asr_ready_t(f["v"]), f["eot"])
                ev = self.head_pol.update(f["eot"], f["vad"])
                if ev is not None and self._fires("head"):
                    events.append((self._asr_ready_t(f["v"]), "head", ev, f["v"]))
            m = {"type": "frame", "t": round((f["v"] + 1) * FRAME_MS / 1000, 3), "vad": round(f["vad"], 4),
                 "eot": None if self.last_eot is None else round(self.last_eot, 5),
                 "speakers": [round(min(max(float(p), 0.0), 1.0), 4) for p in row], "primary": self.timeout.primary}
            if self.e.debug:
                m["spk_t"] = round((spk_v + 1) * FRAME_MS / 1000, 3) if spk_v >= 0 else None
            fr_msgs.append(m)
        if lvl >= 2 and len(fr_msgs) > 1:  # one message instead of one per frame
            out.append({"type": "frames", "items": [{k: v for k, v in m.items() if k != "type"} for m in fr_msgs]})
            self._count_only("frames_batched", len(fr_msgs))
        else:
            out += fr_msgs
        out += enrolled_msgs
        if self.tsvad is not None and self.tsvad.events:
            # the print itself (192 numbers) goes with it, so a client can store a print taken live (e.g. after an
            # onboarding prompt) and send it back with {"type": "enroll", "embedding": [...]} next session
            emb = [round(float(x), 6) for x in self.tsvad.print] if self.tsvad.print is not None else []
            out += [{"type": "voiceprint", **ev, "embedding": emb} for ev in self.tsvad.events]
            self.tsvad.events = []
        if self.asr.lid_events:
            out += [m for m in self.asr.lid_events if math.isfinite(m.get("confidence", 0.0))]
            self.asr.lid_events = []
        if self.sil is not None:
            events += self._silero_events(self.sil.feed(x), final)
        events = sorted(events, key=lambda e: e[0])
        if self.cfg.turn_policy in HYBRID_POLICIES:
            events = self._hybrid(events)
        for t_dec, pol, ev, v in events:
            self._turn_end(pol, ev, v, t_dec, out)
        if self.asr.n_frames - self.seg_frame0 >= MAX_SEGMENT_S * 1000 / FRAME_MS and not final:
            # no policy cut this segment for MAX_SEGMENT_S: cut it with a final (no turn_end) so the partial and
            # the segment text stay bounded
            self._notice("segment_cap", f"segment open for {MAX_SEGMENT_S:.0f} s without a turn_end: cut with a final")
            out += self.take_notices()
            self._emit_final(self.asr.n_frames * FRAME_MS / 1000, self.asr.n_frames, self.timeout.primary, out)
        if self.la is not None and not final:
            self._lookahead_emit(out)
        text = self._seg_text()
        if text != self.last_partial and lvl < 2:
            self.last_partial = text
            out.append({"type": "partial", "t": round(self.asr.n_frames * FRAME_MS / 1000, 3), "text": text})
        elif text != self.last_partial:
            self._count_only("shed_partials")
        dt = time.perf_counter() - t0
        self.proc_s += dt
        self._rate.append((len(x), dt))
        while len(self._rate) > 1 and sum(s for s, _ in self._rate) - self._rate[0][0] >= SHED_WINDOW_S * SR:
            self._rate.popleft()
        # per 160 ms of audio: a block's time is spread over its samples (exact for the 20 ms blocks of a paced
        # stream; for large backlog blocks it is the average cost of the chunks in the block)
        pos, rate = self.samples - len(x), dt / max(1, len(x))
        while self.samples >= self._next_chunk:
            self._acc += (self._next_chunk - pos) * rate
            pos = self._next_chunk
            self.chunk_ms.append(self._acc * 1000)
            self._acc = 0.0
            self._next_chunk += CHUNK_SAMPLES
        self._acc += (self.samples - pos) * rate if len(x) else dt
        if arrived is not None:
            lag = (time.perf_counter() - arrived) * 1000
            self.send_lag_ms.append(lag)
            self.send_lag_ms_max = max(self.send_lag_ms_max, lag)
            if self.first_partial_ms is None and any(m["type"] == "partial" and m["text"] for m in out):
                self.first_partial_ms = round(lag, 1)
        return out

    def _count_only(self, code: str, n: int = 1) -> None:
        """A degradation counted in stats.degraded and the engine counters without an error message."""
        self.degraded[code] = self.degraded.get(code, 0) + n
        self.e.count(code, n)

    def finish(self, samples: np.ndarray | None = None, arrived: float | None = None) -> list[dict]:
        """End of stream: flush both models, emit the last final and the stats message."""
        if self.finished:
            return []
        self.finished = True  # (set before the flush: _asr_frames_at then counts the final partial chunk)
        out = self.process(np.zeros(0, np.float32) if samples is None else samples, final=True, arrived=arrived)
        spk, extra = self._attribute(self.seg_frame0, self.asr.n_frames, self.timeout.primary)
        out.append({"type": "final", "t": self.t, "text": self._seg_text(), "speaker": spk, **extra})
        if self.e.final_sources:
            out[-1]["source"] = "stream"
        if self.la is not None:
            self._lookahead_due(self.asr.n_frames, self.t, spk, out[-1]["text"])
            self._lookahead_emit(out)
        if self.fa:
            self._final_job(self.samples, self.asr.n_frames, self.t, spk, out, out[-1]["text"])
        self.seg_tok = len(self.asr.tokens)
        out.append(self.stats())
        return out

    def stats(self) -> dict:
        """The ``stats`` message: RTF, latency percentiles, memory (plus debug and option keys when enabled)."""
        audio = self.samples / SR
        m = {"type": "stats", "rtf": round(self.proc_s / audio, 4) if audio > 0 else 0.0,
             "chunk_ms_p50": _pct(self.chunk_ms, 50), "chunk_ms_p95": _pct(self.chunk_ms, 95),
             "first_partial_ms": self.first_partial_ms, "peak_rss_mb": peak_rss_mb()}
        if self.degraded:
            m["degraded"] = dict(sorted(self.degraded.items()))
        if self.registry is not None:
            m["speakers_seen"] = len(self.registry)
        if self.binder is None and self.tsvad is not None and self.e.enroll != "dominant":
            m.update(enroll=self.e.enroll, enrolled=self.tsvad.enrolled, primary_column=None)
        if self.binder is not None:
            m.update(enroll=self.e.enroll, enrolled=self.binder.enrolled, primary_column=self.binder.column)
            if self.e.debug:
                m.update(enroll_ms_p50=_pct(self.enroll_ms, 50), enroll_ms_max=round(max(self.enroll_ms, default=0.0), 1),
                         enroll_ms_mean=round(float(np.mean(self.enroll_ms)), 2) if self.enroll_ms else 0.0,
                         enroll_n_embed=self.binder.n_embed)
        if self.e.final_sources:
            m.update(final_asr=",".join(self.e.final_sources),
                     final_latency_ms={src: {"p50": _pct(lat, 50), "p95": _pct(lat, 95),
                                             "max": round(max(lat, default=0.0), 1), "n": len(lat)}
                                       for src, lat in self.final_lat_ms.items()},
                     final_asr_rss_mb=getattr(self.e.final_asr, "rss_mb", None))
            if self.e.debug and self.la is not None:
                m["lookahead_ms_mean"] = round(self.la.ms / max(1, len(self.chunk_ms)), 3)
        if self.asr.lid is not None:
            m["lang"] = self.asr.lid.current
            if self.e.debug:
                m["lid_ms_mean"] = round(self.asr.lid.ms / max(1, self.asr.n_frames), 4)
        if self.e.debug and self.sil is not None:
            m.update(silero_ms_p50=_pct(self.sil.ms, 50), silero_ms_p95=_pct(self.sil.ms, 95),
                     silero_ms_mean=round(float(np.mean(self.sil.ms)), 3) if self.sil.ms else 0.0,
                     silero_chunks=len(self.sil.ms))
        if self.e.debug:
            m.update(audio_s=round(audio, 3), proc_s=round(self.proc_s, 3), n_chunks=len(self.chunk_ms),
                     asr_ms_p50=_pct(self.asr_ms, 50), asr_ms_p95=_pct(self.asr_ms, 95),
                     diar_ms_p50=_pct(self.diar_ms, 50), diar_ms_p95=_pct(self.diar_ms, 95),
                     turn_ms_p50=round(self.asr.turn_ms / max(1, self.asr.n_frames), 3),
                     backlog_ms_max=round(self.backlog_ms_max, 1),
                     backlog_ms_end=round(self.backlog_ms[-1] if self.backlog_ms else 0.0, 1),
                     send_lag_ms_p50=_pct(self.send_lag_ms, 50), send_lag_ms_p95=_pct(self.send_lag_ms, 95),
                     send_lag_ms_max=round(self.send_lag_ms_max, 1),
                     diar_lag_ms_mean_measured=round(float(np.mean(self.diar_emit_lag)), 1)
                     if self.diar_emit_lag else None, turn_input=self.e.turn_input)
        return m


# --------------------------------------------------------------------------- websocket server
class _Conn:
    def __init__(self):
        self.inbox: deque = deque()  # float32 16 kHz blocks
        self.cmds: list = []  # (samples received when it arrived, 'agent_end' | 'enroll')
        self.n_in = 0  # samples in inbox
        self.received = 0  # samples received (16 kHz)
        self.processed = 0
        self.arrivals: deque = deque()  # (cumulative samples, perf_counter)
        self.carry = b""  # trailing bytes of a binary frame that do not complete a sample frame
        self.end = self.closed = False
        self.fail: tuple[str, str] | None = None  # (error code, detail) set by the reader on an internal failure
        self.session = None  # the Session once audio started (Engine.global_load)
        self.wake = asyncio.Event()
        self.drained = asyncio.Event()  # set after every processing round (reader backpressure)
        self.last_rx = time.perf_counter()

    def take(self, n: int) -> np.ndarray:
        parts, need = [], n
        while need and self.inbox:
            b = self.inbox[0]
            if len(b) <= need:
                parts.append(self.inbox.popleft())
                need -= len(b)
            else:
                parts.append(b[:need])
                self.inbox[0] = b[need:]
                need = 0
        x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        self.n_in -= len(x)
        self.processed += len(x)
        return x

    def arrival_of(self, cum: int) -> float | None:
        while self.arrivals and self.arrivals[0][0] < cum:
            self.arrivals.popleft()
        return self.arrivals[0][1] if self.arrivals else None


def _log(*a):
    log.info(" ".join(map(str, a)))


async def handle(ws, engine: Engine, log=_log):
    """One connection = one session (protocol: docs/PROTOCOL.md). Every client mistake answers with a
    structured ``error`` message (fatal ones close the connection with code 1008 / 1011); nothing a client sends
    can raise out of this coroutine (research/BULLETPROOF.md section 2)."""
    from websockets.exceptions import ConnectionClosed
    loop = asyncio.get_running_loop()
    peer = getattr(ws, "remote_address", None)
    peer_s = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) and len(peer) >= 2 else str(peer)
    cfg, conn = SessionConfig(), _Conn()
    session: Session | None = None
    resampler: Resampler | None = None
    t_conn = time.perf_counter()
    send_lock = asyncio.Lock()
    pending: list[asyncio.Task] = []  # --final-asr deliveries in flight
    engine.conns.add(conn)
    engine.sessions_total += 1
    max_inbox = int(MAX_INBOX_S * SR)
    max_session = int(engine.max_session_s * SR)

    def ev(event: str, **kw):
        engine.event(event, peer=peer_s, **kw)

    async def send(msgs):
        async with send_lock:
            for m in msgs:
                if engine.debug:
                    validate(m, debug=True)
                await ws.send(json.dumps(m, allow_nan=False))

    async def send_error(code: str, detail: str, fatal: bool = False):
        engine.count(f"error_{code}")
        ev("error", code=code, detail=detail, fatal=fatal)
        try:
            await send([error_msg(code, detail, fatal)])
        except Exception:  # noqa: BLE001 - the client may already be gone
            pass

    def submit_finals():
        """--final-asr: start the offline pass of every finished turn now; returns the deliveries to schedule once
        the streaming messages that precede them have been sent."""
        if session is None or engine.final_asr is None:
            return []
        return [(j, engine.submit_final(j["audio"]), time.perf_counter()) for j in session.take_final_jobs()]

    async def deliver(job, fut, t_sub):
        try:
            try:
                res = await asyncio.wait_for(asyncio.wrap_future(fut), FINAL_ASR_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001 - a failed / hung pass must not end the session: stream text
                hung = isinstance(e, asyncio.TimeoutError)
                if hung:
                    fut.cancel()
                    engine.count("final_asr_timeout")
                    e = TimeoutError(f"no answer within {FINAL_ASR_TIMEOUT_S:g} s")
                ev("final_asr_failed", source=job["msg"]["source"], error=f"{type(e).__name__}: {e}", hung=hung)
                engine.final_done(e, hung=hung)
                await send(session.fallback_final(job, e, (time.perf_counter() - t_sub) * 1000))
                return
            engine.final_done()
            await send([session.complete_final(job, res, (time.perf_counter() - t_sub) * 1000)])
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:  # noqa: BLE001
            ev("deliver_failed", error=f"{type(e).__name__}: {e}")

    def schedule(subs):
        for job, fut, t_sub in subs:
            pending.append(asyncio.create_task(deliver(job, fut, t_sub)))

    async def on_text(msg: str):
        nonlocal session
        if len(msg) > MAX_TEXT_MESSAGE:
            await send_error("message_too_large", f"text frame of {len(msg)} bytes ignored (max {MAX_TEXT_MESSAGE})")
            return
        try:
            d = json.loads(msg)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            await send_error("bad_json", f"text frame is not JSON: {msg[:80]!r}")
            return
        if not isinstance(d, dict) or not isinstance(d.get("type"), str):
            await send_error("bad_message", f"expected an object with a string 'type', got {msg[:80]!r}")
            return
        typ = d["type"]
        if typ == "config":
            if resampler is not None and session is None:
                # audio has arrived but has not been processed yet: the session's policy is fixed by the config as
                # it was at the first audio frame, so create it now and let it judge this late config (otherwise
                # a turn_policy sent after audio would be accepted silently)
                session = Session(engine, cfg)
                conn.session = session
            warn = cfg.update(d, audio_started=resampler is not None)
            if session is not None:
                session.apply_config(cfg)
                warn += [m["detail"] for m in session.take_notices()]
            if warn:
                await send_error("bad_config", "; ".join(warn))
            ev("config", config=str(cfg))
        elif typ in ("agent_end", "enroll"):  # --enroll triggers, applied in audio order
            emb = d.get("embedding")  # --turn-input tsvad: a stored voice print (192 floats) instead of live speech
            if emb is not None and (engine.tsvad is None or not isinstance(emb, list) or len(emb) != 192 or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in emb)):
                await send_error("bad_message", "embedding ignored: needs --turn-input tsvad and 192 finite numbers")
                emb = None
            if ENROLL_TRIGGER.get(engine.enroll) == typ or (typ == "enroll" and emb is not None):
                conn.cmds.append((conn.received, typ, emb))  # a stored print (tsvad) is taken under any --enroll
                conn.wake.set()
            else:
                await send_error("unsupported", f"{typ} ignored: the server runs --enroll {engine.enroll}")
        elif typ == "end":
            conn.end = True
            conn.wake.set()
        else:
            await send_error("unknown_type", f"unknown message type {typ!r}")

    async def reader():
        nonlocal resampler
        try:
            async for msg in ws:
                conn.last_rx = time.perf_counter()
                if isinstance(msg, (bytes, bytearray)):
                    while conn.n_in > max_inbox and not conn.closed:  # backpressure: stop reading until drained
                        engine.count("inbox_backpressure")
                        conn.drained.clear()
                        await conn.drained.wait()
                    x, conn.carry, bad = decode_pcm(conn.carry + bytes(msg), cfg)
                    if bad:
                        engine.count("nan_input", bad)
                    if resampler is None:
                        resampler = Resampler(cfg.sample_rate)
                    x = resampler(x)
                    if len(x):
                        conn.inbox.append(x)
                        conn.n_in += len(x)
                        conn.received += len(x)
                        conn.arrivals.append((conn.received, time.perf_counter()))
                        conn.wake.set()
                    continue
                await on_text(msg)
                if conn.end:
                    return
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - never a silent death: the main loop reports and closes
            conn.fail = ("internal_error", f"{type(e).__name__}: {e}")
            ev("reader_failed", error=conn.fail[1], trace=traceback.format_exc(limit=3))
        finally:
            if not conn.end:
                conn.closed = True
            conn.wake.set()
            conn.drained.set()

    async def run(fn, *args):
        """Run a session step on the compute thread; a failure ends the session with a fatal error message."""
        try:
            return await loop.run_in_executor(engine.executor, fn, *args)
        except Exception as e:  # noqa: BLE001
            engine.count("processing_failed")
            ev("processing_failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=5))
            raise _SessionFailed("processing_failed", f"{type(e).__name__}: {e}") from e

    close_code = 1000
    try:
        await send([engine.ready_msg()])
        rtask = asyncio.create_task(reader())
        while True:
            try:
                await asyncio.wait_for(conn.wake.wait(), timeout=engine.idle_timeout_s)
            except asyncio.TimeoutError:
                idle = time.perf_counter() - conn.last_rx
                if idle >= engine.idle_timeout_s and not conn.closed:
                    engine.count("idle_timeout")
                    raise _SessionFailed("idle_timeout", f"no message for {idle:.0f} s", 1008) from None
                continue
            conn.wake.clear()
            if conn.closed:
                break
            if conn.fail is not None:
                raise _SessionFailed(*conn.fail)
            if conn.cmds and session is None:
                session = Session(engine, cfg)
            while conn.cmds:  # arm at the audio position the message arrived at (frames >= it are searched)
                n_recv, kind, emb = conn.cmds.pop(0)
                session.arm_enrollment(kind, n_recv // FRAME_SAMPLES, emb)
                ev("enrollment_armed", trigger=kind, t=round(n_recv / SR, 2))
            while conn.n_in >= FRAME_SAMPLES // 4 or (conn.end and conn.n_in):
                if session is None:
                    session = Session(engine, cfg)
                conn.session = session
                backlog = conn.n_in / SR * 1000
                shed = engine.global_shed()
                x = conn.take(min(conn.n_in, MAX_ROUND_SAMPLES))
                conn.drained.set()
                arrived = conn.arrival_of(conn.processed)
                msgs = await run(session.process, x, False, arrived, backlog, shed)
                subs = submit_finals()
                await send(msgs)
                schedule(subs)
                if conn.closed:
                    break
                if session.samples >= max_session:
                    conn.end = True
                    engine.count("session_limit")
                    await send_error("session_limit", f"session reached --max-session-s {engine.max_session_s:.0f}: "
                                                      f"flushing and closing", fatal=True)
                    close_code = 1008
                    break
            if conn.closed:
                break
            if conn.end:
                if session is None:
                    session = Session(engine, cfg)
                msgs = await run(session.finish, None, time.perf_counter())
                if engine.final_asr is not None:  # the stats wait for every offline final of the session
                    subs = submit_finals()
                    await send(msgs[:-1])
                    schedule(subs)
                    await asyncio.gather(*pending, return_exceptions=True)
                    msgs = [session.stats()]
                await send(msgs)
                st = msgs[-1]
                ev("end", audio_s=round(session.t, 2), wall_s=round(time.perf_counter() - t_conn, 1), stats=st)
                await ws.close(close_code)
                break
        if conn.closed:
            ev("disconnected", audio_s=round(conn.received / SR, 2), note="session dropped")
    except ConnectionClosed:
        ev("closed_while_sending", note="session dropped")
    except _SessionFailed as f:
        await send_error(f.code, f.detail, fatal=True)
        try:
            await ws.close(f.close_code)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001 - the last line of defense: report, close, keep serving
        engine.count("error_internal_error")
        ev("handler_failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=5))
        try:
            await send([error_msg("internal_error", f"{type(e).__name__}: {e}", fatal=True)])
            await ws.close(1011)
        except Exception:  # noqa: BLE001
            pass
    finally:
        engine.conns.discard(conn)
        conn.closed = True
        conn.drained.set()
        try:
            rtask.cancel()
        except NameError:
            pass
        for t in pending:
            t.cancel()


class _SessionFailed(Exception):
    def __init__(self, code: str, detail: str, close_code: int = 1011):
        super().__init__(detail)
        self.code, self.detail, self.close_code = code, detail, close_code


def _health_request(engine: Engine):
    """websockets ``process_request`` hook: GET /health (also /healthz, /stats) answers with Engine.health() JSON
    over plain HTTP; every other path proceeds with the WebSocket handshake."""
    def hook(connection, request):
        path = getattr(request, "path", "") or ""
        if path.split("?")[0] in ("/health", "/healthz", "/stats"):
            try:
                body = json.dumps(engine.health()) + "\n"
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n"
            resp = connection.respond(200, body)
            try:
                resp.headers["Content-Type"] = "application/json"
            except Exception:  # noqa: BLE001
                pass
            return resp
        return None
    return hook


async def serve(engine: Engine, host: str = "127.0.0.1", port: int = 8765, ready_event=None, stop=None):
    """Accept WebSocket connections (and ``GET /health``) on ``host:port`` until ``stop`` resolves."""
    from websockets.asyncio.server import serve as ws_serve
    async with ws_serve(lambda ws: handle(ws, engine), host, port, max_size=2 ** 22, ping_interval=None,
                        process_request=_health_request(engine)) as server:
        port = server.sockets[0].getsockname()[1] if server.sockets else port
        log.info(f"[serve] listening on ws://{host}:{port} (GET http://{host}:{port}/health)")
        if ready_event is not None:
            ready_event(port)
        try:
            await (stop if stop is not None else asyncio.Future())
        finally:  # shutdown mid-session: tell every live client before the sockets go away
            conns = list(server.connections) if hasattr(server, "connections") else []
            for c in conns:
                try:
                    await asyncio.wait_for(c.send(json.dumps(error_msg("server_shutdown", "server stopping", True))),
                                           1.0)
                    await asyncio.wait_for(c.close(1001), 1.0)
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    main()
