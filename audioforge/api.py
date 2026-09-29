"""The in-process Python API: the served stack without a WebSocket.

    import audioforge

    fe = audioforge.load()                       # single-model mode, the models audioforge-download installed
    s = fe.session(turn_policy="hybrid_dyn")
    s.enroll(fe.voiceprint(user_clean_speech))   # the user's stored print: >= 5 s of clean speech (10 s for meetings)
    for block in blocks_of_pcm:                  # any length; int16 or float32, 16 kHz unless sample_rate= says so
        for ev in s.feed(block):
            print(ev["type"], ev)
    print(s.end())                               # the last final and the stats

Events are the protocol's messages as dicts (``partial``, ``turn_end``, ``final``, ``stats``, ...; docs/PROTOCOL.md),
produced by exactly the code the server runs (``audioforge.serve.Engine`` / ``Session``). ``frame`` messages (one
per 80 ms) are left out unless ``frames=True``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Frontend", "Session", "load", "voiceprint"]

log = logging.getLogger(__name__)


class Session:
    """One stream of audio (one call / one room). Create it with ``Frontend.session``."""

    def __init__(self, frontend: Frontend, frames: bool = False, **config: Any) -> None:
        from .serve import SR, SessionConfig
        from .serve import Session as _ServeSession
        from .server.streams import Resampler

        cfg = SessionConfig()
        warnings = cfg.update(config)
        if warnings:
            raise ValueError("; ".join(warnings))
        self.config = cfg
        self.frames = frames
        self._s = _ServeSession(frontend.engine, cfg)
        self._resample = Resampler(cfg.sample_rate) if cfg.sample_rate != SR else None
        self._ended = False

    def _pcm(self, pcm: np.ndarray | bytes) -> np.ndarray:
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            pcm = np.frombuffer(bytes(pcm), dtype="<i2")
        x = np.asarray(pcm)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        x = x.astype(np.float32, copy=False)
        if x.ndim == 2:  # (samples, channels) -> mono
            x = x.mean(1).astype(np.float32)
        return self._resample(x) if self._resample is not None else x

    def _out(self, msgs: list[dict]) -> list[dict]:
        return msgs if self.frames else [m for m in msgs if m["type"] not in ("frame", "frames")]

    def feed(self, pcm: np.ndarray | bytes) -> list[dict]:
        """Process a block of audio (float in [-1, 1] or int16; ``bytes`` = int16 little-endian) and return the
        events it completed, in order."""
        if self._ended:
            raise RuntimeError("session already ended")
        msgs = self._s.process(self._pcm(pcm))
        if self._s.final_jobs:  # --final-asr: the offline finals, right after their streaming final
            msgs += self._s.run_final_jobs_sync()
        return self._out(msgs)

    def agent_end(self) -> bool:
        """The agent's own speech (TTS) ended now: arms ``--enroll after_agent`` / ``after_agent_arm`` binding."""
        return self._s.arm_enrollment("agent_end")

    def enroll(self, embedding: list[float] | None = None) -> bool:
        """``--enroll explicit``: the next utterance (or the given voice print) is the user."""
        return self._s.arm_enrollment("enroll", embedding=embedding)

    def end(self) -> list[dict]:
        """Flush: the last ``final`` (and any offline finals still due) and ``stats``."""
        self._ended = True
        msgs = self._s.finish()
        if self._s.final_jobs:  # the offline finals come before stats
            stats = [m for m in msgs if m["type"] == "stats"]
            msgs = [m for m in msgs if m["type"] != "stats"] + self._s.run_final_jobs_sync() + stats
        return self._out(msgs)


class Frontend:
    """The loaded models, shared by any number of sessions (one at a time per thread)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def voiceprint(self, audio, sample_rate: int = 16000) -> list[float]:
        """The user's voice print (192 floats) from their clean speech (float or int16 array, any rate): what
        ``Session.enroll(embedding)`` and the protocol's ``{"type": "enroll", "embedding": [...]}`` take. Use at least
        5 s of the user alone (10 s for meetings); research/SINGLE_MODEL.md A2 measured shorter and live prints."""
        return _voiceprint(self.engine.asr, audio, sample_rate)

    def session(self, turn_policy: str = "timeout", *, frames: bool = False, **config: Any) -> Session:
        """A new stream. ``config`` takes the protocol's session options: ``timeout_ms``, ``eot_threshold``,
        ``sample_rate`` (default 16000); ``turn_policy`` is one of ``audioforge.serve.POLICIES``."""
        return Session(self, frames=frames, turn_policy=turn_policy, **config)

    def run_file(self, path: str | os.PathLike, turn_policy: str = "timeout", block_ms: int = 160,
                 frames: bool = False) -> list[dict]:
        """Every event of an audio file (any rate, resampled to 16 kHz), fed in ``block_ms`` blocks."""
        from .data import load_wav

        audio = load_wav(str(path), 16000).astype(np.float32)
        s = self.session(turn_policy, frames=frames)
        n = 16000 * block_ms // 1000
        out: list[dict] = []
        for i in range(0, len(audio), n):
            out += s.feed(audio[i:i + n])
        return out + s.end()


def load(diarizer: str | None = None, models_dir: str | Path | None = None, *, asr: str | None = None,
         diar: str | None = None, threads: int = 2, warmup: bool = True, mode: str | None = None,
         device: str = "cpu", **engine_options: Any) -> Frontend:
    """Load the served models (as ``audioforge-serve`` does) and return a ``Frontend``.

    ``mode``: ``single`` (the default, as ``audioforge-serve``) or ``room``; without ``mode``, a ``diarizer``, ``diar``
    or ``final_asr`` selects room. ``diarizer`` (room): ``nemotron3`` (default if downloaded) or ``sortformer``;
    ``models_dir``: where
    ``audioforge-download`` put them (default ``$AUDIOFORGE_HOME``, else ``<repo>/models``, else
    ``~/.cache/audioforge``); ``asr`` / ``diar`` override the paths. ``device``: ``cpu`` (default) or ``cuda`` /
    ``cuda:N`` (as ``audioforge-serve --device``). ``engine_options`` are
    ``audioforge.serve.Engine.load`` keyword arguments (the server flags with underscores, e.g. ``enroll``,
    ``final_asr``, ``diar_labels``). ``mode="single"`` is ``audioforge-serve --mode single``: the one 115M model
    for a known user (TS-VAD turn input, no diarizer loaded, the distilled LID head, no final ASR); pass the user's
    voice print with ``Session.enroll(embedding)`` or let it be taken after ``Session.agent_end()``.
    """
    import torch

    from . import hub
    from .serve import Engine

    def need(key: str) -> str:
        p = hub.find_model(key, models_dir)
        if p is None:
            raise FileNotFoundError(f"model '{key}' not found in {hub.models_dir(models_dir)}; run: audioforge-download"
                                    + (f" --diarizer {key}" if key in hub.DIARIZERS else ""))
        return str(p)

    torch.set_num_threads(threads)
    if mode is None:
        mode = "room" if (diarizer or diar or engine_options.get("final_asr")) else "single"
    if mode == "single":
        from .launch import LID_FILE, TSVAD_FILE, find_head
        from .server.cli import MODES, SINGLE_CONFLICTS
        bad = [k for k, _ in SINGLE_CONFLICTS if engine_options.get(k) not in (None, False, "spk")] + (["diar"] if diar else [])
        if bad:
            raise ValueError(f"mode='single' loads one model; drop {bad}")
        opts = {**MODES["single"], **engine_options}
        if "lid" not in engine_options and opts.get("lid") == "head" and find_head(LID_FILE, None if models_dir is None else str(models_dir)) is None:
            log.warning(f"{LID_FILE} not found, language ID is off (as audioforge-serve)")
            opts["lid"] = None
        if opts.get("tsvad") is None:
            p = find_head(TSVAD_FILE, models_dir)
            if p is None:
                raise FileNotFoundError(f"mode='single' needs the TS-VAD head {TSVAD_FILE}; run: audioforge-download")
            opts["tsvad"] = str(p)
        engine = Engine.load(asr or need("asr"), None, device, threads=threads, **opts)
        if warmup:
            engine.warmup()
        return Frontend(engine)
    elif mode != "room":
        raise ValueError(f"unknown mode {mode!r} (single | room)")
    if diarizer is None and diar is None:
        diarizer = "nemotron3" if hub.find_model("nemotron3", models_dir) or not hub.find_model("sortformer", models_dir) \
            else "sortformer"
    opts = hub.diarizer_defaults(diarizer) if diar is None else {}
    opts.update(engine_options)
    if opts.get("final_asr") == "tdt_v3" and "AUDIOFORGE_TDT_V3" not in os.environ:
        p = hub.find_model("tdt_v3", models_dir)
        if p is not None:
            os.environ["AUDIOFORGE_TDT_V3"] = str(p)
    engine = Engine.load(asr or need("asr"), diar or need(diarizer), device, threads=threads, **opts)
    if warmup:
        engine.warmup()
    return Frontend(engine)


def _voiceprint(model, audio, sample_rate: int = 16000) -> list[float]:
    from .tsvad_stream import voiceprint as vp
    x = np.asarray(audio)
    x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sample_rate != 16000:  # the server's own resampler (no torchaudio in the serve extra)
        from .server.streams import Resampler
        x = Resampler(sample_rate)(x)
    if len(x) < 16000:
        raise ValueError("voiceprint: need at least 1 s of speech (5 s or more recommended)")
    return [round(float(v), 6) for v in vp(model, x)]


def voiceprint(audio, sample_rate: int = 16000, models_dir: str | Path | None = None, asr: str | None = None
               ) -> list[float]:
    """A voice print without loading the whole server stack: only the served ASR model (see Frontend.voiceprint)."""
    from . import hub
    from .train import load_model
    path = asr or hub.find_model("asr", models_dir)
    if path is None:
        raise FileNotFoundError("model 'asr' not found; run: audioforge-download")
    return _voiceprint(load_model(str(path), "cpu").eval(), audio, sample_rate)


EXAMPLE_WAV = Path(__file__).resolve().parents[1] / "examples" / "audio" / "two_party_call_16s.wav"  # single mode
ROOM_EXAMPLE_WAV = Path(__file__).resolve().parents[1] / "examples" / "audio" / "two_speakers_10s.wav"  # room mode
