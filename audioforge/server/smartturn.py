"""``--turn-model smartturn``: Pipecat's smart-turn v3.2 classifier inside the served ``vad_head`` rule, a bridge.

This is a SECOND model (pipecat-ai/smart-turn v3.2, BSD-2-Clause: a Whisper-tiny encoder + linear head, 8.7 MB ONNX,
the one Pipecat bundles) that stands in for our turn head's end-of-turn decision on assistant-directed speech while
the native classifier (turn head v5) is trained. It is opt-in; ``--turn-model head`` (default) is the shipped rule.

When it runs (``VadHeadPolicy(turn_model=...)``): once per silence run, at the frame where ``vad_head``'s silence
reaches its quiet trigger (``SMARTTURN_QUIET_MS``). P(complete) > 0.5 -> ``turn_end`` at once (path ``model``);
otherwise the rule waits: the next silence run is classified again, and the preset's smart-turn fallback
(``TURN_PRESETS[...]["smartturn_fallback_ms"]``) ends the turn if the user stays quiet.

Input, exactly as Pipecat 1.12's ``LocalSmartTurnAnalyzerV3`` / ``BaseSmartTurn`` prepares it
(``audioforge.baselines.turn.pipecat_smartturn_replay`` drives those classes offline):
  * the turn's audio from its first speech frame - ``pre_speech_s`` (0.5 s, Pipecat's ``pre_speech_ms``; Pipecat adds
    its VAD's start delay to reach back to the same onset), never before the previous turn_end (Pipecat clears its
    buffer there), up to the decision time (every sample the rule's frame had at its decision-ready time);
  * int16-quantised (Pipecat receives int16 PCM), the last <= 8 s kept (``max_duration_secs``), left-padded with zeros
    to 8 s, Whisper log-mel with do_normalize (Pipecat's vendored numpy ``compute_whisper_log_mel_features`` when the
    ``pipecat-ai`` package is installed, else transformers' ``WhisperFeatureExtractor(chunk_length=8)``, which that
    file mirrors), -> ONNX -> P(complete).
The model file: ``--smartturn-onnx PATH``, default the copy bundled in the installed ``pipecat-ai`` package.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

from .constants import FRAME_SAMPLES, SR

__all__ = ["SmartTurnModel", "SmartTurnTrigger", "bundled_onnx"]

MAX_S = 8.0  # Pipecat SmartTurnParams.max_duration_secs and the model's input window
PRE_SPEECH_S = 0.5  # Pipecat SmartTurnParams.pre_speech_ms


def _pipecat_dir() -> Path | None:
    spec = importlib.util.find_spec("pipecat")  # locates the package without importing it (its import logs a banner)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "audio" / "turn" / "smart_turn"


def bundled_onnx() -> Path | None:
    """smart-turn-v3.2-cpu.onnx from the installed pipecat-ai package, or None."""
    d = _pipecat_dir()
    p = d / "data" / "smart-turn-v3.2-cpu.onnx" if d is not None else None
    return p if p is not None and p.exists() else None


def _pipecat_features():
    """Pipecat's vendored numpy log-mel (``_whisper_features.compute_whisper_log_mel_features``), loaded from its file
    (numpy only; no pipecat package import), or None."""
    d = _pipecat_dir()
    f = d / "_whisper_features.py" if d is not None else None
    if f is None or not f.exists():
        return None
    spec = importlib.util.spec_from_file_location("_audioforge_pipecat_whisper_features", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.compute_whisper_log_mel_features


class SmartTurnModel:
    """smart-turn v3.2 ONNX, shared by every session of an engine. ``predict(segment)`` -> (P(complete), ms)."""

    def __init__(self, onnx_path: str | Path | None = None, threads: int = 1):
        import onnxruntime as ort
        path = Path(onnx_path) if onnx_path else bundled_onnx()
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"--turn-model smartturn: no smart-turn ONNX ({onnx_path or 'pipecat-ai not installed'}); pass "
                f"--smartturn-onnx PATH (smart-turn-v3.2-cpu.onnx, pipecat-ai/smart-turn, BSD-2-Clause)")
        so = ort.SessionOptions()  # Pipecat LocalSmartTurnAnalyzerV3's session options
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.path, self.threads = str(path), int(threads)
        self.session = ort.InferenceSession(self.path, sess_options=so, providers=["CPUExecutionProvider"])
        self._mel = _pipecat_features()
        self.features_impl = "pipecat" if self._mel is not None else "transformers"
        if self._mel is None:
            from transformers import WhisperFeatureExtractor
            self._fe = WhisperFeatureExtractor(chunk_length=int(MAX_S))

    def features(self, x: np.ndarray) -> np.ndarray:
        n = int(MAX_S * SR)
        x = np.asarray(x, np.float32)[-n:]
        if len(x) < n:
            x = np.pad(x, (n - len(x), 0))
        if self._mel is not None:
            return np.expand_dims(self._mel(x, do_normalize=True), 0).astype(np.float32)
        return self._fe(x, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=n,
                        truncation=True, do_normalize=True).input_features.astype(np.float32)

    def predict(self, segment: np.ndarray) -> tuple[float, float]:
        """P(complete) of one prepared segment (``SmartTurnTrigger``) and the wall time of features + ONNX (ms)."""
        t0 = time.perf_counter()
        if len(segment) == 0:
            return 0.0, 0.0
        p = float(np.asarray(self.session.run(None, {"input_features": self.features(segment)})[0]).reshape(-1)[0])
        return p, (time.perf_counter() - t0) * 1000


class SmartTurnTrigger:
    """The ``turn_model`` callable ``VadHeadPolicy`` calls at its quiet trigger: ``(v, onset_v) -> (P(complete), ms)``.
    ``get_audio(a, b)`` returns absolute samples [a, b) of the session (what is still buffered of them),
    ``ready_sample(v)`` frame v's decision-ready sample; ``turn_ended(sample)`` after every turn_end moves the start
    limit (Pipecat clears its buffer at a turn end). ``calls`` keeps {"v", "p", "ms"} per call."""

    def __init__(self, model, get_audio, ready_sample, pre_speech_s: float = PRE_SPEECH_S):
        self.model, self.get_audio, self.ready_sample, self.pre = model, get_audio, ready_sample, float(pre_speech_s)
        self.limit = 0
        self.calls: list[dict] = []

    def turn_ended(self, sample: int) -> None:
        self.limit = max(self.limit, int(sample))

    def __call__(self, v: int, onset_v: int) -> tuple[float, float]:
        end = int(self.ready_sample(v))
        a = max(int(onset_v) * FRAME_SAMPLES - int(round(self.pre * SR)), self.limit, end - int(MAX_S * SR), 0)
        x = np.asarray(self.get_audio(a, end), np.float32)
        seg = np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.float32) / 32768.0
        p, ms = self.model.predict(seg)
        self.calls.append({"v": int(v), "p": float(p), "ms": float(ms)})
        return p, ms
