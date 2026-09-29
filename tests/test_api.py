"""The in-process Python API (audioforge.load / Frontend.session / Session.feed) on tiny random models."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import audioforge
from audioforge.api import Frontend

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("test_serve_helpers_api", ROOT / "tests" / "test_serve.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def _fe():
    return Frontend(H._engine())


def test_top_level_exports():
    assert {"load", "Frontend", "Session", "__version__"} <= set(audioforge.__all__)
    assert audioforge.load is not None and isinstance(audioforge.__version__, str)


def test_feed_matches_the_server_session_and_drops_frames_by_default():
    from audioforge.serve import Session as ServeSession
    from audioforge.serve import SessionConfig

    audio = H._audio(2.0)
    ref = ServeSession(H._engine(), SessionConfig())
    want = ref.process(audio) + ref.finish()
    s = _fe().session(frames=True)
    got = s.feed(audio) + s.end()
    strip = lambda ms: [{k: v for k, v in m.items() if k not in ("rtf", "chunk_ms_p50", "chunk_ms_p95",  # noqa: E731
                                                                 "first_partial_ms", "peak_rss_mb")} for m in ms]
    assert strip(got) == strip(want)
    s2 = _fe().session()
    quiet = s2.feed(audio) + s2.end()
    assert quiet and not any(m["type"] in ("frame", "frames") for m in quiet) and quiet[-1]["type"] == "stats"


def test_pcm_formats_and_resampling():
    audio = H._audio(1.0)
    i16 = (audio * 32767).astype(np.int16)
    a = _fe().session(frames=True)
    b = _fe().session(frames=True)
    ma, mb = a.feed(i16), b.feed(i16.tobytes())
    assert [m["type"] for m in ma] == [m["type"] for m in mb]
    s = _fe().session(frames=True, sample_rate=8000)
    msgs = s.feed(audio[::2]) + s.end()
    assert sum(m["type"] == "frame" for m in msgs) == pytest.approx(12, abs=2)  # 1 s at 80 ms frames


def test_bad_config_and_use_after_end():
    with pytest.raises(ValueError, match="turn_policy"):
        _fe().session(turn_policy="nope")
    s = _fe().session()
    s.end()
    with pytest.raises(RuntimeError):
        s.feed(np.zeros(160, np.float32))


def test_load_reports_missing_models(tmp_path, monkeypatch):
    from audioforge import hub

    monkeypatch.setattr(hub, "find_model", lambda key, d=None: None)
    with pytest.raises(FileNotFoundError, match="audioforge-download"):
        audioforge.load(models_dir=tmp_path)
