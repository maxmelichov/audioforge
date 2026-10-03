"""The CLEANUP_TODO "suspicious" spots, each pinned by a test."""
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.metrics import pct, pct_dict


@pytest.fixture(autouse=True, scope="module")
def _threads_1():
    """1 torch thread for this module only (restored after it, not at import time for the session)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "examples"))


# ---------------------------------------------------------------- one percentile helper (serve / stream_client / ami)
def test_pct_semantics_cover_the_three_old_helpers():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert pct(xs, 50) == 2.5 and pct(xs, 50, nd=2) == 2.5 and pct([1.234], 90, nd=2) == 1.23
    assert pct([], 50) is None and pct([], 95, empty=0.0) == 0.0  # stream_client / serve empties
    assert pct_dict([]) == {} and pct_dict(xs) == {"p10": 1.3, "p25": 1.75, "p50": 2.5, "p75": 3.25, "p90": 3.7}
    from audioforge import serve
    from audioforge.datasets import ami
    assert serve._pct([], 50) == 0.0 and serve._pct([1.006], 50) == 1.01
    assert ami._pct(np.array([0.5])) == {"p10": 0.5, "p25": 0.5, "p50": 0.5, "p75": 0.5, "p90": 0.5}


# ---------------------------------------------------------------- librispeech: a non-206 answer is not retried
def test_fetch_range_fails_fast_when_range_is_ignored(tmp_path, monkeypatch):
    from audioforge.datasets import librispeech as ls
    calls = []

    class R(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=30):
        calls.append(req.headers.get("Range"))
        return R(b"x" * 10)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ls.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="ignored the Range header"):
        ls._fetch_range("http://x/y.tar.gz", tmp_path / "part0", 0, 99, retries=50)
    assert len(calls) == 1  # was: swallowed and retried `retries` times (~17 min at 2 s each)


# ---------------------------------------------------------------- model.py: CTC dedup state is initialised
def test_streaming_session_ctc_prev_is_an_init_attribute():
    from audioforge.model import SpeechModel, StreamingSession
    from audioforge.tokenizer import CharTokenizer
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none"},
           "encoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1]},
           "heads": {"ctc": {"type": "ctc"}}, "decoding": {"primary": "ctc"}}
    s = StreamingSession(SpeechModel(cfg, CharTokenizer(list("abcd "))).eval())
    assert s._prev == -1
    s.feed(np.zeros(4000, np.float32))
    assert isinstance(s._prev, int)


# ---------------------------------------------------------------- benchmark_teachers: ru_maxrss units
def test_peak_memory_cpu_is_bytes():
    bt = pytest.importorskip("benchmark_teachers")
    with bt.PeakMemory(torch.device("cpu")) as pm:
        pass
    assert pm.gb > 0.01  # this process is > 10 MB; KiB misread as bytes would give ~1e-5


# ---------------------------------------------------------------- turn_error_analysis: --n must match stage 1
def test_turn_error_analysis_refuses_a_score_file_for_another_n(tmp_path, monkeypatch):
    tea = pytest.importorskip("turn_error_analysis")
    np.savez(tmp_path / "scores.npz", oracle=np.array([np.zeros(3)] * 5, dtype=object))
    (tmp_path / "scores_meta.json").write_text(json.dumps({"ckpt": "x", "n": 5, "chunk": 1}))
    monkeypatch.setattr(tea, "load_turns", lambda n, cache: {"on": [0] * n, "en": [1] * n, "val": [None] * n})
    a = type("A", (), dict(scratch=str(tmp_path), n=7, diar_cache=str(tmp_path), out=None))
    with pytest.raises(SystemExit, match="holds 5"):
        tea.stage_analyze(a)


# ---------------------------------------------------------------- livekit demo reads the prepared reference
def test_livekit_demo_load_ref_reads_ref_json(tmp_path):
    pytest.importorskip("livekit.agents")
    lk = pytest.importorskip("livekit_offline_demo")
    wav = tmp_path / "w1.wav"
    (tmp_path / "w1.ref.json").write_text(json.dumps({"onset_s": 1.5}))
    assert lk.load_ref(wav) == {"onset_s": 1.5}
    (tmp_path / "w1.ref.json").unlink()
    (tmp_path / "w1.json").write_text(json.dumps({"onset_s": 2.5}))
    assert lk.load_ref(wav) == {"onset_s": 2.5}
    assert lk.load_ref(tmp_path / "none.wav") == {}
