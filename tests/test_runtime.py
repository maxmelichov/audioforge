"""StreamingRuntime (audioforge/runtime.py) == StreamingSession, int8/trace/ONNX paths, warmup."""
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.data import synthetic_dataset
from audioforge.model import SpeechModel, StreamingSession
from audioforge.runtime import StreamingRuntime, export_chunk_onnx
from audioforge.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "runs" / "voice_agent_frontend.afm"


def _session(m, audio, att=None, push=1000):
    s = StreamingSession(m, att_context_size=att)
    for i in range(0, len(audio), push):
        s.feed(audio[i:i + push])
    s.feed([], final=True)
    return s


def _runtime(rt, audio, push=777):
    rt.reset()
    for i in range(0, len(audio), push):
        rt.feed(audio[i:i + push])
    rt.feed([], final=True)
    return rt


def _random_model(att, seed=0):
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "fixed", "dither": 0.0},
           "encoder": {"d_model": 48, "n_layers": 3, "n_heads": 4, "subsampling_channels": 16, "causal": True,
                       "att_context_size": att, "dropout": 0.0},
           "heads": {"tdt": {"type": "tdt", "pred_hidden": 32, "joint_hidden": 32},
                     "ctc": {"type": "ctc"},
                     "vad": {"type": "frame", "key": "vad", "hidden": 16},
                     "spk": {"type": "speaker", "num_speakers": 3, "emb_dim": 16, "from_layers": "all"}}}
    m = SpeechModel(cfg, CharTokenizer(list("abcdefg "))).eval()
    with torch.no_grad():
        m.layer_mix["spk"].normal_()
    return m


@pytest.mark.parametrize("att", [[16, 1], [8, 3], [16, 0], [6, 1]])
@pytest.mark.parametrize("n_samples", [16000, 16000 + 1234])  # whole chunks / a partial last chunk
def test_runtime_frames_match_session(att, n_samples):
    """Fixed-shape step (masked caches, first/partial chunks) == stream_step, frame by frame."""
    m = _random_model(att)
    x = (np.random.default_rng(1).standard_normal(n_samples) * 0.1).astype(np.float32)
    s = _session(m, x, att)
    rt = _runtime(StreamingRuntime(m, att_context_size=att), x)
    assert len(rt.frame_events["vad"]) == len(s.frame_events["vad"])
    assert np.allclose(rt.frame_events["vad"], s.frame_events["vad"], atol=1e-5)
    assert rt.tokens == s.tokens
    # the streamed layer-mix frames pool into the same embedding as the offline speaker head
    enc, elen, hidden = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]), att, return_hidden=True)
    ref = m.heads["spk"].embed(m.head_input("spk", enc, hidden), elen)
    assert torch.allclose(rt.speaker_embedding(), ref, atol=1e-4)


def test_trace_backend_matches_eager():
    m = _random_model([16, 1])
    x = (np.random.default_rng(2).standard_normal(20000) * 0.1).astype(np.float32)
    a = _runtime(StreamingRuntime(m), x)
    b = _runtime(StreamingRuntime(m, backend="trace"), x)
    assert a.tokens == b.tokens
    assert np.allclose(a.frame_events["vad"], b.frame_events["vad"], atol=1e-5)


@pytest.mark.skipif(not TRAINED.exists(), reason="needs runs/voice_agent_frontend.afm")
def test_runtime_text_equals_streaming_session_trained():
    from audioforge.train import load_model
    m = load_model(TRAINED)
    rt = StreamingRuntime(m)
    for ex in synthetic_dataset("asr", 12, seed=2):
        s = _session(m, ex["audio"])
        r = _runtime(rt, ex["audio"])
        assert r.text == s.text
        assert np.allclose(r.frame_events["vad"], s.frame_events["vad"], atol=1e-4)


def test_int8_path_runs_and_keeps_caller_model():
    m = _random_model([16, 1])
    w = m.encoder.layers[0].ff1[1].weight.clone()
    x = (np.random.default_rng(3).standard_normal(16000) * 0.1).astype(np.float32)
    rt = _runtime(StreamingRuntime(m, quantize="int8", quantize_decoder=True).warmup(), x)
    assert len(rt.frame_events["vad"]) == len(_session(m, x).frame_events["vad"])
    assert isinstance(rt.text, str)
    assert torch.equal(m.encoder.layers[0].ff1[1].weight, w)  # deep-copied, caller untouched
    if TRAINED.exists():  # int8 is (near) transcript-neutral on the trained tiny model
        from audioforge.train import load_model
        tm = load_model(TRAINED)
        q = StreamingRuntime(tm, quantize="int8")
        data = synthetic_dataset("asr", 12, seed=2)
        same = sum(_runtime(q, ex["audio"]).text == _session(tm, ex["audio"]).text for ex in data)
        assert same >= len(data) - 2


def test_warmup_idempotent():
    m = _random_model([16, 1])
    x = (np.random.default_rng(4).standard_normal(24000) * 0.1).astype(np.float32)
    ref = _runtime(StreamingRuntime(m), x)
    rt = StreamingRuntime(m)
    rt.warmup()
    rt.warmup()
    for _ in range(2):
        out = _runtime(rt, x)
        assert out.tokens == ref.tokens and out.frame_events["vad"] == ref.frame_events["vad"]
    rt.warmup()
    assert rt.steps == 0 and rt.tokens == [] and rt.offset == 0


def test_onnx_chunk_export_matches_eager(tmp_path):
    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    pytest.importorskip("onnx")
    m = _random_model([8, 1])
    p = export_chunk_onnx(m, tmp_path / "chunk.onnx", [8, 1])
    x = (np.random.default_rng(5).standard_normal(16000 + 500) * 0.1).astype(np.float32)
    a = _runtime(StreamingRuntime(m, att_context_size=[8, 1]), x)
    b = _runtime(StreamingRuntime(m, att_context_size=[8, 1], backend="onnx", onnx_path=p, threads=1), x)
    assert np.allclose(a.frame_events["vad"], b.frame_events["vad"], atol=1e-4)
    assert a.tokens == b.tokens
