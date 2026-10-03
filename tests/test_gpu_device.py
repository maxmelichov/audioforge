"""serve --device mps / cuda (CUDA: PR #1): the streaming engine with the
models on a GPU emits the same turn_ends and finals as the CPU engine on the bundled two-party clip, in single-model
mode with the stored voice print and with the print taken live after agent_end. Needs that GPU and the downloaded
models (audioforge-download); skipped otherwise. A GPU the process cannot see falls back to cpu
(test_bulletproof.test_device_fallback_and_startup_checks)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
PRINT = ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json"
GPUS = [pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"))]


def _load(device):
    import audioforge
    from audioforge import hub
    if hub.find_model("asr") is None or hub.find_model("tsvad") is None:
        pytest.skip("needs audioforge-download (asr + tsvad)")
    return audioforge.load(device=device)


def _decisions(fe, enroll):
    import soundfile as sf
    pcm, sr = sf.read(str(CLIP), dtype="int16")
    s = fe.session(sample_rate=sr)
    if enroll:
        s.enroll(json.loads(PRINT.read_text()))
    else:
        s.agent_end()
    out = []
    for i in range(0, len(pcm), sr // 50):
        out += s.feed(pcm[i:i + sr // 50])
    out += s.end()
    return [(m["type"], m["t"], m.get("speaker"), m.get("text"), m.get("policy")) for m in out
            if m["type"] in ("turn_end", "final", "voiceprint")]


@pytest.mark.real
@pytest.mark.parametrize("device", GPUS)
def test_gpu_engine_matches_cpu_on_the_bundled_clip(device):
    cpu, gpu = _load("cpu"), _load(device)
    assert next(gpu.engine.asr.parameters()).device.type == device
    assert next(gpu.engine.tsvad.parameters()).device.type == device and not gpu.engine.fast
    for enroll in (True, False):
        ref = _decisions(cpu, enroll)
        assert any(m[0] == "turn_end" for m in ref)
        assert _decisions(gpu, enroll) == ref


@pytest.mark.parametrize("device", GPUS)
def test_column_embedder_runs_on_the_heads_device(device):
    """enrollment.ColumnEmbedder takes host arrays, embeds on the speaker head's device, returns host float32."""
    from audioforge.enrollment import ColumnEmbedder
    from audioforge.heads.audio import SpeakerHead
    torch.manual_seed(0)
    head = SpeakerHead(64, num_speakers=4, emb_dim=16).eval()
    x, valid = np.random.RandomState(0).randn(2, 5, 64).astype(np.float32), np.ones((2, 5), bool)
    ref = ColumnEmbedder(head)(x, valid)
    out = ColumnEmbedder(head.to(device))(x, valid)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32 and np.abs(out - ref).max() < 1e-4
