"""serve --device cuda (research/GPU_RUN_2026-09-29.md): the streaming engine with the models on a GPU emits the same
turn_ends and finals as the CPU engine on the bundled two-party clip, in single-model mode with a stored print and with
the print taken after agent_end. Needs a CUDA GPU and the downloaded models (audioforge-download); skipped otherwise.
Without a GPU, --device cuda still falls back to cpu (test_bulletproof.test_device_fallback_and_startup_checks)."""
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
PRINT = ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json"


def _load(device):
    import audioforge
    from audioforge import hub
    if hub.find_model("asr") is None or hub.find_model("tsvad") is None:
        pytest.skip("needs audioforge-download (asr + tsvad)")
    return audioforge.load(device=device)


def _decisions(fe, enroll):
    import soundfile as sf
    pcm, sr = sf.read(str(CLIP), dtype="int16")
    s = fe.session(turn_policy="hybrid_dyn", sample_rate=sr)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_engine_matches_cpu_on_the_bundled_clip():
    cpu, gpu = _load("cpu"), _load("cuda")
    assert next(gpu.engine.asr.parameters()).is_cuda
    for enroll in (True, False):
        assert _decisions(gpu, enroll) == _decisions(cpu, enroll)
