"""runs/stage1_served_v2.afm: the served checkpoint with the VAD head on block 4 only.
Runs when both real checkpoints exist: config / tensors differ in heads.vad only, the VAD reads exactly block 4, and
the server's ASR pass (ASRStream) emits the same tokens as runs/stage1_served.afm with --asr-vad-gate off."""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = ROOT / "runs/stage1_served.afm", ROOT / "runs/stage1_served_v2.afm"
pytestmark = pytest.mark.skipif(not (V1.exists() and V2.exists()), reason="served checkpoints not built")


@pytest.fixture(scope="module")
def models():
    from audioforge.train import load_model
    return load_model(V1, "cpu").eval(), load_model(V2, "cpu").eval()


def test_v2_differs_in_vad_only(models):
    a, b = models
    assert b.layer_tap["vad"] == [3] and "vad" not in b.layer_mix and a.head_cfg["vad"]["from_layers"] == "all"
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) == set(sa) - {"layer_mix.vad"}
    assert all(torch.equal(sa[k], sb[k]) for k in sb if not k.startswith("heads.vad."))
    assert sum(p.numel() for p in b.heads["vad"].parameters()) == 32897


def test_v2_stream_tokens_identical_gate_off(models):
    from audioforge.server.streams import ASRStream
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 4) / 16000
    x = (0.1 * np.sin(2 * np.pi * 220 * t) * (t % 1 < 0.6) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    out = []
    for m in models:
        s = ASRStream(m, turn=None)
        vad = []
        for i in range(0, len(x), 320):
            vad += [r["vad"] for r in s.feed_frames(x[i: i + 320])]
        vad += [r["vad"] for r in s.feed_frames(np.zeros(0, np.float32), final=True)]
        out.append((list(s.tokens), vad))
    assert out[0][0] == out[1][0]
    assert len(out[0][1]) == len(out[1][1]) > 0
    with torch.no_grad():  # the v2 VAD is the block-4 read-out of the same pass
        m = models[1]
        enc, elen, hid = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]), return_hidden=True)
        assert torch.equal(m.head_input("vad", enc, hid), hid[3])
