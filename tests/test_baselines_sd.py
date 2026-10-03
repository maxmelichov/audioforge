"""audioforge.baselines.sd: label-grid conversions, the NeMo ConvASREncoder port (TitaNet / frame-VAD MarbleNet)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audioforge.baselines import sd  # noqa: E402
from audioforge.datasets.ami import frames as ami_frames  # noqa: E402

NEMO = ROOT / "data" / "nemo"
TITANET = NEMO / "speakerverification_en_titanet_large.nemo"
MARBLENET = NEMO / "frame_vad_multilingual_marblenet_v2.0.nemo"


def test_intervals_match_ami_label_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        iv = sorted((float(s), float(s + d)) for s, d in zip(rng.uniform(0, 19, 5), rng.uniform(0.01, 2, 5)))
        np.testing.assert_array_equal(sd.intervals_to_frames(iv, 251), ami_frames(iv, 251) > 0.5)


def test_pool_probs_any_overlap():
    p = np.zeros(1000)          # 20 ms frames
    p[10] = 0.9                 # [0.20, 0.22) -> 80 ms frame 2
    p[12] = 0.4                 # [0.24, 0.26) -> frame 3
    out = sd.pool_probs(p, 0.02, 251)
    assert out[2] == pytest.approx(0.9) and out[3] == pytest.approx(0.4) and out.sum() == pytest.approx(1.3)
    # a 30 ms frame straddling two label frames marks both
    q = np.zeros(10)
    q[2] = 1.0                  # [0.06, 0.09)
    np.testing.assert_array_equal(np.flatnonzero(sd.pool_probs(q, 0.03, 4)), [0, 1])


def test_segments_to_matrix_top4_arrival_order():
    segs = [(5.0, 6.0, "B"), (1.0, 2.0, "A"), (3.0, 10.0, "C"), (0.5, 0.6, "D"), (12.0, 19.0, "E")]
    M = sd.segments_to_matrix(segs, 251)
    assert M.shape == (251, 4)
    first = [int(np.argmax(M[:, c])) for c in range(4)]
    assert first == sorted(first)                        # arrival order
    assert not any((M[:, c] == sd.intervals_to_frames([(0.5, 0.6)], 251)).all() for c in range(4))  # D dropped


def test_runs_roundtrip():
    a = np.zeros(251, bool)
    a[3:9] = a[40:41] = a[250] = True
    np.testing.assert_array_equal(sd.intervals_to_frames(sd.runs(a, 0.08), 251), a)


def test_pyannote_der_collar0_equals_frame_der():
    pytest.importorskip("pyannote.metrics")
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_sd_baselines import pooled_der, pyannote_der
    rng = np.random.default_rng(1)
    refs = [(rng.random((251, 4)) < p).astype(np.float32) for p in (0.1, 0.3, 0.05)]
    hyps = [np.roll(r, s, 0) * (rng.random(r.shape) < 0.9) for r, s in zip(refs, (1, 3, 0))]
    fd = pooled_der(hyps, refs)["der_pooled"]
    assert pyannote_der(hyps, refs, collars=(0.0,))["collar_0.0"]["der"] == pytest.approx(fd, abs=2e-4)


def _tiny_cfg(se: bool, stride: int = 1):
    blk = dict(repeat=2, kernel=[5], stride=[1], dilation=[1], residual=True, separable=True, se=se)
    return [dict(blk, filters=16, repeat=1, stride=[stride], residual=False), dict(blk, filters=16),
            dict(blk, filters=24, repeat=1, kernel=[1], residual=False)]


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_encoder_masked_batching_exact(stride):
    torch.manual_seed(0)
    enc = sd.ConvEncoder(8, _tiny_cfg(se=True, stride=stride)).eval()
    for m in enc.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2)
    x = torch.randn(2, 8, 50)
    lens = torch.tensor([50, 31])
    y, yl = enc(x, lens)
    y1, yl1 = enc(x[1:, :, :31], lens[1:])
    assert int(yl[1]) == int(yl1[0]) == y1.shape[-1]
    torch.testing.assert_close(y[1, :, : int(yl[1])], y1[0], atol=1e-5, rtol=1e-5)


def test_state_dict_keys_follow_nemo_layout():
    enc = sd.ConvEncoder(8, _tiny_cfg(se=True))
    keys = set(enc.state_dict())
    assert {"encoder.1.mconv.0.conv.weight", "encoder.1.mconv.5.conv.weight", "encoder.1.mconv.7.running_var",
            "encoder.1.mconv.8.fc.0.weight", "encoder.1.res.0.0.conv.weight", "encoder.1.res.0.1.bias"} <= keys


@pytest.mark.skipif(not MARBLENET.exists(), reason="MarbleNet .nemo not downloaded")
def test_marblenet_strict_load():
    m = sd.load_nemo_conv(MARBLENET)
    assert isinstance(m, sd.FrameVAD) and m.import_info["params"] == 91378
    assert m.import_info["fb_max_abs_diff"] < 1e-6
    t = np.arange(32000) / 16000
    x = np.concatenate([np.zeros(16000), 0.3 * np.sin(2 * np.pi * 220 * t[:16000])]).astype(np.float32)
    p = m.probs(x)
    assert p.shape == (101,) and ((p >= 0) & (p <= 1)).all()


@pytest.mark.skipif(not TITANET.exists(), reason="TitaNet .nemo not downloaded")
def test_titanet_strict_load_and_card_pair():
    torch.set_num_threads(2)
    from audioforge.nemo_import import import_titanet
    m = import_titanet(TITANET)
    assert m.import_info["n_loaded"] == m.import_info["n_tensors"] - 2
    rng = np.random.default_rng(0)
    a = [rng.standard_normal(16000).astype(np.float32) * 0.1, rng.standard_normal(24000).astype(np.float32) * 0.1]
    e1, e2 = m.embed(a), m.embed(a, batch_size=2)
    assert e1.shape == (2, 192)
    torch.testing.assert_close(e1, e2, atol=1e-5, rtol=1e-4)
    try:
        import soundfile
        from huggingface_hub import hf_hub_download
        wavs = [soundfile.read(hf_hub_download("nvidia/speakerverification_en_titanet_large", f, local_files_only=True))[0]
                for f in ("an255-fash-b.wav", "cen7-fash-b.wav")]
    except (OSError, ImportError):  # LocalEntryNotFoundError is an OSError
        pytest.skip("card example wavs not cached")
    e = m.embed(wavs)
    assert float(e[0] @ e[1]) > 0.7   # same speaker ("fash"); NeMo's verify_speakers threshold is 0.7
