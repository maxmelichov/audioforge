"""scripts/research/vad_layers.py: encoder truncation, gated streaming with re-prime, ROC helpers (research/VAD_LAYERS.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vad_layers import DeepRunner, GatedEncoder, gate_from_label, roc_stats, truncated_encoder  # noqa: E402

from audioforge.modules.fastconformer import FastConformerEncoder, StreamState  # noqa: E402

ATT = [8, 1]


def _enc(**kw):
    torch.manual_seed(0)
    # the served encoder's streaming flavour: causal, NeMo subsampling padding, rel_pos attention, xscaling
    return FastConformerEncoder(feat_in=40, d_model=64, n_layers=4, n_heads=4, subsampling_channels=32, dropout=0.0,
                                causal=True, att_context_size=ATT, pos_emb="rel_pos", xscaling=True,
                                subsampling_activation="relu", subsampling_padding="nemo", **kw).eval()


def _stream(enc, mel, att=ATT):
    cs = (att[1] + 1) * enc.subsampling_factor
    state, outs, hids = StreamState(), [], []
    for c in range(mel.shape[-1] // cs):
        o, hid, state = enc.stream_step(mel[..., c * cs:(c + 1) * cs], state, att, final=False, return_hidden=True)
        outs.append(o)
        hids.append(hid)
    return torch.cat(outs, 1), [torch.cat([h[i] for h in hids], 1) for i in range(len(enc.layers))]


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_truncated_encoder_equals_full_pass_prefix(k):
    enc = _enc()
    mel = torch.randn(1, 40, 30 * 16)
    full, hidden = _stream(enc, mel)
    sub = truncated_encoder(enc, k)
    assert len(sub.layers) == k and len(enc.layers) == 4  # the original is untouched
    assert all(a is b for a, b in zip(sub.layers, enc.layers))  # shared weights, not copies
    out, _ = _stream(sub, mel)
    ref = hidden[k - 1] if k else enc.pre_encode(mel, torch.tensor([mel.shape[-1]]))[0][:, : out.shape[1]] * enc.xscale
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    # offline forward of the view agrees with the streaming view
    off, _ = sub(mel, torch.tensor([mel.shape[-1]]), ATT)
    assert torch.allclose(off[:, : out.shape[1]], out, atol=1e-4)


def test_gated_encoder_never_gated_equals_ungated():
    enc = _enc()
    mel = torch.randn(1, 40, 24 * 16)
    full, hidden = _stream(enc, mel)
    g = GatedEncoder(enc, 2, ATT, prime=8)
    outs, hids = [], []
    for c in range(24):
        o, hid, sec = g.step(mel[..., c * 16:(c + 1) * 16], gate=False, final=False)
        outs.append(o)
        hids.append(hid)
    out = torch.cat(outs, 1)
    assert torch.allclose(out, full, atol=1e-5), (out - full).abs().max()
    for i in range(4):
        assert torch.allclose(torch.cat([h[i] for h in hids], 1), hidden[i], atol=1e-5)


def test_reprime_with_full_history_is_exact_and_short_history_is_not():
    enc = _enc()
    n = 40
    mel = torch.randn(1, 40, n * 16)
    full, hidden = _stream(enc, mel)
    gate = np.zeros(n, bool)
    gate[10:22] = True  # a gated stretch, then speech resumes at chunk 22
    k, cs = 2, 2

    def run(prime):
        r = DeepRunner(enc, k, ATT, prime)
        errs, secs = [], []
        for c in range(n):
            top, dh, sec = r.step(hidden[k - 1][:, c * cs:(c + 1) * cs], bool(gate[c]))
            if top is None:
                assert gate[c]
                continue
            assert len(dh) == len(enc.layers) - k
            errs.append(float((top - full[:, c * cs:(c + 1) * cs]).abs().max()))
            if sec:
                secs.append(sec)
        assert len(secs) == (1 if prime else 0)  # one re-prime at the resume (chunk 22); the cold start is not one
        return errs

    e_full = run(prime=200)  # more history than exists: the deep blocks recompute everything -> exact
    assert max(e_full) < 1e-4, max(e_full)
    e_left = run(prime=8)  # the attention left context only: the re-primed frames lack their own context
    e_none = run(prime=0)  # no re-prime: the resume chunk attends to nothing -> the largest error
    assert max(e_full) < max(e_left) < max(e_none), (max(e_full), max(e_left), max(e_none))
    assert e_none[0] < 1e-6 and e_left[0] < 1e-6  # before the gate everything is identical


def test_roc_stats_and_gate():
    y = np.r_[np.ones(50), np.zeros(50)].astype(bool)
    perfect = roc_stats(np.r_[np.linspace(0.6, 1, 50), np.linspace(0, 0.4, 50)], y)
    assert perfect["auc"] == 1.0 and perfect["miss_at_fpr0.075"] == 0.0
    inverted = roc_stats(np.r_[np.zeros(50), np.ones(50)], y)
    assert inverted["auc"] == 0.0 and inverted["miss_at_fpr0.075"] == 1.0
    # half the positives score above every negative, the other half below: the ROC is flat at TPR 0.5 between
    # FPR 0 and 1, so the interpolated miss at FPR 0.075 is exactly 0.5 and the AUC 0.5
    s = np.r_[np.ones(25) * 0.9, np.ones(25) * 0.3, np.ones(50) * 0.5]
    r = roc_stats(s, y)
    assert abs(r["miss_at_fpr0.075"] - 0.5) < 1e-6 and abs(r["auc"] - 0.5) < 1e-6
    lab = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], np.float32)
    np.testing.assert_array_equal(gate_from_label(lab, 2, 0), [False, True, True, True, False, True])
    np.testing.assert_array_equal(gate_from_label(lab, 2, 1), [False, False, True, True, False, False])


def test_batched_reprime_equals_chunked_reprime():
    enc = _enc()
    n = 40
    mel = torch.randn(1, 40, n * 16)
    full, hidden = _stream(enc, mel)
    gate = np.zeros(n, bool)
    gate[10:22] = True
    outs = {}
    for batched in (False, True):
        r = DeepRunner(enc, 2, ATT, prime=16, batched=batched)
        outs[batched] = torch.cat([t for t in (r.step(hidden[1][:, c * 2:(c + 1) * 2], bool(gate[c]))[0]
                                              for c in range(n)) if t is not None], 1)
    assert torch.allclose(outs[True], outs[False], atol=1e-5), (outs[True] - outs[False]).abs().max()
