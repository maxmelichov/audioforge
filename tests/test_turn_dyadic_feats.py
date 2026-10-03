"""Dyadic training additions: the causal log-RMS energy input, the multi-horizon
user-activity auxiliary targets, and the corrupted-then-corrected (rebind) conditioning windows.

Causality, shapes, streaming == offline, warm-start identity (zero-initialised projections), label semantics."""
import sys
from pathlib import Path

import pytest
import torch

from audioforge.heads.turn import (
    FRAME_SAMPLES,
    TurnHead,
    causal_standardize,
    energy_features,
    frame_log_rms,
    horizon_edges,
    multi_horizon_targets,
)
from audioforge.model import rebind_cfg, rebind_corruption

sys.path.insert(0, str(Path(__file__).parent))


def _audio(B=2, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    amp = torch.rand(B, T, 1, generator=g) * 0.5 + 0.01  # a different loudness per frame
    return (torch.randn(B, T, FRAME_SAMPLES, generator=g) * amp).reshape(B, -1)


# --------------------------------------------------------------------------- energy
def test_energy_shapes_and_frame_grid():
    x = _audio(2, 30)
    e, st = energy_features(x, None, 30)
    assert e.shape == (2, 30) and all(t.shape == (2,) for t in st)
    lr, ok = frame_log_rms(x, torch.tensor([30 * FRAME_SAMPLES, 10 * FRAME_SAMPLES + 5]), 32)
    assert lr.shape == (2, 32) and ok.shape == (2, 32)
    assert ok[0, :30].all() and not ok[0, 30:].any()
    assert ok[1, :11].all() and not ok[1, 11:].any()  # a frame with >= 1 valid sample counts
    ref = torch.log(x[0, :FRAME_SAMPLES].pow(2).mean().sqrt())
    assert torch.allclose(lr[0, 0], ref, atol=1e-5)
    e2, _ = energy_features(x, torch.tensor([30 * FRAME_SAMPLES, 10 * FRAME_SAMPLES]), 30)
    assert (e2[1, 10:] == 0).all()  # frames past the audio are 0


def test_energy_is_causal():
    x = _audio(1, 40, seed=1)
    e, _ = energy_features(x, None, 40)
    for t in (0, 5, 17, 39):
        y = x.clone()
        y[:, (t + 1) * FRAME_SAMPLES:] = torch.randn_like(y[:, (t + 1) * FRAME_SAMPLES:]) * 3
        f, _ = energy_features(y, None, 40)
        assert torch.equal(e[:, : t + 1], f[:, : t + 1]), t


def test_energy_streaming_equals_offline():
    x = _audio(2, 50, seed=2)
    off, _ = energy_features(x, None, 50)
    st, parts = None, []
    for a, b in ((0, 7), (7, 8), (8, 30), (30, 50)):
        p, st = energy_features(x[:, a * FRAME_SAMPLES: b * FRAME_SAMPLES], None, b - a, state=st)
        parts.append(p)
    assert torch.allclose(torch.cat(parts, 1), off, atol=1e-6)


def test_energy_standardisation_is_gain_invariant():
    x = _audio(1, 60, seed=3)
    e1, _ = energy_features(x, None, 60)
    e2, _ = energy_features(4.0 * x, None, 60)  # +log 4 on every frame: the running mean absorbs it
    assert torch.allclose(e1, e2, atol=1e-4)
    assert e1.abs().max() <= 2.0 + 1e-6  # clip 4, halved


def test_causal_standardize_ignores_invalid_frames():
    x = torch.randn(1, 20)
    valid = torch.ones(1, 20, dtype=torch.bool)
    valid[0, 5:9] = False
    z, (m, v, k) = causal_standardize(x, valid)
    assert (z[0, 5:9] == 0).all() and int(k) == 16


# --------------------------------------------------------------------------- multi-horizon targets
def test_horizon_edges():
    assert horizon_edges([200, 400, 600, 1000]) == [3, 5, 8, 13]
    with pytest.raises(AssertionError):
        horizon_edges([200, 220])  # both round to 3 frames


def test_multi_horizon_targets_bins_and_validity():
    T = 30
    act = torch.zeros(1, T)
    act[0, 10] = 1  # one active frame at 10
    tg, ok = multi_horizon_targets(act, torch.tensor([T]), [3, 5, 8, 13])
    assert tg.shape == (1, T, 4) and ok.shape == (1, T, 4)
    # bin j of frame t covers t + e_{j-1} + 1 .. t + e_j
    for t in range(T):
        for j, (lo, hi) in enumerate(((1, 3), (4, 5), (6, 8), (9, 13))):
            assert tg[0, t, j] == float(t + lo <= 10 <= t + hi), (t, j)
    # the OR of the bins is "active within 13 frames" (future_act_aux's cumulative window at h = 13)
    from audioforge.heads.turn import future_act_targets
    cum, _ = future_act_targets(act, torch.tensor([T]), [13])
    assert torch.equal(tg.max(-1).values, cum[..., 0])
    # validity: the bin must end inside the labelled frames unless a positive was already seen
    assert not ok[0, T - 1].any() and ok[0, T - 14].all()
    tg2, ok2 = multi_horizon_targets(act, torch.tensor([12]), [3, 5, 8, 13])
    assert ok2[0, 12:].sum() == 0  # frames past the length are never labelled
    assert ok2[0, 7, 0] and tg2[0, 7, 0] == 1  # frame 7's first bin (8..10) holds the positive
    assert not ok2[0, 7, 3]  # its last bin (16..20) ends beyond the labels


def test_multi_horizon_targets_look_only_forward():
    act = (torch.rand(2, 40) > 0.5).float()
    tg, _ = multi_horizon_targets(act, None, [3, 5, 8, 13])
    for t in (0, 10, 25):
        a2 = act.clone()
        a2[:, : t + 1] = 1 - a2[:, : t + 1]  # the past / present frames do not enter frame t's targets
        tg2, _ = multi_horizon_targets(a2, None, [3, 5, 8, 13])
        assert torch.equal(tg[:, t], tg2[:, t])


# --------------------------------------------------------------------------- head
def _head(**kw):
    torch.manual_seed(0)
    return TurnHead(16, mode="none", hidden=8, **kw).eval()


def test_zero_init_energy_head_matches_source():
    src = _head()
    new = _head(energy_input=True, multi_horizon_aux={"horizons_ms": [200, 400, 600, 1000], "weight": 0.3})
    missing, unexpected = new.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"energy_in.weight", "mh.weight", "mh.bias"}
    enc, x = torch.randn(2, 25, 16), _audio(2, 25)
    en = new.energy_of(x, None, 25)
    assert en.shape == (2, 25)
    assert torch.allclose(src(enc, None), new(enc, None, energy=en), atol=1e-6)
    assert new.decode(enc, None) is None  # energy head without energy: missing input


def test_energy_head_causal_and_streaming_equals_offline():
    h = _head(energy_input=True)
    with torch.no_grad():
        h.energy_in.weight.normal_(0, 1.0)
    T = 20
    enc, x = torch.randn(1, T, 16), _audio(1, T, seed=5)
    off = h.decode(enc, None, energy=h.energy_of(x, None, T))
    st = h.init_stream()
    outs = [h.step(enc[:, t: t + 1], state=st, audio=x[:, t * FRAME_SAMPLES: (t + 1) * FRAME_SAMPLES]) for t in range(T)]
    assert torch.allclose(torch.cat(outs, 1), off, atol=1e-5)
    y = x.clone()
    y[:, 12 * FRAME_SAMPLES:] *= 5
    alt = h.decode(enc, None, energy=h.energy_of(y, None, T))
    assert torch.allclose(off[:, :12], alt[:, :12], atol=1e-6) and not torch.allclose(off[:, 12:], alt[:, 12:])


def test_multi_horizon_loss_trains_and_old_aux_weight0_skipped():
    h = TurnHead(16, mode="none", hidden=8, future_act_aux={"horizons": [6, 12, 25], "weight": 0.0},
                 multi_horizon_aux={"horizons_ms": [200, 400, 600, 1000], "weight": 0.3}).train()
    T = 30
    act = torch.zeros(2, T)
    act[:, 3:15] = 1
    batch = {"spk_act": act, "audio": _audio(2, T)}
    enc = torch.randn(2, T, 16, requires_grad=True)
    called = []
    h.future_loss = lambda *a: called.append(1) or torch.tensor(0.0)
    l = h.loss(enc, torch.tensor([T, T]), batch)
    l.backward()
    assert h.mh.weight.grad is not None and h.mh.weight.grad.abs().sum() > 0 and not called


# --------------------------------------------------------------------------- rebind (corrupted-then-corrected)
def test_rebind_window_is_corrected_before_the_turn_end():
    B, T, S = 64, 60, 4
    g = torch.Generator().manual_seed(0)
    oracle = torch.zeros(B, T)
    oracle[:, 10:40] = 1
    cols = torch.zeros(B, T, S)
    cols[:, 10:40, 0] = 1
    cols[:, 38:55, 1] = 1  # the other party
    prim = torch.zeros(B, dtype=torch.long)
    out, wins = rebind_corruption(cols, prim, oracle, torch.full((B,), T), 1.0, 12, 50, gen=g)
    assert all(w is not None for w in wins)
    for b, (s0, e0) in enumerate(wins):
        assert 11 <= e0 <= 40 and 0 <= s0 < e0 and e0 - s0 <= 50
        assert torch.equal(out[b, e0:], cols[b, e0:]) and torch.equal(out[b, :s0], cols[b, :s0])
        assert torch.equal(out[b, s0:e0, 0], cols[b, s0:e0, 1]) and torch.equal(out[b, s0:e0, 1], cols[b, s0:e0, 0])
    none, w0 = rebind_corruption(cols, prim, oracle, torch.full((B,), T), 0.0)
    assert torch.equal(none, cols) and all(w is None for w in w0)
    assert rebind_cfg(0.1) == {"p": 0.1, "w_min": 12, "w_max": 50}
    with pytest.raises(ValueError):
        rebind_cfg({"q": 1})
