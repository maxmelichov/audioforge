"""scripts/research/contamination_probe.py (research/CONTAMINATION.md): a transient wrong speaker binding, corrected at t0.

Under test (tiny hermetic models, no checkpoints):
  - the corruptions touch only the window [lo, hi) of the bound track / column (zero, swap, rand, eps, noop);
  - head_run (the TurnHead v3 forward with an explicit GRU / counter state) chained at any frame equals the unsplit
    run, so the state-substitution protocols are exact stream continuations;
  - on a causal speaker-conditioned encoder the frames before the (chunk-aligned) window are bit-identical to the
    baseline and the frames after the correction still differ (the effect the probe measures), and the 'both'
    protocol (baseline encoder frames + fresh state) gives a zero difference;
  - greedy_segmented equals heads.turn.greedy_decode_frames without hooks, keeps the pre-t0 prefix under a decoder
    reset, and is unchanged when the substituted frames are the same frames;
  - sa_examples reproduces the recipe's held-out mixtures (synthetic_dataset('speaker_attributed', seed=1)).
"""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.data import synthetic_dataset
from audioforge.heads.turn import decoded_text_state, greedy_decode_frames

sys.path.insert(0, str(Path(__file__).parent))
from test_turn_v3 import V3, _model  # noqa: E402

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def cp():
    spec = importlib.util.spec_from_file_location("contamination_probe", ROOT / "scripts" / "research" / "contamination_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model, batch = _model(turn=V3)
    return model.eval(), batch


# --------------------------------------------------------------------------- corruption
def test_corrupt_track_touches_only_the_window(cp):
    rng = random.Random(0)
    T, S = 30, 4
    cols = np.random.default_rng(1).random((T, S)).astype(np.float32)
    cols[:, 0] = 1.0  # the bound column speaks throughout
    cols[10:20, 2] = 0.9  # column 2 is the loudest other column on the window
    act = cols[:, 0].copy()
    lo, hi = 12, 18
    for mode in ("zero", "swap", "rand", "eps", "noop"):
        a, c, info = cp.corrupt_track(act, cols, lo, hi, mode, rng)
        outside = np.r_[0:lo, hi:T]
        assert np.array_equal(a[outside], act[outside]) and np.array_equal(c[outside], cols[outside]), mode
        assert np.array_equal(a, c[:, 0]), "act is the bound column"
        assert info["lo"] == lo and info["hi"] == hi
        if mode == "zero":
            assert np.all(a[lo:hi] == 0) and np.all(c[lo:hi, 1:] == cols[lo:hi, 1:])
            assert info["input_delta"] == pytest.approx(1.0)
        elif mode == "swap":
            assert info["partner"] == 2
            assert np.array_equal(c[lo:hi, 0], cols[lo:hi, 2]) and np.array_equal(c[lo:hi, 2], cols[lo:hi, 0])
            assert info["differs_from_zero"]
        elif mode == "rand":
            j = info["partner"]
            assert j in (1, 2, 3)
            assert np.array_equal(c[lo:hi, 0], cols[lo:hi, j]) and np.array_equal(c[lo:hi, j], cols[lo:hi, 0])
        elif mode == "eps":
            assert np.abs(a[lo:hi] - act[lo:hi]).max() <= 1e-4 and info["input_delta"] > 0
        else:
            assert np.array_equal(a, act) and np.array_equal(c, cols) and info["input_delta"] == 0
    # saasr form: no columns, the other speaker's activity
    other = np.zeros(T, np.float32)
    other[lo:hi] = 1.0
    a, c, info = cp.corrupt_track(act, None, lo, hi, "swap", rng, other=other)
    assert c is None and np.all(a[lo:hi] == 1.0) and info["input_delta"] == 0.0 and info["differs_from_zero"]
    a, _, info = cp.corrupt_track(act, None, lo, hi, "zero", rng, other=other)
    assert np.all(a[lo:hi] == 0) and not info["differs_from_zero"]
    with pytest.raises(ValueError):
        cp.corrupt_track(act, cols, lo, hi, "bogus", rng)


def test_bin_curve_and_logit(cp):
    d = np.arange(10, dtype=np.float64)
    c = cp.bin_curve(d, 2, n_bins=6, size=2)
    assert np.allclose(c[:4], [2.5, 4.5, 6.5, 8.5]) and np.isnan(c[4:]).all()
    p = np.array([0.5, 0.998164, 1e-9])
    z = cp.logit(p)
    assert z[0] == pytest.approx(0.0) and z[1] > 6 and np.isfinite(z).all()


# --------------------------------------------------------------------------- turn head: segmented runs
def _turn_inputs(cp, model, batch, act_np, cols_np):
    head = model.heads["turn"]
    audio, alen = batch["audio"][:1], batch["audio_len"][:1]
    T = act_np.shape[1]
    A, Cc = torch.as_tensor(act_np), torch.as_tensor(cols_np)
    P = torch.zeros(A.shape[0], dtype=torch.long)
    with torch.no_grad():
        enc0, elen, hidden = model.encode(audio, alen, return_hidden=True)
        y, n = decoded_text_state(model, head, audio, alen, enc0, hidden, elen, A[:1, : enc0.shape[1]])
        tf = head.text_frames(y, n)[:, :T].expand(A.shape[0], -1, -1)
        enc, _ = model.encode(audio.expand(A.shape[0], -1), alen.expand(A.shape[0]), spk_act=A)
    return head, enc[:, :T], tf, A, Cc, P


def test_head_run_chain_equals_unsplit(cp, tiny):
    model, batch = tiny
    T = min(int(batch["audio_len"][0]) // 1280, 24)
    rng = np.random.default_rng(0)
    cols = (rng.random((1, T, 4)) > 0.5).astype(np.float32)
    act = cols[:, :, 0].copy()
    head, enc, tf, A, Cc, P = _turn_inputs(cp, model, batch, act, cols)
    with torch.no_grad():
        p, h, dur = cp.head_run(head, enc, tf, A, Cc, P)
        for k in (1, 7, T - 1):
            pa, ha, da = cp.head_run(head, enc[:, :k], tf[:, :k], A[:, :k], Cc[:, :k], P)
            pb, _, _ = cp.head_run(head, enc[:, k:], tf[:, k:], A[:, k:], Cc[:, k:], P, h0=ha[:, -1][None].contiguous(), dur0=da)
            assert torch.allclose(torch.cat([pa, pb], 1), p, atol=1e-6), k
        assert all(torch.equal(dur[k][0], v[0]) and torch.equal(dur[k][1], v[1])
                   for k, v in head.v3_features(T, A, Cc, P, None)[1].items())


def test_corruption_is_causal_and_persists(cp, tiny):
    model, batch = tiny
    assert model.encoder.causal
    cs = model.encoder.att_context_size[1] + 1
    T = min(int(batch["audio_len"][0]) // 1280, 24)
    cols = np.zeros((1, T, 4), np.float32)
    cols[:, :, 0] = 1.0
    act = cols[:, :, 0].copy()
    lo, hi = 8, 12
    az, cz, _ = cp.corrupt_track(act[0], cols[0], lo, hi, "zero", random.Random(0))
    head, enc, tf, A, Cc, P = _turn_inputs(cp, model, batch, np.stack([act[0], az]), np.stack([cols[0], cz]))
    with torch.no_grad():
        p, h, _ = cp.head_run(head, enc, tf, A, Cc, P)
        d = (p[1] - p[0]).abs()
        lo_al = lo - lo % cs
        assert d[:lo_al].max() < 1e-6, "frames before the (chunk-aligned) window must not change"
        assert d[hi:].max() > 1e-4, "the perturbed-kernel model must still differ after the correction"
        # 'both' protocol: baseline encoder frames + fresh state -> identical to the baseline under the same protocol
        seg = dict(tf=tf[:, hi:], act=A[:, hi:], cols=Cc[:, hi:], prim=P)
        q, _, _ = cp.head_run(head, enc[:1, hi:].expand(2, -1, -1), h0=None, dur0=None, **seg)
        assert torch.equal(q[0], q[1])
        # enc_sub: baseline frames after hi, the corrupted GRU state -> a difference that comes from the state only
        h0 = h[:, hi - 1][None].contiguous()
        _, dur0 = head.v3_features(hi, A[:, :hi], Cc[:, :hi], P, None)
        q, _, _ = cp.head_run(head, enc[:1, hi:].expand(2, -1, -1), h0=h0, dur0=dur0, **seg)
        assert (q[1] - q[0]).abs().max() > 0
    hd, hy = cp.outcome(p[0].numpy(), act[0], 2, T - 4, cp._ev())
    assert set(hd) == {"fc", "lat_ms"} and set(hy) == {"fc", "lat_ms"}


# --------------------------------------------------------------------------- RNNT: segmented greedy
def test_greedy_segmented_matches_reference(cp, tiny):
    model, _ = tiny
    asr = model.heads["turn"]._asr
    torch.manual_seed(1)
    f = torch.randn(20, model.encoder.d_model) * 3
    ref = greedy_decode_frames(asr, f)
    assert cp.greedy_segmented(asr, f) == ref
    assert cp.greedy_segmented(asr, f, t0=9, reset=False, f_after=f) == ref
    toks, frs = cp.greedy_segmented(asr, f, t0=9, reset=True)
    pre = [k for k, t in zip(*ref) if t < 9]
    assert [k for k, t in zip(toks, frs) if t < 9] == pre
    toks2, frs2 = cp.greedy_segmented(asr, f, t0=9, reset=False, f_after=torch.zeros_like(f))
    assert [k for k, t in zip(toks2, frs2) if t < 9] == pre
    assert cp.edit_distance("abc", "abd") == 1 and cp.wer("a b c", "a b") == pytest.approx(0.5)
    assert cp.flips("red note", "red", "note go") == 1
    assert cp.token_bins([1, 2, 3], [4, 5, 9], 4, 3, size=2) == [[1, 2], [], [3]]


def test_sa_examples_match_recipe_val(cp):
    ex = cp.sa_examples(6)
    ref = synthetic_dataset("speaker_attributed", 6, seed=1)
    for a, b in zip(ex, ref):
        assert np.array_equal(a["audio"], b["audio"]) and np.array_equal(a["spk_targets"], b["spk_targets"])
        assert b["text"] in [s["text"] for s in a["speakers"]] and len(a["speakers"]) == 2
    conds = cp.turn_conditions()
    assert conds[0][0] == "base" and len(conds) == 1 + 3 * 3 + 2
    assert len(cp.saasr_conditions()) == 1 + 3 * 3 + 1
