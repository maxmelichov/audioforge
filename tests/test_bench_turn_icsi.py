"""scripts/research/bench_turn_icsi.py: the frozen-operating-point scoring agrees with eval_stage1's cross-fit / in-sample
selection on synthetic score tracks (no data or models needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("bench_turn_icsi", ROOT / "scripts" / "research" / "bench_turn_icsi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _convs(n=40, T=120, seed=0):
    rng = np.random.default_rng(seed)
    convs, head, sil = [], [], []
    for i in range(n):
        on, en = 10, int(rng.integers(40, 80))
        hes = np.zeros(T, np.float32)
        if i % 3 == 0:
            hes[on + 5: on + 9] = 1
        p = rng.uniform(0, 0.9, T)
        p[en:] = np.minimum(1, p[en:] + rng.uniform(0, 0.4, T - en))  # the head rises after the end
        act = np.zeros(T, np.float32)
        act[on:en] = 1
        act[en + int(rng.integers(0, 6)):] = 0
        convs.append(dict(onset_frame=on, turn_end_frame=en, hes=hes, spk_act=act, meeting=f"M{i % 4}",
                          spk_targets=np.stack([act, np.zeros(T, np.float32)], 1)))
        head.append(p.astype(np.float32))
        sil.append(np.arange(T, dtype=np.float32) * (np.arange(T) >= en + int(rng.integers(0, 10))))
    return convs, head, sil


def test_fixed_point_reproduces_the_selection(M):
    import eval_stage1 as E
    convs, head, sil = _convs()
    on, en, pauses = M._prep(convs)
    folds = np.array([0 if v["meeting"] in ("M0", "M1") else 1 for v in convs])
    single = (head, lambda t: t + 1)
    hybrid = (head, lambda t: t + 1, sil, lambda t: t + 1)
    for spec, hyb in ((single, False), (hybrid, True)):
        pts = M.fit_points(spec, hyb, convs, 75, folds)
        oc, ths = M.sweep_outcomes(spec, hyb, on, en, pauses, 75)
        # the per-fold points are exactly v2_crossfit's thresholds
        _, _, _, th = E.v2_crossfit(oc, folds, M.MAX_FC, "turn", tie_miss=hyb)
        for f in (0, 1):  # v2_crossfit reports the (rounded) threshold, or the grid column of a hybrid
            assert (ths[int(round(th[f]))] if hyb else round(pts[f"fold{f}"], 5)) == (pts[f"fold{f}"] if hyb else th[f])
        # applying a point reproduces the sweep's column at that point
        j = E._select(oc, np.ones(len(convs), bool), M.MAX_FC, "turn", tie_miss=hyb)
        fc, lat, pf, npause = M.outcomes_at(spec, hyb, on, en, pauses, 75, pts["all"])
        assert np.array_equal(fc, oc["fc"][:, j]) and np.array_equal(lat, oc["lat"][:, j])
        assert np.array_equal(pf, oc["pf"][:, j]) and fc.mean() <= M.MAX_FC


def test_fmt_point_frames_convention(M):
    assert M.fmt_point("silero_timeout", 22.6) == {"timeout_threshold": 22.6, "k_frames": 23, "k_ms": 1840.0}
    assert M.fmt_point(M.REF, (0.998, 51.0))["k_frames"] == 52  # score > 51 frames of silence = 52 observed
    assert M.fmt_point(M.CANDIDATE, (0.998, np.inf))["k_frames"] is None


def test_silero_timeout_track_matches_bench_turn_baselines(M):
    from audioforge.baselines import turn as B
    p = np.array([0.9] * 20 + [0.1] * 40 + [0.9] * 10 + [0.1] * 30, np.float32)
    ref = B.silence_frames(B.speech_chunks_pipecat(B.pipecat_vad(p)), 30)
    assert np.array_equal(M.silero_timeout_track(p, 30), ref)


def test_icsi_folds_partition_the_held_out_meetings(M):
    from audioforge.datasets.icsi import DEFAULT_MEETINGS, SPLITS
    held = set(SPLITS["dev"]) | set(SPLITS["eval"])
    assert set(M.ICSI_MEETINGS) == held and set(sum(map(list, M.ICSI_FOLDS), [])) == held
    assert not set(M.ICSI_FOLDS[0]) & set(M.ICSI_FOLDS[1])
    assert not held & set(DEFAULT_MEETINGS["train"])  # nothing the trail6 head or any table was fitted on
