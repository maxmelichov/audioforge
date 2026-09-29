"""scripts/research/bench_turn_icsi_dyn.py: the dynamic-timeout grid scoring, the frozen / cross-fitted points and the
whole-window event statistics agree with eval_stage1 / bench_completeness_dyn on synthetic rows (no data or models)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("bench_turn_icsi_dyn", ROOT / "scripts" / "research" / "bench_turn_icsi_dyn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(n=48, T=140, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        on, en = 10, int(rng.integers(50, 90))
        p = rng.uniform(0, 0.6, T)
        p[en:] = np.minimum(1, p[en:] + rng.uniform(0.2, 0.45, T - en))  # the head rises after the end
        other = np.zeros(T, bool)
        if i % 4 == 0:
            other[en + 12:] = True  # taken end
        sil_start = en + int(rng.integers(0, 6))
        sil = np.maximum(0, np.arange(T) - sil_start).astype(np.float64)
        if i % 5 == 0:  # a within-turn pause with its own silence run
            sil[on + 15: on + 25] = np.arange(10)
        pauses = [(on + 15, on + 25)] if i % 5 == 0 else []
        rows.append(dict(key=f"k{i}", meeting=f"M{i % 4}", T=T, onset=on, end=en, stratum="taken" if i % 4 == 0 else "open",
                         fold=0 if i % 4 < 2 else 1, pauses=pauses, post_avail=T - en, end_reason="trail", other_act=other,
                         head=p, to_primary=sil.copy(), to_silero=sil))
    return rows


def test_grid_outcomes_match_single_policy_outcomes(M):
    rows = _rows()
    cands = M.D.grid(1)[::37]
    oc = M.dyn_grid_outcomes(rows, "to_silero", ("head",), 75, cands)
    assert oc["fc"].shape == (len(rows), len(cands))
    for j, c in enumerate(cands):
        fc, lat, pf, npause = M.dyn_outcomes_at(rows, "to_silero", ("head",), M.dyn_params(c, ("head",)), 75)
        assert np.array_equal(fc, oc["fc"][:, j]) and np.array_equal(lat, oc["lat"][:, j]) and np.array_equal(pf, oc["pf"][:, j])
        assert np.array_equal(npause, oc["npause"])


def test_required_wait_shrinks_with_the_posterior(M):
    rows = _rows(n=6)
    a_lo = M.dyn_params((45, 2, [0]), ("head",))
    a_hi = M.dyn_params((45, 2, [40]), ("head",))
    lo = M.dyn_tracks(rows, "to_silero", ("head",), [a_lo] * len(rows))
    hi = M.dyn_tracks(rows, "to_silero", ("head",), [a_hi] * len(rows))
    for r, x, y in zip(rows, lo, hi):
        assert np.all(y >= x)  # a larger a never asks for more silence
        assert np.allclose(x, r["to_silero"] - 45)  # a = 0: the plain timeout T0
        req = 45 - 40 * r["head"]
        assert np.allclose(y, r["to_silero"] - np.clip(req, 2, 45))
    assert M.as_cand(a_hi, ("head",)) == (45, 2, [40])


def test_fire_conditions_and_summary(M):
    rows = _rows(n=8)
    pts = {"silero_timeout": 20.0, "head_or_silero": {"theta": 0.95, "offset": 30.0}, "hybrid": {"theta": None, "offset": 30.0},
           "head_or_dyn_silero|head": {"theta": 0.95, "offset": 0.0}}
    prm = {"silero|head": [M.dyn_params((30, 2, [20]), ("head",))] * len(rows)}
    conds = M.fire_conditions(rows, pts, prm)
    assert set(conds) == {"silero_timeout", "head_or_silero", "hybrid", "dyn_silero|head", "head_or_dyn_silero|head"}
    for r, c in zip(rows, conds["silero_timeout"]):
        assert np.array_equal(c, r["to_silero"] > 20)
    for r, c, d in zip(rows, conds["dyn_silero|head"], conds["head_or_dyn_silero|head"]):
        req = np.clip(30 - 20 * r["head"], 2, 30)
        assert np.array_equal(c, r["to_silero"] - req > -0.5)  # the fire rule: silence >= required, up to rounding
        assert np.array_equal(d, (r["head"] > 0.95) | (r["to_silero"] - req > 0.0))
    groups, subsets = M.groups_of(rows)
    s = M.D.continuous_summary(rows, conds, groups["open"])
    x = s["silero_timeout"]["all"]
    assert x["n_windows"] == len(rows) and 0 <= x["windows_with_post_event"] <= 1
    # a fixed 20-frame timeout: exactly one rising edge per silence run that reaches 21 frames
    n_ev = sum(len(M.D.events(c)) for c in conds["silero_timeout"])
    assert x["events_per_window"] == round(n_ev / len(rows), 3)
    assert set(subsets) == {"dev", "eval"} and not subsets["dev"].any()  # synthetic meetings are in no ICSI subset


def test_fit_points_are_v2_crossfit_and_in_sample_selection(M):
    import eval_stage1 as E
    rows = _rows()
    folds = np.array([r["fold"] for r in rows])
    sing, hyb = M.ref_specs(rows)
    for spec, hybrid in ((sing["silero_timeout"], False), (hyb["head_or_silero"], True)):
        pts = M.fit_points(spec, hybrid, rows, 75, folds)
        on, en, pauses = M.prep(rows)
        oc, ths = M.M.sweep_outcomes(spec, hybrid, on, en, pauses, 75)
        _, _, _, th = E.v2_crossfit(oc, folds, M.MAX_FC, "turn", tie_miss=hybrid)
        for f in (0, 1):
            assert (ths[int(round(th[f]))] if hybrid else round(pts[f"fold{f}"], 5)) == (pts[f"fold{f}"] if hybrid else th[f])
        j = E._select(oc, np.ones(len(rows), bool), M.MAX_FC, "turn", tie_miss=hybrid)
        fc, lat, pf, npause = M.M.outcomes_at(spec, hybrid, on, en, pauses, 75, pts["all"])
        assert np.array_equal(fc, oc["fc"][:, j]) and fc.mean() <= M.MAX_FC


def test_paired_uses_the_same_resamples_as_frozen_block(M):
    import eval_stage1 as E
    rows = _rows()
    groups, _ = M.groups_of(rows)
    n = len(rows)
    rng = np.random.default_rng(1)
    outs = {(k, "6s"): (rng.random(n) < 0.05, np.where(rng.random(n) < 0.5, np.inf, rng.integers(80, 3000, n).astype(float)))
            for k in ("a", "b")}
    res = M.paired(outs, groups, n, 200, [("a", "b")], {"6s": 75})
    idx = np.random.default_rng(0).integers(0, n, (200, n))
    assert res["a - b | 6s"]["all"] == E._paired(outs[("a", "6s")], outs[("b", "6s")], idx)
    assert set(res["a - b | 6s"]) == {"all", "open", "taken"}
