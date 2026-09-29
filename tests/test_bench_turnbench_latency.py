"""scripts/research/bench_turnbench_latency.py: the bounded-exponential wait, the isotonic calibration (monotone, pointwise, so
the calibrated stream stays causal; fitted on the fitting split only), the silence / velocity / head branches, the
scorer-matching replica and the oto gold rules. No data, no models."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "integrations" / "turnbench_scorer"))


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("bench_turnbench_latency", ROOT / "scripts" / "research" / "bench_turnbench_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- exponential policy
def test_exp_wait_bounds_and_shape(M):
    ph = np.linspace(0, 1, 101)
    w = M.exp_wait(ph, 0.3, 2.0, 3.0, 1.0)
    assert w[0] == pytest.approx(2.0)  # phat = 0 -> Tmax
    assert w[-1] == pytest.approx(0.3 + 1.7 * np.exp(-3.0))  # phat = 1 -> Tmin + (Tmax - Tmin) e^-lam
    assert np.all(np.diff(w) <= 1e-12) and np.all((w >= 0.3) & (w <= 2.0))  # non-increasing, bounded
    assert M.exp_wait(1.0, 0.3, 2.0, 50.0, 1.0) == pytest.approx(0.3)  # large lam: Tmin at phat = 1
    # gamma > 1 keeps the wait long until phat is high; gamma < 1 shortens it early
    assert M.exp_wait(0.5, 0.3, 2.0, 3.0, 4.0) > M.exp_wait(0.5, 0.3, 2.0, 3.0, 1.0) > M.exp_wait(0.5, 0.3, 2.0, 3.0, 0.5)
    assert M.exp_wait(np.array([-1.0, 2.0]), 0.3, 2.0, 3.0, 1.0) == pytest.approx(M.exp_wait(np.array([0.0, 1.0]), 0.3, 2.0, 3.0, 1.0))


def test_lin_wait_is_the_baselines_clamp(M):
    assert M.lin_wait(np.array([0.0, 0.5, 1.0]), 2.0, 2.5, 0.3) == pytest.approx([2.0, 0.75, 0.3])


# ----------------------------------------------------------------------------- isotonic calibration
def test_isotonic_monotone_clipped_and_pointwise_causal(M):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.9, 1.0, 4000)
    y = (rng.uniform(size=4000) < (p - 0.9) * 10).astype(float)
    cal = M.Isotonic().fit(p, y)
    grid = np.linspace(0.85, 1.05, 200)
    out = cal(grid)
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1
    assert cal(0.5) == pytest.approx(cal(p.min())) and cal(2.0) == pytest.approx(cal(p.max()))  # clipped
    # causality: phat[:t] is unchanged by any change of p[t:]
    s = rng.uniform(0.9, 1.0, 300)
    a = cal(s)
    for t in (0, 50, 299):
        s2 = s.copy()
        s2[t:] = rng.uniform(0.0, 1.0, 300 - t)
        assert np.array_equal(cal(s2)[:t], a[:t])
    # the velocity feature built on it is causal too
    r = M.rise(a)
    s3 = s.copy()
    s3[200:] = 0.0
    assert np.array_equal(M.rise(cal(s3))[:200], r[:200])


def test_isotonic_depends_only_on_the_fitting_split(M):
    rng = np.random.default_rng(1)
    pf, yf = rng.uniform(size=500), rng.integers(0, 2, 500).astype(float)
    a = M.Isotonic().fit(pf, yf)
    b = M.Isotonic().fit(pf, yf)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)  # deterministic given the fitting data
    js = a.to_json()
    c = M.Isotonic(js["x"], js["y"])
    assert c(0.5) == pytest.approx(a(0.5), abs=0.05)


def test_event_weighted_calibration(M):
    # one long positive (100 chunks) and ten short negatives (1 chunk each): per-event weights make the rate 1/11
    p = np.full(110, 0.5)
    y = np.r_[np.ones(100), np.zeros(10)]
    w = np.r_[np.full(100, 1 / 100), np.ones(10)]
    assert M.Isotonic().fit(p, y, w)(0.5) == pytest.approx(1 / 11)


# ----------------------------------------------------------------------------- branches
def _chan(M, speech, head, dur=100.0):
    return M.Chan("c", 1, np.asarray(speech, bool), np.asarray(head, np.float64), dur)


def test_silence_branch_fixed_equals_timeout_events(M):
    import bench_turnbench as TB
    rng = np.random.default_rng(2)
    sp = rng.uniform(size=3000) < 0.6
    sp = np.repeat(sp[::10], 10)[:3000]
    ch = _chan(M, sp, np.zeros(int(3000 * 0.032 / 0.08) + 2))
    for k in (0.2, 0.5, 1.0):
        ours = [round(float(t), 4) for t in ch.silence_events(np.full(len(ch.s), k))]
        assert ours == TB.timeout_events(sp, M.CHUNK, k)


def test_silence_branch_dynamic_wait_one_event_per_run(M):
    sp = np.r_[np.ones(10), np.zeros(100), np.ones(5), np.zeros(100)].astype(bool)
    ch = _chan(M, sp, np.zeros(100))
    wait = np.where(ch.run < 50, 0.64, 1.28)  # 20 chunks in run 1, 40 chunks in run 2
    ev = ch.silence_events(wait)
    assert len(ev) == 2
    assert ev[0] == pytest.approx((10 + 20) * M.CHUNK) and ev[1] == pytest.approx((115 + 40) * M.CHUNK)
    assert ch.silence_events(np.full(len(ch.s), 10.0)).size == 0  # never reached


def test_velocity_early_exit_latched_at_tmin(M):
    sp = np.r_[np.ones(10), np.zeros(200)].astype(bool)
    ch = _chan(M, sp, np.zeros(100))
    vel = np.zeros(len(ch.s), bool)
    vel[2] = True  # trigger 3 chunks into the silence, before Tmin
    ev = ch.silence_events(np.full(len(ch.s), 5.0), vel=vel, tmin=0.32)  # Tmin = 10 chunks
    assert ev == pytest.approx([(10 + 10) * M.CHUNK])
    vel[:] = False
    vel[30] = True  # trigger after Tmin: fires at the trigger
    assert ch.silence_events(np.full(len(ch.s), 5.0), vel=vel, tmin=0.32) == pytest.approx([(10 + 31) * M.CHUNK])


def test_head_reading_is_the_latest_emitted_frame(M):
    head = np.arange(20, dtype=float) / 20
    ch = _chan(M, np.r_[np.ones(1), np.zeros(19)].astype(bool), head)
    for tau, f, p in zip(ch.tau, ch.fidx, ch.p):
        assert f == (2 * np.floor(tau / 0.16 + 1e-9) - 1) and (f < 0 or ((f // 2 + 1) * 2) * 0.08 <= tau + 1e-9)
        assert p == (head[f] if f >= 0 else 0.0)


def test_merge_refractory_and_sources(M):
    t, s = M.merged([1.0, 5.0], [1.5, 3.2, 5.1, 9.0], dur=8.0)
    assert t == [1.0, 3.2] and s == ["head", "silence"]  # 1.5, 5.0, 5.1 inside the refractory; 9.0 past the end


def test_claimed_times_matches_scorer(M):
    from turnbench.gold import AnchorEvent, Interval
    from turnbench.score import score_task
    rng = np.random.default_rng(3)
    for _ in range(50):
        pos = sorted(rng.uniform(0, 60, 8).round(2))
        pred = sorted(set(rng.uniform(0, 62, 12).round(3)))
        exc = [Interval(1, 30.0, 31.0)]
        sc = score_task([AnchorEvent(1, float(t)) for t in pos], [], {1: [float(x) for x in pred]}, exc)
        m = M.claimed_times([float(t) for t in pos], [float(x) for x in pred], exc)
        lat = [(x - a) * 1000 for a, x in zip(pos, m) if x is not None]
        assert len(lat) == sc.tp and np.allclose(sorted(lat), sorted(sc.latencies_ms))


# ----------------------------------------------------------------------------- oto gold
def test_oto_segments_backchannel_and_merge(M):
    segs = [(0.0, 2.0), (2.1, 3.0), (3.4, 4.0), (10.0, 10.5), (20.0, 21.0), (21.2, 22.0)]
    turn, bc = M.oto_segments(segs, gap=0.0)
    assert bc == [(10.0, 10.5)]  # isolated short run
    assert (0.0, 2.0) in turn and (2.1, 3.0) in turn
    turn2, _ = M.oto_segments(segs, gap=0.2)
    assert (0.0, 3.0) in turn2 and (3.4, 4.0) in turn2 and (20.0, 22.0) in turn2


def test_oto_gold_uses_the_scorers_floor_construction(M):
    words = [[(0.0, 2.0, ""), (2.5, 4.0, ""), (8.0, 9.0, ""), (9.2, 10.0, "")], [(4.5, 7.0, ""), (8.4, 8.7, "")]]
    g = M.oto_gold(words, [])
    pos = sorted((e.speaker, e.time_s) for e in g.eot_positive_events)
    assert (1, 4.0) in pos and (2, 7.0) in pos and (1, 10.0) in pos  # floor passes / last turn
    assert [(s.speaker, s.start) for s in g.eot_negative_spans] == [(1, 2.0), (1, 9.0)]  # pauses; bc not a turn
