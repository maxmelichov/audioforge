"""eot-bench v2 (research/archive/EOT_BENCH_V2.md): the leak-free options of conversation.eot_bench and the label-free
enrollment rules of scripts/research/eval_stage1.py. Defaults must reproduce the v1 metric bit for bit."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from audioforge.conversation import FRAME_SEC, _q, eot_bench, floor_stratum, pause_runs, silence_scores

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_stage1", ROOT / "scripts" / "research" / "eval_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- v1 reference (verbatim copy)
REF_FN = '''
def _eot_bench_v1(scores: list[np.ndarray], onsets, ends, frame_ms: float = FRAME_SEC * 1000, chunk: int = 1,
              max_fc: float = 0.05, fixed_latency_ms: float = 400.0, thresholds=None) -> dict:
    \"\"\"eot-bench style sweep.

    Per conversation and threshold θ: a *false cutoff* is any frame t in [onset, end) with score > θ
    (the agent would talk over the user); otherwise the *dead-air latency* is the time from the true
    end to when the first post-end firing is available: (emit(t) - end) * frame_ms, where
    emit(t) = (t // chunk + 1) * chunk accounts for the encoder's chunked look-ahead (chunk = R + 1
    frames; 1 for oracle activity). Never firing = latency inf. Latency percentiles are over the
    conversations that were not cut off (as in eot-bench).

    Reports the lowest-latency threshold with false-cutoff rate <= ``max_fc`` and the false-cutoff
    rate of the most conservative threshold whose P50 latency is <= ``fixed_latency_ms``.
    \"\"\"
    onsets, ends = np.asarray(onsets), np.asarray(ends)
    if thresholds is None:
        allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in scores]))
        thresholds = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
    ths = np.sort(np.asarray(thresholds, np.float64))
    n = len(scores)
    fc = np.zeros((n, len(ths)), bool)
    lat = np.full((n, len(ths)), np.inf)
    for i, s in enumerate(scores):
        s = np.asarray(s, np.float64)
        o, e = int(onsets[i]), int(ends[i])
        pre = s[o:e]
        fc[i] = (pre.max() if len(pre) else -np.inf) > ths
        post = np.maximum.accumulate(s[e:]) if len(s) > e else np.zeros(0)
        idx = np.searchsorted(post, ths, side="right")  # first post-end frame with score > θ
        t = e + idx
        emit = (t // chunk + 1) * chunk
        lat[i] = np.where(idx < len(post), (emit - e) * frame_ms, np.inf)
    fc_rate = fc.mean(0)
    stats = []
    for j in range(len(ths)):
        keep = lat[~fc[:, j], j]
        stats.append((_q(keep, 0.5), _q(keep, 0.9), float(np.isinf(keep).mean()) if len(keep) else 1.0))

    def point(j):
        if j is None:
            return dict(threshold=None, fc_rate=None, p50_ms=float("inf"), p90_ms=float("inf"), miss_rate=None)
        return dict(threshold=round(float(ths[j]), 5), fc_rate=round(float(fc_rate[j]), 4),
                    p50_ms=stats[j][0], p90_ms=stats[j][1], miss_rate=round(stats[j][2], 4))

    ok = [j for j in range(len(ths)) if fc_rate[j] <= max_fc]
    best = min(ok, key=lambda j: (stats[j][0], stats[j][1])) if ok else None
    fixed = [j for j in range(len(ths)) if stats[j][0] <= fixed_latency_ms]
    fixed_j = min(fixed, key=lambda j: (fc_rate[j], stats[j][0])) if fixed else None
    sub = np.unique(np.linspace(0, len(ths) - 1, min(len(ths), 60)).astype(int))
    return dict(n=n, at_max_fc=point(best), max_fc=max_fc, fixed_latency_ms=fixed_latency_ms,
                at_fixed_latency=point(fixed_j),
                curve=[(round(float(ths[j]), 4), round(float(fc_rate[j]), 4), stats[j][0], stats[j][1])
                       for j in sub])

'''
_ns = {"np": np, "FRAME_SEC": FRAME_SEC, "_q": _q}
exec(REF_FN, _ns)
_eot_bench_v1 = _ns["_eot_bench_v1"]

HAND = [np.array(x) for x in ([0.1, 0.3, 0.2, 0.1, 0.1, 0.2, 0.6, 0.9, 0.9],
                              [0.1, 0.7, 0.1, 0.2, 0.1, 0.8, 0.9, 0.9, 0.9],
                              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.95])]


def test_default_output_unchanged_hand_made():
    for kw in (dict(frame_ms=80, thresholds=[0.5, 0.75], max_fc=0.4, fixed_latency_ms=200),
               dict(frame_ms=80, thresholds=[0.5, 0.75], max_fc=0.05), dict(frame_ms=80, thresholds=[0.5], chunk=2),
               dict()):
        assert eot_bench(HAND, [0] * 3, [5] * 3, **kw) == _eot_bench_v1(HAND, [0] * 3, [5] * 3, **kw)
    # the documented numbers of tests/test_turn.py
    p = eot_bench(HAND, [0] * 3, [5] * 3, frame_ms=80, thresholds=[0.5, 0.75], max_fc=0.05)["at_max_fc"]
    assert set(p) == {"threshold", "fc_rate", "p50_ms", "p90_ms", "miss_rate"}
    assert p["threshold"] == 0.75 and p["p50_ms"] == 240 and p["p90_ms"] == 320


def test_default_output_unchanged_random():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(1, 30))
        sc = [rng.random(int(rng.integers(3, 60))) for _ in range(n)]
        if trial % 3 == 0:
            sc = [np.round(s * 10) for s in sc]  # integer ties (silence-timeout-like)
        en = [int(rng.integers(1, len(s) + 1)) for s in sc]
        on = [int(rng.integers(0, e)) for e in en]
        kw = dict(chunk=int(rng.integers(1, 4)), max_fc=float(rng.choice([0.0, 0.05, 0.3])))
        assert eot_bench(sc, on, en, **kw) == _eot_bench_v1(sc, on, en, **kw)


# --------------------------------------------------------------------------- per-pause false cutoffs
def test_pause_runs():
    assert pause_runs(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])) == [(1, 3), (5, 6), (7, 10)]
    assert pause_runs(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1]), min_frames=2) == [(1, 3), (7, 10)]
    assert pause_runs(np.zeros(4)) == []


def test_per_pause_fc_hand_made():
    # primary: speech 0-3, pause 3-5 (2 frames), speech 5-7, pause 7-11 (4 frames), speech 11-13, end = 13
    act = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], np.float32)
    hes = np.zeros(20)
    hes[3:5] = 1
    hes[7:11] = 1
    s = silence_scores(act)
    ps = [pause_runs(hes)]
    # 3-frame timeout (θ = 2.5): fires in the 4-frame pause only -> 1 of 2 pauses, and the turn is cut off
    r = eot_bench([s], [0], [13], thresholds=[2.5], max_fc=1.0, pauses=ps)
    p = r["at_max_fc"]
    assert p["fc_per_pause"] == 0.5 and p["n_pauses"] == 2 and p["fc_rate"] == 1.0
    # 1-frame timeout: both pauses
    assert eot_bench([s], [0], [13], thresholds=[0.5], max_fc=1.0, pauses=ps)["at_max_fc"]["fc_per_pause"] == 1.0
    # FC per pause vs per turn: a second turn without pauses, never cut off, fires at +3 frames
    act2 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], np.float32)
    s2 = silence_scores(act2)
    r = eot_bench([s, s2], [0, 0], [13, 4], thresholds=[2.5, 4.5], max_fc=0.5, pauses=ps + [[]])
    p = r["at_max_fc"]  # θ=2.5: turn FC 1/2 (<= 0.5), pause FC 1/2; latency of turn 2 = 3 frames
    assert p["threshold"] == 2.5 and p["fc_rate"] == 0.5 and p["fc_per_pause"] == 0.5 and p["p50_ms"] == 240
    # fc_unit='pause' constrains the per-pause rate instead: max_fc 0.0 -> θ = 4.5 (no pause fired)
    r = eot_bench([s, s2], [0, 0], [13, 4], thresholds=[2.5, 4.5], max_fc=0.0, pauses=ps + [[]], fc_unit="pause")
    assert r["at_max_fc"]["threshold"] == 4.5 and r["at_max_fc"]["fc_per_pause"] == 0.0
    with pytest.raises(ValueError):
        eot_bench([s], [0], [13], fc_unit="pause")


def test_post_end_window_and_censoring():
    s = np.array([0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.9])  # end 3, fires at frame 7 -> 5 frames = 400 ms
    assert eot_bench([s], [0], [3], thresholds=[0.5])["at_max_fc"]["p50_ms"] == 400
    r = eot_bench([s], [0], [3], thresholds=[0.5], post_end_frames=4, censored=[True])["at_max_fc"]
    assert r["p50_ms"] == float("inf") and r["miss_rate"] == 1.0 and r["miss_censored"] == 1
    assert eot_bench([s], [0], [3], thresholds=[0.5], post_end_frames=5)["at_max_fc"]["p50_ms"] == 400


# --------------------------------------------------------------------------- floor strata
def test_floor_stratum_hand_made():
    y = np.zeros((30, 3), np.float32)
    y[0:10, 0] = 1  # primary ends at 10
    assert floor_stratum(y, 10) == "open"
    y2 = y.copy()
    y2[8:15, 1] = 1  # talks over the end
    assert floor_stratum(y2, 10) == "overlap"
    y3 = y.copy()
    y3[16:20, 2] = 1  # starts 6 frames after the end
    assert floor_stratum(y3, 10) == "switch"
    y4 = y.copy()
    y4[23:26, 2] = 1  # starts after the 13-frame horizon
    assert floor_stratum(y4, 10) == "open"
    y5 = y.copy()
    y5[5:10, 1] = 1  # overlap that stops with the primary: nobody takes the floor
    assert floor_stratum(y5, 10) == "open"


def test_strata_split_in_eot_bench():
    # two open turns (fast 80 ms / 160 ms) and two taken turns (one miss, one 400 ms)
    sc = [np.array([0, 0, 0.9, 0.9]), np.array([0, 0, 0.1, 0.9]), np.array([0, 0, 0, 0, 0, 0, 0]),
          np.array([0, 0, 0, 0, 0, 0, 0.9])]
    g = {"open": np.array([1, 1, 0, 0], bool), "taken": np.array([0, 0, 1, 1], bool)}
    p = eot_bench(sc, [0] * 4, [2] * 4, thresholds=[0.5], groups=g, n_boot=50)["at_max_fc"]
    assert p["miss_rate"] == 0.25
    so, st = p["strata"]["open"], p["strata"]["taken"]
    assert so["n"] == 2 and so["miss_rate"] == 0.0 and so["p50_ms"] == 80 and so["p90_ms"] == 160
    assert st["n"] == 2 and st["miss_rate"] == 0.5 and st["p50_ms"] == 400
    assert so["miss_rate_ci"] == [0.0, 0.0] and len(p["miss_rate_ci"]) == 2


# --------------------------------------------------------------------------- label-free enrollment
def _track(T=60, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.random((T, 4)) * 0.3
    p[2:12, 1] = 0.9  # another speaker talks first
    p[15:40, 0] = 0.9  # then the primary
    p[45:50, 2] = 0.8
    return p


def test_first_active_and_causal_dominant_hand_made(ev):
    p = _track()
    assert ev.enroll_first_active(p) == 1
    c = ev.enroll_causal_dominant(p, k=10, s=5)
    assert (c[:2] == -1).all() and (c[2:12] == 1).all()
    # column 1 silent from 12; > 5 silent frames at 18, when column 0 (active since 15) dominates the look-back
    assert (c[12:18] == 1).all() and (c[18:47] == 0).all()
    # column 0 silent from 40 (> 5 frames at 45); column 2 (active from 45) outnumbers it in the look-back at 47
    assert (c[47:] == 2).all()
    # causal: the binding up to t never depends on frames after t
    for t in (10, 20, 47):
        q = p.copy()
        q[t + 1:] = np.random.default_rng(t).random((len(p) - t - 1, 4))
        assert np.array_equal(ev.enroll_causal_dominant(q, 10, 5)[: t + 1], c[: t + 1])


def test_causal_enrollment_never_reads_labels(ev):
    """Mutation test: shuffling / replacing every label field leaves the causal bindings (and the fed tracks)
    unchanged; the oracle rule does depend on them."""
    rng = np.random.default_rng(3)
    T = 60
    p = _track(T, 1)
    act = np.zeros(T, np.float32)
    act[15:40] = 1
    ex = dict(spk_act=act, onset_frame=15, turn_end_frame=40)
    for trial in range(5):
        mut = dict(spk_act=rng.permutation(act) if trial % 2 else rng.integers(0, 2, T).astype(np.float32),
                   onset_frame=int(rng.integers(0, 30)), turn_end_frame=int(rng.integers(31, T)))
        for mode in ("causal_dominant", "first_active"):
            a, b = ev._v2_cols(ex, p, mode), ev._v2_cols(mut, p, mode)
            assert np.array_equal(a[3], b[3]) and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
            assert a[2] == b[2]
            assert np.array_equal(ev.enroll(p, mode, act, 15, 40), ev.enroll(p, mode, mut["spk_act"], 0, 5))
    # the oracle rule reads the labels (sanity of the mutation test itself)
    assert ev.enroll(p, "oracle", act, 15, 40)[0] == 0
    y = np.zeros(T, np.float32)
    y[2:12] = 1
    assert ev.enroll(p, "oracle", y, 2, 12)[0] == 1


def test_bound_track_is_a_per_frame_primary(ev):
    p = _track()
    c = ev.enroll_causal_dominant(p, 10, 5)
    act, cols = ev.bound_track(p, c)
    for t in range(len(p)):
        if c[t] < 0:
            assert act[t] == 0
            continue
        assert act[t] == pytest.approx(p[t, c[t]]) and cols[t, 0] == pytest.approx(p[t, c[t]])
        assert sorted(cols[t].tolist()) == pytest.approx(sorted(p[t].astype(np.float32).tolist()))


def test_arm_frame(ev):
    assert ev.arm_frame(np.array([0, 0, 1, 0])) == 2 and ev.arm_frame(np.zeros(3)) == 3


@pytest.mark.skipif(not (ROOT / "data/ami/annotations").exists() or not (ROOT / "data/ami/cache").exists(),
                    reason="AMI annotations / audio cache not present")
def test_recut_windows_reproduce_library(ev):
    """The re-cut (fixed start, longer trail) reproduces AMI.turn_examples at trail 2 s and keeps onset / end."""
    base, ext, meta, _, meta_base = ev.v2_data(6.0)  # asserts the reproduction internally
    assert len(base) == len(ext) == len(meta)
    assert all(len(u["spk_act"]) >= len(v["spk_act"]) for u, v in zip(ext, base))
    assert all(m["post_avail"] <= 76 for m in meta)
    assert {m["end_reason"] for m in meta} <= {"resume", "trail", "meeting_end"}
