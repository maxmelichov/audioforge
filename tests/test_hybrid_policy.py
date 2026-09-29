"""Hybrid end-of-turn rule "turn head p >= θ OR primary silent >= k frames": the combinator in conversation.py, its
joint (θ, k) sweep (eot_bench_or) and the eval_stage1 rows, and serve.py's turn_policy "hybrid"."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from audioforge.conversation import eot_bench, eot_bench_or, hybrid_fire_frame, pause_runs, silence_scores

ROOT = Path(__file__).resolve().parents[1]


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_stage1", ROOT / "scripts" / "research" / "eval_stage1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- combinator on hand-made sequences
def test_fire_frame_is_min_of_both_and_never_later_than_timeout():
    act = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    sil = silence_scores(act)                      # 0 0 0 1 2 3 4 5 6 7 8 9
    p = np.array([.1, .2, .3, .5, .7, .96, .97, .2, .1, .1, .1, .1])
    assert hybrid_fire_frame(p, sil, 0.95, 5) == 5 == hybrid_fire_frame(p, sil, 0.95, 99)  # head first
    assert hybrid_fire_frame(p, sil, 0.99, 5) == 7 == hybrid_fire_frame(p, sil, 2.0, 5)    # timeout first
    assert hybrid_fire_frame(p, sil, 0.96, 3) == 5                                        # tie: same frame
    assert hybrid_fire_frame(p, sil, 2.0, 99) == len(p)                                   # neither
    assert hybrid_fire_frame(p, sil, 0.5, 5, start=6) == 6
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = (rng.random(40) < 0.5).astype(float)
        s, q = silence_scores(a), rng.random(40)
        th, k = rng.random(), int(rng.integers(1, 8))
        f = hybrid_fire_frame(q, s, th, k)
        assert f == min(hybrid_fire_frame(q, s, th, 99), hybrid_fire_frame(q, s, 2.0, k))
        assert f <= hybrid_fire_frame(q, s, 2.0, k)


def _synthetic(n=150, seed=0):
    rng = np.random.default_rng(seed)
    A, B, on, en, P = [], [], [], [], []
    for _ in range(n):
        T = int(rng.integers(60, 120))
        e, o = int(rng.integers(25, T - 12)), int(rng.integers(0, 4))
        act = (rng.random(T) < 0.85).astype(float)
        act[e:], act[e - 1] = 0, 1
        hes = np.zeros(T)
        a = int(rng.integers(o + 1, e - 4))
        hes[a:a + 3], act[a:a + 3] = 1, 0
        B.append(silence_scores(act, o))
        A.append(np.clip(rng.random(T) * 0.9 + (np.arange(T) >= e) * rng.random() * 0.3, 0, 1))
        on.append(o), en.append(e), P.append(pause_runs(hes))
    return A, B, on, en, P


def _same(pt, ref, key):
    return (pt[key], pt["fc_rate"], pt["p50_ms"], pt["p90_ms"], pt["miss_rate"]) == \
        (ref["threshold"], ref["fc_rate"], ref["p50_ms"], ref["p90_ms"], ref["miss_rate"])


@pytest.mark.parametrize("chunk", [1, 2])
def test_one_detector_restriction_reproduces_eot_bench(chunk):
    A, B, on, en, P = _synthetic()
    head = eot_bench_or(A, B, on, en, chunk_a=chunk, thresholds_b=[], break_ties_by_miss=False)["at_max_fc"]
    assert _same(head, eot_bench(A, on, en, chunk=chunk)["at_max_fc"], "threshold_a") and head["threshold_b"] is None
    to = eot_bench_or(A, B, on, en, chunk_a=chunk, thresholds_a=[], break_ties_by_miss=False)["at_max_fc"]
    assert _same(to, eot_bench(B, on, en)["at_max_fc"], "threshold_b") and to["threshold_a"] is None
    hp = eot_bench_or(A, B, on, en, chunk_a=chunk, thresholds_b=[], break_ties_by_miss=False, pauses=P,
                      fc_unit="pause")["at_max_fc"]
    ref = eot_bench(A, on, en, chunk=chunk, pauses=P, fc_unit="pause")["at_max_fc"]
    assert _same(hp, ref, "threshold_a") and hp["fc_per_pause"] == ref["fc_per_pause"]


def test_joint_sweep_never_worse_than_either_alone_and_matches_bruteforce():
    A, B, on, en, P = _synthetic(seed=1)
    j = eot_bench_or(A, B, on, en, chunk_a=2, pauses=P)
    best = j["at_max_fc"]
    for ref in (eot_bench(A, on, en, chunk=2)["at_max_fc"], eot_bench(B, on, en)["at_max_fc"]):
        assert (best["p50_ms"], best["p90_ms"]) <= (ref["p50_ms"], ref["p90_ms"])
    assert best["fc_rate"] <= 0.05
    # brute force at the chosen point: per-turn first firing of either, exactly as eot_bench counts it
    th, k = best["threshold_a"] if best["threshold_a"] is not None else np.inf, \
        best["threshold_b"] if best["threshold_b"] is not None else np.inf
    fc, lat = [], []
    for a, b, o, e in zip(A, B, on, en):
        fire = (np.asarray(a) > th) | (np.asarray(b) > k)
        if fire[o:e].any():
            fc.append(True), lat.append(np.inf)
            continue
        ta = np.nonzero(np.asarray(a)[e:] > th)[0]
        tb = np.nonzero(np.asarray(b)[e:] > k)[0]
        la = ((e + ta[0]) // 2 + 1) * 2 - e if len(ta) else np.inf  # head: chunk-2 emission
        lb = tb[0] + 1 if len(tb) else np.inf
        fc.append(False), lat.append(min(la, lb) * 80.0)
    fc, lat = np.array(fc), np.array(lat)
    keep = lat[~fc]
    assert round(fc.mean(), 4) == best["fc_rate"]
    assert float(np.quantile(keep, 0.5, method="inverted_cdf")) == best["p50_ms"]
    assert round(float(np.isinf(keep).mean()), 4) == best["miss_rate"]
    # small explicit grid: table has every (θ, k) cell, each OR point fires no later than the timeout cell alone
    g = eot_bench_or(A, B, on, en, thresholds_a=[0.9, 0.95], thresholds_b=[12, 20], never=False, table=True, chunk_a=2)
    assert len(g["table"]) == 4 and {(c["threshold_a"], c["threshold_b"]) for c in g["table"]} == \
        {(0.9, 12.0), (0.95, 12.0), (0.9, 20.0), (0.95, 20.0)}
    for kk in (12, 20):
        to = eot_bench_or(A, B, on, en, thresholds_a=[], thresholds_b=[kk], never=False, table=True)["table"]
        for c in g["table"]:
            if c["threshold_b"] == kk:
                assert c["fc_rate"] >= to[0]["fc_rate"]  # OR can only add cutoffs ...


def test_eval_hybrid_rows_asserts_pure_rows_and_grid():
    ev = _eval_module()
    A, B, on, en, P = _synthetic(seed=2)
    pt = lambda b: {"n": b["n"], "at_5pct_fc": b["at_max_fc"]}  # noqa: E731
    r = ev.hybrid_rows(A, B, on, en, 2, 1, P, pt(eot_bench(A, on, en, chunk=2)), pt(eot_bench(B, on, en)))
    assert set(r) >= {"at_5pct_turn_fc", "at_5pct_pause_fc", "pure_head", "pure_timeout", "grid"}
    assert len(r["grid"]) == len(ev.HYBRID_GRID_K) * len(ev.HYBRID_GRID_THETA)
    assert {c["k_frames"] for c in r["grid"]} == set(ev.HYBRID_GRID_K)
    assert r["pure_timeout"]["at_5pct_turn_fc"]["k_frames"] == int(eot_bench(B, on, en)["at_max_fc"]["threshold"]) + 1
    with pytest.raises(AssertionError):  # a wrong reference row is caught
        ev.hybrid_rows(A, B, on, en, 1, 1, P, pt(eot_bench(A, on, en, chunk=2)), pt(eot_bench(B, on, en)))


def test_stored_hybrid_run_reproduces_stored_pure_rows():
    new, old = ROOT / "runs/turn_v3_hybrid_n200.json", ROOT / "runs/stage1_turn_v3_eval_n200.json"
    if not new.exists():
        pytest.skip("runs/turn_v3_hybrid_n200.json not produced yet")
    n, o = json.loads(new.read_text())["turn"], json.loads(old.read_text())["turn"]
    for k in ("turn_head_oracle", "timeout_primary_oracle", "timeout_any_speaker_oracle"):
        assert n[k] == o[k], k
    for k in ("turn_head_sortformer_stream", "timeout_sortformer_stream", "turn_head_sortformer_offline",
              "timeout_sortformer_offline"):
        assert n["external_diar"][k] == o["external_diar"][k], k
    h = n["hybrid_oracle"]
    assert h["pure_head"]["at_5pct_turn_fc"]["theta"] == o["turn_head_oracle"]["at_5pct_fc"]["threshold"]
    for row in (n["hybrid_oracle"], n["external_diar"]["hybrid_sortformer_stream"]):
        b, t = row["at_5pct_turn_fc"], row["pure_timeout"]["at_5pct_turn_fc"]
        assert b["fc_rate"] <= 0.05 and (b["p50_ms"], b["p90_ms"]) <= (t["p50_ms"], t["p90_ms"])


# --------------------------------------------------------------------------- serve.py turn_policy "hybrid"
def _serve_helpers():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hybrid_config_and_protocol():
    from audioforge.serve import POLICIES, SessionConfig, validate
    assert "hybrid" in POLICIES and SessionConfig().turn_policy == "timeout"  # default unchanged
    c = SessionConfig()
    assert not c.update({"turn_policy": "hybrid", "timeout_ms": 1680, "eot_threshold": 0.95})
    assert (c.turn_policy, c.timeout_ms, c.eot_threshold) == ("hybrid", 1680, 0.95)
    validate({"type": "turn_end", "t": 2.0, "policy": "hybrid", "p": 0.97, "silence_ms": 320})
    validate({"type": "turn_end", "t": 2.0, "policy": "hybrid", "p": None, "silence_ms": 1040})
    for bad in ({"type": "turn_end", "t": 2.0, "policy": "hybrid", "p": 0.9},
                {"type": "turn_end", "t": 2.0, "policy": "hybrid", "p": 0.9, "silence_ms": 80, "k": 1},
                {"type": "turn_end", "t": 2.0, "policy": "or", "p": 0.9, "silence_ms": 80}):
        with pytest.raises(ValueError):
            validate(bad)


def test_hybrid_dedup_keeps_the_earlier_path_per_turn():
    """Session._hybrid on hand-made candidates: head at 1.5 s and timeout at 2.0 s for the same turn (speech ended at
    1.04 s) -> one hybrid event at 1.5 s; a timeout whose speech came after that decision is a new turn."""
    from collections import deque

    from audioforge.serve import Session, SessionConfig
    s = Session.__new__(Session)
    s.cfg = SessionConfig(turn_policy="hybrid")  # the event tag is the session's policy
    s.hyb_t, s.eot_hist = -np.inf, deque([(1.0, 0.4), (1.5, 0.97), (1.9, 0.99), (3.0, 0.2)])
    head = (1.5, "head", {"p": 0.97, "silence_ms": 320}, 17)          # last VAD speech frame 13 -> 1.12 s
    to = (2.0, "timeout", {"silence_ms": 400, "primary": 0}, 17)      # last primary frame 12 -> 1.04 s
    out = s._hybrid([head, to])
    assert [(e[0], e[1], e[2]["p"], e[2]["silence_ms"]) for e in out] == [(1.5, "hybrid", 0.97, 320)]
    s.hyb_t = -np.inf
    out = s._hybrid([(2.0, "timeout", {"silence_ms": 400, "primary": 0}, 17)])  # timeout path alone: p at 2.0 s
    assert [(e[0], e[1], e[2]["p"], e[2]["silence_ms"], e[2]["primary"]) for e in out] == \
        [(2.0, "hybrid", 0.99, 400, 0)]
    late = (4.0, "timeout", {"silence_ms": 400, "primary": 0}, 40)     # speech until frame 35 (2.88 s) > 2.0 s
    stale = (4.5, "head", {"p": 0.99, "silence_ms": 2400}, 40)        # speech frame 10 (0.88 s) < 4.0 s: dropped
    assert [e[0] for e in s._hybrid([late, stale])] == [4.0]


def test_session_hybrid_fires_on_head_and_timeout_paths():
    """Tiny-model session (scripted energy diarizer, head p ~ 0.99 from the start): the head path fires first (one
    hybrid turn_end at the first chunk, p filled), the timeout path fires for each silence at exactly the plain
    timeout's decision time (p filled from the head); one hybrid turn_end per decision, finals cut at each, and
    the event list does not depend on the client's frame size."""
    from audioforge.serve import Session, SessionConfig, validate
    h = _serve_helpers()
    eng, x = h._energy_engine(), h._speech_silence()

    def run(pol, fs=320):
        s = Session(eng, SessionConfig(turn_policy=pol, timeout_ms=400, eot_threshold=0.98))
        msgs = []
        for i in range(0, len(x), fs):
            msgs += s.process(x[i:i + fs])
        msgs += s.finish()
        for m in msgs:
            validate(m)
        return [m for m in msgs if m["type"] in ("turn_end", "final")]

    hy, both = run("hybrid"), run("both")
    te = [m for m in hy if m["type"] == "turn_end"]
    assert all(m["policy"] == "hybrid" and isinstance(m["p"], float) for m in te)
    b_head = [m for m in both if m["type"] == "turn_end" and m["policy"] == "head"]
    b_to = [m for m in both if m["type"] == "turn_end" and m["policy"] == "timeout"]
    # head path: the head's single firing (t <= 0.2 s), same p / silence as the head policy's event
    assert (te[0]["t"], te[0]["p"], te[0]["silence_ms"]) == (b_head[0]["t"], b_head[0]["p"], b_head[0]["silence_ms"])
    # timeout path: speech resumed after the head's decision, so each timeout firing is its own turn
    assert [(m["t"], m["silence_ms"]) for m in te[1:]] == [(m["t"], m["silence_ms"]) for m in b_to]
    assert te[1]["t"] == 2.0 and te[1]["p"] > 0.98
    assert [m["type"] for m in hy[:-1]] == ["turn_end", "final"] * len(te)  # a final cut at every hybrid turn_end
    assert hy[-1]["type"] == "final"
    for fs in (333, 4000):
        assert [(m["type"], m["t"], m.get("policy"), m.get("text")) for m in run("hybrid", fs)] == \
            [(m["type"], m["t"], m.get("policy"), m.get("text")) for m in hy]
    assert [sorted(m) for m in te] == [sorted(m) for m in b_to[:1] * len(te)]  # protocol keys unchanged
