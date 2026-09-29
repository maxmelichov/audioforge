"""audioforge.e2e_metrics (research/E2E_FINAL.md scoring) and scripts/research/e2e_final.py's clip selection helpers."""
import sys
from pathlib import Path

import pytest

from audioforge.e2e_metrics import paired_bootstrap, pooled, score_clip, usage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


TURNS = [(1.0, 4.0), (8.0, 12.0), (20.0, 25.0)]


def test_dead_air_cut_in_and_misses():
    # 4.5: response to end 4.0 (500 ms); 10.0: inside turn 2 -> cut-in; 16.5: response to end 12.0 (4.5 s: answered
    # within 6 s but missed within 3 s); nothing after 25.0 -> missed at both horizons
    s = score_clip([4.5, 10.0, 16.5], TURNS)
    ends = s["ends"]
    assert [e["dead_air_ms"] for e in ends] == [500, 4500, None]
    assert [e["missed_3s"] for e in ends] == [False, True, True]
    assert [e["missed_6s"] for e in ends] == [False, False, True]
    assert s["cut_ins"] == 1 and s["cut_in_t"] == [10.0] and s["other"] == 0


def test_next_turn_bounds_the_response_and_repeats_are_other():
    # the user's next turn starts at 8.0: a response at 9.0 is inside it (cut-in), not a response to end 4.0
    s = score_clip([9.0], TURNS)
    assert s["ends"][0]["response"] is None and s["ends"][0]["missed_6s"] and s["cut_ins"] == 1
    # two responses after end 4.0: the first counts, the second is "other"; one before the first turn is "other"
    s = score_clip([0.5, 4.2, 5.0], TURNS)
    assert s["ends"][0]["dead_air_ms"] == 200 and s["other"] == 2 and s["cut_ins"] == 0
    # a response exactly at the end is a response, not a cut-in
    assert score_clip([4.0], TURNS)["ends"][0]["dead_air_ms"] == 0


def test_late_response_beyond_horizon_is_a_miss_without_dead_air():
    s = score_clip([31.5], TURNS)
    last = s["ends"][-1]
    assert last["response"] == 31.5 and last["dead_air_ms"] is None and last["missed_6s"]


def test_scored_mask_and_interruptions_1s():
    s = score_clip([4.5], TURNS, scored=[True, False, False], user_intervals=[(1.0, 4.0), (5.2, 6.0)])
    assert len(s["ends"]) == 1 and s["interruptions_1s"] == 1  # user speech at 5.2 < 4.5 + 1


def test_overlapping_turns_rejected():
    with pytest.raises(AssertionError):
        score_clip([], [(0.0, 5.0), (4.0, 6.0)])


def test_pooled_and_paired_bootstrap():
    a = [dict(score_clip([4.5, 12.5, 25.5], TURNS), audio_s=30.0) for _ in range(6)]
    b = [dict(score_clip([5.5, 13.5, 26.5, 10.0], TURNS), audio_s=30.0) for _ in range(6)]
    pa, pb = pooled(a), pooled(b)
    assert pa["dead_air_ms_median"] == 500 and pb["dead_air_ms_median"] == 1500
    assert pa["cut_ins"] == 0 and pb["cut_ins"] == 6 and pb["cut_ins_per_min"] == pytest.approx(2.0)
    d = paired_bootstrap(a, b, "dead_air_ms_median", n_boot=200)
    assert d["delta"] == -1000 and d["ci"] == [-1000, -1000]
    d = paired_bootstrap(a, b, "cut_ins_per_clip", n_boot=200)
    assert d["delta"] == -1


def test_usage():
    u = usage([(0.0, 1.0, 100.0), (1.0, 1.5, 300.0), (2.0, 2.0, 200.0)])
    assert u["cpu_pct"] == 50.0 and u["rss_mb_peak"] == 300.0


def test_segment_picker_uses_floor_gaps():
    import e2e_final as EF
    h = [{"start": 30.5, "end": 33.0}, {"start": 40.0, "end": 45.0}, {"start": 52.0, "end": 55.0},
         {"start": 60.0, "end": 64.0}, {"start": 70.0, "end": 74.0}]
    ag = [{"start": 33.4, "end": 39.5}, {"start": 45.4, "end": 51.6}, {"start": 55.3, "end": 59.6}]
    act = [(t["start"], t["end"]) for t in h + ag]
    seg = EF.pick_segment(h, ag, act, 100.0, [], length=20.0, max_len=40.0, start_min=30.0)
    assert seg is not None
    a, b = seg
    gaps = EF._floor_gaps([(s, e) for s, e in act], act, 100.0, min_gap=0.2)
    assert a in gaps and b in gaps and 20.0 <= b - a <= 40.0
    assert not any(s < a < e or s < b < e for s, e in act)
    # an excluded zone over every candidate -> no segment
    assert EF.pick_segment(h, ag, act, 100.0, [(0.0, 100.0)], length=20.0) is None
