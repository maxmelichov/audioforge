"""scripts/research/bench_turnbench.py event commitment rules (no data, no models) and the vendored scorer on a toy gold."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "integrations" / "turnbench_scorer"))


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("bench_turnbench", ROOT / "scripts" / "research" / "bench_turnbench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_timeout_events_commit_at_window_end(M):
    speech = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    # 32 ms windows, k = 0.1 s -> 4 silent windows; the first run reaches 4 at window 6 (end 7 * 0.032)
    ev = M.timeout_events(speech, 0.032, 0.1)
    assert ev == [round(7 * 0.032, 4), round(13 * 0.032, 4)]
    assert M.timeout_events(np.zeros(10, bool), 0.032, 0.1) == []  # never armed: no speech yet


def test_edge_events_rising_edge_and_refractory(M):
    s = np.array([0.1, 0.9, 0.95, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9], np.float32)
    emit = (np.arange(len(s)) // 2 + 1) * 2 * 0.08
    ev = M.edge_events(s, emit, 0.5, refractory=2.0)
    assert ev[0] == pytest.approx(emit[1]) and len(ev) == 2 and ev[1] == pytest.approx(emit[28])  # frame 4 in refractory
    assert M.edge_events(s, emit, 0.5, refractory=0.0) == [pytest.approx(emit[1]), pytest.approx(emit[4]), pytest.approx(emit[28])]


def test_merge_and_segments(M):
    assert M.merge_events([1.0, 4.0], [1.5, 8.0], 2.0) == [1.0, 4.0, 8.0]
    assert M.strictly_increasing([1.0, 1.0, 2.0, 1.5, 3.0]) == [1.0, 2.0, 3.0]
    segs = M.segments(int(150 * M.SR))
    assert segs[0] == (0, 60 * M.SR, 0) and segs[1] == (52 * M.SR, 120 * M.SR, 8 * M.SR)
    assert segs[-1][1] == 150 * M.SR
    p = np.array([0.0, 1.0, 1.0, 0.0, 0.0] * 4, np.float32)  # 20 chunks of 32 ms = 0.64 s -> 8 frames
    tr = M.frame_track(p, 8)
    assert tr.shape == (8,) and 0 <= tr.min() and tr.max() <= 1


def test_vendored_scorer_toy_gold(M):
    """A prediction exactly at a consensus turn end scores recall 1 / fp 0; one inside a mid-turn pause is an FP."""
    from turnbench.data import Conversation
    from turnbench.gold import events_for_conversation
    from turnbench.score import score_conversation
    from turnbench.submission import ConversationPrediction, SpeakerEvents
    seg = lambda s, e, lab="Normal Turn": (s, e, lab, "x")  # noqa: E731
    tracks = {}
    for an in "abc":  # speaker 1: turn 1-3 | pause | 3.5-5 (same speaker resumes) ; speaker 2: turn 6-8
        tracks[(1, an)] = [seg(1.0, 3.0), seg(3.5, 5.0)]
        tracks[(2, an)] = [seg(6.0, 8.0)]
    conv = Conversation(conversation_id="1", duration_s=10.0, annotations=tracks, audio_bytes={})
    ev = events_for_conversation(conv)
    assert [round(e.time_s, 2) for e in ev.eot_positive_events if e.speaker == 1] == [5.0]
    assert any(abs(s.start - 3.0) < 1e-6 for s in ev.eot_negative_spans if s.speaker == 1)
    good = ConversationPrediction(conversation_id="1", speaker_1=SpeakerEvents(eot=[5.3], interruption=[]),
                                  speaker_2=SpeakerEvents(eot=[8.2], interruption=[]))
    sc = score_conversation(good, conv).task_eot
    assert sc.recall == 1.0 and sc.fp_rate == 0.0 and sorted(round(x) for x in sc.latencies_ms) == [200, 300]
    bad = ConversationPrediction(conversation_id="1", speaker_1=SpeakerEvents(eot=[3.2, 5.3], interruption=[]),
                                 speaker_2=SpeakerEvents(eot=[], interruption=[]))
    sc = score_conversation(bad, conv).task_eot
    assert sc.fp == 1 and sc.recall == 0.5
