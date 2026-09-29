"""audioforge/datasets/dyadic.py on a hand-made two-party conversation (no corpus, no models): frame-grid alignment,
turn-end extraction under the floor definition, causality of the agent_end stream, roles, word interpolation.

Conversation T1 (30 s), channel 0 = A, channel 1 = B (labels written straight into the cache layout):
  A: hello there [0.15-0.95] | pause 0.6 s | how are you [1.55-2.35]      -> one floor turn (B's 'yeah' is a backchannel)
  B: yeah [1.10-1.30]                                                     -> backchannel inside A's pause
  B: fine thanks a lot [2.75-4.05]                                        -> B's turn; switch gap 0.4 s after A
  A: okay great then [5.00-6.20] | pause 0.8 s | and then [7.00-7.60]     -> ONE floor turn (pause < max_hold, floor
                                                                             not taken); switch gap 0.95 s after B
  A: more [10.10-10.50]                                                   -> 2.5 s after 'then' (>= max_hold): its
                                                                             own run, a backchannel (1 word, 0.4 s)
  B: right [8.00-8.30]                                                    -> backchannel in A's 0.8 s pause? no: in
                                                                             the 2.5 s hold after A's turn
  B: so let us continue [12.00-13.60]                                     -> B's turn
  A: yes [13.90-14.10] | pause 0.9 s | i think we should stop here now [15.00-17.40]
                                                                          -> one turn: 'yes' is a run that the floor
                                                                             rule merges with the continuation
"""
import json
import math

import numpy as np
import pytest

from audioforge.data import ToneLanguage
from audioforge.datasets import dyadic as D
from audioforge.datasets.ami import FRAME_SEC, SR, frames

A = [(0.15, 0.55, "hello"), (0.55, 0.95, "there"), (1.55, 1.85, "how"), (1.85, 2.05, "are"), (2.05, 2.35, "you"),
     (5.0, 5.4, "okay"), (5.4, 5.8, "great"), (5.8, 6.2, "then"), (7.0, 7.3, "and"), (7.3, 7.6, "then"),
     (10.1, 10.5, "more"), (13.9, 14.1, "yes"), (15.0, 15.3, "i"), (15.3, 15.6, "think"), (15.6, 15.9, "we"),
     (15.9, 16.3, "should"), (16.3, 16.8, "stop"), (16.8, 17.1, "here"), (17.1, 17.4, "now")]
B = [(1.1, 1.3, "yeah"), (2.75, 3.15, "fine"), (3.15, 3.55, "thanks"), (3.55, 3.75, "a"), (3.75, 4.05, "lot"),
     (8.0, 8.3, "right"), (12.0, 12.4, "so"), (12.4, 12.7, "let"), (12.7, 13.0, "us"), (13.0, 13.6, "continue")]
DUR = 30.0


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("dyadic")
    lab = r / "cache" / "labels"
    lab.mkdir(parents=True)
    for cid, words in (("T1", {"0": A, "1": B}), ("T2", {"0": B, "1": A})):  # T2: the channels swapped
        (lab / f"{cid}.json").write_text(json.dumps({"corpus": "behavior_sd", "id": cid, "duration": DUR, "sr": SR,
                                                     "words": words, "zones": [], "speaker_ids": [2001, 2002],
                                                     "activity_source": "labels", "stats": {}, "meta": {}}))
        rng = np.random.default_rng(0)
        np.save(r / "cache" / f"{cid}.mix16k.npy", (rng.standard_normal(int(DUR * SR)) * 0.01).astype(np.float32))
    return r


@pytest.fixture(scope="module")
def ds(root):
    return D.Dyadic(["T1", "T2"], "behavior_sd", root=root, verbose=False)


def test_turns_floor_definition(ds):
    t = [(x["speaker"], round(x["start"], 2), round(x["end"], 2), x["bc"]) for x in ds.turns["T1"]]
    assert t == [("T1:0", 0.15, 2.35, False), ("T1:1", 1.1, 1.3, True), ("T1:1", 2.75, 4.05, False),
                 ("T1:0", 5.0, 7.6, False), ("T1:1", 8.0, 8.3, True), ("T1:0", 10.1, 10.5, True),
                 ("T1:1", 12.0, 13.6, False), ("T1:0", 13.9, 17.4, False)]
    # the hesitation inside A's first turn (the 0.6 s pause a backchannel did not break) and the 0.8 s hold
    assert ds.turns["T1"][0]["hes"] == [(0.95, 1.55)]
    assert ds.turns["T1"][3]["hes"] == [(6.2, 7.0)]
    assert ds.turns["T1"][7]["hes"] == [(14.1, 15.0)]  # 'yes' + 0.9 s + 'i think ...': one turn (floor kept)
    # the swapped conversation has the same turns with the parties exchanged
    assert [(x["speaker"][-1], round(x["end"], 2)) for x in ds.turns["T2"]] == \
        [({"0": "1", "1": "0"}[x["speaker"][-1]], round(x["end"], 2)) for x in ds.turns["T1"]]


def test_roles(root):
    d = D.Dyadic(["T1", "T2"], "behavior_sd", root=root, verbose=False)  # second: B speaks second in T1
    assert d.roles == {"T1": {"agent": 1, "human": 0}, "T2": {"agent": 0, "human": 1}}
    d = D.Dyadic(["T1", "T2"], "behavior_sd", root=root, verbose=False, agent_rule="alternate")
    assert d.roles == {"T1": {"agent": 1, "human": 0}, "T2": {"agent": 0, "human": 1}}
    d = D.Dyadic(["T1", "T2"], "behavior_sd", root=root, verbose=False, agent_rule=0)
    assert d.roles["T1"] == {"agent": 0, "human": 1} and d.roles["T2"] == {"agent": 0, "human": 1}


def test_frame_grid_and_example_keys(ds):
    ex = ds.turn_examples(20.0, 4.0, 2.0, 1.0)
    assert len(ex) == 5 * 2  # 5 non-backchannel turns per conversation (the last one keeps a 2 s trail: DUR = 30)
    for e in ex:
        T = len(e["spk_act"])
        assert T == ToneLanguage.n_frames(len(e["audio"])) == len(e["agent_end"]) == len(e["agent_act"]) == \
            e["spk_targets"].shape[0] == len(e["eot"]) == len(e["hes"])
        assert e["corpus"] == "behavior_sd" and e["word_timing"] == "interpolated"
        assert e["role"] == ("human" if e["party"] == ds.roles[e["meeting"]]["human"] else "agent")
        assert np.array_equal(e["spk_targets"][:, 1], e["agent_act"]) and not e["spk_targets"][:, 2:].any()
        assert e["speaker"] == 2001 + e["party"]
    # A's first turn of T1: window starts at max(0, 0.15 - 4) = 0 -> the frame grid is the conversation's
    e = next(x for x in ex if x["meeting"] == "T1" and x["party"] == 0 and x["onset_frame"] == 1)
    assert e["start"] == 0.0
    # [0.15, 0.95) covers frames 1..11 (int(0.15/0.08) = 1, ceil(0.95/0.08) - 1 = 11); [1.55, 2.35) frames 19..29
    exp = np.zeros(len(e["spk_act"]), np.float32)
    exp[1:12] = 1
    exp[19:30] = 1
    assert np.array_equal(e["spk_act"], exp)
    assert e["turn_end_frame"] == 30 == int(math.ceil(2.35 / FRAME_SEC - 1e-9))
    assert np.array_equal(e["eot"][:30], np.zeros(30)) and e["eot"][30:].all()
    assert e["hes"][12:19].all() and e["hes"].sum() == 7  # the 0.6 s pause: frames 12..18 (silent frames only)
    assert e["text"] == "hello there how are you"
    # B's backchannel 'yeah' [1.10, 1.30) -> frames 13..16 (4 frames) in the other party's column
    assert e["agent_act"][13:17].all() and e["agent_act"].sum() == 4 + len(np.nonzero(frames([(2.75, 4.05)], len(exp)))[0])


def test_agent_end_stream_causal_and_consistent(ds):
    ex = ds.turn_examples(20.0, 4.0, 6.0, 1.0)
    for e in ex:
        oa, ev, ate = e["agent_act"], e["agent_end"], e["agent_turn_end"]
        T = len(oa)
        # causality: the stream on every prefix equals the prefix of the stream
        before = bool(frames(ds.acts[e["meeting"]][f"{e['meeting']}:{1 - e['party']}"], 1, e["start"] - FRAME_SEC)[0]) \
            if e["start"] >= FRAME_SEC else False
        for t in (1, T // 3, T // 2, T - 1, T):
            assert np.array_equal(D.agent_end_stream(oa[:t], before), ev[:t])
        # events are exactly the 1 -> 0 transitions of the agent's activity
        for t in range(1, T):
            assert ev[t] == float(oa[t - 1] > 0.5 and oa[t] <= 0.5)
        # floor turn ends of the agent are a subset of its stop events, and never inside its activity
        assert ((ate > 0) <= (ev > 0)).all()
        # the enrollment anchor is a real event at or before the human's onset
        f = e["agent_end_frame"]
        assert f == -1 or (ev[f] == 1 and f <= e["onset_frame"] and not ev[f + 1: e["onset_frame"] + 1].any())
    # T1, A's turn 'okay great then ... and then' (5.0-7.6): B's turn ended at 4.05 -> event at frame ceil(4.05/0.08) = 51
    e = next(x for x in ex if x["meeting"] == "T1" and x["party"] == 0 and abs(x["start"] + x["onset_frame"] * FRAME_SEC - 5.0) < 0.1)
    abs_ev = e["start"] / FRAME_SEC + np.nonzero(e["agent_end"])[0]
    assert any(abs(v - math.ceil(4.05 / FRAME_SEC)) <= 1 for v in abs_ev)
    assert e["agent_end_frame"] >= 0 and abs(e["start"] / FRAME_SEC + e["agent_end_frame"] - math.ceil(4.05 / FRAME_SEC)) <= 1
    # 'first voice after the agent stops' arms at the human's onset here (B is silent until A starts)
    anyact = np.maximum(e["spk_act"], e["agent_act"])
    assert D.first_voice_after(anyact, e["agent_end"]) == e["onset_frame"]


def test_human_role_filter_and_agent_end_before(ds):
    hum = ds.turn_examples(20.0, 4.0, 6.0, 1.0, roles=("human",))
    assert hum and all(e["role"] == "human" for e in hum)
    assert ds.filtered["turn"]["role"] > 0
    assert D.agent_end_before(np.array([0, 1, 0, 1, 0]), 2) == 1
    assert D.agent_end_before(np.array([0, 1, 0, 1, 0]), 3) == 3
    assert D.agent_end_before(np.array([0, 0, 0]), 2) == -1


def test_interpolate_words_and_bsd_backchannels():
    w = D.interpolate_words("Oh, um, sure.", 3.16, 4.73)
    assert [t for _, _, t in w] == ["oh", "um", "sure"]
    assert w[0][0] == 3.16 and abs(w[-1][1] - 4.73) < 1e-9
    assert all(b[0] == a[1] for a, b in zip(w, w[1:]))
    lens = [b - a for a, b, _ in w]
    assert lens[2] > lens[0] > 0  # 'sure.' (5 chars) longer than 'Oh,' (3)
    assert D.interpolate_words("[laughter]", 0.0, 1.0) == [(0.0, 1.0, "")]
    meta = {"utterances": [
        {"speaker_idx": 1, "start_time": 3.0, "end_time": 7.0, "tts_text": "Hey there, I'm good", "uttr_type": None,
         "backchannels": [{"tts_text": "Yeah?", "start_time": 4.6, "end_time": 5.2}]},
        {"speaker_idx": 0, "start_time": 7.5, "end_time": 9.0, "tts_text": "Oh right", "uttr_type": "interruption",
         "backchannels": []}]}
    words, ev = D.bsd_words(meta)
    assert [t for _, _, t in words[1]] == ["hey", "there", "i'm", "good"]
    assert words[0][0] == (4.6, 5.2, "yeah") and [t for _, _, t in words[0][1:]] == ["oh", "right"]
    assert ev == {"backchannels": [(0, 4.6, 5.2)], "interruptions": [(0, 7.5, "interruption")]}


def test_split_and_energy_channel():
    ids = [str(i) for i in range(12)]
    assert D.split_ids(ids, "dev") == ["0", "5", "10"]
    assert D.split_ids(ids, "train") == [i for i in ids if i not in ("0", "5", "10")]
    assert D.split_ids(ids, "all") == ids
    x = np.zeros((SR, 2), np.float32)
    x[SR // 2:, 1] = 0.5
    assert D.energy_channel(x, SR, 0.6, 0.9)[0] == 1
    assert D.energy_channel(x, SR, 0.1, 0.3)[1] < 2.0  # both silent: ambiguous


def test_recipe_synthetic_fallback():
    cfg = {"data": {"dyadic": {"corpus": "behavior_sd", "mode": "turn"}, "synthetic": {"n_train": 3, "n_val": 2}}}
    tr, va = D.recipe_data(cfg, "train"), D.recipe_data(cfg, "val")
    assert len(tr) == 3 and len(va) == 2 and "spk_act" in tr[0]


def test_recipe_from_labels_cache(root):
    cfg = {"data": {"dyadic": {"corpus": "behavior_sd", "root": str(root), "mode": "turn", "roles": ["human"],
                               "ids": {"train": ["T1"], "val": ["T2"]}, "trail_sec": 6.0}}}
    tr = D.recipe_data(cfg, "train")
    assert tr and all(e["meeting"] == "T1" and e["role"] == "human" for e in tr)
    from audioforge.data import Collate
    b = Collate()(tr[:2])  # the agent-side streams collate like any frame key
    assert b["agent_end"].shape == b["spk_act"].shape and b["spk_targets"].shape[-1] == 4
