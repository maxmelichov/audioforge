"""AMI label construction on a hand-written words-XML fixture (no corpus needed) + optional real-data checks.

Fixture meeting TST001 (30 s), speakers A (FEE001) and B (MEE002):
  A: hello there [0.0-0.8] | pause 0.6 s | how are you [1.4-2.2]         -> one floor turn, 1 hesitation
  B: yeah [1.05-1.2] (backchannel, inside A's pause: does not take the floor)
  B: fine thanks a lot [2.6-3.9]                                         -> switch gap 0.4 s after A
  A: okay great then [5.0-6.2]                                           -> switch gap 1.1 s after B
"""

import numpy as np
import pytest

from audioforge.data import Collate, ToneLanguage
from audioforge.datasets import ami

W = '   <w nite:id="TST001.{a}.words{i}" starttime="{s}" endtime="{e}"{p}>{t}</w>\n'
WORDS = {
    "A": [(0.0, 0.4, "Hello"), (0.4, 0.8, "there"), (0.8, 0.8, ",", 1), (1.4, 1.6, "how"), (1.6, 1.8, "are"),
          (1.8, 2.2, "you"), (2.2, 2.2, "?", 1), (5.0, 5.4, "Okay"), (5.4, 5.8, "great"), (5.8, 6.2, "then")],
    "B": [(1.05, 1.2, "Yeah"), (2.6, 3.0, "Fine"), (3.0, 3.4, "thanks"), (3.4, 3.6, "a"), (3.6, 3.9, "lot"),
          (3.9, 3.9, ".", 1)],
}
MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="meet.00" xmlns:nite="http://nite.sourceforge.net/">
   <meeting nite:id="meet_1" observation="TST001" duration="30">
      <speaker nite:id="TST001_1" nxt_agent="A" global_name="FEE001"/>
      <speaker nite:id="TST001_2" nxt_agent="B" global_name="MEE002"/>
   </meeting>
</nite:root>
"""
DUR = 30.0


def _xml(agent, words, extra=""):
    body = "".join(W.format(a=agent, i=i, s=w[0], e=w[1], t=w[2], p=' punc="true"' if len(w) > 3 else "")
                   for i, w in enumerate(words))
    return ('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
            f'<nite:root nite:id="TST001.{agent}.words" xmlns:nite="http://nite.sourceforge.net/">\n{body}{extra}</nite:root>\n')


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "annotations" / "words").mkdir(parents=True)
    (tmp_path / "annotations" / "corpusResources").mkdir()
    (tmp_path / "annotations" / "corpusResources" / "meetings.xml").write_text(MEETINGS_XML)
    for a, w in WORDS.items():
        extra = '   <vocalsound nite:id="x" starttime="10.0" endtime="11.0" type="laugh"/>\n' if a == "A" else ""
        (tmp_path / "annotations" / "words" / f"TST001.{a}.words.xml").write_text(_xml(a, w, extra))
    (tmp_path / "cache").mkdir()
    rng = np.random.default_rng(0)
    np.save(tmp_path / "cache" / "TST001.f32.npy", (0.01 * rng.standard_normal(int(DUR * ami.SR))).astype(np.float32))
    return tmp_path


def test_parse_and_normalize():
    xml = _xml("A", [(0.0, 0.3, "I&#39;m"), (0.3, 0.3, ",", 1), (0.3, 0.9, "T_V_"), (0.9, 1.2, "Mm-hmm")],
               '   <vocalsound nite:id="v" starttime="2.0" endtime="3.0" type="laugh"/>\n')
    assert ami.parse_words_xml(xml) == [(0.0, 0.3, "i'm"), (0.3, 0.9, "tv"), (0.9, 1.2, "mm hmm")]


def test_turns_floor_and_gap(root):
    words = ami.meeting_words(root / "annotations", "TST001")
    assert set(words) == {"FEE001", "MEE002"}
    t = ami.speaker_turns(words)  # floor definition (default)
    assert [(x["speaker"], x["start"], x["end"], x["bc"]) for x in t] == [
        ("FEE001", 0.0, 2.2, False), ("MEE002", 1.05, 1.2, True), ("MEE002", 2.6, 3.9, False),
        ("FEE001", 5.0, 6.2, False)]
    assert t[0]["text"] == "hello there how are you" and t[0]["hes"] == [(0.8, 1.4)]
    assert ami.next_turn(t, 0)["start"] == 2.6 and ami.next_turn(t, 2)["start"] == 5.0
    g = ami.speaker_turns(words, turn_def="gap")  # literal gap rule: the 0.6 s pause ends A's turn
    a_turns = [x for x in g if x["speaker"] == "FEE001"]
    assert [(x["start"], x["end"]) for x in a_turns] == [(0.0, 0.8), (1.4, 2.2), (5.0, 6.2)]
    assert all(not x["hes"] for x in g)
    # a non-backchannel run of B inside A's pause takes the floor
    w2 = dict(words, MEE002=[(0.85, 1.0, "no"), (1.0, 1.2, "wait"), (1.2, 1.35, "what")])
    assert len([x for x in ami.speaker_turns(w2) if x["speaker"] == "FEE001"]) == 3


def test_stats(root):
    ds = ami.AMI(["TST001"], root)
    st = ds.stats()
    assert st["turns"] == 3 and st["backchannels"] == 1 and st["speakers"] == 2
    assert st["switch_gap"]["n"] == 2 and st["switch_gap"]["p50"] == pytest.approx(0.75)
    assert st["hesitation"]["n"] == 1 and st["hesitation"]["p50"] == pytest.approx(0.6)
    assert st["overlap_frac_of_speech"] == 0.0


def test_diar_windows(root):
    ds = ami.AMI(["TST001"], root)
    assert ds.speaker_ids == ["FEE001", "MEE002"]
    win = ds.diar(window_sec=20.0, hop_sec=10.0)
    assert [w["start"] for w in win] == [0.0, 10.0]
    w = win[0]
    T = ToneLanguage.n_frames(len(w["audio"]))
    assert len(w["audio"]) == 20 * ami.SR and w["spk_targets"].shape == (T, 4) == (251, 4)
    y = w["spk_targets"]
    exp_a = np.zeros(T)
    exp_a[0:10] = exp_a[17:28] = exp_a[62:78] = 1
    exp_b = np.zeros(T)
    exp_b[13:15] = exp_b[32:49] = 1
    np.testing.assert_array_equal(y[:, 0], exp_a)  # A arrives first
    np.testing.assert_array_equal(y[:, 1], exp_b)
    assert not y[:, 2:].any() and w["speakers"] == [0, 1]
    assert not win[1]["spk_targets"].any()  # 10-30 s: silence


def test_turn_examples(root):
    ds = ami.AMI(["TST001"], root)
    ex = ds.turn_examples(window_sec=20.0, lead_sec=4.0, trail_sec=2.0, min_trail=1.0)
    assert [e["text"] for e in ex] == ["hello there how are you", "fine thanks a lot", "okay great then"]
    for e in ex:
        T = ToneLanguage.n_frames(len(e["audio"]))
        assert e["spk_targets"].shape == (T, 4) and len(e["spk_act"]) == len(e["eot"]) == len(e["hes"]) == T
        np.testing.assert_array_equal(e["spk_targets"][:, 0], e["spk_act"])
        assert e["eot"][: e["turn_end_frame"]].sum() == 0 and e["eot"][e["turn_end_frame"]:].all()
        assert e["spk_act"][e["turn_end_frame"] - 1] == 1 and not e["spk_act"][e["turn_end_frame"]:].any()
        assert e["spk_act"][e["onset_frame"]] == 1 and not e["spk_act"][: e["onset_frame"]].any()
    a = ex[0]  # window [0, 4.2]: A's turn, B's backchannel and B's turn start
    assert len(a["audio"]) == int(4.2 * ami.SR) and a["start"] == 0.0 and a["speaker"] == 0
    assert a["onset_frame"] == 0 and a["turn_end_frame"] == 28
    assert list(np.nonzero(a["hes"])[0]) == list(range(10, 17)) and a["n_hesitations"] == 1
    assert a["spk_targets"][13:15, 1].all() and a["spk_targets"][32:49, 1].all()
    b = ex[1]  # window [1.2, 5.9]: starts after B's own backchannel, ends at turn end + trail_sec
    assert b["start"] == pytest.approx(1.2) and len(b["audio"]) == int(round(4.7 * ami.SR))
    assert (b["onset_frame"], b["turn_end_frame"]) == (17, 34) and b["speaker"] == 1
    assert b["spk_targets"][2:13, 1].all()  # A (arrives 1.4 s = frame 2) is column 1
    batch = Collate(None)(ex[:2])
    assert batch["spk_act"].shape == batch["eot"].shape == batch["hes"].shape
    assert batch["spk_targets"].shape[2] == 4 and batch["speaker"].tolist() == [0, 1]


def test_asr_segments(root):
    ds = ami.AMI(["TST001"], root)
    seg = ds.asr(min_sec=0.5, max_sec=15.0)
    texts = {s["text"] for s in seg}
    assert {"hello there", "fine thanks a lot", "okay great then"} <= texts and "yeah" not in texts
    for s in seg:
        assert len(s["vad"]) == len(s["eou"]) == ToneLanguage.n_frames(len(s["audio"]))


def test_recipe_synthetic_fallback():
    for mode in ("turn", "diar", "asr"):
        data = ami.recipe_data({"data": {"ami": {"mode": mode}, "synthetic": {"n_train": 3}}}, "train")
        assert len(data) == 3


REAL = ami.DEFAULT_ROOT / "cache" / f"{ami.DEFAULT_MEETINGS['dev'][0]}.f32.npy"


@pytest.mark.skipif(not REAL.exists(), reason="AMI not prepared (scripts/research/prepare_ami.py)")
def test_real_meeting():
    m = ami.DEFAULT_MEETINGS["dev"][0]
    ds = ami.AMI([m])
    assert len(ds.speaker_ids) > 150
    for mode in ("turn", "diar", "asr"):
        ex = ds.examples(mode)
        assert len(ex) > 10
        for e in ex[:50]:
            T = ToneLanguage.n_frames(len(e["audio"]))
            for k in ("spk_targets", "spk_act", "eot", "vad"):
                if k in e:
                    assert len(e[k]) == T
            assert len(e["audio"]) <= 20 * ami.SR + 1
