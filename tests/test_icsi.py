"""ICSI conversion + labels on a hand-written NXT fixture (no corpus needed) + optional real-data checks.

Fixture meeting TST001 (30 s), raw ICSI NXT format, agents A (me011), B (fe008), C (mn005):
  A: Hello there [0.0-0.8] , | you're = "you" [1.4, ?] + "'re" [?, 1.8] | right now [1.8-2.2] -> one floor turn, 1 hesitation
  B: Mm-hmm = "Mm" [1.05, ?] HYPH "hmm" [?, 1.2]      -> ONE token "mm hmm": backchannel inside A's pause
  B: two hour long: "two" [2.6, ?] "hour" [?, ?] "long" [?, 3.9]   -> chain split by characters 3:4:4
  A: Okay great then [5.0-6.2]
  C: digits "Transcript one two" untimed in segment [10, 14]         -> untimed zone [10, 14]
  C: "@" untimed + okay [20.0-20.5] in segment [18, 21]             -> zone narrowed to [18, 20]; okay = backchannel
  after the last segment end (21 s)                                  -> zone [21, inf)
"""

import numpy as np
import pytest

from audioforge.data import ToneLanguage
from audioforge.datasets import ami, icsi

W = '   <w nite:id="TST001.w.{i}" starttime="{s}" endtime="{e}" c="{c}">{t}</w>\n'
WORDS = {  # (start, end, text, class); "" = missing time
    "A": [(0.0, 0.4, "Hello", "W"), (0.4, 0.8, "there", "W"), (0.8, 0.8, ",", "CM"), (1.4, "", "you", "W"),
          ("", 1.8, "'re", "W"), (1.8, 2.0, "right", "W"), (2.0, 2.2, "now", "W"), (2.2, 2.2, ".", "."), (5.0, 5.4, "Okay", "W"),
          (5.4, 5.8, "great", "W"), (5.8, 6.2, "then", "W")],
    "B": [(1.05, "", "Mm", "W"), ("", "", "-", "HYPH"), ("", 1.2, "hmm", "W"), (2.6, "", "two", "W"),
          ("", "", "hour", "W"), ("", "", "-", "SYM"), ("", 3.9, "long", "W"), (3.9, 3.9, ".", ".")],
    "C": [("", "", "Transcript", "W"), ("", "", "one", "W"), ("", "", "two", "W"), ("", "", "@", "SYM"),
          (20.0, 20.5, "okay", "W")],
}
SEGS = {  # (start, end, first word index, last word index) into WORDS[agent]
    "A": [(0.0, 2.2, 0, 7), (5.0, 6.2, 8, 10)],
    "B": [(1.05, 1.2, 0, 2), (2.6, 3.9, 3, 7)],
    "C": [(10.0, 14.0, 0, 2), (18.0, 21.0, 3, 4)],
}
PART = {"A": "me011", "B": "fe008", "C": "mn005"}
DUR = 30.0


def _ids(agent):
    return {"A": 0, "B": 100, "C": 200}[agent]


def _words_xml(agent):
    base = _ids(agent)
    body = "".join(W.format(i=base + i, s=s, e=e, t=t, c=c) for i, (s, e, t, c) in enumerate(WORDS[agent]))
    extra = ('   <vocalsound nite:id="TST001.vocalsound.1" starttime="" endtime="" description="laugh"/>\n'
             '   <disfmarker nite:id="TST001.disfmarker.1" starttime="" endtime=""/>\n')
    return ('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
            f'<nite:root nite:id="TST001.{agent}.words" xmlns:nite="http://nite.sourceforge.net/">\n{body}{extra}</nite:root>\n')


def _segs_xml(agent):
    base = _ids(agent)
    body = "".join(
        f'   <segment nite:id="TST001.segment.{base + k}" starttime="{s}" endtime="{e}" participant="{PART[agent]}">\n'
        f'      <nite:child href="TST001.{agent}.words.xml#id(TST001.w.{base + a})..id(TST001.w.{base + b})"/>\n'
        "   </segment>\n" for k, (s, e, a, b) in enumerate(SEGS[agent]))
    return ('<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>\n'
            f'<nite:root nite:id="TST001.{agent}.segs" xmlns:nite="http://nite.sourceforge.net/">\n{body}</nite:root>\n')


@pytest.fixture()
def root(tmp_path):
    for sub in ("Words", "Segments"):
        (tmp_path / "raw" / "ICSI" / sub).mkdir(parents=True)
    for a in WORDS:
        (tmp_path / "raw" / "ICSI" / "Words" / f"TST001.{a}.words.xml").write_text(_words_xml(a))
        (tmp_path / "raw" / "ICSI" / "Segments" / f"TST001.{a}.segs.xml").write_text(_segs_xml(a))
    (tmp_path / "cache").mkdir()
    rng = np.random.default_rng(0)
    np.save(tmp_path / "cache" / "TST001.f32.npy", (0.01 * rng.standard_normal(int(DUR * ami.SR))).astype(np.float32))
    return tmp_path


def test_convert_words():
    w, z, c = icsi.convert_words(_words_xml("A"), _segs_xml("A"))
    assert w == [(0.0, 0.4, "hello"), (0.4, 0.8, "there"), (1.4, 1.8, "you're"), (1.8, 2.0, "right"),
                 (2.0, 2.2, "now"), (5.0, 5.4, "okay"), (5.4, 5.8, "great"), (5.8, 6.2, "then")] and z == []
    w, z, c = icsi.convert_words(_words_xml("B"), _segs_xml("B"))
    assert w == [(1.05, 1.2, "mm hmm"), (2.6, 2.955, "two"), (2.955, 3.427, "hour"), (3.427, 3.9, "long")]
    assert c["chain"] == 3 and c["untimed"] == 0
    w, z, c = icsi.convert_words(_words_xml("C"), _segs_xml("C"))
    assert w == [(20.0, 20.5, "okay")] and z == [(10.0, 14.0), (18.0, 20.0)]
    assert c["untimed"] == 4 and c["last_segment_end"] == 21.0


def test_convert_annotations_layout(root):
    icsi.convert_annotations(root / "raw", root / "annotations")
    ann = root / "annotations"
    assert ami.speaker_names(ann) == {"TST001": {"A": "me011", "B": "fe008", "C": "mn005"}}
    words = ami.meeting_words(ann, "TST001")  # AMI's own parser reads the converted files
    assert words["fe008"][0] == (1.05, 1.2, "mm hmm") and len(words["me011"]) == 8


def test_turns_and_stats(root):
    ds = icsi.ICSI(["TST001"], root)
    assert ds.speaker_ids == ["fe008", "me011", "mn005"]
    assert [ds.gid(s) for s in ds.speaker_ids] == [1000, 1001, 1002] and ds.gid("nobody") == -1
    t = ds.turns["TST001"]
    assert [(x["speaker"], x["start"], x["end"], x["bc"]) for x in t] == [
        ("me011", 0.0, 2.2, False), ("fe008", 1.05, 1.2, True), ("fe008", 2.6, 3.9, False),
        ("me011", 5.0, 6.2, False), ("mn005", 20.0, 20.5, True)]
    assert t[0]["text"] == "hello there you're right now" and t[0]["hes"] == [(0.8, 1.4)]
    st = ds.stats()
    assert st["turns"] == 3 and st["backchannels"] == 2 and st["speakers"] == 3
    assert st["switch_gap"]["n"] == 2 and st["switch_gap"]["p50"] == pytest.approx(0.75)
    assert st["hesitation"]["n"] == 1 and st["hesitation"]["p50"] == pytest.approx(0.6)
    assert st["untimed_frac"] == pytest.approx((4 + 2 + 9) / 30, abs=1e-3)
    assert ds.zones["TST001"] == [(10.0, 14.0), (18.0, 20.0), (21.0, float("inf"))]


def test_examples_and_filters(root):
    ds = icsi.ICSI(["TST001"], root)
    win = ds.diar(window_sec=8.0, hop_sec=4.0)  # [0,8) kept; [4,12) .. [20,28) touch a zone
    assert [w["start"] for w in win] == [0.0] and ds.filtered["diar"] == dict(total=6, untimed=5, overfull=0)
    assert win[0]["speakers"] == [1001, 1000]  # A arrives first, B second; namespaced ids
    raw = icsi.ICSI(["TST001"], root, exclude_untimed=False).diar(window_sec=8.0, hop_sec=4.0)
    assert len(raw) == 6
    ex = ds.turn_examples()
    assert [e["text"] for e in ex] == ["hello there you're right now", "two hour long", "okay great then"]
    assert [e["speaker"] for e in ex] == [1001, 1000, 1001]
    for e in ex:
        T = ToneLanguage.n_frames(len(e["audio"]))
        assert e["spk_targets"].shape == (T, 4) and len(e["spk_act"]) == len(e["eot"]) == len(e["hes"]) == T
        np.testing.assert_array_equal(e["spk_targets"][:, 0], e["spk_act"])
        assert e["eot"][: e["turn_end_frame"]].sum() == 0 and e["eot"][e["turn_end_frame"]:].all()
    assert list(np.nonzero(ex[0]["hes"])[0]) == list(range(10, 17))
    seg = ds.asr(min_sec=0.5)
    assert {"hello there", "you're right now", "two hour long", "okay great then"} <= {s["text"] for s in seg}


def test_overfull_filter(root):
    ds = icsi.ICSI(["TST001"], root)
    ex = [dict(dropped_speakers=1, start=0.0, audio=np.zeros(16000), meeting="TST001"),
          dict(dropped_speakers=0, start=0.0, audio=np.zeros(16000), meeting="TST001")]
    assert len(ds._filter("x", ex)) == 1 and ds.filtered["x"]["overfull"] == 1


def test_recipe_synthetic_and_routing():
    from audioforge.train import load_data
    for mode in ("turn", "diar", "asr"):
        cfg = {"data": {"icsi": {"mode": mode}, "synthetic": {"n_train": 3}}}
        assert len(icsi.recipe_data(cfg, "train")) == 3 and len(load_data(cfg, "train")) == 3


def test_splits():
    assert len(icsi.ALL_MEETINGS) == len(icsi.WAV_BYTES) == 75
    assert sorted(sum(icsi.SPLITS.values(), [])) == sorted(icsi.ALL_MEETINGS)
    for s, ms in icsi.DEFAULT_MEETINGS.items():
        assert set(ms) <= set(icsi.SPLITS[s])
    assert icsi.subset({"train": 14})["train"][:12] == icsi.DEFAULT_MEETINGS["train"]


REAL = icsi.DEFAULT_ROOT / "cache" / f"{icsi.DEFAULT_MEETINGS['dev'][0]}.f32.npy"


@pytest.mark.skipif(not REAL.exists(), reason="ICSI not prepared (scripts/research/prepare_icsi.py)")
def test_real_meeting():
    m = icsi.DEFAULT_MEETINGS["dev"][0]
    ds = icsi.ICSI([m])
    assert len(ds.speaker_ids) > 40
    for mode in ("turn", "diar", "asr"):
        ex = ds.examples(mode)
        assert len(ex) > 10
        for e in ex:
            T = ToneLanguage.n_frames(len(e["audio"]))
            for k in ("spk_targets", "spk_act", "eot", "vad"):
                if k in e:
                    assert len(e[k]) == T
            assert len(e["audio"]) <= 20 * ami.SR + 1
            a, b = e["start"], e["start"] + len(e["audio"]) / ami.SR
            assert not any(za < b and zb > a for za, zb in ds.zones[m])
            ids = e["speakers"] if mode == "diar" else [e["speaker"]]
            assert all(i >= icsi.SPEAKER_OFFSET for i in ids)
