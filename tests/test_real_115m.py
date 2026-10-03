"""The shipped 115M v0.4 stack for real, on CPU (audioforge.load(): stage1_served_v4 + tsvad_spk.pt + lid_115m_v2.pt,
single mode) on the bundled two-party clip (examples/audio/two_party_call_16s.wav, the user's turn ends at 10.8 s):
turn decisions, finals, the speech head, LID v2, the recorded-frame fixtures and the 1.12 s dual-rate final. Skipped
only when the shipped models are absent (audioforge-download); ~40 s in all. `chore test-real` runs these with the
RUN_REAL integration tests."""
import json

import numpy as np
import pytest

from tests.conftest import CLIP, CLIP_META, CLIP_PRINT, ROOT

pytestmark = pytest.mark.real
META = json.loads(CLIP_META.read_text())
S0, E1 = META["user_intervals"][0][0], META["user_turn_ends"][0]  # the user's turn: 3.9 .. 10.8 s
TOL = 0.08  # one frame
_CACHE = {}


def _events(fe, frames=False):
    """Every message of the clip fed as int16 20 ms blocks (a microphone) with the stored voice print enrolled."""
    import soundfile as sf
    key = (id(fe), frames)
    if key not in _CACHE:
        pcm, sr = sf.read(str(CLIP), dtype="int16")
        s = fe.session(sample_rate=sr, frames=frames)
        s.enroll(json.loads(CLIP_PRINT.read_text()))
        out = []
        for i in range(0, len(pcm), sr // 50):
            out += s.feed(pcm[i:i + sr // 50])
        _CACHE[key] = out + s.end()
    return _CACHE[key]


def test_e2e_turn_end_after_the_user_turn_and_a_final(real_115m):
    from audioforge.serve import validate
    ev = _events(real_115m)
    for m in ev:
        validate(m)
    te = [m["t"] for m in ev if m["type"] == "turn_end"]
    assert not any(S0 <= t < E1 - TOL for t in te), te  # never inside the user's turn
    assert any(E1 - TOL <= t <= E1 + 2.0 for t in te), te  # and within 2 s after it
    finals = [m for m in ev if m["type"] == "final"]
    assert any(len(m["text"].split()) >= 3 for m in finals), finals
    assert real_115m.engine.asr.encoder.d_model == 512 and real_115m.engine.name == "stage1_served_v4"


def test_speech_head_present_and_tracks_speech(real_115m):
    """heads.speech (v0.4) is the client's frame `vad`: in [0, 1], high inside the clip's annotated speech, low in
    its pauses."""
    assert "speech" in real_115m.engine.asr.heads
    fr = [m for m in _events(real_115m, frames=True) if m["type"] == "frame"]
    assert len(fr) >= 190  # 16 s / 80 ms
    t, v = np.array([m["t"] for m in fr]), np.array([m["vad"] for m in fr])
    assert ((v >= 0) & (v <= 1)).all() and np.isfinite(v).all()
    ivs = META["user_intervals"] + META["other_intervals"]
    inside = np.array([any(a + 0.16 <= x <= b for a, b in ivs) for x in t])  # 160 ms clear of each onset
    outside = np.array([all(x < a - 0.24 or x > b + 0.4 for a, b in ivs) for x in t])  # clear of every span
    assert inside.sum() > 50 and outside.sum() > 5
    assert v[inside].mean() > 0.7 and v[outside].mean() < v[inside].mean() - 0.3, (v[inside].mean(),
                                                                                 v[outside].mean())


def test_lid_v2_language_event(real_115m):
    """The distilled LID head v2 (assets/lid_115m_v2.pt) attaches to the 115M encoder and names English."""
    from audioforge.lid import resolve_head
    e = real_115m.engine
    if e.lid_name is None:
        pytest.skip("lid_115m_v2.pt is absent (public snapshot): LID is off")
    assert resolve_head("head").endswith("lid_115m_v2.pt") and e.lid_name in e.asr.heads
    ev = _events(real_115m)
    lang = [m for m in ev if m["type"] == "language"]
    assert lang and lang[0]["language"] == "en"
    assert 0.0 < lang[0]["confidence"] <= 1.0 and e.lid_min_ms / 1000 <= lang[0]["t"] <= 16.0
    st = [m for m in ev if m["type"] == "stats"]
    assert st and st[-1]["lang"] == "en"


def test_v04_rerecords_the_fixture_frames(real_115m):
    """tests/fixtures/two_party_call_16s_frames.json (float delivery) was recorded on stage1_served_v2; the shipped v4
    gives the same per-frame turn p, VAD and TS-VAD P(user) / P(other) on the same session."""
    import audioforge.serve as S
    from audioforge.data import load_wav
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())
    want = fx["variants"]["float"]
    s = S.Session(real_115m.engine, S.SessionConfig(turn_policy="vad_head"))
    s.arm_enrollment("enroll", 0, embedding=json.loads(CLIP_PRINT.read_text()))
    rec, orig = [], s.asr.run_turn_on_diar

    def rtod(avail, act_fn, flush=False):
        out = orig(avail, act_fn, flush)
        tp = s.asr.tsvad_p
        rec.extend((round(s._asr_ready_t(v), 4), float(p), float(vad), float(tp[v][0]), float(tp[v][1]))
                   for v, p, vad in out)
        return out
    s.asr.run_turn_on_diar = rtod
    x = load_wav(str(CLIP), 16000).astype(np.float32)
    for i in range(0, len(x), 320):
        s.process(x[i:i + 320])
    s.finish()
    got = {k: [r[j] for r in rec] for j, k in enumerate(("t", "p", "vad", "pu", "po"))}
    assert got["t"] == want["t"]
    for k in ("p", "vad", "pu", "po"):
        assert np.abs(np.array(got[k]) - np.array(want[k])).max() < 2e-3, k


def test_dual_rate_1120_fast_finals_equal_single_rate(real_115m):
    """--final-chunk-ms 1120 ([70, 13], research/DUAL_RATE.md) on the real 115M: the fast pass is unchanged (every
    final_fast == the single-rate final, identical turn_ends and partials), and one slow final per fast final."""
    import audioforge
    from audioforge.serve import validate
    one = _events(real_115m)
    fe2 = audioforge.load(warmup=False, final_chunk_ms=1120)
    assert fe2.engine.dual and fe2.engine.asr_lookahead == 13
    assert fe2.engine.ready_msg()["final_chunk_ms"] == 1120
    two = _events(fe2)
    for m in two:
        validate(m)
    pick = lambda ev, t: [m for m in ev if m["type"] == t]  # noqa: E731
    assert pick(two, "turn_end") == pick(one, "turn_end") and pick(one, "turn_end")
    assert pick(two, "partial") == pick(one, "partial")
    fast, slow = pick(two, "final_fast"), pick(two, "final")
    assert [dict(m, type="final") for m in fast] == pick(one, "final")
    assert len(slow) == len(fast) and [m["t"] for m in slow] == [m["t"] for m in fast]
    assert all(m["source"] == "slow" for m in slow) and any(m["text"].strip() for m in slow)


def test_served_afm_equals_base_plus_heads():
    """The served .afm this checkout uses (models/ or the legacy runs/stage1_served_v4.afm) is exactly the NVIDIA base
    + assets/served_heads_v0.4.pt: its tensor hash equals the one the heads file records (build_served checks it only
    at build time)."""
    import torch

    from audioforge import hub
    from audioforge.train import load_model
    from tests.conftest import need_real_115m
    need_real_115m()
    blob = torch.load(ROOT / "assets" / hub.HEADS_FILE, map_location="cpu", weights_only=True)
    assert hub.state_hash(load_model(str(hub.find_model("asr")), "cpu").state_dict()) == blob["state_hash"]
