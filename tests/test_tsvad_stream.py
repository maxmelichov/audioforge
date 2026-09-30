"""serve --turn-input tsvad (audioforge/tsvad_stream.py; research/archive/IMPROVEMENTS.md section 1): the live TS-VAD track,
its voice-print arming, and the server path that feeds it to the speaker-conditioned turn head."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.heads.tsvad import TSVADHead
from audioforge.serve import TSVAD_DYN, Session, SessionConfig, validate
from audioforge.tsvad_stream import ADAPT_BLEND, TSVADTrack, embed_frames, track_probs

ROOT = Path(__file__).resolve().parents[1]


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _h()


def _tsvad(d=32, emb=8):
    torch.manual_seed(0)
    h = TSVADHead(d, emb_dim=emb, hidden=16).eval()
    for p in h.parameters():
        p.data.normal_(0, 0.3)
    return h


def _spk_model():
    """The tiny serve test model plus a single-tap speaker head (layer 1 of 2) with an 8-d embedding."""
    from audioforge.model import SpeechModel
    from audioforge.tokenizer import CharTokenizer
    torch.manual_seed(0)
    base = H._asr_model()
    cfg = dict(base.cfg)
    cfg["heads"] = {**cfg["heads"], "spk": {"type": "speaker", "num_speakers": 5, "emb_dim": 8, "from_layers": [1]}}
    m = SpeechModel(cfg, CharTokenizer(list("abc "))).eval()
    m.load_state_dict(base.state_dict(), strict=False)
    m.heads["rnnt"].max_symbols = 1
    with torch.no_grad():
        m.heads["vad"].net[-1].bias.fill_(5.0)
        m.heads["turn"].out.bias.fill_(8.0)
    return m


def test_track_with_print_equals_offline_decode():
    h = _tsvad()
    x = torch.randn(1, 23, 32)
    e = np.random.default_rng(0).standard_normal(8).astype(np.float32)
    tr = TSVADTrack(h, None)
    tr.set_print(e)
    out = np.concatenate([tr.feed(x[:, i: i + 2], np.zeros(2)[: x[:, i: i + 2].shape[1]]) for i in range(0, 23, 2)])
    ref = h.decode(x, None, torch.as_tensor(e)[None])[0].numpy()
    assert np.allclose(out, ref, atol=1e-6) and tr.enrolled and tr.events[0]["source"] == "explicit"


def test_arm_collects_speech_frames_then_swaps_the_print():
    from audioforge.heads.audio import SpeakerHead
    h = _tsvad()
    torch.manual_seed(1)
    spk = SpeakerHead(32, num_speakers=3, emb_dim=8).eval()
    x = torch.randn(1, 30, 32)
    vad = np.array([0.1] * 5 + [0.9] * 25)
    tr = TSVADTrack(h, spk, print_s=10 * 0.08, adapt_s=0)
    tr.arm(3)
    out = tr.feed(x, vad)
    # frames 5..14 are the first 10 speech frames at or after frame 3: the print is taken at the end of frame 14
    e = embed_frames(spk, x[0, 5:15].numpy())
    assert np.allclose(tr.print, e) and tr.events == [{"t": 1.2, "seconds": 0.8, "source": "arm"}]
    st = h.init_stream(None)
    a = h.step(x[:, :15], st)
    h.set_enrollment(st, e)
    b = h.step(x[:, 15:], st)
    assert np.allclose(out, torch.cat([a, b], 1)[0].numpy(), atol=1e-6)


def test_refresh_reembeds_confident_target_speech():
    from audioforge.heads.audio import SpeakerHead
    h = _tsvad()
    with torch.no_grad():
        h.out.bias.copy_(torch.tensor([20.0, -20.0]))  # P(target) ~ 1, P(other) ~ 0 on every frame
    spk = SpeakerHead(32, num_speakers=3, emb_dim=8).eval()
    tr = TSVADTrack(h, spk, print_s=0.8, refresh_s=0.8)
    tr.set_print(np.ones(8, np.float32))
    tr.feed(torch.randn(1, 25, 32), np.ones(25))
    assert [ev["source"] for ev in tr.events] == ["explicit", "refresh", "refresh"]  # frames 9 and 19


def _accepting_head(other: float = -20.0):
    h = _tsvad()
    with torch.no_grad():
        h.out.bias.copy_(torch.tensor([20.0, other]))  # P(target) ~ 1 on every frame; P(other) ~ sigmoid(other)
    return h


def test_adaptation_blends_the_enrolled_print_with_accepted_speech():
    from audioforge.heads.audio import SpeakerHead
    h = _accepting_head()
    torch.manual_seed(2)
    spk = SpeakerHead(32, num_speakers=3, emb_dim=8).eval()
    x = torch.randn(1, 30, 32)
    e0 = np.ones(8, np.float32) / np.sqrt(8)
    tr = TSVADTrack(h, spk, print_s=10 * 0.08, adapt_s=12 * 0.08, adapt_min_s=0.08)
    tr.set_print(e0)
    tr.feed(x[:, :12], np.ones(12))
    # 12 accepted frames -> one adaptation over the most recent 10 (print_s) of them, anchored to the enrolled print
    want = ADAPT_BLEND * e0 + (1 - ADAPT_BLEND) * embed_frames(spk, x[0, 2:12].numpy())
    assert tr.n_adapt == 1 and np.allclose(tr.print, want / np.linalg.norm(want), atol=1e-6)
    assert np.allclose(tr.anchor, e0) and [ev["source"] for ev in tr.events] == ["explicit"]  # silent
    tr.feed(x[:, 12:], np.ones(18))
    want = ADAPT_BLEND * e0 + (1 - ADAPT_BLEND) * embed_frames(spk, x[0, 14:24].numpy())
    assert tr.n_adapt == 2 and np.allclose(tr.print, want / np.linalg.norm(want), atol=1e-6)  # never compounds
    # a new enrolment is the new anchor and restarts the collection
    tr.set_print(-e0)
    assert np.allclose(tr.anchor, -e0) and tr.buf == [] and tr.since_refresh == 0


def test_adaptation_needs_the_target_alone_and_yields_to_refresh():
    from audioforge.heads.audio import SpeakerHead
    spk = SpeakerHead(32, num_speakers=3, emb_dim=8).eval()
    e0 = np.ones(8, np.float32) / np.sqrt(8)
    tr = TSVADTrack(_accepting_head(other=20.0), spk, adapt_s=0.08, adapt_min_s=0.08)  # P(other) ~ 1: overlap
    tr.set_print(e0)
    tr.feed(torch.randn(1, 20, 32), np.ones(20))
    assert tr.n_adapt == 0 and np.allclose(tr.print, e0)
    tr = TSVADTrack(_accepting_head(), spk, print_s=0.8, refresh_s=0.8, adapt_s=0.08)
    assert tr.adapt == 0  # --tsvad-refresh-s replaces the adaptation
    assert TSVADTrack(_accepting_head(), None).adapt == 0  # no speaker head: nothing to embed with


def test_track_probs_without_adaptation_is_the_offline_decode():
    from audioforge.heads.audio import SpeakerHead
    h = _tsvad()
    spk = SpeakerHead(32, num_speakers=3, emb_dim=8).eval()
    x = torch.randn(1, 40, 32)
    e = np.random.default_rng(1).standard_normal(8).astype(np.float32)
    ref = h.decode(x, None, torch.as_tensor(e)[None])[0].numpy()
    assert np.allclose(track_probs(h, spk, x[0].numpy(), e, adapt_s=0), ref, atol=1e-6)
    assert track_probs(h, spk, x[0].numpy(), e).shape == (40, 2)


def test_print_is_clean():
    from audioforge.enrollment import print_is_clean
    speech = np.r_[np.full(55, 0.9), np.full(5, 0.1)]
    assert print_is_clean(speech) == (True, {"speech": 0.917})
    assert not print_is_clean(np.full(60, 0.3))[0]  # the user is not speaking
    ok, q = print_is_clean(speech, np.r_[np.full(37, 0.8), np.full(23, 0.0)])  # the other party talks over it (tb_160)
    assert not ok and q == {"speech": 0.917, "other": 0.617}
    assert print_is_clean(speech, np.full(60, 0.05))[0]
    assert not print_is_clean([])[0]


def _engine(**kw):
    m = _spk_model()
    return H.EnergyEngine(m, H._diar_model(), name="tiny", threads=1, turn_input="tsvad", tsvad=_tsvad(), **kw)


def _run(s, x, fs=320, arm=None):
    msgs = []
    for i in range(0, len(x), fs):
        if arm is not None and i == arm[0]:
            s.arm_enrollment(*arm[1:])
        msgs += s.process(x[i:i + fs])
    msgs += s.finish()
    return msgs


def test_session_tsvad_explicit_print_feeds_the_turn_head():
    eng = _engine(enroll="explicit")
    assert eng.embedder is None and eng.turn_input == "tsvad" and eng.dyn_offset == TSVAD_DYN[1]
    x = H._speech_silence()
    s = Session(eng, SessionConfig(turn_policy="head"))
    e = np.random.default_rng(3).standard_normal(8).tolist()
    msgs = _run(s, x, arm=(0, "enroll", 0, e))
    for m in msgs:
        validate(m)
    vp = [m for m in msgs if m["type"] == "voiceprint"]
    assert len(vp) == 1 and vp[0]["source"] == "explicit"
    # the track equals the offline head on the session's own block-1 frames with that print
    n = s.asr.n_frames
    assert len(s.asr.tsvad_p) == n and s.asr.pending == type(s.asr.pending)()
    eots = [m["eot"] for m in msgs if m["type"] == "frame"]
    assert any(v is not None for v in eots)
    st = [m for m in msgs if m["type"] == "stats"][0]
    assert st["enroll"] == "explicit" and st["enrolled"] is True and st["primary_column"] is None


def test_session_tsvad_hybrid_dyn_uses_its_own_point_and_arm_after_agent():
    eng = _engine(enroll="after_agent_arm", silero=__file__)  # any existing path; a stand-in model is injected
    eng.silero_model = type("FakeSilero", (), {"new_state": lambda self: None})()  # a fresh clone has no Silero file
    s = Session(eng, SessionConfig(turn_policy="hybrid_dyn"))
    eng.silero_model = None
    assert s.cfg.theta == TSVAD_DYN[0] and s.head_pol.thr == TSVAD_DYN[0]
    s2 = Session(eng, SessionConfig(turn_policy="hybrid_dyn", eot_threshold=0.9))
    assert s2.cfg.theta == 0.9


def test_session_tsvad_after_agent_arm_prints_from_live_speech():
    eng = _engine(enroll="after_agent_arm", tsvad_print_s=0.8)
    s = Session(eng, SessionConfig(turn_policy="head"))
    msgs = _run(s, H._speech_silence(), arm=(0, "agent_end", 0))
    vp = [m for m in msgs if m["type"] == "voiceprint"]
    assert len(vp) == 1 and vp[0]["source"] == "arm" and vp[0]["seconds"] == 0.8
    for m in msgs:
        validate(m)
    assert not s.arm_enrollment("enroll", 0)  # not this mode's trigger


def test_turn_input_tsvad_rejects_a_model_without_a_speaker_tap():
    with pytest.raises(ValueError):
        H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, turn_input="tsvad", tsvad=_tsvad())


def test_diar_off_columns_are_the_tsvad_track():
    eng = _engine(enroll="explicit", diar_off=True)
    s = Session(eng, SessionConfig(turn_policy="hybrid_dyn"))
    eng.silero_model = None
    from audioforge.tsvad_stream import TSVADColumns
    assert isinstance(s.diar, TSVADColumns) and eng.lag == (0.0, 80.0, 40.0)
    s2 = Session(eng, SessionConfig(turn_policy="timeout"))
    msgs = _run(s2, H._speech_silence(), arm=(0, "enroll", 0, [0.1] * 8))
    for m in msgs:
        validate(m)
    n = len(s2.rows)
    assert n == s2.asr.n_frames
    for v in range(n):
        assert np.allclose(s2.rows[v][:2], s2.asr.tsvad_p[v]) and not s2.rows[v][2:].any()
    with pytest.raises(ValueError):
        H.EnergyEngine(_spk_model(), H._diar_model(), name="tiny", threads=1, diar_off=True)


def test_diar_off_binds_the_timeout_primary_to_the_target_column_and_hybrid_theta():
    eng = _engine(enroll="explicit", diar_off=True)
    s = Session(eng, SessionConfig(turn_policy="hybrid"))
    assert s.cfg.theta == TSVAD_DYN[0]
    msgs = _run(s, H._speech_silence(), arm=(0, "enroll", 0, [0.1] * 8))
    assert all(m["primary"] == 0 for m in msgs if m["type"] == "frame" and m["primary"] is not None)
