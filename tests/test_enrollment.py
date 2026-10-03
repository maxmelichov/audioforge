"""Voice enrollment (audioforge/enrollment.py).

Under test:
  - ColumnEmbedder == SpeakerHead.embed on unmasked frames; recent_embeddings == a brute-force look-back;
  - causality: changing the future (tracks, features, and audio through a tiny causal SpeechModel) never changes the
    binding before it;
  - no label access: mutating the labels (spk_act, onset, turn end) leaves voice_first / voice_dominant bit-identical
    (and does change the oracle-identity upper bound, so the mutation reaches the code);
  - hysteresis: margin, hold, the silent-current self-similarity rule;
  - agreement / rebinds on a hand-made case; eval_stage1 reads a stored binding as a per-frame primary.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.enrollment import (
    VOICE_DEFAULTS,
    VOICE_MODES,
    ColumnEmbedder,
    agreement,
    enroll_voice,
    follow_voice,
    rebinds,
    recent_embeddings,
    speaker_frames,
)
from audioforge.heads.audio import SpeakerHead

ROOT = Path(__file__).parent.parent


def _script(name):
    path = ROOT / "scripts" / "research" / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path.name} is a research driver, not part of this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


def _head(D=16, E=8):
    torch.manual_seed(0)
    h = SpeakerHead(D, num_speakers=5, emb_dim=E)
    h.emb[1].running_mean.normal_(); h.emb[1].running_var.uniform_(0.5, 2.0)
    return h.eval()


def _scene(T=160, S=4, D=16, seed=0):
    """Two voices (feature means) over 4 columns with a column swap: voice A speaks in column 0, then (after a pause)
    in column 2; voice B in column 1."""
    rng = np.random.default_rng(seed)
    va, vb = rng.normal(0, 1, D), rng.normal(0, 1, D)
    feats = rng.normal(0, 0.3, (T, D)).astype(np.float32)
    p = np.full((T, S), 0.05, np.float32)
    spk = np.zeros(T, np.float32)
    for a, b, c, v, is_a in ((5, 40, 0, va, 1), (45, 70, 1, vb, 0), (80, 120, 2, va, 1), (125, 150, 1, vb, 0)):
        feats[a:b] += v
        p[a:b, c] = 0.95
        spk[a:b] = is_a
    return feats, p, spk


def test_embedder_matches_speaker_head():
    h = _head()
    x = torch.randn(3, 20, 16)
    ref = h.embed(x, torch.tensor([20, 20, 20])).detach().numpy()
    got = ColumnEmbedder(h)(x.numpy(), np.ones((3, 20), bool))
    np.testing.assert_allclose(got, ref, atol=1e-5)
    # a mask = the frame subset
    m = np.zeros((1, 20), bool); m[0, [2, 5, 6, 11]] = True
    np.testing.assert_allclose(ColumnEmbedder(h)(x[:1].numpy(), m),
                               ColumnEmbedder(h)(x[:1, [2, 5, 6, 11]].numpy(), np.ones((1, 4), bool)), atol=1e-5)


def test_recent_embeddings_brute_force():
    feats, p, _ = _scene()
    emb = ColumnEmbedder(_head())
    e, ok = recent_embeddings(feats, p, emb, win=25, min_frames=8)
    on = p > 0.5
    for t in (0, 10, 13, 30, 60, 100, 159):
        for c in range(4):
            fr = [u for u in range(max(0, t - 25), t) if on[u, c]]
            assert ok[t, c] == (len(fr) >= 8)
            if ok[t, c]:
                np.testing.assert_allclose(e[t, c], emb(feats[fr][None], np.ones((1, len(fr)), bool))[0], atol=1e-5)
            else:
                assert not e[t, c].any()


def test_follows_a_column_swap():
    """Voice A moves from column 0 to column 2: the voice binding follows it (with an embedder that separates the two
    voices: mean pooling), and never binds B's column."""
    feats, p, spk = _scene()

    def mean_emb(x, v):
        x, v = np.asarray(x, np.float32), np.asarray(v, bool)
        m = (x * v[..., None]).sum(1) / np.maximum(v.sum(1, keepdims=True), 1)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    col, info = enroll_voice(p, feats, mean_emb, "voice_first")
    assert info["c0"] == 0 and info["enrolled_at"] == 5 + 19
    assert (col[:5] == -1).all() and (col[5:80] == 0).all()
    assert col[-1] == 2 and info["n_switches"] == 1
    assert not (col == 1).any()
    sw = int(np.nonzero(col == 2)[0][0])
    assert 80 + 8 <= sw <= 80 + 8 + VOICE_DEFAULTS["hold"] + 2  # min_frames of speech, then `hold` frames


def test_causality_tracks_and_features():
    feats, p, spk = _scene(seed=1)
    emb = ColumnEmbedder(_head())
    rng = np.random.default_rng(5)
    for mode in VOICE_MODES:
        kw = dict(ref=spk, oracle_col=0, causal_init=None)
        c1, _ = enroll_voice(p, feats, emb, mode, **kw)
        for t0 in (30, 90, 130):
            f2, p2, s2 = feats.copy(), p.copy(), spk.copy()
            f2[t0:] = rng.normal(0, 3, f2[t0:].shape)
            p2[t0:] = rng.uniform(0, 1, p2[t0:].shape)
            s2[t0:] = rng.integers(0, 2, len(s2) - t0)
            c2, _ = enroll_voice(p2, f2, emb, mode, **dict(kw, ref=s2))
            np.testing.assert_array_equal(c1[:t0], c2[:t0])


def _tiny_model():
    from audioforge.model import SpeechModel
    from audioforge.tokenizer import train_tokenizer
    tok = train_tokenizer("char", ["abcdefghijklmnopqrstuvwxyz0123456789' "])
    cfg = {"name": "tiny", "preprocessor": {"n_mels": 80, "normalize": "NA"},
           "encoder": {"d_model": 32, "n_layers": 3, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                       "conv_norm": "layer", "att_context_size": [70, 1], "att_context_sizes": [[70, 1]]},
           "heads": {"ctc": {"type": "ctc", "weight": 0},
                     "spk": {"type": "speaker", "num_speakers": 10, "from_layers": "all", "weight": 0.3}}}
    torch.manual_seed(0)
    return SpeechModel(cfg, tok).eval()


def test_causality_through_the_encoder():
    """Audio from frame t0 on (80 ms frames) changes no speaker feature of frames <= t0 - 2 (one frame of look-ahead)
    and therefore no voice binding of frames < t0."""
    m = _tiny_model()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 16000 * 8).astype(np.float32)
    f1 = speaker_frames(m, [x])[0]
    T = len(f1)
    p = np.full((T, 4), 0.05, np.float32); p[3:40, 0] = 0.9; p[45:80, 1] = 0.9; p[60:T, 2] = 0.9
    emb = ColumnEmbedder(m.heads["spk"])
    for t0 in (30, 55, 75):
        x2 = x.copy()
        x2[t0 * 1280:] = rng.normal(0, 0.5, len(x2) - t0 * 1280)
        f2 = speaker_frames(m, [x2])[0]
        np.testing.assert_allclose(f1[: t0 - 1], f2[: t0 - 1], atol=1e-5)
        p2 = p.copy(); p2[t0:] = rng.uniform(0, 1, p2[t0:].shape)
        for mode in ("voice_first", "voice_dominant"):
            c1, _ = enroll_voice(p, f1, emb, mode, min_frames=4, enroll_frames=6, dom_frames=10)
            c2, _ = enroll_voice(p2, f2, emb, mode, min_frames=4, enroll_frames=6, dom_frames=10)
            np.testing.assert_array_equal(c1[:t0], c2[:t0])


def test_no_label_access(ev):
    """eval_stage1.v2_voice_bind with mutated labels: the label-free rules are bit-identical, the oracle one moves."""
    feats, p, spk = _scene(seed=2)
    emb = ColumnEmbedder(_head())
    ex = dict(spk_act=spk, onset_frame=5, turn_end_frame=120)
    bad = dict(spk_act=np.roll(1 - spk, 17), onset_frame=50, turn_end_frame=70)
    for mode in ("voice_first", "voice_dominant"):
        a, _ = ev.v2_voice_bind(emb, ex, p, feats, mode)
        b, _ = ev.v2_voice_bind(emb, bad, p, feats, mode)
        np.testing.assert_array_equal(a, b)
    a, _ = ev.v2_voice_bind(emb, ex, p, feats, "voice_oracle")
    b, _ = ev.v2_voice_bind(emb, bad, p, feats, "voice_oracle")
    assert not np.array_equal(a, b)


def test_hysteresis():
    T, S = 40, 3
    ok = np.zeros((T, S), bool)
    e = np.array([1.0, 0.0], np.float32)

    def vec(c):
        return np.array([c, np.sqrt(1 - c * c)], np.float32)

    emb = np.zeros((T, S, 2), np.float32)
    init = np.full(T, 0, np.int64)
    # frames 0-9: col 0 at 0.90, col 1 at 0.95 (beats by < margin): stay
    ok[0:10, :2] = True; emb[0:10, 0] = vec(0.90); emb[0:10, 1] = vec(0.95)
    # frames 10-14: col 1 beats col 0 by 0.5 for 5 frames (< hold 6): stay
    ok[10:15, :2] = True; emb[10:15, 0] = vec(0.5); emb[10:15, 1] = vec(1.0)
    ok[15, :2] = True; emb[15, 0] = vec(0.9); emb[15, 1] = vec(0.9)  # interrupt: counter resets
    # frames 16-21: 6 frames of col 1 beating col 0: switch on the 6th (frame 21)
    ok[16:22, :2] = True; emb[16:22, 0] = vec(0.5); emb[16:22, 1] = vec(1.0)
    col, n = follow_voice(emb, ok, e, 0, 0, init, margin=0.1, hold=6)
    assert (col[:21] == 0).all() and col[21] == 1 and n == 1
    # silent current (frames 22-39, col 1 silent): col 2 must reach the current column's own level (mean 1.0) - margin
    ok[22:30, 2] = True; emb[22:30, 2] = vec(0.85)  # below 0.9: stay
    ok[30:40, 2] = True; emb[30:40, 2] = vec(0.95)  # >= 0.9 for 6 frames: switch at frame 35
    col, n = follow_voice(emb, ok, e, 0, 0, init, margin=0.1, hold=6)
    assert (col[21:35] == 1).all() and (col[35:] == 2).all() and n == 2
    col, n = follow_voice(emb, ok, e, 0, 0, init, margin=0.1, hold=6, silent_switch=False)  # never while silent
    assert (col[21:] == 1).all() and n == 1
    # before `start` the init binding is kept untouched
    col, _ = follow_voice(emb, ok, e, 0, 25, np.full(T, -1, np.int64), margin=0.1, hold=6)
    assert (col[:25] == -1).all() and (col[25:] == 0).all()  # never heard col 0 since binding: no silent switch


def test_agreement_and_rebinds_hand_made():
    cols = [np.array([-1, 0, 0, 1, 1]), np.array([2, 2, 2, 2, 2]), np.array([-1, -1, 3, 3, 0])]
    ref = [1, 2, 3]
    ends = [4, 5, 4]  # last turn frame = end - 1: cols 1, 2, 3 -> agree, agree, agree
    assert agreement(cols, ref, ends) == 1.0
    assert agreement(cols, ref, [2, 5, 5]) == pytest.approx(1 / 3)  # 0 != 1, 2 == 2, 0 != 3
    assert agreement(cols, ref, [2, 5, 5], mask=[True, False, True]) == 0.0
    assert [rebinds(c) for c in cols] == [2, 0, 2]


def test_stored_binding_read_by_v2_cols(ev, tmp_path):
    feats, p, spk = _scene()
    ex = dict(spk_act=spk, onset_frame=5, turn_end_frame=120, meeting="IS1008b", start=1.0,
              audio=np.zeros(len(spk) * 1280, np.float32))
    c = np.where(np.arange(len(spk)) < 80, 0, 2).astype(np.int64)
    q = ev.v2_bind_path(tmp_path, "voice_first", ex)
    q.parent.mkdir(parents=True)
    np.save(q, c)
    ev.V2_BIND_WORK[0] = tmp_path
    try:
        act, cols, prim, ct = ev._v2_cols(ex, p, "voice_first")
    finally:
        ev.V2_BIND_WORK[0] = None
    np.testing.assert_array_equal(ct, c)
    assert prim == 0
    np.testing.assert_allclose(act, p[np.arange(len(c)), c])
    np.testing.assert_allclose(cols[:, 0], act)


def test_oracle_reordered_is_the_oracle_column_in_the_per_frame_representation(ev):
    feats, p, spk = _scene()
    ex = dict(spk_act=spk, onset_frame=80, turn_end_frame=120)
    a0, c0, pr0, ct0 = ev._v2_cols(ex, p, "oracle")
    a1, c1, pr1, ct1 = ev._v2_cols(ex, p, "oracle_reordered")
    assert pr0 == 2 and pr1 == 0 and (ct0 == ct1).all()
    np.testing.assert_allclose(a0, a1)
    np.testing.assert_allclose(c1, p[:, [2, 0, 1, 3]])


# --------------------------------------------------------------------------- §9: TitaNet backend + identity rules
class FakeAudioEmbedder:
    """Stands in for TitaNetEmbedder: an embedding of a set of 80 ms frames from their audio (mean, std, 1),
    unit norm. Voices = DC offsets, so different offsets separate and the same offset matches."""
    dim = 3

    def frames(self, audio, idx_lists):
        audio = np.asarray(audio, np.float32)
        out = []
        for idx in idx_lists:
            seg = np.concatenate([audio[u * 1280:(u + 1) * 1280] for u in np.asarray(idx, np.int64)])
            v = np.array([seg.mean(), seg.std(), 1.0], np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out) if out else np.zeros((0, 3), np.float32)


def _audio_scene(T=160, seed=0):
    """Audio + track of the _scene layout: voice A (offset +0.5) in column 0 (5-40) then column 2 (80-120), voice B
    (offset -0.5) in column 1 (45-70, 125-150)."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(T * 1280) * 0.05).astype(np.float32)
    p = np.full((T, 4), 0.05, np.float32)
    spk = np.zeros(T, np.float32)
    y = np.zeros((T, 2), np.float32)  # spk_targets: column 0 = primary (A), column 1 = B
    for a, b, c, off, is_a in ((5, 40, 0, 0.5, 1), (45, 70, 1, -0.5, 0), (80, 120, 2, 0.5, 1), (125, 150, 1, -0.5, 0)):
        audio[a * 1280:b * 1280] += off
        p[a:b, c] = 0.95
        spk[a:b] = is_a
        y[a:b, 0 if is_a else 1] = 1
    return audio, p, spk, y


def test_recent_embeddings_audio_grid_equals_brute_force_and_holds_between_grid_frames():
    from audioforge.enrollment import recent_embeddings_audio
    audio, p, _, _ = _audio_scene()
    fe = FakeAudioEmbedder()
    emb, ok = recent_embeddings_audio(audio, p, fe, win=25, min_frames=8, stride=5)
    on = p > 0.5
    for t in range(len(p)):
        g = t - t % 5
        for c in range(4):
            fr = [u for u in range(max(0, g - 25), g) if on[u, c]]
            assert ok[t, c] == (len(fr) >= 8)
            if ok[t, c]:
                np.testing.assert_allclose(emb[t, c], fe.frames(audio, [fr])[0], atol=1e-6)
                np.testing.assert_array_equal(emb[t, c], emb[g, c])
            else:
                assert not emb[t, c].any()


def test_recent_embeddings_audio_is_causal():
    from audioforge.enrollment import recent_embeddings_audio
    audio, p, _, _ = _audio_scene(seed=3)
    fe = FakeAudioEmbedder()
    e1, o1 = recent_embeddings_audio(audio, p, fe, stride=5)
    rng = np.random.default_rng(1)
    for t0 in (31, 60, 97):
        a2, p2 = audio.copy(), p.copy()
        a2[t0 * 1280:] = rng.normal(0, 1, len(a2) - t0 * 1280)
        p2[t0:] = rng.uniform(0, 1, p2[t0:].shape)
        e2, o2 = recent_embeddings_audio(a2, p2, fe, stride=5)
        np.testing.assert_array_equal(e1[:t0], e2[:t0])
        np.testing.assert_array_equal(o1[:t0], o2[:t0])


def test_causal_dominant_from_matches_eval_stage1_and_forces_a_binding(ev):
    from audioforge.enrollment import causal_dominant_from
    _, p, _ = _scene(seed=4)
    ref = ev.enroll_causal_dominant(p)
    np.testing.assert_array_equal(causal_dominant_from(p), ref)
    forced = causal_dominant_from(p, force=(3, 50))
    np.testing.assert_array_equal(forced[:50], ref[:50])
    assert forced[50] == 3
    # column 3 never speaks: after > s silent frames the rule re-binds to the running dominant column
    assert (forced[50:50 + 26] == 3).all() and forced[-1] != 3


def test_agent_end_frame_hand_made():
    from audioforge.enrollment import agent_end_frame
    y = np.zeros((100, 3))
    y[60:100, 0] = 1  # primary
    y[10:40, 1] = 1  # another speaker: two runs before the onset
    y[45:55, 2] = 1
    y[80:90, 1] = 1  # after the onset: not the agent turn
    assert agent_end_frame(y, 60) == 55
    assert agent_end_frame(y, 30) == 40  # a run that starts before the onset but ends after it: its end
    assert agent_end_frame(y, 5) is None  # nobody spoke before the onset
    assert agent_end_frame(y[:, :1], 60) is None  # no other speaker at all


def test_after_prev_end_choice_and_rule():
    from audioforge.enrollment import after_prev_end_choice, enroll_voice, recent_embeddings_audio
    audio, p, spk, y = _audio_scene()
    on = p > 0.5
    # after the agent turn (column 1, ends at 70) column 2 becomes active at 80: chosen on its 3rd frame
    assert after_prev_end_choice(on, 70, min_run=3) == (2, 82)
    assert after_prev_end_choice(on, 150, min_run=3) == (None, None)
    q = p.copy(); q[72:74, 3] = 0.9  # a 2-frame blip is not a column becoming active
    assert after_prev_end_choice(q > 0.5, 70, min_run=3) == (2, 82)
    fe = FakeAudioEmbedder()
    eo = recent_embeddings_audio(audio, p, fe, stride=5)
    ci = np.full(len(p), 1, np.int64)
    col, info = enroll_voice(p, None, None, "after_prev_end", causal_init=ci, agent_end=70, emb_ok=eo,
                             enroll_embed=lambda idx: fe.frames(audio, [idx])[0])
    assert info["c0"] == 2 and info["chosen_at"] == 82 and info["enrolled_at"] == 80 + 19 and info["agent_end"] == 70
    assert (col[:82] == 1).all() and (col[82:] == 2).all()  # B's column (voice B) never takes the binding
    # fallbacks: no agent turn / no column becoming active -> the causal_init binding throughout
    col2, info2 = enroll_voice(p, None, None, "after_prev_end", causal_init=ci, agent_end=None, emb_ok=eo,
                               enroll_embed=lambda idx: fe.frames(audio, [idx])[0])
    assert info2["fallback"] == "no_agent_turn" and (col2 == ci).all()
    col3, info3 = enroll_voice(p, None, None, "after_prev_end", causal_init=ci, agent_end=150, emb_ok=eo,
                               enroll_embed=lambda idx: fe.frames(audio, [idx])[0])
    assert info3["fallback"] == "no_column" and (col3 == ci).all()


def test_voice_explicit_enrolls_on_the_primary_first_utterance():
    from audioforge.enrollment import enroll_voice, recent_embeddings_audio
    audio, p, spk, y = _audio_scene()
    fe = FakeAudioEmbedder()
    eo = recent_embeddings_audio(audio, p, fe, stride=5)
    ci = np.full(len(p), 3, np.int64)
    col, info = enroll_voice(p, None, None, "voice_explicit", ref=spk, causal_init=ci, emb_ok=eo,
                             enroll_embed=lambda idx: fe.frames(audio, [idx])[0])
    assert info["c0"] == 0 and info["enrolled_at"] == 5 + 19  # the column dominating the primary's first 19 frames
    assert (col[:24] == 3).all() and (col[24:80] == 0).all() and col[-1] == 2 and not (col == 1).any()


def test_after_prev_end_choice_reads_no_labels_through_v2_voice_bind(ev):
    """Mutating the primary's labels (spk_act / onset / end) leaves the after_prev_end bindings bit-identical as long
    as the agent end (the only label-derived input, the TTS-end stand-in) is the same; the causal control too."""
    from audioforge.enrollment import recent_embeddings_audio
    audio, p, spk, y = _audio_scene()
    fe = FakeAudioEmbedder()
    eo = recent_embeddings_audio(audio, p, fe, stride=5)
    ex = dict(spk_act=spk, spk_targets=y, onset_frame=80, turn_end_frame=120, audio=audio)
    bad = dict(ex, spk_act=np.roll(1 - spk, 17), onset_frame=79, turn_end_frame=100)
    for mode in ("after_prev_end_titanet", "after_prev_end_causal"):
        a, ia = ev.v2_voice_bind(None, ex, p, None, mode, tn=fe, emb_ok=eo)
        b, ib = ev.v2_voice_bind(None, bad, p, None, mode, tn=fe, emb_ok=eo)
        np.testing.assert_array_equal(a, b)
        assert ia["c0"] == 2 and ia["agent_end"] == 70 and ia["chosen_at"] == 82
    a, _ = ev.v2_voice_bind(None, ex, p, None, "after_prev_end_causal", tn=fe, emb_ok=eo)
    assert a[82] == 2 and a[81] == ev.enroll_causal_dominant(p)[81]
    # an onset with no earlier other-speaker turn: the fallback is causal_dominant, flagged
    c, ic = ev.v2_voice_bind(None, dict(ex, onset_frame=5), p, None, "after_prev_end_titanet", tn=fe, emb_ok=eo)
    assert ic["fallback"] == "no_agent_turn" and (c == ev.enroll_causal_dominant(p)).all()


def test_longer_enrollment_variants_parse_and_use_n_frames(ev):
    from audioforge.enrollment import recent_embeddings_audio
    audio, p, spk, y = _audio_scene()
    fe = FakeAudioEmbedder()
    eo = recent_embeddings_audio(audio, p, fe, stride=5)
    ex = dict(spk_act=spk, spk_targets=y, onset_frame=80, turn_end_frame=120, audio=audio)
    assert "after_prev_end_titanet_e40" in ev.V2_STORED_MODES and "voice_explicit_titanet_e60" in ev.V2_STORED_MODES
    _, i19 = ev.v2_voice_bind(None, ex, p, None, "after_prev_end_titanet", tn=fe, emb_ok=eo)
    _, i40 = ev.v2_voice_bind(None, ex, p, None, "after_prev_end_titanet_e40", tn=fe, emb_ok=eo)
    assert i19["enrolled_at"] == 80 + 19 and i40["enrolled_at"] == 80 + 40 and i40["c0"] == 2
    _, i60 = ev.v2_voice_bind(None, ex, p, None, "after_prev_end_titanet_e60", tn=fe, emb_ok=eo)
    assert i60["enrolled_at"] is None  # column 2 has only 40 active frames: never enrolled, stays on the choice
    _, e40 = ev.v2_voice_bind(None, ex, p, None, "voice_explicit_titanet_e40", tn=fe, emb_ok=eo)
    assert e40["enrolled_at"] == 85  # 35 primary label frames in 5-40, then frames 80-84: available from 85


def test_clean_spans_mask_pooling_and_rule():
    from audioforge.enrollment import clean_mask, consistent_mean, enroll_spans, recent_embeddings_spans, spans_of
    audio, p, spk, y = _audio_scene()
    T = len(p)
    vad = np.where(p.max(1) > 0.5, 0.95, 0.1).astype(np.float32)
    vad[20:30] = 0.5  # a VAD dip inside voice A's first stretch
    q = p.copy(); q[30:36, 3] = 0.9  # an overlap: another column active -> not admitted
    adm = clean_mask(q, vad)
    assert adm[10, 0] and not adm[25, 0] and not adm[32, 0] and not adm[32, 3] and adm[38, 0]
    assert [list(s[[0, -1]]) for s in spans_of(adm[:, 0], 0, T, 7)] == [[5, 19], [36, 39]][:1] + []  # 36-39 is 4 frames
    fe = FakeAudioEmbedder()
    # consistent_mean drops the odd voice out (B among As)
    ea = fe.frames(audio, [np.arange(5, 12), np.arange(12, 19), np.arange(80, 87)])
    eb = fe.frames(audio, [np.arange(45, 52)])
    v, n = consistent_mean(np.concatenate([ea, eb]), 0.7)  # the fake voices' cosine is 0.60: threshold above it
    assert n == 3 and float(v @ ea[0]) > float(v @ eb[0])
    assert consistent_mean(np.concatenate([ea, eb]), 0.3)[1] == 4  # below it nothing is dropped
    sp, te = enroll_spans(adm[:, 2], 40, 70, 7)  # column 2: admitted 80-120 = 40 frames in one span
    assert te == 120 and sum(len(x) for x in sp) == 40
    assert enroll_spans(adm[:, 0], 40, 0, 7) == (None, None)
    emb, ok = recent_embeddings_spans(audio, adm, fe, stride=5)
    assert ok[100, 2] and not ok[100, 1] and not ok[10, 0]  # frame 10: only 5 admitted frames before it
    np.testing.assert_allclose(emb[100, 2], fe.frames(audio, [np.arange(80, 100)])[0], atol=1e-5)  # one span, frames 80..99 (< 100, within 25)


def test_clean_modes_through_v2_voice_bind(ev):
    from audioforge.enrollment import clean_mask, recent_embeddings_spans
    audio, p, spk, y = _audio_scene()
    vad = np.where(p.max(1) > 0.5, 0.95, 0.1).astype(np.float32)
    adm = clean_mask(p, vad)
    fe = FakeAudioEmbedder()
    eo = recent_embeddings_spans(audio, adm, fe, stride=5)
    ex = dict(spk_act=spk, spk_targets=y, onset_frame=80, turn_end_frame=120, audio=audio)
    assert set(ev.V2_CLEAN_MODES) <= set(ev.V2_STORED_MODES)
    col, info = ev.v2_voice_bind(None, ex, p, None, "after_prev_end_titanet_clean", tn=fe, emb_ok=eo, clean=adm)
    assert info["c0"] == 2 and info["chosen_at"] == 82 and info["enrolled_at"] == 120 and (col[82:] == 2).all()
    col, info = ev.v2_voice_bind(None, ex, p, None, "voice_explicit_titanet_clean", tn=fe, emb_ok=eo, clean=adm)
    # the chosen column (0) has 35 clean frames only (the primary's later speech is in column 2): never enrolled,
    # the binding stays on the chosen column from the choice on (unlike the label-frame e40 row, which reaches 40)
    assert info["c0"] == 0 and info["chosen_at"] == 24 and info["enrolled_at"] is None and (col[24:] == 0).all()
