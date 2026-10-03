"""The live path's buffering fixes.

* A NeMo-aligned causal encoder (the served model) runs an attention chunk as soon as its frames' mel input is
  complete (encoder frame v ends at mel frame 8v): chunk j after 8R + 1 + j (R + 1) 8 mel frames instead of after a
  whole (R + 1) 8 window. Same outputs, every frame 70 ms earlier; ``frame_ready_samples`` / ``frames_ready`` (the
  decision clock of turn_end.t and the finals' cut) follow the new trigger exactly.
* ``--asr-chunk-ms 80`` serves the ASR pass at [L, 0] (one frame per chunk, no lookahead)."""
import numpy as np
import pytest
import torch

from audioforge.model import SpeechModel, StreamingSession
from audioforge.tokenizer import CharTokenizer

SR, BLOCK = 16000, 320  # 20 ms client blocks


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _aligned_model(seed=0, spk=False, right=1):
    """The served layout on a tiny config: NeMo causal subsampling (nemo_causal), rel-pos attention, a VAD head and a
    speaker-conditioned turn head; ``spk`` adds the single-tap speaker head that --turn-input tsvad needs."""
    torch.manual_seed(seed)
    heads = {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
             "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
             "turn": {"type": "turn", "mode": "kernel", "use_text": True, "condition_on_speaker": True,
                      "hidden": 16, "k_tokens": 2, "text_dim": 8, "dropout": 0.0}}
    if spk:
        heads["spk"] = {"type": "speaker", "num_speakers": 5, "emb_dim": 192, "from_layers": [1]}
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, right], "dropout": 0.0, "pos_emb": "rel_pos", "xscaling": True,
                       "subsampling_padding": "nemo", "speaker_kernel_layers": [0]},
           "heads": heads}
    m = SpeechModel(cfg, CharTokenizer(list("abc "))).eval()
    with torch.no_grad():  # some text: a non-blank token wins now and then
        m.heads["rnnt"].joint.out[-1].bias[m.heads["rnnt"].blank] = -1.0
    return m


def _audio(sec=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0) + 0.02 * rng.standard_normal(len(t))
            ).astype(np.float32)


def _feed_blocks(s, x):
    """Feed 20 ms blocks; -> per frame the samples received when it was first produced."""
    got = []
    for k in range(0, len(x), BLOCK):
        got += [min(k + BLOCK, len(x))] * len(s.feed_frames(x[k:k + BLOCK]))
    return got


def test_aligned_encoder_runs_the_chunk_when_its_mels_are_complete():
    from audioforge.server.streams import ASRStream
    m = _aligned_model()
    assert m.encoder.pre_encode.nemo_causal
    s = ASRStream(m, "vad", None)
    assert (s.chunk_mel, s.chunk_lead) == (16, 9)  # [8, 1]: 8R + 1 mel frames for the first chunk, then 16 each
    assert s.frame_ready_samples(0) == 8 * 160 + 256 == 1536  # 96 ms, was (16 - 1) * 160 + 256 = 2656 (166 ms)
    assert s.frame_ready_samples(2) - s.frame_ready_samples(0) == 2560  # one 160 ms chunk later
    x = _audio()
    got = _feed_blocks(s, x)
    for v, n in enumerate(got):  # each frame appears in the first 20 ms block that completes its chunk's input
        need = s.frame_ready_samples(v)
        assert n >= need and n - BLOCK < need, (v, n, need)
        assert s.frames_ready(n) >= v + 1 and s.frames_ready(need - 1) <= v


def test_early_trigger_changes_no_output():
    """Same heads, tokens and frame count as the old whole-window trigger (chunk_lead = chunk_mel), only earlier."""
    from audioforge.server.streams import ASRStream
    m = _aligned_model(seed=3)
    x = _audio(4.0, seed=1)
    new, old = ASRStream(m, "vad", None), ASRStream(m, "vad", None)
    old.chunk_lead = old.chunk_mel
    fa, fb = [], []
    for k in range(0, len(x), BLOCK):
        fa += new.feed_frames(x[k:k + BLOCK])
        fb += old.feed_frames(x[k:k + BLOCK])
    fa += new.feed_frames(np.zeros(0, np.float32), final=True)
    fb += old.feed_frames(np.zeros(0, np.float32), final=True)
    assert len(fa) == len(fb) > 40
    assert np.allclose([f["vad"] for f in fa], [f["vad"] for f in fb], atol=1e-5)
    assert new.tokens == old.tokens and new.tok_at == old.tok_at
    # and the text equals the plain StreamingSession fed in one go (the offline-equivalent path)
    ss = StreamingSession(m)
    ss.feed(x, final=True)
    assert ss.tokens == new.tokens


def test_non_aligned_encoder_keeps_the_whole_window():
    from audioforge.server.streams import ASRStream
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0},
           "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16}}}
    s = ASRStream(SpeechModel(cfg, CharTokenizer(list("abc "))).eval(), None, None)
    assert s.chunk_lead == s.chunk_mel == 16
    assert s.frame_ready_samples(0) == 15 * 160 + 256


def _single_engine(chunk_ms=None, right=1):
    from audioforge.heads.tsvad import TSVADHead
    from audioforge.serve import Engine
    torch.manual_seed(0)
    tsvad = TSVADHead(32, emb_dim=192, hidden=16).eval()
    return Engine(_aligned_model(spk=True, right=right), None, name="tiny", threads=1, turn_input="tsvad",
                  diar_off=True, tsvad=tsvad, enroll="dominant", asr_chunk_ms=chunk_ms)


def _run(sess, x):
    msgs = []
    for k in range(0, len(x), BLOCK):
        msgs.append((min(k + BLOCK, len(x)), sess.process(x[k:k + BLOCK])))
    return msgs


def test_session_frames_heads_and_turn_clock_match_the_old_trigger_but_earlier():
    """Single mode on the aligned tiny model: frame messages (vad, eot, speakers) are identical to the old trigger;
    each arrives in an earlier (or the same) block; head-path decision times are never later."""
    from audioforge.serve import Session
    eng = _single_engine()
    x = _audio(4.0, seed=2)
    a, b = Session(eng), Session(eng)
    b.asr.chunk_lead = b.asr.chunk_mel
    ra, rb = _run(a, x), _run(b, x)
    ra.append((len(x) + 1, a.finish()))
    rb.append((len(x) + 1, b.finish()))
    fa = [(n, m) for n, ms in ra for m in ms if m["type"] == "frame"]
    fb = [(n, m) for n, ms in rb for m in ms if m["type"] == "frame"]
    assert len(fa) == len(fb) > 40
    for (na, ma), (nb, mb) in zip(fa, fb):
        assert ma["t"] == mb["t"] and abs(ma["vad"] - mb["vad"]) < 1e-5
        assert na <= nb
    # the per-frame TS-VAD columns and turn-head posteriors (frame.speakers / eot carry the latest of a block's batch)
    assert len(a.rows) == len(b.rows) and np.allclose(np.stack(list(a.rows.d)), np.stack(list(b.rows.d)), atol=1e-5)
    assert np.allclose([p for _, p in a.eot_hist], [p for _, p in b.eot_hist], atol=1e-5)
    assert all(ta <= tb for (ta, _), (tb, _) in zip(a.eot_hist, b.eot_hist))
    early = sum(na < nb for (na, _), (nb, _) in zip(fa, fb))
    assert early >= len(fa) // 2  # the first frame of every chunk (and most second frames) come a block or more sooner
    assert a._asr_ready_t(0) == pytest.approx(1536 / SR) and b._asr_ready_t(0) == pytest.approx(2656 / SR)


def test_asr_chunk_ms_80_serves_attention_context_right_0():
    from audioforge.serve import Session
    eng = _single_engine(chunk_ms=80)
    assert eng.asr_att == [8, 0] and eng.chunk_ms == 80 and eng.ready_msg()["chunk_ms"] == 80
    s = Session(eng)
    assert s.asr.att == [8, 0] and s.asr.cs == 1 and (s.asr.chunk_mel, s.asr.chunk_lead) == (8, 1)
    assert s.diar.cfg.chunk_len == 1
    x = _audio(2.0)
    got = []
    for n, ms in _run(s, x):
        got += [n] * sum(m["type"] == "frame" for m in ms)
    for v, n in enumerate(got):  # frame v is computed once mel frame 8v exists: 80 v + 16 ms
        need = 1280 * v + 256
        assert s.asr.frame_ready_samples(v) == need
        assert n >= need and n - BLOCK < need, (v, n, need)
    with pytest.raises(ValueError):
        _single_engine(chunk_ms=100)


def test_asr_chunk_ms_flag_reaches_the_engine():
    from audioforge.server import cli
    ap = cli.build_parser(advanced=True)
    assert ap.parse_args(["--asr-chunk-ms", "80"]).asr_chunk_ms == 80
    assert ap.parse_args([]).asr_chunk_ms is None
    with pytest.raises(SystemExit):
        ap.parse_args(["--asr-chunk-ms", "120"])
