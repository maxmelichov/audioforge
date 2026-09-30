"""audioforge.perf: the CPU inference fast paths keep the served system's outputs (research/archive/PERFORMANCE.md section 3).

Tiny models (tests/test_serve.py helpers) stand in for the checkpoints: every *exact* option must give bit-identical
tensors / tokens / protocol messages, ``fast_subsample`` agrees to float rounding. The real-checkpoint proof is
``scripts/research/bench_serve.py compare`` on the fixed AMI clips (runs/perf.json "identical"); ``RUN_REAL=1`` runs a short
version of it here when runs/stage1_served.afm and runs/nemo_nemotron3_diar.afm exist."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge import perf
from audioforge.model import StreamingSession
from audioforge.serve import Engine, Session, SessionConfig

ROOT = Path(__file__).resolve().parents[1]


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _h()
EXACT = perf.EXACT


def _strip(msgs):
    """Protocol messages without timing (stats, first_partial) so two runs can be compared for equality."""
    return [m for m in msgs if m.get("type") != "stats"]


def _run_session(eng, x, policy="timeout", block=320):
    s = Session(eng, SessionConfig(turn_policy=policy))
    out = []
    for i in range(0, len(x), block):
        out += s.process(x[i:i + block])
    out += s.finish()
    return _strip(out), s


@pytest.fixture(autouse=True)
def _no_global_joint_cache():
    """perf.apply flips a class attribute; leave it as the test found it."""
    was = StreamingSession.cache_joint_pred
    yield
    StreamingSession.cache_joint_pred = was


def test_parse():
    assert perf.parse(None) == perf.DEFAULT and perf.parse("") == perf.DEFAULT
    assert perf.parse("none") == ()
    assert perf.parse("all") == perf.ALL
    assert perf.parse("default,-linear_t") == tuple(o for o in perf.DEFAULT if o != "linear_t")
    assert perf.parse("none,pos_cache") == ("pos_cache",)
    assert perf.parse({"linear_t": True, "pos_cache": False}) == ("linear_t",)
    with pytest.raises(ValueError):
        perf.parse("turbo")


def test_linear_t_same_values_multirow_bit_identical():
    ref, opt = H._asr_model(), H._asr_model()
    opt.load_state_dict(ref.state_dict())
    n = perf.linear_t(opt.encoder)
    assert n > 0
    for (ka, a), (kb, b) in zip(ref.state_dict().items(), opt.state_dict().items()):
        assert ka == kb and torch.equal(a, b)  # values unchanged, only the storage order
    x = torch.from_numpy(H._audio(1.1, seed=5))[None]
    lens = torch.tensor([x.shape[1]])
    e0, _ = ref.encode(x, lens)
    e1, _ = opt.encode(x, lens)
    assert torch.allclose(e0, e1, atol=1e-6), (e0 - e1).abs().max()
    if sys.platform == "darwin":  # Accelerate: the NN and NT GEMMs round identically for >= 2 rows
        assert torch.equal(e0, e1)


def _relpos_model(seed=0, talky=False):
    """The served encoder's layout on the tiny config: NeMo relative-position attention, NeMo causal subsampling,
    speaker kernels at layer 0 and a speaker-conditioned turn head (what share_subsample / pos_cache act on)."""
    from audioforge.model import SpeechModel
    from audioforge.tokenizer import CharTokenizer
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0, "pos_emb": "rel_pos", "xscaling": True,
                       "subsampling_padding": "nemo", "speaker_kernel_layers": [0]},
           "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
                     "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
                     "turn": {"type": "turn", "mode": "kernel", "use_text": True, "condition_on_speaker": True,
                              "hidden": 16, "k_tokens": 2, "text_dim": 8, "dropout": 0.0}}}
    m = SpeechModel(cfg, CharTokenizer(list("abc "))).eval()
    if talky:  # one "a" per frame, VAD on, turn head high (tests/test_serve.py _talky_asr_model)
        m.heads["rnnt"].max_symbols = 1
        with torch.no_grad():
            m.heads["rnnt"].joint.out[-1].bias[m.heads["rnnt"].blank] = -100.0
            m.heads["rnnt"].joint.out[-1].bias[4] = 100.0
            m.heads["vad"].net[-1].bias.fill_(5.0)
            m.heads["turn"].out.bias.fill_(8.0)
    return m


def test_pos_cache_bit_identical_streaming():
    ref, opt = _relpos_model(), _relpos_model()
    opt.load_state_dict(ref.state_dict())
    assert perf.pos_cache(opt.encoder) == len(opt.encoder.layers)
    x = H._audio(1.7, seed=6)
    a, b = StreamingSession(ref), StreamingSession(opt)
    for i in range(0, len(x), 640):
        a.feed(x[i:i + 640])
        b.feed(x[i:i + 640])
    a.feed([], final=True)
    b.feed([], final=True)
    assert a.tokens == b.tokens
    for k in a.frame_events:
        assert a.frame_events[k] == b.frame_events[k]
    att = next(m for m in opt.encoder.modules() if hasattr(m, "_pos_cache"))
    assert len(att._pos_cache) >= 1  # the cache was used


def test_joint_cache_identical_tokens():
    m = H._talky_asr_model()
    x = H._audio(1.5, seed=7)
    toks = []
    for on in (False, True):
        StreamingSession.cache_joint_pred = on
        s = StreamingSession(m)
        for i in range(0, len(x), 480):
            s.feed(x[i:i + 480])
        s.feed([], final=True)
        toks.append(list(s.tokens))
        if on:
            assert s._pg is not None
    assert toks[0] == toks[1] and len(toks[0]) > 0


def test_share_subsample_identical_messages_turn_input_diar():
    asr_a, asr_b, diar = _relpos_model(), _relpos_model(), H._diar_model()
    asr_b.load_state_dict(asr_a.state_dict())
    ea = Engine(asr_a, diar, name="tiny", threads=1, turn_input="diar", perf="none")
    eb = Engine(asr_b, diar, name="tiny", threads=1, turn_input="diar", perf="none")
    assert perf.share_subsample(eb.asr.encoder) is True
    x = H._speech_silence(seed=3)
    ma, _ = _run_session(ea, x)
    mb, sb = _run_session(eb, x)
    assert ma == mb
    memo = eb.asr.encoder._sub_memo
    assert memo.hits > 0 and any(m["type"] == "frame" and m["eot"] is not None for m in mb)


def test_fast_subsample_close():
    ref, opt = H._asr_model(), H._asr_model()
    opt.load_state_dict(ref.state_dict())
    assert perf.fast_subsample(opt.encoder) == 2
    x = torch.from_numpy(H._audio(0.9, seed=8))[None]
    lens = torch.tensor([x.shape[1]])
    e0, _ = ref.encode(x, lens)
    e1, _ = opt.encode(x, lens)
    assert torch.allclose(e0, e1, atol=1e-5), (e0 - e1).abs().max()


@pytest.mark.parametrize("policy", ["timeout", "hybrid"])
def test_apply_exact_set_keeps_every_message(policy):
    asr_a, asr_b = _relpos_model(talky=True), _relpos_model(talky=True)
    asr_b.load_state_dict(asr_a.state_dict())
    diar_a, diar_b = H._diar_model(), H._diar_model()
    diar_b.load_state_dict(diar_a.state_dict())
    ea = Engine(asr_a, diar_a, name="tiny", threads=1, turn_input="diar", perf="none")
    eb = Engine(asr_b, diar_b, name="tiny", threads=1, turn_input="diar", perf="none")
    x = H._speech_silence(seed=4)
    ma, _ = _run_session(ea, x, policy)
    info = perf.apply(eb.asr, eb.diar, EXACT)
    assert info["linear_t"] > 0 and info["pos_cache"] > 0 and info["share_subsample"] and info["joint_cache"]
    mb, _ = _run_session(eb, x, policy)
    assert len(ma) == len(mb) and any(m["type"] == "partial" and m["text"] for m in ma)
    if sys.platform == "darwin":
        assert ma == mb
    else:  # other BLAS: decisions and text identical, probabilities to rounding
        for a, b in zip(ma, mb):
            assert a["type"] == b["type"]
            for k in a:
                if isinstance(a[k], float):
                    assert abs(a[k] - b[k]) < 1e-4, (k, a, b)
                elif k != "speakers":
                    assert a[k] == b[k], (k, a, b)


def _stream_tokens_and_frames(m, x, ctx):
    with ctx:
        s = StreamingSession(m)
        for i in range(0, len(x), 640):
            s.feed(x[i:i + 640])
        s.feed([], final=True)
        enc, _ = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]))
    return list(s.tokens), {k: list(v) for k, v in s.frame_events.items()}, enc.clone()


def test_default_set_under_inference_mode_bit_identical():
    """The cached tensors of the fast paths must work under torch.inference_mode (a weight or cache created there
    has no version counter) and give the same output as under no_grad, whichever mode fills the cache first."""
    x = H._audio(1.7, seed=9)
    ref = _relpos_model()
    perf.apply(ref, torch.nn.Module(), EXACT)  # the optimised path under no_grad is the reference
    ref_out = _stream_tokens_and_frames(ref, x, torch.no_grad())
    assert ref_out[0] == _stream_tokens_and_frames(_relpos_model(), x, torch.no_grad())[0]  # same tokens as unoptimised
    for first, second in ((torch.inference_mode(), torch.no_grad()), (torch.no_grad(), torch.inference_mode())):
        m = _relpos_model()
        m.load_state_dict(ref.state_dict())  # (values are layout-independent)
        with first:  # apply inside the mode too: linear_t then creates inference-tensor weights
            info = perf.apply(m, torch.nn.Module(), EXACT)
        assert info["pos_cache"] == 2
        for ctx in (first, second, first):
            toks, ev, enc = _stream_tokens_and_frames(m, x, ctx)
            assert toks == ref_out[0] and ev == ref_out[1]
            assert torch.equal(enc, ref_out[2])
        att = next(mod for mod in m.encoder.modules() if hasattr(mod, "_pos_cache"))
        assert len(att._pos_cache) >= 1 and not any(p.is_inference() for p in att._pos_cache.values())


def test_engine_default_is_exact_set():
    eng = Engine(_relpos_model(), H._diar_model(), name="tiny", threads=1, turn_input="diar")
    assert eng.perf_opts == perf.DEFAULT and set(perf.DEFAULT) <= set(perf.EXACT)
    assert eng.perf_info["share_subsample"] and eng.perf_info["pos_cache"] == 2 and eng.perf_info["linear_t"] > 0
    off = Engine(_relpos_model(), H._diar_model(), name="tiny", threads=1, turn_input="diar", perf="none")
    assert off.perf_opts == () and off.perf_info == {"joint_cache": False}


@pytest.mark.skipif(not os.environ.get("RUN_REAL"), reason="RUN_REAL=1: served checkpoints, ~1 min")
def test_real_checkpoints_exact_set_identical_messages():
    asr, diar = ROOT / "runs/stage1_served.afm", ROOT / "runs/nemo_nemotron3_diar.afm"
    if not (asr.exists() and diar.exists()):
        pytest.skip("served checkpoints not present")
    import soundfile as sf
    wav = ROOT / "examples/audio/two_speakers_10s.wav"
    x, sr = sf.read(str(wav), dtype="float32")
    assert sr == 16000
    x = x if x.ndim == 1 else x.mean(1)
    outs = []
    for opts in ((), EXACT):
        eng = Engine.load(str(asr), str(diar), "cpu", diar_pool="max", diar_spks=4, threads=2, diar_left=1)
        perf.apply(eng.asr, eng.diar, opts)
        outs.append(_run_session(eng, x)[0])
        del eng
    assert outs[0] == outs[1]
    assert any(m["type"] == "final" and m["text"] for m in outs[0])
    np.testing.assert_equal(len(outs[0]), len(outs[1]))
