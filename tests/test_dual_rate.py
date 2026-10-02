"""Dual-rate engine (research/DUAL_RATE.md, ``audioforge-serve --final-chunk-ms``): the fast pass keeps every head,
partial, turn decision and fast final byte-identical to single rate; the slow pass's finals partition the tokens of the
same encoder at the long chunk (== the offline masked forward); the turn_end flush never changes the slow stream and
does not depend on how the client cut its audio. Tiny random models only."""
import itertools

import numpy as np
import pytest
import torch

from audioforge.model import SpeechModel, StreamingSession
from audioforge.serve import Session, SessionConfig, validate
from audioforge.server.streams import LookaheadStream
from audioforge.tokenizer import CharTokenizer
from tests.test_serve import EnergyEngine, _diar_model, _speech_silence, _talky_asr_model

SLOW_R = 6  # tiny models run at [8, 1]; the slow pass at [8, 6] (7 frames = 560 ms chunks, the turn cuts fall
# inside a slow chunk)


def _nemo_asr_model(seed=0):
    """Tiny ASR with NeMo-aligned causal subsampling (the served cores' encoder: frame v ends at mel 8v)."""
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "att_context_sizes": [[8, 1], [8, SLOW_R]], "dropout": 0.0,
                       "speaker_kernel_layers": [0], "subsampling_padding": "nemo"},
           "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
                     "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
                     "turn": {"type": "turn", "mode": "kernel", "use_text": True, "condition_on_speaker": True,
                              "hidden": 16, "k_tokens": 2, "text_dim": 8, "dropout": 0.0}}}
    m = SpeechModel(cfg, CharTokenizer(list("abc "))).eval()
    with torch.no_grad():  # a VAD that is always on, a turn head that fires: deterministic turn ends
        m.heads["vad"].net[-1].bias.fill_(5.0)
        m.heads["turn"].out.bias.fill_(8.0)
        j = m.heads["rnnt"].joint.out[-1]  # tokens that depend on the encoder output (not one constant symbol)
        j.weight.mul_(40.0)
        j.bias.zero_()
    return m


_ENG = {}


def _engine(kind="nemo", fc=None, flush=True):
    key = (kind, fc, flush)
    if key not in _ENG:
        base = _ENG.setdefault(("model", kind), _nemo_asr_model() if kind == "nemo" else _talky_asr_model())
        _ENG[key] = EnergyEngine(base, _diar_model(), name="tiny", threads=1, debug=True,
                                 final_chunk_ms=fc, final_flush=flush)
    return _ENG[key]


def _run(eng, x, fs=320):
    s = Session(eng, SessionConfig(timeout_ms=400))
    msgs, batches = [], []
    for i in range(0, len(x), fs):
        b = s.process(x[i:i + fs])
        msgs += b
        batches.append(b)
    b = s.finish()
    msgs += b
    batches.append(b)
    return s, msgs, batches


FC = (SLOW_R + 1) * 80


@pytest.mark.parametrize("kind", ["nemo", "talky"])
@pytest.mark.parametrize("flush", [True, False])
def test_fast_path_byte_identical_to_single_rate(kind, flush):
    """Every frame, partial, turn_end and hint is the same as single rate; each final_fast equals today's final
    (type aside); one slow final per final_fast, same t and speaker, source slow / pass slow."""
    x = _speech_silence()
    _, one, _ = _run(_engine(kind), x)
    _, two, _ = _run(_engine(kind, FC, flush), x)
    for m in two:
        validate(m, debug=True)
    fast = [m for m in two if m["type"] == "final_fast"]
    slow = [m for m in two if m["type"] == "final"]
    rest = [m for m in two if m["type"] not in ("final", "final_fast", "stats")]
    assert rest == [m for m in one if m["type"] not in ("final", "stats")]
    assert [dict(m, type="final") for m in fast] == [m for m in one if m["type"] == "final"]
    assert len(fast) >= 2 and len(slow) == len(fast)
    assert [(m["t"], m["speaker"]) for m in slow] == [(m["t"], m["speaker"]) for m in fast]
    assert all(m["source"] == "slow" and m["pass"] == "slow" for m in slow)
    assert _engine(kind, FC, flush).ready_msg()["final_chunk_ms"] == FC
    for a, b in itertools.pairwise(slow):
        assert a["end"] == b["start"]


def _offline_tokens(m, x, right):
    ref = StreamingSession(m, att_context_size=[8, right])
    for i in range(0, len(x), 320):
        ref.feed(x[i:i + 320])
    ref.feed(np.zeros(0, np.float32), final=True)
    return ref


@pytest.mark.parametrize("kind", ["nemo", "talky"])
def test_slow_text_equals_offline_long_context(kind):
    """Without the flush the slow finals partition exactly the tokens of the encoder at [8, 6], which is the offline
    masked forward at [8, 6] (tests/test_streaming.py); with the flush the frame partition is the same."""
    x = _speech_silence()
    eng = _engine(kind, FC, flush=False)
    s, msgs, _ = _run(eng, x)
    slow = [m for m in msgs if m["type"] == "final"]
    ref = _offline_tokens(eng.asr, x, SLOW_R)
    assert s.la.tokens == ref.tokens
    assert "".join(m["text"] for m in slow) == ref.text
    if kind == "nemo":  # the masked offline forward (what research/DUAL_RATE.md's WER table measures)
        off = eng.asr.transcribe([x], head="rnnt", att_context_size=[8, SLOW_R])[0]
        assert off == ref.text
    _, msgs2, _ = _run(_engine(kind, FC, flush=True), x)
    slow2 = [m for m in msgs2 if m["type"] == "final"]
    assert [(m["start"], m["end"]) for m in slow2] == [(m["start"], m["end"]) for m in slow]
    if kind == "talky":  # one token per frame: the flushed texts are exact too
        assert [m["text"] for m in slow2] == [m["text"] for m in slow]


def test_flush_sends_slow_final_at_turn_end_and_off_waits_for_the_chunk():
    """flush on: the slow final is in the same batch as its turn_end, right after final_fast, latency_ms = compute.
    flush off: it waits for the slow chunk; latency_ms = that chunk's audio-time wait past the decision."""
    x = _speech_silence()
    _, _, batches = _run(_engine("nemo", FC, True), x)
    for b in batches:
        types = [m["type"] for m in b]
        for i, t in enumerate(types):
            if t == "final_fast":
                assert types[i + 1] == "final" and b[i + 1]["t"] == b[i]["t"]
    s, msgs, _ = _run(_engine("nemo", FC, False), x)
    slow = [m for m in msgs if m["type"] == "final"][:-1]  # (the end-of-stream final is flushed by finish)
    for m in slow:
        cut = round(m["end"] / 0.08)
        assert m["latency_ms"] == round(max(0.0, s.la.ready_t(cut - 1) - m["t"]) * 1000, 1)
    assert any(m["latency_ms"] > 0 for m in slow)


def test_flush_view_leaves_stream_unchanged_and_is_block_invariant():
    """flush_view is a copy: the stream decodes the same tokens with or without flushes; the flushed tokens at a
    decision time are the same whether the client sent 20 ms blocks or 250 ms blocks (the one-chunk rewind)."""
    m = _nemo_asr_model()
    x = _speech_silence()
    plain = LookaheadStream(m, SLOW_R)
    for i in range(0, len(x), 320):
        plain.feed(x[i:i + 320])
    for fs in (320, 4000):
        la = LookaheadStream(m, SLOW_R, snapshot=True)
        views = {}
        for i in range(0, len(x), fs):
            la.feed(x[i:i + fs])
            end = min(len(x), i + fs)
            for t in range((i // 1600 + 1) * 1600, end + 1, 1600):  # decision times in this block (100 ms grid)
                v = la.flush_view(t)
                assert v is not None
                views[t] = (v.n_frames, list(v.tokens))
        assert la.tokens == plain.tokens
        if fs == 320:
            small = views
        else:
            assert views == {t: small[t] for t in views}
    # a flush at a chunk boundary equals the decoded stream there
    ref = LookaheadStream(m, SLOW_R)
    ref.feed(x[:16000])
    full = ref.frames_ready(16000)
    v = ref.flush_view(16000)
    assert v.n_frames >= full and v.tokens[: ref.tok_at[full - 1]] == ref.tokens[: ref.tok_at[full - 1]]


def test_engine_flag_validation_and_single_rate_default():
    m = _talky_asr_model()
    e = EnergyEngine(m, _diar_model(), name="tiny", threads=1, final_chunk_ms=160)
    assert not e.dual and "final_chunk_ms" not in e.ready_msg() and "final_asr" not in e.ready_msg()
    with pytest.raises(ValueError):
        EnergyEngine(m, _diar_model(), name="tiny", threads=1, final_chunk_ms=200)
    with pytest.raises(ValueError):
        EnergyEngine(m, _diar_model(), name="tiny", threads=1, final_chunk_ms=FC, asr_lookahead=3)
    e = EnergyEngine(m, _diar_model(), name="tiny", threads=1, final_chunk_ms=FC)
    assert e.dual and e.asr_lookahead == SLOW_R and e.ready_msg()["final_asr"] == "slow"


def test_validate_dual_rate_messages():
    validate({"type": "final_fast", "t": 1.0, "text": "a", "speaker": 0})
    validate({"type": "final", "t": 1.0, "text": "a", "speaker": 0, "source": "slow", "pass": "slow", "start": 0.0,
              "end": 0.96, "latency_ms": 3.2})
    with pytest.raises(ValueError):
        validate({"type": "final_fast", "t": 1.0, "text": "a", "speaker": 0, "source": "slow"})


def test_deferred_flush_sends_turn_end_batch_first():
    """defer_slow (what the WebSocket handler sets): process() returns turn_end + final_fast without the slow final;
    flush_slow() right after returns it, the same message as the in-batch flush (latency_ms aside)."""
    x = _speech_silence()
    eng = _engine("nemo", FC, True)
    _, inb, _ = _run(eng, x)
    s = Session(eng, SessionConfig(timeout_ms=400))
    s.defer_slow = True
    msgs = []
    for i in range(0, len(x), 320):
        b = s.process(x[i:i + 320])
        types = [m["type"] for m in b]
        if "final_fast" in types:
            assert "final" not in types and s.la_pending
        msgs += b + s.flush_slow()
    msgs += s.finish()
    strip = lambda ms: [{k: v for k, v in m.items() if k != "latency_ms"} for m in ms if m["type"] != "stats"]
    assert strip(msgs) == strip(inb)


def test_socket_dual_rate_messages():
    """Over the socket: ready has final_chunk_ms; each final_fast is followed by its slow final (same t)."""
    from tests.test_hybrid_asr import _socket_run
    out = [m for _, m in _socket_run(_engine("nemo", FC, True), _speech_silence(), {"timeout_ms": 400})]
    assert out[0]["type"] == "ready" and out[0]["final_chunk_ms"] == FC and out[0]["final_asr"] == "slow"
    fast = [m for m in out if m["type"] == "final_fast"]
    slow = [m for m in out if m["type"] == "final"]
    assert len(fast) == len(slow) >= 2 and [m["t"] for m in fast] == [m["t"] for m in slow]
    for m in out:
        validate(m, debug=True)


def test_final_cut_speech_partitions_tokens_too():
    """final_cut="speech" (the --asr-lookahead rule: 3 frames past the last VAD speech frame) also partitions the
    long-context tokens exactly (flush off) and sends one slow final per final_fast."""
    x = _speech_silence()
    eng = EnergyEngine(_ENG.setdefault(("model", "nemo"), _nemo_asr_model()), _diar_model(), name="tiny", threads=1,
                       debug=True, final_chunk_ms=FC, final_flush=False, final_cut="speech")
    s, msgs, _ = _run(eng, x)
    slow = [m for m in msgs if m["type"] == "final"]
    assert len(slow) == len([m for m in msgs if m["type"] == "final_fast"])
    assert "".join(m["text"] for m in slow) == _offline_tokens(eng.asr, x, SLOW_R).text
    with pytest.raises(ValueError):
        EnergyEngine(eng.asr, _diar_model(), name="tiny", threads=1, final_chunk_ms=FC, final_cut="bogus")
