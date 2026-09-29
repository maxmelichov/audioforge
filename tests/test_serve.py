"""Streaming WebSocket server (audioforge/serve.py) and real-time client (scripts/stream_client.py).

Tiny random models stand in for the 440 MB checkpoints; the real-model test runs only with RUN_REAL=1."""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.model import SpeechModel, StreamingSession
from audioforge.serve import (DEFAULT_DIAR_CONFIG, DIAR_CONFIGS, DIAR_LOW_LATENCY, DIAR_LOW_LATENCY_032, FRAME_MS,
                              Engine, HeadPolicy, Resampler, Session, SessionConfig, TimeoutPolicy, diar_lag_ms,
                              diar_preset, fast_conv, main, serve, validate)
from audioforge.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
websockets = pytest.importorskip("websockets")


def _asr_model(seed=0):
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1], "dropout": 0.0, "speaker_kernel_layers": [0]},
           "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16},
                     "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
                     "turn": {"type": "turn", "mode": "kernel", "use_text": True, "condition_on_speaker": True,
                              "hidden": 16, "k_tokens": 2, "text_dim": 8, "dropout": 0.0}}}
    return SpeechModel(cfg, CharTokenizer(list("abc "))).eval()


def _diar_model(seed=1):
    torch.manual_seed(seed)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "per_feature", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": False,
                       "att_context_size": [-1, -1], "dropout": 0.0},
           "heads": {"diar": {"type": "sortformer", "num_spks": 4, "d_hidden": 16, "n_layers": 1, "dropout": 0.0,
                              "pos_emb": True}}}
    return SpeechModel(cfg, None).eval()


class EnergyDiarizer:
    """Scripted stand-in for StreamingDiarizer with the same interface and the same chunk + right-context emission
    rule: column 0 = 0.9 on 80 ms frames with RMS > 0.01 (else 0.02). Makes the turn path deterministic."""

    def __init__(self, chunk_len=6, right_context=7):
        from audioforge.streaming_diar import AOSCConfig
        self.cfg = AOSCConfig(chunk_len=chunk_len, chunk_right_context=right_context)
        self.x = np.zeros(0, np.float32)
        self.done = 0

    def feed(self, samples, final=False):
        from audioforge.data import ToneLanguage
        self.x = np.concatenate([self.x, np.asarray(samples, np.float32)])
        avail = ToneLanguage.n_frames(len(self.x)) if final and len(self.x) else len(self.x) // 1280
        C, R = self.cfg.chunk_len, self.cfg.chunk_right_context
        out = []
        while avail - self.done >= C + R or (final and avail > self.done):
            n = min(C, avail - self.done)
            for v in range(self.done, self.done + n):
                seg = self.x[v * 1280:(v + 1) * 1280]
                p = np.full(4, 0.02)
                p[0] = 0.9 if len(seg) and np.sqrt(np.mean(seg ** 2)) > 0.01 else 0.02
                out.append(p)
            self.done += n
        return torch.tensor(np.array(out) if out else np.zeros((0, 4)))


class EnergyEngine(Engine):
    def make_diarizer(self):
        return EnergyDiarizer()


class CrosstalkDiarizer(EnergyDiarizer):
    """EnergyDiarizer plus a weaker second speaker (column 1 = 0.6) on frames 14..20, inside the first silence."""

    def feed(self, samples, final=False):
        start = self.done
        out = super().feed(samples, final)
        if len(out):
            idx = np.arange(start, start + len(out))
            out[(idx >= 14) & (idx <= 20), 1] = 0.6
        return out


class CrosstalkEngine(Engine):
    def make_diarizer(self):
        return CrosstalkDiarizer()


def _talky_asr_model():
    """Tiny ASR whose transducer emits exactly one "a" per frame, VAD always on, turn head p ~ 0.99."""
    m = _asr_model()
    m.heads["rnnt"].max_symbols = 1
    with torch.no_grad():
        m.heads["rnnt"].joint.out[-1].bias[m.heads["rnnt"].blank] = -100.0
        m.heads["rnnt"].joint.out[-1].bias[m.tokenizer.token_id("a") if hasattr(m.tokenizer, "token_id") else 4] = 100.0
        m.heads["vad"].net[-1].bias.fill_(5.0)
        m.heads["turn"].out.bias.fill_(8.0)
    return m


def _speech_silence(pattern=((1.0, 0.2), (1.6, 0.0), (1.0, 0.2), (1.6, 0.0)), seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([(rng.standard_normal(int(16000 * d)) * a).astype(np.float32) for d, a in pattern])


_ENGINES = {}


def _energy_engine():
    if "energy" not in _ENGINES:
        _ENGINES["energy"] = EnergyEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1)
    return _ENGINES["energy"]


def _engine(turn_input="session", debug=False, diar_config=DEFAULT_DIAR_CONFIG):
    key = (turn_input, debug, diar_config)
    if key not in _ENGINES:
        _ENGINES[key] = Engine(_asr_model(), _diar_model(), name="tiny", threads=1, turn_input=turn_input,
                               debug=debug, diar_config=diar_config)
    return _ENGINES[key]


def _audio(sec=2.0, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(16000 * sec)) * 0.1).astype(np.float32)


def _n_frames(model, x):
    _, el = model.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]))
    return int(el[0])


# --------------------------------------------------------------------------- protocol shape
GOOD = [
    {"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80, "diar_config": "low_latency_032",
     "column_lag_ms": 160.0},
    {"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0, 0.5, 1, 0.2], "primary": None},
    {"type": "frame", "t": 0.16, "vad": 0.9, "eot": 0.99, "speakers": [0.9, 0.0, 0.0, 0.0], "primary": 0},
    {"type": "frames", "items": [{"t": 0.08, "vad": 0.1, "eot": 0.2, "speakers": [0, 0, 0, 0], "primary": 3}]},
    {"type": "partial", "t": 1.2, "text": "hello"},
    {"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1040},
    {"type": "turn_end", "t": 2.0, "policy": "head", "p": 0.985, "silence_ms": 320},
    {"type": "final", "t": 2.0, "text": "hello", "speaker": 1},
    {"type": "final", "t": 2.0, "text": "", "speaker": None},
    {"type": "stats", "rtf": 0.3, "chunk_ms_p50": 10.0, "chunk_ms_p95": 90.0, "first_partial_ms": None,
     "peak_rss_mb": 900.0},
]
BAD = [
    {"type": "nope"},
    {"type": "ready", "model": "m", "chunk_ms": 160, "diar_config": "x", "column_lag_ms": 1},  # missing key
    {"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80},  # pre-2026-09-26 shape: no diarizer keys
    {"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80, "diar_config": 3, "column_lag_ms": 1},
    {"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0, 0, 0], "primary": None},  # 3 speakers
    {"type": "frame", "t": 0.08, "vad": 1.5, "eot": None, "speakers": [0, 0, 0, 0], "primary": None},
    {"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0, 0, 0, 0], "primary": 4},
    {"type": "turn_end", "t": 2.0, "policy": "vad", "p": None, "silence_ms": 1000},
    {"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1000.5},
    {"type": "final", "t": 2.0, "text": "x", "speaker": True},
    {"type": "partial", "t": 1.0, "text": "x", "extra": 1},  # unexpected key
]


@pytest.mark.parametrize("msg", GOOD)
def test_validate_accepts_protocol_messages(msg):
    validate(msg)
    json.loads(json.dumps(msg))


@pytest.mark.parametrize("msg", BAD)
def test_validate_rejects_bad_messages(msg):
    with pytest.raises(ValueError):
        validate(msg)


def test_debug_keys_only_in_debug_mode():
    m = {"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0, 0, 0, 0], "primary": None,
         "spk_t": None}
    validate(m, debug=True)
    with pytest.raises(ValueError):
        validate(m)


def test_diar_lag_documented():
    assert diar_lag_ms(DIAR_LOW_LATENCY["chunk_len"], DIAR_LOW_LATENCY["chunk_right_context"]) == (560.0, 960.0, 760.0)
    assert diar_lag_ms(DIAR_LOW_LATENCY_032["chunk_len"],
                       DIAR_LOW_LATENCY_032["chunk_right_context"]) == (80.0, 240.0, 160.0)


def test_diar_preset_plumbing():
    from audioforge.streaming_diar import SORTFORMER_PRESETS, AOSCConfig
    assert DEFAULT_DIAR_CONFIG == "low_latency_032" and set(DIAR_CONFIGS) == set(SORTFORMER_PRESETS)
    for name in DIAR_CONFIGS:
        cfg = AOSCConfig.preset(name)
        assert diar_preset(name) == {k: getattr(cfg, k) for k in SORTFORMER_PRESETS[name]} == SORTFORMER_PRESETS[name]
    assert (DIAR_LOW_LATENCY_032["chunk_len"], DIAR_LOW_LATENCY_032["chunk_right_context"]) == (3, 1)
    assert (DIAR_LOW_LATENCY["chunk_len"], DIAR_LOW_LATENCY["chunk_right_context"]) == (6, 7)
    with pytest.raises(ValueError):
        diar_preset("nope")
    e = _engine()  # default preset reaches the per-session StreamingDiarizer
    assert e.diar_config == "low_latency_032" and e.diar_cfg == DIAR_LOW_LATENCY_032 and e.lag == (80.0, 240.0, 160.0)
    d = e.make_diarizer()
    assert (d.cfg.chunk_len, d.cfg.chunk_right_context, d.cfg.fifo_len) == (3, 1, 188)
    e = _engine(diar_config="low_latency")
    d = e.make_diarizer()
    assert e.diar_cfg == DIAR_LOW_LATENCY and (d.cfg.chunk_len, d.cfg.chunk_right_context) == (6, 7)
    o = Engine(e.asr, e.diar, name="tiny", threads=1, diar_config="low_latency", diar_cfg={"chunk_len": 2})
    assert o.diar_cfg["chunk_len"] == 2 and o.diar_cfg["chunk_right_context"] == 7
    assert o.diar_config == "low_latency+custom" and o.ready_msg()["column_lag_ms"] == diar_lag_ms(2, 7)[2]


def test_cli_diar_config_choices():
    with pytest.raises(SystemExit):  # argparse rejects an unknown preset before loading anything
        main(["--asr", "a", "--diar", "b", "--diar-config", "nope"])


@pytest.mark.parametrize("debug", [False, True])
def test_ready_message_shape(debug):
    for name, lag in (("low_latency_032", 160.0), ("low_latency", 760.0)):
        m = _engine(debug=debug, diar_config=name).ready_msg()
        validate(m, debug=debug)
        assert m["diar_config"] == name and m["column_lag_ms"] == lag
        assert {k: m[k] for k in ("type", "model", "chunk_ms", "frame_ms")} == \
            {"type": "ready", "model": "tiny", "chunk_ms": 160, "frame_ms": FRAME_MS}  # existing keys unchanged


# --------------------------------------------------------------------------- turn policies on hand-made tracks
def _rows(*spans, T):
    """spans: (col, start, end, p) -> (T, 4) activity track."""
    r = np.full((T, 4), 0.02)
    for c, s, e, p in spans:
        r[s:e, c] = p
    return r


def _fire_frames(pol, rows):
    return [(i, ev) for i, row in enumerate(rows) if (ev := pol.update(row)) is not None]


def test_timeout_fires_exactly_at_timeout():
    rows = _rows((0, 10, 30, 0.9), T=80)  # primary speaks frames 10..29, then silence
    pol = TimeoutPolicy(timeout_ms=1000)
    fired = _fire_frames(pol, rows)
    # inactive for >= 1000 ms = 13 frames (12 x 80 = 960 < 1000): frames 30..42 -> fires on frame 42, once
    assert [i for i, _ in fired] == [42]
    assert fired[0][1] == {"silence_ms": 1040, "primary": 0}
    pol = TimeoutPolicy(timeout_ms=960)
    assert [i for i, _ in _fire_frames(pol, rows)] == [41]
    assert pol.primary == 0


def test_timeout_plain_ignores_other_speakers_quiet_waits_and_rearms():
    # primary 0 speaks 0..29; speaker 1 talks 30..44 (overlapping the timeout); 0 speaks again 60..69
    rows = _rows((0, 0, 30, 0.9), (1, 30, 45, 0.9), (0, 60, 70, 0.9), T=100)
    # default (policy "timeout"): the plain primary-silence rule fires on frame 42 although column 1 is active
    fired = _fire_frames(TimeoutPolicy(timeout_ms=1000), rows)
    assert [i for i, _ in fired] == [42, 82]
    assert fired[0][1] == {"silence_ms": 1040, "primary": 0} and fired[1][1]["silence_ms"] == 1040
    # "timeout_quiet": column 1 is active until 44 -> fires on 45 (16 frames = 1280 ms)
    fired = _fire_frames(TimeoutPolicy(timeout_ms=1000, require_quiet=True), rows)
    assert [i for i, _ in fired] == [45, 82]
    assert fired[0][1]["silence_ms"] == 1280 and fired[0][1]["primary"] == 0
    assert fired[1][1]["silence_ms"] == 1040


def test_timeout_primary_is_most_active_column_in_last_5s():
    pol = TimeoutPolicy(timeout_ms=1000)
    assert pol.update(np.zeros(4)) is None and pol.primary is None  # nobody spoke yet
    rows = _rows((2, 0, 20, 0.9), (3, 20, 60, 0.8), T=60)
    prim = []
    for r in rows:
        pol.update(r)
        prim.append(pol.primary)
    assert prim[10] == 2 and prim[-1] == 3
    # column 2 leaves the 5 s window (63 frames) entirely -> never primary again; long silence -> None
    for _ in range(64):
        pol.update(np.zeros(4))
    assert pol.primary is None


def test_head_policy_threshold_crossing_and_rearm():
    hp = HeadPolicy(0.98)
    assert hp.update(0.99, vad=0.0) is None  # no speech yet: not armed
    assert hp.update(0.5, vad=0.9) is None
    ev = hp.update(0.985, vad=0.1)  # crosses 0.98 upwards after speech
    assert ev == {"p": 0.985, "silence_ms": 80}
    assert hp.update(0.99, vad=0.1) is None  # once per speech segment
    assert hp.update(0.99, vad=0.9) is None  # speech re-arms, but p never went below the threshold: no crossing
    hp.update(0.2, vad=0.9)
    assert hp.update(0.981, vad=0.0) is not None


def test_session_config_clamps():
    c = SessionConfig()
    assert c.update({"turn_policy": "bogus", "timeout_ms": 10 ** 6, "eot_threshold": 3}) and c.turn_policy == "timeout"
    assert c.timeout_ms <= 4840 and c.eot_threshold == 1.0
    assert not c.update({"turn_policy": "timeout_quiet"}) and c.turn_policy == "timeout_quiet"
    assert SessionConfig().turn_policy == "timeout"  # the plain primary-silence timeout is the default
    assert c.update({"sample_rate": 8000}, audio_started=True) and c.sample_rate == 16000


def test_resampler_streaming_matches_one_shot():
    x = np.sin(np.arange(8000) / 7.0).astype(np.float32)
    one = Resampler(8000)(x)
    r = Resampler(8000)
    blocks = np.concatenate([r(x[i:i + 333]) for i in range(0, len(x), 333)])
    assert abs(len(one) - 15999) <= 1 and np.allclose(blocks[: len(one)], one[: len(blocks)], atol=1e-6)


# --------------------------------------------------------------------------- CPU fast path
@pytest.mark.parametrize("build", [_asr_model, _diar_model])
def test_fast_conv_matches_original(build):
    ref, fast = build(), build()
    for m in (ref, fast):  # non-trivial BatchNorm statistics / LayerNorm affine
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    fast.load_state_dict(ref.state_dict())
    assert fast_conv(fast) == len(fast.encoder.layers)
    x = torch.from_numpy(_audio(1.3, seed=2))[None]
    lens = torch.tensor([x.shape[1]])
    e0, _ = ref.encode(x, lens)
    e1, _ = fast.encode(x, lens)
    assert torch.allclose(e0, e1, atol=1e-5), (e0 - e1).abs().max()
    xb = torch.cat([x, torch.cat([x[:, :15000], torch.zeros(1, x.shape[1] - 15000)], 1)])  # padded batch item
    lb = torch.tensor([x.shape[1], 15000])
    assert torch.allclose(ref.encode(xb, lb)[0], fast.encode(xb, lb)[0], atol=1e-5)
    if ref.encoder.causal:
        a, b = StreamingSession(ref), StreamingSession(fast)
        for i in range(0, x.shape[1], 777):
            a.feed(x[0, i:i + 777].numpy())
            b.feed(x[0, i:i + 777].numpy())
        a.feed([], final=True)
        b.feed([], final=True)
        assert a.tokens == b.tokens


# --------------------------------------------------------------------------- session (no socket)
def test_asr_stream_text_equals_streaming_session():
    eng = _engine()
    x = _audio(2.3, seed=3)
    ref = StreamingSession(eng.asr)
    for i in range(0, len(x), 555):
        ref.feed(x[i:i + 555])
    ref.feed([], final=True)
    s = Session(eng)
    for i in range(0, len(x), 555):
        s.process(x[i:i + 555])
    s.finish()
    assert s.asr.tokens == ref.tokens and s.asr.n_frames == _n_frames(eng.asr, x)


@pytest.mark.parametrize("turn_input", ["session", "diar"])
def test_end_flush_produces_final_and_valid_messages(turn_input):
    eng = _engine(turn_input, debug=True)
    x = _audio(1.7)
    s = Session(eng, SessionConfig(turn_policy="both"))
    msgs = []
    for i in range(0, len(x), 320):  # 20 ms frames
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=True)
    assert msgs[-1]["type"] == "stats" and msgs[-2]["type"] == "final"
    frames = [m for m in msgs if m["type"] == "frame"]
    assert len(frames) == _n_frames(eng.asr, x) == len(s.rows)  # both models on the same 80 ms clock
    assert [round(f["t"] / 0.08) for f in frames] == list(range(1, len(frames) + 1))
    # the head runs in both input modes (diar mode: only once the diarizer's first columns exist, ~1 s in)
    assert all(f["eot"] is not None for f in frames[-5:]) and (turn_input == "diar") == (frames[0]["eot"] is None)
    assert s.finish() == []  # idempotent
    st = msgs[-1]
    assert st["rtf"] > 0 and st["peak_rss_mb"] > 0 and st["n_chunks"] >= 10


def test_turn_events_timing_and_final_cuts():
    """Speech 0-1 s, silence 1-2.6 s, speech 2.6-3.6 s, silence: the timeout fires on the diarizer's clock at the
    documented decision time, finals partition the transcript, and nothing depends on the client's frame size."""
    eng = _energy_engine()
    x = _speech_silence()
    runs = []
    for fs in (320, 333, 4000):
        s = Session(eng, SessionConfig(turn_policy="both", timeout_ms=400, eot_threshold=0.98))
        msgs = []
        for i in range(0, len(x), fs):
            msgs += s.process(x[i:i + fs])
        msgs += s.finish()
        for m in msgs:
            validate(m)
        runs.append([m for m in msgs if m["type"] not in ("frame", "stats")])
    ev = [(m["type"], m["t"], m.get("policy"), m.get("text"), m.get("speaker")) for m in runs[0]
          if m["type"] != "partial"]
    for r in runs[1:]:
        assert ev == [(m["type"], m["t"], m.get("policy"), m.get("text"), m.get("speaker")) for m in r
                      if m["type"] != "partial"]
    te = [e for e in ev if e[0] == "turn_end" and e[2] == "timeout"]
    # speech frames 0..12 (frame 12 = 0.96-1.04 s still has energy); 400 ms = 5 silent frames -> frame 17 (t 1.44 s);
    # its diarizer column exists once chunk [12, 18) + 7 right-context frames arrived: (18 + 7) x 80 ms = 2.00 s
    assert te[0][1] == 2.0 and te[1][1] > 3.6
    heads = [e for e in ev if e[0] == "turn_end" and e[2] == "head"]
    assert heads and heads[0][1] <= 0.2  # p ~ 0.99 from the start: fires once, at the first speech frame's chunk
    finals = [e for e in ev if e[0] == "final"]
    assert len(finals) == 3 and finals[0][4] == 0 and finals[0][1] == 2.0 and finals[-1][1] == round(len(x) / 16000, 3)
    # the first final holds exactly the tokens of the ASR frames available at 2.0 s (one "a" per frame): the chunk
    # [22, 24) ends at 1.92 s and is ready at 1.936 s; frames 24-25 need audio up to 2.096 s -> 24 tokens
    assert finals[0][3] == "a" * 24
    s = Session(eng, SessionConfig(timeout_ms=400))
    for i in range(0, len(x), 320):
        s.process(x[i:i + 320])
    s.finish()
    assert s.asr.tok_at[23] == 24 and s.seg_tok == len(s.asr.tokens)
    partials = [m for m in runs[0] if m["type"] == "partial"]
    assert partials and partials[0]["text"]


def test_session_plain_timeout_vs_timeout_quiet_with_crosstalk():
    """A second column active during the primary's silence: the default plain timeout still fires at the same
    decision time as without it; timeout_quiet waits for the crosstalk to end. Same protocol keys, tag "timeout"."""
    eng = CrosstalkEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1)
    x = _speech_silence()
    res = {}
    for pol in ("timeout", "timeout_quiet"):
        s = Session(eng, SessionConfig(turn_policy=pol, timeout_ms=400))
        msgs = []
        for i in range(0, len(x), 320):
            msgs += s.process(x[i:i + 320])
        msgs += s.finish()
        for m in msgs:
            validate(m)
        assert all(s.prims[v] == 0 for v in range(12, 40))  # column 0 stays the primary
        res[pol] = [m for m in msgs if m["type"] in ("turn_end", "final")]
    te = {p: [m for m in r if m["type"] == "turn_end"] for p, r in res.items()}
    assert all(m["policy"] == "timeout" and m["p"] is None for r in te.values() for m in r)
    # plain: silent frames 13..17 (400 ms) -> frame 17, column ready at (18 + 7) x 80 ms = 2.00 s (as without crosstalk)
    assert te["timeout"][0]["t"] == 2.0 and te["timeout"][0]["silence_ms"] == 400
    # quiet: column 1 active until frame 20 -> frame 21 (9 silent frames = 720 ms), ready at (24 + 7) x 80 = 2.48 s
    assert te["timeout_quiet"][0]["t"] == 2.48 and te["timeout_quiet"][0]["silence_ms"] == 720
    assert len(te["timeout"]) == len(te["timeout_quiet"]) == 2
    for r in res.values():  # each timeout turn_end is followed by its final; shapes identical across policies
        assert [m["type"] for m in r][:2] == ["turn_end", "final"] and r[1]["speaker"] == 0
    assert [sorted(m) for m in res["timeout"]] == [sorted(m) for m in res["timeout_quiet"]]


@pytest.mark.parametrize("diar_config", ["low_latency_032", "low_latency"])
def test_session_with_each_diar_preset(diar_config):
    """Tiny models, real StreamingDiarizer: every frame gets a column, the columns are finalized at the preset's
    chunk + right-context emission rule, and every message is valid."""
    eng = _engine(debug=True, diar_config=diar_config)
    C, R = eng.diar_cfg["chunk_len"], eng.diar_cfg["chunk_right_context"]
    x = _audio(2.0, seed=11)
    s = Session(eng, SessionConfig(turn_policy="both"))
    msgs, ages = [], []
    for i in range(0, len(x), 320):
        out = s.process(x[i:i + 320])
        ages += [m["t"] - m["spk_t"] for m in out if m["type"] == "frame" and m["spk_t"] is not None]
        msgs += out
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=True)
    assert len(s.rows) == len([m for m in msgs if m["type"] == "frame"]) == _n_frames(eng.asr, x)
    lo, hi, mean = eng.lag
    assert abs(msgs[-1]["diar_lag_ms_mean_measured"] - mean) < 2 * FRAME_MS  # in the preset's range
    assert all(lo <= l <= hi for l in s.diar_emit_lag)
    assert s._diar_ready_t(0) == (C + R) * FRAME_MS / 1000
    assert max(ages) <= (hi + 2 * FRAME_MS) / 1000  # column age on the frame messages (ASR 160 ms chunk clock)


def test_end_without_audio_still_final():
    s = Session(_engine())
    out = s.finish()
    assert [m["type"] for m in out] == ["final", "stats"] and out[0]["text"] == ""


# --------------------------------------------------------------------------- end to end over a local socket
async def _with_server(eng, fn):
    loop = asyncio.get_running_loop()
    port_f, stop = loop.create_future(), loop.create_future()
    task = asyncio.create_task(serve(eng, "127.0.0.1", 0, ready_event=port_f.set_result, stop=stop))
    port = await asyncio.wait_for(port_f, 10)
    try:
        return await fn(f"ws://127.0.0.1:{port}")
    finally:
        stop.set_result(None)
        await task


async def _stream(url, x, cfg=None, frame=320, end=True, sr=16000, stop_after=None):
    from websockets.asyncio.client import connect
    out = []
    async with connect(url) as ws:
        out.append(json.loads(await ws.recv()))
        if cfg:
            await ws.send(json.dumps({"type": "config", **cfg}))
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        step = frame * 2 * sr // 16000
        for k, i in enumerate(range(0, len(pcm), step)):
            if stop_after is not None and k >= stop_after:
                return out  # abrupt disconnect (context exit closes the socket)
            await ws.send(pcm[i:i + step])
        if end:
            await ws.send(json.dumps({"type": "end"}))
        async for m in ws:
            out.append(json.loads(m))
    return out


def test_e2e_socket_sequential_connections_disconnect_and_resample():
    eng = _energy_engine()
    x = _speech_silence(((0.8, 0.2), (1.2, 0.0)), seed=5)

    async def body(url):
        a = await _stream(url, x, {"turn_policy": "timeout", "timeout_ms": 400})
        b = await _stream(url, x, stop_after=20)  # client disconnects mid-stream
        await asyncio.sleep(0.3)
        c = await _stream(url, x, {"turn_policy": "timeout", "timeout_ms": 400}, frame=333)  # odd frame size
        x8 = x[::2].copy()
        d = await _stream(url, x8, {"sample_rate": 8000}, sr=8000)
        return a, b, c, d

    a, b, c, d = asyncio.run(_with_server(eng, body))
    n = _n_frames(eng.asr, x)
    for msgs in (a, c):
        for m in msgs:
            validate(m)
        types = [m["type"] for m in msgs]
        assert types[0] == "ready" and types[-1] == "stats" and types[-2] == "final"
        assert types.count("frame") == n and types.count("stats") == 1
        assert msgs[0] == {"type": "ready", "model": "tiny", "chunk_ms": 160, "frame_ms": FRAME_MS,
                           "diar_config": "low_latency_032", "column_lag_ms": 160.0}
        assert "turn_end" in types and types.count("final") == 2 and "partial" in types
    assert b[0]["type"] == "ready" and len(b) >= 1
    # same audio + config -> same events regardless of the client's frame size (both models are chunking-invariant)
    fin = [[(m["type"], m["t"], m.get("text"), m.get("speaker"), m.get("policy")) for m in msgs
            if m["type"] in ("final", "turn_end")] for msgs in (a, c)]
    assert fin[0] == fin[1]
    types_d = [m["type"] for m in d]
    assert types_d[-1] == "stats" and abs(types_d.count("frame") - n) <= 1


def _load_client():
    spec = importlib.util.spec_from_file_location("stream_client", ROOT / "scripts" / "stream_client.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stream_client_script_end_to_end(tmp_path):
    import soundfile as sf
    client = _load_client()
    x = _audio(1.2, seed=7)
    wav = tmp_path / "a.wav"
    sf.write(wav, x, 16000)
    log, summ = tmp_path / "ev.jsonl", tmp_path / "s.json"
    a = argparse.Namespace(audio=str(wav), url=None, speed=0.0, policy="both", timeout_ms=400, eot_threshold=0.5,
                           log=str(log), summary=str(summ), ref=None, ref_text="a b c", verbose=False)

    async def body(url):
        a.url = url
        return await client.run(a)

    s = asyncio.run(_with_server(_engine(), body))
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines[0]["msg"]["type"] == "ready" and lines[-1]["msg"]["type"] == "stats"
    assert all("wall" in l and "msg" in l for l in lines)
    assert s["server_stats"]["rtf"] > 0 and s["n_frames"] == _n_frames(_engine().asr, x)
    assert s["finals"] and "wer_vs_ref" in s and s["keeping_up"] is None  # speed 0: no pacing, no lag


def test_stream_client_real_time_pacing(tmp_path):
    import soundfile as sf
    client = _load_client()
    x = _audio(1.0, seed=8)
    wav = tmp_path / "b.wav"
    sf.write(wav, x, 16000)
    a = argparse.Namespace(audio=str(wav), url=None, speed=1.0, policy="timeout", timeout_ms=1000,
                           eot_threshold=0.98, log=None, summary=None, ref=None, ref_text=None, verbose=False)

    async def body(url):
        a.url = url
        return await client.run(a)

    s = asyncio.run(_with_server(_engine(), body))
    assert 0.95 <= s["send_wall_s"] <= 1.6  # paced at 1x
    assert s["keeping_up"]["keeps_up"] and s["keeping_up"]["frame_lag_ms_p50"] < 500


# --------------------------------------------------------------------------- real checkpoints (slow)
@pytest.mark.skipif(os.environ.get("RUN_REAL") != "1", reason="loads the 440 MB + 450 MB checkpoints; RUN_REAL=1")
def test_real_models_librispeech_utterance():
    from audioforge.data import load_wav
    eng = Engine.load(str(ROOT / "runs/stage1_heads_pretrained.afm"), str(ROOT / "runs/nemo_sortformer_v2.afm"),
                      threads=2)
    x = load_wav(str(ROOT / "data/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac"))
    s = Session(eng)
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m)
    text = " ".join(m["text"] for m in msgs if m["type"] == "final")
    assert "belly" in text and ("counselled" in text or "counseled" in text), text
    assert any(m["type"] == "frame" and m["primary"] is not None for m in msgs)


# --------------------------------------------------------------------------- --enroll (voice binding, EOT_BENCH_V2 §9)
from audioforge.serve import ENROLL_MODES, VoiceBinder  # noqa: E402


class FakeTN:
    """TitaNetEmbedder stand-in: (mean, std, 1) of the frames' audio, unit norm (voices = DC offsets)."""
    dim = 3

    def frames(self, audio, idx_lists):
        audio = np.asarray(audio, np.float32)
        out = []
        for idx in idx_lists:
            seg = np.concatenate([audio[u * 1280:(u + 1) * 1280] for u in np.asarray(idx, np.int64)])
            v = np.array([seg.mean(), seg.std(), 1.0], np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out) if out else np.zeros((0, 3), np.float32)


# scripted two-speaker scene (80 ms frames): agent (voice B, column 1) 0-12, user (voice A) in column 0 20-60,
# the user again in column 2 75-110 (a diarizer column swap), 120 frames
SCENE = ((0, 13, 1, -0.5), (20, 61, 0, 0.5), (75, 111, 2, 0.5))


def _scene_audio(T=120, seed=0):
    x = (np.random.default_rng(seed).standard_normal(T * 1280) * 0.05).astype(np.float32)
    for a, b, _, off in SCENE:
        x[a * 1280:b * 1280] += off
    return x


def _scene_rows(T=120):
    rows = np.full((T, 4), 0.02)
    for a, b, c, _ in SCENE:
        rows[a:b, c] = 0.9
    return rows


class ScriptedDiarizer(EnergyDiarizer):
    def feed(self, samples, final=False):
        start = self.done
        out = super().feed(samples, final)
        if len(out):
            out = torch.tensor(_scene_rows(max(120, start + len(out)))[start:start + len(out)])
        return out


class ScriptedEngine(Engine):
    def make_diarizer(self):
        return ScriptedDiarizer()


def test_validate_enrollment_keys_all_or_nothing():
    base = {"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80, "diar_config": "low_latency_032",
            "column_lag_ms": 160.0}
    validate(base)
    validate({**base, "enroll": "after_agent", "enrolled": False, "primary_column": None})
    validate({**base, "enroll": "explicit", "enrolled": True, "primary_column": 2})
    with pytest.raises(ValueError):
        validate({**base, "enroll": "after_agent"})  # incomplete
    with pytest.raises(ValueError):
        validate({**base, "enrolled": False})  # a stray enrollment key without the mode
    st = {"type": "stats", "rtf": 0.5, "chunk_ms_p50": 1.0, "chunk_ms_p95": 2.0, "first_partial_ms": None,
          "peak_rss_mb": 100.0}
    validate(st)
    validate({**st, "enroll": "after_agent", "enrolled": True, "primary_column": 0})
    validate({"type": "enrolled", "t": 3.04, "column": 0})
    with pytest.raises(ValueError):
        validate({"type": "enrolled", "t": 3.04, "column": None})


def test_voice_binder_after_agent_selects_enrolls_and_follows_a_swap():
    x, rows = _scene_audio(), _scene_rows()
    b = VoiceBinder(FakeTN(), 4, stride=5)
    cols = []
    for v in range(120):
        if v == 13:  # the agent's TTS ended: arm
            b.arm(13)
        cols.append(b.update(v, rows[v], x[v * 1280:(v + 1) * 1280]))
    assert cols[:22] == [None] * 22  # the agent's own column is never chosen (armed after it)
    assert cols[22] == 0 and b.n_arm == 1  # 3rd consecutive active frame of column 0 (20, 21, 22)
    assert b.enrolled_at == 20 + 19 - 1 and b.n_enrolled == 1
    assert all(c == 0 for c in cols[22:75 + 8])  # nobody else speaks: stays
    assert cols[-1] == 2 and 1 not in cols and 3 not in cols  # the swap to column 2 (same voice) is followed
    sw = next(v for v in range(75, 120) if cols[v] == 2)
    assert 75 + 8 <= sw <= 75 + 8 + 5 + 6 + 1  # min_frames of speech, a grid update, `hold` frames
    # re-arming restarts the search (column 2 is still active: chosen again after 3 frames)
    b.arm(100)
    assert b.column is None and not b.enrolled
    for v in range(100, 106):
        b.update(v, rows[v], x[v * 1280:(v + 1) * 1280])
    assert b.column == 2 and b.n_arm == 2


def test_voice_binder_ignores_speech_before_arm_and_the_other_voice():
    x, rows = _scene_audio(), _scene_rows()
    b = VoiceBinder(FakeTN(), 4, stride=5)
    for v in range(120):
        assert b.update(v, rows[v], x[v * 1280:(v + 1) * 1280]) is None  # never armed: never binds
    b = VoiceBinder(FakeTN(), 4, stride=5)
    b.arm(0)  # armed while the agent speaks: the agent's column is the "next active column" (the product must arm
    for v in range(40):  # at its TTS end, as documented); voice A in column 0 must then not be accepted as the agent
        b.update(v, rows[v], x[v * 1280:(v + 1) * 1280])
    assert b.column == 1 and b.enrolled is False  # 13 agent frames < 19: never enrolled, stays on the choice


def test_session_enroll_after_agent_messages_and_default_unchanged():
    eng = ScriptedEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1, enroll="after_agent",
                         embedder=FakeTN(), debug=True)
    r = eng.ready_msg()
    validate(r, debug=True)
    assert r["enroll"] == "after_agent" and r["enrolled"] is False and r["primary_column"] is None
    s = Session(eng, SessionConfig(timeout_ms=800))
    assert s.arm_enrollment("enroll") is False  # the other mode's trigger is ignored
    assert s.arm_enrollment("agent_end", 13) is True
    x = _scene_audio()
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=True)
    enr = [m for m in msgs if m["type"] == "enrolled"]
    assert len(enr) == 1 and enr[0]["column"] == 0 and 3.0 <= enr[0]["t"] <= 4.0  # frame 38 finalized at 3.92 s
    # frame.primary is the latest finalized diarizer frame's primary (frame v is final ~1 s later here: chunk 6 + 7)
    prim = [(m["t"], m["primary"]) for m in msgs if m["type"] == "frame"]
    assert all(p == 1 for t, p in prim if 1.0 <= t <= 1.9)  # before arming: the dominant column (the agent)
    assert all(p == 0 for t, p in prim if 3.0 <= t <= 6.8)  # the bound column, not the dominant one
    assert all(p == 2 for t, p in prim if t >= 9.6)  # followed the swap
    assert all(p != 1 for t, p in prim if t >= 3.0)  # the agent's column never becomes the primary after arming
    st = msgs[-1]
    assert st["enroll"] == "after_agent" and st["enrolled"] is True and st["primary_column"] == 2
    assert st["enroll_n_embed"] >= 1 and st["enroll_ms_p50"] >= 0
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert te and te[0]["policy"] == "timeout" and 5.0 <= te[0]["t"] <= 7.0  # the user's silence after frame 60
    # the default engine's protocol has no enrollment keys and no binder
    d = ScriptedEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1)
    assert d.enroll == "dominant" and d.embedder is None and "enroll" not in d.ready_msg()
    sd = Session(d)
    assert sd.binder is None and sd.arm_enrollment("agent_end") is False
    out = sd.process(x[:16000]) + sd.finish()
    assert "enroll" not in out[-1] and all(m["type"] != "enrolled" for m in out)


def test_engine_rejects_unknown_enroll_mode():
    with pytest.raises(ValueError):
        Engine(_asr_model(), _diar_model(), name="tiny", threads=1, enroll="nope", embedder=FakeTN())
    assert ENROLL_MODES == ("dominant", "after_agent", "after_agent_arm", "explicit")


def test_e2e_socket_agent_end_message_arms_the_session():
    eng = ScriptedEngine(_talky_asr_model(), _diar_model(), name="tiny", threads=1, enroll="after_agent",
                         embedder=FakeTN())
    x = _scene_audio()

    async def run():
        stop = asyncio.get_running_loop().create_future()
        port_f = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(serve(eng, "127.0.0.1", 0, ready_event=port_f.set_result, stop=stop))
        port = await port_f
        got = []
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            got.append(json.loads(await ws.recv()))
            pcm = (x * 32768).astype("<i2").tobytes()
            cut = 13 * 1280 * 2
            await ws.send(pcm[:cut])
            await ws.send(json.dumps({"type": "agent_end"}))  # arrives at the audio position 13 frames
            await ws.send(json.dumps({"type": "enroll"}))  # wrong trigger for this mode: logged, ignored
            await ws.send(pcm[cut:])
            await ws.send(json.dumps({"type": "end"}))
            async for m in ws:
                got.append(json.loads(m))
        stop.set_result(None)
        await task
        return got

    got = asyncio.run(run())
    assert got[0]["type"] == "ready" and got[0]["enroll"] == "after_agent"
    enr = [m for m in got if m["type"] == "enrolled"]
    assert len(enr) == 1 and enr[0]["column"] == 0
    assert got[-1]["type"] == "stats" and got[-1]["enrolled"] is True and got[-1]["primary_column"] == 2
