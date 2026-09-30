"""vad_head's energy gate (--energy-gate, default on) and --turn-model smartturn (research/EOT_ASSISTANT.md "Energy
gate", "smart-turn bridge").

Gate: a per-session noise floor (10th percentile of the 80 ms frame log energies of the last 3 s); a turn is armed
only by an onset (VAD > 0.5 AND energy > floor + 6 dB), nothing fires before 160 ms of onset frames (warm-up guard),
and --energy-quiet-db X also counts energy-quiet frames as silence. smartturn: Pipecat's smart-turn v3.2 at the quiet
trigger, input prepared as Pipecat's LocalSmartTurnAnalyzerV3."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from audioforge.serve import Session, SessionConfig, validate
from audioforge.server.cli import build_parser
from audioforge.server.constants import ENERGY_GATE, SMARTTURN_ENERGY, TURN_PRESETS
from audioforge.server.policies import EnergyGate, VadHeadPolicy, frame_db
from audioforge.server.smartturn import SmartTurnTrigger, bundled_onnx

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("test_serve_shipped_helpers_eg", ROOT / "tests" / "test_serve_shipped.py")
EOT = _load("eot_latency_research_eg", ROOT / "scripts" / "research" / "eot_latency.py")


# --------------------------------------------------------------------------- EnergyGate
def test_frame_db():
    assert frame_db(np.zeros(1280)) == pytest.approx(-100.0)
    assert frame_db([]) == -100.0
    assert frame_db(np.full(1280, 0.1)) == pytest.approx(-20.0, abs=1e-6)


def test_floor_quiet_onset_and_warmup():
    g = EnergyGate(quiet_db=6.0, onset_db=9.0, warmup_frames=4)
    for _ in range(10):  # room tone at -50 dBFS with the VAD head's 0.66: quiet, no onset
        q, o = g.update(-50.0, 0.66)
        assert q and not o
    assert g.floor == pytest.approx(-50.0) and not g.warm
    for i in range(4):  # speech 25 dB over the floor
        q, o = g.update(-25.0, 0.99)
        assert not q and o and g.warm == (i == 3)
    q, o = g.update(-45.0, 0.9)  # 5 dB over the floor: not quiet (X 6 -> quiet), not an onset (Y 9)
    assert q and not o
    assert EnergyGate(quiet_db=None).update(-100.0, 0.1)[0] is False  # quiet_db None: energy never makes silence
    assert EnergyGate(quiet_db=None, abs_db=-85).update(-90.0, 0.9)[0] is True


def test_floor_window_and_admission():
    g = EnergyGate(window_s=0.4)  # 5 frames
    for _ in range(5):
        g.update(-60.0, 0.0)
    for _ in range(6):
        g.update(-20.0, 0.0)  # VAD 0: not an onset, so admitted: the floor follows the new level
    assert g.floor == pytest.approx(-20.0)
    g = EnergyGate(window_s=0.4, admit="quiet")
    for _ in range(5):
        g.update(-60.0, 0.0)
    for _ in range(20):
        g.update(-20.0, 0.99)  # onsets are kept out of the window: the floor stays at the room
    assert g.floor == pytest.approx(-60.0)
    with pytest.raises(ValueError):
        EnergyGate(admit="some")


def _pol(**kw):
    return VadHeadPolicy(0.99, 2, 8, 0.4, others=None, gate=EnergyGate(quiet_db=None, onset_db=6.0, warmup_frames=2),
                         **kw)


def test_leading_room_tone_does_not_fire():
    """The fresh-session VAD reading (0.55) over room tone: without the gate the 640 ms fallback fires before any
    speech; with it nothing is armed until speech."""
    vad = [0.55] * 3 + [0.1] * 12
    en = [-50.0] * 15
    bare = VadHeadPolicy(0.99, 2, 8, 0.4, others=None)
    assert any(bare.update(0.0, q) for q in vad)
    pol = _pol()
    assert not any(pol.update(0.0, q, None, None, e) for q, e in zip(vad, en))


def test_onset_arms_warmup_then_fallback():
    pol = _pol()
    seq = [(0.1, -50.0)] * 5 + [(0.95, -20.0)] * 1 + [(0.1, -50.0)] * 12  # 80 ms of speech: not warm, no turn_end
    assert not any(pol.update(0.0, q, None, None, e) for q, e in seq)
    seq = [(0.95, -20.0)] * 3 + [(0.1, -50.0)] * 9
    evs = [pol.update(0.0, q, None, None, e) for q, e in seq]
    fired = [i for i, ev in enumerate(evs) if ev]
    assert fired == [3 + 7] and evs[10]["path"] == "fallback" and evs[10]["silence_ms"] == 640
    # a speech frame that is not an onset (VAD high, energy at the floor) resets the silence but does not re-arm
    evs = [pol.update(0.0, q, None, None, e) for q, e in [(0.9, -50.0)] + [(0.1, -50.0)] * 10]
    assert not any(evs)


def test_quiet_db_ends_silence_at_the_audible_end():
    """--energy-quiet-db: the VAD head's tail over room tone counts as silence."""
    def run(qdb):
        pol = VadHeadPolicy(0.99, 2, 8, 0.4, others=None,
                            gate=EnergyGate(quiet_db=qdb, onset_db=6.0, warmup_frames=2))
        seq = [(0.1, -50.0)] * 5 + [(0.99, -20.0)] * 5 + [(0.66, -50.0)] * 6 + [(0.1, -100.0)] * 10
        return [i for i, (q, e) in enumerate(seq) if pol.update(0.0, q, None, None, e)]
    assert run(None) == [16 + 7]  # the fallback counts from the end of the VAD tail
    assert run(6.0) == [10 + 7]  # ... or from the audible end


def test_shipped_gate_leaves_the_bundled_clip_as_served():
    """The recorded frames of the six deliveries + their audio energies: the shipped gate answers the question at
    the same time as the VAD-only rule and never cuts (served_check: runs/eot_latency.json served_check)."""
    from audioforge.data import load_wav
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())["variants"]
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
    for name, fn in EOT.CLIP_VARIANTS.items():
        d = {"head": fx[name]}
        te = [t for t, _ in EOT.sim_served_gate(d, fn(x0))]
        assert not any(s0 <= t < e1 - 0.08 for t in te), (name, te)
        first = [t for t in te if t >= e1 - 0.08][0]
        want = [t for t, _ in EOT.sim_room(d, EOT.CHOSEN) if t >= e1 - 0.08][0]
        assert first == want, (name, te)


# --------------------------------------------------------------------------- Session wiring
def _vad_from_energy(s, x, lead_vad=0.55):
    """The tiny model's VAD is constant: speech frames (RMS > 0.01) read 0.95; quiet frames read lead_vad (the served
    head's fresh-session reading) for the first 0.4 s, then 0.05."""
    orig = s.asr.feed_frames

    def feed(samples, final=False):
        out = orig(samples, final)
        for f in out:
            seg = x[f["v"] * 1280:(f["v"] + 1) * 1280]
            loud = len(seg) and float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > 0.01
            f["vad"] = 0.95 if loud else (lead_vad if f["v"] < 5 else 0.05)
        return out
    s.asr.feed_frames = feed


def _speech(pattern, seed=0):
    """Room tone (-60 dBFS) and syllable-modulated speech: amplitude 0.2 x (0.3 .. 1) at 4 Hz."""
    rng = np.random.default_rng(seed)
    out = []
    for d, a in pattern:
        n = int(16000 * d)
        env = 0.65 + 0.35 * np.sin(2 * np.pi * 4 * np.arange(n) / 16000) if a else 1.0
        out.append((rng.standard_normal(n) * (a * env if a else 0.001)).astype(np.float32))
    return np.concatenate(out)


def _run(eng, x, block=320, **cfg):
    s = Session(eng, SessionConfig(turn_policy="vad_head", **cfg))
    _vad_from_energy(s, x)
    msgs = []
    for i in range(0, len(x), block):
        msgs += s.process(x[i:i + block])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    return s, msgs


@pytest.mark.parametrize("gate", [True, False])
def test_session_gate_removes_the_pre_speech_turn_end(gate):
    eng = H._engine(-8.0, energy_gate=gate)  # p ~ 0: the fallback decides
    x = _speech(((1.5, 0.0), (1.0, 0.2), (1.5, 0.0)))
    s, msgs = _run(eng, x)
    te = [m for m in msgs if m["type"] == "turn_end"]
    pre = [m for m in te if m["t"] < 1.5]  # the VAD's 0.55 over the leading room tone -> fallback at ~1.1 s
    assert bool(pre) == (not gate)
    after = [m for m in te if m["t"] >= 2.5]
    assert len(after) == 1 and after[0]["path"] == "fallback" and after[0]["silence_ms"] == 640
    assert (s.vh_pol.gate is not None) == gate
    if gate:
        assert s.vh_pol.gate.warm and s.vh_pol.gate.floor < -50


def test_session_energy_does_not_depend_on_client_blocks():
    """A frame's energy is the audio known at its decision-ready time, not what the client happened to send."""
    x = _speech(((1.5, 0.0), (1.0, 0.2), (0.3, 0.0), (0.8, 0.2), (1.5, 0.0)))
    got = []
    for block in (320, 1000, 8000):
        eng = H._engine(-8.0, energy_quiet_db=6.0)
        s, msgs = _run(eng, x, block=block)
        got.append([(m["t"], m["path"]) for m in msgs if m["type"] == "turn_end"])
        assert s._energy(3) == pytest.approx(frame_db(x[3 * 1280:min(4 * 1280, s._ready_sample(3))]))
    assert got[0] == got[1] == got[2] and got[0]


def test_session_no_hint_before_warmup():
    eng = H._engine(8.0)  # p ~ 1: the head path + a hint on every silence
    x = _speech(((1.0, 0.0), (1.0, 0.2), (1.5, 0.0)))
    _, msgs = _run(eng, x)
    hints = [m for m in msgs if m["type"] == "turn_end_hint"]
    assert hints and all(m["t"] >= 1.0 for m in hints)
    assert [m["path"] for m in msgs if m["type"] == "turn_end"] == ["head"]


def test_flags():
    a = build_parser().parse_args([])
    assert a.energy_gate == "on" and a.energy_quiet_db is None and a.turn_model == "head"
    assert a.smartturn_trigger == "vad"
    a = build_parser().parse_args(["--energy-gate", "off", "--energy-quiet-db", "6", "--turn-model", "smartturn",
                                   "--smartturn-trigger", "energy", "--smartturn-onnx", "x.onnx"])
    assert (a.energy_gate, a.energy_quiet_db, a.turn_model, a.smartturn_trigger) == ("off", 6.0, "smartturn", "energy")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--turn-model", "livekit"])
    assert ENERGY_GATE["quiet_db"] is None and ENERGY_GATE["onset_db"] == 6.0 and ENERGY_GATE["warmup_ms"] == 160
    for kw in ({"energy_quiet_db": 0.0}, {"turn_model": "x"}, {"turn_model": "smartturn", "smartturn_onnx": "/nope.onnx"},
               {"smartturn_trigger": "x"}):
        with pytest.raises((ValueError, FileNotFoundError)):
            H._engine(8.0, **kw)


def test_protocol_turn_end_path_and_model_ms():
    base = {"type": "turn_end", "t": 1.0, "policy": "vad_head", "p": 0.9, "silence_ms": 160}
    validate({**base, "path": "model", "model_ms": 21.5})
    validate({**base, "path": "fallback"})
    with pytest.raises(ValueError):
        validate({**base, "path": "timer"})


# --------------------------------------------------------------------------- --turn-model smartturn
class FakeModel:
    """predict(segment) -> (p, ms) from a script; records the segments."""

    def __init__(self, ps):
        self.ps, self.segs = list(ps), []

    def predict(self, seg):
        self.segs.append(np.asarray(seg).copy())
        return (self.ps.pop(0) if self.ps else 0.0), 20.0


def test_trigger_segment_is_pipecats():
    """Onset - 0.5 s, not before the last turn end, the last 8 s, int16-quantised."""
    x = (np.random.default_rng(0).standard_normal(16000 * 12) * 0.1).astype(np.float32)
    m = FakeModel([0.9, 0.9, 0.9])
    tr = SmartTurnTrigger(m, lambda a, b: x[a:b], lambda v: (v + 2) * 1280)
    tr(20, 10)  # onset frame 10 = sample 12800 -> start 4800, end frame 22 = 28160
    seg = m.segs[-1]
    assert len(seg) == 28160 - 4800
    np.testing.assert_array_equal(seg, np.round(x[4800:28160] * 32768) / 32768)
    tr.turn_ended(10000)
    tr(20, 10)
    assert len(m.segs[-1]) == 28160 - 10000
    tr(140, 10)  # a long turn: the last 8 s
    assert len(m.segs[-1]) == 8 * 16000
    assert [c["p"] for c in tr.calls] == [0.9] * 3


def _st_pol(ps, **kw):
    m = FakeModel(ps)
    calls = []

    def tm(v, onset_v):
        calls.append((v, onset_v))
        return m.predict(np.zeros(10))
    pol = VadHeadPolicy(0.99, 2, 38, 0.4, others=None, gate=EnergyGate(quiet_db=kw.pop("quiet_db", None), onset_db=6.0,
                                                                          warmup_frames=2),
                        turn_model=tm, **kw)
    return pol, calls


def test_smartturn_complete_fires_at_the_trigger():
    pol, calls = _st_pol([0.8])
    seq = [(0.1, -50.0)] * 3 + [(0.95, -20.0)] * 6 + [(0.1, -50.0)] * 5
    evs = [pol.update(0.0, q, None, None, e) for q, e in seq]
    fired = [(i, ev["path"]) for i, ev in enumerate(evs) if ev]
    assert fired == [(10, "model")] and calls == [(10, 3)] and evs[10]["p"] == 0.8


def test_smartturn_incomplete_waits_reruns_then_falls_back():
    pol, calls = _st_pol([0.2, 0.3])
    seq = ([(0.1, -50.0)] * 3 + [(0.95, -20.0)] * 6 + [(0.1, -50.0)] * 6 + [(0.95, -20.0)] * 4
           + [(0.1, -50.0)] * 40)
    evs = [pol.update(0.0, q, None, None, e) for q, e in seq]
    fired = [(i, ev["path"]) for i, ev in enumerate(evs) if ev]
    assert calls == [(10, 3), (20, 3)]  # once per silence run, the turn's first onset both times
    assert fired == [(18 + 38, "fallback")]


def test_smartturn_two_clocks():
    """--smartturn-trigger energy: the classifier on energy-or-VAD quiet (the VAD's tail over room tone counts), the
    fallback timer on the VAD's own silence."""
    seq = [(0.1, -50.0)] * 3 + [(0.95, -20.0)] * 6 + [(0.7, -50.0)] * 5 + [(0.1, -50.0)] * 12
    pol, calls = _st_pol([0.99], quiet_db=6.0, model_quiet_only=True, model_p=0.97)
    pol.fb = 8
    evs = [pol.update(0.0, q, None, None, e) for q, e in seq]
    assert calls == [(10, 3)] and [(i, ev["path"]) for i, ev in enumerate(evs) if ev] == [(10, "model")]
    pol, calls = _st_pol([0.9], quiet_db=6.0, model_quiet_only=True, model_p=0.97)  # below p: the VAD timer
    pol.fb = 8
    evs = [pol.update(0.0, q, None, None, e) for q, e in seq]
    assert [(i, ev["path"]) for i, ev in enumerate(evs) if ev] == [(13 + 8, "fallback")]
    assert SMARTTURN_ENERGY == {"quiet_db": 6.0, "quiet_ms": 240, "p": 0.97}
    assert all(p["smartturn_fallback_ms"] == 3000 for p in TURN_PRESETS.values())


def test_session_smartturn_with_a_fake_model():
    eng = H._engine(-8.0)
    eng.smartturn, eng.turn_model = FakeModel([0.9]), "smartturn"
    x = _speech(((1.0, 0.0), (1.0, 0.2), (1.5, 0.0)))
    s, msgs = _run(eng, x)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(te) == 1 and te[0]["path"] == "model" and te[0]["model_ms"] == 20.0 and te[0]["silence_ms"] == 160
    seg = eng.smartturn.segs[0]  # the turn from 0.5 s before its first speech frame (~1.0 s) to the decision
    assert 0.9 * 16000 < len(seg) < 2.0 * 16000
    st = [m for m in msgs if m["type"] == "stats"][0]["turn_model"]
    assert st["model"] == "smartturn" and st["calls"] == 1 and st["complete"] == 1


@pytest.mark.skipif(bundled_onnx() is None, reason="pipecat-ai (bundled smart-turn-v3.2-cpu.onnx) not installed")
def test_smartturn_model_equals_pipecats_analyzer():
    from audioforge.baselines.turn import pipecat_smartturn_analyzer
    from audioforge.server.smartturn import SmartTurnModel
    m = SmartTurnModel(threads=1)
    an = pipecat_smartturn_analyzer(threads=1)
    rng = np.random.default_rng(1)
    for n in (8000, 16000 * 3, 16000 * 9):
        x = np.round(rng.standard_normal(n).astype(np.float32) * 0.05 * 32768) / 32768
        p, ms = m.predict(x)
        assert p == pytest.approx(an._predict_endpoint(x[-8 * 16000:])["probability"], abs=1e-6) and ms > 0
