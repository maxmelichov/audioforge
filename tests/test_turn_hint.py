"""The early end-of-turn hint (audioforge/server/turn_hint.py, docs/PROTOCOL.md 5.11): turn_end_hint once per user turn
on 80 ms of served-VAD silence with turn-head p >= H, confirmed by the next turn_end (hinted_at) or withdrawn by
turn_end_hint_cancel when the user resumes; turn_end itself unchanged; --turn-hint-off restores the old protocol."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.serve import Session, SessionConfig, VadHeadPolicy, validate
from audioforge.server.cli import build_parser
from audioforge.server.turn_hint import TurnHintLedger, TurnHintTracker, replay

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("test_serve_helpers_hint", ROOT / "tests" / "test_serve.py")


# --------------------------------------------------------------------------- tracker / ledger
def _feed(tr, rows):
    out = []
    for p, vad in rows:
        out += tr.update(p, vad)
    return out


def test_tracker_hint_once_per_silence_then_cancel_on_two_speech_frames():
    tr = TurnHintTracker(0.8)
    sp, q = (0.1, 0.9), (0.9, 0.05)
    c = _feed(tr, [sp] * 5 + [q] * 6)
    assert [x["kind"] for x in c] == ["hint"] and c[0]["v"] == 5 and c[0]["speech_v"] == 4
    assert _feed(tr, [(0.1, 0.9)] + [q] * 3) == []  # one speech frame: not a resume, and no second hint
    c = _feed(tr, [sp, sp])
    assert [x["kind"] for x in c] == ["cancel"]
    c = _feed(tr, [q])
    assert [x["kind"] for x in c] == ["hint"]  # re-armed by the resume


def test_tracker_needs_p_and_quiet():
    tr = TurnHintTracker(0.8)
    assert _feed(tr, [(0.95, 0.9)] * 3 + [(0.5, 0.05)] * 4) == []  # quiet but p < H; high p only while speaking
    assert [x["kind"] for x in _feed(tr, [(0.81, 0.05)])] == ["hint"]
    assert TurnHintTracker(0.8).update(0.99, 0.05) == []  # silence before any speech


def test_ledger_confirm_cancel_and_drop():
    led = TurnHintLedger()
    m = led.hint(1.0, {"v": 10, "p": 0.9, "speech_v": 9}, "hi there")
    assert m == {"type": "turn_end_hint", "t": 1.0, "p": 0.9, "kind": "hint", "text": "hi there"}
    validate(m)
    assert led.hint(1.1, {"v": 11, "p": 0.9, "speech_v": 9}) is None  # one outstanding
    assert led.turn_end(1.2, 12) == 1.0
    assert led.cancel(1.3) is None  # nothing outstanding any more
    assert led.hint(1.4, {"v": 14, "p": 0.95, "speech_v": 9}) is None  # the silence the turn_end already closed
    assert led.hint(2.0, {"v": 20, "p": 0.85, "speech_v": 18}) is not None
    m = led.cancel(2.1)
    assert m == {"type": "turn_end_hint_cancel", "t": 2.1}
    validate(m)
    assert led.turn_end(3.0, 30) is None
    assert led.stats() == {"sent": 2, "confirmed": 1, "cancelled": 1, "open": 0, "lead_ms_p50": 200.0}


def test_protocol_rejects_bad_hints():
    with pytest.raises(ValueError):
        validate({"type": "turn_end_hint", "t": 1.0, "p": 0.9, "kind": "final", "text": ""})
    with pytest.raises(ValueError):
        validate({"type": "turn_end_hint", "t": 1.0, "p": 1.5, "kind": "hint", "text": ""})
    with pytest.raises(ValueError):
        validate({"type": "turn_end_hint_cancel", "t": 1.0, "p": 0.5})
    validate({"type": "turn_end", "t": 1.0, "policy": "vad_head", "p": 0.99, "silence_ms": 160, "hinted_at": 0.84})
    validate({"type": "turn_end", "t": 1.0, "policy": "vad_head", "p": 0.99, "silence_ms": 160, "hinted_at": None})


def test_flags():
    a = build_parser().parse_args([])
    assert a.turn_hint_p == 0.8 and not a.turn_hint_off
    a = build_parser().parse_args(["--turn-hint-p", "0.9", "--turn-hint-off"])
    assert a.turn_hint_p == 0.9 and a.turn_hint_off


# --------------------------------------------------------------------------- the bundled clip's recorded frames
def test_bundled_clip_hints_cancel_inside_the_turn_and_lead_the_turn_end():
    """two_party_call_16s: the user's turn (3.9-10.8 s) pauses at 8.6-9.1 s with p 0.94-0.99; a hint there is withdrawn
    when the user goes on, and the turn end gets a hint before the vad_head turn_end (which is unchanged)."""
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    h = fx["variants"]["int16_trunc (quickstart_client)"]
    r = replay(h["t"], h["p"], h["vad"], VadHeadPolicy(), h["pu"], h["po"])
    inside = [x for x in r["hints"] if s0 <= x[0] < e1 - 0.08]
    assert inside and all(x[3] == "cancelled" for x in inside)
    te = next((t, ha) for t, _, ha in r["turn_ends"] if t >= e1 - 0.08)
    assert te[1] is not None and e1 - 0.08 <= te[1] < te[0]
    plain = []  # the turn_ends without hints are the same
    pol = VadHeadPolicy()
    for v in range(len(h["p"])):
        if pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v]) is not None:
            plain.append(h["t"][v])
    assert [t for t, _, _ in r["turn_ends"]] == plain


# --------------------------------------------------------------------------- a Session
def _energy_vad(s, x):
    orig = s.asr.feed_frames

    def feed(samples, final=False):
        out = orig(samples, final)
        for f in out:
            seg = x[f["v"] * 1280:(f["v"] + 1) * 1280]
            f["vad"] = 0.9 if len(seg) and float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > 0.01 else 0.05
        return out
    s.asr.feed_frames = feed


def _engine(bias, **kw):
    """The tiny engine; the energy gate off by default (its synthetic speech is constant-level noise from t = 0, which
    never rises 6 dB over its own floor; tests/test_energy_gate.py covers the gate)."""
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(bias)
    kw.setdefault("energy_gate", False)
    return H.EnergyEngine(m, H._diar_model(), name="tiny", threads=1, **kw)


def _run(eng, x, policy="vad_head"):
    s = Session(eng, SessionConfig(turn_policy=policy))
    _energy_vad(s, x)
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    return s, msgs


def test_session_hint_then_confirm():
    """1.0 s speech + 2.0 s silence, p ~ 1: the hint on the first quiet frame (13), the vad_head turn_end on frame 14
    (160 ms of silence) confirms it; the hint's text is what the final then holds."""
    eng = _engine(8.0)
    x = H._speech_silence(((1.0, 0.2), (2.0, 0.0)))
    s, msgs = _run(eng, x)
    hints = [m for m in msgs if m["type"] == "turn_end_hint"]
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(hints) == 1 and len(te) == 1 and not any(m["type"] == "turn_end_hint_cancel" for m in msgs)
    assert hints[0]["t"] == round(s._asr_ready_t(13), 3) and hints[0]["t"] < te[0]["t"]
    assert te[0]["t"] == round(s._asr_ready_t(14), 3) and te[0]["hinted_at"] == hints[0]["t"]
    assert msgs.index(hints[0]) < msgs.index(te[0])
    fin = next(m for m in msgs if m["type"] == "final" and m["t"] == te[0]["t"])
    assert hints[0]["text"] and fin["text"].startswith(hints[0]["text"])
    st = msgs[-1]
    assert st["type"] == "stats" and st["turn_hints"]["sent"] == 1 and st["turn_hints"]["confirmed"] == 1


def test_session_pause_hint_is_cancelled_and_turn_end_unchanged():
    """speech, a 240 ms pause, speech, silence, with --vad-wait-ms 320,640 so the pause does not end the turn: the
    pause's hint is cancelled when the user resumes, the real end gets a new, confirmed hint, and the turn_end
    messages equal those of a --turn-hint-off engine."""
    x = H._speech_silence(((1.0, 0.2), (0.24, 0.0), (0.8, 0.2), (2.0, 0.0)))
    s, msgs = _run(_engine(8.0, vad_wait_ms="320,640"), x)
    kinds = [m["type"] for m in msgs if m["type"].startswith("turn_end")]
    assert kinds == ["turn_end_hint", "turn_end_hint_cancel", "turn_end_hint", "turn_end"], kinds
    h1, c, h2, te = [m for m in msgs if m["type"].startswith("turn_end")]
    assert h1["t"] < c["t"] < h2["t"] < te["t"] and te["hinted_at"] == h2["t"]
    assert msgs[-1]["turn_hints"] == {"sent": 2, "confirmed": 1, "cancelled": 1, "open": 0,
                                      "lead_ms_p50": round((te["t"] - h2["t"]) * 1000, 1)}
    _, off = _run(_engine(8.0, vad_wait_ms="320,640", turn_hint_p=None), x)
    te_off = [m for m in off if m["type"] == "turn_end"]
    assert [{k: v for k, v in m.items() if k != "hinted_at"} for m in msgs if m["type"] == "turn_end"] == te_off
    assert not any(m["type"].startswith("turn_end_hint") for m in off)
    assert "hinted_at" not in te_off[0] and "turn_hints" not in off[-1]


def test_session_low_p_no_hint_and_unhinted_turn_end():
    eng = _engine(-8.0)  # p ~ 0: the 640 ms fallback ends the turn, no hint
    x = H._speech_silence(((1.0, 0.2), (2.0, 0.0)))
    _, msgs = _run(eng, x)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(te) == 1 and te[0]["hinted_at"] is None
    assert not any(m["type"] == "turn_end_hint" for m in msgs)


def test_engine_rejects_bad_hint_p():
    with pytest.raises(ValueError):
        _engine(8.0, turn_hint_p=0.0)
    with pytest.raises(ValueError):
        _engine(8.0, turn_hint_p=1.5)
