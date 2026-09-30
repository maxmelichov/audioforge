"""turn_policy vad_head (research/EOT_LATENCY.md; --mode single's default): the served VAD head's silence >= K frames AND
the turn head p >= theta, OR that silence >= FALLBACK, OR (enrolled TS-VAD track) the user's own silence reaching
USER_SIL while P(other) has held >= 0.9 for HOLD; no Silero. The policy object equals the offline simulator the rule was
picked with (scripts/research/eot_latency.py sim_room / sim_vadhead), and a Session fires where the rule says, without
loading Silero."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.serve import POLICIES, POLICY_THETA, Session, SessionConfig, VadHeadPolicy, validate
from audioforge.server.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("test_serve_shipped_helpers", ROOT / "tests" / "test_serve_shipped.py")
EOT = _load("eot_latency_research", ROOT / "scripts" / "research" / "eot_latency.py")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k,fb,th", [(1, 12, 0.95), (2, None, 0.8), (3, 8, 0.5)])
def test_policy_equals_offline_simulator(seed, k, fb, th):
    rng = np.random.default_rng(seed)
    n = 400
    vad = np.clip(np.repeat(rng.random(n // 8) > 0.5, 8) * 0.8 + rng.random(n) * 0.3, 0, 1)
    p = rng.random(n) ** 0.3
    t = (np.arange(n) + 1) * 0.08
    d = {"head": {"t": t.tolist(), "p": p.tolist(), "vad": vad.tolist(), "pu": [0.0] * n}}
    rule = {"family": "vad_head", "mode": "any", "vad_thr": 0.3, "k": k, "th": th, "fallback_f": fb}
    want = [(tt, path) for tt, path in EOT.sim_vadhead(d, rule)]
    pol = VadHeadPolicy(th, k, fb, 0.3, others=None)
    got = []
    for v in range(n):
        ev = pol.update(float(p[v]), float(vad[v]))
        if ev is not None:
            got.append((round(float(t[v]), 4), ev["path"]))
            assert ev["silence_ms"] % 80 == 0 and ev["silence_ms"] >= 80 * k
    assert got == want and len(got) > 0


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k,fb,vt,others", [(4, 10, 0.4, (12, 8)), (1, 12, 0.3, (6, 1)), (2, None, 0.5, (8, 4)),
                                             (3, 8, 0.4, None)])
def test_policy_with_the_others_path_equals_sim_room(seed, k, fb, vt, others):
    rng = np.random.default_rng(100 + seed)
    n = 600
    vad = np.clip(np.repeat(rng.random(n // 8) > 0.4, 8) * 0.8 + rng.random(n) * 0.3, 0, 1)
    pu = np.clip(np.repeat(rng.random(n // 6) > 0.6, 6) * 0.9 + rng.random(n) * 0.2, 0, 1)
    po = np.clip(np.repeat(rng.random(n // 10) > 0.5, 10) * 0.95 + rng.random(n) * 0.1, 0, 1)
    p = rng.random(n) ** 0.3
    t = (np.arange(n) + 1) * 0.08
    d = {"head": {"t": t.tolist(), "p": p.tolist(), "vad": vad.tolist(), "pu": pu.tolist(), "po": po.tolist()}}
    rule = {"family": "room", "src": "vad", "vad_thr": vt, "k": k, "th": 0.95, "fallback_f": fb, "ot": 0.9,
            "om": others[1] if others else 1, "fu": others[0] if others else None, "fw": 0, "ku": None}
    want = [(tt, "others" if path == "others_fallback" else path) for tt, path in EOT.sim_room(d, rule)]
    pol = VadHeadPolicy(0.95, k, fb, vt, others=others)
    got = []
    for v in range(n):
        ev = pol.update(float(p[v]), float(vad[v]), float(pu[v]), float(po[v]))
        if ev is not None:
            got.append((round(float(t[v]), 4), ev["path"]))
    assert got == want and len(got) > 0
    if others:
        assert any(path == "others" for _, path in got)


def test_defaults_are_the_shipped_rule():
    """--mode single's default = scripts/research/eot_latency.py CHOSEN (after the TS-VAD print fix: the fastest rule
    that keeps the fe28a9e rule's calls / AMI numbers and does not cut the bundled clip): VAD < 0.4 for 160 ms AND
    p >= 0.99, OR 640 ms, OR the user quiet 960 ms while P(other) >= 0.9 for 640 ms."""
    pol = VadHeadPolicy()
    c = EOT.CHOSEN
    assert (pol.thr, pol.k, pol.fb, pol.vad_thr) == (c["th"], c["k"], c["fallback_f"], c["vad_thr"])
    assert pol.others == (c["fu"], c["om"]) and pol.others_p == c["ot"] and c["fw"] == 0 and c["ku"] is None
    e = H.H.EnergyEngine(H.H._talky_asr_model(), H.H._diar_model(), name="tiny", threads=1)
    assert (e.vad_head_k, e.vad_head_fb, e.vad_head_others) == (2, 8, (12, 8)) and e.turn_policy == "timeout"


def test_others_path_ends_the_turn_while_someone_else_talks():
    """The room stays loud (VAD 0.9: another speaker), so the VAD silence never comes; the user's own track went quiet
    at frame 5: with P(other) >= 0.9 held for 8 frames, the turn ends when the user's silence reaches 12 frames. Without
    the track (p_user None) the policy waits for the room."""
    pol, pol_novad = VadHeadPolicy(), VadHeadPolicy()
    fired, fired2 = [], []
    for v in range(40):
        pu = 0.9 if v < 5 else 0.05
        po = 0.95 if v >= 3 else 0.0
        ev = pol.update(0.2, 0.9, pu, po)
        if ev is not None:
            fired.append((v, ev["path"], ev["silence_ms"]))
        if pol_novad.update(0.2, 0.9) is not None:
            fired2.append(v)
    assert fired == [(16, "others", 960)] and fired2 == []
    # a backchannel (P(other) high for less than the hold) does not end the turn
    pol = VadHeadPolicy()
    evs = [pol.update(0.2, 0.9, 0.9 if v < 5 else 0.05, 0.95 if 12 <= v < 16 else 0.0) for v in range(40)]
    assert all(e is None for e in evs)


def test_one_firing_per_silence_run_and_fallback():
    pol = VadHeadPolicy(0.95, 1, 3, 0.3)
    seq = [(0.1, 0.9), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (0.99, 0.1), (0.1, 0.9), (0.99, 0.1), (0.99, 0.1)]
    evs = [pol.update(p, vad) for p, vad in seq]
    fired = [(i, e["path"], e["silence_ms"]) for i, e in enumerate(evs) if e is not None]
    assert fired == [(3, "fallback", 240), (6, "head", 80)]


def test_config_protocol_and_flag():
    assert "vad_head" in POLICIES
    assert SessionConfig(turn_policy="vad_head").theta == POLICY_THETA["vad_head"] == 0.99
    validate({"type": "turn_end", "t": 1.0, "policy": "vad_head", "p": 0.97, "silence_ms": 80})
    a = build_parser().parse_args(["--vad-wait-ms", "160,800", "--others-wait-ms", "0", "--turn-policy", "vad_head"])
    assert a.vad_wait_ms == "160,800" and a.others_wait_ms == "0" and a.turn_policy == "vad_head"
    a = build_parser().parse_args([])
    assert a.turn_policy == "timeout" and a.others_wait_ms is None


def _energy_vad(s, x):
    """The tiny test model's VAD head is constant: replace it with frame energy (0.9 speech / 0.05 silence)."""
    orig = s.asr.feed_frames

    def feed(samples, final=False):
        out = orig(samples, final)
        for f in out:
            seg = x[f["v"] * 1280:(f["v"] + 1) * 1280]
            f["vad"] = 0.9 if len(seg) and float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) > 0.01 else 0.05
        return out
    s.asr.feed_frames = feed


@pytest.mark.parametrize("bias,frame,sil_ms", [(8.0, 14, 160), (-8.0, 20, 640)])
def test_session_vad_head_fires_on_the_head_or_the_fallback_without_silero(bias, frame, sil_ms):
    """1.0 s speech + 2.0 s silence: speech frames 0..12. p ~ 1 -> the head path after 160 ms of silence (frame
    12 + 2); p ~ 0 -> the 640 ms fallback (frame 12 + 8). Decided when the frame's 160 ms chunk is ready."""
    eng = H._engine(bias)
    eng.silero_model = None  # vad_head must never ask for Silero
    x = H.H._speech_silence(((1.0, 0.2), (2.0, 0.0)))
    s = Session(eng, SessionConfig(turn_policy="vad_head"))
    assert s.sil is None
    _energy_vad(s, x)
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(te) == 1 and te[0]["policy"] == "vad_head" and te[0]["silence_ms"] == sil_ms
    assert te[0]["t"] == round(s._asr_ready_t(frame), 3)
    assert (te[0]["p"] > 0.99) == (bias > 0)
    assert any(m["type"] == "final" and m["t"] == te[0]["t"] for m in msgs)  # the cut carries a final


def test_engine_rejects_bad_vad_wait():
    m = H.H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(0.0)
    with pytest.raises(ValueError):
        H.H.EnergyEngine(m, H.H._diar_model(), name="tiny", threads=1, vad_wait_ms="160,80")
    e = H.H.EnergyEngine(m, H.H._diar_model(), name="tiny", threads=1, vad_wait_ms="160,0")
    assert e.vad_head_k == 2 and e.vad_head_fb is None
    for bad in ("40,640", "960,-1", "1,2,3"):
        with pytest.raises(ValueError):
            H.H.EnergyEngine(m, H.H._diar_model(), name="tiny", threads=1, others_wait_ms=bad)
    e = H.H.EnergyEngine(m, H.H._diar_model(), name="tiny", threads=1, others_wait_ms="0")
    assert e.vad_head_others is None
    with pytest.raises(ValueError):
        H.H.EnergyEngine(m, H.H._diar_model(), name="tiny", threads=1, turn_policy="nope")


def _clip_turn_ends(pol, h):
    out = []
    for v in range(len(h["p"])):
        ev = pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v])
        if ev is not None:
            out.append(h["t"][v])
    return out


def test_defaults_do_not_cut_the_bundled_clip():
    """The quickstart clip's user turn runs 3.9-10.8 s with a ~0.5 s pause after "life" (8.6-9.1 s), where the turn head
    reads 0.94-0.99 and the served VAD dips in and out of 0.4. The recorded frames of every delivery the research
    script checks (float, int16 truncated as examples/quickstart_client.py sends it, int16 rounded, -6 / +6 dB, dither):
    the defaults end the turn after 10.8 s and never inside it. The rule shipped until fe28a9e (320 ms at p >= 0.95,
    800 ms fallback) cut it at 9.06 s as the quickstart client delivers it."""
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    assert len(fx["variants"]) == 6
    for name, h in fx["variants"].items():
        te = _clip_turn_ends(VadHeadPolicy(), h)
        assert not any(s0 <= t < e1 - 0.08 for t in te), (name, te)
        assert any(e1 - 0.08 <= t < e1 + 1.5 for t in te), (name, te)  # and it does answer the end
    old = _clip_turn_ends(VadHeadPolicy(0.95, 4, 10), fx["variants"]["int16_trunc (quickstart_client)"])
    assert old[0] == 9.056
