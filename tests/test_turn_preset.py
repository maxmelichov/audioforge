"""--turn-preset / config.turn_preset (research/VAD_TAIL.md, research/EOT_LATENCY.md "Turn presets"): vad_head's
constants as a named trade-off. balanced = the shipped rule (--mode single's default); steady (the fast preset before
turn head v5) = VAD < 0.6 for >= 480 ms AND p >= 0.99, OR 720 ms, others path 640,640. The policy each preset builds
equals the offline rule it was scored as (scripts/research/eot_latency.py CHOSEN / FAST), and neither cuts the bundled
clip. The v5 presets (fast, assistant) are tested in tests/test_turn_seg.py."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from audioforge.serve import TURN_PRESETS, Session, SessionConfig, VadHeadPolicy, vad_head_params
from audioforge.server.cli import MODES, build_parser

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _load("test_serve_shipped_helpers_tp", ROOT / "tests" / "test_serve_shipped.py")
EOT = _load("eot_latency_research_tp", ROOT / "scripts" / "research" / "eot_latency.py")


def _rule_params(r):
    return r["k"], r["fallback_f"], r["vad_thr"], (r["fu"], r["om"])


def test_presets_are_the_scored_rules():
    assert set(TURN_PRESETS) == {"balanced", "fast", "steady", "assistant"}
    assert vad_head_params("balanced") == _rule_params(EOT.CHOSEN)
    assert vad_head_params("steady") == _rule_params(EOT.FAST) == (6, 9, 0.6, (8, 8))
    pol = VadHeadPolicy()  # the class defaults stay the balanced rule
    assert (pol.k, pol.fb, pol.vad_thr, pol.others) == vad_head_params("balanced")
    assert EOT.FAST["th"] == EOT.CHOSEN["th"] == 0.99


def test_flags_override_the_preset():
    assert vad_head_params("steady", "160,640") == (2, 8, 0.6, (8, 8))
    assert vad_head_params("steady", None, "0") == (6, 9, 0.6, None)
    with pytest.raises(ValueError):
        vad_head_params("steady", "480,80")


def test_cli_modes_and_config():
    assert build_parser().parse_args([]).turn_preset == "balanced"
    assert build_parser().parse_args(["--turn-preset", "fast"]).turn_preset == "fast"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--turn-preset", "turbo"])
    assert MODES["single"]["turn_preset"] == "balanced"
    c = SessionConfig()
    assert c.turn_preset is None and c.update({"turn_preset": "fast"}) == [] and c.turn_preset == "fast"
    assert c.update({"turn_preset": "turbo"}) and c.turn_preset == "fast"
    assert build_parser().parse_args(["--turn-preset", "assistant"]).turn_preset == "assistant"


def test_engine_and_session_presets():
    e = H.H.EnergyEngine(H.H._talky_asr_model(), H.H._diar_model(), name="tiny", threads=1, turn_preset="steady")
    assert (e.vad_head_k, e.vad_head_fb, e.vad_head_thr, e.vad_head_others) == (6, 9, 0.6, (8, 8))
    with pytest.raises(ValueError):
        H.H.EnergyEngine(H.H._talky_asr_model(), H.H._diar_model(), name="tiny", threads=1, turn_preset="turbo")
    e = H.H.EnergyEngine(H.H._talky_asr_model(), H.H._diar_model(), name="tiny", threads=1)
    s = Session(e, SessionConfig(turn_policy="vad_head"))
    assert (s.vh_pol.k, s.vh_pol.fb, s.vh_pol.vad_thr, s.vh_pol.others) == (2, 8, 0.4, (12, 8))
    s = Session(e, SessionConfig(turn_policy="vad_head", turn_preset="steady"))
    assert (s.vh_pol.k, s.vh_pol.fb, s.vh_pol.vad_thr, s.vh_pol.others, s.vh_pol.thr) == (6, 9, 0.6, (8, 8), 0.99)
    # a preset change after the session started is refused (the rule is built with the session)
    s.take_notices()
    cfg = SessionConfig(turn_policy="vad_head", turn_preset="balanced")
    s.apply_config(cfg)
    assert s.cfg.turn_preset == "steady" and s.vh_pol.vad_thr == 0.6
    assert any("turn_preset" in m["detail"] for m in s.take_notices())


def test_a_model_preset_may_carry_its_own_theta():
    """cfg["turn_presets"] can set a preset's turn-head threshold (the 0.6B core's balanced, research/CORE_0P6B_TURN.md);
    the client's eot_threshold still wins, and models without it keep POLICY_THETA (0.99)."""
    asr = H.H._talky_asr_model()
    asr.cfg = {**(getattr(asr, "cfg", None) or {}), "turn_presets": {"balanced": {"theta": 0.9, "vad_wait_ms": [240, 960]}}}
    e = H.H.EnergyEngine(asr, H.H._diar_model(), name="tiny", threads=1)
    s = Session(e, SessionConfig(turn_policy="vad_head"))
    assert (s.vh_pol.thr, s.vh_pol.k, s.vh_pol.fb) == (0.9, 3, 12)
    s = Session(e, SessionConfig(turn_policy="vad_head", eot_threshold=0.97))
    assert s.vh_pol.thr == 0.97
    s = Session(e, SessionConfig(turn_policy="vad_head", turn_preset="steady"))
    assert s.vh_pol.thr == 0.99
    e = H.H.EnergyEngine(H.H._talky_asr_model(), H.H._diar_model(), name="tiny", threads=1)
    assert Session(e, SessionConfig(turn_policy="vad_head")).vh_pol.thr == 0.99


def test_a_model_may_give_balanced_the_v5_decider():
    """cfg["turn_presets"]["balanced"]["turn_model"] adds the v5 segment classifier to a preset that has none (on top of
    V5_TURN_MODEL); without it balanced stays the per-frame head rule."""
    from audioforge.serve import V5_TURN_MODEL, model_presets
    m = type("M", (), {"cfg": {"turn_presets": {"balanced": {"vad_wait_ms": [320, 720],
                                                             "turn_model": {"vad_thr": 0.4, "p": 0.6}}}}})()
    pr = model_presets(m)
    assert pr["balanced"]["turn_model"] == {**V5_TURN_MODEL, "vad_thr": 0.4, "p": 0.6}
    assert vad_head_params("balanced", presets=pr)[:2] == (4, 9)
    assert pr["fast"] == TURN_PRESETS["fast"] and "turn_model" not in model_presets(object())["balanced"]


@pytest.mark.parametrize("seed", range(6))
def test_steady_policy_equals_sim_room(seed):
    rng = np.random.default_rng(300 + seed)
    n = 600
    vad = np.clip(np.repeat(rng.random(n // 8) > 0.4, 8) * 0.8 + rng.random(n) * 0.3, 0, 1)
    pu = np.clip(np.repeat(rng.random(n // 6) > 0.6, 6) * 0.9 + rng.random(n) * 0.2, 0, 1)
    po = np.clip(np.repeat(rng.random(n // 10) > 0.5, 10) * 0.95 + rng.random(n) * 0.1, 0, 1)
    p = rng.random(n) ** 0.1
    t = (np.arange(n) + 1) * 0.08
    d = {"head": {"t": t.tolist(), "p": p.tolist(), "vad": vad.tolist(), "pu": pu.tolist(), "po": po.tolist()}}
    want = [(tt, "others" if path == "others_fallback" else path) for tt, path in EOT.sim_room(d, EOT.FAST)]
    k, fb, thr, others = vad_head_params("steady")
    pol = VadHeadPolicy(0.99, k, fb, thr, others=others)
    got = []
    for v in range(n):
        ev = pol.update(float(p[v]), float(vad[v]), float(pu[v]), float(po[v]))
        if ev is not None:
            got.append((round(float(t[v]), 4), ev["path"]))
    assert got == want and len(got) > 0


def test_steady_does_not_cut_the_bundled_clip():
    """The recorded frames of all six deliveries (tests/fixtures/two_party_call_16s_frames.json): the steady preset ends
    the user's turn after its reference end (10.8 s) and never inside it; served_check --preset fast (runs/
    eot_latency.json served_check_fast) gives 11.616 s (11.776 s at +6 dB) on the real engine."""
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    k, fb, thr, others = vad_head_params("steady")
    assert len(fx["variants"]) == 6
    for name, h in fx["variants"].items():
        pol = VadHeadPolicy(0.99, k, fb, thr, others=others)
        te = [h["t"][v] for v in range(len(h["p"])) if pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v])]
        assert not any(s0 <= t < e1 - 0.08 for t in te), (name, te)
        assert any(e1 - 0.08 <= t < e1 + 1.5 for t in te), (name, te)
