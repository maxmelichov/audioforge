"""Turn head v5 (research/TURN_V5.md): the served SegTurnStream equals the offline window classifier, and the
streaming prosody equals the whole-clip prosody."""
import numpy as np
import torch

from audioforge.heads.prosody import N_FEATS, Prosody, prosody_frames
from audioforge.heads.turn_seg import SegTurn, SegTurnStream, text_feats


def _offline(m, x, ex, n, y, v):
    W = m.win
    a0 = max(0, v - W + 1)
    L = v + 1 - a0
    xx = torch.zeros(1, W, x.shape[1])
    ee = torch.zeros(1, W, ex.shape[1])
    xm = torch.zeros(1, W, dtype=torch.bool)
    xx[0, W - L:] = torch.from_numpy(x[a0: v + 1])
    ee[0, W - L:] = torch.from_numpy(ex[a0: v + 1])
    xm[0, W - L:] = True
    U = m.max_tok
    tok = torch.full((1, U), m.cfg["vocab"], dtype=torch.long)
    tm = torch.zeros(1, U, dtype=torch.bool)
    t = y[: n[v]][-U:]
    if len(t):
        tok[0, U - len(t):] = torch.from_numpy(t)
        tm[0, U - len(t):] = True
    last = max([u for u in range(v + 1) if n[u] > (n[u - 1] if u else 0)], default=-1)
    since = v - last if last >= 0 else v + 1
    tf = torch.tensor([text_feats(int(n[v]), int(n[v - W]) if v - W >= 0 else 0, since)], dtype=torch.float32)
    with torch.no_grad():
        return float(m.prob(xx, xm, ee, tok, tm, tf)[0])


def test_segturn_stream_equals_offline():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = SegTurn(d_in=16, n_extra=3, d=32, n_layers=1, heads=2, ff=64, win=12, d_text=16, text_layers=1,
                max_tok=6).eval()
    T = 40
    x = rng.normal(size=(T, 16)).astype(np.float32)
    ex = rng.uniform(size=(T, 3)).astype(np.float32)
    n = np.minimum(np.cumsum(rng.uniform(size=T) < 0.3), 30).astype(np.int64)
    y = rng.integers(0, 1024, 40).astype(np.int64)
    s = SegTurnStream(m)
    for v in range(T):
        s.push(x[v], *ex[v], int(n[v]))
        for u in (v, max(0, v - 3)):  # the policy may ask for a frame behind the newest one
            assert abs(s.prob(u, list(y)) - _offline(m, x, ex, n, y, u)) < 1e-5


def test_prosody_stream_equals_clip():
    rng = np.random.default_rng(1)
    x = (0.1 * np.sin(np.arange(16000 * 2) * 2 * np.pi * 150 / 16000) * (rng.uniform(size=32000) > 0.3)).astype(np.float32)
    P = prosody_frames(x, 25)
    assert P.shape == (25, N_FEATS)
    q, fr = Prosody(), []
    for i in range(0, 25 * 1280, 320):
        fr += q.feed(x[i:i + 320])
    assert np.abs(np.stack(fr)[:25] - P).max() < 1e-6


# --------------------------------------------------------------------------- presets and the served session
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from audioforge.model import SpeechModel  # noqa: E402
from audioforge.serve import TURN_PRESETS, Session, SessionConfig, VadHeadPolicy, vad_head_params  # noqa: E402
from audioforge.server.constants import ENERGY_GATE, FRAME_MS  # noqa: E402
from audioforge.server.policies import EnergyGate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers_seg", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_v5_presets():
    assert TURN_PRESETS["fast"]["turn_model"] == {"model": "v5", "vad_thr": 0.6, "p": 0.7, "reask": True, "quiet_db": None}
    assert vad_head_params("fast") == (1, 8, 0.4, (12, 8))
    assert TURN_PRESETS["assistant"]["turn_model"]["quiet_db"] == 6.0 and TURN_PRESETS["assistant"]["turn_model"]["p"] == 0.9
    assert vad_head_params("assistant") == (3, 37, 0.4, (12, 8))
    assert "turn_model" not in TURN_PRESETS["balanced"] and "turn_model" not in TURN_PRESETS["steady"]


def test_model_reask():
    """model_reask: an "incomplete" answer is asked again at the next quiet frame (default: once per silence run)."""
    for reask, want in ((False, 1), (True, 5)):
        asked = []
        pol = VadHeadPolicy(0.99, 1, None, 0.4, others=None, turn_model=lambda v, o: (asked.append(v) or 0.1, 0.0),
                            model_reask=reask)
        for vad in [0.9] * 4 + [0.1] * 5:
            pol.update(0.0, vad)
        assert len(asked) == want


def _v5_policy(preset, p5):
    k, fb, thr, others = vad_head_params(preset)
    tmc = TURN_PRESETS[preset]["turn_model"]
    g = ENERGY_GATE
    gate = EnergyGate(quiet_db=tmc["quiet_db"], onset_db=g["onset_db"], window_s=g["window_s"], pct=g["pct"],
                      warmup_frames=int(math.ceil(g["warmup_ms"] / FRAME_MS)))
    return VadHeadPolicy(0.99, k, fb, thr, others=others, gate=gate, turn_model=lambda v, o: (p5[v], 0.0),
                         model_quiet_only=tmc["quiet_db"] is not None, model_vad_thr=tmc["vad_thr"], model_p=tmc["p"],
                         model_reask=tmc["reask"])


def test_fast_does_not_cut_the_bundled_clip():
    """The served frames + v5 p of all six deliveries (tests/fixtures/two_party_call_16s_v5.json, recorded by
    scripts/research/turn_v5.py served_presets on runs/stage1_served_v3.afm, where served == this offline policy):
    --turn-preset fast ends the user's turn after its reference end and never inside it."""
    fx = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_v5.json").read_text())["variants"]
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    assert len(fx) == 6
    for name, h in fx.items():
        pol = _v5_policy("fast", h["p5"])
        te = [h["t"][v] for v in range(len(h["p5"]))
              if pol.update(0.0, h["vad"][v], h["pu"][v], h["po"][v], h["db"][v]) is not None]
        assert not any(s0 <= t < e1 - 0.08 for t in te), (name, te)
        assert any(e1 - 0.08 <= t < e1 + 1.5 for t in te), (name, te)


def _seg_asr_model():
    H = _h()
    m = H._talky_asr_model()
    cfg = json.loads(json.dumps(m.cfg))
    cfg["heads"]["turn_seg"] = {"type": "turn_seg", "weight": 0.0, "d_in": 32, "n_extra": 3, "d": 16, "n_layers": 1,
                                "heads": 2, "ff": 32, "win": 12, "d_text": 8, "text_layers": 1, "max_tok": 6,
                                "block": "1", "dropout": 0.0}
    m2 = SpeechModel(cfg, m.tokenizer).eval()
    sd = m2.state_dict()
    sd.update(m.state_dict())
    m2.load_state_dict(sd)
    return H, m2


def test_v5_preset_needs_the_classifier():
    H = _h()
    with pytest.raises(ValueError, match="v5"):
        H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, turn_preset="fast")
    e = H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1)
    s = Session(e, SessionConfig(turn_policy="vad_head", turn_preset="assistant"))  # falls back with a notice
    assert s.vh_pol.turn_model is None and any("v5" in m["detail"] for m in s.take_notices())


def test_v5_session_runs_the_classifier():
    H, m = _seg_asr_model()
    e = H.EnergyEngine(m, H._diar_model(), name="tiny", threads=1, turn_preset="fast")
    s = Session(e, SessionConfig(turn_policy="vad_head"))
    assert s.asr.seg is not None and s.vh_pol.turn_model is not None and s.vh_pol.model_reask
    # the tiny VAD is always on, so use the assistant preset (energy-quiet frames count for the classifier's clock)
    s = Session(e, SessionConfig(turn_policy="vad_head", turn_preset="assistant"))
    assert s.vh_pol.model_quiet_only and s.vh_pol.gate.quiet_db == 6.0
    x = H._speech_silence()
    for i in range(0, len(x), 320):
        s.process(x[i:i + 320])
    s.finish()
    assert s.asr.seg.v + 1 == s.asr.n_frames
    assert len(s.vh_pol.model_calls) > 0 and all(0.0 <= c["p"] <= 1.0 for c in s.vh_pol.model_calls)


def test_heads_v03_and_v02_still_selectable(tmp_path, monkeypatch):
    from audioforge import hub
    assert hub.HEADS_VERSION == "0.4" and hub.SERVED == "stage1_served_v4.afm"
    assert hub.HEADS["0.2"][3] == "stage1_served_v2.afm" and hub.HEADS["0.1"][3] == "stage1_served.afm"
    seen = {}

    def fake_install(keys, directory, *a, **k):
        seen["heads_version"] = k.get("heads_version", a[5] if len(a) > 5 else None)
        return {}
    monkeypatch.setattr(hub, "install", fake_install)
    hub.main(["--dir", str(tmp_path), "--yes", "--heads-version", "0.2"])
    assert seen["heads_version"] == "0.2"
    hub.main(["--dir", str(tmp_path), "--yes"])
    assert seen["heads_version"] == "0.4"
