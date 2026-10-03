"""The turn presets that ship, read from the shipped heads files themselves (no base model needed, ~30 MB each): the
0.6B v0.4 heads carry their own cfg["turn_presets"], merged by serve.model_presets over TURN_PRESETS; the 115M v0.4
heads carry none and use TURN_PRESETS unchanged. The expected values below are the shipped behaviour (frames of
80 ms); a change to a heads file, to TURN_PRESETS or to the merge shows up here. Also checks that the recorded-frame
fixtures (recorded on stage1_served_v2 / v3) are still what v0.4 computes: every tensor and head config they read is
unchanged in v0.4."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from audioforge import hub
from audioforge.serve import TURN_PRESETS, model_presets, vad_head_params

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
_BLOBS = {}


def _blob(name):
    if name not in _BLOBS:
        p = ROOT / "assets" / name
        if not p.exists():
            pytest.skip(f"{p} is absent")
        _BLOBS[name] = torch.load(p, map_location="cpu", weights_only=True)
    return _BLOBS[name]


def _presets(registry, version):
    cfg = _blob(registry[version][0])["config"]
    return cfg, model_presets(SimpleNamespace(cfg=cfg))


def _params(pr):
    return {name: vad_head_params(name, presets=pr) for name in ("balanced", "fast", "assistant", "steady")}


def test_0p6b_v04_presets_merged_from_its_heads_file():
    cfg, pr = _presets(hub.HEADS_0P6B, "0.4")
    assert {"vad", "turn", "turn_seg", "turn_seg_a", "turn_vad", "speech", "spk"} <= set(cfg["heads"])
    # (K frames, fallback frames, VAD threshold, others (user-silence, hold) frames)
    assert _params(pr) == {"balanced": (4, 9, 0.4, (12, 8)),      # 320 / 720 ms
                           "fast": (1, 9, 0.4, (12, 8)),          # 80 / 720 ms
                           "assistant": (2, 43, 0.4, (12, 8)),    # 160 / 3440 ms
                           "steady": (6, 9, 0.6, (8, 8))}         # not in the file: TURN_PRESETS
    v5 = {"model": "v5", "vad_thr": 0.4, "reask": True}
    assert pr["balanced"]["turn_model"] == {**v5, "p": 0.6, "quiet_db": None}
    assert pr["fast"]["turn_model"] == {**v5, "p": 0.5, "quiet_db": None}
    assert pr["assistant"]["turn_model"] == {**v5, "head": "turn_seg_a", "p": 0.9, "quiet_db": 6.0}
    assert pr["steady"] == TURN_PRESETS["steady"] and "turn_model" not in pr["steady"]
    for name in ("balanced", "fast", "assistant"):
        assert pr[name]["smartturn_fallback_ms"] == 3000 and "theta" not in pr[name]  # theta stays 0.99


def test_115m_v04_heads_use_turn_presets_unchanged():
    cfg, pr = _presets(hub.HEADS, "0.4")
    assert "turn_presets" not in cfg and pr == TURN_PRESETS
    assert {"vad", "turn", "turn_seg", "speech", "spk"} <= set(cfg["heads"]) and "turn_seg_a" not in cfg["heads"]
    assert _params(pr) == {"balanced": (2, 8, 0.4, (12, 8)),      # 160 / 640 ms
                           "fast": (1, 8, 0.4, (12, 8)),          # 80 / 640 ms
                           "assistant": (3, 37, 0.4, (12, 8)),    # 240 / 2960 ms
                           "steady": (6, 9, 0.6, (8, 8))}         # 480 / 720 ms, others 640,640
    assert "turn_model" not in pr["balanced"]
    assert pr["fast"]["turn_model"] == {"model": "v5", "vad_thr": 0.6, "p": 0.7, "reask": True, "quiet_db": None}
    assert pr["assistant"]["turn_model"] == {"model": "v5", "vad_thr": 0.4, "p": 0.9, "reask": True, "quiet_db": 6.0}


@pytest.mark.parametrize("fixture,version,heads", [
    ("two_party_call_16s_frames.json", "0.2", ("vad", "turn", "spk", "diar", "eou")),
    ("two_party_call_16s_v5.json", "0.3", ("vad", "turn", "turn_seg", "spk", "diar", "eou")),
])
def test_recorded_frame_fixtures_still_match_the_v04_heads(fixture, version, heads):
    """The preset tests replay frames recorded on an older served model (test_turn_preset, test_turn_seg,
    test_vad_head_policy). v0.4 = that model + new heads only: every tensor of the recording model's heads file is
    bit-identical in v0.4 and their head configs are equal, so v0.4 emits the same frames (same base, same code).
    test_real_115m.test_v04_rerecords_the_fixture_frames checks it on the real model when it is present."""
    meta = json.loads((FIX / fixture).read_text())
    rec = "stage1_served_v2.afm" if version == "0.2" else "turn_v5.py"
    assert rec in meta.get("source", meta.get("note", ""))
    old, new = _blob(hub.HEADS[version][0]), _blob(hub.HEADS["0.4"][0])
    assert old["tensors"] and set(old["tensors"]) <= set(new["tensors"])
    for k, v in old["tensors"].items():
        assert torch.equal(v, new["tensors"][k]), k
    for h in heads:
        assert old["config"]["heads"][h] == new["config"]["heads"][h], h
    assert old["config"]["encoder"] == new["config"]["encoder"]
    assert old["base"] == new["base"]
