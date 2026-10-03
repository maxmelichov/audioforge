"""Shared pytest setup: make the research drivers importable.

The tests import the research scripts by module name (``import eval_stage1``) or by file path. The scripts live in
``scripts/research/`` (user-facing ones stay in ``scripts/``); both directories go on ``sys.path`` here so the tests
and the scripts' own sibling imports resolve without per-file path juggling.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT, ROOT / "scripts", ROOT / "scripts" / "research"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import os

import pytest

CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"  # the bundled single-mode clip (real speech)
CLIP_META = ROOT / "examples" / "audio" / "two_party_call_16s.json"
CLIP_PRINT = ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long soak / timing tests (still run by default)")
    config.addinivalue_line("markers", "real: loads the shipped checkpoints (skipped when they are absent)")


@pytest.fixture(autouse=True)
def _restore_threads_and_hub_env():
    """Every test leaves the torch thread count and HF_HUB_OFFLINE as it found them, so a file run alone behaves as it
    does inside the full suite."""
    import torch
    n, off = torch.get_num_threads(), os.environ.get("HF_HUB_OFFLINE")
    yield
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)
    if off is None:
        os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = off


def served_0p6b():
    """(served 0.6B v0.4 .afm, TS-VAD head) of the shipped 0.6B core. Skips only when the NVIDIA base .nemo is absent;
    otherwise a missing served model is built once (hub.build_served: base + assets/served_heads_0p6b_v0.4.pt, hash
    checked) into runs/, where hub.find_model and audioforge.load(core="0.6b") pick it up."""
    from audioforge import hub
    from audioforge.paths import DATA_ROOT
    t = hub.find_model("tsvad_0p6b")
    p = hub.find_model("asr_0p6b")
    if p is None:
        base = DATA_ROOT / "nemo" / hub.COMPONENTS["asr_0p6b"].filename
        if not base.exists():
            pytest.skip(f"{base} is absent (the NVIDIA 0.6B base; audioforge-download --core 0.6b)")
        heads = hub.heads_path(None, hub.models_dir(), hub.HEADS_0P6B_VERSION, hub.HEADS_0P6B)
        p = ROOT / "runs" / hub.SERVED_0P6B
        p.parent.mkdir(exist_ok=True)
        hub.build_served(base, heads, p)
    assert t is not None, "tsvad_0p6b.pt ships in assets/"
    return p, t


def need_real_115m():
    from audioforge import hub
    for key in ("asr", "tsvad"):
        if hub.find_model(key) is None:
            pytest.skip(f"the shipped 115M '{key}' is absent (audioforge-download)")


@pytest.fixture(scope="session")
def real_115m():
    """audioforge.load() on the shipped 115M v0.4 stack (single mode, CPU, 2 threads), loaded once per session."""
    need_real_115m()
    import audioforge
    return audioforge.load(warmup=False)
