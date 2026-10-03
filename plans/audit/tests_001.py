# /// script
# requires-python = ">=3.10"
# dependencies = ["audioforge"]
#
# [tool.uv.sources]
# audioforge = { path = "../..", editable = true }
# ///
"""Companion to plans/audit/tests_001.md: which test gates are open on this machine, the turn presets the shipped heads
carry, and (--hash) whether the legacy served .afm files in runs/ are exactly base + heads (hub.state_hash).

    uv run plans/audit/tests_001.py            # cheap: file gates + presets (loads two ~27 MB heads files)
    uv run plans/audit/tests_001.py --hash     # also loads each served .afm on CPU (~0.5-2.5 GB each)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", action="store_true", help="verify runs/*.afm state hashes against the heads blobs")
    a = ap.parse_args()
    import torch

    from audioforge import hub
    print("== gates")
    for k in ("asr", "tsvad", "lid", "asr_0p6b", "tsvad_0p6b", "lid_0p6b"):
        print(f"  find_model({k!r}) = {hub.find_model(k)}")
    for name in ("RUN_REAL", "AUDIOFORGE_BIG_TESTS"):
        print(f"  ${name} = {os.environ.get(name)!r}")
    print(f"  cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}")
    print(f"  0.6B base .nemo present: {(ROOT / 'data/nemo' / hub.COMPONENTS['asr_0p6b'].filename).exists()}")
    print("== turn presets carried by the shipped heads (serve.model_presets merges these over TURN_PRESETS)")
    pairs = [(hub.HEADS, hub.HEADS_VERSION), (hub.HEADS_0P6B, hub.HEADS_0P6B_VERSION)]
    for reg, ver in pairs:
        name = reg[ver][0]
        blob = torch.load(ROOT / "assets" / name, map_location="cpu", weights_only=False)
        print(f"  {name}: heads={sorted(blob['config']['heads'])}")
        print(f"    turn_presets={json.dumps(blob['config'].get('turn_presets'))}")
    if a.hash:
        from audioforge.train import load_model
        print("== served .afm == base + heads (state_hash)")
        for reg in (hub.HEADS, hub.HEADS_0P6B):
            for ver, (name, _size, _sha, out) in reg.items():
                afm = ROOT / "runs" / out
                if not afm.exists():
                    print(f"  {out}: absent")
                    continue
                want = torch.load(ROOT / "assets" / name, map_location="cpu", weights_only=False)["state_hash"]
                got = hub.state_hash(load_model(afm, "cpu").state_dict())
                print(f"  {out}: {'match' if got == want else 'MISMATCH'}")


if __name__ == "__main__":
    main()
