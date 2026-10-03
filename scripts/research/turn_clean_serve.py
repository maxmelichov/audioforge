"""plans/fixwave/turn_clean.md: the 115M served-heads v0.5 candidate with the clean (held-out) presets, not the
default, and its latency gate. Reads runs/turn_clean.json heldout.115m (turn_clean.py pick).

  build115 --base <115M .nemo>   stage1_served_v4.afm + the picked classifiers + cfg turn_presets
                                 -> TCW/stage1_served_v5_cand.afm, assets/served_heads_v0.5_candidate.pt
  cost115 --device mps|cpu       engine ms per 160 ms chunk, v0.4 vs the candidate, interleaved, per preset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from turn_clean import T6, TCW, V5W, load_json, log, save  # noqa: E402


def preset_cfg(track, r, shipped_track):
    """A rule twin -> the served cfg["turn_presets"] entry (serve.model_presets merges it over TURN_PRESETS)."""
    o = {"vad_wait_ms": [r["k"] * 80, r["fb"] * 80], "vad_thr": r["vad_thr"],
         "others_wait_ms": [x * 80 for x in (r.get("others") or (12, 8))]}
    if r["mode"] == "head":
        o["theta"] = r["th"]
        return o
    tm = {"vad_thr": r["mvt"], "p": r["mp"], "reask": bool(r.get("reask")), "quiet_db": r.get("quiet_db")}
    if track != shipped_track:
        tm["head"] = "turn_seg_a"
    o["turn_model"] = tm
    return o


def stage_build115(a):
    """The 115M served-heads v0.5 candidate with the clean presets (not the default): stage1_served_v4.afm with
    heads.turn_seg = the fast pick's classifier, heads.turn_seg_a = the assistant pick's when it differs, and
    cfg turn_presets -> TCW/stage1_served_v5_cand.afm + assets/served_heads_v0.5_candidate.pt (hub.export_heads)."""
    import copy

    import torch

    from audioforge import hub
    from audioforge.model import SpeechModel
    from audioforge.train import load_model, save_model
    ho = {pn: dict(h, pick=h.get("pick_servable", h["pick"])) for pn, h in load_json("heldout")["115m"].items()}
    base = load_model(str(ROOT / "runs" / "stage1_served_v4.afm"), "cpu")
    cfg = copy.deepcopy(base.cfg)
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    main = ho["fast"]["pick"]["track"]
    cls = {main}
    for pn in ("balanced", "assistant"):
        if ho[pn]["pick"]["track"] not in ("head", main):
            cls.add(ho[pn]["pick"]["track"])
    assert len(cls) <= 2, cls
    second = next((t for t in cls if t != main), None)
    for hname, tag in (("turn_seg", main), ("turn_seg_a", second)):
        if tag is None:
            continue
        ck = torch.load(V5W / tag / "model.pt", map_location="cpu", weights_only=False)
        cfg["heads"][hname] = {"type": "turn_seg", "weight": 0.0, **ck["cfg"]}
        for k in [k for k in sd if k.startswith(f"heads.{hname}.")]:
            del sd[k]
        sd.update({f"heads.{hname}.{k}": v for k, v in ck["state_dict"].items()})
    cfg["turn_presets"] = {pn: preset_cfg(ho[pn]["pick"]["track"], ho[pn]["pick"]["rule"], main) for pn in ho}
    cfg["name"] = "stage1_served_v5_candidate"
    m = SpeechModel(cfg, base.tokenizer)
    m.load_state_dict(sd, strict=True)
    out = TCW / "stage1_served_v5_cand.afm"
    save_model(m.eval(), out)
    heads = ROOT / "assets" / "served_heads_v0.5_candidate.pt"
    info = hub.export_heads(out, Path(a.base), heads)
    got = hub.build_served(Path(a.base), heads, TCW / "rebuilt.afm")
    assert got == info["state_hash"], (got, info)
    rr = T6.served_rules(out)
    for pn in ho:  # the served model resolves each preset to the picked rule
        want = ho[pn]["pick"]["rule"]
        assert all(rr[pn].get(k) == want.get(k) for k in ("k", "fb", "vad_thr", "mode")), (pn, rr[pn], want)
    rec = {"afm": str(out), "heads": str(heads), "size": heads.stat().st_size, "sha256": hub.sha256_file(heads),
           "turn_seg": main, "turn_seg_a": second, "turn_presets": cfg["turn_presets"], **info}
    log(rec)
    save("build115", rec)


def stage_cost115(a):
    """Latency gate of the v0.5 candidate: the full single-mode served engine (mps_115m protocol: bundled two-party clip,
    its print, warm-up, best of 3) for stage1_served_v4 and the candidate, interleaved old / new per preset, ms per
    160 ms chunk; and the turn_end times of every preset on that clip (balanced / fast must be identical)."""
    import mps_115m as M
    import torch

    import audioforge.serve as S
    from audioforge.server.cli import MODES
    torch.set_num_threads(2)
    pcm = M.load_clip().astype(np.float32) / 32768.0
    pr = json.loads(M.PRINT.read_text())
    afms = {"v0.4": ROOT / "runs" / "stage1_served_v4.afm", "v0.5_candidate": TCW / "stage1_served_v5_cand.afm"}
    opts = {**MODES["single"], "enroll": "explicit",
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True}
    eng = {k: S.Engine.load(str(v), None, a.device, threads=2, **opts) for k, v in afms.items()}
    for e in eng.values():
        e.warmup()
    res = {}
    for preset in ("balanced", "fast", "assistant"):
        runs = {k: [] for k in eng}
        ends = {}
        for _ in range(3):
            for k, e in eng.items():
                s_ = S.Session(e, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
                s_.arm_enrollment("enroll", 0, embedding=pr)
                t0, ev = time.perf_counter(), []
                for i in range(0, len(pcm), M.CHUNK):
                    ev += [m for m in (s_.process(pcm[i:i + M.CHUNK]) or []) if isinstance(m, dict) and m.get("type") == "turn_end"]
                ev += [m for m in (s_.finish() or []) if isinstance(m, dict) and m.get("type") == "turn_end"]
                runs[k].append({"rtf": (time.perf_counter() - t0) / (len(pcm) / 16000), "chunk_ms": list(s_.chunk_ms)})
                ends[k] = [round(float(m.get("t", m.get("time", -1))), 3) for m in ev]
        r = {}
        for k, rr in runs.items():
            c = np.array(min(rr, key=lambda x: x["rtf"])["chunk_ms"])
            r[k] = {"chunk_ms_p50": round(float(np.median(c)), 2), "chunk_ms_p95": round(float(np.percentile(c, 95)), 2),
                    "turn_end_t": ends[k]}
        r["delta_p50_ms"] = round(r["v0.5_candidate"]["chunk_ms_p50"] - r["v0.4"]["chunk_ms_p50"], 2)
        r["turn_ends_identical"] = ends["v0.4"] == ends["v0.5_candidate"]
        res[preset] = r
        log(preset, {k: (v["chunk_ms_p50"], v["chunk_ms_p95"], v["turn_end_t"]) for k, v in r.items() if isinstance(v, dict)},
            r["delta_p50_ms"], r["turn_ends_identical"])
    save("cost115", {a.device: res})


def stage_served115(a):
    """Served == offline for the candidate's assistant preset (the bundled call has no assistant turn_end): --n held-out
    st3 clips (smart-turn train split + 3 s silence, no print, as core_0p6b_turn hoprep) through the served v0.5
    candidate; their turn_end decision times against the turn_v5.run_policy twin on the cached held-out signals."""
    import torch
    import turn_clean as TC
    import turn_v5 as V5

    import audioforge.serve as S
    from audioforge.datasets import smartturn as ST
    from audioforge.server.cli import MODES
    torch.set_num_threads(2)
    ho = load_json("heldout")["115m"]["assistant"]
    h = ho["shipped"] if a.which == "v0.4" else ho.get("pick_servable", ho["pick"])
    rule = dict(h["rule"], others=tuple(h["rule"]["others"]))
    U = TC.units("115m", h["track"])["ho:st3"][: a.n]
    meta_st = json.loads(ST.build_cache(verbose=False).read_text())
    wav = np.load(ST.DEFAULT_ROOT / "cache" / "human_5_all.npy", mmap_mode="r")
    o = meta_st["offsets"]
    opts = {**MODES["single"], "enroll": "explicit",
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True}
    afm = ROOT / "runs" / "stage1_served_v4.afm" if a.which == "v0.4" else TCW / "stage1_served_v5_cand.afm"
    eng = S.Engine.load(str(afm), None, a.device, threads=2, **opts)
    eng.warmup()
    rows, same = [], 0
    for key, hh, db, _ss in U:
        idx = int(key.split("_")[1])
        x = np.concatenate([np.asarray(wav[o[idx]: o[idx + 1]], np.float32), np.zeros(int(3.0 * 16000), np.float32)])
        s_ = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset="assistant"))
        ev = []
        for i in range(0, len(x), 2560):
            ev += [m for m in (s_.process(x[i:i + 2560]) or []) if isinstance(m, dict) and m.get("type") == "turn_end"]
        ev += [m for m in (s_.finish() or []) if isinstance(m, dict) and m.get("type") == "turn_end"]
        served = [round(float(m["t"]), 3) for m in ev]
        off = [round(t, 3) for t, _ in V5.run_policy({"head": hh}, db, rule, {})]
        rows.append({"clip": key, "served": served, "offline": off})
        same += served == off
    log(f"served115 {a.which}: {same}/{len(rows)} clips identical", rows[:3])
    d = load_json("served115", {})
    d = d if "rows" not in d else {}
    d[a.which] = {"track": h["track"], "rule": rule, "n": len(rows), "identical": same, "rows": rows}
    save("served115", d)


STAGES = {"build115": stage_build115, "cost115": stage_cost115, "served115": stage_served115}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--base", default="/Volumes/ExternalSSD/nvidia-audio-models/data/nemo/stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--which", default="v0.5_candidate", choices=("v0.4", "v0.5_candidate"))
    STAGES[ap.parse_known_args()[0].stage](ap.parse_known_args()[0])


if __name__ == "__main__":
    main()
