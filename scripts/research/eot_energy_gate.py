"""research/EOT_ASSISTANT.md "Energy gate": the energy-aware quiet gate of vad_head (``--quiet-gate energy``) and the
``--turn-model smartturn`` bridge, scored offline on the three end-of-turn harnesses with the SERVED classes.

The served VAD head reads ~0.66 on -50 dBFS room tone and ~0.55 on a fresh session's first frames, so vad_head's
silence (VAD < thr) starts only at digital silence and its 640 ms fallback can fire before the user speaks. The gate
(audioforge.server.policies.EnergyGate) adds a per-session noise floor (the 10th percentile of the 80 ms frame log
energies of the last 3 s): a frame is quiet if VAD < thr OR energy < floor + X dB, a turn is armed only by an onset
(VAD > 0.5 AND energy > floor + Y dB), and nothing fires before 320 ms of onset frames (warm-up guard).

The frames the rule reads (turn head p, served VAD, P(user) / P(other), decision-ready times) do not depend on the
rule, so every rule is replayed through ``audioforge.server.policies.VadHeadPolicy`` + ``EnergyGate`` (the served
objects, not a re-implementation) on the stored per-frame dumps, with the frame energies recomputed from the same
audio (``frame_db`` of samples [1280 v, 1280 (v + 1)), as ``Session`` computes them):

  * assistant: scripts/research/eot_assistant.py's 399 smart-turn v3.2-test clips (its dump, sessions, score_system)
  * calls / AMI: scripts/research/eot_latency.py's dump (E2E calls user channels, 200 AMI dev turns; score_session)
  * the bundled clip's six deliveries (eot_latency CLIP_VARIANTS; the recorded frames in
    tests/fixtures/two_party_call_16s_frames.json): no cut

  sweep       X in 3..12 dB x Y x floor admission x warm-up, both presets -> runs/eot_energy_gate.json "sweep"
  smartturn   --turn-model smartturn (smart-turn v3.2 ONNX at the quiet trigger, Pipecat's input preparation) on the
              three harnesses, offline with the served SmartTurnModel -> "smartturn" (onnx; gate)
  served      the served eot_assistant dumps (dump_gate, dump_smartturn) scored + checked against the replay

    PYTHONPATH=. .venv/bin/python scripts/research/eot_energy_gate.py sweep
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_energy_gate.py smartturn
    PYTHONPATH=. .venv/bin/python scripts/research/eot_energy_gate.py served   # after the two served dumps
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import eot_assistant as A  # noqa: E402
import eot_latency as E  # noqa: E402

from audioforge.server.policies import EnergyGate, VadHeadPolicy, frame_db  # noqa: E402

OUT = ROOT / "runs" / "eot_energy_gate.json"
CACHE = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/eot_energy_gate")
FS = 1280
BAL = {"k": 2, "fb": 8, "thr": 0.4, "others": (12, 8)}
FAST = {"k": 6, "fb": 9, "thr": 0.6, "others": (8, 8)}
PRESETS = {"balanced": BAL, "fast": FAST}
log = E.log


def energies(x: np.ndarray, t) -> np.ndarray:
    """Per recorded frame v, the log energy of samples [1280 v, min(1280 (v + 1), ready sample)) with the ready sample
    = the frame's decision-ready time (``Session._energy``: an ASR frame can be ready before its 80 ms end)."""
    return np.asarray([frame_db(x[v * FS:min((v + 1) * FS, int(round(float(tv) * E.SR)))]) if int(round(float(tv) * E.SR)) > v * FS
                       else -100.0 for v, tv in enumerate(t)], np.float64)


def save(key, val):
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[key] = val
    prev["generated"] = time.strftime("%Y-%m-%d %H:%M")
    OUT.write_text(json.dumps(prev, indent=1, default=float))


# --------------------------------------------------------------------------- data
def ctx():
    """{assistant: (sess, dump, comp, en), calls_ami: (sess, dump, comp, en), clip: (frames, en, meta)}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    adump = A.load_dump()
    asess = A.sessions(adump)
    acomp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in adump.items()}
    f = CACHE / "energy_assistant_ready.json"
    if f.exists():
        aen = {k: np.asarray(v) for k, v in json.loads(f.read_text()).items()}
    else:
        aen = {s["key"]: energies(A.audio(s["clip"]), adump[s["key"]]["head"]["t"]) for s in asess}
        f.write_text(json.dumps({k: v.round(3).tolist() for k, v in aen.items()}))
    ldump = E.load_dump()
    lsess = [s for s in E.sessions() if s["key"] in ldump]
    lcomp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in ldump.items()}
    f = CACHE / "energy_calls_ami_ready.json"
    if f.exists():
        len_ = {k: np.asarray(v) for k, v in json.loads(f.read_text()).items()}
    else:
        len_ = {s["key"]: energies(E.read_audio(s), ldump[s["key"]]["head"]["t"]) for s in lsess}
        f.write_text(json.dumps({k: v.round(3).tolist() for k, v in len_.items()}))
    from audioforge.data import load_wav
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), E.SR).astype(np.float32)
    frames = json.loads((ROOT / "tests" / "fixtures" / "two_party_call_16s_frames.json").read_text())["variants"]
    xs = {var: fn(x0) for var, fn in E.CLIP_VARIANTS.items()}
    cen = {var: energies(xs[var], frames[var]["t"]) for var in xs}
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    return {"assistant": (asess, adump, acomp, aen), "calls_ami": (lsess, ldump, lcomp, len_),
            "clip": (frames, cen, meta, xs)}


# --------------------------------------------------------------------------- the served policy on a dump
class CachedModel:
    """SmartTurnModel.predict memoised on the segment (the trigger points of many rules share segments); ms = the
    first measured call of that segment."""

    def __init__(self, model):
        self.m, self.cache, self.n_run = model, {}, 0

    def predict(self, seg):
        k = (len(seg), hash(np.asarray(seg, np.float32).tobytes()))
        if k not in self.cache:
            self.cache[k] = self.m.predict(seg)
            self.n_run += 1
        return self.cache[k]


def run(head: dict, en: np.ndarray | None, preset: dict, gate: dict | None, track: bool, fb: int | None = -1,
        model=None, x: np.ndarray | None = None) -> list[tuple[float, str, float]]:
    """VadHeadPolicy (+ EnergyGate(**gate), + the smart-turn trigger on audio x when ``model``) over one session's
    frames -> [(decision-ready t, path, model ms of the deciding call or 0)]."""
    from audioforge.server.smartturn import SmartTurnTrigger
    t, p, vad = head["t"], head["p"], head["vad"]
    tm = None
    if model is not None:
        tm = SmartTurnTrigger(model, lambda a, b: x[a:b], lambda v: int(round(float(t[v]) * E.SR)))
    g = EnergyGate(**gate) if gate is not None else None
    pol = VadHeadPolicy(preset.get("th", 0.99), preset["k"], preset["fb"] if fb == -1 else fb, preset["thr"],
                        others=preset["others"], gate=g, turn_model=tm, model_quiet_only=preset.get("mq", False),
                        model_vad_thr=preset.get("mthr"), model_p=preset.get("mp", 0.5))
    pu, po = head.get("pu"), head.get("po")
    out = []
    for v in range(len(p)):
        ev = pol.update(p[v], vad[v], pu[v] if track else None, po[v] if track else None,
                        None if en is None else float(en[v]))
        if ev is not None:
            ms = tm.calls[-1]["ms"] if ev["path"] == "model" else 0.0
            out.append((round(float(t[v]), 4), ev["path"], ms / 1000))
            if tm is not None:
                tm.turn_ended(int(round(float(t[v]) * E.SR)))
    return out


def score_calls_ami(c, preset, gate, fb=-1, model=None) -> dict:
    sess, dump, comp, en = c[:4]
    per = {sc: [] for sc in E.SCOPES}
    for ss in sess:
        d = dump[ss["key"]]
        te = run(d["head"], en[ss["key"]], preset, gate, bool(d.get("has_print")), fb, model,
                 E.read_audio(ss) if model is not None else None)
        t = [x + comp[ss["key"]] + m for x, _, m in te]
        cc = [comp[ss["key"]] + m for _, _, m in te]
        for sc, sets in E.SCOPES.items():
            if ss["set"] in sets:
                per[sc].append(E.score_session(t, ss, cc))
    return {"calls": E.pool(per["two_party_user"]), "ami": E.pool(per["ami"])}


def score_assistant(c, preset, gate, fb=-1, model=None) -> dict:
    sess, dump, comp, en = c[:4]
    times, cc, paths = {}, {}, {}
    for s in sess:
        d = dump[s["key"]]
        te = run(d["head"], en[s["key"]], preset, gate, False, fb, model, A.audio(s["clip"]) if model is not None else None)
        times[s["key"]] = [x + comp[s["key"]] + m for x, _, m in te]
        cc[s["key"]] = [comp[s["key"]] + m for _, _, m in te]
        paths[s["key"]] = te
    r = A.score_system(sess, times, cc)
    # which path answers each clip (first turn_end at or after e - 80 ms, after the speech onset)
    cnt = {}
    for s in sess:
        e = s["user_turns"][0][1]
        ev = [pth for t, pth, _ in paths[s["key"]] if t >= max(s["onset"], e - E.TOL)]
        key = ("complete " if s["complete"] else "incomplete ") + (ev[0] if ev else "none")
        cnt[key] = cnt.get(key, 0) + 1
    r["answering_path"] = cnt
    return r


def clip_check(c, preset, gate, fb=-1, model=None) -> dict:
    frames, cen, meta, xs = c
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    out = {}
    for var, h in frames.items():
        te = [t for t, _, _ in run(h, cen[var], preset, gate, True, fb, model, xs[var])]
        hit = [t for t in te if t >= e1 - E.TOL]
        out[var] = {"turn_ends": te, "cut": any(s0 <= t < e1 - E.TOL for t in te),
                    "answer_ms": round((hit[0] - e1) * 1000) if hit else None}
    return {"any_cut": any(v["cut"] for v in out.values()), "variants": out}


def row(c, preset, gate, fb=-1, model=None) -> dict:
    a = score_assistant(c["assistant"], preset, gate, fb, model)
    l_ = score_calls_ami(c["calls_ami"], preset, gate, fb, model)
    k = clip_check(c["clip"], preset, gate, fb, model)
    return {"assistant": a, **l_, "clip": k}


def brief(r) -> str:
    a, c, m = r["assistant"], r["calls"], r["ami"]
    return (f"asst {a['complete']['eot_total_ms_p50']}/{a['complete']['eot_total_ms_p95']} FF {a['incomplete']['false_fire_pct']} "
            f"miss {a['complete']['missed_pct']} early {a['complete']['early_fire_pct']} acc {a['accuracy_pct']} pre {a['clips_with_pre_speech_turn_end']} | "
            f"calls {c['eot_total_ms_p50']}/{c['eot_total_ms_p95']} {c['false_interruption_pct']}/{c['missed_pct']} | "
            f"AMI {m['eot_total_ms_p50']}/{m['eot_total_ms_p95']} {m['false_interruption_pct']}/{m['missed_pct']} | "
            f"clip cut {r['clip']['any_cut']} ans {[v['answer_ms'] for v in r['clip']['variants'].values()]}")


LIMITS = {"balanced": {"calls_fi": 20.2, "calls_miss": 7.3, "ami_fi": 10.5, "ami_miss": 33.5},
          "fast": {"calls_fi": 24.8, "calls_miss": 3.7, "ami_fi": 16.0, "ami_miss": 30.5}}


def holds(r, preset_name) -> bool:
    g = LIMITS[preset_name]
    return (r["calls"]["false_interruption_pct"] <= g["calls_fi"] and r["calls"]["missed_pct"] <= g["calls_miss"]
            and r["ami"]["false_interruption_pct"] <= g["ami_fi"] and r["ami"]["missed_pct"] <= g["ami_miss"]
            and not r["clip"]["any_cut"])


def slim(r):
    return {"assistant": {k: r["assistant"][k] for k in ("complete", "incomplete", "accuracy_pct",
                                                         "clips_with_pre_speech_turn_end", "answering_path")},
            "calls": r["calls"], "ami": r["ami"],
            "clip": {"any_cut": r["clip"]["any_cut"],
                     "answer_ms": {k: v["answer_ms"] for k, v in r["clip"]["variants"].items()}}}


SHIPPED_GATE = {"quiet_db": None, "onset_db": 6.0, "warmup_frames": 2}  # constants.ENERGY_GATE (160 ms warm-up)


def cmd_sweep(a):
    """Three gate families on both presets: (q) quiet = VAD < thr OR energy < floor + X (onset Y = X + 3, warm-up 4
    frames) with the fallback re-tuned; (o) onset arming + warm-up only (the silence stays the VAD's); (d) digital
    silence: energy < an absolute dBFS counts as quiet (+ onset Y 6, warm-up 2)."""
    c = ctx()
    res = {"baseline": {}, "grid": [], "best": {}}
    for pn, pr in PRESETS.items():
        r = row(c, pr, None)
        res["baseline"][pn] = slim(r)
        log(pn, "VAD only", brief(r))
    grid = []
    for X, fb in itertools.product((3, 4, 6, 8, 10, 12), (0, 4, 8)):
        grid.append(("q", {"quiet_db": X, "onset_db": X + 3, "warmup_frames": 4}, fb))
    for Y, wu in itertools.product((3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4)):
        grid.append(("o", {"quiet_db": None, "onset_db": Y, "warmup_frames": wu}, 0))
    for ab in (-90, -85, -80, -75):
        grid.append(("d", {**SHIPPED_GATE, "abs_db": ab}, 0))
    for pn, pr in PRESETS.items():
        for fam, gt, dfb in grid:
            r = row(c, pr, gt, pr["fb"] + dfb)
            ok = holds(r, pn)
            res["grid"].append({"preset": pn, "family": fam, "gate": gt, "fallback_ms": (pr["fb"] + dfb) * 80,
                                "holds": ok, **slim(r)})
            log(pn, fam, gt, (pr["fb"] + dfb) * 80, "OK " if ok else "-- ", brief(r))
    for pn in PRESETS:
        cand = [g for g in res["grid"] if g["preset"] == pn and g["holds"]]
        cand.sort(key=lambda g: (g["assistant"]["complete"]["eot_total_ms_p50"], -g["assistant"]["accuracy_pct"],
                                 g["assistant"]["clips_with_pre_speech_turn_end"]))
        res["best"][pn] = cand[:5]
        n_q = sum(g["holds"] for g in res["grid"] if g["preset"] == pn and g["family"] == "q")
        log(pn, f"quiet-gate rows that hold calls/AMI/clip: {n_q}")
    for pn, pr in PRESETS.items():
        res["shipped_" + pn] = slim(row(c, pr, SHIPPED_GATE))
        log("shipped gate", pn, brief(row(c, pr, SHIPPED_GATE)))
    save("sweep", res)


# (preset, energy quiet X or None, trigger frames, fallback frames, two clocks, P(complete) threshold, model VAD thr)
# one clock: the trigger and the fallback count the same silence; two clocks ("mq"): the classifier on energy-or-VAD
# quiet, the fallback timer on the VAD's own silence
ST_CONFIGS = {
    "vad trigger balanced (--smartturn-trigger vad): VAD < 0.4 160 ms, fallback 3 s": ("balanced", None, 2, 38, False, 0.5, None),
    "vad trigger fast: VAD < 0.6 160 ms, fallback 3 s": ("fast", None, 2, 38, False, 0.5, None),
    "vad trigger 80 ms, fallback 3 s": ("balanced", None, 1, 38, False, 0.5, None),
    "vad trigger 240 ms, fallback 3 s": ("balanced", None, 3, 38, False, 0.5, None),
    "vad trigger 160 ms, fallback 2 s": ("balanced", None, 2, 25, False, 0.5, None),
    "vad trigger 160 ms, fallback 1 s": ("balanced", None, 2, 12, False, 0.5, None),
    "vad trigger 160 ms, fallback 640 ms": ("balanced", None, 2, 8, False, 0.5, None),
    "one clock, energy X 6 dB 160 ms, fallback 3 s": ("balanced", 6.0, 2, 38, False, 0.5, None),
    "one clock, energy X 3 dB 160 ms, fallback 3 s": ("balanced", 3.0, 2, 38, False, 0.5, None),
    "two clocks, energy X 6 160 ms, p > 0.5, VAD fallback 640": ("balanced", 6.0, 2, 8, True, 0.5, None),
    "two clocks, energy X 6 240 ms, p > 0.5, VAD fallback 640": ("balanced", 6.0, 3, 8, True, 0.5, None),
    "two clocks, energy X 3 160 ms, p > 0.5, VAD fallback 640": ("balanced", 3.0, 2, 8, True, 0.5, None),
    "two clocks, VAD < 0.5 160 ms, p > 0.5, VAD fallback 640": ("balanced", None, 2, 8, True, 0.5, 0.5),
    "two clocks, energy X 6 160 ms, p > 0.97, VAD fallback 640": ("balanced", 6.0, 2, 8, True, 0.97, None),
    "two clocks, energy X 6 240 ms, p > 0.9, VAD fallback 640": ("balanced", 6.0, 3, 8, True, 0.9, None),
    "two clocks balanced (--smartturn-trigger energy): X 6 240 ms, p > 0.97, VAD fallback 640": ("balanced", 6.0, 3, 8, True, 0.97, None),
    "two clocks fast (--smartturn-trigger energy): X 6 240 ms, p > 0.97, VAD fallback 720": ("fast", 6.0, 3, 9, True, 0.97, None),
    "two clocks, energy X 6 240 ms, p > 0.99, VAD fallback 640": ("balanced", 6.0, 3, 8, True, 0.99, None),
    "two clocks, energy X 6 320 ms, p > 0.9, VAD fallback 640": ("balanced", 6.0, 4, 8, True, 0.9, None),
}


def cmd_smartturn(a):
    """--turn-model smartturn offline on the three harnesses: the served SmartTurnModel + SmartTurnTrigger inside the
    served VadHeadPolicy (EnergyGate as shipped, plus an energy-quiet trigger variant), total = decision + chunk p50 +
    the deciding smart-turn call's measured compute."""
    from audioforge.server.smartturn import SmartTurnModel
    c = ctx()
    m = SmartTurnModel(threads=2)
    cm = CachedModel(m)
    res = {"model": m.path, "features": m.features_impl, "threads": 2, "rows": {}}
    for name, (pn, X, k, fb, mq, mp, mthr) in ST_CONFIGS.items():
        t0 = time.time()
        r = row(c, {**PRESETS[pn], "k": k, "mq": mq, "mp": mp, "mthr": mthr}, {**SHIPPED_GATE, "quiet_db": X}, fb,
                model=cm)
        res["rows"][name] = slim(r)
        log(name, brief(r), f"{time.time() - t0:.0f} s, {cm.n_run} model runs")
    ms = np.asarray([v[1] for v in cm.cache.values()])
    res["compute_ms_per_call"] = {"n": int(len(ms)), "p50": round(float(np.median(ms)), 1),
                                  "p95": round(float(np.percentile(ms, 95)), 1)}
    log("compute per call", res["compute_ms_per_call"])
    save("smartturn", res)


def cmd_served(a):
    """The served dumps (eot_assistant dump with EOT_ASSISTANT_DUMP=dump_gate / dump_smartturn): scored as eot_assistant
    scores its own, and checked against this script's offline replay of the same configuration."""
    from audioforge.server.smartturn import SmartTurnModel
    c = ctx()
    sess = c["assistant"][0]
    out = {}
    cm = None
    for name, sub, kw, gt in (("gate", "dump_gate", {}, SHIPPED_GATE),
                              ("smartturn", "dump_smartturn", {"k": 2, "fb": 38}, SHIPPED_GATE),
                              ("smartturn_energy", "dump_smartturn_energy", {"k": 3, "mq": True, "mp": 0.97},
                               {**SHIPPED_GATE, "quiet_db": 6.0})):
        d = A.WORK / sub
        dump = {x["key"]: x for x in (json.loads(q.read_text()) for q in sorted(d.glob("*.json")))}
        if len(dump) < len(sess):
            log(f"{sub}: {len(dump)} / {len(sess)} clips; skipped")
            continue
        comp = {k: v["chunk_ms"]["p50"] / 1000 for k, v in dump.items()}
        times = {k: [m["t"] + comp[k] + (m.get("model_ms") or 0) / 1000 for m in v["turn_ends"]] for k, v in dump.items()}
        cc = {k: [comp[k] + (m.get("model_ms") or 0) / 1000 for m in v["turn_ends"]] for k, v in dump.items()}
        r = A.score_system(sess, times, cc)
        r_raw = A.score_system(sess, times, cc, drop_pre=False)
        # offline replay of the same configuration on these frames
        st = name.startswith("smartturn")
        if st and cm is None:
            cm = CachedModel(SmartTurnModel(threads=2))
        differ = 0
        for s in sess:
            v = dump[s["key"]]
            off = run(v["head"], c["assistant"][3][s["key"]], {**BAL, **kw}, gt, False,
                      kw.get("fb", -1), cm if st else None, A.audio(s["clip"]) if st else None)
            differ += [round(t, 3) for t, _, _ in off] != [m["t"] for m in v["turn_ends"]]
        mm = [m["ms"] for v in dump.values() for m in (v.get("turn_model_calls") or [])]
        out[name] = {"assistant": r, "incl_pre_speech_accuracy_pct": r_raw["accuracy_pct"],
                     "offline_replay_differs_clips": differ, "n": len(sess),
                     "compute_ms_chunk_p50_median": round(1000 * float(np.median(list(comp.values()))), 1)}
        if mm:
            out[name]["smartturn_calls"] = {"n": len(mm), "per_clip": round(len(mm) / len(sess), 2),
                                            "ms_p50": round(float(np.median(mm)), 1),
                                            "ms_p95": round(float(np.percentile(mm, 95)), 1)}
        cmp_ = r["complete"]
        log(name, f"p50 {cmp_['eot_total_ms_p50']} p95 {cmp_['eot_total_ms_p95']} FF {r['incomplete']['false_fire_pct']} "
            f"miss {cmp_['missed_pct']} early {cmp_['early_fire_pct']} acc {r['accuracy_pct']} "
            f"(incl. pre-speech {r_raw['accuracy_pct']}); offline differs on {differ} clips", out[name].get("smartturn_calls"))
    save("served_assistant", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("sweep", "smartturn", "served"):
        sub.add_parser(n)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
