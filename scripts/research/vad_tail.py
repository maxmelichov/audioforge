"""research/VAD_TAIL.md: the served VAD head's tail after the end of speech, and whether it can be removed.

Stages (numpy stages need no gate; every model-loading stage through scripts/dev/gate.sh, resumable, --budget s):
  tail      the tail on the eot_latency dump (calls + AMI), per VAD threshold 0.3..0.7, vs Silero, split by what the
            tail region holds (Silero speech / energy only / nothing), and on the AMI / ICSI dev VAD sets vs their frame
            labels -> runs/vad_tail.json "tail"
  oracle    first look: the shipped rule family on the served VAD and on a tail-free VAD (Silero on the frame grid)
  bounds    what removing the tail could buy: pause vs end tail, backdated silence clock, tail-free oracle + hangover,
            tail-free VAD on the head path only -> "bounds"
  fast_scan the --turn-preset fast selection -> "fast_scan"
  presets   balanced / fast on both label sets with a session bootstrap -> "presets"
  b4feats   (gate.sh, MPS) block-4 feature cache for a VAD-head retrain (stopped: the oracle bound answers it)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/vad_tail")
OUT = ROOT / "runs" / "vad_tail.json"
LABELS_AE = ROOT / "runs" / "turn_v4_labels_eval.json"
FRAME, CHUNK_S, SR = 0.08, 512 / 16000, 16000
THRS = (0.3, 0.4, 0.5, 0.6, 0.7)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def pct(x, qs=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    return {f"p{q}": int(round(1000 * float(np.percentile(x, q)))) for q in qs} if len(x) else None


def run_end(on: np.ndarray, step: float, e: float, stop: float) -> float:
    """End (s) of the detector's speech around a reference end e: if the step containing e - eps is on, follow the on
    run forward (not past ``stop``); else the end of the last on step before e. -inf if none."""
    n = len(on)
    j = int(np.floor((e - 1e-6) / step))
    j = min(max(j, 0), n - 1)
    if on[j]:
        while j + 1 < n and on[j + 1] and (j + 1) * step < stop:
            j += 1
        return (j + 1) * step
    k = np.nonzero(on[: j + 1])[0]
    return (k[-1] + 1) * step if len(k) else -np.inf


def energy_db(x: np.ndarray, hop=160) -> np.ndarray:
    n = len(x) // hop
    f = x[: n * hop].reshape(n, hop)
    return 10 * np.log10((f.astype(np.float64) ** 2).mean(1) + 1e-12)


def cmd_tail(a):
    import eot_latency as E
    dump = E.load_dump()
    ae = json.loads(LABELS_AE.read_text())
    rows = []
    for ss in E.sessions():
        d = dump.get(ss["key"])
        if d is None:
            continue
        vad = np.zeros(max(d["head"]["v"]) + 1)
        vad[np.asarray(d["head"]["v"])] = d["head"]["vad"]
        conf = np.asarray(d["conf"], float)
        x = E.read_audio(ss)
        db = energy_db(x)
        floor = float(np.percentile(db, 10))
        lvl = float(np.percentile(db, 95))
        act = db > max(floor + 12.0, lvl - 40.0)  # 10 ms frames above the noise floor
        for i, (s0, e1) in enumerate(ss["user_turns"]):
            if not ss["scored"][i]:
                continue
            nxt = ss["next_onset"][i]
            stop = min(e1 + 3.0, nxt if nxt is not None else 1e9, len(vad) * FRAME)
            ea = ae[ss["key"]][i][1]
            r = {"key": ss["key"], "set": ss["set"], "e": e1, "ea": ea, "stop": stop}
            for thr in THRS:
                r[f"vad{thr}"] = run_end(vad > thr, FRAME, e1, stop)
                r[f"sil{thr}"] = run_end(conf > thr, CHUNK_S, e1, stop)
            r["en"] = run_end(act, 0.01, e1, stop)
            # what the tail region (label end .. the VAD's fall at 0.5) holds
            v_end = r["vad0.5"]
            if v_end > e1 + 0.04:
                j0, j1 = int(np.ceil((e1 + 0.04) / CHUNK_S)), int(np.floor(v_end / CHUNK_S))
                k0, k1 = int(np.ceil((e1 + 0.04) / 0.01)), int(np.floor(v_end / 0.01))
                sil_sp = bool(j1 > j0 and (conf[j0:j1] > 0.5).any())
                en_frac = float(act[k0:k1].mean()) if k1 > k0 else 0.0
                r["tail_kind"] = "silero_speech" if sil_sp else ("energy" if en_frac >= 0.3 else "quiet")
                r["tail_en_frac"] = round(en_frac, 3)
                r["tail_db_over_floor"] = round(float(np.median(db[k0:k1]) - floor), 1) if k1 > k0 else None
            else:
                r["tail_kind"] = "none"
            rows.append(r)
    (WORK / "tail_rows.json").write_text(json.dumps(rows, default=float))
    res = {"n_ends": {}, "definition": "tail = end of the detector's on-run that contains the reference end (followed "
           "forward; if the detector is already off at the end, the end of its last on step before it, negative) - "
           "the reference end. Served VAD on the 80 ms frame grid [0.08 v, 0.08 (v + 1)) (the label convention; "
           "frame v's pass-1 input ends at 0.08 v + 16 ms for odd v and 0.08 (v + 1) + 16 ms for even v), Silero on its "
           "32 ms chunks, energy on 10 ms frames above max(floor + 12 dB, p95 - 40 dB)."}
    for scope, sets in (("calls", ("turnbench", "oto")), ("turnbench", ("turnbench",)), ("oto", ("oto",)),
                        ("ami", ("ami",))):
        R = [r for r in rows if r["set"] in sets]
        res["n_ends"][scope] = len(R)
        out = {}
        for ref in ("e", "ea"):
            for thr in THRS:
                out[f"vad{thr}_vs_{ref}"] = pct([r[f"vad{thr}"] - r[ref] for r in R if np.isfinite(r[f"vad{thr}"])])
                out[f"silero{thr}_vs_{ref}"] = pct([r[f"sil{thr}"] - r[ref] for r in R if np.isfinite(r[f"sil{thr}"])])
            out[f"energy_vs_{ref}"] = pct([r["en"] - r[ref] for r in R if np.isfinite(r["en"])])
        out["vad0.5_vs_silero0.5"] = pct([r["vad0.5"] - r["sil0.5"] for r in R if np.isfinite(r["vad0.5"]) and np.isfinite(r["sil0.5"])])
        out["vad0.4_vs_silero0.5"] = pct([r["vad0.4"] - r["sil0.5"] for r in R if np.isfinite(r["vad0.4"]) and np.isfinite(r["sil0.5"])])
        out["vad0.5_vs_energy"] = pct([r["vad0.5"] - r["en"] for r in R if np.isfinite(r["vad0.5"]) and np.isfinite(r["en"])])
        kinds = {}
        for r in R:
            kinds.setdefault(r["tail_kind"], []).append(r["vad0.5"] - r["e"])
        out["tail_kind"] = {k: {"n": len(v), "tail_ms": pct(v, (50, 90))} for k, v in sorted(kinds.items())}
        res[scope] = out
    # frame-labelled VAD sets (research/VAD_SINGLE.md; the served block-4 head's saved scores): tail at label ends
    # followed by >= 400 ms of label silence
    lab = {}
    for name in ("ami_dev", "icsi_dev"):
        f = Path("/Volumes/ExternalSSD/nvidia-audio-models/runs/vad_single") / f"eval_{name}.npz"
        z = np.load(f, allow_pickle=True)
        tails = {thr: [] for thr in THRS}
        for y, s in zip(z["labels"], z["L3"]):
            y, s = np.asarray(y, bool), np.asarray(s, float)
            for v in range(1, len(y) - 6):
                if y[v - 1] and not y[v] and not y[v: v + 5].any():
                    for thr in THRS:
                        e_ = run_end(s > thr, FRAME, v * FRAME, (v + 12) * FRAME)
                        if np.isfinite(e_):
                            tails[thr].append(e_ - v * FRAME)
        lab[name] = {f"vad{thr}_vs_label_end": pct(t) for thr, t in tails.items()}
        lab[name]["n_ends"] = len(tails[0.5])
        lab[name]["share_tail_ge_160ms@0.5"] = round(float(np.mean(np.asarray(tails[0.5]) >= 0.16 - 1e-6)), 3)
    res["frame_label_sets"] = lab
    save("tail", res)
    print(json.dumps(res, indent=1))




# --------------------------------------------------------------------------- rule scoring on a (modified) dump
GOAL = {"calls_fi": 20.2, "calls_miss": 7.3, "ami_fi": 10.5, "ami_miss": 33.5}


def load_ctx(dump_dir=None, labels=None):
    """(dump, sessions, compute) with EOT_DUMP / EOT_LABELS applied (eot_latency reads them at import)."""
    import importlib
    if dump_dir is not None:
        os.environ["EOT_DUMP"] = str(dump_dir)
    if labels is not None:
        os.environ["EOT_LABELS"] = str(labels)
    elif "EOT_LABELS" in os.environ:
        del os.environ["EOT_LABELS"]
    import eot_latency as E
    E = importlib.reload(E)
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump]
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    return E, dump, sess, comp


def score_rule(E, dump, sess, comp, rule) -> dict:
    per = {sc: [] for sc in E.SCOPES}
    for ss in sess:
        d = dump[ss["key"]]
        t = [x + comp[ss["key"]] for x, _ in E.simulate(d, rule)]
        for sc, sets in E.SCOPES.items():
            if ss["set"] in sets:
                per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
    return {"calls": E.pool(per["two_party_user"]), "ami": E.pool(per["ami"])}


def vad_grid(E, vts=(0.3, 0.4, 0.5, 0.6), ks=(1, 2, 3, 4, 5, 6), ths=(0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 2.0),
             Fs=(3, 4, 5, 6, 7, 8, 9, 10), fus=(8, 12)):
    import itertools
    R = []
    for vt, k, th, F, fu in itertools.product(vts, ks, ths, Fs, fus):
        if th > 1 and k != 1:
            continue
        R.append({**E.CHOSEN, "vad_thr": vt, "k": k, "th": th, "fallback_f": F, "fu": fu})
    return R


def meets(x, g):
    c, m = x["calls"], x["ami"]
    return (c["eot_total_ms_p50"] is not None and c["false_interruption_pct"] <= g["calls_fi"] and c["missed_pct"] <= g["calls_miss"]
            and m["false_interruption_pct"] <= g["ami_fi"] and m["missed_pct"] <= g["ami_miss"])


def fmt(x):
    c, m = x["calls"], x["ami"]
    return (f"calls {c['eot_total_ms_p50']}/{c['eot_total_ms_p95']} {c['false_interruption_pct']}/{c['missed_pct']} | "
            f"AMI {m['eot_total_ms_p50']}/{m['eot_total_ms_p95']} {m['false_interruption_pct']}/{m['missed_pct']}")


def rname(r):
    s = f"VAD<{r['vad_thr']}"
    if r.get("vad_backdate"):
        s += f" backdate{r['vad_backdate']}"
    if r["th"] <= 1:
        s += f" sil>={r['k'] * 80}&p>={r['th']}"
    return s + f" | fb {r['fallback_f'] * 80} | others {r['fu'] * 80}"


def scan(E, dump, sess, comp, rules, goal=GOAL, clip_frames=None, top=8):
    rows = [{"rule": r, "name": rname(r), **score_rule(E, dump, sess, comp, r)} for r in rules]
    ok = sorted([x for x in rows if meets(x, goal)],
                key=lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"], x["ami"]["eot_total_ms_p50"]))
    best = None
    if clip_frames is not None:
        meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
        for x in ok:
            x["clip_cut_in"] = [v for v, dd in clip_frames.items() if E.clip_cuts(dict(dd), x["rule"], meta)["cuts_user_turn"]]
            if not x["clip_cut_in"]:
                best = x
                break
    # Pareto: calls p50 vs calls FI with calls missed and both AMI numbers within the goal
    fr = sorted([x for x in rows if x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["missed_pct"] <= goal["calls_miss"]
                 and x["ami"]["missed_pct"] <= goal["ami_miss"] and x["ami"]["false_interruption_pct"] <= goal["ami_fi"]],
                key=lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"]))
    par, bfi = [], 1e9
    for x in fr:
        if x["calls"]["false_interruption_pct"] < bfi:
            bfi = x["calls"]["false_interruption_pct"]
            par.append(x)
    return {"n_rules": len(rows), "n_meet_goal": len(ok), "fastest_meeting_goal": ok[:top], "best_no_clip_cut": best,
            "pareto": par}


def silero_frames(d) -> np.ndarray:
    """Oracle frame VAD from the Silero track: max Silero confidence over the 32 ms chunks overlapping frame v's label
    window [0.08 v, 0.08 (v + 1))."""
    conf = np.asarray(d["conf"], float)
    v = np.asarray(d["head"]["v"])
    out = np.zeros(len(v))
    for i, vv in enumerate(v):
        j0, j1 = int(np.floor(vv * FRAME / CHUNK_S)), int(np.ceil((vv + 1) * FRAME / CHUNK_S))
        out[i] = conf[j0:j1].max(initial=0.0)
    return out


def cmd_oracle(a):
    """Upper bound before any retraining: the shipped rule family on a tail-free VAD (the Silero track on the frame
    grid, an oracle) and on the served VAD, scored with the harness's simulator; plus the tail at mid-turn pauses."""
    t0 = time.time()
    E, dump, sess, comp = load_ctx()
    cv = json.loads((E.WORK / "clip_frames.json").read_text())
    res = {}
    R = vad_grid(E)
    log(f"{len(R)} rules")
    res["served_vad"] = scan(E, dump, sess, comp, R)
    log("served", time.time() - t0)
    for d in dump.values():
        d["head"]["vad"] = [float(x) for x in silero_frames(d)]
        d.pop("_cache", None)
    res["silero_oracle_vad"] = scan(E, dump, sess, comp, R)
    log("oracle", time.time() - t0)
    for k, v in res.items():
        log(k, v["n_meet_goal"], "meet goal")
        for x in v["fastest_meeting_goal"][:3]:
            log("   ", fmt(x), x["name"])
        for x in v["pareto"][:12]:
            log("   pareto", fmt(x), x["name"])
    save("oracle", res)



# --------------------------------------------------------------------------- block-4 feature cache (pass 1, [70,1])
B4 = WORK / "b4"
N_OTO = 800


def _frames_from(ivs, T):
    y = np.zeros(T, np.float32)
    for a0, b0 in ivs:  # the label convention: [s, e) covers frames int(s / 0.08) .. ceil(e / 0.08) - 1
        y[int(a0 / FRAME): max(int(a0 / FRAME), int(np.ceil(b0 / FRAME - 1e-9)))] = 1.0
    return y


def cmd_b4feats(a):
    """Block-4 features (the served VAD head's input) of: AMI train diar windows (any-speaker word labels), oto user
    channels (their Silero activity), AMI / ICSI dev VAD sets, the 232 eot sessions and the six clip deliveries
    -> B4/<set>/<id>.npz {b4 float16 (T, 512), y (T,) or none}. Resumable, --budget s."""
    import torch
    import eot_latency as E
    from audioforge.train import load_model
    torch.set_num_threads(2)
    t0 = time.time()
    m = load_model(str(E.SERVED_AFM), a.device).eval()

    def enc(xs):
        with torch.no_grad():
            x, xl = m._pad(list(xs))
            e, el, hid = m.encode(x, xl, [70, 1], return_hidden=True)
            b4 = m.head_input("vad", e, hid)
            v = m.heads["vad"](b4).sigmoid()
            return [(b4[i, : int(el[i])].float().cpu().numpy().astype(np.float16), v[i, : int(el[i])].float().cpu().numpy())
                    for i in range(len(xs))]

    def todo_items():
        if a.set == "eval":
            dump = E.load_dump()
            for s in E.sessions():
                if s["key"] in dump:
                    yield s["key"], (lambda s=s: E.read_audio(s)), None
            from audioforge.data import load_wav
            x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
            for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items()):
                yield f"clip_{i}", (lambda fn=fn: fn(x0)), None
        elif a.set in ("ami_train", "ami_dev", "icsi_dev"):
            from vad_layers import load_set
            val = load_set({"ami_train": "train", "ami_dev": "ami_dev", "icsi_dev": "icsi_dev"}[a.set], n_train=a.n)
            for j, v in enumerate(val):
                yield f"{j:05d}", (lambda v=v: np.asarray(v["audio"], np.float32)), np.asarray(v["vad"], np.float32)
        elif a.set == "oto":
            import turn_v4 as T4
            man = json.loads((T4.WORK / "manifest.json").read_text())
            cs = [c for c in man if c["src"] == "oto"]
            cs = cs[:: max(1, len(cs) // N_OTO)][:N_OTO]
            au = T4.ClipAudio()
            for c in cs:
                yield c["id"], (lambda c=c: au(c, man)[0]), ("oto", c["act"])
    od = B4 / a.set
    od.mkdir(parents=True, exist_ok=True)
    batch, n = [], 0

    def flush():
        nonlocal batch, n
        outs = enc([x for _, x, _ in batch])
        for (k, x, y), (b4, v) in zip(batch, outs):
            T = len(b4)
            if isinstance(y, tuple):
                y = _frames_from(y[1], T)
            elif y is not None:
                y = np.asarray(y[:T], np.float32)
                if len(y) < T:
                    y = np.concatenate([y, np.zeros(T - len(y), np.float32)])
            d = {"b4": b4, "vad": v.astype(np.float32)}
            if y is not None:
                d["y"] = y
            np.savez(od / f"{k}.tmp.npz", **d)
            (od / f"{k}.tmp.npz").rename(od / f"{k}.npz")
            n += 1
        batch = []
    for k, fx, y in todo_items():
        if (od / f"{k}.npz").exists():
            continue
        if time.time() - t0 > a.budget:
            break
        batch.append((k, fx(), y))
        if len(batch) >= (1 if a.set == "eval" else a.batch):
            flush()
    if batch:
        flush()
    log(f"b4feats {a.set}: {n} new in {time.time() - t0:.0f} s; total {len(list(od.glob('*.npz')))}")


def cmd_presets(a):
    """--turn-preset balanced / fast (audioforge.server.constants.TURN_PRESETS) scored by the harness on both label
    sets, with a session-bootstrap 90 % interval of the calls p50 / p95 difference -> runs/vad_tail.json "presets"."""
    res = {}
    for lab_name, lab in (("original", None), ("audible_end", LABELS_AE)):
        E, dump, sess, comp = load_ctx(labels=lab)
        rows = {}
        per = {}
        for name, rule in (("balanced", E.CHOSEN), ("fast", E.FAST)):
            rows[name] = score_rule(E, dump, sess, comp, rule)
            per[name] = {ss["key"]: E.score_session([x + comp[ss["key"]] for x, _ in E.simulate(dump[ss["key"]], rule)],
                                                     ss, comp[ss["key"]]) for ss in sess}
        calls = [ss["key"] for ss in sess if ss["set"] != "ami"]
        rng = np.random.default_rng(0)
        d50, d95 = [], []
        for _ in range(2000):
            ks = rng.choice(calls, len(calls))
            lb = np.concatenate([per["balanced"][k]["lat"] for k in ks])
            lf = np.concatenate([per["fast"][k]["lat"] for k in ks])
            d50.append(np.median(lf) - np.median(lb))
            d95.append(np.percentile(lf, 95) - np.percentile(lb, 95))
        rows["calls_fast_minus_balanced_ms_90ci"] = {
            "p50": [int(round(1000 * np.percentile(d50, q))) for q in (5, 95)],
            "p95": [int(round(1000 * np.percentile(d95, q))) for q in (5, 95)]}
        res[lab_name] = rows
        for name in ("balanced", "fast"):
            log(lab_name, name, fmt(rows[name]))
        log(lab_name, "fast - balanced 90 % CI", rows["calls_fast_minus_balanced_ms_90ci"])
    res["rules"] = {"balanced": "VAD < 0.4 for >= 160 ms AND p >= 0.99, OR 640 ms; others path 960,640",
                    "fast": "VAD < 0.6 for >= 480 ms AND p >= 0.99, OR 720 ms; others path 640,640"}
    save("presets", res)


def cmd_bounds(a):
    """What removing the tail could buy, before any retraining (research/VAD_TAIL.md section 2), on the calls + AMI
    dump with the harness's scorer:
      pause_tail   the VAD0.4 fall after Silero's end at in-turn pauses >= 160 ms vs at turn ends (calls)
      backdate     option (a): the silence clock timed from the VAD's fall (last frame >= vad_hi) once VAD < vad_thr
      oracle       a tail-free VAD (Silero on the frame grid) + 0-320 ms hangover in place of the served VAD
      head_only    the tail-free VAD for the head path only, the served VAD for the fallback
    Each: the rules meeting GOAL and the fastest calls p50 at calls FI <= 20.2 % / missed <= 7.3 %."""
    import itertools
    E, dump, sess, comp = load_ctx()
    res = {}
    # ---- pause tail vs end tail
    ends, pauses, bridged = [], [], []
    for ss in sess:
        if ss["set"] == "ami":
            continue
        d = dump[ss["key"]]
        vad = np.zeros(max(d["head"]["v"]) + 1)
        vad[np.asarray(d["head"]["v"])] = d["head"]["vad"]
        on = np.asarray(d["conf"]) > 0.5
        for i, (s0, e1) in enumerate(ss["user_turns"]):
            j0, j1 = int(s0 / CHUNK_S), int(e1 / CHUNK_S)
            k = j0
            while k < j1:
                if on[k] and k + 1 < j1 and not on[k + 1]:
                    g = k + 1
                    while g < j1 and not on[g]:
                        g += 1
                    if (g - k - 1) * CHUNK_S >= 0.16 and g < j1:
                        pe, ge = (k + 1) * CHUNK_S, g * CHUNK_S
                        f = run_end(vad > 0.4, FRAME, pe, ge + 1.0)
                        pauses.append(min(f, ge) - pe)
                        bridged.append(bool(f >= ge - 1e-6))
                    k = g
                else:
                    k += 1
            if ss["scored"][i]:
                se = run_end(on, CHUNK_S, e1, e1 + 3)
                if np.isfinite(se):
                    ends.append(run_end(vad > 0.4, FRAME, se, e1 + 3) - se)
    p_, b_ = np.asarray(pauses), np.asarray(bridged)
    res["pause_tail"] = {"ends_vad0.4_after_silero_end": pct(ends, (25, 50, 75)), "n_ends": len(ends),
                         "n_pauses_ge_160ms": len(p_), "share_pauses_bridged_by_vad0.4": round(float(b_.mean()), 3),
                         "unbridged_pauses_vad0.4_after_silero_start": pct(p_[~b_], (25, 50, 75))}
    log("pause_tail", res["pause_tail"])

    def summ(rows):
        ok = sorted([x for x in rows if meets(x, GOAL)], key=lambda x: x["calls"]["eot_total_ms_p50"])
        fr = [x for x in calls_front(rows) if x["calls"]["false_interruption_pct"] <= GOAL["calls_fi"]]
        return {"n_rules": len(rows), "n_meet_goal": len(ok), "fastest_meeting_goal": ok[:3],
                "fastest_calls_fi_le_20.2_miss_le_7.3": fr[:1]}
    orig = E.simulate
    E.simulate = lambda d, r: (sim_backdate(d, r) if "vad_hi_bd" in r else sim_two(d, r) if "vad2_thr" in r
                               else orig(d, r))
    # ---- (a) backdated silence clock
    rows = []
    for lo, hi, k, th, F in itertools.product((0.3, 0.4, 0.5), (0.5, 0.6, 0.7, 0.8, 0.9), range(1, 7),
                                              (0.95, 0.98, 0.99, 0.995, 2.0), range(4, 13)):
        if hi < lo or (th > 1 and k != 1):
            continue
        r = {**E.CHOSEN, "vad_thr": lo, "vad_hi_bd": hi, "k": k, "th": th, "fallback_f": F}
        rows.append({"rule": r, "name": f"VAD<{lo} timed from last>={hi} sil>={k * 80}&p>={th} | fb {F * 80}",
                     **score_rule(E, dump, sess, comp, r)})
    res["backdate"] = summ(rows)
    log("backdate", res["backdate"]["n_meet_goal"], [fmt(x) + " " + x["name"] for x in res["backdate"]["fastest_calls_fi_le_20.2_miss_le_7.3"]])
    # ---- head path on the tail-free VAD, fallback on the served VAD
    sf = {k: silero_frames(d) for k, d in dump.items()}
    for k, d in dump.items():
        d["head"]["vad2"] = [float(q) for q in sf[k]]
    rows = []
    for v2, k, th, F in itertools.product((0.3, 0.5, 0.7), range(1, 7), (0.9, 0.95, 0.98, 0.99, 0.995), range(7, 11)):
        r = {**E.CHOSEN, "vad2_thr": v2, "k": k, "th": th, "fallback_f": F}
        rows.append({"rule": r, "name": f"head path on Silero<{v2} {k * 80}ms & p>={th} | fallback served {F * 80}",
                     **score_rule(E, dump, sess, comp, r)})
    res["head_only"] = summ(rows)
    log("head_only", res["head_only"]["n_meet_goal"])
    # ---- oracle tail-free VAD + hangover
    R = vad_grid(E, vts=(0.3, 0.4, 0.5, 0.6), ks=(1, 2, 3, 4, 5, 6), ths=(0.9, 0.95, 0.98, 0.99, 0.995, 2.0),
                 Fs=(4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16), fus=(12,))
    res["oracle"] = {}
    for h in (0, 1, 2, 3, 4):
        for k, d in dump.items():
            s_ = sf[k]
            d["head"]["vad"] = [float(s_[max(0, i - h): i + 1].max()) for i in range(len(s_))]
            d.pop("_cache", None)
        rows = [{"rule": r, "name": f"tail-free VAD + {h * 80} ms hangover: " + rname(r), **score_rule(E, dump, sess, comp, r)}
                for r in R]
        res["oracle"][f"hangover_{h * 80}ms"] = summ(rows)
        log("oracle", h, res["oracle"][f"hangover_{h * 80}ms"]["n_meet_goal"],
            [fmt(x) for x in res["oracle"][f"hangover_{h * 80}ms"]["fastest_calls_fi_le_20.2_miss_le_7.3"]])
    save("bounds", res)




def calls_front(rows, miss=7.3):
    fr = sorted([x for x in rows if x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["missed_pct"] <= miss],
                key=lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"]))
    par, bfi = [], 1e9
    for x in fr:
        if x["calls"]["false_interruption_pct"] < bfi:
            bfi = x["calls"]["false_interruption_pct"]
            par.append(x)
    return par


def sim_two(d: dict, rule: dict) -> list:
    """sim_room's head path / fallback / others path with the head path's silence on a second VAD track
    (d["head"]["vad2"], threshold rule["vad2_thr"]) and the fallback's on the served one (research/VAD_TAIL.md: a
    tail-free VAD for the head path only). One firing per turn: a firing disarms until either track hears speech."""
    ht, hp = np.asarray(d["head"]["t"], float), np.asarray(d["head"]["p"], float)
    sp1 = np.asarray(d["head"]["vad"], float) >= rule["vad_thr"]
    sp2 = np.asarray(d["head"]["vad2"], float) >= rule["vad2_thr"]
    pu, po = np.asarray(d["head"]["pu"], float), np.asarray(d["head"]["po"], float)
    usp = pu >= rule.get("ut", 0.5)
    k, th, F, FU, om, ot = rule["k"], rule["th"], rule["fallback_f"], rule["fu"], rule["om"], rule["ot"]
    out, l1, l2, lu, armed, armu, orun = [], -1, -1, -1, False, False, 0
    for v in range(len(hp)):
        if sp1[v]:
            l1, armed = v, True
        if sp2[v]:
            l2 = v
            armed = armed or l1 >= 0
        if usp[v]:
            lu, armu = v, True
        path = None
        if armed and l1 >= 0:
            s2 = v - max(l2, l1 if rule.get("both") else -1) if l2 >= 0 else 0
            if not sp1[v] or not rule.get("both"):
                if s2 >= k and hp[v] >= th and not sp2[v]:
                    path = "head"
            if path is None and not sp1[v] and v - l1 >= F:
                path = "fallback"
        orun = orun + 1 if po[v] >= ot else 0
        if path is None and armu and orun >= om and FU <= v - lu <= FU:
            path = "others_fallback"
        if path is not None:
            armed = armu = False
            out.append((round(float(ht[v]), 4), path))
    return out


def sim_backdate(d: dict, rule: dict) -> list:
    """sim_room (src "vad", head path + fallback + others fallback) with the silence clock backdated to the VAD's
    fall: once the VAD is below ``vad_thr`` the silence counts from the last frame with VAD >= ``vad_hi`` (>= vad_thr),
    not from the last frame >= vad_thr (research/VAD_TAIL.md option (a); causal). vad_hi == vad_thr is sim_room."""
    ht, hp = np.asarray(d["head"]["t"], float), np.asarray(d["head"]["p"], float)
    vad = np.asarray(d["head"]["vad"], float)
    pu, po = np.asarray(d["head"]["pu"], float), np.asarray(d["head"]["po"], float)
    lo, hi = rule["vad_thr"], rule["vad_hi_bd"]
    k, th, F, FU, om, ot = rule["k"], rule["th"], rule["fallback_f"], rule["fu"], rule["om"], rule["ot"]
    out, last_lo, last_hi, lu, orun = [], -1, -1, -1, 0
    fired, firedu = -2, -2
    for v in range(len(hp)):
        if vad[v] >= lo:
            last_lo = v
        if vad[v] >= hi:
            last_hi = v
        if pu[v] >= 0.5:
            lu = v
        path = None
        if last_lo >= 0 and last_lo != fired and vad[v] < lo:
            sil = v - (last_hi if last_hi >= 0 else last_lo)
            if sil >= k and hp[v] >= th:
                path = "head"
            elif sil >= F:
                path = "fallback"
        orun = orun + 1 if po[v] >= ot else 0
        if path is None and lu >= 0 and lu != firedu and orun >= om and FU <= v - lu <= FU + rule.get("fw", 0):
            path = "others_fallback"
        if path is not None:
            fired, firedu = last_lo, lu
            out.append((round(float(ht[v]), 4), path))
    return out


def cmd_fast_scan(a):
    """The --turn-preset fast selection: the fastest calls p50 among vad_head-family rules (sim_room) with calls FI <=
    30 %, calls missed <= 7.3 %, AMI FI <= 16 % that do not cut the bundled clip under any of the six deliveries; the
    coordinator then picked the 886 ms runner-up (fewer false interruptions and misses, 16 ms slower) as TURN_PRESETS
    ["fast"] -> runs/vad_tail.json "fast_scan"."""
    E, dump, sess, comp = load_ctx()
    R = vad_grid(E, vts=(0.3, 0.35, 0.4, 0.45, 0.5, 0.6), ks=(1, 2, 3, 4, 5, 6),
                 ths=(0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 2.0), Fs=(3, 4, 5, 6, 7, 8, 9, 10, 12), fus=(8, 10, 12))
    rows = [{"rule": r, "name": rname(r), **score_rule(E, dump, sess, comp, r)} for r in R]
    g = {"calls_fi": 30.0, "calls_miss": 7.3, "ami_fi": 16.0, "ami_miss": 100.0}
    ok = sorted([x for x in rows if meets(x, g)], key=lambda x: (x["calls"]["eot_total_ms_p50"],
                                                              x["calls"]["false_interruption_pct"], x["ami"]["missed_pct"]))
    cv = json.loads((E.WORK / "clip_frames.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    fastest_any, no_cut = ok[:5], []
    for x in ok:
        x["clip_cut_in"] = [v for v, dd in cv.items() if E.clip_cuts(dict(dd), x["rule"], meta)["cuts_user_turn"]]
        if not x["clip_cut_in"]:
            no_cut.append(x)
            if len(no_cut) >= 12:
                break
    for x in fastest_any[:3]:
        log("fastest (cuts in", len(x.get("clip_cut_in", [])), "deliveries):", fmt(x), x["name"])
    for x in no_cut:
        log("no cut:", fmt(x), x["name"])
    save("fast_scan", {"goal": g, "n_rules": len(rows), "n_meet": len(ok), "fastest_meeting": fastest_any,
                       "fastest_no_clip_cut": no_cut, "shipped_fast": rname(E.FAST)})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tail")
    sub.add_parser("oracle")
    sub.add_parser("presets")
    sub.add_parser("bounds")
    sub.add_parser("fast_scan")
    b = sub.add_parser("b4feats")
    b.add_argument("--set", required=True)
    b.add_argument("--n", type=int, default=600)
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--budget", type=float, default=540)
    b.add_argument("--device", default="mps")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
