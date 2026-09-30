"""Dynamic-timeout policies on eot-bench v2 (scripts/research/bench_completeness.py --stage dyntimeout; research/archive/COMPLETENESS.md §2.4).

Policy: fire at frame t when  silence(t) >= required(t),  required(t) = clamp(T0 - a * s(t), Tmin, Tmax)  (frames),
  silence = the Silero/Pipecat any-speaker silence run ("silero") or our Sortformer-primary silence run ("primary");
  s       = the trail6 head posterior ("head"), smart-turn v3.2 P(complete) held from the matching trigger ("st":
            Silero triggers for the Silero run, Sortformer-primary triggers for the primary run), or both
            (required = T0 - a_head * head - a_st * st).
Because s >= 0, required <= T0, so Tmax < T0 only lowers the wait at s = 0: the grid keeps Tmax = T0 (reported as such)
and the free parameters are (T0, a, Tmin). As a score track: silence - required, fired when > -0.5 (>= 0 frames).
Fitting = the scorer's own operating-point rule (lowest P50, then P90, then miss, subject to <= 5 % per-turn FC) over the
grid on the OTHER meeting fold, applied to this fold: exactly what v2_crossfit does over a threshold axis, here over the
policy grid. Emission = the slowest input's rule. Reference rows (Silero timeout, head-OR-Silero, hybrid) are
recomputed with the same code and asserted equal to the committed numbers.

Continuous evaluation: the fitted policy (per fold) is run over the WHOLE window (lead + turn + trail); an event is a
rising edge of the fire condition (one per silence run at most). Reported per window: events in the lead (before the
primary's onset: other speakers' silences), inside the turn (cutoffs), the first post-end event (the detection), extra
post-end events ("duplicates"), and post-end events after another speaker has started since the end ("late fires").
"""
from __future__ import annotations

import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np

import bench_turn_baselines as BT
import eval_stage1 as E
from audioforge.conversation import bootstrap_ci, eot_outcomes, outcome_metrics
from bench_completeness_fusion import Scorer, emit_fn

T0_GRID = (9, 13, 18, 24, 30, 37, 45, 55, 65, 75)  # 0.72 .. 6.0 s
A_GRID = (0, 5, 10, 15, 20, 30, 40, 55)
TMIN_GRID = (2, 4, 7, 10)
A_BOTH = (0, 10, 20, 35, 50)
POLICIES = {  # name -> (silence stream, s streams, emission kind)
    "silero|head": ("to_silero", ("head",), "head"),
    "silero|st": ("to_silero", ("st_silero",), "frame"),
    "silero|head+st": ("to_silero", ("head", "st_silero"), "head"),
    "primary|head": ("to_primary", ("head",), "head"),
    "primary|st": ("to_primary", ("st_sortformer",), "sf"),
    "primary|head+st": ("to_primary", ("head", "st_sortformer"), "head"),
    # our completeness head (stage streams) in smart-turn's place; the head-only policies with a 160 ms emission
    "silero|cmp": ("to_silero", ("cmp",), "eou"),
    "silero|head+cmp": ("to_silero", ("head", "cmp"), "head"),
}


def required(params, s_tracks):
    T0, tmin, coefs = params
    r = np.full_like(s_tracks[0], float(T0))
    for a, s in zip(coefs, s_tracks):
        r = r - a * s
    return np.clip(r, tmin, T0)


def policy_scores(rows, sil, s_names, params):
    return [np.asarray(r[sil], np.float64) - required(params, [np.asarray(r[n], np.float64) for n in s_names]) for r in rows]


def grid(n_s: int):
    A = A_GRID if n_s == 1 else A_BOTH
    return [(T0, tmin, coefs) for T0 in T0_GRID for tmin in TMIN_GRID if tmin <= T0
            for coefs in itertools.product(A, repeat=n_s)]


def stage_dyntimeout(a, work: Path):
    t0 = time.time()
    out = {"policy": "fire when silence >= clamp(T0 - sum_i a_i s_i, Tmin, Tmax = T0) frames; grid over (T0, Tmin, a_i)",
           "grid": {"T0_frames": T0_GRID, "Tmin_frames": TMIN_GRID, "a_single": A_GRID, "a_both": A_BOTH},
           "selection": "scorer rule (P50, P90, miss s.t. per-turn FC <= 5 %) on the other meeting fold", "blocks": {}}
    ref = json.loads(BT.REF_JSON.read_text())["turn_v2"]
    base_json = json.loads((BT.ROOT / "runs" / "baselines_turn.json").read_text())
    ext_key = None
    for blk, hz in (("C", {"2s": 25, "6s": 75}), ("A", {"2s": None})):
        rows = pickle.loads((work / f"streams_{blk}.pkl").read_bytes())
        if blk == "C":
            ext_key = [r["key"] for r in rows]
        cmp_dir = work / "cmp"
        have_cmp = all((cmp_dir / f"{k}.npy").exists() for k in ext_key)
        if have_cmp:
            for i, r in enumerate(rows):
                r["cmp"] = np.load(cmp_dir / f"{ext_key[i]}.npy")[: r["T"]].astype(np.float64)
        policies = {k: v for k, v in POLICIES.items() if have_cmp or "cmp" not in k}
        out["completeness_stream"] = have_cmp
        sc = Scorer(rows, hz, a.n_boot)
        head = [r["head"].astype(np.float64) for r in rows]
        tsil = [r["to_silero"].astype(np.float64) for r in rows]
        top = [r["to_primary"].astype(np.float64) for r in rows]
        eh, es, ef = E._stream_emit(BT.C_SF, BT.R_SF, 2), E._stream_emit(BT.C_SF, BT.R_SF), BT.EMIT["frame"]
        sc.single("silero_timeout", tsil, ef)
        sc.single("timeout_primary", top, es)
        sc.hybrid("hybrid", head, eh, top, es)
        sc.hybrid("head_or_silero", head, eh, tsil, ef)
        blk_key = "C_extended_windows_stream" if blk == "C" else "A_default_windows_all"
        for mine, theirs, src in (("hybrid", "hybrid_stream_causal_dominant", ref), ("silero_timeout", "silero_timeout", base_json),
                                  ("timeout_primary", "timeout_stream_causal_dominant", ref),
                                  ("head_or_silero", "head_trail6+silero_timeout", base_json)):
            for h in hz:
                x = sc.res["systems"][mine][h]["fixed_5pct_turn_fc"]
                y = src[blk_key]["systems"][theirs][h]["fixed_5pct_turn_fc"]
                assert (x["miss_rate"], x["fc_rate"]) == (y["miss_rate"], y["fc_rate"]), (blk, mine, h)
        fitted = {}
        for name, (sil, s_names, kind) in policies.items():
            emit = emit_fn(kind)
            cands = grid(len(s_names))
            r = {}
            for h, L in hz.items():
                # per-window outcomes of every candidate at the fixed fire rule (score > -0.5)
                FC = np.zeros((len(rows), len(cands)), bool)
                LAT = np.full((len(rows), len(cands)), np.inf)
                PF = np.zeros((len(rows), len(cands)), np.int64)
                for j, prm in enumerate(cands):
                    z = E._emit_transform(policy_scores(rows, sil, s_names, prm), sc.en, emit)
                    oc = eot_outcomes(z, sc.on, sc.en, [-0.5], post_end_frames=L, pauses=sc.pauses)
                    FC[:, j], LAT[:, j], PF[:, j] = oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0]
                    npause = oc["npause"]
                oc_all = {"fc": FC, "lat": LAT, "pf": PF, "npause": npause, "ths": np.arange(len(cands), dtype=np.float64)}
                fc, lat, pf, th = E.v2_crossfit(oc_all, sc.folds, 0.05, "turn", tie_miss=True)
                chosen = {f: {"T0": cands[int(j)][0], "Tmin": cands[int(j)][1], "Tmax": cands[int(j)][0],
                              "a": dict(zip(s_names, cands[int(j)][2]))} for f, j in th.items()}
                pt = outcome_metrics(fc, lat, pf, npause)
                pt.update(bootstrap_ci(fc, lat, a.n_boot, 0), params_by_fold=chosen)
                pt["strata"] = {g: {**outcome_metrics(fc[m], lat[m], pf[m], npause[m]), **bootstrap_ci(fc[m], lat[m], a.n_boot, 0)}
                                for g, m in sc.groups.items()}
                sc.outs[(f"dyn_{name}", h)] = (fc, lat)
                r[h] = {"fixed_5pct_turn_fc": pt}
                fitted[(name, h)] = chosen
            sc.res["systems"][f"dyn_{name}"] = r
            for ref_name in ("head_or_silero", "silero_timeout", "hybrid"):
                sc.pair(f"dyn_{name}", ref_name)
            if name in ("silero|head", "silero|head+st", "silero|cmp"):
                # the head OR the fitted dynamic timeout (the hybrid with the dynamic policy in the timeout's place; the
                # policy's per-fold parameters are those fitted at each horizon, the joint (theta, offset) grid is then
                # cross-fitted on the same folds as every hybrid)
                dyn_tracks = {h: [policy_scores([r], sil, s_names, (fitted[(name, h)][r["fold"]]["T0"], fitted[(name, h)][r["fold"]]["Tmin"],
                                                                     [fitted[(name, h)][r["fold"]]["a"][n] for n in s_names]))[0] for r in rows]
                              for h in hz}
                hn = f"head_or_dyn_{name}"
                rr = {}
                for h, L in hz.items():
                    sub = Scorer(rows, {h: L}, a.n_boot)
                    sub.hybrid(hn, head, eh, dyn_tracks[h], emit)
                    rr[h] = sub.res["systems"][hn][h]
                    sc.outs[(hn, h)] = sub.outs[(hn, h)]
                sc.res["systems"][hn] = rr
                for ref_name in ("head_or_silero", "silero_timeout", "hybrid", f"dyn_{name}"):
                    sc.pair(hn, ref_name)
        sc.pair("head_or_silero", "silero_timeout")
        if blk == "C":
            sc.res["continuous_whole_window"] = continuous(rows, sc, fitted)
        out["blocks"][blk_key] = sc.res
        print(f"  {blk}: {time.time() - t0:.0f}s", flush=True)
    out["sec"] = round(time.time() - t0, 1)
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    prev["dyntimeout"] = out
    Path(a.out).write_text(json.dumps(prev, indent=1, default=float))
    print(f"wrote {a.out}")
    print_dyn(out)


# --------------------------------------------------------------------------- continuous run over whole windows
def events(cond: np.ndarray) -> np.ndarray:
    c = np.asarray(cond, bool)
    return np.nonzero(c & ~np.concatenate([[False], c[:-1]]))[0]


def window_events(r, ev: np.ndarray) -> dict:
    o, e, T = r["onset"], r["end"], r["T"]
    other_since_end = np.zeros(T, bool)
    if e < T:
        other_since_end[e:] = np.maximum.accumulate(r["other_act"][e:])
    post = ev[ev >= e]
    return {"lead": int((ev < o).sum()), "turn": int(((ev >= o) & (ev < e)).sum()), "post": int(len(post)),
            "dup": int(max(0, len(post) - 1)), "late": int(other_since_end[post].sum()) if len(post) else 0,
            "first_late": bool(len(post) and other_since_end[post[0]]), "n": int(len(ev))}


def continuous(rows, sc: Scorer, fitted: dict) -> dict:
    """Fire conditions over the whole window at the 6 s-fitted parameters of each window's fold (policies) and at the
    cross-fitted thresholds of the references."""
    res = {"definition": "event = rising edge of the fire condition over the whole window (frame grid, no emission "
                         "delay); lead = before the primary's onset, turn = inside it (cutoffs), post = after the end "
                         "(first = detection, dup = further ones), late = post-end events after another speaker has "
                         "started since the end", "systems": {}}
    conds = {}
    for name, (sil, s_names, kind) in POLICIES.items():
        if (name, "6s") not in fitted:
            continue
        ch = fitted[(name, "6s")]
        conds[f"dyn_{name}"] = [(np.asarray(r[sil], np.float64) - required(
            (ch[r["fold"]]["T0"], ch[r["fold"]]["Tmin"], [ch[r["fold"]]["a"][n] for n in s_names]),
            [np.asarray(r[n], np.float64) for n in s_names])) >= 0 for r in rows]
    th = sc.res["systems"]["silero_timeout"]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
    conds["silero_timeout"] = [r["to_silero"] > th[r["fold"]] for r in rows]
    th = sc.res["systems"]["head_or_silero"]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
    conds["head_or_silero"] = [((r["head"] > (th[r["fold"]]["theta_a"] if th[r["fold"]]["theta_a"] is not None else np.inf))
                                | (r["to_silero"] > (th[r["fold"]]["theta_b"] if th[r["fold"]]["theta_b"] is not None else np.inf)))
                               for r in rows]
    th = sc.res["systems"]["hybrid"]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
    conds["hybrid"] = [((r["head"] > (th[r["fold"]]["theta_a"] if th[r["fold"]]["theta_a"] is not None else np.inf))
                        | (r["to_primary"] > (th[r["fold"]]["theta_b"] if th[r["fold"]]["theta_b"] is not None else np.inf)))
                       for r in rows]
    res["systems"] = continuous_summary(rows, conds, sc.groups["open"])
    return res


def continuous_summary(rows, conds: dict, open_: np.ndarray) -> dict:
    """{system: {all / open / taken: per-window event statistics}} of fire conditions (one bool track per window) over
    the whole windows; ``rows`` need onset / end / T / other_act (bench_completeness stream rows)."""
    out = {}
    for name, cs in conds.items():
        w = [window_events(r, events(c)) for r, c in zip(rows, cs)]

        def agg(sel):
            ws = [x for x, m in zip(w, sel) if m]
            n = len(ws)
            return {"n_windows": n, "events_per_window": round(sum(x["n"] for x in ws) / n, 3),
                    "lead_events_per_window": round(sum(x["lead"] for x in ws) / n, 3),
                    "windows_with_lead_event": round(sum(x["lead"] > 0 for x in ws) / n, 4),
                    "windows_with_turn_event(cutoff)": round(sum(x["turn"] > 0 for x in ws) / n, 4),
                    "windows_with_post_event": round(sum(x["post"] > 0 for x in ws) / n, 4),
                    "duplicate_post_events_per_window": round(sum(x["dup"] for x in ws) / n, 3),
                    "windows_with_duplicate": round(sum(x["dup"] > 0 for x in ws) / n, 4),
                    "late_post_events_per_window": round(sum(x["late"] for x in ws) / n, 3),
                    "windows_whose_first_post_event_is_late": round(sum(x["first_late"] for x in ws) / n, 4),
                    "late_share_of_post_events": round(sum(x["late"] for x in ws) / max(1, sum(x["post"] for x in ws)), 4)}
        out[name] = {"all": agg(np.ones(len(rows), bool)), "open": agg(open_), "taken": agg(~open_)}
    return out


def print_dyn(out):
    for blk, r in out["blocks"].items():
        print(f"\n### {blk}")
        hzs = [h for h in ("6s", "2s") if h in next(iter(r["systems"].values()))]
        for name, s in r["systems"].items():
            cells = []
            for h in hzs:
                p = s[h]["fixed_5pct_turn_fc"]
                o = p["strata"]["open"]
                prm = p.get("params_by_fold")
                cells.append(f"{h}: {100*p['miss_rate']:.1f} [{100*p['miss_rate_ci'][0]:.1f}, {100*p['miss_rate_ci'][1]:.1f}] all, "
                             f"{100*o['miss_rate']:.1f} [{100*o['miss_rate_ci'][0]:.1f}, {100*o['miss_rate_ci'][1]:.1f}] open, "
                             f"{100*p['strata']['taken']['miss_rate']:.1f} taken, FC {100*p['fc_rate']:.1f}/{100*(p.get('fc_per_pause') or 0):.1f}, "
                             f"P50 open {o['p50_ms']}" + (f", params {prm}" if prm else ""))
            print(f"| {name} | " + " | ".join(cells))
        print("paired:")
        for k, d in r["paired"].items():
            print(f"  {k}: all {100*d['all']['miss_diff']:+.1f} [{100*d['all']['miss_diff_ci'][0]:+.1f}, {100*d['all']['miss_diff_ci'][1]:+.1f}]"
                  f" open {100*d['open']['miss_diff']:+.1f} [{100*d['open']['miss_diff_ci'][0]:+.1f}, {100*d['open']['miss_diff_ci'][1]:+.1f}]"
                  f" fc {100*d['all']['fc_a']:.1f} vs {100*d['all']['fc_b']:.1f}")
        if "continuous_whole_window" in r:
            print(json.dumps(r["continuous_whole_window"]["systems"], indent=1))
