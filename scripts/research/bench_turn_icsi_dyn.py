"""Held-out ICSI confirmation of the dynamic-timeout policies (research/archive/COMPLETENESS.md §2.4; research/archive/BASELINES.md,
"Dynamic timeout: ICSI held-out confirmation").

COMPLETENESS.md §2.4 picked, after seeing the AMI dev table, the dynamic Silero timeout modulated by the trail6 head
posterior,  fire when silence_silero(t) >= clamp(T0 - a * p_head(t), Tmin, Tmax = T0) frames,  and the head OR that
policy (the dynamic rule in the timeout's place inside the OR). This scores them on the 1312 held-out ICSI turns of
scripts/research/bench_turn_icsi.py exactly as head-OR-Silero was confirmed there:

  frozen    parameters FITTED ON ALL 974 AMI DEV TURNS (the in-sample <= 5 % per-turn FC point of the same grid, the
            scorer's own rule: lowest P50, then P90, then miss) applied to ICSI unchanged, plus the AMI table's two
            per-fold points (runs/completeness.json, re-derived here and asserted equal) as a sensitivity range;
            references (Silero timeout, hybrid, head OR Silero, head) at their frozen points from bench_turn_icsi
            (asserted equal to runs/baselines_turn_icsi.json); paired bootstraps on the same resamples
  crossfit  the AMI protocol re-run on ICSI (grid point chosen on one leave-meetings-out half, applied to the other)
  continuous  bench_completeness_dyn's whole-window event statistics (events / window, lead fires, cutoffs, duplicate
            and late post-end fires) at the frozen and the cross-fitted parameters (block C)

The smart-turn variant (Silero | head + st) needs smart-turn v3.2 scores on the ICSI windows; they were never computed
(no <work>/smartturn), so it is skipped and recorded as such.

Stages (CPU, 2 threads; the tracks / vad / scores stages of bench_turn_icsi.py must have run):
  extract  stream rows (head, Silero silence run, Sortformer-primary silence run, labels) of the AMI dev and ICSI
           blocks -> <work>/dyn_rows_{ami,icsi}_{C,A}.pkl (bench_completeness's row format)
  report   -> runs/baselines_turn_icsi_dyn.json

  W=<scratch>/icsi_turn/work
  .venv/bin/python scripts/research/bench_turn_icsi_dyn.py --stage extract --work $W
  .venv/bin/python scripts/research/bench_turn_icsi_dyn.py --stage report --work $W --out runs/baselines_turn_icsi_dyn.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_completeness_dyn as D  # noqa: E402
import bench_turn_icsi as M  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.conversation import eot_outcomes, pause_runs  # noqa: E402
from bench_completeness_fusion import Scorer, emit_fn  # noqa: E402

BT = M.BT
MAX_FC = M.MAX_FC
BLOCKS = {"C": ("C_extended_windows_stream", {"2s": 25, "6s": 75}), "A": ("A_default_windows_all", {"2s": None})}
POLICIES = {k: v for k, v in D.POLICIES.items() if k in ("silero|head", "silero|head+st")}
REFS = ("silero_timeout", "hybrid", "head_or_silero", "head")
REF_NAMES = {  # our name -> bench_turn_icsi's
    "silero_timeout": "silero_timeout", "hybrid": M.REF, "head_or_silero": M.CANDIDATE,
    "head": "head_v3_stream_causal_dominant", "timeout_primary": "timeout_stream_causal_dominant"}
FIRE = -0.5  # a policy score track (silence - required) fires when > -0.5, i.e. silence >= required


# --------------------------------------------------------------------------- rows
def rows_from_block(convs, meta, singles, folds) -> list[dict]:
    """bench_completeness's stream rows (without the smart-turn streams) from bench_turn_icsi.systems_for output."""
    from audioforge.conversation import floor_stratum
    head = singles["head_v3_stream_causal_dominant"][0]
    to = singles["timeout_stream_causal_dominant"][0]
    sil = singles["silero_timeout"][0]
    rows = []
    for i, v in enumerate(convs):
        T = len(v["spk_act"])
        assert len(head[i]) == T and len(sil[i]) == T, (i, len(head[i]), len(sil[i]), T)
        rows.append(dict(
            key=f"{v['meeting']}_{int(round(float(v['start']) * 1000)):08d}", meeting=v["meeting"], T=T, onset=int(v["onset_frame"]), end=int(v["turn_end_frame"]),
            stratum=floor_stratum(v["spk_targets"], int(v["turn_end_frame"]), E.V2_FLOOR_HORIZON),
            fold=0 if v["meeting"] in folds[0] else 1, pauses=pause_runs(v["hes"]),
            post_avail=int(meta[i]["post_avail"]), end_reason=meta[i]["end_reason"],
            other_act=(np.asarray(v["spk_targets"])[:, 1:] > 0.5).any(1),
            head=np.asarray(head[i], np.float64), to_primary=np.asarray(to[i], np.float64)[:T],
            to_silero=np.asarray(sil[i], np.float64)[:T]))
    return rows


def rows_path(work: Path, corpus: str, blk: str) -> Path:
    return work / f"dyn_rows_{corpus}_{blk}.pkl"


def stage_extract(work: Path):
    t0 = time.time()
    for corpus, systems, folds in (("ami", M.ami_systems, E.V2_DEV_FOLDS), ("icsi", lambda: M.icsi_systems(work), M.ICSI_FOLDS)):
        blocks = systems()
        for blk, (bk, _) in BLOCKS.items():
            convs, mt, sing, _, _ = blocks[bk]
            rows = rows_from_block(convs, mt, sing, folds)
            rows_path(work, corpus, blk).write_bytes(pickle.dumps(rows))
            print(f"  {corpus} {blk}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
        del blocks


# --------------------------------------------------------------------------- specs and points
def prep(rows):
    return (np.array([r["onset"] for r in rows]), np.array([r["end"] for r in rows]), [r["pauses"] for r in rows])


def ref_specs(rows) -> tuple[dict, dict]:
    """(singles, hybrids) reference systems on stream rows, with bench_turn_icsi's emission rules."""
    head = [r["head"] for r in rows]
    to = [r["to_primary"] for r in rows]
    sil = [r["to_silero"] for r in rows]
    eh, es, ef = emit_fn("head"), emit_fn("sf"), emit_fn("frame")
    return ({"silero_timeout": (sil, ef), "head": (head, eh), "timeout_primary": (to, es)},
            {"hybrid": (head, eh, to, es), "head_or_silero": (head, eh, sil, ef)})


def fit_points(spec, hybrid: bool, rows, L, folds) -> dict:
    """bench_turn_icsi.fit_points on stream rows: 'all' = in-sample on every row, 'fold<f>' = on the rows not in f."""
    on, en, pauses = prep(rows)
    oc, ths = M.sweep_outcomes(spec, hybrid, on, en, pauses, L)
    pts = {"all": ths[E._select(oc, np.ones(len(rows), bool), MAX_FC, "turn", tie_miss=hybrid)]}
    for f in np.unique(folds):
        pts[f"fold{int(f)}"] = ths[E._select(oc, folds != f, MAX_FC, "turn", tie_miss=hybrid)]
    return pts


def dyn_grid_outcomes(rows, sil: str, s_names, L, cands) -> dict:
    """eot_outcomes-shaped matrices of every grid candidate of a dynamic policy at the fixed fire rule (score > -0.5);
    the columns index ``cands``."""
    on, en, pauses = prep(rows)
    emit = emit_fn("head")
    FC = np.zeros((len(rows), len(cands)), bool)
    LAT = np.full((len(rows), len(cands)), np.inf)
    PF = np.zeros((len(rows), len(cands)), np.int64)
    npause = None
    for j, prm in enumerate(cands):
        z = E._emit_transform(D.policy_scores(rows, sil, s_names, prm), en, emit)
        oc = eot_outcomes(z, on, en, [FIRE], post_end_frames=L, pauses=pauses)
        FC[:, j], LAT[:, j], PF[:, j] = oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0]
        npause = oc["npause"]
    return {"fc": FC, "lat": LAT, "pf": PF, "npause": npause, "ths": np.arange(len(cands), dtype=np.float64)}


def dyn_params(cand, s_names) -> dict:
    T0, tmin, coefs = cand
    return {"T0": int(T0), "Tmin": int(tmin), "Tmax": int(T0), "a": {n: int(a) for n, a in zip(s_names, coefs)}}


def as_cand(p: dict, s_names):
    return (p["T0"], p["Tmin"], [p["a"][n] for n in s_names])


def dyn_tracks(rows, sil, s_names, params_by_row) -> list[np.ndarray]:
    """The policy score track (silence - required) of each row at its own parameters (a dict per row)."""
    return [D.policy_scores([r], sil, s_names, as_cand(p, s_names))[0] for r, p in zip(rows, params_by_row)]


def dyn_outcomes_at(rows, sil, s_names, params: dict, L):
    on, en, pauses = prep(rows)
    z = E._emit_transform(D.policy_scores(rows, sil, s_names, as_cand(params, s_names)), en, emit_fn("head"))
    oc = eot_outcomes(z, on, en, [FIRE], post_end_frames=L, pauses=pauses)
    return oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0], oc["npause"]


def fmt_hybrid_point(th, dyn=None) -> dict:
    a, b = th
    d = {"theta": None if not np.isfinite(a) else float(a), "offset": None if not np.isfinite(b) else float(b)}
    if dyn is not None:
        d["dyn"] = dyn
    return d


def groups_of(rows):
    strata = np.array([r["stratum"] for r in rows])
    meet = np.array([r["meeting"] for r in rows])
    return ({"open": strata == "open", "taken": strata != "open"},
            {s: np.isin(meet, ms) for s, ms in M.SUBSETS.items()})


def paired(outs: dict, groups: dict, n: int, n_boot: int, pairs, hz) -> dict:
    """bench_turn_icsi.frozen_block's paired bootstraps (same resamples: seeds 0 / 1) over (system, horizon) outcomes."""
    res = {}
    idx = np.random.default_rng(0).integers(0, n, (n_boot, n))
    for a_, b_ in pairs:
        for h in hz:
            if (a_, h) not in outs or (b_, h) not in outs:
                continue
            A, Bb = outs[(a_, h)], outs[(b_, h)]
            d = {"all": E._paired(A, Bb, idx)}
            for g in ("open", "taken"):
                sub = np.nonzero(groups[g])[0]
                gi = np.random.default_rng(1).integers(0, len(sub), (n_boot, len(sub)))
                d[g] = E._paired((A[0][sub], A[1][sub]), (Bb[0][sub], Bb[1][sub]), gi)
            res[f"{a_} - {b_} | {h}"] = d
    return res


def fire_conditions(rows, points: dict, params_by_row: dict) -> dict:
    """Whole-window fire conditions (bool per frame; no emission delay, as bench_completeness_dyn.continuous) of the
    reference rules at ``points`` and of the dynamic policies at ``params_by_row`` {policy: [params per row]} and the
    head-OR-dyn rules at points[f'head_or_dyn_{policy}'] = {theta, offset}."""
    conds = {}
    th = points["silero_timeout"]
    conds["silero_timeout"] = [r["to_silero"] > th for r in rows]
    for name, key_b in (("head_or_silero", "to_silero"), ("hybrid", "to_primary")):
        ta, tb = points[name]["theta"], points[name]["offset"]
        ta, tb = (np.inf if ta is None else ta), (np.inf if tb is None else tb)
        conds[name] = [(r["head"] > ta) | (r[key_b] > tb) for r in rows]
    for name, prm in params_by_row.items():
        sil, s_names, _ = POLICIES[name]
        tr = dyn_tracks(rows, sil, s_names, prm)
        conds[f"dyn_{name}"] = [t > FIRE for t in tr]
        hn = f"head_or_dyn_{name}"
        if hn in points:
            ta, tb = points[hn]["theta"], points[hn]["offset"]
            ta, tb = (np.inf if ta is None else ta), (np.inf if tb is None else tb)
            conds[hn] = [(r["head"] > ta) | (t > tb) for r, t in zip(rows, tr)]
    return conds


# --------------------------------------------------------------------------- report
def frozen_block(ami_rows, icsi_rows, hz: dict, policies: dict, n_boot: int, committed: dict, cmp_json: dict, bk: str) -> dict:
    """Every system at its all-AMI point (and the two AMI fold points), scored on ICSI; references are asserted equal
    to runs/baselines_turn_icsi.json, the AMI fold points of the policies to runs/completeness.json."""
    folds_a = np.array([r["fold"] for r in ami_rows])
    on, en, pauses = prep(icsi_rows)
    groups, subsets = groups_of(icsi_rows)
    n = len(icsi_rows)
    res = {"systems": {}, "checks": {}}
    outs = {}
    sing_a, hyb_a = ref_specs(ami_rows)
    sing_i, hyb_i = ref_specs(icsi_rows)
    # references at bench_turn_icsi's frozen points
    for name in REFS:
        hybrid = name in hyb_a
        spec_a, spec_i = (hyb_a if hybrid else sing_a)[name], (hyb_i if hybrid else sing_i)[name]
        r = {}
        for h, L in hz.items():
            pts = fit_points(spec_a, hybrid, ami_rows, L, folds_a)
            r[h] = {"ami_points": {k: M.fmt_point(REF_NAMES[name], th) for k, th in pts.items()}}
            for k, th in pts.items():
                fc, lat, pf, npause = M.outcomes_at(spec_i, hybrid, on, en, pauses, L, th)
                r[h][f"icsi_at_ami_{k}"] = M.point_metrics(fc, lat, pf, npause, groups, n_boot if k == "all" else 0,
                                                           subsets if k == "all" else None)
                if k == "all":
                    outs[(name, h)] = (fc, lat)
            c = committed[bk]["frozen_ami"]["systems"][REF_NAMES[name]][h]["icsi_at_ami_all"]
            mine = r[h]["icsi_at_ami_all"]
            ok = (mine["miss_rate"], mine["fc_rate"]) == (c["miss_rate"], c["fc_rate"])
            res["checks"][f"{name} | {h} = baselines_turn_icsi.json"] = bool(ok)
            assert ok, (name, h, mine["miss_rate"], mine["fc_rate"], c["miss_rate"], c["fc_rate"])
        res["systems"][name] = r
    # dynamic policies and head OR dyn
    for name, (sil, s_names, kind) in policies.items():
        cands = D.grid(len(s_names))
        r, rh = {}, {}
        for h, L in hz.items():
            oc = dyn_grid_outcomes(ami_rows, sil, s_names, L, cands)
            pts = {"all": dyn_params(cands[E._select(oc, np.ones(len(ami_rows), bool), MAX_FC, "turn", tie_miss=True)], s_names)}
            for f in np.unique(folds_a):
                pts[f"fold{int(f)}"] = dyn_params(cands[E._select(oc, folds_a != f, MAX_FC, "turn", tie_miss=True)], s_names)
            cj = cmp_json["blocks"][bk]["systems"][f"dyn_{name}"][h]["fixed_5pct_turn_fc"]["params_by_fold"]
            ok = all(pts[f"fold{f}"] == cj[str(f)] for f in (0, 1))
            res["checks"][f"dyn_{name} | {h} fold points = completeness.json"] = bool(ok)
            assert ok, (name, h, pts, cj)
            r[h] = {"ami_points": pts}
            for k, prm in pts.items():
                fc, lat, pf, npause = dyn_outcomes_at(icsi_rows, sil, s_names, prm, L)
                r[h][f"icsi_at_ami_{k}"] = M.point_metrics(fc, lat, pf, npause, groups, n_boot if k == "all" else 0,
                                                           subsets if k == "all" else None)
                if k == "all":
                    outs[(f"dyn_{name}", h)] = (fc, lat)
            # head OR dyn: the joint (theta, offset) grid on AMI. 'all' = in-sample with the all-AMI policy; the fold
            # points = COMPLETENESS.md's protocol (each AMI row's track at its own fold's policy, selection on the other fold)
            eh = emit_fn("head")
            head_a, head_i = [x["head"] for x in ami_rows], [x["head"] for x in icsi_rows]
            hn = f"head_or_dyn_{name}"
            rh[h] = {"ami_points": {}}
            for k in pts:
                per_row = [pts[k] if k == "all" else pts[f"fold{x['fold']}"] for x in ami_rows]
                spec_a = (head_a, eh, dyn_tracks(ami_rows, sil, s_names, per_row), eh)
                on_a, en_a, pa_a = prep(ami_rows)
                oc_h, ths = M.sweep_outcomes(spec_a, True, on_a, en_a, pa_a, L)
                sel = np.ones(len(ami_rows), bool) if k == "all" else folds_a != int(k[-1])
                th = ths[E._select(oc_h, sel, MAX_FC, "turn", tie_miss=True)]
                rh[h]["ami_points"][k] = fmt_hybrid_point(th, pts[k])
                if k != "all":
                    cj = cmp_json["blocks"][bk]["systems"][hn][h]["fixed_5pct_turn_fc"]["thresholds_by_fold"][k[-1]]
                    ok = all(abs((x if x is not None else np.inf) - (y if y is not None else np.inf)) < 1e-6
                             for x, y in ((rh[h]["ami_points"][k]["theta"], cj["theta_a"]), (rh[h]["ami_points"][k]["offset"], cj["theta_b"])))
                    res["checks"][f"{hn} | {h} {k} point = completeness.json"] = bool(ok)
                    assert ok, (hn, h, k, rh[h]["ami_points"][k], cj)
                spec_i = (head_i, eh, dyn_tracks(icsi_rows, sil, s_names, [pts[k]] * n), eh)
                fc, lat, pf, npause = M.outcomes_at(spec_i, True, on, en, pauses, L, th)
                rh[h][f"icsi_at_ami_{k}"] = M.point_metrics(fc, lat, pf, npause, groups, n_boot if k == "all" else 0,
                                                            subsets if k == "all" else None)
                if k == "all":
                    outs[(hn, h)] = (fc, lat)
        res["systems"][f"dyn_{name}"] = r
        res["systems"][hn] = rh
    for name, r in res["systems"].items():
        print("    frozen " + name + ": " + ", ".join(
            f"{h} miss {r[h]['icsi_at_ami_all']['miss_rate']:.3f} open {r[h]['icsi_at_ami_all']['strata']['open']['miss_rate']:.3f} "
            f"fc {r[h]['icsi_at_ami_all']['fc_rate']:.3f}" for h in hz), flush=True)
    pairs = []
    for name in policies:
        pairs += [(f"dyn_{name}", b) for b in ("head_or_silero", "silero_timeout", "hybrid", "head")]
        pairs += [(f"head_or_dyn_{name}", b) for b in ("head_or_silero", "silero_timeout", "hybrid", "head", f"dyn_{name}")]
    pairs += [("head_or_silero", "silero_timeout"), ("head_or_silero", "hybrid")]
    res["paired"] = paired(outs, groups, n, n_boot, pairs, hz)
    return res


def crossfit_block(icsi_rows, hz: dict, policies: dict, n_boot: int, committed: dict, bk: str) -> dict:
    """The AMI protocol on ICSI (bench_completeness_fusion.Scorer over ICSI folds): references, the policies over the
    grid, and head OR dyn with each row's fold policy inside the joint grid."""
    from audioforge.conversation import bootstrap_ci, outcome_metrics
    sc = Scorer(icsi_rows, hz, n_boot)
    sing, hyb = ref_specs(icsi_rows)
    sc.single("silero_timeout", *sing["silero_timeout"])
    sc.single("head", *sing["head"])
    sc.hybrid("hybrid", *hyb["hybrid"])
    sc.hybrid("head_or_silero", *hyb["head_or_silero"])
    checks = {}
    for name in REFS:
        for h in hz:
            mine = sc.res["systems"][name][h]["fixed_5pct_turn_fc"]
            c = committed[bk]["crossfit_icsi"]["systems"][REF_NAMES[name]][h]["fixed_5pct_turn_fc"]
            ok = (mine["miss_rate"], mine["fc_rate"]) == (c["miss_rate"], c["fc_rate"])
            checks[f"{name} | {h} = baselines_turn_icsi.json"] = bool(ok)
            assert ok, (name, h, mine["miss_rate"], mine["fc_rate"], c["miss_rate"], c["fc_rate"])
    fitted = {}
    for name, (sil, s_names, kind) in policies.items():
        cands = D.grid(len(s_names))
        r = {}
        for h, L in hz.items():
            oc = dyn_grid_outcomes(icsi_rows, sil, s_names, L, cands)
            fc, lat, pf, th = E.v2_crossfit(oc, sc.folds, MAX_FC, "turn", tie_miss=True)
            chosen = {f: dyn_params(cands[int(j)], s_names) for f, j in th.items()}
            pt = outcome_metrics(fc, lat, pf, oc["npause"])
            pt.update(bootstrap_ci(fc, lat, n_boot, 0), params_by_fold=chosen)
            pt["strata"] = {g: {**outcome_metrics(fc[m], lat[m], pf[m], oc["npause"][m]), **bootstrap_ci(fc[m], lat[m], n_boot, 0)}
                            for g, m in sc.groups.items()}
            sc.outs[(f"dyn_{name}", h)] = (fc, lat)
            r[h] = {"fixed_5pct_turn_fc": pt}
            fitted[(name, h)] = chosen
        sc.res["systems"][f"dyn_{name}"] = r
        for ref_name in ("head_or_silero", "silero_timeout", "hybrid", "head"):
            sc.pair(f"dyn_{name}", ref_name)
        hn, rr = f"head_or_dyn_{name}", {}
        head = [x["head"] for x in icsi_rows]
        for h, L in hz.items():
            tr = dyn_tracks(icsi_rows, sil, s_names, [fitted[(name, h)][x["fold"]] for x in icsi_rows])
            sub = Scorer(icsi_rows, {h: L}, n_boot)
            sub.hybrid(hn, head, emit_fn("head"), tr, emit_fn("head"))
            rr[h] = sub.res["systems"][hn][h]
            rr[h]["fixed_5pct_turn_fc"]["thresholds_by_fold"] = {
                f: dict(v, dyn=fitted[(name, h)][int(f)]) for f, v in rr[h]["fixed_5pct_turn_fc"]["thresholds_by_fold"].items()}
            sc.outs[(hn, h)] = sub.outs[(hn, h)]
        sc.res["systems"][hn] = rr
        for ref_name in ("head_or_silero", "silero_timeout", "hybrid", "head", f"dyn_{name}"):
            sc.pair(hn, ref_name)
    sc.pair("head_or_silero", "silero_timeout")
    sc.pair("head_or_silero", "hybrid")
    for name, r in sc.res["systems"].items():
        print("    crossfit " + name + ": " + ", ".join(
            f"{h} miss {r[h]['fixed_5pct_turn_fc']['miss_rate']:.3f} open {r[h]['fixed_5pct_turn_fc']['strata']['open']['miss_rate']:.3f} "
            f"fc {r[h]['fixed_5pct_turn_fc']['fc_rate']:.3f}" for h in hz), flush=True)
    sc.res["checks"] = checks
    sc.res["fitted"] = {f"{k[0]} | {k[1]}": v for k, v in fitted.items()}
    return sc.res, fitted


def continuous_block(icsi_rows, frozen: dict, cross: dict, fitted: dict, policies: dict) -> dict:
    """Whole-window statistics at the 6 s points: frozen all-AMI parameters and the ICSI cross-fitted ones (per fold)."""
    groups, _ = groups_of(icsi_rows)
    fs = frozen["systems"]
    pts = {"silero_timeout": fs["silero_timeout"]["6s"]["ami_points"]["all"]["timeout_threshold"]}
    for name in ("head_or_silero", "hybrid"):
        p = fs[name]["6s"]["ami_points"]["all"]
        pts[name] = {"theta": p["theta"], "offset": p["timeout_threshold"]}
    prm = {}
    for name in policies:
        prm[name] = [fs[f"dyn_{name}"]["6s"]["ami_points"]["all"]] * len(icsi_rows)
        p = fs[f"head_or_dyn_{name}"]["6s"]["ami_points"]["all"]
        pts[f"head_or_dyn_{name}"] = {"theta": p["theta"], "offset": p["offset"]}
    out = {"definition": "event = rising edge of the fire condition over the whole window (frame grid, no emission "
                         "delay); lead = before the primary's onset, turn = inside it (cutoffs), post = after the end "
                         "(first = detection, dup = further ones), late = post-end events after another speaker has "
                         "started since the end; 6 s points",
           "frozen_ami": D.continuous_summary(icsi_rows, fire_conditions(icsi_rows, pts, prm), groups["open"])}
    # cross-fitted: each window at its own fold's point (fold-wise references are single tracks, so build per row)
    cs = cross["systems"]
    conds = {}
    th = cs["silero_timeout"]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
    conds["silero_timeout"] = [r["to_silero"] > th[r["fold"]] for r in icsi_rows]
    for name, key_b in (("head_or_silero", "to_silero"), ("hybrid", "to_primary")):
        th = cs[name]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
        conds[name] = [(r["head"] > (th[r["fold"]]["theta_a"] if th[r["fold"]]["theta_a"] is not None else np.inf))
                       | (r[key_b] > (th[r["fold"]]["theta_b"] if th[r["fold"]]["theta_b"] is not None else np.inf))
                       for r in icsi_rows]
    for name in policies:
        sil, s_names, _ = POLICIES[name]
        tr = dyn_tracks(icsi_rows, sil, s_names, [fitted[(name, "6s")][r["fold"]] for r in icsi_rows])
        conds[f"dyn_{name}"] = [t > FIRE for t in tr]
        th = cs[f"head_or_dyn_{name}"]["6s"]["fixed_5pct_turn_fc"]["thresholds_by_fold"]
        conds[f"head_or_dyn_{name}"] = [(r["head"] > (th[r["fold"]]["theta_a"] if th[r["fold"]]["theta_a"] is not None else np.inf))
                                        | (t > (th[r["fold"]]["theta_b"] if th[r["fold"]]["theta_b"] is not None else np.inf))
                                        for r, t in zip(icsi_rows, tr)]
    out["crossfit_icsi"] = D.continuous_summary(icsi_rows, conds, groups["open"])
    return out


def stage_report(a, work: Path) -> dict:
    t0 = time.time()
    committed = json.loads((ROOT / "runs" / "baselines_turn_icsi.json").read_text())
    cmp_json = json.loads((ROOT / "runs" / "completeness.json").read_text())["dyntimeout"]
    st_icsi = (work / "smartturn").is_dir()
    policies = {k: v for k, v in POLICIES.items() if st_icsi or "st" not in k}
    out = {"protocol": "eot-bench v2 on the held-out ICSI turns of bench_turn_icsi.py; frozen = dynamic-timeout parameters "
                       "(T0, Tmin, a) and the head-OR-dyn (theta, offset) fitted in-sample on all 974 AMI dev turns (<= 5 % "
                       "per-turn FC, scorer rule) applied to ICSI; crossfit = the AMI leave-meetings-out protocol on ICSI",
           "policy": "fire when silence_silero >= clamp(T0 - sum_i a_i s_i, Tmin, Tmax = T0) frames (80 ms); s = trail6 head posterior",
           "grid": {"T0_frames": D.T0_GRID, "Tmin_frames": D.TMIN_GRID, "a_single": D.A_GRID},
           "policies": list(policies), "smartturn_icsi_available": st_icsi,
           "skipped": [] if st_icsi else ["dyn silero|head+st and head OR dyn(silero|head+st): no smart-turn scores on the ICSI windows"],
           "meetings": committed["meetings"], "folds": committed["folds"], "subsets": committed["subsets"], "n_boot": a.n_boot}
    for blk, (bk, hz) in BLOCKS.items():
        ami_rows = pickle.loads(rows_path(work, "ami", blk).read_bytes())
        icsi_rows = pickle.loads(rows_path(work, "icsi", blk).read_bytes())
        out.setdefault("n", len(icsi_rows))
        print(f"  {bk}: frozen ({time.time() - t0:.0f}s)", flush=True)
        r = {"frozen_ami": frozen_block(ami_rows, icsi_rows, hz, policies, a.n_boot, committed, cmp_json, bk)}
        print(f"  {bk}: crossfit ({time.time() - t0:.0f}s)", flush=True)
        r["crossfit_icsi"], fitted = crossfit_block(icsi_rows, hz, policies, a.n_boot, committed, bk)
        if blk == "C":
            print(f"  {bk}: continuous ({time.time() - t0:.0f}s)", flush=True)
            r["continuous_whole_window"] = continuous_block(icsi_rows, r["frozen_ami"], r["crossfit_icsi"], fitted, policies)
        out[bk] = r
    out["sec"] = round(time.time() - t0, 1)
    return out


def print_report(out):
    for bk in ("C_extended_windows_stream", "A_default_windows_all"):
        r = out[bk]
        print(f"\n### {bk} frozen")
        for name, s in r["frozen_ami"]["systems"].items():
            cells = []
            for h, d in s.items():
                p = d["icsi_at_ami_all"]
                o = p["strata"]["open"]
                cells.append(f"{h}: {100*p['miss_rate']:.1f} [{100*p['miss_rate_ci'][0]:.1f}, {100*p['miss_rate_ci'][1]:.1f}] all, "
                             f"{100*o['miss_rate']:.1f} [{100*o['miss_rate_ci'][0]:.1f}, {100*o['miss_rate_ci'][1]:.1f}] open, "
                             f"{100*p['strata']['taken']['miss_rate']:.1f} taken, FC {100*p['fc_rate']:.1f}/{100*(p.get('fc_per_pause') or 0):.1f}, "
                             f"P50 open {o['p50_ms']}, point {d['ami_points']['all']}; folds "
                             + " / ".join(f"{100*d[f'icsi_at_ami_fold{f}']['miss_rate']:.1f} (open {100*d[f'icsi_at_ami_fold{f}']['strata']['open']['miss_rate']:.1f}, FC {100*d[f'icsi_at_ami_fold{f}']['fc_rate']:.1f}) @ {d['ami_points'][f'fold{f}']}" for f in (0, 1)))
                if "subsets" in p:
                    cells.append("subsets " + ", ".join(f"{k}: {100*v['miss_rate']:.1f} open {100*v['strata']['open']['miss_rate']:.1f} FC {100*v['fc_rate']:.1f}" for k, v in p["subsets"].items()))
            print(f"| {name} | " + " | ".join(cells))
        print("paired (frozen):")
        for k, d in r["frozen_ami"]["paired"].items():
            print(f"  {k}: all {100*d['all']['miss_diff']:+.1f} [{100*d['all']['miss_diff_ci'][0]:+.1f}, {100*d['all']['miss_diff_ci'][1]:+.1f}]"
                  f" open {100*d['open']['miss_diff']:+.1f} [{100*d['open']['miss_diff_ci'][0]:+.1f}, {100*d['open']['miss_diff_ci'][1]:+.1f}]"
                  f" taken {100*d['taken']['miss_diff']:+.1f} [{100*d['taken']['miss_diff_ci'][0]:+.1f}, {100*d['taken']['miss_diff_ci'][1]:+.1f}]")
        print(f"\n### {bk} crossfit")
        for name, s in r["crossfit_icsi"]["systems"].items():
            cells = []
            for h, d in s.items():
                p = d["fixed_5pct_turn_fc"]
                o = p["strata"]["open"]
                cells.append(f"{h}: {100*p['miss_rate']:.1f} [{100*p['miss_rate_ci'][0]:.1f}, {100*p['miss_rate_ci'][1]:.1f}] all, "
                             f"{100*o['miss_rate']:.1f} [{100*o['miss_rate_ci'][0]:.1f}, {100*o['miss_rate_ci'][1]:.1f}] open, "
                             f"{100*p['strata']['taken']['miss_rate']:.1f} taken, FC {100*p['fc_rate']:.1f}/{100*(p.get('fc_per_pause') or 0):.1f}, "
                             f"P50 open {o['p50_ms']}, {p.get('params_by_fold') or p.get('thresholds_by_fold')}")
            print(f"| {name} | " + " | ".join(cells))
        print("paired (crossfit):")
        for k, d in r["crossfit_icsi"]["paired"].items():
            print(f"  {k}: all {100*d['all']['miss_diff']:+.1f} [{100*d['all']['miss_diff_ci'][0]:+.1f}, {100*d['all']['miss_diff_ci'][1]:+.1f}]"
                  f" open {100*d['open']['miss_diff']:+.1f} [{100*d['open']['miss_diff_ci'][0]:+.1f}, {100*d['open']['miss_diff_ci'][1]:+.1f}]"
                  f" taken {100*d['taken']['miss_diff']:+.1f} [{100*d['taken']['miss_diff_ci'][0]:+.1f}, {100*d['taken']['miss_diff_ci'][1]:+.1f}]")
        if "continuous_whole_window" in r:
            for k in ("frozen_ami", "crossfit_icsi"):
                print(f"\ncontinuous ({k}):")
                for name, d in r["continuous_whole_window"][k].items():
                    x = d["all"]
                    print(f"  {name}: ev/w {x['events_per_window']}, lead {x['lead_events_per_window']}, cutoff {100*x['windows_with_turn_event(cutoff)']:.1f} %, "
                          f"post {100*x['windows_with_post_event']:.1f} %, dup {x['duplicate_post_events_per_window']} ({100*x['windows_with_duplicate']:.1f} %), "
                          f"first late {100*x['windows_whose_first_post_event_is_late']:.1f} / {100*d['open']['windows_whose_first_post_event_is_late']:.1f} / {100*d['taken']['windows_whose_first_post_event_is_late']:.1f} %")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["extract", "report"])
    p.add_argument("--work", default=str(BT.SCRATCH / "icsi_turn" / "work"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(ROOT / "runs" / "baselines_turn_icsi_dyn.json"))
    a = p.parse_args()
    BT._torch2()
    work = Path(a.work)
    if a.stage == "extract":
        stage_extract(work)
        return
    res = stage_report(a, work)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {a.out} ({res['sec']}s)")
    print_report(res)


if __name__ == "__main__":
    main()
