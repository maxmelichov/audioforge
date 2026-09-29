"""Fusion stage of scripts/research/bench_completeness.py (research/COMPLETENESS.md §2): learned fusion of our trail6 head,
smart-turn v3.2 and the two silence timeouts on eot-bench v2, cross-fitted by the v2 meeting folds, plus the
error complementarity of smart-turn and our hybrid on floor-open ends."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

import bench_turn_baselines as BT
import eval_stage1 as E
from audioforge.baselines import turn as B
from audioforge.conversation import bootstrap_ci, eot_outcomes, eot_outcomes_or, outcome_metrics

DUR_MAX = 64  # frames: the head's own dur_max (log-clipped counters)
FEATS = ("head_logit", "log_to_primary", "log_to_silero", "st_silero_p", "st_silero_on", "st_sortformer_p",
         "st_sortformer_on", "cmp_logit")  # cmp_logit: our completeness head's P(complete so far) (stage streams)
FEATURE_SETS = {  # name -> (feature names, emission kind of the fused decision = the slowest input's)
    "all": (FEATS[:7], "head"),
    "no_head": (("log_to_primary", "log_to_silero", "st_silero_p", "st_silero_on", "st_sortformer_p",
                 "st_sortformer_on"), "sf"),
    "speaker_unaware": (("log_to_silero", "st_silero_p", "st_silero_on"), "frame"),
    "head+silero": (("head_logit", "log_to_silero"), "head"),
}
CMP_SETS = {  # the same, with our completeness head in place of / next to smart-turn
    "all+cmp": (FEATS, "head"),
    "head+cmp": (("head_logit", "cmp_logit"), "head"),
    "head+silero+cmp": (("head_logit", "log_to_silero", "cmp_logit"), "head"),
    "silero+cmp": (("log_to_silero", "cmp_logit"), "eou"),
    "silero+st+cmp": (("log_to_silero", "st_silero_p", "st_silero_on", "cmp_logit"), "eou"),
}
MODELS = ("logit", "gbm")


def emit_fn(kind: str):
    """Emission rule of a fused decision: that of its slowest input (head = Sortformer rule + 160 ms head chunk)."""
    if kind == "head":
        return E._stream_emit(BT.C_SF, BT.R_SF, 2)
    return BT.EMIT[kind]


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def features(r: dict) -> np.ndarray:
    """(T, len(FEATS)) per-frame features of one window from its stored streams (all causal by construction)."""
    f = {"head_logit": _logit(r["head"]),
         "log_to_primary": np.log1p(np.minimum(r["to_primary"], DUR_MAX)) / np.log1p(DUR_MAX),
         "log_to_silero": np.log1p(np.minimum(r["to_silero"], DUR_MAX)) / np.log1p(DUR_MAX),
         "st_silero_p": r["st_silero"], "st_silero_on": (r["st_silero"] > 0).astype(np.float64),
         "st_sortformer_p": r["st_sortformer"], "st_sortformer_on": (r["st_sortformer"] > 0).astype(np.float64),
         "cmp_logit": _logit(r["cmp"]) if "cmp" in r else np.zeros(r["T"])}
    return np.stack([f[k] for k in FEATS], 1)


def train_frames(rows, horizon: int):
    """Training rows: frames [onset, min(T, end + horizon)) of every window; label = post-end; per-window weights
    give the pre-end and post-end parts of a turn the same total weight (the scorer is per turn)."""
    X, y, w = [], [], []
    for r in rows:
        F = features(r)
        o, e, T = r["onset"], r["end"], r["T"]
        hi = min(T, e + horizon)
        if hi <= o:
            continue
        X.append(F[o:hi])
        lab = (np.arange(o, hi) >= e).astype(np.float64)
        y.append(lab)
        n1, n0 = max(1, int(lab.sum())), max(1, int((1 - lab).sum()))
        w.append(np.where(lab > 0, 0.5 / n1, 0.5 / n0))
    return np.concatenate(X), np.concatenate(y), np.concatenate(w)


def fit_model(kind: str, X, y, w, cols):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    Xc = X[:, cols]
    if kind == "logit":
        m = LogisticRegression(C=1.0, max_iter=2000)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, max_leaf_nodes=8,
                                           l2_regularization=1.0, random_state=0)
    m.fit(Xc, y, sample_weight=w)
    return m


def oof_scores(rows, horizon: int, kind: str, cols) -> tuple[list[np.ndarray], dict]:
    """Out-of-fold fused score per window: the model fitted on the other v2 meeting fold scores this one."""
    folds = np.array([r["fold"] for r in rows])
    out = [None] * len(rows)
    info = {}
    for f in np.unique(folds):
        tr = [r for r, g in zip(rows, folds) if g != f]
        X, y, w = train_frames(tr, horizon)
        m = fit_model(kind, X, y, w, cols)
        if kind == "logit":
            info[f"fold{int(f)}_coef"] = {FEATS[c]: round(float(v), 3) for c, v in zip(cols, m.coef_[0])}
            info[f"fold{int(f)}_intercept"] = round(float(m.intercept_[0]), 3)
        for i in np.nonzero(folds == f)[0]:
            F = features(rows[i])[:, cols]
            out[i] = m.predict_proba(F)[:, 1].astype(np.float64)
    return out, info


# --------------------------------------------------------------------------- scorer (eval_stage1 / bench_turn_baselines)
class Scorer:
    """score_block's fixed <= 5 % per-turn FC operating point (cross-fitted by fold) for singles and OR-hybrids on one
    block, keeping the per-turn outcomes for paired bootstraps and the complementarity analysis."""

    def __init__(self, rows, horizons: dict, n_boot: int):
        self.rows, self.hz, self.n_boot = rows, horizons, n_boot
        self.on = np.array([r["onset"] for r in rows])
        self.en = np.array([r["end"] for r in rows])
        strata = np.array([r["stratum"] for r in rows])
        self.groups = {"open": strata == "open", "taken": strata != "open"}
        self.pauses = [r["pauses"] for r in rows]
        self.folds = np.array([r["fold"] for r in rows])
        self.avail = np.array([r["post_avail"] for r in rows])
        self.reason = np.array([r["end_reason"] for r in rows])
        self.outs: dict = {}
        self.res = {"n": len(rows), "strata_counts": {g: int(m.sum()) for g, m in self.groups.items()},
                    "systems": {}, "paired": {}}

    def _fixed(self, name, hz, L, oc, th_fmt=None):
        cens = ((self.avail < L) if L is not None else np.ones(len(self.rows), bool)) & (self.reason != "resume")
        fc, lat, pf, th = E.v2_crossfit(oc, self.folds, 0.05, "turn", tie_miss=th_fmt is not None)
        if th_fmt is not None:
            th = {f: th_fmt(j) for f, j in th.items()}
        pt = outcome_metrics(fc, lat, pf, oc["npause"])
        pt.update(bootstrap_ci(fc, lat, self.n_boot, 0), thresholds_by_fold=th)
        m = np.isinf(lat) & ~fc
        pt.update(n_miss=int(m.sum()), miss_censored=int((m & cens).sum()))
        pt["strata"] = {g: {**outcome_metrics(fc[msk], lat[msk], pf[msk], oc["npause"][msk]),
                            **bootstrap_ci(fc[msk], lat[msk], self.n_boot, 0)} for g, msk in self.groups.items()}
        self.outs[(name, hz)] = (fc, lat)
        return {"fixed_5pct_turn_fc": pt}

    def single(self, name, scores, emit):
        r = {}
        for hz, L in self.hz.items():
            z = E._emit_transform(scores, self.en, emit)
            allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in z]))
            ths = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
            oc = eot_outcomes(z, self.on, self.en, ths, post_end_frames=L, pauses=self.pauses)
            r[hz] = self._fixed(name, hz, L, oc)
        self.res["systems"][name] = r
        return r

    def hybrid(self, name, sa, ea, sb, eb):
        r = {}
        for hz, L in self.hz.items():
            za, zb = E._emit_transform(sa, self.en, ea), E._emit_transform(sb, self.en, eb)
            ga, gb = E.v2_hybrid_grid(za, self.on, self.en, self.pauses), E.v2_hybrid_grid(zb, self.on, self.en, self.pauses)
            oc = eot_outcomes_or(za, zb, self.on, self.en, ga, gb, post_end_frames=L, pauses=self.pauses)

            def th_fmt(j, oc=oc):
                ta, tb_ = oc["ths"][int(round(j))]
                return {"theta_a": float(ta) if np.isfinite(ta) else None,
                        "theta_b": float(tb_) if np.isfinite(tb_) else None}
            oc_i = dict(oc, ths=np.arange(oc["fc"].shape[1], dtype=np.float64))
            r[hz] = self._fixed(name, hz, L, oc_i, th_fmt=th_fmt)
        self.res["systems"][name] = r
        return r

    def pair(self, a, b):
        n = len(self.rows)
        idx = np.random.default_rng(0).integers(0, n, (self.n_boot, n))
        for hz in self.hz:
            if (a, hz) not in self.outs or (b, hz) not in self.outs:
                continue
            A, Bb = self.outs[(a, hz)], self.outs[(b, hz)]
            d = {"all": E._paired(A, Bb, idx)}
            for g in ("open", "taken"):
                sub = np.nonzero(self.groups[g])[0]
                gi = np.random.default_rng(1).integers(0, len(sub), (self.n_boot, len(sub)))
                d[g] = E._paired((A[0][sub], A[1][sub]), (Bb[0][sub], Bb[1][sub]), gi)
            self.res["paired"][f"{a} - {b} | {hz}"] = d


def pause_vs_end_auc(rows, key: str, post_frames: int = 13) -> dict:
    """AUC of a per-frame stream between within-turn pauses (max over each pause's frames; the negatives) and the first
    1.04 s after each turn end (max; the positives): BASELINES.md's smart-turn AMI diagnostic (0.58 there)."""
    from audioforge.heads.completeness import auc
    neg, pos = [], []
    for r in rows:
        x = np.asarray(r[key], np.float64)
        e = r["end"]
        if e < len(x):
            pos.append(float(x[e: e + post_frames].max()))
        for a_, b_ in r["pauses"]:
            seg = x[max(a_, 0): min(b_, len(x))]
            if len(seg):
                neg.append(float(seg.max()))
    y = [False] * len(neg) + [True] * len(pos)
    return {"auc": round(auc(y, neg + pos), 4), "n_pauses": len(neg), "n_ends": len(pos)}


# --------------------------------------------------------------------------- complementarity
def complementarity(rows, sc: Scorer, within_sec: float = 1.0) -> dict:
    """Floor-open ends at the 6 s horizon: among the turns a fitted system misses (not cut off, no firing within the
    horizon), the fraction where smart-turn's own decision (P >= 0.5 at a trigger whose decision time lies within
    ``within_sec`` after the true end) says 'complete'; and the converse: among open ends where smart-turn does not say
    complete within that time, the fraction each fitted system detects within 1 s / within the horizon."""
    open_ = np.nonzero(sc.groups["open"])[0]
    res = {"within_sec": within_sec, "n_open": int(len(open_))}

    def st_says(r, which, thr=0.5):
        e = r["end"] * B.FRAME
        if which == "silero":
            taus, ps = r["pc_tau"], r["pc_p"]
        else:
            taus, ps = (r["sf_frames"] + 1) * B.FRAME, r["sf_p"]
        ok = (taus >= e - 1e-9) & (taus <= e + within_sec + 1e-9)
        return bool(np.any(ps[ok] >= thr)), bool(ok.any())

    for which in ("silero", "sortformer"):
        says = np.array([st_says(rows[i], which)[0] for i in open_])
        trig = np.array([st_says(rows[i], which)[1] for i in open_])
        res[f"smartturn_{which}"] = {"says_complete_within": int(says.sum()), "triggered_within": int(trig.sum()),
                                     "frac_open_says_complete": round(float(says.mean()), 4)}
        for name in ("hybrid", "head", "head_or_silero", "silero_timeout", "fusion_logit_all", "fusion_gbm_all", "cmp",
                     "cmp+silero_timeout", "fusion_logit_all+cmp"):
            if (name, "6s") not in sc.outs:
                continue
            fc, lat = sc.outs[(name, "6s")]
            fc, lat = fc[open_], lat[open_]
            missed = ~fc & np.isinf(lat)
            caught1 = ~fc & (lat <= within_sec * 1000 + 1e-6)
            caught6 = ~fc & np.isfinite(lat)
            d = {"n_missed": int(missed.sum()),
                 "missed_and_smartturn_complete": int((missed & says).sum()),
                 "frac_missed_that_smartturn_gets": round(float(says[missed].mean()), 4) if missed.any() else None,
                 "missed_and_smartturn_triggered": int((missed & trig).sum()),
                 "n_smartturn_not_complete": int((~says).sum()),
                 "frac_smartturn_misses_caught_1s": round(float(caught1[~says].mean()), 4) if (~says).any() else None,
                 "frac_smartturn_misses_caught_horizon": round(float(caught6[~says].mean()), 4) if (~says).any() else None,
                 "frac_smartturn_hits_caught_1s": round(float(caught1[says].mean()), 4) if says.any() else None}
            res[f"smartturn_{which}"][name] = d
    # the union of the two triggers
    says = np.array([st_says(rows[i], "silero")[0] or st_says(rows[i], "sortformer")[0] for i in open_])
    res["smartturn_either"] = {"frac_open_says_complete": round(float(says.mean()), 4)}
    for name in ("hybrid", "head_or_silero"):
        fc, lat = sc.outs[(name, "6s")]
        fc, lat = fc[open_], lat[open_]
        missed = ~fc & np.isinf(lat)
        res["smartturn_either"][name] = {"n_missed": int(missed.sum()),
                                         "frac_missed_that_smartturn_gets": round(float(says[missed].mean()), 4)}
    return res


def stage_fusion(a, work: Path):
    t0 = time.time()
    ref = json.loads(BT.REF_JSON.read_text())["turn_v2"]
    base_json = json.loads((BT.ROOT / "runs" / "baselines_turn.json").read_text())
    out = {"protocol": "eot-bench v2, same windows / scorer / folds as runs/baselines_turn.json; fusion models fitted "
                       "on the other meeting fold (out-of-fold scores), thresholds cross-fitted on the same folds",
           "features": FEATS, "feature_sets": {k: list(v[0]) for k, v in FEATURE_SETS.items()},
           "models": {"logit": "sklearn LogisticRegression(C=1)", "gbm": "HistGradientBoosting(200 x depth 3, lr 0.05)"},
           "train_frames": "[onset, end + horizon) per window, label = post-end, pre/post halves weighted equally per turn",
           "n_boot": a.n_boot, "blocks": {}}
    ext_key = None
    for blk, hz in (("C", {"2s": 25, "6s": 75}), ("A", {"2s": None})):
        rows = pickle.loads((work / f"streams_{blk}.pkl").read_bytes())
        cmp_dir = work / "cmp"
        have_cmp = cmp_dir.exists() and all((cmp_dir / f"{r['key'] if blk == 'C' else ext_key[i]}.npy").exists()
                                            for i, r in enumerate(rows)) if blk == "C" or ext_key else False
        if have_cmp:
            for i, r in enumerate(rows):
                r["cmp"] = np.load(cmp_dir / f"{r['key'] if blk == 'C' else ext_key[i]}.npy")[: r["T"]].astype(np.float64)
        sc = Scorer(rows, hz, a.n_boot)
        head = [r["head"].astype(np.float64) for r in rows]
        top = [r["to_primary"].astype(np.float64) for r in rows]
        tsil = [r["to_silero"].astype(np.float64) for r in rows]
        eh, es, ef = E._stream_emit(BT.C_SF, BT.R_SF, 2), E._stream_emit(BT.C_SF, BT.R_SF), BT.EMIT["frame"]
        sc.single("head", head, eh)
        sc.single("silero_timeout", tsil, ef)
        sc.single("smartturn_silero", [r["st_silero"].astype(np.float64) for r in rows], ef)
        sc.hybrid("hybrid", head, eh, top, es)
        sc.hybrid("head_or_silero", head, eh, tsil, ef)
        if blk == "C":
            ext_key = [r["key"] for r in rows]  # block A windows share their (extended) window's stream, cropped
        sets = dict(FEATURE_SETS)
        if have_cmp:
            cmp = [r["cmp"] for r in rows]
            sc.single("cmp", cmp, emit_fn("eou"))
            sc.hybrid("cmp+silero_timeout", cmp, emit_fn("eou"), tsil, ef)
            sc.hybrid("head_or_cmp", head, eh, cmp, emit_fn("eou"))
            sets.update(CMP_SETS)
        out.setdefault("completeness_stream", have_cmp)
        horizon = 75 if blk == "C" else 25
        fits = {}
        for fs_name, (cols_names, kind) in sets.items():
            cols = [FEATS.index(c) for c in cols_names]
            for mk in MODELS:
                name = f"fusion_{mk}_{fs_name}"
                z, info = oof_scores(rows, horizon, mk, cols)
                fits[name] = info
                sc.single(name, z, emit_fn(kind))
                if fs_name in ("all", "no_head", "all+cmp", "head+cmp"):
                    sc.hybrid(name + "+silero_timeout", z, emit_fn(kind), tsil, ef)
        names = [n for n in sc.res["systems"] if n.startswith("fusion") or n.startswith("cmp") or n == "head_or_cmp"]
        for n in names:
            for r_ in ("hybrid", "head_or_silero", "silero_timeout"):
                sc.pair(n, r_)
        if have_cmp:  # the marginal value of the completeness stream next to the same features without it
            for mk in MODELS:
                sc.pair(f"fusion_{mk}_head+silero+cmp", f"fusion_{mk}_head+silero")
                sc.pair(f"fusion_{mk}_all+cmp", f"fusion_{mk}_all")
                sc.pair(f"fusion_{mk}_silero+cmp", f"fusion_{mk}_speaker_unaware")
            sc.pair("cmp+silero_timeout", "smartturn_silero+timeout") if "smartturn_silero+timeout" in sc.res["systems"] else None
            sc.res["cmp_pause_vs_end_auc"] = pause_vs_end_auc(rows, "cmp")
            sc.res["smartturn_pause_vs_end_auc"] = pause_vs_end_auc(rows, "st_silero")
        sc.pair("head_or_silero", "hybrid")
        # our rows must reproduce the committed runs exactly
        blk_key = "C_extended_windows_stream" if blk == "C" else "A_default_windows_all"
        for mine, theirs, src in (("hybrid", "hybrid_stream_causal_dominant", ref), ("head", "head_v3_stream_causal_dominant", ref),
                                  ("silero_timeout", "silero_timeout", base_json), ("head_or_silero", "head_trail6+silero_timeout", base_json),
                                  ("smartturn_silero", "smartturn_silero", base_json)):
            for h in hz:
                x = sc.res["systems"][mine][h]["fixed_5pct_turn_fc"]
                y = src[blk_key]["systems"][theirs][h]["fixed_5pct_turn_fc"]
                assert (x["miss_rate"], x["fc_rate"], x["miss_rate_ci"]) == (y["miss_rate"], y["fc_rate"], y["miss_rate_ci"]), \
                    (blk, mine, h, x["miss_rate"], y["miss_rate"])
        sc.res["reference_rows_reproduced"] = True
        sc.res["fits"] = fits
        if blk == "C":
            sc.res["complementarity_open_6s"] = complementarity(rows, sc)
        out["blocks"][blk_key] = sc.res
        print(f"  {blk}: {time.time() - t0:.0f}s", flush=True)
    out["sec"] = round(time.time() - t0, 1)
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    prev["fusion"] = out
    Path(a.out).write_text(json.dumps(prev, indent=1, default=float))
    print(f"wrote {a.out}")
    print_table(out)


def print_table(out):
    for blk, r in out["blocks"].items():
        print(f"\n### {blk}")
        hzs = [h for h in ("6s", "2s") if h in next(iter(r["systems"].values()))]
        print("| system | " + " | ".join(f"{h} miss all | {h} open | {h} taken | {h} FC turn/pause" for h in hzs) + " |")
        for name, s in r["systems"].items():
            cells = []
            for h in hzs:
                p = s[h]["fixed_5pct_turn_fc"]
                o, t = p["strata"]["open"], p["strata"]["taken"]
                cells.append(f"{100*p['miss_rate']:.1f} [{100*p['miss_rate_ci'][0]:.1f}, {100*p['miss_rate_ci'][1]:.1f}] | "
                             f"{100*o['miss_rate']:.1f} [{100*o['miss_rate_ci'][0]:.1f}, {100*o['miss_rate_ci'][1]:.1f}] | "
                             f"{100*t['miss_rate']:.1f} | {100*p['fc_rate']:.1f} / {100*(p.get('fc_per_pause') or 0):.1f}")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print("\npaired (a - b): miss diff [CI] all / open")
        for k, d in r["paired"].items():
            print(f"  {k}: {100*d['all']['miss_diff']:+.1f} [{100*d['all']['miss_diff_ci'][0]:+.1f}, {100*d['all']['miss_diff_ci'][1]:+.1f}]"
                  f" / {100*d['open']['miss_diff']:+.1f} [{100*d['open']['miss_diff_ci'][0]:+.1f}, {100*d['open']['miss_diff_ci'][1]:+.1f}]"
                  f"  fc {100*d['all']['fc_a']:.1f} vs {100*d['all']['fc_b']:.1f}")
        if "complementarity_open_6s" in r:
            print("\ncomplementarity:", json.dumps(r["complementarity_open_6s"], indent=1))
