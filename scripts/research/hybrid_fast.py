"""research/IMPROVEMENTS.md section 2b: a fast path for the head-gated turn rule (the user's comparison with
smart-turn's ~225 ms decision: 200 ms Silero stop + 25 ms inference).

hybrid_dyn (shipped) = head >= theta OR Silero silence - clamp(75 - 55 p, 2, 75) > offset: it cannot fire before
7 frames (560 ms) of silence even at p -> 1. hybrid_fast adds one arm: at the frame where the Silero silence (the
Pipecat VAD state machine's silence since the last speech chunk, frames) first reaches m (m = 2 / 3 / 4 / 6 frames =
160 / 240 / 320 / 480 ms) the head's posterior on that frame is read once, and the turn ends if p >= tau_fast; else
hybrid_dyn decides. The arm's decision waits for the head's emission of that frame.

eot-bench v2 (AMI dev 974, ICSI held-out 1312, 6 s horizon), both speaker-activity inputs of research/IMPROVE_115M.md
A.2: the Sortformer causal-dominant track (served head, stored scores) and the TS-VAD 5 s-print track. Operating points:
(theta, offset, tau_fast) jointly, <= 5 % per-turn false cuts, cross-fitted on leave-meetings-out halves (the A.2
protocol); ICSI also with the AMI-fitted point frozen. Grids: theta in hybrid_dyn's own per-fold choices, offset in
1..6 frames (0.5 steps), tau in 30 quantiles of the fast arm's values + never.

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/hybrid_fast.py --out runs/hybrid_fast.json
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
import tsvad as T  # noqa: E402

MINS = (2, 3, 4, 6)
OFFSETS = tuple(np.arange(1.0, 6.01, 0.5))


def load(corpus, inp):
    import eval_stage1 as E
    import bench_turn_icsi as M
    ext, meta, ds = T.bench_windows(corpus)
    convs = [dict(v) for v in ext]
    for v in convs:
        v.pop("audio", None)
    Tl = [len(v["spk_act"]) for v in convs]
    sil = [M.silero_timeout_track(np.load(T.SILERO[corpus] / f"{T.wkey(v)}.npy"), t)[:t] for v, t in zip(ext, Tl)]
    if inp == "sortformer":
        head = [T.stored_scores(corpus, "causal_dominant", v)[:t] for v, t in zip(ext, Tl)]
        emit = E._stream_emit(T.C_SF, T.R_SF, T.CHUNK)
    else:
        head = [T.stored_scores(corpus, "tsvad_spk", v)[:t] for v, t in zip(ext, Tl)]
        emit = lambda t: (t // T.CHUNK + 1) * T.CHUNK  # noqa: E731
    fold = T.folds_of(corpus, convs)
    return convs, sil, head, emit, fold


def outcomes(convs, sil, head, emit, fold, thetas, mins=MINS, taus_fixed=None, offsets=OFFSETS, ref=True):
    """Joint outcome table over (m, theta, offset, tau) -> dict(fc, lat, pf, npause, cfg list)."""
    import eval_stage1 as E
    from audioforge.conversation import eot_outcomes, eot_outcomes_or, pause_runs
    on = np.array([v["onset_frame"] for v in convs])
    en = np.array([v["turn_end_frame"] for v in convs])
    pauses = [pause_runs(v["hes"]) for v in convs]
    zh = E._emit_transform(head, en, emit)
    zd = E._emit_transform([T.dyn_track(s, h) for s, h in zip(sil, head)], en, emit)
    blocks, cfgs = [], []
    # hybrid_dyn alone (the reference, same grid): the fast arm never fires
    for m in ((None,) if ref else ()) + tuple(mins):
        if m is None:
            zf = [np.zeros_like(np.asarray(z, np.float64)) - 1.0 for z in zh]
            taus = [np.inf]
        else:
            fast = [np.where(np.asarray(s)[: len(h)] == m, np.asarray(h, np.float64), -1.0) for s, h in zip(sil, head)]
            zf = E._emit_transform(fast, en, emit)
            vals = np.concatenate([np.asarray(f)[np.asarray(f) >= 0] for f in fast])
            taus = list(np.unique(np.quantile(vals, np.linspace(0.5, 0.999, 30)))) + [np.inf] \
                if taus_fixed is None else list(taus_fixed)
        for th in thetas:
            for off in offsets:
                ind = [np.maximum(np.asarray(a) > th, np.asarray(b) > off).astype(np.float64) for a, b in zip(zh, zd)]
                oc = eot_outcomes_or(zf, ind, on, en, taus, [0.5], post_end_frames=75, pauses=pauses)
                blocks.append(oc)
                cfgs += [{"m": m, "theta": float(th), "offset": float(off), "tau": float(t)} for t in np.sort(taus)]
    oc = {k: np.concatenate([b[k] for b in blocks], 1) for k in ("fc", "lat", "pf")}
    oc["npause"] = blocks[0]["npause"]
    oc["ths"] = np.arange(oc["fc"].shape[1], dtype=np.float64)
    return oc, cfgs


def point_metrics(fc, lat, strata):
    from audioforge.conversation import bootstrap_ci, outcome_metrics
    r = outcome_metrics(fc, lat, np.zeros(len(fc), np.int64), np.zeros(len(fc), np.int64))
    keep = lat[~fc]
    r["p10_ms"] = float(np.quantile(keep, 0.1)) if len(keep) else None
    r.update(bootstrap_ci(fc, lat, 1000, 0))
    r["open_miss"] = float(np.isinf(lat[strata == "open"]).mean())
    return r


def main():
    import eval_stage1 as E
    from audioforge.conversation import floor_stratum
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "runs" / "hybrid_fast.json"))
    p.add_argument("--inputs", default="sortformer,tsvad")
    a = p.parse_args()
    ref = json.loads((T.WORK / "report.json").read_text())
    res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    t0 = time.time()
    for inp in a.inputs.split(","):
        frozen_cfg = None
        for corpus in ("ami", "icsi"):
            convs, sil, head, emit, fold = load(corpus, inp)
            en = [v["turn_end_frame"] for v in convs]
            strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
            # hybrid_dyn's own theta per fold (A.2 report for tsvad; for sortformer: refit below on the same grid)
            thetas = sorted({round(float(x), 6) for x in np.quantile(np.concatenate([np.asarray(h) for h in head]),
                                                                      [0.97, 0.98, 0.99, 0.995, 0.998])})
            if inp == "tsvad":
                thetas = sorted(set(thetas) | {round(v[0], 6) for v in ref[corpus]["systems"]["hybrid_dyn_tsvad_spk"]["6s"]["thresholds_by_fold"].values()})
            oc, cfgs = outcomes(convs, sil, head, emit, fold, thetas)
            out = {}
            for m in (None,) + MINS:
                cols = np.array([j for j, c in enumerate(cfgs) if c["m"] == m])
                sub = {k: (oc[k][:, cols] if k in ("fc", "lat", "pf") else oc[k]) for k in ("fc", "lat", "pf", "npause")}
                sub["ths"] = np.arange(len(cols), dtype=np.float64)
                fc, lat, pf, th = E.v2_crossfit(sub, fold, 0.05, "turn", tie_miss=True)
                r = point_metrics(fc, lat, strata)
                r["cfg_by_fold"] = {f: cfgs[cols[int(j)]] for f, j in th.items()}
                key = "hybrid_dyn" if m is None else f"hybrid_fast_{m * 80}ms"
                out[key] = r
                if corpus == "ami":  # the AMI-wide point for the frozen ICSI row
                    j = E._select(sub, np.ones(len(fold), bool), 0.05, "turn", tie_miss=True)
                    out[key]["cfg_all"] = cfgs[cols[j]]
                elif frozen_cfg is not None and key in frozen_cfg:
                    c = frozen_cfg[key]
                    j = next(i for i, cc in enumerate(cfgs) if cc == c) if c in cfgs else None
                    if j is None:  # the frozen theta may not be in ICSI's grid: rebuild for it
                        oc2, cfgs2 = outcomes(convs, sil, head, emit, fold, [c["theta"]],
                                              mins=(c["m"],) if c["m"] else (), taus_fixed=[c["tau"]],
                                              offsets=[c["offset"]], ref=c["m"] is None)
                        j = 0
                        fcz, latz = oc2["fc"][:, j], oc2["lat"][:, j]
                    else:
                        fcz, latz = oc["fc"][:, j], oc["lat"][:, j]
                    out[key]["frozen_ami"] = point_metrics(fcz, latz, strata)
                log = out[key]
                print(f"{inp:10s} {corpus} {key:20s} miss {log['miss_rate'] * 100:5.1f} FC {log['fc_rate'] * 100:4.1f} "
                      f"P10 {log['p10_ms']} P50 {log['p50_ms']} open {log['open_miss'] * 100:5.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if corpus == "ami":
                frozen_cfg = {k: v["cfg_all"] for k, v in out.items()}
            res.setdefault(inp, {})[corpus] = out
            Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
