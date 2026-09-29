"""research/IMPROVEMENTS.md section 2c: latency / false-cutoff frontier of the served turn head (the product knob).

From the stored score arrays only (no model runs): eot-bench v2 AMI dev (974) and ICSI held-out (1312), 6 s horizon.
Systems on the TS-VAD 5 s-print track (research/IMPROVE_115M.md A.2): the served head alone (threshold swept),
hybrid_dyn (head threshold x dyn offset swept jointly), the plain silence timeout on P(target) (timeout_ms swept) and
the Silero any-speaker timeout (timeout_ms swept). For each FC budget (3 / 5 / 7.5 / 10 / 15 % per turn) the point
with the lowest P50 (then miss) whose realised FC over all turns is within the budget: an in-sample frontier (the
operating point is chosen on the same turns), to show the trade-off, not a held-out number. The Sortformer-track head
scores were lost with the scratch (tsvad_chain.sh rebuilds them), so that input is not in the table.

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import tsvad as T  # noqa: E402

BUDGETS = (0.03, 0.05, 0.075, 0.10, 0.15)
OUT = ROOT / "runs" / "eot_frontier.json"


def _q(x, q):
    """Quantile of latencies with misses as +inf; None when it falls on a miss (> the 6 s horizon)."""
    if not len(x):
        return None
    v = np.sort(np.asarray(x, np.float64))[min(len(x) - 1, int(np.floor(q * (len(x) - 1))))]
    return float(v) if np.isfinite(v) else None


def frontier(oc, strata, cfgs):
    fc, lat = oc["fc"], oc["lat"]
    rate = fc.mean(0)
    rows = {}
    for b in BUDGETS:
        best = None
        for j in np.nonzero(rate <= b)[0]:
            keep = lat[~fc[:, j], j]
            p50 = float(np.quantile(keep, 0.5)) if len(keep) else np.inf
            p50 = np.inf if not np.isfinite(p50) else p50
            miss = float(np.isinf(lat[:, j]).mean())
            key = (p50, miss)
            if best is None or key < best[0]:
                best = (key, j)
        if best is None:
            rows[str(b)] = None
            continue
        j = best[1]
        keep = lat[~fc[:, j], j]
        rows[str(b)] = {"fc": round(float(rate[j]), 4), "miss": round(float(np.isinf(lat[:, j]).mean()), 4),
                        "open_miss": round(float(np.isinf(lat[strata == "open", j]).mean()), 4),
                        "p10_ms": _q(keep, 0.1), "p50_ms": _q(keep, 0.5), "cfg": cfgs[j]}
    return rows


def main():
    import eval_stage1 as E
    import bench_turn_icsi as M
    from audioforge.conversation import eot_outcomes, eot_outcomes_or, floor_stratum, pause_runs
    ec = lambda t: (t // T.CHUNK + 1) * T.CHUNK  # noqa: E731
    ef = lambda t: t + 1  # noqa: E731
    res = {}
    for corpus in ("ami", "icsi"):
        ext, meta, ds = T.bench_windows(corpus)
        convs = [dict(v) for v in ext]
        for v in convs:
            v.pop("audio", None)
        Tl = [len(v["spk_act"]) for v in convs]
        on = np.array([v["onset_frame"] for v in convs])
        en = np.array([v["turn_end_frame"] for v in convs])
        strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
        pauses = [pause_runs(v["hes"]) for v in convs]
        sil = [M.silero_timeout_track(np.load(T.SILERO[corpus] / f"{T.wkey(v)}.npy"), t)[:t] for v, t in zip(ext, Tl)]
        head = [T.stored_scores(corpus, "tsvad_spk", v)[:t] for v, t in zip(ext, Tl)]
        acts = [T.binding_inputs(corpus, "tsvad_spk", v)[0][:t] for v, t in zip(ext, Tl)]
        to = [E.silence_scores(x, E.arm_frame(x)) for x in acts]
        zh = E._emit_transform(head, en, ec)
        zt = E._emit_transform(to, en, ec)
        zs = E._emit_transform(sil, en, ef)
        r = {}
        grid_h = np.unique(np.quantile(np.concatenate([np.asarray(x) for x in head]), np.linspace(0.5, 0.9999, 400)))
        oc = eot_outcomes(zh, on, en, grid_h, post_end_frames=75)
        r["head"] = frontier(oc, strata, [{"theta": float(t)} for t in np.sort(grid_h)])
        for name, z in (("timeout_tsvad", zt), ("timeout_silero", zs)):
            g = np.arange(1, 76, dtype=np.float64) - 0.5
            oc = eot_outcomes(z, on, en, g, post_end_frames=75)
            r[name] = frontier(oc, strata, [{"timeout_ms": int((t + 0.5) * 80)} for t in np.sort(g)])
        blocks, cfgs = [], []
        thetas = np.unique(np.quantile(np.concatenate([np.asarray(x) for x in head]), np.linspace(0.5, 0.9995, 40)))
        zd = E._emit_transform([T.dyn_track(s, h) for s, h in zip(sil, head)], en, ec)
        offs = np.arange(-6.0, 8.01, 0.5)
        for th in thetas:
            ind = [(np.asarray(a) > th).astype(np.float64) for a in zh]
            oc = eot_outcomes_or(ind, zd, on, en, [0.5], offs, post_end_frames=75)
            blocks.append(oc)
            cfgs += [{"theta": float(th), "offset": float(o)} for o in np.sort(offs)]
        oc = {k: np.concatenate([b[k] for b in blocks], 1) for k in ("fc", "lat")}
        r["hybrid_dyn"] = frontier(oc, strata, cfgs)
        res[corpus] = r
        for name, rows in r.items():
            for b, x in rows.items():
                if x:
                    print(f"{corpus} {name:15s} FC<= {float(b) * 100:4.1f}: FC {x['fc'] * 100:4.1f} miss {x['miss'] * 100:5.1f} "
                          f"open {x['open_miss'] * 100:5.1f} P10 {x['p10_ms']} P50 {x['p50_ms']}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
