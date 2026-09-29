"""Our completeness head on TurnBench dev with the OFFICIAL scorer (research/COMPLETENESS.md §3.4), built on
scripts/research/bench_turnbench.py (its stages, event rules, scorer wrapper and operating-point rule are imported unchanged;
this script only adds the completeness systems and writes to its own work dir / output).

Systems (per speaker k, both channels are given to every TurnBench baseline):
  cmp                 P(complete so far) of the completeness head (runs/stage1_completeness.afm, frozen 115M encoder,
                      [70, 1] streaming mask) on CHANNEL k; events = rising edges above theta with the 2 s refractory,
                      committed at the 160 ms chunk end (smart-turn's own TurnBench baseline is per channel too)
  cmp+silero_timeout  cmp OR Pipecat's Silero state machine timeout on channel k (their deployable cascade)
  silero_timeout      the reference cascade alone (recomputed here so the rows share one vad stage)
Stages:  vad (their stage_vad, this work dir)  ->  cmp (MPS or CPU, resumable)  ->  sweep
  W=<scratch>/completeness/turnbench
  .venv/bin/python scripts/research/bench_turnbench_cmp.py --stage vad --work $W
  .venv/bin/python scripts/research/bench_turnbench_cmp.py --stage cmp --work $W --device mps
  .venv/bin/python scripts/research/bench_turnbench_cmp.py --stage sweep --work $W
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_turn_baselines as BT  # noqa: E402
import bench_turnbench as TB  # noqa: E402
from audioforge.datasets.ami import FRAME_SEC, SR  # noqa: E402

WORK = BT.SCRATCH / "completeness" / "turnbench"
CKPT = ROOT / "runs" / "stage1_completeness.afm"


def stage_cmp(a, ds, work):
    import torch
    from audioforge.data import ToneLanguage
    from audioforge.train import load_model
    model = load_model(str(a.ckpt), a.device)
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "completeness")
    head = model.heads[name]
    items = [f"{cid}_ch{k}" for cid in ds.meetings for k in (1, 2)]
    stat = {"sec": 0.0, "audio": 0.0}

    def fn(item, out):
        cid, k = item.rsplit("_ch", 1)
        x_all = np.asarray(ds.channels16k(cid), np.float32)[int(k) - 1]
        Tc = ToneLanguage.n_frames(len(x_all))
        scores = np.zeros(Tc, np.float32)
        with torch.no_grad():
            for s0, s1, keep in TB.segments(len(x_all)):
                x = torch.as_tensor(x_all[s0:s1])[None].to(a.device)
                t0 = time.perf_counter()
                enc, elen, hidden = model.encode(x, torch.tensor([x.shape[1]], device=a.device), return_hidden=True)
                z = head.decode(model.head_input(name, enc, hidden), elen)[0].float().cpu().numpy()
                stat["sec"] += time.perf_counter() - t0
                stat["audio"] += (s1 - s0) / SR
                f0 = int(round(s0 / SR / FRAME_SEC))
                fk = int(round(keep / SR / FRAME_SEC))
                n = min(len(z) - fk, Tc - (f0 + fk))
                if n > 0:
                    scores[f0 + fk: f0 + fk + n] = z[fk: fk + n]
        np.save(out.with_suffix(".tmp.npy"), scores)
        out.with_suffix(".tmp.npy").replace(out)
    left = TB.budget_loop("cmp", items, fn, work, a.budget, ".npy")
    BT.add_timing(work, "cmp_segment", 0, stat["sec"], audio_sec=stat["audio"], device=a.device)
    return left


class CmpSystems(TB.Systems):
    def __init__(self, ds, work: Path):
        super().__init__(ds, work)
        self.cmp = {(c, k): np.load(work / "cmp" / f"{c}_ch{k}.npy") for c in self.cids for k in (1, 2)
                    if (work / "cmp" / f"{c}_ch{k}.npy").exists()}
        self.have_cmp = len(self.cmp) == 2 * len(self.cids)

    def events(self, system: str, c: str, k: int, pt) -> list[float]:
        if system == "cmp":
            s = self.cmp[(c, k)]
            return TB.edge_events(s, self.emit_frames(len(s)), pt)
        if system == "cmp+silero_timeout":
            th, ks = pt
            return TB.merge_events(self.events("cmp", c, k, th), self.events("silero_timeout", c, k, ks))
        return super().events(system, c, k, pt)


def stage_sweep(a, ds, work) -> dict:
    from turnbench.data import resolve_dataset
    t0 = time.time()
    dataset = resolve_dataset(TB.D.TB_REPO, skip_audio=True)
    S = CmpSystems(ds, work)
    assert S.have_cmp, "run --stage cmp to completion first"
    vals = np.concatenate(list(S.cmp.values()))
    cq = [float(x) for x in np.unique(np.quantile(vals, TB.Q_GRID))]
    hq = [float(x) for x in np.unique(np.quantile(vals, TB.HYB_Q))]
    grids = {"silero_timeout": list(TB.K_GRID), "cmp": cq, "cmp+silero_timeout": [(th, ks) for th in hq for ks in TB.HYB_K]}
    out = {"scorer": "integrations/turnbench_scorer (SesameAILabs/turnbench, MIT), EOT task", "fp_budget": TB.FP_BUDGET,
           "n_conversations": len(S.cids), "ckpt": str(a.ckpt),
           "commit_rule": "cmp: rising edge above theta, 2 s refractory, committed at the 160 ms chunk end; timeouts: "
                          "end of the window completing k s of silence", "systems": {}, "published_dev_rescored": {}}
    (work / "preds").mkdir(exist_ok=True)
    for name, grid in grids.items():
        curve = []
        for pt in grid:
            curve.append({"point": pt, "score": TB.score_sub(S.submission(name, pt), dataset)})
        best = TB.pick(curve)
        r = {"curve": curve, "operating_point": best}
        if best is not None:
            p = work / "preds" / f"{name}.json"
            p.write_text(S.submission(name, best["point"]).model_dump_json(indent=1))
            r["predictions"] = str(p)
        out["systems"][name] = r
        b = best["score"] if best else None
        print(f"  {name:20s} " + (f"recall {b['recall']:.3f} fp {b['fp_rate']:.3f} p50 {b['latency_ms']['p50']} ms at {best['point']}"
                                  if b else "no point within the FP budget") + f" ({time.time() - t0:.0f}s)", flush=True)
    from turnbench.submission import load_submission
    for b in ("smart_turn_v3", "vap", "rms_vad", "kyutai_semantic_vad"):
        p = ROOT / "integrations" / "turnbench_scorer" / "baselines_dev" / f"{b}.json"
        if p.exists():
            out["published_dev_rescored"][b] = TB.score_sub(load_submission(p), dataset)
    out["timing"] = json.loads((work / "timing.json").read_text()) if (work / "timing.json").exists() else {}
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["vad", "cmp", "sweep"])
    p.add_argument("--work", default=str(WORK))
    p.add_argument("--budget", type=float, default=500.0)
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "runs" / "turnbench_dev_completeness.json"))
    a = p.parse_args()
    BT._torch2()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ds = TB.ds_all()
    if a.stage == "vad":
        print(TB.stage_vad(a, ds, work))
    elif a.stage == "cmp":
        print(stage_cmp(a, ds, work))
    else:
        res = stage_sweep(a, ds, work)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
