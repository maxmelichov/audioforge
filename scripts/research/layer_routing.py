"""Layer routing for the diarization and turn heads (research/archive/LAYER_ROUTING.md). Train / evaluate / report stages.

  train      run a head-only recipe (GPU, one job on the machine), then assert that every tensor outside the trained
             prefixes (--trained, default the diar head + its mix) is bit-identical to the init checkpoint.
  diar_eval  diarization on the BASELINES.md windows: AMI dev 64 x 20 s diar windows (bench_sd_baselines.diar_windows,
             the exact code of the BASELINES diarization table: pooled frame DER with eval_stage1.der_parts + miss / FA /
             confusion, and pyannote.metrics DER at collar 0 / 0.25 / 0.5 total width) and the same on ICSI dev
             (64 x 20 s diar windows, seeded cap, never trained on) as the held-out corpus. Also the head's effective
             layer weights. CPU, 2 threads. --ckpt may be a ported Sortformer (runs/nemo_sortformer_v2.afm).
  report     markdown tables from --json.

Everything merges into --json (default runs/layer_routing.json) under ``diar[<tag>]`` / ``turn[<tag>]``.

  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 .venv/bin/python scripts/research/layer_routing.py train \\
      --recipe research/recipes/diar_layer_route.yaml --out <scratch>/diar_b4.afm --tag diar_b4 heads.diar.from_layers=[3]
  .venv/bin/python scripts/research/layer_routing.py diar_eval --ckpt <scratch>/diar_b4.afm --tag diar_b4
  .venv/bin/python scripts/research/layer_routing.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

DIAR_TRAINED = ("heads.diar.", "layer_mix.diar")


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _merge(path: Path, section: str, tag: str, part: str, rec: dict):
    res = _json(path)  # re-read: other stages may have merged meanwhile
    res.setdefault(section, {}).setdefault(tag, {})[part] = rec
    path.write_text(json.dumps(res, indent=1))


def frozen_check(init: str, out: str, trained: tuple[str, ...]) -> dict:
    """Every tensor of ``init`` outside ``trained`` prefixes is bit-identical in ``out`` (asserted)."""
    from audioforge.train import load_model
    src = load_model(init, "cpu").state_dict()
    sd = load_model(out, "cpu").state_dict()
    changed = [k for k in src if k in sd and not torch.equal(src[k], sd[k])]
    bad = [k for k in changed if not k.startswith(trained)]
    assert not bad, f"tensors outside {trained} changed: {bad[:5]}"
    return {"frozen_tensors_identical": sum(1 for k in src if k in sd and not k.startswith(trained)),
            "changed_tensors": len(changed), "dropped_from_init": [k for k in src if k not in sd],
            "new_in_out": [k for k in sd if k not in src]}


def layer_weights(model, head: str):
    lw = getattr(model, "layer_weights", None)
    w = lw(head) if lw is not None and head in model.head_cfg else None
    return None if w is None else [round(float(x), 4) for x in w]


def stage_train(a):
    from audioforge.train import load_model, load_recipe, run_recipe
    cfg = load_recipe(a.recipe, a.overrides)
    init = cfg["init"]["from"]
    trained = tuple(a.trained.split(","))
    t0 = time.time()
    _, metrics = run_recipe(a.recipe, a.overrides, out=a.out)
    sec = time.time() - t0
    chk = frozen_check(init, a.out, trained)
    head = a.head
    out = load_model(a.out, "cpu")
    rec = {"recipe": a.recipe, "overrides": a.overrides, "ckpt": a.out, "init": init, "sec": round(sec, 1),
           "max_steps": cfg["trainer"]["max_steps"], "final_eval": metrics, **chk,
           "head_cfg": out.head_cfg[head], "layer_weights": layer_weights(out, head)}
    _merge(Path(a.json), a.section, a.tag, "train", rec)
    print(f"[train] {a.tag}: {sec:.0f}s, {chk['frozen_tensors_identical']} frozen tensors identical, "
          f"{chk['changed_tensors']} changed (all under {trained}); weights {rec['layer_weights']}", flush=True)


# --------------------------------------------------------------------------- diarization
def icsi_diar_windows(n: int = 64) -> list[dict]:
    from audioforge.datasets.icsi import recipe_data
    from audioforge.train import derive_labels
    val = recipe_data({"data": {"icsi": {"mode": "diar", "val_split": "dev", "n_val": n, "seed": 0}}}, "val")
    return [derive_labels(dict(v), ["vad"]) for v in val]


def score_diar(pd: list[np.ndarray], val: list[dict]) -> dict:
    """bench_sd_baselines.stage_model's diarization scoring on precomputed probabilities (T, S) per window."""
    from bench_sd_baselines import _fit, pooled_der, pyannote_der
    refs = [np.asarray(v["spk_targets"], np.float32) for v in val]
    hard = [(_fit(p, len(r)) > 0.5).astype(np.float32) for p, r in zip(pd, refs)]
    from eval_stage1 import der_parts
    parts = [[round(x, 1) for x in der_parts(torch.as_tensor(h).float(), torch.as_tensor(r).float())]
             for h, r in zip(hard, refs)]  # per window (miss, fa, conf, speech) frames: paired bootstraps over windows
    return {"n": len(val), "per_window": parts, "frame_der": pooled_der(hard, refs), "pyannote": pyannote_der(hard, refs),
            "mean_ref_speakers": round(float(np.mean([(r.sum(0) > 0).sum() for r in refs])), 2),
            "mean_hyp_speakers": round(float(np.mean([(h.sum(0) > 0).sum() for h in hard])), 2)}


def stage_diar_eval(a):
    from bench_sd_baselines import _load_ours, diar_windows, model_pass
    from audioforge.heads.turn import _diar_name
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    model = _load_ours(a.ckpt)
    diar = _diar_name(model)
    rec = {"ckpt": a.ckpt, "head": diar, "head_cfg": model.head_cfg.get(diar),
           "layer_weights": layer_weights(model, diar),
           "head_params": int(sum(p.numel() for p in model.heads[diar].parameters()))}
    for corpus, fn in (("ami", diar_windows), ("icsi", icsi_diar_windows)):
        val = fn(a.n)
        t0 = time.time()
        pd, _ = model_pass(model, val)
        r = score_diar(pd, val)
        r["sec"] = round(time.time() - t0, 1)
        rec[corpus] = r
        print(a.tag, corpus, r["frame_der"], {k: v["der"] for k, v in r["pyannote"].items()}, flush=True)
    _merge(Path(a.json), "diar", a.tag, "eval", rec)


# --------------------------------------------------------------------------- turn: eot-bench v2 paired vs trail6
def stage_turn_pair(a):
    """eot-bench v2 (research/archive/EOT_BENCH_V2.md §7 protocol) on stored head scores (eval_stage1 --bench v2 --v2-stage
    scores --v2-tag <tag>): causal_dominant binding, block A (default windows, 2 s) and block C (extended windows, 2 s /
    6 s emission horizons), threshold cross-fitted at <= 5 % per-turn FC (2-fold leave-meetings-out); miss, held-out FC
    and P50 per system, and paired bootstraps (tag minus --base) over all / floor-open / taken turns (the §7 pair code)."""
    import eval_stage1 as ev
    from audioforge.conversation import _q, floor_stratum, pause_runs
    torch.set_num_threads(a.threads)
    W, tags = Path(a.work), [a.base] + [t for t in a.tags.split(",") if t]
    base, ext, meta, _, meta_base = ev.v2_data(6.0)
    C, R = ev.SORTFORMER_LOW_LATENCY["chunk_len"], ev.SORTFORMER_LOW_LATENCY["chunk_right_context"]
    emit = ev._stream_emit(C, R, 2)
    out = {"protocol": "eot-bench v2, causal_dominant, cross-fitted <= 5 % per-turn FC", "base": a.base, "n_boot": a.n_boot,
           "systems": {}, "paired": {}}
    for block, convs, mt, sfx, hz in (("A", base, meta_base, "@2s", {"2s": None}), ("C", ext, meta, "", {"2s": 25, "6s": 75})):
        n = len(convs)
        on = np.array([v["onset_frame"] for v in convs]); en = np.array([v["turn_end_frame"] for v in convs])
        strata = np.array([floor_stratum(v["spk_targets"], int(e), ev.V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
        groups = {"open": strata == "open", "taken": strata != "open"}
        pauses = [pause_runs(v["hes"]) for v in convs]
        folds = np.array([0 if v["meeting"] in ev.V2_DEV_FOLDS[0] else 1 for v in convs])
        avail = np.array([m["post_avail"] for m in mt]); reason = np.array([m["end_reason"] for m in mt])
        idx = np.random.default_rng(0).integers(0, n, (a.n_boot, n))
        o = {}
        for tag in tags:
            sc = [np.load(ev.v2_score_path(W, "causal_dominant" + sfx, v, tag)) for v in convs]
            for h, L in hz.items():
                cens = ((avail < L) if L is not None else np.ones(n, bool)) & (reason != "resume")
                b = ev.eot_bench_emit(sc, on, en, emit, pauses=pauses, post_end_frames=L, censored=cens, groups=groups,
                                      n_boot=0, return_outcomes=True)
                fc, lat, pf, th = ev.v2_crossfit(b["_outcomes"], folds, 0.05, "turn")
                o[(tag, h)] = (fc, lat)
                rec = {"thresholds": th}
                for g, m in (("all", np.ones(n, bool)), *groups.items()):
                    k = lat[m][~fc[m]]
                    rec[g] = {"n": int(m.sum()), "miss": round(float(np.isinf(k).mean()), 4), "fc": round(float(fc[m].mean()), 4),
                              "p50_ms": _q(k, 0.5) if np.isfinite(_q(k, 0.5)) else None,
                              "p50_fired_ms": float(np.median(k[np.isfinite(k)])) if np.isfinite(k).any() else None}
                out["systems"][f"{block} | {tag} | {h}"] = rec
                print(block, tag, h, {g: (rec[g]["miss"], rec[g]["fc"], rec[g]["p50_fired_ms"]) for g in ("all", "open", "taken")},
                      flush=True)
        for tag in tags[1:]:
            for h in hz:
                A, B = o[(tag, h)], o[(a.base, h)]
                d = {"all": ev._paired(A, B, idx)}
                for g in ("open", "taken"):
                    sub = np.nonzero(groups[g])[0]
                    gi = np.random.default_rng(1).integers(0, len(sub), (a.n_boot, len(sub)))
                    d[g] = ev._paired((A[0][sub], A[1][sub]), (B[0][sub], B[1][sub]), gi)
                out["paired"][f"{block} | {tag} - {a.base} | {h}"] = d
                print(block, tag, "-", a.base, h, {g: (d[g]["miss_diff"], d[g]["miss_diff_ci"]) for g in d}, flush=True)
    res = _json(Path(a.json))
    res.setdefault("turn", {})["eot_bench_v2" + ("" if a.base == "trail6" else f"_vs_{a.base}")] = json.loads(json.dumps(out, default=str))
    Path(a.json).write_text(json.dumps(res, indent=1))


def paired_der(a: list, b: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """Pooled-DER difference a - b over the same windows (per-window (miss, fa, conf, speech)), window bootstrap CI."""
    A, B = np.asarray(a, float), np.asarray(b, float)
    der = lambda X, i: X[i, :3].sum() / max(1.0, X[i, 3].sum())  # noqa: E731
    idx = np.random.default_rng(seed).integers(0, len(A), (n_boot, len(A)))
    d = np.array([der(A, i) - der(B, i) for i in idx])
    full = np.arange(len(A))
    return {"diff": round(float(der(A, full) - der(B, full)), 4),
            "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]}


def best_at_fp(curve: list, fp: float):
    """TurnBench's rule on a stored curve: the highest-recall point with fp_rate <= fp (None if none)."""
    ok = [c for c in curve if c["score"]["fp_rate"] <= fp]
    return max(ok, key=lambda c: (c["score"]["recall"], -c["score"]["fp_rate"])) if ok else None


def stage_turnbench(a):
    """Merge bench_turnbench.py --stage sweep --tag outputs (head alone + head OR Silero per channel) into --json: the
    dev operating point (max recall at fp <= 0.10) and, for 'confident earlier', the same rule at fp <= 0.05 / 0.15."""
    rec = {}
    for spec in a.tags.split(","):
        tag, path = spec.split("=", 1)
        d = _json(Path(path))
        r = {"file": path}
        for sysname, v in d["systems"].items():
            kind = "head" if sysname.startswith("head_") else "hybrid"
            r[kind] = {"dev_op": v["operating_point"]}
            for fp in (0.05, 0.10, 0.15):
                b = best_at_fp(v["curve"], fp)
                r[kind][f"fp<={fp}"] = None if b is None else {"point": b["point"], "recall": b["score"]["recall"],
                                                               "fp_rate": b["score"]["fp_rate"],
                                                               "p50_ms": b["score"]["latency_ms"]["p50"],
                                                               "p10_ms": b["score"]["latency_ms"]["p10"],
                                                               "p90_ms": b["score"]["latency_ms"]["p90"]}
            if kind == "head":  # recall / P50 linearly interpolated at matched fp along the head's threshold curve
                c = sorted((x["score"]["fp_rate"], x["score"]["recall"], x["score"]["latency_ms"]["p50"]) for x in v["curve"])
                fps = np.array([x[0] for x in c])
                r[kind]["interp_matched_fp"] = {
                    str(f): {"recall": round(float(np.interp(f, fps, [x[1] for x in c])), 4),
                             "p50_ms": round(float(np.interp(f, fps, [x[2] for x in c]))),
                             "extrapolated": bool(f > fps.max() or f < fps.min())} for f in (0.03, 0.05, 0.07, 0.10)}
        rec[tag] = r
        for kind in ("head", "hybrid"):
            print(tag, kind, {k: (v["recall"], v["fp_rate"], v["p50_ms"]) if v else None for k, v in r[kind].items() if k.startswith("fp<=")})
    res = _json(Path(a.json))
    res.setdefault("turn", {})["turnbench_dev"] = rec
    Path(a.json).write_text(json.dumps(res, indent=1))


# --------------------------------------------------------------------------- report
def _blocks(w):
    if w is None:
        return "top (17)"
    nz = [i for i, x in enumerate(w) if x > 0]
    if len(nz) == 1:
        return f"block {nz[0] + 1}"
    return f"mix of blocks {nz[0] + 1}-{nz[-1] + 1}" if nz == list(range(nz[0], nz[-1] + 1)) else f"mix {nz}"


def stage_report(a):
    res = _json(Path(a.json))
    d = res.get("diar", {})
    print("| variant | input | AMI frame DER (miss / FA / conf) | AMI pyannote c0 / c0.25 / c0.5 | ICSI frame DER (miss / FA / conf) "
          "| ICSI pyannote c0 / c0.25 / c0.5 | hyp / ref spk per window (AMI) |")
    print("|---|---|---|---|---|---|---|")
    for tag, r in d.items():
        e = r.get("eval")
        if not e:
            continue
        cells = []
        for c in ("ami", "icsi"):
            f, p = e[c]["frame_der"], e[c]["pyannote"]
            cells.append(f"{f['der_pooled']:.3f} ({f['miss']:.3f} / {f['fa']:.3f} / {f['confusion']:.3f})")
            cells.append(" / ".join(f"{p[k]['der']:.3f}" for k in ("collar_0.0", "collar_0.25", "collar_0.5")))
        print(f"| {tag} | {_blocks(e['layer_weights'])} | {' | '.join(cells)} | "
              f"{e['ami']['mean_hyp_speakers']} / {e['ami']['mean_ref_speakers']} |")
    ctl = (d.get(a.control) or {}).get("eval")
    if ctl and "per_window" in ctl["ami"]:
        print(f"\npaired pooled-DER difference vs {a.control} (window bootstrap 95 % CI): AMI | ICSI")
        for tag, r in d.items():
            e = r.get("eval")
            if e and tag != a.control and "per_window" in e["ami"] and len(e["ami"]["per_window"]) == len(ctl["ami"]["per_window"]):
                pa, pi = (paired_der(e[c]["per_window"], ctl[c]["per_window"]) for c in ("ami", "icsi"))
                print(f"{tag}: {pa['diff']:+.3f} {pa['ci95']} | {pi['diff']:+.3f} {pi['ci95']}")
    print()
    for tag, r in d.items():
        w = (r.get("eval") or {}).get("layer_weights")
        if w and sum(x > 0 for x in w) > 1:
            print(f"{tag}: " + " ".join(f"{x:.3f}" for x in w))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("train", "diar_eval", "turn_pair", "turnbench", "report"))
    ap.add_argument("--recipe")
    ap.add_argument("--out")
    ap.add_argument("--ckpt", default="runs/stage1_served.afm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--section", default="diar", help="train: json section (diar | turn)")
    ap.add_argument("--head", default="diar", help="train: the head whose config / layer weights are recorded")
    ap.add_argument("--trained", default=",".join(DIAR_TRAINED), help="train: comma-separated trained prefixes")
    ap.add_argument("--json", default="runs/layer_routing.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--work", help="turn_pair: the eot-bench v2 work dir holding scores_<binding>__<tag>/")
    ap.add_argument("--base", default="trail6", help="turn_pair: reference score tag")
    ap.add_argument("--tags", default="", help="turn_pair: comma-separated score tags compared with --base; "
                    "turnbench: comma-separated tag=<sweep json>")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--control", default="diar_all17", help="report: the diar control for paired differences")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    a.tag = a.tag or Path(a.out or a.ckpt).stem
    {"train": stage_train, "diar_eval": stage_diar_eval, "turn_pair": stage_turn_pair, "turnbench": stage_turnbench, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
