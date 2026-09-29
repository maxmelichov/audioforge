"""Held-out confirmation of the turn-detection rules on ICSI (research/BASELINES.md, "Turn detection: ICSI held-out
confirmation").

research/BASELINES.md picked "our trail6 head OR the Silero VAD timeout" after seeing the AMI dev table. This script
rebuilds eot-bench v2 (scripts/research/eval_stage1.py --bench v2, research/EOT_BENCH_V2.md) on the ICSI meetings that no
fitting ever touched (the Kaldi dev + eval meetings: Bmr021 Bns001 + Bmr013 Bmr018 Bro021; 5 of the 17 meetings with
audio, the other 12 are the train subset) with the same window / horizon rules (block C: 6 s-extended windows,
horizons 25 / 75 post-end emission frames; block A: the default 2 s-trail windows; causal_dominant enrollment on the
1.04 s streaming Sortformer v2 tracks; floor-open / taken strata; untimed-zone and > 4-speaker windows dropped as
icsi.recipe_data drops them) and scores two readings of every rule:

  frozen    every operating point (θ, k_frames, timeout frames) FITTED ON ALL 974 AMI DEV TURNS (the in-sample
            <= 5 % per-turn FC point of the same grids the AMI table used) and applied to ICSI unchanged, plus the
            AMI table's two per-fold points as a sensitivity range; paired bootstraps vs our hybrid on ICSI
  crossfit  the AMI protocol re-run on ICSI (operating point chosen on one leave-meetings-out half, applied to the
            other; scripts/bench_turn_baselines.score_block with ICSI folds)

Systems: Sortformer-primary timeout, Silero v5 timeout (Pipecat VAD state machine), any-speaker Sortformer timeout,
our trail6 head, hybrid trail6 (head OR primary timeout), head OR Silero timeout, and the oracle timeouts (labels).

Stages (each one process <= 10 min, resumable; CPU, 2 threads):
  tracks   streaming Sortformer tracks of the extended windows (eval_stage1.v2_tracks: real-audio right padding,
           1.04 s low-latency config; the same code / flags as the AMI trail6 dev tracks), floor-open turns first
  vad      Silero VAD v5 probabilities (32 ms) of every extended window
  scores   trail6 head scores (runs/stage1_turn_v3_trail6.afm) with causal_dominant binding, 6 s and 2 s windows
  report   -> runs/baselines_turn_icsi.json

  W=<scratch>/icsi_turn/work
  TMPDIR=<scratch> .venv/bin/python scripts/research/bench_turn_icsi.py --stage tracks --work $W   # repeat until "0 left"
  .venv/bin/python scripts/research/bench_turn_icsi.py --stage vad --work $W
  TMPDIR=<scratch> .venv/bin/python scripts/research/bench_turn_icsi.py --stage scores --work $W  # repeat until "0 left"
  .venv/bin/python scripts/research/bench_turn_icsi.py --stage report --work $W --out runs/baselines_turn_icsi.json
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

import bench_turn_baselines as BT  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402

HEAD_CKPT = ROOT / "runs" / "stage1_turn_v3_trail6.afm"
HEAD_TAG = "trail6"
AMI_SILERO = BT.SCRATCH / "baselines_turn" / "work" / "silero"  # AMI dev Silero probabilities (bench_turn_baselines --stage vad)
ICSI_MEETINGS = ("Bmr021", "Bns001", "Bmr013", "Bmr018", "Bro021")  # icsi.SPLITS dev + eval
ICSI_FOLDS = (("Bmr021", "Bmr018"), ("Bns001", "Bmr013", "Bro021"))  # leave-meetings-out halves (619 / 693 turns)
SUBSETS = {"dev": ("Bmr021", "Bns001"), "eval": ("Bmr013", "Bmr018", "Bro021")}
REF = BT.REF  # hybrid_stream_causal_dominant
CANDIDATE = "head_trail6+silero_timeout"
C_SF, R_SF = BT.C_SF, BT.R_SF
MAX_FC = 0.05


def key(ex) -> str:
    return BT.key(ex)


# --------------------------------------------------------------------------- data
def icsi_data(meetings=ICSI_MEETINGS, trail_sec: float = 6.0):
    """(base, ext, meta, ds, meta_base) as eval_stage1.v2_data, on ICSI: the library's default dev-style turn windows
    (20 s, 2 s trail; ICSI.turn_examples drops untimed-zone / > 4-speaker windows) and the same turns re-cut with a
    ``trail_sec`` trail from the same start. An extended window that reaches into an untimed zone (unlabelled speech
    in its trail) or holds > 4 speakers is dropped too, so every scored frame is labelled."""
    from audioforge.datasets import ext_tracks as xt
    from audioforge.datasets.icsi import ICSI
    ds = ICSI(list(meetings), verbose=False)
    kept = ds.turn_examples(20.0)
    chk, meta_b = E.turn_windows(ds, 2.0)  # the unfiltered re-cut: keys / labels reproduce the library's windows
    pos = {xt.example_key(v): i for i, v in enumerate(chk)}
    idx = [pos[xt.example_key(v)] for v in kept]
    for i, v in zip(idx, kept):
        assert np.array_equal(chk[i]["spk_targets"], v["spk_targets"]) and np.array_equal(chk[i]["hes"], v["hes"])
    ext_all, meta_all = E.turn_windows(ds, trail_sec, starts=[v["start"] for v in chk])
    idx = [i for i in idx if ds._keep(ext_all[i]) is None]
    base, ext = [chk[i] for i in idx], [ext_all[i] for i in idx]
    for u, v in zip(ext, base):
        assert u["onset_frame"] == v["onset_frame"] and u["turn_end_frame"] == v["turn_end_frame"]
    return base, ext, [meta_all[i] for i in idx], ds, [meta_b[i] for i in idx]


def drop_audio(convs):
    for v in convs:
        v.pop("audio", None)


# --------------------------------------------------------------------------- stages
def stage_tracks(a, ext, ds, work):
    from audioforge.conversation import floor_stratum
    op = [floor_stratum(v["spk_targets"], v["turn_end_frame"], E.V2_FLOOR_HORIZON) == "open" for v in ext]
    order = [i for i in range(len(ext)) if op[i]] + [i for i in range(len(ext)) if not op[i]]
    return E.v2_tracks(ext, ds, work, a.budget, order=order)


def stage_vad(a, ext, work):
    vad = B.SileroVAD(BT.SILERO_V5)

    def fn(ex, out):
        p = vad.probs(np.asarray(ex["audio"], np.float32))
        tmp = out.with_name(out.stem + ".tmp.npy")
        np.save(tmp, p)
        tmp.replace(out)
    return BT.run_resumable("silero", ext, fn, work, a.budget, ".npy")


def stage_scores(a, base, ext, work):
    from audioforge.train import load_model
    model = load_model(str(HEAD_CKPT), "cpu")
    assert E.chunk_of(model) == 2, E.chunk_of(model)
    return E.v2_scores(model, base, ext, work, ["causal_dominant", "causal_dominant@2s"], a.budget, a.batch_size,
                       HEAD_TAG, tracks_dir=work / "tracks")


# --------------------------------------------------------------------------- score tracks
def silero_timeout_track(probs: np.ndarray, T: int) -> np.ndarray:
    """The Silero timeout score (frames of silence since the last Pipecat-VAD speech chunk), as bench_turn_baselines
    builds it."""
    return B.silence_frames(B.speech_chunks_pipecat(B.pipecat_vad(probs)), T)


def systems_for(convs, tracks, head, silero, two: bool):
    """{name: (scores, emit)} singles and {name: (scores_a, emit_a, scores_b, emit_b)} hybrids of one block, built
    exactly as bench_turn_baselines.our_systems / build_tracks (stream emission for Sortformer rows, the head's
    160 ms chunk, t + 1 for frame-level rows)."""
    T = [len(v["spk_act"]) for v in convs]
    tracks = [np.stack([E._fit(p[:, j], t) for j in range(p.shape[1])], 1) for p, t in zip(tracks, T)]
    acts = [E._v2_cols(v, p, "causal_dominant")[0] for v, p in zip(convs, tracks)]
    es, eh = E._stream_emit(C_SF, R_SF), E._stream_emit(C_SF, R_SF, 2)
    ef = BT.EMIT["frame"]
    to = [E.silence_scores(x, E.arm_frame(x)) for x in acts]
    anys = [E.silence_scores(p.max(1), E.arm_frame(p.max(1))) for p in tracks]
    sil = [silero_timeout_track(p, t)[:t] for p, t in zip(silero, T)]
    on = [v["onset_frame"] for v in convs]
    singles = {"timeout_stream_causal_dominant": (to, es), "head_v3_stream_causal_dominant": (head, eh),
               "timeout_any_speaker_stream": (anys, es), "silero_timeout": (sil, ef),
               "timeout_primary_oracle": ([E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)], ef),
               "timeout_any_speaker_oracle": ([E.silence_scores(np.asarray(v["spk_targets"]).max(1), o)
                                               for v, o in zip(convs, on)], ef)}
    hybrids = {REF: (head, eh, to, es), CANDIDATE: (head, eh, sil, ef)}
    return singles, hybrids


def ami_systems():
    """The AMI dev blocks (C: extended windows, A: default windows) with the stored trail6 tracks / head scores and
    the Silero probabilities of bench_turn_baselines: {block: (convs, meta, singles, hybrids, horizons)}."""
    base, ext, meta, ds, meta_base = E.v2_data(6.0)
    keys = [key(v) for v in ext]  # one key per turn (the 2 s window is a prefix of the 6 s one)
    sil = [np.load(AMI_SILERO / f"{k}.npy") for k in keys]
    out = {}
    for blk, convs, mt, hz, two in (("C_extended_windows_stream", ext, meta, {"2s": 25, "6s": 75}, False),
                                    ("A_default_windows_all", base, meta_base, {"2s": None}, True)):
        tracks = [E.v2_base_track(v) for v in convs] if two else [np.load(E.v2_track_path(BT.TRAIL6_WORK, v)) for v in convs]
        sub = "causal_dominant" + ("@2s" if two else "")
        head = [np.load(E.v2_score_path(BT.TRAIL6_WORK, sub, v, HEAD_TAG)) for v in convs]
        out[blk] = (convs, mt) + systems_for(convs, tracks, head, sil, two) + (hz,)
    drop_audio(base)
    drop_audio(ext)
    del ds
    return out


def icsi_systems(work: Path):
    base, ext, meta, ds, meta_base = icsi_data()
    keys = [key(v) for v in ext]
    sil = [np.load(work / "silero" / f"{k}.npy") for k in keys]
    tr = [np.load(E.v2_track_path(work, v)) for v in ext]
    out = {}
    for blk, convs, mt, hz, two in (("C_extended_windows_stream", ext, meta, {"2s": 25, "6s": 75}, False),
                                    ("A_default_windows_all", base, meta_base, {"2s": None}, True)):
        tracks = [p[: len(v["spk_act"])] for p, v in zip(tr, convs)]  # block A: the same stream, cropped (flush fix)
        sub = "causal_dominant" + ("@2s" if two else "")
        head = [np.load(E.v2_score_path(work, sub, v, HEAD_TAG)) for v in convs]
        out[blk] = (convs, mt) + systems_for(convs, tracks, head, sil, two) + (hz,)
    drop_audio(base)
    drop_audio(ext)
    del ds
    return out


# --------------------------------------------------------------------------- operating points
def _prep(convs):
    from audioforge.conversation import pause_runs
    on = np.array([v["onset_frame"] for v in convs])
    en = np.array([v["turn_end_frame"] for v in convs])
    return on, en, [pause_runs(v["hes"]) for v in convs]


def sweep_outcomes(spec, hybrid: bool, on, en, pauses, L):
    """eot_outcomes over the grid bench_turn_baselines.score_block sweeps (all score values, at most 3000 quantiles;
    hybrids: v2_hybrid_grid x v2_hybrid_grid). Returns (oc, ths) with ths[j] the threshold (pair) of column j."""
    from audioforge.conversation import eot_outcomes, eot_outcomes_or
    if hybrid:
        ha, ea, tb, eb = spec
        za, zb = E._emit_transform(ha, en, ea), E._emit_transform(tb, en, eb)
        ga, gb = E.v2_hybrid_grid(za, on, en, pauses), E.v2_hybrid_grid(zb, on, en, pauses)
        oc = eot_outcomes_or(za, zb, on, en, ga, gb, post_end_frames=L, pauses=pauses)
        ths = [(float(x), float(y)) for x, y in oc["ths"]]
        return dict(oc, ths=np.arange(len(ths), dtype=np.float64)), ths
    sc, emit = spec
    z = E._emit_transform(sc, en, emit)
    allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in z]))
    ths = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
    oc = eot_outcomes(z, on, en, ths, post_end_frames=L, pauses=pauses)
    return oc, [float(x) for x in oc["ths"]]


def fit_points(spec, hybrid: bool, convs, L, folds) -> dict:
    """Operating points at <= MAX_FC per-turn FC (eot_bench's rule; ties to the lower miss on a joint grid):
    'all' = chosen on every turn (in-sample), 'fold<f>' = chosen on the turns NOT in fold f (the cross-fit's
    per-fold point, as the AMI table reports it)."""
    on, en, pauses = _prep(convs)
    oc, ths = sweep_outcomes(spec, hybrid, on, en, pauses, L)
    pts = {"all": ths[E._select(oc, np.ones(len(convs), bool), MAX_FC, "turn", tie_miss=hybrid)]}
    for f in np.unique(folds):
        pts[f"fold{int(f)}"] = ths[E._select(oc, folds != f, MAX_FC, "turn", tie_miss=hybrid)]
    return pts


def outcomes_at(spec, hybrid: bool, on, en, pauses, L, th):
    """(fc, lat, pf, npause) of a system at ONE fixed operating point (θ, or (θ_a, θ_b) for a hybrid)."""
    from audioforge.conversation import eot_outcomes, eot_outcomes_or
    if hybrid:
        ha, ea, tb, eb = spec
        oc = eot_outcomes_or(E._emit_transform(ha, en, ea), E._emit_transform(tb, en, eb), on, en, [th[0]], [th[1]],
                             post_end_frames=L, pauses=pauses)
    else:
        sc, emit = spec
        oc = eot_outcomes(E._emit_transform(sc, en, emit), on, en, [th], post_end_frames=L, pauses=pauses)
    return oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0], oc["npause"]


def fmt_point(name, th):
    if name in (REF, CANDIDATE):
        a, b = th
        return {"theta": None if not np.isfinite(a) else float(a),
                "timeout_threshold": None if not np.isfinite(b) else float(b),
                "k_frames": None if not np.isfinite(b) else int(np.floor(b)) + 1}
    if name.startswith("head"):
        return {"theta": float(th)}
    return {"timeout_threshold": float(th), "k_frames": int(np.floor(th)) + 1,
            "k_ms": round((int(np.floor(th)) + 1) * 80.0, 1)}


def point_metrics(fc, lat, pf, npause, groups, n_boot, subsets=None):
    from audioforge.conversation import bootstrap_ci, outcome_metrics
    pt = outcome_metrics(fc, lat, pf, npause)
    pt.update(bootstrap_ci(fc, lat, n_boot, 0))
    pt["strata"] = {g: {**outcome_metrics(fc[m], lat[m], pf[m], npause[m]), **bootstrap_ci(fc[m], lat[m], n_boot, 0)}
                    for g, m in groups.items()}
    if subsets:
        pt["subsets"] = {}
        for s, m in subsets.items():
            d = outcome_metrics(fc[m], lat[m], pf[m], npause[m])
            d["strata"] = {g: outcome_metrics(fc[m & gm], lat[m & gm], pf[m & gm], npause[m & gm])
                           for g, gm in groups.items()}
            pt["subsets"][s] = d
    return pt


def frozen_block(ami, icsi, n_boot: int, pairs) -> dict:
    """Every system at the AMI-fitted points ('all' = the frozen rule; the two AMI fold points as a range), scored on
    ICSI, with paired bootstraps (same resamples) vs the hybrid at the 'all' points."""
    from audioforge.conversation import floor_stratum
    convs_a, _, sing_a, hyb_a, hz = ami
    convs_i, _, sing_i, hyb_i, _ = icsi
    folds_a = np.array([0 if v["meeting"] in E.V2_DEV_FOLDS[0] else 1 for v in convs_a])
    on, en, pauses = _prep(convs_i)
    n = len(convs_i)
    strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs_i, en)])
    groups = {"open": strata == "open", "taken": strata != "open"}
    meet = np.array([v["meeting"] for v in convs_i])
    subsets = {s: np.isin(meet, ms) for s, ms in SUBSETS.items()}
    res = {"systems": {}, "paired": {}}
    outs = {}
    for name in list(sing_a) + list(hyb_a):
        hybrid = name in hyb_a
        spec_a = hyb_a[name] if hybrid else sing_a[name]
        spec_i = hyb_i[name] if hybrid else sing_i[name]
        r = {}
        for h, L in hz.items():
            pts = fit_points(spec_a, hybrid, convs_a, L, folds_a)
            r[h] = {"ami_points": {k: fmt_point(name, th) for k, th in pts.items()}}
            for k, th in pts.items():
                fc, lat, pf, npause = outcomes_at(spec_i, hybrid, on, en, pauses, L, th)
                r[h][f"icsi_at_ami_{k}"] = point_metrics(fc, lat, pf, npause, groups, n_boot if k == "all" else 0,
                                                         subsets if k == "all" else None)
                if k == "all":
                    outs[(name, h)] = (fc, lat)
        res["systems"][name] = r
        print(f"    frozen {name}: " + ", ".join(f"{h} miss {r[h]['icsi_at_ami_all']['miss_rate']:.3f} "
                                                 f"fc {r[h]['icsi_at_ami_all']['fc_rate']:.3f}" for h in hz), flush=True)
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
            res["paired"][f"{a_} - {b_} | {h}"] = d
    return res


def stage_report(a, work: Path) -> dict:
    t0 = time.time()
    print("  loading AMI dev (frozen operating points)", flush=True)
    ami = ami_systems()
    print(f"  loading ICSI ({time.time() - t0:.0f}s)", flush=True)
    icsi = icsi_systems(work)
    convs = icsi["C_extended_windows_stream"][0]
    out = {"protocol": "eot-bench v2 (research/EOT_BENCH_V2.md) on held-out ICSI meetings; frozen = operating points "
                       "fitted on all 974 AMI dev turns (in-sample, <= 5 % per-turn FC), crossfit = the AMI "
                       "leave-meetings-out protocol re-run on ICSI",
           "meetings": list(ICSI_MEETINGS), "folds": [list(f) for f in ICSI_FOLDS], "subsets": {k: list(v) for k, v in SUBSETS.items()},
           "n": len(convs), "n_per_meeting": {m: int(sum(v["meeting"] == m for v in convs)) for m in ICSI_MEETINGS},
           "head": str(HEAD_CKPT.relative_to(ROOT)), "n_boot": a.n_boot,
           "ami_folds_for_points": [list(f) for f in E.V2_DEV_FOLDS]}
    pairs = [(s, REF) for s in list(icsi["C_extended_windows_stream"][2]) + [CANDIDATE]] + \
            [(CANDIDATE, "silero_timeout"), (CANDIDATE, "head_v3_stream_causal_dominant")]
    for blk in ("C_extended_windows_stream", "A_default_windows_all"):
        print(f"  {blk}: frozen ({time.time() - t0:.0f}s)", flush=True)
        r = {"frozen_ami": frozen_block(ami[blk], icsi[blk], a.n_boot, pairs)}
        print(f"  {blk}: crossfit ({time.time() - t0:.0f}s)", flush=True)
        E.V2_DEV_FOLDS = ICSI_FOLDS
        cv, mt, sing, hyb, hz = icsi[blk]
        r["crossfit_icsi"] = BT.score_block(cv, mt, sing, hyb, hz, a.n_boot, pairs)
        E.V2_DEV_FOLDS = out["ami_folds_for_points"]
        out[blk] = r
    del ami
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["tracks", "vad", "scores", "report"])
    p.add_argument("--work", default=str(BT.SCRATCH / "icsi_turn" / "work"))
    p.add_argument("--budget", type=float, default=520.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(ROOT / "runs" / "baselines_turn_icsi.json"))
    a = p.parse_args()
    BT._torch2()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        res = stage_report(a, work)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")
        return
    t0 = time.time()
    base, ext, meta, ds, meta_base = icsi_data()
    print(f"  icsi data: {len(ext)} turns ({time.time() - t0:.0f}s)", flush=True)
    if a.stage == "tracks":
        print(stage_tracks(a, ext, ds, work))
    elif a.stage == "vad":
        print(stage_vad(a, ext, work))
    else:
        print(stage_scores(a, base, ext, work))


if __name__ == "__main__":
    main()
