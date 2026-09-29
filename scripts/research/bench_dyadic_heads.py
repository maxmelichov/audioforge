"""Scoring of the dyadic-trained turn heads (research/DYADIC.md section 8) against the AMI-trained trail6 head.

Heads (tags): trail6 (runs/stage1_turn_v3_trail6.afm, the control), dyadic (a), energy (b), mh (c), energy_mh (d).

Sets and protocols (all eot-bench v2, research/EOT_BENCH_V2.md sections 1-3):
  oto dev      the HUMAN party's non-backchannel turns of the 60 dev conversations of research/recipes/stage1_turn_dyadic.yaml
               (first 160 conversations of the slice, positions i % 8 in {0, 1, 2}; never trained on), 20 s windows
               with a 2 s trail (block A) and the same starts with a 6 s trail (block C). Input = the two-channel
               product's: mixed mono + spk_act = the human's Silero activity, cols = [human, agent, 0, 0]. Systems per
               head: head_<tag> (160 ms emission) and hybrid_<tag> = head OR the primary-activity timeout; plus the
               timeout itself. Operating points: (i) cross-fitted <= 5 % per-turn FC over conversation halves
               (even / odd positions), 1000 bootstraps, paired differences vs trail6 on the same ends; (ii) frozen AMI
               = the point fitted on all 974 AMI dev turns with the SAME head on the oracle party track, unchanged.
  AMI dev      974 turns, the deployable cascade: cached streaming Sortformer tracks (1.04 s) with causal_dominant
               enrollment (the trail6 work dir of EOT_BENCH_V2.md section 7), head_<tag> and hybrid_<tag> = head OR the
               bound-track timeout; paired vs trail6 (61.9 / 45.9 hybrid at 6 s is the reference row).
  TurnBench    scripts/research/bench_turnbench.py --tag <tag> (official scorer; head per channel = spk_act channel k's Silero).

Stages (each process <= 45 min; MPS for scoring when no training runs, CPU otherwise):
  scores  --tag T --what oto,ami,ami_stream [--device mps]    resumable within --budget
  check   --tag T      every tensor except heads.turn.* identical to trail6's (encoder, RNNT, ...); energy / mh new
  points  AMI-frozen operating points of every head (oracle track, 6 s): JSON for bench_turnbench --frozen
  report  -> runs/dyadic_train.json (+ printed tables)
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
import bench_turn_dyadic as BD  # noqa: E402
import bench_turn_icsi as BI  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.datasets import dyadic as D  # noqa: E402

HEADS = {"trail6": "runs/stage1_turn_v3_trail6.afm", "dyadic": "runs/stage1_turn_dyadic.afm",
         "energy": "runs/stage1_turn_dyadic_energy.afm", "mh": "runs/stage1_turn_dyadic_mh.afm",
         "energy_mh": "runs/stage1_turn_dyadic_energy_mh.afm"}
SPLIT = dict(n_conv=160, split_mod=8, dev_res=(0, 1, 2))
WORK = BT.SCRATCH / "dyt" / "work"
OTO_WORK = WORK / "oto"
AMI_ORACLE_WORK = BD.AMI_WORK  # trail6's AMI dev oracle-track scores live here already (scores_oracle_track__trail6)
AMI_STREAM_WORK = BT.SCRATCH / "trail6" / "work"  # tracks/ + scores_causal_dominant[_2s]__trail6
POINTS_JSON = WORK / "frozen_points.json"
OUT = ROOT / "runs" / "dyadic_train.json"
HZ_C, HZ_A = {"2s": 25, "6s": 75}, {"2s": None}


def dev_ids() -> list[str]:
    return D.split_ids(D.list_ids("oto")[: SPLIT["n_conv"]], "dev", SPLIT["split_mod"], SPLIT["dev_res"])


def oto_dev_data(trail_sec: float = 6.0):
    """(base, ext, meta, ds, meta_base) of the human turns of the dev conversations (bench_turn_dyadic.dyadic_data on
    an explicit id list; no turn cap)."""
    ds = D.Dyadic(dev_ids(), "oto", verbose=False, cache_dtype="float16")
    with ds.tag_primary() as names:
        base, meta_base = E.turn_windows(ds, 2.0)
    ds.resolve_primary(base, names)
    with ds.tag_primary() as names:
        ext, meta = E.turn_windows(ds, trail_sec, starts=[v["start"] for v in base])
    ds.resolve_primary(ext, names)
    idx = [i for i, v in enumerate(ext) if v["role"] == "human" and ds._keep(v) is None and ds._keep(base[i]) is None]
    return [base[i] for i in idx], [ext[i] for i in idx], [meta[i] for i in idx], ds, [meta_base[i] for i in idx]


def score_path(work: Path, tag: str, ex: dict) -> Path:
    return Path(work) / f"scores_oracle_track__{tag}" / f"{BT.key(ex)}.npy"


def load_head(tag: str, device: str):
    import torch
    from audioforge.train import load_model
    m = load_model(str(ROOT / HEADS[tag]), "cpu")
    return m.to(torch.device(device)).eval()


# --------------------------------------------------------------------------- scores
def oracle_scores(model, tag, convs, work, budget, bs) -> int:
    BD.HEAD_TAG = tag  # bench_turn_dyadic.oracle_score_path names the dir after it
    left = BD.head_oracle_scores(model, convs, work, budget, bs)
    BD.HEAD_TAG = "trail6"
    return left


def stage_scores(a):
    import torch
    t0 = time.time()
    model = load_head(a.tag, a.device)
    assert E.chunk_of(model) == BD.HEAD_CHUNK
    out = {}
    for what in a.what.split(","):
        rem = a.budget - (time.time() - t0)
        if rem <= 0:
            out[what] = "no budget"
            continue
        with torch.no_grad():
            if what == "oto":
                base, ext, meta, ds, mb = oto_dev_data()
                out[what] = oracle_scores(model, a.tag, ext, OTO_WORK, rem, a.batch_size)
            elif what == "ami":
                base, ext, meta, ds, mb = E.v2_data(6.0)
                out[what] = oracle_scores(model, a.tag, ext, AMI_ORACLE_WORK, rem, a.batch_size)
            elif what == "ami_stream":
                base, ext, meta, ds, mb = E.v2_data(6.0)
                out[what] = E.v2_scores(model, base, ext, AMI_STREAM_WORK, ["causal_dominant", "causal_dominant@2s"],
                                        rem, a.batch_size, a.tag)
        print(f"  {what}: {out[what]} ({time.time() - t0:.0f}s)", flush=True)
    return out


def stage_check(a) -> dict:
    """The dyadic runs train heads.turn only: every other tensor must equal trail6's bit for bit."""
    import torch
    from audioforge.train import load_model
    r = load_model(str(ROOT / HEADS["trail6"]), "cpu").state_dict()
    n = load_model(str(ROOT / HEADS[a.tag]), "cpu").state_dict()
    frozen = [k for k in r if not k.startswith("heads.turn.")]
    diff = [k for k in frozen if k not in n or not torch.equal(r[k], n[k])]
    turn_changed = [k for k in r if k.startswith("heads.turn.") and k in n and not torch.equal(r[k], n[k])]
    new = sorted(set(n) - set(r))
    res = {"tag": a.tag, "frozen_tensors": len(frozen), "frozen_changed": diff,
           "rnnt_tensors": sum(k.startswith("heads.rnnt.") for k in frozen),
           "turn_tensors_changed": len(turn_changed), "new_tensors": new}
    assert not diff, f"{a.tag}: frozen tensors changed: {diff[:5]}"
    print(json.dumps(res), flush=True)
    return res


# --------------------------------------------------------------------------- systems
def tags_with(work: Path, convs, tags) -> list[str]:
    return [t for t in tags if all(score_path(work, t, v).exists() for v in convs)]


def oto_systems(convs, ext, tags):
    T = [len(v["spk_act"]) for v in convs]
    on = [v["onset_frame"] for v in convs]
    to = [E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)]
    singles = {"timeout_primary_oracle": (to, BD.EMIT_FRAME)}
    hybrids = {}
    for t in tags:
        h = [np.load(score_path(OTO_WORK, t, u))[:n] for u, n in zip(ext, T)]
        singles[f"head_{t}"] = (h, BD.EMIT_HEAD)
        hybrids[f"hybrid_{t}"] = (h, BD.EMIT_HEAD, to, BD.EMIT_FRAME)
    return singles, hybrids


def ami_oracle_blocks(tags):
    """AMI dev (974 turns, oracle party track) singles / hybrids per head for the frozen points."""
    base, ext, meta, ds, mb = E.v2_data(6.0)
    tags = tags_with(AMI_ORACLE_WORK, ext, tags)
    out = {}
    for blk, convs, hz in (("C", ext, HZ_C), ("A", base, HZ_A)):
        T = [len(v["spk_act"]) for v in convs]
        on = [v["onset_frame"] for v in convs]
        to = [E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)]
        singles, hybrids = {"timeout_primary_oracle": (to, BD.EMIT_FRAME)}, {}
        for t in tags:
            h = [np.load(score_path(AMI_ORACLE_WORK, t, u))[:n] for u, n in zip(ext, T)]
            singles[f"head_{t}"] = (h, BD.EMIT_HEAD)
            hybrids[f"hybrid_{t}"] = (h, BD.EMIT_HEAD, to, BD.EMIT_FRAME)
            if all((BI.AMI_SILERO / f"{BT.key(u)}.npy").exists() for u in ext):
                sil = [BI.silero_timeout_track(np.load(BI.AMI_SILERO / f"{BT.key(u)}.npy"), n)[:n] for u, n in zip(ext, T)]
                hybrids[f"head_{t}+silero_timeout"] = (h, BD.EMIT_HEAD, sil, BD.EMIT_FRAME)
        out[blk] = (convs, None, singles, hybrids, hz)
    BI.drop_audio(base)
    BI.drop_audio(ext)
    return out, tags


def frozen_points(ami_blk) -> dict:
    """{system: {horizon: (theta...) }} fitted on all AMI dev turns (in-sample, <= 5 % per-turn FC)."""
    convs_a, _, sing_a, hyb_a, hz = ami_blk
    folds_a = np.array([0 if v["meeting"] in E.V2_DEV_FOLDS[0] else 1 for v in convs_a])
    pts = {}
    for name in list(sing_a) + list(hyb_a):
        hybrid = name in hyb_a
        spec = hyb_a[name] if hybrid else sing_a[name]
        pts[name] = {h: BI.fit_points(spec, hybrid, convs_a, L, folds_a)["all"] for h, L in hz.items()}
    return pts


def frozen_eval(pts, convs_d, sing_d, hyb_d, hz, n_boot, pairs) -> dict:
    """Every system at its frozen AMI point on the dyadic set + paired differences at those points."""
    from audioforge.conversation import floor_stratum
    on, en, pauses = BI._prep(convs_d)
    strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs_d, en)])
    groups = {"open": strata == "open", "taken": strata != "open"}
    res, outs = {"systems": {}, "paired": {}}, {}
    for name, ph in pts.items():
        hybrid = name in hyb_d
        if name not in (hyb_d if hybrid else sing_d):
            continue
        spec = hyb_d[name] if hybrid else sing_d[name]
        r = {}
        for h, L in hz.items():
            th = ph[h]
            fc, lat, pf, npause = BI.outcomes_at(spec, hybrid, on, en, pauses, L, th)
            th_l = [float(x) if np.isfinite(x) else None for x in np.atleast_1d(th)]
            r[h] = {"ami_point": th_l, "at_ami_point": BI.point_metrics(fc, lat, pf, npause, groups, n_boot)}
            outs[(name, h)] = (fc, lat)
        res["systems"][name] = r
    n = len(convs_d)
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


def pairs_for(tags, prefix=("head", "hybrid")):
    out = []
    for t in tags:
        if t == "trail6":
            continue
        for p in prefix:
            out.append((f"{p}_{t}", f"{p}_trail6"))
    for t in tags:
        out.append((f"hybrid_{t}", f"head_{t}"))
    return out


def ami_stream_block(tags, n_boot):
    """AMI dev deployable cascade (causal_dominant binding on the cached 1.04 s streaming tracks), per head."""
    base, ext, meta, ds, mb = E.v2_data(6.0)
    res, have = {}, []
    for t in tags:
        if all(E.v2_score_path(AMI_STREAM_WORK, "causal_dominant", v, t).exists() for v in ext) and \
                all(E.v2_score_path(AMI_STREAM_WORK, "causal_dominant@2s", v, t).exists() for v in base):
            have.append(t)
    es, eh = E._stream_emit(BI.C_SF, BI.R_SF), E._stream_emit(BI.C_SF, BI.R_SF, BD.HEAD_CHUNK)
    for blk, convs, mt, hz, two in (("C_extended_windows_stream", ext, meta, HZ_C, False),
                                    ("A_default_windows_all", base, mb, HZ_A, True)):
        T = [len(v["spk_act"]) for v in convs]
        tracks = [E.v2_base_track(v) if two else np.load(E.v2_track_path(AMI_STREAM_WORK, v)) for v in convs]
        tracks = [np.stack([E._fit(p[:, j], t) for j in range(p.shape[1])], 1) for p, t in zip(tracks, T)]
        acts = [E._v2_cols(v, p, "causal_dominant")[0] for v, p in zip(convs, tracks)]
        to = [E.silence_scores(x, E.arm_frame(x)) for x in acts]
        singles = {"timeout_stream_causal_dominant": (to, es)}
        hybrids = {}
        for t in have:
            sp = [E.v2_score_path(AMI_STREAM_WORK, "causal_dominant" + ("@2s" if two else ""), v, t) for v in convs]
            h = [np.load(q)[:n] for q, n in zip(sp, T)]
            singles[f"head_{t}"] = (h, eh)
            hybrids[f"hybrid_{t}"] = (h, eh, to, es)
        res[blk] = BT.score_block(convs, mt, singles, hybrids, hz, n_boot, pairs_for(have))
        print(f"  AMI stream {blk}: {len(have)} heads", flush=True)
    BI.drop_audio(base)
    BI.drop_audio(ext)
    return res, have


def stage_points(a):
    ami, tags = ami_oracle_blocks(list(HEADS))
    pts = frozen_points(ami["C"])
    j = {name: {h: [float(x) if np.isfinite(x) else None for x in np.atleast_1d(th)] for h, th in ph.items()}
         for name, ph in pts.items()}
    POINTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    POINTS_JSON.write_text(json.dumps(j, indent=1))
    print(json.dumps(j, indent=1))
    return j


def stage_report(a) -> dict:
    t0 = time.time()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    base, ext, meta, ds, mb = oto_dev_data()
    tags = tags_with(OTO_WORK, ext, list(HEADS))
    folds = BD.folds_of(ext)
    res["oto_dev"] = {"protocol": "eot-bench v2 on the oto dev conversations (human-party turns; oracle party track = "
                                  "the human's Silero activity, cols [human, agent, 0, 0]); cross-fitted <= 5 % per-turn "
                                  "FC over conversation halves; frozen_ami = the same head's point fitted on all 974 "
                                  "AMI dev turns (oracle party track) applied unchanged",
                      "conversations": len(set(v["meeting"] for v in ext)), "n": len(ext), "heads": tags,
                      "folds": [list(f) for f in folds], "n_boot": a.n_boot,
                      "end_reasons": {r: int(sum(m["end_reason"] == r for m in meta)) for r in ("resume", "trail", "meeting_end")}}
    saved = E.V2_DEV_FOLDS
    ami, ami_tags = ami_oracle_blocks(list(HEADS)) if a.frozen else (None, [])
    for blk, convs, mt, hz, key in (("C_extended_windows", ext, meta, HZ_C, "C"), ("A_default_windows", base, mb, HZ_A, "A")):
        E.V2_DEV_FOLDS = folds
        sing, hyb = oto_systems(convs, ext, tags)
        r = {"crossfit": BT.score_block(convs, mt, sing, hyb, hz, a.n_boot, pairs_for(tags))}
        E.V2_DEV_FOLDS = saved
        if ami is not None:
            pts = frozen_points(ami[key])
            r["frozen_ami"] = frozen_eval(pts, convs, sing, hyb, hz, a.n_boot, pairs_for(tags))
        res["oto_dev"][blk] = r
        print(f"  oto {blk} ({time.time() - t0:.0f}s)", flush=True)
    E.V2_DEV_FOLDS = saved
    if a.ami_stream:
        blk, have = ami_stream_block(list(HEADS), a.n_boot)
        res["ami_dev_stream"] = {"protocol": "eot-bench v2 AMI dev, 974 turns, cached 1.04 s streaming Sortformer tracks, "
                                             "causal_dominant enrollment, cross-fitted <= 5 % per-turn FC (2-fold "
                                             "leave-meetings-out), hybrid = head OR bound-track timeout",
                                 "heads": have, **blk}
    res["sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(tables(res))
    print(f"wrote {OUT}")
    return res


# --------------------------------------------------------------------------- tables
def _row(s6, s2=None):
    ci = s6["miss_rate_ci"]
    txt = (f"{100 * s6['fc_rate']:.1f} | {100 * s6['miss_rate']:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
           + (f"{100 * s2['miss_rate']:.1f} | " if s2 is not None else "")
           + f"{s6['p50_ms']:.0f} | {100 * s6['strata']['open']['miss_rate']:.1f} | "
           f"{100 * s6['strata']['taken']['miss_rate']:.1f}")
    return txt


def _pd(d):
    x = d["all"]
    return (f"{100 * x['miss_diff']:+.1f} [{100 * x['miss_diff_ci'][0]:+.1f}, {100 * x['miss_diff_ci'][1]:+.1f}] | "
            f"{100 * d['open']['miss_diff']:+.1f} [{100 * d['open']['miss_diff_ci'][0]:+.1f}, "
            f"{100 * d['open']['miss_diff_ci'][1]:+.1f}] | {100 * d['taken']['miss_diff']:+.1f} | "
            f"{x['p50_a'] - x['p50_b']:+.0f}")


def tables(res) -> str:
    L = []
    o = res.get("oto_dev")
    if o:
        c = o["C_extended_windows"]
        L.append(f"### oto dev: {o['n']} human turn ends, {o['conversations']} conversations\n")
        L.append("Cross-fitted <= 5 % per-turn FC:\n\n| system | FC % | miss 6 s % [CI] | miss 2 s % | P50 6 s ms | "
                 "open miss % | taken miss % |\n|---|---|---|---|---|---|---|")
        for name, r in c["crossfit"]["systems"].items():
            s6 = r["6s"]["fixed_5pct_turn_fc"]
            s2 = r["2s"]["fixed_5pct_turn_fc"]
            L.append(f"| {name} | {_row(s6, s2)} |")
        L.append("\nPaired at 6 s (miss points, 95 % CI; P50 difference ms):\n\n| a - b | all | open | taken | P50 |\n|---|---|---|---|---|")
        for k, d in c["crossfit"]["paired"].items():
            if k.endswith("| 6s"):
                L.append(f"| {k[:-5]} | {_pd(d)} |")
        if "frozen_ami" in c:
            L.append("\nFrozen AMI points (same head, oracle party track, all 974 AMI dev turns):\n\n| system | AMI point | "
                     "FC % | miss 6 s % [CI] | miss 2 s % | P50 6 s ms | open miss % | taken miss % |\n|---|---|---|---|---|---|---|---|")
            for name, r in c["frozen_ami"]["systems"].items():
                s6, s2 = r["6s"]["at_ami_point"], r["2s"]["at_ami_point"]
                pt = ", ".join("never" if x is None else f"{x:.5g}" for x in r["6s"]["ami_point"])
                L.append(f"| {name} | {pt} | {_row(s6, s2)} |")
            L.append("\nPaired at the frozen points, 6 s:\n\n| a - b | all | open | taken | P50 |\n|---|---|---|---|---|")
            for k, d in c["frozen_ami"]["paired"].items():
                if k.endswith("| 6s"):
                    L.append(f"| {k[:-5]} | {_pd(d)} |")
    s = res.get("ami_dev_stream")
    if s:
        c = s["C_extended_windows_stream"]
        L.append(f"\n### AMI dev, deployable cascade (causal_dominant, 1.04 s tracks), n = {c['n']}\n\n| system | FC % | "
                 "miss 6 s % [CI] | miss 2 s (A) % | P50 6 s ms | open miss % | taken miss % |\n|---|---|---|---|---|---|---|")
        for name, r in c["systems"].items():
            s6 = r["6s"]["fixed_5pct_turn_fc"]
            s2 = s["A_default_windows_all"]["systems"].get(name, {}).get("2s", {}).get("fixed_5pct_turn_fc")
            L.append(f"| {name} | {_row(s6, s2)} |")
        L.append("\n| a - b (6 s) | all | open | taken | P50 |\n|---|---|---|---|---|")
        for k, d in c["paired"].items():
            if k.endswith("| 6s"):
                L.append(f"| {k[:-5]} | {_pd(d)} |")
    return "\n".join(L)


def stage_tb_policy(a) -> dict:
    """TurnBench dev (official gold / score_task, per speaker; research/DYADIC.md section 7's replica with TP-to-branch
    attribution) of the 'fixed' policy p > theta OR Silero silence >= k for one head, theta over the head's own score
    quantiles (plus off) x section 7's k grid: the max-recall point at fp <= 0.10 (section 4's dev rule), the lowest
    P50 with recall >= 0.80 and fp <= 0.10 (section 7's fit rule, in-sample), the lowest P50 at fp <= 0.10 at any
    recall, each with the head-branch share of TPs and P50 per branch."""
    import bench_turnbench_latency as BL
    work = BL.WORK_TB
    hd = "head" if a.tag == "trail6" else f"head_{a.tag}"
    G = BL.tb_gold()
    dur = json.loads((D.TB_SCORER / "turnbench" / "durations-dev.json").read_text())["durations"]
    chans = []
    for c in sorted(G, key=int):
        z = np.load(work / "vad" / f"{c}.npz")
        for k in (1, 2):
            sp = BL.pipecat_speech(z[f"p{k}"], work / "pipecat" / f"{c}_{k}.npy")
            chans.append(BL.Chan(c, k, sp, np.load(work / hd / f"{c}_spk{k}.npy"), float(dur[c])))
    corpus = BL.Corpus("turnbench_dev", chans, G)
    allp = np.concatenate([ch.head for ch in chans])
    thetas = [float(x) for x in np.unique(np.quantile(allp, np.concatenate([np.linspace(0.5, 0.95, 10),
                                                                                np.linspace(0.955, 0.9995, 30)])))] + [float("inf")]
    rows = []
    for th in thetas:
        for k in list(BL.K_GRID) + [1e9]:
            m = BL.aggregate(corpus.score(BL.events_fn("fixed", (th, k))), corpus.cids)
            rows.append({"theta": None if not np.isfinite(th) else th, "k_s": None if k >= 1e9 else k, **m})
    ok = [r for r in rows if r["fp_rate"] <= BL.FP_BUDGET and r["p50"] is not None]
    pick = lambda xs, key: min(xs, key=key) if xs else None  # noqa: E731
    out = {"head": a.tag, "n_points": len(rows),
           "max_recall_fp10": pick(ok, lambda r: (-r["recall"], r["fp_rate"], r["p50"])),
           "lowest_p50_recall80_fp10": pick([r for r in ok if r["recall"] >= BL.MIN_RECALL], lambda r: (r["p50"], -r["recall"])),
           "lowest_p50_fp10_any_recall": pick(ok, lambda r: (r["p50"], -r["recall"])),
           "head_alone_max_recall_fp10": pick([r for r in ok if r["k_s"] is None], lambda r: (-r["recall"], r["p50"])),
           "reaches_p50_700_fp10_recall80": any(r["p50"] <= 700 and r["recall"] >= BL.MIN_RECALL for r in ok)}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("turnbench_policy", {})[a.tag] = out
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))
    return out


OTO_TB_WORK = BT.SCRATCH / "dyadic" / "oto" / "work_tb"  # section 7's continuous oto streams (vad / pipecat)
# the bins' probabilities are small (TurnBench dev medians 0.017 / 0.027 for bins 1 / 2): a grid dense at the low end
P_GRID = (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05, 0.1, 0.2)
P_GRID_OR = (0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03)


def oto_tb_dev_ids() -> list[str]:
    """Section 7's 16 oto conversations that are DEV conversations of the dyadic runs (never trained on)."""
    import bench_turnbench_latency as BL
    dev = set(dev_ids())
    return [c for c in BL.oto_ids() if c in dev]


def stage_oto_tb_head(a):
    """Continuous per-channel head (+ multi-horizon bins) streams of the dev conversations among section 7's oto set
    (bench_turnbench.stage_head on the in-memory channels), into <OTO_TB_WORK>/head_<tag>[__mhbins]."""
    import bench_turnbench as TB
    import bench_turnbench_latency as BL
    ds = BL.oto_ds()
    ds.meetings = oto_tb_dev_ids()
    a.ckpt = str(ROOT / HEADS[a.tag])
    return TB.stage_head(a, ds, OTO_TB_WORK)


def _corpus(name, tag):
    import bench_turnbench_latency as BL
    hd = "head" if tag == "trail6" else f"head_{tag}"
    if name == "tb":
        work, G = BL.WORK_TB, BL.tb_gold()
        dur = json.loads((D.TB_SCORER / "turnbench" / "durations-dev.json").read_text())["durations"]
        cids = sorted(G, key=int)
    else:
        work = OTO_TB_WORK
        ds = D.Dyadic(oto_tb_dev_ids(), "oto", verbose=False)
        G = BL.oto_gold_all(ds)
        dur = {m: ds.duration(m) for m in ds.meetings}
        cids = list(ds.meetings)
    chans, mh, speech = [], {}, {}
    for c in cids:
        z = np.load(work / "vad" / f"{c}.npz")
        for k in (1, 2):
            sp = BL.pipecat_speech(z[f"p{k}"], work / "pipecat" / f"{c}_{k}.npy")
            ch = BL.Chan(c, k, sp, np.load(work / hd / f"{c}_spk{k}.npy"), float(dur[c]))
            chans.append(ch)
            q = work / f"{hd}__mhbins" / f"{c}_spk{k}.npy"
            if tag != "trail6" and q.exists():  # trail6's dir is 'head': 'head_mh' would be the mh head's EOT scores
                mh[id(ch)] = np.load(q)
            speech[id(ch)] = np.asarray(sp, bool)
    return BL.Corpus(name, chans, {c: G[c] for c in cids}), mh, speech


def _pred_events(ch, pm, t1, t2):
    """Predictive trigger: rising edges of P(bin1) < t1 AND P(bin2) < t2 (the user predicted silent at ~+200 and
    ~+400 ms), committed at the 160 ms chunk end, 2 s refractory; no silence timer."""
    import bench_turnbench_latency as BL
    cond = (pm[:, 0] < t1) & (pm[:, 1] < t2)
    rises = np.nonzero(cond & ~np.concatenate([[False], cond[:-1]]))[0]
    out, last = [], -np.inf
    for t in rises:
        if ch.emit[t] - last >= BL.REFRACTORY:
            out.append(round(float(ch.emit[t]), 4))
            last = ch.emit[t]
    return out


def _metrics(corpus, fn, cids, speech=None):
    """aggregate() + P10 (negative = before the gold end) + share of TPs before the end + share of ALL fires that land
    while the user's own channel is in a Silero speech chunk (before their last word ends)."""
    import bench_turnbench_latency as BL
    per = corpus.score(fn)
    m = BL.aggregate(per, cids)
    lat = np.concatenate([per[c][4] for c in cids])
    m["tp_before_end"] = round(float((lat < 0).mean()), 3) if len(lat) else None
    if speech is not None:
        n_in = n_all = 0
        for ch in corpus.chans:
            if ch.cid not in cids:
                continue
            times, _ = fn(ch)
            sp = speech[id(ch)]
            j = np.minimum((np.asarray(times) / BL.CHUNK - 1e-9).astype(int), len(sp) - 1)
            n_in += int(sp[j].sum()) if len(j) else 0
            n_all += len(times)
        m["fires_during_user_speech"] = round(n_in / max(n_all, 1), 3)
        m["n_fires"] = n_all
    return m


def stage_predictive(a) -> dict:
    """The coordinator's predictive trigger for the multi-horizon heads on TurnBench dev (official gold / scorer):
    (t1, t2) [and k for the OR with the Silero-per-channel timeout] fitted by the max-recall-at-fp<=0.10 rule on (i) oto
    dev (section 7 gold construction, the dev conversations among its 16) -> all of TurnBench dev, (ii) one TurnBench
    half -> the other (both directions, pooled); against the reactive head (p > theta, and p > theta OR silence >= k)
    fitted the same way."""
    import bench_turnbench_latency as BL
    tb, tb_mh, tb_sp = _corpus("tb", a.tag)
    try:  # the continuous oto dev streams exist for trail6 (section 7) and the multi-horizon heads (stage oto_tb_head)
        oto, oto_mh, _ = _corpus("oto", a.tag)
    except FileNotFoundError:
        oto, oto_mh = None, {}
    has_mh = bool(tb_mh)
    allp = np.concatenate([ch.head for ch in tb.chans])
    thetas = [float(x) for x in np.unique(np.quantile(allp, np.concatenate([np.linspace(0.5, 0.95, 10), np.linspace(0.955, 0.9995, 30)])))]
    fams = {"reactive_head": [(th, None) for th in thetas],
            "reactive_head_or_silero": [(th, k) for th in thetas + [float("inf")] for k in BL.K_GRID]}
    if has_mh:
        fams["predictive"] = [(t1, t2, None) for t1 in P_GRID for t2 in P_GRID]
        fams["predictive_or_silero"] = [(t1, t2, k) for t1 in P_GRID_OR for t2 in P_GRID_OR for k in BL.K_GRID]

    def fn_of(fam, pt, mh):
        def fn(ch):
            if fam.startswith("reactive"):
                sil = ch.silence_events(np.full(len(ch.s), pt[1])) if pt[1] is not None else []
                return BL.merged(ch.head_events(pt[0]), sil, ch.dur)
            sil = ch.silence_events(np.full(len(ch.s), pt[2])) if pt[2] is not None else []
            return BL.merged(_pred_events(ch, mh[id(ch)], pt[0], pt[1]), sil, ch.dur)
        return fn

    def fit(corpus, mh, cids, grid, fam):
        best = None
        for pt in grid:
            m = BL.aggregate(corpus.score(fn_of(fam, pt, mh)), cids)
            if m["fp_rate"] <= BL.FP_BUDGET and (best is None or (m["recall"], -(m["p50"] or 1e9)) > (best[1]["recall"], -(best[1]["p50"] or 1e9))):
                best = (pt, m)
        return best

    A, Bh = BL.tb_halves(tb.cids)
    out = {"head": a.tag, "oto_fit_conversations": oto.cids if oto is not None else None, "families": {}}
    for fam, grid in fams.items():
        r = {}
        f = fit(oto, oto_mh, oto.cids, grid, fam) if oto is not None and (not fam.startswith("predictive") or oto_mh) else None
        if f:
            r["oto->tb"] = {"point": f[0], "fit": f[1], "held_out": _metrics(tb, fn_of(fam, f[0], tb_mh), tb.cids, tb_sp)}
        halves = {}
        for name, fh, eh in (("tbA->tbB", A, Bh), ("tbB->tbA", Bh, A)):
            fr = fit(tb, tb_mh, fh, grid, fam)
            if fr:
                halves[name] = {"point": fr[0], "fit": fr[1], "held_out": _metrics(tb, fn_of(fam, fr[0], tb_mh), eh, tb_sp)}
        r.update(halves)
        if len(halves) == 2:  # pooled held-out halves
            per = {}
            for name, eh in (("tbA->tbB", Bh), ("tbB->tbA", A)):
                per.update({c: v for c, v in tb.score(fn_of(fam, halves[name]["point"], tb_mh)).items() if c in eh})
            pm = BL.aggregate(per, tb.cids)
            lat = np.concatenate([per[c][4] for c in tb.cids])
            pm["tp_before_end"] = round(float((lat < 0).mean()), 3) if len(lat) else None
            r["tb_halves_pooled"] = pm
        out["families"][fam] = r
        pr = r.get("tb_halves_pooled") or {}
        print(f"  {fam}: pooled held-out recall {pr.get('recall')} fp {pr.get('fp_rate')} P10 {pr.get('p10')} "
              f"P50 {pr.get('p50')} | oto->tb {r.get('oto->tb', {}).get('held_out', {}).get('recall')}", flush=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res.setdefault("turnbench_predictive", {})[a.tag] = out
    OUT.write_text(json.dumps(res, indent=1, default=float))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["scores", "check", "points", "report", "tables", "tb_policy", "oto_tb_head", "predictive"])
    p.add_argument("--tag", default="trail6", choices=list(HEADS))
    p.add_argument("--what", default="oto,ami,ami_stream")
    p.add_argument("--device", default="cpu")
    p.add_argument("--budget", type=float, default=2400.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-frozen", dest="frozen", action="store_false")
    p.add_argument("--no-ami-stream", dest="ami_stream", action="store_false")
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    if a.stage == "scores":
        r = stage_scores(a)
        tot = sum((sum(v.values()) if isinstance(v, dict) else (v if isinstance(v, int) else 1)) for v in r.values())
        print(r)
        print(f"LEFT {tot}")
    elif a.stage == "check":
        stage_check(a)
    elif a.stage == "points":
        stage_points(a)
    elif a.stage == "oto_tb_head":
        print(stage_oto_tb_head(a))
    elif a.stage == "predictive":
        stage_predictive(a)
    elif a.stage == "tb_policy":
        stage_tb_policy(a)
    elif a.stage == "tables":
        print(tables(json.loads(OUT.read_text())))
    else:
        stage_report(a)


if __name__ == "__main__":
    main()
