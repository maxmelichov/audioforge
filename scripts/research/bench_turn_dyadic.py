"""eot-bench v2 (research/archive/EOT_BENCH_V2.md) on the dyadic slices (research/archive/DYADIC.md), NO retraining: how the AMI-trained
trail6 head and the timeout / Silero / Parakeet-EOU baselines behave on two-party audio where the "user" is one channel.

Set: the HUMAN party's non-backchannel turns (datasets/dyadic.py roles; agent = the party who speaks second) of the
first conversations of a slice, as the library's turn windows (20 s, lead 4 s) with a 2 s trail (block A) and re-cut
with a 6 s trail from the same start (block C; horizons 25 / 75 post-end emission frames), >= 500 turns per corpus.
Folds for the cross-fitted <= 5 % per-turn FC operating point: conversations at even / odd positions.

Systems (block C and A; 'oracle' = the party's own labelled activity, which a two-channel product has for free):
  timeout_primary_oracle        silence timeout on the human party's activity (label-armed at the onset)
  timeout_any_speaker_oracle    the same on the union of both parties (speaker-unaware)
  silero_timeout                Pipecat's Silero v5 state machine on the MIXED MONO (speaker-unaware, deployable)
  head_trail6_oracle_track      our trail6 head on the mixed mono with spk_act = the human's oracle activity and cols =
                                [human, agent, 0, 0] (the agent-aware input a product has), 160 ms emission
  hybrid_oracle                 head OR primary timeout; head_trail6+silero_timeout: head OR Silero timeout
  eou_posterior                 Parakeet-Realtime-EOU log P(<EOU>) on the mixed mono (+ 0.32 s of real audio, cropped)
  natives                       1040 ms (13 frames) primary timeout, 1000 ms Silero timeout, <EOU> emitted
  [subset] timeout / head / any-speaker on streaming Sortformer v2 tracks with causal_dominant enrollment: only on
           --max-tracks windows (39 s of CPU per 20 s window in window mode: see research/archive/DYADIC.md), reported as a
           separate block together with the oracle rows on the same windows.
Block A's head / EOU scores are the first frames of the block C streams (causal models: the 2 s window is a prefix
of the 6 s one; no re-run with an end-of-file flush 2 s after the end).

Stages (each one process <= 10 min, resumable; CPU, 2 threads):
  data      build / count the windows (labels + mix caches: scripts/research/prepare_dyadic.py --labels)
  vad       Silero VAD v5 probabilities (32 ms) of the extended windows' mono mix
  head      trail6 head scores with the oracle party track (extended windows)
  eou       Parakeet-EOU per-frame log posteriors (extended windows)
  tracks    [optional] streaming Sortformer tracks of the first --max-tracks windows
  scores    [optional] trail6 head scores with causal_dominant binding on those tracks
  ami_head  AMI dev (974 turns) trail6 scores with the ORACLE party track, for the frozen-AMI operating points
  report    -> runs/dyadic_bench_<corpus>.json

  W=<scratch>/dyadic/<corpus>/work
  .venv/bin/python scripts/research/bench_turn_dyadic.py --corpus behavior_sd --stage vad --work $W
  .venv/bin/python scripts/research/bench_turn_dyadic.py --corpus behavior_sd --stage head --work $W   # repeat until "0 left"
  ...
  .venv/bin/python scripts/research/bench_turn_dyadic.py --corpus behavior_sd --stage report --work $W
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
import bench_turn_icsi as BI  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402
from audioforge.datasets import dyadic as D  # noqa: E402

HEAD_CKPT = ROOT / "runs" / "stage1_turn_v3_trail6.afm"
HEAD_TAG = "trail6"
HEAD_CHUNK = 2  # 160 ms encoder chunks -> emission (t // 2 + 1) * 2
SCRATCH = BT.SCRATCH / "dyadic"
AMI_WORK = SCRATCH / "ami" / "work"  # AMI dev oracle-track head scores (stage ami_head)
MAX_FC = 0.05
DEFAULTS = {"behavior_sd": dict(hours=5.0), "dailytalk": dict(hours=3.0), "oto": dict(hours=None), "turnbench": dict(hours=None)}
MIN_TURNS = 500
EMIT_FRAME = lambda t: t + 1  # noqa: E731
EMIT_HEAD = lambda t: (t // HEAD_CHUNK + 1) * HEAD_CHUNK  # noqa: E731
EMIT_EOU = BT.EMIT["eou"]
REF = "hybrid_oracle"
PAIRS = [("head_trail6_oracle_track", "timeout_primary_oracle"), ("hybrid_oracle", "timeout_primary_oracle"),
         ("hybrid_oracle", "head_trail6_oracle_track"), ("silero_timeout", "timeout_primary_oracle"),
         ("head_trail6_oracle_track", "silero_timeout"), ("head_trail6+silero_timeout", "silero_timeout"),
         ("eou_posterior", "silero_timeout"), ("eou_posterior", "timeout_primary_oracle"),
         ("timeout_any_speaker_oracle", "timeout_primary_oracle"),
         ("head_v3_stream_causal_dominant", "head_trail6_oracle_track"),
         ("timeout_stream_causal_dominant", "timeout_primary_oracle"),
         ("head_v3_stream_causal_dominant", "timeout_stream_causal_dominant")]


def silero_path() -> Path:
    """The Silero v5 ONNX of the AMI benchmark if present (read-only), else the silero_vad package's model."""
    if BT.SILERO_V5.exists():
        return BT.SILERO_V5
    import silero_vad
    return Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"


def key(ex) -> str:
    return BT.key(ex)


# --------------------------------------------------------------------------- data
def dyadic_data(corpus: str, hours, n_conv: int | None, max_turns: int, trail_sec: float = 6.0, roles=("human",)):
    """(base, ext, meta, ds, meta_base) as eval_stage1.v2_data on a dyadic slice: the human party's turns, base = the
    default 2 s-trail windows, ext = the same starts with a ``trail_sec`` trail; windows over an excluded zone
    dropped from both. Conversations are taken in slice order until ``max_turns`` human turns (>= MIN_TURNS)."""
    ids = D.slice_ids(corpus, None, hours)
    if n_conv:
        ids = ids[:n_conv]
    ds = D.Dyadic(ids, corpus, verbose=False)
    with ds.tag_primary() as names:
        base, meta_base = E.turn_windows(ds, 2.0)
    ds.resolve_primary(base, names)
    with ds.tag_primary() as names:
        ext, meta = E.turn_windows(ds, trail_sec, starts=[v["start"] for v in base])
    ds.resolve_primary(ext, names)
    for u, v in zip(ext, base):
        assert u["onset_frame"] == v["onset_frame"] and u["turn_end_frame"] == v["turn_end_frame"] and u["party"] == v["party"]
    idx = [i for i, v in enumerate(ext) if (not roles or v["role"] in roles) and ds._keep(v) is None and ds._keep(base[i]) is None]
    if max_turns and len(idx) > max_turns:  # whole conversations, in order, until the cap
        keep, seen = [], []
        for i in idx:
            if ext[i]["meeting"] not in seen:
                if len(keep) >= max_turns:
                    break
                seen.append(ext[i]["meeting"])
            keep.append(i)
        idx = keep
    return [base[i] for i in idx], [ext[i] for i in idx], [meta[i] for i in idx], ds, [meta_base[i] for i in idx]


def folds_of(convs) -> tuple:
    ids = []
    for v in convs:
        if v["meeting"] not in ids:
            ids.append(v["meeting"])
    return tuple(ids[0::2]), tuple(ids[1::2])


# --------------------------------------------------------------------------- stages
def stage_vad(a, ext, work):
    vad = B.SileroVAD(silero_path())

    def fn(ex, out):
        p = vad.probs(np.asarray(ex["audio"], np.float32))
        np.save(out.with_name(out.stem + ".tmp.npy"), p)
        out.with_name(out.stem + ".tmp.npy").replace(out)
    return BT.run_resumable("silero", ext, fn, work, a.budget, ".npy")


def oracle_score_path(work: Path, ex: dict) -> Path:
    return Path(work) / f"scores_oracle_track__{HEAD_TAG}" / f"{key(ex)}.npy"


def head_oracle_scores(model, convs, work: Path, budget: float, batch_size: int) -> int:
    """trail6 head scores with spk_act = the oracle primary and cols = spk_targets (human, agent, 0, 0), prim 0."""
    import torch
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    todo = [v for v in convs if not oracle_score_path(work, v).exists()]
    oracle_score_path(work, convs[0]).parent.mkdir(parents=True, exist_ok=True)
    todo.sort(key=lambda v: len(v["audio"]))
    t0, done = time.time(), 0
    with torch.no_grad():
        for j in range(0, len(todo), batch_size):
            if time.time() - t0 > budget:
                break
            cc = todo[j: j + batch_size]
            sc = E.turn_scores_given_act(model, name, cc, [v["spk_act"] for v in cc], batch_size,
                                         [np.asarray(v["spk_targets"], np.float32) for v in cc], [0] * len(cc))
            for v, s in zip(cc, sc):
                np.save(oracle_score_path(work, v), s)
            done += len(cc)
    print(f"  head oracle-track scores: {done} done in {time.time() - t0:.0f}s, {len(todo) - done} left", flush=True)
    return len(todo) - done


def stage_head(a, ext, work):
    from audioforge.train import load_model
    model = load_model(str(HEAD_CKPT), "cpu")
    assert E.chunk_of(model) == HEAD_CHUNK, E.chunk_of(model)
    return head_oracle_scores(model, ext, work, a.budget, a.batch_size)


def stage_ami_head(a):
    """AMI dev (all 974 turns, the trail6 extended windows) with the oracle party track: the frozen operating point of
    the same input condition. Written to AMI_WORK (never into another run's work dir)."""
    from audioforge.train import load_model
    base, ext, meta, ds, meta_base = E.v2_data(6.0)
    model = load_model(str(HEAD_CKPT), "cpu")
    AMI_WORK.mkdir(parents=True, exist_ok=True)
    return head_oracle_scores(model, ext, AMI_WORK, a.budget, a.batch_size)


def stage_eou(a, ext, ds, work):
    from audioforge.nemo_import import import_eou
    m = import_eou(BT.EOU_NEMO)
    ids = [m.eou_ids["<EOU>"], m.eou_ids["<EOB>"]]

    def fn(ex, out):
        T = len(ex["spk_act"])
        x = BT.padded_audio(ds, ex)
        toks, logpost, em = B.rnnt_frame_decode(m, x, watch_ids=ids)
        BT.save_npz(out, logpost=BT.B_fit(logpost, T, -np.inf), emitted=BT.B_fit(em, T, False),
                    text=np.array(m.tokenizer.decode([k for tt in toks[:T] for k in tt])))
    return BT.run_resumable("eou", ext, fn, work, a.budget)


def stage_tracks(a, ext, ds, work):
    from audioforge.conversation import floor_stratum
    sub = ext[: a.max_tracks]
    op = [floor_stratum(v["spk_targets"], v["turn_end_frame"], E.V2_FLOOR_HORIZON) == "open" for v in sub]
    order = [i for i in range(len(sub)) if op[i]] + [i for i in range(len(sub)) if not op[i]]
    return E.v2_tracks(sub, ds, work, a.budget, order=order)


def stage_scores(a, base, ext, work):
    from audioforge.train import load_model
    model = load_model(str(HEAD_CKPT), "cpu")
    n = a.max_tracks
    return E.v2_scores(model, base[:n], ext[:n], work, ["causal_dominant", "causal_dominant@2s"], a.budget,
                       a.batch_size, HEAD_TAG, tracks_dir=work / "tracks")


# --------------------------------------------------------------------------- systems
def systems_for(convs, work: Path, two: bool, tracks=None, head_stream=None, ext=None):
    """{name: (scores, emit)} singles and {name: (scores_a, emit_a, scores_b, emit_b)} hybrids of one block. The
    stored streams are keyed by the EXTENDED window (``ext``, same order); a default window reads their prefix."""
    ext = ext or convs
    T = [len(v["spk_act"]) for v in convs]
    on = [v["onset_frame"] for v in convs]
    sil = [BI.silero_timeout_track(np.load(work / "silero" / f"{key(u)}.npy"), t)[:t] for u, t in zip(ext, T)]
    head = [np.load(oracle_score_path(work, u))[:t] for u, t in zip(ext, T)]
    eou = [np.load(work / "eou" / f"{key(u)}.npz")["logpost"][:t, 0] for u, t in zip(ext, T)]
    eou_em = [np.load(work / "eou" / f"{key(u)}.npz")["emitted"][:t, 0].astype(np.float32) for u, t in zip(ext, T)]
    to = [E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)]
    anys = [E.silence_scores(np.asarray(v["spk_targets"]).max(1), o) for v, o in zip(convs, on)]
    singles = {"timeout_primary_oracle": (to, EMIT_FRAME), "timeout_any_speaker_oracle": (anys, EMIT_FRAME),
               "silero_timeout": (sil, EMIT_FRAME), "head_trail6_oracle_track": (head, EMIT_HEAD),
               "eou_posterior": (eou, EMIT_EOU)}
    hybrids = {REF: (head, EMIT_HEAD, to, EMIT_FRAME), "head_trail6+silero_timeout": (head, EMIT_HEAD, sil, EMIT_FRAME)}
    natives = {"timeout_1040ms_primary_oracle": (to, EMIT_FRAME, 12.0, None, None, None),
               "silero_timeout_1000ms": (sil, EMIT_FRAME, 12.5 - 1e-6, None, None, None),
               "eou_native (<EOU> emitted)": (eou_em, EMIT_EOU, 0.5, None, None, None)}
    if tracks is not None:
        tr = [np.stack([E._fit(p[:, j], t) for j in range(p.shape[1])], 1) for p, t in zip(tracks, T)]
        acts = [E._v2_cols(v, p, "causal_dominant")[0] for v, p in zip(convs, tr)]
        es, eh = E._stream_emit(BI.C_SF, BI.R_SF), E._stream_emit(BI.C_SF, BI.R_SF, HEAD_CHUNK)
        singles.update({"timeout_stream_causal_dominant": ([E.silence_scores(x, E.arm_frame(x)) for x in acts], es),
                        "head_v3_stream_causal_dominant": (head_stream, eh),
                        "timeout_any_speaker_stream": ([E.silence_scores(p.max(1), E.arm_frame(p.max(1))) for p in tr], es)})
        hybrids["hybrid_stream_causal_dominant"] = (head_stream, eh, singles["timeout_stream_causal_dominant"][0], es)
    return singles, hybrids, natives


def ami_oracle_systems():
    """AMI dev blocks with the oracle-track head scores (stage ami_head), the label timeouts and the AMI Silero
    probabilities (bench_turn_baselines' work dir, read-only) if present: {block: (convs, meta, singles, hybrids, hz)}."""
    base, ext, meta, ds, meta_base = E.v2_data(6.0)
    if not all(oracle_score_path(AMI_WORK, v).exists() for v in ext):
        return None
    have_sil = all((BI.AMI_SILERO / f"{key(v)}.npy").exists() for v in ext)
    out = {}
    for blk, convs, mt, hz in (("C_extended_windows", ext, meta, {"2s": 25, "6s": 75}),
                               ("A_default_windows", base, meta_base, {"2s": None})):
        T = [len(v["spk_act"]) for v in convs]
        on = [v["onset_frame"] for v in convs]
        head = [np.load(oracle_score_path(AMI_WORK, u))[:t] for u, t in zip(ext, T)]
        to = [E.silence_scores(v["spk_act"], o) for v, o in zip(convs, on)]
        singles = {"timeout_primary_oracle": (to, EMIT_FRAME),
                   "timeout_any_speaker_oracle": ([E.silence_scores(np.asarray(v["spk_targets"]).max(1), o)
                                                   for v, o in zip(convs, on)], EMIT_FRAME),
                   "head_trail6_oracle_track": (head, EMIT_HEAD)}
        hybrids = {REF: (head, EMIT_HEAD, to, EMIT_FRAME)}
        if have_sil:
            sil = [BI.silero_timeout_track(np.load(BI.AMI_SILERO / f"{key(u)}.npy"), t)[:t] for u, t in zip(ext, T)]
            singles["silero_timeout"] = (sil, EMIT_FRAME)
            hybrids["head_trail6+silero_timeout"] = (head, EMIT_HEAD, sil, EMIT_FRAME)
        out[blk] = (convs, mt, singles, hybrids, hz)
    BI.drop_audio(base)
    BI.drop_audio(ext)
    return out


def frozen_from_ami(ami_blk, convs_d, sing_d, hyb_d, n_boot: int) -> dict:
    """Every dyadic system that also exists on AMI, at the operating point fitted on ALL 974 AMI dev turns (in-sample,
    <= 5 % per-turn FC) and applied unchanged (bench_turn_icsi.fit_points / outcomes_at)."""
    from audioforge.conversation import floor_stratum
    convs_a, _, sing_a, hyb_a, hz = ami_blk
    folds_a = np.array([0 if v["meeting"] in E.V2_DEV_FOLDS_AMI[0] else 1 for v in convs_a])
    on, en, pauses = BI._prep(convs_d)
    strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs_d, en)])
    groups = {"open": strata == "open", "taken": strata != "open"}
    res = {}
    for name in list(sing_a) + list(hyb_a):
        hybrid = name in hyb_a
        if name not in (hyb_d if hybrid else sing_d):
            continue
        spec_a = hyb_a[name] if hybrid else sing_a[name]
        spec_d = hyb_d[name] if hybrid else sing_d[name]
        r = {}
        for h, L in hz.items():
            pts = BI.fit_points(spec_a, hybrid, convs_a, L, folds_a)
            th = pts["all"]
            fc, lat, pf, npause = BI.outcomes_at(spec_d, hybrid, on, en, pauses, L, th)
            r[h] = {"ami_point": BI.fmt_point(name, th) if name in (BI.REF, BI.CANDIDATE) or not hybrid
                    else {"theta": float(th[0]) if np.isfinite(th[0]) else None,
                          "timeout_threshold": float(th[1]) if np.isfinite(th[1]) else None},
                    "at_ami_point": BI.point_metrics(fc, lat, pf, npause, groups, n_boot)}
        res[name] = r
        print(f"    frozen-AMI {name}: " + ", ".join(f"{h} miss {r[h]['at_ami_point']['miss_rate']:.3f} fc "
                                                     f"{r[h]['at_ami_point']['fc_rate']:.3f}" for h in hz), flush=True)
    return res


def stage_report(a, base, ext, meta, meta_base, ds, work: Path) -> dict:
    t0 = time.time()
    folds = folds_of(ext)
    E.V2_DEV_FOLDS_AMI = E.V2_DEV_FOLDS
    E.V2_DEV_FOLDS = folds
    st = ds.stats()
    out = {"protocol": "eot-bench v2 (research/archive/EOT_BENCH_V2.md) on a dyadic slice: human-party turns, cross-fitted "
                       "<= 5 % per-turn FC operating point (folds = conversations at even / odd positions), 1000 "
                       "bootstraps; frozen_ami = the operating point fitted on all 974 AMI dev turns applied unchanged",
           "corpus": a.corpus, "hours_slice": a.hours, "roles_scored": a.roles,
           "conversations": len(set(v["meeting"] for v in ext)),
           "n": len(ext), "roles": ds.roles, "agent_rule": ds.agent_rule, "word_timing": ds.word_timing,
           "activity_source": "labels" if a.corpus == "behavior_sd" else ds.activity_source,
           "folds": [list(f) for f in folds], "head": str(HEAD_CKPT.relative_to(ROOT)), "n_boot": a.n_boot,
           "silero_model": str(silero_path()), "label_stats": st,
           "post_avail": {k: int(np.percentile([m["post_avail"] for m in meta], q)) for k, q in (("p5", 5), ("p50", 50), ("p95", 95))},
           "end_reasons": {r: int(sum(m["end_reason"] == r for m in meta)) for r in ("resume", "trail", "meeting_end")},
           "agent_end_frame_available": int(sum(v["agent_end_frame"] >= 0 for v in ext))}
    ami = ami_oracle_systems() if a.frozen_ami else None
    out["frozen_ami_available"] = ami is not None
    for blk, convs, mt, hz, two in (("C_extended_windows", ext, meta, {"2s": 25, "6s": 75}, False),
                                    ("A_default_windows", base, meta_base, {"2s": None}, True)):
        print(f"  {blk} ({time.time() - t0:.0f}s)", flush=True)
        sing, hyb, nat = systems_for(convs, work, two, ext=ext)
        r = {"crossfit": BT.score_block(convs, mt, sing, hyb, hz, a.n_boot, PAIRS, natives=nat)}
        if ami is not None:
            r["frozen_ami"] = frozen_from_ami(ami["C_extended_windows" if not two else "A_default_windows"], convs,
                                              sing, hyb, a.n_boot)
        out[blk] = r
    # streaming-track subset (if any tracks exist)
    n = a.max_tracks
    have = [i for i in range(min(n, len(ext))) if E.v2_track_path(work, ext[i]).exists()
            and E.v2_score_path(work, "causal_dominant", ext[i], HEAD_TAG).exists()]
    if have:
        sub = [ext[i] for i in have]
        tr = [np.load(E.v2_track_path(work, v)) for v in sub]
        hs = [np.load(E.v2_score_path(work, "causal_dominant", v, HEAD_TAG)) for v in sub]
        E.V2_DEV_FOLDS = folds_of(sub)
        sing, hyb, nat = systems_for(sub, work, False, tracks=tr, head_stream=hs)
        r = BT.score_block(sub, [meta[i] for i in have], sing, hyb, {"2s": 25, "6s": 75}, a.n_boot, PAIRS, natives=nat)
        r["track_quality"] = E.v2_track_quality(sub, [np.stack([E._fit(p[:, j], len(v["spk_act"])) for j in range(p.shape[1])], 1)
                                                      for p, v in zip(tr, sub)], np.array([v["onset_frame"] for v in sub]),
                                                np.array([v["turn_end_frame"] for v in sub]), BI.C_SF, BI.R_SF)
        out["C_stream_subset"] = {"n": len(sub), "folds": [list(f) for f in E.V2_DEV_FOLDS], **r}
    E.V2_DEV_FOLDS = E.V2_DEV_FOLDS_AMI
    out["sec"] = round(time.time() - t0, 1)
    return out


def brief(res: dict) -> str:
    """One line per system: miss 6 s / 2 s (block C), open / taken 6 s, P50 6 s, FC."""
    rows = []
    c = res["C_extended_windows"]["crossfit"]
    for name, r in list(c["systems"].items()) + list(c["native"].items()):
        s6 = r["6s"].get("fixed_5pct_turn_fc", r["6s"])
        s2 = r["2s"].get("fixed_5pct_turn_fc", r["2s"])
        rows.append(f"{name:36s} miss6 {100 * s6['miss_rate']:5.1f} [{100 * s6['miss_rate_ci'][0]:.1f},{100 * s6['miss_rate_ci'][1]:.1f}]"
                    f" open {100 * s6['strata']['open']['miss_rate']:5.1f} taken {100 * s6['strata']['taken']['miss_rate']:5.1f}"
                    f" P50 {s6['p50_ms']:.0f} ms fc {100 * s6['fc_rate']:.1f}% | miss2 {100 * s2['miss_rate']:5.1f}")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", required=True, choices=list(D.CORPORA))
    p.add_argument("--stage", required=True, choices=["data", "vad", "head", "eou", "tracks", "scores", "ami_head", "report"])
    p.add_argument("--work", default=None)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--n-conv", type=int, default=None, help="first N conversations of the slice")
    p.add_argument("--max-turns", type=int, default=650, help="human turns: whole conversations until this cap")
    p.add_argument("--max-tracks", type=int, default=0, help="streaming Sortformer subset size (0 = none)")
    p.add_argument("--roles", default="human", choices=["human", "both"], help="whose turn ends are scored")
    p.add_argument("--budget", type=float, default=500.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-frozen-ami", dest="frozen_ami", action="store_false")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    BT._torch2()
    if a.hours is None:
        a.hours = DEFAULTS[a.corpus]["hours"]
    work = Path(a.work) if a.work else SCRATCH / a.corpus / "work"
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "ami_head":
        print(stage_ami_head(a))
        return
    t0 = time.time()
    base, ext, meta, ds, meta_base = dyadic_data(a.corpus, a.hours, a.n_conv, a.max_turns,
                                                 roles=("human",) if a.roles == "human" else None)
    ncv = len(set(v["meeting"] for v in ext))
    print(f"  {a.corpus}: {len(ext)} human turns from {ncv} conversations ({time.time() - t0:.0f}s); "
          f"filtered {ds.filtered.get('turn')}; agent_end available for {sum(v['agent_end_frame'] >= 0 for v in ext)}", flush=True)
    if len(ext) < MIN_TURNS:
        print(f"  WARNING: fewer than {MIN_TURNS} turns; add conversations (--n-conv / --hours)", flush=True)
    if a.stage == "data":
        from audioforge.conversation import floor_stratum
        st = [floor_stratum(v["spk_targets"], v["turn_end_frame"], E.V2_FLOOR_HORIZON) for v in ext]
        print({s: st.count(s) for s in ("open", "overlap", "switch")}, {r: sum(m["end_reason"] == r for m in meta) for r in ("resume", "trail", "meeting_end")})
        return
    if a.stage == "vad":
        print(stage_vad(a, ext, work))
    elif a.stage == "head":
        print(stage_head(a, ext, work))
    elif a.stage == "eou":
        print(stage_eou(a, ext, ds, work))
    elif a.stage == "tracks":
        print(stage_tracks(a, ext, ds, work))
    elif a.stage == "scores":
        print(stage_scores(a, base, ext, work))
    else:
        res = stage_report(a, base, ext, meta, meta_base, ds, work)
        out = Path(a.out) if a.out else ROOT / "runs" / f"dyadic_bench_{a.corpus}.json"
        out.write_text(json.dumps(res, indent=1, default=float))
        print(brief(res))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
