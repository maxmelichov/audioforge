"""smart-turn x our turn head: learned fusion on eot-bench v2 and the completeness head (research/COMPLETENESS.md).

Stages (CPU, 2 threads, each < 10 min):
  extract   per eot-bench v2 window (block C: 6 s-extended windows; block A: default 2 s windows) the per-frame score
            streams already computed by scripts/research/bench_turn_baselines.py and the trail6 run: our trail6 head p, the
            Sortformer-primary silence run (our timeout), the Silero/Pipecat silence run, smart-turn v3.2 P(complete)
            held from Silero triggers and from Sortformer-primary triggers (+ the raw trigger times / values), plus
            onset / end / strata / folds / pauses -> <work>/streams_<blk>.pkl. Nothing is recomputed.
  fusion    cross-fitted (by the v2 meeting folds) logistic / gradient-boosted fusion of those streams, scored with
            the v2 scorer (<= 5 % per-turn FC fixed on the other fold, 6 s and 2 s horizons, floor-open / taken
            strata, bootstrap CIs, paired bootstraps vs our hybrid and vs head-OR-Silero); the error
            complementarity of smart-turn and the hybrid on open ends -> runs/completeness.json ["fusion"]. When
            <work>/cmp/ holds the completeness-head stream of every window (stage streams) it is fused too.
  dyntimeout "dynamic timeout" policies: fire when the silence run >= clamp(T0 - a * s, Tmin, Tmax) frames, s = the
            trail6 head posterior / smart-turn P(complete) / both, silence = Silero any-speaker or our primary track;
            (T0, a, Tmin, Tmax) chosen on the other meeting fold under <= 5 % per-turn FC (the scorer's own rule),
            scored as every other row; plus a continuous run over the whole windows (lead + turn + trail): events per
            window, duplicate post-end fires, fires after another speaker has taken the floor -> ["dyntimeout"].
  human5    the completeness head (--ckpt) vs smart-turn v3.2's own ONNX on the same held-out human_5 clips
            (datasets/smartturn.py eval split): accuracy / AUC at the labelled utterance end and at the clip end,
            CPU per call -> runs/completeness.json ["human5"].
  streams   per-frame P(complete so far) of the completeness head on every extended v2 window (frozen encoder, [70, 1]
            streaming mask = what cache-aware streaming computes; 0.32 s of following meeting audio, cropped) ->
            <work>/cmp/<key>.npy (resumable, --budget s per process); block A = the same track cropped.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "2")  # 2-thread budget: BLAS, OpenMP (sklearn's HistGradientBoosting) and torch

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

import bench_turn_baselines as BT  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402
from audioforge.conversation import (floor_stratum,  # noqa: E402
                                     pause_runs)

SCRATCH = BT.SCRATCH
WORK = SCRATCH / "completeness"
OUT_JSON = ROOT / "runs" / "completeness.json"
BLOCKS = {"C": {"2s": 25, "6s": 75}, "A": {"2s": None}}
STREAMS = ("head", "to_primary", "to_silero", "st_silero", "st_sortformer")
EMIT_KIND = {"head": "head", "to_primary": "sf", "to_silero": "frame", "st_silero": "frame", "st_sortformer": "sf"}


# --------------------------------------------------------------------------- extract
def stage_extract(work: Path):
    base, ext, meta, ds, meta_base = BT.load()
    tr = BT.build_tracks(ext, BT.TRAIL6_WORK.parent.parent / "baselines_turn" / "work")
    for blk, convs, mt, two in (("C", ext, meta, False), ("A", base, meta_base, True)):
        tracks = ([E.v2_base_track(v) for v in convs] if two else [np.load(E.v2_track_path(BT.TRAIL6_WORK, v)) for v in convs])
        singles, _ = BT.our_systems(convs, tracks, "trail6", two)
        head, to = singles["head_v3_stream_causal_dominant"][0], singles["timeout_stream_causal_dominant"][0]
        rows = []
        for i, v in enumerate(convs):
            T = len(v["spk_act"])
            d = tr[i]
            s = np.load(BT.SCRATCH / "baselines_turn" / "work" / "smartturn" / f"{BT.key(ext[i])}.npz")
            rows.append(dict(
                key=BT.key(v), meeting=v["meeting"], T=T, onset=int(v["onset_frame"]), end=int(v["turn_end_frame"]),
                stratum=floor_stratum(v["spk_targets"], int(v["turn_end_frame"]), E.V2_FLOOR_HORIZON),
                fold=0 if v["meeting"] in E.V2_DEV_FOLDS[0] else 1, pauses=pause_runs(v["hes"]),
                post_avail=int(mt[i]["post_avail"]), end_reason=mt[i]["end_reason"],
                prim_act=(np.asarray(v["spk_act"]) > 0.5), other_act=(np.asarray(v["spk_targets"])[:, 1:] > 0.5).any(1),
                head=np.asarray(head[i], np.float32), to_primary=np.asarray(to[i], np.float32)[:T],
                to_silero=d["silero_timeout"][0][:T], st_silero=d["smartturn_silero"][0][:T],
                st_sortformer=d["smartturn_sortformer"][0][:T],
                pc_tau=s["pc_tau"], pc_p=s["pc_p"], sf_frames=s["sf_frames"], sf_p=s["sf_p"]))
            assert len(rows[-1]["head"]) == T, (blk, i, len(rows[-1]["head"]), T)
        work.mkdir(parents=True, exist_ok=True)
        (work / f"streams_{blk}.pkl").write_bytes(pickle.dumps(rows))
        print(f"  {blk}: {len(rows)} windows -> {work / f'streams_{blk}.pkl'}", flush=True)


# --------------------------------------------------------------------------- completeness head vs smart-turn
def _cmp_name(model) -> str:
    return next(k for k, v in model.head_cfg.items() if v["type"] == "completeness")


def stage_human5(a, work: Path):
    """Held-out human_5 clips: smart-turn v3.2 ONNX (its own inference: last <= 8 s, left zero-pad) vs our head."""
    import torch
    from audioforge.datasets.smartturn import SmartTurnClips
    from audioforge.heads.completeness import accuracy, auc
    from audioforge.train import load_model
    from audioforge.data import Collate, to_device
    ds = SmartTurnClips(split="eval")
    st = B.SmartTurn(BT.SMART_TURN)
    m = load_model(a.ckpt, a.device)
    name = _cmp_name(m)
    head = m.heads[name]
    ys, p_st, p_utt, p_utt_clip, p_frame, dur = [], [], [], [], [], []
    t_st = t_ours = 0.0
    for i in range(len(ds)):
        ex = ds[i]
        x = ex["audio"]
        ys.append(bool(ex["complete"]))
        dur.append(len(x) / B.SR)
        t0 = time.perf_counter()
        p_st.append(st.predict(x))
        t_st += time.perf_counter() - t0
        b = to_device(Collate(None)([ex]), a.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            enc, elen, hidden = m.encode(b["audio"], b["audio_len"], return_hidden=True)
            e = m.head_input(name, enc, hidden)
            end = torch.minimum(b["utt_end_frame"].reshape(-1).long(), elen.long()).clamp(min=1)
            pu = head.utterance_prob(e, elen, end)
            pc = head.utterance_prob(e, elen)
            z = head.decode(e, elen)
        t_ours += time.perf_counter() - t0
        p_utt.append(float(pu[0]))
        p_utt_clip.append(float(pc[0]))
        p_frame.append(float(z[0, int(end[0]) - 1]))
    # incremental cost of the head alone per 160 ms chunk (the encoder runs anyway for ASR / VAD / turn)
    with torch.no_grad():
        e = torch.randn(1, 2, m.encoder.d_model, device=a.device)
        s = head.init_stream(1)
        t0 = time.perf_counter()
        for _ in range(200):
            head.step(e, s)
        head_ms = (time.perf_counter() - t0) / 200 * 1000
    def row(p):
        return {"acc": round(accuracy(ys, p), 4), "auc": round(auc(ys, p), 4)}
    n = len(ys)
    res = {"split": ds.how, "n": n, "n_complete": int(sum(ys)), "mean_clip_sec": round(float(np.mean(dur)), 2),
           "ckpt": str(a.ckpt), "smartturn_model": str(BT.SMART_TURN.name),
           "smartturn_v3.2": {**row(p_st), "ms_per_call": round(t_st / n * 1000, 2)},
           "ours_utterance_at_vad_end": row(p_utt), "ours_utterance_at_clip_end": row(p_utt_clip),
           "ours_frame_at_vad_end": row(p_frame),
           "ours_ms_per_clip_encoder+head": round(t_ours / n * 1000, 2), "ours_head_ms_per_160ms_chunk": round(head_ms, 3),
           "bootstrap_auc_diff_ci": _auc_diff_ci(ys, p_utt, p_st), "threads": 2, "device": a.device,
           "per_clip": {"complete": ys, "smartturn": [round(float(v), 4) for v in p_st],
                        "ours_utt": [round(v, 4) for v in p_utt], "ours_frame": [round(v, 4) for v in p_frame]}}
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    prev["human5"] = res
    Path(a.out).write_text(json.dumps(prev, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))


def _auc_diff_ci(y, pa, pb, n_boot: int = 1000):
    from audioforge.heads.completeness import auc
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(y), len(y))
        d.append(auc(y[ii], pa[ii]) - auc(y[ii], pb[ii]))
    d = np.array(d)
    return {"ours - smartturn": round(float(auc(y, pa) - auc(y, pb)), 4),
            "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]}


def stage_streams(a, work: Path):
    """P(complete so far) per frame of every extended v2 window (block C); resumable."""
    import torch
    from audioforge.train import load_model
    base, ext, meta, ds, meta_base = BT.load()
    m = load_model(a.ckpt, a.device)
    assert list(m.encoder.att_context_size) == [70, 1], m.encoder.att_context_size
    name = _cmp_name(m)
    head = m.heads[name]
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        T = len(ex["spk_act"])
        x = torch.as_tensor(BT.padded_audio(ds, ex))[None].to(a.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            enc, elen, hidden = m.encode(x, torch.tensor([x.shape[1]], device=a.device), return_hidden=True)
            z = head.decode(m.head_input(name, enc, hidden), elen)[0].float().cpu().numpy()
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += 1
        np.save(out.with_name(out.stem + ".tmp.npy"), BT.B_fit(z, T, 0.0).astype(np.float32))
        out.with_name(out.stem + ".tmp.npy").replace(out)
    left = BT.run_resumable("cmp", ext, fn, work, a.budget, ".npy")
    BT.add_timing(work, "cmp_offline_window", stat["calls"], stat["sec"], ckpt=str(a.ckpt), device=a.device)
    return left


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["extract", "fusion", "human5", "streams", "dyntimeout"])
    p.add_argument("--ckpt", default=str(ROOT / "runs" / "stage1_completeness.afm"))
    p.add_argument("--budget", type=float, default=480.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--work", default=str(WORK))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(OUT_JSON))
    a = p.parse_args()
    BT._torch2()
    work = Path(a.work)
    if a.stage == "extract":
        stage_extract(work)
    elif a.stage == "fusion":
        from bench_completeness_fusion import stage_fusion
        stage_fusion(a, work)
    elif a.stage == "dyntimeout":
        from bench_completeness_dyn import stage_dyntimeout
        stage_dyntimeout(a, work)
    elif a.stage == "human5":
        stage_human5(a, work)
    elif a.stage == "streams":
        stage_streams(a, work)


if __name__ == "__main__":
    main()
