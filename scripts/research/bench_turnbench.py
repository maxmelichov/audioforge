"""TurnBench dev (research/archive/OUTSIDE.md 2.3, research/archive/DYADIC.md) scored with the OFFICIAL MIT scorer (vendored unchanged in
integrations/turnbench_scorer): our systems run CAUSALLY on the 38 two-channel conversations and commit discrete EOT
events per speaker; the scorer does the rest (consensus gold, [t - 0.25, min(t + 3, next)] windows, one FP per
negative span). Timestamps are commit times: the end of the 32 ms Silero chunk / 20 ms RMS window / 160 ms encoder
chunk that revealed the decision (audio heard up to that time only). No thresholds are re-fitted per conversation;
the dev operating point of every system is the highest EOT recall at fp_rate <= 0.10 (their rule), swept over the
system's own score quantiles (their rule 2), and frozen in a predictions JSON.

Inputs, per speaker k (both channels are given to every TurnBench baseline; ours use them as follows):
  rms_timeout        their energy VAD (20 ms RMS > 0.01 on channel k) + a silence timeout of k_s seconds
  silero_timeout     Pipecat's Silero v5 state machine on channel k + a silence timeout (the deployable cascade)
  silero_mix_timeout the same on the MIXED MONO (speaker-unaware: fires for either speaker's silence)
  head_trail6        our AMI-trained trail6 head on the MIXED MONO with spk_act = channel k's Silero activity and
                     cols = [channel k, other channel, 0, 0] (the agent-aware input a two-channel product has); events
                     = rising edges of p above theta with a 2 s refractory (their commit rule)
  hybrid             head OR silero_timeout (channel k); head+silero_mix: head OR silero_mix_timeout
  eou                Parakeet-Realtime-EOU on channel k: posterior log P(<EOU>) with theta (+ refractory), and the
                     native "<EOU> emitted" point
Not run: the streaming Sortformer track cascade (window mode costs 39 s of CPU per 20 s of audio on this machine:
7.3 h x 2 speakers is >= 28 CPU-hours; research/archive/DYADIC.md).

Stages (each one process <= 10 min, resumable; CPU, 2 threads; long audio is processed in 60 s segments with 8 s of
left overlap, the recurrent head / decoder state restarting per segment):
  vad     Silero (ch 1, ch 2, mix) + RMS masks per conversation
  head    trail6 scores per (conversation, speaker)
  eou     Parakeet-EOU per (conversation, channel)
  sweep   operating points + predictions JSONs (<work>/preds/<system>.json) + the published dev files re-scored
          -> runs/turnbench_dev.json
  W=<scratch>/dyadic/turnbench/work_official
  .venv/bin/python scripts/research/bench_turnbench.py --stage vad --work $W        # repeat until "0 left"
  .venv/bin/python scripts/research/bench_turnbench.py --stage head --work $W       # repeat until "0 left" (~1.4 CPU-h)
  .venv/bin/python scripts/research/bench_turnbench.py --stage eou --work $W        # repeat until "0 left" (~1.5 CPU-h)
  .venv/bin/python scripts/research/bench_turnbench.py --stage sweep --work $W
Our eot-bench v2 protocol on the same audio: scripts/research/bench_turn_dyadic.py --corpus turnbench --roles both.
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
sys.path.insert(0, str(ROOT / "integrations" / "turnbench_scorer"))

import bench_turn_baselines as BT  # noqa: E402
import bench_turn_dyadic as BD  # noqa: E402
import eval_stage1 as E  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402
from audioforge.datasets import dyadic as D  # noqa: E402
from audioforge.datasets.ami import FRAME_SEC, SR  # noqa: E402

SEG_SEC, OVL_SEC = 60.0, 8.0
CHUNK = B.CHUNK_SEC  # 32 ms Silero
RMS_WIN, RMS_THR = 0.02, 0.01  # their baselines/rms_vad rule (on the 16 kHz resample here)
REFRACTORY = 2.0
FP_BUDGET = 0.10
K_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
Q_GRID = tuple(np.round(np.concatenate([np.linspace(0.50, 0.95, 10), [0.97, 0.98, 0.99, 0.995, 0.998, 0.999]]), 4))
HYB_Q = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
HYB_K = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
PUBLISHED_TEST = {  # research/archive/OUTSIDE.md 2.2 (test split; leaderboard-test.json holds the rest)
    "vap": (0.845, 0.055, 368), "smart_turn_v3": (0.752, 0.047, None), "kyutai_semantic_vad": (0.773, 0.059, 1007)}
DEV_BASELINES = ("vap", "smart_turn_v3", "kyutai_semantic_vad", "rms_vad", "espnet_turntaking",
                 "espnet_turntaking_perchannel", "wavlm_large_causal", "mimi_endpointer", "openai_server_vad",
                 "openai_semantic_vad", "oracle_annotator")


def ds_all():
    return D.Dyadic(D.list_ids("turnbench"), "turnbench", verbose=False)


def budget_loop(name, items, fn, work: Path, budget: float, suffix: str):
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (d / f"{it}{suffix}").exists()]
    t0, n = time.time(), 0
    for it in todo:
        if time.time() - t0 > budget:
            break
        fn(it, d / f"{it}{suffix}")
        n += 1
    print(f"  {name}: {n} done in {time.time() - t0:.0f}s, {len(todo) - n} left", flush=True)
    return len(todo) - n


def segments(n: int):
    """[(start sample, end sample, first sample to keep)] of SEG_SEC pieces with OVL_SEC left overlap."""
    seg, ovl = int(SEG_SEC * SR), int(OVL_SEC * SR)
    out, s = [], 0
    while s < n:
        a = max(0, s - ovl)
        out.append((a, min(n, s + seg), s - a))
        s += seg
    return out


# --------------------------------------------------------------------------- stages
def stage_vad(a, ds, work):
    vad = B.SileroVAD(BD.silero_path())

    def fn(cid, out):
        ch = np.asarray(ds.channels16k(cid), np.float32)
        mix = D.mix_mono(ch)
        p1, p2, pm = vad.probs(ch[0]), vad.probs(ch[1]), vad.probs(mix)
        w = int(RMS_WIN * SR)
        rms = [np.sqrt((c[: len(c) // w * w].reshape(-1, w) ** 2).mean(1)) > RMS_THR for c in ch]
        BT.save_npz(out, p1=p1, p2=p2, pm=pm, rms1=rms[0], rms2=rms[1])
    return budget_loop("vad", ds.meetings, fn, work, a.budget, ".npz")


def frame_track(p: np.ndarray, T: int) -> np.ndarray:
    """Per-32 ms probabilities -> per-80 ms soft activity (mean of the chunks that end inside the frame)."""
    ends = (np.arange(len(p)) + 1) * CHUNK
    f = np.minimum((ends / FRAME_SEC - 1e-9).astype(np.int64), T - 1)
    s = np.bincount(f, weights=p, minlength=T)[:T]
    c = np.bincount(f, minlength=T)[:T]
    return np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)


def head_dir(tag: str) -> str:
    """Score directory of a head: 'head' for trail6 (the committed run), 'head_<tag>' for the others."""
    return "head" if tag == "trail6" else f"head_{tag}"


def stage_head(a, ds, work):
    import torch
    from audioforge.data import ToneLanguage
    from audioforge.train import load_model
    model = load_model(str(a.ckpt or BD.HEAD_CKPT), "cpu").to(torch.device(a.device)).eval()
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    items = [f"{cid}_spk{k}" for cid in ds.meetings for k in (1, 2)]

    def fn(item, out):
        cid, k = item.rsplit("_spk", 1)
        k = int(k)
        z = np.load(work / "vad" / f"{cid}.npz")
        # the mixed mono exactly as Dyadic._load_audio builds TurnBench's mix cache (from the float16 channel cache),
        # computed in memory: no second 1.7 GB cache on a full disk
        mix = D.mix_mono(np.asarray(ds.channels16k(cid), np.float32))
        Tc = ToneLanguage.n_frames(len(mix))
        tr = {1: frame_track(z["p1"], Tc), 2: frame_track(z["p2"], Tc)}
        scores = np.zeros(Tc, np.float32)
        has_mh = getattr(model.heads[name], "mh", None) is not None
        mh = np.zeros((Tc, model.heads[name].mh.out_features), np.float32) if has_mh else None
        with torch.no_grad():
            for s0, s1, keep in segments(len(mix)):
                x = mix[s0:s1]
                T = ToneLanguage.n_frames(len(x))
                f0 = int(round(s0 / SR / FRAME_SEC))
                act = np.zeros(T, np.float32)
                cols = np.zeros((T, 4), np.float32)
                for j, kk in enumerate((k, 3 - k)):
                    seg = tr[kk][f0: f0 + T]
                    cols[: len(seg), j] = seg
                act[:] = cols[:, 0]
                ax = [] if has_mh else None
                sc = E.turn_scores_given_act(model, name, [dict(audio=x, spk_act=act, spk_targets=cols)], [act], 1,
                                             [cols], [0], aux=ax)[0]
                fk = int(round(keep / SR / FRAME_SEC))
                n = min(len(sc) - fk, Tc - (f0 + fk))
                if n > 0:
                    scores[f0 + fk: f0 + fk + n] = sc[fk: fk + n]
                    if has_mh:
                        mh[f0 + fk: f0 + fk + n] = ax[0][fk: fk + n]
        if has_mh:  # multi-horizon user-activity bins (research/archive/DYADIC.md section 8's predictive trigger)
            q = out.parent.parent / (out.parent.name + "__mhbins") / out.name
            q.parent.mkdir(parents=True, exist_ok=True)
            np.save(q.with_suffix(".tmp.npy"), mh)
            q.with_suffix(".tmp.npy").replace(q)
        np.save(out.with_suffix(".tmp.npy"), scores)
        out.with_suffix(".tmp.npy").replace(out)
    return budget_loop(head_dir(a.tag), items, fn, work, a.budget, ".npy")


def stage_eou(a, ds, work):
    from audioforge.data import ToneLanguage
    from audioforge.nemo_import import import_eou
    m = import_eou(BT.EOU_NEMO)
    ids = [m.eou_ids["<EOU>"], m.eou_ids["<EOB>"]]
    items = [f"{cid}_ch{k}" for cid in ds.meetings for k in (1, 2)]
    pad = int(BT.PAD_SEC * SR)

    def fn(item, out):
        cid, k = item.rsplit("_ch", 1)
        ch = np.asarray(ds.channels16k(cid)[int(k) - 1], np.float32)
        Tc = ToneLanguage.n_frames(len(ch))
        lp = np.full(Tc, -np.inf)
        em = np.zeros(Tc, bool)
        for s0, s1, keep in segments(len(ch)):
            x = ch[s0: min(len(ch), s1 + pad)]
            _, logpost, emitted = B.rnnt_frame_decode(m, x, watch_ids=ids)
            f0, fk = int(round(s0 / SR / FRAME_SEC)), int(round(keep / SR / FRAME_SEC))
            T = ToneLanguage.n_frames(s1 - s0)
            n = min(T - fk, Tc - (f0 + fk), len(logpost) - fk)
            if n > 0:
                lp[f0 + fk: f0 + fk + n] = logpost[fk: fk + n, 0]
                em[f0 + fk: f0 + fk + n] = emitted[fk: fk + n, 0]
        BT.save_npz(out, logpost=lp, emitted=em)
    return budget_loop("eou", items, fn, work, a.budget, ".npz")


# --------------------------------------------------------------------------- events
def timeout_events(speech: np.ndarray, step: float, k_s: float) -> list[float]:
    """Commit times of 'silence >= k_s after speech' on a per-window speech mask (window length ``step`` s): the end
    of the first silent window at which the silent run reaches k_s (one event per silence run)."""
    out, run, armed = [], 0, False
    need = int(np.ceil(k_s / step - 1e-9))
    for j, sp in enumerate(speech):
        if sp:
            run, armed = 0, True
        else:
            run += 1
            if armed and run == need:
                out.append(round((j + 1) * step, 4))
    return out


def edge_events(score: np.ndarray, emit_s: np.ndarray, theta: float, refractory: float = REFRACTORY) -> list[float]:
    """Rising edges of ``score`` above theta, committed at ``emit_s[t]``, with a refractory period."""
    out, last, above = [], -np.inf, False
    for t in range(len(score)):
        s = score[t]
        if s > theta and not above and emit_s[t] - last >= refractory:
            out.append(round(float(emit_s[t]), 4))
            last = emit_s[t]
        above = s > theta
    return out


def merge_events(a: list[float], b: list[float], refractory: float = REFRACTORY) -> list[float]:
    out, last = [], -np.inf
    for t in sorted(a + b):
        if t - last >= refractory:
            out.append(t)
            last = t
    return out


def strictly_increasing(ts: list[float]) -> list[float]:
    out = []
    for t in ts:
        if not out or t > out[-1]:
            out.append(t)
    return out


class Systems:
    """Per-conversation raw streams (loaded once) -> event lists per (system, operating point)."""

    def __init__(self, ds, work: Path, head: str = "head"):
        self.ds, self.work = ds, work
        self.cids = ds.meetings
        self.vad = {c: np.load(work / "vad" / f"{c}.npz") for c in self.cids}
        self.head = {(c, k): np.load(work / head / f"{c}_spk{k}.npy") for c in self.cids for k in (1, 2)
                     if (work / head / f"{c}_spk{k}.npy").exists()}
        self.eou = {(c, k): np.load(work / "eou" / f"{c}_ch{k}.npz") for c in self.cids for k in (1, 2)
                    if (work / "eou" / f"{c}_ch{k}.npz").exists()}
        self.have_head = len(self.head) == 2 * len(self.cids)
        self.have_eou = len(self.eou) == 2 * len(self.cids)
        self.pc = {(c, k): B.speech_chunks_pipecat(B.pipecat_vad(self.vad[c][f"p{k}"])) for c in self.cids for k in (1, 2)}
        self.pcm = {c: B.speech_chunks_pipecat(B.pipecat_vad(self.vad[c]["pm"])) for c in self.cids}
        self.dur = {c: ds.duration(c) for c in self.cids}

    def emit_frames(self, T):
        return ((np.arange(T) // BD.HEAD_CHUNK + 1) * BD.HEAD_CHUNK) * FRAME_SEC

    def quantiles(self, kind: str) -> np.ndarray:
        src = self.head if kind == "head" else self.eou
        vals = np.concatenate([(v if kind == "head" else v["logpost"]) for v in src.values()])
        vals = vals[np.isfinite(vals)]
        return np.unique(np.quantile(vals, Q_GRID))

    def events(self, system: str, c: str, k: int, pt) -> list[float]:
        if system == "rms_timeout":
            return timeout_events(self.vad[c][f"rms{k}"], RMS_WIN, pt)
        if system == "silero_timeout":
            return timeout_events(self.pc[(c, k)], CHUNK, pt)
        if system == "silero_mix_timeout":
            return timeout_events(self.pcm[c], CHUNK, pt)
        if system == "head_trail6":
            s = self.head[(c, k)]
            return edge_events(s, self.emit_frames(len(s)), pt)
        if system == "eou_posterior":
            s = self.eou[(c, k)]["logpost"]
            return edge_events(np.where(np.isfinite(s), s, -1e9), self.emit_frames(len(s)), pt)
        if system == "eou_native":
            em = self.eou[(c, k)]["emitted"].astype(np.float32)
            return edge_events(em, self.emit_frames(len(em)), 0.5, refractory=0.0)
        if system == "hybrid":
            th, ks = pt
            return merge_events(self.events("head_trail6", c, k, th), self.events("silero_timeout", c, k, ks))
        if system == "head+silero_mix":
            th, ks = pt
            return merge_events(self.events("head_trail6", c, k, th), self.events("silero_mix_timeout", c, k, ks))
        raise KeyError(system)

    def submission(self, system: str, pt):
        from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission
        preds = []
        for c in self.cids:
            sp = {}
            for k in (1, 2):
                ev = [t for t in strictly_increasing(self.events(system, c, k, pt)) if 0 <= t < self.dur[c]]
                sp[k] = SpeakerEvents(eot=ev, interruption=[])
            preds.append(ConversationPrediction(conversation_id=c, speaker_1=sp[1], speaker_2=sp[2]))
        return Submission(schema_version=SCHEMA_VERSION, predictions=preds)


def score_sub(sub, dataset) -> dict:
    from turnbench.score import score_submission
    sc = score_submission(sub, dataset).task_eot
    lat = sc.latency()  # latencies_ms are milliseconds already
    return {"recall": round(float(sc.recall), 4), "fp_rate": round(float(sc.fp_rate), 4), "tp": sc.tp, "fn": sc.fn,
            "fp": sc.fp, "tn": sc.tn, "latency_ms": {q: (None if not np.isfinite(v) else round(float(v))) for q, v in
                                                     (("p10", lat.p10), ("p50", lat.p50), ("p90", lat.p90))}}


def pick(curve: list[dict]) -> dict | None:
    """Their rule: the highest recall at fp_rate <= FP_BUDGET (ties: lower fp, then lower p50)."""
    ok = [r for r in curve if r["score"]["fp_rate"] <= FP_BUDGET]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["score"]["recall"], -r["score"]["fp_rate"], -(r["score"]["latency_ms"]["p50"] or 1e9)))


def stage_sweep_tag(a, ds, work) -> dict:
    """--tag <head> (not trail6): the head alone and head OR Silero-per-channel at the dev operating point (max recall
    at fp <= 0.10, their rule) and at the head's AMI-frozen points (--frozen: bench_dyadic_heads.py --stage points;
    head theta, and (theta, k_frames x 80 ms) of head+silero_timeout fitted on AMI dev at <= 5 % per-turn FC, 6 s)."""
    from turnbench.data import resolve_dataset
    t0 = time.time()
    dataset = resolve_dataset(D.TB_REPO, skip_audio=True)
    S = Systems(ds, work, head_dir(a.tag))
    assert S.have_head, f"missing head scores in {work / head_dir(a.tag)}"
    out = {"scorer": "integrations/turnbench_scorer (SesameAILabs/turnbench, MIT), EOT task", "fp_budget": FP_BUDGET,
           "head": a.tag, "ckpt": str(a.ckpt), "n_conversations": len(S.cids), "systems": {}, "frozen_ami": {}}
    hq = [float(x) for x in S.quantiles("head")]
    hy = [float(x) for x in np.unique(np.quantile(np.concatenate(list(S.head.values())), HYB_Q))]
    grids = {"head_trail6": hq, "hybrid": [(th, ks) for th in hy for ks in HYB_K]}
    for name, grid in grids.items():
        curve = [{"point": pt, "score": score_sub(S.submission(name, pt), dataset)} for pt in grid]
        best = pick(curve)
        disp = f"head_{a.tag}" if name == "head_trail6" else f"hybrid_{a.tag}"
        out["systems"][disp] = {"curve": curve, "operating_point": best}
        b = best["score"] if best else None
        print(f"  {disp:20s} " + (f"recall {b['recall']:.3f} fp {b['fp_rate']:.3f} p50 {b['latency_ms']['p50']} ms at {best['point']}"
                                  if b else "no point within the FP budget") + f" ({time.time() - t0:.0f}s)", flush=True)
    if a.frozen:
        pts = json.loads(Path(a.frozen).read_text())
        th = pts[f"head_{a.tag}"]["6s"][0]
        hth, hk = pts[f"head_{a.tag}+silero_timeout"]["6s"]
        fz = {f"head_{a.tag}": ("head_trail6", th if th is not None else 2.0),
              f"hybrid_{a.tag}": ("hybrid", (hth if hth is not None else 2.0,
                                             (int(np.floor(hk)) + 1) * FRAME_SEC if hk is not None else 1e9))}
        for disp, (name, pt) in fz.items():
            sc = score_sub(S.submission(name, pt), dataset)
            out["frozen_ami"][disp] = {"point": pt, "score": sc}
            print(f"  frozen {disp:20s} recall {sc['recall']:.3f} fp {sc['fp_rate']:.3f} p50 {sc['latency_ms']['p50']} ms "
                  f"at {pt}", flush=True)
    out["sec"] = round(time.time() - t0, 1)
    return out


def stage_sweep(a, ds, work) -> dict:
    from turnbench.data import resolve_dataset
    from turnbench.submission import load_submission
    t0 = time.time()
    dataset = resolve_dataset(D.TB_REPO, skip_audio=True)  # gold columns only (HTTP range reads, cached)
    S = Systems(ds, work)
    out = {"scorer": "integrations/turnbench_scorer (SesameAILabs/turnbench, MIT), EOT task", "fp_budget": FP_BUDGET,
           "n_conversations": len(S.cids), "commit_rule": "timeouts: end of the window completing k s of silence; "
           "head / EOU: rising edge above theta, 2 s refractory, committed at the 160 ms chunk end",
           "head_complete": S.have_head, "eou_complete": S.have_eou, "systems": {}, "published_dev_rescored": {},
           "published_test": PUBLISHED_TEST}
    grids = {"rms_timeout": list(K_GRID), "silero_timeout": list(K_GRID), "silero_mix_timeout": list(K_GRID)}
    if S.have_head:
        hq = S.quantiles("head")
        grids["head_trail6"] = [float(x) for x in hq]
        hy = [float(x) for x in np.unique(np.quantile(np.concatenate(list(S.head.values())), HYB_Q))]
        grids["hybrid"] = [(th, ks) for th in hy for ks in HYB_K]
        grids["head+silero_mix"] = [(th, ks) for th in hy for ks in HYB_K]
    if S.have_eou:
        grids["eou_posterior"] = [float(x) for x in S.quantiles("eou")]
        grids["eou_native"] = [None]
    (work / "preds").mkdir(exist_ok=True)
    for name, grid in grids.items():
        curve = []
        for pt in grid:
            sub = S.submission(name, pt)
            curve.append({"point": pt, "score": score_sub(sub, dataset)})
        best = pick(curve)
        r = {"curve": curve, "operating_point": best}
        if best is not None:
            sub = S.submission(name, best["point"])
            p = work / "preds" / f"{name}.json"
            p.write_text(sub.model_dump_json(indent=1))
            r["predictions"] = str(p)
        out["systems"][name] = r
        b = best["score"] if best else None
        print(f"  {name:20s} " + (f"recall {b['recall']:.3f} fp {b['fp_rate']:.3f} p50 {b['latency_ms']['p50']} ms at {best['point']}"
                                  if b else "no point within the FP budget") + f" ({time.time() - t0:.0f}s)", flush=True)
    for b in DEV_BASELINES:
        p = ROOT / "integrations" / "turnbench_scorer" / "baselines_dev" / f"{b}.json"
        if p.exists():
            out["published_dev_rescored"][b] = score_sub(load_submission(p), dataset)
            print(f"  published {b:28s} " + json.dumps(out["published_dev_rescored"][b]), flush=True)
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["vad", "head", "eou", "sweep"])
    p.add_argument("--work", default=str(BD.SCRATCH / "turnbench" / "work_official"))
    p.add_argument("--budget", type=float, default=500.0)
    p.add_argument("--out", default=None, help="default runs/turnbench_dev.json (trail6) / runs/turnbench_dev_<tag>.json")
    p.add_argument("--tag", default="trail6", help="head name: scores in <work>/head_<tag> (trail6: <work>/head)")
    p.add_argument("--ckpt", default=None, help="stage head: the turn model (default the trail6 checkpoint)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--frozen", default=None, help="sweep --tag: AMI-frozen points JSON (bench_dyadic_heads.py --stage points)")
    a = p.parse_args()
    if a.out is None:
        a.out = str(ROOT / "runs" / ("turnbench_dev.json" if a.tag == "trail6" else f"turnbench_dev_{a.tag}.json"))
    BT._torch2()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ds = ds_all()
    if a.stage == "vad":
        print(stage_vad(a, ds, work))
    elif a.stage == "head":
        print(stage_head(a, ds, work))
    elif a.stage == "eou":
        print(stage_eou(a, ds, work))
    else:
        res = stage_sweep(a, ds, work) if a.tag == "trail6" and not a.frozen else stage_sweep_tag(a, ds, work)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
