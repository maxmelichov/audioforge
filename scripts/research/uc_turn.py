"""User-channel predictive turn head (research/archive/IMPROVEMENTS.md section 1): features, training, inference, TurnBench.

The head (scripts/research/uc_turn_head.py) reads the frozen streaming encoder's top layer computed on the USER'S OWN
channel (the served ASR pass: no extra encoder cost), the user's Silero activity, the agent's activity (the other
party's channel on recorded audio; the TTS timeline in a product), duration counters and the user channel's energy,
and predicts both parties' future voice activity.

Stages (CPU, 2 threads, every process < 10 min and resumable):
  feats  --set oto_train|oto_dev|tb   per conversation: per channel the encoder's top layer ([70, 1] = 160 ms chunks,
         60 s segments with 8 s left overlap, bench_turnbench.segments, float16), Silero v5 probabilities (32 ms) and
         raw log-RMS per 80 ms frame; oto also the label activity (the corpus's Silero segments) as targets.
         -> data/cache/uc_turn/<set>/<id>.npz
  train  --tag T [--hidden --layers --no-energy --vap --steps --lr]   both parties of every oto_train conversation
         as the user in turn (the other = agent); random 40 s crops; checkpoints in <scratch>/uc_turn/<T>.ckpt;
         final head runs/uc_turn_<T>.pt
  infer  --tag T --set tb|oto_tb     continuous per-channel streams (one GRU pass per whole conversation, as served)
         -> <work>/head_uc_<T>/<id>_spk<k>.npy (P(user quiet 2 s)) and <work>/head_uc_<T>__mhbins/<id>_spk<k>.npy
         (columns: user bins 1-5, quiet 1.04 s, quiet 2 s[, VAP user-silent mass])
  eval   --tag T     TurnBench dev with the official gold / scorer (bench_turnbench_latency replica with the TP
         attribution checks): predictive trigger (bins 1-2) and quiet-threshold families, alone and OR the user
         channel's Silero timeout; fitted (i) on one TurnBench half -> the other (pooled), (ii) on oto dev (the 6 dev
         conversations of DYADIC section 7) -> all of TurnBench. -> runs/uc_turn.json

  PYTHONPATH=. .venv/bin/python scripts/research/uc_turn.py feats --set oto_train      # repeat until "0 left"
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

FEATS = ROOT / "data" / "cache" / "uc_turn"
SCRATCH = Path("/private/tmp/claude-501/-Users-maxm/54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad")
CKPT_DIR = SCRATCH / "uc_turn"
OUT = ROOT / "runs" / "uc_turn.json"
ENC_MODEL = ROOT / "runs" / "stage1_served.afm"  # the served model: its encoder is the frozen NVIDIA 115M streaming one
SR = 16000
FRAME = 0.08
ATT = [70, 1]


# --------------------------------------------------------------------------- ids
def oto_train_ids() -> list[str]:
    import bench_dyadic_heads as BH
    from audioforge.datasets import dyadic as D
    return D.split_ids(D.list_ids("oto")[: BH.SPLIT["n_conv"]], "train", BH.SPLIT["split_mod"], BH.SPLIT["dev_res"])


def oto_dev_ids() -> list[str]:
    """The dev conversations we need: the 6 of DYADIC section 7's set (oto -> TurnBench fits) + the 16 E2E_FINAL oto
    clip conversations (live dead-air clips). All are dev conversations of the dyadic split (never trained on)."""
    import bench_dyadic_heads as BH
    ids = list(BH.oto_tb_dev_ids())
    e2e = json.loads((ROOT / "runs" / "e2e_final.json").read_text())["clips"]
    ids += [v["conversation"] for v in e2e.values() if v["set"] == "oto" and v["conversation"] not in ids]
    dev = set(BH.dev_ids())
    assert all(c in dev for c in ids)
    return ids


def tb_ids() -> list[str]:
    from audioforge.datasets import dyadic as D
    return D.list_ids("turnbench")


def set_ids(name: str) -> list[str]:
    return {"oto_train": oto_train_ids, "oto_dev": oto_dev_ids, "tb": tb_ids}[name]()


# --------------------------------------------------------------------------- features
def channels(set_name: str, cid: str) -> np.ndarray:
    """(2, n) float32 16 kHz channels."""
    from audioforge.datasets import dyadic as D
    if set_name == "tb":
        ds = D.Dyadic([cid], "turnbench", verbose=False)
        return np.asarray(ds.channels16k(cid), np.float32)
    x, sr, _ = D.read_stereo("oto", D.OTO_ROOT, cid)
    ch = np.asarray(D.resample16k(x, sr), np.float32)
    return ch if ch.shape[0] == 2 else ch.T


def encode_channel(model, x: np.ndarray, T: int) -> np.ndarray:
    import bench_turnbench as TB
    import torch
    enc = np.zeros((T, model.encoder.d_model if hasattr(model.encoder, "d_model") else 512), np.float16)
    with torch.no_grad():
        for s0, s1, keep in TB.segments(len(x)):
            seg = torch.from_numpy(np.ascontiguousarray(x[s0:s1]))[None]
            e, el, *_ = model.encode(seg, torch.tensor([seg.shape[1]]), att_context_size=ATT)
            e = e[0, : int(el[0])].numpy()
            f0 = int(round(s0 / SR / FRAME))
            fk = int(round(keep / SR / FRAME))
            n = min(len(e) - fk, T - (f0 + fk))
            if n > 0:
                enc[f0 + fk: f0 + fk + n] = e[fk: fk + n]
    return enc


def label_frames(cid: str, T: int) -> np.ndarray:
    """(2, T) label activity of an oto conversation (the corpus labels: Silero segments per channel)."""
    from audioforge.datasets import dyadic as D
    from audioforge.datasets.ami import activity, frames
    lab = json.loads((D.OTO_ROOT / "cache" / "labels" / f"{cid}.json").read_text())
    return np.stack([frames(activity([tuple(w) for w in lab["words"][str(c)]]), T, 0.0) for c in (0, 1)])


def stage_feats(a):
    import bench_turn_dyadic as BD
    from audioforge.baselines import turn as B
    from audioforge.data import ToneLanguage
    from uc_turn_head import frame_log_rms_np
    from audioforge.train import load_model
    out_dir = FEATS / a.set
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = set_ids(a.set)
    todo = [c for c in ids if not (out_dir / f"{c}.npz").exists()]
    if not todo:
        print("  feats: 0 left")
        return
    model = load_model(str(ENC_MODEL), "cpu").eval()
    vad = B.SileroVAD(BD.silero_path())
    t0, n = time.time(), 0
    for cid in todo:
        if time.time() - t0 > a.budget:
            break
        ch = channels(a.set, cid)
        T = ToneLanguage.n_frames(ch.shape[1])
        z = {}
        for k in (0, 1):
            z[f"enc{k}"] = encode_channel(model, ch[k], T)
            z[f"p{k}"] = vad.probs(ch[k]).astype(np.float32)
            z[f"rms{k}"] = frame_log_rms_np(ch[k], T).astype(np.float32)
        if a.set.startswith("oto"):
            z["lab"] = label_frames(cid, T)
        tmp = out_dir / f"{cid}.tmp.npz"
        np.savez(tmp, **z)
        tmp.replace(out_dir / f"{cid}.npz")
        n += 1
        print(f"  {cid}: {ch.shape[1] / SR / 60:.1f} min ({time.time() - t0:.0f}s)", flush=True)
    print(f"  feats {a.set}: {n} done in {time.time() - t0:.0f}s, {len(todo) - n} left", flush=True)


# --------------------------------------------------------------------------- training
def frame_track(p: np.ndarray, T: int) -> np.ndarray:
    import bench_turnbench as TB
    return TB.frame_track(p, T)


class Pool:
    """All oto_train conversations in memory; items = (conversation, user channel)."""

    def __init__(self, set_name: str = "oto_train", ids=None):
        self.items = []
        for cid in ids or set_ids(set_name):
            p = FEATS / set_name / f"{cid}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            T = z["enc0"].shape[0]
            act = np.stack([frame_track(z["p0"], T), frame_track(z["p1"], T)])
            conv = dict(enc=[z["enc0"], z["enc1"]], act=act, rms=[z["rms0"], z["rms1"]], lab=z["lab"], T=T)
            for k in (0, 1):
                self.items.append((conv, k))
        self.frames = sum(c["T"] for c, k in self.items)

    def batch(self, rng: random.Random, bs: int, W: int, aug: bool = True):
        import torch
        enc, act, rms, tu, ta, L = [], [], [], [], [], []
        weights = [c["T"] for c, _ in self.items]
        for _ in range(bs):
            conv, k = rng.choices(self.items, weights)[0]
            T = conv["T"]
            s = rng.randrange(0, max(1, T - W))
            e = min(T, s + W)
            a_u = conv["act"][k, s:e].copy()
            a_a = conv["act"][1 - k, s:e].copy()
            if aug:
                r = rng.random()
                if r < 0.3:  # exact agent timeline (a product's TTS state)
                    a_a = conv["lab"][1 - k, s:e].astype(np.float32)
                elif r < 0.6:
                    a_a = (a_a > 0.5).astype(np.float32)
            n = e - s
            pad = W - n
            enc.append(np.pad(conv["enc"][k][s:e].astype(np.float32), ((0, pad), (0, 0))))
            act.append(np.pad(np.stack([a_u, a_a], -1), ((0, pad), (0, 0))))
            rms.append(np.pad(conv["rms"][k][s:e], (0, pad)))
            tu.append(np.pad(conv["lab"][k, s:e], (0, pad)))
            ta.append(np.pad(conv["lab"][1 - k, s:e], (0, pad)))
            L.append(n)
        f = lambda x: torch.from_numpy(np.stack(x))  # noqa: E731
        Lt = torch.tensor(L)
        valid = torch.arange(W)[None] < Lt[:, None]
        return f(enc), f(act).float(), f(rms).float(), valid, f(tu).float(), f(ta).float(), Lt


def head_cfg(a) -> dict:
    return dict(hidden=a.hidden, layers=a.layers, energy=not a.no_energy, vap=a.vap, dropout=a.dropout)


def stage_train(a):
    import torch
    from uc_turn_head import UCTurnHead, save_head
    torch.set_num_threads(2)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ck = CKPT_DIR / f"{a.tag}.ckpt"
    torch.manual_seed(a.seed)
    head = UCTurnHead(**head_cfg(a))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    step, log = 0, []
    if ck.exists():
        s = torch.load(ck, weights_only=False)
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        step, log = s["step"], s["log"]
        assert s["cfg"] == head.cfg, "checkpoint config differs"
    if step >= a.steps:
        print(f"  train {a.tag}: done ({step} steps)")
        return
    t_load = time.time()
    pool = Pool()
    print(f"  pool: {len(pool.items)} items, {pool.frames * FRAME / 3600:.1f} h ({time.time() - t_load:.0f}s)", flush=True)
    rng = random.Random(a.seed * 100003 + step)
    sched = lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))  # noqa: E731
    t0 = time.time()
    head.train()
    acc = {}
    while step < a.steps and time.time() - t0 < a.budget:
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        enc, act, rms, valid, tu, ta, L = pool.batch(rng, a.batch_size, a.window)
        out, _ = head(enc, act, rms, valid)
        loss, parts = head.loss(out, tu, ta, L)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % 100 == 0:
            row = {"step": step, **{k: round(v / 100, 4) for k, v in acc.items()}, "t": round(time.time() - t0)}
            log.append(row)
            acc = {}
            print(f"  {row}", flush=True)
    torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "log": log, "cfg": head.cfg}, ck)
    if step >= a.steps:
        head.eval()
        save_head(head, ROOT / "runs" / f"uc_turn_{a.tag}.pt",
                  {"tag": a.tag, "steps": step, "log": log[-5:], "train_hours": round(pool.frames * FRAME / 3600, 2),
                   "encoder": str(ENC_MODEL.name), "att_context": ATT, "args": vars(a)})
        print(f"  saved runs/uc_turn_{a.tag}.pt")
    print(f"  train {a.tag}: step {step}/{a.steps}")


# --------------------------------------------------------------------------- inference
def run_head(head, enc, act, rms):
    """One continuous causal pass -> dict of (T, k) probabilities."""
    import torch
    with torch.no_grad():
        out, _ = head(torch.from_numpy(enc.astype(np.float32))[None], torch.from_numpy(act.astype(np.float32))[None],
                      torch.from_numpy(rms.astype(np.float32))[None])
        return {k: v.numpy() for k, v in head.probs(out).items()}


def mh_columns(p: dict) -> np.ndarray:
    cols = [p["user_bins"], p["user_quiet"]]
    if "p_user_silent_vap" in p:
        cols.append(p["p_user_silent_vap"])
    return np.concatenate(cols, -1).astype(np.float32)


def stage_infer(a):
    import bench_dyadic_heads as BH
    import bench_turnbench_latency as BL
    import torch
    from uc_turn_head import load_head
    torch.set_num_threads(2)
    head = load_head(ROOT / "runs" / f"uc_turn_{a.tag}.pt")
    if a.set == "tb":
        work, fset, ids = BL.WORK_TB, "tb", tb_ids()
    else:
        work, fset, ids = BH.OTO_TB_WORK, "oto_dev", BH.oto_tb_dev_ids()
    hd, md = work / f"head_uc_{a.tag}", work / f"head_uc_{a.tag}__mhbins"
    hd.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for cid in ids:
        z = np.load(FEATS / fset / f"{cid}.npz")
        T = z["enc0"].shape[0]
        vz = np.load(work / "vad" / f"{cid}.npz")  # the benchmark's own Silero streams (identical to z["p*"])
        tr = [frame_track(vz["p1"], T), frame_track(vz["p2"], T)]
        for k in (1, 2):
            u = k - 1
            act = np.stack([tr[u], tr[1 - u]], -1)
            p = run_head(head, z[f"enc{u}"], act, z[f"rms{u}"])
            np.save(hd / f"{cid}_spk{k}.npy", p["user_quiet"][:, -1].astype(np.float32))
            np.save(md / f"{cid}_spk{k}.npy", mh_columns(p))
    print(f"  infer {a.tag} {a.set}: {len(ids)} conversations in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- evaluation
P_GRID = (0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05,
          0.07, 0.1, 0.15, 0.2, 0.3)
P_GRID_OR = (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05, 0.1)


def _rise_events(ch, cond):
    import bench_turnbench_latency as BL
    rises = np.nonzero(cond & ~np.concatenate([[False], cond[:-1]]))[0]
    out, last = [], -np.inf
    for t in rises:
        if ch.emit[t] - last >= BL.REFRACTORY:
            out.append(round(float(ch.emit[t]), 4))
            last = ch.emit[t]
    return out


def evaluate(tag: str, fams_sel=None) -> dict:
    import bench_dyadic_heads as BH
    import bench_turnbench_latency as BL
    tb, tb_mh, tb_sp = BH._corpus("tb", f"uc_{tag}")
    oto, oto_mh, _ = BH._corpus("oto", f"uc_{tag}")
    allq = np.concatenate([tb_mh[id(ch)][:, 6] for ch in tb.chans])
    q_grid = sorted(set(float(x) for x in np.quantile(allq, np.concatenate([np.linspace(0.5, 0.95, 10),
                                                                              np.linspace(0.955, 0.9995, 30)]))))
    fams = {"predictive": [(t1, t2, None) for t1 in P_GRID for t2 in P_GRID],
            "predictive_or_silero": [(t1, t2, k) for t1 in P_GRID_OR for t2 in P_GRID_OR for k in BL.K_GRID],
            "quiet2s": [(th, None) for th in q_grid],
            "quiet2s_or_silero": [(th, k) for th in q_grid for k in BL.K_GRID],
            "silero": [(float("inf"), k) for k in BL.K_GRID]}
    if fams_sel:
        fams = {k: v for k, v in fams.items() if k in fams_sel}

    def fn_of(fam, pt, mh):
        def fn(ch):
            m = mh[id(ch)]
            k = pt[-1]
            sil = ch.silence_events(np.full(len(ch.s), k)) if k is not None else []
            if fam.startswith("predictive"):
                ev = _rise_events(ch, (m[:, 0] < pt[0]) & (m[:, 1] < pt[1]))
            elif np.isfinite(pt[0]):
                ev = _rise_events(ch, m[:, 6] > pt[0])
            else:
                ev = []
            return BL.merged(ev, sil, ch.dur)
        return fn

    def fit(corpus, mh, cids, grid, fam):
        best = None
        for pt in grid:
            m = BL.aggregate(corpus.score(fn_of(fam, pt, mh)), cids)
            if m["fp_rate"] <= BL.FP_BUDGET and (best is None or (m["recall"], -(m["p50"] or 1e9)) >
                                                 (best[1]["recall"], -(best[1]["p50"] or 1e9))):
                best = (pt, m)
        return best

    def fit_fast(corpus, mh, cids, grid, fam):
        """Lowest P50 with recall >= 0.83 at fp <= 0.10 (the IMPROVEMENTS.md target rule)."""
        best = None
        for pt in grid:
            m = BL.aggregate(corpus.score(fn_of(fam, pt, mh)), cids)
            if m["fp_rate"] <= BL.FP_BUDGET and m["recall"] >= 0.83 and m["p50"] is not None and (
                    best is None or m["p50"] < best[1]["p50"]):
                best = (pt, m)
        return best

    A, Bh = BL.tb_halves(tb.cids)
    out = {"tag": tag, "families": {}}
    for fam, grid in fams.items():
        r = {}
        for rule, fitter in (("max_recall", fit), ("fast_083", fit_fast)):
            rr = {}
            f = fitter(oto, oto_mh, oto.cids, grid, fam)
            if f:
                rr["oto->tb"] = {"point": f[0], "fit": f[1], "held_out": BH._metrics(tb, fn_of(fam, f[0], tb_mh), tb.cids, tb_sp)}
            halves = {}
            for name, fh, eh in (("tbA->tbB", A, Bh), ("tbB->tbA", Bh, A)):
                fr = fitter(tb, tb_mh, fh, grid, fam)
                if fr:
                    halves[name] = {"point": fr[0], "fit": fr[1], "held_out": BH._metrics(tb, fn_of(fam, fr[0], tb_mh), eh, tb_sp)}
            rr.update(halves)
            if len(halves) == 2:
                per = {}
                for name, eh in (("tbA->tbB", Bh), ("tbB->tbA", A)):
                    per.update({c: v for c, v in tb.score(fn_of(fam, halves[name]["point"], tb_mh)).items() if c in eh})
                pm = BL.aggregate(per, tb.cids)
                lat = np.concatenate([per[c][4] for c in tb.cids])
                pm["tp_before_end"] = round(float((lat < 0).mean()), 3) if len(lat) else None
                rr["tb_halves_pooled"] = pm
            r[rule] = rr
            pr = rr.get("tb_halves_pooled") or {}
            print(f"  {fam} [{rule}]: pooled held-out recall {pr.get('recall')} fp {pr.get('fp_rate')} P10 {pr.get('p10')} "
                  f"P50 {pr.get('p50')} | oto->tb {rr.get('oto->tb', {}).get('held_out', {})}", flush=True)
        out["families"][fam] = r
    return out


def stage_eval(a):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    r = evaluate(a.tag, a.families.split(",") if a.families else None)
    res.setdefault("turnbench", {})[a.tag] = r
    OUT.write_text(json.dumps(res, indent=1, default=float))


# --------------------------------------------------------------------------- E2E_FINAL clips (offline, live logic)
E2E_WORK = SCRATCH / "e2e_final"


def clip_decisions(head, silero, user: np.ndarray, other: np.ndarray, model, t1, t2, k_s, rule="predictive",
                   theta=0.5, agent="pipecat") -> dict:
    """The live UCTurnStream on one two-party clip: user channel audio + encoder frames (offline [70, 1] encode from
    the clip start == the streaming pass), agent state = the other channel's Pipecat-VAD speech state changes
    (committed at the 32 ms chunk end: the TTS timeline stand-in). Returns responses (decision times) and sources."""
    import torch
    from audioforge.baselines import turn as B
    from audioforge.data import ToneLanguage
    from uc_stream import UCTurnStream
    T = ToneLanguage.n_frames(len(user))
    enc = torch.from_numpy(encode_channel(model, user, T).astype(np.float32))[None]
    st = UCTurnStream(head, silero, t1=t1, t2=t2, k_s=k_s, rule=rule, theta=theta)
    po = B.SileroVAD(silero.path).probs(other)
    sp = B.speech_chunks_pipecat(B.pipecat_vad(po))
    prev = False
    for j, s in enumerate(sp):
        if bool(s) != prev:
            st.set_agent(bool(s), (j + 1) * B.CHUNK_SEC)
            prev = bool(s)
    out, step = [], 2560
    v = 0
    for pos in range(0, len(user), step):
        st.feed_audio(user[pos: pos + step])
        ready = min(T, (pos + step) // 1280)
        if ready > v:
            st.feed_enc(enc[:, v:ready])
            v = ready
        out += st.poll()
    st.end()
    out += st.poll()
    return {"responses": [t for t, _, _ in out], "sources": [s for _, s, _ in out]}


def stage_clips(a):
    import soundfile as sf
    import torch
    import bench_turn_dyadic as BD
    from audioforge.baselines import turn as B
    from audioforge.e2e_metrics import pooled, score_clip
    from uc_turn_head import load_head
    from audioforge.train import load_model
    torch.set_num_threads(2)
    head = load_head(ROOT / "runs" / f"uc_turn_{a.tag}.pt")
    model = load_model(str(ENC_MODEL), "cpu").eval()
    silero = B.SileroStream(BD.silero_path())
    refs = {r["name"]: r for r in json.loads((E2E_WORK / "clips.json").read_text())}
    k_s = None if a.k < 0 else a.k
    res = {"tag": a.tag, "point": {"rule": a.rule, "t1": a.t1, "t2": a.t2, "theta": a.theta, "k_s": k_s}, "sets": {}}
    per = {}
    for n, r in refs.items():
        if r["set"] == "ami":
            continue
        user = sf.read(str(E2E_WORK / "clips" / f"{n}.user.wav"), dtype="float32")[0]
        mono = sf.read(str(E2E_WORK / "clips" / f"{n}.mono.wav"), dtype="float32")[0]
        other = np.clip(mono - user, -1, 1)
        d = clip_decisions(head, silero, user, other, model, a.t1, a.t2, k_s, a.rule, a.theta)
        s = score_clip(d["responses"], [tuple(t) for t in r["user_turns"]], scored=r["scored"],
                       user_intervals=r["user_intervals"])
        s["audio_s"] = len(user) / SR
        per[n] = (r["set"], s, d)
    for st in ("turnbench", "oto"):
        sc = [s for (ss, s, _) in per.values() if ss == st]
        m = pooled(sc)
        m["n_clips"] = len(sc)
        res["sets"][st] = m
        print(f"  {st}: dead air P50 {m.get('dead_air_ms_median')} P90 {m.get('dead_air_ms_p90')} missed_3s "
              f"{m.get('missed_3s')} cut-ins {m.get('cut_ins')} (n ends {m.get('n_ends')})", flush=True)
    res["per_clip"] = {n: {"responses": d["responses"], "sources": d["sources"], "cut_ins": s["cut_ins"]}
                       for n, (_, s, d) in per.items()}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("clips", {})[f"{a.tag}|{a.rule}|{a.t1}|{a.t2}|{a.theta}|{k_s}"] = res
    OUT.write_text(json.dumps(out, indent=1, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["feats", "train", "infer", "eval", "clips"])
    p.add_argument("--set", default="oto_train")
    p.add_argument("--tag", default="v1")
    p.add_argument("--budget", type=float, default=520.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--vap", action="store_true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="")
    p.add_argument("--rule", default="predictive", choices=["predictive", "quiet"])
    p.add_argument("--t1", type=float, default=0.01)
    p.add_argument("--t2", type=float, default=0.01)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--k", type=float, default=-1.0, help="Silero timeout s for the OR (-1: none)")
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    {"feats": stage_feats, "train": stage_train, "infer": stage_infer, "eval": stage_eval,
     "clips": stage_clips}[a.stage](a)


if __name__ == "__main__":
    main()
