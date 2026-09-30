"""Latency of the TurnBench decision policy fitted on DYADIC data (research/archive/DYADIC.md section 7): the per-channel
inputs of DYADIC.md section 4 (user-channel Silero v5 + Pipecat state machine; trail6 head on the mixed mono with the
user's channel as the party track) under four policy families, fitted on otoSpeech or on one half of TurnBench dev and
scored on TurnBench dev with the OFFICIAL scorer's gold and matching (integrations/turnbench_scorer, unchanged).

Policies (per speaker channel; one silence-branch event per user silence run, head-branch events = rising edges of the
emitted head posterior above theta; the two merged with the scorer's 2 s refractory):
  fixed   head p > theta OR silence >= k                                       (the DYADIC.md hybrid; reference)
  lin     head p > theta OR silence >= clamp(T0 - a * p, Tmin, T0)             (BASELINES.md dynamic timeout)
  exp     phat > theta OR silence >= Tmin + (Tmax - Tmin) * exp(-lam * phat**gam), phat = isotonic(p) fitted on the
          FITTING split only (a fixed monotone map: phat[t] depends on p[t] alone, so the policy stays causal)
  vel     exp + early exit at Tmin: if phat > 0.85 and phat rose by > 0.5 within the last 2 head frames at any time
          during the current silence run, fire at max(that time, silence start + Tmin)
p at time tau = the head score of the latest 80 ms frame emitted by tau (160 ms encoder chunk: frame t is emitted at
((t // 2) + 1) * 160 ms); silence = time since the end of the last Pipecat-VAD speech chunk on the user's channel.

otoSpeech gold (no human TurnBench labels exist for this corpus): the scorer's OWN floor construction
(turnbench.gold.build_conversation_events) applied to our per-channel Silero segments, with every speech run
(segments merged across gaps < --oto-gap s) labelled Turn and runs <= 1 s bounded by >= 1 s of own silence on both
sides labelled Backchannel; redactions are excluded intervals. Scored with turnbench.score.score_task.

Stages (CPU, 2 threads, each process <= 10 min, resumable):
  oto_vad   Silero per channel on the 16 oto conversations of runs/dyadic_bench_oto.json   (-> <work_oto>/vad)
  oto_head  trail6 head per (conversation, speaker) exactly as bench_turnbench.stage_head   (-> <work_oto>/head)
  sweep     policy grids x {oto, TurnBench dev, TurnBench halves}, fits, held-out scores    (-> runs/turnbench_latency.json)
  v2        the chosen points through eot-bench v2 (bench_turn_dyadic windows, streams sliced from the whole
            conversations) on the oto and TurnBench-derived sets                            (merged into the same JSON)
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
import bench_turnbench as TB  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402
from audioforge.datasets import dyadic as D  # noqa: E402
from audioforge.datasets.ami import FRAME_SEC, SR  # noqa: E402

CHUNK = B.CHUNK_SEC  # 32 ms
HEAD_EMIT_FRAMES = BD.HEAD_CHUNK  # 2 frames = 160 ms
REFRACTORY = TB.REFRACTORY
FP_BUDGET, MIN_RECALL = 0.10, 0.80
WORK_TB = BD.SCRATCH / "turnbench" / "work_official"
WORK_OTO = BD.SCRATCH / "oto" / "work_tb"
OUT = ROOT / "runs" / "turnbench_latency.json"


def oto_ids() -> list[str]:
    d = json.loads((ROOT / "runs" / "dyadic_bench_oto.json").read_text())
    ids = [c for f in d["folds"] for c in f]
    order = D.list_ids("oto")
    return sorted(ids, key=order.index)


def oto_ds():
    ds = D.Dyadic(oto_ids(), "oto", verbose=False)

    def channels16k(m):  # in memory: no 1 GB channel cache on a full disk
        x, sr, _ = D.read_stereo("oto", ds.root, m)
        return D.resample16k(x, sr)
    ds.channels16k = channels16k
    return ds


# --------------------------------------------------------------------------- gold
BC_MAX_SEC, BC_RUN_GAP, BC_MAX_SEGS = 1.0, 0.5, 2  # DYADIC.md's oto backchannel rule (datasets/ami.speaker_turns)


def oto_segments(segs: list, gap: float = 0.0) -> tuple[list, list]:
    """One channel's VAD segments [(s, e), ...] -> (Turn segments, Backchannel segments). A run = segments separated
    by < BC_RUN_GAP s; a run of <= BC_MAX_SEGS segments lasting <= BC_MAX_SEC is a backchannel (DualTurn / our oto
    rule). Turn segments of the other runs are merged across gaps < ``gap`` (0 = the VAD segmentation as is, like
    TurnBench's VAD-segmented annotation)."""
    segs = sorted((float(a), float(b)) for a, b, *_ in segs)
    runs, cur = [], []
    for s, e in segs:
        if cur and s - cur[-1][1] >= BC_RUN_GAP:
            runs.append(cur)
            cur = []
        cur.append((s, e))
    if cur:
        runs.append(cur)
    turn, bc = [], []
    for r in runs:
        if len(r) <= BC_MAX_SEGS and r[-1][1] - r[0][0] <= BC_MAX_SEC:
            bc.extend(r)
        else:
            for s, e in r:
                if turn and gap > 0 and s - turn[-1][1] < gap:
                    turn[-1] = (turn[-1][0], e)
                else:
                    turn.append((s, e))
    # a Turn run may start after a backchannel of the same speaker within BC_RUN_GAP only across runs: keep sorted
    return sorted(turn), sorted(bc)


def oto_gold(words: dict, zones: list, gap: float = 0.0):
    """The scorer's floor construction (turnbench.gold.build_conversation_events) on our per-channel VAD segments:
    speaker k = channel k - 1; redactions -> excluded intervals on both speakers."""
    from turnbench.gold import ConsensusEvent, ConsensusViews, Interval, build_conversation_events
    turn_ev, lab_ev = [], []
    for k in (1, 2):
        t, b = oto_segments(words[k - 1], gap)
        turn_ev += [ConsensusEvent(k, round(s, 4), round(e, 4), "Turn") for s, e in t]
        lab_ev += [ConsensusEvent(k, round(s, 4), round(e, 4), "Turn") for s, e in t]
        lab_ev += [ConsensusEvent(k, round(s, 4), round(e, 4), "Backchannel") for s, e in b]
    excl = [Interval(k, float(a), float(b)) for a, b in zones for k in (1, 2)]
    return build_conversation_events(ConsensusViews(turn_ev, excl, lab_ev, []))


def tb_gold() -> dict:
    """{conversation id: turnbench.gold.ConversationEvents} of TurnBench dev (the scorer's own gold)."""
    from turnbench.data import conversation, conversation_ids, resolve_dataset
    from turnbench.gold import events_for_conversation
    ds = resolve_dataset(D.TB_REPO, skip_audio=True)
    return {c: events_for_conversation(conversation(ds, c)) for c in conversation_ids(ds)}


def oto_gold_all(ds, gap: float = 0.0) -> dict:
    return {m: oto_gold([ds.words[m][f"{m}:{c}"] for c in (0, 1)], ds.zones[m], gap) for m in ds.meetings}


def gold_stats(G: dict) -> dict:
    neg = np.array([s.end - s.start for g in G.values() for s in g.eot_negative_spans])
    npos = sum(len(g.eot_positive_events) for g in G.values())
    return {"conversations": len(G), "eot_pos": int(npos), "eot_neg": int(len(neg)),
            "neg_per_pos": round(len(neg) / max(npos, 1), 3),
            "neg_span_s_p10_p50_p90": [round(float(x), 2) for x in np.percentile(neg, [10, 50, 90])] if len(neg) else None}


# --------------------------------------------------------------------------- policy engine
THETAS = (0.9985, 0.9988, 0.999, 0.9992, 0.9994, 0.9996, float("inf"))  # head branch (raw p; inf = off)
K_GRID = (0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0)
LIN_T0 = (1.0, 1.5, 2.0, 2.5, 3.0)
LIN_A = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5)  # seconds per unit of p
LIN_TMIN = (0.2, 0.3, 0.5, 0.7)
EXP_TMIN = (0.2, 0.3, 0.4, 0.5, 0.7)
EXP_TMAX = (1.2, 1.5, 2.0, 2.5, 3.0)
EXP_LAM = (1.0, 2.0, 3.0, 5.0, 8.0)
EXP_GAM = (0.5, 1.0, 2.0, 4.0)
VEL_P, VEL_RISE, VEL_FRAMES = 0.85, 0.5, 2  # the task's velocity rule
VEL_GRID_RISE, VEL_GRID_FRAMES = (0.5, 0.3, 0.2, 0.1), (2, 4, 6)  # sensitivity: looser rules on the chosen exp point


def exp_wait(phat, tmin: float, tmax: float, lam: float, gam: float):
    """Bounded-exponential required silence (s): Tmin + (Tmax - Tmin) * exp(-lam * phat**gam); phat in [0, 1]."""
    phat = np.clip(np.asarray(phat, np.float64), 0.0, 1.0)
    return tmin + (tmax - tmin) * np.exp(-lam * phat ** gam)


def lin_wait(p, t0: float, a: float, tmin: float):
    """BASELINES.md linear clamp: clamp(T0 - a * p, Tmin, T0) (s)."""
    return np.clip(t0 - a * np.asarray(p, np.float64), tmin, t0)


class Isotonic:
    """Monotone calibration p -> phat fitted with sklearn's IsotonicRegression (increasing, clipped out of range) and
    applied pointwise by linear interpolation of its knots: phat[t] = f(p[t]) reads nothing but p[t], so the
    calibrated stream is exactly as causal as the raw one."""

    def __init__(self, x=None, y=None):
        self.x = None if x is None else np.asarray(x, np.float64)
        self.y = None if y is None else np.asarray(y, np.float64)

    def fit(self, p, lab, weight=None):
        from sklearn.isotonic import IsotonicRegression
        p, lab = np.asarray(p, np.float64), np.asarray(lab, np.float64)
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p, lab, sample_weight=weight)
        self.x, self.y = np.asarray(ir.X_thresholds_, np.float64), np.asarray(ir.y_thresholds_, np.float64)
        return self

    def __call__(self, p):
        return np.interp(np.asarray(p, np.float64), self.x, self.y)

    def to_json(self, max_knots: int = 200) -> dict:
        j = np.unique(np.linspace(0, len(self.x) - 1, min(max_knots, len(self.x))).astype(int))
        return {"n_knots": int(len(self.x)), "x": [float(v) for v in self.x[j]], "y": [round(float(v), 5) for v in self.y[j]]}


def rise(phat_frames: np.ndarray, n: int = VEL_FRAMES) -> np.ndarray:
    """phat[t] - min(phat[t-n .. t-1]) (0 for t < 1): the rise within the last n head frames (causal)."""
    x = np.asarray(phat_frames, np.float64)
    out = np.zeros(len(x))
    for d in range(1, n + 1):
        prev = np.concatenate([np.full(d, np.inf), x[:-d]]) if len(x) > d else np.full(len(x), np.inf)
        out = np.maximum(out, np.where(np.isfinite(prev), x - prev, 0.0))
    return out


class Chan:
    """One speaker channel: the silent chunks of its Pipecat-VAD stream (after the first speech) with the silence
    duration, silence-run id, chunk end time and the index of the latest emitted head frame."""

    def __init__(self, cid: str, k: int, speech: np.ndarray, head: np.ndarray, dur: float):
        self.cid, self.k, self.head, self.dur = cid, k, np.asarray(head, np.float64), dur
        T = len(head)
        self.emit = ((np.arange(T) // HEAD_EMIT_FRAMES + 1) * HEAD_EMIT_FRAMES) * FRAME_SEC
        sp = np.asarray(speech, bool)
        idx = np.arange(len(sp))
        last = np.maximum.accumulate(np.where(sp, idx, -1))  # last speech chunk at or before j
        sil = (~sp) & (last >= 0)
        j = idx[sil]
        self.tau = (j + 1) * CHUNK
        self.s = self.tau - (last[sil] + 1) * CHUNK
        self.run = last[sil]
        m = np.floor(self.tau / (HEAD_EMIT_FRAMES * FRAME_SEC) + 1e-9).astype(np.int64)
        self.fidx = np.minimum(HEAD_EMIT_FRAMES * m - 1, T - 1)  # latest emitted frame (-1: none yet)
        self.p = np.where(self.fidx >= 0, self.head[np.maximum(self.fidx, 0)], 0.0)
        self._edges = {}

    def calibrated(self, cal, frames=(VEL_FRAMES,)):
        """(phat at the silent chunks, {n: rise of phat over the last n frames at the silent chunks})."""
        ph = cal(self.head)
        ok = self.fidx >= 0
        f = np.maximum(self.fidx, 0)
        return np.where(ok, ph[f], 0.0), {n: np.where(ok, rise(ph, n)[f], 0.0) for n in frames}

    def first_per_run(self, cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(run ids, positions in the silent-chunk arrays) of the first chunk of each run where cond holds."""
        pos = np.nonzero(cond)[0]
        runs, first = np.unique(self.run[pos], return_index=True)
        return runs, pos[first]

    def silence_events(self, wait: np.ndarray, vel: np.ndarray | None = None, tmin: float | None = None) -> np.ndarray:
        """Commit times of the silence branch: per silence run, the first chunk with silence >= wait (s, per chunk);
        with ``vel`` (bool per chunk), also max(first vel chunk, first chunk with silence >= tmin) of that run."""
        runs, pos = self.first_per_run(self.s >= wait - 1e-9)
        if vel is not None and vel.any():
            vr, vp = self.first_per_run(vel)
            mr, mp = self.first_per_run(self.s >= tmin - 1e-9)
            mm = dict(zip(mr.tolist(), mp.tolist()))
            best = dict(zip(runs.tolist(), pos.tolist()))
            for r, p in zip(vr.tolist(), vp.tolist()):
                if r in mm:
                    q = max(p, mm[r])  # positions of one run are contiguous and s increases along them
                    best[r] = min(best.get(r, q), q)
            pos = np.array(sorted(best.values()), np.int64)
        return self.tau[pos]

    def head_events(self, theta: float) -> list:
        if not np.isfinite(theta):
            return []
        if theta not in self._edges:
            ab = self.head > theta
            rises = np.nonzero(ab & ~np.concatenate([[False], ab[:-1]]))[0]
            out, last = [], -np.inf
            for t in rises:
                if self.emit[t] - last >= REFRACTORY:
                    out.append(round(float(self.emit[t]), 4))
                    last = self.emit[t]
            self._edges[theta] = out
        return self._edges[theta]


def merged(head_ev: list, sil_ev, dur: float) -> tuple[list, list]:
    """OR of the two branches with the 2 s refractory (bench_turnbench.merge_events), kept in [0, dur) and strictly
    increasing; returns (times, source per time: 'head' | 'silence')."""
    ev = sorted([(t, 0) for t in head_ev] + [(round(float(t), 4), 1) for t in sil_ev])
    out, src, last = [], [], -np.inf
    for t, s in ev:
        if t - last >= REFRACTORY and 0 <= t < dur and (not out or t > out[-1]):
            out.append(t)
            src.append("head" if s == 0 else "silence")
            last = t
    return out, src


def claimed_times(pos_times: list, pred: list, excluded: list) -> list:
    """The scorer's positive-claiming loop (turnbench.score.score_task, one speaker) returning the matched prediction
    time per positive (None = FN); used only to attribute each TP to a branch, and checked against the scorer."""
    import bisect
    from turnbench.gold import TAU_MAX_S, TAU_PRE_S
    times = [t for t in pred if not any(iv.start <= t <= iv.end for iv in excluded)]
    anchors = sorted(pos_times)
    claimed, out = set(), []
    for t in anchors:
        i = bisect.bisect_right(anchors, t)
        nxt = anchors[i] if i < len(anchors) else float("inf")
        lo, hi = t - TAU_PRE_S, min(t + TAU_MAX_S, nxt)
        j = bisect.bisect_left(times, lo)
        m = None
        while j < len(times) and times[j] <= hi:
            if j not in claimed:
                claimed.add(j)
                m = times[j]
                break
            j += 1
        out.append(m)
    return out


class Corpus:
    """Channels + gold of one corpus; ``score(events_fn)`` -> per-conversation counts with branch attribution."""

    def __init__(self, name: str, chans: list, gold: dict):
        self.name, self.chans, self.gold = name, chans, gold
        self.cids = list(dict.fromkeys(c.cid for c in chans))
        self.by = {}
        for c in chans:
            g = gold[c.cid]
            self.by[(c.cid, c.k)] = ([e for e in g.eot_positive_events if e.speaker == c.k],
                                     [s for s in g.eot_negative_spans if s.speaker == c.k],
                                     [iv for iv in g.eot_excluded if iv.speaker == c.k])

    def score(self, events_fn, check: bool = False) -> dict:
        """events_fn(chan) -> (times, sources). Returns {cid: [tp, fn, fp, tn, lat_ms (array), src (array of 0 head /
        1 silence)]} with the counts of turnbench.score.score_task per speaker."""
        from turnbench.score import score_task
        res = {c: [0, 0, 0, 0, [], []] for c in self.cids}
        for ch in self.chans:
            pos, neg, exc = self.by[(ch.cid, ch.k)]
            times, src = events_fn(ch)
            sc = score_task(pos, neg, {ch.k: times}, exc)
            m = claimed_times([e.time_s for e in pos], times, exc)
            smap = dict(zip(times, src))
            lat = [(t - a) * 1000.0 for a, t in zip(sorted(e.time_s for e in pos), m) if t is not None]
            s = [0 if smap[t] == "head" else 1 for t in m if t is not None]
            if check:
                assert len(lat) == sc.tp and np.allclose(sorted(lat), sorted(sc.latencies_ms)), (ch.cid, ch.k)
            r = res[ch.cid]
            r[0] += sc.tp
            r[1] += sc.fn
            r[2] += sc.fp
            r[3] += sc.tn
            r[4] += lat  # our matched latencies (equal to the scorer's when check=True), aligned with the sources
            r[5] += s
        return {c: (r[0], r[1], r[2], r[3], np.round(np.asarray(r[4], np.float32), 1), np.asarray(r[5], np.int8))
                for c, r in res.items()}


def aggregate(per_conv: dict, cids) -> dict:
    tp = sum(per_conv[c][0] for c in cids)
    fn = sum(per_conv[c][1] for c in cids)
    fp = sum(per_conv[c][2] for c in cids)
    tn = sum(per_conv[c][3] for c in cids)
    lat = np.concatenate([per_conv[c][4] for c in cids]) if cids else np.zeros(0)
    src = np.concatenate([per_conv[c][5] for c in cids]) if cids else np.zeros(0, np.int8)
    q = (lambda x, pc: None if not len(x) else round(float(np.percentile(np.asarray(x, np.float64), pc))))
    out = {"recall": round(tp / max(tp + fn, 1), 4), "fp_rate": round(fp / max(fp + tn, 1), 4), "tp": tp, "fn": fn,
           "fp": fp, "tn": tn, "p10": q(lat, 10), "p50": q(lat, 50), "p90": q(lat, 90)}
    h = src == 0
    out["head_share"] = round(float(h.mean()), 3) if len(src) else None
    out["p50_head"], out["p50_silence"] = q(lat[h], 50), q(lat[~h], 50)
    out["p90_head"], out["p90_silence"] = q(lat[h], 90), q(lat[~h], 90)
    return out


# --------------------------------------------------------------------------- data
def pipecat_speech(p: np.ndarray, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    sp = B.speech_chunks_pipecat(B.pipecat_vad(p))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, sp)
    return sp


def load_tb(work: Path) -> Corpus:
    G = tb_gold()
    dur = json.loads((D.TB_SCORER / "turnbench" / "durations-dev.json").read_text())["durations"]
    chans = []
    for c in sorted(G, key=int):
        z = np.load(work / "vad" / f"{c}.npz")
        for k in (1, 2):
            sp = pipecat_speech(z[f"p{k}"], work / "pipecat" / f"{c}_{k}.npy")
            chans.append(Chan(c, k, sp, np.load(work / "head" / f"{c}_spk{k}.npy"), float(dur[c])))
    return Corpus("turnbench_dev", chans, G)


def load_oto(work: Path, gap: float) -> tuple[Corpus, dict]:
    ds = D.Dyadic(oto_ids(), "oto", verbose=False)
    G = oto_gold_all(ds, gap)
    chans = []
    for c in ds.meetings:
        z = np.load(work / "vad" / f"{c}.npz")
        for k in (1, 2):
            sp = pipecat_speech(z[f"p{k}"], work / "pipecat" / f"{c}_{k}.npy")
            chans.append(Chan(c, k, sp, np.load(work / "head" / f"{c}_spk{k}.npy"), ds.duration(c)))
    return Corpus("oto", chans, G), {m: ds.roles[m]["human"] + 1 for m in ds.meetings}


def tb_halves(cids) -> tuple[list, list]:
    ids = sorted(cids, key=int)
    return ids[0::2], ids[1::2]


def calib_samples(corpus: Corpus, cids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(raw head p, label, weight) at the user's silent chunks of ``cids`` (where the policy reads phat): 1 inside
    (anchor, anchor + TAU_MAX] of the speaker's EOT positives, 0 inside the speaker's mid-turn-pause negatives; other
    chunks unused. Weight = 1 / (chunks of that event), so every gold event counts once: phat estimates P(this
    silence is a turn end | p) at the event level, not per 32 ms of silence (3 s post-end windows would otherwise
    outweigh the 0.4 s median pause 10:1)."""
    from turnbench.gold import TAU_MAX_S
    P, Y, W = [], [], []
    cs = set(cids)
    for ch in corpus.chans:
        if ch.cid not in cs:
            continue
        pos, neg, _ = corpus.by[(ch.cid, ch.k)]
        ok0 = ch.fidx >= 0
        for lab, spans in ((0, [(sp.start, sp.end, True) for sp in neg]),
                           (1, [(e.time_s, e.time_s + TAU_MAX_S, False) for e in pos])):
            for a, b, closed in spans:
                m = ok0 & (ch.tau >= a if closed else ch.tau > a) & (ch.tau <= b)
                n = int(m.sum())
                if n:
                    P.append(ch.p[m])
                    Y.append(np.full(n, lab, np.int8))
                    W.append(np.full(n, 1.0 / n))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(W)


# --------------------------------------------------------------------------- policies -> events
def policy_grid(family: str) -> list[tuple]:
    if family == "fixed":
        return [(th, k) for th in THETAS for k in K_GRID]
    if family in ("lin", "lin_cal"):
        return [(th, t0, a, tm) for th in THETAS for t0 in LIN_T0 for a in LIN_A for tm in LIN_TMIN if tm < t0]
    if family in ("exp", "vel"):
        return [(th, tm, tx, lam, gam) for th in THETAS for tm in EXP_TMIN for tx in EXP_TMAX for lam in EXP_LAM
                for gam in EXP_GAM if tm < tx]
    raise KeyError(family)


def events_fn(family: str, pt: tuple, calcache: dict | None = None):
    """chan -> (times, sources) of one policy point. ``calcache[id(chan)]`` = (phat, rise) at the silent chunks."""
    def fn(ch):
        th = pt[0]
        if family == "fixed":
            sil = ch.silence_events(np.full(len(ch.s), pt[1]))
        elif family == "lin":
            sil = ch.silence_events(lin_wait(ch.p, *pt[1:]))
        elif family == "lin_cal":
            sil = ch.silence_events(lin_wait(calcache[id(ch)][0], *pt[1:]))
        else:
            ph, ri = calcache[id(ch)]
            w = exp_wait(ph, *pt[1:5])
            if family == "exp":
                sil = ch.silence_events(w)
            else:
                vp, vr, vn = pt[5:8] if len(pt) > 5 else (VEL_P, VEL_RISE, VEL_FRAMES)
                r = ri[vn] if isinstance(ri, dict) else ri
                sil = ch.silence_events(w, vel=(ph > vp) & (r > vr), tmin=pt[1])
        return merged(ch.head_events(th), sil, ch.dur)
    return fn


def point_json(family: str, pt: tuple) -> dict:
    th = None if not np.isfinite(pt[0]) else pt[0]
    if family == "fixed":
        return {"theta": th, "k_s": pt[1]}
    if family in ("lin", "lin_cal"):
        return {"theta": th, "T0_s": pt[1], "a_s": pt[2], "Tmin_s": pt[3]}
    vel = pt[5:8] if len(pt) > 5 else (VEL_P, VEL_RISE, VEL_FRAMES)
    return {"theta": th, "Tmin_s": pt[1], "Tmax_s": pt[2], "lambda": pt[3], "gamma": pt[4],
            **({"vel": {"phat_gt": vel[0], "rise_gt": vel[1], "frames": vel[2]}} if family.startswith("vel") else {})}


# --------------------------------------------------------------------------- selection
def feasible(m: dict, min_recall: float = MIN_RECALL, max_fp: float = FP_BUDGET, max_p50=None) -> bool:
    return (m["recall"] >= min_recall and m["fp_rate"] <= max_fp and m["p50"] is not None
            and (max_p50 is None or m["p50"] <= max_p50))


def choose(rows: list, split: str, **kw):
    """The task's objective on the FITTING split: lowest P50 with recall >= 0.80 and fp <= 0.10 (ties: higher recall,
    lower fp). Returns the row or None."""
    ok = [r for r in rows if feasible(r[split], **kw)]
    if not ok:
        return None
    return min(ok, key=lambda r: (r[split]["p50"], -r[split]["recall"], r[split]["fp_rate"]))


def best_recall(rows: list, split: str, max_p50: float, max_fp: float = FP_BUDGET):
    """Highest recall with fp <= max_fp and P50 <= max_p50 on ``split`` (ties: lower P50)."""
    ok = [r for r in rows if r[split]["fp_rate"] <= max_fp and r[split]["p50"] is not None and r[split]["p50"] <= max_p50]
    if not ok:
        return None
    return max(ok, key=lambda r: (r[split]["recall"], -r[split]["p50"]))


def lowest_fp(rows: list, split: str, max_p50: float, min_recall: float = MIN_RECALL):
    """What a P50 target costs: the lowest fp with P50 <= max_p50 and recall >= min_recall on ``split``."""
    ok = [r for r in rows if r[split]["p50"] is not None and r[split]["p50"] <= max_p50 and r[split]["recall"] >= min_recall]
    return min(ok, key=lambda r: (r[split]["fp_rate"], -r[split]["recall"])) if ok else None


def pareto(rows: list, split: str, max_fp: float = FP_BUDGET) -> list:
    """Points with fp <= max_fp on ``split`` not dominated in (recall up, P50 down); sorted by P50."""
    ok = sorted([r for r in rows if r[split]["fp_rate"] <= max_fp and r[split]["p50"] is not None],
                key=lambda r: (r[split]["p50"], -r[split]["recall"]))
    out, best = [], -1.0
    for r in ok:
        if r[split]["recall"] > best + 1e-12:
            out.append(r)
            best = r[split]["recall"]
    return out


# --------------------------------------------------------------------------- sweep
OTO_GAP = 0.2  # Turn segments merged across < 0.2 s: oto's pause-span p10/p50/p90 then match TurnBench dev's (section 7)
CALS = ("oto", "tbA", "tbB")


def parts() -> list[tuple]:
    out = [("fixed", None), ("lin", None)]
    for c in CALS:
        out += [("lin_cal", c), ("exp", c), ("vel", c)]
    return out


def part_path(work: Path, fam: str, cal) -> Path:
    return work / "sweep" / f"{fam}__{cal or 'raw'}.pkl"


class Ctx:
    def __init__(self, a):
        self.tb = load_tb(Path(a.work_tb))
        self.oto, self.oto_human = load_oto(Path(a.work_oto), OTO_GAP)
        self.A, self.B = tb_halves(self.tb.cids)
        self.cals = {}
        for c, (corp, ids) in {"oto": (self.oto, self.oto.cids), "tbA": (self.tb, self.A),
                               "tbB": (self.tb, self.B)}.items():
            p, y, w = calib_samples(corp, ids)
            self.cals[c] = Isotonic().fit(p, y, w)
            self.cals[c].n, self.cals[c].pos_rate = int(round(w.sum())), float((w * y).sum() / w.sum())
        self._cc = {}

    def calcache(self, cal):
        if cal not in self._cc:
            self._cc[cal] = {id(ch): ch.calibrated(self.cals[cal], VEL_GRID_FRAMES) for corp in (self.tb, self.oto)
                             for ch in corp.chans}
        return self._cc[cal]

    def splits(self, cal):
        """{split: (corpus, conversation ids)} evaluated for a part."""
        sp = {"tb": (self.tb, self.tb.cids), "tbA": (self.tb, self.A), "tbB": (self.tb, self.B)}
        if cal in (None, "oto"):
            sp["oto"] = (self.oto, self.oto.cids)
            sp["oto_human"] = (self.oto, self.oto.cids)
        return sp

    def eval_point(self, fam, cal, pt, check=False) -> dict:
        fn = events_fn(fam, pt, self.calcache(cal) if cal else None)
        row = {"family": fam, "cal": cal, "pt": pt}
        per = {}
        for name, (corp, ids) in self.splits(cal).items():
            if name == "oto_human":
                continue
            if corp.name not in per:
                per[corp.name] = corp.score(fn, check=check)
            row[name] = aggregate(per[corp.name], ids)
        if "oto" in row:  # the human party's ends only (role 'human' of bench_turn_dyadic), same predictions
            row["oto_human"] = self.oto_human_metrics(fn)
        return row

    def oto_human_metrics(self, fn) -> dict:
        sub = Corpus("oto_h", [c for c in self.oto.chans if c.k == self.oto_human[c.cid]], self.oto.gold)
        return aggregate(sub.score(fn), sub.cids)


def stage_sweep(a) -> int:
    import pickle
    work = Path(a.work_oto)
    todo = [(f, c) for f, c in parts() if not part_path(work, f, c).exists()]
    if not todo:
        print("  sweep: 0 left")
        return 0
    t0 = time.time()
    ctx = Ctx(a)
    print(f"  loaded ({time.time() - t0:.0f}s); calibration samples " +
          ", ".join(f"{c}: n {v.n} pos {v.pos_rate:.3f}" for c, v in ctx.cals.items()), flush=True)
    for fam, cal in todo:
        pp = part_path(work, fam, cal)
        tmp = pp.with_suffix(".partial.pkl")
        rows = pickle.loads(tmp.read_bytes()) if tmp.exists() else []
        grid = policy_grid(fam)
        for pt in grid[len(rows):]:
            if time.time() - t0 > a.budget:
                break
            rows.append(ctx.eval_point(fam, cal, pt, check=len(rows) == 0))
        pp.parent.mkdir(parents=True, exist_ok=True)
        if len(rows) == len(grid):
            pp.write_bytes(pickle.dumps(rows))
            tmp.unlink(missing_ok=True)
        else:
            tmp.write_bytes(pickle.dumps(rows))
        print(f"  {fam}@{cal}: {len(rows)}/{len(grid)} ({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > a.budget:
            break
    left = sum(not part_path(work, f, c).exists() for f, c in parts())
    print(f"  sweep: {left} left")
    return left


# --------------------------------------------------------------------------- report
FAMILIES = ("fixed", "lin", "lin_cal", "exp", "vel")
PROTOCOLS = {"oto->tb": ("oto", "oto", "tb"), "tbA->tbB": ("tbA", "tbA", "tbB"), "tbB->tbA": ("tbB", "tbB", "tbA")}
KEYS = ("recall", "fp_rate", "p10", "p50", "p90", "head_share", "p50_head", "p50_silence", "p90_head", "p90_silence",
        "tp", "fn", "fp", "tn")


def slim(m: dict | None) -> dict | None:
    return None if m is None else {k: m[k] for k in KEYS if k in m}


def load_rows(work: Path) -> dict:
    import pickle
    return {(f, c): pickle.loads(part_path(work, f, c).read_bytes()) for f, c in parts()}


def fam_rows(rows: dict, fam: str, cal: str) -> list:
    return rows[(fam, cal if fam in ("lin_cal", "exp", "vel") else None)]


def pooled_crossfit(ctx, fam: str, ptA, ptB) -> dict:
    """The two held-out halves pooled: point fitted on A scored on B's conversations, point fitted on B on A's."""
    rA = ctx.tb.score(events_fn(fam, ptA, ctx.calcache("tbA") if fam not in ("fixed", "lin") else None))
    rB = ctx.tb.score(events_fn(fam, ptB, ctx.calcache("tbB") if fam not in ("fixed", "lin") else None))
    per = {**{c: rA[c] for c in ctx.B}, **{c: rB[c] for c in ctx.A}}
    return aggregate(per, ctx.tb.cids)


def stage_report(a) -> dict:
    t0 = time.time()
    work = Path(a.work_oto)
    rows = load_rows(work)
    ctx = Ctx(a)
    tbres = json.loads((ROOT / "runs" / "turnbench_dev.json").read_text())
    out = {"protocol": "TurnBench dev official scorer (score_task on the scorer's own gold, checked equal to "
                       "score_submission at the DYADIC.md hybrid point); oto = the scorer's floor construction on our "
                       "per-channel Silero segments (gap %.1f s); fit = lowest P50 s.t. recall >= %.2f and fp <= %.2f on "
                       "the FITTING split; held-out numbers are never selected on their own split" % (OTO_GAP, MIN_RECALL, FP_BUDGET),
           "inputs": "user-channel Silero v5 + Pipecat state machine (silence); trail6 head on the mixed mono with the "
                     "user's channel Silero track (DYADIC.md section 4), 160 ms emission",
           "gold": {"turnbench_dev": gold_stats(ctx.tb.gold), "oto": gold_stats(ctx.oto.gold),
                    "oto_human_party_eot_pos": int(sum(1 for m, g in ctx.oto.gold.items() for e in g.eot_positive_events
                                                       if e.speaker == ctx.oto_human[m]))},
           "tb_halves": {"A": ctx.A, "B": ctx.B}, "oto_conversations": ctx.oto.cids,
           "calibrations": {c: {"events": v.n, "pos_rate_event_weighted": round(v.pos_rate, 3), **v.to_json(40)}
                            for c, v in ctx.cals.items()},
           "grids": {f: len(policy_grid(f)) for f in FAMILIES},
           "velocity_rule": {"phat_gt": VEL_P, "rise_gt": VEL_RISE, "frames": VEL_FRAMES},
           "reference": {"dyadic_md_hybrid_in_sample": slim(ctx.eval_point("fixed", None, (0.9988547563552856, 1.5))["tb"]),
                         "published_dev_rescored": {k: tbres["published_dev_rescored"][k] for k in
                                                    ("vap", "espnet_turntaking", "smart_turn_v3", "kyutai_semantic_vad",
                                                     "mimi_endpointer")}},
           "families": {}}
    for fam in FAMILIES:
        F = {"protocols": {}}
        chosen = {}
        for proto, (cal, fit, ev) in PROTOCOLS.items():
            R = fam_rows(rows, fam, cal)
            c = choose(R, fit)
            rule = "min P50 s.t. recall >= 0.80, fp <= 0.10"
            if c is None:  # infeasible on the fitting split: fall back to the scorer's rule
                ok = [r for r in R if r[fit]["fp_rate"] <= FP_BUDGET]
                c = max(ok, key=lambda r: (r[fit]["recall"], -(r[fit]["p50"] or 1e9))) if ok else None
                rule = "INFEASIBLE -> highest recall at fp <= 0.10"
            chosen[proto] = c
            P = {"rule": rule, "point": point_json(fam, c["pt"]) if c else None, "fit": slim(c[fit]) if c else None,
                 "heldout": slim(c[ev]) if c else None,
                 "pareto_fit": [{"point": point_json(fam, r["pt"]), "fit": slim(r[fit]), "heldout": slim(r[ev])}
                                for r in pareto(R, fit)]}
            if proto == "oto->tb" and c:
                P["fit_oto_human_party"] = slim(c.get("oto_human"))
            for lim in (700, 500):
                b = best_recall(R, fit, lim)
                P[f"best_recall_p50_le_{lim}"] = None if b is None else {
                    "point": point_json(fam, b["pt"]), "fit": slim(b[fit]), "heldout": slim(b[ev])}
            F["protocols"][proto] = P
        if chosen["tbA->tbB"] and chosen["tbB->tbA"]:
            F["crossfit_pooled_heldout"] = slim(pooled_crossfit(ctx, fam, chosen["tbA->tbB"]["pt"], chosen["tbB->tbA"]["pt"]))
            for lim in (700, 500):
                bA = best_recall(fam_rows(rows, fam, "tbA"), "tbA", lim)
                bB = best_recall(fam_rows(rows, fam, "tbB"), "tbB", lim)
                F[f"crossfit_pooled_best_recall_p50_le_{lim}"] = (
                    slim(pooled_crossfit(ctx, fam, bA["pt"], bB["pt"])) if bA and bB else None)
        Rin = fam_rows(rows, fam, "oto")  # calibration from oto: the TurnBench-dev front is in-sample only in the grid
        F["tb_in_sample_pareto"] = [{"point": point_json(fam, r["pt"]), "tb": slim(r["tb"])} for r in pareto(Rin, "tb")]
        for lim in (700, 500):
            b = best_recall(Rin, "tb", lim)
            F[f"tb_in_sample_best_recall_p50_le_{lim}"] = None if b is None else {
                "point": point_json(fam, b["pt"]), "tb": slim(b["tb"])}
            b = lowest_fp(Rin, "tb", lim)
            F[f"tb_in_sample_lowest_fp_p50_le_{lim}_recall_ge_0.80"] = None if b is None else {
                "point": point_json(fam, b["pt"]), "tb": slim(b["tb"])}
            b = lowest_fp(Rin, "tb", lim, 0.0)
            F[f"tb_in_sample_lowest_fp_p50_le_{lim}"] = None if b is None else {
                "point": point_json(fam, b["pt"]), "tb": slim(b["tb"])}
        mn = min((r for r in Rin if r["tb"]["fp_rate"] <= FP_BUDGET and r["tb"]["p50"] is not None),
                 key=lambda r: r["tb"]["p50"], default=None)
        F["tb_in_sample_min_p50_fp_le_0.10"] = None if mn is None else {"point": point_json(fam, mn["pt"]), "tb": slim(mn["tb"])}
        mo = min((r for r in fam_rows(rows, fam, "oto") if feasible(r["oto"], 0.0)), key=lambda r: r["oto"]["p50"], default=None)
        F["oto_min_p50_fp_le_0.10"] = None if mo is None else {"point": point_json(fam, mo["pt"]), "oto": slim(mo["oto"]),
                                                              "heldout_tb": slim(mo["tb"])}
        out["families"][fam] = F
        print(f"  {fam}: " + " | ".join(f"{p}: {F['protocols'][p]['heldout']}" for p in PROTOCOLS) + f" ({time.time() - t0:.0f}s)", flush=True)
    # velocity sensitivity: looser early-exit rules on top of each protocol's chosen exp point, fitted the same way
    vg = {}
    for proto, (cal, fit, ev) in PROTOCOLS.items():
        base = out["families"]["exp"]["protocols"][proto]
        cb = choose(fam_rows(rows, "exp", cal), fit)
        if cb is None:
            continue
        cand = [ctx.eval_point("exp", cal, cb["pt"])] + [ctx.eval_point("vel", cal, tuple(cb["pt"]) + (VEL_P, r, n))
                                                         for r in VEL_GRID_RISE for n in VEL_GRID_FRAMES]
        c = choose(cand, fit)
        vg[proto] = {"base_exp": base["point"], "chosen": point_json(c["family"], c["pt"]) if c else None,
                     "fit": slim(c[fit]) if c else None, "heldout": slim(c[ev]) if c else None,
                     "grid": [{"point": point_json(r["family"], r["pt"]), "fit": slim(r[fit]), "heldout": slim(r[ev])}
                              for r in cand]}
    out["vel_sensitivity"] = vg
    out["chosen_points_for_v2"] = {fam: out["families"][fam]["protocols"]["oto->tb"]["point"] for fam in FAMILIES}
    out["sec"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- eot-bench v2 with the chosen points
V2_SETS = {"oto": dict(corpus="oto", n_conv=16, roles=("human",)), "turnbench": dict(corpus="turnbench", n_conv=10, roles=None)}
V2_HORIZON = 75  # 6 s of post-end frames (block C)


def v2_windows(name: str):
    """The block-C windows of DYADIC.md section 3 (bench_turn_dyadic.dyadic_data, same arguments); audio is never read
    (the streams come from the whole-conversation runs), so the mix loader is replaced by silence of the right length."""
    cfg = V2_SETS[name]
    orig = D.Dyadic._load_audio
    D.Dyadic._load_audio = lambda self, m: np.zeros(int(round(self.durations[m] * SR)) + SR, np.float32)
    try:
        base, ext, meta, ds, _ = BD.dyadic_data(cfg["corpus"], BD.DEFAULTS[cfg["corpus"]]["hours"], cfg["n_conv"], 650,
                                                roles=cfg["roles"])
    finally:
        D.Dyadic._load_audio = orig
    for v in ext:
        v.pop("audio", None)
    return ext, ds


def v2_track(v: dict, times: list) -> np.ndarray:
    """Policy events (commit times, s, whole conversation) -> the window's per-frame score: 1 on the frame whose end
    is the first frame end at or after the commit (decided by the end of that frame), else 0."""
    T = len(v["spk_act"])
    x = np.zeros(T, np.float32)
    rel = np.asarray(times, np.float64) - float(v["start"])
    f = np.ceil(rel / FRAME_SEC - 1e-9).astype(np.int64) - 1
    f = f[(f >= 0) & (f < T)]
    x[f] = 1.0
    return x


def stage_v2(a) -> dict:
    import bench_turn_icsi as BI
    from audioforge.conversation import eot_outcomes, floor_stratum
    res = json.loads(Path(a.out).read_text())
    pts = res["chosen_points_for_v2"]
    ctx = Ctx(a)
    systems = {"hybrid_dyadic_md (theta 0.99885, k 1.5 s; TurnBench-dev-selected)": ("fixed", None, (0.9988547563552856, 1.5))}
    for fam, pj in pts.items():
        if pj is None:
            continue
        th = pj["theta"] if pj["theta"] is not None else float("inf")
        pt = {"fixed": lambda: (th, pj["k_s"]), "lin": lambda: (th, pj["T0_s"], pj["a_s"], pj["Tmin_s"]),
              "lin_cal": lambda: (th, pj["T0_s"], pj["a_s"], pj["Tmin_s"])}.get(
            fam, lambda: (th, pj["Tmin_s"], pj["Tmax_s"], pj["lambda"], pj["gamma"]))()
        systems[f"{fam} (oto-fitted)"] = (fam, "oto" if fam in ("lin_cal", "exp", "vel") else None, pt)
    out = {"protocol": "eot-bench v2 block C (6 s trail, horizon 75 frames) on the DYADIC.md section 3 windows; the "
                       "policies run CONTINUOUSLY on the whole conversation (same streams as the TurnBench runs) and the "
                       "window reads their committed events; operating points frozen (oto-fitted), no v2 re-fit",
           "sets": {}}
    for name in V2_SETS:
        ext, ds = v2_windows(name)
        corp = ctx.oto if name == "oto" else ctx.tb
        chan = {(c.cid, c.k): c for c in corp.chans}
        on, en, pauses = BI._prep(ext)
        strata = np.array([floor_stratum(v["spk_targets"], int(e), BD.E.V2_FLOOR_HORIZON) for v, e in zip(ext, en)])
        groups = {"open": strata == "open", "taken": strata != "open"}
        S = {"n": len(ext), "conversations": len(set(v["meeting"] for v in ext)), "systems": {}}
        for sname, (fam, cal, pt) in systems.items():
            fn = events_fn(fam, pt, ctx.calcache(cal) if cal else None)
            ev = {}
            tracks = []
            for v in ext:
                key_ = (v["meeting"], int(str(v["party"]).rsplit(":", 1)[-1]) + 1)
                if key_ not in ev:
                    ev[key_] = fn(chan[key_])
                tracks.append(v2_track(v, ev[key_][0]))
            oc = eot_outcomes(tracks, on, en, [0.5], post_end_frames=V2_HORIZON, pauses=pauses)
            m = BI.point_metrics(oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0], oc["npause"], groups, a.n_boot)
            S["systems"][sname] = {"point": point_json(fam, pt), **{k: m[k] for k in ("fc_rate", "fc_per_pause",
                                   "miss_rate", "miss_rate_ci", "p50_ms", "p90_ms")},
                                   "open_miss": m["strata"]["open"]["miss_rate"], "taken_miss": m["strata"]["taken"]["miss_rate"],
                                   "n_open": int(groups["open"].sum())}
            print(f"  v2 {name} {sname}: " + json.dumps(S["systems"][sname]), flush=True)
        out["sets"][name] = S
    return out


# --------------------------------------------------------------------------- markdown
def _m(m, keys=("recall", "fp_rate", "p50", "p90")) -> str:
    if m is None:
        return "–"
    f = {"recall": "{:.3f}", "fp_rate": "{:.3f}", "p50": "{}", "p90": "{}"}
    return " / ".join(f[k].format(m[k]) for k in keys)


def _pt(p) -> str:
    if p is None:
        return "–"
    th = "off" if p["theta"] is None else f"{p['theta']}"
    if "k_s" in p:
        return f"θ {th}, k {p['k_s']} s"
    if "T0_s" in p:
        return f"θ {th}, T0 {p['T0_s']}, a {p['a_s']}, Tmin {p['Tmin_s']}"
    v = f", vel ({p['vel']['phat_gt']}, {p['vel']['rise_gt']}, {p['vel']['frames']} fr)" if "vel" in p else ""
    return f"θ {th}, Tmin {p['Tmin_s']}, Tmax {p['Tmax_s']}, λ {p['lambda']}, γ {p['gamma']}{v}"


def tables(res: dict) -> str:
    L = []
    L.append("| family | protocol | chosen point | fit: recall / fp / P50 / P90 | HELD-OUT: recall / fp / P50 / P90 | held-out head share | held-out P50 head / silence ms |")
    L.append("|---|---|---|---|---|---|---|")
    for fam, F in res["families"].items():
        for proto, P in F["protocols"].items():
            h = P["heldout"]
            L.append(f"| {fam} | {proto} | {_pt(P['point'])}{' (INFEASIBLE on fit)' if 'INFEAS' in P['rule'] else ''} | "
                     f"{_m(P['fit'])} | {_m(h)} | {h['head_share'] if h else '–'} | "
                     f"{(h['p50_head'] if h else '–')} / {(h['p50_silence'] if h else '–')} |")
        if "crossfit_pooled_heldout" in F:
            h = F["crossfit_pooled_heldout"]
            L.append(f"| {fam} | TB halves pooled | (per half) | | {_m(h)} | {h['head_share']} | {h['p50_head']} / {h['p50_silence']} |")
    L.append("")
    L.append("| family | TB-dev in-sample Pareto front at fp <= 0.10 (recall / fp / P50 / P90; point) |")
    L.append("|---|---|")
    for fam, F in res["families"].items():
        L.append(f"| {fam} | " + "<br>".join(f"{_m(r['tb'])} ({_pt(r['point'])})" for r in F["tb_in_sample_pareto"]) + " |")
    L.append("")
    L.append("| family | oto Pareto front at fp <= 0.10: fit (oto) -> held-out TB dev |")
    L.append("|---|---|")
    for fam, F in res["families"].items():
        L.append(f"| {fam} | " + "<br>".join(f"{_m(r['fit'])} -> {_m(r['heldout'])}" for r in F["protocols"]["oto->tb"]["pareto_fit"]) + " |")
    L.append("")
    L.append("| family | TB in-sample min P50 at fp <= 0.10 | best recall P50 <= 700, fp <= 0.10 (in-sample / pooled cross-fit) | best recall P50 <= 500 | lowest fp at P50 <= 700, recall >= 0.80 (in-sample) | lowest fp at P50 <= 500, recall >= 0.80 |")
    L.append("|---|---|---|---|---|---|")
    for fam, F in res["families"].items():
        g = lambda k, sub="tb": (_m(F[k][sub]) if F.get(k) else "none")  # noqa: E731
        L.append(f"| {fam} | {g('tb_in_sample_min_p50_fp_le_0.10')} | {g('tb_in_sample_best_recall_p50_le_700')} / "
                 f"{_m(F.get('crossfit_pooled_best_recall_p50_le_700')) if F.get('crossfit_pooled_best_recall_p50_le_700') else 'none'} | "
                 f"{g('tb_in_sample_best_recall_p50_le_500')} | {g('tb_in_sample_lowest_fp_p50_le_700_recall_ge_0.80')} | "
                 f"{g('tb_in_sample_lowest_fp_p50_le_500_recall_ge_0.80')} |")
    if "eot_bench_v2" in res:
        L.append("")
        L.append("| set | system (frozen point) | FC % turn / pause | miss 6 s % [CI] | P50 / P90 ms | open miss % (n) | taken miss % |")
        L.append("|---|---|---|---|---|---|---|")
        for sname, S in res["eot_bench_v2"]["sets"].items():
            for k, r in S["systems"].items():
                L.append(f"| {sname} ({S['n']}) | {k} | {100 * r['fc_rate']:.1f} / {100 * (r['fc_per_pause'] or 0):.1f} | "
                         f"{100 * r['miss_rate']:.1f} [{100 * r['miss_rate_ci'][0]:.1f}, {100 * r['miss_rate_ci'][1]:.1f}] | "
                         f"{r['p50_ms']} / {r['p90_ms']} | {100 * r['open_miss']:.1f} ({r['n_open']}) | {100 * r['taken_miss']:.1f} |")
    return "\n".join(L)


# --------------------------------------------------------------------------- compute stages (oto)
def stage_oto_vad(a):
    ds = oto_ds()
    vad = B.SileroVAD(BD.silero_path())

    def fn(cid, out):
        ch = np.asarray(ds.channels16k(cid), np.float32)
        BT.save_npz(out, p1=vad.probs(ch[0]), p2=vad.probs(ch[1]))
    return TB.budget_loop("vad", ds.meetings, fn, Path(a.work_oto), a.budget, ".npz")


def stage_oto_head(a):
    ds = oto_ds()
    for k, v in (("ckpt", None), ("device", "cpu"), ("tag", "trail6")):  # bench_turnbench.stage_head's newer options
        setattr(a, k, getattr(a, k, v))
    return TB.stage_head(a, ds, Path(a.work_oto))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["oto_vad", "oto_head", "sweep", "report", "v2", "tables"])
    p.add_argument("--work-oto", default=str(WORK_OTO))
    p.add_argument("--work-tb", default=str(WORK_TB))
    p.add_argument("--budget", type=float, default=480.0)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    BT._torch2()
    Path(a.work_oto).mkdir(parents=True, exist_ok=True)
    if a.stage == "oto_vad":
        print(stage_oto_vad(a))
    elif a.stage == "oto_head":
        print(stage_oto_head(a))
    elif a.stage == "sweep":
        stage_sweep(a)
    elif a.stage == "report":
        res = stage_report(a)
        old = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
        if "eot_bench_v2" in old:
            res["eot_bench_v2"] = old["eot_bench_v2"]
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")
    elif a.stage == "tables":
        print(tables(json.loads(Path(a.out).read_text())))
    elif a.stage == "v2":
        v2 = stage_v2(a)
        res = json.loads(Path(a.out).read_text())
        res["eot_bench_v2"] = v2
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
