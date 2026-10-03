"""Scoring pieces of research/FINAL_COMPARE.md added by the evaluation fix wave (plans/fixwave/eval_fixes.md, after
the audits plans/audit/{metrics,fairness,leakage}_001.md). final_compare.py calls these from its score_* functions.

  meta      the meeting id of every cached AMI / ICSI test item (WER sets, VAD windows) and the audio length of every
            VAD window -> WORK/meta/*.json, so CIs can resample meetings and VAD can score frames inside the audio only
  stdder    standard DER (md-eval style: 0.25 s collar on each side of every reference boundary, overlap scored,
            pyannote.metrics) of the open diarizers' cached tracks on the eot-bench v2 test windows, every speaker
            -> WORK/meta/stdder.json (our tracker outputs one target track, so standard DER is not defined for it)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_scoring.py meta|stdder

What is here:
- VAD: only label frames that start inside the audio are scored (the 251st frame of a 20 s window lies past the end and
  was a forced miss for Silero / TEN / pyannote); 1000-resample bootstrap over meetings.
- WER / speaker EER: bootstrap groups = meetings for the AMI / ICSI test sets.
- Live-call WER with repetitions and false starts collapsed in both reference and hypothesis (one rule for all).
- Turn, assistant clips: accuracy / false fire at windows 2.0 / 2.5 / 3.0 / 3.5 s and the headline-window rule.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
SR = 16000
FRAME_S = 0.08
N_BOOT = 1000


def _work() -> Path:
    import final_compare as FC
    return FC.WORK


def meta_dir() -> Path:
    return _work() / "meta"


# =========================================================================== meta: meetings and audio lengths
def stage_meta(a):
    """Rebuild the AMI / ICSI test item lists exactly as final_compare.asr_set / vad_layers.load_set built them and
    store each item's meeting (asserting the cached references / labels match item by item)."""
    import random
    import final_compare as FC
    d = meta_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name in ("ami_eval", "icsi_eval"):
        f = d / f"asr_{name}.json"
        if f.exists():
            continue
        if name == "ami_eval":
            from audioforge.datasets.ami import AMI, subset
            ds = AMI(subset({"eval": 4})["eval"], verbose=False)
        else:
            from audioforge.datasets.icsi import ICSI, subset
            ds = ICSI(subset()["eval"], verbose=False)
        segs = ds.asr(1.0, 15.0)
        idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
        refs = json.loads((FC.WORK / "asr_sets" / f"{name}_refs.json").read_text())
        assert [segs[j]["text"] for j in idx] == refs, f"{name}: rebuilt item list differs from the cached one"
        z = np.load(FC.WORK / "asr_sets" / f"{name}.npz")
        assert all(len(z[f"a{i}"]) == len(segs[j]["audio"]) for i, j in enumerate(idx)), name
        f.write_text(json.dumps({"meeting": [segs[j]["meeting"] for j in idx],
                                 "start": [round(float(segs[j]["start"]), 3) for j in idx]}))
        print(name, "meetings", sorted(set(segs[j]["meeting"] for j in idx)), flush=True)
    import vad_layers as V
    for sn in FC.VAD_SETS:
        f = d / f"vad_{sn}.json"
        if f.exists():
            continue
        val = V.load_set(sn)
        y = np.concatenate([np.asarray(v["vad"], np.float32) for v in val])
        z = np.load(FC.WORK / "vad" / f"silero_v5_{sn}.npz")
        assert np.array_equal(z["y"], y), f"{sn}: labels differ from the cached VAD set"
        f.write_text(json.dumps({"meeting": [v.get("meeting") for v in val],
                                 "n_samples": [int(len(v["audio"])) for v in val],
                                 "n_frames": [int(len(v["vad"])) for v in val]}))
        print(sn, "meetings", sorted(set(v.get("meeting") for v in val)), flush=True)
    print("STAGE_COMPLETE", flush=True)


DEFAULT_META = SSD / "scratch" / "final_compare" / "meta"  # same items whatever WORK a candidate run uses


def _meta(kind: str, name: str):
    for d in (meta_dir(), DEFAULT_META):
        f = d / f"{kind}_{name}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


def asr_groups(name: str):
    """Bootstrap groups of a WER set: meetings for the AMI / ICSI test sets (None when the meta file is missing)."""
    m = _meta("asr", name)
    return m["meeting"] if m else None


# =========================================================================== bootstrap over groups
def group_draws(groups, n_boot=N_BOOT, seed=0):
    """-> list of index arrays, each the items of a resample of the groups (with replacement)."""
    g = np.asarray(groups)
    ug = sorted(set(g.tolist()), key=str)
    members = [np.nonzero(g == k)[0] for k in ug]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]


def ci95(x, nd=4, scale=1.0):
    x = np.asarray(x, float)
    return [round(scale * float(np.percentile(x, q)), nd) for q in (2.5, 97.5)]


# =========================================================================== VAD
def vad_inside(lens, n_samples) -> np.ndarray:
    """Mask over the concatenated label frames: True for a frame that starts inside its window's audio."""
    out = []
    for T, n in zip(lens, n_samples):
        t0 = np.arange(int(T)) * FRAME_S
        out.append(t0 < n / SR - 1e-9)
    return np.concatenate(out)


def vad_metrics(p, y):
    from sklearn.metrics import roc_auc_score
    t = y > 0.5
    d = p > 0.5
    tp, fp, fn = float((d & t).sum()), float((d & ~t).sum()), float((~d & t).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    auc = roc_auc_score(t, p)
    th = np.quantile(p[~t], 1 - 0.075)  # miss at FPR 7.5 % (vad_layers.FPR_POINT)
    return f1, auc, float((p[t] <= th).mean())


def onset_lags(p, y, lens, thr=0.5, min_sil=5, horizon=12):
    lags, missed, i0 = [], 0, 0
    for n in lens:
        pp, yy = p[i0:i0 + n], y[i0:i0 + n] > 0.5
        for i in range(min_sil, n):
            if yy[i] and not yy[i - min_sil:i].any():
                j = np.nonzero(pp[i:i + horizon + 1] > thr)[0]
                if len(j):
                    lags.append(int(j[0]))
                else:
                    missed += 1
        i0 += n
    return np.array(lags), missed


def score_vad(work: Path, systems, sets) -> dict:
    """Every system on every VAD set, frames inside the audio only; CIs = 1000 resamples of meetings (windows kept
    together by meeting); paired F1 / AUC deltas against each core on the same resamples."""
    res = {}
    for sn in sets:
        Z = {s: np.load(work / "vad" / f"{s}_{sn}.npz") for s in systems if (work / "vad" / f"{s}_{sn}.npz").exists()}
        if not Z:
            continue
        ref = next(iter(Z.values()))
        y_all, lens = ref["y"], ref["lens"]
        for s, z in Z.items():
            assert np.array_equal(z["y"], y_all) and np.array_equal(z["lens"], lens), f"{s} {sn}: labels differ"
        meta = _meta("vad", sn)
        assert meta is not None and meta["n_frames"] == [int(x) for x in lens], f"{sn}: run final_scoring.py meta"
        keep = vad_inside(lens, meta["n_samples"])
        win = np.repeat(np.arange(len(lens)), lens)[keep]
        lens_k = np.bincount(win, minlength=len(lens))
        y = y_all[keep]
        meet_of_win = np.asarray(meta["meeting"])
        # resample meetings; each draw = the kept frames of every window of the drawn meetings
        frames_of_win = [np.nonzero(win == w)[0] for w in range(len(lens))]
        wdraws = group_draws(meet_of_win)
        boots = [np.concatenate([frames_of_win[w] for w in wd]) for wd in wdraws]
        o = {"n_windows": int(len(lens)), "n_frames": int(keep.sum()), "frames_dropped_past_audio_end": int((~keep).sum()),
             "meetings": sorted(set(meta["meeting"]), key=str), "speech_frac": round(float((y > 0.5).mean()), 4),
             "bootstrap": "1000 resamples of meetings"}
        B = {}
        for s, z in Z.items():
            p = z["p"].astype(np.float64)[keep]
            f1, auc, miss = vad_metrics(p, y)
            B[s] = np.array([vad_metrics(p[ii], y[ii]) for ii in boots])
            lags, missed = onset_lags(p, y, lens_k)
            o[s] = {"f1_at_0.5": round(f1, 4), "f1_ci95": ci95(B[s][:, 0]), "auc": round(auc, 4),
                    "auc_ci95": ci95(B[s][:, 1]), "miss_at_fpr_7.5_pct": round(100 * miss, 2),
                    "miss_ci95_pct": ci95(B[s][:, 2], 2, 100),
                    "onset_lag_ms_p50": int(80 * np.median(lags)) if len(lags) else None,
                    "onset_lag_ms_p90": int(80 * np.percentile(lags, 90)) if len(lags) else None,
                    "onsets": int(len(lags) + missed),
                    "onsets_detected_within_1s_pct": round(100 * len(lags) / max(len(lags) + missed, 1), 1),
                    "compute_s": round(float(z["sec"]), 1)}
        for base in ("core_0p6b", "core_115m"):
            if base in B:
                for s in B:
                    if s != base:
                        dd = B[s] - B[base]
                        o[f"{s} - {base}"] = {"f1_delta_ci95": ci95(dd[:, 0]), "auc_delta_ci95": ci95(dd[:, 1])}
        res[sn] = o
    return res


# =========================================================================== speaker EER, meeting bootstrap
def eer_pairs(E, spk, meet):
    """Within-meeting pairs: per meeting (cosine scores, same-speaker labels)."""
    X = E / np.linalg.norm(E, axis=1, keepdims=True)
    out = {}
    for m in sorted(set(meet.tolist()), key=str):
        idx = np.nonzero(meet == m)[0]
        iu = np.triu_indices(len(idx), 1)
        s = (X[idx][iu[0]] * X[idx][iu[1]]).sum(1)
        out[m] = (s, (spk[idx][iu[0]] == spk[idx][iu[1]]).astype(np.int64))
    return out


def eer_meeting_boot(pairs: dict, n_boot=N_BOOT, seed=0):
    import torch
    from audioforge.metrics import eer
    ms = list(pairs)
    f = lambda keys: float(eer(torch.tensor(np.concatenate([pairs[k][0] for k in keys])),  # noqa: E731
                               torch.tensor(np.concatenate([pairs[k][1] for k in keys]))))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n_boot):
        keys = [ms[i] for i in rng.integers(0, len(ms), len(ms))]
        lab = np.concatenate([pairs[k][1] for k in keys])
        if lab.min() == lab.max():  # a resample with only one class has no EER
            continue
        b.append(f(keys))
    return f(ms), b


# =========================================================================== live calls: disfluency-neutral WER
_FRAG = re.compile(r"^[\w']+-$")


def drop_fragments(text: str) -> str:
    """Raw text: drop false-start fragments written with a trailing hyphen ("I- I believe" -> "I believe")."""
    return " ".join(w for w in text.split() if not _FRAG.match(w.strip(",.;:!?\"")))


def collapse_repeats(words: list[str], max_n: int = 3) -> list[str]:
    """Normalised words: collapse immediate repetitions of 1-3 word n-grams ("i i believe" -> "i believe")."""
    w = list(words)
    changed = True
    while changed:
        changed = False
        for n in range(max_n, 0, -1):
            i = 0
            out = []
            while i < len(w):
                if i + 2 * n <= len(w) and w[i:i + n] == w[i + n:i + 2 * n]:
                    i += n  # drop the first copy
                    changed = True
                    continue
                out.append(w[i])
                i += 1
            w = out
    return w


def disfl_norm(base):
    """The rule, applied the same way to references and every system's hypotheses."""
    return lambda s: " ".join(collapse_repeats(base(drop_fragments(s)).split()))


# =========================================================================== turn: false-fire windows
WINDOWS = (2.0, 2.5, 3.0, 3.5)
NEAR_S = 0.25  # a window "lands on" a fallback timer when it is within this many seconds of it
FALLBACKS_S = {"Pipecat smart-turn (STOP_SECS, pipecat 1.12)": 3.0,
               "LiveKit EnglishModel (max_endpointing_delay, livekit-agents 1.8.3)": 3.0}


def headline_window(our_fallbacks: dict) -> dict:
    """The largest candidate window that is more than NEAR_S away from every system's nominal fallback timer."""
    fb = {**FALLBACKS_S, **our_fallbacks}
    ok = [w for w in WINDOWS if all(abs(w - v) > NEAR_S for v in fb.values())]
    return {"window_s": max(ok) if ok else min(WINDOWS), "candidates_s": list(WINDOWS), "fallbacks_s": fb,
            "rule": f"the largest of {list(WINDOWS)} s that is more than {NEAR_S} s away from every system's nominal "
                    "fallback timer (the time after which it ends the turn without a 'complete' decision)"}


def asst_point(asess, times, comps):
    import eot_assistant as EA
    sc = EA.score_system(asess, times, comps)
    return {"accuracy_pct": sc["accuracy_pct"], "false_fire_pct": sc["incomplete"]["false_fire_pct"]}


def asst_curve(asess, times, comps, boot=N_BOOT, seed=0) -> dict:
    """Accuracy / false fire at each window (eot_assistant.FF_WINDOW set per window), clip bootstrap CIs, plus where
    each system's first turn end on an incomplete clip falls relative to the audible end."""
    import eot_assistant as EA
    old = EA.FF_WINDOW
    out = {}
    try:
        rng = np.random.default_rng(seed)
        draws = [rng.integers(0, len(asess), len(asess)) for _ in range(boot)]
        for w in WINDOWS:
            EA.FF_WINDOW = w
            pt = asst_point(asess, times, comps)
            bs = [asst_point([asess[i] for i in d], times, comps) for d in draws]
            out[f"{w:g}"] = {**pt, "ci95": {q: ci95([b[q] for b in bs if b[q] is not None], 1)
                                            for q in ("accuracy_pct", "false_fire_pct")}}
    finally:
        EA.FF_WINDOW = old
    off = []
    for s in asess:
        if s["complete"]:
            continue
        t = np.asarray(times[s["key"]], float)
        t = t[(t >= s["onset"]) & (t < s["next_onset"][0])]
        if len(t):
            off.append(float(t.min()) - s["user_turns"][0][1])
    off = np.asarray(off)
    n_inc = sum(not s["complete"] for s in asess)
    out["first_fire_after_audible_end_incomplete"] = {
        "n_incomplete": n_inc, "n_fired": int(len(off)),
        "p50_s": round(float(np.median(off)), 3) if len(off) else None,
        **{f"share_within_{NEAR_S:g}s_of_{w:g}s_pct": round(100 * float(np.mean(np.abs(off - w) <= NEAR_S)), 1)
           if len(off) else None for w in WINDOWS}}
    return out


def ours_asst_times(A: dict, rule: dict, asess) -> tuple[dict, dict]:
    """Turn-end totals of one preset on the assistant clips: core_0p6b_turn.ev_score's replay, verbatim."""
    import turn_v5 as V5
    pk = "p" if rule["mode"] == "head" else rule.get("p5key", "p5")
    times, comps = {}, {}
    for s_ in asess:
        d = A[s_["key"]]
        h = dict(d["head"])
        h["p"] = h[pk]
        comp = d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"asst_{s_['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        times[s_["key"]] = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        comps[s_["key"]] = comp
    return times, comps


def ours_ami_score(D: dict, rule: dict, boot=200, seed=0) -> dict:
    """One preset on the AMI sessions of D only (e.g. the no-print dumps): ev_score's replay and eot_latency's scorer,
    session bootstrap."""
    import eot_latency as E
    import turn_v5 as V5
    pk = "p" if rule["mode"] == "head" else rule.get("p5key", "p5")
    per = []
    for ss in E.sessions():
        if ss["set"] != "ami" or ss["key"] not in D:
            continue
        d = D[ss["key"]]
        h = dict(d["head"])
        h["p"] = h[pk]
        comp = d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"{ss['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        per.append(E.score_session(tt, ss, comp))
    r = E.pool(per)
    rng = np.random.default_rng(seed)
    bs = [E.pool([per[i] for i in rng.integers(0, len(per), len(per))]) for _ in range(boot)]
    r["ci95"] = {q: ci95([b[q] for b in bs if b[q] is not None], 1)
                 for q in ("eot_total_ms_p50", "eot_total_ms_p95", "false_interruption_pct", "missed_pct")}
    r["n_sessions"] = len(per)
    return r


# =========================================================================== standard DER of the diarizers
STD_TEST = {"ami_eval": None, "icsi": ("Bmr013", "Bmr018", "Bro021")}


def _annotation(intervals, uri):
    from pyannote.core import Annotation, Segment
    ann = Annotation(uri=uri)
    for k, (a, b, lab) in enumerate(intervals):
        if b > a:
            ann[Segment(a, b), k] = lab
    return ann


def _runs(col: np.ndarray):
    on = np.flatnonzero(np.diff(np.r_[0, col.astype(np.int8), 0]))
    return list(zip(on[::2], on[1::2]))


def stage_stdder(a):
    """md-eval style DER (collar 0.25 s each side, overlap scored) per window, pooled; meeting bootstrap."""
    import tsvad as TV
    from eval_stage1 import _fit
    from pyannote.core import Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    arms = {"nemotron3": lambda c: SSD / "scratch" / "tsvad" / c / "n3" / "tracks",
            "pyannote31": lambda c: _work() / "pya" / c / "tracks"}
    out = {"metric": "DER = (missed + false alarm + speaker confusion) / reference speech, every speaker of the window, "
                     "0.25 s collar on each side of each reference boundary, overlapping speech scored (pyannote.metrics "
                     "DiarizationErrorRate(collar=0.5, skip_overlap=False)); reference = the corpus speaker intervals; "
                     "hypothesis = the cached diarizer track thresholded at 0.5 on its 80 ms grid",
           "bootstrap": "1000 resamples of meetings"}
    for corpus, test in STD_TEST.items():
        ext, meta, ds = TV.bench_windows(corpus)
        wins = [v for v in ext if test is None or v["meeting"] in test]
        comp = {arm: [] for arm in arms}
        for v in wins:
            key = TV.wkey(v)
            T = len(v["spk_act"])
            a0 = float(v["start"])
            b0 = a0 + len(v["audio"]) / SR
            ref = []
            for spk, iv in ds.acts[v["meeting"]].items():
                ref += [(max(x, a0) - a0, min(y, b0) - a0, str(spk)) for x, y in iv if y > a0 and x < b0]
            R = _annotation(ref, key)
            uem = Segment(0.0, b0 - a0)
            for arm, d in arms.items():
                P = np.load(d(corpus) / f"{key}.stream_rc.npy")
                P = np.stack([_fit(P[:, j], T) for j in range(P.shape[1])], 1) > 0.5
                hyp = [(i * FRAME_S, min(j * FRAME_S, b0 - a0), f"h{c}") for c in range(P.shape[1]) for i, j in _runs(P[:, c])]
                m = DiarizationErrorRate(collar=0.5, skip_overlap=False)
                r = m(R, _annotation(hyp, key), uem=uem, detailed=True)
                comp[arm].append((v["meeting"], r["missed detection"], r["false alarm"], r["confusion"], r["total"]))
        o = {"n_windows": len(wins), "meetings": sorted({v["meeting"] for v in wins})}
        for arm, rows in comp.items():
            M = np.array([x[1:] for x in rows], float)
            meet = [x[0] for x in rows]
            tot = M.sum(0)
            draws = group_draws(meet)
            b = [M[d].sum(0) for d in draws]
            o[arm] = {"der_pct": round(100 * (tot[0] + tot[1] + tot[2]) / tot[3], 2),
                      "ci95": ci95([100 * (x[0] + x[1] + x[2]) / x[3] for x in b], 2),
                      "missed_pct": round(100 * tot[0] / tot[3], 2), "false_alarm_pct": round(100 * tot[1] / tot[3], 2),
                      "confusion_pct": round(100 * tot[2] / tot[3], 2), "ref_speech_s": round(float(tot[3]), 1)}
        out[corpus] = o
        print(corpus, json.dumps(o), flush=True)
    meta_dir().mkdir(parents=True, exist_ok=True)
    (meta_dir() / "stdder.json").write_text(json.dumps(out, indent=1))
    print("STAGE_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("meta", "stdder"))
    a = p.parse_args()
    {"meta": stage_meta, "stdder": stage_stdder}[a.stage](a)


if __name__ == "__main__":
    main()
