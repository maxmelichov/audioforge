"""Conversational action tokens <YIELD> / <HOLD> in the RNNT vocabulary, scored on eot-bench v2 (research/YIELD_TOKENS.md).

Model: runs/stage1_rnnt_yieldhold.afm (research/recipes/stage1_rnnt_yieldhold.yaml: frozen NVIDIA encoder, RNNT decoder + joint
trained on AMI train action transcripts). Same 974 AMI dev turn windows, folds, strata, cross-fitted <= 5 % per-turn FC
operating points, horizons and bootstrap seeds as scripts/research/bench_turn_baselines.py (whose cached baseline tracks -
Silero timeout, Parakeet-Realtime-EOU posterior, our trail6 hybrid - are reused and re-scored in the same call).

Decision scores (audioforge.baselines.turn.rnnt_frame_decode, exactly how the EOU baseline was scored: raw window audio
+ 0.32 s of following meeting audio, [70, 1] chunked mask = 160 ms cache-aware streaming, greedy decoding, no diarizer):
  yield_posterior   per frame the max over the frame's symbol steps of log P(<YIELD>) at the joint (log domain)
  yield_vs_hold     log P(<YIELD>) - log P(<HOLD>) on the same steps
  yield_native      <YIELD> actually emitted by greedy decoding (the model's own rule; no threshold fitting)
  + OR-hybrids with the Silero VAD timeout (joint (theta, k) grid, as the other "+ timeout" rows)
Emission rule 'eou' = 160 ms chunks ((t // 2 + 1) * 2), as for the EOU model.

Stages (CPU, 2 threads; resumable):
  decode   per window: tokens per frame, log posteriors and emission flags of <YIELD> / <HOLD>
  wer      WER of the base (stage1_heads_pretrained) and the new model: LibriSpeech gate subset, AMI dev single-speaker
           segments, AMI dev turn windows (primary words; from the stored per-frame tokens)
  segments end-token test on clean single-speaker AMI dev segments (<YIELD> vs <HOLD> at the segment end)
  report   eot-bench v2 tables, paired deltas, token precision / recall at pause sites, emission timing -> runs/yield_tokens.json
  --ckpt selects the model (default runs/stage1_rnnt_yieldhold.afm; run 1a: runs/stage1_rnnt_yieldhold_1a.afm)

  W=<scratch>/yield_tokens/work
  .venv/bin/python scripts/research/bench_yield_tokens.py --stage decode --work $W     (repeat until "0 left")
  .venv/bin/python scripts/research/bench_yield_tokens.py --stage wer --work $W
  .venv/bin/python scripts/research/bench_yield_tokens.py --stage report --work $W --out runs/yield_tokens.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
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

CKPT = ROOT / "runs" / "stage1_rnnt_yieldhold.afm"
BASE_CKPT = BT.ASR_CKPT  # runs/stage1_heads_pretrained.afm
BT_WORK = BT.SCRATCH / "baselines_turn" / "work"  # cached baseline tracks (silero, eou, asr, ...)
TOKENS = ("<YIELD>", "<HOLD>")
POST_FRAMES = 75  # 6 s horizon of block C


def _norm(s: str) -> str:
    try:
        from audioforge.teachers import normalize_text
        return normalize_text(s)
    except Exception:  # noqa: BLE001
        return " ".join(re.sub(r"[^\w\s']", " ", s.lower().replace("-", " ")).split())


def strip_tokens(s: str) -> str:
    for t in TOKENS:
        s = s.replace(t, " ")
    return " ".join(s.split())


# --------------------------------------------------------------------------- stages
def stage_decode(a, ext, ds, work):
    BT._torch2()
    from audioforge.train import load_model
    m = load_model(str(CKPT), "cpu")
    assert list(m.encoder.att_context_size) == [70, 1], m.encoder.att_context_size
    ids = [m.tokenizer.token_id(t) for t in TOKENS]
    assert ids == [1025, 1024], ids
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        T = len(ex["spk_act"])
        x = BT.padded_audio(ds, ex)
        t0 = time.perf_counter()
        toks, logpost, em = B.rnnt_frame_decode(m, x, watch_ids=ids)
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += 1
        toks = (toks + [[] for _ in range(T)])[:T]
        BT.save_npz(out, logpost=BT.B_fit(logpost, T, -np.inf), emitted=BT.B_fit(em, T, False),
                    toks=np.array(json.dumps(toks)), text=np.array(m.tokenizer.decode([k for tt in toks for k in tt])))
    left = BT.run_resumable("yield", ext, fn, work, a.budget)
    BT.add_timing(work, "yield_offline_window", stat["calls"], stat["sec"], ckpt=str(CKPT.name))
    return left


def _wer_pairs(m, items, bs=8):
    from audioforge.metrics import wer
    hyps = []
    for i in range(0, len(items), bs):
        hyps += m.transcribe([x for x, _ in items[i:i + bs]], head="rnnt")
    return wer([_norm(r) for _, r in items], [_norm(h) for h in hyps])


def stage_wer(a, ext, ds, work):
    BT._torch2()
    from audioforge.data import load_wav
    from audioforge.metrics import wer
    from audioforge.train import load_model
    out_p = work / "wer.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    lines = [json.loads(x) for x in (ROOT / "data/librispeech/test-clean-first200.jsonl").read_text().splitlines() if x.strip()][:100]
    libri = [(load_wav(x["audio_filepath"], 16000), x["text"]) for x in lines]
    segs = ds.asr(1.0, 15.0)
    idx = sorted(random.Random(0).sample(range(len(segs)), min(200, len(segs))))
    segs = [(np.asarray(segs[i]["audio"], np.float32), segs[i]["text"]) for i in idx]
    res["n"] = {"librispeech": len(libri), "ami_dev_segments": len(segs), "ami_dev_windows": len(ext)}
    for name, ck in (("base", BASE_CKPT), ("yieldhold", CKPT)):
        if name in res and "ami_dev_windows" in res[name]:
            continue
        m = load_model(str(ck), "cpu")
        t0 = time.time()
        r = {"ckpt": str(ck.relative_to(ROOT)), "librispeech": _wer_pairs(m, libri), "ami_dev_segments": _wer_pairs(m, segs, 4)}
        # turn windows: hyps from the stored per-frame tokens (same padded, [70, 1]-masked decode for both models)
        refs, hyps = [], []
        for ex in ext:
            if name == "base":
                toks = json.loads((BT_WORK / "asr" / f"{BT.key(ex)}.json").read_text())
            else:
                toks = json.loads(str(np.load(work / "yield" / f"{BT.key(ex)}.npz")["toks"]))
            hyps.append(_norm(m.tokenizer.decode([k for tt in toks for k in tt])))
            refs.append(_norm(strip_tokens(ex["text"])))
        r["ami_dev_windows"] = wer(refs, hyps)
        r["sec"] = round(time.time() - t0, 1)
        res[name] = r
        out_p.write_text(json.dumps(res, indent=1))
        print(name, r, flush=True)
    return 0


def stage_segments(a, ext, ds, work):
    """End-token test on clean single-speaker AMI dev segments (ami.asr with action tokens, run 1a's label rule): the
    segment audio stops 0.1 s after the last word, and its label is <YIELD> (the last word ends the speaker's turn) or
    <HOLD> (the run ends but the turn continues). Reports the greedy end token (last emitted token if it is one of the
    two, else none) against the label, and the AUC of log P(<YIELD>) - log P(<HOLD>) over the last 5 frames."""
    BT._torch2()
    from audioforge.metrics import wer
    from audioforge.train import load_model
    m = load_model(str(CKPT), "cpu")
    ids = [m.tokenizer.token_id(t) for t in TOKENS]
    segs = ds.asr(1.0, 15.0, action_tokens={"hes_gap": 0.3})
    rows = []
    for e in segs:
        toks, lp, em = B.rnnt_frame_decode(m, np.asarray(e["audio"], np.float32), watch_ids=ids)
        flat = [k for t in toks for k in t]
        last = flat[-1] if flat else None
        pred = "<YIELD>" if last == ids[0] else "<HOLD>" if last == ids[1] else "none"
        tail = lp[-5:]
        rows.append({"label": e["end_token"] if e["end_token"] in TOKENS else "none", "pred": pred,
                     "score": float(np.max(tail[:, 0] - tail[:, 1])), "max_logp_yield": float(lp[:, 0].max()),
                     "max_logp_hold": float(lp[:, 1].max()), "n_yield": int(em[:, 0].sum()), "n_hold": int(em[:, 1].sum()),
                     "hyp": _norm(m.tokenizer.decode(flat)), "ref": _norm(e["text_plain"])})
    lab = [r for r in rows if r["label"] != "none"]
    conf = {l: {p: sum(1 for r in lab if r["label"] == l and r["pred"] == p) for p in ("<YIELD>", "<HOLD>", "none")}
            for l in TOKENS}
    y = np.array([r["label"] == "<YIELD>" for r in lab])
    sc = np.array([r["score"] for r in lab])
    # AUC by rank statistic
    order = sc.argsort()
    ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    auc = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)) if n1 and n0 else None
    dec = sum(1 for r in lab if r["pred"] == r["label"])
    res = {"ckpt": str(CKPT.relative_to(ROOT)), "n_segments": len(segs), "n_labelled_ends": len(lab),
           "labels": {t: int(sum(1 for r in lab if r["label"] == t)) for t in TOKENS},
           "confusion_label_x_pred": conf, "end_token_accuracy": dec / max(1, len(lab)),
           "end_token_emitted_rate": sum(1 for r in lab if r["pred"] != "none") / max(1, len(lab)),
           "auc_yield_vs_hold_score": auc,
           "max_logp_yield_p50_p90": [float(x) for x in np.percentile([r["max_logp_yield"] for r in rows], [50, 90])],
           "max_logp_hold_p50_p90": [float(x) for x in np.percentile([r["max_logp_hold"] for r in rows], [50, 90])],
           "wer_segments_plain": wer([r["ref"] for r in rows], [r["hyp"] for r in rows]),
           "emissions": {"yield": int(sum(r["n_yield"] for r in rows)), "hold": int(sum(r["n_hold"] for r in rows))}}
    (work / "segments.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return 0


# --------------------------------------------------------------------------- pause-site token statistics
def pause_site_stats(convs, ems, post_frames: int = POST_FRAMES, slack: int = 2) -> dict:
    """Token behaviour at the labelled sites. ``ems`` = per window (yield_emitted (T,), hold_emitted (T,) or None).
    Pause site = a within-turn pause run [a, b) of ``hes`` plus ``slack`` frames after it (the token is often emitted
    when speech resumes). End site = [turn_end, turn_end + post_frames). Also counts YIELD emissions inside the turn
    [onset, end) (the false yields of the native rule) and post-end."""
    from audioforge.conversation import pause_runs
    c = dict(pauses=0, hold_at_pause=0, yield_at_pause=0, ends=0, yield_at_end=0, hold_at_end=0,
             yield_emissions_in_turn=0, yield_emissions_post_end=0, turns_with_yield_in_turn=0, turns_with_hold=0,
             hold_emissions_total=0, yield_emissions_total=0)
    for v, (ey, eh) in zip(convs, ems):
        T = len(ey)
        on, en = int(v["onset_frame"]), int(v["turn_end_frame"])
        eh = np.zeros(T, bool) if eh is None else eh
        for a_, b_ in pause_runs(v["hes"]):
            s = slice(a_, min(T, b_ + slack))
            c["pauses"] += 1
            c["hold_at_pause"] += bool(eh[s].any())
            c["yield_at_pause"] += bool(ey[s].any())
        e = slice(en, min(T, en + post_frames))
        c["ends"] += 1
        c["yield_at_end"] += bool(ey[e].any())
        c["hold_at_end"] += bool(eh[e].any())
        c["yield_emissions_in_turn"] += int(ey[on:en].sum())
        c["yield_emissions_post_end"] += int(ey[en:].sum())
        c["turns_with_yield_in_turn"] += bool(ey[on:en].any())
        c["turns_with_hold"] += bool(eh.any())
        c["hold_emissions_total"] += int(eh.sum())
        c["yield_emissions_total"] += int(ey.sum())
    r = dict(c)
    n_p, n_e = max(1, c["pauses"]), max(1, c["ends"])
    r["rates"] = {"hold_recall_at_pause": c["hold_at_pause"] / n_p, "yield_false_at_pause": c["yield_at_pause"] / n_p,
                  "yield_recall_at_end_6s": c["yield_at_end"] / n_e, "hold_false_at_end": c["hold_at_end"] / n_e,
                  "turns_with_yield_in_turn": c["turns_with_yield_in_turn"] / n_e,
                  "yield_precision_post_end": c["yield_emissions_post_end"] / max(1, c["yield_emissions_total"])}
    return r


def train_label_stats() -> dict:
    from audioforge.datasets.ami import AMI, subset
    ds = AMI(subset({"train": 12})["train"], verbose=False, audio=False)
    # the recipe's windows without audio: count tokens from the words directly (same rule as turn_examples)
    from audioforge.datasets.ami import action_transcript
    n_h = n_y = n_w = 0
    per = []
    for m in ds.meetings:
        for t in ds.turns[m]:
            if t["bc"]:
                continue
            s = action_transcript(t["words"])
            h = s.count("<HOLD>")
            n_h += h
            n_y += 1
            n_w += t["n_words"]
            per.append(h)
    return {"meetings": len(ds.meetings), "turns": n_y, "hold_tokens": n_h, "yield_tokens": n_y, "words": n_w,
            "turns_with_hold": int(np.sum(np.array(per) > 0)),
            "note": "whole turns (windows crop the lead only; the recipe's 3274 16 s windows carry 454 <HOLD>)"}


# --------------------------------------------------------------------------- emission timing at the turn end
HIST_LO, HIST_HI, HIST_BIN = -0.32, 6.08, 0.16


def emission_timing(convs, last_word_end, fire_frames, emit_fn, pre_frames: int = 2) -> dict:
    """Lag of the first firing at a turn end relative to the LAST WORD's end (label word timings), per turn.
    ``fire_frames`` = per window the frames on which the detector fires (greedy emission frames, or frames where the
    score is >= the fold's fitted threshold); the firing considered is the first one at frame >= turn_end - pre_frames
    (i.e. inside the last 160 ms of the last word or later; earlier firings are the scorer's false cutoffs and are
    counted separately). Emission time = emit_fn(frame) * 0.08 s (the 160 ms chunk rule). Reports P10/P50/P90 of the
    lag, a histogram in 160 ms bins, the share of firings at <= 160 ms after the word end (from the phonemes, not the
    silence), and the median in-turn pause (hes run) length on those turns."""
    from audioforge.conversation import pause_runs
    lags, in_turn, pauses, turns_fired = [], 0, [], 0
    for v, lw, ff in zip(convs, last_word_end, fire_frames):
        on, en = int(v["onset_frame"]), int(v["turn_end_frame"])
        ff = np.asarray(ff, np.int64)
        in_turn += int(((ff >= on) & (ff < en - pre_frames)).any())
        cand = ff[ff >= en - pre_frames]
        if len(cand) == 0:
            continue
        turns_fired += 1
        lags.append(emit_fn(int(cand[0])) * B.FRAME - lw)
        pauses += [(b_ - a_) * B.FRAME for a_, b_ in pause_runs(v["hes"])]
    lags = np.asarray(lags)
    edges = np.round(np.arange(HIST_LO, HIST_HI + 1e-9, HIST_BIN), 2)
    hist = np.histogram(lags, edges)[0] if len(lags) else np.zeros(len(edges) - 1, int)
    return {"turns": len(convs), "turns_fired_at_end": turns_fired, "turns_fired_in_turn": in_turn,
            "lag_p10_p50_p90_s": [round(float(x), 3) for x in np.percentile(lags, [10, 50, 90])] if len(lags) else None,
            "lag_mean_s": round(float(lags.mean()), 3) if len(lags) else None,
            "share_within_160ms_of_word_end": round(float((lags <= 0.16).mean()), 4) if len(lags) else None,
            "share_before_word_end": round(float((lags <= 0.0).mean()), 4) if len(lags) else None,
            "median_in_turn_pause_s_on_fired_turns": round(float(np.median(pauses)), 3) if pauses else None,
            "n_in_turn_pauses_on_fired_turns": len(pauses),
            "hist_160ms": {"edges_s": edges.tolist(), "counts": hist.tolist(),
                           "below": int((lags < HIST_LO).sum()), "above": int((lags >= HIST_HI).sum())}}


def _fold_thresholds(sysrow: dict, folds) -> list[float]:
    th = sysrow["thresholds_by_fold"]
    return [float(th[str(int(f))]["exact"] if str(int(f)) in th else th[int(f)]["exact"]) for f in folds]


# --------------------------------------------------------------------------- report
YIELD_SINGLES = ["yield_posterior", "yield_vs_hold"]
REF = BT.REF


def stage_report(a, base, ext, meta, meta_base, work, ds=None):
    t0 = time.time()
    tr = BT.build_tracks(ext, BT_WORK)
    ys = []
    for ex in ext:
        z = np.load(work / "yield" / f"{BT.key(ex)}.npz")
        lp, em = z["logpost"], z["emitted"]
        ys.append({"yield_posterior": (lp[:, 0].astype(np.float64), "eou"),
                   "yield_vs_hold": ((lp[:, 0] - lp[:, 1]).astype(np.float64), "eou"),
                   "yield_emitted": (em[:, 0].astype(np.float32), "eou"), "hold_emitted": (em[:, 1].astype(np.float32), "eou")})
    for d, y in zip(tr, ys):
        d.update(y)
    ref = json.loads(BT.REF_JSON.read_text())["turn_v2"]
    out = {"protocol": "eot-bench v2 (research/EOT_BENCH_V2.md); yield/hold tokens on the same windows / scorer as "
                       "runs/baselines_turn.json", "ckpt": str(CKPT.relative_to(ROOT)), "n_boot": a.n_boot}
    for blk, convs, mt, hz, two in (("C_extended_windows_stream", ext, meta, {"2s": 25, "6s": 75}, False),
                                    ("A_default_windows_all", base, meta_base, {"2s": None}, True)):
        T = [len(v["spk_act"]) for v in convs]
        tracks = ([E.v2_base_track(v) for v in convs] if two else [np.load(E.v2_track_path(BT.TRAIL6_WORK, v)) for v in convs])
        singles, hybrids = BT.our_systems(convs, tracks, "trail6", two)
        col = lambda nm: [d[nm][0][:t] for d, t in zip(tr, T)]  # noqa: E731
        emit = lambda nm: BT.EMIT[tr[0][nm][1]]  # noqa: E731
        for nm in ("silero_timeout", "eou_posterior") + tuple(YIELD_SINGLES):
            singles[nm] = (col(nm), emit(nm))
        hybrids["yield_posterior+silero_timeout"] = (col("yield_posterior"), emit("yield_posterior"), col("silero_timeout"), emit("silero_timeout"))
        hybrids["yield_posterior+our_timeout"] = (col("yield_posterior"), emit("yield_posterior")) + singles["timeout_stream_causal_dominant"]
        hybrids["eou_posterior+silero_timeout"] = (col("eou_posterior"), emit("eou_posterior"), col("silero_timeout"), emit("silero_timeout"))
        hybrids["head_trail6+silero_timeout"] = singles["head_v3_stream_causal_dominant"] + (col("silero_timeout"), emit("silero_timeout"))
        nat = {"yield_native (<YIELD> emitted)": (col("yield_emitted"), emit("yield_emitted"), 0.5, None, None, None),
               "yield_native OR 3 s Silero silence": (col("yield_emitted"), emit("yield_emitted"), 0.5, col("silero_timeout"),
                                                      emit("silero_timeout"), 37.5 - 1e-6),
               "eou_native (<EOU> emitted)": (col("eou_emitted"), emit("eou_emitted"), 0.5, None, None, None)}
        mine = YIELD_SINGLES + ["yield_posterior+silero_timeout", "yield_posterior+our_timeout"]
        pairs = [(s, r) for s in mine for r in (REF, "eou_posterior", "silero_timeout", "timeout_stream_causal_dominant",
                                                "head_trail6+silero_timeout", "eou_posterior+silero_timeout")]
        r = BT.score_block(convs, mt, singles, hybrids, hz, a.n_boot, pairs, nat)
        for s in BT.OURS:  # our rows and the two reused baselines must reproduce the committed runs exactly
            for h in hz:
                x, y = r["systems"][s][h]["fixed_5pct_turn_fc"], ref[blk]["systems"][s][h]["fixed_5pct_turn_fc"]
                assert (x["miss_rate"], x["fc_rate"], x["miss_rate_ci"]) == (y["miss_rate"], y["fc_rate"], y["miss_rate_ci"]), (blk, s, h)
        bref = json.loads((ROOT / "runs" / "baselines_turn.json").read_text())[blk]["systems"]
        for s in ("silero_timeout", "eou_posterior"):
            for h in hz:
                x, y = r["systems"][s][h]["fixed_5pct_turn_fc"], bref[s][h]["fixed_5pct_turn_fc"]
                assert (x["miss_rate"], x["miss_rate_ci"]) == (y["miss_rate"], y["miss_rate_ci"]), (blk, s, h)
        r["reproduced"] = ["ours: runs/turn_trail6_hybrid_leakfree.json", "baselines: runs/baselines_turn.json"]
        out[blk] = r
        print(f"  {blk}: {time.time() - t0:.0f}s", flush=True)
    if ds is not None:  # emission timing at the turn end (block C, 6 s operating point)
        lw = [max(e for _, e, _ in w) for w in BT.turn_words(ds, ext)]
        folds = np.array([0 if v["meeting"] in E.V2_DEV_FOLDS[0] else 1 for v in ext])
        en = np.array([v["turn_end_frame"] for v in ext])
        emit_eou = BT.EMIT["eou"]
        timing = {}
        for nm, sc_name, em_name in (("yield", "yield_posterior", "yield_emitted"), ("parakeet_eou", "eou_posterior", "eou_emitted")):
            row = out["C_extended_windows_stream"]["systems"][sc_name]["6s"]["fixed_5pct_turn_fc"]
            ths = _fold_thresholds(row, folds)
            greedy = [np.nonzero(d[em_name][0] > 0.5)[0] for d in tr]
            z = E._emit_transform([d[sc_name][0] for d in tr], en, emit_eou)  # scores on the emission grid, as scored
            cross = [np.nonzero(np.asarray(zz) >= th)[0] for zz, th in zip(z, ths)]
            timing[nm] = {"greedy_emission": emission_timing(ext, lw, greedy, emit_eou),
                          # post-end index j of the transformed track = emission frame j + 1 (eval_stage1._emit_transform)
                          "threshold_crossing_6s_crossfit": emission_timing(ext, lw, cross, lambda t: t + 1),
                          "thresholds_by_fold_exact": row["thresholds_by_fold"]}
        timing["note"] = ("lag = first firing at frame >= turn_end - 2 (last 160 ms of the last word or later); emission time = "
                          "(t // 2 + 1) * 160 ms for greedy tokens, the scorer's emission grid for threshold crossings; "
                          "last word end from the AMI word timings")
        out["emission_timing"] = timing
    ems = [(d["yield_emitted"][0] > 0.5, d["hold_emitted"][0] > 0.5) for d in tr]
    out["tokens_at_sites"] = {"yieldhold": pause_site_stats(ext, ems),
                              "parakeet_eou": pause_site_stats(ext, [(d["eou_emitted"][0] > 0.5, None) for d in tr])}
    out["train_labels"] = train_label_stats()
    if (work / "wer.json").exists():
        out["wer"] = json.loads((work / "wer.json").read_text())
    if (work / "segments.json").exists():
        out["single_speaker_segments"] = json.loads((work / "segments.json").read_text())
    out["timing"] = json.loads((work / "timing.json").read_text()) if (work / "timing.json").exists() else {}
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    global CKPT
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["decode", "wer", "segments", "report"])
    p.add_argument("--work", default=str(BT.SCRATCH / "yield_tokens" / "work"))
    p.add_argument("--budget", type=float, default=520.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(ROOT / "runs" / "yield_tokens.json"))
    p.add_argument("--ckpt", default=str(CKPT), help="the yield/hold model (default runs/stage1_rnnt_yieldhold.afm)")
    a = p.parse_args()
    CKPT = Path(a.ckpt).resolve()
    BT._torch2()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, meta_base = BT.load()
    if a.stage == "report":
        res = stage_report(a, base, ext, meta, meta_base, work, ds)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")
        return
    {"decode": stage_decode, "wer": stage_wer, "segments": stage_segments}[a.stage](a, ext, ds, work)


if __name__ == "__main__":
    main()
