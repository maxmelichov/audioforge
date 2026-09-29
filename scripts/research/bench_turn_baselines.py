"""Dedicated open turn-detection models vs our turn detection on eot-bench v2 (research/BASELINES.md, "Turn detection").

Same data and scorer as scripts/research/eval_stage1.py --bench v2 (research/EOT_BENCH_V2.md): the 974 AMI dev turn windows
(block A: default 2 s-trail windows, horizon = the window's trail; block C: 6 s-extended windows, horizons 25 / 75
post-end emission frames), label-free causal enrollment where a speaker track is needed, fixed threshold at <= 5 %
false cutoffs per turn chosen on one leave-meetings-out half and applied to the other, floor-open / taken strata,
bootstrap CIs, paired bootstraps vs our hybrid. Every baseline is a per-frame (80 ms) score track
(audioforge/baselines/turn.py); our rows are recomputed from the stored trail6 head scores and streaming Sortformer
tracks and asserted equal to runs/turn_trail6_hybrid_leakfree.json.

Stages (each <= 10 min, resumable; one model per process; CPU, 2 threads):
  vad        Silero VAD v5 probabilities (32 ms) of every extended window
  smartturn  Pipecat smart-turn v3.2 at (a) Pipecat-VAD stops (Silero, speaker-unaware) and (b) 200 ms silences of
             the causal-bound Sortformer primary track (speaker-aware trigger); <= 8 s of audio before the trigger
  lkaudio    LiveKit turn-detector-v1-mini (audio, 1.2 s) at LiveKit-Silero end-of-speech events
  eou        nvidia/parakeet_realtime_eou_120m-v1 (nemo_import.import_eou): per-frame <EOU> posterior / emission
  asr        our streaming RNNT (runs/stage1_heads_pretrained.afm, 160 ms): tokens emitted per frame
  lktext     LiveKit text EOU (EnglishModel, MultilingualModel) on (a) our ASR partials, (b) oracle primary words,
             at every frame after a LiveKit-Silero end-of-speech (until speech resumes)
  report     score everything -> runs/baselines_turn.json

  W=<scratch>/baselines_turn
  .venv/bin/python scripts/research/bench_turn_baselines.py --stage vad --work $W/work
  ... --stage smartturn | lkaudio | eou | asr | lktext   (repeat a stage until it prints "0 left")
  .venv/bin/python scripts/research/bench_turn_baselines.py --stage report --work $W/work --out runs/baselines_turn.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import eval_stage1 as E  # noqa: E402
from audioforge.baselines import turn as B  # noqa: E402

SCRATCH = Path(os.environ.get("AUDIOFORGE_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch"))  # machine rule: scratch on the SSD
TRAIL6_WORK = SCRATCH / "trail6" / "work"  # streaming Sortformer tracks (1.04 s config) + trail6 head scores
REF_JSON = ROOT / "runs" / "turn_trail6_hybrid_leakfree.json"
SILERO_V5 = ROOT / "data" / "silero" / "silero_vad_v5.onnx"  # github snakers4/silero-vad v5.1.2
SMART_TURN = SCRATCH / "baselines_turn" / "models" / "smart-turn-v3" / "smart-turn-v3.2-cpu.onnx"
EOU_NEMO = ROOT / "data" / "nemo" / "parakeet_realtime_eou_120m-v1.nemo"
ASR_CKPT = ROOT / "runs" / "stage1_heads_pretrained.afm"
C_SF, R_SF = 6, 7  # Sortformer low-latency config (chunk, right context)
EOU_CHUNK = 2  # att_context [70, 1]: 2-frame (160 ms) chunks
PAD_SEC = 0.32  # real meeting audio fed after the window to streaming ASR models (no end-of-file flush), cropped
SF_TRIGGER_FRAMES = 3  # >= 200 ms of primary silence on the 80 ms grid
LK_AUDIO_MIN_SILENCE = 0.2  # livekit.agents.inference.eot.base.MIN_SILENCE_DURATION_MS
LK_MAX_DELAY = 3.0  # LiveKit endpointing max_delay (s); Pipecat SmartTurnParams.stop_secs is also 3 s
LK_MODELS = ("en", "multilingual")
TEXT_SOURCES = ("asr", "oracle")


def key(ex) -> str:
    from audioforge.datasets import ext_tracks as xt
    return xt.example_key(ex)


def load():
    base, ext, meta, ds, meta_base = E.v2_data(6.0)
    return base, ext, meta, ds, meta_base


def padded_audio(ds, ex, pad: float = PAD_SEC) -> np.ndarray:
    a = float(ex["start"])
    b = a + len(ex["audio"]) / B.SR
    return np.asarray(ds._clip(ex["meeting"], a, min(ds.duration(ex["meeting"]), b + pad)), np.float32)


def bound_act(ex) -> np.ndarray:
    p = np.load(E.v2_track_path(TRAIL6_WORK, ex))
    return E._v2_cols(ex, p, "causal_dominant")[0]


def turn_words(ds, ext) -> list[list[tuple]]:
    """Per extended window: the primary turn's words [(start, end, word)] relative to the window start (the words
    turn_windows put in ex['text'])."""
    out = []
    for ex in ext:
        a = float(ex["start"])
        hits = []
        for t in ds.turns[ex["meeting"]]:
            if t["bc"] or ds.gid(t["speaker"]) != ex["speaker"] or t["end"] <= a:
                continue
            w = [x for x in t["words"] if x[1] > a]
            if " ".join(x[2] for x in w) == ex["text"] and abs((t["end"] - a) / B.FRAME - ex["turn_end_frame"]) < 3:
                hits.append(w)
        assert len(hits) == 1, (key(ex), len(hits))
        out.append([(s - a, e - a, word) for s, e, word in hits[0]])
    return out


def run_resumable(name, items, fn, work: Path, budget: float, suffix=".npz"):
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    todo = [ex for ex in items if not (d / f"{key(ex)}{suffix}").exists()]
    t0, n = time.time(), 0
    for ex in todo:
        if time.time() - t0 > budget:
            break
        fn(ex, d / f"{key(ex)}{suffix}")
        n += 1
    print(f"  {name}: {n} done in {time.time() - t0:.0f}s, {len(todo) - n} left", flush=True)
    return len(todo) - n


def save_npz(path: Path, **kw):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **kw)
    tmp.replace(path)


def add_timing(work: Path, name: str, calls: int, sec: float, **extra):
    p = work / "timing.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    r = d.get(name, {"calls": 0, "sec": 0.0})
    r["calls"] += int(calls)
    r["sec"] += float(sec)
    r.update(extra)
    d[name] = r
    p.write_text(json.dumps(d, indent=1))


# --------------------------------------------------------------------------- stages
def stage_vad(a, ext, ds, work):
    vad = B.SileroVAD(SILERO_V5)
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        x = np.asarray(ex["audio"], np.float32)
        t0 = time.perf_counter()
        p = vad.probs(x)
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += len(p)
        np.save(out.with_name(out.stem + ".tmp.npy"), p)
        out.with_name(out.stem + ".tmp.npy").replace(out)
    left = run_resumable("silero", ext, fn, work, a.budget, ".npy")
    add_timing(work, "silero_chunk", stat["calls"], stat["sec"])
    return left


def pipecat_triggers(p) -> tuple[np.ndarray, np.ndarray]:
    """(stop decision times s, resume decision times s) of the Pipecat VAD."""
    v = B.pipecat_vad(p)
    return (v["stops"] + 1) * B.CHUNK_SEC, (v["starts"] + 1) * B.CHUNK_SEC


def stage_smartturn(a, ext, ds, work):
    st = B.SmartTurn(SMART_TURN)
    stat = {"calls": 0, "sec": 0.0}

    def pred(x, tau):
        t0 = time.perf_counter()
        v = st.predict(x[max(0, int(round((tau - 8.0) * B.SR))): int(round(tau * B.SR))])
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += 1
        return v

    def fn(ex, out):
        x = np.asarray(ex["audio"], np.float32)
        p = np.load(work / "silero" / f"{key(ex)}.npy")
        stops, _ = pipecat_triggers(p)
        pc = np.array([pred(x, tau) for tau in stops], np.float32)
        trig, _ = B.frame_silence_triggers(bound_act(ex), SF_TRIGGER_FRAMES)
        sf = np.array([pred(x, (t + 1) * B.FRAME) for t in trig], np.float32)
        save_npz(out, pc_tau=stops, pc_p=pc, sf_frames=trig, sf_p=sf)
    left = run_resumable("smartturn", ext, fn, work, a.budget)
    add_timing(work, "smartturn_call", stat["calls"], stat["sec"], model=str(SMART_TURN.name))
    return left


def stage_lkaudio(a, ext, ds, work):
    m = B.LiveKitAudio()
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        x = np.asarray(ex["audio"], np.float32)
        v = B.livekit_vad(np.load(work / "silero" / f"{key(ex)}.npy"))
        taus = (v["eos"] + 1) * B.CHUNK_SEC
        # LiveKit requests the prediction once the raw accumulated silence reaches MIN_SILENCE_DURATION_MS (200 ms)
        # while the VAD still reports speaking, on the last 1.2 s; the result is used at END_OF_SPEECH (tau)
        t_in = np.minimum(v["speech_end"] + LK_AUDIO_MIN_SILENCE, taus)
        ps = []
        for ti in t_in:
            t0 = time.perf_counter()
            ps.append(m.predict(x[: int(round(ti * B.SR))]))
            stat["sec"] += time.perf_counter() - t0
            stat["calls"] += 1
        save_npz(out, tau=taus, t_in=t_in, p=np.array(ps, np.float32))
    left = run_resumable("lkaudio", ext, fn, work, a.budget)
    add_timing(work, "lkaudio_call", stat["calls"], stat["sec"])
    return left


def _torch2():
    import torch
    torch.set_num_threads(2)
    return torch


def stage_eou(a, ext, ds, work):
    _torch2()
    from audioforge.nemo_import import import_eou
    m = import_eou(EOU_NEMO)
    ids = [m.eou_ids["<EOU>"], m.eou_ids["<EOB>"]]
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        T = len(ex["spk_act"])
        x = padded_audio(ds, ex)
        t0 = time.perf_counter()
        toks, logpost, em = B.rnnt_frame_decode(m, x, watch_ids=ids)
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += 1
        logpost, em = B_fit(logpost, T, -np.inf), B_fit(em, T, False)
        text = m.tokenizer.decode([k for tt in toks[:T] for k in tt])
        save_npz(out, logpost=logpost, emitted=em, text=np.array(text))
    left = run_resumable("eou", ext, fn, work, a.budget)
    add_timing(work, "eou_offline_window", stat["calls"], stat["sec"])
    if not left:
        eou_stream_timing(m, ext, work)
    return left


def eou_stream_timing(m, ext, work, n=5):
    """Per-160 ms-chunk compute of the EOU model in audioforge's streaming path (model.StreamingSession)."""
    p = work / "timing.json"
    if p.exists() and "eou_stream_chunk" in json.loads(p.read_text()):
        return
    from audioforge.model import StreamingSession
    calls, sec = 0, 0.0
    for ex in ext[:n]:
        x = np.asarray(ex["audio"], np.float32)
        ss = StreamingSession(m)
        step = 2560
        for s in range(0, len(x) - step + 1, step):
            t0 = time.perf_counter()
            ss.feed(x[s:s + step])
            sec += time.perf_counter() - t0
            calls += 1
    add_timing(work, "eou_stream_chunk", calls, sec, path="model.StreamingSession, 160 ms feeds")


def B_fit(x, T, fill=0):
    """Crop / pad (with ``fill``: nothing decoded there) a per-frame array to T frames."""
    x = np.asarray(x)
    if len(x) >= T:
        return x[:T]
    return np.concatenate([x, np.full((T - len(x),) + x.shape[1:], fill, x.dtype)])


def stage_asr(a, ext, ds, work):
    _torch2()
    from audioforge.train import load_model
    m = load_model(str(ASR_CKPT), "cpu")
    assert list(m.encoder.att_context_size) == [70, 1], m.encoder.att_context_size
    stat = {"calls": 0, "sec": 0.0}

    def fn(ex, out):
        T = len(ex["spk_act"])
        x = padded_audio(ds, ex)
        t0 = time.perf_counter()
        toks, _, _ = B.rnnt_frame_decode(m, x)
        stat["sec"] += time.perf_counter() - t0
        stat["calls"] += 1
        toks = (toks + [[] for _ in range(T)])[:T]
        tmp = out.with_name(out.stem + ".tmp.json")
        tmp.write_text(json.dumps(toks))
        tmp.replace(out)
    left = run_resumable("asr", ext, fn, work, a.budget, ".json")
    add_timing(work, "asr_offline_window", stat["calls"], stat["sec"])
    if not left:  # the partial text per frame needs the tokenizer: store it once (no model needed later)
        tok = m.tokenizer
        d = work / "asr"
        for ex in ext:
            q = d / f"{key(ex)}.text.json"
            if q.exists():
                continue
            toks = json.loads((d / f"{key(ex)}.json").read_text())
            q.write_text(json.dumps(asr_partials(toks, tok)))
    return left


def asr_partials(toks, tok) -> list[str]:
    """Partial transcript available at each frame t: tokens of frames u with emission (u // 2 + 1) * 2 <= t + 1."""
    T = len(toks)
    out, acc, u = [], [], 0
    for t in range(T):
        while u < T and (u // EOU_CHUNK + 1) * EOU_CHUNK <= t + 1:
            acc += toks[u]
            u += 1
        out.append(tok.decode(acc))
    return out


def lk_gate(p, T):
    """LiveKit gate on the frame grid: frames at/after an end-of-speech decision and before speech restarts, plus
    per frame the silence since the speech end (s) for the max_delay timeout; 0 / -1 elsewhere."""
    v = B.livekit_vad(p)
    gate = np.zeros(T, bool)
    since = np.zeros(T, np.float32)
    sos_t = (v["sos"] + 1) * B.CHUNK_SEC
    for j, se in zip(v["eos"], v["speech_end"]):
        tau = (j + 1) * B.CHUNK_SEC
        a0 = B.frame_of_time(tau)
        k = np.searchsorted(sos_t, tau, side="right")
        b0 = T if k >= len(sos_t) else B.frame_of_time(sos_t[k])
        if a0 >= T:
            continue
        gate[a0:b0] = True
        fe = (np.arange(a0, min(b0, T)) + 1) * B.FRAME
        since[a0:b0] = fe - se
    return gate, since


def oracle_partials(words, T) -> list[str]:
    ends = np.array([e for _, e, _ in words])
    out = []
    for t in range(T):
        n = int(np.sum(ends <= (t + 1) * B.FRAME + 1e-9))
        out.append(" ".join(w for _, _, w in words[:n]))
    return out


def stage_lktext(a, ext, ds, work):
    words = turn_words(ds, ext)
    wmap = {key(ex): w for ex, w in zip(ext, words)}
    left = 0
    for mname in LK_MODELS:
        m = B.LiveKitText(mname)

        def fn(ex, out, m=m):
            T = len(ex["spk_act"])
            gate, _ = lk_gate(np.load(work / "silero" / f"{key(ex)}.npy"), T)
            srcs = {"asr": json.loads((work / "asr" / f"{key(ex)}.text.json").read_text()),
                    "oracle": oracle_partials(wmap[key(ex)], T)}
            res = {}
            for s, texts in srcs.items():
                sc = np.zeros(T, np.float32)
                for t in np.nonzero(gate)[0]:
                    sc[t] = m.predict(texts[t])
                res[s] = sc
            save_npz(out, **res, threshold=np.array(m.threshold))
        left += run_resumable(f"lktext_{mname}", ext, fn, work, a.budget / len(LK_MODELS))
        add_timing(work, f"lktext_{mname}_call", m.calls, m.sec, revision=m.revision, threshold=m.threshold)
    return left


# --------------------------------------------------------------------------- score tracks
def build_tracks(ext, work):
    """Per extended window: {system: (scores (T,), emission kind)}; 'frame' = emit t + 1, 'sf' = Sortformer stream
    rule, 'eou' = 160 ms chunks. Timeouts: 'to_pc' Pipecat-VAD silence, 'to_lk' LiveKit silence since speech end."""
    out = []
    for ex in ext:
        k, T = key(ex), len(ex["spk_act"])
        p = np.load(work / "silero" / f"{k}.npy")
        vp = B.pipecat_vad(p)
        to_pc = B.silence_frames(B.speech_chunks_pipecat(vp), T)
        _, resumes = pipecat_triggers(p)
        s = np.load(work / "smartturn" / f"{k}.npz")
        st_pc = B.held_scores(T, list(zip(s["pc_tau"], s["pc_p"])), resumes)
        act = bound_act(ex)
        trig, res = B.frame_silence_triggers(act, SF_TRIGGER_FRAMES)
        assert np.array_equal(trig, s["sf_frames"])
        st_sf = B.held_frames(T, trig, s["sf_p"], res)
        gate, since = lk_gate(p, T)
        to_lk = np.where(gate, since / B.FRAME, 0.0).astype(np.float32)
        la = np.load(work / "lkaudio" / f"{k}.npz")
        v = B.livekit_vad(p)
        sos_t = (v["sos"] + 1) * B.CHUNK_SEC
        lk_audio = B.held_scores(T, list(zip(la["tau"], la["p"])), sos_t)
        eo = np.load(work / "eou" / f"{k}.npz")
        d = {"silero_timeout": (to_pc, "frame"), "to_lk": (to_lk, "frame"),
             "smartturn_silero": (st_pc, "frame"), "smartturn_sortformer": (st_sf, "sf"),
             "livekit_audio_mini": (lk_audio, "frame"),
             "eou_posterior": (eo["logpost"][:, 0], "eou"),  # log P(<EOU>) (monotone in P)
             "eou_emitted": (eo["emitted"][:, 0].astype(np.float32), "eou")}
        for mname in LK_MODELS:
            lt = np.load(work / f"lktext_{mname}" / f"{k}.npz")
            for src in TEXT_SOURCES:
                d[f"livekit_text_{mname}_{src}"] = (lt[src].astype(np.float32), "frame")
        out.append(d)
    return out


EMIT = {"frame": lambda t: t + 1, "sf": E._stream_emit(C_SF, R_SF),
        "eou": lambda t: (t // EOU_CHUNK + 1) * EOU_CHUNK}

# (display name, detector a, detector b or None, speaker awareness, deployed OR-timeout partner)
BASELINES = [
    ("silero_timeout", "silero_timeout", None),
    ("smartturn_silero", "smartturn_silero", None),
    ("smartturn_silero+timeout", "smartturn_silero", "silero_timeout"),
    ("smartturn_sortformer", "smartturn_sortformer", None),
    ("smartturn_sortformer+timeout", "smartturn_sortformer", "OUR_TIMEOUT"),
    ("livekit_audio_mini", "livekit_audio_mini", None),
    ("livekit_audio_mini+timeout", "livekit_audio_mini", "to_lk"),
    ("eou_posterior", "eou_posterior", None),
] + [(f"livekit_text_{m}_{s}", f"livekit_text_{m}_{s}", None) for m in LK_MODELS for s in TEXT_SOURCES] \
  + [(f"livekit_text_{m}_{s}+timeout", f"livekit_text_{m}_{s}", "to_lk") for m in LK_MODELS for s in TEXT_SOURCES]

OURS = ["timeout_stream_causal_dominant", "head_v3_stream_causal_dominant", "hybrid_stream_causal_dominant",
        "timeout_any_speaker_stream"]
REF = "hybrid_stream_causal_dominant"

NATIVE = {  # out-of-the-box operating points (no threshold fitting): (detector a, θ_a, detector b, θ_b)
    "pipecat_native (smart-turn P>0.5 OR 3 s silence)": ("smartturn_silero", 0.5, "silero_timeout", 37.5 - 1e-6),
    "livekit_audio_native (P>=0.36 at EOS OR 3 s)": ("livekit_audio_mini", B.LiveKitAudio.THRESHOLD_EN - 1e-9,
                                                     "to_lk", LK_MAX_DELAY / B.FRAME - 1e-6),
    "eou_native (<EOU> emitted)": ("eou_emitted", 0.5, None, None),
}


def our_systems(convs, tracks, work_scores_tag, two_s: bool):
    """Our rows exactly as eval_stage1.v2_block builds them (causal binding): scores + emission functions."""
    tracks = [np.stack([E._fit(p[:, j], len(v["spk_act"])) for j in range(p.shape[1])], 1) for v, p in zip(convs, tracks)]
    acts = [E._v2_cols(v, p, "causal_dominant")[0] for v, p in zip(convs, tracks)]
    es, eh = E._stream_emit(C_SF, R_SF), E._stream_emit(C_SF, R_SF, 2)  # --v2-chunk 2 (160 ms head)
    sub = "causal_dominant" + ("@2s" if two_s else "")
    head = [np.load(E.v2_score_path(TRAIL6_WORK, sub, v, work_scores_tag)) for v in convs]
    to = [E.silence_scores(x, E.arm_frame(x)) for x in acts]
    anys = [E.silence_scores(p.max(1), E.arm_frame(p.max(1))) for p in tracks]
    return {"timeout_stream_causal_dominant": (to, es), "head_v3_stream_causal_dominant": (head, eh),
            "timeout_any_speaker_stream": (anys, es)}, {"hybrid_stream_causal_dominant": (head, eh, to, es)}


def score_block(convs, meta, singles: dict, hybrids: dict, horizons: dict, n_boot: int, pairs, natives=None):
    """eval_stage1.v2_block's scoring of given systems (same folds, strata, grids, cross-fit, bootstrap seeds)."""
    from audioforge.conversation import (bootstrap_ci, eot_outcomes, eot_outcomes_or, floor_stratum, outcome_metrics,
                                         pause_runs)
    n = len(convs)
    on = np.array([v["onset_frame"] for v in convs])
    en = np.array([v["turn_end_frame"] for v in convs])
    strata = np.array([floor_stratum(v["spk_targets"], int(e), E.V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
    groups = {"open": strata == "open", "taken": strata != "open"}
    pauses = [pause_runs(v["hes"]) for v in convs]
    folds = np.array([0 if v["meeting"] in E.V2_DEV_FOLDS[0] else 1 for v in convs])
    avail = np.array([m["post_avail"] for m in meta])
    reason = np.array([m["end_reason"] for m in meta])
    res = {"n": n, "strata_counts": {g: int(m.sum()) for g, m in groups.items()},
           "n_pauses": int(sum(len(p) for p in pauses)), "systems": {}, "paired": {}, "native": {}}
    outs = {}

    def fixed(name, r, hz, L, cens, oc, th_fmt=None):
        fc, lat, pf, th = E.v2_crossfit(oc, folds, 0.05, "turn", tie_miss=th_fmt is not None)
        if th_fmt is not None:
            th = {f: th_fmt(j) for f, j in th.items()}
        else:  # v2_crossfit rounds to 5 decimals; keep the exact values too (tiny posteriors)
            th = {f: {"rounded": v, "exact": float(oc["ths"][E._select(oc, folds != f, 0.05, "turn")])}
                  for f, v in th.items()}
        pt = outcome_metrics(fc, lat, pf, oc["npause"])
        pt.update(bootstrap_ci(fc, lat, n_boot, 0), thresholds_by_fold=th)
        m = np.isinf(lat) & ~fc
        pt.update(n_miss=int(m.sum()), miss_censored=int((m & cens).sum()))
        pt["strata"] = {g: {**outcome_metrics(fc[msk], lat[msk], pf[msk], oc["npause"][msk]),
                            **bootstrap_ci(fc[msk], lat[msk], n_boot, 0)} for g, msk in groups.items()}
        r[hz] = {"fixed_5pct_turn_fc": pt}
        outs[(name, hz)] = (fc, lat)

    for name, spec in list(singles.items()) + list(hybrids.items()):
        r = {}
        for hz, L in horizons.items():
            cens = ((avail < L) if L is not None else np.ones(n, bool)) & (reason != "resume")
            if name in hybrids:
                ha, ea, tb, eb = spec
                za, zb = E._emit_transform(ha, en, ea), E._emit_transform(tb, en, eb)
                ga, gb = E.v2_hybrid_grid(za, on, en, pauses), E.v2_hybrid_grid(zb, on, en, pauses)
                oc = eot_outcomes_or(za, zb, on, en, ga, gb, post_end_frames=L, pauses=pauses)

                def th_fmt(j, oc=oc):
                    ta, tb_ = oc["ths"][int(round(j))]
                    return {"theta_a": float(ta) if np.isfinite(ta) else None,
                            "theta_b": float(tb_) if np.isfinite(tb_) else None}
                oc_i = dict(oc, ths=np.arange(oc["fc"].shape[1], dtype=np.float64))
                fixed(name, r, hz, L, cens, oc_i, th_fmt=th_fmt)
                continue
            sc, emit = spec
            z = E._emit_transform(sc, en, emit)
            allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in z]))
            ths = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
            oc = eot_outcomes(z, on, en, ths, post_end_frames=L, pauses=pauses)
            fixed(name, r, hz, L, cens, oc)
        res["systems"][name] = r
    for name, (sa, ea, ta, sb, eb, tb) in (natives or {}).items():
        r = {}
        for hz, L in horizons.items():
            za = E._emit_transform(sa, en, ea)
            if sb is None:
                oc = eot_outcomes(za, on, en, [ta], post_end_frames=L, pauses=pauses)
            else:
                zb = E._emit_transform(sb, en, eb)
                oc = eot_outcomes_or(za, zb, on, en, [ta], [tb], post_end_frames=L, pauses=pauses)
            fc, lat, pf = oc["fc"][:, 0], oc["lat"][:, 0], oc["pf"][:, 0]
            pt = outcome_metrics(fc, lat, pf, oc["npause"])
            pt.update(bootstrap_ci(fc, lat, n_boot, 0))
            pt["strata"] = {g: {**outcome_metrics(fc[m], lat[m], pf[m], oc["npause"][m]),
                                **bootstrap_ci(fc[m], lat[m], n_boot, 0)} for g, m in groups.items()}
            r[hz] = pt
        res["native"][name] = r
    idx = np.random.default_rng(0).integers(0, n, (n_boot, n))
    for a_, b_ in pairs:
        if (a_, next(iter(horizons))) not in outs or (b_, next(iter(horizons))) not in outs:
            continue
        for hz in horizons:
            A, Bb = outs[(a_, hz)], outs[(b_, hz)]
            d = {"all": E._paired(A, Bb, idx)}
            for g in ("open", "taken"):
                sub = np.nonzero(groups[g])[0]
                gi = np.random.default_rng(1).integers(0, len(sub), (n_boot, len(sub)))
                d[g] = E._paired((A[0][sub], A[1][sub]), (Bb[0][sub], Bb[1][sub]), gi)
            res["paired"][f"{a_} - {b_} | {hz}"] = d
    return res


def stage_report(a, base, ext, meta, meta_base, work):
    t0 = time.time()
    tr = build_tracks(ext, work)
    ref = json.loads(REF_JSON.read_text())["turn_v2"]
    out = {"protocol": "eot-bench v2 (research/EOT_BENCH_V2.md), baselines on the same windows / scorer",
           "ours_from": str(REF_JSON.relative_to(ROOT)), "n_boot": a.n_boot}
    for blk, convs, mt, hz, two in (("C_extended_windows_stream", ext, meta, {"2s": 25, "6s": 75}, False),
                                    ("A_default_windows_all", base, meta_base, {"2s": None}, True)):
        T = [len(v["spk_act"]) for v in convs]
        tracks = ([E.v2_base_track(v) for v in convs] if two else [np.load(E.v2_track_path(TRAIL6_WORK, v)) for v in convs])
        singles, hybrids = our_systems(convs, tracks, "trail6", two)
        col = lambda nm: [d[nm][0][:t] for d, t in zip(tr, T)]  # noqa: E731
        emit = lambda nm: EMIT[tr[0][nm][1]]  # noqa: E731
        for disp, da, db in BASELINES:
            if db is None:
                singles[disp] = (col(da), emit(da))
            elif db == "OUR_TIMEOUT":
                hybrids[disp] = (col(da), emit(da)) + singles["timeout_stream_causal_dominant"]
            else:
                hybrids[disp] = (col(da), emit(da), col(db), emit(db))
        # exploratory: our head OR the speaker-unaware Silero timeout (instead of the Sortformer-primary timeout)
        hybrids["head_trail6+silero_timeout"] = singles["head_v3_stream_causal_dominant"] + (
            col("silero_timeout"), emit("silero_timeout"))
        nat = {}
        for nm, (da, ta, db, tb) in NATIVE.items():
            nat[nm] = (col(da), emit(da), ta, None if db is None else col(db), None if db is None else emit(db), tb)
        for mname in LK_MODELS:
            thr = float(np.load(work / f"lktext_{mname}" / f"{key(ext[0])}.npz")["threshold"])
            for src in TEXT_SOURCES:
                da = f"livekit_text_{mname}_{src}"
                nat[f"livekit_text_{mname}_{src}_native (P>={thr} at EOS OR 3 s)"] = (
                    col(da), emit(da), thr - 1e-9, col("to_lk"), emit("to_lk"), LK_MAX_DELAY / B.FRAME - 1e-6)
        pairs = [(d, REF) for d, _, _ in BASELINES] + [(d, "timeout_stream_causal_dominant") for d, _, _ in BASELINES] \
            + [("head_trail6+silero_timeout", REF), ("head_trail6+silero_timeout", "silero_timeout")]
        r = score_block(convs, mt, singles, hybrids, hz, a.n_boot, pairs, nat)
        # our rows must reproduce the committed leak-free run exactly
        for s in OURS:
            for h in hz:
                x, y = r["systems"][s][h]["fixed_5pct_turn_fc"], ref[blk]["systems"][s][h]["fixed_5pct_turn_fc"]
                assert (x["miss_rate"], x["fc_rate"], x["miss_rate_ci"]) == (y["miss_rate"], y["fc_rate"], y["miss_rate_ci"]), \
                    (blk, s, h, x["miss_rate"], y["miss_rate"])
        r["ours_reproduced"] = True
        out[blk] = r
        print(f"  {blk}: {time.time() - t0:.0f}s", flush=True)
    out["timing"] = json.loads((work / "timing.json").read_text()) if (work / "timing.json").exists() else {}
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=["vad", "smartturn", "lkaudio", "eou", "asr", "lktext", "report"])
    p.add_argument("--work", default=str(SCRATCH / "baselines_turn" / "work"))
    p.add_argument("--budget", type=float, default=520.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", default=str(ROOT / "runs" / "baselines_turn.json"))
    a = p.parse_args()
    _torch2()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, meta_base = load()
    if a.stage == "report":
        res = stage_report(a, base, ext, meta, meta_base, work)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(f"wrote {a.out}")
        return
    fn = {"vad": stage_vad, "smartturn": stage_smartturn, "lkaudio": stage_lkaudio, "eou": stage_eou,
          "asr": stage_asr, "lktext": stage_lktext}[a.stage]
    fn(a, ext, ds, work)


if __name__ == "__main__":
    main()
