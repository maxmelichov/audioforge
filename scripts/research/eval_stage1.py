"""Stage-1 real-speech evaluation of the fused front-end on AMI dev (unseen meetings / speakers).

Tasks (each reuses the repo's own metric code; nothing is re-implemented):
  turn   AMI turn windows: eot-bench sweep (conversation.eot_bench) of the turn head with oracle primary
         activity and with the model's own diarization (heads.turn.turn_scores act_source="diar"), plus the
         silence-timeout baselines (conversation.baselines: primary oracle, any-speaker oracle).
         --diar-ckpt <afm>: also the primary track of an EXTERNAL Sortformer (own preprocessor/encoder; offline
         pass = upper bound, StreamingDiarizer card low-latency = real time), column chosen by overlap with the
         oracle primary on [onset, turn_end) only, fed to the turn head as spk_act, plus the diarizer+timeout
         cascade (research/STAGE1.md, "Turn head with an external diarizer"). A turn head with act_columns > 1
         (TurnHead v3) is also fed all the diarizer's columns + the chosen column (its duration features are computed
         inside the head from the fed track). Extra non-learned cascade: "duration rule" = fire when the primary
         column has been silent >= k frames AND no other column is active (k swept like the timeout's).
         --diar-cache <dir> (e.g. data/ami/cache/sortformer/dev): read the streaming tracks from the per-example cache
         of scripts/research/make_sortformer_tracks.py instead of recomputing them (same procedure, identical arrays; the cheap
         offline pass is recomputed as without a cache); a .pkl path = the old resumable cache.
         --hybrid: also the OR decision rule "turn head p > θ OR primary silent >= k frames" (serve.py turn_policy
         "hybrid"), swept jointly over (θ, k) by conversation.eot_bench_or, on oracle activity (hybrid_oracle) and,
         with --diar-ckpt, on the Sortformer streaming track (external_diar.hybrid_sortformer_stream); operating points
         at <= 5 % per-turn and per-pause FC, a small (k, θ) grid, and the pure rows re-derived from the same sweep
         (asserted equal to turn_head_* / timeout_*). Off by default: the default output is unchanged.
  diar   AMI diar windows: frame DER (metrics.frame_der) of the Sortformer head (offline pass over the
         20 s window, as Trainer.evaluate), miss / FA / confusion at the best permutation, trivial
         baselines; VAD head frame accuracy / recall vs the derived any-speaker label (train.derive_labels).
  spk    AMI asr-mode single-speaker segments: speaker-verification EER (metrics.eer) over all pairs of
         speaker-head embeddings (cosine), all pairs and within-meeting pairs; baseline = mean-pooled
         layer-mix features of the same head input (no trained projection).
  asr    AMI asr-mode segments: WER of the frozen RNNT head (teachers.normalize_text on ref and hyp).

Usage (CPU, 2 threads, one process at a time; results are merged into --out, so tasks can be split
across processes to keep each under the time budget):
  TMPDIR=<scratch> .venv/bin/python scripts/research/eval_stage1.py --ckpt runs/stage1_heads_pretrained.step500.afm \
      --n 64 --tasks turn,diar,spk,asr --out runs/stage1_step500_eval.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audioforge.conversation import baselines, eot_bench, eot_bench_or, pause_runs, silence_scores  # noqa: E402
from audioforge.data import Collate, to_device  # noqa: E402
from audioforge.datasets.ami import recipe_data  # noqa: E402
from audioforge.heads.turn import _diar_name, _onset_end, turn_scores  # noqa: E402
from audioforge.metrics import eer, frame_der, wer  # noqa: E402
from audioforge.teachers import normalize_text  # noqa: E402
from audioforge.train import derive_labels, load_model  # noqa: E402


def ami_dev(mode: str, n: int, seed: int = 0) -> list[dict]:
    """The recipe path (data.val: {ami: {mode, val_split: dev, n_val}}): 4 default dev meetings, seeded cap."""
    return recipe_data({"data": {"ami": {"mode": mode, "val_split": "dev", "n_val": n, "seed": seed}}}, "val")


def _pt(b: dict) -> dict:
    """Operating points of an eot_bench result (no curve)."""
    return {"n": b["n"], "at_5pct_fc": b["at_max_fc"], "at_p50_le_400ms": b["at_fixed_latency"]}


def chunk_of(model) -> int:
    att = model.encoder.att_context_size
    return att[1] + 1 if att[1] >= 0 else 10 ** 6


# --------------------------------------------------------------------------- turn
@torch.no_grad()
def eval_turn(model, n: int, batch_size: int, diar_ckpt: str | None = None, diar_cache: Path | None = None,
              diar_budget: float | None = None, enroll_mode: str = "oracle", hybrid: bool = False) -> dict:
    val = ami_dev("turn", n)
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    on, en = _onset_end(val)  # as TurnHead.evaluate_model
    chunk = chunk_of(model)
    res = {"n": len(val), "chunk_frames": chunk, "turn_head": name}
    for src in ["oracle"] + (["diar"] if _diar_name(model) else []):
        st, t0 = {}, time.time()
        sc = turn_scores(model, name, val, act_source=src, batch_size=batch_size, stats=st)
        res[f"turn_head_{src}"] = _pt(eot_bench(sc, on, en, chunk=chunk))
        if src == "oracle":
            sc_oracle = sc
        if src == "diar" and st.get("speech"):
            res["diar_primary_act_miss"] = round(st["miss"] / st["speech"], 4)
        print(f"  turn {src}: {time.time() - t0:.0f}s {res[f'turn_head_{src}']}", flush=True)
    vv = [dict(v, onset_frame=o, turn_end_frame=e, primary_act=np.asarray(v["spk_act"]))
          for v, o, e in zip(val, on, en)]
    bl = baselines(vv)  # zero look-ahead silence timeouts on oracle activity
    res["timeout_primary_oracle"] = _pt(bl["timeout_primary_oracle"])
    res["timeout_any_speaker_oracle"] = _pt(bl["timeout_any_speaker_oracle"])
    if hybrid:  # head (turn-chunk emission) OR timeout (zero look-ahead), both on oracle primary activity
        res["hybrid_oracle"] = hybrid_rows(sc_oracle, [silence_scores(v["primary_act"], o) for v, o in zip(vv, on)],
                                           on, en, chunk, 1, [pause_runs(v["hes"]) for v in val],
                                           res["turn_head_oracle"], res["timeout_primary_oracle"])
    if _diar_name(model):
        res["diar_primary_act_miss_offline"] = offline_primary_miss(model, val, batch_size)
    post = np.array([len(v["spk_act"]) - e for v, e in zip(val, en)]) * 80.0  # measurable dead-air ceiling
    res["post_end_ms_min_p50_max"] = [float(post.min()), float(np.median(post)), float(post.max())]
    res["mean_sec"] = round(float(np.mean([len(v["audio"]) / 16000 for v in val])), 2)
    res["onset_clipped_frac"] = round(float(np.mean([v["onset_clipped"] for v in val])), 3)
    res["with_hesitation_frac"] = round(float(np.mean([v["n_hesitations"] > 0 for v in val])), 3)
    if diar_ckpt:
        res["external_diar"] = eval_turn_external(model, name, val, on, en, chunk, diar_ckpt, batch_size,
                                                  diar_cache, diar_budget, enroll_mode, hybrid)
    return res


@torch.no_grad()
def offline_primary_miss(model, val, batch_size) -> float:
    """Diagnostic: primary-frame miss of the diar head's arrival-rank column from ONE offline pass over the
    whole window (non-causal upper bound of the 'diar' source, which re-runs the head on growing prefixes)."""
    from audioforge.heads.turn import primary_column
    diar, col = _diar_name(model), Collate(model.tokenizer)
    miss = speech = 0
    for i in range(0, len(val), batch_size):
        b = col(val[i: i + batch_size])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        p = model.heads[diar](model.head_input(diar, enc, hidden), elen).sigmoid()
        c = primary_column(b["spk_targets"]).clamp(max=p.shape[2] - 1)
        T = min(p.shape[1], b["spk_act"].shape[1])
        act = p[:, :T].gather(2, c[:, None, None].expand(-1, T, 1))[..., 0]
        ref = (b["spk_act"][:, :T] > 0.5) & (torch.arange(T)[None] < elen[:, None])
        miss += int((ref & (act <= 0.5)).sum())
        speech += int(ref.sum())
    return round(miss / max(1, speech), 4)


# --------------------------------------------------------------------------- turn head + external diarizer
# NVIDIA streaming Sortformer v2 card "low latency" setting (research/SORTFORMER_IMPORT.md), in 80 ms frames.
SORTFORMER_LOW_LATENCY = dict(chunk_len=6, chunk_right_context=7, fifo_len=188, spkcache_update_period=144,
                              spkcache_len=188)
SORTFORMER_ENC_LEFT = 188  # window-mode encoder left context (the non-causal encoder is re-run per step)


def eot_bench_emit(scores, onsets, ends, emit_fn, **kw) -> dict:
    """eot_bench with an arbitrary per-frame emission time emit_fn(t) (monotone, >= t + 1), exactly.

    The pre-end part of each score track is kept (so false cutoffs are unchanged: a firing on a pre-end
    frame is a cutoff even when its decision is emitted after the end); the post-end part is re-indexed
    by emission time: z[k] = max score over post-end frames t with emit(t) - end <= k + 1. eot_bench with
    chunk = 1 then reports latency (k + 1) * 80 ms = (emit(t*) - end) * 80 ms for the first firing t*."""
    return eot_bench(_emit_transform(scores, ends, emit_fn), onsets, ends, chunk=1, **kw)


def _emit_transform(scores, ends, emit_fn) -> list[np.ndarray]:
    """The score re-indexing of eot_bench_emit (pre-end part kept, post-end part indexed by emission time)."""
    out = []
    floor = min(float(np.min(s)) for s in scores if len(s)) - 1.0  # 'nothing emitted yet' (finite: no NaN
    for s, e in zip(scores, ends):                                  # in eot_bench's threshold quantiles)
        s = np.asarray(s, np.float64)
        e = int(e)
        if len(s) <= e:
            out.append(s)
            continue
        t = np.arange(e, len(s))
        d = np.array([emit_fn(int(x)) for x in t]) - e  # >= 1, non-decreasing
        z = np.full(int(d[-1]), floor)
        np.maximum.at(z, d - 1, s[e:])
        out.append(np.concatenate([s[:e], np.maximum.accumulate(z)]))
    return out


HYBRID_GRID_K = (13, 18, 21, 24, 30)  # frames of primary silence
HYBRID_GRID_THETA = (0.9, 0.95, 0.98, 0.99)


def hybrid_rows(head, sil, on, en, chunk_head: int, chunk_sil: int, pauses, pure_head: dict, pure_timeout: dict) -> dict:
    """The hybrid decision rule "turn head p > θ OR primary silent >= k frames" (serve.py turn_policy "hybrid") on
    one input condition: conversation.eot_bench_or over eot_bench's own θ candidates x every k (plus 'never' for
    each, so the pure head and pure timeout are inside the sweep). Operating points at <= 5 % per-turn and per-pause
    FC (eot_bench's rule: lowest P50, then P90; ties: lower miss), the pure rows re-derived from the same sweep with
    eot_bench's tie rule (asserted equal to ``pure_head`` / ``pure_timeout``, the _pt rows of this run), and the
    (k, θ) grid of HYBRID_GRID_K x HYBRID_GRID_THETA at fixed points. k_frames = silence threshold + 1."""
    def fmt(pt: dict) -> dict:
        d = {"theta": pt["threshold_a"], "k_frames": None if pt["threshold_b"] is None else int(pt["threshold_b"]) + 1}
        return {**d, **{k: v for k, v in pt.items() if k not in ("threshold_a", "threshold_b")}}

    kw = dict(chunk_a=chunk_head, chunk_b=chunk_sil, pauses=pauses)
    res = {"rule": "fire at the first frame with head p > theta OR primary silent >= k_frames (either emission)",
           "chunk_head": chunk_head, "chunk_timeout": chunk_sil}
    for unit in ("turn", "pause"):
        b = eot_bench_or(head, sil, on, en, fc_unit=unit, **kw)
        res[f"at_5pct_{unit}_fc"] = fmt(b["at_max_fc"])
        if unit == "turn":
            res["at_p50_le_400ms"] = fmt(b["at_fixed_latency"])
            res["n_thresholds_theta_k"], res["n_pauses"] = b["n_thresholds"], b["n_pauses"]
        for tag, only, ref in (("pure_head", dict(thresholds_b=[]), pure_head),
                               ("pure_timeout", dict(thresholds_a=[]), pure_timeout)):
            r = eot_bench_or(head, sil, on, en, fc_unit=unit, break_ties_by_miss=False, **only, **kw)["at_max_fc"]
            if unit == "turn":  # the 1-D restriction reproduces eot_bench's row of this run exactly
                x = ref["at_5pct_fc"]
                th = r["threshold_a"] if tag == "pure_head" else r["threshold_b"]
                assert (th, r["fc_rate"], r["p50_ms"], r["p90_ms"], r["miss_rate"]) == \
                    (x["threshold"], x["fc_rate"], x["p50_ms"], x["p90_ms"], x["miss_rate"]), (tag, r, x)
            res.setdefault(tag, {})[f"at_5pct_{unit}_fc"] = fmt(r)
    g = eot_bench_or(head, sil, on, en, thresholds_a=HYBRID_GRID_THETA, thresholds_b=[k - 1 for k in HYBRID_GRID_K],
                     never=False, table=True, **kw)
    res["grid"] = [fmt(p) for p in g["table"]]
    return res


def _stream_emit(C: int, R: int, turn_chunk: int = 1):
    """Emission frame of frame t: the streaming diarizer finalizes frame t after its chunk (C frames)
    plus R right-context frames have arrived; the turn head additionally waits for its own encoder chunk."""
    return lambda t: max((t // turn_chunk + 1) * turn_chunk, (t // C + 1) * C + R)


@torch.no_grad()
def external_diar_probs(val, diar_ckpt: str, batch_size: int, cache: Path | None, budget_sec: float | None):
    """Per-example (T_d, S) speaker probabilities of an external Sortformer model (its own preprocessor /
    encoder / head): 'offline' = one head pass over the whole window; 'stream' = StreamingDiarizer, window
    mode, card low-latency config. Cached (resumable) in ``cache``; exits after ``budget_sec`` if unfinished."""
    import pickle
    from audioforge.streaming_diar import StreamingDiarizer
    key = [len(v["audio"]) for v in val]
    if cache is not None and cache.is_dir():  # per-example cache of scripts/research/make_sortformer_tracks.py
        # streaming tracks from the cache (per window: identical to recomputing them); the offline pass is cheap
        # and batch-composition dependent at ~1e-5, so it is recomputed below exactly as without a cache
        from audioforge.datasets import ext_tracks as xt
        man = xt.manifest(directory=cache)
        if man.get("diar_ckpt") and Path(man["diar_ckpt"]).name != Path(diar_ckpt).name:
            raise SystemExit(f"{cache} holds tracks of {man['diar_ckpt']}, not {diar_ckpt}")
        paths = [xt.track_path(cache, xt.example_key(v), "stream") for v in val]
        miss = [str(q) for q in paths if not q.exists()]
        if miss:
            raise SystemExit(f"{len(miss)} streaming tracks missing in {cache} (e.g. {miss[0]}): run "
                             f"scripts/research/make_sortformer_tracks.py --track-source stream for this split")
        print(f"  sortformer streaming tracks from {cache} ({len(val)})", flush=True)
        st = {"key": key, "ckpt": diar_ckpt, "offline": [], "stream": [np.load(q) for q in paths]}
        cache = None  # nothing to write back
    else:
        st = pickle.loads(cache.read_bytes()) if cache and cache.exists() else {}
    if st.get("key") != key:
        st = {"key": key, "ckpt": diar_ckpt, "offline": [], "stream": []}
    if len(st["offline"]) == len(val) and len(st["stream"]) == len(val):
        return st
    t0 = time.time()
    dm = load_model(diar_ckpt, "cpu")
    dname = _diar_name(dm)
    print(f"  loaded diarizer {diar_ckpt} ({time.time() - t0:.0f}s) head={dname} "
          f"n_mels={dm.preprocessor.n_mels}", flush=True)
    col = Collate(None)
    if len(st["offline"]) < len(val):
        st["offline"] = []
        for i in range(0, len(val), batch_size):
            b = col([{"audio": v["audio"]} for v in val[i: i + batch_size]])
            enc, elen = dm.encode(b["audio"], b["audio_len"])
            p = dm.heads[dname](enc, elen).sigmoid()
            st["offline"] += [p[j, : int(elen[j])].numpy() for j in range(len(elen))]
        print(f"  sortformer offline: {time.time() - t0:.0f}s", flush=True)
    for i in range(len(st["stream"]), len(val)):
        sd = StreamingDiarizer(dm, diar_head=dname, mode="window", enc_left_context=SORTFORMER_ENC_LEFT,
                               **SORTFORMER_LOW_LATENCY)
        sd.feed(np.asarray(val[i]["audio"], np.float32), final=True)
        st["stream"].append(sd.all_probs.numpy())
        if cache:
            cache.write_bytes(pickle.dumps(st))
        if cache and budget_sec and time.time() - t0 > budget_sec and i + 1 < len(val):
            raise SystemExit(f"streaming diarizer: {i + 1}/{len(val)} done in {time.time() - t0:.0f}s; "
                             f"cached in {cache}, re-run the same command to resume")
    print(f"  sortformer streaming: {time.time() - t0:.0f}s", flush=True)
    del dm
    return st


def enroll_column(p: np.ndarray, ref: np.ndarray, onset: int, end: int) -> int:
    """'Enrollment by who is talking': the diarizer column that best overlaps the oracle primary activity
    on frames [onset, end) only - never after the turn end. Hard overlap (p > 0.5 on primary frames),
    ties broken by the soft overlap."""
    T = min(len(p), len(ref), end)
    pp, rr = p[onset:T], ref[onset:T] > 0.5
    hard = ((pp > 0.5) & rr[:, None]).sum(0)
    soft = (pp * rr[:, None]).sum(0)
    return int(np.lexsort((-soft, -hard))[0])


def act_stats(acts, val, ends) -> dict:
    """Primary-activity frame errors of an activity track vs the oracle spk_act (common length):
    miss / FA per primary speech frame (DER convention), FA per non-primary frame, and the fraction of
    post-turn-end frames marked active (what keeps a silence timeout from firing)."""
    miss = fa = sp = nonsp = post_on = post = 0
    for a, v, e in zip(acts, val, ends):
        ref = np.asarray(v["spk_act"]) > 0.5
        T = min(len(a), len(ref))
        h, r = np.asarray(a[:T]) > 0.5, ref[:T]
        miss += int((r & ~h).sum()); fa += int((~r & h).sum())
        sp += int(r.sum()); nonsp += int((~r).sum())
        post_on += int(h[e:].sum()); post += max(0, T - e)
    return {"miss": round(miss / max(1, sp), 4), "fa_per_speech": round(fa / max(1, sp), 4),
            "fa_per_nonspeech": round(fa / max(1, nonsp), 4), "post_end_active": round(post_on / max(1, post), 4)}


@torch.no_grad()
def turn_scores_given_act(model, name: str, convs: list[dict], acts: list[np.ndarray], batch_size: int,
                          cols: list[np.ndarray] | None = None, prims: list[int] | None = None, aux: list | None = None):
    """heads.turn.turn_scores with the primary activity supplied from outside (one (T_i,) track per conv,
    already on the turn model's 80 ms frame grid): the same speaker-conditioned encode, greedy-decoded text
    state and head.decode; only the ``act`` tensor differs. (audioforge.heads.turn.turn_scores has no hook
    for an external track - patch intent: an ``act_override`` argument.)
    cols / prims (TurnHead v3, act_columns > 1): the diarizer's (T_i, S) columns and the primary's column index.
    aux (a list, heads with multi_horizon_aux): the per-bin user-activity probabilities (T_i, H) are appended to it."""
    from audioforge.heads.turn import decoded_text_state
    head, hc = model.heads[name], model.head_cfg[name]
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(convs), batch_size):
        cc = convs[i: i + batch_size]
        b = Collate(model.tokenizer)(cc)
        if dev.type != "cpu":
            b = to_device(b, dev)
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        act = torch.zeros(enc.shape[0], enc.shape[1], device=dev)
        for j, a in enumerate(acts[i: i + batch_size]):
            n = min(len(a), enc.shape[1])
            act[j, :n] = torch.as_tensor(np.asarray(a[:n], np.float32), device=dev)
        if hc.get("condition_on_speaker"):
            e = model.cond_head_input(name, b["audio"], b["audio_len"], act)
        else:
            e = model.head_input(name, enc, hidden)
        text = decoded_text_state(model, head, b["audio"], b["audio_len"], enc, hidden, elen, act) \
            if head.use_text else None
        if getattr(head, "v3_in", None) is None and getattr(head, "energy_in", None) is None:  # v2 heads: unchanged
            p = head.decode(e, elen, spk_act=act if head.concat else None, text=text)
        else:
            ct = pt = None
            if head.needs_cols:
                if cols is None or prims is None:
                    raise ValueError(f"turn head has act_columns={head.act_columns}: pass cols and prims")
                ct = torch.zeros(enc.shape[0], enc.shape[1], head.act_columns, device=dev)
                for j, c in enumerate(cols[i: i + batch_size]):
                    c = np.asarray(c, np.float32)
                    n, S = min(len(c), enc.shape[1]), min(c.shape[1], head.act_columns)
                    ct[j, :n, :S] = torch.as_tensor(c[:n, :S], device=dev)
                pt = torch.as_tensor([int(x) for x in prims[i: i + batch_size]], device=dev)
            en = head.energy_of(b["audio"], b["audio_len"], e.shape[1]) if hasattr(head, "energy_of") else None
            if aux is not None and getattr(head, "mh", None) is not None:  # + the multi-horizon bins (T_i, H)
                p, pm = head.decode_aux(e, elen, spk_act=act, text=text, cols=ct, prim=pt, energy=en)
                for j in range(len(cc)):
                    aux.append(pm[j, : min(int(elen[j]), len(cc[j]["spk_act"]))].float().cpu().numpy())
            else:
                p = head.decode(e, elen, spk_act=act, text=text, cols=ct, prim=pt, energy=en)
        for j in range(len(cc)):
            n = min(int(elen[j]), len(cc[j]["spk_act"]))
            out.append(p[j, :n].float().cpu().numpy())
    return out


@torch.no_grad()
def own_diar_acts(model, val, batch_size: int) -> list[np.ndarray]:
    """The 'diar' source's activity track (heads.turn.streaming_diar_act, arrival-rank column), per conv."""
    from audioforge.heads.turn import primary_column, streaming_diar_act
    diar, out = _diar_name(model), []
    for i in range(0, len(val), batch_size):
        b = Collate(model.tokenizer)(val[i: i + batch_size])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        act = streaming_diar_act(model, model.head_input(diar, enc, hidden), elen,
                                 primary_column(b["spk_targets"]), chunk_of(model))
        out += [act[j, : int(elen[j])].numpy() for j in range(len(elen))]
    return out


def _fit(a: np.ndarray, T: int) -> np.ndarray:
    """Crop / edge-pad a diarizer track to T frames (frame counts can differ by one)."""
    a = np.asarray(a, np.float32)
    return a[:T] if len(a) >= T else np.concatenate([a, np.repeat(a[-1:] if len(a) else [0.0], T - len(a))])


def duration_rule_scores(p: np.ndarray, col: int, onset: int = 0) -> np.ndarray:
    """Learned-free duration rule on a diarizer's (T, S) track: score = frames the primary column has been silent
    (silence_scores) on frames where no other column is active (p > 0.5), else 0. eot_bench's threshold sweep
    then sweeps k: fire when the primary has been silent >= k frames AND no other speaker is active."""
    p = np.asarray(p, np.float32)
    other = np.delete(p, col, 1).max(1) if p.shape[1] > 1 else np.zeros(len(p), np.float32)
    return silence_scores(p[:, col], onset) * (other <= 0.5)


def eval_turn_external(model, name, val, on, en, chunk, diar_ckpt, batch_size, cache, budget,
                       enroll_mode: str = "oracle", hybrid: bool = False) -> dict:
    st = external_diar_probs(val, diar_ckpt, batch_size, cache, budget)
    C, R = SORTFORMER_LOW_LATENCY["chunk_len"], SORTFORMER_LOW_LATENCY["chunk_right_context"]
    T_turn = [len(np.asarray(v["spk_act"])) for v in val]  # == turn-model encoder frames (checked below)
    res = {"diar_ckpt": diar_ckpt, "stream_config": dict(SORTFORMER_LOW_LATENCY, enc_left_context=SORTFORMER_ENC_LEFT),
           "stream_input_buffer_ms": (C + R) * 80.0,
           "column_choice": "argmax_s |{t in [onset, turn_end): p_s(t) > 0.5 and oracle primary(t)}| "
                            "(ties: soft overlap); chosen per example, per source, never looks past turn_end"}
    # frame alignment
    for src in ("offline", "stream"):
        d = np.array([len(p) for p in st[src]]) - np.array(T_turn)
        res[f"frames_{src}_minus_turn"] = {str(k): int((d == k).sum()) for k in np.unique(d)}
    ends = en
    acts, cols = {}, {}
    full_v2 = {}
    for src in ("offline", "stream"):
        if enroll_mode == "oracle":  # the v1 rule (oracle overlap on [onset, turn_end)): the default
            cs = [enroll_column(p, np.asarray(v["spk_act"]), o, e) for p, v, o, e in zip(st[src], val, on, en)]
            acts[src] = [_fit(p[:, c], T) for p, c, T in zip(st[src], cs, T_turn)]
        else:  # eot-bench v2 label-free enrollment (per-frame binding for causal_dominant: columns re-ordered)
            xs = [_v2_cols(v, p, enroll_mode) for v, p in zip(val, st[src])]
            cs = [x[2] for x in xs]
            acts[src] = [x[0] for x in xs]
            full_v2[src] = [x[1] for x in xs]
            orc = [enroll_column(p, np.asarray(v["spk_act"]), o, e) for p, v, o, e in zip(st[src], val, on, en)]
            res[f"enroll_agree_oracle_at_turn_end_{src}"] = round(float(np.mean(
                [x[3][e - 1] == c for x, e, c in zip(xs, en, orc)])), 4)
        cols[src] = cs
        res[f"column_hist_{src}"] = {str(k): int(np.sum(np.array(cs) == k)) for k in range(st[src][0].shape[1])}
    if enroll_mode != "oracle":
        res["enroll"] = enroll_mode
        res["column_choice"] = f"{enroll_mode} (label-free, eot-bench v2; research/EOT_BENCH_V2.md)"
    res["columns_agree_offline_stream"] = round(float(np.mean(np.array(cols["offline"]) == np.array(cols["stream"]))), 4)
    own = own_diar_acts(model, val, batch_size) if _diar_name(model) else None
    res["activity_vs_oracle"] = {"own_diar_stream": act_stats(own, val, ends) if own else None,
                                 "sortformer_offline": act_stats(acts["offline"], val, ends),
                                 "sortformer_stream": act_stats(acts["stream"], val, ends)}
    # sanity: the emission-time re-indexing reproduces eot_bench's chunk rule exactly
    orc = [silence_scores(np.asarray(v["spk_act"]), o) for v, o in zip(val, on)]
    assert _pt(eot_bench_emit(orc, on, en, lambda t: (t // chunk + 1) * chunk)) == _pt(eot_bench(orc, on, en, chunk=chunk))
    t0 = time.time()
    ncol = getattr(model.heads[name], "act_columns", 1)
    full = {src: [np.stack([_fit(p[:, s], T) for s in range(p.shape[1])], 1) for p, T in zip(st[src], T_turn)]
            for src in ("offline", "stream")}
    full.update(full_v2)
    xin = {src: ((full[src], cols[src]) if ncol > 1 else ()) for src in ("offline", "stream")}
    if ncol > 1:
        res["turn_head_input"] = f"act_columns={ncol}: all diarizer columns + one-hot of the chosen column"
    sc_off = turn_scores_given_act(model, name, val, acts["offline"], batch_size, *xin["offline"])
    res["turn_head_sortformer_offline"] = _pt(eot_bench(sc_off, on, en, chunk=chunk))
    sc_st = turn_scores_given_act(model, name, val, acts["stream"], batch_size, *xin["stream"])
    res["turn_head_sortformer_stream"] = _pt(eot_bench_emit(sc_st, on, en, _stream_emit(C, R, chunk)))
    print(f"  turn head x sortformer: {time.time() - t0:.0f}s", flush=True)
    # cascade: silence timeout on the diarizer's primary track (zero extra look-ahead beyond the diarizer's)
    res["timeout_sortformer_offline"] = _pt(eot_bench([silence_scores(a, o) for a, o in zip(acts["offline"], on)], on, en))
    res["timeout_sortformer_stream"] = _pt(eot_bench_emit([silence_scores(a, o) for a, o in zip(acts["stream"], on)],
                                                          on, en, _stream_emit(C, R)))
    if hybrid:  # head OR timeout on the streaming track, each on its own streaming emission rule
        res["hybrid_sortformer_stream"] = hybrid_rows(
            _emit_transform(sc_st, en, _stream_emit(C, R, chunk)),
            _emit_transform([silence_scores(a, o) for a, o in zip(acts["stream"], on)], en, _stream_emit(C, R)),
            on, en, 1, 1, [pause_runs(v["hes"]) for v in val], res["turn_head_sortformer_stream"],
            res["timeout_sortformer_stream"])
    if own:
        res["timeout_own_diar_stream"] = _pt(eot_bench([silence_scores(a, o) for a, o in zip(own, on)], on, en, chunk=chunk))
    # learned-free duration rule on the same tracks: primary silent >= k AND no other column active
    res["duration_rule_sortformer_offline"] = _pt(eot_bench(
        [duration_rule_scores(p, c, o) for p, c, o in zip(full["offline"], cols["offline"], on)], on, en))
    res["duration_rule_sortformer_stream"] = _pt(eot_bench_emit(
        [duration_rule_scores(p, c, o) for p, c, o in zip(full["stream"], cols["stream"], on)], on, en, _stream_emit(C, R)))
    return res


# --------------------------------------------------------------------------- eot-bench v2 (leak-free protocol)
# research/EOT_BENCH_V2.md. Everything below is opt-in (--bench v2 / --enroll); the default turn task is unchanged.
ENROLL_MODES = ("oracle", "causal_dominant", "first_active")
ENROLL_K, ENROLL_S = 25, 25  # causal_dominant: 2.0 s look-back, re-bind after > 2.0 s of silence (AMI max_hold)
V2_DEV_FOLDS = (("IS1008b", "ES2011b"), ("TS3004b", "IB4002"))  # leave-meetings-out halves for the fixed threshold
V2_DIAR_CKPT = "runs/nemo_sortformer_v2.afm"
V2_FLOOR_HORIZON = 13  # frames (1.04 s): floor-open = no other speaker active within this after the end


def _dominant(hard: np.ndarray, soft: np.ndarray) -> int:
    """Column with the most active frames (ties: the larger soft sum)."""
    return int(np.lexsort((-soft, -hard))[0])


def enroll_causal_dominant(p: np.ndarray, k: int = ENROLL_K, s: int = ENROLL_S, thr: float = 0.5) -> np.ndarray:
    """Causal, label-free enrollment: per frame t the bound primary column c(t) (-1 = nothing heard yet).

    Binds, at the first frame on which any column is active (p > thr), the column with the most active frames in the
    last ``k`` frames [t - k + 1, t]; afterwards re-binds (to that running dominant column) only when the bound column
    has been silent for more than ``s`` consecutive frames and another column dominates the look-back. Reads only
    the diarizer's own probabilities up to frame t (never labels, never the future)."""
    p = np.asarray(p, np.float64)
    T, S = p.shape
    on = p > thr
    ch = np.concatenate([np.zeros((1, S)), np.cumsum(on, 0)])
    cs = np.concatenate([np.zeros((1, S)), np.cumsum(p, 0)])
    out = np.full(T, -1, np.int64)
    c, silent = -1, 0
    for t in range(T):
        lo = max(0, t - k + 1)
        hard, soft = ch[t + 1] - ch[lo], cs[t + 1] - cs[lo]
        if c < 0:
            if on[t].any():
                c, silent = _dominant(hard, soft), 0
        else:
            silent = 0 if on[t, c] else silent + 1
            if silent > s:
                new = _dominant(hard, soft)
                if new != c and hard[new] > 0:
                    c, silent = new, 0
        out[t] = c
    return out


def enroll_first_active(p: np.ndarray, thr: float = 0.5) -> int:
    """Causal, label-free enrollment: the first column active after the window start (ties at that frame: the
    highest probability); column 0 if nothing is ever active."""
    p = np.asarray(p, np.float64)
    act = np.nonzero((p > thr).any(1))[0]
    return int(np.argmax(p[act[0]])) if len(act) else 0


def enroll(p: np.ndarray, mode: str, ref=None, onset: int | None = None, end: int | None = None,
           k: int = ENROLL_K, s: int = ENROLL_S) -> np.ndarray:
    """Per-frame primary column (T,) of a (T, S) diarizer track under an enrollment rule: 'oracle' (enroll_column:
    overlap with the oracle primary on [onset, end) - the enrollment LEAK of the v1 protocol), 'first_active' and
    'causal_dominant' (label-free; ``ref`` / ``onset`` / ``end`` are ignored). Static rules repeat one column."""
    assert mode in ENROLL_MODES, mode
    T = len(p)
    if mode == "oracle":
        return np.full(T, enroll_column(p, np.asarray(ref), int(onset), int(end)), np.int64)
    if mode == "first_active":
        return np.full(T, enroll_first_active(p), np.int64)
    return enroll_causal_dominant(p, k, s)


def bound_track(p: np.ndarray, col_t: np.ndarray):
    """(act (T,), cols (T, S)) for a per-frame binding: act(t) = p[t, c(t)] (0 while unbound, c = -1); cols = the
    columns re-ordered per frame so the bound column is column 0 (others keep their order) - fed to a TurnHead v3
    with prim = 0, this is exactly a per-frame primary index (its one-hot / counters follow the binding)."""
    p = np.asarray(p, np.float32)
    T, S = p.shape
    c = np.asarray(col_t)
    act = np.where(c >= 0, p[np.arange(T), np.clip(c, 0, S - 1)], 0.0).astype(np.float32)
    cc = np.clip(c, 0, S - 1)
    order = np.array([[j] + [i for i in range(S) if i != j] for j in range(S)])[cc]  # (T, S)
    return act, np.take_along_axis(p, order, 1)


def arm_frame(act) -> int:
    """First active frame of an activity track (len if never): a deployable timeout arms here, not at the label onset."""
    nz = np.nonzero(np.asarray(act) > 0.5)[0]
    return int(nz[0]) if len(nz) else len(act)


def turn_windows(ds, trail_sec: float = 2.0, starts=None, window_sec: float = 20.0, lead_sec: float = 4.0,
                 min_trail: float = 1.0, max_spks: int = 4):
    """AMI.turn_examples re-cut from the same meeting audio + word labels, optionally with each window's START fixed
    (``starts[k]`` for the k-th example; used to extend the trail of the default windows without moving their start,
    so the pre-end part is unchanged). Returns (examples, meta); meta[k] = {end_reason: 'resume' (the primary's next
    turn starts: natural end) | 'trail' | 'meeting_end', post_avail: frames after the turn end, stop_sec}."""
    from audioforge.data import ToneLanguage
    from audioforge.datasets.ami import activity, frames, spk_matrix
    out, meta = [], []
    for m in ds.meetings:
        turns, dur = ds.turns[m], ds.duration(m)
        prev, nxt, by = {}, {}, {}
        for t in turns:
            by.setdefault(t["speaker"], []).append(t)
        for ts in by.values():
            for j, t in enumerate(ts):
                prev[id(t)] = ts[j - 1]["end"] if j else 0.0
                nxt[id(t)] = ts[j + 1]["start"] if j + 1 < len(ts) else dur
        for t in turns:
            if t["bc"]:
                continue
            b = min(dur, t["end"] + trail_sec, nxt[id(t)])
            if b - t["end"] < min_trail:
                continue
            a = max(0.0, prev[id(t)], t["start"] - lead_sec, b - window_sec) if starts is None else float(starts[len(out)])
            x = ds._clip(m, a, b)
            T = ToneLanguage.n_frames(len(x))
            prim = frames(activity([w for w in t["words"] if w[1] > a], ds.cfg["act_bridge"]), T, a)
            if not prim.any():
                continue
            y, order, dropped = spk_matrix(ds.acts[m], T, a, b, max_spks, first=t["speaker"])
            y[:, 0] = prim
            nz = np.nonzero(prim)[0]
            onset, end = int(nz[0]), int(nz[-1]) + 1
            eot = np.zeros(T, np.float32)
            eot[end:] = 1
            hes = frames([h for h in t["hes"] if h[1] > a], T, a) * (1 - prim)
            out.append(dict(audio=x, spk_targets=y, spk_act=prim, primary_act=prim, eot=eot, hes=hes,
                            turn_end_frame=end, onset_frame=onset,
                            text=" ".join(w[2] for w in t["words"] if w[1] > a),
                            speaker=ds.gid(t["speaker"]), meeting=m, start=float(a),
                            onset_clipped=bool(t["start"] < a), n_hesitations=len(t["hes"]), dropped_speakers=dropped))
            reason = "resume" if b == nxt[id(t)] else ("trail" if b == t["end"] + trail_sec else "meeting_end")
            meta.append(dict(end_reason=reason, post_avail=T - end, stop_sec=float(b)))
    if starts is not None:
        assert len(out) == len(starts), (len(out), len(starts))
    return out, meta


def v2_data(trail_sec: float = 6.0):
    """(base, ext, meta, ds, meta_base): the default dev turn windows (all 974, trail 2 s; keys of the cached Sortformer tracks) and
    the same windows with the trail extended to ``trail_sec`` (same start, stop at the primary's next turn)."""
    from audioforge.datasets.ami import AMI, subset
    from audioforge.datasets import ext_tracks as xt
    base = ami_dev("turn", 10 ** 6)
    ds = AMI(subset({"dev": 4})["dev"], verbose=False)
    chk, meta_base = turn_windows(ds, 2.0)  # the re-cut reproduces the library's windows exactly
    assert [xt.example_key(v) for v in chk] == [xt.example_key(v) for v in base]
    assert all(np.array_equal(u["spk_targets"], v["spk_targets"]) and np.array_equal(u["hes"], v["hes"])
               for u, v in zip(chk, base))
    ext, meta = turn_windows(ds, trail_sec, starts=[v["start"] for v in base])
    for u, v in zip(ext, base):
        assert u["onset_frame"] == v["onset_frame"] and u["turn_end_frame"] == v["turn_end_frame"]
    return base, ext, meta, ds, meta_base


V2_DIAR_CONFIGS = ("low_latency", "low_latency_032")  # audioforge.streaming_diar.SORTFORMER_PRESETS names


def v2_diar_cfg(name: str = "low_latency") -> dict:
    """StreamingDiarizer config of a --v2-diar-config name (low_latency = SORTFORMER_LOW_LATENCY, the default)."""
    from audioforge.streaming_diar import SORTFORMER_PRESETS
    assert name in V2_DIAR_CONFIGS, name
    return SORTFORMER_LOW_LATENCY if name == "low_latency" else dict(SORTFORMER_PRESETS[name])


def v2_track_path(work: Path, ex: dict, tracks_dir=None) -> Path:
    """<work>/tracks/<key>.stream_rc.npy, or <tracks_dir>/<key>.stream_rc.npy for a separate track set
    (--v2-tracks-dir, e.g. another diarizer config): same keys, other directory."""
    from audioforge.datasets import ext_tracks as xt
    return (Path(tracks_dir) if tracks_dir else Path(work) / "tracks") / f"{xt.example_key(ex)}.stream_rc.npy"


def v2_tracks(ext, ds, work: Path, budget: float, diar_ckpt: str = V2_DIAR_CKPT, order=None,
              cfg_name: str = "low_latency", device: str = "cpu", tracks_dir=None) -> dict:
    """Streaming Sortformer tracks (card low-latency config, as the cache; or the preset ``cfg_name``) of the extended
    windows, computed on the window audio plus (C + R + 1) frames of right padding from the meeting, cropped to the
    window's frames: every scored frame gets its full streaming right context (no end-of-file flush inside the scoring
    window). Resumable; ``order`` = example indices in priority order (default: all, as listed). ``tracks_dir``: write
    there instead of <work>/tracks; ``device``: where the diarizer runs (tracks saved as CPU float32)."""
    from audioforge.streaming_diar import StreamingDiarizer
    from audioforge.datasets.ami import SR
    cfg = v2_diar_cfg(cfg_name)
    order = list(range(len(ext))) if order is None else list(order)
    todo = [i for i in order if not v2_track_path(work, ext[i], tracks_dir).exists()]
    v2_track_path(work, ext[0], tracks_dir).parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        return {"done": len(ext), "todo": 0}
    C, R = cfg["chunk_len"], cfg["chunk_right_context"]
    t0 = time.time()
    dm = load_model(diar_ckpt, device)
    dname = _diar_name(dm)
    n = 0
    with torch.no_grad():
        for i in todo:
            v = ext[i]
            m, a = v["meeting"], float(v["start"])
            b = a + len(v["audio"]) / SR
            x = ds._clip(m, a, min(ds.duration(m), b + (C + R + 1) * 0.08))
            sd = StreamingDiarizer(dm, diar_head=dname, mode="window", enc_left_context=SORTFORMER_ENC_LEFT, **cfg)
            sd.feed(np.asarray(x, np.float32), final=True)
            p = sd.all_probs.float().cpu().numpy()[: len(v["spk_act"])]
            q = v2_track_path(work, v, tracks_dir)
            np.save(q.with_suffix(".tmp.npy"), p)
            (q.with_suffix(".tmp.npy")).replace(q)
            n += 1
            if time.time() - t0 > budget:
                break
    done = sum(v2_track_path(work, v, tracks_dir).exists() for v in ext)
    print(f"  v2 tracks: {n} computed in {time.time() - t0:.0f}s, {len(ext) - done} left", flush=True)
    return {"done": done, "todo": len(ext) - done}


def _v2_cols(ex, p, mode, k=ENROLL_K, s=ENROLL_S):
    """-> (act (T,), cols (T, S), prim, col_t) for the turn head under an enrollment rule, on the example's grid."""
    T = len(np.asarray(ex["spk_act"]))
    p = np.stack([_fit(p[:, j], T) for j in range(p.shape[1])], 1)
    ct = (v2_stored_binding(ex, mode, T) if mode in V2_STORED_MODES
          else enroll(p, "oracle" if mode == "oracle_reordered" else mode, ex["spk_act"], ex["onset_frame"],
                      ex["turn_end_frame"], k, s))
    if mode in ("causal_dominant", "oracle_reordered") or mode in V2_STORED_MODES:
        act, cols = bound_track(p, ct)
        return act, cols, 0, ct
    return p[:, int(ct[0])].copy(), p, int(ct[0]), ct


# --------------------------------------------------------------------------- voice enrollment (EOT_BENCH_V2.md §8)
# audioforge/enrollment.py: follow the primary by voice. Stages: embed (the speaker head's per-frame input of the
# extended windows, once), bind (per-frame voice bindings of the extended and default windows, stored), then scores /
# report as for the other bindings (the stored binding is read by _v2_cols).
from audioforge.enrollment import VOICE_MODES as V2_VOICE_MODES  # noqa: E402
from audioforge.enrollment import ALL_MODES as V2_ALL_VOICE_MODES  # noqa: E402

# §9: the same rules followed with the TitaNet-L backend ('<rule>_titanet', audio embeddings on a 5-frame grid), the
# two §9 identity rules (after_prev_end, voice_explicit) and the causal_dominant-followed control of after_prev_end.
# '_e<N>': enrollment on the first N active frames instead of 19 (1.5 s): 40 = 3.2 s, 60 = 4.8 s (OUTSIDE.md §4:
# target-speaker enrollment saturates at 3-5 s); the following is unchanged.
V2_ENROLL_LENS = (40, 60)
V2_TITANET_MODES = tuple(f"{m}_titanet" for m in V2_ALL_VOICE_MODES) + tuple(
    f"{m}_titanet_e{n}" for m in ("voice_explicit", "after_prev_end") for n in V2_ENROLL_LENS)
V2_CLEAN_MODES = ("after_prev_end_titanet_clean", "voice_explicit_titanet_clean")  # §9b clean-span masks
V2_CONTROL_MODES = ("after_prev_end_causal",) + V2_CLEAN_MODES
V2_STORED_MODES = V2_VOICE_MODES + V2_TITANET_MODES + V2_CONTROL_MODES  # per-frame bindings read from bind_<mode>/
V2_BIND_WORK: list = [None]  # set by eval_turn_v2: where bind_<mode>/ lives
# 'oracle_reordered': the oracle column, but fed to the head in the per-frame-binding representation (bound_track:
# bound column first, prim = 0) that causal_dominant and the voice bindings use - separates the head's sensitivity to
# the column order from the binding itself. Head rows only (its timeout is the oracle one).
V2_REPR_MODES = ("oracle_reordered",)


def v2_feat_path(work: Path, ex: dict) -> Path:
    from audioforge.datasets import ext_tracks as xt
    return Path(work) / "spkfeat" / f"{xt.example_key(ex)}.npy"


def v2_bind_path(work: Path, mode: str, ex: dict) -> Path:
    """<work>/bind_<mode>/<key>.npy: the per-frame voice binding of one window (keys differ between the default and
    the extended window of a turn: they include the sample count)."""
    from audioforge.datasets import ext_tracks as xt
    return Path(work) / f"bind_{mode}" / f"{xt.example_key(ex)}.npy"


def v2_stored_binding(ex: dict, mode: str, T: int) -> np.ndarray:
    assert V2_BIND_WORK[0] is not None, "voice bindings need --v2-work with a bind stage run"
    c = np.load(v2_bind_path(V2_BIND_WORK[0], mode, ex))
    assert len(c) == T, (len(c), T)
    return c


@torch.no_grad()
def v2_embed(model, ext, work: Path, budget: float, batch_size: int) -> dict:
    """Stage embed: the speaker head's per-frame input (enrollment.speaker_frames, float16) of every extended window.
    The default windows share the start and the encoder is causal, so their features are these, cropped."""
    from audioforge.enrollment import speaker_frames
    t0 = time.time()
    todo = [i for i in range(len(ext)) if not v2_feat_path(work, ext[i]).exists()]
    v2_feat_path(work, ext[0]).parent.mkdir(parents=True, exist_ok=True)
    todo.sort(key=lambda i: len(ext[i]["audio"]))
    n = 0
    for j in range(0, len(todo), batch_size):
        if time.time() - t0 > budget:
            break
        idx = todo[j: j + batch_size]
        for i, f in zip(idx, speaker_frames(model, [ext[i]["audio"] for i in idx], batch_size)):
            q = v2_feat_path(work, ext[i])
            np.save(q.with_suffix(".tmp.npy"), f.astype(np.float16))
            q.with_suffix(".tmp.npy").replace(q)
        n += len(idx)
    left = len(todo) - n
    print(f"  v2 embed: {n} computed in {time.time() - t0:.0f}s, {left} left", flush=True)
    return {"done": len(ext) - left, "todo": left}


def v2_voice_bind(embed, ex, p, feats, mode, tn=None, **kw):
    """(col (T,), info) of one window under a voice binding on the example's grid (tracks fitted as in _v2_cols).

    ``mode``: a §8 rule on the own-head features ``feats`` (``embed`` = ColumnEmbedder), or a '<rule>_titanet' rule
    with ``tn`` = TitaNetEmbedder (enrollment embedding from the window's audio; ``kw['emb_ok']`` = the stored
    TitaNet look-back embeddings), or 'after_prev_end_causal' (the after_prev_end choice followed by causal_dominant).
    after_prev_end's agent end comes from the labels (enrollment.agent_end_frame), the column choice does not."""
    from audioforge.enrollment import (after_prev_end_choice, agent_end_frame, causal_dominant_from, enroll_voice,
                                       TITANET_DEFAULTS, VOICE_DEFAULTS)
    T = len(np.asarray(ex["spk_act"]))
    p = np.stack([_fit(p[:, j], T) for j in range(p.shape[1])], 1)
    if "_e" in mode and mode.rsplit("_e", 1)[1].isdigit():  # '<rule>_titanet_e40': longer enrollment
        mode, n_enr = mode.rsplit("_e", 1)
        kw = dict(kw, enroll_frames=int(n_enr))
    clean = None
    if mode.endswith("_clean"):  # §9b: kw['clean'] = admitted mask (clean_mask), emb_ok from recent_embeddings_spans
        mode = mode.removesuffix("_clean")
        clean = kw.pop("clean")
    base = mode.removesuffix("_titanet").removesuffix("_causal")
    ci = enroll_causal_dominant(p) if base in ("voice_dominant", "after_prev_end", "voice_explicit") else None
    ae = agent_end_frame(ex["spk_targets"], int(ex["onset_frame"])) if base == "after_prev_end" else None
    if mode.endswith("_causal"):
        assert base == "after_prev_end", mode
        info = {"c0": -1, "enrolled_at": None, "n_switches": 0}
        if ae is None:
            return ci, dict(info, fallback="no_agent_turn")
        c0, tc = after_prev_end_choice(p > VOICE_DEFAULTS["thr"], ae, TITANET_DEFAULTS["min_run"])
        if c0 is None:
            return ci, dict(info, fallback="no_column")
        return causal_dominant_from(p, force=(c0, tc)), dict(info, c0=c0, chosen_at=tc, agent_end=ae)
    oc = enroll_column(p, np.asarray(ex["spk_act"]), ex["onset_frame"], ex["turn_end_frame"]) \
        if base == "voice_oracle" else None
    if mode.endswith("_titanet"):
        assert tn is not None and kw.get("emb_ok") is not None, "TitaNet modes need tn and the stored emb_ok"
        audio = np.asarray(ex["audio"], np.float32)
        if clean is not None:
            from audioforge.enrollment import consistent_mean, CLEAN_DEFAULTS
            return enroll_voice(p, None, None, base, ref=ex["spk_act"], oracle_col=oc, causal_init=ci, agent_end=ae,
                                clean=clean, enroll_embed=lambda spans: consistent_mean(
                                    tn.frames(audio, spans), CLEAN_DEFAULTS["span_min_cos"])[0], **kw)
        return enroll_voice(p, None, None, base, ref=ex["spk_act"], oracle_col=oc, causal_init=ci, agent_end=ae,
                            enroll_embed=lambda idx: tn.frames(audio, [idx])[0], **kw)
    f = np.asarray(feats, np.float32)
    f = f[:T] if len(f) >= T else np.concatenate([f, np.zeros((T - len(f), f.shape[1]), np.float32)])
    return enroll_voice(p, f, embed, base, ref=ex["spk_act"], oracle_col=oc, causal_init=ci, agent_end=ae, **kw)


def v2_titanet_path(work: Path, ex: dict, clean: bool = False) -> Path:
    """<work>/spkemb_titanet[_clean]/<key>.npz: TitaNet look-back embeddings of an extended window on the 5-frame grid."""
    from audioforge.datasets import ext_tracks as xt
    return Path(work) / ("spkemb_titanet_clean" if clean else "spkemb_titanet") / f"{xt.example_key(ex)}.npz"


def v2_load_titanet(work: Path, ex: dict, T: int, clean: bool = False):
    """The stored grid embeddings expanded to per-frame (emb (T, S, E), ok (T, S)) as recent_embeddings_audio
    returns them; ``T`` <= the stored window's length (a default window = the extended window's prefix)."""
    z = np.load(v2_titanet_path(work, ex, clean))
    stride = int(z["stride"])
    emb, ok = z["emb"].astype(np.float32), z["ok"]
    rep = np.minimum(stride, T - np.arange(0, T, stride))
    g = len(rep)
    return np.repeat(emb[:g], rep, 0), np.repeat(ok[:g], rep, 0)


def v2_vad_path(work: Path, ex: dict) -> Path:
    from audioforge.datasets import ext_tracks as xt
    return Path(work) / "vad" / f"{xt.example_key(ex)}.npy"


@torch.no_grad()
def v2_vad(model, ext, work: Path, batch_size: int = 8) -> int:
    """The model's own VAD probability per frame of every extended window (<work>/vad/<key>.npy; causal encode)."""
    todo = [i for i in range(len(ext)) if not v2_vad_path(work, ext[i]).exists()]
    if not todo:
        return 0
    v2_vad_path(work, ext[0]).parent.mkdir(parents=True, exist_ok=True)
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "frame" and v.get("key") == "vad")
    todo.sort(key=lambda i: len(ext[i]["audio"]))
    for j in range(0, len(todo), batch_size):
        idx = todo[j: j + batch_size]
        x, lens = model._pad([ext[i]["audio"] for i in idx])
        enc, elen, hidden = model.encode(x, lens, return_hidden=True)
        pr = torch.sigmoid(model.heads[name](model.head_input(name, enc, hidden))).float().cpu().numpy()
        for k, i in enumerate(idx):
            np.save(v2_vad_path(work, ext[i]), pr[k, : int(elen[k])].astype(np.float16))
    return len(todo)


def v2_clean_mask(work: Path, ex_ext: dict, p_fit: np.ndarray):
    from audioforge.enrollment import CLEAN_DEFAULTS, VOICE_DEFAULTS, clean_mask
    vad = np.load(v2_vad_path(work, ex_ext)).astype(np.float32)
    return clean_mask(p_fit, vad, VOICE_DEFAULTS["thr"], CLEAN_DEFAULTS["vad_thr"])


def v2_embed_titanet_clean(tn, ext, work: Path, budget: float, tracks_dir=None) -> dict:
    """Stage embed --v2-embedder titanet_clean (§9b): recent_embeddings_spans on the clean mask of every extended
    window -> <work>/spkemb_titanet_clean/<key>.npz (same layout as spkemb_titanet)."""
    from audioforge.enrollment import CLEAN_DEFAULTS, TITANET_DEFAULTS, VOICE_DEFAULTS, recent_embeddings_spans
    t0 = time.time()
    todo = [i for i in range(len(ext)) if not v2_titanet_path(work, ext[i], clean=True).exists()
            and v2_track_path(work, ext[i], tracks_dir).exists()]
    v2_titanet_path(work, ext[0], clean=True).parent.mkdir(parents=True, exist_ok=True)
    n, stride = 0, TITANET_DEFAULTS["stride"]
    for i in todo:
        if time.time() - t0 > budget:
            break
        ex = ext[i]
        T = len(np.asarray(ex["spk_act"]))
        p = np.load(v2_track_path(work, ex, tracks_dir))
        p = np.stack([_fit(p[:, j], T) for j in range(p.shape[1])], 1)
        adm = v2_clean_mask(work, ex, p)
        emb, ok = recent_embeddings_spans(ex["audio"], adm, tn, VOICE_DEFAULTS["win"], VOICE_DEFAULTS["min_frames"],
                                          stride, CLEAN_DEFAULTS["min_span"], CLEAN_DEFAULTS["span_min_cos"])
        q = v2_titanet_path(work, ex, clean=True)
        np.savez(q.with_suffix(".tmp.npz"), emb=emb[::stride].astype(np.float16), ok=ok[::stride], stride=stride)
        q.with_suffix(".tmp.npz").replace(q)
        n += 1
    left = len(todo) - n
    print(f"  v2 embed titanet_clean: {n} computed in {time.time() - t0:.0f}s, {left} left", flush=True)
    return {"done": len(ext) - left, "todo": left}


def v2_embed_titanet(tn, ext, work: Path, budget: float, tracks_dir=None) -> dict:
    """Stage embed --v2-embedder titanet: recent_embeddings_audio of every extended window on its streaming track
    (stride TITANET_DEFAULTS['stride']); grid rows stored float16. The default windows reuse them cropped (same
    audio prefix; the cached 2 s tracks differ from the cropped extended ones by <= 1.8e-4)."""
    from audioforge.enrollment import TITANET_DEFAULTS, VOICE_DEFAULTS, recent_embeddings_audio
    t0 = time.time()
    todo = [i for i in range(len(ext)) if not v2_titanet_path(work, ext[i]).exists()
            and v2_track_path(work, ext[i], tracks_dir).exists()]
    v2_titanet_path(work, ext[0]).parent.mkdir(parents=True, exist_ok=True)
    n, stride = 0, TITANET_DEFAULTS["stride"]
    for i in todo:
        if time.time() - t0 > budget:
            break
        ex = ext[i]
        T = len(np.asarray(ex["spk_act"]))
        p = np.load(v2_track_path(work, ex, tracks_dir))
        p = np.stack([_fit(p[:, j], T) for j in range(p.shape[1])], 1)
        emb, ok = recent_embeddings_audio(ex["audio"], p, tn, VOICE_DEFAULTS["win"], VOICE_DEFAULTS["min_frames"],
                                          VOICE_DEFAULTS["thr"], stride)
        q = v2_titanet_path(work, ex)
        np.savez(q.with_suffix(".tmp.npz"), emb=emb[::stride].astype(np.float16), ok=ok[::stride], stride=stride)
        q.with_suffix(".tmp.npz").replace(q)
        n += 1
    left = len(todo) - n
    print(f"  v2 embed titanet: {n} computed in {time.time() - t0:.0f}s, {left} left", flush=True)
    return {"done": len(ext) - left, "todo": left}


def v2_bind(model, base, ext, work: Path, modes, budget: float, tracks_dir=None, tn=None) -> dict:
    """Stage bind: per-frame voice bindings (enrollment.enroll_voice) of the extended windows (their streaming tracks)
    and the default windows (the dev cache's tracks, v2_base_track; features = the extended window's, cropped).
    Resumable, one .npy per window and mode. Only the speaker head of ``model`` is used; with ``tn`` (TitaNetEmbedder)
    the modes are the '<rule>_titanet' / control ones and read the embed-titanet stage's files instead (``model``
    unused). Also writes bind_<mode>/_info.json (c0, enrolled_at, fallback per window key) for the decomposition."""
    from audioforge.enrollment import ColumnEmbedder, recent_embeddings, VOICE_DEFAULTS, speaker_head_name
    from audioforge.datasets import ext_tracks as xt
    embed = ColumnEmbedder(model.heads[speaker_head_name(model)]) if tn is None else None
    infos = {m: {} for m in modes}
    for m in modes:
        q = Path(work) / f"bind_{m}" / "_info.json"
        if q.exists():
            infos[m] = json.loads(q.read_text())
    t0, left = time.time(), 0
    for m in modes:
        for w in (base, ext):
            v2_bind_path(work, m, w[0]).parent.mkdir(parents=True, exist_ok=True)
    for i in range(len(ext)):
        jobs = [(w, m) for w in (base, ext) for m in modes if not v2_bind_path(work, m, w[i]).exists()]
        if not jobs:
            continue
        if time.time() - t0 > budget:
            left += 1
            continue
        feats = np.load(v2_feat_path(work, ext[i])).astype(np.float32) if tn is None else None
        for w in (base, ext):
            ms = [m for ww, m in jobs if ww is w]
            if not ms:
                continue
            ex = w[i]
            p = (v2_base_track(ex) if tracks_dir is None else v2_base_track(ex, ext[i], tracks_dir)) if w is base \
                else np.load(v2_track_path(work, ex, tracks_dir))
            T = len(np.asarray(ex["spk_act"]))
            pf = np.stack([_fit(p[:, j], T) for j in range(p.shape[1])], 1)
            if tn is None:
                ff = feats[:T] if len(feats) >= T else np.concatenate(
                    [feats, np.zeros((T - len(feats), feats.shape[1]), np.float32)])
                eo = recent_embeddings(ff, pf, embed, VOICE_DEFAULTS["win"], VOICE_DEFAULTS["min_frames"],
                                       VOICE_DEFAULTS["thr"])
            else:
                eo = v2_load_titanet(work, ext[i], T)
                eoc = v2_load_titanet(work, ext[i], T, clean=True) if any(m.endswith("_clean") for m in ms) else None
            for m in ms:
                if m.endswith("_clean"):
                    col, info = v2_voice_bind(embed, ex, p, feats, m, tn=tn, emb_ok=eoc, clean=v2_clean_mask(work, ext[i], pf))
                else:
                    col, info = v2_voice_bind(embed, ex, p, feats, m, tn=tn, emb_ok=eo)
                np.save(v2_bind_path(work, m, ex), col)
                infos[m][xt.example_key(ex)] = {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                                                for k, v in info.items()}
    for m in modes:
        (Path(work) / f"bind_{m}" / "_info.json").write_text(json.dumps(infos[m]))
    print(f"  v2 bind: {left} windows left ({time.time() - t0:.0f}s)", flush=True)
    return {"left": left}


V2_BINDINGS = ("oracle@2s", "causal_dominant@2s", "first_active@2s", "oracle", "causal_dominant", "first_active")


def v2_score_path(work: Path, binding: str, ex: dict, tag: str = "") -> Path:
    """Stored head scores: <work>/scores_<binding>[__<tag>]/<key>.npy. ``tag`` (--v2-tag, e.g. the checkpoint's name)
    keeps the scores of different turn checkpoints apart in one work dir (the tracks are shared); '' = the untagged
    layout of the first (v3) run."""
    from audioforge.datasets import ext_tracks as xt
    sub = f"scores_{binding.replace('@', '_')}" + (f"__{tag}" if tag else "")
    return Path(work) / sub / f"{xt.example_key(ex)}.npy"


def v2_base_track(ex, ext_ex=None, tracks_dir=None) -> np.ndarray:
    """The streaming track of a default (2 s) window: the dev cache's flush-fixed <key>.stream.npy; with a separate
    track set (``tracks_dir``) the same-start extended window's track (``ext_ex``) cropped to the window's frames,
    which is the same array (flush fix: every window frame has its full right context; checked on the 1.04 s set:
    max |crop - cache| = 1.8e-4, MPS vs CPU)."""
    from audioforge.datasets import ext_tracks as xt
    if tracks_dir is not None:
        return np.load(v2_track_path(None, ext_ex, tracks_dir))[: len(np.asarray(ex["spk_act"]))]
    return np.load(xt.track_path(xt.cache_dir(split="dev"), xt.example_key(ex), "stream"))


@torch.no_grad()
def v2_scores(model, base, ext, work: Path, bindings, budget: float, batch_size: int, tag: str = "",
              tracks_dir=None) -> dict:
    """Turn-head scores per example and binding (resumable; per-example .npy). '<rule>@2s' = the default windows
    with their cached streaming tracks (all turns); '<rule>' = the extended windows, for the turns whose extended
    streaming track exists (v2_tracks). ``tracks_dir``: a separate extended-window track set (then the default
    windows use its tracks cropped, v2_base_track)."""
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    t0, left = time.time(), {}
    for bnd in bindings:
        two = bnd.endswith("@2s")
        convs = base if two else ext
        mode = bnd.split("@")[0]
        ok = [i for i, v in enumerate(convs) if (two and tracks_dir is None) or v2_track_path(work, ext[i], tracks_dir).exists()]
        todo = [i for i in ok if not v2_score_path(work, bnd, convs[i], tag).exists()]
        v2_score_path(work, bnd, convs[0], tag).parent.mkdir(parents=True, exist_ok=True)
        todo.sort(key=lambda i: len(convs[i]["audio"]))
        done = 0
        for j in range(0, len(todo), batch_size):
            if time.time() - t0 > budget:
                break
            idx = todo[j: j + batch_size]
            cc = [convs[i] for i in idx]
            tr = [(v2_base_track(v) if tracks_dir is None else v2_base_track(v, ext[i], tracks_dir)) if two
                  else np.load(v2_track_path(work, v, tracks_dir)) for i, v in zip(idx, cc)]
            xs = [_v2_cols(v, p, mode) for v, p in zip(cc, tr)]
            sc = turn_scores_given_act(model, name, cc, [x[0] for x in xs], batch_size,
                                       [x[1] for x in xs], [x[2] for x in xs])
            for v, sv in zip(cc, sc):
                np.save(v2_score_path(work, bnd, v, tag), sv)
            done += len(idx)
        left[bnd] = len(todo) - done
        print(f"  v2 scores {bnd}: {done} done, {left[bnd]} left of {len(ok)} ({time.time() - t0:.0f}s)", flush=True)
    return left


def _select(oc: dict, rows: np.ndarray, max_fc: float, unit: str, tie_miss: bool = False) -> int:
    """eot_bench's operating-point rule (lowest P50, then P90, s.t. FC <= max_fc) on a subset of conversations;
    the most conservative threshold if none qualifies. Remaining ties: the first candidate (on one ascending
    threshold axis = the lowest threshold), or with ``tie_miss`` the lower miss rate (eot_bench_or's rule; needed on
    a joint (θ, k) grid, whose column order has no meaning, whenever P50 = P90 = inf)."""
    from audioforge.conversation import _q
    fc, lat = oc["fc"][rows], oc["lat"][rows]
    rate = fc.mean(0) if unit == "turn" else oc["pf"][rows].sum(0) / max(1, int(oc["npause"][rows].sum()))
    best, key = len(oc["ths"]) - 1, None
    for j in np.nonzero(rate <= max_fc)[0]:
        keep = lat[~fc[:, j], j]
        kj = (_q(keep, 0.5), _q(keep, 0.9)) + ((float(np.isinf(keep).mean()) if len(keep) else 1.0,) if tie_miss else ())
        if key is None or kj < key:
            best, key = int(j), kj
    return best


def v2_crossfit(oc: dict, folds: np.ndarray, max_fc: float, unit: str, tie_miss: bool = False):
    """Fixed operating point: the threshold chosen on one leave-meetings-out half applied to the other half; returns
    per-conversation (fc, lat, pf) at the cross-fitted thresholds and the thresholds per fold."""
    n = len(folds)
    fc, lat, pf = np.zeros(n, bool), np.full(n, np.inf), np.zeros(n, np.int64)
    th = {}
    for f in np.unique(folds):
        j = _select(oc, folds != f, max_fc, unit, tie_miss)
        r = folds == f
        fc[r], lat[r], pf[r] = oc["fc"][r, j], oc["lat"][r, j], oc["pf"][r, j]
        th[int(f)] = round(float(oc["ths"][j]), 5)
    return fc, lat, pf, th


def _paired(a, b, idx) -> dict:
    """Paired bootstrap of (system a - system b): miss rate and P50 at their fixed thresholds (same resamples)."""
    from audioforge.conversation import _q
    dm, dp = [], []
    for ii in idx:
        ka, kb = a[1][ii][~a[0][ii]], b[1][ii][~b[0][ii]]
        dm.append((np.isinf(ka).mean() if len(ka) else 1.0) - (np.isinf(kb).mean() if len(kb) else 1.0))
        pa, pb = _q(ka, 0.5), _q(kb, 0.5)
        dp.append(pa - pb if np.isfinite(pa) and np.isfinite(pb) else (0.0 if pa == pb else np.sign(pa - pb) * np.inf))
    ka, kb = a[1][~a[0]], b[1][~b[0]]
    dm, dp = np.array(dm), np.array(dp)
    pa, pb = _q(ka, 0.5), _q(kb, 0.5)
    return dict(miss_diff=round(float(np.isinf(ka).mean() - np.isinf(kb).mean()), 4),
                miss_diff_ci=[round(float(np.quantile(dm, 0.025, method="inverted_cdf")), 4),
                              round(float(np.quantile(dm, 0.975, method="inverted_cdf")), 4)],
                fc_a=round(float(a[0].mean()), 4), fc_b=round(float(b[0].mean()), 4), p50_a=pa, p50_b=pb,
                p50_diff_ci=[float(np.quantile(dp, 0.025, method="inverted_cdf")),
                             float(np.quantile(dp, 0.975, method="inverted_cdf"))])


V2_PAIRS = [("head_v3_stream_causal_dominant", "timeout_stream_causal_dominant"),
            ("head_v3_stream_oracle", "timeout_stream_oracle_labelarm"),
            ("head_v3_stream_first_active", "timeout_stream_first_active"),
            ("head_v3_stream_causal_dominant", "timeout_any_speaker_stream"),
            ("timeout_stream_causal_dominant", "timeout_stream_oracle_labelarm"),
            ("timeout_stream_causal_dominant_labelarm", "timeout_stream_oracle_labelarm"),
            ("head_v3_stream_causal_dominant", "head_v3_stream_oracle")]


V2_HYBRID_PAIRS = [("hybrid_stream_causal_dominant", "timeout_stream_causal_dominant"),
                   ("hybrid_stream_causal_dominant", "head_v3_stream_causal_dominant"),
                   ("hybrid_stream_oracle", "timeout_stream_oracle"),
                   ("hybrid_stream_oracle", "head_v3_stream_oracle"),
                   ("hybrid_stream_causal_dominant", "hybrid_stream_oracle")]
# voice enrollment (§8): each voice binding vs causal_dominant and vs the oracle binding, per system; and the
# oracle-identity upper bound vs the label-free ones
V2_VOICE_PAIRS = [(f"{sys_}_{m}", f"{sys_}_{ref}") for m in V2_VOICE_MODES
                  for sys_ in ("timeout_stream", "head_v3_stream", "hybrid_stream")
                  for ref in ("causal_dominant", "oracle")] + \
                 [(f"{sys_}_{m}", f"{sys_}_voice_oracle") for m in ("voice_first", "voice_dominant")
                  for sys_ in ("timeout_stream", "head_v3_stream", "hybrid_stream")] + \
                 [(f"{sys_}_{a}", f"{sys_}_{b}") for sys_ in ("head_v3_stream", "hybrid_stream")
                  for a, b in (("oracle_reordered", "oracle"), ("voice_oracle", "oracle_reordered"),
                               ("causal_dominant", "oracle_reordered"))]
# §9: TitaNet-followed rules vs causal_dominant / oracle, vs their own-head counterpart (the backend effect), the
# identity rules vs the oracle-identity bound, and after_prev_end's two followers
V2_VOICE_PAIRS += [(f"{sys_}_{m}", f"{sys_}_{ref}") for m in V2_TITANET_MODES + V2_CONTROL_MODES
                   for sys_ in ("timeout_stream", "head_v3_stream", "hybrid_stream")
                   for ref in ("causal_dominant", "oracle")] + \
                  [(f"{sys_}_{a}", f"{sys_}_{b}") for sys_ in ("timeout_stream", "head_v3_stream", "hybrid_stream")
                   for a, b in [(f"{m}_titanet", m) for m in V2_VOICE_MODES] +
                   [("voice_dominant_titanet", "voice_oracle_titanet"), ("voice_first_titanet", "voice_oracle_titanet"),
                    ("after_prev_end_titanet", "voice_oracle_titanet"), ("voice_explicit_titanet", "voice_oracle_titanet"),
                    ("after_prev_end_titanet", "after_prev_end_causal"), ("after_prev_end_causal", "causal_dominant"),
                    ("voice_oracle_titanet", "oracle_reordered")] +
                   [(f"{m}_titanet_e{n}", ref) for m in ("voice_explicit", "after_prev_end") for n in V2_ENROLL_LENS
                    for ref in (f"{m}_titanet", "causal_dominant", "oracle")] +
                   [(f"{m}_titanet_clean", ref) for m in ("voice_explicit", "after_prev_end")
                    for ref in (f"{m}_titanet_e40", f"{m}_titanet", "causal_dominant", "oracle")]]


def v2_voice_modes_available(convs) -> tuple:
    """The voice bindings whose stored per-frame binding exists for every window (bind stage run)."""
    w = V2_BIND_WORK[0]
    if w is None:
        return ()
    return tuple(m for m in V2_STORED_MODES if all(v2_bind_path(w, m, v).exists() for v in convs))


def v2_repr_modes_available(convs, work, score_tag, tag) -> tuple:
    """oracle_reordered when its head scores exist for every window."""
    if score_tag is None:
        return ()
    return tuple(m for m in V2_REPR_MODES if all(v2_score_path(work, m + score_tag, v, tag).exists() for v in convs))


V2_HYBRID_QMIN = 0.85  # hybrid grid: candidate thresholds = pre-end / pause maxima at or above this quantile


def v2_hybrid_grid(sc, on, en, pauses, qmin: float = V2_HYBRID_QMIN, cap=None) -> np.ndarray:
    """Candidate thresholds of one detector for the hybrid's joint sweep. A threshold only changes the false-cutoff
    sets at a turn's pre-end maximum (or a pause's maximum), and between two such values the lowest one fires
    earliest, so the maxima themselves are the only candidates an operating point can need; at <= 5 % FC only those
    at or above their ``qmin`` quantile can qualify (0.85 leaves room for the per-fold choice). + inf (never fires,
    so each detector alone is inside the grid); ``cap``: values >= cap fire no earlier than never within the horizon
    and are dropped (the timeout: silence runs beyond the post-end horizon)."""
    pre = np.array([float(np.max(np.asarray(x, np.float64)[o:e])) if e > o else -np.inf for x, o, e in zip(sc, on, en)])
    pm = []
    for x, ps in zip(sc, pauses):
        x = np.asarray(x, np.float64)
        pm += [float(x[max(a, 0): min(b, len(x))].max()) for a, b in ps if len(x[max(a, 0): min(b, len(x))])]
    pm = np.asarray(pm)
    c = [pre[pre >= np.quantile(pre, qmin)]]
    if len(pm):
        c.append(pm[pm >= np.quantile(pm, qmin)])
    c = np.unique(np.concatenate(c))
    if cap is not None:
        c = c[c < cap]
    return np.unique(np.append(c, np.inf))


def v2_track_quality(convs, tracks, on, en, C: int, R: int, k: int = ENROLL_K, s: int = ENROLL_S) -> dict:
    """Streaming-track quality vs the labels: pooled frame DER over all columns (der_parts, best permutation, no
    collar), the bound primary track's miss / FA / post-end activity (act_stats) and onset / offset lag per binding,
    and the diarizer's nominal emission delay (_stream_emit: mean over frames of emit(t) - t)."""
    tot = np.zeros(4)
    for v, p in zip(convs, tracks):
        m, f, c, sp = der_parts(torch.from_numpy((p > 0.5).astype(np.float32)),
                                torch.from_numpy((np.asarray(v["spk_targets"]) > 0.5).astype(np.float32)))
        tot += (m, f, c, sp)
    em = _stream_emit(C, R)
    out = {"der": round(float(tot[:3].sum() / tot[3]), 4), "der_miss": round(float(tot[0] / tot[3]), 4),
           "der_fa": round(float(tot[1] / tot[3]), 4), "der_confusion": round(float(tot[2] / tot[3]), 4),
           "emission_delay_frames_mean": float(np.mean([em(t) - t for t in range(12 * C)])),
           "input_buffer_ms": (C + R) * 80.0}
    for mode in ("oracle", "causal_dominant"):
        acts = [_v2_cols(v, p, mode, k, s)[0] for v, p in zip(convs, tracks)]
        q = act_stats(acts, convs, en)
        onl, offl = [], []
        for a_, o, e in zip(acts, on, en):
            h = np.asarray(a_) > 0.5
            nz = np.nonzero(h[o:])[0]
            onl.append(float(nz[0]) if len(nz) else np.nan)
            z = np.nonzero(~h[e:])[0]  # first inactive frame at / after the label end
            offl.append(float(z[0]) if len(z) else np.nan)
        onl, offl = np.array(onl), np.array(offl)
        q.update(onset_lag_frames_median=float(np.nanmedian(onl)), onset_lag_frames_mean=round(float(np.nanmean(onl)), 3),
                 offset_lag_frames_median=float(np.nanmedian(offl)),
                 offset_lag_frames_mean=round(float(np.nanmean(offl)), 3),
                 never_off_after_end=int(np.isnan(offl).sum()))
        out[mode] = q
    return out


def v2_block(convs, meta, tracks, work: Path, score_tag: str | None, horizons: dict, chunk: int, n_boot: int,
             k: int = ENROLL_K, s: int = ENROLL_S, score_dir_tag: str = "", cfg_name: str = "low_latency",
             hybrid: bool = False) -> dict:
    """One evaluation block: systems x horizon on a list of windows (``tracks`` None = label-only systems), per
    stratum, in-sample and cross-fitted (leave-meetings-out) 5 % FC operating points with bootstrap CIs, and paired
    head-vs-timeout comparisons. ``score_tag``: suffix of the stored head scores ('@2s' default windows, '' extended);
    None = no head rows. ``horizons`` {name: post_end_frames or None (= the window's own trail)}. ``cfg_name``: the
    diarizer config the tracks were made with (its C, R set the emission rule). ``hybrid``: also the head-OR-timeout
    system per binding with head scores (hybrid_stream_<binding>: fire at the earlier of the head at θ and the
    deployable timeout at k on the same bound track), (θ, k) cross-fitted jointly like the single thresholds, plus
    the track-quality summary."""
    from audioforge.conversation import bootstrap_ci, eot_outcomes_or, floor_stratum, outcome_metrics, pause_runs
    cfg = v2_diar_cfg(cfg_name)
    C, R = cfg["chunk_len"], cfg["chunk_right_context"]
    n = len(convs)
    on = np.array([v["onset_frame"] for v in convs])
    en = np.array([v["turn_end_frame"] for v in convs])
    strata = np.array([floor_stratum(v["spk_targets"], int(e), V2_FLOOR_HORIZON) for v, e in zip(convs, en)])
    groups = {"open": strata == "open", "taken": strata != "open", "overlap": strata == "overlap",
              "switch": strata == "switch"}
    pauses = [pause_runs(v["hes"]) for v in convs]
    folds = np.array([0 if v["meeting"] in V2_DEV_FOLDS[0] else 1 for v in convs])
    avail = np.array([m["post_avail"] for m in meta])
    reason = np.array([m["end_reason"] for m in meta])
    res = {"n": n, "strata_counts": {g: int(m.sum()) for g, m in groups.items()},
           "n_pauses": int(sum(len(p) for p in pauses)), "turns_with_pause": int(sum(bool(p) for p in pauses)),
           "end_reason_counts": {r: int((reason == r).sum()) for r in np.unique(reason)},
           "post_end_frames_pct_5_50_95": [float(np.percentile(avail, q)) for q in (5, 50, 95)],
           "fold_counts": [int((folds == f).sum()) for f in (0, 1)]}
    systems, vmodes = {}, ()
    if tracks is not None:
        tracks = [np.stack([_fit(p[:, j], len(v["spk_act"])) for j in range(p.shape[1])], 1)
                  for v, p in zip(convs, tracks)]
        orc = np.array([enroll_column(p, v["spk_act"], o, e) for p, v, o, e in zip(tracks, convs, on, en)])
        fa = np.array([enroll_first_active(p) for p in tracks])
        agree = {"first_active_vs_oracle": round(float((fa == orc).mean()), 4),
                 "first_active_vs_oracle_open": round(float((fa == orc)[groups["open"]].mean()), 4)}
        for kk, ss in ((k, s), (25, 12), (50, 25), (25, 50), (12, 12)):
            cd = [enroll_causal_dominant(p, kk, ss) for p in tracks]
            at_end = np.array([c[e - 1] == o for c, e, o in zip(cd, en, orc)])
            agree[f"causal_dominant_k{kk}_s{ss}"] = {
                "at_turn_end": round(float(at_end.mean()), 4),
                "at_turn_end_open": round(float(at_end[groups["open"]].mean()), 4),
                "frames_onset_to_end": round(float(np.sum([(c[a_:e] == o).sum() for c, a_, e, o in zip(cd, on, en, orc)])
                                                   / np.sum(en - on)), 4),
                "rebinds_per_turn": round(float(np.mean([int((np.diff(c) != 0).sum()) for c in cd])), 3)}
        vmodes = v2_voice_modes_available(convs)
        for m in vmodes:  # voice enrollment (EOT_BENCH_V2.md §8): the stored per-frame bindings
            vc = [_v2_cols(v, p, m)[3] for v, p in zip(convs, tracks)]
            at_end = np.array([c[e - 1] == o for c, e, o in zip(vc, en, orc)])
            agree[m] = {"at_turn_end": round(float(at_end.mean()), 4),
                        "at_turn_end_open": round(float(at_end[groups["open"]].mean()), 4),
                        "frames_onset_to_end": round(float(np.sum([(c[a_:e] == o).sum() for c, a_, e, o in
                                                                   zip(vc, on, en, orc)]) / np.sum(en - on)), 4),
                        "rebinds_per_turn": round(float(np.mean([int((np.diff(c) != 0).sum()) for c in vc])), 3)}
        res["enrollment_agreement"] = agree
        if hybrid:
            res["track_quality"] = v2_track_quality(convs, tracks, on, en, C, R, k, s)
        emit_stream, emit_head = _stream_emit(C, R), _stream_emit(C, R, chunk)
        rmodes = v2_repr_modes_available(convs, work, score_tag, score_dir_tag)
        for mode in rmodes:  # head (and hybrid) rows only; the deployable timeout on the oracle column is shared
            sp = [v2_score_path(work, mode + score_tag, v, score_dir_tag) for v in convs]
            systems[f"head_v3_stream_{mode}"] = ([np.load(q) for q in sp], emit_head)
        for mode in ENROLL_MODES + vmodes:
            acts = [_v2_cols(v, p, mode, k, s)[0] for v, p in zip(convs, tracks)]
            # timeout cascade on the bound track: '_labelarm' = silence counted from the LABEL onset (v1), else from
            # the bound track's first active frame (deployable)
            tag = "timeout_stream_oracle" if mode == "oracle" else f"timeout_stream_{mode}"
            systems[f"{tag}_labelarm"] = ([silence_scores(a, o) for a, o in zip(acts, on)], emit_stream)
            systems[tag] = ([silence_scores(a, arm_frame(a)) for a in acts], emit_stream)
            if score_tag is not None:
                sp = [v2_score_path(work, mode + score_tag, v, score_dir_tag) for v in convs]
                if all(q.exists() for q in sp):
                    systems[f"head_v3_stream_{mode}"] = ([np.load(q) for q in sp], emit_head)
        anys = [p.max(1) for p in tracks]
        systems["timeout_any_speaker_stream"] = ([silence_scores(a, arm_frame(a)) for a in anys], emit_stream)
        # sensitivity of the causal cascade to (k, s) (no model)
        for kk, ss in ((25, 12), (50, 25), (25, 50), (12, 12)):
            acts = [bound_track(p, enroll_causal_dominant(p, kk, ss))[0] for p in tracks]
            systems[f"timeout_stream_causal_dominant_k{kk}_s{ss}"] = (
                [silence_scores(a, arm_frame(a)) for a in acts], emit_stream)
    systems["timeout_any_speaker_oracle"] = ([silence_scores(v["spk_targets"].max(1), o) for v, o in zip(convs, on)], None)
    systems["timeout_primary_oracle"] = ([silence_scores(v["spk_act"], o) for v, o in zip(convs, on)], None)
    res["systems"], outs = {}, {}

    def fixed_points(name, r, hz, L, cens, oc, th_fmt=None):
        for unit in ("turn", "pause"):
            fc, lat, pf, th = v2_crossfit(oc, folds, 0.05, unit, tie_miss=th_fmt is not None)
            if th_fmt is not None:  # joint (θ, k) grid: report the chosen pair per fold
                th = {f: th_fmt(j) for f, j in th.items()}
            pt = outcome_metrics(fc, lat, pf, oc["npause"])
            pt.update(bootstrap_ci(fc, lat, n_boot, 0), thresholds_by_fold=th)
            m = np.isinf(lat) & ~fc
            pt.update(n_miss=int(m.sum()), miss_censored=int((m & cens).sum()),
                      miss_primary_resumed=int((m & (reason == "resume") & ((avail < L) if L else True)).sum()))
            pt["strata"] = {g: {**outcome_metrics(fc[msk], lat[msk], pf[msk], oc["npause"][msk]),
                                **bootstrap_ci(fc[msk], lat[msk], n_boot, 0)} for g, msk in groups.items()}
            r[hz][f"fixed_5pct_{unit}_fc"] = pt
            outs[(name, hz, unit)] = (fc, lat)

    hyb = {}  # hybrid_stream_<mode>: (head scores, head emission, timeout scores, timeout emission)
    if hybrid and tracks is not None:
        for mode in ENROLL_MODES + vmodes + V2_REPR_MODES:
            tag = "timeout_stream_oracle" if mode in ("oracle", "oracle_reordered") else f"timeout_stream_{mode}"
            if f"head_v3_stream_{mode}" in systems:
                hyb[f"hybrid_stream_{mode}"] = systems[f"head_v3_stream_{mode}"] + systems[tag]
    for name, spec in list(systems.items()) + list(hyb.items()):
        r = {}
        for hz, L in horizons.items():
            cens = ((avail < L) if L is not None else np.ones(n, bool)) & (reason != "resume")
            if name in hyb:  # joint (θ, k) sweep on the emission-indexed tracks (both detectors' own emission rule)
                ha, ea, tb, eb = spec
                za, zb = _emit_transform(ha, en, ea), _emit_transform(tb, en, eb)
                ga = v2_hybrid_grid(za, on, en, pauses)
                gb = v2_hybrid_grid(zb, on, en, pauses)
                oc = eot_outcomes_or(za, zb, on, en, ga, gb, post_end_frames=L, pauses=pauses)

                def th_fmt(j, oc=oc):
                    ta, tb_ = oc["ths"][int(j)] if np.ndim(j) == 0 else j
                    return {"theta": round(float(ta), 5) if np.isfinite(ta) else None,
                            "k_frames": int(tb_) + 1 if np.isfinite(tb_) else None}

                ji = _select(oc, np.ones(n, bool), 0.05, "turn", tie_miss=True)
                pt = outcome_metrics(oc["fc"][:, ji], oc["lat"][:, ji], oc["pf"][:, ji], oc["npause"])
                r[hz] = {"in_sample": {"at_5pct_fc": {**th_fmt(ji), **pt}},
                         "grid_size": [int(len(ga)), int(len(gb))]}
                # v2_crossfit reports oc["ths"][j] as a float: give it the column index, then format
                oc_i = dict(oc, ths=np.arange(oc["fc"].shape[1], dtype=np.float64))
                fixed_points(name, r, hz, L, cens, oc_i, th_fmt=lambda j, oc=oc: th_fmt(oc["ths"][int(round(j))]))
                del oc, oc_i
                continue
            sc, emit = spec
            kw = dict(pauses=pauses, post_end_frames=L, censored=cens, groups=groups, n_boot=n_boot,
                      return_outcomes=True)
            b = eot_bench_emit(sc, on, en, emit, **kw) if emit else eot_bench(sc, on, en, **kw)
            oc = b.pop("_outcomes")
            r[hz] = {"in_sample": {"at_5pct_fc": b["at_max_fc"], "at_p50_le_400ms": b["at_fixed_latency"]}}
            fixed_points(name, r, hz, L, cens, oc)
        if "2s" in horizons and "6s" in horizons:  # 2 s misses caused by the window limit: they fire by 6 s
            for unit in ("turn", "pause"):
                f2, l2 = outs[(name, "2s", unit)]
                f6, l6 = outs[(name, "6s", unit)]
                m2 = np.isinf(l2) & ~f2
                fired = m2 & ~f6 & np.isfinite(l6)
                r["2s"][f"fixed_5pct_{unit}_fc"]["miss_fired_by_6s"] = int(fired.sum())
                r["2s"][f"fixed_5pct_{unit}_fc"]["miss_fired_by_6s_open"] = int((fired & groups["open"]).sum())
        res["systems"][name] = r
    idx = np.random.default_rng(0).integers(0, n, (n_boot, n))
    res["paired"] = {}
    known = set(systems) | set(hyb)
    for a_, b_ in V2_PAIRS + (V2_HYBRID_PAIRS if hyb else []) + V2_VOICE_PAIRS:
        if a_ not in known or b_ not in known:
            continue
        for hz in horizons:
            for unit in ("turn", "pause"):
                A, B = outs[(a_, hz, unit)], outs[(b_, hz, unit)]
                d = {"all": _paired(A, B, idx)}
                for g in ("open", "taken"):
                    sub = np.nonzero(groups[g])[0]
                    gi = np.random.default_rng(1).integers(0, len(sub), (n_boot, len(sub)))
                    d[g] = _paired((A[0][sub], A[1][sub]), (B[0][sub], B[1][sub]), gi)
                res["paired"][f"{a_} - {b_} | {hz} | fixed_5pct_{unit}_fc"] = d
    return res


def eval_turn_v2(a) -> dict | None:
    """--bench v2 stages (each one process <= 10 min, resumable): tracks (Sortformer only), scores (turn model
    only), report (no model)."""
    from audioforge.conversation import floor_stratum
    work = Path(a.v2_work)
    work.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, meta_base = v2_data(a.trail_sec)
    print(f"  v2 data: {len(ext)} turns, trail {a.trail_sec}s", flush=True)
    if a.v2_stage == "tracks":  # floor-open turns first (the stratum that matters), then the rest
        op = [floor_stratum(v["spk_targets"], v["turn_end_frame"], V2_FLOOR_HORIZON) == "open" for v in ext]
        order = [i for i in range(len(ext)) if op[i]] + [i for i in range(len(ext)) if not op[i]]
        cfg_name, td = getattr(a, "v2_diar_config", "low_latency"), getattr(a, "v2_tracks_dir", None)
        assert cfg_name == "low_latency" or td, "a non-default --v2-diar-config needs its own --v2-tracks-dir"
        kw = {} if cfg_name == "low_latency" and td is None else dict(cfg_name=cfg_name, tracks_dir=td)
        if getattr(a, "device", "cpu") != "cpu":
            kw["device"] = a.device
        return {"tracks": v2_tracks(ext, ds, work, a.v2_budget, a.diar_ckpt or V2_DIAR_CKPT, order, **kw)}
    V2_BIND_WORK[0] = work
    if a.v2_stage in ("embed", "bind"):
        t0 = time.time()
        td = getattr(a, "v2_tracks_dir", None)
        if getattr(a, "v2_embedder", "head") in ("titanet", "titanet_clean"):  # §9 / §9b: TitaNet-L backend
            from audioforge.enrollment import TitaNetEmbedder
            tn = TitaNetEmbedder(path=a.titanet)
            if a.v2_stage == "embed" and a.v2_embedder == "titanet_clean":  # needs the turn model's VAD head once
                if any(not v2_vad_path(work, v).exists() for v in ext):
                    n = v2_vad(load_model(a.ckpt, "cpu"), ext, work, a.batch_size)
                    print(f"  v2 vad: {n} windows ({time.time() - t0:.0f}s)", flush=True)
                return {"embed_titanet_clean": v2_embed_titanet_clean(tn, ext, work, a.v2_budget - (time.time() - t0), td)}
            if a.v2_stage == "embed":
                return {"embed_titanet": v2_embed_titanet(tn, ext, work, a.v2_budget - (time.time() - t0), td)}
            tmodes = V2_TITANET_MODES + V2_CONTROL_MODES
            modes = [m for m in a.v2_bindings.split(",") if m in tmodes] or list(tmodes)
            return {"bind": v2_bind(None, base, ext, work, modes, a.v2_budget - (time.time() - t0), td, tn=tn)}
        model = load_model(a.ckpt, "cpu")
        if a.v2_stage == "embed":
            return {"embed": v2_embed(model, ext, work, a.v2_budget - (time.time() - t0), a.batch_size)}
        modes = [m for m in a.v2_bindings.split(",") if m in V2_VOICE_MODES] or list(V2_VOICE_MODES)
        return {"bind": v2_bind(model, base, ext, work, modes, a.v2_budget - (time.time() - t0), td)}
    if a.v2_stage == "scores":
        t0 = time.time()
        model = load_model(a.ckpt, "cpu")
        print(f"  loaded {a.ckpt} in {time.time() - t0:.0f}s (chunk {chunk_of(model)})", flush=True)
        assert chunk_of(model) == a.v2_chunk, (chunk_of(model), a.v2_chunk)
        td = getattr(a, "v2_tracks_dir", None)
        left = v2_scores(model, base, ext, work, a.v2_bindings.split(","), a.v2_budget - (time.time() - t0),
                         a.batch_size, a.v2_tag, **({"tracks_dir": td} if td else {}))
        return {"scores_left": left}
    t0 = time.time()
    res = {"protocol": "eot-bench v2 (research/EOT_BENCH_V2.md)", "enroll_params": {"k_frames": ENROLL_K,
           "s_frames": ENROLL_S}, "folds": [list(f) for f in V2_DEV_FOLDS], "floor_horizon_frames": V2_FLOOR_HORIZON,
           "score_tag": a.v2_tag}
    cfg_name, td = getattr(a, "v2_diar_config", "low_latency"), getattr(a, "v2_tracks_dir", None)
    hyb = bool(getattr(a, "v2_hybrid", False))
    bkw = {} if cfg_name == "low_latency" and not hyb else dict(cfg_name=cfg_name, hybrid=hyb)
    if bkw or td:
        res.update(diar_config=dict(v2_diar_cfg(cfg_name), name=cfg_name, enc_left_context=SORTFORMER_ENC_LEFT),
                   tracks_dir=td, hybrid=hyb)
    btr = (lambda v, i: v2_base_track(v)) if td is None else (lambda v, i: v2_base_track(v, ext[i], td))
    # A: all turns, default windows (2 s trail), the cached streaming tracks (= v1 inputs)
    res["A_default_windows_all"] = v2_block(base, meta_base, [btr(v, i) for i, v in enumerate(base)], work, "@2s",
                                            {"2s": None}, a.v2_chunk, a.n_boot, score_dir_tag=a.v2_tag, **bkw)
    print(f"  block A {time.time() - t0:.0f}s", flush=True)
    # B: all turns, extended windows, label-only systems, 2 s and 6 s horizons
    res["B_extended_windows_labels"] = v2_block(ext, meta, None, work, None, {"2s": 25, "6s": 75}, a.v2_chunk, a.n_boot)
    print(f"  block B {time.time() - t0:.0f}s", flush=True)
    # C: the turns with an extended streaming track (floor-open first), all systems, 2 s and 6 s horizons
    have = [i for i, v in enumerate(ext) if v2_track_path(work, v, td).exists()]
    if have:
        sub = lambda xs: [xs[i] for i in have]  # noqa: E731
        res["C_extended_windows_stream"] = v2_block(sub(ext), sub(meta), [np.load(v2_track_path(work, ext[i], td))
                                                                          for i in have],
                                                    work, "", {"2s": 25, "6s": 75}, a.v2_chunk, a.n_boot,
                                                    score_dir_tag=a.v2_tag, **bkw)
        print(f"  block C (n={len(have)}) {time.time() - t0:.0f}s", flush=True)
    # v1 protocol rows at n = 974 (default windows, cached tracks, oracle binding, label arming, in-sample threshold)
    from audioforge.conversation import baselines
    on, en = _onset_end(base)
    C, R = v2_diar_cfg(cfg_name)["chunk_len"], v2_diar_cfg(cfg_name)["chunk_right_context"]
    acts = [_v2_cols(v, btr(v, i), "oracle")[0] for i, v in enumerate(base)]
    v1 = {"timeout_stream_oracle": _pt(eot_bench_emit([silence_scores(x, o) for x, o in zip(acts, on)], on, en,
                                                      _stream_emit(C, R)))}
    vv = [dict(v, onset_frame=o, turn_end_frame=e, primary_act=np.asarray(v["spk_act"])) for v, o, e in zip(base, on, en)]
    bl = baselines(vv)
    v1.update(timeout_primary_oracle=_pt(bl["timeout_primary_oracle"]),
              timeout_any_speaker_oracle=_pt(bl["timeout_any_speaker_oracle"]))
    sp = [v2_score_path(work, "oracle@2s", v, a.v2_tag) for v in base]
    if all(q.exists() for q in sp):
        v1["head_v3_stream_oracle"] = _pt(eot_bench_emit([np.load(q) for q in sp], on, en,
                                                         _stream_emit(C, R, a.v2_chunk)))
    res["v1_protocol_n974"] = v1
    return res


# --------------------------------------------------------------------------- diar + vad
def der_parts(pred: torch.Tensor, ref: torch.Tensor):
    """Miss / FA / confusion frame counts at the permutation frame_der picks (same cost), + speech."""
    S = max(ref.shape[1], pred.shape[1])
    ref, pred = (torch.nn.functional.pad(x, (0, S - x.shape[1])) for x in (ref, pred))
    best = None
    for perm in itertools.permutations(range(S)):
        p = pred[:, list(perm)]
        n_ref, n_hyp = ref.sum(1), p.sum(1)
        correct = torch.minimum(p, ref).sum(1)
        err = float((torch.maximum(n_ref, n_hyp) - correct).sum())
        if best is None or err < best[0]:
            miss = float((n_ref - n_hyp).clamp(min=0).sum())
            fa = float((n_hyp - n_ref).clamp(min=0).sum())
            best = (err, miss, fa, err - miss - fa)
    return best[1], best[2], best[3], float(ref.sum())


@torch.no_grad()
def eval_diar_vad(model, n: int, batch_size: int) -> dict:
    val = [derive_labels(dict(v), ["vad"]) for v in ami_dev("diar", n)]  # vad = any speaker active
    diar = _diar_name(model)
    vad = next((k for k, v in model.head_cfg.items() if v["type"] == "frame" and v.get("key") == "vad"), None)
    col = Collate(model.tokenizer)
    acc = {k: {"der_sum": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0, "speech": 0.0}
           for k in ("head", "all_silence", "one_speaker_always", "one_speaker_oracle_vad")}
    v = dict(correct=0, total=0, tp=0, pos=0, fp=0, base_correct=0)
    overlap = n_items = 0
    for i in range(0, len(val), batch_size):
        b = col(val[i: i + batch_size])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        pred = model.heads[diar].decode(model.head_input(diar, enc, hidden), elen)  # offline, thresh 0.5
        for j in range(len(elen)):
            T = min(int(elen[j]), int(b["spk_targets_len"][j]))
            ref = b["spk_targets"][j, :T].float()
            overlap += int((ref.sum(1) > 1).sum())
            n_items += 1
            cands = {"head": pred[j, :T].cpu(), "all_silence": torch.zeros_like(ref),
                     "one_speaker_always": torch.cat([torch.ones(T, 1), torch.zeros(T, ref.shape[1] - 1)], 1),
                     # oracle any-speaker VAD as a single speaker: the best a speaker-blind diarizer can do
                     "one_speaker_oracle_vad": torch.cat([(ref.sum(1, keepdim=True) > 0).float(),
                                                          torch.zeros(T, ref.shape[1] - 1)], 1)}
            for k, p in cands.items():
                a = acc[k]
                a["der_sum"] += frame_der(p, ref)
                m, f, c, s = der_parts(p, ref)
                a["miss"] += m
                a["fa"] += f
                a["conf"] += c
                a["speech"] += s
        if vad:
            p = model.heads[vad].decode(model.head_input(vad, enc, hidden), elen) > 0.5
            lab = b["vad"] > 0.5
            T = min(p.shape[1], lab.shape[1])
            ar = torch.arange(T)[None]
            valid = (ar < elen[:, None]) & (ar < b["vad_len"][:, None])
            p, lab = p[:, :T], lab[:, :T]
            v["correct"] += int(((p == lab) & valid).sum())
            v["total"] += int(valid.sum())
            v["tp"] += int((p & lab & valid).sum())
            v["fp"] += int((p & ~lab & valid).sum())
            v["pos"] += int((lab & valid).sum())
            v["base_correct"] += int((lab & valid).sum())  # always-speech baseline is right on speech frames
    res = {"n": n_items, "window_sec": 20.0, "overlap_frac_of_frames": None}
    speech_frames = acc["head"]["speech"]
    for k, a in acc.items():
        sp = max(1.0, a["speech"])
        res[f"diar_{k}"] = {"der_mean_of_windows": round(a["der_sum"] / max(1, n_items), 4),
                            "der_pooled": round((a["miss"] + a["fa"] + a["conf"]) / sp, 4),
                            "miss": round(a["miss"] / sp, 4), "fa": round(a["fa"] / sp, 4),
                            "confusion": round(a["conf"] / sp, 4)}
    res["speaker_frames_total"] = int(speech_frames)
    res["overlap_frames"] = overlap
    if vad:
        res["vad_head"] = {"acc": round(v["correct"] / v["total"], 4), "recall": round(v["tp"] / max(1, v["pos"]), 4),
                           "precision": round(v["tp"] / max(1, v["tp"] + v["fp"]), 4),
                           "speech_frac": round(v["pos"] / v["total"], 4)}
        res["vad_always_speech"] = {"acc": round(v["base_correct"] / v["total"], 4), "recall": 1.0,
                                    "precision": round(v["pos"] / v["total"], 4)}
    del res["overlap_frac_of_frames"]
    return res


# --------------------------------------------------------------------------- speaker EER
@torch.no_grad()
def eval_spk(model, n: int, batch_size: int) -> dict:
    val = ami_dev("asr", n)
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "speaker")
    head, col = model.heads[name], Collate(None)
    embs, pooled = [], []
    for i in range(0, len(val), batch_size):
        b = col([{"audio": v["audio"]} for v in val[i: i + batch_size]])
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        x = model.head_input(name, enc, hidden)
        embs.append(head.embed(x, elen))
        m = (torch.arange(x.shape[1])[None] < elen[:, None]).float()[..., None]
        pooled.append(torch.nn.functional.normalize((x * m).sum(1) / m.sum(1), dim=-1))
    spk = torch.tensor([v["speaker"] for v in val])
    meet = [v["meeting"] for v in val]
    iu = torch.triu_indices(len(val), len(val), 1)
    lab = (spk[iu[0]] == spk[iu[1]]).long()
    same_meet = torch.tensor([meet[a] == meet[b] for a, b in iu.T.tolist()])
    res = {"n_segments": len(val), "n_speakers": int(spk.unique().numel()), "n_trials": int(len(lab)),
           "n_target_trials": int(lab.sum()), "n_within_meeting_trials": int(same_meet.sum()),
           "mean_sec": round(float(np.mean([v["duration"] for v in val])), 2)}
    for tag, E in (("spk_head", torch.cat(embs)), ("meanpool_layer_mix", torch.cat(pooled))):
        s = (E[iu[0]] * E[iu[1]]).sum(-1)
        res[f"eer_{tag}"] = round(eer(s, lab), 4)
        res[f"eer_{tag}_within_meeting"] = round(eer(s[same_meet], lab[same_meet]), 4)
    return res


# --------------------------------------------------------------------------- ASR sanity
@torch.no_grad()
def eval_asr(model, n: int, batch_size: int, head: str = "rnnt") -> dict:
    val = ami_dev("asr", n)
    hyps = []
    for i in range(0, len(val), batch_size):
        hyps += model.transcribe([v["audio"] for v in val[i: i + batch_size]], head=head)
    refs = [normalize_text(v["text"]) for v in val]
    hyps = [normalize_text(h) for h in hyps]
    return {"n": len(val), "head": head, "wer": round(wer(refs, hyps), 4),
            "ref_words": sum(len(r.split()) for r in refs),
            "mean_sec": round(float(np.mean([v["duration"] for v in val])), 2),
            "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:5]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/stage1_heads_pretrained.step500.afm")
    ap.add_argument("--n", type=int, default=64, help="examples per set (asr uses n // 2)")
    ap.add_argument("--n-asr", type=int, default=None)
    ap.add_argument("--tasks", default="turn,diar,spk,asr")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/stage1_step500_eval.json")
    ap.add_argument("--diar-ckpt", default=None,
                    help="turn task: also condition the turn head on an EXTERNAL Sortformer .afm's primary track "
                         "(offline pass + StreamingDiarizer low-latency) and add its silence-timeout cascade")
    ap.add_argument("--diar-cache", default=None, help="pickle of the external diarizer's per-example probs "
                                                        "(resumable across processes; without it no budget stop), or "
                                                        "a make_sortformer_tracks.py cache dir (data/ami/cache/"
                                                        "sortformer/dev): read its cached streaming tracks")
    ap.add_argument("--diar-budget", type=float, default=360.0,
                    help="seconds of external-diarizer compute per process before exiting to resume")
    ap.add_argument("--enroll", default="oracle", choices=ENROLL_MODES,
                    help="turn task, external diarizer: primary-column rule (oracle = v1 default; causal_dominant / "
                         "first_active = label-free, eot-bench v2)")
    ap.add_argument("--hybrid", action="store_true",
                    help="turn task: add the head-OR-timeout rows (serve.py turn_policy 'hybrid'), joint (θ, k) sweep")
    ap.add_argument("--bench", default="v1", choices=("v1", "v2"),
                    help="turn task: v2 = the leak-free eot-bench v2 on all dev turns (research/EOT_BENCH_V2.md), "
                         "run as stages --v2-stage tracks|scores|report")
    ap.add_argument("--v2-stage", default="report", choices=("tracks", "embed", "bind", "scores", "report"),
                    help="v2 stage; embed / bind = the voice enrollment (audioforge/enrollment.py, EOT_BENCH_V2.md §8)")
    ap.add_argument("--v2-work", default=None, help="v2: scratch dir for extended-window tracks and head scores")
    ap.add_argument("--v2-embedder", default="head", choices=("head", "titanet", "titanet_clean"),
                    help="v2 embed/bind: the voice-enrollment embedding (head = the model's speaker head, §8; "
                         "titanet = TitaNet-L on the window audio, 5-frame update grid, §9: modes '<rule>_titanet' + "
                         "after_prev_end_causal)")
    ap.add_argument("--titanet", default=None, help="TitaNet-L .nemo (default enrollment.TITANET_NEMO)")
    ap.add_argument("--v2-budget", type=float, default=500.0, help="v2 tracks/scores: seconds per process")
    ap.add_argument("--v2-bindings", default=",".join(V2_BINDINGS))
    ap.add_argument("--v2-chunk", type=int, default=2, help="v2 report: the turn model's encoder chunk (frames)")
    ap.add_argument("--v2-tag", default="", help="v2 scores/report: suffix of the stored head-score dirs "
                    "(scores_<binding>__<tag>) so several checkpoints share one work dir; '' = untagged (v3)")
    ap.add_argument("--trail-sec", type=float, default=6.0, help="v2: post-end audio of the re-cut windows")
    ap.add_argument("--v2-diar-config", default="low_latency", choices=V2_DIAR_CONFIGS,
                    help="v2 tracks/report: StreamingDiarizer card preset of the tracks (low_latency = 1.04 s default; "
                         "low_latency_032 = 0.32 s; needs --v2-tracks-dir); sets the emission rule in the report")
    ap.add_argument("--v2-tracks-dir", default=None,
                    help="v2: extended-window tracks in this dir instead of <work>/tracks (same keys); the default "
                         "windows then use these tracks cropped (v2_base_track) instead of the dev cache")
    ap.add_argument("--v2-hybrid", action="store_true",
                    help="v2 report: add hybrid_stream_<binding> (head at θ OR deployable timeout at k, joint "
                         "cross-fitted (θ, k)) and the track-quality summary")
    ap.add_argument("--device", default="cpu", help="v2 tracks: where the diarizer runs (cpu | mps)")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    if a.bench == "v2":  # stages load at most one model each (report: none)
        r = eval_turn_v2(a)
        if a.v2_stage == "report":
            res.update(ckpt=a.ckpt, turn_v2=dict(r, sec=round(time.time() - t0, 1)))
            out.write_text(json.dumps(res, indent=1))
        print(f"[turn v2 {a.v2_stage}] {time.time() - t0:.0f}s {json.dumps(r)[:600]}")
        return
    model = load_model(a.ckpt, "cpu")
    res["ckpt"] = a.ckpt
    res["att_context_size"] = list(model.encoder.att_context_size)
    print(f"loaded {a.ckpt} in {time.time() - t0:.0f}s heads={list(model.heads)}", flush=True)
    cache = Path(a.diar_cache) if a.diar_cache else None
    fns = {"turn": lambda: eval_turn(model, a.n, a.batch_size, a.diar_ckpt, cache, a.diar_budget, a.enroll,
                                     a.hybrid),
           "diar": lambda: eval_diar_vad(model, a.n, a.batch_size),
           "spk": lambda: eval_spk(model, a.n, a.batch_size),
           "asr": lambda: eval_asr(model, a.n_asr or a.n // 2, a.batch_size)}
    for t in a.tasks.split(","):
        t1 = time.time()
        res[t] = fns[t]()
        res[t]["sec"] = round(time.time() - t1, 1)
        print(f"[{t}] {time.time() - t1:.0f}s {json.dumps(res[t])[:1500]}", flush=True)
        out.write_text(json.dumps(res, indent=1))
    print(f"total {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
