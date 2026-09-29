"""Contamination probe: a speaker-conditioned streaming model gets a WRONG speaker assignment for a short window,
then the correct one again. How long does the damage persist after the correction (state contamination), beyond the
corrupted window itself? research/CONTAMINATION.md.

Two subjects:
  turn   the turn head runs/stage1_turn_v3_trail6.afm (speaker kernels at encoder layers 0 / 2, causal chunked
         attention [70, 1], GRU head with duration counters) on the AMI dev extended turn windows of eot-bench v2 with
         their cached streaming Sortformer tracks under the causal_dominant binding (scripts/research/eval_stage1.py).
  saasr  the speaker-attributed ASR recipe (research/recipes/speaker_attributed_asr.yaml: the activity track steers the
         encoder through speaker kernels; RNNT decodes the bound speaker's text) on held-out synthetic mixtures.

Audio is identical in every condition; only the conditioning track changes on the window [t0 - W, t0). Both encoders
are causal (offline forward == stream_step, tests/test_streaming.py), so a corrupted-then-restored track run offline
is exactly a stream whose binding was wrong for W and corrected at t0, and every frame before the window is bit-
identical to the baseline (checked). The localization (which state carries the damage) uses the same equivalence:
  head/decoder reset   the recurrent state (GRU hidden + duration counters; RNNT prediction net) restarts at t0;
  encoder cache sub    the frames >= t0 come from the baseline encoder run (== a stream whose KV / conv caches were
                       replaced by correctly conditioned ones at t0), the recurrent state stays the corrupted one;
  re-prime (turn)      the deployable version of the cache swap: the encoder is re-run from t0 - 70 frames (its left
                       context) with the corrected track, the old caches are dropped;
  both                 everything is reset (must give a zero difference; a check).
Every protocol is applied to the baseline too, and the difference is measured against the baseline under the same
protocol; the protocol's own cost (baseline under protocol vs plain baseline) is reported next to it.

Usage:
  python scripts/research/contamination_probe.py turn --work <eot-bench v2 work dir with tracks/> --out <dir> [--n 350]
  python scripts/research/contamination_probe.py saasr --ckpt runs/speaker_attributed_asr.afm --out <dir> [--tag noncausal]
  python scripts/research/contamination_probe.py report --turn <dir>/turn.json --saasr <dir>/saasr_*.json --out runs/contamination.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.conversation import eot_outcomes, silence_scores  # noqa: E402
from audioforge.data import Collate, ToneLanguage  # noqa: E402
from audioforge.heads.turn import decoded_text_state  # noqa: E402
from audioforge.train import load_model  # noqa: E402

FRAME = 0.08
BIN = 2  # 160 ms persistence bins (turn head chunk)
N_BINS = 25  # 4 s
THETA, K_FRAMES = 0.998164, 53  # frozen hybrid operating point (research/EOT_BENCH_V2.md section 7)
LEFT_CTX = 70  # encoder left context in frames (att_context_size [70, 1])
TURN_WINDOWS = {"0.5s": 6, "1s": 12, "2s": 25}
SAASR_WINDOWS = {"160ms": 2, "320ms": 4, "480ms": 6}  # the synthetic utterances are short (median target span 0.7 s)
SAASR_BUFFERS = {"0.5s": 6, "1s": 12, "2s": 25}  # bounded replay buffers (the mixtures are about 2 s long)
REPLAY_BUFFERS = {"1s": 12, "2s": 25, "4s": 50, "6s": 75, "10s": 125, "15s": 188}  # bounded replay (turn head)
DELAYS = {"0s": 0, "0.5s": 6, "1s": 12, "2s": 25, "4s": 50}  # correction delay after the window end
MODES = ("zero", "swap", "rand")
CONTROLS = ("noop", "eps")


def _ev():
    spec = importlib.util.spec_from_file_location("eval_stage1", ROOT / "scripts" / "research" / "eval_stage1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wait_for_load(max_load: float = 10.0):
    import subprocess
    while True:
        try:
            l1 = float(subprocess.check_output(["sysctl", "-n", "vm.loadavg"]).decode().split()[1])
        except Exception:
            return
        if l1 <= max_load:
            return
        print(f"  load {l1:.1f} > {max_load}: waiting", flush=True)
        time.sleep(30)


# --------------------------------------------------------------------------- corruption
def corrupt_track(act: np.ndarray, cols: np.ndarray | None, lo: int, hi: int, mode: str, rng: random.Random,
                  other: np.ndarray | None = None):
    """Corrupt the bound track on frames [lo, hi). act (T,) = the bound column; cols (T, S) with the bound column
    first (turn head v3, bound_track) or None (saasr: ``other`` (T,) is the other speaker's activity).
    zero: the bound speaker reads silent; swap: the bound column shows the most active OTHER column of the window
    (the enrollment points at the wrong speaker); rand: swap with a random other column (control); eps: +-1e-4 noise
    (numerical-noise bound); noop: unchanged (determinism check). -> (act_x, cols_x, info)."""
    act_x = np.array(act, np.float32, copy=True)
    cols_x = None if cols is None else np.array(cols, np.float32, copy=True)
    info = {"mode": mode, "lo": int(lo), "hi": int(hi), "partner": None}
    if mode == "noop" or hi <= lo:
        pass
    elif mode == "eps":  # towards the interior of [0, 1], so a saturated track (exactly 0 / 1) is perturbed too
        noise = np.array([rng.uniform(1e-5, 1e-4) for _ in range(hi - lo)], np.float32)
        act_x[lo:hi] = np.where(act_x[lo:hi] > 0.5, act_x[lo:hi] - noise, act_x[lo:hi] + noise)
        if cols_x is not None:
            cols_x[lo:hi, 0] = act_x[lo:hi]
    elif mode == "zero":
        act_x[lo:hi] = 0.0
        if cols_x is not None:
            cols_x[lo:hi, 0] = 0.0
    elif mode in ("swap", "rand"):
        if cols_x is not None:
            S = cols_x.shape[1]
            j = (1 + int(np.argmax(cols[lo:hi, 1:].mean(0)))) if mode == "swap" else rng.randrange(1, S)
            a, b = cols[lo:hi, 0].copy(), cols[lo:hi, j].copy()
            cols_x[lo:hi, 0], cols_x[lo:hi, j] = b, a
            act_x[lo:hi] = b
            info["partner"] = j
        else:
            act_x[lo:hi] = other[lo:hi]
            info["partner"] = 1
    else:
        raise ValueError(mode)
    info["input_delta"] = float(np.abs(act_x[lo:hi] - act[lo:hi]).mean()) if hi > lo else 0.0
    # a swap whose partner is silent on the window is the zero corruption; flag the ones that differ from it
    info["differs_from_zero"] = bool(hi > lo and mode in ("swap", "rand") and float(np.abs(act_x[lo:hi]).mean()) > 0.1)
    return act_x, cols_x, info


def bin_curve(delta: np.ndarray, t0: int, n_bins: int = N_BINS, size: int = BIN) -> np.ndarray:
    """Mean of ``delta`` (T,) over the bins [t0 + b*size, t0 + (b+1)*size), NaN where the track has no frames."""
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        seg = delta[t0 + b * size: t0 + (b + 1) * size]
        if len(seg):
            out[b] = float(np.mean(seg))
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


# --------------------------------------------------------------------------- turn head: segmented head runs
@torch.no_grad()
def head_run(head, enc, tf, act, cols, prim, h0=None, dur0=None):
    """The TurnHead v3 forward on a block of frames with an explicit initial state: enc (B,n,D), tf (B,n,text_dim)
    text features, act (B,n) kernel track, cols (B,n,S), prim (B,). h0 (1,B,H) GRU state, dur0 counters (or None =
    fresh). -> p (B,n), h (B,n,H), dur state after the block. A run split at any frame and chained through these
    states equals the unsplit run (the streaming ``step`` uses the same pieces)."""
    v3, dur = head.v3_features(enc.shape[1], act, cols, prim, dur0)
    x = head._fuse(enc, None, tf, None, v3=v3)
    h, _ = head.rnn(x, h0)
    return head.out(h).squeeze(-1).sigmoid(), h, dur


def outcome(p: np.ndarray, act: np.ndarray, onset: int, end: int, ev, theta=THETA, k=K_FRAMES, horizon=75):
    """(head, hybrid) outcomes of one turn at the frozen point: each {'fc': bool, 'lat_ms': float (inf = miss)}.
    Head: fires at p > theta, emitted at its 160 ms chunk. Timeout: the bound track silent >= k frames, armed at its
    first active frame, emitted under the stream rule (C = 6, R = 7). Post-end horizon 75 emission frames (6 s)."""
    hp = ev._emit_transform([p], [end], lambda t: (t // BIN + 1) * BIN)
    oh = eot_outcomes(hp, [onset], [end], [theta], chunk=1, post_end_frames=horizon)
    sil = silence_scores(act, ev.arm_frame(act))
    sp = ev._emit_transform([sil], [end], ev._stream_emit(6, 7))
    ot = eot_outcomes(sp, [onset], [end], [k - 0.5], chunk=1, post_end_frames=horizon)
    head = {"fc": bool(oh["fc"][0, 0]), "lat_ms": float(oh["lat"][0, 0])}
    hyb = {"fc": bool(oh["fc"][0, 0] or ot["fc"][0, 0]), "lat_ms": float(min(oh["lat"][0, 0], ot["lat"][0, 0]))}
    return head, hyb


def _cls(o):
    return "fc" if o["fc"] else ("miss" if np.isinf(o["lat_ms"]) else "fire")


def turn_conditions(windows=TURN_WINDOWS):
    conds = [("base", None, 0)]
    conds += [(m, w, n) for m in MODES for w, n in windows.items()]
    conds += [("noop", "1s", windows["1s"]), ("eps", "1s", windows["1s"])]
    return conds


def eligible_turns(ext, d_end: int, w_max: int, margin: int = 6, post: int = N_BINS * BIN):
    """Turns where t0 = end - d_end leaves the largest corruption window inside the turn (>= margin frames after
    the onset) and the 4 s persistence window inside the track."""
    out = []
    for i, v in enumerate(ext):
        on, en, T = int(v["onset_frame"]), int(v["turn_end_frame"]), len(v["spk_act"])
        t0 = en - d_end
        if t0 - w_max >= on + margin and t0 + post <= T:
            out.append(i)
    return out


@torch.no_grad()
def probe_turn(model, ex: dict, track: np.ndarray, ev, t0: int, rng: random.Random, conds, windows=TURN_WINDOWS,
               reprime_conds=(("zero", "1s"), ("swap", "1s"))) -> dict:
    """One turn: every condition x protocol -> per-bin |dp|, |dlogit|, decision flips, outcomes."""
    head = model.heads["turn"]
    dev = next(model.parameters()).device
    b = Collate(model.tokenizer)([ex])
    audio, alen = b["audio"].to(dev), b["audio_len"].to(dev)
    T = len(ex["spk_act"])
    act, cols, prim, _ = ev._v2_cols(ex, track, "causal_dominant")
    on, en = int(ex["onset_frame"]), int(ex["turn_end_frame"])
    # conditions -> tracks
    acts, colss, infos = [], [], []
    for mode, wname, w in conds:
        if mode == "base":
            a, c, info = act.copy(), cols.copy(), {"mode": "base", "lo": t0, "hi": t0, "input_delta": 0.0}
        else:
            a, c, info = corrupt_track(act, cols, t0 - w, t0, mode, rng)
        acts.append(a)
        colss.append(c)
        infos.append(info)
    B = len(conds)
    A = torch.as_tensor(np.stack(acts), device=dev)
    Cc = torch.as_tensor(np.stack(colss), device=dev)
    P = torch.zeros(B, dtype=torch.long, device=dev)
    # unconditioned encode: text state (the RNNT head is not speaker-conditioned; checked by the caller)
    enc0, elen, hidden = model.encode(audio, alen, return_hidden=True)
    y, n = decoded_text_state(model, head, audio, alen, enc0, hidden, elen, A[:1, : enc0.shape[1]])
    tf = head.text_frames(y, n)[:, :T].expand(B, -1, -1)
    # conditioned encodes, all conditions in one batch (same audio, same length: no padding)
    enc, _ = model.encode(audio.expand(B, -1), alen.expand(B), spk_act=torch.nn.functional.pad(A, (0, 1)))
    enc = enc[:, :T]
    p_full, h_full, _ = head_run(head, enc, tf, A, Cc, P)
    h0 = h_full[:, t0 - 1][None].contiguous()  # (1,B,H) GRU state after frame t0-1
    _, dur0 = head.v3_features(t0, A[:, :t0], Cc[:, :t0], P, None)
    seg = dict(tf=tf[:, t0:], act=A[:, t0:], cols=Cc[:, t0:], prim=P)
    enc_base_rest = enc[:1, t0:].expand(B, -1, -1)
    base_h0 = h0[:, :1].expand(-1, B, -1).contiguous()
    base_dur0 = {k: (s[:1].expand(B), r[:1].expand(B)) for k, (s, r) in dur0.items()}
    runs = {"full": p_full[:, t0:]}
    runs["head_reset"] = head_run(head, enc[:, t0:], h0=None, dur0=None, **seg)[0]
    runs["gru_reset"] = head_run(head, enc[:, t0:], h0=None, dur0=dur0, **seg)[0]
    runs["dur_reset"] = head_run(head, enc[:, t0:], h0=h0, dur0=None, **seg)[0]
    runs["enc_sub"] = head_run(head, enc_base_rest, h0=h0, dur0=dur0, **seg)[0]
    runs["both"] = head_run(head, enc_base_rest, h0=None, dur0=None, **seg)[0]
    # deployable re-prime: encoder re-run from t0 - LEFT_CTX with the CORRECT track (the corrupted window lies inside
    # the re-primed span, so all conditions get the same encoder frames), recurrent state kept / reset
    s0 = max(0, t0 - LEFT_CTX)
    smp = s0 * 1280
    a_rp = torch.as_tensor(act[s0:], device=dev)[None]
    enc_rp, _ = model.encode(audio[:, smp:], alen - smp, spk_act=torch.nn.functional.pad(a_rp, (0, 1)))
    enc_rp = enc_rp[:, t0 - s0: T - s0]
    if enc_rp.shape[1] < T - t0:  # the cropped signal can yield one frame less (subsampling rounding): repeat the last
        enc_rp = torch.cat([enc_rp, enc_rp[:, -1:].expand(-1, T - t0 - enc_rp.shape[1], -1)], 1)
    enc_rp = enc_rp.expand(B, -1, -1)
    runs["reprime"] = head_run(head, enc_rp, h0=h0, dur0=dur0, **seg)[0]
    # baseline-state substitution into the corrupted run (the mirror of enc_sub: corrupted encoder, clean state),
    # and its split into the GRU state and the duration counters
    runs["state_sub"] = head_run(head, enc[:, t0:], h0=base_h0, dur0=base_dur0, **seg)[0]
    runs["gru_sub"] = head_run(head, enc[:, t0:], h0=base_h0, dur0=dur0, **seg)[0]
    runs["dur_sub"] = head_run(head, enc[:, t0:], h0=h0, dur0=base_dur0, **seg)[0]
    # re-prime from twice the left context
    s1 = max(0, t0 - 2 * LEFT_CTX)
    a_rp2 = torch.as_tensor(act[s1:], device=dev)[None]
    enc_rp2, _ = model.encode(audio[:, s1 * 1280:], alen - s1 * 1280, spk_act=torch.nn.functional.pad(a_rp2, (0, 1)))
    enc_rp2 = enc_rp2[:, t0 - s1: T - s1]
    if enc_rp2.shape[1] < T - t0:
        enc_rp2 = torch.cat([enc_rp2, enc_rp2[:, -1:].expand(-1, T - t0 - enc_rp2.shape[1], -1)], 1)
    runs["reprime2x"] = head_run(head, enc_rp2.expand(B, -1, -1), h0=h0, dur0=dur0, **seg)[0]
    # deployable head replay: on correction, restore the head state snapshot from the (chunk-aligned) window start
    # and re-run the head over the window with the CORRECTED track / columns on the encoder frames as they are
    # (still computed under the wrong binding); no encoder recompute. Frames before the aligned window start are
    # identical to the baseline (causality), so the snapshot equals the baseline's state there.
    rep = []
    for j, (mode, wname, w) in enumerate(conds):
        lo = t0 - (w or 0)
        lo -= lo % BIN
        if lo <= 0:
            lo = BIN
        hj = h_full[j:j + 1, lo - 1][None].contiguous()
        _, dj = head.v3_features(lo, A[j:j + 1, :lo], Cc[j:j + 1, :lo], P[:1], None)
        q, _, _ = head_run(head, enc[j:j + 1, lo:], tf[:1, lo:], A[:1, lo:], Cc[:1, lo:], P[:1], h0=hj, dur0=dj)
        rep.append(q[0, t0 - lo:])
    runs["head_replay"] = torch.stack(rep)
    # deployable repair with an encoder recompute: re-prime the encoder from 2 x left context before the window with
    # the corrected track, then replay the head from the window start on those frames (same state snapshot)
    rep = []
    for j, (mode, wname, w) in enumerate(conds):
        lo = t0 - (w or 0)
        lo -= lo % BIN
        if lo <= 0:
            lo = BIN
        s2 = max(0, lo - 2 * LEFT_CTX)
        a2 = torch.as_tensor(act[s2:], device=dev)[None]
        e2, _ = model.encode(audio[:, s2 * 1280:], alen - s2 * 1280, spk_act=torch.nn.functional.pad(a2, (0, 1)))
        e2 = e2[:, lo - s2: T - s2]
        if e2.shape[1] < T - lo:
            e2 = torch.cat([e2, e2[:, -1:].expand(-1, T - lo - e2.shape[1], -1)], 1)
        hj = h_full[j:j + 1, lo - 1][None].contiguous()
        _, dj = head.v3_features(lo, A[j:j + 1, :lo], Cc[j:j + 1, :lo], P[:1], None)
        q, _, _ = head_run(head, e2, tf[:1, lo:], A[:1, lo:], Cc[:1, lo:], P[:1], h0=hj, dur0=dj)
        rep.append(q[0, t0 - lo:])
    runs["reprime2x_replay"] = torch.stack(rep)
    out = {"t0": t0, "onset": on, "end": en, "T": T, "conds": [], "input_delta": [i["input_delta"] for i in infos],
           "partner": [i.get("partner") for i in infos], "differs_from_zero": [bool(i.get("differs_from_zero")) for i in infos]}
    pf = p_full.cpu().numpy()
    base_full = pf[0]
    pre = {}
    for j, (mode, wname, w) in enumerate(conds):
        lo = t0 - (w or 0)
        lo -= lo % BIN  # the encoder's attention chunk is 2 frames (right context 1): frame lo - 1 may see frame lo
        pre[j] = float(np.abs(pf[j, :lo] - base_full[:lo]).max()) if lo > 0 else 0.0
    out["pre_window_max"] = [pre[j] for j in range(B)]
    out["in_window_max"] = [float(np.abs(pf[j, t0 - (w or 0): t0] - base_full[t0 - (w or 0): t0]).max())
                            if w else 0.0 for j, (m, wn, w) in enumerate(conds)]
    res = {}
    for prot, q in runs.items():
        q = q.cpu().numpy()  # (B, T - t0)
        ref = q[0]  # the baseline under the same protocol
        for j in range(B):
            d = np.abs(q[j] - ref)
            dl = np.abs(logit(q[j]) - logit(ref))
            fl = ((q[j] > THETA) != (ref > THETA)).astype(np.float64)
            full_p = np.concatenate([pf[j, :t0], q[j]])
            oh, ohy = outcome(full_p, acts[j], on, en, ev)
            res[(prot, j)] = {"dp": bin_curve(d, 0), "dlogit": bin_curve(dl, 0), "flip": bin_curve(fl, 0),
                              "dp_max_4s": float(d[: N_BINS * BIN].max()), "dp_tail": float(d[N_BINS * BIN:].max())
                              if len(d) > N_BINS * BIN else 0.0, "head": oh, "hybrid": ohy}
        # protocol cost: the baseline under this protocol vs the plain baseline
        res[(prot, "cost")] = {"dp": bin_curve(np.abs(ref - base_full[t0:]), 0),
                               "dlogit": bin_curve(np.abs(logit(ref) - logit(base_full[t0:])), 0)}
    out["res"] = res
    return out


def _ci(x: np.ndarray, n_boot: int, rng) -> tuple:
    """Percentile bootstrap (over rows) of the column means; NaNs ignored. -> (mean, lo, hi) each (bins,)."""
    x = np.asarray(x, np.float64)
    m = np.nanmean(x, 0)
    if n_boot <= 0 or len(x) < 2:
        return m, m, m
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    bs = np.stack([np.nanmean(x[i], 0) for i in idx])
    return m, np.nanquantile(bs, 0.025, 0), np.nanquantile(bs, 0.975, 0)


def summarize_turn(records: list[dict], conds, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    prots = sorted({k[0] for r in records for k in r["res"]})
    B = len(conds)
    out = {"n": len(records), "conditions": [{"mode": m, "window": w, "frames": n} for m, w, n in conds],
           "protocols": prots, "bins_ms": [int(b * BIN * FRAME * 1000) for b in range(N_BINS)], "curves": {},
           "outcomes": {}, "cost": {}, "checks": {}}
    inp = np.array([r["input_delta"] for r in records])  # (n, B)
    out["checks"]["pre_window_max_dp"] = float(np.max([r["pre_window_max"] for r in records]))
    out["checks"]["in_window_max_dp_by_cond"] = [float(np.mean([r["in_window_max"][j] for r in records])) for j in range(B)]
    out["input_delta_mean_by_cond"] = inp.mean(0).tolist()
    eff = inp > 0.25  # the corruption actually changed the input on the window
    out["n_effective_by_cond"] = eff.sum(0).tolist()
    dfz = np.array([r["differs_from_zero"] for r in records])  # swap / rand whose partner column spoke on the window
    out["n_swap_differs_from_zero_by_cond"] = dfz.sum(0).tolist()
    for prot in prots:
        for j, (mode, wname, w) in enumerate(conds):
            if j == 0:
                continue
            key = f"{prot}/{mode}/{wname}"
            dp = np.stack([r["res"][(prot, j)]["dp"] for r in records])
            dl = np.stack([r["res"][(prot, j)]["dlogit"] for r in records])
            fl = np.stack([r["res"][(prot, j)]["flip"] for r in records])
            m, lo, hi = _ci(dp, n_boot, rng)
            me, loe, hie = _ci(dp[eff[:, j]], n_boot, rng) if eff[:, j].sum() >= 2 else (m * np.nan,) * 3
            aff = np.nanmean(dp > 0.01, 0)
            mz, loz, hiz = _ci(dp[dfz[:, j]], n_boot, rng) if dfz[:, j].sum() >= 2 else (m * np.nan,) * 3
            out["curves"][key] = {
                "dp_mean_partner_active": mz.round(6).tolist(), "dp_lo_partner_active": loz.round(6).tolist(),
                "dp_hi_partner_active": hiz.round(6).tolist(), "n_partner_active": int(dfz[:, j].sum()),
                "dp_mean": m.round(6).tolist(), "dp_lo": lo.round(6).tolist(), "dp_hi": hi.round(6).tolist(),
                "dp_mean_effective": me.round(6).tolist(), "dp_lo_effective": loe.round(6).tolist(),
                "dp_hi_effective": hie.round(6).tolist(),
                "dlogit_mean": np.nanmean(dl, 0).round(4).tolist(), "flip_rate": np.nanmean(fl, 0).round(5).tolist(),
                "affected_frac_gt_0.01": aff.round(4).tolist(),
                "dp_max_4s_mean": float(np.mean([r["res"][(prot, j)]["dp_max_4s"] for r in records])),
                "dp_max_4s_p90": float(np.quantile([r["res"][(prot, j)]["dp_max_4s"] for r in records], 0.9)),
                "dp_tail_gt4s_max": float(np.max([r["res"][(prot, j)]["dp_tail"] for r in records])),
                "turns_dp_gt_0.01_after_1s": float(np.mean(np.nanmax(dp[:, 6:], 1) > 0.01)),
                "turns_dp_gt_0.01_after_2s": float(np.mean(np.nanmax(dp[:, 12:], 1) > 0.01)),
            }
            # outcomes at the frozen point vs the baseline under the same protocol
            oc = {}
            for det in ("head", "hybrid"):
                base = [r["res"][(prot, 0)][det] for r in records]
                cor = [r["res"][(prot, j)][det] for r in records]
                changed = [(_cls(a) != _cls(b)) or (a["lat_ms"] != b["lat_ms"]) for a, b in zip(base, cor)]
                cls_changed = [_cls(a) != _cls(b) for a, b in zip(base, cor)]
                oc[det] = {"changed_frac": float(np.mean(changed)), "class_changed_frac": float(np.mean(cls_changed)),
                           "n_changed": int(np.sum(changed)),
                           "base": {c: int(sum(_cls(a) == c for a in base)) for c in ("fire", "miss", "fc")},
                           "corrupted": {c: int(sum(_cls(a) == c for a in cor)) for c in ("fire", "miss", "fc")},
                           "transitions": {}}
                for a, b in zip(base, cor):
                    if _cls(a) != _cls(b):
                        k = f"{_cls(a)}->{_cls(b)}"
                        oc[det]["transitions"][k] = oc[det]["transitions"].get(k, 0) + 1
            out["outcomes"][key] = oc
        cost = np.stack([r["res"][(prot, "cost")]["dp"] for r in records])
        m, lo, hi = _ci(cost, n_boot, rng)
        out["cost"][prot] = {"dp_mean": m.round(6).tolist(), "dp_lo": lo.round(6).tolist(), "dp_hi": hi.round(6).tolist(),
                             "dlogit_mean": np.nanmean(np.stack([r["res"][(prot, "cost")]["dlogit"] for r in records]), 0).round(4).tolist()}
    return out


def run_turn(a):
    torch.set_num_threads(a.threads)
    ev = _ev()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, _ = ev.v2_data(6.0)
    model = load_model(a.ckpt, "cpu")
    assert not model.head_cfg[model.heads["turn"].text_head].get("condition_on_speaker"), \
        "the text state would depend on the track: not handled"
    conds = turn_conditions()
    d_end = int(round(a.t0_before_end / FRAME))
    idx = eligible_turns(ext, d_end, max(TURN_WINDOWS.values()))
    idx = [i for i in idx if ev.v2_track_path(a.work, ext[i]).exists()]
    print(f"[turn] eligible {len(idx)} of {len(ext)} (t0 = end - {d_end} frames); using {min(a.n, len(idx))}", flush=True)
    idx = idx[: a.n]
    pk = out_dir / "turn_records.pkl"
    records = pickle.load(open(pk, "rb")) if pk.exists() else {}
    t_start = time.time()
    for c, i in enumerate(idx):
        if i in records:
            continue
        wait_for_load()
        ex = ext[i]
        track = np.load(ev.v2_track_path(a.work, ex))
        t0 = int(ex["turn_end_frame"]) - d_end
        records[i] = probe_turn(model, ex, track, ev, t0, random.Random(a.seed * 100003 + i), conds)
        records[i]["idx"] = i
        records[i]["meeting"] = ex["meeting"]
        if (c + 1) % 25 == 0 or c + 1 == len(idx):
            pickle.dump(records, open(pk, "wb"))
            print(f"  {c + 1}/{len(idx)} turns, {time.time() - t_start:.0f}s", flush=True)
        if c + 1 == a.early_check and a.early_check:
            s = summarize_turn([records[k] for k in idx if k in records], conds, n_boot=200)
            noise = max(np.nanmax(s["curves"]["full/eps/1s"]["dp_mean"]), np.nanmax(s["curves"]["full/noop/1s"]["dp_mean"]))
            eff = max(np.nanmax(s["curves"][f"full/{m}/2s"]["dp_lo"][1:]) for m in ("zero", "swap"))
            print(f"  early check after {c + 1} turns: noise {noise:.2e}, min effect CI lower bound (bins >= 1) {eff:.2e}",
                  flush=True)
            if eff <= 2 * noise:
                print("  effect indistinguishable from noise after the first bin: stopping early", flush=True)
                break
        if a.budget and time.time() - t_start > a.budget:
            print("  budget reached", flush=True)
            break
    recs = [records[k] for k in idx if k in records]
    pickle.dump(records, open(pk, "wb"))
    s = summarize_turn(recs, conds, n_boot=a.n_boot, seed=a.seed)
    s["setup"] = {"ckpt": a.ckpt, "work": str(a.work), "binding": "causal_dominant", "t0_before_end_frames": d_end,
                  "theta": THETA, "k_frames": K_FRAMES, "left_ctx_frames": LEFT_CTX, "windows_frames": TURN_WINDOWS,
                  "n_eligible": len(idx), "meetings": sorted({r["meeting"] for r in recs})}
    json.dump(s, open(out_dir / "turn.json", "w"), indent=1)
    print(f"[turn] wrote {out_dir / 'turn.json'} ({len(recs)} turns)", flush=True)
    return s


# --------------------------------------------------------------------------- speaker-attributed ASR
def sa_examples(n: int, seed: int = 1) -> list[dict]:
    """The recipe's held-out synthetic mixtures (data.synthetic_dataset('speaker_attributed', seed=1) draws the same
    mixtures; the first 64 are the recipe's val set) with BOTH speakers' texts kept."""
    lang = ToneLanguage(seed=0)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        m = lang.mixture(rng)
        rng.randrange(len(m["speakers"]))  # keep the rng in step with synthetic_dataset
        out.append(m)
    return out


@torch.no_grad()
def greedy_segmented(asr, f: torch.Tensor, t0: int | None = None, reset: bool = False, f_after: torch.Tensor | None = None):
    """heads.turn.greedy_decode_frames with a state hook at frame t0: ``reset`` restarts the prediction net (blank
    prefix) when the decoder first reaches frame t0; ``f_after`` replaces the encoder frames >= t0. Without both it
    is exactly greedy_decode_frames. -> (tokens, frames)."""
    V1, dev = asr.vocab_size + 1, f.device
    hyp, frames = [], []
    g, state = asr.pred(torch.tensor([[asr.blank]], device=dev), None)
    fe = asr.joint.enc(f)
    if f_after is not None and t0 is not None:
        fe = torch.cat([fe[:t0], asr.joint.enc(f_after)[t0:]], 0)
    t, T, emitted, done = 0, f.shape[0], 0, False
    while t < T:
        if t0 is not None and t >= t0 and not done:
            done = True
            if reset:
                g, state = asr.pred(torch.tensor([[asr.blank]], device=dev), None)
        z = asr.joint.out(fe[t][None, None] + asr.joint.pred(g))[0, 0]
        k = int(z[:V1].argmax())
        if asr.is_tdt:
            d = asr.durations[int(z[V1:].argmax())]
            if k == asr.blank and d == 0:
                d = 1
        else:
            d = 1 if k == asr.blank else 0
        if k != asr.blank:
            hyp.append(k)
            frames.append(t)
            g, state = asr.pred(torch.tensor([[k]], device=dev), state)
            emitted += 1
        if d == 0 and emitted >= asr.max_symbols:
            d = 1
        if d > 0:
            emitted = 0
        t += d
    return hyp, frames


def edit_distance(a, b) -> int:
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(hyp: str, ref: str) -> float:
    r = ref.split()
    return edit_distance(hyp.split(), r) / max(1, len(r))


def flips(hyp: str, ref: str, other: str) -> int:
    """Words of the hypothesis that belong to the OTHER speaker's text and not to the target's (attribution flips)."""
    rs, os_ = set(ref.split()), set(other.split())
    return sum(1 for w in hyp.split() if w in os_ and w not in rs)


def token_bins(tokens, frames, t0: int, n_bins: int, size: int = BIN):
    out = [[] for _ in range(n_bins)]
    for k, t in zip(tokens, frames):
        b = (t - t0) // size
        if 0 <= b < n_bins:
            out[b].append(k)
    return out


def saasr_conditions(windows=SAASR_WINDOWS):
    conds = [("base", None, 0)]
    conds += [(m, w, n) for m in MODES for w, n in windows.items()]
    conds += [("noop", list(windows)[1], list(windows.values())[1])]
    return conds


@torch.no_grad()
def probe_saasr(model, ex: dict, spk: int, t0: int, rng: random.Random, conds, n_bins: int = 13) -> dict:
    asr = model.heads["asr"]
    dev = next(model.parameters()).device
    audio = torch.as_tensor(ex["audio"], device=dev)[None]
    alen = torch.tensor([audio.shape[1]], device=dev)
    tg = ex["spk_targets"]
    act, other = tg[:, spk], tg[:, 1 - spk]
    ref, oth_text = ex["speakers"][spk]["text"], ex["speakers"][1 - spk]["text"]
    acts, infos = [], []
    for mode, wname, w in conds:
        if mode == "base":
            a, info = act.copy(), {"input_delta": 0.0}
        else:
            a, _, info = corrupt_track(act, None, t0 - w, t0, mode, rng, other=other)
        acts.append(a)
        infos.append(info)
    B = len(conds)
    A = torch.as_tensor(np.stack(acts), device=dev)
    enc, elen = model.encode(audio.expand(B, -1), alen.expand(B), spk_act=A)
    n = int(elen[0])
    res = {}
    dec = {}
    # bounded replay at t0: encoder re-run from t0 - L (empty caches) with the CORRECT track, decoder continues from
    # its (corrupted-run) prefix at t0 - L on the recomputed frames
    rep_enc = {}
    for name, L in SAASR_BUFFERS.items():
        s = max(0, t0 - L)
        a = torch.as_tensor(act[s:], device=dev)[None]
        e, _ = model.encode(audio[:, s * 1280:], alen - s * 1280, spk_act=a)
        F_ = enc[0, :n].clone()
        m_ = min(e.shape[1], n - s)
        F_[s: s + m_] = e[0, :m_]
        rep_enc[name] = (s, F_)
    for j in range(B):
        f = enc[j, :n]
        dec[("full", j)] = greedy_segmented(asr, f)
        dec[("dec_reset", j)] = greedy_segmented(asr, f, t0, reset=True)
        dec[("enc_sub", j)] = greedy_segmented(asr, f, t0, reset=False, f_after=enc[0, :n])
        dec[("both", j)] = greedy_segmented(asr, f, t0, reset=True, f_after=enc[0, :n])
        for name, (s, F_) in rep_enc.items():
            dec[(f"replay_{name}", j)] = greedy_segmented(asr, f, s, reset=False, f_after=F_)
    for (prot, j), (toks, frs) in dec.items():
        rt, rf = dec[("full", 0)] if prot.startswith("replay_") else dec[(prot, 0)]
        tb, rb = token_bins(toks, frs, t0, n_bins), token_bins(rt, rf, t0, n_bins)
        avail = [(t0 + b * BIN) < n for b in range(n_bins)]
        change = [(edit_distance(x, y) / max(1, len(x), len(y)) if (x or y) else 0.0) if av else np.nan
                  for x, y, av in zip(tb, rb, avail)]
        differs = [float(x != y) if av else np.nan for x, y, av in zip(tb, rb, avail)]
        hyp = model.tokenizer.decode(toks)
        lo = t0 - (conds[j][2] or 0)
        pre_tokens = [k for k, t in zip(toks, frs) if t < lo]
        pre_ref = [k for k, t in zip(rt, rf) if t < lo]
        res[(prot, j)] = {"change": change, "differs": differs, "wer": wer(hyp, ref), "flips": flips(hyp, ref, oth_text),
                          "hyp_changed": float(toks != rt), "pre_window_changed": float(pre_tokens != pre_ref),
                          "n_ref_words": len(ref.split()), "hyp": hyp}
    return {"t0": t0, "T": n, "spk": spk, "ref": ref, "res": res, "input_delta": [i["input_delta"] for i in infos]}


def summarize_saasr(records, conds, n_boot=1000, seed=0, n_bins=13) -> dict:
    rng = np.random.default_rng(seed)
    prots = ["full", "dec_reset", "enc_sub", "both"] + [f"replay_{k}" for k in SAASR_BUFFERS]
    out = {"n": len(records), "conditions": [{"mode": m, "window": w, "frames": n} for m, w, n in conds],
           "protocols": prots, "bins_ms": [int(b * BIN * FRAME * 1000) for b in range(n_bins)], "curves": {},
           "utterance": {}, "checks": {}}
    inp = np.array([r["input_delta"] for r in records])
    out["input_delta_mean_by_cond"] = inp.mean(0).tolist()
    eff = inp > 0.25
    out["n_effective_by_cond"] = eff.sum(0).tolist()
    base_wer = np.array([r["res"][("full", 0)]["wer"] for r in records])
    out["baseline_wer"] = float(base_wer.mean())
    out["checks"]["pre_window_changed_frac_by_cond"] = [float(np.mean([r["res"][("full", j)]["pre_window_changed"]
                                                                       for r in records])) for j in range(len(conds))]
    for prot in prots:
        for j, (mode, wname, w) in enumerate(conds):
            if j == 0 and not prot.startswith("replay_"):
                continue
            key = f"{prot}/{mode}/{wname}"
            ch = np.array([r["res"][(prot, j)]["change"] for r in records], np.float64)
            df = np.array([r["res"][(prot, j)]["differs"] for r in records], np.float64)
            m, lo, hi = _ci(ch, n_boot, rng)
            out["curves"][key] = {"change_mean": np.round(m, 4).tolist(), "change_lo": np.round(lo, 4).tolist(),
                                  "change_hi": np.round(hi, 4).tolist(), "differs_frac": np.round(np.nanmean(df, 0), 4).tolist(),
                                  "n_avail": np.sum(~np.isnan(ch), 0).tolist()}
            w_c = np.array([r["res"][(prot, j)]["wer"] for r in records])
            w_b = np.array([r["res"][(prot, 0)]["wer"] for r in records])
            nw = np.array([r["res"][(prot, j)]["n_ref_words"] for r in records])
            fl_c = np.array([r["res"][(prot, j)]["flips"] for r in records])
            fl_b = np.array([r["res"][(prot, 0)]["flips"] for r in records])
            out["utterance"][key] = {"wer": float(w_c.mean()), "wer_base": float(w_b.mean()),
                                     "wer_pooled": float((w_c * nw).sum() / nw.sum()), "wer_pooled_base": float((w_b * nw).sum() / nw.sum()),
                                     "hyp_changed_frac": float(np.mean([r["res"][(prot, j)]["hyp_changed"] for r in records])),
                                     "flips_per_utt": float(fl_c.mean()), "flips_per_utt_base": float(fl_b.mean()),
                                     "utts_with_new_flip": float(np.mean(fl_c > fl_b))}
    return out


@torch.no_grad()
def saasr_timing(model, audio: np.ndarray, buffers=REPLAY_BUFFERS, reps: int = 5) -> dict:
    """Bounded replay wall-clock for the SA-ASR model: encoder over the last L s + greedy RNNT over those frames, vs one
    160 ms step (stream_step on a 16-mel chunk with primed caches + greedy over its 2 frames; causal models only)."""
    asr = model.heads["asr"]
    dev = next(model.parameters()).device
    x = torch.as_tensor(audio, device=dev)[None]
    out = {}
    for name, L in buffers.items():
        seg = x[:, : L * 1280]
        n = torch.tensor([seg.shape[1]], device=dev)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            e, _ = model.encode(seg, n, spk_act=torch.ones(1, L + 1, device=dev))
            greedy_segmented(asr, e[0])
            ts.append(time.perf_counter() - t)
        out[name] = {"frames": L, "ms": 1000 * float(np.median(ts))}
    out["chunk_ms"] = BIN * FRAME * 1000
    out["chunk_step_ms"] = None
    if model.encoder.causal:
        import copy
        from audioforge.modules.fastconformer import StreamState
        feats, flen = model.features(x[:, : 90 * 1280], torch.tensor([90 * 1280], device=dev))
        st, cm, pos = StreamState(), model.encoder.stream_chunk_frames(), 0
        while pos + cm <= 70 * 8:
            _, st = model.encoder.stream_step(feats[..., pos: pos + cm], st, spk_act=torch.ones(1, 2, device=dev))
            pos += cm
        ts = []
        for _ in range(20):
            st2 = copy.deepcopy(st)
            t = time.perf_counter()
            e, st2 = model.encoder.stream_step(feats[..., pos: pos + cm], st2, spk_act=torch.ones(1, 2, device=dev))
            greedy_segmented(asr, e[0])
            ts.append(time.perf_counter() - t)
        out["chunk_step_ms"] = 1000 * float(np.median(ts))
    for name in buffers:
        out[name]["realtime_chunks"] = out[name]["ms"] / out["chunk_ms"]
        out[name]["x_chunk_step"] = None if out["chunk_step_ms"] is None else out[name]["ms"] / out["chunk_step_ms"]
    return out


def run_saasr(a):
    torch.set_num_threads(a.threads)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(a.ckpt, "cpu")
    exs = sa_examples(a.n_eval)
    long_audio = np.concatenate([e["audio"] for e in exs[:12]])[: 16 * 16000]
    timing = saasr_timing(model, long_audio)
    print("[saasr] timing: " + ", ".join(f"{k} {v['ms']:.0f} ms" for k, v in timing.items() if isinstance(v, dict))
          + f"; chunk step {timing['chunk_step_ms']}", flush=True)
    conds = saasr_conditions()
    w_max = max(SAASR_WINDOWS.values())
    cases = []
    for i, ex in enumerate(exs):
        for spk in range(len(ex["speakers"])):
            nz = np.nonzero(ex["spk_targets"][:, spk] > 0.5)[0]
            if not len(nz):
                continue
            a0, a1 = int(nz[0]), int(nz[-1])
            t0 = a0 + int(round(0.6 * (a1 - a0 + 1)))
            if t0 - w_max >= a0 + 1 and t0 < a1:
                cases.append((i, spk, t0))
    print(f"[saasr] {len(cases)} cases (utterance x target speaker) from {len(exs)} mixtures; causal={model.encoder.causal}", flush=True)
    records = []
    t_start = time.time()
    for c, (i, spk, t0) in enumerate(cases):
        wait_for_load()
        records.append(probe_saasr(model, exs[i], spk, t0, random.Random(a.seed * 7919 + c), conds))
        if (c + 1) % 50 == 0:
            print(f"  {c + 1}/{len(cases)} cases, {time.time() - t_start:.0f}s", flush=True)
    s = summarize_saasr(records, conds, n_boot=a.n_boot, seed=a.seed)
    s["timing"] = timing
    s["setup"] = {"ckpt": a.ckpt, "causal": bool(model.encoder.causal), "att_context_size": list(model.encoder.att_context_size),
                  "n_mixtures": a.n_eval, "n_cases": len(cases), "windows_frames": SAASR_WINDOWS,
                  "t0": "60 % through the target's active span", "note": "corruption swaps in the other speaker's activity"}
    json.dump(s, open(out_dir / f"saasr_{a.tag}.json", "w"), indent=1)
    print(f"[saasr] wrote {out_dir / f'saasr_{a.tag}.json'}", flush=True)
    return s


def run_report(a):
    out = {"question": "How long does a transient wrong speaker binding keep changing a speaker-conditioned streaming "
                       "model's outputs after the binding is corrected?", "doc": "research/CONTAMINATION.md"}
    if a.turn:
        out["turn_head"] = json.load(open(a.turn))
    if a.turn_extra:
        out["turn_head_replay_delay_rebind"] = json.load(open(a.turn_extra))
    out["speaker_attributed_asr"] = {}
    for p in a.saasr or []:
        d = json.load(open(p))
        out["speaker_attributed_asr"][Path(p).stem.replace("saasr_", "")] = d
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


# =========================================================================== scope addition (coordinator):
# replay cost baseline, correction-delay sweep, real rebind events (research/CONTAMINATION.md sections 6-8)


def _encode_from(model, audio, alen, act_np, s: int, T: int, dev):
    """Encoder re-run from frame s (empty caches) with the track ``act_np`` -> frames [s, T) as (1, T - s, D)."""
    a = torch.as_tensor(act_np[s:], device=dev)[None]
    e, _ = model.encode(audio[:, s * 1280:], alen - s * 1280, spk_act=torch.nn.functional.pad(a, (0, 1)))
    e = e[:, : T - s]
    if e.shape[1] < T - s:
        e = torch.cat([e, e[:, -1:].expand(-1, T - s - e.shape[1], -1)], 1)
    return e


@torch.no_grad()
def turn_timing(model, audio: np.ndarray, buffers=REPLAY_BUFFERS, reps: int = 5) -> dict:
    """Wall-clock (2 threads) of the bounded replay (encoder over the last L s + head over the same frames) vs one
    normal 160 ms step (encoder.stream_step on a 16-mel chunk with primed caches + TurnHead.step on 2 frames)."""
    head = model.heads["turn"]
    dev = next(model.parameters()).device
    x = torch.as_tensor(audio, device=dev)[None]
    out = {}
    for name, L in buffers.items():
        seg = x[:, : L * 1280]
        n = torch.tensor([seg.shape[1]], device=dev)
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            e, _ = model.encode(seg, n, spk_act=torch.zeros(1, L + 1, device=dev))
            T = e.shape[1]
            tf = torch.zeros(1, T, head.text_dim, device=dev)
            head_run(head, e, tf, torch.zeros(1, T, device=dev), torch.zeros(1, T, head.act_columns, device=dev),
                     torch.zeros(1, dtype=torch.long, device=dev))
            ts.append(time.perf_counter() - t)
        out[name] = {"frames": L, "ms": 1000 * float(np.median(ts))}
    # one streaming step: prime the encoder caches with 70 frames, then time a 16-mel-frame chunk + head step
    feats, flen = model.features(x[:, : 90 * 1280], torch.tensor([90 * 1280], device=dev))
    from audioforge.modules.fastconformer import StreamState
    st = StreamState()
    cm = model.encoder.stream_chunk_frames()
    mel = feats
    pos = 0
    while pos + cm <= 70 * 8:
        _, st = model.encoder.stream_step(mel[..., pos: pos + cm], st, spk_act=torch.zeros(1, 2, device=dev))
        pos += cm
    ts = []
    hs = head.init_stream()
    g, _ = head._asr.pred(torch.tensor([[head._asr.blank]], device=dev), None)
    import copy
    for _ in range(20):
        st2 = copy.deepcopy(st)
        t = time.perf_counter()
        e, st2 = model.encoder.stream_step(mel[..., pos: pos + cm], st2, spk_act=torch.zeros(1, 2, device=dev))
        head.step(e, pred_state=g, last_tokens=[[]], spk_act=torch.zeros(1, e.shape[1], device=dev), state=hs,
                  cols=torch.zeros(1, e.shape[1], head.act_columns, device=dev),
                  prim=torch.zeros(1, dtype=torch.long, device=dev))
        ts.append(time.perf_counter() - t)
    out["chunk_step_ms"] = 1000 * float(np.median(ts))
    out["chunk_ms"] = BIN * FRAME * 1000
    for name in buffers:
        out[name]["x_chunk_step"] = out[name]["ms"] / out["chunk_step_ms"]
        out[name]["realtime_chunks"] = out[name]["ms"] / out["chunk_ms"]
    return out


@torch.no_grad()
def probe_replay_and_delay(model, ex, track, ev, t0: int, rng, buffers=REPLAY_BUFFERS, delays=DELAYS, W: int = 12) -> dict:
    """One turn, corrupted window [t0 - W, t0) (zero and swap):
    replay:  bounded replay at t0 with buffer L: encoder from t0 - L (empty caches, corrected track), head from its
             snapshot at t0 - L (clean whenever L >= W), -> |dp| vs the full correct history after t0;
    delay:   the correction arrives at t_c = t0 + d: until t_c the system runs the corrupted history (track correct
             again after the window), at t_c it repairs by (i) nothing, (ii) head replay from the window start on the
             stale encoder frames, (iii) encoder replay from lo - 2*LEFT_CTX + head replay, (iv) the same repair
             pointing the window at a WRONG third column. -> |dp| vs baseline after t_c, outcomes on the composite."""
    head = model.heads["turn"]
    dev = next(model.parameters()).device
    b = Collate(model.tokenizer)([ex])
    audio, alen = b["audio"].to(dev), b["audio_len"].to(dev)
    T = len(ex["spk_act"])
    act, cols, prim, _ = ev._v2_cols(ex, track, "causal_dominant")
    on, en = int(ex["onset_frame"]), int(ex["turn_end_frame"])
    lo = t0 - W
    lo_al = lo - lo % BIN
    az, cz, _ = corrupt_track(act, cols, lo, t0, "zero", rng)
    asw, csw, isw = corrupt_track(act, cols, lo, t0, "swap", rng)
    # wrong correction: the repair believes the window belonged to a third column j (not the true one, not the swap partner)
    others = [j for j in range(1, cols.shape[1]) if j != isw.get("partner")]
    j_wrong = rng.choice(others)
    a_wr, c_wr = act.copy(), cols.copy()
    a_wr[lo:t0] = cols[lo:t0, j_wrong]
    c_wr[lo:t0, 0], c_wr[lo:t0, j_wrong] = cols[lo:t0, j_wrong], cols[lo:t0, 0]
    A = torch.as_tensor(np.stack([act, az, asw]), device=dev)
    Cc = torch.as_tensor(np.stack([cols, cz, csw]), device=dev)
    B = 3
    P = torch.zeros(B, dtype=torch.long, device=dev)
    enc0, elen, hidden = model.encode(audio, alen, return_hidden=True)
    y, n = decoded_text_state(model, head, audio, alen, enc0, hidden, elen, A[:1, : enc0.shape[1]])
    tf = head.text_frames(y, n)[:, :T].expand(B, -1, -1)
    enc, _ = model.encode(audio.expand(B, -1), alen.expand(B), spk_act=torch.nn.functional.pad(A, (0, 1)))
    enc = enc[:, :T]
    p_full, h_full, _ = head_run(head, enc, tf, A, Cc, P)
    pf = p_full.cpu().numpy()
    base = pf[0]
    out = {"t0": t0, "T": T, "replay": {}, "delay": {}, "swap_partner_active": bool(isw.get("differs_from_zero"))}

    def snapshot(j, s):
        if s <= 0:
            return None, None
        hj = h_full[j:j + 1, s - 1][None].contiguous()
        _, dj = head.v3_features(s, A[j:j + 1, :s], Cc[j:j + 1, :s], P[:1], None)
        return hj, dj

    # --- bounded replay at t0 (state snapshot at t0 - L is clean for every L >= W: the result is the same for both
    # corruptions, so one run per L; recorded as the deviation of the bounded replay from the full correct history)
    # The head replays from the (chunk-aligned) window start with its state snapshot there; the encoder cold-starts
    # L before t0, so the buffer beyond the window is encoder warm-up. (Replaying the head from the buffer start
    # instead feeds it the encoder's cold-start frames and costs |dp| about 0.45 for every buffer up to 10 s.)
    for name, L in buffers.items():
        s = max(0, t0 - L)
        hs = max(s, lo_al)
        e = _encode_from(model, audio, alen, act, s, T, dev)[:, hs - s:]
        hj, dj = snapshot(1, hs)  # corrupted run's snapshot (== baseline's for hs <= lo)
        q, _, _ = head_run(head, e, tf[:1, hs:], A[:1, hs:], Cc[:1, hs:], P[:1], h0=hj, dur0=dj)
        q = q[0, t0 - hs:].cpu().numpy()
        d = np.abs(q - base[t0:])
        # encoder-level cold-start error on the window and on the 2 s after t0, relative to the encoder's scale
        scale = float(enc[0].abs().mean())
        enc_err_win = float((e[0, : t0 - hs] - enc[0, hs:t0]).abs().mean() / scale) if t0 > hs else 0.0
        enc_err_post = float((e[0, t0 - hs: t0 - hs + 25] - enc[0, t0: t0 + 25]).abs().mean() / scale)
        oh, ohy = outcome(np.concatenate([pf[1, :t0], q]), az, on, en, ev)
        ob, oby = outcome(base, act, on, en, ev)
        out["replay"][name] = {"dp": bin_curve(d, 0), "dp_max_4s": float(d[: N_BINS * BIN].max()), "clipped": s == 0,
                               "enc_err_window_rel": enc_err_win, "enc_err_post_rel": enc_err_post,
                               "frames": min(L, t0), "hybrid_changed": (_cls(ohy) != _cls(oby)) or ohy["lat_ms"] != oby["lat_ms"],
                               "head_changed": (_cls(oh) != _cls(ob)) or oh["lat_ms"] != ob["lat_ms"]}
    # --- correction delay
    s2 = max(0, lo_al - 2 * LEFT_CTX)
    e_rep = _encode_from(model, audio, alen, act, s2, T, dev)[:, lo_al - s2:]  # corrected encoder frames [lo_al, T)
    e_wr = _encode_from(model, audio, alen, a_wr, s2, T, dev)[:, lo_al - s2:]
    for j, cname in ((1, "zero"), (2, "swap")):
        hj, dj = snapshot(j, lo_al)
        rep_enc = head_run(head, e_rep, tf[:1, lo_al:], A[:1, lo_al:], Cc[:1, lo_al:], P[:1], h0=hj, dur0=dj)[0][0].cpu().numpy()
        rep_head = head_run(head, enc[j:j + 1, lo_al:], tf[:1, lo_al:], A[:1, lo_al:], Cc[:1, lo_al:], P[:1], h0=hj, dur0=dj)[0][0].cpu().numpy()
        A_wr = torch.as_tensor(a_wr, device=dev)[None]
        C_wr = torch.as_tensor(c_wr, device=dev)[None]
        rep_wrong = head_run(head, e_wr, tf[:1, lo_al:], A_wr[:, lo_al:], C_wr[:, lo_al:], P[:1], h0=hj, dur0=dj)[0][0].cpu().numpy()
        for dname, d in delays.items():
            tc = t0 + d
            if tc >= T - 1:
                continue
            variants = {"no_repair": pf[j], "head_replay": np.concatenate([pf[j, :tc], rep_head[tc - lo_al:]]),
                        "enc_head_replay": np.concatenate([pf[j, :tc], rep_enc[tc - lo_al:]]),
                        "wrong_repair": np.concatenate([pf[j, :tc], rep_wrong[tc - lo_al:]])}
            ob, oby = outcome(base, act, on, en, ev)
            for vname, p in variants.items():
                dd = np.abs(p[tc:] - base[tc:])
                oh, ohy = outcome(p, acts_for(j, az, asw), on, en, ev)
                out["delay"][(cname, dname, vname)] = {
                    "dp": bin_curve(dd, 0), "dp_max_4s": float(dd[: N_BINS * BIN].max()) if len(dd) else np.nan,
                    "dp_uncorrected_mean": float(np.abs(pf[j, t0:tc] - base[t0:tc]).mean()) if tc > t0 else 0.0,
                    "hybrid": ohy, "head": oh, "hybrid_base": oby, "head_base": ob}
    return out


def acts_for(j, az, asw):
    return az if j == 1 else asw


def rebind_events(p: np.ndarray, ex: dict, ev, min_pre: int = 3, min_post: int = 12):
    """Real delayed corrections on a cached track: the first causal_dominant re-bind that lands on the ORACLE column
    from a wrong one. -> (t_a, t_r, old, oracle_col) or None. t_a = start of the wrong stretch."""
    T = len(np.asarray(ex["spk_act"]))
    pf = np.stack([ev._fit(p[:, j], T) for j in range(p.shape[1])], 1)
    ct = ev.enroll_causal_dominant(pf)
    oc = ev.enroll_column(pf, np.asarray(ex["spk_act"]), int(ex["onset_frame"]), int(ex["turn_end_frame"]))
    for t in range(1, T):
        if ct[t] != ct[t - 1] and ct[t - 1] >= 0 and ct[t] == oc and ct[t - 1] != oc:
            ta = t - 1
            while ta > 0 and ct[ta - 1] == ct[t - 1]:
                ta -= 1
            if t - ta >= min_pre and T - t >= min_post:
                return ta, t, int(ct[t - 1]), int(oc), pf
            return None
    return None


@torch.no_grad()
def probe_rebind(model, ex, pf, ev, ta: int, tr: int, old: int, oc: int, rng) -> dict:
    """Baseline: the oracle column bound throughout (bound_track representation). Real: the same, but on [ta, tr)
    the follower's actual wrong column is bound (act / cols follow it). Synthetic on the same window: zero, swap.
    -> |dp| after tr (full, state_sub, enc_sub) and the pre-rebind input difference."""
    head = model.heads["turn"]
    dev = next(model.parameters()).device
    b = Collate(model.tokenizer)([ex])
    audio, alen = b["audio"].to(dev), b["audio_len"].to(dev)
    T = pf.shape[0]
    col_o = np.full(T, oc, np.int64)
    act_o, cols_o = ev.bound_track(pf, col_o)
    col_r = col_o.copy()
    col_r[ta:tr] = old
    act_r, cols_r = ev.bound_track(pf, col_r)
    az, cz, _ = corrupt_track(act_o, cols_o, ta, tr, "zero", rng)
    asw, csw, isw = corrupt_track(act_o, cols_o, ta, tr, "swap", rng)
    A = torch.as_tensor(np.stack([act_o, act_r, az, asw]), device=dev)
    Cc = torch.as_tensor(np.stack([cols_o, cols_r, cz, csw]), device=dev)
    B = 4
    P = torch.zeros(B, dtype=torch.long, device=dev)
    enc0, elen, hidden = model.encode(audio, alen, return_hidden=True)
    y, n = decoded_text_state(model, head, audio, alen, enc0, hidden, elen, A[:1, : enc0.shape[1]])
    tf = head.text_frames(y, n)[:, :T].expand(B, -1, -1)
    enc, _ = model.encode(audio.expand(B, -1), alen.expand(B), spk_act=torch.nn.functional.pad(A, (0, 1)))
    enc = enc[:, :T]
    p_full, h_full, _ = head_run(head, enc, tf, A, Cc, P)
    h0 = h_full[:, tr - 1][None].contiguous()
    _, dur0 = head.v3_features(tr, A[:, :tr], Cc[:, :tr], P, None)
    seg = dict(tf=tf[:, tr:], act=A[:, tr:], cols=Cc[:, tr:], prim=P)
    base_h0 = h0[:, :1].expand(-1, B, -1).contiguous()
    base_dur0 = {k: (s[:1].expand(B), r[:1].expand(B)) for k, (s, r) in dur0.items()}
    runs = {"full": p_full[:, tr:], "state_sub": head_run(head, enc[:, tr:], h0=base_h0, dur0=base_dur0, **seg)[0],
            "enc_sub": head_run(head, enc[:1, tr:].expand(B, -1, -1), h0=h0, dur0=dur0, **seg)[0]}
    out = {"ta": ta, "tr": tr, "W": tr - ta, "T": T, "old": old, "oracle": oc, "res": {},
           "input_delta_real": float(np.abs(act_r[ta:tr] - act_o[ta:tr]).mean()),
           "real_track_mean_on_window": float(act_r[ta:tr].mean()), "oracle_track_mean_on_window": float(act_o[ta:tr].mean()),
           "swap_partner_active": bool(isw.get("differs_from_zero"))}
    for prot, q in runs.items():
        q = q.cpu().numpy()
        for j, name in enumerate(("base", "real", "zero", "swap")):
            d = np.abs(q[j] - q[0])
            out["res"][(prot, name)] = {"dp": bin_curve(d, 0), "dp_max_4s": float(d[: N_BINS * BIN].max())}
    return out


def _curve_stats(rows, n_boot, rng):
    x = np.stack(rows)
    m, lo, hi = _ci(x, n_boot, rng)
    return {"dp_mean": m.round(6).tolist(), "dp_lo": lo.round(6).tolist(), "dp_hi": hi.round(6).tolist(),
            "affected_frac_gt_0.01": np.nanmean(x > 0.01, 0).round(4).tolist(), "n": int(len(x))}


def summarize_turn_extra(recs_rd: list, recs_rb: list, timing: dict, n_boot=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {"timing": timing, "bins_ms": [int(b * BIN * FRAME * 1000) for b in range(N_BINS)],
           "replay": {}, "delay": {}, "rebind": {}}
    for name in REPLAY_BUFFERS:
        rows = [r["replay"][name] for r in recs_rd]
        st = _curve_stats([x["dp"] for x in rows], n_boot, rng)
        st.update({"clipped_frac": float(np.mean([x["clipped"] for x in rows])),
                   "enc_err_window_rel_mean": float(np.mean([x["enc_err_window_rel"] for x in rows])),
                   "enc_err_post_rel_mean": float(np.mean([x["enc_err_post_rel"] for x in rows])),
                   "frames_median": float(np.median([x["frames"] for x in rows])),
                   "dp_max_4s_mean": float(np.mean([x["dp_max_4s"] for x in rows])),
                   "turns_max_dp_lt_0.01": float(np.mean([x["dp_max_4s"] < 0.01 for x in rows])),
                   "hybrid_outcome_changed_frac": float(np.mean([x["hybrid_changed"] for x in rows])),
                   "head_outcome_changed_frac": float(np.mean([x["head_changed"] for x in rows]))})
        out["replay"][name] = st
    out["replay_n"] = len(recs_rd)
    for cname in ("zero", "swap"):
        for dname in DELAYS:
            for vname in ("no_repair", "head_replay", "enc_head_replay", "wrong_repair"):
                rows = [r["delay"][(cname, dname, vname)] for r in recs_rd if (cname, dname, vname) in r["delay"]]
                if not rows:
                    continue
                st = _curve_stats([x["dp"] for x in rows], n_boot, rng)
                st["dp_uncorrected_mean"] = float(np.mean([x["dp_uncorrected_mean"] for x in rows]))
                for det in ("hybrid", "head"):
                    ch = [(_cls(x[det]) != _cls(x[det + "_base"])) or x[det]["lat_ms"] != x[det + "_base"]["lat_ms"] for x in rows]
                    cc = [_cls(x[det]) != _cls(x[det + "_base"]) for x in rows]
                    st[f"{det}_changed_frac"] = float(np.mean(ch))
                    st[f"{det}_class_changed_frac"] = float(np.mean(cc))
                    st[f"{det}_corrupted"] = {c: int(sum(_cls(x[det]) == c for x in rows)) for c in ("fire", "miss", "fc")}
                    st[f"{det}_base"] = {c: int(sum(_cls(x[det + "_base"]) == c for x in rows)) for c in ("fire", "miss", "fc")}
                out["delay"][f"{cname}/{dname}/{vname}"] = st
    if recs_rb:
        Ws = np.array([r["W"] for r in recs_rb])
        out["rebind"]["n"] = len(recs_rb)
        out["rebind"]["W_frames_pct_10_50_90"] = [float(np.percentile(Ws, q)) for q in (10, 50, 90)]
        out["rebind"]["input_delta_real_mean"] = float(np.mean([r["input_delta_real"] for r in recs_rb]))
        out["rebind"]["real_track_mean_on_window"] = float(np.mean([r["real_track_mean_on_window"] for r in recs_rb]))
        out["rebind"]["oracle_track_mean_on_window"] = float(np.mean([r["oracle_track_mean_on_window"] for r in recs_rb]))
        out["rebind"]["swap_partner_active_frac"] = float(np.mean([r["swap_partner_active"] for r in recs_rb]))
        out["rebind"]["tr_minus_onset_frames_median"] = None
        for prot in ("full", "state_sub", "enc_sub"):
            for name in ("real", "zero", "swap"):
                out["rebind"][f"{prot}/{name}"] = _curve_stats([r["res"][(prot, name)]["dp"] for r in recs_rb], n_boot, rng)
                out["rebind"][f"{prot}/{name}"]["dp_max_4s_mean"] = float(np.mean([r["res"][(prot, name)]["dp_max_4s"] for r in recs_rb]))
    return out


def run_turn_extra(a):
    torch.set_num_threads(a.threads)
    ev = _ev()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, _ = ev.v2_data(6.0)
    model = load_model(a.ckpt, "cpu")
    d_end = int(round(a.t0_before_end / FRAME))
    idx = [i for i in eligible_turns(ext, d_end, max(TURN_WINDOWS.values())) if ev.v2_track_path(a.work, ext[i]).exists()]
    idx_rd = idx[: a.n]
    t_start = time.time()
    timing = turn_timing(model, np.asarray(ext[idx_rd[0]]["audio"], np.float32))
    print("[turn-extra] timing: " + ", ".join(f"{k} {v['ms']:.0f} ms" for k, v in timing.items() if isinstance(v, dict))
          + f"; one 160 ms chunk step {timing['chunk_step_ms']:.1f} ms", flush=True)
    pk = out_dir / "turn_extra_records.pkl"
    st = pickle.load(open(pk, "rb")) if pk.exists() else {"rd": {}, "rb": {}}
    for c, i in enumerate(idx_rd):
        if i in st["rd"]:
            continue
        wait_for_load()
        ex = ext[i]
        track = np.load(ev.v2_track_path(a.work, ex))
        st["rd"][i] = probe_replay_and_delay(model, ex, track, ev, int(ex["turn_end_frame"]) - d_end, random.Random(a.seed * 31 + i))
        if (c + 1) % 25 == 0:
            pickle.dump(st, open(pk, "wb"))
            print(f"  replay/delay {c + 1}/{len(idx_rd)} turns, {time.time() - t_start:.0f}s", flush=True)
    pickle.dump(st, open(pk, "wb"))
    # real rebind events over all windows with a track
    n_ev = 0
    for i, ex in enumerate(ext):
        if i in st["rb"] or not ev.v2_track_path(a.work, ex).exists():
            continue
        p = np.load(ev.v2_track_path(a.work, ex))
        evn = rebind_events(p, ex, ev)
        st["rb"][i] = None
        if evn is None:
            continue
        wait_for_load()
        ta, tr, old, oc, pf = evn
        st["rb"][i] = probe_rebind(model, ex, pf, ev, ta, tr, old, oc, random.Random(a.seed * 17 + i))
        n_ev += 1
        if n_ev % 25 == 0:
            pickle.dump(st, open(pk, "wb"))
            print(f"  rebind events {n_ev}, {time.time() - t_start:.0f}s", flush=True)
        if a.max_rebinds and n_ev >= a.max_rebinds:
            break
    pickle.dump(st, open(pk, "wb"))
    recs_rd = [st["rd"][i] for i in idx_rd if i in st["rd"]]
    recs_rb = [r for r in st["rb"].values() if r is not None]
    s = summarize_turn_extra(recs_rd, recs_rb, timing, n_boot=a.n_boot, seed=a.seed)
    s["setup"] = {"ckpt": a.ckpt, "n_replay_delay_turns": len(recs_rd), "n_rebind_events": len(recs_rb),
                  "windows_scanned_for_rebinds": len(st["rb"]), "buffers_frames": REPLAY_BUFFERS, "delays_frames": DELAYS,
                  "delay_window_frames": 12, "t0_before_end_frames": d_end}
    json.dump(s, open(out_dir / "turn_extra.json", "w"), indent=1)
    print(f"[turn-extra] wrote {out_dir / 'turn_extra.json'}", flush=True)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("turn")
    t.add_argument("--ckpt", default="runs/stage1_turn_v3_trail6.afm")
    t.add_argument("--work", required=True, help="eot-bench v2 work dir with tracks/<key>.stream_rc.npy")
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=350)
    t.add_argument("--t0-before-end", type=float, default=2.0, help="seconds before the true turn end")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--n-boot", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--budget", type=float, default=0)
    t.add_argument("--early-check", type=int, default=60)
    s = sub.add_parser("saasr")
    s.add_argument("--ckpt", default="runs/speaker_attributed_asr.afm")
    s.add_argument("--out", required=True)
    s.add_argument("--tag", default="noncausal")
    s.add_argument("--n-eval", type=int, default=256)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    x = sub.add_parser("turn-extra")
    x.add_argument("--ckpt", default="runs/stage1_turn_v3_trail6.afm")
    x.add_argument("--work", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--n", type=int, default=150)
    x.add_argument("--max-rebinds", type=int, default=0)
    x.add_argument("--t0-before-end", type=float, default=2.0)
    x.add_argument("--threads", type=int, default=2)
    x.add_argument("--n-boot", type=int, default=1000)
    x.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--turn")
    r.add_argument("--turn-extra")
    r.add_argument("--saasr", nargs="*")
    r.add_argument("--out", default="runs/contamination.json")
    a = ap.parse_args(argv)
    return {"turn": run_turn, "saasr": run_saasr, "turn-extra": run_turn_extra, "report": run_report}[a.cmd](a)


if __name__ == "__main__":
    main()
