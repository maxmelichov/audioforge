"""Turn-head error analysis on AMI dev: why the v3 turn head works with oracle speaker activity and fails with
NVIDIA's streaming Sortformer track, and why a silence timeout on the same noisy track does better
(research/archive/TURN_ERRORS.md).

Two stages, so every model process stays small (CPU, 2 threads, one model load, resumable):

  scores   load the turn model once, run the turn head on the same n AMI dev turns with several INPUT variants and
           save the per-turn score tracks to <scratch>/scores.npz (resumable: variants already there are skipped;
           stops before a group that would overrun --budget seconds). Variants are defined by three inputs:
             enc  = the activity track fed to the speaker-conditioned encoder (speaker kernels) + the text state;
             prim = the head's primary track (its duration counters and the chosen column of ``cols``);
             oth  = the head's other diarizer columns.
           Each is taken from the oracle (O), the Sortformer streaming track (S) or a simulated track:
             oracle        O/O/O  (= scripts/research/eval_stage1.py turn_head_oracle: spk_targets, primary in column 0)
             stream        S/S/S  (= eval_stage1 turn_head_sortformer_stream: all SF columns + enrollment column)
             hyb_oprim     O/O/S  oracle primary column + Sortformer's other columns
             hyb_sprim     S/S/O  Sortformer primary column + oracle other columns
             enc_o_head_s  O/S/S  encoder/text see the oracle, the head's explicit inputs see Sortformer
             enc_s_head_o  S/O/O  the reverse
             mix25..mix100 all three from a "partially repaired" Sortformer track: each contiguous error run of
                           each column (vs the oracle mapped into Sortformer's column layout) is replaced by the
                           oracle with probability q (seeded) -> a track-quality -> miss curve; mix100 = oracle in
                           the Sortformer layout (soft values kept where SF was right)
             fix_post      SF track with the primary column forced to the oracle (off) on frames >= turn end
             fix_pre       SF track with the primary column forced to the oracle on frames < turn end
             hang4, hang8  SF track with a causal k-frame hangover on the primary column (max over the last k frames)
             enc_hang8     the hangover track reaches the encoder/text only; the head's inputs see the raw SF track
  analyze  no model: per-turn track-quality features, per-turn outcomes at each variant's own <=5 % FC threshold
           (recomputed exactly as eot_bench / eval_stage1.eot_bench_emit), cross-tabs, score histograms, false-cutoff
           turns' hesitations, combined decision rules (head AND/OR timeout), -> JSON.

Emission: anything derived from the streaming diarizer is scored with eval_stage1's streaming emission rule
(_stream_emit(6, 7, turn chunk)); the oracle variant with the turn chunk only (and additionally with the streaming
delay, "oracle@stream_emit", to separate the diarizer's latency from its errors).

  .venv/bin/python scripts/research/turn_error_analysis.py scores  --ckpt runs/stage1_turn_v3.afm --groups base --scratch <dir>
  .venv/bin/python scripts/research/turn_error_analysis.py scores  --groups mix,fix,hang --scratch <dir>   # repeat until done
  .venv/bin/python scripts/research/turn_error_analysis.py analyze --scratch <dir> --out runs/turn_v3_error_analysis.json
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import eval_stage1 as ev  # noqa: E402
from audioforge.conversation import eot_bench, silence_scores  # noqa: E402
from audioforge.datasets import ext_tracks as xt  # noqa: E402
from audioforge.heads.turn import _onset_end  # noqa: E402

C_SF, R_SF = ev.SORTFORMER_LOW_LATENCY["chunk_len"], ev.SORTFORMER_LOW_LATENCY["chunk_right_context"]
FRAME_MS = 80.0
S = 4
GROUPS = {"base": ["oracle", "stream", "hyb_oprim", "hyb_sprim", "enc_o_head_s", "enc_s_head_o"],
          "mix": ["mix25", "mix50", "mix75", "mix100"],
          "fix": ["fix_post", "fix_pre"],
          "hang": ["hang4", "hang8", "enc_hang8"]}
MIX_Q = {"mix25": 0.25, "mix50": 0.5, "mix75": 0.75, "mix100": 1.0}


# --------------------------------------------------------------------------- data (no model)
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of the True runs of a 1-D bool array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def max_run(mask) -> int:
    r = runs(mask)
    return max((b - a for a, b in r), default=0)


def oracle_in_sf_layout(sf: np.ndarray, tg: np.ndarray, col: int) -> np.ndarray:
    """Oracle spk_targets (T, S; primary in column 0) re-ordered into the Sortformer track's column layout: the
    primary goes to the enrollment column ``col``, the other oracle speakers to the remaining columns by maximal
    whole-window overlap (hard, ties by soft). Simulation only (looks at the whole window)."""
    T = len(sf)
    tg = np.asarray(tg, np.float32)[:T]
    tg = np.pad(tg, ((0, 0), (0, max(0, S - tg.shape[1]))))[:, :S]
    out = np.zeros((T, S), np.float32)
    out[:, col] = tg[:, 0]
    free = [s for s in range(S) if s != col]
    best, bp = None, None
    for perm in itertools.permutations(free):
        hard = sum(float(((sf[:, s] > 0.5) & (tg[:, j + 1] > 0.5)).sum()) for j, s in enumerate(perm))
        soft = sum(float((sf[:, s] * tg[:, j + 1]).sum()) for j, s in enumerate(perm))
        if best is None or (hard, soft) > best:
            best, bp = (hard, soft), perm
    for j, s in enumerate(bp):
        out[:, s] = tg[:, j + 1]
    return out


def repair(sf: np.ndarray, osf: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    """Replace each contiguous error run (per column, hard decisions differ) of ``sf`` by the oracle with prob. q."""
    out = sf.copy()
    for s in range(sf.shape[1]):
        for a, b in runs((sf[:, s] > 0.5) != (osf[:, s] > 0.5)):
            if rng.random() < q:
                out[a:b, s] = osf[a:b, s]
    return out


def load_turns(n: int, cache: Path) -> dict:
    """The eval's n AMI dev turns + their cached streaming Sortformer tracks (on the turn grid) + enrollment column."""
    val = ev.ami_dev("turn", n)
    on, en = _onset_end(val)
    T = [len(np.asarray(v["spk_act"])) for v in val]
    sf, cols, osf = [], [], []
    for v, o, e, t in zip(val, on, en, T):
        p = np.load(xt.track_path(cache, xt.example_key(v), "stream"))
        c = ev.enroll_column(p, np.asarray(v["spk_act"]), o, e)
        full = np.stack([ev._fit(p[:, s], t) for s in range(p.shape[1])], 1)
        sf.append(full)
        cols.append(c)
        osf.append(oracle_in_sf_layout(full, np.asarray(v["spk_targets"]), c))
    return dict(val=val, on=on, en=en, T=T, sf=sf, col=cols, osf=osf)


def oracle_cols(v: dict, T: int) -> np.ndarray:
    tg = np.asarray(v["spk_targets"], np.float32)[:T]
    return np.pad(tg, ((0, 0), (0, max(0, S - tg.shape[1]))))[:, :S]


def variant_inputs(name: str, d: dict, seed: int = 0) -> tuple[str, list, list, list]:
    """-> (enc_key, enc_acts, cols, prims) for a variant (module docstring). enc_key groups variants that share
    the speaker-conditioned encode + text state."""
    val, T, sf, col, osf = d["val"], d["T"], d["sf"], d["col"], d["osf"]
    ora = [np.asarray(v["spk_act"], np.float32)[:t] for v, t in zip(val, T)]
    sfp = [p[:, c] for p, c in zip(sf, col)]
    if name == "oracle":
        return "O", ora, [oracle_cols(v, t) for v, t in zip(val, T)], [0] * len(val)
    if name == "stream":
        return "S", sfp, sf, col
    if name in ("hyb_oprim", "enc_o_head_s"):
        if name == "enc_o_head_s":
            return "O", ora, sf, col
        cs = []
        for p, c, a in zip(sf, col, ora):
            x = p.copy()
            x[:, c] = a
            cs.append(x)
        return "O", ora, cs, col
    if name in ("hyb_sprim", "enc_s_head_o"):
        if name == "enc_s_head_o":
            return "S", sfp, [oracle_cols(v, t) for v, t in zip(val, T)], [0] * len(val)
        cs = []
        for v, t, a in zip(val, T, sfp):
            x = oracle_cols(v, t)
            x[:, 0] = a
            cs.append(x)
        return "S", sfp, cs, [0] * len(val)
    if name in MIX_Q:
        rng = np.random.default_rng(seed + int(MIX_Q[name] * 100))
        cs = [repair(p, o, MIX_Q[name], rng) for p, o in zip(sf, osf)]
        return name, [x[:, c] for x, c in zip(cs, col)], cs, col
    if name in ("fix_post", "fix_pre"):
        cs = []
        for p, c, o, e in zip(sf, col, osf, d["en"]):
            x = p.copy()
            sl = slice(e, None) if name == "fix_post" else slice(0, e)
            x[sl, c] = o[sl, c]
            cs.append(x)
        return name, [x[:, c] for x, c in zip(cs, col)], cs, col
    if name in ("hang4", "hang8", "enc_hang8"):
        k = 4 if name == "hang4" else 8
        hp = [hangover(a, k) for a in sfp]
        if name == "enc_hang8":
            return name, hp, sf, col
        cs = []
        for p, c, a in zip(sf, col, hp):
            x = p.copy()
            x[:, c] = a
            cs.append(x)
        return name, hp, cs, col
    raise KeyError(name)


def hangover(a: np.ndarray, k: int) -> np.ndarray:
    """Causal hangover: a[t] -> max(a[t-k .. t]) (bridges spurious gaps <= k frames; delays every offset by k)."""
    a = np.asarray(a, np.float32)
    return np.max(np.stack([np.concatenate([np.zeros(j, np.float32), a[: len(a) - j]]) for j in range(k + 1)]), 0)


def mixed_tracks(name: str, d: dict, seed: int = 0) -> list[np.ndarray]:
    """The full (T, S) track a variant feeds (for track-quality stats of simulated tracks)."""
    return variant_inputs(name, d, seed)[2]


# --------------------------------------------------------------------------- stage 1: scores (model)
@torch.no_grad()
def compute_scores(model, d: dict, names: list[str], batch_size: int) -> dict[str, list[np.ndarray]]:
    """Turn-head P(EOT) per turn for each variant; the speaker-conditioned encode + text state are computed once per
    (batch, enc track) and shared by the variants with that enc track (= turn_scores_given_act, split in two)."""
    from audioforge.data import Collate
    from audioforge.heads.turn import decoded_text_state
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    head, hc = model.heads[name], model.head_cfg[name]
    asr_cond = bool(model.head_cfg[head.text_head].get("condition_on_speaker")) if head.use_text else False
    inp = {v: variant_inputs(v, d) for v in names}
    print(f"  turn head {name}: condition_on_speaker={hc.get('condition_on_speaker')} asr_text_speaker_conditioned="
          f"{asr_cond}", flush=True)
    keys = sorted({x[0] for x in inp.values()})
    val = d["val"]
    out = {v: [] for v in names}
    for i in range(0, len(val), batch_size):
        cc = val[i: i + batch_size]
        b = Collate(model.tokenizer)(cc)
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        B, Te = enc.shape[0], enc.shape[1]

        def tens(lst, dims=None):
            x = torch.zeros(B, Te, *(dims or ()))
            for j, a in enumerate(lst[i: i + batch_size]):
                a = np.asarray(a, np.float32)
                m = min(len(a), Te)
                x[j, :m] = torch.as_tensor(a[:m, :S] if a.ndim == 2 else a[:m])
            return x
        text_shared = None
        for k in keys:
            vs = [v for v in names if inp[v][0] == k]
            act = tens(inp[vs[0]][1])
            e = model.encode(b["audio"], b["audio_len"], spk_act=act)[0] if hc.get("condition_on_speaker") \
                else model.head_input(name, enc, hidden)
            if not head.use_text:
                text = None
            elif asr_cond or text_shared is None:
                text = decoded_text_state(model, head, b["audio"], b["audio_len"], enc, hidden, elen, act)
                text_shared = None if asr_cond else text
            else:
                text = text_shared
            for v in vs:
                _, _, cols, prims = inp[v]
                ct = tens(cols, (S,))
                pt = torch.as_tensor([int(x) for x in prims[i: i + batch_size]])
                ha = ct.gather(2, pt[:, None, None].expand(-1, Te, 1))[..., 0]  # head's primary track = cols[prim]
                p = head.decode(e, elen, spk_act=ha, text=text, cols=ct, prim=pt)
                for j in range(len(cc)):
                    m = min(int(elen[j]), len(cc[j]["spk_act"]))
                    out[v].append(p[j, :m].float().cpu().numpy())
    return out


def stage_scores(a):
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    f = scratch / "scores.npz"
    have = dict(np.load(f, allow_pickle=True)) if f.exists() else {}
    todo = [v for g in a.groups.split(",") for v in GROUPS[g] if v not in have]
    if not todo:
        print("nothing to do", sorted(have))
        return
    t0 = time.time()
    d = load_turns(a.n, Path(a.diar_cache))
    print(f"data {time.time() - t0:.0f}s n={len(d['val'])}", flush=True)
    from audioforge.train import load_model
    model = load_model(a.ckpt, "cpu")
    model.eval()
    print(f"model {time.time() - t0:.0f}s", flush=True)
    # one enc track per call group: variants sharing an enc key are computed together
    by_key = {}
    for v in todo:
        by_key.setdefault(variant_inputs(v, d)[0], []).append(v)
    per = None
    for k, vs in by_key.items():
        if per and a.budget and time.time() - t0 + per > a.budget:
            print(f"budget: stopping before {vs}; re-run to resume", flush=True)
            break
        t1 = time.time()
        sc = compute_scores(model, d, vs, a.batch_size)
        for v in vs:
            have[v] = np.array(sc[v], dtype=object)
        np.savez(f, **have)
        per = max(per or 0, time.time() - t1)
        print(f"  {vs}: {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
    meta = {"ckpt": a.ckpt, "n": len(d["val"]), "chunk": ev.chunk_of(model)}
    (scratch / "scores_meta.json").write_text(json.dumps(meta))


# --------------------------------------------------------------------------- stage 2: analysis (no model)
def emit_transform(scores, ends, emit_fn) -> list[np.ndarray]:
    """The score re-indexing of eval_stage1.eot_bench_emit (post-end part indexed by emission time), so per-turn
    outcomes can be read with chunk = 1 exactly as that function's eot_bench call sees them."""
    out = []
    floor = min(float(np.min(s)) for s in scores if len(s)) - 1.0
    for s, e in zip(scores, ends):
        s = np.asarray(s, np.float64)
        e = int(e)
        if len(s) <= e:
            out.append(s)
            continue
        t = np.arange(e, len(s))
        dd = np.array([emit_fn(int(x)) for x in t]) - e
        z = np.full(int(dd[-1]), floor)
        np.maximum.at(z, dd - 1, s[e:])
        out.append(np.concatenate([s[:e], np.maximum.accumulate(z)]))
    return out


def bench(scores, on, en, emit: str, chunk: int) -> tuple[dict, list[np.ndarray], int]:
    """(eot_bench result, the scores it saw, its chunk) for emission rule 'turn' (chunk) or 'stream'."""
    if emit == "stream":
        z = emit_transform(scores, en, ev._stream_emit(C_SF, R_SF, chunk))
        return eot_bench(z, on, en, chunk=1), z, 1
    return eot_bench(scores, on, en, chunk=chunk), [np.asarray(s, np.float64) for s in scores], chunk


def exact_threshold(z, point: dict) -> float:
    """The unrounded eot_bench threshold behind a reported (5-decimal) operating point: eot_bench's own candidate set,
    nearest candidate that reproduces the point's FC rate."""
    if point["threshold"] is None:
        return float("inf")
    allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in z]))
    ths = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
    r = point["threshold"]
    for t in ths[np.argsort(np.abs(ths - r))][:20]:
        if abs(t - r) <= 5e-6 + 1e-12:
            return float(t)
    raise ValueError(f"no eot_bench threshold near {r}")


def outcomes(z, on, en, th: float, chunk: int) -> list[dict]:
    """Per-turn outcome at threshold th, exactly eot_bench's rule: fc if any score > th on [onset, end); else the
    first post-end frame with score > th gives latency (emit - end) * 80 ms; none = miss."""
    res = []
    for s, o, e in zip(z, on, en):
        pre, post = s[o:e], s[e:]
        mpre = float(pre.max()) if len(pre) else -np.inf
        mpost = float(post.max()) if len(post) else -np.inf
        if mpre > th:
            res.append(dict(out="fc", lat=None, max_pre=mpre, max_post=mpost))
            continue
        idx = np.nonzero(post > th)[0]
        if len(idx):
            t = e + int(idx[0])
            res.append(dict(out="fire", lat=((t // chunk + 1) * chunk - e) * FRAME_MS, max_pre=mpre, max_post=mpost))
        else:
            res.append(dict(out="miss", lat=None, max_pre=mpre, max_post=mpost))
    return res


def summary(oc: list[dict]) -> dict:
    n = len(oc)
    fc = sum(o["out"] == "fc" for o in oc)
    keep = [o for o in oc if o["out"] != "fc"]
    lat = np.array([o["lat"] if o["out"] == "fire" else np.inf for o in keep])
    q = (lambda p: float(np.quantile(lat, p, method="inverted_cdf")) if len(lat) else float("inf"))
    return dict(n=n, fc=fc, fire=sum(o["out"] == "fire" for o in oc), miss=sum(o["out"] == "miss" for o in oc),
                fc_rate=round(fc / n, 4), miss_rate=round(float(np.isinf(lat).mean()) if len(lat) else 1.0, 4),
                p50_ms=q(0.5), p90_ms=q(0.9))


def track_features(d: dict) -> list[dict]:
    """Per-turn quality of the streaming Sortformer primary column (enrollment column) vs the oracle primary."""
    feats = []
    for v, o, e, T, p, c, osf in zip(d["val"], d["on"], d["en"], d["T"], d["sf"], d["col"], d["osf"]):
        r = np.asarray(v["spk_act"])[:T] > 0.5
        h = p[:, c] > 0.5
        oth_sf = np.delete(p, c, 1).max(1) > 0.5
        tg = oracle_cols(v, T)
        oth_or = tg[:, 1:].max(1) > 0.5
        post = slice(e, T)
        n_post = max(1, T - e)
        in_turn = slice(o, e)
        rr, hh = r[in_turn], h[in_turn]
        fa_post = h[post]
        # lag of the end: >0 frames the column stays on after the true end, <=0 it went off before the end
        if e < T and h[e]:
            offs = np.nonzero(~h[e:])[0]
            lag = int(offs[0]) if len(offs) else T - e
        else:
            ons = np.nonzero(h[:e])[0]
            lag = int(ons[-1]) + 1 - e if len(ons) else -(e - o)
        ons_t = np.nonzero(h[max(0, o - 12):e])[0]
        hes = np.asarray(v.get("hes", np.zeros(T)))[:T] > 0.5
        nxt = np.nonzero(oth_or[e:])[0]
        feats.append(dict(
            turn_frames=e - o, post_frames=T - e, col=int(c),
            in_turn_miss=round(float((rr & ~hh).sum() / max(1, rr.sum())), 4),
            in_turn_fa=round(float((~rr & hh).sum() / max(1, (~rr).sum())), 4) if (~rr).any() else 0.0,
            post_fa=round(float(fa_post.sum() / n_post), 4),
            post_fa_first13=round(float(h[e:e + 13].mean()), 4) if e < T else 0.0,
            clean_post=bool(not h[e + 2:].any()),
            sf_other_on_post=bool(oth_sf[post].any()),
            oracle_other_on_post=bool(oth_or[post].any()),
            oracle_next_gap=int(nxt[0]) if len(nxt) else None,
            overlap_at_end=bool(e > 0 and oth_or[e - 1]),
            fa_is_next_speaker=round(float((fa_post & oth_or[post]).sum() / max(1, fa_post.sum())), 4)
            if fa_post.any() else None,
            max_miss_run=max_run(rr & ~hh), max_sf_silence_in_turn=max_run(~hh),
            max_oracle_silence_in_turn=max_run(~rr), max_hes_run=max_run(hes[in_turn]),
            n_hesitations=int(v.get("n_hesitations", 0)),
            lag_end=lag, lag_onset=(int(ons_t[0]) + max(0, o - 12) - o) if len(ons_t) else None,
            track_err=round(float(((p > 0.5) != (osf > 0.5)).sum() / max(1, (osf > 0.5).sum())), 4)))
    return feats


def track_err_stats(tracks, d) -> dict:
    """Frame error of a (T,S) track vs the oracle in the same layout (miss+FA over oracle speech frames, a DER with a
    fixed mapping) and the primary column's miss / post-end activity."""
    err = sp = pm = ps = pon = pn = 0
    for x, o, e, c in zip(tracks, d["osf"], d["en"], d["col"]):
        hx, ho = x > 0.5, o > 0.5
        err += int((hx != ho).sum())
        sp += int(ho.sum())
        pm += int((ho[:, c] & ~hx[:, c]).sum())
        ps += int(ho[:, c].sum())
        pon += int(hx[e:, c].sum())
        pn += max(0, len(x) - e)
    return dict(frame_err=round(err / max(1, sp), 4), prim_miss=round(pm / max(1, ps), 4),
                prim_post_end_active=round(pon / max(1, pn), 4))


def hist(vals, edges) -> list[tuple[str, int]]:
    vals = np.asarray(vals, np.float64)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append((f"[{lo:g},{hi:g})", int(((vals >= lo) & (vals < hi)).sum())))
    out.append((f">={edges[-1]:g}", int((vals >= edges[-1]).sum())))
    return out


def crosstab(rows, cols_key, row_key) -> dict:
    t = {}
    for r in rows:
        t.setdefault(str(r[row_key]), {}).setdefault(str(r[cols_key]), 0)
        t[str(r[row_key])][str(r[cols_key])] += 1
    return t


def bin_of(x, edges, labels):
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        if lo <= x < hi:
            return lab
    return labels[-1]


def stage_analyze(a):
    scratch = Path(a.scratch)
    sc = dict(np.load(scratch / "scores.npz", allow_pickle=True))
    meta = json.loads((scratch / "scores_meta.json").read_text())
    chunk = int(meta["chunk"])
    d = load_turns(a.n, Path(a.diar_cache))
    on, en = d["on"], d["en"]
    n = len(on)
    if int(meta.get("n", n)) != n or any(len(sc[v]) != n for v in sc):  # scored with another --n: misaligned
        raise SystemExit(f"--n {a.n} loads {n} turns but {scratch / 'scores.npz'} holds {meta.get('n')} "
                         f"(re-run stage 1 with this --n, or pass --n {meta.get('n')})")
    feats = track_features(d)
    res = {"n": n, "ckpt": meta["ckpt"], "chunk_frames": chunk, "variants": {}, "per_turn": []}
    emit = {v: ("turn" if v == "oracle" else "stream") for v in sc}
    oc, zs, ths = {}, {}, {}
    order = [v for g in GROUPS.values() for v in g if v in sc]
    for v in order + (["oracle@stream_emit"] if "oracle" in sc else []):
        base = v.split("@")[0]
        em = "stream" if v.endswith("@stream_emit") else emit[base]
        b, z, ch = bench(list(sc[base]), on, en, em, chunk)
        th = exact_threshold(z, b["at_max_fc"])
        o = outcomes(z, on, en, th, ch)
        s = summary(o)
        # the per-turn replay must reproduce eot_bench's operating point exactly
        assert s["fc_rate"] == b["at_max_fc"]["fc_rate"] and s["miss_rate"] == b["at_max_fc"]["miss_rate"] and \
            s["p50_ms"] == b["at_max_fc"]["p50_ms"], (v, s, b["at_max_fc"])
        oc[v], zs[v], ths[v] = o, z, th
        res["variants"][v] = dict(emit=em, at_5pct_fc=b["at_max_fc"], at_p50_le_400ms=b["at_fixed_latency"],
                                  counts={k: s[k] for k in ("fc", "fire", "miss")})
    # timeout on the same streaming track (eval_stage1 timeout_sortformer_stream)
    sfp = [p[:, c] for p, c in zip(d["sf"], d["col"])]
    sil = [silence_scores(a_, o) for a_, o in zip(sfp, on)]
    b, z, ch = bench(sil, on, en, "stream", chunk)
    oc["timeout"] = outcomes(z, on, en, exact_threshold(z, b["at_max_fc"]), ch)
    res["variants"]["timeout_stream"] = dict(emit="stream", at_5pct_fc=b["at_max_fc"],
                                             counts={k: summary(oc["timeout"])[k] for k in ("fc", "fire", "miss")})
    # timeouts on the simulated tracks
    for v in [x for x in order if x in MIX_Q or x.startswith(("fix", "hang", "enc_hang"))]:
        tr = mixed_tracks(v, d)
        bb, _, _ = bench([silence_scores(x[:, c], o) for x, c, o in zip(tr, d["col"], on)], on, en, "stream", chunk)
        res["variants"][v]["timeout_same_track_at_5pct_fc"] = bb["at_max_fc"]
        res["variants"][v]["track"] = track_err_stats(tr, d)
    res["variants"]["stream"]["track"] = track_err_stats(d["sf"], d)
    res["variants"]["stream"]["timeout_same_track_at_5pct_fc"] = b["at_max_fc"]

    # ---- per-turn table
    for i in range(n):
        row = dict(i=i, key=xt.example_key(d["val"][i]), **feats[i])
        for v in oc:
            row[f"{v}_out"] = oc[v][i]["out"]
            row[f"{v}_lat"] = oc[v][i]["lat"]
            if v != "timeout":
                row[f"{v}_max_pre"] = round(oc[v][i]["max_pre"], 4)
                row[f"{v}_max_post"] = round(oc[v][i]["max_post"], 4)
        res["per_turn"].append(row)
    P = res["per_turn"]
    for r in P:
        r["post_fa_bin"] = bin_of(r["post_fa"], [0, 1e-9, 0.25, 0.5, 1.01], ["0", "(0,.25]", "(.25,.5]", ">.5"])
        r["sf_sil_bin"] = bin_of(r["max_sf_silence_in_turn"], [0, 3, 6, 11, 21, 10 ** 6],
                                 ["0-2", "3-5", "6-10", "11-20", ">20"])
        r["miss_bin"] = bin_of(r["in_turn_miss"], [0, 0.05, 0.2, 0.5, 1.01], ["<5%", "5-20%", "20-50%", ">50%"])

    # ---- cross-tabs (head fed the streaming track, at its own <=5 % FC threshold)
    X = {}
    X["stream_x_clean_post"] = crosstab(P, "stream_out", "clean_post")
    X["stream_x_post_fa"] = crosstab(P, "stream_out", "post_fa_bin")
    X["stream_x_sf_silence_in_turn"] = crosstab(P, "stream_out", "sf_sil_bin")
    X["stream_x_in_turn_miss"] = crosstab(P, "stream_out", "miss_bin")
    X["timeout_x_clean_post"] = crosstab(P, "timeout_out", "clean_post")
    X["timeout_x_post_fa"] = crosstab(P, "timeout_out", "post_fa_bin")
    X["timeout_x_sf_silence_in_turn"] = crosstab(P, "timeout_out", "sf_sil_bin")
    X["stream_x_timeout"] = crosstab(P, "timeout_out", "stream_out")
    X["oracle_x_stream"] = crosstab(P, "stream_out", "oracle_out")
    if "hyb_oprim" in sc:
        X["stream_x_hyb_oprim"] = crosstab(P, "hyb_oprim_out", "stream_out")
    res["crosstabs"] = X

    # (a) clean post-end window: does the head still miss?
    clean = [r for r in P if r["clean_post"]]
    dirty = [r for r in P if not r["clean_post"]]
    res["split_a_clean_post"] = {
        "n_clean": len(clean), "n_post_fa": len(dirty),
        "clean": {k: {o: sum(r[f"{k}_out"] == o for r in clean) for o in ("fire", "miss", "fc")}
                  for k in ("stream", "timeout", "oracle")},
        "post_fa": {k: {o: sum(r[f"{k}_out"] == o for r in dirty) for o in ("fire", "miss", "fc")}
                    for k in ("stream", "timeout", "oracle")},
        "clean_head_miss_max_post_hist": hist([r["stream_max_post"] for r in clean if r["stream_out"] == "miss"],
                                              [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]),
        "threshold_stream": ths["stream"], "threshold_oracle": ths["oracle"]}
    # where does the timeout win?
    win = [r for r in P if r["timeout_out"] == "fire" and r["stream_out"] != "fire"]
    lose = [r for r in P if r["stream_out"] == "fire" and r["timeout_out"] != "fire"]

    def prof(rows):
        if not rows:
            return {"n": 0}
        keys = ["post_fa", "in_turn_miss", "max_sf_silence_in_turn", "max_miss_run", "lag_end", "stream_max_post",
                "stream_max_pre", "oracle_max_post"]
        out = {"n": len(rows), "clean_post": sum(r["clean_post"] for r in rows),
               "head_out": {o: sum(r["stream_out"] == o for r in rows) for o in ("fire", "miss", "fc")}}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[f"median_{k}"] = round(float(np.median(vals)), 4) if vals else None
        return out
    res["split_b_timeout_wins"] = {"timeout_fires_head_not": prof(win), "head_fires_timeout_not": prof(lose),
                                   "timeout_fire_latency_ms_p50_on_wins":
                                   float(np.median([r["timeout_lat"] for r in win])) if win else None}
    # (c) oracle-fed fires, stream-fed does not
    cset = [r for r in P if r["oracle_out"] == "fire" and r["stream_out"] != "fire"]
    both = [r for r in P if r["oracle_out"] == "fire" and r["stream_out"] == "fire"]
    res["split_c_oracle_fires_stream_not"] = {"n": len(cset), "stream_out": {o: sum(r["stream_out"] == o for r in cset)
                                                                           for o in ("miss", "fc")},
                                              "profile": prof(cset), "profile_both_fire": prof(both)}
    if "hyb_oprim" in sc:
        res["split_c_oracle_fires_stream_not"]["hyb_oprim_out"] = {o: sum(r["hyb_oprim_out"] == o for r in cset)
                                                                  for o in ("fire", "miss", "fc")}
    # input differences on (c): mean head inputs over the first 13 post-end frames and inside the turn
    res["split_c_inputs"] = input_diff(d, cset, both)

    # ---- score distributions (max over [onset, end) and over the post-end part as eot_bench sees it)
    E = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
    res["score_hist"] = {v: {"max_pre": hist([o["max_pre"] for o in oc[v]], E),
                             "max_post": hist([o["max_post"] for o in oc[v]], E)}
                         for v in ("oracle", "stream", "hyb_oprim", "hyb_sprim") if v in oc}
    # is the threshold forced high by a few FC-prone turns?
    pre_st = np.array([o["max_pre"] for o in oc["stream"]])
    post_st = np.array([o["max_post"] for o in oc["stream"]])
    fcurve = []
    for fcmax in (0.05, 0.075, 0.10, 0.15, 0.20):
        bb = eot_bench(zs["stream"], on, en, chunk=1, max_fc=fcmax)["at_max_fc"]
        fcurve.append(dict(max_fc=fcmax, threshold=bb["threshold"], miss_rate=bb["miss_rate"], p50_ms=bb["p50_ms"]))
    top = np.argsort(-pre_st)[:15]
    res["threshold_pressure"] = {
        "miss_vs_allowed_fc": fcurve,
        "top15_max_pre": [dict(i=int(i), max_pre=round(float(pre_st[i]), 4), **{k: P[i][k] for k in (
            "max_hes_run", "max_oracle_silence_in_turn", "max_sf_silence_in_turn", "max_miss_run", "in_turn_miss",
            "n_hesitations", "oracle_out")}) for i in top],
        "turns_pre_above_oracle_threshold": int((pre_st > ths["oracle"]).sum()),
        "turns_post_above_oracle_threshold": int((post_st > ths["oracle"]).sum()),
        "separable_turns_frac": round(float((post_st > pre_st).mean()), 4),
        "note": "separable = the turn's own max post-end score exceeds its max pre-end score: a per-turn threshold "
                "between them would fire correctly (an upper bound for per-turn calibration)"}
    fc_turns = {v: [dict(i=r["i"], max_hes_run_frames=r["max_hes_run"], max_oracle_silence=r["max_oracle_silence_in_turn"],
                         max_sf_silence=r["max_sf_silence_in_turn"], max_miss_run=r["max_miss_run"],
                         n_hesitations=r["n_hesitations"]) for r in P if r[f"{v}_out"] == "fc"]
                for v in ("oracle", "stream", "timeout")}
    res["false_cutoff_turns"] = fc_turns
    res["hesitation_ms_fc_vs_rest"] = {
        v: {"fc_median_max_hes_ms": float(np.median([t["max_hes_run_frames"] for t in fc_turns[v]]) * FRAME_MS)
            if fc_turns[v] else None,
            "fc_with_hes": sum(t["max_hes_run_frames"] > 0 for t in fc_turns[v]),
            "rest_with_hes_frac": round(float(np.mean([r["max_hes_run"] > 0 for r in P if r[f"{v}_out"] != "fc"])), 4)}
        for v in fc_turns}

    # ---- decision combinations on the streaming track
    res["combos"] = combos(sc["stream"], sil, on, en, chunk)
    if "oracle" in sc:
        res["combos_oracle_track"] = combos(sc["oracle"], [silence_scores(np.asarray(v["spk_act"]), o)
                                                           for v, o in zip(d["val"], on)], on, en, chunk, emit="turn")
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, default=float))
    print_report(res)
    print(f"-> {out}")


def input_diff(d, cset, both) -> dict:
    """Mean head inputs for two turn sets: primary column / other columns (SF vs oracle), inside the turn's last
    13 frames and in the first 13 post-end frames."""
    def one(rows):
        acc = {k: [] for k in ("sf_prim_post", "sf_other_post", "or_other_post", "sf_prim_last13", "or_prim_last13",
                               "sf_prim_since_at_e6")}
        for r in rows:
            i = r["i"]
            p, c, e, v, T = d["sf"][i], d["col"][i], d["en"][i], d["val"][i], d["T"][i]
            tg = oracle_cols(v, T)
            acc["sf_prim_post"].append(float(p[e:e + 13, c].mean()))
            acc["sf_other_post"].append(float(np.delete(p, c, 1)[e:e + 13].max(1).mean()))
            acc["or_other_post"].append(float(tg[e:e + 13, 1:].max(1).mean()))
            acc["sf_prim_last13"].append(float(p[max(0, e - 13):e, c].mean()))
            acc["or_prim_last13"].append(float(tg[max(0, e - 13):e, 0].mean()))
            h = p[: e + 6, c] > 0.5  # frames since the SF primary was last on, 6 frames after the true end
            ons = np.nonzero(h)[0]
            acc["sf_prim_since_at_e6"].append(float(e + 5 - ons[-1]) if len(ons) else float(e + 6))
        return {k: round(float(np.mean(x)), 4) if x else None for k, x in acc.items()}
    return {"oracle_fires_stream_not": one(cset), "both_fire": one(both)}


def combos(head_sc, sil, on, en, chunk, emit="stream") -> dict:
    """Head AND / OR a silence timeout on the same primary track; best miss at <= 5 % FC over the timeout k."""
    out = {}
    for mode in ("and", "or"):
        best = None
        for k in range(0, 26):
            if mode == "and":
                s = [np.asarray(h, np.float64) * (np.asarray(t[: len(h)]) >= k) for h, t in zip(head_sc, sil)]
            else:
                s = [np.where(np.asarray(t[: len(h)]) >= k, 2.0, np.asarray(h, np.float64)) for h, t in zip(head_sc, sil)]
            b = bench(s, on, en, emit, chunk)[0]["at_max_fc"]
            if b["threshold"] is None:
                continue
            key = (b["miss_rate"], b["p50_ms"])
            if best is None or key < best[0]:
                best = (key, dict(k_frames=k, **b))
        out[f"head_{mode}_timeout"] = best[1] if best else None
    return out


def print_report(res):
    V = res["variants"]
    print("variant                 thr      fc    miss   p50   p90   (fc/fire/miss)")
    for v, x in V.items():
        p = x["at_5pct_fc"]
        print(f"  {v:22s} {p['threshold']!s:8.8} {p['fc_rate']!s:6} {p['miss_rate']!s:6} {p['p50_ms']:6} "
              f"{p['p90_ms']:6}  {x['counts']}")
    for k, t in res["crosstabs"].items():
        print(k, json.dumps(t))
    for k in ("split_a_clean_post", "split_b_timeout_wins", "split_c_oracle_fires_stream_not", "split_c_inputs",
              "score_hist", "threshold_pressure", "hesitation_ms_fc_vs_rest", "combos", "combos_oracle_track"):
        print(k, json.dumps(res.get(k), default=float)[:2500])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["scores", "analyze"])
    ap.add_argument("--ckpt", default="runs/stage1_turn_v3.afm")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--diar-cache", default="data/ami/cache/sortformer/dev")
    ap.add_argument("--scratch", default="runs/turn_error_scratch", help="per-variant score tracks (scores.npz)")
    ap.add_argument("--groups", default="base", help=f"scores: comma list of {list(GROUPS)}")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--budget", type=float, default=420.0, help="scores: seconds per process (stops between groups)")
    ap.add_argument("--out", default="runs/turn_v3_error_analysis.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    (stage_scores if a.stage == "scores" else stage_analyze)(a)


if __name__ == "__main__":
    main()
