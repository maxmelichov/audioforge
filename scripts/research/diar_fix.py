"""Multi-speaker diarization in the served system: reproduce, diagnose, measure (research/DIARIZATION_FIX.md).

Subcommands (every model / server run goes through scripts/dev/gate.sh):

  mix     --out DIR [--seed 0 --n-spk 6 --turns 30]      6-speaker LibriSpeech test-clean mix with turn-taking + overlap
  clip    --corpus icsi|ami --meeting M --out DIR [--dur 60 --min-spk 4]   a real meeting window with >= 4 speakers
  run     --url ws://.. --clip DIR/name [--speed 1 --sessions 1 --out JSON]  stream through a live server, score
  offline --asr .. --diar .. --clip X | --ami 64 [--diar-spks K --diar-pool max --diar-left 1 --shed 0|1
          --labels legacy|registry --embed spk|titanet --out JSON --budget S]   the Session in-process (resumable)
  score   --log events.json --ref clip.json                                   re-score a saved event log

A clip is ``<stem>.wav`` (16 kHz mono) + ``<stem>.json``: {"n_spk", "speakers": [names], "turns": [{"spk": k,
"start", "end"}], "frames": path of an (T, n_spk) 0/1 .npy at 80 ms}. Scores per run: distinct speaker ids in the
finals, speaker-count error, per-final speaker accuracy after a Hungarian map (emitted id -> reference speaker),
per-final purity (share of the span's speech that is its dominant reference speaker), shed / overloaded events,
and, offline, the pooled frame DER of the diarizer rows the session actually used (Hungarian over all columns).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SR = 16000
FRAME = 0.08
SCRATCH = Path(os.environ.get("DIAR_SCRATCH", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/diar"))


# --------------------------------------------------------------------------- references
def frames_of(turns, T, n_spk):
    y = np.zeros((T, n_spk), np.float32)
    for t in turns:
        a, b = max(0, int(t["start"] / FRAME)), min(T, int(math.ceil(t["end"] / FRAME - 1e-9)))
        if b > a:
            y[a:b, t["spk"]] = 1
    return y


def write_clip(stem: Path, audio: np.ndarray, turns, speakers, extra=None):
    import soundfile as sf
    stem.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(stem) + ".wav", audio, SR, subtype="PCM_16")
    T = int(math.ceil(len(audio) / SR / FRAME))
    y = frames_of(turns, T, len(speakers))
    np.save(str(stem) + ".ref.npy", y)
    ref = {"n_spk": len(speakers), "speakers": list(speakers), "turns": turns, "frames": str(stem) + ".ref.npy",
           "seconds": round(len(audio) / SR, 2), "active_speakers": int((y.sum(0) >= 1 / FRAME).sum()),
           **(extra or {})}
    Path(str(stem) + ".json").write_text(json.dumps(ref, indent=1))
    print(f"[clip] {stem}: {ref['seconds']} s, {ref['n_spk']} speakers ({ref['active_speakers']} with >= 1 s), "
          f"{len(turns)} turns, overlap frames {int((y.sum(1) > 1).sum())}", flush=True)
    return ref


def load_clip(path: str):
    import soundfile as sf
    stem = str(path)[:-5] if str(path).endswith(".json") else str(path)[:-4] if str(path).endswith(".wav") else str(path)
    ref = json.loads(Path(stem + ".json").read_text())
    x, sr = sf.read(stem + ".wav", dtype="float32")
    assert sr == SR
    if x.ndim > 1:
        x = x.mean(1)
    ref["frames_np"] = np.load(ref["frames"])
    ref["stem"] = stem
    return x.astype(np.float32), ref


# --------------------------------------------------------------------------- mix
def cmd_mix(a):
    from audioforge.datasets import librispeech as L
    import soundfile as sf
    rng = random.Random(a.seed)
    utts = L.scan("test-clean")
    root = L._root(None)
    by_spk = {}
    for u in utts:
        if 2.0 * SR <= u["n"] <= 12.0 * SR:
            by_spk.setdefault(u["speaker_id"], []).append(u)
    spks = sorted(s for s, us in by_spk.items() if len(us) >= 6)
    chosen = rng.sample(spks, a.n_spk)
    pools = {s: rng.sample(by_spk[s], len(by_spk[s])) for s in chosen}
    # speaker sequence: every speaker appears in the first n_spk turns (arrival order = index), then a shuffled
    # round robin with no immediate repetition, so everyone re-enters after the others spoke
    seq = list(range(a.n_spk))
    while len(seq) < a.turns:
        order = list(range(a.n_spk))
        rng.shuffle(order)
        if order[0] == seq[-1]:
            order.append(order.pop(0))
        seq += order
    seq = seq[: a.turns]
    turns, pieces, t = [], [], 0.0
    for k, s in enumerate(seq):
        u = pools[chosen[s]].pop(0)
        x, _ = sf.read(str(root / u["path"]), dtype="float32")
        dur = rng.uniform(2.0, 5.0)
        x = x[: int(dur * SR)]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-8) * 0.05  # equal loudness, -26 dBFS RMS
        if k > 0:
            gap = -rng.uniform(0.3, 0.8) if rng.random() < a.p_overlap else rng.uniform(0.3, 1.2)
            t = max(0.0, t + gap)
        turns.append({"spk": s, "start": round(t, 3), "end": round(t + len(x) / SR, 3), "utt": u["id"]})
        pieces.append((int(t * SR), x))
        t = t + len(x) / SR
    n = int(math.ceil((t + 1.0) * SR))
    mix = np.zeros(n, np.float32)
    for s0, x in pieces:
        mix[s0: s0 + len(x)] += x
    peak = np.abs(mix).max()
    if peak > 0.95:
        mix *= 0.95 / peak
    stem = Path(a.out) / f"mix{a.n_spk}_s{a.seed}"
    write_clip(stem, mix, turns, [f"libri-{s}" for s in chosen], {"source": "librispeech test-clean",
                                                                     "p_overlap": a.p_overlap})


# --------------------------------------------------------------------------- real meeting window
def cmd_clip(a):
    if a.corpus == "icsi":
        from audioforge.datasets.icsi import ICSI as DS
    else:
        from audioforge.datasets.ami import AMI as DS
    ds = DS([a.meeting], verbose=False)
    m = a.meeting
    acts = ds.acts[m]
    dur = ds.duration(m)
    best = None
    for s0 in np.arange(0.0, dur - a.dur, 5.0):
        s1 = s0 + a.dur
        sec = {s: sum(min(e, s1) - max(b, s0) for b, e in iv if e > s0 and b < s1) for s, iv in acts.items()}
        n = sum(1 for v in sec.values() if v >= a.min_sec)
        speech = sum(sec.values())
        key = (n, speech)
        if best is None or key > best[0]:
            best = (key, s0)
    (n, speech), s0 = best
    if n < a.min_spk:
        sys.exit(f"no {a.dur} s window of {m} with {a.min_spk} speakers speaking >= {a.min_sec} s (best {n})")
    s1 = s0 + a.dur
    x = np.asarray(ds._clip(m, s0, s1), np.float32)
    order = sorted(acts, key=lambda s: min([b for b, e in acts[s] if e > s0 and b < s1] or [1e9]))
    order = [s for s in order if any(e > s0 and b < s1 for b, e in acts[s])]
    idx = {s: i for i, s in enumerate(order)}
    turns = []
    for tr in ds.turns[m]:
        if tr["end"] > s0 and tr["start"] < s1 and tr["speaker"] in idx:
            turns.append({"spk": idx[tr["speaker"]], "start": round(max(0.0, tr["start"] - s0), 3),
                          "end": round(min(a.dur, tr["end"] - s0), 3)})
    # frame references from the word-level activity (as the AMI/ICSI DER labels), not from the coarse turns
    T = int(math.ceil(len(x) / SR / FRAME))
    y = np.zeros((T, len(order)), np.float32)
    for s in order:
        for b, e in acts[s]:
            if e > s0 and b < s1:
                fa, fb = max(0, int((b - s0) / FRAME)), min(T, int(math.ceil((e - s0) / FRAME - 1e-9)))
                if fb > fa:
                    y[fa:fb, idx[s]] = 1
    stem = Path(a.out) / f"{a.corpus}_{m}_{int(s0)}s"
    import soundfile as sf
    stem.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(stem) + ".wav", x, SR, subtype="PCM_16")
    np.save(str(stem) + ".ref.npy", y)
    ref = {"n_spk": len(order), "speakers": order, "turns": turns, "frames": str(stem) + ".ref.npy",
           "seconds": round(len(x) / SR, 2), "active_speakers": int((y.sum(0) >= 1 / FRAME).sum()),
           "source": f"{a.corpus} {m} [{s0:.0f}, {s1:.0f}) s", "window_start_s": float(s0)}
    Path(str(stem) + ".json").write_text(json.dumps(ref, indent=1))
    print(f"[clip] {stem}: {ref['n_spk']} speakers ({ref['active_speakers']} >= 1 s), {len(turns)} turns, "
          f"speech {speech:.0f} s, overlap frames {int((y.sum(1) > 1).sum())}", flush=True)


# --------------------------------------------------------------------------- scoring
def hungarian_der(pred: np.ndarray, ref: np.ndarray, thr: float = 0.5):
    """Pooled frame DER parts (miss, fa, conf, speech) with the best injective column map (exact: the matched term
    is separable per pair, so the assignment problem gives the same optimum as enumerating permutations)."""
    from scipy.optimize import linear_sum_assignment
    T = min(len(pred), len(ref))
    p = (np.asarray(pred[:T]) > thr).astype(np.float64)
    r = np.asarray(ref[:T], np.float64)
    if p.shape[1] < r.shape[1]:
        p = np.pad(p, ((0, 0), (0, r.shape[1] - p.shape[1])))
    elif r.shape[1] < p.shape[1]:
        r = np.pad(r, ((0, 0), (0, p.shape[1] - r.shape[1])))
    ov = r.T @ p  # (S_ref, S_hyp) matched frames per pair
    ri, ci = linear_sum_assignment(-ov)
    correct = ov[ri, ci].sum()
    n_ref, n_hyp = r.sum(1), p.sum(1)
    err = np.maximum(n_ref, n_hyp).sum() - correct
    miss = np.clip(n_ref - n_hyp, 0, None).sum()
    fa = np.clip(n_hyp - n_ref, 0, None).sum()
    return float(miss), float(fa), float(err - miss - fa), float(r.sum())


def count_speakers(mat: np.ndarray, thr: float = 0.5, min_s: float = 1.0) -> int:
    return int(((np.asarray(mat) > thr).sum(0) >= min_s / FRAME).sum())


def score_events(msgs: list[dict], ref: dict, rows: np.ndarray | None = None) -> dict:
    """Decision-level scores of one session against the clip reference."""
    from scipy.optimize import linear_sum_assignment
    y = ref["frames_np"]
    T, n_spk = y.shape
    finals, prev_t, errors, stats = [], 0.0, [], None
    for m in msgs:
        if m["type"] == "final":
            if m.get("source") not in (None, "stream"):
                continue
            a, b = int(prev_t / FRAME), min(T, int(math.ceil(m["t"] / FRAME)))
            span = y[a:b]
            speech = span.sum(0)
            gt = int(speech.argmax()) if speech.sum() > 0 else None
            purity = float(speech.max() / speech.sum()) if speech.sum() > 0 else None
            finals.append({"t": m["t"], "t0": round(prev_t, 3), "speaker": m.get("speaker"), "text": m.get("text", ""),
                           "gt": gt, "purity": purity, "n_gt": int((speech >= 1 / FRAME).sum()),
                           "conf": m.get("speaker_conf"), "shed": m.get("diar_shed")})
            prev_t = m["t"]
        elif m["type"] == "error":
            errors.append({"t": m.get("_t"), "code": m["code"], "detail": m["detail"][:120]})
        elif m["type"] == "stats":
            stats = m
    scored = [f for f in finals if f["gt"] is not None and f["text"].strip()]
    ids = sorted({f["speaker"] for f in scored if f["speaker"] is not None})
    gts = sorted({f["gt"] for f in scored})
    acc = None
    if scored and ids:
        C = np.zeros((len(ids), len(gts)))
        for f in scored:
            if f["speaker"] is not None:
                C[ids.index(f["speaker"]), gts.index(f["gt"])] += 1
        ri, ci = linear_sum_assignment(-C)
        acc = float(C[ri, ci].sum() / len(scored))
        mapping = {ids[i]: gts[j] for i, j in zip(ri, ci)}
    else:
        mapping = {}
    # how many distinct emitted ids each reference speaker's turns received (column permutation / merging)
    per_gt = {}
    for f in scored:
        per_gt.setdefault(f["gt"], []).append(f["speaker"])
    n_active = int((y.sum(0) >= 1 / FRAME).sum())
    out = {"n_finals": len(finals), "n_scored": len(scored), "ids_emitted": ids, "n_ids": len(ids),
           "n_ref_speakers": n_active, "count_error": len(ids) - n_active,
           "final_speaker_acc": None if acc is None else round(acc, 3),
           "null_speaker_finals": sum(1 for f in scored if f["speaker"] is None),
           "purity_mean": round(float(np.mean([f["purity"] for f in scored])), 3) if scored else None,
           "finals_multi_speaker": sum(1 for f in scored if f["n_gt"] >= 2),
           "ids_per_ref_speaker": {int(k): sorted({str(v) for v in vs}) for k, vs in sorted(per_gt.items())},
           "map": {str(k): int(v) for k, v in mapping.items()},
           "errors": errors[:20], "n_overloaded": sum(1 for e in errors if e["code"] == "overloaded"),
           "degraded": (stats or {}).get("degraded"), "rtf": (stats or {}).get("rtf"),
           "speakers_seen": (stats or {}).get("speakers_seen"), "finals": finals}
    prims = [it["primary"] for m in msgs for it in ([m] if m["type"] == "frame" else m.get("items", []))
             if m["type"] in ("frame", "frames")]
    out["primaries"] = sorted({p for p in prims if p is not None})
    if rows is not None and len(rows):
        miss, fa, conf, speech = hungarian_der(rows, y)
        out["der"] = {"der": round((miss + fa + conf) / max(speech, 1), 4), "miss": round(miss / max(speech, 1), 4),
                      "fa": round(fa / max(speech, 1), 4), "conf": round(conf / max(speech, 1), 4)}
        out["der_speech"] = speech
        out["diar_count"] = count_speakers(rows)
        out["diar_count_error"] = out["diar_count"] - n_active
    return out


def brief(s: dict) -> str:
    return (f"finals {s['n_scored']}/{s['n_finals']} ids {s['ids_emitted']} (ref {s['n_ref_speakers']}, err "
            f"{s['count_error']:+d}) acc {s['final_speaker_acc']} purity {s['purity_mean']} null {s['null_speaker_finals']} "
            f"overloaded {s['n_overloaded']} degraded {s['degraded']} rtf {s['rtf']}"
            + (f" DER {s['der']['der']} (m {s['der']['miss']} fa {s['der']['fa']} c {s['der']['conf']}) diar_count "
               f"{s['diar_count']} ({s['diar_count_error']:+d})" if "der" in s else ""))


# --------------------------------------------------------------------------- live server
async def _session(url: str, x: np.ndarray, speed: float, policy: str, timeout_ms: int, tag: str):
    import websockets
    msgs = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config", "turn_policy": policy, "timeout_ms": timeout_ms}))
        block = int(SR * 0.02)
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        done = asyncio.Event()

        async def rx():
            try:
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    m["_wall"] = time.perf_counter()
                    msgs.append(m)
                    if m.get("type") == "stats":
                        break
            finally:
                done.set()

        task = asyncio.create_task(rx())
        t0 = time.perf_counter()
        for i in range(0, len(pcm), block * 2):
            if done.is_set():
                break
            await ws.send(pcm[i: i + block * 2])
            if speed > 0:
                due = t0 + (i / 2 + block) / SR / speed
                dt = due - time.perf_counter()
                if dt > 0:
                    await asyncio.sleep(dt)
        if not done.is_set():
            await ws.send(json.dumps({"type": "end"}))
        try:
            await asyncio.wait_for(done.wait(), 120)
        except asyncio.TimeoutError:
            pass
        task.cancel()
    t_sent0 = t0
    for m in msgs:  # audio-time stamp of each error message (its arrival relative to the audio clock)
        if m.get("type") == "error":
            m["_t"] = round((m["_wall"] - t_sent0) * speed, 2) if speed > 0 else None
    print(f"[run] {tag}: {len(msgs)} messages, {sum(1 for m in msgs if m['type'] == 'final')} finals", flush=True)
    return msgs


def cmd_run(a):
    x, ref = load_clip(a.clip)
    x2, ref2 = load_clip(a.clip2) if a.sessions > 1 and a.clip2 else (x, ref)

    async def go():
        tasks = [_session(a.url, x if i == 0 else x2, a.speed, a.policy, a.timeout_ms, f"s{i}")
                 for i in range(a.sessions)]
        return await asyncio.gather(*tasks)

    logs = asyncio.run(go())
    res = {"clip": ref["stem"], "clip2": ref2["stem"], "url": a.url, "speed": a.speed, "sessions": a.sessions,
           "policy": a.policy, "load": os.getloadavg(), "sessions_scored": []}
    for i, msgs in enumerate(logs):
        s = score_events(msgs, ref if i == 0 else ref2)
        res["sessions_scored"].append(s)
        print(f"[score] s{i}: {brief(s)}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**res, "events": [[{k: v for k, v in m.items() if k != '_wall'}
                                                                for m in msgs] for msgs in logs]}, default=float))
        print(f"[run] wrote {a.out}")


def cmd_score(a):
    d = json.loads(Path(a.log).read_text())
    x, ref = load_clip(a.ref or d["clip"])
    x2, ref2 = load_clip(a.ref2 or d.get("clip2") or (a.ref or d["clip"]))
    for i, msgs in enumerate(d["events"]):
        sc = score_events(msgs, ref if i == 0 else ref2)
        print(f"[score] s{i}: {brief(sc)}")
        if a.finals:
            for f in sc["finals"]:
                print(f"    {f['t0']:7.2f}-{f['t']:7.2f} spk {f['speaker']!s:>4} gt {f['gt']!s:>4} purity "
                      f"{'-' if f['purity'] is None else round(f['purity'], 2)} n_gt {f['n_gt']} "
                      f"{('conf ' + str(f['conf'])) if f['conf'] is not None else ''}{'shed ' if f['shed'] else ''}"
                      f"'{f['text'][:60]}'")


# --------------------------------------------------------------------------- offline (in-process Session)
def _engine(a):
    import torch
    import audioforge.serve as S
    torch.set_num_threads(a.threads)
    kw = dict(threads=a.threads, diar_left=a.diar_left, debug=True)
    if a.labels != "legacy" or a.shed_diar != "vad" or a.embed:
        kw.update(diar_labels=a.labels, shed_diar=a.shed_diar)
        if a.embed:
            kw["diar_embed"] = a.embed
        if a.titanet:
            kw["titanet"] = a.titanet
    eng = S.Engine.load(a.asr, a.diar, "cpu", diar_pool=a.diar_pool, diar_spks=a.diar_spks, **kw)
    return eng


def _run_session(eng, x: np.ndarray, shed: int, timeout_ms: int, block: int = 2560, policy: str = "timeout"):
    import audioforge.serve as S
    s = S.Session(eng, S.SessionConfig(turn_policy=policy, timeout_ms=timeout_ms))
    msgs, t0 = [], time.perf_counter()
    for i in range(0, len(x), block):
        msgs += s.process(x[i: i + block], shed=shed, backlog_ms=0.0)
    msgs += s.finish()
    wall = time.perf_counter() - t0
    rows = np.stack([s.rows[v] for v in range(max(0, len(s.rows) - len(s.rows.d)), len(s.rows))]) if len(s.rows) else None
    return msgs, rows, wall


def cmd_offline(a):
    import audioforge.serve as S
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() and a.resume else {"items": {}}
    res.update(asr=a.asr, diar=a.diar, diar_spks=a.diar_spks, diar_pool=a.diar_pool, labels=a.labels,
               shed=a.shed, shed_diar=a.shed_diar, embed=a.embed, threads=a.threads, policy=a.policy,
               load=os.getloadavg())
    if a.shed:
        S.SHED_RTF = 0.0  # enter the shedding level regardless of the (unpaced) recent RTF
    t_start = time.time()
    eng = _engine(a)
    items = []
    if a.clip:
        for c in a.clip:
            x, ref = load_clip(c)
            items.append((Path(ref["stem"]).name, x, ref))
    if a.ami:
        from audioforge.datasets.ami import AMI, DEFAULT_MEETINGS
        ds = AMI(DEFAULT_MEETINGS["dev"], verbose=False)
        data = ds.diar(20.0, 10.0, max_spks=8)
        idx = sorted(random.Random(0).sample(range(len(data)), a.ami)) if a.ami < len(data) else range(len(data))
        for i in idx:
            d = data[i]
            ref = {"frames_np": d["spk_targets"], "n_spk": d["spk_targets"].shape[1]}
            items.append((f"ami_{d['meeting']}_{d['start']:.0f}", np.asarray(d["audio"], np.float32), ref))
    done = 0
    for name, x, ref in items:
        if name in res["items"]:
            continue
        if time.time() - t_start > a.budget:
            print(f"[offline] budget reached after {done} new items; rerun with --resume", flush=True)
            break
        msgs, rows, wall = _run_session(eng, x, a.shed, a.timeout_ms, policy=a.policy)
        s = score_events(msgs, ref, rows)
        s["wall_rtf"] = round(wall / (len(x) / SR), 3)
        s.pop("finals", None) if a.ami else None
        res["items"][name] = s
        done += 1
        print(f"[offline] {name}: {brief(s)} wall-rtf {s['wall_rtf']}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, default=float))
    its = list(res["items"].values())
    if its:
        agg = {"n": len(its), "wall_rtf_mean": round(float(np.mean([i["wall_rtf"] for i in its])), 3),
               "count_error_mean": round(float(np.mean([i["diar_count_error"] for i in its if "diar_count_error" in i])), 3),
               "count_abs_error_mean": round(float(np.mean([abs(i["diar_count_error"]) for i in its if "diar_count_error" in i])), 3),
               "final_ids_count_error_mean": round(float(np.mean([i["count_error"] for i in its])), 3),
               "final_speaker_acc_mean": round(float(np.mean([i["final_speaker_acc"] for i in its
                                                              if i["final_speaker_acc"] is not None] or [float("nan")])), 3),
               "null_speaker_finals": int(sum(i["null_speaker_finals"] for i in its)),
               "n_scored_finals": int(sum(i["n_scored"] for i in its)),
               "shed_diar_frames": int(sum((i.get("degraded") or {}).get("shed_diar_frames", 0) for i in its))}
        if all("der" in i for i in its):  # pooled DER = speech-weighted mean of the per-item rates (exact)
            w = np.array([i.get("der_speech", 1.0) for i in its])
            for k in ("der", "miss", "fa", "conf"):
                agg[f"pooled_{k}"] = round(float(np.average([i["der"][k] for i in its], weights=w)), 4)
        res["aggregate"] = agg
        out.write_text(json.dumps(res, default=float))
        print(f"[offline] aggregate {json.dumps(agg)}", flush=True)


# --------------------------------------------------------------------------- registry threshold calibration
def cmd_calib(a):
    """Same- vs different-speaker cosines of per-turn embeddings on a clip's reference turns: the served speaker
    head (block-4 frames of the turn, tsvad_stream.voiceprint) and TitaNet-L on the turn's audio."""
    import torch
    from audioforge.train import load_model
    from audioforge.tsvad_stream import voiceprint
    from audioforge.enrollment import TitaNetEmbedder
    torch.set_num_threads(a.threads)
    x, ref = load_clip(a.clip)
    model = load_model(a.asr, "cpu")
    tita = TitaNetEmbedder(path=a.titanet) if a.titanet != "none" else None
    embs = {"spk": [], "titanet": []}
    spk = []
    for t in ref["turns"]:
        seg = x[int(t["start"] * SR): int(t["end"] * SR)]
        if len(seg) < SR:
            continue
        spk.append(t["spk"])
        embs["spk"].append(voiceprint(model, seg))
        if tita is not None:
            embs["titanet"].append(tita.frames(seg, [np.arange(len(seg) // 1280)])[0])
    spk = np.array(spk)
    out = {"clip": ref["stem"], "n_turns": len(spk)}
    for k, es in embs.items():
        if not es:
            continue
        E = np.stack(es)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        C = E @ E.T
        iu = np.triu_indices(len(E), 1)
        same = C[iu][spk[iu[0]] == spk[iu[1]]]
        diff = C[iu][spk[iu[0]] != spk[iu[1]]]
        # the registry's operating rule: a turn joins the closest known speaker if cos >= thr; sweep thr for the
        # per-turn accuracy of a sequential registry over the clip's turn order
        from audioforge.speaker_registry import SpeakerRegistry
        best = None
        sweep = {}
        for thr in np.arange(0.1, 0.9, 0.05):
            reg = SpeakerRegistry(float(thr))
            ids = [reg.assign(e)[0] for e in E]
            from scipy.optimize import linear_sum_assignment
            M = np.zeros((max(ids) + 1, spk.max() + 1))
            for i, g in zip(ids, spk):
                M[i, g] += 1
            r, c = linear_sum_assignment(-M)
            acc = M[r, c].sum() / len(ids)
            sweep[round(float(thr), 2)] = {"acc": round(float(acc), 3), "n_ids": len(reg)}
            if best is None or (acc, -abs(len(reg) - ref["n_spk"])) > best[0]:
                best = ((acc, -abs(len(reg) - ref["n_spk"])), float(thr))
        out[k] = {"same_mean": round(float(same.mean()), 3), "same_p10": round(float(np.percentile(same, 10)), 3),
                  "diff_mean": round(float(diff.mean()), 3), "diff_p90": round(float(np.percentile(diff, 90)), 3),
                  "diff_max": round(float(diff.max()), 3), "best_thr": round(best[1], 2), "sweep": sweep}
        print(f"[calib] {k}: same {out[k]['same_mean']} (p10 {out[k]['same_p10']}) diff {out[k]['diff_mean']} "
              f"(p90 {out[k]['diff_p90']}, max {out[k]['diff_max']}) best thr {out[k]['best_thr']} "
              f"acc {sweep[round(best[1], 2)]}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mix")
    p.add_argument("--out", default=str(SCRATCH / "clips"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-spk", type=int, default=6)
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--p-overlap", type=float, default=0.25)
    p.set_defaults(fn=cmd_mix)
    p = sub.add_parser("clip")
    p.add_argument("--corpus", choices=["icsi", "ami"], default="icsi")
    p.add_argument("--meeting", default="Bmr021")
    p.add_argument("--out", default=str(SCRATCH / "clips"))
    p.add_argument("--dur", type=float, default=60.0)
    p.add_argument("--min-spk", type=int, default=4)
    p.add_argument("--min-sec", type=float, default=3.0)
    p.set_defaults(fn=cmd_clip)
    p = sub.add_parser("run")
    p.add_argument("--url", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--clip2", default=None, help="audio of the second and later sessions (default: the same clip)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--policy", default="timeout")
    p.add_argument("--timeout-ms", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score")
    p.add_argument("--log", required=True)
    p.add_argument("--ref", default=None)
    p.add_argument("--ref2", default=None)
    p.add_argument("--finals", action="store_true")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("offline")
    p.add_argument("--asr", default=str(ROOT / "runs/stage1_served.afm"))
    p.add_argument("--diar", default=str(ROOT / "runs/nemo_nemotron3_diar.afm"))
    p.add_argument("--diar-spks", type=int, default=None)
    p.add_argument("--diar-pool", default=None)
    p.add_argument("--diar-left", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--clip", action="append", default=[])
    p.add_argument("--ami", type=int, default=0)
    p.add_argument("--shed", type=int, default=0)
    p.add_argument("--shed-diar", default="vad")
    p.add_argument("--labels", default="legacy")
    p.add_argument("--embed", default=None)
    p.add_argument("--titanet", default=None)
    p.add_argument("--timeout-ms", type=int, default=1000)
    p.add_argument("--policy", default="timeout")
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(fn=cmd_offline)
    p = sub.add_parser("calib")
    p.add_argument("--clip", required=True)
    p.add_argument("--asr", default=str(ROOT / "runs/stage1_served.afm"))
    p.add_argument("--titanet", default=str(ROOT / "data/nemo/speakerverification_en_titanet_large.nemo"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_calib)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
