"""research/IMPROVEMENTS.md section 4: how good must the TS-VAD voice print be? Print length and live prints.

On the eot-bench v2 windows (AMI dev 974, ICSI held-out 1312; the primary is the target), per window a TS-VAD track
[P(target), P(other)] from the served head (runs/tsvad_spk.pt) with the print taken by one of these rules:

  offline prints (the product holds a stored print): L = 1.5 / 3 / 5 / 10 s of the primary's single-speaker speech
      from elsewhere in the meeting (segments overlapping the window +-2 s excluded; tsvad.clip_from with the bank
      seeds of stage_vprints, so the 5.0 / 1.5 s rows reproduce A.1), embedded from a separate block-4 pass
      -> track names vp1p5, vp3p0, vp5p0 (== A.2's tsvad_spk), vp10p0
  live arm, per turn (``--enroll after_agent_arm`` without memory): armed at the last other-speaker speech end before
      the primary's onset inside the window (window start if none); the print is the first L s of speech after it
      (speech = the head's own no-print output P > 0.5, the plain-VAD mode; frames of anyone who talks then are
      taken, as the server would), embedded from the stream's block-4 frames; no print (plain VAD) before
      -> arm<L>
  live arm, session memory: the print was armed the same way at the primary's PREVIOUS turn in the meeting (its
      first L s of speech after the preceding other-speaker end; the audio before the window, contaminated or not)
      and is in use from the window start -> mem<L>
  memory + refresh: mem<L> plus in-window refresh (TSVADTrack refresh every 2 s of confident target speech over the
      last L s) -> memref<L>

Frame metrics (F1 / miss / FA of P(target) against the primary's labels) come from ``frame``; turn-level misses come
from running the turn heads on these tracks (tsvad_turn.py scores --track <name>, report).

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/tsvad_enroll.py tracks --budget 540   # repeat
  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/tsvad_enroll.py frame
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import tsvad as T  # noqa: E402

WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/tsvad_enroll")
OUT = ROOT / "runs" / "tsvad_enroll.json"
LENS = (1.5, 3.0, 5.0, 10.0)
LIVE_LENS = (1.5, 3.0, 5.0)


def log(*a):
    print(*a, flush=True)


def tname(prefix, L):
    return f"{prefix}{L:.1f}".replace(".", "p")


def arm_frame(v) -> int:
    """Last other-speaker speech end (label frames) at or before the primary's onset, else 0."""
    y = np.asarray(v["spk_targets"])
    on = int(v["onset_frame"])
    if y.shape[1] < 2:
        return 0
    oth = y[: on + 1, 1:].max(1) > 0.5
    idx = np.nonzero(oth)[0]
    return int(idx[-1] + 1) if len(idx) else 0


def live_track(head, spk_head, feats, arm, L, e0=None, refresh_s=0.0):
    """Offline replica of serve's TSVADTrack: arm at frame ``arm`` (None = never), print from the first L s of
    speech after it (speech = the head's no-print P(target) > 0.5 on the same frames), optional initial print e0."""
    from audioforge.tsvad_stream import TSVADTrack
    null = T.tsvad_probs(head, feats, None)[:, 0]
    tr = TSVADTrack(head, spk_head, print_s=L, refresh_s=refresh_s)
    if e0 is not None:
        tr.set_print(e0)
    if arm is not None:
        tr.arm(arm)
    return tr.feed(torch.as_tensor(feats)[None], null), tr


def prev_turn_audio_ivs(ds, v, L):
    """The primary's previous turn in the meeting before the window: (arm time, [its speech from then on]) where the
    arm is the last other-speaker speech end before that turn's onset; None if the primary has no earlier turn."""
    m = v["meeting"]
    spk = ds.speaker_ids[int(v["speaker"]) - getattr(ds, "speaker_offset", 0)]
    a0 = float(v["start"])
    own = sorted(iv for iv in ds.acts[m][spk] if iv[1] <= a0 - 0.5)
    if not own:
        return None
    # the start of the last own turn: walk back over own segments separated by < 1 s
    j = len(own) - 1
    while j > 0 and own[j][0] - own[j - 1][1] < 1.0:
        j -= 1
    onset = own[j][0]
    oth = sorted(iv for o, ivs in ds.acts[m].items() if o != spk for iv in ivs if iv[1] <= onset + 1e-6)
    arm = max([e for _, e in oth], default=max(0.0, onset - 2.0))
    return arm, min(a0 - 0.1, arm + max(4 * L, L + 6.0))


def stage_tracks(a):
    T.set_threads(2)
    model = T.load_served()
    head, _ = T.load_head(ROOT / "runs" / "tsvad_spk.pt")
    spk_head = model.heads["spk"]
    t0 = time.time()
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        kidx, VP = T.load_vprints(corpus)
        bank_path = WORK / corpus / "bank.npz"
        names = [tname("vp", L) for L in LENS] + [tname(p, L) for p in ("arm", "mem", "memref") for L in LIVE_LENS]
        for n in names:
            (WORK / corpus / f"track_{n}").mkdir(parents=True, exist_ok=True)
        # prints of lengths 3 and 10 s (1.5 / 5 s: A.1's stored vprints), same bank rule as stage_vprints
        if bank_path.exists():
            bank = dict(np.load(bank_path))
        else:
            segs_m, bank = {}, {"keys": [], "vp3p0": [], "vp10p0": [], "has3p0": [], "has10p0": []}
            for v in ext:
                m = v["meeting"]
                if m not in segs_m:
                    segs_m[m] = T.single_segments(ds, m)
                s = T.window_speakers(ds, v)[0]
                a0, b0 = float(v["start"]), float(v["start"]) + len(v["audio"]) / T.SR
                for L in (3.0, 10.0):
                    e, ok = np.zeros(192, np.float32), False
                    for i in range(T.VP_BANK):
                        ivs = T.clip_from(segs_m[m].get(s, []), L, random.Random(f"{m}_{s}_{L}_{i}"))
                        if ivs is None or any(x < b0 + 2 and y > a0 - 2 for x, y in ivs):
                            continue
                        e = T.spk_embed(model, T.block_feats(model, [T.clip_audio(ds, m, ivs)]))[0]
                        ok = True
                        break
                    bank[tname("vp", L)].append(e)
                    bank[f"has{L:.1f}".replace(".", "p")].append(ok)
                bank["keys"].append(T.wkey(v))
                if time.time() - t0 > a.budget:
                    log(f"  {corpus}: bank not finished in budget ({len(bank['keys'])}/{len(ext)}); rerun")
                    return
            bank = {k: np.array(x) for k, x in bank.items()}
            T.save_npz(bank_path, **bank)
            log(f"  {corpus}: bank of 3 / 10 s prints ({time.time() - t0:.0f}s)")
        bidx = {str(k): i for i, k in enumerate(bank["keys"])}
        done = 0
        for v in ext:
            key = T.wkey(v)
            if all((WORK / corpus / f"track_{n}" / f"{key}.npy").exists() for n in names):
                continue
            if time.time() - t0 > a.budget:
                break
            f = T.win_feats(corpus, v)
            k, b = kidx[key], bidx[key]
            prints = {1.5: (VP["spk_1p5"][k, 0], bool(VP["has_1p5"][k, 0])),
                      5.0: (VP["spk_5p0"][k, 0], bool(VP["has_5p0"][k, 0])),
                      3.0: (bank["vp3p0"][b], bool(bank["has3p0"][b])), 10.0: (bank["vp10p0"][b], bool(bank["has10p0"][b]))}
            for L, (e, ok) in prints.items():
                np.save(WORK / corpus / f"track_{tname('vp', L)}" / f"{key}.npy",
                        T.tsvad_probs(head, f, e if ok else None).astype(np.float16))
            arm = arm_frame(v)
            pv = prev_turn_audio_ivs(ds, v, max(LIVE_LENS))
            pfe = None
            if pv is not None:
                aud = np.asarray(ds._clip(v["meeting"], pv[0], pv[1]), np.float32)
                pfe = T.block_feats(model, [aud])[0] if len(aud) > T.SR else None
            for L in LIVE_LENS:
                p_arm, _ = live_track(head, spk_head, f, arm, L)
                np.save(WORK / corpus / f"track_{tname('arm', L)}" / f"{key}.npy", p_arm.astype(np.float16))
                e0 = None
                if pfe is not None:
                    _, tr = live_track(head, spk_head, pfe, 0, L)
                    e0 = tr.print
                p_mem, _ = live_track(head, spk_head, f, None, L, e0)
                np.save(WORK / corpus / f"track_{tname('mem', L)}" / f"{key}.npy", p_mem.astype(np.float16))
                p_ref, _ = live_track(head, spk_head, f, None if e0 is not None else arm, L, e0, refresh_s=2.0)
                np.save(WORK / corpus / f"track_{tname('memref', L)}" / f"{key}.npy", p_ref.astype(np.float16))
            done += 1
        left = sum(1 for v in ext if not all((WORK / corpus / f"track_{n}" / f"{T.wkey(v)}.npy").exists() for n in names))
        log(f"  {corpus} tracks: {done} done, {left} left ({time.time() - t0:.0f}s)")
        if left:
            return


def stage_frame(a):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    names = [tname("vp", L) for L in LENS] + [tname(p, L) for p in ("arm", "mem", "memref") for L in LIVE_LENS]
    for corpus in a.corpora.split(","):
        ext, meta, ds = T.bench_windows(corpus)
        r = {}
        for n in names:
            P, Y = [], []
            for v in ext:
                p = np.load(WORK / corpus / f"track_{n}" / f"{T.wkey(v)}.npy").astype(np.float32)
                y = np.asarray(v["spk_act"], np.float32)
                t = min(len(p), len(y))
                P.append(p[:t, 0])
                Y.append(y[:t])
            pr, rc, f1, fa = T.frame_prf(np.concatenate(P), np.concatenate(Y))
            r[n] = {"f1": round(f1, 4), "precision": round(pr, 4), "miss": round(1 - rc, 4), "fa": round(fa, 4)}
            log(f"  {corpus} {n:10s} F1 {f1:.3f} miss {1 - rc:.3f} FA {fa:.3f}")
        res.setdefault("frame", {})[corpus] = r
        OUT.write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage")
    p.add_argument("--budget", type=float, default=540)
    p.add_argument("--corpora", default="ami,icsi")
    a = p.parse_args()
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
