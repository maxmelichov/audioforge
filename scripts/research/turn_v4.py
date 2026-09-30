"""research/TURN_V4.md: audible-end turn labels, and a turn head trained to be confident fast (heads only, on cached
served features). Evaluation goes through scripts/research/eot_latency.py unchanged (EOT_DUMP / EOT_LABELS / EOT_OUT).

  labels_eval   audible-end reference labels for the eot_latency sessions (calls + AMI) from the dumped served VAD
                head and Silero track -> runs/turn_v4_labels_eval.json (+ the audit in runs/turn_v4.json "labels")

Audible end of a labelled turn [s, e]: the end of the last 80 ms frame with served VAD > 0.5 or the last 32 ms chunk
with Silero > 0.5 that overlaps [s, e] (the later of the two, clipped to e); e itself when neither hears speech there.
An end only moves earlier, never later.

    PYTHONPATH=. .venv/bin/python scripts/research/turn_v4.py labels_eval
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

WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v4")
LABELS_EVAL = ROOT / "runs" / "turn_v4_labels_eval.json"
OUT = ROOT / "runs" / "turn_v4.json"
FRAME, CHUNK_S = 0.08, 512 / 16000
VAD_ON, SIL_ON = 0.5, 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def audible_end(s0: float, e1: float, vad: np.ndarray, conf: np.ndarray | None) -> float:
    """The audible end of a labelled turn [s0, e1] (module doc)."""
    ends = []
    v = np.nonzero(vad > VAD_ON)[0]
    v = v[((v + 1) * FRAME > s0) & (v * FRAME < e1)]
    if len(v):
        ends.append(min((v[-1] + 1) * FRAME, e1))
    if conf is not None:
        j = np.nonzero(conf > SIL_ON)[0]
        j = j[((j + 1) * CHUNK_S > s0) & (j * CHUNK_S < e1)]
        if len(j):
            ends.append(min((j[-1] + 1) * CHUNK_S, e1))
    return round(float(max(ends)), 4) if ends else e1


def move_stats(moved: list[float]) -> dict:
    m = np.asarray(moved)
    return {"n_ends": int(len(m)), "moved_any": int((m > 1e-6).sum()), "moved_gt_80ms": int((m > 0.08).sum()),
            "moved_gt_120ms": int((m > 0.12).sum()), "moved_gt_240ms": int((m > 0.24).sum()),
            "moved_gt_500ms": int((m > 0.5).sum()),
            "pct_moved_gt_120ms": round(100 * float(np.mean(m > 0.12)), 1) if len(m) else None,
            "moved_ms_p50_p75_p90_max": [int(round(1000 * float(np.percentile(m, q)))) for q in (50, 75, 90, 100)]
            if len(m) else None,
            "moved_ms_p50_of_moved": int(round(1000 * float(np.median(m[m > 1e-6])))) if (m > 1e-6).any() else 0}


def cmd_labels_eval(a):
    import eot_latency as E
    dump = E.load_dump()
    out, stats, diag = {}, {}, {}
    for ss in E.sessions():
        d = dump.get(ss["key"])
        if d is None:
            continue
        vad = np.zeros(max(d["head"]["v"]) + 1)
        vad[np.asarray(d["head"]["v"])] = d["head"]["vad"]
        conf = np.asarray(d["conf"], float) if d.get("conf") else None
        turns = []
        for i, (s0, e1) in enumerate(ss["user_turns"]):
            e2 = audible_end(s0, e1, vad, conf)
            turns.append([s0, e2])
            if ss["scored"][i]:
                for sc in {"ami" if ss["set"] == "ami" else "calls", ss["set"]}:
                    stats.setdefault(sc, []).append(e1 - e2)
                # the smart-turn audit's view (Silero alone at 0.7, Pipecat's start confidence): how far is the label
                # end past that detector's last speech in the turn
                if conf is not None:
                    j = np.nonzero(conf > 0.7)[0]
                    j = j[((j + 1) * CHUNK_S > s0) & (j * CHUNK_S < e1)]
                    if len(j):
                        diag.setdefault(ss["set"], []).append(e1 - min((j[-1] + 1) * CHUNK_S, e1))
        out[ss["key"]] = turns
    LABELS_EVAL.write_text(json.dumps(out))
    res = {"definition": __doc__.split("Audible end of a labelled turn")[1].split("\n\n")[0].strip(),
           "file": str(LABELS_EVAL.relative_to(ROOT)), "scored_ends": {k: move_stats(v) for k, v in stats.items()},
           "diag_silero_0.7_only": {k: move_stats(v) for k, v in diag.items()}}
    save("labels_eval", res)
    for k, v in {**res["scored_ends"], **{f"silero0.7:{k}": v for k, v in res["diag_silero_0.7_only"].items()}}.items():
        log(k, v)


# --------------------------------------------------------------------------- served turn-head inputs, offline
ATT = [70, 1]


class Extractor:
    """The served single-mode turn-head inputs of a clip, computed offline (research/TURN_V4.md "Features"):
    pass 1 (masked [70,1] forward) -> VAD head, block 4 -> the served TS-VAD track (``tsvad_stream.track_probs``: the
    stored print set before frame 0, anchored adaptation on, as ``Session.arm_enrollment(embedding=...)``), the RNNT
    greedy decode frame by frame (tokens + the frame each was emitted at); pass 2 = the encoder conditioned on P(user)
    (the speaker kernels at layers 1 and 3), read at the top layer. Checked against the served dump (``verify``)."""

    def __init__(self, afm=None, device="mps"):
        import torch
        from audioforge.train import load_model
        from audioforge.tsvad_stream import load_tsvad
        import eot_latency as E
        self.torch = torch
        self.dev = device
        self.m = load_model(str(afm or E.SERVED_AFM), device).eval()
        self.cpu = load_model(str(afm or E.SERVED_AFM), "cpu").eval()  # decode + TS-VAD track (per-frame loops)
        self.tsvad = load_tsvad(str(ROOT / "runs" / "tsvad_spk.pt"))
        self.turn_name = next(k for k, v in self.m.head_cfg.items() if v["type"] == "turn")
        self.head = self.m.heads[self.turn_name]
        self.asr_name = self.head.text_head

    def __call__(self, audios: list, prints: list) -> list[dict]:
        """audios: list of float32 16 kHz arrays; prints: list of 192-d prints (or None) -> per clip dict of
        e (T, 512) float16, vad (T,), pu (T,), po (T,), y (U,) tokens, n (T,) tokens emitted at frames <= t."""
        from audioforge.heads.turn import greedy_decode_frames, token_counts
        from audioforge.tsvad_stream import track_probs
        torch, m = self.torch, self.m
        with torch.no_grad():
            x, xl = m._pad(audios)
            enc, elen, hid = m.encode(x, xl, ATT, return_hidden=True)
            vad = m.heads["vad"](m.head_input("vad", enc, hid)).sigmoid().float().cpu().numpy()
            b4 = m.head_input("spk", enc, hid).float().cpu().numpy()
            fa = m.head_input(self.asr_name, enc, hid).float().cpu()
            T = enc.shape[1]
            outs, acts = [], torch.zeros(len(audios), T)
            for i in range(len(audios)):
                L = int(elen[i])
                P = track_probs(self.tsvad, self.cpu.heads["spk"], b4[i, :L], prints[i])
                hyp, fr = greedy_decode_frames(self.cpu.heads[self.asr_name], fa[i, :L])
                y = torch.tensor(hyp, dtype=torch.long)
                n = token_counts(torch.tensor([fr], dtype=torch.long) if fr else torch.full((1, 1), -1),
                                 torch.tensor([len(hyp)]), L)[0].numpy() if hyp else np.zeros(L, np.int64)
                acts[i, :L] = torch.from_numpy(P[:, 0])
                outs.append({"vad": vad[i, :L].astype(np.float16), "pu": P[:, 0].astype(np.float16),
                             "po": P[:, 1].astype(np.float16), "y": np.asarray(hyp, np.int32), "n": n.astype(np.int32)})
            e2 = m.encode(x, xl, ATT, spk_act=acts.to(self.dev))[0].float().cpu().numpy()
            for i, o in enumerate(outs):
                o["e"] = e2[i, : len(o["pu"])].astype(np.float16)
        return outs


def head_probs(head, f: dict, dev="cpu", state_dict=None) -> np.ndarray:
    """A turn head's per-frame p on one clip's cached inputs, exactly as served (act = P(user), columns [P(user),
    P(other), 0, 0], primary 0, the decoded text state)."""
    import torch
    with torch.no_grad():
        e = torch.as_tensor(np.asarray(f["e"], np.float32), device=dev)[None]
        T = e.shape[1]
        pu = torch.as_tensor(np.asarray(f["pu"], np.float32), device=dev)[None]
        po = torch.as_tensor(np.asarray(f["po"], np.float32), device=dev)[None]
        cols = torch.stack([pu, po, torch.zeros_like(pu), torch.zeros_like(pu)], -1)
        y = torch.as_tensor(np.asarray(f["y"], np.int64), device=dev)[None]
        if y.shape[1] == 0:
            y = torch.zeros(1, 1, dtype=torch.long, device=dev)
        n = torch.as_tensor(np.asarray(f["n"], np.int64), device=dev)[None]
        return head.decode(e, torch.tensor([T], device=dev), spk_act=pu, text=(y, n), cols=cols,
                           prim=torch.zeros(1, dtype=torch.long, device=dev))[0].float().cpu().numpy()


def cmd_verify(a):
    """The offline Extractor + the served head vs the served dump's p on a few eval sessions."""
    import torch
    import eot_latency as E
    torch.set_num_threads(2)
    ex = Extractor(device=a.device)
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump]
    pick = [s for s in sess if s["set"] != "ami"][:: max(1, 32 // a.n)][: a.n] + [s for s in sess if s["set"] == "ami"][: a.n]
    res = []
    head = ex.cpu.heads[ex.turn_name]
    for s in pick:
        f = ex([E.read_audio(s)], [s["embedding"]])[0]
        p = head_probs(head, f)
        d = dump[s["key"]]["head"]
        q = np.asarray(d["p"])
        L = min(len(p), len(q))
        pu_d = np.asarray(d["pu"])[:L]
        r = {"key": s["key"], "T": [len(p), len(q)], "p_absdiff_max": float(np.abs(p[:L] - q[:L]).max()),
             "p_absdiff_p99": float(np.percentile(np.abs(p[:L] - q[:L]), 99)),
             "pu_absdiff_max": float(np.abs(np.asarray(f["pu"], float)[:L] - pu_d).max()),
             "vad_absdiff_max": float(np.abs(np.asarray(f["vad"], float)[:L] - np.asarray(d["vad"])[:L]).max()),
             "p>=0.99 agree": float(np.mean((p[:L] >= 0.99) == (q[:L] >= 0.99)))}
        res.append(r)
        log(r)
    save("verify_extractor", res)


# --------------------------------------------------------------------------- training manifest
EVAL_OTO = None  # the 16 eval conversations (E2E clips.json), never trained on
WIN_S = 40.0  # oto window (a cold-started session, like the eval clips)
N_OTO_CONV, N_OTO_WIN = 200, 8
N_MEET = 900  # turn windows per meeting corpus
ST_PAD = 1.2  # s of trailing near-silence after a smart-turn clip


def eval_oto_ids() -> set:
    import eot_latency as E
    return {c["conversation"] for c in json.loads((E.E2E / "clips.json").read_text()) if c["set"] == "oto"}


def _intervals(fr: np.ndarray) -> list:
    """0/1 frames -> [[a, b], ...] seconds."""
    out, on = [], None
    for v, x in enumerate(list(fr) + [0]):
        if x and on is None:
            on = v
        elif not x and on is not None:
            out.append([round(on * FRAME, 3), round(v * FRAME, 3)])
            on = None
    return out


def oto_single(ds, m, guard: float = 0.2) -> dict:
    """{party: [(a, b), ...]} stretches of >= 1 s where the party's Silero track is on and the other party's is off
    within +-guard s (the stored-print picker's "the other party is silent"; tsvad.single_segments needs words)."""
    out = {}
    for c in (0, 1):
        me, oth = ds.acts[m][f"{m}:{c}"], ds.acts[m][f"{m}:{1 - c}"]
        segs = []
        for p, q in me:
            cuts = [(max(p, o0 - guard), min(q, o1 + guard)) for o0, o1 in oth if o0 - guard < q and o1 + guard > p]
            cur = p
            for c0, c1 in sorted(cuts):
                if c0 - cur >= 1.0:
                    segs.append((cur, c0))
                cur = max(cur, c1)
            if q - cur >= 1.0:
                segs.append((cur, q))
        out[f"{m}:{c}"] = segs
    return out


def cmd_manifest(a):
    """Training clips (module doc) -> WORK/manifest.json; oto channels cached at 16 kHz on first use (resumable)."""
    import random
    import torch
    torch.set_num_threads(2)
    import tsvad as T
    from audioforge.datasets import dyadic as D
    t0 = time.time()
    rng = random.Random(0)
    man = WORK / "manifest.json"
    clips = []
    # ---- oto: user channel alone, both channels of each conversation as "the user"
    ev = eval_oto_ids()
    ids = [i for i in D.list_ids("oto") if i not in ev and (D._root(None, "oto") / "cache" / "labels" / f"{i}.json").exists()]
    ids = ids[:N_OTO_CONV]
    ds = D.Dyadic(ids, "oto", verbose=False)
    for m in ids:
        if time.time() - t0 > a.budget:
            log(f"budget: oto channels cached up to {m}; rerun")
            return
        ds.channels16k(m)
    log(f"oto: {len(ids)} conversations (eval conversations excluded: {len(ev)})")
    for m in ids:
        dur = ds.duration(m)
        segs = oto_single(ds, m)
        for c in (0, 1):
            party = f"{m}:{c}"
            turns = [t for t in ds.turns[m] if t["speaker"] == party and not t["bc"]]
            nxt = [turns[j + 1]["start"] if j + 1 < len(turns) else dur for j in range(len(turns))]
            cand = [t["end"] for t in turns]
            used = []
            for _ in range(40):
                if len(used) >= N_OTO_WIN or not cand:
                    break
                e = rng.choice(cand)
                a0 = round(max(0.0, e - rng.uniform(8.0, WIN_S - 3.0)), 2)
                b0 = min(dur, a0 + WIN_S)
                if any(abs(a0 - u) < WIN_S for u in used) or any(za < b0 and zb > a0 for za, zb in ds.zones[m]):
                    continue
                ivs = T.clip_from(segs.get(party, []), 5.0, rng, exclude=(a0 - 1.0, b0 + 1.0))
                if ivs is None:
                    continue
                tt = [[round(max(t["start"], a0) - a0, 3), round(t["end"] - a0, 3), round(min(n, b0) - a0, 3),
                       bool(t["start"] >= a0)] for t, n in zip(turns, nxt) if a0 < t["end"] <= b0 - 0.5]
                if not tt:
                    continue
                act = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in ds.acts[m][party]
                       if q > a0 and p < b0]
                used.append(a0)
                clips.append({"id": f"oto_{m}_{c}_{int(a0 * 100)}", "src": "oto", "meeting": m, "ch": c, "a": a0,
                              "b": round(b0, 2), "turns": tt, "act": act,
                              "print": {"meeting": m, "ch": c, "ivs": [[round(p, 3), round(q, 3)] for p, q in ivs]}})
    log(f"oto clips: {len(clips)}")
    # ---- AMI / ICSI train meetings (local): eot-bench v2 turn windows, as eot_latency.prepare_ami cuts the dev ones
    import eval_stage1 as ES
    from audioforge.datasets.ami import AMI, split_lists
    from audioforge.datasets import icsi as I
    for src, mk in (("ami", lambda ms: AMI(ms, verbose=False)), ("icsi", lambda ms: I.ICSI(ms, verbose=False))):
        root = ROOT / "data" / src / "audio"
        pool = split_lists()["train"] if src == "ami" else I.SPLITS["train"]
        ms = sorted(m for m in pool if (root / f"{m}.Mix-Headset.wav").exists())
        dsm = mk(ms)
        ext, meta = ES.turn_windows(dsm, 6.0)
        idx = sorted(random.Random(1).sample(range(len(ext)), min(N_MEET, len(ext))))
        segc, n = {}, 0
        for i in idx:
            v, mt = ext[i], meta[i]
            m, a0 = v["meeting"], float(v["start"])
            on, en = int(v["onset_frame"]) * FRAME, int(v["turn_end_frame"]) * FRAME
            b0 = min(a0 + en + 6.5, dsm.duration(m))
            nxt = round(len(v["spk_act"]) * FRAME, 3) if mt.get("end_reason") == "resume" else round(b0 - a0, 3)
            spk = dsm.speaker_ids[int(v["speaker"]) - getattr(dsm, "speaker_offset", 0)]
            if m not in segc:
                segc[m] = T.single_segments(dsm, m)
            ivs = T.clip_from(segc[m].get(spk, []), 5.0, random.Random(f"tv4_{src}_{i}"), exclude=(a0 - 2.0, b0 + 2.0))
            if ivs is None:
                continue
            clips.append({"id": f"{src}_{m}_{i:05d}", "src": src, "meeting": m, "ch": None, "a": round(a0, 3),
                          "b": round(b0, 3), "turns": [[round(on, 3), round(en, 3), nxt, not v["onset_clipped"]]],
                          "act": _intervals(np.asarray(v["spk_act"]) > 0.5),
                          "print": {"meeting": m, "ch": None, "ivs": [[round(p, 3), round(q, 3)] for p, q in ivs]}})
            n += 1
        log(f"{src}: {len(ms)} local train meetings, {len(ext)} windows, {n} clips")
    # ---- smart-turn human_5_all train split (BSD-2): one utterance, + ST_PAD s of near-silence after it
    from audioforge.datasets import smartturn as ST
    mp = ST.build_cache(verbose=False)
    meta = json.loads(mp.read_text())
    sp = ST.split_indices(meta)
    for i in sp["train"]:
        clips.append({"id": f"st_{i:05d}", "src": "st", "idx": i, "complete": bool(meta["complete"][i]),
                      "print": {"self": True}})
    log(f"smart-turn train clips: {len(sp['train'])} ({sp['how']}; eval {len(sp['eval'])} held out)")
    man.write_text(json.dumps(clips))
    log(f"manifest: {len(clips)} clips -> {man}")


# --------------------------------------------------------------------------- feature cache
FEATS = WORK / "feats"


class ClipAudio:
    """Audio + print audio of a manifest clip."""

    def __init__(self):
        self.ds = {}

    def _d(self, src, man):
        if src not in self.ds:
            from audioforge.datasets import dyadic as D
            from audioforge.datasets.ami import AMI
            from audioforge.datasets import icsi as I
            ms = sorted({c.get("meeting") for c in man if c["src"] == src and src != "st"})
            if src == "oto":
                self.ds[src] = D.Dyadic(ms, "oto", verbose=False)
            elif src == "ami":
                self.ds[src] = AMI(ms, verbose=False)
            elif src == "icsi":
                self.ds[src] = I.ICSI(ms, verbose=False)
            else:
                from audioforge.datasets import smartturn as ST
                meta = json.loads(ST.build_cache(verbose=False).read_text())
                self.ds[src] = (meta, np.load(ST.DEFAULT_ROOT / "cache" / "human_5_all.npy", mmap_mode="r"))
        return self.ds[src]

    def __call__(self, c, man):
        SR = 16000
        d = self._d(c["src"], man)
        if c["src"] == "oto":
            ch = d.channels16k(c["meeting"])[c["ch"]]
            x = np.asarray(ch[int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)
            pr = np.concatenate([np.asarray(ch[int(round(p * SR)): int(round(q * SR))], np.float32)
                                 for p, q in c["print"]["ivs"]])
        elif c["src"] in ("ami", "icsi"):
            x = np.asarray(d._clip(c["meeting"], c["a"], c["b"]), np.float32)
            pr = np.concatenate([np.asarray(d._clip(c["meeting"], p, q), np.float32) for p, q in c["print"]["ivs"]])
        else:
            meta, wav = d
            o = meta["offsets"]
            u = np.asarray(wav[o[c["idx"]]: o[c["idx"] + 1]], np.float32)
            noise = np.random.default_rng(c["idx"]).normal(0, 10 ** (-70 / 20), int(ST_PAD * SR)).astype(np.float32)
            x, pr = np.concatenate([u, noise]), u
        return x, pr


def cmd_feats(a):
    """Served turn-head inputs of every manifest clip -> FEATS/<id>.npz (resumable; --budget s per call)."""
    import torch
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    man = json.loads((WORK / "manifest.json").read_text())
    FEATS.mkdir(parents=True, exist_ok=True)
    todo = [c for c in man if not (FEATS / f"{c['id']}.npz").exists()]
    if a.src:
        todo = [c for c in todo if c["src"] in a.src.split(",")]
    log(f"feats: {len(todo)} of {len(man)} clips to do")
    if not todo:
        return
    t0 = time.time()
    ex = Extractor(device=a.device)
    au = ClipAudio()
    done = 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        xs, prs = zip(*[au(c, man) for c in cs])
        prints = [voiceprint(ex.cpu, p) if len(p) >= 8000 else None for p in prs]
        outs = ex(list(xs), prints)
        for c, o in zip(cs, outs):
            tmp = FEATS / f"{c['id']}.tmp.npz"
            np.savez(tmp, **o)
            tmp.rename(FEATS / f"{c['id']}.npz")
        done += len(cs)
    el = time.time() - t0
    log(f"feats: {done} clips in {el:.0f} s ({el / max(done, 1):.2f} s/clip); {len(todo) - done} left")


# --------------------------------------------------------------------------- targets
W_PAUSE = 3.0  # mid-turn pause frames (the hard negatives: 45 % of call turns hold a >= 160 ms pause)
END_W = {-2: 0.3, -1: 0.6, 0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 1.5}  # frames from the first silent frame
TAIL_W = 0.3  # later "turn over" frames until the user's next onset (<= 6 s)


def targets(c: dict, vad: np.ndarray, T: int):
    """(y (T,), w (T,), ends [(fe, next_frame)], pauses [(f0, f1)]) for a clip: the early-confidence objective of
    research/TURN_V4.md. Reference ends are the audible ends (``audible_end`` on the served VAD, clipped to the
    labelled end); frames outside the user's turns and their answer windows get weight 0."""
    y, w = np.zeros(T, np.float32), np.zeros(T, np.float32)
    y0, w0 = np.zeros(T, np.float32), np.zeros(T, np.float32)  # the served head's own objective (eot = 1 after the end)
    ends, pauses = [], []
    if c["src"] == "st":
        on = np.nonzero(vad > VAD_ON)[0]
        if not len(on):
            targets.orig = (y0, w0)
            return y, w, ends, pauses
        s0, e1 = on[0] * FRAME, (on[-1] + 1) * FRAME
        turns, act = [[s0, e1, T * FRAME, True]], [[s0, e1]]
    else:
        turns, act = c["turns"], c["act"]
    for s0, e0, nxt, has_on in turns:
        e1 = audible_end(s0, e0, vad, None)
        fs, fe = int(s0 / FRAME), int(np.ceil(e1 / FRAME - 1e-9))
        nf = min(T, int(nxt / FRAME), fe + 75)
        w[fs: max(fs, fe - 2)] = 1.0
        y[fs: max(fs, fe - 2)] = 0.0
        w0[fs: nf] = 1.0
        y0[fe: nf] = 0.0 if (c["src"] == "st" and not c["complete"]) else 1.0
        for (p0, p1), (q0, q1) in zip(act[:-1], act[1:]):  # pauses between two of the user's runs inside the turn
            if p1 >= s0 - 1e-6 and q0 <= e1 and q0 - p1 >= 0.16:
                f0, f1 = int(np.ceil(p1 / FRAME)), int(q0 / FRAME)
                if f1 > f0:
                    w[f0:f1] = W_PAUSE
                    pauses.append((f0, f1))
        if c["src"] == "st" and not c["complete"]:  # the utterance goes on: the trailing silence is a pause
            w[max(0, fe - 2):] = W_PAUSE
            y[max(0, fe - 2):] = 0.0
            pauses.append((fe, T))
            continue
        for k, wk in END_W.items():
            v = fe + k
            if 0 <= v < T and (k < 0 or v < nf):
                y[v], w[v] = 1.0, wk
        if nf > fe + 6:
            y[fe + 6: nf], w[fe + 6: nf] = 1.0, TAIL_W
        ends.append((fe, nf))
    targets.orig = (y0, w0)
    return y, w, ends, pauses


# --------------------------------------------------------------------------- training
def load_train(split_seed=0, val_frac=0.08):
    """All cached clips with targets; a validation split by conversation / meeting / clip (never the eval sets)."""
    import random
    man = json.loads((WORK / "manifest.json").read_text())
    groups = sorted({(c["src"], c.get("meeting", c["id"])) for c in man})
    rng = random.Random(split_seed)
    val_g = set(rng.sample(groups, int(val_frac * len(groups))))
    tr, va = [], []
    for c in man:
        f = FEATS / f"{c['id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        d = {k: z[k] for k in z.files}
        T = len(d["pu"])
        d["tgt"], d["wt"], d["ends"], d["pauses"] = targets(c, d["vad"].astype(np.float32), T)
        d["tgt0"], d["wt0"] = targets.orig
        d["src"], d["id"] = c["src"], c["id"]
        (va if (c["src"], c.get("meeting", c["id"])) in val_g else tr).append(d)
    return tr, va


def collate(items, dev):
    import torch
    B, T = len(items), max(len(d["pu"]) for d in items)
    U = max(1, max(len(d["y"]) for d in items))
    e = torch.zeros(B, T, items[0]["e"].shape[1])
    pu, po, tg, wt, tg0, wt0 = (torch.zeros(B, T) for _ in range(6))
    y = torch.zeros(B, U, dtype=torch.long)
    n = torch.zeros(B, T, dtype=torch.long)
    L = torch.zeros(B, dtype=torch.long)
    for i, d in enumerate(items):
        t = len(d["pu"])
        L[i] = t
        e[i, :t] = torch.from_numpy(d["e"].astype(np.float32))
        pu[i, :t], po[i, :t] = torch.from_numpy(d["pu"].astype(np.float32)), torch.from_numpy(d["po"].astype(np.float32))
        tg[i, :t], wt[i, :t] = torch.from_numpy(d["tgt"]), torch.from_numpy(d["wt"])
        tg0[i, :t], wt0[i, :t] = torch.from_numpy(d["tgt0"]), torch.from_numpy(d["wt0"])
        y[i, : len(d["y"])] = torch.from_numpy(d["y"].astype(np.int64))
        n[i, :t] = torch.from_numpy(d["n"].astype(np.int64))
        n[i, t:] = n[i, t - 1]
    cols = torch.stack([pu, po, torch.zeros_like(pu), torch.zeros_like(pu)], -1)
    to = lambda x: x.to(dev)  # noqa: E731
    return to(e), to(L), to(pu), to(cols), (to(y), to(n)), to(tg), to(wt), to(tg0), to(wt0)


def head_logits(head, b):
    import torch
    e, L, pu, cols, text = b[:5]
    h = head.hidden_states(e, L, pu, text, cols, torch.zeros(len(L), dtype=torch.long, device=e.device))
    return head.out(h).squeeze(-1).float()


def val_metrics(head, va, dev, ths=(0.9, 0.95, 0.99)) -> dict:
    """Weighted BCE on the val clips, and at each theta: % of ends with p >= theta on a frame within +240 ms of the
    audible end (frames fe .. fe + 2), % of mid-turn pauses (>= 160 ms) with p >= theta on a frame inside them."""
    import torch
    import torch.nn.functional as F
    head.eval()
    ls = ws = 0.0
    reach = {th: [] for th in ths}
    fa = {th: [] for th in ths}
    es, ps = [], []
    with torch.no_grad():
        for i in range(0, len(va), 16):
            its = va[i: i + 16]
            b = collate(its, dev)
            z = head_logits(head, b)
            l = F.binary_cross_entropy_with_logits(z, b[5], reduction="none")
            ls += float((l * b[6]).sum())
            ws += float(b[6].sum())
            p = z.sigmoid().cpu().numpy()
            for j, d in enumerate(its):
                for fe, nf in d["ends"]:
                    m = p[j, fe: min(fe + 3, len(d["pu"]))].max(initial=0.0)
                    es.append(m)
                    for th in ths:
                        reach[th].append(m >= th)
                for f0, f1 in d["pauses"]:
                    m = p[j, f0: f1].max(initial=0.0)
                    ps.append(p[j, f0 + 1: min(f1, f0 + 5)].max(initial=0.0))
                    for th in ths:
                        fa[th].append(m >= th)
    head.train()
    es, ps = np.asarray(es), np.asarray(ps)
    # end-vs-pause AUC: P(an end's max p in its first 240 ms > a pause's max p over its 160-400 ms frames)
    auc = float((es[:, None] > ps[None]).mean() + 0.5 * (es[:, None] == ps[None]).mean()) if len(es) and len(ps) else None
    return {"val_bce": round(ls / max(ws, 1), 4), "auc": round(auc, 4) if auc is not None else None,
            "reach@pause_fa10": round(100 * float(np.mean(es > np.percentile(ps, 90))), 1) if len(ps) else None,
            **{f"reach240@{th}": round(100 * float(np.mean(reach[th])), 1) for th in ths},
            **{f"pause_fa@{th}": round(100 * float(np.mean(fa[th])), 1) for th in ths},
            "n_ends": len(reach[ths[0]]), "n_pauses": len(fa[ths[0]])}


def cmd_train(a):
    """Heads only: the served turn head (warm start), everything else frozen and absent from the graph (its inputs are
    the cached pass-2 frames). -> WORK/<tag>/head.pt (best val BCE) + history.json. Resumable (last.pt)."""
    import random
    import torch
    import torch.nn.functional as F
    from audioforge.train import load_model
    import eot_latency as E
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    od = WORK / a.tag
    od.mkdir(parents=True, exist_ok=True)
    dev = a.device
    m = load_model(str(E.SERVED_AFM), dev).eval()
    name = next(k for k, v in m.head_cfg.items() if v["type"] == "turn")
    head = m.heads[name]
    last = od / "last.pt"
    if a.scratch and not last.exists():  # fresh init of the head (same architecture)
        for mod in head.modules():
            if hasattr(mod, "reset_parameters") and mod is not head:
                mod.reset_parameters()
    for p in m.parameters():
        p.requires_grad_(False)
    params = [p for n_, p in head.named_parameters() if not n_.startswith("fut.")]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    step, hist, best = 0, [], 1e9
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
        log(f"resume at step {step}")
    if step >= a.steps:
        log("done")
        return
    global W_PAUSE, TAIL_W, END_W
    W_PAUSE, TAIL_W = a.w_pause, a.tail_w
    END_W = {k: v * a.end_scale for k, v in END_W.items()}
    tr, va = load_train()
    by = {}
    for d in tr:
        by.setdefault("meet" if d["src"] in ("ami", "icsi") else d["src"], []).append(d)
    mix = {"oto": a.p_oto, "meet": a.p_meet, "st": 1 - a.p_oto - a.p_meet}
    log(f"train clips {len(tr)} ({ {k: len(v) for k, v in by.items()} }), val {len(va)}; mix {mix}")
    if step == 0:
        hist.append({"step": 0, **val_metrics(head, va, dev)})
        log(hist[-1])
    rng = random.Random(a.seed + step)
    head.train()
    t0 = time.time()
    while step < a.steps and time.time() - t0 < a.budget:
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * min(step / a.steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        src = rng.choices(list(mix), weights=list(mix.values()))[0]
        its = rng.sample(by[src], min(a.batch, len(by[src])))
        b = collate(its, dev)
        z = head_logits(head, b)
        l = F.binary_cross_entropy_with_logits(z, b[5], reduction="none")
        loss = (l * b[6]).sum() / b[6].sum().clamp(min=1)
        if a.aux_orig > 0:  # the served head's objective (BCE, pos_weight 2) so the fine-tune keeps its calibration
            l0 = F.binary_cross_entropy_with_logits(z, b[7], reduction="none", pos_weight=torch.tensor(2.0, device=z.device))
            loss = loss + a.aux_orig * (l0 * b[8]).sum() / b[8].sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            r = {"step": step, "lr": round(lr, 6), "loss": round(float(loss), 4), **val_metrics(head, va, dev)}
            hist.append(r)
            log(r)
            if -r["auc"] < best:
                best = -r["auc"]
                torch.save({"state_dict": {k: v.cpu() for k, v in head.state_dict().items()}, "step": step,
                            "val": r, "name": name}, od / "head.pt")
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist, "best": best},
                       last)
            (od / "history.json").write_text(json.dumps({"args": vars(a), "hist": hist}, indent=1))
    log(f"stopped at step {step} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------- evaluation inputs
EVALF = WORK / "evalfeats"


def cmd_evalfeats(a):
    """Served turn-head inputs of the 232 eot_latency sessions (one clip per batch: exactly the dumped session) and
    of the bundled clip under the six CLIP_VARIANTS deliveries (for the sweep's clip-cut check)."""
    import torch
    import eot_latency as E
    from audioforge.data import load_wav
    torch.set_num_threads(2)
    EVALF.mkdir(parents=True, exist_ok=True)
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump and not (EVALF / f"{s['key']}.npz").exists()]
    ex = Extractor(device=a.device)
    t0 = time.time()
    for s in sess:
        if time.time() - t0 > a.budget:
            log("budget; rerun")
            return
        o = ex([E.read_audio(s)], [s["embedding"]])[0]
        np.savez(EVALF / f"{s['key']}.npz", **o)
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
    emb = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json").read_text())
    for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items()):
        f = EVALF / f"clip_{i}.npz"
        if not f.exists():
            np.savez(f, **ex([fn(x0)], [emb])[0])
    log(f"evalfeats done ({time.time() - t0:.0f} s)")


def load_head(path=None, dev="cpu"):
    """The served turn head (bound to the served ASR's PredictionNet), optionally with trained weights."""
    import torch
    from audioforge.train import load_model
    import eot_latency as E
    m = load_model(str(E.SERVED_AFM), dev).eval()
    name = next(k for k, v in m.head_cfg.items() if v["type"] == "turn")
    if path:
        m.heads[name].load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    return m.heads[name].eval()


def cmd_evaldump(a):
    """A copy of the served dump with the turn head's p replaced by head ``--head``'s (computed on the cached served
    inputs) -> WORK/dump_<tag>/ and WORK/dump_<tag>_clip_frames.json; score with
    EOT_DUMP=... EOT_CLIP_FRAMES=... eot_latency.py sweep."""
    import torch
    import eot_latency as E
    torch.set_num_threads(2)
    head = load_head(a.head)
    od = WORK / f"dump_{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    diffs = []
    for f in sorted(E.DUMP.glob("*.json")):
        d = json.loads(f.read_text())
        z = np.load(EVALF / f"{d['key']}.npz")
        p = head_probs(head, {k: z[k] for k in z.files})
        old = np.asarray(d["head"]["p"])
        assert len(p) == len(old), (d["key"], len(p), len(old))
        if a.head is None:
            diffs.append(float(np.abs(p - old).max()))
        d["head"]["p"] = [float(x) for x in p]
        d["turn_ends"] = []  # the served turn_ends of the old head no longer apply
        (od / f.name).write_text(json.dumps(d))
    cv = json.loads((E.WORK / "clip_frames.json").read_text())
    for i, var in enumerate(E.CLIP_VARIANTS):
        z = np.load(EVALF / f"clip_{i}.npz")
        p = head_probs(head, {k: z[k] for k in z.files})
        assert len(p) == len(cv[var]["head"]["p"]), (var, len(p), len(cv[var]["head"]["p"]))
        if a.head is None:
            diffs.append(float(np.abs(p - np.asarray(cv[var]["head"]["p"])).max()))
        cv[var]["head"]["p"] = [float(x) for x in p]
    (WORK / f"dump_{a.tag}_clip_frames.json").write_text(json.dumps(cv))
    if diffs:
        log(f"served head re-computed offline: max |p - dumped p| = {max(diffs):.2e} over {len(diffs)} sessions")
    log(f"-> {od}")


# --------------------------------------------------------------------------- rule scan (the harness's simulator + scorer)
SHIPPED_GOAL = None  # filled from the old head's shipped-rule numbers per label set


def scan_grid() -> list[dict]:
    """The vad_head family (sim_room: head path + fallback + others path) on a wider grid than the harness, and the
    dyn-wait family (the fallback wait a function of p) with and without a head path."""
    import itertools
    import eot_latency as E
    base = dict(E.CHOSEN)
    R = []
    for vt, k, th, F, fu in itertools.product((0.3, 0.35, 0.4, 0.45, 0.5), range(1, 7),
                                              (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995,
                                               0.997, 0.999),
                                              (4, 5, 6, 7, 8, 9, 10), (8, 10, 12)):
        R.append({**base, "vad_thr": vt, "k": k, "th": th, "fallback_f": F, "fu": fu})
    for vt, cap, fl, p1, hp, fu in itertools.product((0.35, 0.4, 0.5), (6, 8, 10, 12), (1, 2, 3), (0.7, 0.9, 0.95, 0.99, 1.0),
                                                     ((2, 0.99), (1, 0.995), None), (8, 12)):
        k, th = hp if hp else (1, 2.0)
        R.append({**base, "vad_thr": vt, "k": k, "th": th, "fallback_f": cap, "fu": fu,
                  "dyn": (cap, fl, (cap - fl) / p1)})
    for vt, ve, ke, te, F, (k, th), fu in itertools.product((0.4, 0.5), (0.5, 0.6, 0.7, 0.8, 1.01), (1, 2),
                                                            (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999), (6, 8, 10),
                                                            ((2, 0.99), (2, 0.95), (1, 2.0)), (8, 12)):
        R.append({**base, "vad_thr": vt, "k": k, "th": th, "fallback_f": F, "fu": fu, "early": (ve, ke, te)})
    return R


def rule_label(r: dict) -> str:
    s = ""
    if r.get("early"):
        ve, ke, te = r["early"]
        s = f"early: p>={te} after {ke * 80}ms of VAD<{ve} | " if ve <= 1 else f"early: p>={te} ({ke} frames, any VAD) | "
    s += f"ourVAD<{r['vad_thr']}"
    if r["th"] <= 1.0:
        s += f" sil>={r['k'] * 80}ms & p>={r['th']}"
    if r.get("dyn"):
        c, f, a_ = r["dyn"]
        s += f" | dyn wait clamp({c * 80}-{a_ * 80:.0f}*p, {f * 80}, {c * 80}) ms"
    else:
        s += f" | fallback {r['fallback_f'] * 80}ms"
    return s + f" | others {r['fu'] * 80}ms"


def cmd_scan(a):
    """Score scan_grid() + the shipped rule on a dump (EOT_DUMP / EOT_LABELS / EOT_CLIP_FRAMES set by the caller),
    select under the goal (FI and missed no worse than the reference numbers on both corpora, no clip cut)."""
    import eot_latency as E
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump]
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    cv = json.loads(Path(os.environ.get("EOT_CLIP_FRAMES", E.WORK / "clip_frames.json")).read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())

    def score(rule):
        per = {sc: [] for sc in E.SCOPES}
        for ss in sess:
            t = [x + comp[ss["key"]] for x, _ in E.simulate(dump[ss["key"]], rule)]
            for sc, sets in E.SCOPES.items():
                if ss["set"] in sets:
                    per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
        return {sc: E.pool(v) for sc, v in per.items()}
    ref = json.loads(a.goal)  # {"calls_fi", "calls_miss", "ami_fi", "ami_miss"}
    rows = []
    t0 = time.time()
    for r in [dict(E.CHOSEN)] + scan_grid():
        sc = score(r)
        rows.append({"rule": r, "name": rule_label(r), "calls": sc["two_party_user"], "ami": sc["ami"]})
    log(f"{len(rows)} rules in {time.time() - t0:.0f} s")
    ok = [x for x in rows if x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["false_interruption_pct"] <= ref["calls_fi"]
          and x["calls"]["missed_pct"] <= ref["calls_miss"] and x["ami"]["false_interruption_pct"] <= ref["ami_fi"]
          and x["ami"]["missed_pct"] <= ref["ami_miss"]]
    ok.sort(key=lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"], x["ami"]["eot_total_ms_p50"]))
    best = None
    for x in ok:
        x["clip_cut_in"] = [v for v, dd in cv.items() if E.clip_cuts(dict(dd), x["rule"], meta)["cuts_user_turn"]]
        if not x["clip_cut_in"]:
            best = x
            break
    # Pareto (calls p50 vs calls FI) among rules with calls missed and both AMI numbers within the goal
    fr = sorted([x for x in rows if x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["missed_pct"] <= ref["calls_miss"]
                 and x["ami"]["missed_pct"] <= ref["ami_miss"] and x["ami"]["false_interruption_pct"] <= ref["ami_fi"]],
                key=lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"]))
    par, bfi = [], 1e9
    for x in fr:
        if x["calls"]["false_interruption_pct"] < bfi:
            bfi = x["calls"]["false_interruption_pct"]
            par.append(x)
    # the fastest dyn-wait rule under the goal, separately
    dyn_ok = [x for x in ok if x["rule"].get("dyn")]
    res = {"dump": str(E.DUMP), "labels": os.environ.get("EOT_LABELS") or "original", "goal": ref, "n_rules": len(rows),
           "shipped_rule": rows[0], "n_meet_goal": len(ok), "best": best,
           "best_dyn": dyn_ok[0] if dyn_ok else None, "top_meeting_goal": ok[:15], "pareto": par,
           "head_reach": E.head_reach(dump, sess, ths=(0.5, 0.8, 0.9, 0.95, 0.98, 0.99))}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    f = lambda x: (f"{x['calls']['eot_total_ms_p50']}/{x['calls']['eot_total_ms_p95']} {x['calls']['false_interruption_pct']}/"  # noqa
                   f"{x['calls']['missed_pct']} | AMI {x['ami']['eot_total_ms_p50']}/{x['ami']['eot_total_ms_p95']} "
                   f"{x['ami']['false_interruption_pct']}/{x['ami']['missed_pct']} | {x['name']}")
    log("shipped rule:", f(rows[0]))
    log(f"{len(ok)} meet the goal; best (no clip cut):", f(best) if best else None)
    if dyn_ok:
        log("best dyn-wait:", f(dyn_ok[0]))
    for x in par:
        log("  pareto:", f(x))


# --------------------------------------------------------------------------- smart-turn's own test clips
def cmd_stest(a):
    """Accuracy on the 399 human_5_all clips in smart-turn-data-v3.2-test (runs/smartturn_audit.json protocol: label
    endpoint_bool; smart-turn 97.0 %). Each clip is served as a session (+ ST_PAD s of -70 dBFS noise, the clip's own
    speech as the print, as in training); "complete" = the turn head's p on the frame 160 ms after the clip's last
    served-VAD speech frame (the head path's first chance under the shipped rule) >= theta. theta: 0.5, 0.99 (the
    shipped rule's), and the accuracy-best theta on the training clips (so the test clips pick nothing)."""
    import torch
    from audioforge.tsvad_stream import voiceprint
    from audioforge.datasets import smartturn as ST
    torch.set_num_threads(2)
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    sp = ST.split_indices(meta)
    lab = np.asarray(meta["complete"], bool)
    od = WORK / "stest"
    od.mkdir(parents=True, exist_ok=True)
    ex = None
    au = ClipAudio()
    for i in sp["eval"]:
        f = od / f"st_{i:05d}.npz"
        if f.exists():
            continue
        ex = ex or Extractor(device=a.device)
        c = {"src": "st", "idx": i}
        x, pr = au(c, [])
        np.savez(f, **ex([x], [voiceprint(ex.cpu, pr)])[0])
    heads = {"served": None, **{h.split("=")[0]: h.split("=")[1] for h in a.heads}}

    def decide(head, d):
        p = head_probs(head, d)
        v = np.nonzero(np.asarray(d["vad"], float) > VAD_ON)[0]
        j = min(len(p) - 1, (v[-1] if len(v) else 0) + 2)
        return float(p[j])
    res = {}
    for name, path in heads.items():
        head = load_head(path)
        pt = np.array([decide(head, dict(np.load(od / f"st_{i:05d}.npz"))) for i in sp["eval"]])
        ptr = np.array([decide(head, dict(np.load(FEATS / f"st_{i:05d}.npz"))) for i in sp["train"]])
        yt, ytr = lab[sp["eval"]], lab[sp["train"]]
        grid = np.linspace(0.01, 0.999, 200)
        th_best = float(grid[np.argmax([((ptr >= t) == ytr).mean() for t in grid])])
        r = {}
        for tag, th in (("0.5", 0.5), ("0.99", 0.99), ("train_best", th_best)):
            pred = pt >= th
            r[tag] = {"theta": round(th, 3), "accuracy": round(100 * float((pred == yt).mean()), 1),
                      "recall_complete": round(100 * float(pred[yt].mean()), 1),
                      "recall_incomplete": round(100 * float((~pred[~yt]).mean()), 1)}
        res[name] = r
        log(name, r)
    save("smartturn_test", {"n": len(sp["eval"]), "how": sp["how"], "smart_turn_v3.2": 96.99, "heads": res})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("labels_eval")
    mf = sub.add_parser("manifest")
    mf.add_argument("--budget", type=float, default=540)
    fe = sub.add_parser("feats")
    fe.add_argument("--budget", type=float, default=500)
    fe.add_argument("--batch", type=int, default=8)
    fe.add_argument("--device", default="mps")
    fe.add_argument("--src", default=None)
    t = sub.add_parser("train")
    t.add_argument("--tag", default="h1")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--p-oto", type=float, default=0.35)
    t.add_argument("--p-meet", type=float, default=0.4)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--budget", type=float, default=540)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--w-pause", type=float, default=3.0)
    t.add_argument("--tail-w", type=float, default=0.3)
    t.add_argument("--end-scale", type=float, default=1.0)
    t.add_argument("--scratch", action="store_true")
    t.add_argument("--aux-orig", type=float, default=0.0)
    t.add_argument("--device", default="mps")
    ef = sub.add_parser("evalfeats")
    ef.add_argument("--budget", type=float, default=540)
    ef.add_argument("--device", default="mps")
    ed = sub.add_parser("evaldump")
    ed.add_argument("--head", default=None)
    ed.add_argument("--tag", required=True)
    sc = sub.add_parser("scan")
    sc.add_argument("--goal", required=True)
    sc.add_argument("--out", required=True)
    st = sub.add_parser("stest")
    st.add_argument("--heads", nargs="*", default=[])
    st.add_argument("--device", default="mps")
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=3)
    v.add_argument("--device", default="mps")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
