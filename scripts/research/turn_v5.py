"""research/TURN_V5.md: turn head v5. Main line: a smart-turn style SEGMENT classifier on our frozen encoder
(attention pooling over the last <= 8 s of encoder frames + the RNNT text so far), read when our VAD goes quiet.
Per-frame variants (causal Transformer / bigger GRU on the turn_v4 cache) are ablations.

Stages (W = scratch/turn_v5; the turn_v4 cache = scratch/turn_v4 is reused: pass-2 frames, VAD, TS-VAD, RNNT text):
  blocks      pass-1 encoder blocks 4 / 8 / 12 / 17 of every training clip, eval session, clip delivery and
              smart-turn test clip -> W/blk/<id>.npz (MPS, resumable)
  cuts        smart-turn style negatives: training utterances cut inside speech at an energy dip (a word boundary
              proxy) + 640 ms of the clip's own quiet audio, through the full served-input extractor -> W/cuts/
  smartturn   smart-turn v3.2 (Pipecat's features + ONNX) at offsets around every end / pause / cut -> W/st/<id>.npz
  texts       the decoded RNNT text of every clip as words (+ token -> word index) -> W/texts.jsonl
  lm          (venv_mlx) Qwen2.5-7B-Instruct-4bit: log P(sentence-final punctuation | words so far) -> W/lm/<id>.npy
  prosody     per-80 ms prosody (audioforge.heads.prosody) -> W/pros/<id>.npy
  probe       which encoder block: a small segment classifier per block, val end-vs-pause AUC
  train       segment classifier (or --arch frame: per-frame causal model) -> W/<tag>/
  evaldump    p of a trained model on every eval-session frame -> W/dump_<tag> (+ clip frames) for eot_latency / scan
  report      head reach, frame AUC, calibration, smart-turn test accuracy of a dump
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

W = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v5")
W4 = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v4")
FEATS4, EVALF4, STEST4 = W4 / "feats", W4 / "evalfeats", W4 / "stest"
BLK, CUTS, STT, LMD, PROS = W / "blk", W / "cuts", W / "st", W / "lm", W / "pros"
OUT = ROOT / "runs" / "turn_v5.json"
FRAME, SR = 0.08, 16000
BLOCKS = (4, 8, 12, 17)
ST_OFFS_MS = (-240, -80, 0, 80, 160, 240, 400)
ST_ONNX = ROOT / ".venv/lib/python3.12/site-packages/pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def manifest() -> list[dict]:
    return json.loads((W4 / "manifest.json").read_text())


def npz(path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- blocks
def eval_items():
    """(id, audio fn, dir of the served inputs) of the eval sessions, the six clip deliveries, the smart-turn tests."""
    import eot_latency as E
    from audioforge.data import load_wav
    dump = E.load_dump()
    out = [(f"ev_{s['key']}", (lambda s=s: E.read_audio(s)), EVALF4 / f"{s['key']}.npz")
           for s in E.sessions() if s["key"] in dump]
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
    for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items()):
        out.append((f"clip_{i}", (lambda fn=fn: fn(x0)), EVALF4 / f"clip_{i}.npz"))
    return out


def cmd_blocks(a):
    import torch
    import turn_v4 as V4
    from audioforge.train import load_model
    import eot_latency as E
    torch.set_num_threads(2)
    BLK.mkdir(parents=True, exist_ok=True)
    man = manifest()
    au = V4.ClipAudio()
    items = []
    if a.which in ("all", "train"):
        items += [(c["id"], (lambda c=c: au(c, man)[0]), FEATS4 / f"{c['id']}.npz") for c in man]
    if a.which in ("all", "eval"):
        items += eval_items()
        from audioforge.datasets import smartturn as ST
        meta = json.loads(ST.build_cache(verbose=False).read_text())
        for i in ST.split_indices(meta)["eval"]:
            items.append((f"stt_{i:05d}", (lambda i=i: au({"src": "st", "idx": i}, [])[0]), STEST4 / f"st_{i:05d}.npz"))
    if a.which == "cuts":
        cm = json.loads((W / "cuts.json").read_text())
        items = [(c["id"], (lambda c=c: cut_audio(c, au, man)), CUTS / f"{c['id']}.npz") for c in cm]
    todo = [it for it in items if not (BLK / f"{it[0]}.npz").exists()]
    log(f"blocks: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    m = load_model(str(E.SERVED_AFM), a.device).eval()
    t0, done = time.time(), 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        xs = [fn() for _, fn, _ in cs]
        with torch.no_grad():
            x, xl = m._pad(xs)
            enc, elen, hid = m.encode(x, xl, V4.ATT, return_hidden=True)
            hs = [hid[b - 1].half().cpu().numpy() for b in BLOCKS]
        for j, (cid, _, ref) in enumerate(cs):
            L = int(elen[j])
            if ref.exists():
                T = len(np.load(ref)["pu"])
                assert abs(T - L) <= 1, (cid, T, L)
                L = T
            tmp = BLK / f"{cid}.tmp.npz"
            np.savez(tmp, **{f"b{b}": h[j, :L] for b, h in zip(BLOCKS, hs)})
            tmp.rename(BLK / f"{cid}.npz")
        done += len(cs)
    el = time.time() - t0
    log(f"blocks: {done} in {el:.0f} s; {len(todo) - done} left")


# --------------------------------------------------------------------------- cuts (smart-turn style negatives)
CUT_PAD_S = 0.64
CUT_CTX_S = 14.0  # 8 s window + the encoder's 5.6 s left context


def frame_db(x: np.ndarray, hop: int = 160) -> np.ndarray:
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, 1) + 1e-10)


def cmd_cutplan(a):
    """One or two cut points per oto / AMI / ICSI clip: inside a labelled user turn (>= 1 s after its start, >= 0.6 s
    before its end), at a 10 ms energy minimum over +-50 ms that is >= 10 dB below the turn's median level and where
    the served VAD says speech on both sides (a word boundary inside running speech, not a pause)."""
    import random
    import turn_v4 as V4
    man = manifest()
    au = V4.ClipAudio()
    rng = random.Random(5)
    plan = []
    t0 = time.time()
    for c in man:
        if c["src"] == "st":
            continue
        x, _ = au(c, man)
        db = frame_db(x)
        vad = np.load(FEATS4 / f"{c['id']}.npz")["vad"].astype(np.float32)
        cands = []
        for s0, e0, nxt, _ in c["turns"]:
            a0, b0 = int((s0 + 1.0) * 100), int((e0 - 0.6) * 100)
            if b0 - a0 < 20:
                continue
            med = np.median(db[int(s0 * 100): int(e0 * 100)])
            for j in range(a0, b0):
                if j < 5 or j + 5 >= len(db):
                    continue
                if db[j] == db[j - 5: j + 6].min() and db[j] < med - 10:
                    v = int(j / 8)
                    if 1 <= v < len(vad) - 2 and vad[v - 1] > 0.5 and vad[min(len(vad) - 1, v + 2)] > 0.5:
                        cands.append(j / 100)
        for t in rng.sample(cands, min(a.per_clip, len(cands))):
            plan.append({"id": f"cut_{c['id']}_{int(t * 100)}", "src": c["src"], "clip": c["id"], "t": round(t, 2),
                         "a": round(max(0.0, t - CUT_CTX_S), 2)})
    (W / "cuts.json").write_text(json.dumps(plan))
    log(f"cut plan: {len(plan)} cuts ({time.time() - t0:.0f} s)")


def cut_audio(cp: dict, au, man) -> np.ndarray:
    c = next(x for x in man if x["id"] == cp["clip"]) if not hasattr(cut_audio, "idx") else cut_audio.idx[cp["clip"]]
    x, _ = au(c, man)
    db = frame_db(x)
    # the clip's own quietest 0.5 s as the pad (room noise, not digital silence)
    k = 50
    cs = np.convolve(db, np.ones(k) / k, "valid")
    j = int(np.argmin(cs))
    q = x[j * 160: (j + k) * 160]
    n = int(CUT_PAD_S * SR)
    pad = np.tile(q, n // max(1, len(q)) + 1)[:n] if len(q) else np.zeros(n, np.float32)
    seg = x[int(cp["a"] * SR): int(cp["t"] * SR)]
    return np.concatenate([seg, pad]).astype(np.float32)


class _HidTap:
    """Wraps model.encode so the pass-1 hidden states of turn_v4.Extractor's first call are kept (blocks)."""

    def __init__(self, m):
        self.m, self.enc, self.hid = m, m.encode, None

    def __call__(self, *a, **k):
        out = self.enc(*a, **k)
        if k.get("return_hidden"):
            self.hid = out[2]
        return out


ST3_PAD_S = 3.0


def extract_items(which, man, au):
    """(id, audio fn, print fn or None, out dir) of a set: cuts | st3 (smart-turn train clips + 3 s of digital
    silence, no print: the eot_assistant protocol) | asst (the 399 eot_assistant clips exactly as that benchmark
    feeds them)."""
    if which == "cuts":
        cut_audio.idx = {c["id"]: c for c in man}
        plan = json.loads((W / "cuts.json").read_text())
        return [(c["id"], (lambda c=c: cut_audio(c, au, man)), c["clip"], CUTS) for c in plan]
    from audioforge.datasets import smartturn as ST
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    wav = np.load(ST.DEFAULT_ROOT / "cache" / "human_5_all.npy", mmap_mode="r")
    o = meta["offsets"]
    z = np.zeros(int(ST3_PAD_S * SR), np.float32)
    if which == "st3":
        idx = ST.split_indices(meta)["train"]
        return [(f"st3_{i:05d}", (lambda i=i: np.concatenate([np.asarray(wav[o[i]: o[i + 1]], np.float32), z])), None,
                W / "st3") for i in idx]
    if which == "st32":
        return [(c["id"], (lambda c=c: st32_audio(c)), None, W / "st32") for c in st32_clips()]
    import eot_assistant as EA
    return [(f"asst_{c['key']}", (lambda c=c: EA.audio(c)), None, W / "asst") for c in EA.clips()]


ST32 = Path("/Volumes/ExternalSSD/nvidia-audio-models/data/smart_turn_v3_2")
_ST32 = {}


def st32_clips() -> list[dict]:
    """English rows of smart-turn-data-v3.2-train (CC BY 4.0) fetched shard by shard (data/smart_turn_v3_2)."""
    out = []
    for jf in sorted(ST32.glob("eng_*.json")):
        m = json.loads(jf.read_text())
        sh = jf.stem
        for j in range(len(m["ids"])):
            dur = (m["offsets"][j + 1] - m["offsets"][j]) / SR
            if dur < 0.5 or dur > 16:
                continue
            out.append({"id": f"st32_{sh[4:]}_{j:04d}", "src": "st32", "shard": sh, "j": j,
                        "complete": bool(m["complete"][j]), "group": ("st32", f"{sh}_{j}")})
    return out


def st32_audio(c) -> np.ndarray:
    if c["shard"] not in _ST32:
        m = json.loads((ST32 / f"{c['shard']}.json").read_text())
        _ST32[c["shard"]] = (m, np.load(ST32 / f"{c['shard']}.npy", mmap_mode="r"))
    m, a = _ST32[c["shard"]]
    x = np.asarray(a[m["offsets"][c["j"]]: m["offsets"][c["j"] + 1]], np.float32) / 32768.0
    return np.concatenate([x, np.zeros(int(ST3_PAD_S * SR), np.float32)])


def cmd_extract(a):
    """Served turn-head inputs (turn_v4.Extractor: VAD, TS-VAD, RNNT text, pass 2) + pass-1 blocks of a set."""
    import torch
    import turn_v4 as V4
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    man = manifest()
    au = V4.ClipAudio()
    items = extract_items(a.set, man, au)
    items[0][3].mkdir(parents=True, exist_ok=True)
    BLK.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (it[3] / f"{it[0]}.npz").exists()]
    log(f"extract {a.set}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    ex = V4.Extractor(device=a.device)
    tap = _HidTap(ex.m)
    ex.m.encode = tap
    idx = {c["id"]: c for c in man}
    prc = {}
    t0, done = time.time(), 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        xs = [fn() for _, fn, _, _ in cs]
        prints = []
        for _, _, pk, _ in cs:
            if pk is None:
                prints.append(None)
                continue
            if pk not in prc:
                _, pr = au(idx[pk], man)
                prc[pk] = voiceprint(ex.cpu, pr) if len(pr) >= 8000 else None
            prints.append(prc[pk])
        outs = ex(xs, prints)
        hs = [tap.hid[b - 1].half().cpu().numpy() for b in BLOCKS]
        for j, ((cid, _, pk, od), o) in enumerate(zip(cs, outs)):
            T = len(o["pu"])
            if pk is None:  # no print: the served session is not enrolled, the head reads P(user) = P(other) = 0
                o["pu"] = np.zeros(T, np.float16)
                o["po"] = np.zeros(T, np.float16)
            np.savez(BLK / f"{cid}.tmp.npz", **{f"b{b}": h[j, :T] for b, h in zip(BLOCKS, hs)})
            (BLK / f"{cid}.tmp.npz").rename(BLK / f"{cid}.npz")
            np.savez(od / f"{cid}.tmp.npz", **o)
            (od / f"{cid}.tmp.npz").rename(od / f"{cid}.npz")
        done += len(cs)
    el = time.time() - t0
    log(f"extract {a.set}: {done} in {el:.0f} s ({el / max(done, 1):.2f} s each); {len(todo) - done} left")


# --------------------------------------------------------------------------- events (ends, pauses, cuts) per clip
def clip_events(c: dict, d: dict) -> tuple:
    """(targets tuple from turn_v4.targets, [(kind, frame, turn_start_s)]) with kind end / pause / cut; frame = the
    first quiet frame (an end: the audible end's frame; a pause: its first frame; a cut: the first pad frame)."""
    import turn_v4 as V4
    vad = d["vad"].astype(np.float32)
    T = len(vad)
    if c["src"] == "cut":
        fe = int(round((c["t"] - c["a"]) / FRAME))
        return None, [("cut", fe, 0.0)]
    y, w, ends, pauses = V4.targets(c, vad, T)
    ev = []
    if c["src"] == "st":
        on = np.nonzero(vad > V4.VAD_ON)[0]
        s0 = on[0] * FRAME if len(on) else 0.0
        for fe, nf in ends:
            ev.append(("end", fe, s0))
        for f0, f1 in pauses:
            ev.append(("pause", f0, s0))
        return (y, w, ends, pauses), ev
    starts = [t[0] for t in c["turns"]]
    for fe, nf in ends:
        s0 = max([s for s in starts if s < fe * FRAME] or [0.0])
        ev.append(("end", fe, s0))
    for f0, f1 in pauses:
        s0 = max([s for s in starts if s < f0 * FRAME] or [0.0])
        ev.append(("pause", f0, s0))
    return (y, w, ends, pauses), ev


def all_clips(with_cuts=True) -> list[dict]:
    man = manifest()
    if (W / "st3").exists():
        man += [{"id": "st3" + c["id"][2:], "src": "st3", "idx": c["idx"], "complete": c["complete"],
                 "group": ("st", c["id"])} for c in list(man) if c["src"] == "st"]
    if (W / "st32").exists() and USE_ST32:
        man += [c for c in st32_clips() if (W / "st32" / f"{c['id']}.npz").exists()]
    if with_cuts and (W / "cuts.json").exists():
        for cp in json.loads((W / "cuts.json").read_text()):
            man.append({**cp, "src": "cut", "orig_src": cp["src"]})
    return man


USE_ST32 = True


def feat_path(c):
    return {"cut": CUTS, "st3": W / "st3", "st32": W / "st32"}.get(c["src"], FEATS4) / f"{c['id']}.npz"


# --------------------------------------------------------------------------- smart-turn targets
def cmd_smartturn(a):
    """smart-turn v3.2 P(complete) at ST_OFFS_MS around every event, as Pipecat calls it: the audio from the turn's
    speech start - 0.5 s to the call time, last <= 8 s, Pipecat's numpy log-mel + the bundled ONNX. Events are sampled
    (--frac per kind) so the total stays inside --max-calls."""
    import random
    import onnxruntime as ort
    import turn_v4 as V4
    from pipecat.audio.turn.smart_turn._whisper_features import compute_whisper_log_mel_features
    STT.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
    sess = ort.InferenceSession(str(ST_ONNX), sess_options=so, providers=["CPUExecutionProvider"])
    man = all_clips()
    idx = {c["id"]: c for c in man}
    au = V4.ClipAudio()
    rng = random.Random(7)
    todo = [c for c in man if c["src"] != "st3" and not (STT / f"{c['id']}.npz").exists() and feat_path(c).exists()]
    log(f"smartturn: {len(todo)} of {len(man)} clips to do")
    t0, calls, done = time.time(), 0, 0
    for c in todo:
        if time.time() - t0 > a.budget:
            break
        d = npz(feat_path(c))
        _, ev = clip_events(c, d)
        if c["src"] == "cut":
            cut_audio.idx = idx
            x = cut_audio(c, au, man)
        else:
            x, _ = au(c, man)
        fr, pr, kd = [], [], []
        for kind, f, s0 in ev:
            if kind == "pause" and rng.random() > a.p_pause:
                continue
            for off in ST_OFFS_MS:
                if kind == "cut" and off < 0:
                    continue
                t = f * FRAME + off / 1000
                v = int(round(t / FRAME)) - 1
                if v < 0 or v >= len(d["vad"]) or t * SR > len(x):
                    continue
                seg = x[max(0, int((s0 - 0.5) * SR)): int(t * SR)][-8 * SR:]
                if len(seg) < 1600:
                    continue
                seg = np.pad(seg, (8 * SR - len(seg), 0))
                mel = compute_whisper_log_mel_features(seg, do_normalize=True)[None].astype(np.float32)
                p = float(sess.run(None, {"input_features": mel})[0].reshape(-1)[0])
                fr.append(v), pr.append(p), kd.append({"end": 0, "pause": 1, "cut": 2}[kind] * 1000 + off)
                calls += 1
        np.savez(STT / f"{c['id']}.npz", v=np.asarray(fr, np.int32), p=np.asarray(pr, np.float32),
                 k=np.asarray(kd, np.int32))
        done += 1
    el = time.time() - t0
    log(f"smartturn: {done} clips, {calls} calls in {el:.0f} s ({1000 * el / max(calls, 1):.1f} ms/call); "
        f"{len(todo) - done} clips left")


# --------------------------------------------------------------------------- texts + LM completeness
def cmd_texts(a):
    from audioforge.train import load_model
    import eot_latency as E
    m = load_model(str(E.SERVED_AFM), "cpu")
    sp = m.tokenizer
    rows = []
    items = [(c["id"], feat_path(c)) for c in all_clips()] + [(i, p) for i, _, p in eval_items()]
    for cid, p in items:
        if not p.exists():
            continue
        y = [int(t) for t in np.load(p)["y"]]
        pieces = [sp.decode([t]) for t in y]
        raw = [sp.sp.id_to_piece(t) for t in y]
        w_of, words = [], []
        for t, pc, rw in zip(y, pieces, raw):
            new = (rw.startswith("▁") if rw is not None else pc.startswith(" ")) or not words
            if new:
                words.append(pc.strip())
            else:
                words[-1] += pc.strip()
            w_of.append(len(words) - 1)
        rows.append({"id": cid, "words": words, "w_of": w_of})
    with open(W / "texts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"texts: {len(rows)} clips; example: {' '.join(rows[0]['words'][:30])}")


LM_PROMPT = "The following is a transcript of one person speaking on a phone call. Transcript:\n"


def cmd_lm(a):
    """(venv_mlx) per word: log P(next token is sentence-final punctuation | prompt + words so far), one causal
    forward per clip, batched (Qwen2.5-7B-Instruct 4-bit, Apache-2.0, local). Cut clips and st3 clips are skipped
    (st3 reads its st twin's scores; cuts carry no completeness target)."""
    import mlx.core as mx
    from mlx_lm import load
    LMD.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(W / "texts.jsonl")]
    rows = [r for r in rows if not r["id"].startswith(("cut_", "st3_", "clip_"))]
    todo = sorted([r for r in rows if not (LMD / f"{r['id']}.npy").exists()], key=lambda r: len(r["words"]))
    log(f"lm: {len(todo)} of {len(rows)}")
    if not todo:
        return
    m, tok = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    punct = mx.array([i for t, i in tok.get_vocab().items()
                      if t in (".", "?", "!", ".Ċ", "?Ċ", "!Ċ", ".ĊĊ", "?ĊĊ", "<|endoftext|>")])
    pre = tok.encode(LM_PROMPT)
    pad = tok.eos_token_id or 0
    t0, n = time.time(), 0
    i = 0
    while i < len(todo) and time.time() - t0 < a.budget:
        seqs = []
        L0 = None
        batch = []
        while i < len(todo) and len(batch) < a.lm_batch:
            r = todo[i]
            words = r["words"][-400:]
            ids, bounds = list(pre), []
            for w in words:
                ids += tok.encode(" " + w)
                bounds.append(len(ids) - 1)
            if L0 is not None and len(ids) > 1.5 * L0 + 32 and batch:
                break
            L0 = L0 or len(ids)
            batch.append((r, words, ids, bounds))
            i += 1
        Lm = max(len(x[2]) for x in batch)
        X = mx.array([x[2] + [pad] * (Lm - len(x[2])) for x in batch])
        lg = m(X)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        sc = np.array(mx.logsumexp(lp[:, :, punct], axis=-1).astype(mx.float32))
        for j, (r, words, ids, bounds) in enumerate(batch):
            out = np.full(len(r["words"]), -20.0, np.float32)
            if words:
                out[len(r["words"]) - len(words):] = sc[j, np.asarray(bounds)]
            np.save(LMD / f"{r['id']}.npy", out)
            n += 1
    log(f"lm: {n} clips in {time.time() - t0:.0f} s; {len(todo) - n} left")


def lm_frames(cid: str, n: np.ndarray, w_of: list) -> np.ndarray:
    """Per frame: the LM score of the transcript so far (tokens emitted at frames <= t); -20 before any word."""
    f = LMD / f"{cid}.npy"
    s = np.load(f) if f.exists() else None
    out = np.full(len(n), -20.0, np.float32)
    if s is None or not len(w_of) or not len(s):
        return out
    k = n > 0
    wi = np.asarray(w_of)[np.clip(n[k] - 1, 0, len(w_of) - 1)]
    out[k] = s[np.clip(wi, 0, len(s) - 1)]
    return out


# --------------------------------------------------------------------------- prosody
def cmd_prosody(a):
    import turn_v4 as V4
    from audioforge.heads.prosody import prosody_frames
    PROS.mkdir(parents=True, exist_ok=True)
    man = [c for c in all_clips() if c["src"] != "st32"]
    cut_audio.idx = {c["id"]: c for c in man}
    au = V4.ClipAudio()
    items = [(c["id"], (lambda c=c: cut_audio(c, au, man) if c["src"] == "cut" else au(c, man)[0]), feat_path(c))
             for c in man]
    items += [(i, fn, p) for i, fn, p in eval_items()]
    from audioforge.datasets import smartturn as ST
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    items += [(f"stt_{i:05d}", (lambda i=i: au({"src": "st", "idx": i}, [])[0]), STEST4 / f"st_{i:05d}.npz")
              for i in ST.split_indices(meta)["eval"]]
    import eot_assistant as EA
    items += [(f"asst_{c['key']}", (lambda c=c: EA.audio(c)), W / "asst" / f"asst_{c['key']}.npz") for c in EA.clips()]
    man = [c for c in man if c["src"] != "st32"]
    todo = [it for it in items if not (PROS / f"{it[0]}.npy").exists() and it[2].exists()]
    log(f"prosody: {len(todo)} of {len(items)}")
    t0, done, ms = time.time(), 0, []
    for cid, fn, ref in todo:
        if time.time() - t0 > a.budget:
            break
        T = len(np.load(ref)["pu"])
        x = fn()
        t1 = time.perf_counter()
        P = prosody_frames(x, T)
        ms.append((time.perf_counter() - t1) * 1000 / max(T, 1))
        np.save(PROS / f"{cid}.npy", P.astype(np.float16))
        done += 1
    log(f"prosody: {done} in {time.time() - t0:.0f} s ({np.mean(ms) if ms else 0:.3f} ms/frame); {len(todo) - done} left")



# --------------------------------------------------------------------------- segment classifier: data
WIN = 100
Q_VAD = 0.4  # "quiet" = served VAD below this (the shipped rule's line)


def quiet_run(vad: np.ndarray, thr: float = Q_VAD) -> np.ndarray:
    q = np.zeros(len(vad), np.int32)
    c = 0
    for i, x in enumerate(vad):
        c = c + 1 if x < thr else 0
        q[i] = c
    return q


def since_token(n: np.ndarray) -> np.ndarray:
    out = np.zeros(len(n), np.int32)
    last = -1
    prev = 0
    for v, k in enumerate(n):
        if k > prev:
            last = v
        prev = k
        out[v] = v - last if last >= 0 else v + 1
    return out


def group_of(c):
    if c["src"] == "cut":
        return (c["orig_src"], c["clip"].split("_")[1] if c["orig_src"] == "oto" else c["clip"].split("_")[1])
    return (c["src"], c.get("meeting", c["id"]))


def val_groups(man, frac=0.08, seed=0):
    """turn_v4's split (same seed, same groups) so v4 and v5 validate on the same held-out conversations."""
    import random
    groups = sorted({(c["src"], c.get("meeting", c["id"])) for c in man if c["src"] not in ("cut", "st3", "st32")})
    return set(random.Random(seed).sample(groups, int(frac * len(groups))))


def cut_group(c, idx):
    o = idx[c["clip"]]
    return (o["src"], o.get("meeting", o["id"]))


def samples_of(c: dict, d: dict, T: int):
    """Training frames (v, label, weight, kind) of one clip. kind: 0 end (first 12 frames after the audible end),
    1 in-turn pause, 2 quiet dip inside a turn, 3 cut (smart-turn style negative), 4 smart-turn clip end."""
    import turn_v4 as V4
    vad = d["vad"].astype(np.float32)
    q = quiet_run(vad)
    S = []
    if c["src"] == "cut":
        fc = int(round((c["t"] - c["a"]) / FRAME))
        for v in range(fc, min(T, fc + 8)):
            S.append((v, 0.0, 1.0, 3))
        return S
    if c["src"] in ("st3", "st32"):  # 3 s of digital silence after the utterance: every frame after the end carries the label
        on = np.nonzero(vad > V4.VAD_ON)[0]
        if not len(on):
            return S
        fe = int(on[-1]) + 1
        for v in range(fe, min(T, fe + 38)):
            S.append((v, 1.0 if c["complete"] else 0.0, 1.0, 4))
        return S
    y, w, ends, pauses = V4.targets(c, vad, T)
    if c["src"] == "st":
        for fe, nf in ends:
            for v in range(fe, min(T, fe + 12)):
                S.append((v, 1.0, 1.0, 4))
        for f0, f1 in pauses:
            for v in range(f0, min(T, f0 + 12)):
                S.append((v, 0.0, 1.0, 4))
        return S
    inpause = np.zeros(T, bool)
    for f0, f1 in pauses:
        inpause[f0:f1] = True
        for v in range(f0, min(f1, T)):
            S.append((v, 0.0, 1.0, 1))
    for fe, nf in ends:
        for v in range(fe, min(T, nf, fe + 12)):
            S.append((v, 1.0, 1.0, 0))
    for s0, e0, nxt, _ in c["turns"]:
        e1 = V4.audible_end(s0, e0, vad, None)
        fs, fe = int(s0 / FRAME), int(np.ceil(e1 / FRAME - 1e-9))
        for v in range(fs + 3, max(fs + 3, fe - 1)):
            if q[v] >= 1 and not inpause[v]:
                S.append((v, 0.0, 1.0, 2))
    return S


def load_block(d, bp, block):
    """block '17' / '4' / ... (pass 1), 'p2' (the pass-2 top of the turn_v4 cache) or '4+8' (concatenated)."""
    if block == "p2":
        return d["e"]
    z = np.load(bp)
    return np.concatenate([z[f"b{b}"] for b in str(block).split("+")], 1)


class SegData:
    """Every clip's arrays in RAM + the sample table. arrays: x (T, 512) of the chosen block (or 'p2' = the pass-2
    top layer of the turn_v4 cache), extra (T, 3 [+12]), n, y, since; samples: (clip, v, label, weight, kind,
    st target or -1, LM completeness score)."""

    def __init__(self, block="17", pros=False, clips=None, max_clips=None, seed=0):
        import random
        man = all_clips()
        idx = {c["id"]: c for c in man}
        vg = val_groups(man)
        if max_clips:
            rng = random.Random(seed)
            base = [c for c in man if c["src"] not in ("cut", "st3", "st32")]
            keep = set(c["id"] for c in rng.sample(base, min(max_clips, len(base))))
            man = [c for c in man if (c["id"] in keep) or (c["src"] == "cut" and c["clip"] in keep)
                   or (c["src"] == "st3" and c["group"][1] in keep)]
        self.clips, self.S = [], []
        texts = {}
        if (W / "texts.jsonl").exists():
            for l in open(W / "texts.jsonl"):
                r = json.loads(l)
                texts[r["id"]] = r["w_of"]
        t0 = time.time()
        for c in man:
            fp, bp = feat_path(c), BLK / f"{c['id']}.npz"
            if not fp.exists() or (block != "p2" and not bp.exists()):
                continue
            d = npz(fp)
            T = len(d["pu"])
            x = load_block(d, bp, block)[:T]
            if len(x) < T:
                continue
            ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float16)
            if pros:
                pf = PROS / f"{c['id']}.npy"
                P = np.load(pf)[:T] if pf.exists() else np.zeros((T, 12), np.float16)
                ex = np.concatenate([ex, P.astype(np.float16)], 1)
            n = d["n"].astype(np.int32)
            ci = len(self.clips)
            val = (cut_group(c, idx) if c["src"] == "cut" else c["group"] if "group" in c
                   else (c["src"], c.get("meeting", c["id"]))) in vg
            self.clips.append({"id": c["id"], "src": c["src"], "x": x.astype(np.float16), "ex": ex, "n": n,
                               "y": d["y"].astype(np.int32), "since": since_token(n), "val": val})
            st = {}
            sf = STT / f"{c['id']}.npz"
            if sf.exists():
                z = np.load(sf)
                st = dict(zip(z["v"].tolist(), z["p"].tolist()))
            lid = "st" + c["id"][3:] if c["src"] == "st3" else c["id"]
            lm = lm_frames(lid, n, texts.get(c["id"], [])) if c["id"] in texts else np.full(T, -20.0, np.float32)
            for v, lab, wt, kind in samples_of(c, d, T):
                self.S.append((ci, v, lab, wt, kind, st.get(v, -1.0), float(lm[v]), val))
        self.S = np.array(self.S, dtype=np.float64)
        self.block, self.pros = block, pros
        log(f"SegData block {block} pros {pros}: {len(self.clips)} clips, {len(self.S)} samples "
            f"(pos {int(self.S[:, 2].sum())}, val {int(self.S[:, 7].sum())}, with st target {(self.S[:, 5] >= 0).sum()}) "
            f"in {time.time() - t0:.0f} s")

    def batch(self, rows, dev, max_tok=48):
        import torch
        B = len(rows)
        D = self.clips[0]["x"].shape[1]
        E_ = self.clips[0]["ex"].shape[1]
        x = np.zeros((B, WIN, D), np.float32)
        ex = np.zeros((B, WIN, E_), np.float32)
        xm = np.zeros((B, WIN), bool)
        tok = np.full((B, max_tok), 1024, np.int64)
        tm = np.zeros((B, max_tok), bool)
        tf = np.zeros((B, 2), np.float32)
        from audioforge.heads.turn_seg import text_feats
        for i, r in enumerate(rows):
            c = self.clips[int(r[0])]
            v = int(r[1])
            a0 = max(0, v - WIN + 1)
            L = v + 1 - a0
            x[i, WIN - L:] = c["x"][a0: v + 1]
            ex[i, WIN - L:] = c["ex"][a0: v + 1]
            xm[i, WIN - L:] = True
            nn_ = int(c["n"][v])
            t = c["y"][:nn_][-max_tok:]
            if len(t):
                tok[i, max_tok - len(t):] = t
                tm[i, max_tok - len(t):] = True
            n0 = int(c["n"][v - WIN]) if v - WIN >= 0 else 0
            tf[i] = text_feats(nn_, n0, int(c["since"][v]))
        to = lambda a: torch.from_numpy(a).to(dev)  # noqa: E731
        return to(x), to(xm), to(ex), to(tok), to(tm), to(tf)


LM_AB = (1.0, 0.0)


def fit_lm_calib(S) -> tuple:
    """Logistic fit P(end) ~ sigmoid(a * lm + b) on the training samples with a word (the completeness target)."""
    m = (S[:, 7] == 0) & (S[:, 6] > -19.5) & (S[:, 4] <= 2)
    if m.sum() < 200:
        return None
    xs, ys = S[m, 6], S[m, 2]
    a, b = 0.5, 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(a * xs + b)))
        ga, gb = np.mean((p - ys) * xs), np.mean(p - ys)
        a, b = a - 0.5 * ga, b - 0.5 * gb
    return float(a), float(b)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def seg_eval(model, data, rows, dev):
    """Sample-level metrics on rows (val): AUC end vs pause/dip/cut, AUC on the first 3 quiet frames only, BCE."""
    import torch
    import torch.nn.functional as F
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            b = data.batch(rows[i: i + 512], dev)
            ps.append(model(*b)["main"].float().cpu().numpy())
    model.train()
    z = np.concatenate(ps)
    p = 1 / (1 + np.exp(-z))
    lab, kind = rows[:, 2], rows[:, 4]
    early = np.zeros(len(rows), bool)
    # "early": the first 4 frames after an end / into a pause (the decision the rule needs)
    prev = None
    cnt = 0
    for i, r in enumerate(rows):
        key = (r[0], r[4])
        cnt = cnt + 1 if (prev is not None and key == prev[0] and r[1] == prev[1] + 1) else 0
        prev = (key, r[1])
        early[i] = cnt < 4
    bce = float(F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(lab)).item())
    nat = kind <= 2
    return {"auc": auc(p[(lab == 1) & nat], p[(lab == 0) & nat]),
            "auc_early": auc(p[(lab == 1) & nat & early], p[(lab == 0) & (kind == 1) & early]),
            "auc_end_vs_cut": auc(p[(lab == 1) & (kind == 0)], p[kind == 3]),
            "acc_st": float(((p >= 0.5) == (lab == 1))[kind == 4].mean()) if (kind == 4).any() else None,
            "bce": round(bce, 4), "n": int(len(rows))}, z


def cmd_train(a):
    import random
    import torch
    import torch.nn.functional as F
    from audioforge.heads.turn_seg import SegTurn
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    od = W / a.tag
    od.mkdir(parents=True, exist_ok=True)
    dev = a.device
    global USE_ST32
    USE_ST32 = not a.no_st32
    data = SegData(a.block, a.pros, max_clips=a.max_clips)
    S = data.S
    global LM_AB
    LM_AB = fit_lm_calib(S)
    tr = S[S[:, 7] == 0]
    va = S[S[:, 7] == 1]
    if len(va) > a.max_val:
        va = va[np.sort(np.random.default_rng(0).choice(len(va), a.max_val, replace=False))]
    D = data.clips[0]["x"].shape[1]
    model = SegTurn(d_in=D, n_extra=3, d=a.d, n_layers=a.layers, heads=4, ff=4 * a.d, win=WIN,
                    use_text=not a.no_text, n_pros=12 if a.pros else 0, dropout=a.dropout,
                    block=a.block).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    step, hist, best = 0, [], -1.0
    last = od / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, hist, best = ck["step"], ck["hist"], ck["best"]
    log(f"{a.tag}: {npar / 1e6:.2f}M params; train {len(tr)} val {len(va)}; lm calib {LM_AB}; step {step}")
    if step >= a.steps:
        return
    # sampling weights: balance kinds (ends, pauses, dips, cuts, smart-turn clips)
    kinds = tr[:, 4].astype(int)
    kw = {0: 1.0, 1: a.w_pause, 2: 0.5, 3: a.w_cut, 4: a.w_st}
    wts = np.array([kw[k] for k in kinds]) / np.bincount(kinds, minlength=5)[kinds]
    wts /= wts.sum()
    rng = np.random.default_rng(a.seed + step)
    t0 = time.time()
    model.train()
    while step < a.steps and time.time() - t0 < a.budget:
        lr = a.lr * min(1.0, (step + 1) / 200) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        rows = tr[rng.choice(len(tr), a.batch, p=wts)]
        b = data.batch(rows, dev)
        if a.pu_drop > 0:  # no voice print (an unenrolled session reads P(user) = P(other) = 0)
            drop = torch.rand(len(rows), device=dev) < a.pu_drop
            b[2][drop, :, 1:3] = 0.0
        out = model(*b)
        lab = torch.tensor(rows[:, 2], dtype=torch.float32, device=dev)
        npos = lab.sum().clamp(min=1)
        pw = ((len(lab) - npos) / npos).clamp(0.1, 10.0)
        loss = F.binary_cross_entropy_with_logits(out["main"], lab, pos_weight=pw)
        stt = torch.tensor(rows[:, 5], dtype=torch.float32, device=dev)
        ms = stt >= 0
        if a.w_st_aux > 0 and ms.any():
            loss = loss + a.w_st_aux * F.binary_cross_entropy_with_logits(out["st"][ms], stt[ms])
        if a.w_compl > 0 and "compl" in out and LM_AB is not None:
            lm = torch.tensor(rows[:, 6], dtype=torch.float32, device=dev)
            ml = lm > -19.5
            if ml.any():
                ct = torch.sigmoid(LM_AB[0] * lm + LM_AB[1])
                loss = loss + a.w_compl * F.binary_cross_entropy_with_logits(out["compl"][ml], ct[ml])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % a.eval_every == 0 or step == a.steps:
            r, _ = seg_eval(model, data, va, dev)
            r = {"step": step, "loss": round(float(loss), 4), **r}
            hist.append(r)
            log(r)
            if r["auc"] > best:
                best = r["auc"]
                torch.save({"cfg": model.cfg, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "step": step, "val": r, "args": vars(a), "lm_ab": LM_AB}, od / "model.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist,
                        "best": best}, last)
            (od / "history.json").write_text(json.dumps({"args": vars(a), "params": npar, "hist": hist}, indent=1))
    log(f"stopped at {step} ({time.time() - t0:.0f} s)")
    if step >= a.steps or a.calibrate_now:
        calibrate(od, data, va, dev)


def calibrate(od, data, va, dev):
    """Temperature + bias on the held-out split (BCE), stored in the model's ``temp`` buffer."""
    import torch
    from audioforge.heads.turn_seg import SegTurn
    ck = torch.load(od / "model.pt", map_location="cpu", weights_only=False)
    model = SegTurn(**ck["cfg"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    _, z = seg_eval(model, data, va, dev)
    y = va[:, 2]
    T_, b_ = 1.0, 0.0
    zt, yt = torch.tensor(z), torch.tensor(y)
    p = torch.tensor([T_, b_], requires_grad=True)
    o = torch.optim.LBFGS([p], max_iter=200)

    def cl():
        o.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(zt / p[0] + p[1], yt)
        l.backward()
        return l
    o.step(cl)
    ck["state_dict"]["temp"] = p.detach().float()
    ck["calib"] = [float(p[0]), float(p[1])]
    torch.save(ck, od / "model.pt")
    log(f"calibrated: T {float(p[0]):.3f} b {float(p[1]):.3f}")


def cmd_calib(a):
    import torch
    ck = torch.load(W / a.tag / "model.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    data = SegData(ar["block"], ar["pros"], max_clips=ar.get("max_clips"))
    va = data.S[data.S[:, 7] == 1]
    if len(va) > ar["max_val"]:
        va = va[np.sort(np.random.default_rng(0).choice(len(va), ar["max_val"], replace=False))]
    calibrate(W / a.tag, data, va, a.device)


def load_seg(path, dev="cpu"):
    import torch
    from audioforge.heads.turn_seg import SegTurn
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SegTurn(**ck["cfg"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def seg_probs(model, x, ex, n, y, dev="cpu", bs=256) -> np.ndarray:
    """p at every frame of one clip (the window ending at each frame), calibrated."""
    import torch
    T = len(n)
    c = {"x": x, "ex": ex, "n": n, "y": y, "since": since_token(n)}
    fake = type("D", (), {})()
    fake.clips = [c]
    rows = np.array([[0, v] for v in range(T)], np.float64)
    out = []
    with torch.no_grad():
        for i in range(0, T, bs):
            b = SegData.batch(fake, rows[i: i + bs], dev, model.max_tok)
            out.append(model.prob(*b).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def eval_inputs(key_or_clip: str, block, pros):
    """(x, ex, n, y) of an eval session / clip delivery / smart-turn test clip."""
    if key_or_clip.startswith("stt_"):
        d = npz(STEST4 / f"st_{key_or_clip[4:]}.npz")
        bid = key_or_clip
    elif key_or_clip.startswith("clip_"):
        d = npz(EVALF4 / f"{key_or_clip}.npz")
        bid = key_or_clip
    else:
        d = npz(EVALF4 / f"{key_or_clip}.npz")
        bid = f"ev_{key_or_clip}"
    T = len(d["pu"])
    x = load_block(d, BLK / f"{bid}.npz", block)[:T]
    ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
    if pros:
        ex = np.concatenate([ex, np.load(PROS / f"{bid}.npy")[:T].astype(np.float32)], 1)
    return x.astype(np.float32), ex, d["n"].astype(np.int32), d["y"].astype(np.int32), d


def cmd_evaldump(a):
    import torch
    import eot_latency as E
    torch.set_num_threads(2)
    model, ck = load_seg(W / a.tag / "model.pt", a.device)
    block = str(ck["args"]["block"])
    pros = ck["args"]["pros"]
    od = W / f"dump_{a.tag}"
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for f in sorted(E.DUMP.glob("*.json")):
        d = json.loads(f.read_text())
        x, ex, n, y, _ = eval_inputs(d["key"], block, pros)
        p = seg_probs(model, x, ex, n, y, a.device)
        assert len(p) == len(d["head"]["p"]), (d["key"], len(p), len(d["head"]["p"]))
        d["head"]["p"] = [float(v) for v in p]
        d["turn_ends"] = []
        (od / f.name).write_text(json.dumps(d))
    cv = json.loads((E.WORK / "clip_frames.json").read_text())
    for i, var in enumerate(E.CLIP_VARIANTS):
        x, ex, n, y, _ = eval_inputs(f"clip_{i}", block, pros)
        p = seg_probs(model, x, ex, n, y, a.device)
        assert len(p) == len(cv[var]["head"]["p"])
        cv[var]["head"]["p"] = [float(v) for v in p]
    (W / f"dump_{a.tag}_clip_frames.json").write_text(json.dumps(cv))
    log(f"evaldump {a.tag}: {time.time() - t0:.0f} s -> {od}")


def cmd_stest(a):
    """The 399 smart-turn v3.2 test clips (turn_v4.cmd_stest protocol): p on the frame 160 ms after the clip's last
    served-VAD speech frame >= theta (0.5; and the accuracy-best theta on the smart-turn train clips)."""
    import torch
    import turn_v4 as V4
    from audioforge.datasets import smartturn as ST
    torch.set_num_threads(2)
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    sp = ST.split_indices(meta)
    lab = np.asarray(meta["complete"], bool)
    res = {}
    for tag in a.tags:
        model, ck = load_seg(W / tag / "model.pt", a.device)
        block, pros = str(ck["args"]["block"]), ck["args"]["pros"]

        def decide(x, ex, n, y, d, off=2):
            p = seg_probs(model, x, ex, n, y, a.device)
            v = np.nonzero(np.asarray(d["vad"], float) > V4.VAD_ON)[0]
            return float(p[min(len(p) - 1, (v[-1] if len(v) else 0) + off)])
        pt = np.array([decide(*eval_inputs(f"stt_{i:05d}", block, pros)) for i in sp["eval"]])
        ptr = []
        for i in sp["train"][:: 3]:
            cid = f"st_{i:05d}"
            d = npz(FEATS4 / f"{cid}.npz")
            T = len(d["pu"])
            x = load_block(d, BLK / f"{cid}.npz", block)[:T]
            ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
            if pros:
                ex = np.concatenate([ex, np.load(PROS / f"{cid}.npy")[:T].astype(np.float32)], 1)
            ptr.append(decide(x.astype(np.float32), ex, d["n"], d["y"], d))
        ptr = np.array(ptr)
        yt, ytr = lab[sp["eval"]], lab[sp["train"][:: 3]]
        grid = np.linspace(0.01, 0.999, 200)
        thb = float(grid[np.argmax([((ptr >= t) == ytr).mean() for t in grid])])
        r = {}
        for nm, th in (("0.5", 0.5), ("train_best", thb)):
            pred = pt >= th
            r[nm] = {"theta": round(th, 3), "accuracy": round(100 * float((pred == yt).mean()), 1),
                     "recall_complete": round(100 * float(pred[yt].mean()), 1),
                     "recall_incomplete": round(100 * float((~pred[~yt]).mean()), 1)}
        r["auc"] = auc(pt[yt], pt[~yt])
        res[tag] = r
        log(tag, r)
    d = json.loads(OUT.read_text()).get("smartturn_test", {}) if OUT.exists() else {}
    d.update(res)
    save("smartturn_test", d)


def frame_stats(dump_dir: Path) -> dict:
    """On a dump: frame AUC (frames of the first 320 ms after each scored end vs frames inside the turn; all frames
    and quiet frames only = served VAD < 0.4), median p inside turns (quiet frames) / in pauses >= 160 ms / at ends,
    and head reach (eot_latency.head_reach)."""
    import eot_latency as E
    os.environ["EOT_DUMP"] = str(dump_dir)
    E.DUMP = Path(dump_dir)
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump]
    out = {"head_reach": E.head_reach(dump, sess, ths=(0.5, 0.8, 0.9, 0.95, 0.98, 0.99))}
    for scope, sets in E.SCOPES.items():
        pe, pin, pinq, pp = [], [], [], []
        for ss in sess:
            if ss["set"] not in sets:
                continue
            h = dump[ss["key"]]["head"]
            v = np.asarray(h["v"])
            t0 = v * FRAME  # label grid time of each frame's start
            p, vad = np.asarray(h["p"]), np.asarray(h["vad"])
            for i, (s0, e1) in enumerate(ss["user_turns"]):
                if not ss["scored"][i]:
                    continue
                pe += list(p[(t0 >= e1) & (t0 < e1 + 0.32)])
                m = (t0 >= s0) & (t0 + FRAME <= e1 - 0.08)
                pin += list(p[m])
                pinq += list(p[m & (vad < Q_VAD)])
                q = quiet_run(vad)
                pp += list(p[m & (q >= 2)])  # >= 160 ms quiet inside the turn: pause frames
        out[scope] = {"frame_auc_all": auc(pe, pin), "frame_auc_quiet": auc(pe, pinq),
                      "p50_in_turn": float(np.median(pin)) if pin else None,
                      "p50_in_turn_quiet": float(np.median(pinq)) if pinq else None,
                      "p50_pause": float(np.median(pp)) if pp else None,
                      "p50_end_320ms": float(np.median(pe)) if pe else None,
                      "n_end_frames": len(pe), "n_turn_frames": len(pin), "n_pause_frames": len(pp)}
    return out


def cmd_report(a):
    d = W / f"dump_{a.tag}" if a.tag != "served" else W4 / "dump_served"
    r = frame_stats(d)
    for sc in ("two_party_user", "ami"):
        log(a.tag, sc, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[sc].items()})
        hr = r["head_reach"][sc]
        log("  reach p>=0.9 / 0.95 within 200/300/500:", list(hr["p>=0.9"].values()), list(hr["p>=0.95"].values()))
    res = json.loads(OUT.read_text()).get("frame_stats", {}) if OUT.exists() else {}
    res[a.tag] = r
    save("frame_stats", res)


# --------------------------------------------------------------------------- quiet gate (energy floor) + warm-up guard
ENERGY = W / "energy"


def frame_abs_db(x: np.ndarray, T: int) -> np.ndarray:
    """dBFS of each 80 ms frame v (samples [1280 v, 1280 (v + 1)))."""
    n = T * 1280
    x = np.pad(np.asarray(x, np.float32), (0, max(0, n - len(x))))[:n]
    return (10 * np.log10(np.mean(x.reshape(T, 1280).astype(np.float64) ** 2, 1) + 1e-10)).astype(np.float32)


def cmd_energy(a):
    import eot_latency as E
    import eot_assistant as EA
    ENERGY.mkdir(parents=True, exist_ok=True)
    dump = E.load_dump()
    n = 0
    for s in E.sessions():
        if s["key"] in dump and not (ENERGY / f"{s['key']}.npy").exists():
            T = len(np.load(EVALF4 / f"{s['key']}.npz")["pu"])
            np.save(ENERGY / f"{s['key']}.npy", frame_abs_db(E.read_audio(s), T))
            n += 1
    for c in EA.clips():
        f = ENERGY / f"asst_{c['key']}.npy"
        if not f.exists():
            T = len(np.load(W / "asst" / f"asst_{c['key']}.npz")["pu"])
            np.save(f, frame_abs_db(EA.audio(c), T))
            n += 1
    from audioforge.data import load_wav
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
    for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items()):
        T = len(np.load(EVALF4 / f"clip_{i}.npz")["pu"])
        np.save(ENERGY / f"clip_{i}.npy", frame_abs_db(fn(x0), T))
    log(f"energy: {n} new")


class QuietGate:
    """Served-path quiet gate (streaming, per frame): the frame counts as quiet for the turn rule when the served VAD
    is below its line OR the frame's level is < floor + x_db (floor: a minimum follower rising 0.5 dB/s) OR < abs_db.
    Warm-up guard: nothing is quiet until the VAD has heard >= guard_frames speech frames (> 0.5) since the start,
    so no turn_end can fire before the user's first speech."""

    def __init__(self, x_db=None, abs_db=-60.0, guard_frames=4, rise_db=0.04):
        self.x, self.abs, self.guard, self.rise = x_db, abs_db, guard_frames, rise_db
        self.floor = None
        self.nsp = 0

    def __call__(self, vad: float, db: float) -> float:
        """-> the VAD value the rule should read (0.0 when the energy gate says quiet, 1.0 before the guard)."""
        self.floor = db if self.floor is None else min(db, self.floor + self.rise)
        if vad > 0.5 and db > self.floor + 10:
            self.nsp += 1
        if self.guard and self.nsp < self.guard:
            return 1.0
        if self.x is not None and (db < self.floor + self.x or db < self.abs):
            return 0.0
        return vad


def gated_vad(vad, db, x_db, guard):
    g = QuietGate(x_db, guard_frames=guard)
    return np.array([g(float(a), float(b)) for a, b in zip(vad, db)])


# --------------------------------------------------------------------------- joint scan: calls + AMI + assistant
GOAL = {"calls_fi": 20.2, "calls_miss": 7.3, "ami_fi": 10.5, "ami_miss": 33.5}


def asst_probs(tag: str) -> dict:
    """{key: p per frame} of a trained model on the 399 eot_assistant sessions (W/asst inputs, no print)."""
    import eot_assistant as EA
    model, ck = load_seg(W / tag / "model.pt", "cpu")
    block, pros = str(ck["args"]["block"]), ck["args"]["pros"]
    out = {}
    for c in EA.clips():
        cid = f"asst_{c['key']}"
        d = npz(W / "asst" / f"{cid}.npz")
        T = len(d["pu"])
        x = load_block(d, BLK / f"{cid}.npz", block)[:T]
        ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)
        if pros:
            ex = np.concatenate([ex, np.load(PROS / f"{cid}.npy")[:T].astype(np.float32)], 1)
        out[c["key"]] = seg_probs(model, x.astype(np.float32), ex, d["n"].astype(np.int32), d["y"].astype(np.int32))
    return out


def load_sets(tag: str):
    """(calls/AMI dump, assistant dump, clip frames) with the tag's p (tag 'served' = today's head)."""
    import eot_latency as E
    import eot_assistant as EA
    dd = W4 / "dump_served" if tag == "served" else W / f"dump_{tag}"
    cvf = W4 / "dump_served_clip_frames.json" if tag == "served" else W / f"dump_{tag}_clip_frames.json"
    dump = {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(dd.glob("*.json")))}
    ad = EA.load_dump()
    if tag != "served":
        cache = W / f"asst_p_{tag}.json"
        if cache.exists():
            ap = {k: np.asarray(v) for k, v in json.loads(cache.read_text()).items()}
        else:
            ap = asst_probs(tag)
            cache.write_text(json.dumps({k: [round(float(x), 5) for x in v] for k, v in ap.items()}))
        for k, d in ad.items():
            v = np.asarray(d["head"]["v"])
            d["head"]["p"] = [float(ap[k][min(j, len(ap[k]) - 1)]) for j in v]
    cv = json.loads(cvf.read_text())
    return dump, ad, cv


def transform(dump: dict, x_db, guard, prefix="") -> dict:
    """Copies of the sessions whose VAD track is the quiet gate's (QuietGate) output; no-op for (None, 0)."""
    out = {}
    for k, d in dump.items():
        d2 = {kk: vv for kk, vv in d.items() if kk != "_cache"}
        if x_db is not None or guard:
            h = dict(d["head"])
            f = ENERGY / f"{prefix}{k}.npy"
            db = np.load(f)
            v = np.asarray(h["v"])
            h["vad"] = gated_vad(np.asarray(h["vad"], float), db[np.clip(v, 0, len(db) - 1)], x_db, guard).tolist()
            d2["head"] = h
        out[k] = d2
    return out


def scan_rules() -> list:
    import itertools
    import eot_latency as E
    base = dict(E.CHOSEN)
    R = []
    for vt, k, th in itertools.product((0.4, 0.5), (1, 2, 3), (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
        for F in (6, 8, 10):
            R.append({**base, "vad_thr": vt, "k": k, "th": th, "fallback_f": F})
        for cap, fl, pstar in itertools.product((12, 20, 37), (6, 8, 10), (0.2, 0.35, 0.5)):
            if fl >= cap:
                continue
            R.append({**base, "vad_thr": vt, "k": k, "th": th, "fallback_f": None, "dyn": (cap, fl, (cap - fl) / pstar)})
    return R


def rname(r):
    s = f"VAD<{r['vad_thr']} sil>={r['k'] * 80}ms & p>={r['th']}"
    if r.get("dyn"):
        c, f, a_ = r["dyn"]
        s += f" | wait clamp({c * 80}-{a_ * 80:.0f}*p, {f * 80}, {c * 80}) ms"
    else:
        s += f" | fallback {r['fallback_f'] * 80}ms"
    return s + (f" | gate {r['gate']}" if r.get("gate") else "")


def cmd_scan3(a):
    import eot_latency as E
    import eot_assistant as EA
    if a.labels:
        os.environ["EOT_LABELS"] = a.labels
        E.LABELS = a.labels
    dump0, ad0, cv0 = load_sets(a.tag)
    sess = [s for s in E.sessions() if s["key"] in dump0]
    asess = EA.sessions(ad0)
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump0.items()}
    acomp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in ad0.items()}
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    gates = [(None, 0), (None, 4), (6.0, 4), (10.0, 4)] if not a.quick else [(None, 0), (12.0, 4)]
    rows = []
    t0 = time.time()
    for xg, gd in gates:
        dump = transform(dump0, xg, gd)
        ad = transform(ad0, xg, gd, prefix="asst_")
        cv = cv0
        if xg is not None or gd:  # the clip deliveries under the same gate
            cv = {}
            for i, (var, dd) in enumerate(cv0.items()):
                h = dict(dd["head"])
                db = np.load(ENERGY / f"clip_{i}.npy")
                v = np.arange(len(h["vad"]))
                h["vad"] = gated_vad(np.asarray(h["vad"], float), db[np.clip(v, 0, len(db) - 1)], xg, gd).tolist()
                cv[var] = {**{kk: vv for kk, vv in dd.items() if kk != "_cache"}, "head": h}
        rules = [dict(E.CHOSEN)] + scan_rules() if not a.quick else [dict(E.CHOSEN)]
        for r in rules:
            r = {**r, "gate": f"x{xg}_g{gd}" if (xg is not None or gd) else None}
            per = {sc: [] for sc in E.SCOPES}
            for ss in sess:
                t = [x + comp[ss["key"]] for x, _ in E.sim_room(dump[ss["key"]], r)]
                for sc, sets in E.SCOPES.items():
                    if ss["set"] in sets:
                        per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
            pc = {sc: E.pool(v) for sc, v in per.items()}
            at = {s["key"]: [x + acomp[s["key"]] for x, _ in E.sim_room(ad[s["key"]], r)] for s in asess}
            asc = EA.score_system(asess, at, acomp)
            rows.append({"rule": r, "name": rname(r), "calls": pc["two_party_user"], "ami": pc["ami"],
                         "asst": {"acc": asc["accuracy_pct"], "p50": asc["complete"]["eot_total_ms_p50"],
                                  "p95": asc["complete"]["eot_total_ms_p95"],
                                  "ff_inc": asc["incomplete"]["false_fire_pct"],
                                  "miss_c": asc["complete"]["missed_pct"], "early_c": asc["complete"]["early_fire_pct"],
                                  "pre_speech": asc["clips_with_pre_speech_turn_end"]},
                         "_cv": cv})
        log(f"gate {xg},{gd}: {len(rows)} rules so far ({time.time() - t0:.0f} s)")

    def cut(x):
        if "clip_cut_in" not in x:
            x["clip_cut_in"] = [v for v, dd in x["_cv"].items() if E.clip_cuts(dict(dd), x["rule"], meta)["cuts_user_turn"]]
        return x["clip_cut_in"]
    g = GOAL if not a.labels else {"calls_fi": 19.3, "calls_miss": 6.4, "ami_fi": 10.5, "ami_miss": 33.5}
    ok = lambda x, fi=g["calls_fi"]: (x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["false_interruption_pct"] <= fi  # noqa
                                      and x["calls"]["missed_pct"] <= g["calls_miss"] and x["ami"]["false_interruption_pct"] <= g["ami_fi"]
                                      and x["ami"]["missed_pct"] <= g["ami_miss"])
    key = lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"])  # noqa
    best = next((x for x in sorted([x for x in rows if ok(x)], key=key) if not cut(x)), None)
    fast = next((x for x in sorted([x for x in rows if x["calls"]["eot_total_ms_p50"] is not None
                                    and x["calls"]["false_interruption_pct"] <= 25.0
                                    and x["calls"]["missed_pct"] <= g["calls_miss"]], key=key) if not cut(x)), None)
    asst_ok = [x for x in rows if x["asst"]["acc"] is not None and x["asst"]["acc"] >= 90 and (x["asst"]["p50"] or 1e9) <= 400]
    asst_best = max(rows, key=lambda x: (x["asst"]["acc"] or 0, -(x["asst"]["p50"] or 1e9)))
    both = [x for x in sorted([x for x in rows if ok(x)], key=lambda x: -(x["asst"]["acc"] or 0)) if not cut(x)][:10]
    strip = lambda x: None if x is None else {k: v for k, v in x.items() if k != "_cv"}  # noqa
    res = {"tag": a.tag, "labels": a.labels or "original", "goal": g, "n_rules": len(rows), "shipped": strip(rows[0]),
           "best_goal": strip(best), "best_fast_fi25": strip(fast), "n_asst_goal": len(asst_ok),
           "asst_best": strip(asst_best), "goal_rules_by_asst_acc": [strip(x) for x in both],
           "asst_goal_fastest_calls": strip(min(asst_ok, key=key)) if asst_ok else None,
           "gate_shipped_rule": {x["rule"]["gate"] or "none": strip(x) for x in rows
                                 if {kk: vv for kk, vv in x["rule"].items() if kk != "gate"} == dict(E.CHOSEN)}}
    tagl = "audible" if a.labels else "orig"
    (W / f"scan3_{a.tag}_{tagl}.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda x: None if x is None else (f"calls {x['calls']['eot_total_ms_p50']}/{x['calls']['eot_total_ms_p95']} "  # noqa
                                          f"{x['calls']['false_interruption_pct']}/{x['calls']['missed_pct']} | AMI "
                                          f"{x['ami']['eot_total_ms_p50']} {x['ami']['false_interruption_pct']}/"
                                          f"{x['ami']['missed_pct']} | asst acc {x['asst']['acc']} p50 {x['asst']['p50']} "
                                          f"ffinc {x['asst']['ff_inc']} | {x['name']}")
    log("shipped:", f(rows[0]))
    for gname, x in res["gate_shipped_rule"].items():
        log(f"  shipped rule + gate {gname}:", f(x))
    log("best under goal:", f(best))
    log("best FI<=25:", f(fast))
    log("asst best:", f(asst_best))
    log(f"asst >= 90 % & <= 400 ms: {len(asst_ok)} rules; fastest calls among them:", f(res["asst_goal_fastest_calls"]))
    for x in both[:3]:
        log("  goal + best asst:", f(x))


# --------------------------------------------------------------------------- scan4: the SERVED policy class (energy gate + turn_model)
class _ReplayGate:
    """Stands in for policies.EnergyGate with its outputs precomputed per session (the gate reads only the energy
    and the VAD, never the rule, so its (quiet, onset) sequence is the same under every rule)."""

    def __init__(self, qo, warmup_frames):
        self.qo, self.i, self.n_onset, self.warmup_frames = qo, 0, 0, warmup_frames

    @property
    def warm(self):
        return self.n_onset >= self.warmup_frames

    def update(self, e, vad):
        q, o = self.qo[self.i]
        self.i += 1
        self.n_onset += o
        return q, o


def gate_trace(vad, db, quiet_db):
    import math
    from audioforge.server.constants import ENERGY_GATE as G, FRAME_MS
    from audioforge.server.policies import EnergyGate
    g = EnergyGate(quiet_db=quiet_db, onset_db=G["onset_db"], window_s=G["window_s"], pct=G["pct"],
                   warmup_frames=int(math.ceil(G["warmup_ms"] / FRAME_MS)))
    return [g.update(float(e), float(v)) for v, e in zip(vad, db)], g.warmup_frames


def run_policy(d, db, rule, gcache):
    """The served VadHeadPolicy (+ EnergyGate replay, + the v5 classifier as turn_model) over one session's frames
    -> [(t, path)]."""
    from audioforge.server.policies import VadHeadPolicy
    h = d["head"]
    p = h["p"]
    gk = rule["quiet_db"]
    if gk not in gcache:
        gcache[gk] = gate_trace(h["vad"], db, gk)
    qo, wf = gcache[gk]
    tm = None
    if rule["mode"] == "model":
        tm = lambda v, onset_v: (p[v], 0.0)  # noqa: E731
    pol = VadHeadPolicy(rule.get("th", 0.99), rule["k"], rule["fb"], rule["vad_thr"], others=rule.get("others", (12, 8)),
                        gate=_ReplayGate(qo, wf) if rule["gate"] else None, turn_model=tm,
                        model_vad_thr=rule.get("mvt"), model_p=rule.get("mp", 0.5),
                        model_quiet_only=rule.get("mqo", False))
    out = []
    pu, po, vad, t = h["pu"], h["po"], h["vad"], h["t"]
    reask = rule.get("reask", False)
    for v in range(len(p)):
        n0 = len(pol.model_calls)
        ev = pol.update(p[v], vad[v], pu[v], po[v], float(db[v]) if rule["gate"] else None)
        if reask and len(pol.model_calls) > n0 and not pol.model_calls[-1]["complete"]:
            pol._asked = -1  # re-classify at every further quiet frame of the same run
        if ev is not None:
            out.append((float(t[v]), ev["path"]))
    return out


def scan4_rules(quick=False):
    import itertools
    R = [{"mode": "head", "gate": False, "quiet_db": None, "k": 2, "th": 0.99, "fb": 8, "vad_thr": 0.4, "name": "shipped rule, no gate"},
         {"mode": "head", "gate": True, "quiet_db": None, "k": 2, "th": 0.99, "fb": 8, "vad_thr": 0.4, "name": "shipped rule + gate"}]
    if quick:
        return R
    for k, th, fb in itertools.product((1, 2, 3), (0.5, 0.7, 0.8, 0.9, 0.95, 0.99), (8, 12)):
        R.append({"mode": "head", "gate": True, "quiet_db": None, "k": k, "th": th, "fb": fb, "vad_thr": 0.4})
    for k, mvt, mp, fb, qd, ra in itertools.product((1, 2, 3), (0.4, 0.5, 0.6), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97),
                                                    (8, 9, 12, 37), (None, 6.0), (False, True)):
        R.append({"mode": "model", "gate": True, "quiet_db": qd, "mqo": qd is not None, "k": k, "mvt": mvt, "mp": mp,
                  "fb": fb, "vad_thr": 0.4, "reask": ra})
    for k, mvt, mp, fb, oth, vt in itertools.product((1, 2, 3), (0.4, 0.5, 0.6), (0.4, 0.5, 0.6, 0.7, 0.8), (8, 9),
                                                     ((12, 8), (8, 8), (10, 8)), (0.4, 0.5, 0.6)):
        R.append({"mode": "model", "gate": True, "quiet_db": None, "k": k, "mvt": mvt, "mp": mp, "fb": fb,
                  "vad_thr": vt, "reask": True, "others": oth})
    return R


def r4name(r):
    if r.get("name"):
        return r["name"]
    if r["mode"] == "head":
        return f"head: VAD<{r['vad_thr']} sil>={r['k'] * 80}ms & p>={r['th']} | fallback {r['fb'] * 80}ms | gate"
    return (f"v5 classifier at {r['k'] * 80}ms quiet (VAD<{r['mvt']}{' or energy<floor+%g dB' % r['quiet_db'] if r['quiet_db'] else ''})"
            f" p>{r['mp']}{' (re-asked every quiet frame)' if r.get('reask') else ''} | fallback {r['fb'] * 80}ms (VAD<{r['vad_thr']})"
            f"{' | others %d,%d' % tuple(x * 80 for x in r['others']) if r.get('others') else ''} | gate")


def cmd_scan4(a):
    import eot_latency as E
    import eot_assistant as EA
    if a.labels:
        os.environ["EOT_LABELS"] = a.labels
        E.LABELS = a.labels
    dump, ad, cv = load_sets(a.tag)
    sess = [s for s in E.sessions() if s["key"] in dump]
    asess = EA.sessions(ad)
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    acomp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in ad.items()}
    dbs = {k: np.load(ENERGY / f"{k}.npy") for k in dump}
    adbs = {k: np.load(ENERGY / f"asst_{k}.npy") for k in ad}
    cdbs = {var: np.load(ENERGY / f"clip_{i}.npy") for i, var in enumerate(cv)}
    for dd, dbd in ((dump, dbs), (ad, adbs)):
        for k, d in dd.items():
            v = np.asarray(d["head"]["v"])
            dbd[k] = dbd[k][np.clip(v, 0, len(dbd[k]) - 1)]
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta.get("user_turn", [3.9, 10.8]) if isinstance(meta, dict) else (3.9, 10.8)
    gc = {k: {} for k in list(dump) + [f"a:{k}" for k in ad] + [f"c:{k}" for k in cv]}
    rows = []
    t0 = time.time()
    rules = scan4_rules(a.quick)
    for i, r in enumerate(rules):
        per = {sc: [] for sc in E.SCOPES}
        for ss in sess:
            t = [x + comp[ss["key"]] for x, _ in run_policy(dump[ss["key"]], dbs[ss["key"]], r, gc[ss["key"]])]
            for sc, sets in E.SCOPES.items():
                if ss["set"] in sets:
                    per[sc].append(E.score_session(t, ss, comp[ss["key"]]))
        pc = {sc: E.pool(v) for sc, v in per.items()}
        paths = {}
        at = {}
        for s_ in asess:
            ev = run_policy(ad[s_["key"]], adbs[s_["key"]], r, gc["a:" + s_["key"]])
            at[s_["key"]] = [x + acomp[s_["key"]] for x, _ in ev]
        asc = EA.score_system(asess, at, acomp)
        cut = []
        for var, dd in cv.items():
            te = run_policy(dd, cdbs[var], r, gc["c:" + var])
            if any(3.9 <= t_ < 10.8 - 0.08 for t_, _ in te):
                cut.append(var)
        rows.append({"rule": r, "name": r4name(r), "calls": pc["two_party_user"], "ami": pc["ami"],
                     "asst": {"acc": asc["accuracy_pct"], "p50": asc["complete"]["eot_total_ms_p50"],
                              "p95": asc["complete"]["eot_total_ms_p95"], "ff_inc": asc["incomplete"]["false_fire_pct"],
                              "miss_c": asc["complete"]["missed_pct"], "early_c": asc["complete"]["early_fire_pct"],
                              "pre_speech": asc["clips_with_pre_speech_turn_end"]}, "clip_cut_in": cut})
        if i % 50 == 0:
            log(f"{i + 1}/{len(rules)} rules ({time.time() - t0:.0f} s)")
    g = GOAL if not a.labels else {"calls_fi": 19.3, "calls_miss": 6.4, "ami_fi": 10.5, "ami_miss": 33.5}
    okc = lambda x, fi: (x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["false_interruption_pct"] <= fi  # noqa
                         and x["calls"]["missed_pct"] <= g["calls_miss"] and x["ami"]["false_interruption_pct"] <= g["ami_fi"]
                         and x["ami"]["missed_pct"] <= g["ami_miss"] and not x["clip_cut_in"])
    key = lambda x: (x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"])  # noqa
    goal = sorted([x for x in rows if okc(x, g["calls_fi"])], key=key)
    fast = sorted([x for x in rows if x["calls"]["eot_total_ms_p50"] is not None and x["calls"]["false_interruption_pct"] <= 25.0
                   and x["calls"]["missed_pct"] <= g["calls_miss"] and not x["clip_cut_in"]], key=key)
    aok = [x for x in rows if (x["asst"]["acc"] or 0) >= 90 and (x["asst"]["p50"] or 1e9) <= 400]
    abest = sorted(rows, key=lambda x: (-(x["asst"]["acc"] or 0), x["asst"]["p50"] or 1e9))
    res = {"tag": a.tag, "labels": a.labels or "original", "goal": g, "n_rules": len(rows),
           "shipped_no_gate": rows[0], "shipped_gate": rows[1], "best_goal": goal[0] if goal else None,
           "goal_top": goal[:10], "best_fast_fi25": fast[0] if fast else None, "fast_top": fast[:5],
           "asst_goal_n": len(aok), "asst_goal_fastest_calls": min(aok, key=key) if aok else None,
           "asst_top": abest[:10],
           "goal_by_asst_acc": sorted([x for x in rows if okc(x, g["calls_fi"])], key=lambda x: -(x["asst"]["acc"] or 0))[:10],
           "all": rows if a.keep_all else None}
    tagl = "audible" if a.labels else "orig"
    (W / f"scan4_{a.tag}_{tagl}.json").write_text(json.dumps(res, indent=1, default=float))
    f = lambda x: None if x is None else (f"calls {x['calls']['eot_total_ms_p50']}/{x['calls']['eot_total_ms_p95']} "  # noqa
                                          f"{x['calls']['false_interruption_pct']}/{x['calls']['missed_pct']} | AMI "
                                          f"{x['ami']['eot_total_ms_p50']} {x['ami']['false_interruption_pct']}/"
                                          f"{x['ami']['missed_pct']} | asst acc {x['asst']['acc']} p50 {x['asst']['p50']} "
                                          f"ff {x['asst']['ff_inc']} | cut {len(x['clip_cut_in'])} | {x['name']}")
    log("shipped, no gate:", f(rows[0]))
    log("shipped + gate:", f(rows[1]))
    log("best under goal:", f(res["best_goal"]))
    log("best FI<=25:", f(res["best_fast_fi25"]))
    log(f"asst >= 90 % & <= 400 ms: {len(aok)}; fastest calls:", f(res["asst_goal_fastest_calls"]))
    for x in abest[:3]:
        log("  asst top:", f(x))
    for x in res["goal_by_asst_acc"][:3]:
        log("  goal + best asst:", f(x))


def cmd_blend(a):
    """dump_<out> = the mean of the tags' p (calls / AMI dumps, clip deliveries, assistant p)."""
    tags = a.tags
    od = W / f"dump_{a.out}"
    od.mkdir(parents=True, exist_ok=True)
    for f in sorted((W / f"dump_{tags[0]}").glob("*.json")):
        ds = [json.loads((W / f"dump_{t}" / f.name).read_text()) for t in tags]
        d = ds[0]
        d["head"]["p"] = list(np.mean([np.asarray(x["head"]["p"]) for x in ds], 0))
        (od / f.name).write_text(json.dumps(d, default=float))
    cvs = [json.loads((W / f"dump_{t}_clip_frames.json").read_text()) for t in tags]
    for var in cvs[0]:
        cvs[0][var]["head"]["p"] = list(np.mean([np.asarray(c[var]["head"]["p"]) for c in cvs], 0))
    (W / f"dump_{a.out}_clip_frames.json").write_text(json.dumps(cvs[0], default=float))
    aps = [json.loads((W / f"asst_p_{t}.json").read_text()) for t in tags]
    (W / f"asst_p_{a.out}.json").write_text(json.dumps({k: list(np.mean([np.asarray(x[k]) for x in aps], 0))
                                                         for k in aps[0]}, default=float))
    log(f"blend {tags} -> {a.out}")


def cmd_served_v5(a):
    """The v5 classifier inside the real served single-mode session (ASRStream.attach_seg) on the bundled clip under
    the six deliveries, and on a few eval sessions: served seg_prob(v) vs the offline seg_probs on the cached inputs;
    plus the classifier's compute per call and the prosody cost per chunk."""
    import torch
    from single_model import single_engine
    import audioforge.serve as S
    import eot_latency as E
    from audioforge.data import load_wav
    from audioforge.heads.turn_seg import load_seg_turn
    torch.set_num_threads(2)
    model = load_seg_turn(W / a.tag / "model.pt")
    _, ck = load_seg(W / a.tag / "model.pt")
    block, pros = str(ck["args"]["block"]), ck["args"]["pros"]
    eng = single_engine(afm=E.SERVED_AFM)
    eng.warmup()
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), SR).astype(np.float32)
    emb = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json").read_text())
    res = {}
    items = [(f"clip_{i}", fn(x0), emb) for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items())]
    sess = {s["key"]: s for s in E.sessions()}
    for k in a.keys:
        items.append((k, E.read_audio(sess[k]), sess[k]["embedding"]))
    for name, x, e in items:
        s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))
        s.arm_enrollment("enroll", 0, embedding=e)
        s.asr.attach_seg(model)
        for i in range(0, len(x), 320):
            s.process(x[i:i + 320])
        s.finish()
        T = s.asr.seg.v + 1
        served = np.array([s.asr.seg_prob(v) for v in range(max(0, T - 150), T)])
        xx, ex, n, y, _ = eval_inputs(name, block, pros)
        off = seg_probs(model, xx, ex, n, y)[max(0, T - 150): T]
        L = min(len(off), len(served))
        dif = np.abs(off[:L] - served[:L])
        res[name] = {"frames": int(L), "max_abs_diff": float(dif.max()), "p99_abs_diff": float(np.percentile(dif, 99)),
                     "ms_per_call": round(s.asr.seg.ms / max(1, s.asr.seg.calls), 2)}
        log(name, res[name])
    save(f"served_v5_{a.tag}", res)


def ready_db(x: np.ndarray, T: int, v_list, t_list) -> np.ndarray:
    """Per frame v the served energy (Session._energy): samples [1280 v, min(1280 (v + 1), decision-ready sample))."""
    from audioforge.server.policies import frame_db
    out = frame_abs_db(x, T)
    for v, t in zip(v_list, t_list):
        b = min((v + 1) * 1280, int(round(float(t) * SR)))
        if v < T:
            out[v] = frame_db(x[v * 1280: b]) if b > v * 1280 else -100.0
    return out


def cmd_energy_ready(a):
    import eot_latency as E
    import eot_assistant as EA
    dump = E.load_dump()
    for s in E.sessions():
        d = dump.get(s["key"])
        if d is None:
            continue
        T = len(np.load(EVALF4 / f"{s['key']}.npz")["pu"])
        np.save(ENERGY / f"{s['key']}.npy", ready_db(E.read_audio(s), T, d["head"]["v"], d["head"]["t"]))
    ad = EA.load_dump()
    for c in EA.clips():
        d = ad[c["key"]]
        T = len(np.load(W / "asst" / f"asst_{c['key']}.npz")["pu"])
        np.save(ENERGY / f"asst_{c['key']}.npy", ready_db(EA.audio(c), T, d["head"]["v"], d["head"]["t"]))
    from audioforge.data import load_wav
    cv = json.loads((E.WORK / "clip_frames.json").read_text())
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), 16000).astype(np.float32)
    for i, (var, fn) in enumerate(E.CLIP_VARIANTS.items()):
        T = len(np.load(EVALF4 / f"clip_{i}.npz")["pu"])
        t = cv[var]["head"]["t"]
        np.save(ENERGY / f"clip_{i}.npy", ready_db(fn(x0), T, range(len(t)), t))
    log("energy at the decision-ready time: done")


def cmd_summary(a):
    """runs/turn_v5.json "candidates": per tag the val metrics, frame stats, head reach, smart-turn test accuracy and
    the scan4 operating points (both label sets where scanned)."""
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    fs, stt = res.get("frame_stats", {}), res.get("smartturn_test", {})
    out = {}
    for tag in a.tags:
        r = {}
        hf = W / tag / "history.json"
        if hf.exists():
            h = json.loads(hf.read_text())
            best = max(h["hist"], key=lambda x: x["auc"])
            r["params_M"] = round(h["params"] / 1e6, 2)
            r["args"] = {k: h["args"][k] for k in ("block", "pros", "no_text", "w_st_aux", "w_compl", "lr", "layers",
                                                   "dropout", "no_st32", "steps") if k in h["args"]}
            r["val_best"] = {k: best[k] for k in ("step", "auc", "auc_early", "auc_end_vs_cut", "acc_st", "bce")}
        if tag in fs:
            r["frame"] = {sc: {k: fs[tag][sc][k] for k in ("frame_auc_all", "frame_auc_quiet", "p50_in_turn_quiet",
                                                          "p50_pause", "p50_end_320ms")} for sc in ("two_party_user", "ami")}
            r["head_reach"] = {sc: {th: fs[tag]["head_reach"][sc][f"p>={th}"] for th in (0.5, 0.9, 0.95)}
                               for sc in ("two_party_user", "ami")}
        if tag in stt:
            r["smartturn_test"] = stt[tag]
        for lab in ("orig", "audible"):
            f = W / f"scan4_{tag}_{lab}.json"
            if not f.exists():
                continue
            rows = json.loads(f.read_text())["all"] or []
            g = GOAL if lab == "orig" else {"calls_fi": 19.3, "calls_miss": 6.4, "ami_fi": 10.5, "ami_miss": 33.5}
            c = lambda x: x["calls"]  # noqa: E731
            base = [x for x in rows if c(x)["eot_total_ms_p50"] is not None and c(x)["missed_pct"] <= g["calls_miss"]]
            amiok = [x for x in base if x["ami"]["false_interruption_pct"] <= g["ami_fi"] and x["ami"]["missed_pct"] <= g["ami_miss"]]
            key = lambda x: (c(x)["eot_total_ms_p50"], c(x)["false_interruption_pct"])  # noqa: E731
            pick = lambda L: None if not L else {k: L[0][k] for k in ("name", "calls", "ami", "asst", "clip_cut_in")}  # noqa
            r[f"scan_{lab}"] = {
                "n_rules": len(rows),
                "goal_no_cut": pick(sorted([x for x in amiok if c(x)["false_interruption_pct"] <= g["calls_fi"] and not x["clip_cut_in"]], key=key)),
                "goal_ignoring_clip": pick(sorted([x for x in amiok if c(x)["false_interruption_pct"] <= g["calls_fi"]], key=key)),
                "fi25_ami_goal_no_cut": pick(sorted([x for x in amiok if c(x)["false_interruption_pct"] <= 25 and not x["clip_cut_in"]], key=key)),
                "fi25_ami_goal_ignoring_clip": pick(sorted([x for x in amiok if c(x)["false_interruption_pct"] <= 25], key=key)),
                "fi25_no_cut_ami_free": pick(sorted([x for x in base if c(x)["false_interruption_pct"] <= 25 and not x["clip_cut_in"]], key=key)),
                "assistant_acc90_fastest": pick(sorted([x for x in rows if (x["asst"]["acc"] or 0) >= 90], key=lambda x: x["asst"]["p50"])),
                "assistant_best_acc": pick(sorted(rows, key=lambda x: (-(x["asst"]["acc"] or 0), x["asst"]["p50"] or 1e9))),
            }
        out[tag] = r
    res["candidates"] = {**res.get("candidates", {}), **out}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for tag, r in out.items():
        v = r.get("val_best", {})
        sc = r.get("scan_orig", {})
        f = lambda x: None if not x else (f"{x['calls']['eot_total_ms_p50']} {x['calls']['false_interruption_pct']}/{x['calls']['missed_pct']} "  # noqa
                                          f"AMI {x['ami']['false_interruption_pct']}/{x['ami']['missed_pct']} cut {len(x['clip_cut_in'])}")
        fa = lambda x: None if not x else f"{x['asst']['acc']}% {x['asst']['p50']}ms ff {x['asst']['ff_inc']}"  # noqa
        log(tag, "val auc", round(v.get("auc", 0), 3), "early", round(v.get("auc_early", 0), 3),
            "| calls qAUC", r.get("frame", {}).get("two_party_user", {}).get("frame_auc_quiet"),
            "AMI qAUC", r.get("frame", {}).get("ami", {}).get("frame_auc_quiet"),
            "| stest", r.get("smartturn_test", {}).get("0.5", {}).get("accuracy"),
            "| goal", f(sc.get("goal_ignoring_clip")), "| fi25", f(sc.get("fi25_ami_goal_ignoring_clip")),
            "| fi25 nocut", f(sc.get("fi25_no_cut_ami_free")), "| asst90", fa(sc.get("assistant_acc90_fastest")))


F5 = {"mode": "model", "gate": True, "quiet_db": None, "k": 1, "mvt": 0.6, "mp": 0.7, "fb": 8, "vad_thr": 0.4,
      "reask": True, "others": (12, 8)}  # the v5 fast operating point (research/TURN_V5.md)
A5 = {"mode": "model", "gate": True, "quiet_db": 6.0, "mqo": True, "k": 2, "mvt": 0.4, "mp": 0.9, "fb": 37,
      "vad_thr": 0.4, "reask": True, "others": (12, 8)}  # the v5 assistant operating point
BAL = {"mode": "head", "gate": True, "quiet_db": None, "k": 2, "th": 0.99, "fb": 8, "vad_thr": 0.4, "others": (12, 8)}


def cmd_boot(a):
    """Session bootstrap (2000 draws) of the calls / AMI p50 difference: tag's rule vs today's balanced rule."""
    import eot_latency as E
    if a.labels:
        os.environ["EOT_LABELS"] = a.labels
        E.LABELS = a.labels
    rule = json.loads(a.rule) if a.rule else F5
    rule["others"] = tuple(rule.get("others", (12, 8)))
    res = {}
    per = {}
    for tag, r in ((a.tag, rule), ("served", BAL)):
        dump, _, _ = load_sets(tag)
        sess = [s for s in E.sessions() if s["key"] in dump]
        comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
        gc = {}
        for ss in sess:
            d = dump[ss["key"]]
            db = np.load(ENERGY / f"{ss['key']}.npy")
            db = db[np.clip(np.asarray(d["head"]["v"]), 0, len(db) - 1)]
            t = [x + comp[ss["key"]] for x, _ in run_policy(d, db, r, gc.setdefault(ss["key"], {}))]
            per.setdefault(tag, {})[ss["key"]] = (ss["set"], E.score_session(t, ss, comp[ss["key"]]))
    rng = np.random.default_rng(0)
    for sc, sets in E.SCOPES.items():
        keys = [k for k, (st, _) in per[a.tag].items() if st in sets]
        diffs = []
        for _ in range(2000):
            ks = rng.choice(keys, len(keys))
            m = [np.median(np.concatenate([per[t][k][1]["lat"] for k in ks] or [[np.nan]])) for t in (a.tag, "served")]
            diffs.append(1000 * (m[0] - m[1]))
        res[sc] = {"p50_diff_ms": round(float(np.nanmedian(diffs))), "ci90": [round(float(np.nanpercentile(diffs, q))) for q in (5, 95)]}
        log(sc, res[sc])
    save(f"bootstrap_{a.tag}_{'audible' if a.labels else 'orig'}", {"rule": {k: v for k, v in rule.items()}, **res})


def cmd_build_v3(a):
    """runs/stage1_served_v3.afm = stage1_served_v2.afm + heads.turn_seg (the v5 classifier ``--tag``, weight 0: never
    trained by the Trainer), and assets/served_heads_v0.3.pt = its non-NVIDIA tensors (hub.export_heads); checks
    that hub.build_served rebuilds v3 bit-identically from the NVIDIA .nemo + the v0.3 heads."""
    import torch
    from audioforge import hub
    from audioforge.model import SpeechModel
    from audioforge.train import load_model, save_model
    import eot_latency as E
    v2 = load_model(str(E.SERVED_AFM), "cpu")
    ck = torch.load(W / a.tag / "model.pt", map_location="cpu", weights_only=False)
    cfg = json.loads(json.dumps(v2.cfg))
    cfg["heads"]["turn_seg"] = {"type": "turn_seg", "weight": 0.0, **ck["cfg"]}
    m = SpeechModel(cfg, v2.tokenizer)
    sd = v2.state_dict()
    sd.update({f"heads.turn_seg.{k}": v for k, v in ck["state_dict"].items()})
    m.load_state_dict(sd, strict=True)
    out = ROOT / "runs" / "stage1_served_v3.afm"
    save_model(m.eval(), out)
    back = load_model(str(out), "cpu")
    assert hub.state_hash(back.state_dict()) == hub.state_hash(m.state_dict())
    base = Path(a.base)
    heads = ROOT / "assets" / "served_heads_v0.3.pt"
    info = hub.export_heads(out, base, heads)
    tmp = W / "rebuilt_v3.afm"
    got = hub.build_served(base, heads, tmp)
    assert got == info["state_hash"], (got, info)
    log(f"v3 built: {info}; heads {heads.stat().st_size} bytes sha256 {hub.sha256_file(heads)}; rebuild hash ok")
    save("served_heads_v0.3", {"from_tag": a.tag, **info, "size": heads.stat().st_size,
                               "sha256": hub.sha256_file(heads), "afm": out.name})


PRESET_RULES = {"balanced": BAL, "steady": {"mode": "head", "gate": True, "quiet_db": None, "k": 6, "th": 0.99, "fb": 9,
                                           "vad_thr": 0.6, "others": (8, 8)},
                "fast": F5, "assistant": {**A5, "k": 3, "mvt": 0.4}}


def cmd_served_presets(a):
    """The real single-mode engine on runs/stage1_served_v3.afm, the bundled clip under the six deliveries, every
    --turn-preset: served turn_ends vs the offline policy twin (run_policy) on the session's own frames, the clip cut
    check, the v5 compute; writes tests/fixtures/two_party_call_16s_v5.json (the frames + the v5 p + energies)."""
    import torch
    from single_model import single_engine
    import audioforge.serve as S
    import eot_latency as E
    from audioforge.data import load_wav
    torch.set_num_threads(2)
    eng = single_engine(afm=ROOT / "runs" / "stage1_served_v3.afm")
    eng.warmup()
    x0 = load_wav(str(ROOT / "examples" / "audio" / "two_party_call_16s.wav"), SR).astype(np.float32)
    emb = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    res, fx = {}, {}
    for preset in a.presets:
        rule = dict(PRESET_RULES[preset])
        for var, fn in E.CLIP_VARIANTS.items():
            x = fn(x0)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
            s.arm_enrollment("enroll", 0, embedding=emb)
            rec = []
            orig = s.asr.run_turn_on_diar

            def rtod(avail, act_fn, flush=False, _o=orig, _s=s, rec=rec):
                out = _o(avail, act_fn, flush)
                tp = _s.asr.tsvad_p
                rec.extend((int(v), round(_s._asr_ready_t(v), 4), float(p), float(vad), float(tp[v][0]), float(tp[v][1]))
                           for v, p, vad in out)
                return out
            s.asr.run_turn_on_diar = rtod
            msgs = []
            for i in range(0, len(x), 320):
                msgs += s.process(x[i:i + 320])
            msgs += s.finish()
            te = [m for m in msgs if m["type"] == "turn_end"]
            h = {k: [r[j] for r in rec] for j, k in enumerate(("v", "t", "p", "vad", "pu", "po"))}
            db = ready_db(x, max(h["v"]) + 1, h["v"], h["t"])
            if s.asr.seg is not None:
                h5 = dict(h)
                h5["p"] = [s.asr.seg_prob(v) for v in h["v"]]
            else:
                h5 = h
            off = run_policy({"head": h5}, db[np.asarray(h["v"])], rule, {})
            st = s.stats_message() if hasattr(s, "stats_message") else {}
            r = {"served": [{k: m.get(k) for k in ("t", "path", "p", "silence_ms", "model_ms")} for m in te],
                 "offline": [round(t_, 4) for t_, _ in off],
                 "served_equals_offline": [m["t"] for m in te] == [round(t_, 3) for t_, _ in off],
                 "cuts_user_turn": any(s0 <= m["t"] < e1 - 0.08 for m in te),
                 "answers_end_at": next((m["t"] for m in te if m["t"] >= e1 - 0.08), None),
                 "seg_ms_per_call": round(s.asr.seg.ms / max(1, s.asr.seg.calls), 2) if s.asr.seg is not None else None,
                 "chunk_ms_p50": round(float(np.median(list(s.chunk_ms))), 1)}
            res.setdefault(preset, {})[var] = r
            log(preset, var, [m["t"] for m in te], r["offline"], "EQUAL" if r["served_equals_offline"] else "DIFF",
                "CUT" if r["cuts_user_turn"] else "no cut", r["seg_ms_per_call"], r["chunk_ms_p50"])
            if preset == "fast":
                fx[var] = {"t": h["t"], "p5": [round(float(q), 6) for q in h5["p"]], "vad": h["vad"], "pu": h["pu"],
                           "po": h["po"], "db": [round(float(q), 3) for q in db[np.asarray(h["v"])]]}
    if fx:
        (ROOT / "tests" / "fixtures" / "two_party_call_16s_v5.json").write_text(json.dumps(
            {"note": "served frames of the bundled clip under the six CLIP_VARIANTS deliveries with the turn head v5 p "
                     "(scripts/research/turn_v5.py served_presets); db = the frame energy at its decision-ready time",
             "variants": fx}))
    save("served_presets", res)


def cmd_asst_served(a):
    """Score the served dump of the 399 assistant clips under a v5 preset (eot_assistant.py dump with
    EOT_ASSISTANT_ENGINE='{"afm": "runs/stage1_served_v3.afm", "turn_preset": "assistant"}') with eot_assistant's
    scorer; compare with the offline twin (run_policy on the dump's frames with the offline v5 p)."""
    import eot_assistant as EA
    dd = W / a.dump
    dump = {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(dd.glob("*.json")))}
    sess = EA.sessions(dump)
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    served = EA.score_system(sess, {k: [m["t"] + comp[k] for m in d["turn_ends"]] for k, d in dump.items()}, comp)
    ap = json.loads((W / f"asst_p_{a.tag}.json").read_text())
    rule = dict(PRESET_RULES[a.preset])
    off, same = {}, 0
    for k, d in dump.items():
        h = dict(d["head"])
        v = np.asarray(h["v"])
        h["p"] = [float(ap[k][min(j, len(ap[k]) - 1)]) for j in v]
        db = ready_db(EA.audio(next(c for c in EA.clips() if c["key"] == k)), int(v.max()) + 1, h["v"], h["t"])[v]
        ev = run_policy({"head": h}, db, rule, {})
        off[k] = [t + comp[k] for t, _ in ev]
        same += [round(t, 3) for t, _ in ev] == [m["t"] for m in d["turn_ends"]]
    offline = EA.score_system(sess, off, comp)
    paths = {}
    for d in dump.values():
        for m in d["turn_ends"]:
            paths[m.get("policy")] = paths.get(m.get("policy"), 0) + 1
    res = {"served": served, "offline_twin": offline, "clips_served_equals_offline": f"{same}/{len(dump)}",
           "chunk_ms_p50_median": round(1000 * float(np.median(list(comp.values()))), 1)}
    save(f"assistant_served_{a.preset}", res)
    for nm, r in (("served", served), ("offline", offline)):
        log(nm, r["accuracy_pct"], r["complete"]["eot_total_ms_p50"], r["complete"]["eot_total_ms_p95"],
            r["incomplete"]["false_fire_pct"], r["complete"]["missed_pct"], r["complete"]["early_fire_pct"])
    log("served == offline on", res["clips_served_equals_offline"], "chunk p50", res["chunk_ms_p50_median"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blocks")
    b.add_argument("--which", default="all")
    b.add_argument("--budget", type=float, default=540)
    b.add_argument("--batch", type=int, default=16)
    b.add_argument("--device", default="mps")
    cp = sub.add_parser("cutplan")
    cp.add_argument("--per-clip", type=int, default=2)
    cu = sub.add_parser("extract")
    cu.add_argument("--set", required=True)
    cu.add_argument("--budget", type=float, default=540)
    cu.add_argument("--batch", type=int, default=16)
    cu.add_argument("--device", default="mps")
    st = sub.add_parser("smartturn")
    st.add_argument("--budget", type=float, default=540)
    st.add_argument("--p-pause", type=float, default=0.5)
    sub.add_parser("texts")
    lm = sub.add_parser("lm")
    lm.add_argument("--budget", type=float, default=540)
    lm.add_argument("--lm-batch", type=int, default=8)
    pr = sub.add_parser("prosody")
    pr.add_argument("--budget", type=float, default=540)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True)
    t.add_argument("--block", default="17")
    t.add_argument("--pros", action="store_true")
    t.add_argument("--no-text", action="store_true")
    t.add_argument("--max-clips", type=int, default=None)
    t.add_argument("--max-val", type=int, default=30000)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d", type=int, default=256)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--w-pause", type=float, default=1.5)
    t.add_argument("--w-cut", type=float, default=0.7)
    t.add_argument("--w-st", type=float, default=0.5)
    t.add_argument("--w-st-aux", type=float, default=0.5)
    t.add_argument("--w-compl", type=float, default=0.3)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--pu-drop", type=float, default=0.25)
    t.add_argument("--calibrate-now", action="store_true")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--no-st32", action="store_true")
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--budget", type=float, default=540)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="mps")
    cb = sub.add_parser("calib")
    cb.add_argument("--tag", required=True)
    cb.add_argument("--device", default="mps")
    ed = sub.add_parser("evaldump")
    ed.add_argument("--tag", required=True)
    ed.add_argument("--device", default="cpu")
    sub.add_parser("energy")
    asv = sub.add_parser("asst_served")
    asv.add_argument("--dump", default="asst_dump_assistant")
    asv.add_argument("--tag", default="c5")
    asv.add_argument("--preset", default="assistant")
    sp = sub.add_parser("served_presets")
    sp.add_argument("--presets", nargs="+", default=["balanced", "steady", "fast", "assistant"])
    b3 = sub.add_parser("build_v3")
    b3.add_argument("--tag", default="c5")
    b3.add_argument("--base", default="/Volumes/ExternalSSD/nvidia-audio-models/data/nemo/stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
    bo = sub.add_parser("boot")
    bo.add_argument("--tag", required=True)
    bo.add_argument("--rule", default=None)
    bo.add_argument("--labels", default=None)
    sm = sub.add_parser("summary")
    sm.add_argument("--tags", nargs="+")
    sub.add_parser("energy_ready")
    sv = sub.add_parser("served_v5")
    sv.add_argument("--tag", required=True)
    sv.add_argument("--keys", nargs="*", default=[])
    bl = sub.add_parser("blend")
    bl.add_argument("--tags", nargs="+")
    bl.add_argument("--out", required=True)
    s4 = sub.add_parser("scan4")
    s4.add_argument("--tag", required=True)
    s4.add_argument("--labels", default=None)
    s4.add_argument("--quick", action="store_true")
    s4.add_argument("--keep-all", action="store_true")
    s3 = sub.add_parser("scan3")
    s3.add_argument("--tag", required=True)
    s3.add_argument("--labels", default=None)
    s3.add_argument("--quick", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--tag", required=True)
    se = sub.add_parser("stest")
    se.add_argument("--tags", nargs="+")
    se.add_argument("--device", default="cpu")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
