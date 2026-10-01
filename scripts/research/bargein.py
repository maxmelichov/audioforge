"""research/BARGEIN.md: evaluation + training data for a barge-in / backchannel classifier, baselines, a first probe.

While the agent is speaking the user makes a sound. Classes: backchannel (the agent keeps talking), interruption (the
user takes the floor: the agent must stop), noise (cough / breath / other non-speech: keep talking). The "agent" is
the other party of a human-human conversation (AMI / ICSI meetings, otoSpeech two-channel calls). Label rules: see
``label_run`` and research/BARGEIN.md. CPU only; every stage runs under scripts/dev/gate.sh.

Stages (W = scratch/bargein):
  build       events of every local AMI / ICSI meeting and labelled oto conversation -> W/events.json, counts ->
              runs/bargein.json "data"
  otocache    16 kHz channel caches of the 16 held-out oto conversations (resumable, --budget)
  evalfeats   eval events: pass-1 blocks 4 / 8 (masked [70, 1] forward of the served 115M encoder, the turn_v5 cache
              recipe) over [onset - 8 s, onset + 2 s] of the user channel (oto) / the headset mix (AMI, ICSI) + the
              RNNT greedy words (oto) -> W/evalfeats/<id>.npz (resumable, --budget)
  trainextra  train events outside every cached turn_v5/blk window: the same blocks -> W/trainx/<id>.npy
  trainfeats  train events (turn_v5/blk cache, else W/trainx) + eval events -> W/{train,eval}_X.npy + _meta.json
  baselines   VAD-only, duration rule, min-words (+ backchannel-word filter) on the eval events -> "baselines"
  probe       tiny attention-pool head on block 4 + 8, trained on the train events, scored at 240 / 400 / 560 ms

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

W = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/bargein")
BLK5 = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v5/blk")
MAN4 = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v4/manifest.json")
E2E_CLIPS = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad/clips.json")
OUT = ROOT / "runs" / "bargein.json"
SR, FRAME = 16000, 0.08
DECISION_MS = (240, 400, 560)
CLASSES = ("backchannel", "interruption", "noise")

# ---- label rule constants (research/BARGEIN.md "Label rules")
BRIDGE = 0.3        # s: gaps <= this inside one party's speech are bridged (a VAD hangover): one "run"
PRE_SIL = 1.0       # s: the user is silent this long before the onset (a fresh onset)
AGENT_MIN = 1.0     # s: the agent has been speaking at least this long at the onset
BC_MAX = 1.0        # s: a backchannel run lasts at most this long
BC_RESUME = 1.0     # s: ... and the agent speaks again within this long after it (the agent held the floor)
INT_MIN = 1.5       # s: an interruption run lasts at least this long
INT_YIELD = 0.5     # s: ... and the agent stops at least this long before the user's run ends,
AGENT_BC_MAX = 1.0  # s: ... saying nothing but runs <= this long (its own backchannels) until the user's run ends
NOISE_MAX = 1.5     # s: a noise run (non-speech sounds only) lasts at most this long
THIRD_GUARD = 1.0   # s: meetings: no third party speaks in [onset - this, run end + this] (two-party condition)
WIN_PRE, WIN_POST = 8.0, 2.0  # eval feature window around the onset

BC_WORDS = {"mm", "hmm", "mhm", "mmhm", "mmhmm", "uh huh", "uhhuh", "huh", "yeah", "yea", "yes", "yep", "yup", "ok",
            "okay", "right", "alright", "all right", "sure", "oh", "ah", "wow", "ooh", "true", "exactly", "cool",
            "nice", "great", "good", "really", "i see", "indeed", "absolutely", "definitely", "fine", "mm hmm", "uh",
            "no", "mmm", "mmmm", "kay", "'kay"}
# 'uh' only counts inside 'uh huh'; 'no' only inside 'oh no'. Handled in is_bc().
LAUGH = {"laugh"}
NOISE_SOUNDS = {"cough", "sneeze", "sigh", "sighs", "yawn", "yawns", "sharp inhale", "sharp exhale", "loud exhale",
                "loud inhale", "sharp intake of air", "blows nose", "clicks tongue", "tongue clicks", "tongue clicking",
                "hiccough", "gasp", "snort", "whistling", "whistles", "humming", "hums"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def is_bc(text: str) -> bool:
    """All tokens of a (normalized) user run are backchannel tokens."""
    t = " " + " ".join(text.split()) + " "
    for multi in ("uh huh", "mm hmm", "all right", "i see", "oh no", "oh okay", "oh yeah", "oh right", "oh wow"):
        t = t.replace(" " + multi + " ", " X ")
    toks = t.split()
    return bool(toks) and all(w == "X" or (w in BC_WORDS and w not in ("uh", "no")) for w in toks)


# --------------------------------------------------------------------------- corpora -> per-party timelines
def runs_of(items, bridge=BRIDGE):
    """[(s, e, text, kind)] -> runs [[s, e, [items]]] with gaps <= bridge merged."""
    out = []
    for it in sorted(items, key=lambda x: (x[0], x[1])):
        if out and it[0] - out[-1][1] <= bridge:
            out[-1][1] = max(out[-1][1], it[1])
            out[-1][2].append(it)
        else:
            out.append([it[0], it[1], [it]])
    return out


def ami_sounds(ann: Path, meeting: str, names: dict) -> dict:
    """AMI timed <vocalsound> elements per speaker: [(s, e, type, 'sound')] (zero-length ones dropped)."""
    out = {}
    for p in sorted((ann / "words").glob(f"{meeting}.*.words.xml")):
        agent = p.name.split(".")[1]
        spk = names.get(agent) or f"{meeting}.{agent}"
        for el in ET.parse(p).getroot().iter("vocalsound"):
            s, e = el.get("starttime"), el.get("endtime")
            if not s or not e or float(e) - float(s) < 0.05:
                continue
            out.setdefault(spk, []).append((float(s), float(e), el.get("type", "other").strip().lower(), "sound"))
    return out


def meeting_timelines(corpus: str):
    """{meeting: {"split", "parties": {name: [(s, e, text, kind)]}, "zones": [(a, b)], "words_only": bool}}"""
    from audioforge.datasets import ami as A
    from audioforge.datasets import icsi as I
    out = {}
    if corpus == "ami":
        splits = A.DEFAULT_MEETINGS
        ann = A.DEFAULT_ROOT / "annotations"
        names_all = A.speaker_names(ann)
        for split, ms in splits.items():
            for m in ms:
                if not (A.DEFAULT_ROOT / "cache" / f"{m}.f32.npy").exists():
                    continue
                words = A.meeting_words(ann, m)
                snd = ami_sounds(ann, m, names_all.get(m, {}))
                parties = {s: [(a, b, t, "word") for a, b, t in w if b > a] for s, w in words.items()}
                for s, v in snd.items():
                    parties.setdefault(s, []).extend(v)
                out[m] = {"split": SPLIT_MAP[split], "parties": parties, "zones": []}
    else:
        root = I.DEFAULT_ROOT
        zones = json.loads((root / "annotations" / "untimed.json").read_text())
        for split, ms in I.DEFAULT_MEETINGS.items():
            for m in ms:
                if not (root / "cache" / f"{m}.f32.npy").exists():
                    continue
                words = A.meeting_words(root / "annotations", m)
                parties = {s: [(a, b, t, "word") for a, b, t in w if b > a] for s, w in words.items()}
                out[m] = {"split": SPLIT_MAP[split], "parties": parties,
                          "zones": [(a, b if b >= 0 else 1e9) for a, b in zones.get(m, [])]}
    return out


SPLIT_MAP = {"train": "train", "dev": "eval", "eval": "test"}  # corpus split -> ours (corpus dev = our eval set)


def eval_oto_ids() -> list[str]:
    return sorted({c["conversation"] for c in json.loads(E2E_CLIPS.read_text()) if c["set"] == "oto"})


def oto_timelines():
    from audioforge.datasets import dyadic as D
    ev = set(eval_oto_ids())
    out = {}
    for p in sorted((D.OTO_ROOT / "cache" / "labels").glob("*.json")):
        if "." in p.stem:
            continue
        d = json.loads(p.read_text())
        parties = {str(c): [(float(a), float(b), "", "vad") for a, b, _ in d["words"][str(c)]] for c in (0, 1)}
        out[p.stem] = {"split": "eval" if p.stem in ev else "train", "parties": parties,
                       "zones": [tuple(z) for z in d.get("zones", [])], "duration": d["duration"]}
    return out


# --------------------------------------------------------------------------- label rules
def label_run(corpus, u_run, u_prev_end, agent_runs, a_idx, third_busy, has_words=True):
    """Label one user run given the agent's runs. Returns (label, subtype, info) with label in CLASSES or
    'ambiguous' / None (None = not a barge-in candidate)."""
    s, e, items = u_run
    if s - u_prev_end < PRE_SIL:
        return None, "no_pre_silence", {}
    ar = agent_runs[a_idx]
    if not (ar[0] <= s - AGENT_MIN and ar[1] > s):
        return None, "agent_not_speaking", {}
    if third_busy:
        return None, "third_party", {}
    dur = e - s
    words = [it for it in items if it[3] in ("word", "vad")]
    sounds = [it for it in items if it[3] == "sound"]
    text = " ".join(it[2] for it in items if it[3] == "word").strip()
    t_a = ar[1]  # the agent's floor run ends here
    later = [r for r in agent_runs[a_idx + 1:] if r[0] < e + BC_RESUME + 5]
    resumed = t_a >= e or any(r[0] <= e + BC_RESUME for r in later)
    agent_talk_after_ta = sum(min(r[1], e) - r[0] for r in later if r[0] < e)
    agent_long_after_ta = any(r[0] < e and (min(r[1], e) - r[0]) > AGENT_BC_MAX for r in later)
    yielded = t_a < e - INT_YIELD and not agent_long_after_ta
    info = {"dur": round(dur, 3), "text": text, "sounds": [it[2] for it in sounds], "agent_end": round(t_a, 3),
            "agent_resumed": bool(resumed), "agent_yielded": bool(yielded),
            "agent_talk_after": round(agent_talk_after_ta, 3)}
    if corpus in ("ami", "icsi") and not words:
        kinds = {it[2] for it in sounds}
        if kinds and kinds <= LAUGH:
            return ("backchannel" if dur <= BC_MAX and resumed else "ambiguous"), "laugh", info
        if kinds and kinds <= NOISE_SOUNDS and dur <= NOISE_MAX:
            return "noise", "sound:" + ",".join(sorted(kinds)), info
        return "ambiguous", "sound_other", info
    lexical_bc = is_bc(text) if corpus in ("ami", "icsi") else True
    if dur <= BC_MAX and resumed and lexical_bc and not (sounds and not all(x[2] in LAUGH for x in sounds)):
        return "backchannel", "bc_lex" if corpus != "oto" else "bc_vad", info
    if dur >= INT_MIN and yielded:
        return "interruption", "yield", info
    if dur <= BC_MAX and not lexical_bc:
        return "ambiguous", "short_non_bc_words", info
    if dur <= BC_MAX and not resumed:
        return "ambiguous", "short_agent_stops", info
    if dur >= INT_MIN:
        return "ambiguous", "long_agent_keeps_talking", info
    return "ambiguous", "mid_duration", info


def events_of(corpus, mid, tl):
    parties, zones = tl["parties"], tl["zones"]
    wruns = {p: runs_of([it for it in v if it[3] in ("word", "vad")]) for p, v in parties.items()}
    aruns = {p: runs_of(v) for p, v in parties.items()}  # words + sounds (the user side)
    out, rej = [], {}
    for u, runs in aruns.items():
        prev_end = -1e9
        for r in runs:
            s, e = r[0], r[1]
            pe, prev_end = prev_end, e
            if any(a < e + 0.5 and b > s - PRE_SIL for a, b in zones):
                rej["zone"] = rej.get("zone", 0) + 1
                continue
            # the agent: the (single) other party speaking at the onset
            cands = []
            for a, ar in wruns.items():
                if a == u:
                    continue
                for j, x in enumerate(ar):
                    if x[0] <= s < x[1]:
                        cands.append((a, j))
                        break
            if not cands:
                rej["agent_not_speaking"] = rej.get("agent_not_speaking", 0) + 1
                continue
            if len(cands) > 1:
                rej["third_party"] = rej.get("third_party", 0) + 1
                continue
            a, j = cands[0]
            third = any(x[0] < e + THIRD_GUARD and x[1] > s - THIRD_GUARD
                        for p, ar in wruns.items() if p not in (a, u) for x in ar)
            lab, sub, info = label_run(corpus, r, pe, wruns[a], j, third)
            if lab is None:
                rej[sub] = rej.get(sub, 0) + 1
                continue
            ev = {"id": f"{corpus}_{mid}_{u.replace(' ', '')}_{int(round(s * 1000))}", "corpus": corpus,
                  "meeting": mid, "split": tl["split"], "user": u, "agent": a, "onset": round(s, 3),
                  "end": round(e, 3), "label": lab, "subtype": sub, **info}
            if corpus == "oto":
                ev["ch"] = int(u)
            else:
                ev["words"] = [[round(it[0], 3), round(it[1], 3), it[2]] for it in r[2] if it[3] == "word"]
            out.append(ev)
    return out, rej


def cmd_build(a):
    W.mkdir(parents=True, exist_ok=True)
    evs, rej_all, meets = [], {}, {}
    for corpus in ("ami", "icsi", "oto"):
        tls = oto_timelines() if corpus == "oto" else meeting_timelines(corpus)
        for mid, tl in tls.items():
            e, rej = events_of(corpus, mid, tl)
            evs += e
            meets.setdefault(corpus, {}).setdefault(tl["split"], []).append(mid)
            for k, v in rej.items():
                rej_all.setdefault(corpus, {})[k] = rej_all.setdefault(corpus, {}).get(k, 0) + v
        log(f"{corpus}: {len(tls)} conversations")
    (W / "events.json").write_text(json.dumps(evs))
    counts = {}
    for ev in evs:
        c = counts.setdefault(ev["corpus"], {}).setdefault(ev["split"], {})
        c[ev["label"]] = c.get(ev["label"], 0) + 1
        sub = counts[ev["corpus"]].setdefault("subtypes_" + ev["split"], {})
        sub[ev["label"] + "/" + ev["subtype"]] = sub.get(ev["label"] + "/" + ev["subtype"], 0) + 1
    bc_text = {}
    for ev in evs:
        if ev["corpus"] != "oto" and ev["label"] in ("backchannel", "ambiguous") and ev["dur"] <= BC_MAX:
            k = (ev["label"][:3], ev["text"] or "<" + ",".join(ev["sounds"]) + ">")
            bc_text[k] = bc_text.get(k, 0) + 1
    top = sorted(bc_text.items(), key=lambda kv: -kv[1])[:40]
    data = {"events_file": str(W / "events.json"), "n_events": len(evs), "counts": counts,
            "rejected_candidates": rej_all,
            "conversations": {c: {s: len(v) for s, v in d.items()} for c, d in meets.items()},
            "eval_ids": {c: sorted(d.get("eval", [])) for c, d in meets.items()},
            "test_reserved_ids": {c: sorted(d.get("test", [])) for c, d in meets.items()},
            "top_short_texts": [[k[0], k[1], v] for k, v in top],
            "rules": {k: globals()[k] for k in ("BRIDGE", "PRE_SIL", "AGENT_MIN", "BC_MAX", "BC_RESUME", "INT_MIN",
                                                "INT_YIELD", "AGENT_BC_MAX", "NOISE_MAX", "THIRD_GUARD")}}
    save("data", data)
    print(json.dumps({"counts": {c: {s: v for s, v in d.items() if not s.startswith("subtypes")}
                                 for c, d in counts.items()}, "conversations": data["conversations"]}, indent=1))
    print(json.dumps(data["top_short_texts"]))
    print(json.dumps(rej_all))


# --------------------------------------------------------------------------- features
def load_events(split=None, labels=CLASSES):
    evs = json.loads((W / "events.json").read_text())
    return [e for e in evs if (split is None or e["split"] == split) and (labels is None or e["label"] in labels)]


class Audio:
    """16 kHz source of an event: the user's own channel (oto) or the headset mix (AMI / ICSI)."""

    def __init__(self):
        self.cache = {}

    def src(self, ev):
        k = (ev["corpus"], ev["meeting"], ev.get("ch"))
        if k not in self.cache:
            if ev["corpus"] == "oto":
                from audioforge.datasets import dyadic as D
                ds = D.Dyadic([ev["meeting"]], "oto", verbose=False)
                self.cache[k] = ds.channels16k(ev["meeting"])[ev["ch"]]
            else:
                root = ROOT / "data" / ev["corpus"] / "cache"
                self.cache[k] = np.load(root / f"{ev['meeting']}.f32.npy", mmap_mode="r")
        return self.cache[k]

    def window(self, ev, pre=WIN_PRE, post=WIN_POST):
        x = self.src(ev)
        t0 = max(0.0, ev["onset"] - pre)
        a, b = int(round(t0 * SR)), int(round((ev["onset"] + post) * SR))
        return np.asarray(x[a:b], np.float32), t0


def cmd_evalfeats(a):
    """Blocks 4 / 8 (+ RNNT words on oto) of every eval event, clean and ambiguous."""
    import copy
    import torch
    import turn_v4 as V4
    from audioforge.heads.turn import greedy_decode_frames
    from audioforge.train import load_model
    from prepare_dyadic_asr import words_of
    torch.set_num_threads(2)
    out = W / "evalfeats"
    out.mkdir(parents=True, exist_ok=True)
    evs = load_events("eval", labels=None)
    todo = [e for e in evs if not (out / f"{e['id']}.npz").exists()]
    log(f"evalfeats: {len(todo)} of {len(evs)} to do")
    if not todo:
        return
    import eot_latency as E
    m = load_model(str(E.SERVED_AFM), "cpu").eval()
    asr = copy.deepcopy(m.heads["rnnt"]).eval()
    au, t_start, done = Audio(), time.time(), 0
    todo.sort(key=lambda e: (e["corpus"], e["meeting"]))
    B = 8
    for i in range(0, len(todo), B):
        if time.time() - t_start > a.budget:
            break
        cs = todo[i: i + B]
        ws = [au.window(e) for e in cs]
        with torch.no_grad():
            x, xl = m._pad([w for w, _ in ws])
            enc, elen, hid = m.encode(x, xl, V4.ATT, return_hidden=True)
            for j, (e, (w, t0)) in enumerate(zip(cs, ws)):
                L = int(elen[j])
                words = []
                if e["corpus"] == "oto":
                    hyp, fr = greedy_decode_frames(asr, enc[j, :L].float())
                    words = words_of(m.tokenizer, hyp, [t0 + (f + 1) * FRAME for f in fr])
                tmp = out / f"{e['id']}.tmp.npz"
                np.savez(tmp, b4=hid[3][j, :L].half().numpy(), b8=hid[7][j, :L].half().numpy(), t0=np.float64(t0),
                         words=np.array(json.dumps(words)))
                tmp.rename(out / f"{e['id']}.npz")
        done += len(cs)
    log(f"evalfeats: {done} in {time.time() - t_start:.0f} s; {len(todo) - done} left")


def cmd_trainextra(a):
    """Train events outside every cached turn_v5/blk window: the same blocks (masked [70, 1] forward of the served
    encoder) over [onset - 8 s, onset + 0.8 s], only the onset frames kept -> W/trainx/<id>.npy (resumable)."""
    import torch
    import turn_v4 as V4
    from audioforge.train import load_model
    import eot_latency as E
    torch.set_num_threads(2)
    out = W / "trainx"
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(MAN4.read_text())
    cov = {}
    for c in man:
        if c["src"] in ("ami", "icsi", "oto"):
            cov.setdefault((c["src"], c["meeting"], c.get("ch")), []).append((c["a"], c["b"]))
    todo = [e for e in load_events("train") if not (out / f"{e['id']}.npy").exists()
            and not any(p + 3.0 <= e["onset"] and e["onset"] + 0.7 <= q
                        for p, q in cov.get((e["corpus"], e["meeting"], e.get("ch")), []))]
    log(f"trainextra: {len(todo)} to do")
    if not todo:
        return
    m = load_model(str(E.SERVED_AFM), "cpu").eval()
    au, t_start, done, B = Audio(), time.time(), 0, 8
    todo.sort(key=lambda e: (e["corpus"], e["meeting"]))
    for i in range(0, len(todo), B):
        if time.time() - t_start > a.budget:
            break
        cs = todo[i: i + B]
        ws = [au.window(e, WIN_PRE, 0.8) for e in cs]
        with torch.no_grad():
            x, xl = m._pad([w for w, _ in ws])
            _, elen, hid = m.encode(x, xl, V4.ATT, return_hidden=True)
        for j, (e, (_, t0)) in enumerate(zip(cs, ws)):
            L = int(elen[j])
            f = onset_frames(hid[3][j, :L].half().numpy(), hid[7][j, :L].half().numpy(), t0, e["onset"])
            if f is not None:
                np.save(out / f"{e['id']}.tmp.npy", f)
                (out / f"{e['id']}.tmp.npy").rename(out / f"{e['id']}.npy")
        done += len(cs)
    log(f"trainextra: {done} in {time.time() - t_start:.0f} s; {len(todo) - done} left")


PRE_FR, MAX_FR = 4, 4 + int(round(max(DECISION_MS) / 80))  # frames before the onset, frames kept (at 560 ms)


def onset_frames(b4, b8, t0, onset):
    """(MAX_FR, 1024) float16 frames [onset - 4 frames, onset + 560 ms) of a window starting at t0 (None if short)."""
    k0 = int(np.floor((onset - t0) / FRAME)) - PRE_FR
    if k0 < 0 or k0 + MAX_FR > len(b4):
        return None
    return np.concatenate([b4[k0: k0 + MAX_FR], b8[k0: k0 + MAX_FR]], 1)


def cmd_trainfeats(a):
    """Train events inside a cached turn_v5/blk window (>= 3 s after its start, same source: the user's channel for
    oto, the mix for AMI / ICSI) -> onset-aligned frames. Eval events -> the same layout from W/evalfeats."""
    man = json.loads(MAN4.read_text())
    by = {}
    for c in man:
        if c["src"] in ("ami", "icsi", "oto"):
            by.setdefault((c["src"], c["meeting"], c.get("ch")), []).append(c)
    X, meta, miss = [], [], 0
    evs = load_events("train")
    t_start = time.time()
    for e in evs:
        cl = by.get((e["corpus"], e["meeting"], e.get("ch")), [])
        hit = [c for c in cl if c["a"] + 3.0 <= e["onset"] and e["onset"] + 0.7 <= c["b"]]
        f, src = None, None
        if hit:
            c = max(hit, key=lambda c: e["onset"] - c["a"])
            z = np.load(BLK5 / f"{c['id']}.npz")
            f, src = onset_frames(z["b4"], z["b8"], c["a"], e["onset"]), "turn_v5/blk:" + c["id"]
        if f is None and (W / "trainx" / f"{e['id']}.npy").exists():
            f, src = np.load(W / "trainx" / f"{e['id']}.npy"), "bargein/trainx"
        if f is None:
            miss += 1
            continue
        X.append(f)
        meta.append({"id": e["id"], "corpus": e["corpus"], "meeting": e["meeting"], "label": e["label"],
                     "src": src})
    log(f"train: {len(X)} events with cached frames, {miss} outside every cached window ({time.time() - t_start:.0f} s)")
    np.save(W / "train_X.npy", np.stack(X))
    (W / "train_meta.json").write_text(json.dumps(meta))
    X, meta = [], []
    for e in load_events("eval"):
        z = np.load(W / "evalfeats" / f"{e['id']}.npz")
        f = onset_frames(z["b4"], z["b8"], float(z["t0"]), e["onset"])
        assert f is not None, e["id"]
        X.append(f)
        meta.append({"id": e["id"], "corpus": e["corpus"], "meeting": e["meeting"], "label": e["label"]})
    np.save(W / "eval_X.npy", np.stack(X))
    (W / "eval_meta.json").write_text(json.dumps(meta))
    cnt = {}
    for mm in json.loads((W / "train_meta.json").read_text()):
        cnt.setdefault(mm["corpus"], {})[mm["label"]] = cnt.setdefault(mm["corpus"], {}).get(mm["label"], 0) + 1
    nsrc = sum(mm["src"].startswith("turn_v5") for mm in json.loads((W / "train_meta.json").read_text()))
    save("train_coverage", {"with_frames": cnt, "from_turn_v5_cache": nsrc, "missing": miss,
                            "eval_events_with_frames": len(X)})
    log(f"train coverage {cnt}; eval {len(X)}")


# --------------------------------------------------------------------------- metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(p, 4), "r": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def score(labels, preds, lat_ms=None):
    """Per-class P / R / F1 (+ macro), the stop decision's view (interruption vs the rest): false-stop rate on
    backchannels + noise, share of VAD barge-ins rejected (LiveKit's 51 %), and the median decision latency."""
    labels, preds = list(labels), list(preds)
    out = {"n": len(labels), "per_class": {}}
    for c in CLASSES:
        tp = sum(1 for y, q in zip(labels, preds) if y == c and q == c)
        fp = sum(1 for y, q in zip(labels, preds) if y != c and q == c)
        fn = sum(1 for y, q in zip(labels, preds) if y == c and q != c)
        out["per_class"][c] = {**prf(tp, fp, fn), "support": sum(1 for y in labels if y == c)}
    present = [c for c in CLASSES if out["per_class"][c]["support"]]
    out["macro_f1"] = round(float(np.mean([out["per_class"][c]["f1"] for c in present])), 4)
    neg = [q for y, q in zip(labels, preds) if y != "interruption"]
    out["false_stop_rate"] = round(sum(q == "interruption" for q in neg) / len(neg), 4) if neg else None
    out["vad_bargeins_rejected"] = round(sum(q != "interruption" for q in preds) / len(preds), 4)
    if lat_ms is not None:
        tl = [l for y, q, l in zip(labels, preds, lat_ms) if y == "interruption" and q == "interruption"]
        out["latency_ms_p50_true_stops"] = float(np.median(tl)) if tl else None
    return out


def by_corpus(evs, preds, lat=None):
    res = {"all": score([e["label"] for e in evs], preds, lat)}
    for c in ("ami", "icsi", "oto"):
        idx = [i for i, e in enumerate(evs) if e["corpus"] == c]
        if idx:
            res[c] = score([evs[i]["label"] for i in idx], [preds[i] for i in idx],
                           [lat[i] for i in idx] if lat is not None else None)
    return res


# --------------------------------------------------------------------------- baselines
def eval_words(ev):
    """(time, word) the user produced after the onset: our RNNT greedy on the user channel (oto, measured) or the
    reference words stamped at their end (AMI / ICSI: an oracle, no ASR delay)."""
    if ev["corpus"] == "oto":
        z = np.load(W / "evalfeats" / f"{ev['id']}.npz")
        return [(t, w) for t, w in json.loads(str(z["words"])) if t > ev["onset"]]
    return [(w[1], w[2]) for w in ev.get("words", []) if w[1] > ev["onset"]]


def cmd_baselines(a):
    evs = load_events("eval")
    res = {"n_eval": len(evs), "systems": {}}
    S = res["systems"]
    S["vad_only"] = {"rule": "every user onset while the agent speaks stops the agent (decision at the VAD onset)",
                     **by_corpus(evs, ["interruption"] * len(evs), [0.0] * len(evs))}
    for T in (0, 240, 400, 500, 560, 800, 1000, 1200, 1500):
        pr = ["interruption" if e["dur"] * 1000 > T else "backchannel" for e in evs]
        S[f"duration_{T}ms"] = {"rule": f"stop if the user's VAD run (gaps <= {BRIDGE} s bridged) lasts > {T} ms",
                                **by_corpus(evs, pr, [float(T)] * len(evs))}
    words = {e["id"]: eval_words(e) for e in evs}
    for N in (1, 2, 3):
        for H in (560, 1000, 1500):
            for filt in (False, True):
                pr, lat = [], []
                for e in evs:
                    ws = [(t, w) for t, w in words[e["id"]] if t <= e["onset"] + H / 1000
                          and (not filt or not is_bc(w))]
                    ok = len(ws) >= N
                    pr.append("interruption" if ok else "backchannel")
                    lat.append((ws[N - 1][0] - e["onset"]) * 1000 if ok else None)
                name = f"min_words_{N}{'_nonbc' if filt else ''}_{H}ms"
                S[name] = {"rule": f"stop once >= {N} {'non-backchannel ' if filt else ''}words are transcribed "
                                   f"within {H} ms of the onset (oto: our streaming RNNT on the user channel; AMI / ICSI:"
                                   f" reference words at their end time, no ASR delay = an upper bound)",
                           **by_corpus(evs, pr, lat)}
    save("baselines", res)
    rows = ["vad_only"] + [k for k in S if k.startswith("duration")] + [k for k in S if k.startswith("min_words")]
    print(f"{'system':32s} {'int P':>6s} {'int R':>6s} {'int F1':>6s} {'bc F1':>6s} {'mF1':>6s} {'fstop':>6s} {'rej':>6s} {'lat':>6s}")
    for k in rows:
        r = S[k]["all"]
        pc = r["per_class"]
        print(f"{k:32s} {pc['interruption']['p']:6.3f} {pc['interruption']['r']:6.3f} {pc['interruption']['f1']:6.3f} "
              f"{pc['backchannel']['f1']:6.3f} {r['macro_f1']:6.3f} {r['false_stop_rate']:6.3f} "
              f"{r['vad_bargeins_rejected']:6.3f} {r.get('latency_ms_p50_true_stops') or 0:6.0f}")


# --------------------------------------------------------------------------- probe head
def n_frames(d_ms: int) -> int:
    """Frames the head may read at decision time d: the encoder's [70, 1] context gives each 80 ms frame one frame of
    look-ahead, so the last usable frame ends 80 ms before onset + d (no audio after onset + d is seen)."""
    return PRE_FR + d_ms // 80 - 1


def make_head(din, hid=32, n_cls=3):
    import torch
    import torch.nn as nn

    class Head(nn.Module):
        """LayerNorm -> Linear(din, 32) + frame position -> GELU -> attention pool ++ mean pool -> Linear(64, 3)."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(din)
            self.proj = nn.Linear(din, hid)
            self.pos = nn.Parameter(torch.zeros(MAX_FR, hid))
            self.att = nn.Linear(hid, 1)
            self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * hid, n_cls))

        def forward(self, x, n):  # x (B, MAX_FR, din), n (B,) frames usable
            h = torch.nn.functional.gelu(self.proj(self.norm(x)) + self.pos)
            mask = torch.arange(x.shape[1])[None] < n[:, None]
            w = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
            att = (w[..., None] * h).sum(1)
            mean = (h * mask[..., None]).sum(1) / n[:, None]
            return self.out(torch.cat([att, mean], -1))

    return Head()


def cmd_probe(a):
    import torch
    torch.set_num_threads(2)
    Xtr = np.load(W / "train_X.npy")
    mtr = json.loads((W / "train_meta.json").read_text())
    Xev = np.load(W / "eval_X.npy")
    mev = json.loads((W / "eval_meta.json").read_text())
    allev = {e["id"]: e for e in load_events(None)}
    ev_list = [allev[m["id"]] for m in mev]
    ytr = np.array([CLASSES.index(m["label"]) for m in mtr])
    convs = sorted({m["meeting"] for m in mtr})
    val_conv = {c for i, c in enumerate(convs) if i % 10 == 0}  # every 10th train conversation: val (thresholds)
    va = np.array([m["meeting"] in val_conv for m in mtr])
    res = {"train_events": int((~va).sum()), "val_events": int(va.sum()), "val_conversations": len(val_conv),
           "decision_ms": list(DECISION_MS), "variants": {}}
    cnt = np.bincount(ytr[~va], minlength=3).astype(float)
    wcls = torch.tensor(np.minimum(cnt.sum() / (3 * np.maximum(cnt, 1)), 20.0), dtype=torch.float32)
    dsel = {"b4": slice(0, 512), "b8": slice(512, 1024), "b4+b8": slice(0, 1024)}
    for var in a.variants.split(","):
        sl = dsel[var]
        per_seed = []
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            head = make_head(sl.stop - sl.start)
            opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=5e-2)
            Xt = torch.from_numpy(Xtr[~va][:, :, sl].astype(np.float32))
            yt = torch.from_numpy(ytr[~va])
            steps = a.epochs * (len(yt) // 128 + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
            head.train()
            for _ in range(steps):
                idx = torch.from_numpy(rng.integers(0, len(yt), 128))
                n = torch.from_numpy(rng.choice([n_frames(d) for d in (160, 240, 320, 400, 480, 560)], 128))
                x = Xt[idx] * (torch.rand(len(idx), MAX_FR, 1) > 0.1)  # frame dropout
                loss = torch.nn.functional.cross_entropy(head(x, n), yt[idx], weight=wcls)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            head.eval()

            def probs(X, d):
                with torch.no_grad():
                    Xx = torch.from_numpy(X[:, :, sl].astype(np.float32))
                    return head(Xx, torch.full((len(Xx),), n_frames(d))).softmax(-1).numpy()
            sr = {}
            for d in DECISION_MS:
                pv, pe = probs(Xtr[va], d), probs(Xev, d)
                yv = ytr[va]
                # stop threshold on P(interruption) for >= 95 % recall on the VAL conversations
                pos = np.sort(pv[yv == 1, 1])
                th = float(pos[int(np.floor(0.05 * len(pos)))]) if len(pos) else 0.5
                am = [CLASSES[k] for k in pe.argmax(1)]
                stop = ["interruption" if p >= th else ("noise" if q[2] > q[0] else "backchannel")
                        for p, q in zip(pe[:, 1], pe)]
                # oracle: the precision at 100 % eval recall (LiveKit's operating point; threshold chosen on eval)
                yint = np.array([e["label"] == "interruption" for e in ev_list])
                th100 = float(pe[yint, 1].min())
                or100 = ["interruption" if p >= th100 else "backchannel" for p in pe[:, 1]]
                # fused with the duration rule: stop iff the user's VAD run is still on at d AND P(int) >= th_g
                # (th_g for >= 95 % val recall among the val events that pass the gate)
                dv = np.array([allev[m["id"]]["dur"] for m, v in zip(mtr, va) if v])
                gv = dv * 1000 > d
                posg = np.sort(pv[(yv == 1) & gv, 1])
                thg = float(posg[int(np.floor(0.05 * len(posg)))]) if len(posg) else 0.5
                fused = ["interruption" if (e["dur"] * 1000 > d and p >= thg) else "backchannel"
                         for e, p in zip(ev_list, pe[:, 1])]
                from sklearn.metrics import roc_auc_score
                sr[str(d)] = {"auc_int_val": round(float(roc_auc_score(yv == 1, pv[:, 1])), 4),
                              "auc_int_eval": round(float(roc_auc_score(yint, pe[:, 1])), 4),
                              "fused_duration_gate": {"threshold": round(thg, 4),
                                                      **by_corpus(ev_list, fused, [float(d)] * len(fused))},
                              "argmax": by_corpus(ev_list, am, [float(d)] * len(am)),
                              "stop_at_val_recall95": {"threshold": round(th, 4),
                                                       **by_corpus(ev_list, stop, [float(d)] * len(stop))},
                              "oracle_recall100": score([e["label"] for e in ev_list], or100)}
            per_seed.append(sr)
            log(var, seed, {d: (sr[str(d)]["argmax"]["all"]["macro_f1"],
                                sr[str(d)]["stop_at_val_recall95"]["all"]["per_class"]["interruption"]["p"],
                                sr[str(d)]["stop_at_val_recall95"]["all"]["per_class"]["interruption"]["r"])
                            for d in DECISION_MS})
            if var == "b4+b8" and seed == 0:
                torch.save(head.state_dict(), W / "probe_b4b8_seed0.pt")
        summ = {}
        for d in DECISION_MS:
            def ms(get):
                v = [get(sr[str(d)]) for sr in per_seed]
                return [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
            summ[str(d)] = {
                "auc_int_val": ms(lambda r: r["auc_int_val"]),
                "auc_int_eval": ms(lambda r: r["auc_int_eval"]),
                "fused_int_p": ms(lambda r: r["fused_duration_gate"]["all"]["per_class"]["interruption"]["p"]),
                "fused_int_r": ms(lambda r: r["fused_duration_gate"]["all"]["per_class"]["interruption"]["r"]),
                "fused_int_f1": ms(lambda r: r["fused_duration_gate"]["all"]["per_class"]["interruption"]["f1"]),
                "fused_false_stop": ms(lambda r: r["fused_duration_gate"]["all"]["false_stop_rate"]),
                "fused_rejected": ms(lambda r: r["fused_duration_gate"]["all"]["vad_bargeins_rejected"]),
                "argmax_macro_f1": ms(lambda r: r["argmax"]["all"]["macro_f1"]),
                "argmax_f1": {c: ms(lambda r, c=c: r["argmax"]["all"]["per_class"][c]["f1"]) for c in CLASSES},
                "argmax_p": {c: ms(lambda r, c=c: r["argmax"]["all"]["per_class"][c]["p"]) for c in CLASSES},
                "argmax_r": {c: ms(lambda r, c=c: r["argmax"]["all"]["per_class"][c]["r"]) for c in CLASSES},
                "stop95_int_p": ms(lambda r: r["stop_at_val_recall95"]["all"]["per_class"]["interruption"]["p"]),
                "stop95_int_r": ms(lambda r: r["stop_at_val_recall95"]["all"]["per_class"]["interruption"]["r"]),
                "stop95_false_stop": ms(lambda r: r["stop_at_val_recall95"]["all"]["false_stop_rate"]),
                "stop95_rejected": ms(lambda r: r["stop_at_val_recall95"]["all"]["vad_bargeins_rejected"]),
                "oracle100_int_p": ms(lambda r: r["oracle_recall100"]["per_class"]["interruption"]["p"]),
                "oracle100_rejected": ms(lambda r: r["oracle_recall100"]["vad_bargeins_rejected"]),
                "stop95_int_p_by_corpus": {c: ms(lambda r, c=c: r["stop_at_val_recall95"][c]["per_class"]["interruption"]["p"])
                                           for c in ("ami", "icsi", "oto")},
                "stop95_int_r_by_corpus": {c: ms(lambda r, c=c: r["stop_at_val_recall95"][c]["per_class"]["interruption"]["r"])
                                           for c in ("ami", "icsi", "oto")},
            }
        res["variants"][var] = {"summary_mean_std": summ, "seed0_detail": per_seed[0]}
        log(var, json.dumps(summ))
    n_par = sum(p.numel() for p in make_head(1024).parameters())
    res["head"] = {"params_b4b8": int(n_par), "epochs": a.epochs, "seeds": a.seeds,
                   "train_lengths_ms": [160, 240, 320, 400, 480, 560], "class_weights": wcls.tolist()}
    save("probe", res)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage")
    p.add_argument("--budget", type=float, default=520.0)
    p.add_argument("--variants", default="b4+b8,b4,b8")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    a = p.parse_args()
    globals()["cmd_" + a.stage](a)


if __name__ == "__main__":
    main()
