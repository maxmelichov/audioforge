"""research/TURN_DATA.md: real per-speaker conversational channels for the 0.6B turn heads -> runs/turn_data.json.

The blocker of CORE_0P6B_TURN.md / FIXALL.md step 2: no training or held-out audio looked like the TurnBench user
channels (quiet speech, a real noise floor, weak crosstalk), so the better "Q" heads could not be selected safely.
Source used here: the AMI Meeting Corpus individual headset channels (IHM, CC BY 4.0): every participant's own
close-talk microphone, recorded in a real room, with the room's noise floor and the other participants' crosstalk.
One participant's headset = "the user channel"; turns = ami.speaker_turns on the manual word timings of all
participants (the floor rule every other corpus here uses; no forced alignment needed).

Splits: AMI dev meetings (pyannote list) = the new held-out scope `ihm`; AMI train meetings = training windows.
The AMI test meetings (the AMI test turn rows) are never read.

Stages (each one process, < 10 min, resumable; run through scripts/dev/gate.sh, one torch/MPS job at a time):
  man   --split dev|train        user-channel windows (40 s, around user turn ends) -> TD/man_<split>.json
  stats --split dev              channel statistics (speech level, noise floor, crosstalk) of the windows next to the
                                 oto held-out windows and the evaluation call clips (audio statistics only)
  feats --split S                the 0.6B served inputs + blocks 8/12/24 of every window (C.TurnExtractor, print from
                                 the user's own single-speaker speech elsewhere on the same channel) -> TD/inp|blk/
  p6    --vad --seg --turn       a 0.6B candidate's per-frame signals on the held-out windows -> TD/p6_<combo>.npz
  test                           does the scope reproduce the TurnBench ordering (v0.2 vs Q false interruptions)?

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/turn_data.py <stage> [...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import core_0p6b_turn as T6  # noqa: E402  (imports core_0p6b_heads as T6.C)

C = T6.C
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
IHM = SSD / "data" / "ami_ihm"
TD = SSD / "scratch" / "turndata"
OUT = ROOT / "runs" / "turn_data.json"
SR, FRAME, HOP = 16000, 0.08, 1280
WIN_S = 40.0
log = T6.log


def save(key, val, sub=None):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    if sub is None:
        d[key] = val
    else:
        d.setdefault(key, {})[sub] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def load_json(key, default=None):
    return json.loads(OUT.read_text()).get(key, default) if OUT.exists() else default


# --------------------------------------------------------------------------- AMI individual headsets
def channels() -> dict:
    """{meeting: {global speaker name: headset channel}} from the AMI meetings.xml."""
    p = ROOT / "data" / "ami" / "annotations" / "corpusResources" / "meetings.xml"
    return {m.get("observation"): {s.get("global_name"): int(s.get("channel")) for s in m.iter("speaker")}
            for m in ET.parse(p).getroot().iter("meeting")}


def split_meetings(split):
    from audioforge.datasets.ami import split_lists
    hold = set(C.VAD_HOLD)  # TS3011b / ES2015c: the turn_v4 / VAD held-out meetings stay out of training
    ms = split_lists()[{"dev": "dev", "test": "eval"}.get(split, "train")]
    return [m for m in ms if split in ("dev", "test") or m not in hold]


_HS = {}


def headset(m, ch):
    """The headset channel as float32 16 kHz (memory-mapped .npy cache next to the wav)."""
    k = (m, ch)
    if k not in _HS:
        npy = IHM / "cache" / f"{m}.Headset-{ch}.f32.npy"
        if not npy.exists():
            import soundfile as sf
            x, sr = sf.read(str(IHM / f"{m}.Headset-{ch}.wav"), dtype="float32", always_2d=True)
            assert sr == SR, sr
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, np.ascontiguousarray(x.mean(1)))
        _HS[k] = np.load(npy, mmap_mode="r")
    return _HS[k]


def ihm_audio(c):
    """(window audio, print audio) of a manifest window: the user's own headset channel."""
    x = headset(c["meeting"], c["ch"])
    w = np.asarray(x[int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)
    pr = np.concatenate([np.asarray(x[int(round(p * SR)): int(round(q * SR))], np.float32) for p, q in c["print"]])
    return w, pr


def stage_man(a):
    """User-channel windows of the --split meetings whose headsets are all on disk. Per participant (= channel): up
    to --per windows of 40 s, each placed around a randomly drawn user turn end (offset 8 .. 37 s into the window, as
    turn_v4's oto windows), non-overlapping; turns = the user's non-backchannel floor turns ending inside the window
    (start, end, next user onset clipped to the window, start inside the window); act = the user's word activity;
    print = 5 s of the user's single-speaker speech (nobody else within 0.2 s) outside the window, same channel."""
    import tsvad as TS
    from audioforge.datasets.ami import AMI
    chans = channels()
    ms = [m for m in split_meetings(a.split)
          if m in chans and all((IHM / f"{m}.Headset-{c}.wav").exists() for c in chans[m].values())]
    ds = AMI(ms, audio=False, verbose=False)
    rng = random.Random({"dev": 0, "train": 1, "test": 2}[a.split])
    clips = []
    for m in ms:
        dur = min(len(headset(m, c)) for c in chans[m].values()) / SR
        segs = TS.single_segments(ds, m)
        for spk, ch in sorted(chans[m].items(), key=lambda t: t[1]):
            turns = [t for t in ds.turns[m] if t["speaker"] == spk and not t["bc"]]
            if not turns:
                continue
            nxt = [turns[j + 1]["start"] if j + 1 < len(turns) else dur for j in range(len(turns))]
            cand = [t["end"] for t in turns if t["end"] < dur - 3]
            used = []
            for _ in range(60):
                if len(used) >= a.per or not cand:
                    break
                e = rng.choice(cand)
                a0 = round(max(0.0, e - rng.uniform(8.0, WIN_S - 3.0)), 2)
                b0 = min(dur, a0 + WIN_S)
                if b0 - a0 < 20 or any(abs(a0 - u) < WIN_S for u in used):
                    continue
                ivs = TS.clip_from(segs.get(spk, []), 5.0, rng, exclude=(a0 - 1.0, b0 + 1.0))
                if ivs is None:
                    continue
                tt = [[round(max(t["start"], a0) - a0, 3), round(t["end"] - a0, 3), round(min(n, b0) - a0, 3),
                       bool(t["start"] >= a0)] for t, n in zip(turns, nxt) if a0 < t["end"] <= b0 - 0.5]
                if not tt:
                    continue
                act = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in ds.acts[m][spk]
                       if q > a0 and p < b0]
                oth = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for s, ivs_ in ds.acts[m].items()
                       if s != spk for p, q in ivs_ if q > a0 and p < b0]
                used.append(a0)
                clips.append({"id": f"ihm_{m}_{ch}_{int(a0 * 100)}", "src": "ihm", "meeting": m, "ch": ch,
                              "a": a0, "b": round(b0, 2), "turns": tt, "act": act, "oth": oth,
                              "print": [[round(p, 3), round(q, 3)] for p, q in ivs]})
    TD.mkdir(parents=True, exist_ok=True)
    (TD / f"man_{a.split}.json").write_text(json.dumps(clips))
    n_t = sum(len(c["turns"]) for c in clips)
    h = sum(c["b"] - c["a"] for c in clips) / 3600
    log(f"man {a.split}: {len(ms)} meetings, {len(clips)} windows ({h:.1f} h), {n_t} user turns")
    save("man", {"meetings": ms, "windows": len(clips), "hours": round(h, 2), "turns": n_t,
                 "audio_hours_total": round(sum(min(len(headset(m, c)) for c in chans[m].values()) / SR * len(chans[m])
                                                for m in ms) / 3600, 1)}, a.split)


# --------------------------------------------------------------------------- otoSpeech 280 h (raw channels)
OTO280 = SSD / "data" / "oto280"
BLEED_DB = 15.0  # a user-channel segment is the other party's bleed when the other channel is this much louder


def oto280_ids(split):
    """dev = sessions of the downloaded slice that are NOT in the local 141 h slice (whose processed audio trained
    the shipped heads) and not an evaluation conversation, every 4th in index order; train = the rest minus the
    evaluation conversations and the turn_v4 / v5 held-out conversations."""
    import turn_v4 as V4
    from audioforge.datasets import dyadic as D
    ids = [i for i in D.list_ids("oto", root=OTO280)
           if D.oto_index(OTO280)[i]["tar"] <= "data/train/shard-000023.tar"]  # the fixed slice: shards 0-23
    ev = V4.eval_oto_ids()
    i141 = set(D.list_ids("oto"))
    vg = {m for s_, m in T6._val_groups() if s_ == "oto"}
    fresh = [i for i in ids if i not in i141 and i not in ev]
    dev = fresh[::4][:32]
    if split == "dev":
        return dev
    train = [i for i in ids if i not in set(dev) and i not in ev and i not in vg][:64]
    if split == "train":
        return train
    assert split == "test", split  # never touched: fresh sessions in neither dev nor train (only public-row use)
    return [i for i in fresh if i not in set(dev) and i not in set(train)][:32]


def _bleed_filter(words, ch16, other_words):
    """Drops user-channel VAD segments that are the other party's bleed: the segment overlaps the other party's
    speech for >= 80 % of its length and the other channel is >= BLEED_DB louder over it."""
    out = []
    x0, x1 = ch16
    for a0, b0, t in words:
        ov = sum(max(0.0, min(b0, q) - max(a0, p)) for p, q, _ in other_words)
        if ov >= 0.8 * (b0 - a0):
            i0, i1 = int(a0 * SR), int(b0 * SR)
            e0 = 10 * np.log10(np.mean(np.asarray(x0[i0:i1], np.float64) ** 2) + 1e-10)
            e1 = 10 * np.log10(np.mean(np.asarray(x1[i0:i1], np.float64) ** 2) + 1e-10)
            if e1 - e0 >= BLEED_DB:
                continue
        out.append((a0, b0, t))
    return out


def stage_oman(a):
    """otoSpeech-280h (raw) user-channel windows, turn_v4's oto recipe on the RAW channels: labels = Silero VAD v5 on
    each raw channel (the 141 h corpus's rule, audioforge.datasets.dyadic, cached under <oto280>/cache/labels) minus
    the other party's bleed (_bleed_filter), then ami.speaker_turns (turn_gap 0.5 s, floor rule, max_hold 2 s,
    backchannels = isolated segments <= 1 s); windows of 40 s around user turn ends, none overlapping a redaction;
    print = 5 s of the user's own single-party speech outside the window -> TD/man_o<split>.json. Resumable through
    the label cache (Silero ~1 min per session)."""
    import tsvad as TS
    import turn_v4 as V4
    from audioforge.datasets import ami
    from audioforge.datasets import dyadic as D
    ids = oto280_ids(a.split)
    t0 = time.time()
    done = []
    if a.shard:  # label-cache workers: --shard k/n builds the labels of ids[k::n] only
        k, n = map(int, a.shard.split("/"))
        for m in ids[k::n]:
            D.Dyadic([m], "oto", root=OTO280, verbose=False)
        log(f"oman {a.split} shard {a.shard}: done")
        return
    for m in ids:
        if time.time() - t0 > a.budget:
            break
        D.Dyadic([m], "oto", root=OTO280, verbose=False)  # builds / reads the label cache
        done.append(m)
    if len(done) < len(ids):
        log(f"oman {a.split}: labels {len(done)} / {len(ids)}; rerun")
        return
    ds = D.Dyadic(ids, "oto", root=OTO280, verbose=False)
    rng = random.Random({"dev": 10, "train": 11, "test": 12}[a.split])
    clips, nbleed = [], 0
    for m in ids:
        ch = ds.channels16k(m)
        w = {}
        for c in (0, 1):
            ww = ds.words[m][f"{m}:{c}"]
            w[f"{m}:{c}"] = _bleed_filter(ww, (ch[c], ch[1 - c]), ds.words[m][f"{m}:{1 - c}"])
            nbleed += len(ww) - len(w[f"{m}:{c}"])
        ds.words[m] = w
        ds.turns[m] = ami.speaker_turns(w, **ds.cfg)
        ds.acts[m] = {s_: ami.activity(ww, ds.cfg["act_bridge"]) for s_, ww in w.items()}
        dur = ds.duration(m)
        segs = V4.oto_single(ds, m)
        for c in (0, 1):
            party = f"{m}:{c}"
            turns = [t for t in ds.turns[m] if t["speaker"] == party and not t["bc"]]
            nxt = [turns[j + 1]["start"] if j + 1 < len(turns) else dur for j in range(len(turns))]
            cand = [t["end"] for t in turns]
            used = []
            for _ in range(40):
                if len(used) >= a.per or not cand:
                    break
                e = rng.choice(cand)
                a0 = round(max(0.0, e - rng.uniform(8.0, WIN_S - 3.0)), 2)
                b0 = min(dur, a0 + WIN_S)
                if any(abs(a0 - u) < WIN_S for u in used) or any(za < b0 and zb > a0 for za, zb in ds.zones[m]):
                    continue
                ivs = TS.clip_from(segs.get(party, []), 5.0, rng, exclude=(a0 - 1.0, b0 + 1.0))
                if ivs is None:
                    continue
                tt = [[round(max(t["start"], a0) - a0, 3), round(t["end"] - a0, 3), round(min(n, b0) - a0, 3),
                       bool(t["start"] >= a0)] for t, n in zip(turns, nxt) if a0 < t["end"] <= b0 - 0.5]
                if not tt:
                    continue
                act = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in ds.acts[m][party]
                       if q > a0 and p < b0]
                oth = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in ds.acts[m][f"{m}:{1 - c}"]
                       if q > a0 and p < b0]
                used.append(a0)
                clips.append({"id": f"o2_{m[:12]}_{c}_{int(a0 * 100)}", "src": "oto280", "meeting": m, "ch": c,
                              "a": a0, "b": round(b0, 2), "turns": tt, "act": act, "oth": oth,
                              "print": [[round(p, 3), round(q, 3)] for p, q in ivs]})
    TD.mkdir(parents=True, exist_ok=True)
    (TD / f"man_o{a.split}.json").write_text(json.dumps(clips))
    n_t = sum(len(c["turns"]) for c in clips)
    h = sum(c["b"] - c["a"] for c in clips) / 3600
    log(f"oman {a.split}: {len(ids)} sessions ({sum(ds.duration(m) for m in ids) / 3600:.1f} h), {len(clips)} windows "
        f"({h:.1f} h), {n_t} user turns; {nbleed} bleed segments dropped")
    save("man", {"sessions": ids, "windows": len(clips), "hours": round(h, 2), "turns": n_t, "bleed_dropped": nbleed,
                 "session_hours": round(sum(ds.duration(m) for m in ids) / 3600, 1)}, "o" + a.split)


# --------------------------------------------------------------------------- AppTek call-center dialogues (held-out only)
APPTEK = SSD / "data" / "apptek"
APPTEK_LOCS = {"atdev": ("en-US_General", "en-GB", "en-US_Southern", "en-CA"),  # held-out (selection)
               "attest": ("en-AU", "en-IE", "en-ZA", "en-IN")}  # never touched (public rows only)


def apptek_calls(split="atdev"):
    """[(locale, stem, {role: channel number}, segments)] of the downloaded calls (both channel files on disk)."""
    out = []
    for loc in APPTEK_LOCS[split]:
        roles = {}
        for l in open(APPTEK / "test" / loc / "metadata.jsonl"):
            r = json.loads(l)
            stem, ch = r["file_name"][len("audio/"):-len(".wav")].rsplit("_channel", 1)
            roles.setdefault(stem, {})[r["role"]] = int(ch)
        for l in open(APPTEK / "diarization" / loc / "metadata.jsonl"):
            r = json.loads(l)
            stem = r["file_name"][len("audio/"):-len(".wav")]
            if all((APPTEK / "test" / loc / "audio" / f"{stem}_channel{c}.wav").exists() for c in (1, 2)) \
                    and set(roles.get(stem, {}).values()) == {1, 2}:
                out.append((loc, stem, roles[stem], r["segments"]))
    return out


_AT = {}


def apptek_channel(loc, stem, ch):
    k = (loc, stem, ch)
    if k not in _AT:
        npy = APPTEK / "cache" / f"{stem}_channel{ch}.f32.npy"
        if not npy.exists():
            import soundfile as sf
            from audioforge.datasets.dyadic import resample16k
            x, sr = sf.read(str(APPTEK / "test" / loc / "audio" / f"{stem}_channel{ch}.wav"), dtype="float32", always_2d=True)
            npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy, resample16k(x, sr).mean(0).astype(np.float32))
        _AT[k] = np.load(npy, mmap_mode="r")
    return _AT[k]


def stage_aman(a):
    """AppTek call-center dialogues (CC BY-SA 4.0; the card excludes training: held-out ONLY) as user-channel windows:
    each call's two channel files are the two parties (role -> channel from the default config); words = the
    reference segments of the diarization config (one token per segment, as Behavior-SD's utterances: pauses inside a
    segment are not visible), turns = ami.speaker_turns (floor rule); 40 s windows around user turn ends (both
    parties in turn as the user); print = 5 s of the user's own segments with the other party silent (+-0.2 s)
    outside the window -> TD/man_atdev.json."""
    import tsvad as TS
    from audioforge.datasets import ami
    rng = random.Random(20 if a.split == "atdev" else 21)
    clips = []
    calls = apptek_calls(a.split)
    for loc, stem, roles, segs in calls:
        chmap = {"agent": roles["agent"], "customer": roles["customer"]}
        words = {f"{stem}:{chmap[r]}": [] for r in chmap}
        for g in segs:
            words[f"{stem}:{chmap[g['role']]}"].append((float(g["start"]), float(g["end"]), "x"))
        words = {k: sorted(v) for k, v in words.items()}
        cfg = dict(ami.LABEL_DEFAULTS)
        turns_all = ami.speaker_turns(words, **cfg)
        acts = {k: ami.activity(v, cfg["act_bridge"]) for k, v in words.items()}
        dur = min(len(apptek_channel(loc, stem, c)) for c in (1, 2)) / SR
        for c in (1, 2):
            party, other = f"{stem}:{c}", f"{stem}:{3 - c}"
            turns = [t for t in turns_all if t["speaker"] == party and not t["bc"]]
            nxt = [turns[j + 1]["start"] if j + 1 < len(turns) else dur for j in range(len(turns))]
            single = [(p, q) for p, q in acts[party] if q - p >= 1.0
                      and not any(o0 - 0.2 < q and o1 + 0.2 > p for o0, o1 in acts[other])]
            cand = [t["end"] for t in turns if t["end"] < dur - 3]
            used = []
            for _ in range(40):
                if len(used) >= a.per or not cand:
                    break
                e = rng.choice(cand)
                a0 = round(max(0.0, e - rng.uniform(8.0, WIN_S - 3.0)), 2)
                b0 = min(dur, a0 + WIN_S)
                if b0 - a0 < 20 or any(abs(a0 - u) < WIN_S for u in used):
                    continue
                ivs = TS.clip_from(single, 5.0, rng, exclude=(a0 - 1.0, b0 + 1.0))
                if ivs is None:
                    continue
                tt = [[round(max(t["start"], a0) - a0, 3), round(t["end"] - a0, 3), round(min(n, b0) - a0, 3),
                       bool(t["start"] >= a0)] for t, n in zip(turns, nxt) if a0 < t["end"] <= b0 - 0.5]
                if not tt:
                    continue
                act = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in acts[party] if q > a0 and p < b0]
                oth = [[round(max(p, a0) - a0, 3), round(min(q, b0) - a0, 3)] for p, q in acts[other] if q > a0 and p < b0]
                used.append(a0)
                clips.append({"id": f"at_{stem.split('_')[-1]}_{c}_{int(a0 * 100)}", "src": "apptek", "loc": loc,
                              "meeting": stem, "ch": c, "a": a0, "b": round(b0, 2), "turns": tt, "act": act,
                              "oth": oth, "print": [[round(p, 3), round(q, 3)] for p, q in ivs]})
    (TD / f"man_{a.split}.json").write_text(json.dumps(clips))
    n_t = sum(len(c["turns"]) for c in clips)
    log(f"aman: {len(calls)} calls, {len(clips)} windows ({sum(c['b'] - c['a'] for c in clips) / 3600:.1f} h), {n_t} user turns")
    save("man", {"calls": [c[1] for c in calls], "windows": len(clips), "turns": n_t}, a.split)


def at_audio(c):
    x = apptek_channel(c["loc"], c["meeting"], c["ch"])
    w = np.asarray(x[int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)
    pr = np.concatenate([np.asarray(x[int(round(p * SR)): int(round(q * SR))], np.float32) for p, q in c["print"]])
    return w, pr


_O2 = {}


def o2_audio(c):
    from audioforge.datasets import dyadic as D
    m = c["meeting"]
    if m not in _O2:
        _O2[m] = D.Dyadic([m], "oto", root=OTO280, verbose=False).channels16k(m)
    x = _O2[m][c["ch"]]
    w = np.asarray(x[int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)
    pr = np.concatenate([np.asarray(x[int(round(p * SR)): int(round(q * SR))], np.float32) for p, q in c["print"]])
    return w, pr


def codec(x, kind):
    """The telephone chain through ffmpeg: 16 kHz float -> 8 kHz G.711 mu-law ('ulaw') or 8 kHz Opus at N kbit/s
    ('opus12', 'opus16') -> back to 16 kHz float; aligned to the input (Opus pre-skip removed by the decoder) and cut /
    padded to its length."""
    import subprocess
    x = np.asarray(x, np.float32)
    if kind == "ulaw":
        enc = ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "mulaw", "-"]
        dec_in = ["-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "-"]
    else:
        enc = ["-ar", "8000", "-c:a", "libopus", "-b:a", f"{kind[4:]}k", "-application", "voip", "-f", "ogg", "-"]
        dec_in = ["-f", "ogg", "-i", "-"]
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    b = subprocess.run(base + ["-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-"] + enc, input=x.tobytes(),
                       capture_output=True, check=True).stdout
    y = np.frombuffer(subprocess.run(base + dec_in + ["-ar", "16000", "-ac", "1", "-f", "f32le", "-"], input=b,
                                     capture_output=True, check=True).stdout, np.float32)
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)].copy()


def clip_audio(c):
    x, pr = o2_audio(c) if c["src"] == "oto280" else at_audio(c) if c["src"] == "apptek" else ihm_audio(c)
    if c.get("gain_db") is not None:  # a level-varied copy (stage level): the whole channel scaled, floor included
        g = np.float32(10 ** (c["gain_db"] / 20))
        x, pr = x * g, pr * g
    if c.get("codec"):  # a telephone-chain copy (stage codec)
        x, pr = codec(x, c["codec"]), codec(pr, c["codec"])
    return x, pr


def stage_codec(a):
    """Telephone-chain copies of --src-split: every window gets one chain drawn per window (G.711 mu-law 8 kHz, or
    8 kHz Opus at 12 / 16 kbit/s, 'voip' mode) -> TD/man_<split>.json (ids + '_c')."""
    import zlib
    out = []
    for c in manifest(a.src_split):
        rng = np.random.default_rng(zlib.crc32(("codec" + c["id"]).encode()))
        out.append(dict(c, id=c["id"] + "_c", codec=str(rng.choice(["ulaw", "opus12", "opus16"]))))
    (TD / f"man_{a.split}.json").write_text(json.dumps(out))
    log(f"codec {a.src_split} -> {a.split}: {len(out)} windows")


LEVEL_DB = (-48.0, -34.0)  # the user's speech level drawn per window (TurnBench user channels: p10 -43.9, p50 -35.6)


def stage_level(a):
    """Level variation on real channels: every window of --src-split whose channel has a real noise floor (median
    frame outside the user's speech > -85 dBFS, i.e. not gated to digital zero) gets a copy with the WHOLE channel
    scaled (speech, floor, bleed alike) so the user's speech level is drawn from LEVEL_DB -> TD/man_<split>.json
    (ids + '_q'). Nothing is synthesised: the noise floor, room and crosstalk stay the recording's own."""
    import zlib
    src = manifest(a.src_split)
    st = json.loads((TD / f"stats_{a.src_split}.json").read_text()) if (TD / f"stats_{a.src_split}.json").exists() else {}
    out = []
    for c in src:
        if c["id"] not in st:
            x, _ = clip_audio(c)
            st[c["id"]] = chan_stats(x, c["act"], clip_mix(c, len(x)))
        s_ = st[c["id"]]
        if not (s_["floor_db"] > -85) or not np.isfinite(s_["speech_db"]):
            continue
        rng = np.random.default_rng(zlib.crc32(c["id"].encode()))
        tgt = float(rng.uniform(*LEVEL_DB))
        out.append(dict(c, id=c["id"] + "_q", gain_db=round(tgt - s_["speech_db"], 2)))
    (TD / f"stats_{a.src_split}.json").write_text(json.dumps(st))
    (TD / f"man_{a.split}.json").write_text(json.dumps(out))
    log(f"level {a.src_split} -> {a.split}: {len(out)} of {len(src)} windows (real floor), "
        f"{sum(len(c['turns']) for c in out)} turns; gain p50 {np.median([c['gain_db'] for c in out]):.1f} dB")


def clip_mix(c, n):
    if c["src"] == "apptek":
        out = np.zeros(n, np.float32)
        for ch in (1, 2):
            y = np.asarray(apptek_channel(c["loc"], c["meeting"], ch)[int(round(c["a"] * SR)): int(round(c["b"] * SR))],
                           np.float32)[:n]
            out[: len(y)] += y
        return out
    if c["src"] == "oto280":
        ch = _O2[c["meeting"]] if c["meeting"] in _O2 else (o2_audio(c), _O2[c["meeting"]])[1]
        return np.asarray(ch[0][int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)[:n] + \
            np.asarray(ch[1][int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)[:n]
    chans = channels()
    return sum(np.asarray(headset(c["meeting"], ch)[int(round(c["a"] * SR)): int(round(c["b"] * SR))], np.float32)[:n]
               for ch in chans[c["meeting"]].values())


def manifest(split):
    return json.loads((TD / f"man_{split}.json").read_text())


# --------------------------------------------------------------------------- channel statistics
def _frames_in(ivs, T):
    f = np.zeros(T, bool)
    for p, q in ivs:
        f[int(p / FRAME): int(np.ceil(q / FRAME))] = True
    return f


def chan_stats(x, act_ivs, mix=None):
    """speech dBFS (power mean over the user's active frames), floor dBFS (median of frames outside the user's
    activity), crosstalk = corr(user-channel dB, mix dB) over those outside frames (mix = both parties)."""
    T = len(x) // HOP
    db = 10 * np.log10(np.mean(x[: T * HOP].astype(np.float64).reshape(T, HOP) ** 2, 1) + 1e-10)
    act = _frames_in(act_ivs, T)
    sp = 10 * np.log10(np.mean(10 ** (db[act] / 10))) if act.any() else np.nan
    out = {"speech_db": float(sp), "floor_db": float(np.median(db[~act])) if (~act).any() else np.nan,
           "zero_frac": float(np.mean(db[~act] < -95)) if (~act).any() else np.nan}
    if mix is not None and (~act).sum() > 10:
        mdb = 10 * np.log10(np.mean(mix[: T * HOP].astype(np.float64).reshape(T, HOP) ** 2, 1) + 1e-10)
        g = ~act & (db > -95)
        out["xtalk_corr"] = float(np.corrcoef(db[g], mdb[g])[0, 1]) if g.sum() > 10 else np.nan
    return out


def stage_stats(a):
    """Speech level / floor / crosstalk of the held-out windows next to the oto held-out windows (clean and quiet
    variants) and the evaluation call clips (user channel vs the two-party mix; audio statistics only, no model
    output is read)."""
    import eot_latency as E
    import turn_v4 as V4
    rows = {}
    chans = channels()
    for sp in a.split.split(","):
        r = []
        for c in manifest(sp):
            x, _ = clip_audio(c)
            r.append(chan_stats(x, c["act"], clip_mix(c, len(x))))
        rows["scope_" + sp] = r
    man = T6._turn_manifest()
    au = V4.ClipAudio()
    vg = T6._val_groups()
    oto = [c for c in man if c["src"] == "oto" and (c["src"], c["meeting"]) in vg][: a.n] if a.n else []
    r, rq = [], []
    for c in oto:
        x, _ = au(c, man)
        d = au._d("oto", man)
        both = np.asarray(d.channels16k(c["meeting"]), np.float32)
        mix = (both[0] + both[1])[int(round(c["a"] * SR)): int(round(c["b"] * SR))][: len(x)]
        r.append(chan_stats(x, c["act"], mix))
        xq = T6.quiet_audio(c, au, man)[0]
        rq.append(chan_stats(xq, c["act"], mix))
    if oto:  # the old held-out scopes (selection audio: not for public numbers; --n 0 skips them)
        rows["oto_heldout"], rows["oto_quiet_heldout"] = r, rq
    for st in ("turnbench", "oto"):
        r = []
        for s in E.sessions():
            if s["set"] != st:
                continue
            x = E.read_audio(s)[: -int(s["pad_s"] * SR)]
            mono = Path(s["wav"].replace(".user.wav", ".mono.wav"))
            mix = None
            if mono.exists():
                import soundfile as sf
                mix = sf.read(str(mono), dtype="float32", always_2d=True)[0].mean(1)[: len(x)]
                x = x[: len(mix)]
            r.append(chan_stats(x, s["user_turns"], mix))
        rows[f"eval_{st}"] = r
    summ = {}
    for k, v in rows.items():
        summ[k] = {"n": len(v)}
        for q in ("speech_db", "floor_db", "xtalk_corr", "zero_frac"):
            arr = np.array([x_.get(q, np.nan) for x_ in v], float)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                summ[k][q] = [round(float(np.percentile(arr, p)), 2) for p in (10, 50, 90)]
        log(k, summ[k])
    save("stats", summ)


# --------------------------------------------------------------------------- features
def inp_dir(split):
    return TD / f"inp_{split}", TD / f"blk_{split}"


def stage_feats(a):
    """C.TurnExtractor (the served 0.6B inputs: VAD head, TS-VAD track from the print, greedy tokens + blocks 8 / 12 /
    24) on every window of --split; also the frame energies (dBFS per 80 ms frame). Resumable (--budget s)."""
    import torch
    import turn_v5 as V5
    torch.set_num_threads(2)
    di, db_ = inp_dir(a.split)
    for d_ in (di, db_, TD / "db"):
        d_.mkdir(parents=True, exist_ok=True)
    cl = manifest(a.split)
    if a.n:
        cl = cl[: a.n]
    todo = [c for c in cl if not (di / f"{c['id']}.npz").exists()]
    log(f"feats {a.split}: {len(todo)} of {len(cl)} to do")
    if not todo:
        return
    t0 = time.time()
    ex6 = C.TurnExtractor(a.device)
    done = 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        aud = [clip_audio(c) for c in cs]
        pr6 = ex6.prints([q[1] for q in aud])
        outs = ex6([q[0] for q in aud], pr6, (8, 12, 24))
        for c, (inp, blk), q in zip(cs, outs, aud):
            np.savez(db_ / f"{c['id']}.npz", **blk)
            np.savez(di / f"{c['id']}.npz", **inp)
            np.save(TD / "db" / f"{c['id']}.npy", V5.frame_abs_db(q[0], len(inp["pu"])))
        done += len(cs)
    log(f"feats {a.split}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")


def stage_teach115(a):
    """The distillation target of the v5 classifier on the IHM windows, as core_0p6b_turn's stage teacher makes it
    for every other clip: the 115M's shipped v5 (c5) on its own served inputs (turn_v4.Extractor: VAD, TS-VAD from the
    same print, tokens) and its pass-1 block 8 -> core_0p6b_turn TF/teacher/<id>.npy (fp16). Resumable."""
    import torch
    import turn_v4 as V4
    import turn_v5 as V5
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    od = T6.TF / "teacher"
    cl = manifest(a.split)
    todo = [c for c in cl if not (od / f"{c['id']}.npy").exists() and (inp_dir(a.split)[0] / f"{c['id']}.npz").exists()]
    log(f"teach115 {a.split}: {len(todo)} of {len(cl)} to do")
    if not todo:
        return
    t0 = time.time()
    model, _ = V5.load_seg(T6._V5_ORIG["W"] / "c5" / "model.pt", a.device)
    ex1 = V4.Extractor(device=a.device)
    done = 0
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        aud = [clip_audio(c) for c in cs]
        xs = [q[0] for q in aud]
        prints = [voiceprint(ex1.cpu, q[1]) if len(q[1]) >= 8000 else None for q in aud]
        o1 = ex1(xs, prints)
        with torch.no_grad():
            x_, xl = ex1.m._pad(xs)
            enc, elen, hid = ex1.m.encode(x_, xl, V4.ATT, return_hidden=True)
            b8 = hid[7].float().cpu().numpy()
        for j, (c, o) in enumerate(zip(cs, o1)):
            L = len(o["pu"])
            ex = np.stack([o["vad"], o["pu"], o["po"]], 1).astype(np.float32)
            p = V5.seg_probs(model, b8[j, :L].astype(np.float32), ex, np.asarray(o["n"], np.int32),
                             np.asarray(o["y"], np.int32), a.device, bs=512)
            np.save(od / f"{c['id']}.npy", p.astype(np.float16))
        done += len(cs)
    log(f"teach115 {a.split}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")


# --------------------------------------------------------------------------- held-out signals and scoring
def stage_p6(a):
    """A 0.6B candidate (--vad tag | served, --seg tag | s12, --turn tag | f1; core_0p6b_turn's heads) on the
    held-out windows -> TD/p6_<combo>.npz (vad, p5, p, pu, po per window)."""
    import torch
    torch.set_num_threads(2)
    name = T6.combo_name(a.vad, a.seg, a.turn)
    f = TD / f"p6_{a.split}_{name}.npz"
    if f.exists() and not a.force:
        log("exists", f)
        return
    vad_fn = None if a.vad == "served" else T6.load_vad(a.vad, a.device)[0]
    seg = T6._load_seg(a.seg, a.device)
    turn = T6._load_turn_head(a.turn, a.device)
    di, db_ = inp_dir(a.split)
    out = {}
    t0 = time.time()
    for c in manifest(a.split):
        if not (di / f"{c['id']}.npz").exists():
            continue
        z = np.load(di / f"{c['id']}.npz")
        d = {k: z[k] for k in z.files}
        zb = np.load(db_ / f"{c['id']}.npz")
        d.update({k: zb[k] for k in zb.files})
        r = T6.heads_on(d, vad_fn, seg, turn, a.device)
        for k in ("vad", "p5", "p"):
            out[f"{c['id']}|{k}"] = r[k].astype(np.float32)
        out[f"{c['id']}|pu"] = d["pu"].astype(np.float32)
        out[f"{c['id']}|po"] = d["po"].astype(np.float32)
    np.savez(f, **out)
    log(f"p6 {name}: {len(out) // 5} windows in {time.time() - t0:.0f} s")


def load_sig(split, name):
    z = np.load(TD / f"p6_{split}_{name}.npz")
    out = {}
    for k in z.files:
        cid, kk = k.split("|")
        out.setdefault(cid, {})[kk] = z[k]
    return out


def score(sig, split, rule, comp=T6.COMP["0p6b"], gcache=None, per_out=None):
    """One rule on the windows (eot_latency.score_session per window, pooled; turns whose start is outside the
    window are scored too, as turn_v4's oto windows)."""
    import eot_latency as E
    import turn_v5 as V5
    gcache = {} if gcache is None else gcache
    pk = "p" if rule["mode"] == "head" else rule.get("p5key", "p5")  # p5b: a preset's own classifier (turn_seg_a)
    per = []
    for c in manifest(split):
        s = sig.get(c["id"])
        if s is None:
            continue
        T = len(s["vad"])
        h = {"v": list(range(T)), "t": [round((v + 1) * FRAME, 4) for v in range(T)], "p": s[pk][:T],
             "vad": s["vad"][:T], "pu": s["pu"][:T], "po": s["po"][:T]}
        if "vad_m" in s:
            h["vad_m"] = s["vad_m"][:T]
        db = np.load(TD / "db" / f"{c['id']}.npy")[:T]
        tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, gcache.setdefault(c["id"], {}))]
        ss = {"user_turns": [(t[0], t[1]) for t in c["turns"]], "scored": [True] * len(c["turns"]),
              "next_onset": [t[2] for t in c["turns"]]}
        per.append(E.score_session(tt, ss, comp))
    if per_out is not None:
        per_out.extend(per)
    return E.pool(per)


def boot_ci(per, boot=1000, seed=0, groups=None):
    import eot_latency as E
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(boot):
        rows.append(E.pool([per[i] for i in rng.integers(0, len(per), len(per))]))
    out = {}
    for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct"):
        x = np.array([r[q] for r in rows if r[q] is not None], float)
        out[q] = [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)]
    return out


def paired_diff(pa, pb, boot=1000, seed=0):
    """Paired window bootstrap of (a - b) for FI and missed (percentage points): point, 95 % CI."""
    import eot_latency as E
    rng = np.random.default_rng(seed)
    n = len(pa)
    assert n == len(pb)

    def d(ix):
        A, B = E.pool([pa[i] for i in ix]), E.pool([pb[i] for i in ix])
        return [A["false_interruption_pct"] - B["false_interruption_pct"], A["missed_pct"] - B["missed_pct"]]
    pt = d(range(n))
    bs = np.array([d(rng.integers(0, n, n)) for _ in range(boot)])
    return {"fi_pp": [round(pt[0], 1)] + [round(float(np.percentile(bs[:, 0], q)), 1) for q in (2.5, 97.5)],
            "missed_pp": [round(pt[1], 1)] + [round(float(np.percentile(bs[:, 1], q)), 1) for q in (2.5, 97.5)]}


V02 = ("served__s12__f1", "S_v01heads_heldout_constants")
QC = ("mlp12oq_sa__kd1stq__b12", "Q")


def stage_test(a):
    """The test of the data (before any training): v0.2 (shipped heads + constants) and Q (heads + its held-out
    rules), each preset, on the new held-out scope; next to the same two on the evaluation calls split by corpus
    (TurnBench / oto) from the cached evaluation signals of CORE_0P6B_TURN (no new evaluation run)."""
    rules = T6.load_json("evverify")
    res, pers = {}, {}
    for nm, (combo, rk) in (("v0.2", V02), ("Q", QC)):
        sig = load_sig(a.split, combo)
        for pn, r in rules[rk]["rules"].items():
            if pn == "assistant":
                continue
            per = []
            p = score(sig, a.split, r, per_out=per)
            p["ci95"] = boot_ci(per)
            res.setdefault(nm, {})[pn] = p
            pers[(nm, pn)] = per
            log(f"{a.split} {nm} {pn}: p50 {p['eot_total_ms_p50']} FI {p['false_interruption_pct']} "
                f"{p['ci95']['false_interruption_pct']} missed {p['missed_pct']} {p['ci95']['missed_pct']} (n {p['n_turns']})")
    for pn in ("balanced", "fast"):
        res.setdefault("paired_Q_minus_v0.2", {})[pn] = paired_diff(pers[("Q", pn)], pers[("v0.2", pn)])
        log(f"{a.split} paired Q - v0.2 {pn}:", res["paired_Q_minus_v0.2"][pn])
    # the evaluation calls by corpus (cached signals; reported, never used to choose)
    import eot_latency as E
    import turn_v5 as V5
    ev = {}
    for nm, (combo, rk) in (("v0.2", ("dump", V02[1])), ("Q", (QC[0], "Q"))):
        D, _ = T6.ev_sets(combo)
        for pn, r in rules[rk]["rules"].items():
            if pn == "assistant":
                continue
            pk = "p" if r["mode"] == "head" else "p5"
            per = {"turnbench": [], "oto": []}
            for ss in E.sessions():
                if ss["key"] not in D or ss["set"] not in per:
                    continue
                d = D[ss["key"]]
                h = dict(d["head"])
                h["p"] = h[pk]
                comp = d["chunk_ms"]["p50"] / 1000
                db = np.load(V5.ENERGY / f"{ss['key']}.npy")
                db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
                tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, r, {})]
                per[ss["set"]].append(E.score_session(tt, ss, comp))
            for st, P in per.items():
                ev.setdefault(nm, {}).setdefault(pn, {})[st] = E.pool(P)
            log(f"eval {nm} {pn}: TurnBench {T6.short({'calls': ev[nm][pn]['turnbench'], 'meet': ev[nm][pn]['oto'], 'asst': {'accuracy_pct': '', 'p50': '', 'false_fire_pct': ''}})}")
    save("test", {"heldout": res, "eval_by_corpus": ev}, a.split)


def _quiet_runs(db, turns, vads, rel_db=20.0, min_len=3):
    """In-turn quiet runs: inside each user turn [s, e - 80 ms), runs of >= min_len frames whose energy is more than
    rel_db under the session's in-turn speech level (power mean). Per run: length (frames) and, per VAD, the longest
    stretch of frames with VAD < 0.4 inside it (what the silence clock can count)."""
    T = len(db)
    inturn = np.zeros(T, bool)
    for s0, e1 in turns:
        inturn[int(s0 / FRAME): max(int(s0 / FRAME), int((e1 - 0.08) / FRAME))] = True
    if not inturn.any():
        return []
    lvl = 10 * np.log10(np.mean(10 ** (db[inturn] / 10)))
    q = inturn & (db < lvl - rel_db)
    out, v = [], 0
    while v < T:
        if not q[v]:
            v += 1
            continue
        u = v
        while u < T and q[u]:
            u += 1
        if u - v >= min_len:
            r = {"len": u - v}
            for k, vad in vads.items():
                if k.startswith("max_"):
                    r[k] = float(np.max(vad[v:u]))
                    continue
                best = cur = 0
                for x in vad[v:u]:
                    cur = cur + 1 if x < 0.4 else 0
                    best = max(best, cur)
                r[k] = best
            out.append(r)
        v = u
    return out


def stage_diag(a):
    """Why a scope does or does not reproduce TurnBench (diagnostic; evaluation signals of CORE_0P6B_TURN's cache are
    read but nothing is chosen on them): in-turn quiet runs (_quiet_runs) per scope, their length, and how much of
    each run v0.2's served VAD and Q's VAD leave below 0.4 (the time the 720 / 960 ms fallbacks can count)."""
    import eot_latency as E
    import turn_v5 as V5
    scopes = {}
    D2, _ = T6.ev_sets("dump")
    DQ, _ = T6.ev_sets(QC[0])
    for ss in E.sessions():
        if ss["key"] not in D2 or ss["set"] not in ("turnbench", "oto"):
            continue
        h2, hq = D2[ss["key"]]["head"], DQ[ss["key"]]["head"]
        db = np.load(V5.ENERGY / f"{ss['key']}.npy")
        db = db[np.clip(np.asarray(h2["v"]), 0, len(db) - 1)]
        scopes.setdefault("eval_" + ss["set"], []).extend(
            _quiet_runs(db, ss["user_turns"], {"v02": np.asarray(h2["vad"]), "Q": np.asarray(hq["vad"]),
                                               "max_pQ": np.asarray(hq["p"]), "max_p5Q": np.asarray(hq["p5"]),
                                               "max_p5v02": np.asarray(h2["p5"]), "max_po": np.asarray(h2["po"]),
                                               "max_pu": np.asarray(h2["pu"])}))
    for sp in a.split.split(","):
        s2, sq = load_sig(sp, V02[0]), load_sig(sp, QC[0])
        for c in manifest(sp):
            if c["id"] not in s2:
                continue
            db = np.load(TD / "db" / f"{c['id']}.npy")
            scopes.setdefault("scope_" + sp, []).extend(
                _quiet_runs(db, [(t[0], t[1]) for t in c["turns"]],
                            {"v02": s2[c["id"]]["vad"], "Q": sq[c["id"]]["vad"], "max_pQ": sq[c["id"]]["p"],
                             "max_p5Q": sq[c["id"]]["p5"], "max_p5v02": s2[c["id"]]["p5"], "max_po": s2[c["id"]]["po"],
                             "max_pu": s2[c["id"]]["pu"]}))
    res = {}
    for k, rs in scopes.items():
        L = np.array([r["len"] for r in rs])
        v2 = np.array([r["v02"] for r in rs])
        vq = np.array([r["Q"] for r in rs])
        res[k] = {"runs": len(rs), "len_ms_p50": float(np.median(L) * 80) if len(L) else None,
                  "runs_ge_720ms_pct": round(100 * float(np.mean(L >= 9)), 1),
                  "v02_clock_ge_720ms_pct": round(100 * float(np.mean(v2 >= 9)), 1),
                  "Q_clock_ge_960ms_pct": round(100 * float(np.mean(vq >= 12)), 1),
                  "Q_clock_ge_720ms_pct": round(100 * float(np.mean(vq >= 9)), 1),
                  "v02_share_of_run_below_0.4": round(float(np.sum(v2) / max(1, np.sum(L))), 3),
                  "Q_share_of_run_below_0.4": round(float(np.sum(vq) / max(1, np.sum(L))), 3)}
        for kk in ("max_pQ", "max_p5Q", "max_p5v02", "max_po", "max_pu"):
            x = np.array([r[kk] for r in rs])
            res[k][kk + "_ge_0.9_pct"] = round(100 * float(np.mean(x >= 0.9)), 1)
            res[k][kk + "_p50"] = round(float(np.median(x)), 3)
        log(k, res[k])
    save("diag", res)


# --------------------------------------------------------------------------- hybrid rules on every held-out scope
HO_SPLITS = ("dev", "devq", "odev", "odevq", "atdev")  # the real-channel held-out scopes (+ core_0p6b_turn's "ho")


def hsig(split, base, clock, seg):
    """Per-clip signals of a hybrid candidate: vad / p / pu / po from the --base combo (the served GRU VAD), vad_m =
    the --clock combo's (stateless) VAD, p5 = the --seg combo's classifier. split 'ho' = core_0p6b_turn's held-out."""
    ld = (lambda c: T6.load_ho(f"p6_{c}.npz")) if split == "ho" else (lambda c: load_sig(split, c))
    B, K, S = ld(base), ld(clock) if clock else None, ld(seg)
    out = {}
    for k, d in B.items():
        x = dict(d)
        if K is not None:
            x["vad_m"] = K[k]["vad"]
        x["p5"] = S[k]["p5"]
        out[k] = x
    return out


def all_scores(sigs, rule, gcs):
    """{scope: pooled result} of one rule on every held-out scope (ho -> calls / quiet / meet / asst)."""
    r = T6.ho_score(sigs["ho"], T6.ho_meta(), rule, T6.COMP["0p6b"], gcs.setdefault("ho", {}))
    out = {"calls": r["calls"], "quiet": r["quiet"], "meet": r["meet"], "asst": r["asst"]}
    if rule.get("fam") != "assistant":
        for sp in HO_SPLITS:
            out[sp] = score(sigs[sp], sp, rule, gcache=gcs.setdefault(sp, {}))
    return out


MCLOCKS = (True,)


def hgrid(fam):
    import itertools
    G = []
    if fam in ("balanced", "fast"):
        for mc, k, mvt, mp, fb, rt in itertools.product(MCLOCKS, (1, 2, 3, 4), (0.4, 0.5), (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
                                                        (9, 12, 16), (None, 0.25)):
            G.append({"mode": "model", "gate": True, "quiet_db": None, "k": k, "mvt": mvt, "mp": mp, "fb": fb,
                      "vad_thr": 0.4, "reask": True, "others": (12, 8), "rt": rt, "mclock": mc})
    else:
        for mc, k, mvt, mp, qd, fb in itertools.product(MCLOCKS, (2, 3, 4), (0.4, 0.5), (0.9, 0.95, 0.97, 0.98, 0.99),
                                                        (6.0, None), (37, 40, 43)):
            G.append({"mode": "model", "gate": True, "quiet_db": qd, "mqo": qd is not None, "k": k, "mvt": mvt,
                      "mp": mp, "fb": fb, "vad_thr": 0.4, "reask": True, "others": (12, 8), "mclock": mc,
                      "fam": "assistant"})
    return G


def nonreg(r, b, fam):
    """r no worse than b on every held-out scope: fast / balanced FI and missed on calls, quiet, meet and every
    real-channel scope; assistant accuracy, false fires and p50 on st3."""
    if fam == "assistant":
        a_, b_ = r["asst"], b["asst"]
        return a_["p50"] is not None and a_["p50"] <= b_["p50"] and a_["false_fire_pct"] <= b_["false_fire_pct"] \
            and a_["accuracy_pct"] >= b_["accuracy_pct"]
    for sc in ("calls", "quiet", "meet") + HO_SPLITS:
        if r[sc]["eot_total_ms_p50"] is None or r[sc]["false_interruption_pct"] > b[sc]["false_interruption_pct"] \
                or r[sc]["missed_pct"] > b[sc]["missed_pct"]:
            return False
    return True


def stage_hscan(a):
    """Every hybrid rule of hgrid(--fam) for one candidate (--base / --clock / --seg combos) on every held-out scope
    -> TD/hscan_<tag>_<fam>.json (resumable, --budget), with the v0.2 baseline (served heads, its rules) first."""
    rules = T6.load_json("evverify")
    v02 = rules[V02[1]]["rules"]
    splits = ("ho",) + HO_SPLITS
    sig02 = {sp: hsig(sp, V02[0], None, V02[0]) for sp in splits}
    global MCLOCKS
    MCLOCKS = (False, True) if a.both_clocks else (False,) if not a.clock else (True,)
    sigs = {sp: hsig(sp, a.base, a.clock, a.seg) for sp in splits}
    t0 = time.time()
    for fam in a.fam.split(","):
        f = TD / f"hscan_{a.tag}_{fam}.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        if "base" not in d:
            d["base"] = all_scores(sig02, dict(v02[fam], fam=fam), {})
            d["rows"] = []
        G = hgrid(fam)
        gcs = {}
        for r in G[len(d["rows"]):]:
            d["rows"].append({"rule": r, **all_scores(sigs, r, gcs)})
            if time.time() - t0 > a.budget:
                break
        f.write_text(json.dumps(d))
        ok = [x for x in d["rows"] if nonreg(x, d["base"], fam)]
        log(f"hscan {a.tag} {fam}: {len(d['rows'])}/{len(G)} rules ({time.time() - t0:.0f} s), {len(ok)} no worse than v0.2")
        if len(d["rows"]) < len(G):
            return


def pick_rule(d, fam):
    """The pre-registered held-out pick (research/TURN_DATA.md §2.2) from an hscan file's rows: balanced / fast = the
    fastest (calls + quiet p50) rule no worse than the baseline on every conversational scope, else None; assistant =
    the fastest st3 p50 with accuracy >= and false fires <= the baseline's (ties: accuracy, then false fires)."""
    b = d["base"]
    if fam == "assistant":
        ok = [r for r in d["rows"] if r["asst"]["p50"] is not None and r["asst"]["accuracy_pct"] >= b["asst"]["accuracy_pct"]
              and r["asst"]["false_fire_pct"] <= b["asst"]["false_fire_pct"]]
        key = lambda r: (r["asst"]["p50"], -r["asst"]["accuracy_pct"], r["asst"]["false_fire_pct"],  # noqa: E731
                         bool(r["rule"].get("mclock")))  # ties: the plain clock (no second VAD needed)
        return (min(ok, key=key) if ok else None), len(ok)
    ok = [r for r in d["rows"] if nonreg(r, b, fam)]
    return (min(ok, key=lambda r: (r["calls"]["eot_total_ms_p50"] + r["quiet"]["eot_total_ms_p50"],
                                   r["calls"]["false_interruption_pct"])) if ok else None), len(ok)


def stage_pick(a):
    """The held-out picks of --tags (hscan files) per family -> runs/turn_data.json pick.<tag>.<fam>."""
    res = load_json("pick", {})
    for tag in a.tags.split(","):
        for fam in a.fam.split(","):
            f = TD / f"hscan_{tag}_{fam}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            pk, n = pick_rule(d, fam)
            res.setdefault(tag, {})[fam] = {"n_rules": len(d["rows"]), "n_pass": n, "pick": pk, "base": d["base"]}
            log(f"{tag} {fam}: {n}/{len(d['rows'])} pass; pick {pk['rule'] if pk else None}")
    save("pick", res)


def stage_hocheck(a):
    """The served candidate (--afm) equals its offline rule twin on HELD-OUT audio (--split windows, first --n), per
    preset: the served session's turn_end times vs core_0p6b_turn.served_rules replayed on the session's own frames
    (the servedcheck protocol without touching an evaluation set) -> runs/turn_data.json hocheck.<afm stem>."""
    import torch
    import audioforge.serve as S
    import turn_v5 as V5
    torch.set_num_threads(2)
    afm = Path(a.afm)
    eng = T6.engine_cand(afm, a.device)
    eng.warmup()
    rules = T6.served_rules(afm)
    res = {}
    cl = manifest(a.split)[: a.n]
    for preset in a.fam.split(","):
        same = 0
        for c in cl:
            x, pr = clip_audio(c)
            s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
            if s.asr.seg is None and eng.seg_name is not None:
                sh = eng.asr.heads[eng.seg_name]
                s.asr.attach_seg(sh, next(sh.parameters()).device)
            rec = []
            C._record(s, rec, enrolled_only=True)
            msgs = []
            for i in range(0, len(x), 320):
                msgs += s.process(x[i:i + 320])
            msgs += s.finish()
            h = {k: [r[j] for r in rec] for j, k in enumerate(("v", "t", "p", "vad", "pu", "po", "p5", "vad_m", "p5b"))}
            r = dict(rules[preset])
            if r["mode"] == "model":
                h["p"] = h[r.get("p5key", "p5")]
            db = V5.ready_db(x, max(h["v"]) + 1, h["v"], h["t"])
            off = [round(t_, 3) for t_, _ in V5.run_policy({"head": h}, db[np.asarray(h["v"])], r, {})]
            served = [m_["t"] for m_ in msgs if m_["type"] == "turn_end"]
            same += served == off
        res[preset] = f"{same}/{len(cl)}"
        log(afm.stem, preset, res[preset])
    save("hocheck", res, afm.stem)


def stage_same(a):
    """balanced / fast unchanged by construction: the served --old and --afm engines give identical turn_end times on
    --n HELD-OUT windows (--split), per preset (--fam) -> runs/turn_data.json same.<new stem>."""
    import torch
    import audioforge.serve as S
    torch.set_num_threads(2)
    engs = {k: T6.engine_cand(Path(f), a.device) for k, f in (("old", a.old), ("new", a.afm))}
    for e in engs.values():
        e.warmup()
    res = {}
    cl = manifest(a.split)[: a.n]
    for preset in a.fam.split(","):
        same = 0
        for c in cl:
            x, _ = clip_audio(c)
            te = {}
            for k, eng in engs.items():
                s = S.Session(eng, S.SessionConfig(turn_policy="vad_head", turn_preset=preset))
                msgs = []
                for i in range(0, len(x), 320):
                    msgs += s.process(x[i:i + 320])
                msgs += s.finish()
                te[k] = [round(m_["t"], 3) for m_ in msgs if m_["type"] == "turn_end"]
            same += te["old"] == te["new"]
        res[preset] = f"{same}/{len(cl)}"
        log(preset, res[preset])
    save("same", res, Path(a.afm).stem)


FC_V03 = SSD / "scratch" / "final_compare" / "eot"  # FINAL_COMPARE's 0.6B v0.3 dumps (the "before")
FC_V04 = TD / "fc_v04" / "eot"  # this run's v0.4 dumps (final_compare eotdump, FINAL_0P6B_AFM = v0.4)
AFM_V03 = SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.3.afm"
AFM_V04 = SSD / "scratch" / "core_0p6b" / "served_0p6b_v0.4.afm"


def stage_evalv04(a):
    """The single verification of the shipped heads: FINAL_COMPARE's turn sets (smart-turn v3.2 test clips; AMI test
    turns; the calls = TurnBench dev + oto, not a public test split), v0.3 (before) and v0.4 (after) served dumps, each
    preset as the served model resolves it, core_0p6b_turn.ev_score with a 1000-resample session bootstrap; and the
    assistant preset on the never-touched real-channel scopes (offline twin; served == offline is checked separately).
    -> runs/turn_data.json evalv04."""
    import eot_assistant as EA
    import eot_latency as E
    assert E.AMI_SPLIT == "eval", "EOT_AMI_SPLIT=eval (the AMI test turns)"
    sd = EA.load_dump()
    sess = E.sessions()
    clips = EA.clips()
    res = load_json("evalv04", {}) if a.never_only else {}
    for tag, d0, afm in () if a.never_only else (("v0.3", FC_V03, AFM_V03), ("v0.4", FC_V04, AFM_V04)):
        D = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted((d0 / "0p6b_calls").glob("*.json")))}
        A = {x["key"]: x for x in (json.loads(p_.read_text()) for p_ in sorted((d0 / "0p6b_asst").glob("*.json")))}
        assert all(s_["key"] in D for s_ in sess) and all(c["key"] in A for c in clips), (tag, len(D), len(A))
        D = {s_["key"]: D[s_["key"]] for s_ in sess}
        for kk, d in A.items():
            d["conf"] = sd[kk]["conf"]
        rules = T6.served_rules(afm)
        o = {"afm": str(afm), "rules": rules,
             "chunk_ms_p50_median": round(float(np.median([d["chunk_ms"]["p50"] for d in D.values()])), 2)}
        for pn, r in rules.items():
            o[pn] = T6.ev_score(D, A, r, boot=a.boot)
            x = o[pn]
            log(tag, pn, "asst", x["asst"]["accuracy_pct"], x["asst"]["p50"], x["asst"]["false_fire_pct"], x["asst"]["ci95"],
                "| ami", x["ami"]["eot_total_ms_p50"], x["ami"]["false_interruption_pct"], x["ami"]["missed_pct"],
                "| calls", x["calls"]["eot_total_ms_p50"], x["calls"]["false_interruption_pct"], x["calls"]["missed_pct"])
        res[tag] = o
    # the assistant preset on the never-touched real-channel scopes (conversational audio: interrupt / missed / p50)
    rules = T6.load_json("evverify")
    v02a = dict(rules[V02[1]]["rules"]["assistant"])
    pk = load_json("pick")["hkr"]["assistant"]["pick"]["rule"]
    nv = {}
    for sp in ("test", "testq", "otest", "otestq", "attest"):
        before = score(load_sig(sp, V02[0]), sp, v02a, per_out=(pb := []))
        after = score(hsig(sp, V02[0], None, "mlp12r__kd1stqr__f1"), sp, pk, per_out=(pa := []))
        before["ci95"], after["ci95"] = boot_ci(pb), boot_ci(pa)
        nv[sp] = {"v0.3": before, "v0.4": after, "paired_v04_minus_v03": paired_diff(pa, pb)}
        log("never", sp, "assistant before", before["eot_total_ms_p50"], before["false_interruption_pct"], before["missed_pct"],
            "after", after["eot_total_ms_p50"], after["false_interruption_pct"], after["missed_pct"], nv[sp]["paired_v04_minus_v03"])
    res["never_touched_assistant"] = nv
    save("evalv04", res)


def stage_splits(a):
    """The three disjoint sets, checked by recording id (meeting / session / call), not by clip:
    train (new + what the shipped heads were trained on), held-out (selection: new + core_0p6b_turn's turn_v4 / v5
    split), never touched (new sources' public-row parts) and the evaluation sets (smart-turn v3.2 test, AMI test
    meetings, the call sessions: TurnBench dev + 16 oto conversations). -> TD/splits.json + runs/turn_data.json
    splits (every pairwise intersection that must be empty, with its size)."""
    import turn_v4 as V4
    from audioforge.datasets import smartturn as ST
    from audioforge.datasets.ami import split_lists
    from audioforge.datasets import icsi as I
    import eot_latency as E
    sl = split_lists()
    rec = lambda src, ids: {f"{src}:{i}" for i in ids}  # noqa: E731
    man = lambda sp: {c["meeting"] for c in manifest(sp)} if (TD / f"man_{sp}.json").exists() else set()  # noqa: E731
    vg = T6._val_groups()
    tv4 = T6._turn_manifest()
    old_train = set()
    for c in tv4:
        g = (c["src"], c.get("meeting", c["id"]))
        if c["src"] in ("oto", "ami", "icsi") and g not in vg:
            old_train.add(f"{'oto' if c['src'] == 'oto' else c['src']}:{g[1]}")
    old_train |= rec("ami", [m for m in sl["train"] if m not in C.VAD_HOLD])  # VAD / turn training meetings
    old_train |= rec("icsi", I.SPLITS["train"])
    old_held = {f"{s_}:{m}" for s_, m in vg if s_ in ("oto", "ami", "icsi")} | rec("ami", C.VAD_HOLD)
    ev_calls = set()
    for x in E.sessions():
        if x["set"] in ("oto", "turnbench"):
            ev_calls.add(x["key"])
    oto_eval = V4.eval_oto_ids()
    evals = rec("oto", oto_eval) | rec("ami", sl["eval"])  # + TurnBench dev conversations (a corpus of its own)
    meta = json.loads(ST.build_cache(verbose=False).read_text())
    sp_ = ST.split_indices(meta)
    st_test, st_train = set(sp_["eval"]), set(sp_["train"])
    sets = {
        "train_new": rec("ami", man("train")) | rec("oto", man("otrain")),
        "heldout_new": rec("ami", man("dev")) | rec("oto", man("odev")) | rec("apptek", man("atdev")),
        "never_new": rec("ami", man("test")) | rec("oto", man("otest")) | rec("apptek", man("attest")),
        "train_shipped": old_train, "heldout_old": old_held, "eval": evals,
    }
    pairs = [("train_new", "heldout_new"), ("train_new", "never_new"), ("heldout_new", "never_new"),
             ("train_new", "eval"), ("heldout_new", "eval"), ("heldout_new", "train_shipped"),
             ("never_new", "train_shipped"), ("never_new", "heldout_old"), ("train_new", "heldout_old"),
             ("heldout_new", "heldout_old"), ("train_shipped", "eval")]
    res = {"sizes": {k: len(v) for k, v in sets.items()}, "smart_turn": {"train": len(st_train), "test": len(st_test),
                                                                        "overlap": len(st_train & st_test)}}
    for x, y in pairs:
        inter = sorted(sets[x] & sets[y])
        res[f"{x} & {y}"] = len(inter)
        if inter:
            res[f"{x} & {y} ids"] = inter[:10]
    # never_new & eval: the AMI test meetings are evaluation audio in both (allowed: neither is trained / selected on)
    res["never_new & eval (allowed: both evaluation-only)"] = len(sets["never_new"] & sets["eval"])
    (TD / "splits.json").write_text(json.dumps({k: sorted(v) for k, v in sets.items()}, indent=0))
    for k, v in res.items():
        log(k, v)
    bad = [k for k, v in res.items() if "&" in k and "allowed" not in k and not k.endswith("ids") and v]
    save("splits", res)
    if bad:
        raise SystemExit(f"NOT DISJOINT: {bad}")
    log("splits: all required intersections are empty")


STAGES = {"man": stage_man, "oman": stage_oman, "aman": stage_aman, "level": stage_level, "codec": stage_codec, "stats": stage_stats, "feats": stage_feats, "teach115": stage_teach115, "p6": stage_p6,
          "test": stage_test, "diag": stage_diag, "hscan": stage_hscan, "splits": stage_splits, "pick": stage_pick, "hocheck": stage_hocheck, "same": stage_same, "evalv04": stage_evalv04}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--vad", default="served")
    ap.add_argument("--seg", default="s12")
    ap.add_argument("--turn", default="f1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default="")
    ap.add_argument("--src-split", default="odev")
    ap.add_argument("--base", default="served__s12__f1")
    ap.add_argument("--clock", default="mlp12oq_sa__kd1stq__b12")
    ap.add_argument("--tag", default="h")
    ap.add_argument("--both-clocks", action="store_true")
    ap.add_argument("--tags", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--never-only", action="store_true")
    ap.add_argument("--afm", default="")
    ap.add_argument("--old", default="/Volumes/ExternalSSD/nvidia-audio-models/scratch/core_0p6b/served_0p6b_v0.3.afm")
    ap.add_argument("--fam", default="balanced,fast")
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
