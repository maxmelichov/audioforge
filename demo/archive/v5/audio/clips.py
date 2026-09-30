"""v5 AUDIO agent: example clips chosen for intelligibility, processed, scored, with subtitles and agent-voice placements.

    G="scripts/dev/gate.sh env PYTHONPATH=.:demo/v5 .venv/bin/python demo/archive/v5/audio/clips.py"
    $G transcribe [--budget 480]   # stage 1: Whisper small + large-v3-turbo word timestamps on whole raw user channels (cached)
    $G score      [--budget 480]   # stage 2: candidate spans -> processed WAV, Whisper small before/after, WER (cached)
    $G choose                      # stage 3: pick per example, write clips/*.wav, clips_manifest.json, subs.json, table

Outputs ($DEMO_OUT/v5_data/): clips/<id>.wav (48 kHz mono float, exactly the span [t0, t1], fades baked in),
clips_manifest.json, subs.json, clips/intelligibility.md. Clip time = the clip WAV's time in
scratch/e2e_tsvad/clips (the per_clip timeline of runs/e2e_final.json); room: AMI IS1008b meeting time - 1670 s.

Processing chain (ffmpeg): high-pass 80 Hz, +3 dB presence (one octave around 2.83 kHz = 2-4 kHz), light compression
(2:1 above -22 dBFS, soft knee), two-pass loudnorm to -16 LUFS integrated (linear; if loudnorm refuses linear mode, the
measured linear gain instead), true-peak limiter at -1 dBTP (limiter run at 4x oversampling, 192 kHz), 150 ms fade in/out.
WER: Whisper small (faster-whisper int8, beam 5, en) on the raw cut (before) and on the processed clip (after), against
the reference words whose midpoint lies in the span (AMI word XML; TurnBench annotator a aligned to Whisper turbo word
times; otoSpeech has no transcript: Whisper large-v3-turbo on the whole raw channel is the pseudo-reference).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "demo" / "v5")]
import examples as EX  # noqa: E402  (text normalisation, reference loaders, TurnBench aligner: read-only reuse)

DEMO_OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
V5 = DEMO_OUT / "v5_data"
CLIPS = V5 / "clips"
WORK = CLIPS / "_work"
CACHE = WORK / "cache.json"
SRC_DIR = EX.CLIP_DIR
ROOM_WAV, ROOM_T0 = EX.ROOM_WAV, EX.ROOM_WAV_T0
SR = 48000
FADE = 0.15
CHAIN = ("highpass=f=80:poles=2,equalizer=f=2830:t=o:w=1:g=3,"
         "acompressor=threshold=-22dB:ratio=2:attack=15:release=200:knee=6:makeup=1")
TP_LIMIT = "aresample=192000,alimiter=limit=0.85:attack=1:release=60:level=false:asc=1,aresample=48000"
AGENT_JSON = DEMO_OUT / "agent_voice_v5.json"


# ------------------------------------------------------------------ io / cache
def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(c):
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c))
    tmp.replace(CACHE)


def ff(args):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-threads", "2", *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])
    return p.stderr


def measure(path):
    err = ff(["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"])
    s = err[err.rindex("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", s).group(1))
    m = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", s)
    return lufs, (float(m.group(1)) if m and m.group(1) != "-inf" else None)


def src_of(clip, ch="user"):
    return ROOM_WAV if clip.startswith("room_") else SRC_DIR / f"{clip}.{ch}.wav"


# ------------------------------------------------------------------ processing
def cut_raw(src, t0, t1, dst):
    ff(["-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(dst)])


def process(src, t0, t1, dst):
    """Exactly [t0, t1] of src -> dst (48 kHz mono float) through the clarity chain, -16 LUFS, -1 dBTP, 150 ms fades."""
    d = t1 - t0
    cut = ["-ss", f"{t0:.3f}", "-t", f"{d:.3f}", "-i", str(src)]
    base = f"aresample={SR},{CHAIN}"
    fades = f"afade=t=in:st=0:d={FADE},afade=t=out:st={d - FADE:.3f}:d={FADE}"
    err = ff([*cut, "-af", f"{base},loudnorm=I=-16:TP=-1:LRA=11:print_format=json", "-f", "null", "-"])
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json")
    err = ff([*cut, "-af", f"{base},{ln},{TP_LIMIT},{fades}", "-t", f"{d:.3f}", "-ac", "1", "-c:a", "pcm_f32le", "-y", str(dst)])
    m2 = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    lufs, tp = measure(dst)
    method = "loudnorm two-pass linear"
    if m2.get("normalization_type") != "linear" or abs(lufs + 16) > 0.3:
        gain = -16.0 - float(m["input_i"])
        for _ in range(6):
            ff([*cut, "-af", f"{base},volume={gain:.2f}dB,{TP_LIMIT},{fades}", "-t", f"{d:.3f}", "-ac", "1", "-c:a", "pcm_f32le",
                "-y", str(dst)])
            lufs, tp = measure(dst)
            applied = gain
            if abs(lufs + 16) <= 0.2:
                break
            gain += -16.0 - lufs
        method = (f"loudnorm pass 1 measured {float(m['input_i']):.1f} LUFS; pass 2 linear refused (true peak), so linear gain "
                  f"{applied:+.1f} dB into the -1 dBTP limiter")
    x, _ = sf.read(dst, dtype="float32")
    return {"lufs": lufs, "true_peak_dbtp": tp, "method": method, "input_lufs": float(m["input_i"]),
            "input_tp": float(m["input_tp"]), "samples": len(x)}


# ------------------------------------------------------------------ whisper
_models = {}


def whisper(kind):
    if kind not in _models:
        from faster_whisper import WhisperModel
        _models[kind] = WhisperModel(EX.WHISPER_SMALL if kind == "small" else EX.WHISPER_TURBO, device="cpu",
                                     compute_type="int8", cpu_threads=2)
    return _models[kind]


def transcribe(kind, path, offset=0.0):
    segs, _ = whisper(kind).transcribe(str(path), language="en", beam_size=5, word_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=False)
    return [{"w": w.word.strip(), "s": round(w.start + offset, 3), "e": round(w.end + offset, 3)}
            for s in segs for w in (s.words or [])]


# ------------------------------------------------------------------ data
def load():
    pc, clips = EX.load()
    return pc, clips


def rec(pc, clips, name, fw, sy, ch="user"):
    return pc.get(f"{clips[name]['set']}|{ch}|{fw}|{sy}|{name}")


def words_ref(name, clips, cache):
    """Reference words (dicts w, n, s, e, spk, timed) in clip time."""
    if name.startswith("room_"):
        return [r for r in EX.ami_ref_words("IS1008b", ROOM_T0) if 0.0 <= r["s"] and r["e"] <= 60.0], "AMI word XML (speakers A-D)"
    if name.startswith("ami_"):
        c = clips[name]
        return EX.ami_ref_words(c["conversation"], c["start"]), "AMI word XML (all speakers)"
    turbo = [dict(h, n=EX.norm_tok(h["w"])) for h in cache["full"][name]["turbo"]]
    turbo = [h for h in turbo if h["n"]]
    if name.startswith("tb_"):
        ref = [r for r in EX.tb_ref_words(clips[name], "user") if r["spk"] == "user"]
        EX.align_tb(ref, turbo)
        return ref, "TurnBench annotator a (user speaker), word times aligned to Whisper large-v3-turbo"
    return ([dict(h, spk="user", timed=True) for h in turbo],
            "Whisper large-v3-turbo on the whole raw channel (pseudo-reference; otoSpeech has no transcript)")


def wer(ref, hyp, t0, t1):
    r = [x["n"] for x in ref if t0 <= (x["s"] + x["e"]) / 2 < t1]
    h = [EX.norm_tok(x["w"]) for x in hyp if t0 <= (x["s"] + x["e"]) / 2 < t1]
    h = [x for x in h if x]
    return EX.edit_distance(r, h) / max(1, len(r)), len(r)


def miss_rate(inw, subset, hyp, t0, t1):
    """Share of `subset` reference words not matched exactly when the whole in-span reference is aligned to the hypothesis."""
    import difflib
    rn = [r["n"] for r in inw]
    hn = [h for h in (EX.norm_tok(x["w"]) for x in hyp if t0 <= (x["s"] + x["e"]) / 2 < t1) if h]
    hit = set()
    for b in difflib.SequenceMatcher(None, rn, hn, autojunk=False).get_matching_blocks():
        hit |= set(range(b.a, b.a + b.size))
    ids = {id(r) for r in subset}
    idx = [i for i, r in enumerate(inw) if id(r) in ids]
    return sum(1 for i in idx if i not in hit) / max(1, len(idx)), len(idx)


# ------------------------------------------------------------------ candidates
INTERRUPT_CLIPS = ["tb_22", "tb_42", "tb_106", "tb_113", "tb_128", "tb_138", "oto_f619bed328134a054bbd860d0e753862",
                   "oto_b49a3e43a039db495da6d043c86ae60b", "oto_5bc1e19ecd43cc6d45d0c1b9b78be5aa",
                   "oto_5f9148c5dfff61031a7c9ef711020cad", "oto_629ad1f171"]


KEEP, ROOM_KEEP = 8, 16


def all_user_clips(clips):
    return [n for n, c in clips.items() if c["set"] in ("turnbench", "oto")]


def gaps(words, dur):
    """(gap_start, gap_end) between consecutive words (plus both clip ends)."""
    ws = sorted((w["s"], w["e"]) for w in words)
    out, prev = [], 0.0
    for s, e in ws:
        if s - prev >= 0.12:
            out.append((prev, s))
        prev = max(prev, e)
    out.append((prev, dur))
    return out


def starts_ends(words, dur):
    """Cut points at word boundaries only (never inside a word): a start sits just before a word, an end just after one,
    halfway into the pause when the pause is short."""
    ws = sorted((w["s"], w["e"]) for w in words)
    starts, ends = set(), set()
    prev_e = 0.0
    for i, (s, e) in enumerate(ws):
        if s >= prev_e - 0.01:
            starts.add(round(max(0.0, s - min(0.25, max(0.0, s - prev_e) / 2)), 3))
        nxt = ws[i + 1][0] if i + 1 < len(ws) else dur
        if nxt >= e - 0.01:
            ends.add(round(min(dur, e + min(0.3, max(0.0, nxt - e) / 2)), 3))
        if nxt - e > 0.6:  # inside a pause: any point on a 0.25 s grid
            for g in np.arange(e + 0.3, nxt - 0.29, 0.25):
                ends.add(round(float(g), 3)); starts.add(round(float(g), 3))
        prev_e = max(prev_e, e)
    return sorted(starts), sorted(ends)


def interrupt_cands(pc, clips, cache):
    out = []
    for n in all_user_clips(clips):
        A, C = rec(pc, clips, n, "pipecat", "A"), rec(pc, clips, n, "pipecat", "C")
        if not A or not C or len(A["cut_in_t"]) < 2 or n not in cache.get("full", {}):
            continue
        dur = clips[n]["dur"]
        st, en = starts_ends(cache["full"][n]["turbo"], dur)
        cresp = [e["response"] for e in C["ends"] if e.get("response") is not None]
        best = {}
        for t0 in st:
            for t1 in en:
                if not 6.0 <= t1 - t0 <= 10.0:
                    continue
                a_in = [c for c in A["cut_in_t"] if t0 + 0.8 <= c <= t1 - 0.6]
                a_all = [c for c in A["cut_in_t"] if t0 <= c <= t1]
                if len(a_in) < 2 or len(a_all) != len(a_in) or any(t0 <= c <= t1 for c in C["cut_in_t"]):
                    continue
                c_in = [r for r in cresp if max(a_in) < r <= t1 - 0.2 and r >= t0]
                score = 2.0 * bool(c_in) + 0.2 * min(len(a_in), 4) - 0.1 * (t1 - t0)
                key = (round(2 * t0) / 2, bool(c_in))
                if key not in best or score > best[key]["score"]:
                    best[key] = {"clip": n, "t0": t0, "t1": t1, "A_cuts": a_in, "C_resp": c_in, "score": round(score, 3)}
        out += sorted(best.values(), key=lambda x: -x["score"])[:KEEP]
    return out


E2E_TABLE = None


def table():
    global E2E_TABLE
    if E2E_TABLE is None:
        E2E_TABLE = json.loads((ROOT / "runs" / "e2e_final.json").read_text())["table"]
    return E2E_TABLE


def max_gap(words, t0, t1):
    """Longest stretch without a word between t0 and the last word before t1 (a long silence reads as a dead clip)."""
    ts = sorted((w["s"], w["e"]) for w in words if w["e"] > t0 and w["s"] < t1)
    prev, g = t0, 0.0
    for a, b in ts:
        g = max(g, a - prev)
        prev = max(prev, b)
    return g


RATE_MAX = 4.5      # cut-ins per minute inside the span: within ~2x of the default's 2.1-2.2 / min average


def typical_interrupt_cands(pc, clips, cache, lo=12.0, hi=20.0, kind="interrupt"):
    """A 12-20 s stretch where Pipecat's default cuts in 1-2 times at <= RATE_MAX / min, audioforge 0 times, and audioforge
    answers a turn that ends inside the span. call_today: 8-15 s with exactly one default cut-in (only the default shown)."""
    out = []
    for n in all_user_clips(clips):
        A, C = rec(pc, clips, n, "pipecat", "A"), rec(pc, clips, n, "pipecat", "C")
        if not A or not C or not A["cut_in_t"] or n not in cache.get("full", {}):
            continue
        words = cache["full"][n]["turbo"]
        st, en = starts_ends(words, clips[n]["dur"])
        best = {}
        for t0 in st:
            for t1 in en:
                L = t1 - t0
                if not lo <= L <= hi:
                    continue
                a_all = [c for c in A["cut_in_t"] if t0 <= c <= t1]
                a_in = [c for c in a_all if t0 + 1.5 <= c <= t1 - 1.0]
                if not a_all or len(a_in) != len(a_all):
                    continue
                rate = 60.0 * len(a_all) / L
                if kind == "interrupt":
                    if len(a_all) > 2 or rate > RATE_MAX or any(t0 <= c <= t1 for c in C["cut_in_t"]):
                        continue
                    ans = [e for e in C["ends"] if e.get("response") is not None and t0 + 2 <= e["end"] <= e["response"] <= t1 - 0.3]
                    if not ans:
                        continue
                    last = max(e["response"] for e in ans)
                    if max_gap(words, t0, last) > 3.0:
                        continue
                    score = -abs(rate - 2.2) - 0.05 * L
                    extra = {"C_resp": [e["response"] for e in ans], "C_turn_end": [e["end"] for e in ans]}
                else:
                    if len(a_all) != 1 or max_gap(words, t0, t1) > 2.5:
                        continue
                    score = -abs(rate - 2.2) - 0.05 * L
                    extra = {"C_resp": []}
                key = round(t0)
                if key not in best or score > best[key]["score"]:
                    best[key] = dict({"clip": n, "t0": t0, "t1": t1, "A_cuts": a_all, "rate_in_span": round(rate, 2),
                                      "score": round(score, 3)}, **extra)
        out += sorted(best.values(), key=lambda x: -x["score"])[:4]
    return out


def typical_words_cands(pc, clips, cache):
    """LiveKit default first text 2.5-4.5 s after onset (median on real calls 3.4 s), audioforge 0.8-1.6 s (median 1.2 s)."""
    out = []
    for n in all_user_clips(clips):
        B, C = rec(pc, clips, n, "livekit", "B"), rec(pc, clips, n, "livekit", "C")
        if not B or not C or n not in cache.get("full", {}):
            continue
        fb, fc = B.get("first_text_ms_after_onset"), C.get("first_text_ms_after_onset")
        if fb is None or fc is None or not (2500 <= fb <= 4500 and 600 <= fc <= 1600):
            continue
        on = clips[n].get("first_onset_user") or clips[n]["user_turns"][0][0]
        st, en = starts_ends(cache["full"][n]["turbo"], clips[n]["dur"])
        t0 = max([s_ for s_ in st if s_ <= on + 0.05] or [max(0.0, on - 0.25)])
        for t1 in en:
            if t1 < on + fb / 1000 + 0.8 or not 6.0 <= t1 - t0 <= 10.0 or any(t0 <= c <= t1 for c in C["cut_in_t"]):
                continue
            out.append({"clip": n, "t0": t0, "t1": t1, "onset": on, "B_first": round(on + fb / 1000, 3), "C_first": round(on + fc / 1000, 3),
                        "lead_s": round((fb - fc) / 1000, 3), "score": -abs((fb - fc) / 1000 - 2.2) - 0.02 * (t1 - t0)})
            if sum(1 for o in out if o["clip"] == n) >= 4:
                break
    return out


def words_cands(pc, clips, cache):
    out = []
    for n in all_user_clips(clips):
        B, C = rec(pc, clips, n, "livekit", "B"), rec(pc, clips, n, "livekit", "C")
        if not B or not C or n not in cache.get("full", {}):
            continue
        fb, fc = B.get("first_text_ms_after_onset"), C.get("first_text_ms_after_onset")
        if fb is None or fc is None or fb - fc < 2000:
            continue
        on = clips[n].get("first_onset_user") or clips[n]["user_turns"][0][0]
        tb_abs, tc_abs = on + fb / 1000, on + fc / 1000
        dur = clips[n]["dur"]
        st, en = starts_ends(cache["full"][n]["turbo"], dur)
        t0c = [s for s in st if s <= on + 0.05]
        if not t0c:
            t0c = [max(0.0, on - 0.25)]
        t0 = max(t0c)
        for t1 in en:
            if t1 < tb_abs + 0.6 or not 6.0 <= t1 - t0 <= 10.0:
                continue
            if any(t0 <= c <= t1 for c in C["cut_in_t"]):
                continue
            out.append({"clip": n, "t0": t0, "t1": t1, "onset": on, "B_first": round(tb_abs, 3), "C_first": round(tc_abs, 3),
                        "lead_s": round((fb - fc) / 1000, 3), "score": round((fb - fc) / 1000 - 0.1 * (t1 - t0), 3)})
            break  # the shortest span that shows the default's first words
    return out


def room_cands(cache):
    ref, _ = words_ref("room_IS1008b_1670s", None, cache)
    st, en = starts_ends(ref, 60.0)
    out = []
    for t0 in st:
        for t1 in en:
            if not 6.0 <= t1 - t0 <= 10.0:
                continue
            spk = {r["spk"] for r in ref if t0 <= (r["s"] + r["e"]) / 2 < t1}
            if len(spk) < 3:
                continue
            # overlapped speech time (two or more speakers at once) hurts intelligibility
            grid = np.arange(t0, t1, 0.02)
            cnt = np.zeros(len(grid))
            for s_ in spk:
                act = np.zeros(len(grid), bool)
                for r in ref:
                    if r["spk"] == s_:
                        act |= (grid >= r["s"]) & (grid < r["e"])
                cnt += act
            ov = float((cnt >= 2).mean())
            out.append({"clip": "room_IS1008b_1670s", "t0": t0, "t1": t1, "speakers": sorted(spk), "overlap_frac": round(ov, 3),
                        "score": round(len(spk) - 3 * ov - 0.05 * (t1 - t0), 3)})
    out.sort(key=lambda x: -x["score"])
    res = []
    for x in out:
        if all(abs(x["t0"] - y["t0"]) > 0.6 or abs(x["t1"] - y["t1"]) > 0.6 for y in res):
            res.append(x)
    return res[:ROOM_KEEP]


# ------------------------------------------------------------------ stages
def cmd_transcribe(a):
    pc, clips = load()
    cache = load_cache()
    full = cache.setdefault("full", {})
    t_start = time.time()
    for kind in ("turbo",):
        for n in all_user_clips(clips):
            if kind in full.get(n, {}):
                continue
            if time.time() - t_start > a.budget:
                save_cache(cache); print("budget reached; run again"); sys.exit(3)
            full.setdefault(n, {})[kind] = transcribe(kind, src_of(n))
            print(kind, n, len(full[n][kind]), f"{time.time() - t_start:.0f}s", flush=True)
            save_cache(cache)
    print("transcribe done")


def cand_list(pc, clips, cache):
    """v2 (typical spans, after 'the demos look rigged'): interrupt / words / call_today at typical rates; room unchanged."""
    c = []
    for x in typical_interrupt_cands(pc, clips, cache):
        c.append(dict(x, kind="interrupt"))
    for x in typical_interrupt_cands(pc, clips, cache, 10.0, 15.0, "call_today"):
        c.append(dict(x, kind="call_today"))
    for x in typical_words_cands(pc, clips, cache):
        c.append(dict(x, kind="words"))
    for x in room_cands(cache):
        c.append(dict(x, kind="room"))
    for x in c:
        x["key"] = f"{x['clip']}.{x['t0']:.2f}-{x['t1']:.2f}"
    return c


def cmd_score(a):
    pc, clips = load()
    cache = load_cache()
    spans = cache.setdefault("spans", {})
    cands = cand_list(pc, clips, cache)
    print(len(cands), "candidates", flush=True)
    t_start = time.time()
    for x in cands:
        k = x["key"]
        e = spans.setdefault(k, {})
        if "after" in e and "before" in e:
            continue
        if time.time() - t_start > a.budget:
            save_cache(cache); print("budget reached; run again"); sys.exit(3)
        src = src_of(x["clip"])
        raw, proc = WORK / f"{k}.raw.wav", WORK / f"{k}.proc.wav"
        cut_raw(src, x["t0"], x["t1"], raw)
        e["proc"] = process(src, x["t0"], x["t1"], proc)
        e["before"] = transcribe("small", raw, x["t0"])
        e["after"] = transcribe("small", proc, x["t0"])
        e["raw_lufs"] = measure(raw)[0]
        print(k, x["kind"], f"{time.time() - t_start:.0f}s", flush=True)
        save_cache(cache)
    print("score done")


# ------------------------------------------------------------------ agent voice placement
def agent_lines():
    """Full reply, the short reply, and 'Sure,' = the full reply ended at its first word boundary (never mid-word)."""
    j = json.loads(AGENT_JSON.read_text())
    full, short = j["units"][0], j["units"][1]
    x, sr = sf.read(full["wav"], dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    hop = int(0.01 * sr)
    r = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop] + 1e-12)
    rdb = 20 * np.log10(r / r.max())
    on = int(np.argmax(rdb > -30))
    # first pause (>= 40 ms under -35 dB) after 150 ms of speech: the comma after "Sure"
    cut = None
    for i in range(on + 15, len(rdb) - 4):
        if (rdb[i: i + 4] < -35).all():
            cut = i
            break
    sure_dur = round((cut or on + 30) * 0.01 + 0.03, 3)
    lines = {"full": {"text": full.get("caption") or full["text"], "wav": full["wav"], "dur": round(full["end"] - full["start"], 3), "trim": None},
             "short": {"text": short.get("caption") or short["text"], "wav": short["wav"], "dur": round(short["end"] - short["start"], 3), "trim": None},
             "sure": {"text": "Sure,", "wav": full["wav"], "dur": sure_dur, "trim": sure_dur}}
    return lines


def place_agent(events, t1, lines):
    """events: sorted [(t, kind)]. Longest line that ends 0.1 s before the next event; none if even 'Sure,' does not fit."""
    out = []
    for i, (t, kind) in enumerate(events):
        nxt = events[i + 1][0] if i + 1 < len(events) else None
        room = (nxt - t - 0.1) if nxt is not None else 99.0
        pick = next((k for k in ("full", "short", "sure") if lines[k]["dur"] <= room), None)
        if kind == "cut_in" and pick == "full" and nxt is None:
            pick = "full"
        ent = {"t": round(t, 3), "event": kind, "line": pick}
        if pick:
            L = lines[pick]
            ent.update(text=L["text"], dur=L["dur"], end=round(t + L["dur"], 3), wav=L["wav"], trim_s=L["trim"],
                       past_span_end=round(max(0.0, t + L["dur"] - t1), 3), gain_db_rel_user=-3.0)
        else:
            ent.update(text="", dur=0.0, reason=f"next event {nxt - t:.2f} s later: no line fits without cutting a word")
        out.append(ent)
    return out


# ------------------------------------------------------------------ choose
KIND_SYS = {"interrupt": {"L": ("pipecat", "A"), "R": ("pipecat", "C")}, "phone": {"L": ("pipecat", "A"), "R": ("pipecat", "C")},
            "call_today": {"P": ("pipecat", "A")},
            "words": {"L": ("livekit", "B"), "R": ("livekit", "C")}}
LABEL = {"pipecat/A": "Pipecat default", "pipecat/C": "audioforge (Pipecat)", "livekit/B": "LiveKit default", "livekit/C": "audioforge (LiveKit)"}


def room_diar(t0, t1):
    d = json.loads((V5 / "data.json").read_text())["clips"]["room_IS1008b_1670s"]
    cols = set()
    for fr in d["frames"]:
        if t0 <= fr[0] < t1:
            cols |= {i for i, p in enumerate(fr[1]) if p >= 0.5}
    return sorted(cols)


def cmd_choose(a):
    pc, clips = load()
    cache = load_cache()
    cands = cand_list(pc, clips, cache)
    lines = agent_lines()
    rows = []
    for x in cands:
        e = cache.get("spans", {}).get(x["key"])
        if not e or "after" not in e:
            continue
        ref, rsrc = words_ref(x["clip"], clips, cache)
        x["wer_after"], x["n_ref"] = wer(ref, e["after"], x["t0"], x["t1"])
        x["wer_before"], _ = wer(ref, e["before"], x["t0"], x["t1"])
        x["lufs"], x["tp"], x["raw_lufs"], x["method"] = e["proc"]["lufs"], e["proc"]["true_peak_dbtp"], e["raw_lufs"], e["proc"]["method"]
        x["ref_source"] = rsrc
        x["ref_text"] = " ".join(r["w"] for r in ref if x["t0"] <= (r["s"] + r["e"]) / 2 < x["t1"])
        x["hyp_text"] = " ".join(h["w"] for h in e["after"] if x["t0"] <= (h["s"] + h["e"]) / 2 < x["t1"])
        x["wps"] = x["n_ref"] / (x["t1"] - x["t0"])
        if x["kind"] == "room":  # words not overlapped by another speaker's word (backchannels under the main voice)
            inw = [r for r in ref if x["t0"] <= (r["s"] + r["e"]) / 2 < x["t1"]]
            solo = [r for r in inw if not any(o["spk"] != r["spk"] and o["s"] < r["e"] and o["e"] > r["s"] for o in inw)]
            x["wer_solo_after"], x["n_solo"] = miss_rate(inw, solo, e["after"], x["t0"], x["t1"])
            x["wer_solo_before"], _ = miss_rate(inw, solo, e["before"], x["t0"], x["t1"])
        rows.append(x)
    overrides = json.loads(a.pick) if a.pick else {}
    chosen = {}
    for kind in ("interrupt", "words", "call_today", "room"):
        pool = [x for x in rows if x["kind"] == kind]
        if kind in overrides:
            chosen[kind] = next(x for x in rows if x["key"] == overrides[kind])
            continue
        used = {c["clip"] for c in chosen.values()}
        ok = [x for x in pool if x["wer_after"] <= 0.15 and x["n_ref"] >= 8 and x["wps"] >= 1.2 and x["clip"] not in used]
        chosen[kind] = max(ok, key=lambda x: (x["score"], -x["wer_after"])) if ok else min(pool, key=lambda x: x["wer_after"])
    manifest, subs = {"generated_by": "demo/archive/v5/audio/clips.py", "processing": {
        "chain": CHAIN, "loudness": "two-pass loudnorm to -16 LUFS integrated, linear", "true_peak": TP_LIMIT + " (-1 dBTP)",
        "fades_ms": 150, "sr": SR, "file_span": "each WAV is exactly [t0, t1] of its source (offset = t0)"},
        "agent_voice": {k: {kk: v[kk] for kk in ("text", "dur", "wav", "trim")} for k, v in lines.items()},
        "agent_voice_rule": "agent line sits 3 dB below the clip's loudness (-19 LUFS for a -16 LUFS clip); the longest line "
                            "that ends 0.1 s before the next event; 'Sure,' = the full reply ended at its first word boundary "
                            "(trim_s, then a 30 ms fade); no line when even that does not fit",
        "clips": {}}, {"generated_by": "demo/archive/v5/audio/clips.py", "time_base": "clip time (same as t0/t1)", "clips": {}}
    for kind, x in chosen.items():
        cid = f"ex_{kind}"
        src = src_of(x["clip"])
        dst = CLIPS / f"{cid}.wav"
        proc = process(src, x["t0"], x["t1"], dst)
        ref, rsrc = words_ref(x["clip"], clips, cache)
        sub = [[round(r["s"], 3), round(r["e"], 3), r["w"], r["spk"]] for r in ref if x["t0"] <= (r["s"] + r["e"]) / 2 < x["t1"]]
        if x["clip"].startswith("oto_"):
            # no reference transcript: Whisper turbo word times on the processed clip
            hyp = transcribe("turbo", dst, x["t0"])
            sub = [[h["s"], h["e"], h["w"], "user"] for h in hyp if x["t0"] <= (h["s"] + h["e"]) / 2 < x["t1"]]
            sub_src = "Whisper large-v3-turbo word timestamps on the processed clip (otoSpeech has no transcript)"
        else:
            sub_src = rsrc
        subs["clips"][cid] = {"source_clip": x["clip"], "t0": x["t0"], "t1": x["t1"], "source": sub_src,
                              "words": sub, "speakers": sorted({w[3] for w in sub})}
        ent = {"clip_id": cid, "kind": kind, "source_clip": x["clip"],
               "source": {"interrupt": "TurnBench dev (Mundo AI, evaluation licence) user channel" if x["clip"].startswith("tb_") else "otoSpeech (CC-BY-4.0) user channel",
                          "call_today": "TurnBench dev (Mundo AI, evaluation licence) user channel" if x["clip"].startswith("tb_") else "otoSpeech (CC-BY-4.0) user channel",
                          "phone": "otoSpeech full-duplex (CC-BY-4.0), the caller's own channel",
                          "words": "TurnBench dev user channel" if x["clip"].startswith("tb_") else "otoSpeech (CC-BY-4.0) user channel",
                          "room": "AMI IS1008b (CC-BY-4.0), mixed headset audio, meeting time 1670 s + clip time"}[kind],
               "source_wav": str(src), "wav": str(dst), "offset": x["t0"],
               "span": [x["t0"], x["t1"]], "dur_s": round(x["t1"] - x["t0"], 3),
               "wer_before": round(x["wer_before"], 4), "wer_after": round(x["wer_after"], 4), "wer_ref_words": x["n_ref"],
               "ref_source": x["ref_source"], "ref_text": x["ref_text"], "whisper_small_after": x["hyp_text"],
               "raw_lufs": x["raw_lufs"], "lufs": proc["lufs"], "true_peak_dbtp": proc["true_peak_dbtp"], "loudness_method": proc["method"],
               "subs_ref": f"subs.json#clips.{cid}", "speaker": "speaker", "systems": {}, "agent_voice": {}}
        if kind == "room":
            ent["wer_nonoverlapped_before"] = round(x["wer_solo_before"], 4)
            ent["wer_nonoverlapped_after"] = round(x["wer_solo_after"], 4)
            ent["wer_nonoverlapped_ref_words"] = x["n_solo"]
            ent["speakers_in_span"] = x["speakers"]
            ent["overlap_frac"] = x["overlap_frac"]
            ent["diarizer_columns_active_in_span"] = room_diar(x["t0"], x["t1"])
            ent["meeting_time"] = [ROOM_T0 + x["t0"], ROOM_T0 + x["t1"]]
        else:
            for side, (fw, sy) in KIND_SYS[kind].items():
                r = rec(pc, clips, x["clip"], fw, sy)
                cuts = [c for c in r["cut_in_t"] if x["t0"] <= c <= x["t1"]]
                resp = [q["response"] for q in r["ends"] if q.get("response") is not None and x["t0"] <= q["response"] <= x["t1"]]
                on = clips[x["clip"]].get("first_onset_user") or clips[x["clip"]]["user_turns"][0][0]
                ft = r.get("first_text_ms_after_onset")
                sysd = {"system": f"{fw}/{sy}", "label": LABEL[f"{fw}/{sy}"],
                        "record": f"runs/e2e_final.json per_clip['{clips[x['clip']]['set']}|user|{fw}|{sy}|{x['clip']}']",
                        "cut_ins": cuts, "responses": resp, "first_text_ms_after_onset": ft,
                        "first_text": round(on + ft / 1000, 3) if ft is not None else None, "onset": on}
                ent["systems"][side] = sysd
                if kind in ("interrupt", "phone", "call_today"):
                    evs = sorted([(c, "cut_in") for c in cuts] + [(q, "response") for q in resp])
                    ent["agent_voice"][side] = place_agent(evs, x["t1"], lines)
            T = table()
            st_ = clips[x["clip"]]["set"]
            L_ = x["t1"] - x["t0"]
            if kind in ("interrupt", "call_today"):
                tA, tC = T[f"{st_}|user|pipecat|A"], T[f"{st_}|user|pipecat|C"]
                ent["typicality"] = {
                    "default_cut_ins_in_span": len(ent["systems"]["P" if kind == "call_today" else "L"]["cut_ins"]),
                    "default_rate_in_span_per_min": round(60 * len(ent["systems"]["P" if kind == "call_today" else "L"]["cut_ins"]) / L_, 2),
                    "default_avg_per_min": tA["cut_ins_per_min"], "default_avg_scope": f"runs/e2e_final.json table['{st_}|user|pipecat|A'] (16 calls)",
                    "ours_rate_in_span_per_min": (round(60 * len(ent["systems"]["R"]["cut_ins"]) / L_, 2) if "R" in ent["systems"] else None),
                    "ours_avg_per_min": tC["cut_ins_per_min"], "ours_avg_scope": f"runs/e2e_final.json table['{st_}|user|pipecat|C']",
                    "pooled_default_avg_per_min": round((T["oto|user|pipecat|A"]["cut_ins"] + T["turnbench|user|pipecat|A"]["cut_ins"]) /
                                                        ((T["oto|user|pipecat|A"]["cut_ins"] / T["oto|user|pipecat|A"]["cut_ins_per_min"]) +
                                                         (T["turnbench|user|pipecat|A"]["cut_ins"] / T["turnbench|user|pipecat|A"]["cut_ins_per_min"])), 3),
                    "note": "the span is chosen so the default's in-span rate is within ~2x of its measured average; audioforge also cuts in "
                            "on real calls (average above), just not in this span"}
            if kind == "words":
                tB, tC = T[f"{st_}|user|livekit|B"], T[f"{st_}|user|livekit|C"]
                ent["typicality"] = {
                    "default_first_text_s": round(ent["systems"]["L"]["first_text_ms_after_onset"] / 1000, 2),
                    "ours_first_text_s": round(ent["systems"]["R"]["first_text_ms_after_onset"] / 1000, 2),
                    "default_median_s": tB["first_text_ms_median"] / 1000, "ours_median_s": tC["first_text_ms_median"] / 1000,
                    "median_scope": f"runs/e2e_final.json table['{st_}|user|livekit|B/C'] first_text_ms_median (16 calls)",
                    "median_oto": [T["oto|user|livekit|B"]["first_text_ms_median"] / 1000, T["oto|user|livekit|C"]["first_text_ms_median"] / 1000],
                    "median_turnbench": [T["turnbench|user|livekit|B"]["first_text_ms_median"] / 1000, T["turnbench|user|livekit|C"]["first_text_ms_median"] / 1000]}
                ent["first_words_lead_s"] = round(ent["systems"]["L"]["first_text"] - ent["systems"]["R"]["first_text"], 3)
        manifest["clips"][cid] = ent
    (V5 / "clips_manifest.json").write_text(json.dumps(manifest, indent=1))
    (V5 / "subs.json").write_text(json.dumps(subs, indent=1))
    # table
    T = ["| example | clip | span (s) | dur | events in span | ref words | WER before | WER after | LUFS raw -> processed | true peak | chosen |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    ch_keys = {x["key"]: k for k, x in chosen.items()}
    for x in sorted(rows, key=lambda x: (x["kind"], x["wer_after"])):
        if x["kind"] in ("interrupt", "phone", "call_today"):
            ev = (f"default cuts in {len(x['A_cuts'])}x ({x.get('rate_in_span', 0):.1f}/min in span)"
                  + (", ours 0" if x["kind"] != "call_today" else "") + (f", ours answers {x['C_resp'][-1]:.2f}" if x["C_resp"] else ""))
        elif x["kind"] == "words":
            ev = f"first words: default {x['B_first']:.2f}, ours {x['C_first']:.2f} (lead {x['lead_s']:.1f} s)"
        else:
            ev = (f"speakers {''.join(x['speakers'])}, overlap {100 * x['overlap_frac']:.0f} %; non-overlapped words missed "
                  f"{100 * x['wer_solo_before']:.0f} -> {100 * x['wer_solo_after']:.0f} % (n={x['n_solo']})")
        T.append(f"| {x['kind']} | {x['clip'][:16]} | {x['t0']:.2f}-{x['t1']:.2f} | {x['t1'] - x['t0']:.1f} | {ev} | {x['n_ref']} | "
                 f"{100 * x['wer_before']:.1f} % | {100 * x['wer_after']:.1f} % | {x['raw_lufs']:.1f} -> {x['lufs']:.1f} | "
                 f"{x['tp'] if x['tp'] is not None else float('nan'):.1f} | {'**' + ch_keys[x['key']] + '**' if x['key'] in ch_keys else ''} |")
    (CLIPS / "intelligibility.md").write_text("\n".join(T) + "\n")
    print("\n".join(T))
    for kind, x in chosen.items():
        print(f"\n{kind}: {x['key']}  WER {x['wer_before']:.3f} -> {x['wer_after']:.3f}\n  ref: {x['ref_text']}\n  hyp: {x['hyp_text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["transcribe", "score", "choose", "cands"])
    ap.add_argument("--budget", type=float, default=480)
    ap.add_argument("--pick", default="", help='choose: JSON {kind: span key} overrides')
    a = ap.parse_args()
    if a.cmd == "cands":
        pc, clips = load()
        for x in cand_list(pc, clips, load_cache()):
            print(x["kind"], x["key"], {k: v for k, v in x.items() if k not in ("kind", "key", "clip", "t0", "t1")})
        return
    {"transcribe": cmd_transcribe, "score": cmd_score, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    main()
