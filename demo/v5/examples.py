"""v5 real-example picker: chooses the example windows of the film by their recorded events, then by intelligibility.

    scripts/dev/gate.sh env PYTHONPATH=. .venv/bin/python demo/v5/examples.py
      -> $DEMO_OUT/v5_data/examples.json, $DEMO_OUT/v5_data/ex_audio/<example>.wav, $DEMO_OUT/v5_data/examples_wer.md

Examples (every time is the stored per_clip timeline of runs/e2e_final.json, i.e. clip time = the clip WAV's time):
  interrupt  two-party, user channel: 2-4 Pipecat-default (pipecat/A) cut-ins in the window, 0 for ours (pipecat/C)
  words      LiveKit default (livekit/B) first text > 3 s after the user's onset, ours (livekit/C) < 1.5 s and no
             livekit/C cut-in in the window; window = [onset, B first text + 0.3 s], at most 8 s
  phone      like interrupt, another clip (otoSpeech phone call preferred, scored against a Whisper large-v3-turbo
             pseudo-reference because otoSpeech has no transcript)
  call_today 4-6 s with >= 1 pipecat/A cut-in (opening shot, only A shown)
  room       AMI IS1008b [1670.2, 1675.4] s meeting time, all four speakers from the AMI word XMLs
Intelligibility: each candidate window is cut with 0.25 s context, processed (highpass 80 Hz, +3 dB presence at 3 kHz,
3:1 compressor, two-pass linear loudnorm to -16 LUFS / -1 dBTP, limiter 0.89), transcribed by Whisper small (int8,
CPU, 2 threads, en, beam 5, word timestamps) and scored by WER against the reference words inside the window
(lowercase, punctuation stripped, bracketed tags, fillers and cut-off word fragments dropped).
TurnBench references are turn-level segments; their words are timed by aligning them to Whisper's word timestamps
(unmatched words interpolated between matched neighbours of the same segment).
"""
from __future__ import annotations

import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEMO_OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
OUT = DEMO_OUT / "v5_data"
EX_AUDIO = OUT / "ex_audio"
CAND_AUDIO = OUT / "ex_cand_audio"
CACHE = OUT / "ex_cand_cache_v2.json"
CLIPS_JSON = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad/clips.json")
CLIP_DIR = CLIPS_JSON.parent / "clips"
ROOM_WAV = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/diar/clips/ami_IS1008b_1670s.wav")
ROOM_WAV_T0, ROOM_WIN = 1670.0, (1670.2, 1675.4)
AMI_WORDS = ROOT / "data" / "ami" / "annotations" / "words"
HF = Path.home() / ".cache" / "huggingface" / "hub"
WHISPER_SMALL = sorted(glob.glob(str(HF / "models--Systran--faster-whisper-small" / "snapshots" / "*")))[-1]
WHISPER_TURBO = sorted(glob.glob(str(HF / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots" / "*")))[-1]
PAD = 0.25
SR_OUT = 48000
FILLERS = {"um", "uh", "uhm", "umm", "uhh", "mm", "mmm", "hmm", "hm", "mhm", "mmhmm", "uhhuh"}
CHAIN = "highpass=f=80,equalizer=f=3000:t=q:w=1.2:g=3,acompressor=threshold=-24dB:ratio=3:attack=10:release=150"


# ============================================================================ text
def norm_tok(w: str) -> str:
    w = w.lower().replace("-", "")
    w = re.sub(r"[^a-z0-9']", "", w).strip("'")
    return "" if w in FILLERS else w


def clean_ref_text(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"\b(\w{1,4})-(?=\1)", "", t, flags=re.I)          # stutter prefix: f-for -> for
    t = re.sub(r"(?<![\w-])\w+-(?=[\s,.?!]|$)", " ", t)              # cut-off fragment: oth-, I-
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(r: list, h: list) -> int:
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


# ============================================================================ data
def load():
    pc = json.loads((ROOT / "runs" / "e2e_final.json").read_text())["per_clip"]
    clips = {c["name"]: c for c in json.loads(CLIPS_JSON.read_text())}
    return pc, clips


def rec(pc, clip, ch, fw, sy):
    return pc.get(f"{clip['set']}|{ch}|{fw}|{sy}|{clip['name']}")


def onset_of(clip, ch):
    o = clip.get(f"first_onset_{ch}")
    return (o, f"first_onset_{ch}") if o is not None else (clip["user_turns"][0][0], "user_turns[0][0]")


def tb_ref_words(clip, ch) -> list[dict]:
    """TurnBench annotator-a segments -> words with char-interpolated times (clip time), per speaker label."""
    from audioforge.datasets import dyadic as D
    cid, h = clip["conversation"], clip["human_channel"]
    spk = [h + 1] if ch == "user" else [1, 2]
    row = D.tb_row(D.TB_ROOT, cid, ["conversation_id"] + [f"speaker_{k}_annotation_a" for k in (1, 2)])
    out = []
    for k in spk:
        for si, e in enumerate(row[f"speaker_{k}_annotation_a"]):
            a, b = e["start_s"] - clip["start"], e["end_s"] - clip["start"]
            if b < -1 or a > clip["dur"] + 1:
                continue
            toks = [t for t in clean_ref_text(e["text"] or "").split() if norm_tok(t)]
            n = sum(len(t) + 1 for t in toks)
            c = 0
            for t in toks:
                s = a + (b - a) * c / n
                c += len(t) + 1
                out.append({"w": t, "n": norm_tok(t), "s": s, "e": a + (b - a) * c / n,
                            "spk": "user" if k == h + 1 else "agent", "seg": (k, si), "timed": False})
    return sorted(out, key=lambda x: (x["seg"][0], x["s"]))


def ami_ref_words(meeting: str, offset: float) -> list[dict]:
    """AMI word XMLs (all speakers) -> words in clip time (meeting time - offset)."""
    out = []
    for f in sorted(AMI_WORDS.glob(f"{meeting}.*.words.xml")):
        letter = f.name.split(".")[1]
        for w in ET.parse(f).getroot():
            if not w.tag.endswith("w") or w.get("punc") == "true" or w.get("starttime") is None:
                continue
            t = (w.text or "").strip()
            if not norm_tok(t):
                continue
            out.append({"w": t, "n": norm_tok(t), "s": float(w.get("starttime")) - offset,
                        "e": float(w.get("endtime")) - offset, "spk": letter, "seg": None, "timed": True})
    return sorted(out, key=lambda x: x["s"])


def align_tb(ref: list[dict], hyp: list[dict]) -> None:
    """Give TurnBench reference words Whisper's times where the tokens match; interpolate the rest per segment."""
    rn = [r["n"] for r in ref]
    hn = [h["n"] for h in hyp]
    for blk in difflib.SequenceMatcher(None, rn, hn, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            r, h = ref[blk.a + k], hyp[blk.b + k]
            r["s"], r["e"], r["timed"] = h["s"], h["e"], True
    i = 0
    while i < len(ref):
        if ref[i]["timed"]:
            i += 1
            continue
        j = i
        while j < len(ref) and not ref[j]["timed"] and ref[j]["seg"] == ref[i]["seg"]:
            j += 1
        prv = ref[i - 1] if i > 0 and ref[i - 1]["timed"] and ref[i - 1]["seg"] == ref[i]["seg"] else None
        nxt = ref[j] if j < len(ref) and ref[j]["timed"] and ref[j]["seg"] == ref[i]["seg"] else None
        run = ref[i:j]
        if prv and nxt:
            a, b = prv["e"], max(nxt["s"], prv["e"])
            n = sum(len(r["w"]) + 1 for r in run)
            c = 0
            for r in run:
                r["s"] = a + (b - a) * c / n
                c += len(r["w"]) + 1
                r["e"] = a + (b - a) * c / n
        elif prv:
            for k, r in enumerate(run):
                r["s"], r["e"] = prv["e"] + 0.3 * k, prv["e"] + 0.3 * (k + 1)
        elif nxt:
            for k, r in enumerate(run):
                r["s"], r["e"] = nxt["s"] - 0.3 * (len(run) - k), nxt["s"] - 0.3 * (len(run) - k - 1)
        i = j


def in_win(w, t0, t1):
    return t0 <= (w["s"] + w["e"]) / 2 < t1


# ============================================================================ audio
def ff(args: list[str]) -> str:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-threads", "2", *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return p.stderr


def process(src: Path, t0: float, t1: float, dst: Path, src_dur: float) -> dict:
    """Cut [t0-PAD, t1+PAD] (silence where the source has none), clarity chain, two-pass linear loudnorm, limiter."""
    a = t0 - PAD
    lead = max(0.0, -a)
    ss = max(0.0, a)
    total = t1 - t0 + 2 * PAD
    pre = (f"adelay={lead * 1000:.0f}:all=1," if lead > 0 else "") + f"apad=whole_dur={total:.3f},"
    cut = ["-ss", f"{ss:.3f}", "-t", f"{total - lead:.3f}", "-i", str(src)]
    base = pre + CHAIN
    err = ff([*cut, "-af", base + ",loudnorm=I=-16:TP=-1:LRA=11:print_format=json", "-f", "null", "-"])
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    ln = (f"loudnorm=I=-16:TP=-1:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          f"linear=true:print_format=json")
    err = ff([*cut, "-af", f"{base},{ln},alimiter=limit=0.89:level=false,aresample={SR_OUT}", "-t", f"{total:.3f}",
              "-ac", "1", "-c:a", "pcm_f32le", "-y", str(dst)])
    m2 = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    method = "loudnorm two-pass linear + alimiter 0.89"
    lufs, tp = measure(dst)
    if m2.get("normalization_type") != "linear":
        # loudnorm refuses linear mode when the linear gain would push the true peak over -1 dBTP and silently
        # switches to its dynamic (AGC) mode; keep the normalisation linear instead: the measured linear gain as a
        # plain volume step, the peaks caught by the limiter (0.89 = -1 dBFS), one correction step on the result
        gain = -16.0 - float(m["input_i"])
        for _ in range(2):
            ff([*cut, "-af", f"{base},volume={gain:.2f}dB,alimiter=limit=0.89:level=false,aresample={SR_OUT}",
                "-t", f"{total:.3f}", "-ac", "1", "-c:a", "pcm_f32le", "-y", str(dst)])
            lufs, tp = measure(dst)
            gain += -16.0 - lufs
        method = (f"loudnorm pass-1 measure; linear gain {gain - (-16.0 - lufs):+.1f} dB + alimiter 0.89 "
                  f"(loudnorm linear refused: true peak)")
    return {"lufs": lufs, "true_peak_dbfs": tp, "loudnorm_pass2_type": m2.get("normalization_type"),
            "method": method, "pass1_input_i": float(m["input_i"]), "pass1_input_tp": float(m["input_tp"])}


def measure(p: Path) -> tuple[float, float | None]:
    err = ff(["-i", str(p), "-af", "ebur128=peak=true", "-f", "null", "-"])
    summ = err[err.rindex("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
    tp = re.search(r"Peak:\s+(-?[\d.inf]+) dBFS", summ)
    return lufs, (float(tp.group(1)) if tp else None)


def src_dur(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


# ============================================================================ candidates
def cuts_in(r, t0, t1):
    return [t for t in (r or {}).get("cut_in_t", []) if t0 <= t <= t1]


def responses(r):
    return [{"end": e.get("end"), "response": e.get("response")} for e in (r or {}).get("ends", [])]


def turn_of(clip, t):
    return next((u for u in clip["user_turns"] if u[0] - 0.3 <= t <= u[1] + 0.3), None)


def interrupt_cands(pc, clips) -> list[dict]:
    out = []
    for c in clips.values():
        if c["set"] not in ("turnbench", "oto"):
            continue
        A, C = rec(pc, c, "user", "pipecat", "A"), rec(pc, c, "user", "pipecat", "C")
        if not A or not C:
            continue
        cuts = sorted(A["cut_in_t"])
        best = {}
        for i in range(len(cuts)):
            for j in range(i + 1, min(i + 4, len(cuts))):
                if cuts[j] - cuts[i] > 6.5:
                    continue
                tu = turn_of(c, cuts[i])
                t0 = max(0.0, cuts[i] - 1.5)
                if tu and cuts[i] - 2.5 <= tu[0] <= cuts[i]:
                    t0 = max(0.0, tu[0] - 0.2)
                t1 = cuts[j] + 1.0
                tj = turn_of(c, cuts[j])
                resp = None
                if tj:
                    resp = next((e["response"] for e in C["ends"] if e.get("response") and e["response"] >= tj[1] - 0.05
                                 and e["response"] > cuts[j]), None)
                if resp and resp + 0.5 - t0 <= 8.0:
                    t1 = max(t1, resp + 0.5)
                elif resp and resp + 0.5 - cuts[i] + 0.3 <= 8.0:
                    t1 = resp + 0.5
                    t0 = t1 - 8.0
                t1 = min(max(t1, t0 + 4.0), c["dur"])
                if t1 - t0 > 8.0:
                    continue
                na, nc = len(cuts_in(A, t0, t1)), len(cuts_in(C, t0, t1))
                if not 2 <= na <= 4 or nc:
                    continue
                inside = bool(resp and t0 <= resp <= t1)
                cand = {"kind": "interrupt", "clip": c["name"], "channel": "user", "t0": round(t0, 3),
                        "t1": round(t1, 3), "n_cuts_A": na, "C_resp": resp, "C_resp_inside": inside,
                        "event_score": 2 * inside + (1 if resp and resp - t1 < 1.5 else 0) + 0.1 * na}
                key = round(t0, 1)
                if key not in best or cand["event_score"] > best[key]["event_score"]:
                    best[key] = cand
        out += sorted(best.values(), key=lambda x: -x["event_score"])[:2]
    return out


def words_cands(pc, clips) -> list[dict]:
    out = []
    for c in clips.values():
        for ch in ("user", "mono"):
            if ch == "mono" and c["set"] != "ami":
                continue  # two-party: user channel only (clean reference alignment)
            B, C = rec(pc, c, ch, "livekit", "B"), rec(pc, c, ch, "livekit", "C")
            if not B or not C or B.get("first_text_ms_after_onset") is None or C.get("first_text_ms_after_onset") is None:
                continue
            fb, fc = B["first_text_ms_after_onset"] / 1000, C["first_text_ms_after_onset"] / 1000
            on, _ = onset_of(c, ch)
            if fb < 3.0 or fc >= 1.5 or fb + 0.3 > 8.3:
                continue
            t1 = on + fb + 0.3
            t0 = min(on, t1 - 4.0)
            if cuts_in(C, t0, t1):
                continue  # ours must not cut in inside the window
            out.append({"kind": "words", "clip": c["name"], "channel": ch, "t0": round(t0, 3), "t1": round(t1, 3),
                        "B_ft": fb, "C_ft": fc, "event_score": fb - fc})
    return out


def call_today_cands(pc, clips, n_max=16) -> list[dict]:
    out = []
    for c in clips.values():
        ch = "mono" if c["set"] == "ami" else "user"
        A = rec(pc, c, ch, "pipecat", "A")
        if not A:
            continue
        for t in A["cut_in_t"]:
            tu = turn_of(c, t)
            if not tu or t - tu[0] < 2.0:
                continue
            t0 = max(0.0, tu[0] - 0.1, t - 3.6)
            t1 = min(t0 + 5.0, c["dur"])
            if t1 - t0 < 4.0:
                continue
            out.append({"kind": "call_today", "clip": c["name"], "channel": ch, "t0": round(t0, 3), "t1": round(t1, 3),
                        "n_cuts_A": len(cuts_in(A, t0, t1)), "event_score": t - tu[0]})
    # AMI (the recognisable meeting look) and the rest, most lead-in speech first
    out.sort(key=lambda x: -x["event_score"])
    seen, res = set(), []
    for x in out:
        if (x["clip"], round(x["t0"])) in seen:
            continue
        seen.add((x["clip"], round(x["t0"])))
        res.append(x)
    ami = [x for x in res if x["clip"].startswith("ami")][: n_max // 2]
    return ami + [x for x in res if not x["clip"].startswith("ami")][: n_max - len(ami)]


# ============================================================================ main
def main():
    for d in (OUT, EX_AUDIO, CAND_AUDIO):
        d.mkdir(parents=True, exist_ok=True)
    pc, clips = load()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cands = interrupt_cands(pc, clips) + words_cands(pc, clips) + call_today_cands(pc, clips)
    room = {"kind": "room", "clip": ROOM_WAV.stem, "channel": "mono", "t0": round(ROOM_WIN[0] - ROOM_WAV_T0, 3),
            "t1": round(ROOM_WIN[1] - ROOM_WAV_T0, 3)}
    cands.append(room)
    # one processed WAV per distinct window
    wins = {}
    for x in cands:
        x["key"] = f"{x['clip']}.{x['channel']}.{x['t0']:.2f}-{x['t1']:.2f}"
        wins.setdefault(x["key"], x)
    print(f"{len(cands)} candidates, {len(wins)} distinct windows", flush=True)
    for k, x in wins.items():
        src = ROOM_WAV if x["kind"] == "room" else CLIP_DIR / f"{x['clip']}.{x['channel']}.wav"
        dst = CAND_AUDIO / f"{k}.wav"
        ent = cache.setdefault(k, {})
        if "proc" not in ent or not dst.exists():
            ent["proc"] = process(src, x["t0"], x["t1"], dst, src_dur(src))
        ent["src"] = str(src)
    CACHE.write_text(json.dumps(cache))

    from faster_whisper import WhisperModel

    def transcribe(model, path, t0):
        segs, _ = model.transcribe(str(path), language="en", beam_size=5, word_timestamps=True,
                                   condition_on_previous_text=False, vad_filter=False)
        ws = []
        for s in segs:
            for w in s.words or []:
                ws.append({"w": w.word.strip(), "s": round(w.start + t0 - PAD, 3), "e": round(w.end + t0 - PAD, 3)})
        return ws

    todo = [k for k in wins if "small" not in cache[k]]
    if todo:
        m = WhisperModel(WHISPER_SMALL, device="cpu", compute_type="int8", cpu_threads=2)
        for i, k in enumerate(todo):
            cache[k]["small"] = transcribe(m, CAND_AUDIO / f"{k}.wav", wins[k]["t0"])
            print(f"small {i + 1}/{len(todo)} {k}", flush=True)
            CACHE.write_text(json.dumps(cache))
        del m
    todo = [k for k, x in wins.items() if x["clip"].startswith("oto_") and "turbo" not in cache[k]]
    if todo:
        m = WhisperModel(WHISPER_TURBO, device="cpu", compute_type="int8", cpu_threads=2)
        for i, k in enumerate(todo):
            cache[k]["turbo"] = transcribe(m, CAND_AUDIO / f"{k}.wav", wins[k]["t0"])
            print(f"turbo {i + 1}/{len(todo)} {k}", flush=True)
            CACHE.write_text(json.dumps(cache))
        del m

    # ---- score
    tbref = {}
    for x in cands:
        ent = cache[x["key"]]
        hyp = [dict(h, n=norm_tok(h["w"])) for h in ent["small"]]
        hyp = [h for h in hyp if h["n"]]
        t0, t1 = x["t0"], x["t1"]
        if x["kind"] == "room":
            ref = ami_ref_words("IS1008b", ROOM_WAV_T0)
            src_ref = "AMI word XML (A-D)"
        elif x["clip"].startswith("ami_"):
            c = clips[x["clip"]]
            ref = ami_ref_words(c["conversation"], c["start"])
            src_ref = "AMI word XML (all speakers)"
        elif x["clip"].startswith("tb_"):
            k = (x["clip"], x["channel"])
            if k not in tbref:
                tbref[k] = tb_ref_words(clips[x["clip"]], x["channel"])
            ref = [dict(r) for r in tbref[k] if r["e"] > t0 - 3 * PAD - 2 and r["s"] < t1 + 3 * PAD + 2]
            align_tb(ref, hyp)
            src_ref = "TurnBench annotator a (user speaker)"
        else:
            ref = [dict(h, n=norm_tok(h["w"]), spk="user" if x["channel"] == "user" else "?", timed=True)
                   for h in ent["turbo"]]
            ref = [r for r in ref if r["n"]]
            src_ref = "Whisper large-v3-turbo pseudo-reference (otoSpeech has no transcript)"
        r_in = [r for r in ref if in_win(r, t0, t1)]
        h_in = [h for h in hyp if in_win(h, t0, t1)]
        rn, hn = [r["n"] for r in r_in], [h["n"] for h in h_in]
        x["wer"] = round(edit_distance(rn, hn) / max(1, len(rn)), 4)
        x["n_ref"] = len(rn)
        x["ref_source"] = src_ref
        x["ref_text"] = " ".join(r["w"] for r in r_in)
        x["whisper_text"] = " ".join(h["w"] for h in h_in)
        x["lufs"] = ent["proc"]["lufs"]
        x["proc"] = ent["proc"]
        x["words"] = [[round(r["s"], 3), round(r["e"], 3), r["w"], r["spk"]] for r in ref
                      if (r["s"] + r["e"]) / 2 >= t0 - PAD and (r["s"] + r["e"]) / 2 < t1 + PAD]
        x["words"].sort(key=lambda w: w[0])
        x["ref_timed_frac"] = round(sum(1 for r in r_in if r.get("timed")) / max(1, len(r_in)), 3)

    # ---- choose
    def ok(x):  # enough speech in the window to read along (>= 1.5 reference words per second)
        return x["n_ref"] >= 6 and x["n_ref"] / (x["t1"] - x["t0"]) >= 1.5

    def pick(kind, key, exclude=()):
        pool = [x for x in cands if x["kind"] == kind and x["clip"] not in exclude and ok(x)]
        good = [x for x in pool if x["wer"] <= 0.15]
        return max(good, key=key) if good else min(pool, key=lambda x: x["wer"])

    chosen = {}
    chosen["interrupt"] = pick("interrupt", lambda x: (x["C_resp_inside"], x["clip"].startswith("tb_"), -x["wer"]))
    chosen["phone"] = dict(pick("interrupt", lambda x: (x["C_resp_inside"], x["clip"].startswith("oto_"), -x["wer"]),
                                exclude={chosen["interrupt"]["clip"]}), kind="phone")
    chosen["words"] = pick("words", lambda x: (x["clip"].startswith("tb_"), x["B_ft"] >= 5.0, x["t1"] - x["t0"] <= 8.0,
                                               -x["wer"]))
    chosen["call_today"] = pick("call_today", lambda x: (x["clip"].startswith("ami"), -x["wer"], x["n_ref"]),
                                exclude={chosen["interrupt"]["clip"], chosen["phone"]["clip"]})
    chosen["room"] = next(x for x in cands if x["kind"] == "room")

    systems = {"interrupt": {"L": ("pipecat", "A"), "R": ("pipecat", "C")},
               "phone": {"L": ("pipecat", "A"), "R": ("pipecat", "C")},
               "words": {"L": ("livekit", "B"), "R": ("livekit", "C")},
               "call_today": {"P": ("pipecat", "A")}, "room": {}}
    examples = {}
    for name, x in chosen.items():
        dst = EX_AUDIO / f"{name}.wav"
        shutil.copyfile(CAND_AUDIO / f"{x['key']}.wav", dst)
        e = {"clip": x["clip"], "set": "ami" if x["kind"] == "room" else clips[x["clip"]]["set"],
             "channel": x["channel"], "source_wav": cache[x["key"]]["src"], "processed_wav": str(dst),
             "pad_s": PAD, "t0": x["t0"], "t1": x["t1"], "dur_s": round(x["t1"] - x["t0"], 3),
             "processed_sr": SR_OUT, "systems": {}, "per_system": {}}
        if x["kind"] != "room":
            c = clips[x["clip"]]
            e["clip_start_in_recording"] = c["start"]
            e["user_turns"] = c["user_turns"]
            on, on_src = onset_of(c, x["channel"])
            e["user_onset"], e["user_onset_source"] = on, on_src
            for side, (fw, sy) in systems[name].items():
                r = rec(pc, c, x["channel"], fw, sy)
                e["systems"][side] = f"{fw}/{sy}"
                ft = r.get("first_text_ms_after_onset")
                e["per_system"][f"{fw}/{sy}"] = {
                    "cut_in_t": r["cut_in_t"], "cut_in_t_in_window": cuts_in(r, x["t0"], x["t1"]),
                    "responses": responses(r),
                    "responses_in_window": [q["response"] for q in responses(r)
                                            if q["response"] is not None and x["t0"] <= q["response"] <= x["t1"]],
                    "user_turns": c["user_turns"], "first_text_ms_after_onset": ft,
                    "first_text": round(on + ft / 1000, 3) if ft is not None else None}
        else:
            e["meeting_window"] = list(ROOM_WIN)
            e["wav_starts_at_meeting_s"] = ROOM_WAV_T0
            e["note"] = "clip time = meeting time - 1670.0; systems: none (live diarization session shot)"
        if name in ("interrupt", "phone"):
            e["C_response_inside_window"] = x["C_resp_inside"]
            e["C_response_after_turn"] = x["C_resp"]
        e.update({"words": x["words"], "words_speakers": "user = the human on the user channel" if x["channel"] == "user"
                  else "AMI speaker letters", "wer_after_processing": x["wer"], "wer_ref_words": x["n_ref"],
                  "ref_source": x["ref_source"], "ref_word_timing_from_whisper_frac": x["ref_timed_frac"],
                  "whisper_text": x["whisper_text"], "ref_text": x["ref_text"], "lufs": x["lufs"],
                  "true_peak_dbfs": x["proc"]["true_peak_dbfs"], "loudness_method": x["proc"]["method"]})
        examples[name] = e
    (OUT / "examples.json").write_text(json.dumps({"generated_by": "demo/v5/examples.py", "examples": examples},
                                                  indent=1))

    # ---- table
    lines = ["# v5 example candidates: intelligibility (Whisper small int8 on the processed window)", "",
             "WER vs reference words inside the window (lowercase, no punctuation, bracketed tags / fillers / cut-off "
             "fragments dropped). oto rows: vs Whisper large-v3-turbo pseudo-reference. * = chosen.", "",
             "| kind | clip | ch | t0 | t1 | events | ref words | WER | LUFS | chosen |", "|---|---|---|---|---|---|---|---|---|---|"]
    chosen_keys = {(n if n != "phone" else "interrupt", x["key"]): n for n, x in chosen.items()}
    for x in sorted(cands, key=lambda x: (x["kind"], x["wer"])):
        if x["kind"] in ("interrupt",):
            ev = f"A cuts {x['n_cuts_A']}, C resp {x['C_resp']} ({'in' if x['C_resp_inside'] else 'out'})"
        elif x["kind"] == "words":
            ev = f"B first text {x['B_ft']:.2f} s, C {x['C_ft']:.2f} s"
        elif x["kind"] == "call_today":
            ev = f"A cuts {x['n_cuts_A']}"
        else:
            ev = "-"
        ch = chosen_keys.get((x["kind"], x["key"]), "")
        lines.append(f"| {x['kind']} | {x['clip']} | {x['channel']} | {x['t0']:.2f} | {x['t1']:.2f} | {ev} | "
                     f"{x['n_ref']} | {100 * x['wer']:.1f} % | {x['lufs']:.1f} | {'* ' + ch if ch else ''} |")
    (OUT / "examples_wer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for n, e in examples.items():
        print(n, e["clip"], e["channel"], e["t0"], e["t1"], "WER", e["wer_after_processing"], "LUFS", e["lufs"])


if __name__ == "__main__":
    main()
