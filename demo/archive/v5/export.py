"""v5 data export: everything the HTML shots draw, as one JSON (no model is loaded; plain numpy over recorded data).

    PYTHONPATH=. .venv/bin/python demo/archive/v5/export.py   ->  $DEMO_OUT/v5_data/data.json

Per clip: the audio envelope (10 ms hop, the live waveform bars), and per system side the user turns, cut-ins and
responses from the recorded E2E session (runs/e2e_final.json per_clip), and the transcript as timed change points:
  ours    = the served model's streaming words from demo/events/<clip>.json (word granularity, split tokens mended),
            only the user's turn (words that appeared before the user's onset belong to the previous speaker);
  default = the reference text of the clip as ONE block at the measured first-text time and nothing added later
            (the session texts were not retained); the block holds the words spoken before that moment (words
            placed along the user's speech energy). Shown muted and labelled as reference text.
Room: the 8-column Nemotron-3 live session on AMI IS1008b [1670, 1730) s (research/archive/DIARIZATION_FIX.md fix run,
scratch/diar/logs/new_timeout_any_ami_IS1008b_1670s.json): per-frame speaker activity and the streaming words, each
new word attributed to the diarizer's primary column at its arrival. The architecture shot reads the same session
(VAD, speaker, end-of-turn values, turn ends, words) and one Parakeet-TDT rewrite from demo/events/ami_IS1008b_003_agentend.json.
Numbers: demo/archive/numbers.json (each with its runs/*.json path).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import render_v2 as V2  # noqa: E402  (data classes only: Data, Events, ClipAudio)

OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out")) / "v5_data"
HOP = 0.01


def envelope(audio: V2.ClipAudio):
    n = int(audio.dur / HOP)
    env = audio.env_at(0, n * HOP, n)
    # a bit of smoothing so bars breathe instead of flicker
    k = np.ones(3) / 3
    env = np.convolve(env, k, mode="same")
    return [round(float(min(1.0, v)), 3) for v in env]


def ours_text(ev: V2.Events, onset: float, t_end: float):
    """Change points [t, text] of the served stream, rebuilt from raw finals + partial (overlap on raw tokens), mended."""
    stream_finals = [(f["arr"], f["text"].strip()) for f in ev.finals if f.get("source", "stream") == "stream" and (f.get("text") or "").strip()]

    def raw_at(now):
        fin = " ".join(t for a, t in stream_finals if a <= now).split()
        txt, nxt = "", None
        for arr, text in ev.partials:
            if arr <= now:
                txt = text
            else:
                nxt = text; break
        # word granularity: hold back a last token that the next partial extends ("me" -> "messed")
        if txt and nxt is not None and nxt.startswith(txt) and not nxt.startswith(txt + " "):
            txt = txt.rsplit(" ", 1)[0] if " " in txt else ""
        pw = txt.split()
        k = 0
        for n in range(min(len(fin), len(pw)), 0, -1):
            if fin[-n:] == pw[:n]:
                k = n; break
        return fin + pw[k:]

    pre = len(raw_at(onset))
    out, prev = [], None
    for i in range(int(t_end * 30) + 1):
        t = i / 30
        words = raw_at(t)[pre:]
        s = ev._mend(" ".join(words))
        if s != prev:
            out.append([round(t, 3), s]); prev = s
    # a word the stream later rewrote ("me" -> "messed"): show only the stable prefix until the rewrite lands
    for i in range(len(out) - 2, -1, -1):
        a, b = out[i][1].split(), out[i + 1][1].split()
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        if k < len(a):
            out[i][1] = " ".join(a[:k])
    return out


def default_text(ref: str, env_full, turns, releases):
    words = ref.split()
    a, b = turns[0]
    i0, i1 = int(a / HOP), int(b / HOP)
    e = np.array(env_full[i0:i1])
    w = (e > 0.12).astype(float) + 1e-3
    cum = np.cumsum(w) / w.sum()
    t_word = [a + (np.searchsorted(cum, (k + 1) / len(words)) * HOP) for k in range(len(words))]
    out = [[0.0, ""]]
    for r, upto in releases:
        n = sum(1 for tw in t_word if tw <= upto)
        if n:
            out.append([round(r, 3), " ".join(words[:n])])
    return out


def side(D, clip, fw, sysm, env_full):
    s = D.session(clip, fw, sysm)
    rec = {"label": s["label"], "sub": s["sub"], "ours": s["ours"], "user_turns": s["user_turns"],
           "cut_ins": s["cut_ins"], "responses": [e["response"] for e in s["ends"] if e.get("response") is not None],
           "first_text": s["first_text"]}
    onset = s["user_turns"][0][0]
    if s["ours"]:
        rec["text"] = ours_text(D.events(clip), onset, D.audio(clip).dur)
    else:
        ft = s["first_text"]
        rec["text"] = default_text(s["text"], env_full, s["user_turns"], [(ft, ft)])
    return rec


def logmel(w, sr, secs, n_mels=80):
    """80-bin log-mel at 25 ms / 10 ms (the encoder's front end, Slaney mel scale), first `secs` seconds, as base64
    uint8 (time-major, 80 values per 10 ms frame), for the scrolling spectrogram in the architecture shot."""
    import base64
    w = w[: int(secs * sr)]
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        w = np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w); sr = 16000
    w = np.append(w[0], w[1:] - 0.97 * w[:-1])
    n_fft, win, hop = 512, 400, 160
    k = 1 + (len(w) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(k)[:, None]
    fr = w[idx] * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(fr, n_fft)) ** 2
    def hz2mel(f):
        f = np.asarray(f, float); m = f / (200.0 / 3)
        return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27), m)
    def mel2hz(m):
        m = np.asarray(m, float); f = m * 200.0 / 3
        return np.where(m >= 15, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)), f)
    mpts = mel2hz(np.linspace(hz2mel(0), hz2mel(8000), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = mpts[i], mpts[i + 1], mpts[i + 2]
        fb[i] = np.maximum(0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c))) * 2 / (r - l)
    lm = np.log(spec @ fb.T + 1e-10)
    lo, hi = np.percentile(lm, 5), np.percentile(lm, 99.7)
    q = np.clip((lm - lo) / (hi - lo), 0, 1)
    return base64.b64encode((q * 255).astype(np.uint8).tobytes()).decode()


DIAR = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/diar")


def room():
    """Live 8-column session: frames (t, 4 people's activity, vad, eot, primary), turn ends, words with speaker."""
    import soundfile as sf
    log = json.loads((DIAR / "logs" / "new_timeout_any_ami_IS1008b_1670s.json").read_text())
    ev = log["events"][0]
    fr = [e for e in ev if e["type"] == "frame"]
    frames = [[round(f["t"], 2), [round(v, 3) for v in f["speakers"][:8]], round(f["vad"], 3),
               None if f.get("eot") is None else round(float(f["eot"]), 3), f["primary"]] for f in fr]
    prim_t = np.array([f["t"] for f in fr]); prim = [f["primary"] for f in fr]
    # words with speakers, as change points: [t, [[col, word], ...]] (committed turns + the live partial)
    def prim_at(t):
        return prim[max(0, int(np.searchsorted(prim_t, t - 0.4)) - 1)]
    committed, buf, prev, words = [], [], [], []
    for p in (e for e in ev if e["type"] in ("partial", "final")):
        w = (p.get("text") or "").split()
        if p["type"] == "final":
            if w:
                committed += buf
            buf, prev = [], []
            continue
        k = 0
        while k < min(len(w), len(prev)) and w[k] == prev[k]:
            k += 1
        if k == 0 and prev and not (w and prev and (w[0].startswith(prev[0]) or prev[0].startswith(w[0]))):
            committed += buf; buf = []
        nb = []
        for i, x in enumerate(w):
            spk = buf[i][0] if i < len(buf) else prim_at(p["t"])
            nb.append([spk, x])
        buf, prev = nb, w
        words.append([round(p["t"], 2), committed + buf])
    te = [round(e["t"], 2) for e in ev if e["type"] == "turn_end"]
    finals = [[round(e["t"], 2), e.get("speaker"), e.get("text", "")] for e in ev if e["type"] == "final" and (e.get("text") or "").strip()]
    w, sr = sf.read(DIAR / "clips" / "ami_IS1008b_1670s.wav", dtype="float32")
    audio = V2.ClipAudio(DIAR / "clips" / "ami_IS1008b_1670s.wav")
    tdt = V2.Events(HERE.parent / "events" / "ami_IS1008b_003_agentend.json")
    st = next(f["text"] for f in tdt.finals if f.get("source") == "stream" and f["text"].strip())
    tt = next(f["text"] for f in tdt.finals if f.get("source") == "tdt_v3" and f["text"].strip())
    return {"env": envelope(audio), "hop": HOP, "mel": logmel(w, sr, 40.0), "dur": audio.dur, "frames": frames, "words": words, "turn_ends": te, "finals": finals,
            "tdt_example": {"stream": st, "tdt": tt}, "wav": str(DIAR / "clips" / "ami_IS1008b_1670s.wav")}


def examples(D):
    """The film's real examples (demo/archive/v5/examples.py picks them by event rules + Whisper intelligibility and writes
    loudness-processed audio + word-timed subtitles to $DEMO_OUT/v5_data/examples.json). Each becomes a clip entry
    "ex_<key>" on the stored per-clip timeline (clip time), with the recorded cut-ins / responses / first-text times."""
    import soundfile as sf
    ex = json.loads((OUT / "examples.json").read_text())["examples"]
    out = {}
    for key, e in ex.items():
        w, sr = sf.read(e["processed_wav"], dtype="float32")
        off = e["t0"] - e["pad_s"]
        n = int((e["t1"] + e["pad_s"] + 0.5) / HOP)
        step = int(sr * HOP)
        env = np.zeros(n, np.float32)
        k = len(w) // step
        seg = np.abs(w[: k * step]).reshape(k, step).max(1)
        seg = seg / max(1e-6, np.percentile(seg, 99.5))
        i0 = int(round(off / HOP))
        for j in range(k):
            if 0 <= i0 + j < n:
                env[i0 + j] = seg[j]
        env = np.convolve(env, np.ones(3) / 3, mode="same")
        entry = {"env": [round(float(min(1.0, v)), 3) for v in env], "hop": HOP, "dur": e["t1"] + e["pad_s"],
                 "t0": e["t0"], "t1": e["t1"], "subs": [[round(a, 3), round(b, 3), wd, sp] for a, b, wd, sp in e["words"]],
                 "processed_wav": e["processed_wav"], "offset": off, "wer": e["wer_after_processing"], "lufs": e["lufs"], "sides": {}}
        names = {"pipecat/A": "Pipecat default", "livekit/B": "LiveKit default", "pipecat/C": "audioforge", "livekit/C": "audioforge"}
        for pk, sysk in e.get("systems", {}).items():
            ps = e["per_system"][sysk]
            ours = sysk.endswith("/C")
            side_rec = {"label": names[sysk], "ours": ours, "user_turns": ps["user_turns"], "cut_ins": ps["cut_in_t"],
                        "responses": [r["response"] for r in ps["responses"] if r.get("response") is not None],
                        "first_text": ps.get("first_text"), "text": [[0.0, ""]]}
            if key == "words" and ours:
                ev = V2.Events(HERE.parent / "events" / f"{e['clip']}.user.json")
                side_rec["text"] = ours_text(ev, 0.0, e["t1"] + 1)
                side_rec["text_source"] = "fresh live session of the same audio, fixed defaults (demo/events/tb_160.user.json)"
            elif not ours and ps.get("first_text") is not None:
                ft = ps["first_text"]
                block = " ".join(wd for a, b, wd, sp in e["words"] if b <= ft - 0.2 and sp in ("user", "C"))
                side_rec["text"] = [[0.0, ""], [round(ft, 3), block]] if block else [[0.0, ""]]
            entry["sides"][pk] = side_rec
        out[f"ex_{key}"] = entry
    return out


def merge_manifest(D, clips):
    """The AUDIO agent's clips (demo/archive/v5/audio/clips.py): $DEMO_OUT/v5_data/clips_manifest.json + subs.json. Each
    manifest clip replaces/updates the data.json record of the same id: processed audio (exactly [t0, t1] of the
    source, offset = t0), span, recorded per-system events inside the identical span, the agent-voice schedule, the
    speaker's words. Pages read only data.json."""
    import soundfile as sf
    mp, sp = OUT / "clips_manifest.json", OUT / "subs.json"
    if not mp.exists():
        return
    man = json.loads(mp.read_text())["clips"]
    subs = json.loads(sp.read_text())["clips"] if sp.exists() else {}
    op = OUT / "opening.json"
    if op.exists() and "ex_call_today" not in man:   # fallback opening call (demo/archive/v5/opening.py) when the manifest has none
        o = json.loads(op.read_text()); man["ex_call_today"] = o["clip"]; subs["ex_call_today"] = {"words": o["subs"]}
    for cid, m in man.items():
        rec = dict(clips.get(cid, {}))
        w, sr = sf.read(m["wav"], dtype="float32")
        t0, t1 = m["span"]
        n = int((t1 + 0.5) / HOP); step = int(sr * HOP); k = len(w) // step
        seg = np.abs(w[: k * step]).reshape(k, step).max(1); seg = seg / max(1e-6, np.percentile(seg, 99.5))
        env = np.zeros(n, np.float32); i0 = int(round(m["offset"] / HOP))
        env[i0: i0 + k] = seg[: max(0, n - i0)]
        env = np.convolve(env, np.ones(3) / 3, mode="same")
        rec.update({"env": [round(float(min(1.0, v)), 3) for v in env], "hop": HOP, "dur": t1 + 0.3, "t0": t0, "t1": t1,
                    "processed_wav": m["wav"], "offset": m["offset"], "wer": m.get("wer_after"), "lufs": m.get("lufs"),
                    "agent_voice": m.get("agent_voice", {}), "source_clip": m.get("source_clip"), "source": m.get("source"),
                    "speakers_in_span": m.get("speakers_in_span")})
        if cid in subs:
            rec["subs"] = subs[cid]["words"]
        meta = D.clips.get(m.get("source_clip"), {})
        sides = {}
        for pk, sy in (m.get("systems") or {}).items():
            ours = sy["system"].endswith("/C")
            sides[pk] = {"label": sy["label"], "ours": ours, "system": sy["system"], "record": sy.get("record"),
                         "user_turns": meta.get("user_turns") or [[sy.get("onset", t0), t1]],
                         "cut_ins": sy["cut_ins"], "responses": sy["responses"], "first_text": sy.get("first_text"),
                         "onset": sy.get("onset"), "text": [[0.0, ""]]}
            if cid == "ex_call_today" and sy.get("first_text") is not None:
                ft = sy["first_text"]; block = " ".join(x[2] for x in rec.get("subs", []) if x[1] <= ft - 0.2)
                sides[pk]["text"] = [[0.0, ""], [round(ft, 3), block]] if block else [[0.0, ""]]
            if cid == "ex_words" and ours:
                ev = V2.Events(HERE.parent / "events" / f"{m['source_clip']}.user.json")
                sides[pk]["text"] = ours_text(ev, 0.0, t1 + 1)
            elif cid == "ex_words" and sy.get("first_text") is not None:
                ft = sy["first_text"]
                block = " ".join(x[2] for x in rec.get("subs", []) if x[1] <= ft - 0.2)
                sides[pk]["text"] = [[0.0, ""], [round(ft, 3), block]] if block else [[0.0, ""]]
        if sides:
            rec["sides"] = sides
        clips[cid] = rec


def main():
    D = V2.Data()
    clips = {}
    E2E = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad/clips")
    plan = {"ami_IB4002_119": [("pipecat", "A"), ("pipecat", "C")],
            "ami_ES2011b_027": [("livekit", "B"), ("livekit", "C")],
            "ami_TS3004b_064": [("pipecat", "A"), ("pipecat", "C")],
            "oto_5bc1e19ecd43cc6d45d0c1b9b78be5aa": [("pipecat", "A"), ("pipecat", "C")]}
    for clip, systems in plan.items():
        if clip.startswith("oto_"):   # real phone call (otoSpeech, CC-BY-4.0): audio from the E2E run, no transcript
            D._audio[clip] = V2.ClipAudio(E2E / f"{clip}.mono.wav")
        env = envelope(D.audio(clip))
        sides = {}
        for fw, sm in systems:
            if clip.startswith("oto_"):
                s = D.session(clip, fw, sm)
                sides[f"{fw}/{sm}"] = {"label": s["label"], "sub": s["sub"], "ours": s["ours"], "user_turns": s["user_turns"],
                                       "cut_ins": s["cut_ins"], "responses": [e["response"] for e in s["ends"] if e.get("response") is not None],
                                       "first_text": s["first_text"], "text": [[0.0, ""]]}
            else:
                sides[f"{fw}/{sm}"] = side(D, clip, fw, sm, env)
        clips[clip] = {"env": env, "hop": HOP, "dur": D.audio(clip).dur, "sides": sides}
    clips["room_IS1008b_1670s"] = room()
    exs = examples(D)
    exs["ex_call_today"]["sides"]["P"]["text"] = clips["ami_IB4002_119"]["sides"]["pipecat/A"]["text"]   # AMI reference block at its measured arrival
    clips.update(exs)
    rm = dict(clips["room_IS1008b_1670s"]); rm.update({k: exs["ex_room"][k] for k in ("env", "subs", "t0", "t1", "processed_wav", "offset", "wer", "lufs")})
    clips["ex_room"] = rm
    merge_manifest(D, clips)
    nums = {k: v for k, v in D.N.items()}
    e2e = json.loads((HERE.parents[1] / "runs" / "e2e_final.json").read_text())
    nums["cutins_per_set"] = {"value": {k.split("|")[0] + "|" + k.split("|")[1]: {s2: e2e["table"][f"{k.rsplit('|', 1)[0]}|{s2}"]["cut_ins_per_min"] for s2 in ("A", "C")}
                                        for k in e2e["table"] if k.endswith("|pipecat|A")}, "source": "runs/e2e_final.json", "path": "table > *|pipecat|A,C > cut_ins_per_min"}
    # plain numbers for the result charts (one number per system, whole percents / one-decimal seconds); every value
    # keeps its runs/*.json path in "src"
    t = e2e["table"]
    nb = json.loads((HERE / "shots" / "numbers_b.json").read_text())
    def row(k, f):
        return {"value": t[k][f], "src": f"runs/e2e_final.json > table > {k} > {f}"}
    live = {}
    for st, lab in (("oto", "phone"), ("turnbench", "turnbench")):
        live[lab] = {
            "unanswered_3s": {"ours": row(f"{st}|user|pipecat|C", "missed_3s"), "livekit": row(f"{st}|user|livekit|B", "missed_3s"), "pipecat": row(f"{st}|user|pipecat|A", "missed_3s")},
            "cutins_per_min": {"ours": row(f"{st}|user|pipecat|C", "cut_ins_per_min"), "pipecat": row(f"{st}|user|pipecat|A", "cut_ins_per_min")},
            "first_text_ms": {"ours": row(f"{st}|user|pipecat|C", "first_text_ms_median"), "pipecat": row(f"{st}|user|pipecat|A", "first_text_ms_median"), "livekit": row(f"{st}|user|livekit|B", "first_text_ms_median")},
            "n_ends": t[f"{st}|user|pipecat|C"]["n_ends"], "n_clips": t[f"{st}|user|pipecat|C"]["n_clips"]}
    caught = {k: {"value": round(100 - D.N[f"turn_miss_all/{k}"]["value"], 1), "src": D.N[f"turn_miss_all/{k}"]["source"] + " (100 - miss rate)"}
              for k in ("ours_hybrid", "parakeet_eou", "smartturn_silero", "silero_timeout", "livekit_text")}
    try:   # legacy charts.html numbers (the film now uses the charts agent's pages and numbers_b.js)
        icsi = {"ours": {"value": round(100 - nb["icsi/ours"]["value"], 1), "src": nb["icsi/ours"]["source"]},
                "silero": {"value": round(100 - nb["icsi/silero"]["value"], 1), "src": nb["icsi/silero"]["source"]},
                "ours_fc": {"value": nb["icsi/ours/fc"]["value"], "src": nb["icsi/ours/fc"]["source"]},
                "silero_fc": {"value": nb["icsi/silero/fc"]["value"], "src": nb["icsi/silero/fc"]["source"]}}
        tb = {k: {"value": 100 * nb[f"tb/{k}"]["value"], "src": nb[f"tb/{k}"]["source"]} for k in ("ours", "silero", "kyutai", "eou", "smartturn")}
        wer = {k: {"value": nb[f"wer/{k}"]["value"], "src": nb[f"wer/{k}"]["source"]} for k in ("ours", "whisper_turbo", "parakeet_ctc", "whisper_small")}
    except KeyError:
        icsi = tb = wer = None
    nums["plain"] = {"live": live, "meetings_caught": caught, "icsi_caught": icsi, "turnbench_caught": tb, "wer": wer}
    nums["server_rtf_nemotron"] = {"value": max(e2e["components_live"][k]["server_total"]["rtf_wall"] for k in ("pipecat:CN", "livekit:CN")),
                                   "source": "runs/e2e_final.json", "path": "components_live > {pipecat,livekit}:CN > server_total > rtf_wall (max)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({"clips": clips, "numbers": nums}))
    for clip, c in clips.items():
        for k, s in c.get("sides", {}).items():
            print(clip, k, "cut_ins", s["cut_ins"], "resp", s["responses"], "text pts", len(s["text"]), "|", s["text"][-1][1][:90])
    print("wrote", OUT / "data.json")


if __name__ == "__main__":
    main()
