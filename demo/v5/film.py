"""v5 film renderer: HTML/CSS/JS shots (demo/v5/shots/*.html; the frame is a pure function of t) rendered by headless
Chromium via Playwright at 30 fps into PNG frames on the SSD, then encoded with ffmpeg (-threads 2) with the narration,
the example audio (real recordings + the demo agent voice at the recorded times) and the generated music bed.

    V=/Volumes/afdev/venvs/video/bin/python
    $V demo/v5/film.py --cut pilot2 --stills           # one still per shot + DOM metrics -> $DEMO_OUT/v5_stills/<cut>_<W>x<H>/
    $V demo/v5/film.py --cut pilot2 --frames           # resumable, --budget s per call (exit 3 = run again); DOM QA samples at 2 fps
    $V demo/v5/film.py --cut pilot2 --mux              # audio mix (+ stems) + H.264/AAC -> $DEMO_OUT/showcase_v5_<cut>[_square].mp4
    $V demo/v5/qa.py --cut pilot2                      # QA on the rendered film (frames, DOM samples, audio stems)
    (--size 1080x1080: the square cut, re-laid out by each page, never cropped)

A shot is a list of segments: ("say", scene) = one narration line; ("clip", key, t0, t1) = a real recording plays
alone (narration silent, music muted) for panel `key` from clip time t0 to t1; ("hold", s) = picture only.
Inputs: $DEMO_OUT/v5_data/data.json (demo/v5/export.py), $DEMO_OUT/narration_v5*.json + agent_voice_v5.json (demo/tts.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
SHOTS_DIR = HERE / "shots"
OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
FPS, SR = 30, 48000
BPM = 100.0
BEAT = 60.0 / BPM
LEAD, GAP, TAIL = 0.3, 0.35, 0.5
sys.path.insert(0, str(HERE))
import audio as A  # noqa: E402
from shots_def import SHOTS, CUTS  # noqa: E402


def load_units(cut):
    units = {}
    for name in (f"narration_v5_{cut}.json", "narration_v5.json"):
        p = OUT / name
        if p.exists():
            for u in json.loads(p.read_text())["units"]:
                units.setdefault(u["scene"], u)
    return units


def agent_line():
    """The demo agent's two lines: the full reply and a short one for tight gaps (never chopped mid-word)."""
    j = json.loads((OUT / "agent_voice_v5.json").read_text())
    full, short = j["units"][0], j["units"][1]
    mk = lambda u: {"wav": u["wav"], "dur": u["end"] - u["start"], "text": u.get("caption") or u["text"]}  # noqa: E731
    return {"full": mk(full), "short": mk(short)}


def caption_chunks(text, max_chars):
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    out = []
    for s in sents:
        if out and len(out[-1]) + 1 + len(s) <= max_chars and len(s) < 18:
            out[-1] += " " + s
        elif len(s) <= max_chars:
            out.append(s)
        else:
            acc = ""
            for c in [c.strip() for c in re.split(r"(?<=[,;:])\s+", s)]:
                if acc and len(acc) + 1 + len(c) > max_chars:
                    out.append(acc); acc = c
                else:
                    acc = (acc + " " + c).strip()
            if acc:
                out.append(acc)
    # a chunk still too long (no clause break): split at word boundaries into near-equal parts
    res = []
    for c in out:
        if len(c) <= max_chars:
            res.append(c); continue
        words, k = c.split(), -(-len(c) // max_chars)
        target, cur = len(c) / k, ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max(target, 1) + 6:
                res.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            res.append(cur)
    return res


def fit_label(label, max_chars):
    """Clip labels are ' · '-separated; keep as many leading parts as fit."""
    parts, out = label.split(" · "), ""
    for p in parts:
        cand = p if not out else out + " · " + p
        if len(cand) > max_chars:
            break
        out = cand
    return out or parts[0][:max_chars]


def timeline(cut, W):
    units = load_units(cut)
    data = json.loads((OUT / "v5_data" / "data.json").read_text())
    agent = agent_line() if (OUT / "agent_voice_v5.json").exists() else None
    max_chars = 62 if W > 1500 else 36
    t, shots = 0.0, []
    for name in CUTS[cut]:
        sh = dict(SHOTS[name], name=name)
        segs, lt = [], LEAD
        voice, clips, agents, captions = [], [], [], []
        for sg in sh["segments"]:
            if sg[0] == "say":
                u = units.get(sg[1])
                ud = (u["end"] - u["start"]) if u else 5.0
                cap = (u.get("caption") or u["text"]) if u else f"[{sg[1]}]"
                segs.append({"kind": "say", "scene": sg[1], "start": lt, "end": lt + ud})
                if u:
                    voice.append({"start": t + lt, "wav": u["wav"], "dur": ud})
                chunks = caption_chunks(cap, max_chars)
                tot = sum(len(c) for c in chunks) or 1
                acc = lt
                for c in chunks:
                    d = ud * len(c) / tot
                    captions.append([acc, acc + d, c]); acc += d
                # cues: phrase -> local time (picture leads the voice by 0.25 s)
                for k, phrase in (sh.get("cues") or {}).items():
                    i = cap.find(phrase)
                    if i >= 0:
                        sh.setdefault("cue_t", {})[k] = max(0.0, lt + ud * i / max(1, len(cap)) - 0.25)
                lt += ud + GAP
            elif sg[0] == "clip":
                key, t0, t1 = sg[1], sg[2], sg[3]
                rec = data["clips"].get(sh["clip"], {})
                if t0 is None:   # the span comes from the clip record (identical for both sides)
                    t0, t1 = rec["t0"], rec["t1"]
                lt += 0.25                        # a breath of silence before the real audio
                segs.append({"kind": "clip", "key": key, "t0": t0, "t1": t1, "start": lt, "end": lt + (t1 - t0)})
                clips.append({"start": t + lt, "clip": sh["clip"], "t0": t0, "t1": t1})
                lab = sg[4] if len(sg) > 4 else ""
                if isinstance(lab, (tuple, list)):
                    lab = lab[0] if W > 1500 else lab[1]
                captions.append([lt, lt + (t1 - t0), fit_label(lab, max_chars if W > 1500 else 40)])
                # the demo agent's voice at the recorded times of this panel's system
                over = t1 - t0
                av = {} if sh.get("noagent") else (rec.get("agent_voice") or {})   # words example: no replies (both would overlap)
                if av:   # the AUDIO agent's schedule: one line per recorded event, never chopped
                    for pk, lines in av.items():
                        if key != "LR" and pk != key:
                            continue
                        for ln in lines:
                            sh.setdefault("agent_lines", {}).setdefault(pk, []).append([ln["t"], ln["text"], ln["dur"]])
                            agents.append({"start": t + lt + (ln["t"] - t0), "wav": ln["wav"], "dur": ln["dur"], "trim": ln.get("trim_s")})
                            over = max(over, ln["t"] - t0 + ln["dur"])
                elif not sh.get("noagent"):
                  for pk in ((list(sh["panels"]) if key == "LR" else [key]) if sh.get("panels") else []):
                    side = data["clips"][sh["clip"]]["sides"][sh["panels"][pk]]
                    evs = sorted([c for c in side["cut_ins"] if t0 <= c < t1] + [r for r in side["responses"] if t0 <= r < t1])
                    for i, c in enumerate(evs):
                        gap = (evs[i + 1] - c) if i + 1 < len(evs) else 99.0
                        line = None
                        if agent and gap >= agent["full"]["dur"] + 0.15:
                            line = agent["full"]
                        elif agent and gap >= agent["short"]["dur"] + 0.1:
                            line = agent["short"]
                        sh.setdefault("agent_lines", {}).setdefault(pk, []).append([c, line["text"] if line else "", line["dur"] if line else 0.0])
                        if line:
                            agents.append({"start": t + lt + (c - t0), "wav": line["wav"], "dur": line["dur"]})
                            over = max(over, c - t0 + line["dur"])
                segs[-1]["end_audio"] = lt + over
                lt += over + 0.35
            elif sg[0] == "hold":
                segs.append({"kind": "hold", "start": lt, "end": lt + sg[1]}); lt += sg[1]
        dur = max(sh.get("min_dur", 0), lt - GAP + sh.get("tail", TAIL))
        dur = math.ceil(dur / (BEAT / 2) - 1e-6) * (BEAT / 2)   # cut on the (half) beat
        sh.update(start=t, dur=dur, end=t + dur, segs=segs, voice=voice, clips=clips, agents=agents, captions=captions)
        sh.setdefault("cue_t", {})
        sh.setdefault("agent_lines", {})
        shots.append(sh); t += dur
    return shots, t


def ctx_for(sh, lt, sq=False):
    cap = ""
    for a, b, c in sh["captions"]:
        if a - 0.05 <= lt < b + 0.3:
            cap = c
    return {"caption": cap, "fine": sh.get("fine_sq" if sq and sh.get("fine_sq") else "fine", ""), "dur": sh["dur"], "cues": sh["cue_t"], "segs": sh["segs"],
            "agent_lines": sh.get("agent_lines", {}), "light": sh.get("light", 1), "breathe": sh.get("breathe", False)}


PAGE_KEYS = ("page", "clip", "panels", "side", "offset", "chart", "mode", "name", "dur", "cue_t", "params", "fine", "segs", "notext", "beat", "subs", "who")


class Browser:
    def __init__(self, W, H):
        from playwright.sync_api import sync_playwright
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/Volumes/afdev/venvs/video/browsers")
        self.pw = sync_playwright().start()
        self.b = self.pw.chromium.launch(args=["--disable-gpu", "--font-render-hinting=none", "--renderer-process-limit=1"])
        self.p = self.b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        self.W, self.H = W, H
        self.data = json.loads((OUT / "v5_data" / "data.json").read_text())
        self.cur = None

    def open(self, sh):
        if self.cur == sh["name"]:
            return
        self.p.goto((SHOTS_DIR / sh["page"]).as_uri())
        self.p.evaluate("document.fonts.ready")
        params = {k: sh[k] for k in PAGE_KEYS if k in sh}
        if self.W == self.H and sh.get("fine_sq"):
            params["fine"] = sh["fine_sq"]
        self.p.evaluate("([d, p, W, H]) => { window.PARAMS = p; init(d, p, W, H); }", [self.data, params, self.W, self.H])
        self.cur = sh["name"]

    def shot(self, sh, lt, path):
        self.open(sh)
        # chart shots skip their empty pre-build (title-only / blank first frames): page time = shot time + skip
        self.p.evaluate("([t, c]) => render(t, c)", [lt + sh.get("skip", 0.0), ctx_for(sh, lt, self.W == self.H)])
        self.p.screenshot(path=str(path), type="png")

    def metrics(self):
        return self.p.evaluate((HERE / "metrics.js").read_text())

    def close(self):
        self.b.close(); self.pw.stop()


def frame_dir(cut, W, H):
    d = OUT / "v5_frames" / f"{cut}_{W}x{H}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_frames(a, W, H):
    shots, total = timeline(a.cut, W)
    n = int(round(total * FPS))
    d = frame_dir(a.cut, W, H)
    qa_path = d / "dom_samples.jsonl"
    done_qa = set()
    if qa_path.exists() and not a.force:
        done_qa = {json.loads(line)["f"] for line in qa_path.open()}
    elif qa_path.exists():
        qa_path.unlink()
    t0 = time.time()
    br, done = None, 0
    with qa_path.open("a") as qa:
        for f in range(n):
            p = d / f"{f:05d}.png"
            need_qa = f % 15 == 0 and f not in done_qa
            if p.exists() and not a.force and not need_qa:
                continue
            if time.time() - t0 > a.budget:
                print(f"budget reached at frame {f}/{n}; run again", flush=True)
                if br:
                    br.close()
                sys.exit(3)
            T = f / FPS
            sh = next(s for s in shots if s["start"] <= T < s["end"] + 1e-9)
            if br is None:
                br = Browser(W, H)
            br.shot(sh, T - sh["start"], p)
            if f % 15 == 0:
                m = br.metrics()
                qa.write(json.dumps({"f": f, "t": T, "shot": sh["name"], "lt": T - sh["start"], **m}) + "\n"); qa.flush()
            done += 1
            if done % 300 == 0:
                print(f"frame {f}/{n}  {done / (time.time() - t0):.1f} fps", flush=True)
    if br:
        br.close()
    # drop frames beyond the end (a shorter re-cut)
    for p in d.glob("*.png"):
        if int(p.stem) >= n:
            p.unlink()
    print(f"frames complete: {n} in {d}")


def cmd_stills(a, W, H):
    shots, total = timeline(a.cut, W)
    d = OUT / "v5_stills" / f"{a.cut}_{W}x{H}"
    d.mkdir(parents=True, exist_ok=True)
    br = Browser(W, H)
    meta = {}
    for sh in shots:
        if a.only and sh["name"] not in a.only.split(","):
            continue
        lt = sh.get("still", sh["dur"] * 0.62) if a.at is None else a.at
        p = d / f"{sh['name']}.png"
        br.shot(sh, lt, p)
        meta[sh["name"]] = dict(br.metrics(), t=lt, dur=sh["dur"])
        print(f"{sh['name']:16s} {sh['start']:6.2f}-{sh['end']:6.2f}  still at {lt:.2f} -> {p}")
    br.close()
    old = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    old.update(meta)
    (d / "metrics.json").write_text(json.dumps(old, indent=1))
    print(f"total {total:.2f} s")


def cmd_mux(a, W, H):
    shots, total = timeline(a.cut, W)
    meta = json.loads((OUT / "v5_data" / "data.json").read_text())["clips"]
    mix, stems = A.mix(shots, total, SR, {k: {"processed_wav": v.get("processed_wav"), "offset": v.get("offset")} for k, v in meta.items()})
    tag = f"v5_{a.cut}"
    if hasattr(A, "qa_audio"):   # the AUDIO agent's checks; the build fails if they do not pass
        aq = A.qa_audio(mix, stems, shots, SR, total=total)
        (OUT / f"{tag}_{W}x{H}_audio_qa.json").write_text(json.dumps(aq, indent=1, default=str))
        if not aq.get("ok", False):
            print("audio QA failed:", json.dumps(aq, default=str)[:2000]); sys.exit(2)
    sf.write(OUT / f"{tag}_mix.wav", mix, SR)
    for k, v in stems.items():
        sf.write(OUT / f"{tag}_stem_{k}.wav", v, SR)
    d = frame_dir(a.cut, W, H)
    base = "showcase_v6" if a.cut == "v6" else f"showcase_v5{'' if a.cut == 'full' else '_' + a.cut}"
    name = f"{base}{'_square' if W == H else ''}.mp4"
    out = OUT / name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-framerate", str(FPS), "-i", str(d / "%05d.png"), "-i", str(OUT / f"{tag}_mix.wav"),
                    "-vf", f"fade=t=out:st={max(0.0, total - 1.5):.3f}:d=1.5",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-threads", "2",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    tl = [{k: s[k] for k in ("name", "start", "end", "segs", "voice", "clips", "agents", "captions")} for s in shots]
    (OUT / f"{out.stem}.timeline.json").write_text(json.dumps({"total": total, "shots": tl}, indent=1))
    print("wrote", out, f"{total:.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="v6", choices=list(CUTS))
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--frames", action="store_true"); ap.add_argument("--stills", action="store_true")
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--at", type=float, default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--only", default="", help="stills: comma-separated shot names")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    if a.stills:
        cmd_stills(a, W, H)
    if a.frames:
        cmd_frames(a, W, H)
    if a.mux:
        cmd_mux(a, W, H)


if __name__ == "__main__":
    main()
