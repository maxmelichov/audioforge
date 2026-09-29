"""Showcase renderer v2: light, high-contrast, "show, don't say". Split screens replay the recorded E2E sessions
(runs/e2e_final.json per-clip records: cut-in times, response times, first-text latency) of the framework default
stacks next to ours on the same audio, with our live transcript from the recorded server events.

    PYTHONPATH=. .venv/bin/python demo/render_v2.py --cut full --storyboard /Volumes/ExternalSSD/.../storyboard_v2
    PYTHONPATH=. .venv/bin/python demo/render_v2.py --cut short --size 1080x1080 --out $DEMO_OUT/showcase_v2_square.mp4 [--seg i --nseg n | --mux]

Inputs: demo/numbers.json, runs/e2e_final.json, $E2E_TSVAD/clips.json + runs/*.jsonl (system T = served TS-VAD path),
demo/clips/<clip>.mono.wav, demo/events/<clip>.mono.json (demo/record_events.py), $DEMO_OUT/narration_v2_<cut>.json.
Design: near-white background, ink text (>= 7:1), one accent per role (ours = blue, framework defaults = red, decisions
= amber), 12-column grid, 96 px margins, nothing under 28 px at 1080p, 300 ms eases, no flashes, captions in a solid bar.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
E2E_TSVAD = Path(os.environ.get("E2E_TSVAD", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad"))
SCRATCH = OUT_DIR / "scratch"
FPS = 30
SR = 48000

# ----------------------------------------------------------------------------- palette (WCAG: ink 16.5:1, muted 7.1:1, roles >= 5.6:1 on BG)
BG = (246, 247, 249)
PANEL = (255, 255, 255)
PANEL2 = (233, 237, 242)
INK = (17, 24, 39)
MUTED = (75, 85, 99)
GRID = (201, 209, 219)
OURS = (11, 92, 173)        # blue
OURS_L = (214, 229, 247)
BASE = (180, 35, 24)        # red: framework default stacks
BASE_L = (250, 222, 218)
ACCENT = (154, 74, 7)       # amber: decisions / turn end
ACCENT_L = (253, 233, 211)
GOOD = (27, 127, 76)
SPK = [(11, 92, 173), (196, 82, 24), (109, 40, 217), (27, 127, 76)]
BAR = (17, 24, 39)          # caption bar
FONT_UI = "/System/Library/Fonts/Avenir Next.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_IDX = {"heavy": 8, "demi": 2, "medium": 5, "regular": 7}
REPO_URL = "github.com/maxmelichov/audioforge"
_fonts: dict = {}


def font(kind, size):
    key = (kind, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_MONO if kind == "mono" else FONT_UI, size, index=0 if kind == "mono" else FONT_IDX[kind])
    return _fonts[key]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def mix(c1, c2, x):
    return tuple(int(round(a + (b - a) * x)) for a, b in zip(c1, c2))


# ----------------------------------------------------------------------------- data
class ClipAudio:
    def __init__(self, path: Path):
        w, sr = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        self.wav, self.sr, self.dur = w, sr, len(w) / sr
        step = int(sr * 0.004)
        k = len(w) // step
        env = np.abs(w[: k * step]).reshape(k, step).max(1)
        self.env = env / max(1e-6, np.percentile(env, 99.5))
        self.step = 0.004

    def env_at(self, t0, t1, n):
        idx = ((t0 + (t1 - t0) * (np.arange(n) + 0.5) / n) / self.step).astype(int)
        out = np.zeros(n, np.float32)
        ok = (idx >= 0) & (idx < len(self.env))
        out[ok] = self.env[idx[ok]]
        return out


class Events:
    """Recorded server events of one clip (demo/record_events.py): partials, finals, frames, turn ends, enrolled."""

    def __init__(self, path: Path | None):
        self.partials, self.finals, self.turn_ends, self.enrolled, self.languages = [], [], [], [], []
        self.f_t = np.zeros(0); self.f_vad = np.zeros(0); self.f_spk = np.zeros((0, 4)); self.f_prim = np.zeros(0); self.f_eot = np.zeros(0)
        if path is None or not path.exists():
            return
        ev = json.loads(path.read_text())["events"]
        self.partials = [(e["rel_ms"] / 1000, e["msg"]["text"]) for e in ev if e["msg"]["type"] == "partial"]
        self.finals = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "final"]
        self.turn_ends = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "turn_end"]
        self.enrolled = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "enrolled"]
        self.languages = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "language"]
        fr = [e["msg"] for e in ev if e["msg"]["type"] == "frame"]
        if fr:
            self.f_t = np.array([f["t"] for f in fr]); self.f_vad = np.array([f["vad"] for f in fr])
            self.f_spk = np.array([(f["speakers"] + [0, 0, 0, 0])[:4] for f in fr], dtype=np.float32)
            self.f_prim = np.array([-1 if f["primary"] is None else f["primary"] for f in fr])
            self.f_eot = np.array([0.0 if f.get("eot") is None else float(f["eot"]) for f in fr])

    def _vocab(self):
        """Words of the offline Parakeet-TDT rewrites of this recording: the dictionary used to re-join a streaming token
        the RNNT emitted split ("me ssed" -> "messed" only if the TDT final contains "messed")."""
        if not hasattr(self, "_voc"):
            import re
            self._voc = set()
            for f in self.finals:
                if f.get("source") == "tdt_v3":
                    self._voc |= set(re.findall(r"[a-z']+", (f.get("text") or "").lower()))
        return self._voc

    _DICT = None

    @classmethod
    def _dict(cls):
        if cls._DICT is None:
            words = set()
            try:
                words = {w.strip().lower() for w in open("/usr/share/dict/words", errors="ignore")}
            except OSError:
                pass
            cls._DICT = words
        return cls._DICT

    def _mend(self, text):
        """Re-join a streaming token the RNNT emitted split ("me ssed" -> "messed"): only when the second piece is not a
        word and the joined form is one (system dictionary + this recording's TDT rewrite)."""
        voc = self._vocab() | self._dict()
        toks, out = text.split(), []
        i = 0
        while i < len(toks):
            a, b = toks[i], toks[i + 1] if i + 1 < len(toks) else None
            joined = (a + b) if b is not None else ""
            stems = {joined} | {joined[: -len(suf)] for suf in ("ed", "d", "s", "es", "ing") if joined.endswith(suf) and len(joined) > len(suf) + 2}
            if b is not None and b not in voc and len(b) >= 2 and any(w in voc for w in stems):
                out.append(a + b); i += 2
            else:
                out.append(a); i += 1
        return " ".join(out)

    def finals_text(self, now):
        """Accumulated streaming final text of the window so far (same decoder as the partials, so the text reads as
        one continuous transcript; the TDT rewrite is shown in the meeting scene)."""
        return self._mend(" ".join(f["text"].strip() for f in self.finals if f["arr"] <= now and f.get("source", "stream") == "stream" and (f.get("text") or "").strip()))

    def live_text(self, now):
        """Finals so far + the current partial, with the overlap removed (after a streaming final the partial buffer
        still starts with the words that were just finalised)."""
        fin, part = self.finals_text(now), self.partial_at(now)
        if fin and part:
            fw, pw = fin.split(), part.split()
            k = 0
            for n in range(min(len(fw), len(pw)), 0, -1):
                if fw[-n:] == pw[:n]:
                    k = n; break
            part = " ".join(pw[k:])
        return (fin + " " + part).strip(), fin, part

    def partial_at(self, now):
        """The live partial at ``now`` at word granularity: the last token is shown only once the next partial proves it
        is a whole word (it is followed by a space or the text is final), so a split token such as "me ssed" never shows."""
        txt, nxt = "", None
        for arr, text in self.partials:
            if arr <= now:
                txt = text
            else:
                nxt = text
                break
        if not txt or nxt is None:
            return self._mend(txt)
        if nxt.startswith(txt + " ") or not nxt.startswith(txt):
            return self._mend(txt)
        return self._mend(txt.rsplit(" ", 1)[0] if " " in txt else "")


class Data:
    def __init__(self):
        self.N = json.loads((DEMO / "numbers.json").read_text())
        e2e = json.loads((ROOT / "runs" / "e2e_final.json").read_text())
        self.per_clip = e2e["per_clip"]
        self.clips = {c["name"]: c for c in json.loads((E2E_TSVAD / "clips.json").read_text())} if (E2E_TSVAD / "clips.json").exists() else {}
        self.T = {}
        for f in glob.glob(str(E2E_TSVAD / "runs" / "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                self.T[r["clip"]] = r
        self._audio, self._events = {}, {}

    def num(self, k):
        return self.N[k]["value"]

    def src(self, *keys):
        return sorted({self.N[k]["source"] for k in keys})

    def audio(self, clip):
        if clip not in self._audio:
            p = DEMO / "clips" / f"{clip}.mono.wav"
            if not p.exists():
                p = DEMO / "clips" / f"{clip}.wav"
            self._audio[clip] = ClipAudio(p)
        return self._audio[clip]

    def events(self, clip):
        if clip not in self._events:
            p = DEMO / "events" / f"{clip}.mono.json"
            if not p.exists():
                p = DEMO / "events" / f"{clip}.json"
            self._events[clip] = Events(p if p.exists() else None)
        return self._events[clip]

    def session(self, clip, fw, sysm):
        """One side of a split screen: user turns, cut-in times, response times, first-text time, transcript source."""
        meta = self.clips.get(clip, {})
        user_turns = meta.get("user_turns") or []
        if sysm == "T":
            r = self.T[clip]["raw"]
            ends = [{"end": ut[1], "response": next((x for x in r["responses"] if x >= ut[1]), None)} for ut in user_turns]
            return {"user_turns": user_turns, "cut_ins": [x for x in r["responses"] if any(a < x < b for a, b in user_turns)],
                    "ends": ends, "first_text": r["first_text_t"], "text": meta.get("text_mono", ""), "ours": True,
                    "label": "OURS + 5 s VOICE PRINT", "sub": "serve --turn-input tsvad, Pipecat 1.12", "clip": clip}
        v = self.per_clip[f"{meta.get('set', clip.split('_')[0])}|mono|{fw}|{sysm}|{clip}"]
        onset = user_turns[0][0] if user_turns else 0.0
        names = {"A": ("PIPECAT DEFAULT STACK", "Silero VAD + smart-turn v3.2 + Whisper small"),
                 "B": ("LIVEKIT DEFAULT STACK", "Silero VAD + EnglishModel + Whisper small"),
                 "C": ("OURS", f"one streaming model + Nemotron-3, in {fw.capitalize()}"),
                 "D": ("OURS, hybrid rule", f"hybrid_dyn + agent-end arming, in {fw.capitalize()}")}
        return {"user_turns": user_turns, "cut_ins": v["cut_in_t"], "ends": v["ends"],
                "first_text": onset + (v["first_text_ms_after_onset"] or 0) / 1000 if v["first_text_ms_after_onset"] is not None else None,
                "text": meta.get("text_mono", ""), "ours": sysm in ("C", "D"), "label": names[sysm][0], "sub": names[sysm][1], "clip": clip}


# ----------------------------------------------------------------------------- timeline (same contract as render.py)
SCENE_SPEC = {
    "full": {"ex_cutin": {"clip": "ami_IB4002_119", "offset": 0.0, "min_len": 19.0},
             "ex_deadair": {"clip": "ami_IS1008b_003", "offset": 7.0, "min_len": 11.0},
             "ex_partial": {"clip": "ami_ES2011b_027", "offset": 0.0, "min_len": 11.5},
             "multispeaker": {"clip": "ami_IS1008b_003", "events": "ami_IS1008b_003_agentend", "offset": 0.0, "min_len": 16.5},
             "voiceprint": {"clip": "ami_TS3004b_064", "offset": 0.0, "min_len": 19.5},
             "architecture": {"min_len": 14.0}, "end": {"min_len": 5.0}},
    "short": {"ex_cutin": {"clip": "ami_IB4002_119", "offset": 3.0, "min_len": 15.0},
              "ex_deadair": {"clip": "ami_IS1008b_003", "offset": 9.0, "min_len": 9.0},
              "ex_partial": {"clip": "ami_ES2011b_027", "offset": 0.0, "min_len": 10.5},
              "architecture": {"min_len": 8.0}, "end": {"min_len": 4.0}},
}
LEAD_S, GAP_S, SCENE_GAP_S = 0.5, 0.35, 0.5


def build_timeline(narr, cut):
    spec = SCENE_SPEC[cut]
    scenes, t = [], 0.0
    for u in narr["units"]:
        if not scenes or scenes[-1]["name"] != u["scene"]:
            if scenes:
                s = scenes[-1]
                s["end"] = max(s["narr_end"] + SCENE_GAP_S, s["start"] + spec.get(s["name"], {}).get("min_len", 0.0))
                t = s["end"]
            scenes.append({"name": u["scene"], "start": t, "units": [], **spec.get(u["scene"], {})})
            t += LEAD_S
        u = dict(u)
        dur = u["end"] - u["start"]
        u["start"], u["end"] = t, t + dur
        scenes[-1]["units"].append(u)
        scenes[-1]["narr_end"] = t + dur
        t += dur + GAP_S
    s = scenes[-1]
    s["end"] = max(s["narr_end"] + 1.0, s["start"] + spec.get(s["name"], {}).get("min_len", 0.0))
    return scenes


def dry_units(cut, spc=0.085):
    units, scene, t = [], "title", 0.0
    for line in (DEMO / f"script_v2_{cut}.md").read_text().splitlines():
        line = line.strip()
        if line.startswith("## scene:"):
            scene = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            dur = spc * len(line)
            units.append({"scene": scene, "text": line[2:].strip(), "start": t, "end": t + dur, "wav": ""})
            t += dur
    return {"units": units}


def split_caption(text, max_chars):
    import re
    if len(text) <= max_chars:
        return [text]
    sents = re.split(r"(?<=[.?!:;,])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:
        if len(c) <= max_chars:
            out.append(c); continue
        words, cur = c.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                out.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            out.append(cur)
    return out


def captions_for(units, max_chars):
    out = []
    for u in units:
        parts = split_caption(u["text"], max_chars)
        total = sum(len(p) for p in parts)
        t = u["start"]
        for i, p in enumerate(parts):
            d = (u["end"] - u["start"]) * len(p) / total
            out.append((t, t + d + (0.35 if i == len(parts) - 1 else 0.0), p))
            t += d
    return out


# ----------------------------------------------------------------------------- canvas
class Canvas:
    ARROWS = "→↑↓←"

    def __init__(self, W, H):
        self.W, self.H, self.sq = W, H, W == H
        self.s = min(W, H) / 1080
        self.m = self.px(96 if not self.sq else 64)
        self.gutter = self.px(24)
        self.ncol = 12
        self.colw = (W - 2 * self.m - (self.ncol - 1) * self.gutter) / self.ncol
        self.cap_h = self.px(104 if not self.sq else 150)  # caption bar height (reserved at the bottom)
        self.legend_h = self.px(44)

    def px(self, v):
        return int(round(v * self.s))

    def col(self, i, span=1):
        """x0, x1 of columns i..i+span-1 (0-based)."""
        x0 = self.m + i * (self.colw + self.gutter)
        return int(round(x0)), int(round(x0 + span * self.colw + (span - 1) * self.gutter))

    def _runs(self, s, kind, size):
        f, fa = font(kind, self.px(size)), font("mono", self.px(size))
        runs, cur, arrow = [], "", False
        for ch in s:
            a = ch in self.ARROWS
            if cur and a != arrow:
                runs.append((cur, fa if arrow else f)); cur = ""
            cur += ch; arrow = a
        if cur:
            runs.append((cur, fa if arrow else f))
        return runs

    def text(self, d, xy, s, kind="medium", size=32, fill=INK, anchor="la", alpha=1.0, bg=None):
        if alpha <= 0 or not s:
            return
        col = mix(bg or BG, fill, alpha) if alpha < 1 else fill
        runs = self._runs(s, kind, size)
        if len(runs) == 1:
            d.text(xy, s, font=runs[0][1], fill=col, anchor=anchor); return
        total = sum(f.getlength(t) for t, f in runs)
        x, y = xy
        x -= {"l": 0.0, "m": total / 2, "r": total}[anchor[0]]
        for t, f in runs:
            d.text((x, y), t, font=f, fill=col, anchor="l" + anchor[1]); x += f.getlength(t)

    def tw(self, s, kind="medium", size=32):
        return sum(f.getlength(t) for t, f in self._runs(s, kind, size))

    def wrap(self, s, kind, size, width):
        f = font(kind, self.px(size))
        lines, cur = [], ""
        for w in s.split():
            t = (cur + " " + w).strip()
            if f.getlength(t) > width and cur:
                lines.append(cur); cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        return lines

    def para(self, d, xy, s, kind, size, width, fill=INK, alpha=1.0, leading=1.3, bg=None):
        x, y = xy
        for line in self.wrap(s, kind, size, width):
            self.text(d, (x, y), line, kind, size, fill, alpha=alpha, bg=bg)
            y += self.px(size) * leading
        return y

    def panel(self, d, box, fill=PANEL, outline=GRID, radius=None, width=2):
        d.rounded_rectangle(box, radius=radius or self.px(16), fill=fill, outline=outline, width=width)

    def pill(self, d, xy, s, fill, fg=(255, 255, 255), size=28, alpha=1.0):
        x, y = xy
        w = self.tw(s, "demi", size) + self.px(28)
        h = self.px(size) + self.px(16)
        d.rounded_rectangle([x, y, x + w, y + h], radius=h / 2, fill=mix(BG, fill, alpha))
        self.text(d, (x + w / 2, y + h / 2), s, "demi", size, mix(fill, fg, alpha), anchor="mm", bg=fill)
        return w, h

    def source(self, d, srcs, y=None):
        s = "source: " + ", ".join(srcs)
        self.text(d, (self.W - self.m, (y or self.H - self.cap_h) - self.px(12)), s, "mono", 26 if not self.sq else 24, MUTED, anchor="rd")

    def header(self, d, title, sub=None, alpha=1.0):
        y = self.px(64 if not self.sq else 48)
        self.text(d, (self.m, y), title, "heavy", 56 if not self.sq else 44, INK, alpha=alpha)
        y += self.px(72 if not self.sq else 58)
        if sub:
            y = self.para(d, (self.m, y), sub, "medium", 30 if not self.sq else 26, self.W - 2 * self.m, MUTED, alpha=alpha) + self.px(10)
        return y + self.px(12)


def new_frame(cv):
    return Image.new("RGB", (cv.W, cv.H), BG)


# ----------------------------------------------------------------------------- split-screen side
def draw_side(cv, d, box, sess, audio, events, now, *, ours, timer=True, transcript=True, big_text=False):
    """One side: label, waveform of the user's audio (played up to ``now``), user-turn shading, cut-in and answer
    markers, agent row, transcript as received, dead-air timer."""
    x0, y0, x1, y1 = (int(v) for v in box)
    W = x1 - x0
    role = OURS if ours else BASE
    cv.panel(d, box)
    pad = cv.px(24 if not cv.sq else 20)
    sq = cv.sq
    # label
    pw, ph = cv.pill(d, (x0 + pad, y0 + pad), sess["label"], role)
    if not sq:
        cv.text(d, (x0 + pad, y0 + pad + ph + cv.px(10)), sess["sub"], "medium", 28, MUTED, bg=PANEL)
    # waveform window: fixed clip window [0, dur] so both sides align
    wy0 = y0 + pad + ph + (cv.px(98) if not sq else cv.px(44))
    wh = cv.px(150 if not sq else 96)
    dur = audio.dur
    px_per_s = (W - 2 * pad) / dur
    wx0 = x0 + pad
    # user-turn shading
    for a, b in sess["user_turns"]:
        d.rounded_rectangle([wx0 + a * px_per_s, wy0, wx0 + b * px_per_s, wy0 + wh], radius=cv.px(8), fill=PANEL2)
    if sess["user_turns"]:
        a, b = sess["user_turns"][0]
        cv.text(d, (wx0 + a * px_per_s, wy0 - cv.px(6)), "user speaking", "medium", 28, MUTED, anchor="ld", bg=PANEL)
    n = W - 2 * pad
    env = audio.env_at(0, dur, n)
    mid = wy0 + wh // 2
    for i in range(n):
        t = (i + 0.5) / px_per_s
        h = max(1, int(env[i] * wh * 0.42))
        col = INK if t <= now else GRID
        d.line([(wx0 + i, mid - h), (wx0 + i, mid + h)], fill=col)
    # playhead
    xp = wx0 + min(now, dur) * px_per_s
    d.line([(xp, wy0 - cv.px(6)), (xp, wy0 + wh + cv.px(6))], fill=INK, width=3)
    # agent row
    ay = wy0 + wh + cv.px(18 if not sq else 10)
    ah = cv.px(40 if not sq else 34)
    cv.text(d, (wx0, ay + ah / 2), "agent", "medium", 28, MUTED, anchor="lm", bg=PANEL)
    ax0 = wx0 + cv.px(96)
    d.line([(ax0, ay + ah / 2), (wx0 + n, ay + ah / 2)], fill=GRID, width=2)
    n_cut = 0
    for tci in sess["cut_ins"]:
        if tci <= now:
            n_cut += 1
            a = ease((now - tci) / 0.3)
            x = wx0 + tci * px_per_s
            if x + cv.px(10) <= wx0 + n:
                d.rounded_rectangle([x, ay + ah * (1 - a) / 2, x + cv.px(10), ay + ah - ah * (1 - a) / 2], radius=cv.px(4), fill=BASE)
            d.line([(x + cv.px(5), wy0), (x + cv.px(5), ay)], fill=BASE, width=2)
    for e in sess["ends"]:
        r = e.get("response")
        if r is not None and r <= now:
            a = ease((now - r) / 0.3)
            x = wx0 + r * px_per_s
            xe = min(wx0 + n, x + (now - r) * px_per_s + cv.px(10))
            if xe > x + 2:
                d.rounded_rectangle([x, ay + ah * (1 - a) / 2, xe, ay + ah - ah * (1 - a) / 2], radius=cv.px(4), fill=role)
            d.line([(x + cv.px(5), wy0), (x + cv.px(5), ay)], fill=role, width=2)
    # badges: cut-ins and dead-air timer
    by = ay + ah + cv.px(18 if not sq else 12)
    if n_cut:
        cv.pill(d, (wx0, by), f"cut-in ×{n_cut}: answers while the user is still speaking" if not sq else f"cut-in ×{n_cut}", BASE, size=28)
    if timer and sess["ends"]:
        e = sess["ends"][0]
        end, r = e["end"], e.get("response")
        if now >= end:
            if r is None or now < r:
                val, lab, col = now - end, "waiting …", ACCENT
            else:
                val, lab, col = r - end, "answered after", role
            if r is None and now - end > 6.0:
                val, lab, col = 6.0, "no answer within", BASE
            if sq:
                cv.text(d, (x1 - pad, y0 + pad + ph / 2), f"{lab} {val:.1f} s", "heavy", 32, col, anchor="rm", bg=PANEL)
            else:
                cv.text(d, (x1 - pad, y0 + pad), lab, "medium", 28, MUTED, anchor="ra", bg=PANEL)
                cv.text(d, (x1 - pad, y0 + pad + cv.px(30)), f"{val:.1f} s", "heavy", 64, col, anchor="ra", bg=PANEL)
    # transcript
    if transcript:
        ft = sess.get("first_text")
        size = 40 if big_text and not sq else (32 if not sq else 30)
        fin_txt = ""
        if ours and events.partials:
            fin_txt = events.finals_text(now)
            txt = events.partial_at(now)
            state = "text" if (txt or fin_txt) else "dots"
        else:
            txt = sess["text"]
            if ft is not None and now >= ft:
                words = txt.split()
                txt = " ".join(words[: max(1, int(len(words) * min(1.0, (now - ft) / 0.4)))]) if not ours else txt
                state = "text"
            else:
                state = "none"
        if sq:  # one line: the tail of the transcript, plus the first-text pill in the badge row
            ty = y1 - pad - cv.px(size) - cv.px(6)
            if state == "text":
                full = (fin_txt + " " + txt).strip() if ours else txt
                f = font("medium", cv.px(size))
                cut = full
                while f.getlength(cut) > n - cv.px(40) and " " in cut:
                    cut = cut.split(" ", 1)[1]
                cv.text(d, (wx0, ty), ("… " if cut != full else "") + cut + (" |" if ours else ""), "medium", size, INK, bg=PANEL)
            else:
                cv.text(d, (wx0, ty), "(nothing yet)" if state == "none" else "…", "medium", size, GRID, bg=PANEL)
            if ft is not None and sess["user_turns"]:
                onset = sess["user_turns"][0][0]
                bx = wx0 + (cv.tw(f"cut-in ×{n_cut}", "demi", 28) + cv.px(44) if n_cut else 0)
                if now >= ft:
                    cv.pill(d, (bx, by), f"first text {ft - onset:.1f} s", role if ours else BASE, size=28)
                elif now >= onset:
                    cv.text(d, (bx, by + cv.px(10)), f"waiting for text … {now - onset:.1f} s", "medium", 28, MUTED, bg=PANEL)
        else:
            ty = by + cv.px(70)
            cv.text(d, (wx0, ty), "TRANSCRIPT AS RECEIVED", "demi", 28, MUTED, bg=PANEL)
            ty += cv.px(40)
            if state == "text":
                if ours:
                    ty = draw_rich(cv, d, (wx0, ty), [(fin_txt, MUTED), (txt + " |", INK)], "medium", size, n, bg=PANEL, max_y=y1 - pad - cv.px(70))
                else:
                    cv.para(d, (wx0, ty), txt, "medium", size, n, INK, bg=PANEL)
            else:
                cv.text(d, (wx0, ty), "(nothing yet)" if state == "none" else "…", "medium", size, GRID, bg=PANEL)
            if not ours:
                cv.para(d, (wx0, y1 - pad - cv.px(120)), "baseline text = reference transcript shown at its measured arrival time (session text not retained)", "medium", 24, n, MUTED, bg=PANEL)
            if ft is not None and sess["user_turns"]:
                onset = sess["user_turns"][0][0]
                if now >= ft:
                    cv.pill(d, (wx0, y1 - pad - cv.px(52)), f"first text {ft - onset:.1f} s after speech onset", role if ours else BASE, size=28)
                elif now >= onset:
                    cv.text(d, (wx0, y1 - pad - cv.px(40)), f"waiting for text … {now - onset:.1f} s", "medium", 28, MUTED, bg=PANEL)


def draw_rich(cv, d, xy, runs, kind, size, width, bg=None, leading=1.3, max_y=None):
    """Wrapped paragraph of (text, colour) runs; keeps the LAST lines when the block would overflow max_y."""
    f = font(kind, cv.px(size))
    words = []
    for text, col in runs:
        words += [(w, col) for w in text.split()]
    lines, cur, cur_w = [], [], 0.0
    sp = f.getlength(" ")
    for w, col in words:
        ww = f.getlength(w)
        if cur and cur_w + sp + ww > width:
            lines.append(cur); cur, cur_w = [(w, col)], ww
        else:
            cur.append((w, col)); cur_w += (sp if cur_w else 0) + ww
    if cur:
        lines.append(cur)
    lh = cv.px(size) * leading
    x, y = xy
    if max_y is not None:
        keep = max(1, int((max_y - y) // lh))
        if len(lines) > keep:
            lines = lines[-keep:]
            lines[0] = [("…", MUTED)] + lines[0]
    for line in lines:
        xx = x
        for w, col in line:
            cv.text(d, (xx, y), w, kind, size, col, bg=bg)
            xx += f.getlength(w) + sp
        y += lh
    return y


def draw_legend(cv, d, items):
    y = cv.H - cv.cap_h - cv.legend_h
    x = cv.m
    for col, lab in items:
        d.rounded_rectangle([x, y + cv.px(10), x + cv.px(22), y + cv.px(32)], radius=cv.px(4), fill=col)
        cv.text(d, (x + cv.px(32), y + cv.px(21)), lab, "medium", 28, MUTED, anchor="lm")
        x += cv.px(32) + cv.tw(lab, "medium", 28) + cv.px(40)


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, cut, W, H, narr, data: Data):
        self.cut, self.cv, self.D = cut, Canvas(W, H), data
        self.scenes = build_timeline(narr, cut)
        self.duration = self.scenes[-1]["end"]
        units = [u for s in self.scenes for u in s["units"]]
        self.captions = captions_for(units, 60 if not self.cv.sq else 40)

    def frame(self, T):
        cv = self.cv
        img = new_frame(cv)
        scene = next((s for s in self.scenes if s["start"] <= T < s["end"]), self.scenes[-1])
        st = T - scene["start"]
        ui, prog = -1, 0.0
        for i, u in enumerate(scene["units"]):
            if T >= u["start"]:
                ui, prog = i, min(1.0, (T - u["start"]) / max(0.01, u["end"] - u["start"]))
        getattr(self, "scene_" + scene["name"])(img, scene, st, ui, prog, T)
        self.draw_caption(img, T)
        return img

    def draw_caption(self, img, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        d.rectangle([0, cv.H - cv.cap_h, cv.W, cv.H], fill=BAR)
        cap = next((c for c in self.captions if c[0] <= T < c[1]), None)
        if not cap:
            return
        size = 40 if not cv.sq else 36
        lines = cv.wrap(cap[2], "demi", size, cv.W - 2 * cv.m)
        lh = cv.px(size) * 1.25
        y = cv.H - cv.cap_h + (cv.cap_h - lh * len(lines)) / 2
        for l in lines:
            cv.text(d, (cv.W / 2, y), l, "demi", size, (255, 255, 255), anchor="ma", bg=BAR)
            y += lh

    # -- split screen scenes -------------------------------------------------------------------------------
    def _split(self, img, scene, st, left, right, legend, *, big_text=False, timer=True, top_extra=0, sources=None):
        cv = self.cv
        d = ImageDraw.Draw(img)
        clip = scene["clip"]
        now = scene["offset"] + st
        audio = self.D.audio(clip)
        ev = self.D.events(scene.get("events", clip))
        top = cv.px(150 if not cv.sq else 130) + top_extra
        bottom = cv.H - cv.cap_h - cv.legend_h - cv.px(16)
        if cv.sq:
            h = int((bottom - top - cv.gutter) / 2)
            boxes = [(cv.m, top, cv.W - cv.m, top + h), (cv.m, top + h + cv.gutter, cv.W - cv.m, bottom)]
        else:
            lx = cv.col(0, 6); rx = cv.col(6, 6)
            boxes = [(lx[0], top, lx[1], bottom), (rx[0], top, rx[1], bottom)]
        draw_side(cv, d, boxes[0], left, audio, ev, now, ours=left["ours"], big_text=big_text, timer=timer)
        draw_side(cv, d, boxes[1], right, audio, ev, now, ours=right["ours"], big_text=big_text, timer=timer)
        meta = self.D.clips.get(clip, {})
        if cv.sq:
            cv.text(d, (cv.m, cv.px(64)), f"AMI {meta.get('conversation', '')} · same audio into both · t = {now:4.1f} s", "medium", 30, MUTED)
            draw_legend(cv, d, legend[:3])
            cv.text(d, (cv.W - cv.m, cv.H - cv.cap_h - cv.legend_h + cv.px(21)), "source: runs/e2e_final.json", "mono", 24, MUTED, anchor="rm")
        else:
            cv.text(d, (cv.m, cv.px(64)), f"AMI meeting {meta.get('conversation', '')} · {audio.dur:.0f} s window · same audio into both · live at 1x · t = {now:4.1f} s",
                    "medium", 30, MUTED)
            draw_legend(cv, d, legend)
            cv.source(d, sources or ["runs/e2e_final.json (per-clip session records)", "demo/events/*.json"], y=cv.H - cv.cap_h - cv.legend_h)

    LEGEND = [(BASE, "default stack"), (OURS, "ours"), (PANEL2, "user speaking"), (INK, "audio played so far")]

    def scene_ex_cutin(self, img, scene, st, ui, prog, T):
        D = self.D
        d = ImageDraw.Draw(img)
        self.cv.text(d, (self.cv.m, self.cv.px(20)), "EXAMPLE 1 · CUT-INS", "demi", 28, BASE)
        self._split(img, scene, st, D.session(scene["clip"], "pipecat", "A"), D.session(scene["clip"], "pipecat", "C"), self.LEGEND)

    def scene_ex_deadair(self, img, scene, st, ui, prog, T):
        D = self.D
        d = ImageDraw.Draw(img)
        self.cv.text(d, (self.cv.m, self.cv.px(20)), "EXAMPLE 2 · DEAD AIR AFTER THE USER STOPS", "demi", 28, ACCENT)
        self._split(img, scene, st, D.session(scene["clip"], "pipecat", "A"), D.session(scene["clip"], "pipecat", "C"), self.LEGEND)

    def scene_ex_partial(self, img, scene, st, ui, prog, T):
        D = self.D
        d = ImageDraw.Draw(img)
        self.cv.text(d, (self.cv.m, self.cv.px(20)), "EXAMPLE 3 · FIRST WORDS", "demi", 28, OURS)
        self._split(img, scene, st, D.session(scene["clip"], "livekit", "B"), D.session(scene["clip"], "livekit", "C"), self.LEGEND, big_text=True, timer=False)

    def scene_voiceprint(self, img, scene, st, ui, prog, T):
        D = self.D
        d = ImageDraw.Draw(img)
        cv = self.cv
        cv.text(d, (cv.m, cv.px(20)), "EXAMPLE 5 · THE SAME MODEL WITH A 5 s VOICE PRINT (served TS-VAD path · one live clip)", "demi", 28, ACCENT)
        left = D.session(scene["clip"], "pipecat", "C"); left["label"] = "OURS, NO VOICE PRINT"; left["sub"] = "label-free primary speaker (product default C)"
        right = D.session(scene["clip"], "pipecat", "T")
        band = cv.px(120)
        self._split(img, scene, st, left, right, [(OURS, "ours"), (BASE, "cut-in"), (PANEL2, "user speaking")], top_extra=band,
                    sources=["runs/e2e_final.json (C)", "e2e_tsvad runs (T)", "runs/improve_115m.json"])
        N = D.num
        a = ease((T - scene["units"][1]["start"]) / 0.3) if len(scene["units"]) > 1 else 0.0
        box = (cv.m, cv.px(150) - cv.px(10), cv.W - cv.m, cv.px(150) + band - cv.px(26))
        cv.panel(d, box, fill=mix(BG, ACCENT_L, a), outline=mix(BG, ACCENT, a))
        if a > 0:
            cv.text(d, (box[0] + cv.px(24), (box[1] + box[3]) / 2), "offline benchmark (this is one live clip) · missed turn ends at 6 s:", "medium", 30, mix(ACCENT_L, MUTED, a), anchor="lm", bg=ACCENT_L)
            cv.text(d, (box[2] - cv.px(24), (box[1] + box[3]) / 2),
                    f"AMI {N('tsvad_turn/ami/hybrid_sortformer'):.1f} % → {N('tsvad_turn/ami/hybrid_tsvad'):.1f} %     held-out ICSI {N('tsvad_turn/icsi/hybrid_sortformer'):.1f} % → {N('tsvad_turn/icsi/hybrid_tsvad'):.1f} %",
                    "heavy", 36, mix(ACCENT_L, INK, a), anchor="rm", bg=ACCENT_L)

    # -- multi-speaker live view -----------------------------------------------------------------------------
    def scene_multispeaker(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        clip = scene["clip"]
        now = scene["offset"] + st
        audio, ev = self.D.audio(clip), self.D.events(scene.get("events", clip))
        cv.text(d, (cv.m, cv.px(20)), "EXAMPLE 4 · A MEETING: WHO HOLDS THE FLOOR, AND WHEN THE TURN ENDS", "demi", 28, OURS)
        top = cv.px(80)
        bottom = cv.H - cv.cap_h - cv.legend_h - cv.px(16)
        box = (cv.m, top, cv.W - cv.m, bottom)
        cv.panel(d, box)
        pad = cv.px(24)
        pw, ph = cv.pill(d, (box[0] + pad, top + pad), "OURS · LIVE", OURS)
        cv.text(d, (box[0] + pad + pw + cv.px(16), top + pad + ph / 2), "stage1_served + Nemotron-3 diarizer · hybrid_dyn · user bound after the agent's TTS end", "medium", 28, MUTED, anchor="lm", bg=PANEL)
        wy0 = top + pad + ph + cv.px(24)
        wh = cv.px(160)
        wx0, n = box[0] + pad, box[2] - box[0] - 2 * pad
        px_per_s = n / audio.dur
        env = audio.env_at(0, audio.dur, n)
        mid = wy0 + wh // 2
        for i in range(n):
            t = (i + 0.5) / px_per_s
            h = max(1, int(env[i] * wh * 0.42))
            d.line([(wx0 + i, mid - h), (wx0 + i, mid + h)], fill=INK if t <= now else GRID)
        # speaker bands from the recorded frames (only frames that have arrived)
        by0 = wy0 + wh + cv.px(20)
        row = cv.px(30)
        labels = ["VAD"] + [f"speaker {k}" for k in range(4)]
        for k, lab in enumerate(labels):
            cv.text(d, (wx0, by0 + row * k + row / 2), lab, "medium", 28, MUTED if k == 0 else SPK[k - 1], anchor="lm", bg=PANEL)
        lx = wx0 + cv.px(150)
        vis = ev.f_t <= now
        for t, v, s, p in zip(ev.f_t[vis], ev.f_vad[vis], ev.f_spk[vis], ev.f_prim[vis]):
            xa, xb = lx + max(0.0, t - 0.08) * (n - cv.px(150)) / audio.dur, lx + t * (n - cv.px(150)) / audio.dur
            if v > 0.5:
                d.rectangle([xa, by0 + 4, xb, by0 + row - 6], fill=mix(PANEL, INK, 0.35 + 0.65 * float(v)))
            for k in range(4):
                if s[k] > 0.5:
                    yy = by0 + row * (k + 1)
                    d.rectangle([xa, yy + 4, xb, yy + row - 6], fill=mix(PANEL, SPK[k], 0.45 + 0.55 * float(s[k])))
            if p >= 0:
                yy = by0 + row * (p + 1)
                d.rectangle([xa, yy + row - 8, xb, yy + row - 5], fill=INK)
        bands_bottom = by0 + row * 5 + cv.px(8)
        def marker(t, lab, col, dy):
            x = lx + t * (n - cv.px(150)) / audio.dur
            d.line([(x, wy0), (x, bands_bottom)], fill=col, width=3)
            w = cv.tw(lab, "demi", 28) + cv.px(28)
            px_ = x + cv.px(8) if x + cv.px(8) + w < box[2] - pad else x - cv.px(8) - w
            cv.pill(d, (px_, wy0 + dy), lab, col, size=28)
        for e in ev.enrolled:
            if e["arr"] <= now:
                marker(e["t"], f"user bound: speaker {e['column']}", GOOD, cv.px(6))
        for e in ev.turn_ends:
            if e["arr"] <= now:
                marker(e["t"], "turn end", ACCENT, cv.px(62))
        xp = lx + now * (n - cv.px(150)) / audio.dur
        d.line([(xp, wy0 - cv.px(6)), (xp, bands_bottom)], fill=INK, width=3)
        # transcript
        ty = bands_bottom + cv.px(24)
        cv.text(d, (wx0, ty), "LIVE PARTIAL", "demi", 28, MUTED, bg=PANEL)
        ty = cv.para(d, (wx0, ty + cv.px(38)), (ev.partial_at(now) or "…") + " |", "medium", 34, n, INK, bg=PANEL) + cv.px(10)
        fins = [f for f in ev.finals if f["arr"] <= now and f.get("source") == "tdt_v3" and (f.get("text") or "").strip()]
        if fins:
            f = fins[-1]
            cv.text(d, (wx0, ty), f"FINAL, rewritten by Parakeet-TDT 0.6B v3 at t = {f['t']:.1f} s", "demi", 28, GOOD, bg=PANEL)
            cv.para(d, (wx0, ty + cv.px(38)), f["text"], "medium", 32, n, INK, bg=PANEL)
        draw_legend(cv, d, [(INK, "VAD"), (SPK[0], "speaker 0"), (SPK[1], "speaker 1"), (GOOD, "user bound"), (ACCENT, "turn end")])
        cv.source(d, ["demo/events/ami_IS1008b_003_agentend.json (recorded from audioforge.serve, agent_end at 3.12 s)"], y=cv.H - cv.cap_h - cv.legend_h)

    # -- title / stacks / architecture / numbers / limits / end ----------------------------------------------
    def scene_title(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        a = ease(st / 0.3)
        y = cv.px(60 if not cv.sq else 40)
        cv.text(d, (cv.m, y), "Today we are presenting", "medium", 36 if not cv.sq else 30, MUTED, alpha=a)
        cv.text(d, (cv.m, y + cv.px(56)), "audioforge", "heavy", 120 if not cv.sq else 88, INK, alpha=a)
        cv.para(d, (cv.m, y + cv.px(190 if not cv.sq else 150)), "One streaming model that does a voice agent's listening: the words, the speech, who is talking, and when your turn is over.",
                "medium", 40 if not cv.sq else 32, cv.col(0, 10)[1] - cv.m, INK, alpha=a)
        # four chips
        cy = y + cv.px(320 if not cv.sq else 280)
        x = cv.m
        for i, (lab, col) in enumerate([("transcript", OURS), ("voice activity", INK), ("speaker", SPK[2]), ("end of turn", ACCENT)]):
            ai = ease((st - 0.15 * i) / 0.3)
            w, h = cv.pill(d, (x, cy), lab, col, size=32 if not cv.sq else 28, alpha=ai)
            x += w + cv.px(16)
            if cv.sq and i == 1:
                x = cv.m; cy += h + cv.px(12)
        # live event strip from a real recording, building while the pills appear
        clip = "ami_IS1008b_003"
        audio, ev = self.D.audio(clip), self.D.events("ami_IS1008b_003_agentend")
        sy0 = cy + cv.px(90 if not cv.sq else 80)
        sy1 = cv.H - cv.cap_h - cv.px(70)
        box = (cv.m, sy0, cv.W - cv.m, sy1)
        cv.panel(d, box)
        pad = cv.px(20)
        wx0, n = box[0] + pad, box[2] - box[0] - 2 * pad
        now = min(audio.dur, st * 1.0)
        wh = int((sy1 - sy0) * 0.42)
        wy0 = sy0 + pad + cv.px(44)
        cv.text(d, (wx0, sy0 + pad), f"live, from a recorded session: AMI meeting, t = {now:4.1f} s", "medium", 28, MUTED, bg=PANEL)
        px_per_s = n / audio.dur
        env = audio.env_at(0, audio.dur, n)
        mid = wy0 + wh // 2
        for i in range(n):
            tt = (i + 0.5) / px_per_s
            h = max(1, int(env[i] * wh * 0.42))
            d.line([(wx0 + i, mid - h), (wx0 + i, mid + h)], fill=INK if tt <= now else GRID)
        by0 = wy0 + wh + cv.px(10)
        row = cv.px(18)
        vis = ev.f_t <= now
        for tt, v, sp, pr in zip(ev.f_t[vis], ev.f_vad[vis], ev.f_spk[vis], ev.f_prim[vis]):
            xa, xb = wx0 + max(0.0, tt - 0.08) * px_per_s, wx0 + tt * px_per_s
            for k in range(2):
                if sp[k] > 0.5:
                    d.rectangle([xa, by0 + row * k + 2, xb, by0 + row * (k + 1) - 3], fill=mix(PANEL, SPK[k], 0.45 + 0.55 * float(sp[k])))
        for e in ev.enrolled:
            if e["arr"] <= now:
                x = wx0 + e["t"] * px_per_s
                d.line([(x, wy0), (x, by0 + 2 * row)], fill=GOOD, width=3)
                cv.pill(d, (x + cv.px(6), wy0 - cv.px(4)), "speaker bound", GOOD, size=28)
        for e in ev.turn_ends:
            if e["arr"] <= now:
                x = wx0 + e["t"] * px_per_s
                d.line([(x, wy0), (x, by0 + 2 * row)], fill=ACCENT, width=3)
                cv.pill(d, (x - cv.px(6) - cv.tw("turn end", "demi", 28) - cv.px(28), wy0 - cv.px(4)), "turn end", ACCENT, size=28)
        xp = wx0 + now * px_per_s
        d.line([(xp, wy0 - cv.px(4)), (xp, by0 + 2 * row)], fill=INK, width=3)
        ptxt = ev.partial_at(now)
        if ptxt:
            f = font("medium", cv.px(30))
            while f.getlength(ptxt) > n - cv.px(160) and " " in ptxt:
                ptxt = ptxt.split(" ", 1)[1]
            cv.text(d, (wx0, by0 + 2 * row + cv.px(10)), "partial: " + ptxt + " |", "medium", 30, INK, bg=PANEL)
        cv.text(d, (cv.m, cv.H - cv.cap_h - cv.px(24)), "one frozen NVIDIA streaming encoder · small heads · NVIDIA's diarizer beside it · two laptop CPU threads", "medium", 30 if not cv.sq else 24, MUTED, anchor="ls", alpha=a)

    def _chain_box(self, d, box, title, sub, col, fill, a):
        cv = self.cv
        if a <= 0:
            return
        cv.panel(d, box, fill=mix(BG, fill, a), outline=mix(BG, col, a), width=3)
        cx = (box[0] + box[2]) / 2
        cv.text(d, (cx, box[1] + cv.px(18)), title, "demi", 32 if not cv.sq else 28, mix(fill, col, a), anchor="ma", bg=fill)
        yy = box[1] + cv.px(62)
        for s in sub:
            cv.text(d, (cx, yy), s, "medium", 28, mix(fill, INK, a), anchor="ma", bg=fill); yy += cv.px(34)

    def _arrow(self, d, p0, p1, col, a=1.0):
        cv = self.cv
        if a <= 0:
            return
        c = mix(BG, col, a)
        d.line([p0, p1], fill=c, width=max(3, cv.px(4)))
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0]); L = cv.px(18)
        d.polygon([p1, (p1[0] - L * math.cos(ang - 0.5), p1[1] - L * math.sin(ang - 0.5)), (p1[0] - L * math.cos(ang + 0.5), p1[1] - L * math.sin(ang + 0.5))], fill=c)

    def scene_stacks(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = cv.header(d, "What exists today, and ours", "the framework default stacks chain separate models; ours is one pass plus a diarizer")
        if cv.sq:
            self._stacks_square(d, top, st, ui); return
        # left: chained stack, top to bottom
        lx0, lx1 = cv.col(0, 5)
        cv.pill(d, (lx0, top), "PIPECAT / LIVEKIT DEFAULT STACKS", BASE)
        bw, bh, gap = lx1 - lx0, cv.px(104), cv.px(30)
        y = top + cv.px(70)
        chain = [("audio, 16 kHz", [], MUTED, PANEL), ("Silero VAD  ·  0.24 M", ["speech / silence"], BASE, BASE_L),
                 ("turn detector", ["smart-turn v3.2 (Pipecat) · EnglishModel (LiveKit)"], BASE, BASE_L),
                 ("Whisper small  ·  244 M, offline", ["the words, after the turn is declared over"], BASE, BASE_L)]
        for i, (t, sub, c, f) in enumerate(chain):
            a = ease((st - 0.2 * i) / 0.3)
            h = bh if sub else cv.px(56)
            self._chain_box(d, (lx0, y, lx0 + bw, y + h), t, sub, c, f, a)
            if i < len(chain) - 1:
                self._arrow(d, (lx0 + bw / 2, y + h), (lx0 + bw / 2, y + h + gap - cv.px(4)), MUTED, ease((st - 0.2 * (i + 1)) / 0.3))
            y += h + gap
        a = ease((st - 0.9) / 0.3)
        cv.para(d, (lx0, y + cv.px(6)), "three models, three delays. Nobody in the chain knows who is speaking, and the words arrive only after the turn is declared over.", "medium", 30, bw, MUTED, alpha=a)
        # right: ours
        rx0, rx1 = cv.col(6, 6)
        a2 = ease((st - 1.2) / 0.3)
        cv.pill(d, (rx0, top), "OURS", OURS, alpha=a2)
        enc = (rx0, top + cv.px(70), rx1, top + cv.px(70) + cv.px(150))
        self._chain_box(d, enc, "one frozen NVIDIA streaming encoder  ·  115 M", ["FastConformer, cache-aware, loaded without NeMo", "one pass every 160 ms"], OURS, OURS_L, a2)
        heads = [("VAD", INK), ("speaker", SPK[2]), ("turn", ACCENT), ("words", OURS)]
        hw = (rx1 - rx0 - 3 * cv.px(16)) / 4
        hy = enc[3] + cv.px(50)
        for i, (t, c) in enumerate(heads):
            ai = ease((st - 1.5 - 0.15 * i) / 0.3)
            hx = rx0 + i * (hw + cv.px(16))
            self._arrow(d, (hx + hw / 2, enc[3]), (hx + hw / 2, hy - cv.px(4)), c, ai)
            cv.panel(d, (hx, hy, hx + hw, hy + cv.px(64)), fill=mix(BG, PANEL, ai), outline=mix(BG, c, ai), width=3)
            cv.text(d, (hx + hw / 2, hy + cv.px(32)), t + " head", "demi", 30, mix(PANEL, c, ai), anchor="mm", bg=PANEL)
        a3 = ease((st - 2.2) / 0.3)
        dy = hy + cv.px(64) + cv.px(34)
        self._chain_box(d, (rx0, dy, rx1, dy + bh), "NVIDIA Nemotron-3 diarizer beside it  ·  100 M", ["who is speaking, on the same 80 ms clock"], SPK[1], PANEL, a3)
        cv.para(d, (rx0, dy + bh + cv.px(40)), "one pass every 160 ms. Partial words while the user speaks, and a turn decision that follows the bound speaker.", "medium", 30, rx1 - rx0, MUTED, alpha=a3)
        cv.source(d, ["research/E2E_FINAL.md §2 (systems A, B, C)", "audioforge/serve.py"])

    def _stacks_square(self, d, top, st, ui):
        cv = self.cv
        w = cv.W - 2 * cv.m
        cv.pill(d, (cv.m, top), "DEFAULT STACKS", BASE)
        y = top + cv.px(64)
        bw = (w - 2 * cv.px(16)) / 3; bh = cv.px(110)
        for i, (t, s) in enumerate([("Silero VAD", ["speech / silence"]), ("turn detector", ["smart-turn · EnglishModel"]), ("Whisper small", ["words after the turn"])]):
            a = ease((st - 0.2 * i) / 0.3); x = cv.m + i * (bw + cv.px(16))
            self._chain_box(d, (x, y, x + bw, y + bh), t, s, BASE, BASE_L, a)
        y2 = y + bh + cv.px(60)
        a2 = ease((st - 1.0) / 0.3)
        cv.pill(d, (cv.m, y2), "OURS", OURS, alpha=a2)
        self._chain_box(d, (cv.m, y2 + cv.px(64), cv.W - cv.m, y2 + cv.px(64) + cv.px(150)), "one frozen NVIDIA encoder, one pass / 160 ms", ["VAD · speaker · turn · words from the same pass", "+ NVIDIA Nemotron-3 diarizer beside it"], OURS, OURS_L, a2)

    def scene_architecture(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = cv.header(d, "How it is built", "one frozen NVIDIA streaming encoder · heads trained by us · NVIDIA's diarizer and Parakeet-TDT beside it · events out to Pipecat / LiveKit")
        if cv.sq:
            self._stacks_square(d, top, st, ui); return
        bottom = cv.H - cv.cap_h - cv.px(70)
        y = top + cv.px(10)
        H = bottom - y
        ax0, ax1 = cv.col(0, 2); ex0, ex1 = cv.col(2, 3); hx0, hx1 = cv.col(5, 4); ox0, ox1 = cv.col(9, 3)
        enc_h = int(H * 0.62)
        a1 = ease(st / 0.3)
        self._chain_box(d, (ax0, y + enc_h // 2 - cv.px(80), ax1, y + enc_h // 2 + cv.px(80)), "16 kHz audio", ["160 ms chunks", "log-mel 80"], MUTED, PANEL, a1)
        a2 = ease((st - 0.4) / 0.3)
        self._arrow(d, (ax1, y + enc_h // 2), (ex0 - cv.px(4), y + enc_h // 2), MUTED, a2)
        enc = (ex0, y, ex1, y + enc_h)
        self._chain_box(d, enc, "FastConformer encoder", ["", "NVIDIA streaming weights,", "kept frozen · 115 M params", "", "cache-aware, one pass", "per 160 ms chunk", "80 ms output frames", "", "loaded without NeMo"], OURS, OURS_L, a2)
        heads = [("VAD head", "speech / silence · 33 K", INK), ("speaker head", "block-4 tap, TitaNet-distilled · 0.5 M", SPK[2]),
                 ("turn head", "speaker-conditioned GRU · end of turn", ACCENT), ("RNNT partials", "words while the user speaks", OURS)]
        hgap = cv.px(18)
        hh = (enc_h - 3 * hgap) / 4
        for i, (t, sub, c) in enumerate(heads):
            ai = ease((st - 0.9 - 0.35 * i) / 0.3)
            yy = enc[1] + i * (hh + hgap)
            self._arrow(d, (ex1, (enc[1] + enc[3]) / 2), (hx0 - cv.px(4), yy + hh / 2), c, ai)
            cv.panel(d, (hx0, yy, hx1, yy + hh), fill=mix(BG, PANEL, ai), outline=mix(BG, c, ai), width=3)
            cv.text(d, (hx0 + cv.px(20), yy + hh / 2 - cv.px(34)), t, "demi", 34, mix(PANEL, c, ai), bg=PANEL)
            cv.text(d, (hx0 + cv.px(20), yy + hh / 2 + cv.px(6)), sub, "medium", 28, mix(PANEL, MUTED, ai), bg=PANEL)
        a3 = ease((st - 2.5) / 0.3)
        dy = enc[3] + cv.px(30)
        dh = bottom - dy
        self._chain_box(d, (ax0, dy, ex1, dy + dh), "NVIDIA Nemotron-3 diarizer  ·  100 M", ["beside the encoder, same 80 ms clock", "→ who is speaking"], SPK[1], PANEL, a3)
        a4 = ease((st - 2.9) / 0.3)
        self._chain_box(d, (hx0, dy, hx1, dy + dh), "Parakeet-TDT 0.6B v3", ["rewrites each finished turn,", "in its own process"], GOOD, PANEL, a4)
        a5 = ease((st - 3.3) / 0.3)
        oy = enc[1] + enc_h // 2 - cv.px(110)
        self._arrow(d, (hx1, enc[1] + enc_h // 2), (ox0 - cv.px(4), enc[1] + enc_h // 2), INK, a5)
        self._chain_box(d, (ox0, oy, ox1, oy + cv.px(220)), "events out", ["partial · turn_end · final", "speaker · language", "", "→ Pipecat / LiveKit"], INK, PANEL, a5)
        cv.source(d, ["runs/stage1_served.afm", "runs/nemo_nemotron3_diar.afm", "audioforge/serve.py"])

    def _bars(self, d, top, rows, unit, t, vmax, label_w, bh=None, gap=None):
        cv = self.cv
        x0 = cv.m + label_w
        x1 = cv.W - cv.m - cv.px(200 if not cv.sq else 150)
        bh, gap = bh or cv.px(52), gap or cv.px(22)
        y = top
        for i, (label, val, ours) in enumerate(rows):
            a = ease((t - 0.15 * i) / 0.3)
            col = OURS if ours else BASE
            cv.text(d, (cv.m, y + bh / 2), label, "demi" if ours else "medium", 32, INK if ours else MUTED, anchor="lm")
            w = (x1 - x0) * (val / vmax) * a
            d.rounded_rectangle([x0, y, x0 + max(w, 2), y + bh], radius=cv.px(8), fill=col if ours else mix(BG, BASE, 0.75))
            cv.text(d, (x0 + w + cv.px(16), y + bh / 2), f"{val:.1f}{unit}", "demi", 32, INK if ours else MUTED, anchor="lm", alpha=a)
            y += bh + gap
        return y

    def scene_numbers(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        N = self.D.num
        t2 = T - scene["units"][1]["start"] if len(scene["units"]) > 1 else st
        top = cv.header(d, "Measured, not claimed", f"top: AMI dev, n = {N('turn_n')} turn ends, every system held to ≤ 5 % false cutoffs, 6 s horizon · bottom: 37 recorded conversations live inside Pipecat / LiveKit · lower is better")
        sq = cv.sq
        rows = [("ours: speaker-aware hybrid" if not sq else "ours", N("turn_miss_all/ours_hybrid"), True), ("NVIDIA Parakeet-Realtime-EOU" if not sq else "NVIDIA Parakeet-EOU", N("turn_miss_all/parakeet_eou"), False),
                ("smart-turn v3.2 + Silero (Pipecat)" if not sq else "smart-turn + Silero", N("turn_miss_all/smartturn_silero"), False), ("Silero VAD timeout" if not sq else "Silero timeout", N("turn_miss_all/silero_timeout"), False),
                ("LiveKit turn detector + timeout" if not sq else "LiveKit detector", N("turn_miss_all/livekit_text"), False)]
        cv.text(d, (cv.m, top), "missed turn ends within 6 s", "demi", 32 if not sq else 28, INK)
        y = self._bars(d, top + cv.px(58), rows, " %", st, 100, cv.px(560 if not sq else 330), bh=cv.px(40 if not sq else 34), gap=cv.px(14 if not sq else 10))
        if not sq:
            cv.text(d, (cv.m, y + cv.px(2)), "the margin comes from ends where the next speaker takes the floor within about a second: a speaker-unaware detector sees no silence there", "medium", 28, MUTED)
        y2 = y + cv.px(56 if not sq else 24)
        ci_a, ci_c = N("e2e_cut_ins_per_min/pipecat_default"), N("e2e_cut_ins_per_min/ours_timeout")
        ft_a, ft_b, ft_c = N("e2e_first_text_ms/pipecat_default"), N("e2e_first_text_ms/livekit_default"), N("e2e_first_text_ms/ours_timeout")
        if sq:
            cols = [(cv.m, cv.W - cv.m)]
        else:
            cols = [cv.col(0, 6), cv.col(6, 6)]
        charts = [("cut-ins per minute (Pipecat pipeline, range over clip sets)", [("Pipecat default stack", ci_a, BASE), ("ours", ci_c, OURS)], "", 8.0),
                  ("first partial transcript after speech onset", [("LiveKit default", [v / 1000 for v in ft_b], BASE), ("Pipecat default", [v / 1000 for v in ft_a], BASE), ("ours", [v / 1000 for v in ft_c], OURS)], " s", 12.0)]
        for ci, (title, rws, unit, vmax) in enumerate(charts):
            x0, x1 = cols[ci % len(cols)]
            yy = y2 if not sq or ci == 0 else y2 + cv.px(200)
            cv.text(d, (x0, yy), title, "demi", 30 if not sq else 26, INK, alpha=ease(t2 / 0.3))
            yy += cv.px(46)
            for i, (lab, rng, col) in enumerate(rws):
                a = ease((t2 - 0.15 * i) / 0.3)
                bx0, bx1 = x0, x1 - cv.px(20)
                cv.text(d, (bx0, yy), lab, "medium", 28, MUTED, alpha=a)
                lo, hi = rng
                cv.text(d, (bx1, yy), f"{lo:.1f} – {hi:.1f}{unit}", "demi", 28, col, anchor="ra", alpha=a)
                d.rounded_rectangle([bx0, yy + cv.px(36), bx1, yy + cv.px(56)], radius=cv.px(6), fill=PANEL2)
                d.rounded_rectangle([bx0 + (bx1 - bx0) * lo / vmax, yy + cv.px(36), bx0 + (bx1 - bx0) * min(hi, vmax) / vmax, yy + cv.px(56)], radius=cv.px(6), fill=mix(BG, col, a))
                yy += cv.px(70)
        cv.source(d, self.D.src("turn_miss_all/ours_hybrid", "turn_n") + ["runs/e2e_final.json"])

    def scene_limits(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        N = self.D.num
        top = cv.header(d, "What it is not good at yet", "all measured, research/FINAL_REPORT.md scorecard")
        vap, ours = N("turnbench_p50/vap"), N("turnbench_p50/ours_predictive")
        items = [("polite, not fast", f"TurnBench two-party calls: recall {N('turnbench_recall/ours_predictive'):.2f} vs VAP {N('turnbench_recall/vap'):.2f}, but median latency {ours} ms vs {vap} ms — we wait for the silence, VAP predicts it"),
                 ("speaker verification", f"{N('spk_eer/shipped'):.1f} % EER vs TitaNet-L 8.2 % (AMI, within meeting)"),
                 ("diarization head", "frame DER 0.394 vs Sortformer v2 0.201 — so the product ships NVIDIA's diarizer"),
                 ("language ID head", "75 % at 2 s vs AmberNet 95 % — off by default, AmberNet wired in"),
                 ("floor-open turn ends", f"best rule {N('turn_miss_open/ours_head_or_silero'):.1f} % missed vs a Silero timeout {N('turn_miss_open/silero_timeout'):.1f} % (AMI, n = 236)"),
                 ("compute", "1.7–3.6x the CPU of the default stacks; keeps up at 1x on two threads"),
                 ("cut-ins: fewer, not zero", f"ours {N('e2e_cut_ins_per_min/ours_timeout')[0]:.1f}–{N('e2e_cut_ins_per_min/ours_timeout')[1]:.1f} per minute vs Pipecat default {N('e2e_cut_ins_per_min/pipecat_default')[0]:.1f}–{N('e2e_cut_ins_per_min/pipecat_default')[1]:.1f}; on the clips above, ours (LiveKit) cut in at 4.7 s on IS1008b_003 and at 16.8 s on ES2011b_027:")]
        y = top + cv.px(10)
        for i, (a_, b_) in enumerate(items):
            al = ease((st - 0.25 * i) / 0.3)
            cv.text(d, (cv.m, y), a_, "demi", 32 if not cv.sq else 26, BASE, alpha=al)
            y = cv.para(d, (cv.col(3)[0], y), b_, "medium", 30 if not cv.sq else 24, cv.W - cv.m - cv.col(3)[0], INK, alpha=al) + cv.px(18 if not cv.sq else 8)
        # mini agent rows for the two honest cut-ins
        al = ease((st - 0.25 * len(items)) / 0.3)
        for j, (clip, tci, dur, ut) in enumerate((("IS1008b_003", 4.72, 15.21, (1.28, 13.28)), ("ES2011b_027", 16.78, 20.0, (0.0, 18.0)))):
            x0, x1 = (cv.col(3, 4) if j == 0 else cv.col(8, 4)) if not cv.sq else ((cv.m, cv.W // 2 - cv.px(12)) if j == 0 else (cv.W // 2 + cv.px(12), cv.W - cv.m))
            yy = y + cv.px(6)
            pps = (x1 - x0) / dur
            d.rounded_rectangle([x0 + ut[0] * pps, yy, x0 + ut[1] * pps, yy + cv.px(22)], radius=cv.px(6), fill=mix(BG, PANEL2, al))
            d.line([(x0, yy + cv.px(40)), (x1, yy + cv.px(40))], fill=mix(BG, GRID, al), width=2)
            xc = x0 + tci * pps
            d.rounded_rectangle([xc, yy + cv.px(28), xc + cv.px(8), yy + cv.px(52)], radius=cv.px(3), fill=mix(BG, BASE, al))
            cv.text(d, (x0, yy + cv.px(60)), f"AMI {clip} · ours in LiveKit · cut-in at {tci:.1f} s", "medium", 26, MUTED, alpha=al)
        cv.source(d, self.D.src("turnbench_p50/vap", "turnbench_p50/ours_predictive", "spk_eer/shipped", "turn_miss_open/silero_timeout"))

    def scene_end(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        a = ease(st / 0.3)
        cy = (cv.H - cv.cap_h) / 2
        cv.text(d, (cv.W / 2, cy - cv.px(200)), "audioforge", "heavy", 110 if not cv.sq else 84, INK, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cy - cv.px(60)), "one streaming model · transcript + voice activity + speaker + end of turn", "medium", 34 if not cv.sq else 26, MUTED, anchor="ma", alpha=a)
        cv.panel(d, (cv.m, cy + cv.px(10), cv.W - cv.m, cy + cv.px(120)), fill=mix(BG, PANEL, a), outline=mix(BG, OURS, a), width=3)
        cv.text(d, (cv.W / 2, cy + cv.px(65)), REPO_URL, "mono", 44 if not cv.sq else 30, OURS, anchor="mm", bg=PANEL, alpha=a)
        cv.text(d, (cv.W / 2, cy + cv.px(150)), "code · benchmarks · negative results · corrections log", "medium", 32 if not cv.sq else 26, INK, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cv.H - cv.cap_h - cv.px(20)), "voice-over: Qwen3-TTS 1.7B (Apache-2.0) · NVIDIA weights CC-BY-4.0 · AMI / otoSpeech / LibriSpeech CC-BY-4.0", "mono", 24, MUTED, anchor="md", alpha=a)


# ----------------------------------------------------------------------------- audio (same mix as v1, no music by default)
def resample(x, sr, out_sr):
    if sr == out_sr:
        return x.astype(np.float32)
    n = int(round(len(x) * out_sr / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def build_audio(r: Renderer):
    n = int(math.ceil(r.duration * SR)) + SR
    voice = np.zeros(n, np.float32)
    for s in r.scenes:
        for u in s["units"]:
            w, sr = sf.read(u["wav"], dtype="float32")
            w = resample(w, sr, SR)
            i = int(u["start"] * SR)
            voice[i: i + len(w)] += w[: n - i]
    rms = np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2)) if np.any(np.abs(voice) > 0.01) else 0.1
    voice *= min(6.0, 10 ** (-14 / 20) / max(rms, 1e-6))
    voice = np.tanh(voice * 1.15) / np.tanh(1.15)
    clipmix = np.zeros(n, np.float32)
    for s in r.scenes:
        if "clip" not in s:
            continue
        c = r.D.audio(s["clip"])
        w = resample(c.wav, c.sr, SR)
        off = int(s["offset"] * SR)
        seg = w[off: off + int((s["end"] - s["start"]) * SR)].copy()
        f = min(len(seg), int(0.4 * SR))
        seg[-f:] *= np.linspace(1, 0, f)
        i = int(s["start"] * SR)
        clipmix[i: i + len(seg)] += seg[: n - i] * 10 ** (-8 / 20)
    env = np.convolve(np.abs(voice), np.ones(int(0.05 * SR)) / int(0.05 * SR), mode="same")
    duck = 1 - 0.6 * np.clip(env / 0.08, 0, 1)
    d2 = np.empty_like(duck)
    acc, a_att, a_rel = 1.0, math.exp(-1 / (0.03 * SR)), math.exp(-1 / (0.4 * SR))
    for i in range(len(duck)):
        a = a_att if duck[i] < acc else a_rel
        acc = a * acc + (1 - a) * duck[i]
        d2[i] = acc
    out = voice + clipmix * d2
    out = np.tanh(out * 1.05) / np.tanh(1.05) * 0.95
    return out[: int(r.duration * SR)]


# ----------------------------------------------------------------------------- main
STORYBOARD_T = {  # representative moment of each scene (seconds into the scene) for the still
    "title": 1.5, "stacks": 3.0, "ex_cutin": 14.5, "ex_deadair": 10.2, "ex_partial": 5.0, "multispeaker": 16.0,
    "voiceprint": 18.6, "numbers": 2.0, "architecture": 4.5, "limits": 3.0, "end": 1.0,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", choices=["full", "short"], required=True)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--storyboard", default=None, help="write one PNG per scene (dry timings) into this directory")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--seg", type=int, default=None)
    ap.add_argument("--nseg", type=int, default=1)
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    W, H = (int(v) for v in a.size.lower().split("x"))
    npath = OUT_DIR / f"narration_v2_{a.cut}.json"
    narr = json.loads(npath.read_text()) if npath.exists() and not a.dry else dry_units(a.cut)
    r = Renderer(a.cut, W, H, narr, Data())
    print(f"{a.cut} {W}x{H}: {r.duration:.1f} s, scenes:")
    for s in r.scenes:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  {s['name']}")
    if a.storyboard:
        out = Path(a.storyboard); out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(r.scenes):
            t = s["start"] + min(STORYBOARD_T.get(s["name"], 2.0), s["end"] - s["start"] - 0.1)
            # a second still for two-unit scenes (numbers, voiceprint)
            times = [t] + ([s["units"][1]["start"] + 1.5] if len(s["units"]) > 1 and s["name"] in ("numbers", "voiceprint", "stacks") else [])
            for j, tt in enumerate(times):
                r.frame(tt).save(out / f"{i + 1:02d}{'' if j == 0 else chr(97 + j)}_{s['name']}_{W}x{H}.png")
        print("storyboard ->", out)
        return
    if a.preview is not None:
        r.frame(a.preview).save(str(a.out) + ".png"); print("preview ->", str(a.out) + ".png"); return
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)
    n_frames = int(r.duration * FPS)
    import time
    t0 = time.time()
    if not a.mux:
        lo = 0 if a.seg is None else a.seg * n_frames // a.nseg
        hi = n_frames if a.seg is None else (a.seg + 1) * n_frames // a.nseg
        seg_out = SCRATCH / f"{out.stem}.seg{0 if a.seg is None else a.seg:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-an", "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-profile:v", "high", str(seg_out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(lo, hi):
            p.stdin.write(r.frame(i / FPS).tobytes())
            if (i - lo) % 300 == 0:
                print(f"  frame {i}/{n_frames}  {time.time() - t0:.0f} s", flush=True)
        p.stdin.close(); p.wait()
        if p.returncode:
            sys.exit(f"ffmpeg failed on {seg_out}")
        print("wrote", seg_out)
        if a.seg is not None:
            return
        segs = [seg_out]
    else:
        segs = sorted(SCRATCH.glob(f"{out.stem}.seg*.mp4"))
        if len(segs) != a.nseg:
            sys.exit(f"expected {a.nseg} segments, found {len(segs)}")
    wav_path = SCRATCH / (out.stem + ".mix.wav")
    sf.write(wav_path, build_audio(r), SR)
    lst = SCRATCH / (out.stem + ".segs.txt")
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav_path),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    wav_path.unlink(missing_ok=True)
    print("wrote", out, f"{r.duration:.1f} s in {time.time() - t0:.0f} s")
    out.with_suffix(".timeline.json").write_text(json.dumps({"duration_s": r.duration, "scenes": [
        {"name": s["name"], "start": round(s["start"], 2), "end": round(s["end"], 2), "clip": s.get("clip")} for s in r.scenes]}, indent=1))


if __name__ == "__main__":
    main()
