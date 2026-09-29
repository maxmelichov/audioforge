"""Deterministic renderer for the showcase video: PIL frames piped to ffmpeg, audio mixed with numpy.

    PYTHONPATH=. .venv/bin/python demo/render.py --cut full  --size 1920x1080 --out demo/showcase_full.mp4
    PYTHONPATH=. .venv/bin/python demo/render.py --cut short --size 1080x1080 --out demo/showcase_square.mp4

Inputs (all produced by other demo/ scripts, nothing here is hand-typed):
  demo/narration_<cut>.json + *_units/*.wav   narration units with scene names (demo/tts.py)
  demo/events/<clip>.json                     recorded server events (demo/record_events.py)
  demo/numbers.json                           every on-screen number with its runs/*.json source (demo/extract_numbers.py)
  demo/clips/*.wav                            the audio the server heard

Style: cinematic dark theme, large type, one accent for decisions, live waveform + speaker bands + captions
(see demo/README.md, "Style brief"). The first frame is a full composition (no fade-in); captions are burned in.
"""
from __future__ import annotations

import argparse
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
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))  # narration in, renders out
SCRATCH = OUT_DIR / "scratch"  # ffmpeg temp / mixed audio / video segments (large intermediates, never the internal disk)
FPS = 30
SR = 48000  # output audio rate

# ----------------------------------------------------------------------------- palette / fonts
BG = (11, 13, 18)
PANEL = (20, 23, 31)
INK = (240, 242, 246)
MUTED = (150, 158, 172)
DIM = (70, 76, 90)
ACCENT = (255, 179, 71)     # decisions (turn end)
WAVE = (90, 200, 250)       # audio
GOOD = (94, 214, 154)
BAD = (240, 96, 96)
SPK = [(90, 200, 250), (255, 140, 105), (170, 140, 255), (120, 220, 160)]  # diarizer columns 0-3
FONT_UI = "/System/Library/Fonts/Avenir Next.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"
FONT_IDX = {"heavy": 8, "demi": 2, "medium": 5, "regular": 7}
REPO_URL = "github.com/maxmelichov/audioforge"

_fonts: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _fonts:
        if kind == "mono":
            _fonts[key] = ImageFont.truetype(FONT_MONO, size, index=0)
        else:
            _fonts[key] = ImageFont.truetype(FONT_UI, size, index=FONT_IDX[kind])
    return _fonts[key]


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def lerp(a, b, x):
    return a + (b - a) * x


def mix(c1, c2, x):
    return tuple(int(round(lerp(a, b, x))) for a, b in zip(c1, c2))


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{int(round(ms))} ms"


# ----------------------------------------------------------------------------- recorded events
class Clip:
    """One recorded session: the audio the server heard and every event with its arrival time."""

    def __init__(self, path: Path):
        d = json.loads(path.read_text())
        self.meta = d
        self.name = path.stem
        wav, sr = sf.read(d["audio"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        self.sr = sr
        self.wav = wav
        self.dur = len(wav) / sr
        step = int(sr * 0.002)  # 2 ms envelope
        k = len(wav) // step
        env = np.abs(wav[: k * step]).reshape(k, step).max(1)
        self.env = env / max(1e-6, np.percentile(env, 99.5))
        self.env_step = 0.002
        ev = d["events"]
        self.ready = next((e["msg"] for e in ev if e["msg"]["type"] == "ready"), {})
        self.stats = next((e["msg"] for e in ev if e["msg"]["type"] == "stats"), {})
        fr = [e for e in ev if e["msg"]["type"] == "frame"]
        self.f_t = np.array([e["msg"]["t"] for e in fr])
        self.f_arr = np.array([e["rel_ms"] / 1000 for e in fr])
        self.f_vad = np.array([e["msg"]["vad"] for e in fr])
        self.f_eot = np.array([e["msg"]["eot"] if e["msg"]["eot"] is not None else 0.0 for e in fr])
        self.f_spk = np.array([(e["msg"]["speakers"] + [0, 0, 0, 0])[:4] for e in fr], dtype=np.float32)
        self.f_prim = np.array([-1 if e["msg"]["primary"] is None else e["msg"]["primary"] for e in fr])
        self.partials = [(e["rel_ms"] / 1000, e["msg"]["t"], e["msg"]["text"]) for e in ev if e["msg"]["type"] == "partial"]
        self.turn_ends = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "turn_end"]
        self.finals = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "final"]
        self.enrolled = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "enrolled"]
        self.agent_ends = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "client_agent_end"]
        self.languages = [dict(arr=e["rel_ms"] / 1000, **e["msg"]) for e in ev if e["msg"]["type"] == "language"]
        # dead air of every turn end = decision time - end of the last served-VAD speech frame before it
        for te in self.turn_ends:
            sp = self.f_t[(self.f_vad > 0.5) & (self.f_t <= te["t"])]
            te["last_speech_t"] = float(sp[-1]) if len(sp) else None
            te["dead_air_ms"] = round(1000 * (te["t"] - sp[-1])) if len(sp) else None
            te["compute_ms"] = round(1000 * (te["arr"] - te["t"]))

    def env_at(self, t0: float, t1: float, n: int) -> np.ndarray:
        """n envelope samples for audio time [t0, t1) (0 outside the clip)."""
        idx = ((t0 + (t1 - t0) * (np.arange(n) + 0.5) / n) / self.env_step).astype(int)
        out = np.zeros(n, np.float32)
        ok = (idx >= 0) & (idx < len(self.env))
        out[ok] = self.env[idx[ok]]
        return out

    def partial_at(self, now: float) -> str:
        txt = ""
        for arr, t, text in self.partials:
            if arr <= now:
                txt = text
            else:
                break
        return txt

    def frames_until(self, now: float) -> int:
        return int(np.searchsorted(self.f_arr, now, side="right"))


# ----------------------------------------------------------------------------- timeline
SCENE_SPEC = {
    # clip / clip offset / minimum scene length (s); demo scenes keep playing after the narration ends
    "full": {"hook": {"clip": "oto_f619bed3", "offset": 0.0, "bg": True},
             "demo_oto": {"clip": "oto_f619bed3", "offset": 0.0, "min_len": 23.0},
             "demo_ami": {"clip": "ami_IS1008b_003", "offset": 0.0, "min_len": 18.5},
             "demo_libri": {"clip": "libri_1089-134686-0000", "offset": 0.0, "min_len": 13.0},
             "endcard": {"min_len": 5.0}},
    "short": {"hook": {"clip": "oto_f619bed3", "offset": 0.0, "bg": True},
              "demo_oto": {"clip": "oto_f619bed3", "offset": 9.0, "min_len": 12.5},
              "endcard": {"min_len": 4.5}},
}
LEAD_S = 0.6       # narration starts this long after a scene begins
GAP_S = 0.35
SCENE_GAP_S = 0.5


def build_timeline(narr: dict, cut: str) -> list[dict]:
    spec = SCENE_SPEC[cut]
    scenes: list[dict] = []
    units = narr["units"]
    t = 0.0
    for u in units:
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


def captions_for(units: list[dict], max_chars: int) -> list[tuple[float, float, str]]:
    """Split each unit into caption chunks (at punctuation, <= max_chars) with time proportional to characters."""
    out = []
    for u in units:
        text = u["text"]
        parts = [p.strip() for p in _split_caption(text, max_chars)]
        total = sum(len(p) for p in parts)
        t = u["start"]
        for i, p in enumerate(parts):
            d = (u["end"] - u["start"]) * len(p) / total
            hold = 0.35 if i == len(parts) - 1 else 0.0
            out.append((t, t + d + hold, p))
            t += d
    return out


def _split_caption(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    import re
    sents = re.split(r"(?<=[.?!:;,])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:  # a single long sentence: split on words
        if len(c) <= max_chars:
            out.append(c)
            continue
        words, cur = c.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            out.append(cur)
    return out


# ----------------------------------------------------------------------------- drawing helpers
class Canvas:
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.sq = W == H
        self.s = min(W, H) / 1080  # type scale
        self.m = int(64 * self.s)  # side gutter

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    ARROWS = "→↑↓←"

    def _runs(self, s: str, kind: str, size: int):
        """Split text into (segment, font) runs: Avenir has no arrow glyphs, Menlo does."""
        f, fa = font(kind, self.px(size)), font("mono", self.px(size))
        runs, cur, cur_arrow = [], "", False
        for ch in s:
            is_arrow = ch in self.ARROWS
            if cur and is_arrow != cur_arrow:
                runs.append((cur, fa if cur_arrow else f)); cur = ""
            cur += ch; cur_arrow = is_arrow
        if cur:
            runs.append((cur, fa if cur_arrow else f))
        return runs

    def text(self, d: ImageDraw.ImageDraw, xy, s: str, kind="medium", size=40, fill=INK, anchor="la", alpha=1.0):
        if alpha <= 0:
            return
        col = mix(BG, fill, alpha) if alpha < 1 else fill
        if not any(ch in s for ch in self.ARROWS):
            d.text(xy, s, font=font(kind, self.px(size)), fill=col, anchor=anchor)
            return
        runs = self._runs(s, kind, size)
        total = sum(f.getlength(t) for t, f in runs)
        x, y = xy
        x -= {"l": 0.0, "m": total / 2, "r": total}[anchor[0]]
        for t, f in runs:
            d.text((x, y), t, font=f, fill=col, anchor="l" + anchor[1])
            x += f.getlength(t)

    def tw(self, s: str, kind="medium", size=40) -> float:
        return sum(f.getlength(t) for t, f in self._runs(s, kind, size))

    def wrap(self, s: str, kind: str, size: int, width: float) -> list[str]:
        f = font(kind, self.px(size))
        lines, cur = [], ""
        for w in s.split():
            t = (cur + " " + w).strip()
            if f.getlength(t) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        return lines

    def para(self, d, xy, s, kind, size, width, fill=INK, alpha=1.0, leading=1.25) -> float:
        x, y = xy
        for line in self.wrap(s, kind, size, width):
            self.text(d, (x, y), line, kind, size, fill, alpha=alpha)
            y += self.px(size) * leading
        return y

    def pill(self, d, box, fill, radius=None):
        radius = radius or self.px(14)
        d.rounded_rectangle(box, radius=radius, fill=fill)

    def source(self, d, srcs):
        """Source file(s) in small mono type in the bottom-right corner of a numbers scene."""
        s = "source: " + ", ".join(sorted(set(srcs)))
        self.text(d, (self.W - self.m, self.H - self.px(26)), s, "mono", 22, MUTED, anchor="rd")


def new_frame(cv: Canvas) -> Image.Image:
    img = Image.new("RGB", (cv.W, cv.H), BG)
    return img


# ----------------------------------------------------------------------------- live demo component
def draw_live(cv: Canvas, img: Image.Image, clip: Clip, now: float, box, *, background=False, title=None):
    """The live view at clip time ``now``: scrolling waveform + VAD + speaker bands, events, partial text, counters."""
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    W = x1 - x0
    span_past, span_fut = (5.5, 2.5) if not cv.sq else (4.5, 1.5)
    head = x0 + W * span_past / (span_past + span_fut)
    px_per_s = W / (span_past + span_fut)
    t_left = now - span_past
    dimf = 0.35 if background else 1.0

    # --- waveform strip
    wh = int((y1 - y0) * (0.34 if not background else 0.6))
    wy0 = y0 + cv.px(46 if title else 8)
    if title and not background:
        cv.text(d, (x0, y0), title, "demi", 26, MUTED)
    env = clip.env_at(t_left, now + span_fut, W)
    mid = wy0 + wh // 2
    for i in range(W):
        t = t_left + (i + 0.5) / px_per_s
        if t < 0 or t > clip.dur:
            continue
        a = env[i]
        h = max(1, int(a * wh * 0.48))
        if t <= now:
            col = mix(BG, WAVE, dimf * (0.55 + 0.45 * min(1.0, a * 1.5)))
        else:
            col = mix(BG, WAVE, 0.18 * dimf)
        d.line([(x0 + i, mid - h), (x0 + i, mid + h)], fill=col)
    # --- VAD + speaker bands (only frames that have ARRIVED by now)
    n = clip.frames_until(now)
    by0 = wy0 + wh + cv.px(10)
    row = cv.px(16 if not cv.sq else 14)
    if not background:
        cv.text(d, (x0, by0), "VAD", "mono", 18, MUTED, anchor="ls")
        for k in range(4):
            cv.text(d, (x0, by0 + row * (k + 1) + row), f"spk {k}", "mono", 18, mix(BG, SPK[k], 0.9), anchor="ls")
    lx = x0 + cv.px(64) if not background else x0
    if n:
        ft, fv, fs, fp = clip.f_t[:n], clip.f_vad[:n], clip.f_spk[:n], clip.f_prim[:n]
        vis = (ft >= t_left) & (ft <= now + 0.1)
        for t, v, s, p in zip(ft[vis], fv[vis], fs[vis], fp[vis]):
            xa = int(head + (t - 0.08 - now) * px_per_s)
            xb = int(head + (t - now) * px_per_s)
            if xb <= lx:
                continue
            xa = max(xa, lx)
            if v > 0.5:
                d.rectangle([xa, by0 + 2, xb, by0 + row - 3], fill=mix(BG, INK, dimf * (0.35 + 0.65 * v)))
            for k in range(4):
                if s[k] > 0.5:
                    col = mix(BG, SPK[k], dimf * (0.4 + 0.6 * float(s[k])))
                    yy = by0 + row * (k + 1) + 2
                    d.rectangle([xa, yy, xb, yy + row - 4], fill=col)
            if p >= 0 and not background:
                yy = by0 + row * (p + 1) + 2
                d.rectangle([xa, yy + row - 6, xb, yy + row - 4], fill=INK)
    bands_bottom = by0 + row * 5 + cv.px(6)
    # --- markers: agent_end, enrolled, turn_end (+ flash)
    def marker(t, label, col, yy, dash=False):
        x = int(head + (t - now) * px_per_s)
        if x < lx or x > x1:
            return
        if dash:
            for yy2 in range(wy0, bands_bottom, 8):
                d.line([(x, yy2), (x, yy2 + 4)], fill=col)
        else:
            d.line([(x, wy0), (x, bands_bottom)], fill=col, width=2)
        if not background:
            cv.text(d, (x + cv.px(6), yy), label, "mono", 18, col, anchor="ls")
    for ae in clip.agent_ends:
        if ae["arr"] <= now:
            marker(ae["t"], "agent TTS end", MUTED, wy0 + cv.px(20), dash=True)
    for en in clip.enrolled:
        if en["arr"] <= now:
            marker(en["t"], f"user enrolled: spk {en['column']}", GOOD, wy0 + cv.px(42))
    for lg in clip.languages:
        if lg["arr"] <= now:
            marker(lg["t"], f"language: {lg['language']} ({lg['confidence']:.2f})", SPK[3], wy0 + cv.px(86), dash=True)
    for te in clip.turn_ends:
        if te["arr"] <= now:
            marker(te["t"], "turn end", ACCENT, wy0 + cv.px(64))
            if te.get("last_speech_t") is not None:
                xa = int(head + (te["last_speech_t"] - now) * px_per_s)
                xb = int(head + (te["t"] - now) * px_per_s)
                if xb > lx:
                    d.line([(max(xa, lx), bands_bottom + 3), (xb, bands_bottom + 3)], fill=ACCENT, width=3)
    # playhead
    d.line([(head, wy0 - cv.px(4)), (head, bands_bottom)], fill=mix(BG, INK, 0.8 * dimf), width=2)
    # flash on a turn-end arrival (700 ms), dead-air label persists 3.5 s
    for te in clip.turn_ends:
        age = now - te["arr"]
        if 0 <= age < 0.7:
            r = int(cv.px(30) + ease_out(age / 0.7) * cv.px(140))
            a = 1 - age / 0.7
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(ov)
            od.ellipse([head - r, mid - r, head + r, mid + r], outline=ACCENT + (int(220 * a),), width=max(2, cv.px(6)))
            od.ellipse([head - r // 3, mid - r // 3, head + r // 3, mid + r // 3], fill=ACCENT + (int(90 * a),))
            img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))
            d = ImageDraw.Draw(img)
        if 0 <= age < 3.5 and not background:
            a = 1.0 if age < 2.8 else 1 - (age - 2.8) / 0.7
            bx = x1 - cv.px(20)
            top = wy0 + cv.px(4)
            da = te["dead_air_ms"]
            big = fmt_ms(da) if da is not None else "n/a"
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).rounded_rectangle([bx - cv.px(560 if not cv.sq else 470), top - cv.px(8), bx + cv.px(12), top + cv.px(152)],
                                                 radius=cv.px(14), fill=(0, 0, 0, int(170 * a)))
            img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))
            d = ImageDraw.Draw(img)
            cv.text(d, (bx, top), "TURN END", "heavy", 26, ACCENT, anchor="ra", alpha=a)
            cv.text(d, (bx, top + cv.px(30)), big, "heavy", 64, ACCENT, anchor="ra", alpha=a)
            cv.text(d, (bx, top + cv.px(102)), "dead air: last speech frame to decision", "medium", 20, MUTED, anchor="ra", alpha=a)
            cv.text(d, (bx, top + cv.px(126)), f"+{te['compute_ms']} ms until the event arrived  ·  policy {te['policy']}",
                    "medium", 20, MUTED, anchor="ra", alpha=a)
    if background:
        return bands_bottom
    # --- text panel: live partial + finals
    ty = bands_bottom + cv.px(26)
    tw = W
    cv.text(d, (x0, ty), "LIVE PARTIAL", "demi", 20, MUTED)
    ty += cv.px(30)
    partial = clip.partial_at(now)
    if partial.strip():
        ty = cv.para(d, (x0, ty), partial.strip() + " |", "medium", 44 if not cv.sq else 40, tw, INK)
    else:
        cv.text(d, (x0, ty), "…", "medium", 44, DIM)
        ty += cv.px(56)
    ty += cv.px(14)
    fins = [f for f in clip.finals if f["arr"] <= now]
    shown = {}
    for f in fins:  # a tdt_v3 final replaces the stream final with the same t
        key = round(f["t"], 3)
        if f.get("source") == "tdt_v3" or key not in shown:
            shown[key] = f
    fin_list = list(shown.values())[-2:]
    if fin_list:
        cv.text(d, (x0, ty), "FINALS", "demi", 20, MUTED)
        ty += cv.px(30)
    for f in reversed(fin_list):
        spk = f.get("speaker")
        col = SPK[spk] if spk is not None and 0 <= spk < 4 else MUTED
        d.ellipse([x0, ty + cv.px(10), x0 + cv.px(16), ty + cv.px(26)], fill=col)
        src = f.get("source", "stream")
        tag = "rewritten by Parakeet-TDT 0.6B v3" + (f"  ·  {int(f['latency_ms'])} ms" if f.get("latency_ms") is not None else "") \
            if src == "tdt_v3" else "streaming final"
        cv.text(d, (x0 + cv.px(28), ty), f"t = {f['t']:.2f} s  ·  {tag}", "mono", 18, GOOD if src == "tdt_v3" else MUTED)
        ty += cv.px(26)
        txt = (f["text"] or "(no speech)").strip()
        ty = cv.para(d, (x0 + cv.px(28), ty), txt, "regular", 30 if not cv.sq else 28, tw - cv.px(28), mix(BG, INK, 0.85))
        ty += cv.px(12)
        if ty > y1 - cv.px(60):
            break
    # --- counters
    lag_ms = None
    if n:
        lag_ms = int(round((clip.f_arr[n - 1] - clip.f_t[n - 1]) * 1000))
    rtf = clip.stats.get("rtf")
    ready = clip.ready
    items = [f"audio {now:5.1f} s", f"frame lag {lag_ms} ms" if lag_ms is not None else "frame lag –",
             f"session RTF {rtf:.2f}" if rtf is not None else "RTF –", f"chunk {ready.get('chunk_ms', 160)} ms",
             f"policy {clip.meta.get('policy')}", "2 CPU threads"]
    cy = y1 - cv.px(8)
    cv.text(d, (x0, cy), "   ·   ".join(items), "mono", 20, MUTED, anchor="ls")


# ----------------------------------------------------------------------------- scenes
class Renderer:
    def __init__(self, cut: str, W: int, H: int, narr: dict, numbers: dict, clips: dict[str, Clip]):
        self.cut, self.cv, self.narr, self.N, self.clips = cut, Canvas(W, H), narr, numbers, clips
        self.scenes = build_timeline(narr, cut)
        self.duration = self.scenes[-1]["end"]
        units = [u for s in self.scenes for u in s["units"]]
        self.captions = captions_for(units, 54 if not self.cv.sq else 40)

    def num(self, key):
        return self.N[key]["value"]

    def src(self, *keys):
        return [self.N[k]["source"] for k in keys]

    # -- scene dispatch
    def frame(self, T: float) -> Image.Image:
        cv = self.cv
        img = new_frame(cv)
        scene = next((s for s in self.scenes if s["start"] <= T < s["end"]), self.scenes[-1])
        st = T - scene["start"]
        # which unit of the scene is speaking (or last spoken)
        ui, prog = -1, 0.0
        for i, u in enumerate(scene["units"]):
            if T >= u["start"]:
                ui, prog = i, min(1.0, (T - u["start"]) / max(0.01, u["end"] - u["start"]))
        getattr(self, "scene_" + scene["name"])(img, scene, st, ui, prog, T)
        self.draw_caption(img, T)
        return img

    def draw_caption(self, img, T):
        cv = self.cv
        cap = next((c for c in self.captions if c[0] <= T < c[1]), None)
        if not cap:
            return
        d = ImageDraw.Draw(img)
        size = 46 if not cv.sq else 44
        width = cv.W - 2 * cv.m - cv.px(40)
        lines = cv.wrap(cap[2], "demi", size, width)
        lh = cv.px(size) * 1.22
        bh = int(lh * len(lines) + cv.px(28))
        y1 = cv.H - cv.px(56 if not cv.sq else 120)
        y0 = y1 - bh
        maxw = max(cv.tw(l, "demi", size) for l in lines) + cv.px(44)
        x0 = (cv.W - maxw) / 2
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle([x0, y0, x0 + maxw, y1], radius=cv.px(16), fill=(0, 0, 0, 175))
        img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))
        d = ImageDraw.Draw(img)
        y = y0 + cv.px(14)
        for l in lines:
            cv.text(d, (cv.W / 2, y), l, "demi", size, INK, anchor="ma")
            y += lh

    def header(self, d, title, sub=None):
        cv = self.cv
        cv.text(d, (cv.m, cv.px(44)), title, "heavy", 54 if not cv.sq else 44, INK)
        y = cv.px(44) + cv.px(66 if not cv.sq else 60)
        if sub:
            y = cv.para(d, (cv.m, y), sub, "medium", 26 if not cv.sq else 22, cv.W - 2 * cv.m, MUTED) + cv.px(14)
        else:
            y += cv.px(10)
        return y

    # -- hook: live waveform background + headline + key number in the first frame
    def scene_hook(self, img, scene, st, ui, prog, T):
        cv = self.cv
        clip = self.clips[scene["clip"]]
        now = scene["offset"] + st
        if cv.sq:
            draw_live(cv, img, clip, now, (cv.m, cv.px(560), cv.W - cv.m, cv.H - cv.px(220)), background=True)
        else:
            draw_live(cv, img, clip, now, (cv.m, cv.px(600), cv.W - cv.m, cv.H - cv.px(150)), background=True)
        d = ImageDraw.Draw(img)
        # headline (no fade: fully visible on frame 0; the second line slides in)
        y = cv.px(90 if not cv.sq else 70)
        h1 = ["Voice agents interrupt you.", "Or they leave you hanging."] if self.cut == "short" else \
             ["Voice agents interrupt.", "Or they wait too long."]
        size = 96 if not cv.sq else 62
        cv.text(d, (cv.m, y), h1[0], "heavy", size, INK)
        a2 = ease_out((st - 0.9) / 0.5) if st > 0.9 else 0.0
        if self.cut == "short":
            a2 = 1.0  # sound-off: everything readable at once
        cv.text(d, (cv.m + cv.px(24) * (1 - a2), y + cv.px(size) * 1.15), h1[1], "heavy", size, ACCENT, alpha=a2)
        # key number tile, visible from frame 0
        ours, best = self.num("turn_miss_all/ours_hybrid"), self.num("turn_miss_all/parakeet_eou")
        tx = cv.m
        ty = y + cv.px(size) * 2.45
        cv.text(d, (tx, ty), f"{ours:.1f}%", "heavy", 88 if not cv.sq else 76, WAVE)
        cv.text(d, (tx + cv.tw(f"{ours:.1f}%", "heavy", 88 if not cv.sq else 76) + cv.px(24), ty + cv.px(30)),
                f"vs ≥ {best:.0f}%", "demi", 44 if not cv.sq else 38, MUTED)
        cv.text(d, (tx, ty + cv.px(100 if not cv.sq else 90)),
                "turn ends missed: one speaker-aware model vs smart-turn, LiveKit, NVIDIA EOU", "medium", 28 if not cv.sq else 25, INK)
        cv.text(d, (tx, ty + cv.px(136 if not cv.sq else 124)),
                f"AMI meetings, n = {self.num('turn_n')}, all held to ≤ 5% false cutoffs, 6 s horizon · lower is better",
                "medium", 22, MUTED)
        # tagline
        cv.text(d, (tx, ty + cv.px(176 if not cv.sq else 160)),
                "one streaming model · ASR + VAD + speaker + turn-taking · laptop CPU", "demi", 28 if not cv.sq else 24, MUTED)
        cv.source(d, self.src("turn_miss_all/ours_hybrid", "turn_miss_all/parakeet_eou"))

    # -- architecture diagram: one encoder, many heads
    def scene_architecture(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = self.header(d, "One encoder, many heads", "the NVIDIA recipe, rebuilt in plain PyTorch (no NeMo); encoder frozen, heads trained" if not cv.sq else
                          "NVIDIA recipe in plain PyTorch · encoder frozen, heads trained")
        reveal = ease_out(st / 0.6)
        if cv.sq:
            self._arch_square(d, top, st, reveal)
        else:
            self._arch_wide(d, top, st, reveal)
        cv.text(d, (cv.W - cv.m, cv.H - cv.px(26)), "runs/stage1_served.afm · runs/nemo_nemotron3_diar.afm · audioforge/serve.py",
                "mono", 22, MUTED, anchor="rd")

    def _box(self, d, box, title, sub, col, alpha=1.0, fill=PANEL):
        cv = self.cv
        if alpha <= 0:
            return
        x0, y0, x1, y1 = box
        d.rounded_rectangle(box, radius=cv.px(18), fill=mix(BG, fill, alpha), outline=mix(BG, col, alpha), width=max(2, cv.px(3)))
        compact = (y1 - y0) < cv.px(100)
        cv.text(d, ((x0 + x1) / 2, y0 + cv.px(12 if compact else 22)), title, "demi", (26 if compact else 30) if not cv.sq else 26, col, anchor="ma", alpha=alpha)
        y = y0 + cv.px(48 if compact else 64)
        for line in sub:
            cv.text(d, ((x0 + x1) / 2, y), line, "medium", (20 if compact else 22) if not cv.sq else 20, INK, anchor="ma", alpha=alpha)
            y += cv.px(28)

    def _arrow(self, d, p0, p1, col, alpha=1.0):
        cv = self.cv
        if alpha <= 0:
            return
        c = mix(BG, col, alpha)
        d.line([p0, p1], fill=c, width=max(2, cv.px(4)))
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        L = cv.px(16)
        d.polygon([p1, (p1[0] - L * math.cos(ang - 0.5), p1[1] - L * math.sin(ang - 0.5)),
                   (p1[0] - L * math.cos(ang + 0.5), p1[1] - L * math.sin(ang + 0.5))], fill=c)

    def _arch_wide(self, d, top, st, reveal):
        cv = self.cv
        y0 = top + cv.px(40)
        bh = cv.px(150)
        col_w = cv.px(300)
        xa = cv.m
        # stage 1: audio -> mel
        self._box(d, (xa, y0 + cv.px(120), xa + col_w, y0 + cv.px(120) + bh), "16 kHz audio", ["int16 PCM in", "160 ms chunks"], MUTED, reveal)
        xb = xa + col_w + cv.px(70)
        a2 = ease_out((st - 0.4) / 0.6)
        self._arrow(d, (xa + col_w, y0 + cv.px(195)), (xb, y0 + cv.px(195)), MUTED, a2)
        self._box(d, (xb, y0 + cv.px(120), xb + col_w, y0 + cv.px(120) + bh), "log-mel 80", ["10 ms hop", "frame-local norm"], MUTED, a2)
        xc = xb + col_w + cv.px(70)
        a3 = ease_out((st - 0.8) / 0.6)
        self._arrow(d, (xb + col_w, y0 + cv.px(195)), (xc, y0 + cv.px(195)), MUTED, a3)
        enc_w = cv.px(380)
        self._box(d, (xc, y0 + cv.px(40), xc + enc_w, y0 + cv.px(40) + cv.px(310)), "FastConformer encoder",
                  ["NVIDIA streaming weights, frozen", "115 M params, 8× subsampling", "cache-aware, [70, 1] context", "one pass per 160 ms chunk", "80 ms output frames"],
                  WAVE, a3, fill=(24, 34, 44))
        # heads
        xd = xc + enc_w + cv.px(90)
        heads = [("RNNT transcript", "partial text every chunk", WAVE), ("VAD head", "speech / silence", INK),
                 ("speaker head", "block-4 tap, TitaNet-distilled", SPK[2]), ("turn head", "GRU, end-of-turn p", ACCENT)]
        hh = cv.px(84)
        for i, (t, s, c) in enumerate(heads):
            a = ease_out((st - 1.4 - 0.25 * i) / 0.5)
            yy = y0 + cv.px(4) + i * (hh + cv.px(14))
            self._arrow(d, (xc + enc_w, y0 + cv.px(195)), (xd, yy + hh / 2), c, a)
            self._box(d, (xd, yy, xd + cv.px(330), yy + hh), t, [s], c, a)
        # diarizer beside
        a5 = ease_out((st - 2.8) / 0.6)
        yy = y0 + cv.px(400)
        self._box(d, (xa, yy, xd - cv.px(30), yy + cv.px(120)), "NVIDIA streaming diarizer · Nemotron-3, 100 M · same 80 ms frame clock",
                  ["4 speaker columns → primary speaker (armed at the agent's TTS end)", "→ silence timeout OR turn head → turn_end"], SPK[1], a5)
        a6 = ease_out((st - 3.4) / 0.6)
        self._box(d, (xd, yy, xd + cv.px(330), yy + cv.px(120)), "per-turn final ASR", ["Parakeet-TDT 0.6B v3, offline,", "own process, rewrites the final"], GOOD, a6)
        cv.text(d, (cv.m, yy + cv.px(180)), "events out: ready · frame (80 ms) · partial · turn_end · final · enrolled · language · stats   —   WebSocket, one session per connection",
                "mono", 22, MUTED, anchor="ls", alpha=a6)

    def _arch_square(self, d, top, st, reveal):
        cv = self.cv
        y = top + cv.px(20)
        w = cv.W - 2 * cv.m
        self._box(d, (cv.m, y, cv.m + w, y + cv.px(90)), "16 kHz audio → log-mel 80", ["160 ms chunks in"], MUTED, reveal)
        y2 = y + cv.px(130)
        a3 = ease_out((st - 0.6) / 0.6)
        self._arrow(d, (cv.W / 2, y + cv.px(90)), (cv.W / 2, y2), MUTED, a3)
        self._box(d, (cv.m, y2, cv.m + w, y2 + cv.px(150)), "FastConformer encoder (NVIDIA, frozen, 115 M)",
                  ["cache-aware streaming, one pass per 160 ms chunk", "80 ms output frames"], WAVE, a3, fill=(24, 34, 44))
        y3 = y2 + cv.px(200)
        heads = [("transcript", WAVE), ("VAD", INK), ("speaker", SPK[2]), ("turn end", ACCENT)]
        bw = (w - 3 * cv.px(16)) / 4
        for i, (t, c) in enumerate(heads):
            a = ease_out((st - 1.2 - 0.2 * i) / 0.5)
            x = cv.m + i * (bw + cv.px(16))
            self._arrow(d, (x + bw / 2, y2 + cv.px(150)), (x + bw / 2, y3), c, a)
            self._box(d, (x, y3, x + bw, y3 + cv.px(90)), t, ["small head"], c, a)
        y4 = y3 + cv.px(130)
        a5 = ease_out((st - 2.4) / 0.6)
        self._box(d, (cv.m, y4, cv.m + w, y4 + cv.px(96)), "+ NVIDIA streaming diarizer (Nemotron-3), same frame clock",
                  ["primary speaker → timeout OR turn head → turn_end"], SPK[1], a5)

    # -- live demos
    def _demo(self, img, scene, st, title, short=None):
        cv = self.cv
        if cv.sq and short:
            title = short
        clip = self.clips[scene["clip"]]
        now = scene["offset"] + st
        if cv.sq:
            box = (cv.m, cv.px(40), cv.W - cv.m, cv.H - cv.px(290))
        else:
            box = (cv.m, cv.px(40), cv.W - cv.m, cv.H - cv.px(200))
        draw_live(cv, img, clip, now, box, title=title)

    def scene_demo_oto(self, img, scene, st, ui, prog, T):
        self._demo(img, scene, st, "LIVE · two-party call (otoSpeech, CC-BY-4.0) · stage1_served + Nemotron-3 diarizer · hybrid_dyn · enroll after_agent_arm",
                   "LIVE · two-party call (otoSpeech) · served model + Nemotron-3")

    def scene_demo_ami(self, img, scene, st, ui, prog, T):
        self._demo(img, scene, st, "LIVE · AMI meeting IS1008b, two speakers overlapping · same server · final ASR: Parakeet-TDT v3",
                   "LIVE · AMI meeting IS1008b · same server")

    def scene_demo_libri(self, img, scene, st, ui, prog, T):
        self._demo(img, scene, st, "LIVE · LibriSpeech test-clean 1089-134686-0000 · read speech")

    # -- scorecard: one panel per narration unit
    def scene_scorecard(self, img, scene, st, ui, prog, T):
        d = ImageDraw.Draw(img)
        k = max(ui, 0)
        if self.cut == "short":
            panels = [self._panel_turn, self._panel_tsvad, self._panel_rtf]
        else:
            panels = [self._panel_turn, self._panel_turnbench, self._panel_wer, self._panel_rtf]
        k = min(k, len(panels) - 1)
        # time since this panel started
        t_panel = T - (scene["units"][k]["start"] if k < len(scene["units"]) else scene["start"])
        if ui < 0:
            t_panel = st
        panels[k](d, t_panel)

    def _bars(self, d, top, rows, unit, higher_better, t, vmax=None, note=None, label_w=None):
        """Horizontal bars: rows = (label, value, is_ours, extra)."""
        cv = self.cv
        vmax = vmax or max(r[1] for r in rows) * 1.12
        label_w = label_w or cv.px(520 if not cv.sq else 380)
        x0 = cv.m + label_w
        x1 = cv.W - cv.m - cv.px(200 if not cv.sq else 150)
        bh = cv.px(46 if not cv.sq else 42)
        gap = cv.px(22 if not cv.sq else 18)
        y = top
        for i, (label, val, ours, extra) in enumerate(rows):
            a = ease_out((t - 0.15 * i) / 0.6)
            col = WAVE if ours else DIM
            cv.text(d, (cv.m, y + bh / 2), label, "demi" if ours else "medium", 30 if not cv.sq else 22, INK if ours else MUTED, anchor="lm")
            w = (x1 - x0) * (val / vmax) * a
            d.rounded_rectangle([x0, y, x0 + max(w, 2), y + bh], radius=cv.px(8), fill=col)
            vtxt = f"{val:.1f}{unit}" if isinstance(val, float) and unit == "%" else (f"{val:.3f}" if unit == "" else f"{val}{unit}")
            cv.text(d, (x0 + w + cv.px(16), y + bh / 2), vtxt, "demi", 30 if not cv.sq else 24, INK if ours else MUTED, anchor="lm", alpha=a)
            if extra:
                cv.text(d, (cv.W - cv.m, y + bh / 2), extra, "mono", 20, MUTED, anchor="rm", alpha=a)
            y += bh + gap
        if note:
            y = cv.para(d, (cv.m, y + cv.px(10)), note, "medium", 24 if not cv.sq else 20, cv.W - 2 * cv.m, MUTED)
        return y

    def _panel_intro(self, d, t):
        cv = self.cv
        self.header(d, "The numbers", "measured on real audio · same metric code for every row · every figure traceable to runs/*.json")
        items = [("end-of-turn", "eot-bench v2, AMI dev, n = 974", ACCENT), ("TurnBench", "Sesame's two-party benchmark", WAVE),
                 ("speaker head", "within-meeting EER", SPK[2]), ("final transcript", "WER on meeting segments", GOOD), ("cost", "RTF on 2 CPU threads", INK)]
        y = cv.px(260)
        for i, (a, b, c) in enumerate(items):
            al = ease_out((t - 0.2 * i) / 0.5)
            cv.text(d, (cv.m, y), a, "heavy", 48 if not cv.sq else 40, c, alpha=al)
            cv.text(d, (cv.m + cv.px(460 if not cv.sq else 330), y + cv.px(12)), b, "medium", 30 if not cv.sq else 24, MUTED, alpha=al)
            y += cv.px(100 if not cv.sq else 92)
        cv.source(d, ["runs/*.json"])

    def _panel_turn(self, d, t):
        cv = self.cv
        n = self.num("turn_n")
        top = self.header(d, "End of turn on real meetings", f"AMI dev, n = {n} turn ends · every system held to ≤ 5% false cutoffs · 6 s horizon · lower is better")
        sq = cv.sq
        rows = [("ours: speaker-aware hybrid", self.num("turn_miss_all/ours_hybrid"), True, ""),
                ("NVIDIA Parakeet-Realtime-EOU" if not sq else "NVIDIA Parakeet-EOU", self.num("turn_miss_all/parakeet_eou"), False, ""),
                ("smart-turn v3.2 + Silero (Pipecat)" if not sq else "smart-turn v3.2 + Silero", self.num("turn_miss_all/smartturn_silero"), False, ""),
                ("Silero VAD timeout", self.num("turn_miss_all/silero_timeout"), False, ""),
                ("LiveKit turn detector + timeout" if not sq else "LiveKit detector + timeout", self.num("turn_miss_all/livekit_text"), False, "")]
        self._bars(d, top + cv.px(20), rows, "%", False, t, vmax=100,
                       note="% of turn ends missed within 6 s. The margin comes from ends where the next speaker takes the floor within ~1 s.")
        cv.source(d, self.src("turn_miss_all/ours_hybrid", "turn_n"))

    def _panel_turnbench(self, d, t):
        cv = self.cv
        top = self.header(d, "TurnBench (two-party), Sesame's official scorer", "recall at ≤ 10% false positives · thresholds frozen from AMI, no retraining · higher is better")
        rows = [("ours: predictive trigger OR Silero", self.num("turnbench_recall/ours_predictive"), True, f"median {self.num('turnbench_p50/ours_predictive')} ms"),
                ("VAP (published reference)", self.num("turnbench_recall/vap"), False, f"median {self.num('turnbench_p50/vap')} ms"),
                ("ours: head OR Silero", self.num("turnbench_recall/ours_hybrid"), False, f"median {self.num('turnbench_p50/ours_hybrid')} ms"),
                ("Kyutai STT semantic VAD", self.num("turnbench_recall/kyutai_semantic_vad"), False, f"median {self.num('turnbench_p50/kyutai_semantic_vad')} ms"),
                ("smart-turn v3", self.num("turnbench_recall/smart_turn_v3"), False, f"median {self.num('turnbench_p50/smart_turn_v3')} ms")]
        self._bars(d, top + cv.px(20), rows, "", True, t, vmax=1.0,
                   note="the gap to VAP is latency, not recall: VAP fires before the silence, we fire after it starts")
        cv.source(d, self.src("turnbench_recall/ours_predictive", "turnbench_recall/vap"))

    def _panel_spk(self, d, t):
        cv = self.cv
        top = self.header(d, "Speaker head: 32 → 14 % EER", "within-meeting equal-error rate, AMI dev (n = 64 segments) · 0.5 M parameters on the shared encoder pass · lower is better")
        a, b = self.num("spk_eer/original"), self.num("spk_eer/shipped")
        y = top + cv.px(70)
        al = ease_out(t / 0.6)
        cv.text(d, (cv.m, y), f"{a:.1f}%", "heavy", 150 if not cv.sq else 120, DIM, alpha=al)
        x2 = cv.m + cv.tw(f"{a:.1f}%", "heavy", 150 if not cv.sq else 120) + cv.px(60)
        a2 = ease_out((t - 0.5) / 0.6)
        cv.text(d, (x2, y + cv.px(40)), "→", "heavy", 90 if not cv.sq else 70, MUTED, alpha=a2)
        cv.text(d, (x2 + cv.px(120), y), f"{b:.1f}%", "heavy", 150 if not cv.sq else 120, SPK[2], alpha=a2)
        y += cv.px(220 if not cv.sq else 180)
        for i, line in enumerate(["original head: a near-uniform mix of all 17 encoder layers",
                                  "shipped head: block 4 alone + relational distillation from TitaNet-L",
                                  "TitaNet-L itself: 8.2 % (research/BASELINES.md) — still better, at its own encoder's cost"]):
            cv.text(d, (cv.m, y), line, "medium", 30 if not cv.sq else 24, INK if i < 2 else MUTED, alpha=ease_out((t - 1.0 - 0.3 * i) / 0.5))
            y += cv.px(48 if not cv.sq else 40)
        cv.source(d, self.src("spk_eer/original", "spk_eer/shipped"))

    def _panel_wer(self, d, t):
        cv = self.cv
        top = self.header(d, "Transcript quality", "word error rate, AMI dev single-speaker segments (n = 200), Whisper-normalized · lower is better")
        rows = [("streaming pass, 160 ms chunks (live partials)", self.num("wer/stream_ami"), False, ""),
                ("per-turn final: Parakeet-TDT 0.6B v3", self.num("wer/tdt_v3_ami"), True, ""),
                ("Whisper large-v3-turbo, offline (reference)", self.num("wer/whisper_turbo_ami"), False, "")]
        self._bars(d, top + cv.px(30), rows, "%", False, t, vmax=30, label_w=cv.px(720 if not cv.sq else 400),
                       note=f"read speech (LibriSpeech test-clean): streaming {self.num('wer/stream_libri'):.2f} % WER")
        cv.source(d, self.src("wer/stream_ami", "wer/tdt_v3_ami", "wer/whisper_turbo_ami", "wer/stream_libri"))

    def _panel_rtf(self, d, t):
        cv = self.cv
        lo, hi = self.num("rtf/server_nemotron_range")
        top = self.header(d, "Cost: real-time factor on 2 CPU threads", "whole server (ASR + heads + Nemotron-3 diarizer + turn rule), median per run, live in Pipecat/LiveKit · Apple M5")
        y = top + cv.px(70)
        al = ease_out(t / 0.6)
        cv.text(d, (cv.m, y), f"RTF {lo:.2f} – {hi:.2f}", "heavy", 120 if not cv.sq else 84, WAVE, alpha=al)
        y += cv.px(170 if not cv.sq else 120)
        lines = ["< 1.0 means it keeps up with the audio; measured live across AMI, TurnBench and otoSpeech clips (0.63–0.65 on the two-party sets)",
                 f"streaming ASR alone: RTF {self.num('rtf/stream_asr_cpu2'):.3f}  ·  with the Sortformer v2 diarizer instead: {self.num('rtf/server_sortformer_range')[0]:.2f} – {self.num('rtf/server_sortformer_range')[1]:.2f}",
                 "1.7–3.6x the CPU of the Pipecat / LiveKit default stacks (0.43 / 0.28 CPU s per audio second); 1.5 GB RSS with Nemotron-3"]
        for i, line in enumerate(lines):
            cv.text(d, (cv.m, y), line, "medium", 30 if not cv.sq else 24, INK if i == 0 else MUTED, alpha=ease_out((t - 0.6 - 0.3 * i) / 0.5))
            y += cv.px(50 if not cv.sq else 40)
        cv.source(d, self.src("rtf/server_nemotron_range", "rtf/stream_asr_cpu2"))

    # -- TS-VAD: the target-speaker track feeding the turn head (research/IMPROVE_115M.md A.2, offline benchmark)
    def _panel_tsvad(self, d, t):
        cv = self.cv
        if cv.sq:
            top = self.header(d, "Next step: a target-speaker track", "missed turn ends at 6 s · ≤ 5% false cutoffs · 5 s voice print · offline, not served yet")
        else:
            top = self.header(d, "Next step, measured: a target-speaker track for the turn head",
                              "missed turn ends at 6 s, ≤ 5% false cutoffs · 5 s voice print · offline eot-bench, not served yet · lower is better")
        N = self.num
        y = top + cv.px(20)
        colw = (cv.W - 2 * cv.m) / 2
        cols = [("AMI dev, n = 974", "ami"), ("ICSI held-out, n = 1312", "icsi")]
        big = 96 if not cv.sq else 72
        rowh = cv.px(big * 1.2) + cv.px(44) + cv.px(90)
        for ci, (title, ds) in enumerate(cols):
            x = cv.m if cv.sq else cv.m + ci * colw
            yb = y + ci * rowh if cv.sq else y
            al = ease_out((t - 0.3 * ci) / 0.6)
            cv.text(d, (x, yb), title, "demi", 30 if not cv.sq else 24, MUTED, alpha=al)
            a, b = N(f"tsvad_turn/{ds}/hybrid_sortformer"), N(f"tsvad_turn/{ds}/hybrid_tsvad")
            yy = yb + cv.px(44)
            cv.text(d, (x, yy), f"{a:.1f}%", "heavy", big, DIM, alpha=al)
            x2 = x + cv.tw(f"{a:.1f}%", "heavy", big) + cv.px(24)
            a2 = ease_out((t - 0.5 - 0.3 * ci) / 0.6)
            cv.text(d, (x2, yy + cv.px(big * 0.3)), "→", "heavy", int(big * 0.55), MUTED, alpha=a2)
            cv.text(d, (x2 + cv.px(big * 0.7), yy), f"{b:.1f}%", "heavy", big, GOOD, alpha=a2)
            lo, hi = N(f"tsvad_turn/{ds}/hybrid_tsvad/ci")
            cv.text(d, (x, yy + cv.px(big * 1.2)), f"diarizer column → turn head   vs   TS-VAD track → turn head  [{lo:.1f}, {hi:.1f}]",
                    "medium", 22 if not cv.sq else 19, MUTED, alpha=a2)
            dyn = N(f"tsvad_turn/{ds}/hybrid_dyn_tsvad")
            dopen = N(f"tsvad_turn/{ds}/hybrid_dyn_tsvad/open")
            cv.text(d, (x, yy + cv.px(big * 1.2) + cv.px(32)), f"with the Silero fusion (hybrid_dyn): {dyn:.1f}% all · {dopen:.1f}% floor-open",
                    "medium", 22 if not cv.sq else 19, MUTED, alpha=a2)
        y2 = y + (cv.px(300) if not cv.sq else 2 * rowh + cv.px(10))
        lines = [f"the head: {N('tsvad_params') / 1e6:.2f} M parameters on block 4 of the encoder pass that already runs; no diarizer call on the turn path",
                 f"oracle speaker on AMI: {N('tsvad_turn/ami/hybrid_oracle'):.1f}% · Silero timeout: {N('tsvad_turn/ami/silero_timeout'):.1f}% (AMI) / {N('tsvad_turn/icsi/silero_timeout'):.1f}% (ICSI)",
                 "caveats: the voice print comes from elsewhere in the meeting; the head is untrained on this input; no paired CI yet; no served path"]
        for i, line in enumerate(lines):
            y2 = cv.para(d, (cv.m, y2), line, "medium", 26 if not cv.sq else 20, cv.W - 2 * cv.m, INK if i == 0 else MUTED,
                         alpha=ease_out((t - 1.2 - 0.3 * i) / 0.5)) + cv.px(12)
        cv.source(d, self.src("tsvad_turn/ami/hybrid_tsvad", "tsvad_turn/icsi/hybrid_tsvad", "tsvad_turn/icsi/hybrid_sortformer"))

    def scene_tsvad(self, img, scene, st, ui, prog, T):
        d = ImageDraw.Draw(img)
        self._panel_tsvad(d, st)

    # -- head-to-head scatter (dead air vs cut-ins) inside Pipecat / LiveKit
    def scene_headtohead(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = self.header(d, "Inside Pipecat and LiveKit, on meeting audio", "5 AMI windows, mono microphone, live · x: median dead air after the user's last word · y: cut-ins per minute · lower-left is better")
        keys = [(fw, s) for fw in ("pipecat", "livekit") for s in ("A", "B", "C", "CN", "D", "DN")]
        pts = []
        for fw, s in keys:
            k = f"e2e/ami/{fw}/{s}"
            if k in self.N:
                v = self.N[k]["value"]
                pts.append((fw, s, v["dead_air_ms_median"], v["cut_ins_per_min"], v["name"]))
        px0, py0 = cv.m + cv.px(90), top + cv.px(30)
        px1, py1 = cv.W - cv.m - cv.px(40), cv.H - cv.px(250 if not cv.sq else 300)
        xmax, ymax = 3200, 8.5
        d.line([(px0, py1), (px1, py1)], fill=DIM, width=2)
        d.line([(px0, py0), (px0, py1)], fill=DIM, width=2)
        for xv in range(0, xmax + 1, 800):
            x = px0 + (px1 - px0) * xv / xmax
            d.line([(x, py1), (x, py1 + cv.px(8))], fill=DIM)
            cv.text(d, (x, py1 + cv.px(14)), f"{xv / 1000:.1f} s", "mono", 20, MUTED, anchor="ma")
        for yv in range(0, 9, 2):
            y = py1 - (py1 - py0) * yv / ymax
            d.line([(px0 - cv.px(8), y), (px0, y)], fill=DIM)
            cv.text(d, (px0 - cv.px(14), y), f"{yv}", "mono", 20, MUTED, anchor="rm")
        cv.text(d, ((px0 + px1) / 2, py1 + cv.px(44)), "median dead air after the user's last word → lower is faster", "medium", 22, MUTED, anchor="ma")
        cv.text(d, (px0 - cv.px(70), py0 - cv.px(6)), "cut-ins / min ↑", "medium", 22, MUTED, anchor="ld")
        names = {"A": "Pipecat default: Silero + smart-turn v3.2 + Whisper small", "B": "LiveKit default: Silero + EnglishModel + Whisper small",
                 "C": "ours, 1 s timeout (Sortformer v2)", "CN": "ours, 1 s timeout (Nemotron-3)",
                 "D": "ours, hybrid_dyn + arm (Sortformer v2)", "DN": "ours, hybrid_dyn + arm (Nemotron-3)"}
        # legend, top-left of the plot (the data sit lower right)
        ly = py0 + cv.px(10)
        for i, s in enumerate(("A", "B", "C", "CN", "D", "DN")):
            a = ease_out((st - 0.12 * i) / 0.5)
            col = BAD if s in ("A", "B") else (ACCENT if s.startswith("D") else WAVE)
            cv.text(d, (px0 + cv.px(24), ly), s, "heavy", 24, col, alpha=a)
            cv.text(d, (px0 + cv.px(70), ly + cv.px(2)), names[s], "medium", 22, MUTED, alpha=a)
            ly += cv.px(32)
        for i, (fw, s, x, y, name) in enumerate(pts):
            a = ease_out((st - 0.12 * i) / 0.5)
            if a <= 0:
                continue
            X = px0 + (px1 - px0) * min(x, xmax) / xmax
            Y = py1 - (py1 - py0) * min(y, ymax) / ymax
            col = BAD if s in ("A", "B") else (ACCENT if s.startswith("D") else WAVE)
            r = cv.px(17) * a
            if fw == "pipecat":
                d.ellipse([X - r, Y - r, X + r, Y + r], fill=mix(BG, col, a))
            else:
                d.rectangle([X - r, Y - r, X + r, Y + r], fill=mix(BG, col, a))
            cv.text(d, (X, Y), s, "heavy", 17, BG, anchor="mm", alpha=a)
        # marker legend
        lx = cv.m
        lyy = py1 + cv.px(80)
        d.ellipse([lx, lyy, lx + cv.px(18), lyy + cv.px(18)], fill=MUTED)
        cv.text(d, (lx + cv.px(28), lyy - cv.px(4)), "Pipecat 1.12", "medium", 22, MUTED)
        d.rectangle([lx + cv.px(200), lyy, lx + cv.px(218), lyy + cv.px(18)], fill=MUTED)
        cv.text(d, (lx + cv.px(228), lyy - cv.px(4)), "LiveKit Agents 1.8", "medium", 22, MUTED)
        cv.text(d, (lx + cv.px(470), lyy - cv.px(4)), "red: framework default stack · blue: ours, 1 s timeout · amber: ours, opt-in hybrid rule (0 cut-ins, more dead air) · n = 5, CIs wide",
                "medium", 22, MUTED)
        # second unit: first-partial latency across all 37 clips x both frameworks
        a2 = ease_out((st - (scene["units"][1]["start"] - scene["start"])) / 0.6) if len(scene["units"]) > 1 else 0.0
        if a2 > 0:
            N = self.num
            o, p, l = N("e2e_first_text_ms/ours_timeout"), N("e2e_first_text_ms/pipecat_default"), N("e2e_first_text_ms/livekit_default")
            box = (cv.W / 2 - cv.px(40), top + cv.px(40), cv.W - cv.m - cv.px(40), top + cv.px(250 if not cv.sq else 230))
            d.rounded_rectangle(box, radius=cv.px(16), fill=mix(BG, PANEL, a2))
            x, y = box[0] + cv.px(24), box[1] + cv.px(18)
            cv.text(d, (x, y), "first partial transcript after speech onset (median, all sets)", "demi", 24 if not cv.sq else 20, MUTED, alpha=a2)
            rows = [("ours", f"{o[0] / 1000:.1f} – {o[1] / 1000:.1f} s", WAVE), ("Pipecat default", f"{p[0] / 1000:.1f} – {p[1] / 1000:.1f} s", BAD),
                    ("LiveKit default", f"{l[0] / 1000:.1f} – {l[1] / 1000:.1f} s", BAD)]
            y += cv.px(40)
            for lab, val, col in rows:
                cv.text(d, (x, y), lab, "medium", 26 if not cv.sq else 20, INK, alpha=a2)
                cv.text(d, (box[2] - cv.px(24), y), val, "heavy", 30 if not cv.sq else 22, col, anchor="ra", alpha=a2)
                y += cv.px(48 if not cv.sq else 40)
        cv.source(d, ["runs/e2e_final.json"])

    # -- tradeoff
    def scene_tradeoff(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = self.header(d, "The honest tradeoff: polite, not fast",
                          "TurnBench dev, official scorer · median latency, true end of turn → decision" if cv.sq else
                          "TurnBench dev, official scorer · median latency from the true end of turn to the decision · held-out operating points")
        N = self.num
        vap, ours = N("turnbench_p50/vap"), N("turnbench_p50/ours_predictive")
        rv, ro = N("turnbench_recall/vap"), N("turnbench_recall/ours_predictive")
        y = top + cv.px(30)
        al = ease_out(st / 0.6)
        colw = (cv.W - 2 * cv.m) / 2
        big = 100 if not cv.sq else 72
        cv.text(d, (cv.m, y), f"VAP · recall {rv:.2f} · FP 0.045", "demi", 30 if not cv.sq else 24, MUTED, alpha=al)
        cv.text(d, (cv.m, y + cv.px(40)), f"{vap} ms", "heavy", big, DIM, alpha=al)
        x2, y2 = (cv.m + colw, y) if not cv.sq else (cv.m, y + cv.px(40) + cv.px(big * 1.25) + cv.px(20))
        cv.text(d, (x2, y2), f"ours, predictive OR Silero · recall {ro:.2f} · FP 0.092", "demi", 30 if not cv.sq else 24, MUTED, alpha=al)
        cv.text(d, (x2, y2 + cv.px(40)), f"{ours} ms", "heavy", big, ACCENT, alpha=al)
        a2 = ease_out((st - 0.8) / 0.6)
        yp = y2 + cv.px(40) + cv.px(big * 1.25) + cv.px(16)
        yp = cv.para(d, (cv.m, yp), f"≈ {(ours - vap) / 1000:.1f} s later at twice the false positives: we fire after the silence begins, VAP predicts it (its P10 latency is negative). VAP dominates.",
                     "medium", 28 if not cv.sq else 22, cv.W - 2 * cv.m, INK, alpha=a2)
        if self.cut == "full":
            a3 = ease_out((st - (scene["units"][1]["start"] - scene["start"])) / 0.6) if len(scene["units"]) > 1 else a2
            y2 = yp + cv.px(50 if not cv.sq else 30)
            cv.text(d, (cv.m, y2), "what it is not good at (all measured, research/FINAL_REPORT.md scorecard):", "demi", 28 if not cv.sq else 22, MUTED, alpha=a3)
            o, s = N("turn_miss_open/ours_head_or_silero"), N("turn_miss_open/silero_timeout")
            items = [("speaker verification", f"{N('spk_eer/shipped'):.1f}% EER vs TitaNet-L 8.2% (AMI, within meeting)"),
                     ("diarization head", "frame DER 0.394 vs Sortformer v2 0.201 — the product ships NVIDIA's diarizer"),
                     ("language ID head", "75% at 2 s vs AmberNet 95% — off by default, AmberNet wired in"),
                     ("floor-open turn ends", f"best rule {o:.1f}% missed vs Silero timeout {s:.1f}% (AMI, n = 236): a timeout is as good or better"),
                     ("compute", "1.7–3.6x the CPU of the Pipecat / LiveKit default stacks (keeps up at 1x)")]
            yy = y2 + cv.px(44)
            for i, (a, b) in enumerate(items):
                ai = ease_out((st - (scene["units"][1]["start"] - scene["start"]) - 0.25 * i) / 0.5) if len(scene["units"]) > 1 else a2
                cv.text(d, (cv.m, yy), a, "demi", 26 if not cv.sq else 20, BAD, alpha=ai)
                cv.text(d, (cv.m + cv.px(400 if not cv.sq else 250), yy), b, "medium", 26 if not cv.sq else 18, INK, alpha=ai)
                yy += cv.px(42 if not cv.sq else 36)
            cv.source(d, self.src("turnbench_p50/vap", "turnbench_p50/ours_predictive", "turn_miss_open/ours_head_or_silero", "turn_miss_open/silero_timeout", "spk_eer/shipped"))
        else:
            cv.source(d, self.src("turnbench_p50/vap", "turnbench_p50/ours_predictive", "turnbench_recall/ours_predictive"))

    # -- what's next
    def scene_next(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        top = self.header(d, "What is next")
        items = [("serve the target-speaker head", "TS-VAD track → turn head: 61.9 → 39.3 % (AMI) and 68.5 → 20.4 % (ICSI) missed ends offline; needs its own operating point and a served path"),
                 ("bind the user from the agent's own TTS end", "measured with a label stand-in: 61.9 → 51.6 % misses with TitaNet-L following (research/EOT_BENCH_V2.md §9)"),
                 ("a trigger that fires before the silence", "predictive head on real two-party audio; VAP-style negative latency; the encoder stays frozen (WER gate)"),
                 ("everything is open", "code, benchmarks, negative results and the corrections log")]
        y = top + cv.px(50)
        for i, (a, b) in enumerate(items):
            al = ease_out((st - 0.35 * i) / 0.5)
            cv.text(d, (cv.m, y), f"{i + 1}", "heavy", 60 if not cv.sq else 48, ACCENT, alpha=al)
            cv.text(d, (cv.m + cv.px(80), y + cv.px(6)), a, "demi", 42 if not cv.sq else 32, INK, alpha=al)
            cv.para(d, (cv.m + cv.px(80), y + cv.px(64 if not cv.sq else 50)), b, "medium", 26 if not cv.sq else 22, cv.W - 2 * cv.m - cv.px(80), MUTED, alpha=al)
            y += cv.px(150 if not cv.sq else 130)
        cv.text(d, (cv.W - cv.m, cv.H - cv.px(26)), "research/IMPROVE_115M.md · research/EOT_BENCH_V2.md §9 · research/DYADIC.md §8", "mono", 22, MUTED, anchor="rd")

    # -- end card
    def scene_endcard(self, img, scene, st, ui, prog, T):
        cv = self.cv
        d = ImageDraw.Draw(img)
        al = ease_out(st / 0.4)
        cy = cv.H / 2 - cv.px(120 if not cv.sq else 160)
        cv.text(d, (cv.W / 2, cy - cv.px(120)), "one streaming model · ASR + VAD + speaker + turn-taking", "demi", 34 if not cv.sq else 28, MUTED, anchor="ma", alpha=al)
        cv.text(d, (cv.W / 2, cy - cv.px(60)), "audioforge", "heavy", 120 if not cv.sq else 96, INK, anchor="ma", alpha=al)
        d.rounded_rectangle([cv.m, cy + cv.px(110), cv.W - cv.m, cy + cv.px(210 if not cv.sq else 240)], radius=cv.px(20), fill=mix(BG, PANEL, al))
        url_size = 50 if not cv.sq else 40
        cv.text(d, (cv.W / 2, cy + cv.px(160 if not cv.sq else 175)), REPO_URL, "mono", url_size, ACCENT, anchor="mm", alpha=al)
        cv.text(d, (cv.W / 2, cy + cv.px(250 if not cv.sq else 280)), "code · benchmarks · negative results · corrections log", "medium", 28 if not cv.sq else 24, MUTED, anchor="ma", alpha=al)
        credit = "voice-over: Qwen3-TTS 1.7B (Apache-2.0) · NVIDIA weights CC-BY-4.0 · AMI / otoSpeech / LibriSpeech CC-BY-4.0"
        if cv.sq:
            cv.text(d, (cv.W / 2, cv.H - cv.px(84)), "voice-over: Qwen3-TTS 1.7B (Apache-2.0) · NVIDIA weights CC-BY-4.0", "mono", 18, DIM, anchor="md")
            cv.text(d, (cv.W / 2, cv.H - cv.px(56)), "AMI / otoSpeech / LibriSpeech CC-BY-4.0", "mono", 18, DIM, anchor="md")
        else:
            cv.text(d, (cv.W / 2, cv.H - cv.px(40)), credit, "mono", 18, DIM, anchor="md")


# ----------------------------------------------------------------------------- audio
def resample(x: np.ndarray, sr: int, out_sr: int) -> np.ndarray:
    if sr == out_sr:
        return x.astype(np.float32)
    n = int(round(len(x) * out_sr / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def music_bed(dur: float, sr: int, seed: int = 7) -> np.ndarray:
    """A generated ambient pad (public domain by construction): slow detuned sines on two alternating chords, low-passed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    chords = [[110.0, 164.81, 220.0, 261.63], [98.0, 146.83, 196.0, 246.94]]  # A min7-ish, G add9-ish, low
    out = np.zeros_like(t)
    period = 16.0
    for ci, ch in enumerate(chords):
        w = 0.5 * (1 + np.sin(2 * np.pi * (t / period) - np.pi / 2 + np.pi * ci))  # crossfade between chords
        w = w ** 2
        for f in ch:
            det = 1 + rng.normal(0, 0.0015)
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * t * rng.uniform(0.05, 0.12) + rng.uniform(0, 6.28))
            out += w * lfo * np.sin(2 * np.pi * f * det * t + rng.uniform(0, 6.28)) / len(ch)
    # one-pole low-pass, twice
    y = out.copy()
    a = math.exp(-2 * math.pi * 900 / sr)
    for _ in range(2):
        acc = 0.0
        for i in range(len(y)):
            acc = a * acc + (1 - a) * y[i]
            y[i] = acc
    y /= max(1e-6, np.abs(y).max())
    fade = int(sr * 2.5)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y.astype(np.float32)


def build_audio(r: Renderer, narr_dir: Path, music: bool) -> np.ndarray:
    n = int(math.ceil(r.duration * SR)) + SR
    voice = np.zeros(n, np.float32)
    for s in r.scenes:
        for u in s["units"]:
            w, sr = sf.read(u["wav"] if Path(u["wav"]).is_absolute() else ROOT / u["wav"], dtype="float32")
            w = resample(w, sr, SR)
            i = int(u["start"] * SR)
            voice[i: i + len(w)] += w[: n - i]
    # narration loudness: about -14 dBFS RMS on speech (the mix lands near -17 LUFS integrated, LinkedIn-safe), soft-limited
    rms = np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2)) if np.any(np.abs(voice) > 0.01) else 0.1
    voice *= min(6.0, 10 ** (-14 / 20) / max(rms, 1e-6))
    voice = np.tanh(voice * 1.15) / np.tanh(1.15)  # gentle peak limiting instead of clipping
    # clip audio during demo scenes, ducked under the narration
    clipmix = np.zeros(n, np.float32)
    for s in r.scenes:
        if "clip" not in s:
            continue
        c = r.clips[s["clip"]]
        w = resample(c.wav, c.sr, SR)
        off = int(s["offset"] * SR)
        seg = w[off: off + int((s["end"] - s["start"]) * SR)]
        i = int(s["start"] * SR)
        seg = seg.copy()
        f = min(len(seg), int(0.4 * SR))
        seg[-f:] *= np.linspace(1, 0, f)
        gain = 10 ** (-9 / 20) if not s.get("bg") else 10 ** (-20 / 20)
        clipmix[i: i + len(seg)] += seg[: n - i] * gain
    env = np.abs(voice)
    k = int(0.05 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    duck = 1 - 0.65 * np.clip(env / 0.08, 0, 1)
    # smooth the duck (attack 30 ms, release 400 ms)
    d2 = np.empty_like(duck)
    acc, a_att, a_rel = 1.0, math.exp(-1 / (0.03 * SR)), math.exp(-1 / (0.4 * SR))
    for i in range(len(duck)):
        a = a_att if duck[i] < acc else a_rel
        acc = a * acc + (1 - a) * duck[i]
        d2[i] = acc
    out = voice + clipmix * d2
    if music:
        bed = music_bed(len(out) / SR, SR)[: len(out)]
        out += bed * (10 ** (-30 / 20)) * (0.55 + 0.45 * d2)
    out = np.tanh(out * 1.05) / np.tanh(1.05) * 0.95  # final soft limiter, peaks <= -0.45 dBFS
    return out[: int(r.duration * SR)]


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", choices=["full", "short"], required=True)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--preview", type=float, default=None, help="render one frame at this time to <out>.png and exit")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--seg", type=int, default=None, help="render only video segment SEG of --nseg (machine rule: short calls); then --mux")
    ap.add_argument("--nseg", type=int, default=1)
    ap.add_argument("--mux", action="store_true", help="concatenate the rendered segments, mix the audio and write --out")
    ap.add_argument("--dry", action="store_true", help="no narration yet: estimate unit timings from the script (previews only)")
    a = ap.parse_args(argv)
    W, H = (int(v) for v in a.size.lower().split("x"))
    npath = OUT_DIR / f"narration_{a.cut}.json"
    if a.dry or not npath.exists():
        if a.preview is None:
            sys.exit(f"{npath} missing: run demo/tts.py synth first (--dry works for --preview only)")
        units, scene, t = [], "intro", 0.0
        for line in (DEMO / f"script_{a.cut}.md").read_text().splitlines():
            line = line.strip()
            if line.startswith("## scene:"):
                scene = line.split(":", 1)[1].strip()
            elif line.startswith("- "):
                dur = 0.07 * len(line)  # ~14 chars/s at a slow narration pace
                units.append({"scene": scene, "text": line[2:].strip(), "start": t, "end": t + dur, "wav": ""})
                t += dur
        narr = {"units": units}
    else:
        narr = json.loads(npath.read_text())
    numbers = json.loads((DEMO / "numbers.json").read_text())
    clips = {p.stem: Clip(p) for p in sorted((DEMO / "events").glob("*.json"))}
    r = Renderer(a.cut, W, H, narr, numbers, clips)
    print(f"{a.cut} {W}x{H}: {r.duration:.1f} s, scenes:")
    for s in r.scenes:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  {s['name']}")
    if a.preview is not None:
        r.frame(a.preview).save(str(a.out) + ".png")
        print("preview ->", str(a.out) + ".png")
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    n_frames = int(r.duration * FPS)
    import time
    t0 = time.time()
    if not a.mux:  # video only (one segment or all of it)
        lo = 0 if a.seg is None else a.seg * n_frames // a.nseg
        hi = n_frames if a.seg is None else (a.seg + 1) * n_frames // a.nseg
        seg_out = SCRATCH / f"{out.stem}.seg{0 if a.seg is None else a.seg:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
               "-profile:v", "high", str(seg_out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(lo, hi):
            p.stdin.write(r.frame(i / FPS).tobytes())
            if (i - lo) % 300 == 0:
                print(f"  frame {i}/{n_frames}  {time.time() - t0:.0f} s", flush=True)
        p.stdin.close()
        p.wait()
        if p.returncode:
            sys.exit(f"ffmpeg failed on {seg_out}")
        print("wrote", seg_out, f"frames {lo}-{hi} in {time.time() - t0:.0f} s")
        if a.seg is not None:
            return
        segs = [seg_out]
    else:
        segs = sorted(SCRATCH.glob(f"{out.stem}.seg*.mp4"))
        if len(segs) != a.nseg:
            sys.exit(f"expected {a.nseg} segments for {out.stem}, found {len(segs)}: {segs}")
    audio = build_audio(r, DEMO, music=not a.no_music)
    wav_path = SCRATCH / (out.stem + ".mix.wav")
    sf.write(wav_path, audio, SR)
    lst = SCRATCH / (out.stem + ".segs.txt")
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav_path),
           "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    wav_path.unlink(missing_ok=True)
    print("wrote", out, f"{r.duration:.1f} s in {time.time() - t0:.0f} s")
    (out.with_suffix(".timeline.json")).write_text(json.dumps({"duration_s": r.duration, "scenes": [
        {"name": s["name"], "start": round(s["start"], 2), "end": round(s["end"], 2), "clip": s.get("clip")} for s in r.scenes]}, indent=1))


if __name__ == "__main__":
    main()
