"""Showcase v4 (90-120 s): v3's split screens and plain narration as the spine, plus an animated "how it works" flow,
the voice-print flow, four animated graphs (WER, missed turn ends, product, compute), a navy-to-indigo gradient with a
moving light, warm amber for the user, cool cyan for ours, red only for cut-ins, and 300 ms slides on the beat.

    PYTHONPATH=. .venv/bin/python demo/render_v4.py --storyboard $DEMO_OUT/storyboard_v4 --dry [--size 1080x1080]
    PYTHONPATH=. .venv/bin/python demo/render_v4.py --size 1920x1080 --out $DEMO_OUT/showcase_v4.mp4 [--seg i --nseg n | --mux]

Data: render_v2.Data (E2E per-clip records, served TS-VAD session, recorded server events, clip audio) and
demo/numbers.json (every number with its runs/*.json path). v3's call-window drawing is reused with the v4 palette.
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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_v2 as V2  # noqa: E402
import render_v3 as V3  # noqa: E402
from render_v2 import Data, font, ease, mix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
SCRATCH = OUT_DIR / "scratch"
FPS, SR = 30, 48000
BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

# ----------------------------------------------------------------------------- palette
NAVY = (10, 16, 46)
INDIGO = (30, 28, 92)
CARD = (24, 30, 72)
CARD2 = (34, 42, 96)
INK = (247, 248, 252)        # 15.5:1 on NAVY, 12.3:1 on CARD
MUTED = (186, 192, 224)      # 8.5:1 on CARD
DIM = (120, 128, 170)
USER = (255, 184, 77)        # amber: the user
USER_D = (242, 169, 59)      # user bubble: amber #F2A93B; the bars in it are navy (9.5:1)
OURS = (72, 214, 255)        # cyan: ours
OURS_D = (20, 110, 150)      # ours bubble fill
GREY = (110, 118, 150)       # default stack's agent bubble
RED = (255, 96, 96)
GOOD = (96, 224, 160)
SPK = [(72, 214, 255), (255, 184, 77), (200, 150, 255), (110, 230, 170)]


def contrast(a, b):
    la, lb = sorted((V3.lum(a), V3.lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


assert contrast(INK, NAVY) >= 7 and contrast(INK, CARD) >= 7 and contrast(MUTED, CARD) >= 7 and contrast(NAVY, USER_D) >= 7, "palette contrast"

# v3's call-window drawing reads its module palette at call time: repaint it once with the v4 colours
V3.BG, V3.BG2, V3.CARD, V3.BUBBLE = NAVY, INDIGO, CARD, USER_D
V3.INK, V3.MUTED, V3.DIM = INK, MUTED, MUTED  # the header timer used DIM at 3:1; MUTED keeps it >= 7:1
V3.ACCENT, V3.ACCENT_SOFT, V3.GREY_AGENT, V3.RED, V3.SPK = OURS_D, (18, 84, 116), GREY, RED, SPK
V3.BUBBLE_INK = NAVY

# ----------------------------------------------------------------------------- shots
SHOTS = [
    dict(name="title", bars=2, phrase=None, fine=None, ui="title", clip="ami_ES2011b_027", events="ami_ES2011b_027.mono", offset=0.0),
    dict(name="two_systems", bars=3, phrase=None, fine=None, ui="split", clip="ami_IB4002_119", offset=3.4, left=("pipecat", "A"), right=("pipecat", "C")),
    dict(name="interrupt", bars=3, phrase="Fewer interruptions", fine="1.6–5.5x fewer cut-ins than the Pipecat default stack across 37 recorded conversations · runs/e2e_final.json",
         ui="split", clip="ami_IB4002_119", offset=9.0, left=("pipecat", "A"), right=("pipecat", "C")),
    dict(name="deadair", bars=3, phrase=None, fine="dead air after the user stops: similar to the defaults across 37 conversations (lower on three clip sets, higher on two) · runs/e2e_final.json",
         ui="split", clip="ami_IS1008b_003", offset=12.0, left=("pipecat", "A"), right=("pipecat", "C"), deadair=True),
    dict(name="words", bars=3, phrase="Words while you speak", fine="first partial transcript 0.8–1.2 s after speech onset vs 2.6–3.7 s (Pipecat default) and 3.4–11.3 s (LiveKit default) · runs/e2e_final.json",
         ui="split", clip="ami_ES2011b_027", offset=0.0, left=("livekit", "B"), right=("livekit", "C")),
    dict(name="room", bars=4, phrase="Knows whose turn it is", fine="AMI meetings, 974 turn ends: 61.9 % missed vs 70–74 % for smart-turn, LiveKit and NVIDIA EOU, all at ≤ 5 % false cutoffs · runs/baselines_turn.json",
         ui="room", clip="ami_IS1008b_003", events="ami_IS1008b_003_agentend", offset=6.0),
    dict(name="how", bars=8, phrase="How it works", fine="live values from a recorded session (AMI IS1008b) · encoder: stt_en_fastconformer_hybrid_large_streaming_multi, frozen · heads: audioforge · diarizer: Nemotron-3 (4 of 8 columns served) · final: Parakeet-TDT 0.6B v3",
         ui="how", clip="ami_IS1008b_003", events="ami_IS1008b_003_agentend", offset=0.0),
    dict(name="vpflow", bars=3, phrase="Give it a voice print", fine="TS-VAD head, 0.26 M parameters on block 4 of the same encoder pass; 5 s print embedded by the served speaker head · served session on AMI TS3004b · research/IMPROVEMENTS.md §1",
         ui="vpflow", clip="ami_TS3004b_064", events="ami_TS3004b_064_tsvad", offset=9.5),
    dict(name="voiceprint", bars=3, phrase=None, fine="missed turn ends 61.9 → 39.3 % on AMI, 68.5 → 20.4 % on held-out ICSI, offline benchmark; this clip is one live session · runs/improve_115m.json",
         ui="split", clip="ami_TS3004b_064", offset=12.4, left=("pipecat", "C"), right=("pipecat", "T"), tags=("no voice print", "voice print"), deadair=True),
    dict(name="graph_wer", bars=3, phrase="Transcripts", fine="word error rate, Whisper-normalised, 200 utterances each · LibriSpeech test-clean and AMI dev single-speaker segments · runs/final_asr.json, runs/hybrid_asr.json", ui="graph_wer"),
    dict(name="graph_turn", bars=3, phrase="Missed turn ends", fine="6 s horizon, every system held to ≤ 5 % false cutoffs · AMI dev n = 974, ICSI held-out n = 1312 (AMI thresholds frozen; TS-VAD rows cross-fitted on ICSI) · runs/baselines_turn*.json, runs/improve_115m.json", ui="graph_turn"),
    dict(name="graph_e2e", bars=3, phrase="Inside Pipecat and LiveKit", fine="37 recorded conversations live at 1x · ranges over clip sets · runs/e2e_final.json", ui="graph_e2e"),
    dict(name="graph_rtf", bars=2, phrase="One model. Two CPU threads.", fine="whole server real-time factor, Nemotron-3 diarizer, Apple M5 · runs/e2e_final.json; perf pass: server compute 0.62x after the CPU fast paths · runs/perf.json", ui="graph_rtf"),
    dict(name="honest", bars=2, phrase=None, fine="ours: 1.1–4.5 cut-ins per minute vs 2.1–7.5 (Pipecat default); TurnBench two-party calls: 0.7 s behind VAP · runs/e2e_final.json, runs/dyadic_train.json",
         ui="honest", clip="ami_IS1008b_003", offset=1.5),
    dict(name="end", bars=2, phrase="audioforge", fine="baseline transcripts are the reference text shown at the measured arrival time (session text not retained); all timings from the recorded sessions", ui="end"),
]
LEAD = 0.3
TRANS = 0.3


def build_timeline(units=None):
    t, shots = 0.0, []
    for i, sh in enumerate(SHOTS):
        s = dict(sh)
        dur = s["bars"] * BAR
        u = units[i] if units and i < len(units) else None
        if u is not None:
            need = LEAD + (u["end"] - u["start"]) + 0.3
            while dur < need:
                dur += BAR
            s["unit"] = dict(u, start=t + LEAD, end=t + LEAD + (u["end"] - u["start"]))
        s["start"], s["end"] = t, t + dur
        t += dur
        shots.append(s)
    return shots


def caption_chips(text, max_chars):
    """One whole sentence per chip (two short ones at most); a long sentence breaks only at a clause boundary."""
    import re
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    chips = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        if i + 1 < len(sents) and len(cur) + 1 + len(sents[i + 1]) <= max_chars and (len(cur) < 40 or len(sents[i + 1]) < 40):
            cur = cur + " " + sents[i + 1]; i += 1
        i += 1
        if len(cur) <= max_chars:
            chips.append(cur); continue
        clauses = [c.strip() for c in re.split(r"(?<=[,;:])\s+", cur) if c.strip()]
        acc = ""
        for c in clauses:
            if acc and len(acc) + 1 + len(c) > max_chars:
                chips.append(acc); acc = c
            else:
                acc = (acc + " " + c).strip()
        if acc:
            chips.append(acc)
    return chips or [text]


def dry_units(spc=0.062):
    units, scene, t = [], "title", 0.0
    for line in (DEMO / "script_v4.md").read_text().splitlines():
        line = line.strip()
        if line.startswith("## scene:"):
            scene = line.split(":", 1)[1].strip()
        elif line.startswith("- "):
            d = spc * len(line)
            units.append({"scene": scene, "text": line[2:].strip(), "start": t, "end": t + d, "wav": ""}); t += d
    return units


# ----------------------------------------------------------------------------- canvas / background
class Canvas(V3.Canvas):
    def text(self, d, xy, s, kind="medium", size=40, fill=INK, anchor="la", alpha=1.0, bg=None):
        V2.Canvas.text(self, d, xy, s, kind, size, fill, anchor, alpha, bg or NAVY)


_GRAD = {}


def background(cv, T):
    """Vertical navy → indigo gradient with a soft light that drifts slowly (idle motion on every shot)."""
    key = (cv.W, cv.H)
    if key not in _GRAD:
        g = np.linspace(0, 1, cv.H)[:, None]
        rows = (np.array(NAVY) * (1 - g) + np.array(INDIGO) * g).astype(np.uint8)
        _GRAD[key] = Image.fromarray(np.repeat(rows[:, None, :], cv.W, axis=1), "RGB")
    img = _GRAD[key].copy()
    small = Image.new("L", (cv.W // 10, cv.H // 10), 0)
    d = ImageDraw.Draw(small)
    cx = cv.W / 20 + math.sin(T * 0.21) * cv.W / 32
    cy = cv.H / 20 + math.cos(T * 0.17) * cv.H / 30
    r = cv.W / 26
    d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8], fill=255)
    small = small.filter(ImageFilter.GaussianBlur(cv.W / 90)).resize((cv.W, cv.H))
    light = Image.new("RGB", (cv.W, cv.H), (58, 52, 140))
    return Image.composite(light, img, small.point(lambda v: int(v * 0.55)))


def glow_panel(cv, d, box, fill=CARD, outline=None, radius=None):
    d.rounded_rectangle(box, radius=radius or cv.px(26), fill=fill, outline=outline, width=cv.px(3) if outline else 0)


# ----------------------------------------------------------------------------- graphs
def count(v, a):
    return v * ease(a)


def grouped_bars(cv, d, box, groups, series, colors, t, vmax, unit="%", fmt="{:.1f}", note_better="lower is better", stagger=0.12):
    """Vertical grouped bars growing from zero with the numbers counting up. groups: names; series: {name: [v per group]}."""
    x0, y0, x1, y1 = box
    ng, ns = len(groups), len(series)
    gw = (x1 - x0) / ng
    bw = gw * 0.8 / ns
    base = y1 - cv.px(60)
    top = y0 + cv.px(60)
    d.line([(x0, base), (x1, base)], fill=mix(NAVY, MUTED, 0.5), width=2)
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + gw * 0.1
        cv.text(d, (gx + gw * 0.4, base + cv.px(14)), g, "demi", 30, INK, anchor="ma")
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None:
                continue
            a = ease((t - stagger * (si + gi * ns)) / 0.7)
            h = (base - top) * min(v, vmax) / vmax * a
            bx = gx + si * bw
            col = colors[name]
            d.rounded_rectangle([bx + cv.px(4), base - h, bx + bw - cv.px(4), base], radius=cv.px(8), fill=mix(NAVY, col, 0.35 + 0.65 * a))
            cv.text(d, (bx + bw / 2, base - h - cv.px(8)), fmt.format(count(v, a)) + (unit if not cv.sq else ""), "demi", 26 if not cv.sq else 20, INK, anchor="md", alpha=a)
    # legend, wrapped into rows; the "lower is better" note on its own line
    lx, ly = x0, y0 - cv.px(70)
    maxw = x1 - x0
    for name, col in colors.items():
        if name not in series:
            continue
        w = cv.px(32) + cv.tw(name, "medium", 26) + cv.px(40)
        if lx + w > x0 + maxw:
            lx, ly = x0, ly + cv.px(36)
        d.rounded_rectangle([lx, ly + cv.px(8), lx + cv.px(22), ly + cv.px(30)], radius=cv.px(5), fill=col)
        cv.text(d, (lx + cv.px(32), ly + cv.px(6)), name, "medium", 26, MUTED)
        lx += w
    cv.text(d, (x1, y0 + cv.px(6)), note_better, "medium", 26, MUTED, anchor="ra")


def hbar_ranges(cv, d, box, rows, t, vmax, unit):
    """Horizontal range bars (lo–hi) growing from the left with the numbers counting up. rows: (label, (lo, hi), colour)."""
    x0, y0, x1, y1 = box
    y = y0
    for i, (lab, (lo, hi), col) in enumerate(rows):
        a = ease((t - 0.15 * i) / 0.7)
        cv.text(d, (x0, y), lab, "medium", 28, MUTED, alpha=a)
        cv.text(d, (x1, y), f"{count(lo, a):.1f} – {count(hi, a):.1f}{unit}", "demi", 30, col, anchor="ra", alpha=a)
        d.rounded_rectangle([x0, y + cv.px(42), x1, y + cv.px(74)], radius=cv.px(10), fill=CARD2)
        d.rounded_rectangle([x0 + (x1 - x0) * lo / vmax * a, y + cv.px(42), x0 + (x1 - x0) * min(hi, vmax) / vmax * a, y + cv.px(74)], radius=cv.px(10), fill=col)
        y += cv.px(120)
    return y


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, W, H, data: Data, narr=None):
        self.cv, self.D = Canvas(W, H), data
        self.shots = build_timeline(narr["units"] if narr else None)
        self.duration = self.shots[-1]["end"]
        self.N = data.num

    # -- frame with a 300 ms slide on the cut
    def frame(self, T):
        i = next((k for k, s in enumerate(self.shots) if s["start"] <= T < s["end"]), len(self.shots) - 1)
        sh = self.shots[i]
        st = T - sh["start"]
        cur = self.render_shot(sh, st, T)
        if i > 0 and st < TRANS:
            prev = self.shots[i - 1]
            old = self.render_shot(prev, prev["end"] - prev["start"] - 0.02, prev["end"] - 0.02)
            a = ease(st / TRANS)
            dx = int(self.cv.W * (1 - a))
            out = old.copy()
            out.paste(cur, (dx, 0))
            # a thin light seam on the wipe edge
            ImageDraw.Draw(out).line([(dx, 0), (dx, self.cv.H)], fill=(120, 110, 200), width=max(2, self.cv.px(3)))
            return out
        return cur

    def render_shot(self, sh, st, T):
        cv = self.cv
        img = background(cv, T)
        d = ImageDraw.Draw(img)
        getattr(self, "shot_" + sh["ui"])(img, d, sh, st, T)
        self.overlay(d, sh, st, T)
        return img

    def overlay(self, d, sh, st, T):
        cv = self.cv
        a = min(ease((st - 0.5) / 0.4), 1 - ease((st - (sh["end"] - sh["start"]) + 0.4) / 0.3))
        if sh.get("phrase") and sh["ui"] not in ("end", "title"):
            size = 64 if not cv.sq else 50
            # slide up 20 px while fading in
            cv.text(d, (cv.W / 2, cv.px(74 if not cv.sq else 52) + cv.px(20) * (1 - ease((st - 0.5) / 0.4))), sh["phrase"], "demi", size, INK, anchor="ma", alpha=a)
        if sh.get("fine"):
            lines = cv.wrap(sh["fine"], "medium", 22, cv.W - 2 * cv.m)
            y = cv.H - cv.px(34) - cv.px(28) * len(lines)
            for line in lines:
                cv.text(d, (cv.W / 2, y), line, "medium", 22, MUTED, anchor="ma", alpha=a); y += cv.px(28)
        u = sh.get("unit")
        if u and u["start"] - 0.1 <= T <= u["end"] + 0.4:
            parts = caption_chips(u["text"], 84 if not cv.sq else 56)
            total = sum(len(p) for p in parts)
            t0, txt = u["start"], parts[-1]
            for p in parts:
                dd = (u["end"] - u["start"]) * len(p) / total
                if T < t0 + dd + 0.3:
                    txt = p; break
                t0 += dd
            nfine = len(cv.wrap(sh["fine"], "medium", 22, cv.W - 2 * cv.m)) if sh.get("fine") else 0
            size = 32 if not cv.sq else 28
            lines = cv.wrap(txt, "medium", size, cv.W - 2 * cv.m - cv.px(80))[:2]
            lh = cv.px(size) * 1.25
            fy = cv.H - cv.px(34) - cv.px(28) * nfine - cv.px(24) - lh * len(lines) - cv.px(8)
            w = max(cv.tw(l, "medium", size) for l in lines) + cv.px(40)
            d.rounded_rectangle([cv.W / 2 - w / 2, fy - cv.px(8), cv.W / 2 + w / 2, fy + lh * len(lines) + cv.px(6)], radius=cv.px(12), fill=(6, 8, 24))
            for l in lines:
                cv.text(d, (cv.W / 2, fy), l, "medium", size, INK, anchor="ma", bg=(6, 8, 24)); fy += lh

    # -- layouts shared with v3
    def _boxes(self, split, tall=False):
        cv = self.cv
        if not cv.sq:
            h = cv.px(600 if tall else 520)
            top = cv.px(200)
        else:
            top, h = cv.px(140), cv.H - cv.px(140) - cv.px(260)
        bottom = top + h
        if not split:
            w = cv.px(780 if not cv.sq else 720)
            return [(cv.W / 2 - w / 2, top, cv.W / 2 + w / 2, bottom)]
        if cv.sq:
            hh = (bottom - top - cv.px(20)) / 2
            return [(cv.m, top, cv.W - cv.m, top + hh), (cv.m, top + hh + cv.px(20), cv.W - cv.m, bottom)]
        gap = cv.px(40)
        w = (cv.W - 2 * cv.m - gap) / 2
        return [(cv.m, top, cv.m + w, bottom), (cv.m + w + gap, top, cv.W - cv.m, bottom)]

    def _scale_box(self, box, st):
        """Bubbles / cards scale in over 300 ms."""
        a = ease(st / 0.3)
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = 0.92 + 0.08 * a
        return (cx - (cx - x0) * s, cy - (cy - y0) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s)

    def shot_title(self, img, d, sh, st, T):
        cv = self.cv
        a = ease(st / 0.5)
        cv.text(d, (cv.W / 2, cv.px(90 if not cv.sq else 60)), "audioforge", "heavy", 120 if not cv.sq else 88, INK, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cv.px(236 if not cv.sq else 170)), "one model that listens for your voice agent", "medium", 36 if not cv.sq else 28, MUTED, anchor="ma", alpha=a)
        chips = [("words", OURS), ("speech", INK), ("speaker", USER), ("turn", GOOD)]
        total = sum(cv.tw(c, "demi", 30) + cv.px(28) for c, _ in chips) + cv.px(16) * 3
        x = cv.W / 2 - total / 2
        for i, (c, col) in enumerate(chips):
            ai = ease((st - 0.4 - 0.15 * i) / 0.3)
            w, h = cv.pill(d, (x, cv.px(300 if not cv.sq else 225)), c, mix(NAVY, col, ai), fg=NAVY if col != INK else NAVY, size=30, alpha=ai)
            x += w + cv.px(16)
        # live window under it
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh["events"])
        sess = self.D.session(sh["clip"], "livekit", "C")
        box = self._boxes(False)[0]
        box = (box[0], box[1] + cv.px(180), box[2], box[3] + cv.px(60)) if not cv.sq else (box[0], box[1] + cv.px(180), box[2], box[3])
        V3.draw_app(cv, d, self._scale_box(box, st - 0.6), "audioforge", sess, audio, ev, now, ours=True)

    def shot_split(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh.get("events", sh["clip"]))
        L = self.D.session(sh["clip"], *sh["left"]); R = self.D.session(sh["clip"], *sh["right"])
        tags = sh.get("tags", ("default", "audioforge"))
        b = self._boxes(True)
        V3.draw_app(cv, d, self._scale_box(b[0], st), tags[0], L, audio, ev, now, ours=bool(sh.get("tags")), agent_col=(V3.ACCENT_SOFT if sh.get("tags") else None), deadair=sh.get("deadair", False))
        V3.draw_app(cv, d, self._scale_box(b[1], st - 0.1), tags[1], R, audio, ev, now, ours=True, deadair=sh.get("deadair", False))

    def shot_room(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        V3.draw_room(cv, d, self._scale_box(self._boxes(False, tall=True)[0], st), self.D.audio(sh["clip"]), self.D.events(sh["events"]), now)

    def shot_honest(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        sess = self.D.session(sh["clip"], "livekit", "C")
        V3.draw_app(cv, d, self._scale_box(self._boxes(False)[0], st), "audioforge", sess, self.D.audio(sh["clip"]), self.D.events(sh["clip"]), now, ours=True)

    # -- how it works: live flow driven by a recorded session
    def shot_how(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh["events"])
        n = int(np.searchsorted(ev.f_t, now, side="right"))
        vad = float(ev.f_vad[n - 1]) if n else 0.0
        spk = ev.f_spk[n - 1] if n else np.zeros(4)
        eot = float(ev.f_eot[n - 1]) if n and len(ev.f_eot) else 0.0
        if cv.sq:
            return self._how_square(d, sh, st, now, audio, ev, vad, spk, eot)
        top = cv.px(170 if not cv.sq else 130)
        # column x positions (12-col grid)
        cx0, cx1 = cv.col(0, 2); ex0, ex1 = cv.col(2, 3); hx0, hx1 = cv.col(5, 4); dx0, dx1 = cv.col(9, 3)
        # 1. chunk conveyor: a tile every 160 ms slides into the encoder
        a1 = ease(st / 0.4)
        cy = top + cv.px(150)
        cv.text(d, (cx0, top), "audio, 160 ms chunks", "demi", 28, MUTED, alpha=a1)
        ph = (now % 0.8) / 0.8
        for k in range(4):
            u = (k + ph) / 4
            x = cx0 + (ex0 - cv.px(20) - cx0) * u
            e = audio.env_at(now - (1 - u) * 0.64, now - (1 - u) * 0.64 + 0.16, 1)[0]
            hh = cv.px(18) + e * cv.px(60)
            d.rounded_rectangle([x, cy - hh / 2, x + cv.px(28), cy + hh / 2], radius=cv.px(6), fill=mix(NAVY, USER, 0.35 + 0.65 * u) if a1 > 0 else NAVY)
        # 2. encoder block lights on each chunk arrival
        a2 = ease((st - 0.3) / 0.4)
        glow = 0.55 + 0.45 * (1 - ph)
        enc = (ex0, top, ex1, top + cv.px(420))
        glow_panel(cv, d, enc, fill=mix(NAVY, CARD2, a2), outline=mix(NAVY, OURS, a2 * glow))
        cv.text(d, ((ex0 + ex1) / 2, top + cv.px(24)), "NVIDIA FastConformer", "demi", 30, mix(CARD2, OURS, a2), anchor="ma", bg=CARD2)
        for i, line in enumerate(["streaming encoder", "frozen · 115 M", "one pass per chunk"]):
            cv.text(d, ((ex0 + ex1) / 2, top + cv.px(70) + i * cv.px(34)), line, "medium", 26, mix(CARD2, MUTED, a2), anchor="ma", bg=CARD2)
        # a pulse travelling down the block
        py = top + cv.px(250) + ph * cv.px(140)
        d.rounded_rectangle([ex0 + cv.px(24), py, ex1 - cv.px(24), py + cv.px(6)], radius=cv.px(3), fill=mix(CARD2, OURS, a2 * (1 - ph)))
        # 3. heads with live widgets
        heads = [("speech?", INK), ("who?", USER), ("turn over?", GOOD), ("words", OURS)]
        hh = cv.px(96)
        for i, (t, c) in enumerate(heads):
            ai = ease((st - 0.9 - 0.35 * i) / 0.4)
            yy = top + i * (hh + cv.px(12))
            # self-drawing line encoder -> head
            p0 = (ex1, top + cv.px(210)); p1 = (hx0, yy + hh / 2)
            pts = [(p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * (3 * u * u - 2 * u ** 3)) for u in np.linspace(0, 1, 30)]
            k = max(2, int(len(pts) * ai))
            if ai > 0:
                d.line(pts[:k], fill=mix(NAVY, c, 0.8), width=cv.px(3))
            glow_panel(cv, d, (hx0, yy, hx1, yy + hh), fill=mix(NAVY, CARD, ai), outline=mix(NAVY, c, ai), radius=cv.px(18))
            cv.text(d, (hx0 + cv.px(18), yy + hh / 2), t, "demi", 30, mix(CARD, c, ai), anchor="lm", bg=CARD)
            wx = hx0 + cv.px(190)
            if ai <= 0:
                continue
            if i == 0:  # VAD lamp
                on = vad > 0.5
                d.ellipse([wx, yy + hh / 2 - cv.px(14), wx + cv.px(28), yy + hh / 2 + cv.px(14)], fill=mix(CARD, GOOD if on else DIM, 0.9), outline=INK if on else None, width=2)
                cv.text(d, (wx + cv.px(44), yy + hh / 2), f"speech  p = {vad:.2f}" if on else f"silence  p = {vad:.2f}", "medium", 26, INK, anchor="lm", bg=CARD)
            elif i == 1:  # speaker map: dot moves to the active speaker's anchor
                anchors = [(0.2, 0.3), (0.75, 0.25), (0.3, 0.8), (0.8, 0.75)]
                mw, mh = cv.px(170), hh - cv.px(16)
                d.rounded_rectangle([wx, yy + cv.px(8), wx + mw, yy + cv.px(8) + mh], radius=cv.px(8), fill=mix(CARD, NAVY, 0.6))
                for kk, (ax_, ay_) in enumerate(anchors):
                    d.ellipse([wx + ax_ * mw - cv.px(4), yy + cv.px(8) + ay_ * mh - cv.px(4), wx + ax_ * mw + cv.px(4), yy + cv.px(8) + ay_ * mh + cv.px(4)], fill=mix(CARD, SPK[kk], 0.6))
                act = int(np.argmax(spk)) if spk.max() > 0.5 else None
                if act is not None:
                    ax_, ay_ = anchors[act]
                    d.ellipse([wx + ax_ * mw - cv.px(10), yy + cv.px(8) + ay_ * mh - cv.px(10), wx + ax_ * mw + cv.px(10), yy + cv.px(8) + ay_ * mh + cv.px(10)], fill=SPK[act], outline=INK, width=2)
                cv.text(d, (wx + mw + cv.px(16), yy + hh / 2), f"speaker {act}" if act is not None else "–", "medium", 26, INK, anchor="lm", bg=CARD)
            elif i == 2:  # turn meter rising to its threshold; fires at the recorded turn_end
                mw = cv.px(230)
                d.rounded_rectangle([wx, yy + hh / 2 - cv.px(10), wx + mw, yy + hh / 2 + cv.px(10)], radius=cv.px(6), fill=mix(CARD, NAVY, 0.6))
                thr = 0.998
                lvl = min(1.0, max(0.0, (eot - 0.5) / (thr - 0.5)))
                fired = any(0 <= now - te["t"] < 0.6 for te in ev.turn_ends)
                d.rounded_rectangle([wx, yy + hh / 2 - cv.px(10), wx + mw * lvl, yy + hh / 2 + cv.px(10)], radius=cv.px(6), fill=GOOD if fired else mix(CARD, GOOD, 0.6))
                d.line([(wx + mw * 0.97, yy + hh / 2 - cv.px(16)), (wx + mw * 0.97, yy + hh / 2 + cv.px(16))], fill=INK, width=2)
                cv.text(d, (wx + mw + cv.px(16), yy + hh / 2), "turn end" if fired else f"p = {eot:.3f}", "demi" if fired else "medium", 26, GOOD if fired else INK, anchor="lm", bg=CARD)
            else:  # words
                txt = ev.live_text(now)[0]
                f = font("medium", cv.px(26))
                while f.getlength(txt) > (hx1 - wx - cv.px(30)) and " " in txt:
                    txt = txt.split(" ", 1)[1]
                cv.text(d, (wx, yy + hh / 2), (txt + " |") if txt else "…", "medium", 26, INK, anchor="lm", bg=CARD)
        # 4. diarizer lanes beside
        a4 = ease((st - 2.6) / 0.4)
        dy0 = top
        glow_panel(cv, d, (dx0, dy0, dx1, dy0 + cv.px(420)), fill=mix(NAVY, CARD, a4), outline=mix(NAVY, USER, a4))
        cv.text(d, (dx0 + cv.px(18), dy0 + cv.px(18)), "Nemotron-3 diarizer", "demi", 28, mix(CARD, USER, a4), bg=CARD)
        cv.text(d, (dx0 + cv.px(18), dy0 + cv.px(52)), "eight speaker lanes · four served here", "medium", 24, mix(CARD, MUTED, a4), bg=CARD)
        lane_y = dy0 + cv.px(92)
        vis = ev.f_t <= now
        win = 6.0
        lw = dx1 - dx0 - 2 * cv.px(18)
        for k in range(8):
            ly = lane_y + k * cv.px(40)
            d.rounded_rectangle([dx0 + cv.px(18), ly, dx1 - cv.px(18), ly + cv.px(28)], radius=cv.px(6), fill=mix(CARD, NAVY, (0.5 if k < 4 else 0.25) * a4))
            if k >= 4:
                continue
            for tt, s_ in zip(ev.f_t[vis], ev.f_spk[vis]):
                if tt < now - win or s_[k] <= 0.5:
                    continue
                x = dx0 + cv.px(18) + (tt - (now - win)) / win * lw
                d.rectangle([x - 1, ly + cv.px(5), x + max(1, lw / (win / 0.08)), ly + cv.px(23)], fill=mix(CARD, SPK[k], (0.4 + 0.6 * float(s_[k])) * a4))
        # 5. TDT rewrite below
        a5 = ease((st - 3.4) / 0.4)
        ty = top + cv.px(450)
        glow_panel(cv, d, (ex0, ty, dx1, ty + cv.px(150)), fill=mix(NAVY, CARD, a5), outline=mix(NAVY, GOOD, a5), radius=cv.px(18))
        cv.text(d, (ex0 + cv.px(18), ty + cv.px(14)), "Parakeet-TDT 0.6B v3 · rewrites each finished turn", "demi", 26, mix(CARD, GOOD, a5), bg=CARD)
        fins = [f for f in ev.finals if f["arr"] <= now and f.get("source") == "tdt_v3" and (f.get("text") or "").strip()]
        if fins:
            f_ = fins[-1]
            age = now - f_["arr"]
            txt = f_["text"]
            nchar = int(len(txt) * min(1.0, age / 0.6))
            cv.para(d, (ex0 + cv.px(18), ty + cv.px(52)), txt[:nchar], "medium", 26, dx1 - ex0 - cv.px(36), INK, bg=CARD)
        else:
            cv.text(d, (ex0 + cv.px(18), ty + cv.px(52)), "waiting for the first turn end …", "medium", 26, mix(CARD, MUTED, a5), bg=CARD)
        cv.text(d, (cv.m, top + cv.px(620)), f"t = {now:4.1f} s · AMI meeting IS1008b · every value on this screen is a recorded event", "medium", 24, MUTED)

    def _how_square(self, d, sh, st, now, audio, ev, vad, spk, eot):
        """1:1 layout: encoder strip, four head rows, four diarizer lanes, TDT box, stacked."""
        cv = self.cv
        x0, x1 = cv.m, cv.W - cv.m
        y = cv.px(120)
        ph = (now % 0.8) / 0.8
        a2 = ease((st - 0.3) / 0.4)
        # conveyor + encoder in one strip
        for k in range(4):
            u = (k + ph) / 4
            x = x0 + (cv.px(150)) * u
            e = audio.env_at(now - (1 - u) * 0.64, now - (1 - u) * 0.64 + 0.16, 1)[0]
            hh = cv.px(12) + e * cv.px(40)
            d.rounded_rectangle([x, y + cv.px(45) - hh / 2, x + cv.px(20), y + cv.px(45) + hh / 2], radius=cv.px(5), fill=mix(NAVY, USER, 0.35 + 0.65 * u))
        enc = (x0 + cv.px(190), y, x1, y + cv.px(90))
        glow_panel(cv, d, enc, fill=mix(NAVY, CARD2, a2), outline=mix(NAVY, OURS, a2 * (0.55 + 0.45 * (1 - ph))), radius=cv.px(18))
        cv.text(d, (enc[0] + cv.px(18), y + cv.px(12)), "NVIDIA FastConformer · frozen · 115 M", "demi", 26, mix(CARD2, OURS, a2), bg=CARD2)
        cv.text(d, (enc[0] + cv.px(18), y + cv.px(50)), "160 ms chunks in, one pass per chunk", "medium", 24, mix(CARD2, MUTED, a2), bg=CARD2)
        y += cv.px(110)
        heads = [("speech?", INK), ("who?", USER), ("turn over?", GOOD), ("words", OURS)]
        hh = cv.px(64)
        for i, (t, c) in enumerate(heads):
            ai = ease((st - 0.9 - 0.35 * i) / 0.4)
            yy = y + i * (hh + cv.px(10))
            glow_panel(cv, d, (x0, yy, x1, yy + hh), fill=mix(NAVY, CARD, ai), outline=mix(NAVY, c, ai), radius=cv.px(16))
            cv.text(d, (x0 + cv.px(16), yy + hh / 2), t, "demi", 26, mix(CARD, c, ai), anchor="lm", bg=CARD)
            wx = x0 + cv.px(170)
            if ai <= 0:
                continue
            if i == 0:
                on = vad > 0.5
                d.ellipse([wx, yy + hh / 2 - cv.px(12), wx + cv.px(24), yy + hh / 2 + cv.px(12)], fill=mix(CARD, GOOD if on else DIM, 0.9))
                cv.text(d, (wx + cv.px(36), yy + hh / 2), f"{'speech' if on else 'silence'}  p = {vad:.2f}", "medium", 24, INK, anchor="lm", bg=CARD)
            elif i == 1:
                act = int(np.argmax(spk)) if spk.max() > 0.5 else None
                for kk in range(4):
                    cx = wx + kk * cv.px(40)
                    r = cv.px(12) if act == kk else cv.px(7)
                    d.ellipse([cx - r, yy + hh / 2 - r, cx + r, yy + hh / 2 + r], fill=SPK[kk] if act == kk else mix(CARD, SPK[kk], 0.5))
                cv.text(d, (wx + cv.px(180), yy + hh / 2), f"speaker {act}" if act is not None else "–", "medium", 24, INK, anchor="lm", bg=CARD)
            elif i == 2:
                mw = cv.px(360)
                lvl = min(1.0, max(0.0, (eot - 0.5) / (0.998 - 0.5)))
                fired = any(0 <= now - te["t"] < 0.6 for te in ev.turn_ends)
                d.rounded_rectangle([wx, yy + hh / 2 - cv.px(9), wx + mw, yy + hh / 2 + cv.px(9)], radius=cv.px(5), fill=mix(CARD, NAVY, 0.6))
                d.rounded_rectangle([wx, yy + hh / 2 - cv.px(9), wx + mw * lvl, yy + hh / 2 + cv.px(9)], radius=cv.px(5), fill=GOOD if fired else mix(CARD, GOOD, 0.6))
                cv.text(d, (wx + mw + cv.px(14), yy + hh / 2), "turn end" if fired else f"p = {eot:.3f}", "demi" if fired else "medium", 24, GOOD if fired else INK, anchor="lm", bg=CARD)
            else:
                txt = ev.live_text(now)[0]
                f = font("medium", cv.px(24))
                while f.getlength(txt) > (x1 - wx - cv.px(30)) and " " in txt:
                    txt = txt.split(" ", 1)[1]
                cv.text(d, (wx, yy + hh / 2), (txt + " |") if txt else "…", "medium", 24, INK, anchor="lm", bg=CARD)
        y += 4 * (hh + cv.px(10)) + cv.px(10)
        a4 = ease((st - 2.6) / 0.4)
        glow_panel(cv, d, (x0, y, x1, y + cv.px(150)), fill=mix(NAVY, CARD, a4), outline=mix(NAVY, USER, a4), radius=cv.px(16))
        cv.text(d, (x0 + cv.px(16), y + cv.px(10)), "Nemotron-3 diarizer · a lane per voice", "demi", 24, mix(CARD, USER, a4), bg=CARD)
        vis = ev.f_t <= now
        win, lw = 6.0, x1 - x0 - 2 * cv.px(16)
        for k in range(4):
            ly = y + cv.px(46) + k * cv.px(24)
            d.rounded_rectangle([x0 + cv.px(16), ly, x1 - cv.px(16), ly + cv.px(18)], radius=cv.px(4), fill=mix(CARD, NAVY, 0.5 * a4))
            for tt, s_ in zip(ev.f_t[vis], ev.f_spk[vis]):
                if tt < now - win or s_[k] <= 0.5:
                    continue
                x = x0 + cv.px(16) + (tt - (now - win)) / win * lw
                d.rectangle([x - 1, ly + 3, x + max(1, lw / (win / 0.08)), ly + cv.px(15)], fill=mix(CARD, SPK[k], (0.4 + 0.6 * float(s_[k])) * a4))
        y += cv.px(165)
        a5 = ease((st - 3.4) / 0.4)
        glow_panel(cv, d, (x0, y, x1, y + cv.px(110)), fill=mix(NAVY, CARD, a5), outline=mix(NAVY, GOOD, a5), radius=cv.px(16))
        cv.text(d, (x0 + cv.px(16), y + cv.px(10)), "Parakeet-TDT v3 · rewrites each finished turn", "demi", 24, mix(CARD, GOOD, a5), bg=CARD)
        fins = [f for f in ev.finals if f["arr"] <= now and f.get("source") == "tdt_v3" and (f.get("text") or "").strip()]
        if fins:
            f_ = fins[-1]; txt = f_["text"]; nchar = int(len(txt) * min(1.0, (now - f_["arr"]) / 0.6))
            lines = cv.wrap(txt[:nchar], "medium", 24, x1 - x0 - cv.px(32))[:2]
            for j, l in enumerate(lines):
                cv.text(d, (x0 + cv.px(16), y + cv.px(46) + j * cv.px(30)), l, "medium", 24, INK, bg=CARD)
        else:
            cv.text(d, (x0 + cv.px(16), y + cv.px(46)), "waiting for the first turn end …", "medium", 24, mix(CARD, MUTED, a5), bg=CARD)

    # -- voice-print flow
    def shot_vpflow(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio = self.D.audio(sh["clip"])
        ev = self.D.events(sh["events"])
        top = cv.px(190 if not cv.sq else 130)
        # voice print chip slides in
        a1 = ease(st / 0.5)
        px0, px1 = cv.col(0, 3) if not cv.sq else (cv.m, cv.m + cv.px(400))
        chip = (px0 - cv.px(200) * (1 - a1), top, px1 - cv.px(200) * (1 - a1), top + cv.px(120))
        glow_panel(cv, d, chip, fill=CARD, outline=USER, radius=cv.px(20))
        cv.text(d, (chip[0] + cv.px(18), top + cv.px(14)), "5 s voice print", "demi", 28, USER, bg=CARD)
        V3.draw_waveform_bars(cv, d, (chip[0] + cv.px(18), top + cv.px(54), chip[2] - cv.px(18), top + cv.px(106)), audio, 5.0, USER, window=5.0, bars=36)
        # arrow to the TS-VAD head box
        hx0, hx1 = cv.col(4, 4) if not cv.sq else (cv.m + cv.px(470), cv.W - cv.m)
        a2 = ease((st - 0.5) / 0.4)
        d.line([(chip[2], top + cv.px(60)), (hx0 - cv.px(6), top + cv.px(60))], fill=mix(NAVY, USER, a2), width=cv.px(3))
        head = (hx0, top - cv.px(10), hx1, top + cv.px(130))
        glow_panel(cv, d, head, fill=mix(NAVY, CARD, a2), outline=mix(NAVY, OURS, a2))
        cv.text(d, ((hx0 + hx1) / 2, top + cv.px(10)), "target-speaker head", "demi", 30 if not cv.sq else 26, mix(CARD, OURS, a2), anchor="ma", bg=CARD)
        cv.text(d, ((hx0 + hx1) / 2, top + cv.px(52)), "0.26 M parameters · block 4 of the same encoder pass" if not cv.sq else "0.26 M params · block 4", "medium", 24, mix(CARD, MUTED, a2), anchor="ma", bg=CARD)
        cv.text(d, ((hx0 + hx1) / 2, top + cv.px(84)), "P(this is the user) every 80 ms", "medium", 24, mix(CARD, MUTED, a2), anchor="ma", bg=CARD)
        # two lanes: all speech vs the target track
        a3 = ease((st - 1.0) / 0.5)
        ly = top + cv.px(190)
        lx0, lx1 = cv.m, cv.W - cv.m
        dur = audio.dur  # the whole clip on one axis, so the two lanes can be compared at a glance
        vis = ev.f_t <= now
        LH = cv.px(70)
        def X(tt):
            return lx0 + tt / dur * (lx1 - lx0)
        def lane(y, label, col, values, alpha):
            cv.text(d, (lx0, y - cv.px(36)), label, "demi", 28, mix(NAVY, col, alpha))
            d.rounded_rectangle([lx0, y, lx1, y + LH], radius=cv.px(10), fill=mix(NAVY, CARD, alpha))
            for tt, v in zip(ev.f_t[vis], values[vis]):
                if v <= 0.5:
                    continue
                d.rectangle([X(tt) - 1, y + cv.px(10), X(tt) + 2, y + LH - cv.px(10)], fill=mix(CARD, col, (0.4 + 0.6 * float(v)) * alpha))
            xp = X(min(now, dur))
            d.line([(xp, y - cv.px(4)), (xp, y + LH + cv.px(4))], fill=mix(NAVY, INK, alpha), width=2)
        a3b = ease((st - 1.4) / 0.5)
        lane(ly, "all speech in the room (VAD)", MUTED, ev.f_vad, a3)
        target = ev.f_spk[:, 0] if len(ev.f_spk) else np.zeros(0)
        lane(ly + cv.px(200), "the user only (target-speaker track)", OURS, target, a3b)
        # annotate the stretches where someone else talks: speech on the top lane, nothing on the bottom one
        other = (ev.f_vad > 0.5) & (target < 0.5) & vis if len(target) else np.zeros(0, bool)
        spans, start = [], None
        for i, o in enumerate(other):
            if o and start is None:
                start = ev.f_t[i]
            if (not o or i == len(other) - 1) and start is not None:
                if ev.f_t[i] - start >= 0.4:
                    spans.append((start, ev.f_t[i]))
                start = None
        for (s0, s1) in spans:
            xa, xb = X(s0), X(s1)
            d.rounded_rectangle([xa - cv.px(6), ly - cv.px(6), xb + cv.px(6), ly + LH + cv.px(6)], radius=cv.px(8), outline=USER, width=cv.px(3))
            d.rounded_rectangle([xa - cv.px(6), ly + cv.px(200) - cv.px(6), xb + cv.px(6), ly + cv.px(200) + LH + cv.px(6)], radius=cv.px(8), outline=USER, width=cv.px(3))
            anchor = "la" if xa < cv.W / 2 else "ra"
            ax = xa - cv.px(6) if anchor == "la" else xb + cv.px(6)
            cv.text(d, (ax, ly + LH + cv.px(12)), "someone else talking", "demi", 26, USER, anchor=anchor)
            cv.text(d, (ax, ly + cv.px(200) + LH + cv.px(12)), "user lane stays empty", "demi", 26, USER, anchor=anchor)
        # the turn head connected to the target lane only; it fires at the served turn ends
        a4 = ease((st - 1.9) / 0.5)
        ty = ly + cv.px(200) + LH + cv.px(70)
        fired = any(0 <= now - te["arr"] < 1.0 for te in ev.turn_ends)
        glow_panel(cv, d, (cv.W / 2 - cv.px(260), ty, cv.W / 2 + cv.px(260), ty + cv.px(80)), fill=mix(NAVY, CARD if not fired else (30, 90, 70), a4), outline=mix(NAVY, GOOD, a4), radius=cv.px(20))
        d.line([(cv.W / 2, ly + cv.px(200) + LH), (cv.W / 2, ty)], fill=mix(NAVY, GOOD, a4), width=cv.px(3))
        cv.text(d, (cv.W / 2, ty + cv.px(40)), "turn end!" if fired else "turn head reads this lane only", "demi", 30, mix(CARD, GOOD if not fired else INK, a4), anchor="mm", bg=CARD)
        if not cv.sq:
            cv.text(d, (lx1, ly - cv.px(36)), f"t = {now:4.1f} s · whole clip, live values from the served TS-VAD session", "medium", 24, MUTED, anchor="ra")

    # -- graphs
    def shot_graph_wer(self, img, d, sh, st, T):
        cv = self.cv
        N = self.N
        box = (cv.m, cv.px(250 if not cv.sq else 200), cv.W - cv.m, cv.H - cv.px(190 if not cv.sq else 250))
        series = {"ours, streaming": [N("wer/stream_libri"), N("wer/stream_ami")], "Whisper small": [N("wer/whisper_small_libri"), N("wer/whisper_small_ami")],
                  "Parakeet-CTC 0.6B": [N("wer/parakeet_ctc_libri"), N("wer/parakeet_ctc_ami")], "Whisper turbo": [N("wer/whisper_turbo_libri"), N("wer/whisper_turbo_ami")],
                  "TDT v3 per turn (ours)": [N("wer/tdt_v3_libri"), N("wer/tdt_v3_ami")]}
        colors = {"ours, streaming": OURS, "Whisper small": GREY, "Parakeet-CTC 0.6B": (150, 160, 200), "Whisper turbo": (190, 196, 230), "TDT v3 per turn (ours)": GOOD}
        grouped_bars(cv, d, box, ["LibriSpeech, read speech", "AMI meetings"], series, colors, st - 0.3, 24.0, unit=" %", note_better="word error rate · lower is better")
        cv.source(d, self.D.src("wer/stream_ami", "wer/tdt_v3_ami"), y=cv.H - cv.px(100)) if False else None

    def shot_graph_turn(self, img, d, sh, st, T):
        cv = self.cv
        N = self.N
        box = (cv.m, cv.px(250 if not cv.sq else 200), cv.W - cv.m, cv.H - cv.px(190 if not cv.sq else 250))
        series = {"ours (label-free)": [N("turn_miss_all/ours_hybrid"), N("tsvad_turn/icsi/hybrid_sortformer")],
                  "NVIDIA Parakeet-EOU": [N("turn_miss_all/parakeet_eou"), None], "smart-turn + Silero": [N("turn_miss_all/smartturn_silero"), None],
                  "Silero timeout": [N("turn_miss_all/silero_timeout"), N("tsvad_turn/icsi/silero_timeout")], "LiveKit detector": [N("turn_miss_all/livekit_text"), None],
                  "ours + voice print (offline)": [N("tsvad_turn/ami/hybrid_tsvad"), N("tsvad_turn/icsi/hybrid_tsvad")]}
        colors = {"ours (label-free)": OURS, "NVIDIA Parakeet-EOU": GREY, "smart-turn + Silero": (150, 160, 200), "Silero timeout": (170, 176, 210), "LiveKit detector": (190, 196, 230), "ours + voice print (offline)": GOOD}
        grouped_bars(cv, d, box, ["AMI dev", "ICSI held-out"], series, colors, st - 0.3, 100.0, unit=" %", note_better="missed turn ends at 6 s · lower is better")

    def shot_graph_e2e(self, img, d, sh, st, T):
        cv = self.cv
        N = self.N
        lx0, lx1 = (cv.col(0, 6)[0], cv.col(0, 6)[1] - cv.px(30)) if not cv.sq else (cv.m, cv.W - cv.m)
        rx0, rx1 = (cv.col(6, 6)[0] + cv.px(30), cv.col(6, 6)[1]) if not cv.sq else (cv.m, cv.W - cv.m)
        top = cv.px(250 if not cv.sq else 150)
        cv.text(d, (lx0, top), "cut-ins per minute (Pipecat pipeline)", "demi", 30, INK, alpha=ease(st / 0.4))
        y = hbar_ranges(cv, d, (lx0, top + cv.px(50), lx1, 0), [("Pipecat default stack", N("e2e_cut_ins_per_min/pipecat_default"), RED), ("ours", N("e2e_cut_ins_per_min/ours_timeout"), OURS)], st - 0.3, 8.0, "")
        t2 = top if not cv.sq else y + cv.px(4)
        cv.text(d, (rx0, t2), "first partial transcript after speech onset", "demi", 30, INK, alpha=ease((st - 0.4) / 0.4))
        ft_a, ft_b, ft_c = N("e2e_first_text_ms/pipecat_default"), N("e2e_first_text_ms/livekit_default"), N("e2e_first_text_ms/ours_timeout")
        hbar_ranges(cv, d, (rx0, t2 + cv.px(50), rx1, 0), [("LiveKit default", [v / 1000 for v in ft_b], GREY), ("Pipecat default", [v / 1000 for v in ft_a], GREY), ("ours", [v / 1000 for v in ft_c], OURS)], st - 0.7, 12.0, " s")
        cv.para(d, (cv.m, cv.H - cv.px(260 if not cv.sq else 190)), "dead air is similar to the defaults; against LiveKit's default we cut in more on mono mixes, where it stays quiet", "medium", 26, cv.W - 2 * cv.m, MUTED, alpha=ease((st - 1.5) / 0.4))

    def shot_graph_rtf(self, img, d, sh, st, T):
        cv = self.cv
        N = self.N
        lo, hi = N("rtf/server_nemotron_range")
        a = ease(st / 0.8)
        y = cv.px(300 if not cv.sq else 200)
        x0, x1 = cv.m, cv.W - cv.m
        cv.text(d, (x0, y), "server real-time factor on 2 CPU threads (1.0 = the audio's own speed)", "demi", 30, INK, alpha=ease(st / 0.4))
        d.rounded_rectangle([x0, y + cv.px(60), x1, y + cv.px(110)], radius=cv.px(10), fill=CARD2)
        d.rounded_rectangle([x0, y + cv.px(60), x0 + (x1 - x0) * 0.64 * a, y + cv.px(110)], radius=cv.px(10), fill=OURS)
        d.line([(x1, y + cv.px(50)), (x1, y + cv.px(120))], fill=INK, width=3)
        cv.text(d, (x1, y + cv.px(130)), "1.0 = real time", "medium", 26, MUTED, anchor="ra")
        cv.text(d, (x0 + (x1 - x0) * 0.64 * a + cv.px(16), y + cv.px(85)), f"{count(0.64, a):.2f}", "heavy", 56, INK, anchor="lm", alpha=a)
        cv.text(d, (x0, y + cv.px(130)), f"median {lo:.2f} – {hi:.2f} across the live runs, Nemotron-3 diarizer", "medium", 26, MUTED, alpha=a)
        a2 = ease((st - 0.9) / 0.6)
        y2 = y + cv.px(230)
        cv.text(d, (x0, y2), "server compute after the CPU fast paths (perf pass)", "demi", 30, INK, alpha=a2)
        d.rounded_rectangle([x0, y2 + cv.px(60), x1, y2 + cv.px(110)], radius=cv.px(10), fill=CARD2)
        d.rounded_rectangle([x0, y2 + cv.px(60), x0 + (x1 - x0) * 0.62 * a2, y2 + cv.px(110)], radius=cv.px(10), fill=GOOD)
        cv.text(d, (x0 + (x1 - x0) * 0.62 * a2 + cv.px(16), y2 + cv.px(85)), f"{count(0.62, a2):.2f}x", "heavy", 56, INK, anchor="lm", alpha=a2)
        cv.text(d, (x0, y2 + cv.px(130)), "of the compute before the pass · same events out, bit-identical", "medium", 26, MUTED, alpha=a2)

    def shot_end(self, img, d, sh, st, T):
        cv = self.cv
        a = ease(st / 0.5)
        cy = cv.H / 2 - cv.px(80)
        cv.text(d, (cv.W / 2, cy - cv.px(90)), "audioforge", "heavy", 120 if not cv.sq else 88, INK, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cy + cv.px(80)), "open code · open benchmarks · open negative results", "medium", 36 if not cv.sq else 28, MUTED, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cy + cv.px(150)), V2.REPO_URL, "mono", 40 if not cv.sq else 28, OURS, anchor="ma", alpha=a)


# ----------------------------------------------------------------------------- audio (v3 mix, lower music under narration, no pumping: slow release)
def build_audio(r):
    n = int(math.ceil(r.duration * SR)) + SR
    voice = np.zeros(n, np.float32)
    for sh in r.shots:
        u = sh.get("unit")
        if not u:
            continue
        w, sr = sf.read(u["wav"], dtype="float32")
        w = V2.resample(w, sr, SR)
        i = int(u["start"] * SR)
        voice[i: i + len(w)] += w[: n - i]
    if np.any(np.abs(voice) > 0.01):
        rms = np.sqrt(np.mean(voice[np.abs(voice) > 0.01] ** 2))
        voice *= min(6.0, 10 ** (-14 / 20) / max(rms, 1e-6))
        voice = np.tanh(voice * 1.15) / np.tanh(1.15)
    call = np.zeros(n, np.float32)
    for sh in r.shots:
        if "clip" not in sh:
            continue
        c = r.D.audio(sh["clip"])
        w = V2.resample(c.wav, c.sr, SR)
        off = int(sh["offset"] * SR)
        seg = w[off: off + int((sh["end"] - sh["start"]) * SR)].copy()
        f = min(len(seg), int(0.3 * SR))
        if len(seg):
            seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        i = int(sh["start"] * SR)
        call[i: i + len(seg)] += seg[: n - i] * 10 ** (-10 / 20)
    env = np.convolve(np.abs(voice), np.ones(int(0.1 * SR)) / int(0.1 * SR), mode="same")
    duck = 1 - 0.55 * np.clip(env / 0.06, 0, 1)
    d2 = np.empty_like(duck)
    acc, a_att, a_rel = 1.0, math.exp(-1 / (0.05 * SR)), math.exp(-1 / (0.8 * SR))  # slow release: no pumping
    for i in range(len(duck)):
        a = a_att if duck[i] < acc else a_rel
        acc = a * acc + (1 - a) * duck[i]
        d2[i] = acc
    bed = V3.music_bed(len(voice) / SR + 1.0, SR)[: n]
    bed = np.pad(bed, (0, n - len(bed)))
    out = voice + call * d2 + bed * (10 ** (-21 / 20)) * (0.6 + 0.4 * d2)
    out = np.tanh(out * 1.05) / np.tanh(1.05) * 0.95
    return out[: int(r.duration * SR)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1920x1080"); ap.add_argument("--out", default=None)
    ap.add_argument("--storyboard", default=None); ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--crf", type=int, default=20); ap.add_argument("--seg", type=int, default=None); ap.add_argument("--nseg", type=int, default=1)
    ap.add_argument("--mux", action="store_true"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    W, H = (int(v) for v in a.size.lower().split("x"))
    npath = OUT_DIR / "narration_v4.json"
    narr = json.loads(npath.read_text()) if npath.exists() and not a.dry else {"units": dry_units()}
    r = Renderer(W, H, Data(), narr)
    print(f"v4 {W}x{H}: {r.duration:.1f} s, {len(r.shots)} shots")
    for s in r.shots:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  {s['name']}")
    if a.storyboard:
        out = Path(a.storyboard); out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(r.shots):
            t = s["start"] + 0.6 * (s["end"] - s["start"])
            r.frame(t).save(out / f"{i + 1:02d}_{s['name']}_{W}x{H}.png")
        print("storyboard ->", out); return
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
    out.with_suffix(".timeline.json").write_text(json.dumps({"duration_s": r.duration, "shots": [
        {"name": s["name"], "start": round(s["start"], 2), "end": round(s["end"], 2), "clip": s.get("clip")} for s in r.shots]}, indent=1))


if __name__ == "__main__":
    main()
