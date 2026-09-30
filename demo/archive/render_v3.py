"""Showcase v3: the product in use, filmed like a product spot. A minimal call UI (user bubble with a live waveform and
words, agent reply bubble), two of them side by side on the same call audio (framework default on the left, ours on
the right), replaying the recorded E2E sessions. Almost no words on screen: at most 8 short phrases, one at a time,
with the test basis as fine print. Music bed generated in numpy (public domain by construction), narration one or
two plain sentences per shot, cuts on the beat (100 BPM, shots are whole bars).

    PYTHONPATH=. .venv/bin/python demo/archive/render_v3.py --storyboard $DEMO_OUT/storyboard_v3 [--size 1080x1080]
    PYTHONPATH=. .venv/bin/python demo/archive/render_v3.py --size 1920x1080 --out $DEMO_OUT/showcase_v3.mp4 [--seg i --nseg n | --mux]

Reuses the data layer of render_v2 (E2E per-clip records, T records, recorded server events, clip audio).
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
from render_v2 import Data, ease, mix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
OUT_DIR = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
SCRATCH = OUT_DIR / "scratch"
FPS = 30
SR = 48000
BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT  # 2.4 s

# ----------------------------------------------------------------------------- palette (dark slate, one accent)
BG = (15, 20, 28)
BG2 = (22, 29, 40)
CARD = (28, 36, 50)
BUBBLE = (44, 54, 72)
BUBBLE_INK = None            # colour of the bars inside the user bubble (None = INK)
INK = (245, 247, 250)          # 16.9:1 on BG
MUTED = (170, 180, 196)        # 8.4:1 on BG
DIM = (96, 108, 128)
ACCENT = (70, 150, 255)        # ours (bubble fill; white text on it 4.6:1 for large text)
ACCENT_SOFT = (48, 96, 160)
GREY_AGENT = (92, 102, 120)    # default stack's agent bubble
RED = (235, 90, 80)
SPK = [(70, 150, 255), (245, 160, 90), (190, 140, 255), (110, 210, 160)]


def lum(c):
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


assert contrast(INK, BG) >= 7 and contrast(MUTED, BG) >= 7 and contrast(INK, CARD) >= 7 and contrast(INK, BUBBLE) >= 7


# ----------------------------------------------------------------------------- shots
# Each shot: name, bars (x 2.4 s), narration, on-screen phrase (or None), fine print (or None), scene spec.
SHOTS = [
    dict(name="title", bars=2, narr="Today we are presenting audioforge.", phrase=None, fine=None,
         clip="ami_ES2011b_027", events="ami_ES2011b_027.mono", offset=0.0, ui="single_ours"),
    dict(name="two_systems", bars=3, narr="Same call, two systems. The left one is the default stack. The right one is ours.",
         phrase=None, fine=None, clip="ami_IB4002_119", offset=0.0, ui="split", left=("pipecat", "A"), right=("pipecat", "C")),
    dict(name="interrupt", bars=3, narr="Watch the left: it answers while she is still talking. Ours waits.",
         phrase="Fewer interruptions", fine="1.6–5.5x fewer cut-ins than the Pipecat default stack across 37 recorded conversations · runs/e2e_final.json",
         clip="ami_IB4002_119", offset=7.2, ui="split", left=("pipecat", "A"), right=("pipecat", "C")),
    dict(name="deadair", bars=3, narr="She stops. The left takes three seconds to reply. Ours takes one and a half.",
         phrase=None, fine="median dead air after the user stops: similar to the defaults across 37 conversations (lower on three clip sets, higher on two) · runs/e2e_final.json",
         clip="ami_IS1008b_003", offset=10.5, ui="split", left=("pipecat", "A"), right=("pipecat", "C"), deadair=True),
    dict(name="words", bars=3, narr="Here the words show up while she speaks. The left side has nothing yet.",
         phrase="Words while you speak", fine="first partial transcript 0.8–1.2 s after speech onset, vs 2.6–3.7 s (Pipecat default) and 3.4–11.3 s (LiveKit default) · runs/e2e_final.json",
         clip="ami_ES2011b_027", offset=0.0, ui="split", left=("livekit", "B"), right=("livekit", "C")),
    dict(name="room", bars=3, narr="A room with several people. Ours knows which one is the user, and answers only them.",
         phrase="Knows whose turn it is", fine="AMI meetings, 974 turn ends: 61.9 % missed vs 70–74 % for smart-turn, LiveKit and NVIDIA EOU, all at ≤ 5 % false cutoffs · runs/baselines_turn.json",
         clip="ami_IS1008b_003", events="ami_IS1008b_003_agentend", offset=0.0, ui="room"),
    dict(name="onemodel", bars=2, narr="All of this is one model, on two laptop CPU threads.",
         phrase="One model. Two CPU threads.", fine="whole server at real-time factor 0.64 with the Nemotron-3 diarizer, 2 threads, Apple M5 · runs/e2e_final.json",
         ui="fan"),
    dict(name="voiceprint", bars=3, narr="Give it five seconds of your voice, and it misses far fewer of your turn ends.",
         phrase="Give it a voice print", fine="missed turn ends 61.9 → 39.3 % on AMI, 68.5 → 20.4 % on held-out ICSI, offline benchmark; this clip is one live session · runs/improve_115m.json",
         clip="ami_TS3004b_064", offset=10.0, ui="split", left=("pipecat", "C"), right=("pipecat", "T"), tags=("no voice print", "voice print")),
    dict(name="honest", bars=2, narr="It is not perfect. Sometimes it still interrupts, and it is slower than the best two-party detector.",
         phrase=None, fine="ours: 1.1–4.5 cut-ins per minute vs 2.1–7.5 (Pipecat default); TurnBench two-party calls: 0.7 s behind VAP · runs/e2e_final.json, runs/dyadic_train.json",
         clip="ami_IS1008b_003", offset=1.5, ui="single_ours_lk"),
    dict(name="end", bars=2, narr="It is open, and it runs on a laptop.", phrase="audioforge",
         fine="baseline transcripts are the reference text shown at the measured arrival time (session text not retained); all timings from the recorded sessions", ui="end"),
]
LEAD = 0.4  # narration starts this long after the cut


def build_timeline(narr_units=None):
    """Shots on the bar grid; a shot grows to the next bar if its narration does not fit."""
    t = 0.0
    shots = []
    for i, sh in enumerate(SHOTS):
        s = dict(sh)
        dur = s["bars"] * BAR
        unit = narr_units[i] if narr_units and i < len(narr_units) else None
        if unit is not None:
            need = LEAD + (unit["end"] - unit["start"]) + 0.5
            while dur < need:
                dur += BAR
        s["start"], s["end"] = t, t + dur
        if unit is not None:
            s["unit"] = dict(unit, start=t + LEAD, end=t + LEAD + (unit["end"] - unit["start"]))
        t += dur
        shots.append(s)
    return shots


# ----------------------------------------------------------------------------- drawing
class Canvas(V2.Canvas):
    def __init__(self, W, H):
        super().__init__(W, H)
        self.cap_h = 0
        self.m = self.px(96 if not self.sq else 56)

    def text(self, d, xy, s, kind="medium", size=40, fill=INK, anchor="la", alpha=1.0, bg=None):
        super().text(d, xy, s, kind, size, fill, anchor, alpha, bg or BG)


def vignette(cv):
    img = Image.new("RGB", (cv.W, cv.H), BG)
    ov = Image.new("L", (cv.W // 8, cv.H // 8), 0)
    od = ImageDraw.Draw(ov)
    od.ellipse([-cv.W // 16, -cv.H // 16, cv.W // 8 + cv.W // 16, cv.H // 8 + cv.H // 16], fill=255)
    ov = ov.filter(ImageFilter.GaussianBlur(cv.W // 40)).resize((cv.W, cv.H))
    light = Image.new("RGB", (cv.W, cv.H), BG2)
    return Image.composite(light, img, ov)


def draw_waveform_bars(cv, d, box, audio, now, col, *, window=2.4, bars=28, active=True):
    """A chat-style live waveform: the last ``window`` seconds as thin rounded bars."""
    x0, y0, x1, y1 = box
    n = bars
    env = audio.env_at(now - window, now, n)
    if not active:
        env = env * 0.25
    bw = (x1 - x0) / n
    mid = (y0 + y1) / 2
    for i in range(n):
        h = max(cv.px(4), env[i] * (y1 - y0) * 0.9)
        x = x0 + i * bw + bw * 0.25
        d.rounded_rectangle([x, mid - h / 2, x + bw * 0.5, mid + h / 2], radius=bw * 0.25, fill=col)


def draw_app(cv, d, box, tag, sess, audio, events, now, *, ours, agent_col=None, show_text=True, room=False, alpha=1.0, deadair=False):
    """A minimal call window: header with a tag, the user's bubble (live waveform + words), the agent's reply bubbles."""
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = cv.px(28)
    d.rounded_rectangle(box, radius=cv.px(28), fill=CARD)
    # header
    hy = y0 + pad
    dot = ACCENT if ours else GREY_AGENT
    d.ellipse([x0 + pad, hy + cv.px(4), x0 + pad + cv.px(18), hy + cv.px(22)], fill=dot)
    cv.text(d, (x0 + pad + cv.px(30), hy), tag, "demi", 26, MUTED, bg=CARD)
    cv.text(d, (x1 - pad, hy), f"{now:4.1f} s", "mono", 24, DIM, anchor="ra", bg=CARD)
    W = x1 - x0
    # speaking state from the session's user turns (baseline side) or the served VAD (ours)
    speaking = any(a <= now <= b for a, b in sess["user_turns"])
    ended = bool(sess["user_turns"]) and now > sess["user_turns"][0][1]
    # user bubble (left)
    by = hy + cv.px(60 if not cv.sq else 50)
    bw = int(W * 0.72)
    bh_ = cv.px(110 if not cv.sq else 80)
    bub = (x0 + pad, by, x0 + pad + bw, by + bh_)
    d.rounded_rectangle(bub, radius=cv.px(22), fill=BUBBLE)
    d.ellipse([x0 + pad - cv.px(6), by + bh_ - cv.px(30), x0 + pad + cv.px(26), by + bh_ + cv.px(2)], fill=BUBBLE)
    bink = BUBBLE_INK or INK
    draw_waveform_bars(cv, d, (bub[0] + cv.px(24), bub[1] + cv.px(22), bub[2] - cv.px(24), bub[3] - cv.px(22)), audio, now,
                       bink if speaking else mix(BUBBLE, bink, 0.45), active=speaking)
    ty = bub[3] + cv.px(18)
    # words under the user bubble
    if show_text:
        ft = sess.get("first_text")
        if ours and events.partials:
            full, fin, part = events.live_text(now)
            state = "text" if full else ("dots" if speaking or ended else "none")
        else:
            full = sess["text"]
            state = "text" if (ft is not None and now >= ft) else ("dots" if speaking or ended else "none")
            if state == "text" and not ours:
                words = full.split()
                full = " ".join(words[: max(1, int(len(words) * min(1.0, (now - ft) / 0.5)))])
        maxw = bw
        if state == "text":
            keep = 2 if not cv.sq else 1
            allines = cv.wrap(full, "medium", 30, maxw)
            lines = allines[-keep:]
            if len(allines) > keep:
                lines[0] = "… " + lines[0]
            for line in lines:
                cv.text(d, (x0 + pad, ty), line, "medium", 30, INK, bg=CARD); ty += cv.px(38)
        elif state == "dots":
            ph = (now * 2.5) % 3
            for k in range(3):
                cx = x0 + pad + cv.px(10) + k * cv.px(26)
                r = cv.px(7) if k <= ph else cv.px(5)
                d.ellipse([cx - r, ty + cv.px(16) - r, cx + r, ty + cv.px(16) + r], fill=MUTED if k <= ph else DIM)
            ty += cv.px(38)
    # agent bubbles (right): one per response / cut-in that has happened
    ay = max(ty + cv.px(24), by + cv.px(190 if not cv.sq else 130))
    acol = agent_col or (ACCENT if ours else GREY_AGENT)
    fired = sorted([(t, True) for t in sess["cut_ins"] if t <= now] + [(e["response"], False) for e in sess["ends"] if e.get("response") is not None and e["response"] <= now])
    room_h = y1 - pad - ay
    per = cv.px(64) + cv.px(18)
    fired = fired[-max(1, int(room_h // per)):] if room_h >= cv.px(64) else []
    for t, is_cut in fired:
        a = ease((now - t) / 0.3)
        w = int(W * 0.46 * a) + cv.px(10)
        h = cv.px(64)
        rb = (x1 - pad - w, ay, x1 - pad, ay + h)
        d.rounded_rectangle(rb, radius=cv.px(20), fill=mix(CARD, acol, a))
        d.ellipse([x1 - pad - cv.px(26), ay + h - cv.px(30), x1 - pad + cv.px(6), ay + h + cv.px(2)], fill=mix(CARD, acol, a))
        # reply "voice" bars inside the bubble
        k = 9
        for i in range(k):
            ph = 0.5 + 0.5 * math.sin((now - t) * 9 + i)
            bh = cv.px(10) + ph * cv.px(28) * a
            xx = rb[0] + cv.px(22) + i * cv.px(16)
            if xx + cv.px(8) < rb[2] - cv.px(16):
                d.rounded_rectangle([xx, ay + h / 2 - bh / 2, xx + cv.px(8), ay + h / 2 + bh / 2], radius=cv.px(4), fill=mix(acol, INK, 0.9 * a))
        if is_cut:
            d.rounded_rectangle([rb[0] - cv.px(14), ay + cv.px(8), rb[0] - cv.px(8), ay + h - cv.px(8)], radius=cv.px(3), fill=mix(CARD, RED, a))
        elif deadair and sess["ends"]:
            end = sess["ends"][0]["end"]
            cv.text(d, (rb[0] - cv.px(20), ay + h / 2), f"{t - end:.1f} s", "demi", 34, mix(CARD, INK, a), anchor="rm", bg=CARD)
        ay += h + cv.px(18)
    return ay


def draw_room(cv, d, box, audio, events, now):
    """The meeting variant of the app: several people in the header; the user's turns bound to one of them."""
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = cv.px(28)
    d.rounded_rectangle(box, radius=cv.px(28), fill=CARD)
    hy = y0 + pad
    d.ellipse([x0 + pad, hy + cv.px(4), x0 + pad + cv.px(18), hy + cv.px(22)], fill=ACCENT)
    cv.text(d, (x0 + pad + cv.px(30), hy), "audioforge · meeting room", "demi", 26, MUTED, bg=CARD)
    cv.text(d, (x1 - pad, hy), f"{now:4.1f} s", "mono", 24, DIM, anchor="ra", bg=CARD)
    # current frame state
    n = int(np.searchsorted(events.f_t, now, side="right"))
    spk = events.f_spk[n - 1] if n else np.zeros(4)
    bound = next((e["column"] for e in events.enrolled if e["arr"] <= now), None)
    # people: the meeting's four participants (AMI has four), speaking rings on whoever is active
    py = hy + cv.px(70)
    names = ["person A", "person B", "person C", "person D"]
    for k in range(4):
        cx = x0 + pad + cv.px(70) + k * cv.px(170 if not cv.sq else 160)
        r = cv.px(34)
        active = spk[k] > 0.5
        if active:
            d.ellipse([cx - r - cv.px(8), py - cv.px(8), cx + r + cv.px(8), py + 2 * r + cv.px(8)], outline=SPK[k], width=cv.px(4))
        d.ellipse([cx - r, py, cx + r, py + 2 * r], fill=mix(CARD, SPK[k], 0.35 if not active else 0.9))
        cv.text(d, (cx, py + 2 * r + cv.px(10)), names[k], "medium", 24, MUTED, anchor="ma", bg=CARD)
        if bound == k:
            cv.pill(d, (cx - cv.px(34), py + 2 * r + cv.px(44)), "user", ACCENT, size=22)
    # user bubble: waveform lit only while the bound speaker talks
    by = py + cv.px(190)
    W = x1 - x0
    bw = int(W * 0.72)
    bub = (x0 + pad, by, x0 + pad + bw, by + cv.px(110))
    d.rounded_rectangle(bub, radius=cv.px(22), fill=BUBBLE)
    speaking = bound is not None and spk[bound] > 0.5
    bink = BUBBLE_INK or INK
    draw_waveform_bars(cv, d, (bub[0] + cv.px(24), bub[1] + cv.px(22), bub[2] - cv.px(24), bub[3] - cv.px(22)), audio, now, bink if speaking else mix(BUBBLE, bink, 0.45), active=speaking)
    ty = bub[3] + cv.px(18)
    full = events.live_text(now)[0]
    if full:
        lines = cv.wrap(full, "medium", 30, bw)
        if len(lines) > 2:
            lines = ["… " + lines[-2], lines[-1]]
        for line in lines:
            cv.text(d, (x0 + pad, ty), line, "medium", 30, INK, bg=CARD); ty += cv.px(38)
    # agent reply at the served turn end
    ay = max(ty + cv.px(24), by + cv.px(190))
    for te in events.turn_ends:
        if te["arr"] <= now:
            a = ease((now - te["arr"]) / 0.3)
            w = int(W * 0.46 * a) + cv.px(10)
            h = cv.px(64)
            rb = (x1 - pad - w, ay, x1 - pad, ay + h)
            d.rounded_rectangle(rb, radius=cv.px(20), fill=mix(CARD, ACCENT, a))
            for i in range(9):
                ph = 0.5 + 0.5 * math.sin((now - te["arr"]) * 9 + i)
                bh = cv.px(10) + ph * cv.px(28) * a
                xx = rb[0] + cv.px(22) + i * cv.px(16)
                if xx + cv.px(8) < rb[2] - cv.px(16):
                    d.rounded_rectangle([xx, ay + h / 2 - bh / 2, xx + cv.px(8), ay + h / 2 + bh / 2], radius=cv.px(4), fill=mix(ACCENT, INK, 0.9 * a))
            ay += h + cv.px(18)


class Renderer:
    def __init__(self, W, H, data: Data, narr=None):
        self.cv, self.D = Canvas(W, H), data
        units = narr["units"] if narr else None
        self.shots = build_timeline(units)
        self.duration = self.shots[-1]["end"]

    def frame(self, T):
        cv = self.cv
        img = vignette(cv)
        d = ImageDraw.Draw(img)
        sh = next((s for s in self.shots if s["start"] <= T < s["end"]), self.shots[-1])
        st = T - sh["start"]
        getattr(self, "shot_" + sh["ui"])(img, d, sh, st, T)
        self.overlay(d, sh, st, T)
        return img

    # -- shared overlay: phrase, fine print, narration caption (small, since the narrator carries the explanation)
    def overlay(self, d, sh, st, T):
        cv = self.cv
        a_in = ease((st - 0.6) / 0.4)
        a_out = 1 - ease((st - (sh["end"] - sh["start"]) + 0.5) / 0.4)
        a = min(a_in, a_out)
        if sh.get("phrase") and sh["ui"] != "end":
            size = 72 if not cv.sq else 56
            cv.text(d, (cv.W / 2, cv.px(84 if not cv.sq else 60)), sh["phrase"], "demi", size, INK, anchor="ma", alpha=a)
        if sh.get("fine"):
            lines = cv.wrap(sh["fine"], "medium", 22, cv.W - 2 * cv.m)
            y = cv.H - cv.px(40) - cv.px(28) * len(lines)
            for line in lines:
                cv.text(d, (cv.W / 2, y), line, "medium", 22, MUTED, anchor="ma", alpha=a)
                y += cv.px(28)
        # narration caption (LinkedIn autoplays muted): a single line, small, above the fine print
        u = sh.get("unit")
        if u and u["start"] <= T <= u["end"] + 0.4:
            parts = V2.split_caption(u["text"], 64 if not cv.sq else 42)
            total = sum(len(p) for p in parts)
            t0 = u["start"]
            for p in parts:
                dd = (u["end"] - u["start"]) * len(p) / total
                if t0 <= T < t0 + dd + 0.4:
                    txt = p; break
                t0 += dd
            else:
                txt = parts[-1]
            fy = cv.H - cv.px(40) - cv.px(28) * (len(cv.wrap(sh["fine"], "medium", 22, cv.W - 2 * cv.m)) if sh.get("fine") else 0) - cv.px(56)
            w = cv.tw(txt, "medium", 32 if not cv.sq else 28) + cv.px(40)
            d.rounded_rectangle([cv.W / 2 - w / 2, fy - cv.px(8), cv.W / 2 + w / 2, fy + cv.px(44)], radius=cv.px(12), fill=(0, 0, 0))
            cv.text(d, (cv.W / 2, fy), txt, "medium", 32 if not cv.sq else 28, INK, anchor="ma", bg=(0, 0, 0))

    # -- layouts
    def _boxes(self, split, tall=False):
        cv = self.cv
        if not cv.sq:
            h = cv.px(600 if tall else 520)
            top = cv.px(210)
            bottom = top + h
        else:
            top = cv.px(150)
            bottom = cv.H - cv.px(250)
        if not split:
            w = cv.px(760 if not cv.sq else 700)
            return [(cv.W / 2 - w / 2, top, cv.W / 2 + w / 2, bottom)]
        if cv.sq:
            h = (bottom - top - cv.px(20)) / 2
            return [(cv.m, top, cv.W - cv.m, top + h), (cv.m, top + h + cv.px(20), cv.W - cv.m, bottom)]
        gap = cv.px(40)
        w = (cv.W - 2 * cv.m - gap) / 2
        return [(cv.m, top, cv.m + w, bottom), (cv.m + w + gap, top, cv.W - cv.m, bottom)]

    def shot_single_ours(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh.get("events", sh["clip"]))
        sess = self.D.session(sh["clip"], "livekit", "C")
        box = self._boxes(False)[0]
        draw_app(cv, d, box, "audioforge", sess, audio, ev, now, ours=True)
        a = ease((st - 0.3) / 0.5)
        cv.text(d, (cv.W / 2, cv.px(84 if not cv.sq else 60)), "audioforge", "demi", 72 if not cv.sq else 56, INK, anchor="ma", alpha=a)

    def shot_single_ours_lk(self, img, d, sh, st, T):  # the honest one: ours in LiveKit cutting in once
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh["clip"])
        sess = self.D.session(sh["clip"], "livekit", "C")
        draw_app(cv, d, self._boxes(False)[0], "audioforge", sess, audio, ev, now, ours=True)

    def shot_split(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh.get("events", sh["clip"]))
        L = self.D.session(sh["clip"], *sh["left"]); R = self.D.session(sh["clip"], *sh["right"])
        tags = sh.get("tags", ("default", "audioforge"))
        b = self._boxes(True)
        draw_app(cv, d, b[0], tags[0], L, audio, ev, now, ours=L["ours"] and sh["left"][1] != "C" or sh["left"][1] == "C" and sh.get("tags") is not None,
                 agent_col=(ACCENT_SOFT if sh.get("tags") else None), deadair=sh.get("deadair", False))
        draw_app(cv, d, b[1], tags[1], R, audio, ev, now, ours=True, deadair=sh.get("deadair", False))

    def shot_room(self, img, d, sh, st, T):
        cv = self.cv
        now = sh["offset"] + st
        audio, ev = self.D.audio(sh["clip"]), self.D.events(sh["events"])
        draw_room(cv, d, self._boxes(False, tall=True)[0], audio, ev, now)

    def shot_fan(self, img, d, sh, st, T):
        cv = self.cv
        a = ease(st / 0.5)
        cx, cy = cv.W / 2, cv.H / 2 + cv.px(20)
        bw, bh = cv.px(260), cv.px(150)
        d.rounded_rectangle([cx - cv.px(420) - bw / 2, cy - bh / 2, cx - cv.px(420) + bw / 2, cy + bh / 2], radius=cv.px(24), fill=mix(BG, CARD, a), outline=mix(BG, ACCENT, a), width=cv.px(3))
        cv.text(d, (cx - cv.px(420), cy), "one model", "demi", 36, mix(CARD, INK, a), anchor="mm", bg=CARD)
        labels = ["words", "speech", "speaker", "turn"]
        for i, lab in enumerate(labels):
            ai = ease((st - 0.3 - 0.2 * i) / 0.4)
            ty = cy - cv.px(210) + i * cv.px(140)
            p0 = (cx - cv.px(420) + bw / 2, cy)
            p1 = (cx + cv.px(240), ty)
            # thin curved line as a polyline
            pts = []
            for k in range(41):
                u = k / 40
                x = p0[0] + (p1[0] - p0[0]) * u
                y = p0[1] + (p1[1] - p0[1]) * (3 * u * u - 2 * u * u * u)
                pts.append((x, y))
            npts = max(2, int(len(pts) * ai))
            if ai > 0:
                d.line(pts[:npts], fill=mix(BG, ACCENT, 0.9), width=cv.px(3))
            cv.text(d, (cx + cv.px(270), ty), lab, "medium", 40, INK, anchor="lm", alpha=ai)

    def shot_end(self, img, d, sh, st, T):
        cv = self.cv
        a = ease(st / 0.5)
        cy = cv.H / 2 - cv.px(60)
        cv.text(d, (cv.W / 2, cy - cv.px(80)), "audioforge", "heavy", 120 if not cv.sq else 88, INK, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cy + cv.px(90)), "open code, open benchmarks, open negative results", "medium", 36 if not cv.sq else 28, MUTED, anchor="ma", alpha=a)
        cv.text(d, (cv.W / 2, cy + cv.px(160)), V2.REPO_URL, "mono", 40 if not cv.sq else 28, ACCENT, anchor="ma", alpha=a)


# ----------------------------------------------------------------------------- audio: music bed + narration + call audio
def music_bed(dur, sr, seed=3):
    """Generated: a slow pad (two alternating chords, detuned sines, low-passed) and a soft pulse on every beat.
    Public domain by construction (no samples)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    chords = [[110.0, 164.81, 220.0, 277.18], [98.0, 146.83, 196.0, 246.94]]
    pad = np.zeros(n)
    for ci, ch in enumerate(chords):
        w = 0.5 * (1 + np.sin(2 * np.pi * t / (8 * BAR) - np.pi / 2 + np.pi * ci)) ** 2
        for f in ch:
            det = 1 + rng.normal(0, 0.002)
            lfo = 0.7 + 0.3 * np.sin(2 * np.pi * t * rng.uniform(0.05, 0.1) + rng.uniform(0, 6.28))
            pad += w * lfo * np.sin(2 * np.pi * f * det * t + rng.uniform(0, 6.28)) / len(ch)
    # one-pole low-pass x2 (vectorised via lfilter-like recursion in numpy is slow; use FFT low-pass instead)
    spec = np.fft.rfft(pad)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec *= 1 / (1 + (freqs / 700.0) ** 4)
    pad = np.fft.irfft(spec, n)
    pad /= max(1e-6, np.abs(pad).max())
    # pulse on each beat: a short sine burst at 55 Hz with a soft click, accent on the bar
    pulse = np.zeros(n)
    k = int(0.18 * sr)
    env = np.exp(-np.arange(k) / (0.05 * sr))
    burst = env * np.sin(2 * np.pi * 55 * np.arange(k) / sr)
    b = 0.0
    i = 0
    while b < dur:
        s0 = int(b * sr)
        g = 1.0 if i % 4 == 0 else 0.55
        pulse[s0: s0 + k] += g * burst[: max(0, min(k, n - s0))]
        b += BEAT; i += 1
    out = 0.8 * pad + 0.5 * pulse
    fade = int(sr * 2.0)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return (out / max(1e-6, np.abs(out).max())).astype(np.float32)


def build_audio(r: Renderer):
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
        seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        i = int(sh["start"] * SR)
        call[i: i + len(seg)] += seg[: n - i] * 10 ** (-9 / 20)
    env = np.convolve(np.abs(voice), np.ones(int(0.05 * SR)) / int(0.05 * SR), mode="same")
    duck = 1 - 0.6 * np.clip(env / 0.08, 0, 1)
    # smooth (attack 30 ms, release 400 ms) via exponential filters
    d2 = np.empty_like(duck)
    acc, a_att, a_rel = 1.0, math.exp(-1 / (0.03 * SR)), math.exp(-1 / (0.4 * SR))
    for i in range(len(duck)):
        a = a_att if duck[i] < acc else a_rel
        acc = a * acc + (1 - a) * duck[i]
        d2[i] = acc
    bed = music_bed(len(voice) / SR + 1.0, SR)[: n]
    bed = np.pad(bed, (0, n - len(bed)))
    out = voice + call * d2 + bed * (10 ** (-22 / 20)) * (0.6 + 0.4 * d2)
    out = np.tanh(out * 1.05) / np.tanh(1.05) * 0.95
    return out[: int(r.duration * SR)]


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", default=None)
    ap.add_argument("--storyboard", default=None)
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--seg", type=int, default=None)
    ap.add_argument("--nseg", type=int, default=1)
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    W, H = (int(v) for v in a.size.lower().split("x"))
    npath = OUT_DIR / "narration_v3.json"
    narr = json.loads(npath.read_text()) if npath.exists() and not a.dry else None
    r = Renderer(W, H, Data(), narr)
    print(f"v3 {W}x{H}: {r.duration:.1f} s, {len(r.shots)} shots")
    for s in r.shots:
        print(f"  {s['start']:6.1f}-{s['end']:6.1f}  {s['name']}")
    if a.storyboard:
        out = Path(a.storyboard); out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(r.shots):
            t = s["start"] + 0.62 * (s["end"] - s["start"])
            r.frame(t).save(out / f"{i + 1:02d}_{s['name']}_{W}x{H}.png")
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
    out.with_suffix(".timeline.json").write_text(json.dumps({"duration_s": r.duration, "shots": [
        {"name": s["name"], "start": round(s["start"], 2), "end": round(s["end"], 2), "clip": s.get("clip")} for s in r.shots]}, indent=1))


if __name__ == "__main__":
    main()
