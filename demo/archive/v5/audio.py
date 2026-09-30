"""v5 audio: narration units placed per shot, the example recordings, the demo agent's voice and a generated music bed
(public domain by construction: detuned sines and a soft 55 Hz pulse on every beat, no samples).

Mix rules (checked by qa_audio):
  * loudness: every narration line is normalised to -16 LUFS integrated; every example clip plays at -16 LUFS (the clips
    in $DEMO_OUT/v5_data/clips_manifest.json are processed to -16 LUFS / -1 dBTP by demo/archive/v5/audio/clips.py; any other
    clip audio is measured and brought to -16 LUFS here); the agent voice sits 3 dB below its clip (-19 LUFS).
  * narration is silent while an example plays (the timeline puts clips between lines; qa_audio verifies the stems).
  * music: one constant bed level (-34 LUFS, no ducking between narration lines, so no swells); muted across each
    example block (first clip start to last clip / agent end of the shot, narration inside the block included), with
    0.4 s raised-cosine ramps outside the block.
  * edges: every clip / agent / narration edge has a fade (clips 150 ms, baked into the manifest WAVs); the whole film
    fades out over its last 1.5 s and the last 0.1 s is digital silence.
  * a single lookahead peak limiter (-1.5 dBFS sample peak ~ -1 dBTP) on the sum; the same gain curve is applied to every
    stem, so the stems still add up to the mix.
Only numpy + soundfile + the ffmpeg binary (the video venv has no scipy).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DEMO_OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
MANIFEST = DEMO_OUT / "v5_data" / "clips_manifest.json"
BPM = 100.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
TARGET_LUFS = -16.0
AGENT_REL_DB = -3.0
MUSIC_LUFS = -34.0
CLIP_FADE = 0.15
END_FADE = 1.5
END_SILENCE = 0.1
LIMIT_DBFS = -1.5
MUSIC_RAMP = 0.4


# ------------------------------------------------------------------ basics
def resample(x, sr, out_sr):
    if sr == out_sr:
        return x.astype(np.float32)
    n = int(round(len(x) * out_sr / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def db(v):
    return 10 ** (v / 20)


def fade(x, sr, ms_in=20, ms_out=20):
    """Raised-cosine fade in / out (in place)."""
    a, b = min(len(x), int(sr * ms_in / 1000)), min(len(x), int(sr * ms_out / 1000))
    if a:
        x[:a] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))).astype(x.dtype)
    if b:
        x[-b:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, b))).astype(x.dtype)
    return x


def read_mono(path, sr_out):
    w, sr = sf.read(path, dtype="float32")
    w = w.mean(1) if w.ndim > 1 else w
    return resample(w, sr, sr_out)


_LCACHE: dict = {}


def loudness(x, sr):
    """(integrated LUFS, true peak dBTP) of a mono float signal, by ffmpeg's ebur128 (BS.1770-4)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    key = hashlib.sha1(x.tobytes()).hexdigest() + str(sr)
    if key in _LCACHE:
        return _LCACHE[key]
    if len(x) < int(0.45 * sr) or not np.any(x):
        return (-70.0, -120.0)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                        "-af", "ebur128=peak=true", "-f", "null", "-"], input=x.tobytes(), capture_output=True)
    err = p.stderr.decode(errors="replace")
    s = err[err.rindex("Summary:"):] if "Summary:" in err else err
    m = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", s)
    t = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", s)
    val = (float(m.group(1)) if m and m.group(1) != "-inf" else -70.0, float(t.group(1)) if t and t.group(1) != "-inf" else -120.0)
    _LCACHE[key] = val
    return val


def to_lufs(x, sr, target, max_gain_db=30.0):
    L, _ = loudness(x, sr)
    if L <= -69:
        return x
    return (x * db(float(np.clip(target - L, -max_gain_db, max_gain_db)))).astype(np.float32)


def limiter(x, sr, thr_db=LIMIT_DBFS, look_ms=5.0, release_ms=120.0):
    """Lookahead peak limiter: gain per 1 ms block = min(1, thr / peak over the lookahead window), instant attack,
    exponential release; returns the per-sample gain curve (apply it to the mix and to every stem)."""
    thr = db(thr_db)
    hop = max(1, int(sr / 1000))
    nb = int(math.ceil(len(x) / hop))
    pad = np.zeros(nb * hop, np.float32)
    pad[: len(x)] = np.abs(x)
    pk = pad.reshape(nb, hop).max(1)
    k = int(look_ms)
    pk2 = pk.copy()
    for s in range(1, k + 1):          # max over [i - k, i + k] blocks
        pk2[:-s] = np.maximum(pk2[:-s], pk[s:])
        pk2[s:] = np.maximum(pk2[s:], pk[:-s])
    g = np.minimum(1.0, thr / np.maximum(pk2, 1e-9))
    if g.min() >= 1.0:
        return np.ones(len(x), np.float32)
    rel = math.exp(-1.0 / release_ms)
    out = np.empty(nb)
    acc = 1.0
    for i in range(nb):
        acc = g[i] if g[i] < acc else rel * acc + (1 - rel) * g[i]
        out[i] = acc
    return np.interp(np.arange(len(x)), np.arange(nb) * hop + hop / 2, out).astype(np.float32)


# ------------------------------------------------------------------ music
def music_bed(dur, sr, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    chords = [[110.0, 164.81, 220.0, 277.18], [98.0, 146.83, 196.0, 246.94]]
    pad = np.zeros(n)
    for ci, ch in enumerate(chords):
        # equal-power crossfade between the two chords: the sum stays level (no 8-bar swell)
        w = np.sqrt(0.5 * (1 + np.sin(2 * np.pi * t / (8 * BAR) - np.pi / 2 + np.pi * ci)))
        for f in ch:
            det = 1 + rng.normal(0, 0.002)
            lfo = 0.85 + 0.15 * np.sin(2 * np.pi * t * rng.uniform(0.05, 0.1) + rng.uniform(0, 6.28))
            pad += w * lfo * np.sin(2 * np.pi * f * det * t + rng.uniform(0, 6.28)) / len(ch)
    spec = np.fft.rfft(pad)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec *= 1 / (1 + (freqs / 700.0) ** 4)
    pad = np.fft.irfft(spec, n)
    pad /= max(1e-6, np.abs(pad).max())
    pulse = np.zeros(n)
    k = int(0.18 * sr)
    burst = np.exp(-np.arange(k) / (0.05 * sr)) * np.sin(2 * np.pi * 55 * np.arange(k) / sr)
    burst[:48] *= np.linspace(0, 1, 48)                            # no click at the pulse onset
    b, i = 0.0, 0
    while b < dur:
        s0 = int(b * sr)
        m = max(0, min(k, n - s0))
        pulse[s0: s0 + m] += (1.0 if i % 4 == 0 else 0.55) * burst[:m]
        b += BEAT; i += 1
    out = 0.8 * pad + 0.5 * pulse
    fd = int(sr * 1.5)
    out[:fd] *= np.linspace(0, 1, fd)
    return (out / max(1e-6, np.abs(out).max())).astype(np.float32)


# ------------------------------------------------------------------ clips
CLIP_PATHS = {"room_IS1008b_1670s": "/Volumes/ExternalSSD/nvidia-audio-models/scratch/diar/clips/ami_IS1008b_1670s.wav",
              "oto_5bc1e19ecd43cc6d45d0c1b9b78be5aa": "/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad/clips/oto_5bc1e19ecd43cc6d45d0c1b9b78be5aa.mono.wav"}


def clip_wav(name):
    p = Path(CLIP_PATHS[name]) if name in CLIP_PATHS else ROOT / "demo" / "clips" / f"{name}.mono.wav"
    if not p.exists():
        p = ROOT / "demo" / "clips" / f"{name}.wav"
    w, sr = sf.read(p, dtype="float32")
    return (w.mean(1) if w.ndim > 1 else w), sr


def load_manifest():
    try:
        return json.loads(MANIFEST.read_text())["clips"]
    except Exception:
        return {}


def clip_segment(c, SR, clips_meta, manifest):
    """Audio of clip c = {clip, t0, t1} at -16 LUFS with 150 ms edge fades, and where it came from."""
    t0, t1 = c["t0"], c["t1"]
    m = manifest.get(c["clip"])
    if m and m["span"][0] - 0.02 <= t0 and t1 <= m["span"][1] + 0.02:
        w = read_mono(m["wav"], SR)
        a0, a1 = int(round((t0 - m["offset"]) * SR)), int(round((t1 - m["offset"]) * SR))
        seg = w[max(0, a0): max(0, a1)].copy()
        inner_in, inner_out = abs(t0 - m["span"][0]) > 0.02, abs(t1 - m["span"][1]) > 0.02
        fade(seg, SR, CLIP_FADE * 1000 if inner_in else 0, CLIP_FADE * 1000 if inner_out else 0)
        return seg, {"source": "manifest", "wav": m["wav"], "lufs_file": m.get("lufs")}
    meta = (clips_meta or {}).get(c["clip"], {})
    if meta.get("processed_wav"):
        w = read_mono(meta["processed_wav"], SR)
        off = meta.get("offset") or 0.0
        seg = w[max(0, int(round((t0 - off) * SR))): max(0, int(round((t1 - off) * SR)))].copy()
        src = meta["processed_wav"]
    else:
        w, sr = clip_wav(c["clip"])
        w = resample(w, sr, SR)
        seg = w[int(t0 * SR): int(t1 * SR)].copy()
        src = c["clip"]
    seg = to_lufs(seg, SR, TARGET_LUFS)
    fade(seg, SR, CLIP_FADE * 1000, CLIP_FADE * 1000)
    return seg, {"source": "fallback (measured and normalised here)", "wav": str(src)}


# ------------------------------------------------------------------ mix
def example_blocks(shots, pre=0.05, post=0.05):
    """One mute block per example shot: first clip start -> last clip / agent end (narration inside included)."""
    out = []
    for sh in shots:
        ev = [(c["start"], c["start"] + (c["t1"] - c["t0"])) for c in sh.get("clips", [])]
        ev += [(a["start"], a["start"] + a["dur"]) for a in sh.get("agents", [])]
        if ev:
            out.append((min(a for a, _ in ev) - pre, max(b for _, b in ev) + post))
    return out


def mix(shots, total, SR, clips_meta=None):
    """Returns (final mix, stems). See the module docstring for the rules."""
    n = int(math.ceil(total * SR)) + SR
    voice = np.zeros(n, np.float32)
    clipst = np.zeros(n, np.float32)
    agentst = np.zeros(n, np.float32)
    manifest = load_manifest()
    for sh in shots:
        for v in sh["voice"]:
            w = read_mono(v["wav"], SR)
            w = to_lufs(w, SR, TARGET_LUFS)
            fade(w, SR, 5, 80)
            i = int(round(v["start"] * SR))
            voice[i: i + len(w)] += w[: n - i]
        clip_lufs = None
        for c in sh["clips"]:
            seg, _ = clip_segment(c, SR, clips_meta, manifest)
            clip_lufs = TARGET_LUFS
            i = int(round(c["start"] * SR))
            clipst[i: i + len(seg)] += seg[: n - i]
        for ag in sh["agents"]:
            full = read_mono(ag["wav"], SR)
            L, _ = loudness(full, SR)                     # gain from the whole unit (a trimmed 'Sure,' is too short to gate)
            g = db((clip_lufs or TARGET_LUFS) + AGENT_REL_DB - L) if L > -69 else 1.0
            w = full[: int(round(ag["dur"] * SR))].copy() * g
            fade(w, SR, 10, 30)
            i = int(round(ag["start"] * SR))
            agentst[i: i + len(w)] += w[: n - i]
    # music: constant bed, muted across every example block with raised-cosine ramps outside the block
    bed = music_bed(n / SR, SR)[:n]
    bed = np.pad(bed, (0, n - len(bed)))
    mid = bed[int(10 * SR): int(40 * SR)] if n > 40 * SR else bed
    Lb, _ = loudness(mid, SR)
    music = bed * db(MUSIC_LUFS - Lb)
    gain = np.ones(n, np.float32)
    r = int(MUSIC_RAMP * SR)
    ramp_down = (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))).astype(np.float32)
    for a, b in example_blocks(shots):
        ia, ib = max(0, int(a * SR)), min(n, int(b * SR))
        gain[ia:ib] = 0.0
        s = max(0, ia - r)
        gain[s:ia] = np.minimum(gain[s:ia], ramp_down[r - (ia - s):])
        e = min(n, ib + r)
        gain[ib:e] = np.minimum(gain[ib:e], ramp_down[::-1][: e - ib])
    music = (music * gain).astype(np.float32)
    # end: 1.5 s raised-cosine fade to silence, then END_SILENCE of digital silence
    L = int(round(total * SR))
    endf = np.ones(n, np.float32)
    z = max(0, L - int(END_SILENCE * SR))
    f0 = max(0, z - int(END_FADE * SR))
    endf[f0:z] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, z - f0))
    endf[z:] = 0.0
    voice *= endf; clipst *= endf; agentst *= endf; music *= endf
    out = voice + clipst + agentst + music
    g = limiter(out[:L], SR)
    stems = {"voice": voice[:L] * g, "clip": clipst[:L] * g, "agent": agentst[:L] * g, "music": music[:L] * g}
    stems["example"] = stems["clip"] + stems["agent"]
    return (out[:L] * g).astype(np.float32), {k: v.astype(np.float32) for k, v in stems.items()}


# ------------------------------------------------------------------ QA
def _rms_db(x):
    return 20 * math.log10(math.sqrt(float(np.mean(x.astype(np.float64) ** 2))) + 1e-12)


def transients(x, sr, times, win=0.03, ratio=6.0, floor_db=-50.0):
    """Click detector at edit boundaries: the largest second difference within +-win of each boundary compared with the
    95th percentile of the second difference in the surrounding 400 ms (the recording's own transients set the bar)."""
    d2 = np.abs(np.diff(x.astype(np.float64), 2))
    hits = []
    for t in times:
        i = int(t * sr)
        a, b = max(0, i - int(win * sr)), min(len(d2), i + int(win * sr))
        c0, c1 = max(0, i - int(0.2 * sr)), min(len(d2), i + int(0.2 * sr))
        if b <= a or c1 - c0 < 10:
            continue
        ctx = np.concatenate([d2[c0:a], d2[b:c1]])
        ref = np.percentile(ctx, 95) if len(ctx) else 0.0
        pk = float(d2[a:b].max())
        if 20 * math.log10(pk + 1e-12) > floor_db and pk > ratio * max(ref, 1e-7):
            hits.append({"t": round(t, 3), "peak_d2_db": round(20 * math.log10(pk), 1), "ratio": round(pk / max(ref, 1e-9), 1)})
    return hits


def hard_cuts(x, sr, hop_ms=10, loud_db=-30.0, quiet_db=-80.0):
    """Level falling from above loud_db to digital silence (below quiet_db) within one 10 ms hop (a 150 ms fade on
    speech at -16 LUFS ends around -40 dBFS in its last hop; an unfaded cut drops from about -15 dBFS)."""
    hop = int(sr * hop_ms / 1000)
    k = len(x) // hop
    r = 20 * np.log10(np.sqrt(np.mean(x[: k * hop].astype(np.float64).reshape(k, hop) ** 2, 1)) + 1e-12)
    idx = np.where((r[:-1] > loud_db) & (r[1:] < quiet_db))[0]
    return [round((i + 1) * hop / sr, 3) for i in idx]


def qa_audio(mix_, stems, shots, SR, total=None, fps=30):
    """Measures every mix rule; returns a dict with 'ok' and the failures. Call after mix() with its outputs."""
    total = total or len(mix_) / SR
    rep = {"fail": []}
    I, TP = loudness(mix_, SR)
    rep["mix_integrated_lufs"], rep["mix_true_peak_dbtp"] = I, TP
    if TP > -0.9:
        rep["fail"].append(f"true peak {TP:.1f} dBTP > -1")
    # narration lines and clips at -16 LUFS; agent 3 dB under its clip
    lines, clips, agents = [], [], []
    for sh in shots:
        for v in sh["voice"]:
            a, b = int(v["start"] * SR), int((v["start"] + v["dur"]) * SR)
            lines.append((sh["name"], round(v["start"], 2), loudness(stems["voice"][a:b], SR)[0]))
        for c in sh["clips"]:
            a, b = int(c["start"] * SR), int((c["start"] + c["t1"] - c["t0"]) * SR)
            clips.append((sh["name"], round(c["start"], 2), loudness(stems["clip"][a:b], SR)[0]))
        for g in sh["agents"]:
            a, b = int(g["start"] * SR), int((g["start"] + g["dur"]) * SR)
            agents.append((sh["name"], round(g["start"], 2), round(g["dur"], 2), _rms_db(stems["agent"][a:b])))
    rep["narration_lufs"] = [{"shot": s, "t": t, "lufs": L} for s, t, L in lines]
    rep["clip_lufs"] = [{"shot": s, "t": t, "lufs": L} for s, t, L in clips]
    for s, t, L in lines + clips:
        if L > -69 and abs(L - TARGET_LUFS) > 1.5:
            rep["fail"].append(f"{s} @ {t:.2f}: {L:.1f} LUFS (target -16 +-1.5)")
    # agent vs clip (RMS dB, same shot; the agent units are short so RMS rather than gated LUFS)
    for s, t, d, r in agents:
        cl = [c for c in shots if c["name"] == s][0]["clips"]
        a, b = int(cl[0]["start"] * SR), int((cl[0]["start"] + cl[0]["t1"] - cl[0]["t0"]) * SR)
        act = stems["clip"][a:b]
        hop = int(0.05 * SR)
        k = len(act) // hop
        blk = np.sqrt(np.mean(act[: k * hop].reshape(k, hop) ** 2, 1)) if k else np.zeros(1)
        speech = blk[blk > blk.max() * db(-25)] if blk.max() > 0 else blk
        rep.setdefault("agent_vs_clip_db", []).append({"shot": s, "t": t, "dur": d,
                                                        "agent_rms_minus_clip_speech_rms_db": round(r - 20 * math.log10(float(np.sqrt(np.mean(speech ** 2))) + 1e-12), 1)})
    # narration silent during clips; music muted during example blocks
    spans = [(c["start"], c["start"] + c["t1"] - c["t0"]) for sh in shots for c in sh["clips"]]
    for a, b in spans:
        v = np.abs(stems["voice"][int(a * SR): int(b * SR)]).max(initial=0)
        if v > 1e-4:
            rep["fail"].append(f"narration during clip {a:.2f}-{b:.2f} (peak {20 * math.log10(v):.1f} dBFS)")
    for a, b in example_blocks(shots, 0, 0):
        m = np.abs(stems["music"][int(a * SR): int(b * SR)]).max(initial=0)
        if m > 1e-5:
            rep["fail"].append(f"music during example block {a:.2f}-{b:.2f} (peak {20 * math.log10(m):.1f} dBFS)")
    # music steadiness outside the blocks and their ramps (3 s RMS windows, film fade-in / fade-out excluded)
    blocks = example_blocks(shots, MUSIC_RAMP + 0.1, MUSIC_RAMP + 0.1)
    w = int(3.0 * SR)
    lv = []
    for s0 in range(int(2 * SR), int((total - END_FADE - 0.5) * SR) - w, int(0.5 * SR)):
        t0_, t1_ = s0 / SR, (s0 + w) / SR
        if any(t0_ < b and t1_ > a for a, b in blocks):
            continue
        lv.append(_rms_db(stems["music"][s0: s0 + w]))
    if lv:
        rep["music_level_db_p5_p95"] = [round(float(np.percentile(lv, 5)), 1), round(float(np.percentile(lv, 95)), 1)]
        spread = rep["music_level_db_p5_p95"][1] - rep["music_level_db_p5_p95"][0]
        rep["music_level_spread_db"] = round(spread, 1)
        if spread > 3.0:
            rep["fail"].append(f"music bed not steady between lines: {spread:.1f} dB spread (3 s windows)")
    # end: last frame silent, fade present
    last = mix_[-int(SR / fps):]
    rep["last_frame_peak"] = float(np.abs(last).max(initial=0))
    if rep["last_frame_peak"] > 1e-4:
        rep["fail"].append(f"last frame not silent (peak {rep['last_frame_peak']:.2e})")
    # clicks at every edit boundary, per stem that was edited there, and on the mix
    bounds = {"voice": [], "clip": [], "agent": [], "music": []}
    for sh in shots:
        bounds["voice"] += [x for v in sh["voice"] for x in (v["start"], v["start"] + v["dur"])]
        bounds["clip"] += [x for c in sh["clips"] for x in (c["start"], c["start"] + c["t1"] - c["t0"])]
        bounds["agent"] += [x for g in sh["agents"] for x in (g["start"], g["start"] + g["dur"])]
    bounds["music"] = [x for a, b in example_blocks(shots) for x in (a - MUSIC_RAMP, a, b, b + MUSIC_RAMP)]
    allb = sorted({round(x, 3) for v in bounds.values() for x in v} | {round(sh["start"], 3) for sh in shots} | {round(total - END_FADE, 3)})
    rep["clicks"] = {k: transients(stems[k], SR, v) for k, v in bounds.items()}
    # on the mix: only transients the summing / limiter created (mix peak well above every stem's own peak there)
    mixhits = []
    for h in transients(mix_, SR, allb):
        i, wn = int(h["t"] * SR), int(0.03 * SR)
        own = max(float(np.abs(np.diff(stems[k][max(0, i - wn): i + wn].astype(np.float64), 2)).max(initial=0))
                  for k in ("voice", "clip", "agent", "music"))
        if 10 ** (h["peak_d2_db"] / 20) > 1.5 * own:
            mixhits.append(h)
    rep["clicks"]["mix"] = mixhits
    for k, v in rep["clicks"].items():
        for h in v:
            rep["fail"].append(f"click on {k} at {h['t']:.3f} s (x{h['ratio']})")
    rep["hard_cuts"] = {k: hard_cuts(stems[k], SR) for k in ("voice", "clip", "agent", "music")}
    rep["hard_cuts"]["mix"] = hard_cuts(mix_, SR)
    for k, v in rep["hard_cuts"].items():
        for t in v:
            rep["fail"].append(f"hard cut to silence on {k} at {t:.3f} s")
    rep["n_boundaries_checked"] = len(allb)
    rep["ok"] = not rep["fail"]
    return rep


if __name__ == "__main__":
    # self-test on the current timeline: python demo/archive/v5/audio.py [timeline.json]
    import sys
    tl = Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO_OUT / "showcase_v5.timeline.json"
    j = json.loads(tl.read_text())
    data = json.loads((DEMO_OUT / "v5_data" / "data.json").read_text())["clips"]
    meta = {k: {"processed_wav": v.get("processed_wav"), "offset": v.get("offset")} for k, v in data.items()}
    SR = 48000
    m, st = mix(j["shots"], j["total"], SR, meta)
    out = DEMO_OUT / "v5_data" / "clips" / "_work"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "selftest_mix.wav", m, SR)
    for k, v in st.items():
        sf.write(out / f"selftest_stem_{k}.wav", v, SR)
    rep = qa_audio(m, st, j["shots"], SR, j["total"])
    print(json.dumps({k: v for k, v in rep.items() if k not in ("clicks",)}, indent=1)[:6000])
