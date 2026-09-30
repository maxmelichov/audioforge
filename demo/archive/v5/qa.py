"""QA pass over a rendered v5 cut: picture (2 fps DOM samples taken during the frame render + the PNG frames) and
sound (the mix and its stems). Fails loudly; writes $DEMO_OUT/v5_qa_<cut>_<W>x<H>.json.

    /Volumes/afdev/venvs/video/bin/python demo/archive/v5/qa.py --cut pilot2 [--size 1920x1080]

Picture checks, on every 2 fps sample:
  edge      no text within 48 px of the frame edge
  overflow  no clipped text or ellipsis (elements marked data-clipok keep only their newest lines by design)
  panel     no element crossing its panel (data-panel) boundary
  empty     at most 60 % of the frame in flat 40 px blocks (background or empty panels)
  caption   no spelled-out numbers in captions (captions show digits)
  number    a number with data-final is within 5 % of its final value once its animation (data-done) has ended
  headline  a shot's headline changes at most once every 2 s
  type      body/transcript >= 44 px, headline >= 96 px (84 in 1:1), caption >= 40, fine print and labels >= 28;
            one family; weights 400/700; contrast >= 7:1
Sound checks: narration never overlaps an example span by > 50 ms (timeline and voice-stem energy), music silent
inside example spans (final mix minus example stem), no clicks (sample jumps), loudness per narration line and
per example span (active RMS, dBFS).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))
NUMWORDS = r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|percent|point)\b"


def role(t, sq):
    c = str(t["cls"])
    if "h1" in c:
        return "headline", 84 if sq else 96
    if c == "caption":
        return "caption", 40
    if c in ("w", "ph") or "words" in c or c.startswith("tx"):
        return "body", 44
    return "label", 28


def flat_fraction(png, blk=40):
    """Share of the frame that is empty background: 40 px blocks that are flat (std < 2.5 levels) AND within 10 levels
    of the navy -> indigo background gradient at their row (empty panels and the drifting light are not background)."""
    rgb = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)
    H, W, _ = rgb.shape
    h, w = H // blk, W // blk
    g = np.linspace(0, 1, H)[:, None]
    ref = np.array([11, 11, 12]) * (1 - g) + np.array([11, 11, 12]) * g   # flat near-black background
    lum = rgb.mean(2)
    b = lum[: h * blk, : w * blk].reshape(h, blk, w, blk)
    flat = b.std(axis=(1, 3)) < 2.5
    mean = rgb[: h * blk, : w * blk].reshape(h, blk, w, blk, 3).mean(axis=(1, 3))
    refb = ref[: h * blk].reshape(h, blk, 3).mean(1)[:, None, :]
    bg = np.abs(mean - refb).max(2) < 10
    return float((flat & bg).mean())


def act_rms_db(x, sr):
    hop = int(0.02 * sr)
    k = len(x) // hop
    if k == 0:
        return None
    r = np.sqrt(np.mean(x[: k * hop].reshape(k, hop) ** 2, 1) + 1e-12)
    a = r[r > r.max() * 10 ** (-30 / 20)]
    return round(float(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="pilot2"); ap.add_argument("--size", default="1920x1080")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x")); sq = W == H
    fd = OUT / "v5_frames" / f"{a.cut}_{W}x{H}"
    samples = [json.loads(l) for l in (fd / "dom_samples.jsonl").open()]
    samples.sort(key=lambda s: s["f"])
    fails = {k: [] for k in ("motion", "text_overlap", "edge", "overflow", "panel", "geometry", "empty", "caption", "number", "headline", "type", "contrast", "font")}
    fams, weights, empties = set(), set(), []
    last_h = {}
    for s in samples:
        tag = f"{s['shot']}@{s['lt']:.1f}s"
        for t in s["texts"]:
            fams.add(t["family"]); weights.add(str(t["weight"]))
            if t["x0"] < 48 or t["y0"] < 48 or t["x1"] > W - 48 or t["y1"] > H - 48:
                fails["edge"].append(f"{tag} '{t['text']}' at {[round(t[k]) for k in ('x0', 'y0', 'x1', 'y1')]}")
            r, need = role(t, sq)
            if t["size"] + 0.5 < need:
                fails["type"].append(f"{tag} {r} {t['size']:.0f}px '{t['text']}'")
            if t["contrast"] < 7:
                fails["contrast"].append(f"{tag} {t['contrast']}:1 '{t['text']}'")
        # no two text elements may overlap (e.g. a callout over the headline)
        tx = [t for t in s["texts"] if t["x1"] - t["x0"] > 2]
        seen_pairs = set()
        for i in range(len(tx)):
            for j in range(i + 1, len(tx)):
                A_, B_ = tx[i], tx[j]
                ox = min(A_["x1"], B_["x1"]) - max(A_["x0"], B_["x0"]); oy = min(A_["y1"], B_["y1"]) - max(A_["y0"], B_["y0"])
                hmin = min(A_["y1"] - A_["y0"], B_["y1"] - B_["y0"]); wmin = min(A_["x1"] - A_["x0"], B_["x1"] - B_["x0"])
                # line boxes of large type include leading below the glyphs; count an overlap of glyph areas only
                if ox > max(2, 0.1 * wmin) and oy > max(4, 0.25 * hmin):
                    key = (A_["text"], B_["text"])
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        fails["text_overlap"].append(f"{tag} '{A_['text']}' x '{B_['text']}'")
        for o in s["overflow"]:
            fails["overflow"].append(f"{tag} {o}")
        for c in s["crossings"]:
            fails["panel"].append(f"{tag} {c}")
        for g in s.get("geom", []):
            fails["geometry"].append(f"{tag} label '{g['el']}' over drawn geometry ({g['covered']:.0%} of its area)")
        if s.get("caption") and re.search(NUMWORDS, s["caption"].lower()):
            fails["caption"].append(f"{tag} '{s['caption']}'")
        for n in s["numbers"]:
            m = re.search(r"-?\d+(\.\d+)?", n["text"].replace(",", ""))
            if m and s["lt"] >= n["done"] + 0.05 and n["final"] and abs(float(m.group(0)) - n["final"]) > 0.05 * abs(n["final"]):
                fails["number"].append(f"{tag} shows {n['text']} final {n['final']}")
        h = tuple(s["h1"])
        prev = last_h.get(s["shot"])
        if prev and not prev[0] and h:      # a headline appearing on an empty frame is not a change
            last_h[s["shot"]] = (h, s["lt"])
        elif prev and h and prev[0] != h:
            if s["lt"] - prev[1] < 2.0 - 1e-6:
                fails["headline"].append(f"{tag} '{' '.join(h)}' after {s['lt'] - prev[1]:.1f}s")
            last_h[s["shot"]] = (h, s["lt"])
        elif not prev:
            last_h[s["shot"]] = (h, s["lt"])
        png = fd / f"{s['f']:05d}.png"
        if png.exists():
            e = flat_fraction(png); empties.append(e)
            # v6 flat near-black palette (keynote restraint, user request): black space is intentional, so the check
            # flags only frames that are essentially empty (> 95 % flat background; a single headline number on black is the reference style); v5's gradient used 60 %
            if e > 0.95:
                fails["empty"].append(f"{tag} {e:.0%} flat")
    # motion: something visibly changes in every 500 ms window (frame f vs f+15, 4x downsampled, > 12 levels)
    tlm = json.loads((OUT / f"{'showcase_v6' if a.cut == 'v6' else 'showcase_v5' + ('' if a.cut == 'full' else '_' + a.cut)}{'_square' if sq else ''}.timeline.json").read_text())
    for sh in tlm["shots"]:
        f0, f1 = int(round(sh["start"] * 30)), int(round(sh["end"] * 30))
        for f in range(f0, f1 - 15, 15):
            pa, pb = fd / f"{f:05d}.png", fd / f"{f + 15:05d}.png"
            if not (pa.exists() and pb.exists()):
                continue
            A_ = np.asarray(Image.open(pa).convert("L").reduce(4)).astype(np.int16)
            B_ = np.asarray(Image.open(pb).convert("L").reduce(4)).astype(np.int16)
            ch = float((np.abs(A_ - B_) > 12).mean())
            if ch < 0.002:
                fails["motion"].append(f"{sh['name']}@{f / 30 - sh['start']:.1f}s only {ch:.4f} of pixels change in 500 ms")
    if len(fams) != 1:
        fails["font"].append(f"families {sorted(fams)}")
    if not weights <= {"400", "700"}:
        fails["font"].append(f"weights {sorted(weights)}")
    # ---- sound
    stem = f"{'showcase_v6' if a.cut == 'v6' else 'showcase_v5' + ('' if a.cut == 'full' else '_' + a.cut)}{'_square' if sq else ''}"
    tl = json.loads((OUT / f"{stem}.timeline.json").read_text())
    tag = f"v5_{a.cut}"
    mix, sr = sf.read(OUT / f"{tag}_mix.wav", dtype="float32")
    voice, _ = sf.read(OUT / f"{tag}_stem_voice.wav", dtype="float32")
    example, _ = sf.read(OUT / f"{tag}_stem_example.wav", dtype="float32")
    sound = {"overlap": [], "music_in_example": [], "clicks": [], "loudness": {"voice_lines_dbfs": [], "example_spans_dbfs": []}}
    vspans = [(v["start"], v["start"] + v["dur"]) for sh in tl["shots"] for v in sh["voice"]]
    cspans = [(c["start"], c["start"] + c["t1"] - c["t0"]) for sh in tl["shots"] for c in sh["clips"]] + \
             [(g["start"], g["start"] + g["dur"]) for sh in tl["shots"] for g in sh["agents"]]
    for a0, b0 in cspans:
        for a1, b1 in vspans:
            ov = min(b0, b1) - max(a0, a1)
            if ov > 0.05:
                sound["overlap"].append(f"timeline: voice {a1:.2f}-{b1:.2f} overlaps example {a0:.2f}-{b0:.2f} by {ov:.2f}s")
        seg = voice[int(a0 * sr): int(b0 * sr)]
        hop = int(0.01 * sr)
        k = len(seg) // hop
        if k:
            e = np.abs(seg[: k * hop]).reshape(k, hop).max(1)
            n_loud = int((e > 10 ** (-50 / 20)).sum())
            if n_loud > 5:
                sound["overlap"].append(f"energy: voice stem active for {n_loud * 10} ms inside example {a0:.2f}-{b0:.2f}")
        res = mix[int(a0 * sr): int(b0 * sr)] - example[int(a0 * sr): int(b0 * sr)]
        rdb = 20 * np.log10(np.sqrt(np.mean(res ** 2)) + 1e-12) if len(res) else -200
        if rdb > -60:
            sound["music_in_example"].append(f"{a0:.2f}-{b0:.2f}: residual {rdb:.1f} dBFS")
        sound["loudness"]["example_spans_dbfs"].append(act_rms_db(example[int(a0 * sr): int(b0 * sr)], sr))
    for a1, b1 in vspans:
        sound["loudness"]["voice_lines_dbfs"].append(act_rms_db(voice[int(a1 * sr): int(b1 * sr)], sr))
    # clicks: an edit (start / end of a narration line, a clip, an agent line) must not add a step: check each
    # boundary on the stem that was edited there (the recordings' own transients are not edits)
    stems = {k: sf.read(OUT / f"{tag}_stem_{k}.wav", dtype="float32")[0] for k in ("voice", "clip", "agent")}
    def step(x, b):
        dx = np.abs(np.diff(x[max(0, int(b * sr) - int(0.05 * sr)): int(b * sr) + int(0.05 * sr)]))
        if not len(dx):
            return None
        mid = len(dx) // 2; w0 = dx[max(0, mid - int(0.003 * sr)): mid + int(0.003 * sr)]
        return (w0.max(), np.median(dx)) if len(w0) else None
    for sh in tl["shots"]:
        bl = [("voice", v["start"]) for v in sh["voice"]] + [("voice", v["start"] + v["dur"]) for v in sh["voice"]] + \
             [("clip", c["start"]) for c in sh["clips"]] + [("clip", c["start"] + c["t1"] - c["t0"]) for c in sh["clips"]] + \
             [("agent", g["start"]) for g in sh["agents"]] + [("agent", g["start"] + g["dur"]) for g in sh["agents"]]
        for k, b in bl:
            r = step(stems[k], b)
            if r and r[0] > 0.02 and r[0] > 12 * (r[1] + 1e-4):
                sound["clicks"].append(f"{k} edit at {b:.3f}s: step {r[0]:.3f} vs median {r[1]:.4f}")
        r = step(mix, sh["start"])
        if r and r[0] > 0.05 and r[0] > 12 * (r[1] + 1e-4):
            sound["clicks"].append(f"shot cut at {sh['start']:.3f}s: step {r[0]:.3f}")
    peak = float(np.abs(mix).max())
    n_frames = len(list(fd.glob("*.png")))
    summary = {"samples": len(samples), "frames": n_frames, "expected_frames": int(round(tl["total"] * 30)),
               "fonts": sorted(fams), "weights": sorted(weights), "empty_max": round(max(empties), 3) if empties else None,
               "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
               "picture_fail_counts": {k: len(v) for k, v in fails.items()},
               "sound_fail_counts": {k: len(v) for k, v in sound.items() if k != "loudness"}, "loudness": sound["loudness"]}
    ok = all(len(v) == 0 for v in fails.values()) and not sound["overlap"] and not sound["music_in_example"] and not sound["clicks"] and summary["frames"] == summary["expected_frames"]
    summary["PASS"] = ok
    out = {"summary": summary, "picture": {k: v[:40] for k, v in fails.items()}, "sound": sound}
    (OUT / f"v5_qa_{a.cut}_{W}x{H}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    for k, v in fails.items():
        for x in v[:8]:
            print(f"FAIL {k}: {x}")
    for k in ("overlap", "music_in_example", "clicks"):
        for x in sound[k][:8]:
            print(f"FAIL {k}: {x}")
    if summary["frames"] != summary["expected_frames"]:
        print(f"FAIL frames: {summary['frames']} rendered of {summary['expected_frames']} expected")
    print("QA", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
