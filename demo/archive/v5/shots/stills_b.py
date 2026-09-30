"""Stills + design checks for the B shots (demo/v5/shots_b.json), in 16:9 and 1:1, with the film's Playwright venv.

    V=/Volumes/afdev/venvs/video/bin/python
    $V demo/archive/v5/shots/stills_b.py [--only chart_icsi,scorecard] [--at 5.0]
      -> $DEMO_OUT/v5_stills/shots_b/<name>_<W>x<H>.png + checks.json

One Chromium, one page. Per still: DOM metrics (demo/archive/v5/metrics.js) and these checks:
  edge      every visible text >= 64 px from the frame edge
  type      headline >= 96 px, fine print >= 28 px, setup line >= 36 px, every other text >= 44 px; one family; weights 400/700
  contrast  >= 7:1 for every visible text
  overflow  no clipped text / ellipsis;  panel: nothing crosses its data-panel box
  primary   the data-primary element spans >= 80 % of the width or the height
  number    data-final numbers equal their final value once their animation is done
  countup   during the bar growth every visible data-final label already shows its final value (or nothing)
  overlap   no two visible text boxes intersect
  jargon    no F1 / RTF / recall / FP / CI / "points" / number ranges on screen outside the fine print
  motion    over 500 ms windows after the build (4 windows from the still time), >= 0.2 % of pixels change by > 12 levels
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out")) / "v5_stills" / "shots_b"
JARGON = re.compile(r"\bF1\b|\bRTF\b|recall|\bFP\b|\bCI\b|\bpoints?\b|\d\s*[–-]\s*\d|±")


def role(t):
    c = str(t["cls"])
    if "h1" in c:
        return "headline", 96
    if c == "fine":
        return "fine print", 28
    if c == "setup":
        return "setup", 36
    if c == "caption":
        return "caption", 40
    return "label", 44


def check(m, W, H, name, lt):
    fails = []
    for t in m["texts"]:
        r, need = role(t)
        if t["size"] + 0.5 < need:
            fails.append(f"type {r} {t['size']:.0f}px < {need}: '{t['text']}'")
        if t["contrast"] < 7:
            fails.append(f"contrast {t['contrast']}:1 '{t['text']}'")
        if min(t["x0"], t["y0"], W - t["x1"], H - t["y1"]) < 64 - 0.5:
            fails.append(f"edge {min(t['x0'], t['y0'], W - t['x1'], H - t['y1']):.0f}px '{t['text']}'")
    fams = {t["family"] for t in m["texts"]}
    wts = {str(t["weight"]) for t in m["texts"]}
    if len(fams) > 1:
        fails.append(f"fonts {fams}")
    if not wts <= {"400", "700"}:
        fails.append(f"weights {wts}")
    vis = [t for t in m["texts"] if role(t)[0] not in ("fine print", "caption")]
    for t in vis:
        if JARGON.search(t["text"]):
            fails.append(f"jargon '{t['text']}'")
    for i, a in enumerate(vis):
        for b in vis[i + 1:]:
            ix = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]); iy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if ix > 2 and iy > 0.25 * min(a["y1"] - a["y0"], b["y1"] - b["y0"]):
                fails.append(f"overlap '{a['text']}' x '{b['text']}'")
    for o in m["overflow"]:
        fails.append(f"overflow {o}")
    for c in m["crossings"]:
        fails.append(f"panel {c}")
    p = m["primary"]
    span = None
    if p:
        span = {"w": round((p["x1"] - p["x0"]) / W, 3), "h": round((p["y1"] - p["y0"]) / H, 3)}
        if max(span.values()) < 0.8:
            fails.append(f"primary span {span}")
    else:
        fails.append("no data-primary element")
    for nm in m["numbers"]:
        mt = re.search(r"-?\d+(?:\.\d+)?", nm["text"].replace("\u2212", "-"))
        if lt >= nm["done"] and (not mt or abs(float(mt.group()) - nm["final"]) > abs(nm["final"]) * 0.006 + 1e-9):
            fails.append(f"number {nm}")
    return {"fails": fails, "primary_span": span, "type_min": {r: min([t["size"] for t in m["texts"] if role(t)[0] == r] or [0]) for r in ("headline", "label", "fine print")},
            "contrast_min": min([t["contrast"] for t in m["texts"]] or [0]), "edge_min_px": round(min([min(t["x0"], t["y0"], W - t["x1"], H - t["y1"]) for t in m["texts"]] or [0]), 1),
            "fonts": sorted(fams), "weights": sorted(wts), "h1": m["h1"]}


def changed(a, b):
    a = np.asarray(Image.open(a).convert("L").reduce(4)).astype(np.int16)
    b = np.asarray(Image.open(b).convert("L").reduce(4)).astype(np.int16)
    return float((np.abs(a - b) > 12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--at", type=float, default=None)
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    OUT = Path(a.out) if a.out else OUT
    spec = json.loads((V5 / "shots_b.json").read_text())
    # a shot may list several still times ("stills": [...]): one still (and one set of checks) per time
    spec["shots"] = [dict(sh, name=f"{sh['name']}_{i + 1}", still=t) if len(sh.get("stills", [])) > 1 else sh
                     for sh in spec["shots"] for i, t in enumerate(sh.get("stills", [sh["still"]]))]
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/Volumes/afdev/venvs/video/browsers")
    from playwright.sync_api import sync_playwright
    metrics_js = (V5 / "metrics.js").read_text()
    res = json.loads((OUT / "checks.json").read_text()) if (OUT / "checks.json").exists() else {}
    tmp = OUT / "_motion"
    tmp.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--disable-gpu", "--font-render-hinting=none", "--renderer-process-limit=1"])
        for W, H in ((1920, 1080), (1080, 1080)):
            pg = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            for sh in spec["shots"]:
                if a.only and sh["name"] not in a.only.split(",") and sh["name"].rsplit("_", 1)[0] not in a.only.split(","):
                    continue
                errs.clear()
                pg.goto((HERE / sh["page"]).as_uri())
                pg.evaluate("document.fonts.ready")
                params = dict(sh.get("params", {}), page=sh["page"], name=sh["name"], dur=sh["dur"])
                pg.evaluate("([p, W, H]) => { window.PARAMS = p; init({numbers: {}, clips: {}}, p, W, H); }", [params, W, H])
                cap = re.sub(r"\{([^|}]*)\|[^}]*\}", r"\1", sh.get("narration", ""))
                words, cap = cap.split(), ""                     # film.py shows the narration in chunks; use the first one
                while words and len(cap) + len(words[0]) < (38 if H == W else 60):
                    cap = (cap + " " + words.pop(0)).strip()
                ctx = {"caption": cap, "fine": pg.evaluate("window.FINE || ''"), "dur": sh["dur"], "cues": {}, "segs": []}
                # count-up check: sample the growth window; a visible value label must already read its final value
                cu = []
                for tt in np.arange(0.0, 1.8, 0.1):
                    pg.evaluate("([t, c]) => render(t, c)", [float(tt), ctx])
                    for nm in pg.evaluate(metrics_js)["numbers"]:
                        mt = re.search(r"-?\d+(?:\.\d+)?", nm["text"].replace("\u2212", "-"))
                        if nm["text"] and (not mt or abs(float(mt.group()) - nm["final"]) > 1e-9):
                            cu.append(f"t={tt:.1f} '{nm['text']}' != {nm['final']}")
                lt = sh["still"] if a.at is None else a.at
                pg.evaluate("([t, c]) => render(t, c)", [lt, ctx])
                png = OUT / f"{sh['name']}_{W}x{H}.png"
                pg.screenshot(path=str(png), type="png")
                m = pg.evaluate(metrics_js)
                r = check(m, W, H, sh["name"], lt)
                r["fails"] += [f"countup {c}" for c in cu[:5]]
                r["shown"] = [nm["text"] for nm in m["numbers"]]
                if errs:
                    r["fails"].insert(0, f"page errors: {errs}")
                # motion over 500 ms windows from the end of the build to the end of the shot (sampled)
                if not a.no_motion:
                    ts = [x for x in np.arange(1.4, sh["dur"] - 0.5 + 1e-6, max(0.5, (sh["dur"] - 1.9) / 6))]
                    vals = []
                    for t0 in ts:
                        for k, tt in enumerate((t0, t0 + 0.5)):
                            pg.evaluate("([t, c]) => render(t, c)", [float(tt), ctx])
                            pg.screenshot(path=str(tmp / f"m{k}.png"), type="png")
                        vals.append(round(changed(tmp / "m0.png", tmp / "m1.png"), 4))
                    r["motion"] = vals
                    if min(vals) < 0.002:
                        r["fails"].append(f"motion min {min(vals)} < 0.002")
                res[f"{sh['name']}_{W}x{H}"] = r
                print(f"{sh['name']:16s} {W}x{H} t={lt:.2f} span {r['primary_span']} type {r['type_min']} contrast {r['contrast_min']} edge {r['edge_min_px']} "
                      f"motion {min(r.get('motion') or [0]):.4f}  FAIL: {r['fails'] or 'none'}", flush=True)
            pg.close()
        br.close()
    (OUT / "checks.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
