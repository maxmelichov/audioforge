"""Static result graph for the post: results.png (demo/archive/v5/shots/results_img.html?view=overview), rendered with the
film's Playwright venv at 2x, plus checks.

    V=/Volumes/afdev/venvs/video/bin/python
    scripts/dev/gate.sh $V demo/archive/v5/shots/images_b.py
      -> $DEMO_OUT/images/results.png (3840x2160), results_square.png (2160x2160), checks_results.json
      -> demo/archive/images/results.png (1920x1080), results_square.png (1080x1080)   (half-size copies)

Checks (in CSS px, before the 2x): every text >= 64 px from the frame edge; no two text boxes intersect; contrast
>= 7:1 against the background; no clipped / overflowing text; every value label equals its numbers_b "shown" value;
every text inside its panel; smallest type >= 20 px (40 px in the PNG).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out")) / "images"
REPO = ROOT / "demo" / "images"

METRICS = r"""() => {
  const lum = c => { const m = c.match(/[\d.]+/g).map(Number); const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]); };
  const bg = lum(getComputedStyle(document.getElementById("stage")).backgroundColor);
  const texts = [];
  const walk = el => { for (const n of el.childNodes) {
    if (n.nodeType === 3 && n.textContent.trim()) { const r = document.createRange(); r.selectNodeContents(n); const b = r.getBoundingClientRect(); const p = n.parentElement, cs = getComputedStyle(p);
      const l = lum(cs.color), c = (Math.max(l, bg) + 0.05) / (Math.min(l, bg) + 0.05);
      const pn = p.closest("[data-panel]");
      texts.push({ text: n.textContent.trim().slice(0, 60), x0: b.left, y0: b.top, x1: b.right, y1: b.bottom, size: parseFloat(cs.fontSize), contrast: +c.toFixed(2),
                   panel: pn ? pn.dataset.panel : null, pbox: pn ? (r2 => [r2.left, r2.top, r2.right, r2.bottom])(pn.getBoundingClientRect()) : null,
                   overflow: p.scrollWidth > p.clientWidth + 1 && getComputedStyle(p).display !== "inline" && p.className !== "cell" });
    } else if (n.nodeType === 1) walk(n); } };
  walk(document.getElementById("stage"));
  const vals = [...document.querySelectorAll("[data-final]")].map(e => ({ final: e.dataset.final, text: e.textContent }));
  return { texts, vals, W: innerWidth, H: innerHeight };
}"""


def check(m):
    W, H, T, fails = m["W"], m["H"], m["texts"], []
    for t in T:
        e = min(t["x0"], t["y0"], W - t["x1"], H - t["y1"])
        if e < 63.5:
            fails.append(f"edge {e:.0f}px '{t['text']}'")
        if t["contrast"] < 7:
            fails.append(f"contrast {t['contrast']} '{t['text']}'")
        if t["size"] < 20:
            fails.append(f"size {t['size']}px '{t['text']}'")
        if t["overflow"]:
            fails.append(f"overflow '{t['text']}'")
        if t["pbox"]:
            x0, y0, x1, y1 = t["pbox"]
            if t["x0"] < x0 - 1 or t["x1"] > x1 + 1 or t["y0"] < y0 - 1 or t["y1"] > y1 + 1:
                fails.append(f"outside panel {t['panel']} '{t['text']}'")
    for i, a in enumerate(T):
        for b in T[i + 1:]:
            if min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) > 0 and min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]) > 0:
                fails.append(f"overlap '{a['text']}' x '{b['text']}'")
    for v in m["vals"]:
        if not v["text"].replace(" ", " ").startswith(v["final"] + " ") and v["text"] != v["final"]:
            fails.append(f"value '{v['text']}' != {v['final']}")
    return {"fails": fails, "min_edge": round(min(min(t["x0"], t["y0"], W - t["x1"], H - t["y1"]) for t in T), 1),
            "min_contrast": min(t["contrast"] for t in T), "min_size": min(t["size"] for t in T), "values": [v["text"] for v in m["vals"]]}


def main():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/Volumes/afdev/venvs/video/browsers")
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    REPO.mkdir(parents=True, exist_ok=True)
    res = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(args=["--disable-gpu", "--font-render-hinting=none"])
        for W, H, name in ((1920, 1080, "results"), (1080, 1080, "results_square")):
            pg = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto((HERE / "results_img.html").as_uri() + "?view=overview")
            pg.evaluate("document.fonts.ready")
            pg.evaluate("([W, H]) => build('overview', W, H)", [W, H])
            png = OUT / f"{name}.png"
            pg.screenshot(path=str(png), type="png")
            r = check(pg.evaluate(METRICS))
            if errs:
                r["fails"].insert(0, f"page errors {errs}")
            im = Image.open(png)
            r["size_px"] = list(im.size)
            im.resize((W, H), Image.LANCZOS).save(REPO / f"{name}.png", optimize=True)
            res[name] = r
            print(f"{name:15s} {im.size} edge {r['min_edge']} contrast {r['min_contrast']} type {r['min_size']}px values {r['values']}  FAIL: {r['fails'] or 'none'}")
            pg.close()
        br.close()
    (OUT / "checks_results.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT, REPO)


if __name__ == "__main__":
    main()
