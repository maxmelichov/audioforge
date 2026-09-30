"""Render the static architecture image at 2x with Playwright and run the static checks.

    /Volumes/afdev/venvs/video/bin/python demo/archive/v5/images/render_images.py
      -> $DEMO_OUT/images/architecture.png (3840x2160), architecture_square.png (2160x2160)
         demo/archive/images/architecture.png, architecture_square.png (half size)

Checks (from demo/archive/v5/metrics.js): no text within 48 px (1x) of an edge, no overlapping text, no clipped text,
contrast >= 4.5:1 for every text (small labels use the secondary grey), smallest text size reported.
"""
import json
import os
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PAGE = "architecture2.html"   # v2: plain-language figure (architecture.html = v1, engineering detail)
ROOT = HERE.parents[2]
OUT = Path(os.environ.get("DEMO_OUT", "/Volumes/ExternalSSD/nvidia-audio-models/demo_out"))


def main():
    from playwright.sync_api import sync_playwright
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/Volumes/afdev/venvs/video/browsers")
    clip = json.loads((OUT / "v5_data" / "data.json").read_text())["clips"]["room_IS1008b_1670s"]
    data = {"env": clip["env"], "mel": clip["mel"]}
    metrics = (HERE.parent / "metrics.js").read_text()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "demo" / "images").mkdir(parents=True, exist_ok=True)
    report, ok = {}, True
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--font-render-hinting=none"])
        for W, H, name in ((1920, 1080, "architecture"), (1080, 1080, "architecture_square")):
            pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            pg.goto((HERE / PAGE).as_uri())
            pg.evaluate("document.fonts.ready")
            pg.evaluate("([W, H, d]) => build(W, H, d)", [W, H, data])
            n_te = pg.evaluate("document.querySelectorAll('[data-te]:not(.foot)').length")
            out = OUT / "images" / f"{name}.png"
            pg.screenshot(path=str(out), type="png")
            m = pg.evaluate(metrics)
            fails = []
            tx = m["texts"]
            for t in tx:
                if t["x0"] < 48 or t["y0"] < 48 or t["x1"] > W - 48 or t["y1"] > H - 48:
                    fails.append(f"edge: '{t['text']}'")
                if t["contrast"] < 4.5:
                    fails.append(f"contrast {t['contrast']}: '{t['text']}'")
            for i in range(len(tx)):
                for j in range(i + 1, len(tx)):
                    A, B = tx[i], tx[j]
                    ox = min(A["x1"], B["x1"]) - max(A["x0"], B["x0"]); oy = min(A["y1"], B["y1"]) - max(A["y0"], B["y0"])
                    hmin = min(A["y1"] - A["y0"], B["y1"] - B["y0"])
                    if ox > 2 and oy > max(4, 0.25 * hmin) and A["y0"] != B["y0"] and abs(A["y1"] - B["y1"]) > 2:   # same line = inline runs
                        fails.append(f"overlap: '{A['text']}' x '{B['text']}'")
            fails += [f"overflow: {o}" for o in m["overflow"]]
            if n_te > 12:
                fails.append(f"{n_te} text elements above the footnote (max 12)")
            if min(t["size"] for t in tx) < 23.5:
                fails.append(f"smallest text {min(t['size'] for t in tx):.1f} px")
            report[name] = {"size_px": [W * 2, H * 2], "text_elements": n_te, "min_text_px_at_1x": min(t["size"] for t in tx), "fails": fails}
            ok = ok and not fails
            Image.open(out).resize((W, H), Image.LANCZOS).save(ROOT / "demo" / "images" / f"{name}.png", optimize=True)
            pg.close()
        b.close()
    (OUT / "images" / "checks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
