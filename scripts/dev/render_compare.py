# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright>=1.40"]
# ///
"""Render the README comparison images (demo/images/compare_<key>.png) from demo/images/redesign/arch_vs.html.

    python3 demo/images/redesign/export_final.py                                   # numbers_final.js from runs/
    /Volumes/afdev/venvs/video/bin/python scripts/dev/render_compare.py [asr,turn,vad,spk,lid]
    chore images                                                                   # both steps

The page reads its numbers from demo/images/redesign/numbers_final.js (export_final.py writes it from
runs/final_compare.json), so run the export first. One Chromium, one 1920x1080 page at device scale 2.
With `uv run scripts/dev/render_compare.py` run `uv run --with playwright playwright install chromium` once.

Every page is checked before its screenshot (CHECK below): no two text boxes overlap, no text spills out of its chart
card or the 1920x1080 frame, no chart card overlaps another block, and barchart() reported no overlap it could not fix
by shrinking the x labels. Problems are printed per page and the script exits 1. `--check-only` skips the PNGs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "demo" / "images" / "redesign" / "arch_vs.html"
KEYS = ["asr", "turn", "vad", "spk", "lid"]

# runs in the page after buildVs(): returns a list of problems (empty = clean)
CHECK = r"""
() => {
  const W = 1920, H = 1080, out = [];
  const vis = e => { const cs = getComputedStyle(e); if (cs.display === "none" || cs.visibility === "hidden") return false;
    for (let a = e; a; a = a.parentElement) { const c = getComputedStyle(a); if (c.display === "none" || a.classList.contains("dim")) return false; }
    const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const txt = e => (e.textContent || "").trim().slice(0, 40);
  // the box of the element and of its text (a long word can overflow a fixed-width box)
  const box = e => { const r = e.getBoundingClientRect(), g = document.createRange(); g.selectNodeContents(e); const q = g.getBoundingClientRect();
    const t = e.classList.contains("xl") || e.classList.contains("val") || e.classList.contains("dl");   // chart labels: the text itself
    return q.width > 0 && t ? { e, l: q.left, t: q.top, r: q.right, b: q.bottom }
         : q.width > 0 ? { e, l: Math.min(r.left, q.left), t: Math.min(r.top, q.top), r: Math.max(r.right, q.right), b: Math.max(r.bottom, q.bottom) }
         : { e, l: r.left, t: r.top, r: r.right, b: r.bottom }; };
  const inter = (a, b) => Math.min(a.r, b.r) - Math.max(a.l, b.l) > 1 && Math.min(a.b, b.b) - Math.max(a.t, b.t) > 1;
  const sc = document.getElementById("score"), st = document.getElementById("stage");
  // text leaves on the page: score texts + the lit diagram labels
  const leaves = [...sc.querySelectorAll(".col-h, .col-s, .tb .m, .tb .a, .bc .ph, .bc .pu, .bc .val, .bc .xl, .bc .dl, .bc .nm")]
    .concat([...st.querySelectorAll(".l1, .l2, .box, .abs")]).filter(vis);
  const B = leaves.map(box);
  for (const b of B) if (b.l < -1 || b.t < -1 || b.r > W + 1 || b.b > H + 1) out.push(`off-frame: "${txt(b.e)}"`);
  for (let i = 0; i < B.length; i++) for (let j = i + 1; j < B.length; j++) {
    const a = B[i], b = B[j]; if (a.e.contains(b.e) || b.e.contains(a.e)) continue;
    if (st.contains(a.e) && st.contains(b.e)) continue;   // the diagram's own layout is checked by arch_v10
    const lab = e => /\b(xl|val|dl)\b/.test(e.className), gap = a.e.classList.contains("xl") && b.e.classList.contains("xl") ? 6 : lab(a.e) && lab(b.e) ? 3 : 0;
    // chart labels must keep a gap: 6 px between x labels, 3 px between other chart labels
    if (inter(a, b) || (gap && Math.min(a.r, b.r) - Math.max(a.l, b.l) > -gap && Math.min(a.b, b.b) - Math.max(a.t, b.t) > 1))
      out.push(`overlap: "${txt(a.e)}" x "${txt(b.e)}"`); }
  // chart cards: children inside the card, cards clear of every other block
  const cards = [...sc.querySelectorAll(".bc")].map(box);
  for (const c of cards) for (const k of c.e.querySelectorAll(".ph, .pu, .val, .xl, .dl, .nm")) { const b = box(k);
    if (b.l < c.l - 1 || b.r > c.r + 1 || b.t < c.t - 1 || b.b > c.b + 1) out.push(`spills out of chart "${txt(c.e.querySelector(".ph"))}": "${txt(k)}"`); }
  const blocks = cards.concat([...sc.querySelectorAll(".tb, .col-h, .col-s")].filter(vis).map(box))
    .concat([...st.querySelectorAll(".box, .blk, .abs, .l1")].filter(vis).map(box));
  for (let i = 0; i < cards.length; i++) for (const b of blocks) { if (b.e === cards[i].e || cards[i].e.contains(b.e)) continue;
    if (inter(cards[i], b)) out.push(`chart "${txt(cards[i].e.querySelector(".ph"))}" overlaps "${txt(b.e)}"`); }
  if (window.OVERLAPS) out.push(`barchart() left ${window.OVERLAPS} chart(s) with overlapping labels`);
  return [...new Set(out)];
}
"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check_only = "--check-only" in sys.argv
    keys = args[0].split(",") if args else KEYS
    bad = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--font-render-hinting=none"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        for k in keys:
            page.goto(PAGE.as_uri())
            page.wait_for_timeout(500)
            page.evaluate("window.OVERLAPS = 0")
            page.evaluate(f"buildVs(1920, 1080, '{k}')")
            page.wait_for_timeout(800)
            probs = page.evaluate(CHECK)
            print(f"check {k}: {'clean' if not probs else str(len(probs)) + ' problem(s)'}")
            for q in probs:
                print("   ", q)
            bad += bool(probs)
            if check_only:
                continue
            out = ROOT / "demo" / "images" / f"compare_{k}.png"
            page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1920, "height": 1080})
            print("rendered", out.relative_to(ROOT))
        browser.close()
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
