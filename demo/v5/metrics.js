// DOM metrics of the current frame for the design / QA checks (evaluated by film.py after render(t)).
() => {
  const W = innerWidth, H = innerHeight;
  const lum = c => { const m = c.match(/[\d.]+/g).map(Number); const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return [0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]), m.length > 3 ? m[3] : 1]; };
  const opacityOf = el => { let op = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement) { const cs = getComputedStyle(e); op *= parseFloat(cs.opacity); if (cs.visibility === "hidden" || cs.display === "none") return 0; } return op; };
  const bgOf = el => { for (let e = el; e && e.nodeType === 1; e = e.parentElement) { const b = getComputedStyle(e).backgroundColor; if (b && !/rgba\(0, 0, 0, 0\)|transparent/.test(b)) { const [l, a] = lum(b); if (a > 0.5) return [b, l]; } } return ["background", lum("rgb(11,11,12)")[0]]; };
  const texts = [], overflow = [], crossings = [], numbers = [];
  const stage = document.getElementById("stage");
  const walk = el => {
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim()) {
        const p = n.parentElement; if (p.closest("svg") && p.tagName !== "text") continue;
        const op = opacityOf(p); if (op < 0.9) continue;
        const range = document.createRange(); range.selectNodeContents(n); const r = range.getBoundingClientRect();
        if (r.width < 1) continue;
        const cs = getComputedStyle(p); const [bg, lb] = bgOf(p); const [lf] = lum(cs.fill && p.tagName === "text" ? cs.fill : cs.color);
        const hi = Math.max(lf, lb), lo = Math.min(lf, lb);
        const scale = p.tagName === "text" ? 1 : (p.getBoundingClientRect().height / (p.offsetHeight || 1)) || 1;
        texts.push({ text: n.textContent.trim().slice(0, 50), size: parseFloat(cs.fontSize) * scale, weight: cs.fontWeight, family: cs.fontFamily.split(",")[0].replace(/"/g, ""),
                     contrast: +((hi + 0.05) / (lo + 0.05)).toFixed(2), bg, color: cs.color, cls: String(p.className && p.className.baseVal !== undefined ? p.className.baseVal : p.className) || p.id || p.tagName,
                     x0: r.left, y0: r.top, x1: r.right, y1: r.bottom });
      } else if (n.nodeType === 1) walk(n);
    }
  };
  walk(stage);
  // overflow / ellipsis on any visible element with clipped content
  stage.querySelectorAll("*").forEach(e => {
    if (e.closest("svg")) return;
    const cs = getComputedStyle(e);
    if (opacityOf(e) < 0.9 || !e.textContent.trim()) return;
    const clip = /hidden|clip/.test(cs.overflow + cs.overflowX + cs.overflowY);
    if (cs.textOverflow === "ellipsis" && e.scrollWidth > e.clientWidth + 1) overflow.push({ el: e.id || e.className, kind: "ellipsis", text: e.textContent.trim().slice(0, 40) });
    else if (clip && e.dataset.clipok === undefined && (e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1)) overflow.push({ el: e.id || e.className, kind: "overflow", sw: e.scrollWidth, cw: e.clientWidth, sh: e.scrollHeight, ch: e.clientHeight, text: e.textContent.trim().slice(0, 40) });
  });
  // elements crossing their panel (data-panel) boundary
  stage.querySelectorAll("[data-panel]").forEach(pn => {
    if (opacityOf(pn) < 0.05) return;
    const pr = pn.getBoundingClientRect();
    pn.querySelectorAll("*").forEach(c => {
      if (opacityOf(c) < 0.3 || c.closest("[data-free]")) return;
      const r = c.getBoundingClientRect(); if (r.width < 1 || r.height < 1) return;
      if (r.left < pr.left - 2 || r.right > pr.right + 2 || r.top < pr.top - 2 || r.bottom > pr.bottom + 2)
        crossings.push({ panel: pn.dataset.panel, el: c.className || c.tagName, r: [r.left, r.top, r.right, r.bottom].map(Math.round), p: [pr.left, pr.top, pr.right, pr.bottom].map(Math.round) });
    });
  });
  // numbers with a final value and the local time their animation ends
  stage.querySelectorAll("[data-final]").forEach(e => { if (opacityOf(e) < 0.9) return; numbers.push({ final: parseFloat(e.dataset.final), done: parseFloat(e.dataset.done || "0"), text: e.textContent.trim() }); });
  // labels over drawn 3D geometry (pages that publish window.GEOM): share of a label's area covered by faces
  const geom = [];
  if (window.GEOM && window.GEOM.length) {
    const inside = (x, y, ps) => { let c = false; for (let i = 0, j = ps.length - 1; i < ps.length; j = i++) { const [xi, yi] = ps[i], [xj, yj] = ps[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
    const boxes = window.GEOM.map(ps => [ps, Math.min(...ps.map(p => p[0])), Math.max(...ps.map(p => p[0])), Math.min(...ps.map(p => p[1])), Math.max(...ps.map(p => p[1]))]);
    stage.querySelectorAll(".lbl, .h1, .badge, .chain").forEach(e => {
      if (opacityOf(e) < 0.3) return; const r = e.getBoundingClientRect(); if (r.width < 1) return;
      let n = 0, hit = 0;
      for (let y = r.top + 6; y < r.bottom; y += 12) for (let x = r.left + 6; x < r.right; x += 12) { n++;
        for (const [ps, x0, x1, y0, y1] of boxes) if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && inside(x, y, ps)) { hit++; break; } }
      if (n && hit / n > 0.02) geom.push({ el: (e.textContent || "").trim().slice(0, 30), covered: +(hit / n).toFixed(3) });
    });
  }
  const h1 = [...stage.querySelectorAll(".h1")].filter(e => opacityOf(e) > 0.5).map(e => e.textContent.trim());
  const prim = document.querySelector("[data-primary]"); const pr = prim ? prim.getBoundingClientRect() : null;
  const cap = document.getElementById("caption");
  return { texts, overflow, crossings: crossings.slice(0, 20), numbers, h1, geom, caption: cap && opacityOf(cap) > 0.5 ? cap.textContent : "",
           primary: pr ? { x0: pr.left, y0: pr.top, x1: pr.right, y1: pr.bottom } : null, W, H };
}
