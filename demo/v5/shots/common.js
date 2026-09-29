// v5 shared runtime. Every shot exposes init(data, params, W, H) and render(t, ctx); the frame is a pure function of t.
const U = {
  clamp: (x, a = 0, b = 1) => Math.min(b, Math.max(a, x)),
  lerp: (a, b, x) => a + (b - a) * x,
  ease: x => { x = Math.min(1, Math.max(0, x)); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; },
  eout: x => { x = Math.min(1, Math.max(0, x)); return 1 - Math.pow(1 - x, 3); },
  seg: (t, a, d = 0.3) => Math.min(1, Math.max(0, (t - a) / d)),
  el(tag, cls, parent, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; if (parent) parent.appendChild(e); return e; },
  css(e, o) { for (const k in o) e.style[k] = typeof o[k] === "number" && !/opacity|zIndex|flex|fontWeight/.test(k) ? o[k] + "px" : o[k]; return e; },
  textAt(points, now) { let s = ""; for (const p of points) { if (p[0] <= now) s = p[1]; else break; } return s; },
  // time each word index first appears in a change-point list
  wordTimes(points) { const out = []; for (const [t, s] of points) { const n = s ? s.split(" ").length : 0; while (out.length < n) out.push(t); } return out; },
  env(clip, t) { const i = Math.floor(t / clip.hop); return i >= 0 && i < clip.env.length ? clip.env[i] : 0; },
};

const S = { W: 1920, H: 1080, sq: false, M: 48 };

function stage(W, H) {
  if (window.PARAMS && window.PARAMS.fine != null) window.FINE = window.PARAMS.fine;
  S.W = W; S.H = H; S.sq = H / W > 0.9; S.M = 64;   // safe margin for every text element and panel
  const st = U.el("div", null, document.body); st.id = "stage"; U.css(st, { width: W, height: H });
  const lt = U.el("div", null, st); lt.id = "light";
  const ct = U.el("div", null, st); ct.id = "content";
  const cap = U.el("div", null, st); cap.id = "caption";
  const fine = U.el("div", null, st); fine.id = "fine";
  // fine print: measured at its real width (1-2 lines), never truncated; the content area ends above it
  U.css(fine, { left: S.M, right: S.M, bottom: S.M });
  fine.textContent = window.FINE || "";
  const fineH = Math.max(36, Math.ceil(fine.scrollHeight));
  S.bottom = H - S.M - fineH - 12 - 76 - 18;   // content ends above the caption bar + fine print
  if (window.PARAMS && window.PARAMS.subs) {    // example shots: a subtitle band with the speaker's words above the caption bar
    const sb = U.el("div", null, st); sb.id = "subs"; S.bottom -= 84;
    U.css(sb, { left: S.M, right: S.M, bottom: S.M + fineH + 12 + 76 + 14, height: 70 });
  }
  U.css(fine, { height: fineH });
  U.css(cap, { bottom: S.M + fineH + 12, maxWidth: W - 2 * S.M });
  return ct;
}

function chrome(t, ctx) {
  const lt = document.getElementById("light");
  // static, near-invisible vignette (no drifting light in the new palette)
  const cap = document.getElementById("caption");
  const c = ctx.caption || "";
  cap.textContent = c; cap.style.opacity = c ? 1 : 0;
  document.getElementById("fine").textContent = ctx.fine || "";
  const ct = document.getElementById("content");
  const push = ctx.push == null ? 0.015 : ctx.push;   // slow push-in, bounded so text stays >= 48 px from the edge
  let k = 1 + push * U.clamp(t / Math.max(1, ctx.dur || 8));
  // shots with idle stretches (examples waiting for their clip, the flat architecture between beats): a slow 4 s
  // breathing scale of at most 0.8 % keeps the frame alive without moving text within 48 px of the edge
  // idle breathing: a triangle wave (constant speed, no stall at the turning points), push-in off while breathing
  if (ctx.breathe) { const ph = (t % 4) / 4; k = 1 + 0.012 * (ph < 0.5 ? 2 * ph : 2 - 2 * ph); }
  k = Math.min(k, 1.018);   // push-in + breathe never exceed 1.8 %: text stays >= 48 px from the edge (64 px margin)
  ct.style.transform = `scale(${k.toFixed(5)})`;
}

// live waveform: n bars over the last `win` seconds of the clip envelope
function makeWave(parent, n, box, color) {
  const w = U.el("div", "wave", parent); U.css(w, box);
  const bars = []; const bw = box.width / n * 0.56;
  for (let i = 0; i < n; i++) { const b = U.el("i", null, w); U.css(b, { width: bw, height: 8, background: color }); bars.push(b); }
  return { w, bars, box, n };
}
function drawWave(wv, clip, now, active, win = 2.4) {
  for (let i = 0; i < wv.n; i++) {
    const tt = now - win + win * (i + 0.5) / wv.n;
    let v = U.env(clip, tt); v = Math.pow(v, 0.8);
    if (!active) v *= 0.25;
    wv.bars[i].style.height = Math.max(10, v * wv.box.height * 0.95) + "px";
    wv.bars[i].style.opacity = active ? 1 : 0.55;
  }
}

// words appear at word granularity with a 150 ms rise; `color` for the text
function drawWords(box, points, now, stagger = 0) {
  const s = U.textAt(points, now);
  const words = s ? s.split(" ") : [];
  const times = U.wordTimes(points);
  let html = "";
  words.forEach((w, i) => {
    const a = U.eout((now - (times[i] || 0) - (stagger ? (i - firstOfBlock(times, i)) * stagger : 0)) / 0.15);
    html += `<span class="w" style="opacity:${a.toFixed(3)};display:inline-block;transform:translateY(${((1 - a) * 12).toFixed(1)}px)">${w}</span> `;
  });
  box.innerHTML = `<p>${html}</p>`;
  // keep the newest lines: drop leading words while the text overflows the box
  if (box.scrollHeight > box.clientHeight + 1) {
    const sp = box.querySelectorAll("p > span"); let k = 0;
    while (k < sp.length - 1 && box.scrollHeight > box.clientHeight + 1) { sp[k].remove(); k++; }
    const first = box.querySelector("p > span"); if (first) first.textContent = "… " + first.textContent;
    const sp2 = box.querySelectorAll("p > span"); let j = 0;
    while (j < sp2.length - 1 && box.scrollHeight > box.clientHeight + 1) { sp2[j].remove(); j++; }
  }
  return words.length;
}
function firstOfBlock(times, i) { let j = i; while (j > 0 && times[j - 1] === times[i]) j--; return j; }

// live subtitles of the speaker's words (reference transcript timings): the words of the last few seconds, the current
// one highlighted; `words` = [[t0, t1, word, speaker]] in clip time
function drawSubs(words, at, playing, label) {
  const el = document.getElementById("subs"); if (!el) return;
  if (!playing || !words) { el.style.opacity = 0; return; }
  const past = words.filter(w => w[0] <= at && w[0] >= at - 4.0);
  const cur = words.filter(w => w[0] <= at && at < w[1] + 0.15);
  const SPC = { user: "#FFB84D", A: "#48D6FF", B: "#FFB84D", C: "#C4BFF0", D: "#7CF0B8" };
  let html = "", prevLab = null;
  past.slice(-9).forEach(w => {
    const spk = w[3] || "user", lab = label(spk);
    if (lab !== prevLab) { html += `<b style="background:#FFB84D">${lab}</b> `; prevLab = lab; }
    html += `<span style="color:${cur.includes(w) ? "#FFB84D" : "#F7F8FC"}">${w[2]}</span> `;
  });
  el.innerHTML = html || "&nbsp;"; el.style.opacity = 1;
  while (el.scrollWidth > el.clientWidth + 1 && el.querySelectorAll("span").length > 1) { const f = el.querySelector("span"); f.remove(); }
}
window.drawSubs = drawSubs;
window.U = U; window.S = S; window.stage = stage; window.chrome = chrome; window.makeWave = makeWave; window.drawWave = drawWave; window.drawWords = drawWords;
