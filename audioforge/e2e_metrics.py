"""User-perceived turn-taking metrics for end-to-end voice-agent runs (research/E2E_FINAL.md).

Every system is reduced to one list of *response moments*: the audio times (s, clip clock) at which the agent would
start answering (Pipecat: the ``LLMContextFrame`` reaching the LLM; LiveKit: the committed user turn). The labels
(user floor turns) are used only here, after the run.

Definitions (one per clip; ``turns`` = the user's floor turns as sorted, non-overlapping (start, end) spans):

* **response to end e**: the first response moment tau with e <= tau < (the user's next turn start, else +inf).
  **dead air** = tau - e. **missed within h** = no response in [e, min(next start, e + h)).
* **cut-in**: a response moment inside a user turn span [start, end) (the user's turn is still going on, speech or a
  within-turn pause).
* **other**: response moments that are neither a first response nor a cut-in (repeat answers, answers to the other
  party only, answers before the user's first turn).
* ``interruptions_1s`` (AMI comparability with research/INTEGRATION.md section 4): moments followed by user speech
  (``user_intervals``) within 1 s.

Aggregation pools the scored ends of all clips; paired comparisons resample clips (bootstrap) and recompute the pooled
statistic of both systems on the same resampled clips.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

HORIZONS = (3.0, 6.0)


def _active(intervals, a: float, b: float) -> bool:
    return any(e > a + 1e-9 and s < b for s, e in intervals)


def score_clip(responses: Iterable[float], turns: Sequence[Sequence[float]], *, horizons=HORIZONS,
               user_intervals: Sequence[Sequence[float]] | None = None, scored: Sequence[bool] | None = None,
               resume_s: float = 1.0) -> dict:
    """Score one clip. ``scored[i]`` = whether turn i's end is scored (default all). Returns per-end rows and counts."""
    taus = sorted(float(t) for t in responses)
    turns = [(float(s), float(e)) for s, e in turns]
    assert all(turns[i][1] <= turns[i + 1][0] + 1e-9 for i in range(len(turns) - 1)), "turns must not overlap"
    scored = list(scored) if scored is not None else [True] * len(turns)
    hmax = max(horizons)
    ends, used = [], set()
    for i, (_s, e) in enumerate(turns):
        nxt = turns[i + 1][0] if i + 1 < len(turns) else math.inf
        tau = next((t for t in taus if e - 1e-9 <= t < nxt), None)
        if tau is not None:
            used.add(tau)
        if not scored[i]:
            continue
        row = {"end": round(e, 3), "next_start": None if nxt == math.inf else round(nxt, 3),
               "response": None if tau is None else round(tau, 3),
               "dead_air_ms": None if tau is None or tau - e > hmax + 1e-9 else round((tau - e) * 1000)}
        for h in horizons:
            row[f"missed_{h:g}s"] = not (tau is not None and tau < min(nxt, e + h))
        ends.append(row)
    cut = [t for t in taus if any(s - 1e-9 <= t < e - 1e-9 for s, e in turns)]
    other = [t for t in taus if t not in used and t not in cut]
    out = {"ends": ends, "n_responses": len(taus), "cut_ins": len(cut), "cut_in_t": [round(t, 3) for t in cut],
           "other": len(other)}
    if user_intervals is not None:
        out["interruptions_1s"] = sum(_active(user_intervals, t, t + resume_s) for t in taus)
    return out


def pooled(clips: Sequence[dict], horizons=HORIZONS) -> dict:
    """Pool scored clips (``score_clip`` outputs, optionally with ``audio_s``): dead-air median / P90 over ends answered
    within max(horizons), miss rates over all ends, cut-ins (total, per clip, per audio minute)."""
    ends = [r for c in clips for r in c["ends"]]
    da = np.array([r["dead_air_ms"] for r in ends if r["dead_air_ms"] is not None], float)
    out = {"n_clips": len(clips), "n_ends": len(ends), "n_answered": int(len(da)),
           "dead_air_ms_median": round(float(np.median(da))) if len(da) else None,
           "dead_air_ms_p90": round(float(np.percentile(da, 90))) if len(da) else None,
           "dead_air_ms_mean": round(float(da.mean())) if len(da) else None}
    for h in horizons:
        out[f"missed_{h:g}s"] = round(float(np.mean([r[f"missed_{h:g}s"] for r in ends])), 4) if ends else None
    ci = [c["cut_ins"] for c in clips]
    out["cut_ins"] = int(sum(ci))
    out["cut_ins_per_clip"] = round(float(np.mean(ci)), 3) if ci else None
    mins = sum(c.get("audio_s", 0.0) for c in clips) / 60
    out["cut_ins_per_min"] = round(sum(ci) / mins, 3) if mins > 0 else None
    out["other"] = int(sum(c["other"] for c in clips))
    if all("interruptions_1s" in c for c in clips) and clips:
        out["interruptions_1s"] = int(sum(c["interruptions_1s"] for c in clips))
    return out


STATS: dict[str, Callable[[Sequence[dict]], float | None]] = {
    "dead_air_ms_median": lambda cs: pooled(cs)["dead_air_ms_median"],
    "dead_air_ms_p90": lambda cs: pooled(cs)["dead_air_ms_p90"],
    "missed_3s": lambda cs: pooled(cs)["missed_3s"],
    "missed_6s": lambda cs: pooled(cs)["missed_6s"],
    "cut_ins_per_clip": lambda cs: pooled(cs)["cut_ins_per_clip"],
}


def paired_bootstrap(a: Sequence[dict], b: Sequence[dict], stat: str | Callable = "dead_air_ms_median",
                     n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Statistic(a) - statistic(b) on the same clips (a[i] and b[i] = one clip under two systems), with a percentile
    CI from resampling clips with replacement. Resamples where either side is undefined are skipped."""
    assert len(a) == len(b) and len(a) > 0, (len(a), len(b))
    f = STATS[stat] if isinstance(stat, str) else stat
    d0a, d0b = f(a), f(b)
    delta = None if d0a is None or d0b is None else d0a - d0b
    rng = np.random.default_rng(seed)
    n, ds = len(a), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        va, vb = f([a[i] for i in idx]), f([b[i] for i in idx])
        if va is not None and vb is not None:
            ds.append(va - vb)
    lo, hi = ((float(np.percentile(ds, 100 * (1 - ci) / 2)), float(np.percentile(ds, 100 * (1 + ci) / 2)))
              if ds else (None, None))
    r = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"stat": stat if isinstance(stat, str) else getattr(stat, "__name__", "custom"), "a": r(d0a), "b": r(d0b),
            "delta": r(delta), "ci": [r(lo), r(hi)], "n_clips": n, "n_boot_valid": len(ds)}


def usage(samples: Sequence[tuple[float, float, float]]) -> dict:
    """Process usage from (wall s, cpu s (user + sys, all threads), rss MB) samples: mean CPU % of one core over the
    span and the peak RSS."""
    if len(samples) < 2:
        return {"cpu_pct": None, "rss_mb_peak": max((s[2] for s in samples), default=None)}
    (w0, c0, _), (w1, c1, _) = samples[0], samples[-1]
    return {"cpu_pct": round(100 * (c1 - c0) / max(w1 - w0, 1e-9), 1), "rss_mb_peak": round(max(s[2] for s in samples), 1),
            "cpu_s": round(c1 - c0, 3), "wall_s": round(w1 - w0, 3)}
