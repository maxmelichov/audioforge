"""The early end-of-turn hint (``turn_end_hint``, audioforge/server/turn_hint.py) measured on the stored per-frame dumps.

Read-only: the per-frame dumps of scripts/research/eot_latency.py (``EOT_DUMP``, default scratch/tswer_fix/eot_dump:
decision time, turn-head p, served VAD, TS-VAD P(user) / P(other) per 80 ms frame) and its reference turns
(``eot_latency.sessions``). No model runs: numpy plus the server's own ``TurnHintTracker`` / ``TurnHintLedger`` /
``VadHeadPolicy`` through ``turn_hint.replay``, so the numbers are what a served session emits on the same frames.

    .venv/bin/python scripts/research/turn_hint.py [--out runs/turn_hint.json]

Per hint threshold H (80 ms of served-VAD silence and p >= H), per corpus (calls = TurnBench + one-to-one user
channels, AMI = 200 dev turns), with the served ``vad_head`` turn_end (VadHeadPolicy defaults; the others path only
with an enrolled print, as served) unchanged:

* hint latency: the confirmed hint of each answered reference end (the hint the answering turn_end carries in
  ``hinted_at``, if it fired at or after the end - 80 ms), minus the reference end; p50 / p95 (ms);
* precision: of the hints inside a scored reference turn window [start, min(end + 6 s, next onset)), the % that
  fired at or after the end - 80 ms, i.e. the user did not resume inside the same reference turn; ``confirm_rate``:
  the % of those hints the server confirmed (turn_end) rather than cancelled;
* recall: the % of answered reference ends whose answering turn_end carries such a hint, strictly earlier;
* effective response latency with the LLM + TTS prep of 300 / 600 ms started at the hint: max(hint + prep,
  turn_end) - end (a hint-less or stale-hinted end: turn_end + prep - end), vs turn_end + prep - end; p50 / p95 and
  the mean saving (ms);
* wasted preps: cancelled hints per scored turn (each is an LLM call thrown away).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from audioforge.server.policies import VadHeadPolicy  # noqa: E402
from audioforge.server.turn_hint import TurnHintTracker, replay  # noqa: E402

HS = (0.7, 0.8, 0.9, 0.95, 0.99)
PREPS_MS = (300, 600)
SCOPES = {"calls": ("turnbench", "oto"), "ami": ("ami",)}


def _eot():
    spec = importlib.util.spec_from_file_location("eot_latency_ro", ROOT / "scripts" / "research" / "eot_latency.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_session(d: dict, H: float) -> dict:
    h = d["head"]
    enrolled = bool(d.get("has_print"))
    return replay(h["t"], h["p"], h["vad"], VadHeadPolicy(), h["pu"] if enrolled else None,
                  h["po"] if enrolled else None, tracker=TurnHintTracker(H))


def score(r: dict, ss: dict, tol: float, horizon: float) -> dict:
    """One session's replay vs its reference turns (module docstring)."""
    ends = r["turn_ends"]
    out = {"n": 0, "answered": 0, "intr": 0, "hint_at_end": 0, "hint_early": 0, "confirmed": 0, "cancelled": 0,
           "recall": 0, "stale": 0, "lat_hint": [], "lat_end": [], "pairs": []}
    for i, (s0, e1) in enumerate(ss["user_turns"]):
        if not ss["scored"][i]:
            continue
        out["n"] += 1
        hz = e1 + horizon if ss["next_onset"][i] is None else min(e1 + horizon, ss["next_onset"][i])
        if any(s0 <= t < e1 - tol for t, _, _ in ends):
            out["intr"] += 1
        for th, _, _, outcome in r["hints"]:
            if s0 <= th < hz:
                out["hint_at_end" if th >= e1 - tol else "hint_early"] += 1
                out["confirmed"] += outcome == "confirmed"
                out["cancelled"] += outcome == "cancelled"
        ans = [(t, ha) for t, _, ha in ends if e1 - tol <= t < hz]
        if not ans:
            continue
        te, ha = ans[0]
        out["answered"] += 1
        out["lat_end"].append(te - e1)
        valid = ha is not None and ha >= e1 - tol and ha < te
        if ha is not None and not valid:
            out["stale"] += 1  # confirmed a hint from inside the turn: the prepared reply misses the end
        if valid:
            out["recall"] += 1
            out["lat_hint"].append(ha - e1)
        out["pairs"].append((te - e1, (ha - e1) if valid else None))
    return out


def pool(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    ans = sum(r["answered"] for r in rows)
    hin = sum(r["hint_at_end"] + r["hint_early"] for r in rows)
    lh = np.concatenate([np.asarray(r["lat_hint"], float) for r in rows]) if rows else np.zeros(0)
    le = np.concatenate([np.asarray(r["lat_end"], float) for r in rows]) if rows else np.zeros(0)
    pairs = [p for r in rows for p in r["pairs"]]
    ms = lambda x: None if x is None else int(round(1000 * x))  # noqa: E731
    pct = lambda a, q: ms(float(np.percentile(a, q))) if len(a) else None  # noqa: E731
    res = {"n_turns": n, "n_answered": ans, "n_hints_in_turn_windows": hin,
           "hint_latency_ms_p50": pct(lh, 50), "hint_latency_ms_p95": pct(lh, 95),
           "turn_end_latency_ms_p50": pct(le, 50), "turn_end_latency_ms_p95": pct(le, 95),
           "precision_pct": round(100 * sum(r["hint_at_end"] for r in rows) / hin, 1) if hin else None,
           "confirm_rate_pct": round(100 * sum(r["confirmed"] for r in rows) / hin, 1) if hin else None,
           "recall_pct": round(100 * sum(r["recall"] for r in rows) / ans, 1) if ans else None,
           "stale_hinted_ends": sum(r["stale"] for r in rows),
           "wasted_preps_per_turn": round(sum(r["cancelled"] for r in rows) / n, 3) if n else None,
           "false_interruption_pct": round(100 * sum(r["intr"] for r in rows) / n, 1) if n else None,
           "missed_pct": round(100 * (n - ans) / n, 1) if n else None}
    for prep in PREPS_MS:
        pr = prep / 1000
        base = np.array([te + pr for te, _ in pairs])
        eff = np.array([max(h + pr, te) if h is not None else te + pr for te, h in pairs])
        res[f"prep{prep}"] = {"response_ms_p50_at_turn_end": pct(base, 50), "response_ms_p95_at_turn_end": pct(base, 95),
                              "response_ms_p50_at_hint": pct(eff, 50), "response_ms_p95_at_hint": pct(eff, 95),
                              "mean_saving_ms": ms(float(np.mean(base - eff))) if len(pairs) else None}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "turn_hint.json"))
    a = ap.parse_args(argv)
    logging.getLogger("audioforge.serve").setLevel(logging.WARNING)  # the per-hint outcome lines
    t0 = time.time()
    E = _eot()
    dump = E.load_dump()
    sess = {s["key"]: s for s in E.sessions() if s["key"] in dump}
    # the served vad_head replayed here must be the rule eot_latency scores (sim_room with its CHOSEN rule)
    agree = tot = 0
    for k in sess:
        d = dump[k]
        want = [round(t, 3) for t, _ in E.sim_room(d, E.CHOSEN)] if d.get("has_print") else None
        if want is None:
            continue
        got = [round(t, 3) for t, _, _ in run_session(d, 0.8)["turn_ends"]]
        tot += 1
        agree += got == want
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "dump": str(E.DUMP), "n_sessions": len(sess),
           "turn_end_rule": "VadHeadPolicy() defaults (served vad_head); others path only with an enrolled print",
           "vad_head_defaults": {k: getattr(VadHeadPolicy(), k) for k in ("thr", "k", "fb", "vad_thr", "others")},
           "hint_rule": "first frame with served-VAD silence (VAD < 0.4) >= 80 ms and p >= H; cancel on VAD > 0.5 x 2",
           "replay_equals_eot_latency_sim_room": f"{agree}/{tot} enrolled sessions",
           "tol_s": E.TOL, "horizon_s": E.HORIZON, "table": {}}
    for H in HS:
        per = {sc: [] for sc in SCOPES}
        for k, s in sess.items():
            sc = next((n for n, sets in SCOPES.items() if s["set"] in sets), None)
            if sc is None:
                continue
            per[sc].append(score(run_session(dump[k], H), s, E.TOL, E.HORIZON))
        res["table"][str(H)] = {sc: pool(rows) for sc, rows in per.items()}
        for sc in SCOPES:
            r = res["table"][str(H)][sc]
            print(f"H={H:<5} {sc:5s} hint p50/p95 {r['hint_latency_ms_p50']}/{r['hint_latency_ms_p95']} ms "
                  f"(turn_end {r['turn_end_latency_ms_p50']}/{r['turn_end_latency_ms_p95']}) "
                  f"prec {r['precision_pct']}% confirm {r['confirm_rate_pct']}% recall {r['recall_pct']}% "
                  f"waste/turn {r['wasted_preps_per_turn']} | 300: {r['prep300']['response_ms_p50_at_turn_end']}->"
                  f"{r['prep300']['response_ms_p50_at_hint']} p95 {r['prep300']['response_ms_p95_at_turn_end']}->"
                  f"{r['prep300']['response_ms_p95_at_hint']} | 600: {r['prep600']['response_ms_p50_at_turn_end']}->"
                  f"{r['prep600']['response_ms_p50_at_hint']} p95 {r['prep600']['response_ms_p95_at_turn_end']}->"
                  f"{r['prep600']['response_ms_p95_at_hint']}", flush=True)
    res["compute_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"replay == sim_room(CHOSEN): {res['replay_equals_eot_latency_sim_room']}; -> {a.out}")


if __name__ == "__main__":
    main()
