"""Collect runs/dyadic_bench_<corpus>.json (+ runs/turnbench_dev.json) into runs/dyadic_bench.json and print the
markdown tables of research/archive/DYADIC.md.   .venv/bin/python scripts/research/summarize_dyadic.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPORA = ("behavior_sd", "dailytalk", "oto", "turnbench")
ORDER = ["timeout_primary_oracle", "timeout_1040ms_primary_oracle", "timeout_any_speaker_oracle", "silero_timeout",
         "silero_timeout_1000ms", "head_trail6_oracle_track", "hybrid_oracle", "head_trail6+silero_timeout",
         "eou_posterior", "eou_native (<EOU> emitted)", "timeout_stream_causal_dominant", "head_v3_stream_causal_dominant",
         "timeout_any_speaker_stream", "hybrid_stream_causal_dominant"]


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def ci(d):
    c = d.get("miss_rate_ci")
    return f"{pct(d['miss_rate'])} [{pct(c[0])}, {pct(c[1])}]" if c else pct(d["miss_rate"])


def p50(d):
    v = d.get("p50_ms")
    return "inf" if v is None or v == float("inf") else f"{v:.0f}"


def block_rows(res: dict, blk: str, section: str = "crossfit") -> list[str]:
    c = res[blk][section]
    rows = []
    for name in ORDER:
        r = c["systems"].get(name) or c.get("native", {}).get(name)
        if r is None:
            continue
        s6 = r["6s"].get("fixed_5pct_turn_fc", r["6s"])
        s2 = r["2s"].get("fixed_5pct_turn_fc", r["2s"])
        rows.append(f"| {name} | {pct(s6['fc_rate'])} | {ci(s6)} | {ci(s2)} | {p50(s6)} | "
                    f"{pct(s6['strata']['open']['miss_rate'])} ({s6['strata']['open']['n']}) | "
                    f"{pct(s6['strata']['taken']['miss_rate'])} ({s6['strata']['taken']['n']}) |")
    return rows


def frozen_rows(res: dict, blk: str) -> list[str]:
    f = res[blk].get("frozen_ami") or {}
    rows = []
    for name in ORDER:
        if name not in f:
            continue
        r6, r2 = f[name]["6s"]["at_ami_point"], f[name]["2s"]["at_ami_point"]
        rows.append(f"| {name} | {json.dumps(f[name]['6s']['ami_point'])} | {pct(r6['fc_rate'])} | {ci(r6)} | {ci(r2)} | {p50(r6)} | "
                    f"{pct(r6['strata']['open']['miss_rate'])} | {pct(r6['strata']['taken']['miss_rate'])} |")
    return rows


def paired_rows(res: dict, blk: str) -> list[str]:
    out = []
    for k, d in res[blk]["crossfit"]["paired"].items():
        if not k.endswith("| 6s"):
            continue
        a = d["all"]
        out.append(f"| {k[:-5]} | {100 * a['miss_diff']:+.1f} [{100 * a['miss_diff_ci'][0]:+.1f}, {100 * a['miss_diff_ci'][1]:+.1f}] | "
                   f"{100 * d['open']['miss_diff']:+.1f} | {100 * d['taken']['miss_diff']:+.1f} |")
    return out


def main():
    summary, md = {}, []
    for c in CORPORA:
        p = ROOT / "runs" / f"dyadic_bench_{c}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        hdr = ("| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |\n"
               "|---|---|---|---|---|---|---|")
        md.append(f"### {c}: {res['n']} {res.get('roles_scored', 'human')}-party turn ends, {res['conversations']} conversations "
                  f"(post-end frames p50 {res['post_avail']['p50']}, end reasons {res['end_reasons']})\n")
        md.append("Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):\n\n" + hdr)
        md += block_rows(res, "C_extended_windows")
        md.append("\nPaired differences at 6 s (miss points, 95 % CI; open / taken point estimates):\n\n| a - b | all | open | taken |\n|---|---|---|---|")
        md += paired_rows(res, "C_extended_windows")
        if res.get("frozen_ami_available"):
            md.append("\nFrozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):\n\n"
                      "| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |\n|---|---|---|---|---|---|---|---|")
            md += frozen_rows(res, "C_extended_windows")
        if "C_stream_subset" in res:
            sub = res["C_stream_subset"]
            md.append(f"\nStreaming Sortformer subset (n = {sub['n']}):\n\n" + hdr)
            md += block_rows({"X": {"crossfit": sub}}, "X")
        md.append("")
        summary[c] = {"n": res["n"], "conversations": res["conversations"], "block_C_6s": {
            name: {k: (r["6s"].get("fixed_5pct_turn_fc", r["6s"]))[k] for k in ("fc_rate", "miss_rate", "miss_rate_ci", "p50_ms")}
            for name, r in list(res["C_extended_windows"]["crossfit"]["systems"].items()) + list(res["C_extended_windows"]["crossfit"]["native"].items())},
            "frozen_ami_6s": {name: {k: f["6s"]["at_ami_point"][k] for k in ("fc_rate", "miss_rate", "miss_rate_ci", "p50_ms")}
                              for name, f in (res["C_extended_windows"].get("frozen_ami") or {}).items()}}
    tb = ROOT / "runs" / "turnbench_dev.json"
    if tb.exists():
        t = json.loads(tb.read_text())
        md.append("### TurnBench dev (official scorer, EOT task, dev operating point = highest recall at fp_rate <= 0.10)\n")
        md.append("| system | operating point | recall | fp_rate | latency p10 / p50 / p90 ms |\n|---|---|---|---|---|")
        for name, r in t["systems"].items():
            b = r["operating_point"]
            if b is None:
                md.append(f"| {name} | none within budget | - | - | - |")
                continue
            s, l = b["score"], b["score"]["latency_ms"]
            md.append(f"| {name} | {b['point']} | {s['recall']:.3f} | {s['fp_rate']:.3f} | {l['p10']} / {l['p50']} / {l['p90']} |")
        md.append("\nPublished dev predictions re-scored with the same vendored scorer (their committed operating points):\n\n"
                  "| baseline | recall | fp_rate | p50 ms | published test recall / fp / p50 |\n|---|---|---|---|---|")
        for name, s in t["published_dev_rescored"].items():
            pt = t["published_test"].get(name)
            md.append(f"| {name} | {s['recall']:.3f} | {s['fp_rate']:.3f} | {s['latency_ms']['p50']} | "
                      + (f"{pt[0]} / {pt[1]} / {pt[2]}" if pt else "-") + " |")
        summary["turnbench_dev"] = {name: (r["operating_point"] or {}).get("score") for name, r in t["systems"].items()}
        summary["turnbench_published_dev_rescored"] = t["published_dev_rescored"]
    (ROOT / "runs" / "dyadic_bench.json").write_text(json.dumps(summary, indent=1, default=float))
    text = "\n".join(md)
    print(text)
    if "--write" in sys.argv:
        (ROOT / "runs" / "dyadic_tables.md").write_text(text)


if __name__ == "__main__":
    main()
