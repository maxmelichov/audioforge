"""The README / SINGLE_MODEL.md "live_69" table on this box: every system's pooled live metrics over its sessions,
scored exactly as scripts/research/single_model.py._live does (e2e_final.score_record + group_metrics on the rebuilt
37-clip set), next to the published row of runs/single_model.json table.live_69.

    PYTHONPATH=. python scripts/research/gpu_run_5090/live_table.py --data <dir> --out live.json

<dir> holds the e2e work dirs of research/GPU_RUN_2026-09-29.md: e2e/ (clips.json, the cpu runs of S, A and B),
e2e_cpu_loaded/ (CN on cpu, overloaded), e2e_cuda/ (S with --device cuda) and e2e_cn_cuda/ (CN with --device cuda).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import e2e_final as E2E  # noqa: E402

D = Path(".")  # --data
KEYS = ("n_clips", "missed_3s", "missed_6s", "cut_ins_per_session", "dead_air_ms_median", "first_text_ms_median", "wer",
        "server_rtf_median", "server_rtf_max", "rss_mb_server_peak", "n_flagged_rtf_gt_1", "load1_start_range")
# ours: (work dir, framework, system) ; published: the key in runs/single_model.json table.live_69.systems
ROWS = {
    "single_S (A1 rule, --mode single, live)": [("cpu", "e2e", "pipecat", "S"), ("cuda", "e2e_cuda", "pipecat", "S")],
    # CN on 2 cpu threads ran next to three other queues first and is overloaded at RTF 1.04 on its own (every session
    # flagged): kept as the record of that, e2e_cpu_loaded/ (the note, section 4)
    "default_CN (Nemotron-3, timeout)": [("cpu", "e2e_cpu_loaded", "pipecat", "CN"), ("cuda", "e2e_cn_cuda", "pipecat", "CN")],
    "pipecat_default_A": [("cpu", "e2e", "pipecat", "A")],
    "livekit_default_B": [("cpu", "e2e", "livekit", "B")],
    "default_CN_livekit": [("cpu", "e2e_cpu_loaded", "livekit", "CN"), ("cuda", "e2e_cn_cuda", "livekit", "CN")],
}


def ours(work: Path, fw: str, sy: str) -> dict:
    refs = {r["name"]: r for r in json.loads((D / "e2e" / "clips.json").read_text())}
    recs = [r for r in E2E.load_records(work) if r["framework"] == fw and r["system"] == sy]
    if not recs:
        return {}
    items = [(r, E2E.score_record(r, refs[r["clip"]])) for r in recs]
    m = E2E.group_metrics(items)
    m["cut_ins_per_session"] = round(m["cut_ins"] / len(items), 3)
    m["n_sessions"] = len(items)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, help="parent of the e2e work dirs")
    a = ap.parse_args()
    global D
    D = Path(a.data)
    pub = json.loads((ROOT / "runs" / "single_model.json").read_text())["table"]["live_69"]["systems"]
    res = {}
    for name, variants in ROWS.items():
        row = {"published_m5_cpu": pub.get(name)}
        for dev, wd, fw, sy in variants:
            m = ours(D / wd, fw, sy)
            row[f"5090box_{dev}"] = ({k: m.get(k) for k in KEYS + ("n_sessions",)} | {"complete": m["n_sessions"] == 69}
                                     if m else None)
        res[name] = row
        print(name)
        for k, v in row.items():
            if v:
                print(f"  {k:18s} n={v.get('n_sessions', v.get('n_clips'))} miss3 {v.get('missed_3s')} miss6 {v.get('missed_6s')} "
                      f"cut/sess {v.get('cut_ins_per_session')} dead {v.get('dead_air_ms_median')} first {v.get('first_text_ms_median')} "
                      f"wer {v.get('wer')} rtf {v.get('server_rtf_median')} rss {v.get('rss_mb_server_peak')} "
                      f"flagged {v.get('n_flagged_rtf_gt_1')}")
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
