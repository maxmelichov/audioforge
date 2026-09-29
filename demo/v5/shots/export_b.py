"""Numbers for the B shots (chart_icsi, chart_live, chart_turnbench, chart_wer, scorecard, end), read from runs/*.json
at build time (nothing typed by hand).

    python3 demo/v5/shots/export_b.py   ->  demo/v5/shots/numbers_b.js (window.NB = {...}) + numbers_b.json

Each entry: value (raw, in the unit named in scope), shown (the one string the film puts on screen, rounded once, half
away from zero), dec, scope, source (file) and path (keys joined by " > "). A missing path raises, so a stale number
cannot ship. Pages load numbers_b.js with a <script> tag (file:// safe) and put `shown` on screen; bar lengths use
`value`. The same quantity is exported once and reused by every page (charts and scorecard), so it cannot drift.

The script also checks every number in the shots_b.json narration captions ({caption|spoken}) against the shown
values, so a narration line cannot quote a number the pictures do not show.
"""
from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
ROOT = HERE.parents[2]
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")


def get(d, path):
    for k in path.split(" > "):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


_cache: dict = {}


def load(rel):
    if rel not in _cache:
        p = SSD / rel[len("ssd/"):] if rel.startswith("ssd/") else ROOT / rel
        _cache[rel] = json.loads(p.read_text())
    return _cache[rel]


def rnd(v, dec):
    """one rounding for the whole film: half away from zero on the decimal string"""
    return str(Decimal(str(v)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


OUT: dict = {}


def put(key, src, path, scope, dec=None, fn=lambda v: v):
    v = fn(get(load(src), path))
    OUT[key] = {"value": v, "scope": scope, "source": src, "path": path}
    if dec is not None:
        OUT[key].update(shown=rnd(v, dec), dec=dec)


def pct(v):            # fraction -> percent, exact decimal arithmetic
    return float(Decimal(str(v)) * 100)


def caught_pct(v):     # miss fraction -> % caught
    return float((1 - Decimal(str(v))) * 100)


def main():
    # ---- 1. ICSI held-out, thresholds frozen from AMI (eot-bench v2, 6 s horizon) ---------------------------------
    s = "runs/baselines_turn_icsi_dyn.json"
    base = "C_extended_windows_stream > frozen_ami"
    pol = "head_or_dyn_silero|head"
    for k, sysn in {"ours": pol, "silero": "silero_timeout"}.items():
        p = f"{base} > systems > {sysn} > 6s > icsi_at_ami_all"
        put(f"icsi/{k}/caught", s, p + " > miss_rate", "% of turn ends caught within the horizon (100 - miss rate)", 0, caught_pct)
        put(f"icsi/{k}/caught_ci", s, p + " > miss_rate_ci", "95 % bootstrap CI of % caught", None,
            lambda c: [rnd(caught_pct(c[1]), 1), rnd(caught_pct(c[0]), 1)])
        put(f"icsi/{k}/fc", s, p + " > fc_rate", "% of turns cut off too early (false cut-offs per turn)", 1, pct)
    pp = f"{base} > paired > {pol} - silero_timeout | 6s > all"
    put("icsi/diff_ci", s, pp + " > miss_diff_ci", "95 % paired bootstrap CI of the caught difference, points", None,
        lambda c: [rnd(-100 * c[1], 1), rnd(-100 * c[0], 1)])
    put("icsi/n", s, "n", "ICSI turn ends scored")
    put("icsi/meetings", s, "meetings", "held-out ICSI meetings", None, len)
    OUT["icsi/policy"] = {"value": pol, "scope": "audioforge policy name", "source": s, "path": f"{base} > systems > {pol}"}
    OUT["icsi/horizon_s"] = {"value": 6, "scope": "turn-end horizon, s", "source": s, "path": f"key '6s' in {base} > systems > {pol} > 6s"}
    assert "6s" in get(load(s), f"{base} > systems > {pol}")
    OUT["icsi/ami_n"] = {"value": int(re.search(r"all (\d+) AMI dev turns", get(load(s), "protocol")).group(1)),
                         "scope": "AMI turns the frozen thresholds were fitted on", "source": s, "path": "protocol (text: 'all N AMI dev turns')"}
    fo, fs = OUT["icsi/ours/fc"]["value"], OUT["icsi/silero/fc"]["value"]
    OUT["icsi/fc_ratio"] = {"value": round(fo / fs, 3), "scope": "audioforge false cut-offs / Silero timeout false cut-offs", "source": s,
                            "path": "icsi/ours/fc / icsi/silero/fc"}
    assert 0.45 <= fo / fs <= 0.55, "'half as often' no longer holds"
    co, cs = OUT["icsi/ours/caught"]["value"], OUT["icsi/silero/caught"]["value"]
    OUT["icsi/caught_ratio"] = {"value": co / cs, "shown": rnd(co / cs, 1), "dec": 1, "scope": "turn ends caught, audioforge / Silero timeout (x as many)",
                                "source": s, "path": "(1 - icsi/ours miss_rate) / (1 - icsi/silero miss_rate)"}

    # ---- 2. live user channel inside Pipecat and LiveKit (runs/e2e_final.json) ------------------------------------
    s = "runs/e2e_final.json"
    SYS = {"ours": "pipecat|C", "pipecat": "pipecat|A", "livekit": "livekit|B"}
    for st in ("oto", "turnbench"):
        for k, sy in SYS.items():
            p = f"table > {st}|user|{sy}"
            put(f"live/{st}/{k}/unanswered", s, p + " > missed_3s", "% of turns not answered within 3 s", 0, pct)
            put(f"live/{st}/{k}/first", s, p + " > first_text_ms_median", "s from speech onset to the first words on screen (median)", 1, lambda v: v / 1000)
            if k != "livekit":      # interruptions are compared with Pipecat's default only
                put(f"live/{st}/{k}/cutins", s, p + " > cut_ins_per_min", "times per minute the agent talks over the user", 1)
        put(f"live/{st}/clips", s, f"table > {st}|user|pipecat|A > n_clips", "recorded conversations")
        put(f"live/{st}/ends", s, f"table > {st}|user|pipecat|A > n_ends", "user turn ends")
        r = OUT[f"live/{st}/ours/cutins"]["value"] / OUT[f"live/{st}/pipecat/cutins"]["value"]
        OUT[f"live/{st}/cutins_ratio"] = {"value": round(r, 3), "scope": "audioforge / Pipecat default cut-ins per minute", "source": s,
                                          "path": f"live/{st}/ours/cutins / live/{st}/pipecat/cutins"}

    # "Interrupts nearly 40 % less": % fewer cut-ins per set, the smaller of the two, rounded to 10 for the headline
    less = {st: float((1 - Decimal(str(OUT[f"live/{st}/ours/cutins"]["value"])) / Decimal(str(OUT[f"live/{st}/pipecat/cutins"]["value"]))) * 100)
            for st in ("oto", "turnbench")}
    for st, v in less.items():
        OUT[f"live/{st}/cutins_less"] = {"value": v, "shown": rnd(v, 0), "dec": 0, "scope": "% fewer cut-ins per minute than Pipecat's default",
                                         "source": s, "path": f"1 - table > {st}|user|pipecat|C / {st}|user|pipecat|A > cut_ins_per_min"}
    lo = min(int(rnd(v, 0)) for v in less.values())
    r10 = int(rnd(lo / 10, 0)) * 10
    OUT["live/cutins_less"] = {"value": lo, "shown": str(r10), "dec": 0, "word": "nearly" if r10 > lo else "about",
                               "scope": "headline: the smaller per-set % fewer cut-ins, rounded to 10 ('nearly' if rounded up)", "source": s,
                               "path": "min(live/oto/cutins_less, live/turnbench/cutins_less), rounded to 10"}

    # ---- 3. TurnBench dev, official scorer, FP <= 0.10 ------------------------------------------------------------
    put("tb/ours", "runs/turnbench_latency.json", "families > fixed > crossfit_pooled_heldout > recall", "% of turn ends detected, held-out halves pooled", 0, pct)
    put("tb/ours/fp", "runs/turnbench_latency.json", "families > fixed > crossfit_pooled_heldout > fp_rate", "false-positive rate", 3)
    for k, sysn in {"silero": "silero_timeout", "eou": "eou_posterior"}.items():
        put(f"tb/{k}", "runs/turnbench_dev.json", f"systems > {sysn} > operating_point > score > recall", "% detected at its best dev point", 0, pct)
        put(f"tb/{k}/fp", "runs/turnbench_dev.json", f"systems > {sysn} > operating_point > score > fp_rate", "false-positive rate", 3)
    pub = "turnbench_published_dev_rescored > smart_turn_v3"
    put("tb/smartturn", "runs/dyadic_bench.json", pub + " > recall", "% detected, published predictions rescored", 0, pct)
    put("tb/smartturn/fp", "runs/dyadic_bench.json", pub + " > fp_rate", "false-positive rate", 3)
    put("tb/fp_budget", "runs/turnbench_dev.json", "fp_budget", "false-positive budget", 2)
    put("tb/conversations", "runs/turnbench_dev.json", "n_conversations", "TurnBench dev conversations")
    tl = load("runs/turnbench_latency.json")["families"]["fixed"]["crossfit_pooled_heldout"]
    OUT["tb/ends"] = {"value": tl["tp"] + tl["fn"], "scope": "turn ends scored (tp + fn)", "source": "runs/turnbench_latency.json",
                      "path": "families > fixed > crossfit_pooled_heldout > tp + fn"}
    best = max(("silero", "eou", "smartturn"), key=lambda k: OUT[f"tb/{k}"]["value"])
    OUT["tb/best"] = dict(OUT[f"tb/{best}"], scope=f"best framework baseline ({best})")

    # ---- 4. final transcript WER, AMI dev 200 segments, Whisper-normalised ----------------------------------------
    put("wer/ours", "runs/hybrid_asr.json", "english > results > tdt_v3/ami > wer_whisper_norm > wer", "% word errors, Parakeet-TDT 0.6B v3 per finished turn", 1, pct)
    put("wer/ours/ci", "runs/hybrid_asr.json", "english > results > tdt_v3/ami > wer_whisper_norm > ci95", "95 % CI", None, lambda c: [rnd(pct(x), 1) for x in c])
    for k, sysn in {"whisper_turbo": "whisper_turbo", "parakeet_ctc": "parakeet", "whisper_small": "whisper_small"}.items():
        put(f"wer/{k}", "runs/final_asr.json", f"results > {sysn}/ami > wer_whisper_norm > wer", "% word errors, offline", 1, pct)
        put(f"wer/{k}/ci", "runs/final_asr.json", f"results > {sysn}/ami > wer_whisper_norm > ci95", "95 % CI", None, lambda c: [rnd(pct(x), 1) for x in c])
    put("wer/n", "runs/hybrid_asr.json", "english > results > tdt_v3/ami > wer_whisper_norm > n", "AMI dev segments")

    # ---- 5. who is talking: the real AMI meeting of the room example (IS1008b, 1670 s) ----------------------------
    room = "ssd/demo_out/v5_data/data.json"
    words = get(load(room), "clips > ex_room > words")
    OUT["room/found"] = {"value": len({sp for _, ws in words for sp, _ in ws if sp is not None}), "scope": "people the recorded session labelled",
                         "source": room, "path": "clips > ex_room > words > * > * > speaker (distinct)"}
    put("room/ref", "ssd/scratch/diar/offline/n3_R4_tita_any.json", "items > ami_IS1008b_1670s > n_ref_speakers", "people in the meeting (AMI reference)")

    # ---- 6. voice activity, AMI dev windows, threshold 0.5 (streaming detectors) ------------------------------------
    sv = "runs/baselines_sd.json"
    put("vad/ours", sv, "ours > vad_sweep > 0.5 > f1", "F1 at 0.5, as %", 1, pct)
    put("vad/marblenet", sv, "vad > marblenet_frame_vad_v2 > sweep > 0.5 > f1", "F1 at 0.5, as %", 1, pct)
    put("vad/silero", sv, "vad > silero_v5.1.2 > sweep > 0.5 > f1", "F1 at 0.5, as %", 1, pct)
    put("vad/n", sv, "vad > n", "AMI dev windows")

    # ---- 7. target speaker: missed turn ends at 6 s, AMI / held-out ICSI, 5 s voice print (TS-VAD track -> turn head)
    base = "C_extended_windows_stream > {} > hybrid_stream_causal_dominant > 6s > fixed_5pct_turn_fc"
    for st, src, sub in (("ami", "runs/baselines_turn.json", "systems"), ("icsi", "runs/baselines_turn_icsi.json", "crossfit_icsi > systems")):
        pb = base.format(sub)
        put(f"ts/{st}/before", src, pb + " > miss_rate", "% missed turn ends, shipped hybrid (no voice print)", 0, pct)
        put(f"ts/{st}/before_fc", src, pb + " > fc_rate", "% false cut-offs, shipped hybrid", 1, pct)
        pa = f"turn_bench > {st} > systems > hybrid_tsvad_spk > 6s"
        put(f"ts/{st}/after", "runs/improve_115m.json", pa + " > miss_rate", "% missed turn ends with the 5 s voice print", 0, pct)
        put(f"ts/{st}/after_fc", "runs/improve_115m.json", pa + " > fc_rate", "% false cut-offs with the voice print", 1, pct)
        put(f"ts/{st}/n", "runs/improve_115m.json", f"turn_bench > {st} > n", "turn ends scored")

    # framework versions of the live runs (research/E2E_FINAL.md text)
    md = (ROOT / "research" / "E2E_FINAL.md").read_text()
    for k, pat in {"pipecat": r"Pipecat (\d+\.\d+(?:\.\d+)?)", "livekit": r"LiveKit Agents (\d+\.\d+(?:\.\d+)?)"}.items():
        m = re.search(pat, md)
        OUT[f"version/{k}"] = {"value": m.group(1), "shown": m.group(1), "scope": "framework version in the live runs",
                               "source": "research/E2E_FINAL.md", "path": f"text: '{pat}'"}

    (HERE / "numbers_b.json").write_text(json.dumps(OUT, indent=1))
    (HERE / "numbers_b.js").write_text("// generated by export_b.py from runs/*.json; do not edit\nwindow.NB = " + json.dumps(OUT) + ";\n")
    for k, v in OUT.items():
        print(f"{k:30s} {json.dumps(v.get('shown', v['value']))[:60]:14s} <- {v['source']} > {v['path']}")
    print("wrote", HERE / "numbers_b.js")
    verify_narration()


def verify_narration():
    shown = {v["shown"] for v in OUT.values() if "shown" in v} | {str(v["value"]) for v in OUT.values() if isinstance(v["value"], int)}
    spec = json.loads((V5 / "shots_b.json").read_text())
    bad = []
    for sh in spec["shots"]:
        for cap in re.findall(r"\{([^|}]*)\|", sh.get("narration", "")):
            for num in re.findall(r"\d+(?:\.\d+)?", cap):
                if num not in shown:
                    bad.append((sh["name"], cap))
    if bad:
        raise SystemExit(f"narration numbers not in numbers_b: {bad}")
    print("narration numbers: all match numbers_b")


if __name__ == "__main__":
    main()
