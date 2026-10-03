# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Audit: every published number (README.md, research/FINAL_COMPARE.md headline tables, the numbers the
compare_*.png images read through demo/images/redesign/arch_vs.html -> numbers_final.json) against
runs/final_compare.json, runs/dual_rate.json and runs/turn_data.json, after rounding to the printed decimals.

    uv run plans/audit/fairness_001.py          # or: python3 plans/audit/fairness_001.py
    uv run plans/audit/fairness_001.py -v       # also print every OK cell

Read-only. Sections: [NF] numbers_final.json vs its source path, [IMG] image keys, [TAB] markdown table cells,
[TXT] prose claims, [STALE] stale numbers / head versions / images in public docs, [PROV] provenance (afm files).
Exit code 1 if any MISMATCH / NOSRC / STALE line is printed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERBOSE = "-v" in sys.argv
J = {n: json.loads((ROOT / f"runs/{n}.json").read_text()) for n in ("final_compare", "dual_rate", "turn_data")}
NF = json.loads((ROOT / "demo/images/redesign/numbers_final.json").read_text())
COUNT = {"OK": 0, "MISMATCH": 0, "NOSRC": 0, "STALE": 0, "NOTE": 0}


def out(kind, msg):
    COUNT[kind] = COUNT.get(kind, 0) + 1
    if kind != "OK" or VERBOSE:
        print(f"{kind:8s} {msg}")


def rnd(x, dec):
    return Decimal(str(x)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)


def get(path):
    """'fc:a > b > 0' (fc / dr / td = final_compare / dual_rate / turn_data); ' @x0.001' suffix scales."""
    scale = 1.0
    if " @x" in path:
        path, s = path.split(" @x")
        scale = float(s)
    src, p = path.split(":", 1)
    d = J[{"fc": "final_compare", "dr": "dual_rate", "td": "turn_data"}[src]]
    for k in p.split(" > "):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return None if d is None else float(d) * scale


NUM = re.compile(r"(?<![\w.\-+])[-+]?\d+(?:\.\d+)?(?![\w])")


def nums(text):
    t = text.replace("−", "-").replace("**", "").replace("*", "")
    t = re.sub(r"\[([^\]]*)\]", lambda m: " " + m.group(1).replace(",", " ") + " ", t)
    return NUM.findall(t)


def check(where, shown, path):
    if path is None:
        return
    if path == "NOSRC":
        out("NOSRC", f"{where}: {shown} has no source in runs/final_compare|dual_rate|turn_data.json")
        return
    try:
        v = get(path)
    except (KeyError, IndexError, TypeError, ValueError):
        out("NOSRC", f"{where}: {shown} -> path missing: {path}")
        return
    dec = len(shown.split(".")[1]) if "." in shown else 0
    want = rnd(v, dec)
    if Decimal(shown.lstrip("+")) == want:
        out("OK", f"{where}: {shown} == {path} ({v})")
    else:
        out("MISMATCH", f"{where}: shown {shown}, json {path} = {v} -> {want}")


# ---------------------------------------------------------------- [NF] + [IMG]
def check_nf():
    for k, e in NF.items():
        src = {"runs/final_compare.json": "fc", "runs/dual_rate.json": "dr"}[e["source"]]
        try:
            v = get(f"{src}:{e['path']}")
        except (KeyError, IndexError, TypeError, ValueError):
            v = None
        if v is None:
            if e["value"] is not None:
                out("MISMATCH", f"[NF] {k}: value {e['value']} but {e['path']} missing")
            continue
        if e["value"] is None or abs(round(v, 6) - e["value"]) > 1e-9 or e["shown"] != str(rnd(v, e["dec"])):
            out("MISMATCH", f"[NF] {k}: numbers_final {e['value']} / '{e['shown']}' vs json {v}")
        else:
            out("OK", f"[NF] {k}")
    html = (ROOT / "demo/images/redesign/arch_vs.html").read_text()
    vs = html[html.index("const VS = {"):html.index("function barchart")]
    keys = sorted(set(re.findall(r'"(final/[^"]+)"', vs)) | {"final/" + k for k in re.findall(r'F\+"([^"]+)"', vs)})
    missing = [k for k in keys if k not in NF]
    for k in missing:
        out("MISMATCH", f"[IMG] arch_vs.html reads {k}: not in numbers_final.json (bar would say 'still running')")
    nulls = [k for k in keys if k in NF and NF[k]["value"] is None]
    for k in nulls:
        out("MISMATCH", f"[IMG] {k}: null in numbers_final.json (dashed 'still running' bar)")
    out("OK", f"[IMG] {len(keys)} image keys, {len(missing)} missing, {len(nulls)} null")
    # which words each tWER bar is on (pyannote31 / nofilter keys are the 115M-words rows)
    for c in ("icsi", "ami"):
        words = {k: NF[f"final/twer/{c}/{k}"]["path"].split(" > ")[-2] for k in
                 ("ours_0p6b", "pyannote31", "nemotron3_0p6bwords", "oracle_0p6bwords", "nofilter")
                 if f"final/twer/{c}/{k}" in NF and f"final/twer/{c}/{k}" in keys}
        if len({w.rsplit("_", 1)[-1] for w in words.values()}) > 1:
            out("NOTE", f"[IMG] compare_spk tWER {c}: bars mix 115M and 0.6B words: {words} "
                f"(pyannote on 0.6B words would be {get(f'fc:speaker_test > {c if c == 'icsi' else 'ami_eval'} > twer > pyannote31_best_0p6b > twer')})")
    # the words image shows the opt-in 1.12 s final for ours, the README table next to it the 160 ms pass
    for s in ("live", "ami", "icsi") if any(k.startswith("final/dualrate/wer/") for k in keys) else ():   # only while the image reads the 1.12 s keys
        a = NF.get(f"final/dualrate/wer/{s}/0p6b/1120", {}).get("shown")
        b = NF.get(f"final/wer/{s}/ours_0p6b", {}).get("shown")
        out("NOTE", f"[IMG] compare_asr {s}: ours 0.6B bar = 1.12 s final {a}; README/FINAL_COMPARE 160 ms value {b}")
    # staleness: for each committed compare_*.png, did any key it reads change in numbers_final.json since that commit?
    for png in sorted((ROOT / "demo/images").glob("compare_*.png")):
        job = png.stem.split("_", 1)[1]
        blk = re.split(r"\n  \w+: \{", vs.split(f"\n  {job}: {{", 1)[1])[0]
        jk = [k for k in keys if f'"{k[len("final/"):]}"' in blk or f'"{k}"' in blk]
        rev = subprocess.run(["git", "log", "-1", "--format=%h", "--", str(png)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(png)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        try:
            old = json.loads(subprocess.run(["git", "show", f"{rev}:demo/images/redesign/numbers_final.json"], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            out("NOTE", f"[IMG] {png.name}: cannot read numbers_final.json at {rev}")
            continue
        changed = [(k, old.get(k, {}).get("value"), NF[k]["value"]) for k in jk if old.get(k, {}).get("value") != NF.get(k, {}).get("value")]
        if changed:
            out("STALE", f"[IMG] {png.name} (committed {rev}{', dirty' if dirty else ''}) reads {len(jk)} keys; changed since: {changed}")
        else:
            out("OK", f"[IMG] {png.name} (committed {rev}): its {len(jk)} keys unchanged since the render")


# ---------------------------------------------------------------- [TAB]
def tables(path):
    """yield (header_line_no, header_cells, [(line_no, cells)])"""
    lines = (ROOT / path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and re.match(r"^\|[-:| ]+\|$", lines[i + 1]):
            hdr = [c.strip() for c in lines[i].strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append((j + 1, [c.strip() for c in lines[j].strip("|").split("|")]))
                j += 1
            yield i + 1, hdr, rows
            i = j
        else:
            i += 1


def W(st, sysk, sub="all"):
    return f"fc:words > {st} > whisper_norm|{sub} > {sysk} > wer_pct"


def WCI(st, sysk, sub="all"):
    p = f"fc:words > {st} > whisper_norm|{sub} > {sysk}"
    return [p + " > wer_pct", p + " > ci95 > 0", p + " > ci95 > 1"]


def CI(base, pt, ci):
    return [f"{base} > {pt}", f"{base} > {ci} > 0", f"{base} > {ci} > 1"]


ASR = {"audioforge 0.6B": "core_0p6b", "ours 0.6B": "core_0p6b", "audioforge 115M": "core_115m", "ours 115M": "core_115m",
       "audioforge 115M `--beam 8` (option)": "core_115m_beam8", "NVIDIA Parakeet-TDT 0.6B v3": "tdt_v3",
       "Parakeet-TDT 0.6B v3": "tdt_v3", "Whisper large-v3": "whisper_large_v3", "Whisper large-v3-turbo": "whisper_turbo",
       "Whisper small (LiveKit default STT)": "whisper_small", "Whisper small (LiveKit default)": "whisper_small",
       "Whisper small, beam 5": "whisper_small_b5", "Whisper small, beam 5 (faster-whisper defaults)": "whisper_small_b5",
       "Whisper small, beam 1": "whisper_small"}
TURN = {"audioforge 115M `assistant`": ("ours_115m", "assistant"), "audioforge 115M `balanced` (default)": ("ours_115m", "balanced"),
        "audioforge 115M `fast`": ("ours_115m", "fast"), "audioforge 0.6B `assistant`": ("ours_0p6b", "assistant"),
        "audioforge 0.6B `balanced` (default)": ("ours_0p6b", "balanced"), "audioforge 0.6B `fast`": ("ours_0p6b", "fast"),
        "audioforge 115M `balanced`": ("ours_115m", "balanced"), "audioforge 0.6B `balanced`": ("ours_0p6b", "balanced"),
        "audioforge 0.6B `balanced` ⚑": ("ours_0p6b", "balanced"), "audioforge 0.6B `fast` ⚑": ("ours_0p6b", "fast"),
        "audioforge 115M `balanced` (held-out pick = shipped)": ("ours_115m", "balanced"),
        "audioforge 115M `fast` (held-out pick = shipped)": ("ours_115m", "fast"),
        "Pipecat smart-turn v3.2 + Silero (defaults)": ("pipecat_smartturn_v3.2_silero", None),
        "Pipecat smart-turn v3.2 + Silero": ("pipecat_smartturn_v3.2_silero", None),
        "LiveKit Agents 1.8 EnglishModel + Silero": ("livekit_en_turn_detector_silero", None),
        "LiveKit Agents 1.8 + Silero": ("livekit_en_turn_detector_silero", None),
        "LiveKit turn detector + Silero": ("livekit_en_turn_detector_silero", None),
        "NVIDIA Parakeet-Realtime-EOU 120M": ("parakeet_realtime_eou", None), "NVIDIA Parakeet-Realtime-EOU": ("parakeet_realtime_eou", None),
        "ours 115M": ("ours_115m", "assistant"), "ours 0.6B": ("ours_0p6b", "assistant"),
        "smart-turn v3.2 classifier alone (upper bound: one call per whole clip, no timing)": ("smartturn_classifier_alone", None)}


def tb(sys_, preset, ds):
    return f"fc:turn > {sys_}" + (f" > {preset}" if preset else "") + f" > {ds}"


RES = ("README.md", "docs/RESULTS.md")  # the README results moved to docs/RESULTS.md (docs split, 2026-10-03)


def spec_for(file, hdr):
    """return fn(row_label) -> list of per-column path lists (None = skip column), or None if table not audited."""
    h = " | ".join(hdr)
    if file in RES and h.startswith("system | LibriSpeech test-clean"):
        return lambda r: (lambda s: [[W("ls_clean", s)], [W("ls_other", s)], [W("ami_eval", s)], [W("icsi_eval", s)],
                                     [W("live", s, "user_channel")]])(ASR[r])
    if file in RES and h.startswith('system | right about "done"'):
        def f(r):
            if r.startswith("ours 115M, held-out pick"):
                b = "fc:turn_clean > test > 115m > assistant > clean_servable > smartturn_test_w2.5"
                return [[b + " > accuracy_pct"], [b + " > p50"], [b + " > false_fire_pct"]]
            b = tb(*TURN[r], "asst")
            return [[b + "_windows > 2.5 > accuracy_pct"], [b + " > p50"], [b + "_windows > 2.5 > false_fire_pct"]]
        return f
    if file in RES and h.startswith("system | device | p50 | WER of the timed texts"):
        K = {"ours 115M": "ours_115m_1120", "ours 0.6B": "ours_0p6b_1120", "Parakeet-TDT 0.6B v3 (offline)": "parakeet_tdt_mps",
             "Whisper large-v3-turbo": "whisper_turbo", "Whisper small, beam 5": "whisper_small_b5"}
        def f(r):
            b = f"fc:final_latency > systems > {K[r]}"
            return [None, [b + " > p50"], [b + " > wer_timed_texts > wer_pct"]]
        return f
    if file in RES and h.startswith(" | 115M (default)"):
        return "STALE_CORE"
    if h.startswith("system | LibriSpeech test-clean (300)"):
        return lambda r: None if r not in ASR else (lambda s: [WCI("ls_clean", s), WCI("ls_other", s), WCI("ami_eval", s),
                         WCI("icsi_eval", s), WCI("live", s), WCI("live", s, "user_channel"), WCI("fleurs_en", s), None])(ASR[r])
    if h.startswith("baseline − 0.6B"):
        def f(r):
            s = ASR.get({"Parakeet-TDT v3": "NVIDIA Parakeet-TDT 0.6B v3", "Whisper small": "Whisper small (LiveKit default STT)"}.get(r, r))
            return [CI(f"fc:words > {st} > whisper_norm|all > {s} - core_0p6b", "delta_pp", "ci95")
                    for st in ("ls_clean", "ls_other", "ami_eval", "icsi_eval", "live", "fleurs_en")]
        return f
    if h.startswith("system | device | p50 ms"):
        K = {"audioforge 115M, `--final-chunk-ms 1120`": "ours_115m_1120", "audioforge 0.6B, `--final-chunk-ms 1120`": "ours_0p6b_1120",
             "NVIDIA Parakeet-TDT 0.6B v3 (offline)": "parakeet_tdt", "NVIDIA Parakeet-TDT 0.6B v3 (offline) @ MPS, fp32": "parakeet_tdt_mps",
             "Whisper large-v3-turbo": "whisper_turbo", "Whisper large-v3": "whisper_large", "Whisper small (LiveKit default STT)": "whisper_small",
             "Whisper small, beam 5 (faster-whisper defaults)": "whisper_small_b5", "Whisper small, beam 1": "whisper_small",
             "Whisper small, beam 5 (transformers)": "whisper_small_mps"}
        def f(r):
            b = f"fc:final_latency > systems > {K.get(r, K.get(r.split(' @ ')[0]))}"
            return [None, CI(b, "p50", "p50_ci95"), CI(b, "p95", "p95_ci95"), [b + " > by_length_p50 > lt2s > p50"],
                    [b + " > by_length_p50 > 2to5s > p50"], [b + " > by_length_p50 > gt5s > p50"],
                    CI(b + " > wer_timed_texts", "wer_pct", "ci95")]
        return f
    if h.startswith("system | AMI test F1 @0.5"):
        K = {"audioforge 115M speech head (v0.4)": "core_115m", "audioforge 0.6B speech head (v0.3)": "core_0p6b",
             "audioforge 115M speech head (v0.4; ICSI: retrained, speakers unseen)": "core_115m",
             "audioforge 0.6B speech head (v0.3 = v0.4; ICSI: retrained, speakers unseen)": "core_0p6b",
             "Silero VAD v5 (Pipecat / LiveKit)": "silero_v5", "TEN VAD": "ten_vad", "NVIDIA MarbleNet v2 frame VAD": "marblenet_v2",
             "pyannote segmentation-3.0": "pyannote_seg3"}
        def f(r):
            a, i = f"fc:vad > ami_eval > {K[r]}", f"fc:{'vad_speakers_unseen' if 'speakers unseen' in r else 'vad'} > icsi_eval > {K[r]}"
            return [CI(a, "f1_at_0.5", "f1_ci95"), [a + " > auc"], [a + " > miss_at_fpr_7.5_pct"], CI(i, "f1_at_0.5", "f1_ci95"),
                    [i + " > auc"], [i + " > miss_at_fpr_7.5_pct"], [a + " > onset_lag_ms_p50", a + " > onset_lag_ms_p90"], None]
        return f
    if h.startswith("system | assistant: accuracy"):
        def f(r):
            s, p = TURN[r]
            a, m = tb(s, p, "asst"), tb(s, p, "ami")
            if s == "smartturn_classifier_alone":
                return [[a + " > accuracy_pct", a + " > ci95 > accuracy_pct > 0", a + " > ci95 > accuracy_pct > 1"], None, [a + " > false_fire_pct"], [a + " > missed_pct"], None, None, None]
            return [[a + " > accuracy_pct", a + " > ci95 > accuracy_pct > 0", a + " > ci95 > accuracy_pct > 1"], [a + " > p50", a + " > p95"],
                    [a + " > false_fire_pct"], [a + " > missed_pct"],
                    [m + " > eot_total_ms_p50", m + " > ci95 > eot_total_ms_p50 > 0", m + " > ci95 > eot_total_ms_p50 > 1"],
                    [m + " > false_interruption_pct", m + " > ci95 > false_interruption_pct > 0", m + " > ci95 > false_interruption_pct > 1"],
                    [m + " > missed_pct", m + " > ci95 > missed_pct > 0", m + " > ci95 > missed_pct > 1"]]
        return f
    if h.startswith("system (calls, not a test split)"):
        def f(r):
            c = tb(*TURN[r], "calls")
            return [[c + " > eot_total_ms_p50", c + " > eot_total_ms_p95"],
                    [c + " > false_interruption_pct", c + " > ci95 > false_interruption_pct > 0", c + " > ci95 > false_interruption_pct > 1"],
                    [c + " > missed_pct", c + " > ci95 > missed_pct > 0", c + " > ci95 > missed_pct > 1"]]
        return f
    if h.startswith("system | AMI test tWER"):   # AMI-only after the fix wave: tWER [CI] | target miss + FA | tracking F1
        T = {"audioforge 115M (115M words)": ("ours_115m", "ours_115m"), "audioforge 0.6B (0.6B words)": ("ours_0p6b", "ours_0p6b"),
             "Nemotron-3-Diarization + print, 0.6B words": ("nemotron3_best_0p6b", "nemotron3"),
             "pyannote speaker-diarization-3.1 + print, 0.6B words": ("pyannote31_best_0p6b", "pyannote31"),
             "Nemotron-3, oracle binding (upper bound)": (None, "nemotron3_oracle_binding"),
             "pyannote 3.1, oracle binding (upper bound)": (None, "pyannote31_oracle_binding")}
        A = "fc:speaker_test > ami_eval"
        def f(r):
            if r.startswith("no filter") or r.startswith("oracle filter"):
                k = "none" if r.startswith("no") else "oracle"
                return [[f"{A} > twer > {k}_115m > twer", f"{A} > twer > {k}_0p6b > twer"], None, None]
            tw, fr = T[r]
            return [None if tw is None else [f"{A} > twer > {tw} > twer", f"{A} > twer > {tw} > ci_meeting > 0", f"{A} > twer > {tw} > ci_meeting > 1"],
                    [f"{A} > frame_best > {fr} > der_pct"], [f"{A} > frame_best > {fr} > f1"]]
        return f
    if h.startswith("embedder | AMI test EER"):
        K = {"audioforge 115M speaker head (block 4)": "core_115m", "audioforge 0.6B speaker head (block 5)": "core_0p6b",
             "NVIDIA TitaNet-L (the teacher)": "titanet_l", "pyannote WeSpeaker ResNet34 (in pyannote 3.1)": "wespeaker_pyannote"}
        return lambda r: [CI(f"fc:speaker_eer > {c}_eval > {K[r]}", "eer_within_meeting_pct", "ci95") for c in ("ami", "icsi")]
    if h.startswith("system | 2 s from speech onset"):
        K = {"Whisper large-v3": "whisper_large_v3", "Whisper large-v3-turbo": "whisper_turbo", "Whisper small": "whisper_small",
             "NVIDIA AmberNet (the teacher; reused, `data/lid/preds/ambernet`)": "ambernet", "audioforge 0.6B LID head v2": "core_0p6b",
             "audioforge 115M LID head v2": "core_115m"}
        def f(r):
            r = r.replace("⚑", "").strip()
            if r.startswith("previous heads"):
                q = "fc:lid_previous_heads"
                return [[f"{q} > core_115m > acc_2s_pct", f"{q} > core_0p6b > acc_2s_pct"], [f"{q} > core_115m > acc_full_pct", f"{q} > core_0p6b > acc_full_pct"]]
            if r not in K:
                return [["NOSRC"] * 2, ["NOSRC"] * 2]
            b = f"fc:lid > {K[r]}"
            return [CI(b, "acc_2s_pct", "acc_2s_ci95"), CI(b, "acc_full_pct", "acc_full_ci95")]
        return f
    if h.startswith("core | ms per 160 ms chunk"):
        def f(r):
            c = "115m" if "115M" in r else "0p6b"
            e = f"fc:cost > {c} > engine"
            return [[f"{e} > mps > chunk_ms_p50", f"{e} > mps > chunk_ms_p95"], [f"{e} > cpu > chunk_ms_p50", f"{e} > cpu > chunk_ms_p95"],
                    [f"fc:cost_summary > {c} > mps > realtime_streams"], [f"fc:cost_summary > {c} > cpu > realtime_streams"],
                    [f"{e} > mps > peak_rss_mb @x0.001", f"{e} > mps > mps_driver_mb @x0.001"]]
        return f
    if h.startswith("model | params"):
        K = {"audioforge 115M (whole served model, heads v0.4)": ["core_115m", "NOSRC"], "audioforge 0.6B (whole served model, heads v0.3)": ["core_0p6b", "NOSRC"],
             "Whisper small / large-v3-turbo / large-v3": ["whisper_small", "whisper_turbo", "whisper_large_v3"], "Parakeet-TDT 0.6B v3": ["tdt_v3"],
             "Parakeet-Realtime-EOU": ["parakeet_realtime_eou"], "smart-turn v3.2 (ONNX)": ["smartturn_v3.2"],
             "LiveKit turn detector (largest ONNX in the cache)": ["livekit_turn_detector/onnx/model_q8.onnx"],
             "Silero VAD v5 (TorchScript, its 8 kHz and 16 kHz branches together)": ["silero_v5"],
             "MarbleNet v2 frame VAD": ["marblenet_v2"], "pyannote segmentation-3.0 / WeSpeaker ResNet34": ["pyannote_seg3", "wespeaker_resnet34"],
             "TEN VAD": ["TENMB"], "Nemotron-3-Diarization": ["nemotron3_diar"], "TitaNet-L / AmberNet": ["titanet_l", "ambernet"]}
        def f(r):
            ks = K.get(r)
            if ks is None:
                return [None]
            return [[None if k is None else "NOSRC" if k == "NOSRC" else "fc:params > ten_vad > file_mb" if k == "TENMB" else f"fc:params > {k} > params @x1e-06" for k in ks]]
        return f
    return None


def check_tables(file, only_lines=None):
    for hl, hdr, rows in tables(file):
        if only_lines and not (only_lines[0] <= hl <= only_lines[1]):
            continue
        spec = spec_for(file, hdr)
        if spec is None:
            continue
        if spec == "STALE_CORE":
            stale_core_table(file, hl, rows)
            continue
        for ln, cells in rows:
            label = cells[0].replace("**", "").strip().strip("*").strip()
            if hdr[:3] == ["system", "device", "p50 ms"] and cells[1].startswith("MPS") and "Parakeet" in label:
                label += " @ " + cells[1]
            try:
                cols = spec(label)
            except KeyError:
                cols = None
            if cols is None:
                if any(nums(c) for c in cells[1:]):
                    out("NOSRC", f"{file}:{ln} row '{label}' not mapped")
                continue
            for ci, cell in enumerate(cells[1:]):
                paths = cols[ci] if ci < len(cols) else None
                ns = nums(cell)
                if paths is None:
                    continue
                for k, n in enumerate(ns):
                    p = paths[k] if k < len(paths) else "NOSRC"
                    check(f"{file}:{ln} [{label} | {hdr[ci + 1]}]", n, p)


def stale_core_table(file, hl, rows):
    """README '--core 0.6b' table: first-pass dev numbers (heads v0.2 / v0.3). Show the current test value."""
    now = {"WER, AMI / ICSI meetings": [W("ami_eval", "core_115m"), W("icsi_eval", "core_115m"), W("ami_eval", "core_0p6b"), W("icsi_eval", "core_0p6b")],
           "WER, live two-party calls (all words / user channel)": [W("live", "core_115m"), W("live", "core_115m", "user_channel"),
                                                                   W("live", "core_0p6b"), W("live", "core_0p6b", "user_channel")],
           "target-speaker WER, AMI / ICSI": [f"fc:speaker_test > {c} > twer > ours_{m} > twer" for m in ("115m", "0p6b") for c in ("ami_eval", "icsi")],
           "speaker EER within a meeting, AMI / ICSI": [f"fc:speaker_eer > {c}_eval > core_{m} > eer_within_meeting_pct" for m in ("115m", "0p6b") for c in ("ami", "icsi")],
           "VAD F1, AMI / ICSI": [f"fc:vad > {c}_eval > core_{m} > f1_at_0.5" for m in ("115m", "0p6b") for c in ("ami", "icsi")],
           "turn end, calls (`balanced`): p50, false interruptions, missed": [tb(f"ours_{m}", "balanced", "calls") + f" > {q}" for m in ("115m", "0p6b") for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")],
           "turn end, calls (`fast`): p50, false interruptions, missed": [tb(f"ours_{m}", "fast", "calls") + f" > {q}" for m in ("115m", "0p6b") for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")],
           "turn end, AMI (`balanced`): p50, false interruptions, missed": [tb(f"ours_{m}", "balanced", "ami") + f" > {q}" for m in ("115m", "0p6b") for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")],
           "speech to an agent (`assistant`): accuracy, p50, false fires": [tb(f"ours_{m}", "assistant", "asst") + f" > {q}" for m in ("115m", "0p6b") for q in ("accuracy_pct", "p50", "false_fire_pct")],
           "compute per 160 ms chunk, CPU 2 threads / Apple GPU": [f"fc:cost > {m} > engine > {d} > chunk_ms_p50" for m in ("115m", "0p6b") for d in ("cpu", "mps")],
           "real-time streams, CPU 2 threads / Apple GPU": [f"fc:cost_summary > {m} > {d} > realtime_streams" for m in ("115m", "0p6b") for d in ("cpu", "mps")],
           "memory": ["fc:cost > 115m > engine > mps > peak_rss_mb @x0.001", "fc:cost > 0p6b > engine > mps > peak_rss_mb @x0.001", "fc:cost > 0p6b > engine > mps > mps_driver_mb @x0.001"]}
    # README rewrite after the fix wave (test-split rows, heads v0.4; the 115M agent row = the held-out pick)
    TC = "fc:turn_clean > test > 115m > assistant > clean_servable > smartturn_test_w2.5"
    now.update({
           "WER, AMI / ICSI test meetings": now["WER, AMI / ICSI meetings"],
           "target-speaker WER, AMI test": [f"fc:speaker_test > ami_eval > twer > ours_{m} > twer" for m in ("115m", "0p6b")],
           "speaker EER within a meeting, AMI test": [f"fc:speaker_eer > ami_eval > core_{m} > eer_within_meeting_pct" for m in ("115m", "0p6b")],
           "speech detection F1 at 0.5, AMI test": [f"fc:vad > ami_eval > core_{m} > f1_at_0.5" for m in ("115m", "0p6b")],
           "turn end, AMI test (`balanced`, with the print): p50, false interruptions, missed": now["turn end, AMI (`balanced`): p50, false interruptions, missed"],
           "speech to an agent (`assistant`, 2.5 s window): accuracy, p50, cut-offs": [TC + " > accuracy_pct", TC + " > p50", TC + " > false_fire_pct",
               tb("ours_0p6b", "assistant", "asst_windows") + " > 2.5 > accuracy_pct", tb("ours_0p6b", "assistant", "asst") + " > p50",
               tb("ours_0p6b", "assistant", "asst_windows") + " > 2.5 > false_fire_pct"],
           "final text after you stop (`--final-chunk-ms 1120`, Mac GPU): p50, WER of the timed texts": [
               f"fc:final_latency > systems > ours_{m}_1120 > {q}" for m in ("115m", "0p6b") for q in ("p50", "wer_timed_texts > wer_pct")]})
    for ln, cells in rows:
        label = cells[0].strip()
        ns = [n for c in cells[1:] for n in nums(c)]
        ps = now.get(label, [])
        for k, n in enumerate(ns):
            p = ps[k] if k < len(ps) else None
            if p is None:
                out("NOSRC", f"{file}:{ln} [--core table | {label}] {n}: no test-split source")
                continue
            v = get(p)
            dec = len(n.split(".")[1]) if "." in n else 0
            if Decimal(n) != rnd(v, dec):
                out("STALE", f"{file}:{ln} [--core table | {label}] shows {n}; current test value {rnd(v, dec)} ({p})")
            else:
                out("OK", f"{file}:{ln} [--core table | {label}] {n}")


# ---------------------------------------------------------------- [TXT] prose claims: "text with {number}" -> path
TXT = [("docs/RESULTS.md", s, p) for s, p in [  # README prose after the fix-wave rewrite, moved to docs/RESULTS.md (2026-10-03)
    ("ours 115M {1527} ms", "fc:turn > ours_115m > balanced > ami > eot_total_ms_p50"),
    ("1527 ms / {15.5} % interruptions", "fc:turn > ours_115m > balanced > ami > false_interruption_pct"),
    ("interruptions / {36.0} % missed", "fc:turn > ours_115m > balanced > ami > missed_pct"),
    ("ours 0.6B {1498} ms", "fc:turn > ours_0p6b > balanced > ami > eot_total_ms_p50"),
    ("ours 0.6B 1498 ms / {10.0} %", "fc:turn > ours_0p6b > balanced > ami > false_interruption_pct"),
    ("ours 0.6B 1498 ms / 10.0 % / {33.5} %", "fc:turn > ours_0p6b > balanced > ami > missed_pct"),
    ("ours 115M {1508} ms", "fc:turn > ours_115m > balanced > ami_noprint > eot_total_ms_p50"),
    ("ours 115M 1508 ms / {6.5} %", "fc:turn > ours_115m > balanced > ami_noprint > false_interruption_pct"),
    ("ours 115M 1508 ms / 6.5 % / {74.0} %", "fc:turn > ours_115m > balanced > ami_noprint > missed_pct"),
    ("ours 0.6B {1499} ms", "fc:turn > ours_0p6b > balanced > ami_noprint > eot_total_ms_p50"),
    ("ours 0.6B 1499 ms / {5.5} %", "fc:turn > ours_0p6b > balanced > ami_noprint > false_interruption_pct"),
    ("ours 0.6B 1499 ms / 5.5 % / {73.5} %", "fc:turn > ours_0p6b > balanced > ami_noprint > missed_pct"),
    ("LiveKit {745} ms", "fc:turn > livekit_en_turn_detector_silero > ami > eot_total_ms_p50"),
    ("LiveKit 745 ms / {13.0} %", "fc:turn > livekit_en_turn_detector_silero > ami > false_interruption_pct"),
    ("LiveKit 745 ms / 13.0 % / {72.0} %", "fc:turn > livekit_en_turn_detector_silero > ami > missed_pct"),
    ("Pipecat {752} ms", "fc:turn > pipecat_smartturn_v3.2_silero > ami > eot_total_ms_p50"),
    ("Pipecat 752 ms / {39.5} %", "fc:turn > pipecat_smartturn_v3.2_silero > ami > false_interruption_pct"),
    ("Pipecat 752 ms / 39.5 % / {53.0} %", "fc:turn > pipecat_smartturn_v3.2_silero > ami > missed_pct"),
    ("Parakeet-EOU {1251} ms", "fc:turn > parakeet_realtime_eou > ami > eot_total_ms_p50"),
    ("Parakeet-EOU 1251 ms / {4.5} %", "fc:turn > parakeet_realtime_eou > ami > false_interruption_pct"),
    ("Parakeet-EOU 1251 ms / 4.5 % / {86.0} %", "fc:turn > parakeet_realtime_eou > ami > missed_pct"),
    ("F1 at threshold 0.5 ({0.959}\n115M", "fc:vad > ami_eval > core_115m > f1_at_0.5"),
    ("115M / {0.957} 0.6B against", "fc:vad > ami_eval > core_0p6b > f1_at_0.5"),
    ("MarbleNet v2 {0.941}", "fc:vad > ami_eval > marblenet_v2 > f1_at_0.5"),
    ("TEN VAD {0.930}, Silero", "fc:vad > ami_eval > ten_vad > f1_at_0.5"),
    ("Silero v5 {0.901}", "fc:vad > ami_eval > silero_v5 > f1_at_0.5"),
    ("(AUC {0.967} against", "fc:vad > ami_eval > marblenet_v2 > auc"),
    ("against {0.966} / 0.967)", "fc:vad > ami_eval > core_115m > auc"),
    ("against 0.966 / {0.967})", "fc:vad > ami_eval > core_0p6b > auc"),
    ("false alarms\n({9.6} against", "fc:vad > ami_eval > marblenet_v2 > miss_at_fpr_7.5_pct"),
    ("({9.6} against {12.5} / 12.6 %)".replace("{9.6}", "9.6"), "fc:vad > ami_eval > core_115m > miss_at_fpr_7.5_pct"),
    ("against 12.5 / {12.6} %)", "fc:vad > ami_eval > core_0p6b > miss_at_fpr_7.5_pct"),
    ("best on AMI (F1 {0.977})", "fc:vad > ami_eval > pyannote_seg3 > f1_at_0.5"),
    ("(115M {0.930} against Silero", "fc:vad_speakers_unseen > icsi_eval > core_115m > f1_at_0.5"),
    ("against Silero {0.924}", "fc:vad > icsi_eval > silero_v5 > f1_at_0.5"),
    ("and TEN VAD {0.922}; the", "fc:vad > icsi_eval > ten_vad > f1_at_0.5"),
    ("the 0.6B's {0.922} ties", "fc:vad_speakers_unseen > icsi_eval > core_0p6b > f1_at_0.5"),
    ("(AUC {0.941} / 0.945 against", "fc:vad_speakers_unseen > icsi_eval > core_115m > auc"),
    ("(AUC 0.941 / {0.945} against", "fc:vad_speakers_unseen > icsi_eval > core_0p6b > auc"),
    ("against at most {0.932};", "fc:vad > icsi_eval > ten_vad > auc"),
    ("{16.0} / 15.5 % speech missed", "fc:vad_speakers_unseen > icsi_eval > core_115m > miss_at_fpr_7.5_pct"),
    ("16.0 / {15.5} % speech missed", "fc:vad_speakers_unseen > icsi_eval > core_0p6b > miss_at_fpr_7.5_pct"),
    ("at\nleast {20.5} %", "fc:vad > icsi_eval > ten_vad > miss_at_fpr_7.5_pct"),
    ("ours 0.6B {47.1}, ours 115M", "fc:speaker_test > ami_eval > twer > ours_0p6b > twer"),
    ("ours 115M {51.5}, Nemotron-3", "fc:speaker_test > ami_eval > twer > ours_115m > twer"),
    ("Nemotron-3 diarizer {64.4}", "fc:speaker_test > ami_eval > twer > nemotron3_best_0p6b > twer"),
    ("pyannote 3.1 {77.8}", "fc:speaker_test > ami_eval > twer > pyannote31_best_0p6b > twer"),
    ("(tracking F1 {0.795} against", "fc:speaker_test > ami_eval > frame_best > nemotron3_oracle_binding > f1"),
    ("against {0.811} / 0.829)", "fc:speaker_test > ami_eval > frame_best > ours_115m > f1"),
    ("against 0.811 / {0.829})", "fc:speaker_test > ami_eval > frame_best > ours_0p6b > f1"),
    ("Whisper large-v3 {95.9} %", "fc:lid > whisper_large_v3 > acc_2s_pct"),
    ("AmberNet {95.1} %", "fc:lid > ambernet > acc_2s_pct"),
    ("ours 0.6B\n{92.7} %", "fc:lid > core_0p6b > acc_2s_pct"),
    ("ours 115M {92.4} %", "fc:lid > core_115m > acc_2s_pct"),
    ("Whisper small {90.6} %", "fc:lid > whisper_small > acc_2s_pct"),
    ("115M {28.6} ms on the GPU", "fc:cost > 115m > engine > mps > chunk_ms_p50"),
    ("{30.4} ms on 2 CPU", "fc:cost > 115m > engine > cpu > chunk_ms_p50"),
    ("{5} real-time streams on the GPU", "fc:cost_summary > 115m > mps > realtime_streams"),
    ("streams on the GPU, {1.2} GB", "fc:cost > 115m > engine > mps > peak_rss_mb @x0.001"),
    ("0.6B {42.9} ms GPU", "fc:cost > 0p6b > engine > mps > chunk_ms_p50"),
    ("{97.1} ms CPU", "fc:cost > 0p6b > engine > cpu > chunk_ms_p50"),
    ("{3} streams on the GPU", "fc:cost_summary > 0p6b > mps > realtime_streams"),
    ("{4.9} GB.", "fc:cost > 0p6b > engine > mps > peak_rss_mb @x0.001"),
    ("(211 ms against {351} / 381 ms", "fc:turn > ours_0p6b > assistant > asst > p50"),
    ("(211 ms against 351 / {381} ms", "fc:turn_clean > test > 115m > assistant > clean_servable > smartturn_test_w2.5 > p50"),
    ("answers assistant speech sooner ({211} ms", "fc:turn > pipecat_smartturn_v3.2_silero > asst > p50"),
    ("at {29} % cut-offs", "fc:turn > pipecat_smartturn_v3.2_silero > asst_windows > 2.5 > false_fire_pct"),
    ("against our {0.9} / 2.7 %", "fc:turn > ours_0p6b > assistant > asst_windows > 2.5 > false_fire_pct"),
    ("against our 0.9 / {2.7} %", "fc:turn_clean > test > 115m > assistant > clean_servable > smartturn_test_w2.5 > false_fire_pct"),
    ("ICSI test meetings ({7.6} against", W("icsi_eval", "tdt_v3")),
    ("(7.6 against {10.3} %)", W("icsi_eval", "core_0p6b")),
    ("LibriSpeech\n  ({3.4} against", W("ls_other", "tdt_v3")),
    ("(3.4 against {5.7} % on test-other)", W("ls_other", "core_0p6b")),
    ("FLEURS English ({7.0} against", W("fleurs_en", "tdt_v3")),
    ("(7.0 against {8.0} %)", W("fleurs_en", "core_0p6b")),
    ("tie on the user channel ({5.4} % each)", W("live", "core_0p6b", "user_channel").replace("whisper_norm|", "whisper_norm_disfl|")),
    ("(95.9 against {92.7} %)", "fc:lid > core_0p6b > acc_2s_pct"),
    ("EER AMI test\n  {1.9} % against", "fc:speaker_eer > ami_eval > titanet_l > eer_within_meeting_pct"),
    ("1.9 % against {3.8}-5.0 %", "fc:speaker_eer > ami_eval > core_0p6b > eer_within_meeting_pct"),
    ("1.9 % against 3.8-{5.0} %", "fc:speaker_eer > ami_eval > core_115m > eer_within_meeting_pct"),
    ("fast final texts have {14.2} % WER", "fc:final_latency > systems > ours_115m_1120 > wer_timed_texts > wer_pct"),
    ("about {4.7}× sooner", None),
]] + [("docs/USAGE.md", s, p) for s, p in [  # the README dual-rate paragraph, moved to docs/USAGE.md
    ("the slow `final` follows {9}-48 ms later", "dr:live > 115m_mps_F1120 > slow_final > flush_wall_ms > p50"),
    ("follows 9-{48} ms later", "dr:live > 0p6b_mps_F1120 > slow_final > flush_wall_ms > p95"),
    ("(~{105} ms for the 0.6B on CPU", "dr:live > 0p6b_cpu_F1120 > slow_final > flush_wall_ms > p50"),
    ("ICSI test −{3.5} (115M)", "dr:wer > 115m|icsi_eval > r13 > delta_vs_160_pp @x-1"),
    ("/ −{1.9} (0.6B)", "dr:wer > 0p6b|icsi_eval > r13 > delta_vs_160_pp @x-1"),
    ("live calls −{1.7} / −0.8", "dr:wer > 115m|live > r13 > delta_vs_160_pp @x-1"),
    ("live calls −1.7 / −{0.8}", "dr:wer > 0p6b|live > r13 > delta_vs_160_pp @x-1"),
    ("AMI test\n−{0.6} / −0.1", "dr:wer > 115m|ami_eval > r13 > delta_vs_160_pp @x-1"),
    ("−0.6 / −{0.1} (not", "dr:wer > 0p6b|ami_eval > r13 > delta_vs_160_pp @x-1"),
    ("115M on the GPU {4} → 3", "dr:cost > 115m|F160 > realtime_streams_mps"),
    ("115M on the GPU 4 → {3}", "dr:cost > 115m|F1120 > realtime_streams_mps"),
    ("0.6B on the GPU {3} → 1", "dr:cost > 0p6b|F160 > realtime_streams_mps"),
    ("0.6B on the GPU 3 → {1}", "dr:cost > 0p6b|F1120 > realtime_streams_mps"),
]] + [("README.md", s, p) for s, p in [  # the short README results table (docs split, 2026-10-03)
    ("(AMI test, WER) | **{7.9} %**", W("ami_eval", "core_0p6b")),
    ("WER) | **7.9 %** | Parakeet-TDT v3 {8.3} %", W("ami_eval", "tdt_v3")),
    ("Parakeet-TDT v3 8.3 %, Whisper large-v3 {10.7} %", W("ami_eval", "whisper_large_v3")),
    ("(Mac GPU, median) | **{49} ms**", "fc:final_latency > systems > ours_0p6b_1120 > p50"),
    ("| Parakeet-TDT v3 {232} ms", "fc:final_latency > systems > parakeet_tdt_mps > p50"),
    ("Whisper large-v3-turbo {282} ms", "fc:final_latency > systems > whisper_turbo > p50"),
    ("cuts the user off | **{97.7} % / 0.9 %**", tb("ours_0p6b", "assistant", "asst_windows") + " > 2.5 > accuracy_pct"),
    ("cuts the user off | **97.7 % / {0.9} %**", tb("ours_0p6b", "assistant", "asst_windows") + " > 2.5 > false_fire_pct"),
    ("| LiveKit {85.2} % / 22.3 %", tb("livekit_en_turn_detector_silero", None, "asst_windows") + " > 2.5 > accuracy_pct"),
    ("| LiveKit 85.2 % / {22.3} %", tb("livekit_en_turn_detector_silero", None, "asst_windows") + " > 2.5 > false_fire_pct"),
    ("Pipecat {75.9} % / 29.0 %", tb("pipecat_smartturn_v3.2_silero", None, "asst_windows") + " > 2.5 > accuracy_pct"),
    ("Pipecat 75.9 % / {29.0} %", tb("pipecat_smartturn_v3.2_silero", None, "asst_windows") + " > 2.5 > false_fire_pct"),
    ("knows the user's voice) | **{33.5} %**", "fc:turn > ours_0p6b > balanced > ami > missed_pct"),
    ("| Pipecat {53.0} %, LiveKit", "fc:turn > pipecat_smartturn_v3.2_silero > ami > missed_pct"),
    ("Pipecat 53.0 %, LiveKit {72.0} %", "fc:turn > livekit_en_turn_detector_silero > ami > missed_pct"),
    ("(AMI test, F1) | **{0.959}**", "fc:vad > ami_eval > core_115m > f1_at_0.5"),
    ("| MarbleNet v2 {0.941}, Silero", "fc:vad > ami_eval > marblenet_v2 > f1_at_0.5"),
    ("MarbleNet v2 0.941, Silero v5 {0.901}", "fc:vad > ami_eval > silero_v5 > f1_at_0.5"),
    ("the 0.6B scores {0.957})", "fc:vad > ami_eval > core_0p6b > f1_at_0.5"),
    ("(AMI test) | **{47.1} %**", "fc:speaker_test > ami_eval > twer > ours_0p6b > twer"),
    ("| Nemotron-3 diarizer {64.4} %", "fc:speaker_test > ami_eval > twer > nemotron3_best_0p6b > twer"),
    ("pyannote 3.1 {77.8} % |", "fc:speaker_test > ami_eval > twer > pyannote31_best_0p6b > twer"),
    ("assistant sooner ({211} ms against 351 ms)", "fc:turn > pipecat_smartturn_v3.2_silero > asst > p50"),
    ("(211 ms against {351} ms)", "fc:turn > ours_0p6b > assistant > asst > p50"),
]] + [("research/FINAL_COMPARE.md", s, p) for s, p in [
    ("streaming: a word shows up about {0.3} s", "fc:stt_latency > 0p6b > p50_ms @x0.001"),
    ("p50 {274} ms [250, 291], p95 571 ms", "fc:stt_latency > 115m > p50_ms"),
    ("p50 274 ms [250, 291], p95 {571} ms", "fc:stt_latency > 115m > p95_ms"),
    ("p50 {291} ms [272, 311]", "fc:stt_latency > 0p6b > p50_ms"),
    ("[272, 311], p95 {591} ms", "fc:stt_latency > 0p6b > p95_ms"),
    ("- 0.6B: {7.76} [6.09, 9.51]", W("fleurs_en", "core_0p6b_f1120")),
    ("- 115M: {10.51} [8.61, 12.40]", W("fleurs_en", "core_115m_f1120")),
    ("in 29 ms per 160 ms of audio", None),
    ("The 115M runs a whole session in {29} ms", "fc:cost > 115m > engine > mps > chunk_ms_p50"),
    ("0.6B needs {43} ms", "fc:cost > 0p6b > engine > mps > chunk_ms_p50"),
    ("before (v0.3) {93.0} % [90.5, 95.5]", "td:evalv04 > v0.3 > assistant > asst > accuracy_pct"),
    ("[90.5, 95.5] / {379} ms", "td:evalv04 > v0.3 > assistant > asst > p50"),
    ("379 ms / {4.5} %", "td:evalv04 > v0.3 > assistant > asst > false_fire_pct"),
    ("AMI test {1619} / 4.0 % / 53.0 %", "td:evalv04 > v0.3 > assistant > ami > eot_total_ms_p50"),
    ("AMI test 1619 / {4.0} % / 53.0 %", "td:evalv04 > v0.3 > assistant > ami > false_interruption_pct"),
    ("AMI test 1619 / 4.0 % / {53.0} %", "td:evalv04 > v0.3 > assistant > ami > missed_pct"),
    ("p50, 2-5 s (17) | p50, > 5 s (30) |", None),
]]


def check_txt():
    for file, snip, path in TXT:
        if path is None:
            continue
        text = (ROOT / file).read_text()
        m = re.search(r"\{([^}]*)\}", snip)
        pat = re.escape(snip[:m.start()]) + r"(" + re.escape(m.group(1)) + r")" + re.escape(snip[m.end():])
        pat = pat.replace(r"\ ", r"\s+").replace(r"\n", r"\s+").replace("\\\n", r"\s+")
        mm = re.search(pat, text)
        if not mm:
            out("NOTE", f"[TXT] {file}: claim text not found (doc changed?): {snip!r}")
            continue
        ln = text[:mm.start(1)].count("\n") + 1
        check(f"{file}:{ln} [prose '{snip.replace(chr(10), ' ')}']", m.group(1), path)


# ---------------------------------------------------------------- [STALE]
FIRST_PASS = {  # FINAL_COMPARE "What changed" first-pass (dev) column + README --core table + FOCUS rows in arch_vs.html
    "10.4", "20.6", "9.5", "12.4", "13.6", "26.2", "10.4", "12.8", "0.951", "0.937", "0.898", "0.906", "0.935", "1326",
    "1180", "1177", "63.2", "62.1", "71.6", "85.3", "31.7", "37.1", "60.0", "74.8", "19.8", "12.0", "11.9", "91.0", "87.6",
    "87.4", "27.7", "43.7", "92.2", "292", "24.4", "27.3", "11.2", "14.4", "22.5", "13.7", "17.1", "956", "725", "487"}
PUBLIC = ["README.md", "docs/README.md", "docs/USAGE.md", "docs/TRAINING.md", "docs/RESULTS.md", "docs/ARCHITECTURE.md", "docs/CONFIGURATION.md", "docs/MODELS.md", "docs/PROTOCOL.md",
          "docs/SERVER_INTERNALS.md", "docs/RELEASE_CHECKLIST.md", "docs/REPO_LAYOUT.md", "demo/README.md",
          "demo/images/EXPLAINER.md", "research/README.md", "research/METRICS.md", "CHANGELOG.md",
          "packages/audioforge-client/README.md", "examples/audio/README.md"]


def check_stale():
    for f in PUBLIC:
        p = ROOT / f
        if not p.exists():
            continue
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            hits = sorted({n for n in NUM.findall(line.replace("−", "-")) if n in FIRST_PASS})
            if hits and re.search(r"%|ms|F1|WER|EER|AMI|ICSI|0\.\d{3}", line):
                out("STALE", f"{f}:{ln} first-pass/dev-era numbers {hits}: {line.strip()[:150]}")
            if re.search(r"heads v0\.[23]\b(?! *\+)|\(heads v0\.[23]\)", line) and "v0.4" not in line:
                out("STALE", f"{f}:{ln} old head version: {line.strip()[:150]}")
            for img in re.findall(r"[\w./-]+\.png", line):
                ip = next((q for q in (ROOT / img, ROOT / "demo/images" / Path(img).name) if q.exists()), None)
                if ip and "compare_" not in img:
                    older = os.path.getmtime(ip) < min(os.path.getmtime(q) for q in (ROOT / "demo/images").glob("compare_*.png"))
                    out("STALE" if older and f in ("README.md",) or re.search(r"results_v|architecture_v[89]", img) else "NOTE",
                        f"{f}:{ln} references {img} (mtime {'older' if older else 'newer'} than compare_*.png)")
    html = (ROOT / "demo/images/redesign/arch_vs.html").read_text()
    for ln, line in enumerate(html.splitlines(), 1):
        if "rows: [[" in line or line.strip().startswith('["'):
            hits = sorted({n for n in NUM.findall(line) if n in FIRST_PASS})
            if hits:
                out("STALE", f"demo/images/redesign/arch_vs.html:{ln} FOCUS rows (focus_*.png, not compare_*) hard-code first-pass {hits}")


def check_prov():
    fc = J["final_compare"]
    out("NOTE", f"[PROV] final_compare.json afm: 115m={Path(fc['afm']['115m']).name}, 0p6b={Path(fc['afm']['0p6b']).name}; "
        f"turn.ours_0p6b afm={Path(fc['turn']['ours_0p6b']['afm']).name}; params core_115m from {Path(fc['params']['core_115m']['file']).name}, "
        f"core_0p6b from {Path(fc['params']['core_0p6b']['file']).name} (doc rows say heads v0.4 / v0.4)")
    d, f = J["dual_rate"]["cost"], fc["cost_summary"]
    for c in ("115m", "0p6b"):
        for dev in ("mps", "cpu"):
            a, b = d[f"{c}|F160"][f"realtime_streams_{dev}"], f[c][dev]["realtime_streams"]
            if a != b:
                out("NOTE", f"[PROV] real-time streams {c} {dev}: final_compare {b} vs dual_rate F160 {a} (same engine, two runs)")
    for c in ("115m", "0p6b"):
        for st in ("ls_clean", "ls_other", "ami_eval", "icsi_eval", "live"):
            a, b = J["dual_rate"]["wer"][f"{c}|{st}"]["r1"]["wer_pct"], fc["words"][st]["whisper_norm|all"][f"core_{c}"]["wer_pct"]
            if abs(a - b) > 0.005:
                out("NOTE", f"[PROV] 160 ms WER {c} {st}: dual_rate r1 {a} vs final_compare {b}")
    L = J["dual_rate"]["live"]
    for c in ("115m", "0p6b"):
        for dev in ("cpu", "mps"):
            try:
                x = L[f"{c}_{dev}_F1120"]["turn_end_delivery_ms"]["p95"] - L[f"{c}_{dev}_F160"]["turn_end_delivery_ms"]["p95"]
                out("NOTE", f"[PROV] dual-rate turn-end delivery p95 cost {c} {dev}: +{x:.1f} ms (docs/USAGE.md says +9-27 ms, +93 ms for the 0.6B on CPU)")
            except KeyError:
                pass


if __name__ == "__main__":
    print("== [NF]/[IMG] numbers_final.json and the image keys"); check_nf()
    print("== [TAB] README.md / docs/RESULTS.md tables"); check_tables("README.md"); check_tables("docs/RESULTS.md")
    print("== [TAB] research/FINAL_COMPARE.md headline tables")
    t = (ROOT / "research/FINAL_COMPARE.md").read_text().splitlines()
    end = next(i for i, l in enumerate(t, 1) if l.startswith("## What changed"))
    check_tables("research/FINAL_COMPARE.md", (1, end))
    print("== [TXT] prose claims"); check_txt()
    print("== [STALE] public docs"); check_stale()
    print("== [PROV]"); check_prov()
    print("== summary", {k: v for k, v in COUNT.items()})
    sys.exit(1 if COUNT["MISMATCH"] + COUNT["NOSRC"] + COUNT["STALE"] else 0)
