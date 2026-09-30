"""Every number shown in the showcase video, read from its runs/*.json (no hand-typed values).

    PYTHONPATH=. .venv/bin/python demo/archive/extract_numbers.py  -> demo/archive/numbers.json

Each entry: value, unit/scope, and the source file + JSON path it was read from; demo/archive/render.py prints the source file
in the corner of every numbers scene. If a path is missing the script fails, so a stale number cannot ship.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(d, path):
    """``path`` = keys joined by " > " (JSON keys may themselves contain "/")."""
    for k in path.split(" > "):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def load(name):
    return json.loads((ROOT / "runs" / name).read_text())


def main():
    out = {}

    def put(key, value, scope, src, path, fmt=None):
        out[key] = {"value": value, "scope": scope, "source": f"runs/{src}", "path": path}
        if fmt:
            out[key]["text"] = fmt

    # ---- end-of-turn, eot-bench v2 (AMI dev n=974, <=5 % false cutoffs, 6 s horizon) ------------------------
    bt = load("baselines_turn.json")
    sec = "C_extended_windows_stream > systems"
    turn = {"ours_hybrid": "hybrid_stream_causal_dominant", "silero_timeout": "silero_timeout",
            "parakeet_eou": "eou_posterior", "smartturn_silero": "smartturn_silero+timeout",
            "livekit_text": "livekit_text_en_asr+timeout", "ours_head_or_silero": "head_trail6+silero_timeout"}
    for k, sysname in turn.items():
        p = f"{sec} > {sysname} > 6s > fixed_5pct_turn_fc"
        r = get(bt, p)
        put(f"turn_miss_all/{k}", round(100 * r["miss_rate"], 1), "% missed turn ends, all ends, AMI dev n=974, 6 s, <=5 % FC",
            "baselines_turn.json", p + " > miss_rate")
        put(f"turn_miss_open/{k}", round(100 * r["strata"]["open"]["miss_rate"], 1),
            "% missed, floor-open ends (n=236)", "baselines_turn.json", p + " > strata > open > miss_rate")
    put("turn_n", get(bt, "C_extended_windows_stream > n"), "turn ends scored", "baselines_turn.json", "C_extended_windows_stream > n")

    # ---- TurnBench dev (Sesame's official scorer, <=10 % FP) ----------------------------------------------------
    db = load("dyadic_bench.json")
    pub = "turnbench_published_dev_rescored"
    for k in ["vap", "smart_turn_v3", "kyutai_semantic_vad"]:
        put(f"turnbench_recall/{k}", get(db, f"{pub} > {k} > recall"), "recall, TurnBench dev", "dyadic_bench.json", f"{pub} > {k} > recall")
        put(f"turnbench_p50/{k}", get(db, f"{pub} > {k} > latency_ms > p50"), "median latency ms", "dyadic_bench.json", f"{pub} > {k} > latency_ms > p50")
    # ours: the hybrid point of research/archive/DYADIC.md (in-sample), and the head alone
    ours = _find_key(db, "hybrid", must={"recall": 0.8351})
    put("turnbench_recall/ours_hybrid", ours[1]["recall"], "recall, TurnBench dev, head OR Silero per channel", "dyadic_bench.json", ours[0] + " > recall")
    put("turnbench_p50/ours_hybrid", ours[1]["latency_ms"]["p50"], "median latency ms", "dyadic_bench.json", ours[0] + " > latency_ms > p50")
    dt = load("dyadic_train.json")
    pred = _find_key(dt, "tb_halves_pooled", must={"recall": 0.8619})
    put("turnbench_recall/ours_predictive", pred[1]["recall"], "recall, TurnBench dev, predictive trigger OR Silero, held-out halves pooled",
        "dyadic_train.json", pred[0] + " > recall")
    put("turnbench_p50/ours_predictive", pred[1]["p50"], "median latency ms", "dyadic_train.json", pred[0] + " > p50")

    # ---- speaker head ---------------------------------------------------------------------------------------
    sh = load("spk_head.json")
    for k, var in (("original", "stage1_spk_head"), ("shipped", "stage1_served")):
        p = f"variants > {var} > eval > ami_n64 > eer_within_meeting"
        put(f"spk_eer/{k}", round(100 * get(sh, p), 1), f"% within-meeting EER, AMI dev n=64 ({var})", "spk_head.json", p)

    # ---- ASR ------------------------------------------------------------------------------------------------
    fa = load("final_asr.json")
    ha = load("hybrid_asr.json")
    put("wer/stream_libri", round(100 * get(fa, "results > served/libri > wer_whisper_norm > wer"), 2), "% WER LibriSpeech test-clean (200 utt), streaming 160 ms",
        "final_asr.json", "results > served/libri > wer_whisper_norm > wer")
    put("wer/stream_ami", round(100 * get(fa, "results > served/ami > wer_whisper_norm > wer"), 1), "% WER AMI dev segments (200), streaming 160 ms",
        "final_asr.json", "results > served/ami > wer_whisper_norm > wer")
    put("wer/tdt_v3_ami", round(100 * get(ha, "english > results > tdt_v3/ami > wer_whisper_norm > wer"), 1), "% WER AMI dev segments (200), Parakeet-TDT 0.6B v3 per-turn final",
        "hybrid_asr.json", "english > results > tdt_v3/ami > wer_whisper_norm > wer")
    put("wer/whisper_turbo_ami", round(100 * get(fa, "results > whisper_turbo/ami > wer_whisper_norm > wer"), 1), "% WER AMI, Whisper large-v3-turbo offline",
        "final_asr.json", "results > whisper_turbo/ami > wer_whisper_norm > wer")
    for sysk, name in (("whisper_small", "whisper_small"), ("parakeet", "parakeet_ctc"), ("whisper_turbo", "whisper_turbo")):
        for st in ("libri", "ami"):
            put(f"wer/{name}_{st}", round(100 * get(fa, f"results > {sysk}/{st} > wer_whisper_norm > wer"), 2 if st == "libri" else 1),
                f"% WER {st} (200), {sysk} offline, Whisper-normalised", "final_asr.json", f"results > {sysk}/{st} > wer_whisper_norm > wer")
    put("wer/tdt_v3_libri", round(100 * get(ha, "english > results > tdt_v3/libri > wer_whisper_norm > wer"), 2), "% WER LibriSpeech (200), Parakeet-TDT 0.6B v3 per turn",
        "hybrid_asr.json", "english > results > tdt_v3/libri > wer_whisper_norm > wer")
    put("rtf/stream_asr_cpu2", get(fa, "results > served/ami > rtf_cpu2"), "RTF, streaming ASR alone, 2 CPU threads", "final_asr.json", "results > served/ami > rtf_cpu2")

    # ---- VAD ------------------------------------------------------------------------------------------------
    sd = load("baselines_sd.json")
    ours_vad = _find_key(sd, "f1", must=None, prefer=0.949)
    put("vad_f1/ours", ours_vad[1], "F1, AMI dev 80 ms frames", "baselines_sd.json", ours_vad[0])
    put("vad_f1/silero", get(sd, "vad > silero_v5.1.2 > sweep > 0.5 > f1"), "F1, Silero v5 at 0.5", "baselines_sd.json", "vad > silero_v5.1.2 > sweep > 0.5 > f1")

    # ---- product head-to-head (e2e_final.json) -------------------------------------------------------------
    e2e = load("e2e_final.json")
    names = {"A": "Pipecat default", "B": "LiveKit default", "C": "ours, 1 s timeout", "CN": "ours + Nemotron-3, 1 s timeout",
             "D": "ours, hybrid_dyn + arm", "DN": "ours + Nemotron-3, hybrid_dyn + arm"}
    for key, row in e2e["table"].items():
        st, cond, fw, sysm = key.split("|")
        if cond != "mono" or fw == "offline":
            continue
        put(f"e2e/{st}/{fw}/{sysm}", {"dead_air_ms_median": row["dead_air_ms_median"], "cut_ins_per_min": row["cut_ins_per_min"],
                                     "missed_6s": row["missed_6s"], "n_clips": row["n_clips"], "rtf": row["server_rtf_median"],
                                     "rss_mb": row["rss_mb_server_peak"], "name": names[sysm]},
            f"{st} clips, mono mic, {fw}", "e2e_final.json", f"table > {key}")
    rtfs = [row["server_rtf_median"] for k, row in e2e["table"].items() if k.endswith("|CN") or k.endswith("|DN")]
    put("rtf/server_nemotron_range", [min(rtfs), max(rtfs)], "server RTF median range, ours + Nemotron-3 diarizer, all sets/frameworks, 2 CPU threads",
        "e2e_final.json", "table > *|CN,DN > server_rtf_median")
    rtfs2 = [row["server_rtf_median"] for k, row in e2e["table"].items() if k.endswith("|C") or k.endswith("|D")]
    put("rtf/server_sortformer_range", [min(rtfs2), max(rtfs2)], "server RTF median range, ours + Sortformer v2", "e2e_final.json", "table > *|C,D > server_rtf_median")


    # ---- TS-VAD track feeding the turn head (research/IMPROVE_115M.md A.2; runs/improve_115m.json) ---------------
    im = load("improve_115m.json")
    for ds, n in (("ami", 974), ("icsi", 1312)):
        for k, sysname in (("hybrid_sortformer", "hybrid_causal_dominant"), ("hybrid_tsvad", "hybrid_tsvad_spk"),
                           ("hybrid_dyn_tsvad", "hybrid_dyn_tsvad_spk"), ("hybrid_oracle", "hybrid_oracle"),
                           ("silero_timeout", "silero_timeout")):
            p = f"turn_bench > {ds} > systems > {sysname} > 6s"
            try:
                r = get(im, p)
            except KeyError:
                continue  # ICSI Sortformer rows live in baselines_turn_icsi.json (below)
            put(f"tsvad_turn/{ds}/{k}", round(100 * r["miss_rate"], 1), f"% missed turn ends at 6 s, {ds.upper()} n={n}, <=5 % FC, 5 s voice print",
                "improve_115m.json", p + " > miss_rate")
            put(f"tsvad_turn/{ds}/{k}/ci", [round(100 * c, 1) for c in r["miss_rate_ci"]], "95 % CI", "improve_115m.json", p + " > miss_rate_ci")
            put(f"tsvad_turn/{ds}/{k}/open", round(100 * r["strata"]["open"]["miss_rate"], 1), "% missed, floor-open ends", "improve_115m.json", p + " > strata > open > miss_rate")
    bi = load("baselines_turn_icsi.json")
    p = "C_extended_windows_stream > crossfit_icsi > systems > hybrid_stream_causal_dominant > 6s > fixed_5pct_turn_fc"
    r = get(bi, p)
    put("tsvad_turn/icsi/hybrid_sortformer", round(100 * r["miss_rate"], 1), "% missed at 6 s, ICSI n=1312, Sortformer causal-dominant column -> shipped hybrid, cross-fitted on ICSI",
        "baselines_turn_icsi.json", p + " > miss_rate")
    put("tsvad_frame_f1/ami", get(im, "frame > ami > primary > tsvad_spk_vp5p0 > all > f1"), "target-speaker frame F1, AMI dev, 5 s print", "improve_115m.json", "frame > ami > primary > tsvad_spk_vp5p0 > all > f1")
    put("tsvad_params", 260_000, "TS-VAD head parameters (research/IMPROVE_115M.md: 260k on block 4)", "improve_115m.json", "(text)")

    # ---- E2E first-partial latency and cut-ins ranges (research/E2E_FINAL.md section 1) -------------------------
    ft = {s: [] for s in ("A", "B", "C")}
    ci = {s: [] for s in ("A", "B", "C")}
    for key, row in e2e["table"].items():
        st, cond, fw, sysm = key.split("|")
        if sysm in ft and fw != "offline":
            ft[sysm].append(row["first_text_ms_median"]); ci[sysm].append(row["cut_ins_per_min"])
    for s, name in (("A", "pipecat_default"), ("B", "livekit_default"), ("C", "ours_timeout")):
        put(f"e2e_first_text_ms/{name}", [min(ft[s]), max(ft[s])], "median first-partial latency after speech onset, ms, range over clip sets x frameworks", "e2e_final.json", f"table > *|{s} > first_text_ms_median")
        put(f"e2e_cut_ins_per_min/{name}", [round(min(ci[s]), 2), round(max(ci[s]), 2)], "cut-ins per minute, range over clip sets x frameworks", "e2e_final.json", f"table > *|{s} > cut_ins_per_min")

    (ROOT / "demo" / "numbers.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:40s} {str(v['value'])[:60]:60s} <- {v['source']}:{v['path'][:70]}")


def _find_key(d, name, must=None, prefer=None, path=""):
    """Depth-first search for a dict key ``name`` whose value satisfies ``must`` (subset) / equals ``prefer``."""
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path} > {k}" if path else k
            if k == name:
                if must is not None and isinstance(v, dict) and all(abs(v.get(mk, -9) - mv) < 1e-4 for mk, mv in must.items()):
                    return p, v
                if must is None and prefer is not None and isinstance(v, (int, float)) and abs(v - prefer) < 5e-4:
                    return p, v
            r = _find_key(v, name, must, prefer, p)
            if r:
                return r
    elif isinstance(d, list):
        for i, v in enumerate(d):
            r = _find_key(v, name, must, prefer, f"{path} > {i}")
            if r:
                return r
    return None


if __name__ == "__main__":
    main()
