"""Numbers for the final comparison images (research/FINAL_COMPARE.md), read from runs/final_compare.json at build
time; nothing typed by hand.

    python3 demo/images/redesign/export_final.py  ->  numbers_final.json + numbers_final.js (window.NF = {...})

Same entry shape as numbers_single.json: value, shown (rounded half away from zero), dec, scope, source, path. A system
that was not measured gets value null and shown "not measured" (with the reason in scope).
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = "runs/final_compare.json"


def rnd(x, dec):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def get(d, path):
    for k in path.split(" > "):
        d = d[k]
    return d


def main():
    R = json.loads((ROOT / SRC).read_text())
    out = {}

    def put(key, path, dec, scope, scale=1.0):
        try:
            v = get(R, path)
        except (KeyError, TypeError):
            v = None
        if v is None:
            out[key] = {"value": None, "shown": "not measured", "dec": dec, "scope": scope, "source": SRC, "path": path}
            return
        v = round(float(v) * scale, 6)
        out[key] = {"value": v, "shown": rnd(v, dec), "dec": dec, "scope": scope, "source": SRC, "path": path}

    # ---- words: WER %, Whisper normalizer
    asr = {"ours_115m": "core_115m", "ours_0p6b": "core_0p6b", "whisper_small": "whisper_small",
           "whisper_turbo": "whisper_turbo", "whisper_large": "whisper_large_v3", "parakeet_tdt": "tdt_v3"}
    for st, label in (("libri", "LibriSpeech test-clean first 200"), ("ami", "AMI dev 200 segments"),
                      ("icsi", "ICSI dev 200 segments")):
        for k, s in asr.items():
            put(f"final/wer/{st}/{k}", f"words > {st} > whisper_norm|all > {s} > wer_pct", 1, f"WER %, {label}, Whisper normalizer")
    for sub, label in (("all", "live"), ("user_channel", "live_user")):
        for k, s in asr.items():
            put(f"final/wer/{label}/{k}", f"words > live > whisper_norm|{sub} > {s} > wer_pct", 1,
                f"WER %, 32 live two-party sessions ({sub}), every word, Whisper normalizer")
    for st in ("libri", "ami", "icsi"):
        put(f"final/wer/{st}/nemotron35", f"words_nemotron35 > {st} > whisper_norm > wer_pct", 1,
            "WER %, Nemotron 3.5 ASR 0.6B (multilingual), en-US prompt (runs/core_3p5.json via final_compare)")
    for c in ("115m", "0p6b"):
        put(f"final/sttlat/p50/ours_{c}", f"stt_latency > {c} > p50_ms", 0, "streaming word latency p50 (ms), AMI test windows")
    # ---- speech detection
    vad = {"ours_115m": "core_115m", "ours_0p6b": "core_0p6b", "silero": "silero_v5", "marblenet": "marblenet_v2",
           "pyannote": "pyannote_seg3", "ten_vad": "ten_vad"}
    for sn, lab in (("ami_dev", "ami"), ("icsi_dev", "icsi")):
        for k, s in vad.items():
            put(f"final/vad/{lab}/f1/{k}", f"vad > {sn} > {s} > f1_at_0.5", 3, f"VAD F1 at 0.5, {sn}")
            put(f"final/vad/{lab}/auc/{k}", f"vad > {sn} > {s} > auc", 3, f"VAD ROC-AUC, {sn}")
            put(f"final/vad/{lab}/miss/{k}", f"vad > {sn} > {s} > miss_at_fpr_7.5_pct", 1, f"speech missed at 7.5 % false alarms, {sn}")
    # ---- turn taking
    base = {"pipecat": "pipecat_smartturn_v3.2_silero", "livekit": "livekit_en_turn_detector_silero",
            "parakeet_eou": "parakeet_realtime_eou"}
    for k, s in base.items():
        for m, q, d in (("acc", "accuracy_pct", 1), ("p50", "p50", 0), ("p95", "p95", 0), ("ff", "false_fire_pct", 1),
                        ("missed", "missed_pct", 1)):
            put(f"final/asst/{m}/{k}", f"turn > {s} > asst > {q}", d, "smart-turn v3.2 test, 399 assistant-directed clips")
        for ds in ("calls", "ami"):
            for m, q, d in (("p50", "eot_total_ms_p50", 0), ("p95", "eot_total_ms_p95", 0),
                            ("fi", "false_interruption_pct", 1), ("missed", "missed_pct", 1)):
                put(f"final/{ds}/{m}/{k}", f"turn > {s} > {ds} > {q}", d, f"end of turn, {ds}")
    put("final/asst/acc/smartturn_alone", "turn > smartturn_classifier_alone > asst > accuracy_pct", 1,
        "smart-turn v3.2 classifier alone, one call per whole clip (upper bound, no timing)")
    for c in ("115m", "0p6b"):
        for preset in ("balanced", "fast", "assistant"):
            sfx = "" if preset == "assistant" else f"_{preset}"
            for m, q, d in (("acc", "accuracy_pct", 1), ("p50", "p50", 0), ("p95", "p95", 0),
                            ("ff", "false_fire_pct", 1), ("missed", "missed_pct", 1)):
                put(f"final/asst/{m}/ours_{c}{sfx}", f"turn > ours_{c} > {preset} > asst > {q}", d,
                    f"audioforge {c} --turn-preset {preset}, assistant clips")
            for ds in ("calls", "ami"):
                for m, q, d in (("p50", "eot_total_ms_p50", 0), ("p95", "eot_total_ms_p95", 0),
                                ("fi", "false_interruption_pct", 1), ("missed", "missed_pct", 1)):
                    put(f"final/{ds}/{m}/ours_{c}_{preset}", f"turn > ours_{c} > {preset} > {ds} > {q}", d,
                        f"audioforge {c} --turn-preset {preset}, {ds}")
    # ---- speaker
    for corpus in ("ami", "icsi"):
        for k, path in (("ours_115m", f"speaker > twer > {corpus} > core115_tsvad_d2 > twer"),
                        ("ours_0p6b", f"speaker > twer > {corpus} > core0p6b_tsvad_d2 > twer"),
                        ("nemotron3", f"speaker > twer > {corpus} > nemotron3_best > twer"),
                        ("nemotron3_0p6bwords", f"speaker > twer > {corpus} > 0p6b_nemotron3_best > twer"),
                        ("pyannote31", f"speaker > twer > {corpus} > pyannote31_best > twer"),
                        ("nofilter", f"speaker > twer > {corpus} > none > twer"),
                        ("oracle", f"speaker > twer > {corpus} > oracle_d2 > twer")):
            put(f"final/twer/{corpus}/{k}", path, 1, f"target-speaker WER %, {corpus} eot-bench windows, primary, 5 s print")
        for k in ("ours_115m", "ours_0p6b", "nemotron3", "pyannote31"):
            put(f"final/der/{corpus}/{k}", f"speaker > frame > {corpus} > {k} > der_pct", 1, f"target-speaker DER %, {corpus}")
            put(f"final/trackf1/{corpus}/{k}", f"speaker > frame > {corpus} > {k} > f1", 3, f"tracking F1, {corpus}")
        for k, s in (("ours_115m", "core_115m"), ("ours_0p6b", "core_0p6b"), ("titanet", "titanet_l"),
                     ("wespeaker", "wespeaker_pyannote")):
            put(f"final/eer/{corpus}/{k}", f"speaker_eer > {corpus} > {s} > eer_within_meeting_pct", 1,
                f"within-meeting speaker EER %, {corpus} dev, 200 segments")
    # ---- language
    for k, s in (("ours_115m", "core_115m"), ("ours_0p6b", "core_0p6b"), ("whisper_small", "whisper_small"),
                 ("whisper_turbo", "whisper_turbo"), ("whisper_large", "whisper_large_v3"), ("ambernet", "ambernet")):
        put(f"final/lid/2s/{k}", f"lid > {s} > acc_2s_pct", 1, "FLEURS-17 test, 2 s from speech onset")
        put(f"final/lid/full/{k}", f"lid > {s} > acc_full_pct", 1, "FLEURS-17 test, full clip")
    # ---- cost
    for c in ("115m", "0p6b"):
        for dev in ("mps", "cpu"):
            put(f"final/cost/{c}/{dev}/chunk_ms", f"cost > {c} > engine > {dev} > chunk_ms_p50", 1,
                f"ms per 160 ms chunk, full single-mode engine, {dev}")
            put(f"final/cost/{c}/{dev}/streams", f"cost_summary > {c} > {dev} > realtime_streams", 0,
                f"real-time streams, {dev}")
            put(f"final/cost/{c}/{dev}/rss_mb", f"cost > {c} > engine > {dev} > peak_rss_mb", 0, f"peak RSS MB, {dev}")
    (HERE / "numbers_final.json").write_text(json.dumps(out, indent=1))
    (HERE / "numbers_final.js").write_text("// generated by export_final.py from runs/final_compare.json; do not edit\n"
                                           "window.NF = " + json.dumps(out) + ";\n")
    n_none = sum(v["value"] is None for v in out.values())
    print(f"{len(out)} entries, {n_none} not measured -> numbers_final.json")


if __name__ == "__main__":
    main()
