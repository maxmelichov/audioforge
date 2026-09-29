"""Numbers for the single-model images (architecture_single / results_single, EXPLAINER.md), read from the run files at
build time; nothing typed by hand.

    python3 demo/images/redesign/export_single.py  ->  numbers_single.json + numbers_single.js (window.NS = {...})

Each entry: value, shown (the one string on the image, rounded half away from zero), dec, scope, source, path. A
missing path raises. Entries shared with the v5 results image are copied from demo/v5/shots/numbers_b.json (which
export_b.py builds from the same runs), so the two images cannot drift apart.
"""
from __future__ import annotations

import glob
import json
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
NB = json.loads((ROOT / "demo/v5/shots/numbers_b.json").read_text())
CONDS = ["ami|mono", "turnbench|mono", "turnbench|user", "oto|mono", "oto|user"]   # the 69 live sessions (37 clips)


def load(rel):
    return json.loads((ROOT / rel).read_text())


def get(d, path):
    for k in path.split(" > "):
        d = d[k]
    return d


def rnd(x, dec):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def main():
    out = {}

    def put(key, value, dec, scope, source, path):
        out[key] = {"value": value, "shown": rnd(value, dec), "dec": dec, "scope": scope, "source": source, "path": path}

    # 1. target-speaker tracking F1 (primary speaker, all frames), 5 s voice print
    imp = load("runs/improve_115m.json")
    for ds in ("icsi", "ami"):
        base = f"frame > {ds} > primary"
        p = f"{base} > tsvad_spk_vp5p0 > all > f1"
        put(f"find/{ds}/ours", get(imp, p), 2, "target-speaker F1, audioforge TS-VAD head (single model), 5 s print",
            "runs/improve_115m.json", p)
        cands = {b: get(imp, f"{base} > sortformer_vp_{b}_vp5p0 > all > f1") for b in ("spk", "titanet")}
        best = max(cands, key=cands.get)
        put(f"find/{ds}/sortformer", cands[best], 2,
            f"target-speaker F1, NVIDIA Streaming Sortformer v2 column bound by the same 5 s print (best of the two binders: {best})",
            "runs/improve_115m.json", f"max({base} > sortformer_vp_spk_vp5p0 / sortformer_vp_titanet_vp5p0 > all > f1)")
    # 2. missed turn ends, single model with the print vs the two-model stack without it; 3. VAD; 6. WER (copied)
    for k in ("ts/ami/before", "ts/ami/after", "ts/icsi/before", "ts/icsi/after", "ts/ami/before_fc", "ts/ami/after_fc",
              "ts/icsi/before_fc", "ts/icsi/after_fc", "ts/ami/n", "ts/icsi/n", "vad/ours", "vad/marblenet", "vad/silero",
              "vad/n", "wer/ours", "wer/whisper_small", "wer/n", "version/pipecat", "version/livekit"):
        out[k] = dict(NB[k])
    # 4. cut-ins per live session, pooled over the same 69 Pipecat sessions
    tv, fin = load("runs/e2e_tsvad.json")["table"], load("runs/e2e_final.json")["table"]
    for key, tab, fmt, name in (("T", tv, "{c}|T", "audioforge single model (--turn-input tsvad --diar-off, hybrid_dyn)"),
                                ("C", fin, "{c}|pipecat|C", "audioforge two-model default (Sortformer v2, timeout)"),
                                ("A", fin, "{c}|pipecat|A", "Pipecat default stack")):
        rows = [tab[fmt.format(c=c)] for c in CONDS]
        n = sum(r["n_clips"] for r in rows)
        src = "runs/e2e_tsvad.json" if tab is tv else "runs/e2e_final.json"
        put(f"live/{key}/cutins", sum(r["cut_ins"] for r in rows) / n, 2, f"cut-ins per session, {name}", src,
            "sum(table > " + fmt.format(c="<set|cond>") + " > cut_ins) / sum(n_clips), over " + ", ".join(CONDS))
        if key in ("T", "C"):
            ends = sum(r["n_ends"] for r in rows)
            put(f"live/{key}/missed3s", 100 * sum(r["missed_3s"] * r["n_ends"] for r in rows) / ends, 0,
                f"% of user turns not answered within 3 s, {name}", src, "n_ends-weighted table > missed_3s over the 5 conditions")
    out["live/sessions"] = {"value": sum(tv[c + "|T"]["n_clips"] for c in CONDS), "scope": "live Pipecat sessions",
                            "source": "runs/e2e_tsvad.json", "path": "sum(table > <set|cond>|T > n_clips)"}
    out["live/sessions"]["shown"] = str(out["live/sessions"]["value"])
    nclips = len(load("runs/e2e_final.json")["clips"])
    out["live/clips"] = {"value": nclips, "shown": str(nclips), "scope": "distinct clips behind the live sessions (rebuilt identically for T)",
                         "source": "runs/e2e_final.json", "path": "len(clips)"}
    # 5. server real-time factor, CPU 2 threads: single model (69 raw session records) vs the two-model default
    rt = []
    for f in sorted(glob.glob(str(SSD / "scratch/e2e_tsvad/runs/pipecat_T_*.jsonl"))):
        for line in open(f):
            s = (json.loads(line).get("raw") or {}).get("server_stats") or {}
            if "rtf" in s:
                rt.append(s["rtf"])
    assert len(rt) == 69, len(rt)
    put("rtf/T", statistics.median(rt), 2, "server real-time factor, single model, median of 69 live sessions",
        "ssd/scratch/e2e_tsvad/runs/pipecat_T_*.jsonl", "median(raw > server_stats > rtf)")
    put("rtf/C", statistics.median([fin[c + "|pipecat|C"]["server_rtf_median"] for c in CONDS]), 2,
        "server real-time factor, two-model default, median of the five per-set medians", "runs/e2e_final.json",
        "median(table > <set|cond>|pipecat|C > server_rtf_median)")
    # not shown on the images, quoted in EXPLAINER.md
    hy = load("runs/hybrid_asr.json")
    p = "english > results > served/ami > wer_normalize_text > wer"
    put("wer/stream_ami", 100 * get(hy, p), 0, "% word errors of the streaming transcript on AMI meetings", "runs/hybrid_asr.json", p)
    # runs/single_model.json (research/SINGLE_MODEL.md): A1 offline replay of the 32 two-party sessions, A2 print length
    sm = load("runs/single_model.json")
    A1 = "a1 > table > all69|off_"
    for key, row, name in (("served", "T", "served single-model rule (hybrid_dyn, served wait)"),
                           ("tuned", "dynF12", "tuned rule --dyn-wait-ms 2000,960"),
                           ("default", "CN", "two-model default (Nemotron-3, timeout 1 s), replayed")):
        r = sm["a1"]["table"]["all69|off_" + row]
        put(f"a1/{key}/missed3s", 100 * r["missed_3s"], 1, f"% of user turns not answered within 3 s, {name}, offline replay",
            "runs/single_model.json", A1 + row + " > missed_3s")
        put(f"a1/{key}/cutins", r["cut_ins_per_session"], 2, f"cut-ins per session, {name}, offline replay",
            "runs/single_model.json", A1 + row + " > cut_ins_per_session")
    tu = sm["a1"]["table"]["all69|off_dynF12"]
    assert abs(tu["missed_3s"] - 0.1009) < 1e-4 and abs(tu["cut_ins_per_session"] - 1.062) < 1e-3   # SINGLE_MODEL.md: the 2000,960 row
    out["a1/sessions"] = {"value": tu["n_clips"], "shown": str(tu["n_clips"]), "scope": "two-party sessions in the offline replay",
                          "source": "runs/single_model.json", "path": A1 + "dynF12 > n_clips"}
    assert not any(k.startswith("all69|live_S") for k in sm["a1"]["table"])   # no live confirmation of the tuned rule yet
    for ds, tk in (("ami", "turn_offline_ami"), ("icsi", "turn_offline_icsi")):
        t = sm["table"][tk]
        put(f"done2/{ds}/single", 100 * t["single"]["hybrid_dyn"]["miss"], 0, "% missed turn ends, single model (hybrid_dyn on the TS-VAD track, 5 s print)",
            "runs/single_model.json", f"table > {tk} > single > hybrid_dyn > miss")
        put(f"done2/{ds}/single_fc", 100 * t["single"]["hybrid_dyn"]["fc"], 1, "% false cut-offs, single model", "runs/single_model.json", f"table > {tk} > single > hybrid_dyn > fc")
        put(f"done2/{ds}/default", 100 * t["default"]["hybrid (shipped rule)"]["miss"], 0, "% missed turn ends, two-model stack (shipped hybrid)",
            "runs/single_model.json", f"table > {tk} > default > hybrid (shipped rule) > miss")
        put(f"done2/{ds}/default_fc", 100 * t["default"]["hybrid (shipped rule)"]["fc"], 1, "% false cut-offs, two-model stack", "runs/single_model.json", f"table > {tk} > default > hybrid (shipped rule) > fc")
        d = float(out[f"done2/{ds}/default"]["shown"]) - float(out[f"done2/{ds}/single"]["shown"])
        out[f"done2/{ds}/delta"] = {"value": d, "shown": rnd(d, 0), "dec": 0, "source": "runs/single_model.json",
                                    "scope": "points fewer missed turn ends = shown default - shown single", "path": f"done2/{ds}/default.shown - done2/{ds}/single.shown"}
        tr = sm["a2"]["turn"][ds]
        for L, key in (("1.5", "vp1p5"), ("3", "vp3p0"), ("5", "vp5p0 (spk)"), ("10", "vp10p0"), ("live5", "arm5p0")):
            put(f"a2/{ds}/{L}", 100 * tr[key]["hybrid_dyn"]["miss"], 1, f"% missed turn ends, voice print {L} s (hybrid_dyn)",
                "runs/single_model.json", f"a2 > turn > {ds} > {key} > hybrid_dyn > miss")
    # results_v6: NVIDIA Nemotron-3 bound by the same 5 s print (best of the two binders per corpus; commit 8b685fa)
    for ds in ("icsi", "ami"):
        base = f"frame > {ds} > primary"
        cands = {bd: get(imp, f"{base} > nemotron3_vp_{bd}_vp5p0 > all > f1") for bd in ("spk", "titanet")}
        best = max(cands, key=cands.get)
        put(f"find/{ds}/nemotron3", cands[best], 2, f"target-speaker F1, NVIDIA Nemotron-3 column bound by the same 5 s print (best binder: {best})",
            "runs/improve_115m.json", f"max({base} > nemotron3_vp_spk_vp5p0 / nemotron3_vp_titanet_vp5p0 > all > f1)")
    # results_v5: external systems only
    bt = load("runs/baselines_turn.json")["C_extended_windows_stream"]["systems"]
    for key, sysn, name in (("eou", "eou_posterior", "NVIDIA Parakeet-Realtime-EOU"), ("smartturn", "smartturn_silero+timeout", "Pipecat smart-turn v3 + Silero timeout"),
                            ("silero", "silero_timeout", "Silero timeout"), ("livekit", "livekit_text_en_asr+timeout", "LiveKit turn detector (text, en) + timeout")):
        pth = f"C_extended_windows_stream > systems > {sysn} > 6s > fixed_5pct_turn_fc"
        r = get(load("runs/baselines_turn.json"), pth)
        put(f"ext/ami/{key}", 100 * r["miss_rate"], 1, f"% missed turn ends, AMI, {name}, <= 5 % per-turn FC", "runs/baselines_turn.json", pth + " > miss_rate")
    pth = "C_extended_windows_stream > crossfit_icsi > systems > silero_timeout > 6s > fixed_5pct_turn_fc"
    put("ext/icsi/silero", 100 * get(load("runs/baselines_turn_icsi.json"), pth)["miss_rate"], 1, "% missed turn ends, ICSI, Silero timeout", "runs/baselines_turn_icsi.json", pth + " > miss_rate")
    for ds in ("ami", "icsi"):   # single-model hybrid_dyn at the same budget, one decimal for the ranked lists
        tk = f"turn_offline_{ds}"
        put(f"ext/{ds}/ours", 100 * sm["table"][tk]["single"]["hybrid_dyn"]["miss"], 1, f"% missed turn ends, {ds.upper()}, audioforge single (hybrid_dyn, 5 s stored print)",
            "runs/single_model.json", f"table > {tk} > single > hybrid_dyn > miss")
    ev = load("runs/e2e_tsvad.json")["table"]["oto|user|T"]
    put("live2/ours/unanswered", 100 * ev["missed_3s"], 0, "% of user turns not answered within 3 s, live oto user channel, audioforge single (served hybrid_dyn rule)",
        "runs/e2e_tsvad.json", "table > oto|user|T > missed_3s")
    for key, sysn in (("livekit", "livekit|B"), ("pipecat", "pipecat|A")):
        put(f"live2/{key}/unanswered", 100 * fin[f"oto|user|{sysn}"]["missed_3s"], 0, f"% not answered within 3 s, live oto user, {key} default", "runs/e2e_final.json", f"table > oto|user|{sysn} > missed_3s")
    rows = [fin[c + "|livekit|B"] for c in CONDS]
    put("live/B/cutins", sum(r["cut_ins"] for r in rows) / sum(r["n_clips"] for r in rows), 2, "cut-ins per session, LiveKit default, same 69 sessions (its own runs)",
        "runs/e2e_final.json", "sum(table > <set|cond>|livekit|B > cut_ins) / sum(n_clips)")
    fa = load("runs/final_asr.json")
    for key, k in (("turbo", "whisper_turbo"), ("ctc", "parakeet")):
        pth = f"results > {k}/ami > wer_whisper_norm > wer"; put(f"wer/{key}", 100 * get(fa, pth), 1, f"% word errors, offline, {k}", "runs/final_asr.json", pth)
    # results_v6: first words on screen, live (research/SINGLE_MODEL.md: median over the 64 two-party sessions)
    L = sm["table"]["live_69"]["systems"]
    for key, row, name in (("ours", "single_S (A1 rule, --mode single, live)", "audioforge single (--mode single rule), live"),
                           ("pipecat", "pipecat_default_A", "Pipecat default"), ("livekit", "livekit_default_B", "LiveKit default")):
        put(f"first/{key}", L[row]["first_text_ms_median"] / 1000, 2, f"s from the user's first onset to the first words on screen (median), {name}",
            "runs/single_model.json", f"table > live_69 > systems > {row} > first_text_ms_median")
    # results_v6 (shipped --mode single rule on all 69 live sessions)
    for key, row in (("ours", "single_S (A1 rule, --mode single, live)"), ("pipecat", "pipecat_default_A"), ("livekit", "livekit_default_B")):
        base = f"table > live_69 > systems > {row}"
        put(f"s69/{key}/missed3s", 100 * L[row]["missed_3s"], 1, f"% of user turns not answered within 3 s, all 69 live sessions, {key}", "runs/single_model.json", base + " > missed_3s")
        put(f"s69/{key}/cutins", L[row]["cut_ins_per_session"], 2, f"interruptions per call, all 69 live sessions, {key}", "runs/single_model.json", base + " > cut_ins_per_session")
    # results_v7 "Hears speech": speech missed at a false-alarm rate of about 7.5 %, AMI dev 64 windows (FINAL_REPORT section 2)
    vs = load("runs/vad_single.json")["eval"]["ami_dev"]["L3"]["at_fpr0.075"]
    put("vad2/ours/miss", 100 * vs["miss"], 1, f"% speech frames missed at FPR {vs['fpr']}, shipped block-4 VAD head (L3)", "runs/vad_single.json",
        "eval > ami_dev > L3 > at_fpr0.075 > miss")
    put("vad2/ours/fpr", 100 * vs["fpr"], 1, "false-alarm rate of that point, %", "runs/vad_single.json", "eval > ami_dev > L3 > at_fpr0.075 > fpr")
    bsd = load("runs/baselines_sd.json")["vad"]
    for key, sysn in (("marblenet", "marblenet_frame_vad_v2"), ("silero", "silero_v5.1.2")):
        sw = bsd[sysn]["sweep"]; thr = min(sw, key=lambda t: abs(sw[t]["fpr"] - 0.075))
        assert abs(sw[thr]["fpr"] - vs["fpr"]) < 0.002, (sysn, sw[thr]["fpr"])   # same operating point, or stop
        put(f"vad2/{key}/miss", 100 * sw[thr]["miss"], 1, f"% speech frames missed at FPR {sw[thr]['fpr']} (sweep threshold {thr}), {sysn}", "runs/baselines_sd.json",
            f"vad > {sysn} > sweep > {thr} > miss")
    put("vad2/rel", 100 * (1 - out["vad2/ours/miss"]["value"] / out["vad2/marblenet/miss"]["value"]), 0,
        "% less missed speech than NVIDIA MarbleNet at the same false-alarm rate (relative)", "runs/vad_single.json + runs/baselines_sd.json", "1 - vad2/ours/miss / vad2/marblenet/miss")
    # results_v7 secondary lines (relative, from the raw values)
    put("v7/missed_rel", 100 * (1 - out["ext/ami/ours"]["value"] / out["ext/ami/eou"]["value"]), 0,
        "% fewer missed turn ends than NVIDIA Parakeet-Realtime-EOU, AMI (relative)", "runs/single_model.json + runs/baselines_turn.json", "1 - ext/ami/ours / ext/ami/eou")
    put("v7/first_x", out["first/pipecat"]["value"] / out["first/ours"]["value"], 1,
        "times faster to first words than Pipecat default", "runs/single_model.json", "first/pipecat / first/ours")
    put("v7/cut_rel", 100 * (1 - out["s69/ours/cutins"]["value"] / out["s69/livekit/cutins"]["value"]), 0,
        "% fewer interruptions per call than LiveKit default (relative)", "runs/single_model.json", "1 - s69/ours/cutins / s69/livekit/cutins")
    # derived, relative or absolute, each with its formula (v3 results cards)
    for ds in ("icsi", "ami"):   # points of missed turn ends removed: difference of the two values as shown (raw in scope)
        b_, a_ = out[f"ts/{ds}/before"], out[f"ts/{ds}/after"]
        d = float(b_["shown"]) - float(a_["shown"])
        out[f"done/{ds}/delta"] = {"value": d, "shown": rnd(d, 0), "dec": 0, "source": b_["source"] + " + " + a_["source"],
                                   "scope": f"points fewer missed turn ends = shown before - shown after (raw difference {b_['value'] - a_['value']:.2f})",
                                   "path": f"ts/{ds}/before.shown - ts/{ds}/after.shown"}
    put("cut/rel", 100 * (1 - out["live/T/cutins"]["value"] / out["live/A/cutins"]["value"]), 0,
        "% fewer interruptions per call than Pipecat's default (relative)", "runs/e2e_tsvad.json + runs/e2e_final.json", "1 - live/T/cutins / live/A/cutins")
    assert out["rtf/T"]["value"] / out["rtf/C"]["value"] < 0.5   # the card says "less than half"
    put("wer/rel", 100 * (1 - out["wer/ours"]["value"] / out["wer/whisper_small"]["value"]), 0,
        "% fewer word errors than Whisper small (relative: (14.4 - 9.5) / 14.4)", "runs/hybrid_asr.json + runs/final_asr.json", "1 - wer/ours / wer/whisper_small")
    put("wer/pp", out["wer/whisper_small"]["value"] - out["wer/ours"]["value"], 1,
        "percentage points fewer word errors (absolute)", "runs/hybrid_asr.json + runs/final_asr.json", "wer/whisper_small - wer/ours")
    (HERE / "numbers_single.json").write_text(json.dumps(out, indent=1))
    (HERE / "numbers_single.js").write_text("// generated by export_single.py from the run files; do not edit\nwindow.NS = " + json.dumps(out) + ";\n")
    for k, v in out.items():
        print(f"{k:24s} {str(v.get('shown', v['value'])):>6s}  {v['source']}")


if __name__ == "__main__":
    main()
