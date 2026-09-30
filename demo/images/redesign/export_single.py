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
import re
import statistics
import subprocess
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


def load_committed(rel):
    """A run file as committed at HEAD (not the working tree), with the commit that last touched it. Used for the tWER
    files: the working tree can hold another agent's uncommitted re-run made with uncommitted code."""
    blob = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"])
    h = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", rel]).decode().strip()
    return json.loads(blob), f"{rel} @ {h} (committed)"


def v8(put, out):
    """results_v8: turn-taking on calls and in meetings (EOT latency p50, false-interruption rate, missed turn ends),
    target-speaker WER, VAD F1 and miss rate at a fixed false-alarm rate, compute per 160 ms chunk, and the footer WER
    line. Every key is v8/...; every entry names its file and path."""
    # 1-2. turn-taking, same turn ends for every system (runs/eot_latency.json table)
    eot = load("runs/eot_latency.json")
    ours = eot["selection"]["fastest_print_fix_goal_no_clip_cut"]   # the shipped vad_head rule (EOT_LATENCY.md, constants.py; commit 6ad219a)
    assert ours.startswith("(g) head theta 0.99: ourVAD<0.4 sil>=160ms & p>=0.99 | fallback 640ms | others"), ours
    systems = (("ours", ours), ("livekit", "LiveKit EnglishModel + Silero (defaults)"),
               ("pipecat", "Pipecat smart-turn v3.2 + Silero (defaults)"))
    for ds, tk in (("calls", "two_party_user"), ("ami", "ami")):
        for key, row in systems:
            r = eot["table"][tk][row]
            base = f"table > {tk} > {row}"
            put(f"v8/{ds}/{key}/eot", r["eot_total_ms_p50"], 0, f"end-of-turn latency p50, ms, {tk}, {row}", "runs/eot_latency.json", base + " > eot_total_ms_p50")
            put(f"v8/{ds}/{key}/fi", r["false_interruption_pct"], 0, f"false-interruption rate, % of user turns, {tk}, {row}", "runs/eot_latency.json", base + " > false_interruption_pct")
            put(f"v8/{ds}/{key}/miss", r["missed_pct"], 0, f"missed turn ends, % of turn ends, {tk}, {row}", "runs/eot_latency.json", base + " > missed_pct")
        n = eot["table"][tk][ours]["n_turns"]
        assert all(eot["table"][tk][row]["n_turns"] == n for _, row in systems)   # the same turn ends for every system
        put(f"v8/{ds}/n", n, 0, f"reference turn ends scored, {tk}", "runs/eot_latency.json", f"table > {tk} > <rule> > n_turns")
        put(f"v8/{ds}/sessions", eot["n_sessions"]["two_party_user" if ds == "calls" else "ami"], 0, f"sessions / windows, {tk}", "runs/eot_latency.json", f"n_sessions > {tk}")
    # 3. target-speaker WER (the user's words only), committed run files
    tl, tl_src = load_committed("runs/tswer_live.json")
    for key, arm in (("none", "none"), ("ours", "tsvad_d2"), ("oracle", "oracle_d2")):
        put(f"v8/twer/calls/{key}", tl["results"]["mono"][arm]["wer"], 0, f"tWER %, the 16 two-party mono-mix sessions (both voices in one channel), arm {arm}",
            tl_src, f"results > mono > {arm} > wer")
    put("v8/twer/calls/n", tl["results"]["mono"]["n_sessions"], 0, "mono-mix sessions scored", tl_src, "results > mono > n_sessions")
    tw, tw_src = load_committed("runs/tswer.json")
    arms = tw["results"]["icsi"]["primary"]["arms"]
    put("v8/twer/icsi/ours", arms["tsvad_d2"]["twer"], 0, "tWER %, ICSI held-out, primary speaker, audioforge TS-VAD filter (5 s print)", tw_src, "results > icsi > primary > arms > tsvad_d2 > twer")
    put("v8/twer/icsi/none", arms["none"]["twer"], 0, "tWER %, ICSI held-out, primary speaker, no filter (every word)", tw_src, "results > icsi > primary > arms > none > twer")
    n3 = {b: arms[f"n3_{b}_d2"]["twer"] for b in ("spk", "tn")}
    best = min(n3, key=n3.get)   # the better of the two binders for NVIDIA's system
    put("v8/twer/icsi/nemotron3", n3[best], 0, f"tWER %, ICSI, NVIDIA Nemotron-3-Diarization column bound by the same 5 s print (better binder: {best}; spk {n3['spk']}, TitaNet {n3['tn']})",
        tw_src, f"min(results > icsi > primary > arms > n3_spk_d2 / n3_tn_d2 > twer)")
    # 4. VAD, AMI dev 64 x 20 s windows: F1 at the 0.5 threshold and the miss rate at the same ~7.5 % false-alarm rate
    va = load("runs/vad_auc.json")
    for key, k in (("ours", "audioforge_block4_head"), ("silero", "silero_v5.1.2"), ("marblenet", "marblenet_frame_vad_v2")):
        put(f"v8/vad/{key}/f1", va[k]["f1_at_0.5"], 3, f"VAD F1 at threshold 0.5, AMI dev, {k}", "runs/vad_auc.json", f"{k} > f1_at_0.5")
    vs = load("runs/vad_single.json")["eval"]["ami_dev"]["L3"]
    assert abs(vs["f1"] - va["audioforge_block4_head"]["f1_at_0.5"]) < 1e-4   # the two files agree on the shipped head
    for key in ("ours", "silero", "marblenet"):
        out[f"v8/vad/{key}/miss"] = dict(out[f"vad2/{key}/miss"])
    put("v8/vad/fpr", 100 * vs["at_fpr0.075"]["fpr"], 1, "false-alarm rate of the fixed operating point, %", "runs/vad_single.json", "eval > ami_dev > L3 > at_fpr0.075 > fpr")
    put("v8/vad/n", load("runs/baselines_sd.json")["vad"]["n"], 0, "AMI dev windows (20 s)", "runs/baselines_sd.json", "vad > n")
    # 6. cost: compute per 160 ms chunk, the full single-mode engine (Session.chunk_ms) on the bundled clip, 2 threads, CPU and
    # the Mac GPU from the same run (runs/mps_115m.json, research/MPS_115M.md); the RTX 5090 figure as cited in runs/
    mp = load("runs/mps_115m.json")
    for key, dev, name in (("cpu", "cpu", "Apple M5 CPU, 2 threads"), ("mps", "mps", "Apple M5 GPU (MPS)")):
        put(f"v8/cost/{key}", mp["engine"][dev]["chunk_ms_p50"], 1, f"ms of compute per 160 ms chunk, p50, full --mode single engine on the bundled clip, {name}",
            "runs/mps_115m.json", f"engine > {dev} > chunk_ms_p50")
    note = load("runs/stt_latency.json")["gpu_estimate"]["note"]
    m = re.search(r"5090's (\d+) ms/chunk \((research/GPU_RUN_[\d-]+\.md), PR #1\)", note)
    assert m, note
    put("v8/cost/gpu", int(m.group(1)), 0, f"ms per 160 ms chunk on an RTX 5090, measured in PR #1 ({m.group(2)} on that branch; no GPU run file in runs/ locally)",
        "runs/stt_latency.json", "gpu_estimate > note (cites PR #1)")
    # footer: word accuracy of the 115M streaming model
    hy = load("runs/hybrid_asr.json")
    for key, k in (("ours", "served"), ("whisper_small", "whisper_small")):
        p = f"english > results > {k}/libri > wer_normalize_text > wer"
        put(f"v8/wer/libri/{key}", 100 * get(hy, p), 1, f"WER %, LibriSpeech test-clean 200 utterances, {k}", "runs/hybrid_asr.json", p)
    L = load("runs/single_model.json")["table"]["live_69"]["systems"]
    for key, row in (("ours", "single_S (A1 rule, --mode single, live)"), ("livekit", "livekit_default_B"), ("pipecat", "pipecat_default_A")):
        put(f"v8/wer/live/{key}", 100 * L[row]["wer"], 1, f"WER %, live sessions (the 32 with transcripts), {row}", "runs/single_model.json", f"table > live_69 > systems > {row} > wer")


def v9(put, out):
    """results_v9: the four winning cards of v8 as paired vertical bars (audioforge vs one competitor), each with the
    relative change computed here from the SAME rounded values printed on the bars (the "shown" strings), so a reader
    who divides the two bar labels gets the printed %. The raw-value change is kept in "raw" (and in the scope).
    Never typed. Only improvements are shown, so each must be > 0 or the export stops. Every key is v9/...."""
    def sh(k):
        return float(out[k]["shown"])
    def rel_lower(key, ours, other, what):   # lower is better: % fewer
        o, b = out[ours], out[other]
        r, raw = 100 * (1 - sh(ours) / sh(other)), 100 * (1 - o["value"] / b["value"])
        assert r > 0 and raw > 0, (key, o["value"], b["value"])
        put(key, r, 0, f"% relative reduction, {what}: 1 - {o['shown']} / {b['shown']} (bar labels); raw values 1 - {o['value']} / {b['value']} = {raw:.2f}",
            o["source"] + (" + " + b["source"] if b["source"] != o["source"] else ""), f"1 - {ours}.shown / {other}.shown")
        out[key]["raw"] = raw
    rel_lower("v9/calls/fi_rel", "v8/calls/ours/fi", "v8/calls/livekit/fi", "false interruptions on calls, audioforge vs LiveKit")
    rel_lower("v9/calls/miss_rel", "v8/calls/ours/miss", "v8/calls/livekit/miss", "missed turn ends on calls, audioforge vs LiveKit")
    rel_lower("v9/ami/fi_rel", "v8/ami/ours/fi", "v8/ami/pipecat/fi", "false interruptions in AMI meetings, audioforge vs Pipecat")
    rel_lower("v9/ami/miss_rel", "v8/ami/ours/miss", "v8/ami/pipecat/miss", "missed turn ends in AMI meetings, audioforge vs Pipecat")
    rel_lower("v9/twer/rel", "v8/twer/calls/ours", "v8/twer/calls/none", "target-speaker WER on mono-mix calls, audioforge vs our own STT with no speaker filter")
    # VAD F1, higher is better: % higher. MarbleNet's F1 must agree with the baselines file the task names.
    o, b = out["v8/vad/ours/f1"], out["v8/vad/marblenet/f1"]
    mb = load("runs/baselines_sd.json")["vad"]["marblenet_frame_vad_v2"]["sweep"]["0.5"]["f1"]
    assert abs(mb - b["value"]) < 1e-4, (mb, b["value"])
    r, raw = 100 * (sh("v8/vad/ours/f1") / sh("v8/vad/marblenet/f1") - 1), 100 * (o["value"] / b["value"] - 1)
    assert r > 0 and raw > 0, r
    put("v9/vad/rel", r, 1, f"% relative increase in VAD F1 at 0.5, audioforge vs NVIDIA MarbleNet: {o['shown']} / {b['shown']} - 1 (bar labels); raw {o['value']} / {b['value']} - 1 = {raw:.2f}",
        "runs/vad_auc.json (+ runs/vad_single.json, runs/baselines_sd.json agree)", "v8/vad/ours/f1.shown / v8/vad/marblenet/f1.shown - 1")
    out["v9/vad/rel"]["raw"] = raw


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
    # results_v7 first words in milliseconds (the measured unit), exact
    for key, row in (("ours", "single_S (A1 rule, --mode single, live)"), ("pipecat", "pipecat_default_A"), ("livekit", "livekit_default_B")):
        put(f"firstms/{key}", L[row]["first_text_ms_median"], 0, f"ms from the user's first onset to the first words on screen (median), {key}",
            "runs/single_model.json", f"table > live_69 > systems > {row} > first_text_ms_median")
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
    # ---- results_v8 / architecture_v8 (standard metric names only; research/METRICS.md, EOT_LATENCY.md, TSWER.md) ----
    v8(put, out)
    # ---- results_v9 (the user's mock-up: paired bars, relative change per chart) ----
    v9(put, out)
    (HERE / "numbers_single.json").write_text(json.dumps(out, indent=1))
    (HERE / "numbers_single.js").write_text("// generated by export_single.py from the run files; do not edit\nwindow.NS = " + json.dumps(out) + ";\n")
    for k, v in out.items():
        print(f"{k:24s} {str(v.get('shown', v['value'])):>6s}  {v['source']}")


if __name__ == "__main__":
    main()
