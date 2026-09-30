"""The live voice-agent metrics of research/METRICS.md, per test condition, in industry-standard terms -> runs/metrics.json.

No model runs here: it re-scores the stored live records with the E2E_FINAL scorer (audioforge/e2e_metrics.py).
  audioforge single (S)  <scratch>/e2e_tsvad/runs/pipecat_S_*.jsonl  (Pipecat 1.12, --mode single, stored 5 s print)
  audioforge room (CN)   runs/e2e_final.json per_clip (Pipecat and LiveKit)
  Pipecat default (A)    runs/e2e_final.json per_clip (Silero + smart-turn v3 + Whisper small, in Pipecat)
  LiveKit default (B)    runs/e2e_final.json per_clip (Silero + LiveKit EOU + Whisper small, in LiveKit Agents)

Conditions: `user` = the user's own channel of 32 two-party calls (16 TurnBench + 16 otoSpeech), the realistic input
(the agent hears only the user); `mono` = the same 32 calls as one mixed channel, where the recorded partner answers
in the same audio (no system gets silence to fire on; a test artifact); `ami` = 5 AMI meeting windows; `all` = all 69.

Metrics per condition (one definition each; research/METRICS.md):
  eot_ms_p50 / p90    end-of-turn latency: user turn end (reference) -> the framework receives the end-of-turn decision
                      (stub LLM/TTS answer at once), over ends answered within 6 s
  answered_pct        % of user turn ends that get a response before the user speaks again (no time limit)
  late_or_missed_3s   % of user turn ends with no response within 3 s (missed or late)
  false_int_per_min   false interruptions: responses while the user is still mid-turn (speech or a within-turn pause),
                      per minute of audio; also per session
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
TSV = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad")
COND = {"user": [("turnbench", "user"), ("oto", "user")], "mono": [("turnbench", "mono"), ("oto", "mono")],
        "ami": [("ami", "mono")]}
COND["all"] = COND["user"] + COND["mono"] + COND["ami"]


def turns_cut(c: dict, ref: dict) -> tuple[int, int]:
    """(user turns with at least one response inside the turn span, user turns): the scorer's cut-in rule per turn."""
    turns = ref["user_turns"]
    return sum(any(s - 1e-9 <= t < e - 1e-9 for t in c.get("cut_in_t", [])) for s, e in turns), len(turns)


def final_latency(rec: dict, ref: dict) -> list[float]:
    """Per scored user turn end e: the first non-empty final received at or after e (before the user's next turn),
    minus e, in ms. Finals are the framework-received times of the server's `final` messages."""
    fin = sorted(f["t"] for f in rec["raw"].get("finals", []) if f.get("text", "").strip())
    turns, out = ref["user_turns"], []
    for i, (_s, e) in enumerate(turns):
        if not ref["scored"][i]:
            continue
        nxt = turns[i + 1][0] if i + 1 < len(turns) else float("inf")
        t = next((f for f in fin if e - 1e-9 <= f < nxt), None)
        if t is not None and t - e <= 6.0:
            out.append((t - e) * 1000)
    return out


def summarize(clips: list[dict], refs: dict) -> dict:
    ends = [e for c in clips for e in c["ends"]]
    tc = [turns_cut(c, refs[c["_clip"]]) for c in clips]
    da = [e["dead_air_ms"] for e in ends if e["dead_air_ms"] is not None and not e["missed_6s"]]
    mins = sum(c["audio_s"] for c in clips) / 60
    ci = sum(c["cut_ins"] for c in clips)
    return {"n_sessions": len(clips), "n_turn_ends": len(ends), "audio_min": round(mins, 1),
            "eot_ms_p50": round(float(np.median(da))) if da else None,
            "eot_ms_p90": round(float(np.percentile(da, 90))) if da else None,
            "eot_ms_p95": round(float(np.percentile(da, 95))) if da else None,
            "false_int_pct_of_turns": round(100 * sum(a for a, _ in tc) / max(1, sum(b for _, b in tc)), 1),
            "n_user_turns": sum(b for _, b in tc),
            "answered_pct": round(100 * float(np.mean([e["response"] is not None for e in ends])), 1),
            "answered_within_6s_pct": round(100 * float(np.mean([not e["missed_6s"] for e in ends])), 1),
            "late_or_missed_3s_pct": round(100 * float(np.mean([e["missed_3s"] for e in ends])), 1),
            "false_int_per_min": round(ci / mins, 2), "false_int_per_session": round(ci / len(clips), 2)}


def main():
    import e2e_final as E2E
    refs = {r["name"]: r for r in json.loads((TSV / "clips.json").read_text())}
    per, fl_S = {}, {}
    for p in sorted((TSV / "runs").glob("pipecat_S_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            per[(r["set"], r["cond"], "pipecat", "S", r["clip"])] = E2E.score_record(r, refs[r["clip"]])
            fl_S.setdefault(r["cond"] if r["set"] != "ami" else "ami", []).extend(final_latency(r, refs[r["clip"]]))
    ef = json.loads((ROOT / "runs" / "e2e_final.json").read_text())
    for k, v in ef["per_clip"].items():
        st, cond, fw, sy, clip = k.split("|")
        if sy in ("A", "B", "CN"):
            per[(st, cond, fw, sy, clip)] = v
    for k, v in per.items():
        v["_clip"] = k[4]
    systems = {"audioforge_single|pipecat": ("pipecat", "S"), "audioforge_room|pipecat": ("pipecat", "CN"),
               "audioforge_room|livekit": ("livekit", "CN"), "pipecat_default|pipecat": ("pipecat", "A"),
               "livekit_default|livekit": ("livekit", "B")}
    out = {"generated_by": "scripts/research/metrics_table.py", "definitions": __doc__, "live": {}}
    for name, (fw, sy) in systems.items():
        for cond, groups in COND.items():
            clips = [v for (st, cd, f, s, _), v in per.items() if f == fw and s == sy and (st, cd) in groups]
            if clips:
                out["live"][f"{name}|{cond}"] = summarize(clips, refs)
    out["final_latency_single"] = {
        c: {"n": len(v), "p50_ms": round(float(np.median(v))), "p95_ms": round(float(np.percentile(v, 95)))}
        for c, v in fl_S.items() if v}
    out["final_latency_single"]["definition"] = (
        "user turn end (reference) -> the framework receives the server's final transcript of that turn (Pipecat, "
        "--mode single); the final is sent with the turn_end, so this is the end-of-turn latency seen as text; "
        "the default stacks' records keep no final times: not measured for them")
    # compute: CPU seconds per audio second of the whole server process (the unit the default stacks were measured in)
    cpu, rtf, chunk = [], [], []
    for p in sorted((TSV / "runs").glob("pipecat_S_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            cpu.append(r["usage"]["server"]["cpu_s"] / r["raw"]["total_s"])
            rtf.append(r["raw"]["server_stats"]["rtf"])
            chunk.append(r["raw"]["server_stats"]["chunk_ms_p50"])
    cl = ef["components_live"]
    out["compute"] = {
        "audioforge_single": {"cpu_s_per_audio_s": round(float(np.median(cpu)), 3), "wall_rtf": round(float(np.median(rtf)), 3),
                              "chunk_ms_p50_cpu2": round(float(np.median(chunk)), 1), "n_sessions": len(cpu),
                              "note": "server process, Apple M5 CPU, 2 threads, live Pipecat sessions"},
        "pipecat_default": {"cpu_s_per_audio_s": cl["pipecat:A"]["_process"]["driver_cpu_s_per_audio_s"],
                            "note": "the whole Pipecat process (Silero + smart-turn + Whisper small), same machine"},
        "livekit_default": {"cpu_s_per_audio_s": cl["livekit:B"]["_process"]["driver_cpu_s_per_audio_s"],
                            "note": "the whole LiveKit agent process (Silero + EOU + Whisper small), same machine"},
        "whisper_small_call_ms": {"pipecat_p50": cl["pipecat:A"]["stt_whisper"]["call_ms_p50"],
                                  "pipecat_p95": cl["pipecat:A"]["stt_whisper"]["call_ms_p95"],
                                  "livekit_p50": cl["livekit:B"]["stt_whisper"]["call_ms_p50"],
                                  "livekit_p95": cl["livekit:B"]["stt_whisper"]["call_ms_p95"]},
    }
    # end-of-turn precision / recall / F1 on TurnBench dev (official scorer counts: tp/fn over the 1904 turn ends,
    # fp over the 1063 within-turn pauses; every system at its FP-rate <= 0.10 point on this set)
    tb = json.loads((ROOT / "runs" / "turnbench_dev.json").read_text())

    def prf(sc):
        p_, r_ = sc["tp"] / (sc["tp"] + sc["fp"]), sc["tp"] / (sc["tp"] + sc["fn"])
        return {"precision": round(p_, 3), "recall": round(r_, 3), "f1": round(2 * p_ * r_ / (p_ + r_), 3),
                "fp_rate": sc["fp_rate"], "latency_ms_p50": (sc.get("latency_ms") or sc).get("p50"),
                "latency_ms_p90": (sc.get("latency_ms") or sc).get("p90")}
    tl = json.loads((ROOT / "runs" / "turnbench_latency.json").read_text())["families"]["fixed"]["crossfit_pooled_heldout"]
    out["turnbench_eot"] = {
        "audioforge_turn_head_fixed_policy_heldout (thresholds fitted on the other half of TurnBench dev, pooled; "
        "turn head + Silero on the user's channel, not the exact --mode single input)": prf(tl),
        "audioforge_turn_head_hybrid_in_sample (thresholds picked on TurnBench dev itself)":
            prf(tb["systems"]["hybrid"]["operating_point"]["score"]),
        "silero_timeout": prf(tb["systems"]["silero_timeout"]["operating_point"]["score"]),
        "nvidia_parakeet_realtime_eou_posterior": prf(tb["systems"]["eou_posterior"]["operating_point"]["score"]),
        "pipecat_smart_turn_v3 (published dev predictions, rescored)": prf(tb["published_dev_rescored"]["smart_turn_v3"]),
        "vap (published dev predictions, rescored)": prf(tb["published_dev_rescored"]["vap"]),
        "livekit_turn_detector": "not measured (text model; no TurnBench run)"}
    # target-speaker DER: frame level (80 ms), no collar, overlap scored, from the stored frame counts of
    # IMPROVE_115M A.1 / A.1b: (missed target frames + false-alarm frames) / target frames
    fr = json.loads((ROOT / "runs" / "improve_115m.json").read_text())["frame"]
    tsder = {}
    for c in ("ami", "icsi"):
        for k in ("tsvad_spk_vp5p0", "nemotron3_vp_spk_vp5p0", "nemotron3_vp_titanet_vp5p0", "nemotron3_oracle_column"):
            a = fr[c]["primary"][k]["all"]
            tsder[f"{c}|{k}"] = {"ts_der_pct": round(100 * (a["miss"] * a["target_frames"] + a["fa"] * a["nontarget_frames"])
                                                     / a["target_frames"], 1), "f1": a["f1"], "miss": a["miss"],
                                 "fa_rate": a["fa"], "target_frames": a["target_frames"]}
    out["target_speaker_der"] = tsder
    (ROOT / "runs" / "metrics.json").write_text(json.dumps(out, indent=1))
    for k, v in out["live"].items():
        print(f"{k:32s} {v}")
    print(json.dumps(out["compute"], indent=1))


if __name__ == "__main__":
    main()
