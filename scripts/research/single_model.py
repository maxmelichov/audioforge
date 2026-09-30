"""research/SINGLE_MODEL.md: single-model mode (``audioforge-serve --mode single``) measured, and experiment A1.

A1: a faster turn rule on the target (TS-VAD) track for two-party calls. The live 69-session Pipecat test
(runs/e2e_tsvad.json, system T = hybrid_dyn on the TS-VAD track) missed 45 % of user turn ends within 3 s against 35 %
for the product default C (1 s timeout on a diarizer column). ``replay`` runs the exact server session
(``audioforge.serve.Session``, the single-mode engine: stage1_served.afm, TS-VAD columns, no diarizer loaded, the LID
head) in-process over E2E_FINAL's 37 clips (scripts/research/e2e_tsvad.py rebuilt them; same pads) with the user's
stored 5 s voice print, for one turn-rule variant at a time; the server's ``turn_end`` audio times plus the measured
Pipecat delivery offset stand in for the response moments, which ``score`` scores with E2E_FINAL's scorer against C's
stored live records (paired clip bootstrap). The offline T row is checked against T's live records first (calibration).

Variants (VARIANTS): T (hybrid_dyn at TSVAD_DYN, the live T), to1000 (plain 1 s timeout on P(target)), to800,
hy800 (head OR 0.8 s timeout on P(target)), hy1000 (= the live Th), dyn40 / dyn25 (hybrid_dyn with the Silero-silence
cap T0 lowered from 75 to 40 / 25 frames).

    W=/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/single_model.py replay --variant T --budget 540
    PYTHONPATH=. .venv/bin/python scripts/research/single_model.py score
    PYTHONPATH=. .venv/bin/python scripts/research/single_model.py table        # -> runs/single_model.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SR = 16000
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad")
OUT = ROOT / "runs" / "single_model.json"
PADS = {"ami": 7.0, "turnbench": 6.0, "oto": 6.0}  # e2e_final.PADS
# variant -> (turn_policy, timeout_ms, DYN_T0 override or None, DYN_A override or None). hybrid_dyn waits
# clamp(T0 - A p, 2, T0) + offset frames of Silero silence (p = the head's posterior; served T0 75, A 55: 6.3 s at
# p = 0, 1.9 s at p = 1). dyn40 / dyn25 lower the cap T0 with the slope kept (the p = 1 wait shrinks to the 2-frame
# floor); dynC40 / dynC30 lower the cap and keep the p = 1 wait at 20 frames (A = T0 - 20); dynF12 / dynF8 cap 25 / 20
# frames with a 12 / 8-frame wait at p = 1
VARIANTS = {"T": ("hybrid_dyn", 1000, None, None), "to1000": ("timeout", 1000, None, None),
            "to800": ("timeout", 800, None, None), "hy800": ("hybrid", 800, None, None),
            "hy1000": ("hybrid", 1000, None, None), "dyn40": ("hybrid_dyn", 1000, 40.0, None),
            "dyn25": ("hybrid_dyn", 1000, 25.0, None), "dynC40": ("hybrid_dyn", 1000, 40.0, 20.0),
            "dynC30": ("hybrid_dyn", 1000, 30.0, 10.0), "dynF12": ("hybrid_dyn", 1000, 25.0, 13.0),
            "dynF8": ("hybrid_dyn", 1000, 20.0, 12.0),
            "CN": ("timeout", 1000, None, None)}  # the product default replayed (Nemotron-3, E2E_FINAL's CN flags)
GROUPS = [("ami", "mono"), ("turnbench", "mono"), ("turnbench", "user"), ("oto", "mono"), ("oto", "user")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def single_engine(threads=2, afm=None, **kw):
    """The --mode single engine (no diarizer loaded) with Silero available for the hybrid_dyn variants (``afm``: the
    ASR + heads checkpoint, default the measured v1 runs/stage1_served.afm; ``kw``: further Engine options, e.g.
    turn_model="smartturn")."""
    from audioforge.serve import Engine
    from audioforge.server.cli import MODES
    opts = {**MODES["single"], "enroll": "explicit", "tsvad": str(ROOT / "runs" / "tsvad_spk.pt"),
            "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True, **kw}
    return Engine.load(str(afm or ROOT / "runs" / "stage1_served.afm"), None, "cpu", threads=threads, **opts)


def default_engine(threads=2):
    """E2E_FINAL's system CN: Nemotron-3-Diarization (max pool, frame-local encoder, 4 columns), --enroll dominant."""
    from audioforge.serve import Engine
    return Engine.load(str(ROOT / "runs" / "stage1_served.afm"), str(ROOT / "runs" / "nemo_nemotron3_diar.afm"), "cpu",
                       threads=threads, diar_pool="max", diar_left=1, diar_spks=4, shed_diar="hold")


def cmd_replay(a):
    import torch
    from e2e_final import _wav_read

    import audioforge.serve as S
    torch.set_num_threads(2)
    work = Path(a.work)
    policy, tms, t0_override, a_override = VARIANTS[a.variant]
    if t0_override is not None:
        S.DYN_T0 = t0_override  # the Session reads the module constants (research only; not a server option)
    if a_override is not None:
        S.DYN_A = a_override
    refs = json.loads((work / "clips.json").read_text())
    prints = json.loads((work / "prints.json").read_text())
    out = work / "replay" / f"{a.variant}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(x)["clip"], json.loads(x)["cond"]) for x in out.read_text().splitlines()} if out.exists() else set()
    eng = default_engine() if a.variant == "CN" else single_engine()
    eng.warmup()
    t_start = time.time()
    n = 0
    for r in refs:
        for cond in (("mono",) if r["set"] == "ami" else ("mono", "user")):
            if a.sets and r["set"] not in a.sets.split(","):
                continue
            if a.conds and cond not in a.conds.split(","):
                continue
            if (r["name"], cond) in done:
                continue
            if time.time() - t_start > a.budget:
                log(f"budget: {n} sessions this call; rerun")
                return
            x = _wav_read(work / "clips" / f"{r['name']}.{cond}.wav")
            x = np.concatenate([x, np.zeros(int(PADS[r["set"]] * SR), np.float32)])
            s = S.Session(eng, S.SessionConfig(turn_policy=policy, timeout_ms=tms))
            emb = (prints.get(r["name"]) or {}).get("5.0")
            if emb is not None and a.variant != "CN":
                s.arm_enrollment("enroll", 0, embedding=emb["embedding"])
            msgs, tp = [], time.perf_counter()
            blk = SR * 160 // 1000
            for i in range(0, len(x), blk):
                msgs += s.process(x[i:i + blk])
            msgs += s.finish()
            wall = time.perf_counter() - tp
            te = [{"t": m["t"], "policy": m["policy"]} for m in msgs if m["type"] == "turn_end"]
            fin = [{"t": m["t"], "text": m["text"]} for m in msgs if m["type"] == "final"]
            part = [m["t"] for m in msgs if m["type"] == "partial"]
            st = [m for m in msgs if m["type"] == "stats"][-1]
            rec = {"clip": r["name"], "set": r["set"], "cond": cond, "variant": a.variant, "policy": policy,
                   "timeout_ms": tms, "dyn_t0": t0_override, "dyn_a": a_override, "has_print": emb is not None, "turn_ends": te,
                   "finals": fin, "first_partial_t": part[0] if part else None, "audio_s": round(len(x) / SR, 3),
                   "wall_s": round(wall, 2), "stats": {k: st.get(k) for k in ("rtf", "peak_rss_mb", "first_partial_ms")}}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            log(f"{a.variant} {r['name']} {cond}: {len(te)} turn ends, rtf {wall / (len(x) / SR):.3f}")
    log(f"replay {a.variant}: all sessions done")


# --------------------------------------------------------------------------- scoring
def _live_records(work: Path, system: str) -> dict:
    out = {}
    for p in sorted((work / "runs").glob(f"pipecat_{system}_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[(r["clip"], r["cond"])] = r
    return out


def delivery_offset(work: Path) -> float:
    """Median (Pipecat response time - server turn_end audio time) over T's live records with a matching end."""
    d = []
    for r in _live_records(work, "T").values():
        te = [e["t"] for e in r["raw"].get("server_turn_ends", [])]
        for t in r["raw"]["responses"]:
            c = [t - x for x in te if 0 <= t - x < 0.5]
            if c:
                d.append(min(c))
    return float(np.median(d)) if d else 0.075


def _scores(work: Path, off: float) -> dict:
    """per[(set, cond, system, clip)] = score_clip dict for the offline variants, the live T / Th and the stored C / D /
    CN / DN Pipecat records."""
    import e2e_final as E2E
    refs = {r["name"]: r for r in json.loads((work / "clips.json").read_text())}
    per = {}
    for p in sorted((work / "replay").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            raw = {"responses": [round(e["t"] + off, 3) for e in r["turn_ends"]], "audio_s": r["audio_s"],
                   "finals": r["finals"], "first_text_t": r["first_partial_t"]}
            per[(r["set"], r["cond"], "off_" + r["variant"], r["clip"])] = E2E.score_record(
                {"raw": raw, "cond": r["cond"]}, refs[r["clip"]])
    for sy in ("T", "Th", "S"):
        for (clip, cond), r in _live_records(work, sy).items():
            per[(r["set"], cond, "live_" + sy, clip)] = E2E.score_record(r, refs[clip])
    ef = json.loads((ROOT / "runs" / "e2e_final.json").read_text())
    for k, v in ef["per_clip"].items():
        st, cond, fw, sy, clip = k.split("|")
        if fw == "pipecat" and sy in ("A", "C", "D", "CN", "DN"):
            per[(st, cond, "live_" + sy, clip)] = v
    return per


def _pooled_over(per, system, keys):
    from audioforge.e2e_metrics import pooled
    sc = [per[(s, c, system, k)] for s, c, k in keys if (s, c, system, k) in per]
    m = pooled(sc) if sc else {}
    if sc:
        m["cut_ins_per_session"] = round(m["cut_ins"] / len(sc), 3)
    return m


def _paired(per, x, y, keys, n_boot=2000, seed=0):
    from audioforge.e2e_metrics import pooled
    kk = [k for k in keys if (k[0], k[1], x, k[2]) in per and (k[0], k[1], y, k[2]) in per]
    A = [per[(s, c, x, k)] for s, c, k in kk]
    B = [per[(s, c, y, k)] for s, c, k in kk]
    rng = np.random.default_rng(seed)
    out = {"n_sessions": len(kk)}
    for stat in ("missed_3s", "missed_6s", "cut_ins", "dead_air_ms_median"):
        def val(sc, stat=stat):
            m = pooled(sc)
            return m["cut_ins"] / len(sc) if stat == "cut_ins" else m[stat]
        if not kk:
            continue
        a_, b_ = val(A), val(B)
        bs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(kk), len(kk))
            va, vb = val([A[i] for i in idx]), val([B[i] for i in idx])
            if va is not None and vb is not None:
                bs.append(va - vb)
        out[stat] = {"a": a_, "b": b_, "delta": None if a_ is None or b_ is None else round(a_ - b_, 4),
                     "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
    return out


def cmd_score(a):
    work = Path(a.work)
    off = delivery_offset(work)
    per = _scores(work, off)
    systems = sorted({k[2] for k in per})
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "delivery_offset_s": round(off, 3), "table": {}, "paired": {}}
    scopes = {"all69": GROUPS, "two_party_user": [("turnbench", "user"), ("oto", "user")],
              "two_party_mono": [("turnbench", "mono"), ("oto", "mono")], "ami": [("ami", "mono")]}
    clips = sorted({k[3] for k in per})
    for scope, groups in scopes.items():
        keys = [(s, c, k) for s, c in groups for k in clips if any((s, c, sy, k) in per for sy in systems)]
        for sy in systems:
            m = _pooled_over(per, sy, keys)
            if m:
                res["table"][f"{scope}|{sy}"] = m
        for x in [s for s in systems if s.startswith(("off_", "live_T", "live_Th", "live_S"))]:
            for y in ("live_C", "live_CN", "live_D", "live_DN", "off_CN", "off_T", "live_T"):
                res["paired"][f"{scope}|{x}-{y}"] = _paired(per, x, y, keys)
        res["paired"][f"{scope}|off_T-live_T"] = _paired(per, "off_T", "live_T", keys)
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev["a1"] = res
    OUT.write_text(json.dumps(prev, indent=1, default=float))
    for k, v in res["table"].items():
        if k.split("|")[0] in ("all69", "two_party_user"):
            log(k, {q: v.get(q) for q in ("n_clips", "dead_air_ms_median", "missed_3s", "missed_6s", "cut_ins_per_session")})


# --------------------------------------------------------------------------- the one table
def _j(path):
    return json.loads((ROOT / path).read_text())


def _six(sysd):
    s = sysd["6s"].get("fixed_5pct_turn_fc", sysd["6s"])
    return {"miss": s["miss_rate"], "ci": s.get("miss_rate_ci"), "fc": s["fc_rate"], "open_miss": s["strata"]["open"]["miss_rate"]}


def _live(work: Path, system: str, framework: str = "pipecat") -> dict:
    """Pooled live metrics of a system over the 69 sessions: stored records (e2e_final / e2e_tsvad runs)."""
    import e2e_final as E2E
    refs = {r["name"]: r for r in json.loads((work / "clips.json").read_text())}
    if system.startswith("e2e_final:"):
        ef = _j("runs/e2e_final.json")
        sy = system.split(":")[1]
        ks = [k for k in ef["per_clip"] if k.split("|")[2] == framework and k.split("|")[3] == sy]
        sc = [ef["per_clip"][k] for k in ks]
        from audioforge.e2e_metrics import pooled
        m = pooled(sc)
        # first words on the two-party clips only (the rebuilt AMI clips of e2e_tsvad carry no onset), so every row
        # of the table is over the same 64 sessions
        ft = [ef["per_clip"][k]["first_text_ms_after_onset"] for k in ks if k.split("|")[0] != "ami"
              and ef["per_clip"][k].get("first_text_ms_after_onset") is not None]
        we, wn = sum(v.get("wer_err", 0) for v in sc), sum(v.get("wer_n", 0) for v in sc)
        tab = [v for k, v in ef["table"].items() if k.split("|")[2] == framework and k.split("|")[3] == sy]
        rt = [v["server_rtf_median"] for v in tab if v.get("server_rtf_median") is not None]
        rss = [v["rss_mb_server_peak"] for v in tab if v.get("rss_mb_server_peak") is not None]
        return {**m, "cut_ins_per_session": round(m["cut_ins"] / len(sc), 3), "wer": round(we / wn, 4) if wn else None,
                "first_text_ms_median": round(float(np.median(ft))) if ft else None,
                "server_rtf_median": round(float(np.median(rt)), 3) if rt else None,
                "rss_mb_server_peak": max(rss) if rss else None}
    items = []
    run_dirs = [work / "runs"] + ([work / "runs_single"] if (work / "runs_single").exists() else [])
    for d in run_dirs:
        for p in sorted(d.glob(f"pipecat_{system}_*.jsonl")):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                items.append((r, E2E.score_record(r, refs[r["clip"]])))
    if not items:
        return {}
    m = E2E.group_metrics(items)
    m["cut_ins_per_session"] = round(m["cut_ins"] / len(items), 3)
    return m


def cmd_table(a):
    work = Path(a.work)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    imp, bt, bti = _j("runs/improve_115m.json"), _j("runs/baselines_turn.json"), _j("runs/baselines_turn_icsi.json")
    ami, icsi = imp["turn_bench"]["ami"]["systems"], imp["turn_bench"]["icsi"]["systems"]
    bami = bt["C_extended_windows_stream"]["systems"]
    bicsi = bti["C_extended_windows_stream"]["crossfit_icsi"]["systems"]
    fa, lid, fasr, hy = imp["frame"], _j("runs/lid.json"), _j("runs/final_asr.json")["results"], _j("runs/hybrid_asr.json")
    rows = {}
    rows["turn_offline_ami"] = {
        "what": "turn-end misses, AMI dev n = 974, 6 s horizon, <= 5 % per-turn false cut-offs (cross-fitted)",
        "single": {"hybrid_dyn": _six(ami["hybrid_dyn_tsvad_spk"]), "hybrid": _six(ami["hybrid_tsvad_spk"]),
                   "head": _six(ami["head_tsvad_spk"]), "timeout": _six(ami["timeout_tsvad_spk"])},
        "default": {"timeout (Sortformer v2 1.04 s column; Nemotron-3 offline not measured)": _six(bami["timeout_stream_causal_dominant"]),
                    "hybrid (shipped rule)": _six(bami["hybrid_stream_causal_dominant"])},
        "pipecat": {"smart-turn v3 + Silero timeout": _six(bami["smartturn_silero+timeout"])},
        "livekit": {"LiveKit EOU (text, en) + timeout": _six(bami["livekit_text_en_asr+timeout"])},
        "source": "runs/improve_115m.json turn_bench.ami; runs/baselines_turn.json"}
    rows["turn_offline_icsi"] = {
        "what": "turn-end misses, held-out ICSI n = 1312, 6 s, <= 5 % per-turn false cut-offs (cross-fitted)",
        "single": {"hybrid_dyn": _six(icsi["hybrid_dyn_tsvad_spk"]), "hybrid": _six(icsi["hybrid_tsvad_spk"]),
                   "head": _six(icsi["head_tsvad_spk"]), "timeout": _six(icsi["timeout_tsvad_spk"])},
        "default": {"timeout (Sortformer v2 column)": _six(bicsi["timeout_stream_causal_dominant"]),
                    "hybrid (shipped rule)": _six(bicsi["hybrid_stream_causal_dominant"])},
        "pipecat": {"Silero timeout (smart-turn not run on ICSI)": _six(bicsi["silero_timeout"])},
        "livekit": None, "source": "runs/improve_115m.json turn_bench.icsi; runs/baselines_turn_icsi.json"}
    live = {"single_T (hybrid_dyn, e2e_tsvad)": _live(work, "T"), "single_Th (hybrid 1 s, e2e_tsvad)": _live(work, "Th"),
            "default_CN (Nemotron-3, timeout)": _live(work, "e2e_final:CN"), "default_C (Sortformer v2, timeout)": _live(work, "e2e_final:C"),
            "pipecat_default_A": _live(work, "e2e_final:A"), "livekit_default_B": _live(work, "e2e_final:B", "livekit"),
            "default_CN_livekit": _live(work, "e2e_final:CN", "livekit")}
    for sy in ("S",):  # the A1 winner, live, true single mode (no diarizer loaded)
        m = _live(work, sy)
        if m:
            live[f"single_{sy} (A1 rule, --mode single, live)"] = m
    rows["live_69"] = {"what": "live, 37 clips / 69 sessions through Pipecat (LiveKit where named), 1x, CPU 2 threads",
                       "systems": {k: {q: v.get(q) for q in ("n_clips", "missed_3s", "missed_6s", "cut_ins_per_session",
                                                            "dead_air_ms_median", "first_text_ms_median", "wer",
                                                            "server_rtf_median", "rss_mb_server_peak")} for k, v in live.items()},
                       "source": "runs/e2e_final.json per_clip; scratch/e2e_tsvad/runs*/ (scored with e2e_final.score_record)"}
    rows["wer"] = {"what": "WER, normalize_text, n = 200 each",
                   "single": {"ami_streaming": fasr["served/ami"]["wer_normalize_text"]["wer"],
                              "libri_streaming": fasr["served/libri"]["wer_normalize_text"]["wer"]},
                   "default": {"ami_tdt_v3_per_turn": hy["english"]["results"]["tdt_v3/ami"]["wer_normalize_text"]["wer"],
                               "libri_tdt_v3": hy["english"]["results"]["tdt_v3/libri"]["wer_normalize_text"]["wer"]},
                   "pipecat_livekit": {"whisper_small_ami": fasr["whisper_small/ami"]["wer_normalize_text"]["wer"],
                                       "whisper_small_libri": fasr["whisper_small/libri"]["wer_normalize_text"]["wer"]},
                   "source": "runs/final_asr.json, runs/hybrid_asr.json english"}
    s = lid["systems"]
    rows["lid"] = {"what": "FLEURS-17 test (n = 2550), accuracy at 2 s / full utterance",
                   "single": [s["lid_distill_vadgated"]["2s"]["acc"], s["lid_distill_vadgated"]["full"]["acc"]],
                   "default": "off (opt-in --lid ambernet: " + f"{s['ambernet']['2s']['acc']:.3f} / {s['ambernet']['full']['acc']:.3f})",
                   "pipecat_livekit": "not part of their default stacks (Whisper small, language fixed to en)",
                   "source": "runs/lid.json systems"}
    rows["tracking"] = {"what": "speaker-tracking F1 of the user's track (primary = target, 5 s print), all frames",
                        "single": {c: fa[c]["primary"]["tsvad_spk_vp5p0"]["all"]["f1"] for c in ("ami", "icsi")},
                        "default": {c: {"sortformer_bound_by_print_spk": fa[c]["primary"]["sortformer_vp_spk_vp5p0"]["all"]["f1"],
                                        "sortformer_bound_by_print_titanet": fa[c]["primary"]["sortformer_vp_titanet_vp5p0"]["all"]["f1"],
                                        "sortformer_oracle_column": fa[c]["primary"]["sortformer_oracle_column"]["all"]["f1"]}
                                    for c in ("ami", "icsi")},
                        "note": "Nemotron-3 columns not scored this way; Pipecat / LiveKit defaults track no speaker",
                        "source": "runs/improve_115m.json frame (research/IMPROVE_115M.md A.1)"}
    res["table"] = rows
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"-> {OUT}")
    print(json.dumps(rows["live_69"], indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("replay", "score", "table"):
        s = sub.add_parser(n)
        s.add_argument("--work", default=str(WORK))
        if n == "replay":
            s.add_argument("--variant", required=True, choices=list(VARIANTS))
            s.add_argument("--budget", type=float, default=540)
            s.add_argument("--sets", default="")
            s.add_argument("--conds", default="")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
