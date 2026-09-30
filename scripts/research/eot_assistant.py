"""research/EOT_ASSISTANT.md: end-of-turn speed and accuracy on assistant-directed speech (smart-turn's own test clips).

Data: the 399 pipecat-ai/human_5_all clips (BSD-2-Clause) in smart-turn-data-v3.2-test (data/smartturn, the split of
audioforge.datasets.smartturn.split_indices; 175 complete, 224 incomplete). Each clip is one utterance spoken to a
voice assistant that ends either at a real end ("complete") or at a mid-utterance cut ("incomplete"). Every clip is
padded with PAD_S = 3 s of digital silence so a decision can fire after it.

Reference point e: the clip's audible end (``speech_end``: the later of the last Silero chunk > 0.5 and the energy-VAD
utterance end, clipped to the clip end; median 256 / 427 ms before the clip end on complete / incomplete clips). Not
turn_v4's ``audible_end``: that also reads our served VAD head, whose tail puts the end at the clip end on 90 % of
these clips (it would call Pipecat's correct stops "early"). The decision window is the padded clip, [0, clip end + 3 s).

Systems (the same audio, the same harness as scripts/research/eot_latency.py):
  * ours, balanced: the served --mode single engine (audioforge.serve.Session, stage1_served_v2.afm, turn_policy
    vad_head with its shipped defaults), the clip alone as the user channel, NO voice print (the TS-VAD others path
    needs another talker, and there is none here, so it never fires); its own turn_end messages.
  * ours, fast: the --turn-preset fast rule (VAD < 0.6 for >= 480 ms AND p >= 0.99, OR 720 ms) applied by
    eot_latency.sim_room to the per-frame signals of the same served session (``check``: sim_room(CHOSEN) reproduces
    the served balanced turn_ends exactly, so the frames are what the server reads).
  * the turn_end_hint (p >= 0.8 after 80 ms of VAD silence) of the served balanced session.
  * Pipecat smart-turn v3.2 + Silero (Pipecat 1.12 defaults) and LiveKit EnglishModel + Silero (0.5 / 3.0 s): the
    replay of eot_latency.cmd_baselines unchanged (Silero v5 confidences of the padded clip, LiveKit text = our served
    streaming ASR's text ready at each end of speech).
Times: decision audio time + measured compute (eot_latency's ``total``: ours + the session's chunk p50, Pipecat + the
smart-turn call).

Metrics (``score``):
  complete    EOT latency = first turn_end at or after e - 80 ms minus e (p50 / p95, eot_latency.score_session with
              horizon clip end + 3 s); missed = no turn_end in [e - 80 ms, clip end + 3 s); early = a turn_end before
              e - 80 ms (the user was still talking); correct = answered and not early
  incomplete  false fire = any turn_end before e + 3 s (smart-turn's sense: the agent takes the turn after a
              mid-utterance stop the user would resume from); split into before the cut and within 0.5 / 1 / 2 / 3 s
              after it (the baselines' own silence timeouts, LiveKit's 3.0 s max delay and Pipecat's 3 s stop_secs,
              sit at the 3 s edge, so read the 2 s column too)
  accuracy    (complete correct + incomplete not fired) / n (smart-turn v3.2's classifier: 97.0 %, runs/smartturn_audit.json)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py dump --budget 540   # until done
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py baselines --budget 540
    PYTHONPATH=. .venv/bin/python scripts/research/eot_assistant.py score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import eot_latency as E  # noqa: E402

SR = 16000
PAD_S = 3.0
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/eot_assistant")
# EOT_ASSISTANT_DUMP / EOT_ASSISTANT_ENGINE (research/EOT_ASSISTANT.md "Energy gate"): dump another served
# configuration elsewhere, e.g. dump_gate (the energy gate, the default since then; dump = before it) or dump_smartturn
# with EOT_ASSISTANT_ENGINE='{"turn_model": "smartturn"}' (Engine options as JSON)
DUMP = Path(os.environ.get("EOT_ASSISTANT_DUMP", WORK / "dump"))
ENGINE_KW = json.loads(os.environ.get("EOT_ASSISTANT_ENGINE", "{}"))
BASE = WORK / "baselines"
OUT = ROOT / "runs" / "eot_assistant.json"
SILERO = ROOT / "data" / "silero" / "silero_vad_v5.onnx"
# --turn-preset fast (TURN_PRESETS["fast"]); eot_latency.FAST once that is committed, the same numbers otherwise
FAST = getattr(E, "FAST", {**E.CHOSEN, "vad_thr": 0.6, "k": 6, "fallback_f": 9, "fu": 8})
AFTER = (0.5, 1.0, 2.0, 3.0)
FF_WINDOW = 3.0  # an incomplete clip is a false fire if a turn_end comes before its audible end + 3 s
log = E.log


def clips() -> list[dict]:
    from audioforge.datasets import smartturn as D
    root = ROOT / "data" / "smartturn"
    meta = json.loads((root / "cache" / "human_5_all.json").read_text())
    sp = D.split_indices(meta, root)
    assert sp["how"].startswith("smart-turn-data-v3.2-test"), sp["how"]
    off = meta["offsets"]
    return [{"key": Path(meta["ids"][i]).stem, "i": i, "a": off[i], "b": off[i + 1], "complete": bool(meta["complete"][i])}
            for i in sp["eval"]]


_WAV = None


def audio(c: dict) -> np.ndarray:
    global _WAV
    if _WAV is None:
        _WAV = np.load(ROOT / "data" / "smartturn" / "cache" / "human_5_all.npy", mmap_mode="r")
    x = np.asarray(_WAV[c["a"]:c["b"]], np.float32)
    return np.concatenate([x, np.zeros(int(PAD_S * SR), np.float32)])


def cmd_dump(a):
    import torch
    from single_model import single_engine

    import audioforge.serve as S
    from audioforge.baselines.turn import SileroStream
    torch.set_num_threads(2)
    DUMP.mkdir(parents=True, exist_ok=True)
    todo = [c for c in clips() if not (DUMP / f"{c['key']}.json").exists()]
    if not todo:
        log("dump: all clips done")
        return
    kw = dict(ENGINE_KW)
    eng = single_engine(afm=kw.pop("afm", E.SERVED_AFM), **kw)  # "afm": e.g. runs/stage1_served_v3.afm (turn head v5)
    eng.warmup()
    sil = SileroStream(SILERO)
    tok = eng.asr.tokenizer
    specials = {tok.token_id(x) for x in getattr(tok, "specials", [])}
    t0, n = time.time(), 0
    for c in todo:
        if time.time() - t0 > a.budget:
            log(f"budget: {n} clips this call, {len(todo) - n} left; rerun")
            return
        x = audio(c)
        s = S.Session(eng, S.SessionConfig(turn_policy="vad_head"))  # no voice print (module doc)
        rec = []
        orig = s.asr.run_turn_on_diar

        def rtod(avail, act_fn, flush=False, _o=orig, _s=s, rec=rec):
            out = _o(avail, act_fn, flush)
            # the served vad_head policy reads P(user) / P(other) only once a print is enrolled (never here)
            tp = _s.asr.tsvad_p if _s.tsvad is not None and _s.tsvad.enrolled else []
            for v, p, vad in out:
                pu, po = (float(tp[v][0]), float(tp[v][1])) if v < len(tp) else (0.0, 0.0)
                rec.append((int(v), round(_s._asr_ready_t(v), 4), float(p), round(float(vad), 5), pu, po))
            return out
        s.asr.run_turn_on_diar = rtod
        msgs = []
        for i in range(0, len(x), 320):  # 20 ms blocks, as a client streams them
            msgs += s.process(x[i:i + 320])
        msgs += s.finish()
        st = sil.new_state()
        conf = [sil.step(st, x[j * 512:(j + 1) * 512]) for j in range(len(x) // 512)]
        cm = np.asarray(list(s.chunk_ms), float)
        d = {"key": c["key"], "complete": c["complete"], "clip_s": round((c["b"] - c["a"]) / SR, 4),
             "audio_s": round(len(x) / SR, 4), "silero_loaded_in_session": s.sil is not None,
             "head": {k: [r[j] for r in rec] for j, k in enumerate(("v", "t", "p", "vad", "pu", "po"))},
             "conf": [round(float(q), 5) for q in conf],
             "tok_at": [int(q) for q in s.asr.tok_at],
             "pieces": ["" if t in specials else tok.sp.id_to_piece(int(t)) for t in s.asr.tokens],
             "asr_clock": {"half": int(s.asr.half), "hop": int(s.asr.hop), "chunk_mel": int(s.asr.chunk_mel),
                           "chunk_lead": int(s.asr.chunk_lead), "cs": int(s.asr.cs)},
             "chunk_ms": {"p50": round(float(np.median(cm)), 2), "p95": round(float(np.percentile(cm, 95)), 2)},
             "turn_ends": [{k: m.get(k) for k in ("t", "policy", "p", "silence_ms", "hinted_at", "path", "model_ms")}
                           for m in msgs if m["type"] == "turn_end"],
             "engine": {"energy_gate": eng.energy_gate, "energy_quiet_db": eng.energy_quiet_db,
                        "turn_model": eng.turn_model},
             "turn_model_calls": list(s.st_trig.calls) if s.st_trig is not None else None,
             "hints": [{k: m.get(k) for k in ("t", "p", "kind")} for m in msgs
                       if m["type"] in ("turn_end_hint", "turn_end_hint_cancel")]}
        (DUMP / f"{c['key']}.json").write_text(json.dumps(d))
        n += 1
        if n % 20 == 0:
            log(f"{n} clips ({time.time() - t0:.0f} s); last {c['key']}: {len(d['turn_ends'])} turn ends")
    log(f"dump: {n} clips this call; all clips done")


def load_dump() -> dict:
    return {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(DUMP.glob("*.json")))}


def speech_end(x: np.ndarray, conf: np.ndarray, clip_s: float) -> float:
    """The clip's audible end, from two VADs that are not the system under test (ours reads its own VAD head, whose
    tail stays > 0.5 to the clip end on 90 % of these clips; research/VAD_TAIL.md): the later of the last 32 ms Silero
    chunk > 0.5 and the energy-VAD utterance end (audioforge.datasets.smartturn.utterance_end_frame, the end the
    smart-turn training labels use), clipped to the clip end."""
    from audioforge.datasets.smartturn import utterance_end_frame
    n = int(round(clip_s * SR)) // 512
    j = np.nonzero(conf[:n] > 0.5)[0]
    ends = [utterance_end_frame(x) * E.FRAME] + ([(j[-1] + 1) * 512 / SR] if len(j) else [])
    return round(float(min(max(ends), clip_s)), 4)


def sessions(dump: dict) -> list[dict]:
    """eot_latency-style sessions: one reference turn [0, audible end], horizon = the padded clip end."""
    out = []
    for c in clips():
        d = dump.get(c["key"])
        if d is None:
            continue
        conf = np.asarray(d["conf"])
        e = speech_end(audio(c)[:c["b"] - c["a"]], conf, d["clip_s"])
        on = np.nonzero(conf[:int(d["clip_s"] * SR) // 512] > 0.5)[0]
        out.append({"key": c["key"], "set": "smartturn", "complete": c["complete"], "clip": c, "clip_s": d["clip_s"],
                    "onset": round(float(on[0]) * 512 / SR, 4) if len(on) else 0.0,
                    "user_turns": [(0.0, e)], "scored": [True], "next_onset": [round(d["clip_s"] + PAD_S, 4)]})
    return out


def cmd_baselines(a):
    """eot_latency.cmd_baselines as is, pointed at these clips (its sessions / audio / dump / output dir)."""
    dump = load_dump()
    sess = sessions(dump)
    E.sessions = lambda: sess
    E.read_audio = lambda ss: audio(ss["clip"])
    E.load_dump = lambda: dump
    E.BASE = BASE
    E.cmd_baselines(a)


def load_baselines() -> dict:
    return {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(BASE.glob("*.json")))}


def _pct(a, b):
    return round(100 * a / b, 1) if b else None


def score_system(sess: list[dict], times: dict, compute: dict, drop_pre: bool = True) -> dict:
    """times[key] = turn_end totals (s); compute[key] = per-time compute (s) or one value. ``drop_pre``: ignore
    turn_ends decided before the clip's speech onset (first Silero chunk > 0.5): nothing has been said, the final is
    empty, and no agent answers it (ours: the served VAD head reads ~0.55 on the first frames of a fresh session, so
    its 640 ms fallback can fire in a clip's leading silence; each clip here is a fresh session)."""
    n_pre = 0
    times, compute = dict(times), dict(compute)
    for s in sess:
        k = s["key"]
        t = np.asarray(times[k], float)
        c = np.broadcast_to(np.asarray(compute[k], float), t.shape)
        keep = t >= s["onset"]
        n_pre += bool((~keep).any())
        if drop_pre:
            times[k], compute[k] = t[keep].tolist(), c[keep].tolist()
    comp = [s for s in sess if s["complete"]]
    inc = [s for s in sess if not s["complete"]]
    per = [E.score_session(times[s["key"]], s, compute[s["key"]]) for s in comp]
    r = E.pool(per)
    correct_c = sum(p["intr"] == 0 and p["miss"] == 0 for p in per)
    fired, before, after, padded = 0, 0, {w: 0 for w in AFTER}, 0
    for s in inc:
        t = np.asarray(times[s["key"]], float)
        e, h = s["user_turns"][0][1], s["next_onset"][0]
        t = t[t < h]
        padded += bool(len(t))
        fired += bool(np.any(t < e + FF_WINDOW))
        if len(t) and t.min() < e - E.TOL:
            before += 1
        for w in AFTER:
            after[w] += bool(np.any(t < e + w))
    from_clip_end = []  # complete clips: first answering turn_end minus the clip end (where the digital pad starts)
    for s in comp:
        t = np.asarray(times[s["key"]], float)
        e, h = s["user_turns"][0][1], s["next_onset"][0]
        t = t[(t >= e - E.TOL) & (t < h)]
        if len(t):
            from_clip_end.append(float(t.min()) - s["clip_s"])
    lat = np.concatenate([np.asarray(p["lat"]) for p in per]) if per else np.zeros(0)
    return {"complete": {"n": len(comp), "eot_total_ms_p50": r["eot_total_ms_p50"], "eot_total_ms_p95": r["eot_total_ms_p95"],
                         "eot_decision_ms_p50": r["eot_decision_ms_p50"], "eot_decision_ms_p95": r["eot_decision_ms_p95"],
                         "answered_within_500ms_pct": _pct(int(np.sum(lat <= 0.5)), len(comp)),
                         "answered_within_1s_pct": _pct(int(np.sum(lat <= 1.0)), len(comp)),
                         "missed_pct": r["missed_pct"], "early_fire_pct": r["false_interruption_pct"],
                         "correct_pct": _pct(correct_c, len(comp)),
                         "eot_from_clip_end_ms_p50": int(round(1000 * float(np.median(from_clip_end))))
                         if from_clip_end else None},
            "incomplete": {"n": len(inc), "false_fire_pct": _pct(fired, len(inc)),
                           "fired_before_cut_pct": _pct(before, len(inc)),
                           "fired_anywhere_in_padded_clip_pct": _pct(padded, len(inc)),
                           **{f"fired_by_cut+{w:g}s_pct": _pct(after[w], len(inc)) for w in AFTER}},
            "accuracy_pct": _pct(correct_c + len(inc) - fired, len(sess)),
            "clips_with_pre_speech_turn_end": n_pre, "pre_speech_turn_ends_dropped": drop_pre}


def score_hint(sess: list[dict], dump: dict, compute: dict) -> dict:
    sent = good = 0
    lat, rec_c, fired_i = [], 0, 0
    for s in sess:
        e, h = s["user_turns"][0][1], s["next_onset"][0]
        ht = [m["t"] + compute[s["key"]] for m in dump[s["key"]]["hints"] if m["kind"] == "hint"]
        ht = [t for t in ht if s["onset"] <= t < h]
        sent += len(ht)
        if s["complete"]:
            ok = [t for t in ht if t >= e - E.TOL]
            good += len(ok)
            if ok:
                rec_c += 1
                lat.append(ok[0] - e)
        else:
            fired_i += bool(ht)
    nc = sum(s["complete"] for s in sess)
    ni = len(sess) - nc
    r = lambda q: int(round(float(np.percentile(lat, q)) * 1000)) if lat else None  # noqa: E731
    return {"hints_sent": sent, "precision_pct": _pct(good, sent),
            "recall_complete_pct": _pct(rec_c, nc), "fired_on_incomplete_pct": _pct(fired_i, ni),
            "latency_total_ms_p50": r(50), "latency_total_ms_p95": r(95),
            "note": "precision = hints in [e - 80 ms, clip end + 3 s) of a complete clip / all hints; a hint on an "
                    "incomplete clip or before a complete clip's end counts against it"}


def cmd_score(a):
    dump, bl = load_dump(), load_baselines()
    sess = sessions(dump)
    assert len(sess) == len(clips()), f"dump incomplete: {len(sess)} / {len(clips())}"
    # check: the offline rule on the recorded frames == the served session's own turn_ends (balanced)
    bad = [k for k, d in dump.items()
           if [round(t, 3) for t, _ in E.sim_room(d, E.CHOSEN)] != [m["t"] for m in d["turn_ends"]]]
    log(f"check: sim_room(CHOSEN) vs served turn_ends: {len(bad)} / {len(dump)} clips differ")
    comp = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    systems = {}
    systems["ours balanced (shipped)"] = score_system(
        sess, {k: [m["t"] + comp[k] for m in d["turn_ends"]] for k, d in dump.items()}, comp)
    systems["ours fast (--turn-preset fast)"] = score_system(
        sess, {k: [t + comp[k] for t, _ in E.sim_room(d, FAST)] for k, d in dump.items()}, comp)
    bnames = {"pipecat_smartturn": "Pipecat smart-turn v3.2 + Silero (defaults)",
              "livekit_eou": "LiveKit EnglishModel + Silero (0.5 / 3.0 s)"}
    if all(s["key"] in bl for s in sess):
        for bk, label in bnames.items():
            tt, cc = {}, {}
            for s in sess:
                ev = bl[s["key"]][bk]
                cc[s["key"]] = [e["compute_ms"] / 1000 if bk == "pipecat_smartturn" else 0.0 for e in ev]
                tt[s["key"]] = [e["t"] + c for e, c in zip(ev, cc[s["key"]])]
            systems[label] = score_system(sess, tt, cc)
    else:
        log(f"baselines incomplete ({sum(s['key'] in bl for s in sess)} / {len(sess)}): rows skipped")
    # which path of our rule answers each clip (first turn_end at or after e - 80 ms, after the speech onset)
    paths = {}
    for name, rule in (("balanced", E.CHOSEN), ("fast", FAST)):
        for kind in ("complete", "incomplete"):
            cnt = {}
            for s in sess:
                if s["complete"] != (kind == "complete"):
                    continue
                e = s["user_turns"][0][1]
                ev = [pth for t, pth in E.sim_room(dump[s["key"]], rule) if t >= max(s["onset"], e - E.TOL)]
                cnt[ev[0] if ev else "none"] = cnt.get(ev[0] if ev else "none", 0) + 1
            paths[f"{name} {kind}"] = cnt
    # the smart-turn classifier alone (one call on the clip, p > 0.5), our harness (runs/smartturn_audit.json)
    audit = json.loads((ROOT / "runs" / "smartturn_audit.json").read_text())["eval_human_5_all"]["eval"]
    moved = np.asarray([s["clip_s"] - s["user_turns"][0][1] for s in sess])
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "data": {"source": "pipecat-ai/human_5_all (BSD-2-Clause), smart-turn-data-v3.2-test ids",
                    "n": len(sess), "n_complete": sum(s["complete"] for s in sess),
                    "pad_s": PAD_S, "clip_s_p50": round(float(np.median([s["clip_s"] for s in sess])), 2),
                    "clip_end_minus_audible_end_ms_p50_p90": [int(round(1000 * float(np.percentile(moved, q))))
                                                              for q in (50, 90)],
                    "other_assistant_directed_sets_local": "none (data/: calls, meetings, TTS dialogue only)"},
           "protocol": __doc__.split("Metrics (``score``):")[0].strip(),
           "voice_print": "none: the clip alone as the user channel, no enrollment; the TS-VAD others path cannot fire",
           "fast_rule": FAST, "check_sim_equals_served_balanced": {"n_clips": len(dump), "n_differ": len(bad)},
           "compute_ms": {"ours_chunk_p50_median": round(1000 * float(np.median(list(comp.values()))), 1)},
           "systems": systems, "ours_answering_path": paths,
           "systems_raw_incl_pre_speech_turn_ends": {
               k: score_system(sess, {kk: [m["t"] + comp[kk] for m in d["turn_ends"]] if "balanced" in k else
                                      [t + comp[kk] for t, _ in E.sim_room(d, FAST)] for kk, d in dump.items()},
                               comp, drop_pre=False) for k in list(systems)[:2]}, "turn_end_hint (balanced session, p >= 0.8)": score_hint(sess, dump, comp),
           "smart_turn_classifier_alone": {"accuracy_pct": audit["accuracy"], "recall_complete_pct": audit["recall_complete"],
                                           "recall_incomplete_pct": audit["recall_incomplete"],
                                           "note": "one smart-turn call on the unpadded clip, p > 0.5 (no VAD, no timing)"}}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for k, v in systems.items():
        c, i = v["complete"], v["incomplete"]
        log(f"{k}: p50 {c['eot_total_ms_p50']} p95 {c['eot_total_ms_p95']} | missed {c['missed_pct']} early "
            f"{c['early_fire_pct']} | FF inc {i['false_fire_pct']} (before cut {i['fired_before_cut_pct']}, "
            f"+1s {i['fired_by_cut+1s_pct']}) | acc {v['accuracy_pct']}")
    log("hint", json.dumps(res["turn_end_hint (balanced session, p >= 0.8)"]))
    log(f"-> {OUT}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "baselines"):
        b = sub.add_parser(name)
        b.add_argument("--budget", type=float, default=540)
    sub.add_parser("score")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
