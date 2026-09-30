"""Audit of the Pipecat smart-turn / LiveKit baseline rows in research/EOT_LATENCY.md (is 2.3 s real or the harness?).

  inputs    sample rate / channels / range / level of the clips, the Silero track vs the audio, the feature extractor
            (transformers vs Pipecat's vendored numpy log-mel) on real turn windows
  replay    Pipecat 1.12's OWN LocalSmartTurnAnalyzerV3 (BaseSmartTurn buffering, pre-speech, 8 s cap, stop_secs
            silence fallback) driven chunk by chunk with the Pipecat VAD state (as TurnAnalyzerUserTurnStopStrategy
            feeds it), under a simulated clock; checks it against eot_latency.py's baselines3; logs per reference end
            the first smart-turn call, its p / decision and how the turn_end fired; sweeps SmartTurnParams.stop_secs
  livekit   the LiveKit row with max_endpointing_delay swept (min 0.5 s)
  eval      smart-turn v3.2 through audioforge.baselines.turn.SmartTurn on smart-turn's own labelled data
            (pipecat-ai/human_5_all: the 402 clips in smart-turn-data-v3.2-test, and a slice of the rest)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/smartturn_audit.py inputs
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/smartturn_audit.py replay
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/smartturn_audit.py livekit
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/smartturn_audit.py eval
Writes runs/smartturn_audit.json (one key per command).
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
import eot_latency as E  # noqa: E402

SR, CH = 16000, 512
CH_S = CH / SR
OUT = ROOT / "runs" / "smartturn_audit.json"
ONNX = ROOT / ".venv/lib/python3.12/site-packages/pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx"


def save(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1))


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


# --------------------------------------------------------------------------- inputs
def cmd_inputs(a):
    import soundfile as sf
    from audioforge.baselines import turn as B
    from pipecat.audio.turn.smart_turn._whisper_features import compute_whisper_log_mel_features
    dump = E.load_dump()
    rows, lv = [], {"calls": [], "ami": []}
    st = B.SmartTurn(ONNX)
    fdiff, pdiff = [], []
    for ss in E.sessions():
        info = sf.info(ss["wav"])
        x = E.read_audio(ss)
        d = dump[ss["key"]]
        conf = np.asarray(d["conf"])
        # Silero track recomputed on this audio (the dump's conf must be the same audio, same chunking)
        sp = B.pipecat_vad(conf)["state"] >= 2
        speech = np.concatenate([x[j * CH:(j + 1) * CH] for j in np.nonzero(sp)[0]]) if sp.any() else x[:1]
        grp = "ami" if ss["set"] == "ami" else "calls"
        lv[grp].append(dbfs(speech))
        rows.append({"key": ss["key"], "sr": info.samplerate, "channels": info.channels, "subtype": info.subtype,
                     "n_conf": len(conf), "n_chunks_audio": len(x) // CH, "peak": round(float(np.abs(x).max()), 3),
                     "speech_dbfs": round(dbfs(speech), 1)})
        if grp == "calls" and len(fdiff) < 40:  # feature / probability parity on the last 8 s before each ref end
            for s0, e1 in ss["user_turns"]:
                w = x[max(0, int((e1 + 0.2) * SR) - 8 * SR): int((e1 + 0.2) * SR)]
                w8 = np.pad(w, (8 * SR - len(w), 0))
                f_tf = st.features(w)[0]
                f_pc = compute_whisper_log_mel_features(w8, do_normalize=True)
                fdiff.append(float(np.abs(f_tf - f_pc).max()))
                p_tf = float(st.session.run(None, {"input_features": f_tf[None]})[0].reshape(-1)[0])
                p_pc = float(st.session.run(None, {"input_features": f_pc[None].astype(np.float32)})[0].reshape(-1)[0])
                pdiff.append(abs(p_tf - p_pc))
    # the silero chunks recomputed on the audio for 3 calls sessions vs the dump
    sil = B.SileroVAD(ROOT / "data/silero/silero_vad_v5.onnx") if (ROOT / "data/silero/silero_vad_v5.onnx").exists() else None
    vad_chk = []
    if sil is not None:
        for ss in [s for s in E.sessions() if s["set"] != "ami"][:3]:
            x = E.read_audio(ss)
            pr = sil.probs(x)
            c = np.asarray(dump[ss["key"]]["conf"])
            n = min(len(pr), len(c))
            vad_chk.append({"key": ss["key"], "max_abs_diff": round(float(np.abs(pr[:n] - c[:n]).max()), 4),
                            "len_audio_chunks": len(pr), "len_dump": len(c)})
    out = {"sessions": rows,
           "speech_dbfs": {g: {"p10": round(float(np.percentile(v, 10)), 1), "p50": round(float(np.median(v)), 1),
                               "p90": round(float(np.percentile(v, 90)), 1)} for g, v in lv.items()},
           "sr_set": sorted({r["sr"] for r in rows}), "channels_set": sorted({r["channels"] for r in rows}),
           "max_peak": max(r["peak"] for r in rows),
           "conf_len_mismatch": [r["key"] for r in rows if abs(r["n_conf"] - r["n_chunks_audio"]) > 1],
           "features_tf_vs_pipecat_max_abs": round(max(fdiff), 5), "p_tf_vs_pipecat_max_abs": round(max(pdiff), 6),
           "n_feature_windows": len(fdiff), "silero_recomputed_vs_dump": vad_chk}
    save("inputs", out)
    print(json.dumps({k: v for k, v in out.items() if k != "sessions"}, indent=1))


# --------------------------------------------------------------------------- Pipecat replay
def replay(an, x, conf):
    """audioforge.baselines.turn.pipecat_smartturn_replay -> (events [{t, kind 'call'|'fallback', ...}], turn_ends
    [{t, path, p, ms}])."""
    from audioforge.baselines import turn as B
    calls, ends = B.pipecat_smartturn_replay(an, x, conf)
    ends = [{**e, "ms": e["compute_ms"]} for e in ends]
    events = sorted([{**c, "kind": "call"} for c in calls] + [{"t": e["t"], "kind": "fallback"} for e in ends
                                                             if e["path"] == "fallback"], key=lambda v: v["t"])
    return events, ends


def per_end(ss, events, ends):
    """For each scored reference end e: the first smart-turn call in [e - 80 ms, next onset / e + 6 s), its p, and the
    turn_end that answers it (first in the same window) with its path."""
    out = []
    for i, (s0, e1) in enumerate(ss["user_turns"]):
        if not ss["scored"][i]:
            continue
        h = e1 + E.HORIZON if ss["next_onset"][i] is None else min(e1 + E.HORIZON, ss["next_onset"][i])
        calls = [ev for ev in events if ev["kind"] == "call" and e1 - E.TOL <= ev["t"] < h]
        ans = [en for en in ends if e1 - E.TOL <= en["t"] + en["ms"] / 1000 < h]
        intr = [en for en in ends if s0 <= en["t"] + en["ms"] / 1000 < e1 - E.TOL]
        out.append({"key": ss["key"], "end": e1, "gap_to_next_onset": None if ss["next_onset"][i] is None
                    else round(ss["next_onset"][i] - e1, 3),
                    "first_call_dt": round(calls[0]["t"] - e1, 3) if calls else None,
                    "first_call_p": calls[0]["p"] if calls else None,
                    "first_call_complete": calls[0]["complete"] if calls else None,
                    "n_calls": len(calls),
                    "turn_end_dt": round(ans[0]["t"] + ans[0]["ms"] / 1000 - e1, 3) if ans else None,
                    "turn_end_path": ans[0]["path"] if ans else None, "interrupted": bool(intr)})
    return out


def score(sess, res):
    """E.pool per scope + the robustness of its p50: a seeded session bootstrap 90 % interval and the share of answered
    ends within 0.5 s / 1 s (the Pipecat latencies are bimodal: ~0.2 s model path vs ~3.1 s fallback)."""
    tab = {}
    rng = np.random.default_rng(0)
    for sc, sets in E.SCOPES.items():
        per = [E.score_session([e["t"] + e["ms"] / 1000 for e in res[s["key"]]], s, [e["ms"] / 1000 for e in res[s["key"]]])
               for s in sess if s["set"] in sets and s["key"] in res]
        tab[sc] = E.pool(per)
        lat = np.concatenate([np.asarray(p["lat"]) for p in per])
        bs = np.asarray([np.nanmedian(np.concatenate([np.asarray(per[i]["lat"], float)
                                                      for i in rng.integers(0, len(per), len(per))] + [np.array([np.nan])]))
                         for _ in range(1000)])
        tab[sc]["p50_bootstrap90_ms"] = [int(round(np.nanpercentile(bs, 5) * 1000)), int(round(np.nanpercentile(bs, 95) * 1000))]
        tab[sc]["answered_within_500ms_pct"] = round(100 * float(np.mean(lat <= 0.5)), 1) if len(lat) else None
        tab[sc]["answered_within_1s_pct"] = round(100 * float(np.mean(lat <= 1.0)), 1) if len(lat) else None
    return tab


def label_diag(sess, evs, dump):
    """Per scored calls end: the Pipecat VAD's last speaking chunk end in (turn start, next onset / e + 6 s) minus the
    labelled end; the last smart-turn call in that span (= the call at the turn's audible end) and its decision; and
    whether the other party then holds the floor (> 1 s of agent speech before the user's next turn)."""
    from audioforge.baselines import turn as B
    clips = {c["name"] + ".user": c for c in json.loads((E.E2E / "clips.json").read_text())}
    off, last, kinds = [], [], {}
    for s in sess:
        sp = np.isin(B.pipecat_vad(np.asarray(dump[s["key"]]["conf"]))["state"], (2, 3))
        c = clips[s["key"]]
        for i, (s0, e1) in enumerate(s["user_turns"]):
            if not s["scored"][i]:
                continue
            h = e1 + E.HORIZON if s["next_onset"][i] is None else min(e1 + E.HORIZON, s["next_onset"][i])
            js = np.nonzero(sp[:int(h / CH_S)])[0]
            js = js[js * CH_S > s0]
            if len(js):
                off.append((js[-1] + 1) * CH_S - e1)
            calls = [v for v in evs[s["key"]] if v["kind"] == "call" and s0 < v["t"] < h]
            ag = sum(max(0.0, min(b, h) - max(a_, e1)) for a_, b in c["agent_intervals"])
            k = "other_takes_floor" if ag > 1.0 else ("no_next_turn" if s["next_onset"][i] is None else "user_continues")
            kd = kinds.setdefault(k, {"n": 0, "with_call": 0, "last_call_complete": 0})
            kd["n"] += 1
            if calls:
                last.append(calls[-1]["complete"])
                kd["with_call"] += 1
                kd["last_call_complete"] += int(calls[-1]["complete"])
    off = np.asarray(off)
    return {"vad_speech_end_minus_label_end_s": {q: round(float(np.percentile(off, int(q[1:]))), 3)
                                                 for q in ("p10", "p25", "p50", "p75", "p90")},
            "share_label_end_later_than_vad_end_by_120ms": round(float(np.mean(off < -0.12)), 3),
            "last_call_in_turn": {"n": len(last), "complete": int(sum(last))}, "by_what_follows": kinds}


def cmd_replay(a):
    from audioforge.baselines import turn as B
    dump = E.load_dump()
    old = E.WORK / "baselines3"  # the re-implemented Pipecat row published before this audit
    bl = {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(old.glob("*.json")))}
    sess = [s for s in E.sessions() if s["key"] in dump]
    audio = {s["key"]: E.read_audio(s) for s in sess}
    out = {"sweep": {}}
    for stop in a.stop_secs:
        an = B.pipecat_smartturn_analyzer(stop)
        res, evs = {}, {}
        for s in sess:
            evs[s["key"]], res[s["key"]] = replay(an, audio[s["key"]], np.asarray(dump[s["key"]]["conf"]))
        tab = score(sess, res)
        pe = [r for s in sess if s["set"] != "ami" for r in per_end(s, evs[s["key"]], res[s["key"]])]
        pa = [r for s in sess if s["set"] == "ami" for r in per_end(s, evs[s["key"]], res[s["key"]])]
        fc = lambda rows: {"n_ends": len(rows), "with_call": sum(r["first_call_p"] is not None for r in rows),  # noqa
                           "first_call_complete": sum(bool(r["first_call_complete"]) for r in rows),
                           "answered_by_model": sum(r["turn_end_path"] == "model" for r in rows),
                           "answered_by_fallback": sum(r["turn_end_path"] == "fallback" for r in rows),
                           "missed": sum(r["turn_end_path"] is None for r in rows)}
        out["sweep"][str(stop)] = {"table": tab, "calls_ends": fc(pe), "ami_ends": fc(pa),
                                   "n_calls": sum(ev["kind"] == "call" for v in evs.values() for ev in v),
                                   "call_ms_p50": round(float(np.median([ev["ms"] for v in evs.values() for ev in v
                                                                         if ev["kind"] == "call"])), 1)}
        if stop == 3.0:
            out["per_end_calls"] = pe
            # parity with eot_latency.py baselines3 (the published row)
            mism, n = [], 0
            for s in sess:
                if s["key"] not in bl:
                    continue
                a_ = [(round(e["t"], 3), e["path"]) for e in bl[s["key"]]["pipecat_smartturn"]]
                b_ = [(round(e["t"], 3), e["path"]) for e in res[s["key"]]]
                n += len(a_)
                if a_ != b_:
                    mism.append({"key": s["key"], "harness": a_[:8], "pipecat": b_[:8]})
            out["parity_vs_old_harness_baselines3"] = {"n_turn_ends": n, "sessions_differ": len(mism), "examples": mism[:5]}
            out["label_diag_calls"] = label_diag([s for s in sess if s["set"] != "ami"], evs, dump)
            ps = np.array([r["first_call_p"] for r in pe if r["first_call_p"] is not None])
            out["first_call_p_hist_calls"] = np.histogram(ps, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])[0].tolist()
        print(stop, json.dumps(out["sweep"][str(stop)]), flush=True)
    save("replay", out)
    print(json.dumps({k: v for k, v in out["parity_vs_old_harness_baselines3"].items() if k != "examples"}))


# --------------------------------------------------------------------------- LiveKit
def cmd_livekit(a):
    from audioforge.baselines import turn as B
    lk = B.LiveKitText("en", threads=2)
    dump = E.load_dump()
    sess = [s for s in E.sessions() if s["key"] in dump]
    out = {}
    first = []
    for mx in a.max_delay:
        res = {}
        for ss in sess:
            d = dump[ss["key"]]
            conf = np.asarray(d["conf"], float)
            lv = B.livekit_vad(conf)
            sos = sorted(int(j) for j in lv["sos"])
            tok_at, pieces, c = d["tok_at"], d["pieces"], d["asr_clock"]
            commit, outl = 0, []
            for j, t_se in zip(lv["eos"], lv["speech_end"]):
                t_eos = (int(j) + 1) * CH_S
                mels = max((int(round(t_eos * SR)) - c["half"]) // c["hop"] + 1, 0)
                lead = c.get("chunk_lead", c["chunk_mel"])
                nfr = min((0 if mels < lead else (mels - lead) // c["chunk_mel"] + 1) * c["cs"], len(tok_at))
                ntok = tok_at[nfr - 1] if nfr > 0 else 0
                text = "".join(pieces[commit:ntok]).replace("▁", " ").strip()
                if not text:
                    continue
                p = lk.predict(text)
                delay = a.min_delay if p >= lk.threshold else mx
                t_fire = max(t_eos, float(t_se) + delay)
                nxt = next((s for s in sos if s > j), None)
                if nxt is not None and (nxt + 1) * CH_S < t_fire:
                    continue
                outl.append({"t": round(t_fire, 4), "ms": 0.0, "path": "min" if delay == a.min_delay else "max", "p": p,
                             "t_se": float(t_se)})
                commit = ntok
            res[ss["key"]] = outl
            if mx == 3.0 and ss["set"] != "ami":
                for i, (s0, e1) in enumerate(ss["user_turns"]):
                    h = e1 + E.HORIZON if ss["next_onset"][i] is None else min(e1 + E.HORIZON, ss["next_onset"][i])
                    m = [e for e in outl if e1 - E.TOL <= e["t"] < h]
                    first.append(m[0]["path"] if m else None)
        out[str(mx)] = score(sess, res)
        print(mx, json.dumps(out[str(mx)]), flush=True)
    out["calls_ends_path_at_default"] = {"min_delay (P>=thr)": first.count("min"), "max_delay": first.count("max"),
                                         "missed": first.count(None), "n": len(first)}
    save("livekit", out)
    print(out["calls_ends_path_at_default"])


# --------------------------------------------------------------------------- smart-turn on its own test data
def cmd_eval(a):
    from audioforge.baselines import turn as B
    from audioforge.datasets import smartturn as D
    st = B.SmartTurn(ONNX)
    root = ROOT / "data" / "smartturn"
    meta = json.loads((root / "cache" / "human_5_all.json").read_text())
    wav = np.load(root / "cache" / "human_5_all.npy", mmap_mode="r")
    sp = D.split_indices(meta, root)
    off, lab = meta["offsets"], np.asarray(meta["complete"], bool)
    out = {"split": sp["how"]}
    for name in ("eval", "train"):
        idx = sp[name] if name == "eval" else sorted(__import__("random").Random(0).sample(sp[name], a.n_train))
        p, lvl, dur = [], [], []
        for i in idx:
            x = np.asarray(wav[off[i]:off[i + 1]], np.float32)
            nz = x[np.abs(x) > 1e-4]
            lvl.append(dbfs(nz) if len(nz) else -120.0)
            dur.append(len(x) / SR)
            p.append(st.predict(x))
        y, p = lab[idx], np.array(p)
        pred = p > 0.5
        out[name] = {"n": len(idx), "n_complete": int(y.sum()), "accuracy": round(float((pred == y).mean()) * 100, 2),
                     "recall_complete": round(float(pred[y].mean()) * 100, 2),
                     "recall_incomplete": round(float((~pred[~y]).mean()) * 100, 2),
                     "clip_active_dbfs_p10_p50_p90": [round(float(np.percentile(lvl, q)), 1) for q in (10, 50, 90)],
                     "dur_s_p50": round(float(np.median(dur)), 2)}
        print(name, json.dumps(out[name]), flush=True)
    save("eval_human_5_all", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inputs")
    r = sub.add_parser("replay")
    r.add_argument("--stop-secs", type=float, nargs="+", default=[3.0, 2.0, 1.5, 1.0, 0.8, 0.5])
    lk = sub.add_parser("livekit")
    lk.add_argument("--min-delay", type=float, default=0.5)
    lk.add_argument("--max-delay", type=float, nargs="+", default=[3.0, 6.0, 1.5, 1.0, 0.8])
    ev = sub.add_parser("eval")
    ev.add_argument("--n-train", type=int, default=600)
    a = ap.parse_args()
    {"inputs": cmd_inputs, "replay": cmd_replay, "livekit": cmd_livekit, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
