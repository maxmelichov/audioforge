"""research/EOT_LATENCY.md: single mode's end-of-turn latency, measured the standard way, and a faster turn rule.

Standard metrics (``score``): per reference user turn (start s, end e):
  * EOT latency   = first turn_end at or after e - 80 ms (and before the user's next turn / 6 s) minus e, in ms, p50 / p95
                    over the answered ends. ``total`` (the standard number) = decision + the measured compute of that
                    decision; ``decision`` = the audio time at which the deciding input was complete (ours: the
                    frame's decision-ready time on the served ASR clock, ``Session._asr_ready_t``, which already holds
                    the chunk + lookahead buffering; baselines: the end of the deciding Silero chunk) minus e.
  * false interruption = % of reference turns with a turn_end in [s, e - 80 ms) (the user was still in the turn and
                    speaks again after it).
  * missed        = % of reference turn ends with no turn_end in [e - 80 ms, min(e + 6 s, next user onset)).

Data: the served single model (``audioforge.serve.Session``, the --mode single engine, stage1_served_v2.afm, TS-VAD track,
stored 5 s print) on (1) the 32 two-party user-channel sessions of E2E_FINAL's clips (user audio alone, 6 s zero pad;
109 scored ends) and (2) 200 AMI dev eot-bench v2 turns (seeded sample of the 974; audio = the meeting from the window
start to the turn end + 6.5 s, the target's print from other single-speaker speech in the same meeting).

  prepare_ami  write the AMI clips + prints (torch; gate)
  dump         run the served session once per clip and store every per-frame signal a rule reads (torch; gate; resumable)
  check        the numpy simulator reproduces the server's own turn_end times (shipped rule)
  baselines    Pipecat smart-turn v3.2 (+ Pipecat Silero VAD) and LiveKit EnglishModel (+ LiveKit Silero VAD) offline on
               the same audio (onnx; gate; resumable)
  sweep        rule families in numpy + baselines -> runs/eot_latency.json

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py prepare_ami
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py dump --budget 540   # until done
    PYTHONPATH=. .venv/bin/python scripts/research/eot_latency.py check
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py baselines --budget 540
    PYTHONPATH=. .venv/bin/python scripts/research/eot_latency.py sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SR = 16000
FRAME = 0.08
CHUNK = 512
CHUNK_S = CHUNK / SR
E2E = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad")
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/eot_latency")
# re-dumped after the TS-VAD clean-print check + anchored print adaptation (fe28a9e, research/TSWER.md): the TS-VAD
# columns (P(user) / P(other)) and so the turn head's p changed; dump3 = the dump after the 70 ms chunk-trigger fix
# (d832fca), before fe28a9e; dump2 = the old clock
# EOT_DUMP / EOT_OUT / EOT_LABELS (research/TURN_V4.md): score another head's dump, write elsewhere, or score against
# another reference label set ({session key: [[start, end], ...]} replacing user_turns, e.g. the audible-end labels)
DUMP = Path(os.environ.get("EOT_DUMP", "/Volumes/ExternalSSD/nvidia-audio-models/scratch/tswer_fix/eot_dump"))
LABELS = os.environ.get("EOT_LABELS")
DUMP_PRE_PRINTFIX = WORK / "dump3"
DUMP_OLD = WORK / "dump2"
# baselines4: the Pipecat row from Pipecat's own LocalSmartTurnAnalyzerV3 (scripts/research/smartturn_audit.py);
# baselines3 = our re-implementation of it (on dump3: LiveKit text on the post-d832fca ASR clock); "baselines" = dump2
BASE = WORK / "baselines4"
OUT = Path(os.environ.get("EOT_OUT", ROOT / "runs" / "eot_latency.json"))
N_AMI = 200
TOL = 0.08  # one frame of label tolerance at the reference end
HORIZON = 6.0
SERVED_AFM = ROOT / "runs" / "stage1_served_v2.afm"  # what ships (block-4 VAD head); dump2 used stage1_served.afm (v1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- data
def sessions() -> list[dict]:
    """[{key, set, cond, wav, pad_s, user_turns, scored, next_onset(list per turn), print}] for both corpora."""
    out = []
    refs = json.loads((E2E / "clips.json").read_text())
    prints = json.loads((E2E / "prints.json").read_text())
    for r in refs:
        if r["set"] == "ami":
            continue
        turns = [tuple(x) for x in r["user_turns"]]
        nxt = [turns[i + 1][0] if i + 1 < len(turns) else None for i in range(len(turns))]
        emb = (prints.get(r["name"]) or {}).get("5.0")
        out.append({"key": f"{r['name']}.user", "set": r["set"], "cond": "user", "wav": str(E2E / "clips" / f"{r['name']}.user.wav"),
                    "pad_s": 6.0, "user_turns": turns, "scored": r["scored"], "next_onset": nxt,
                    "embedding": emb["embedding"] if emb else None})
    ami = WORK / "ami" / "clips.json"
    if ami.exists():
        for r in json.loads(ami.read_text()):
            out.append({"key": f"{r['name']}.mono", "set": "ami", "cond": "mono", "wav": str(WORK / "ami" / f"{r['name']}.wav"),
                        "pad_s": 0.0, "user_turns": [tuple(r["user_turn"])], "scored": [True],
                        "next_onset": [r["next_onset"]], "embedding": r["embedding"]})
    if LABELS:  # another reference label set: same turns, other ends (next onsets unchanged)
        lab = json.loads(Path(LABELS).read_text())
        for s in out:
            if s["key"] in lab:
                s["user_turns_orig"] = s["user_turns"]
                s["user_turns"] = [tuple(x) for x in lab[s["key"]]]
                assert len(s["user_turns"]) == len(s["user_turns_orig"]), s["key"]
    return out


def read_audio(s) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(s["wav"], dtype="float32", always_2d=True)
    assert sr == SR
    x = x.mean(1)
    return np.concatenate([x, np.zeros(int(s["pad_s"] * SR), np.float32)])


def cmd_prepare_ami(a):
    import eval_stage1 as ES
    import soundfile as sf
    import torch
    import tsvad as T

    from audioforge.train import load_model
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    out = WORK / "ami"
    out.mkdir(parents=True, exist_ok=True)
    base, ext, meta, ds, _ = ES.v2_data()
    idx = sorted(random.Random(0).sample(range(len(ext)), N_AMI))
    model = load_model(str(ROOT / "runs" / "stage1_served.afm"), "cpu").eval()
    refs, segcache = [], {}
    for i in idx:
        v, mt = ext[i], meta[i]
        m, a0 = v["meeting"], float(v["start"])
        on, en = int(v["onset_frame"]) * FRAME, int(v["turn_end_frame"]) * FRAME
        dur_m = len(ds._audio[m]) / SR
        b0 = min(a0 + en + 6.5, dur_m)
        name = f"ami_{m}_{i:04d}"
        x = np.asarray(ds._clip(m, a0, b0), np.float32)
        nxt = round(len(v["spk_act"]) * FRAME, 3) if mt.get("end_reason") == "resume" else None
        spk = ds.speaker_ids[int(v["speaker"]) - getattr(ds, "speaker_offset", 0)]
        if m not in segcache:
            segcache[m] = T.single_segments(ds, m)
        ivs = T.clip_from(segcache[m].get(spk, []), 5.0, random.Random(f"eot_{name}"), exclude=(a0 - 2.0, b0 + 2.0))
        if ivs is None:
            log(f"  {name}: no 5 s print; skipped")
            continue
        emb = voiceprint(model, T.clip_audio(ds, m, ivs))
        sf.write(str(out / f"{name}.wav"), np.clip(x, -1, 1), SR, subtype="PCM_16")
        refs.append({"name": name, "meeting": m, "start": a0, "dur": round(len(x) / SR, 3),
                     "user_turn": [round(on, 3), round(en, 3)], "next_onset": nxt, "end_reason": mt.get("end_reason"),
                     "speaker": int(v["speaker"]), "print_ivs": [[round(p, 3), round(q, 3)] for p, q in ivs],
                     "embedding": [round(float(z), 6) for z in emb]})
    (out / "clips.json").write_text(json.dumps(refs))
    log(f"{len(refs)} AMI clips -> {out}")


# --------------------------------------------------------------------------- dump
def cmd_dump(a):
    import torch
    from single_model import single_engine

    import audioforge.serve as S
    torch.set_num_threads(2)
    DUMP.mkdir(parents=True, exist_ok=True)
    eng = single_engine(afm=SERVED_AFM)
    eng.warmup()
    tok = eng.asr.tokenizer
    specials = {tok.token_id(x) for x in getattr(tok, "specials", [])}
    t_start, n = time.time(), 0
    for ss in sessions():
        f = DUMP / f"{ss['key']}.json"
        if f.exists():
            continue
        if time.time() - t_start > a.budget:
            log(f"budget: {n} sessions this call; rerun")
            return
        x = read_audio(ss)
        s = S.Session(eng, S.SessionConfig(turn_policy="hybrid_dyn", timeout_ms=1000))
        if ss["embedding"] is not None:
            s.arm_enrollment("enroll", 0, embedding=ss["embedding"])
        rec = {"head": [], "sil": [], "conf": [], "tsvad": {}}
        orig_rtod, orig_feed, orig_step = s.asr.run_turn_on_diar, s.sil.feed, s.sil.sm.step

        def rtod(avail, act_fn, flush=False, _o=orig_rtod, _s=s, _r=rec):
            def af(v, _f=act_fn):
                q = _f(v)
                _r["tsvad"][v] = (round(float(q[1][0]), 5), round(float(q[1][1]), 5))
                return q
            res = _o(avail, af, flush)
            for v, p, vad in res:
                _r["head"].append((int(v), round(_s._asr_ready_t(v), 4), float(p), round(float(vad), 5)))
            return res

        def feed(xx, _o=orig_feed, _r=rec):
            o = _o(xx)
            _r["sil"] += [(int(v), float(sil), round(float(t), 4), int(ls)) for v, sil, t, ls in o]
            return o

        def step(c, _o=orig_step, _r=rec):
            _r["conf"].append(float(c))
            return _o(c)
        s.asr.run_turn_on_diar, s.sil.feed, s.sil.sm.step = rtod, feed, step
        msgs = []
        blk = SR * 160 // 1000
        for i in range(0, len(x), blk):
            msgs += s.process(x[i:i + blk])
        msgs += s.finish()
        head = rec["head"]
        ts = [rec["tsvad"].get(v, (0.0, 0.0)) for v, *_ in head]
        cm = np.asarray(list(s.chunk_ms), float)
        d = {"key": ss["key"], "set": ss["set"], "cond": ss["cond"], "has_print": ss["embedding"] is not None,
             "audio_s": round(len(x) / SR, 3),
             "head": {"v": [h[0] for h in head], "t": [h[1] for h in head], "p": [h[2] for h in head],
                      "vad": [h[3] for h in head], "pu": [q[0] for q in ts], "po": [q[1] for q in ts]},
             "sil": {"v": [q[0] for q in rec["sil"]], "sil": [q[1] for q in rec["sil"]],
                     "t": [q[2] for q in rec["sil"]], "last_sp": [q[3] for q in rec["sil"]]},
             "conf": rec["conf"],
             "tok_at": [int(q) for q in s.asr.tok_at],
             "pieces": ["" if t in specials else tok.sp.id_to_piece(int(t)) for t in s.asr.tokens],
             "asr_clock": {"half": int(s.asr.half), "hop": int(s.asr.hop), "chunk_mel": int(s.asr.chunk_mel),
                           "chunk_lead": int(s.asr.chunk_lead), "cs": int(s.asr.cs)},
             "chunk_ms": {"p50": round(float(np.median(cm)), 2), "p95": round(float(np.percentile(cm, 95)), 2),
                          "mean": round(float(cm.mean()), 2), "n": int(len(cm))},
             "silero_ms_per_chunk": round(float(np.mean(s.sil.ms)), 3) if s.sil.ms else None,
             "turn_ends": [{k: m.get(k) for k in ("t", "policy", "p", "silence_ms")} for m in msgs if m["type"] == "turn_end"]}
        f.write_text(json.dumps(d))
        n += 1
        log(f"{ss['key']}: {len(d['turn_ends'])} turn ends, {len(head)} frames, chunk p50 {d['chunk_ms']['p50']} ms")
    log("dump: all sessions done")


def load_dump() -> dict:
    return {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(DUMP.glob("*.json")))}


# --------------------------------------------------------------------------- the rule simulator
SHIPPED = {"family": "shipped", "cap_f": 25.0, "floor_f": 12.0, "tmin_f": 2.0, "offset_f": 3.22, "theta": 0.99748}


def ourvad_frames(vad: np.ndarray, thr: float):
    """Silence on the served VAD head (block 4): frames since the last frame with VAD >= thr (0 on a speech frame)
    and the index of the last speech frame (-1 before any)."""
    sil = np.zeros(len(vad))
    lsp = np.full(len(vad), -1)
    last = -1
    for v, q in enumerate(vad):
        if q >= thr:
            last = v
        sil[v] = 0.0 if last in (-1, v) else float(v - last)
        lsp[v] = last
    return sil, lsp


def _hybrid_merge(events):
    """Session._hybrid: one event per turn (a candidate is kept only if its path heard speech after the previous kept
    event's decision time)."""
    events.sort(key=lambda e: e[0])
    out, hyb_t = [], -1.0
    for t, path, v, p, silf in events:
        t_speech = (v - int(round(silf * 80)) // 80 + 1) * FRAME
        if t_speech <= hyb_t:
            continue
        hyb_t = t
        out.append((round(float(t), 4), path))
    return out


def sim_shipped(d: dict, rule: dict) -> list:
    """hybrid_dyn on Silero silence (the served rule; cap / floor / offset overridable) OR the head crossing
    (HeadPolicy, re-armed by served VAD > 0.5)."""
    h = d["head"]
    ht, hp, hvad = (np.asarray(h[k], float) for k in ("t", "p", "vad"))
    s = d["sil"]
    sil, srdy, slsp = np.asarray(s["sil"]), np.asarray(s["t"]), np.asarray(s["last_sp"])
    ns = min(len(sil), len(hp))
    ev = []
    cap, floor, tmin, off = rule["cap_f"], rule["floor_f"], rule.get("tmin_f", 0.0), rule.get("offset_f", 0.0)
    A = cap - floor
    fired = -2
    for v in range(ns):
        req = min(max(cap - A * hp[v], tmin), cap)
        if slsp[v] >= 0 and slsp[v] != fired and sil[v] - req > off:
            fired = slsp[v]
            ev.append((max(srdy[v], ht[v]), "dyn", v, float(hp[v]), float(sil[v])))
    th = rule.get("theta")
    if th is not None:
        armed, prev, silent = False, 0.0, 0
        for v in range(len(hp)):
            if hvad[v] > 0.5:
                armed, silent = True, 0
            else:
                silent += 1
            if armed and hp[v] >= th > prev:
                armed = False
                ev.append((ht[v], "head", v, float(hp[v]), float(silent)))
            prev = hp[v]
    return _hybrid_merge(ev)


def sim_vadhead(d: dict, rule: dict) -> list:
    """Silero-free rules on our VAD head (block 4) silence and the turn head posterior p, per silence run:
      (a) mode "any":  fire at the first frame with silence >= k frames and p >= th; optional fallback: silence >= F.
      (d) mode "once": at the frame where silence first reaches k frames, fire if p >= th; else wait for silence >= F.
    One firing per silence run (re-armed by a speech frame)."""
    c = d.setdefault("_cache", {})
    if ("np",) not in c:
        c[("np",)] = tuple(np.asarray(d["head"][k], float) for k in ("t", "p", "vad"))
    ht, hp, hvad = c[("np",)]
    src = rule.get("vad_src", "vad")  # "vad": the served VAD head (any speaker); "tsvad": P(user) of the TS-VAD track
    key = ("ov", src, rule["vad_thr"])
    if key not in c:
        pu = np.asarray(d["head"]["pu"], float)
        if src == "vad":
            sig, thr = hvad, rule["vad_thr"]
        elif src == "tsvad":
            sig, thr = pu, rule["vad_thr"]
        else:  # "either": the user is silent when the VAD head is below vad_thr OR P(user) < 0.5 (someone else talks)
            sig, thr = np.where((hvad >= rule["vad_thr"]) & (pu >= 0.5), 1.0, 0.0), 0.5
        c[key] = ourvad_frames(sig, thr)
    sil, lsp = c[key]
    k, th, F, mode = rule["k"], rule["th"], rule.get("fallback_f"), rule["mode"]
    out, fired = [], -2
    for v in range(len(hp)):
        if lsp[v] < 0 or lsp[v] == fired or sil[v] <= 0:
            continue
        head_ok = (sil[v] >= k and hp[v] >= th) if mode == "any" else (sil[v] == k and hp[v] >= th)
        if head_ok or (F is not None and sil[v] >= F):
            fired = lsp[v]
            out.append((round(float(ht[v]), 4), "head" if head_ok else "fallback"))
    return out


def _room_sig(d: dict, src: str, vt: float, ut: float):
    """Per frame speech indicator of a silence source: "vad" = the served VAD head >= vt (any speaker); "tsvad" = the
    target's P(user) >= ut; "both" = either of the two (silence = the user AND the room quiet); "either" = both of them
    (silence = the user OR the room quiet)."""
    hvad, pu = np.asarray(d["head"]["vad"], float), np.asarray(d["head"]["pu"], float)
    if src == "vad":
        return hvad >= vt
    if src == "tsvad":
        return pu >= ut
    if src == "both":
        return (hvad >= vt) | (pu >= ut)
    if src == "either":
        return (hvad >= vt) & (pu >= ut)
    raise ValueError(src)


def sim_room(d: dict, rule: dict) -> list:
    """Room-aware turn rule (no Silero), per 80 ms frame v with the turn head's p, P(user) / P(other) of the TS-VAD
    track and the served VAD head:
      A (head path)     silence on ``src`` >= k frames AND p >= th
      B (fallback)      silence on ``src`` >= F frames (F None = off); B' (dyn-wait, rule["dyn"] = (cap, floor, a)):
                        silence >= clamp(cap - a * p, floor, cap) frames; E (early, rule["early"] = (vt_e, k_e, th_e)):
                        p >= th_e after k_e frames of VAD < vt_e, checked before A
      C (others path)   P(other) >= ot (someone else is talking, so the user's turn ended when the user went quiet, no
                        wait for room silence) AND the user's own silence (P(user) < ut) >= kU frames AND p >= thC
                        (kU None = off), or that silence >= FU frames (the others fallback; FU None = off); P(other) >= ot must have
                        held for the last ``om`` frames (another speaker has the floor, not a backchannel)
    One firing per user turn: A / B re-arm on a speech frame of ``src``, C on a frame with P(user) >= ut; a firing of
    either disarms both (``arm`` "user": A / B also need a P(user) >= ut frame since the last firing)."""
    c = d.setdefault("_cache", {})
    if ("np",) not in c:
        c[("np",)] = tuple(np.asarray(d["head"][k], float) for k in ("t", "p", "vad"))
    ht, hp, _ = c[("np",)]
    src, vt, ut = rule["src"], rule.get("vad_thr", 0.3), rule.get("ut", 0.5)
    vh = rule.get("vad_hi")  # hysteresis on the VAD (src "vad"): speech from VAD >= vad_hi until VAD < vad_thr
    key = ("room", src, vt, ut, vh)
    if key not in c:
        if vh is None:
            sig = _room_sig(d, src, vt, ut)
        else:
            assert src == "vad"
            sig = _schmitt(np.asarray(d["head"]["vad"], float), vt, vh)
        c[key] = (sig, _room_sig(d, "tsvad", vt, ut), np.asarray(d["head"]["po"], float))
    sp, usp, po = c[key]
    msp = rule.get("min_sp", 0)  # the head path / fallback need >= min_sp speech frames of src since the last firing
    nsp = 0
    k, th, F = rule["k"], rule["th"], rule.get("fallback_f")
    # dyn-wait (Silero-free hybrid_dyn, research/TURN_V4.md): the silence wait in frames is clamp(cap - a * p, floor,
    # cap), so a moderately sure head gets a shorter wait than an unsure one; (cap_f, floor_f, a_f) or None
    dyn = rule.get("dyn")
    # early head path (research/TURN_V4.md): p >= th_e once the VAD has been below vt_e for k_e frames, even while
    # src still counts speech (the served VAD's tail); (vt_e, k_e, th_e) or None. One firing per turn: after an early
    # firing during src speech, nothing fires again until src has gone quiet and heard new speech
    early = rule.get("early")
    evad = np.asarray(d["head"]["vad"], float)
    erun, blocked = 0, False
    kU, ot, thC, FU = rule.get("ku"), rule.get("ot", 0.5), rule.get("th_c", 0.0), rule.get("fu")
    om = max(1, rule.get("om", 1))  # frames in a row with P(other) >= ot
    FW = rule.get("fw", 10 ** 9)  # the others fallback fires only while the user's silence is in [FU, FU + FW]
    orun = 0
    arm_user = rule.get("arm") == "user"
    out = []
    last = lastu = -1
    fired = firedu = -2
    for v in range(len(hp)):
        if sp[v]:
            last = v
            nsp += 1
        if usp[v]:
            lastu = v
        silv = v - last if last >= 0 else 0
        silu = v - lastu if lastu >= 0 else 0
        if early is not None:
            erun = erun + 1 if evad[v] < early[0] else 0
            if blocked and not sp[v]:  # the early firing's turn has gone quiet on src: its speech is spent
                blocked, fired = False, last
        path = None
        if early is not None and not blocked and last >= 0 and last != fired and erun >= early[1] and hp[v] >= early[2]:
            path = "early"
        elif last >= 0 and last != fired and silv > 0 and (not arm_user or lastu != firedu) and nsp >= msp:
            if silv >= k and hp[v] >= th:
                path = "head"
            elif dyn is not None and silv >= min(max(dyn[0] - dyn[2] * hp[v], dyn[1]), dyn[0]):
                path = "dyn"
            elif F is not None and silv >= F:
                path = "fallback"
        orun = orun + 1 if po[v] >= ot else 0
        if path is None and lastu >= 0 and lastu != firedu and orun >= om:
            if kU is not None and silu >= kU and hp[v] >= thC:
                path = "others"
            elif FU is not None and FU <= silu <= FU + FW:
                path = "others_fallback"
        if path is not None:
            fired, firedu, nsp = last, lastu, 0
            blocked = path == "early" and bool(sp[v])  # fired while src still hears speech: wait for its silence
            out.append((round(float(ht[v]), 4), path))
    return out


def _schmitt(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Hysteresis speech indicator: speech from a frame with x >= hi until a frame with x < lo (quiet before any)."""
    out = np.zeros(len(x), bool)
    on = False
    for v, q in enumerate(x):
        on = q >= hi or (on and q >= lo)
        out[v] = on
    return out


def sim_hyst(d: dict, rule: dict) -> list:
    """(b) head-only with hysteresis: fire when p >= hi, re-armed once p < lo after a served-VAD speech frame."""
    h = d["head"]
    ht, hp, hvad = (np.asarray(h[k], float) for k in ("t", "p", "vad"))
    hi, lo = rule["hi"], rule["lo"]
    out, armed, speech = [], True, False
    for v in range(len(hp)):
        if hvad[v] >= 0.5:
            speech = True
        if hp[v] < lo:
            armed = True
        if armed and speech and hp[v] >= hi:
            out.append((round(float(ht[v]), 4), "head"))
            armed, speech = False, False
    return out


def simulate(d: dict, rule: dict) -> list:
    fam = rule["family"]
    if fam in ("shipped", "silero_floor"):
        return sim_shipped(d, rule)
    if fam in ("vad_head", "vad_head_once"):
        return sim_vadhead(d, rule)
    if fam == "hyst":
        return sim_hyst(d, rule)
    if fam == "room":
        return sim_room(d, rule)
    raise ValueError(fam)


def cmd_check(a):
    dump = load_dump()
    bad = n = 0
    for k, d in dump.items():
        sim = [t for t, _ in simulate(d, SHIPPED)]
        srv = [e["t"] for e in d["turn_ends"]]
        n += len(srv)
        if len(sim) != len(srv) or any(abs(x - y) > 1e-3 for x, y in zip(sim, srv)):
            bad += 1
            log("MISMATCH", k, sim[:6], srv[:6])
    log(f"check: {len(dump)} sessions, {n} server turn_ends, {bad} sessions differ")
    # the served VadHeadPolicy with its defaults (the shipped rule) on the dumped frames == sim_room(CHOSEN)
    from audioforge.server.policies import VadHeadPolicy
    bad2 = n2 = 0
    for k, d in dump.items():
        h = d["head"]
        pol = VadHeadPolicy()
        got = []
        for v in range(len(h["p"])):
            ev = pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v])
            if ev is not None:
                got.append((round(float(h["t"][v]), 4), ev["path"]))
        want = [(t, "others" if path == "others_fallback" else path) for t, path in sim_room(d, CHOSEN)]
        n2 += len(want)
        bad2 += got != want
    log(f"check: served VadHeadPolicy defaults vs sim_room(CHOSEN): {n2} turn_ends, {bad2} sessions differ")
    # --turn-preset fast: the policy built from TURN_PRESETS["fast"] (audioforge.serve.vad_head_params) == sim_room(FAST)
    from audioforge.serve import vad_head_params
    k, fb, thr, others = vad_head_params("fast")
    bad3 = n3 = 0
    for k_, d in dump.items():
        h = d["head"]
        pol = VadHeadPolicy(0.99, k, fb, thr, others=others)
        got = []
        for v in range(len(h["p"])):
            ev = pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v])
            if ev is not None:
                got.append((round(float(h["t"][v]), 4), ev["path"]))
        want = [(t, "others" if path == "others_fallback" else path) for t, path in sim_room(d, FAST)]
        n3 += len(want)
        bad3 += got != want
    log(f"check: served VadHeadPolicy --turn-preset fast vs sim_room(FAST): {n3} turn_ends, {bad3} sessions differ")
    return bad + bad2 + bad3


# --------------------------------------------------------------------------- baselines
def cmd_baselines(a):
    """Pipecat 1.12 default (Silero VAD confidence 0.7, start / stop 0.2 s; smart-turn v3.2 at each VAD stop through
    Pipecat's own LocalSmartTurnAnalyzerV3 = audioforge.baselines.turn.pipecat_smartturn_replay: turn audio from
    speech start - 0.7 s (no earlier than the previous turn end, where Pipecat clears its buffer), <= 8 s;
    incomplete -> the stop_secs 3 s silence fallback, counted from the chunk after the VAD stop) and LiveKit agents 1.8 default (Silero plugin VAD activation 0.5, min silence 0.55 s; at each
    END_OF_SPEECH the EnglishModel EOU on the current turn's transcript = our served streaming ASR text ready by then;
    turn end = max(EOS + compute, speech end + (0.5 s if P >= threshold else 3.0 s)), cancelled by a new
    START_OF_SPEECH). Both read the Silero v5 confidences the served session computed on the same audio (the same
    ONNX model, 512-sample chunks)."""
    from audioforge.baselines import turn as B
    BASE.mkdir(parents=True, exist_ok=True)
    st = B.pipecat_smartturn_analyzer(stop_secs=3.0)
    lk = B.LiveKitText("en", threads=2)
    dump = load_dump()
    t0, n = time.time(), 0
    for ss in sessions():
        f = BASE / f"{ss['key']}.json"
        if f.exists() or ss["key"] not in dump:
            continue
        if time.time() - t0 > a.budget:
            log(f"budget: {n} sessions this call; rerun")
            return
        d = dump[ss["key"]]
        x = read_audio(ss)
        conf = np.asarray(d["conf"], float)
        # ---- Pipecat + smart-turn: Pipecat's own analyzer classes on a simulated clock
        calls, pc_out = B.pipecat_smartturn_replay(st, x, conf)
        st_ms = [c["ms"] for c in calls]
        # ---- LiveKit + EnglishModel
        lv = B.livekit_vad(conf)
        sos = set(int(j) for j in lv["sos"])
        tok_at, pieces = d["tok_at"], d["pieces"]
        c = d["asr_clock"]
        commit_tok, lk_out, lk_ms = 0, [], []
        eos = list(zip(lv["eos"], lv["speech_end"]))
        for i, (j, t_se) in enumerate(eos):
            t_eos = (int(j) + 1) * CHUNK_S
            # frames decoded by t_eos on the served ASR clock (StreamingSession.frames_ready: the first chunk after
            # chunk_lead mel frames, then every chunk_mel; dumps before the d832fca trigger fix: chunk_mel throughout)
            mels = max((int(round(t_eos * SR)) - c["half"]) // c["hop"] + 1, 0)
            lead = c.get("chunk_lead", c["chunk_mel"])
            nfr = min((0 if mels < lead else (mels - lead) // c["chunk_mel"] + 1) * c["cs"], len(tok_at))
            ntok = tok_at[nfr - 1] if nfr > 0 else 0
            text = "".join(pieces[commit_tok:ntok]).replace("▁", " ").strip()
            if not text:
                continue
            tc = time.perf_counter()
            p = lk.predict(text)
            ms = (time.perf_counter() - tc) * 1000
            lk_ms.append(ms)
            delay = 0.5 if p >= lk.threshold else 3.0
            t_fire = max(t_eos + ms / 1000, float(t_se) + delay)
            nxt_sos = min((s for s in sos if s > j), default=None)
            if nxt_sos is not None and (nxt_sos + 1) * CHUNK_S < t_fire:
                continue  # the user resumed before the endpointing delay ran out
            lk_out.append({"t": round(t_fire, 4), "path": "min_delay" if delay == 0.5 else "max_delay", "p": round(p, 5),
                           "compute_ms": round(ms, 2), "text": text[-80:]})
            commit_tok = ntok
        f.write_text(json.dumps({"key": ss["key"], "pipecat_smartturn": pc_out, "livekit_eou": lk_out,
                                 "smartturn_ms": st_ms, "livekit_ms": lk_ms, "lk_threshold": lk.threshold}))
        n += 1
        log(f"{ss['key']}: smart-turn {len(pc_out)} ends ({len(st_ms)} calls), LiveKit {len(lk_out)} ends")
    log("baselines: all sessions done")


def load_baselines() -> dict:
    return {d["key"]: d for d in (json.loads(p.read_text()) for p in sorted(BASE.glob("*.json")))}


# --------------------------------------------------------------------------- scoring
def score_session(times: list[float], ss: dict, compute: float | list = 0.0) -> dict:
    """Per session: latencies (s) of answered scored ends (``lat`` = total: the decision's input complete + its compute;
    ``lat_dec`` = decision: the audio time at which its input was complete, i.e. total - compute), # scored turns,
    # interrupted, # missed. ``times`` are totals; ``compute`` (s) per time or one value."""
    tt = np.asarray(times, float)
    cc = np.broadcast_to(np.asarray(compute, float), tt.shape)
    o = np.argsort(tt, kind="stable")
    T, C = tt[o], cc[o]
    lat, lat_dec, n, intr, miss = [], [], 0, 0, 0
    for i, (s0, e1) in enumerate(ss["user_turns"]):
        if not ss["scored"][i]:
            continue
        n += 1
        if np.any((T >= s0) & (T < e1 - TOL)):
            intr += 1
        h = e1 + HORIZON
        if ss["next_onset"][i] is not None:
            h = min(h, ss["next_onset"][i])
        m = (T >= e1 - TOL) & (T < h)
        if m.any():
            j = int(np.argmax(m))
            lat.append(float(T[j] - e1))
            lat_dec.append(float(T[j] - C[j] - e1))
        else:
            miss += 1
    return {"lat": lat, "lat_dec": lat_dec, "n": n, "intr": intr, "miss": miss}


def pool(scores: list[dict]) -> dict:
    lat = np.concatenate([np.asarray(s["lat"]) for s in scores]) if scores else np.zeros(0)
    dec = np.concatenate([np.asarray(s["lat_dec"]) for s in scores]) if scores else np.zeros(0)
    n = sum(s["n"] for s in scores)
    r = lambda x: None if x is None else int(round(x * 1000))  # noqa: E731
    p50 = float(np.median(lat)) if len(lat) else None
    p95 = float(np.percentile(lat, 95)) if len(lat) else None
    return {"n_turns": n, "n_answered": int(len(lat)),
            "eot_total_ms_p50": r(p50), "eot_total_ms_p95": r(p95),
            "eot_decision_ms_p50": r(float(np.median(dec)) if len(dec) else None),
            "eot_decision_ms_p95": r(float(np.percentile(dec, 95)) if len(dec) else None),
            "false_interruption_pct": round(100 * sum(s["intr"] for s in scores) / n, 1) if n else None,
            "missed_pct": round(100 * sum(s["miss"] for s in scores) / n, 1) if n else None}


SCOPES = {"two_party_user": ("turnbench", "oto"), "ami": ("ami",)}


def rule_grid() -> dict:
    R = {"today: hybrid_dyn 2000,960 (Silero)": dict(SHIPPED)}
    for floor in (0, 100, 200, 300, 400):  # (c) the shipped rule with its p = 1 wait (FLOOR) lowered
        for off in (3.22, 0.0):
            R[f"(c) Silero hybrid_dyn cap 2000 floor {floor}{' +offset' if off else ''}"] = {
                **SHIPPED, "family": "silero_floor", "floor_f": floor / 80, "tmin_f": 0.0, "offset_f": off}
    ths = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.99748, 0.999)
    for vt in (0.3, 0.5):
        for th in ths:
            for k in range(1, 9):  # (a) our VAD silence >= k frames AND p >= th (no fallback / with fallback)
                for F in (None, 10, 12, 15, 19, 25):
                    R[f"(a) ourVAD<{vt} sil>={k * 80}ms & p>={th}" + (f" | fallback {F * 80}ms" if F else "")] = {
                        "family": "vad_head", "mode": "any", "vad_thr": vt, "k": k, "th": th, "fallback_f": F}
            for k in (2, 3):  # (d) smart-turn style: decide once at silence ~200 ms, else fall back
                for F in (8, 10, 12, 15, 19):
                    R[f"(d) ourVAD<{vt} at sil={k * 80}ms p>={th} else fallback {F * 80}ms"] = {
                        "family": "vad_head_once", "mode": "once", "vad_thr": vt, "k": k, "th": th, "fallback_f": F}
    for th in ths:  # the same two families on the target's own silence (TS-VAD P(user) < 0.5), no Silero
        for k in range(1, 9):
            for F in (None, 10, 12, 15, 19, 25):
                R[f"(a') TS-VAD P(user)<0.5 sil>={k * 80}ms & p>={th}" + (f" | fallback {F * 80}ms" if F else "")] = {
                    "family": "vad_head", "mode": "any", "vad_src": "tsvad", "vad_thr": 0.5, "k": k, "th": th,
                    "fallback_f": F}
        for k in (2, 3):
            for F in (8, 10, 12, 15, 19):
                R[f"(d') TS-VAD P(user)<0.5 at sil={k * 80}ms p>={th} else fallback {F * 80}ms"] = {
                    "family": "vad_head_once", "mode": "once", "vad_src": "tsvad", "vad_thr": 0.5, "k": k, "th": th,
                    "fallback_f": F}
    for vt in (0.3, 0.5):  # silence = VAD head < vt OR TS-VAD P(user) < 0.5
        for th in ths:
            for k in range(1, 9):
                for F in (None, 10, 12, 15, 19, 25):
                    R[f"(a'') ourVAD<{vt} or P(user)<0.5 sil>={k * 80}ms & p>={th}" + (f" | fallback {F * 80}ms" if F else "")] = {
                        "family": "vad_head", "mode": "any", "vad_src": "either", "vad_thr": vt, "k": k, "th": th,
                        "fallback_f": F}
    for hi in (0.9, 0.95, 0.98, 0.99, 0.995, 0.99748, 0.999):  # (b) head only, hysteresis
        for lo in (0.3, 0.5, 0.7):
            R[f"(b) head-only p>={hi} rearm p<{lo}"] = {"family": "hyst", "hi": hi, "lo": lo}
    # (e) the room-aware simulator's head path + fallback on each silence source (sim_room without the others path)
    for src in ROOM_SRC:
        for vt in (0.3, 0.4, 0.5):
            if src == "tsvad" and vt != 0.3:
                continue  # P(user) < 0.5 only; the VAD threshold does not enter
            for k in range(1, 6):
                for th in (0.8, 0.9, 0.93, 0.95, 0.97, 0.98):
                    for F in (10, 12, 15, 19, None):
                        r = {"family": "room", "src": src, "vad_thr": vt, "k": k, "th": th, "fallback_f": F}
                        R[room_name(r)] = r
    R.update(demo_guard_grid())
    return R


def demo_guard_grid() -> dict:
    """(g) guards against the bundled clip's mid-question cut: in the user's 8.6-9.1 s pause the served VAD reads 0.32,
    0.40, 0.21, 0.37, 0.20, 0.12 (as quickstart_client delivers it, int16 by truncation) while the turn head's p is
    0.94-0.98, so the fe28a9e rule's head path fired at 9.06 s. Each guard is a change of that rule (SHIPPED_FE28,
    with its others path): VAD hysteresis (silence starts when VAD < lo, speech resumes at VAD >= hi), the head path
    at 400 ms, a minimum of VAD speech frames since the last firing before the head path / fallback may fire, and a
    surer head (theta 0.97-0.995) with a shorter wait and fallback."""
    R = {CHOSEN_NAME: dict(SHIPPED_FE28)}
    for lo in (0.25, 0.3, 0.35):
        for hi in (0.45, 0.5, 0.6):
            for k in (3, 4, 5):
                R[f"(g) hysteresis VAD<{lo} starts / >={hi} resets, sil>={k * 80}ms & p>=0.95 | fallback 800ms | others"] = {
                    **SHIPPED_FE28, "vad_thr": lo, "vad_hi": hi, "k": k}
    R["(g) head path at 400 ms (ourVAD<0.4 sil>=400ms & p>=0.95 | fallback 800ms | others)"] = {**SHIPPED_FE28, "k": 5}
    R["(g) head path at 400 ms, others path at 800 ms user silence"] = {**SHIPPED_FE28, "k": 5, "fu": 10}
    for th in (0.97, 0.98, 0.99, 0.995):  # a surer head (theta), with a shorter wait and fallback
        for k in (2, 3, 4, 5):
            for fb in (8, 10):
                R[f"(g) head theta {th}: ourVAD<0.4 sil>={k * 80}ms & p>={th} | fallback {fb * 80}ms | others"] = {
                    **SHIPPED_FE28, "th": th, "k": k, "fallback_f": fb}
    for m in (2, 4, 6, 8, 12):
        R[f"(g) min {m * 80} ms of VAD speech before an end | shipped"] = {**SHIPPED_FE28, "min_sp": m}
    # the fastest no-cut rule of a wider scratch grid (VAD 0.35-0.5, 160-640 ms, theta 0.9-0.995, fallback, others
    # path) that meets the calls / AMI-miss goal; it costs 3.5 points of AMI false interruptions
    R["(g) ourVAD<0.5 sil>=480ms & p>=0.99 | fallback 640ms | others"] = {
        **SHIPPED_FE28, "vad_thr": 0.5, "k": 6, "th": 0.99, "fallback_f": 8}
    return R


def clip_cuts(d: dict, rule: dict, meta: dict) -> dict:
    """The rule on the bundled clip's recorded frames: its turn_ends, and whether one falls inside the user's turn
    (a cut: [first user onset, reference end - 80 ms))."""
    te = [t for t, _ in simulate(d, rule)]
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    return {"turn_ends": te, "cuts_user_turn": any(s0 <= t < e1 - TOL for t in te)}


ROOM_SRC = {"vad": "ourVAD<{vt}", "tsvad": "P(user)<0.5", "both": "ourVAD<{vt} & P(user)<0.5",
            "either": "ourVAD<{vt} or P(user)<0.5"}


def room_name(r: dict) -> str:
    n = (f"({'f' if r.get('fu') or r.get('ku') else 'e'}) {ROOM_SRC[r['src']].format(vt=r.get('vad_thr', 0.3))} "
         f"sil>={r['k'] * 80}ms & p>={r['th']}" + (f" | fallback {r['fallback_f'] * 80}ms" if r.get("fallback_f") else ""))
    if r.get("fu") or r.get("ku"):
        n += f" | others P(other)>={r['ot']} for {r.get('om', 1) * 80}ms:"
        if r.get("ku"):
            n += f" user sil>={r['ku'] * 80}ms & p>={r['th_c']}"
        if r.get("fu"):
            fw = r.get("fw", 10 ** 9)
            n += (f" user sil={r['fu'] * 80}ms" if fw == 0 else f" user sil>={r['fu'] * 80}ms")
    return n


def others_grid(bases: list[dict]) -> dict:
    """(f) the others path (sim_room C) on top of each base rule: P(other) >= ot held for om frames, and the user's own
    silence reaching FU frames (fw 0: checked once, at that frame; else any frame after) and / or kU frames with the head
    p >= thC."""
    R = {}
    for b in bases:
        for ot in (0.5, 0.7, 0.9):
            for om in (1, 4, 8):
                for fu in (4, 6, 8, 10, 12, 15, 19, 25):
                    for fw in (0, 10 ** 9):
                        for ku, thc in ((None, 0.0), (4, 0.9), (2, 0.95)):
                            r = {**b, "ot": ot, "om": om, "fu": fu, "fw": fw, "ku": ku, "th_c": thc}
                            R[room_name(r)] = r
    return R


# the goal of the shipped rule (the brief, 2026-09-30): faster than today on two-party calls with false interruptions
# and misses no worse than today's there (the pre-d832fca dump's 23.9 / 8.3 %), and on AMI no more misses than today's
# 38.5 % at false interruptions <= ~10 %
GOAL = {"calls_fi": 23.9, "calls_miss": 8.3, "ami_miss": 38.5, "ami_fi": 10.0}
# the fe28a9e shipped rule's own numbers on the post-print-fix dump (research/TSWER.md); a guard must not lose them
GOAL_PRINTFIX = {"calls_fi": 22.9, "calls_miss": 7.3, "ami_miss": 34.0, "ami_fi": 10.5}  # AMI fi: 10.0 + one turn
N_BASES = 20


def head_reach(dump, sess, ths=(0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.99748)) -> dict:
    """% of reference ends where the head posterior reaches >= th (on a frame ready) within 200 / 300 / 500 ms of the
    end, and the same for mid-turn pauses (user silence >= 200 ms inside a turn), by decision time."""
    out = {}
    for scope, sets in SCOPES.items():
        ends, pauses = [], []
        for ss in sess:
            if ss["set"] not in sets or ss["key"] not in dump:
                continue
            h = dump[ss["key"]]["head"]
            ht, hp = np.asarray(h["t"]), np.asarray(h["p"])
            for i, (s0, e1) in enumerate(ss["user_turns"]):
                if ss["scored"][i]:
                    ends.append([hp[(ht >= e1) & (ht <= e1 + w)].max(initial=0.0) for w in (0.2, 0.3, 0.5)])
        E = np.asarray(ends)
        out[scope] = {"n_ends": len(E), **{f"p>={th}": {f"within_{w}ms": round(float(100 * np.mean(E[:, j] >= th)), 1)
                                                         for j, w in enumerate((200, 300, 500))} for th in ths}}
    return out


def cmd_sweep(a):
    dump, bl = load_dump(), load_baselines()
    sess = [s for s in sessions() if s["key"] in dump]
    compute = {k: d["chunk_ms"]["p50"] / 1000 for k, d in dump.items()}
    rules = rule_grid()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"),
           "definitions": {"eot_total_ms": "first turn_end at/after the reference end - 80 ms (before min(end + 6 s, next "
                                           "user onset)) minus the reference end; decision audio time + measured compute",
                           "eot_decision_ms": "the audio time at which the deciding input was complete (ours: the frame's "
                                              "decision-ready time on the served ASR clock, which already holds the "
                                              "chunk + lookahead buffering; baselines: the end of the deciding Silero "
                                              "chunk) minus the reference end = eot_total - compute",
                           "false_interruption_pct": "% of reference user turns with a turn_end in [start, end - 80 ms)",
                           "missed_pct": "% of reference turn ends with no turn_end in [end - 80 ms, min(end + 6 s, next "
                                         "user onset))"},
           "n_sessions": {sc: sum(s["set"] in sets for s in sess) for sc, sets in SCOPES.items()}}
    comp = {sc: [d["chunk_ms"] for k, d in dump.items() if d["set"] in sets] for sc, sets in SCOPES.items()}
    res["compute_ms"] = {
        "ours_chunk_p50_median_over_sessions": {sc: round(float(np.median([c["p50"] for c in v])), 1) for sc, v in comp.items()},
        "ours_chunk_p95_median_over_sessions": {sc: round(float(np.median([c["p95"] for c in v])), 1) for sc, v in comp.items()},
        "silero_per_32ms_chunk": round(float(np.median([d["silero_ms_per_chunk"] for d in dump.values()])), 3)}
    table = {sc: {} for sc in SCOPES}

    def score_rules(rr: dict):
        for name, rule in rr.items():
            per = {sc: [] for sc in SCOPES}
            for ss in sess:
                times = [t + compute[ss["key"]] for t, _ in simulate(dump[ss["key"]], rule)]
                for sc, sets in SCOPES.items():
                    if ss["set"] in sets:
                        per[sc].append(score_session(times, ss, compute[ss["key"]]))
            for sc in SCOPES:
                table[sc][name] = pool(per[sc])
    t0 = time.time()
    score_rules(rules)
    # (f): the others path on the N_BASES fastest (e) rules that meet the calls goal
    tpc = table["two_party_user"]
    bases = sorted((n for n, r in rules.items() if r["family"] == "room" and tpc[n]["eot_total_ms_p50"] is not None
                    and tpc[n]["false_interruption_pct"] <= GOAL["calls_fi"] and tpc[n]["missed_pct"] <= GOAL["calls_miss"]),
                   key=lambda n: (tpc[n]["eot_total_ms_p50"], tpc[n]["false_interruption_pct"]))[:N_BASES]
    extra = others_grid([rules[n] for n in bases])
    score_rules(extra)
    rules = {**rules, **extra}
    log(f"{len(rules)} rules in {time.time() - t0:.0f} s")
    # baselines
    bnames = {"pipecat_smartturn": "Pipecat smart-turn v3.2 + Silero (defaults)",
              "livekit_eou": "LiveKit EnglishModel + Silero (defaults)"}
    bcomp = {}
    for bk, label in bnames.items():
        per = {sc: [] for sc in SCOPES}
        for ss in sess:
            if ss["key"] not in bl:
                continue
            # Pipecat: the smart-turn call runs after the VAD stop (+ its compute); LiveKit's t already holds the
            # EnglishModel compute when EOS + compute binds (max(EOS + compute, speech end + delay))
            cmp = [e["compute_ms"] / 1000 if bk == "pipecat_smartturn" else 0.0 for e in bl[ss["key"]][bk]]
            times = [e["t"] + c for e, c in zip(bl[ss["key"]][bk], cmp)]
            for sc, sets in SCOPES.items():
                if ss["set"] in sets:
                    per[sc].append(score_session(times, ss, cmp))
        for sc in SCOPES:
            table[sc][label] = pool(per[sc])
        ms = np.asarray([x for k in bl for x in bl[k]["smartturn_ms" if bk == "pipecat_smartturn" else "livekit_ms"]])
        bcomp[label] = {"calls": int(len(ms)), "p50": round(float(np.median(ms)), 1) if len(ms) else None,
                        "p95": round(float(np.percentile(ms, 95)), 1) if len(ms) else None}
    res["compute_ms"]["baseline_model_per_decision"] = bcomp
    # the JSON keeps every rule except the (f) ones that miss the goal by more than 2 AMI false-interruption points
    # (8640 others-path variants per run)
    keep = [n for n, r in rules.items() if n not in extra or (
        table["two_party_user"][n]["false_interruption_pct"] <= GOAL["calls_fi"]
        and table["two_party_user"][n]["missed_pct"] <= GOAL["calls_miss"]
        and table["ami"][n]["missed_pct"] <= GOAL["ami_miss"] and table["ami"][n]["false_interruption_pct"] <= GOAL["ami_fi"] + 2)]
    res["n_rules_scored"] = len(rules)
    rules = {n: rules[n] for n in keep}
    table = {sc: {n: v for n, v in t.items() if n in rules or n in bnames.values()} for sc, t in table.items()}
    res["rules"] = rules
    res["table"] = table
    # selection on two_party_user (AMI = held-out corpus)
    tp = table["two_party_user"]
    today = tp["today: hybrid_dyn 2000,960 (Silero)"]
    lkr = tp[bnames["livekit_eou"]]
    ours = [n for n in rules]

    def fastest(fi_max, miss_max):
        ok = [n for n in ours if tp[n]["eot_total_ms_p50"] is not None and tp[n]["false_interruption_pct"] <= fi_max
              and tp[n]["missed_pct"] <= miss_max]
        return min(ok, key=lambda n: (tp[n]["eot_total_ms_p50"], tp[n]["false_interruption_pct"])) if ok else None
    have_lk = lkr["missed_pct"] is not None
    miss_cap = today["missed_pct"]  # a faster rule must not miss more turn ends than today's
    sel = {"miss_cap_pct": miss_cap,
           "fastest_fi<=livekit": fastest(lkr["false_interruption_pct"], miss_cap) if have_lk else None,
           "fastest_fi<=today": fastest(today["false_interruption_pct"], miss_cap)}
    # at the latency budget: decision p50 <= 250 ms -> the fewest false interruptions (then misses)
    bud = [n for n in ours if tp[n]["eot_decision_ms_p50"] is not None and tp[n]["eot_decision_ms_p50"] <= 250
           and tp[n]["missed_pct"] <= miss_cap]
    sel["best_at_decision_p50<=250ms"] = min(bud, key=lambda n: (tp[n]["false_interruption_pct"], tp[n]["missed_pct"])) if bud else None
    am = table["ami"]
    good = [n for n in ours if tp[n]["eot_total_ms_p50"] is not None and am[n]["missed_pct"] is not None
            and tp[n]["false_interruption_pct"] <= GOAL["calls_fi"] and tp[n]["missed_pct"] <= GOAL["calls_miss"]
            and am[n]["missed_pct"] <= GOAL["ami_miss"] and am[n]["false_interruption_pct"] <= GOAL["ami_fi"]]
    sel["goal"] = GOAL
    sel["n_meeting_goal"] = len(good)
    sel["fastest_meeting_goal_both_corpora"] = min(
        good, key=lambda n: (tp[n]["eot_total_ms_p50"], am[n]["eot_total_ms_p50"], tp[n]["false_interruption_pct"])) if good else None
    # after the TS-VAD print fix (fe28a9e) the shipped rule cut the bundled clip mid-question: the fastest (g) guard (or
    # the shipped rule) with calls false interruptions / misses <= the fe28a9e shipped rule's (22.9 / 7.3 %), AMI misses
    # <= its 34.0 % and AMI false interruptions <= its 10.0 % + one turn, and no turn_end inside the clip's user turn
    # under any of the ways a client may deliver the clip (the frames served_check recorded, CLIP_VARIANTS)
    clipf = Path(os.environ.get("EOT_CLIP_FRAMES", WORK / "clip_frames.json"))
    demo = {}
    if clipf.exists():
        cv = json.loads(clipf.read_text())
        meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
        for n in [n for n in rules if n.startswith("(g)")] + [CHOSEN_NAME]:
            demo[n] = {v: clip_cuts(d, rules[n], meta) for v, d in cv.items()}
            demo[n]["cut_in"] = [v for v in cv if demo[n][v]["cuts_user_turn"]]
    res["demo_clip"] = {"frames": str(clipf), "variants": list(CLIP_VARIANTS), "rules": demo}
    ok = [n for n in [*demo] if tp[n]["false_interruption_pct"] <= GOAL_PRINTFIX["calls_fi"]
          and tp[n]["missed_pct"] <= GOAL_PRINTFIX["calls_miss"] and am[n]["missed_pct"] <= GOAL_PRINTFIX["ami_miss"]
          and am[n]["false_interruption_pct"] <= GOAL_PRINTFIX["ami_fi"] and not demo[n]["cut_in"]]
    sel["goal_print_fix"] = GOAL_PRINTFIX
    sel["fastest_print_fix_goal_no_clip_cut"] = min(
        ok, key=lambda n: (tp[n]["eot_total_ms_p50"], tp[n]["false_interruption_pct"], am[n]["eot_total_ms_p50"])) if ok else None
    res["selection"] = sel
    # Pareto front (two_party_user): EOT total p50 vs false interruption, among rules with misses <= cap
    pts = sorted(((tp[n]["eot_total_ms_p50"], tp[n]["false_interruption_pct"], n) for n in ours
                  if tp[n]["eot_total_ms_p50"] is not None and tp[n]["missed_pct"] <= miss_cap))
    front, best = [], 1e9
    for lat, fi, n in pts:
        if fi < best:
            front.append(n)
            best = fi
    res["pareto_two_party_user"] = [{"rule": n, **tp[n], "ami": table["ami"][n]} for n in front]
    res["head_reach"] = head_reach(dump, sess)
    # the floor of any rule on our 80 ms frames: the first frame wholly after the reference end (k = 1), ready with its
    # 160 ms chunk (decision), + compute (total)
    fl = {}
    for sc, sets in SCOPES.items():
        x, xd = [], []
        for ss in sess:
            if ss["set"] not in sets:
                continue
            ht = np.asarray(dump[ss["key"]]["head"]["t"])
            for i, (s0, e1) in enumerate(ss["user_turns"]):
                v = int(e1 / FRAME) + 1
                if ss["scored"][i] and v < len(ht):
                    x.append(ht[v] + compute[ss["key"]] - e1)
                    xd.append(ht[v] - e1)
        x = np.asarray(x)
        fl[sc] = {"total_ms_p50": int(round(1000 * np.median(x))), "total_ms_p95": int(round(1000 * np.percentile(x, 95))),
                  "decision_ms_p50": int(round(1000 * np.median(xd)))}
    res["structural_floor_k1"] = fl
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "served_check" in prev:
        res["served_check"] = prev["served_check"]
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for k in ("today: hybrid_dyn 2000,960 (Silero)", CHOSEN_NAME, sel["fastest_print_fix_goal_no_clip_cut"],
              sel["fastest_meeting_goal_both_corpora"], sel["fastest_fi<=livekit"],
              sel["fastest_fi<=today"],
              sel["best_at_decision_p50<=250ms"], *bnames.values()):
        if k:
            log(k, "|", tp.get(k), "| AMI", table["ami"].get(k))
    log(f"-> {OUT}")


# the rule shipped as --mode single's default (turn_policy vad_head with its defaults: --vad-wait-ms 160,640,
# VAD < 0.4, theta 0.99, --others-wait-ms 960,640 at P(other) >= 0.9): selection["fastest_print_fix_goal_no_clip_cut"]
# (after fe28a9e; until then SHIPPED_FE28, selection["fastest_meeting_goal_both_corpora"] on dump3)
CHOSEN = {"family": "room", "src": "vad", "vad_thr": 0.4, "k": 2, "th": 0.99, "fallback_f": 8,
          "ot": 0.9, "om": 8, "fu": 12, "fw": 0, "ku": None, "th_c": 0.0}
# the rule shipped at a8b8c67-fe28a9e (--vad-wait-ms 320,800, VAD < 0.4, theta 0.95, --others-wait-ms 960,640)
SHIPPED_FE28 = {"family": "room", "src": "vad", "vad_thr": 0.4, "k": 4, "th": 0.95, "fallback_f": 10,
                "ot": 0.9, "om": 8, "fu": 12, "fw": 0, "ku": None, "th_c": 0.0}
# --turn-preset fast (research/VAD_TAIL.md, "Turn presets" below): VAD < 0.6 for >= 480 ms AND p >= 0.99, OR 720 ms,
# others path 640,640 (audioforge.server.constants.TURN_PRESETS["fast"])
FAST = {**CHOSEN, "vad_thr": 0.6, "k": 6, "fallback_f": 9, "fu": 8}
CHOSEN_NAME = "(g0) shipped at fe28a9e: ourVAD<0.4 sil>=320ms & p>=0.95 | fallback 800ms | others"
PREV_VAD_HEAD = {"family": "vad_head", "mode": "any", "vad_thr": 0.3, "k": 1, "th": 0.95, "fallback_f": 12}


def _i16_trunc(x):  # what examples/quickstart_client.py sends: float -> int16 by truncation
    return (np.clip(x, -1, 1) * 32767).astype("<i2").astype(np.float32) / 32768.0


def _i16_round(x):
    return np.round(np.clip(x, -1, 1) * 32767).astype("<i2").astype(np.float32) / 32768.0


# the ways a client may deliver the bundled clip (the server scales int16 by 1 / 32768): the fe28a9e rule's 9.06 s cut
# came from the quickstart client's truncation alone, so a rule is checked on all of them
CLIP_VARIANTS = {
    "float": lambda x: x,
    "int16_trunc (quickstart_client)": _i16_trunc,
    "int16_round": _i16_round,
    "gain_0.5_int16_trunc": lambda x: _i16_trunc(0.5 * x),
    "gain_2.0_int16_trunc": lambda x: _i16_trunc(2.0 * x),
    "dither_-70dBFS_int16_round": lambda x: _i16_round(
        x + np.random.default_rng(0).normal(0, 10 ** (-70 / 20), len(x)).astype(np.float32)),
}


def sim_served_gate(d: dict, x: np.ndarray, preset: str = "balanced", quiet_db=None) -> list:
    """The served vad_head WITH the energy gate (--energy-gate on, the default since research/EOT_ASSISTANT.md
    "Energy gate"): audioforge's VadHeadPolicy + EnergyGate built as Session._vad_head_policy builds them, over the
    recorded frames d["head"] and the frame energies of the delivered audio x -> [(t, path)]."""
    import math

    from audioforge.serve import vad_head_params
    from audioforge.server.constants import ENERGY_GATE as G
    from audioforge.server.constants import FRAME_MS
    from audioforge.server.policies import EnergyGate, VadHeadPolicy, frame_db
    k, fb, thr, others = vad_head_params(preset)
    pol = VadHeadPolicy(0.99, k, fb, thr, others=others, gate=EnergyGate(
        quiet_db=quiet_db, onset_db=G["onset_db"], window_s=G["window_s"], pct=G["pct"],
        warmup_frames=int(math.ceil(G["warmup_ms"] / FRAME_MS))))
    h, out = d["head"], []
    for v in range(len(h["p"])):
        b = min((v + 1) * 1280, int(round(float(h["t"][v]) * SR)))  # the samples known at the decision-ready time
        ev = pol.update(h["p"][v], h["vad"][v], h["pu"][v], h["po"][v], frame_db(x[v * 1280:b]) if b > v * 1280 else -100.0)
        if ev is not None:
            out.append((round(float(h["t"][v]), 4), ev["path"]))
    return out


def _clip_session(eng, S, x, emb, pol, preset=None):
    """The served session on the clip in 20 ms blocks -> (turn_end messages, per-frame signals, session)."""
    s = S.Session(eng, S.SessionConfig(turn_policy=pol, turn_preset=preset))
    s.arm_enrollment("enroll", 0, embedding=emb)
    rec = []
    orig = s.asr.run_turn_on_diar

    def rtod(avail, act_fn, flush=False):
        out = orig(avail, act_fn, flush)
        tp = s.asr.tsvad_p
        rec.extend((round(s._asr_ready_t(v), 4), float(p), float(vad), float(tp[v][0]), float(tp[v][1]))
                   for v, p, vad in out)
        return out
    s.asr.run_turn_on_diar = rtod
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    d = {"head": {k: [r[j] for r in rec] for j, k in enumerate(("t", "p", "vad", "pu", "po"))}}
    return [m for m in msgs if m["type"] == "turn_end"], d, s


def cmd_served_check(a):
    """The served turn_policy vad_head (defaults: --vad-wait-ms 160,640, VAD < 0.4, theta 0.99, --others-wait-ms
    960,640) on the bundled clip (examples/audio/two_party_call_16s.wav + its stored voice print) vs the offline rule
    (sim_room with CHOSEN) on the per-frame signals recorded from the same session, under every CLIP_VARIANTS delivery
    (the frames go to WORK / clip_frames.json for the sweep's demo-clip check); and the shipped hybrid_dyn session on
    the float clip for contrast."""
    import torch
    from single_model import single_engine

    import audioforge.serve as S
    from audioforge.data import load_wav
    torch.set_num_threads(2)
    preset = getattr(a, "preset", "balanced")
    tmod = getattr(a, "turn_model", "head")  # --turn-model smartturn (research/EOT_ASSISTANT.md): no sim_room twin
    rule = FAST if preset == "fast" else CHOSEN
    eng = single_engine(afm=SERVED_AFM, turn_model=tmod)
    eng.warmup()
    wav = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
    x0 = load_wav(str(wav), SR).astype(np.float32)
    emb = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json").read_text())
    meta = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
    s0, e1 = meta["user_intervals"][0][0], meta["user_turn_ends"][0]
    res = {"clip": str(wav.relative_to(ROOT)), "reference_user_turn_ends": meta["user_turn_ends"],
           "user_turn": [s0, e1]}
    frames = {}
    for var, fn in CLIP_VARIANTS.items():
        for pol in ("vad_head", "hybrid_dyn") if var == "float" and preset == "balanced" and tmod == "head" else ("vad_head",):
            te, d, s = _clip_session(eng, S, fn(x0), emb, pol, preset if pol == "vad_head" else None)
            r = {"served_turn_ends": [{k: m.get(k) for k in ("t", "p", "silence_ms", "path", "model_ms") if k in m}
                                      for m in te],
                 "cuts_user_turn": any(s0 <= m["t"] < e1 - TOL for m in te),
                 "silero_loaded_in_session": s.sil is not None,
                 "chunk_ms_p50": round(float(np.median(list(s.chunk_ms))), 1)}
            if pol == "vad_head" and tmod == "head":
                sim_v = [t for t, _ in sim_room(d, rule)]  # the VAD-only rule (--energy-gate off)
                sim = ([t for t, _ in sim_served_gate(d, fn(x0), preset, eng.energy_quiet_db)] if eng.energy_gate
                       else sim_v)
                r["offline_rule_turn_ends"] = sim
                r["offline_vad_only_rule_turn_ends"] = sim_v
                r["served_equals_offline"] = [m["t"] for m in te] == [round(t, 3) for t in sim]
                d.pop("_cache", None)
                frames[var] = {**d, "turn_ends": r["served_turn_ends"]}
            hit = [m["t"] for m in te if m["t"] >= e1 - TOL]
            r[f"eot_decision_audio_ms_after_{e1}"] = round((hit[0] - e1) * 1000) if hit else None
            res[pol if var == "float" else f"vad_head [{var}]"] = r
            log(var, pol, r["served_turn_ends"], "cut" if r["cuts_user_turn"] else "no cut",
                r.get("served_equals_offline"))
    if preset == "balanced" and tmod == "head":  # the frames are the same under either preset (the rule reads them)
        (WORK / "clip_frames.json").write_text(json.dumps(frames))
    res["engine"] = {"energy_gate": eng.energy_gate, "energy_quiet_db": eng.energy_quiet_db, "turn_model": tmod}
    log(json.dumps(res, indent=1))
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    key = "served_check" if preset == "balanced" else f"served_check_{preset}"
    prev[key if tmod == "head" else f"{key}_{tmod}"] = res
    OUT.write_text(json.dumps(prev, indent=1, default=float))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare_ami")
    d = sub.add_parser("dump")
    d.add_argument("--budget", type=float, default=540)
    sub.add_parser("check")
    b = sub.add_parser("baselines")
    b.add_argument("--budget", type=float, default=540)
    sub.add_parser("sweep")
    sc = sub.add_parser("served_check")
    sc.add_argument("--preset", choices=("balanced", "fast"), default="balanced")
    sc.add_argument("--turn-model", dest="turn_model", choices=("head", "smartturn"), default="head")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
