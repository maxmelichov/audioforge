"""plans/fixwave/turn_clean.md: held-out-only re-selection of the turn presets (115M: classifier + constants; 0.6B:
constants on the v0.4 heads), then one scoring pass on the public test rows (smart-turn v3.2 test 399 at false-fire
windows 2.0 / 2.5 / 3.0 / 3.5 s; AMI test turns). -> runs/turn_clean.json.

Selection audio (never the 399, AMI test, TurnBench or the 16 oto evaluation conversations): core_0p6b_turn's held-out
set (ho: oto calls, quiet oto, AMI ES2015c, smart-turn train-split st3), the eot-bench AMI dev turn windows, and
TURN_DATA's held-out scopes (dev, devq, odev, odevq, atdev).

Stages (each one process, < 10 min, resumable; heavy ones through scripts/dev/gate.sh, one at a time):
  x115 --split S      the 115M's served inputs + pass-2 top + pass-1 blocks 4 / 8 + prosody on a held-out scope that
                      has no 115M cache yet (S = quiet | dev | devq | odev | odevq | atdev) -> TCW/x115/S/<id>.npz
  p115 --tags T       each 115M classifier's calibrated p on every held-out clip -> TCW/p115/<tag>.npz (+ base.npz:
                      the served VAD / TS-VAD / per-frame head p of the scopes x115 made)
  scan --sys C --tag T --fam conv|asst   every rule of the pre-registered grid on every held-out scope (resumable)
  pick --sys C         the pre-registered pick -> runs/turn_clean.json heldout.<core>
  test --sys C         shipped and clean picks, once, on the test rows -> runs/turn_clean.json test.<core>
The 115M v0.5 candidate build and its latency gate: turn_clean_serve.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import core_0p6b_turn as T6  # noqa: E402  (imports core_0p6b_heads, which repoints turn_v5 at the 0.6B caches)
import turn_v5 as V5  # noqa: E402

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
TCW = SSD / "scratch" / "turn_clean"
TD = SSD / "scratch" / "turndata"
OUT = ROOT / "runs" / "turn_clean.json"
V5W, V5BLK, V5NPZ = T6._V5_ORIG["W"], T6._V5_ORIG["BLK"], T6._V5_ORIG["npz"]
TAGS_115 = ("c1", "c2", "a_notext", "a_nost", "a_nocompl", "c3", "c4", "a_pool", "a_big", "c5", "c5s1", "c23")
# tracks the server cannot run today (picked tracks only get this note; the rule itself is unchanged)
NOT_SERVABLE = {"c3": "reads 12 prosody features per frame that the served engine does not compute (TURN_V5: not shipped)",
                "c23": "the mean of two classifiers (c2 and c3); c3 reads prosody features the served engine does not compute"}
TD_SPLITS = ("dev", "devq", "odev", "odevq", "atdev")
CONV = ("ho:calls", "ho:quiet", "ho:meet", "amidev") + TD_SPLITS
COMP = {"115m": 0.030, "0p6b": 0.041}
WINDOWS = (2.0, 2.5, 3.0, 3.5)
FRAME = 0.08
AMI_DEV = {"IS1008b", "ES2011b", "TS3004b", "IB4002"}
AMI_TEST = {"IS1009b", "ES2004b", "TS3003b", "EN2002a"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(key, val, sub=None):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    if sub is None:
        d[key] = val
    else:
        d.setdefault(key, {})[sub] = val
    OUT.write_text(json.dumps(d, indent=1, default=float))


def load_json(key, default=None):
    return json.loads(OUT.read_text()).get(key, default) if OUT.exists() else default


# --------------------------------------------------------------------------- grids (plans/fixwave/turn_clean.md §0.2)
def model_grid():
    return [{"mode": "model", "gate": True, "quiet_db": None, "k": k, "mvt": mvt, "mp": mp, "fb": fb, "vad_thr": 0.4,
             "reask": True, "others": (12, 8)}
            for k, mvt, mp, fb in itertools.product((1, 2, 3, 4), (0.4, 0.5, 0.6), (0.5, 0.6, 0.7, 0.8, 0.9, 0.95), (8, 9, 12))]


def head_grid():
    return [{"mode": "head", "gate": True, "quiet_db": None, "k": k, "th": th, "fb": fb, "vad_thr": vt, "others": (12, 8)}
            for k, th, fb, vt in itertools.product((1, 2, 3, 4), (0.9, 0.95, 0.97, 0.99), (6, 8, 10, 12), (0.4, 0.5, 0.6))]


def asst_grid():
    return [{"mode": "model", "gate": True, "quiet_db": qd, "mqo": qd is not None, "k": k, "mvt": mvt, "mp": mp, "fb": fb,
             "vad_thr": 0.4, "reask": True, "others": (12, 8)}
            for k, mvt, mp, qd, fb in itertools.product((2, 3, 4, 5), (0.4, 0.5), (0.8, 0.85, 0.9, 0.95, 0.97, 0.99),
                                                        (6.0, None), (37, 40, 43))]


def rkey(r):
    """A rule's identity (the fields run_policy reads)."""
    f = ("mode", "quiet_db", "k", "mvt", "mp", "th", "fb", "vad_thr", "rt")
    return tuple((k, r.get(k)) for k in f) + (("others", tuple(r.get("others") or ())), ("mqo", bool(r.get("mqo"))),
                                              ("reask", bool(r.get("reask"))), ("mclock", bool(r.get("mclock"))))


def shipped_rules(core):
    """The shipped presets as rule twins: 115M = turn_v5's PRESET_RULES (classifier c5, balanced = the v2 head);
    0.6B = the v0.4 served model's cfg turn_presets (assistant reads turn_seg_a = p5b)."""
    if core == "115m":
        P = V5.PRESET_RULES
        return {"balanced": ("head", dict(P["balanced"])), "fast": ("c5", dict(P["fast"])), "assistant": ("c5", dict(P["assistant"]))}
    rr = T6.served_rules(T6.cand_afm("v0.4"))
    out = {}
    for pn, r in rr.items():
        r = {k: (tuple(v) if k == "others" and v else v) for k, v in r.items()}
        out[pn] = ("head" if r["mode"] == "head" else "kd1stqr" if r.get("p5key") == "p5b" else "s12", r)
    return out


# --------------------------------------------------------------------------- 115M features on the scopes without a cache
def x115_items(split):
    """[(id, audio fn -> (x, print))] of a scope."""
    import turn_data as TDm
    if split == "quiet":
        import turn_v4 as V4
        au = V4.ClipAudio()
        man = T6._turn_manifest()
        return [(f"q_{c['id']}", (lambda c=c: T6.quiet_audio(c, au, man)[:2])) for c in T6._quiet_clips("val")]
    return [(c["id"], (lambda c=c: TDm.clip_audio(c))) for c in TDm.manifest(split)]


def stage_x115(a):
    import torch
    import turn_v4 as V4

    from audioforge.heads.prosody import prosody_frames
    from audioforge.tsvad_stream import voiceprint
    torch.set_num_threads(2)
    od = TCW / "x115" / a.split
    od.mkdir(parents=True, exist_ok=True)
    items = x115_items(a.split)
    todo = [it for it in items if not (od / f"{it[0]}.npz").exists()]
    log(f"x115 {a.split}: {len(todo)} of {len(items)} to do")
    if not todo:
        return
    t0, done = time.time(), 0
    ex1 = V4.Extractor(device=a.device)
    for i in range(0, len(todo), a.batch):
        if time.time() - t0 > a.budget:
            break
        cs = todo[i: i + a.batch]
        aud = [fn() for _, fn in cs]
        xs = [np.asarray(q[0], np.float32) for q in aud]
        prints = [voiceprint(ex1.cpu, q[1]) if len(q[1]) >= 8000 else None for q in aud]
        o1 = ex1(xs, prints)
        with torch.no_grad():
            x_, xl = ex1.m._pad(xs)
            enc, elen, hid = ex1.m.encode(x_, xl, V4.ATT, return_hidden=True)
            b4, b8 = hid[3].half().cpu().numpy(), hid[7].half().cpu().numpy()
        for j, ((cid, _), o) in enumerate(zip(cs, o1)):
            L = len(o["pu"])
            pros = prosody_frames(xs[j], L).astype(np.float16)
            tmp = od / f"{cid}.tmp.npz"
            np.savez(tmp, **o, b4=b4[j, :L], b8=b8[j, :L], pros=pros)
            tmp.rename(od / f"{cid}.npz")
        done += len(cs)
    log(f"x115 {a.split}: {done} in {time.time() - t0:.0f} s; {len(todo) - done} left")


# --------------------------------------------------------------------------- 115M classifier p on every held-out clip
def ho_inputs(cid, src):
    """(d, blocks npz path or None, prosody) of a held-out clip from the turn_v5 / x115 caches."""
    if src == "otoq":
        z = np.load(TCW / "x115" / "quiet" / f"{cid}.npz")
        d = {k: z[k] for k in z.files}
        return d, None, d["pros"]
    f = (V5W / "st3" / f"{cid}.npz") if src == "st3" else (V5.FEATS4 / f"{cid}.npz")
    return V5NPZ(f), V5BLK / f"{cid}.npz", np.load(V5.PROS / f"{cid}.npy")


def all_clips_115():
    """[(scope, id, loader -> (d, blk path or None, pros))] of every held-out clip with a 115M input."""
    out = [("ho", cid, (lambda cid=cid, s=m["src"]: ho_inputs(cid, s))) for cid, m in T6.ho_meta().items()]
    for sp in TD_SPLITS:
        d_ = TCW / "x115" / sp
        for f in sorted(d_.glob("*.npz")):
            if not f.name.endswith(".tmp.npz"):
                out.append((sp, f.stem, (lambda f=f: (lambda z: ({k: z[k] for k in z.files}, None, z["pros"]))(np.load(f)))))
    return out


def stage_p115(a):
    import torch
    import turn_v4 as V4
    torch.set_num_threads(2)
    (TCW / "p115").mkdir(parents=True, exist_ok=True)
    clips = all_clips_115()
    fb = TCW / "p115" / "base.npz"
    if not fb.exists():  # served VAD / TS-VAD / per-frame head p of the TURN_DATA scopes (ho: p_115m.npz)
        head = V4.load_head(None, a.device)
        out = {}
        for sc, cid, ld in clips:
            if sc == "ho":
                continue
            d, _, _ = ld()
            T = len(d["pu"])
            out[f"{cid}|p"] = V4.head_probs(head, d, a.device).astype(np.float32)[:T]
            for k in ("vad", "pu", "po"):
                out[f"{cid}|{k}"] = d[k].astype(np.float32)[:T]
        np.savez(fb, **out)
        log(f"p115 base: {len(out) // 4} clips")
    t0 = time.time()
    for tag in a.tags.split(","):
        f = TCW / "p115" / f"{tag}.npz"
        if f.exists() or tag == "c23":
            continue
        model, ck = V5.load_seg(V5W / tag / "model.pt", a.device)
        block, pros = str(ck["args"]["block"]), bool(ck["args"]["pros"])
        out = {}
        for _sc, cid, ld in clips:
            d, bp, pr = ld()
            T = len(d["pu"])
            x = (np.concatenate([d[f"b{b}"] for b in block.split("+")], 1) if bp is None else V5.load_block(d, bp, block))[:T]
            ex = np.stack([d["vad"], d["pu"], d["po"]], 1).astype(np.float32)[:T]
            if pros:
                ex = np.concatenate([ex, np.asarray(pr[:T], np.float32)], 1)
            out[cid] = V5.seg_probs(model, x.astype(np.float32), ex, d["n"][:T].astype(np.int32),
                                    d["y"].astype(np.int32), a.device).astype(np.float16)
        np.savez(f, **out)
        log(f"p115 {tag}: {len(out)} clips, {time.time() - t0:.0f} s")
        if time.time() - t0 > a.budget:
            break
    if "c23" in a.tags.split(",") and not (TCW / "p115" / "c23.npz").exists() \
            and all((TCW / "p115" / f"{t}.npz").exists() for t in ("c2", "c3")):
        z2, z3 = np.load(TCW / "p115" / "c2.npz"), np.load(TCW / "p115" / "c3.npz")
        np.savez(TCW / "p115" / "c23.npz", **{k: ((z2[k].astype(np.float32) + z3[k].astype(np.float32)) / 2).astype(np.float16)
                                              for k in z2.files})
        log("p115 c23 = mean(c2, c3)")


# --------------------------------------------------------------------------- scoring units
def ami_sessions(split):
    """eot_latency's AMI turn windows of the dev (selection) or eval (test) meetings, read directly (no env)."""
    import eot_latency as E
    d = E.WORK / ("ami_eval" if split == "eval" else "ami")
    out = []
    for r in json.loads((d / "clips.json").read_text()):
        out.append({"key": f"{r['name']}.mono", "set": "ami", "user_turns": [tuple(r["user_turn"])], "scored": [True],
                    "next_onset": [r["next_onset"]]})
    mt = {s["key"].split("_")[1] for s in out}
    assert mt <= (AMI_TEST if split == "eval" else AMI_DEV), mt
    return out


def _dumps(d):
    return {x["key"]: x for x in (json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json")))}


def units(core, track):
    """{scope: [(key, h, db, ss)]} of every held-out scope; h['p'] = the track's p ('head' = the per-frame head,
    else the classifier tag: 115M c1 ... c23; 0.6B s12 / kd1stqr)."""
    import turn_data as TDm
    meta = T6.ho_meta()
    U = {s: [] for s in CONV + ("ho:st3",)}
    if core == "115m":
        base = T6.load_ho("p_115m.npz")
        pc = None if track == "head" else np.load(TCW / "p115" / f"{track}.npz")
        zb = np.load(TCW / "p115" / "base.npz")
        bt = {}
        for k in zb.files:
            cid, kk = k.split("|")
            bt.setdefault(cid, {})[kk] = zb[k]
        get = lambda cid: dict(base.get(cid) or bt[cid], p5=None if pc is None else pc[cid].astype(np.float32))  # noqa: E731
    else:
        name = "mlp12r__kd1stqr__f1" if track == "kd1stqr" else "served__s12__f1"
        ho6 = T6.load_ho(f"p6_{name}.npz")
        if track == "kd1stqr":  # the GRU VAD of the served session; p5 = turn_seg_a (reads the turn VAD as input)
            hs = T6.load_ho("p6_served__s12__f1.npz")
            ho6 = {k: dict(hs[k], p5=v["p5"]) for k, v in ho6.items()}
        tds = {sp: TDm.load_sig(sp, "served__s12__f1") for sp in TD_SPLITS} if track != "kd1stqr" else {}
        get = lambda cid: ho6.get(cid)  # noqa: E731
    pk = "p" if track == "head" else "p5"
    for cid, m in meta.items():
        s = get(cid)
        T = min(len(s["vad"]), m["T"])
        h = {"t": [round((v + 1) * FRAME, 4) for v in range(T)], "p": np.asarray(s[pk][:T], np.float32),
             "vad": s["vad"][:T], "pu": s["pu"][:T], "po": s["po"][:T]}
        db = np.load(T6.HO / "db" / f"{cid}.npy")[:T]
        sc = {"oto": "ho:calls", "otoq": "ho:quiet", "st3": "ho:st3"}.get(m["src"], "ho:meet")
        U[sc].append((cid, h, db, {"key": cid, **m}))
    if core == "0p6b" and track == "kd1stqr":
        return {"ho:st3": U["ho:st3"]}
    for sp in TD_SPLITS:
        for c in TDm.manifest(sp):
            s = get(c["id"]) if core == "115m" else tds[sp].get(c["id"])
            if s is None:
                continue
            db = np.load(TD / "db" / f"{c['id']}.npy")
            T = min(len(s["vad"]), len(db))
            h = {"t": [round((v + 1) * FRAME, 4) for v in range(T)], "p": np.asarray(s[pk][:T], np.float32),
                 "vad": s["vad"][:T], "pu": s["pu"][:T], "po": s["po"][:T]}
            ss = {"user_turns": [(t[0], t[1]) for t in c["turns"]], "scored": [True] * len(c["turns"]),
                  "next_onset": [t[2] for t in c["turns"]]}
            U[sp].append((c["id"], h, db[:T], ss))
    # AMI dev turn windows from the served dumps (115M: turn_v4 dump_served + turn_v5 dump_<tag>; 0.6B: the v0.1 dump)
    if core == "115m":
        D = _dumps(V5.W4 / "dump_served")
        Dt = None if track == "head" else _dumps(V5W / f"dump_{track}")
    else:
        D, Dt = _dumps(T6.C.EOT_DUMP), None
    for ss in ami_sessions("dev"):
        d = D[ss["key"]]
        hd = d["head"]
        p = hd["p"] if track == "head" else (Dt[ss["key"]]["head"]["p"] if Dt is not None else hd["p5"])
        v = np.asarray(hd["v"])
        db = np.load(V5.ENERGY / f"{ss['key']}.npy")
        h = {"t": hd["t"], "p": np.asarray(p, np.float32), "vad": hd["vad"], "pu": hd["pu"], "po": hd["po"]}
        U["amidev"].append((ss["key"], h, db[np.clip(v, 0, len(db) - 1)], ss))
    return U


def score_scope(us, rule, comp, gcs, asst=False):
    import eot_assistant as EA
    import eot_latency as E
    per, times, comps, ss_ = [], {}, {}, []
    for key, h, db, ss in us:
        tt = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, gcs.setdefault(key, {}))]
        if asst:
            times[key], comps[key] = tt, comp
            ss_.append(ss)
        else:
            per.append(E.score_session(tt, ss, comp))
    if asst:
        a_ = EA.score_system(ss_, times, comps)
        return {"accuracy_pct": a_["accuracy_pct"], "p50": a_["complete"]["eot_total_ms_p50"],
                "false_fire_pct": a_["incomplete"]["false_fire_pct"]}
    r = E.pool(per)
    return {k: r[k] for k in ("n_turns", "eot_total_ms_p50", "false_interruption_pct", "missed_pct")}


def stage_scan(a):
    """--fam conv: the model grid (track = a classifier) or the head grid (--tag head) on the 9 conversational scopes;
    --fam asst: the assistant grid on st3. -> TCW/scan/<core>_<tag>_<fam>.json (resumable, --budget)."""
    (TCW / "scan").mkdir(parents=True, exist_ok=True)
    f = TCW / "scan" / f"{a.core}_{a.tag}_{a.fam}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    G = grid_for(a.core, a.tag, a.fam)
    if len(rows) >= len(G):
        log("done", f)
        return
    U = units(a.core, a.tag)
    comp = COMP[a.core]
    gcs = {}
    t0 = time.time()
    for r in G[len(rows):]:
        if a.fam == "asst":
            res = {"ho:st3": score_scope(U["ho:st3"], r, comp, gcs, asst=True)}
        else:
            res = {sc: score_scope(U[sc], r, comp, gcs) for sc in CONV}
        rows.append({"rule": r, "res": res})
        if time.time() - t0 > a.budget:
            break
    f.write_text(json.dumps(rows))
    log(f"scan {a.core} {a.tag} {a.fam}: {len(rows)}/{len(G)} rules, {time.time() - t0:.0f} s")


def grid_for(core, tag, fam):
    """The pre-registered grid, with the shipped rule of that track prepended when the grid lacks it."""
    G = asst_grid() if fam == "asst" else head_grid() if tag == "head" else model_grid()
    have = {rkey(r) for r in G}
    extra = []
    for pn, (tr, r) in shipped_rules(core).items():
        if tr == tag and (pn == "assistant") == (fam == "asst") and rkey(r) not in have:
            extra.append(dict(r))
            have.add(rkey(r))
    return extra + G


# --------------------------------------------------------------------------- the pick (plans/fixwave/turn_clean.md §0.3)
def tracks(core, preset):
    if core == "115m":
        return (("head",) + TAGS_115) if preset == "balanced" else TAGS_115
    return {"balanced": ("head", "s12"), "fast": ("s12",), "assistant": ("kd1stqr",)}[preset]


def rows_of(core, preset):
    fam = "asst" if preset == "assistant" else "conv"
    out = []
    for tr in tracks(core, preset):
        f = TCW / "scan" / f"{core}_{tr}_{fam}.json"
        rows = json.loads(f.read_text())
        assert len(rows) == len(grid_for(core, tr, fam)), (f, len(rows))
        out += [{"track": tr, **x} for x in rows]
    return out


def mean_p50(res):
    return float(np.mean([res[s]["eot_total_ms_p50"] for s in CONV]))


def violations(res, ref, preset):
    """(number of violated constraints, summed excess in pp) against the reference's held-out results."""
    if preset == "assistant":
        a_, b_ = res["ho:st3"], ref["ho:st3"]
        ex = [max(0.0, b_["accuracy_pct"] - a_["accuracy_pct"]), max(0.0, a_["false_fire_pct"] - b_["false_fire_pct"])]
    else:
        ex = []
        for s in CONV:
            if res[s]["eot_total_ms_p50"] is None:
                return 99, 1e9
            ex += [max(0.0, res[s]["false_interruption_pct"] - ref[s]["false_interruption_pct"]),
                   max(0.0, res[s]["missed_pct"] - ref[s]["missed_pct"])]
    return sum(e > 1e-9 for e in ex), round(sum(ex), 2)


def choose(rows, ref, preset):
    def speed(x):
        r = x["res"]
        if preset == "assistant":
            a_ = r["ho:st3"]
            return (a_["p50"] if a_["p50"] is not None else 1e9, -a_["accuracy_pct"], a_["false_fire_pct"])
        if any(r[s]["eot_total_ms_p50"] is None for s in CONV):
            return (1e9,)
        return (mean_p50(r), float(np.mean([r[s]["false_interruption_pct"] for s in CONV])),
                float(np.mean([r[s]["missed_pct"] for s in CONV])))
    ok = [x for x in rows if violations(x["res"], ref, preset)[0] == 0]
    if ok:
        return min(ok, key=speed), len(ok), True
    return min(rows, key=lambda x: (*violations(x["res"], ref, preset), speed(x))), 0, False


def find(rows, track, rule):
    k = rkey(rule)
    return next(x for x in rows if x["track"] == track and rkey(x["rule"]) == k)


def stage_pick(a):
    res = load_json("heldout", {})
    ship = shipped_rules(a.core)
    out = {}
    for preset in ("balanced", "fast", "assistant"):
        rows = rows_of(a.core, preset)
        tr, r = ship[preset]
        shipped = find(rows, tr, r)
        if a.core == "115m":
            ref, ref_name = shipped["res"], "115M shipped preset (held-out replay)"
        else:
            p115 = res["115m"][preset]["pick"]
            ref, ref_name = p115["res"], "115M clean held-out pick (held-out replay)"
        pk, n_ok, passed = choose(rows, ref, preset)
        same = pk["track"] == tr and rkey(pk["rule"]) == rkey(r)
        out[preset] = {"reference": ref_name, "n_candidates": len(rows), "n_pass": n_ok, "strict_pass": passed,
                       "pick": pk, "shipped": shipped, "pick_equals_shipped": same,
                       "shipped_violations_vs_reference": violations(shipped["res"], ref, preset)}
        if pk["track"] in NOT_SERVABLE:  # post-hoc filter, stated as such: the same rule over servable tracks only
            pks, n_s, _ = choose([x for x in rows if x["track"] not in NOT_SERVABLE], ref, preset)
            out[preset].update({"pick_servable": pks, "n_pass_servable": n_s,
                                "servable_equals_shipped": pks["track"] == tr and rkey(pks["rule"]) == rkey(r),
                                "not_servable_reason": NOT_SERVABLE[pk["track"]]})
        f = lambda x, pr=preset: (x["track"], {k: v for k, v in x["rule"].items() if v is not None},  # noqa: E731
                                  (x["res"]["ho:st3"] if pr == "assistant" else round(mean_p50(x["res"]))))
        log(f"{a.core} {preset}: {n_ok}/{len(rows)} pass (strict {passed}); pick {f(pk)}; shipped {f(shipped)}; same {same}")
    res[a.core] = out
    save("heldout", res)


# --------------------------------------------------------------------------- test rows, scored once
def test_sets(core):
    """(AMI test dumps, assistant dumps) of the served engines (final_compare eotdump; 0.6B = the v0.4 dumps)."""
    import eot_assistant as EA
    W = SSD / "scratch" / ("final_compare/eot" if core == "115m" else "turndata/fc_v04/eot")
    keys = {s["key"] for s in ami_sessions("eval")}
    D = {k: v for k, v in _dumps(W / f"{core}_calls").items() if k in keys}
    A = _dumps(W / f"{core}_asst")
    sd = EA.load_dump()
    for k, d in A.items():
        d["conf"] = sd[k]["conf"]
    assert len(D) == len(keys) == 200 and len(A) == 399, (len(D), len(A))
    return D, A


def test_p(core, track, D, A):
    """p of a track on the test dumps' frames: the dumped p (head) / p5 (shipped classifier) / p5b (0.6B turn_seg_a);
    another 115M classifier on the 399: turn_v5's offline asst_p_<tag> (no AMI-test cache: raises)."""
    k = {"head": "p", "c5": "p5", "s12": "p5", "kd1stqr": "p5b"}.get(track)
    if k is not None:
        return ({kk: d["head"][k] for kk, d in D.items()}, {kk: d["head"][k] for kk, d in A.items()})
    ap = json.loads((V5W / f"asst_p_{track}.json").read_text())
    pa = {kk: [ap[kk][min(j, len(ap[kk]) - 1)] for j in d["head"]["v"]] for kk, d in A.items()}
    z = np.load(TCW / "p115" / f"amitest_{track}.npz")  # stage x115test (scoring only, after the pick)
    pd_ = {kk: [float(z[kk][min(j, len(z[kk]) - 1)]) for j in d["head"]["v"]] for kk, d in D.items()}
    return pd_, pa


def stage_x115test(a):
    """After the pick only: the 115M inputs of the 200 AMI test turn windows (one window per batch, as turn_v4's
    evalfeats made the dev ones: the served print, the masked forward) and --tags' classifier p on every frame
    -> TCW/p115/amitest_<tag>.npz. Also c5, to check the extraction against the served dumps' p5."""
    import eot_latency as E
    import torch
    import turn_v4 as V4

    from audioforge.heads.prosody import prosody_frames
    torch.set_num_threads(2)
    od = TCW / "x115" / "amitest"
    od.mkdir(parents=True, exist_ok=True)
    refs = json.loads((E.WORK / "ami_eval" / "clips.json").read_text())
    todo = [r for r in refs if not (od / f"{r['name']}.mono.npz").exists()]
    t0 = time.time()
    if todo:
        ex1 = V4.Extractor(device=a.device)
        for r in todo:
            if time.time() - t0 > a.budget:
                log(f"x115test: budget, {len(todo)} left; rerun")
                return
            x = E.read_audio({"wav": str(E.WORK / "ami_eval" / f"{r['name']}.wav"), "pad_s": 0.0})
            o = ex1([x], [r["embedding"]])[0]
            with torch.no_grad():
                x_, xl = ex1.m._pad([x])
                _, _, hid = ex1.m.encode(x_, xl, V4.ATT, return_hidden=True)
            L = len(o["pu"])
            np.savez(od / f"{r['name']}.mono.npz", **o, b4=hid[3][0, :L].half().cpu().numpy(),
                     b8=hid[7][0, :L].half().cpu().numpy(), pros=prosody_frames(x, L).astype(np.float16))
    for tag in a.tags.split(","):
        f = TCW / "p115" / f"amitest_{tag}.npz"
        if f.exists():
            continue
        if tag == "c23":
            z2, z3 = np.load(TCW / "p115" / "amitest_c2.npz"), np.load(TCW / "p115" / "amitest_c3.npz")
            np.savez(f, **{k: (z2[k] + z3[k]) / 2 for k in z2.files})
            continue
        model, ck = V5.load_seg(V5W / tag / "model.pt", a.device)
        block, pros = str(ck["args"]["block"]), bool(ck["args"]["pros"])
        out = {}
        for r in refs:
            z = np.load(od / f"{r['name']}.mono.npz")
            T = len(z["pu"])
            x = np.concatenate([z[f"b{b}"] for b in block.split("+")], 1)[:T].astype(np.float32)
            ex = np.stack([z["vad"], z["pu"], z["po"]], 1).astype(np.float32)
            if pros:
                ex = np.concatenate([ex, z["pros"][:T].astype(np.float32)], 1)
            out[f"{r['name']}.mono"] = V5.seg_probs(model, x, ex, z["n"].astype(np.int32), z["y"].astype(np.int32), a.device)
        np.savez(f, **out)
        log(f"x115test {tag}: {len(out)} windows, {time.time() - t0:.0f} s")


def score_test(D, A, pD, pA, rule, boot=1000, seed=0):
    import eot_assistant as EA
    import eot_latency as E
    sess = ami_sessions("eval")
    per = []
    for ss in sess:
        d = D[ss["key"]]
        h = dict(d["head"], p=pD[ss["key"]])
        comp = d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"{ss['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        per.append(E.score_session([x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})], ss, comp))
    asess = EA.sessions(A)
    times, comps = {}, {}
    for s_ in asess:
        d = A[s_["key"]]
        h = dict(d["head"], p=pA[s_["key"]])
        comp = d["chunk_ms"]["p50"] / 1000
        db = np.load(V5.ENERGY / f"asst_{s_['key']}.npy")
        db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
        times[s_["key"]] = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        comps[s_["key"]] = comp

    def asst(ss):
        sc = EA.score_system(ss, times, comps)
        return {"accuracy_pct": sc["accuracy_pct"], "p50": sc["complete"]["eot_total_ms_p50"],
                "p95": sc["complete"]["eot_total_ms_p95"], "false_fire_pct": sc["incomplete"]["false_fire_pct"],
                "missed_pct": sc["complete"]["missed_pct"]}

    def ci(rows, q):
        x = np.array([r_[q] for r_ in rows if r_[q] is not None], float)
        return [round(float(np.percentile(x, 2.5)), 1), round(float(np.percentile(x, 97.5)), 1)] if len(x) else None
    rng = np.random.default_rng(seed)
    out = {"ami_test": E.pool(per)}
    bs = [E.pool([per[i] for i in rng.integers(0, len(per), len(per))]) for _ in range(boot)]
    out["ami_test"]["ci95"] = {q: ci(bs, q) for q in ("eot_total_ms_p50", "false_interruption_pct", "missed_pct")}
    for w in WINDOWS:
        EA.FF_WINDOW = w
        o = asst(asess)
        rng = np.random.default_rng(seed)
        ba = [asst([asess[i] for i in rng.integers(0, len(asess), len(asess))]) for _ in range(boot)]
        o["ci95"] = {q: ci(ba, q) for q in ("accuracy_pct", "p50", "false_fire_pct")}
        out[f"smartturn_test_w{w:g}"] = o
    EA.FF_WINDOW = 3.0
    return out


def stage_test(a):
    """Once: the shipped presets and the clean held-out picks on the test rows -> runs/turn_clean.json test.<core>."""
    ho = load_json("heldout")[a.core]
    D, A = test_sets(a.core)
    res = load_json("test", {})
    out = {}
    for preset, h in ho.items():
        cfgs = {"shipped": (h["shipped"]["track"], h["shipped"]["rule"])}
        if not h["pick_equals_shipped"]:
            cfgs["clean"] = (h["pick"]["track"], h["pick"]["rule"])
        if "pick_servable" in h and not h["servable_equals_shipped"]:
            cfgs["clean_servable"] = (h["pick_servable"]["track"], h["pick_servable"]["rule"])
        out[preset] = {}
        for lab, (tr, r) in cfgs.items():
            r = dict(r, others=tuple(r["others"]) if r.get("others") else None)
            pD, pA = test_p(a.core, tr, D, A)
            o = score_test(D, A, pD, pA, r, boot=a.boot)
            out[preset][lab] = {"track": tr, "rule": r, **o}
            s3 = o["smartturn_test_w3"]
            log(f"{a.core} {preset} {lab} ({tr}): AMI test {o['ami_test']['eot_total_ms_p50']} / "
                f"{o['ami_test']['false_interruption_pct']} / {o['ami_test']['missed_pct']}; smart-turn test (3 s) "
                f"{s3['accuracy_pct']} / {s3['p50']} / {s3['false_fire_pct']}")
        out[preset]["clean_equals_shipped"] = h["pick_equals_shipped"]
    res[a.core] = out
    save("test", res)


STAGES = {"x115": stage_x115, "x115test": stage_x115test, "p115": stage_p115, "scan": stage_scan, "pick": stage_pick, "test": stage_test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tags", default=",".join(TAGS_115))
    ap.add_argument("--tag", default="c5")
    ap.add_argument("--sys", dest="core", default="115m", choices=("115m", "0p6b"))  # not --core: core_0p6b_heads reads argv
    ap.add_argument("--fam", default="conv", choices=("conv", "asst"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--budget", type=float, default=500)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
