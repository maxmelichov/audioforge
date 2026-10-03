# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "jiwer>=3", "transformers>=4.40", "soundfile"]
# ///
"""Independent re-derivation of a sample of runs/final_compare.json numbers from the saved per-item outputs
(plans/audit/metrics_001.md). Read-only: reads the SSD scratch outputs, writes nothing.

    .venv/bin/python plans/audit/metrics_001.py asr|vad|lid|turn|all
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch")
W = SSD / "final_compare"
FC = json.loads((ROOT / "runs/final_compare.json").read_text())


def rows(p):
    return {r["i"]: r for r in map(json.loads, Path(p).read_text().splitlines()) if r} if Path(p).exists() else {}


# ------------------------------------------------------------------ ASR
def refs_for(name):
    if name in ("libri", "ami"):
        return json.loads((SSD / "final_asr" / f"{name}_refs.json").read_text())
    if name == "icsi":
        return json.loads((SSD / "hybrid_asr" / "icsi_refs.json").read_text())
    if name.endswith("_eval"):
        return json.loads((W / "asr_sets" / f"{name}_refs.json").read_text())
    if name == "live":
        out = []
        for c in json.loads((SSD / "e2e_tsvad" / "clips.json").read_text()):
            if c["set"] != "turnbench" or not c.get("text_user"):
                continue
            out += [c["text_mono"], c["text_user"]]
        return out
    if name == "fleurs_en":
        tsv = ROOT / "data/lid/fleurs/_tsv/data/en_us/test.tsv"
        ref = {r[1].rsplit(".", 1)[0]: r[2] for r in (ln.split("\t") for ln in tsv.read_text().splitlines() if ln)}
        rr = [r for r in map(json.loads, (ROOT / "data/lid/fleurs/manifest.jsonl").read_text().splitlines())
              if r.get("lang") == "en" and r.get("split") == "test"]
        return [ref[r["id"]] for r in rr]
    split = {"ls_clean": "test-clean", "ls_other": "test-other"}[name]
    items = []
    for tr in sorted((ROOT / "data/librispeech/LibriSpeech" / split).glob("*/*/*.trans.txt")):
        for ln in tr.read_text().splitlines():
            items.append(ln.split(" ", 1)[1].lower())
    return [items[i] for i in sorted(random.Random(0).sample(range(len(items)), 300))]


def whisper_norm():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    hub = Path.home() / ".cache/huggingface/hub/models--openai--whisper-tiny/snapshots"
    nj = next(hub.glob("*/normalizer.json"))
    return EnglishTextNormalizer(json.loads(nj.read_text()))


def asr():
    import jiwer
    norm = whisper_norm()
    systems = ["whisper_small", "whisper_turbo", "whisper_large_v3", "tdt_v3", "core_115m", "core_115m_beam8",
               "core_0p6b", "core_115m_f1120", "core_0p6b_f1120"]
    for s_name in ("libri", "ami", "icsi", "live", "ami_eval", "icsi_eval", "ls_clean", "ls_other", "fleurs_en"):
        refs = refs_for(s_name)
        rep = FC["words"].get(s_name, {}).get("whisper_norm|all", {})
        for s in systems:
            h = rows(W / "asr" / f"{s}_{s_name}.jsonl")
            if len(h) < len(refs):
                if h:
                    print(f"{s_name:10s} {s:18s} INCOMPLETE {len(h)}/{len(refs)}")
                continue
            R = [norm(r) for r in refs]
            H = [norm(h[i]["hyp"]) for i in range(len(refs))]
            err = nref = 0
            empty_h = sum(1 for x in H if not x.strip())
            for r, y in zip(R, H):
                nr = len(r.split())
                if nr == 0:
                    err += len(y.split())
                    continue
                o = jiwer.process_words(r, y if y.strip() else "")  # jiwer 3 accepts empty hyp
                err += o.substitutions + o.deletions + o.insertions
                nref += nr
            # long-form completeness: hyp words per audio second vs ref
            mine = 100 * err / nref
            r_ = rep.get(s, {})
            flag = "" if abs(mine - r_.get("wer_pct", -99)) < 0.006 else "  <-- MISMATCH"
            print(f"{s_name:10s} {s:18s} mine {mine:6.2f} ({err}/{nref})  reported {r_.get('wer_pct')} "
                  f"({r_.get('errors')}/{r_.get('ref_words')}) empty_hyp={empty_h}{flag}")


def longform():
    """Whisper outputs on items > 30 s: hyp words vs ref words (truncation check)."""
    norm = whisper_norm()
    for s_name in ("live", "libri", "ami", "icsi", "fleurs_en", "ls_clean", "ls_other", "ami_eval", "icsi_eval"):
        refs = refs_for(s_name)
        for s in ("whisper_small", "whisper_turbo", "whisper_large_v3", "tdt_v3", "core_0p6b"):
            h = rows(W / "asr" / f"{s}_{s_name}.jsonl")
            long_ = [i for i in h if h[i]["audio_sec"] > 30]
            if not long_:
                continue
            rat = [len(norm(h[i]["hyp"]).split()) / max(1, len(norm(refs[i]).split())) for i in sorted(long_)]
            print(f"{s_name} {s}: {len(long_)} items >30 s; hyp/ref word ratio min {min(rat):.2f} "
                  f"median {np.median(rat):.2f}  per item {[round(x, 2) for x in rat]}")


# ------------------------------------------------------------------ VAD
def vad():
    from sklearn.metrics import roc_auc_score
    for sn in ("ami_dev", "icsi_dev", "ami_eval", "icsi_eval"):
        rep = FC["vad"].get(sn, {})
        for s in ("silero_v5", "marblenet_v2", "pyannote_seg3", "ten_vad", "core_115m", "core_0p6b"):
            f = W / "vad" / f"{s}_{sn}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            p, t = z["p"].astype(np.float64), z["y"] > 0.5
            d = p > 0.5
            tp, fp, fn = (d & t).sum(), (d & ~t).sum(), (~d & t).sum()
            f1 = 2 * tp / (2 * tp + fp + fn)
            auc = roc_auc_score(t, p)
            neg = np.sort(p[~t])
            th = np.quantile(neg, 0.925)
            fpr_real = (neg > th).mean()
            miss = (p[t] <= th).mean()
            r = rep.get(s, {})
            print(f"{sn:9s} {s:14s} F1 {f1:.4f} (rep {r.get('f1_at_0.5')})  AUC {auc:.4f} (rep {r.get('auc')})  "
                  f"miss@7.5 {100 * miss:.2f} (rep {r.get('miss_at_fpr_7.5_pct')})  real FPR at th {100 * fpr_real:.2f}% "
                  f"p in [{p.min():.3f},{p.max():.3f}] uniq {len(np.unique(p))} zeros {(p == 0).mean():.3f}")


def vad_lastframe():
    """Label frame 250 of each 20 s window spans [20.00, 20.08) s, past the audio: silero / ten / pyannote score it 0.
    Re-score every system without that frame."""
    from sklearn.metrics import roc_auc_score
    for sn in ("ami_dev", "icsi_dev", "ami_eval", "icsi_eval"):
        for s in ("silero_v5", "marblenet_v2", "pyannote_seg3", "ten_vad", "core_115m", "core_0p6b"):
            z = np.load(W / "vad" / f"{s}_{sn}.npz")
            off = np.cumsum(z["lens"])
            keep = np.ones(len(z["y"]), bool)
            keep[off - 1] = False
            p, t = z["p"][keep].astype(np.float64), z["y"][keep] > 0.5
            d = p > 0.5
            tp, fp, fn = (d & t).sum(), (d & ~t).sum(), (~d & t).sum()
            th = np.quantile(p[~t], 0.925)
            r = FC["vad"][sn][s]
            print(f"{sn:9s} {s:14s} F1 {2 * tp / (2 * tp + fp + fn):.4f} (rep {r['f1_at_0.5']})  AUC "
                  f"{roc_auc_score(t, p):.4f} (rep {r['auc']})  miss {100 * (p[t] <= th).mean():.2f} "
                  f"(rep {r['miss_at_fpr_7.5_pct']})")


def lid():
    rep = FC["lid"]
    codes = rep["languages"]
    for s in ("whisper_small", "whisper_turbo", "whisper_large_v3", "core_0p6b", "core_115m"):
        f = W / "lid" / f"{s}.npz"
        if not f.exists():
            print(s, "no npz; source", rep.get(s, {}).get("source"))
            continue
        z = np.load(f)
        y = z["y"]
        a2 = 100 * (z["P"][:, 0].argmax(-1) == y).mean()
        af = 100 * (z["P"][:, 1].argmax(-1) == y).mean()
        r = rep[s]
        print(f"{s:16s} acc2s {a2:.2f} (rep {r['acc_2s_pct']})  full {af:.2f} (rep {r['acc_full_pct']})  n {len(y)} "
              f"rowsum {z['P'][:, 0].sum(-1).mean():.3f} src {r['source'][-40:]}")
    d = ROOT / "data/lid/preds/ambernet"
    P = np.concatenate([np.load(d / f"{c}.npz")["P"] for c in codes])
    y = np.concatenate([[i] * len(np.load(d / f"{c}.npz")["P"]) for i, c in enumerate(codes)])
    print(f"ambernet         acc2s {100 * (P[:, 1].argmax(-1) == y).mean():.2f} (rep {rep['ambernet']['acc_2s_pct']}) "
          f"full {100 * (P[:, -1].argmax(-1) == y).mean():.2f} (rep {rep['ambernet']['acc_full_pct']})")


# ------------------------------------------------------------------ turn
TOL, HOR, FF = 0.08, 6.0, 3.0


def my_calls(times, sess):
    """own scorer: per scored turn, interrupted if a fire in [s0, e1-TOL); answered by the first fire in
    [e1-TOL, min(e1+6, next onset)); latency = fire - e1."""
    lat, n, intr, miss = [], 0, 0, 0
    for ss in sess:
        T = np.sort(np.asarray(times[ss["key"]], float))
        for (s0, e1), sc, nx in zip(ss["user_turns"], ss["scored"], ss["next_onset"]):
            if not sc:
                continue
            n += 1
            intr += bool(((T >= s0) & (T < e1 - TOL)).any())
            h = e1 + HOR if nx is None else min(e1 + HOR, nx)
            a = T[(T >= e1 - TOL) & (T < h)]
            if len(a):
                lat.append(a[0] - e1)
            else:
                miss += 1
    return {"n": n, "p50": int(round(1000 * np.median(lat))), "p95": int(round(1000 * np.percentile(lat, 95))),
            "fi": round(100 * intr / n, 1), "miss": round(100 * miss / n, 1)}


def my_asst(times, asess):
    ok, comp_lat, n_c, miss_c, ff, n_i = 0, [], 0, 0, 0, 0
    for s in asess:
        T = np.sort(np.asarray(times[s["key"]], float))
        T = T[T >= s["onset"]]  # session-start rule: fires before the first Silero speech chunk are ignored
        e, h = s["user_turns"][0][1], s["next_onset"][0]
        if s["complete"]:
            n_c += 1
            early = ((T < e - TOL)).any()
            a = T[(T >= e - TOL) & (T < min(e + HOR, h))]
            if len(a):
                comp_lat.append(a[0] - e)
            else:
                miss_c += 1
            ok += (not early) and len(a) > 0
        else:
            n_i += 1
            f = bool((T[T < h] < e + FF).any())
            ff += f
            ok += not f
    return {"acc": round(100 * ok / len(asess), 1), "p50": int(round(1000 * np.median(comp_lat))),
            "p95": int(round(1000 * np.percentile(comp_lat, 95))), "ff": round(100 * ff / n_i, 1),
            "miss": round(100 * miss_c / n_c, 1)}


def _cmp(name, mine, rep_calls, rep_ami, rep_asst):
    def show(lbl, m, r, keys):
        bad = [f"{a}: mine {m[a]} rep {r.get(b)}" for a, b in keys if r.get(b) is not None and abs(m[a] - r[b]) > 0.11]
        print(f"  {name:34s} {lbl:5s} {m}  {'MATCH' if not bad else 'DIFF ' + '; '.join(bad)}")
    kc = [("p50", "eot_total_ms_p50"), ("p95", "eot_total_ms_p95"), ("fi", "false_interruption_pct"), ("miss", "missed_pct")]
    show("calls", mine["calls"], rep_calls, kc)
    show("ami", mine["ami"], rep_ami, kc)
    show("asst", mine["asst"], rep_asst, [("acc", "accuracy_pct"), ("p50", "p50"), ("p95", "p95"), ("ff", "false_fire_pct"),
                                          ("miss", "missed_pct")])


def turn():
    import os
    os.environ["EOT_AMI_SPLIT"] = FC["ami_turn_split"]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]
    import eot_assistant as EA
    import eot_latency as E
    rep = FC["turn"]
    sess = E.sessions()
    calls = [s for s in sess if s["set"] in ("turnbench", "oto")]
    ami = [s for s in sess if s["set"] == "ami"]
    asess = EA.sessions(EA.load_dump())
    print(f"sessions: calls {len(calls)} ami {len(ami)} asst {len(asess)}")

    def run(name, tc, ta):
        mine = {"calls": my_calls(tc, calls), "ami": my_calls(tc, ami), "asst": my_asst(ta, asess)}
        r = rep[name] if "|" not in name else rep[name.split("|")[0]][name.split("|")[1]]
        _cmp(name, mine, r["calls"], r["ami"], r["asst"])
        return mine
    bc = {d["key"]: d for d in (json.loads(p.read_text()) for p in E.BASE.glob("*.json"))}
    ba = {d["key"]: d for d in (json.loads(p.read_text()) for p in EA.BASE.glob("*.json"))}
    for bk, name in (("pipecat_smartturn", "pipecat_smartturn_v3.2_silero"), ("livekit_eou", "livekit_en_turn_detector_silero")):
        add = (lambda e: e["t"] + e["compute_ms"] / 1000) if bk == "pipecat_smartturn" else (lambda e: e["t"])
        run(name, {k: [add(e) for e in bc[k][bk]] for k in bc}, {k: [add(e) for e in ba[k][bk]] for k in ba})
    ec = {p.stem: json.loads(p.read_text()) for p in (W / "eou" / "calls").glob("*.json")}
    ea = {p.stem: json.loads(p.read_text()) for p in (W / "eou" / "asst").glob("*.json")}
    f = lambda D: {k: [e["t_dec"] + e["compute_s"] for e in d["ends"]] for k, d in D.items()}  # noqa: E731
    run("parakeet_realtime_eou", f(ec), f(ea))
    # ours: rule twin times (turn_v5.run_policy, the served VadHeadPolicy) + median chunk compute, own scoring
    import turn_v5 as V5
    for sysn, d0 in (("115m", W / "eot"), ("0p6b", SSD / "turndata/fc_v04/eot")):
        rules = rep[f"ours_{sysn}"]["rules"]
        D = {p.stem: json.loads(p.read_text()) for p in (d0 / f"{sysn}_calls").glob("*.json")}
        A = {p.stem: json.loads(p.read_text()) for p in (d0 / f"{sysn}_asst").glob("*.json")}
        for pn, rule in rules.items():
            pk = "p" if rule["mode"] == "head" else rule.get("p5key", "p5")
            tc, ta = {}, {}
            for DD, out, pre in ((D, tc, ""), (A, ta, "asst_")):
                for k, d in DD.items():
                    h = dict(d["head"])
                    h["p"] = h[pk]
                    comp = d["chunk_ms"]["p50"] / 1000
                    db = np.load(V5.ENERGY / f"{pre}{k}.npy")
                    db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
                    out[k] = [x + comp for x, _ in V5.run_policy({"head": h}, db, rule, {})]
            run(f"ours_{sysn}|{pn}", tc, ta)


# ------------------------------------------------------------------ speaker EER
def _my_eer(s, lab):
    """EER by a threshold sweep over every score: the point where FAR and FRR cross (mean of the two there)."""
    o = np.argsort(-s)
    lab = lab[o].astype(bool)
    tp = np.cumsum(lab)
    fp = np.cumsum(~lab)
    frr = 1 - tp / lab.sum()
    far = fp / (~lab).sum()
    i = np.argmin(np.abs(frr - far))
    return (frr[i] + far[i]) / 2


def eer():
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]
    import final_compare as F
    rep = FC["speaker_eer"]
    for c in ("ami", "icsi", "ami_eval", "icsi_eval"):
        val = F.spk_segments(c)
        spk = np.array([v["speaker"] for v in val])
        meet = np.array([v["meeting"] for v in val])
        iu = np.triu_indices(len(val), 1)
        same_m = meet[iu[0]] == meet[iu[1]]
        lab = (spk[iu[0]] == spk[iu[1]])[same_m]
        print(f"{c}: {len(val)} segments, {len(set(meet))} meetings, {same_m.sum()} within-meeting pairs "
              f"({lab.sum()} target)")
        for s in ("core_115m", "core_0p6b", "titanet_l", "wespeaker_pyannote"):
            f = W / "spk" / f"{s}_{c}.npy"
            if not f.exists():
                continue
            X = np.load(f)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            sc = (X[iu[0]] * X[iu[1]]).sum(1)[same_m]
            print(f"   {s:20s} EER {100 * _my_eer(sc, lab):.2f}  (rep {rep[c].get(s, {}).get('eer_within_meeting_pct')})")


def dualrate():
    """The 1.12 s final rows (runs/dual_rate.json wer) re-derived from scratch/dualrate/wer/*.jsonl."""
    norm = whisper_norm()
    dr = json.loads((ROOT / "runs/dual_rate.json").read_text())["wer"]
    for core in ("115m", "0p6b"):
        for s in ("ls_clean", "ls_other", "ami_eval", "icsi_eval", "live"):
            refs = refs_for(s)
            R = [norm(r) for r in refs]
            out = []
            for r in (0, 1, 6, 13):
                h = rows(SSD / "dualrate/wer" / f"{core}_r{r}_{s}.jsonl")
                if len(h) < len(refs):
                    continue
                e = n = 0
                for i, ref in enumerate(R):
                    a, b = ref.split(), norm(h[i]["hyp"]).split()
                    e += _lev(a, b)
                    n += len(a)
                rep = dr.get(f"{core}|{s}", {}).get(f"r{r}", {}).get("wer_pct")
                out.append(f"r{r} {100 * e / n:.2f} (rep {rep})")
            fc1 = rows(W / "asr" / f"core_{core}_{s}.jsonl")
            h1 = rows(SSD / "dualrate/wer" / f"{core}_r1_{s}.jsonl")
            same = sum(fc1[i]["hyp"] == h1[i]["hyp"] for i in range(len(refs))) if fc1 and len(h1) >= len(refs) else None
            print(f"{core} {s:9s} " + "  ".join(out) + f"  r1 hyps == final_compare core rows: {same}/{len(refs)}")


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def turn_window():
    """Sensitivity of the assistant-clip accuracy / false fire to the 3 s 'not cut' window (Pipecat's stop_secs and
    LiveKit's max_delay defaults are both 3.0 s, so their fallbacks land on the window edge)."""
    global FF
    import os
    os.environ["EOT_AMI_SPLIT"] = FC["ami_turn_split"]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts/research")]
    import eot_assistant as EA
    import turn_v5 as V5
    asess = EA.sessions(EA.load_dump())
    ba = {d["key"]: d for d in (json.loads(p.read_text()) for p in EA.BASE.glob("*.json"))}
    S = {"pipecat": {k: [e["t"] + e["compute_ms"] / 1000 for e in ba[k]["pipecat_smartturn"]] for k in ba},
         "livekit": {k: [e["t"] for e in ba[k]["livekit_eou"]] for k in ba}}
    for sysn, d0 in (("115m", W / "eot"), ("0p6b", SSD / "turndata/fc_v04/eot")):
        A = {p.stem: json.loads(p.read_text()) for p in (d0 / f"{sysn}_asst").glob("*.json")}
        rule = FC["turn"][f"ours_{sysn}"]["rules"]["assistant"]
        pk = rule.get("p5key", "p5")
        tt = {}
        for k, d in A.items():
            h = dict(d["head"])
            h["p"] = h[pk]
            db = np.load(V5.ENERGY / f"asst_{k}.npy")
            db = db[np.clip(np.asarray(h["v"]), 0, len(db) - 1)]
            tt[k] = [x + d["chunk_ms"]["p50"] / 1000 for x, _ in V5.run_policy({"head": h}, db, rule, {})]
        S[f"ours_{sysn}_assistant"] = tt
    for name, T in S.items():
        row = []
        for w in (2.0, 3.0, 3.5, 4.0):
            FF = w
            r = my_asst(T, asess)
            row.append(f"{w:g}s acc {r['acc']} ff {r['ff']}")
        print(f"{name:24s} " + " | ".join(row))
    FF = 3.0


# MAIN
if __name__ == "__main__":
    STAGES = [k for k in ("asr", "longform", "dualrate", "vad", "vad_lastframe", "lid", "turn", "turn_window", "eer") if k in globals()]
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in STAGES:
        if what in (name, "all"):
            print(f"==== {name}")
            globals()[name]()
