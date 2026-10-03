"""Speech head without the ICSI test speakers (plans/audit/leakage_001.md finding 3, research/FIXALL.md step 1).

All 13 speakers of the ICSI test meetings (Bmr013, Bmr018, Bro021) appear in the ICSI train meetings of FIXALL's
`icsi600` window set. This retrains the step-1 speech head of each core with the FIXALL recipe, minus every ICSI
meeting that holds any of those speakers (`fixall.py vtrain --icsi-exclude-test-speakers`, cached features only),
selects it on the FIXALL held-out scopes, then scores it once on the ICSI / AMI test windows of FINAL_COMPARE
(final_compare.py vad, run into a scratch work dir) next to the shipped head. -> runs/fixall.json speech_clean.<core>.

Stages (each < 10 min; run through scripts/dev/gate.sh, one heavy job at a time):
  spk                           test speakers and the excluded meetings (ICSI annotations) -> speech_clean.<core>
  heldout --core C --tag T      new head HEADS/vad_<core>_<T>.pt vs the shipped one on fixall.vho_sets
  build   --core C --tag T      candidate served model = shipped afm with heads.speech replaced -> FX/<name>.afm
  score   --core C              test windows (ami_eval, icsi_eval), frames inside the audio only, window bootstrap
  ship    --core C --tag T      assets/speech_<core>_v2.pt (+ size, sha256) when the held-out ship rule passes
  latency --core C              engine ms per 160 ms chunk old vs new (final_compare cost outputs) + servedeq record

  PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/speech_clean.py <stage> --core 115m|0.6b [...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SSD = Path("/Volumes/ExternalSSD/nvidia-audio-models")
SC = SSD / "scratch" / "speech_clean"
TEST_ICSI = ("Bmr013", "Bmr018", "Bro021")
# the shipped speech heads (research/FIXALL.md step 1): fixall vtrain tags and the served models that carry them
SHIPPED = {"115m": ("mix26_none_sa_cal", SSD / "scratch/fixall/stage1_served_v4.afm"),
           "0p6b": ("mix_none_sa", SSD / "scratch/fixall/served_0p6b_v0.3.afm")}
CAND = {"115m": "speech115_clean", "0p6b": "speech6_clean"}
TEST_SETS = {"icsi_test": "icsi_eval", "ami_test": "ami_eval"}


@lru_cache(None)
def icsi_speakers() -> dict:
    """{meeting: sorted participant tags} from the transcriber segments of ICSI_core_NXT.zip."""
    from audioforge.datasets.icsi import DEFAULT_ROOT
    z = zipfile.ZipFile(DEFAULT_ROOT / "ICSI_core_NXT.zip")
    out: dict[str, set] = {}
    for n in z.namelist():
        if "/Segments/" in n and n.endswith(".segs.xml"):
            m = n.rsplit("/", 1)[-1].split(".")[0]
            out.setdefault(m, set()).update(re.findall(r'participant="([^"]+)"', z.read(n).decode("utf8", "ignore")))
    return {m: sorted(v) for m, v in out.items()}


def test_speakers() -> list[str]:
    sp = icsi_speakers()
    return sorted(set().union(*[sp[m] for m in TEST_ICSI]))


def excluded_meetings(meetings=None) -> list[str]:
    """ICSI meetings (default: the 12 cached train meetings of icsi600) that contain any ICSI test speaker."""
    from audioforge.datasets.icsi import DEFAULT_MEETINGS
    sp, ts = icsi_speakers(), set(test_speakers())
    return [m for m in (meetings or DEFAULT_MEETINGS["train"]) if ts & set(sp[m])]


def save(core, key, val):
    import fixall as F
    res = F.load_json("speech_clean", {}) or {}
    res.setdefault(core, {})[key] = val
    F.save("speech_clean", res)


def stage_spk(a):
    from audioforge.datasets.icsi import DEFAULT_MEETINGS, SPLITS
    import fixall as F
    sp, ts = icsi_speakers(), test_speakers()
    ex = excluded_meetings()
    ex_all = excluded_meetings(SPLITS["train"])
    rec = {"test_speakers": ts, "excluded_meetings": ex,
           "icsi600_train_meetings_kept": [m for m in DEFAULT_MEETINGS["train"] if m not in ex],
           "held_out_meetings_with_test_speakers": [m for m in F.ICSI_HOLD if set(ts) & set(sp[m])],
           "all_icsi_train_split_meetings_with_test_speakers": ex_all,
           "speakers_per_excluded_meeting": {m: sorted(set(ts) & set(sp[m])) for m in ex}}
    print(json.dumps(rec, indent=1))
    for core in ("115m", "0p6b"):
        for k in ("test_speakers", "excluded_meetings"):
            save(core, k, rec[k])
        save(core, "speakers", {k: v for k, v in rec.items() if k not in ("test_speakers", "excluded_meetings")})


def _load_head(tag):
    import torch
    import fixall as F
    ck = torch.load(F.HEADS / f"vad_{F.CORE}_{tag}.pt", map_location="cpu", weights_only=False)
    net = F._vnet(F.C.D, len(ck["blocks"]), ck.get("hidden", 64))
    net.load_state_dict(ck["state_dict"])
    return net, ck


def stage_heldout(a):
    """New head (--tag) vs the shipped head on the FIXALL held-out scopes (fixall.vho_score: AMI-held TS3011b / ES2015c,
    ICSI-held Bro026 / Bmr022, oto-held, quiet-oto-held, held-out room tone). Also checks that the shipped head file
    equals heads.speech of the shipped served model."""
    import torch
    import fixall as F
    from audioforge.train import load_model
    torch.set_num_threads(2)
    core = F.CORE
    stag, safm = SHIPPED[core]
    snet, sck = _load_head(stag)
    nnet, nck = _load_head(a.tag)
    assert sck["blocks"] == nck["blocks"]
    msd = load_model(str(safm), "cpu").state_dict()
    same = all(torch.equal(msd[f"heads.speech.{k}"], v) for k, v in sck["state_dict"].items() if k != "mix") and \
        torch.equal(msd["layer_mix.speech"], sck["state_dict"]["mix"])
    assert same, "shipped head file != served heads.speech"
    H = F.vho_sets(sck["blocks"])
    out = {}
    for k, net in (("shipped", snet), ("new", nnet)):
        net = net.to(a.device)
        out[k] = F.vho_score({s: F.vpredict(net, v[0], v[1], a.device) for s, v in H.items()}, H)
        F.log(k, json.dumps(out[k]))
    r = nck["res"]
    out["new_train"] = {"tag": a.tag, "args": r["args"], "calib_shift": r["calib_shift"], "best_sel_uncalibrated": r["ho"]["sel"],
                        "last_step": r["hist"][-1]["step"], "sec": r["sec"]}
    out["shipped_tag"] = stag
    out["shipped_head_equals_served"] = bool(same)
    n, s = out["new"], out["shipped"]
    out["ships"] = bool(n["sel"] >= s["sel"] and n["ami_ho"]["f1"] >= s["ami_ho"]["f1"] and n["icsi_ho"]["f1"] >= s["icsi_ho"]["f1"])
    out["caveat"] = ("AMI-held TS3011b: the shipped turn heads trained on 81 TS3011b clips (leakage_001 finding 11b); "
                     "ICSI-held Bro026 / Bmr022 contain ICSI test speakers (selection only, never trained on here)")
    F.log("ships", out["ships"])
    save(core, "heldout", out)
    save(core, "ships", out["ships"])


def stage_build(a):
    """Candidate served model in scratch: the shipped served afm with heads.speech = HEADS/vad_<core>_<tag>.pt."""
    import fixall as F
    out, changed = F.build_cand(CAND[F.CORE], a.tag, base_afm=SHIPPED[F.CORE][1])
    save(F.CORE, "candidate", {"afm": str(out), "changed": changed, "base": str(SHIPPED[F.CORE][1])})


def _metrics(p, t):
    from sklearn.metrics import roc_auc_score
    d = p > 0.5
    tp, fp, fn = float((d & t).sum()), float((d & ~t).sum()), float((~d & t).sum())
    th = np.quantile(p[~t], 1 - 0.075)
    return 2 * tp / max(2 * tp + fp + fn, 1), roc_auc_score(t, p), float((p[t] <= th).mean())


def stage_score(a):
    """The FINAL_COMPARE test windows (vad_layers ami_eval / icsi_eval, 64 x 20 s each) scored by final_compare.py vad
    into SC/fc_<core>_{shipped,new}/vad. Only label frames inside the audio (drop frame T-1 when T*0.08 > audio s,
    metrics_001 finding 1), for both heads. F1 at 0.5, ROC-AUC, miss at 7.5 % FA; 1000 window-bootstrap resamples
    (seed 0), paired new - shipped deltas."""
    import fixall as F
    import vad_layers as V
    core = F.CORE
    sysname = "core_115m" if core == "115m" else "core_0p6b"
    res = {}
    for key, sn in TEST_SETS.items():
        Z = {k: np.load(SC / f"fc_{core}_{k}" / "vad" / f"{sysname}_{sn}.npz") for k in ("shipped", "new")}
        y, lens = Z["shipped"]["y"], Z["shipped"]["lens"]
        assert np.array_equal(Z["new"]["y"], y) and np.array_equal(Z["new"]["lens"], lens)
        val = V.load_set(sn)
        assert len(val) == len(lens)
        off = np.r_[0, np.cumsum(lens)]
        rows, dropped = [], 0
        for w, v in enumerate(val):
            T = int(lens[w])
            n_in = T - 1 if T * F.FRAME > len(v["audio"]) / F.SR + 1e-9 else T
            dropped += T - n_in
            rows.append(np.arange(off[w], off[w] + n_in))
        keep = np.concatenate(rows)
        t = y > 0.5
        rng = np.random.default_rng(0)
        boots = [np.concatenate([rows[k] for k in rng.integers(0, len(rows), len(rows))]) for _ in range(1000)]
        o = {"n_windows": int(len(lens)), "n_frames_scored": int(len(keep)), "frames_dropped_past_audio": int(dropped),
             "speech_frac": round(float(t[keep].mean()), 4)}
        B = {}
        for k, z in Z.items():
            p = z["p"].astype(np.float64)
            f1, auc, miss = _metrics(p[keep], t[keep])
            B[k] = np.array([_metrics(p[ii], t[ii]) for ii in boots])
            ci = lambda j: [round(float(np.percentile(B[k][:, j], q)), 4) for q in (2.5, 97.5)]  # noqa: E731
            f1a, auca, missa = _metrics(p, t)
            o[k] = {"f1_at_0.5": round(f1, 4), "f1_ci95": ci(0), "auc": round(auc, 4), "auc_ci95": ci(1),
                    "miss_at_fpr_7.5_pct": round(100 * miss, 2), "miss_ci95_pct": [round(100 * x, 2) for x in ci(2)],
                    "all_frames_incl_past_audio": {"f1_at_0.5": round(f1a, 4), "auc": round(auca, 4),
                                                   "miss_at_fpr_7.5_pct": round(100 * missa, 2)}}
        dd = B["new"] - B["shipped"]
        o["new_minus_shipped"] = {
            nm: {"delta": round(float((o["new"][kk] - o["shipped"][kk])), 4),
                 "ci95": [round(float(sc * np.percentile(dd[:, j], q)), 4) for q in (2.5, 97.5)]}
            for j, (nm, kk, sc) in enumerate((("f1", "f1_at_0.5", 1), ("auc", "auc", 1), ("miss_pct", "miss_at_fpr_7.5_pct", 100)))}
        if key == "icsi_test":
            o["label"] = "speakers unseen in training (new head); shipped head trained on all 13 of these speakers"
        F.log(key, json.dumps({k: o[k] for k in ("new", "shipped", "new_minus_shipped")}))
        res[key] = o
    save(core, "test", res)


def stage_ship(a):
    import shutil
    import fixall as F
    from audioforge import hub
    rec = (F.load_json("speech_clean", {}) or {}).get(F.CORE, {})
    assert rec.get("ships"), "held-out ship rule failed: nothing to ship"
    dst = ROOT / "assets" / f"speech_{F.CORE}_v2.pt"
    shutil.copyfile(F.HEADS / f"vad_{F.CORE}_{a.tag}.pt", dst)
    save(F.CORE, "asset", {"path": str(dst.relative_to(ROOT)), "size": dst.stat().st_size, "sha256": hub.sha256_file(dst)})


def stage_latency(a):
    """Collect the back-to-back engine cost (final_compare.py cost --sub engine, SC/cost_<core>_{old,new}) and the
    fixall servedeq record of the candidate (turn_end events old vs new)."""
    import fixall as F
    core = F.CORE
    out = {}
    for k in ("old", "new"):
        f = SC / f"cost_{core}_{k}" / "cost" / core / "engine.json"
        out[k] = json.loads(f.read_text()).get("mps") if f.exists() else None
    if out["old"] and out["new"]:
        out["delta_p50_ms"] = round(out["new"]["chunk_ms_p50"] - out["old"]["chunk_ms_p50"], 2)
        out["delta_p95_ms"] = round(out["new"]["chunk_ms_p95"] - out["old"]["chunk_ms_p95"], 2)
        out["pass_le_1ms"] = bool(out["delta_p50_ms"] <= 1.0)
    eq = (F.load_json(f"servedeq_{core}", {}) or {}).get("speech_clean")
    if eq:
        out["turn_ends_equal_every_preset"] = all(v["turn_ends_equal"] for v in eq.values())
        out["servedeq"] = {p: {k: v[k] for k in ("turn_ends_equal", "client_vad_changed_frames", "n_frames")} for p, v in eq.items()}
    F.log(json.dumps(out))
    save(core, "latency", out)


STAGES = {"spk": stage_spk, "heldout": stage_heldout, "build": stage_build, "score": stage_score, "ship": stage_ship,
          "latency": stage_latency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--core", default="0.6b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    STAGES[a.stage](a)


if __name__ == "__main__":
    main()
