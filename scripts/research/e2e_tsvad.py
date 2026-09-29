"""research/IMPROVEMENTS.md section 1: the served TS-VAD turn path live, through the Pipecat adapter, on E2E_FINAL's clips.

System T = audioforge.serve --turn-input tsvad --diar-off --enroll explicit (the product has the user's voice print: a 5 s print
of the user's single-speaker speech from elsewhere in the same conversation / meeting, sent once at connect as
{"type": "enroll", "embedding": [...]}), turn policy hybrid_dyn at the TS-VAD point (serve.TSVAD_DYN). Everything
else is research/E2E_FINAL.md's Pipecat protocol (scripts/research/e2e_final.py: WAV transport at 1x, mock LLM/TTS,
the same pads and scoring); C and D are compared from runs/e2e_final.json's stored per-clip records (same clips, same
protocol, same scorer). Variant Ta = --enroll after_agent_arm (print = the first 5 s of speech after each agent_end).

The scratchpad that held E2E_FINAL's clips was lost in a reboot, so ``prepare`` rebuilds them from the stored clip
list: TurnBench / oto clips with e2e_final.two_party_clip on the same conversations (the start / duration must match
runs/e2e_final.json), the 5 AMI windows from the eot-bench v2 AMI dev windows with the same meeting / start (primary =
the window's speaker; agent_end = the last other-speaker speech end before the primary's onset, INTEGRATION section
8's stand-in).

    W=/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/e2e_tsvad.py prepare --work $W
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/e2e_tsvad.py prints --work $W
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/e2e_tsvad.py queue --work $W --budget 540  # repeat
    PYTHONPATH=. .venv/bin/python scripts/research/e2e_tsvad.py report --work $W
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import e2e_final as E2E  # noqa: E402

SR = 16000
WORK = Path("/Volumes/ExternalSSD/nvidia-audio-models/scratch/e2e_tsvad")
# E2E_FINAL's quiet check allows 150 % of one core for other processes; other agents keep this shared laptop above
# that for long stretches, so these runs allow 250 % and record the load and server RTF per session (a session with
# server RTF > 1 is flagged, as in E2E_FINAL)
E2E.MAX_OTHER_CPU = float(os.environ.get("E2E_MAX_OTHER_CPU", 250.0))
SYSTEMS = {"T": ("explicit", "hybrid_dyn"), "Ta": ("after_agent_arm", "hybrid_dyn"),  # system -> (enroll, policy)
           "Th": ("explicit", "hybrid"),  # post hoc: head OR 1000 ms timeout on P(target) (primary = column 0)
           # research/SINGLE_MODEL.md A1: the rule picked offline (--policy / --timeout-ms), served as audioforge-serve
           # --mode single (no diarizer loaded, the LID head); the stored print is sent at connect as for T
           "S": ("explicit", None)}
OUT_JSON = ROOT / "runs" / "e2e_tsvad.json"


def _runs_iv(act, fs=0.08):
    on = np.asarray(act) > 0.5
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            out.append((round(i * fs, 3), round(j * fs, 3)))
            i = j
        else:
            i += 1
    return out


def cmd_prepare(a):
    from audioforge.datasets import dyadic as D
    import eval_stage1 as ES
    import bench_dyadic_heads as BH  # noqa: F401 - e2e_final imports it lazily
    work = Path(a.work)
    out = work / "clips"
    out.mkdir(parents=True, exist_ok=True)
    stored = json.loads((ROOT / "runs" / "e2e_final.json").read_text())["clips"]
    refs = []
    # AMI: the eot-bench v2 windows with the same meeting and start
    base, ext, meta, ds, _ = ES.v2_data()
    for name, c in stored.items():
        if c["set"] != "ami":
            continue
        cand = [v for v in base if v["meeting"] == c["conversation"] and abs(float(v["start"]) - c["start"]) < 0.02]
        assert len(cand) == 1, (name, len(cand))
        v = cand[0]
        x = np.asarray(ds._clip(v["meeting"], c["start"], c["start"] + c["dur"]), np.float32)
        E2E._wav_write(out / f"{name}.mono.wav", x)
        on, en = int(v["onset_frame"]) * 0.08, int(v["turn_end_frame"]) * 0.08
        prim = _runs_iv(v["spk_act"])
        oth = np.asarray(v["spk_targets"])[:, 1:].max(1) if np.asarray(v["spk_targets"]).shape[1] > 1 else np.zeros(1)
        o_ends = [e for s, e in _runs_iv(oth) if e <= on + 1e-6]
        refs.append({"name": name, "set": "ami", "conversation": v["meeting"], "start": c["start"],
                     "dur": round(len(x) / SR, 3), "user_turns": [(round(on, 3), round(en, 3))], "scored": [True],
                     "user_intervals": prim, "agent_ends": o_ends[-1:], "speaker": int(v["speaker"]),
                     "text_mono": v.get("text"), "text_user": None, "has_text": False,
                     "first_onset_mono": None, "note": "rebuilt from the eot-bench v2 window (e2e_tsvad prepare)"})
        print(f"  {name}: onset {on:.2f} end {en:.2f} agent_end {o_ends[-1:]}", flush=True)
    # TurnBench / oto: the same conversations through e2e_final.two_party_clip
    for corpus, key in (("turnbench", "turnbench"), ("oto", "oto")):
        cids = [c["conversation"] for c in stored.values() if c["set"] == key]
        dsd = D.Dyadic(cids, corpus, verbose=False, **({"cache_dtype": "float16"} if corpus == "oto" else {}))
        for cid in cids:
            ref, mono, user = E2E.two_party_clip(dsd, cid, corpus, 35.0)
            st = stored[ref["name"]]
            assert abs(ref["start"] - st["start"]) < 0.01 and abs(ref["dur"] - st["dur"]) < 0.01, (ref["name"], ref["start"], st)
            E2E._wav_write(out / f"{ref['name']}.mono.wav", mono)
            E2E._wav_write(out / f"{ref['name']}.user.wav", user)
            refs.append(ref)
            print(f"  {ref['name']}: {ref['start']:.1f}+{ref['dur']:.1f}s", flush=True)
    (work / "clips.json").write_text(json.dumps(refs, indent=1))
    print(f"{len(refs)} clips -> {out}")


def _print_ivs(segs, excl, name, L=5.0):
    import tsvad as T
    return T.clip_from(segs, L, random.Random(f"e2e_{name}_{L}"), exclude=excl)


def cmd_prints(a):
    """Per clip, the user's voice print(s) from single-speaker speech outside [clip start - 2 s, clip end + 2 s]:
    the served block-4 speaker head over the clip audio alone (audioforge.tsvad_stream.voiceprint)."""
    import torch
    from audioforge.datasets import dyadic as D
    from audioforge.train import load_model
    from audioforge.tsvad_stream import voiceprint
    import tsvad as T
    import eval_stage1 as ES
    torch.set_num_threads(2)
    work = Path(a.work)
    refs = json.loads((work / "clips.json").read_text())
    out = work / "prints.json"
    have = json.loads(out.read_text()) if out.exists() else {}
    model = load_model(str(ROOT / "runs" / "stage1_served.afm"), "cpu").eval()
    lens = [float(x) for x in a.lens.split(",")]
    ami_ds = None
    dy = {}
    for r in refs:
        if r["name"] in have and all(str(L) in have[r["name"]] for L in lens):
            continue
        a0, b0 = r["start"] - 2.0, r["start"] + r["dur"] + 2.0
        rec = have.get(r["name"], {})
        if r["set"] == "ami":
            if ami_ds is None:
                _, _, _, ami_ds, _ = ES.v2_data()
            m = r["conversation"]
            spk = ami_ds.speaker_ids[r["speaker"] - getattr(ami_ds, "speaker_offset", 0)]
            segs = T.single_segments(ami_ds, m).get(spk, [])
            get = lambda ivs: T.clip_audio(ami_ds, m, ivs)  # noqa: E731
        else:
            cid = r["conversation"]
            if r["set"] not in dy:
                cids = [x["conversation"] for x in refs if x["set"] == r["set"]]
                dy[r["set"]] = D.Dyadic(cids, r["set"], verbose=False,
                                       **({"cache_dtype": "float16"} if r["set"] == "oto" else {}))
            ds = dy[r["set"]]
            h = r["human_channel"]
            own = ds.acts[cid][f"{cid}:{h}"]
            oth = sorted(ds.acts[cid][f"{cid}:{1 - h}"])
            segs = []
            for s, e in own:  # the human's speech with the other party silent (0.2 s guard), >= 1 s
                cut = [(s, e)]
                for os_, oe in oth:
                    nxt = []
                    for x, y in cut:
                        if oe + 0.2 <= x or os_ - 0.2 >= y:
                            nxt.append((x, y))
                        else:
                            if os_ - 0.2 > x:
                                nxt.append((x, os_ - 0.2))
                            if oe + 0.2 < y:
                                nxt.append((oe + 0.2, y))
                    cut = nxt
                segs += [(x, y) for x, y in cut if y - x >= 1.0]
            if r["set"] == "turnbench":
                chan = np.asarray(ds.channels16k(cid)[h], np.float32)
            else:
                xx, sr, _ = D.read_stereo(r["set"], ds.root, cid)
                ch = D.resample16k(xx, sr)
                chan = np.asarray(ch.T if ch.shape[0] != 2 else ch, np.float32)[h]
            get = lambda ivs: np.concatenate([chan[int(x * SR): int(y * SR)] for x, y in ivs])  # noqa: E731
        for L in lens:
            ivs = _print_ivs(segs, (a0, b0), r["name"], L)
            if ivs is None:
                rec[str(L)] = None
                continue
            rec[str(L)] = {"ivs": [[round(x, 3), round(y, 3)] for x, y in ivs],
                           "embedding": [round(float(v), 6) for v in voiceprint(model, get(ivs))]}
        have[r["name"]] = rec
        out.write_text(json.dumps(have))
        print(f"  {r['name']}: prints {[L for L in lens if rec.get(str(L))]}", flush=True)
    print(f"prints -> {out}")


def cmd_run(a):
    """One system x condition over clips (Pipecat); T sends the stored print at connect."""
    import asyncio
    import warnings
    warnings.filterwarnings("ignore")
    work = Path(a.work)
    refs = {r["name"]: r for r in json.loads((work / "clips.json").read_text())}
    prints = json.loads((work / "prints.json").read_text())
    enroll, policy = SYSTEMS[a.system]
    policy = policy or a.policy
    out = Path(a.out)
    done = {json.loads(line)["clip"] for line in out.read_text().splitlines()} if out.exists() else set()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    import audioforge.integrations.pipecat as PA
    t_budget = time.time() + a.budget
    for name in a.clips:
        if name in done:
            continue
        if time.time() > t_budget:
            print("budget reached", flush=True)
            break
        ref = refs[name]
        x = E2E._wav_read(work / "clips" / f"{name}.{a.cond}.wav")
        emb = (prints.get(name) or {}).get(str(a.print_s))
        orig = PA.AudioforgeSTTService.__init__

        def init(self, *args, **kw):  # T: queue the stored print; it is sent right after the config at connect
            orig(self, *args, **kw)
            self._timeout_ms = int(a.timeout_ms)  # the session's timeout_ms (1000 = E2E_FINAL's value)
            if enroll == "explicit" and emb is not None:
                self._pending_controls.append({"type": "enroll", "embedding": emb["embedding"]})
        PA.AudioforgeSTTService.__init__ = init
        load0 = E2E.wait_guard()
        sampler = E2E.UsageSampler({"driver": os.getpid(), "server": a.server_pid})
        sampler.start()
        t_start = time.time()
        try:
            raw = asyncio.run(E2E.pipecat_run("T", x, url=a.url, pad_s=E2E.PADS[ref["set"]], policy=policy,
                                              enroll=None if enroll == "explicit" else enroll,
                                              agent_ends=ref.get("agent_ends", [])))
        finally:
            PA.AudioforgeSTTService.__init__ = orig
        use = sampler.stop()
        rtf = (raw.get("server_stats") or {}).get("rtf")
        rec = {"clip": name, "set": ref["set"], "cond": a.cond, "framework": "pipecat", "system": a.system,
               "policy": policy, "timeout_ms": a.timeout_ms, "enroll": enroll, "print_s": a.print_s,
               "has_print": emb is not None,
               "pad_s": E2E.PADS[ref["set"]], "wall_start": t_start, "load1_start": round(load0, 2),
               "usage": use, "flag_rerun": bool(rtf is not None and rtf > 1.0), "raw": raw}
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[pipecat {a.system} {a.cond}] {name}: responses {len(raw['responses'])} rtf {rtf} "
              f"load {load0:.1f}", flush=True)


def cmd_queue(a):
    """Resumable: runs (set, cond) groups of --system until --budget seconds are used; one server per group."""
    work = Path(a.work)
    runs = work / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    refs = json.loads((work / "clips.json").read_text())
    enroll, policy = SYSTEMS[a.system]
    t0 = time.time()
    E2E.WORK = work  # the server's TMPDIR
    for st in ("ami", "turnbench", "oto"):
        for cond in (("mono",) if st == "ami" else ("mono", "user")):
            clips = sorted(r["name"] for r in refs if r["set"] == st)
            tag = f"pipecat_{a.system}_{st}_{cond}"
            out = runs / f"{tag}.jsonl"
            have = {json.loads(line)["clip"] for line in out.read_text().splitlines()} if out.exists() else set()
            todo = [c for c in clips if c not in have]
            left = a.budget - (time.time() - t0)
            if not todo or left < 90:
                continue
            a.port += 1
            srv = start_server(a.port, enroll, runs / f"server_{tag}.log", a.print_s, single=a.system == "S")
            env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--work", str(work), "--system",
                   a.system, "--cond", cond, "--out", str(out), "--url", f"ws://127.0.0.1:{a.port}",
                   "--server-pid", str(srv.pid), "--budget", str(int(left - 60)), "--print-s", str(a.print_s),
                   "--policy", a.policy, "--timeout-ms", str(a.timeout_ms), "--clips", *todo]
            print(f"=== {tag}: {len(todo)} clips {time.strftime('%H:%M:%S')} load {os.getloadavg()[0]:.2f}", flush=True)
            try:
                with (runs / f"driver_{tag}.log").open("a") as f:
                    subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=left)
            except subprocess.TimeoutExpired:
                print("  group timed out (resumable)", flush=True)
            finally:
                srv.terminate()
                try:
                    srv.wait(20)
                except subprocess.TimeoutExpired:
                    srv.kill()
    n = sum(len((runs / f).read_text().splitlines()) for f in os.listdir(runs) if f.endswith(".jsonl")
            and f.startswith(f"pipecat_{a.system}_"))
    print(f"QUEUE: {n} / 69 sessions of {a.system} done", flush=True)


def start_server(port: int, enroll: str, log: Path, print_s: float, single: bool = False):
    """single: audioforge-serve --mode single's flags (no --diar: the diarizer is never loaded; --lid head)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "TMPDIR": str(WORK / "tmp")}
    (WORK / "tmp").mkdir(parents=True, exist_ok=True)
    diar = (["--lid", "head", "--dyn-wait-ms", os.environ.get("E2E_DYN_WAIT_MS", "2000,960")] if single
            else ["--diar", str(ROOT / "runs/nemo_sortformer_v2.afm")])
    cmd = [sys.executable, str(ROOT / "scripts" / "research" / "e2e_final.py"), "serve", "--",
           "--asr", str(ROOT / "runs/stage1_served.afm"), *diar,
           "--port", str(port), "--threads", "2", "--debug-fields", "--enroll", enroll, "--turn-input", "tsvad",
           "--tsvad-print-s", str(print_s), "--diar-off", "--silero", str(ROOT / "data/silero/silero_vad_v5.onnx")]
    f = log.open("w")
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(1)
        txt = log.read_text()
        if "listening" in txt:
            return p
        if "Traceback" in txt or p.poll() is not None:
            break
    p.kill()
    raise RuntimeError(f"server failed: {log.read_text()[-2000:]}")


def cmd_report(a):
    """T (and Ta) vs the stored C / D Pipecat records: pooled metrics per (set, cond) and paired clip bootstraps."""
    from audioforge.e2e_metrics import pooled
    work = Path(a.work)
    refs = {r["name"]: r for r in json.loads((work / "clips.json").read_text())}
    ef = json.loads((ROOT / "runs" / "e2e_final.json").read_text())
    recs = []
    for p in sorted((work / "runs").glob("pipecat_*.jsonl")):
        recs += [json.loads(line) for line in p.read_text().splitlines()]
    per = {}
    for r in recs:
        s = E2E.score_record(r, refs[r["clip"]])
        per[f"{r['set']}|{r['cond']}|pipecat|{r['system']}|{r['clip']}"] = s
    for k, v in ef["per_clip"].items():
        st, cond, fw, sy, clip = k.split("|")
        if fw == "pipecat" and sy in ("C", "D"):
            per[k] = v
    groups = [("ami", "mono"), ("turnbench", "mono"), ("turnbench", "user"), ("oto", "mono"), ("oto", "user")]
    systems = sorted({k.split("|")[3] for k in per})
    table, paired = {}, {}
    rng = np.random.default_rng(0)
    for st, cond in groups:
        clips = sorted({k.split("|")[4] for k in per if k.startswith(f"{st}|{cond}|")})
        for sy in systems:
            sc = [per[f"{st}|{cond}|pipecat|{sy}|{c}"] for c in clips if f"{st}|{cond}|pipecat|{sy}|{c}" in per]
            if sc:
                table[f"{st}|{cond}|{sy}"] = {**pooled(sc), "n_clips": len(sc)}
        for x, y in [("T", "C"), ("T", "D"), ("Ta", "C"), ("Ta", "D"), ("Ta", "T"), ("Th", "C"), ("Th", "D"), ("Th", "T")]:
            cc = [c for c in clips if f"{st}|{cond}|pipecat|{x}|{c}" in per and f"{st}|{cond}|pipecat|{y}|{c}" in per]
            if not cc:
                continue
            A = [per[f"{st}|{cond}|pipecat|{x}|{c}"] for c in cc]
            B = [per[f"{st}|{cond}|pipecat|{y}|{c}"] for c in cc]
            d = {}
            for stat in ("missed_3s", "missed_6s", "cut_ins_per_clip", "dead_air_ms_median"):
                def val(sc):
                    m = pooled(sc)
                    return m.get(stat) if stat != "cut_ins_per_clip" else m["cut_ins"] / len(sc)
                a_, b_ = val(A), val(B)
                bs = []
                for _ in range(2000):
                    idx = rng.integers(0, len(cc), len(cc))
                    va, vb = val([A[i] for i in idx]), val([B[i] for i in idx])
                    if va is not None and vb is not None:
                        bs.append(va - vb)
                d[stat] = {"a": a_, "b": b_, "delta": None if a_ is None or b_ is None else round(a_ - b_, 4),
                           "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
                           if bs else None, "n_clips": len(cc)}
            paired[f"{st}|{cond}|{x}-{y}"] = d
    # pooled over all 69 sessions (the pre-registered bar: misses at 3 s vs C, cut-ins vs D)
    for x, y in [("T", "C"), ("T", "D"), ("Ta", "C"), ("Ta", "D"), ("Th", "C"), ("Th", "D")]:
        keys = [(st, cond, c) for st, cond in groups for c in sorted({k.split("|")[4] for k in per
                if k.startswith(f"{st}|{cond}|")}) if f"{st}|{cond}|pipecat|{x}|{c}" in per
                and f"{st}|{cond}|pipecat|{y}|{c}" in per]
        if not keys:
            continue
        A = [per[f"{s}|{c_}|pipecat|{x}|{c}"] for s, c_, c in keys]
        B = [per[f"{s}|{c_}|pipecat|{y}|{c}"] for s, c_, c in keys]
        d = {}
        for stat in ("missed_3s", "missed_6s", "cut_ins"):
            def val(sc):
                m = pooled(sc)
                return m[stat] / len(sc) if stat == "cut_ins" else m[stat]
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, len(keys), len(keys))
                bs.append(val([A[i] for i in idx]) - val([B[i] for i in idx]))
            d[stat] = {"a": round(val(A), 4), "b": round(val(B), 4), "delta": round(val(A) - val(B), 4),
                       "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                       "n_sessions": len(keys)}
        paired[f"all|{x}-{y}"] = d
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "table": table, "paired": paired,
           "n_records": {sy: sum(1 for k in per if k.split("|")[3] == sy) for sy in systems}}
    OUT_JSON.write_text(json.dumps(res, indent=1))
    for k, v in table.items():
        print(k, {q: v.get(q) for q in ("n_clips", "dead_air_ms_median", "missed_3s", "missed_6s", "cut_ins_per_min")})
    for k, v in paired.items():
        print(k, {q: (w["delta"], w["ci"]) for q, w in v.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for n in ("prepare", "prints", "queue", "report", "run"):
        s = sub.add_parser(n)
        s.add_argument("--work", default=str(WORK))
        if n == "prints":
            s.add_argument("--lens", default="1.5,3.0,5.0,10.0")
        if n in ("queue", "run"):
            s.add_argument("--system", default="T", choices=list(SYSTEMS))
            s.add_argument("--budget", type=float, default=540)
            s.add_argument("--print-s", type=float, default=5.0)
            s.add_argument("--policy", default="timeout", help="system S: the session's turn_policy")
            s.add_argument("--timeout-ms", type=int, default=1000, help="the session's timeout_ms")
        if n == "queue":
            s.add_argument("--port", type=int, default=8960)
        if n == "run":
            s.add_argument("--cond", default="mono")
            s.add_argument("--out", required=True)
            s.add_argument("--url", required=True)
            s.add_argument("--server-pid", type=int)
            s.add_argument("--clips", nargs="+", required=True)
    a = p.parse_args()
    {"prepare": cmd_prepare, "prints": cmd_prints, "queue": cmd_queue, "report": cmd_report, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
