"""Hybrid front end, live part (research/HYBRID_ASR.md): Pipecat / LiveKit sessions through the committed adapters
against audioforge.serve with a second transcript source, on the clips of the final end-to-end comparison.

Clips: scratchpad/e2e_final/clips.json (scripts/research/e2e_final.py `prepare`): the 5 AMI windows of research/INTEGRATION.md
sections 4 / 8 and the 16 TurnBench two-party clips, mono condition (the 16 oto clips have no reference text).
Server: the e2e product default (system C: runs/stage1_served.afm + Sortformer v2 0.32 s, turn policy timeout 1000,
--enroll dominant, 2 threads, --debug-fields) plus
  H  --final-asr tdt_v3 (Parakeet-TDT 0.6B v3 in a child process, CPU, 2 threads of its own)
  L  --asr-lookahead 13 (second text-only pass of the same model at [70, 13])
  HM --final-asr tdt_v3 --final-asr-device mps (the offline model on the Apple GPU; a proxy for a CUDA GPU)
The adapters run with final_source="offline": the agent answers the second source's transcript, and the turn is
committed only when that final has arrived (Pipecat: AudioforgeTurnAnalyzer waits for it; LiveKit: FINAL +
END_OF_SPEECH are emitted on it). So the measured dead air includes the extra wait for the better transcript.
Control: the e2e C runs on the same clips (scratchpad/e2e_final/runs/<fw>_C_<set>_mono.jsonl): stream-final WER,
dead air, and the server's turn_end decision times (which must be identical: streaming decisions unchanged).

Reuses scripts/research/e2e_final.py (pipecat_run / livekit_run / score_record / guard / usage sampler) without changing it.

    PYTHONPATH=. .venv/bin/python scripts/research/hybrid_live.py queue --systems H L --frameworks pipecat livekit
    PYTHONPATH=. .venv/bin/python scripts/research/hybrid_live.py report          # -> runs/hybrid_asr.json["live"]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import e2e_final as E

SCRATCH = E.SCRATCH
E2E_WORK = SCRATCH / "e2e_final"
WORK = SCRATCH / "hybrid_live"
OUT = ROOT / "runs" / "hybrid_asr.json"
SYSTEMS = {"H": ["--final-asr", "tdt_v3", "--final-asr-worker", "process", "--final-asr-threads", "2"],
           "L": ["--asr-lookahead", "13"],
           "HM": ["--final-asr", "tdt_v3", "--final-asr-worker", "process", "--final-asr-device", "mps"]}
CAPTURE: list[dict] = []  # every server final / turn_end seen by the adapter, with its arrival time


def patch_adapters(final_source: str = "offline"):
    import audioforge.integrations.livekit as LA
    import audioforge.integrations.pipecat as PA

    class STT(PA.AudioforgeSTTService):
        def __init__(self, **kw):
            kw.setdefault("final_source", final_source)
            super().__init__(**kw)

        async def handle_server_message(self, msg):
            if msg.get("type") in ("final", "turn_end"):
                CAPTURE.append({"perf": time.perf_counter(), "wall": time.time(), **msg})
            await super().handle_server_message(msg)
    PA.AudioforgeSTTService = STT

    class FE(LA.AudioforgeFrontend):
        def __init__(self, url, **kw):
            kw.setdefault("final_source", final_source)
            super().__init__(url, **kw)
    LA.AudioforgeFrontend = FE
    orig = LA._SpeechMapper.on_message

    def on_message(self, msg, *a, **kw):
        if msg.get("type") in ("final", "turn_end"):
            CAPTURE.append({"perf": time.perf_counter(), "wall": time.time(), **msg})
        return orig(self, msg, *a, **kw)
    LA._SpeechMapper.on_message = on_message


def cmd_run(a):
    warnings.filterwarnings("ignore")
    refs = E.load_clips(E2E_WORK)
    patch_adapters(a.final_source)
    out = Path(a.out)
    done = {json.loads(x)["clip"] for x in out.read_text().splitlines()} if out.exists() else set()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    for name in a.clips:
        if name in done:
            continue
        ref = refs[name]
        x = E._wav_read(E2E_WORK / "clips" / f"{name}.mono.wav")
        load0 = E.wait_guard(poll=60.0, max_wait_s=45 * 60)
        pids = {"driver": os.getpid(), "server": a.server_pid}
        try:
            import psutil
            kids = psutil.Process(a.server_pid).children()
            if kids:
                pids["final_asr_worker"] = kids[0].pid
        except Exception as e:  # noqa: BLE001 - the worker's usage is optional
            print(f"no final-ASR worker pid: {e}", flush=True)
        sampler = E.UsageSampler(pids)
        sampler.start()
        CAPTURE.clear()
        t_start = time.time()
        fn = E.pipecat_run if a.framework == "pipecat" else E.livekit_run
        raw = asyncio.run(fn("C", x, url=a.url, pad_s=E.PADS[ref["set"]], policy="timeout", enroll=None,
                             agent_ends=[]))
        use = sampler.stop()
        rec = {"clip": name, "set": ref["set"], "cond": "mono", "framework": a.framework, "system": a.system,
               "policy": "timeout", "final_source": a.final_source, "wall_start": t_start,
               "load1_start": round(load0, 2), "load1_end": round(os.getloadavg()[0], 2), "usage": use,
               "capture": list(CAPTURE), "raw": raw}
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        st = raw.get("server_stats") or {}
        print(f"[{a.framework} {a.system}] {name}: responses {len(raw['responses'])} rtf {st.get('rtf')} "
              f"final_latency {st.get('final_latency_ms')}", flush=True)


def start_server(system: str, port: int, log: Path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "TMPDIR": str(WORK / "tmp")}
    (WORK / "tmp").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts/research/e2e_final.py"), "serve", "--", "--asr", str(ROOT / "runs/stage1_served.afm"),
           "--diar", str(ROOT / "runs/nemo_sortformer_v2.afm"), "--port", str(port), "--threads", "2",
           "--debug-fields", "--enroll", "dominant", *SYSTEMS[system]]
    f = log.open("w")
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(300):
        time.sleep(1)
        txt = log.read_text()
        if "listening" in txt:
            return p
        if "Traceback" in txt or p.poll() is not None:
            break
    p.kill()
    raise RuntimeError(f"server failed: {log.read_text()[-2000:]}")


def cmd_queue(a):
    WORK.mkdir(parents=True, exist_ok=True)
    runs = WORK / "runs"
    runs.mkdir(exist_ok=True)
    refs = E.load_clips(E2E_WORK)
    port = a.port
    for st in a.sets:
        clips = sorted(n for n, r in refs.items() if r["set"] == st and r.get("has_text"))
        if a.max_clips:
            clips = clips[: a.max_clips]
        for system in a.systems:
            for fw in a.frameworks:
                tag = f"{fw}_{system}_{st}_mono"
                out = runs / f"{tag}.jsonl"
                have = {json.loads(x)["clip"] for x in out.read_text().splitlines()} if out.exists() else set()
                todo = [c for c in clips if c not in have]
                if not todo:
                    continue
                E.wait_guard(poll=60.0, max_wait_s=45 * 60)
                port += 1
                srv = start_server(system, port, runs / f"server_{tag}.log")
                env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
                       "TMPDIR": str(WORK / "tmp"), "TOKENIZERS_PARALLELISM": "false"}
                cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--framework", fw, "--system", system,
                       "--out", str(out), "--url", f"ws://127.0.0.1:{port}", "--server-pid", str(srv.pid),
                       "--clips", *todo]
                print(f"=== {tag}: {len(todo)} clips {time.strftime('%H:%M:%S')} load {os.getloadavg()[0]:.2f}",
                      flush=True)
                try:
                    with (runs / f"driver_{tag}.log").open("a") as f:
                        subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3 * 3600,
                                       check=False)
                finally:
                    srv.terminate()
                    try:
                        srv.wait(30)
                    except subprocess.TimeoutExpired:
                        srv.kill()
                time.sleep(3)
    print("QUEUE DONE", flush=True)


# --------------------------------------------------------------------------- report
def _wer(ref: str, hyp: str) -> tuple[int, int]:
    from audioforge.metrics import edit_distance
    from audioforge.teachers import normalize_text
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return edit_distance(r, h), len(r)


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else None


def summarize(recs: list[dict], ctrl: dict, refs: dict) -> dict:
    """One (framework, system, set) group: WER by source, offline-final latency after turn_end, dead air vs control,
    decisions vs control, server cost."""
    werr = {"delivered": [0, 0], "stream": [0, 0], "offline": [0, 0], "control_stream": [0, 0]}
    lat_srv, lat_client, dead, dead_c, dec_same, dec_n, first_text, pair_a = [], [], [], [], 0, 0, [], []
    rtf, bl, rss, wrss, chunk50, chunk95, lag95, rtf_c, chunk95_c, lag95_c = [], [], [], [], [], [], [], [], [], []
    for r in recs:
        ref = refs[r["clip"]]
        s = E.score_record(r, ref)
        text = ref.get("text_mono")
        cap = r["capture"]
        fins = [c for c in cap if c["type"] == "final"]
        stream_txt = " ".join(c["text"] for c in fins if c.get("source") == "stream")
        off_txt = " ".join(c["text"] for c in fins if c.get("source") not in (None, "stream"))
        if text:
            for k, h in (("stream", stream_txt), ("offline", off_txt)):
                e, n = _wer(text, h)
                werr[k][0] += e
                werr[k][1] += n
            werr["delivered"][0] += s.get("wer_err", 0)
            werr["delivered"][1] += s.get("wer_n", 0)
        tes = {c["t"]: c for c in cap if c["type"] == "turn_end"}
        for c in fins:
            if c.get("source") not in (None, "stream"):
                lat_srv.append(c.get("latency_ms", 0.0))
                te = tes.get(c["t"])
                if te is not None:
                    lat_client.append((c["perf"] - te["perf"]) * 1000)
        st = r["raw"].get("server_stats") or {}
        rtf.append(st.get("rtf"))
        bl.append(st.get("backlog_ms_max"))
        rss.append((r.get("usage", {}).get("server") or {}).get("rss_mb_peak"))
        wrss.append(st.get("final_asr_rss_mb") or (r.get("usage", {}).get("final_asr_worker") or {}).get("rss_mb_peak"))
        chunk50.append(st.get("chunk_ms_p50"))
        chunk95.append(st.get("chunk_ms_p95"))
        lag95.append(st.get("send_lag_ms_p95"))
        if s.get("first_text_ms_after_onset") is not None:
            first_text.append(s["first_text_ms_after_onset"])
        c = ctrl.get(r["clip"])
        if c is not None:
            sc = E.score_record(c, ref)
            if text and "finals" in c["raw"]:
                e, n = _wer(text, " ".join(f["text"] for f in c["raw"]["finals"]))
                werr["control_stream"][0] += e
                werr["control_stream"][1] += n
            a_t = [x["t"] for x in r["raw"].get("server_turn_ends", [])]
            b_t = [x["t"] for x in c["raw"].get("server_turn_ends", [])]
            dec_n += 1
            dec_same += a_t == b_t
            cs = c["raw"].get("server_stats") or {}
            rtf_c.append(cs.get("rtf"))
            chunk95_c.append(cs.get("chunk_ms_p95"))
            lag95_c.append(cs.get("send_lag_ms_p95"))
            dead_c.append(sc)
            pair_a.append(s)
        dead.append(s)
    from audioforge.e2e_metrics import paired_bootstrap, pooled
    pm, pc = pooled(dead), (pooled(dead_c) if dead_c else None)
    paired_da = paired_bootstrap(pair_a, dead_c, "dead_air_ms_median") if dead_c else None

    def w(k):
        e, n = werr[k]
        return round(e / n, 4) if n else None

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.median(xs)), 3) if xs else None

    def mx(xs):
        xs = [x for x in xs if x is not None]
        return round(float(max(xs)), 1) if xs else None
    return {"n_clips": len(recs), "wer": {k: w(k) for k in werr},
            "offline_final_latency_ms_server": {"p50": _pct(lat_srv, 50), "p95": _pct(lat_srv, 95), "n": len(lat_srv)},
            "offline_final_after_turn_end_ms_client": {"p50": _pct(lat_client, 50), "p95": _pct(lat_client, 95)},
            "dead_air": {k: pm.get(k) for k in pm if "dead" in k or "missed" in k or "cut" in k},
            "dead_air_control": {k: pc.get(k) for k in pc if "dead" in k or "missed" in k or "cut" in k} if pc else None,
            "dead_air_median_minus_control_ms": paired_da,
            "decisions_identical_to_control": f"{dec_same}/{dec_n}",
            "server": {"rtf_median": med(rtf), "rtf_max": mx(rtf), "backlog_ms_max": mx(bl),
                       "server_rss_mb_peak": mx(rss), "final_asr_worker_rss_mb": mx(wrss),
                       "chunk_ms_p50_median": med(chunk50), "chunk_ms_p95_median": med(chunk95),
                       "send_lag_ms_p95_median": med(lag95)},
            "control_server": {"rtf_median": med(rtf_c), "chunk_ms_p95_median": med(chunk95_c),
                               "send_lag_ms_p95_median": med(lag95_c)},
            "first_text_ms_median": med(first_text)}


def cmd_report(a):
    refs = E.load_clips(E2E_WORK)
    ctrl_all = {}
    for r in E.load_records(E2E_WORK):
        if r["system"] == "C" and r["cond"] == "mono":
            ctrl_all[(r["framework"], r["clip"])] = r
    groups = {}
    for p in sorted((WORK / "runs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            groups.setdefault((r["framework"], r["system"], r["set"]), []).append(r)
    res = {"protocol": __doc__.split("Reuses")[0].strip(), "groups": {}}
    for (fw, sy, st), recs in sorted(groups.items()):
        ctrl = {c: v for (f, c), v in ctrl_all.items() if f == fw}
        res["groups"][f"{fw}/{sy}/{st}"] = summarize(recs, ctrl, refs)
        print(fw, sy, st, json.dumps(res["groups"][f"{fw}/{sy}/{st}"]), flush=True)
    full = json.loads(OUT.read_text()) if OUT.exists() else {}
    full["live"] = res
    OUT.write_text(json.dumps(full, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--framework", choices=["pipecat", "livekit"], required=True)
    r.add_argument("--system", choices=list(SYSTEMS), required=True)
    r.add_argument("--clips", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--server-pid", type=int, required=True)
    r.add_argument("--final-source", default="offline", choices=["offline", "stream"])
    q = sub.add_parser("queue")
    q.add_argument("--systems", nargs="+", default=["H", "L"])
    q.add_argument("--frameworks", nargs="+", default=["pipecat", "livekit"])
    q.add_argument("--sets", nargs="+", default=["ami", "turnbench"])
    q.add_argument("--port", type=int, default=9300)
    q.add_argument("--max-clips", type=int)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    {"run": cmd_run, "queue": cmd_queue, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
