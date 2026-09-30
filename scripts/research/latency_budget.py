"""research/LATENCY_BUDGET.md: every delay in single mode's live path other than the turn rule's own silence wait.

Stages (each call < 10 min; the torch ones through scripts/dev/gate.sh; one at a time):

  ws        start ``audioforge-serve --mode single`` (child process, 2 threads) from ``--code`` (a checkout: the
            working tree, or a worktree of the commit before the fixes), stream the bundled clip
            (examples/audio/two_party_call_16s.wav + its stored voice print) ``--reps`` times at 1x in 20 ms blocks
            over a real WebSocket, and time every message against the moment its audio was sent:
              * frame v: receive - send of the block that completed the frame's chunk (server compute + socket), and
                receive - send of the frame's label time (v + 1) * 80 ms
              * turn_end: its decision time t (audio clock) vs the user's labelled end (10.8 s), and receive - send of t
              * final: receive - receive of its turn_end; partial: receive - send of the chunk that produced it
              * the server's stats (per-chunk compute, send lag)
  pipecat   the same server and clip through Pipecat (examples/pipecat_local_demo.run_pipeline: WAV transport at 1x,
            AudioforgeSTTService + VAD + turn analyzer, policy hybrid_dyn, a mock LLM): adapter time = mock LLM context
            - server turn_end arrival, and the whole response - send of t (the 76 ms ``delivery_offset_s`` of
            runs/single_model.json)
  wer       LibriSpeech-200 (runs/hybrid_asr.json protocol, final_asr's prepared audio) through the streaming
            session (``StreamingSession.feed`` in 20 ms blocks) at att_context [70, 1] and [70, 0]; WER with
            normalize_text, and the ASR encoder + decoder time per chunk
  compute   the full single-mode session (``audioforge.serve.Session``: encoder, RNNT, VAD, TS-VAD, turn pass 2, LID,
            Silero) on the clip in 20 ms blocks at --asr-chunk-ms 160 and 80: time per chunk and RTF, 2 threads; and
            the turn_end / final times of each setting
  report    -> runs/latency_budget.json (merged)

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/latency_budget.py ws --tag before --code <worktree>
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/latency_budget.py ws --tag after
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/latency_budget.py pipecat --tag after
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/latency_budget.py wer
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/latency_budget.py compute
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

SR, BLOCK_S, FRAME_S = 16000, 0.02, 0.08
CLIP = ROOT / "examples" / "audio" / "two_party_call_16s.wav"
PRINT = ROOT / "examples" / "audio" / "two_party_call_16s.voiceprint.json"
REF = json.loads((ROOT / "examples" / "audio" / "two_party_call_16s.json").read_text())
OUT = ROOT / "runs" / "latency_budget.json"
MODEL = ROOT / "runs" / "stage1_served_v2.afm"  # what audioforge-download ships (hub.SERVED); heads from assets/
PY = str(ROOT / ".venv" / "bin" / "python")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else None


def summ(xs):
    xs = [float(x) for x in xs if x is not None]
    return {"n": len(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "mean": round(float(np.mean(xs)), 1) if xs else None,
            "min": pct(xs, 0), "max": pct(xs, 100)}


def save(key: str, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1))


def clip_audio() -> np.ndarray:
    from audioforge.data import load_wav
    return load_wav(str(CLIP), SR).astype(np.float32)


# --------------------------------------------------------------------------- server child
def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    """``audioforge-serve --mode single`` from checkout ``code`` (its audioforge package), models from this repo."""

    def __init__(self, code: Path, extra: list[str] | None = None):
        self.port = free_port()
        env = {**os.environ, "PYTHONPATH": str(code), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        argv = [PY, "-c", "from audioforge.launch import serve_main; serve_main()", "--port", str(self.port),
                "--asr", str(MODEL), "--silero", str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "--threads", "2"] + (extra or [])
        self.log = open(ROOT / "logs" / f"latency_budget_server_{self.port}.log", "w")
        self.p = subprocess.Popen(argv, env=env, cwd=str(code), stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 180:
            if self.p.poll() is not None:
                raise RuntimeError(f"server exited ({self.p.returncode}); see {self.log.name}")
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.5).close()
                return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError("server did not start")

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()


# --------------------------------------------------------------------------- ws
async def ws_once(url: str, audio: np.ndarray, emb, pad_s: float = 1.5) -> dict:
    from websockets.asyncio.client import connect
    x = np.concatenate([audio, np.zeros(int(pad_s * SR), np.float32)])
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    n = int(SR * BLOCK_S)
    sent: list[tuple[int, float]] = []  # (cumulative samples, perf after send)
    recv: list[tuple[float, dict]] = []
    async with connect(url, max_size=2 ** 22, ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "config", "turn_policy": "hybrid_dyn", "sample_rate": SR}))
        await ws.send(json.dumps({"type": "enroll", "embedding": emb}))

        async def sender():
            t0 = time.perf_counter()
            for i, k in enumerate(range(0, len(pcm), n)):
                due = t0 + (i + 1) * BLOCK_S  # a microphone delivers block i when its last sample has been captured
                await asyncio.sleep(max(0.0, due - time.perf_counter()))
                await ws.send(pcm[k:k + n].tobytes())
                sent.append((min(k + n, len(pcm)), time.perf_counter()))
            await ws.send(json.dumps({"type": "end"}))

        task = asyncio.create_task(sender())
        async for raw in ws:
            recv.append((time.perf_counter(), json.loads(raw)))
        await task
    return {"ready": ready, "sent": sent, "recv": recv}


def sent_at(sent, samples: float) -> float | None:
    """perf time at which the audio up to ``samples`` had been sent."""
    for cum, p in sent:
        if cum >= samples - 1e-6:
            return p
    return None


def analyse_ws(run: dict, lead_mels: int | None, chunk_mel: int = 16, cs: int = 2) -> dict:
    sent, recv = run["sent"], run["recv"]
    hop, half = 160, 256
    lead = lead_mels or chunk_mel

    def ready_samples(v):
        return (lead + (v // cs) * chunk_mel - 1) * hop + half

    fr_compute, fr_label, part = [], [], []
    te, fin, stats = [], [], None
    for p, m in recv:
        if m["type"] == "frame":
            v = int(round(m["t"] / FRAME_S)) - 1
            s1, s2 = sent_at(sent, ready_samples(v)), sent_at(sent, m["t"] * SR)
            if s1:
                fr_compute.append((p - s1) * 1000)
            if s2:
                fr_label.append((p - s2) * 1000)
        elif m["type"] == "partial":
            v = int(round(m["t"] / FRAME_S)) - 1
            s1 = sent_at(sent, ready_samples(v))
            if s1:
                part.append((p - s1) * 1000)
        elif m["type"] == "turn_end":
            s = sent_at(sent, m["t"] * SR)
            te.append({"t": m["t"], "silence_ms": m["silence_ms"], "p": m.get("p"), "recv_perf": p,
                       "deliver_ms": round((p - s) * 1000, 1) if s else None})
        elif m["type"] == "final":
            fin.append({"t": m["t"], "text": m["text"], "speaker": m.get("speaker"), "recv_perf": p})
        elif m["type"] == "stats":
            stats = m
    end = REF["user_turn_ends"][0]
    user_te = [e for e in te if e["t"] >= end - 0.08 and e["t"] < end + 3.0]
    out = {"frame_after_chunk_ready_ms": summ(fr_compute), "frame_after_label_ms": summ(fr_label),
           "partial_after_chunk_ready_ms": summ(part), "turn_ends": te, "finals": fin}
    if user_te:
        e = user_te[0]
        s_end = sent_at(sent, end * SR)
        out["user_turn"] = {"ref_end_s": end, "decision_t": e["t"], "decision_after_end_ms": round((e["t"] - end) * 1000),
                            "silence_ms": e["silence_ms"], "deliver_ms": e["deliver_ms"],
                            "recv_after_end_ms": round((e["recv_perf"] - s_end) * 1000, 1)}
        f = [x for x in fin if abs(x["t"] - e["t"]) < 1e-6]
        if f:
            out["user_turn"]["final_after_turn_end_ms"] = round((f[0]["recv_perf"] - e["recv_perf"]) * 1000, 2)
            out["user_turn"]["final_text"] = f[0]["text"]
    for e in te:
        e.pop("recv_perf")
    for f in fin:
        f.pop("recv_perf")
    if stats:
        out["stats"] = {k: stats.get(k) for k in stats if any(s in k for s in ("chunk_ms", "send_lag", "rtf", "asr_ms",
                                                                                  "diar_ms", "turn_ms", "backlog"))}
    return out


def cmd_ws(a):
    code = Path(a.code).resolve()
    audio, emb = clip_audio(), json.loads(PRINT.read_text())
    srv = Server(code, a.extra.split() if a.extra else None)
    runs = []
    try:
        lead = None if a.tag == "before" else a.lead
        for r in range(a.reps):
            raw = asyncio.run(ws_once(srv.url, audio, emb))
            res = analyse_ws(raw, lead, a.chunk_mel, a.cs)
            ut = res.get("user_turn", {})
            log(f"ws {a.tag} rep {r}: turn_end t={ut.get('decision_t')} deliver {ut.get('deliver_ms')} ms, "
                f"frame {res['frame_after_chunk_ready_ms']['p50']} ms after chunk ready")
            runs.append(res)
    finally:
        srv.close()
    pool = {k: summ([x for r in runs for x in _vals(r, k)]) for k in ("deliver_ms", "recv_after_end_ms",
                                                                       "decision_after_end_ms",
                                                                       "final_after_turn_end_ms")}
    save(f"ws_{a.tag}", {"generated": time.strftime("%Y-%m-%d %H:%M"), "code": str(code), "extra": a.extra,
                         "reps": a.reps, "user_turn_pooled": pool,
                         "frame_after_chunk_ready_ms": summ([x for r in runs for x in _fr(r)]),
                         "runs": runs})
    log(f"ws {a.tag}: done {json.dumps(pool)}")


def _vals(r, k):
    v = r.get("user_turn", {}).get(k)
    return [v] if v is not None else []


def _fr(r):
    return [r["frame_after_chunk_ready_ms"]["p50"]]


# --------------------------------------------------------------------------- pipecat
def cmd_pipecat(a):
    import examples.pipecat_local_demo as PD
    from audioforge.integrations import pipecat as AP
    code = Path(a.code).resolve()
    audio, emb = clip_audio(), json.loads(PRINT.read_text())
    orig = AP.AudioforgeSTTService.__init__

    def init(self, *args, **kw):  # the stored print, sent right after the config (as the quickstart does)
        orig(self, *args, **kw)
        self._pending_controls = [{"type": "enroll", "embedding": emb}]
    AP.AudioforgeSTTService.__init__ = init
    PD.AudioforgeSTTService = AP.AudioforgeSTTService
    srv = Server(code, a.extra.split() if a.extra else None)
    out = []
    try:
        for r in range(a.reps):
            raw = asyncio.run(PD.run_pipeline(audio, url=srv.url, policy="hybrid_dyn", pad_s=2.0))
            t0 = raw["t0"]
            end = REF["user_turn_ends"][0]
            tes = [e for e in raw["server_turn_ends"] if end - 0.08 <= e["t"] < end + 3]
            ctx = [p for p, _ in raw["contexts"]]
            rec = {"turn_ends": [(e["t"], round((e["perf"] - t0) * 1000, 1)) for e in raw["server_turn_ends"]],
                   "contexts_ms": [round((p - t0) * 1000, 1) for p in ctx]}
            if tes:
                e = tes[0]
                c = [p for p in ctx if p >= e["perf"]]
                # the transport pushes block i at t0 + (i + 1) * 20 ms: the decision sample t was pushed at ceil(t / 20 ms)
                pushed = t0 + np.ceil(e["t"] / BLOCK_S - 1e-9) * BLOCK_S
                rec.update(decision_t=e["t"], arrival_after_push_ms=round((e["perf"] - pushed) * 1000, 1),
                           adapter_ms=round((c[0] - e["perf"]) * 1000, 1) if c else None,
                           response_after_t_ms=round((c[0] - (t0 + e["t"])) * 1000, 1) if c else None,
                           response_after_end_ms=round((c[0] - (t0 + end)) * 1000, 1) if c else None)
            rec["frame_lag_ms"] = summ(raw["frame_lag_ms"])
            log(f"pipecat rep {r}: {json.dumps({k: v for k, v in rec.items() if k != 'turn_ends'})}")
            out.append(rec)
    finally:
        srv.close()
    pool = {k: summ([r[k] for r in out if r.get(k) is not None])
            for k in ("arrival_after_push_ms", "adapter_ms", "response_after_t_ms", "response_after_end_ms")}
    save(f"pipecat_{a.tag}", {"generated": time.strftime("%Y-%m-%d %H:%M"), "reps": a.reps, "pooled": pool, "runs": out})
    log(f"pipecat {a.tag}: done {json.dumps(pool)}")


# --------------------------------------------------------------------------- wer
def cmd_wer(a):
    import torch
    torch.set_num_threads(2)
    from hybrid_asr import load_fa_set

    from audioforge.metrics import wer
    from audioforge.model import StreamingSession
    from audioforge.serve import fast_conv
    from audioforge.teachers import normalize_text
    from audioforge.train import load_model
    from audioforge import perf as P
    m = load_model(str(MODEL), "cpu").eval()
    fast_conv(m)
    P.apply(m, None, P.parse("default"))
    audios, refs = load_fa_set("libri")
    audios, refs = audios[: a.n], refs[: a.n]
    res = json.loads(OUT.read_text()).get("wer", {}) if OUT.exists() else {}
    blk = int(BLOCK_S * SR)
    t_start = time.time()
    for r in [int(x) for x in a.rights.split(",")]:
        key = f"70_{r}"
        if key in res and res[key]["n"] >= len(refs):
            continue
        hyps, chunk_ms, audio_s, wall = [], [], 0.0, 0.0
        for x in audios:
            s = StreamingSession(m, att_context_size=[70, r])
            n_ch = 0
            t0 = time.perf_counter()
            for k in range(0, len(x), blk):
                before = len(s.tokens), s.mel_fed
                tc = time.perf_counter()
                s.feed(x[k:k + blk])
                if s.mel_fed != before[1]:
                    chunk_ms.append((time.perf_counter() - tc) * 1000)
                    n_ch += 1
            s.feed(np.zeros(0, np.float32), final=True)
            wall += time.perf_counter() - t0
            audio_s += len(x) / SR
            hyps.append(s.text)
            if time.time() - t_start > a.budget:
                raise SystemExit(f"budget: stopped at {key} item {len(hyps)} (rerun; not resumable within a setting)")
        w = wer([normalize_text(t) for t in refs], [normalize_text(h) for h in hyps])
        res[key] = {"n": len(refs), "wer": round(100 * float(w), 2), "chunk_ms": summ(chunk_ms),
                    "rtf": round(wall / audio_s, 4), "hyps": hyps}
        log(f"wer [70,{r}]: {res[key]['wer']} % n={len(refs)} chunk p50 {res[key]['chunk_ms']['p50']} ms "
            f"rtf {res[key]['rtf']}")
        save("wer", res)
    # paired bootstrap on per-utterance errors
    if all(f"70_{r}" in res for r in (0, 1)):
        import jiwer
        e = {}
        for r in (0, 1):
            e[r] = []
            for ref, hyp in zip(refs, res[f"70_{r}"]["hyps"]):
                o = jiwer.process_words(normalize_text(ref) or "<empty>", normalize_text(hyp) or "<empty>")
                e[r].append((o.substitutions + o.deletions + o.insertions, len(normalize_text(ref).split())))
        rng = np.random.default_rng(0)
        E0, E1, N = (np.array([x[0] for x in e[0]]), np.array([x[0] for x in e[1]]), np.array([x[1] for x in e[0]]))
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(N), len(N))
            bs.append(100 * (E0[i].sum() - E1[i].sum()) / N[i].sum())
        res["delta_0_minus_1"] = {"wer_points": round(100 * (E0.sum() - E1.sum()) / N.sum(), 2),
                                  "ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]}
        save("wer", res)
        log(f"wer delta [70,0]-[70,1]: {res['delta_0_minus_1']}")


# --------------------------------------------------------------------------- compute
def _profile_hooks(eng) -> dict:
    """Wrap the parts of a chunk step with timers (ms summed over the run; '_chunks' = unconditioned encoder calls)."""
    from audioforge.server import streams as ST
    from audioforge.server import policies as PO
    from audioforge import tsvad_stream as TS
    from audioforge import lid as LI
    prof = {"_chunks": 0}

    def timed(obj, name, key, count=False, key_fn=None):
        f = getattr(obj, name)

        def w(*args, **kw):
            t0 = time.perf_counter()
            try:
                return f(*args, **kw)
            finally:
                k = key_fn(args, kw) if key_fn else key
                prof[k] = prof.get(k, 0.0) + (time.perf_counter() - t0) * 1000
                if count and k == key:
                    prof["_chunks"] += 1
        setattr(obj, name, w)
    enc = eng.asr.encoder
    timed(enc, "stream_step", "encoder", count=True,
          key_fn=lambda a, kw: "encoder_conditioned_turn_pass" if kw.get("spk_act") is not None else "encoder")
    timed(ST.ASRStream, "_decode", "rnnt_decode")
    timed(ST.ASRStream, "_turn_step", "turn_head_steps")
    timed(TS.TSVADTrack, "feed", "tsvad_head")
    timed(LI.LIDStream, "feed", "lid_head")
    timed(PO.SileroSilence, "feed", "silero")
    timed(ST.ASRStream, "feed_frames", "asr_feed_frames_total")
    return prof


def cmd_compute(a):
    import torch
    torch.set_num_threads(2)
    from audioforge.serve import Engine, Session
    from audioforge.server.cli import MODES
    from audioforge.server.protocol import SessionConfig
    audio, emb = clip_audio(), json.loads(PRINT.read_text())
    res = {}
    for cms in [int(x) for x in a.chunks.split(",")]:
        opts = {**MODES["single"], "tsvad": str(ROOT / "assets" / "tsvad_spk.pt"), "lid": str(ROOT / "assets" / "lid_distill.pt"),
                "silero": str(ROOT / "data" / "silero" / "silero_vad_v5.onnx"), "preload_silero": True,
                "asr_chunk_ms": cms}
        eng = Engine.load(str(MODEL), None, "cpu", threads=2, **opts)
        eng.warmup()
        prof = _profile_hooks(eng)
        per_rep = []
        for rep in range(a.reps):
            s = Session(eng, SessionConfig(turn_policy="hybrid_dyn"))  # the quickstart's policy
            s.arm_enrollment("enroll", 0, emb)
            blk = int(BLOCK_S * SR)
            x = np.concatenate([audio, np.zeros(int(1.5 * SR), np.float32)])
            block_ms, msgs = [], []
            for k in range(0, len(x), blk):
                t0 = time.perf_counter()
                out = s.process(x[k:k + blk])
                block_ms.append(((time.perf_counter() - t0) * 1000, bool(any(m["type"] == "frame" for m in out))))
                msgs += out
            msgs += s.finish()
            st = msgs[-1]
            te = [(m["t"], m["silence_ms"]) for m in msgs if m["type"] == "turn_end"]
            fin = [(m["t"], m["text"], m.get("speaker")) for m in msgs if m["type"] == "final"]
            chunk_blocks = [ms for ms, fr in block_ms if fr]
            other = [ms for ms, fr in block_ms if not fr]
            per_rep.append({"chunk_block_ms": summ(chunk_blocks), "other_block_ms": summ(other),
                            "rtf": st.get("rtf"), "turn_ends": te, "finals": fin,
                            "turn_ms_mean": st.get("turn_ms_mean")})
            log(f"compute {cms} ms rep {rep}: chunk-block p50 {per_rep[-1]['chunk_block_ms']['p50']} ms "
                f"p95 {per_rep[-1]['chunk_block_ms']['p95']}, other {per_rep[-1]['other_block_ms']['p50']} ms, "
                f"rtf {st.get('rtf')}, turn_ends {te}")
        n_chunks = max(1, prof.pop("_chunks"))
        split = {k: round(v / n_chunks, 2) for k, v in prof.items()}
        log(f"compute {cms} ms: ms per chunk by part {split}")
        res[str(cms)] = {"reps": per_rep, "ms_per_chunk_by_part": split}
        del eng
    save("compute", {"generated": time.strftime("%Y-%m-%d %H:%M"), "threads": 2, "res": res})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("ws")
    w.add_argument("--tag", required=True)
    w.add_argument("--code", default=str(ROOT))
    w.add_argument("--reps", type=int, default=5)
    w.add_argument("--extra", default="", help="more server flags, e.g. '--asr-chunk-ms 80'")
    w.add_argument("--lead", type=int, default=9, help="first-chunk mel frames of the code under test")
    w.add_argument("--chunk-mel", type=int, default=16)
    w.add_argument("--cs", type=int, default=2)
    p = sub.add_parser("pipecat")
    p.add_argument("--tag", required=True)
    p.add_argument("--code", default=str(ROOT))
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--extra", default="")
    r = sub.add_parser("wer")
    r.add_argument("--rights", default="1,0")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--budget", type=float, default=540)
    c = sub.add_parser("compute")
    c.add_argument("--chunks", default="160,80")
    c.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    {"ws": cmd_ws, "pipecat": cmd_pipecat, "wer": cmd_wer, "compute": cmd_compute}[a.cmd](a)


if __name__ == "__main__":
    main()
