"""Real-time WebSocket client for audioforge.serve: stream a WAV/FLAC (any rate -> 16 kHz) in 20 ms frames.

    PYTHONPATH=. .venv/bin/python scripts/stream_client.py audio.wav [--url ws://127.0.0.1:8765] [--speed 1]
        [--policy timeout|head|both] [--timeout-ms 1000] [--eot-threshold 0.98]
        [--log events.jsonl] [--summary summary.json] [--ref ref.json] [--ref-text "..."] [--verbose]

--speed 1 paces the audio at real time (frame i is sent when its last sample would exist: t0 + (i+1)*20 ms);
--speed 0 sends as fast as possible (throughput / RTF only; lags are then meaningless). Every message is logged
with its wall-clock arrival time. The summary reports:

* first_partial: audio time t of the first non-empty partial and its arrival latency measured three ways -
  vs the moment the audio up to t had been sent (server + transport), vs the end of the first word (if --ref
  gives first_word_end_s, e.g. AMI word timings), and vs the energy speech onset of the file;
* turn_end events (t, policy, p, silence_ms, arrival lag), the finals and the joined final transcript
  (WER vs --ref text if given), the server's stats message;
* keeping up at 1x: lag of every frame message (arrival - time its audio was sent), p50/p95/max and the slope
  of lag over the stream (ms of extra lag per s of audio; > ~20 ms/s = queue growth).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audioforge.metrics import pct  # noqa: E402

FRAME_S = 0.02


def speech_onset_s(x: np.ndarray, sr: int = 16000, rel_db: float = -30.0) -> float | None:
    """First 20 ms frame whose RMS is within ``rel_db`` of the loudest frame (a crude energy onset)."""
    n = int(sr * FRAME_S)
    k = len(x) // n
    if k == 0:
        return None
    rms = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, 1) + 1e-12)
    hit = np.nonzero(20 * np.log10(rms / rms.max()) > rel_db)[0]
    return float(hit[0] * FRAME_S) if len(hit) else None


def _pct(xs, q):
    return pct(xs, q, nd=1, empty=None)


async def run(a) -> dict:
    from websockets.asyncio.client import connect

    from audioforge.data import load_wav
    x = load_wav(a.audio, 16000)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    n = int(16000 * FRAME_S)
    frames = [pcm[i: i + n].tobytes() for i in range(0, len(pcm), n)]
    log = open(a.log, "w") if a.log else None
    events: list[dict] = []
    t_start = None  # perf_counter at which audio time 0 "started"
    done = asyncio.Event()

    def stamp(msg):
        now = time.perf_counter()
        rec = {"wall": time.time(), "rel_ms": round((now - t_start) * 1000, 1) if t_start else None, "msg": msg}
        events.append(rec)
        if log:
            log.write(json.dumps(rec) + "\n")
        typ = msg.get("type")
        if typ != "frame" or a.verbose:
            ts = time.strftime("%H:%M:%S", time.localtime(rec["wall"])) + f".{int(rec['wall'] * 1000) % 1000:03d}"
            body = {k: v for k, v in msg.items() if k != "type"}
            print(f"{ts} +{rec['rel_ms'] or 0:8.1f}ms {typ:8s} {json.dumps(body)}", flush=True)
        return rec

    async with connect(a.url, max_size=2 ** 22, ping_interval=None) as ws:
        ready = json.loads(await ws.recv())
        stamp(ready)
        cfg = {"type": "config", "turn_policy": a.policy, "timeout_ms": a.timeout_ms,
               "eot_threshold": a.eot_threshold, "sample_rate": 16000}
        await ws.send(json.dumps(cfg))

        async def receiver():
            # ends on the server's stats, on a fatal ``error`` (the server closes right after it) or on any close;
            # a non-fatal error is a degradation report: logged by stamp(), the session goes on
            try:
                async for m in ws:
                    try:
                        msg = json.loads(m)
                    except ValueError:
                        print(f"[client] ignoring a non-JSON server frame: {m[:80]!r}", flush=True)
                        continue
                    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
                        print(f"[client] ignoring a malformed server message: {m[:80]!r}", flush=True)
                        continue
                    stamp(msg)
                    if msg.get("type") == "stats":
                        break
            except Exception as e:  # noqa: BLE001 - ConnectionClosed etc.: the sender stops, summarize() reports
                print(f"[client] connection ended: {type(e).__name__}: {e}", flush=True)
            finally:
                done.set()

        rtask = asyncio.create_task(receiver())
        t_start = time.perf_counter()
        t_sent = None
        try:
            for i, fr in enumerate(frames):
                if done.is_set():  # the server closed the session (fatal error): stop sending, keep what arrived
                    break
                if a.speed > 0:
                    due = t_start + (i + 1) * FRAME_S / a.speed
                    d = due - time.perf_counter()
                    if d > 0:
                        await asyncio.sleep(d)
                await ws.send(fr)
            t_sent = time.perf_counter()
            if not done.is_set():
                await ws.send(json.dumps({"type": "end"}))
        except Exception as e:  # noqa: BLE001 - the socket went away mid-stream: report, do not crash
            print(f"[client] send failed: {type(e).__name__}: {e}", flush=True)
            t_sent = t_sent or time.perf_counter()
        await done.wait()
        await rtask
    if log:
        log.close()
    return summarize(a, x, events, t_start, t_sent)


def summarize(a, x, events, t_start, t_sent) -> dict:
    speed = a.speed if a.speed > 0 else None
    ref = json.loads(Path(a.ref).read_text()) if a.ref else {}

    def sent_at_ms(t_audio):  # rel ms at which audio up to t_audio had been sent (paced runs only)
        return t_audio * 1000 / speed if speed else None

    msgs = [(e["rel_ms"], e["msg"]) for e in events if e["rel_ms"] is not None]
    fp = next(((r, m) for r, m in msgs if m["type"] == "partial" and m["text"]), None)
    onset = speech_onset_s(x)
    first = None
    if fp:
        r, m = fp
        first = {"t_audio": m["t"], "arrival_rel_ms": r, "text": m["text"]}
        if speed:
            first["latency_vs_audio_sent_ms"] = round(r - sent_at_ms(m["t"]), 1)
            if "first_word_end_s" in ref:
                first["latency_vs_first_word_end_ms"] = round(r - sent_at_ms(ref["first_word_end_s"]), 1)
            if onset is not None:
                first["latency_vs_energy_onset_ms"] = round(r - sent_at_ms(onset), 1)
    turn_ends = []
    for r, m in msgs:
        if m["type"] == "turn_end":
            te = {k: m[k] for k in ("t", "policy", "p", "silence_ms")}
            if speed:
                te["arrival_lag_ms"] = round(r - sent_at_ms(m["t"]), 1)
                if "turn_end_s" in ref:
                    te["after_ref_turn_end_ms"] = round(r - sent_at_ms(ref["turn_end_s"]), 1)
            turn_ends.append(te)
    finals = [m for _, m in msgs if m["type"] == "final"]
    transcript = " ".join(f["text"] for f in finals if f["text"]).strip()
    stats = next((m for _, m in msgs if m["type"] == "stats"), None)
    frame_lag = [(m["t"], r - sent_at_ms(m["t"])) for r, m in msgs if m["type"] == "frame"] if speed else []
    keep = None
    if frame_lag:
        ts, lags = np.array([f[0] for f in frame_lag]), np.array([f[1] for f in frame_lag])
        slope = float(np.polyfit(ts, lags, 1)[0]) if len(ts) > 2 else 0.0
        keep = {"frame_lag_ms_p50": _pct(lags, 50), "frame_lag_ms_p95": _pct(lags, 95),
                "frame_lag_ms_max": round(float(lags.max()), 1), "lag_slope_ms_per_s": round(slope, 2),
                "lag_first_quarter_ms": _pct(lags[: max(1, len(lags) // 4)], 50),
                "lag_last_quarter_ms": _pct(lags[-max(1, len(lags) // 4):], 50),
                "keeps_up": bool(slope < 20.0 and lags[-max(1, len(lags) // 4):].max() < 1500)}
    errors = [m for _, m in msgs if m["type"] == "error"]
    out = {"audio": str(a.audio), "audio_s": round(len(x) / 16000, 3), "speed": a.speed, "policy": a.policy,
           "timeout_ms": a.timeout_ms, "send_wall_s": round(t_sent - t_start, 3),
           "errors": [{k: m.get(k) for k in ("code", "detail", "fatal")} for m in errors],
           "fatal_error": next((m.get("code") for m in errors if m.get("fatal")), None),
           "complete": stats is not None,
           "n_frames": sum(1 for _, m in msgs if m["type"] == "frame"),
           "n_partials": sum(1 for _, m in msgs if m["type"] == "partial"),
           "first_partial": first, "energy_onset_s": onset, "turn_ends": turn_ends,
           "finals": [{"t": f["t"], "speaker": f["speaker"], "text": f["text"]} for f in finals],
           "transcript": transcript, "keeping_up": keep, "server_stats": stats}
    ref_text = a.ref_text or ref.get("all_text") or ref.get("text")
    if ref_text:
        from audioforge.metrics import wer
        from audioforge.teachers import normalize_text
        out["wer_vs_ref"] = round(float(wer([normalize_text(ref_text)], [normalize_text(transcript)])), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stream_client", description=__doc__.split("\n\n")[0])
    ap.add_argument("audio")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = as fast as possible")
    ap.add_argument("--policy", choices=["timeout", "head", "both"], default="timeout")
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--eot-threshold", type=float, default=0.98)
    ap.add_argument("--log", help="JSONL event log (one line per message: wall, rel_ms, msg)")
    ap.add_argument("--summary", help="write the summary JSON here")
    ap.add_argument("--ref", help="reference JSON (first_word_end_s, turn_end_s, all_text / text)")
    ap.add_argument("--ref-text", help="reference transcript for WER")
    ap.add_argument("--verbose", action="store_true", help="also print every frame message")
    a = ap.parse_args(argv)
    s = asyncio.run(run(a))
    txt = json.dumps(s, indent=1)
    if a.summary:
        Path(a.summary).write_text(txt)
    print("SUMMARY " + json.dumps({k: s[k] for k in ("audio_s", "first_partial", "turn_ends", "transcript",
                                                       "keeping_up", "server_stats") if k in s}))
    return s


if __name__ == "__main__":
    main()
