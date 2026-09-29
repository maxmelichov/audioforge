"""LiveKit Agents plugin (audioforge/integrations/livekit.py) against a scripted fake server, the real serve.py with
tiny models, and a real (room-less) AgentSession. The real-checkpoint demo runs only with RUN_REAL=1."""
import asyncio
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

lk_agents = pytest.importorskip("livekit.agents")
websockets = pytest.importorskip("websockets")

from livekit import rtc  # noqa: E402
from livekit.agents import stt, vad  # noqa: E402
from livekit.agents._exceptions import APIConnectionError  # noqa: E402
from livekit.agents.types import APIConnectOptions  # noqa: E402
from livekit.agents.voice.turn import _StreamingTurnDetector, _StreamingTurnDetectorStream  # noqa: E402

from audioforge.integrations.livekit import (  # noqa: E402
    AudioforgeFrontend,
    AudioforgeOptions,
    AudioforgeSTT,
    _SpeechMapper,
    _VadMapper,
)

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
SE = stt.SpeechEventType
VE = vad.VADEventType


# --------------------------------------------------------------------------- scripted fake server
class FakeServer:
    """Stand-in for audioforge.serve speaking the same protocol, driven by the audio's energy:
    frame per 80 ms (vad 0.95 / speaker 0 on frames with RMS > thr), a word "w<n>" per 5 speech frames as partials,
    a timeout turn_end once the primary was silent for timeout_ms (sent ``lag_s`` of audio later, like the
    diarizer), the head's turn_end when its fake eot crosses the threshold, finals at the cutting turn_end and at
    "end", then stats and close. ``drop_after_s`` closes the socket abruptly."""

    def __init__(self, thr=0.02, lag_s=0.56, drop_after_s=None):
        self.thr, self.lag_s, self.drop_after_s = thr, lag_s, drop_after_s
        self.connections, self.configs, self.samples, self.ends = 0, [], [], 0
        self.controls = []  # (connection index, type, samples received before it)
        self.port = None

    async def handler(self, ws):
        self.connections += 1
        idx = len(self.samples)
        self.samples.append(0)
        cfg = {"turn_policy": "timeout", "timeout_ms": 1000, "eot_threshold": 0.98}
        st = dict(buf=np.zeros(0, np.float32), v=0, silent=0, spoke=False, words=0, seg_words=0, speech=0,
                  pending=[], head_armed=False, prev_eot=0.0, recent=[])
        await ws.send(json.dumps({"type": "ready", "model": "fake", "chunk_ms": 160, "frame_ms": 80}))

        def cut():
            return "head" if cfg["turn_policy"] == "head" else "timeout"

        def text():
            return " ".join(f"w{i}" for i in range(st["words"] - st["seg_words"], st["words"]))

        async def emit_due(t_audio, final=False):
            due = [p for p in st["pending"] if p[0] <= t_audio + 1e-9 or final]
            st["pending"] = [p for p in st["pending"] if p not in due]
            for t, pol, sil in due:
                await ws.send(json.dumps({"type": "turn_end", "t": round(t, 3), "policy": pol,
                                          "p": 0.99 if pol == "head" else None, "silence_ms": sil}))
                if pol == cut():
                    await ws.send(json.dumps({"type": "final", "t": round(t, 3), "text": text(), "speaker": 0}))
                    st["seg_words"] = 0

        async for msg in ws:
            if isinstance(msg, str):
                d = json.loads(msg)
                if d["type"] == "config":
                    cfg.update({k: d[k] for k in ("turn_policy", "timeout_ms", "eot_threshold") if k in d})
                    self.configs.append(d)
                    continue
                if d["type"] in ("agent_end", "enroll"):
                    self.controls.append((idx, d["type"], self.samples[idx]))
                    continue
                if d["type"] == "end":
                    self.ends += 1
                    t = self.samples[idx] / SR
                    await emit_due(t, final=True)
                    await ws.send(json.dumps({"type": "final", "t": round(t, 3), "text": text(), "speaker": 0}))
                    await ws.send(json.dumps({"type": "stats", "rtf": 0.1, "chunk_ms_p50": 1.0, "chunk_ms_p95": 2.0,
                                              "first_partial_ms": 5.0, "peak_rss_mb": 10.0}))
                    await ws.close()
                    return
                continue
            if st.get("dropped"):
                continue  # keep draining: websockets pauses reading (and misses the close reply) on a backlog
            x = np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768
            self.samples[idx] += len(x)
            st["buf"] = np.concatenate([st["buf"], x])
            if self.drop_after_s is not None and self.samples[idx] / SR >= self.drop_after_s:
                st["dropped"] = True
                asyncio.ensure_future(ws.close(code=1011))
                continue
            items = []
            while len(st["buf"]) >= 1280:
                seg, st["buf"] = st["buf"][:1280], st["buf"][1280:]
                st["v"] += 1
                t = st["v"] * 0.08
                speech = float(np.sqrt(np.mean(seg ** 2))) > self.thr
                st["recent"] = (st["recent"] + [speech])[-62:]
                if speech:
                    st["silent"], st["spoke"], st["speech"] = 0, True, st["speech"] + 1
                    st["head_armed"] = True
                    if st["speech"] % 5 == 0:
                        st["words"] += 1
                        st["seg_words"] += 1
                else:
                    st["silent"] += 1
                eot = 0.05 if speech else min(0.995, 0.2 + 0.1 * st["silent"])
                prim = 0 if any(st["recent"]) else None
                items.append({"t": round(t, 3), "vad": 0.95 if speech else 0.02, "eot": eot,
                              "speakers": [0.9 if speech else 0.02, 0.02, 0.02, 0.02], "primary": prim})
                if st["spoke"] and st["silent"] * 80 >= cfg["timeout_ms"] and cfg["turn_policy"] != "head":
                    st["pending"].append((t + self.lag_s, "timeout", st["silent"] * 80))
                    st["spoke"] = False
                if cfg["turn_policy"] != "timeout" and st["head_armed"] and eot >= cfg["eot_threshold"] > st["prev_eot"]:
                    st["pending"].append((t, "head", st["silent"] * 80))
                    st["head_armed"] = False
                st["prev_eot"] = eot
            if items:
                await ws.send(json.dumps({"type": "frames", "items": items}))
                if st["words"] and text():
                    await ws.send(json.dumps({"type": "partial", "t": items[-1]["t"], "text": text()}))
            await emit_due(self.samples[idx] / SR)

    async def __aenter__(self):
        from websockets.asyncio.server import serve
        self._srv = await serve(self.handler, "127.0.0.1", 0).__aenter__()
        self.port = self._srv.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *a):
        self._srv.close()
        await self._srv.wait_closed()


def speech_silence(pattern=((1.0, 0.2), (1.6, 0.0), (0.8, 0.2), (1.6, 0.0)), seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([(rng.standard_normal(int(SR * d)) * a).astype(np.float32) for d, a in pattern])


def frames_of(x, sr=SR, ms=20):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = sr * ms // 1000
    return [rtc.AudioFrame(pcm[i:i + n].tobytes(), sr, 1, len(pcm[i:i + n])) for i in range(0, len(pcm), n)]


async def push_all(streams, frames, pace=0.0):
    for f in frames:
        for s in streams:
            (s.push_audio if hasattr(s, "push_audio") else s.push_frame)(f)
        await asyncio.sleep(pace)


async def collect(stream, timeout=10.0):
    out = []

    async def run():
        async for ev in stream:
            out.append((time.time(), ev))
    await asyncio.wait_for(run(), timeout)
    return out


# --------------------------------------------------------------------------- pure mappers
def _wall(t):
    return 1000.0 + t


def test_speech_mapper_turn_sequence_and_metadata():
    m = _SpeechMapper(AudioforgeOptions(), min_speech_ms=160)
    ev = []
    for i in range(1, 4):
        ev += m.on_message({"type": "frame", "t": 0.08 * i, "vad": 0.9, "eot": 0.1, "speakers": [0.9, 0, 0, 0],
                            "primary": 1}, _wall, offset=2.0)
    assert [e.type for e in ev] == [SE.START_OF_SPEECH]  # after 2 frames = 160 ms
    assert ev[0].speech_start_time == pytest.approx(_wall(0.0))
    ev = m.on_message({"type": "partial", "t": 0.32, "text": "hello"}, _wall, offset=2.0)
    assert [e.type for e in ev] == [SE.INTERIM_TRANSCRIPT] and ev[0].alternatives[0].text == "hello"
    assert ev[0].alternatives[0].end_time == pytest.approx(2.32)
    assert m.on_message({"type": "partial", "t": 0.40, "text": "hello"}, _wall) == []  # unchanged text
    assert m.on_message({"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1040},
                        _wall) == []
    ev = m.on_message({"type": "final", "t": 2.0, "text": "hello there", "speaker": 1}, _wall, offset=2.0)
    assert [e.type for e in ev] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH]
    d = ev[0].alternatives[0]
    assert d.text == "hello there" and d.speaker_id == "S1" and d.metadata["audioforge"]["speaker"] == 1
    assert d.metadata["audioforge"]["turn_end"]["silence_ms"] == 1040
    # the primary's last active frame: decision time - mean diarizer lag (0.76 s) - silence
    assert ev[1].speech_end_time == pytest.approx(_wall(2.0 - 0.76 - 1.04))
    m.on_message({"type": "partial", "t": 1.2, "text": "x"}, _wall)
    m.on_message({"type": "turn_end", "t": 3.0, "policy": "timeout", "p": None, "silence_ms": 1040, "frame_t": 2.4},
                 _wall)  # server --debug-fields: exact deciding frame
    ev2 = m.on_message({"type": "final", "t": 3.0, "text": "x", "speaker": 1}, _wall)
    assert ev2[-1].speech_end_time == pytest.approx(_wall(2.4 - 1.04))
    # the next turn needs new speech; the end-of-stream final closes it
    ev = m.on_message({"type": "partial", "t": 2.4, "text": "again"}, _wall)
    assert [e.type for e in ev] == [SE.START_OF_SPEECH, SE.INTERIM_TRANSCRIPT]
    ev = m.on_message({"type": "final", "t": 3.0, "text": "again", "speaker": None}, _wall)
    assert [e.type for e in ev] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH] and ev[0].alternatives[0].speaker_id is None
    ev = m.on_message({"type": "stats", "rtf": 0.3, "chunk_ms_p50": 1, "chunk_ms_p95": 2, "first_partial_ms": 3,
                       "peak_rss_mb": 4}, _wall, audio_s=3.0)
    assert ev[0].type == SE.RECOGNITION_USAGE and ev[0].recognition_usage.audio_duration == 3.0


def test_speech_mapper_both_policy_only_timeout_cuts():
    m = _SpeechMapper(AudioforgeOptions(turn_policy="both"))
    m.on_message({"type": "partial", "t": 0.5, "text": "a"}, _wall)
    assert m.on_message({"type": "turn_end", "t": 0.9, "policy": "head", "p": 0.99, "silence_ms": 80}, _wall) == []
    m.on_message({"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1040}, _wall)
    ev = m.on_message({"type": "final", "t": 2.0, "text": "a b", "speaker": 0}, _wall)
    assert [e.type for e in ev] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH]
    assert ev[0].alternatives[0].metadata["audioforge"]["other_turn_end"]["policy"] == "head"
    m = _SpeechMapper(AudioforgeOptions(turn_policy="head"))
    m.on_message({"type": "partial", "t": 0.5, "text": "a"}, _wall)
    m.on_message({"type": "turn_end", "t": 0.9, "policy": "head", "p": 0.99, "silence_ms": 80}, _wall)
    assert m.pending_turn_end["policy"] == "head"


def test_vad_mapper_hysteresis_and_durations():
    m = _VadMapper(0.5, min_speech_duration=0.16, min_silence_duration=0.56)
    ps = [0.1, 0.9, 0.9, 0.9, 0.2, 0.9] + [0.1] * 7 + [0.1]
    evs = []
    for i, p in enumerate(ps):
        evs.append(m.on_frame({"t": 0.08 * (i + 1), "vad": p}, now=_wall(0.08 * (i + 1)) + 0.1, wall_of=_wall))
    types = [[t for t, _ in e] for e in evs]
    assert all(t[-1] == VE.INFERENCE_DONE for t in types)  # one INFERENCE_DONE per frame
    assert types[2] == [VE.START_OF_SPEECH, VE.INFERENCE_DONE]  # 2 speech frames
    start = evs[2][0][1]
    assert start["speech_duration"] == pytest.approx(0.16) and start["lag"] == pytest.approx(0.1)
    ends = [(i, kw) for i, e in enumerate(evs) for t, kw in e if t == VE.END_OF_SPEECH]
    assert len(ends) == 1 and ends[0][0] == 12  # 7 silent frames after the last speech frame (index 5)
    assert ends[0][1]["silence_duration"] == pytest.approx(0.56)
    assert ends[0][1]["span"] == pytest.approx((0.08, 0.48))
    inf = evs[4][-1][1]
    assert inf["speaking"] and inf["raw_silence"] == pytest.approx(0.08) and inf["speech_duration"] > 0.3


# --------------------------------------------------------------------------- streams over the fake server
def test_stt_stream_events_over_fake_server():
    async def body():
        async with FakeServer() as srv:
            s = AudioforgeSTT(srv.url, timeout_ms=600)
            st = s.stream()
            x = speech_silence()
            task = asyncio.create_task(collect(st))
            await push_all([st], frames_of(x))
            st.end_input()
            evs = await task
            await st.aclose()
            await s.frontend.aclose()
            return srv, x, evs

    srv, x, evs = asyncio.run(body())
    types = [e.type for _, e in evs]
    assert srv.connections == 1 and srv.samples[0] == len(x) and srv.ends == 1
    assert srv.configs[0] == {"type": "config", "turn_policy": "timeout", "timeout_ms": 600,
                              "sample_rate": 16000}  # eot_threshold left to the server unless set
    # two speech bursts -> two turns, each START ... INTERIM ... FINAL, END; usage at the end
    assert types.count(SE.START_OF_SPEECH) == 2 and types.count(SE.END_OF_SPEECH) == 2
    assert types.count(SE.FINAL_TRANSCRIPT) == 2 and types[-1] == SE.RECOGNITION_USAGE
    for i, t in enumerate(types):
        if t == SE.END_OF_SPEECH:
            assert types[i - 1] == SE.FINAL_TRANSCRIPT
    finals = [e.alternatives[0] for _, e in evs if e.type == SE.FINAL_TRANSCRIPT]
    # 13 speech frames then 11 (a word per 5 speech frames): "w0 w1" | "w2 w3"
    assert finals[0].text == "w0 w1" and finals[0].speaker_id == "S0" and finals[1].text == "w2 w3"
    assert evs[-1][1].recognition_usage.audio_duration == pytest.approx(len(x) / SR)


def test_stt_stream_resamples_48k_input():
    async def body():
        async with FakeServer() as srv:
            s = AudioforgeSTT(srv.url)
            st = s.stream()
            x = speech_silence(((0.6, 0.2), (0.4, 0.0)))
            x48 = np.repeat(x, 3)
            task = asyncio.create_task(collect(st))
            await push_all([st], frames_of(x48, sr=48000))
            st.end_input()
            await task
            await st.aclose()
            await s.frontend.aclose()
            return srv, x

    srv, x = asyncio.run(body())
    assert abs(srv.samples[0] - len(x)) <= 400  # LiveKit's resampler (HIGH quality) -> 16 kHz


def test_shared_session_stt_vad_turn_detector():
    """One server session for all three components; only one of them feeds audio."""
    async def body():
        async with FakeServer() as srv:
            fe = AudioforgeFrontend(srv.url, timeout_ms=600)
            s, v, td = fe.stt(), fe.vad(), fe.turn_detector()
            assert isinstance(td, _StreamingTurnDetector)
            ss, vs = s.stream(), v.stream()
            ts = td.stream()
            assert isinstance(ts, _StreamingTurnDetectorStream)
            x = speech_silence()
            fr = frames_of(x)
            t_stt = asyncio.create_task(collect(ss))
            t_vad = asyncio.create_task(collect(vs))
            futs = []
            for i, f in enumerate(fr):
                ss.push_frame(f)
                vs.push_frame(f)
                ts.push_audio(f)
                if i == 55:  # 0.1 s into the first silence (speech ends at 1.0 s)
                    futs.append(ts.predict())
                await asyncio.sleep(0.0005)
            await asyncio.sleep(0.2)
            ss.end_input()
            vs.end_input()
            ts.end_input()
            e_stt, e_vad = await t_stt, await t_vad
            await ss.aclose()
            await vs.aclose()
            await ts.aclose()
            await fe.aclose()
            return srv, x, e_stt, e_vad, [f.result() for f in futs], ts.predictions

    srv, x, e_stt, e_vad, preds, log = asyncio.run(body())
    assert srv.connections == 1 and srv.samples == [len(x)]  # not 3x the audio
    vtypes = [e.type for _, e in e_vad]
    assert vtypes.count(VE.START_OF_SPEECH) == 2 and vtypes.count(VE.END_OF_SPEECH) == 2
    assert vtypes.count(VE.INFERENCE_DONE) == len(x) // 1280
    eos = [e for _, e in e_vad if e.type == VE.END_OF_SPEECH][0]
    assert eos.frames and 0.9 <= sum(f.duration for f in eos.frames) <= 1.2  # the first 1.0 s of speech
    assert preds[0].end_of_turn_probability == 1.0 and log[0]["why"] == "turn_end"
    assert [e.type for _, e in e_stt].count(SE.FINAL_TRANSCRIPT) == 2


def test_turn_detector_cancel_deadline_and_armed():
    async def body():
        async with FakeServer() as srv:
            fe = AudioforgeFrontend(srv.url, timeout_ms=400)
            ts = fe.turn_detector(prediction_timeout=0.3).stream()
            assert await ts.unlikely_threshold(None) == 0.5 and await ts.supports_language(None)
            f1 = ts.predict()
            ts.cancel_inference()  # new speech
            r1 = await f1
            f2 = ts.predict()  # nothing arrives -> deadline
            r2 = await asyncio.wait_for(f2, 2)
            x = speech_silence(((0.5, 0.2), (1.5, 0.0)))
            for f in frames_of(x):
                ts.push_audio(f)
                await asyncio.sleep(0.0005)
            await asyncio.sleep(0.3)
            armed = ts._armed_t is not None  # turn_end arrived with no request pending
            r3 = await ts.predict()
            await ts.aclose()
            await fe.aclose()
            return r1, r2, armed, r3, ts.predictions

    r1, r2, armed, r3, log = asyncio.run(body())
    assert r1.end_of_turn_probability == 0.0 and r2.end_of_turn_probability == 0.0
    assert armed and r3.end_of_turn_probability == 1.0
    assert [p["why"] for p in log] == ["cancelled", "deadline", "turn_end_before_request"]


def test_turn_detector_ignores_stale_turn_end():
    """A timeout turn_end followed by more speech must not answer a later request (it was about an earlier pause)."""
    async def body():
        async with FakeServer() as srv:
            fe = AudioforgeFrontend(srv.url, timeout_ms=400)
            ts = fe.turn_detector().stream()
            # speech 0.5 s | silence 1.2 s (turn_end decided ~0.96 s, sent ~1.52 s) | speech 0.5 s | silence 1.5 s
            x = speech_silence(((0.5, 0.2), (1.2, 0.0), (0.5, 0.2), (1.5, 0.0)))
            fut = None
            for i, f in enumerate(frames_of(x)):
                ts.push_audio(f)
                if i == 115:  # 0.1 s into the second silence
                    fut = ts.predict()
                    immediate = fut.done()
                await asyncio.sleep(0.001)
            r = await asyncio.wait_for(fut, 3)
            await ts.aclose()
            await fe.aclose()
            return immediate, r, ts.predictions

    immediate, r, log = asyncio.run(body())
    assert not immediate and r.end_of_turn_probability == 1.0 and log[0]["why"] == "turn_end"


def test_turn_detector_head_policy_returns_eot():
    async def body():
        async with FakeServer() as srv:
            fe = AudioforgeFrontend(srv.url, turn_policy="both", eot_threshold=0.9)
            ts = fe.turn_detector(policy="head").stream()
            assert await ts.unlikely_threshold(None) == 0.9
            x = speech_silence(((0.5, 0.2), (1.0, 0.0)))
            fr = frames_of(x)
            fut = None
            for i, f in enumerate(fr):
                ts.push_audio(f)
                if i == 35:  # 0.2 s of silence
                    fut = ts.predict()
                await asyncio.sleep(0.0005)
            r = await asyncio.wait_for(fut, 2)
            await ts.aclose()
            await fe.aclose()
            return r

    r = asyncio.run(body())
    assert 0.2 < r.end_of_turn_probability < 0.9  # the fake head's eot a few frames into the silence


def test_connection_refused_raises_api_error():
    async def body():
        s = AudioforgeSTT("ws://127.0.0.1:9", connect_timeout=2)
        st = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2))
        st.push_frame(frames_of(np.zeros(320, np.float32))[0])
        with pytest.raises(APIConnectionError):
            async for _ in st:
                pass
        await st.aclose()

    asyncio.run(body())


def test_server_drop_mid_stream_raises_after_retries():
    async def body():
        async with FakeServer(drop_after_s=0.3) as srv:
            s = AudioforgeSTT(srv.url)
            st = s.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0.05, timeout=2))
            fr = frames_of(speech_silence(((2.0, 0.2),)))

            async def feed():
                for f in fr:
                    try:
                        st.push_frame(f)
                    except RuntimeError:
                        return
                    await asyncio.sleep(0.002)
            ft = asyncio.create_task(feed())
            with pytest.raises(APIConnectionError):
                async for _ in st:
                    pass
            ft.cancel()
            await st.aclose()
            return srv

    srv = asyncio.run(body())
    assert srv.connections == 2  # one retry opened a fresh server session


def test_recognize_batch():
    async def body():
        async with FakeServer() as srv:
            s = AudioforgeSTT(srv.url, timeout_ms=400)
            fr = frames_of(speech_silence(((1.0, 0.2), (1.2, 0.0), (0.4, 0.2))))
            ev = await s.recognize(fr)
            return ev

    ev = asyncio.run(body())
    assert ev.type == SE.FINAL_TRANSCRIPT and ev.alternatives[0].text == "w0 w1 w2"  # 13 + 6 speech frames


# --------------------------------------------------------------------------- real serve.py (tiny models)
def _load_test_serve():
    spec = importlib.util.spec_from_file_location("_ts_for_livekit", ROOT / "tests" / "test_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_server_tiny_models_protocol_compat():
    """audioforge.serve (tiny random ASR + energy diarizer) behind the plugin: the server's timeout turn_end becomes
    FINAL_TRANSCRIPT + END_OF_SPEECH with the server's text and speaker."""
    ts = _load_test_serve()
    from audioforge.serve import serve
    eng = ts._energy_engine()
    x = ts._speech_silence()

    async def body():
        loop = asyncio.get_running_loop()
        port_f, stop = loop.create_future(), loop.create_future()
        task = asyncio.create_task(serve(eng, "127.0.0.1", 0, ready_event=port_f.set_result, stop=stop))
        port = await asyncio.wait_for(port_f, 10)
        try:
            fe = AudioforgeFrontend(f"ws://127.0.0.1:{port}", timeout_ms=400)
            ss, vs = fe.stt().stream(), fe.vad().stream()
            t1, t2 = asyncio.create_task(collect(ss, 30)), asyncio.create_task(collect(vs, 30))
            for f in frames_of(x):
                ss.push_frame(f)
                vs.push_frame(f)
                await asyncio.sleep(0.001)
            ss.end_input()
            vs.end_input()
            e1, e2 = await t1, await t2
            await ss.aclose()
            await vs.aclose()
            await fe.aclose()
            return e1, e2, fe.last_stats
        finally:
            stop.set_result(None)
            await task

    e1, e2, stats = asyncio.run(body())
    types = [e.type for _, e in e1]
    finals = [e.alternatives[0] for _, e in e1 if e.type == SE.FINAL_TRANSCRIPT]
    # test_serve.test_turn_events_timing_and_final_cuts: first cut at t = 2.0 s with 24 tokens, speaker 0
    assert finals[0].text == "a" * 24 and finals[0].speaker_id == "S0"
    assert finals[0].metadata["audioforge"]["t"] == 2.0 and finals[0].metadata["audioforge"]["turn_end"]["policy"] == "timeout"
    i = types.index(SE.FINAL_TRANSCRIPT)
    assert types[i + 1] == SE.END_OF_SPEECH and types[0] == SE.START_OF_SPEECH
    assert types.count(SE.INTERIM_TRANSCRIPT) > 5 and types[-1] == SE.RECOGNITION_USAGE
    assert sum(1 for _, e in e2 if e.type == VE.INFERENCE_DONE) >= len(x) // 1280 - 1  # VAD head always on
    assert stats is not None and stats["rtf"] > 0


# --------------------------------------------------------------------------- AgentSession without a room
def _load_demo():
    spec = importlib.util.spec_from_file_location("_lk_demo", ROOT / "examples" / "livekit_offline_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("mode", ["stt", "detector"])
def test_agent_session_commits_turn_on_our_decision(mode):
    """A real AgentSession (no room: custom AudioInput / AudioOutput, stub LLM + TTS from the demo) commits the user
    turn from our events: mode "stt" = turn_detection="stt" on END_OF_SPEECH, "detector" = AudioforgeTurnDetector."""
    demo = _load_demo()
    x = speech_silence(((1.0, 0.2), (3.0, 0.0)))

    async def body():
        async with FakeServer() as srv:
            res = await demo.run_agent_session(x, srv.url, mode=mode, speed=4.0, timeout_ms=600, tail_s=0.0)
            return res

    res = asyncio.run(body())
    users = [e for e in res["events"] if e["kind"] == "user_turn"]
    assert users and users[0]["text"] == "w0 w1", res["events"]
    assert any(e["kind"] == "agent_reply" for e in res["events"])
    # the fake server decides 600 ms after the last speech frame, plus its 0.56 s "diarizer" lag
    commit_audio_t = users[0]["audio_t"]
    assert 1.0 + 0.6 <= commit_audio_t <= 1.0 + 0.6 + 0.56 + 1.0, users[0]
    # LiveKit commits on our decision: commit minus the arrival of the server's turn_end
    m = demo.agent_metrics(res, {"onset_s": 0.0, "turn_end_s": 1.04})
    assert m["commit_ms_after_decision_arrival"] is not None and 0 <= m["commit_ms_after_decision_arrival"] < 150, m


# --------------------------------------------------------------------------- real checkpoints (slow)
@pytest.mark.skipif(os.environ.get("RUN_REAL") != "1", reason="starts audioforge.serve with the 440 MB + 450 MB "
                    "checkpoints and streams an AMI window in real time; RUN_REAL=1")
def test_real_demo_one_ami_window(tmp_path):
    wav = Path(os.environ.get("LIVEKIT_DEMO_WAV", "")) if os.environ.get("LIVEKIT_DEMO_WAV") else None
    if wav is None or not wav.exists():
        pytest.skip("set LIVEKIT_DEMO_WAV to an AMI window WAV with a sibling .json reference")
    port = int(os.environ.get("LIVEKIT_DEMO_PORT", "8793"))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    srv = subprocess.Popen([sys.executable, "-m", "audioforge.serve", "--asr", "runs/stage1_heads_pretrained.afm",
                            "--diar", "runs/nemo_sortformer_v2.afm", "--port", str(port), "--threads", "2"],
                           cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t0 = time.time()
        while time.time() - t0 < 240:
            line = srv.stdout.readline()
            if "listening" in line:
                break
            if srv.poll() is not None:
                pytest.fail("server exited")
        out = tmp_path / "demo.json"
        r = subprocess.run([sys.executable, "examples/livekit_offline_demo.py", str(wav), "--url",
                            f"ws://127.0.0.1:{port}", "--out", str(out)], cwd=ROOT, env=env, timeout=300)
        assert r.returncode == 0
        res = json.loads(out.read_text())["windows"][0]
        assert res["first_interim_ms_after_first_word_end"] is not None and res["final_transcript"]
        assert math.isfinite(res["wer_all_speakers"])
    finally:
        srv.terminate()
        srv.wait(30)


def test_hybrid_policies_cut_and_agent_end_control_ordering():
    """hybrid_dyn: its turn_end cuts the finals (END_OF_SPEECH); agent_end before the session exists is sent right
    after the config (audio position 0), a later one behind the audio already pushed."""
    from audioforge.integrations.livekit import POLICIES
    assert {"hybrid", "hybrid_silero", "hybrid_dyn"} <= set(POLICIES)
    assert AudioforgeOptions(turn_policy="hybrid_dyn").cut_policy == "hybrid_dyn"
    assert AudioforgeOptions(turn_policy="both").cut_policy == "timeout"
    m = _SpeechMapper(AudioforgeOptions(turn_policy="hybrid_silero"))
    m.on_message({"type": "partial", "t": 0.5, "text": "a"}, _wall)
    m.on_message({"type": "turn_end", "t": 3.2, "policy": "hybrid_silero", "p": 0.3, "silence_ms": 2640}, _wall)
    ev = m.on_message({"type": "final", "t": 3.2, "text": "a b", "speaker": 0}, _wall)
    assert [e.type for e in ev] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH]

    async def body():
        async with FakeServer() as srv:
            fe = AudioforgeFrontend(srv.url, turn_policy="hybrid_dyn", eot_threshold=0.9)
            assert fe.agent_end()  # no session yet: pending
            st = fe.stt().stream()
            x = speech_silence()
            frames = frames_of(x)
            task = asyncio.create_task(collect(st))
            await push_all([st], frames[:100])
            await asyncio.sleep(0.2)
            assert fe.agent_end()
            await push_all([st], frames[100:])
            st.end_input()
            await task
            await st.aclose()
            await fe.aclose()
            return srv, fe

    srv, fe = asyncio.run(body())
    assert srv.configs[0]["turn_policy"] == "hybrid_dyn" and srv.configs[0]["eot_threshold"] == 0.9
    assert [(c[1], c[2]) for c in srv.controls] == [("agent_end", 0), ("agent_end", 100 * 320)]
    assert [c[0] for c in fe.controls] == ["agent_end", "agent_end"] and fe.controls[1][1] == 2.0


def test_attach_sends_agent_end_when_the_agent_stops_speaking():
    class Sess:
        def __init__(self):
            self.cb = {}

        def on(self, name):
            def deco(f):
                self.cb[name] = f
                return f
            return deco

    fe = AudioforgeFrontend("ws://unused")
    sess = Sess()
    fe.attach(sess)
    ev = type("E", (), {})
    for old, new in (("listening", "thinking"), ("thinking", "speaking"), ("speaking", "listening")):
        e = ev()
        e.old_state, e.new_state = old, new
        sess.cb["agent_state_changed"](e)
    assert fe.controls == [("agent_end", 0.0)] and fe._pending_controls == ["agent_end"]


def test_speech_mapper_final_source_offline_holds_turn_until_offline_final():
    """Server --final-asr: with final_source="offline" the streaming final of a turn is held; FINAL_TRANSCRIPT (offline
    text) + END_OF_SPEECH are emitted on the offline final with the same t. Default "stream" ignores offline finals."""
    te = {"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1040}
    fs = {"type": "final", "t": 2.0, "text": "hello there", "speaker": 1, "source": "stream"}
    fo = {"type": "final", "t": 2.0, "text": "Hello, there.", "speaker": 1, "source": "tdt_v3", "start": 0.1,
          "end": 1.4, "latency_ms": 200.0}
    for src, want in (("stream", "hello there"), ("offline", "Hello, there.")):
        m = _SpeechMapper(AudioforgeOptions(final_source=src))
        m.on_message({"type": "partial", "t": 0.4, "text": "hello"}, _wall)
        assert m.on_message(te, _wall) == []
        ev1 = m.on_message(fs, _wall)
        ev2 = m.on_message(fo, _wall)
        evs = ev1 + ev2
        assert [e.type for e in evs] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH]
        assert evs[0].alternatives[0].text == want
        assert (ev1 == []) == (src == "offline")
        assert evs[0].alternatives[0].metadata["audioforge"]["turn_end"]["silence_ms"] == 1040
    # a server without --final-asr: finals have no source and are always used
    m = _SpeechMapper(AudioforgeOptions(final_source="offline"))
    m.on_message({"type": "partial", "t": 0.4, "text": "hi"}, _wall)
    ev = m.on_message({"type": "final", "t": 2.0, "text": "hi", "speaker": 0}, _wall)
    assert [e.type for e in ev] == [SE.FINAL_TRANSCRIPT, SE.END_OF_SPEECH]
    with pytest.raises(ValueError):
        AudioforgeOptions(final_source="bogus")
