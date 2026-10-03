"""Pipecat adapters (audioforge/integrations/pipecat.py) against a fake audioforge server that sends canned events.

The fake server speaks the audioforge.serve protocol (ready / frame / partial / turn_end / final / stats) with a
scripted, energy-based behaviour, so every pipeline run is deterministic. The real-model demo
(examples/pipecat_local_demo.py on an AMI window) runs only with RUN_REAL=1."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pipecat = pytest.importorskip("pipecat")
websockets = pytest.importorskip("websockets")

from loguru import logger  # noqa: E402
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams, VADState  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402

from audioforge.integrations.pipecat import (  # noqa: E402
    AudioforgeHub,
    AudioforgeSTTService,
    AudioforgeTurnAnalyzer,
    AudioforgeVADAnalyzer,
)

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
STATS = {"type": "stats", "rtf": 0.1, "chunk_ms_p50": 1.0, "chunk_ms_p95": 2.0, "first_partial_ms": None,
         "peak_rss_mb": 1.0}
WORDS = ["hello", "world", "again", "and", "more", "words"]


# ------------------------------------------------------------------------------------------ fake server
def energy_script(v, rms, cfg, st):
    """Per 80 ms frame v (None = the client's "end"): VAD = energy; one more word every 4 speech frames; head
    turn_end after 3 silent frames, timeout turn_end after 6 (each once per speech segment); finals cut by the
    timeout policy unless the policy is "head" (as audioforge.serve)."""
    if v is None:
        return [{"type": "final", "t": st.get("t", 0.0), "text": st.get("seg", ""), "speaker": 0}, STATS]
    t = round((v + 1) * 0.08, 3)
    st["t"] = t
    speech = rms > 0.01
    sil = 0 if speech else st.get("sil", 0) + 1
    st["sil"] = sil
    out = [{"type": "frame", "t": t, "vad": 0.95 if speech else 0.02, "eot": 0.99 if sil >= 3 else 0.1,
            "speakers": [0.9 if speech else 0.02, 0.02, 0.02, 0.02], "primary": 0 if st.get("spoke") else None}]
    pol = cfg.get("turn_policy", "timeout")
    if speech:
        st["spoke"] = st["armed_t"] = st["armed_h"] = True
        st["n"] = st.get("n", 0) + 1
        seg = " ".join(WORDS[st.get("w0", 0): st.get("w0", 0) + 1 + (st["n"] - 1) // 4])
        if seg != st.get("seg"):
            st["seg"] = seg
            out.append({"type": "partial", "t": t, "text": seg})
    else:
        cut = None
        if pol in ("head", "both") and st.get("armed_h") and sil == 3:
            st["armed_h"] = False
            out.append({"type": "turn_end", "t": t, "policy": "head", "p": 0.99, "silence_ms": 240})
            cut = "head" if pol == "head" else None
        if pol in ("timeout", "both") and st.get("armed_t") and sil == 6:
            st["armed_t"] = False
            out.append({"type": "turn_end", "t": t, "policy": "timeout", "p": None, "silence_ms": 480})
            cut = "timeout"
        if cut:
            out.append({"type": "final", "t": t, "text": st.get("seg", ""), "speaker": 0})
            st["w0"] = st.get("w0", 0) + len(st.get("seg", "").split())
            st["seg"], st["n"] = "", 0
    return out


class FakeAudioforge:
    def __init__(self, script=energy_script, batch_frames: bool = False):
        self.script, self.batch = script, batch_frames
        self.log: list[tuple] = []  # ("config", dict) / ("audio", n_bytes) / ("end", dict), in arrival order
        self.connections = 0

    async def handler(self, ws):
        self.connections += 1
        await ws.send(json.dumps({"type": "ready", "model": "fake", "chunk_ms": 160, "frame_ms": 80}))
        cfg, st, buf, v = {}, {}, np.zeros(0, np.float32), 0
        async for msg in ws:
            if isinstance(msg, bytes):
                self.log.append(("audio", len(msg)))
                buf = np.concatenate([buf, np.frombuffer(msg, "<i2").astype(np.float32) / 32768.0])
                while len(buf) >= 1280:
                    fr, buf = buf[:1280], buf[1280:]
                    msgs = self.script(v, float(np.sqrt(np.mean(fr ** 2))), cfg, st)
                    v += 1
                    if self.batch:
                        items = [{k: x for k, x in m.items() if k != "type"} for m in msgs if m["type"] == "frame"]
                        msgs = ([{"type": "frames", "items": items}] if items else []) + \
                               [m for m in msgs if m["type"] != "frame"]
                    for m in msgs:
                        await ws.send(json.dumps(m))
                continue
            d = json.loads(msg)
            self.log.append((d["type"], d))
            if d["type"] == "config":
                cfg.update(d)
            elif d["type"] == "end":
                for m in self.script(None, 0.0, cfg, st):
                    await ws.send(json.dumps(m))
                await ws.close()
                return


async def with_fake_server(fake: FakeAudioforge, body):
    from websockets.asyncio.server import serve
    async with serve(fake.handler, "127.0.0.1", 0, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        return await body(f"ws://127.0.0.1:{port}")


def tone(sec_speech: float, sec_silence: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(sec_speech * SR)) / SR
    return np.concatenate([amp * np.sin(2 * np.pi * 220 * t), np.zeros(int(sec_silence * SR))]).astype(np.float32)


def pcm_frames(x: np.ndarray, chunk: int = 320):
    b = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    return [InputAudioRawFrame(audio=b[i:i + 2 * chunk], sample_rate=SR, num_channels=1)
            for i in range(0, len(b), 2 * chunk)]


class Recorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


async def run_frames(processors, frames, *, gap_s: float = 0.02, settle_s: float = 0.3):
    """Push ``frames`` through Pipeline(processors + [Recorder]) (``gap_s`` apart), wait, EndFrame. -> frames seen
    at the end of the pipeline, in order, including the EndFrame."""
    rec = Recorder()
    worker = PipelineWorker(Pipeline(list(processors) + [rec]), params=PipelineParams(audio_in_sample_rate=SR),
                            enable_rtvi=False, cancel_on_idle_timeout=False, idle_timeout_secs=None)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(_w, _f):
        started.set()

    async def push():
        await asyncio.wait_for(started.wait(), 10)
        for f in frames:
            await worker.queue_frame(f)
            await asyncio.sleep(gap_s)
        await asyncio.sleep(settle_s)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.wait_for(asyncio.gather(runner.run(), push()), 60)
    return rec.frames


class LogCapture:
    def __init__(self, level="WARNING"):
        self.records = []
        self.level = level

    def __enter__(self):
        self.hid = logger.add(lambda m: self.records.append(m.record["level"].name + ": " + m.record["message"]),
                              level=self.level)
        return self

    def __exit__(self, *a):
        logger.remove(self.hid)


# ------------------------------------------------------------------------------------------ STT service
@pytest.mark.parametrize("batch", [False, True])
def test_stt_interim_final_metadata_and_end_flush(batch):
    fake = FakeAudioforge(batch_frames=batch)
    x = tone(1.0, 1.0)

    async def body(url):
        hub = AudioforgeHub()
        stt = AudioforgeSTTService(url=url, hub=hub)
        frames = await run_frames([stt], pcm_frames(x), gap_s=0.005)
        return hub, frames

    with LogCapture() as cap:
        hub, frames = asyncio.run(with_fake_server(fake, body))
    kinds = [type(f).__name__ for f in frames]
    interims = [f for f in frames if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in frames if isinstance(f, TranscriptionFrame)]
    # protocol: config first, then all audio, then end
    assert fake.log[0][0] == "config" and fake.log[0][1]["turn_policy"] == "timeout"
    assert fake.log[0][1]["sample_rate"] == SR and fake.log[0][1]["timeout_ms"] == 1000
    assert fake.log[-1][0] == "end"
    assert sum(n for k, n in fake.log if k == "audio") == len(x) * 2
    # frames: interims on partials, one finalized final at the turn end, audio passes through, EndFrame last
    # 13 speech frames (the 13th is half speech): one word more every 4 frames
    assert [f.text for f in interims] == ["hello", "hello world", "hello world again", "hello world again and"]
    assert [f.text for f in finals] == ["hello world again and"] and all(f.finalized for f in finals)
    assert finals[0].metadata["audioforge"] == {"t": finals[0].result["t"], "speaker": 0, "kind": "final"}
    assert interims[-1].metadata["audioforge"]["speaker"] == 0 and interims[-1].metadata["audioforge"]["kind"] == "partial"
    assert kinds.count("InputAudioRawFrame") == len(pcm_frames(x)) and kinds[-1] == "EndFrame"
    assert hub.stats is not None and hub.ready["model"] == "fake"
    assert hub.n_frames == len(x) // 1280 and len(hub.turn_ends) == 1 and hub.turn_ends[0].policy == "timeout"
    assert cap.records == []


def test_stt_end_flush_final_precedes_endframe():
    """Speech still running at the end: the tail arrives only after {"type": "end"} and must precede EndFrame."""
    fake = FakeAudioforge()
    x = tone(0.8, 0.0)

    async def body(url):
        return await run_frames([AudioforgeSTTService(url=url)], pcm_frames(x), gap_s=0.005)

    frames = asyncio.run(with_fake_server(fake, body))
    finals = [i for i, f in enumerate(frames) if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1 and frames[finals[0]].text == "hello world again"  # 10 speech frames
    assert finals[0] < len(frames) - 1 and isinstance(frames[-1], EndFrame)


def test_stt_skips_empty_text_and_requests_analyzer_policy():
    async def silent_script_body(url):
        hub = AudioforgeHub()
        AudioforgeTurnAnalyzer(hub, policy="head")
        return await run_frames([AudioforgeSTTService(url=url, hub=hub)], pcm_frames(np.zeros(SR, np.float32)))

    fake = FakeAudioforge()
    frames = asyncio.run(with_fake_server(fake, silent_script_body))
    assert fake.log[0][1]["turn_policy"] == "head"  # the analyzer's policy reaches the server
    assert not any(isinstance(f, (TranscriptionFrame, InterimTranscriptionFrame)) for f in frames)  # final "" dropped


def test_stt_handle_server_message_unit():
    stt = AudioforgeSTTService(url="ws://unused", turn_policy="both")
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    stt.push_frame = fake_push

    async def go():
        await stt.handle_server_message({"type": "frames", "items": [
            {"t": 0.08, "vad": 0.9, "eot": 0.2, "speakers": [0.1, 0.8, 0, 0], "primary": 1}]})
        await stt.handle_server_message({"type": "partial", "t": 0.16, "text": "  "})
        await stt.handle_server_message({"type": "partial", "t": 0.16, "text": "hi"})
        await stt.handle_server_message({"type": "turn_end", "t": 1.2, "policy": "head", "p": 0.99, "silence_ms": 240})
        await stt.handle_server_message({"type": "final", "t": 1.2, "text": "hi there", "speaker": 3})

    asyncio.run(go())
    assert stt.turn_policy == "both" and stt.hub.primary == 1 and stt.hub.vad == 0.9
    assert [type(f).__name__ for f in pushed] == ["InterimTranscriptionFrame", "TranscriptionFrame"]
    assert pushed[0].metadata["audioforge"]["speaker"] == 1 and pushed[1].metadata["audioforge"]["speaker"] == 3
    assert stt.hub.turn_ends[0].policy == "head" and stt.hub.turn_ends[0].p == 0.99
    assert not stt.supports_ttfs


# ------------------------------------------------------------------------------------------ VAD analyzer
def _vad_states(vad: AudioforgeVADAnalyzer, hub: AudioforgeHub, probs):
    out = []
    for i, p in enumerate(probs):
        hub.on_frame({"t": (i + 1) * 0.08, "vad": p, "eot": None, "speakers": [0] * 4, "primary": None})
        out.append(vad._run_analyzer(b"\x00\x00" * vad.num_frames_required()))
    return out


def test_vad_analyzer_uses_server_prob_with_pipecat_hysteresis():
    hub = AudioforgeHub()
    vad = AudioforgeVADAnalyzer(hub)
    vad.set_sample_rate(SR)
    assert vad.num_frames_required() == 1280 and vad.params.min_volume == 0.0
    Q, S1, SP, ST = VADState.QUIET, VADState.STARTING, VADState.SPEAKING, VADState.STOPPING
    # start_secs = stop_secs = 0.2 s -> round(0.2 / 0.08) = 2 steps of 80 ms
    states = _vad_states(vad, hub, [0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1])
    assert states == [Q, S1, SP, SP, ST, SP, ST, Q, Q]
    # a single speech step is not enough to start; below-threshold confidence (0.6 < 0.7) is silence
    assert _vad_states(vad, hub, [0.9, 0.1, 0.6, 0.6, 0.6]) == [S1, Q, Q, Q, Q]
    assert vad.steps == 14 and len(vad.lag_frames) == 14 and max(vad.lag_frames) == 0


def test_vad_analyzer_params_and_rate():
    hub = AudioforgeHub()
    vad = AudioforgeVADAnalyzer(hub, params=VADParams(confidence=0.5, start_secs=0.08, stop_secs=0.4))
    vad.set_sample_rate(8000)
    assert vad.num_frames_required() == 640
    states = _vad_states(vad, hub, [0.55] + [0.2] * 5)
    assert states[0] == VADState.SPEAKING and states[1:5] == [VADState.STOPPING] * 4 and states[5] == VADState.QUIET
    assert vad.params.min_volume == 0.0  # not set explicitly -> the loudness gate stays off
    gated = AudioforgeVADAnalyzer(hub, params=VADParams(start_secs=0.08, min_volume=0.6))
    gated.set_sample_rate(SR)
    assert _vad_states(gated, hub, [0.99, 0.99]) == [VADState.QUIET] * 2  # explicit min_volume: digital silence


def test_vad_analyzer_lags_when_server_frames_are_late():
    hub = AudioforgeHub()
    vad = AudioforgeVADAnalyzer(hub)
    vad.set_sample_rate(SR)
    buf = b"\x00\x00" * 1280
    vad._run_analyzer(buf)
    vad._run_analyzer(buf)  # no server frame yet: confidence = the last known value (0.0)
    hub.on_frame({"t": 0.08, "vad": 0.95, "speakers": [0] * 4, "primary": None})
    vad._run_analyzer(buf)
    assert vad.lag_frames == [1, 2, 2]


# ------------------------------------------------------------------------------------------ turn analyzer
def _te(hub, policy, t=1.0, p=None):
    hub.on_turn_end({"type": "turn_end", "t": t, "policy": policy, "p": p, "silence_ms": 1000})


@pytest.mark.parametrize("policy,events,complete", [
    ("timeout", ["timeout"], True), ("timeout", ["head"], False), ("head", ["head"], True),
    ("head", ["timeout"], False), ("both", ["head"], True), ("both", ["timeout"], True)])
def test_turn_analyzer_policy_filter(policy, events, complete):
    hub = AudioforgeHub()
    ta = AudioforgeTurnAnalyzer(hub, policy=policy)
    assert hub.policy == policy
    assert ta.append_audio(b"", False) == EndOfTurnState.INCOMPLETE
    for e in events:
        _te(hub, e, p=0.99 if e == "head" else None)
    want = EndOfTurnState.COMPLETE if complete else EndOfTurnState.INCOMPLETE
    assert ta.append_audio(b"", False) == want
    assert asyncio.run(ta.analyze_end_of_turn()) == (want, None)
    assert len(ta.received) == int(complete)


def _fr(hub, t, vad):
    hub.on_frame({"t": round(t, 3), "vad": vad, "eot": None, "speakers": [0] * 4, "primary": 0})


def test_turn_analyzer_latch_wait_for_silence_and_clear():
    hub = AudioforgeHub()
    ta = AudioforgeTurnAnalyzer(hub)
    _te(hub, "timeout", t=1.0)
    assert ta.append_audio(b"", True) == EndOfTurnState.INCOMPLETE  # still audible (someone else): wait, latched
    assert asyncio.run(ta.analyze_end_of_turn())[0] == EndOfTurnState.COMPLETE and ta.pending is not None
    assert ta.append_audio(b"", False) == EndOfTurnState.COMPLETE  # VAD quiet: complete, repeated until cleared
    assert ta.append_audio(b"", False) == EndOfTurnState.COMPLETE and ta.speech_triggered
    ta.clear()
    assert ta.append_audio(b"", False) == EndOfTurnState.INCOMPLETE and not ta.speech_triggered
    nogate = AudioforgeTurnAnalyzer(AudioforgeHub(), wait_for_silence=False)
    _te(nogate._hub, "timeout")
    assert nogate.append_audio(b"", True) == EndOfTurnState.COMPLETE  # no gate: complete even while audible
    with pytest.raises(ValueError):
        AudioforgeTurnAnalyzer(AudioforgeHub(), policy="smart")


def test_turn_analyzer_drops_decision_when_speech_follows_it():
    """resume_ms = 240: three server frames of speech after the decision time make it stale; speech before it,
    one-frame blips and frames of the same batch that precede the event (t <= decision) do not count / do count."""
    hub = AudioforgeHub()
    ta = AudioforgeTurnAnalyzer(hub)
    for i in range(10):
        _fr(hub, 0.08 * (i + 1), 0.9)  # speech up to 0.8 s (before the decision)
    _fr(hub, 0.88, 0.1)
    _te(hub, "timeout", t=0.88)
    assert ta.pending is not None
    _fr(hub, 0.96, 0.9)  # a blip
    _fr(hub, 1.04, 0.1)
    _fr(hub, 1.12, 0.9)
    assert ta.pending is not None and ta.append_audio(b"", False) == EndOfTurnState.COMPLETE
    _fr(hub, 1.20, 0.9)  # third speech frame after the decision -> stale
    assert ta.pending is None and [e.t for e in ta.discarded] == [0.88]
    assert ta.append_audio(b"", False) == EndOfTurnState.INCOMPLETE
    # frames with t > decision that arrived in the same batch before the turn_end message are counted too
    for t in (1.28, 1.36, 1.44):
        _fr(hub, t, 0.9)
    _te(hub, "timeout", t=1.2)
    assert ta.pending is None and len(ta.discarded) == 2
    keep = AudioforgeTurnAnalyzer(AudioforgeHub(), resume_ms=0)
    _te(keep._hub, "timeout", t=0.0)
    for i in range(5):
        _fr(keep._hub, 0.08 * (i + 1), 0.9)
    assert keep.pending is not None  # rule disabled


# ------------------------------------------------------------------------------------------ whole local pipeline
def _demo():
    sys.path.insert(0, str(ROOT))
    import examples.pipecat_local_demo as demo
    return demo


@pytest.mark.parametrize("policy,server_silence_frames", [("timeout", 6), ("head", 3)])
def test_local_pipeline_turn_end_via_pipecat_strategy(policy, server_silence_frames):
    """WAV transport -> STT -> aggregator (our VAD + TurnAnalyzerUserTurnStopStrategy) -> mock, at 2x speed."""
    demo = _demo()
    fake = FakeAudioforge()
    x = tone(1.2, 1.5)

    async def body(url):
        return await demo.run_pipeline(x, url=url, policy=policy, pad_s=0.5, speed=2.0)

    with LogCapture() as cap:
        raw = asyncio.run(with_fake_server(fake, body))
    assert fake.log[0][1]["turn_policy"] == policy
    assert raw["finished"] == ["EndFrame"] and raw["pipeline_errors"] == [] and raw["violations"] == []
    assert cap.records == []
    assert len(raw["starts"]) == 1 and len(raw["decisions"]) == 1
    assert raw["stopped"][0][1] == "TurnAnalyzerUserTurnStopStrategy"
    assert [c[1] for c in raw["contexts"]] == ["hello world again and"]
    assert [f["text"] for f in raw["finals"]] == ["hello world again and"]
    # the decision follows the server's turn_end (frame 15 + server_silence_frames) and precedes the stream end
    at = lambda p: (p - raw["t0"]) * raw["speed"]  # noqa: E731
    te = raw["server_turn_ends"][0]
    assert te["policy"] == policy and te["t"] == pytest.approx((15 + server_silence_frames) * 0.08)
    assert te["t"] <= at(raw["decisions"][0]) < te["t"] + 0.5
    assert [k for k, _ in raw["vad"]] == ["start", "stop"]


@pytest.mark.parametrize("inject_v,kept", [(23, True), (12, False)])
def test_local_pipeline_turn_end_during_speech(inject_v, kept):
    """A turn_end whose decision time is followed by < 240 ms of speech is held while Pipecat's VAD still hears
    speech and ends the turn at the VAD stop (no LLM inference before that); one followed by more speech is stale
    and dropped (nothing ends the turn)."""
    demo = _demo()
    t_dec = round((inject_v + 1) * 0.08, 3)

    def script(v, rms, cfg, st):
        out = [m for m in energy_script(v, rms, cfg, st) if m["type"] not in ("turn_end", "final")]
        if v == inject_v:
            out += [{"type": "turn_end", "t": t_dec, "policy": "timeout", "p": None, "silence_ms": 1000},
                    {"type": "final", "t": t_dec, "text": "hello world", "speaker": 0}]
        return out if v is not None else out + [{"type": "final", "t": 3.0, "text": "", "speaker": 0}]

    fake = FakeAudioforge(script=script)
    x = tone(2.0, 1.0)  # speech frames 0..24 (t <= 2.0 s)

    async def body(url):
        return await demo.run_pipeline(x, url=url, policy="timeout", pad_s=0.3, speed=2.0)

    with LogCapture() as cap:
        raw = asyncio.run(with_fake_server(fake, body))
    assert raw["violations"] == [] and cap.records == []
    at = lambda p: (p - raw["t0"]) * raw["speed"]  # noqa: E731
    vad_stop = [at(p) for k, p in raw["vad"] if k == "stop"]
    if kept:
        assert len(raw["decisions"]) == 1 and vad_stop and raw["turn_discarded"] == []
        assert at(raw["decisions"][0]) >= vad_stop[0] - 0.05 > 2.0  # held until the VAD stop after the speech
        assert len(raw["contexts"]) == 1 and raw["contexts"][0][0] >= raw["decisions"][0] - 0.05  # no early answer
    else:
        assert raw["decisions"] == [] and [e["t"] for e in raw["turn_discarded"]] == [t_dec]
        # the turn is still open at the end: Pipecat's aggregator flushes it after pushing the EndFrame
        assert raw["order_notes"] == ["mock: session-end LLMContextFrame after EndFrame (Pipecat aggregator)"]


def test_score_classifies_decisions():
    demo = _demo()
    t0 = 100.0
    raw = {"t0": t0, "speed": 1.0, "stream_end_perf": t0 + 20, "total_s": 21.0,
           "decisions": [t0 + 2.0, t0 + 5.5, t0 + 11.3, t0 + 30.0], "contexts": [(t0 + 2.0, "x")],
           "server_turn_ends": [
               {"t": 11.0, "policy": "timeout", "p": None, "silence_ms": 1000, "perf": t0 + 11.1, "wall": 0}],
           "finals": [{"text": "a b c"}], "interims": 3, "vad": [], "frame_lag_ms": [100, 120],
           "vad_lag_frames": [1, 2], "max_push_late_ms": 1.0, "server_stats": None, "stopped": [], "watchdog": []}
    ref = {"primary_intervals": [[1.0, 4.0], [6.0, 10.0]], "turn_end_s": 10.0, "onset_s": 1.0, "all_text": "a b d"}
    s = demo.score(raw, ref)
    assert [d["kind"] for d in s["decisions"]] == ["interruption", "interruption", "after_end"]  # 30 s: after end
    assert s["dead_air_ms"] == 1300 and s["interruptions"] == 2 and s["early"] == 0
    assert s["server_turn_end_t_minus_true_end_ms"] == 1000 and s["pipecat_after_server_turn_end_ms"] == 200
    assert s["wer_all_speakers"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["answer_interruptions"] == 1
    raw["decisions"] = [t0 + 4.8, t0 - 0.5]  # 4.8: primary resumes 1.2 s later (early); -0.5: before onset - 1 s
    s = demo.score(raw, ref)
    assert s["early"] == 1 and s["pre_onset"] == 1 and s["interruptions"] == 0
    assert s["dead_air_ms"] is None and s["no_decision_within_ms"] == 11000


# ------------------------------------------------------------------------------------------ real models (RUN_REAL=1)
@pytest.mark.real
@pytest.mark.skipif(os.environ.get("RUN_REAL") != "1", reason="real models + AMI audio; set RUN_REAL=1")
def test_real_demo_on_one_ami_window(tmp_path):
    """Serve the shipped stack (audioforge-serve: --mode single on the 115M v0.4; or use AUDIOFORGE_URL), prepare one
    AMI dev window and run the demo in real time."""
    demo = _demo()
    names = demo.prepare(1, tmp_path)
    url, proc = os.environ.get("AUDIOFORGE_URL"), None
    if not url:
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = open(tmp_path / "server.log", "w")
        proc = subprocess.Popen([sys.executable, "-c", "import sys; from audioforge.launch import serve_main; "
                                 "serve_main(sys.argv[1:])", "--mode", "single", "--port", str(port), "--threads", "2"],
                                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log,
                                stderr=subprocess.STDOUT)
        url = f"ws://127.0.0.1:{port}"
        for _ in range(300):
            if "listening" in (tmp_path / "server.log").read_text() or proc.poll() is not None:
                break
            time.sleep(1)
        assert "listening" in (tmp_path / "server.log").read_text(), (tmp_path / "server.log").read_text()[-2000:]
    try:
        out = tmp_path / "res.json"
        with LogCapture() as cap:
            res = demo.main(["run", str(tmp_path / f"{names[0]}.wav"), "--url", url, "--policy", "timeout",
                             "--out", str(out), "--log-level", "WARNING"])
        r = res[0]
        assert r["pipeline_finished"] == ["EndFrame"] and r["violations"] == [] and r["pipeline_errors"] == []
        assert r["n_finals"] >= 1 and r["wer_all_speakers"] < 0.8
        assert not [c for c in cap.records if c.startswith("ERROR")]
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(30)


# ------------------------------------------------------------------------------------------ enrollment (--enroll)
def test_stt_enroll_after_agent_sends_agent_end_on_bot_stopped_speaking():
    from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
    from websockets.protocol import State

    class FakeWS:
        state = State.OPEN

        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(json.loads(m))

    async def go(mode):
        stt = AudioforgeSTTService(url="ws://unused", enroll=mode)
        ws = stt._websocket = FakeWS()
        try:  # the hook runs before the parent's process_frame (which may reject frames before StartFrame)
            await stt.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await stt.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        except Exception:  # noqa: BLE001
            pass
        ok = await stt.enroll()
        await stt.handle_server_message({"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80,
                                         "diar_config": "low_latency_032", "column_lag_ms": 160.0,
                                         "enroll": mode or "dominant", "enrolled": False, "primary_column": None})
        await stt.handle_server_message({"type": "enrolled", "t": 3.9, "column": 2})
        return stt, ws, ok

    stt, ws, ok = asyncio.run(go("after_agent"))
    assert [m["type"] for m in ws.sent] == ["agent_end"] and ok is False and stt.hub.enroll_sent == 1
    assert stt.hub.enrolled_column == 2
    stt, ws, ok = asyncio.run(go("after_agent_arm"))  # the arming-only server mode uses the same message
    assert [m["type"] for m in ws.sent] == ["agent_end"] and ok is False and stt.hub.enroll_sent == 1
    stt, ws, ok = asyncio.run(go("explicit"))
    assert [m["type"] for m in ws.sent] == ["enroll"] and ok is True
    stt, ws, ok = asyncio.run(go(None))
    assert ws.sent == [] and ok is False
    with pytest.raises(ValueError):
        AudioforgeSTTService(url="ws://unused", enroll="nope")


def test_hybrid_policies_accepted_and_eot_threshold_left_to_the_server():
    from audioforge.integrations.pipecat import POLICIES, AudioforgeTurnAnalyzer, TurnEnd
    assert {"hybrid", "hybrid_silero", "hybrid_dyn"} <= set(POLICIES)
    hub = AudioforgeHub()
    ta = AudioforgeTurnAnalyzer(hub, policy="hybrid_dyn")
    assert hub.policy == "hybrid_dyn"
    stt = AudioforgeSTTService(url="ws://unused", hub=hub)
    assert stt.turn_policy == "hybrid_dyn" and stt._eot_threshold is None
    ev = TurnEnd(2.0, "hybrid_dyn", 0.5, 2400, 0.0, 0.0)
    ta._on_turn_end(ev)
    assert ta.pending is ev  # the analyzer latches its own policy's event
    ta.clear()
    ta._on_turn_end(TurnEnd(2.5, "timeout", None, 1000, 0.0, 0.0))
    assert ta.pending is None


def test_agent_end_before_connect_is_queued_for_the_config():
    from pipecat.frames.frames import BotStoppedSpeakingFrame

    async def go():
        stt = AudioforgeSTTService(url="ws://unused", enroll="after_agent_arm")
        try:
            await stt.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        except Exception:  # noqa: BLE001 - the parent may reject frames before StartFrame; the hook ran first
            pass
        return stt

    stt = asyncio.run(go())
    assert stt._pending_controls == ["agent_end"]


@pytest.mark.parametrize("final_source", ["stream", "offline"])
def test_stt_final_source_with_final_asr_server(final_source):
    """Server --final-asr: each turn has a streaming and an offline final. final_source picks
    which one becomes the TranscriptionFrame; with "offline" the turn analyzer completes only once the turn's offline
    final has arrived. A server without the flag (no "source") is unaffected."""
    hub = AudioforgeHub()
    stt = AudioforgeSTTService(url="ws://unused", hub=hub, final_source=final_source)
    turn = AudioforgeTurnAnalyzer(hub, policy="timeout", wait_for_silence=False)
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    stt.push_frame = fake_push
    states = []

    async def go():
        await stt.handle_server_message({"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80,
                                         "diar_config": "x", "column_lag_ms": 160, "final_asr": "tdt_v3"})
        await stt.handle_server_message({"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1000})
        await stt.handle_server_message({"type": "final", "t": 2.0, "text": "hello there", "speaker": 0,
                                         "source": "stream"})
        states.append(turn.append_audio(b"\x00\x00" * 320, False))
        await stt.handle_server_message({"type": "final", "t": 2.0, "text": "Hello, there.", "speaker": 0,
                                         "source": "tdt_v3", "start": 0.2, "end": 1.5, "latency_ms": 180.0})
        states.append(turn.append_audio(b"\x00\x00" * 320, False))

    asyncio.run(go())
    texts = [f.text for f in pushed if type(f).__name__ == "TranscriptionFrame"]
    assert texts == (["hello there"] if final_source == "stream" else ["Hello, there."])
    assert len(hub.finals) == 2 and hub.offline_final_t == {2.0}
    C, I = EndOfTurnState.COMPLETE, EndOfTurnState.INCOMPLETE
    assert states == ([C, C] if final_source == "stream" else [I, C])
    with pytest.raises(ValueError):
        AudioforgeSTTService(url="ws://unused", final_source="bogus")


# ------------------------------------------------------------------------------------------ turn_end_hint (eager EOT)
def test_stt_hint_events_off_by_default_and_eager_frames_when_on():
    from pipecat.frames.frames import EagerEndOfTurnCancelFrame, EagerTranscriptionFrame

    for on in (False, True):
        stt = AudioforgeSTTService(url="ws://unused", turn_policy="vad_head", turn_hints=on)
        pushed = []

        async def fake_push(frame, direction=FrameDirection.DOWNSTREAM, _p=pushed):
            _p.append(frame)

        stt.push_frame = fake_push

        async def go(stt=stt):
            await stt.handle_server_message({"type": "turn_end_hint", "t": 1.2, "p": 0.9, "kind": "hint",
                                             "text": "hi there"})
            await stt.handle_server_message({"type": "turn_end_hint_cancel", "t": 1.4})
            await stt.handle_server_message({"type": "turn_end_hint_cancel", "t": 1.5})  # nothing outstanding
            await stt.handle_server_message({"type": "turn_end_hint", "t": 2.0, "p": 0.95, "kind": "hint",
                                             "text": "hi there you"})
            await stt.handle_server_message({"type": "turn_end", "t": 2.16, "policy": "vad_head", "p": 0.99,
                                             "silence_ms": 160, "hinted_at": 2.0})

        with LogCapture() as cap:
            asyncio.run(go())
        assert cap.records == []  # known message types, no warnings
        h = stt.hub.hints
        assert [(x.t, x.outcome) for x in h] == [(1.2, "cancelled"), (2.0, "confirmed")]
        assert stt.hub.turn_ends[0].hinted_at == 2.0 and stt.hub.pending_hint is None
        kinds = [type(f) for f in pushed]
        if on:
            assert kinds == [EagerTranscriptionFrame, EagerEndOfTurnCancelFrame, EagerTranscriptionFrame]
            assert pushed[0].text == "hi there" and pushed[0].metadata["audioforge"]["kind"] == "hint"
        else:
            assert kinds == []


def hint_script(v, rms, cfg, st):
    """energy_script (timeout policy: turn_end after 6 silent frames) plus the hint: turn_end_hint on the first
    silent frame of a speech segment (text = the segment so far, or ``st["hint_text"]``), turn_end_hint_cancel after
    2 speech frames following an outstanding hint; turn_end carries hinted_at."""
    out = energy_script(v, rms, cfg, st)
    if v is None:
        return out
    t = round((v + 1) * 0.08, 3)
    speech = rms > 0.01
    st["run"] = st.get("run", 0) + 1 if speech else 0
    if speech:
        st["hint_armed"] = st.get("hint_armed", True) or st["run"] >= 2
        if st.get("hint") is not None and st["run"] >= 2:
            out.append({"type": "turn_end_hint_cancel", "t": t})
            st["hint"], st["hint_armed"] = None, True
    elif st.get("sil") == 1 and st.get("hint_armed", True) and st.get("hint") is None and st.get("seg"):
        st["hint"], st["hint_armed"] = t, False
        out.append({"type": "turn_end_hint", "t": t, "p": 0.9, "kind": "hint",
                    "text": st.get("hint_text", st["seg"])})
    for m in out:
        if m["type"] == "turn_end":
            m["hinted_at"], st["hint"] = st.get("hint"), None
    return out


def _hint_run(x, *, turn_hints, llm_ms=100, script=hint_script):
    demo = _demo()
    fake = FakeAudioforge(script=script)

    async def body(url):
        return await demo.run_pipeline(x, url=url, policy="timeout", pad_s=0.5, speed=2.0, turn_hints=turn_hints,
                                       llm_ms=llm_ms)

    with LogCapture() as cap:
        raw = asyncio.run(with_fake_server(fake, body))
    assert raw["finished"] == ["EndFrame"] and raw["pipeline_errors"] == [] and raw["violations"] == []
    assert [r for r in cap.records if "differs from the recommended" not in r] == []
    at = lambda p: (p - raw["t0"]) * raw["speed"]  # noqa: E731
    return raw, at


@pytest.mark.parametrize("turn_hints", [False, True])
def test_local_pipeline_hint_prepares_the_reply_and_releases_it_at_the_turn_end(turn_hints):
    """1.2 s speech: hint at 1.28 s, timeout turn_end at 1.68 s; the mock LLM needs 100 ms wall (0.2 s audio at 2x).
    Without hints the reply starts ~0.2 s after the decision; with hints it was prepared from the hint (speculative
    inference, held by the SpeculationGate) and is released at the decision, with no second inference."""
    raw, at = _hint_run(tone(1.2, 1.5), turn_hints=turn_hints)
    te = raw["server_turn_ends"][0]
    assert te["t"] == pytest.approx(1.68) and te["hinted_at"] == pytest.approx(1.28)
    assert len(raw["responses"]) == 1 and len(raw["decisions"]) == 1
    dec, resp = at(raw["decisions"][0]), at(raw["responses"][0])
    runs = [sp for _, sp, _ in raw["llm_runs"]]
    if turn_hints:
        assert runs == [True] and len(raw["spec_kept"]) == 1 and raw["spec_discarded"] == []
        assert at(raw["llm_runs"][0][0]) < te["t"] and resp == pytest.approx(dec, abs=0.12)
    else:
        assert runs == [False] and raw["speculated"] == []
        assert resp - dec == pytest.approx(0.2, abs=0.12)
    assert resp >= te["t"]  # never before the server's turn_end
    assert [h["outcome"] for h in raw["hints"]] == ["confirmed"]


def test_local_pipeline_hint_cancelled_by_resumed_speech():
    """speech 1.0 s, pause 0.24 s, speech 0.8 s: the pause's hint starts a speculative inference that the cancel
    discards (nothing is spoken); the real end gets a new hint whose reply is released at the turn_end: one reply."""
    x = np.concatenate([tone(1.0, 0.24), tone(0.8, 1.5)])
    raw, at = _hint_run(x, turn_hints=True, llm_ms=300)
    assert [h["outcome"] for h in raw["hints"]] == ["cancelled", "confirmed"]
    assert [d[2] for d in raw["spec_discarded"]] == ["cancel"] and len(raw["spec_kept"]) == 1
    assert [sp for _, sp, _ in raw["llm_runs"]] == [True, True]
    assert len(raw["responses"]) == 1 and at(raw["responses"][0]) >= raw["server_turn_ends"][0]["t"]
    assert len(raw["decisions"]) == 1


def test_local_pipeline_hint_text_mismatch_answers_the_final():
    """The final transcript differs from the hinted text: the held reply is discarded and inference runs again on
    the final text (the reply comes late, but it answers what was said)."""
    def script(v, rms, cfg, st):
        st["hint_text"] = "hello"
        return hint_script(v, rms, cfg, st)

    raw, at = _hint_run(tone(1.2, 1.5), turn_hints=True, script=script)
    assert [d[2] for d in raw["spec_discarded"]] == ["mismatch"] and raw["spec_kept"] == []
    assert [sp for _, sp, _ in raw["llm_runs"]] == [True, False]
    assert raw["llm_runs"][1][2] == "hello world again and" and len(raw["responses"]) == 1


@pytest.mark.parametrize("final_text", ["fast", "slow"])
def test_stt_final_text_with_dual_rate_server(final_text):
    """Server --final-chunk-ms: each turn has a final_fast (160 ms pass, at the turn end) and
    a final with source "slow". final_text="fast" (default) pushes final_fast and completes at once; "slow" pushes
    the slow text and completes only once the turn's slow final has arrived."""
    hub = AudioforgeHub()
    stt = AudioforgeSTTService(url="ws://unused", hub=hub, final_text=final_text)
    turn = AudioforgeTurnAnalyzer(hub, policy="timeout", wait_for_silence=False)
    pushed = []

    async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    stt.push_frame = fake_push
    states = []

    async def go():
        await stt.handle_server_message({"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80,
                                         "diar_config": "x", "column_lag_ms": 160, "final_asr": "slow",
                                         "final_chunk_ms": 1120})
        await stt.handle_server_message({"type": "turn_end", "t": 2.0, "policy": "timeout", "p": None, "silence_ms": 1000})
        await stt.handle_server_message({"type": "final_fast", "t": 2.0, "text": "hello their", "speaker": 0})
        states.append(turn.append_audio(b"\x00\x00" * 320, False))
        await stt.handle_server_message({"type": "final", "t": 2.0, "text": "hello there", "speaker": 0,
                                         "source": "slow", "pass": "slow", "start": 0.0, "end": 1.92,
                                         "latency_ms": 4.0})
        states.append(turn.append_audio(b"\x00\x00" * 320, False))

    asyncio.run(go())
    texts = [f.text for f in pushed if type(f).__name__ == "TranscriptionFrame"]
    assert texts == (["hello their"] if final_text == "fast" else ["hello there"])
    C, I = EndOfTurnState.COMPLETE, EndOfTurnState.INCOMPLETE
    assert states == ([C, C] if final_text == "fast" else [I, C])
    with pytest.raises(ValueError):
        AudioforgeSTTService(url="ws://unused", final_text="bogus")
    with pytest.raises(ValueError):
        AudioforgeSTTService(url="ws://unused", final_text="slow", final_source="offline")
