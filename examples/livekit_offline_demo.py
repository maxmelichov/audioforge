"""Offline LiveKit test harness: AMI dev windows through the audioforge LiveKit plugin, no LiveKit server.

    PYTHONPATH=. .venv/bin/python examples/livekit_offline_demo.py WIN.wav [WIN.wav ...] --url ws://127.0.0.1:8791 \
        [--speed 1] [--tail 2.5] [--agent-session none|stt|detector|both] [--out results.json] [--quiet]

Needs a running ``python -m audioforge.serve`` (see docs/CONFIGURATION.md). Each WAV may have a sibling
``.json`` reference (onset_s / turn_end_s / first_word_end_s of the primary's turn, all_text = every speaker's words
in the window; ``scripts``-style prep in research/archive/INTEGRATION.md).

Stream mode (always): 20 ms ``rtc.AudioFrame``s (16 kHz mono int16, built from PCM, no room) are pushed at ``--speed``
x real time into ``AudioforgeSTT.stream()`` and ``AudioforgeVAD.stream()`` of one ``AudioforgeFrontend`` (one server
session), followed by ``--tail`` s of digital silence. Every SpeechEvent / VADEvent is printed with the audio time at
which it arrived. Per window:

* first INTERIM latency: arrival of the first INTERIM_TRANSCRIPT minus the moment the audio up to the end of the
  window's first word had been pushed;
* STT END_OF_SPEECH (= the server's timeout turn_end on the diarizer's primary track): ``early`` = ENDs that arrived
  inside the primary's turn (onset_s < arrival < turn_end_s, i.e. the turn was cut), ``dead_air_ms`` = arrival of the
  first END at/after the true turn end minus the moment the audio at turn_end_s had been pushed. Same for the VAD's
  END_OF_SPEECH (any-speaker VAD, 560 ms hangover) for comparison;
* WER of the joined FINAL_TRANSCRIPTs against all speakers' words in the window (``audioforge.teachers.normalize_text``).

Agent-session mode (``--agent-session``): the same audio drives a real ``AgentSession`` without a room: a custom
``io.AudioInput`` (the WAV at real time) and ``io.AudioOutput`` (discards audio), a stub LLM that echoes the user
turn, a stub TTS (a tone). ``stt``: ``turn_detection="stt"`` (LiveKit commits the user turn on our END_OF_SPEECH);
``detector``: ``turn_detection=AudioforgeTurnDetector`` (LiveKit's audio turn-detector protocol; VAD END_OF_SPEECH +
our prediction). Reported: when LiveKit committed each user turn (``conversation_item_added``, role user) and when
the agent's reply started, relative to the true turn end.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import rtc  # noqa: E402
from livekit.agents import Agent, llm, stt, tts, vad  # noqa: E402
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS  # noqa: E402
from livekit.agents.voice import io  # noqa: E402

from audioforge.integrations.livekit import POLICIES, AudioforgeFrontend  # noqa: E402

SR = 16000
FRAME_S = 0.02


def frames_of(x: np.ndarray, sr: int = SR, frame_s: float = FRAME_S) -> list[rtc.AudioFrame]:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    n = int(sr * frame_s)
    return [rtc.AudioFrame(pcm[i:i + n].tobytes(), sr, 1, len(pcm[i:i + n])) for i in range(0, len(pcm), n)]


class Clock:
    """Push wall times of each 20 ms frame: audio time <-> wall clock."""

    def __init__(self):
        self.walls: list[float] = []

    def mark(self):
        self.walls.append(time.time())

    def wall_at(self, t_audio: float) -> float:
        """Wall time at which the audio up to t_audio had been pushed."""
        i = max(0, min(len(self.walls) - 1, int(np.ceil(t_audio / FRAME_S - 1e-9)) - 1))
        return self.walls[i]

    def audio_at(self, wall: float) -> float:
        """Audio seconds pushed by wall time ``wall``."""
        return bisect.bisect_right(self.walls, wall) * FRAME_S


# --------------------------------------------------------------------------- stream mode
async def run_streams(x: np.ndarray, url: str, speed: float = 1.0, tail_s: float = 2.5, timeout_ms: int = 1000,
                      policy: str = "timeout", quiet: bool = False, eot_threshold: float | None = None,
                      agent_end_s: float | None = None) -> dict:
    """``agent_end_s``: audio time of the stand-in TTS-end event: ``fe.agent_end()`` is called once every frame
    before that time has been pushed (server --enroll after_agent[_arm])."""
    fe = AudioforgeFrontend(url, turn_policy=policy, timeout_ms=timeout_ms, eot_threshold=eot_threshold)
    ss, vs = fe.stt().stream(), fe.vad().stream()
    audio = np.concatenate([x, np.zeros(int(tail_s * SR), np.float32)])
    frames, clock = frames_of(audio), Clock()
    s_ev, v_ev = [], []

    def show(w, src, typ, body=""):
        if not quiet:
            print(f"  {clock.audio_at(w):7.2f}s  {src:3s} {typ:20s} {body}", flush=True)

    async def read_stt():
        async for ev in ss:
            w = time.time()
            s_ev.append((w, ev))
            a = ev.alternatives[0] if ev.alternatives else None
            body = ""
            if a is not None:
                body = f"{a.text!r}" + (f" speaker={a.speaker_id}" if a.speaker_id else "")
            elif ev.speech_end_time:
                body = f"speech_end={clock.audio_at(ev.speech_end_time):.2f}s"
            elif ev.speech_start_time:
                body = f"speech_start={clock.audio_at(ev.speech_start_time):.2f}s"
            elif ev.recognition_usage:
                body = f"audio={ev.recognition_usage.audio_duration:.2f}s"
            show(w, "STT", ev.type.name, body)

    async def read_vad():
        async for ev in vs:
            w = time.time()
            v_ev.append((w, ev))
            if ev.type != vad.VADEventType.INFERENCE_DONE:
                show(w, "VAD", ev.type.name, f"speech={ev.speech_duration:.2f}s silence={ev.silence_duration:.2f}s "
                                             f"lag={ev.inference_duration * 1000:.0f}ms frames={len(ev.frames)}")

    tasks = [asyncio.create_task(read_stt()), asyncio.create_task(read_vad())]
    t0 = time.time()
    agent_end_t = None
    for i, f in enumerate(frames):
        if speed > 0:
            d = t0 + (i + 1) * FRAME_S / speed - time.time()
            if d > 0:
                await asyncio.sleep(d)
        if agent_end_s is not None and agent_end_t is None and i * FRAME_S >= agent_end_s - 1e-9:
            fe.agent_end()
            agent_end_t = round(i * FRAME_S, 3)
        ss.push_frame(f)
        vs.push_frame(f)
        clock.mark()
    ss.end_input()
    vs.end_input()
    await asyncio.wait_for(asyncio.gather(*tasks), 60)
    await ss.aclose()
    await vs.aclose()
    await fe.aclose()
    return {"stt": s_ev, "vad": v_ev, "clock": clock, "stats": fe.last_stats, "audio_s": len(x) / SR,
            "tail_s": tail_s, "sessions": fe.sessions_opened, "turn_ends": fe.turn_ends,
            "cut_policy": fe.opts.cut_policy, "agent_end_t": agent_end_t, "controls": fe.controls}


def stream_metrics(res: dict, ref: dict) -> dict:
    from audioforge.metrics import wer
    from audioforge.teachers import normalize_text
    clock: Clock = res["clock"]
    SE, VE = stt.SpeechEventType, vad.VADEventType
    out = {"audio_s": round(res["audio_s"], 2), "server_sessions": res["sessions"]}
    first = next(((w, e) for w, e in res["stt"] if e.type == SE.INTERIM_TRANSCRIPT), None)
    out["first_interim_text"] = first[1].alternatives[0].text if first else None
    out["first_interim_audio_t"] = round(clock.audio_at(first[0]), 2) if first else None
    fwe = ref.get("first_word_end_s")
    out["first_interim_ms_after_first_word_end"] = (round((first[0] - clock.wall_at(fwe)) * 1000)
                                                    if first and fwe is not None else None)
    # server + transport lag of that INTERIM: arrival minus the push of the audio its text covers (SpeechData.end_time)
    out["first_interim_lag_ms"] = (round((first[0] - clock.wall_at(first[1].alternatives[0].end_time)) * 1000)
                                   if first else None)
    onset, end = ref.get("onset_s"), ref.get("turn_end_s")
    # STT END_OF_SPEECH = the server's cutting turn_end. Classified by its *decision time* t (load-independent: the
    # audio time at which the deciding data was complete), then split into decision delay (algorithm: timeout +
    # diarizer lag) and delivery lag (compute backlog + transport; arrival minus the push of audio t).
    tes = [te for te in res["turn_ends"] if te["policy"] == res["cut_policy"]]
    eos = [w for w, e in res["stt"] if e.type == SE.END_OF_SPEECH]
    out["stt_decisions_t"] = [te["t"] for te in tes]
    out["stt_end_of_speech_audio_t"] = [round(clock.audio_at(w), 2) for w in eos]
    vads = [(w, e) for w, e in res["vad"] if e.type == VE.END_OF_SPEECH]
    out["vad_end_of_speech_t"] = [round(e.timestamp, 2) for _, e in vads]
    if end is not None:
        out["stt_early_cuts"] = sum(1 for te in tes if onset is not None and onset < te["t"] < end)
        after = [te for te in tes if te["t"] >= end - 1e-6]
        if after:
            te = after[0]
            w = next((w for w in eos if w >= te["arrival"] - 1e-3), None)
            out["stt_decision_after_end_ms"] = round((te["t"] - end) * 1000)
            out["stt_primary_silence_ms"] = te["silence_ms"]
            out["stt_delivery_lag_ms"] = round((w - clock.wall_at(te["t"])) * 1000) if w else None
            out["stt_dead_air_ms"] = round((w - clock.wall_at(end)) * 1000) if w else None
            out["stt_end_in_tail"] = bool(te["t"] > res["audio_s"])
        out["vad_early_ends"] = sum(1 for _, e in vads if onset is not None and onset < e.timestamp < end)
        va = [(w, e) for w, e in vads if e.timestamp >= end - 1e-6]
        if va:
            out["vad_decision_after_end_ms"] = round((va[0][1].timestamp - end) * 1000)
            out["vad_dead_air_ms"] = round((va[0][0] - clock.wall_at(end)) * 1000)
    lags = [e.inference_duration * 1000 for _, e in res["vad"] if e.type == VE.INFERENCE_DONE]
    if lags:
        out["frame_lag_ms_p50"], out["frame_lag_ms_max"] = round(float(np.median(lags))), round(max(lags))
    finals = [e.alternatives[0] for _, e in res["stt"] if e.type == SE.FINAL_TRANSCRIPT]
    hyp = " ".join(a.text for a in finals).strip()
    out["final_transcript"] = hyp
    out["final_speakers"] = [a.speaker_id for a in finals]
    if ref.get("all_text"):
        out["wer_all_speakers"] = round(float(wer([normalize_text(ref["all_text"])], [normalize_text(hyp)])), 4)
    st = res.get("stats") or {}
    out["agent_end_t"], out["controls"] = res.get("agent_end_t"), res.get("controls")
    out["server_stats"] = st
    out["server_rtf"] = st.get("rtf")
    if "backlog_ms_max" in st:  # server started with --debug-fields
        out["server_backlog_ms_max"] = st["backlog_ms_max"]
    return out


# --------------------------------------------------------------------------- agent-session mode (no room)
class WavAudioInput(io.AudioInput):
    """The WAV as a live microphone: 20 ms frames at ``speed`` x real time, then blocks (the session closes it)."""

    def __init__(self, x: np.ndarray, speed: float = 1.0, at: tuple | None = None):
        """``at`` = (audio seconds, callback): the callback runs once all frames before that time were handed out."""
        super().__init__(label="wav")
        self.frames, self.speed, self.i = frames_of(x), speed, 0
        self.clock, self.t0 = Clock(), None
        self.done = asyncio.Event()
        self.at = at
        self.at_t: float | None = None

    async def __anext__(self) -> rtc.AudioFrame:
        if self.t0 is None:
            self.t0 = time.time()
        if self.i >= len(self.frames):
            self.done.set()
            await asyncio.sleep(3600)
            raise StopAsyncIteration
        if self.speed > 0:
            d = self.t0 + (self.i + 1) * FRAME_S / self.speed - time.time()
            if d > 0:
                await asyncio.sleep(d)
        if self.at is not None and self.at_t is None and self.i * FRAME_S >= self.at[0] - 1e-9:
            self.at[1]()
            self.at_t = round(self.i * FRAME_S, 3)
        f = self.frames[self.i]
        self.i += 1
        self.clock.mark()
        return f


class NullAudioOutput(io.AudioOutput):
    """Discards the agent's audio and reports each segment as played out."""

    def __init__(self):
        super().__init__(label="null", capabilities=io.AudioOutputCapabilities(pause=False))
        self._dur, self._open = 0.0, False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if not self._open:
            self.on_playback_started(created_at=time.time())
        await super().capture_frame(frame)
        self._dur += frame.duration
        self._open = True

    def flush(self) -> None:
        super().flush()
        if self._open:
            d, self._dur, self._open = self._dur, 0.0, False
            asyncio.get_running_loop().call_soon(lambda: self.on_playback_finished(playback_position=d,
                                                                                   interrupted=False))

    def clear_buffer(self) -> None:
        if self._open:
            d, self._dur, self._open = self._dur, 0.0, False
            self.on_playback_finished(playback_position=d, interrupted=True)


class StubLLM(llm.LLM):
    """Function-based LLM stand-in: replies "You said: <last user turn>" (no network)."""

    @property
    def model(self) -> str:
        return "stub-echo"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kw) -> llm.LLMStream:
        return _StubLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _StubLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        last = ""
        for it in reversed(self._chat_ctx.items):
            if getattr(it, "role", None) == "user":
                last = it.text_content or ""
                break
        self._event_ch.send_nowait(llm.ChatChunk(id="stub", delta=llm.ChoiceDelta(role="assistant",
                                                                                   content=f"You said: {last}")))


class StubTTS(tts.TTS):
    """Non-streaming TTS stand-in: a 440 Hz tone, 40 ms per character (max 2 s), 24 kHz."""

    def __init__(self):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)

    @property
    def model(self) -> str:
        return "stub-tone"

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _StubChunked(tts=self, input_text=text, conn_options=conn_options)


class _StubChunked(tts.ChunkedStream):
    async def _run(self, output_emitter) -> None:
        output_emitter.initialize(request_id="stub", sample_rate=24000, num_channels=1, mime_type="audio/pcm")
        n = int(24000 * min(2.0, 0.04 * max(1, len(self._input_text))))
        pcm = (np.sin(2 * np.pi * 440 * np.arange(n) / 24000) * 3000).astype(np.int16)
        output_emitter.push(pcm.tobytes())
        output_emitter.flush()


async def run_agent_session(x: np.ndarray, url: str, mode: str = "stt", speed: float = 1.0, timeout_ms: int = 1000,
                            tail_s: float = 2.5, quiet: bool = True, policy: str = "timeout",
                            eot_threshold: float | None = None, agent_end_s: float | None = None) -> dict:
    """Drive a room-less AgentSession with our STT + VAD (+ turn detector). Returns the recorded events, each with
    wall and audio time (audio seconds pushed when it happened)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from livekit_agent_worker import build_session  # the exact configuration the room worker uses
    fe = AudioforgeFrontend(url, turn_policy=policy, timeout_ms=timeout_ms, eot_threshold=eot_threshold)
    session = build_session(fe, mode)
    audio = np.concatenate([x, np.zeros(int(tail_s * SR), np.float32)])
    inp = WavAudioInput(audio, speed, at=(agent_end_s, fe.agent_end) if agent_end_s is not None else None)
    session.input.audio = inp
    session.output.audio = NullAudioOutput()
    events: list[dict] = []

    def rec(kind, **kw):
        w = time.time()
        e = {"kind": kind, "wall": w, "audio_t": round(inp.clock.audio_at(w), 3), **kw}
        events.append(e)
        if not quiet:
            print(f"  {e['audio_t']:7.2f}s  AGT {kind:18s} {json.dumps(kw)}", flush=True)

    @session.on("user_input_transcribed")
    def _t(ev):
        if ev.is_final:
            rec("user_final", text=ev.transcript, speaker=ev.speaker_id)

    @session.on("conversation_item_added")
    def _c(ev):
        role = getattr(ev.item, "role", None)
        text = getattr(ev.item, "text_content", None)
        rec("user_turn" if role == "user" else "agent_reply" if role == "assistant" else f"item_{role}", text=text)

    @session.on("agent_state_changed")
    def _a(ev):
        rec("agent_state", state=ev.new_state)

    @session.on("user_state_changed")
    def _u(ev):
        rec("user_state", state=ev.new_state)

    @session.on("metrics_collected")
    def _m(ev):
        m = ev.metrics
        if getattr(m, "type", "") == "eou_metrics":
            rec("eou_metrics", end_of_utterance_delay=round(m.end_of_utterance_delay, 3),
                transcription_delay=round(m.transcription_delay, 3))

    await session.start(agent=Agent(instructions="Echo the user (offline test)."), record=False)
    await asyncio.wait_for(inp.done.wait(), len(audio) / SR / max(speed, 1e-3) + 60)
    await asyncio.sleep(1.0)
    await session.aclose()
    await fe.aclose()
    det = session.turn_detection
    preds = [{"audio_t": round(inp.clock.audio_at(p["wall"]), 3), "p": p["p"], "why": p["why"]}
             for p in getattr(det, "predictions", [])]
    cut = [te for te in fe.turn_ends if te["policy"] == fe.opts.cut_policy]
    return {"events": events, "clock": inp.clock, "mode": mode, "predictions": preds,
            "decisions_t": [te["t"] for te in cut], "decision_arrivals": [te["arrival"] for te in cut],
            "agent_end_t": inp.at_t, "stats": fe.last_stats}


def agent_metrics(res: dict, ref: dict) -> dict:
    clock: Clock = res["clock"]
    end = ref.get("turn_end_s")
    onset = ref.get("onset_s")
    users = [e for e in res["events"] if e["kind"] == "user_turn"]
    replies = [e for e in res["events"] if e["kind"] == "agent_reply"]
    out = {"mode": res["mode"], "agent_end_t": res.get("agent_end_t"), "server_stats": res.get("stats"),
           "user_turns_audio_t": [e["audio_t"] for e in users],
           "user_turn_texts": [e["text"] for e in users], "server_decisions_t": res.get("decisions_t")}
    if end is not None:
        out["early_commits"] = sum(1 for e in users if onset is not None and onset < e["audio_t"] < end)
        after = [e for e in users if e["audio_t"] >= end]
        out["commit_ms_after_turn_end"] = round((after[0]["wall"] - clock.wall_at(end)) * 1000) if after else None
        if after:  # LiveKit's own share: commit minus the arrival of the latest server decision before it
            prev = [a for a in res.get("decision_arrivals", []) if a <= after[0]["wall"] + 1e-3]
            out["commit_ms_after_decision_arrival"] = round((after[0]["wall"] - prev[-1]) * 1000) if prev else None
        rep = [e for e in replies if e["audio_t"] >= end]
        out["reply_ms_after_turn_end"] = round((rep[0]["wall"] - clock.wall_at(end)) * 1000) if rep else None
    eou = [e for e in res["events"] if e["kind"] == "eou_metrics"]
    out["eou_metrics"] = [{k: e[k] for k in ("audio_t", "end_of_utterance_delay", "transcription_delay")} for e in eou]
    if res.get("predictions"):  # AudioforgeTurnDetector requests and how each was settled
        out["turn_detector_predictions"] = [p for p in res["predictions"] if p["why"] != "superseded"]
    return out


# --------------------------------------------------------------------------- main
def dump_events(path: Path, r: dict) -> None:
    """Raw stream-mode events (every SpeechEvent, VAD START/END, server turn_ends) with wall + audio time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clock: Clock = r["clock"]
    rows = []
    for w, e in r["stt"]:
        a = e.alternatives[0] if e.alternatives else None
        rows.append({"wall": w, "audio_t": clock.audio_at(w), "src": "stt", "type": e.type.name,
                     "text": a.text if a else None, "speaker": a.speaker_id if a else None,
                     "meta": a.metadata if a else None, "speech_end_time": e.speech_end_time,
                     "speech_start_time": e.speech_start_time})
    for w, e in r["vad"]:
        if e.type != vad.VADEventType.INFERENCE_DONE:
            rows.append({"wall": w, "audio_t": clock.audio_at(w), "src": "vad", "type": e.type.name,
                         "t": e.timestamp, "speech_duration": e.speech_duration,
                         "silence_duration": e.silence_duration, "lag": e.inference_duration})
    for te in r["turn_ends"]:
        rows.append({"wall": te["arrival"], "audio_t": clock.audio_at(te["arrival"]), "src": "server", **te})
    rows.sort(key=lambda z: z["wall"])
    path.write_text("".join(json.dumps(z, default=str) + "\n" for z in rows))


def load_ref(wav: Path) -> dict:
    """``<name>.ref.json`` (what ``pipecat_local_demo.py prepare`` writes), else the older ``<name>.json``."""
    for j in (wav.with_suffix(".ref.json"), wav.with_suffix(".json")):
        if j.exists():
            return json.loads(j.read_text())
    return {}


async def amain(a) -> dict:
    from audioforge.data import load_wav
    results = []
    for path in a.wavs:
        wav = Path(path)
        ref = load_ref(wav)
        x = load_wav(str(wav), SR).astype(np.float32)
        print(f"== {wav.name}: {len(x) / SR:.2f}s, primary turn {ref.get('onset_s')}-{ref.get('turn_end_s')}s, "
              f"first word ends {ref.get('first_word_end_s')}s", flush=True)
        agent_end = getattr(a, "agent_end", None) or {}
        ae = agent_end.get(wav.stem)
        eot = getattr(a, "eot_threshold", None)
        r = await run_streams(x, a.url, a.speed, a.tail, a.timeout_ms, a.policy, a.quiet, eot, ae)
        if a.events_dir:
            dump_events(Path(a.events_dir) / f"{wav.stem}.jsonl", r)
        m = {"window": wav.stem, **{k: ref.get(k) for k in ("onset_s", "turn_end_s", "first_word_end_s")},
             **stream_metrics(r, ref)}
        modes = {"none": [], "stt": ["stt"], "detector": ["detector"], "both": ["stt", "detector"]}[a.agent_session]
        for mode in modes:
            print(f"  -- AgentSession, turn_detection={mode}", flush=True)
            ar = await run_agent_session(x, a.url, mode, a.speed, a.timeout_ms, a.tail, a.quiet, a.policy, eot, ae)
            m[f"agent_{mode}"] = agent_metrics(ar, ref)
        print("  SUMMARY " + json.dumps({k: v for k, v in m.items() if k != "final_transcript"}), flush=True)
        results.append(m)
    out = {"url": a.url, "speed": a.speed, "tail_s": a.tail, "timeout_ms": a.timeout_ms, "policy": a.policy,
           "windows": results}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time; 0 = as fast as possible")
    ap.add_argument("--tail", type=float, default=2.5, help="seconds of silence appended after each window")
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--policy", choices=list(POLICIES), default="timeout")
    ap.add_argument("--eot-threshold", type=float, default=None,
                    help="head threshold (default: the server's for the policy, 0.98 / 0.99828 / 0.998283)")
    ap.add_argument("--agent-end", type=lambda p: json.loads(Path(p).read_text()), default=None,
                    help="JSON {window stem: audio seconds} of the stand-in TTS-end event (fe.agent_end()) per window; "
                         "the server must run with --enroll after_agent[_arm]")
    ap.add_argument("--agent-session", choices=["none", "stt", "detector", "both"], default="none")
    ap.add_argument("--out")
    ap.add_argument("--events-dir", help="write each window's raw stream-mode events here (JSONL)")
    ap.add_argument("--quiet", action="store_true", help="do not print every event")
    a = ap.parse_args(argv)
    return asyncio.run(amain(a))


if __name__ == "__main__":
    main()
