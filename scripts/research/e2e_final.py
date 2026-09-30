"""Final end-to-end comparison: our single-model front end vs the default local stacks of Pipecat and LiveKit.

research/E2E_FINAL.md has the setup, tables and verdict; runs/e2e_final.json the numbers. Systems (CPU, 2 threads):

  A  Pipecat default local stack: SileroVADAnalyzer (Pipecat's VAD state machine) + LocalSmartTurnAnalyzerV3
     (bundled smart-turn-v3.2-cpu, 3 s stop fallback; Pipecat's default stop strategy) + WhisperSTTService
     (faster-whisper "small", Pipecat's own local STT service; segmented on VAD stops).
  B  LiveKit default local stack: livekit-plugins-silero VAD + livekit-plugins-turn-detector EnglishModel (its
     runner, in-process executor) + the same faster-whisper "small" as a non-streaming LiveKit STT (the session
     wraps it in LiveKit's StreamAdapter on the Silero VAD). Evaluation only (LiveKit Model License).
  C  Ours, product default: audioforge.serve, runs/stage1_served.afm + Sortformer v2 (0.32 s), turn policy timeout
     1000 ms, through the committed Pipecat / LiveKit adapters.
  D  Ours, best rules: the same server with --enroll after_agent_arm and turn policy hybrid_dyn; agent_end sent at
     the other party's labelled turn ends (a stand-in for the TTS-end event).
  Dp Ours, predictive trigger OR per-channel Silero (research/archive/DYADIC.md section 8, head (c), oto-fitted point
     0.012 / 0.012 / 1.4 s): not wired into serve.py, so it runs through the offline causal scorer (``predictive``).

A, C, D run through a Pipecat 1.12 pipeline (WAV transport at 1x -> STT -> user aggregator -> mock LLM/TTS);
B, C, D through a room-less LiveKit Agents 1.8 AgentSession (stub LLM / tone TTS). The response moment is what a
caller waits for: Pipecat's LLMContextFrame reaching the LLM, LiveKit's committed user turn.

    PY=.venv/bin/python; W=<scratch>/e2e_final
    PYTHONPATH=. $PY scripts/research/e2e_final.py prepare --work $W                  # clips + labels (labels: scoring only)
    PYTHONPATH=. $PY scripts/research/e2e_final.py queue --work $W --wt <clean worktree>   # timed sessions, guarded
    PYTHONPATH=. $PY scripts/research/e2e_final.py predictive --work $W                # Dp, offline
    PYTHONPATH=. $PY scripts/research/e2e_final.py components --work $W                # RTF by component, isolated
    PYTHONPATH=. $PY scripts/research/e2e_final.py report --work $W --out runs/e2e_final.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SR = 16000
CHUNK_S = 0.02
SCRATCH = Path("/private/tmp/claude-501/-Users-maxm/54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad")
WORK = SCRATCH / "e2e_final"
AMI_DIR = SCRATCH / "wf-integ" / "pipecat" / "ami"  # INTEGRATION section 4's 5 windows (+ .ref.json)
AMI_LK_DIR = SCRATCH / "wf-integ" / "livekit" / "audio"  # same WAVs, refs with first-word times
AMI_AGENT_END = SCRATCH / "deadair_shipped" / "agent_end.json"  # INTEGRATION section 8's stand-in
GUARD_PATTERN = "audioforge.*train|spk_head.py train|layer_routing.py train|lid.*train|train.py"
MAX_LOAD = 6.0
WHISPER_MODEL = "small"
PADS = {"ami": 7.0, "turnbench": 6.0, "oto": 6.0}  # AMI: INTEGRATION section 8's pad; 6 s >= the 6 s horizon
PRED_POINT = (0.012, 0.012, 1.4)  # DYADIC section 8 predictive OR Silero, fitted on oto -> TurnBench
PRED_HEAD = "runs/stage1_turn_dyadic_mh.afm"


# ============================================================================ guard, usage, timers
MAX_OTHER_CPU = 150.0  # % of one core used by other processes (other agents' eval jobs are not "training")


def other_cpu_pct(dt: float = 1.0) -> float:
    """CPU % (of one core) of every process that is not part of this measurement (e2e_final.py processes)."""
    import psutil
    procs = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if "e2e_final.py" in " ".join(p.info["cmdline"] or []):
                continue
            p.cpu_percent(None)
            procs.append(p)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(dt)
    tot = 0.0
    for p in procs:
        try:
            tot += p.cpu_percent(None)
        except Exception:  # noqa: BLE001
            pass
    return tot


def guard_ok() -> tuple[bool, str]:
    """The machine rule (no training process, 1-min load < 6) plus a quiet check of our own: other processes use
    < MAX_OTHER_CPU % (4 performance cores here; other agents' evaluation jobs made our server fall behind)."""
    r = subprocess.run(["pgrep", "-f", GUARD_PATTERN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p.strip() and int(p) != os.getpid()]
    load = os.getloadavg()[0]
    oc = other_cpu_pct() if not pids and load < MAX_LOAD else -1.0
    return (not pids and load < MAX_LOAD and 0 <= oc < MAX_OTHER_CPU), f"train_pids={pids} load1={load:.2f} other_cpu={oc:.0f}%"


def wait_guard(log=print, poll: float = 30.0, max_wait_s: float = 6 * 3600):
    if os.environ.get("E2E_NO_GUARD"):  # dry runs only (setup is allowed while other agents train)
        return os.getloadavg()[0]
    t0 = time.time()
    while True:
        ok, why = guard_ok()
        if ok:
            return os.getloadavg()[0]
        if time.time() - t0 > max_wait_s:
            raise TimeoutError(f"guard not satisfied for {max_wait_s} s: {why}")
        log(f"[guard] waiting: {why}")
        time.sleep(poll)


class UsageSampler(threading.Thread):
    """(wall, cpu user+sys s, rss MB) of each pid every ``dt`` s."""

    def __init__(self, pids: dict[str, int], dt: float = 0.2):
        super().__init__(daemon=True)
        import psutil
        self.procs = {k: psutil.Process(p) for k, p in pids.items() if p}
        self.samples = {k: [] for k in self.procs}
        self.dt, self._halt = dt, threading.Event()

    def _sample(self):
        now = time.perf_counter()
        for k, p in self.procs.items():
            try:
                c = p.cpu_times()
                self.samples[k].append((now, c.user + c.system, p.memory_info().rss / 2 ** 20))
            except Exception:  # noqa: BLE001 - the process may be gone at the end
                pass

    def run(self):
        while not self._halt.is_set():
            self._sample()
            self._halt.wait(self.dt)

    def stop(self) -> dict:
        from audioforge.e2e_metrics import usage
        self._halt.set()
        self.join(2)
        self._sample()
        return {k: usage(v) for k, v in self.samples.items()}


class Timers:
    """Per-component call timings (wall ms via perf_counter, process CPU ms via process_time) by monkeypatching."""

    def __init__(self):
        self.calls: dict[str, list] = {}

    def reset(self):
        self.calls = {}

    def add(self, name, wall_ms, cpu_ms, **kw):
        self.calls.setdefault(name, []).append({"wall_ms": round(wall_ms, 3), "cpu_ms": round(cpu_ms, 3), **kw})

    def wrap(self, cls, attr: str, name: str, extra=None):
        orig = getattr(cls, attr)
        if getattr(orig, "_e2e_wrapped", False):
            return
        timers = self

        if asyncio.iscoroutinefunction(orig):
            async def w(*a, **k):
                t0, c0 = time.perf_counter(), time.process_time()
                r = await orig(*a, **k)
                timers.add(name, (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000,
                           **(extra(a, k, r) if extra else {}))
                return r
        else:
            def w(*a, **k):
                t0, c0 = time.perf_counter(), time.process_time()
                r = orig(*a, **k)
                timers.add(name, (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000,
                           **(extra(a, k, r) if extra else {}))
                return r
        w._e2e_wrapped = True
        setattr(cls, attr, w)

    def summary(self, audio_s: float) -> dict:
        out = {}
        for k, v in self.calls.items():
            wl = np.array([c["wall_ms"] for c in v])
            cp = np.array([c["cpu_ms"] for c in v])
            out[k] = {"n": len(v), "wall_ms_sum": round(float(wl.sum()), 1), "cpu_ms_sum": round(float(cp.sum()), 1),
                      "wall_ms_p50": round(float(np.percentile(wl, 50)), 2),
                      "wall_ms_p95": round(float(np.percentile(wl, 95)), 2),
                      "rtf_wall": round(float(wl.sum()) / 1000 / max(audio_s, 1e-9), 5),
                      "rtf_cpu": round(float(cp.sum()) / 1000 / max(audio_s, 1e-9), 5)}
        return out


TIMERS = Timers()


def _wav_read(p: Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, (p, sr)
    return x.mean(1)


def _wav_write(p: Path, x: np.ndarray):
    import soundfile as sf
    sf.write(str(p), np.clip(np.asarray(x, np.float32), -1, 1), SR, subtype="PCM_16")


# ============================================================================ clips
def _merge(iv, gap=0.0):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _floor_gaps(turn_spans, act, dur, min_gap=0.3):
    """Points (mid of silent gaps >= min_gap between any activity, outside every floor turn span)."""
    busy = _merge(list(act) + list(turn_spans))
    pts, prev = [], 0.0
    for s, e in busy:
        if s - prev >= min_gap:
            pts.append(round((prev + s) / 2, 3))
        prev = max(prev, e)
    if dur - prev >= min_gap:
        pts.append(round((prev + dur) / 2, 3))
    return pts


def pick_segment(human_turns, agent_turns, act, dur, zones, length=35.0, max_len=60.0, start_min=30.0,
                 min_user_ends=3, min_agent_ends=2):
    """First [a, b) with a, b in floor gaps, a >= start_min, length <= b - a <= max_len, no excluded zone, and enough
    user / agent floor-turn ends inside. Turns are dicts with start / end (non-backchannel)."""
    spans = [(t["start"], t["end"]) for t in human_turns + agent_turns]
    gaps = _floor_gaps(spans, act, dur, min_gap=0.2)
    for a in (g for g in gaps if g >= start_min):
        for b in (g for g in gaps if a + length <= g <= a + max_len):  # the shortest qualifying segment from a
            if any(za < b and zb > a for za, zb in zones):
                break
            nu = sum(a + 1.0 < t["end"] <= b for t in human_turns)
            na = sum(a < t["end"] <= b for t in agent_turns)
            if nu >= min_user_ends and na >= min_agent_ends:
                return a, b
    return None


def tb_ref_text(cid: str, a: float, b: float, speakers=(1, 2)) -> str:
    """TurnBench reference text of [a, b): annotator a's segments (both labels: turns and backchannels) whose midpoint
    lies inside, in start order, with every bracketed non-speech tag ([chuckles], [lip smack], [um] ...) removed."""
    import re
    from audioforge.datasets import dyadic as D
    row = D.tb_row(D.TB_ROOT, cid, ["conversation_id"] + [f"speaker_{k}_annotation_a" for k in speakers])
    segs = sorted((e["start_s"], e["text"]) for k in speakers for e in row[f"speaker_{k}_annotation_a"]
                  if a <= (e["start_s"] + e["end_s"]) / 2 < b)
    return re.sub(r"\s+", " ", " ".join(re.sub(r"\[[^\]]*\]", " ", t) for _, t in segs)).strip()


def two_party_clip(ds, cid: str, corpus: str, length: float = 35.0) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """(ref, mono, user channel) for one conversation, or None if no segment qualifies."""
    from audioforge.datasets import dyadic as D
    human = ds.roles[cid]["human"]
    agent = 1 - human
    turns = ds.turns[cid]
    ht = [t for t in turns if t["speaker"] == f"{cid}:{human}" and not t["bc"]]
    at = [t for t in turns if t["speaker"] == f"{cid}:{agent}" and not t["bc"]]
    act = [iv for s in (f"{cid}:0", f"{cid}:1") for iv in ds.acts[cid][s]]
    seg = pick_segment(ht, at, act, ds.duration(cid), ds.zones[cid], length=length)
    if seg is None:
        return None
    a, b = seg
    if corpus == "turnbench":
        ch = np.asarray(ds.channels16k(cid)[:, int(a * SR): int(b * SR)], np.float32)
    else:
        x, sr, _ = D.read_stereo(corpus, ds.root, cid)
        ch = D.resample16k(x[int(a * sr): int(b * sr) + 1], sr)
        ch = np.asarray(ch.T if ch.shape[0] != 2 else ch, np.float32)[:, : int(round((b - a) * SR))]
    mono = D.mix_mono(ch)
    user = np.clip(ch[human], -1, 1)
    clipt = lambda t: round(min(max(t - a, 0.0), b - a), 3)  # noqa: E731
    uturns = [(clipt(t["start"]), clipt(t["end"])) for t in ht if t["end"] > a and t["start"] < b]
    scored = [a + 0.5 < t["end"] <= b for t in ht if t["end"] > a and t["start"] < b]
    ref = {"name": f"{'tb' if corpus == 'turnbench' else 'oto'}_{cid}", "set": corpus, "conversation": cid,
           "start": round(a, 3), "dur": round(b - a, 3), "human_channel": human, "agent_rule": ds.agent_rule,
           "user_turns": uturns, "scored": scored,
           "user_intervals": [(clipt(s), clipt(e)) for s, e in ds.acts[cid][f"{cid}:{human}"] if e > a and s < b],
           "agent_intervals": [(clipt(s), clipt(e)) for s, e in ds.acts[cid][f"{cid}:{agent}"] if e > a and s < b],
           "agent_ends": [clipt(t["end"]) for t in at if a < t["end"] <= b],
           "text_mono": tb_ref_text(cid, a, b) if corpus == "turnbench" else None,
           "text_user": tb_ref_text(cid, a, b, (human + 1,)) if corpus == "turnbench" else None,
           "has_text": corpus == "turnbench",
           "first_onset_mono": clipt(min(s for s, e in act if e > a)) if act else None,
           "first_onset_user": clipt(min((s for s, e in ds.acts[cid][f"{cid}:{human}"] if e > a), default=a)),
           "word_timing": ds.word_timing,
           "note": "labels for scoring only; agent_ends = the other party's floor-turn ends (TTS-end stand-in)"}
    return ref, mono, user


def ami_clips(out: Path) -> list[dict]:
    ae = json.loads(AMI_AGENT_END.read_text())
    refs = []
    for wav in sorted(AMI_DIR.glob("ami_*.wav")):
        r = json.loads(wav.with_suffix(".ref.json").read_text())
        lk = json.loads((AMI_LK_DIR / f"{wav.stem}.json").read_text())
        x = _wav_read(wav)
        _wav_write(out / f"{wav.stem}.mono.wav", x)
        pre = [iv for iv in r["primary_intervals"] if iv[1] <= r["onset_s"] + 1e-6]
        refs.append({"name": wav.stem, "set": "ami", "conversation": r["meeting"], "start": r["start"],
                     "dur": round(len(x) / SR, 3), "user_turns": [(r["onset_s"], r["turn_end_s"])], "scored": [True],
                     "user_intervals": r["primary_intervals"], "pre_onset_primary": pre,
                     "agent_ends": [ae[wav.stem]] if wav.stem in ae else [],
                     "text_mono": r["all_text"], "text_user": None, "has_text": True,
                     "first_onset_mono": lk.get("first_word_start_s"), "first_word_end_mono": lk.get("first_word_end_any_s"),
                     "n_speakers": r["n_speakers"],
                     "note": "INTEGRATION section 4 window; agent_end = INTEGRATION section 8's label stand-in"})
    return refs


def cmd_prepare(a):
    from audioforge.datasets import dyadic as D
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    out = Path(a.work) / "clips"
    out.mkdir(parents=True, exist_ok=True)
    refs = ami_clips(out)
    # TurnBench dev: conversations in id order, first n_tb with a qualifying segment; evaluation only (licence)
    tb_ids = D.list_ids("turnbench")
    tb = D.Dyadic(tb_ids, "turnbench", verbose=False)
    # oto: dev conversations of the dyadic runs (never trained on), minus section 7's 16 (the predictive trigger's
    # oto fit used the dev ones among them)
    import bench_dyadic_heads as BH
    import bench_turnbench_latency as BL
    fit = set(BL.oto_ids())
    oto_ids = [c for c in BH.dev_ids() if c not in fit]
    oto = D.Dyadic(oto_ids, "oto", verbose=False, cache_dtype="float16")
    for corpus, ds, n in (("turnbench", tb, a.n_tb), ("oto", oto, a.n_oto)):
        k = 0
        for cid in ds.meetings:
            if k >= n:
                break
            r = two_party_clip(ds, cid, corpus, a.length)
            if r is None:
                print(f"  {corpus}/{cid}: no qualifying segment", flush=True)
                continue
            ref, mono, user = r
            _wav_write(out / f"{ref['name']}.mono.wav", mono)
            _wav_write(out / f"{ref['name']}.user.wav", user)
            refs.append(ref)
            k += 1
            print(f"  {ref['name']}: {ref['start']:.1f}+{ref['dur']:.1f}s user ends {sum(ref['scored'])} "
                  f"agent ends {len(ref['agent_ends'])}", flush=True)
    (Path(a.work) / "clips.json").write_text(json.dumps(refs, indent=1))
    print(f"{len(refs)} clips -> {out}")


def load_clips(work: Path) -> dict[str, dict]:
    return {r["name"]: r for r in json.loads((Path(work) / "clips.json").read_text())}


# ============================================================================ Pipecat
_WHISPER = {}


def whisper_model():
    """One faster-whisper model per process (Pipecat's WhisperSTTService loads it in __init__; cached here)."""
    if "m" not in _WHISPER:
        from faster_whisper import WhisperModel
        _WHISPER["m"] = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="default")
    return _WHISPER["m"]


def whisper_text(model, pcm: np.ndarray) -> str:
    """Exactly WhisperSTTService.run_stt's call: language en, no hotwords / prompt, segments with no_speech_prob < 0.4."""
    segs, _ = model.transcribe(pcm, language="en", hotwords=None, initial_prompt=None)
    return "".join(f"{s.text} " for s in segs if s.no_speech_prob < 0.4)


def _stub_mlx_whisper():
    """pipecat.services.whisper.stt imports mlx_whisper on Apple Silicon even for the faster-whisper service; the MLX
    class (Apple GPU) is not used here (CPU only), so an empty module stands in for the package."""
    import types
    sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))


def instrument_pipecat():
    _stub_mlx_whisper()
    import faster_whisper
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.services.whisper.stt import WhisperSTTService
    TIMERS.wrap(SileroVADAnalyzer, "voice_confidence", "vad_silero")
    TIMERS.wrap(LocalSmartTurnAnalyzerV3, "_predict_endpoint", "turn_smart_turn",
                extra=lambda a, k, r: {"audio_s": round(len(a[1]) / SR, 2), "p": round(float(r.get("probability", -1)), 4)
                                       if isinstance(r, dict) else None})

    # Pipecat iterates faster-whisper's lazy segment generator on the event loop: time run_stt as a whole
    orig = WhisperSTTService.run_stt
    if not getattr(orig, "_e2e_wrapped", False):
        async def run_stt(self, audio: bytes):
            t0, c0 = time.perf_counter(), time.process_time()
            async for f in orig(self, audio):
                yield f
            TIMERS.add("stt_whisper", (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000,
                       audio_s=round(len(audio) / 2 / SR, 3))
        run_stt._e2e_wrapped = True
        WhisperSTTService.run_stt = run_stt
    # share one model across pipelines
    if not getattr(faster_whisper.WhisperModel, "_e2e_cached", False):
        import pipecat.services.whisper.stt as ws
        ws.WhisperModel = lambda *a, **k: whisper_model()  # noqa: E731
        faster_whisper.WhisperModel._e2e_cached = True


async def pipecat_run(system: str, audio: np.ndarray, *, url: str | None, pad_s: float, policy: str | None,
                      enroll: str | None, agent_ends: list[float]) -> dict:
    from loguru import logger
    from pipecat.frames.frames import (BotStoppedSpeakingFrame, EndFrame, InterimTranscriptionFrame,
                                       TranscriptionFrame)
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMUserAggregatorParams
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.workers.runner import WorkerRunner
    import examples.pipecat_local_demo as PD

    class Transport(PD.WavInputTransport):
        """PD's WAV transport with a list of stand-in TTS-end times (one BotStoppedSpeakingFrame each)."""

        def __init__(self, x, *, ends, **kw):
            super().__init__(x, **kw)
            self.ends, self.ends_sent = sorted(ends), []

        async def _play(self):
            n = int(SR * CHUNK_S) * 2
            chunks = [self._pcm[i:i + n] for i in range(0, len(self._pcm), n)]
            self.t0 = t0 = time.perf_counter()
            k = 0
            for i, c in enumerate(chunks):
                due = t0 + (i + 1) * CHUNK_S
                d = due - time.perf_counter()
                if d > 0:
                    await asyncio.sleep(d)
                self.max_push_late_ms = max(self.max_push_late_ms, (time.perf_counter() - due) * 1000)
                await self.push_audio_frame(PD.InputAudioRawFrame(audio=c, sample_rate=SR, num_channels=1))
                while k < len(self.ends) and (i + 1) * CHUNK_S >= self.ends[k] - 1e-9:
                    f = BotStoppedSpeakingFrame()
                    f.synthetic_agent_end = True
                    self.ends_sent.append(round((time.perf_counter() - t0), 3))
                    await self.push_frame(f)
                    k += 1
            self.t_end = time.perf_counter()
            if self.on_done:
                self.on_done()

    class Tap(FrameProcessor):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.finals, self.interims, self.first_text = [], 0, None
            self.checks = PD._OrderChecks("tap")

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if getattr(frame, "synthetic_agent_end", False):
                return
            self.checks.see(frame, direction)
            now = time.perf_counter()
            if isinstance(frame, TranscriptionFrame):
                if not frame.text.strip():
                    self.checks.violations.append("tap: empty TranscriptionFrame")
                self.finals.append({"perf": now, "text": frame.text})
                self.first_text = self.first_text or now
            elif isinstance(frame, InterimTranscriptionFrame):
                self.interims += 1
                self.first_text = self.first_text or now
            await self.push_frame(frame, direction)

    TIMERS.reset()
    if system == "A":
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.services.whisper.stt import WhisperSTTService
        stt = WhisperSTTService(device="cpu", settings=WhisperSTTService.Settings(model=WHISPER_MODEL))
        params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())  # default strategies: smart-turn v3
        hub, ends = None, []
    else:
        from audioforge.integrations.pipecat import (AudioforgeHub, AudioforgeSTTService, AudioforgeTurnAnalyzer,
                                                     AudioforgeVADAnalyzer)
        from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        hub = AudioforgeHub()
        stt = AudioforgeSTTService(url=url, hub=hub, timeout_ms=1000, enroll=enroll, end_timeout=90.0)  # stats after a backlog
        turn = AudioforgeTurnAnalyzer(hub, policy=policy)
        params = LLMUserAggregatorParams(vad_analyzer=AudioforgeVADAnalyzer(hub), user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()],
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn)]))
        ends = agent_ends if enroll else []
    agg = LLMUserAggregator(LLMContext(), params=params)
    done = asyncio.Event()
    transport = Transport(audio, pad_s=pad_s, on_done=done.set, ends=ends)
    clock = (lambda p: (p - transport.t0) if transport.t0 is not None else float("nan"))
    tap, mock = Tap(), PD.MockLLMTTS(clock)
    worker = PipelineWorker(Pipeline([transport, stt, tap, agg, mock]), params=PipelineParams(audio_in_sample_rate=SR),
                            enable_rtvi=False, cancel_on_idle_timeout=False, idle_timeout_secs=None)
    errors, finished = [], []

    @worker.event_handler("on_pipeline_error")
    async def _err(_w, frame):
        errors.append(str(frame))

    @worker.event_handler("on_pipeline_finished")
    async def _fin(_w, frame):
        finished.append(type(frame).__name__)

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    end_perf = []

    async def ender():
        await done.wait()
        end_perf.append(time.perf_counter())
        await worker.queue_frame(EndFrame())

    cap = PD._LogCapture()
    hid = logger.add(cap, level="WARNING")
    await asyncio.wait_for(asyncio.gather(runner.run(), ender()), len(audio) / SR + pad_s + 200)
    logger.remove(hid)
    t0 = transport.t0
    at = (lambda p: round(p - t0, 3))
    cut = end_perf[0] if end_perf else float("inf")
    raw = {"t0": t0, "audio_s": transport.audio_s, "total_s": transport.total_s,
           "responses": [at(p) for p, _ in mock.contexts if p <= cut],
           "response_texts": [tx for p, tx in mock.contexts if p <= cut],
           "decisions": [at(p) for p in mock.decisions if p <= cut],
           "starts": [at(p) for p in mock.starts if p <= cut],
           "finals": [{"t": at(f["perf"]), "text": f["text"]} for f in tap.finals],
           "first_text_t": at(tap.first_text) if tap.first_text else None, "n_interims": tap.interims,
           "max_push_late_ms": round(transport.max_push_late_ms, 1), "agent_ends_sent": transport.ends_sent,
           "finished": finished, "pipeline_errors": errors,
           "violations": tap.checks.violations + mock.checks.violations,
           "order_notes": tap.checks.notes + mock.checks.notes, "log_warnings": cap.records}
    if hub is not None:
        raw.update(server_stats=hub.stats, server_ready=hub.ready,
                   server_turn_ends=[{"t": e.t, "policy": e.policy, "arrival_t": at(e.perf)} for e in hub.turn_ends],
                   frame_lag_ms=[round((p - hub.sent_perf(t)) * 1000, 1) for t, p in hub.frame_arrivals
                                 if hub.sent_perf(t) is not None], enroll_sent=hub.enroll_sent,
                   enrolled_column=hub.enrolled_column, turn_received=len(turn.received),
                   turn_discarded=[{k: v for k, v in e.__dict__.items() if k != "perf"} for e in turn.discarded])
    return raw


# ============================================================================ LiveKit
class _LocalEOUExecutor:
    """LiveKit's inference executor protocol, in-process: the plugin's own runner (chat template, truncation, q8
    ONNX) on a worker thread; its ONNX session rebuilt with 2 intra-op threads (the 2-threads-per-process rule)."""

    def __init__(self):
        import onnxruntime as ort
        from livekit.plugins.turn_detector.english import _EUORunnerEn
        r = _EUORunnerEn()
        r.initialize()
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
        so.add_session_config_entry("session.dynamic_block_base", "4")
        r._session = ort.InferenceSession(r._session._model_path, providers=["CPUExecutionProvider"], sess_options=so)
        self.r = r

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        t0, c0 = time.perf_counter(), time.process_time()
        out = await asyncio.to_thread(self.r.run, data)
        j = json.loads(out)
        TIMERS.add("turn_livekit_eou", (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000,
                   p=round(j["eou_probability"], 4), n_chars=len(j.get("input", "")))
        return out


_LK = {}


def livekit_b_parts():
    from livekit.plugins import silero
    from livekit.plugins.turn_detector.base import EOUModelBase
    from livekit.plugins.turn_detector.english import EnglishModel
    from livekit.plugins.silero import onnx_model
    if "ex" not in _LK:
        _LK["ex"] = _LocalEOUExecutor()
        TIMERS.wrap(onnx_model.OnnxModel, "__call__", "vad_silero")

    class LocalEnglishModel(EnglishModel):
        def __init__(self):
            EOUModelBase.__init__(self, model_type="en", inference_executor=_LK["ex"])
    return silero.VAD.load(), LocalEnglishModel()


def make_whisper_lk():
    from livekit import rtc
    from livekit.agents import stt
    from livekit.agents.language import LanguageCode

    class FasterWhisperSTT(stt.STT):
        """faster-whisper as a non-streaming LiveKit STT (the session wraps it in stt.StreamAdapter on its VAD)."""

        def __init__(self):
            super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
            self.m = whisper_model()

        @property
        def model(self) -> str:
            return f"faster-whisper-{WHISPER_MODEL}"

        @property
        def provider(self) -> str:
            return "local"

        async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
            f = rtc.combine_audio_frames(buffer)
            pcm = np.frombuffer(f.data, np.int16).astype(np.float32) / 32768.0
            if f.sample_rate != SR:
                from audioforge.serve import Resampler
                pcm = Resampler(f.sample_rate)(pcm)
            t0, c0 = time.perf_counter(), time.process_time()
            text = await asyncio.to_thread(whisper_text, self.m, pcm)
            TIMERS.add("stt_whisper", (time.perf_counter() - t0) * 1000, (time.process_time() - c0) * 1000,
                       audio_s=round(len(pcm) / SR, 3))
            return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                   alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text.strip())])
    return FasterWhisperSTT()


async def livekit_run(system: str, audio: np.ndarray, *, url: str | None, pad_s: float, policy: str | None,
                      enroll: str | None, agent_ends: list[float]) -> dict:
    from livekit.agents import Agent, AgentSession
    import examples.livekit_offline_demo as LD

    class Input(LD.WavAudioInput):
        def __init__(self, x, ats):
            super().__init__(x, 1.0)
            self.ats, self.k, self.at_ts = sorted(ats, key=lambda z: z[0]), 0, []
            self.done_wall = None

        async def __anext__(self):
            if self.t0 is None:
                self.t0 = time.time()
            if self.i >= len(self.frames):
                self.done_wall = self.done_wall or time.time()
                self.done.set()
                await asyncio.sleep(3600)
                raise StopAsyncIteration
            d = self.t0 + (self.i + 1) * LD.FRAME_S - time.time()
            if d > 0:
                await asyncio.sleep(d)
            while self.k < len(self.ats) and self.i * LD.FRAME_S >= self.ats[self.k][0] - 1e-9:
                self.ats[self.k][1]()
                self.at_ts.append(round(self.i * LD.FRAME_S, 3))
                self.k += 1
            f = self.frames[self.i]
            self.i += 1
            self.clock.mark()
            return f

    TIMERS.reset()
    fe = None
    if system == "B":
        vad, det = livekit_b_parts()
        session = AgentSession(stt=make_whisper_lk(), vad=vad, llm=LD.StubLLM(), tts=LD.StubTTS(),
                               turn_handling={"interruption": {"mode": "vad"},
                                              "preemptive_generation": {"enabled": False},
                                              "turn_detection": det}, user_away_timeout=None)
        ats = []
    else:
        sys.path.insert(0, str(Path(LD.__file__).resolve().parent))
        from livekit_agent_worker import build_session
        from audioforge.integrations.livekit import AudioforgeFrontend
        fe = AudioforgeFrontend(url, turn_policy=policy, timeout_ms=1000)
        session = build_session(fe, "stt")
        ats = [(t, fe.agent_end) for t in agent_ends] if enroll else []
    x = np.concatenate([audio, np.zeros(int(pad_s * SR), np.float32)])
    inp = Input(x, ats)
    session.input.audio = inp
    session.output.audio = LD.NullAudioOutput()
    ev = []

    def rec(kind, **kw):
        w = time.time()
        ev.append({"kind": kind, "wall": w, "t": round(inp.clock.audio_at(w), 3), **kw})

    session.on("user_input_transcribed", lambda e: rec("transcribed", final=e.is_final, text=e.transcript))
    session.on("conversation_item_added", lambda e: rec(
        {"user": "user_turn", "assistant": "agent_reply"}.get(getattr(e.item, "role", None), "item"),
        text=getattr(e.item, "text_content", None)))
    session.on("agent_state_changed", lambda e: rec("agent_state", state=e.new_state))
    session.on("user_state_changed", lambda e: rec("user_state", state=e.new_state))
    errors = []
    session.on("error", lambda e: errors.append(str(getattr(e, "error", e))[:300]))
    import logging

    class H(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.records = []

        def emit(self, r):
            self.records.append({"level": r.levelname, "name": r.name, "msg": r.getMessage()[:300]})
    h = H()
    logging.getLogger("livekit").addHandler(h)
    await session.start(agent=Agent(instructions="Echo the user (offline test)."), record=False)
    await asyncio.wait_for(inp.done.wait(), len(x) / SR + 90)
    await asyncio.sleep(1.0)
    stats = None
    if fe is not None and fe.link is not None:  # end the server session before the session close cancels the STT
        stats = await fe.link.end(wait=90)      # stream (a cancelled stream drops the server's final stats)
    await session.aclose()
    logging.getLogger("livekit").removeHandler(h)
    total = len(x) / SR
    ok = lambda e: e["wall"] <= inp.done_wall  # noqa: E731 - events after the last frame (end flush) do not count
    raw = {"audio_s": len(audio) / SR, "total_s": total,
           "responses": [e["t"] for e in ev if e["kind"] == "user_turn" and ok(e)],
           "response_texts": [e["text"] for e in ev if e["kind"] == "user_turn" and ok(e)],
           "replies": [e["t"] for e in ev if e["kind"] == "agent_reply" and ok(e)],
           "finals": [{"t": e["t"], "text": e["text"]} for e in ev if e["kind"] == "transcribed" and e["final"]],
           "first_text_t": next((e["t"] for e in ev if e["kind"] == "transcribed" and e["text"]), None),
           "n_interims": sum(1 for e in ev if e["kind"] == "transcribed" and not e["final"]),
           "agent_ends_sent": inp.at_ts, "errors": errors, "log_warnings": h.records,
           "events": [{k: v for k, v in e.items() if k != "wall"} for e in ev]}
    if fe is not None:
        await fe.aclose()
        raw.update(server_stats=stats or fe.last_stats, server_sessions=fe.sessions_opened,
                   server_turn_ends=[{"t": te["t"], "policy": te["policy"],
                                      "arrival_t": round(inp.clock.audio_at(te["arrival"]), 3)} for te in fe.turn_ends])
    return raw


# ============================================================================ one session (driver process)
def cmd_run(a):
    """One framework x system x condition over a list of clips; appends one JSON line per clip to --out."""
    warnings.filterwarnings("ignore")
    refs = load_clips(a.work)
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if not r.get("flag_rerun") or a.keep_flagged:
                done.add(r["clip"])
    if a.framework == "pipecat" and a.system == "A":
        instrument_pipecat()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    for name in a.clips:
        if name in done and not a.force:
            continue
        ref = refs[name]
        wav = Path(a.work) / "clips" / f"{name}.{a.cond}.wav"
        x = _wav_read(wav)
        load0 = wait_guard()
        pad = PADS[ref["set"]]
        sampler = UsageSampler({"driver": os.getpid(), "server": a.server_pid})
        sampler.start()
        t_start = time.time()
        fn = pipecat_run if a.framework == "pipecat" else livekit_run
        raw = asyncio.run(fn(a.system, x, url=a.url, pad_s=pad, policy=a.policy, enroll=a.enroll,
                             agent_ends=ref.get("agent_ends", [])))
        use = sampler.stop()
        load1 = os.getloadavg()[0]
        oc1 = other_cpu_pct()
        comp = TIMERS.summary(len(x) / SR)
        rtf = (raw.get("server_stats") or {}).get("rtf")
        comp_rtf = sum(v["rtf_wall"] for v in comp.values())
        inflated = (rtf is not None and rtf > 1.0) or comp_rtf > 1.0 or oc1 > 2 * MAX_OTHER_CPU
        rec = {"clip": name, "set": ref["set"], "cond": a.cond, "framework": a.framework, "system": a.system,
               "policy": a.policy, "enroll": a.enroll, "pad_s": pad, "wall_start": t_start, "load1_start": round(load0, 2),
               "load1_end": round(load1, 2), "other_cpu_pct_end": round(oc1), "usage": use, "components": comp, "component_calls": TIMERS.calls,
               "flag_rerun": bool(inflated), "raw": raw}
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[{a.framework} {a.system} {a.cond}] {name}: responses {len(raw['responses'])} rtf {rtf} "
              f"comp_rtf {comp_rtf:.3f} load {load0:.1f}->{load1:.1f}{' INFLATED' if inflated else ''}", flush=True)


# ============================================================================ server wrapper + queue
def cmd_serve(a, rest):
    """audioforge.serve with per-component time sums added to the debug stats (no change to serve.py)."""
    import audioforge.serve as S
    S.DEBUG_KEYS["stats"] |= {"asr_ms_sum", "diar_ms_sum", "turn_ms_sum", "silero_ms_sum", "enroll_ms_sum"}
    orig = S.Session.stats

    def stats(self):
        m = orig(self)
        if self.e.debug:
            m.update(asr_ms_sum=round(float(sum(self.asr_ms)), 1), diar_ms_sum=round(float(sum(self.diar_ms)), 1),
                     turn_ms_sum=round(float(self.asr.turn_ms), 1),
                     silero_ms_sum=round(float(sum(self.sil.ms)), 1) if self.sil is not None else 0.0,
                     enroll_ms_sum=round(float(sum(self.enroll_ms)), 1))
        return m
    S.Session.stats = stats
    S.main(rest)


NEMOTRON_DIAR = ["--diar", str(ROOT / "runs/nemo_nemotron3_diar.afm"), "--diar-pool", "max", "--diar-left", "1",
                 "--diar-spks", "4"]


def start_server(wt: Path, port: int, enroll: str, log: Path, diar_args: list[str] | None = None):
    env = {**os.environ, "PYTHONPATH": str(wt), "OMP_NUM_THREADS": "2", "TMPDIR": str(WORK / "tmp")}
    (WORK / "tmp").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "serve", "--", "--asr", str(ROOT / "runs/stage1_served.afm"),
           *(diar_args or ["--diar", str(ROOT / "runs/nemo_sortformer_v2.afm")]), "--port", str(port), "--threads", "2",
           "--debug-fields", "--enroll", enroll, "--silero", str(ROOT / "data/silero/silero_vad_v5.onnx")]
    f = log.open("w")
    p = subprocess.Popen(cmd, cwd=str(wt), env=env, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(240):
        time.sleep(1)
        txt = log.read_text()
        if "listening" in txt:
            return p
        if "Traceback" in txt or p.poll() is not None:
            break
    p.kill()
    raise RuntimeError(f"server failed: {log.read_text()[-2000:]}")


PLAN_SYSTEMS = {  # (framework, system) -> (server enroll or None, turn policy, driver enroll)
    ("pipecat", "A"): (None, None, None), ("pipecat", "C"): ("dominant", "timeout", None),
    ("pipecat", "D"): ("after_agent_arm", "hybrid_dyn", "after_agent_arm"),
    ("livekit", "B"): (None, None, None), ("livekit", "C"): ("dominant", "timeout", None),
    ("livekit", "D"): ("after_agent_arm", "hybrid_dyn", "after_agent_arm")}
# CN / DN = C / D with Nemotron-3-Diarization (100M, max pooling) as the diarizer instead of Sortformer v2 (117M)
NEMOTRON_SYSTEMS = {"CN": "C", "DN": "D"}
for (_fw, _sy) in list(PLAN_SYSTEMS):
    for _sn, _base in NEMOTRON_SYSTEMS.items():
        if _sy == _base:
            PLAN_SYSTEMS[(_fw, _sn)] = PLAN_SYSTEMS[(_fw, _sy)]


def plan(work: Path, only_sets=None) -> list[dict]:
    refs = load_clips(work)
    by_set = {}
    for n, r in refs.items():
        by_set.setdefault(r["set"], []).append(n)
    out = []
    for st in ("ami", "turnbench", "oto"):
        if only_sets and st not in only_sets:
            continue
        for cond in (("mono",) if st == "ami" else ("mono", "user")):
            for fw, sy in PLAN_SYSTEMS:
                out.append({"set": st, "cond": cond, "framework": fw, "system": sy, "clips": sorted(by_set.get(st, []))})
    return out


def cmd_queue(a):
    work = Path(a.work)
    runs = work / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    wt = Path(a.wt)
    port = a.port
    for s in plan(work, a.sets):
        if a.systems and s["system"] not in a.systems:
            continue
        if a.frameworks and s["framework"] not in a.frameworks:
            continue
        tag = f"{s['framework']}_{s['system']}_{s['set']}_{s['cond']}"
        out = runs / f"{tag}.jsonl"
        clips = s["clips"][: a.max_clips] if a.max_clips else s["clips"]
        have = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                if not r.get("flag_rerun"):
                    have.add(r["clip"])
                elif a.rerun_flagged is False:
                    have.add(r["clip"])
        todo = [c for c in clips if c not in have]
        if not todo:
            continue
        srv_enroll, policy, enroll = PLAN_SYSTEMS[(s["framework"], s["system"])]
        wait_guard()
        srv, url = None, None
        if srv_enroll:
            port += 1
            srv = start_server(wt, port, srv_enroll, runs / f"server_{tag}.log",
                               NEMOTRON_DIAR if s["system"] in NEMOTRON_SYSTEMS else None)
            url = f"ws://127.0.0.1:{port}"
        env = {**os.environ, "PYTHONPATH": str(wt), "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
               "TMPDIR": str(work / "tmp"), "TOKENIZERS_PARALLELISM": "false"}
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--work", str(work), "--framework",
               s["framework"], "--system", s["system"], "--cond", s["cond"], "--out", str(out), "--clips", *todo]
        if url:
            cmd += ["--url", url, "--policy", policy, "--server-pid", str(srv.pid)]
        if enroll:
            cmd += ["--enroll", enroll]
        print(f"=== {tag}: {len(todo)} clips {time.strftime('%H:%M:%S')} load {os.getloadavg()[0]:.2f}", flush=True)
        try:
            with (runs / f"driver_{tag}.log").open("a") as f:
                subprocess.run(cmd, cwd=str(wt), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=4 * 3600)
        finally:
            if srv is not None:
                srv.terminate()
                try:
                    srv.wait(20)
                except subprocess.TimeoutExpired:
                    srv.kill()
        time.sleep(3)
    print("QUEUE DONE", flush=True)


# ============================================================================ Dp: predictive trigger, offline
def cmd_predictive(a):
    """Head (c)'s multi-horizon bins on each two-party clip (mixed mono + both parties' per-channel Silero tracks,
    DYADIC section 4/8's two-channel input), run causally from the clip start; decisions = predictive trigger (rising
    edge of P(bin1) < t1 AND P(bin2) < t2, committed at the 160 ms chunk end, 2 s refractory) OR the user channel's
    Pipecat-VAD silence >= k (committed at the 32 ms chunk end). Offline: decision times, no delivery lag."""
    import torch
    sys.path.insert(0, str(ROOT / "scripts" / "research"))
    torch.set_num_threads(2)
    import bench_turnbench as TB
    import eval_stage1 as E
    from audioforge.baselines import turn as B
    from audioforge.data import ToneLanguage
    from audioforge.train import load_model
    import bench_turn_dyadic as BD
    refs = load_clips(a.work)
    model = load_model(str(ROOT / PRED_HEAD), "cpu").eval()
    name = next(k for k, v in model.head_cfg.items() if v["type"] == "turn")
    vad = B.SileroVAD(BD.silero_path())
    t1, t2, k_s = PRED_POINT
    out = Path(a.work) / "runs" / "offline_Dp.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, r in refs.items():
        if r["set"] == "ami":
            continue
        ch_path = Path(a.work) / "clips" / f"{n}.user.wav"
        mono = _wav_read(Path(a.work) / "clips" / f"{n}.mono.wav")
        # both channels: user = the saved user channel, other = mono - user (mix_mono is a clipped sum)
        user = _wav_read(ch_path)
        other = np.clip(mono - user, -1, 1)
        t0 = time.perf_counter()
        pu, po = vad.probs(user), vad.probs(other)
        Tc = ToneLanguage.n_frames(len(mono))
        tr = {0: TB.frame_track(pu, Tc), 1: TB.frame_track(po, Tc)}
        mh = np.zeros((Tc, model.heads[name].mh.out_features), np.float32)
        with torch.no_grad():  # bench_turnbench.stage_head's windowing, from the clip start
            for s0, s1, keep in TB.segments(len(mono)):
                xs = mono[s0:s1]
                T = ToneLanguage.n_frames(len(xs))
                f0 = int(round(s0 / SR / 0.08))
                cols = np.zeros((T, 4), np.float32)
                for j in (0, 1):
                    seg = tr[j][f0: f0 + T]
                    cols[: len(seg), j] = seg
                act = cols[:, 0].copy()
                ax = []
                E.turn_scores_given_act(model, name, [dict(audio=xs, spk_act=act, spk_targets=cols)], [act], 1,
                                        [cols], [0], aux=ax)
                fk = int(round(keep / SR / 0.08))
                n_ = min(len(ax[0]) - fk, Tc - (f0 + fk))
                if n_ > 0:
                    mh[f0 + fk: f0 + fk + n_] = np.asarray(ax[0])[fk: fk + n_]
        dt = time.perf_counter() - t0
        emit = (np.arange(Tc) // 2 + 1) * 2 * 0.08  # 160 ms chunk end of each 80 ms frame
        cond = (mh[:, 0] < t1) & (mh[:, 1] < t2)
        rises = np.nonzero(cond & ~np.concatenate([[False], cond[:-1]]))[0]
        pred, last = [], -np.inf
        for t in rises:
            if emit[t] - last >= TB.REFRACTORY:
                pred.append(round(float(emit[t]), 3))
                last = emit[t]
        sp = B.speech_chunks_pipecat(B.pipecat_vad(pu))  # the user channel's Pipecat-VAD speech chunks
        sil = TB.timeout_events(np.asarray(sp, bool), TB.CHUNK, k_s)
        dec = TB.merge_events(sorted(pred), sorted(sil))
        lines.append({"clip": n, "set": r["set"], "cond": "two_channel", "framework": "offline", "system": "Dp",
                      "raw": {"responses": dec, "predictive": pred, "silero": sil, "audio_s": len(mono) / SR,
                              "total_s": len(mono) / SR, "compute_s": round(dt, 2)}})
        print(f"  {n}: {len(pred)} predictive + {len(sil)} silero -> {len(dec)} ({dt:.1f}s)", flush=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in lines))


# ============================================================================ report
def score_record(rec: dict, ref: dict) -> dict:
    from audioforge.e2e_metrics import score_clip
    from audioforge.metrics import edit_distance
    from audioforge.teachers import normalize_text
    raw = rec["raw"]
    turns = [tuple(t) for t in ref["user_turns"]]
    s = score_clip(raw["responses"], turns, scored=ref["scored"], user_intervals=ref["user_intervals"])
    s["audio_s"] = raw["audio_s"]
    if ref["set"] == "ami":
        s["pre_onset"] = sum(1 for t in raw["responses"] if t < turns[0][0])
    reft = ref.get("text_user") if rec["cond"] == "user" else ref.get("text_mono")
    if ref.get("has_text") and reft is not None and "finals" in raw:
        r, h = normalize_text(reft).split(), normalize_text(" ".join(f["text"] for f in raw["finals"])).split()
        s["wer_err"], s["wer_n"] = edit_distance(r, h), len(r)
    onset = ref.get("first_onset_user") if rec["cond"] == "user" else ref.get("first_onset_mono")
    if raw.get("first_text_t") is not None and onset is not None:
        s["first_text_ms_after_onset"] = round((raw["first_text_t"] - onset) * 1000)
    return s


def load_records(work: Path) -> list[dict]:
    recs = {}
    for p in sorted((Path(work) / "runs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = (r["framework"], r["system"], r["set"], r["cond"], r["clip"])
            prev = recs.get(key)
            # a rerun replaces a flagged run; a flagged rerun replaces nothing but is kept as the only one if alone
            if prev is None or (prev.get("flag_rerun") and not r.get("flag_rerun")) or \
                    (prev.get("flag_rerun") and r.get("flag_rerun") and r.get("wall_start", 0) > prev.get("wall_start", 0)):
                if prev is not None:
                    r["replaced_flagged"] = True
                recs[key] = r
    return list(recs.values())


def group_metrics(items: list[tuple[dict, dict]]) -> dict:
    from audioforge.e2e_metrics import pooled
    sc = [s for _, s in items]
    m = pooled(sc)
    we, wn = sum(s.get("wer_err", 0) for s in sc), sum(s.get("wer_n", 0) for s in sc)
    m["wer"] = round(we / wn, 4) if wn else None
    ft = [s["first_text_ms_after_onset"] for s in sc if "first_text_ms_after_onset" in s]
    m["first_text_ms_median"] = round(float(np.median(ft))) if ft else None
    recs = [r for r, _ in items]
    cpu_d = [r["usage"]["driver"]["cpu_pct"] for r in recs if r.get("usage", {}).get("driver", {}).get("cpu_pct") is not None]
    cpu_s = [r["usage"]["server"]["cpu_pct"] for r in recs if r.get("usage", {}).get("server", {}).get("cpu_pct") is not None]
    m["cpu_pct_driver_mean"] = round(float(np.mean(cpu_d)), 1) if cpu_d else None
    m["cpu_pct_server_mean"] = round(float(np.mean(cpu_s)), 1) if cpu_s else None
    m["rss_mb_driver_peak"] = max((r["usage"]["driver"]["rss_mb_peak"] for r in recs if r.get("usage", {}).get("driver")), default=None)
    m["rss_mb_server_peak"] = max((r["usage"]["server"]["rss_mb_peak"] for r in recs if r.get("usage", {}).get("server")), default=None)
    st = [r["raw"].get("server_stats") or {} for r in recs]
    rt = [x["rtf"] for x in st if x.get("rtf") is not None]
    m["server_rtf_median"] = round(float(np.median(rt)), 3) if rt else None
    m["server_rtf_max"] = round(float(max(rt)), 3) if rt else None
    bl = [x["backlog_ms_max"] for x in st if x.get("backlog_ms_max") is not None]
    m["server_backlog_ms_max"] = round(float(max(bl)), 1) if bl else None
    m["n_flagged_rtf_gt_1"] = sum(bool(r.get("flag_rerun")) for r in recs)
    m["load1_start_range"] = [min((r.get("load1_start", 0) for r in recs), default=None),
                              max((r.get("load1_start", 0) for r in recs), default=None)]
    return m


def clean_checks(recs: list[dict]) -> dict:
    pc = [r for r in recs if r["framework"] == "pipecat"]
    lk = [r for r in recs if r["framework"] == "livekit"]
    out = {"pipecat_runs": len(pc),
           "pipecat_endframe": sum("EndFrame" in r["raw"].get("finished", []) for r in pc),
           "pipecat_violations": sum(len(r["raw"].get("violations", [])) for r in pc),
           "pipecat_errors": sum(len(r["raw"].get("pipeline_errors", [])) for r in pc),
           "pipecat_warnings": sum(len(r["raw"].get("log_warnings", [])) for r in pc),
           "livekit_runs": len(lk),
           "livekit_errors": sum(len(r["raw"].get("errors", [])) for r in lk),
           "livekit_warnings": sum(len(r["raw"].get("log_warnings", [])) for r in lk),
           "livekit_server_sessions_not_1": sum(1 for r in lk if r["system"] != "B" and r["raw"].get("server_sessions") != 1)}
    warn_kinds = {}
    for r in recs:
        for w in r["raw"].get("log_warnings", []):
            k = f"{r['framework']}:{r['system']}:{w['msg'][:80]}"
            warn_kinds[k] = warn_kinds.get(k, 0) + 1
    out["warning_kinds"] = dict(sorted(warn_kinds.items(), key=lambda kv: -kv[1])[:20])
    return out


def components_live(recs: list[dict]) -> dict:
    """Per framework x system: component compute per audio second from the live runs (driver-side timers for A/B,
    the server's debug sums for C/D), process CPU per audio second, framework overhead."""
    out = {}
    for fw_sys in sorted({(r["framework"], r["system"]) for r in recs if r["framework"] != "offline"}):
        rs = [r for r in recs if (r["framework"], r["system"]) == fw_sys]
        audio = sum(r["raw"]["total_s"] for r in rs)
        comp = {}
        for r in rs:
            for k, v in (r.get("components") or {}).items():
                c = comp.setdefault(k, {"n": 0, "wall_ms_sum": 0.0, "cpu_ms_sum": 0.0, "walls": []})
                c["n"] += v["n"]
                c["wall_ms_sum"] += v["wall_ms_sum"]
                c["cpu_ms_sum"] += v["cpu_ms_sum"]
                c["walls"] += [x["wall_ms"] for x in r.get("component_calls", {}).get(k, [])]
            st = r["raw"].get("server_stats") or {}
            for k in ("asr_ms_sum", "diar_ms_sum", "turn_ms_sum", "silero_ms_sum", "enroll_ms_sum"):
                if k in st:
                    c = comp.setdefault("server_" + k[:-7], {"n": 0, "wall_ms_sum": 0.0, "cpu_ms_sum": None, "walls": []})
                    c["wall_ms_sum"] += st[k]
                    c["n"] += st.get("n_chunks", 0)
            if "proc_s" in st:
                c = comp.setdefault("server_total", {"n": 0, "wall_ms_sum": 0.0, "cpu_ms_sum": None, "walls": []})
                c["wall_ms_sum"] += st["proc_s"] * 1000
        res = {}
        for k, c in comp.items():
            w = np.array(c["walls"]) if c["walls"] else None
            res[k] = {"n_calls": c["n"], "rtf_wall": round(c["wall_ms_sum"] / 1000 / audio, 4),
                      "rtf_cpu": None if c["cpu_ms_sum"] is None else round(c["cpu_ms_sum"] / 1000 / audio, 4),
                      "call_ms_p50": round(float(np.percentile(w, 50)), 2) if w is not None else None,
                      "call_ms_p95": round(float(np.percentile(w, 95)), 2) if w is not None else None,
                      "calls_per_audio_min": round(c["n"] / audio * 60, 1)}
        cpu_d = sum((r.get("usage", {}).get("driver", {}).get("cpu_s") or 0) for r in rs)
        cpu_s = sum((r.get("usage", {}).get("server", {}).get("cpu_s") or 0) for r in rs)
        comp_cpu = sum(c["cpu_ms_sum"] for c in comp.values() if c["cpu_ms_sum"] is not None) / 1000
        res["_process"] = {"audio_s": round(audio, 1), "driver_cpu_s_per_audio_s": round(cpu_d / audio, 4),
                           "server_cpu_s_per_audio_s": round(cpu_s / audio, 4) if cpu_s else None,
                           "driver_cpu_minus_components_per_audio_s": round((cpu_d - comp_cpu) / audio, 4),
                           "rss_mb_driver_peak": max((r["usage"]["driver"]["rss_mb_peak"] for r in rs
                                                      if r.get("usage", {}).get("driver")), default=None),
                           "rss_mb_server_peak": max((r["usage"]["server"]["rss_mb_peak"] for r in rs
                                                      if r.get("usage", {}).get("server")), default=None)}
        out[f"{fw_sys[0]}:{fw_sys[1]}"] = res
    return out


PAIRS = {"pipecat": [("C", "A"), ("D", "A"), ("D", "C"), ("CN", "A"), ("DN", "A"), ("CN", "C"), ("DN", "D")],
         "livekit": [("C", "B"), ("D", "B"), ("D", "C"), ("CN", "B"), ("DN", "B"), ("CN", "C"), ("DN", "D")]}


def cmd_report(a):
    from audioforge.e2e_metrics import paired_bootstrap
    work = Path(a.work)
    refs = load_clips(work)
    recs = load_records(work)
    scored = [(r, score_record(r, refs[r["clip"]])) for r in recs]
    groups = {}
    for r, s in scored:
        groups.setdefault((r["set"], r["cond"], r["framework"], r["system"]), []).append((r, s))
    table = {}
    for k, items in sorted(groups.items()):
        table["|".join(k)] = group_metrics(items)
    pairs = {}
    for (st, cond, fw, _), _items in groups.items():
        for x, y in PAIRS.get(fw, []):
            gx, gy = groups.get((st, cond, fw, x)), groups.get((st, cond, fw, y))
            if not gx or not gy:
                continue
            dx = {r["clip"]: s for r, s in gx}
            dy = {r["clip"]: s for r, s in gy}
            common = sorted(set(dx) & set(dy))
            key = f"{st}|{cond}|{fw}|{x}-{y}"
            if key in pairs or not common:
                continue
            pairs[key] = {stat: paired_bootstrap([dx[c] for c in common], [dy[c] for c in common], stat, n_boot=a.n_boot)
                          for stat in ("dead_air_ms_median", "missed_3s", "missed_6s", "cut_ins_per_clip")}
    # Dp (offline) vs the live systems' decision on the two-party sets
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "table": table, "paired": pairs,
           "clean_checks": clean_checks([r for r in recs if r["framework"] != "offline"]),
           "components_live": components_live(recs),
           "clips": {n: {k: v for k, v in r.items() if k in ("set", "conversation", "start", "dur", "human_channel")}
                     | {"n_scored_ends": sum(r["scored"]), "n_agent_ends": len(r.get("agent_ends", []))}
                     for n, r in refs.items()},
           "per_clip": {f"{r['set']}|{r['cond']}|{r['framework']}|{r['system']}|{r['clip']}":
                        {k: v for k, v in s.items() if k != "ends"} | {"ends": s["ends"]} for r, s in scored}}
    comp = work / "components.json"
    if comp.exists():
        res["components_isolated"] = json.loads(comp.read_text())
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    (work / "tables.md").write_text(markdown_tables(res))
    for k, m in table.items():
        print(k, {q: m[q] for q in ("n_clips", "n_ends", "dead_air_ms_median", "dead_air_ms_p90", "missed_3s", "missed_6s",
                                    "cut_ins", "wer", "first_text_ms_median", "server_rtf_max", "n_flagged_rtf_gt_1")})
    for k, v in pairs.items():
        print(k, {s: (x["delta"], x["ci"]) for s, x in v.items()})


NAMES = {"A": "A Pipecat default (Silero + smart-turn v3.2 + Whisper small)",
         "B": "B LiveKit default (Silero + EnglishModel + Whisper small)",
         "C": "C ours, timeout 1000 (product default)", "D": "D ours, hybrid_dyn + after_agent_arm",
         "CN": "CN ours + Nemotron-3-Diarization 100M, timeout 1000",
         "DN": "DN ours + Nemotron-3-Diarization 100M, hybrid_dyn + after_agent_arm",
         "Dp": "Dp ours, predictive OR Silero (offline, two-channel)"}


def _f(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.0f} %" if pct else f"{x:g}"


def markdown_tables(res: dict) -> str:
    out = []
    t = res["table"]
    for st in ("ami", "turnbench", "oto"):
        for cond in ("mono", "user", "two_channel"):
            rows = [(k, v) for k, v in t.items() if k.startswith(f"{st}|{cond}|")]
            if not rows:
                continue
            out.append(f"\n#### {st} / {cond}\n")
            out.append("| framework | system | clips / ends | dead air median / P90 ms | missed 3 s / 6 s | cut-ins (per min) "
                       "| WER | first text ms | CPU % driver / server | peak RSS MB driver / server | server RTF med / max, backlog max ms |")
            out.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for k, m in rows:
                _, _, fw, sy = k.split("|")
                out.append(f"| {fw} | {NAMES.get(sy, sy)} | {m['n_clips']} / {m['n_ends']} | {_f(m['dead_air_ms_median'])} / "
                           f"{_f(m['dead_air_ms_p90'])} | {_f(m['missed_3s'], 1)} / {_f(m['missed_6s'], 1)} | {m['cut_ins']} "
                           f"({_f(m['cut_ins_per_min'])}) | {_f(m['wer'], 1) if m['wer'] is not None else '–'} | "
                           f"{_f(m['first_text_ms_median'])} | {_f(m['cpu_pct_driver_mean'])} / {_f(m['cpu_pct_server_mean'])} | "
                           f"{_f(m['rss_mb_driver_peak'])} / {_f(m['rss_mb_server_peak'])} | {_f(m['server_rtf_median'])} / "
                           f"{_f(m['server_rtf_max'])}, {_f(m['server_backlog_ms_max'])} |")
    out.append("\n#### paired differences (x - y), bootstrap 95 % CI over clips\n")
    out.append("| set / cond / framework | x - y | dead air median ms | missed 3 s (pts) | missed 6 s (pts) | cut-ins per clip |")
    out.append("|---|---|---|---|---|---|")
    for k, v in res["paired"].items():
        st, cond, fw, xy = k.split("|")
        cell = lambda d, pct=False: "–" if d["delta"] is None else (  # noqa: E731
            f"{100 * d['delta']:+.0f} [{100 * d['ci'][0]:+.0f}, {100 * d['ci'][1]:+.0f}]" if pct else
            f"{d['delta']:+g} [{d['ci'][0]:+g}, {d['ci'][1]:+g}]")
        out.append(f"| {st} / {cond} / {fw} | {xy} | {cell(v['dead_air_ms_median'])} | {cell(v['missed_3s'], 1)} | "
                   f"{cell(v['missed_6s'], 1)} | {cell(v['cut_ins_per_clip'])} |")
    return "\n".join(out) + "\n"


# ============================================================================ components (isolated)
def cmd_components(a):
    """Each component alone on the same clips, in one process at 2 threads, replaying the calls the live runs made
    (smart-turn: fixed 8 s inputs; Whisper: the live segment lengths cut from the clip; LiveKit EOU: the live call
    count on the clip's transcript prefixes; Silero: every 32 ms chunk). Wall and process CPU per audio second, per
    call p50 / p95, RSS delta. Our components come from the server's debug sums in the live runs (components_live)."""
    import psutil
    import torch
    torch.set_num_threads(2)
    work = Path(a.work)
    recs = load_records(work)
    refs = load_clips(work)
    proc = psutil.Process()
    res = {}
    clips = [r for r in recs if r["framework"] == "pipecat" and r["system"] == "A"]
    lkb = [r for r in recs if r["framework"] == "livekit" and r["system"] == "B"]
    if a.max_clips:
        clips, lkb = clips[: a.max_clips], lkb[: a.max_clips]

    def bench(name, calls, audio_s):
        """calls: list of zero-arg callables."""
        r0 = proc.memory_info().rss
        walls, cpus = [], []
        for f in calls:
            t0, c0 = time.perf_counter(), time.process_time()
            f()
            walls.append((time.perf_counter() - t0) * 1000)
            cpus.append((time.process_time() - c0) * 1000)
        w, c = np.array(walls), np.array(cpus)
        res[name] = {"n_calls": len(calls), "audio_s": round(audio_s, 1), "rtf_wall": round(w.sum() / 1000 / audio_s, 5),
                     "rtf_cpu": round(c.sum() / 1000 / audio_s, 5), "call_ms_p50": round(float(np.percentile(w, 50)), 3),
                     "call_ms_p95": round(float(np.percentile(w, 95)), 3), "rss_delta_mb": round((proc.memory_info().rss - r0) / 2 ** 20, 1),
                     "cpu_pct_during_calls": round(100 * c.sum() / max(w.sum(), 1e-9), 1)}
        print(name, res[name], flush=True)

    # Silero (Pipecat's analyzer model: 512-sample chunks at 16 kHz)
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    audio_all = [(_wav_read(work / "clips" / f"{r['clip']}.{r['cond']}.wav"), r) for r in clips]
    tot = sum(len(x) for x, _ in audio_all) / SR
    va = SileroVADAnalyzer(sample_rate=SR)
    va.set_sample_rate(SR)
    calls = []
    for x, _ in audio_all:
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        calls += [(lambda b=pcm[i:i + 1024]: va.voice_confidence(b)) for i in range(0, len(pcm) - 1024, 1024)]
    bench("A/B vad_silero (per 32 ms chunk)", calls, tot)
    # smart-turn v3.2: same number of calls as live, on 8 s windows ending at the live call times
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    st = LocalSmartTurnAnalyzerV3()
    calls = []
    for x, r in audio_all:
        n = len(r.get("component_calls", {}).get("turn_smart_turn", []))
        for j in range(n):
            e = min(len(x), int(SR * (1 + j * 3)))
            seg = x[max(0, e - 8 * SR): e]
            calls.append(lambda s=seg: st._predict_endpoint(s))
    bench("A smart-turn v3.2 (per call; 8 s input)", calls or [lambda: st._predict_endpoint(np.zeros(8 * SR, np.float32))], tot)
    # Whisper small: the live segment lengths, cut from the clip
    m = whisper_model()
    calls = []
    for x, r in audio_all:
        for c in r.get("component_calls", {}).get("stt_whisper", []):
            L = int(c.get("audio_s", 2.0) * SR)
            calls.append(lambda s=x[:L].copy(): whisper_text(m, s))
    bench(f"A/B stt faster-whisper-{WHISPER_MODEL} (per segment)", calls, tot)
    # LiveKit EOU EnglishModel: the live call count per clip, on growing transcript prefixes
    ex = _LocalEOUExecutor()
    calls = []
    lk_tot = sum(r["raw"]["total_s"] for r in lkb) or tot
    for r in lkb:
        ref = refs[r["clip"]]
        words = (ref.get("text_mono") or "hello there how are you doing today").split()
        n = len(r.get("component_calls", {}).get("turn_livekit_eou", []))
        for j in range(n):
            txt = " ".join(words[: max(1, int(len(words) * (j + 1) / max(n, 1)))][-60:])
            data = json.dumps({"chat_ctx": [{"role": "user", "content": txt}]}).encode()
            calls.append(lambda d=data: ex.r.run(d))
    if calls:
        bench("B livekit EnglishModel (per call)", calls, lk_tot)
    ours_components(work, refs, res, a.max_clips or 4)
    (work / "components.json").write_text(json.dumps(res, indent=1))


def ours_components(work: Path, refs: dict, res: dict, n_clips: int = 4):
    """Our server's parts, in process (audioforge.serve.Engine / Session, no websocket), 2 threads, 20 ms blocks as a
    paced client sends them, on the first two-party mono clips: the shared ASR encoder chunk step, RNNT greedy
    decode, VAD head, turn head (+ its speaker-conditioned encoder pass), Sortformer diarizer, Silero branch and the
    after_agent_arm binder, for C (timeout) and D (hybrid_dyn + arm). Wall and process CPU per audio second."""
    import torch
    import audioforge.serve as S
    from audioforge.model import StreamingSession
    torch.set_num_threads(2)
    tm = Timers()
    eng = S.Engine.load(str(ROOT / "runs/stage1_served.afm"), str(ROOT / "runs/nemo_sortformer_v2.afm"), threads=2,
                        silero=str(ROOT / "data/silero/silero_vad_v5.onnx"), enroll="after_agent_arm")
    enc = eng.asr.encoder
    orig_step = enc.stream_step

    def step(*a_, **k_):
        t0, c0 = time.perf_counter(), time.process_time()
        r = orig_step(*a_, **k_)
        tm.add("turn_cond_encoder" if "spk_act" in k_ else "asr_encoder", (time.perf_counter() - t0) * 1000,
               (time.process_time() - c0) * 1000)
        return r
    enc.stream_step = step
    tm.wrap(StreamingSession, "_decode", "asr_rnnt_decode")
    tm.wrap(type(eng.asr.heads[eng.vad_name]), "forward", "vad_head")
    tm.wrap(type(eng.asr.heads[eng.turn_name]), "step", "turn_head")
    from audioforge.streaming_diar import StreamingDiarizer
    tm.wrap(StreamingDiarizer, "feed", "diar_sortformer")
    tm.wrap(S.SileroSilence, "feed", "silero_branch")
    tm.wrap(S.ArmBinder, "update", "arm_binder")
    names = [n for n, r in refs.items() if r["set"] != "ami"][:n_clips]
    for tag, pol in (("C timeout", "timeout"), ("D hybrid_dyn+arm", "hybrid_dyn")):
        tm.reset()
        tot, proc = 0.0, 0.0
        for n in names:
            x = _wav_read(work / "clips" / f"{n}.mono.wav")
            eng.enroll = "after_agent_arm" if pol != "timeout" else "dominant"
            sess = S.Session(eng, S.SessionConfig(turn_policy=pol))
            ends = sorted(refs[n].get("agent_ends", [])) if pol != "timeout" else []
            k, blk = 0, int(0.02 * SR)
            t0 = time.perf_counter()
            for i in range(0, len(x), blk):
                while k < len(ends) and i / SR >= ends[k]:
                    sess.arm_enrollment("agent_end")
                    k += 1
                sess.process(x[i:i + blk])
            sess.finish()
            proc += time.perf_counter() - t0
            tot += len(x) / SR
        summ = tm.summary(tot)
        summ["session_total"] = {"rtf_wall": round(proc / tot, 4)}
        res[f"ours {tag} (in-process, per component)"] = summ
        print(tag, json.dumps(summ), flush=True)


# ============================================================================ main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        rest = argv[2:] if len(argv) > 1 and argv[1] == "--" else argv[1:]
        return cmd_serve(None, rest)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work", default=str(WORK))
    p.add_argument("--n-tb", type=int, default=16)
    p.add_argument("--n-oto", type=int, default=16)
    p.add_argument("--length", type=float, default=35.0)
    r = sub.add_parser("run")
    r.add_argument("--work", default=str(WORK))
    r.add_argument("--framework", choices=["pipecat", "livekit"], required=True)
    r.add_argument("--system", choices=["A", "B", "C", "D", "CN", "DN"], required=True)
    r.add_argument("--cond", choices=["mono", "user"], default="mono")
    r.add_argument("--clips", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--url")
    r.add_argument("--policy")
    r.add_argument("--enroll")
    r.add_argument("--server-pid", type=int)
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-flagged", action="store_true")
    q = sub.add_parser("queue")
    q.add_argument("--work", default=str(WORK))
    q.add_argument("--wt", required=True, help="clean git worktree whose code the server and adapters run")
    q.add_argument("--port", type=int, default=8900)
    q.add_argument("--sets", nargs="*")
    q.add_argument("--systems", nargs="*")
    q.add_argument("--frameworks", nargs="*")
    q.add_argument("--max-clips", type=int)
    q.add_argument("--rerun-flagged", action="store_true", default=False)
    d = sub.add_parser("predictive")
    d.add_argument("--work", default=str(WORK))
    c = sub.add_parser("components")
    c.add_argument("--work", default=str(WORK))
    c.add_argument("--max-clips", type=int)
    o = sub.add_parser("report")
    o.add_argument("--work", default=str(WORK))
    o.add_argument("--out", default=str(ROOT / "runs" / "e2e_final.json"))
    o.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "run": cmd_run, "queue": cmd_queue, "predictive": cmd_predictive,
            "report": cmd_report, "components": cmd_components}[a.cmd](a)


if __name__ == "__main__":
    main()
