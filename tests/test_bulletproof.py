"""Bulletproofing the served front end (research/archive/BULLETPROOF.md): audio, protocol, model/state, watchdog and
determinism robustness of audioforge/serve.py, with tiny random models (the real-model runs are in
scripts/bulletproof.py). Every test asserts the same three things in some form: no crash / hang, a structured
``error`` message for every failure, and a bounded, valid output."""
import asyncio
import importlib.util
import json
import math
import time
import os
import urllib.request
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
import torch

import audioforge.serve as S
from audioforge.serve import (ERROR_CODES, FRAME_MS, MAX_INBOX_S, SHED_DIAR_MS, SHED_PARTIAL_MS, SR, STAT_KEEP,
                              ArmBinder, Engine, Session, SessionConfig, TimeoutPolicy, VoiceBinder, decode_pcm,
                              error_msg, rss_mb, serve, validate)

ROOT = Path(__file__).resolve().parents[1]
# wall-clock bounds are loose by default (the suite runs on a loaded laptop next to training jobs); the tight,
# real-time bounds apply with BULLETPROOF_STRICT_TIMING=1 (what a quiet machine achieves is recorded in each test)
SLACK = 1.0 if os.environ.get("BULLETPROOF_STRICT_TIMING") == "1" else 4.0
websockets = pytest.importorskip("websockets")


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _h()
_shipped = importlib.util.spec_from_file_location("test_serve_shipped_helpers", ROOT / "tests" / "test_serve_shipped.py")
SH = importlib.util.module_from_spec(_shipped)
_shipped.loader.exec_module(SH)


# --------------------------------------------------------------------------- helpers
def _speech(sec, seed=0, amp=0.2):
    return (np.random.default_rng(seed).standard_normal(int(SR * sec)) * amp).astype(np.float32)


def _pattern(pattern, seed=0):
    """[(seconds, amplitude), ...] -> float32 audio (amplitude 0 = digital silence)."""
    rng = np.random.default_rng(seed)
    return np.concatenate([(rng.standard_normal(int(SR * d)) * a).astype(np.float32) for d, a in pattern])


def _run(eng, x, cfg=None, block=320, **kw):
    """Feed ``x`` in blocks through one Session -> (session, messages); every message validates."""
    s = Session(eng, cfg or SessionConfig())
    msgs = []
    for i in range(0, len(x), block):
        msgs += s.process(x[i:i + block], **kw)
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    return s, msgs


def _decisions(msgs, spk=True):
    """The decision-level events (what a voice agent acts on), without timing statistics. ``spk=False`` leaves
    the frames' speakers / primary out: a frame carries the *latest* diarizer row at emission, which depends on
    how the audio was blocked (a 3 s block emits its 37 frames with the row of its last diarizer frame)."""
    out = []
    for m in msgs:
        if m["type"] in ("frame", "frames"):
            for it in ([m] if m["type"] == "frame" else m["items"]):
                out.append(("frame", it["t"], it["vad"], it["eot"]) +
                           ((tuple(it["speakers"]), it["primary"]) if spk else ()))
        elif m["type"] in ("turn_end", "final", "enrolled", "language"):
            out.append((m["type"], m["t"], m.get("policy"), m.get("text"), m.get("speaker"), m.get("p"),
                        m.get("silence_ms")))
    return out


def _finite(msgs):
    for m in msgs:
        for k, v in m.items():
            if isinstance(v, float):
                assert math.isfinite(v), (k, m)
            if isinstance(v, list):
                assert all(math.isfinite(p) for p in v if isinstance(p, float)), (k, m)
    return True


class _Server:
    """A serve() instance on a free port inside an asyncio.run body: ``async with _Server(eng) as srv``."""

    def __init__(self, eng):
        self.eng, self.port, self.task, self.stop = eng, None, None, None

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        port_f, self.stop = loop.create_future(), loop.create_future()
        self.task = asyncio.create_task(serve(self.eng, "127.0.0.1", 0, ready_event=port_f.set_result, stop=self.stop))
        self.port = await port_f
        return self

    async def __aexit__(self, *a):
        if not self.stop.done():
            self.stop.set_result(None)
        await self.task

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as r:
            return json.loads(r.read())

    async def health(self) -> dict:
        """GET /health from a worker thread (a blocking urlopen inside the event loop would starve the server)."""
        return await asyncio.to_thread(self._get, "/health")


async def _recv_all(ws, timeout=20.0):
    got = []
    try:
        while True:
            got.append(json.loads(await asyncio.wait_for(ws.recv(), timeout)))
    except websockets.exceptions.ConnectionClosed:
        pass
    return got


async def _stream(url, x, cfg=None, chunk=1600, pace=0.0, controls=(), end=True, sample_rate=SR):
    """Open a session, stream int16 ``x`` (float, 16 kHz unless the config says otherwise) -> all messages."""
    async with websockets.connect(url, max_size=2 ** 22) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert ready["type"] == "ready"
        if cfg:
            await ws.send(json.dumps({"type": "config", **cfg}))
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), chunk * 2):
            await ws.send(pcm[i:i + chunk * 2])
            for t_at, msg in controls:
                if i // 2 <= t_at * sample_rate < i // 2 + chunk:
                    await ws.send(json.dumps(msg))
            if pace:
                await asyncio.sleep(pace)
        if end:
            await ws.send(json.dumps({"type": "end"}))
        return [ready] + await _recv_all(ws)


def _errors(msgs, code=None):
    return [m for m in msgs if m["type"] == "error" and (code is None or m["code"] == code)]


class SlowEngine(H.EnergyEngine):
    """EnergyEngine whose diarizer costs ``cost_rtf`` seconds of wall time per second of audio (a model slower than
    real time); skipped under load shedding like the real one (``feed(skip=True)``)."""
    cost_rtf = 1.5

    def make_diarizer(self):
        d = super().make_diarizer()
        feed = d.feed

        def slow_feed(samples, final=False, skip=False):
            if not skip:
                time.sleep(self.cost_rtf * len(samples) / SR)
                return feed(samples, final)
            out = feed(samples, final)
            d.skipped = getattr(d, "skipped", 0) + len(out)
            return torch.zeros_like(out)
        d.feed = slow_feed
        return d


class SwapDiarizer(H.EnergyDiarizer):
    """Speaker A (column 0) on 0-2 s, speaker B (column 1) on 3-5 s, then the columns swap: B keeps talking on
    column 0 from 6 s on (a Sortformer arrival-order swap)."""

    def feed(self, samples, final=False):
        start = self.done
        out = super().feed(samples, final)
        for i in range(len(out)):
            v = start + i
            t = v * FRAME_MS / 1000
            on = bool(out[i, 0] > 0.5)
            out[i] = 0.02
            if on:
                out[i, 1 if 3.0 <= t < 5.0 else 0] = 0.9
        return out


class FailingDiarizer(H.EnergyDiarizer):
    def feed(self, samples, final=False):
        if self.done >= 6:
            raise RuntimeError("diarizer exploded")
        return super().feed(samples, final)



class BoundedEnergyDiarizer(H.EnergyDiarizer):
    """EnergyDiarizer that drops the audio it has consumed: the shared test double keeps every sample ever fed
    (O(n) per call), which made the 30-min soak look like a server slowdown (research/archive/BULLETPROOF.md section 1)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.x0 = 0  # samples dropped from the front of self.x

    def feed(self, samples, final=False):
        from audioforge.data import ToneLanguage
        self.x = np.concatenate([self.x, np.asarray(samples, np.float32)])
        total = self.x0 + len(self.x)
        avail = ToneLanguage.n_frames(total) if final and total else total // 1280
        C, R = self.cfg.chunk_len, self.cfg.chunk_right_context
        out = []
        while avail - self.done >= C + R or (final and avail > self.done):
            n = min(C, avail - self.done)
            for v in range(self.done, self.done + n):
                seg = self.x[v * 1280 - self.x0:(v + 1) * 1280 - self.x0]
                p = np.full(4, 0.02)
                p[0] = 0.9 if len(seg) and np.sqrt(np.mean(seg ** 2)) > 0.01 else 0.02
                out.append(p)
            self.done += n
        drop = max(0, self.done * 1280 - self.x0)
        if drop:
            self.x, self.x0 = self.x[drop:], self.x0 + drop
        return torch.tensor(np.array(out) if out else np.zeros((0, 4)))

class _Eng(H.EnergyEngine):
    """EnergyEngine with a pluggable diarizer class."""
    diar_cls = H.EnergyDiarizer

    def make_diarizer(self):
        return self.diar_cls()


def _energy_engine(diar_cls=H.EnergyDiarizer, model=None, **kw):
    e = _Eng(model or H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, **kw)
    e.diar_cls = diar_cls
    return e


def _fresh(model=None, **kw):
    """A private Engine (tiny ASR + tiny real streaming diarizer): the socket tests assert engine-wide counters
    and mutate limits, which must not leak through test_serve's cached engines."""
    return Engine(model or H._asr_model(), H._diar_model(), name="tiny", threads=1, **kw)


def _quiet_talky_model():
    """Talky transducer (one 'a' per decoded frame) with a VAD head that says 'no speech' everywhere: with the VAD
    gate on, decoding must stop after the hangover."""
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["vad"].net[-1].bias.fill_(-6.0)
    return m


# =========================================================================== 1. audio robustness
def test_error_message_shape_and_validate():
    for code in ERROR_CODES:
        validate(error_msg(code, "x" * 1000, True))
    assert len(error_msg("bad_json", "x" * 1000)["detail"]) == 500
    with pytest.raises(ValueError):
        validate({"type": "error", "code": "made_up", "detail": "", "fatal": False})
    with pytest.raises(ValueError):  # nothing non-finite may reach the wire
        validate({"type": "frame", "t": 0.08, "vad": float("nan"), "eot": None, "speakers": [0, 0, 0, 0], "primary": None})
    with pytest.raises(ValueError):
        validate({"type": "partial", "t": float("inf"), "text": ""})
    validate({"type": "stats", "rtf": 0.1, "chunk_ms_p50": 1, "chunk_ms_p95": 2, "first_partial_ms": None,
              "peak_rss_mb": 1, "degraded": {"nan_input": 3}})


def test_nan_inf_input_is_repaired_reported_and_does_not_poison_the_stream():
    eng = H._engine()
    x = _speech(0.4, seed=1)
    x[100], x[200], x[300] = np.nan, np.inf, -np.inf
    full = np.concatenate([x, _speech(1.6, seed=2)])
    s, msgs = _run(eng, full)
    errs = _errors(msgs, "nan_input")
    assert len(errs) == 1 and not errs[0]["fatal"] and "3 non-finite" in errs[0]["detail"]
    assert msgs[0]["type"] == "error"  # reported with the block that carried the NaNs, before its frames
    frames = [m for m in msgs if m["type"] == "frame"]
    assert len(frames) == H._n_frames(eng.asr, full) and _finite(msgs)
    assert all(0 <= f["vad"] <= 1 for f in frames) and s.degraded == {"nan_input": 3}
    assert msgs[-1]["degraded"] == {"nan_input": 3} and eng.counters["nan_input"] == 3
    # the stream after the bad block equals a clean stream fed the same repaired audio (no state poisoning)
    clean = x.copy()
    clean[[100, 200, 300]] = 0.0
    _, ref = _run(eng, np.concatenate([clean, _speech(1.6, seed=2)]))
    assert _decisions(msgs) == _decisions(ref)


@pytest.mark.parametrize("kind", ["huge", "dc", "quiet", "clipped", "int16_range_floats"])
def test_extreme_amplitudes_stay_finite_and_bounded(kind):
    eng = H._engine()
    base = _speech(2.0, seed=3)
    x = {"huge": base * 1e30, "dc": base + 0.9, "quiet": base * 10 ** (-50 / 20) / base.std(),
         "clipped": np.clip(base * 40, -1, 1), "int16_range_floats": base * 32768}[kind].astype(np.float32)
    s, msgs = _run(eng, x, SessionConfig(turn_policy="both"))
    assert _finite(msgs) and sum(m["type"] == "frame" for m in msgs) == H._n_frames(eng.asr, x)
    if kind in ("huge", "int16_range_floats"):  # beyond +-64: clipped and reported once, nothing else
        assert _errors(msgs, "clipped_input") and s.degraded["clipped_input"] > 0 and len(_errors(msgs)) == 1
    else:
        assert not _errors(msgs)
    assert msgs[-1]["type"] == "stats" and msgs[-2]["type"] == "final"


def test_zero_length_frames_one_sample_blocks_and_empty_streams():
    eng = H._engine()
    s = Session(eng)
    assert s.process(np.zeros(0, np.float32)) == [] and s.process(np.zeros(0, np.float32)) == []
    x = _speech(0.5, seed=4)
    ref = _run(eng, x)[1]
    s = Session(eng)
    msgs = []
    for i in range(len(x)):  # 1-sample blocks with empty blocks in between
        msgs += s.process(x[i:i + 1])
        if i % 1000 == 0:
            msgs += s.process(np.zeros(0, np.float32))
    msgs += s.finish()
    assert _decisions(msgs) == _decisions(ref)
    s = Session(eng)  # no audio at all
    out = s.finish()
    assert [m["type"] for m in out] == ["final", "stats"] and out[-1]["rtf"] == 0.0 and out[0]["text"] == ""
    assert s.finish() == []  # a second flush is a no-op


@pytest.mark.parametrize("block", [1, 112, 3 * SR])
def test_odd_chunk_sizes_give_identical_decisions(block):
    """1 sample, 7 ms and 3 s blocks -> the same frames, turn_ends and finals as 20 ms blocks (bitwise), on the
    scripted diarizer (deterministic turns) and on the tiny real streaming diarizer (frames / rows)."""
    x = _pattern(((1.0, 0.2), (1.6, 0.0), (0.7, 0.2), (1.2, 0.0)), seed=5)
    cfg = dict(turn_policy="both", timeout_ms=480)
    spk = block <= 1280  # blocks longer than a frame emit their frames with the block's last diarizer row

    def by_time(d):  # a block's frames are sent before its turn_ends: compare by audio time, not message order
        return sorted(d, key=lambda e: (e[1], e[0]))

    eng = _energy_engine(model=H._asr_model())
    ref = _decisions(_run(eng, x, SessionConfig(**cfg), block=320)[1], spk)
    got = _decisions(_run(eng, x, SessionConfig(**cfg), block=block)[1], spk)
    assert by_time(got) == by_time(ref) and any(d[0] == "turn_end" for d in ref)
    real = _fresh()
    ref = _decisions(_run(real, x, SessionConfig(**cfg), block=320)[1], spk)
    got = _decisions(_run(real, x, SessionConfig(**cfg), block=block)[1], spk)
    assert by_time(got) == by_time(ref)


def test_same_audio_same_events_across_runs_bitwise():
    x = _pattern(((0.8, 0.3), (1.2, 0.0), (0.9, 0.25), (1.5, 0.0)), seed=6)
    eng = _energy_engine(model=H._asr_model(), debug=True)
    runs = [_decisions(_run(eng, x, SessionConfig(turn_policy="both", timeout_ms=400))[1]) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert sum(1 for d in runs[0] if d[0] == "turn_end") >= 1
    real = _fresh(debug=True)
    runs = [_decisions(_run(real, x, SessionConfig(turn_policy="both", timeout_ms=400))[1]) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_silence_and_noise_minutes_bounded_text_and_no_turns():
    """The VAD gate: a transducer that would emit one token per frame emits at most the hangover's worth on
    non-speech (silence and stationary noise), and no turn rule fires without diarizer speech."""
    gated = _energy_engine(model=_quiet_talky_model(), asr_vad_gate=0.2, asr_vad_hangover_ms=1200)
    plain = _energy_engine(model=_quiet_talky_model())
    for name, x in (("silence", np.zeros(SR * 120, np.float32)), ("noise", _speech(120, seed=7, amp=0.004))):
        s, msgs = _run(gated, x, SessionConfig(turn_policy="both"), block=SR // 2)
        text = "".join(m["text"] for m in msgs if m["type"] == "final")
        assert len(text) <= 15 + 2, (name, len(text))  # hangover 15 frames, then gated
        assert s.asr.n_gated >= 1500 - 20 and not any(m["type"] == "turn_end" for m in msgs), name
        assert sum(m["type"] == "partial" for m in msgs) <= 20
        s2, msgs2 = _run(plain, x, SessionConfig(turn_policy="both"), block=SR // 2)
        assert len("".join(m["text"] for m in msgs2 if m["type"] == "final")) == H._n_frames(plain.asr, x)  # ungated
    # with speech (VAD high) the gate never engages: identical text to the ungated model
    talky = _energy_engine(model=H._talky_asr_model(), asr_vad_gate=0.2)
    x = _pattern(((1.0, 0.2), (0.6, 0.0), (1.0, 0.2)), seed=8)
    a = _run(talky, x)[1]
    b = _run(_energy_engine(model=H._talky_asr_model()), x)[1]
    assert [m["text"] for m in a if m["type"] == "final"] == [m["text"] for m in b if m["type"] == "final"]


def test_resampling_paths_8k_and_48k_and_int16_float32_stereo_payloads():
    """8 kHz / 48 kHz int16, float32 and stereo payloads all yield the 16 kHz frame count of the audio; the
    float32 stereo stream (identical channels) decodes to the int16 mono stream within quantization."""
    eng = _fresh()
    t = np.arange(int(SR * 2.0)) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.3)).astype(np.float32)

    async def go():
        async with _Server(eng) as srv:
            ref = await _stream(srv.url, x)
            res = {"ref": ref}
            x8 = x[::2]  # 8 kHz phone band (crude decimation is fine for the frame count)
            res["8k"] = await _stream(srv.url, x8, {"sample_rate": 8000}, sample_rate=8000)
            x48 = np.interp(np.arange(len(x) * 3) / 3, np.arange(len(x)), x).astype(np.float32)
            res["48k"] = await _stream(srv.url, x48, {"sample_rate": 48000}, sample_rate=48000)
            async with websockets.connect(srv.url, max_size=2 ** 22) as ws:  # float32 stereo, odd frame sizes
                await ws.recv()
                await ws.send(json.dumps({"type": "config", "format": "float32", "channels": 2}))
                st = np.stack([x, x], 1).astype("<f4").tobytes()
                for i in range(0, len(st), 4093):
                    await ws.send(st[i:i + 4093])
                await ws.send(json.dumps({"type": "end"}))
                res["f32st"] = await _recv_all(ws)
            return res

    res = asyncio.run(go())
    n_ref = sum(m["type"] == "frame" for m in res["ref"])
    assert n_ref == H._n_frames(eng.asr, x)
    for k, msgs in res.items():
        assert not _errors(msgs), (k, _errors(msgs))
        assert abs(sum(m["type"] == "frame" for m in msgs) - n_ref) <= 1, k
        assert msgs[-1]["type"] == "stats" and _finite(msgs)
    va = [m["vad"] for m in res["ref"] if m["type"] == "frame"]
    vb = [m["vad"] for m in res["f32st"] if m["type"] == "frame"]
    assert np.allclose(va, vb, atol=0.02)


def test_decode_pcm_formats_and_carry():
    cfg = SessionConfig()
    x, carry, bad = decode_pcm(np.array([16384, -16384, 7], "<i2").tobytes()[:5], cfg)
    assert np.allclose(x, [0.5, -0.5]) and carry == b"\x07" and bad == 0
    cfg = SessionConfig(format="float32", channels=2)
    raw = np.array([0.5, 0.5, np.nan, 3.0, 1.0, -1.0], "<f4").tobytes()
    x, carry, bad = decode_pcm(raw + b"\x00\x00", cfg)
    assert np.allclose(x, [0.5, 0.5, 0.0]) and carry == b"\x00\x00" and bad == 2


@pytest.mark.slow
def test_long_session_flat_memory_bounded_state_no_slowdown():
    """30 min of speech / silence / noise through one session: per-frame state stays bounded, RSS is flat after
    warm-up, and the per-block cost of the last 5 min is not above the first 5 min's."""
    eng = _energy_engine(BoundedEnergyDiarizer, model=H._talky_asr_model())  # talky: stresses the token paths
    s = Session(eng, SessionConfig(turn_policy="both", timeout_ms=800))
    rng = np.random.default_rng(9)
    block = SR // 2
    minutes = 30
    costs, rss, n_msgs, max_partial = [], [], 0, 0
    for i in range(minutes * 120):  # 0.5 s blocks
        t = i / 120
        kind = int(t * 7) % 4  # speech / silence / noise / speech bursts, changing every ~8.6 s
        x = (rng.standard_normal(block) * (0.2 if kind in (0, 3) else 0.003 if kind == 2 else 0.0)).astype(np.float32)
        t0 = time.perf_counter()
        out = s.process(x, backlog_ms=0.0)
        costs.append(time.perf_counter() - t0)
        n_msgs += len(out)
        max_partial = max(max_partial, max((len(m["text"]) for m in out if m["type"] == "partial"), default=0))
        if i % 600 == 599:
            rss.append(rss_mb())
    out = s.finish()
    assert out[-1]["type"] == "stats" and _finite(out)
    n_rows = minutes * 60 * 1000 // FRAME_MS
    assert s.samples == minutes * 60 * SR and n_rows <= len(s.rows) <= n_rows + 1  # (+1: the flush's last frame)
    assert len(s.rows.d) <= 2048 and len(s.prims.d) <= 2048 and len(s.vad_ring.d) <= 2048
    for name in ("chunk_ms", "asr_ms", "diar_ms", "backlog_ms", "diar_emit_lag", "send_lag_ms"):
        assert len(getattr(s, name)) <= STAT_KEEP, name
    assert len(getattr(s.diar, "probs", ())) == 0 and len(s.asr.sig) < 2 * SR
    assert max_partial < 4000 and "segment_cap" not in s.degraded  # turn_ends cut the segments; partials bounded
    first, last = np.median(costs[:600]), np.median(costs[-600:])
    # strict (BULLETPROOF_STRICT_TIMING=1): the last 5 min's median block cost <= 1.5x the first's + 2 ms; by default
    # the ratio gets SLACK too (a medians ratio across 30 min drifts with whatever else the laptop runs)
    assert last <= 1.5 * SLACK * first + 0.002 * SLACK, (first, last)
    # RSS: flat after the first 5 min (allocator warm-up); the last 20 min may not add more than 40 MB (strict), or
    # 40 MB x SLACK by default (allocator-dependent)
    assert rss[-1] - rss[1] < 40 * SLACK, rss
    print(f"\n[long] 30 min: rss {rss} MB, block cost p50 first/last {first * 1e3:.2f}/{last * 1e3:.2f} ms, "
          f"{n_msgs} messages, rtf {out[-1]['rtf']}")


def test_segment_cap_cuts_an_endless_segment_with_a_final():
    """A policy that never fires (head, p ~ 0) on a talky model: the segment is cut with a final every
    MAX_SEGMENT_S so partials stay bounded; reported once as a non-fatal segment_cap."""
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(-8.0)
    eng = _energy_engine(model=m)
    S.MAX_SEGMENT_S, cap = 4.0, S.MAX_SEGMENT_S
    try:
        s, msgs = _run(eng, _speech(13.0, seed=10), SessionConfig(turn_policy="head"), block=SR // 2)
    finally:
        S.MAX_SEGMENT_S = cap
    finals = [m for m in msgs if m["type"] == "final"]
    # 3 caps + the end final; a cap is checked once per block (0.5 s = up to 7 frames past the 50-frame cap)
    assert len(finals) == 4 and all(50 <= len(f["text"]) <= 57 for f in finals[:-1])
    assert not any(m["type"] == "turn_end" for m in msgs)
    assert len(_errors(msgs, "segment_cap")) == 1 and s.degraded["segment_cap"] == 3
    assert "".join(f["text"] for f in finals) == "a" * s.asr.n_frames  # nothing lost or duplicated


def test_back_to_back_speakers_and_column_swap():
    """A -> B -> B on a swapped column: every burst ends in a turn_end with a final, the primary follows the
    active column (0 -> 1 -> 0), nothing crashes and nothing is emitted out of order."""
    eng = _energy_engine(SwapDiarizer)
    x = _pattern(((2.0, 0.2), (1.0, 0.0), (2.0, 0.2), (1.0, 0.0), (2.0, 0.2), (1.5, 0.0)), seed=11)
    s, msgs = _run(eng, x, SessionConfig(timeout_ms=560))
    te = [m for m in msgs if m["type"] == "turn_end"]
    fin = [m for m in msgs if m["type"] == "final"]
    assert 3 <= len(te) <= 4 and len(fin) == len(te) + 1 and _finite(msgs)  # (+1: the end-of-stream final)
    assert [t["t"] for t in te] == sorted(t["t"] for t in te) and all(t["silence_ms"] >= 560 for t in te)
    spk = [f["speaker"] for f in fin]
    assert spk[0] == 0 and 1 in spk and spk[-1] == 0  # A, then B (column 1), then B on the swapped column 0
    prims = [m["primary"] for m in msgs if m["type"] == "frame"]
    assert 1 in prims and prims[-1] == 0
    pol = TimeoutPolicy(1000)  # the primary rule alone: a swap re-binds once the new column dominates the 5 s window
    for _ in range(30):
        pol.update([0.9, 0.02, 0.02, 0.02])
    for _ in range(31):
        pol.update([0.02, 0.9, 0.02, 0.02])
    assert pol.primary == 1


def test_bursts_fire_at_most_one_turn_each_and_never_crash():
    """Cough / laugh-like 60 ms bursts every 2 s on every policy: at most one turn_end per burst, every turn_end
    has its final, finite outputs; with a head that never says end-of-turn, no turn at all."""
    parts = []
    for _ in range(6):
        parts += [(0.06, 0.4), (1.94, 0.0)]
    x = _pattern(parts, seed=12)
    for policy in ("timeout", "head", "both", "hybrid"):  # 'both' may fire the head and the timeout per burst
        s, msgs = _run(_energy_engine(), x, SessionConfig(turn_policy=policy, timeout_ms=480))
        te = [m for m in msgs if m["type"] == "turn_end"]
        assert len(te) <= (12 if policy == "both" else 6) and _finite(msgs), policy
        assert 1 <= sum(m["type"] == "final" for m in msgs) <= len(te) + 1, policy  # 'both': one cut per segment
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(-8.0)
    s, msgs = _run(_energy_engine(model=m), x, SessionConfig(turn_policy="head"))
    assert not any(m["type"] == "turn_end" for m in msgs) and _finite(msgs)


# =========================================================================== 2. protocol robustness
def test_malformed_and_unknown_messages_get_structured_errors_and_the_session_lives():
    eng = _fresh()
    x = _speech(1.0, seed=13)

    async def go():
        async with _Server(eng) as srv:
            async with websockets.connect(srv.url, max_size=2 ** 22) as ws:
                await ws.recv()
                for junk in ("not json", json.dumps([1, 2]), json.dumps({"no": "type"}), json.dumps({"type": 7}),
                             json.dumps({"type": "bogus"}), json.dumps({"type": "config", "timeout_ms": "abc",
                                                                         "eot_threshold": None, "sample_rate": 1,
                                                                         "turn_policy": "x", "channels": 99}),
                             "x" * (S.MAX_TEXT_MESSAGE + 1), json.dumps({"type": "agent_end"}),
                             json.dumps({"type": "enroll"}), "\x00\xff", json.dumps("str")):
                    await ws.send(junk)
                await ws.send((x * 32767).astype("<i2").tobytes())
                await ws.send(json.dumps({"type": "end"}))
                await ws.send(json.dumps({"type": "end"}))  # a duplicate end after the flush is harmless
                return await _recv_all(ws), await srv.health()

    msgs, health = asyncio.run(go())
    codes = [m["code"] for m in msgs if m["type"] == "error"]
    assert codes[:6] == ["bad_json", "bad_message", "bad_message", "bad_message", "unknown_type", "bad_config"]
    assert codes.count("message_too_large") == 1 and codes.count("unsupported") == 2
    assert all(not m["fatal"] for m in msgs if m["type"] == "error")
    bad = [m for m in msgs if m["type"] == "error" and m["code"] == "bad_config"][0]["detail"]
    assert "timeout_ms" in bad and "eot_threshold" in bad and "sample_rate 1" in bad and "channels" in bad
    assert sum(m["type"] == "frame" for m in msgs) == H._n_frames(eng.asr, x) and msgs[-1]["type"] == "stats"
    assert health["ok"] and health["sessions_active"] == 0 and health["counters"]["error_bad_json"] == 2
    assert not eng.conns


def test_config_after_audio_and_duplicate_out_of_order_controls():
    eng = SH._engine(enroll="after_agent_arm")  # Silero fake; turn_policy / sample_rate set before audio only
    x = _pattern(((0.5, 0.0), (1.2, 0.2), (1.0, 0.0)), seed=14)

    async def go():
        async with _Server(eng) as srv:
            ctl = [(0.0, {"type": "agent_end"}), (0.0, {"type": "agent_end"}),  # duplicates before any speech
                   (0.6, {"type": "config", "sample_rate": 8000, "turn_policy": "hybrid_dyn"}),  # late: refused
                   (0.9, {"type": "agent_end"})]  # re-arm during speech
            return await _stream(srv.url, x, {"turn_policy": "timeout", "timeout_ms": 400}, chunk=320, controls=ctl)

    msgs = asyncio.run(go())
    errs = _errors(msgs)
    assert [e["code"] for e in errs] == ["bad_config"] and "sample_rate ignored after" in errs[0]["detail"]
    assert "hybrid_dyn ignored after the first audio" in errs[0]["detail"]
    assert msgs[-1]["type"] == "stats" and msgs[-1]["enrolled"] and msgs[-1]["primary_column"] == 0
    assert sum(m["type"] == "enrolled" for m in msgs) >= 1


def test_odd_byte_binary_frames_and_huge_binary_frame():
    eng = _fresh()
    x = _speech(1.0, seed=15)
    pcm = (x * 32767).astype("<i2").tobytes()

    async def go():
        async with _Server(eng) as srv:
            async with websockets.connect(srv.url, max_size=2 ** 22) as ws:
                await ws.recv()
                cuts = [0, 1, 4, 5, 1003, 1004, 20001, len(pcm)]
                for a, b in zip(cuts, cuts[1:]):
                    await ws.send(pcm[a:b])
                await ws.send(b"")  # an empty binary frame
                await ws.send(json.dumps({"type": "end"}))
                odd = await _recv_all(ws)
            ref = await _stream(srv.url, x)
            async with websockets.connect(srv.url, max_size=2 ** 22) as ws:  # a frame beyond max_size: refused
                await ws.recv()
                try:
                    await ws.send(b"\x00" * (2 ** 22 + 2))
                    await _recv_all(ws, 5)
                except websockets.exceptions.ConnectionClosed:
                    pass
                closed = ws.close_code
            after = await _stream(srv.url, x[:SR // 2])  # the server is fine afterwards
            return odd, ref, closed, after, await srv.health()

    odd, ref, closed, after, health = asyncio.run(go())
    assert _decisions(odd) == _decisions(ref[1:]) and not _errors(odd)
    assert closed == 1009 and after[-1]["type"] == "stats" and health["sessions_active"] == 0


def test_client_disconnect_mid_turn_then_reconnect():
    eng = _fresh()
    x = _speech(3.0, seed=16)

    async def go():
        async with _Server(eng) as srv:
            for _ in range(3):
                ws = await websockets.connect(srv.url)
                await ws.recv()
                await ws.send((x[:SR] * 32767).astype("<i2").tobytes())
                await asyncio.sleep(0.05)
                await ws.close(code=1001)
            await asyncio.sleep(0.2)
            h = await srv.health()
            again = await _stream(srv.url, x)
            return h, again, await srv.health()

    h, again, h2 = asyncio.run(go())
    assert h["sessions_active"] == 0 and h["sessions_total"] == 3 and not eng.conns
    assert again[-1]["type"] == "stats" and sum(m["type"] == "frame" for m in again) == H._n_frames(eng.asr, x)
    assert h2["sessions_total"] == 4


def test_fast_client_backpressure_slow_client_and_fairness():
    """A client that dumps 40 s of audio at once on a model 1.5x slower than real time: the inbox is capped (the
    reader stops reading: TCP backpressure), the watchdog sheds load (reported once as 'overloaded'; the diarizer
    is skipped so the backlog drains), and a second short session and /health are served during the flood."""
    eng = SlowEngine(H._talky_asr_model(), H._diar_model(), name="slow", threads=1)
    x = _speech(40.0, seed=17)
    peak = {"n_in": 0}

    async def watch():
        while True:
            for c in list(eng.conns):
                peak["n_in"] = max(peak["n_in"], c.n_in)
            await asyncio.sleep(0.01)

    async def go():
        async with _Server(eng) as srv:
            w = asyncio.create_task(watch())
            t0 = time.perf_counter()
            flood = asyncio.create_task(_stream(srv.url, x, chunk=SR))
            h, t_health = None, 0.0
            for _ in range(200):  # until the flood is queued (a loaded machine delivers it later)
                await asyncio.sleep(0.05)
                th = time.perf_counter()
                h = await srv.health()
                t_health = time.perf_counter() - th
                if h["backlog_ms_max"] > SHED_DIAR_MS:
                    break
            short = await _stream(srv.url, _speech(0.5, seed=18))
            t_short = time.perf_counter() - t0
            msgs = await flood
            w.cancel()
            return msgs, short, h, t_health, t_short, await srv.health(), time.perf_counter() - t0

    msgs, short, h, t_health, t_short, h2, wall = asyncio.run(go())
    assert msgs[-1]["type"] == "stats" and sum(m["type"] in ("frame", "frames") for m in msgs) >= 1
    assert peak["n_in"] <= (MAX_INBOX_S + 1.0) * SR, peak  # the inbox cap held
    assert h2["counters"].get("inbox_backpressure", 0) > 0 and h2["counters"].get("overloaded", 0) > 0
    # fairness: both served during the flood (quiet machine: t_short ~3 s, t_health ~2 ms)
    assert short[-1]["type"] == "stats" and t_short < 8.0 * SLACK and t_health < 1.0 * SLACK
    assert h["backlog_ms_max"] > SHED_DIAR_MS and h["sessions_active"] >= 1
    st = msgs[-1]
    # 40 s of audio at 1.5x real time would take 60 s unshed; shedding drained it in ~3.3 s on a quiet machine
    assert st["degraded"].get("shed_diar_frames", 0) > 0 and wall < 20 * SLACK
    over = _errors(msgs, "overloaded")
    # engaged on load (one notice; a loaded machine that stretches the run past SHED_NOTICE_S may repeat it)
    assert 1 <= len(over) <= 3 and float(over[0]["detail"].split("rtf ")[1].split(":")[0]) >= S.SHED_RTF
    n_fr = sum(len(m["items"]) if m["type"] == "frames" else 1 for m in msgs if m["type"] in ("frame", "frames"))
    assert n_fr == H._n_frames(eng.asr, x)  # every frame delivered, batched or not
    assert h2["sessions_active"] == 0 and h2["load"] == 0.0 and h2["shed_level"] == 0


def test_concurrent_sessions_are_isolated_fair_and_deterministic():
    """12 concurrent sessions on one compute thread, each with its own audio: every session's decisions (frames'
    vad / eot, turn_ends, finals) equal its solo run bitwise (no cross-talk between sessions), and all finish."""
    eng = _fresh()
    n = 12
    auds = [np.roll(_pattern(((0.8, 0.25), (1.0, 0.0), (0.6, 0.2), (0.8, 0.0)), seed=20 + i), i * 1234) for i in range(n)]
    cfg = {"turn_policy": "both", "timeout_ms": 480}
    # the solo reference sees exactly what the socket path decodes (int16 quantization, /32768)
    q = [(np.clip(a, -1, 1) * 32767).astype("<i2").astype(np.float32) / 32768.0 for a in auds]
    solo = [_decisions(_run(eng, a, SessionConfig(**cfg))[1], spk=False) for a in q]

    async def go():
        async with _Server(eng) as srv:
            t0 = time.perf_counter()
            res = await asyncio.gather(*[_stream(srv.url, a, cfg, chunk=320, pace=0.001) for a in auds])
            return res, time.perf_counter() - t0, await srv.health()

    res, wall, health = asyncio.run(go())
    for i, msgs in enumerate(res):
        assert msgs[-1]["type"] == "stats" and not _errors(msgs), i
        got = _decisions(msgs, spk=False)  # bitwise: block sizes, the compute thread and interleaving change nothing
        assert sorted(got, key=lambda d: (d[1], d[0])) == sorted(solo[i], key=lambda d: (d[1], d[0])), f"session {i}"
    assert health["sessions_active"] == 0 and health["sessions_total"] == n
    assert wall < 60 * SLACK


def test_health_endpoint_and_structured_json_logs(capsys):
    eng = _fresh(log_json=True)

    async def go():
        async with _Server(eng) as srv:
            h0 = await srv.health()
            async with websockets.connect(srv.url) as ws:
                await ws.recv()
                await ws.send((_speech(0.5) * 32767).astype("<i2").tobytes())
                await asyncio.sleep(0.3)
                h1 = await srv.health()
                await ws.send("junk")
                await ws.send(json.dumps({"type": "end"}))
                await _recv_all(ws)
            with pytest.raises(Exception):  # a non-health HTTP GET is not a WebSocket handshake: refused, no crash
                await asyncio.to_thread(srv._get, "/other")
            return h0, h1, await srv.health()

    h0, h1, h2 = asyncio.run(go())
    for h in (h0, h1, h2):
        assert h["ok"] and set(h) >= {"uptime_s", "sessions_active", "sessions_total", "backlog_ms_total",
                                      "rss_mb", "counters", "limits", "final_asr", "silero", "model"}
    assert h1["sessions_active"] == 1 and h2["sessions_active"] == 0 and h2["counters"]["error_bad_json"] == 1
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    events = [line["event"] for line in logs]
    assert "error" in events and "end" in events and all("ts" in line and "peer" in line for line in logs)
    err = next(line for line in logs if line["event"] == "error")
    assert err["code"] == "bad_json" and err["fatal"] is False


def test_server_shutdown_mid_session_tells_the_client():
    eng = _fresh()

    async def go():
        async with _Server(eng) as srv:
            ws = await websockets.connect(srv.url)
            await ws.recv()
            await ws.send((_speech(1.0) * 32767).astype("<i2").tobytes())
            await asyncio.sleep(0.3)
            srv.stop.set_result(None)
            got = await _recv_all(ws, 5)
            return got, ws.close_code

    got, code = asyncio.run(go())
    assert got and got[-1] == error_msg("server_shutdown", "server stopping", True) and code == 1001


def test_idle_timeout_and_session_limit_close_with_errors():
    eng = _fresh(idle_timeout_s=0.4, max_session_s=1.5)

    async def go():
        async with _Server(eng) as srv:
            t0 = time.perf_counter()
            async with websockets.connect(srv.url) as ws:  # connects and never sends anything
                await ws.recv()
                idle = await _recv_all(ws, 5)
                idle_code = ws.close_code
            t_idle = time.perf_counter() - t0
            long = await _stream(srv.url, _speech(4.0, seed=21), end=False)
            return idle, idle_code, t_idle, long, await srv.health()

    idle, idle_code, t_idle, long, health = asyncio.run(go())
    assert idle == [error_msg("idle_timeout", idle[0]["detail"], True)] and idle_code == 1008 and t_idle < 3 * SLACK
    lim = _errors(long, "session_limit")
    assert len(lim) == 1 and lim[0]["fatal"] and long[-1]["type"] == "stats" and long[-2]["type"] == "final"
    assert 1.5 <= sum(m["type"] == "frame" for m in long) * FRAME_MS / 1000 <= 2.1  # flushed at the limit
    assert health["counters"]["idle_timeout"] == 1 and health["counters"]["session_limit"] == 1


def test_processing_failure_is_reported_closes_the_session_and_spares_the_server():
    eng = _energy_engine(FailingDiarizer)
    x = _speech(2.0, seed=22)

    async def go():
        async with _Server(eng) as srv:
            async with websockets.connect(srv.url) as ws:
                await ws.recv()
                await ws.send((x * 32767).astype("<i2").tobytes())
                got = await _recv_all(ws, 10)
                code = ws.close_code
            eng.diar_cls = H.EnergyDiarizer
            ok = await _stream(srv.url, x[:SR])
            return got, code, ok, await srv.health()

    got, code, ok, health = asyncio.run(go())
    err = _errors(got)
    assert len(err) == 1 and err[0]["code"] == "processing_failed" and err[0]["fatal"] and "exploded" in err[0]["detail"]
    assert code == 1011 and ok[-1]["type"] == "stats" and health["counters"]["processing_failed"] == 1
    assert not eng.conns


def test_ready_handshake_is_immediate_and_a_silent_connect_is_clean():
    eng = _fresh()

    async def go():
        async with _Server(eng) as srv:
            t0 = time.perf_counter()
            async with websockets.connect(srv.url) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), 2))
                dt = time.perf_counter() - t0
            async with websockets.connect(srv.url):
                pass  # open and close without a single message
            await asyncio.sleep(0.1)
            return ready, dt, await srv.health()

    ready, dt, health = asyncio.run(go())
    validate(ready)
    assert dt < 1.0 * SLACK and health["sessions_active"] == 0 and health["sessions_total"] == 2


# =========================================================================== 3. model / state robustness
class _FakeEmbedder:
    dim = 4

    def frames(self, audio, groups):
        return np.stack([np.array([1.0, 0.0, 0.0, 0.0]) for _ in groups])


def test_enrollment_with_no_speech_and_on_overlap():
    vb = VoiceBinder(_FakeEmbedder(), 4)
    vb.arm(0)
    for v in range(200):  # minutes of silence: never enrolls, never binds, bounded histories
        assert vb.update(v, [0.02] * 4, np.zeros(1280, np.float32)) is None
    assert not vb.enrolled and len(vb.audio_hist) <= vb.win
    vb = VoiceBinder(_FakeEmbedder(), 4)
    vb.arm(0)
    col = None
    for v in range(60):  # two columns start together: a deterministic candidate, an enrollment, no crash
        col = vb.update(v, [0.9, 0.9, 0.02, 0.02], np.ones(1280, np.float32) * 0.1)
    assert vb.enrolled and col in (0, 1)
    ab = ArmBinder(4)
    ab.arm(0)
    assert all(ab.update(v, [0.02] * 4) is None for v in range(100)) and not ab.enrolled
    for v in range(100, 105):
        c = ab.update(v, [0.9, 0.9, 0.02, 0.02])
    assert ab.enrolled and c == 0


def test_hybrid_dyn_with_nan_head_posterior_still_ends_turns():
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(float("nan"))
    eng = _energy_engine(model=m)
    eng.silero_model = SH.FakeSilero()
    x = _pattern(((1.0, 0.2), (7.5, 0.0)), seed=23)
    s, msgs = _run(eng, x, SessionConfig(turn_policy="hybrid_dyn"))
    assert _finite(msgs)
    err = _errors(msgs, "nan_state_reset")
    assert len(err) == 1 and not err[0]["fatal"] and s.asr.n_resets >= 1
    frames = [m for m in msgs if m["type"] == "frame"]
    assert all(f["eot"] == 0.0 for f in frames)  # the NaN posterior reads as 0 (never fires the head path)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(te) == 1 and te[0]["policy"] == "hybrid_dyn" and te[0]["p"] == 0.0  # the Silero path at p = 0
    assert 6.4 <= te[0]["silence_ms"] / 1000 <= 6.6 and msgs[-1]["degraded"]["nan_state_reset"] > 0


def test_silero_missing_at_startup_is_a_clear_error_and_sessions_fall_back():
    with pytest.raises(FileNotFoundError, match="--silero"):
        H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, silero="/nonexistent.onnx",
                       preload_silero=True)
    eng = H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, silero="/nonexistent.onnx")
    x = _pattern(((1.0, 0.2), (1.5, 0.0)), seed=24)
    s, msgs = _run(eng, x, SessionConfig(turn_policy="hybrid_dyn", timeout_ms=480))
    assert msgs[0]["code"] == "silero_unavailable" and s.cfg.turn_policy == "hybrid"
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert te and te[0]["policy"] == "hybrid" and s.cfg.theta == S.POLICY_THETA["hybrid_dyn"]


class _FakeFinalWorker:
    """--final-asr stand-in: dead (submit raises), failing (every job fails), hanging (no job ever answers) until
    ``restart`` is called, then answers with the audio length."""
    source, rss_mb = "tdt_v3", 100.0

    def __init__(self, mode="dead"):
        self.mode, self.alive, self.restarts, self.jobs = mode, mode != "dead", 0, 0

    def submit(self, audio):
        self.jobs += 1
        f = Future()
        if self.mode == "dead":
            raise BrokenPipeError("worker pipe closed")
        if self.mode == "failing":
            f.set_exception(RuntimeError("worker crashed mid-job"))
        elif self.mode == "hanging":
            pass  # never resolved
        else:
            class R:
                text, compute_ms, rss_mb = f"offline {len(audio)}", 1.0, 100.0
            f.set_result(R())
        return f

    def restart(self):
        self.restarts += 1
        self.mode, self.alive = "ok", True


@pytest.mark.parametrize("mode", ["dead", "failing", "hanging"])
def test_final_asr_worker_crash_falls_back_to_the_streaming_final_and_restarts(mode, monkeypatch):
    """dead: restarted at the first submit; failing while alive: restarted after FINAL_ASR_FAILS_BEFORE_RESTART
    consecutive failures; hanging: the delivery times out (FINAL_ASR_TIMEOUT_S), restarted at once. In every case
    the client gets every offline final (with the streaming text when the pass failed) and the stats."""
    worker = _FakeFinalWorker(mode)
    eng = _energy_engine(final_asr=worker)
    monkeypatch.setattr(S, "FINAL_ASR_RESTART_S", 0.0)
    monkeypatch.setattr(S, "FINAL_ASR_TIMEOUT_S", 0.3)
    x = _pattern(((1.0, 0.2), (1.2, 0.0), (1.0, 0.2), (1.2, 0.0), (1.0, 0.2), (1.2, 0.0)), seed=25)

    async def go():
        async with _Server(eng) as srv:
            t0 = time.perf_counter()
            # hanging: paced, so the first turn's timeout and restart happen before the next turn is submitted
            msgs = await _stream(srv.url, x, {"timeout_ms": 480}, chunk=320, pace=0.02 if mode == "hanging" else 0)
            return msgs, await srv.health(), time.perf_counter() - t0

    msgs, health, wall = asyncio.run(go())
    stream = [m for m in msgs if m["type"] == "final" and m["source"] == "stream"]
    off = [m for m in msgs if m["type"] == "final" and m["source"] == "tdt_v3"]
    assert len(stream) == 4 and len(off) == 4 and msgs[-1]["type"] == "stats"
    fails = _errors(msgs, "final_asr_failed")
    assert len(fails) == 1 and not fails[0]["fatal"]
    # the first turn's offline final fell back to the streaming text; after the restart the worker answers
    by_t = {m["t"]: m for m in stream}
    assert off[0]["text"] == by_t[off[0]["t"]]["text"] and off[0]["text"]
    assert worker.restarts >= 1 and off[-1]["text"].startswith("offline"), [m["text"] for m in off]
    assert health["counters"].get("final_asr_failed", 0) >= 1 and health["final_asr"]["alive"]
    if mode == "hanging":
        assert health["counters"]["final_asr_timeout"] >= 1 and wall < 15 * SLACK
    if mode == "failing":
        assert worker.restarts == 1 and health["counters"]["final_asr_failed"] == 2
    for m in msgs:
        validate(m)


def test_lookahead_pass_dropped_under_load_or_failure_falls_back_to_single_pass(monkeypatch):
    monkeypatch.setattr(S, "SHED_RTF", 0.0)  # the tiny model is fast: let the backlog alone drive the shedding
    eng = H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, asr_lookahead=3)
    x = _pattern(((1.0, 0.2), (1.2, 0.0), (1.0, 0.2), (1.2, 0.0)), seed=26)
    s = Session(eng, SessionConfig(timeout_ms=480))
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320], backlog_ms=SHED_PARTIAL_MS if 1.0 <= i / SR < 1.5 else 0.0)
    msgs += s.finish()
    for m in msgs:
        validate(m)
    drop = _errors(msgs, "lookahead_dropped")
    assert len(drop) == 1 and s.la_dropped
    la = [m for m in msgs if m["type"] == "final" and m["source"] == "lookahead"]
    st = [m for m in msgs if m["type"] == "final" and m["source"] == "stream"]
    assert len(la) == len(st) == 3 and all(a["text"] == b["text"] and a["latency_ms"] == 0 for a, b in zip(la, st))
    assert msgs[-1]["degraded"]["lookahead_dropped"] == 1
    # a lookahead pass that raises is dropped the same way
    eng2 = H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, asr_lookahead=3)
    s2 = Session(eng2, SessionConfig(timeout_ms=480))
    s2.la.feed = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lookahead broke"))
    out = []
    for i in range(0, len(x), 320):
        out += s2.process(x[i:i + 320])
    out += s2.finish()
    assert _errors(out, "lookahead_dropped") and "lookahead broke" in _errors(out, "lookahead_dropped")[0]["detail"]
    assert len([m for m in out if m["type"] == "final" and m["source"] == "lookahead"]) == 3


def test_device_fallback_and_startup_checks(monkeypatch, tmp_path, capsys):
    import audioforge.serve as SS
    monkeypatch.setattr("audioforge.train.load_model", lambda p, d: H._asr_model())
    monkeypatch.setattr("audioforge.nemo_import.load_any", lambda p, d: H._diar_model())
    eng = Engine.load("a.afm", "d.afm", device="tpu", threads=1)
    assert eng.counters == {"device_fallback": 1} and "falling back to cpu" in capsys.readouterr().out
    # a GPU device this process cannot see (mps off a Mac, cuda without a GPU) degrades the same way
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    for dev in ("mps", "cuda", "cuda:1"):
        assert Engine.load("a.afm", "d.afm", device=dev, threads=1).counters == {"device_fallback": 1}
    capsys.readouterr()
    with pytest.raises(SystemExit) as ei:
        SS.main(["--asr", str(tmp_path / "missing.afm"), "--diar", str(tmp_path / "also_missing.afm")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--asr file not found" in err and "--diar file not found" in err
    (tmp_path / "a.afm").write_bytes(b"")
    (tmp_path / "d.afm").write_bytes(b"")
    with pytest.raises(SystemExit) as ei:  # TitaNet needed by --enroll after_agent, absent: at startup
        SS.main(["--asr", str(tmp_path / "a.afm"), "--diar", str(tmp_path / "d.afm"), "--enroll", "after_agent",
                 "--titanet", str(tmp_path / "no_titanet.nemo")])
    assert ei.value.code == 2 and "--titanet" in capsys.readouterr().err


def test_diarizer_nan_rows_are_zeroed_and_reported():
    class NanDiarizer(H.EnergyDiarizer):
        def feed(self, samples, final=False):
            out = super().feed(samples, final)
            if len(out) and 10 <= self.done <= 14:
                out[-1] = float("nan")
            return out

    eng = _energy_engine(NanDiarizer)
    s, msgs = _run(eng, _pattern(((1.5, 0.2), (1.0, 0.0)), seed=27), SessionConfig(timeout_ms=480))
    assert _finite(msgs) and _errors(msgs, "nan_state_reset") and msgs[-1]["degraded"]["nan_state_reset"] >= 1
    assert all(0 <= p <= 1 for m in msgs if m["type"] == "frame" for p in m["speakers"])


# =========================================================================== 4. watchdog and limits
def test_shedding_levels_skip_diarizer_then_partials_and_report_once(monkeypatch):
    monkeypatch.setattr(S, "SHED_RTF", 0.0)  # the tiny model is fast: let the backlog alone drive the shedding
    eng = _fresh(H._talky_asr_model(), debug=True)  # a token per frame: the partial changes every block
    x = _pattern(((1.0, 0.2), (1.0, 0.0), (1.0, 0.2)), seed=28)
    s = Session(eng, SessionConfig(turn_policy="both", timeout_ms=480))
    msgs, lvls = [], []
    for i in range(0, len(x), 1600):
        t = i / SR
        backlog = 0.0 if t < 1.0 else SHED_DIAR_MS if t < 2.0 else SHED_PARTIAL_MS
        out = s.process(x[i:i + 1600], backlog_ms=backlog)
        lvls.append(s.shed)
        msgs += out
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=True)
    assert lvls[:10] == [0] * 10 and lvls[10:20] == [1] * 10 and lvls[20:] == [2] * 10
    over = _errors(msgs, "overloaded")
    assert len(over) == 1 and "level 1" in over[0]["detail"]  # one notice per SHED_NOTICE_S
    assert s.diar.skipped > 0 and s.degraded["shed_diar_frames"] == s.diar.skipped
    frames = [m for m in msgs if m["type"] == "frame"]
    batched = [m for m in msgs if m["type"] == "frames"]
    assert batched and sum(len(b["items"]) for b in batched) + len(frames) == H._n_frames(eng.asr, x)
    assert s.degraded["frames_batched"] == sum(len(b["items"]) for b in batched) and s.degraded["shed_partials"] >= 1
    # under level >= 1 the served VAD stands in for the diarizer: column 0 tracks the VAD, the others are 0
    shed_rows = [np.asarray(s.rows[v]) for v in range(len(s.rows)) if 13 <= v < 24]
    assert all(r[1:].max() == 0.0 for r in shed_rows) and any(r[0] > 0 for r in shed_rows)
    st = msgs[-1]
    assert st["degraded"]["shed_diar_frames"] > 0 and st["backlog_ms_max"] == SHED_PARTIAL_MS
    assert eng.counters["overloaded"] == 1 and eng.counters["shed_diar_frames"] == s.diar.skipped
    # decisions are still produced (the timeout ran on the VAD column) and the transcript is intact
    assert any(m["type"] == "turn_end" for m in msgs) and msgs[-2]["type"] == "final"


def test_burst_backlog_on_a_fast_server_is_not_shed():
    """The behaviour-preserving rule: a client that dumps audio faster than real time at a server that is faster
    than real time (recent_rtf < SHED_RTF) builds a backlog that drains by itself, so nothing is shed and the
    decisions equal the paced run's (unpaced eval drivers and paced clients measure the same thing)."""
    eng = _energy_engine(model=H._asr_model())
    x = _pattern(((1.0, 0.2), (1.0, 0.0), (1.0, 0.2), (1.0, 0.0)), seed=30)
    cfg = SessionConfig(turn_policy="both", timeout_ms=480)
    paced = _run(eng, x, cfg, block=320, backlog_ms=0.0)
    burst = _run(eng, x, cfg, block=320, backlog_ms=10 * SHED_PARTIAL_MS)  # "40 s queued behind this block"
    assert _decisions(burst[1]) == _decisions(paced[1]) and any(d[0] == "turn_end" for d in _decisions(paced[1]))
    assert burst[0].shed_max == 0 and not _errors(burst[1]) and burst[0].recent_rtf() < S.SHED_RTF
    assert "shed_diar_frames" not in burst[0].degraded and burst[0].backlog_ms_max == 10 * SHED_PARTIAL_MS
    # the global watchdog follows the same rule: a queued backlog without load is no reason to shed
    eng.conns.clear()
    conn = S._Conn()
    conn.session, conn.n_in = burst[0], int(100 * SR)
    eng.conns.add(conn)
    assert eng.global_shed() == 0 and eng.global_load() < S.SHED_RTF
    burst[0]._rate.clear()
    burst[0]._rate.append((SR, 2.0))  # 2 s of work per second of audio: now it is overload
    assert eng.global_shed() == 2 and eng.global_load() == 2.0
    eng.conns.clear()


def test_shedding_keeps_decisions_identical_when_diarizer_is_a_noop(monkeypatch):
    """Level 2 only drops messages, never audio: the ASR frames and finals of a shed session equal the plain ones."""
    monkeypatch.setattr(S, "SHED_RTF", 0.0)
    eng = _energy_engine(model=H._talky_asr_model())
    x = _pattern(((1.0, 0.2), (1.0, 0.0), (1.0, 0.2), (1.0, 0.0)), seed=29)
    plain = _run(eng, x, SessionConfig(timeout_ms=480), block=1600)[1]
    shed = _run(eng, x, SessionConfig(timeout_ms=480), block=1600, backlog_ms=SHED_PARTIAL_MS)[1]
    fp = [(f["t"], f["vad"]) for f in plain if f["type"] == "frame"]
    fs = [(it["t"], it["vad"]) for m in shed if m["type"] == "frames" for it in m["items"]]
    fs += [(f["t"], f["vad"]) for f in shed if f["type"] == "frame"]
    assert sorted(fp) == sorted(fs)
    assert [m["text"] for m in plain if m["type"] == "final"] == [m["text"] for m in shed if m["type"] == "final"]
    # no partial while shed (the flush at the end runs with no backlog, so it may emit the last one)
    assert sum(m["type"] == "partial" for m in shed) <= 1 < sum(m["type"] == "partial" for m in plain)


def test_stats_and_health_counters_cover_every_degradation():
    """Every degradation code the server can raise appears in ERROR_CODES (so validate accepts it) and is counted."""
    used = set()
    src = (ROOT / "audioforge" / "serve.py").read_text()
    import re
    for m in re.finditer(r'_notice\("([a-z_]+)"', src):
        used.add(m.group(1))
    for m in re.finditer(r'error_msg\("([a-z_]+)"', src):
        used.add(m.group(1))
    for m in re.finditer(r'_SessionFailed\("([a-z_]+)"', src):
        used.add(m.group(1))
    for m in re.finditer(r'send_error\("([a-z_]+)"', src):
        used.add(m.group(1))
    assert used <= set(ERROR_CODES), used - set(ERROR_CODES)
    assert {"nan_input", "overloaded", "final_asr_failed", "lookahead_dropped", "silero_unavailable",
            "processing_failed", "idle_timeout", "session_limit", "server_shutdown"} <= used


# =========================================================================== 5. adapters
def test_pipecat_adapter_accepts_error_messages():
    pytest.importorskip("pipecat")
    from pipecat.processors.frame_processor import FrameDirection  # noqa: F401
    from integrations.pipecat_audioforge import AudioforgeSTTService
    stt = AudioforgeSTTService(url="ws://unused")
    pushed = []

    async def fake_push_error(error_msg, exception=None):
        pushed.append(error_msg)

    stt.push_error = fake_push_error

    async def go():
        await stt.handle_server_message(error_msg("nan_input", "3 samples", False))
        await stt.handle_server_message(error_msg("processing_failed", "boom", True))

    asyncio.run(go())
    assert [e["code"] for e in stt.hub.errors] == ["nan_input", "processing_failed"] and len(pushed) == 1
    assert "processing_failed" in pushed[0]


def test_livekit_link_records_errors_and_fatal_ends_the_session():
    pytest.importorskip("livekit.agents")
    from integrations.livekit_audioforge import AudioforgeOptions, _Link
    eng = H._engine()

    async def go():
        async with _Server(eng) as srv:
            lk = _Link(AudioforgeOptions(url=srv.url))
            await lk.open()
            q = lk.subscribe()
            await lk.ws.send("junk")
            await lk.ws.send(json.dumps({"type": "end"}))
            got = []
            while True:
                item = await asyncio.wait_for(q.get(), 10)
                if item is None:
                    break
                got.append(item[0])
            await lk.aclose()
            return got, list(lk.errors), lk.error

    got, errors, fatal = asyncio.run(go())
    assert [e["code"] for e in errors] == ["bad_json"] and fatal is None and got[-1]["type"] == "stats"


def test_stream_client_survives_a_fatal_server_error_and_reports_it(tmp_path):
    """scripts/stream_client.py against a server whose session dies mid-stream (processing_failed, close 1011):
    the client stops sending, does not raise, and its summary carries the error, complete=False and what arrived;
    against a healthy server the summary has no errors and complete=True."""
    import argparse
    import soundfile as sf
    client = H._load_client()
    x = _speech(2.0, seed=31)
    wav = tmp_path / "c.wav"
    sf.write(wav, x, SR)

    def args(url):
        return argparse.Namespace(audio=str(wav), url=url, speed=0.0, policy="both", timeout_ms=400,
                                  eot_threshold=0.5, log=None, summary=None, ref=None, ref_text=None, verbose=False)

    async def go():
        async with _Server(_energy_engine(FailingDiarizer)) as srv:
            bad = await client.run(args(srv.url))
        async with _Server(_energy_engine()) as srv:
            ok = await client.run(args(srv.url))
        return bad, ok

    bad, ok = asyncio.run(go())
    assert bad["fatal_error"] == "processing_failed" and not bad["complete"] and bad["server_stats"] is None
    assert [e["code"] for e in bad["errors"]] == ["processing_failed"] and bad["errors"][0]["fatal"]
    assert ok["complete"] and ok["errors"] == [] and ok["fatal_error"] is None and ok["finals"]


def test_server_diarizer_keeps_no_per_frame_probabilities():
    """The real streaming diarizer accumulates every emitted column (`probs`) for offline use; the server's
    sessions run it with keep_probs=False so a long session does not grow with it (the soak uses the scripted
    diarizer for speed; this pins the real one)."""
    eng = _fresh()
    s, msgs = _run(eng, _speech(10.0, seed=32), block=SR // 2)
    assert len(s.diar.probs) == 0 and s.diar.keep_probs is False and len(s.rows) >= 120 and msgs[-1]["type"] == "stats"


def test_bounded_energy_diarizer_equals_the_shared_double():
    x = _pattern(((1.0, 0.2), (0.7, 0.0), (0.5, 0.2), (0.3, 0.0)), seed=33)
    a = _run(_energy_engine(), x, SessionConfig(timeout_ms=480), block=1000)[1]
    b = _run(_energy_engine(BoundedEnergyDiarizer), x, SessionConfig(timeout_ms=480), block=1000)[1]
    assert _decisions(a) == _decisions(b) and any(m["type"] == "turn_end" for m in a)
