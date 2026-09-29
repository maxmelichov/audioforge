"""serve.py's shipped rules (research/INTEGRATION.md section 8): turn policies hybrid_silero (head OR any-speaker Silero
silence >= 2.64 s) and hybrid_dyn (head OR Silero silence >= clamp(80 - 55 p, 7, 80) frames), and --enroll
after_agent_arm (the agent-end choice, then causal_dominant). Tiny models + a scripted diarizer and an energy "Silero"
stand in for the checkpoints; the real Silero ONNX test runs when data/silero/silero_vad_v5.onnx exists."""
import asyncio
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from audioforge.baselines import turn as B
from audioforge.serve import (DYN_A, DYN_OFFSET, DYN_T0, DYN_TMIN, POLICIES, POLICY_THETA, SILERO_TIMEOUT_MS,
                              ArmBinder, CausalDominant, Session, SessionConfig, SileroSilence, serve, validate)

ROOT = Path(__file__).resolve().parents[1]
SILERO = ROOT / "data" / "silero" / "silero_vad_v5.onnx"


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _h()


class FakeSilero:
    """SileroStream stand-in: p = 0.95 on a 512-sample chunk with RMS > 0.01, else 0.02 (stateless)."""

    def new_state(self):
        return [0]

    def step(self, state, chunk):
        state[0] += 1
        return 0.95 if float(np.sqrt(np.mean(np.asarray(chunk, np.float64) ** 2))) > 0.01 else 0.02

    def probs(self, audio):
        n = len(audio) // B.CHUNK
        return np.array([self.step([0], audio[j * B.CHUNK:(j + 1) * B.CHUNK]) for j in range(n)], np.float32)


def _ref_silence(x, vad):
    """The benchmark's offline track: silence_frames(speech_chunks_pipecat(pipecat_vad(probs)), T)."""
    T = len(x) // 1280
    return B.silence_frames(B.speech_chunks_pipecat(B.pipecat_vad(vad.probs(x))), T)


def _engine(turn_bias=8.0, **kw):
    m = H._talky_asr_model()
    with torch.no_grad():
        m.heads["turn"].out.bias.fill_(turn_bias)
    eng = H.EnergyEngine(m, H._diar_model(), name="tiny", threads=1, **kw)
    eng.silero_model = FakeSilero()
    return eng


def _run(eng, cfg, x, fs=320):
    s = Session(eng, cfg)
    msgs = []
    for i in range(0, len(x), fs):
        msgs += s.process(x[i:i + fs])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    return s, msgs


# --------------------------------------------------------------------------- Silero stream + silence track
@pytest.mark.skipif(not SILERO.exists(), reason="data/silero/silero_vad_v5.onnx not present")
def test_silero_stream_equals_offline_wrapper():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(8000) * 0.3, np.zeros(6000), rng.standard_normal(9000) * 0.2]).astype(
        np.float32)
    ref = B.SileroVAD(SILERO).probs(x)
    st = B.SileroStream(SILERO)
    s = st.new_state()
    live = np.array([st.step(s, x[j * 512:(j + 1) * 512]) for j in range(len(x) // 512)], np.float32)
    np.testing.assert_allclose(live, ref, atol=1e-6)


@pytest.mark.parametrize("block", [1, 320, 777, 8000])
def test_silero_silence_equals_benchmark_track(block):
    x = H._speech_silence(((0.5, 0.0), (1.0, 0.2), (3.0, 0.0), (0.1, 0.2), (0.4, 0.0), (1.2, 0.2), (2.0, 0.0)))
    ref = _ref_silence(x, FakeSilero())
    ss = SileroSilence(FakeSilero())
    got = []
    for i in range(0, len(x), block):
        got += ss.feed(x[i:i + block])
    assert [g[0] for g in got] == list(range(len(got))) and len(got) >= len(ref) - 1
    np.testing.assert_allclose([g[1] for g in got[:len(ref)]], ref[:len(got)], atol=1e-4)
    for v, _, t_rdy, _ in got:  # ready when the chunk holding the frame end is complete: <= 16 ms after the end
        assert -1e-9 <= t_rdy - (v + 1) * 0.08 < 0.032 + 1e-9
    assert len(ss.ms) == len(x) // 512


# --------------------------------------------------------------------------- policies
def test_config_theta_defaults_and_protocol():
    assert {"hybrid_silero", "hybrid_dyn"} <= set(POLICIES)
    assert SessionConfig().theta == 0.98 and SessionConfig(turn_policy="hybrid").theta == 0.98
    assert SessionConfig(turn_policy="hybrid_silero").theta == POLICY_THETA["hybrid_silero"] == 0.99828
    assert SessionConfig(turn_policy="hybrid_dyn").theta == POLICY_THETA["hybrid_dyn"] == 0.998283
    c = SessionConfig()
    assert not c.update({"turn_policy": "hybrid_dyn", "eot_threshold": 0.9}) and c.theta == 0.9
    validate({"type": "turn_end", "t": 2.0, "policy": "hybrid_dyn", "p": 0.4, "silence_ms": 4000})
    validate({"type": "turn_end", "t": 2.0, "policy": "hybrid_silero", "p": None, "silence_ms": 2640})
    # wait = clamp(80 - 55 p, 7, 80) frames, up to the 3 ms offset rounding of the frozen (75, 55, 2, +4.957) form
    for p in (0.0, 0.5, 0.9, 1.0):
        req = min(max(DYN_T0 - DYN_A * p, DYN_TMIN), DYN_T0) + DYN_OFFSET
        assert abs(req - min(max(80 - 55 * p, 7), 80)) < 0.05


def test_silero_not_loaded_unless_a_silero_policy_is_used():
    eng = H.EnergyEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, silero="/nonexistent.onnx")
    s = Session(eng, SessionConfig(turn_policy="hybrid"))  # no Silero needed: nothing loaded
    assert s.sil is None and eng.silero_model is None
    # a Silero policy without the file degrades to hybrid (head at the Silero policy's threshold OR the diarizer
    # timeout) and reports it once as a non-fatal error (research/BULLETPROOF.md) instead of killing the session
    s2 = Session(eng, SessionConfig(turn_policy="hybrid_silero"))
    assert s2.sil is None and s2.cfg.turn_policy == "hybrid" and s2.cfg.theta == POLICY_THETA["hybrid_silero"]
    msgs = s2.process(np.zeros(3200, np.float32))
    assert msgs[0]["type"] == "error" and msgs[0]["code"] == "silero_unavailable" and not msgs[0]["fatal"]
    assert eng.silero_error and eng.counters["silero_unavailable"] == 1


def test_hybrid_silero_fires_at_264s_of_any_speaker_silence_once_per_run():
    eng = _engine()
    x = H._speech_silence(((1.0, 0.2), (3.2, 0.0), (1.0, 0.2), (3.2, 0.0)))
    s, msgs = _run(eng, SessionConfig(turn_policy="hybrid_silero", eot_threshold=1.0), x)  # head path off
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert [m["policy"] for m in te] == ["hybrid_silero"] * 2
    # speech chunk 31 ends 1.024 s: frame 45 (end 3.68 s) is the first with >= 33 frames; chunk 162 ends 5.216 s ->
    # frame 98 (7.92 s), its chunk complete at 7.936 s
    assert [m["t"] for m in te] == [3.68, 7.936]
    assert [m["silence_ms"] for m in te] == [2656, 2704] and all(isinstance(m["p"], float) for m in te)
    ref = _ref_silence(x, FakeSilero())
    fire = [v for v in range(1, len(ref)) if ref[v] >= SILERO_TIMEOUT_MS / 80 > ref[v - 1]]
    assert fire == [45, 98]
    finals = [m for m in msgs if m["type"] == "final"]
    assert len(finals) == 3  # a final cut at each turn_end + the end final
    for fs in (333, 5000):
        _, m2 = _run(eng, SessionConfig(turn_policy="hybrid_silero", eot_threshold=1.0), x, fs)
        assert [(m["type"], m["t"], m.get("text")) for m in m2 if m["type"] in ("turn_end", "final")] == \
            [(m["type"], m["t"], m.get("text")) for m in msgs if m["type"] in ("turn_end", "final")]


@pytest.mark.parametrize("bias,expect_frame", [(8.0, 37), (-8.0, 92)])
def test_hybrid_dyn_wait_follows_the_head_posterior(bias, expect_frame):
    """p ~ 1 -> wait ~2.0 s (fires at 25.2 frames of silence); p ~ 0 -> 6.4 s (80.2 frames)."""
    eng = _engine(bias, debug=True)
    x = H._speech_silence(((1.0, 0.2), (7.2, 0.0)))
    s, msgs = _run(eng, SessionConfig(turn_policy="hybrid_dyn", eot_threshold=1.0), x)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert len(te) == 1 and te[0]["policy"] == "hybrid_dyn"
    fe = (expect_frame + 1) * 0.08
    assert te[0]["silence_ms"] == round((fe - 1.024) * 1000)
    assert fe <= te[0]["t"] <= fe + 0.2  # the later of the Silero chunk and the head frame
    assert (te[0]["p"] > 0.99) == (bias > 0)
    st = msgs[-1]
    assert st["silero_chunks"] == len(x) // 512 and st["silero_ms_mean"] >= 0


def test_hybrid_dyn_head_path_and_dedup():
    """With the frozen θ the talky head (p 0.9997 from the start) fires once on the first speech; each later silence is
    its own turn through the dyn path; nothing fires twice for one silence."""
    eng = _engine(8.0)
    x = H._speech_silence(((1.0, 0.2), (3.0, 0.0), (1.0, 0.2), (3.0, 0.0)))
    _, msgs = _run(eng, SessionConfig(turn_policy="hybrid_dyn"), x)
    te = [m for m in msgs if m["type"] == "turn_end"]
    assert te[0]["t"] < 0.5 and te[0]["silence_ms"] == 0  # head path (VAD speech, no silence)
    assert len(te) == 3 and all(m["policy"] == "hybrid_dyn" for m in te)
    assert 2.9 < te[1]["t"] < 3.3 and 6.9 < te[2]["t"] < 7.3


# --------------------------------------------------------------------------- --enroll after_agent_arm
def test_causal_dominant_streaming_equals_batch_with_force():
    from audioforge.enrollment import causal_dominant_from
    rng = np.random.default_rng(3)
    for trial in range(30):
        T = 160
        p = np.where(rng.random((T, 4)) < 0.25, rng.uniform(0.5, 1.0, (T, 4)), rng.uniform(0, 0.5, (T, 4)))
        p[rng.integers(0, T, 40)] = 0.01  # silent frames
        force = None if trial % 3 == 0 else (int(rng.integers(0, 4)), int(rng.integers(0, T)))
        ref = causal_dominant_from(p, force=force)
        cd = CausalDominant(4)
        got = [cd.step(p[t], force[0] if force is not None and t == force[1] else None) for t in range(T)]
        assert got == list(ref)


def test_arm_binder_chooses_after_agent_end_then_follows_causal_dominant():
    rows = H._scene_rows()
    b = ArmBinder(4)
    cols = []
    for v in range(120):
        if v == 13:
            b.arm(13)
        cols.append(b.update(v, rows[v]))
    assert cols[:22] == [None] * 22  # before the first choice: the server's default rule
    assert cols[22] == 0 and b.enrolled and b.n_enrolled == 1 and b.n_embed == 0
    # column 0 falls silent at frame 61; 26 silent frames later (frame 86) causal_dominant re-binds to column 2
    assert all(c == 0 for c in cols[22:86]) and all(c == 2 for c in cols[86:111])
    b.arm(115)
    assert not b.enrolled and b.column == 2  # re-armed: the binding holds until the next choice


def test_session_after_agent_arm_messages_no_titanet():
    eng = H.ScriptedEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, enroll="after_agent_arm",
                           debug=True)
    assert eng.embedder is None and eng.enroll_rss_mb == 0.0
    r = eng.ready_msg()
    validate(r, debug=True)
    assert r["enroll"] == "after_agent_arm" and r["enrolled"] is False and r["primary_column"] is None
    s = Session(eng, SessionConfig(timeout_ms=800))
    assert isinstance(s.binder, ArmBinder)
    assert s.arm_enrollment("enroll") is False and s.arm_enrollment("agent_end", 13) is True
    x = H._scene_audio()
    msgs = []
    for i in range(0, len(x), 320):
        msgs += s.process(x[i:i + 320])
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=True)
    enr = [m for m in msgs if m["type"] == "enrolled"]
    assert len(enr) == 1 and enr[0]["column"] == 0  # chosen on frame 22 (finalized ~2.8 s with chunk 6 + 7)
    prim = [(m["t"], m["primary"]) for m in msgs if m["type"] == "frame"]
    assert all(p == 1 for t, p in prim if 1.0 <= t <= 1.9)  # before the choice: the dominant column (the agent)
    assert all(p == 0 for t, p in prim if 3.0 <= t <= 7.7)  # column 86 (re-bind to 2) is final at 7.76 s
    assert all(p == 2 for t, p in prim if t >= 7.8)
    st = msgs[-1]
    assert st["enroll"] == "after_agent_arm" and st["enrolled"] is True and st["primary_column"] == 2
    assert st["enroll_n_embed"] == 0


def test_e2e_socket_hybrid_dyn_with_arm():
    eng = H.ScriptedEngine(H._talky_asr_model(), H._diar_model(), name="tiny", threads=1, enroll="after_agent_arm")
    eng.silero_model = FakeSilero()
    x = H._scene_audio()

    async def run():
        stop = asyncio.get_running_loop().create_future()
        port_f = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(serve(eng, "127.0.0.1", 0, ready_event=port_f.set_result, stop=stop))
        port = await port_f
        import websockets
        got = []
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            got.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({"type": "config", "turn_policy": "hybrid_dyn"}))
            pcm = (x * 32768).astype("<i2").tobytes()
            cut = 13 * 1280 * 2
            await ws.send(pcm[:cut])
            await ws.send(json.dumps({"type": "agent_end"}))
            await ws.send(pcm[cut:])
            await ws.send(json.dumps({"type": "end"}))
            async for m in ws:
                got.append(json.loads(m))
        stop.set_result(None)
        await task
        return got

    got = asyncio.run(run())
    for m in got:
        validate(m)
    assert got[0]["enroll"] == "after_agent_arm"
    assert [m["column"] for m in got if m["type"] == "enrolled"] == [0]
    assert all(m["policy"] == "hybrid_dyn" for m in got if m["type"] == "turn_end")
    assert got[-1]["type"] == "stats" and got[-1]["enroll"] == "after_agent_arm"


def test_cli_flags_parse(monkeypatch, tmp_path):
    import audioforge.serve as S
    seen = {}

    def fake_load(asr, diar, device, **kw):
        seen.update(kw)
        raise SystemExit(0)

    monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
    a, d, x = tmp_path / "a.afm", tmp_path / "d.afm", tmp_path / "x.onnx"
    for f in (a, d, x):  # the startup check wants the files to exist (a missing one exits 2 before loading)
        f.write_bytes(b"")
    with pytest.raises(SystemExit) as ei:
        S.main(["--asr", "a", "--diar", "d", "--enroll", "after_agent_arm"])
    assert ei.value.code == 2 and not seen
    with pytest.raises(SystemExit):
        S.main(["--asr", str(a), "--diar", str(d), "--enroll", "after_agent_arm", "--silero", str(x),
                "--silero-timeout-ms", "2000"])
    seen["silero"] = Path(seen["silero"]).name
    assert seen["enroll"] == "after_agent_arm" and seen["silero"] == "x.onnx" and seen["silero_timeout_ms"] == 2000
    assert seen["preload_silero"] is True
