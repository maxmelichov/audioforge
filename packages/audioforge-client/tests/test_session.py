import asyncio

import pytest

from audioforge_client import (
    AudioforgeConnectionError,
    AudioforgeSession,
    Final,
    FrameEvent,
    Partial,
    ServerError,
    Stats,
    TurnEnd,
)
from audioforge_client.testing import FakeAudioforgeServer, speech_pcm


async def _drain(s):
    return [ev async for ev in s]


@pytest.mark.parametrize("batch", [False, True])
async def test_session_round_trip(batch):
    async with FakeAudioforgeServer(timeout_ms=480, batch_frames=batch) as srv:
        async with AudioforgeSession(srv.url, timeout_ms=480) as s:
            assert s.ready is not None and s.ready.model == "fake-audioforge"
            pcm = speech_pcm(1.0, 1.0)
            for i in range(0, len(pcm), 640):
                await s.send_audio(pcm[i : i + 640])
            assert abs(s.audio_seconds_sent - 2.0) < 1e-6
            stats = await s.end()
            assert isinstance(stats, Stats)
            evs = await _drain(s)
    kinds = [type(e) for e in evs]
    assert kinds.count(FrameEvent) == 25
    assert Partial in kinds
    te = [e for e in evs if isinstance(e, TurnEnd)]
    assert len(te) == 1 and te[0].policy == "timeout" and te[0].silence_ms == 480
    fins = [e for e in evs if isinstance(e, Final)]
    assert fins[0].text.startswith("w0") and fins[0].t == te[0].t and fins[-1].text == ""
    assert isinstance(evs[-1], Stats) and s.error is None
    assert srv.configs[0]["turn_policy"] == "timeout" and srv.configs[0]["timeout_ms"] == 480


@pytest.mark.parametrize(
    "policy,tags",
    [("head", {"head"}), ("both", {"head", "timeout"}), ("hybrid_dyn", {"hybrid_dyn"})],
)
async def test_policies_tag_and_cut(policy, tags):
    async with FakeAudioforgeServer(timeout_ms=480) as srv:
        async with AudioforgeSession(srv.url, turn_policy=policy, timeout_ms=480) as s:
            await s.send_audio(speech_pcm(1.0, 1.0))
            await s.end()
            evs = await _drain(s)
    te = {e.policy for e in evs if isinstance(e, TurnEnd)}
    assert te == tags
    fins = [e for e in evs if isinstance(e, Final) and e.text]
    assert len(fins) == 1


async def test_controls_are_logged_at_audio_position_and_final_asr_twins():
    async with FakeAudioforgeServer(
        timeout_ms=480, enroll="after_agent_arm", final_asr=True
    ) as srv:
        async with AudioforgeSession(srv.url, timeout_ms=480) as s:
            assert s.ready.enroll == "after_agent_arm" and s.ready.final_asr == "tdt_v3"
            await s.send_audio(speech_pcm(0.5))
            assert await s.agent_end()
            await s.send_audio(speech_pcm(0.5, 1.0))
            await s.end()
            evs = await _drain(s)
    assert srv.controls == [(0, "agent_end", 8000)]
    assert s.controls_sent == [("agent_end", 0.5)]
    assert any(e.type == "enrolled" for e in evs)
    fins = [e for e in evs if isinstance(e, Final)]
    assert {f.source for f in fins} == {"stream", "tdt_v3"}
    off = [f for f in fins if f.is_offline and f.text]
    assert off and off[0].text == off[0].text.upper() and off[0].latency_ms == 1.0
    st = evs[-1]
    assert isinstance(st, Stats) and st.raw["enrolled"] is True and st.raw["final_asr"] == "tdt_v3"


async def test_connection_refused_and_server_drop():
    with pytest.raises(AudioforgeConnectionError):
        async with AudioforgeSession("ws://127.0.0.1:1", connect_timeout=1.0):
            pass
    async with FakeAudioforgeServer(drop_after_s=0.5) as srv:
        async with AudioforgeSession(srv.url) as s:
            await s.send_audio(speech_pcm(1.0))
            evs = await asyncio.wait_for(_drain(s), 5.0)
    assert s.error is not None and isinstance(evs[-1], ServerError)
    assert not s.is_open and not await s.send_audio(b"\0\0")
