import json

import pytest

from audioforge_client import (
    Final,
    FrameEvent,
    Ready,
    ServerError,
    Stats,
    TurnEnd,
    TurnHint,
    TurnHintCancel,
    Unknown,
    Voiceprint,
    config_message,
    control_message,
    cut_policy,
    parse_message,
    use_final,
)


def test_parse_each_type_and_batches():
    ready = parse_message(
        json.dumps(
            {
                "type": "ready",
                "model": "m",
                "chunk_ms": 160,
                "frame_ms": 80,
                "diar_config": "low_latency_032",
                "column_lag_ms": 160,
            }
        )
    )[0]
    assert isinstance(ready, Ready) and ready.column_lag_ms == 160 and ready.final_asr is None
    evs = parse_message(
        json.dumps(
            {
                "type": "frames",
                "items": [
                    {"t": 0.08, "vad": 0.9, "eot": None, "speakers": [0.9, 0, 0, 0], "primary": 0},
                    {"t": 0.16, "vad": 0.1, "eot": 0.5, "speakers": [0, 0, 0, 0], "primary": None},
                ],
            }
        )
    )
    assert [type(e) for e in evs] == [FrameEvent, FrameEvent]
    assert evs[0].speakers == (0.9, 0, 0, 0) and evs[1].eot == 0.5 and evs[1].primary is None
    te = parse_message(
        '{"type":"turn_end","t":2.4,"policy":"hybrid_dyn","p":0.999,"silence_ms":2000}'
    )[0]
    assert isinstance(te, TurnEnd) and te.policy == "hybrid_dyn" and te.silence_ms == 2000
    fin = parse_message(
        '{"type":"final","t":2.4,"text":"hi","speaker":1,"source":"tdt_v3","start":0.3,'
        '"end":2.1,"latency_ms":410}'
    )[0]
    assert isinstance(fin, Final) and fin.is_offline and fin.latency_ms == 410 and fin.speaker == 1
    st = parse_message(
        '{"type":"stats","rtf":0.5,"chunk_ms_p50":38,"chunk_ms_p95":200,"first_partial_ms":null,'
        '"peak_rss_mb":3000}'
    )[0]
    assert isinstance(st, Stats) and st.first_partial_ms is None
    assert isinstance(parse_message('{"type":"whatever","x":1}')[0], Unknown)


def test_malformed_messages_become_errors_not_exceptions():
    assert isinstance(parse_message("not json")[0], ServerError)
    assert isinstance(parse_message("[1,2]")[0], ServerError)
    bad = parse_message('{"type":"turn_end","t":"soon"}')[0]
    assert isinstance(bad, ServerError) and "turn_end" in bad.message


def test_cut_policy_and_use_final():
    assert cut_policy("timeout") == cut_policy("timeout_quiet") == cut_policy("both") == "timeout"
    assert cut_policy("head") == "head"
    for p in ("hybrid", "hybrid_silero", "hybrid_dyn", "vad_head"):
        assert cut_policy(p) == p
    assert use_final({"text": "x"}, "stream") and use_final({"text": "x"}, "offline")
    assert use_final({"source": "stream"}, "stream") and not use_final(
        {"source": "stream"}, "offline"
    )
    assert use_final({"source": "tdt_v3"}, "offline") and not use_final(
        {"source": "tdt_v3"}, "stream"
    )
    # --final-chunk-ms: final_fast is the streaming final, source "slow" the slow pass's
    fast, slow = {"type": "final_fast"}, {"type": "final", "source": "slow"}
    assert use_final(fast) and not use_final(slow)
    assert use_final(slow, final_text="slow") and not use_final(fast, final_text="slow")


def test_config_message_validation():
    assert config_message() == {"type": "config", "timeout_ms": 1000, "sample_rate": 16000}
    assert config_message(turn_policy="vad_head", turn_preset="fast", timeout_ms=None) == {
        "type": "config",
        "turn_policy": "vad_head",
        "turn_preset": "fast",
        "sample_rate": 16000,
    }
    assert config_message(eot_threshold=0.99)["eot_threshold"] == 0.99
    with pytest.raises(ValueError):
        config_message(turn_policy="nope")
    with pytest.raises(ValueError):
        config_message(turn_preset="nope")
    assert control_message("enroll", [1, 2]) == {"type": "enroll", "embedding": [1.0, 2.0]}
    with pytest.raises(ValueError):
        control_message("agent_end", [1.0])
    with pytest.raises(ValueError):
        config_message(eot_threshold=1.5)


def test_server_error_message_is_structured():
    ev = parse_message(
        '{"type":"error","code":"bad_config","detail":"timeout_ms clamped","fatal":false}'
    )[0]
    assert isinstance(ev, ServerError)
    assert ev.code == "bad_config" and ev.detail == "timeout_ms clamped" and ev.fatal is False
    assert "bad_config" in ev.message
    fatal = parse_message('{"type":"error","code":"idle_timeout","detail":"","fatal":true}')[0]
    assert (
        isinstance(fatal, ServerError) and fatal.fatal is True and fatal.message == "idle_timeout"
    )
    client_side = parse_message("not json")[0]
    assert isinstance(client_side, ServerError) and client_side.code == "client_parse"


def test_current_protocol_fields():
    te = parse_message(
        '{"type":"turn_end","t":11.776,"policy":"vad_head","p":0.99,"silence_ms":160,'
        '"hinted_at":11.616,"path":"head"}'
    )[0]
    assert isinstance(te, TurnEnd) and te.hinted_at == 11.616 and te.path == "head"
    hint, cancel = (
        parse_message(
            json.dumps(
                {"type": "turn_end_hint", "t": 11.616, "p": 0.878, "kind": "hint", "text": "so um"}
            )
        )[0],
        parse_message('{"type":"turn_end_hint_cancel","t":9.216}')[0],
    )
    assert isinstance(hint, TurnHint) and hint.text == "so um" and hint.p == 0.878
    assert isinstance(cancel, TurnHintCancel) and cancel.t == 9.216
    fast = parse_message(
        '{"type":"final_fast","t":2.4,"text":"hi","speaker":0,'
        '"voice_gender":{"female":0.1,"male":0.9,"speech_ms":640}}'
    )[0]
    assert isinstance(fast, Final) and fast.fast and fast.stream_source == "stream"
    assert not fast.is_second_pass and fast.voice_gender == {
        "female": 0.1,
        "male": 0.9,
        "speech_ms": 640,
    }
    slow = parse_message(
        '{"type":"final","t":2.4,"text":"Hi.","speaker":0,"source":"slow","pass":"slow",'
        '"start":0.1,"end":1.9,"latency_ms":20}'
    )[0]
    assert isinstance(slow, Final) and slow.is_second_pass and slow.raw["pass"] == "slow"
    vp = parse_message(
        '{"type":"voiceprint","t":0.0,"seconds":0.0,"source":"explicit","embedding":[0.5,-0.5]}'
    )[0]
    assert isinstance(vp, Voiceprint) and vp.embedding == (0.5, -0.5) and vp.source == "explicit"
    ready = parse_message(
        '{"type":"ready","model":"stage1_served_v4","chunk_ms":160,"frame_ms":80,'
        '"diar_config":"off","column_lag_ms":0,"final_asr":"slow","final_chunk_ms":1120}'
    )[0]
    assert isinstance(ready, Ready) and ready.final_chunk_ms == 1120 and ready.final_asr == "slow"
