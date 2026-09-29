"""audioforge/baselines/turn.py (dedicated turn-detection baselines as frame score tracks) and nemo_import.import_eou.

Fast unit tests of the VAD state machines / score-track builders; the model tests skip when the checkpoint is absent.
"""
from pathlib import Path

import numpy as np
import pytest

from audioforge.baselines import turn as B

ROOT = Path(__file__).resolve().parents[1]
EOU = ROOT / "data" / "nemo" / "parakeet_realtime_eou_120m-v1.nemo"


def test_frame_of_time_boundaries():
    assert B.frame_of_time(0.0) == 0
    assert B.frame_of_time(0.08) == 0      # decision at the end of frame 0 -> emitted at frame boundary 1
    assert B.frame_of_time(0.0801) == 1
    assert B.frame_of_time(0.16) == 1
    assert B.frame_of_time(0.192) == 2


def test_pipecat_vad_state_machine():
    p = np.array([0.9] * 10 + [0.1] * 10 + [0.9] * 3 + [0.1] * 3 + [0.9] * 8, np.float32)
    v = B.pipecat_vad(p)
    # 6 speaking chunks -> SPEAKING after chunk 5; 6 silent chunks -> QUIET after chunk 15
    assert v["starts"][0] == 5 and v["stops"][0] == 15
    assert v["state"][4] == 2 and v["state"][5] == 3 and v["state"][10] == 4 and v["state"][15] == 1
    # a 3-chunk burst never reaches SPEAKING (start needs 6), the final 8-chunk burst does
    assert list(v["starts"]) == [5, 31] and list(v["stops"]) == [15]
    sp = B.speech_chunks_pipecat(v)
    assert sp[:10].all() and not sp[10:20].any()


def test_pipecat_vad_state_incremental_matches_batch():
    rng = np.random.default_rng(0)
    p = (rng.random(400) > 0.45).astype(np.float32) * 0.9 + 0.05
    v = B.pipecat_vad(p)
    sm = B.PipecatVADState()
    for j, c in enumerate(p):
        st, stopped, started = sm.step(float(c))
        assert st == v["state"][j]
        assert stopped == (j in set(v["stops"].tolist())) and started == (j in set(v["starts"].tolist()))
        assert sm.speaking == (st in (2, 3))


def test_livekit_vad_end_of_speech():
    p = np.array([0.9] * 20 + [0.2] * 30 + [0.9] * 5, np.float32)
    v = B.livekit_vad(p)
    n_si = round(0.55 / B.CHUNK_SEC)  # 17 chunks
    assert list(v["sos"]) == [1, 51]  # min speech 0.05 s = 2 chunks
    assert list(v["eos"]) == [20 + n_si - 1]
    assert v["speech_end"][0] == pytest.approx(20 * B.CHUNK_SEC)
    assert v["speaking"][20 + n_si - 2] and not v["speaking"][20 + n_si - 1]


def test_silence_frames_and_held_scores():
    sp = np.zeros(50, bool)
    sp[5:20] = True  # speech chunks 5..19 end at 0.64 s
    s = B.silence_frames(sp, 20)
    assert (s[:8] == 0).all()  # frames ending inside speech (<= 0.64 s) -> 0
    assert s[8] == pytest.approx((0.72 - 0.64) / 0.08)
    assert s[10] == pytest.approx((0.88 - 0.64) / 0.08)
    h = B.held_scores(20, [(0.5, 0.7), (1.2, 0.9)], np.array([0.9]))
    assert (h[:6] == 0).all() and (h[6:11] == np.float32(0.7)).all() and (h[11:14] == 0).all()
    assert (h[14:] == np.float32(0.9)).all()


def test_frame_silence_triggers_and_held_frames():
    act = np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], np.float32)
    trig, res = B.frame_silence_triggers(act, 3)
    assert list(trig) == [5, 10] and list(res) == [7]
    h = B.held_frames(len(act), trig, [0.4, 0.8], res)
    assert list(h) == pytest.approx([0, 0, 0, 0, 0, 0.4, 0.4, 0, 0, 0, 0.8])


@pytest.mark.skipif(not EOU.exists(), reason="parakeet_realtime_eou_120m-v1.nemo not downloaded")
def test_import_eou_and_streaming_equivalence():
    import torch
    torch.set_num_threads(2)
    from audioforge.model import StreamingSession
    from audioforge.nemo_import import import_eou, import_nemo
    m = import_eou(EOU)
    assert m.eou_ids == {"<EOU>": 1024, "<EOB>": 1025}
    assert m.import_info["zero_biases"] == 187 and m.cfg["nemo_source"]["license"] == "NVIDIA Open Model License"
    assert list(m.encoder.att_context_size) == [70, 1]
    # the generic importer now also zero-fills use_bias=false encoders (0.6B import); it must agree with import_eou
    g = import_nemo(EOU)
    assert g.import_info["zero_biases"] == 187
    rng = np.random.default_rng(0)
    t = np.arange(int(2.4 * 16000)) / 16000
    x = (0.1 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0) + 0.01 * rng.standard_normal(len(t)))
    x = x.astype(np.float32)

    class S(StreamingSession):
        def _decode(self, f):
            self.fs = getattr(self, "fs", []) + [f.clone()]
            super()._decode(f)
    ss = S(m)
    for a in range(0, len(x), 2560):
        ss.feed(x[a:a + 2560])
    ss.feed(np.zeros(0, np.float32), final=True)
    toks, logpost, em = B.rnnt_frame_decode(m, x, watch_ids=[1024])
    with torch.no_grad():
        enc = m.encode(torch.from_numpy(x)[None], torch.tensor([len(x)]))[0][0]
    fs = torch.cat(ss.fs, 0)
    assert float((fs - enc[: len(fs)]).abs().max()) < 1e-3  # cache-aware streaming == chunked_limited mask
    assert [k for tt in toks for k in tt] == ss.tokens
    assert logpost.shape == (len(toks), 1) and np.all(logpost <= 0)
    assert em[:, 0].sum() == sum(1024 in tt for tt in toks)
