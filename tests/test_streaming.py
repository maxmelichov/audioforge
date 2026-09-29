import numpy as np
import pytest
import torch

from audioforge.model import SpeechModel, StreamingSession
from audioforge.modules.fastconformer import FastConformerEncoder, StreamState
from audioforge.tokenizer import CharTokenizer


@pytest.mark.parametrize("att", [[16, 0], [16, 3], [-1, 1], [8, 7]])
def test_cache_aware_streaming_equals_offline(att):
    torch.manual_seed(0)
    enc = FastConformerEncoder(feat_in=40, d_model=64, n_layers=3, n_heads=4, subsampling_channels=32,
                               dropout=0.0, causal=True, att_context_size=att).eval()
    cs = att[1] + 1
    n_chunks = 6
    mel = torch.randn(1, 40, n_chunks * cs * 8)
    off, _ = enc(mel, torch.tensor([mel.shape[-1]]), att)
    state, outs = StreamState(), []
    for c in range(n_chunks):
        o, state = enc.stream_step(mel[..., c * cs * 8:(c + 1) * cs * 8], state, att)
        outs.append(o)
    stream = torch.cat(outs, 1)
    assert stream.shape == off.shape
    assert torch.allclose(stream, off, atol=1e-4), (stream - off).abs().max()


def test_incremental_mel_matches_offline():
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 1, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [8, 1]},
           "heads": {"ctc": {"type": "ctc"}}}
    tok = CharTokenizer(list("ab"))
    m = SpeechModel(cfg, tok).eval()
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    ref, _ = m.preprocessor(torch.from_numpy(x)[None], torch.tensor([len(x)]))
    s = StreamingSession(m)
    pushed = []
    for i in range(0, len(x), 1234):  # odd-sized pushes, mel computed on the trimmed tail
        s.feed(x[i:i + 1234])
        pushed.append(s.mel_done)
    s.feed([], final=True)
    assert s.mel_done == ref.shape[-1]
    # the session's windowed mel of any frame range equals the offline mel
    s2 = StreamingSession(m)
    xt = torch.from_numpy(x)
    s2.sig = torch.cat([xt[:1], xt[1:] - 0.97 * xt[:-1]])
    assert torch.allclose(s2._mel(0, ref.shape[-1]), ref, atol=1e-4)
    assert torch.allclose(s2._mel(37, 51), ref[..., 37:51], atol=1e-4)


def test_streaming_session_matches_offline_decode():
    torch.manual_seed(0)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [16, 1], "dropout": 0.0},
           "heads": {"tdt": {"type": "tdt", "pred_hidden": 32, "joint_hidden": 32},
                     "vad": {"type": "frame", "key": "vad"}}}
    m = SpeechModel(cfg, CharTokenizer(list("abcd "))).eval()
    x = (np.random.default_rng(1).standard_normal(8000 * 3) * 0.1).astype(np.float32)
    x = x[: (len(x) // 1280) * 1280 - 160]  # length that yields whole chunks
    s = StreamingSession(m)
    for i in range(0, len(x), 800):
        s.feed(x[i:i + 800])
    s.feed([], final=True)
    offline = m.transcribe([x], att_context_size=[16, 1])[0]
    assert s.text == offline
    assert len(s.frame_events["vad"]) > 0


@pytest.mark.parametrize("att", [[4, 1], [5, 1], [-1, 0]])
@pytest.mark.parametrize("extra", [0, 21])  # 21 mel frames = a partial last chunk, not a multiple of 8
def test_stream_step_speaker_kernels_and_partial_chunk(att, extra):
    torch.manual_seed(0)
    enc = FastConformerEncoder(feat_in=40, d_model=64, n_layers=3, n_heads=4, subsampling_channels=32,
                               dropout=0.0, causal=True, att_context_size=att, speaker_kernel_layers=[0, 2]).eval()
    with torch.no_grad():  # speaker kernels start as identity; make them matter
        for p in enc.speaker_kernels.parameters():
            p.normal_(0, 0.1)
    cs, n_chunks = att[1] + 1, 5
    mel = torch.randn(1, 40, n_chunks * cs * 8 + extra)
    T = -(-mel.shape[-1] // 8)
    act = torch.rand(1, T)
    off, olen = enc(mel, torch.tensor([mel.shape[-1]]), att, spk_act=act)
    assert int(olen[0]) == T
    state, outs, t = StreamState(), [], 0
    for c in range(0, mel.shape[-1], cs * 8):
        n = min(cs, T - t)
        o, state = enc.stream_step(mel[..., c:c + cs * 8], state, att, spk_act=act[:, t:t + n])
        outs.append(o)
        t += n
    stream = torch.cat(outs, 1)
    assert stream.shape == off.shape
    assert torch.allclose(stream, off, atol=1e-4), (stream - off).abs().max()


@pytest.mark.parametrize("n_samples", [20000, 23456])
def test_streaming_session_any_length_matches_offline(n_samples):
    # the final partial chunk must yield exactly the offline number of frames (no decoding on padding)
    torch.manual_seed(0)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [16, 3], "dropout": 0.0},
           "heads": {"ctc": {"type": "ctc"}, "vad": {"type": "frame", "key": "vad"}}}
    m = SpeechModel(cfg, CharTokenizer(list("abcd "))).eval()
    x = (np.random.default_rng(2).standard_normal(n_samples) * 0.1).astype(np.float32)
    s = StreamingSession(m)
    for i in range(0, len(x), 800):
        s.feed(x[i:i + 800])
    s.feed([], final=True)
    xt, lens = m._pad([x])
    enc, elen = m.encode(xt, lens, [16, 3])
    vad = m.heads["vad"].decode(enc, elen)[0, : int(elen[0])]
    assert len(s.frame_events["vad"]) == int(elen[0])
    assert torch.allclose(torch.tensor(s.frame_events["vad"]), vad, atol=1e-4)
    assert s.text == m.transcribe([x], head="ctc", att_context_size=[16, 3])[0]
