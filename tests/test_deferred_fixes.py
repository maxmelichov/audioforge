"""Deferred fixes (research/archive/VERIFICATION.md §6, items 5-8): explicit att_context_size, per-layer
streaming outputs, and from_layers: all heads in transcribe / StreamingSession."""
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.model import SpeechModel, StreamingSession
from audioforge.modules.fastconformer import FastConformerEncoder, StreamState
from audioforge.tokenizer import CharTokenizer

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- item 7: att_context_size wins
def test_explicit_att_context_size_wins_over_sizes():
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=1, n_heads=2, subsampling_channels=16,
                               causal=True, att_context_size=[16, 3], att_context_sizes=[[16, 1], [16, 3]])
    assert enc.att_context_size == [16, 3]
    assert enc.att_context_sizes == [[16, 1], [16, 3]]
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=1, n_heads=2, subsampling_channels=16,
                               causal=True, att_context_size=[8, 2], att_context_sizes=[[16, 1]])
    assert enc.att_context_size == [8, 2]
    assert [8, 2] in enc.att_context_sizes  # appended so training can sample it too


def test_only_att_context_sizes_keeps_first():  # nemo_import passes only att_context_sizes
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=1, n_heads=2, subsampling_channels=16,
                               causal=True, att_context_sizes=[[70, 13], [70, 1]])
    assert enc.att_context_size == [70, 13]
    assert enc.att_context_sizes == [[70, 13], [70, 1]]
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=1, n_heads=2, subsampling_channels=16)
    assert enc.att_context_size == [-1, -1] and enc.att_context_sizes == [[-1, -1]]


def test_voice_agent_frontend_att_override():
    cfg = yaml.safe_load((ROOT / "research/recipes/voice_agent_frontend.yaml").read_text())
    cfg["encoder"]["att_context_size"] = [16, 3]
    m = SpeechModel(cfg, CharTokenizer(list("abcd "))).eval()
    assert m.encoder.att_context_size == [16, 3]
    assert StreamingSession(m).chunk_mel == 32


# ---------------------------------------------------------------- item 8: stream_step(return_hidden)
@pytest.mark.parametrize("padding", ["ours", "nemo"])
def test_stream_step_return_hidden_matches_offline(padding):
    torch.manual_seed(0)
    att = [8, 1]
    enc = FastConformerEncoder(feat_in=40, d_model=32, n_layers=3, n_heads=2, subsampling_channels=16,
                               dropout=0.0, causal=True, att_context_size=att,
                               subsampling_padding=padding).eval()
    cs = att[1] + 1
    mel = torch.randn(1, 40, 5 * cs * 8)
    off, olen, hid = enc(mel, torch.tensor([mel.shape[-1]]), att, return_hidden=True)
    state, outs, hids = StreamState(), [], [[] for _ in enc.layers]
    for c in range(0, mel.shape[-1], cs * 8):
        last = c + cs * 8 >= mel.shape[-1]
        o, h, state = enc.stream_step(mel[..., c:c + cs * 8], state, att, final=last, return_hidden=True)
        assert len(h) == len(enc.layers) and torch.equal(h[-1], o)
        outs.append(o)
        for i, hi in enumerate(h):
            hids[i].append(hi)
    assert torch.allclose(torch.cat(outs, 1), off, atol=1e-4)
    for i in range(len(enc.layers)):
        assert torch.allclose(torch.cat(hids[i], 1), hid[i], atol=1e-4)


# ---------------------------------------------------------------- items 5, 6: from_layers: all
def _mixed_model(primary):
    torch.manual_seed(0)
    cfg = {"preprocessor": {"n_mels": 40, "normalize": "none", "dither": 0.0},
           "encoder": {"d_model": 32, "n_layers": 3, "n_heads": 2, "subsampling_channels": 16, "causal": True,
                       "att_context_size": [16, 1], "dropout": 0.0},
           "heads": {"ctc": {"type": "ctc", "from_layers": "all"},
                     "tdt": {"type": "tdt", "pred_hidden": 32, "joint_hidden": 32},
                     "vad": {"type": "frame", "key": "vad", "from_layers": "all"},
                     "eou": {"type": "frame", "key": "eou"}},
           "decoding": {"primary": primary}}
    m = SpeechModel(cfg, CharTokenizer(list("abcd "))).eval()
    with torch.no_grad():  # a non-uniform mix, so reading the top layer instead would differ
        for p in m.layer_mix.values():
            p.copy_(torch.tensor([2.0, -1.0, 0.5]))
        m.heads["ctc"].proj.weight.mul_(8.0)  # sharper logits and a suppressed blank:
        m.heads["ctc"].proj.bias[:4] = -1e3  # special tokens (<pad>, <bos>, ...) and blank: non-trivial
        m.heads["ctc"].proj.bias[m.heads["ctc"].blank] = -1e3  # CTC output
    return m


def _stream(m, x, head=None):
    s = StreamingSession(m, head=head)
    for i in range(0, len(x), 800):
        s.feed(x[i:i + 800])
    s.feed([], final=True)
    return s


@pytest.mark.parametrize("n_samples", [20000, 23456])
def test_streaming_frame_head_from_layers_all_equals_analyze(n_samples):
    m = _mixed_model("tdt")
    x = (np.random.default_rng(3).standard_normal(n_samples) * 0.1).astype(np.float32)
    s = _stream(m, x)
    res = m.analyze([x], att_context_size=[16, 1])
    T = res["frames"][0]
    for k in ("vad", "eou"):
        ref = res[k][0, :T]
        assert len(s.frame_events[k]) == T
        assert torch.allclose(torch.tensor(s.frame_events[k]), ref, atol=1e-4), k
    # the layer mix matters: the top-layer VAD output differs
    xt, lens = m._pad([x])
    enc, elen = m.encode(xt, lens, [16, 1])
    assert not torch.allclose(m.heads["vad"].decode(enc, elen)[0, :T], res["vad"][0, :T], atol=1e-3)
    assert s.text == res["tdt"][0] == m.transcribe([x], att_context_size=[16, 1])[0]


def test_streaming_ctc_from_layers_all_equals_offline():
    m = _mixed_model("ctc")
    x = (np.random.default_rng(4).standard_normal(23456) * 0.1).astype(np.float32)
    s = _stream(m, x)
    res = m.analyze([x], att_context_size=[16, 1])
    tr = m.transcribe([x], att_context_size=[16, 1])[0]
    assert tr == res["ctc"][0]  # item 5: transcribe reads the layer mix like analyze
    assert s.text == tr  # item 6
    assert len(tr) > 0
    # the layer mix matters for the CTC logits
    xt, lens = m._pad([x])
    enc, elen, hid = m.encode(xt, lens, [16, 1], return_hidden=True)
    assert not torch.allclose(m.heads["ctc"](enc), m.heads["ctc"](m.head_input("ctc", enc, hid)), atol=1e-3)


def test_empty_audio_list():
    m = _mixed_model("tdt")
    assert m.transcribe([]) == []
    assert m.analyze([]) == {"frames": [], "frame_sec": m.frame_sec}
