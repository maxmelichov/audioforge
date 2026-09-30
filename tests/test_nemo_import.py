"""NeMo .nemo import: key mapping, NeMo-exact rel-pos attention, and cache-aware streaming of the
ported (NeMo-layout) encoder. Uses a tiny synthetic .nemo archive, so no download is needed; the
real-checkpoint test runs only if data/nemo/stt_en_fastconformer_hybrid_large_streaming_multi.nemo exists.
"""
import io
import math
import os
import re
import tarfile
from pathlib import Path

import pytest
import torch
import yaml

from audioforge.modules.relpos import RelPositionMultiHeadAttention, rel_positional_encoding
from audioforge.nemo_import import import_nemo, translate_config
from audioforge.tokenizer import SentencePieceTokenizer

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "nemo" / "stt_en_fastconformer_hybrid_large_streaming_multi.nemo"

# ours -> NeMo key names (inverse of nemo_import's mapping), to fabricate a NeMo-style checkpoint
_INV = [
    (r"encoder\.pre_encode\.convs\.0\.", "encoder.pre_encode.conv.0."),
    (r"encoder\.pre_encode\.convs\.1\.0\.", "encoder.pre_encode.conv.2."),
    (r"encoder\.pre_encode\.convs\.1\.1\.", "encoder.pre_encode.conv.3."),
    (r"encoder\.pre_encode\.convs\.2\.0\.", "encoder.pre_encode.conv.5."),
    (r"encoder\.pre_encode\.convs\.2\.1\.", "encoder.pre_encode.conv.6."),
    (r"\.ff([12])\.0\.", r".norm_feed_forward\1."),
    (r"\.ff([12])\.1\.", r".feed_forward\1.linear1."),
    (r"\.ff([12])\.4\.", r".feed_forward\1.linear2."),
    (r"\.norm_att\.", ".norm_self_att."),
    (r"\.att\.", ".self_attn."),
    (r"\.conv\.pw1\.", ".conv.pointwise_conv1."),
    (r"\.conv\.dw\.", ".conv.depthwise_conv."),
    (r"\.conv\.norm\.", ".conv.batch_norm."),
    (r"\.conv\.pw2\.", ".conv.pointwise_conv2."),
    (r"heads\.rnnt\.pred\.embed\.", "decoder.prediction.embed."),
    (r"heads\.rnnt\.pred\.lstm\.", "decoder.prediction.dec_rnn.lstm."),
    (r"heads\.rnnt\.joint\.out\.2\.", "joint.joint_net.2."),
    (r"heads\.rnnt\.joint\.", "joint."),
    (r"heads\.ctc\.proj\.", "ctc_decoder.decoder_layers.0."),
]


def nemo_config(vocab=40, d=32, causal=True, conv_norm="layer_norm"):
    return {
        "sample_rate": 16000,
        "tokenizer": {"model_path": "nemo:abc123_tokenizer.model", "type": "bpe"},
        "preprocessor": {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                         "dither": 1e-5, "features": 80, "frame_splicing": 1, "n_fft": 512, "normalize": "NA",
                         "pad_to": 0, "sample_rate": 16000, "window": "hann", "window_size": 0.025,
                         "window_stride": 0.01},
        "spec_augment": {"freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width": 0.05},
        "encoder": {"_target_": "nemo.collections.asr.modules.ConformerEncoder", "feat_in": 80, "feat_out": -1,
                    "n_layers": 2, "d_model": d, "n_heads": 4, "subsampling": "dw_striding",
                    "subsampling_factor": 8, "subsampling_conv_channels": 8, "causal_downsampling": causal,
                    "conv_context_size": "causal" if causal else None, "conv_kernel_size": 9,
                    "conv_norm_type": conv_norm, "self_attention_model": "rel_pos", "xscaling": True,
                    "untie_biases": True, "ff_expansion_factor": 4, "reduction": None,
                    "att_context_size": [[4, 1], [4, 0], [6, 2]] if causal else [-1, -1],
                    "att_context_style": "chunked_limited" if causal else "regular"},
        "decoder": {"_target_": "nemo.collections.asr.modules.RNNTDecoder", "blank_as_pad": True,
                    "vocab_size": vocab, "prednet": {"pred_hidden": 16, "pred_rnn_layers": 1}},
        "joint": {"jointnet": {"activation": "relu", "joint_hidden": 24, "encoder_hidden": d, "pred_hidden": 16},
                  "num_classes": vocab},
        "aux_ctc": {"ctc_loss_weight": 0.3, "decoder": {"_target_": "nemo.collections.asr.modules.ConvASRDecoder"}},
        "decoding": {"greedy": {"max_symbols": 10}},
        "target": "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel",
    }


@pytest.fixture(scope="module")
def fake_nemo(tmp_path_factory):
    """A tiny hybrid RNNT+CTC .nemo archive with random weights under NeMo's parameter names."""
    from audioforge.model import SpeechModel
    words = "the cat sat on a mat with hat and bat while rat ran far".split()
    tok = SentencePieceTokenizer.train([" ".join(words[i:] + words[:i]) for i in range(len(words))] * 5,
                                       vocab_size=40, specials=[])
    nc = nemo_config(vocab=tok.vocab_size)
    torch.manual_seed(0)
    src = SpeechModel(translate_config(nc), SentencePieceTokenizer(tok.model_bytes, [])).eval()
    with torch.no_grad():  # non-trivial rel-pos biases and LayerNorm affine params
        for n, p in src.named_parameters():
            if "pos_bias" in n or "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    sd = {}
    for k, v in src.state_dict().items():
        for pat, rep in _INV:
            k = re.sub(pat, rep, k)
        sd[k] = v.unsqueeze(-1) if k == "ctc_decoder.decoder_layers.0.weight" else v
    sd["preprocessor.featurizer.window"] = src.preprocessor.window.clone()
    sd["preprocessor.featurizer.fb"] = src.preprocessor.fb.clone()[None]
    path = tmp_path_factory.mktemp("nemo") / "tiny.nemo"
    with tarfile.open(path, "w") as tar:
        for name, data in [("model_config.yaml", yaml.safe_dump(nc).encode()),
                           ("abc123_tokenizer.model", tok.model_bytes)]:
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf = io.BytesIO()
        torch.save(sd, buf)
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
    return path, src


def test_import_maps_every_weight(fake_nemo):
    path, src = fake_nemo
    m = import_nemo(path)
    assert m.primary == "rnnt" and set(m.heads) == {"rnnt", "ctc"}
    assert m.tokenizer.vocab_size == src.tokenizer.vocab_size
    assert m.import_info["fb_max_abs_diff"] == 0.0
    for k, v in src.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
    audio = torch.randn(1, 16000) * 0.1
    lens = torch.tensor([16000])
    with torch.no_grad():
        a, _ = src.encode(audio, lens)
        b, _ = m.encode(audio, lens)
    assert torch.equal(a, b)


def test_relpos_matches_explicit_formula():
    """rel_shift-based scores == (q+u)k + (q+v)P[rel(i,j)] with rel = query pos - key pos, cache included."""
    torch.manual_seed(0)
    att = RelPositionMultiHeadAttention(16, 2).eval()
    with torch.no_grad():
        att.pos_bias_u.normal_()
        att.pos_bias_v.normal_()
    x_all = torch.randn(1, 7, 16)
    C = 3
    with torch.no_grad():
        k_c = att.linear_k(x_all[:, :C]).view(1, C, 2, 8).transpose(1, 2)
        v_c = att.linear_v(x_all[:, :C]).view(1, C, 2, 8).transpose(1, 2)
        out, _ = att(x_all[:, C:], cache=(k_c, v_c))
        # explicit reference
        L, T = 7, 4
        q = att.linear_q(x_all[:, C:]).view(T, 2, 8)
        k = att.linear_k(x_all).view(L, 2, 8)
        v = att.linear_v(x_all).view(L, 2, 8)
        pe = rel_positional_encoding(L, 16)[0]  # index m <-> relative position L-1-m
        p = att.linear_pos(pe).view(-1, 2, 8)
        s = torch.empty(2, T, L)
        for i in range(T):
            for j in range(L):
                rel = (C + i) - j
                m_ = L - 1 - rel
                s[:, i, j] = ((q[i] + att.pos_bias_u) * k[j]).sum(-1) + ((q[i] + att.pos_bias_v) * p[m_]).sum(-1)
        w = (s / math.sqrt(8)).softmax(-1)
        ref = att.linear_out(torch.einsum("hij,jhd->ihd", w, v).reshape(1, T, 16))
    assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("n_mel", [96, 97, 131, 160, 23])
def test_ported_encoder_streaming_matches_offline(fake_nemo, n_mel):
    """Cache-aware stream_step on the NeMo-layout encoder (causal CausalConv2D subsampling, rel-pos) ==
    offline chunked-limited forward, for every trained context size, incl. a partial last chunk."""
    path, _ = fake_nemo
    m = import_nemo(path)
    enc = m.encoder
    torch.manual_seed(n_mel)
    feats = torch.randn(1, 80, n_mel)
    for att in enc.att_context_sizes:
        with torch.no_grad():
            off, olen = enc(feats, torch.tensor([n_mel]), att)
        cs = enc.stream_chunk_frames(att)
        st, outs = None, []
        for s in range(0, n_mel, cs):
            y, st = enc.stream_step(feats[..., s:s + cs], st, att, final=s + cs >= n_mel)
            outs.append(y)
        on = torch.cat(outs, 1)
        assert on.shape[1] == int(olen), (att, on.shape, olen)
        assert torch.allclose(on, off, atol=1e-5), (att, (on - off).abs().max())


@pytest.mark.skipif(not REAL.exists(), reason="real .nemo checkpoint not downloaded")
def test_real_checkpoint_streams_like_offline():
    from audioforge.model import StreamingSession
    ls = sorted((ROOT / "data" / "librispeech").rglob("1089-134686-0001.flac"))
    if not ls:
        pytest.skip("LibriSpeech test-clean not available")
    from audioforge.teachers import load_audio
    m = import_nemo(REAL)
    audio = load_audio(ls[0])
    x, lens = torch.tensor(audio)[None], torch.tensor([len(audio)])
    feats, fl = m.preprocessor(x, lens)
    att = [70, 13]
    with torch.no_grad():
        off, _ = m.encoder(feats, fl, att)
    cs, T, st, outs = m.encoder.stream_chunk_frames(att), int(fl), None, []
    for s in range(0, T, cs):
        y, st = m.encoder.stream_step(feats[..., s:s + cs], st, att, final=s + cs >= T)
        outs.append(y)
    assert torch.allclose(torch.cat(outs, 1), off, atol=1e-4)
    text = m.transcribe([audio], head="rnnt", att_context_size=att)[0]
    assert text == "stuff it into you his belly counselled him"
    sess = StreamingSession(m, "rnnt", att)
    for s in range(0, len(audio), 1600):
        sess.feed(audio[s:s + 1600])
    assert sess.feed(audio[:0], final=True) == text


# --------------------------------------------------------------------------- nemotron-speech-streaming-en-0.6b style
# (research/archive/ENC_0P6B.md): use_bias false, 2-layer LSTM prediction net, 128 mels, xscaling off, aux_ctc stub without
# CTC weights, plain EncDecRNNTBPEModel target.
REAL_0P6B = ROOT / "data" / "nemo" / "nemotron-speech-streaming-en-0.6b.nemo"
_NO_BIAS = re.compile(r"encoder\.layers\.\d+\.(ff[12]\.[14]|att\.linear_(q|k|v|out)|conv\.(pw1|dw|pw2))\.bias")


def nemotron_config(vocab=40, d=32):
    nc = nemo_config(vocab=vocab, d=d, causal=True, conv_norm="layer_norm")
    nc["preprocessor"]["features"] = 128
    nc["encoder"].update(feat_in=128, use_bias=False, xscaling=False, n_heads=4,
                         att_context_size=[[70, 13], [70, 6], [70, 1], [70, 0]])
    nc["decoder"]["prednet"]["pred_rnn_layers"] = 2
    nc["aux_ctc"] = {"ctc_loss_weight": 0.3, "decoder": {"_target_": "nemo.collections.asr.modules.ConvASRDecoder",
                                                          "feat_in": None, "num_classes": -1, "vocabulary": []}}
    nc["target"] = "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel"
    return nc


@pytest.fixture(scope="module")
def fake_nemotron(tmp_path_factory):
    """A tiny RNNT-only .nemo with the 0.6B model's config quirks; the bias-free tensors are absent from the archive."""
    from audioforge.model import SpeechModel
    words = "the cat sat on a mat with hat and bat while rat ran far".split()
    tok = SentencePieceTokenizer.train([" ".join(words[i:] + words[:i]) for i in range(len(words))] * 5,
                                       vocab_size=40, specials=[])
    nc = nemotron_config(vocab=tok.vocab_size)
    torch.manual_seed(1)
    src = SpeechModel(translate_config(nc, nemo_keys=[]), SentencePieceTokenizer(tok.model_bytes, [])).eval()
    assert set(src.heads) == {"rnnt"} and src.heads["rnnt"].pred.lstm.num_layers == 2
    with torch.no_grad():
        for n, p in src.named_parameters():
            if "pos_bias" in n or "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
            if _NO_BIAS.fullmatch(n):
                p.zero_()  # NeMo has no such tensor; the import fills zeros
    sd = {}
    for k, v in src.state_dict().items():
        if _NO_BIAS.fullmatch(k):
            continue
        for pat, rep in _INV:
            k = re.sub(pat, rep, k)
        sd[k] = v
    sd["preprocessor.featurizer.window"] = src.preprocessor.window.clone()
    sd["preprocessor.featurizer.fb"] = src.preprocessor.fb.clone()[None]
    path = tmp_path_factory.mktemp("nemo") / "tiny_nemotron.nemo"
    with tarfile.open(path, "w") as tar:
        for name, data in [("model_config.yaml", yaml.safe_dump(nc).encode()),
                           ("abc123_tokenizer.model", tok.model_bytes)]:
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf = io.BytesIO()
        torch.save(sd, buf)
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
    return path, src, sd


def test_import_nemotron_style_strict_tensor_count(fake_nemotron):
    """Every archive tensor is loaded exactly once, exactly the 11 bias-free tensors per layer are zero-filled, no CTC
    head is built from the aux_ctc stub, the 2-layer LSTM maps, and the encoder output is bit-identical."""
    path, src, nemo_sd = fake_nemotron
    m = import_nemo(path)
    assert set(m.heads) == {"rnnt"} and m.primary == "rnnt"
    assert m.heads["rnnt"].pred.lstm.num_layers == 2
    assert m.cfg["heads"]["rnnt"]["weight"] == 1.0
    n_layers = len(m.encoder.layers)
    info = m.import_info
    assert info["n_tensors"] == len(nemo_sd)
    assert info["n_loaded"] == len(nemo_sd) - 2  # window + fb are recomputed, not loaded
    assert info["zero_biases"] == 11 * n_layers
    assert info["n_loaded"] + info["zero_biases"] == len(m.state_dict())
    assert info["fb_max_abs_diff"] == 0.0 and m.preprocessor.n_mels == 128
    assert m.encoder.xscale is None and m.encoder.att_context_sizes == [[70, 13], [70, 6], [70, 1], [70, 0]]
    for k, v in src.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
    audio = torch.randn(1, 16000) * 0.1
    lens = torch.tensor([16000])
    with torch.no_grad():
        a, _ = src.encode(audio, lens)
        b, _ = m.encode(audio, lens)
    assert torch.equal(a, b)
    # the config alone (no tensor names) would still describe a CTC head; the importer checks the checkpoint
    nc = yaml.safe_load(tarfile.open(path).extractfile("./model_config.yaml").read())
    assert "ctc" in translate_config(nc)["heads"] and "ctc" not in translate_config(nc, nemo_sd.keys())["heads"]


def test_nemotron_style_streams_like_offline(fake_nemotron):
    path, _, _ = fake_nemotron
    enc = import_nemo(path).encoder
    torch.manual_seed(0)
    n_mel = 173
    feats = torch.randn(1, 128, n_mel)
    for att in ([70, 13], [70, 1]):
        with torch.no_grad():
            off, olen = enc(feats, torch.tensor([n_mel]), att)
        cs = enc.stream_chunk_frames(att)
        st, outs = None, []
        for s in range(0, n_mel, cs):
            y, st = enc.stream_step(feats[..., s:s + cs], st, att, final=s + cs >= n_mel)
            outs.append(y)
        on = torch.cat(outs, 1)
        assert on.shape[1] == int(olen)
        assert torch.allclose(on, off, atol=1e-5), (att, (on - off).abs().max())


@pytest.mark.skipif(not (REAL_0P6B.exists() and os.environ.get("AUDIOFORGE_BIG_TESTS")),
                    reason="needs the 2.4 GB checkpoint and AUDIOFORGE_BIG_TESTS=1 (loads ~2.5 GB; machine rules)")
def test_real_0p6b_tensor_counts():
    torch.set_num_threads(2)
    m = import_nemo(REAL_0P6B)
    info = m.import_info
    assert (info["n_tensors"], info["n_loaded"], info["zero_biases"]) == (653, 651, 264)
    assert info["n_loaded"] + info["zero_biases"] == len(m.state_dict())
    assert m.num_params() == 618_118_161 + 442_368  # checkpoint params + the zero biases
    assert (m.encoder.d_model, len(m.encoder.layers), m.encoder.layers[0].att.h) == (1024, 24, 8)
    assert m.tokenizer.vocab_size == 1024 and set(m.heads) == {"rnnt"}
    assert m.cfg["nemo_source"]["license"] == "NVIDIA Open Model License"
    assert info["fb_max_abs_diff"] < 1e-6 and info["window_max_abs_diff"] < 1e-6
