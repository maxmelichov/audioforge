"""NeMo Sortformer (.nemo SortformerEncLabelModel) import: key mapping, q/k/v fusion and a numeric check of our
SortformerHead(norm_first=False, out_pre_relu=True) against a re-implementation of NeMo's forward
(transformer_encoders.TransformerEncoder post-LN blocks + SortformerModules.forward_speaker_logits).
Uses a tiny synthetic archive; the real-checkpoint test runs only if
data/nemo/diar_streaming_sortformer_4spk-v2.nemo exists.

Also nvidia/Nemotron-3-Diarization (feature stacking + pre-LN RoPE transformer + sub-pixel upsample, 8 speakers):
nemo_import.Nemotron3Diarizer vs an independent re-implementation of the NeMo/HF-transformers forward on a tiny
synthetic bf16 archive, strict load counts, .afm round trip, and the real checkpoints when downloaded.
"""
import io
import math
import re
import tarfile
from pathlib import Path

import pytest
import torch
import yaml
from test_nemo_import import _INV, nemo_config

from audioforge.heads.audio import SortformerHead
from audioforge.nemo_import import LICENSES, import_nemo, license_of, translate_config

torch.set_num_threads(2)  # machine rule: CPU, 2 threads

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "nemo" / "diar_streaming_sortformer_4spk-v2.nemo"
REAL_V21 = ROOT / "data" / "nemo" / "diar_streaming_sortformer_4spk-v2.1.nemo"
REAL_N3 = ROOT / "data" / "nemo" / "Nemotron-3-Diarization.nemo"
D_TF, N_TF, H_TF, S = 16, 2, 4, 4


def sortformer_config(d=32):
    nc = nemo_config(d=d, causal=False, conv_norm="batch_norm")
    for k in ("tokenizer", "decoder", "joint", "aux_ctc", "decoding", "spec_augment"):
        nc.pop(k)
    nc.update(target="nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
              streaming_mode=True, pil_weight=0.5, max_num_of_spks=S,
              model_defaults={"fc_d_model": d, "tf_d_model": D_TF},
              sortformer_modules={"_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
                                  "num_spks": S, "fc_d_model": d, "tf_d_model": D_TF, "dropout_rate": 0.5,
                                  "chunk_len": 188, "chunk_right_context": 1, "fifo_len": 0,
                                  "spkcache_len": 188, "spkcache_update_period": 188},
              transformer_encoder={"_target_": "nemo.collections.asr.modules.transformer.transformer_encoders."
                                               "TransformerEncoder",
                                   "hidden_size": D_TF, "inner_size": 4 * D_TF, "num_attention_heads": H_TF,
                                   "num_layers": N_TF, "hidden_act": "relu", "pre_ln": False,
                                   "pre_ln_final_layer_norm": True, "ffn_dropout": 0.5})
    return nc


def nemo_head_weights(d):
    g = torch.Generator().manual_seed(1)
    r = lambda *s: 0.2 * torch.randn(*s, generator=g)  # noqa: E731
    sd = {"sortformer_modules.encoder_proj.weight": r(D_TF, d), "sortformer_modules.encoder_proj.bias": r(D_TF),
          "sortformer_modules.first_hidden_to_hidden.weight": r(D_TF, D_TF),
          "sortformer_modules.first_hidden_to_hidden.bias": r(D_TF),
          "sortformer_modules.single_hidden_to_spks.weight": r(S, D_TF),
          "sortformer_modules.single_hidden_to_spks.bias": r(S),
          "sortformer_modules.hidden_to_spks.weight": r(S, 2 * D_TF),  # unused by NeMo's forward
          "sortformer_modules.hidden_to_spks.bias": r(S)}
    for n in range(N_TF):
        p = f"transformer_encoder.layers.{n}."
        for x in ("query_net", "key_net", "value_net", "out_projection"):
            sd[p + f"first_sub_layer.{x}.weight"], sd[p + f"first_sub_layer.{x}.bias"] = r(D_TF, D_TF), r(D_TF)
        for ln in ("layer_norm_1", "layer_norm_2"):
            sd[p + f"{ln}.weight"], sd[p + f"{ln}.bias"] = 1 + r(D_TF), r(D_TF)
        sd[p + "second_sub_layer.dense_in.weight"], sd[p + "second_sub_layer.dense_in.bias"] = r(4 * D_TF, D_TF), r(4 * D_TF)
        sd[p + "second_sub_layer.dense_out.weight"], sd[p + "second_sub_layer.dense_out.bias"] = r(D_TF, 4 * D_TF), r(D_TF)
    return sd


def nemo_reference_logits(sd, emb, valid):
    """NeMo: TransformerEncoder(post-LN; MHA with q,k scaled by head_dim**-0.25; -10000 key mask)
    then relu -> first_hidden_to_hidden -> relu -> single_hidden_to_spks."""
    F = torch.nn.functional
    lin = lambda x, k: F.linear(x, sd[k + ".weight"], sd[k + ".bias"])  # noqa: E731
    x = lin(emb, "sortformer_modules.encoder_proj")
    B, T, _ = x.shape
    hd = D_TF // H_TF
    mask = (1 - valid.float())[:, None, None, :] * -10000.0
    for n in range(N_TF):
        p = f"transformer_encoder.layers.{n}."
        split = lambda t: t.view(B, T, H_TF, hd).transpose(1, 2) / math.sqrt(math.sqrt(hd))  # noqa: E731
        q, k = split(lin(x, p + "first_sub_layer.query_net")), split(lin(x, p + "first_sub_layer.key_net"))
        v = lin(x, p + "first_sub_layer.value_net").view(B, T, H_TF, hd).transpose(1, 2)
        a = ((q @ k.transpose(-1, -2)) + mask).softmax(-1) @ v
        a = lin(a.transpose(1, 2).reshape(B, T, D_TF), p + "first_sub_layer.out_projection")
        x = F.layer_norm(a + x, (D_TF,), sd[p + "layer_norm_1.weight"], sd[p + "layer_norm_1.bias"])
        f = lin(torch.relu(lin(x, p + "second_sub_layer.dense_in")), p + "second_sub_layer.dense_out")
        x = F.layer_norm(f + x, (D_TF,), sd[p + "layer_norm_2.weight"], sd[p + "layer_norm_2.bias"])
    h = lin(torch.relu(x), "sortformer_modules.first_hidden_to_hidden")
    return lin(torch.relu(h), "sortformer_modules.single_hidden_to_spks")


@pytest.fixture(scope="module")
def fake_sortformer(tmp_path_factory):
    from audioforge.model import SpeechModel
    nc = sortformer_config()
    torch.manual_seed(0)
    src = SpeechModel(translate_config(nc)).eval()
    sd = {}
    for k, v in src.state_dict().items():
        if k.startswith("heads."):
            continue
        for pat, rep in _INV:
            k = re.sub(pat, rep, k)
        sd[k] = v
    head = nemo_head_weights(32)
    sd.update(head)
    sd["preprocessor.featurizer.window"] = src.preprocessor.window.clone()
    sd["preprocessor.featurizer.fb"] = src.preprocessor.fb.clone()[None]
    path = tmp_path_factory.mktemp("nemo") / "tiny_sortformer.nemo"
    with tarfile.open(path, "w") as tar:
        for name, data in [("model_config.yaml", yaml.safe_dump(nc).encode())]:
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf = io.BytesIO()
        torch.save(sd, buf)
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
    return path, src, head, len(sd)


def test_sortformer_import_matches_nemo_forward(fake_sortformer):
    path, src, head_sd, n_nemo = fake_sortformer
    m = import_nemo(path)
    assert list(m.heads) == ["diar"] and m.tokenizer is None
    h = m.cfg["heads"]["diar"]
    assert h["type"] == "sortformer" and h["norm_first"] is False and h["out_pre_relu"] and not h["pos_emb"]
    # 2 frontend + 2 unused hidden_to_spks tensors dropped; 3 q/k/v -> 1 in_proj per layer and weight/bias
    assert m.import_info["n_tensors"] == n_nemo
    assert m.import_info["n_loaded"] == n_nemo - 4 - 2 * 2 * N_TF == len(m.state_dict())
    for k, v in src.state_dict().items():
        if not k.startswith("heads."):
            assert torch.equal(m.state_dict()[k], v), k
    torch.manual_seed(1)
    enc = torch.randn(2, 11, 32)
    elen = torch.tensor([11, 7])
    valid = torch.arange(11)[None] < elen[:, None]
    with torch.no_grad():
        ours = m.heads["diar"](enc, elen)
        ref = nemo_reference_logits(head_sd, enc, valid)
    assert torch.allclose(ours[valid], ref[valid], atol=1e-5), (ours[valid] - ref[valid]).abs().max()


def test_sortformer_head_defaults_unchanged():
    """The new layout knobs are opt-in: default keys/structure are the original head's."""
    torch.manual_seed(0)
    h = SortformerHead(32)
    assert [k for k in h.state_dict() if k.startswith("out.")] == ["out.0.weight", "out.0.bias",
                                                                    "out.2.weight", "out.2.bias"]
    assert h.tf.layers[0].norm_first is True
    torch.manual_seed(0)
    h2 = SortformerHead(32, norm_first=True, out_pre_relu=False)
    x, n = torch.randn(1, 5, 32), torch.tensor([5])
    h.eval(), h2.eval()
    assert torch.equal(h(x, n), h2(x, n))


def test_license_lookup():
    assert license_of("nvidia/diar_streaming_sortformer_4spk-v2") == "CC-BY-4.0"
    assert license_of("data/nemo/diar_streaming_sortformer_4spk-v2.1.nemo") == LICENSES[
        "diar_streaming_sortformer_4spk-v2.1"] != "CC-BY-4.0"


@pytest.mark.skipif(not REAL.exists(), reason="real Sortformer .nemo not downloaded")
def test_real_sortformer_checkpoint():
    torch.set_num_threads(2)
    m = import_nemo(REAL)
    assert m.import_info["n_tensors"] == 990 and m.import_info["n_loaded"] == 914 == len(m.state_dict())
    assert m.import_info["fb_max_abs_diff"] < 1e-6
    assert m.cfg["preprocessor"]["n_mels"] == 128 and m.cfg["preprocessor"]["normalize"] == "NA"
    assert m.cfg["nemo_source"]["license"] == "CC-BY-4.0"
    with torch.no_grad():
        p = m.heads["diar"](*m.encode(torch.zeros(1, 32000), torch.tensor([32000]))).sigmoid()
    assert p.shape == (1, 26, 4) and float(p.max()) < 0.5  # silence -> no speaker


@pytest.mark.skipif(not REAL_V21.exists(), reason="real Sortformer v2.1 .nemo not downloaded")
def test_real_sortformer_v2_1_checkpoint():
    """v2.1 = v2's architecture: the same importer, strict load, same counts; license tagged."""
    torch.set_num_threads(2)
    m = import_nemo(REAL_V21)
    assert m.import_info["n_tensors"] == 990 and m.import_info["n_loaded"] == 914 == len(m.state_dict())
    assert m.cfg["nemo_source"]["license"] == "NVIDIA Open Model License"
    with torch.no_grad():
        p = m.heads["diar"](*m.encode(torch.zeros(1, 32000), torch.tensor([32000]))).sigmoid()
    assert p.shape == (1, 26, 4) and float(p.max()) < 0.5


# --------------------------------------------------------------------------- Nemotron-3-Diarization
N3 = dict(mels=16, d=32, layers=2, heads=4, tf=16, spk=8, f=8)


def nemotron_config():
    return {"target": "nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel",
            "sample_rate": 16000, "streaming_mode": True, "high_resolution": True, "output_subsampling_factor": 1,
            "max_num_of_spks": N3["spk"], "nemo_version": "3.0.0",
            "preprocessor": {"_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                             "normalize": "NA", "window_size": 0.025, "sample_rate": 16000, "window_stride": 0.01,
                             "window": "hann", "features": N3["mels"], "n_fft": 512, "frame_splicing": 1,
                             "dither": 1e-5},
            "sortformer_modules": {"_target_": "nemo.collections.asr.modules.sortformer_modules.SortformerModules",
                                   "num_spks": N3["spk"], "fc_d_model": N3["d"], "tf_d_model": N3["tf"],
                                   "spkcache_len": 264, "fifo_len": 0, "chunk_len": 264,
                                   "spkcache_update_period": 264, "spkcache_sil_frames_per_spk": 1,
                                   "strong_boost_rate": 0.75},
            "encoder": {"_target_": "nemo.collections.asr.modules.TransformerEncoder", "feat_in": N3["mels"],
                        "feat_out": -1, "n_layers": N3["layers"], "d_model": N3["d"], "n_heads": N3["heads"],
                        "subsampling": "feature_stacking", "subsampling_factor": N3["f"], "ff_expansion": 4.0,
                        "self_attention_model": "rope", "pos_emb_max_len": 5000, "xscaling": False,
                        "qkv_bias": False, "qk_norm": False, "pre_block_norm": True, "attn_mode": "full"}}


def nemotron_weights():
    g = torch.Generator().manual_seed(3)
    r = lambda *s: 0.2 * torch.randn(*s, generator=g)  # noqa: E731
    d, tf, S, f, M = N3["d"], N3["tf"], N3["spk"], N3["f"], N3["mels"]
    sd = {"encoder.pre_encode.proj.weight": r(d, f * M), "encoder.embed_norm.weight": 1 + r(d),
          "encoder.embed_norm.bias": r(d), "encoder.final_norm.weight": 1 + r(d), "encoder.final_norm.bias": r(d)}
    for n in range(N3["layers"]):
        p = f"encoder.layers.{n}."
        sd.update({p + "norm1.weight": 1 + r(d), p + "norm1.bias": r(d), p + "norm2.weight": 1 + r(d),
                   p + "norm2.bias": r(d), p + "attn.w_qkv.weight": r(3 * d, d), p + "attn.out_proj.weight": r(d, d),
                   p + "attn.out_proj.bias": r(d), p + "ffn.net.0.weight": r(4 * d, d), p + "ffn.net.0.bias": r(4 * d),
                   p + "ffn.net.3.weight": r(d, 4 * d), p + "ffn.net.3.bias": r(d)})
    m = "sortformer_modules."
    sd.update({m + "learnable_sil_emb": r(d), m + "hidden_to_spks.weight": r(S, 2 * tf), m + "hidden_to_spks.bias": r(S),
               m + "first_hidden_to_hidden.weight": r(tf, tf), m + "first_hidden_to_hidden.bias": r(tf),
               m + "single_hidden_to_spks.weight": r(S, tf), m + "single_hidden_to_spks.bias": r(S),
               m + "activity_head.0.weight": 1 + r(tf), m + "activity_head.0.bias": r(tf),
               m + "activity_head.1.weight": r(3, tf), m + "activity_head.1.bias": r(3),
               m + "encoder_proj.weight": r(tf, d), m + "encoder_proj.bias": r(tf),
               m + "subpixel_upsample.weight": r(f * tf, tf, 3), m + "subpixel_upsample.bias": r(f * tf)})
    return {k: v.bfloat16() for k, v in sd.items()}  # the real checkpoint is bf16


def nemotron_reference_hr(sd, mel, flen):
    """NeMo / HF Nemotron3DiarizationForAudioFrameClassification (one chunk): stack 8 frames -> proj -> LN ->
    pre-LN blocks with rotate-half RoPE (theta 1e4, positions from 0), key padding mask -> LN -> encoder_proj ->
    Conv1d sub-pixel upsample -> relu/Linear/relu/Linear. mel (B, M, T) -> 10 ms logits (B, T8, S)."""
    F = torch.nn.functional
    w = {k: v.float() for k, v in sd.items()}
    lin = lambda x, k, b=True: F.linear(x, w[k + ".weight"], w[k + ".bias"] if b else None)  # noqa: E731
    ln = lambda x, k: F.layer_norm(x, (x.shape[-1],), w[k + ".weight"], w[k + ".bias"])  # noqa: E731
    f, d, H = N3["f"], N3["d"], N3["heads"]
    x = mel.transpose(1, 2)
    x = F.pad(x, (0, 0, 0, -x.shape[1] % f))
    B, T, M = x.shape
    x = lin(x.reshape(B, T // f, M * f), "encoder.pre_encode.proj", b=False)
    T = x.shape[1]
    valid = torch.arange(T)[None] < ((flen + f - 1) // f)[:, None]
    hd = d // H
    inv = 1.0 / 10000 ** (torch.arange(0, hd, 2).float() / hd)
    ang = torch.arange(T).float()[:, None] * inv[None]
    cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
    rot = lambda t: torch.cat([-t[..., hd // 2:], t[..., : hd // 2]], -1)  # noqa: E731
    h = ln(x, "encoder.embed_norm")
    for n in range(N3["layers"]):
        p = f"encoder.layers.{n}."
        y = ln(h, p + "norm1")
        q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in F.linear(y, w[p + "attn.w_qkv.weight"]).chunk(3, -1))
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + torch.where(valid, 0.0, float("-inf"))[:, None, None, :]
        h = h + lin((att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d), p + "attn.out_proj")
        h = h + lin(F.gelu(lin(ln(h, p + "norm2"), p + "ffn.net.0")), p + "ffn.net.3")
    h = lin(ln(h, "encoder.final_norm"), "sortformer_modules.encoder_proj")
    h = F.conv1d(h.transpose(1, 2), w["sortformer_modules.subpixel_upsample.weight"],
                 w["sortformer_modules.subpixel_upsample.bias"], padding=1).transpose(1, 2).reshape(B, T * f, -1)
    h = lin(torch.relu(h), "sortformer_modules.first_hidden_to_hidden")
    return lin(torch.relu(h), "sortformer_modules.single_hidden_to_spks"), valid


@pytest.fixture(scope="module")
def fake_nemotron(tmp_path_factory):
    from audioforge.features import mel_filterbank
    nc, sd = nemotron_config(), nemotron_weights()
    sd["preprocessor.featurizer.window"] = torch.hann_window(400, periodic=False).bfloat16()
    sd["preprocessor.featurizer.fb"] = mel_filterbank(16000, 512, N3["mels"])[None].bfloat16()
    path = tmp_path_factory.mktemp("nemo") / "Nemotron-3-Diarization.nemo"  # the name selects the license
    with tarfile.open(path, "w") as tar:
        data = yaml.safe_dump(nc).encode()
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        buf = io.BytesIO()
        torch.save(sd, buf)
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = buf.tell()
        buf.seek(0)
        tar.addfile(info, buf)
    return path, sd


def test_nemotron_import_matches_reference_forward(fake_nemotron):
    from audioforge.nemo_import import Nemotron3Diarizer, is_nemotron_diar
    path, sd = fake_nemotron
    assert is_nemotron_diar(nemotron_config()) and not is_nemotron_diar(sortformer_config())
    m = import_nemo(path)
    assert isinstance(m, Nemotron3Diarizer) and m.tokenizer is None and list(m.heads) == ["diar"]
    assert m.cfg["arch"] == "nemotron3_diar" and m.cfg["nemo_source"]["license"] == "OpenMDW-1.1"
    assert m.cfg["heads"]["diar"]["num_spks"] == 8 and m.heads["diar"].num_spks == 8
    # dropped: 2 frontend buffers, 2 hidden_to_spks, 4 activity_head; nothing else, strict load
    assert m.import_info["n_tensors"] == len(sd) and m.import_info["n_loaded"] == len(sd) - 8 == len(m.state_dict())
    torch.manual_seed(0)
    audio = 0.1 * torch.randn(2, 16000)
    alen = torch.tensor([16000, 9000])  # 101 / 57 mel frames -> 13 / 8 encoder frames (padding + partial group)
    with torch.no_grad():
        mel, flen = m.preprocessor(audio, alen)
        enc, elen = m.encoder(mel, flen)
        valid = torch.arange(enc.shape[1])[None] < elen[:, None]
        ours = m.heads["diar"].forward_hr(enc, valid)
        ref, rvalid = nemotron_reference_hr(sd, mel, flen)
        assert torch.equal(valid, rvalid)
        v8 = valid.repeat_interleave(8, 1)
        assert torch.allclose(ours[v8], ref[v8], atol=1e-5), (ours[v8] - ref[v8]).abs().max()
        # 80 ms interface: logit(mean) / logit(max) of the 8 sub-frame probabilities
        p80 = m.heads["diar"](enc, elen).sigmoid()
        pr = ref.sigmoid().view(2, -1, 8, 8)
        assert torch.allclose(p80[valid], pr.mean(2)[valid], atol=1e-5)
        m.heads["diar"].pool = "max"
        assert torch.allclose(m.heads["diar"](enc, elen).sigmoid()[valid], pr.amax(2)[valid], atol=1e-5)
        m.heads["diar"].pool = "mean"
        # streaming interface: an uncached chunk step == the offline pass over the same frames
        e = enc[:1, :13]
        out = m.heads["diar"].forward_chunk(e[:, :0], e[:, :5], e[:, 5:10], e[:, 10:])
        assert torch.allclose(out, m.heads["diar"].forward_emb(e), atol=1e-6)


def test_nemotron_afm_round_trip(fake_nemotron, tmp_path):
    from audioforge.nemo_import import load_any
    from audioforge.train import save_model
    m = import_nemo(fake_nemotron[0])
    save_model(m, tmp_path / "n3.afm")
    m2 = load_any(tmp_path / "n3.afm")
    assert type(m2) is type(m) and m2.cfg == m.cfg
    assert m.state_dict().keys() == m2.state_dict().keys()
    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m.state_dict().items())
    x = 0.1 * torch.randn(1, 12000)
    with torch.no_grad():
        assert torch.equal(m.diarize(x), m2.diarize(x))
    assert m2.diarize(x).shape == (1 + 12000 // 1280, 8)  # 76 mel frames -> 10 encoder frames


def test_top_k_columns():
    from audioforge.nemo_import import top_k_columns
    p = torch.zeros(1, 6, 8)
    p[0, :4, 5], p[0, :2, 1], p[0, :3, 6], p[0, 0, 7] = 0.9, 0.8, 0.7, 0.6
    h = top_k_columns(p, 2)
    assert h[0].sum(0).tolist() == [0, 0, 0, 0, 0, 4, 3, 0]
    assert torch.equal(top_k_columns(p, None), (p > 0.5).float())
    assert top_k_columns(p, 2, lengths=torch.tensor([2]))[0, :2].sum(0).nonzero().flatten().tolist() == [1, 5]


@pytest.mark.skipif(not REAL_N3.exists(), reason="Nemotron-3-Diarization .nemo not downloaded")
def test_real_nemotron_checkpoint():
    torch.set_num_threads(2)
    m = import_nemo(REAL_N3)
    assert m.import_info["n_tensors"] == 363 and m.import_info["n_loaded"] == 355 == len(m.state_dict())
    assert m.import_info["fb_max_abs_diff"] < 1e-3  # bf16 copy of the same filterbank
    assert m.cfg["nemo_source"]["license"] == "OpenMDW-1.1" and m.cfg["heads"]["diar"]["n_layers"] == 31
    p = m.diarize(torch.zeros(32000))
    assert p.shape == (26, 8) and float(p.max()) < 0.5
