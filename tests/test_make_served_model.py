"""scripts/research/make_served_model.py: transplant heads.spk (block-4 tap) into the turn checkpoint, everything else
bit-identical, layer_mix.spk dropped, config differs in heads.spk only."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import load_model, save_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_served_model", ROOT / "scripts" / "research" / "make_served_model.py")
msm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msm)

ENC = {"d_model": 16, "n_layers": 4, "n_heads": 2, "subsampling_channels": 8, "causal": True,
       "att_context_size": [8, 1], "dropout": 0.0}
PRE = {"n_mels": 40, "normalize": "fixed", "dither": 0.0}


def _model(spk_cfg, seed=0, spk_seed=None):
    torch.manual_seed(seed)
    heads = {"rnnt": {"type": "rnnt", "pred_hidden": 8, "joint_hidden": 8},
             "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
             "spk": {"type": "speaker", "num_speakers": 5, "emb_dim": 8, **spk_cfg}}
    m = SpeechModel({"preprocessor": PRE, "encoder": ENC, "heads": heads}, CharTokenizer(list("ab ")))
    if spk_seed is not None:  # a differently trained speaker head on the same encoder
        g = torch.Generator().manual_seed(spk_seed)
        with torch.no_grad():
            for p in m.heads["spk"].parameters():
                p.copy_(torch.randn(p.shape, generator=g))
    return m.eval()


@pytest.fixture
def ckpts(tmp_path):
    turn = _model({"from_layers": "all", "weight": 0.3})
    spk = _model({"from_layers": [3], "weight": 1.0, "aam_weight": 1.0,
                  "distill": {"target": "spk_teacher", "weight": 0.5, "relational_weight": 1.0}}, spk_seed=7)
    # the speaker checkpoint shares the encoder / rnnt / vad weights with the turn checkpoint (head-only training)
    sd = {k: v for k, v in turn.state_dict().items() if not k.startswith("heads.spk.") and k != "layer_mix.spk"}
    missing = spk.load_state_dict(sd, strict=False)
    assert all(k.startswith("heads.spk.") for k in missing.missing_keys) and not missing.unexpected_keys
    pt, ps = tmp_path / "turn.afm", tmp_path / "spk.afm"
    save_model(turn, pt), save_model(spk, ps)
    return str(pt), str(ps), turn, spk


def test_transplant_and_verify(ckpts, tmp_path):
    pt, ps, turn, spk = ckpts
    cfg, sd, rep = msm.transplant(pt, ps)
    assert "layer_mix.spk" not in sd and rep["dropped"] == ["layer_mix.spk"]
    assert cfg["heads"]["spk"] == {"type": "speaker", "num_speakers": 5, "emb_dim": 8, "from_layers": [3],
                                   "aam_weight": 1.0, "weight": 0.3}  # distill (loss-only) dropped, served weight kept
    out = tmp_path / "served.afm"
    m = SpeechModel(cfg, turn.tokenizer)
    m.load_state_dict(sd, strict=True)
    save_model(m.eval(), out)
    v = msm.verify(str(out), pt, ps)
    assert v["spk_tensors_from_relational"] == len([k for k in spk.state_dict() if k.startswith("heads.spk.")])
    assert v["identical_non_spk_tensors"] == len(sd) - v["spk_tensors_from_relational"]
    assert v["layer_weights_spk"] == [0.0, 0.0, 0.0, 1.0]
    served = load_model(out, "cpu")
    # the served speaker head embeds exactly like the speaker checkpoint, the ASR exactly like the turn checkpoint
    x = torch.randn(1, 8000) * 0.1
    l = torch.tensor([8000])
    with torch.no_grad():
        e_s = served.heads["spk"].embed(served.head_input("spk", *_enc(served, x, l)), l_enc(served, x, l))
        e_r = spk.heads["spk"].embed(spk.head_input("spk", *_enc(spk, x, l)), l_enc(spk, x, l))
        assert torch.allclose(e_s, e_r, atol=1e-6)
        assert served.transcribe([x[0].numpy()]) == turn.transcribe([x[0].numpy()])


def _enc(m, x, l):
    enc, _, hid = m.encode(x, l, return_hidden=True)
    return enc, hid


def l_enc(m, x, l):
    return m.encode(x, l)[1]


def test_verify_rejects_a_changed_encoder_tensor(ckpts, tmp_path):
    pt, ps, turn, spk = ckpts
    cfg, sd, _ = msm.transplant(pt, ps)
    k = next(k for k in sd if k.startswith("encoder.") and sd[k].dtype.is_floating_point)
    sd[k] = sd[k] + 1e-3
    m = SpeechModel(cfg, turn.tokenizer)
    m.load_state_dict(sd, strict=True)
    out = tmp_path / "bad.afm"
    save_model(m.eval(), out)
    with pytest.raises(AssertionError, match="non-speaker tensors differ"):
        msm.verify(str(out), pt, ps)


def test_transplant_requires_block4_tap(ckpts, tmp_path):
    pt, ps, turn, spk = ckpts
    with pytest.raises(AssertionError, match="from_layers"):
        msm.transplant(pt, pt)  # the turn checkpoint's own all-layer head is not a block-4 tap


@pytest.mark.skipif(not (ROOT / "runs/served_model_build.json").exists(), reason="served model not built")
def test_real_build_report():
    r = json.loads((ROOT / "runs/served_model_build.json").read_text())
    assert r["verify"]["layer_weights_spk"][3] == 1.0 and r["verify"]["identical_non_spk_tensors"] >= 700
    assert r["wer"]["identical_hyps"] and r["wer"]["wer_served"] == r["wer"]["wer_trail6"]


def test_swap_vad_single_tap(ckpts, tmp_path):
    """--vad-head: heads.vad replaced by a single-tap head, layer_mix.vad dropped, everything else bit-identical."""
    pt, _, turn, _ = ckpts
    from audioforge.model import build_head
    torch.manual_seed(3)
    h = build_head({"type": "frame", "key": "vad", "hidden": 8, "from_layers": [1]}, 16)
    hp = tmp_path / "head.pt"
    torch.save({"cfg": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": [1], "weight": 0.5},
                "state_dict": h.state_dict()}, hp)
    cfg, sd, rep = msm.swap_vad(pt, str(hp))
    assert "layer_mix.vad" not in sd and rep["dropped"] == ["layer_mix.vad"]
    m = SpeechModel(cfg, turn.tokenizer)
    m.load_state_dict(sd, strict=True)
    out = tmp_path / "v2.afm"
    save_model(m.eval(), out)
    v = msm.verify_vad(str(out), pt, str(hp))
    assert v["vad_layer"] == 1
    served = load_model(out, "cpu")
    x, l = torch.randn(1, 8000) * 0.1, torch.tensor([8000])
    with torch.no_grad():
        enc, hid = _enc(served, x, l)
        assert torch.equal(served.head_input("vad", enc, hid), hid[1])
        assert torch.allclose(served.heads["vad"](hid[1]), h(hid[1]))
        assert served.transcribe([x[0].numpy()]) == turn.transcribe([x[0].numpy()])
    with pytest.raises(AssertionError, match="single-layer tap"):
        torch.save({"cfg": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all"},
                    "state_dict": h.state_dict()}, hp)
        msm.swap_vad(pt, str(hp))
