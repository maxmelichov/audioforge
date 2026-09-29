"""Speaker head: single-layer taps (from_layers: [k]), TitaNet distillation loss, teacher attach, head-only
training from a stage-1 checkpoint (research/SPK_HEAD.md)."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from audioforge.data import Collate, attach_teacher, segment_id, synthetic_dataset
from audioforge.heads.audio import SpeakerHead
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import Trainer, build_model, save_model

ENC = {"d_model": 16, "n_layers": 4, "n_heads": 2, "subsampling_channels": 8, "causal": True,
       "att_context_size": [8, 1], "dropout": 0.0}
PRE = {"n_mels": 40, "normalize": "fixed", "dither": 0.0}


def _model(spk_cfg, extra_heads=None):
    torch.manual_seed(0)
    heads = {"spk": {"type": "speaker", "num_speakers": 5, "emb_dim": 8, **spk_cfg}, **(extra_heads or {})}
    return SpeechModel({"preprocessor": PRE, "encoder": ENC, "heads": heads}).eval()


def _audio(n=2, T=6000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, generator=g) * 0.1
    return x, torch.tensor([T, T - 1500][:n])


# ---------------------------------------------------------------- from_layers taps
def test_single_layer_tap_reads_exactly_that_layer():
    for k in (0, 2, -1):
        m = _model({"from_layers": [k]})
        x, lens = _audio()
        enc, elen, hidden = m.encode(x, lens, return_hidden=True)
        e = m.head_input("spk", enc, hidden)
        assert torch.equal(e, hidden[k % len(m.encoder.layers)])
        assert "spk" not in m.layer_mix and m.layer_tap["spk"] == [k % 4]
        w = m.layer_weights("spk")
        assert w.tolist() == F.one_hot(torch.tensor(k % 4), 4).float().tolist()
    m = _model({"from_layers": 3})  # bare int
    assert m.layer_tap["spk"] == [3]


def test_sub_mix_over_listed_layers_and_validation():
    m = _model({"from_layers": [1, 3]})
    assert m.layer_mix["spk"].shape == (2,)
    x, lens = _audio()
    enc, elen, hidden = m.encode(x, lens, return_hidden=True)
    e = m.head_input("spk", enc, hidden)
    assert torch.allclose(e, 0.5 * (hidden[1] + hidden[3]), atol=1e-6)
    assert m.layer_weights("spk").tolist() == pytest.approx([0, 0.5, 0, 0.5])
    with pytest.raises(ValueError):
        _model({"from_layers": [4]})
    with pytest.raises(ValueError):
        _model({"from_layers": [1, 1]})
    # unchanged paths: all -> full mix, none -> the top layer / no weights
    ma = _model({"from_layers": "all"})
    assert ma.layer_mix["spk"].shape == (4,) and "spk" not in ma.layer_tap
    mt = _model({})
    assert mt.layer_weights("spk") is None and torch.equal(mt.head_input("spk", enc, hidden), enc)


# ---------------------------------------------------------------- distillation loss
def _reference_aam(head, e, y):
    cos = e @ F.normalize(head.W, dim=-1).T
    theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    target = torch.cos(theta + head.m)
    logits = torch.where(F.one_hot(y, cos.shape[1]).bool(), target, cos) * head.s
    return F.cross_entropy(logits.float(), y)


def test_speaker_loss_bit_identical_without_distill():
    torch.manual_seed(1)
    head = SpeakerHead(16, num_speakers=5, emb_dim=8).train()
    enc = torch.randn(3, 20, 16)
    elen = torch.tensor([20, 15, 9])
    y = torch.tensor([0, 3, 3])
    torch.manual_seed(2)
    got = head.loss(enc, elen, {"speaker": y})
    torch.manual_seed(2)
    head2 = SpeakerHead(16, num_speakers=5, emb_dim=8).train()
    head2.load_state_dict(head.state_dict())
    ref = _reference_aam(head2, head2.embed(enc, elen), y)
    assert torch.equal(got, ref)  # the pre-distillation code path, unchanged
    assert head.key == "speaker" and head.distill is None


def test_speaker_distill_loss():
    torch.manual_seed(3)
    head = SpeakerHead(16, num_speakers=5, emb_dim=8, distill={"target": "spk_teacher", "weight": 2.0},
                       aam_weight=0.0).train()
    assert head.key == "spk_teacher"
    enc = torch.randn(4, 12, 16)
    elen = torch.tensor([12, 12, 10, 5])
    t = torch.randn(4, 8)
    e = head.embed(enc, elen)
    l = head.loss(enc, elen, {"spk_teacher": t})
    ref = 2.0 * (1 - (e * F.normalize(t, dim=-1)).sum(-1)).mean()
    assert torch.allclose(l, ref, atol=1e-6)
    assert torch.allclose(head.loss(enc, elen, {"spk_teacher": e.detach()}), torch.zeros(()), atol=1e-5)
    # AAM term added only when speaker labels are present and aam_weight > 0
    head.aam_weight = 0.5
    y = torch.tensor([1, 2, 3, 4])
    both = head.loss(enc, elen, {"spk_teacher": t, "speaker": y})
    assert torch.allclose(both, ref + 0.5 * head.aam_loss(e, y), atol=1e-6)
    assert torch.allclose(head.loss(enc, elen, {"spk_teacher": t}), ref, atol=1e-6)
    # gradients flow to the pooling / embedding
    both.backward()
    assert head.emb[0].weight.grad is not None and head.emb[0].weight.grad.abs().sum() > 0


def test_model_forward_distill_head_skips_without_target_and_trains_with_it():
    m = _model({"from_layers": [1], "distill": {"target": "spk_teacher"}, "aam_weight": 0.0}).train()
    x, lens = _audio()
    out = m({"audio": x, "audio_len": lens, "speaker": torch.tensor([0, 1])})
    assert "loss_spk" not in out  # key spk_teacher missing -> skipped (multi-task batches)
    out = m({"audio": x, "audio_len": lens, "spk_teacher": torch.randn(2, 8)})
    assert out["loss_spk"].requires_grad and 0.0 <= float(out["loss_spk"]) <= 2.0


# ---------------------------------------------------------------- teacher cache attach
def test_segment_id_and_attach_teacher(tmp_path):
    data = synthetic_dataset("multitask", 4, seed=0)
    for i, ex in enumerate(data):
        ex.update(meeting="M1", start=1.5 * i)
    ids = [segment_id(e) for e in data]
    assert len(set(ids)) == 4 and ids[0].startswith("M1:0.000:")
    assert segment_id({"id": "103-1240-0000", "audio": np.zeros(3)}) == "103-1240-0000"
    emb = np.random.default_rng(0).standard_normal((3, 6)).astype(np.float32)
    np.savez(tmp_path / "t.npz", ids=np.array(ids[:3]), emb=emb)
    out = attach_teacher(data, str(tmp_path / "t.npz"))
    assert len(out) == 3 and all(np.array_equal(o["spk_teacher"], e) for o, e in zip(out, emb))
    assert "spk_teacher" not in data[0]  # input untouched
    with pytest.raises(KeyError):
        attach_teacher(data, {"file": str(tmp_path / "t.npz"), "missing": "error"})
    with pytest.raises(FileNotFoundError):
        attach_teacher(data, str(tmp_path / "nope.npz"))
    b = Collate(None)(out)
    assert b["spk_teacher"].shape == (3, 6) and b["spk_teacher"].dtype == torch.float32


# ---------------------------------------------------------------- head-only fine-tune from a stage-1 style checkpoint
def test_init_from_changes_from_layers_and_train_only(tmp_path):
    torch.manual_seed(0)
    src = SpeechModel({"preprocessor": PRE, "encoder": ENC,
                       "heads": {"ctc": {"type": "ctc"}, "vad": {"type": "frame", "key": "vad"},
                                 "spk": {"type": "speaker", "num_speakers": 5, "emb_dim": 8, "from_layers": "all",
                                         "weight": 0.3}}}, CharTokenizer(list("abcd ")))
    with torch.no_grad():
        src.layer_mix["spk"].normal_()
    path = tmp_path / "src.afm"
    save_model(src, path)
    recipe = {"init": {"from": str(path), "pretrained_lr_mult": 0, "train_only": ["heads.spk", "layer_mix.spk"]},
              "heads": {"ctc": {"weight": 0}, "vad": {"weight": 0},
                        "spk": {"type": "speaker", "num_speakers": 5, "from_layers": [2], "weight": 1.0}},
              "trainer": {"max_steps": 2, "batch_size": 4, "lr": 1e-2, "device": "cpu", "log_every": 1}}
    model, cfg = build_model(recipe, [])
    assert cfg["heads"]["spk"]["from_layers"] == [2] and cfg["heads"]["spk"]["emb_dim"] == 8  # merged per head
    assert "spk" not in model.layer_mix  # the source's layer_mix.spk was dropped, not an error
    ssd = src.state_dict()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert all(torch.equal(ssd[k], before[k]) for k in before if k in ssd)
    tr = Trainer(model, cfg)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("heads.spk.") for n in trainable)
    data = synthetic_dataset("multitask", 8, seed=0)
    for ex in data:
        ex["speaker"] = int(ex["speaker"]) % 5
    tr.fit(data, collate=Collate(model.tokenizer))
    after = model.state_dict()
    changed = [k for k in after if not torch.equal(before[k], after[k])]
    assert changed and all(k.startswith("heads.spk.") for k in changed)
    assert all(torch.equal(ssd[k], after[k]) for k in ssd if not k.startswith(("heads.spk.", "layer_mix.")))


def test_spk_recipes_parse():
    root = Path(__file__).resolve().parents[1] / "research" / "recipes"
    for n in (2, 4, 8, 12, 17):
        c = yaml.safe_load((root / f"spk_layer_sweep_L{n}.yaml").read_text())
        assert c["heads"]["spk"]["from_layers"] == [n - 1] and c["init"]["train_only"] == ["heads.spk", "layer_mix.spk"]
        assert c["trainer"]["max_steps"] == 1000 and c["trainer"]["batch_size"] == 6
        assert all(c["heads"][h]["weight"] == 0 for h in ("rnnt", "ctc", "vad", "eou", "diar", "turn"))
    c = yaml.safe_load((root / "spk_distill_titanet.yaml").read_text())
    spk = c["heads"]["spk"]
    assert spk["distill"] == {"target": "spk_teacher", "weight": 1.0} and spk["aam_weight"] == 0.0
    assert spk["from_layers"] == "all" and c["trainer"]["max_steps"] == 2000
    assert {e["name"] for e in c["data"]["mix"]} == {"ami_asr", "icsi_asr", "librispeech"}
    assert all("spk_teacher" in e for e in c["data"]["mix"]) and "spk_teacher" in c["data"]["val"]
    c = yaml.safe_load((root / "spk_distill_titanet_bestlayer.yaml").read_text())
    assert isinstance(c["heads"]["spk"]["from_layers"], list) and c["heads"]["spk"]["distill"]


def test_speaker_relational_loss():
    torch.manual_seed(5)
    head = SpeakerHead(16, num_speakers=5, emb_dim=8, distill={"target": "spk_teacher", "weight": 0.0, "relational_weight": 1.0},
                       aam_weight=0.0).train()
    enc = torch.randn(4, 12, 16)
    elen = torch.tensor([12, 12, 10, 5])
    e = head.embed(enc, elen)
    t = torch.randn(4, 192)  # teacher dimension may differ from the student's
    tn = F.normalize(t, dim=-1)
    off = ~torch.eye(4, dtype=torch.bool)
    ref = F.mse_loss((e @ e.T)[off], (tn @ tn.T)[off])
    assert torch.allclose(head.loss(enc, elen, {"spk_teacher": t}), ref, atol=1e-6)
    # identical relations -> zero loss (teacher = a rotation of the student)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    assert float(head.relational_loss(e, {"spk_teacher": e.detach() @ q})) < 1e-10
    # a single item gives a zero relational loss with a gradient path, not an error
    one = head.relational_loss(e[:1], {"spk_teacher": t[:1]})
    assert float(one) == 0.0 and one.requires_grad


def test_init_from_reinitialises_resized_head_tensor(tmp_path):
    torch.manual_seed(0)
    src = SpeechModel({"preprocessor": PRE, "encoder": ENC,
                       "heads": {"spk": {"type": "speaker", "num_speakers": 5, "emb_dim": 8, "from_layers": [1]}}})
    path = tmp_path / "src.afm"
    save_model(src, path)
    recipe = {"init": {"from": str(path), "pretrained_lr_mult": 0, "train_only": ["heads.spk"]},
              "heads": {"spk": {"num_speakers": 12}}, "trainer": {"device": "cpu"}}
    model, cfg = build_model(recipe, [])
    assert model.heads["spk"].W.shape == (12, 8)
    ssd, msd = src.state_dict(), model.state_dict()
    assert all(torch.equal(ssd[k], msd[k]) for k in ssd if k != "heads.spk.W")


def test_librispeech_speaker_offset_recipe_parses():
    root = Path(__file__).resolve().parents[1] / "research" / "recipes"
    c = yaml.safe_load((root / "spk_layer_sweep_L4_libri.yaml").read_text())
    ls = next(e for e in c["data"]["mix"] if e["name"] == "librispeech")["librispeech"]
    assert ls["speaker_offset"] == 1000 and c["heads"]["spk"]["num_speakers"] == 1251
    assert c["heads"]["spk"]["from_layers"] == [3] and c["trainer"]["max_steps"] == 1000
