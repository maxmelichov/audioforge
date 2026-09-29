"""KL anchor (R2 forgetting guard), mode-consistency (MCR-RNNT), WER gate and step checkpoints."""
import json

import numpy as np
import pytest
import torch

from audioforge.data import Collate, save_wav, synthetic_dataset, to_device
from audioforge.losses.consistency import FrozenTeacher, head_kl, transducer_kl
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import Trainer, build_model, load_model, save_model

CFG = {
    "preprocessor": {"n_mels": 40, "normalize": "fixed", "dither": 0.0},
    "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                "att_context_size": [8, 1], "dropout": 0.0},
    "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32}, "ctc": {"type": "ctc"}},
}
TRAIN = {"max_steps": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu", "log_every": 1}


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _data(n=8):
    return synthetic_dataset("multitask", n, seed=0)


def _model(cfg=CFG):
    torch.manual_seed(0)
    return SpeechModel(cfg, CharTokenizer(list("abcdefghijklmnopqrstuvwxyz '")))


def _logits(B=3, T=7, U1=5, V=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, U1, V, generator=g)


def test_transducer_kl_basic_properties():
    s, t = _logits(seed=0), _logits(seed=1)
    el, ul = torch.tensor([7, 5, 3]), torch.tensor([4, 2, 0])
    assert float(transducer_kl(s, s.clone(), el, ul)) == 0.0
    fwd = transducer_kl(s, t, el, ul, symmetric=False)
    rev = transducer_kl(t, s, el, ul, symmetric=False)
    sym = transducer_kl(s, t, el, ul, symmetric=True)
    assert float(fwd) > 0 and float(rev) > 0 and abs(float(sym) - 0.5 * float(fwd + rev)) < 1e-6
    # TDT layout: token and duration logits are separate distributions, KLs add
    tdt = transducer_kl(s, t, el, ul, n_tokens=4)
    ref = transducer_kl(s[..., :4], t[..., :4], el, ul) + transducer_kl(s[..., 4:], t[..., 4:], el, ul)
    assert abs(float(tdt) - float(ref)) < 1e-6


def test_transducer_kl_masks_padding():
    s, t = _logits(seed=0), _logits(seed=1)
    el, ul = torch.tensor([7, 5, 3]), torch.tensor([4, 2, 0])
    base = transducer_kl(s, t, el, ul, reduction="none")
    # garbage (including inf/nan) in padded frames and tokens must not change anything
    s2, t2 = torch.randn(3, 10, 8, 6) * 50, torch.randn(3, 10, 8, 6) * 50
    s2[:, :7, :5], t2[:, :7, :5] = s, t
    for b, (T, U) in enumerate(zip(el.tolist(), ul.tolist())):
        s2[b, T:], t2[b, :, U + 1:] = float("nan"), float("inf")
        s[b, T:], t[b, :, U + 1:] = 123.0, -77.0
    assert torch.allclose(transducer_kl(s2, t2, el, ul, reduction="none"), base, atol=1e-6)
    assert torch.allclose(transducer_kl(s, t, el, ul, reduction="none"), base, atol=1e-6)


def _batch(model, n=4):
    return to_device(Collate(model.tokenizer)(_data(n)), "cpu")


def test_head_kl_fused_matches_full_and_teacher_zero():
    m = _model().eval()
    teacher = FrozenTeacher(m, ["rnnt"])
    b = _batch(m)
    out = m(b, return_enc=True)
    e = out["enc"]
    te, _, _ = teacher.encode(e["feats"], e["flen"], e["att"])
    y, yl = b["text"], b["text_len"]
    assert float(head_kl(m.heads["rnnt"], e["enc"], teacher.heads["rnnt"], te, e["elen"], y, yl).detach()) < 1e-6
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    e = m(b, return_enc=True)["enc"]
    full = head_kl(m.heads["rnnt"], e["enc"], teacher.heads["rnnt"], te, e["elen"], y, yl, fused_batch_size=0)
    fused = head_kl(m.heads["rnnt"], e["enc"], teacher.heads["rnnt"], te, e["elen"], y, yl, fused_batch_size=3)
    assert float(full) > 1e-4 and np.isfinite(float(full)) and abs(float(full) - float(fused)) < 1e-5
    full.backward()  # gradients reach the student, never the teacher
    assert any(p.grad is not None for p in m.encoder.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    # on-demand joint equals the head's own joint
    z, el = teacher.joint_logits("rnnt", e["feats"], e["flen"], y, e["att"])
    assert z.shape[:3] == (4, int(el.max()), int(yl.max()) + 1)


def _init_from(tmp_path, trainer):
    src = _model()
    path = tmp_path / "src.afm"
    save_model(src, path)
    recipe = {"init": {"from": str(path), "pretrained_lr_mult": 0.1}, "heads": {"rnnt": {"weight": 1.0}},
              "trainer": {**TRAIN, **trainer}}
    return build_model(recipe, [])


def test_anchor_zero_for_identical_then_positive(tmp_path):
    model, cfg = _init_from(tmp_path, {"anchor": {"weight": 0.5, "heads": ["rnnt"], "symmetric": True}})
    tr = Trainer(model, cfg)
    b = _batch(model)
    model.eval()  # the joint's dropout is the only train/eval difference
    kl = tr._consistency(b, model(b, return_enc=True)["enc"])["kl_anchor"]
    assert abs(float(kl)) < 1e-6
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.05 * torch.randn_like(p))
    kl = tr._consistency(b, model(b, return_enc=True)["enc"])["kl_anchor"]
    assert float(kl) > 1e-5 and np.isfinite(float(kl))
    # teacher from an .afm path gives the same anchor as the in-memory copy
    tr2 = Trainer(model, {**cfg, "trainer": {**cfg["trainer"], "anchor": {"teacher": str(tmp_path / "src.afm")}}})
    kl2 = tr2._consistency(b, model(b, return_enc=True)["enc"])["kl_anchor"]
    assert abs(float(kl2) - float(kl)) < 1e-6
    # 2-step smoke through fit: kl is logged and the loss stays finite
    tr.fit(_data(), collate=Collate(model.tokenizer))
    assert "kl_anchor" in tr.history[-1] and np.isfinite(tr.history[-1]["loss"])
    assert tr.teacher.encoder.layers[0] is not model.encoder.layers[0]


def test_mode_consistency_smoke_causal():
    m = _model()
    cfg = {**CFG, "trainer": {**TRAIN, "mode_consistency": {"weight": 0.5, "contexts": [[-1, -1], [4, 1]]}}}
    tr = Trainer(m, cfg)
    tr.fit(_data(), collate=Collate(m.tokenizer))
    h = tr.history[-1]
    assert h["step"] == 2 and h["kl_mode"] > 0 and np.isfinite(h["loss"]) and "loss_rnnt_mode2" in h
    assert abs(h["loss"] - (h["loss_rnnt"] + h["loss_ctc"] + h["loss_rnnt_mode2"] + 0.5 * h["kl_mode"])) < 1e-3


def _gate_manifest(tmp_path, n=4):
    lines = []
    for i, ex in enumerate(_data(n)):
        p = tmp_path / f"u{i}.wav"
        save_wav(str(p), np.asarray(ex["audio"]))
        lines.append(json.dumps({"audio_filepath": str(p), "text": ex["text"].upper() + "."}))
    (tmp_path / "gate.jsonl").write_text("\n".join(lines))
    return str(tmp_path / "gate.jsonl")


def test_wer_gate_stops_run_and_checkpoints(tmp_path):
    man = _gate_manifest(tmp_path)
    m = _model()
    gate = {"manifest": man, "every": 2, "max_delta": -1, "n": 4, "head": "rnnt"}
    cfg = {**CFG, "trainer": {**TRAIN, "max_steps": 3, "wer_gate": gate, "checkpoint_every": 1}}
    tr = Trainer(m, cfg)
    out = str(tmp_path / "run.afm")
    tr.fit(_data(), collate=Collate(m.tokenizer), out=out)
    assert tr.stopped and tr.stopped["step"] == 2 and tr.history[-1]["step"] == 2
    assert tr.history[0]["step"] == 0 and "wer_gate" in tr.history[-1]
    # step 1 checkpoint is the last good one; the failing step writes none; model restored from it
    assert tr.checkpoints == [str(tmp_path / "run.step1.afm")] and tr.stopped["restored_from"] == tr.checkpoints[0]
    ck = load_model(tr.checkpoints[0])
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in ck.state_dict().items())
    # a permissive gate lets the run finish
    m2 = _model()
    tr2 = Trainer(m2, {**CFG, "trainer": {**TRAIN, "wer_gate": {**gate, "every": 1, "max_delta": 5.0}}})
    tr2.fit(_data(), collate=Collate(m2.tokenizer))
    assert tr2.stopped is None and tr2.history[-1]["step"] == 2
