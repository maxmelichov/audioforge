"""Train-time activity conditioning (research/archive/TURN_ABLATION.md, "What you need to change elsewhere" 1-2).

The speaker-conditioned encoder pass (TDT / CTC / kernel turn head) was trained only on the clean oracle primary
activity, then evaluated on diarization output (P50 240 ms -> 2240 ms, miss 7 % -> 48 %). ``conditioning:`` feeds it,
per item, the diar head's own detached primary column (probability ``p_diar``) or a noise-augmented oracle
(frame flips, boundary jitter, dropped short segments); the clean track stays in ``spk_act_oracle`` for the labels.
"""
import copy
import random
from pathlib import Path

import pytest
import torch
import yaml

from audioforge.conversation import conversation_dataset
from audioforge.data import Collate
from audioforge.heads.turn import primary_column
from audioforge.model import GradScale, SpeechModel, augment_activity
from audioforge.tokenizer import train_tokenizer
from audioforge.train import run_recipe

RECIPE = Path(__file__).parent.parent / "research" / "recipes" / "speaker_aware_turn.yaml"


def _cfg(conditioning=None, turn_mode="kernel", use_text=True):
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg.pop("conditioning", None)
    if conditioning is not None:
        cfg["conditioning"] = conditioning
    cfg["encoder"].update(n_layers=2, d_model=32, n_heads=2, subsampling_channels=8, speaker_kernel_layers=[0])
    cfg["heads"]["turn"].update(mode=turn_mode, condition_on_speaker=turn_mode == "kernel", use_text=use_text,
                                hidden=16, text_dim=8, decoded_prob=0.0, text_delay=0, text_noise=0.0)
    cfg["heads"]["tdt"].update(pred_hidden=16, joint_hidden=16)
    cfg["heads"]["diar"].update(d_hidden=16, n_layers=1, prefix_prob=0.0)
    return cfg


def _model(conditioning=None, **kw):
    convs = conversation_dataset(4, seed=9)
    tok = train_tokenizer("char", [c["text"] for c in convs])
    torch.manual_seed(0)
    model = SpeechModel(_cfg(conditioning, **kw), tok)
    with torch.no_grad():  # speaker kernels start as identity: perturb them so conditioning changes the encoding
        for p in model.encoder.speaker_kernels.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return model, Collate(tok)(convs)


def _forward_v2(m, batch, att):
    """SpeechModel.forward as it was before ``conditioning`` (reference for the bit-identity test)."""
    feats, flen = m.features(batch["audio"], batch["audio_len"])
    enc, elen, hidden = m.encoder(feats, flen, att, return_hidden=True)
    enc_spk, total = None, 0.0
    for name, head in m.heads.items():
        hc = m.head_cfg[name]
        needed = head.key if hasattr(head, "key") else None
        if needed and needed not in batch:
            continue
        e = m.head_input(name, enc, hidden)
        if hc.get("condition_on_speaker"):
            if "spk_act" not in batch:
                continue
            if enc_spk is None:
                enc_spk, _ = m.encoder(feats, flen, att, spk_act=batch["spk_act"][:, : enc.shape[1]])
            e = enc_spk
        if "grad_scale" in hc:
            e = GradScale.apply(e, float(hc["grad_scale"]))
        total = total + hc.get("weight", 1.0) * head.loss(e, elen, batch)
    return total


def _spy_encoder(model, seen):
    orig = model.encoder.forward

    def spy(feats, flen, att=None, spk_act=None, **k):
        if spk_act is not None:
            seen.append(spk_act.detach().clone())
        return orig(feats, flen, att, spk_act=spk_act, **k)

    model.encoder.forward = spy


def _spy_turn_batch(model, seen):
    orig = model.heads["turn"].loss

    def spy(enc, elen, batch):
        seen.append(batch)
        return orig(enc, elen, batch)

    model.heads["turn"].loss = spy


# --------------------------------------------------------------------------- defaults
@pytest.mark.parametrize("turn_mode,use_text", [("kernel", True), ("kernel", False), ("none", False)])
def test_default_off_is_bit_identical(turn_mode, use_text):
    model, b = _model(None, turn_mode=turn_mode, use_text=use_text)
    assert not model.cond_on
    model.train()
    att = model.encoder.att_context_size
    torch.manual_seed(5), random.seed(5)
    ref = _forward_v2(model, copy.copy(b), att)
    torch.manual_seed(5), random.seed(5)
    out = model(b, att_context_size=att)
    assert torch.equal(out["loss"], ref)
    # an explicit all-zero block is the same as no block
    m2, _ = _model({"p_diar": 0.0, "flip": 0.0, "jitter": 0})
    assert not m2.cond_on


def test_caller_batch_untouched_and_eval_unchanged():
    model, b = _model({"p_diar": 0.5, "flip": 0.3, "jitter": 2, "drop": 0.5})
    keys, act = set(b), b["spk_act"].clone()
    model.train()
    model(b)
    assert set(b) == keys and torch.equal(b["spk_act"], act)
    # eval / inference: the conditioning never applies
    model.eval()
    seen = []
    _spy_encoder(model, seen)
    with torch.no_grad():
        model(b)
    assert torch.equal(seen[0], b["spk_act"][:, : seen[0].shape[1]])


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        _model({"p_diarization": 0.5})


# --------------------------------------------------------------------------- diar source
def test_p_diar_one_uses_the_diar_heads_primary_column():
    model, b = _model({"p_diar": 1.0})
    model.train()
    diar = model.heads["diar"]
    T = {}

    def fake(enc, enc_len):  # a known logit pattern, different per column
        B, Tn, _ = enc.shape
        T.setdefault("T", Tn), T.setdefault("elen", enc_len)
        z = torch.arange(Tn, dtype=torch.float)[None, :, None] / Tn - 0.5 + torch.arange(4.0)[None, None]
        return z.expand(B, Tn, 4) + 0 * enc.sum()

    diar.forward = fake
    b["spk_targets"] = b["spk_targets"].clone()
    b["spk_targets"][0, 0, 0], b["spk_targets"][0, 0, 1] = 0, 1  # item 0: another speaker arrives first
    seen, batches = [], []
    _spy_encoder(model, seen)
    _spy_turn_batch(model, batches)
    out = model(b)
    assert torch.isfinite(out["loss"])
    col = primary_column(b["spk_targets"])  # arrival rank of the primary, the rule the labels / eval use
    assert (col > 0).any() and (col == 0).any()
    Tn, elen = T["T"], T["elen"]
    z = fake(torch.zeros(len(col), Tn, 1), None).sigmoid()
    expect = z.gather(2, col[:, None, None].expand(-1, Tn, 1))[..., 0]
    expect = expect * (torch.arange(Tn)[None] < elen[:, None])  # padding frames are silent
    act = seen[0]
    assert act.shape == expect.shape and torch.allclose(act, expect)
    tb = batches[0]
    assert torch.equal(tb["spk_act"], act) and torch.equal(tb["spk_act_oracle"], b["spk_act"])
    assert not act.requires_grad


def test_p_diar_real_head_detached_and_trains():
    model, b = _model({"p_diar": 1.0})
    model.train()
    out = model(b)
    out["loss"].backward()
    assert model.heads["diar"].proj.weight.grad is not None  # the diar loss still trains it


# --------------------------------------------------------------------------- noise augmentation
def test_noise_changes_spk_act_but_oracle_stays_clean():
    model, b = _model({"flip": 0.1, "jitter": 2, "drop": 1.0, "drop_max": 6})
    model.train()
    clean = b["spk_act"].clone()
    batches = []
    _spy_turn_batch(model, batches)
    torch.manual_seed(1)
    model(b)
    tb = batches[0]
    assert torch.equal(tb["spk_act_oracle"], clean)
    T = tb["spk_act"].shape[1]
    assert not torch.equal(tb["spk_act"], clean[:, :T].float())
    assert set(tb["spk_act"].unique().tolist()) <= {0.0, 1.0}


def test_augment_activity_components():
    torch.manual_seed(0)
    a = torch.zeros(3, 40)
    a[:, 5:15] = 1  # long segment
    a[:, 20:23] = 1  # short segment
    L = torch.tensor([40, 40, 30])
    assert torch.equal(augment_activity(a, L), a)  # all off -> identity
    d = augment_activity(a, L, drop=1.0, drop_max=4)
    assert d[:, 20:23].sum() == 0 and torch.equal(d[:, 5:15], a[:, 5:15])  # only short segments dropped
    for _ in range(20):
        j = augment_activity(a, L, jitter=2)
        for b in range(3):
            nz = j[b].nonzero()[:, 0]
            assert 3 <= int(nz[0]) <= 7 and 20 <= int(nz[-1]) + 1 <= 25
    f = augment_activity(a, L, flip=0.5)
    assert (f != a).float().mean() > 0.3 and f[2, 30:].sum() == 0  # flips stay inside the valid frames
    assert torch.equal(a, a.clone())


# --------------------------------------------------------------------------- _asr_enc
def test_asr_enc_is_the_detached_conditioned_encoding():
    model, b = _model(None, turn_mode="none", use_text=True)  # turn_text arm: the turn head reuses _asr_enc
    model.train()
    batches, seen_enc = [], []
    _spy_turn_batch(model, batches)
    orig = model.encoder.forward

    def spy(feats, flen, att=None, spk_act=None, **k):
        r = orig(feats, flen, att, spk_act=spk_act, **k)
        if spk_act is not None:
            seen_enc.append(r[0])
        return r

    model.encoder.forward = spy
    model(b)
    e = batches[0]["_asr_enc"]
    assert len(seen_enc) == 1  # no second, no-grad re-encode in TurnHead.asr_view
    assert e.shape == seen_enc[0].shape and e.shape[:2] == (len(b["audio_len"]), seen_enc[0].shape[1])
    assert torch.equal(e, seen_enc[0].detach()) and not e.requires_grad


# --------------------------------------------------------------------------- recipe smoke
@pytest.mark.parametrize("use_text", [False, True], ids=["speaker", "speaker+text"])
def test_recipe_smoke_conditioning(use_text, tmp_path):
    cfg = yaml.safe_load(RECIPE.read_text())
    assert cfg["conditioning"] == {"p_diar": 0.5, "flip": 0.05, "jitter": 2, **{k: cfg["conditioning"][k]
                                   for k in cfg["conditioning"] if k not in ("p_diar", "flip", "jitter")}}
    assert cfg["heads"]["diar"]["from_layers"] == "all" and cfg["heads"]["diar"]["prefix_prob"] == 0.5
    ov = ["trainer.max_steps=3", "trainer.batch_size=4", "trainer.device=cpu", "data.synthetic.n_train=8",
          "data.synthetic.n_val=4", "encoder.n_layers=2", "encoder.d_model=64", "encoder.subsampling_channels=16",
          "heads.turn.mode=kernel", "heads.turn.condition_on_speaker=true", f"heads.turn.use_text={str(use_text).lower()}"]
    model, metrics = run_recipe(str(RECIPE), ov, out=str(tmp_path / "m.afm"))
    assert model.cond_on and "eot_turn_p50_ms@5fc" in metrics and "eot_turn_diar_act_miss" in metrics
