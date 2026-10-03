"""Regression tests from the turn-ablation diagnosis (research/archive/TURN_ABLATION.md).

A2  the primary column chosen at inference (``primary_column``) is the column the Sortformer sort loss trains
    the primary into (``sort_by_arrival``), for ties, silent columns, batch padding and label cropping.
A   the diar path the turn head is evaluated with re-runs the Sortformer head on growing prefixes, so the head
    must be trained on prefixes too (``SortformerHead(prefix_prob=...)``); the turn head's labels can come from
    an oracle key while its conditioning activity is noisy (``spk_act_oracle``).
B   the text-branch training alignment must run through the encoding the ASR head actually reads (the
    speaker-conditioned one when the ASR head is conditioned), and the train-time text augmentations
    (delay / substitution / the ASR head's own greedy decode) keep the text state causal and well-formed.
"""
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import audioforge.heads.turn as turn_mod
from audioforge.conversation import conversation_dataset
from audioforge.data import Collate
from audioforge.heads.asr import RNNTHead
from audioforge.heads.audio import SortformerHead, _match_len, sort_by_arrival
from audioforge.heads.turn import TurnHead, primary_column, token_counts
from audioforge.model import SpeechModel
from audioforge.tokenizer import train_tokenizer
from audioforge.train import run_recipe

RECIPE = Path(__file__).parent.parent / "research" / "recipes" / "speaker_aware_turn.yaml"


# --------------------------------------------------------------------------- A2: primary column consistency
def _primary_in_sorted(t: torch.Tensor, T: int | None = None) -> torch.Tensor:
    """The column that holds the primary (column 0) after the sort loss's reordering (with the loss's crop)."""
    s = sort_by_arrival(_match_len(t, T) if T is not None else t)
    ref = _match_len(t, T)[..., 0] if T is not None else t[..., 0]
    hits = (s == ref[..., None]).all(1)  # (B,S) columns identical to the primary track
    return hits


def test_primary_column_matches_sort_loss_order_random():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        B, T, S = 4, 30, 4
        t = torch.zeros(B, T, S)
        for b in range(B):
            for s in range(S):
                if s == 0 or torch.rand(1, generator=g) < 0.7:  # the primary always speaks; others may be silent
                    st = int(torch.randint(0, 20, (1,), generator=g))
                    t[b, st: st + int(torch.randint(1, 10, (1,), generator=g)), s] = 1
        t[0, :, 1] = t[0, :, 0]  # a tie: another speaker starts on the same frame (and is identical)
        pc = primary_column(t)
        onset = (t[..., 0] > 0.5).float().argmax(1)
        for crop in (None, T - 3, T + 5):  # the loss crops/pads labels to the encoder length
            if crop is not None and int(onset.max()) >= crop:
                continue
            hits = _primary_in_sorted(t, crop)
            assert all(bool(hits[b, pc[b]]) for b in range(B)), (crop, pc, hits)
        # ties resolve like the loss (stable sort: the lower original column first)
        assert int(pc[0]) == int(primary_column(t[:1])[0])


def test_primary_column_on_collated_conversations():
    convs = conversation_dataset(24, seed=5)
    b = Collate(None)(convs)
    pc = primary_column(b["spk_targets"])
    for crop in (b["spk_targets"].shape[1], b["spk_targets"].shape[1] - 1):
        srt = sort_by_arrival(_match_len(b["spk_targets"].float(), crop))
        for j, c in enumerate(convs):
            n = min(len(c["primary_act"]), crop)
            assert np.array_equal(srt[j, :n, int(pc[j])].numpy(), c["primary_act"][:n])
    # conversations where another speaker talks first exist, so the primary is not always column 0
    assert (pc > 0).any() and (pc == 0).any()


# --------------------------------------------------------------------------- A: prefix-trained Sortformer
def test_sortformer_prefix_loss():
    torch.manual_seed(0)
    head = SortformerHead(16, num_spks=4, d_hidden=16, n_layers=1, dropout=0.0, prefix_prob=1.0, prefix_min=4).train()
    B, T = 2, 20
    enc, elen = torch.randn(B, T, 16), torch.tensor([20, 15])
    tgt = torch.zeros(B, T, 4)
    tgt[:, 2:8, 1] = 1  # arrives first
    tgt[:, 12:18, 0] = 1  # arrives second: silent in prefixes shorter than 12 frames
    random.seed(3)
    l = head.loss(enc, elen, {"spk_targets": tgt})
    l.backward()
    assert torch.isfinite(l) and head.proj.weight.grad is not None
    random.seed(3)
    L = head._prefix_len(T)
    full = head._loss_on(enc, elen, tgt)
    pre = head._loss_on(enc[:, :L], elen.clamp(max=L), tgt[:, :L])
    assert torch.allclose(l, 0.5 * (full + pre), atol=1e-6)
    assert head.prefix_min <= L <= T
    head.eval()  # no prefix term at eval time
    assert torch.allclose(head.loss(enc, elen, {"spk_targets": tgt}), head._loss_on(enc, elen, tgt))


def test_turn_labels_from_oracle_key():
    torch.manual_seed(0)
    head = TurnHead(16, mode="none", hidden=8, dropout=0.0).eval()
    enc, elen = torch.randn(2, 12, 16), torch.tensor([12, 12])
    clean = torch.zeros(2, 12)
    clean[:, 2:6] = 1
    noisy = clean.clone()
    noisy[:, 9] = 1  # a diarization false alarm after the true end would move the label
    a = head.loss(enc, elen, {"spk_act": noisy, "spk_act_oracle": clean})
    b = head.loss(enc, elen, {"spk_act": clean})
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- B: text alignment input
def _tiny_model(turn_mode="none"):
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg["encoder"].update(n_layers=2, d_model=32, n_heads=2, subsampling_channels=8, speaker_kernel_layers=[0])
    cfg["heads"]["turn"].update(mode=turn_mode, condition_on_speaker=turn_mode == "kernel", use_text=True,
                                hidden=16, text_dim=8)
    cfg["heads"]["tdt"].update(pred_hidden=16, joint_hidden=16)
    cfg["heads"]["diar"].update(d_hidden=16, n_layers=1)
    convs = conversation_dataset(4, seed=9)
    tok = train_tokenizer("char", [c["text"] for c in convs])
    torch.manual_seed(0)
    model = SpeechModel(cfg, tok).eval()
    with torch.no_grad():  # speaker kernels start as identity: perturb them so conditioning changes the encoding
        for p in model.encoder.speaker_kernels.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return model, convs, tok


@pytest.mark.parametrize("turn_mode", ["none", "kernel"])
def test_text_alignment_uses_the_asr_heads_encoding(turn_mode, monkeypatch):
    """The TDT head is speaker-conditioned; the forced alignment for the turn head's text state must run on
    that conditioned encoding (as decoded_text_state does at inference), even when the turn head itself is not
    conditioned (turn_text ablation arm: aligning on the plain encoding gave 0 % greedy success)."""
    model, convs, tok = _tiny_model(turn_mode)
    seen = []
    orig = turn_mod.greedy_align

    def spy(asr, enc, *a, **k):
        seen.append(enc.detach().clone())
        return orig(asr, enc, *a, **k)

    monkeypatch.setattr(turn_mod, "greedy_align", spy)
    b = Collate(tok)(convs)
    with torch.no_grad():
        model(b)
        e_plain, _ = model.encode(b["audio"], b["audio_len"])
        e_cond, _ = model.encode(b["audio"], b["audio_len"], spk_act=_match_len(b["spk_act"].float(), e_plain.shape[1]))
    assert len(seen) == 1
    assert torch.allclose(seen[0], e_cond, atol=1e-5) and not torch.allclose(e_plain, e_cond, atol=1e-3)


# --------------------------------------------------------------------------- B: text augmentation
def _text_head(**kw):
    torch.manual_seed(0)
    asr = RNNTHead(32, vocab_size=12, pred_hidden=16, joint_hidden=16).eval()
    head = TurnHead(32, mode="none", hidden=24, history=4, dropout=0.0, use_text=True, k_tokens=4, text_dim=8,
                    align="uniform", **kw)
    head.bind_asr(asr)
    return head, asr


def test_text_delay_and_noise_augmentation():
    head, asr = _text_head(text_delay=3, text_noise=0.5)
    B, T = 3, 20
    enc, elen = torch.randn(B, T, 32), torch.tensor([20, 14, 9])
    y, yl = torch.randint(0, 12, (B, 6)), torch.tensor([6, 3, 1])
    act = torch.zeros(B, T)
    act[0, 2:12], act[1, 5:8], act[2, 1:3] = 1, 1, 1
    base = head.align(enc, elen, y, yl, act)
    head.train()
    torch.manual_seed(1)
    y2, emit = head.augment_text(y, yl, base, elen)
    for b in range(B):
        u = int(yl[b])
        d = emit[b, :u] - base[b, :u]
        assert (d >= 0).all() and (d <= 3).all() and len(set(d.tolist())) == 1  # one delay per item
        assert int(emit[b, :u].max()) < int(elen[b])  # clamped inside the valid frames
        assert (emit[b, 1:u] >= emit[b, : u - 1]).all()
    assert (y2 != y).any() and (y2 < 12).all()  # substitutions stay in the vocabulary (never blank)
    n = token_counts(emit, yl, T)
    assert (n[:, 1:] >= n[:, :-1]).all()
    head.eval()  # no augmentation at eval time
    y3, e3 = head.augment_text(y, yl, base, elen)
    assert torch.equal(y3, y) and torch.equal(e3, base)


def test_decoded_text_training_path():
    head, asr = _text_head(decoded_prob=1.0)
    head.train()
    enc, elen = torch.randn(2, 12, 32), torch.tensor([12, 10])
    act = torch.zeros(2, 12)
    act[:, 2:8] = 1
    b = {"spk_act": act, "text": torch.randint(0, 12, (2, 4)), "text_len": torch.tensor([4, 2])}
    l = head.loss(enc, elen, b)
    l.backward()
    assert torch.isfinite(l) and head.text_proj[0].weight.grad is not None
    assert all(p.grad is None for p in asr.parameters())
    assert head.align_counts.get("decoded", 0) == 2


@pytest.mark.parametrize("decoded_prob", [0.0, 1.0])
def test_recipe_smoke_diagnostics_reported(tmp_path, decoded_prob):
    """3 recipe steps report the turn diagnostics. Seed-independent (the trainer may seed Python's random): with
    decoded_prob 0 every step aligns the reference text (turn_align_greedy_frac), with 1 every step trains on the
    ASR's own decode (turn_decoded_frac); a 0.5 draw over 3 steps picked one path only, depending on the seed."""
    ov = ["trainer.max_steps=3", "trainer.batch_size=4", "trainer.device=cpu", "data.synthetic.n_train=8",
          "data.synthetic.n_val=4", "encoder.n_layers=2", "encoder.d_model=64", "encoder.subsampling_channels=16",
          "heads.turn.mode=kernel", "heads.turn.condition_on_speaker=true", "heads.turn.use_text=true",
          "heads.turn.text_delay=2", "heads.turn.text_noise=0.1", f"heads.turn.decoded_prob={decoded_prob}",
          "heads.diar.prefix_prob=0.5"]
    _, metrics = run_recipe(str(RECIPE), ov, out=str(tmp_path / "m.afm"))
    assert "eot_turn_diar_act_miss" in metrics and 0.0 <= metrics["eot_turn_diar_act_miss"] <= 1.0
    if decoded_prob == 0.0:
        assert "turn_align_greedy_frac" in metrics and 0.0 <= metrics["turn_align_greedy_frac"] <= 1.0
        assert "turn_decoded_frac" not in metrics
    else:
        assert metrics.get("turn_decoded_frac") == 1.0 and "turn_align_greedy_frac" not in metrics
