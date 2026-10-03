"""Edge cases for audioforge/heads/audio.py (CodecTokenHead upsampling, SortformerHead num_spks/PIL)."""
import itertools

import pytest
import torch
import torch.nn.functional as F

from audioforge.heads.audio import CodecTokenHead, SortformerHead, sort_by_arrival


@pytest.fixture(autouse=True, scope="module")
def _threads_1():
    """1 torch thread for this module only (restored after it, not at import time for the session)."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_codec_upsample_subframes_are_distinguishable():
    torch.manual_seed(0)
    h = CodecTokenHead(16, num_codebooks=1, codebook_size=8, upsample=2, hidden=16, n_layers=1, n_heads=2)
    enc, elen = torch.randn(1, 5, 16), torch.tensor([5])
    codes = torch.tensor([[0, 1] * 5])[..., None]
    batch = {"codes": codes, "codes_len": torch.tensor([10])}
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    for _ in range(300):
        l = h.loss(enc, elen, batch)
        opt.zero_grad()
        l.backward()
        opt.step()
    h.eval()
    lg, _ = h(enc, elen)
    assert float((lg[:, 0::2] - lg[:, 1::2]).abs().max()) > 1e-3
    assert float((h.decode(enc, elen) == codes).float().mean()) > 0.9


def test_codec_upsample1_state_dict_unchanged():
    h = CodecTokenHead(16, num_codebooks=2, codebook_size=8, hidden=16, n_layers=1, n_heads=2)
    assert not any(k.startswith("sub") for k in h.state_dict())


def _diar_batch(S_data=4, B=2, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = (torch.rand(B, T, S_data, generator=g) > 0.6).float()
    return torch.randn(B, T, 16, generator=g), torch.tensor([T, T - 3]), {"spk_targets": tgt}


def test_sortformer_num_spks_differs_from_label_columns():
    torch.manual_seed(0)
    enc, elen, batch = _diar_batch(S_data=4)
    for n in (2, 8):
        h = SortformerHead(16, num_spks=n, d_hidden=16, n_layers=1, n_heads=2, dropout=0.0)
        assert torch.isfinite(h.loss(enc, elen, batch))
    # num_spks=2 with pil off == sort loss against the 2 earliest-arriving speakers
    h = SortformerHead(16, num_spks=2, d_hidden=16, n_layers=1, n_heads=2, dropout=0.0, pil_weight=0.0).eval()
    logits = h(enc, elen)
    valid = (torch.arange(12)[None] < elen[:, None])[..., None].float()
    ref = (F.binary_cross_entropy_with_logits(logits, sort_by_arrival(batch["spk_targets"])[..., :2],
                                              reduction="none") * valid).sum() / (valid.sum() * 2)
    assert torch.allclose(h.loss(enc, elen, batch), ref)


def test_sortformer_pil_honoured_above_4_speakers():
    torch.manual_seed(0)
    enc, elen, batch = _diar_batch(S_data=5)
    h = SortformerHead(16, num_spks=5, d_hidden=16, n_layers=1, n_heads=2, dropout=0.0, pil_weight=1.0).eval()
    l_pil = h.loss(enc, elen, batch)
    h.pil_weight = 0.0
    assert not torch.allclose(l_pil, h.loss(enc, elen, batch))


def test_sortformer_pil_matches_bruteforce():
    torch.manual_seed(0)
    enc, elen, batch = _diar_batch(S_data=3)
    h = SortformerHead(16, num_spks=3, d_hidden=16, n_layers=1, n_heads=2, dropout=0.0, pil_weight=1.0).eval()
    logits, tgt = h(enc, elen), batch["spk_targets"]
    valid = (torch.arange(12)[None] < elen[:, None])[..., None].float()
    best = None
    for perm in itertools.permutations(range(3)):
        l = (F.binary_cross_entropy_with_logits(logits, tgt[..., list(perm)], reduction="none") * valid
             ).sum((1, 2)) / (valid.sum((1, 2)) * 3)
        best = l if best is None else torch.minimum(best, l)
    assert torch.allclose(h.loss(enc, elen, batch), best.mean(), atol=1e-6)
