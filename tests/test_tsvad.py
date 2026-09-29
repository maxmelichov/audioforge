"""TS-VAD head (audioforge/heads/tsvad.py; research/IMPROVE_115M.md Part A)."""
import numpy as np
import torch

from audioforge.heads.tsvad import TSVADHead
from audioforge.model import build_head


def _head(**kw):
    torch.manual_seed(0)
    h = TSVADHead(32, emb_dim=8, hidden=16, **kw).eval()
    for p in h.parameters():  # non-trivial FiLM / null vector
        p.data.normal_(0, 0.3)
    return h


def test_shapes_and_range():
    h = _head()
    x = torch.randn(3, 20, 32)
    e = torch.randn(3, 8)
    p = h.decode(x, None, e)
    assert p.shape == (3, 20, 2) and float(p.min()) >= 0 and float(p.max()) <= 1


def test_streaming_equals_offline():
    h = _head()
    x = torch.randn(1, 23, 32)
    e = torch.randn(8)
    off = h.decode(x, None, e[None])
    st = h.init_stream(e.numpy())
    parts = [h.step(x[:, i: i + 2], st) for i in range(0, 23, 2)]
    assert torch.allclose(torch.cat(parts, 1), off, atol=1e-6)


def test_causal():
    h = _head()
    x = torch.randn(1, 30, 32)
    e = torch.randn(1, 8)
    a = h.decode(x, None, e)
    y = x.clone()
    y[:, 17:] = torch.randn(1, 13, 32)
    b = h.decode(y, None, e)
    assert torch.equal(a[:, :17], b[:, :17]) and not torch.equal(a[:, 17:], b[:, 17:])


def test_enrollment_matters_and_null_vector():
    h = _head()
    x = torch.randn(2, 10, 32)
    e = torch.randn(2, 8)
    with_e = h.decode(x, None, e)
    none = h.decode(x, None, None)
    has0 = h.decode(x, None, e, torch.tensor([False, True]))
    assert not torch.allclose(with_e, none)
    assert torch.allclose(has0[0], none[0]) and torch.allclose(has0[1], with_e[1])
    # the enrollment is used as a direction only (unit-normalised)
    assert torch.allclose(h.decode(x, None, 5 * e), with_e, atol=1e-6)


def test_loss_masks_padding_and_trains():
    h = TSVADHead(32, emb_dim=8, hidden=16)
    x = torch.randn(4, 12, 32)
    y = (torch.rand(4, 12, 2) > 0.5).float()
    batch = {"tsvad_targets": y, "tsvad_enroll": torch.randn(4, 8), "tsvad_has": torch.tensor([1, 1, 0, 1]).bool()}
    l_full = h.loss(x, torch.full((4,), 12), batch)
    l_pad = h.loss(x, torch.tensor([12, 12, 12, 6]), batch)
    assert l_full.item() > 0 and not torch.isclose(l_full, l_pad)
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    l0 = None
    for _ in range(50):
        loss = h.loss(x, torch.full((4,), 12), batch)
        l0 = l0 if l0 is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * l0


def test_build_head_registry():
    h = build_head({"type": "tsvad", "from_layers": [3], "weight": 0.0, "emb_dim": 192, "hidden": 64}, 512)
    assert isinstance(h, TSVADHead)
    assert sum(p.numel() for p in h.parameters()) < 1_000_000


def test_set_enrollment_keeps_state():
    h = _head()
    x = torch.randn(1, 8, 32)
    e1, e2 = np.random.randn(8).astype(np.float32), np.random.randn(8).astype(np.float32)
    st = h.init_stream(e1)
    h.step(x[:, :4], st)
    h0 = st["h"].clone()
    h.set_enrollment(st, e2)
    assert torch.equal(st["h"], h0)
    out = h.step(x[:, 4:], st)
    # reference: GRU state from e1 frames, then e2 conditioning
    st2 = h.init_stream(e1)
    h.step(x[:, :4], st2)
    st2["c"] = h.cond(torch.as_tensor(e2)[None], 1)
    assert torch.allclose(out, h.step(x[:, 4:], st2))
