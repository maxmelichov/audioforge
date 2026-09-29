import numpy as np
import torch
from uc_turn_head import UCTurnHead, load_head, quiet_targets, save_head, vap_targets


def test_quiet_targets():
    act = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    tg, ok = quiet_targets(act, None, [2, 4])
    # h=2: quiet over (t, t+2]
    assert tg[0, :, 0].tolist()[:8] == [1, 1, 0, 0, 1, 1, 1, 1]
    # h=4 at t=0: frames 1..4 include frame 4 active -> 0; at t=5: frames 6..9 quiet -> 1
    assert tg[0, 0, 1] == 0 and tg[0, 5, 1] == 1
    # validity: t=7 with h=4 needs frames 8..11, beyond T=10 and no positive -> invalid
    assert not ok[0, 7, 1] and ok[0, 5, 1]
    # a known positive inside the labelled part is valid even if the window runs past the end
    act2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)
    tg2, ok2 = quiet_targets(act2, None, [4])
    assert ok2[0, 7, 0] and tg2[0, 7, 0] == 0


def test_vap_targets_and_silent_mass():
    u = torch.zeros(1, 20)
    a = torch.zeros(1, 20)
    a[0, 3] = 1
    cls, ok = vap_targets(u, a, None, [2, 5, 8, 10])
    # at t=0 the agent is active in bin 2 (frames 3..5): agent bit index 4 + 1 -> 2^5
    assert int(cls[0, 0]) == 2 ** 5 and ok[0, 0] and not ok[0, 15]
    h = UCTurnHead(vap=True, hidden=16, layers=1)
    assert h.vap_user_silent[0] and h.vap_user_silent[2 ** 5] and not h.vap_user_silent[1]
    assert int(h.vap_user_silent.sum()) == 16


def test_streaming_equals_forward():
    torch.manual_seed(0)
    for energy in (True, False):
        h = UCTurnHead(hidden=32, layers=2, energy=energy, vap=True).eval()
        T = 37
        enc = torch.randn(1, T, 512)
        act = (torch.rand(1, T, 2) > 0.6).float()
        act[0, :5] = 0
        rms = torch.randn(1, T) - 3
        with torch.no_grad():
            full = h.probs(h(enc, act, rms)[0])
        st = h.init_state()
        parts = {k: [] for k in full}
        for s, e in ((0, 2), (2, 3), (3, 11), (11, 12), (12, 30), (30, 37)):
            p, st = h.step(enc[:, s:e], act[0, s:e].numpy(), rms[0, s:e].numpy(), st)
            for k in parts:
                parts[k].append(p[k])
        for k in full:
            np.testing.assert_allclose(torch.cat(parts[k]).numpy(), full[k].numpy(), atol=1e-5)
        assert st.frames == T


def test_loss_and_roundtrip(tmp_path):
    torch.manual_seed(1)
    h = UCTurnHead(hidden=16, layers=1, vap=True)
    enc = torch.randn(2, 40, 512)
    act = (torch.rand(2, 40, 2) > 0.5).float()
    out, _ = h(enc, act, torch.randn(2, 40))
    loss, parts = h.loss(out, act[..., 0], act[..., 1], torch.tensor([40, 30]))
    assert torch.isfinite(loss) and set(parts) == {"user_bins", "user_quiet", "agent_bins", "vap"}
    loss.backward()
    save_head(h, tmp_path / "h.pt", {"x": 1})
    h2 = load_head(tmp_path / "h.pt")
    h.eval()
    with torch.no_grad():
        a = h.probs(h(enc[:1], act[:1])[0])
        b = h2.probs(h2(enc[:1], act[:1])[0])
    for k in a:
        np.testing.assert_allclose(a[k].numpy(), b[k].numpy(), atol=1e-6)
    assert h2.meta == {"x": 1}
