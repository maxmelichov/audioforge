import functools
import math

import torch

from audioforge.losses.transducer import rnnt_loss, tdt_loss


def brute_rnnt(lp, y):
    T, U1, _ = lp.shape
    U, blank = U1 - 1, lp.shape[-1] - 1
    p = lp.exp().double()

    @functools.lru_cache(None)
    def g(t, u):
        b = p[t, u, blank] * (1.0 if (t == T - 1 and u == U) else (g(t + 1, u) if t < T - 1 else 0.0))
        e = p[t, u, y[u]] * g(t, u + 1) if u < U else 0.0
        return b + e
    return -math.log(g(0, 0))


def brute_tdt(lp, dlp, y, durs):
    T, U1, _ = lp.shape
    U, blank = U1 - 1, lp.shape[-1] - 1
    p, pd = lp.exp().double(), dlp.exp().double()

    @functools.lru_cache(None)
    def f(t, u):
        tot = 0.0
        for j, d in enumerate(durs):
            if d >= 1 and t + d <= T:
                nxt = 1.0 if (t + d == T and u == U) else (f(t + d, u) if t + d < T else 0.0)
                tot += p[t, u, blank] * pd[t, u, j] * nxt
            if u < U and t + d < T:
                tot += p[t, u, y[u]] * pd[t, u, j] * f(t + d, u + 1)
        return tot
    return -math.log(f(0, 0))


def test_rnnt_matches_brute_force():
    torch.manual_seed(0)
    T, U, V = 5, 3, 4
    logits = torch.randn(2, T, U + 1, V + 1)
    y = torch.randint(0, V, (2, U))
    tl, ul = torch.tensor([5, 4]), torch.tensor([3, 2])
    got = rnnt_loss(logits, y, tl, ul, blank=V, reduction="none")
    for b in range(2):
        lp = logits[b, : tl[b], : ul[b] + 1].log_softmax(-1)
        assert abs(got[b].item() - brute_rnnt(lp, y[b].tolist())) < 1e-4


def test_tdt_matches_brute_force():
    torch.manual_seed(1)
    T, U, V, durs = 6, 3, 4, [0, 1, 2, 3]
    tok = torch.randn(2, T, U + 1, V + 1)
    dur = torch.randn(2, T, U + 1, len(durs))
    y = torch.randint(0, V, (2, U))
    tl, ul = torch.tensor([6, 5]), torch.tensor([3, 2])
    got = tdt_loss(tok, dur, y, tl, ul, blank=V, durations=durs, reduction="none")
    for b in range(2):
        lp = tok[b, : tl[b], : ul[b] + 1].log_softmax(-1)
        dlp = dur[b, : tl[b], : ul[b] + 1].log_softmax(-1)
        assert abs(got[b].item() - brute_tdt(lp, dlp, y[b].tolist(), durs)) < 1e-4


def test_losses_backprop():
    tok = torch.randn(1, 4, 3, 5, requires_grad=True)
    dur = torch.randn(1, 4, 3, 3, requires_grad=True)
    y = torch.tensor([[1, 2]])
    l = tdt_loss(tok, dur, y, torch.tensor([4]), torch.tensor([2]), 4, [0, 1, 2])
    l.backward()
    assert torch.isfinite(tok.grad).all() and tok.grad.abs().sum() > 0


def test_tdt_without_zero_duration_matches_brute_force():
    # no d=0 -> every emission advances time; the within-frame recursion must be skipped, not
    # evaluated with -1e30 "log-probs" (float32 cancellation there returned 2.8 instead of 11.0)
    torch.manual_seed(1)
    T, U, V = 6, 3, 4
    for durs in ([1, 2, 3], [1]):
        tok = torch.randn(2, T, U + 1, V + 1)
        dur = torch.randn(2, T, U + 1, len(durs))
        y = torch.randint(0, V, (2, U))
        tl, ul = torch.tensor([6, 5]), torch.tensor([3, 2])
        got = tdt_loss(tok, dur, y, tl, ul, blank=V, durations=durs, reduction="none")
        for b in range(2):
            lp = tok[b, : tl[b], : ul[b] + 1].log_softmax(-1)
            dlp = dur[b, : tl[b], : ul[b] + 1].log_softmax(-1)
            assert abs(got[b].item() - brute_tdt(lp, dlp, y[b].tolist(), durs)) < 1e-4, durs


def test_tdt_sigma_under_normalization():
    # sigma is subtracted from every token log-prob (blank and non-blank), as in NeMo's TDT loss
    torch.manual_seed(2)
    T, U, V, durs = 5, 2, 3, [0, 1, 2]
    tok, dur = torch.randn(1, T, U + 1, V + 1), torch.randn(1, T, U + 1, len(durs))
    y, tl, ul = torch.randint(0, V, (1, U)), torch.tensor([T]), torch.tensor([U])
    got = tdt_loss(tok, dur, y, tl, ul, blank=V, durations=durs, sigma=0.05, reduction="none")
    lp = tok[0].log_softmax(-1) - 0.05
    assert abs(got.item() - brute_tdt(lp, dur[0].log_softmax(-1), y[0].tolist(), durs)) < 1e-4


def test_fused_batch_transducer_loss_matches_full_joint():
    # fused_batch_size: joint + loss per sub-batch under activation checkpointing (memory-safe path)
    from audioforge.heads.asr import RNNTHead
    for durs in (None, [0, 1, 2]):
        torch.manual_seed(0)
        full = RNNTHead(16, 6, pred_hidden=8, joint_hidden=8, durations=durs)
        fused = RNNTHead(16, 6, pred_hidden=8, joint_hidden=8, durations=durs, fused_batch_size=2)
        fused.load_state_dict(full.state_dict())
        enc = torch.randn(5, 9, 16, requires_grad=True)
        batch = {"text": torch.randint(0, 6, (5, 4)), "text_len": torch.tensor([4, 1, 3, 2, 4])}
        el = torch.tensor([9, 5, 7, 3, 8])
        grads = []
        for h in (full.eval(), fused.eval()):
            enc.grad = None
            l = h.loss(enc, el, batch)
            l.backward()
            grads.append((l.detach(), enc.grad.clone(), h.joint.out[-1].weight.grad.clone()))
        for a, b in zip(*grads):
            assert torch.allclose(a, b, atol=1e-5), durs


def test_tdt_greedy_duration_semantics():
    from audioforge.heads.asr import RNNTHead

    def head(tok, dur, durs=(0, 1, 2)):
        h = RNNTHead(4, 5, pred_hidden=4, joint_hidden=4, durations=list(durs), max_symbols=3).eval()
        lin = h.joint.out[-1]
        with torch.no_grad():  # force a constant (token, duration) prediction
            lin.weight.zero_(); lin.bias.fill_(-10); lin.bias[tok] = 10; lin.bias[6 + durs.index(dur)] = 10
        return h
    f, L = torch.randn(1, 7, 4), torch.tensor([7])
    assert head(5, 0).decode(f, L) == [[]]  # blank with d=0 still advances one frame (terminates)
    assert head(3, 0).decode(f, L) == [[3] * 21]  # token d=0 stays on the frame, capped by max_symbols
    assert head(3, 2).decode(f, L) == [[3] * 4]  # token d=2 at frames 0, 2, 4, 6
