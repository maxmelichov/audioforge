"""Pure-PyTorch RNNT and TDT (token-and-duration transducer) losses.

Both use the same trick: along a time row the recursion
    alpha[t,u] = logaddexp(a[u], alpha[t,u-1] + e[u-1])
is a linear recurrence in log space, solved in closed form with
``logcumsumexp``. So we loop over T only, vectorized over batch and labels.
That runs on CPU, CUDA and MPS without custom kernels.
"""
from __future__ import annotations

import torch

NEG = -1e30


def _row(a: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Solve alpha[u] = logaddexp(a[u], alpha[u-1] + e[u-1]) for all u. a (B,U+1), e (B,U)."""
    E = torch.cat([torch.zeros_like(a[:, :1]), torch.cumsum(e, dim=1)], dim=1)  # E[u] = sum_{m<u} e[m]
    return E + torch.logcumsumexp(a - E, dim=1)


def _gather_labels(logp_tok: torch.Tensor, targets: torch.Tensor, blank: int):
    """logp_tok (B,T,U+1,V) -> blank (B,T,U+1), emit (B,T,U)."""
    B, T, U1, _ = logp_tok.shape
    blank_lp = logp_tok[..., blank]
    idx = targets[:, None, :, None].expand(B, T, U1 - 1, 1)
    emit_lp = logp_tok[:, :, :-1].gather(-1, idx).squeeze(-1)
    return blank_lp, emit_lp


def rnnt_loss(logits: torch.Tensor, targets: torch.Tensor, logit_lengths: torch.Tensor,
              target_lengths: torch.Tensor, blank: int, reduction: str = "mean") -> torch.Tensor:
    """logits (B,T,U+1,V) raw joint outputs; targets (B,U) padded."""
    logp = logits.float().log_softmax(-1)
    blank_lp, emit_lp = _gather_labels(logp, targets.long(), blank)
    B, T, U1 = blank_lp.shape
    init = torch.full((B, U1), NEG, device=logits.device)
    init[:, 0] = 0.0
    rows = [_row(init, emit_lp[:, 0])]
    for t in range(1, T):
        rows.append(_row(rows[-1] + blank_lp[:, t - 1], emit_lp[:, t]))
    alpha = torch.stack(rows, 1)  # B,T,U+1
    b = torch.arange(B, device=logits.device)
    tl, ul = logit_lengths.long() - 1, target_lengths.long()
    ll = alpha[b, tl, ul] + blank_lp[b, tl, ul]
    return _reduce(-ll, target_lengths, reduction)


def tdt_loss(token_logits: torch.Tensor, duration_logits: torch.Tensor, targets: torch.Tensor,
             logit_lengths: torch.Tensor, target_lengths: torch.Tensor, blank: int,
             durations: list[int], sigma: float = 0.0, reduction: str = "mean") -> torch.Tensor:
    """Token-and-Duration Transducer loss (Xu et al., 2023), as in Parakeet-TDT.

    token_logits (B,T,U+1,V), duration_logits (B,T,U+1,len(durations)).
    Blank must advance time (d>=1); tokens may use d=0 (several tokens per frame).
    ``sigma`` is the logit under-normalization from the paper (0 disables).
    """
    logp = token_logits.float().log_softmax(-1)
    if sigma:
        logp = logp - sigma
    dlp = duration_logits.float().log_softmax(-1)
    blank_lp, emit_lp = _gather_labels(logp, targets.long(), blank)
    B, T, U1 = blank_lp.shape
    dev = token_logits.device
    di = {d: i for i, d in enumerate(durations)}
    assert 1 in di, "durations must include 1"
    e0 = emit_lp + dlp[:, :, :-1, di[0]] if 0 in di else torch.full_like(emit_lp, NEG)
    rows: list[torch.Tensor] = []
    for t in range(T):
        terms = []
        if t == 0:
            init = torch.full((B, U1), NEG, device=dev)
            init[:, 0] = 0.0
            terms.append(init)
        for d in durations:
            if d == 0 or t - d < 0:
                continue
            s, j = t - d, di[d]
            prev = rows[s]
            terms.append(prev + blank_lp[:, s] + dlp[:, s, :, j])  # blank, stay on u
            emit = prev[:, :-1] + emit_lp[:, s] + dlp[:, s, :-1, j]  # token, u -> u+1
            terms.append(torch.cat([torch.full_like(prev[:, :1], NEG), emit], 1))
        a = torch.logsumexp(torch.stack(terms), 0)
        rows.append(_row(a, e0[:, t]) if 0 in di else a)  # no d=0: nothing is emitted within a frame
    alpha = torch.stack(rows, 1)
    b = torch.arange(B, device=dev)
    ul = target_lengths.long()
    fin = []
    for d in durations:
        if d == 0:
            continue
        s = logit_lengths.long() - d
        ok = s >= 0
        sc = s.clamp(min=0)
        v = alpha[b, sc, ul] + blank_lp[b, sc, ul] + dlp[b, sc, ul, di[d]]
        fin.append(torch.where(ok, v, torch.full_like(v, NEG)))
    ll = torch.logsumexp(torch.stack(fin), 0)
    return _reduce(-ll, target_lengths, reduction)


def _reduce(nll, target_lengths, reduction):
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    # per-token mean = NeMo reduction "mean" (NeMo's RNNTLoss default "mean_batch" is nll.mean())
    return (nll / target_lengths.clamp(min=1)).mean()
