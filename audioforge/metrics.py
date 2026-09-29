"""WER/CER, frame-level DER, speaker-verification EER, the shared percentile helpers and process memory."""
from __future__ import annotations

import itertools
import resource
import sys

import numpy as np
import torch


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    e = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(r.split()) for r in refs))


def cer(refs: list[str], hyps: list[str]) -> float:
    e = sum(edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(r) for r in refs))


def frame_der(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Permutation-free frame DER. pred/ref (T,S) binary. (miss + FA + confusion) / speech."""
    S = max(ref.shape[1], pred.shape[1])  # pad to a common speaker count: extra columns are FA / miss
    ref, pred = (torch.nn.functional.pad(x, (0, S - x.shape[1])) for x in (ref, pred))
    best = None
    for perm in itertools.permutations(range(S)):
        p = pred[:, list(perm)]
        n_ref, n_hyp = ref.sum(1), p.sum(1)
        correct = torch.minimum(p, ref).sum(1)
        err = (torch.maximum(n_ref, n_hyp) - correct).sum()
        best = err if best is None else torch.minimum(best, err)
    return float(best / ref.sum().clamp(min=1))


def eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Equal error rate from trial scores and 0/1 labels."""
    order = scores.argsort(descending=True)
    l = labels[order].float()
    tp = l.cumsum(0)
    fp = (1 - l).cumsum(0)
    fnr = 1 - tp / l.sum().clamp(min=1)
    fpr = fp / (1 - l).sum().clamp(min=1)
    i = torch.argmin((fnr - fpr).abs())
    return float((fnr[i] + fpr[i]) / 2)


def pct(xs, q: float, nd: int = 1, empty=None):
    """``q``-th percentile of ``xs`` rounded to ``nd`` places; ``empty`` when there is nothing to summarise."""
    return round(float(np.percentile(xs, q)), nd) if len(xs) else empty


def pct_dict(xs, qs=(10, 25, 50, 75, 90), nd: int = 3) -> dict:
    """``{"p10": .., "p25": .., ...}`` of ``xs`` (empty dict when ``xs`` is empty)."""
    return {f"p{q}": pct(xs, q, nd) for q in qs} if len(xs) else {}


def rss_mb() -> float:
    """Current resident set size (psutil; falls back to the peak)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        return peak_rss_mb()


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024, 1)  # bytes on macOS, KiB on Linux
