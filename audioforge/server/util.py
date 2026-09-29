"""Small helpers shared by the engine, the sessions and the streams: percentiles, process memory and a bounded
per-frame ring."""
from __future__ import annotations

from collections import deque

from ..metrics import pct, peak_rss_mb, rss_mb  # noqa: F401 - re-exported for the server


def _pct(xs, q):
    return pct(xs, q, nd=2, empty=0.0)


class _Ring:
    """Append-only sequence that keeps only the last ``keep`` items: ``len()`` counts every item ever appended and
    ``r[v]`` indexes absolutely (an evicted index returns the oldest kept item; a future one raises IndexError).
    Bounds the per-frame state of a session of any length (rows, primaries, VAD)."""

    def __init__(self, keep: int = 2048):
        self.d: deque = deque(maxlen=keep)
        self.n = 0

    def append(self, x):
        self.d.append(x)
        self.n += 1

    def __len__(self):
        return self.n

    def __bool__(self):
        return self.n > 0

    def __getitem__(self, v):
        if v < 0:
            return self.d[v]
        i = v - (self.n - len(self.d))
        if i >= len(self.d):
            raise IndexError(v)
        return self.d[max(0, i)]
