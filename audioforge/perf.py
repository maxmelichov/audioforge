"""CPU inference fast paths for the served system. Every option keeps the model's math; the
ones marked *exact* give bit-identical outputs on this machine's BLAS (checked by tests/test_perf.py and by
a serve benchmark comparison on the fixed clip set), the others agree to float rounding.

``apply(engine, opts)`` installs a set of options on an ``audioforge.serve.Engine`` (``--perf`` in serve.py);
``DEFAULT`` is the set that is on by default because it is exact.

* ``linear_t`` (*exact* for >= 2 rows): store every encoder ``nn.Linear`` weight transposed (``W.t().contiguous().t()``,
  same values, same shape, column-major storage). PyTorch then calls the BLAS as ``x @ Wt`` ("NN") instead of
  ``x @ W^T`` ("NT"); on Apple Accelerate the NT path is 3-4x slower for the 2-4-row GEMMs of a 160 ms streaming
  step (a 512->2048 layer: 134 -> 30 us at 2 threads) and 5-20 % slower for the diarizer's ~380-row GEMMs. Results
  are bit-identical for inputs of >= 2 rows (the BLAS GEMM computes each output row the same way in both layouts).
  A 1-row input takes a GEMV, and only the row-major GEMV reproduces the original 1-row rounding (~1e-7 relative
  otherwise; note the original itself gives a frame different bits alone than next to another frame). In the
  encoders only the trailing frame of a stream (a 1-frame final chunk) is a 1-row call. Heads (RNNT joint / predictor, turn head, VAD) are left alone: they are tiny and run single rows.
* ``pos_cache`` (*exact*): ``RelPositionMultiHeadAttention`` recomputed ``linear_pos(pe(L))`` - a (2L-1) x d GEMM,
  143 rows at the served [70, 1] cache, i.e. more rows than the rest of the layer's streaming step - on every call.
  It depends only on L and the weights, so it is cached per module (a small LRU of L values; see relpos.py).
* ``share_subsample`` (*exact*): the speaker-conditioned turn pass (``--turn-input diar``) re-ran the encoder's
  convolutional subsampling on the same mel chunks as the ASR pass (its speaker kernels start at layer 0, so the
  conformer layers cannot be shared, but the subsampling output can). The ASR pass keeps its subsampled frames per
  mel chunk and the conditioned pass reuses them when its stream state is at the same position (checked).
* ``joint_cache`` (*exact*): the per-frame RNNT decode recomputed ``joint.pred(g)`` (the prediction-network
  projection of the joint) on every encoder frame although ``g`` only changes when a token is emitted; the last
  projection is kept with the tensor it came from (``model.StreamingSession.cache_joint_pred``).
* ``fast_subsample`` (float rounding, max |diff| ~4e-7 on the subsampled frames): the two depthwise 3x3 stride-2
  ``Conv2d`` of the subsampling (3.0-3.3 ms per call on 2 threads; PyTorch's CPU depthwise conv again) as nine
  strided multiply-adds (0.7 ms).
"""
from __future__ import annotations

import weakref
from collections import OrderedDict

import torch
import torch.nn as nn

DEFAULT = ("linear_t", "pos_cache", "share_subsample", "joint_cache")
EXACT = ("linear_t", "pos_cache", "share_subsample", "joint_cache")
ALL = ("linear_t", "pos_cache", "share_subsample", "joint_cache", "fast_subsample")


# --------------------------------------------------------------------------- linear_t
def linear_t(module: nn.Module, skip=()) -> int:
    """Re-lay out every nn.Linear weight under ``module`` (not under the modules in ``skip``) as column-major storage.
    Returns the number of layers changed. Inference only (the Parameter is replaced, requires_grad kept)."""
    skipped = {id(m) for s in skip for m in s.modules()}
    n = 0
    for mod in module.modules():
        if id(mod) in skipped or type(mod) is not nn.Linear:
            continue
        w = mod.weight
        if w.dim() != 2 or w.stride(0) == 1:
            continue
        mod.weight = nn.Parameter(w.detach().t().contiguous().t(), requires_grad=w.requires_grad)
        n += 1
    return n


# --------------------------------------------------------------------------- pos_cache
def pos_cache(module: nn.Module, size: int = 2) -> int:
    """Turn on the linear_pos(pe) cache of every RelPositionMultiHeadAttention under ``module``."""
    from .modules.relpos import RelPositionMultiHeadAttention
    n = 0
    for mod in module.modules():
        if isinstance(mod, RelPositionMultiHeadAttention):
            mod.pos_cache_size = size
            mod._pos_cache = OrderedDict()
            n += 1
    return n


# --------------------------------------------------------------------------- fast_subsample
def _dw_shifts(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Depthwise 3x3 stride-2 conv (no padding; the caller pads) as nine strided multiply-adds."""
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    To, Fo = (x.shape[2] - kh) // sh + 1, (x.shape[3] - kw) // sw + 1
    w = conv.weight[:, 0]
    acc = None
    for i in range(kh):
        for j in range(kw):
            s = x[:, :, i: i + sh * (To - 1) + 1: sh, j: j + sw * (Fo - 1) + 1: sw] * w[:, i, j][None, :, None, None]
            acc = s if acc is None else acc + s
    return acc + conv.bias[None, :, None, None] if conv.bias is not None else acc


class _DWShift(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return _dw_shifts(self.conv, x)


def fast_subsample(encoder) -> int:
    """Replace the depthwise Conv2d of a DWStridingSubsampling (encoder.pre_encode) by ``_dw_shifts``."""
    pe = getattr(encoder, "pre_encode", None)
    n = 0
    for conv in getattr(pe, "convs", []):
        if isinstance(conv, nn.Sequential) and isinstance(conv[0], nn.Conv2d) and conv[0].groups == conv[0].in_channels \
                and conv[0].groups > 1 and conv[0].padding == (0, 0) and conv[0].dilation == (1, 1):
            conv[0] = _DWShift(conv[0])
            n += 1
    return n


# --------------------------------------------------------------------------- share_subsample
class _SubsampleMemo:
    """Subsampled frames of the ASR pass per mel chunk, for the conditioned pass over the same chunk.

    Keyed by the mel chunk tensor (weakly: an entry dies with its chunk) and checked against the stream position
    (mel frames received before the chunk, first output frame), so a state that is not at the same position simply
    recomputes."""

    def __init__(self):
        self.d: dict[int, tuple] = {}
        self.hits = self.misses = 0

    def put(self, mel, key, x):
        i = id(mel)
        self.d[i] = (weakref.ref(mel, lambda _r, i=i: self.d.pop(i, None)), key, x)

    def get(self, mel, key):
        e = self.d.get(id(mel))
        if e is not None and e[0]() is mel and e[1] == key:
            self.hits += 1
            return e[2]
        self.misses += 1
        return None


def share_subsample(encoder) -> bool:
    """Give a NeMo-aligned causal encoder the subsampling memo that ``FastConformerEncoder._stream_step_aligned``
    consults (per instance; the encoder code has the hook, this only turns it on)."""
    if not getattr(getattr(encoder, "pre_encode", None), "nemo_causal", False):
        return False
    encoder._sub_memo = _SubsampleMemo()
    return True


# --------------------------------------------------------------------------- engine
def parse(spec) -> tuple[str, ...]:
    """'default' | 'none' | 'all' | 'exact' | comma list (a leading '-' removes an option from the set so far)."""
    if spec is None or spec == "":
        return DEFAULT
    if isinstance(spec, dict):
        spec = ",".join(k for k, v in spec.items() if v)
    out: list[str] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("default", "none", "all", "exact"):
            out = list({"default": DEFAULT, "none": (), "all": ALL, "exact": EXACT}[tok])
        elif tok.startswith("-"):
            out = [o for o in out if o != tok[1:]]
        elif tok in ALL:
            out.append(tok) if tok not in out else None
        else:
            raise ValueError(f"unknown perf option {tok!r} (one of {ALL})")
    return tuple(out)


def apply(asr_model, diar_model, opts) -> dict:
    """Install ``opts`` (an iterable of ALL names) on the ASR and diarizer models (``diar_model`` may be None:
    ``--diar-off``); returns counts per option."""
    opts = tuple(opts)
    info = {}
    if "linear_t" in opts:
        info["linear_t"] = linear_t(asr_model.encoder) + (linear_t(diar_model) if diar_model is not None else 0)
    if "pos_cache" in opts:
        info["pos_cache"] = pos_cache(asr_model.encoder) + (pos_cache(diar_model) if diar_model is not None else 0)
    if "share_subsample" in opts:
        info["share_subsample"] = bool(share_subsample(asr_model.encoder))
    if "fast_subsample" in opts:
        info["fast_subsample"] = fast_subsample(asr_model.encoder) + (
            fast_subsample(diar_model.encoder) if diar_model is not None else 0)
    from .model import StreamingSession
    StreamingSession.cache_joint_pred = info["joint_cache"] = "joint_cache" in opts
    return info
