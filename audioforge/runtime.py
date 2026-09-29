"""On-device streaming runtime: the StreamingSession pipeline, rebuilt for deployment.

    samples -> incremental log-mel -> ChunkEncoder (fixed shapes, caches in/out) -> heads -> greedy decode

What changes vs ``StreamingSession`` (``audioforge/model.py``):

* **Fixed shapes.** ``ChunkEncoder`` is a fixed-shape re-implementation of
  ``FastConformerEncoder.stream_step`` that reuses the encoder's own weights. Every step sees
  ``(1, F, lookback + chunk)`` mel frames and fixed-size key/value/conv caches; "not yet valid"
  cache slots, the first chunk (no mel history) and a partial final chunk are handled by masks
  instead of by changing shapes. This is what makes ``torch.jit.trace``/``torch.compile``/ONNX/Core ML
  possible (the standard Riva / sherpa-onnx pattern: caches are explicit inputs and outputs).
* **KV ring buffer.** Keys are stored already RoPE-rotated, so their order does not matter to
  attention: frame ``a`` lives in slot ``a % (left + chunk)``. The torch backends write the new chunk
  into the ring in place (no per-step cache copies); the export graph does the same with
  gather/where and returns the caches.
* **Preallocated buffers.** Signal buffer, mel input window, caches and the decoder token tensor are
  allocated once; the per-step Python work is a handful of slice copies.
* **Cheaper, identical math.** RoPE sin/cos are computed once per step (not 2x per layer), the
  joint's prediction projection is cached per emitted token (not recomputed per frame), and the
  pointwise convs run as ``nn.Linear`` (same weights; lets dynamic int8 quantization cover them).
* **All heads per chunk.** Primary transducer/CTC text, a secondary CTC greedy, frame heads
  (VAD/EOU) and the layer-mix features for ``from_layers: all`` heads (speaker embedding on demand).
* **Backends.** ``eager`` | ``trace`` (``torch.jit.trace``) | ``compile`` (``torch.compile``) |
  ``onnx`` (onnxruntime CPU, per-chunk encoder exported with caches as I/O); optional
  ``quantize="int8"`` (dynamic, Linear + LSTM, qnnpack on ARM) and ``dtype`` bf16/fp16.

fp32 ``eager``/``trace`` produce the same text as ``StreamingSession`` (tests/test_runtime.py).
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads.asr import CTCHead, RNNTHead
from .heads.audio import FrameHead, SpeakerHead
from .model import SpeechModel


def _ceil_div2(x: int) -> int:
    return (x + 1) // 2


# --------------------------------------------------------------------------- fixed-shape encoder step
class _Layer(nn.Module):
    """One FastConformer layer, streaming-only, pointwise convs as Linear (shared weights)."""

    def __init__(self, layer):
        super().__init__()
        self.ff1, self.ff2 = layer.ff1, layer.ff2
        self.norm_att, self.norm_conv, self.norm_out = layer.norm_att, layer.norm_conv, layer.norm_out
        self.qkv, self.proj = layer.att.qkv, layer.att.proj
        self.h, self.dk = layer.att.h, layer.att.dk
        c = layer.conv
        self.kernel, self.norm_type, self.cnorm, self.dw = c.kernel, c.norm_type, c.norm, c.dw
        self.pw1 = nn.Linear(c.pw1.in_channels, c.pw1.out_channels)
        self.pw1.weight = nn.Parameter(c.pw1.weight.detach()[..., 0])  # views: no extra memory
        self.pw1.bias = c.pw1.bias
        self.pw2 = nn.Linear(c.pw2.in_channels, c.pw2.out_channels)
        self.pw2.weight = nn.Parameter(c.pw2.weight.detach()[..., 0])
        self.pw2.bias = c.pw2.bias


def _rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ChunkEncoder(nn.Module):
    """Functional fixed-shape streaming step with explicit caches.

    forward(mel, inv, offset, nvalid, kc, vc, cc) ->
        (enc (1,cs,D), kc', vc', cc', *head_outputs)     functional (export)
        (enc (1,cs,D), *head_outputs)                    inplace=True: kc/vc/cc updated in place

    mel     (1, F, lb + cs*8)  mel history (``lb`` frames, the leading ``inv`` of which are invalid) + chunk
    inv     int64 ()           invalid history frames (lb at the first step, 0 in steady state)
    offset  int64 ()           absolute encoder frame index of this chunk (RoPE position, cache fill)
    nvalid  int64 ()           valid encoder frames in this chunk (cs, fewer for the final partial chunk)
    kc, vc  (N, H, left+cs, dk) ring buffer of rotated keys / values (frame a in slot a % (left+cs))
    cc      (N, D, k-1)        depthwise-conv input cache per layer
    """

    def __init__(self, model: SpeechModel, att_context_size, head_names: list[str], primary: str,
                 inplace: bool = False):
        super().__init__()
        self.inplace = inplace  # True: caches updated in place (eager/trace/compile); False: returned (export)
        enc = model.encoder
        assert enc.causal, "streaming requires a causal encoder"
        L, R = att_context_size
        assert L >= 0 and R >= 0, "the fixed-shape runtime needs a finite [L, R] context"
        self.cs = R + 1
        self.left = (L // self.cs) * self.cs
        self.slots = self.left + self.cs  # KV ring size
        self.fast_dw = True  # torch backends: depthwise conv via unfold (see forward)
        self.lb = enc.pre_encode.lookback
        self.sub = enc.subsampling_factor
        assert self.lb % self.sub == 0
        self.chunk_mel = self.cs * self.sub
        self.pre_convs, self.pre_out = enc.pre_encode.convs, enc.pre_encode.out
        # static per-stage frame positions (no shape-dependent ops in the traced graph: Core ML friendly)
        T, Fr = self.lb + self.chunk_mel, model.preprocessor.n_mels
        for s in range(3):
            self.register_buffer(f"tpos{s}", torch.arange(T), persistent=False)
            T, Fr = _ceil_div2(T), (Fr + 2 - 3) // 2 + 1
        self.pre_T, self.pre_CF = T, enc.pre_encode.convs[0].out_channels * Fr
        self.layers = nn.ModuleList([_Layer(l) for l in enc.layers])
        l0 = self.layers[0]
        self.n_layers, self.d, self.h, self.dk, self.kernel = len(self.layers), enc.d_model, l0.h, l0.dk, l0.kernel
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, self.dk, 2, dtype=torch.float32) / self.dk)),
                             persistent=False)
        self.primary = primary
        self.head_names = list(head_names)
        # only the per-frame parts of each head live here (the transducer's prediction net / joint
        # output and the speaker pooling run outside), so quantization/casting touches just these
        self.kinds, mods = [], {}
        for k in head_names:
            h = model.heads[k]
            if isinstance(h, RNNTHead):
                self.kinds.append("joint_enc")
                mods[k] = h.joint.enc  # joint encoder projection; greedy loop runs outside
            elif isinstance(h, CTCHead):
                self.kinds.append("ctc")
                mods[k] = h
            elif isinstance(h, FrameHead):
                self.kinds.append("sigmoid" if h.num_classes == 1 else "softmax")
                mods[k] = h
            elif isinstance(h, SpeakerHead):
                self.kinds.append("frames")  # layer-mixed frames; pooled on demand by the runtime
                mods[k] = nn.Identity()
            else:
                raise TypeError(f"head {k!r} ({type(h).__name__}) is not streamable")
        self.heads = nn.ModuleDict(mods)
        mix = {}
        for k in head_names:
            w = model.layer_weights(k)  # from_layers: all -> constant softmax weights; taps -> one-hot / sub-mix
            if w is not None:
                mix[k] = w.to(next(model.parameters()).device)
        self.mix_names = list(mix)
        for k, w in mix.items():
            self.register_buffer(f"mix_{k}", w.clone(), persistent=False)

    def output_names(self) -> list[str]:
        caches = [] if self.inplace else ["kc_out", "vc_out", "cc_out"]
        return ["enc"] + caches + [f"head_{k}" for k in self.head_names]

    def init_caches(self, dtype=torch.float32):
        kc = torch.zeros(self.n_layers, self.h, self.slots, self.dk, dtype=dtype)
        cc = torch.zeros(self.n_layers, self.d, self.kernel - 1, dtype=dtype)
        return kc, kc.clone(), cc

    def _pre_encode(self, mel, inv):
        x = mel.transpose(1, 2).unsqueeze(1)  # 1,1,T,F
        for s, conv in enumerate(self.pre_convs):
            bad = getattr(self, f"tpos{s}") < torch.div(inv, 2 ** s, rounding_mode="floor")
            x = x.masked_fill(bad[None, None, :, None], 0.0)
            x = F.silu(conv(F.pad(x, (1, 1, 2, 0))))
        x = self.pre_out(x.permute(0, 2, 1, 3).reshape(1, self.pre_T, self.pre_CF))
        return x[:, self.lb // self.sub:]

    def forward(self, mel, inv, offset, nvalid, kc, vc, cc):
        x = self._pre_encode(mel, inv)  # 1,cs,D
        cs, left, S, dev = self.cs, self.left, self.slots, mel.device
        # RoPE tables once per step (shared by all layers, q and k); same formula as fastconformer.rotary
        pos = offset.to(torch.float32) + torch.arange(cs, device=dev, dtype=torch.float32)
        ang = pos[:, None] * self.inv_freq[None]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        # KV ring buffer of S = left + cs slots: absolute frame a lives in slot a % S. Keys are stored
        # already rotated, so their order is irrelevant to attention; after writing this chunk the ring holds
        # exactly frames [offset - left, offset + cs). Valid keys: a >= 0 and a < offset + nvalid.
        slot = torch.arange(S, device=dev)
        start = offset - left
        a_abs = start + torch.remainder(slot - start, S)
        kmask = ((a_abs >= 0) & (a_abs < offset + nvalid))[None, None, None]
        j = torch.remainder(slot - offset, S)  # position of slot within this chunk (new iff j < cs)
        if self.inplace:
            new_slots = torch.remainder(offset + torch.arange(cs, device=dev), S)
        else:
            is_new = (j < cs)[None, :, None]
            jj = torch.clamp(j, max=cs - 1)
        kcs, vcs, ccs, hidden = [], [], [], []
        for i, l in enumerate(self.layers):
            x = x + 0.5 * l.ff1(x)
            q, k, v = l.qkv(l.norm_att(x)).view(1, cs, 3, l.h, l.dk).permute(2, 0, 3, 1, 4)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            if self.inplace:  # no per-step cache copies: write cs new slots in place
                kc[i].index_copy_(1, new_slots, k[0])
                vc[i].index_copy_(1, new_slots, v[0])
                ki, vi = kc[i:i + 1], vc[i:i + 1]
            else:  # functional (ONNX / Core ML): gather + where, caches returned as outputs
                ki = torch.where(is_new, k[0][:, jj], kc[i])
                vi = torch.where(is_new, v[0][:, jj], vc[i])
                kcs.append(ki)
                vcs.append(vi)
                ki, vi = ki[None], vi[None]
            a = F.scaled_dot_product_attention(q, ki, vi, attn_mask=kmask)
            x = x + l.proj(a.transpose(1, 2).reshape(1, cs, -1))
            c = F.glu(l.pw1(l.norm_conv(x)), dim=-1).transpose(1, 2)  # 1,D,cs
            c = torch.cat([cc[i:i + 1], c], 2)
            if self.inplace:
                cc[i].copy_(c[0, :, -(l.kernel - 1):])
            else:
                ccs.append(c[0, :, -(l.kernel - 1):])
            if self.inplace and self.fast_dw:  # depthwise conv as unfold * w, sum: ~40x faster than Conv1d(groups=D) on a
                # 10-frame input on CPU (PyTorch's depthwise Conv1d CPU path costs ~1.2 ms/call here)
                c = (c.unfold(2, l.kernel, 1) * l.dw.weight[:, 0][None, :, None]).sum(-1) + l.dw.bias[None, :, None]
            else:  # export graphs keep the Conv op
                c = l.dw(c)
            c = l.cnorm(c) if l.norm_type == "batch" else l.cnorm(c.transpose(1, 2)).transpose(1, 2)
            x = x + l.pw2(F.silu(c).transpose(1, 2))
            x = l.norm_out(x + 0.5 * l.ff2(x))
            if self.mix_names:
                hidden.append(x)
        outs = [x] if self.inplace else [x, torch.stack(kcs), torch.stack(vcs), torch.stack(ccs)]
        for k, kind in zip(self.head_names, self.kinds):
            e = x
            if k in self.mix_names:
                w = getattr(self, f"mix_{k}")
                e = sum(w[j] * hj for j, hj in enumerate(hidden))
            z = self.heads[k](e)
            outs.append(z.sigmoid() if kind == "sigmoid" else z.softmax(-1) if kind == "softmax" else z)
        return tuple(outs)


def streamable_heads(model: SpeechModel) -> list[str]:
    ok = (RNNTHead, CTCHead, FrameHead, SpeakerHead)
    return [k for k, h in model.heads.items()
            if isinstance(h, ok) and not model.head_cfg[k].get("condition_on_speaker")]


# --------------------------------------------------------------------------- ONNX export of the step
def export_chunk_onnx(model: SpeechModel, path: str | Path, att_context_size=None, opset: int = 17,
                      dynamo: bool = False):
    """Export the per-chunk encoder step (+ head projections) with caches as ONNX inputs/outputs."""
    model = model.cpu().eval()
    att = list(att_context_size or model.encoder.att_context_size)
    heads = streamable_heads(model)
    primary = model.primary if model.primary in heads else heads[0]
    ce = ChunkEncoder(model, att, heads, primary).eval()
    args = _example_inputs(ce, model.preprocessor.n_mels)
    torch.onnx.export(ce, args, str(path), input_names=["mel", "inv", "offset", "nvalid", "kc", "vc", "cc"],
                      output_names=ce.output_names(), opset_version=opset, dynamo=dynamo)
    return path


def _example_inputs(ce: ChunkEncoder, n_mels: int, dtype=torch.float32):
    kc, vc, cc = ce.init_caches(dtype)
    return (torch.randn(1, n_mels, ce.lb + ce.chunk_mel, dtype=dtype), torch.tensor(ce.lb), torch.tensor(0),
            torch.tensor(ce.cs), kc, vc, cc)


# --------------------------------------------------------------------------- runtime
class StreamingRuntime:
    """Deployment wrapper for cache-aware streaming (see module docstring).

    rt = StreamingRuntime(model, backend="trace", quantize="int8"); rt.warmup()
    for chunk in mic: rt.feed(chunk)        # -> current text
    rt.feed([], final=True); rt.text, rt.frame_events, rt.ctc_text, rt.speaker_embedding()
    """

    BACKENDS = ("eager", "trace", "compile", "onnx", "coreml")

    def __init__(self, model: SpeechModel, head: str | None = None, att_context_size=None,
                 backend: str = "eager", quantize: str | None = None, dtype: torch.dtype = torch.float32,
                 threads: int | None = None, onnx_path: str | Path | None = None, coreml_path=None,
                 coreml_units: str = "CPU_ONLY", decode_secondary: bool = True, copy_model: bool = True,
                 free_encoder: bool = False, quantize_decoder: bool = False):
        assert backend in self.BACKENDS, backend
        pp = model.preprocessor
        assert pp.normalize in (None, "none", "NA", "fixed"), "streaming needs frame-local mel normalization"
        if threads:
            torch.set_num_threads(threads)
        self.threads = threads
        model = model.cpu().eval()
        if (quantize or dtype != torch.float32 or free_encoder) and copy_model:
            model = copy.deepcopy(model)  # never mutate the caller's model (copy_model=False: in place, less RAM)
        self.m, self.backend, self.quantize, self.dtype = model, backend, quantize, dtype
        self.att = list(att_context_size or model.encoder.att_context_size)
        heads = streamable_heads(model)
        self.head_name = head or model.primary
        assert self.head_name in heads, f"primary head {self.head_name!r} is not streamable"
        self.decode_secondary = decode_secondary
        self.ce = ChunkEncoder(model, self.att, heads, self.head_name,
                               inplace=backend in ("eager", "trace", "compile")).eval()
        self.head = model.heads[self.head_name]
        self.head_names = heads
        if quantize == "int8":
            torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines \
                else torch.backends.quantized.engine
            self.ce = torch.ao.quantization.quantize_dynamic(self.ce, {nn.Linear}, dtype=torch.qint8, inplace=True)
            if quantize_decoder and isinstance(self.head, RNNTHead):  # prediction net + joint (decoder side)
                torch.ao.quantization.quantize_dynamic(self.head.pred, {nn.LSTM}, dtype=torch.qint8, inplace=True)
                torch.ao.quantization.quantize_dynamic(self.head.joint, {nn.Linear}, dtype=torch.qint8, inplace=True)
        elif quantize:
            raise ValueError(f"unknown quantize {quantize!r}")
        if dtype != torch.float32:
            self.ce = self.ce.to(dtype)
        self.cs, self.left, self.lb, self.chunk_mel = self.ce.cs, self.ce.left, self.ce.lb, self.ce.chunk_mel
        self.frame_sec = model.frame_sec
        self.chunk_ms = self.cs * self.frame_sec * 1000
        self.hop, self.half, self.n_fft = pp.hop, pp.n_fft // 2, pp.n_fft
        self.preemph = pp.preemph
        self._fn = self._build_backend(onnx_path, coreml_path, coreml_units)
        if free_encoder and backend in ("onnx", "coreml"):  # encoder weights live in the ORT/Core ML model
            self.ce.layers = nn.ModuleList()
            self.ce.pre_convs = nn.ModuleList()
            model.encoder.layers = nn.ModuleList()
            model.encoder.pre_encode.convs = nn.ModuleList()
            model.encoder.pre_encode.out = nn.Identity()
        self.reset()

    # ----------------------------------------------------------------- backends
    def _build_backend(self, onnx_path, coreml_path, coreml_units):
        n_mels = self.m.preprocessor.n_mels
        if self.backend == "eager":
            return self.ce
        if self.backend == "trace":
            with torch.no_grad():
                return torch.jit.freeze(torch.jit.trace(self.ce, _example_inputs(self.ce, n_mels, self.dtype),
                                                        check_trace=False).eval())
        if self.backend == "compile":
            return torch.compile(self.ce, dynamic=False, fullgraph=True)
        if self.backend == "onnx":
            import onnxruntime as ort
            assert self.dtype == torch.float32 and not self.quantize, "use an int8 ONNX file for ORT int8"
            if onnx_path is None or not Path(onnx_path).exists():
                raise FileNotFoundError("export first: export_chunk_onnx(model, path, att)")
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads or 0
            so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            in_names = [i.name for i in sess.get_inputs()]
            out_names = [o.name for o in sess.get_outputs()]

            def run(*args):
                feed = {n: a.numpy() for n, a in zip(in_names, args) if n in in_names}
                return tuple(torch.from_numpy(o) for o in sess.run(out_names, feed))
            self._ort = sess
            return run
        if self.backend == "coreml":
            import coremltools as ct
            units = getattr(ct.ComputeUnit, coreml_units)
            mlm = ct.models.MLModel(str(coreml_path), compute_units=units)
            names = ["mel", "inv", "offset", "nvalid", "kc", "vc", "cc"]
            outs = self.ce.output_names()

            def run(*args):
                feed = {n: (a.reshape(1).numpy().astype(np.int32) if a.dtype == torch.int64 else a.numpy())
                        for n, a in zip(names, args)}  # Core ML: no rank-0 inputs
                r = mlm.predict(feed)
                return tuple(torch.from_numpy(np.asarray(r[o])) for o in outs)
            return run
        raise ValueError(self.backend)

    # ----------------------------------------------------------------- state
    def reset(self):
        F_ = self.m.preprocessor.n_mels
        self.kc, self.vc, self.cc = self.ce.init_caches(self.dtype)
        self.mel_in = torch.zeros(1, F_, self.lb + self.chunk_mel, dtype=self.dtype)
        self.hist_valid = 0
        self.offset = 0
        cap = 1 << 16
        self.sig = torch.zeros(cap)  # pre-emphasized samples; sig[0] is absolute sample sig_start
        self.sig_n = 0
        self.sig_start = 0
        self.last = 0.0
        self.mel_done = 0
        self.seg = torch.zeros((self.chunk_mel - 1) * self.hop + self.n_fft)
        self.tokens: list[int] = []
        self.ctc_tokens: list[int] = []
        self._ctc_prev = self._ctc_prev2 = -1
        self.pred = None
        self.gp = None
        self.skip = 0
        self._tok = torch.zeros(1, 1, dtype=torch.long)
        self.frame_events: dict[str, list] = {k: [] for k in self.head_names
                                              if isinstance(self.m.heads[k], FrameHead)}
        self.spk_frames: dict[str, list] = {k: [] for k in self.head_names
                                            if isinstance(self.m.heads[k], SpeakerHead)}
        self.steps = 0
        self.first_token_step = None
        return self

    @torch.no_grad()
    def warmup(self, n_steps: int = 3):
        """Run a few silent chunks (JIT/compile/allocator warm-up), then reset. Idempotent."""
        for _ in range(n_steps):
            self.feed(np.zeros(self.chunk_mel * self.hop, np.float32))
        return self.reset()

    # ----------------------------------------------------------------- audio -> mel
    def _append(self, x: torch.Tensor):
        n = x.numel()
        if self.sig_n + n > self.sig.numel():
            big = torch.zeros(max(2 * self.sig.numel(), self.sig_n + n))
            big[: self.sig_n] = self.sig[: self.sig_n]
            self.sig = big
        pe = self.preemph
        if pe:
            y = self.sig[self.sig_n: self.sig_n + n]
            y[0] = x[0] - pe * self.last  # same float32 ops as StreamingSession
            if n > 1:
                y[1:] = x[1:] - pe * x[:-1]
        else:
            self.sig[self.sig_n: self.sig_n + n] = x
        self.last = float(x[-1])
        self.sig_n += n

    def _mel(self, a: int, b: int) -> torch.Tensor:
        pp = self.m.preprocessor
        s0, s1 = a * self.hop - self.half, (b - 1) * self.hop + self.half
        seg = self.seg[: s1 - s0] if (b - a) == self.chunk_mel else torch.zeros(s1 - s0)
        seg.zero_()
        lo, hi = max(s0, self.sig_start), min(s1, self.sig_start + self.sig_n)
        if hi > lo:
            seg[lo - s0: hi - s0] = self.sig[lo - self.sig_start: hi - self.sig_start]
        spec = torch.stft(seg[None], pp.n_fft, pp.hop, pp.win, pp.window, center=False, return_complex=True)
        return pp.fixed_norm(torch.log(pp.fb @ (spec.real ** 2 + spec.imag ** 2) + 2 ** -24))  # 1,F,b-a

    def _trim(self):
        keep = self.mel_done * self.hop - self.half - self.hop
        if keep > self.sig_start:
            d = min(keep - self.sig_start, self.sig_n)
            self.sig[: self.sig_n - d] = self.sig[d: self.sig_n].clone()
            self.sig_n -= d
            self.sig_start += d

    # ----------------------------------------------------------------- streaming
    @torch.no_grad()
    def feed(self, samples, final: bool = False) -> str:
        x = torch.as_tensor(np.asarray(samples, dtype=np.float32))
        if x.numel():
            self._append(x)
        total = self.sig_start + self.sig_n
        ready = (total - self.half) // self.hop + 1 if not final else total // self.hop + 1
        while ready - self.mel_done >= self.chunk_mel:
            self._step(self._mel(self.mel_done, self.mel_done + self.chunk_mel), self.chunk_mel)
            self.mel_done += self.chunk_mel
            self._trim()
        if final and ready > self.mel_done:
            n = ready - self.mel_done
            self._step(self._mel(self.mel_done, ready), n)
            self.mel_done = ready
        return self.text

    def _step(self, mel: torch.Tensor, n: int):
        """mel (1,F,n) new frames, n <= chunk_mel."""
        lb, cm = self.lb, self.chunk_mel
        self.mel_in[..., lb: lb + n] = mel
        if n < cm:
            self.mel_in[..., lb + n:] = 0.0
        hv = self.hist_valid
        nvalid = _ceil_div2(_ceil_div2(_ceil_div2(hv + n))) - hv // self.ce.sub
        out = self._fn(self.mel_in, torch.tensor(lb - hv), torch.tensor(self.offset), torch.tensor(nvalid),
                       self.kc, self.vc, self.cc)
        if self.ce.inplace:
            enc, heads_out = out[0], out[1:]
        else:
            enc, self.kc, self.vc, self.cc = out[:4]
            heads_out = out[4:]
        self.offset += nvalid
        # slide the mel history window: last lb frames of [history | chunk]
        self.mel_in[..., :lb] = self.mel_in[..., cm:].clone()
        self.hist_valid = min(lb, hv + n)
        self.steps += 1
        ntok = len(self.tokens)
        for name, o in zip(self.head_names, heads_out):
            o = o[:, :nvalid].float()
            h = self.m.heads[name]
            if name == self.head_name:
                self._decode_primary(o[0])
            elif isinstance(h, FrameHead):
                self.frame_events[name].extend(o[0].tolist())
            elif isinstance(h, SpeakerHead):
                self.spk_frames[name].append(o[0])
            elif isinstance(h, CTCHead) and self.decode_secondary:
                self._ctc_greedy(o[0], self.ctc_tokens)
        if self.first_token_step is None and len(self.tokens) > ntok:
            self.first_token_step = self.steps
        return enc

    def _ctc_greedy(self, lp, toks):
        blank = lp.shape[-1] - 1  # CTCHead: blank = vocab_size = last class
        for p in lp.argmax(-1).tolist():
            if p != blank and p != self._ctc_prev2:
                toks.append(p)
            self._ctc_prev2 = p

    def _decode_primary(self, f):
        h = self.head
        if isinstance(h, CTCHead):
            for p in f.argmax(-1).tolist():
                if p != h.blank and p != self._ctc_prev:
                    self.tokens.append(p)
                self._ctc_prev = p
            return
        # f = joint.enc(enc) (T, J). Same greedy as StreamingSession._decode, joint.pred(g) cached.
        V1 = h.vocab_size + 1
        if self.pred is None:
            self._tok[0, 0] = h.blank
            self.pred = h.pred(self._tok, None)
            self.gp = h.joint.pred(self.pred[0])
        g, st = self.pred
        gp = self.gp
        t, emitted, T = self.skip, 0, f.shape[0]
        durs = h.durations
        while t < T:
            z = h.joint.out(f[t][None, None] + gp)[0, 0]
            k = int(z[:V1].argmax())
            d = durs[int(z[V1:].argmax())] if durs is not None else (1 if k == h.blank else 0)
            if k == h.blank:
                d = max(d, 1)
            else:
                self.tokens.append(k)
                self._tok[0, 0] = k
                g, st = h.pred(self._tok, st)
                gp = h.joint.pred(g)
                emitted += 1
            if d == 0 and emitted >= h.max_symbols:
                d = 1
            if d:
                emitted = 0
            t += d
        self.skip = t - T
        self.pred, self.gp = (g, st), gp

    # ----------------------------------------------------------------- outputs
    @property
    def text(self) -> str:
        return self.m.tokenizer.decode(self.tokens) if self.m.tokenizer is not None else ""

    @property
    def ctc_text(self) -> str:
        return self.m.tokenizer.decode(self.ctc_tokens) if self.m.tokenizer is not None else ""

    @torch.no_grad()
    def speaker_embedding(self, name: str | None = None, last_frames: int | None = None) -> torch.Tensor | None:
        """Pool the streamed layer-mix frames (e.g. over the current turn) -> (1, emb_dim)."""
        if not self.spk_frames:
            return None
        name = name or next(iter(self.spk_frames))
        fr = self.spk_frames[name]
        if not fr:
            return None
        e = torch.cat(fr, 0)
        if last_frames:
            e = e[-last_frames:]
        return self.m.heads[name].embed(e[None], torch.tensor([e.shape[0]]))


# --------------------------------------------------------------------------- helpers for benchmarks
def time_stream(rt: StreamingRuntime, audio: np.ndarray, push_ms: float | None = None,
                spk_every_s: float = 1.0, spk_window_s: float = 5.0) -> dict:
    """Stream ``audio`` through ``rt`` in chunk-sized pushes; per-push wall and process-CPU times (ms).

    CPU time (``time.process_time``: all threads of this process) excludes time spent waiting for a core,
    so on a loaded machine it is the better estimate of the cost on one dedicated core (1-thread runs).
    """
    rt.reset()
    push = int(round((push_ms or rt.chunk_ms) / 1000 * 16000))
    fs = rt.frame_sec
    spk_every = max(1, int(round(spk_every_s / (rt.cs * fs))))
    import resource
    wall, cpu, spk_ms = [], [], []
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    for i in range(0, len(audio), push):
        t0, c0 = time.perf_counter(), time.process_time()
        rt.feed(audio[i:i + push])
        if rt.spk_frames and rt.steps and rt.steps % spk_every == 0:
            t1 = time.perf_counter()
            rt.speaker_embedding(last_frames=int(spk_window_s / fs))
            spk_ms.append((time.perf_counter() - t1) * 1000)
        wall.append((time.perf_counter() - t0) * 1000)
        cpu.append((time.process_time() - c0) * 1000)
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    rt.feed([], final=True)
    # user CPU = compute; system CPU is mostly page-fault / swap handling on a memory-starved machine
    return {"step_ms": wall, "cpu_ms": cpu, "spk_ms": spk_ms, "steps": rt.steps, "tokens": len(rt.tokens),
            "user_ms": (ru1.ru_utime - ru0.ru_utime) * 1000, "sys_ms": (ru1.ru_stime - ru0.ru_stime) * 1000,
            "majflt": ru1.ru_majflt - ru0.ru_majflt, "minflt": ru1.ru_minflt - ru0.ru_minflt}
