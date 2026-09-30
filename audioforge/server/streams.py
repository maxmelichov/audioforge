"""The ASR model's streaming passes (``ASRStream``: encoder + RNNT + VAD / turn / speaker heads per 160 ms chunk;
``LookaheadStream``: the text-only ``--asr-lookahead`` pass), the exact CPU convolution fast path (``fast_conv``)
and the input resampler."""
from __future__ import annotations

import math
import time
import types
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ..model import StreamingSession
from ..modules.fastconformer import StreamState
from .constants import FRAME_MS, SR
from .util import _Ring

__all__ = ["ASRStream", "fast_conv", "LookaheadStream", "Resampler"]


# --------------------------------------------------------------------------- ASR stream
class ASRStream(StreamingSession):
    """StreamingSession that returns per-frame records: {"v", "vad", "eot"} (eot None unless the turn head runs
    on the session's frames). Same mel / encoder path as ``StreamingSession.feed`` (identical text); the
    transducer is decoded frame by frame so the turn head's text state at frame v uses the tokens of frames <= v
    (as ``heads.turn.decoded_text_state``). turn_input "diar": chunks wait in ``pending`` for the diarizer."""

    def __init__(self, model, vad: str | None = "vad", turn: str | None = "turn", turn_input: str = "session",
                 lid=None, vad_gate: float | None = None, vad_hangover_frames: int = 15, att_context_size=None):
        super().__init__(model, att_context_size=att_context_size)
        self.lid = lid  # audioforge.lid.LIDStream (--lid) or None
        # --asr-vad-gate: the transducer is not decoded on a frame whose VAD <= vad_gate once vad_hangover_frames
        # such frames have passed since the last speech frame (the encoder and the heads still run every frame);
        # bounds the text emitted on long non-speech (research/archive/BULLETPROOF.md section 1)
        self.vad_gate = None if vad_gate is None else float(vad_gate)
        self.hangover = int(vad_hangover_frames)
        self.since_speech = 10 ** 9  # non-speech frames since the last speech frame (starts gated)
        self.n_gated = 0
        self.n_resets = 0
        self.lid_events: list[dict] = []
        self.frame_events = {}  # not used (records are returned instead)
        self.vad_name = vad if vad in model.heads else None
        self.turn_name = turn if turn in model.heads else None
        self.turn = model.heads[self.turn_name] if self.turn_name else None
        self.turn_state = self.turn.init_stream(1) if self.turn is not None else None
        self.turn_input = turn_input
        self.kernel = bool(self.turn_name and model.head_cfg[self.turn_name].get("condition_on_speaker"))
        if self.kernel and (self.turn_name in model.layer_tap or self.turn_name in model.layer_mix):
            # the kernel path streams the conditioned encoder's TOP layer (run_turn_on_diar); a from_layers tap on a
            # speaker-conditioned turn head (research/archive/LAYER_ROUTING.md) is not wired here and would be read wrongly
            raise NotImplementedError(f"serve: turn head {self.turn_name!r} is speaker-conditioned with from_layers "
                                      f"{model.head_cfg[self.turn_name].get('from_layers')!r}; not supported")
        self.k = getattr(self.turn, "k_tokens", 4)
        self.cs = self.att[1] + 1  # encoder frames per chunk
        self.n_frames = 0
        self.tok_at: list[int] = []  # tokens emitted at frames <= v (final cut points)
        self.pending: deque = deque()  # diar mode: dicts {mel, last, v0, n, snaps, enc}
        self.cstate = StreamState()  # diar mode, kernel heads: the speaker-conditioned encoder stream
        self.turn_ms = 0.0
        self.tsvad = None  # turn_input "tsvad": the session's tsvad_stream.TSVADTrack (set by Session)
        self.tsvad_p = _Ring()  # its [P(target), P(other)] per ASR frame
        self.keep_spk = False  # --diar-labels registry --diar-embed spk: keep the speaker head's input per frame
        self.spk_feats = _Ring()  # (D,) block-4 frame per ASR frame (last 2048), read by Session._embed
        self.seg = None  # turn head v5 (heads.turn_seg.SegTurnStream, ``attach_seg``): the segment classifier's inputs
        self.seg_block = None
        self.pros = None  # heads.prosody.Prosody when the v5 model reads prosody
        self.pros_frames = _Ring(512)

    def attach_seg(self, model, device="cpu"):
        """Turn head v5 (research/TURN_V5.md): keep the segment classifier's per-frame inputs (its encoder block of
        this pass, VAD, P(user) / P(other) when a print is enrolled, the decoded token count, prosody) so
        ``seg_prob(v)`` can classify the window ending at frame v. No extra encoder work."""
        from ..heads.turn_seg import SegTurnStream
        self.seg = SegTurnStream(model, device)
        self.seg_block = [int(b) for b in str(model.cfg["block"]).split("+")]
        if model.n_pros:
            from ..heads.prosody import Prosody
            self.pros = Prosody()

    def seg_prob(self, v: int) -> float:
        """P(the user's turn is complete) of the v5 classifier on the window ending at frame v."""
        return self.seg.prob(v, self.tokens)

    def reset_state(self):
        """Drop the recurrent state (encoder caches, transducer prediction state, turn GRU) after a non-finite
        output, keeping the frame / token counters and the signal buffer: the stream continues from the next chunk
        with a cold cache instead of carrying a NaN forever."""
        self.enc_state, self.cstate = StreamState(), StreamState()
        self.pred, self.skip = None, 0
        if self.turn is not None:
            self.turn_state = self.turn.init_stream(1)
        self.n_resets += 1

    def _turn_step(self, e, snap, act=None, cols=None, prim=None) -> float:
        g, toks = snap
        h = self.turn
        kw = {}
        if getattr(h, "needs_act", getattr(h, "concat", False)):
            kw["spk_act"] = torch.tensor([[float(act or 0.0)]], device=self.dev)
        if getattr(h, "needs_cols", False):
            kw["cols"] = torch.as_tensor(np.asarray(cols, np.float32), device=self.dev)[None, None]
            kw["prim"] = torch.tensor([int(prim or 0)], device=self.dev)
        return float(h.step(e, (g, None) if g is not None else None, [toks], state=self.turn_state, **kw)[0, 0])

    @torch.no_grad()
    def feed_frames(self, samples, final: bool = False) -> list[dict]:
        """Audio samples -> one record ``{"v", "vad", "eot"}`` per newly completed ASR frame (``final`` flushes)."""
        if self.lid is not None and hasattr(self.lid, "feed_audio"):  # the AmberNet backend keeps the raw audio
            self.lid.feed_audio(samples)
        if self.pros is not None and len(samples):
            for fr in self.pros.feed(np.asarray(samples, np.float32)):
                self.pros_frames.append(fr)
        x = torch.as_tensor(np.asarray(samples, np.float32), device=self.dev)
        pe = self.m.preprocessor.preemph
        if len(x):
            prev = torch.cat([torch.tensor([self.last], device=self.dev), x[:-1]])
            self.last = float(x[-1])
            self.sig = torch.cat([self.sig, x - pe * prev if pe else x])
        total = self.sig_start + len(self.sig)
        if total == 0:  # nothing received (an empty stream has no frames; the diarizer agrees)
            return []
        ready = (total - self.half) // self.hop + 1 if not final else total // self.hop + 1
        if ready > self.mel_done:
            self.mel_buf.append(self._mel(self.mel_done, ready))
            self.mel_done = ready
        keep = self.mel_done * self.hop - self.half - self.hop
        if keep > self.sig_start:
            self.sig = self.sig[keep - self.sig_start:]
            self.sig_start = keep
        mel = torch.cat(self.mel_buf, -1) if self.mel_buf else torch.zeros(1, 1, 0, device=self.dev)
        out = []
        while mel.shape[-1] >= self._next_chunk_mels() or (final and mel.shape[-1]):
            k = self._next_chunk_mels()  # a chunk runs as soon as its frames' mel input is complete (chunk_lead)
            last = bool(final) and mel.shape[-1] <= k
            chunk, mel = mel[..., :k], mel[..., k:]
            self.mel_fed += chunk.shape[-1]
            enc, hid, self.enc_state = self.m.encoder.stream_step(chunk, self.enc_state, self.att, final=last,
                                                                  return_hidden=True)
            n = enc.shape[1]
            if n == 0:
                continue
            f_asr = self.m.head_input(self.head_name, enc, hid)[0]
            vad = (self.m.heads[self.vad_name](self.m.head_input(self.vad_name, enc, hid)).sigmoid()[0].tolist()
                   if self.vad_name else [0.0] * n)
            e_turn = self.m.head_input(self.turn_name, enc, hid) if self.turn is not None else None
            if self.keep_spk:  # the speaker head's tap of this chunk (no extra encoder pass): per-turn voice ids
                for fr in self.m.head_input("spk", enc, hid)[0].float().cpu().numpy():
                    self.spk_feats.append(fr)
            if self.tsvad is not None:  # block 4 of this chunk (the speaker head's tap): no extra encoder pass
                for p in self.tsvad.feed(self.m.head_input("spk", enc, hid), vad):
                    self.tsvad_p.append(p)
            if self.lid is not None:  # same chunk, same per-layer outputs: no extra encoder pass
                ends = [(self.n_frames + j + 1) * FRAME_MS / 1000 for j in range(n)]
                self.lid_events += self.lid.feed(enc, hid, vad, ends)
            if self.seg is not None:
                seg_x = torch.cat([hid[b - 1] for b in self.seg_block], -1)[0].float().cpu().numpy()
            snaps = []
            for j in range(n):
                if self.vad_gate is not None:
                    gated = vad[j] <= self.vad_gate and self.since_speech >= self.hangover
                    self.since_speech = 0 if vad[j] > self.vad_gate else self.since_speech + 1
                else:
                    gated = False
                if gated:  # no transducer step on this frame (as if it decoded blank); TDT skips still count down
                    self.skip = max(0, self.skip - 1)
                    self.n_gated += 1
                else:
                    self._decode(f_asr[j: j + 1])
                snap = (self.pred[0] if self.pred is not None else None, list(self.tokens[-self.k:]))
                eot = None
                if self.turn is not None and self.turn_input == "session":
                    t0 = time.perf_counter()
                    eot = self._turn_step(e_turn[:, j: j + 1], snap)
                    self.turn_ms += (time.perf_counter() - t0) * 1000
                snaps.append(snap)
                self.tok_at.append(len(self.tokens))
                if self.seg is not None:
                    v = self.n_frames
                    en = self.tsvad is not None and getattr(self.tsvad, "enrolled", False) and v < len(self.tsvad_p)
                    pu, po = (float(self.tsvad_p[v][0]), float(self.tsvad_p[v][1])) if en else (0.0, 0.0)
                    pr = None
                    if self.pros is not None:
                        pr = self.pros_frames[v] if v < len(self.pros_frames) else np.zeros(12, np.float32)
                    self.seg.push(seg_x[j], float(vad[j]), pu, po, len(self.tokens), pr)
                out.append({"v": self.n_frames, "vad": float(vad[j]), "eot": eot})
                self.n_frames += 1
            if self.turn is not None and self.turn_input in ("diar", "tsvad"):
                self.pending.append({"mel": chunk, "last": last, "v0": self.n_frames - n, "n": n, "snaps": snaps,
                                     "vad": vad, "enc": None if self.kernel else e_turn})
        self.mel_buf = [mel] if mel.shape[-1] else []
        return out

    @torch.no_grad()
    def run_turn_on_diar(self, avail: int, act_fn, flush: bool = False) -> list[tuple[int, float, float]]:
        """diar mode: run the turn head on pending chunks whose frames all have a diarizer column (< avail).
        act_fn(v) -> (primary activity, columns, primary index). -> [(v, eot, vad)]."""
        res = []
        while self.pending and (flush or self.pending[0]["v0"] + self.pending[0]["n"] <= avail):
            c = self.pending.popleft()
            t0 = time.perf_counter()
            info = [act_fn(v) for v in range(c["v0"], c["v0"] + c["n"])]
            if self.kernel:
                act = torch.tensor([[a for a, _, _ in info]], dtype=torch.float32, device=self.dev)
                e, self.cstate = self.m.encoder.stream_step(c["mel"], self.cstate, self.att, spk_act=act,
                                                            final=c["last"])
            else:
                e = c["enc"]
            for j in range(min(c["n"], e.shape[1])):
                a, cols, prim = info[j]
                res.append((c["v0"] + j, self._turn_step(e[:, j: j + 1], c["snaps"][j], a, cols, prim), c["vad"][j]))
            self.turn_ms += (time.perf_counter() - t0) * 1000
        return res


class LookaheadStream(StreamingSession):
    """--asr-lookahead: text-only cache-aware streaming pass of the ASR model at att_context [70, R] (same weights,
    longer chunks). Decoded frame by frame so ``tok_at[v]`` = tokens emitted at frames <= v; ``ready_t(v)`` = the
    audio time at which frame v is decoded (its chunk's audio + the STFT half window)."""

    def __init__(self, model, right: int):
        left = model.encoder.att_context_size[0]
        super().__init__(model, att_context_size=[left, int(right)])
        self.frame_events = {}  # text only: no frame heads
        self.cs = int(right) + 1
        self.n_frames = 0
        self.tok_at: list[int] = []
        self.ms = 0.0

    def _decode(self, f):
        for j in range(f.shape[0]):
            super()._decode(f[j: j + 1])
            self.tok_at.append(len(self.tokens))
            self.n_frames += 1

    def ready_t(self, v: int) -> float:
        """Audio time (s) by which the lookahead pass can decode frame ``v`` (its chunk and lookahead have arrived)."""
        return self.frame_ready_samples(v) / SR


# --------------------------------------------------------------------------- CPU fast path
def _fast_conv_forward(self, x, pad_mask=None, cache=None):
    """modules.fastconformer.ConvModule.forward with the same math: pointwise convs as F.linear and the depthwise
    conv as unfold * w (PyTorch's CPU depthwise Conv1d costs ~1.2 ms per call on a 2-frame chunk; runtime.py uses
    the same trick). Installed per instance by ``fast_conv``; outputs match the original to float rounding."""
    k = self.kernel
    h = F.glu(F.linear(x, self.pw1.weight[..., 0], self.pw1.bias), dim=-1).transpose(1, 2)  # B,D,T
    if pad_mask is not None:
        h = h.masked_fill(pad_mask[:, None], 0.0)
    if cache is not None:
        h = torch.cat([cache, h], dim=2)
        new_cache = h[:, :, -(k - 1):]
    else:
        new_cache = None
        left = k - 1 if self.causal else (k - 1) // 2
        h = F.pad(h, (left, k - 1 - left))
    y = (h.unfold(2, k, 1) * self.dw.weight[:, 0][None, :, None]).sum(-1) + self.dw.bias[None, :, None]
    y = self.norm(y) if self.norm_type == "batch" else self.norm(y.transpose(1, 2)).transpose(1, 2)
    return F.linear(F.silu(y).transpose(1, 2), self.pw2.weight[..., 0], self.pw2.bias), new_cache


def fast_conv(model) -> int:
    """Install ``_fast_conv_forward`` on every ConvModule of ``model`` (inference only). Returns the count."""
    from ..modules.fastconformer import ConvModule
    n = 0
    for mod in model.modules():
        if isinstance(mod, ConvModule) and mod.dw.bias is not None and mod.pw1.bias is not None:
            mod.forward = types.MethodType(_fast_conv_forward, mod)
            n += 1
    return n


# --------------------------------------------------------------------------- engine + session
class Resampler:
    """Streaming linear-interpolation resampler (state carried across blocks). Send 16 kHz for best quality."""

    def __init__(self, sr_in: int, sr_out: int = SR):
        self.ratio = sr_in / sr_out
        self.pos = 0.0  # position of the next output sample, relative to self.tail[0]
        self.tail = np.zeros(0, np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1.0:
            return x
        buf = np.concatenate([self.tail, x])
        if len(buf) < 2:
            self.tail = buf
            return np.zeros(0, np.float32)
        n = int(math.floor((len(buf) - 1 - self.pos) / self.ratio)) + 1
        n = max(0, n)
        idx = self.pos + np.arange(n) * self.ratio
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        self.pos = self.pos + n * self.ratio - (len(buf) - 1)
        self.tail = buf[-1:]
        return out
