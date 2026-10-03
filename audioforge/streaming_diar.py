"""Streaming Sortformer with an Arrival-Order Speaker Cache (AOSC), and live speaker-attributed ASR.

Streaming Sortformer (Medennikov et al., arXiv 2507.18446; NeMo ``diar_streaming_sortformer_4spk``,
Nemotron-3-Diarization) runs an offline-trained, arrival-order Sortformer chunk by chunk. Each step
sees ``[speaker cache ‖ FIFO ‖ chunk ‖ right context]``:

* the **FIFO** holds the embeddings of the most recent frames (short-term context);
* the **speaker cache** holds a few high-confidence frames of every speaker heard so far, laid out
  slot 0 first, then slot 1, ... (arrival order). The model was trained with the sort loss to
  put the first-appearing speaker in output 0, and the cache makes the first speaker in the *input*
  be the one already assigned to slot 0, so slot identities stay fixed across chunks.

All lengths are in 80 ms encoder frames, with NeMo's names: ``chunk_len``, ``chunk_right_context``,
``fifo_len``, ``spkcache_len``, ``spkcache_update_period``. Input-buffer latency = (chunk_len +
chunk_right_context) x 80 ms, as on the NVIDIA model cards.

Classes
    AOSCConfig           the streaming configuration
    SpeakerCache         the cache/FIFO state and the cache-update (compression) rule
    StreamingDiarizer    feed(samples) -> per-frame (T_new, S) speaker probabilities
    StreamingSpeakerASR  StreamingDiarizer + speaker-kernel ASR: incremental text per speaker slot

Encoders. A causal encoder (``causal: true``) is run with its cache-aware ``stream_step`` exactly
once per frame (its outputs never change, so right-context frames are just buffered). A non-causal
encoder is re-run per step on an audio window [left context ‖ chunk ‖ right context].

Deviation from the paper: the cache and FIFO store the Sortformer head's per-frame embeddings of
the *encoder output*, and only the head transformer is re-run over them. NVIDIA caches pre-encode
(subsampling) outputs and re-runs the whole FastConformer + transformer over the concatenation.
Ours keeps one encoder pass per frame; theirs lets the encoder itself attend across the cache.
Because our head then needs to know where the cache is, SortformerHead gets sinusoidal positions
(``pos_emb: true``).

Training (``python -m audioforge.streaming_diar train recipes/streaming_sortformer.yaml``) adds
multi-turn conversations with reappearing speakers and a *streaming-simulation* loss: the head is
also trained on [cache ‖ fifo ‖ chunk ‖ rc] sequences whose cache is selected from ground truth
(teacher forcing) with the same slot-major layout as inference, and whose targets use the global
arrival order of the whole conversation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ToneLanguage, synthetic_dataset
from .heads.asr import CTCHead
from .heads.audio import _match_len, sort_by_arrival
from .metrics import edit_distance, frame_der
from .modules.fastconformer import StreamState

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- configuration
@dataclass
class AOSCConfig:
    chunk_len: int = 6               # frames emitted per step
    chunk_right_context: int = 2     # look-ahead frames the head sees (predictions discarded)
    fifo_len: int = 20               # most recent past frames kept verbatim
    spkcache_len: int = 40           # speaker-cache capacity in frames (0 disables the cache)
    spkcache_update_period: int = 10  # frames moved FIFO -> cache per update (min pop size)
    sil_frames: int = 4              # non-speech frames kept in the cache (NeMo keeps a few too)
    strong_boost_rate: float = 0.75  # fraction of the per-speaker share guaranteed to each speaker
    threshold: float = 0.5

    @property
    def latency_ms(self) -> float:
        return (self.chunk_len + self.chunk_right_context) * 80.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "AOSCConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    @classmethod
    def preset(cls, name: str) -> "AOSCConfig":
        """A named NVIDIA card configuration (SORTFORMER_PRESETS); fields not in the preset keep their defaults."""
        return cls.from_dict(SORTFORMER_PRESETS[name])


# NVIDIA streaming Sortformer v2 card configurations (table
# "Configuration"), in 80 ms frames. Input buffer latency = (chunk_len + chunk_right_context) * 80 ms. The card lists
# the same FIFO / update period / cache for both low-latency rows. Used with StreamingDiarizer(mode="window",
# enc_left_context=188, **SORTFORMER_PRESETS[name]) (scripts/make_sortformer_tracks.py, scripts/eval_stage1.py).
SORTFORMER_PRESETS = {
    "low_latency": dict(chunk_len=6, chunk_right_context=7, fifo_len=188, spkcache_update_period=144,
                        spkcache_len=188),      # card "low latency", 1.04 s (the cached tracks)
    "low_latency_032": dict(chunk_len=3, chunk_right_context=1, fifo_len=188, spkcache_update_period=144,
                            spkcache_len=188),  # card "ultra low latency", 0.32 s
}


# --------------------------------------------------------------------------- cache update rule
def cache_scores(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """(N,S) probabilities -> (N,S) log-score that frame n is speaker s *alone*:

        score[n,s] = log p[n,s] + sum_{s' != s} log(1 - p[n,s'])     (-inf where p[n,s] <= threshold)

    High = confident and non-overlapped (NeMo's ``_get_log_pred_scores`` up to a constant).
    """
    p = preds.float().clamp(1e-6, 1 - 1e-6)
    l1 = torch.log1p(-p)
    s = p.log() - l1 + l1.sum(-1, keepdim=True)
    return s.masked_fill(p <= threshold, float("-inf"))


def select_cache_frames(preds: torch.Tensor, times: torch.Tensor, cap: int, sil_frames: int = 4,
                        strong_boost_rate: float = 0.75, threshold: float = 0.5) -> torch.Tensor:
    """The AOSC compression rule. Returns indices into the candidates, in cache layout order.

    1. score every (frame, speaker) with ``cache_scores``; a frame can serve only one speaker;
    2. each speaker is guaranteed its best ``floor(strong_boost_rate * (cap - n_sil) / S)`` frames;
    3. the remaining capacity goes to the best remaining (frame, speaker) pairs overall;
    4. up to ``sil_frames`` non-speech frames (all p <= threshold; lowest max-probability first);
    5. layout: slot 0's frames (in time order), slot 1's, ..., then the silence frames. Slot-major
       = arrival order, because the sort-loss model numbers speakers by arrival.
    """
    N, S = preds.shape
    if cap <= 0 or N == 0:
        return torch.zeros(0, dtype=torch.long)
    sc = cache_scores(preds, threshold).cpu()
    silent = (preds.max(-1).values <= threshold).cpu()
    n_sil = min(sil_frames, int(silent.sum()), cap)
    cap_spk = cap - n_sil
    q = int(strong_boost_rate * cap_spk / S)
    slot = torch.full((N,), -1, dtype=torch.long)
    taken = 0
    if q > 0:
        for s in range(S):
            k = 0
            for n in sc[:, s].argsort(descending=True).tolist():
                if k >= q or not math.isfinite(sc[n, s]):
                    break
                if slot[n] < 0:
                    slot[n], k = s, k + 1
            taken += k
    flat = sc.flatten()
    for i in flat.argsort(descending=True).tolist():
        if taken >= cap_spk or not math.isfinite(flat[i]):
            break
        n, s = divmod(i, S)
        if slot[n] < 0:
            slot[n], taken = s, taken + 1
    t = times.cpu()
    order = []
    for s in range(S):
        idx = (slot == s).nonzero().flatten()
        order.append(idx[t[idx].argsort()])
    sil_idx = (silent & (slot < 0)).nonzero().flatten()
    sil_idx = sil_idx[preds.max(-1).values.cpu()[sil_idx].argsort()[:n_sil]]
    order.append(sil_idx[t[sil_idx].argsort()])
    return torch.cat(order)


class SpeakerCache:
    """AOSC + FIFO state. ``step`` runs the head over [cache ‖ fifo ‖ chunk ‖ rc] and updates the state.

    head_fn(cache, fifo, chunk, rc) -> (N,S) probabilities for the concatenated sequence, where every
    argument is an (N_i, D) tensor of per-frame embeddings.

    Update (after each step, as NeMo's ``streaming_update``):
      * the cache's and FIFO's stored predictions are refreshed with this step's outputs;
      * the chunk (embeddings + predictions + absolute frame times) is appended to the FIFO;
      * if the FIFO exceeds ``fifo_len``, its oldest max(spkcache_update_period, overflow) frames are
        popped and appended to the cache (in time order while the cache has room);
      * if the cache then exceeds ``spkcache_len`` it is compressed with ``select_cache_frames``.
    """

    def __init__(self, cfg: AOSCConfig, num_spks: int):
        self.cfg, self.S = cfg, num_spks
        self.c_emb = self.f_emb = None
        self.c_pred = self.f_pred = torch.zeros(0, num_spks)
        self.c_t = self.f_t = torch.zeros(0, dtype=torch.long)
        self.n_compress = 0

    def _empty(self, like):
        return like.new_zeros(0, like.shape[-1])

    @torch.no_grad()
    def step(self, chunk: torch.Tensor, rc: torch.Tensor, t0: int, head_fn):
        if self.c_emb is None:
            self.c_emb, self.f_emb = self._empty(chunk), self._empty(chunk)
            self.c_pred = self.f_pred = chunk.new_zeros(0, self.S)
        probs = head_fn(self.c_emb, self.f_emb, chunk, rc)
        nc, nf, n = len(self.c_emb), len(self.f_emb), len(chunk)
        self.c_pred, self.f_pred = probs[:nc], probs[nc: nc + nf]
        cp, rp = probs[nc + nf: nc + nf + n], probs[nc + nf + n:]
        self.f_emb = torch.cat([self.f_emb, chunk])
        self.f_pred = torch.cat([self.f_pred, cp])
        self.f_t = torch.cat([self.f_t, torch.arange(t0, t0 + n)])
        cfg = self.cfg
        if len(self.f_emb) > cfg.fifo_len:
            pop = min(len(self.f_emb), max(cfg.spkcache_update_period, len(self.f_emb) - cfg.fifo_len))
            self.c_emb = torch.cat([self.c_emb, self.f_emb[:pop]])
            self.c_pred = torch.cat([self.c_pred, self.f_pred[:pop]])
            self.c_t = torch.cat([self.c_t, self.f_t[:pop]])
            self.f_emb, self.f_pred, self.f_t = self.f_emb[pop:], self.f_pred[pop:], self.f_t[pop:]
            if len(self.c_emb) > cfg.spkcache_len:
                idx = select_cache_frames(self.c_pred, self.c_t, cfg.spkcache_len, cfg.sil_frames,
                                          cfg.strong_boost_rate, cfg.threshold)
                dev = self.c_emb.device
                self.c_emb, self.c_pred, self.c_t = self.c_emb[idx.to(dev)], self.c_pred[idx.to(dev)], self.c_t[idx]
                self.n_compress += 1
        return cp, rp


# --------------------------------------------------------------------------- incremental front ends
class _MelStream:
    """Incremental log-mel identical to the offline preprocessor (same code as StreamingSession)."""

    def __init__(self, model):
        pp = model.preprocessor
        self.pp, self.hop, self.half = pp, pp.hop, pp.n_fft // 2
        self.dev = next(model.parameters()).device
        self.sig = torch.zeros(0, device=self.dev)
        self.sig_start, self.last, self.mel_done = 0, 0.0, 0

    def _mel(self, a, b):
        pp = self.pp
        s0, s1 = a * self.hop - self.half, (b - 1) * self.hop + self.half
        seg = torch.zeros(s1 - s0, device=self.dev)
        lo, hi = max(s0, self.sig_start), min(s1, self.sig_start + len(self.sig))
        if hi > lo:
            seg[lo - s0: hi - s0] = self.sig[lo - self.sig_start: hi - self.sig_start]
        spec = torch.stft(seg[None], pp.n_fft, pp.hop, pp.win, pp.window, center=False, return_complex=True)
        return pp.fixed_norm(torch.log(pp.fb @ (spec.real ** 2 + spec.imag ** 2) + 2 ** -24))

    def feed(self, x: torch.Tensor, final: bool) -> torch.Tensor:
        pe = self.pp.preemph
        if len(x):
            prev = torch.cat([torch.tensor([self.last], device=self.dev), x[:-1]])
            self.last = float(x[-1])
            self.sig = torch.cat([self.sig, x - pe * prev if pe else x])
        total = self.sig_start + len(self.sig)
        ready = (total - self.half) // self.hop + 1 if not final else total // self.hop + 1
        out = None
        if ready > self.mel_done:
            out = self._mel(self.mel_done, ready)
            self.mel_done = ready
        keep = self.mel_done * self.hop - self.half - self.hop
        if keep > self.sig_start:
            self.sig = self.sig[keep - self.sig_start:]
            self.sig_start = keep
        return out if out is not None else torch.zeros(1, self.pp.n_mels, 0, device=self.dev)


class _IncDecoder:
    """Incremental greedy CTC / RNNT / TDT decoding over successive encoder frames (as StreamingSession)."""

    def __init__(self, head, tokenizer):
        self.h, self.tok = head, tokenizer
        self.tokens, self.pred, self.skip, self.prev = [], None, 0, -1

    @torch.no_grad()
    def feed(self, f: torch.Tensor):
        h = self.h
        if f.shape[0] == 0:
            return
        if isinstance(h, CTCHead):
            for p in h(f[None]).argmax(-1)[0].tolist():
                if p != h.blank and p != self.prev:
                    self.tokens.append(p)
                self.prev = p
            return
        V1, dev = h.vocab_size + 1, f.device
        if self.pred is None:
            self.pred = h.pred(torch.tensor([[h.blank]], device=dev), None)
        g, st = self.pred
        fe = h.joint.enc(f)
        t, emitted = self.skip, 0
        while t < f.shape[0]:
            z = h.joint.out(fe[t][None, None] + h.joint.pred(g))[0, 0]
            k = int(z[:V1].argmax())
            d = h.durations[int(z[V1:].argmax())] if h.is_tdt else (1 if k == h.blank else 0)
            if k == h.blank:
                d = max(d, 1)
            else:
                self.tokens.append(k)
                g, st = h.pred(torch.tensor([[k]], device=dev), st)
                emitted += 1
            if d == 0 and emitted >= h.max_symbols:
                d = 1
            if d:
                emitted = 0
            t += d
        self.skip = t - f.shape[0]
        self.pred = (g, st)

    @property
    def text(self) -> str:
        return self.tok.decode(self.tokens)


# --------------------------------------------------------------------------- streaming diarizer
class StreamingDiarizer:
    """Chunked Sortformer inference with an arrival-order speaker cache.

    ``feed(samples, final=False)`` accepts any number of new 16 kHz samples and returns the
    probabilities (T_new, S) of the frames finalized by this call. After ``final=True`` the total
    number of frames equals the offline encoder's frame count.

    mode: "causal" (cache-aware ``stream_step``; needs ``causal: true`` and frame-local mel
    normalization), "window" (re-encode [enc_left_context ‖ chunk ‖ rc] frames of audio per step;
    works for any encoder), or "auto" (causal if the encoder is causal).
    Config: the model's ``streaming:`` recipe block, overridden by keyword arguments.
    """

    def __init__(self, model, diar_head: str = "diar", mode: str = "auto", enc_left_context: int = 24,
                 keep_mel: bool = False, keep_probs: bool = True, **cfg):
        self.keep_probs = keep_probs  # False (audioforge.serve): do not accumulate every emitted column (unbounded)
        self.skipped = 0  # frames emitted without running the model (feed(skip=True), load shedding)
        self.m = model.eval()
        self.head_name = diar_head
        self.head = model.heads[diar_head]
        assert diar_head not in model.layer_mix, "streaming needs the diar head on the top encoder layer"
        base = {k: v for k, v in (model.cfg.get("streaming") or {}).items() if k != "train"}
        self.cfg = AOSCConfig.from_dict({**base, **cfg})
        self.mode = ("causal" if model.encoder.causal else "window") if mode == "auto" else mode
        self.dev = next(model.parameters()).device
        self.fs = model.preprocessor.hop * model.encoder.subsampling_factor  # samples per frame (1280)
        self.cache = SpeakerCache(self.cfg, self.head.num_spks)
        self.done = 0            # frames emitted
        self.listeners = []      # callbacks(start, chunk_probs, rc_probs)
        self.probs = []          # emitted (n,S) tensors
        self.steps = 0
        self.enc_frames_computed = 0  # encoder frame computations (cost accounting)
        if self.mode == "causal":
            assert model.encoder.causal, "causal mode needs a causal encoder"
            assert model.preprocessor.normalize in (None, "none", "NA", "fixed"), \
                "causal streaming needs frame-local mel normalization (normalize: fixed or none)"
            self.att = list(model.encoder.att_context_size)
            self.chunk_mel = model.encoder.stream_chunk_frames(self.att)
            self.mel = _MelStream(model)
            self.mel_buf = torch.zeros(1, model.preprocessor.n_mels, 0, device=self.dev)
            self.enc_state = StreamState()
            self._emb = None       # head embeddings of encoded-but-not-emitted frames
            self._encoded = 0      # frames encoded so far
            self.keep_mel = keep_mel
            self.mel_chunks = deque()  # (start_frame, n_frames, mel) for downstream (ASR) re-encoding
        else:
            self.left = enc_left_context
            self.audio = torch.zeros(0, device=self.dev)
            self.audio_start = 0
            self.total = 0

    # ---- head
    def _head_fn(self, c, f, x, r):
        return self.head.forward_chunk(c[None], f[None], x[None], r[None])[0].float().sigmoid()

    # ---- encoder front ends
    def _push(self, x: torch.Tensor, final: bool) -> int:
        """Add samples; returns the number of frames available for emission (absolute)."""
        if self.mode == "causal":
            mel = torch.cat([self.mel_buf, self.mel.feed(x, final)], -1)
            new = []
            while mel.shape[-1] >= self.chunk_mel or (final and mel.shape[-1]):
                c, mel = mel[..., : self.chunk_mel], mel[..., self.chunk_mel:]
                enc, self.enc_state = self.m.encoder.stream_step(c, self.enc_state, self.att)
                n = enc.shape[1]
                if self.keep_mel:
                    self.mel_chunks.append((self._encoded, n, c))
                self._encoded += n
                self.enc_frames_computed += n
                new.append(self.head.embed_frames(enc[0]))
            self.mel_buf = mel
            if new:
                self._emb = torch.cat(([self._emb] if self._emb is not None else []) + new)
            return self._encoded
        self.audio = torch.cat([self.audio, x])
        self.total += len(x)
        if final:
            return ToneLanguage.n_frames(self.total) if self.total else 0
        return self.total // self.fs

    def window(self, a: int, b: int):
        """Window-mode audio for frames [a - left, b): returns (samples, first_frame)."""
        s0 = max(0, a - self.left)
        lo, hi = s0 * self.fs - self.audio_start, min(b * self.fs, self.total) - self.audio_start
        return self.audio[lo:hi], s0

    def _frames(self, a: int, b: int) -> torch.Tensor:
        if self.mode == "causal":
            off = self._encoded - len(self._emb)
            return self._emb[a - off: b - off]
        x, s0 = self.window(a, b)
        enc, _ = self.m.encode(x[None], torch.tensor([len(x)], device=self.dev))
        self.enc_frames_computed += enc.shape[1]
        return self.head.embed_frames(enc[0, a - s0: b - s0])

    def _trim(self):
        if self.mode == "causal":
            off = self._encoded - len(self._emb)
            self._emb = self._emb[self.done - off:]
        else:
            keep = max(0, self.done - self.left) * self.fs
            if keep > self.audio_start:
                self.audio = self.audio[keep - self.audio_start:]
                self.audio_start = keep

    @torch.no_grad()
    def feed(self, samples, final: bool = False, skip: bool = False) -> torch.Tensor:
        """Add samples -> the newly finalized columns (T_new, S). ``skip=True`` (load shedding): the frames that would
        be finalized now are emitted as zeros without running the encoder or the head (the speaker cache is not
        updated; window mode resumes with full left context, causal mode still encodes for its own cache)."""
        x = torch.as_tensor(np.asarray(samples, dtype=np.float32), device=self.dev).flatten()
        avail = self._push(x, final)
        C, R = self.cfg.chunk_len, self.cfg.chunk_right_context
        out = []
        while avail - self.done >= C + R or (final and avail > self.done):
            n = min(C, avail - self.done)
            r = min(R, avail - self.done - n)
            if skip:
                cp, rp = torch.zeros(n, self.head.num_spks, device=self.dev), None
                self.skipped += n
            else:
                emb = self._frames(self.done, self.done + n + r)
                cp, rp = self.cache.step(emb[:n], emb[n:], self.done, self._head_fn)
            start, self.done = self.done, self.done + n
            self.steps += 1
            if self.keep_probs:
                self.probs.append(cp)
            out.append(cp)
            for cb in self.listeners:
                cb(start, cp, rp)
            self._trim()
        return torch.cat(out) if out else torch.zeros(0, self.head.num_spks, device=self.dev)

    @property
    def all_probs(self) -> torch.Tensor:
        if len(self.probs) > 1:
            self.probs = [torch.cat(self.probs)]
        return self.probs[0] if self.probs else torch.zeros(0, self.head.num_spks, device=self.dev)


# --------------------------------------------------------------------------- streaming speaker-attributed ASR
class StreamingSpeakerASR:
    """Live "who said what": StreamingDiarizer + speaker-kernel ASR (multitalker-Parakeet style).

    Diarization uses one encoder pass. Every *active* speaker slot additionally runs its own
    encoder pass with speaker kernels steered by that slot's streaming activity, and its own
    incremental transducer decoder. Cost: (1 + #active slots) encoder passes per frame.

    causal mode: each slot keeps a cache-aware encoder state; a chunk of mel is re-encoded for a slot
    as soon as the diarizer has *finalized* that chunk's probabilities (so ASR adds no latency beyond
    the diarizer's right context). A slot starts when its probability exceeds ``threshold`` on
    ``min_active_frames`` frames; it then replays up to ``replay_frames`` of buffered mel so the
    encoder has left context. window mode: each slot re-encodes the diarizer's window with its
    activity (right-context activity is the current, provisional prediction).
    """

    def __init__(self, model, asr_head: str | None = None, diar_head: str = "diar", threshold: float = 0.5,
                 min_active_frames: int = 2, replay_frames: int = 32, **diar_kw):
        self.m = model.eval()
        self.asr_name = asr_head or next(k for k, v in model.head_cfg.items() if v.get("condition_on_speaker"))
        self.asr = model.heads[self.asr_name]
        self.diar = StreamingDiarizer(model, diar_head, keep_mel=True, **diar_kw)
        self.diar.listeners.append(self._on_chunk)
        self.thr, self.min_active, self.replay = threshold, min_active_frames, replay_frames
        S = self.diar.head.num_spks
        self.active_count = torch.zeros(S)
        self.slots: dict[int, dict] = {}
        self.hist = deque()  # causal: (start, n, mel) already re-encoded chunks, for replay
        self.asr_frames_computed = 0
        self.events: list[tuple[float, int, str]] = []  # (time_sec, slot, text so far) on text change

    def _new_slot(self, s):
        slot = {"dec": _IncDecoder(self.asr, self.m.tokenizer), "state": StreamState()}
        self.slots[s] = slot
        if self.diar.mode == "causal":
            for start, n, mel in self.hist:
                self._run_causal(s, start, n, mel)
        return slot

    def _act(self, a, b, s):
        return self.diar.all_probs[a:b, s]

    def _run_causal(self, s, start, n, mel):
        slot = self.slots[s]
        act = self._act(start, start + n, s)[None]
        enc, slot["state"] = self.m.encoder.stream_step(mel, slot["state"], self.diar.att, spk_act=act)
        self.asr_frames_computed += enc.shape[1]
        slot["dec"].feed(enc[0])

    @torch.no_grad()
    def _on_chunk(self, start, cp, rp):
        d = self.diar
        self.active_count += (cp > self.thr).sum(0).float().cpu()
        new = [s for s in range(len(self.active_count))
               if s not in self.slots and self.active_count[s] >= self.min_active]
        before = {s: sl["dec"].text for s, sl in self.slots.items()}
        if d.mode == "causal":
            ready = []
            while d.mel_chunks and d.mel_chunks[0][0] + d.mel_chunks[0][1] <= d.done:
                ready.append(d.mel_chunks.popleft())
            for s in new:
                self._new_slot(s)
            for start_c, n, mel in ready:
                for s in self.slots:
                    self._run_causal(s, start_c, n, mel)
                self.hist.append((start_c, n, mel))
                while sum(h[1] for h in self.hist) > self.replay:
                    self.hist.popleft()
        else:
            for s in new:
                self._new_slot(s)
            n, r = len(cp), len(rp)
            if self.slots:
                x, s0 = d.window(start, start + n + r)
                act_all = torch.cat([d.all_probs[s0:start + n], rp])  # past + chunk final, rc provisional
                for s, slot in self.slots.items():
                    enc, _ = self.m.encode(x[None], torch.tensor([len(x)], device=d.dev), spk_act=act_all[None, :, s])
                    self.asr_frames_computed += enc.shape[1]
                    slot["dec"].feed(enc[0, start - s0: start - s0 + n])
        t = (start + len(cp)) * 0.08
        for s, sl in self.slots.items():
            if sl["dec"].text != before.get(s, ""):
                self.events.append((t, s, sl["dec"].text))

    def feed(self, samples, final: bool = False) -> dict[int, str]:
        self.diar.feed(samples, final)
        return self.texts

    @property
    def texts(self) -> dict[int, str]:
        return {s: sl["dec"].text for s, sl in sorted(self.slots.items())}

    def transcript(self) -> list[dict]:
        return [{"speaker": s, "text": t} for s, t in self.texts.items() if t]


# --------------------------------------------------------------------------- data: conversations
def compose_conversation(lang: ToneLanguage, rng: random.Random, speakers: list[int], n_segments: int,
                         gap=(0.2, 0.8), max_spks: int = 4) -> dict:
    """Concatenate ``n_segments`` 2-speaker overlapped mixtures (as ToneLanguage.mixture) drawn from
    one speaker pool, so speakers reappear after others spoke. Columns of ``spk_targets`` are in
    global arrival order; ``speakers[i]['text']`` is speaker i's turns joined in time order."""
    sr = lang.sr
    pool = list(speakers)
    unseen = pool[:]
    parts, t0 = [], int(sr * rng.uniform(0.1, 0.3))
    for seg in range(n_segments):
        if seg == 0:
            pair = unseen[:2]
        else:
            others = [s for s in pool if s not in unseen[:1]]
            pair = unseen[:1] + rng.sample(others, 1) if unseen else rng.sample(pool, min(2, len(pool)))
            rng.shuffle(pair)
        unseen = [s for s in unseen if s not in pair]
        cur = t0
        for j, s in enumerate(pair):
            text = lang.sentence(rng, (1, 3))
            x, act = lang.render(text, s, rng)
            start = cur + (int(sr * rng.uniform(0.1, 0.3)) if j else 0)
            parts.append((s, text, x, act, start))
            cur = start + int(len(x) * rng.uniform(0.6, 1.0))  # overlap up to 40%
        t0 = max(st + len(x) for _, _, x, _, st in parts) + int(sr * rng.uniform(*gap))
    N = max(st + len(x) for _, _, x, _, st in parts) + int(0.3 * sr)
    mix = np.zeros(N, np.float32)
    T = lang.n_frames(N)
    order = []
    for s, *_ in sorted(parts, key=lambda p: p[4]):
        if s not in order:
            order.append(s)
    order = order[:max_spks]
    tgt = np.zeros((T, max_spks), np.float32)
    texts = {s: [] for s in order}
    full = {s: np.zeros(N, bool) for s in order}
    for s, text, x, act, st in sorted(parts, key=lambda p: p[4]):
        mix[st: st + len(x)] += x
        if s in full:
            full[s][st: st + len(x)] |= act
            texts[s].append(text)
    for i, s in enumerate(order):
        tgt[:, i] = lang.frames(full[s], T)
    mix += 0.005 * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(N).astype(np.float32)
    return dict(audio=mix, spk_targets=tgt, speakers=[dict(speaker=s, text=" ".join(texts[s])) for s in order])


def conversation_dataset(n: int, seed: int = 0, max_segments: int = 3, pool=(2, 3), p_single: float = 0.3,
                         lang: ToneLanguage | None = None, attributed: bool = True) -> list[dict]:
    """Training data: short mixtures and multi-segment conversations with reappearing speakers.
    attributed=True adds one random speaker's activity + text for the speaker-kernel ASR head."""
    lang = lang or ToneLanguage(seed=0)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < p_single:
            k, nseg = 2, 1
        else:
            k = rng.choice(list(pool))
            nseg = rng.randint(max(2, k - 1), max(2, max_segments))
        conv = compose_conversation(lang, rng, rng.sample(range(len(lang.speakers)), k), nseg)
        if attributed:
            i = rng.randrange(len(conv["speakers"]))
            conv.update(spk_act=conv["spk_targets"][:, i], text=conv["speakers"][i]["text"])
        out.append(conv)
    return out


def long_conversations(n: int, seed: int = 3, n_segments: int = 10, n_spk: int = 3) -> list[dict]:
    lang, rng = ToneLanguage(seed=0), random.Random(seed)
    return [compose_conversation(lang, rng, rng.sample(range(len(lang.speakers)), n_spk), n_segments)
            for _ in range(n)]


# --------------------------------------------------------------------------- streaming-simulation training
def teacher_cache(y: torch.Tensor, cap: int, sil_frames: int, rng: random.Random) -> torch.Tensor:
    """Ground-truth version of the cache selection (training only): single-speaker frames shared
    round-robin among speakers (random picks), laid out slot-major in time order, then silence."""
    single = (y > 0.5).sum(1) == 1
    S = y.shape[1]
    cand = [((y[:, s] > 0.5) & single).nonzero().flatten().tolist() for s in range(S)]
    silent = ((y > 0.5).sum(1) == 0).nonzero().flatten().tolist()
    n_sil = min(sil_frames, len(silent), cap)
    for c in cand:
        rng.shuffle(c)
    picked = [[] for _ in range(S)]
    budget = cap - n_sil
    while budget > 0 and any(cand):
        for s in range(S):
            if cand[s] and budget > 0:
                picked[s].append(cand[s].pop())
                budget -= 1
    idx = [i for s in range(S) for i in sorted(picked[s])] + sorted(rng.sample(silent, n_sil))
    return torch.tensor(idx, dtype=torch.long)


def streaming_sim_loss(head, enc, elen, spk_targets, scfg: dict, rng: random.Random) -> torch.Tensor:
    """BCE of the head on simulated streaming steps [cache ‖ fifo ‖ chunk ‖ rc], global arrival order."""
    cfg = AOSCConfig.from_dict(scfg)
    tr = scfg.get("train", {})
    chunk_lens, rcs = tr.get("chunk_lens", [cfg.chunk_len]), tr.get("right_contexts", [cfg.chunk_right_context])
    k = tr.get("seqs_per_item", 2)
    B, T = enc.shape[:2]
    tgt = sort_by_arrival(_match_len(spk_targets.float(), T))
    tgt_cpu, lens = tgt.detach().cpu(), elen.tolist()  # index building on the CPU: no per-item syncs
    rows = []
    for b in range(B):
        Tb = int(lens[b])
        for _ in range(k):
            C, R = rng.choice(chunk_lens), rng.choice(rcs)
            if Tb <= C:
                continue
            s = rng.randrange(0, Tb - C + 1)
            fl = max(0, cfg.fifo_len - rng.randrange(max(1, cfg.spkcache_update_period)))
            f0 = max(0, s - fl)
            if f0 <= cfg.spkcache_len:
                cidx = torch.arange(f0)
            else:
                cidx = teacher_cache(tgt_cpu[b, :f0], cfg.spkcache_len, cfg.sil_frames, rng)
            rows.append(b * T + torch.cat([cidx, torch.arange(f0, min(Tb, s + C + R))]))
    L = max(len(r) for r in rows)
    n = torch.tensor([len(r) for r in rows])
    flat = torch.stack([F.pad(r, (0, L - len(r))) for r in rows]).to(enc.device)  # (M,L) into B*T
    valid = (torch.arange(L)[None] < n[:, None]).to(enc.device)
    emb = head.embed_frames(enc).reshape(B * T, -1)[flat]
    y = tgt.reshape(B * T, -1)[flat]
    logits = head.forward_emb(emb, valid).float()
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * valid[..., None]
    return l.sum() / (valid.sum() * y.shape[-1])


class _TrainWrapper(nn.Module):
    """SpeechModel.forward (offline sort+PIL diar loss, speaker-kernel ASR loss) + streaming-sim loss.
    Duplicates SpeechModel.forward's loop because that method does not expose the encoder output."""

    def __init__(self, model, scfg: dict, seed: int = 0):
        super().__init__()
        self.m, self.scfg = model, scfg
        self.tokenizer = model.tokenizer
        self.rng = random.Random(seed)
        self.w = scfg.get("train", {}).get("weight", 0.3)

    def forward(self, batch):
        m = self.m
        feats, flen = m.features(batch["audio"], batch["audio_len"])
        att = self.rng.choice(m.encoder.att_context_sizes)
        enc, elen = m.encoder(feats, flen, att)
        out, total = {}, 0.0
        for name, head in m.heads.items():
            hc = m.head_cfg[name]
            e = enc
            if hc.get("condition_on_speaker"):
                e, _ = m.encoder(feats, flen, att, spk_act=batch["spk_act"][:, : enc.shape[1]])
            l = head.loss(e, elen, batch)
            out[f"loss_{name}"] = l
            total = total + hc.get("weight", 1.0) * l
            if hc["type"] == "sortformer" and self.w > 0:
                ls = streaming_sim_loss(head, enc, elen, batch["spk_targets"], self.scfg, self.rng)
                out[f"loss_{name}_stream"] = ls
                total = total + self.w * ls
        out["loss"] = total
        return out


def train(recipe: str, overrides: list[str] | None = None, out: str | None = None):
    from .model import SpeechModel
    from .train import Trainer, build_tokenizer, load_recipe, save_model
    cfg = load_recipe(recipe, overrides)
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    cc = dict(cfg["data"].get("conversations", {}))
    n = cc.pop("n_train", 3000)
    t0 = time.time()
    train_set = conversation_dataset(n, seed=0, **cc)
    log.info(f"data: {len(train_set)} conversations in {time.time() - t0:.0f}s, "
          f"mean {np.mean([len(x['audio']) for x in train_set]) / 16000:.1f}s")
    tok = build_tokenizer(cfg, train_set)
    model = SpeechModel(cfg, tok)
    log.info(f"[{cfg.get('name')}] params={model.num_params() / 1e6:.2f}M heads={list(model.heads)}")
    wrapper = _TrainWrapper(model, cfg.get("streaming", {}))
    tr = Trainer(wrapper, cfg)
    t0 = time.time()
    tr.fit(train_set)
    log.info(f"train time {time.time() - t0:.0f}s")
    model.eval()
    if out:
        save_model(model.cpu(), out)
        log.info("%s %s", "saved", out)
    return model


# --------------------------------------------------------------------------- evaluation
def cp_wer(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    """Concatenated minimum-permutation WER counts (errors, ref words) over speaker slots."""
    S = max(len(refs), len(hyps))
    refs, hyps = refs + [""] * (S - len(refs)), hyps + [""] * (S - len(hyps))
    best = min(sum(edit_distance(refs[i].split(), hyps[p[i]].split()) for i in range(S))
               for p in itertools.permutations(range(S)))
    return best, sum(len(r.split()) for r in refs)


def slot_flips(pred: torch.Tensor, ref: torch.Tensor, chunk: int) -> tuple[int, int]:
    """Speaker-permutation consistency: per chunk, each active reference speaker is mapped to the
    predicted slot that overlaps it most; a flip = a speaker's slot differs from its slot in its
    previous active chunk. Returns (flips, transitions)."""
    last, flips, trans = {}, 0, 0
    for a in range(0, len(ref), chunk):
        p, r = pred[a: a + chunk], ref[a: a + chunk]
        for j in range(r.shape[1]):
            if r[:, j].sum() < 1:
                continue
            ov = (p * r[:, j: j + 1]).sum(0)
            if ov.max() <= 0:
                continue
            s = int(ov.argmax())
            if j in last:
                trans += 1
                flips += int(last[j] != s)
            last[j] = s
    return flips, trans


@torch.no_grad()
def evaluate(model, items: list[dict], asr: bool = True, push_ms: int = 160, **stream_kw) -> dict:
    """Offline vs streaming DER, slot-flip rate, cpWER (streaming ASR vs offline transcribe_speakers)."""
    dev = next(model.parameters()).device
    diar = next(k for k, v in model.head_cfg.items() if v["type"] == "sortformer")
    has_asr = asr and any(v.get("condition_on_speaker") for v in model.head_cfg.values())
    acc = dict(der_off=[], der_str=[], flips_off=0, flips_str=0, trans_off=0, trans_str=0,
               wer_off=[0, 0], wer_str=[0, 0], sec=0.0, t_str=0.0, t_diar=0.0, enc_frames=0, asr_frames=0)
    cfg = None
    for ex in items:
        x = ex["audio"]
        ref = torch.as_tensor(ex["spk_targets"]).float()
        xt = torch.as_tensor(x, device=dev)[None]
        enc, elen = model.encode(xt, torch.tensor([len(x)], device=dev))
        p_off = model.heads[diar](enc, elen).sigmoid()[0].cpu()
        T = min(len(ref), len(p_off))
        step = int(16000 * push_ms / 1000)
        t0 = time.time()
        d = StreamingDiarizer(model, diar, **stream_kw)
        for i in range(0, len(x), step):
            d.feed(x[i: i + step])
        d.feed([], final=True)
        acc["t_diar"] += time.time() - t0
        p_str = d.all_probs.cpu()
        assert len(p_str) == len(p_off), (len(p_str), len(p_off))
        cfg = d.cfg
        b_off, b_str = (p_off[:T] > 0.5).float(), (p_str[:T] > 0.5).float()
        acc["der_off"].append(frame_der(b_off, ref[:T]))
        acc["der_str"].append(frame_der(b_str, ref[:T]))
        for tag, b in (("off", b_off), ("str", b_str)):
            f, n = slot_flips(b, ref[:T], cfg.chunk_len)
            acc[f"flips_{tag}"] += f
            acc[f"trans_{tag}"] += n
        acc["sec"] += len(x) / 16000
        acc["enc_frames"] += d.enc_frames_computed
        if has_asr:
            refs = [s["text"] for s in ex["speakers"]]
            off = model.transcribe_speakers(x, diar_head=diar)
            e, w = cp_wer(refs, [o["text"] for o in off])
            acc["wer_off"][0] += e
            acc["wer_off"][1] += w
            t0 = time.time()
            sa = StreamingSpeakerASR(model, diar_head=diar, **stream_kw)
            for i in range(0, len(x), step):
                sa.feed(x[i: i + step])
            sa.feed([], final=True)
            acc["t_str"] += time.time() - t0
            acc["asr_frames"] += sa.asr_frames_computed
            e, w = cp_wer(refs, [t for t in sa.texts.values() if t])
            acc["wer_str"][0] += e
            acc["wer_str"][1] += w
    r = {
        "n": len(items), "mean_sec": round(acc["sec"] / len(items), 2),
        "latency_ms": cfg.latency_ms, "config": asdict(cfg), "mode": d.mode,
        "der_offline": round(float(np.mean(acc["der_off"])), 4),
        "der_streaming": round(float(np.mean(acc["der_str"])), 4),
        "slot_flip_rate_offline": round(acc["flips_off"] / max(1, acc["trans_off"]), 4),
        "slot_flip_rate_streaming": round(acc["flips_str"] / max(1, acc["trans_str"]), 4),
        "slot_transitions": acc["trans_str"],
        "rtf_diar": round(acc["t_diar"] / acc["sec"], 4),
        "enc_frames_per_sec_diar": round(acc["enc_frames"] / acc["sec"], 2),
    }
    if has_asr:
        r.update({"cpwer_offline": round(acc["wer_off"][0] / max(1, acc["wer_off"][1]), 4),
                  "cpwer_streaming": round(acc["wer_str"][0] / max(1, acc["wer_str"][1]), 4),
                  "rtf_diar_plus_asr": round(acc["t_str"] / acc["sec"], 4),
                  "asr_enc_frames_per_sec": round(acc["asr_frames"] / acc["sec"], 2)})
    return r


def eval_sets(n_mix: int = 64, n_long: int = 32):
    mix = synthetic_dataset("diar", n_mix, seed=2)
    return {"mixtures_seed2": mix, "long_conversations": long_conversations(n_long, seed=3)}


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="audioforge.streaming_diar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("recipe")
    t.add_argument("overrides", nargs="*")
    t.add_argument("-o", "--out")
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--n-mix", type=int, default=64)
    e.add_argument("--n-long", type=int, default=32)
    e.add_argument("--json")
    e.add_argument("--no-asr", action="store_true")
    e.add_argument("--set", action="append", default=[], help="streaming override key=value (repeatable)")
    e.add_argument("--configs", default="default", help="comma list: default,nocache,lowlat,offline_chunk")
    d = sub.add_parser("demo")
    d.add_argument("model")
    d.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    if a.cmd == "train":
        train(a.recipe, a.overrides, a.out)
        return
    from .train import load_model
    model = load_model(a.model, "cpu")
    if a.cmd == "demo":
        conv = long_conversations(1, seed=a.seed, n_segments=4)[0]
        print("reference:", [(i, s["text"]) for i, s in enumerate(conv["speakers"])])
        sa = StreamingSpeakerASR(model)
        x = conv["audio"]
        for i in range(0, len(x), 2560):
            sa.feed(x[i: i + 2560])
        sa.feed([], final=True)
        for t_, s, txt in sa.events:
            print(f"{t_:6.2f}s  spk{s}: {txt}")
        print("final:", sa.transcript())
        return
    base = dict(kv.split("=", 1) for kv in a.set)
    base = {k: int(v) if v.lstrip("-").isdigit() else float(v) for k, v in base.items()}
    presets = {"default": {}, "nocache": {"spkcache_len": 0}, "lowlat": {"chunk_len": 2, "chunk_right_context": 1},
               "bigfifo_nocache": {"spkcache_len": 0, "fifo_len": 60}}
    sets = eval_sets(a.n_mix, a.n_long)
    res = {}
    for c in a.configs.split(","):
        for name, items in sets.items():
            kw = {**base, **presets[c]}
            t0 = time.time()
            r = evaluate(model, items, asr=not a.no_asr, **kw)
            r["eval_sec"] = round(time.time() - t0, 1)
            res[f"{c}/{name}"] = r
            print(c, name, json.dumps(r), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
