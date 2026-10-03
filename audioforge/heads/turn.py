"""TurnHead (v2 core, v3 duration/activity inputs, v5 mode concat): speaker- and text-aware end-of-turn (EOT) on the shared encoder.

Per-frame logit "the PRIMARY speaker's turn is over" from up to three inputs:

(a) acoustics - the encoder features;
(b) text state (``use_text: true``) - the ASR transducer's PredictionNet state after the tokens known
    at that frame, plus the embeddings of the last ``k_tokens`` tokens: a running summary of the
    transcript ("is the sentence open? did it end in 'uh' or a cut-off word?"). The PredictionNet is
    the ASR head's own (``text_head``, default the first rnnt/tdt head), run without gradient so the
    turn loss never moves the ASR decoder. Training: teacher-forced on ``batch["text"]``, with each
    token placed at a frame by a monotonic alignment proxy - greedy forced alignment through the ASR
    head's joint (walk the (t, u) lattice of the reference: emit y_u at frame t when its logit beats
    blank, else advance t), and uniform spreading of the tokens over the primary's speech frames for
    items where greedy fails (incomplete, or ends before the speech does - e.g. an untrained ASR).
    Inference: the greedy-decoded prefix, each token available from the frame that emitted it.
(c) the primary speaker's activity ``spk_act`` (diarization + enrollment at inference, oracle in
    training):
    * ``mode: kernel`` - the encoder itself is steered by the primary's activity through the speaker
      kernels (set ``condition_on_speaker: true`` on the head; SpeechModel.forward then feeds this head
      the speaker-conditioned encoding).
    * ``mode: concat`` - the activity and its last ``history`` frames are concatenated to the features.
    * ``mode: kernel+concat`` - both. ``mode: none`` - no speaker information.

Ablation grid: mode none|kernel x use_text false|true = {acoustic, +speaker, +text, +speaker+text}.

The temporal model is a unidirectional GRU, so the head is causal and streams frame by frame
(``init_stream`` / ``step``); frame t's text state only uses tokens emitted at frames <= t. Loss: BCE
with ``pos_weight`` over valid frames; targets = frames after the primary's last active frame
(``eot_targets``), so the stock ``data.Collate`` batch (``spk_act``, ``text``) is enough.

v3 inputs (n=200: with a real streaming diarizer the head misses 69 % vs 38 % for a silence
timeout on the same track). All default OFF: a head without them has exactly the v2 modules, state_dict and outputs.
  * ``act_columns: S`` (> 1) - the head also reads the diarizer's full (T, S) track ``cols`` plus a one-hot of the
    primary's column ``prim`` (the enrollment rule, datasets/ext_tracks.enroll_column), so a next speaker who lands
    in the primary's slot is visible in the other columns. Training: SpeechModel.conditioning_act puts the
    conditioning track's columns in ``batch["spk_cols"]`` / ``batch["spk_prim"]`` (``spk_act`` = its primary
    column); without them the oracle ``spk_targets`` (column 0 = primary) is used.
  * ``duration_feats: true`` - causal per-frame counters computed from the (noisy) INPUT tracks, never from labels:
    frames since the primary column was last active and its current active run, the same for "any non-primary
    column" (act_columns > 1) and for the track fed to the encoder kernels (``spk_act``); log-scaled, clipped at
    ``dur_max`` frames (``duration_features``). Lets the head emulate (and refine) a silence timeout.
  * ``future_act_aux: {horizons: [6, 12, 25], weight: 0.3}`` - VAP-style auxiliary logits on the GRU state:
    "the primary / any other speaker is active within the next h frames" (labels from the oracle activity,
    ``future_act_targets``); training only, added to the loss with ``weight``.
  The new inputs enter through a separate zero-initialised projection added to the v2 input layer, so a v3 head
  initialised from a v2 checkpoint starts exactly as the v2 head (init.from loads the old tensors unchanged).

Dyadic additions (both default OFF, a head without them is unchanged):
  * ``energy_input: true`` (or {tau_frames: 125, clip: 4.0}) - one causal scalar per 80 ms frame from the RAW audio
    the model hears: log-RMS of the frame's 1280 samples [1280 t, 1280 (t + 1)), standardised by a causal running
    mean / variance (EMA with alpha_t = max(1 / (t + 1), 1 / tau_frames): the cumulative mean for the first
    tau_frames frames, then a ~10 s exponential window), clipped to +-clip and halved (``energy_features``). Enters
    through its own zero-initialised projection added to the input layer (= a column concatenated to the head input),
    so a warm start begins exactly as the source head. Streaming: ``step(..., energy_state)`` carries (mean, var, t).
  * ``multi_horizon_aux: {horizons_ms: [200, 400, 600, 1000], weight: 0.3}`` - auxiliary logits on the GRU state for
    the PRIMARY (user) party only: "the user is active somewhere in bin j" with bins between consecutive horizons,
    (t, t + e1], (t + e1, t + e2], ... where e_j = ceil(h_j / 80 ms) frames (3 / 5 / 8 / 13 = 240 / 400 / 640 /
    1040 ms; ``multi_horizon_targets``). Relation to ``future_act_aux``: that one predicts CUMULATIVE windows
    (t, t + h] at 480 / 960 / 2000 ms for the primary and for "any other speaker"; this one predicts disjoint,
    finer bins inside the first second for the user alone (the OR of all bins = the user speaks within 1040 ms, the
    quantity an EOT decision hinges on), a VAP-style projection of the user's near future. Training only.
    A recipe that switches to it keeps ``future_act_aux`` with ``weight: 0`` so the warm-start tensors still load;
    weight 0 skips that loss entirely.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .asr import Head
from .audio import _match_len, _pad_mask, arrival_order

log = logging.getLogger(__name__)

MODES = ("none", "concat", "kernel", "kernel+concat")
ALIGNS = ("greedy", "uniform")


def eot_targets(act: torch.Tensor, act_len: torch.Tensor | None = None) -> torch.Tensor:
    """(B,T) activity -> (B,T) float, 1 from the frame after the last active frame on."""
    B, T = act.shape
    ar = torch.arange(T, device=act.device)[None].expand(B, T)
    active = act > 0.5
    if act_len is not None:
        active = active & (ar < act_len[:, None])
    end = torch.where(active, ar, torch.full_like(ar, -1)).max(1).values + 1  # 0 if never active
    return (ar >= end[:, None]).float()


def act_history(act: torch.Tensor, T: int, history: int) -> torch.Tensor:
    """(B,T0) -> (B,T,history): [a_t, a_{t-1}, ..., a_{t-history+1}] (zeros before the start)."""
    a = _match_len(act.float(), T)
    return torch.stack([F.pad(a, (k, 0))[:, :T] for k in range(history)], -1)


# --------------------------------------------------------------------------- v3: duration counters, future labels
NEVER = 10 ** 6  # "frames since last active" of a track that has not been active yet (clipped to dur_max anyway)


def duration_counters(active: torch.Tensor, since0: torch.Tensor | None = None, run0: torch.Tensor | None = None):
    """Causal counters of a (B,n) bool activity track -> (since, run), both (B,n) long.

    since[t] = frames since the track was last active (0 on an active frame; NEVER + t + 1 if never active);
    run[t]   = length of the active run ending at t (0 on an inactive frame).
    since0 / run0 (B,): the values at the frame before this block (streaming state; default: never active, no
    run), so running block by block gives exactly the counters of the whole sequence.
    Computed on the CPU (int64 cummax; B x n is tiny) and returned on ``active``'s device (MPS-safe)."""
    B, n = active.shape
    dev, active = active.device, active.detach().cpu()
    since0 = torch.full((B,), NEVER, dtype=torch.long) if since0 is None else since0.detach().cpu().long()
    run0 = torch.zeros(B, dtype=torch.long) if run0 is None else run0.detach().cpu().long()
    ar = torch.arange(n)[None].expand(B, n)
    last_on = torch.where(active, ar, (-1 - since0)[:, None].expand(B, n)).cummax(1).values
    last_off = torch.where(~active, ar, (-1 - run0)[:, None].expand(B, n)).cummax(1).values
    return (ar - last_on).to(dev), (ar - last_off).to(dev)


def log_clip(x: torch.Tensor, dur_max: int) -> torch.Tensor:
    """Counter -> [0, 1]: log1p(min(x, dur_max)) / log1p(dur_max)."""
    return torch.log1p(x.clamp(max=dur_max).float()) / math.log1p(dur_max)


def future_act_targets(act: torch.Tensor, lengths: torch.Tensor | None, horizons) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T) activity -> targets (B,T,H): any(act[t+1 .. t+h] > 0.5) per horizon h, and a (B,T,H) validity mask:
    a frame is labelled when its whole future window lies inside the item's labelled frames or a positive
    was already found inside them (the future beyond the window end is unknown)."""
    B, T = act.shape
    L = torch.full((B,), T, device=act.device) if lengths is None else lengths.to(act.device).clamp(max=T)
    ar = torch.arange(T, device=act.device)[None]
    a = ((act > 0.5) & (ar < L[:, None])).float()
    c = F.pad(a.cumsum(1), (1, 0))  # c[:, i] = active frames in [0, i)
    tg, ok = [], []
    for h in horizons:
        hi = (ar + 1 + h).clamp(max=T).expand(B, T)
        lo = (ar + 1).clamp(max=T).expand(B, T)
        pos = (c.gather(1, hi) - c.gather(1, lo)) > 0
        tg.append(pos.float())
        ok.append((((ar + h) < L[:, None]) | pos) & (ar < L[:, None]))
    return torch.stack(tg, -1), torch.stack(ok, -1)


FRAME_SAMPLES = 1280  # 80 ms at 16 kHz: the encoder frame grid
ENERGY_FLOOR = 1e-5  # RMS floor (digital silence) before the log


def frame_log_rms(audio: torch.Tensor, audio_len: torch.Tensor | None, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,n) 16 kHz audio -> log RMS (B,T) of samples [1280 t, 1280 (t + 1)) and a (B,T) bool mask of frames with at
    least one valid sample (frames past the audio are 0 / False). Frame t uses samples up to its own end only."""
    B, n = audio.shape
    L = torch.full((B,), n, device=audio.device) if audio_len is None else audio_len.to(audio.device).clamp(max=n)
    need = T * FRAME_SAMPLES
    x = audio.float()
    x = F.pad(x, (0, need - n)) if n < need else x[:, :need]
    ar = torch.arange(need, device=audio.device)[None]
    m = (ar < L[:, None]).float()
    x2 = (x * m) ** 2
    cnt = m.reshape(B, T, FRAME_SAMPLES).sum(-1)
    ms = x2.reshape(B, T, FRAME_SAMPLES).sum(-1) / cnt.clamp(min=1)
    return torch.log(ms.sqrt().clamp(min=ENERGY_FLOOR)) * (cnt > 0), cnt > 0


def causal_standardize(x: torch.Tensor, valid: torch.Tensor, tau: int = 125, clip: float = 4.0,
                       state: tuple | None = None) -> tuple[torch.Tensor, tuple]:
    """Causal running standardisation of (B,T) values: m_t, v_t = EMA mean / variance with alpha_t = max(1/(k+1),
    1/tau) (k = valid frames seen before), updated with x_t, then z_t = clip((x_t - m_t) / sqrt(v_t + 1e-2)) / 2.
    ``state`` = (m, v, k) (B,) each from a previous block -> identical to one pass over the concatenation.
    Invalid frames give 0 and do not update the statistics."""
    B, T = x.shape
    if state is None:
        m = x.new_zeros(B)
        v = x.new_zeros(B)
        k = x.new_zeros(B)
    else:
        m, v, k = (t.to(x.device, x.dtype).clone() for t in state)
    out = []
    for t in range(T):
        ok = valid[:, t].to(x.dtype)
        a = torch.maximum(1.0 / (k + 1), torch.full_like(k, 1.0 / tau))
        d = x[:, t] - m
        m_new = m + a * d
        v_new = (1 - a) * (v + a * d * d)
        m = torch.where(ok > 0, m_new, m)
        v = torch.where(ok > 0, v_new, v)
        k = k + ok
        z = ((x[:, t] - m) / torch.sqrt(v + 1e-2)).clamp(-clip, clip) * 0.5
        out.append(z * ok)
    return torch.stack(out, 1) if out else x.new_zeros(B, 0), (m, v, k)


def energy_features(audio: torch.Tensor, audio_len: torch.Tensor | None, T: int, tau: int = 125, clip: float = 4.0,
                    state: tuple | None = None) -> tuple[torch.Tensor, tuple]:
    """(B,n) raw audio -> (B,T) causal standardised log-RMS per 80 ms frame (see the module docstring)."""
    x, valid = frame_log_rms(audio, audio_len, T)
    return causal_standardize(x, valid, tau, clip, state)


def horizon_edges(horizons_ms, frame_ms: float = 80.0) -> list[int]:
    """Horizons in ms -> bin edges in frames, ceil(h / frame_ms), strictly increasing."""
    e = [int(math.ceil(float(h) / frame_ms - 1e-9)) for h in horizons_ms]
    assert all(b > a for a, b in zip([0] + e, e)), f"horizons must give increasing frame edges: {horizons_ms} -> {e}"
    return e


def multi_horizon_targets(act: torch.Tensor, lengths: torch.Tensor | None, edges) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T) activity -> targets (B,T,H): any(act[t + e_{j-1} + 1 .. t + e_j] > 0.5) for bins between consecutive
    edges (e_0 = 0), and a (B,T,H) validity mask (the whole bin lies inside the labelled frames, or a positive was
    already found inside them)."""
    B, T = act.shape
    L = torch.full((B,), T, device=act.device) if lengths is None else lengths.to(act.device).clamp(max=T)
    ar = torch.arange(T, device=act.device)[None]
    a = ((act > 0.5) & (ar < L[:, None])).float()
    c = F.pad(a.cumsum(1), (1, 0))  # c[:, i] = active frames in [0, i)
    tg, ok = [], []
    lo_e = 0
    for e in edges:
        lo = (ar + 1 + lo_e).clamp(max=T).expand(B, T)
        hi = (ar + 1 + e).clamp(max=T).expand(B, T)
        pos = (c.gather(1, hi) - c.gather(1, lo)) > 0
        tg.append(pos.float())
        ok.append((((ar + e) < L[:, None]) | pos) & (ar < L[:, None]))
        lo_e = e
    return torch.stack(tg, -1), torch.stack(ok, -1)


# --------------------------------------------------------------------------- token -> frame alignment
# Byte budget for one item's alignment lattice temporaries (the ReLU'd joint hidden h and the h * W_y product,
# each (T_b, U_chunk, D_joint) fp32). Items above it are processed in chunks over U. One AMI turn window
# (20 s, T=250) with a whole-turn transcript of ~1.6k tokens needs 2 x 1.04 GB unchunked.
ALIGN_MAX_BYTES = 256 * 2 ** 20


def align_chunk(Tb: int, Ub: int, d_joint: int, max_bytes: int | None = None, elem: int = 4) -> int:
    """Largest number of lattice columns u per chunk so that the two (Tb, chunk, d_joint) temporaries of
    greedy_align stay within ``max_bytes`` (default ALIGN_MAX_BYTES); at least 1, at most Ub."""
    budget = ALIGN_MAX_BYTES if max_bytes is None else max_bytes
    return max(1, min(Ub, budget // max(1, 2 * Tb * d_joint * elem)))


@torch.no_grad()
def greedy_align(asr, enc, enc_len, y, yl, act=None, max_per_frame: int = 4, slack: int = 3,
                 min_inside: float = 0.8, max_bytes: int | None = None):
    """Greedy forced alignment of the reference ``y`` through a transducer head's joint.

    Walks the (t, u) lattice from (0, 0): emit y_u at frame t if logit(y_u) > logit(blank) at (t, u)
    (and fewer than ``max_per_frame`` tokens were emitted at t), else t += 1. TDT durations are ignored
    (blank = advance one frame). -> emit (B,U) frame of each token (-1 = not placed), ok (B,) bool:
    all tokens placed and, if ``act`` is given, a plausible path: the first token no earlier than ``slack``
    frames before the primary's onset, the last no earlier than ``slack`` frames before its last active
    frame, and >= ``min_inside`` of the tokens within ``slack`` frames of its speech (rejects the paths an
    untrained or confused joint produces; the caller falls back to uniform spreading).

    Memory: only the two logits the walk compares (label y_u and blank) are computed - never the full
    V-way joint - one item at a time, cropped to (T_b, U_b), in chunks over U so that each item's
    (T_b, chunk, D_joint) temporaries stay within ``max_bytes`` (``align_chunk``). Identical results."""
    B, T, _ = enc.shape
    U = y.shape[1]
    g, _ = asr.pred(asr.pred.prepend_sos(y))
    fe, gp, lin = asr.joint.enc(enc), asr.joint.pred(g), asr.joint.out[-1]
    emit = torch.full((B, U), -1, dtype=torch.long, device=enc.device)
    ok = torch.zeros(B, dtype=torch.bool)
    for b in range(B):
        Tb, Ub = int(enc_len[b]), int(yl[b])
        if Ub == 0:
            ok[b] = True
            continue
        uc = align_chunk(Tb, Ub, fe.shape[-1], max_bytes, fe.element_size())
        go_t = torch.empty(Tb, Ub, dtype=torch.bool, device=enc.device)
        for u0 in range(0, Ub, uc):
            u1 = min(Ub, u0 + uc)
            h = torch.relu(fe[b, :Tb, None] + gp[b, None, u0:u1])  # Tb, chunk, Dj
            blank = h @ lin.weight[asr.blank] + lin.bias[asr.blank]
            lab = (h * lin.weight[y[b, u0:u1]][None]).sum(-1) + lin.bias[y[b, u0:u1]]
            go_t[:, u0:u1] = lab > blank
            del h, blank, lab
        go = go_t.tolist()
        t = u = per = 0
        e = []
        while t < Tb and u < Ub:
            if go[t][u] and per < max_per_frame:
                e.append(t)
                u, per = u + 1, per + 1
            else:
                t, per = t + 1, 0
        if u < Ub:
            continue
        emit[b, :Ub] = torch.tensor(e, device=enc.device)
        ok[b] = True
        if act is not None and (act[b, :Tb] > 0.5).any():
            sp = (act[b, :Tb] > 0.5).float()
            nz = torch.nonzero(sp)[:, 0]
            near = F.max_pool1d(sp[None, None], 2 * slack + 1, 1, slack)[0, 0] > 0  # speech +- slack frames
            inside = near[emit[b, :Ub]].float().mean()
            ok[b] = bool(e[0] >= int(nz[0]) - slack and e[-1] >= int(nz[-1]) - slack and inside >= min_inside)
    return emit, ok


def uniform_align(yl, T: int, enc_len, act=None, U: int | None = None):
    """Spread each item's tokens evenly over its speech frames (act > 0.5; all valid frames if no
    activity): token u is placed at the frame where the (u+1)/U share of speech frames is complete."""
    B = len(yl)
    U = int(yl.max()) if U is None else U
    emit = torch.full((B, U), -1, dtype=torch.long, device=yl.device)
    for b in range(B):
        Tb, Ub = min(int(enc_len[b]), T), int(yl[b])
        fr = torch.nonzero(act[b, :Tb] > 0.5)[:, 0] if act is not None else torch.zeros(0, dtype=torch.long)
        if len(fr) == 0:
            fr = torch.arange(Tb, device=yl.device)
        S = len(fr)
        for u in range(Ub):
            emit[b, u] = fr[max(0, math.ceil((u + 1) * S / Ub) - 1)]
    return emit


def token_counts(emit: torch.Tensor, yl: torch.Tensor, T: int) -> torch.Tensor:
    """(B,U) emission frames -> (B,T) number of tokens available at each frame (emitted at <= t)."""
    U = emit.shape[1]
    valid = (emit >= 0) & (torch.arange(U, device=emit.device)[None] < yl[:, None])
    ar = torch.arange(T, device=emit.device)
    return ((emit[:, None, :] <= ar[None, :, None]) & valid[:, None, :]).sum(-1)


@torch.no_grad()
def greedy_decode_frames(asr, f: torch.Tensor) -> tuple[list[int], list[int]]:
    """RNNTHead._greedy that also returns the frame each token was emitted at (inference text path)."""
    V1, dev = asr.vocab_size + 1, f.device
    hyp, frames = [], []
    g, state = asr.pred(torch.tensor([[asr.blank]], device=dev), None)
    fe = asr.joint.enc(f)
    t, T, emitted = 0, f.shape[0], 0
    while t < T:
        z = asr.joint.out(fe[t][None, None] + asr.joint.pred(g))[0, 0]
        k = int(z[:V1].argmax())
        if asr.is_tdt:
            d = asr.durations[int(z[V1:].argmax())]
            if k == asr.blank and d == 0:
                d = 1
        else:
            d = 1 if k == asr.blank else 0
        if k != asr.blank:
            hyp.append(k)
            frames.append(t)
            g, state = asr.pred(torch.tensor([[k]], device=dev), state)
            emitted += 1
        if d == 0 and emitted >= asr.max_symbols:
            d = 1
        if d > 0:
            emitted = 0
        t += d
    return hyp, frames


def pad_tokens(seqs: list[list[int]], fill: int = 0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(1, max(map(len, seqs)))
    y = torch.full((len(seqs), L), fill, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        y[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return y, torch.tensor([len(s) for s in seqs], device=device)


# --------------------------------------------------------------------------- head
@dataclass
class TurnStreamState:
    h: torch.Tensor | None = None  # GRU hidden (n_layers, B, hidden)
    act_buf: torch.Tensor | None = None  # (B, history - 1) past primary activity, oldest first
    dur: dict | None = None  # v3 duration_feats: {track: (since, run)} counters at the last frame fed
    energy: tuple | None = None  # energy_input: (mean, var, frames seen) of the causal standardisation


class TurnHead(Head):
    key = "spk_act"  # the batch field it needs (conditioning input and the source of its labels)

    # Sizes (shipped on both cores: hidden 96, n_layers 1, history 8, k_tokens 4, text_dim 64):
    # placeholder: never swept
    def __init__(self, d_model: int, mode: str = "concat", hidden: int = 96, n_layers: int = 1,
                 history: int = 8, pos_weight: float = 2.0, dropout: float = 0.1, use_text: bool = False,
                 text_head: str | None = None, k_tokens: int = 4, text_dim: int = 64, align: str = "greedy",
                 max_per_frame: int = 4, text_delay: int = 0, text_noise: float = 0.0, decoded_prob: float = 0.0,
                 act_columns: int = 1, duration_feats: bool = False, dur_max: int = 64,
                 future_act_aux: dict | None = None, init_partial: bool = False, energy_input=False,
                 multi_horizon_aux: dict | None = None):
        super().__init__()
        # init.from warm start into a differently shaped head (v5: mode concat, hidden 128 over a v2 kernel head):
        # tensors whose shape differs from the checkpoint's start fresh instead of failing load_state_dict
        self.init_partial = bool(init_partial)
        assert mode in MODES, f"mode must be one of {MODES}"
        assert align in ALIGNS, f"align must be one of {ALIGNS}"
        self.mode, self.history, self.pos_weight = mode, history, pos_weight
        self.concat = "concat" in mode
        self.use_text, self.text_head, self.k_tokens, self.text_dim = use_text, text_head, k_tokens, text_dim
        self.align_mode, self.max_per_frame = align, max_per_frame
        # train-time text augmentation: at inference the greedy-decoded tokens
        # arrive ~1 frame after the aligned reference (26-33 % of turn-final tokens >= 2 frames late) and carry
        # ~24 % WER. text_delay: shift each item's token frames by d ~ U{0..text_delay}; text_noise: substitute
        # each token with this probability; decoded_prob: per step, train on the ASR head's own greedy decode
        # (tokens + emission frames, exactly the inference text path) instead of the aligned reference.
        self.text_delay, self.text_noise, self.decoded_prob = text_delay, text_noise, decoded_prob
        self.align_counts = {"greedy": 0, "uniform": 0, "decoded": 0}  # items per text source (diagnostics)
        d_in = d_model + (history if self.concat else 0) + (text_dim if use_text else 0)
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, 1)
        # v3 inputs (module docstring); nothing below exists when they are off (v2 state_dict unchanged)
        self.act_columns, self.duration_feats, self.dur_max = int(act_columns), bool(duration_feats), int(dur_max)
        assert self.act_columns >= 1, "act_columns must be >= 1 (1 = the single primary track, v2)"
        fa = dict(future_act_aux or {})
        self.fut_horizons = [int(h) for h in fa.get("horizons", [])] if fa.get("enabled", True) else []
        self.fut_weight = float(fa.get("weight", 0.3))
        self.n_dur = (6 if self.act_columns > 1 else 2) if self.duration_feats else 0
        d_v3 = (2 * self.act_columns if self.act_columns > 1 else 0) + self.n_dur
        self.v3_in = None
        if d_v3:
            self.v3_in = nn.Linear(d_v3, hidden, bias=False)
            nn.init.zeros_(self.v3_in.weight)  # starts as the v2 head (warm start from a v2 checkpoint)
        self.fut = nn.Linear(hidden, 2 * len(self.fut_horizons)) if self.fut_horizons else None
        # dyadic additions (module docstring): causal log-RMS energy input, multi-horizon user-activity auxiliary
        ei = energy_input if isinstance(energy_input, dict) else ({"enabled": True} if energy_input else {})
        self.energy_tau, self.energy_clip = int(ei.get("tau_frames", 125)), float(ei.get("clip", 4.0))
        self.energy_in = None
        if ei and ei.get("enabled", True):
            self.energy_in = nn.Linear(1, hidden, bias=False)
            nn.init.zeros_(self.energy_in.weight)  # starts as the source head (warm start)
        mh = dict(multi_horizon_aux or {})
        self.mh_edges = horizon_edges(mh.get("horizons_ms", [])) if mh and mh.get("enabled", True) else []
        self.mh_weight = float(mh.get("weight", 0.3))
        self.mh = nn.Linear(hidden, len(self.mh_edges)) if self.mh_edges else None
        self.text_proj = None
        object.__setattr__(self, "_asr", None)  # not a submodule: the ASR head is owned by the model
        object.__setattr__(self, "_model", None)  # set by bind(): re-encodes for the ASR head's view
        self._asr_view_differs = False  # ASR head speaker-conditioned but this head not (or vice versa)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        if self.init_partial:  # drop mismatched tensors before the children load them (they stay freshly initialised)
            own = self.state_dict()
            bad = [k for k in list(state_dict) if k.startswith(prefix) and k[len(prefix):] in own
                   and tuple(state_dict[k].shape) != tuple(own[k[len(prefix):]].shape)]
            for k in bad:
                del state_dict[k]
            if bad:
                log.info(f"[turn init_partial] {len(bad)} tensors with a new shape start fresh: "
                      f"{[k[len(prefix):] for k in bad][:6]}")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    # ------------------------------------------------------------------ text state
    def bind(self, model):
        """Called by SpeechModel.__init__: attach the ASR head whose PredictionNet gives the text state."""
        if not self.use_text:
            return
        name = self.text_head or next((k for k, v in model.head_cfg.items() if v["type"] in ("rnnt", "tdt")), None)
        if name is None or not hasattr(model.heads[name], "pred"):
            raise ValueError("TurnHead(use_text=true) needs a transducer (rnnt/tdt) head for its PredictionNet")
        self.text_head = name
        self.bind_asr(model.heads[name])
        me = next((k for k, h in model.heads.items() if h is self), None)
        if me is not None:
            object.__setattr__(self, "_model", model)
            self._asr_view_differs = bool(model.head_cfg[name].get("condition_on_speaker")) != bool(
                model.head_cfg[me].get("condition_on_speaker"))

    def bind_asr(self, asr):
        object.__setattr__(self, "_asr", asr)
        if self.text_proj is None:
            H = asr.pred.embed.embedding_dim
            self.text_proj = nn.Sequential(nn.Linear(H * (1 + self.k_tokens), self.text_dim), nn.SiLU())

    def _text_feats(self, g: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """g (B,T,H) PredictionNet state, last (B,T,k) token ids (most recent first, blank = none)."""
        e = self._asr.pred.embed(last).flatten(2)  # B,T,k*H
        return self.text_proj(torch.cat([g, e], -1).detach())

    def text_frames(self, y: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Teacher-forced text state per frame. y (B,U) tokens, n (B,T) tokens known at each frame
        (0..U, non-decreasing). Frame t depends on y[:, :n_t] only."""
        pred, B, T = self._asr.pred, *n.shape
        with torch.no_grad():
            g, _ = pred(pred.prepend_sos(y))  # (B,U+1,H): g[:, u] = state after u tokens
        gt = g.gather(1, n[..., None].expand(B, T, g.shape[-1]))
        ypad = pred.prepend_sos(y)  # ypad[:, i] = token i-1, ypad[:, 0] = blank
        last = torch.stack([ypad.gather(1, (n - j).clamp(min=0)) for j in range(self.k_tokens)], -1)
        return self._text_feats(gt, last)

    @torch.no_grad()
    def asr_view(self, enc, batch) -> torch.Tensor:
        """The encoding the ASR head reads (for the forced alignment / decoded text in training). Equals
        ``enc`` unless the ASR head and this head differ in speaker conditioning (turn_text arm: TDT is
        conditioned, the turn head is not -> aligning on ``enc`` failed for 99.8 % of items). Uses
        ``batch["_asr_enc"]`` when the model provides it, else re-encodes (no grad, no SpecAugment)."""
        if not self._asr_view_differs or self._model is None:
            return enc
        if "_asr_enc" in batch:
            return batch["_asr_enc"]
        m = self._model
        cond = bool(m.head_cfg[self.text_head].get("condition_on_speaker"))
        feats, flen = m.preprocessor(batch["audio"], batch["audio_len"])
        att = m.encoder.att_context_size
        spk = batch["spk_act"][:, : enc.shape[1]].float() if cond else None
        e, _ = m.encoder(feats, flen, att, spk_act=spk)
        return e[:, : enc.shape[1]]

    def augment_text(self, y, yl, emit, enc_len):
        """Train-time only: per-item delay d ~ U{0..text_delay} of the token frames (clamped to the valid
        frames, monotonic) and token substitution with probability text_noise. -> (y, emit)."""
        if not self.training or (self.text_delay <= 0 and self.text_noise <= 0):
            return y, emit
        B, U = y.shape
        valid = (emit >= 0) & (torch.arange(U, device=y.device)[None] < yl[:, None])
        if self.text_delay > 0:
            d = torch.randint(0, self.text_delay + 1, (B, 1), device=emit.device)
            emit = torch.where(valid, torch.minimum(emit + d, (enc_len[:, None] - 1).to(emit)), emit)
        if self.text_noise > 0:
            sub = valid & (torch.rand(B, U, device=y.device) < self.text_noise)
            y = torch.where(sub, torch.randint(0, self._asr.vocab_size, (B, U), device=y.device), y)
        return y, emit

    def decoded_text(self, enc, enc_len):
        """The ASR head's greedy decode on ``enc`` (its own view) -> y (B,L), yl (B,), emit (B,L)."""
        hyps, frames = zip(*[greedy_decode_frames(self._asr, enc[b, : int(enc_len[b])].detach())
                             for b in range(enc.shape[0])])
        y, yl = pad_tokens(list(hyps), device=enc.device)
        emit, _ = pad_tokens(list(frames), fill=-1, device=enc.device)
        return y, yl, emit

    def align(self, enc, enc_len, y, yl, act=None) -> torch.Tensor:
        """Training alignment proxy -> (B,U) emission frames (greedy through the ASR joint, uniform
        over the primary's speech frames where greedy fails or ``align: uniform``)."""
        T = enc.shape[1]
        act = _match_len(act.float(), T) if act is not None else None
        uni = uniform_align(yl, T, enc_len, act, U=y.shape[1])
        if self.align_mode == "uniform":
            self.align_counts["uniform"] += len(yl)
            return uni
        emit, ok = greedy_align(self._asr, enc.detach(), enc_len, y, yl, act, self.max_per_frame)
        ok = ok.to(emit.device)
        self.align_counts["greedy"] += int(ok.sum())
        self.align_counts["uniform"] += int((~ok).sum())
        return torch.where(ok[:, None], emit, uni)

    # ------------------------------------------------------------------ v3 inputs
    @property
    def needs_act(self) -> bool:
        """The head itself reads ``spk_act`` (concat history or duration counters of the kernel track)."""
        return self.concat or (self.duration_feats and self.act_columns == 1)

    @property
    def needs_cols(self) -> bool:
        return self.act_columns > 1

    def _cols(self, cols, prim, T: int):
        """-> cols (B,T,S) in [0,1] (frames / columns cropped or zero-padded), prim (B,) long."""
        S = self.act_columns
        c = _match_len(cols.float(), T).clamp(0, 1)
        if c.shape[2] != S:
            c = c[..., :S] if c.shape[2] > S else F.pad(c, (0, S - c.shape[2]))
        prim = torch.as_tensor(prim, device=c.device)
        if prim.dim() == 2:  # one-hot / scores
            prim = prim.argmax(1)
        return c, prim.long().reshape(-1).clamp(0, S - 1)

    def v3_features(self, T: int, spk_act=None, cols=None, prim=None, state: dict | None = None):
        """(B,T,d_v3) extra per-frame inputs [cols, one-hot(prim), duration features] and the counter state after
        the last frame. ``state`` (streaming): the counters before this block. None if the v3 inputs are off."""
        if self.v3_in is None:
            return None, state
        x, new = [], {}
        if self.act_columns > 1:
            if cols is None or prim is None:
                raise ValueError(f"TurnHead(act_columns={self.act_columns}) needs cols (B,T,S) and prim (B,)")
            c, pr = self._cols(cols, prim, T)
            oh = (torch.arange(self.act_columns, device=c.device)[None] == pr[:, None]).to(c)[:, None].expand(-1, T, -1)
            x += [c, oh]
            p_col = c.gather(2, pr[:, None, None].expand(-1, T, 1))[..., 0]
            other = (c * (1 - oh)).max(-1).values
            kern = _match_len(spk_act.float(), T) if spk_act is not None else p_col
            tracks = {"prim": p_col, "other": other, "kernel": kern}
        else:
            if spk_act is None:
                raise ValueError("TurnHead(duration_feats=true) needs spk_act (the primary's activity)")
            tracks = {"kernel": _match_len(spk_act.float(), T)}
        if self.duration_feats:
            for k, a in tracks.items():
                s0, r0 = (state or {}).get(k, (None, None))
                since, run = duration_counters(a > 0.5, s0, r0)
                x += [log_clip(since, self.dur_max)[..., None], log_clip(run, self.dur_max)[..., None]]
                new[k] = (since[:, -1].clamp(max=NEVER), run[:, -1].clamp(max=NEVER))
        return torch.cat(x, -1), new

    # ------------------------------------------------------------------ core
    def _fuse(self, enc, spk_act=None, text=None, act_hist=None, v3=None, energy=None):
        x = [enc]
        if self.concat:
            if spk_act is None and act_hist is None:
                raise ValueError("TurnHead(mode=concat) needs spk_act (the primary speaker's activity)")
            x.append((act_hist if act_hist is not None else act_history(spk_act, enc.shape[1], self.history)).to(enc.dtype))
        if self.use_text:
            if text is None:
                raise ValueError("TurnHead(use_text=true) needs text=(tokens, per-frame token counts)")
            x.append(text.to(enc.dtype))
        if self.energy_in is not None and energy is None:
            raise ValueError("TurnHead(energy_input=true) needs energy (B,T): heads.turn.energy_features(audio, ...)")
        if v3 is None and self.energy_in is None:
            return self.inp(torch.cat(x, -1))
        pre = self.inp[0](torch.cat(x, -1))
        if v3 is not None:
            pre = pre + self.v3_in(v3.to(enc.dtype))
        if self.energy_in is not None:
            pre = pre + self.energy_in(_match_len(energy.to(enc.dtype), enc.shape[1])[..., None])
        return self.inp[1:](pre)

    def energy_of(self, audio, audio_len, T: int):
        """(B,T) energy input of this head from the raw audio (None when the head has no energy input)."""
        if self.energy_in is None:
            return None
        # computed on the CPU (a frame-by-frame recurrence of tiny ops: slow on MPS / CUDA), returned on audio's device
        al = audio_len.detach().cpu() if audio_len is not None else None
        e = energy_features(audio.detach().float().cpu(), al, T, self.energy_tau, self.energy_clip)[0]
        return e.to(audio.device)

    def hidden_states(self, enc, enc_len, spk_act=None, text=None, cols=None, prim=None, energy=None):
        tf = self.text_frames(*text) if (self.use_text and text is not None) else None
        v3, _ = self.v3_features(enc.shape[1], spk_act, cols, prim)
        h, _ = self.rnn(self._fuse(enc, spk_act, tf, v3=v3, energy=energy))
        return h

    def forward(self, enc, enc_len, spk_act=None, text=None, cols=None, prim=None, energy=None):
        """text: (y (B,U), n (B,T)) - tokens and the number of them known at each frame. cols (B,T,S) / prim (B,):
        the diarizer's columns and the primary's column (act_columns > 1). energy (B,T): energy_input heads."""
        if self.v3_in is None and self.energy_in is None:  # v2 path, unchanged
            tf = self.text_frames(*text) if (self.use_text and text is not None) else None
            h, _ = self.rnn(self._fuse(enc, spk_act, tf))
            return self.out(h).squeeze(-1)  # (B,T) logits; causal: frame t sees frames <= t only
        return self.out(self.hidden_states(enc, enc_len, spk_act, text, cols, prim, energy)).squeeze(-1)

    def batch_columns(self, batch):
        """(cols, prim) for training: the conditioning track's columns (SpeechModel.conditioning_act), else the
        oracle spk_targets with the primary in column 0 (the AMI / conversation convention)."""
        if "spk_cols" in batch:
            return batch["spk_cols"], batch["spk_prim"]
        if "spk_targets" in batch:
            t = batch["spk_targets"]
            return t, torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
        a = batch["spk_act"].float()
        return a[..., None], torch.zeros(a.shape[0], dtype=torch.long, device=a.device)

    def future_loss(self, h, valid, batch):
        """VAP-style auxiliary BCE: primary / any other speaker active within the next h frames (oracle labels)."""
        k = "spk_act_oracle" if "spk_act_oracle" in batch else "spk_act"
        T = h.shape[1]
        prim = _match_len(batch[k].float(), T)
        L = valid.sum(1)
        tp, okp = future_act_targets(prim, L, self.fut_horizons)
        if "spk_targets" in batch and batch["spk_targets"].shape[-1] > 1:  # column 0 = primary (AMI / conversation)
            oth = _match_len(batch["spk_targets"].float(), T)[..., 1:].max(-1).values
            to, oko = future_act_targets(oth, L, self.fut_horizons)
        else:
            to, oko = torch.zeros_like(tp), torch.zeros_like(okp)
        tg, ok = torch.cat([tp, to], -1), torch.cat([okp, oko], -1) & valid[..., None]
        z = self.fut(h).float()
        l = F.binary_cross_entropy_with_logits(z, tg, reduction="none")
        return (l * ok).sum() / ok.sum().clamp(min=1)

    def loss(self, enc, enc_len, batch):
        act = batch.get("spk_act")
        text = None
        if self.use_text:
            e_asr = self.asr_view(enc, batch)
            # no transcript in this batch (e.g. a source with drop_keys: [text]) -> the ASR's own decode,
            # which is what the head sees at inference anyway
            if "text" not in batch or (self.training and self.decoded_prob > 0 and random.random() < self.decoded_prob):
                y, yl, emit = self.decoded_text(e_asr, enc_len)
                self.align_counts["decoded"] += len(yl)
            else:
                y, yl = batch["text"], batch["text_len"]
                emit = self.align(e_asr, enc_len, y, yl, act)
                y, emit = self.augment_text(y, yl, emit, enc_len)
            text = (y, token_counts(emit, yl, enc.shape[1]))
        h = None
        energy = self.energy_of(batch["audio"], batch.get("audio_len"), enc.shape[1]) if self.energy_in is not None else None
        if self.v3_in is None and self.fut is None and self.mh is None and self.energy_in is None:
            z = self(enc, enc_len, act, text).float()
        else:
            cols, prim = self.batch_columns(batch) if self.needs_cols else (None, None)
            h = self.hidden_states(enc, enc_len, act, text, cols, prim, energy)
            z = self.out(h).squeeze(-1).float()
        # labels: explicit eot, else from the oracle activity (spk_act_oracle when the conditioning spk_act
        # is diarization output / noise-augmented), else from spk_act
        if "eot" in batch:
            tgt = batch["eot"]
        else:
            k = "spk_act_oracle" if "spk_act_oracle" in batch else "spk_act"
            tgt = eot_targets(batch[k], batch.get(k + "_len", batch.get("spk_act_len")))
        tgt = _match_len(tgt.float(), z.shape[1])
        valid = _pad_mask(enc, enc_len)
        if "spk_act_len" in batch:
            valid = valid & (torch.arange(z.shape[1], device=z.device)[None] < batch["spk_act_len"][:, None])
        pw = torch.tensor(self.pos_weight, device=z.device)
        l = F.binary_cross_entropy_with_logits(z, tgt, reduction="none", pos_weight=pw)
        l = (l * valid).sum() / valid.sum().clamp(min=1)
        if self.fut is not None and self.training and self.fut_weight > 0:
            l = l + self.fut_weight * self.future_loss(h, valid, batch)
        if self.mh is not None and self.training and self.mh_weight > 0:
            l = l + self.mh_weight * self.multi_horizon_loss(h, valid, batch)
        return l

    def multi_horizon_loss(self, h, valid, batch):
        """Auxiliary BCE: the PRIMARY (user) party active in each future bin (oracle labels, multi_horizon_targets)."""
        k = "spk_act_oracle" if "spk_act_oracle" in batch else "spk_act"
        prim = _match_len(batch[k].float(), h.shape[1])
        tg, ok = multi_horizon_targets(prim, valid.sum(1), self.mh_edges)
        ok = ok & valid[..., None]
        z = self.mh(h).float()
        l = F.binary_cross_entropy_with_logits(z, tg, reduction="none")
        return (l * ok).sum() / ok.sum().clamp(min=1)

    @torch.no_grad()
    def decode(self, enc, enc_len, spk_act=None, text=None, cols=None, prim=None, energy=None, **_):
        """Per-frame P(primary's turn is over). None if a required input (spk_act / text / cols / energy) is missing."""
        if (self.needs_act and spk_act is None) or (self.use_text and text is None) or \
                (self.needs_cols and (cols is None or prim is None)) or (self.energy_in is not None and energy is None):
            return None
        if self.v3_in is None and self.energy_in is None:
            return self(enc, enc_len, spk_act, text).sigmoid()
        return self(enc, enc_len, spk_act, text, cols, prim, energy).sigmoid()

    @torch.no_grad()
    def decode_aux(self, enc, enc_len, spk_act=None, text=None, cols=None, prim=None, energy=None):
        """(P(EOT) (B,T), P(user active in each multi_horizon_aux bin) (B,T,H)) from one pass; the second is None
        without multi_horizon_aux. The dyadic predictive trigger reads bins 1-2."""
        h = self.hidden_states(enc, enc_len, spk_act, text, cols, prim, energy)
        return self.out(h).squeeze(-1).sigmoid(), (self.mh(h).sigmoid() if self.mh is not None else None)

    # ------------------------------------------------------------------ streaming
    def init_stream(self, batch_size: int = 1) -> TurnStreamState:
        return TurnStreamState()

    @torch.no_grad()
    def step(self, enc_frames, pred_state=None, last_tokens=None, spk_act=None, state: TurnStreamState | None = None,
             cols=None, prim=None, audio=None):
        """Strictly causal streaming step -> probs (B,n); ``state`` (from ``init_stream``) is updated in place.

        enc_frames (B,n,D) new encoder frames; pred_state: the ASR PredictionNet output after the decoded
        prefix - (B,H) / (B,1,H) tensor or StreamingSession's ``(g, lstm_state)`` tuple; last_tokens: the
        decoded prefix (B,L) tensor or list of token lists (only the last ``k_tokens`` are used);
        spk_act (B,n) the primary's activity on these frames; cols (B,n,S) / prim (B,) the diarizer's columns and
        the primary's column (act_columns > 1). The duration counters continue from ``state.dur``. The text state
        is held constant over the n frames, so feed one frame at a time with the prefix decoded up to that frame
        for exact equivalence with the offline forward. audio (B, 1280 n): the raw samples of these n frames
        (energy_input heads; the standardisation continues from ``state.energy``)."""
        state = state if state is not None else TurnStreamState()
        B, n, _ = enc_frames.shape
        act_hist = None
        if self.concat:
            if spk_act is None:
                raise ValueError("TurnHead(mode=concat) needs spk_act")
            a = spk_act.float().reshape(B, n).to(enc_frames.device)
            buf = state.act_buf if state.act_buf is not None else a.new_zeros(B, self.history - 1)
            full = torch.cat([buf, a], 1)  # B, history-1+n
            act_hist = torch.stack([full[:, self.history - 1 - k: self.history - 1 - k + n]
                                    for k in range(self.history)], -1)
            state.act_buf = full[:, full.shape[1] - (self.history - 1):]
        tf = None
        if self.use_text:
            g = pred_state[0] if isinstance(pred_state, tuple) else pred_state
            if g is None:  # empty prefix: state after <sos>
                g, _ = self._asr.pred(torch.full((B, 1), self._asr.blank, dtype=torch.long, device=enc_frames.device))
            g = g.reshape(B, -1)
            blank = self._asr.blank
            if last_tokens is None:
                last_tokens = [[] for _ in range(B)]
            if torch.is_tensor(last_tokens):
                last_tokens = last_tokens.tolist()
            ids = [([blank] * self.k_tokens + list(s))[-self.k_tokens:][::-1] for s in last_tokens]
            last = torch.tensor(ids, dtype=torch.long, device=enc_frames.device)  # most recent first
            tf = self._text_feats(g[:, None].expand(B, n, g.shape[-1]), last[:, None].expand(B, n, self.k_tokens))
        v3 = None
        if self.v3_in is not None:
            sa = spk_act.float().reshape(B, n).to(enc_frames.device) if spk_act is not None else None
            v3, state.dur = self.v3_features(n, sa, cols, prim, state.dur)
        en = None
        if self.energy_in is not None:
            if audio is None:
                raise ValueError("TurnHead(energy_input=true).step needs audio (the raw samples of these frames)")
            a = torch.as_tensor(audio, dtype=torch.float32, device=enc_frames.device).reshape(B, -1)
            en, state.energy = energy_features(a, None, n, self.energy_tau, self.energy_clip, state.energy)
        h, state.h = self.rnn(self._fuse(enc_frames, None, tf, act_hist, v3=v3, energy=en), state.h)
        return self.out(h).squeeze(-1).sigmoid()

    # ------------------------------------------------------------------ evaluation hook (Trainer.evaluate)
    @torch.no_grad()
    def evaluate_model(self, model, name: str, val: list[dict]) -> dict:
        from ..conversation import baselines, eot_bench
        if "spk_act" not in val[0]:
            return {}
        on, en = _onset_end(val)
        chunk = model.encoder.att_context_size[1] + 1 if model.encoder.att_context_size[1] >= 0 else 10 ** 6
        res = {}
        sources = ["oracle"] + (["diar"] if self.mode != "none" and _diar_name(model) else [])
        for src in sources:
            st = {}
            b = eot_bench(turn_scores(model, name, val, act_source=src, stats=st), on, en, chunk=chunk)
            tag = f"eot_{name}" + ("" if src == "oracle" else "_diar")
            res.update(_flat(tag, b))
            if src == "diar" and st.get("speech"):  # primary frames the diar activity misses (0.91 in v2)
                res[f"{tag}_act_miss"] = round(st["miss"] / st["speech"], 4)
        vv = [dict(v, onset_frame=o, turn_end_frame=e, primary_act=np.asarray(v["spk_act"]))
              for v, o, e in zip(val, on, en)]
        res.update(_flat("eot_timeout", baselines(vv)["timeout_primary_oracle"]))
        ref_items = self.align_counts["greedy"] + self.align_counts["uniform"]
        if self.use_text and ref_items:
            res[f"{name}_align_greedy_frac"] = round(self.align_counts["greedy"] / ref_items, 4)
        if self.use_text and self.align_counts["decoded"]:
            res[f"{name}_decoded_frac"] = round(self.align_counts["decoded"] / (ref_items + self.align_counts["decoded"]), 4)
        return res


def _flat(tag, b):
    p, f = b["at_max_fc"], b["at_fixed_latency"]
    return {f"{tag}_p50_ms@5fc": p["p50_ms"], f"{tag}_p90_ms@5fc": p["p90_ms"], f"{tag}_fc@5fc": p["fc_rate"],
            f"{tag}_fc@p50<=400ms": f["fc_rate"]}


def _onset_end(convs):
    on, en = [], []
    for c in convs:
        nz = np.nonzero(np.asarray(c["spk_act"]) > 0.5)[0]
        on.append(int(nz[0]) if len(nz) else 0)
        en.append(int(nz[-1]) + 1 if len(nz) else 0)
    return on, en


def _diar_name(model):
    return next((k for k, v in model.head_cfg.items() if v["type"] == "sortformer"), None)


def primary_column(spk_targets: torch.Tensor) -> torch.Tensor:
    """Arrival rank of column 0 (the primary) - its column in Sortformer's arrival-ordered output.
    Stands in for enrollment: the system knows who the user is, not when they speak."""
    return (arrival_order(spk_targets) == 0).float().argmax(1)


@torch.no_grad()
def streaming_diar_act(model, enc, elen, col, chunk: int) -> torch.Tensor:
    """Primary activity from the Sortformer head, computed *causally*: the head (non-causal transformer)
    is re-run on every growing prefix, and frame t keeps the value from the prefix ending at t's chunk."""
    head = model.heads[_diar_name(model)]
    B, T, _ = enc.shape
    act = torch.zeros(B, T, device=enc.device)
    for ce in range(chunk, T + chunk, chunk):
        ce = min(ce, T)
        p = head(enc[:, :ce], elen.clamp(max=ce)).sigmoid()  # B,ce,S
        s = max(0, ce - chunk)
        act[:, s:ce] = p[:, s:ce].gather(2, col[:, None, None].expand(B, ce - s, 1))[..., 0]
    return act


@torch.no_grad()
def streaming_diar_probs(model, enc, elen, chunk: int) -> torch.Tensor:
    """All S columns of the Sortformer head, computed causally as ``streaming_diar_act`` (growing prefixes;
    frame t keeps the value from the prefix ending at t's chunk) -> (B,T,S). For act_columns > 1 heads."""
    head = model.heads[_diar_name(model)]
    B, T, _ = enc.shape
    out = None
    for ce in range(chunk, T + chunk, chunk):
        ce = min(ce, T)
        p = head(enc[:, :ce], elen.clamp(max=ce)).sigmoid()  # B,ce,S
        out = p.new_zeros(B, T, p.shape[2]) if out is None else out
        s = max(0, ce - chunk)
        out[:, s:ce] = p[:, s:ce]
    return out


@torch.no_grad()
def decoded_text_state(model, head: TurnHead, audio, audio_len, enc, hidden, elen, act) -> tuple:
    """Inference text path: greedy-decode the ASR head (speaker-conditioned if it is) and return
    (y, n): decoded tokens and the number of them emitted at frames <= t (causal)."""
    asr_name = head.text_head
    if model.head_cfg[asr_name].get("condition_on_speaker"):
        e, _ = model.encode(audio, audio_len, spk_act=act)
    else:
        e = model.head_input(asr_name, enc, hidden)
    hyps, frames = zip(*[greedy_decode_frames(head._asr, e[b, : int(elen[b])]) for b in range(e.shape[0])])
    y, yl = pad_tokens(list(hyps), device=e.device)
    emit, _ = pad_tokens(list(frames), fill=-1, device=e.device)
    return y, token_counts(emit, yl, enc.shape[1])


@torch.no_grad()
def turn_scores(model, name: str, convs: list[dict], act_source: str = "oracle", batch_size: int = 32,
                stats: dict | None = None, act_override: list | None = None, cols_override: list | None = None,
                prim_override: list | None = None) -> list[np.ndarray]:
    """Per-frame P(EOT) for each conversation. act_source: oracle (ground-truth primary activity) or
    diar (causal Sortformer output, primary column chosen by arrival rank = enrollment stand-in).
    act_override: one (T_i,) activity track per conversation (e.g. an external diarizer's primary column,
    datasets/ext_tracks.py), already on the model's 80 ms grid; it replaces ``act`` (act_source is ignored):
    the first min(T_i, T_enc) frames are copied, the rest is 0.
    With ``use_text`` the text state comes from the ASR head's own greedy decoding (never the reference).
    act_columns > 1 heads also get the diarizer's columns + the primary's column: oracle = spk_targets (primary in
    column 0), diar = all columns of the causal prefix re-run (primary = arrival rank), override = cols_override
    (one (T_i, S) track per conversation, copied like act_override) and prim_override (one column index each)."""
    from ..data import Collate, to_device
    model.eval()
    head, hc = model.heads[name], model.head_cfg[name]
    assert ("kernel" in head.mode) == bool(hc.get("condition_on_speaker")), \
        "mode kernel needs condition_on_speaker: true (and vice versa)"
    dev = next(model.parameters()).device
    att = model.encoder.att_context_size
    chunk = att[1] + 1 if att[1] >= 0 else 10 ** 6
    out = []
    for i in range(0, len(convs), batch_size):
        chunk_convs = convs[i: i + batch_size]
        b = to_device(Collate(model.tokenizer)(chunk_convs), dev)
        enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
        act = _match_len(b["spk_act"].float(), enc.shape[1])
        cols = prim = None
        if head.needs_cols:
            if act_override is not None:
                if cols_override is None or prim_override is None:
                    raise ValueError(f"turn head {name!r} has act_columns={head.act_columns}: pass cols_override "
                                     "and prim_override with act_override")
                cols = torch.zeros(enc.shape[0], enc.shape[1], head.act_columns, device=dev)
                for j, c in enumerate(cols_override[i: i + batch_size]):
                    c = np.asarray(c, np.float32)
                    n, S = min(len(c), enc.shape[1]), min(c.shape[1], head.act_columns)
                    cols[j, :n, :S] = torch.as_tensor(c[:n, :S], device=dev)
                prim = torch.as_tensor([int(x) for x in prim_override[i: i + batch_size]], device=dev)
            elif act_source == "diar":
                cols = streaming_diar_probs(model, model.head_input(_diar_name(model), enc, hidden), elen, chunk)
                prim = primary_column(b["spk_targets"]).clamp(max=cols.shape[2] - 1)
            else:
                cols = b["spk_targets"].float() if "spk_targets" in b else act[..., None]
                prim = torch.zeros(enc.shape[0], dtype=torch.long, device=dev)
        if act_override is not None:
            act = torch.zeros(enc.shape[0], enc.shape[1], device=dev)
            for j, a in enumerate(act_override[i: i + batch_size]):
                n = min(len(a), enc.shape[1])
                act[j, :n] = torch.as_tensor(np.asarray(a[:n], np.float32), device=dev)
        elif act_source == "diar":
            act = streaming_diar_act(model, model.head_input(_diar_name(model), enc, hidden), elen,
                                     primary_column(b["spk_targets"]), chunk)
            if stats is not None:  # primary-frame miss of the activity the turn head is conditioned on
                ref = _match_len(b["spk_act"].float(), enc.shape[1]) > 0.5
                ref &= torch.arange(enc.shape[1], device=dev)[None] < elen[:, None]
                stats["miss"] = stats.get("miss", 0) + int((ref & (act <= 0.5)).sum())
                stats["speech"] = stats.get("speech", 0) + int(ref.sum())
        if hc.get("condition_on_speaker"):
            e = model.cond_head_input(name, b["audio"], b["audio_len"], act)
        else:
            e = model.head_input(name, enc, hidden)
        text = decoded_text_state(model, head, b["audio"], b["audio_len"], enc, hidden, elen, act) \
            if head.use_text else None
        if head.v3_in is None and head.energy_in is None:  # v2 heads: the unchanged call
            p = head.decode(e, elen, spk_act=act if head.concat else None, text=text)
        else:
            p = head.decode(e, elen, spk_act=act, text=text, cols=cols, prim=prim,
                            energy=head.energy_of(b["audio"], b["audio_len"], e.shape[1]))
        for j in range(len(chunk_convs)):
            n = min(int(elen[j]), len(chunk_convs[j]["spk_act"]))
            out.append(p[j, :n].float().cpu().numpy())
    return out
