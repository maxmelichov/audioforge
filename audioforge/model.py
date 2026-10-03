"""SpeechModel: the NVIDIA blueprint as one configurable module.

    waveform -> LogMel -> SpecAugment -> FastConformer (8x, 80 ms) -> {head_1, head_2, ...}

Parakeet-TDT-CTC, Canary, Sortformer, Nemotron-ASR-Streaming, EOU and NEST are
all points in this space: same front end, same encoder, different heads.
The multi-head form also enables the *new* recipes in ``research/recipes/``.
"""
from __future__ import annotations

import copy
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import LogMel, SpecAugment
from .heads.asr import AEDHead, CTCHead, RNNTHead
from .heads.audio import CodecTokenHead, FrameHead, LanguageHead, SortformerHead, SpeakerHead
from .modules.fastconformer import FastConformerEncoder, StreamState

log = logging.getLogger(__name__)

TEXT_HEADS = {"ctc", "rnnt", "tdt", "aed"}


# Train-time activity conditioning (research/archive/TURN_ABLATION.md, A): the speaker-conditioned pass is trained on the
# clean oracle primary activity but runs on diarization output at inference. ``conditioning:`` (top level, or
# ``training.activity_conditioning``) replaces the conditioning track per item: with probability p_diar by the diar
# head's own detached sigmoid primary column (arrival-rank rule, heads/turn.py primary_column; offline pass, or the
# causal prefix re-run eval uses when diar_causal), else by the oracle with frame flips (rate flip), boundary jitter
# (+-jitter frames) and dropped short segments (prob drop, length <= drop_max). All off by default.
# p_ext: with this probability (applied last, so it overrides the others) the track is the batch's ``spk_act_ext``,
# a REAL external diarizer's primary track (datasets/ext_tracks.py: cached NVIDIA Sortformer output, enrollment-picked
# column), used as is (soft probabilities, as at inference); items / batches without a valid ext track keep the
# oracle / noisy oracle / own-diar track. ``spk_act_oracle`` always keeps the clean labels.
# Location: top-level ``conditioning:``, ``training.activity_conditioning``, or ``trainer.conditioning`` (the only one
# that survives train.init_from_afm, which copies ``trainer`` but not the top-level key from the recipe).
# ext_noise (TurnHead v3, research/archive/STAGE1.md n=200): corrupt the external track of the items that use it, aimed at the
# failure "the next speaker lands in my slot" - with probability p per item, the primary's column is swapped with the
# most active other column for <= swap_max frames around a speaker change (the oracle turn end with probability
# at_turn_end, else a frame where the track's active set changes), plus, with probability drop, one short dropout
# (<= drop_max frames) of a column. The kernel track spk_act is the corrupted track's primary column. A float is p.
# Columns for TurnHead(act_columns > 1): when a head reads them, conditioning_act also builds the conditioning track's
# (B,T,S) columns (batch["spk_cols"]) and the primary's column (batch["spk_prim"]): the ext track's columns and its
# enrollment column (spk_prim_ext) for ext items, the own diar head's columns for p_diar items, else the oracle
# spk_targets with the (noisy) primary track in its column and the columns randomly permuted (the column index
# carries no information, as with a real diarizer).
# Head-only conditioning (TurnHead v5, research/archive/TURN_ERRORS.md: "drop kernel conditioning so the track reaches only
# the head"): the conditioning above also runs when no head is speaker-conditioned but a head READS the activity
# itself (TurnHead mode concat, duration_feats or act_columns > 1: head.needs_act / needs_cols). Then the noisy /
# external track reaches that head only (batch spk_act / spk_cols / spk_prim), spk_act_oracle keeps the clean labels,
# and the encoder runs once, unconditioned. Recipes with a condition_on_speaker head behave exactly as before.
# rebind (research/archive/CONTAMINATION.md section 9, research/archive/DYADIC.md section 8): corrupted-then-CORRECTED windows. With
# probability p per item the final conditioning track (whatever its source) binds the WRONG party for W ~ U{w_min ..
# w_max} frames ending at a uniform frame between the primary's onset and its turn end, then is correct again: the
# primary column is swapped with the most active other column over the window (the real failure, a follower rebind,
# is a swap), and the kernel track is that corrupted primary column. The labels stay the clean ones, so the head is
# asked to RECOVER after the correction (the GRU state the probe found contaminated for seconds).
COND_DEFAULTS = {"p_diar": 0.0, "flip": 0.0, "jitter": 0, "drop": 0.0, "drop_max": 5, "diar_causal": False,
                 "p_ext": 0.0, "ext_noise": 0.0, "rebind": 0.0}
EXT_NOISE_DEFAULTS = {"p": 0.0, "swap_max": 12, "drop": 0.0, "drop_max": 4, "at_turn_end": 0.5}
REBIND_DEFAULTS = {"p": 0.0, "w_min": 12, "w_max": 50}


def rebind_cfg(v) -> dict:
    """``rebind``: a float (= p) or a dict with REBIND_DEFAULTS keys -> the full dict."""
    d = {"p": float(v)} if isinstance(v, (int, float)) else dict(v or {})
    unknown = set(d) - set(REBIND_DEFAULTS)
    if unknown:
        raise ValueError(f"conditioning.rebind: unknown keys {sorted(unknown)} (known: {sorted(REBIND_DEFAULTS)})")
    return {**REBIND_DEFAULTS, **d}


def rebind_corruption(cols: torch.Tensor, prim: torch.Tensor, oracle: torch.Tensor, lengths: torch.Tensor,
                      p: float, w_min: int = 12, w_max: int = 50, gen: torch.Generator | None = None):
    """Corrupted-then-corrected copy of a conditioning track (see rebind above). cols (B,T,S) the track's columns,
    prim (B,) the primary's column, oracle (B,T) the clean primary activity (onset / turn end), lengths (B,) valid
    frames. -> (cols', windows): windows[b] = (s0, e0) of the wrong stretch or None. Frames >= e0 are untouched."""
    B, T, S = cols.shape
    out, wins = cols.clone(), []
    for b in range(B):
        n = min(int(lengths[b]), T)
        on = (oracle[b, :n] > 0.5).nonzero()[:, 0]
        if S < 2 or len(on) == 0 or float(torch.rand((), generator=gen)) >= p:
            wins.append(None)
            continue
        onset, end = int(on[0]), int(on[-1]) + 1
        e0 = int(torch.randint(onset + 1, max(onset + 2, end + 1), (), generator=gen))
        W = int(torch.randint(int(w_min), int(w_max) + 1, (), generator=gen))
        s0 = max(0, e0 - W)
        pb = int(prim[b])
        others = [k for k in range(S) if k != pb]
        mass = out[b, s0:e0][:, others].sum(0)
        o = others[int(mass.argmax())] if float(mass.max()) > 0 else others[0]
        seg = out[b, s0:e0, pb].clone()
        out[b, s0:e0, pb] = out[b, s0:e0, o]
        out[b, s0:e0, o] = seg
        wins.append((s0, e0))
    return out, wins


def ext_noise_cfg(v) -> dict:
    """``ext_noise``: a float (= p) or a dict with EXT_NOISE_DEFAULTS keys -> the full dict."""
    d = {"p": float(v)} if isinstance(v, (int, float)) else dict(v or {})
    unknown = set(d) - set(EXT_NOISE_DEFAULTS)
    if unknown:
        raise ValueError(f"conditioning.ext_noise: unknown keys {sorted(unknown)} (known: {sorted(EXT_NOISE_DEFAULTS)})")
    return {**EXT_NOISE_DEFAULTS, **d}


def ext_track_noise(cols: torch.Tensor, prim: torch.Tensor, lengths: torch.Tensor, turn_end: torch.Tensor | None = None,
                    p: float = 0.5, swap_max: int = 12, drop: float = 0.0, drop_max: int = 4,
                    at_turn_end: float = 0.5) -> torch.Tensor:
    """Corrupted copy of an external diarizer track cols (B,T,S), primary column prim (B,), valid frames lengths (B,)
    (see ext_noise above). turn_end (B,): the oracle primary's turn end (one past its last active frame)."""
    B, T, S = cols.shape
    out = cols.clone()
    for b in range(B):
        n = min(int(lengths[b]), T)
        if n < 2 or float(torch.rand(())) >= p:
            continue
        pb = int(prim[b])
        on = out[b, :n] > 0.5
        cands = ((on[1:] != on[:-1]).any(1).nonzero()[:, 0] + 1).tolist()  # frames where the active set changes
        te = int(turn_end[b]) if turn_end is not None else -1
        if 0 < te < n and (float(torch.rand(())) < at_turn_end or not cands):
            t0 = te
        elif cands:
            t0 = cands[int(torch.randint(len(cands), ()))]
        else:
            t0 = None
        if t0 is not None and S > 1:
            L = int(torch.randint(1, swap_max + 1, ()))
            s0 = max(0, t0 - int(torch.randint(0, L // 2 + 1, ())))
            e0 = min(n, s0 + L)
            others = [k for k in range(S) if k != pb]
            mass = out[b, t0:min(n, t0 + L)][:, others].sum(0)  # the next speaker = the most active other column
            o = others[int(mass.argmax())] if float(mass.max()) > 0 else others[int(torch.randint(len(others), ()))]
            seg = out[b, s0:e0, pb].clone()
            out[b, s0:e0, pb] = out[b, s0:e0, o]
            out[b, s0:e0, o] = seg
        if drop > 0 and float(torch.rand(())) < drop:
            k = pb if float(torch.rand(())) < 0.5 else int(torch.randint(S, ()))
            L = int(torch.randint(1, drop_max + 1, ()))
            s0 = int(torch.randint(0, max(1, n - L + 1), ()))
            out[b, s0:s0 + L, k] = 0
    return out


def _fit_cols(x: torch.Tensor, T: int, S: int) -> torch.Tensor:
    """(B,T0,S0) -> (B,T,S): frames and columns cropped or zero-padded."""
    x = x[:, :T]
    x = F.pad(x, (0, 0, 0, T - x.shape[1])) if x.shape[1] < T else x
    return x[..., :S] if x.shape[2] >= S else F.pad(x, (0, S - x.shape[2]))


def augment_activity(act: torch.Tensor, lengths: torch.Tensor, flip: float = 0.0, jitter: int = 0,
                     drop: float = 0.0, drop_max: int = 5) -> torch.Tensor:
    """(B,T) activity -> noisy binary (B,T) float copy, only inside each item's first ``lengths[b]`` frames.
    Per active segment: dropped with probability ``drop`` if it is <= ``drop_max`` frames, else its start and end
    move independently by U{-jitter..jitter} (>= 1 frame kept); then each valid frame flips with rate ``flip``."""
    B, T = act.shape
    on = (act > 0.5).float()
    if not (flip > 0 or jitter > 0 or drop > 0):
        return on
    out = torch.zeros_like(on)
    for b in range(B):
        n = min(int(lengths[b]), T)
        x = on[b, :n]
        edges = torch.diff(F.pad(x, (1, 1))).nonzero()[:, 0].tolist()  # alternating starts / ends
        for s, e in zip(edges[0::2], edges[1::2]):
            if drop > 0 and e - s <= drop_max and float(torch.rand(())) < drop:
                continue
            if jitter > 0:
                s, e = (int(v) + int(torch.randint(-jitter, jitter + 1, ())) for v in (s, e))
                s = min(max(s, 0), n - 1)
                e = min(max(e, s + 1), n)
            out[b, s:e] = 1
        if flip > 0:
            f = torch.rand(n, device=act.device) < flip
            out[b, :n] = torch.where(f, 1 - out[b, :n], out[b, :n])
    return out


class GradScale(torch.autograd.Function):
    """Identity forward; scales the gradient flowing back into the shared encoder."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


def build_head(cfg: dict, d_model: int, tokenizer=None) -> nn.Module:
    cfg = dict(cfg)
    t = cfg.pop("type")
    for k in ("weight", "condition_on_speaker", "grad_scale", "from_layers",
              "vad_input"):  # vad_input: a serve-time option of turn_seg heads (research/TURN_DATA.md)
        cfg.pop(k, None)
    V = tokenizer.vocab_size if tokenizer is not None else None
    if t == "ctc":
        return CTCHead(d_model, V, **cfg)
    if t in ("rnnt", "tdt"):
        if t == "tdt":
            cfg.setdefault("durations", [0, 1, 2, 3, 4])
        return RNNTHead(d_model, V, **cfg)
    if t == "aed":
        return AEDHead(d_model, V, tokenizer.token_id("<bos>"), tokenizer.token_id("<eos>"),
                       tokenizer.token_id("<pad>"), **cfg)
    if t == "sortformer":
        return SortformerHead(d_model, **cfg)
    if t == "speaker":
        return SpeakerHead(d_model, **cfg)
    if t == "language":  # spoken language ID, running posterior (research/archive/LID.md)
        return LanguageHead(d_model, **cfg)
    if t == "frame":
        return FrameHead(d_model, **cfg)
    if t == "frame_gru":  # causal GRU frame head (the 0.6B core's VAD, heads/audio.py FrameGRUHead)
        from .heads.audio import FrameGRUHead
        return FrameGRUHead(d_model, **cfg)
    if t == "codec_tokens":
        return CodecTokenHead(d_model, **cfg)
    if t == "turn":  # speaker-aware end-of-turn (heads/turn.py)
        from .heads.turn import TurnHead
        return TurnHead(d_model, **cfg)
    if t == "tsvad":  # target-speaker VAD conditioned on an enrollment embedding (heads/tsvad.py)
        from .heads.tsvad import TSVADHead
        return TSVADHead(d_model, **cfg)
    if t == "turn_seg":  # turn head v5: segment end-of-turn classifier on the session's window (heads/turn_seg.py)
        from .heads.turn_seg import SegTurn
        return SegTurn(**cfg)
    if t == "voice_gender":  # optional perceived voice gender (heads/voice_gender.py, research/VOICE_GENDER.md)
        from .heads.voice_gender import VoiceGenderHead
        return VoiceGenderHead(d_model, **cfg)
    if t == "completeness":  # utterance completeness, smart-turn's task (heads/completeness.py)
        from .heads.completeness import CompletenessHead
        return CompletenessHead(d_model, **cfg)
    raise ValueError(f"unknown head type {t!r}")


class SpeechModel(nn.Module):
    def __init__(self, cfg: dict, tokenizer=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.tokenizer = tokenizer
        pre = dict(cfg.get("preprocessor", {}))
        pre.setdefault("sample_rate", cfg.get("sample_rate", 16000))
        self.preprocessor = LogMel(**pre)
        sa = cfg.get("spec_augment")
        self.spec_augment = SpecAugment(**sa) if sa else None
        enc = dict(cfg["encoder"])
        enc.setdefault("feat_in", self.preprocessor.n_mels)
        self.encoder = FastConformerEncoder(**enc)
        d = self.encoder.d_model
        self.head_cfg = {k: dict(v) for k, v in cfg["heads"].items()}
        self.heads = nn.ModuleDict({k: build_head(v, d, tokenizer) for k, v in self.head_cfg.items()})
        for h in self.heads.values():  # heads that read another head (turn: the ASR PredictionNet state)
            if hasattr(h, "bind"):
                h.bind(self)
        # from_layers: all -> the head reads a learned softmax-weighted sum of every encoder layer
        # (NEST-style; speaker identity lives in middle layers, ASR training erases it at the top)
        self.layer_mix = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(len(self.encoder.layers)))
            for k, v in self.head_cfg.items() if v.get("from_layers") == "all"})
        # from_layers: k | [k, ...] -> the head reads exactly encoder layer k (0-based block index, as
        # speaker_kernel_layers), or a learned softmax mix restricted to the listed layers (research/archive/SPK_HEAD.md)
        self.layer_tap: dict[str, list[int]] = {}
        for k, v in self.head_cfg.items():
            fl = v.get("from_layers")
            if fl is None or fl == "all":
                continue
            idx = [int(i) for i in (fl if isinstance(fl, (list, tuple)) else [fl])]
            n = len(self.encoder.layers)
            idx = [i + n if i < 0 else i for i in idx]
            if not idx or any(i < 0 or i >= n for i in idx) or len(set(idx)) != len(idx):
                raise ValueError(f"heads.{k}.from_layers: {fl!r} must be 'all' or distinct layer indices in [0, {n})")
            self.layer_tap[k] = idx
            if len(idx) > 1:
                self.layer_mix[k] = nn.Parameter(torch.zeros(len(idx)))
        self.primary = cfg.get("decoding", {}).get("primary") or next(
            (k for k, v in self.head_cfg.items() if v["type"] in TEXT_HEADS), None)
        cc = (cfg.get("conditioning") or cfg.get("training", {}).get("activity_conditioning")
              or (cfg.get("trainer") or {}).get("conditioning") or {})
        unknown = set(cc) - set(COND_DEFAULTS)
        if unknown:
            raise ValueError(f"conditioning: unknown keys {sorted(unknown)} (known: {sorted(COND_DEFAULTS)})")
        self.cond = {**COND_DEFAULTS, **cc}
        self.ext_noise = ext_noise_cfg(self.cond["ext_noise"])
        self.rebind = rebind_cfg(self.cond["rebind"])
        self.cond_on = any(self.cond[k] for k in ("p_diar", "flip", "jitter", "drop", "p_ext")) or self.rebind["p"] > 0
        # TurnHead(act_columns > 1): the conditioning also builds the track's columns (conditioning_act with_cols)
        self.n_act_columns = max([int(getattr(h, "act_columns", 1)) for h in self.heads.values()] + [1])
        self._skip_logged: set = set()  # forward(): head sets whose skipped loss was already reported

    # ------------------------------------------------------------------ core
    def features(self, audio, audio_len):
        feats, flen = self.preprocessor(audio, audio_len)
        if self.spec_augment is not None and self.training:
            feats = self.spec_augment(feats, flen)
        return feats, flen

    def encode(self, audio, audio_len, att_context_size=None, spk_act=None, return_hidden=False):
        feats, flen = self.features(audio, audio_len)
        return self.encoder(feats, flen, att_context_size, spk_act=spk_act, return_hidden=return_hidden)

    def head_input(self, name, enc, hidden):
        if name in self.layer_tap:
            idx = self.layer_tap[name]
            if len(idx) == 1:  # single-layer tap: exactly that layer's output
                return hidden[idx[0]]
            w = self.layer_mix[name].softmax(0)
            return sum(wi * hidden[i] for wi, i in zip(w, idx))
        if name not in self.layer_mix:
            return enc
        w = self.layer_mix[name].softmax(0)
        return sum(wi * h for wi, h in zip(w, hidden))

    def cond_head_input(self, name, audio, audio_len, spk_act):
        """Input of a condition_on_speaker head: the speaker-conditioned encoder pass read at the head's from_layers
        (the top layer when it has none, exactly ``encode(..., spk_act=...)[0]`` as before)."""
        if name not in self.layer_tap and name not in self.layer_mix:
            return self.encode(audio, audio_len, spk_act=spk_act)[0]
        enc, _, hidden = self.encode(audio, audio_len, spk_act=spk_act, return_hidden=True)
        return self.head_input(name, enc, hidden)

    @torch.no_grad()
    def layer_weights(self, name: str):
        """Effective per-layer weights (n_layers,) of head ``name``'s input, or None when it reads the top layer:
        the softmax mix (from_layers: all), a one-hot (single tap) or the sub-mix scattered onto all layers."""
        n = len(self.encoder.layers)
        if name in self.layer_tap:
            w = torch.zeros(n)
            idx = self.layer_tap[name]
            w[idx] = 1.0 if len(idx) == 1 else self.layer_mix[name].detach().softmax(0).cpu()
            return w
        if name in self.layer_mix:
            return self.layer_mix[name].detach().softmax(0).cpu()
        return None

    @property
    def head_reads_act(self) -> bool:
        """A head that is NOT speaker-conditioned reads the activity track itself (TurnHead concat / duration_feats /
        act_columns > 1): the conditioning then reaches that head only (head-only conditioning, v5)."""
        return any(not self.head_cfg[k].get("condition_on_speaker")
                   and (bool(getattr(h, "needs_act", False)) or bool(getattr(h, "needs_cols", False)))
                   for k, h in self.heads.items())

    @torch.no_grad()
    def conditioning_act(self, batch, enc, elen, hidden, att, with_cols: bool = False):
        """Train-time conditioning track (B,T) for the speaker-conditioned pass (see COND_DEFAULTS). with_cols: return
        (act, cols (B,T,S), prim (B,)) - the same track's columns and primary column, act == cols[prim] on valid frames."""
        from .heads.turn import _diar_name, primary_column, streaming_diar_act, streaming_diar_probs
        c, T = self.cond, enc.shape[1]
        oracle = batch["spk_act"][:, :T].float()
        oracle = torch.cat([oracle, oracle.new_zeros(oracle.shape[0], T - oracle.shape[1])], 1)
        lens = batch.get("spk_act_len", elen).clamp(max=T)
        act = augment_activity(oracle, lens, c["flip"], int(c["jitter"]), c["drop"], int(c["drop_max"]))
        nc = self.n_act_columns  # columns the TurnHead reads (with_cols)
        if with_cols:  # oracle columns: noisy primary track in column 0, then a random permutation per item
            B = act.shape[0]
            oc = (_fit_cols(batch["spk_targets"].float().to(act.device), T, nc) if "spk_targets" in batch
                  else act.new_zeros(B, T, nc))
            oc[..., 0] = act
            oc = oc * (torch.arange(T, device=act.device)[None] < torch.minimum(lens.to(act.device), elen)[:, None])[..., None]
            perm = torch.rand(B, nc, device=act.device).argsort(1)  # new column j holds old column perm[:, j]
            cols = oc.gather(2, perm[:, None, :].expand(B, T, nc))
            prim = (perm == 0).float().argmax(1)
        diar = _diar_name(self)
        if c["p_diar"] > 0 and diar is not None and "spk_targets" in batch:
            use = torch.rand(act.shape[0], device=act.device) < c["p_diar"]
            if use.any():
                head, was = self.heads[diar], self.heads[diar].training
                head.eval()  # the inference-time track: no dropout
                e = self.head_input(diar, enc, hidden).detach()
                col = primary_column(batch["spk_targets"])  # arrival rank of the primary = its output column
                S = head.num_spks
                use &= col < S  # primary beyond the head's columns: keep the oracle for that item
                col = col.clamp(max=S - 1)
                if c["diar_causal"]:
                    chunk = att[1] + 1 if att[1] >= 0 else 10 ** 6
                    d = streaming_diar_act(self, e, elen, col, chunk)
                    if with_cols:
                        p = streaming_diar_probs(self, e, elen, chunk).float()
                else:
                    p = head(e, elen).float().sigmoid()
                    d = p.gather(2, col[:, None, None].expand(-1, T, 1))[..., 0]
                head.train(was)
                d = d * (torch.arange(T, device=d.device)[None] < elen[:, None])
                act = torch.where(use[:, None], d, act)
                if with_cols:
                    pc = _fit_cols(p, T, nc) * (torch.arange(T, device=d.device)[None] < elen[:, None])[..., None]
                    cols = torch.where(use[:, None, None], pc, cols)
                    prim = torch.where(use, col.clamp(max=nc - 1), prim)
        if c["p_ext"] > 0 and "spk_act_ext" in batch:
            ext = batch["spk_act_ext"][:, :T].float().to(act.device)
            ext = torch.cat([ext, ext.new_zeros(ext.shape[0], T - ext.shape[1])], 1)
            elen_x = batch.get("spk_act_ext_len", elen).to(act.device).clamp(max=T)
            valid = torch.arange(T, device=act.device)[None] < torch.minimum(elen_x, elen)[:, None]
            ok = ~((ext < 0) & valid).any(1)  # ext_tracks.MISSING (-1): no cached track for this item
            use = (torch.rand(act.shape[0], device=act.device) < c["p_ext"]) & ok
            xn = self.ext_noise
            if (with_cols or xn["p"] > 0) and "spk_targets_ext" in batch:
                xc = _fit_cols(batch["spk_targets_ext"].float().to(act.device), T, batch["spk_targets_ext"].shape[-1])
                xc = xc.clamp(0, 1) * valid[..., None]
                if "spk_prim_ext" in batch:  # the enrollment column (datasets/ext_tracks.py)
                    xp = batch["spk_prim_ext"].to(act.device).float().argmax(1)
                else:  # older caches: the column that equals spk_act_ext
                    xp = ((xc - (ext.clamp(0, 1) * valid)[..., None]).abs().sum(1)).argmin(1)
                if xn["p"] > 0:
                    ar = torch.arange(T, device=act.device)[None]
                    end = torch.where((oracle > 0.5) & (ar < lens[:, None].to(act.device)), ar, -1).max(1).values + 1
                    xc = ext_track_noise(xc, xp, torch.minimum(elen_x, elen), end, xn["p"], int(xn["swap_max"]),
                                         xn["drop"], int(xn["drop_max"]), xn["at_turn_end"])
                    ext = xc.gather(2, xp[:, None, None].expand(-1, T, 1))[..., 0]
                if with_cols:
                    cols = torch.where(use[:, None, None], _fit_cols(xc, T, nc), cols)
                    prim = torch.where(use, xp.clamp(max=nc - 1), prim)
            act = torch.where(use[:, None], ext.clamp(0, 1) * valid, act)
        rb = self.rebind
        if rb["p"] > 0:  # corrupted-then-corrected binding on the FINAL track (any source)
            if with_cols:
                cols, _ = rebind_corruption(cols, prim, oracle, lens, rb["p"], int(rb["w_min"]), int(rb["w_max"]))
                act = cols.gather(2, prim[:, None, None].expand(-1, T, 1))[..., 0].to(act)
            elif "spk_targets" in batch and batch["spk_targets"].shape[-1] > 1:
                tc = _fit_cols(batch["spk_targets"].float().to(act.device), T, batch["spk_targets"].shape[-1])
                tc[..., 0] = act
                tc, _ = rebind_corruption(tc, torch.zeros(act.shape[0], dtype=torch.long, device=act.device), oracle, lens,
                                          rb["p"], int(rb["w_min"]), int(rb["w_max"]))
                act = tc[..., 0]
        act = act.to(batch["spk_act"].dtype if batch["spk_act"].is_floating_point() else torch.float)
        return (act, cols, prim) if with_cols else act

    def forward(self, batch: dict, att_context_size=None, return_enc: bool = False) -> dict:
        batch = dict(batch)  # derived keys (_asr_enc, spk_act_oracle) never leak into the caller's batch
        feats, flen = self.features(batch["audio"], batch["audio_len"])
        # one context size per step, shared by both encoder passes (multi-lookahead training)
        att = att_context_size or (random.choice(self.encoder.att_context_sizes) if self.training
                                   else self.encoder.att_context_size)
        enc, elen, hidden = self.encoder(feats, flen, att, return_hidden=True)
        enc_spk = None
        if (self.training and self.cond_on and "spk_act" in batch
                and (any(hc.get("condition_on_speaker") for hc in self.head_cfg.values()) or self.head_reads_act)):
            # clean track kept for the labels (TurnHead reads spk_act_oracle), noisy one conditions the encoder
            # (condition_on_speaker heads) and / or is read by the head itself (head-only, v5: no second pass)
            batch.setdefault("spk_act_oracle", batch["spk_act"])
            if self.n_act_columns > 1:  # TurnHead(act_columns > 1) reads the same track's columns
                batch["spk_act"], batch["spk_cols"], batch["spk_prim"] = self.conditioning_act(
                    batch, enc, elen, hidden, att, with_cols=True)
            else:
                batch["spk_act"] = self.conditioning_act(batch, enc, elen, hidden, att)
        out, total, skipped = {}, 0.0, []
        for name, head in self.heads.items():
            hc = self.head_cfg[name]
            needed = head.key if hasattr(head, "key") else None
            if needed and needed not in batch:
                continue  # multi-task batches: skip heads without labels
            if float(hc.get("weight", 1.0)) == 0:  # skipped below anyway: never pay for its conditioned pass
                skipped.append(name)
                continue
            e = self.head_input(name, enc, hidden)
            if hc.get("condition_on_speaker"):
                if "spk_act" not in batch:
                    continue
                if enc_spk is None:
                    enc_spk, _, hid_spk = self.encoder(feats, flen, att, spk_act=batch["spk_act"][:, : enc.shape[1]],
                                                       return_hidden=True)
                    batch["_asr_enc"] = enc_spk.detach()  # TurnHead.asr_view reuses it (no extra no-grad pass)
                # from_layers on a speaker-conditioned head: its tap / mix of the CONDITIONED pass (default: top)
                e = self.head_input(name, enc_spk, hid_spk)
            w = float(hc.get("weight", 1.0))
            if w == 0 or (torch.is_grad_enabled() and not e.requires_grad
                          and not any(p.requires_grad for p in head.parameters())):
                # the loss could not produce a gradient (weight 0, or this head and everything upstream of
                # it frozen, e.g. the pretrained RNNT/CTC over a frozen encoder). Under no_grad (evaluation)
                # frozen heads are still reported; only weight-0 heads are skipped.
                skipped.append(name)
                continue
            if "grad_scale" in hc:  # auxiliary heads: learn fully, but nudge the shared encoder gently
                e = GradScale.apply(e, float(hc["grad_scale"]))
            l = head.loss(e, elen, batch)
            out[f"loss_{name}"] = l
            total = total + w * l
        if skipped and tuple(skipped) not in self._skip_logged:
            self._skip_logged.add(tuple(skipped))
            log.info(f"[model] skipping loss for heads {skipped}: weight 0 or no trainable parameter "
                  f"(head + upstream frozen)")
        out["loss"] = total
        if return_enc:  # for the trainer's KL terms (losses/consistency.py): same features, same encoding
            out["enc"] = {"feats": feats, "flen": flen, "enc": enc, "elen": elen, "hidden": hidden, "att": att}
        return out

    # ------------------------------------------------------------------ inference
    def _pad(self, audios):
        dev = next(self.parameters()).device
        lens = torch.tensor([len(a) for a in audios], device=dev)
        x = torch.zeros(len(audios), int(lens.max()), device=dev)
        for i, a in enumerate(audios):
            x[i, : len(a)] = torch.as_tensor(a, device=dev)
        return x, lens

    @torch.no_grad()
    def transcribe(self, audios, head: str | None = None, prompt: str | None = None,
                   att_context_size=None, spk_act=None) -> list[str]:
        """audios: list of 1-D float arrays at ``sample_rate``. prompt: AED task string."""
        self.eval()
        if len(audios) == 0:
            return []
        head = head or self.primary
        x, lens = self._pad(audios)
        enc, elen, hidden = self.encode(x, lens, att_context_size, spk_act, return_hidden=True)
        kw = {}
        if isinstance(self.heads[head], AEDHead) and prompt:
            kw["prompt"] = [self.tokenizer.encode(prompt)] * len(audios)
        # from_layers: all -> decode from the head's layer mix, as analyze() and training do
        # (condition_on_speaker heads train on the top speaker-conditioned layer, so they keep enc)
        e = enc if self.head_cfg[head].get("condition_on_speaker") else self.head_input(head, enc, hidden)
        ids = self.heads[head].decode(e, elen, **kw)
        return [self.tokenizer.decode(i) for i in ids]

    @torch.no_grad()
    def analyze(self, audios, att_context_size=None) -> dict:
        """Run every head once on a shared encoding (voice-agent front end)."""
        self.eval()
        if len(audios) == 0:
            return {"frames": [], "frame_sec": self.frame_sec}
        x, lens = self._pad(audios)
        enc, elen, hidden = self.encode(x, lens, att_context_size, return_hidden=True)
        res = {"frames": elen.tolist(), "frame_sec": self.frame_sec}
        for name, head in self.heads.items():
            if self.head_cfg[name].get("condition_on_speaker"):
                continue
            out = head.decode(self.head_input(name, enc, hidden), elen)
            if self.head_cfg[name]["type"] in TEXT_HEADS:
                out = [self.tokenizer.decode(i) for i in out]
            res[name] = out
        return res

    @torch.no_grad()
    def transcribe_speakers(self, audio, diar_head: str = "diar", asr_head: str | None = None,
                            threshold: float = 0.5) -> list[dict]:
        """Speaker-attributed ASR: diarize, then re-encode once per speaker with speaker kernels."""
        self.eval()
        asr_head = asr_head or next(k for k, v in self.head_cfg.items() if v.get("condition_on_speaker"))
        x, lens = self._pad([audio])
        enc, elen, hidden = self.encode(x, lens, return_hidden=True)
        act = self.heads[diar_head](self.head_input(diar_head, enc, hidden), elen).sigmoid()[0]  # T,S
        out = []
        for s in range(act.shape[1]):
            if (act[:, s] > threshold).sum() == 0:
                continue
            text = self.transcribe([audio], head=asr_head, spk_act=act[None, :, s])[0]
            out.append({"speaker": s, "text": text, "active_sec": float((act[:, s] > threshold).sum()) * self.frame_sec})
        return out

    @property
    def frame_sec(self) -> float:
        return self.preprocessor.hop * self.encoder.subsampling_factor / self.preprocessor.sample_rate

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class StreamingSession:
    """Cache-aware streaming: feed raw audio, get incremental text and frame events.

    Works with CTC, RNNT and TDT heads plus any frame heads (VAD/EOU). Mel
    frames are computed incrementally, so the preprocessor must use
    a frame-local normalization (``fixed`` global stats, or ``none`` as in NeMo).
    """

    cache_joint_pred = False  # audioforge.perf "joint_cache": reuse joint.pred(g) across frames until a token is emitted

    def __init__(self, model: SpeechModel, head: str | None = None, att_context_size=None):
        assert model.preprocessor.normalize in (None, "none", "NA", "fixed"), \
            "streaming needs a frame-local normalization (normalize: fixed or none)"
        self.m = model.eval()
        self.head_name = head or model.primary
        self.head = model.heads[self.head_name]
        self.att = list(att_context_size or model.encoder.att_context_size)
        self.chunk_mel = model.encoder.stream_chunk_frames(self.att)
        # mel frames the FIRST chunk needs. A NeMo-aligned causal encoder (pre_encode.nemo_causal) ends encoder frame v
        # at mel frame 8v, so the chunk of frames [v0, v0 + R] is complete once mel frame 8 (v0 + R) exists: chunk j
        # runs after chunk_lead + j * chunk_mel mel frames (8R + 1, then every (R + 1) * 8), not after a whole
        # (R + 1) * 8 window, which held every frame back by 7 mel frames = 70 ms (research/LATENCY_BUDGET.md). The
        # encoder's outputs are the same (``_stream_step_aligned`` carries the 7 frames over). Other encoders need the
        # full window.
        f = model.encoder.subsampling_factor
        self.chunk_lead = (f * self.att[1] + 1 if getattr(getattr(model.encoder, "pre_encode", None), "nemo_causal",
                                                          False) else self.chunk_mel)
        self.mel_fed = 0  # mel frames handed to the encoder so far
        pp = model.preprocessor
        self.hop, self.half = pp.hop, pp.n_fft // 2
        dev = next(model.parameters()).device
        self.dev = dev
        self.sig = torch.zeros(0, device=dev)  # pre-emphasized signal, trimmed
        self.sig_start = 0  # absolute sample index of sig[0]
        self.last = 0.0
        self._prev = -1  # CTC greedy: last argmax id (collapse repeats across chunks)
        self.mel_done = 0  # absolute mel frames emitted to the encoder
        self.mel_buf = []
        self.enc_state = StreamState()
        self.tokens: list[int] = []
        self.pred = None
        self.skip = 0
        self._pg = None  # (g, joint.pred(g)) of the last decoded frame (cache_joint_pred)
        self.frame_events: dict[str, list] = {k: [] for k, v in model.head_cfg.items() if v["type"] == "frame"}

    def _next_chunk_mels(self) -> int:
        """Mel frames the next encoder chunk takes (``chunk_lead`` for the first, then ``chunk_mel``)."""
        return self.chunk_lead if self.mel_fed == 0 else self.chunk_mel

    def frame_ready_samples(self, v: int) -> int:
        """Samples of audio after which encoder frame ``v`` is computed (its chunk's last mel frame is complete:
        mel frame k needs samples up to k * hop + n_fft / 2)."""
        cs = self.att[1] + 1
        return (self.chunk_lead + (v // cs) * self.chunk_mel - 1) * self.hop + self.half

    def frames_ready(self, samples: int) -> int:
        """Encoder frames computed once ``samples`` of audio have arrived (inverse of ``frame_ready_samples``)."""
        mels = max((int(samples) - self.half) // self.hop + 1, 0)
        chunks = 0 if mels < self.chunk_lead else (mels - self.chunk_lead) // self.chunk_mel + 1
        return chunks * (self.att[1] + 1)

    def _mel(self, a: int, b: int) -> torch.Tensor:
        pp = self.m.preprocessor
        s0, s1 = a * self.hop - self.half, (b - 1) * self.hop + self.half
        seg = torch.zeros(s1 - s0, device=self.dev)
        lo, hi = max(s0, self.sig_start), min(s1, self.sig_start + len(self.sig))
        if hi > lo:
            seg[lo - s0: hi - s0] = self.sig[lo - self.sig_start: hi - self.sig_start]
        spec = torch.stft(seg[None], pp.n_fft, pp.hop, pp.win, pp.window, center=False, return_complex=True)
        return pp.fixed_norm(torch.log(pp.fb @ (spec.real ** 2 + spec.imag ** 2) + 2 ** -24))  # 1,F,b-a

    @torch.no_grad()
    def feed(self, samples, final: bool = False) -> str:
        x = torch.as_tensor(samples, dtype=torch.float32, device=self.dev)
        pe = self.m.preprocessor.preemph
        if len(x):
            prev = torch.cat([torch.tensor([self.last], device=self.dev), x[:-1]])
            self.last = float(x[-1])
            self.sig = torch.cat([self.sig, x - pe * prev if pe else x])
        total = self.sig_start + len(self.sig)
        ready = (total - self.half) // self.hop + 1 if not final else total // self.hop + 1
        if ready > self.mel_done:
            self.mel_buf.append(self._mel(self.mel_done, ready))
            self.mel_done = ready
        keep = self.mel_done * self.hop - self.half - self.hop  # trim old samples
        if keep > self.sig_start:
            self.sig = self.sig[keep - self.sig_start:]
            self.sig_start = keep
        mel = torch.cat(self.mel_buf, -1) if self.mel_buf else torch.zeros(1, 1, 0, device=self.dev)
        # final: run the partial last chunk as is (stream_step supports it) instead of padding it,
        # so no frames past the end of the audio are decoded and the frame count equals offline
        while mel.shape[-1] >= self._next_chunk_mels() or (final and mel.shape[-1]):
            n = self._next_chunk_mels()
            last = bool(final) and mel.shape[-1] <= n  # NeMo-aligned encoders emit their trailing frame
            chunk, mel = mel[..., :n], mel[..., n:]
            self._before_chunk(chunk)
            self.mel_fed += chunk.shape[-1]
            enc, hid, self.enc_state = self.m.encoder.stream_step(chunk, self.enc_state, self.att, final=last,
                                                                  return_hidden=True)
            # heads with from_layers: all read their layer mix of this chunk's per-layer outputs (= analyze())
            self._decode(self.m.head_input(self.head_name, enc, hid)[0])
            for k in self.frame_events:
                e = self.m.head_input(k, enc, hid)
                self.frame_events[k].extend(self.m.heads[k].decode(e, torch.tensor([e.shape[1]]))[0].tolist())
        self.mel_buf = [mel] if mel.shape[-1] else []
        return self.text

    def _before_chunk(self, chunk: torch.Tensor) -> None:
        """Hook: called with each mel chunk right before it is encoded (serve's dual-rate pass snapshots its state
        here, audioforge.server.streams.LookaheadStream)."""

    def _decode(self, f):
        h = self.head
        if isinstance(h, CTCHead):
            best = h(f[None]).argmax(-1)[0].tolist()
            for p in best:
                if p != h.blank and p != self._prev:
                    self.tokens.append(p)
                self._prev = p
            return
        V1, dev = h.vocab_size + 1, f.device
        if self.pred is None:
            self.pred = h.pred(torch.tensor([[h.blank]], device=dev), None)
        g, st = self.pred
        fe = h.joint.enc(f)
        t, emitted = self.skip, 0
        # joint.pred(g) changes only when a token is emitted: keep the last projection with the tensor it came from
        # (audioforge.perf "joint_cache"; the same op on the same tensor, so the sum below is bit-identical)
        pg = self._pg[1] if self.cache_joint_pred and self._pg is not None and self._pg[0] is g else None
        while t < f.shape[0]:
            if pg is None:
                pg = h.joint.pred(g)
                if self.cache_joint_pred:
                    self._pg = (g, pg)
            z = h.joint.out(fe[t][None, None] + pg)[0, 0]
            k = int(z[:V1].argmax())
            d = (h.durations[int(z[V1:].argmax())] if h.is_tdt else (1 if k == h.blank else 0))
            if k == h.blank:
                d = max(d, 1)
            else:
                self.tokens.append(k)
                g, st = h.pred(torch.tensor([[k]], device=dev), st)
                pg = None
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
        return self.m.tokenizer.decode(self.tokens)
