"""Non-text heads on the shared encoder.

* SortformerHead      - Sortformer diarization: transformer + per-speaker sigmoid,
                        trained with the arrival-order *sort loss* (+ optional PIL,
                        exhaustive over permutations, so skipped with a warning above 8 speakers).
* SpeakerHead         - TitaNet-style attentive-statistics pooling + AAM-softmax.
* LanguageHead        - causal spoken-language ID: running attentive-stats posterior (research/archive/LID.md).
* FrameHead           - per-frame classifier (VAD, end-of-utterance, ...).
* CodecTokenHead      - NEW: predicts FSQ codec tokens per encoder frame. The
                        FastConformer frame rate (80 ms = 12.5 Hz) equals NVIDIA's
                        12.5 fps NanoCodec, so tokens align 1:1, no decoder needed.
"""
from __future__ import annotations

import itertools
import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asr import Head


def _pad_mask(enc, enc_len):
    return torch.arange(enc.shape[1], device=enc.device)[None] < enc_len[:, None]  # True = valid


def _match_len(t: torch.Tensor, T: int) -> torch.Tensor:
    """Crop/pad frame labels (B,T0,...) to T frames."""
    if t.shape[1] >= T:
        return t[:, :T]
    pad = torch.zeros((t.shape[0], T - t.shape[1], *t.shape[2:]), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], 1)


# --------------------------------------------------------------------------- diarization
def arrival_order(targets: torch.Tensor) -> torch.Tensor:
    """(B,T,S) activity -> (B,S) column indices sorted by first active frame (stable; silent columns last)."""
    B, T, S = targets.shape
    active = targets > 0.5
    first = torch.where(active.any(1), active.float().argmax(1), torch.full((B, S), T, device=targets.device))
    return first.argsort(dim=1, stable=True)


def sort_by_arrival(targets: torch.Tensor) -> torch.Tensor:
    """Reorder speaker columns by first active frame (Sortformer's arrival-time order).

    targets (B,T,S) in {0,1}; silent speakers go last.
    """
    B, T, S = targets.shape
    return targets.gather(2, arrival_order(targets)[:, None].expand(B, T, S))


class SortformerHead(Head):
    key = "spk_targets"
    MAX_PIL_SPKS = 8  # PIL enumerates num_spks! permutations; skipped (with a warning) above this

    def __init__(self, d_model: int, num_spks: int = 4, d_hidden: int = 192, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, pil_weight: float = 0.5, pos_emb: bool = False,
                 prefix_prob: float = 0.0, prefix_min: int = 8, norm_first: bool = True,
                 out_pre_relu: bool = False, distill: dict | None = None):
        super().__init__()
        self.num_spks, self.pil_weight = num_spks, pil_weight
        # distill: {key: diar_teacher, weight: w} (research/archive/DIARIZATION_FIX.md section 5): the same PIL + sorted-BCE
        # loss against a cached teacher diarizer's per-frame posteriors under batch[key] (T, S_t), mixed with the
        # ground-truth loss as (1 - w) * gt + w * teacher; None (default) leaves the head and its loss unchanged
        self.distill = dict(distill) if distill else None
        # prefix_prob: with this probability per training step, add the same loss on a random prefix
        # enc[:, :L] (L ~ U[prefix_min, T]); arrival order is causal, so the cropped targets re-sorted are
        # the prefix's correct labels. Needed when the offline head is re-run on growing prefixes
        # (heads/turn.py streaming_diar_act): trained on whole episodes only, it outputs ~0 activity on
        # short prefixes (research/archive/TURN_ABLATION.md, A).
        self.prefix_prob, self.prefix_min = prefix_prob, prefix_min
        # pos_emb: sinusoidal positions over the transformer input sequence. Streaming Sortformer
        # (audioforge/streaming_diar.py) relies on it: the arrival-order speaker cache is laid out
        # speaker 0 first, so sequence position is what tells the head which cached speaker is "first".
        self.pos_emb = pos_emb
        # Layout knobs for hosting NVIDIA's pretrained Sortformer (audioforge/nemo_import.py); the defaults
        # are this head's own layout. norm_first=False: post-LN blocks (NeMo transformer_encoder pre_ln:
        # false). out_pre_relu=True: ReLU before the first output Linear (NeMo
        # SortformerModules.forward_speaker_logits: relu -> first_hidden_to_hidden -> relu -> single_hidden_to_spks).
        self.proj = nn.Linear(d_model, d_hidden)
        layer = nn.TransformerEncoderLayer(d_hidden, n_heads, 4 * d_hidden, dropout, batch_first=True,
                                           norm_first=norm_first)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        out = [nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, num_spks)]
        self.out = nn.Sequential(*([nn.ReLU()] + out if out_pre_relu else out))

    def embed_frames(self, enc):
        """Per-frame embedding the transformer consumes (B,T,d_hidden); what the speaker cache/FIFO store."""
        return self.proj(enc)

    def _add_pos(self, x):
        if not self.pos_emb:
            return x
        T, D = x.shape[1], x.shape[2]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
        inv = torch.exp(torch.arange(0, D, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe = torch.zeros(T, D, device=x.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * inv), torch.cos(pos * inv)[:, : D // 2]
        return x + pe.to(x.dtype)[None]

    def forward_emb(self, x, valid=None):
        """Logits (B,N,S) from pre-embedded frames x (B,N,d_hidden); valid (B,N) True = real frame."""
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return self.out(self.tf(self._add_pos(x), src_key_padding_mask=~valid))

    def forward_chunk(self, spkcache, fifo, chunk, right_context=None):
        """Streaming step: run the transformer over [spkcache ‖ fifo ‖ chunk ‖ right_context].

        Every input is (B,N_i,d_hidden) pre-embedded frames (``embed_frames``); empty parts allowed.
        Returns logits for the whole sequence (B,N,S); the caller slices the chunk / cache / fifo parts.
        """
        parts = [p for p in (spkcache, fifo, chunk, right_context) if p is not None and p.shape[1] > 0]
        return self.forward_emb(torch.cat(parts, 1))

    def forward(self, enc, enc_len):
        return self.forward_emb(self.embed_frames(enc), _pad_mask(enc, enc_len))  # logits (B,T,S)

    def _prefix_len(self, T: int) -> int:
        return random.randint(min(self.prefix_min, T), T)

    def loss(self, enc, enc_len, batch):
        tch = batch.get(self.distill["key"]) if self.distill else None

        def on(e, el, sl):
            l_gt = self._loss_on(e, el, batch["spk_targets"][:, sl])
            if tch is None:
                return l_gt
            w = float(self.distill.get("weight", 0.5))
            return (1 - w) * l_gt + w * self._loss_on(e, el, tch[:, sl].float())

        full = on(enc, enc_len, slice(None))
        if not (self.training and self.prefix_prob > 0):
            return full
        if self.prefix_prob < 1 and random.random() >= self.prefix_prob:
            return full
        L = self._prefix_len(enc.shape[1])
        pre = on(enc[:, :L], enc_len.clamp(max=L), slice(0, L))
        return 0.5 * (full + pre)

    def _loss_on(self, enc, enc_len, spk_targets):
        logits = self(enc, enc_len).float()
        S = self.num_spks
        # Labels may carry a different number of speaker columns than the head (data producers emit 4):
        # sort by arrival first so cropping keeps the earliest speakers; missing slots are silent.
        sort_t = sort_by_arrival(_match_len(spk_targets.float(), logits.shape[1]))
        if sort_t.shape[2] >= S:
            sort_t = sort_t[..., :S]
        else:
            sort_t = torch.cat([sort_t, sort_t.new_zeros(*sort_t.shape[:2], S - sort_t.shape[2])], -1)
        valid = _pad_mask(enc, enc_len)[..., None].float()
        l_sort = (F.binary_cross_entropy_with_logits(logits, sort_t, reduction="none") * valid).sum() / (
            valid.sum() * S)
        if self.pil_weight <= 0:
            return l_sort
        if S > self.MAX_PIL_SPKS:
            if not getattr(self, "_warned_pil", False):
                warnings.warn(f"SortformerHead: pil_weight ignored for num_spks={S} > {self.MAX_PIL_SPKS}", stacklevel=2)
                self._warned_pil = True
            return l_sort
        # cost[b,i,j] = masked BCE of output i vs target j (PIL is order-free, so sorted targets are fine)
        bce = F.binary_cross_entropy_with_logits(logits[..., :, None].expand(-1, -1, S, S),
                                                 sort_t[..., None, :].expand(-1, -1, S, S), reduction="none")
        cost = (bce * valid[..., None]).sum(1) / (valid.sum((1, 2)) * S)[:, None, None]  # (B,S,S)
        perms = torch.tensor(list(itertools.permutations(range(S))), device=cost.device)  # (P,S)
        rows = torch.arange(S, device=cost.device).expand_as(perms)
        best = cost[:, rows, perms].sum(-1).min(1).values  # (B,)
        return (1 - self.pil_weight) * l_sort + self.pil_weight * best.mean()

    @torch.no_grad()
    def decode(self, enc, enc_len, threshold: float = 0.5, **_):
        return (self(enc, enc_len).sigmoid() > threshold).float()


# --------------------------------------------------------------------------- speaker embedding
class AttentiveStatsPool(nn.Module):
    def __init__(self, d: int, bottleneck: int = 128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(3 * d, bottleneck), nn.Tanh(), nn.Linear(bottleneck, d))

    def forward(self, x, valid):
        m = valid[..., None].float()
        n = m.sum(1, keepdim=True).clamp(min=1)
        mean = (x * m).sum(1, keepdim=True) / n
        std = (((x - mean) ** 2 * m).sum(1, keepdim=True) / n).clamp(min=1e-6).sqrt()
        ctx = torch.cat([x, mean.expand_as(x), std.expand_as(x)], -1)
        w = self.att(ctx).masked_fill(~valid[..., None], -1e4).softmax(1)
        mu = (w * x).sum(1)
        sd = ((w * x * x).sum(1) - mu * mu).clamp(min=1e-6).sqrt()
        return torch.cat([mu, sd], -1)


class SpeakerHead(Head):
    """Attentive-stats pooling -> ``emb_dim`` unit vector; trained with AAM-softmax over ``num_speakers`` ids
    (``batch["speaker"]``) and / or distilled onto a teacher embedding (research/archive/SPK_HEAD.md):
    ``distill: {target: spk_teacher, weight: 1.0}`` adds ``weight * mean(1 - cos(student, teacher))`` against
    ``batch[target]`` (any dimension: an ``emb_dim``-d unit vector is expected, e.g. TitaNet-L's 192-d) and
    makes ``target`` the head's label key; ``distill.relational_weight`` adds the MSE between the student's and
    the teacher's within-batch cosine-similarity matrices (relational distillation, works across dimensions); ``aam_weight`` (default 1.0) scales the AAM term, which is then
    only applied to batches that carry ``speaker``. Without ``distill`` the loss is the plain AAM loss."""

    key = "speaker"

    def __init__(self, d_model: int, num_speakers: int, emb_dim: int = 192, margin: float = 0.2,
                 scale: float = 30.0, distill: dict | None = None, aam_weight: float = 1.0, hidden: int = 0):
        super().__init__()
        self.pool = AttentiveStatsPool(d_model)
        # hidden > 0: a two-layer projection (research/FIXALL.md step 4); 0 = the shipped single linear layer
        self.emb = (nn.Sequential(nn.Linear(2 * d_model, hidden), nn.SiLU(), nn.Linear(hidden, emb_dim),
                                  nn.BatchNorm1d(emb_dim)) if hidden else
                    nn.Sequential(nn.Linear(2 * d_model, emb_dim), nn.BatchNorm1d(emb_dim)))
        self.W = nn.Parameter(torch.randn(num_speakers, emb_dim) * 0.01)
        self.m, self.s = margin, scale
        self.distill = dict(distill) if distill else None
        self.aam_weight = float(aam_weight)
        if self.distill:
            self.key = str(self.distill.get("target", "spk_teacher"))

    def embed(self, enc, enc_len):
        return F.normalize(self.emb(self.pool(enc, _pad_mask(enc, enc_len))), dim=-1)

    def aam_loss(self, e, y):
        cos = e @ F.normalize(self.W, dim=-1).T
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)  # additive angular margin (ArcFace / AAM-softmax)
        logits = torch.where(F.one_hot(y, cos.shape[1]).bool(), target, cos) * self.s
        return F.cross_entropy(logits.float(), y)

    def distill_loss(self, e, batch):
        t = F.normalize(batch[self.key].float().reshape(e.shape[0], -1), dim=-1)
        return (1.0 - (e * t).sum(-1)).mean()

    def relational_loss(self, e, batch):
        """Relational distillation (MSA-ASR style): MSE between the student's and the teacher's within-batch
        cosine-similarity matrices (off-diagonal pairs; the diagonal is 1 for both)."""
        t = F.normalize(batch[self.key].float().reshape(e.shape[0], -1), dim=-1)
        B = e.shape[0]
        if B < 2:
            return e.sum() * 0.0
        off = ~torch.eye(B, dtype=torch.bool, device=e.device)
        return F.mse_loss((e @ e.T)[off], (t @ t.T)[off])

    def loss(self, enc, enc_len, batch):
        e = self.embed(enc, enc_len)
        if self.distill is None:
            l = self.aam_loss(e, batch["speaker"])
            return l if self.aam_weight == 1.0 else self.aam_weight * l
        cw, rw = float(self.distill.get("weight", 1.0)), float(self.distill.get("relational_weight", 0.0))
        l = e.sum() * 0.0  # keeps a gradient path when every term is off for this batch
        if cw > 0:  # cosine term: needs the teacher's dimension == emb_dim
            l = l + cw * self.distill_loss(e, batch)
        if rw > 0:  # relational term: any teacher dimension
            l = l + rw * self.relational_loss(e, batch)
        if self.aam_weight > 0 and "speaker" in batch:
            l = l + self.aam_weight * self.aam_loss(e, batch["speaker"])
        return l

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return self.embed(enc, enc_len)


# --------------------------------------------------------------------------- spoken language identification
class LanguageHead(Head):
    """Causal spoken-language ID (research/archive/LID.md): per-frame MLP -> attentive-statistics pooling over *all frames so
    far* -> classifier, i.e. a running posterior that can be read after any encoder frame.

    The pooling weights are per frame and per channel, ``w_t = exp(5 tanh(a(h_t) / 5))`` (bounded, so the running sums
    need no max-subtraction), and depend on frame t only - no utterance-level context as in ``AttentiveStatsPool`` - so
    the pooled mean / std after frame t are sums over frames <= t: ``running_logits`` (offline, cumulative sums) and
    ``step`` (streaming, carried sums) compute the same numbers. Labels: ``batch["lang"]`` (class index);
    ``loss`` = cross-entropy of the running posterior averaged over every valid frame from ``min_frames`` on (an
    anytime classifier) plus the cross-entropy at the last valid frame. ``labels`` (language codes) are carried in the
    config for serving."""

    key = "lang"
    A_MAX = 5.0

    def __init__(self, d_model: int, num_languages: int, hidden: int = 256, att_hidden: int = 128,
                 cls_hidden: int = 256, dropout: float = 0.1, min_frames: int = 6, labels: list | None = None,
                 context: int = 0, rnn: int = 0):
        super().__init__()
        self.num_languages, self.min_frames = num_languages, int(min_frames)
        # ``context`` > 0 (research/archive/LID.md fix pass): a causal depthwise-separable conv over the last ``context`` + 1
        # per-frame features (residual), so each pooled frame sees a short left context; streaming carries the last
        # ``context`` frames. 0 = the original head (no extra tensors).
        self.context = int(context)
        # ``rnn`` > 0: a causal GRU (that many units) over the per-frame features, added back through a linear map
        # (residual), so pooled frames carry sequence (phonotactic) context; streaming carries the GRU state.
        self.rnn = int(rnn)
        self.labels = list(labels) if labels else [str(i) for i in range(num_languages)]
        assert len(self.labels) == num_languages, "labels must have num_languages entries"
        self.frame = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.att = nn.Sequential(nn.Linear(hidden, att_hidden), nn.Tanh(), nn.Linear(att_hidden, hidden))
        self.cls = nn.Sequential(nn.Linear(2 * hidden, cls_hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(cls_hidden, num_languages))
        if self.context:
            self.tconv_dw = nn.Conv1d(hidden, hidden, self.context + 1, groups=hidden)
            self.tconv_pw = nn.Conv1d(hidden, hidden, 1)
        if self.rnn:
            self.gru = nn.GRU(hidden, self.rnn, batch_first=True)
            self.gru_out = nn.Linear(self.rnn, hidden)

    def _frames(self, x, buf=None, hstate=None):
        """Per-frame features (B, T, hidden), the new conv left-context buffer and the new GRU state (None when the
        option is off)."""
        h = self.frame(x).float()
        nbuf = None
        if self.context:
            if buf is None:
                buf = h.new_zeros(h.shape[0], self.context, h.shape[2])
            hp = torch.cat([buf, h], 1)
            h = h + self.tconv_pw(F.relu(self.tconv_dw(hp.transpose(1, 2)))).transpose(1, 2)
            nbuf = hp[:, -self.context:]
        nh = None
        if self.rnn:
            r, nh = self.gru(h, hstate)
            h = h + self.gru_out(r)
        return h, nbuf, nh

    def _pool_terms(self, h):
        w = torch.exp(self.A_MAX * torch.tanh(self.att(h) / self.A_MAX))
        return w, w * h, w * h * h

    def _terms(self, x):
        """Per-frame pooling terms (w, w h, w h^2), each (B, T, hidden)."""
        return self._pool_terms(self._frames(x)[0])

    def _stream_slots(self):
        """Indices of the extra streaming state (conv buffer, GRU state) after the three pooled sums."""
        k = 3
        c = k if self.context else None
        k += bool(self.context)
        g = k if self.rnn else None
        return c, g

    def _classify(self, sw, swx, swxx):
        sw = sw.clamp(min=1e-6)
        mu = swx / sw
        sd = (swxx / sw - mu * mu).clamp(min=1e-6).sqrt()
        return self.cls(torch.cat([mu, sd], -1))

    def running_logits(self, enc, enc_len):
        """(B, T, L) logits of the posterior after each frame (padding frames repeat the last valid one)."""
        w, wx, wxx = self._terms(enc)
        m = _pad_mask(enc, enc_len)[..., None].float()
        return self._classify((w * m).cumsum(1), (wx * m).cumsum(1), (wxx * m).cumsum(1))

    def forward(self, enc, enc_len):
        """(B, L) logits after the last valid frame (the whole-utterance decision)."""
        w, wx, wxx = self._terms(enc)
        m = _pad_mask(enc, enc_len)[..., None].float()
        return self._classify((w * m).sum(1), (wx * m).sum(1), (wxx * m).sum(1))

    def init_stream(self, batch: int = 1, device=None):
        z = torch.zeros(batch, self.cls[0].in_features // 2, device=device)
        st = [z, z.clone(), z.clone()]
        if self.context:  # left-context buffer of the causal conv
            st.append(torch.zeros(batch, self.context, z.shape[1], device=device))
        if self.rnn:  # GRU state
            st.append(torch.zeros(1, batch, self.rnn, device=device))
        return st

    def step(self, x, state, keep=None, decay: float = 1.0):
        """Streaming update with a chunk of frames x (B, n, D); ``state`` (from ``init_stream``) is updated in place.
        -> (B, n, L) logits after each of the n frames (== running_logits on the concatenated stream).
        Serving options (not used in training): ``keep`` (B, n) bool - frames with False are not pooled (e.g. VAD
        gating; the posterior is held); ``decay`` < 1 - the running sums are multiplied by ``decay`` per pooled frame
        (exponential forgetting, so a long session can follow a language change)."""
        if self.context or self.rnn:
            c, g = self._stream_slots()
            h, nbuf, nh = self._frames(x, state[c] if c is not None else None, state[g] if g is not None else None)
            if c is not None:
                state[c] = nbuf
            if g is not None:
                state[g] = nh
            w, wx, wxx = self._pool_terms(h)
        else:
            w, wx, wxx = self._terms(x)
        if keep is not None:
            k = keep.to(w.dtype)[..., None]
            w, wx, wxx = w * k, wx * k, wxx * k
        if decay == 1.0:
            cw, cx, cxx = (state[0][:, None] + w.cumsum(1), state[1][:, None] + wx.cumsum(1),
                           state[2][:, None] + wxx.cumsum(1))
        else:
            k = torch.ones(w.shape[:2], device=w.device) if keep is None else keep.to(w.dtype)
            outs, s = [], list(state)
            for j in range(w.shape[1]):
                f = 1.0 - (1.0 - decay) * k[:, j:j + 1]  # decay only on pooled frames
                s = [s[0] * f + w[:, j], s[1] * f + wx[:, j], s[2] * f + wxx[:, j]]
                outs.append(s)
            cw, cx, cxx = (torch.stack([o[i] for o in outs], 1) for i in range(3))
        state[0], state[1], state[2] = cw[:, -1], cx[:, -1], cxx[:, -1]
        return self._classify(cw, cx, cxx)

    def loss(self, enc, enc_len, batch):
        y = batch[self.key].long()
        z = self.running_logits(enc, enc_len)
        B, T, L = z.shape
        t = torch.arange(T, device=enc.device)[None]
        use = (t < enc_len[:, None]) & (t >= torch.clamp(enc_len[:, None] - 1, max=self.min_frames - 1))
        ce = F.cross_entropy(z.reshape(-1, L), y[:, None].expand(B, T).reshape(-1), reduction="none").view(B, T)
        anytime = (ce * use).sum(1) / use.sum(1).clamp(min=1)
        last = ce.gather(1, (enc_len - 1).clamp(min=0)[:, None])[:, 0]
        return 0.5 * (anytime.mean() + last.mean())

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return self(enc, enc_len).softmax(-1)


# --------------------------------------------------------------------------- frame-level
class FrameHead(Head):
    """Per-encoder-frame classifier. num_classes=1 -> sigmoid/BCE (VAD, EOU)."""

    def __init__(self, d_model: int, key: str = "vad", num_classes: int = 1, hidden: int = 0,
                 pos_weight: float = 1.0):
        super().__init__()
        self.key, self.num_classes, self.pos_weight = key, num_classes, pos_weight
        self.net = (nn.Sequential(nn.Linear(d_model, hidden), nn.SiLU(), nn.Linear(hidden, num_classes))
                    if hidden else nn.Linear(d_model, num_classes))

    def forward(self, enc):
        z = self.net(enc)
        return z.squeeze(-1) if self.num_classes == 1 else z

    def loss(self, enc, enc_len, batch):
        z = self(enc).float()
        tgt = _match_len(batch[self.key], z.shape[1])
        valid = _pad_mask(enc, enc_len)
        if self.num_classes == 1:
            pw = torch.tensor(self.pos_weight, device=z.device)
            l = F.binary_cross_entropy_with_logits(z, tgt.float(), reduction="none", pos_weight=pw)
            return (l * valid).sum() / valid.sum()
        return F.cross_entropy(z[valid], tgt[valid].long())

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        z = self(enc)
        return z.sigmoid() if self.num_classes == 1 else z.softmax(-1)


class FrameGRUHead(Head):
    """Causal per-frame classifier with a small recurrent state (``type: frame_gru``; the 0.6B core's VAD head,
    research/CORE_0P6B.md): Linear(d_model, hidden)-SiLU-GRU(hidden)-Linear(hidden, 1). ``forward`` runs a whole
    sequence from a zero state; ``init_stream`` / ``step`` run the same GRU chunk by chunk (equal to ``forward`` on
    the concatenation), so a streaming session keeps one hidden vector per head."""

    def __init__(self, d_model: int, key: str = "vad", hidden: int = 64, pos_weight: float = 1.0):
        super().__init__()
        self.key, self.num_classes, self.pos_weight = key, 1, pos_weight
        self.inp = nn.Linear(d_model, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, enc, state=None):
        h, hn = self.rnn(F.silu(self.inp(enc)), None if state is None else state.get("h"))
        if state is not None:
            state["h"] = hn
        return self.out(h).squeeze(-1)

    def init_stream(self, batch: int = 1):
        return {"h": None}

    def step(self, enc, state):
        """enc (B, t, D) of the next frames -> logits (B, t); updates ``state`` in place."""
        return self(enc, state)

    def loss(self, enc, enc_len, batch):
        z = self(enc).float()
        tgt = _match_len(batch[self.key], z.shape[1])
        valid = _pad_mask(enc, enc_len)
        pw = torch.tensor(self.pos_weight, device=z.device)
        l = F.binary_cross_entropy_with_logits(z, tgt.float(), reduction="none", pos_weight=pw)
        return (l * valid).sum() / valid.sum()

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return self(enc).sigmoid()


class CodecTokenHead(Head):
    """Non-autoregressive prediction of K codebook indices per frame.

    Targets ``codes`` (B,T,K). Frame rates must match (e.g. FastConformer 12.5 Hz
    <-> NanoCodec 12.5 fps); ``upsample`` handles integer rate ratios: each encoder frame is
    repeated ``upsample`` times and a learned sub-frame embedding tells the copies apart.
    """

    key = "codes"

    def __init__(self, d_model: int, num_codebooks: int, codebook_size: int, upsample: int = 1,
                 hidden: int = 512, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.K, self.V, self.up = num_codebooks, codebook_size, upsample
        self.inp = nn.Linear(d_model, hidden)
        if upsample > 1:  # breaks the symmetry between the repeated copies of a frame
            self.sub = nn.Parameter(torch.randn(upsample, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, n_heads, 4 * hidden, 0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False) if n_layers else nn.Identity()
        self.out = nn.Linear(hidden, num_codebooks * codebook_size)

    def forward(self, enc, enc_len):
        x = enc.repeat_interleave(self.up, dim=1) if self.up > 1 else enc
        valid = _pad_mask(x, enc_len * self.up)
        h = self.inp(x)
        if self.up > 1:
            h = h + self.sub.repeat(enc.shape[1], 1).to(h.dtype)[None]
        h = self.tf(h, src_key_padding_mask=~valid) if isinstance(self.tf, nn.TransformerEncoder) else h
        B, T, _ = h.shape
        return self.out(h).view(B, T, self.K, self.V), valid

    def loss(self, enc, enc_len, batch):
        logits, valid = self(enc, enc_len)
        tgt = _match_len(batch["codes"].long(), logits.shape[1])
        valid = valid & (torch.arange(logits.shape[1], device=enc.device)[None] < batch["codes_len"][:, None])
        return F.cross_entropy(logits[valid].float().reshape(-1, self.V), tgt[valid].reshape(-1))

    @torch.no_grad()
    def decode(self, enc, enc_len, **_):
        return self(enc, enc_len)[0].argmax(-1)
