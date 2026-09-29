"""KL consistency between transducer joints: frozen-teacher anchor (R2 forgetting guard) and
mode-consistency regularization (MCR-RNNT, arXiv 2604.19079).

Both compare two full RNNT/TDT joint distributions over the same (t, u) lattice of one utterance:
    anchor: student joint vs. a frozen copy of the pretrained head+encoder (same input, same context)
    MCR:    the same head on two encodings of one batch (e.g. offline [-1,-1] vs. streaming [70,1])
Per utterance the KL is averaged over valid lattice cells (t < T_b, u <= U_b), then over the batch,
as the paper does ("normalizing consistency over valid lattice elements").

Memory: ``head_kl`` follows the RNNT head's ``fused_batch_size`` path. With it > 0 the two joints are
built only for one sub-batch at a time, cropped to that sub-batch's max T/U, and wrapped in activation
checkpointing, so backward keeps only the (small) encoder/prediction inputs and recomputes the joints.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

TRANSDUCERS = ("rnnt", "tdt")


def _log_probs(z: torch.Tensor, n_tokens: int | None) -> list[torch.Tensor]:
    """Raw joint output -> log-distributions. TDT: tokens and durations are separate softmaxes."""
    z = z.float()
    if n_tokens is None or n_tokens >= z.shape[-1]:
        return [z.log_softmax(-1)]
    return [z[..., :n_tokens].log_softmax(-1), z[..., n_tokens:].log_softmax(-1)]


def _kl(lp_p: torch.Tensor, lp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) over the last dim, from log-probabilities."""
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)


def transducer_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, enc_len: torch.Tensor,
                  tgt_len: torch.Tensor, symmetric: bool = True, n_tokens: int | None = None,
                  detach_targets: bool = False, reduction: str = "mean") -> torch.Tensor:
    """KL between two joint outputs (B,T,U+1,V[+durations]) on the valid lattice.

    symmetric=False: KL(teacher || student) (teacher distribution as target, as in distillation and
    the paper's eq. 3). symmetric=True: 0.5 * [KL(t||s) + KL(s||t)] (paper eq. 4).
    ``n_tokens``: V+1 for TDT heads, so the duration logits get their own softmax (KLs are summed).
    ``detach_targets``: stop the gradient through the target side of each direction (CR-CTC style);
    the value is unchanged. reduction: "mean" (batch mean of per-utterance means) or "none" (B,).
    """
    B, T, U1 = student_logits.shape[:3]
    kl = 0.0
    for ls, lt in zip(_log_probs(student_logits, n_tokens), _log_probs(teacher_logits, n_tokens)):
        k = _kl(lt.detach() if detach_targets else lt, ls)
        if symmetric:
            k = 0.5 * (k + _kl(ls.detach() if detach_targets else ls, lt))
        kl = kl + k
    dev = student_logits.device
    mask = ((torch.arange(T, device=dev)[None] < enc_len.to(dev)[:, None])[:, :, None]
            & (torch.arange(U1, device=dev)[None] <= tgt_len.to(dev)[:, None])[:, None, :])
    per = torch.where(mask, kl, torch.zeros_like(kl)).sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)
    return per if reduction == "none" else per.mean()


def head_kl(student_head, student_enc: torch.Tensor, teacher_head, teacher_enc: torch.Tensor,
            enc_len: torch.Tensor, y: torch.Tensor, yl: torch.Tensor, symmetric: bool = True,
            detach_targets: bool = False, fused_batch_size: int | None = None) -> torch.Tensor:
    """Build both heads' joints for targets ``y`` and return the masked KL (scalar).

    The heads may be the same module (MCR: two encodings) or student/frozen teacher (anchor).
    ``fused_batch_size`` defaults to the student head's own setting (0 = whole batch at once).
    """
    fbs = student_head.fused_batch_size if fused_batch_size is None else fused_batch_size
    n_tok = student_head.vocab_size + 1 if student_head.is_tdt else None
    sos = student_head.pred.prepend_sos(y)
    gs, _ = student_head.pred(sos)
    if any(p.requires_grad for p in teacher_head.parameters()):
        gt, _ = teacher_head.pred(sos)
    else:
        with torch.no_grad():
            gt, _ = teacher_head.pred(sos)

    def one(se, sg, te, tg, el, ul):
        return transducer_kl(student_head.joint(se, sg), teacher_head.joint(te, tg), el, ul, symmetric,
                             n_tok, detach_targets, reduction="none")

    if not fbs:
        return one(student_enc, gs, teacher_enc, gt, enc_len, yl).mean()
    per = []
    for i in range(0, student_enc.shape[0], fbs):
        s = slice(i, i + fbs)
        T, U = int(enc_len[s].max()), int(yl[s].max())
        args = (student_enc[s, :T], gs[s, :U + 1], teacher_enc[s, :T], gt[s, :U + 1], enc_len[s], yl[s])
        per.append(checkpoint(one, *args, use_reentrant=False) if torch.is_grad_enabled() else one(*args))
    return torch.cat(per).mean()


class FrozenTeacher(nn.Module):
    """Frozen copy of a model's encoder + named transducer heads (eval mode, no grad).

    Built from the live model at trainer start (= the ``init.from`` weights) or from an .afm path.
    It is kept outside the student module, so it is neither optimized nor saved in checkpoints.
    """

    def __init__(self, model, heads: list[str] | None = None):
        super().__init__()
        heads = list(heads or [k for k, v in model.head_cfg.items() if v["type"] in TRANSDUCERS])
        bad = [k for k in heads if model.head_cfg.get(k, {}).get("type") not in TRANSDUCERS]
        if bad:
            raise ValueError(f"FrozenTeacher: {bad} are not transducer (rnnt/tdt) heads of the model")
        self.head_cfg = {k: dict(model.head_cfg[k]) for k in heads}
        self.preprocessor = copy.deepcopy(model.preprocessor)
        self.encoder = copy.deepcopy(model.encoder)
        self.heads = nn.ModuleDict({k: copy.deepcopy(model.heads[k]) for k in heads})
        self.layer_mix = nn.ParameterDict({k: nn.Parameter(model.layer_mix[k].detach().clone())
                                           for k in heads if k in model.layer_mix})
        self.layer_tap = {k: list(v) for k, v in getattr(model, "layer_tap", {}).items() if k in heads}
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_afm(cls, path: str, heads: list[str] | None = None, device="cpu") -> "FrozenTeacher":
        from ..train import load_model
        return cls(load_model(path), heads).to(device)

    def train(self, mode: bool = True):  # always eval: no dropout, deterministic targets
        return super().train(False)

    @torch.no_grad()
    def encode(self, feats, flen, att_context_size=None):
        """Teacher encoding of the student's (already augmented) features, same attention context."""
        return self.encoder(feats, flen, att_context_size or self.encoder.att_context_size, return_hidden=True)

    def head_input(self, name, enc, hidden):
        if name in self.layer_tap:
            idx = self.layer_tap[name]
            if len(idx) == 1:
                return hidden[idx[0]]
            w = self.layer_mix[name].softmax(0)
            return sum(wi * hidden[i] for wi, i in zip(w, idx))
        if name not in self.layer_mix:
            return enc
        w = self.layer_mix[name].softmax(0)
        return sum(wi * h for wi, h in zip(w, hidden))

    @torch.no_grad()
    def joint_logits(self, name: str, feats, flen, y, att_context_size=None):
        """On-demand teacher joint (B,T,U+1,n_out) for a batch (whole batch; see head_kl for sub-batching)."""
        enc, elen, hidden = self.encode(feats, flen, att_context_size)
        head = self.heads[name]
        g, _ = head.pred(head.pred.prepend_sos(y))
        return head.joint(self.head_input(name, enc, hidden), g), elen
