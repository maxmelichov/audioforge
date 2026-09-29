"""User-channel predictive turn head (research/IMPROVEMENTS.md section 1).

The product input: the user's OWN audio channel goes through the frozen streaming encoder (the pass the ASR already
makes, so the head costs no extra encoder compute), and the agent's activity is known exactly from its TTS timeline.
Per 80 ms frame the head reads

* the encoder's top-layer output on the user channel (512),
* two activity columns: the user's (Silero on the user channel, soft per-frame probability) and the agent's
  (TTS state in a product; the other party's channel activity on recorded two-party audio),
* causal duration counters of both columns (frames since last active, current run length; log-clipped),
* optionally the causal standardised log-RMS of the user channel (heads/turn.py energy features),

and predicts the future voice activity of both parties (VAP-style projection, research/OUTSIDE.md 2.1):

* ``user_bins``: P(user active anywhere in bin j), disjoint bins with frame edges ``edges`` (default 3, 5, 8, 13, 25
  = (0,240], (240,400], (400,640], (640,1040], (1040,2000] ms; the first four are head (c)'s bins),
* ``user_quiet``: P(user silent throughout (t, t + h]) for h in ``quiet`` frames (default 13, 25: 1.04 s, 2 s),
* ``agent_bins``: the same bins for the agent (training signal: the other party taking the floor),
* optional ``vap``: the 2^(2 n) joint classes of VAP over ``vap_edges`` (user bits then agent bits, bit = any activity
  in the bin); ``p_user_silent_vap`` = the mass of classes with all user bits 0.

Decision rules live outside the head (scripts/research/uc_turn.py): e.g. the predictive trigger
"rising edge of P(bin1) < t1 AND P(bin2) < t2" of research/DYADIC.md section 8, or "P(user quiet 2 s) > theta".

Streaming: ``init_state`` / ``step`` process any number of frames with carried GRU, duration and energy state and give
exactly the outputs of one ``forward`` over the concatenation (tests/test_uc_turn.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audioforge.heads.turn import causal_standardize, duration_counters, frame_log_rms, log_clip, multi_horizon_targets

DEFAULTS = dict(d_enc=512, hidden=128, layers=2, dropout=0.1, edges=(3, 5, 8, 13, 25), quiet=(13, 25),
                energy=True, vap=False, vap_edges=(2, 5, 8, 10), dur_max=64, act_threshold=0.5, eot=False,
                energy_tau=125, energy_clip=4.0)


def quiet_targets(act: torch.Tensor, lengths: torch.Tensor | None, horizons) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,T) activity -> (B,T,H) 1 if act[t+1 .. t+h] <= 0.5 throughout, and validity (whole window labelled, or an
    active frame already found inside the labelled part: then the target is 0 and known)."""
    B, T = act.shape
    L = torch.full((B,), T, device=act.device) if lengths is None else lengths.to(act.device).clamp(max=T)
    ar = torch.arange(T, device=act.device)[None]
    a = ((act > 0.5) & (ar < L[:, None])).float()
    c = F.pad(a.cumsum(1), (1, 0))
    tg, ok = [], []
    for h in horizons:
        hi = (ar + 1 + h).clamp(max=T).expand(B, T)
        lo = (ar + 1).clamp(max=T).expand(B, T)
        pos = (c.gather(1, hi) - c.gather(1, lo)) > 0
        tg.append((~pos).float())
        ok.append((((ar + h) < L[:, None]) | pos) & (ar < L[:, None]))
    return torch.stack(tg, -1), torch.stack(ok, -1)


def vap_targets(user: torch.Tensor, agent: torch.Tensor, lengths: torch.Tensor | None, edges) -> tuple[torch.Tensor, torch.Tensor]:
    """VAP class index (B,T) over 2 x len(edges) bits (user bins then agent bins, bit j = any activity in bin j) and a
    (B,T) validity mask (every bin fully labelled)."""
    tu, ou = multi_horizon_targets(user, lengths, edges)
    ta, oa = multi_horizon_targets(agent, lengths, edges)
    bits = torch.cat([tu, ta], -1).long()
    w = (2 ** torch.arange(bits.shape[-1], device=bits.device))
    B, T = user.shape
    L = torch.full((B,), T, device=user.device) if lengths is None else lengths.to(user.device).clamp(max=T)
    ar = torch.arange(T, device=user.device)[None]
    valid = (ar + edges[-1] < L[:, None])
    return (bits * w).sum(-1), valid


@dataclass
class UCState:
    h: torch.Tensor | None = None  # GRU state (layers, B, H)
    since: list = field(default_factory=lambda: [None, None])  # duration counters per column (B,) long
    run: list = field(default_factory=lambda: [None, None])
    energy: tuple | None = None  # causal_standardize state
    frames: int = 0


class UCTurnHead(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        c = {**DEFAULTS, **cfg}
        c["edges"], c["quiet"], c["vap_edges"] = list(c["edges"]), list(c["quiet"]), list(c["vap_edges"])
        self.cfg = c
        H = int(c["hidden"])
        self.n_small = 2 + 4 + (1 if c["energy"] else 0)
        self.enc_in = nn.Linear(int(c["d_enc"]), H)
        self.small_in = nn.Linear(self.n_small, H)
        self.drop = nn.Dropout(float(c["dropout"]))
        self.rnn = nn.GRU(H, H, int(c["layers"]), batch_first=True,
                          dropout=float(c["dropout"]) if int(c["layers"]) > 1 else 0.0)
        self.user_bins = nn.Linear(H, len(c["edges"]))
        self.user_quiet = nn.Linear(H, len(c["quiet"]))
        self.agent_bins = nn.Linear(H, len(c["edges"]))
        self.vap = nn.Linear(H, 2 ** (2 * len(c["vap_edges"]))) if c["vap"] else None
        # optional end-of-turn output (eot-bench's label: 1 from the target's turn end on, 0 inside the turn)
        self.eot = nn.Linear(H, 1) if c.get("eot") else None
        if self.vap is not None:
            nv = len(c["vap_edges"])
            idx = torch.arange(2 ** (2 * nv))
            self.register_buffer("vap_user_silent", ((idx & (2 ** nv - 1)) == 0), persistent=False)

    # ----------------------------------------------------------------- features
    def small_features(self, act: torch.Tensor, log_rms: torch.Tensor | None, valid: torch.Tensor | None,
                       state: UCState | None = None) -> tuple[torch.Tensor, UCState]:
        """act (B,T,2) soft activity [user, agent]; log_rms (B,T) raw frame log-RMS of the user channel (or None);
        valid (B,T) bool (frames with audio). -> (B,T,n_small), updated state."""
        c = self.cfg
        st = state if state is not None else UCState()
        B, T, _ = act.shape
        feats = [act.float()]
        on = act > float(c["act_threshold"])
        for j in (0, 1):
            since, run = duration_counters(on[..., j], st.since[j], st.run[j])
            st.since[j], st.run[j] = since[:, -1].clone() if T else st.since[j], run[:, -1].clone() if T else st.run[j]
            feats += [log_clip(since, c["dur_max"])[..., None], log_clip(run, c["dur_max"])[..., None]]
        if c["energy"]:
            if log_rms is None:
                z = act.new_zeros(B, T)
            else:
                v = valid if valid is not None else torch.ones(B, T, dtype=torch.bool, device=act.device)
                z, st.energy = causal_standardize(log_rms.float(), v, int(c["energy_tau"]), float(c["energy_clip"]),
                                                  st.energy)
            feats.append(z[..., None])
        return torch.cat(feats, -1), st

    def forward(self, enc: torch.Tensor, act: torch.Tensor, log_rms: torch.Tensor | None = None,
                valid: torch.Tensor | None = None, state: UCState | None = None) -> tuple[dict, UCState]:
        """enc (B,T,d_enc), act (B,T,2), log_rms (B,T) -> ({name: logits}, state)."""
        small, st = self.small_features(act, log_rms, valid, state)
        x = self.drop(F.silu(self.enc_in(enc.float()) + self.small_in(small)))
        y, st.h = self.rnn(x, st.h)
        y = self.drop(y)
        out = {"user_bins": self.user_bins(y), "user_quiet": self.user_quiet(y), "agent_bins": self.agent_bins(y)}
        if self.vap is not None:
            out["vap"] = self.vap(y)
        if self.eot is not None:
            out["eot"] = self.eot(y)
        st.frames += enc.shape[1]
        return out, st

    def init_state(self) -> UCState:
        return UCState()

    @torch.no_grad()
    def step(self, enc: torch.Tensor, act, log_rms=None, state: UCState | None = None) -> tuple[dict, UCState]:
        """Streaming: enc (1,n,d) frames, act (n,2) or (1,n,2), log_rms (n,) or None -> probabilities {name: (n,k)}."""
        act = torch.as_tensor(np.asarray(act, np.float32)).reshape(1, -1, 2)
        lr = None if log_rms is None else torch.as_tensor(np.asarray(log_rms, np.float32)).reshape(1, -1)
        out, st = self.forward(enc, act, lr, None, state)
        return self.probs(out), st

    @torch.no_grad()
    def probs(self, out: dict) -> dict:
        p = {k: torch.sigmoid(v)[0] for k, v in out.items() if k != "vap"}
        if "vap" in out:
            pv = torch.softmax(out["vap"][0].float(), -1)
            p["p_user_silent_vap"] = pv[:, self.vap_user_silent].sum(-1, keepdim=True)
        return p

    # ----------------------------------------------------------------- training
    def loss(self, out: dict, user: torch.Tensor, agent: torch.Tensor, lengths: torch.Tensor,
             weights: dict | None = None, eot: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """user / agent (B,T) target activity (0/1), lengths (B,) labelled frames; eot (B,T) end-of-turn label with
        -1 = no label (only with cfg eot)."""
        c = self.cfg
        w = {"user_bins": 1.0, "user_quiet": 1.0, "agent_bins": 0.5, "vap": 1.0, "eot": 1.0, **(weights or {})}
        parts = {}
        tu, ou = multi_horizon_targets(user, lengths, c["edges"])
        parts["user_bins"] = _masked_bce(out["user_bins"], tu, ou)
        tq, oq = quiet_targets(user, lengths, c["quiet"])
        parts["user_quiet"] = _masked_bce(out["user_quiet"], tq, oq)
        ta, oa = multi_horizon_targets(agent, lengths, c["edges"])
        parts["agent_bins"] = _masked_bce(out["agent_bins"], ta, oa)
        if "vap" in out:
            cls, ok = vap_targets(user, agent, lengths, c["vap_edges"])
            ce = F.cross_entropy(out["vap"].reshape(-1, out["vap"].shape[-1]), cls.reshape(-1), reduction="none")
            parts["vap"] = (ce * ok.reshape(-1).float()).sum() / ok.float().sum().clamp(min=1)
        if "eot" in out and eot is not None:
            ok = (eot >= 0)[..., None]
            parts["eot"] = _masked_bce(out["eot"], eot.clamp(min=0).float()[..., None], ok)
        total = sum(w[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


def _masked_bce(logits, target, ok):
    l = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (l * ok.float()).sum() / ok.float().sum().clamp(min=1)


def frame_log_rms_np(x: np.ndarray, T: int) -> np.ndarray:
    """numpy wrapper: 16 kHz mono -> raw log-RMS per 80 ms frame (T,)."""
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None]
    return frame_log_rms(t, None, T)[0][0].numpy()


def save_head(head: UCTurnHead, path, meta: dict | None = None):
    torch.save({"cfg": head.cfg, "state_dict": head.state_dict(), "meta": meta or {}}, path)


def load_head(path, device="cpu") -> UCTurnHead:
    ck = torch.load(path, map_location=device, weights_only=False)
    h = UCTurnHead(**ck["cfg"])
    h.load_state_dict(ck["state_dict"])
    h.meta = ck.get("meta", {})
    return h.eval()
