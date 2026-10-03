"""Label-free voice enrollment: follow the primary speaker's diarizer column by VOICE.

A streaming diarizer's columns are slots, not identities: after a pause the same person can come back in another
column (a column swap), so a binding that picks a column once (or re-binds by "who dominates") follows the wrong slot.
Here the primary is an identity instead: an enrollment embedding of the user's first speech, and at every frame the
column whose recent speech sounds most like it, with hysteresis.

Pieces (all numpy / torch, no labels anywhere except the explicitly-oracle ``voice_oracle`` rule):
  * ``speaker_frames``    - the speaker head's per-frame input (the model's plain causal encode, the head's layer mix).
  * ``ColumnEmbedder``    - a SpeakerHead's attentive-statistics pooling + projection over any set of frames.
  * ``recent_embeddings`` - per frame t and column c: the embedding of c's active frames in [t - win, t - 1].
  * ``follow_voice``      - the per-frame primary column from an enrollment embedding (hysteresis).
  * ``enroll_voice``      - the bindings ``voice_first`` / ``voice_dominant`` / ``voice_oracle``.
  * ``agreement``         - agreement of a per-frame binding with a reference column at the turn end.

TitaNet-L backend (the own-head embedding above has 34 % within-window EER):
  * ``TitaNetEmbedder``        - embeds any set of 80 ms frames of a waveform (their audio concatenated) with the
                                 ported NVIDIA TitaNet-Large (``nemo_import.import_titanet``, 192-d, unit norm).
  * ``recent_embeddings_audio`` - the per-frame / per-column look-back embeddings from audio instead of features, on a
                                 grid of every ``stride`` frames (the embedding and the "enough speech" flag of grid
                                 frame g are held for frames g .. g + stride - 1; TitaNet costs ~45 ms per embedding on
                                 2 CPU threads, so per-frame updates are not a product setting either).
  * ``VoiceFollower``          - the hysteresis of ``follow_voice`` as a stateful per-frame step (the server uses it).
  * ``after_prev_end`` rule    - product-realistic identity: the primary is the first column active (>= ``min_run``
                                 consecutive frames) after the agent stopped speaking (``agent_end_frame``: in the
                                 benchmark the end of the previous other-speaker turn, from the labels, as a stand-in
                                 for the agent's own TTS end); ``causal_dominant_from`` seeds the causal_dominant
                                 rule at that choice (the control follower).
  * ``voice_explicit`` rule    - the "say hello" UX: enroll on the primary's first utterance (its timing from the
                                 labels), column = the one dominating those frames, then follow by voice.

Causality. The encoder is causal with one frame of look-ahead (att_context [70, 1], causal convolutions), so the
feature of frame u depends on audio up to frame u + 1. Every decision at frame t reads features of frames <= t - 1
and diarizer probabilities of frames <= t - 1 (plus the enrollment choice, which reads probabilities <= t), i.e.
audio up to frame t: the same information the other bindings use at frame t (the diarizer's own emission lag is
applied downstream by the benchmark's emission rule, as for every binding).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .paths import DATA_ROOT

VOICE_MODES = ("voice_first", "voice_dominant", "voice_oracle")
PRIMARY_MODES = ("after_prev_end", "voice_explicit")  # §9 identity rules (need agent_end / ref respectively)
ALL_MODES = VOICE_MODES + PRIMARY_MODES
# win: look-back of the "recent speech" embedding (2 s); min_frames: speech needed in it (0.64 s); enroll_frames:
# enrollment speech (1.5 s = 19 x 80 ms); margin / hold: hysteresis (switch only if the challenger beats the current
# column by `margin` cosine for `hold` consecutive frames); dom_frames: voice_dominant's "who to enroll" look (5 s).
VOICE_DEFAULTS = dict(win=25, min_frames=8, enroll_frames=19, margin=0.10, hold=6, dom_frames=62, thr=0.5,
                      silent_switch=True)
# stride: TitaNet look-back update period in frames (5 = 400 ms, fixed a priori from its CPU cost before any scoring;
# `hold` = 6 frames then means two consecutive updates); min_run: after_prev_end's "column became active" run.
TITANET_DEFAULTS = dict(stride=5, min_run=3)
# §9b "clean spans": admitted frames = column active AND own VAD > vad_thr AND no other column active; only contiguous
# admitted spans >= min_span frames (0.56 s) are embedded, one embedding per span; a set of span embeddings is pooled by
# consistent_mean (spans whose mean cosine to the others is < span_min_cos are dropped, majority kept); enrollment on
# clean_frames (40 = 3.2 s) admitted frames. Fixed a priori.
CLEAN_DEFAULTS = dict(vad_thr=0.9, min_span=7, span_min_cos=0.3, clean_frames=40)
FRAME_SAMPLES = 1280  # 80 ms at 16 kHz
TITANET_NEMO = str(DATA_ROOT / "nemo" / "speakerverification_en_titanet_large.nemo")


def speaker_head_name(model) -> str:
    return next(k for k, v in model.head_cfg.items() if v["type"] == "speaker")


@torch.no_grad()
def speaker_frames(model, audios, batch_size: int = 8, head: str | None = None) -> list[np.ndarray]:
    """The speaker head's per-frame input for each audio (T_i, D) float32: the plain (unconditioned) causal encode and
    the head's layer mix (``from_layers: all``), exactly what ``SpeechModel.analyze`` feeds the head."""
    model.eval()
    head = head or speaker_head_name(model)
    out = []
    for i in range(0, len(audios), batch_size):
        x, lens = model._pad(audios[i: i + batch_size])
        enc, elen, hidden = model.encode(x, lens, return_hidden=True)
        f = model.head_input(head, enc, hidden)
        out += [f[j, : int(elen[j])].float().cpu().numpy() for j in range(len(elen))]
    return out


class ColumnEmbedder:
    """Unit-norm speaker embeddings of arbitrary frame sets through a SpeakerHead (``pool`` + ``emb``, eval mode):
    ``SpeakerHead.embed`` with a frame mask instead of a length."""

    def __init__(self, head):
        self.head = head.eval()

    @torch.no_grad()
    def __call__(self, x, valid) -> np.ndarray:
        """x (N, L, D), valid (N, L) bool -> (N, E) float32 unit vectors."""
        dev = next(self.head.parameters()).device  # cpu, or the GPU of an --device cuda / mps engine
        x = torch.as_tensor(np.asarray(x, np.float32), device=dev)
        valid = torch.as_tensor(np.asarray(valid, bool), device=dev)
        if x.shape[0] == 0:
            return np.zeros((0, self.head.emb[0].out_features), np.float32)
        return F.normalize(self.head.emb(self.head.pool(x, valid)), dim=-1).cpu().numpy().astype(np.float32)


def recent_embeddings(feats, p, embed, win: int = 25, min_frames: int = 8, thr: float = 0.5, chunk: int = 256):
    """Per frame t and column c the embedding of c's active frames (p > thr) among frames [t - win, t - 1].

    Returns (emb (T, S, E) float32, ok (T, S) bool): ok = at least ``min_frames`` active frames in that look-back;
    emb is 0 where not ok. Frame t never reads feats / p of frames >= t (causal by construction)."""
    feats = np.asarray(feats, np.float32)
    p = np.asarray(p, np.float32)
    T, S = p.shape
    assert len(feats) >= T, (len(feats), T)
    on = p > thr
    cs = np.concatenate([np.zeros((1, S), np.int64), np.cumsum(on, 0)])
    t_idx = np.arange(T)
    cnt = cs[t_idx] - cs[np.maximum(0, t_idx - win)]  # active frames in [t - win, t - 1]
    ok = cnt >= min_frames
    D = feats.shape[1]
    fp = np.concatenate([np.zeros((win, D), np.float32), feats[:T]])  # fp[t : t + win] = frames t - win .. t - 1
    op = np.concatenate([np.zeros((win, S), bool), on])
    tt, cc = np.nonzero(ok)
    E = None
    emb = None
    for i in range(0, len(tt), chunk):
        a, c = tt[i: i + chunk], cc[i: i + chunk]
        rows = a[:, None] + np.arange(win)[None]  # (n, win)
        e = embed(fp[rows], op[rows, c[:, None]])
        if emb is None:
            E = e.shape[1]
            emb = np.zeros((T, S, E), np.float32)
        emb[a, c] = e
    if emb is None:
        emb = np.zeros((T, S, embed(np.zeros((1, 1, D), np.float32), np.ones((1, 1), bool)).shape[1]), np.float32)
    return emb, ok


def enroll_frames(on_col, n: int, start: int = 0):
    """The first ``n`` frames >= start where ``on_col`` is True, and the frame from which an embedding of them is
    available (last frame + 1: its feature sees audio up to that frame). (None, None) if fewer than n exist."""
    idx = np.nonzero(np.asarray(on_col, bool)[start:])[0] + start
    if len(idx) < n:
        return None, None
    return idx[:n], int(idx[n - 1]) + 1


class VoiceFollower:
    """The hysteresis of ``follow_voice`` as a stateful per-frame step, so the benchmark and the server run the same
    rule. ``step(emb_t, ok_t)`` with the (S, E) recent embeddings and (S,) flags of one frame returns the current
    column after that frame."""

    def __init__(self, e_enr, c0: int, margin: float = 0.10, hold: int = 6, silent_switch: bool = True):
        self.e_enr = np.asarray(e_enr, np.float32)
        self.margin, self.hold, self.silent_switch = margin, hold, silent_switch
        self.cur, self.chal, self.run, self.n_switches = int(c0), -1, 0, 0
        self.self_sum, self.self_n = 0.0, 0

    def step(self, emb_t, ok_t) -> int:
        ok_t = np.asarray(ok_t, bool)
        s = np.where(ok_t, np.asarray(emb_t, np.float32) @ self.e_enr, -np.inf)
        cur = self.cur
        if ok_t[cur]:
            self.self_sum += float(s[cur]); self.self_n += 1
        b = int(np.argmax(s))
        qual = False
        if b != cur and np.isfinite(s[b]):
            if ok_t[cur]:
                qual = s[b] - s[cur] >= self.margin
            elif self.self_n and self.silent_switch:
                qual = s[b] >= self.self_sum / self.self_n - self.margin
        if qual:
            self.run = self.run + 1 if b == self.chal else 1
            self.chal = b
            if self.run >= self.hold:
                self.cur, self.chal, self.run, self.n_switches = b, -1, 0, self.n_switches + 1
                self.self_sum, self.self_n = float(s[b]), 1  # the new column's self level starts from this frame
        else:
            self.chal, self.run = -1, 0
        return self.cur


def follow_voice(emb, ok, e_enr, c0: int, start: int, init, margin: float = 0.10, hold: int = 6,
                 silent_switch: bool = True):
    """Per-frame primary column from an enrollment embedding.

    Frames < start keep ``init``. From ``start`` the current column is c0; at each frame the challenger is the column
    (with enough recent speech, ``ok``) whose recent embedding is closest (cosine) to ``e_enr``. It replaces the
    current column once it has qualified on ``hold`` consecutive frames, where qualifying means
      * the current column has recent speech: sim(challenger) - sim(current) >= margin;
      * the current column is silent: sim(challenger) >= (mean sim of the current column over the frames it was
        heard since binding) - margin, i.e. it sounds as much like the user as the user's own column did
        (a label-free, per-window self-similarity level; before the current column has been heard: no switch;
        ``silent_switch=False``: never).
    Returns (col (T,) int64, n_switches)."""
    col = np.asarray(init, np.int64).copy()
    T = len(col)
    f = VoiceFollower(e_enr, c0, margin, hold, silent_switch)
    for t in range(max(0, start), T):
        col[t] = f.step(emb[t], ok[t])
    return col, f.n_switches


def _dominant_cols(on, p, a: int, b: int) -> int:
    hard, soft = on[a:b].sum(0), p[a:b].sum(0)
    return int(np.lexsort((-soft, -hard))[0])


class TitaNetEmbedder:
    """Unit-norm TitaNet-L embeddings of arbitrary sets of 80 ms frames of a 16 kHz waveform: the audio of the
    selected frames is concatenated (the column's active speech only) and embedded as one utterance (TitaNet's own
    per-utterance feature normalisation runs on that excerpt, as it does on any short clip)."""

    dim = 192

    def __init__(self, model=None, path: str | None = None, frame_samples: int = FRAME_SAMPLES, batch_size: int = 8):
        if model is None:
            from .nemo_import import import_titanet
            model = import_titanet(path or TITANET_NEMO)
        self.model = model.eval()
        self.fs, self.batch_size = frame_samples, batch_size

    def segments(self, audio, idx_lists) -> list[np.ndarray]:
        audio = np.asarray(audio, np.float32)
        out = []
        for idx in idx_lists:
            idx = np.asarray(idx, np.int64)
            seg = np.concatenate([audio[u * self.fs: (u + 1) * self.fs] for u in idx]) if len(idx) else np.zeros(0, np.float32)
            if len(seg) < self.fs:  # never happens for >= 1 full frame; keeps the conv stack safe
                seg = np.concatenate([seg, np.zeros(self.fs - len(seg), np.float32)])
            out.append(seg)
        return out

    @torch.no_grad()
    def frames(self, audio, idx_lists) -> np.ndarray:
        """(N, 192) float32: one embedding per frame set (frame u = samples [u fs, (u + 1) fs))."""
        if not len(idx_lists):
            return np.zeros((0, self.dim), np.float32)
        return self.model.embed(self.segments(audio, idx_lists), self.batch_size).float().cpu().numpy()


def recent_embeddings_audio(audio, p, embedder, win: int = 25, min_frames: int = 8, thr: float = 0.5,
                            stride: int = 5, batch: int = 64):
    """``recent_embeddings`` from audio: at every grid frame g (multiples of ``stride``) and column c with >=
    ``min_frames`` active frames among [g - win, g - 1], the embedding of that speech; frames g .. g + stride - 1
    hold grid frame g's embedding and flag (a follower that updates every ``stride`` frames).

    Returns (emb (T, S, E) float32, ok (T, S) bool). Frame t reads audio of frames <= t - 1 and probabilities of
    frames <= t - 1 only (causal by construction)."""
    p = np.asarray(p, np.float32)
    T, S = p.shape
    on = p > thr
    cs = np.concatenate([np.zeros((1, S), np.int64), np.cumsum(on, 0)])
    E = getattr(embedder, "dim", None)
    grid = np.arange(0, T, stride)
    cnt = cs[grid] - cs[np.maximum(0, grid - win)]
    okg = cnt >= min_frames  # (G, S)
    gg, cc = np.nonzero(okg)  # grid index, column
    embg = None
    for i in range(0, len(gg), batch):
        g, c = gg[i: i + batch], cc[i: i + batch]
        idx = [np.nonzero(on[max(0, a - win): a, b])[0] + max(0, a - win) for a, b in zip(grid[g], c)]
        e = embedder.frames(audio, idx)
        if embg is None:
            E = e.shape[1]
            embg = np.zeros((len(grid), S, E), np.float32)
        embg[g, c] = e
    if embg is None:
        embg = np.zeros((len(grid), S, E or 1), np.float32)
    rep = np.minimum(stride, T - grid)
    return np.repeat(embg, rep, 0), np.repeat(okg, rep, 0)


def causal_dominant_from(p, k: int = 25, s: int = 25, thr: float = 0.5, force=None) -> np.ndarray:
    """eval_stage1.enroll_causal_dominant (bind the running dominant column, re-bind after > s silent frames), with
    an optional forced binding ``force=(c0, t0)``: at frame t0 the bound column becomes c0 (silence count reset) and
    the rule continues from there. ``force=None`` reproduces enroll_causal_dominant exactly."""
    p = np.asarray(p, np.float64)
    T, S = p.shape
    on = p > thr
    ch = np.concatenate([np.zeros((1, S)), np.cumsum(on, 0)])
    cs = np.concatenate([np.zeros((1, S)), np.cumsum(p, 0)])
    out = np.full(T, -1, np.int64)
    c, silent = -1, 0
    c0, t0 = force if force is not None else (None, None)
    for t in range(T):
        lo = max(0, t - k + 1)
        hard, soft = ch[t + 1] - ch[lo], cs[t + 1] - cs[lo]
        if t0 is not None and t == t0:
            c, silent = int(c0), 0
        elif c < 0:
            if on[t].any():
                c, silent = int(np.lexsort((-soft, -hard))[0]), 0
        else:
            silent = 0 if on[t, c] else silent + 1
            if silent > s:
                new = int(np.lexsort((-soft, -hard))[0])
                if new != c and hard[new] > 0:
                    c, silent = new, 0
        out[t] = c
    return out


def agent_end_frame(spk_targets, onset: int, thr: float = 0.5):
    """The benchmark's stand-in for "the agent stopped speaking": the end (exclusive frame) of the last activity run
    of any NON-primary label speaker (spk_targets columns 1..; column 0 = the primary) that starts before the primary's
    onset. None if no other speaker spoke before the onset. Labels are used for this time only, never for the column
    choice (in the product the agent knows its own TTS end)."""
    y = np.asarray(spk_targets)
    if y.ndim != 2 or y.shape[1] < 2:
        return None
    other = (y[:, 1:] > thr).any(1)
    d = np.diff(np.concatenate([[0], other.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = starts < onset
    return int(ends[keep][-1]) if keep.any() else None


def after_prev_end_choice(on, t_from: int, min_run: int = 3):
    """Label-free column choice: the first column with ``min_run`` consecutive active frames at or after ``t_from``
    (ties at the same frame: the lower column index, as np.argmax). Returns (column, decision frame = the last frame
    of that run) or (None, None)."""
    on = np.asarray(on, bool)
    T, S = on.shape
    run = np.zeros(S, np.int64)
    for t in range(max(0, int(t_from)), T):
        run = np.where(on[t], run + 1, 0)
        if (run >= min_run).any():
            return int(np.argmax(run >= min_run)), t
    return None, None


# --------------------------------------------------------------------------- §9b clean-span masks
def clean_mask(p, vad, thr: float = 0.5, vad_thr: float = 0.9) -> np.ndarray:
    """(T, S) admitted frames: column active AND the model's own VAD > vad_thr AND exactly one column active."""
    on = np.asarray(p, np.float32) > thr
    v = np.asarray(vad, np.float32)[: len(on)]
    if len(v) < len(on):
        v = np.concatenate([v, np.zeros(len(on) - len(v), np.float32)])
    return on & (v[:, None] > vad_thr) & (on.sum(1, keepdims=True) == 1)


def spans_of(mask_col, lo: int, hi: int, min_span: int) -> list:
    """Contiguous runs of True in mask_col[lo:hi] with >= min_span frames (as frame-index arrays)."""
    m = np.asarray(mask_col, bool)[lo:hi].astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return [np.arange(a, b) + lo for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_span]


def consistent_mean(E, min_cos: float = 0.3):
    """Unit-norm mean of a set of unit embeddings after dropping, one at a time, the span whose mean cosine to the
    others is lowest while that mean is < min_cos and >= 2 spans remain (the majority voice wins). -> (vec, n_kept)."""
    E = np.asarray(E, np.float32)
    keep = list(range(len(E)))
    while len(keep) >= 2:
        G = E[keep] @ E[keep].T
        mean_other = (G.sum(1) - 1.0) / (len(keep) - 1)
        j = int(np.argmin(mean_other))
        if mean_other[j] >= min_cos:
            break
        keep.pop(j)
    v = E[keep].mean(0)
    return v / max(np.linalg.norm(v), 1e-8), len(keep)


def recent_embeddings_spans(audio, adm, embedder, win: int = 25, min_frames: int = 8, stride: int = 5,
                            min_span: int = 7, min_cos: float = 0.3, batch: int = 64):
    """``recent_embeddings_audio`` on an admitted mask (clean_mask) with span pooling: at grid frame g and column c
    the spans (>= min_span admitted frames) inside [g - win, g - 1] are embedded one by one and pooled by
    consistent_mean; ok = their frames total >= min_frames. Causal as recent_embeddings_audio."""
    adm = np.asarray(adm, bool)
    T, S = adm.shape
    grid = np.arange(0, T, stride)
    E = getattr(embedder, "dim", 192)
    embg, okg = np.zeros((len(grid), S, E), np.float32), np.zeros((len(grid), S), bool)
    cells, jobs = [], []
    for gi, g in enumerate(grid):
        for c in range(S):
            sp = spans_of(adm[:, c], max(0, g - win), g, min_span)
            if sp and sum(len(x) for x in sp) >= min_frames:
                cells.append((gi, c, len(jobs), len(sp)))
                jobs += sp
    embs = np.concatenate([embedder.frames(audio, jobs[i: i + batch]) for i in range(0, len(jobs), batch)]) \
        if jobs else np.zeros((0, E), np.float32)
    for gi, c, a, n in cells:
        embg[gi, c], _ = consistent_mean(embs[a: a + n], min_cos)
        okg[gi, c] = True
    rep = np.minimum(stride, T - grid)
    return np.repeat(embg, rep, 0), np.repeat(okg, rep, 0)


def enroll_spans(adm_col, n: int, start: int = 0, min_span: int = 7):
    """The first spans (>= min_span admitted frames, from ``start``) whose frames total >= n, and the frame from which
    their embedding is available (last frame + 1). (None, None) if the window never reaches n."""
    tot, out = 0, []
    for sp in spans_of(adm_col, int(start), len(adm_col), min_span):
        out.append(sp)
        tot += len(sp)
        if tot >= n:
            return out, int(sp[-1]) + 1
    return None, None


def enroll_voice(p, feats, embed, mode: str, ref=None, oracle_col: int | None = None, causal_init=None,
                 emb_ok=None, enroll_embed=None, agent_end: int | None = None, clean=None, **kw):
    """Per-frame primary column (T,) of a (T, S) diarizer track under a voice binding, plus diagnostics.

    mode:
      * 'voice_first':    enroll the first column to become active (the user speaks first); embedding of its first
                          ``enroll_frames`` active frames; before that, the first active column (-1 before any speech).
      * 'voice_dominant': enroll the column with the most active frames over the first ``dom_frames`` frames after the
                          first speech (decided at that frame; before it the ``causal_init`` binding, e.g.
                          causal_dominant); embedding of its first ``enroll_frames`` active frames.
      * 'voice_oracle':   upper bound - enrollment identity from the labels: the oracle column ``oracle_col`` until
                          enrolled, embedding of the first ``enroll_frames`` frames of the oracle primary's speech
                          (``ref``); following is label-free as for the other rules.
      * 'after_prev_end': product-realistic identity (§9): before the agent's end (``agent_end``) the ``causal_init``
                          binding; from it, the first column with ``min_run`` consecutive active frames
                          (``after_prev_end_choice``); embedding of that column's first ``enroll_frames`` active
                          frames after ``agent_end``. ``agent_end`` None (no previous agent turn) or no column
                          becoming active: the causal_init binding throughout (info['fallback'] = 'no_agent_turn' /
                          'no_column').
      * 'voice_explicit': the "say hello" UX (§9): enroll on the primary's first ``enroll_frames`` label frames
                          (``ref``; their timing stands in for the prompt's answer), column = the one dominating
                          those frames; before that the ``causal_init`` binding.
    ``emb_ok``: precomputed ``recent_embeddings`` / ``recent_embeddings_audio`` (emb, ok) of this track (else computed
    here from ``feats``). ``enroll_embed``: callable(frame indices) -> enrollment embedding (default: ``embed`` over
    ``feats``; the TitaNet backend passes ``lambda idx: TitaNetEmbedder.frames(audio, [idx])[0]``).
    ``clean``: §9b admitted mask (clean_mask): the column choice is unchanged, but the enrollment is the chosen
    column's first admitted spans totalling ``clean_frames`` frames (enroll_spans) and ``enroll_embed`` receives that
    list of spans (the caller pools them with consistent_mean); ``emb_ok`` must then come from recent_embeddings_spans.
    Returns (col (T,), info dict: c0, enrolled_at, n_switches[, fallback])."""
    cfg = {**VOICE_DEFAULTS, **TITANET_DEFAULTS, **CLEAN_DEFAULTS, **kw}
    assert mode in ALL_MODES, mode
    p = np.asarray(p, np.float32)
    feats = None if feats is None else np.asarray(feats, np.float32)
    if enroll_embed is None:
        assert feats is not None, "enroll_embed or feats is needed for the enrollment embedding"
        enroll_embed = lambda idx: embed(feats[idx][None], np.ones((1, len(idx)), bool))[0]  # noqa: E731
    T, S = p.shape
    on = p > cfg["thr"]
    any_on = np.nonzero(on.any(1))[0]
    init = np.full(T, -1, np.int64)
    if mode in PRIMARY_MODES:
        ci = np.asarray(causal_init, np.int64) if causal_init is not None else causal_dominant_from(p, thr=cfg["thr"])
        init[:] = ci
        info = {"c0": -1, "enrolled_at": None, "n_switches": 0}
        if mode == "after_prev_end":
            if agent_end is None:
                return init, dict(info, fallback="no_agent_turn")
            c0, tc = after_prev_end_choice(on, agent_end, cfg["min_run"])
            if c0 is None:
                return init, dict(info, fallback="no_column")
            init[tc:] = c0
            idx, t_e = (enroll_spans(clean[:, c0], cfg["clean_frames"], int(agent_end), cfg["min_span"]) if clean is not None
                        else enroll_frames(on[:, c0], cfg["enroll_frames"], start=int(agent_end)))
            info.update(c0=c0, chosen_at=tc, agent_end=int(agent_end))
            if t_e is not None:
                t_e = max(t_e, tc)
        else:
            idx, t_e = enroll_frames(np.asarray(ref)[:T] > 0.5, cfg["enroll_frames"])
            if t_e is None:
                return init, dict(info, fallback="no_primary_speech")
            c0 = _dominant_cols(on, p, int(idx[0]), t_e)
            init[t_e:] = c0
            info.update(c0=c0, chosen_at=t_e)
            if clean is not None:  # §9b: enrollment on the chosen column's clean spans from the utterance on
                idx, t_e2 = enroll_spans(clean[:, c0], cfg["clean_frames"], int(idx[0]), cfg["min_span"])
                t_e = None if t_e2 is None else max(t_e, t_e2)
    elif mode == "voice_oracle":
        c0 = int(oracle_col)
        init[:] = c0
        idx, t_e = enroll_frames(np.asarray(ref)[:T] > 0.5, cfg["enroll_frames"])
    else:
        if not len(any_on):
            return init, {"c0": -1, "enrolled_at": None, "n_switches": 0}
        f0 = int(any_on[0])
        if mode == "voice_first":
            c0 = int(np.argmax(p[f0]))  # = eval_stage1.enroll_first_active
            init[f0:] = c0
            idx, t_e = enroll_frames(on[:, c0], cfg["enroll_frames"])
        else:
            td = f0 + cfg["dom_frames"]
            ci = np.asarray(causal_init, np.int64) if causal_init is not None else np.where(
                np.arange(T) >= f0, int(np.argmax(p[f0])), -1)
            init[:] = ci
            if td >= T:  # never decided inside the window: the causal binding throughout
                return init, {"c0": -1, "enrolled_at": None, "n_switches": 0}
            c0 = _dominant_cols(on, p, f0, td)
            init[td:] = c0
            idx, t_e = enroll_frames(on[:, c0], cfg["enroll_frames"])
            if t_e is not None:
                t_e = max(t_e, td)
    if t_e is None or t_e >= T:
        out = {"c0": c0, "enrolled_at": None, "n_switches": 0}
        if mode in PRIMARY_MODES:
            out.update({k: v for k, v in info.items() if k not in out})
        return init, out
    e_enr = enroll_embed(idx if clean is not None else np.asarray(idx, np.int64))
    if emb_ok is None:
        emb_ok = recent_embeddings(feats, p, embed, cfg["win"], cfg["min_frames"], cfg["thr"])
    col, n_sw = follow_voice(emb_ok[0], emb_ok[1], e_enr, c0, t_e, init, cfg["margin"], cfg["hold"],
                             cfg["silent_switch"])
    out = {"c0": c0, "enrolled_at": t_e, "n_switches": n_sw}
    if mode in PRIMARY_MODES:
        out.update({k: v for k, v in info.items() if k not in out})
    return col, out


def agreement(cols, ref_cols, ends, mask=None) -> float:
    """Fraction of turns whose per-frame binding equals the reference column at the last frame of the turn
    (frame end - 1); ``mask`` selects a subset (e.g. the floor-open stratum)."""
    a = np.array([int(np.asarray(c)[e - 1]) == int(r) for c, r, e in zip(cols, ref_cols, ends)])
    if mask is not None:
        a = a[np.asarray(mask, bool)]
    return float(a.mean()) if len(a) else float("nan")


def rebinds(col) -> int:
    """Binding changes along a per-frame binding (the -1 -> first column step counts, as for causal_dominant)."""
    return int((np.diff(np.asarray(col)) != 0).sum())


def print_is_clean(own_vad, other_vad=None, min_speech: float = 0.8, max_other: float = 0.1,
                   thr: float = 0.5) -> tuple[bool, dict]:
    """Quality check of a voice-print segment: the user must be speaking on at least
    ``min_speech`` of its 80 ms frames (the served VAD head on the user's audio), and where a second channel is
    available (the other party's microphone), the other party at most ``max_other`` of them. A print cut from a
    stretch where the other party talks or laughs over the user (TurnBench tb_160: 62 % of the frames) is a mixture of
    both voices; the TS-VAD head then calls the user's own voice "other". -> (ok, {"speech", "other"} fractions)."""
    own = np.asarray(own_vad, np.float32).reshape(-1)
    q = {"speech": round(float((own > thr).mean()) if len(own) else 0.0, 3)}
    ok = len(own) > 0 and q["speech"] >= min_speech
    if other_vad is not None:
        oth = np.asarray(other_vad, np.float32).reshape(-1)
        q["other"] = round(float((oth > thr).mean()) if len(oth) else 0.0, 3)
        ok = ok and q["other"] <= max_other
    return bool(ok), q
