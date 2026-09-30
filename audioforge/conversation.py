"""Synthetic multi-party conversations for *speaker-aware* end-of-turn (EOT) detection (v2).

Built from the same audio primitive as ``data.ToneLanguage`` (harmonic tones per character, speakers
differ in pitch scale and timbre). One conversation = one episode of a designated PRIMARY speaker (the
user talking to the agent, 1-3 speech segments) plus 0-2 other speakers.

Turn-taking follows the four-move model of NVIDIA's FastMSS simulator (arXiv 2605.15442, sec. 3):
after every utterance a move z in {TH, TS, IR, BC} is drawn from a first-order Markov chain
P[i, j] = p(z_t = j | z_{t-1} = i) (default: (1 - s) * 1 p^T + s * I, stationary distribution p):

  TH  turn hold      the same speaker continues after a pause ~ Exp(beta_th)
  TS  turn switch    another speaker starts after a gap ~ Exp(beta_ts)
  IR  interruption   another speaker starts before the current utterance ends; the overlapped part
                     is a ratio r ~ TruncExp(beta_ir) on [0, 1] of the current utterance
  BC  backchannel    another speaker says "mm hm", placed uniformly within the preceding utterance;
                     the floor stays with the current speaker

No speaker ever overlaps themselves; the next speaker is uniform over everyone but the current one.
Presets (``TurnTakingConfig.preset``): ``diar`` = the paper's CALLHOME-fitted move probabilities
(best for Sortformer), ``asr`` = CALLHOME with overlap boost (IR and BC x2, renormalised; best for
DiCoW), ``default`` = halfway (x sqrt 2) since the turn recipe trains ASR + diarization + turn jointly,
``flat`` = the paper's flat prior.

Primary disfluencies (FDB-v3 categories), each with its own rate: filler "uh" before a hesitation,
repetition (a word said twice), self-correction (a false start of a different word, then the intended
one), mid-word cutoff before a hesitation (the word is restarted after the pause). ``text`` is the
verbatim transcript (fillers written as the word ``e``, false starts / cutoffs as the partial word);
``text_clean`` is the intended fluent sentence.

Prosodic cue (the stand-in for syntax/prosody in real speech): a turn-final word is usually rendered
with *final lowering* (pitch glides down ~18 %, lengthened, softer). Every speaker follows the same
rule, so a bystander finishing a sentence sounds exactly like the primary finishing theirs: only
knowing *who* is speaking separates them. Cue reliability is imperfect on purpose (10 % of true
ends are not lowered, 5 % of hesitations are).

Labels per 80 ms encoder frame (T = ToneLanguage.n_frames(len(audio))):
  spk_targets (T,4)   activity of every speaker (column 0 = primary; Sortformer sorts by arrival)
  primary_act (T,)    = spk_targets[:, 0]; also emitted as ``spk_act`` (conditioning input)
  eot (T,)            1 on every frame from the primary's TRUE end on (never after hesitations, while
                      other speakers hold the floor mid-episode, or when a bystander stops)
  turn_end_frame      first true-EOT frame = last primary-active frame + 1
  onset_frame         first primary-active frame
  text                the primary's verbatim transcript (``text_clean``: fluent version)

``eot`` is a deterministic function of ``spk_act`` (frames after its last active frame), so the
stock ``data.Collate`` (which keeps ``spk_act`` but not ``eot``) is enough for training:
``heads.turn.eot_targets`` rebuilds the label. ``ConversationCollate`` also pads ``eot`` explicitly.

The metric (``eot_bench``) follows LiveKit eot-bench: sweep the decision threshold, count a firing
before the true turn end as a false cutoff, and report dead-air latency (P50/P90) at the threshold
that gives <= 5 % false cutoffs. ``silence_scores`` turns an activity track into the silence-timeout
baseline under the same sweep.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .data import FRAME_SEC, Collate, ToneLanguage

FILLER = "e"  # the filler "uh" reuses the tone of an existing character, flat and sustained
MOVES = ("TH", "TS", "IR", "BC")


# --------------------------------------------------------------------------- audio primitives
def tone(lang: ToneLanguage, c: str, speaker: int, n: int, p0: float = 1.0, p1: float = 1.0,
         amp: float = 1.0) -> np.ndarray:
    """ToneLanguage's character tone with an optional linear pitch glide p0 -> p1 (multipliers)."""
    spk, sr = lang.speakers[speaker], lang.sr
    t = np.arange(n) / sr
    f = lang.freq[c] * spk["scale"] * (p0 + (p1 - p0) * np.arange(n) / max(1, n))
    phase = 2 * np.pi * np.cumsum(f) / sr
    x = sum(h * np.sin((k + 1) * phase) for k, h in enumerate(spk["harm"]))
    env = np.minimum(1, np.minimum(t, t[::-1]) / 0.008)
    return (0.3 * amp * x * env).astype(np.float32)


def _word(lang: ToneLanguage, w: str, speaker: int, rng: random.Random, lowered: bool = False,
          dur: float = 1.0, amp: float = 1.0):
    """One word (no surrounding gaps). lowered: final lowering (pitch glides down ~18 %, lengthened,
    energy decays). -> audio, per-sample activity."""
    sr, xs = lang.sr, []
    for ci, c in enumerate(w):
        n = int(sr * rng.uniform(*lang.char_ms) / 1000 * dur * (1.3 if lowered else 1.0))
        if lowered:
            p0, p1 = 1 - 0.18 * ci / len(w), 1 - 0.18 * (ci + 1) / len(w)
            xs.append(tone(lang, c, speaker, n, p0, p1, amp * (1 - 0.3 * (ci + 1) / len(w))))
        else:
            xs.append(tone(lang, c, speaker, n, amp=amp))
    x = np.concatenate(xs)
    return x, np.ones(len(x), bool)


def render_words(lang: ToneLanguage, words: list[str], speaker: int, rng: random.Random,
                 ending: str = "plain", dur: float = 1.0, amp: float = 1.0):
    """Render words with word gaps. ending: plain | lowered (final lowering on the last word) |
    filler (append a sustained "uh"). -> audio, per-sample activity."""
    sr, pieces, act = lang.sr, [], []

    def add(x, active):
        pieces.append(x)
        act.append(np.full(len(x), active, bool))

    for wi, w in enumerate(words):
        if wi:
            add(np.zeros(int(sr * rng.uniform(*lang.char_ms) / 1000 * dur), np.float32), False)
        x, a = _word(lang, w, speaker, rng, wi == len(words) - 1 and ending == "lowered", dur, amp)
        pieces.append(x)
        act.append(a)
    if ending == "filler":
        add(np.zeros(int(sr * rng.uniform(0.0, 0.08)), np.float32), False)
        add(tone(lang, FILLER, speaker, int(sr * rng.uniform(0.25, 0.45)), 0.92, 0.92, 0.8 * amp), True)
    return np.concatenate(pieces), np.concatenate(act)


def backchannel(lang: ToneLanguage, speaker: int, rng: random.Random, amp: float = 1.0):
    """'mm hm': two short soft bursts."""
    return render_words(lang, ["mm", "hm"], speaker, rng, dur=0.8, amp=0.7 * amp)


# --------------------------------------------------------------------------- turn-taking config
CALLHOME = (0.15, 0.21, 0.44, 0.20)  # FastMSS Table 1, CALLHOME-fitted [p_TH, p_TS, p_IR, p_BC]


def boost_overlap(p, factor: float) -> tuple:
    """Scale the overlapping moves (IR, BC) by ``factor`` and renormalise (the paper's 'OV boost' is
    factor 2: CALLHOME -> (0.09, 0.13, 0.54, 0.24))."""
    q = [p[0], p[1], p[2] * factor, p[3] * factor]
    return tuple(x / sum(q) for x in q)


@dataclass(frozen=True)
class TurnTakingConfig:
    # participants and primary turn shape
    p_others: tuple = (0.2, 0.45, 0.35)  # P(0 / 1 / 2 other speakers)
    p_hes: tuple = (0.35, 0.45, 0.2)  # P(0 / 1 / 2 segment boundaries in the primary episode)
    words_per_segment: tuple = (1, 3)
    other_words: tuple = (1, 3)
    # FastMSS four-move Markov model
    p_moves: tuple = boost_overlap(CALLHOME, math.sqrt(2))  # stationary [TH, TS, IR, BC]
    stickiness: float = 0.2  # P = (1 - s) * 1 p^T + s * I
    transition: tuple | None = None  # explicit 4x4 row-stochastic matrix (overrides p_moves/stickiness)
    beta_th: float = 1.6  # 1/s: hold pause ~ Exp(beta_th), mean 0.62 s
    beta_ts: float = 2.5  # 1/s: switch gap ~ Exp(beta_ts), mean 0.4 s
    beta_ir: float = 3.0  # overlap ratio ~ TruncExp(beta_ir) on [0, 1], mean 0.28
    pause_clip: tuple = (0.1, 2.5)
    gap_clip: tuple = (0.05, 2.0)
    max_excursion: int = 2  # other-speaker utterances before the floor must return to the primary
    max_backchannels: int = 2  # per utterance
    p_lead_other: float = 0.15  # another speaker talks first (onset > 0, speech before the primary)
    # prosody
    p_final_lowered: float = 0.9
    p_decoy: float = 0.05  # final lowering on a mid-episode segment end
    # FDB-v3 disfluencies of the primary
    p_filler: float = 0.6  # per segment boundary: "uh" before the pause
    p_cutoff: float = 0.15  # per segment boundary: last word cut mid-word, restarted after the pause
    p_repeat: float = 0.04  # per word: repetition ("go go")
    p_correct: float = 0.03  # per word: self-correction (false start of another word, then the word)
    filler_in_text: bool = True
    trail_sec: float = 2.6

    def matrix(self) -> np.ndarray:
        if self.transition is not None:
            P = np.asarray(self.transition, np.float64)
        else:
            p = np.asarray(self.p_moves, np.float64)
            P = (1 - self.stickiness) * np.tile(p / p.sum(), (4, 1)) + self.stickiness * np.eye(4)
        return P / P.sum(1, keepdims=True)

    @classmethod
    def preset(cls, name: str = "default", **kw) -> "TurnTakingConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; one of {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **kw})


PRESETS = {
    "default": {},  # CALLHOME x sqrt(2) overlap: between the two task optima (joint ASR+diar+turn)
    "asr": dict(p_moves=boost_overlap(CALLHOME, 2.0)),  # paper: overlap boost helps ASR
    "diar": dict(p_moves=CALLHOME),  # paper: natural CALLHOME statistics best for diarization
    "flat": dict(p_moves=(0.25, 0.25, 0.25, 0.25)),  # paper's flat prior
}


# --------------------------------------------------------------------------- conversations
def _words(lang, rng, n):
    return [rng.choice(lang.LEXICON) for _ in range(rng.randint(*n))]


def _trunc_exp(rng, beta: float) -> float:
    """Sample from Exp(beta) truncated to [0, 1] (inverse CDF)."""
    return -math.log(1 - rng.random() * (1 - math.exp(-beta))) / beta


def _clipped_exp(rng, beta: float, clip) -> float:
    return min(max(rng.expovariate(beta), clip[0]), clip[1])


def _primary_segment(lang, words, speaker, rng, cfg: TurnTakingConfig, final: bool, ev: dict):
    """Render one primary segment with FDB-v3 disfluencies. -> audio, act, verbatim tokens, carry
    (the word cut mid-word at the end, to be restarted at the start of the next segment, or None)."""
    sr, pieces, act, toks = lang.sr, [], [], []

    def add(x, active):
        pieces.append(x)
        act.append(np.full(len(x), active, bool))

    def gap(scale=1.0):
        add(np.zeros(int(sr * rng.uniform(*lang.char_ms) / 1000 * scale), np.float32), False)

    def word(w, lowered=False):
        x, a = _word(lang, w, speaker, rng, lowered)
        pieces.append(x)
        act.append(a)

    lowered_last, carry = False, None
    if final:
        lowered_last = rng.random() < cfg.p_final_lowered
        ev["lowered_final"] = lowered_last
    else:
        ev["n_boundaries"] += 1
        if rng.random() < cfg.p_cutoff and len(words[-1]) > 1:
            carry = words[-1]
            ev["cutoffs"] += 1
        elif rng.random() < cfg.p_decoy:
            lowered_last = True
            ev["decoys"] += 1
    for wi, w in enumerate(words):
        last = wi == len(words) - 1
        if wi:
            gap()
        if last and carry is not None:  # mid-word cutoff right before the hesitation
            k = rng.randint(1, len(w) - 1)
            word(w[:k])
            toks.append(w[:k])
            continue
        ev["n_words"] += 1
        r = rng.random()
        if r < cfg.p_repeat:
            word(w)
            gap()
            toks.append(w)
            ev["repetitions"] += 1
        elif r < cfg.p_repeat + cfg.p_correct:
            wrong = rng.choice([v for v in lang.LEXICON if v != w])
            k = rng.randint(1, max(1, len(wrong) - 1))
            word(wrong[:k])
            gap(1.5)
            toks.append(wrong[:k])
            ev["corrections"] += 1
        word(w, lowered=last and lowered_last)
        toks.append(w)
    if not final and rng.random() < cfg.p_filler:
        add(np.zeros(int(sr * rng.uniform(0.0, 0.08)), np.float32), False)
        add(tone(lang, FILLER, speaker, int(sr * rng.uniform(0.25, 0.45)), 0.92, 0.92, 0.8), True)
        ev["fillers"] += 1
        if cfg.filler_in_text:
            toks.append(FILLER)
    return np.concatenate(pieces), np.concatenate(act), toks, carry


def conversation(lang: ToneLanguage, rng: random.Random, cfg: TurnTakingConfig | None = None,
                 preset: str = "default", **kw) -> dict:
    """One episode. ``cfg`` (or ``preset`` + field overrides in ``kw``) is a ``TurnTakingConfig``."""
    cfg = replace(cfg, **kw) if cfg is not None else TurnTakingConfig.preset(preset, **kw)
    sr, P = lang.sr, cfg.matrix()
    p0 = np.asarray(cfg.p_moves, np.float64) if cfg.transition is None else P.mean(0)
    n_o = rng.choices(range(len(cfg.p_others)), weights=cfg.p_others)[0]
    spks = rng.sample(range(len(lang.speakers)), 1 + n_o)
    others = list(range(1, 1 + n_o))
    gain = {c: rng.uniform(0.5, 1.0) for c in others}
    tracks, free = [], {}  # (column, start_sample, audio, act); column -> sample where it is free again
    ev = dict(n_others=n_o, moves_drawn=[], moves=[], hesitations=[], th_pauses=[], fillers=0, cutoffs=0,
              repetitions=0, corrections=0, n_words=0, n_boundaries=0, decoys=0, lowered_final=False,
              backchannels=0, bystanders=0, next_speaker=0, lead_other=0)
    last_z = [None]

    def draw() -> str:
        row = P[MOVES.index(last_z[0])] if last_z[0] else p0 / p0.sum()
        z = MOVES[rng.choices(range(4), weights=row.tolist())[0]]
        last_z[0] = z
        ev["moves_drawn"].append(z)
        return z

    def place(col, st, x, a):
        st = max(st, free.get(col, st))  # no self-overlap
        tracks.append((col, st, x, a))
        free[col] = st + len(x)
        return st, st + len(x)

    def start_after(z, span):
        if z == "IR":
            return span[1] - int(_trunc_exp(rng, cfg.beta_ir) * (span[1] - span[0]))
        if z == "TS":
            return span[1] + int(sr * _clipped_exp(rng, cfg.beta_ts, cfg.gap_clip))
        p = _clipped_exp(rng, cfg.beta_th, cfg.pause_clip)
        ev["th_pauses"].append(round(p, 3))
        return span[1] + int(sr * p)

    def utter_other(col, st, lowered=None):
        low = rng.random() < cfg.p_final_lowered if lowered is None else lowered
        x, a = render_words(lang, _words(lang, rng, cfg.other_words), spks[col], rng,
                            "lowered" if low else "plain", amp=gain[col])
        return place(col, st, x, a)

    def moves_after(span, cur):
        """Draw moves after an utterance of ``cur``; backchannels are realised here (the floor stays),
        the first non-BC move is returned (TH if nobody else can take the floor)."""
        nbc = 0
        while True:
            z = draw()
            cands = [c for c in others if c != cur]
            if z == "BC":
                if cands and nbc < cfg.max_backchannels and _backchannel_in(span, cands):
                    nbc += 1
                    ev["moves"].append("BC")
                    continue
                z = "TH" if cur == 0 else "TS"
            if not cands and cur == 0:
                z = "TH"
            ev["moves"].append(z)
            return z

    def _backchannel_in(span, cands):
        col = rng.choice(cands)
        x, a = backchannel(lang, spks[col], rng, gain[col])
        lo, hi = max(span[0], free.get(col, span[0])), span[1] - len(x)
        if lo > span[1] - len(x) // 2:
            return False
        place(col, rng.randint(lo, max(lo, hi)), x, a)  # uniform within the preceding utterance
        ev["backchannels"] += 1
        return True

    # ---- optional lead-in by another speaker, then the floor goes to the primary
    start = 0
    if others and rng.random() < cfg.p_lead_other:
        col = rng.choice(others)
        span = utter_other(col, 0)
        ev["lead_other"] = 1
        z = draw()
        z = z if z in ("TS", "IR") else "TS"
        ev["moves"].append(z)
        start = start_after(z, span)

    # ---- primary episode: segments, with FastMSS moves after each one
    n_seg = rng.choices(range(len(cfg.p_hes)), weights=cfg.p_hes)[0] + 1
    seg_words = [_words(lang, rng, cfg.words_per_segment) for _ in range(n_seg)]
    verbatim, clean, carry, prim_spans = [], [], None, []
    for j in range(n_seg):
        final = j == n_seg - 1
        clean += seg_words[j]
        x, a, toks, carry = _primary_segment(lang, ([carry] if carry else []) + seg_words[j], spks[0], rng,
                                             cfg, final, ev)
        span = place(0, start, x, a)
        prim_spans.append(span)
        verbatim += toks
        z = moves_after(span, 0)
        if final:  # closing move: another speaker takes the floor (TS / IR) or silence (TH)
            if z in ("TS", "IR"):
                utter_other(rng.choice(others), start_after(z, span), lowered=True)
                ev["next_speaker"] += 1
            break
        if z == "TH":
            start = start_after("TH", span)
            continue
        # excursion: other speakers hold the floor mid-episode, then it returns to the primary
        col, st, k = rng.choice(others), start_after(z, span), 0
        while True:
            span_o = utter_other(col, st)
            k += 1
            ev["bystanders"] += 1
            z2 = moves_after(span_o, col)
            if z2 == "TH" and k < cfg.max_excursion:
                st = start_after("TH", span_o)
                continue
            if z2 == "TH":
                z2 = "TS"  # excursion cap reached: forced switch back
                ev["moves"][-1] = "TS"
            nxt = rng.choice([0] + [c for c in others if c != col]) if k < cfg.max_excursion else 0
            if nxt == 0:
                start = start_after(z2, span_o)
                break
            col, st = nxt, start_after(z2, span_o)
    ev["hesitations"] = [round((max(b[0], a[1]) - a[1]) / sr, 3) for a, b in zip(prim_spans, prim_spans[1:])]
    p_end = prim_spans[-1][1]

    # ---- mix
    lead = int(sr * rng.uniform(0.2, 0.6))
    off = lead - min(st for _, st, _, _ in tracks)
    N = max(p_end + off + int(cfg.trail_sec * sr), max(st + off + len(x) for _, st, x, _ in tracks) + int(0.3 * sr))
    mix = np.zeros(N, np.float32)
    T = lang.n_frames(N)
    spk_t = np.zeros((T, 4), np.float32)
    full = [np.zeros(N, bool) for _ in spks]
    for col, st, x, a in tracks:
        mix[st + off: st + off + len(x)] += x
        full[col][st + off: st + off + len(x)] |= a
    for col in range(len(spks)):
        spk_t[:, col] = lang.frames(full[col], T)
    mix += rng.uniform(0.002, 0.01) * np.random.default_rng(rng.randrange(1 << 30)).standard_normal(N).astype(np.float32)

    prim = spk_t[:, 0].copy()
    nz = np.nonzero(prim)[0]
    onset, end = int(nz[0]), int(nz[-1]) + 1
    eot = np.zeros(T, np.float32)
    eot[end:] = 1
    others_act = spk_t[:, 1:].max(1)
    n_act = (spk_t > 0).sum(1)
    spans = {}
    for col, st, x, _ in tracks:
        spans.setdefault(col, []).append((st, st + len(x)))
    ev["self_overlaps"] = sum(b[0] < a[1] for sp in spans.values() for a, b in zip(sorted(sp), sorted(sp)[1:]))
    ev.update(overlap_frames=int(((prim > 0) & (others_act > 0)).sum()),
              speech_frames=int((n_act > 0).sum()), overlap2_frames=int((n_act > 1).sum()),
              other_ends_before_primary=int(any(
                  (np.nonzero(spk_t[:, c])[0].max() + 1 < end) for c in range(1, len(spks)) if spk_t[:, c].any())),
              other_speech_after_end=int(others_act[end:].any()),
              max_midturn_silence_frames=int(_max_run(prim[onset:end] == 0)))
    return dict(audio=mix, text=" ".join(verbatim), text_clean=" ".join(clean), spk_targets=spk_t,
                primary_act=prim, spk_act=prim, eot=eot, turn_end_frame=end, onset_frame=onset,
                speaker=spks[0], events=ev)


def _max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


class ConversationDataset(Sequence):
    """Lazy, deterministic dataset: conversation i is generated on access from seed (seed, i).
    Holds no audio (~2 ms to render one), so a 2500-conversation training set costs no memory."""

    def __init__(self, n: int, seed: int = 0, lang: ToneLanguage | None = None, preset: str = "default",
                 cfg: TurnTakingConfig | None = None, **kw):
        self.n, self.seed = n, seed
        self.cfg = replace(cfg, **kw) if cfg is not None else TurnTakingConfig.preset(preset, **kw)
        self.lang = lang or ToneLanguage(seed=0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        if not -self.n <= i < self.n:
            raise IndexError(i)
        return conversation(self.lang, random.Random(1_000_003 * (self.seed + 1) + (i % self.n)), self.cfg)


def conversation_dataset(n: int, seed: int = 0, lang: ToneLanguage | None = None, preset: str = "default",
                         **kw) -> ConversationDataset:
    """``preset``: default | asr | diar | flat; ``kw`` overrides any ``TurnTakingConfig`` field."""
    return ConversationDataset(n, seed, lang, preset=preset, **kw)


def event_stats(convs: list[dict]) -> dict:
    ev = [c["events"] for c in convs]
    hes = [h for e in ev for h in e["hesitations"]]
    n = len(ev)
    drawn = [z for e in ev if e["n_others"] for z in e["moves_drawn"]]
    real = [z for e in ev for z in e["moves"]]
    nb, nw = max(1, sum(e["n_boundaries"] for e in ev)), max(1, sum(e["n_words"] for e in ev))
    return dict(
        n=n, mean_sec=round(float(np.mean([len(c["audio"]) / 16000 for c in convs])), 2),
        others_0_1_2=[sum(e["n_others"] == k for e in ev) / n for k in range(3)],
        moves_drawn={z: round(drawn.count(z) / max(1, len(drawn)), 3) for z in MOVES},
        moves_realised={z: round(real.count(z) / max(1, len(real)), 3) for z in MOVES},
        overlap_fraction=round(sum(e["overlap2_frames"] for e in ev) / max(1, sum(e["speech_frames"] for e in ev)), 4),
        with_hesitation=sum(bool(e["hesitations"]) for e in ev) / n,
        hesitations_per_conv=len(hes) / n,
        hesitation_sec_mean=round(float(np.mean(hes)), 3) if hes else 0.0,
        hesitation_with_filler=sum(e["fillers"] for e in ev) / nb,
        hesitation_cutoff=sum(e["cutoffs"] for e in ev) / nb,
        hesitation_lowered_decoy=sum(e["decoys"] for e in ev) / nb,
        repetition_per_word=round(sum(e["repetitions"] for e in ev) / nw, 4),
        correction_per_word=round(sum(e["corrections"] for e in ev) / nw, 4),
        true_end_lowered=sum(e["lowered_final"] for e in ev) / n,
        with_backchannel=sum(e["backchannels"] > 0 for e in ev) / n,
        with_bystander=sum(e["bystanders"] > 0 for e in ev) / n,
        with_overlap=sum(e["overlap_frames"] > 0 for e in ev) / n,
        other_ends_before_primary=sum(e["other_ends_before_primary"] for e in ev) / n,
        other_speech_after_end=sum(e["other_speech_after_end"] for e in ev) / n,
        next_speaker=sum(e["next_speaker"] > 0 for e in ev) / n,
        midturn_silence_ge_1s=sum(e["max_midturn_silence_frames"] * FRAME_SEC >= 1.0 for e in ev) / n,
    )


class ConversationCollate(Collate):
    """data.Collate plus the conversation-only keys (eot, primary_act, turn_end_frame, onset_frame)."""

    def __call__(self, batch):
        out = super().__call__(batch)
        for k in ("eot", "primary_act"):
            if k in batch[0]:
                out[k], out[k + "_len"] = self._frames(batch, k)
        for k in ("turn_end_frame", "onset_frame"):
            if k in batch[0]:
                out[k] = torch.tensor([ex[k] for ex in batch])
        return out


# --------------------------------------------------------------------------- eot-bench style metric
def silence_scores(act: np.ndarray, onset: int = 0) -> np.ndarray:
    """Silence-timeout detector as a score: length (frames) of the silent run ending at each frame
    (0 before ``onset``). Firing when score > k - 0.5 == 'k frames of silence observed'."""
    act = np.asarray(act) > 0.5
    s = np.zeros(len(act), np.float32)
    run = 0
    for t in range(len(act)):
        run = 0 if (act[t] or t < onset) else run + 1
        s[t] = run
    return s


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q, method="inverted_cdf")) if len(x) else float("inf")


def pause_runs(hes, min_frames: int = 1) -> list[tuple[int, int]]:
    """Within-turn pauses [start, end) (frames) = runs of a ``hes`` mask (AMI turn examples: the primary's silent
    frames inside hesitations >= hes_gap = 0.3 s, datasets/ami.py), keeping runs of >= ``min_frames``."""
    h = (np.asarray(hes) > 0.5).astype(np.int8)
    d = np.diff(np.concatenate([[0], h, [0]]))
    return [(int(a), int(b)) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]) if b - a >= min_frames]


def floor_stratum(spk_targets, end: int, horizon: int = 13, primary_col: int = 0) -> str:
    """Floor state at a turn end from the (T, S) multi-speaker activity (labels; scoring side only):
    'open' = no other speaker active in [end, end + horizon) (13 frames = 1.04 s: the agent should reply),
    'overlap' = else, and another speaker is active on the primary's last frame (end - 1), 'switch' = else
    (someone else starts within the horizon). floor-taken = overlap + switch."""
    y = np.asarray(spk_targets) > 0.5
    if y.ndim == 1 or y.shape[1] <= 1:
        return "open"
    o = np.delete(y, primary_col, 1).any(1)
    if not o[end: end + horizon].any():
        return "open"
    return "overlap" if end >= 1 and o[end - 1] else "switch"


def eot_outcomes(scores: list[np.ndarray], onsets, ends, thresholds, frame_ms: float = FRAME_SEC * 1000,
                 chunk: int = 1, post_end_frames: int | None = None, pauses=None) -> dict:
    """Per-conversation outcomes of eot_bench at every threshold θ (ths ascending):
    fc (n, J) bool   any frame in [onset, end) with score > θ (the per-TURN false cutoff);
    lat (n, J)       dead-air latency of the first post-end firing, inf = never (within ``post_end_frames`` frames
                     after the end if given, else the whole remaining track);
    pf (n, J) int    number of the conversation's within-turn pauses (``pauses[i]`` = [(start, end)] frames) with a
                     firing on one of their frames (per-PAUSE false cutoffs; zeros without ``pauses``);
    npause (n,)      number of pauses."""
    onsets, ends = np.asarray(onsets), np.asarray(ends)
    ths = np.sort(np.asarray(thresholds, np.float64))
    n = len(scores)
    fc = np.zeros((n, len(ths)), bool)
    lat = np.full((n, len(ths)), np.inf)
    pf = np.zeros((n, len(ths)), np.int64)
    npause = np.zeros(n, np.int64)
    for i, s in enumerate(scores):
        s = np.asarray(s, np.float64)
        o, e = int(onsets[i]), int(ends[i])
        pre = s[o:e]
        fc[i] = (pre.max() if len(pre) else -np.inf) > ths
        post = np.maximum.accumulate(s[e:]) if len(s) > e else np.zeros(0)
        if post_end_frames is not None:
            post = post[:int(post_end_frames)]
        idx = np.searchsorted(post, ths, side="right")  # first post-end frame with score > θ
        t = e + idx
        emit = (t // chunk + 1) * chunk
        lat[i] = np.where(idx < len(post), (emit - e) * frame_ms, np.inf)
        if pauses is not None:
            for a, b in pauses[i]:
                seg = s[max(a, 0): min(b, len(s))]
                if len(seg):
                    pf[i] += seg.max() > ths
                    npause[i] += 1
    return dict(ths=ths, fc=fc, lat=lat, pf=pf, npause=npause)


def outcome_metrics(fc, lat, pf=None, npause=None) -> dict:
    """eot-bench metrics of per-conversation outcomes at ONE threshold (fc (n,) bool, lat (n,)): per-turn FC rate,
    per-pause FC rate (if pauses), and P50 / P90 latency and miss rate over the conversations not cut off."""
    fc, lat = np.asarray(fc, bool), np.asarray(lat, np.float64)
    keep = lat[~fc]
    out = dict(n=int(len(fc)), fc_rate=round(float(fc.mean()), 4) if len(fc) else None,
               p50_ms=_q(keep, 0.5), p90_ms=_q(keep, 0.9),
               miss_rate=round(float(np.isinf(keep).mean()), 4) if len(keep) else 1.0)
    if pf is not None:
        tot = int(np.sum(npause))
        out.update(n_pauses=tot, fc_per_pause=round(float(np.sum(pf) / tot), 4) if tot else None)
    return out


def bootstrap_ci(fc, lat, n_boot: int = 1000, seed: int = 0, idx=None) -> dict:
    """95 % percentile-bootstrap CIs (resampling conversations) of miss rate and P50 latency at one threshold."""
    fc, lat = np.asarray(fc, bool), np.asarray(lat, np.float64)
    n = len(fc)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n)) if idx is None else idx
    miss, p50 = np.empty(len(idx)), np.empty(len(idx))
    for b, ii in enumerate(idx):
        keep = lat[ii][~fc[ii]]
        miss[b] = np.isinf(keep).mean() if len(keep) else 1.0
        p50[b] = _q(keep, 0.5)
    return dict(miss_rate_ci=[round(_q(miss, 0.025), 4), round(_q(miss, 0.975), 4)],
                p50_ms_ci=[_q(p50, 0.025), _q(p50, 0.975)])


def eot_bench(scores: list[np.ndarray], onsets, ends, frame_ms: float = FRAME_SEC * 1000, chunk: int = 1,
              max_fc: float = 0.05, fixed_latency_ms: float = 400.0, thresholds=None, fc_unit: str = "turn",
              pauses=None, post_end_frames: int | None = None, censored=None, groups: dict | None = None,
              n_boot: int = 0, seed: int = 0, return_outcomes: bool = False) -> dict:
    """eot-bench style sweep.

    Per conversation and threshold θ: a *false cutoff* is any frame t in [onset, end) with score > θ
    (the agent would talk over the user); otherwise the *dead-air latency* is the time from the true
    end to when the first post-end firing is available: (emit(t) - end) * frame_ms, where
    emit(t) = (t // chunk + 1) * chunk accounts for the encoder's chunked look-ahead (chunk = R + 1
    frames; 1 for oracle activity). Never firing = latency inf. Latency percentiles are over the
    conversations that were not cut off (as in eot-bench).

    Reports the lowest-latency threshold with false-cutoff rate <= ``max_fc`` and the false-cutoff
    rate of the most conservative threshold whose P50 latency is <= ``fixed_latency_ms``.

    Options (eot-bench v2, research/archive/EOT_BENCH_V2.md; all off by default = the output above, unchanged):
      pauses           per conversation [(start, end)] within-turn pause frames (``pause_runs(hes)``): adds
                       ``fc_per_pause`` = pauses with a firing on one of their frames / all pauses;
      fc_unit          'turn' (default) | 'pause': which FC rate ``max_fc`` (and the fixed-latency point) constrains;
      post_end_frames  post-end scoring window: a firing later than this many frames after the end = miss
                       (default: the whole remaining track, i.e. the window's trail);
      censored         (n,) bool, the conversation's audio ended before the post-end window for an artificial
                       reason (window cut): adds ``miss_censored`` = how many misses are such conversations;
      groups           {name: (n,) bool}: every metric per stratum at the chosen thresholds;
      n_boot           > 0: 95 % bootstrap CIs (conversations resampled) of miss rate and P50 per point / stratum;
      return_outcomes  adds the per-conversation outcome arrays under ``_outcomes`` (not JSON-serialisable).
    """
    assert fc_unit in ("turn", "pause"), fc_unit
    if fc_unit == "pause" and pauses is None:
        raise ValueError("fc_unit='pause' needs pauses")
    if thresholds is None:
        allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in scores]))
        thresholds = allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))
    oc = eot_outcomes(scores, onsets, ends, thresholds, frame_ms, chunk, post_end_frames, pauses)
    ths, fc, lat = oc["ths"], oc["fc"], oc["lat"]
    n = len(scores)
    fc_rate = fc.mean(0)
    fc_pause = oc["pf"].sum(0) / max(1, int(oc["npause"].sum()))
    fc_sel = fc_rate if fc_unit == "turn" else fc_pause
    stats = []
    for j in range(len(ths)):
        keep = lat[~fc[:, j], j]
        stats.append((_q(keep, 0.5), _q(keep, 0.9), float(np.isinf(keep).mean()) if len(keep) else 1.0))
    extra = pauses is not None or censored is not None or groups or n_boot
    bidx = np.random.default_rng(seed).integers(0, n, (n_boot, n)) if n_boot else None

    def v2(j) -> dict:
        d = {}
        if pauses is not None:
            d["fc_per_pause"] = round(float(fc_pause[j]), 4)
            d["n_pauses"] = int(oc["npause"].sum())
        if censored is not None:
            m = np.isinf(lat[:, j]) & ~fc[:, j]
            d["miss_censored"] = int((m & np.asarray(censored, bool)).sum())
            d["n_miss"] = int(m.sum())
        if n_boot:
            d.update(bootstrap_ci(fc[:, j], lat[:, j], idx=bidx))
        if groups:
            d["strata"] = {}
            for g, mask in groups.items():
                mask = np.asarray(mask, bool)
                gm = outcome_metrics(fc[mask, j], lat[mask, j], oc["pf"][mask, j] if pauses is not None else None,
                                     oc["npause"][mask] if pauses is not None else None)
                if n_boot and mask.any():
                    gm.update(bootstrap_ci(fc[mask, j], lat[mask, j], n_boot, seed))
                d["strata"][g] = gm
        return d

    def point(j):
        if j is None:
            return dict(threshold=None, fc_rate=None, p50_ms=float("inf"), p90_ms=float("inf"), miss_rate=None)
        p = dict(threshold=round(float(ths[j]), 5), fc_rate=round(float(fc_rate[j]), 4),
                 p50_ms=stats[j][0], p90_ms=stats[j][1], miss_rate=round(stats[j][2], 4))
        return {**p, **v2(j)} if extra else p

    ok = [j for j in range(len(ths)) if fc_sel[j] <= max_fc]
    best = min(ok, key=lambda j: (stats[j][0], stats[j][1])) if ok else None
    fixed = [j for j in range(len(ths)) if stats[j][0] <= fixed_latency_ms]
    fixed_j = min(fixed, key=lambda j: (fc_sel[j], stats[j][0])) if fixed else None
    sub = np.unique(np.linspace(0, len(ths) - 1, min(len(ths), 60)).astype(int))
    out = dict(n=n, at_max_fc=point(best), max_fc=max_fc, fixed_latency_ms=fixed_latency_ms,
               at_fixed_latency=point(fixed_j),
               curve=[(round(float(ths[j]), 4), round(float(fc_rate[j]), 4), stats[j][0], stats[j][1])
                      for j in sub])
    if extra or fc_unit != "turn" or post_end_frames is not None:
        out.update(fc_unit=fc_unit, post_end_frames=post_end_frames)
    if return_outcomes:
        out["_outcomes"] = oc
    return out


# --------------------------------------------------------------------------- OR-combination of two detectors
def hybrid_fire_frame(p, silence, theta: float, k: float, start: int = 0) -> int:
    """First frame t >= ``start`` where the turn head's probability p[t] >= theta OR the primary has been silent
    silence[t] >= k frames (``silence_scores``); len if neither fires. Equals min(head-only frame, timeout-only
    frame), so it is never later than the timeout alone (the rule of serve.py's ``hybrid`` policy)."""
    p, s = np.asarray(p, np.float64), np.asarray(silence, np.float64)
    n = min(len(p), len(s))
    hit = np.nonzero((p[start:n] >= theta) | (s[start:n] >= k))[0]
    return int(start + hit[0]) if len(hit) else n


def _default_ths(scores) -> np.ndarray:
    """eot_bench's default threshold candidates for a list of score tracks."""
    allv = np.unique(np.concatenate([np.asarray(s, np.float64) for s in scores]))
    return allv if len(allv) <= 3000 else np.unique(np.quantile(allv, np.linspace(0, 1, 3000)))


def _pause_max(scores, pauses) -> np.ndarray:
    """Max score on each within-turn pause (P,), pauses counted exactly as in eot_outcomes."""
    mx = []
    for s, ps in zip(scores, pauses):
        s = np.asarray(s, np.float64)
        for a, b in ps:
            seg = s[max(a, 0): min(b, len(s))]
            if len(seg):
                mx.append(float(seg.max()))
    return np.asarray(mx, np.float64)


def eot_bench_or(scores_a, scores_b, onsets, ends, thresholds_a=None, thresholds_b=None, chunk_a: int = 1,
                 chunk_b: int = 1, max_fc: float = 0.05, fixed_latency_ms: float = 400.0, fc_unit: str = "turn",
                 pauses=None, frame_ms: float = FRAME_SEC * 1000, never: bool = True,
                 break_ties_by_miss: bool = True, table: bool = False) -> dict:
    """eot_bench of the OR of two detectors, swept jointly over (θ_a, θ_b): the decision fires at the first frame
    where score_a > θ_a OR score_b > θ_b (e.g. a = turn-head probability, b = silence_scores, where θ_b = k - 1 means
    'silent >= k frames'). Per conversation: false cutoff = either fires on [onset, end); dead-air latency = the
    smaller of the two detectors' latencies, each with its own emission rule (chunk_a / chunk_b as in eot_bench; for
    another emission rule pass tracks re-indexed like eval_stage1.eot_bench_emit, with chunk 1).

    Thresholds default to eot_bench's own candidates per detector. ``never`` adds θ = +inf to both sets, so each
    detector alone (a alone: θ_b = inf; b alone: θ_a = inf) is inside the sweep and the chosen point is never worse
    (P50, then P90) than either alone at the same FC budget. Operating point as in eot_bench: lowest P50, then P90,
    s.t. FC <= max_fc (per turn, or per pause with fc_unit='pause', which needs ``pauses``). Remaining ties go to the
    lower miss rate (``break_ties_by_miss``). With False, the first candidate in (θ_b, θ_a) ascending order wins,
    which on a one-detector sweep is exactly eot_bench's choice. ``table``: also every (θ_a, θ_b) point (small
    grids only)."""
    import warnings
    assert fc_unit in ("turn", "pause"), fc_unit
    if fc_unit == "pause" and pauses is None:
        raise ValueError("fc_unit='pause' needs pauses")
    ths = []
    for sc, th in ((scores_a, thresholds_a), (scores_b, thresholds_b)):
        th = _default_ths(sc) if th is None else np.unique(np.asarray(th, np.float64))
        ths.append(np.unique(np.append(th, np.inf)) if never or not len(th) else th)  # empty set = never fires
    ta, tb = ths
    oa = eot_outcomes(scores_a, onsets, ends, ta, frame_ms, chunk_a)
    ob = eot_outcomes(scores_b, onsets, ends, tb, frame_ms, chunk_b)
    if pauses is not None:
        pa, pb = _pause_max(scores_a, pauses), _pause_max(scores_b, pauses)
        fa_p, fb_p = pa[:, None] > ta[None], pb[:, None] > tb[None]  # (P, Ja), (P, Jb)
    FC, FCP, P50, P90, MISS = (np.zeros((len(tb), len(ta))) for _ in range(5))
    for jb in range(len(tb)):
        fc = oa["fc"] | ob["fc"][:, jb: jb + 1]
        lat = np.minimum(oa["lat"], ob["lat"][:, jb: jb + 1])
        m = (~fc).sum(0)
        with warnings.catch_warnings():  # all-cut-off columns: nan -> inf below
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanquantile(np.where(fc, np.nan, lat), [0.5, 0.9], axis=0, method="inverted_cdf")
        P50[jb], P90[jb] = np.where(m > 0, q[0], np.inf), np.where(m > 0, q[1], np.inf)
        MISS[jb] = np.where(m > 0, (np.isinf(lat) & ~fc).sum(0) / np.maximum(m, 1), 1.0)
        FC[jb] = fc.mean(0)
        if pauses is not None:
            FCP[jb] = (fa_p | fb_p[:, jb: jb + 1]).sum(0) / max(1, len(pa))
    sel = FC if fc_unit == "turn" else FCP

    def point(ij):
        if ij is None:
            return dict(threshold_a=None, threshold_b=None, fc_rate=None, p50_ms=float("inf"), p90_ms=float("inf"),
                        miss_rate=None)
        jb, ja = ij
        d = dict(threshold_a=round(float(ta[ja]), 5) if np.isfinite(ta[ja]) else None,  # None = never fires
                 threshold_b=round(float(tb[jb]), 5) if np.isfinite(tb[jb]) else None,
                 fc_rate=round(float(FC[jb, ja]), 4), p50_ms=float(P50[jb, ja]), p90_ms=float(P90[jb, ja]),
                 miss_rate=round(float(MISS[jb, ja]), 4))
        if pauses is not None:
            d["fc_per_pause"] = round(float(FCP[jb, ja]), 4)
        return d

    ok = [tuple(x) for x in np.argwhere(sel <= max_fc)]  # row-major = (θ_b, θ_a) ascending
    tie = (lambda ij: MISS[ij]) if break_ties_by_miss else (lambda ij: 0.0)
    best = min(ok, key=lambda ij: (P50[ij], P90[ij], tie(ij))) if ok else None
    fx = [tuple(x) for x in np.argwhere(P50 <= fixed_latency_ms)]
    fixed = min(fx, key=lambda ij: (sel[ij], P50[ij])) if fx else None
    out = dict(n=len(scores_a), fc_unit=fc_unit, max_fc=max_fc, at_max_fc=point(best),
               fixed_latency_ms=fixed_latency_ms, at_fixed_latency=point(fixed),
               n_thresholds=[int(len(ta)), int(len(tb))])
    if pauses is not None:
        out["n_pauses"] = int(len(pa))
    if table:
        out["table"] = [point((jb, ja)) for jb in range(len(tb)) for ja in range(len(ta))]
    return out


def pause_fire(scores, pauses, thresholds) -> tuple[np.ndarray, np.ndarray]:
    """(fire (P, J) bool, conv (P,)): per within-turn pause (counted exactly as in eot_outcomes) whether the detector
    fires on one of its frames at each threshold, and the conversation index of the pause."""
    ths = np.sort(np.asarray(thresholds, np.float64))
    mx, cv = [], []
    for i, (s, ps) in enumerate(zip(scores, pauses)):
        s = np.asarray(s, np.float64)
        for a, b in ps:
            seg = s[max(a, 0): min(b, len(s))]
            if len(seg):
                mx.append(float(seg.max()))
                cv.append(i)
    return np.asarray(mx, np.float64)[:, None] > ths[None], np.asarray(cv, np.int64)


def or_outcomes(oa: dict, ob: dict, pa=None, pb=None, pconv=None) -> dict:
    """Per-conversation outcomes of the OR of two detectors (hybrid: fire at the earlier of detector a at θ_a and
    detector b at θ_b) over their joint threshold grid, from each detector's eot_outcomes on the same conversations,
    horizon and emission indexing. Column j = jb * Ja + ja (θ_b outer, both ascending):
      fc  = fc_a | fc_b                         (either fires on [onset, end): a per-turn false cutoff)
      lat = min(lat_a, lat_b)                   (the earlier firing; inf only if neither fires)
      pf  = pauses where either fires           (needs pause_fire matrices pa (P, Ja), pb (P, Jb) and pconv (P,);
                                                 else zeros)
    ths (J, 2) = (θ_a, θ_b); npause from ``oa``. The format is eot_outcomes' (per-conversation arrays), so the same
    operating-point / cross-fitting code applies to it."""
    n, Ja, Jb = len(oa["fc"]), len(oa["ths"]), len(ob["ths"])
    fc = (oa["fc"][:, None, :] | ob["fc"][:, :, None]).reshape(n, Jb * Ja)
    lat = np.minimum(oa["lat"][:, None, :], ob["lat"][:, :, None]).reshape(n, Jb * Ja)
    pf = np.zeros((n, Jb * Ja), np.int32)
    if pa is not None and len(pa):
        for i in np.unique(pconv):
            r = pconv == i
            pf[i] = (pa[r][:, None, :] | pb[r][:, :, None]).sum(0).reshape(-1)
    ths = np.stack([np.tile(oa["ths"], Jb), np.repeat(ob["ths"], Ja)], 1)
    return dict(ths=ths, fc=fc, lat=lat, pf=pf, npause=np.asarray(oa["npause"]).copy())


def eot_outcomes_or(scores_a, scores_b, onsets, ends, thresholds_a, thresholds_b, frame_ms: float = FRAME_SEC * 1000,
                    chunk_a: int = 1, chunk_b: int = 1, post_end_frames: int | None = None, pauses=None) -> dict:
    """or_outcomes of two score-track lists (eot_outcomes of each, same horizon; per-pause firing if ``pauses``)."""
    oa = eot_outcomes(scores_a, onsets, ends, thresholds_a, frame_ms, chunk_a, post_end_frames, pauses)
    ob = eot_outcomes(scores_b, onsets, ends, thresholds_b, frame_ms, chunk_b, post_end_frames)
    if pauses is None:
        return or_outcomes(oa, ob)
    pa, pconv = pause_fire(scores_a, pauses, oa["ths"])
    pb, _ = pause_fire(scores_b, pauses, ob["ths"])
    return or_outcomes(oa, ob, pa, pb, pconv)


def baselines(convs: list[dict], **kw) -> dict:
    """Silence timeout on ground-truth PRIMARY activity (oracle speaker-aware VAD) and on any-speaker
    activity (plain VAD, speaker-unaware), both with zero look-ahead."""
    on = [c["onset_frame"] for c in convs]
    en = [c["turn_end_frame"] for c in convs]
    prim = [silence_scores(c["primary_act"], c["onset_frame"]) for c in convs]
    anyv = [silence_scores(c["spk_targets"].max(1), c["onset_frame"]) for c in convs]
    return {"timeout_primary_oracle": eot_bench(prim, on, en, **kw),
            "timeout_any_speaker_oracle": eot_bench(anyv, on, en, **kw)}


# --------------------------------------------------------------------------- CLI: final benchmark
def _main():
    import argparse
    import json

    from .heads.turn import turn_scores
    from .train import load_model, pick_device

    ap = argparse.ArgumentParser(description="eot-bench style evaluation on synthetic conversations")
    ap.add_argument("models", nargs="*", help=".afm archives with a turn head")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    convs = conversation_dataset(a.n, seed=a.seed, preset=a.preset)
    res = {"events": event_stats(convs), **baselines(convs)}
    dev = pick_device()
    for p in a.models:
        m = load_model(p, dev)
        name = next(k for k, v in m.head_cfg.items() if v["type"] == "turn")
        chunk = m.encoder.att_context_size[1] + 1
        sources = ["oracle"] + (["diar"] if m.heads[name].mode != "none" and "diar" in m.heads else [])
        for src in sources:
            sc = turn_scores(m, name, convs, act_source=src)
            res[f"{Path(p).stem}:{src}"] = eot_bench(sc, [c["onset_frame"] for c in convs],
                                                     [c["turn_end_frame"] for c in convs], chunk=chunk)
    for k, v in res.items():
        if k != "events":
            print(f"{k:45s} @<=5%FC: {v['at_max_fc']}  @P50<=400ms: {v['at_fixed_latency']}")
    print(json.dumps(res["events"]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    _main()
