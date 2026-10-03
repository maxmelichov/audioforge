"""Multi-speaker diarization in the served system (research/archive/DIARIZATION_FIX.md): stable speaker ids on the finals
(``--diar-labels registry``, audioforge.speaker_registry) and the last-stable-column rule under load shedding
(``--shed-diar hold``). Tiny models + scripted diarizers and a scripted voice embedder stand in for the checkpoints.
Pinned: no speaker-0 collapse under shedding, the same id for the same voice across a column permutation, and the
6-speaker count recovered on a synthetic 6-speaker mix that a 4-column diarizer cannot hold."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import audioforge.serve as S
from audioforge.serve import Engine, Session, SessionConfig, validate
from audioforge.speaker_registry import DEFAULT_THR, SpeakerRegistry, held_row, turn_column


@pytest.fixture(autouse=True)
def _seeded():
    """Fixed torch / numpy / random seeds per test: random inputs do not depend on which tests ran before."""
    import random

    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


ROOT = Path(__file__).resolve().parents[1]
SR = 16000
FR = 1280  # samples per 80 ms frame


def _h():
    spec = importlib.util.spec_from_file_location("test_serve_helpers", ROOT / "tests" / "test_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = _h()


# --------------------------------------------------------------------------- scripted voices, diarizers, embedder
def _voice(k: int, sec: float, seed: int = 0) -> np.ndarray:
    """Speaker k = noise band-passed around a speaker-specific centre frequency (the scripted embedder reads it)."""
    rng = np.random.default_rng(seed * 101 + k)
    n = int(sec * SR)
    x = rng.standard_normal(n + 2 * SR)
    f = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    c = 300 + 450 * k
    f *= np.exp(-((freqs - c) / 120) ** 2)
    y = np.fft.irfft(f)[SR: SR + n]
    return (y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.1).astype(np.float32)


def _script(turns, gap=0.6, total=None):
    """[(speaker, seconds), ...] -> (audio, per-frame speaker (-1 = silence), [(spk, f0, f1)])."""
    pieces, spans, t = [], [], 0
    for k, sec in turns:
        v = _voice(k, sec)
        pieces += [v, np.zeros(int(gap * SR), np.float32)]
        f0, f1 = t // FR, (t + len(v)) // FR
        spans.append((k, f0, f1))
        t += len(v) + int(gap * SR)
    x = np.concatenate(pieces)
    if total and len(x) < total * SR:
        x = np.concatenate([x, np.zeros(int(total * SR) - len(x), np.float32)])
    T = int(np.ceil(len(x) / FR))
    lab = np.full(T, -1)
    for k, f0, f1 in spans:
        lab[f0:f1] = k
    return x, lab, spans


class ScriptDiarizer:
    """StreamingDiarizer stand-in: column ``col_of(speaker, frame)`` = 0.9 on the speaker's frames, the rest 0.02;
    feed(skip=True) emits zeros like the real one (load shedding)."""

    def __init__(self, labels, num_spks=4, col_of=None, chunk_len=3, right_context=1):
        from audioforge.streaming_diar import AOSCConfig
        self.cfg = AOSCConfig(chunk_len=chunk_len, chunk_right_context=right_context)
        self.labels, self.S = labels, num_spks
        self.col_of = col_of or (lambda k, v: k % num_spks)
        self.total, self.done, self.skipped = 0, 0, 0

    def feed(self, samples, final=False, skip=False):
        from audioforge.data import ToneLanguage
        self.total += len(np.asarray(samples))
        avail = ToneLanguage.n_frames(self.total) if final and self.total else self.total // FR
        C, R = self.cfg.chunk_len, self.cfg.chunk_right_context
        out = []
        while avail - self.done >= C + R or (final and avail > self.done):
            n = min(C, avail - self.done)
            for v in range(self.done, self.done + n):
                p = np.zeros(self.S) if skip else np.full(self.S, 0.02)
                k = self.labels[v] if v < len(self.labels) else -1
                if k >= 0 and not skip:
                    p[self.col_of(k, v)] = 0.9
                out.append(p)
            self.done += n
            if skip:
                self.skipped += n
        return torch.tensor(np.array(out) if out else np.zeros((0, self.S)))


class BandEmbedder:
    """Scripted TitaNet stand-in: a unit vector over 8 frequency bands of the selected frames' audio, so the same
    scripted voice always lands near the same direction (cos ~ 1) and different voices are near-orthogonal."""

    dim = 8

    def frames(self, audio, idx_lists):
        audio = np.asarray(audio, np.float32)
        out = []
        for idx in idx_lists:
            seg = np.concatenate([audio[u * FR:(u + 1) * FR] for u in np.asarray(idx, np.int64)])
            spec = np.abs(np.fft.rfft(seg)) ** 2
            freqs = np.fft.rfftfreq(len(seg), 1 / SR)
            e = np.array([spec[(freqs >= 300 + 450 * k - 225) & (freqs < 300 + 450 * k + 225)].sum() for k in range(8)])
            out.append(e / (np.linalg.norm(e) + 1e-9))
        return np.stack(out).astype(np.float32)


class _Eng(Engine):
    def __init__(self, *a, diar_factory=None, **k):
        self._factory = diar_factory
        super().__init__(*a, **k)
        self.num_spks = getattr(diar_factory(), "S", self.num_spks)  # the scripted diarizer's columns (the model's, live)

    def make_diarizer(self):
        return self._factory()


_MODELS = {}


def _models():
    if not _MODELS:
        _MODELS["asr"], _MODELS["diar"] = H._talky_asr_model(), H._diar_model()
    return _MODELS["asr"], _MODELS["diar"]


def _engine(factory, **kw):
    asr, diar = _models()
    return _Eng(asr, diar, name="tiny", threads=1, diar_factory=factory, **kw)


def _run(eng, x, shed=0, block=2560, cfg=None):
    s = Session(eng, cfg or SessionConfig(timeout_ms=480))
    msgs = []
    for i in range(0, len(x), block):
        msgs += s.process(x[i:i + block], shed=shed, backlog_ms=0.0)
    msgs += s.finish()
    for m in msgs:
        validate(m, debug=eng.debug)
    return s, msgs


def _finals(msgs):
    return [m for m in msgs if m["type"] == "final" and m["text"].strip()]


def _gt(lab, msgs):
    """Reference speaker of each non-empty final = the speaker with most frames in [previous final, this final)."""
    out, prev = [], 0.0
    for m in msgs:
        if m["type"] != "final":
            continue
        a, b = int(prev / 0.08), int(np.ceil(m["t"] / 0.08))
        seg = lab[a:b]
        seg = seg[seg >= 0]
        prev = m["t"]
        if m["text"].strip():
            out.append(int(np.bincount(seg).argmax()) if len(seg) else None)
    return out


def _hungarian_acc(ids, gts):
    from scipy.optimize import linear_sum_assignment
    pairs = [(i, g) for i, g in zip(ids, gts) if i is not None and g is not None]
    I, G = sorted({i for i, _ in pairs}), sorted({g for _, g in pairs})
    M = np.zeros((len(I), len(G)))
    for i, g in pairs:
        M[I.index(i), G.index(g)] += 1
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / max(1, len(pairs))


# --------------------------------------------------------------------------- unit pieces
def test_registry_assigns_new_ids_below_threshold_and_reuses_above():
    reg = SpeakerRegistry(0.5)
    a, b = np.eye(8)[0], np.eye(8)[1]
    assert reg.assign(a) == (0, 1.0, True)
    assert reg.assign(b)[0] == 1 and reg.assign(b)[2] is False
    near = a * 0.9 + b * 0.1
    sid, conf, new = reg.assign(near)
    assert sid == 0 and not new and conf >= 0.5
    assert len(reg) == 2 and reg.state()["per_speaker"] == [2, 2]
    assert reg.assign(np.zeros(8)) == (None, None, False)  # a degenerate embedding never registers
    cap = SpeakerRegistry(0.99, max_speakers=2)
    cap.assign(a), cap.assign(b)
    assert cap.assign(np.eye(8)[2])[0] in (0, 1) and len(cap) == 2  # bounded


def test_turn_column_and_held_row():
    rows = np.array([[0.9, 0.1, 0, 0]] * 3 + [[0.1, 0.8, 0, 0]] * 5)
    assert turn_column(rows) == 1
    assert turn_column(np.full((4, 4), 0.1)) is None and turn_column(np.zeros((0, 4))) is None
    r = held_row(np.array([0.1, 0.9, 0.0, 0.0]), 0.95, primary=3, S=4)
    assert list(r) == [0, 0.95, 0, 0]  # the last real row's column carries the VAD
    assert list(held_row(np.zeros(4), 0.7, primary=2, S=4)) == [0, 0, 0.7, 0]  # else the primary
    assert list(held_row(None, 0.7, primary=None, S=4)) == [0.7, 0, 0, 0]  # else column 0
    assert list(held_row(np.zeros(4), 0.0, primary=1, S=4)) == [0, 0, 0, 0]  # silence stays silence


def test_validate_accepts_8_columns_and_the_optional_final_keys():
    validate({"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0.0] * 8, "primary": 7})
    with pytest.raises(ValueError):
        validate({"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0.0] * 8, "primary": 8})
    with pytest.raises(ValueError):
        validate({"type": "frame", "t": 0.08, "vad": 0.1, "eot": None, "speakers": [0.0] * 3, "primary": None})
    validate({"type": "final", "t": 1.0, "text": "a", "speaker": 5, "speaker_conf": 0.7, "diar_shed": False})
    validate({"type": "final", "t": 1.0, "text": "a", "speaker": None, "speaker_conf": None, "diar_shed": True})
    validate({"type": "stats", "rtf": 0.1, "chunk_ms_p50": 1, "chunk_ms_p95": 1, "first_partial_ms": None,
              "peak_rss_mb": 1, "speakers_seen": 6})
    assert "registry_failed" in S.ERROR_CODES and set(DEFAULT_THR) == {"spk", "titanet"}


def test_engine_rejects_bad_modes_and_keeps_legacy_by_default():
    asr, diar = _models()
    with pytest.raises(ValueError):
        Engine(asr, diar, name="t", threads=1, diar_labels="voice")
    with pytest.raises(ValueError):
        Engine(asr, diar, name="t", threads=1, shed_diar="drop")
    with pytest.raises(ValueError):  # the tiny ASR model has no speaker head
        Engine(asr, diar, name="t", threads=1, diar_labels="registry", diar_embed="spk")
    e = Engine(asr, diar, name="t", threads=1)
    assert (e.diar_labels, e.shed_diar, e.registry_embedder) == ("column", "vad", None)


# --------------------------------------------------------------------------- 1. no speaker-0 collapse under shedding
def test_no_speaker_0_collapse_under_shedding(monkeypatch):
    """Speaker 1 (column 1) talks for the whole second half. Legacy shedding relabels every final to 0; ``hold``
    keeps the last stable column; ``registry`` keeps the voice-keyed ids (with diar_shed flagged)."""
    monkeypatch.setattr(S, "SHED_RTF", 0.0)  # enter the shedding level regardless of the (unpaced) recent RTF
    x, lab, _ = _script([(0, 2.0), (1, 2.0), (1, 2.0), (1, 2.0)])
    kw = dict(diar_embed="titanet", embedder=BandEmbedder())

    def fin(**ekw):
        eng = _engine(lambda: ScriptDiarizer(lab), **ekw)
        s, msgs = _run(eng, x, shed=1)
        f = _finals(msgs)
        assert f and (s.degraded.get("shed_diar_frames", 0) > 0)
        return s, f

    s0, legacy = fin()
    assert {f["speaker"] for f in legacy} == {0} and all("diar_shed" not in f for f in legacy)  # the bug
    s1, hold = fin(shed_diar="hold")
    assert all(f["diar_shed"] for f in hold) and all(f["speaker_conf"] is None for f in hold)
    assert hold[-1]["speaker"] == 1 and {f["speaker"] for f in hold} == {0, 1}  # last stable column kept
    assert s1.diar.skipped < s0.diar.skipped  # level 1 halves the cadence instead of dropping the diarizer
    s2, reg = fin(shed_diar="hold", diar_labels="registry", **kw)
    ids = [f["speaker"] for f in reg]
    assert {ids[0]} != set(ids[1:]) and None not in ids and all(f["diar_shed"] for f in reg)
    assert _hungarian_acc(ids, _gt(lab, [m for m in reg])) == 1.0
    assert s2.stats()["speakers_seen"] == 2


# --------------------------------------------------------------------------- 2. stable ids across a column permutation
def test_stable_ids_across_a_column_permutation():
    """Two voices alternate; after frame 100 the diarizer swaps their columns (the AOSC re-numbering). The legacy
    label follows the column, so the same person changes id; the registry keeps one id per voice."""
    turns = [(0, 1.6), (1, 1.6)] * 5
    x, lab, spans = _script(turns)
    swap = 100

    def col_of(k, v):
        return k if v < swap else 1 - k

    def run(**ekw):
        eng = _engine(lambda: ScriptDiarizer(lab, col_of=col_of), **ekw)
        _, msgs = _run(eng, x)
        f = _finals(msgs)
        return [m["speaker"] for m in f], _gt(lab, msgs)

    ids, gts = run()
    by = {g: {i for i, gg in zip(ids, gts) if gg == g} for g in (0, 1)}
    assert by[0] == {0, 1} and by[1] == {0, 1}  # legacy: each voice got both ids
    ids, gts = run(diar_labels="registry", diar_embed="titanet", embedder=BandEmbedder())
    by = {g: {i for i, gg in zip(ids, gts) if gg == g} for g in (0, 1)}
    assert len(by[0]) == 1 and len(by[1]) == 1 and by[0] != by[1]  # one id per voice, across the swap
    assert _hungarian_acc(ids, gts) == 1.0


# --------------------------------------------------------------------------- 3. six speakers recovered
@pytest.mark.parametrize("num_cols", [4, 8])
def test_six_speaker_count_recovered_on_the_synthetic_mix(num_cols):
    """Six voices take turns twice each. A 4-column diarizer folds them onto 4 columns (legacy: <= 4 ids); the
    registry recovers 6 ids with the right per-turn speaker (Hungarian accuracy 1) and stats.speakers_seen 6."""
    order = [0, 1, 2, 3, 4, 5, 3, 0, 5, 1, 4, 2]
    x, lab, _ = _script([(k, 1.6) for k in order])
    eng = _engine(lambda: ScriptDiarizer(lab, num_spks=num_cols))
    _, msgs = _run(eng, x)
    ids, gts = [f["speaker"] for f in _finals(msgs)], _gt(lab, msgs)
    assert len(set(ids)) <= num_cols and (num_cols == 8 or len(set(ids)) <= 4)
    n_legacy = len(ids)
    eng = _engine(lambda: ScriptDiarizer(lab, num_spks=num_cols), diar_labels="registry", diar_embed="titanet",
                  embedder=BandEmbedder())
    # timeout_any: every speaker's turn ends (the primary-only timeout merges back-to-back turns of others)
    s, msgs = _run(eng, x, cfg=SessionConfig(turn_policy="timeout_any", timeout_ms=480))
    assert len(_finals(msgs)) >= n_legacy
    assert {m["policy"] for m in msgs if m["type"] == "turn_end"} <= {"timeout", "change"}
    ids, gts = [f["speaker"] for f in _finals(msgs)], _gt(lab, msgs)
    assert len(ids) >= 10 and None not in ids
    assert len(set(ids)) == 6 and s.stats()["speakers_seen"] == 6
    assert _hungarian_acc(ids, gts) == 1.0
    assert all(0 <= f["speaker_conf"] <= 1 for f in _finals(msgs))
    fr = [m for m in msgs if m["type"] == "frame"]
    assert all(len(m["speakers"]) == num_cols for m in fr)  # the frame carries every column of the diarizer


# --------------------------------------------------------------------------- 4. distillation option (section 5)
def test_sortformer_distill_option_is_bit_identical_when_unset_and_mixes_the_teacher_loss():
    from audioforge.heads.audio import SortformerHead
    torch.manual_seed(0)
    plain = SortformerHead(16, num_spks=4, d_hidden=16, n_layers=1, dropout=0.0).eval()
    dist = SortformerHead(16, num_spks=4, d_hidden=16, n_layers=1, dropout=0.0,
                          distill={"key": "diar_teacher", "weight": 0.5}).eval()
    dist.load_state_dict(plain.state_dict())
    enc, el = torch.randn(2, 30, 16), torch.tensor([30, 24])
    y = (torch.rand(2, 30, 4) > 0.7).float()
    teacher = torch.rand(2, 30, 8)
    l0 = plain.loss(enc, el, {"spk_targets": y})
    assert torch.equal(dist.loss(enc, el, {"spk_targets": y}), l0)  # no teacher in the batch: unchanged
    lt = dist.loss(enc, el, {"spk_targets": y, "diar_teacher": teacher})
    l_teacher = plain.loss(enc, el, {"spk_targets": teacher})  # the same PIL + sorted BCE on the soft targets
    assert torch.allclose(lt, 0.5 * l0 + 0.5 * l_teacher)
