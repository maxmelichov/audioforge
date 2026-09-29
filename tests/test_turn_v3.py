"""TurnHead v3 (research/STAGE1.md, n = 200: with a real streaming diarizer the head misses 69 % of turn ends at <= 5 %
false cutoffs vs 38 % for a silence timeout on the same track).

Diagnosis -> change under test:
  (1) no explicit "how long has the primary been silent"      -> duration_feats: causal counters of the INPUT track;
  (2) sees one primary track, blind to the other columns      -> act_columns: 4 (+ primary one-hot), ext_noise;
  (3) trained on offline tracks, evaluated on streaming ones  -> ext_tracks: stream (fallback offline);
  plus future_act_aux (VAP-style "active within h frames" targets). Everything defaults off, bit-identically.
"""
import copy
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from audioforge.conversation import conversation_dataset, eot_bench, silence_scores
from audioforge.data import Collate
from audioforge.datasets import ext_tracks as xt
from audioforge.heads.asr import RNNTHead
from audioforge.heads.turn import (
    NEVER,
    TurnHead,
    act_history,
    duration_counters,
    future_act_targets,
    log_clip,
    token_counts,
    turn_scores,
)
from audioforge.model import SpeechModel, ext_noise_cfg, ext_track_noise

sys.path.insert(0, str(Path(__file__).parent))
from test_cond import _cfg, _forward_v2, _spy_turn_batch  # noqa: E402

ROOT = Path(__file__).parent.parent
V3 = dict(act_columns=4, duration_feats=True, future_act_aux={"horizons": [2, 5], "weight": 0.3})


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


def _perturbed(model):
    with torch.no_grad():  # speaker kernels start as identity: perturb them so conditioning changes the encoding
        for p in model.encoder.speaker_kernels.parameters():
            p.add_(0.5 * torch.randn_like(p))
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, TurnHead) and m.v3_in is not None:  # zero-init would hide the v3 inputs in tests
                m.v3_in.weight.normal_(0, 0.5)
    return model


def _model(conditioning=None, turn=None, **kw):
    """test_cond's tiny speaker-aware turn model with optional TurnHead v3 flags."""
    from audioforge.tokenizer import train_tokenizer
    convs = conversation_dataset(4, seed=9)
    tok = train_tokenizer("char", [c["text"] for c in convs])
    cfg = _cfg(conditioning, **kw)
    cfg["heads"]["turn"].update(turn or {})
    torch.manual_seed(0)
    return _perturbed(SpeechModel(cfg, tok)), Collate(tok)(convs)


def _with_ext(b, seed=3, missing=(), S=4):
    """A batch with a random external track: spk_targets_ext (B,T,S), spk_prim_ext one-hot, spk_act_ext its column."""
    g = torch.Generator().manual_seed(seed)
    b = dict(b)
    B, T = b["spk_act"].shape
    cols = (torch.rand(B, T, S, generator=g) > 0.6).float() * 0.8 + 0.1 * torch.rand(B, T, S, generator=g)
    prim = torch.randint(0, S, (B,), generator=g)
    b["spk_targets_ext"] = cols
    b["spk_prim_ext"] = F.one_hot(prim, S).float()
    b["spk_act_ext"] = cols.gather(2, prim[:, None, None].expand(B, T, 1))[..., 0].clone()
    for i in missing:
        b["spk_act_ext"][i] = xt.MISSING
        b["spk_targets_ext"][i] = xt.MISSING
        b["spk_prim_ext"][i] = xt.MISSING
    for k in ("spk_act_ext", "spk_targets_ext", "spk_prim_ext"):
        b[k + "_len"] = b["spk_act_len"].clone() if k != "spk_prim_ext" else torch.full((B,), S)
    return b, prim


# --------------------------------------------------------------------------- (1) duration counters
def test_duration_counters_hand_made_blockwise_and_causal():
    a = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 1, 0]]).bool()
    since, run = duration_counters(a)
    assert since[0].tolist() == [NEVER + 1, NEVER + 2, 0, 0, 1, 2, 3, 0, 1]
    assert run[0].tolist() == [0, 0, 1, 2, 0, 0, 0, 1, 0]
    f = log_clip(since, 64)
    assert f[0, 0] == 1.0 and f[0, 2] == 0.0 and torch.isclose(f[0, 5], torch.log1p(torch.tensor(2.0)) / np.log1p(64))
    # block by block with the carried state == whole sequence (streaming)
    s0 = r0 = None
    parts = []
    for lo, hi in ((0, 3), (3, 4), (4, 9)):
        s, r = duration_counters(a[:, lo:hi], s0, r0)
        parts.append((s, r))
        s0, r0 = s[:, -1], r[:, -1]
    assert torch.equal(torch.cat([p[0] for p in parts], 1), since) and torch.equal(torch.cat([p[1] for p in parts], 1), run)
    # causal: changing frames >= k never changes the counters before k (random tracks)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 40, generator=g) > 0.5
    s1, r1 = duration_counters(x)
    y = x.clone()
    y[:, 17:] = ~y[:, 17:]
    s2, r2 = duration_counters(y)
    assert torch.equal(s1[:, :17], s2[:, :17]) and torch.equal(r1[:, :17], r2[:, :17])
    # brute-force definition on random tracks
    for b in range(3):
        last, runv = None, 0
        for t in range(40):
            last = t if x[b, t] else last
            runv = runv + 1 if x[b, t] else 0
            assert int(s1[b, t]) == (t - last if last is not None else NEVER + t + 1) and int(r1[b, t]) == runv


def test_duration_features_come_from_the_noisy_input_track():
    torch.manual_seed(1)
    head = TurnHead(16, mode="kernel", hidden=8, dropout=0.0, act_columns=4, duration_feats=True).eval()
    B, T = 2, 30
    cols = torch.rand(B, T, 4)  # soft diarizer probabilities
    prim = torch.tensor([2, 0])
    kern = torch.rand(B, T)  # the track fed to the encoder kernels
    v3, st = head.v3_features(T, kern, cols, prim)
    assert v3.shape == (B, T, 8 + 6)
    pc = cols.gather(2, prim[:, None, None].expand(B, T, 1))[..., 0]
    oth = cols.clone()
    oth[torch.arange(B), :, prim] = 0
    for j, trk in enumerate((pc, oth.max(-1).values, kern)):
        s, r = duration_counters(trk > 0.5)
        assert torch.allclose(v3[..., 8 + 2 * j], log_clip(s, 64)) and torch.allclose(v3[..., 9 + 2 * j], log_clip(r, 64))
    assert set(st) == {"prim", "other", "kernel"}
    # single-track head: counters of spk_act only
    h1 = TurnHead(16, mode="kernel", hidden=8, dropout=0.0, duration_feats=True).eval()
    v, _ = h1.v3_features(T, kern)
    assert v.shape == (B, T, 2) and h1.v3_in.in_features == 2 and h1.needs_act and not h1.needs_cols
    with pytest.raises(ValueError):
        h1.v3_features(T, None)


# --------------------------------------------------------------------------- (2) act_columns + primary one-hot
def test_act_columns_shapes_one_hot_and_batch_invariance():
    torch.manual_seed(0)
    head = TurnHead(32, mode="kernel", hidden=24, dropout=0.0, act_columns=4, duration_feats=True).eval()
    head.v3_in.weight.data.normal_(0, 0.5)
    lens = [17, 30, 9]
    B, T = 3, 30
    enc, cols, prim = torch.randn(B, T, 32), torch.rand(B, T, 4), torch.tensor([3, 0, 1])
    act = cols.gather(2, prim[:, None, None].expand(B, T, 1))[..., 0]
    v3, _ = head.v3_features(T, act, cols, prim)
    assert torch.equal(v3[..., :4], cols) and torch.equal(v3[..., 4:8], F.one_hot(prim, 4).float()[:, None].expand(B, T, 4))
    z = head(enc, torch.tensor(lens), act, None, cols, prim)
    assert z.shape == (B, T)
    assert torch.equal(z, head(enc, torch.tensor(lens), act, None, cols, F.one_hot(prim, 4).float()))  # one-hot prim
    for i, n in enumerate(lens):  # padded batch == alone
        one = head(enc[i:i + 1, :n], torch.tensor([n]), act[i:i + 1, :n], None, cols[i:i + 1, :n], prim[i:i + 1])
        assert torch.allclose(one[0], z[i, :n], atol=1e-5)
    # causal in every input
    c2, a2 = cols.clone(), act.clone()
    c2[:, 12:] = 1 - c2[:, 12:]
    a2[:, 12:] = 1 - a2[:, 12:]
    z2 = head(enc, torch.tensor(lens), a2, None, c2, prim)
    assert torch.allclose(z[:, :12], z2[:, :12], atol=1e-6) and not torch.allclose(z[:, 12:], z2[:, 12:])
    # the primary column matters (same track, other enrollment -> other scores)
    assert not torch.allclose(z, head(enc, torch.tensor(lens), act, None, cols, (prim + 1) % 4))
    assert head.decode(enc, torch.tensor(lens), spk_act=act) is None  # cols missing
    with pytest.raises(ValueError):
        head(enc, torch.tensor(lens), act)
    # fewer diarizer columns than act_columns: zero-padded
    z3 = head(enc, torch.tensor(lens), act, None, cols[..., :3], prim.clamp(max=2))
    assert torch.equal(z3, head(enc, torch.tensor(lens), act, None, F.pad(cols[..., :3], (0, 1)), prim.clamp(max=2)))


def test_attach_primary_one_hot_and_stream_fallback(tmp_path):
    rng = np.random.default_rng(0)

    def ex(start, T=30):
        a = np.zeros(T, np.float32)
        a[5:18] = 1
        return {"audio": np.zeros(T * 1280, np.float32), "spk_act": a, "meeting": "ES2003b", "start": start}

    exs = [ex(1.0), ex(2.0), ex(3.0)]
    off = [rng.random((30, 4)).astype(np.float32) for _ in exs]
    st = rng.random((30, 4)).astype(np.float32)
    for e, p in zip(exs[:2], off):
        np.save(xt.track_path(tmp_path, xt.example_key(e)), p)
    np.save(xt.track_path(tmp_path, xt.example_key(exs[0]), "stream"), st)
    res = xt.attach(exs, directory=tmp_path, source="stream", fallback="offline")
    assert res == {"found": 2, "missing": 1, "fallback": 1}
    assert np.array_equal(exs[0]["spk_targets_ext"], st) and np.array_equal(exs[1]["spk_targets_ext"], off[1])
    for e, p in ((exs[0], st), (exs[1], off[1])):
        c = xt.enroll_column(p, e["spk_act"], 5, 18)
        assert np.array_equal(e["spk_prim_ext"], np.eye(4, dtype=np.float32)[c])
        assert np.array_equal(e["spk_act_ext"], e["spk_targets_ext"][:, c])
    assert (exs[2]["spk_prim_ext"] == xt.MISSING).all()
    assert xt.attach([ex(2.0)], directory=tmp_path, source="stream") == {"found": 0, "missing": 1}  # no fallback
    b = Collate(None)(exs)
    assert b["spk_prim_ext"].shape == (3, 4)
    # the recipe hook: `stream` prefers the streaming track and falls back to the offline one by default
    from audioforge.datasets import ami
    e2 = [ex(1.0), ex(2.0)]
    assert ami.attach_ext_tracks(e2, {"source": "stream", "dir": str(tmp_path)}, "train") == \
        {"ext_tracks": "stream", "found": 2, "missing": 0, "fallback": 1}
    assert ami.attach_ext_tracks([ex(2.0)], {"source": "stream", "dir": str(tmp_path), "fallback": None}, "train") == \
        {"ext_tracks": "stream", "found": 0, "missing": 1}


def test_conditioning_builds_the_same_tracks_columns():
    model, b = _model({"p_ext": 1.0, "flip": 0.05}, turn=V3)
    assert model.n_act_columns == 4
    model.train()
    b, prim = _with_ext(b, missing=(1,))
    batches = []
    _spy_turn_batch(model, batches)
    torch.manual_seed(3)
    out = model(b)
    assert torch.isfinite(out["loss"])
    tb = batches[0]
    act, cols, pr = tb["spk_act"], tb["spk_cols"], tb["spk_prim"]
    B, T, S = cols.shape
    assert S == 4 and act.shape == (B, T) and pr.shape == (B,)
    assert torch.equal(tb["spk_act_oracle"], b["spk_act"])  # labels stay clean
    sel = cols.gather(2, pr[:, None, None].expand(B, T, 1))[..., 0]
    assert torch.allclose(sel, act)  # the kernel track is the fed columns' primary column
    L0 = min(int(b["spk_act_len"][0]), T)
    assert int(pr[0]) == int(prim[0]) and torch.allclose(cols[0, :L0], b["spk_targets_ext"][0, :L0])  # ext item
    # item 1 has no track -> oracle columns, permuted, noisy primary track in the primary's new column
    orc = b["spk_targets"][1, :T].float()
    others = sorted(orc[:, 1:].T.tolist())
    got = [cols[1, :, k].tolist() for k in range(S) if k != int(pr[1])]
    L1 = min(int(b["spk_act_len"][1]), T)
    assert sorted([g[:L1] for g in got]) == sorted([o[:L1] for o in others])
    out["loss"].backward()
    assert model.heads["turn"].v3_in.weight.grad is not None and model.heads["turn"].fut.weight.grad is not None


@pytest.mark.parametrize("causal", [False, True], ids=["offline", "causal"])
def test_conditioning_columns_from_the_own_diar_head(causal):
    model, b = _model({"p_diar": 1.0, "diar_causal": causal}, turn=dict(V3, act_columns=5))  # 5 != num_spks 4
    model.train()
    batches = []
    _spy_turn_batch(model, batches)
    torch.manual_seed(2)
    model(b)
    tb = batches[0]
    cols, pr, act = tb["spk_cols"], tb["spk_prim"], tb["spk_act"]
    B, T, S = cols.shape
    assert S == 5 and (cols[..., 4] == 0).all()  # the diar head's 4 columns, zero-padded
    assert torch.allclose(cols.gather(2, pr[:, None, None].expand(B, T, 1))[..., 0], act)
    from audioforge.heads.turn import primary_column
    assert torch.equal(pr, primary_column(b["spk_targets"]).clamp(max=3))


def test_ext_noise_swaps_the_next_speaker_into_the_primary_slot():
    assert ext_noise_cfg(0.3)["p"] == 0.3 and ext_noise_cfg({"p": 0.2, "swap_max": 5})["swap_max"] == 5
    with pytest.raises(ValueError):
        ext_noise_cfg({"prob": 0.3})
    T = 30
    cols = torch.zeros(1, T, 4)
    cols[0, 2:10, 1] = 0.9  # primary (column 1) ends at frame 10
    cols[0, 11:25, 3] = 0.8  # the next speaker (column 3) starts at frame 11
    torch.manual_seed(0)
    for _ in range(20):
        out = ext_track_noise(cols, torch.tensor([1]), torch.tensor([T]), torch.tensor([10]), p=1.0, swap_max=8,
                              at_turn_end=1.0)
        diff = (out != cols).any(-1)[0].nonzero()[:, 0]
        assert torch.equal(out.sort(-1).values, cols.sort(-1).values)  # a swap: per-frame multiset unchanged
        assert torch.equal(out[..., [0, 2]], cols[..., [0, 2]])  # only the primary and the next speaker's columns
        if len(diff):
            assert int(diff.max()) - int(diff.min()) + 1 <= 8 and 6 <= int(diff.min()) <= 17
    leaked = [ext_track_noise(cols, torch.tensor([1]), torch.tensor([T]), torch.tensor([10]), p=1.0, swap_max=12,
                              at_turn_end=1.0)[0, 11:, 1].max() for _ in range(20)]
    assert max(leaked) > 0.5  # the next speaker lands in the primary's slot after the turn end
    assert torch.equal(ext_track_noise(cols, torch.tensor([1]), torch.tensor([T]), p=0.0), cols)
    d = ext_track_noise(cols, torch.tensor([1]), torch.tensor([T]), p=1.0, swap_max=1, drop=1.0, drop_max=3)
    assert d.sum() < cols.sum() + 1e-6
    # in the model: the kernel track is the corrupted primary column, labels stay clean
    model, b = _model({"p_ext": 1.0, "ext_noise": {"p": 1.0, "swap_max": 12}}, turn=V3)
    model.train()
    b, _ = _with_ext(b)
    batches = []
    _spy_turn_batch(model, batches)
    torch.manual_seed(4)
    model(b)
    tb = batches[0]
    B, T2, _ = tb["spk_cols"].shape
    sel = tb["spk_cols"].gather(2, tb["spk_prim"][:, None, None].expand(B, T2, 1))[..., 0]
    assert torch.allclose(sel, tb["spk_act"]) and torch.equal(tb["spk_act_oracle"], b["spk_act"])
    L = torch.minimum(b["spk_act_len"], torch.tensor(T2))
    assert any(not torch.allclose(tb["spk_cols"][i, :L[i]], b["spk_targets_ext"][i, :L[i]]) for i in range(B))


# --------------------------------------------------------------------------- streaming
def _v3_text_head(mode, **kw):
    torch.manual_seed(0)
    asr = RNNTHead(32, vocab_size=12, pred_hidden=16, joint_hidden=16).eval()
    head = TurnHead(32, mode=mode, hidden=24, history=4, dropout=0.0, use_text=True, k_tokens=4, text_dim=8,
                    **kw).eval()
    head.bind_asr(asr)
    head.v3_in.weight.data.normal_(0, 0.5)
    return head, asr


@pytest.mark.parametrize("mode,flags", [("kernel", dict(act_columns=4, duration_feats=True)),
                                        ("concat", dict(act_columns=1, duration_feats=True)),
                                        ("kernel", dict(act_columns=4, duration_feats=False))],
                         ids=["kernel-cols-dur", "concat-dur", "kernel-cols"])
def test_step_equals_offline(mode, flags):
    head, asr = _v3_text_head(mode, **flags)
    B, T, U = 2, 20, 5
    g = torch.Generator().manual_seed(1)
    enc, elen = torch.randn(B, T, 32, generator=g), torch.tensor([T, T])
    cols, prim = torch.rand(B, T, 4, generator=g), torch.tensor([1, 3])
    act = cols.gather(2, prim[:, None, None].expand(B, T, 1))[..., 0] if flags["act_columns"] > 1 else \
        (torch.rand(B, T, generator=g) > 0.4).float()
    y, yl = torch.randint(0, 12, (B, U), generator=g), torch.tensor([U, U])
    emit = torch.tensor([[1, 3, 6, 9, 12], [0, 2, 4, 10, 11]])
    n = token_counts(emit, yl, T)
    c = cols if flags["act_columns"] > 1 else None
    p = prim if flags["act_columns"] > 1 else None
    z = head(enc, elen, act, (y, n), c, p).sigmoid()
    gs, _ = asr.pred(asr.pred.prepend_sos(y))
    st = head.init_stream(B)
    probs = []
    for t in range(T):
        nt = n[:, t]
        gt = torch.stack([gs[b, nt[b]] for b in range(B)])
        prefix = [y[b, : nt[b]].tolist() for b in range(B)]
        probs.append(head.step(enc[:, t:t + 1], (gt[:, None], None), prefix, act[:, t:t + 1], state=st,
                               cols=c[:, t:t + 1] if c is not None else None, prim=p))
    assert torch.allclose(torch.cat(probs, 1), z, atol=1e-5)


def test_step_in_chunks_equals_offline_without_text():
    torch.manual_seed(2)
    head = TurnHead(32, mode="kernel", hidden=24, dropout=0.0, act_columns=4, duration_feats=True).eval()
    head.v3_in.weight.data.normal_(0, 0.5)
    B, T = 3, 23
    enc, cols, prim = torch.randn(B, T, 32), torch.rand(B, T, 4), torch.tensor([0, 2, 3])
    act = torch.rand(B, T)
    z = head(enc, torch.full((B,), T), act, None, cols, prim).sigmoid()
    st, out = head.init_stream(B), []
    for lo in range(0, T, 5):
        out.append(head.step(enc[:, lo:lo + 5], spk_act=act[:, lo:lo + 5], state=st, cols=cols[:, lo:lo + 5], prim=prim))
    assert torch.allclose(torch.cat(out, 1), z, atol=1e-5) and set(st.dur) == {"prim", "other", "kernel"}


# --------------------------------------------------------------------------- future activity auxiliary targets
def test_future_act_targets_hand_made_and_aux_loss():
    a = torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).float()
    tg, ok = future_act_targets(a, torch.tensor([10]), [1, 3])
    assert tg[0, :, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    assert ok[0, :, 0].tolist() == [True] * 9 + [False]  # frame 9's next frame is outside the item
    assert tg[0, :, 1].tolist() == [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert ok[0, :, 1].tolist() == [True] * 7 + [False] * 3
    tg, ok = future_act_targets(a, torch.tensor([6]), [3])  # padded item: frames >= 6 unknown
    assert ok[0, :, 0].tolist() == [True, True, True, True, False, False, False, False, False, False]
    assert tg[0, 3, 0] == 1 and tg[0, 4, 0] == 0  # frame 3: a positive inside the item; frame 4: future unknown
    # the aux loss: training only, gradient into the future head; others = spk_targets columns 1..
    torch.manual_seed(0)
    head = TurnHead(16, mode="kernel", hidden=8, dropout=0.0, future_act_aux={"horizons": [2, 5], "weight": 0.3})
    assert head.fut.out_features == 4 and head.v3_in is None
    enc = torch.randn(2, 12, 16)
    tgt = torch.zeros(2, 12, 4)
    tgt[:, 2:6, 0] = 1
    tgt[:, 7:10, 2] = 1
    batch = {"spk_act": tgt[..., 0], "spk_act_len": torch.tensor([12, 10]), "spk_targets": tgt}
    head.train()
    l_train = head.loss(enc, torch.tensor([12, 10]), batch)
    l_train.backward()
    assert head.fut.weight.grad is not None and head.fut.weight.grad.abs().sum() > 0
    head.eval()
    l_eval = head.loss(enc, torch.tensor([12, 10]), batch)
    head.fut = None
    assert torch.allclose(l_eval, head.loss(enc, torch.tensor([12, 10]), batch))  # no aux term outside training


# --------------------------------------------------------------------------- default off: bit identity
def test_default_off_is_bit_identical_and_v3_warm_starts_from_v2():
    h2 = TurnHead(32, mode="kernel", use_text=False)
    assert h2.v3_in is None and h2.fut is None and not any(k.startswith(("v3_in", "fut")) for k in h2.state_dict())
    for off in (dict(act_columns=1), dict(duration_feats=False), dict(future_act_aux=None), dict(future_act_aux={})):
        h = TurnHead(32, mode="kernel", use_text=False, **off)
        assert set(h.state_dict()) == set(h2.state_dict())
    # the tiny model's training loss is the pre-conditioning reference, with or without ext columns in the batch
    for cond in (None, {"p_ext": 1.0}):
        model, b = _model(cond)
        model.train()
        att = model.encoder.att_context_size
        torch.manual_seed(5), random.seed(5)
        ref = _forward_v2(model, copy.copy(b), att) if cond is None else model(_with_ext(b)[0], att_context_size=att)["loss"]
        torch.manual_seed(5), random.seed(5)
        bb = _with_ext(b)[0]
        if cond is None:
            out = model(bb, att_context_size=att)["loss"]
        else:  # the new keys (spk_targets_ext / spk_prim_ext) consume no randomness when unused
            bb.pop("spk_targets_ext"), bb.pop("spk_prim_ext")
            out = model(bb, att_context_size=att)["loss"]
        assert torch.equal(out, ref)
    # warm start: a v3 head loaded from a v2 head's weights (v3_in zero-init) gives exactly the v2 scores
    torch.manual_seed(0)
    v2 = TurnHead(32, mode="kernel", hidden=16, dropout=0.0).eval()
    v3 = TurnHead(32, mode="kernel", hidden=16, dropout=0.0, **V3).eval()
    missing, unexpected = v3.load_state_dict(v2.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"v3_in.weight", "fut.weight", "fut.bias"}
    enc, cols = torch.randn(2, 15, 32), torch.rand(2, 15, 4)
    assert torch.allclose(v2(enc, torch.tensor([15, 15])), v3(enc, torch.tensor([15, 15]), cols[..., 0], None, cols,
                                                               torch.tensor([0, 2])), atol=1e-6)


def _v2_reference_scores(model, name, convs):
    """TurnHead v2's decode written out (inp -> GRU -> out), independent of the v3 code paths."""
    from audioforge.heads.turn import decoded_text_state
    head = model.heads[name]
    b = Collate(model.tokenizer)(convs)
    enc, elen, hidden = model.encode(b["audio"], b["audio_len"], return_hidden=True)
    act = b["spk_act"].float()[:, : enc.shape[1]]
    act = F.pad(act, (0, enc.shape[1] - act.shape[1]))
    e = model.encode(b["audio"], b["audio_len"], spk_act=act)[0] if model.head_cfg[name].get("condition_on_speaker") \
        else model.head_input(name, enc, hidden)
    x = [e]
    if head.concat:
        x.append(act_history(act, e.shape[1], head.history))
    if head.use_text:
        y, n = decoded_text_state(model, head, b["audio"], b["audio_len"], enc, hidden, elen, act)
        x.append(head.text_frames(y, n))
    h, _ = head.rnn(head.inp(torch.cat(x, -1)))
    p = head.out(h).squeeze(-1).sigmoid()
    return [p[j, : min(int(elen[j]), len(c["spk_act"]))].numpy() for j, c in enumerate(convs)]


@pytest.mark.skipif(not (ROOT / "runs" / "turn_speaker_text_v3.afm").exists(), reason="existing tiny model not present")
def test_existing_tiny_model_loads_and_scores_unchanged():
    from audioforge.train import load_model
    torch.set_num_threads(2)
    model = load_model(ROOT / "runs" / "turn_speaker_text_v3.afm")  # strict load: the v2 state_dict still fits
    head = model.heads["turn"]
    assert head.v3_in is None and head.fut is None and model.n_act_columns == 1
    convs = conversation_dataset(3, seed=11)
    with torch.no_grad():
        got = turn_scores(model, "turn", convs, batch_size=3)
        ref = _v2_reference_scores(model, "turn", convs)
    assert all(np.array_equal(a, r) for a, r in zip(got, ref))


# --------------------------------------------------------------------------- evaluation
def test_turn_scores_v3_paths_and_eval_copy(ev):
    model, _ = _model(None, turn=V3)
    model.eval()
    convs = conversation_dataset(5, seed=9)
    rng = np.random.default_rng(4)
    cols = [rng.random((len(c["spk_act"]) + int(rng.integers(-2, 3)), 4)).astype(np.float32) for c in convs]
    prims = [int(rng.integers(0, 4)) for _ in convs]
    acts = [c_[:, p] for c_, p in zip(cols, prims)]
    got = turn_scores(model, "turn", convs, batch_size=2, act_override=acts, cols_override=cols, prim_override=prims)
    ref = ev.turn_scores_given_act(model, "turn", convs, acts, 2, cols, prims)
    assert all(np.array_equal(a, b) for a, b in zip(got, ref))
    with pytest.raises(ValueError):
        turn_scores(model, "turn", convs, batch_size=2, act_override=acts)
    # the oracle path feeds spk_targets (primary = column 0), the diar path all causal diar columns
    orc = turn_scores(model, "turn", convs, batch_size=2)
    via = turn_scores(model, "turn", convs, batch_size=2, act_override=[c["spk_act"] for c in convs],
                      cols_override=[c["spk_targets"] for c in convs], prim_override=[0] * len(convs))
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(orc, via))
    dia = turn_scores(model, "turn", convs, act_source="diar", batch_size=2)
    assert len(dia) == len(convs) and all(np.isfinite(d).all() for d in dia)


def test_duration_rule_baseline(ev):
    p = np.zeros((12, 3), np.float32)
    p[1:5, 1] = 0.9  # primary (column 1) talks on frames 1-4
    p[6:8, 2] = 0.8  # someone else on frames 6-7
    s = ev.duration_rule_scores(p, 1, onset=1)
    assert s.tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 7]  # silent run of the primary, 0 while others talk
    assert np.array_equal(s, silence_scores(p[:, 1], 1) * (p[:, [0, 2]].max(1) <= 0.5))
    r = eot_bench([s], [1], [5], thresholds=[3.5], max_fc=0.0)
    assert r["at_max_fc"]["p50_ms"] == (8 + 1 - 5) * 80  # fires at frame 8 (>= 4 silent frames, nobody else)


# --------------------------------------------------------------------------- recipe
def test_recipe_v3_yaml():
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3.yaml").read_text())
    old = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_on_sortformer.yaml").read_text())
    assert cfg["init"] == old["init"] and cfg["init"]["from"] == "runs/stage1_heads_pretrained.afm"
    assert not any("heads.turn".startswith(p) for p in cfg["init"]["pretrained_scope"])
    assert cfg["encoder"] == old["encoder"]
    t = cfg["heads"]["turn"]
    assert t["act_columns"] == 4 and t["duration_feats"] is True and t["use_text"] is True
    assert t["future_act_aux"] == {"horizons": [6, 12, 25], "weight": 0.3}
    ami_src = cfg["data"]["mix"][0]["ami"]
    assert ami_src["mode"] == "turn" and ami_src["ext_tracks"]["source"] == "stream"
    tr = cfg["trainer"]
    assert tr["max_steps"] == 2000 and tr["batch_size"] == 6 and tr["checkpoint_every"] == 500
    assert tr["conditioning"]["p_ext"] == 0.9 and tr["conditioning"]["flip"] == 0.01
    assert tr["wer_gate"] == old["trainer"]["wer_gate"]
    ab = (ROOT / "research" / "recipes" / "stage1_turn_v3_ablation.md").read_text()
    for arm in ("heads.turn.act_columns=1 heads.turn.duration_feats=true heads.turn.future_act_aux=null",
                "heads.turn.act_columns=4 heads.turn.duration_feats=false heads.turn.future_act_aux=null",
                "heads.turn.act_columns=4 heads.turn.duration_feats=true heads.turn.future_act_aux=null"):
        assert arm in ab
    SpeechModel({**_cfg(), "trainer": {"conditioning": tr["conditioning"]}}, _model()[0].tokenizer)  # keys accepted


def _tiny_stage1_afm(path):
    """A tiny stand-in for runs/stage1_heads_pretrained.afm: the same head names / types, 160 ms context."""
    from audioforge.tokenizer import train_tokenizer
    from audioforge.train import save_model
    tok = train_tokenizer("char", ["abcdefghijklmnopqrstuvwxyz0123456789' "])
    cfg = {"name": "tiny_stage1_standin",
           "preprocessor": {"n_mels": 80, "normalize": "NA"},
           "encoder": {"d_model": 32, "n_layers": 3, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                       "conv_norm": "layer",
                       "att_context_size": [70, 1], "att_context_sizes": [[70, 1]], "speaker_kernel_layers": [0, 2]},
           "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 16, "joint_hidden": 16, "max_symbols": 3, "weight": 0},
                     "ctc": {"type": "ctc", "weight": 0},
                     "vad": {"type": "frame", "key": "vad", "hidden": 8, "from_layers": "all", "weight": 0.5},
                     "eou": {"type": "frame", "key": "eou", "hidden": 8, "pos_weight": 5.0, "weight": 0.5},
                     "spk": {"type": "speaker", "num_speakers": 190, "from_layers": "all", "weight": 0.3},
                     "diar": {"type": "sortformer", "num_spks": 4, "d_hidden": 16, "n_layers": 1, "from_layers": "all",
                              "weight": 1.0},
                     "turn": {"type": "turn", "mode": "kernel", "use_text": True, "condition_on_speaker": True,
                              "k_tokens": 4, "text_dim": 64, "align": "greedy", "hidden": 96, "history": 8,
                              "pos_weight": 2.0, "weight": 1.0}}}
    torch.manual_seed(0)
    save_model(SpeechModel(cfg, tok), path)


@pytest.mark.skipif(not xt.has_tracks(split="train", source="stream"), reason="AMI train streaming tracks not cached")
def test_recipe_v3_smoke_on_ami(tmp_path):
    """3 steps of research/recipes/stage1_turn_v3.yaml on CPU: tiny stand-in init model, 8 AMI turn items with their cached
    streaming Sortformer tracks, everything but heads.turn frozen."""
    from audioforge.datasets.ami import recipe_data
    from audioforge.train import load_model, run_recipe
    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_v3.yaml").read_text())
    cfg["data"]["mix"][0]["ami"].update(n_train=8, seed=0)
    cfg["data"]["mix"][0]["ami"]["ext_tracks"]["require"] = True  # tracks present (stream, else offline)
    cfg["data"]["mix"][1]["synthetic"]["n_train"] = 8
    cfg["data"]["val"]["ami"]["n_val"] = 3
    probe = recipe_data({"data": {"ami": cfg["data"]["mix"][0]["ami"]}}, "train")
    assert len(probe) == 8 and all((e["spk_prim_ext"] >= 0).all() for e in probe)
    if xt.manifest(split="train").get("stream", {}).get("n") == xt.manifest(split="train").get("n_examples"):
        d = xt.cache_dir(None, "train")  # complete streaming cache: every item trains on its streaming track
        assert all(xt.track_path(d, xt.example_key(e), "stream").exists() for e in probe)
    rp = tmp_path / "v3_smoke.yaml"
    rp.write_text(yaml.safe_dump(cfg))
    init = tmp_path / "init.afm"
    _tiny_stage1_afm(init)
    ov = [f"init.from={init}", "trainer.max_steps=3", "trainer.device=cpu", "trainer.wer_gate=null",
          "trainer.log_every=1", "trainer.warmup_steps=1"]
    before = {k: v.clone() for k, v in load_model(init).state_dict().items()}
    model, metrics = run_recipe(str(rp), ov, out=str(tmp_path / "m.afm"))
    head = model.heads["turn"]
    assert head.act_columns == 4 and head.duration_feats and head.fut_horizons == [6, 12, 25] and model.n_act_columns == 4
    assert model.cond["p_ext"] == 0.9 and "eot_turn_p50_ms@5fc" in metrics
    sd = model.state_dict()
    assert all(torch.equal(sd[k].cpu(), v) for k, v in before.items() if not k.startswith("heads.turn."))  # frozen
    assert any(not torch.equal(sd[k].cpu(), v) for k, v in before.items() if k.startswith("heads.turn."))  # trains
    assert sd["heads.turn.v3_in.weight"].abs().sum() > 0  # left zero-init: the new inputs get gradient
    loaded = load_model(tmp_path / "m.afm")
    assert loaded.heads["turn"].act_columns == 4 and loaded.heads["turn"].fut is not None
