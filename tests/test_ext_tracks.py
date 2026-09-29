"""External diarizer tracks for the turn head (datasets/ext_tracks.py, model.py conditioning p_ext, heads/turn.py
turn_scores(act_override=...)): the turn head trains on the activity a real diarizer (cached NVIDIA Sortformer
output) produces, with the primary column picked exactly as scripts/research/eval_stage1.py picks it at evaluation."""
import copy
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.conversation import conversation_dataset
from audioforge.data import Collate
from audioforge.datasets import ext_tracks as xt
from audioforge.heads.turn import turn_scores
from audioforge.model import SpeechModel

sys.path.insert(0, str(Path(__file__).parent))
from test_cond import _cfg, _forward_v2, _model, _spy_encoder, _spy_turn_batch  # noqa: E402

ROOT = Path(__file__).parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ev():
    return _script("eval_stage1")


def _ex(T=30, meeting="ES2003b", start=12.345, on=5, end=18):
    act = np.zeros(T, np.float32)
    act[on:end] = 1
    return {"audio": np.zeros(T * 1280, np.float32), "spk_act": act, "meeting": meeting, "start": start}


# --------------------------------------------------------------------------- lookup / cropping
def test_key_lookup_crop_pad_and_missing(tmp_path):
    rng = np.random.default_rng(0)
    exs = [_ex(30), _ex(30, start=40.0), _ex(30, start=80.0)]
    assert xt.example_key(exs[0]) == "ES2003b_00012345_38400"
    long_ = rng.random((33, 4)).astype(np.float32)   # diarizer gives more frames -> cropped
    short = rng.random((28, 4)).astype(np.float32)   # fewer -> last frame repeated
    np.save(xt.track_path(tmp_path, xt.example_key(exs[0])), long_)
    np.save(xt.track_path(tmp_path, xt.example_key(exs[1])), short)
    np.save(xt.track_path(tmp_path, xt.example_key(exs[1]), "stream"), long_)
    res = xt.attach(exs, directory=tmp_path)
    assert res == {"found": 2, "missing": 1}
    a, b, c = exs
    assert a["spk_targets_ext"].shape == (30, 4) and np.array_equal(a["spk_targets_ext"], long_[:30])
    assert np.array_equal(b["spk_targets_ext"][:28], short) and np.array_equal(b["spk_targets_ext"][28:], short[[-1, -1]])
    col = xt.enroll_column(long_, a["spk_act"], 5, 18)
    assert np.array_equal(a["spk_act_ext"], long_[:30, col])
    assert (c["spk_act_ext"] == xt.MISSING).all() and c["spk_targets_ext"].shape == (30, 4)
    with pytest.raises(FileNotFoundError):
        xt.attach([_ex(30, start=99.0)], directory=tmp_path, require=True)
    # the stream selector reads <key>.stream.npy
    e = _ex(30, start=40.0)
    xt.attach([e], directory=tmp_path, source="stream")
    assert np.array_equal(e["spk_targets_ext"], long_[:30])
    # batches collate the new keys (every item carries them, missing ones as -1)
    batch = Collate(None)(exs)
    assert batch["spk_act_ext"].shape == (3, 30) and batch["spk_targets_ext"].shape == (3, 30, 4)


def test_recipe_data_ext_tracks_option(tmp_path, monkeypatch):
    from audioforge.datasets import ami
    data = [_ex(20, start=float(i)) for i in range(3)]
    np.save(xt.track_path(tmp_path, xt.example_key(data[0])), np.ones((20, 4), np.float32))
    res = ami.attach_ext_tracks(data, {"source": "offline", "dir": str(tmp_path)}, "train")
    assert res == {"ext_tracks": "offline", "found": 1, "missing": 2}
    assert data[0]["spk_act_ext"].shape == (20,)


# --------------------------------------------------------------------------- primary-column rule == eval_stage1
def test_enroll_column_and_fit_equal_eval_stage1(ev):
    rng = np.random.default_rng(1)
    for _ in range(300):
        T = int(rng.integers(5, 60))
        p = rng.random((T + int(rng.integers(-2, 3)), 4)).astype(np.float32)
        if rng.random() < 0.3:
            p = (p > 0.5).astype(np.float32)  # hard ties -> soft tie-break
        ref = (rng.random(T) < 0.4).astype(np.float32)
        on = int(rng.integers(0, T))
        end = int(rng.integers(on, T + 1))
        assert xt.enroll_column(p, ref, on, end) == ev.enroll_column(p, ref, on, end)
        assert np.array_equal(xt.fit(p[:, 0], T), ev._fit(p[:, 0], T))
    # never looks after turn_end: changing the track after `end` cannot change the choice
    p = rng.random((40, 4)).astype(np.float32)
    ref = np.zeros(40, np.float32); ref[5:20] = 1
    c = xt.enroll_column(p, ref, 5, 20)
    q = p.copy(); q[20:] = 0; q[20:, (c + 1) % 4] = 1
    assert xt.enroll_column(q, ref, 5, 20) == c
    # onset / end = eval's _onset_end on spk_act
    from audioforge.heads.turn import _onset_end
    ex = _ex(30, on=3, end=11)
    assert xt.onset_end(ex["spk_act"]) == tuple(x[0] for x in _onset_end([ex]))


def test_act_stats_equal_eval_stage1(ev):
    rng = np.random.default_rng(2)
    exs = []
    for i in range(6):
        e = _ex(40, start=float(i), on=int(rng.integers(0, 10)), end=int(rng.integers(15, 35)))
        e["spk_act_ext"] = rng.random(40).astype(np.float32)
        exs.append(e)
    ends = [xt.onset_end(e["spk_act"])[1] for e in exs]
    mine = xt.act_stats(exs)
    ref = ev.act_stats([e["spk_act_ext"] for e in exs], exs, ends)
    assert mine.pop("n") == 6 and mine == ref


def test_stream_config_matches_eval_stage1(ev):
    mk = _script("make_sortformer_tracks")
    assert mk.SORTFORMER_LOW_LATENCY == ev.SORTFORMER_LOW_LATENCY and mk.SORTFORMER_ENC_LEFT == ev.SORTFORMER_ENC_LEFT


# --------------------------------------------------------------------------- p_ext conditioning
def _with_ext(b, seed=3, missing=()):
    g = torch.Generator().manual_seed(seed)
    b = dict(b)
    B, T = b["spk_act"].shape
    b["spk_act_ext"] = torch.rand(B, T, generator=g)
    for i in missing:
        b["spk_act_ext"][i] = xt.MISSING
    b["spk_act_ext_len"] = b["spk_act_len"].clone()
    return b


def test_p_ext_one_conditions_on_the_ext_track_and_keeps_oracle():
    model, b = _model({"p_ext": 1.0})
    assert model.cond_on
    model.train()
    b = _with_ext(b, missing=(1,))
    seen, batches = [], []
    _spy_encoder(model, seen)
    _spy_turn_batch(model, batches)
    out = model(b)
    assert torch.isfinite(out["loss"])
    act = seen[0]
    T = act.shape[1]
    tb = batches[0]
    assert torch.equal(tb["spk_act"], act) and torch.equal(tb["spk_act_oracle"], b["spk_act"])  # labels stay clean
    L = int(b["spk_act_len"][0])
    assert torch.allclose(act[0, :min(L, T)], b["spk_act_ext"][0, :min(L, T)])  # soft track as is
    oracle1 = (b["spk_act"][1, :T] > 0.5).float()
    assert torch.equal(act[1], oracle1)  # no cached track -> oracle (no noise configured)
    out["loss"].backward()
    assert model.heads["turn"].out.weight.grad is not None


def test_p_ext_zero_and_unset_are_bit_identical():
    model, b = _model(None)
    model.train()
    att = model.encoder.att_context_size
    torch.manual_seed(5), random.seed(5)
    ref = _forward_v2(model, copy.copy(b), att)
    torch.manual_seed(5), random.seed(5)
    out = model(_with_ext(b), att_context_size=att)  # an ext track in the batch changes nothing when p_ext unset
    assert torch.equal(out["loss"], ref)
    m2, _ = _model({"p_ext": 0.0})
    assert not m2.cond_on
    # with other conditioning on, p_ext = 0 consumes no randomness: same loss with and without the ext track
    m3, b3 = _model({"flip": 0.1, "jitter": 1, "p_ext": 0.0})
    m3.train()
    torch.manual_seed(7), random.seed(7)
    l1 = m3(b3, att_context_size=att)["loss"]
    torch.manual_seed(7), random.seed(7)
    l2 = m3(_with_ext(b3), att_context_size=att)["loss"]
    assert torch.equal(l1, l2)


def test_trainer_conditioning_location():
    import audioforge.model as M
    cfg = _cfg()
    cfg["trainer"] = {"conditioning": {"p_ext": 0.7, "flip": 0.02, "jitter": 1}}
    tok = _model()[0].tokenizer
    m = SpeechModel(cfg, tok)
    assert m.cond_on and m.cond["p_ext"] == 0.7 and m.cond["p_diar"] == 0.0
    assert set(M.COND_DEFAULTS) >= {"p_ext"}


# --------------------------------------------------------------------------- act_override == eval's copied path
def test_act_override_agrees_with_eval_stage1_copy(ev):
    model, _ = _model(None)
    model.eval()
    convs = conversation_dataset(5, seed=9)
    rng = np.random.default_rng(4)
    acts = [rng.random(len(c["spk_act"]) + int(rng.integers(-3, 3))).astype(np.float32) for c in convs]
    ref = ev.turn_scores_given_act(model, "turn", convs, acts, batch_size=2)
    got = turn_scores(model, "turn", convs, batch_size=2, act_override=acts)
    assert len(ref) == len(got) and all(np.array_equal(a, b) for a, b in zip(ref, got))
    # no override -> unchanged path; the oracle track as an override gives the oracle scores
    orc = turn_scores(model, "turn", convs, batch_size=2)
    via = turn_scores(model, "turn", convs, batch_size=2,
                      act_override=[np.asarray(c["spk_act"], np.float32) for c in convs])
    assert all(np.allclose(a, b) for a, b in zip(orc, via))
    assert not all(np.allclose(a, b) for a, b in zip(orc, got))


# --------------------------------------------------------------------------- recipe
def test_turn_on_sortformer_recipe():
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_turn_on_sortformer.yaml").read_text())
    assert cfg["init"]["from"] == "runs/stage1_heads_pretrained.afm" and cfg["init"]["pretrained_lr_mult"] == 0
    scope = cfg["init"]["pretrained_scope"]
    assert not any("heads.turn".startswith(p) for p in scope) and "encoder." in scope
    assert cfg["encoder"]["att_context_size"] == [70, 1] and cfg["encoder"]["att_context_sizes"] == [[70, 1]]
    assert cfg["trainer"]["conditioning"] == {"p_ext": 0.7, "p_diar": 0, "flip": 0.02, "jitter": 1}
    assert cfg["trainer"]["max_steps"] == 1500 and cfg["trainer"]["batch_size"] == 6
    assert cfg["data"]["mix"][0]["ami"]["ext_tracks"] == {"source": "offline"}
    base = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_heads_pretrained.yaml").read_text())
    assert "ext_tracks" not in str(base["data"]) and "conditioning" not in base  # default stage 1 unchanged
