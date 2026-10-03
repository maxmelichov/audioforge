"""Completeness head + smart-turn dataset loader (research/archive/COMPLETENESS.md)."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.data import Collate, ToneLanguage
from audioforge.datasets import smartturn as st
from audioforge.heads.completeness import CompletenessHead, auc
from audioforge.model import SpeechModel


@pytest.fixture(autouse=True)
def _seeded():
    """Fixed torch / numpy / random seeds per test: random inputs do not depend on which tests ran before."""
    import random

    import numpy as np
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


ROOT = Path(__file__).parent.parent


# --------------------------------------------------------------------------- head
def _head(D=16, H=8, pool=5):
    torch.manual_seed(0)
    return CompletenessHead(D, hidden=H, pool_frames=pool, dropout=0.0).eval()


def test_frame_logits_are_causal():
    h = _head()
    x = torch.randn(2, 12, 16)
    z = h(x)
    for t in (0, 3, 7):
        y = x.clone()
        y[:, t + 1:] += torch.randn_like(y[:, t + 1:])
        assert torch.allclose(h(y)[:, : t + 1], z[:, : t + 1], atol=1e-6)
        assert not torch.allclose(h(y)[:, t + 1:], z[:, t + 1:])


def test_streaming_step_equals_offline():
    h = _head()
    x = torch.randn(2, 11, 16)
    z = h.decode(x, torch.tensor([11, 11]))
    xp, hs = h.hidden_states(x)
    utt_off = torch.stack([h.utterance_logit(xp, hs, torch.tensor([e, e])).sigmoid() for e in range(1, 12)], 1)
    s = h.init_stream(2)
    fr, ut = [], []
    for t in range(11):
        a, b = h.step(x[:, t: t + 1], s)
        fr.append(a)
        ut.append(b)
    assert torch.allclose(torch.cat(fr, 1), z, atol=1e-6)
    assert torch.allclose(torch.stack(ut, 1), utt_off, atol=1e-6)
    # chunked feeding gives the same
    s = h.init_stream(2)
    a1, _ = h.step(x[:, :4], s)
    a2, u2 = h.step(x[:, 4:], s)
    assert torch.allclose(torch.cat([a1, a2], 1), z, atol=1e-6)
    assert torch.allclose(u2, utt_off[:, -1], atol=1e-6)


def test_utterance_logit_pools_last_frames_before_end():
    h = _head(pool=3)
    x = torch.randn(1, 10, 16)
    xp, hs = h.hidden_states(x)
    z = h.utterance_logit(xp, hs, torch.tensor([6]))
    y = x.clone()
    y[:, 6:] += 5.0  # after the end: no effect
    y[:, :2] += 5.0  # before the pooled window (frames 3..5) but inside the GRU history: affects h only
    xp2, hs2 = h.hidden_states(y)
    m = torch.zeros(1, 10, 1)
    m[:, 3:6] = 1
    assert torch.allclose((xp2 * m).sum(1), (xp * m).sum(1))
    yy = x.clone()
    yy[:, 6:] += 5.0
    xp3, hs3 = h.hidden_states(yy)
    assert torch.allclose(h.utterance_logit(xp3, hs3, torch.tensor([6])), z, atol=1e-6)


def test_loss_uses_labels_and_learns():
    torch.manual_seed(0)
    h = CompletenessHead(16, hidden=8, pool_frames=4, dropout=0.0)
    B, T = 8, 12
    x = torch.randn(B, T, 16)
    lab = torch.tensor([i % 2 == 0 for i in range(B)])
    x[lab, 6:] += 2.0  # complete clips look different after the end
    y = torch.zeros(B, T)
    y[lab, 8:] = 1
    w = torch.full((B, T), 0.1)
    w[:, 8:] = 1
    batch = {"completeness": y, "completeness_w": w, "utt_end_frame": torch.full((B, 1), 8), "completeness_len": torch.full((B,), T)}
    el = torch.full((B,), T)
    # without an explicit label the first post-end frame carries it; with one it is used directly
    assert torch.allclose(h.loss(x, el, batch), h.loss(x, el, {**batch, "complete_label": lab.float()[:, None]}))
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    l0 = float(h.loss(x, el, batch).detach())
    for _ in range(60):
        opt.zero_grad()
        l = h.loss(x, el, batch)
        l.backward()
        opt.step()
    assert float(l) < 0.5 * l0
    p = h.utterance_prob(x, el, torch.full((B,), 8))
    assert auc(lab.tolist(), p.tolist()) > 0.9


# --------------------------------------------------------------------------- dataset
def test_frame_labels_and_end_frame():
    y, w = st.frame_labels(10, 6, True, 0.1)
    assert y.tolist() == [0] * 6 + [1] * 4 and w[:6].tolist() == [pytest.approx(0.1)] * 6 and w[6:].tolist() == [1] * 4
    y, w = st.frame_labels(10, 6, False, 0.1)
    assert not y.any() and w[6:].all()
    sr = st.SR
    x = np.concatenate([np.zeros(sr // 2), 0.3 * np.sin(np.arange(sr) * 0.05), np.zeros(int(0.4 * sr))]).astype(np.float32)
    end = st.utterance_end_frame(x)
    T = ToneLanguage.n_frames(len(x))
    assert abs(end - round(1.5 / 0.08)) <= 1 and end < T


def test_match_test_ids_and_seeded_split(tmp_path):
    ids = ["a/x1.flac", "b/x2.flac", "x3.flac", "x4.flac"]
    m = st.match_test_ids(ids, ["x2", "clips/x4.wav"])
    assert m.tolist() == [False, True, False, True]
    u = "18eaa525-67c7-4a4d-aee8-f0f64dc1c25a"
    m = st.match_test_ids([f"complete_{u}_normalized.flac", "incomplete_ffc70d78-18f9-4471-acec-6a338133e4da_normalized.flac"], [u])
    assert m.tolist() == [True, False]
    meta = {"ids": [f"c{i}" for i in range(40)], "complete": [i % 2 == 0 for i in range(40)]}
    sp = st.split_indices(meta, tmp_path, seed=0, eval_frac=0.25)
    assert len(sp["eval"]) == 10 and sum(meta["complete"][i] for i in sp["eval"]) == 5
    assert not (set(sp["eval"]) & set(sp["train"])) and sp["how"].startswith("seeded")
    assert sp == st.split_indices(meta, tmp_path, seed=0, eval_frac=0.25)
    # id file with enough matches -> aligned split
    (tmp_path / st.IDS_JSON).write_text(json.dumps({"test": [{"id": f"c{i}"} for i in range(0, 40, 4)] * 20}))
    meta2 = {"ids": [f"c{i}" for i in range(400)], "complete": [True] * 400}
    (tmp_path / st.IDS_JSON).write_text(json.dumps({"test": [{"id": f"c{i}"} for i in range(0, 400, 4)]}))
    sp = st.split_indices(meta2, tmp_path)
    assert len(sp["eval"]) == 100 and sp["how"].startswith("smart-turn")


def test_parquet_round_trip_16k_mono(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    import soundfile as sf
    rows, labels = [], []
    for i, sr in enumerate((16000, 24000, 16000)):
        x = 0.2 * np.sin(np.arange(int(0.7 * sr)) * 0.03).astype(np.float32)
        if i == 1:
            x = np.stack([x, x], 1)  # stereo, 24 kHz
        b = io.BytesIO()
        sf.write(b, x, sr, format="FLAC")
        rows.append({"bytes": b.getvalue(), "path": f"clip{i}.flac"})
        labels.append(i % 2 == 0)
    t = pa.table({"audio": rows, "endpoint_bool": labels})
    root = tmp_path / "root"
    (root / "human_5_all" / "data").mkdir(parents=True)
    pq.write_table(t, root / st.PARQUET)
    ds = st.SmartTurnClips(root, "train", verbose=False)
    ds_ev = st.SmartTurnClips(root, "eval", verbose=False)
    assert len(ds) + len(ds_ev) == 3
    meta = json.loads((root / "cache" / "human_5_all.json").read_text())
    assert meta["licence"] == "bsd-2-clause" and meta["ids"] == ["clip0.flac", "clip1.flac", "clip2.flac"]
    for j in range(3):
        x = ds.audio(j)
        assert x.ndim == 1 and abs(len(x) - int(0.7 * 16000)) <= 2 and x.dtype == np.float32
    ex = ds[0]
    T = ToneLanguage.n_frames(len(ex["audio"]))
    assert ex["completeness"].shape == (T,) and ex["utt_end_frame"].shape == (1,) and 0 < int(ex["utt_end_frame"][0]) <= T
    b = Collate(None)([ds[i] for i in range(len(ds))])
    assert set(b) >= {"audio", "audio_len", "completeness", "completeness_len", "completeness_w", "utt_end_frame", "complete_label"}
    assert b["complete_label"].reshape(-1).tolist() == [float(ds[i]["complete"]) for i in range(len(ds))]


# --------------------------------------------------------------------------- model + recipe
def _tiny_cfg():
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_completeness.yaml").read_text())
    cfg.pop("init")
    cfg["encoder"] = dict(n_layers=2, d_model=32, n_heads=2, subsampling_channels=8, att_context_size=[70, 1],
                          att_context_sizes=[[70, 1]])
    cfg["heads"] = {"vad": {"type": "frame", "key": "vad", "weight": 0}, "completeness": dict(cfg["heads"]["completeness"], hidden=16)}
    return cfg


def test_recipe_and_model_train_on_synthetic_clips():
    cfg = yaml.safe_load((ROOT / "research" / "recipes" / "stage1_completeness.yaml").read_text())
    assert cfg["init"]["pretrained_lr_mult"] == 0 and "smartturn" in cfg["data"]
    assert all(v.get("weight", 1) == 0 for k, v in cfg["heads"].items() if k != "completeness")
    assert cfg["heads"]["completeness"]["type"] == "completeness" and cfg["trainer"]["checkpoint_every"] == 250
    torch.manual_seed(0)
    m = SpeechModel(_tiny_cfg())
    data = st.synthetic_clips(6, seed=0)
    assert sum(ex["complete"] for ex in data) == 3
    b = Collate(None)(data)
    out = m(b)
    assert "loss_completeness" in out and torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    m.eval()
    res = m.heads["completeness"].evaluate_model(m, "completeness", data)
    assert set(res) >= {"completeness_utt_acc", "completeness_utt_auc", "completeness_frame_auc"} and res["completeness_n"] == 6
    # recipe_data stand-in path
    cfg2 = dict(_tiny_cfg(), data={"smartturn": {}, "synthetic": {"n_train": 4, "n_val": 2}})
    assert len(st.recipe_data(cfg2, "train")) == 4 and len(st.recipe_data(cfg2, "val")) == 2
