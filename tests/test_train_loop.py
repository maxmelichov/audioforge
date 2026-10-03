"""The training loop's run machinery (accelerate, run dir, time-based eval, last / best, resume) and the data side
it reads (the JSONL manifest format, the adapter that writes it, seeded split, the epoch sampler)."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.data import (
    ManifestData,
    check_disjoint,
    epoch_batches,
    read_manifest,
    seeded_split,
    shard_batches,
    synthetic_dataset,
    write_manifest,
)
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import Trainer, load_data, main, size_flags

pytest.importorskip("accelerate")
ENC = {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
       "att_context_size": [8, 1], "dropout": 0.0}
PRE = {"n_mels": 40, "normalize": "fixed", "dither": 0.0}
TOK = list("abcdefghijklmnopqrstuvwxyz '")


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _same(a: dict, b: dict) -> bool:
    for k, v in a.items():
        w = b[k]
        if isinstance(v, np.ndarray):
            if not (isinstance(w, np.ndarray) and v.dtype == w.dtype and np.array_equal(v, w)):
                return False
        elif v != w and not (isinstance(v, tuple) and list(v) == w):
            return False
    return True


# --------------------------------------------------------------------------- data: manifest format + adapter
def test_write_manifest_round_trip_is_lossless(tmp_path):
    rows = synthetic_dataset("multitask", 5, seed=0) + synthetic_dataset("diar", 3, seed=1)
    path = write_manifest(rows, tmp_path, "train")
    data = ManifestData(path)
    assert len(data) == 8 and [round(d, 3) for d in data.durations] == [round(len(r["audio"]) / 16000, 3) for r in rows]
    for src, got in zip(rows, data):
        assert _same(src, got), sorted(src)
    assert all(_same(x, y) for x, y in zip(read_manifest(path), data))  # eager reader = lazy reader


def test_manifest_recipe_single_file_split_is_seeded_and_disjoint(tmp_path):
    path = write_manifest(synthetic_dataset("asr", 40, seed=0), tmp_path, "all")
    cfg = {"data": {"manifest": str(path), "val_fraction": 0.25}, "seed": 3}
    tr, va = load_data(cfg, "train"), load_data(cfg, "val")
    assert len(tr) + len(va) == 40 and 0 < len(va) < 40
    assert not {m["audio_filepath"] for m in tr.rows} & {m["audio_filepath"] for m in va.rows}
    assert [m["audio_filepath"] for m in load_data(cfg, "val").rows] == [m["audio_filepath"] for m in va.rows]
    rows = ManifestData(path).rows
    assert seeded_split(rows, 0.25, 3) == seeded_split(rows, 0.25, 3) != seeded_split(rows, 0.25, 4)
    assert check_disjoint(tr.rows, va.rows) == 0 and check_disjoint(rows, va.rows) == len(va)


def _old_batches(n, bs, rng, src=None):  # the pre-accelerate Trainer._batches, kept verbatim as the reference
    if src is None:
        idx = list(range(n))
        rng.shuffle(idx)
        return [idx[i: i + bs] for i in range(0, n - bs + 1, bs)]
    groups = {}
    for i, s in enumerate(src):
        groups.setdefault(s, []).append(i)
    out = []
    for _, g in sorted(groups.items()):
        rng.shuffle(g)
        out += [g[i: i + bs] for i in range(0, len(g) - bs + 1, bs)]
    rng.shuffle(out)
    return out


def test_epoch_batches_matches_old_order_and_buckets():
    src = [i % 3 for i in range(50)]
    for s in (None, src):
        r1, r2 = random.Random(0), random.Random(0)
        assert [epoch_batches(50, 4, r1, s) for _ in range(3)] == [_old_batches(50, 4, r2, s) for _ in range(3)]
    dur = [random.Random(i).uniform(1, 20) for i in range(64)]
    plain, bucketed = epoch_batches(64, 4, random.Random(0)), epoch_batches(64, 4, random.Random(0), durations=dur, bucket=4)
    pad = lambda bs: sum(max(dur[i] for i in b) * len(b) - sum(dur[i] for i in b) for b in bs)  # noqa: E731
    assert sorted(sum(bucketed, [])) == sorted(sum(plain, [])) and pad(bucketed) < pad(plain)
    assert shard_batches(list(range(7)), 1, 2) == [1, 3, 5] and shard_batches([1, 2], 0, 1) == [1, 2]


# --------------------------------------------------------------------------- loop: run dir, eval, last / best, resume
def _trainer(tmp_path, **t):
    torch.manual_seed(0)
    random.seed(0)
    cfg = {"preprocessor": PRE, "encoder": ENC, "heads": {"ctc": {"type": "ctc"}},
           "trainer": {"max_steps": 6, "batch_size": 4, "lr": 1e-3, "device": "cpu", "log_every": 1,
                       "progress": False, **t}}
    return Trainer(SpeechModel(cfg, CharTokenizer(TOK)), cfg)


def test_run_dir_eval_checkpoints_and_resume_continue_exactly(tmp_path):
    train, val = synthetic_dataset("asr", 12, seed=0), synthetic_dataset("asr", 4, seed=1)
    ref = _trainer(tmp_path, eval_minutes=1e-9)  # every step is past the eval time
    ref.fit(train, val, run_dir=tmp_path / "ref")
    run = tmp_path / "ref"
    assert len(ref.evals) == 7 and all("loss" in e for e in ref.evals)  # 6 timed evals + the final one
    st = json.loads((run / "last" / "trainer_state.json").read_text())
    assert st["step"] == 6 and st["best"] == min(e["loss"] for e in ref.evals)
    assert (run / "best" / "model.afm").exists() and list((run / "tb").rglob("events.out.tfevents.*"))
    with pytest.raises(FileExistsError):  # a finished run is never silently overwritten
        _trainer(tmp_path).fit(train, val, run_dir=run)

    cut = _trainer(tmp_path)  # stop after step 3 (what SIGINT / SIGTERM do), then resume in a fresh trainer
    cut._stop_requested = lambda acc: cut.history[-1]["step"] == 3
    cut.fit(train, val, run_dir=tmp_path / "cut")
    assert cut.stopped["step"] == 3 and (tmp_path / "cut" / "last").is_dir()
    res = _trainer(tmp_path)
    res.fit(train, val, run_dir=tmp_path / "cut", resume=True)
    assert [h["step"] for h in res.history] == [4, 5, 6]
    assert [h["loss"] for h in res.history] == [h["loss"] for h in ref.history[3:]]  # timed evals changed nothing


def test_plain_fit_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = _trainer(tmp_path, eval_minutes=1e-9)
    tr.fit(synthetic_dataset("asr", 8, seed=0))
    assert tr.history[-1]["step"] == 6 and not tr.evals and not list(tmp_path.iterdir())


def test_main_flags_sizes_and_run_dir(tmp_path):
    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump({"name": "t", "preprocessor": PRE, "encoder": ENC, "tokenizer": {"kind": "char"},
                                      "heads": {"ctc": {"type": "ctc"}, "diar": {"type": "sortformer", "num_spks": 4,
                                                                                 "d_hidden": 16, "n_layers": 1}},
                                      "data": {"synthetic": {"kind": "multitask", "n_train": 8, "n_val": 4}},
                                      "trainer": {"max_steps": 50, "batch_size": 4, "device": "cpu"}}))
    flags = size_flags(yaml.safe_load(recipe.read_text()))
    assert {"encoder.d_model", "encoder.n_layers", "heads.diar.d_hidden", "heads.diar.n_layers"} <= set(flags)
    assert main([str(recipe), "--runs", str(tmp_path / "runs"), "--max-steps", "2", "--encoder.d_model", "16",
                 "--heads.diar.d_hidden", "8", "--no-progress"]) == 0
    run = tmp_path / "runs" / "t"
    cfg = yaml.safe_load((run / "recipe.yaml").read_text())
    assert cfg["encoder"]["d_model"] == 16 and cfg["heads"]["diar"]["d_hidden"] == 8
    assert cfg["trainer"]["max_steps"] == 2 and cfg["trainer"]["eval_minutes"] == 10
    assert json.loads((run / "last" / "trainer_state.json").read_text())["step"] == 2
    assert Path(run / "best" / "model.afm").exists()
