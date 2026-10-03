"""train.py fixes (verification items 1-4), mixed data sources (data.mix) and the
stage-1/stage-2 pretrained recipes' config plumbing (tiny stand-in .afm, never the real one)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audioforge.data import Collate, save_wav, synthetic_dataset
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import MixCollate, Trainer, load_data, load_model, run_recipe, save_model

ROOT = Path(__file__).parent.parent
ENC = {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
       "att_context_size": [8, 1], "dropout": 0.0}
PRE = {"n_mels": 40, "normalize": "fixed", "dither": 0.0}
TRAIN = {"max_steps": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu", "log_every": 1}
TOK = list("abcdefghijklmnopqrstuvwxyz '")


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _model(heads, enc=None):
    torch.manual_seed(0)
    cfg = {"preprocessor": PRE, "encoder": {**ENC, **(enc or {})}, "heads": heads, "trainer": TRAIN}
    return SpeechModel(cfg, CharTokenizer(TOK)), cfg


# --------------------------------------------------------------------------- §6.1 frame heads in evaluate
def test_evaluate_multiclass_frame_head_and_absent_key():
    m, cfg = _model({"cls": {"type": "frame", "key": "cls", "num_classes": 3},
                     "vad": {"type": "frame", "key": "vad"},  # key absent from val: skipped
                     "eou": {"type": "frame", "key": "eou"}})
    val = synthetic_dataset("multitask", 6, seed=1)
    for ex in val:
        ex["cls"] = (ex["vad"] * (1 + (np.arange(len(ex["vad"])) % 2))).astype(np.int64)  # classes 0/1/2
        del ex["vad"]
    res = Trainer(m, cfg).evaluate(val)
    assert "acc_cls" in res and 0.0 <= res["acc_cls"] <= 1.0
    assert "recall_cls_1" in res and "recall_cls_2" in res and "recall_cls_0" not in res
    assert "acc_vad" not in res and "acc_eou" in res and "recall_eou" in res


# --------------------------------------------------------------------------- §6.2 save before evaluate
def _tiny_recipe(tmp_path, heads, data):
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"name": "t", "preprocessor": PRE, "encoder": ENC, "tokenizer": {"kind": "char"},
                                 "heads": heads, "data": data, "trainer": TRAIN}))
    return str(p)


def test_run_recipe_saves_before_evaluate_and_reports_eval_error(tmp_path, monkeypatch):
    def boom(self, val, n=64):
        raise RuntimeError("eval exploded")
    monkeypatch.setattr(Trainer, "evaluate", boom)
    out = tmp_path / "m.afm"
    model, metrics = run_recipe(_tiny_recipe(tmp_path, {"ctc": {"type": "ctc"}},
                                             {"synthetic": {"kind": "asr", "n_train": 8, "n_val": 4}}),
                                out=str(out))
    assert out.exists() and "eval_error" in metrics and "eval exploded" in metrics["eval_error"]
    loaded = load_model(out)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())


# --------------------------------------------------------------------------- §6.3 too little data
@pytest.mark.parametrize("n", [0, 3])
def test_fit_raises_when_train_smaller_than_batch(n):
    m, cfg = _model({"ctc": {"type": "ctc"}})
    with pytest.raises(ValueError, match="batch_size"):
        Trainer(m, cfg).fit(synthetic_dataset("asr", n, seed=0))


# --------------------------------------------------------------------------- §6.4 oracle-WER without other heads
def test_condition_on_speaker_oracle_wer_without_non_text_heads():
    m, cfg = _model({"ctc": {"type": "ctc", "condition_on_speaker": True}}, enc={"speaker_kernel_layers": [0]})
    res = Trainer(m, cfg).evaluate(synthetic_dataset("speaker_attributed", 4, seed=1))
    assert "wer_ctc_oracle_act" in res


# --------------------------------------------------------------------------- data.mix
MIX = {"mix": [{"synthetic": {"kind": "multitask", "n_train": 8, "n_val": 4}, "weight": 2.0, "name": "asr"},
               {"synthetic": {"kind": "diar", "n_train": 8, "n_val": 4}, "weight": 0.5, "name": "diar"}],
       "val": {"synthetic": {"kind": "diar", "n_val": 4}}}


def test_mix_weights_sources_and_val():
    train = load_data({"data": MIX}, "train")
    assert len(train) == 8 * 2 + 4 and train.names == ["asr", "diar"]
    assert train.counts == {"asr": 16, "diar": 4}
    assert [train.source_ids.count(i) for i in range(2)] == [16, 4]
    assert all(("text" in train[i]) == (train.source_ids[i] == 0) for i in range(len(train)))
    val = load_data({"data": MIX}, "val")
    assert len(val) == 4 and all("spk_targets" in ex and "text" not in ex for ex in val)
    # derived labels: vad from spk_targets (any speaker active) for items that lack it
    d = {**MIX, "derive": ["vad"]}
    ex = load_data({"data": d}, "val")[0]
    assert np.array_equal(ex["vad"], (ex["spk_targets"].max(1) > 0.5).astype(np.float32))
    # drop_keys per source; standin replaces every source's size
    d = {"mix": [{**MIX["mix"][0], "drop_keys": ["speaker"]}], "standin": {"n_train": 3, "n_val": 2}}
    tr = load_data({"data": d}, "train")
    assert len(tr) == 6 and "speaker" not in tr[0] and "text" in tr[0]


def test_mix_source_batches_are_homogeneous_and_train():
    m, cfg = _model({"ctc": {"type": "ctc"}, "vad": {"type": "frame", "key": "vad"},
                     "diar": {"type": "sortformer", "num_spks": 4, "d_hidden": 16, "n_layers": 1}})
    train = load_data({"data": MIX}, "train")
    tr = Trainer(m, {**cfg, "trainer": {**TRAIN, "max_steps": 5}})
    batches = tr._batches(train, __import__("random").Random(0))
    assert {len({train.source_ids[i] for i in b}) for b in batches} == {1} and all(len(b) == 4 for b in batches)
    tr.fit(train, collate=Collate(m.tokenizer))
    h = tr.history
    assert h[-1]["step"] == 5 and all(np.isfinite(r["loss"]) for r in h)
    assert any("loss_diar" in r and "loss_ctc" not in r for r in h) and any("loss_ctc" in r for r in h)


def test_mix_collate_presence_mask():
    a = synthetic_dataset("multitask", 2, seed=0)
    b = synthetic_dataset("diar", 2, seed=0)
    with pytest.raises(ValueError):
        Collate(CharTokenizer(TOK))(a + b)  # plain Collate refuses partial keys
    out = MixCollate(Collate(CharTokenizer(TOK)))(a + b)
    assert out["vad_present"].tolist() == [True, True, False, False]
    assert out["spk_targets_present"].tolist() == [False, False, True, True]
    assert out["text_len"][2:].tolist() == [0, 0] and out["vad_len"][2:].tolist() == [0, 0]
    assert out["speaker"][2:].tolist() == [0, 0] and out["speaker_present"].tolist() == [True, True, False, False]
    assert "text_present" in out and out["spk_targets_len"][:2].tolist() == [0, 0]


# --------------------------------------------------------------------------- stage-1 / stage-2 recipes
def _stand_in_afm(tmp_path):
    torch.manual_seed(0)
    src = SpeechModel({"preprocessor": PRE, "encoder": {**ENC, "att_context_sizes": [[8, 1], [8, 0]]},
                       "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32},
                                 "ctc": {"type": "ctc"}}}, CharTokenizer(TOK))
    save_model(src, tmp_path / "src.afm")
    return str(tmp_path / "src.afm")


def _gate(tmp_path, n=3):
    lines = []
    for i, ex in enumerate(synthetic_dataset("asr", n, seed=5)):
        save_wav(str(tmp_path / f"g{i}.wav"), np.asarray(ex["audio"]))
        lines.append(json.dumps({"audio_filepath": str(tmp_path / f"g{i}.wav"), "text": ex["text"]}))
    (tmp_path / "gate.jsonl").write_text("\n".join(lines))
    return str(tmp_path / "gate.jsonl")


SHRINK = ["trainer.device=cpu", "trainer.batch_size=4", "trainer.max_steps=2", "trainer.log_every=1",
          "trainer.checkpoint_every=1", "data.standin={n_train: 8, n_val: 4}", "encoder.speaker_kernel_layers=[0, 1]",
          "heads.spk.emb_dim=16", "heads.diar.d_hidden=16", "heads.diar.n_layers=1", "heads.turn.hidden=16",
          "heads.turn.text_dim=16"]


@pytest.mark.parametrize("stage", [1, 2])
def test_pretrained_stage_recipes_plumbing(stage, tmp_path):
    recipe = ROOT / "recipes" / ("stage1_heads_pretrained.yaml" if stage == 1 else "stage2_unfreeze_pretrained.yaml")
    cfg = yaml.safe_load(recipe.read_text())
    real = cfg["init"]["from"]
    assert real == ("runs/nemo_hybrid_streaming_multi.afm" if stage == 1 else "runs/stage1_heads_pretrained.afm")
    assert cfg["trainer"]["wer_gate"]["max_delta"] == 0.002
    ov = SHRINK + [f"init.from={_stand_in_afm(tmp_path)}", f"trainer.wer_gate.manifest={_gate(tmp_path)}",
                   "trainer.wer_gate.n=3", "trainer.wer_gate.every=1"]
    if stage == 2:
        ov += ["trainer.mode_consistency.contexts=[[8, 1], [8, 0]]"]
    out = tmp_path / f"stage{stage}.afm"
    model, metrics = run_recipe(str(recipe), ov, out=str(out))
    assert out.exists() and metrics and "eval_error" not in metrics, metrics
    assert set(model.heads) == {"rnnt", "ctc", "vad", "eou", "spk", "diar", "turn"}
    assert model.heads["spk"].W.shape[0] == 190 and model.head_cfg["rnnt"]["fused_batch_size"] == 2
    assert (tmp_path / f"stage{stage}.step1.afm").exists()
    if stage == 1:  # frozen encoder: pretrained tensors unchanged
        src = load_model(tmp_path / "src.afm").state_dict()
        assert all(torch.equal(src[k], model.state_dict()[k].cpu()) for k in src)
