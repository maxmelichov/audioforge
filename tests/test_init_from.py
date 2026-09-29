"""init.from: start a recipe from a trained/ported .afm, add heads, keep pretrained weights."""
import numpy as np
import torch

from audioforge.data import Collate, synthetic_dataset
from audioforge.model import SpeechModel
from audioforge.tokenizer import CharTokenizer
from audioforge.train import Trainer, build_model, save_model

SRC_CFG = {
    "preprocessor": {"n_mels": 40, "normalize": "fixed", "dither": 0.0},
    "encoder": {"d_model": 32, "n_layers": 2, "n_heads": 2, "subsampling_channels": 8, "causal": True,
                "att_context_size": [8, 1], "dropout": 0.0},
    "heads": {"rnnt": {"type": "rnnt", "pred_hidden": 32, "joint_hidden": 32}, "ctc": {"type": "ctc"}},
}


def _source(tmp_path):
    torch.manual_seed(0)
    src = SpeechModel(SRC_CFG, CharTokenizer(list("abcd ")))
    path = tmp_path / "src.afm"
    save_model(src, path)
    return src, path


def test_init_from_keeps_weights_adds_heads_and_lr_groups(tmp_path):
    src, path = _source(tmp_path)
    recipe = {
        "name": "ft", "init": {"from": str(path), "pretrained_lr_mult": 0.1},
        "encoder": {"att_context_sizes": [[8, 1], [8, 0]], "speaker_kernel_layers": [0]},
        "heads": {"rnnt": {"weight": 0.5}, "vad": {"type": "frame", "key": "vad"},
                  "spk": {"type": "speaker", "num_speakers": 8, "emb_dim": 16, "from_layers": "all"}},
        "trainer": {"max_steps": 2, "batch_size": 4, "lr": 1e-3, "device": "cpu", "log_every": 1},
        "data": {"synthetic": {"kind": "multitask", "n_train": 8, "n_val": 4}},
    }
    model, cfg = build_model(recipe, [])
    # pretrained tensors are identical; recipe adjusted only the rnnt weight, not its architecture
    ssd, msd = src.state_dict(), model.state_dict()
    assert all(torch.equal(ssd[k], msd[k]) for k in ssd)
    assert cfg["heads"]["rnnt"]["pred_hidden"] == 32 and cfg["heads"]["rnnt"]["weight"] == 0.5
    assert set(model.heads) == {"rnnt", "ctc", "vad", "spk"}
    assert model.encoder.att_context_sizes == [[8, 1], [8, 0]] and "0" in model.encoder.speaker_kernels
    assert model.tokenizer.vocab_size == src.tokenizer.vocab_size
    # untrained speaker kernels are identity, so the pretrained RNNT transcribes exactly as before
    audio = [np.random.default_rng(0).standard_normal(8000).astype(np.float32) * 0.1]
    assert model.transcribe(audio, head="rnnt") == src.transcribe(audio, head="rnnt")
    # two optimizer groups: pretrained at lr*0.1, new at lr
    tr = Trainer(model, cfg)
    lrs = sorted(g["lr"] for g in tr.opt.param_groups)
    assert len(lrs) == 2 and abs(lrs[0] - 1e-4) < 1e-12 and abs(lrs[1] - 1e-3) < 1e-12
    n_pre = sum(p.numel() for g in tr.opt.param_groups if g["lr"] < 5e-4 for p in g["params"])
    assert n_pre == sum(p.numel() for n, p in model.named_parameters() if n in model.pretrained_keys)
    tr.fit(synthetic_dataset("multitask", 8, seed=0), collate=Collate(model.tokenizer))
    assert tr.history[-1]["step"] == 2 and np.isfinite(tr.history[-1]["loss"])


def test_init_from_freeze(tmp_path):
    _, path = _source(tmp_path)
    recipe = {"init": {"from": str(path), "pretrained_lr_mult": 0}, "heads": {"vad": {"type": "frame", "key": "vad"}},
              "trainer": {"max_steps": 1, "batch_size": 2, "device": "cpu"}}
    model, cfg = build_model(recipe, [])
    tr = Trainer(model, cfg)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(n in model.pretrained_keys for n in frozen)
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("heads.vad"))
    assert len(tr.opt.param_groups) == 1
