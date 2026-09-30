"""Every recipe builds, trains a few steps, evaluates, saves and reloads."""
from pathlib import Path

import pytest
import torch
import yaml

from audioforge.train import load_model, run_recipe


def _synthetic(p: Path) -> bool:  # real-data recipes (LibriSpeech etc.) need downloads; not smoke-tested
    return "synthetic" in (yaml.safe_load(p.read_text()).get("data") or {})


RECIPES = sorted(p for p in Path(__file__).parent.parent.joinpath("research", "recipes").glob("*.yaml") if _synthetic(p))
SMALL = ["trainer.max_steps=3", "trainer.batch_size=4", "trainer.device=cpu",
         "data.synthetic.n_train=16", "data.synthetic.n_val=4",
         "encoder.n_layers=2", "encoder.d_model=64", "encoder.subsampling_channels=16"]


@pytest.mark.parametrize("recipe", RECIPES, ids=[r.stem for r in RECIPES])
def test_recipe_smoke(recipe, tmp_path):
    root = Path(__file__).parent.parent
    cfg = yaml.safe_load(recipe.read_text())
    codec = (cfg.get("data") or {}).get("codec_path")
    if codec and not (root / codec).exists():  # a trained artifact (runs/codec.pt) that a fresh clone does not have
        pytest.skip(f"{recipe.name} needs {codec}: python -m audioforge.cli train-codec --mel -o {codec}")
    model, metrics = run_recipe(str(recipe), SMALL, out=str(tmp_path / "m.afm"))
    assert metrics, "evaluation produced no metrics"
    loaded = load_model(tmp_path / "m.afm")
    sd1, sd2 = model.state_dict(), loaded.state_dict()
    assert all(torch.equal(sd1[k].cpu(), sd2[k]) for k in sd1)
