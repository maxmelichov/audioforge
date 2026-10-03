# /// script
# requires-python = ">=3.10"
# ///
"""Trainer parity: first 20 steps' losses for fixed seeds/configs, dumped as JSON.

Run from the repo root with the project venv (it imports audioforge):
    .venv/bin/python plans/trainer/parity_001.py <out.json>
    .venv/bin/python plans/trainer/parity_001.py --compare before.json after.json
"""
import json
import random
import sys

RECIPES = ["research/recipes/parakeet_tdt_ctc.yaml", "research/recipes/canary_aed.yaml",
           "research/recipes/sortformer_diar.yaml", "research/recipes/speaker_aware_turn.yaml"]
SMALL = ["trainer.max_steps=20", "trainer.batch_size=4", "trainer.device=cpu", "trainer.log_every=100",
         "data.synthetic.n_train=24", "data.synthetic.n_val=4",
         "encoder.n_layers=2", "encoder.d_model=64", "encoder.subsampling_channels=16"]


def run() -> dict:
    import numpy as np
    import torch

    from audioforge.train import Trainer, build_model, load_data, load_recipe
    torch.set_num_threads(1)
    res = {}
    for r in RECIPES:
        cfg = load_recipe(r, SMALL)
        random.seed(cfg.get("seed", 0))  # global random drives att-context / prefix / decoded_prob draws
        torch.manual_seed(cfg.get("seed", 0))
        np.random.seed(cfg.get("seed", 0))
        train, val = load_data(cfg, "train"), load_data(cfg, "val")
        model, cfg = build_model(cfg, train)
        tr = Trainer(model, cfg)
        tr.fit(train, val)
        res[r] = [h["loss"] for h in tr.history if "loss" in h]
    return res


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        a, b = (json.load(open(p)) for p in sys.argv[2:4])
        worst = max(abs(x - y) for r in a for x, y in zip(a[r], b[r], strict=True))
        print(f"max |delta loss| over {sum(map(len, a.values()))} steps: {worst:.3g}")
        sys.exit(0 if worst <= 1e-6 else 1)
    json.dump(run(), open(sys.argv[1], "w"), indent=1)
