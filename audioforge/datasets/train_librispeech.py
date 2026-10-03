"""Train a ``data: {librispeech: ...}`` recipe on MPS without the per-shape graph-cache leak.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.35 \\
    .venv/bin/python -m audioforge.datasets.train_librispeech recipes/voice_agent_frontend_librispeech.yaml \\
        -o runs/voice_agent_frontend_librispeech.afm [key=value ...]

Same steps as ``audioforge.train.run_recipe`` (and the same unmodified ``Trainer``), but batches go
through ``QuantizedCollate`` (``data.librispeech.pad_to: {audio_sec, tokens}``) so MPS sees a
bounded set of shapes, a checkpoint is written every ``--ckpt-every`` steps, the model is saved
before the final evaluation, and that evaluation runs on CPU. ``audioforge.cli train`` also works, but on MPS every new batch shape costs memory
(free-form shapes: host memory +15 MB/step; ~60 shapes: GPU pool OOM after ~1750 steps).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..train import Trainer, build_model, load_data, load_recipe, save_model
from .librispeech import QuantizedCollate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("-o", "--out")
    ap.add_argument("--ckpt-every", type=int, default=500, help="save <out>.ckpt.afm every N steps (0 = off)")
    a = ap.parse_args(argv)
    cfg = load_recipe(a.recipe, a.overrides)
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    train, val = load_data(cfg, "train"), load_data(cfg, "val")
    model, cfg = build_model(cfg, train)  # fresh, or init.from an .afm (e.g. ported NVIDIA weights)
    tok = model.tokenizer
    print(f"[{cfg.get('name')}] params={model.num_params() / 1e6:.2f}M vocab={tok.vocab_size if tok else '-'} "
          f"heads={list(model.heads)} frame={model.frame_sec * 1000:.0f}ms", flush=True)
    tr = Trainer(model, cfg)
    if a.out and a.ckpt_every:  # Trainer.fit has no checkpoints: piggy-back on its one sched.step() per step
        ckpt, sched_step = str(Path(a.out).with_suffix(".ckpt.afm")), tr.sched.step

        def step_and_checkpoint(*args, **kw):
            sched_step(*args, **kw)
            if tr.sched.last_epoch % a.ckpt_every == 0:
                save_model(model, ckpt)  # save_model moves the state dict to CPU itself
                print(f"checkpoint step={tr.sched.last_epoch} -> {ckpt}", flush=True)

        tr.sched.step = step_and_checkpoint
    pad = cfg["data"].get("librispeech", {}).get("pad_to", {})
    tr.fit(train, val, collate=QuantizedCollate(tok, **pad))
    model.cpu()
    if a.out:  # save first: a late MPS failure must not lose the weights
        save_model(model, a.out)
        print("saved", a.out, flush=True)
    metrics = tr.evaluate(val)  # on CPU: new eval shapes would add MPS graph/buffer memory
    print("final", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
