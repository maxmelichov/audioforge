# Training the heads

How to train or retrain audioforge's heads. The NVIDIA encoder is never trained: every fine-tuning attempt made
LibriSpeech worse ([`ARCHITECTURE.md`](ARCHITECTURE.md)), so everything this project learns lives in small heads that
read the frozen encoder's layers. The binding rules for training code are in [`PROJECT.md`](PROJECT.md); this page
is the how-to.

## Set up

```bash
uv sync --extra train                # accelerate, TensorBoard, tqdm, pyinject; then `uv run python -m audioforge.train ...`
```

Datasets live under `$AUDIOFORGE_DATA` (default `<repo>/data`), checkpoints and logs under `runs/`. Both can be
symlinks to external storage.

## The trainer in one command

```bash
python -m audioforge.train RECIPE.yaml [key=value ...] [--name NAME] [--resume]
```

- **Recipe.** A YAML file: encoder, heads, data and trainer settings. The training recipes are in
  [`recipes/`](../recipes) (for a laptop-sized smoke run, `parakeet_tdt_ctc.yaml` trains a tiny
  model on synthetic data). Any recipe key can be overridden on the command line as `dotted.key=value`, for example
  `trainer.lr=5e-4`.
- **Run directory.** Everything goes to `runs/<name>/` (default name: the recipe's `name`, else its file name):
  `recipe.yaml` (the recipe with your overrides), `tb/` (TensorBoard), `last/` and `best/` (checkpoints), and
  `best/model.afm` (the trained model). `-o out.afm` also writes the final model elsewhere; `--runs DIR` changes the
  parent directory.
- **Evaluation by the clock.** Every `--eval-minutes` (default 10) the trainer evaluates on the validation rows,
  writes `last/`, and updates `best/` only when the eval loss improves.
- **Stop and resume.** Ctrl-C or SIGTERM writes a checkpoint and stops. `--resume` continues `runs/<name>/last` at the
  same step, batch and random state, so no steps are lost and the data order is the same as an uninterrupted run.
- **Progress.** A tqdm bar per epoch with the live loss, and one per evaluation; `--no-progress` turns them off.
- **Watching a live run.** `tensorboard --logdir runs/<name>/tb`. With pyinject installed, `pyinject <pid> 'step,
  float(loss)'` from another shell reads values from the running job.

Trainer flags (each overrides the recipe's `trainer.<key>`): `--max-steps`, `--batch-size`, `--lr`, `--device`
(`auto | cpu | mps | cuda`), `--eval-minutes`, `--eval-items`, `--num-workers`, `--bucket` (duration bucketing),
`--sampler-seed`, `--mixed-precision` (`no | fp16 | bf16`).

**Multi-GPU.** The loop runs on [accelerate](https://huggingface.co/docs/accelerate), so the same command scales out:

```bash
accelerate launch -m audioforge.train RECIPE.yaml --name NAME
```

## Data: one manifest format

The trainer reads one general format: JSONL manifests with NeMo's keys (`audio_filepath`, `duration`, `text`,
`speaker`, ...), one row per example; array labels go in an `.npz` per row (`labels_file`). A recipe points at them:

```yaml
data: {manifest: {train: data/manifests/NAME/train.jsonl, val: data/manifests/NAME/val.jsonl}}
```

A single file (`data: {manifest: all.jsonl}`) is split into train and validation by a seeded split
(`data.val_fraction`, default 0.05). Audio is decoded and preprocessed on the fly, so training starts at once.

Any of the built-in sources (LibriSpeech, AMI, ICSI, two-party conversations, smart-turn clips, synthetic) converts
to manifests with the trainer's own loader:

```bash
python -m audioforge.datasets.to_manifest RECIPE.yaml -o data/manifests/NAME [key=value ...] [--splits train val]
```

It prints the `data:` line to put in the recipe; the rows are identical to what the source produced
(`tests/test_train_loop.py`). A `mix` of sources loses its per-source batching when flattened: convert each source on
its own. Put the corpora under `$AUDIOFORGE_DATA` before converting them.

## Sizes

No head size is chosen by feel ([`PROJECT.md`](PROJECT.md) "Sizing"). Every size in a recipe is also a flag: the
trainer finds the recipe's size keys (`--encoder.d_model`, `--heads.<head>.hidden`, `--heads.<head>.n_layers`, ...)
and lists them with their current values in `python -m audioforge.train RECIPE.yaml --help`.

The default for each size is measured with `scripts/sweep_capacity.py`: every candidate trains for the same
wall-clock budget, and the smallest one within 1 % of the best eval loss is picked.

```bash
uv run scripts/sweep_capacity.py speech --core 115m --sizes 16,32,64,128,256 --budget 120
uv run scripts/sweep_capacity.py speech --core 0p6b --grid hidden=32,64,128 depth=1,2 --budget 60
```

Each sweep prints its table and writes it, with the raw numbers, under `runs/sweeps/`. Sweep again when the data,
the objective or a neighbouring head changes. The sweep currently knows the `speech` head; another head is added by
writing a `Spec` in the script.

## How the shipped heads were made

The recipes for the served turn and speaker heads, and for the experiments around them, are grouped in
[`recipes/README.md`](../recipes/README.md). A served model is the NVIDIA encoder plus a heads file. To write the heads file from a trained served
checkpoint:

```bash
audioforge-download --export-heads SERVED.afm OUT.pt
```

[ARCHITECTURE.md "Models and files"](ARCHITECTURE.md#models-and-files) lists the shipped files.

## Data licences

- Train only on data whose licence allows training a commercial model (for example CC BY 4.0: AMI, LibriSpeech,
  FLEURS, otoSpeech-280h). Non-commercial or research-only data (CC BY-NC, research-only licences, paid LDC corpora
  without a licence) is never used, not even for augmentation noise. The datasets behind the shipped heads and
  their licences are listed in [ARCHITECTURE.md](ARCHITECTURE.md#training-data-not-shipped).
- Never train or select on a test split. Validation rows come from the training corpora's own dev data or held-out
  speakers; the trainer warns when validation rows are also training rows.
- Respect each dataset's terms beyond the licence (otoSpeech: do not try to identify the speakers).
- A head distilled from another model inherits that model's terms (the LID head is trained on AmberNet outputs, NGC
  Terms of Use; see [ARCHITECTURE.md](ARCHITECTURE.md#licences-of-the-weights)).

## Machine safety

Training is a heavy job. Run one training job per machine at a time. Run long jobs in the background with their
output in a log under `runs/`, so progress can be read from the log tail. Keep large datasets and checkpoints on
external storage through `$AUDIOFORGE_DATA` and a `runs/` symlink. Stop a run only after a checkpoint (Ctrl-C writes
one) and bring it back with `--resume`.
