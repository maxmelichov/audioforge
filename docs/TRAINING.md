# Training the heads

How to train or retrain audioforge's heads. The NVIDIA encoder is never trained: every fine-tuning attempt made
LibriSpeech worse ([`ARCHITECTURE.md`](ARCHITECTURE.md)), so everything this project learns lives in small heads that
read the frozen encoder's layers. The binding rules for training code are in [`PROJECT.md`](PROJECT.md); this page
is the how-to.

## Set up

```bash
pip install -e ".[train]"            # accelerate, TensorBoard, tqdm, pyinject
```

Datasets live under `$AUDIOFORGE_DATA` (default `<repo>/data`), checkpoints and logs under `runs/`. Both can be
symlinks to external storage.

## The trainer in one command

```bash
python -m audioforge.train RECIPE.yaml [key=value ...] [--name NAME] [--resume]
```

- **Recipe.** A YAML file: encoder, heads, data and trainer settings. The recipes behind the research are in
  [`research/recipes/`](../research/recipes) (for a laptop-sized smoke run, `parakeet_tdt_ctc.yaml` trains a tiny
  model on synthetic data). Any recipe key can be overridden on the command line as `dotted.key=value`, for example
  `trainer.lr=5e-4`.
- **Run directory.** Everything goes to `runs/<name>/` (default name: the recipe's `name`, else its file name):
  `recipe.yaml` (the recipe with your overrides), `tb/` (TensorBoard), `last/` and `best/` (checkpoints), and
  `best/model.afm` (the trained model). `-o out.afm` also writes the final model elsewhere; `--runs DIR` changes the
  parent directory.
- **Evaluation by the clock.** Every `--eval-minutes` (default 10) the trainer evaluates on the validation rows,
  writes `last/`, and updates `best/` only when the eval loss improves.
- **Stop and resume.** Ctrl-C or SIGTERM writes a checkpoint and stops. `--resume` continues `runs/<name>/last` at the
  same step, batch and random state, so no steps are lost and the data order is the same as an uninterrupted run
  (`plans/trainer/resume_001.py` checks this bit for bit).
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
its own. The download and preparation scripts for each corpus are in
[`scripts/research/README.md`](../scripts/research/README.md) ("Data preparation").

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

Each sweep writes its table to `plans/sweeps/<head>_<date>.md` and its raw numbers to `runs/sweeps/`. The past
sweeps, and which sizes are still unmeasured placeholders, are listed in
[`plans/sweeps/INVENTORY.md`](../plans/sweeps/INVENTORY.md). Sweep again when the data, the objective or a
neighbouring head changes. The sweep currently knows the `speech` head; another head is added by writing a `Spec` in
the script.

## How the shipped heads were made

The shipped heads were trained by the research drivers in [`scripts/research/`](../scripts/research) (for example
`fixall.py` for the speech detector and LID v2, `tsvad.py` for the target-speaker head), each described in its note
in [`research/`](../research/README.md). A served model is the NVIDIA encoder plus a heads file; maintainers write the
heads file from a trained served checkpoint with
`audioforge-download --export-heads SERVED.afm OUT.pt`, and [`MODELS.md`](MODELS.md) lists the shipped files.

## Data licences

- Train only on data whose licence allows training a commercial model (for example CC BY 4.0: AMI, LibriSpeech,
  FLEURS, otoSpeech-280h). Non-commercial or research-only data (CC BY-NC, research-only licences, paid LDC corpora
  without a licence) is never used, not even for augmentation noise. The licence survey for the turn data is in
  [`research/TURN_DATA.md`](../research/TURN_DATA.md) §1.1.
- Never train or select on a test split. Validation rows come from the training corpora's own dev data or held-out
  speakers; the trainer warns when validation rows are also training rows.
- Respect each dataset's terms beyond the licence (otoSpeech: do not try to identify the speakers).
- A head distilled from another model inherits that model's terms (the LID head is trained on AmberNet outputs, NGC
  Terms of Use; see [`MODELS.md`](MODELS.md)).

## Machine safety

Training is a heavy job. On a shared machine (the project's laptop rules, [`CONTRIBUTING.md`](../CONTRIBUTING.md)
"Heavy jobs"):

- one training at a time;
- anything over about 20 s runs in the background with its output in a log under `runs/`, never through `| tail`:
  `scripts/dev/logged.sh -b NAME scripts/dev/gate.sh python -m audioforge.train RECIPE.yaml`, then `chore log NAME`;
- the gate (`scripts/dev/gate.sh`) waits for a free machine (load, swap, at least 10 GB of free disk) and runs the
  job at `nice 5` on 2 threads;
- large datasets and checkpoints go to external storage through `$AUDIOFORGE_DATA` and `runs/` symlinks, never more
  than 1 GB on the internal disk;
- stop a run only after a checkpoint (Ctrl-C does that) and bring it back with `--resume`.
