# Trainer parity: pre-change vs accelerate trainer (2026-10-03)

Base commit 4895faf (pre-change `audioforge/train.py`) vs the accelerate loop. Same seed and config, CPU, 1 thread,
first 20 optimizer steps of four synthetic recipes (tiny encoder: `n_layers=2 d_model=64 subsampling_channels=16`,
`batch_size=4`, `n_train=24`). Reproduce:

    .venv/bin/python plans/trainer/parity_001.py plans/trainer/parity_after.json
    .venv/bin/python plans/trainer/parity_001.py --compare plans/trainer/parity_before.json plans/trainer/parity_after.json

Result: **max |delta loss| = 0 over 80 steps** (bit-identical; the 1e-6 bar is met with margin).
`parity_before.json` was recorded on the unmodified trainer before any edit.

What keeps it identical: single-process accelerate leaves the model unwrapped and in place (`device_placement=False`),
`accelerator.backward` / `clip_grad_norm_` / the wrapped optimizer and scheduler do what the plain calls did,
the batch order is the old `random.Random(0)` epoch shuffle (now `data.epoch_batches`, bucketing off by default),
and the DataLoader that feeds batches gets its own `torch.Generator` so iterating it never draws from the global
torch RNG (without that, every epoch shifted SpecAugment / dropout draws).

## Pre-existing nondeterminism found (not introduced, written down instead of hidden)

`speaker_aware_turn.yaml` gave different losses on two runs of the *unchanged* trainer (|delta| 0.029 at step 1):
`run_recipe` seeded torch and numpy but not Python's `random`, which drives `SpeechModel.forward`'s att-context
draw, the sortformer prefix draw (`heads/audio.py`) and the turn head's `decoded_prob` draw (`heads/turn.py`).
The parity script seeds `random` for both sides.

**Not fixed, stop-and-write-down:** seeding `random` in `run_recipe` (the rule "loader, sampler, split and masking
are seeded") was tried and reverted: `tests/test_diag_turn.py::test_recipe_smoke_diagnostics_reported` then fails
every time (with seed 0 all three steps' `decoded_prob=0.5` draws pick "decoded", so `turn_align_greedy_frac` is
never reported). Unseeded, that test is flaky on the base commit itself (measured on 4895faf train.py/data.py: 4 of 8 runs fail; with the new loop and no seed: 0 of 5 and 1 of 5 in two batches; 3/3 fail with
the seed). The fix needs files outside the training path: make the test seed-independent
(e.g. `decoded_prob=0.0` for the greedy-alignment assertion, or more steps) and then seed `random` in
`run_recipe`. Within a run the global `random` state *is* checkpointed and restored on `--resume`
(accelerate save_state / load_state), so resumes are exact either way.

## Behaviour that is new only when asked for

- Run dir (`fit(run_dir=...)`, `trainer.run_dir`, or `python -m audioforge.train`): TensorBoard, timed evals,
  `last`/`best`, signal handler, `--resume`. Existing callers (`audioforge.cli train`, scripts, tests) pass no run
  dir and get the old loop: no files, no handlers, no extra evals.
- `trainer.bucket`, `trainer.num_workers`, `trainer.mixed_precision`, `trainer.sampler_seed` (default 0 = old order).
- tqdm bars per epoch / eval (stderr); `trainer.progress: false` or `--no-progress` turns them off.

## Behaviour changes that could not be avoided (stop-and-write-down list)

1. `Trainer.fit` now needs `accelerate` (and tqdm). Base `pip install audioforge` users (serve only) are unaffected
   (accelerate/tqdm are imported inside `fit`); training needs `pip install "audioforge[train]"`. `dev` lists
   accelerate / tensorboard / tqdm so CI keeps running the trainer tests.
2. `data.manifest` recipes now read rows lazily (`ManifestData`, same items as `read_manifest`, audio decoded per
   access instead of all up front). `read_manifest` still returns the eager list.
3. A val manifest that shares rows with the train manifest is now warned about (`check_disjoint`), not refused.
4. Named dataset sources (librispeech / ami / icsi / dyadic / smartturn) still load directly in the trainer. Routing
   them through the manifest by default would change start-up cost and, for lazily generated mixes, the items'
   materialisation; instead `python -m audioforge.datasets.to_manifest` converts any recipe's data to the manifest
   format losslessly (float32 WAV + npz labels, round trip tested), and a recipe then trains from `data.manifest`.

## Step losses

| recipe | step | before | after | abs delta |
|---|---|---|---|---|
| parakeet_tdt_ctc.yaml | 1 | 4.391630173 | 4.391630173 | 0 |
| parakeet_tdt_ctc.yaml | 2 | 4.495429516 | 4.495429516 | 0 |
| parakeet_tdt_ctc.yaml | 3 | 4.168879509 | 4.168879509 | 0 |
| parakeet_tdt_ctc.yaml | 4 | 4.292806625 | 4.292806625 | 0 |
| parakeet_tdt_ctc.yaml | 5 | 4.330666542 | 4.330666542 | 0 |
| parakeet_tdt_ctc.yaml | 6 | 4.588917732 | 4.588917732 | 0 |
| parakeet_tdt_ctc.yaml | 7 | 4.352891922 | 4.352891922 | 0 |
| parakeet_tdt_ctc.yaml | 8 | 4.115071774 | 4.115071774 | 0 |
| parakeet_tdt_ctc.yaml | 9 | 3.945665836 | 3.945665836 | 0 |
| parakeet_tdt_ctc.yaml | 10 | 4.274981022 | 4.274981022 | 0 |
| parakeet_tdt_ctc.yaml | 11 | 4.160243988 | 4.160243988 | 0 |
| parakeet_tdt_ctc.yaml | 12 | 3.875901937 | 3.875901937 | 0 |
| parakeet_tdt_ctc.yaml | 13 | 3.933950901 | 3.933950901 | 0 |
| parakeet_tdt_ctc.yaml | 14 | 3.755329609 | 3.755329609 | 0 |
| parakeet_tdt_ctc.yaml | 15 | 3.842600346 | 3.842600346 | 0 |
| parakeet_tdt_ctc.yaml | 16 | 4.193862915 | 4.193862915 | 0 |
| parakeet_tdt_ctc.yaml | 17 | 3.789740801 | 3.789740801 | 0 |
| parakeet_tdt_ctc.yaml | 18 | 3.917537689 | 3.917537689 | 0 |
| parakeet_tdt_ctc.yaml | 19 | 3.843187809 | 3.843187809 | 0 |
| parakeet_tdt_ctc.yaml | 20 | 3.797488928 | 3.797488928 | 0 |
| canary_aed.yaml | 1 | 4.192436218 | 4.192436218 | 0 |
| canary_aed.yaml | 2 | 4.398442745 | 4.398442745 | 0 |
| canary_aed.yaml | 3 | 4.417659283 | 4.417659283 | 0 |
| canary_aed.yaml | 4 | 4.281785965 | 4.281785965 | 0 |
| canary_aed.yaml | 5 | 4.221023083 | 4.221023083 | 0 |
| canary_aed.yaml | 6 | 4.010397434 | 4.010397434 | 0 |
| canary_aed.yaml | 7 | 4.013031006 | 4.013031006 | 0 |
| canary_aed.yaml | 8 | 4.119941235 | 4.119941235 | 0 |
| canary_aed.yaml | 9 | 3.937855721 | 3.937855721 | 0 |
| canary_aed.yaml | 10 | 3.911304951 | 3.911304951 | 0 |
| canary_aed.yaml | 11 | 3.761414528 | 3.761414528 | 0 |
| canary_aed.yaml | 12 | 4.013255596 | 4.013255596 | 0 |
| canary_aed.yaml | 13 | 3.823221207 | 3.823221207 | 0 |
| canary_aed.yaml | 14 | 3.825926542 | 3.825926542 | 0 |
| canary_aed.yaml | 15 | 3.675003052 | 3.675003052 | 0 |
| canary_aed.yaml | 16 | 3.626789093 | 3.626789093 | 0 |
| canary_aed.yaml | 17 | 3.583343506 | 3.583343506 | 0 |
| canary_aed.yaml | 18 | 3.634305000 | 3.634305000 | 0 |
| canary_aed.yaml | 19 | 3.649948597 | 3.649948597 | 0 |
| canary_aed.yaml | 20 | 3.668887138 | 3.668887138 | 0 |
| sortformer_diar.yaml | 1 | 0.702736020 | 0.702736020 | 0 |
| sortformer_diar.yaml | 2 | 0.696867526 | 0.696867526 | 0 |
| sortformer_diar.yaml | 3 | 0.676803946 | 0.676803946 | 0 |
| sortformer_diar.yaml | 4 | 0.664789379 | 0.664789379 | 0 |
| sortformer_diar.yaml | 5 | 0.648933589 | 0.648933589 | 0 |
| sortformer_diar.yaml | 6 | 0.623873234 | 0.623873234 | 0 |
| sortformer_diar.yaml | 7 | 0.607589006 | 0.607589006 | 0 |
| sortformer_diar.yaml | 8 | 0.582831562 | 0.582831562 | 0 |
| sortformer_diar.yaml | 9 | 0.566902995 | 0.566902995 | 0 |
| sortformer_diar.yaml | 10 | 0.543179154 | 0.543179154 | 0 |
| sortformer_diar.yaml | 11 | 0.543525457 | 0.543525457 | 0 |
| sortformer_diar.yaml | 12 | 0.537288308 | 0.537288308 | 0 |
| sortformer_diar.yaml | 13 | 0.509696364 | 0.509696364 | 0 |
| sortformer_diar.yaml | 14 | 0.506820500 | 0.506820500 | 0 |
| sortformer_diar.yaml | 15 | 0.486261100 | 0.486261100 | 0 |
| sortformer_diar.yaml | 16 | 0.507252574 | 0.507252574 | 0 |
| sortformer_diar.yaml | 17 | 0.490281254 | 0.490281254 | 0 |
| sortformer_diar.yaml | 18 | 0.493473053 | 0.493473053 | 0 |
| sortformer_diar.yaml | 19 | 0.474169284 | 0.474169284 | 0 |
| sortformer_diar.yaml | 20 | 0.488836557 | 0.488836557 | 0 |
| speaker_aware_turn.yaml | 1 | 4.404268742 | 4.404268742 | 0 |
| speaker_aware_turn.yaml | 2 | 4.158022881 | 4.158022881 | 0 |
| speaker_aware_turn.yaml | 3 | 4.358156204 | 4.358156204 | 0 |
| speaker_aware_turn.yaml | 4 | 3.700739145 | 3.700739145 | 0 |
| speaker_aware_turn.yaml | 5 | 6.003275394 | 6.003275394 | 0 |
| speaker_aware_turn.yaml | 6 | 5.055829048 | 5.055829048 | 0 |
| speaker_aware_turn.yaml | 7 | 4.639379501 | 4.639379501 | 0 |
| speaker_aware_turn.yaml | 8 | 3.926448822 | 3.926448822 | 0 |
| speaker_aware_turn.yaml | 9 | 4.145437241 | 4.145437241 | 0 |
| speaker_aware_turn.yaml | 10 | 4.095814705 | 4.095814705 | 0 |
| speaker_aware_turn.yaml | 11 | 3.514319658 | 3.514319658 | 0 |
| speaker_aware_turn.yaml | 12 | 4.545298576 | 4.545298576 | 0 |
| speaker_aware_turn.yaml | 13 | 4.316381454 | 4.316381454 | 0 |
| speaker_aware_turn.yaml | 14 | 3.973038435 | 3.973038435 | 0 |
| speaker_aware_turn.yaml | 15 | 3.624776602 | 3.624776602 | 0 |
| speaker_aware_turn.yaml | 16 | 3.642712593 | 3.642712593 | 0 |
| speaker_aware_turn.yaml | 17 | 3.499233007 | 3.499233007 | 0 |
| speaker_aware_turn.yaml | 18 | 3.523156404 | 3.523156404 | 0 |
| speaker_aware_turn.yaml | 19 | 3.380639553 | 3.380639553 | 0 |
| speaker_aware_turn.yaml | 20 | 4.064692974 | 4.064692974 | 0 |
