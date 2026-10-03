# audioforge/train.py split proposal (not applied: the 700-line rule says ask first)

train.py was 885 lines at 4895faf and is ~1200 after the accelerate loop. It does three jobs; the loop is the one
that belongs in train.py. Proposed split, by responsibility, public API unchanged (train.py re-exports every moved
name, so `from audioforge.train import load_model, run_recipe, Trainer, load_data, MixCollate, ...` keeps working
for the ~90 call sites in audioforge/, scripts/ and tests/):

| part | lines now | moves to | contents |
|---|---|---|---|
| training loop | ~700 | stays in `audioforge/train.py` | `Trainer` (optimizer groups, R2 guards, accelerate fit, timed eval, last/best, resume, signals, evaluate), `run_recipe`, `main` + size flags |
| recipe data assembly | ~190 | `audioforge/data.py` side (new `audioforge/datasets/recipe.py`, data.py would pass 700) | `load_data`, `_load_source`, `manifest_data`, `derive_labels`, `MixedData`, `load_mix`, `MixCollate` |
| model archive / build | ~170 | model side, new `audioforge/archive.py` (model.py is already 721) | `save_model`, `load_model`, `export_onnx`, `init_from_afm`, `build_model`, `build_tokenizer`, `build_codec` |
| side trainers | ~130 | `audioforge/train_aux.py` | `train_codec`, `_train_mel_codec`, `F_l1`, `train_speech_llm` (own loops for the codec and SALM demos, used only by `audioforge.cli train-codec / train-salm`) |

Order: archive first (most importers: serve, hub, api, conversation import `load_model` lazily from `.train`; the
re-export keeps them), then recipe data (tests/test_train_fixes.py's mix tests move with it), then side trainers.
Each step is a pure move; verify with `plans/trainer/parity_001.py --compare` (bit-identical losses) and the suite's
pass/skip counts.
