# Contributing

## Set up

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # the server extras + pytest, ruff, pre-commit, build
pre-commit install                 # ruff, large-file and config-doc checks on every commit
```

`pip install -e ".[research]"` adds what the research drivers and baseline benchmarks import (transformers,
pyannote, speechbrain, faster-whisper, ...). You only need it to reproduce the notes in `research/`.

## Tests and lint

```bash
ruff check audioforge tests integrations examples scripts     # F, E9, I, B (pyproject.toml)
python -m pytest -q $(grep -v "^#" tests/fast_ci.txt)          # the CPU subset CI runs: no checkpoints, ~90 s
python -m pytest -q                                            # everything
python scripts/dev/gen_config_doc.py --check                   # docs/CONFIGURATION.md matches the flag table
```

Tests that need real checkpoints (`runs/*.afm`, `models/`) or datasets (`$AUDIOFORGE_DATA`) skip themselves when the
files are absent. The heaviest ones are opt-in: `RUN_REAL=1`, `AUDIOFORGE_BIG_TESTS=1`. The server's regression
suite is `tests/test_serve*.py`, `test_bulletproof.py`, `test_tsvad_stream.py` and `test_diarization_fix.py`. Keep it
passing without editing it when you refactor the server.

CI (`.github/workflows/ci.yml`) runs ruff, mypy on the public surface, the config-doc check, a CLI smoke test and the
fast tests on Python 3.10 and 3.12, plus the `packages/audioforge-client` tests.

## Heavy jobs: the machine-safety gate

Anything that loads a model, trains, benchmarks or starts a server goes through the gate:

```bash
scripts/dev/gate.sh python -m pytest -q tests/test_serve.py
scripts/dev/gate.sh python scripts/research/<driver>.py ...
```

The gate (written for macOS: it reads `sysctl vm.swapusage`) waits until fewer than 3 busy project Python processes are running, swap is under 14 GB and the internal
disk has at least 10 GB free. It then runs the command at `nice 5` with 2 threads (`OMP_NUM_THREADS=2`). The rules
behind it, learned on a shared laptop:

- one training at a time; keep each call under about 10 minutes, and make long jobs resumable stage by stage
  (see `scripts/research/tsvad_chain.sh`);
- nothing over 1 GB on the internal disk: large datasets and checkpoints live under `$AUDIOFORGE_DATA` /
  `$AUDIOFORGE_HOME`, which can point at external storage;
- never write to `~/.cache/huggingface`; `audioforge-download` keeps its own files;
- never delete `data/` or `runs/` files: `runs/*.json` hold every reported number.

## Where things live

[`docs/REPO_LAYOUT.md`](docs/REPO_LAYOUT.md) is the one-screen map of the tree. In short: package code in
`audioforge/`, tests in `tests/`, user docs in `docs/`, experiment drivers in `scripts/research/`, dated notes in
`research/` (current reports at the top, superseded ones in `research/archive/`), their result JSONs in `runs/`, and
the launch images in `demo/images/`. Anything under an `archive/` directory is kept for the record and not maintained.

| you want to change | look at |
|---|---|
| server behaviour | `audioforge/serve.py` (engine, session, connection loop) and `audioforge/server/` |
| a server flag | `audioforge/server/cli.py` (`FLAGS`), then `python scripts/dev/gen_config_doc.py` |
| the wire protocol | `audioforge/server/protocol.py` (`SCHEMA`, `validate`) and `docs/PROTOCOL.md` |
| the Python API | `audioforge/api.py` |
| Pipecat / LiveKit adapters | `audioforge/integrations/` |
| model code, heads, training | `audioforge/model.py`, `audioforge/heads/`, `audioforge/train.py`; recipes in `research/recipes/` |
| an experiment | a driver in `scripts/research/` (indexed in its README) and a dated note in `research/` |

Research code stays out of the package's serving path. A new driver goes in `scripts/research/` with a docstring
that names its note, and its numbers go to `runs/<name>.json`. A result is cited in a note only with its JSON.
`research/README.md` indexes the notes. Notes are dated records: correct them in place, marked "Corrected <date>",
rather than rewriting history.

## Style

- Library code logs through `logging.getLogger(__name__)` (the `audioforge` namespace); only CLI `main()` functions
  print.
- Public functions and classes get type hints and a docstring; package modules declare `__all__`.
- The version lives in `audioforge/__init__.py` only.
- Commit messages: a short `area: what changed` subject, then why, with the evidence (tests, measurements).
