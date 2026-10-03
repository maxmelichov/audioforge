# Contributing

The binding rules are in [`AGENTS.md`](AGENTS.md) (workflow: tooling, plans, file size, commits, no long sleeps) and
[`docs/PROJECT.md`](docs/PROJECT.md) (the ML project: training loop, commands, sizing). This page is the how-to;
where the two disagree, those files win.

## Set up

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # the server extras + pytest, ruff, pre-commit, build
pre-commit install                 # ruff, large-file and config-doc checks on every commit
```

`pip install -e ".[research]"` adds what the research drivers and baseline benchmarks import (transformers,
pyannote, speechbrain, faster-whisper, ...). You only need it to reproduce the notes in `research/`.

Tasks run through [`chore`](https://github.com/getchore/chore) (MIT, one binary) from the root `chorefile`:

```bash
curl -fsSL https://getchore.github.io/install.sh | sh     # or the release tarball from GitHub into ~/.local/bin
chore list                                                # every task with its one-line description
chore <task> --dry                                        # print what a task would run, run nothing
```

Standalone Python scripts carry a [PEP 723](https://peps.python.org/pep-0723/) header, so `uv run scripts/x.py`
builds its own cached environment (the package comes in as an editable path dependency) and never touches `.venv`.
`.venv/bin/python scripts/x.py` keeps working unchanged. JavaScript tooling uses `pnpm` only.

## Tests and lint

```bash
chore lint          # ruff (F, E9, I, B; pyproject.toml), the client package, docs/CONFIGURATION.md vs the flag table
chore test-fast     # the CPU subset CI runs (tests/fast_ci.txt): no checkpoints, ~90 s, in the background
chore test          # everything, in the background through the gate
chore log test      # tail of the newest runs/logs/test_*.log
```

The test tasks print a pid and a log path and return at once; the result is the log's last lines (`== exit 0`).
The plain commands underneath are `ruff check audioforge tests integrations examples scripts`,
`python -m pytest -q $(grep -v "^#" tests/fast_ci.txt)`, `python -m pytest -q` and
`python scripts/dev/gen_config_doc.py --check`.

Tests that need real checkpoints (`runs/*.afm`, `models/`) or datasets (`$AUDIOFORGE_DATA`) skip themselves when the
files are absent. The heaviest ones are opt-in: `RUN_REAL=1`, `AUDIOFORGE_BIG_TESTS=1`. The server's regression
suite is `tests/test_serve*.py`, `test_bulletproof.py`, `test_tsvad_stream.py` and `test_diarization_fix.py`. Keep it
passing without editing it when you refactor the server.

CI (`.github/workflows/ci.yml`) runs ruff, mypy on the public surface, the config-doc check, a CLI smoke test and the
fast tests on Python 3.10 and 3.12, plus the `packages/audioforge-client` tests.

## Heavy jobs: the machine-safety gate

Anything that loads a model, trains, benchmarks or starts a server goes through the gate:

```bash
scripts/dev/gate.sh python -m pytest -q tests/test_serve.py                      # short: foreground is fine
scripts/dev/logged.sh -b <name> scripts/dev/gate.sh python scripts/research/<driver>.py ...   # long: background
chore log <name>                                                                 # tail of runs/logs/<name>_*.log
```

Anything over about 20 s runs in the background (or tmux) with its output tee'd to a log under `runs/`
(docs/PROJECT.md "Commands"). `scripts/dev/logged.sh` does that: `runs/logs/<name>_<UTC stamp>.log`, unbuffered,
ending with an `== exit <code>` line. Never pipe a long run through `| tail` or `| head`: they buffer until the end,
so nothing is readable while it runs and a crash loses the output. Report the log tail; do not `sleep` waiting for it.

The gate (written for macOS: it reads `sysctl vm.swapusage`) waits until fewer than 3 busy project Python processes are running, swap is under 14 GB and the internal
disk has at least 10 GB free. It then runs the command at `nice 5` with 2 threads (`OMP_NUM_THREADS=2`). The rules
behind it, learned on a shared laptop:

- one training at a time; keep each call under about 10 minutes, and make long jobs resumable stage by stage
  (see `scripts/research/tsvad_chain.sh`);
- nothing over 1 GB on the internal disk: large datasets and checkpoints live under `$AUDIOFORGE_DATA` /
  `$AUDIOFORGE_HOME`, which can point at external storage;
- never write to `~/.cache/huggingface`; `audioforge-download` keeps its own files;
- never delete `data/` or `runs/` files: `runs/*.json` hold every reported number.

## Plans and scratch files

The root holds only the project files (README, LICENSE, NOTICE, CHANGELOG, CITATION, CONTRIBUTING, AGENTS.md,
CLAUDE.md, pyproject.toml, chorefile) and the top directories. Every plan, survey, sweep table or scratch note goes
under `plans/<name>/`. A plan that can be checked gets a validation script `plans/<name>/<name>_001.py` (a standalone
`uv` script, then `_002`, ...) with a `.md` next to it that records what it checked and its output.
`chore rules` (`plans/repo_rules/repo_rules_001.py`) checks the repository rules themselves.

## File size

700 lines at most per file (AGENTS.md "File size"). When a file reaches it, ask before splitting; then split by
responsibility into halves, keep the public API where callers import it, and move tests with their code. The files
already over the limit and the plan for them are in `plans/file_size/`.

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
- Commit messages: a plain subject line (`area: what changed` is the house style), optionally a short body with
  the why and the evidence. No `Co-Authored-By`, session or other trailers (AGENTS.md "Commits").
