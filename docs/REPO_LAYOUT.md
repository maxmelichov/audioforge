# Repository layout

Rules for what goes where and how work is done: [`../AGENTS.md`](../AGENTS.md) (workflow) and
[`PROJECT.md`](PROJECT.md) (ML project). How-to: [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

| path | what it is |
|---|---|
| `audioforge/` | the Python package: model and heads, streaming server (`serve.py`, `server/`), Python API (`api.py`), Pipecat / LiveKit adapters (`integrations/`), model download (`hub.py`), training |
| `tests/` | pytest suite; `tests/fast_ci.txt` is the CPU subset CI runs |
| `examples/` | runnable demos (Pipecat, LiveKit, plain client) and the bundled `audio/` clips |
| `docs/` | user and operator docs: architecture, configuration (generated), protocol, models, server internals, release checklist |
| `assets/` | the small shipped head checkpoints (`*.pt`) that single-model mode attaches to the NVIDIA encoder |
| `packages/audioforge-client/` | the standalone WebSocket client package (own `pyproject.toml` and tests) |
| `integrations/` | old import names of the adapters (now in `audioforge/integrations/`) and `turnbench_scorer/`, the vendored MIT TurnBench scorer |
| `scripts/` | `stream_client.py` (reference client), `sweep_capacity.py` (head sizing sweeps); `dev/` tooling (`gate.sh` job gate, `logged.sh` background runs with a log under `runs/logs/`, `build_public.sh` public snapshot, `render_compare.py` README comparison images, config-doc generator); `research/` the drivers behind every number (indexed in its README); `archive/` retired one-off drivers. Scripts carry a PEP 723 header: `uv run scripts/<x>.py` |
| `plans/` | every plan, survey, sweep table and scratch note, one directory per topic: `plans/<name>/<name>_001.py` validation scripts (standalone `uv` scripts) with a `.md` of what they checked and printed |
| `research/` | lab notes. Current reports at the top (`METRICS.md` = the numbers to quote, `FINAL_REPORT.md`, `SINGLE_MODEL.md`, ...); `recipes/` training configs; `archive/` superseded notes and raw Hugging Face pulls; `README.md` indexes all of it |
| `runs/` | committed result JSONs that the notes, docs and images cite (checkpoints `*.afm` / `*.pt` are git-ignored); `archive/` results no current doc reads |
| `demo/` | launch images (`images/`, current `*_v8*` / `*_v9*` plus the `redesign/` pipeline); `archive/` earlier video pipelines and image versions |

Top-level files (the root holds nothing else; plans and scratch go under `plans/`): `README.md`, `CHANGELOG.md`,
`CONTRIBUTING.md`, `AGENTS.md` (agent and workflow rules; `CLAUDE.md` is a symlink to it), `LICENSE` (Apache-2.0),
`NOTICE` (third-party terms), `CITATION.cff`, `pyproject.toml`, `chorefile` (the tasks: `chore list`),
`.gitignore`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`. `chore rules` checks this mechanically.

Rules that shape the tree: 700 lines at most per file (the known exceptions and their split plan are in
`plans/file_size/`); commit messages are a plain subject line with an optional short body and no trailers.

Not in git: `data/` (datasets, `$AUDIOFORGE_DATA`), `models/` (written by `audioforge-download`), `runs/*.afm`,
renders (`$DEMO_OUT`), `logs/` (and `runs/logs/`, the task logs), `scratch/`, `.chore/`.

Archives are moved with `git mv`, never rewritten: a note in `research/archive/` reads as it did on its date, and
every path that pointed at it was updated when it moved (2026-09-30).
