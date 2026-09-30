# Repository layout

| path | what it is |
|---|---|
| `audioforge/` | the Python package: model and heads, streaming server (`serve.py`, `server/`), Python API (`api.py`), Pipecat / LiveKit adapters (`integrations/`), model download (`hub.py`), training |
| `tests/` | pytest suite; `tests/fast_ci.txt` is the CPU subset CI runs |
| `examples/` | runnable demos (Pipecat, LiveKit, plain client) and the bundled `audio/` clips |
| `docs/` | user and operator docs: architecture, configuration (generated), protocol, models, server internals, release checklist |
| `assets/` | the small shipped head checkpoints (`*.pt`) that single-model mode attaches to the NVIDIA encoder |
| `packages/audioforge-client/` | the standalone WebSocket client package (own `pyproject.toml` and tests) |
| `integrations/` | old import names of the adapters (now in `audioforge/integrations/`) and `turnbench_scorer/`, the vendored MIT TurnBench scorer |
| `scripts/` | `stream_client.py` (reference client); `dev/` tooling (`gate.sh` job gate, config-doc generator); `research/` the drivers behind every number (indexed in its README); `archive/` retired one-off drivers |
| `research/` | lab notes. Current reports at the top (`METRICS.md` = the numbers to quote, `FINAL_REPORT.md`, `SINGLE_MODEL.md`, ...); `recipes/` training configs; `archive/` superseded notes and raw Hugging Face pulls; `README.md` indexes all of it |
| `runs/` | committed result JSONs that the notes, docs and images cite (checkpoints `*.afm` / `*.pt` are git-ignored); `archive/` results no current doc reads |
| `demo/` | launch images (`images/`, current `*_v8*` / `*_v9*` plus the `redesign/` pipeline); `archive/` earlier video pipelines and image versions |

Top-level files: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE` (Apache-2.0), `NOTICE` (third-party
terms), `CITATION.cff`, `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`.

Not in git: `data/` (datasets, `$AUDIOFORGE_DATA`), `models/` (written by `audioforge-download`), `runs/*.afm`,
renders (`$DEMO_OUT`), `logs/`, `scratch/`.

Archives are moved with `git mv`, never rewritten: a note in `research/archive/` reads as it did on its date, and
every path that pointed at it was updated when it moved (2026-09-30).
