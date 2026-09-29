# Changelog

All notable changes to audioforge. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Every measured number cited here has its source in
`runs/*.json` (see `research/FINAL_REPORT.md`, Appendix A).

## [Unreleased]

### Added (2026-09-29): `--device cuda` for the streaming server (research/GPU_RUN_2026-09-29.md)
- `audioforge-serve --device cuda`, `audioforge-bench --device cuda` and `audioforge.load(device="cuda")` run the
  streaming engine with the models on an NVIDIA GPU (opt-in; cpu stays the default, other devices still fall back
  to cpu). The events equal cpu's on the bundled clips and on 69 live Pipecat sessions. On an RTX 5090 the 160 ms
  block compute went from 77 to 20 ms (single mode), room mode from RTF 1.02 to 0.16, and one server process held 4
  live sessions (2 CPU threads: 1).
- TF32 is off in that mode (full fp32). PyTorch's default cuDNN TF32 moved the encoder output 1.5 % from cpu and
  changed live transcripts (2 of 3 runs in a real LiveKit room).

### Fixed (2026-09-29, fresh Linux clone)
- `audioforge.load()` / `audioforge-bench` found the downloaded Silero only under `$AUDIOFORGE_DATA/silero`, so
  hybrid_dyn fell back to hybrid; the engine now also looks in the models directory.
- `audioforge.load()` / `audioforge-bench` raised without `lid_distill.pt` (not in `assets/`, and the release asset
  is missing); they now turn language ID off, as `audioforge-serve` does.
- `scripts/dev/gate.sh` works on Linux (it read macOS-only `sysctl` / `df -g` and never passed).
- `--final-asr` (process worker): a child that dies while loading now raises instead of hanging the server.

### Changed (2026-09-29): the shipped model's VAD head reads block 4
- `audioforge-download` now builds **`stage1_served_v2.afm`** from `assets/served_heads_v0.2.pt` (104 tensors,
  19,629,563 bytes, sha256 `cb5aa069…`; pinned in `hub.HEADS`). Its VAD head reads encoder block 4 only (the speaker
  and TS-VAD heads' tap): AMI VAD F1 0.951 vs 0.949, no downstream regression (research/VAD_SINGLE.md,
  runs/vad_single.json). Every other tensor is bit-identical to `stage1_served.afm`.
- `audioforge-serve`, `audioforge.load()` and `audioforge-bench` load that build (`hub.SERVED`). An existing
  `models/stage1_served.afm` is no longer picked up: re-run `audioforge-download`.
- `runs/stage1_served.afm` stays the 2026-09-27 measured checkpoint that the research documents cite.
  `audioforge-download --heads-version 0.1` rebuilds it from `assets/served_heads_v0.1.pt`, which is kept.
- The heads file now has its size and sha256 checked, not only the merged model's tensor hash.

### Changed (2026-09-29): single-model mode is the default
- **`audioforge-serve` runs single-model mode by default.** One 115M model follows the known user by a stored voice
  print (target diarization) and loads no diarizer. The previous two-model stack is **`--mode room`**: general
  diarization with NVIDIA Nemotron-3-Diarization (the room default when downloaded, else Sortformer v2), optionally
  with `--final-asr tdt_v3`. Without `--mode`, `--diarizer`, `--diar` or `--final-asr` select room mode, so older
  command lines keep working. `--mode default` is gone (renamed `room`). `audioforge.load()` and `audioforge-bench`
  follow the same rule.
- **Voice sample requirement.** Send the user's stored print, `{"type": "enroll", "embedding": [...]}`, made from
  ≥ 5 s of clean speech (10 s for meetings). Live grabs after `agent_end` cost 16-55 points of missed turn ends in
  meetings.
  - New: `audioforge.voiceprint(audio)` and `Frontend.voiceprint`.
  - `voiceprint` messages now carry the `embedding`.
  - Both adapters send a stored print (`AudioforgeSTTService.enroll(embedding=...)`,
    `AudioforgeFrontend.enroll(embedding=...)`) under any enroll mode.
  - `examples/quickstart_client.py` streams a 16 s otoSpeech two-party call (`examples/audio/two_party_call_16s.wav`,
    CC BY 4.0) and sends its user's stored print. The LibriSpeech clip `two_speakers_10s.wav` stays as room mode's
    example.
- **`audioforge-download` defaults to `asr + tsvad + lid + silero`.** The TS-VAD head (`tsvad_spk.pt`, 1.0 MB) and
  the LID head (`lid_distill.pt`, 3.7 MB) are sha256-pinned components, shipped in `assets/` and attached to the
  v0.1.0 release. `--diarizer nemotron3|sortformer` adds room mode's diarizer. A failed head download stops with one
  line naming the file.
- In single-model mode `final.speaker` is 0 for the user and 1 for someone else (the dominant TS-VAD column of the
  turn).

### Added (2026-09-29, single-model mode, research/SINGLE_MODEL.md)
- **`audioforge-serve --mode single`** (config key `mode: single`, `audioforge.load(mode="single")`,
  `audioforge-bench --mode single`): everything from the one 115M checkpoint for a known user. It adds `--turn-input
  tsvad --diar-off --lid head --enroll after_agent_arm --dyn-wait-ms 2000,960`, loads no diarizer and no final-ASR
  worker, and refuses options that would load a second model. The default mode is unchanged.
- `--diar-off` without `--diar` no longer loads a diarizer at all; `ready.diar_config` is `"off"` then.
- An `enroll` message with a stored `embedding` is accepted under every `--enroll` mode when `--turn-input tsvad` is on.
- `--dyn-wait-ms CAP,FLOOR` sets `hybrid_dyn`'s silence wait.
- `scripts/research/single_model_distill/`: the GPU package for target-speaker and turn heads trained on more data
  (prepared, smoke-tested, not run).

### Changed (2026-09-29 cleanup for the public release)
- **Layout.** Research drivers and drafts left the top level: `scripts/bench_serve.py`, `uc_turn.py`,
  `make_served_model.py`, `run_all.sh`, `audioforge/uc_stream.py` and `audioforge/heads/uc_turn.py` (now
  `scripts/research/uc_turn_head.py`) moved to `scripts/research/` (indexed in its new README); `recipes/` moved to
  `research/recipes/` (with an index; `runs/*.json` provenance still cites the old paths). `scripts/bench.py` and
  `scripts/download_models.py` were deleted (duplicates of `audioforge-bench` / `audioforge-download`).
- **Server.** `audioforge/serve.py` (2755 lines) was split: the engine, sessions and connection loop stay in
  `serve.py`; the protocol, turn policies, speaker binding, ASR streams, flags and helpers are in `audioforge/server/`.
  `audioforge.serve.X` and `python -m audioforge.serve` are unchanged. The design docstring moved to
  `docs/SERVER_INTERNALS.md`.
- **Config.** One flag table (`audioforge/server/cli.py`) produces `--help` (everyday flags, one screen),
  `--help-advanced` (all; `--help-serve` still works), `--config FILE` / `$AUDIOFORGE_CONFIG` (YAML or JSON, flags
  win) and the flag reference in `docs/CONFIGURATION.md`, generated by `scripts/dev/gen_config_doc.py` and checked by
  `tests/test_config_doc.py` and CI. The Silero default path follows `$AUDIOFORGE_DATA`.
- **Adapters.** The Pipecat and LiveKit adapters ship in the package as `audioforge.integrations.pipecat` /
  `.livekit` (`pip install "audioforge[pipecat]"` is enough); `integrations.pipecat_audioforge` /
  `integrations.livekit_audioforge` remain as aliases.
- **Logging.** Library code logs under the `audioforge` logger (plain messages to stdout at INFO by default, so the
  console output is unchanged); `audioforge.use_app_logging()` hands records to the application's handlers.
- `audioforge-bench --diarizer nemotron3` now applies the launcher's settings (all 8 columns, hold on shed): the
  product diarizer settings live once in `hub.diarizer_defaults`.
- README cut to what a user needs; its synthetic-results, early live-server and corrections-log sections moved to
  `research/EARLY_RESULTS.md`.

### Added (2026-09-29)
- Python API: `audioforge.load()` -> `Frontend.session()` -> `Session.feed(pcm)` / `end()` returning the protocol's
  messages; `examples/python_api.py`; `tests/test_api.py`.
- `docs/ARCHITECTURE.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.pre-commit-config.yaml`.
- Lint: ruff now also enforces import order (I) and bugbear (B); mypy runs on the public surface in CI; CI checks the
  generated config doc and smoke-tests the CLIs.

### Earlier unreleased changes

### Added
- `audioforge/perf.py` + `audioforge-serve --perf` (default on): exact CPU inference fast paths (column-major
  `nn.Linear` weights for Accelerate's NN GEMM path, cached relative-position projection, one shared subsampling for
  the ASR and turn passes, cached RNNT joint prediction projection). Server compute 0.62x with identical protocol
  decisions (0 differences in 1313 messages, probabilities within 1e-5; A/B on the 5 AMI windows; `runs/perf.json`,
  `research/PERFORMANCE.md`); `scripts/research/bench_serve.py`
  (`run` / `components` / `ab` / `compare` / `table`) and `tests/test_perf.py`.
- Many-speaker diarization (2026-09-28, `research/DIARIZATION_FIX.md`, `tests/test_diarization_fix.py`):
  `serve --diar-labels registry` (stable voice-keyed speaker ids per session from the served speaker head, or
  TitaNet-L with `--diar-embed titanet`; `final.speaker_conf`, `final.diar_shed`, `stats.speakers_seen`),
  `--shed-diar hold` (no speaker-0 collapse under load shedding) and session config `turn_policy: "timeout_any"`.
- TS-VAD turn path (2026-09-28, behind flags; `research/IMPROVEMENTS.md` §1): `audioforge/tsvad_stream.py`,
  `serve --turn-input tsvad [--diar-off] [--tsvad-print-s S] [--tsvad-refresh-s S]`, the `enroll` message with a stored
  `embedding`, the `voiceprint` message; Pipecat adapter `enroll(embedding=...)`; `tests/test_tsvad_stream.py`.
- Server hardening (2026-09-28, `research/BULLETPROOF.md`): structured `error` messages, `GET /health`, `--log-json`,
  RTF-gated load shedding, final-ASR watchdog, idle / session limits; `tests/test_bulletproof.py` (48 tests).
- `scripts/research/distill_0p6b_to_115m/`: the GPU package that distils the 0.6B streaming model into the 115M core
  (setup, preflight, KD with a WER gate and a no-teacher control; smoke-tested on ~1 h; ~6 A100 hours estimated).
- `research/VERIFICATION_2026-09-28.md`: the final report's numbers read back from their JSON (551 claims).
- `research/PERFORMANCE.md`: every RTF figure in the repository labelled by what it measures (model-only offline
  batch decode 0.015 / 0.022 vs live server 0.64 / 0.80 vs the on-device runtime study 0.16), per-component cost of
  the served path per 160 ms, concurrency, accepted / rejected optimisations.
- `pyproject.toml`: pip-installable `audioforge` with a minimal core (torch, numpy, pyyaml, sentencepiece,
  soundfile, huggingface_hub) and extras `serve`, `pipecat`, `livekit`, `nemo-import`, `dev`, `research`; console
  entry points `audioforge`, `audioforge-serve`, `audioforge-download`, `audioforge-bench`.
- `audioforge-download` (`audioforge/hub.py`): fetches the pinned NVIDIA / Silero checkpoints by Hugging Face
  revision or fixed URL, verifies size and sha256, shows each model licence and asks for acceptance, converts
  `.nemo` to `.afm`, and rebuilds the served model bit-identically from NVIDIA's weights plus
  `assets/served_heads_v0.1.pt` (tensor-hash asserted). Reuses files already under `$AUDIOFORGE_DATA/nemo` and
  `--from-local` directories instead of downloading.
- `audioforge-serve` (`audioforge/launch.py`): starts `audioforge.serve` with the downloaded models;
  `--diarizer nemotron3` applies the measured Nemotron-3 configuration (`--diar-pool max --diar-left 1`).
- `audioforge-bench`: in-process RTF / per-block latency / events on a WAV file, no server.
- `AUDIOFORGE_DATA` (datasets, `.nemo`, Silero; default `<repo>/data`) and `AUDIOFORGE_HOME` (downloaded models;
  default `<repo>/models` in a checkout, else `~/.cache/audioforge`), so data can live on external storage.
- `examples/quickstart_client.py` and the bundled 10 s two-speaker clip `examples/audio/two_speakers_10s.wav`
  (LibriSpeech, CC-BY-4.0).
- `docs/CONFIGURATION.md` (every server flag with its measured cost), `docs/PROTOCOL.md` (the WebSocket wire
  protocol), `research/README.md` (index of the research notes), `docs/RELEASE_CHECKLIST.md`.
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff plus the fast CPU tests (`tests/fast_ci.txt`) on Python
  3.10 and 3.12, and the `audioforge-client` package tests; no checkpoints are downloaded.
- `packages/audioforge-client`: a torch-free WebSocket client and protocol framing for `audioforge.serve`.
- `LICENSE` (Apache-2.0) and `NOTICE` (model-weight and vendored-code licences). The licence choice is still to be
  confirmed by the maintainer (see `docs/RELEASE_CHECKLIST.md`).

### Changed
- Repository layout: research drivers moved from `scripts/` to `scripts/research/`; `scripts/` keeps the user-facing
  `bench.py`, `download_models.py`, `stream_client.py`, `make_served_model.py`, `run_all.sh`; machine helpers in
  `scripts/dev/`. `tests/conftest.py` puts both script directories on `sys.path`. Research notes under `research/`
  still cite the old `scripts/<name>.py` paths.
- README: the results section is now the final scorecard from `research/FINAL_REPORT.md` (14 rows, each traced to
  its `runs/*.json`); the 2026-09-26 "one model vs the dedicated models" table it replaces showed the original
  speaker head (32.2 % EER) and a product verdict that were both superseded.
- `.gitignore`: `data/`, `models/`, `runs/*.afm|pt|onnx`, `*.log`, demo renders and scratch venvs are ignored.
- `audioforge-serve --diarizer nemotron3` no longer adds `--diar-spks 4` (it hid speakers 5-8): `frame.speakers` now
  carries 8 columns for Nemotron-3 (`--diar-spks 4` restores 4); the launcher adds `--shed-diar hold` for either
  diarizer unless given (2026-09-28). `python -m audioforge.serve` defaults are unchanged.
- `research/FINAL_REPORT.md` / `.html` refreshed on 2026-09-28 (scorecard rows 15-26, §4.1, §5.6-5.8, §7.1-7.3, §8.1);
  README scorecard, live-server and demo sections follow it. Doc corrections from the verification pass:
  `DIARIZATION_FIX.md` §3.4 (AMI-64 DER after the fix is 0.245, not 0.229: the old row was a 57-of-64-window partial),
  `BULLETPROOF.md` (48 tests, not 45), `IMPROVEMENTS.md` §3 (TitaNet row columns).

### Measured (2026-09-27, `research/FINAL_REPORT.md`)
- One frozen 115M NVIDIA streaming encoder with small trained heads, plus NVIDIA's diarizer, at 160 ms chunks on a
  2-thread laptop CPU: best streaming VAD measured on AMI (F1 0.949), fewer missed turn ends than every dedicated
  turn detector at matched false cutoffs on AMI (hybrid 61.9 % vs Silero timeout 72.7 %) and held-out ICSI, and
  1.6-5.5x fewer cut-ins than Pipecat's default stack end to end; worse than the dedicated models at meeting ASR
  (24.4 % WER, fixed by a per-turn Parakeet-TDT v3 pass: 9.7 %), speaker verification, diarization and language ID.

### Measured (2026-09-28, `research/FINAL_REPORT.md` rows 15-26)
- Many-speaker rooms: DER 0.395 -> 0.247 and the speaker count exact on three 4-6-speaker clips, forced-shedding DER
  0.809 -> 0.329, AMI 20 s windows unchanged (0.245). Server compute 0.62x with the fast paths.
- TS-VAD turn path: missed turn ends 61.9 -> 39.3 % (AMI) and 68.5 -> 20.4 % (ICSI) offline with a 5 s voice print;
  live on 69 mostly two-party sessions +10.8 [+5.6, +15.8] points of misses at 3 s vs the default, equal to the
  no-cut-in rule at 0.43x compute (behind flags).
- 0.6B streaming core not adopted (AMI `hybrid_dyn` misses +2.8 [+0.1, +5.3], five categories equal; meeting WER -13
  points). Decoder-only adaptation AMI 24.43 -> 22.50 %; `--asr-lookahead 13` -1.41 / -2.50 / -0.37 WER points;
  crop-level speaker head 19.8 -> 17.1-17.4 % EER; transcript LID 100 % / 91.7 %.
- Negative: TS-VAD-trained predictive turn head, `hybrid_fast`, diarization distillation into a 1.9 M head.

### Open before release (2026-09-28; details in `docs/RELEASE_CHECKLIST.md`)
- Licence confirmation (Apache-2.0 is a placeholder), author e-mail in the commit metadata, Hugging Face token
  rotation, the upstream PRs of `docs/UPSTREAM_PRS.md`, the GPU distillation run, a human listen of the demo video.

### Tests (2026-09-28)
- After the final-report refresh (one gated run each): `ruff check audioforge tests integrations examples scripts`
  reports 4 findings (F401 in `audioforge/heads/uc_turn.py` and `scripts/research/uc_turn.py`, F841 in
  `tests/test_diarization_fix.py` and `tests/test_uc_turn.py`), none in files this refresh touched, so CI's ruff step
  fails until they are fixed (`ruff check --fix` handles the two F401); the fast CI subset (`tests/fast_ci.txt`):
  524 tests, 523 passed, 1 skipped, 0 failed.
- Full suite on the development machine (`scripts/dev/gate.sh .venv/bin/python -m pytest -q tests`, 10 min 47 s):
  718 passed, 5 skipped, 7 failed; all 7 failures are in the untracked, in-progress `tests/test_bulletproof.py`
  against an uncommitted `audioforge/serve.py` (another workstream), none in committed code.
- CI subset (`tests/fast_ci.txt`) in the fresh quickstart venv: 536 passed, 7 skipped, 84 s for the superset it was
  cut from; the 11 failures in that superset were research-only imports and those four files are excluded.

## [0.1.0] - unreleased
Initial public version; the entries above describe it.
