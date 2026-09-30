# Changelog

All notable changes to audioforge. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Every measured number cited here has its source in
`runs/*.json` (see `research/FINAL_REPORT.md`, Appendix A).

## [Unreleased]

### Results (2026-09-30): where single mode stands after today's changes
- **Turn rule:** single mode's default is `vad_head` (VAD head < 0.4 for ≥ 160 ms and turn head p ≥ 0.99, or 640 ms
  of silence). Calls: EOT p50 956 ms, 20 % false interruptions, 7 % missed; AMI: 1326 ms, 11 %, 34 %
  (research/EOT_LATENCY.md, `runs/eot_latency.json`).
- **Silero dropped** from single mode: not downloaded by default, not loaded; `hybrid_dyn` still works with `--silero`.
- **70 ms chunk-trigger fix** (d832fca): each ASR chunk runs once its mels are complete, so every frame, token and
  head output is ready 70 ms earlier, with identical outputs (research/LATENCY_BUDGET.md).
- **Voice-print check + anchored adaptation** (fe28a9e): clean prints and a working print kept at 0.8 × the enrolled
  one. Target-speaker WER on mixed calls 44.6 → 40.2 %, user channel 24.4 → 18.1 % (research/TSWER.md).
- **`--device mps`** (2bdf3e9): same events and decodes as CPU; 28.7 vs 29.8 ms per 160 ms chunk, 5 vs 4 real-time
  streams per process (research/MPS_115M.md).
- **README rewritten** around single mode: one results table with every number linked to its source.

### Changed (2026-09-30): `vad_head` waits for a surer head; the quickstart clip is no longer cut (research/EOT_LATENCY.md)
- **Why:** after the TS-VAD print fix (fe28a9e), the rule below cut the bundled quickstart clip mid-question at
  9.06 s when streamed with `examples/quickstart_client.py`. In the user's half-second pause the turn head reads
  0.94-0.98 and the served VAD dips in and out of 0.4, so the client's int16 truncation decided the cut.
- **The rule's defaults now:** VAD below 0.4 for ≥ 160 ms and turn head p ≥ 0.99, or 640 ms of that silence
  (`--vad-wait-ms 160,640`, `POLICY_THETA["vad_head"]` 0.99). The others path is unchanged. No code path changed.
- **Measured offline** on the post-fix dump, against the fe28a9e rule:
  - Two-party calls: EOT p50 / p95 956 / 1919 vs 916 / 2020 ms, false interruptions 20.2 vs 22.9 %, missed 7.3 vs
    7.3 %.
  - AMI: 1326 / 3758 vs 1327 / 4162 ms, false interruptions 10.5 vs 10.0 %, missed 33.5 vs 34.0 %.
  - The quickstart clip is not cut under any of six deliveries (float, int16 truncated / rounded, -6 / +6 dB,
    dither), in process and over the websocket (3 runs at 1x). VAD hysteresis, the head path at 400 ms and a
    minimum-speech guard were tried and each still cut it or missed more AMI ends.
  - Source: `runs/eot_latency.json` (`selection.fastest_print_fix_goal_no_clip_cut`, `demo_clip`, `served_check`).

### Changed (2026-09-30): single mode's turn rule is `vad_head`, no Silero (research/EOT_LATENCY.md)
- **`--mode single` adds `--turn-policy vad_head`.** This is the new server default for a session whose `config` names
  no `turn_policy`; a client's `config` still wins. Room mode keeps `timeout`.
- **The rule:** the served VAD head below 0.4 for ≥ 320 ms and turn head p ≥ 0.95, or 800 ms of that silence. With an
  enrolled TS-VAD track there is also the others path: the user's own silence reaching 960 ms while P(other) ≥ 0.9
  has held for 640 ms.
  - New flag `--others-wait-ms USER_SIL,HOLD`.
  - `--vad-wait-ms` now defaults to `320,800`.
- **Measured offline** on the served clock after the chunk-trigger fix, against `hybrid_dyn 2000,960`:
  - Two-party calls: EOT p50 / p95 881 / 1789 vs 1287 / 1923 ms, false interruptions 22.9 vs 22.9 %, missed 7.3 vs
    7.3 %.
  - AMI: 1330 / 4160 vs 1794 / 4285 ms, false interruptions 10.0 vs 8.5 %, missed 33.5 vs 37.5 %.
  - Source: `runs/eot_latency.json`.
- **Silero is no longer part of single mode.**
  - `audioforge-download`'s default set is `asr + tsvad` (+ `lid`).
  - `audioforge-serve --mode single` no longer passes `--silero`.
  - `hybrid_dyn` still works with `--silero` or `--with silero`.
- **`audioforge.load()` sessions** default to the engine's policy (`Frontend.session(turn_policy=None)`).
- **`scripts/research/eot_latency.py`:**
  - Re-dumped on `stage1_served_v2.afm` after d832fca: every frame is ready 70 ms earlier.
  - "Decision" latency is now total − compute; it no longer subtracts a 240 ms constant.
  - LiveKit's transcript lookup follows the new ASR clock.
  - New room-aware family `sim_room`.

### Changed (2026-09-29): one standard scorecard (research/METRICS.md)
- **New `research/METRICS.md`:** every published number in standard voice-agent metrics, one definition each. STT:
  WER on the full LibriSpeech test-clean / test-other (Whisper normalizer) and AMI, RTFx, partial latency (word end
  to word shown) and final latency. Turn detection: end-of-turn latency p50 / p95, false-interruption rate, response
  rate, TurnBench precision / recall / F1. VAD: ROC-AUC and F1. Speaker: target-speaker DER. Efficiency: compute per
  160 ms chunk, real-time streams per process, CPU seconds per audio second. The README's Results table and
  `research/SINGLE_MODEL.md` now use it.
- **Measured for it:**
  - Partial latency: 441 / 732 ms p50 / p95 on CPU (`scripts/research/stt_latency.py`, `runs/stt_latency.json`).
  - Full-LibriSpeech WER: 2.48 / 6.13 % (`scripts/research/asr_leaderboard.py`, `runs/asr_leaderboard.json`).
  - VAD AUC: 0.972 vs Silero 0.956 and MarbleNet 0.959 (`scripts/research/vad_auc.py`, `runs/vad_auc.json`).
  - 3 real-time sessions per process on 2 CPU threads (`scripts/research/streams_cpu.py`, `runs/streams_cpu.json`).
  - The live records re-scored per condition, with the target-speaker DER and TurnBench F1 derived from stored
    counts (`scripts/research/metrics_table.py`, `runs/metrics.json`).
- **Removed:**
  - The "first words" metric (1081 ms): it counted from the start of speech.
  - The pooled "missed within 3 s" headline: 32 of its 69 sessions were mono mixes, where the recorded partner
    answers in the same audio and no system can respond in time.
  - The name "dead air" (now end-of-turn latency) and "cut-ins" (now false interruptions).
- `research/FINAL_REPORT.md` is marked historical (room-mode era).


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
