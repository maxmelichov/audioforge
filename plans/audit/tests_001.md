# Test suite audit 001 (read-only)

Date: 2026-10-03. Scope: `tests/` at HEAD 671fa71 (baseline 954 passed, 10 skipped, `plans/baseline/pytest.txt`).
Nothing under `tests/` or `audioforge/` was edited. I did not run the full suite. What I did run, all on CPU:
`pytest --setup-only -rs tests/` (this evaluates every skip marker without running any test body),
`tests/test_core_0p6b.py`, and two small scratch checks. Their outputs are in `runs/audit/tests_001_*.txt`.
The companion script `plans/audit/tests_001.py` (a PEP 723 uv script) repeats the gate, preset and hash checks.

Severity: **H** = a shipped behaviour can regress with every test still green. **M** = a test is weaker than it
looks, or can fail for reasons unrelated to the code. **L** = hygiene.

## 1. The 10 skips

| # | test | reason | does it hide a real failure here? |
|---|------|--------|------------------------------------|
| 1-2 | `test_gpu_device.py:44,55` `[cuda]` | no CUDA on this Mac | No. The `[mps]` variants ran and passed. |
| 3 | `test_hybrid_asr.py:165` | needs `AUDIOFORGE_BIG_TESTS=1` (the TDT 0.6B checkpoint is present) | No, this opt-in gate is on purpose. Nothing runs it: there is no chore task for it. |
| 4 | `test_nemo_import.py:316` | needs `AUDIOFORGE_BIG_TESTS=1` (the 0.6B `.nemo` is present) | Same as #3. |
| 5 | `test_serve.py:593` | `RUN_REAL=1` | Not hidden, but stale: it loads `runs/stage1_heads_pretrained.afm` + `runs/nemo_sortformer_v2.afm` (the old room stack), not the served v0.4 core (see H5). |
| 6 | `test_pipecat_integration.py:490` | `RUN_REAL=1` | Stale in the same way as #5. |
| 7 | `test_livekit_integration.py:568` | `RUN_REAL=1` | Stale in the same way as #5. |
| 8 | `test_perf.py:233` | `RUN_REAL=1` | Stale: it uses `runs/stage1_served.afm` (v0.1). |
| 9-10 | `test_core_0p6b.py:98` and `:128` (both through `_served_0p6b`, line 94) | "the 0.6B core is not downloaded" | **Yes. See H1.** The base `data/nemo/nemotron-speech-streaming-en-0.6b.nemo` and `assets/served_heads_0p6b_v0.4.pt` are both on this machine. Only the built `served_0p6b_v0.4.afm` is missing (`runs/` holds the v0.1 and v0.3 builds). The skip message is wrong, and the default 0.6B core is never streamed by any test. |

Every other skip guard (data/AMI/ICSI/Silero/Sortformer/TitaNet/`runs/*.afm`, plus the in-body skips in
`test_baselines_sd.py:124`, `test_edge_teachers.py:87`, `test_trail6.py:248/253`, `test_recipes.py:27`,
`test_nemo_import.py:185` and `test_hybrid_policy.py:130`) is open on this machine, and those tests ran. Every
module-level `importorskip` (websockets, pipecat, livekit, accelerate, pyarrow, pyannote, onnxruntime) is satisfied.
If any of them were missing, the skip count would be above 10.

## 2. Findings ranked by severity

### H1. The default 0.6B core (heads v0.4) is never run end to end, and the skip message hides it
`tests/test_core_0p6b.py:90-133`. `_served_0p6b()` skips when `hub.find_model("asr_0p6b")` is None. That lookup only
checks `models/served_0p6b_v0.4.afm` and `runs/served_0p6b_v0.4.afm`. It does not notice that the base `.nemo` and
the pinned heads, the two inputs `hub.build_served` needs, are both present.
**Fix:** skip only when the base `.nemo` is absent. Otherwise build the served model once into a session tmp dir
(or `models/`) with `hub.build_served`, then run. Change the skip text to name the missing file.

### H2. The per-core turn presets that ship are pinned nowhere
`assets/served_heads_0p6b_v0.4.pt` carries its own `config.turn_presets`:
- balanced: 320/720 ms, v5 at p 0.6
- fast: 80/720 ms, p 0.5
- assistant: 160/3440 ms, `head: turn_seg_a`, p 0.9, quiet_db 6

`serve.model_presets` merges these over `TURN_PRESETS`. No test loads the real file.
`tests/test_turn_preset.py:96-107` uses a hand-typed dict that looks like the balanced entry. The assistant entry's
`head: turn_seg_a` routing and the `turn_vad` head are covered only on tiny models (`test_turn_data.py`).
The 115M v0.4 heads carry no `turn_presets`, so they use `TURN_PRESETS`. Those are checked as policies on recorded
frames (`test_turn_preset.py:129`, `test_turn_seg.py:115`, `test_vad_head_policy.py:192`). Those fixtures were recorded
on `stage1_served_v2.afm` / `v3.afm` (their `source` / `note` fields say so), and nothing checks they still match what
v4 emits.
**Fix:**
- Add a test that does `torch.load(assets/served_heads_0p6b_v0.4.pt)["config"]` (27 MB, no base model needed) and
  asserts `model_presets(...)` plus `vad_head_params(...)` for balanced, fast and assistant, including
  `turn_model["head"] == "turn_seg_a"`. Do the same for the 115M v0.4 config (expected: `TURN_PRESETS` unchanged).
- Add a `RUN_REAL` test that re-records the bundled clip's frames on the served v4 and compares them to the fixture.

### H3. CI never runs the tests that guard the turn rules, dual rate or the 0.6B registry
`.github/workflows/ci.yml:56` runs only `tests/fast_ci.txt`. That list leaves out:
- `test_turn_preset`, `test_turn_seg`, `test_vad_head_policy`, `test_energy_gate`
- `test_dual_rate`, `test_serve`, `test_bulletproof`, `test_latency_budget`
- `test_core_0p6b`. Its first six tests are pure registry and launcher checks that take milliseconds.

A refactor that breaks a preset, the dual-rate partition or the 0.6B pins passes CI.
**Fix:** time these files on CPU and add the ones under about 20 s to `fast_ci.txt` (at least `test_core_0p6b`,
`test_turn_preset`, `test_turn_seg`, `test_vad_head_policy`, `test_dual_rate`). Mark the slow socket and soak tests
`slow` instead of dropping the whole file.

### H4. The 1.12 s dual-rate final is untested
`tests/test_dual_rate.py` runs only `SLOW_R = 6` (`FC = 560`) on tiny models. Nothing builds an engine with
`final_chunk_ms=1120` (`[70,13]`, `asr_lookahead == 13`), even on a tiny model. The real `[70,13]` stream == offline
check (`test_nemo_import.py:180`) covers the base 115M encoder only. It does not cover the served engine's slow
finals or the 0.6B core.
**Fix:**
- Parametrize `test_dual_rate` over `SLOW_R in (6, 13)`, with `att_context_sizes` holding `[8, 13]`, and assert
  `ready_msg()["final_chunk_ms"] == 1120` / `asr_lookahead == 13`.
- Add a skip-if-absent real test: slow finals on the bundled clip == `transcribe(att_context_size=[70,13])` on the
  served v4.

### H5. No CPU test runs the shipped 115M v0.4 for real, and the RUN_REAL tests run an outdated stack
The only test that loads the real `stage1_served_v4.afm`, `tsvad_spk.pt` and `lid_115m_v2.pt` is
`test_gpu_device.py:43`. It runs only with MPS or CUDA, and it only checks that GPU output equals CPU output on
`turn_end`/`final`/`voiceprint`. The four `RUN_REAL` tests (#5-8 above) load v0.1 or the pre-served room stack.
There is no chore task that runs them.
**Fix:**
- Switch the `RUN_REAL` tests to `hub.find_model("asr")` with `--mode single`.
- Add one CPU real-model smoke test (skip if absent): `audioforge.load().run_file(two_party_call_16s.wav)` gives a
  `turn_end` after 10.8 s and none inside the user turn, a non-empty final, and `language == "en"`.
- Add `chore test-real`.

### M1. Speech head and LID v2 are tested only on tiny models
Covered on tiny models: `test_fixall.py:34` (speech) and `test_lid.py` (LID). The real `heads.speech` and
`assets/lid_*_v2.pt` load only inside the MPS test above, and its `_decisions` (`test_gpu_device.py:30-41`) throws
away the speech and language fields.
**Fix:** cover them in the H5 smoke test: assert the `speech` field is in [0,1] and tracks VAD, and assert the
language event's label and timing. For 0.6B, assert `lid_0p6b_v2.pt` attaches to the 0.6B encoder (d_model 1024).

### M2. The public snapshot path (no `assets/lid_*.pt`) is skipped over, not tested
`test_single_mode.py:143` and `test_core_0p6b.py:35` `continue` when the LID file is absent. In the snapshot that
just checks less. The launcher branch "`lid_115m_v2.pt` not found, language ID is off" (`audioforge/launch.py`,
`resolve_models`) has no test. Neither does `scripts/dev/build_public.sh`.
**Fix:** monkeypatch `launch.find_head` to return None for the LID file. Assert that the argv has no `--lid`, that
stderr carries the message, and that a tiny-model `audioforge.load` still streams.

### M3. No test checks that the legacy `runs/*.afm` files equal base + heads
`hub.find_model` makes tests and the server use `runs/stage1_served_v4.afm` / `runs/served_0p6b_v*.afm` in this
checkout. `build_served` asserts the hash only at build time.
I checked by hand on CPU (`runs/audit/tests_001_hashchk.txt`): `stage1_served_v4.afm` and `served_0p6b_v0.3.afm`
both match their heads' `state_hash`.
**Fix:** add a skip-if-absent test (or run `uv run plans/audit/tests_001.py --hash`) comparing
`hub.state_hash(load_model(afm).state_dict())` to the heads blob's `state_hash`.

### M4. Wall-clock thresholds that will flake on a loaded machine
- `test_hybrid_asr.py:353`: `gaps.max() < 0.6` s, with no `SLACK`. The process variant spins the CPU on purpose.
- `test_bulletproof.py:459`: `last <= 1.5 * first + 0.002 * SLACK`, a ratio of medians across a 30-minute soak.
- `test_bulletproof.py:462`: RSS growth `< 40` MB, which depends on the allocator.

`test_bulletproof.py:677/709/819/938` already scale by `SLACK = 4`, which is fine.
**Fix:** multiply the first by `SLACK`. Mark the soak test `slow` and run its strict bounds only under
`BULLETPROOF_STRICT_TIMING=1`.

### M5. Likely-vacuous text equality
`test_core_0p6b.py:124` compares the streaming and offline transcripts of 3 s of Gaussian noise. Both are probably
`""`. I could not confirm because the v0.4 build is absent.
**Fix:** use the bundled clip and assert the text is non-empty.
(I checked the similar `test_served_v2.py:28`: v0.1 emits 4 tokens on its sine input, so it is not vacuous, just thin.)

### L1. Tests that only compare stored JSON and cannot fail from a code change
- `test_make_served_model.py:106` `test_real_build_report` (reads `runs/served_model_build.json`).
- `test_hybrid_policy.py:127` `test_stored_hybrid_run_reproduces_stored_pure_rows` (two stored `runs/*.json`).

**Fix:** rename them as data-integrity checks, or regenerate the rows from code on a small subset.

### L2. Broad excepts that turn regressions into skips
- `test_baselines_sd.py:123`: `except Exception` -> skip.
- `test_trail6.py:52`: `_have_ami_train` swallows every error, so the test silently skips.
- `test_edge_teachers.py:86`.
- `test_pipecat_integration.py:547,592`: `except Exception: pass`. The outcomes are still asserted afterwards.

**Fix:** catch `FileNotFoundError`, `OSError` or `huggingface_hub.errors.LocalEntryNotFoundError` only.

### L3. Ordering and working-directory dependence
- `test_recipe_yaml.py:33` reads `Path("research/recipes/...")` relative to the cwd, so it fails if pytest is
  started outside the repo root. Fix: use `ROOT /`.
- `torch.set_num_threads(1)` runs at import time in `test_deferred_fixes.py:14`, `test_edge_data.py:20` and
  `test_edge_audio.py:9`. Because import happens at collection, the whole session runs single-threaded, so a file
  run alone behaves differently from the same file in the full suite.
- `test_baselines_sd.py:109`, `test_baselines_turn.py:81` and `test_edge_teachers.py:105` change the thread count
  without restoring it.
- `test_edge_teachers.py:78` leaks `HF_HUB_OFFLINE` through `os.environ.setdefault`.

**Fix:** use fixtures that restore the thread count (as `test_latency_budget.py:18` does) and `monkeypatch.setenv`.

### L4. Unseeded randomness
`torch.randn` without a seed or generator appears in `test_components.py:15,24,60,118`,
`test_diarization_fix.py:320-322`, `test_completeness.py:27-80`, `test_diag_turn.py:82-195` and others. These are
mostly shape checks and property equalities with tolerances, so the risk is low. The values depend on which tests ran
before. Fix: add `torch.manual_seed` or a `Generator`.

### L5. No hang protection
`pytest-timeout` is not installed. A stuck socket test would block the 9-minute suite. Most awaits use `wait_for`;
`test_pipecat_integration.py:499` binds to port 0, closes the socket and then reuses the port, which can race
(RUN_REAL only).

### L6. PEP 723 scripts and chorefile tasks are untested
Five scripts carry `# /// script` headers: `scripts/stream_client.py`, `scripts/sweep_capacity.py`,
`scripts/research/voice_gender.py`, `scripts/dev/gen_config_doc.py` and `scripts/dev/render_compare.py`. Only
`gen_config_doc.py --check` runs, through `sys.executable` (`test_config_doc.py:16`), never through `uv run`. The
`chorefile` tasks (`test-fast` reads `tests/fast_ci.txt`; every entry exists today) have no smoke test.
**Fix:** a test that parses each header with `tomllib`, checks that its third-party imports are listed in
`dependencies`, and runs `--help`. If `chore` is on PATH, run `chore <task> --dry` for each task.

### L7. Dead branch
`test_core_0p6b.py:26` skips "if not sha", but the sha is always pinned now. Remove it.

## 3. What is solid (checked, no action)
- **Heads pins load the real files.** `test_single_mode.py:137-146` checks size + sha256 of every
  `assets/served_heads_v0.*.pt` plus `tsvad_spk.pt` / `lid_115m_v2.pt`. `test_core_0p6b.py:22-37` does the same for
  the 0.6B v0.4 heads plus `tsvad_0p6b.pt` / `lid_0p6b_v2.pt`. Neither skips here.
- **The offline simulators are independent.** `test_vad_head_policy` / `test_turn_preset` compare `VadHeadPolicy`
  against `scripts/research/eot_latency.py:sim_room`, which is its own implementation, not an import of the policy.
- **Tiny-model dual-rate checks are real.** Fast path byte-identical to single rate, slow finals == the offline masked
  forward, flush block-invariance: these are equality tests against independent references, with `len >= 2` guards.
- **No assertion-free tests.** I found no `assert True`, self-comparison or assert-free test (AST scan). The three
  tests without an `assert` call `validate` / `onnx.checker` / a strict YAML loader, which raise on failure.
