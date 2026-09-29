# Issues fixed (2026-09-28 pass)

Running log of the "fix every known open issue and finish the unfinished evaluations" pass. One row per item:
what was wrong, what changed, the test that pins it, the commit.

## A. CLEANUP_TODO.md "Suspicious" and "Ruff findings"

| Item | What was wrong | What changed | Test | Commit |
|---|---|---|---|---|
| `datasets/librispeech.py` `_fetch_range` | a non-206 answer raised `RuntimeError` *inside* the `try`, the bare `except Exception` swallowed it and retried 500 x 2 s (~17 min) before failing | `_NoRange(RuntimeError)` re-raised past the retry `except` (fail fast); transient errors still resume | `test_issues_fixed.py::test_fetch_range_fails_fast_when_range_is_ignored` (one request, not 50) | 7b477b4 |
| `scripts/benchmark_teachers.py` `PeakMemory` | `ru_maxrss` divided by 2**30 on every platform; it is KiB on Linux (bytes on macOS) | bytes on darwin, `* 1024` elsewhere (same rule as bench_ondevice) | `test_peak_memory_cpu_is_bytes` | 7b477b4 |
| `scripts/bench_ondevice.py` `profile()` | backend mapping missed `trace-int8`/`bf16`/`fp16`/`coreml-*` (they hit `StreamingRuntime` with an unknown backend); the module-split guard was `be == "onnx"` although the split patches the *eager* submodules, so trace/compile silently profiled the eager encoder and `rt._fn = ce` never restored the traced step; `bench()` reported the upper-median repeat for even `--repeats` while `first_result_ms` used `statistics.median` | same mapping as `bench()`; onnx/coreml return after the coarse stages, trace/compile go straight to the op table with the original step function, eager keeps the split and restores `orig[1]`; one median (`statistics.median`, and the repeat closest to it for the per-field numbers) | ruff + `--help` smoke (no torch-backend CI on this Mac); logic-only change | 7b477b4 |
| `examples/livekit_offline_demo.py` `load_ref` | read `<wav>.json`, but `pipecat_local_demo.py prepare` writes `<name>.ref.json` -> the demo always scored without a reference | `.ref.json` first, legacy `.json` fallback; usage text updated. `--policy` was already passed through (`run_agent_session(..., a.policy, ...)`, line 444) - verified, no change | `test_livekit_demo_load_ref_reads_ref_json` | 7b477b4 |
| ws port 8765 vs 8791 | code defaults 8765 everywhere, INTEGRATION.md said "use 8791" without saying 8765 is the default | kept **8765** as the default (server, `stream_client`, both adapters, both examples, `AUDIOFORGE_URL`); INTEGRATION.md now says so and explains 8791 as the explicit override on this Mac (port held by Cursor) | docs only | 7b477b4 |
| `scripts/turn_error_analysis.py` | `analyze` reloaded the turns with its own `--n` and never compared it to `scores_meta.json` -> silent per-turn misalignment; `set_num_threads` called in `main()` and again in stage 1 | analyze checks `meta["n"]` and every score array against the loaded turn count and exits with the fix; the stage-1 call removed | `test_turn_error_analysis_refuses_a_score_file_for_another_n` | 7b477b4 |
| `datasets/icsi.py` md5 registry | `reg[m]` read outside the lock while other workers write it (benign race) | the whole read (`known`, `seen`) under the lock; `stat()` once | existing `test_icsi.py` | 7b477b4 |
| `model.py` `StreamingSession` CTC | dedup state via `getattr(self, "_prev", -1)` | `self._prev = -1` in `__init__` | `test_streaming_session_ctc_prev_is_an_init_attribute` | 7b477b4 |
| `heads/turn.py` docstring | still "TurnHead v2" | "TurnHead (v2 core, v3 duration/activity inputs, v5 mode concat)" | - | 7b477b4 |
| `.gitignore` | `runs/*.log` not ignored (7 untracked strays) | `runs/*.log` ignored; the 25 logs committed earlier stay tracked | `git status` clean of `runs/*.log` | 7b477b4 |
| percentile helpers x3 | `serve.py`, `scripts/stream_client.py`, `datasets/ami.py` each had a `_pct` with different rounding / empty semantics | `audioforge.metrics.pct(xs, q, nd, empty)` and `pct_dict`; the three (and `hybrid_asr.py`'s) are one-line wrappers keeping their old semantics | `test_pct_semantics_cover_the_three_old_helpers` | 7b477b4 |
| ruff F/E9 (52 findings) | unused imports/variables, an f-string without placeholders, `types` shadowed by a loop variable in `serve.py`, `to_device` redefined in `eval_stage1.py` | all fixed (35 auto, 17 by hand: dead assignments removed, `types` -> `tys`, `# noqa: F811` for the pytest fixture re-imported by name in `test_icsi_tracks.py`) | `ruff check --select F,E9 .` -> "All checks passed" | 7b477b4 |
| datasets `_root` / `prepare_*` duplication | three identical one-line `_root` with different `DEFAULT_ROOT`, two ~70-line prepare mains | **not consolidated** (judged not safe/worth it): the mains differ in 6 of 35 lines (AMI `fetch_split_lists`; ICSI `--all`, `WAV_BYTES` sizes, default stats path) and a shared module would need more parameters than the lines it saves; the `_root` one-liners are the per-module default binding | - | - |

Full suite after A: 674 passed, 5 skipped (`pytest -q`).

## B. Unfinished evaluations

| Item | What was unfinished | What was done | Test / evidence | Commit |
|---|---|---|---|---|
| B.2 dual lookahead WER | `serve --asr-lookahead` shipped without a WER measurement of the [70,13] pass | `hybrid_asr.py prep_icsi` (ICSI-200 dev segments, the AMI protocol) + `lookahead`: the served model at [70,0] / [70,1] / [70,13] on AMI-200, ICSI-200, LibriSpeech-200, paired bootstraps -> `runs/hybrid_asr.json["english"]`. [70,13] - [70,1]: **-1.41 [-2.48, -0.32]** AMI, **-2.50 [-3.80, -1.32]** ICSI, **-0.37 [-0.64, -0.11]** LibriSpeech (WER points); [70,0] - [70,1]: +1.25 / +2.58 / +0.19. Tables in `research/HYBRID_ASR.md` section 3 | `tests/test_hybrid_asr.py::test_lookahead_heads_bit_identical_and_text_partition` (every frame / turn_end / partial / streaming final identical with and without the flag; 4 lookahead tests pass) | b4a085e |
| B.1 TS-VAD evaluation | `research/IMPROVE_115M.md` had the AMI frame rows only; no ICSI frame rows, no turn-level rows | ICSI frame rows (A.1) and the eot-bench v2 turn rows with the TS-VAD track feeding the served trail6 head (A.2), AMI dev 974 + ICSI held-out 1312, cross-fitted <= 5 % FC, CIs: missed turn ends at 6 s **61.9 -> 39.3 %** (AMI, shipped `hybrid` -> TS-VAD-fed hybrid) and **68.5 -> 20.4 %** (ICSI); `hybrid_dyn` 34.2 / 18.7 %. All intermediate files were rebuilt on the SSD scratch after the reboot (`scripts/research/tsvad.py winfeats/vprints/bind/scores/report`, gated, <= 540 s calls) | `tests/test_tsvad.py` (7); `runs/improve_115m.json["turn_bench"]` | 89ba05a |
| B.1b paired CIs vs the Sortformer rows | the per-turn outcomes of the Sortformer-column systems (2026-09-27) were in the wiped scratch | Sortformer tracks (AMI 974 + ICSI 1312 windows), Silero, trail6 reference scores and TitaNet column embeddings are being rebuilt by the same chain (`eval_stage1.py --bench v2`, `bench_turn_icsi.py`); the report then adds the paired `tsvad - {vp_spk, vp_titanet, causal_dominant, oracle}` deltas. See the hand-back note for where it stopped | - | pending |
| B.3 transcript-text LID / FLEURS-17 | `hybrid_asr.json["lid_text"]` empty in the report; 7 of 17 FLEURS languages untranscribed | FLEURS-17 TDT v3 rows with the sample size stated (150 for the 10 supported languages, **30** for pl, uk, tr, fa, hi, zh, ja: `--n-lang 30`, the report merges per language and keeps the larger-n row); `lid_text`: langid.py on the transcript, **100 % (1260/1260)** on the supported set for full utterances, **91.7 % (275/300)** at 2 s, 0 % for every unsupported language (AmberNet on the same utterances 99.9 / 95.1 %). `research/HYBRID_ASR.md` section 5 | `tests/test_hybrid_asr.py` (decoder / import tests; scoring is data) | 52c3422 |
| B.4 meeting-ASR decoder adaptation | recipe written, never run; the first attempt built a 21 GB LibriSpeech float32 cache on the internal disk and helped crash the machine | `research/recipes/asr_meeting_decoder_adapt.yaml` caps the anchor at `max_sec 12` so it reads the *existing* `train-clean-100.1-12s` cache on the SSD (3.8 GB; `data/librispeech` is now a symlink there; nothing decoded anew). Run: 500 steps head-only (RNNT pred + joint, 5.33M params), 380 s on MPS, one gated call; wer_gate on LibriSpeech never fired. **AMI-200 24.43 -> 22.50 (paired -1.93 [-2.77, -1.24])**, ICSI-200 -0.12 [-0.78, +0.56], LibriSpeech-200 +0.04 [-0.09, +0.18]; 748/759 tensors bit-identical to the served model (only `heads.rnnt.pred/joint` changed). `research/IMPROVE_115M.md` Part B | `hybrid_asr.py adapt_report` -> `runs/hybrid_asr.json["adapt"]`; tensor diff | d92db12 |

## C. Machine / scratch changes made in this pass

- `scripts/research/{tsvad,hybrid_asr,bench_turn_baselines}.py`: `SCRATCH` is `$AUDIOFORGE_SCRATCH` or
  `/Volumes/ExternalSSD/nvidia-audio-models/scratch` (was the session scratchpad under `/private/tmp`, wiped by the
  reboot); `bench_turn_baselines.SILERO_V5` reads `data/silero/silero_vad_v5.onnx` (the scratch copy is gone).
- exFAT caveat: 3 of 974 AMI window-feature `.npy` files written while the machine was under pressure came back as
  zero-filled headers (numpy reads them as "pickled data"); `tsvad.py` was rerun for those keys after a scan
  (`np.load` every file). Worth a checksum/verify step if the SSD scratch is kept.
- `hybrid_asr.py run --n-lang N` / `report --n-lang N --fleurs-only`: per-language subsets with the size recorded
  (`"subset"`), previous larger-n rows kept.
- `tsvad.py bind --bindings ...` only computes the requested bindings (the Sortformer track is loaded only for `vp_*`);
  `report` skips a reference whose stored scores are absent instead of failing.

## D. Full suite at the end of the pass

`pytest -q` (one gated run, 2026-09-28 03:10-03:25, machine shared with other agents' jobs): **743 tests, 736 passed,
6 skipped, 1 failed**. The failure is `tests/test_bulletproof.py::test_fast_client_backpressure_slow_client_and_fairness`
(the bulletproof pass's untracked test file, a timing-based fairness assertion under load). Every test of this pass
(`test_issues_fixed.py`, `test_hybrid_asr.py`, `test_tsvad.py`, `test_serve_shipped.py`, `test_streaming_diar.py`) passes.
