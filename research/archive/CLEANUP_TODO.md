# Cleanup backlog (from the 2026-09-26 cleanup pass; apply once the files are no longer being edited)

**Status 2026-09-29 (release cleanup, CHANGELOG.md):** the ruff findings below are all fixed (`ruff check` with F, E9,
I, B passes); the percentile helpers now all call `audioforge.metrics.pct` and the peak-RSS helper exists once
(`audioforge.metrics.peak_rss_mb`); `.gitignore` covers `runs/*.log`. Still open: the dataset `_root` /
`recipe_data` duplication, the PCM framing loop shared by `scripts/stream_client.py` and the examples, and every item
under "Suspicious" except the last.

Done in that pass (490 tests before/after): ruff F/E9 fixes, shared `arrival_order()` and `frame_db()`
helpers, stale comments/docstrings, package `__init__` docstrings. Nothing deleted.

## Ruff findings in files that were off-limits at the time
- audioforge/conversation.py:59 unused `fields` import
- audioforge/train.py:5 unused `import io`
- audioforge/nemo_import.py:144 unused `d`
- audioforge/vocab.py:67 unused `missing, unexpected` (strict=True already raises)
- audioforge/serve.py:222 loop var `types` shadows the module import (F402)
- scripts/contamination_probe.py:420 `rng`, :556 `T` unused; :1042 f-string without placeholders
- scripts/eval_stage1.py:50 unused `to_device`
- audioforge/datasets/train_librispeech.py:22 unused `TEXT_HEADS`, `SpeechModel`
- tests: test_contamination.py:23, test_livekit_integration.py:26, test_recipes.py:8, test_mem.py:9 unused imports

## Duplicates to consolidate
- `_root(root)` and the recipe_data/subset/download/attach_ext_tracks structure across datasets/{ami,icsi,librispeech}.py; prepare_ami.py vs prepare_icsi.py main() near line-for-line -> shared `datasets/_meeting.py`
- three percentile helpers: serve.py:613, scripts/stream_client.py:46, datasets/ami.py:606 (rounding/empty semantics differ) -> `audioforge.metrics`
- PCM framing one-liner + paced-send loop duplicated in stream_client.py and the three examples; `frames_of` re-implemented in tests/test_livekit_integration.py:147

## Suspicious (possible bugs), not touched
- datasets/librispeech.py:76-91 RuntimeError raised inside try and swallowed -> retries ~17 min instead of failing fast
- scripts/benchmark_teachers.py:87-91 ru_maxrss is KiB on Linux but divided by 2**30 (bench_ondevice.py:261 does it right)
- scripts/bench_ondevice.py:449-479 module-split guard is `be == "onnx"` (comment says eager); trace/compile passes swap in eager `ce` and never restore; profile() does not map coreml-*/bf16/fp16; `med` vs statistics.median for even repeats
- examples/livekit_offline_demo.py:401 load_ref reads `<wav>.json` but prep writes `<name>.ref.json`; `--policy` ignored in agent-session mode (line 300)
- default ws://127.0.0.1:8765 in stream_client/livekit_agent_worker/livekit_offline_demo/pipecat_local_demo while INTEGRATION.md documents 8791
- scripts/turn_error_analysis.py:453 analyze reloads turns with its own --n; scores_meta.json n never compared -> silent misalignment; set_num_threads called twice
- datasets/icsi.py:318 `reg[m]` read outside its lock (benign)
- model.py:524 CTC state via getattr(self, "_prev", -1) instead of __init__ attribute
- heads/turn.py top docstring still says "TurnHead v2"
- .gitignore does not ignore runs/*.log (25 tracked, 7 untracked strays)
