# Tests fix wave: changes needed outside tests/ (2026-10-03)

Found while fixing plans/audit/tests_001.md. The tests owner did not edit package code; each item names its owner.

## 1. Trainer owner: seed Python's `random` in `run_recipe` (audioforge/train.py)

`tests/test_diag_turn.py::test_recipe_smoke_diagnostics_reported` is now seed-independent: it is parametrized over
`heads.turn.decoded_prob` 0.0 (asserts `turn_align_greedy_frac`, no `turn_decoded_frac`) and 1.0 (asserts
`turn_decoded_frac == 1.0`, no greedy frac), instead of one 0.5 draw over 3 steps (with seed 0 all three draws picked
"decoded", plans/trainer/parity.md). Checked with `random` seeded inside `run_recipe` (simulated: seed + 0, 1, 2):
both cases pass every time.

So `run_recipe` can now seed it, which the data-order rule asks for, e.g. next to the torch / numpy seeds:

    random.seed(cfg.get("seed", 0))

and the NOTE comment above them (train.py ~line 959) can go.

## 2. API owner: `audioforge.load()` fails in the public snapshot (no LID head)

`audioforge.load()` (115M, mode single) passes `lid="head"` from `MODES["single"]` to `Engine.load`, and
`lid.resolve_head("head")` raises `FileNotFoundError` when `lid_115m_v2.pt` is in neither the models dir, assets/
nor runs/. That is the public snapshot (`scripts/dev/build_public.sh` drops assets/lid_*.pt). The launcher
(`launch._single`) handles the same case: it prints "lid_115m_v2.pt not found, language ID is off" and drops --lid.
The 0.6B branch of `api.load` already does `find_head(...)` and passes None.

Fix: in `api.load`'s single branch, for the 115M as for the 0.6B, resolve the head with `find_head(lid_file,
models_dir)` when `opts.get("lid") == "head"` and `"lid" not in engine_options`, and pass None (with the same stderr
notice) if it is absent.

`tests/test_single_mode.py::test_snapshot_api_load_without_lid_head_turns_lid_off` pins this as
`xfail(strict=True, raises=FileNotFoundError)`. When the fix lands it XPASSes and fails the suite: then drop the
xfail marker.

## 3. Serve owner (minor): `--final-asr` latency_ms is taken after the submit

`audioforge/serve.py` `submit_finals()` builds `(j, engine.submit_final(j["audio"]), time.perf_counter())`: the
start time is read after the job was handed to the worker, so when the event loop is descheduled right after the
submit, `latency_ms` under-reports the pass (978 ms for a 1.2 s fake pass, seen under heavy swap in
`tests/test_hybrid_asr.py::test_final_asr_socket_protocol_and_non_blocking[process]`). Reading the clock before
`submit_final` makes `latency_ms >= pass time` hold always. The test now divides its 1150 ms bound by its SLACK
(3, or 1 with BULLETPROOF_STRICT_TIMING=1) until then.
