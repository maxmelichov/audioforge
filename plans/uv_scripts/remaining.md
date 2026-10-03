# uv standalone scripts: what is left

Date: 2026-10-03. Rule (AGENTS.md "Tooling"): Python scripts are `uv` standalone scripts, so `uv run <script>` works.

Done in the first pass (PEP 723 header, package as an editable path dependency, `python <script>` unchanged):

- `scripts/stream_client.py` (`audioforge[serve]`)
- `scripts/dev/gen_config_doc.py` (`audioforge`)
- `scripts/dev/render_compare.py` (new; `playwright`)
- `scripts/sweep_capacity.py` (already had one; owned by the heads agent)
- `plans/file_size/file_size_001.py`, `plans/repo_rules/repo_rules_001.py` (no dependencies)

Header used (the path is relative to the script's directory, `..` from `scripts/`, `../..` from `scripts/<dir>/`):

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["audioforge[research]"]   # or [serve] / plain, by what the script imports
#
# [tool.uv.sources]
# audioforge = { path = "../..", editable = true }
# ///
```

Scripts keep their `sys.path.insert(0, ROOT)` lines: they make `python <script>` work from the project venv, and the
sibling-driver imports in `scripts/research/` (`import eval_stage1 as E`) depend on the second insert.

## Left for a later pass

`scripts/archive/` is retired and kept as written (ruff excludes it too), so it is not on this list to convert:
`scripts/archive/bench_ondevice.py`.

`plans/trainer/parity_001.py` has a header without dependencies and is run with `.venv/bin/python` (it imports
`audioforge`); its owner decides whether to add the path dependency.

`scripts/research/` (each needs its imports checked: most use `[research]` extras such as transformers, pyannote,
speechbrain; modules imported by other drivers, such as `eval_stage1.py` or `uc_stream.py`, still get a header so
they run standalone):

- `scripts/research/asr_leaderboard.py`
- `scripts/research/bargein.py`
- `scripts/research/bench_completeness.py`
- `scripts/research/bench_completeness_dyn.py`
- `scripts/research/bench_completeness_fusion.py`
- `scripts/research/bench_dyadic_heads.py`
- `scripts/research/bench_sd_baselines.py`
- `scripts/research/bench_serve.py`
- `scripts/research/bench_turn_baselines.py`
- `scripts/research/bench_turn_dyadic.py`
- `scripts/research/bench_turn_icsi.py`
- `scripts/research/bench_turn_icsi_dyn.py`
- `scripts/research/bench_turnbench.py`
- `scripts/research/bench_turnbench_cmp.py`
- `scripts/research/bench_turnbench_latency.py`
- `scripts/research/bench_yield_tokens.py`
- `scripts/research/benchmark_teachers.py`
- `scripts/research/cache_titanet.py`
- `scripts/research/contamination_probe.py`
- `scripts/research/core_0p6b.py`
- `scripts/research/core_0p6b_heads.py`
- `scripts/research/core_0p6b_turn.py`
- `scripts/research/diar_distill.py`
- `scripts/research/diar_fix.py`
- `scripts/research/distill_0p6b_to_115m/distill.py`
- `scripts/research/dual_rate.py`
- `scripts/research/e2e_final.py`
- `scripts/research/e2e_tsvad.py`
- `scripts/research/enc0p6b.py`
- `scripts/research/eot_assistant.py`
- `scripts/research/eot_energy_gate.py`
- `scripts/research/eot_frontier.py`
- `scripts/research/eot_latency.py`
- `scripts/research/eval_stage1.py`
- `scripts/research/final_asr.py`
- `scripts/research/final_compare.py`
- `scripts/research/final_latency.py`
- `scripts/research/final_report_html.py`
- `scripts/research/fixall.py`
- `scripts/research/frontier_codec_probe.py`
- `scripts/research/frontier_sa_captions.py`
- `scripts/research/hybrid_asr.py`
- `scripts/research/hybrid_fast.py`
- `scripts/research/hybrid_live.py`
- `scripts/research/latency_budget.py`
- `scripts/research/layer_routing.py`
- `scripts/research/lid.py`
- `scripts/research/lid_data.py`
- `scripts/research/lid_fix.py`
- `scripts/research/make_served_model.py`
- `scripts/research/make_sortformer_tracks.py`
- `scripts/research/metrics_table.py`
- `scripts/research/mps_115m.py`
- `scripts/research/prepare_ami.py`
- `scripts/research/prepare_dyadic.py`
- `scripts/research/prepare_dyadic_asr.py`
- `scripts/research/prepare_icsi.py`
- `scripts/research/prepare_librispeech.py`
- `scripts/research/single_model.py`
- `scripts/research/single_model_distill/single_distill.py`
- `scripts/research/smartturn_audit.py`
- `scripts/research/spk_frame.py`
- `scripts/research/spk_head.py`
- `scripts/research/spk_overlap.py`
- `scripts/research/streams_cpu.py`
- `scripts/research/stt_latency.py`
- `scripts/research/summarize_dyadic.py`
- `scripts/research/tsvad.py`
- `scripts/research/tsvad_enroll.py`
- `scripts/research/tsvad_serve_check.py`
- `scripts/research/tsvad_turn.py`
- `scripts/research/tswer.py`
- `scripts/research/turn_data.py`
- `scripts/research/turn_error_analysis.py`
- `scripts/research/turn_hint.py`
- `scripts/research/turn_v4.py`
- `scripts/research/turn_v5.py`
- `scripts/research/uc_stream.py`
- `scripts/research/uc_turn.py`
- `scripts/research/uc_turn_head.py`
- `scripts/research/vad_auc.py`
- `scripts/research/vad_layers.py`
- `scripts/research/vad_single.py`
- `scripts/research/vad_tail.py`
