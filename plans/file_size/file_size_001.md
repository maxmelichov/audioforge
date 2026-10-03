# File size: plan to bring product files under 700 lines

2026-10-03, main at 4895faf. Rule (AGENTS.md "File size"): 700 lines max; ask before splitting; split by
responsibility into halves, keep the public API where callers expect it, move tests with their code.
Read-only survey: nothing here is applied yet.

## Summary (approve in one line)

15 product files are over 700 lines: 11 in `audioforge/`, 4 in `tests/`. Every one has a clean split
by responsibility. Every old import path keeps working: the moved public names are re-exported, or
reached through a lazy module `__getattr__` where a plain re-export would create an import cycle. After
the splits the largest product file is about 650 lines (`server/session.py`).

- **Reply "approve all product splits"** to do all 15 in the order below.
  **Or reply "approve these N: ..."** with a list of files.
- **Order** (3 waves; within a wave, one agent per row, no file shared between agents):
  1. Wave 1, files nobody else is editing:
     - (A) `integrations/livekit.py` + `integrations/pipecat.py`, with their two test files
     - (B) `conversation.py`, `streaming_diar.py`, `model.py`
     - (C) `nemo_import.py`
  2. Wave 2, the server, which is the biggest and riskiest:
     - (D) `serve.py` + `tests/test_serve.py` + `tests/test_bulletproof.py`, done as one change
  3. Wave 3, files other agents are editing right now. These start only after those agents have
     committed and released them:
     - (E) `train.py`
     - (F) `datasets/dyadic.py` + `datasets/ami.py`
     - (C, again) `heads/turn.py`
- **Gate after each wave:** the full fast test suite passes unchanged, with no test edits except the
  moved tests. Then one commit per file group.
- **ETA** (AGENTS.md formula): about 6,400 lines move, 5 agents, 3 stages.
  6,400 × 40 / (6,000 × 5) ≈ 9 min, plus 3 × 2 min = **about 15 min**. Wave 3 also waits for
  the other agents to release `train.py`, `datasets/` and `heads/`.
- **Not proposed:** research scripts, research docs, `demo/archive` and the two large docs.
  Section 3 proposes what the rule should mean for them. You decide.

| # | File (lines now) | Split into (est. lines after) | Main risk |
|---|---|---|---|
| 1 | audioforge/serve.py (2144) | serve.py facade ~240; server/engine.py ~545; server/session.py ~645; server/session_turns.py ~270; server/session_finals.py ~275; server/connection.py ~380 | patched constants such as `S.SHED_RTF` (see the table) |
| 2 | integrations/livekit.py (980) | livekit.py ~600; livekit_frontend.py ~420 | none (no cycle) |
| 3 | heads/turn.py (964) | turn.py ~575; turn_inputs.py ~305; turn_eval.py ~165 | patched `greedy_align` / `_diar_name` (both keep working) |
| 4 | conversation.py (907) | conversation.py ~560; eot_bench.py ~380 | none |
| 5 | streaming_diar.py (899) | streaming_diar.py ~560; streaming_diar_train.py ~380 | `python -m audioforge.streaming_diar train` must keep working |
| 6 | nemo_import.py (898) | nemo_import.py ~615; nemotron_diar.py ~310 | lazy re-export of 3 names |
| 7 | train.py (885) | train.py ~590; train_data.py ~200; train_codec_llm.py ~140 | other agents are editing it now |
| 8 | datasets/dyadic.py (810) | dyadic.py ~420; dyadic_corpora.py ~430 | other agents are editing it now |
| 9 | datasets/ami.py (762) | ami.py ~565; ami_labels.py ~230 | patched `ami._root` / `DEFAULT_ROOT` stay put |
| 10 | integrations/pipecat.py (753) | pipecat.py ~410; pipecat_turn.py ~375 | none |
| 11 | model.py (721) | model.py ~570; streaming.py ~170 | none |
| 12 | tests/test_bulletproof.py (1193) | test_bulletproof.py ~610; test_bulletproof_server.py ~610 | none |
| 13 | tests/test_pipecat_integration.py (797) | test_pipecat_integration.py ~520; test_pipecat_turn.py ~310 | none |
| 14 | tests/test_serve.py (790) | test_serve.py ~520; test_server_parts.py ~300 | about 10 test files load test_serve.py as a helper module, so the test doubles stay in it |
| 15 | tests/test_livekit_integration.py (770) | test_livekit_integration.py ~590; test_livekit_mappers.py ~205 | none |

## 1. Inventory (plans/file_size/file_size_001.py)

`uv run plans/file_size/file_size_001.py` lists every tracked source or doc file over 700 lines
(`git ls-files`, then a line count), grouped by area. Output at 4895faf:

```
## product package (audioforge/): 11 file(s) over 700
    2144  audioforge/serve.py
     980  audioforge/integrations/livekit.py
     964  audioforge/heads/turn.py
     907  audioforge/conversation.py
     899  audioforge/streaming_diar.py
     898  audioforge/nemo_import.py
     885  audioforge/train.py
     810  audioforge/datasets/dyadic.py
     762  audioforge/datasets/ami.py
     753  audioforge/integrations/pipecat.py
     721  audioforge/model.py

## tests/: 4 file(s) over 700
    1193  tests/test_bulletproof.py
     797  tests/test_pipecat_integration.py
     790  tests/test_serve.py
     770  tests/test_livekit_integration.py

## scripts/research/: 19 file(s) over 700
    3419  scripts/research/core_0p6b_heads.py
    2101  scripts/research/core_0p6b_turn.py
    1935  scripts/research/turn_v5.py
    1656  scripts/research/eval_stage1.py
    1525  scripts/research/fixall.py
    1503  scripts/research/final_compare.py
    1392  scripts/research/e2e_final.py
    1297  scripts/research/tsvad.py
    1238  scripts/research/turn_data.py
    1130  scripts/research/contamination_probe.py
    1126  scripts/research/lid_fix.py
    1108  scripts/research/eot_latency.py
     946  scripts/research/turn_v4.py
     945  scripts/research/hybrid_asr.py
     928  scripts/research/bench_turnbench_latency.py
     795  scripts/research/tswer.py
     747  scripts/research/single_model_distill/single_distill.py
     742  scripts/research/lid.py
     728  scripts/research/bargein.py

## scripts/ (top): 0 file(s) over 700

## research/ docs: 7 file(s) over 700
    3094  research/FINAL_REPORT.html
    2358  research/archive/raw/cards/canary-1b-v2.md
    1341  research/archive/raw/cards/parakeet-tdt-0.6b-v3.md
    1011  research/FINAL_REPORT.md
     811  research/archive/DYADIC.md
     810  research/archive/EOT_BENCH_V2.md
     763  research/archive/raw/cards/nemotron-3.5-asr-streaming-0.6b.md

## demo/archive: 3 file(s) over 700
    1206  demo/archive/render.py
    1199  demo/archive/render_v2.py
     821  demo/archive/render_v4.py

## other: 2 file(s) over 700
     821  docs/CONFIGURATION.md
     707  docs/UPSTREAM_PRS.md

TOTAL: 46 tracked file(s) over 700 lines
```

## 2. Product splits, one table per file

Conventions used in every table:

- "Re-export" means the old module imports the moved names back, so `from audioforge.X import Y` keeps
  working everywhere: scripts, tests and docs.
- Where the new module has to import from the old one, a plain re-export would create an import cycle.
  Those cases use a lazy re-export, a module-level `__getattr__` (Python's PEP 562).
- Line counts are estimates: the moved range, plus about 20–40 lines of imports, plus the module docstring.
- Monkeypatch facts come from grepping every `monkeypatch.setattr` / `X.NAME =` in `tests/` and
  `scripts/`.

### 2.1 audioforge/serve.py (2144 lines)

The `audioforge/server/` package already exists and holds constants, protocol, policies, binding,
streams and cli. serve.py still does four jobs:

1. the engine: lines 236–736 (`gpu_available`, `model_presets`, `vad_head_params`, `Engine`);
2. the per-connection `Session`: lines 737–1794, about 1,060 lines, over the limit on its own;
3. the WebSocket connection loop: lines 1795–2144;
4. 235 lines of docstring, re-exports and `__all__`.

`Session` has three responsibilities of its own:

- how a turn end is decided (the vad_head policy, hints, Silero/hybrid events);
- what each final carries (the final-ASR jobs, the lookahead finals, speaker attribution);
- the processing loop itself.

These are split as mixins, so every `self.` attribute and method stays exactly where it is at runtime.
This is not a line-count cut: each mixin is one job. A collaborator-object refactor would be cleaner, but
it changes behaviour paths and is not proposed.

| | |
|---|---|
| Responsibilities found | engine (models, worker threads, global shedding, health); session core (state, config, notices, processing, shedding, stats); turn decision; finals; WebSocket loop |
| → `audioforge/server/engine.py` (~545) | `gpu_available`, `model_presets`, `vad_head_params`, `Engine` |
| → `audioforge/server/session_turns.py` (~270) | `SessionTurnsMixin`: `_vad_head_policy`, `_vad_head_theta`, `_seg_model`, `_seg2_model`, `_energy`, `_feed_energy`, `ST_KEEP_S`, `_st_audio`, `_ready_sample`, `_hint_update`, `_vad_head_update`, `_tsvad_theta`, `_act_tsvad`, `_cut`, `_fires`, `_p_at`, `_hybrid`, `_silero_events`, `_turn_end` |
| → `audioforge/server/session_finals.py` (~275) | `SessionFinalsMixin`: `_attribute`, `_embed`, `_emit_final`, `_final_job`, `_lookahead_due`, `_drop_lookahead`, `_lookahead_emit`, `flush_slow`, `take_final_jobs`, `run_final_jobs_sync`, `complete_final`, `fallback_final` |
| → `audioforge/server/session.py` (~645) | `class Session(SessionTurnsMixin, SessionFinalsMixin)`: `__init__`, `_notice`, `take_notices`, `apply_config`, `arm_enrollment`, the frame/time helpers (`t`, `_latest_row`, `_seg_text`, `_text_upto`, `_frame_audio`, `_act`, `_asr_ready_t`, `_diar_ready_t`, `_asr_frames_at`), `_sanitize`, `recent_rtf`, `_shed_level`, `process`, `_count_only`, `finish`, `stats` |
| → `audioforge/server/connection.py` (~380) | `_Conn`, `_log`, `handle`, `_SessionFailed`, `_health_request`, `serve` |
| Stays in `audioforge/serve.py` (~240) | the module docstring (with the in-process usage example) and every current re-export. It adds `Engine`, `Session`, `handle`, `serve`, `gpu_available`, `model_presets`, `vad_head_params`, so `__all__` and `audioforge.serve.X` are unchanged. The `python -m audioforge.serve` entry point stays. `server/cli.py` keeps its lazy `from ..serve import Engine` / `serve`. |
| Tests that move | see 2.12 and 2.14: the connection, engine-degradation and client-adapter tests go to `tests/test_bulletproof_server.py`; the protocol, policy and enrollment unit tests go to `tests/test_server_parts.py` |
| Risk 1: patched module limits | serve.py documents that `SHED_RTF`, `MAX_SEGMENT_S` and `FINAL_ASR_*` "may be overridden on this module". These are set on the serve module: by tests/test_bulletproof.py (`S.SHED_RTF`, `S.FINAL_ASR_TIMEOUT_S`, `S.FINAL_ASR_RESTART_S`, `S.MAX_SEGMENT_S`), tests/test_diarization_fix.py (`S.SHED_RTF`), scripts/research/diar_fix.py (`S.SHED_RTF`) and scripts/research/single_model.py (`S.DYN_T0`, `S.DYN_A`). Once the code lives in `server/*`, a plain module global would ignore these overrides. **Fix:** the moved code reads the overridable names (`SHED_RTF`, `MAX_SEGMENT_S`, `FINAL_ASR_TIMEOUT_S`, `FINAL_ASR_RESTART_S`, `FINAL_ASR_FAILS_BEFORE_RESTART`, `DYN_T0`, `DYN_A`, `DYN_TMIN`, `DYN_OFFSET`) at call time through a 5-line `server.constants.live(name)`. It returns `sys.modules["audioforge.serve"].NAME` when serve is loaded, else the constant. The contract, the tests and the scripts stay unchanged. The existing shedding, segment-cap and final-ASR tests prove it. |
| Risk 2: lazy relative imports | about 30 lazy relative imports inside methods (`from .lid`, `.tsvad_stream`, `.speaker_registry`, `.streaming_diar`, `.final_asr`, `.enrollment`, `.nemo_import`, `.train`, `.server.smartturn` …) become `..x` / `.smartturn`. This is mechanical, but a missed one fails only on that code path. Grep `from \.` in the 5 new files, then run test_single_mode, test_lid, test_tsvad_stream and test_hybrid_asr. |
| Risk 3: other patches | `S.Engine.load` (test_serve_shipped) and `serve_mod.main` (test_single_mode) patch attributes of re-exported objects, so they still work. `ni.load_any` and `en.TitaNetEmbedder` are looked up lazily at call time, so they are unaffected. |
| Import cycles | none. engine → session is a one-way import, and connection imports engine and session; `server/*` never imports `audioforge.serve` at top level. |

### 2.2 audioforge/integrations/livekit.py (980 lines)

| | |
|---|---|
| Responsibilities found | (a) the one shared server session per agent session: options, PCM conversion, the WebSocket `_Link`, `AudioforgeFrontend`; (b) the three LiveKit adapters on top of it: STT, VAD, turn detector, and their event mappers |
| → `audioforge/integrations/livekit_frontend.py` (~420) | `SR`, `FRAME_S`, `DEFAULT_URL`, `DIAR_LAG_MAX_S`, `DIAR_LAG_MEAN_S`, `POLICIES`, `HYBRID_POLICIES`, `DEFAULT_EOT_THRESHOLD`, `AudioforgeOptions`, `_PcmConverter`, `_cat`, `_Link`, `AudioforgeFrontend`. The factory methods `stt()` / `vad()` / `turn_detector()` import their classes from `.livekit` at call time. |
| Stays in `livekit.py` (~600) | the module docstring (the user-facing mapping doc), `_SpeechMapper`, `AudioforgeSTT`, `AudioforgeSpeechStream`, `_VadMapper`, `AudioforgeVAD`, `AudioforgeVADStream`, `AudioforgeTurnDetector`, `AudioforgeTurnStream`, and a re-export of everything moved to livekit_frontend, so `from audioforge.integrations.livekit import AudioforgeFrontend, AudioforgeOptions, POLICIES` keeps working (tests, demo and docs use these) |
| Tests that move | see 2.15: the mapper unit tests go to `tests/test_livekit_mappers.py` |
| Risk | Low. livekit_frontend imports nothing from livekit at top level, so there is no cycle. tests/test_bulletproof.py `test_livekit_link_records_errors…` reaches `_Link`: re-export the private `_Link` too. No monkeypatches. |

### 2.3 audioforge/heads/turn.py (964 lines)

| | |
|---|---|
| Responsibilities found | (a) the `TurnHead` module itself (lines 369–820, about 450): forward, loss, decode, streaming step, `evaluate_model`; (b) the head's inputs and targets as pure tensor functions (lines 86–368, about 280): EOT and future and multi-horizon targets, duration counters, energy features, greedy and uniform token alignment, greedy frame decode; (c) the turn scoring and evaluation drivers (lines 821–964, about 145) |
| Why three, not two | moving only (b) leaves turn.py at about 690–720 lines, with no headroom. (c) is a separate job (whole-model scoring used by `conversation._main`, `model.py` and 8 research scripts). |
| → `audioforge/heads/turn_inputs.py` (~305) | `eot_targets`, `act_history`, `NEVER`, `duration_counters`, `log_clip`, `future_act_targets`, `FRAME_SAMPLES`, `ENERGY_FLOOR`, `frame_log_rms`, `causal_standardize`, `energy_features`, `horizon_edges`, `multi_horizon_targets`, `ALIGN_MAX_BYTES`, `align_chunk`, `greedy_align`, `uniform_align`, `token_counts`, `greedy_decode_frames`, `pad_tokens` |
| → `audioforge/heads/turn_eval.py` (~165) | `_flat`, `_onset_end`, `_diar_name`, `primary_column`, `streaming_diar_act`, `streaming_diar_probs`, `decoded_text_state`, `turn_scores` |
| Stays in `heads/turn.py` (~575) | docstring, `MODES`, `ALIGNS`, `TurnStreamState`, `TurnHead`, and re-exports of every name above. model.py's `from .heads.turn import _diar_name, primary_column, streaming_diar_act, streaming_diar_probs`, conversation's `turn_scores` and the scripts' `_diar_name` all keep working. |
| Tests that move | none required: test_turn.py, test_turn_v3.py, test_turn_v5.py, test_diag_turn.py and test_mem.py import through `heads.turn`. Their imports can optionally be pointed at the new modules later. |
| Risk: monkeypatches | `turn_mod.greedy_align` (test_diag_turn) **keeps working**, because `TurnHead.align` stays in turn.py and looks up `greedy_align` there, as a turn.py global. `turn._diar_name` (test_trail6, test_track_flush, test_hyb6, test_icsi_tracks) **keeps working**, because those tests patch it for scripts/research/make_sortformer_tracks.py, which imports it lazily from `audioforge.heads.turn` at call time. Inside turn_eval, `streaming_diar_act` would use turn_eval's own `_diar_name`. No test patches that path. |
| Import cycles | none. turn_eval imports from turn_inputs only; turn.py imports both; neither imports turn. |
| Note | other agents are editing `heads/` right now, so this is wave 3. |

### 2.4 audioforge/conversation.py (907 lines)

| | |
|---|---|
| Responsibilities found | (a) the synthetic multi-party conversation generator, its dataset and its collate (lines 1–511); (b) the EOT benchmark metrics: false cuts, latency, bootstrap CIs, OR-fusion, silence baselines (lines 512–870) |
| → `audioforge/eot_bench.py` (~380) | `silence_scores`, `_q`, `pause_runs`, `floor_stratum`, `eot_outcomes`, `outcome_metrics`, `bootstrap_ci`, `eot_bench`, `hybrid_fire_frame`, `_default_ths`, `_pause_max`, `eot_bench_or`, `pause_fire`, `or_outcomes`, `eot_outcomes_or`, `baselines` (imports only `.data.FRAME_SEC`) |
| Stays in `conversation.py` (~560) | generator, `TurnTakingConfig`, `PRESETS`, `ConversationDataset`, `conversation_dataset`, `event_stats`, `ConversationCollate`, the `_main` CLI (it needs both halves), and a plain re-export of all the names above, including the private `_q` that 3 scripts import |
| Tests that move | none required: about 31 files import the metrics as `from audioforge.conversation import …`, and the re-export keeps them all working |
| Risk | Low. eot_bench never imports conversation, so there is no cycle. No monkeypatches. |

### 2.5 audioforge/streaming_diar.py (899 lines)

| | |
|---|---|
| Responsibilities found | (a) streaming inference: `AOSCConfig`, `SORTFORMER_PRESETS`, cache rule, `SpeakerCache`, `_MelStream`, `_IncDecoder`, `StreamingDiarizer`, `StreamingSpeakerASR` (lines 1–542); (b) research training and evaluation: synthetic conversations (`compose_conversation`, `conversation_dataset`, `long_conversations`), the streaming-simulation loss (`teacher_cache`, `streaming_sim_loss`, `_TrainWrapper`, `train`), metrics (`cp_wer`, `slot_flips`, `evaluate`, `eval_sets`) and the `main` CLI (lines 543–899) |
| → `audioforge/streaming_diar_train.py` (~380) | all of (b) |
| Stays in `streaming_diar.py` (~560) | all of (a); `if __name__ == "__main__":` delegates to `streaming_diar_train.main`, so `python -m audioforge.streaming_diar train …` (research/recipes/streaming_sortformer.yaml) keeps working. A lazy `__getattr__` re-exports the (b) names, because streaming_diar_train imports `StreamingDiarizer` and friends from here and a plain re-export would be a cycle. |
| Tests that move | none required. tests/test_streaming_diar.py has 158 lines and uses `compose_conversation` and `slot_flips` as fixtures for the inference tests too, so it stays as it is. Pointing those two imports at `streaming_diar_train` turns the lazy re-export into a direct import. |
| Risk | Low. `sdm.StreamingDiarizer` (test_hyb6) is patched where it stays. The scripts import `StreamingDiarizer`, `SORTFORMER_PRESETS`, `AOSCConfig` and `_MelStream`, which all stay. |

### 2.6 audioforge/nemo_import.py (898 lines)

| | |
|---|---|
| Responsibilities found | (a) the .nemo reader plus FastConformer/Sortformer config and weight translation, and the entry points `import_nemo`, `import_titanet`, `import_ambernet`, `import_eou`, `load_any`, `main`; (b) Nemotron-3-Diarization, a different architecture with its own model classes, config translation, weight map and importer (lines 402–694) |
| → `audioforge/nemotron_diar.py` (~310) | `is_nemotron_diar`, `_rotate_half`, `_RoPEAttention`, `_PreLNBlock`, `FeatureStackingEncoder`, `Nemotron3DiarHead`, `top_k_columns`, `Nemotron3Diarizer`, `translate_nemotron_diar_config`, `SKIP_NEMOTRON`, `_NEMOTRON`, `map_nemotron_state_dict`, `_import_nemotron_diar`. It imports `_translate_preprocessor`, `check_frontend` and `license_of` from nemo_import, a one-way import. |
| Stays in `nemo_import.py` (~615) | (a), with `import_nemo` and `load_any` importing from `.nemotron_diar` inside the function. A lazy `__getattr__` serves `Nemotron3Diarizer`, `top_k_columns`, `is_nemotron_diar` (and the rest of the moved names), so `from audioforge.nemo_import import Nemotron3Diarizer` (tests/test_sortformer_import.py) keeps working. |
| Tests that move | the Nemotron tests in tests/test_sortformer_import.py could move to `tests/test_nemotron_diar.py`, but this is optional: that file is under 700 lines and the lazy re-export covers it |
| Risk | Low. `ni.load_any` (test_single_mode) stays in nemo_import. A cycle is avoided only by the lazy imports; a test asserting `import audioforge.nemotron_diar` works first, on its own, guards it. |

### 2.7 audioforge/train.py (885 lines)

| | |
|---|---|
| Responsibilities found | (a) the training loop: `pick_device`, `load_recipe`, `Trainer`, the .afm archive (`save_model`, `load_model`, `export_onnx`, `init_from_afm`), `build_model`, `run_recipe`; (b) recipe → examples: source dispatch, mixing, label derivation, mix collate, codec and tokenizer builders (lines 50–232); (c) two unrelated toy trainers, codec and speech-LLM (lines 759–885) |
| Why three | moving only (b) leaves about 710 lines. (c) is its own job, used only by `cli.py`'s `train-codec` / `train-speech-llm`. |
| → `audioforge/train_data.py` (~200) | `load_data`, `_load_source`, `MIX_META`, `derive_labels`, `MixedData`, `load_mix`, `MixCollate`, `build_codec`, `build_tokenizer`. The dataset imports stay lazy. This is kept out of `data.py` because PROJECT.md says the loader stays general purpose, and this layer is the recipe-shaped dispatch. |
| → `audioforge/train_codec_llm.py` (~140) | `train_codec`, `_train_mel_codec`, `F_l1`, `train_speech_llm` |
| Stays in `train.py` (~590) | (a), plus re-exports of everything in train_data. train_codec_llm imports `pick_device` from train, so its names are served by a lazy `__getattr__`, or cli.py's two lazy imports are pointed at the new module. |
| Tests that move | none required: `load_data`, `derive_labels`, `MixCollate` and `MIX_META` are imported through `audioforge.train` (8+8+1+1 sites) |
| Risk | `train.load_model` (patched in 4 tests) and `Trainer.evaluate` (test_train_fixes) stay in train.py. **Other agents are editing train.py now, so this is wave 3, after they commit.** |

### 2.8 audioforge/datasets/dyadic.py (810 lines)

| | |
|---|---|
| Responsibilities found | (a) reading the four raw corpora into per-party words and activity: roots and licences, resampling, the Behavior-SD / DailyTalk / oto / TurnBench readers, word interpolation, energy channel, Silero / VAD-head segments, `channel_check`, `build_labels`, `attach_asr_text` (lines 87–436 and 703–760); (b) the `Dyadic` dataset, built on `ami.AMI`, with agent-side streams, splits and the recipe hook |
| → `audioforge/datasets/dyadic_corpora.py` (~430) | `CORPORA`, the `*_REPO` / `*_ROOT` / `*_LICENSE_NOTE` constants, `ROOTS`, `TB_SCORER`, `SPEAKER_OFFSET`, `WORD_TIMING`, `ACTIVITY_SOURCES`, `VAD_HEAD_CKPT`, `_root`, `resample16k`, `mix_mono`, `list_ids`, `bsd_dirs`, `bsd_file`, `oto_index`, `_oto_member`, `tb_index`, `tb_row`, `tb_words`, `read_stereo`, `_tok_text`, `interpolate_words`, `bsd_words`, `energy_channel`, `dt_words`, `silero_model` (+ `_SILERO` cache), `vad_segments`, `vad_head_model` (+ `_VAD_HEAD`), `channel_check`, `build_labels`, `attach_asr_text` |
| Stays in `dyadic.py` (~420) | docstring, `AGENT_RULES`, `SPLIT_MOD`, `agent_end_stream`, `agent_end_before`, `first_voice_after`, `split_ids`, `slice_ids`, `conversation_duration`, `Dyadic`, `recipe_data`, and a plain re-export of every corpora name (scripts/research uses many of them) |
| Tests that move | none required: no dyadic test file is over the limit, and the re-exports cover all imports |
| Risk | No monkeypatches and no module-attribute assignments found. The corpora module never imports dyadic, so there is no cycle. **Wave 3, because datasets are being edited now.** |

### 2.9 audioforge/datasets/ami.py (762 lines)

| | |
|---|---|
| Responsibilities found | (a) download and split lists (lines 118–241); (b) annotation parsing and label construction, as pure functions from words to turns and frames (lines 242–449); (c) the `AMI` dataset, corpus stats, recipe hooks and `attach_ext_tracks` |
| → `audioforge/datasets/ami_labels.py` (~230) | `LABEL_DEFAULTS`, `MAX_RUN_SEC`, `normalize`, `parse_words_xml`, `speaker_names`, `global_speakers`, `meeting_words`, `activity`, `_runs`, `_turn`, `ACTION_DEFAULTS`, `action_transcript`, `window_action_transcript`, `speaker_turns`, `next_turn`, `frames`, `spk_matrix` |
| Stays in `ami.py` (~565) | constants and URLs, `DEFAULT_ROOT`, `DEFAULT_MEETINGS`, `_root`, the download functions, `AMI`, `corpus_stats`, `recipe_data`, `attach_ext_tracks`, and a plain re-export of all the label names (icsi.py, dyadic.py and scripts use `ami.speaker_turns` and friends) |
| Why labels, not download | `ami._root` and `ami.DEFAULT_ROOT` are monkeypatched (test_trail6), and the download code reads them. Moving download out would break those patches; the label functions are pure and touch neither. |
| Tests that move | none required |
| Risk | Low. `LABEL_DEFAULTS` stays the same dict object, so any in-place mutation is still shared. **Wave 3.** |

### 2.10 audioforge/integrations/pipecat.py (753 lines)

| | |
|---|---|
| Responsibilities found | (a) `AudioforgeSTTService`, which owns the WebSocket and turns server messages into Pipecat frames; (b) the turn and VAD side: the shared `AudioforgeHub` state (turn ends, hints, frame probabilities) and the analyzers and stop strategy that read it |
| → `audioforge/integrations/pipecat_turn.py` (~375) | `POLICIES`, `ENROLL_MODES`, `AGENT_END_MODES`, `FRAME_MS`, `NUM_SPKS`, `TurnEnd`, `TurnHint`, `AudioforgeHub`, `AudioforgeVADAnalyzer`, `AudioforgeTurnParams`, `AudioforgeTurnAnalyzer`, `AudioforgeEagerTurnStopStrategy` |
| Stays in `pipecat.py` (~410) | the module docstring (the user-facing mapping doc), `AudioforgeSTTService`, and a re-export of every pipecat_turn name. `from audioforge.integrations.pipecat import AudioforgeVADAnalyzer, AudioforgeHub, …` keeps working. |
| Why the hub goes with the analyzers | the STT service writes to the hub and the analyzers read from it. Putting the hub with the analyzers makes the import one-way (pipecat → pipecat_turn). Putting it with the STT service would make the analyzers import back into pipecat.py, a cycle. |
| Tests that move | see 2.13 |
| Risk | Low. No monkeypatches and no cycle. |

### 2.11 audioforge/model.py (721 lines)

| | |
|---|---|
| Responsibilities found | (a) `SpeechModel`, the batch model: head building, conditioning corruptions, forward, transcribe and analyze; (b) `StreamingSession`, the chunked streaming inference wrapper (lines 567–721) |
| → `audioforge/streaming.py` (~170) | `StreamingSession` (it uses only `StreamState` and the model instance it is given; it does not import model.py) |
| Stays in `model.py` (~570) | everything else, plus `from .streaming import StreamingSession` as a re-export. About 30 import sites (server/streams.py subclasses it, runtime, perf, scripts and tests) keep working. |
| Tests that move | none: tests/test_streaming.py already exists and keeps importing through `audioforge.model` |
| Risk | Low. `SpeechModel.head_reads_act` (test_turn_v5) is patched on a class that stays. |

### 2.12 tests/test_bulletproof.py (1193 lines)

| | |
|---|---|
| Responsibilities found | (a) session robustness, in process: audio input (NaN, amplitude, zero-length, odd blocks, resampling, PCM formats), determinism, memory, segment cap, speaker swaps, bursts, enrollment, policies (lines 261–528 and 823–878); (b) connection loop and engine degradations: malformed or binary messages, disconnects, backpressure, concurrency, health, shutdown, idle timeout, processing failure, ready handshake (lines 529–822), the final-ASR worker crash and lookahead drop, device fallback, diarizer NaN, load shedding, stats counters (lines 880–1105), and the Pipecat and LiveKit client adapters plus the stream client (lines 1106–1179) |
| → `tests/test_bulletproof_server.py` (~610) | all of (b). It reaches the shared helpers (`_run`, `_errors`, `_speech`, `_pattern`, `_energy_engine`, `_fresh`, `_Server`, `_stream`, the fake engines) by loading test_bulletproof.py by path. The suite already loads helper modules this way (`_h()` loads test_serve.py). |
| Stays (~610) | the helpers and (a) |
| Mirrors | (a) tests server/session.py; (b) tests server/connection.py and server/engine.py, plus the adapters |
| Risk | `monkeypatch.setattr(S, "SHED_RTF" / "FINAL_ASR_*")` keeps working through the `live()` accessor from 2.1. Pytest does not collect tests from a module loaded under another name. |

### 2.13 tests/test_pipecat_integration.py (797 lines)

| | |
|---|---|
| → `tests/test_pipecat_turn.py` (~310) | the tests of pipecat_turn.py: VAD analyzer units (`_vad_states`, the 3 vad tests), turn analyzer units (`_te`, `_fr`, the 3 turn tests), the local-pipeline turn end through the stop strategy (2 tests), and the hint prepare / cancel / mismatch pipeline tests with `hint_script` and `_hint_run`. `FakeAudioforge` and `run_frames` are loaded by path. |
| Stays (~520) | the fake server, the pipeline helpers, the STT service tests, the real-demo test, enrollment, policy acceptance, final-source and dual-rate tests |
| Risk | none (no monkeypatches) |

### 2.14 tests/test_serve.py (790 lines)

| | |
|---|---|
| Constraint | about 10 test files load test_serve.py by path as helper module `H`. They use `H._talky_asr_model`, `_diar_model`, `EnergyEngine`, `_speech_silence`, `_engine`, `_asr_model`, `_n_frames`, `_audio`, `EnergyDiarizer`, `_scene_audio`, `_scene_rows`, `ScriptedEngine`, `_with_server`, `_load_client`. **All test doubles stay in test_serve.py.** |
| → `tests/test_server_parts.py` (~300) | tests of the parts in `audioforge/server/`: protocol shape (lines 139–231: validate, debug keys, diarizer lag and preset plumbing, CLI choices, ready shape); turn policies on hand-made tracks with `_rows` and `_fire_frames` (lines 232–314); enrollment and voice-binder units (lines 662–759). It uses `H` for the doubles. |
| Stays (~520) | the doubles, the CPU fast-path test, the session (no-socket) tests, the socket end-to-end tests, the stream-client tests, the real-model test, and the e2e agent_end test |
| Risk | none (no monkeypatches in the moved tests) |

### 2.15 tests/test_livekit_integration.py (770 lines)

| | |
|---|---|
| → `tests/test_livekit_mappers.py` (~205) | the pure mapper tests (`_SpeechMapper`, `_VadMapper`, no socket): lines 192–261 and 658–770 |
| Stays (~590) | `FakeServer`, the helpers, and every stream, turn-detector, AgentSession, real-server and demo test |
| Risk | none |

## 3. Research scripts, research docs, demo archive, other docs: what should the rule mean?

| Group | Over 700 | What they are | Recommendation |
|---|---|---|---|
| scripts/research/ | 19 (728–3,419) | one experiment driver per write-up. Most are frozen. `core_0p6b_heads.py` (3,419) was touched 6 h ago. | **Exempt as research artifacts**, with two rules. (1) No growth: a script edited past 700 lines, or already over and being extended, is split at that point, after asking. (2) Scripts that tests load as libraries are de facto product code: `eval_stage1.py` 1,656 (9 test files), `tsvad.py` 1,297 (6), `eot_latency.py` 1,108 (3), `lid.py` 742 (3), `e2e_final.py`, `contamination_probe.py` and `bench_turnbench_latency.py` (1 each). Next time one is touched, promote the tested functions into `audioforge/` or split the script. Start with `eval_stage1.py` and `tsvad.py`. |
| research/*.md, *.html | 7 | `FINAL_REPORT.md` / `.html` (reports); `archive/raw/cards/*.md` (verbatim upstream model cards); `archive/DYADIC.md`, `EOT_BENCH_V2.md` (frozen write-ups) | **Exempt as documents.** Line limits help code review; a report or a copied model card does not get better when split. |
| demo/archive/ | 3 (`render.py` 1,206, `render_v2.py` 1,199, `render_v4.py` 821) | superseded render scripts, kept for provenance | **Exempt as archive.** Never edited. If one is ever revived, it moves out of archive and is split then. |
| other docs | `docs/CONFIGURATION.md` 821, `docs/UPSTREAM_PRS.md` 707 | CONFIGURATION.md's flag tables are generated from `server/cli.py` FLAGS (`scripts/dev/gen_config_doc.py`, checked by tests/test_config_doc.py); UPSTREAM_PRS.md is a status doc | **Exempt** (a generated reference and a document) |

Suggested wording for AGENTS.md (your call): *"700 lines max for code in `audioforge/`, `tests/`,
`scripts/` (top level) and `integrations/`. Research scripts, docs, generated files and `archive/`
folders are exempt, but a research script imported by tests counts as code, and no exempt script is
grown past 700 without asking."* With that wording, file_size_001.py could take an `--enforced` flag
that prints only the enforced groups. It is not added yet.
