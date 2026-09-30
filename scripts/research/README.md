# Research drivers

Scripts behind the numbers in `research/` and `runs/*.json`. Nobody needs them to install, download, serve or
benchmark audioforge (that is `audioforge-download` / `audioforge-serve` / `audioforge-bench`). Most need the
research extra (`pip install -e ".[research]"`), datasets under `$AUDIOFORGE_DATA` and checkpoints under `runs/`;
the note named in each docstring says how it was run. Run anything that loads a model through
`scripts/dev/gate.sh` (see `CONTRIBUTING.md`). The scripts import each other by module name: run them from the
repository root with `PYTHONPATH=.`, or through pytest (`tests/conftest.py` puts this directory on `sys.path`).

`distill_0p6b_to_115m/` is the self-contained GPU package for the 0.6B -> 115M distillation (its own README).

## Data preparation

| script | what it does |
|---|---|
| `prepare_ami.py` | Download, verify and cache an AMI Meeting Corpus subset (headset mix + manual annotations v1.6.2). |
| `prepare_icsi.py` | Download, verify, convert and cache an ICSI Meeting Corpus subset (headset mix + NXT core annotations v1.0). |
| `prepare_librispeech.py` | Download, verify, extract and cache LibriSpeech splits for audioforge. |
| `prepare_dyadic.py` | Download the dyadic slices (research/archive/DYADIC.md): Behavior-SD (CC BY 4.0, synthetic two-channel dialogues, one HF tar shard) and kyutai/DailyTalkContiguous (CC BY-SA 4.0, stereo acted dialogues with word timestamps). |
| `prepare_dyadic_asr.py` | Per-channel ASR transcripts of oto conversations (research/archive/DYADIC.md section 8): oto has no transcripts, but the turn head's text branch was trained on each window's PRIMARY words (AMI reference transcript, aligned through the ... |
| `lid_data.py` | Spoken-LID data (research/archive/LID.md): a small FLEURS subset, streamed (no full tarball on disk). |
| `make_sortformer_tracks.py` | Cache NVIDIA streaming Sortformer v2 activity tracks for every AMI / ICSI turn example (train + dev), so the turn head can TRAIN on real diarizer output (audioforge/datasets/ext_tracks.py; research/recipes/stage1_turn_on_sortformer.yaml). |
| `cache_titanet.py` | Cache frozen NVIDIA TitaNet-Large embeddings (the speaker-distillation teacher) per single-speaker segment. |

## Served model (maintainers)

| script | what it does |
|---|---|
| `make_served_model.py` | Build the served checkpoint: the trail6 turn model with the relational speaker head transplanted. |
| `bench_serve.py` | Reproducible CPU benchmark of the served system (audioforge.serve Engine / Session, in process, no websocket). |
| `run_all.sh` | Train every recipe at laptop scale and save models + logs under runs/. |

## Training and evaluation of the heads

| script | what it does |
|---|---|
| `eval_stage1.py` | Stage-1 real-speech evaluation of the fused front-end on AMI dev (unseen meetings / speakers). |
| `spk_head.py` | Speaker head: architecture vs supervision (research/archive/SPK_HEAD.md). |
| `spk_frame.py` | research/archive/IMPROVEMENTS.md section 3: the block-4 speaker head, crop-level ("frame-level") TitaNet distillation with every LibriSpeech train-clean-100 speaker + the relational loss, trained on cached block-4 features. |
| `spk_overlap.py` | Overlap robustness of speaker representations on AMI dev turn windows (research/archive/SPK_HEAD.md, "Overlap robustness"). |
| `layer_routing.py` | Layer routing for the diarization and turn heads (research/archive/LAYER_ROUTING.md). |
| `vad_layers.py` | Frame VAD per encoder block, and what a "gated encoder" would save (research/archive/VAD_LAYERS.md). |
| `tsvad.py` | Target-speaker VAD head on the frozen 115M streaming encoder (research/IMPROVE_115M.md, Part A). |
| `tsvad_enroll.py` | research/archive/IMPROVEMENTS.md section 4: how good must the TS-VAD voice print be? Print length and live prints. |
| `tsvad_turn.py` | research/archive/IMPROVEMENTS.md section 2: a predictive turn head trained on TS-VAD tracks (heads only, frozen encoder). |
| `diar_distill.py` | research/archive/DIARIZATION_FIX.md section 5: distil NVIDIA's diarizer into our own head on the frozen served encoder. |
| `lid.py` | Spoken language identification on the single-model front end vs dedicated LID models (research/archive/LID.md). |
| `turn_error_analysis.py` | Turn-head error analysis on AMI dev: why the v3 turn head works with oracle speaker activity and fails with NVIDIA's streaming Sortformer track, and why a silence timeout on the same noisy track does better ... |
| `contamination_probe.py` | Contamination probe: a speaker-conditioned streaming model gets a WRONG speaker assignment for a short window, then the correct one again. |
| `tsvad_chain.sh` | research/IMPROVE_115M.md A.2 / A.3: the whole TS-VAD turn-bench chain, resumable stage by stage (re-run this script; finished. |

## Benchmarks against dedicated models

| script | what it does |
|---|---|
| `bench_sd_baselines.py` | Dedicated VAD / diarization / speaker-verification models vs our heads, on the eval_stage1 AMI dev data. |
| `bench_turn_baselines.py` | Dedicated open turn-detection models vs our turn detection on eot-bench v2 (research/archive/BASELINES.md, "Turn detection"). |
| `bench_turn_icsi.py` | Held-out confirmation of the turn-detection rules on ICSI (research/archive/BASELINES.md, "Turn detection: ICSI held-out confirmation"). |
| `bench_turn_icsi_dyn.py` | Held-out ICSI confirmation of the dynamic-timeout policies (research/archive/COMPLETENESS.md §2.4; research/archive/BASELINES.md, "Dynamic timeout: ICSI held-out confirmation"). |
| `bench_turn_dyadic.py` | eot-bench v2 (research/archive/EOT_BENCH_V2.md) on the dyadic slices (research/archive/DYADIC.md), NO retraining: how the AMI-trained trail6 head and the timeout / Silero / Parakeet-EOU baselines behave on two-party audio where the "user" is ... |
| `bench_dyadic_heads.py` | Scoring of the dyadic-trained turn heads (research/archive/DYADIC.md section 8) against the AMI-trained trail6 head. |
| `bench_turnbench.py` | TurnBench dev (research/archive/OUTSIDE.md 2.3, research/archive/DYADIC.md) scored with the OFFICIAL MIT scorer (vendored unchanged in integrations/turnbench_scorer): our systems run CAUSALLY on the 38 two-channel conversations and commit ... |
| `bench_turnbench_cmp.py` | Our completeness head on TurnBench dev with the OFFICIAL scorer (research/archive/COMPLETENESS.md §3.4), built on scripts/research/bench_turnbench.py (its stages, event rules, scorer wrapper and operating-point rule are imported ... |
| `bench_turnbench_latency.py` | Latency of the TurnBench decision policy fitted on DYADIC data (research/archive/DYADIC.md section 7): the per-channel inputs of DYADIC.md section 4 (user-channel Silero v5 + Pipecat state machine; trail6 head on the mixed mono with ... |
| `bench_completeness.py` | smart-turn x our turn head: learned fusion on eot-bench v2 and the completeness head (research/archive/COMPLETENESS.md). |
| `bench_completeness_dyn.py` | Dynamic-timeout policies on eot-bench v2 (scripts/research/bench_completeness.py --stage dyntimeout; research/archive/COMPLETENESS.md §2.4). |
| `bench_completeness_fusion.py` | Fusion stage of scripts/research/bench_completeness.py (research/archive/COMPLETENESS.md §2): learned fusion of our trail6 head, smart-turn v3.2 and the two silence timeouts on eot-bench v2, cross-fitted by the v2 meeting folds, plus ... |
| `bench_yield_tokens.py` | Conversational action tokens <YIELD> / <HOLD> in the RNNT vocabulary, scored on eot-bench v2 (research/archive/YIELD_TOKENS.md). |
| `benchmark_teachers.py` | Benchmark NVIDIA teachers (and our students) on the first N LibriSpeech test-clean utterances. |
| `summarize_dyadic.py` | Collect runs/dyadic_bench_<corpus>.json (+ runs/turnbench_dev.json) into runs/dyadic_bench.json and print the markdown tables of research/archive/DYADIC.md. |
| `eot_frontier.py` | research/archive/IMPROVEMENTS.md section 2c: latency / false-cutoff frontier of the served turn head (the product knob). |

## End-to-end and live-server studies

| script | what it does |
|---|---|
| `e2e_final.py` | Final end-to-end comparison: our single-model front end vs the default local stacks of Pipecat and LiveKit. |
| `e2e_tsvad.py` | research/archive/IMPROVEMENTS.md section 1: the served TS-VAD turn path live, through the Pipecat adapter, on E2E_FINAL's clips. |
| `hybrid_asr.py` | Hybrid front end, offline part (research/archive/HYBRID_ASR.md): NVIDIA Parakeet-TDT 0.6B v3 as the per-turn final ASR. |
| `hybrid_live.py` | Hybrid front end, live part (research/archive/HYBRID_ASR.md): Pipecat / LiveKit sessions through the committed adapters against audioforge.serve with a second transcript source, on the clips of the final end-to-end comparison. |
| `hybrid_fast.py` | research/archive/IMPROVEMENTS.md section 2b: a fast path for the head-gated turn rule (the user's comparison with smart-turn's ~225 ms decision: 200 ms Silero stop + 25 ms inference). |
| `final_asr.py` | ASR on real meeting audio for the final report (research/FINAL_REPORT.md): WER + CPU RTF, same sets and normalisation. |
| `diar_fix.py` | Multi-speaker diarization in the served system: reproduce, diagnose, measure (research/archive/DIARIZATION_FIX.md). |
| `tsvad_serve_check.py` | research/archive/IMPROVEMENTS.md section 1: does the served --turn-input tsvad path reproduce the offline benchmark? |
| `core_0p6b.py` | research/archive/IMPROVEMENTS.md section 6: nemotron-speech-streaming-en-0.6b as the GPU streaming core (the user's request: 22.5 % AMI WER for the streaming partials is too high, production is GPU). |
| `enc0p6b.py` | nvidia/nemotron-speech-streaming-en-0.6b as a voice-agent front-end encoder (research/archive/ENC_0P6B.md). |

## Drafts and pilots (not served; kept because a note or test cites them)

| script | what it does |
|---|---|
| `uc_turn.py` | User-channel predictive turn head (research/archive/IMPROVEMENTS.md section 1): features, training, inference, TurnBench. |
| `uc_turn_head.py` | User-channel predictive turn head (research/archive/IMPROVEMENTS.md section 1). |
| `uc_stream.py` | Live two-channel predictive end-of-turn for one session (research/archive/IMPROVEMENTS.md section 1). |
| `frontier_codec_probe.py` | FRONTIER pilot 2: does the frozen 12.5 Hz streaming ASR encoder predict 12.5 Hz codec tokens frame-synchronously? (research/archive/FRONTIER.md, P2: the precondition for ASR-encoder-driven streaming speech-to-speech / voice conversion.) |
| `frontier_sa_captions.py` | FRONTIER pilot 1: speaker-attributed live captions from pieces we already serve (research/archive/FRONTIER.md, Part 2). |

## Reports

| script | what it does |
|---|---|
| `final_report_html.py` | research/FINAL_REPORT.md -> research/FINAL_REPORT.html: one self-contained file, identical content. |

## Archived

One-off drivers that no note, test or other script uses any more moved to [`scripts/archive/`](../archive/) (2026-09-30):

| script | what it did |
|---|---|
| `train_turn_ablation.sh` | Turn-taking ablation v3 (research/archive/TURN_ABLATION.md), strictly sequential, load-guarded (waits for load<8 before. |
| `run_v3_then_stage1.sh` | Sequential GPU chain: v3 turn ablation -> stage 1 (frozen NVIDIA encoder + our heads). One training at a time. |
| `bench_ondevice.py` | On-device streaming benchmark for the unified front-end (research/archive/ONDEVICE.md). |
| `run_dyadic_bench.sh` | Drive a resumable bench stage under the machine rules: one process at a time, <= 10 min each, a new process only. |
