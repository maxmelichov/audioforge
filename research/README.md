# research/

Lab notes behind audioforge: model surveys, imports of NVIDIA checkpoints, benchmarks, ablations and negative results.
**Layout.** The current reports sit here at the top: [METRICS.md](METRICS.md), [FINAL_REPORT.md](FINAL_REPORT.md)
(+ `.html`), [SINGLE_MODEL.md](SINGLE_MODEL.md), [EOT_LATENCY.md](EOT_LATENCY.md), [TSWER.md](TSWER.md),
[VAD_SINGLE.md](VAD_SINGLE.md), [LATENCY_BUDGET.md](LATENCY_BUDGET.md), [IMPROVE_115M.md](IMPROVE_115M.md),
[MPS_115M.md](MPS_115M.md) (Mac GPU), [BARGEIN.md](BARGEIN.md) (barge-in eval set + baselines), [E2E_FINAL.md](E2E_FINAL.md), [EARLY_RESULTS.md](EARLY_RESULTS.md),
[ANALYSIS.md](ANALYSIS.md) and [VERIFICATION_2026-09-28.md](VERIFICATION_2026-09-28.md). Every other note (superseded,
negative or background) moved unchanged to [`archive/`](archive/) on 2026-09-30, with the raw Hugging Face pulls in
`archive/raw/`; the tables below index both, one line per doc. Training recipes are in [`recipes/`](recipes/README.md).

Each doc is a dated note, kept as written. When a later measurement supersedes a number, it is corrected in place
(marked "Corrected <date>") or listed in the corrections log in [EARLY_RESULTS.md](EARLY_RESULTS.md#corrections-log-2026-09-25);
otherwise the newer doc wins. **The numbers to quote are in [FINAL_COMPARE.md](FINAL_COMPARE.md)** (2026-10-03: both cores against
the best open models on public test splits) and its summary scorecard [METRICS.md](METRICS.md) (one definition per
metric). Numbers in older notes are often on dev splits (AMI / ICSI dev); they document why a default was chosen, not
the current result. For the history, start with [FINAL_REPORT.md](FINAL_REPORT.md) (the single-document final report, also rendered as
[FINAL_REPORT.html](FINAL_REPORT.html)), then the product comparison [E2E_FINAL.md](E2E_FINAL.md);
[INTERIM_SCORECARD.md](archive/INTERIM_SCORECARD.md) is the earlier snapshot it supersedes (its disagreements with the JSON
are listed in FINAL_REPORT Appendix B).

Scripts cited in these docs as `scripts/<name>.py` now live at `scripts/research/<name>.py` (moved 2026-09-28;
[`scripts/research/README.md`](../scripts/research/README.md) lists them), and recipes cited as `recipes/<name>.yaml`
live in [`recipes/`](recipes/README.md) here (moved 2026-09-29). Run a cited script from the repository root as
`PYTHONPATH=. python scripts/research/<name>.py ...`. The laptop-scale synthetic results, the first live-server
measurements and the 2026-09-25 corrections log, formerly in the top-level README, are in
[EARLY_RESULTS.md](EARLY_RESULTS.md).

Key numbers are copied from each doc. "FC" = false cutoffs; "open" = floor-open turn ends (nobody else speaks).
Dates come from the doc header or git history (first commit → last change where they differ).

## Overview and plans

| doc | what it answers | key number | date |
|---|---|---|---|
| [FINAL_COMPARE.md](FINAL_COMPARE.md) | both cores (heads v0.4, LID head v2) against the best open models on public test splits, every baseline re-run on the same audio; leakage-flagged rows in its Appendix A | AMI test WER 7.9 % (0.6B) vs 8.3 % Parakeet-TDT v3 (offline) and 10.7 % Whisper large-v3; assistant clips 97.7 % accuracy at 351 ms p50 (0.6B) vs 85.2 % LiveKit, 75.9 % Pipecat | 2026-10-03 |
| [CORE_0P6B.md](CORE_0P6B.md) | nemotron-speech-streaming-en-0.6b as a second served core (`--core 0.6b`), every head retrained, side by side with the 115M on the same benchmarks | test splits (FINAL_COMPARE.md), 115M → 0.6B: WER 16.1 → 7.9 % (AMI test), live calls 20.3 → 11.6 %; speaker EER 5.0 → 3.8 % (AMI test); AMI test turns (`balanced`) 15.5 / 36.0 → 10.0 / 33.5 % false interruptions / missed; 30 vs 97 ms per chunk on CPU, 4 vs 1 streams | 2026-10-01 |
| [LAYER_SWEEP_0P6B.md](LAYER_SWEEP_0P6B.md) | which of the 0.6B's 24 blocks each head should read (every block, the mix, the top-3 concat; held-out selection) | speaker / TS-VAD best at blocks 4-6 (moved from 11), v5 classifier mid (12), LID deep (18-20); VAD peaks differ by corpus (AMI 13-14, ICSI 2-4) | 2026-10-01 |
| [METRICS.md](METRICS.md) | the standard scorecard on the test splits, both cores: WER, partial / final latency, end-of-turn accuracy / latency, false interruptions, missed, VAD AUC / F1, tWER, speaker EER, LID, compute | partial latency 274 / 291 ms p50 (115M / 0.6B, MPS); AMI test `balanced` 15.5 / 10.0 % false interruptions vs 13.0 % LiveKit, 39.5 % Pipecat | 2026-09-29 → 2026-10-03 |
| [INTERIM_SCORECARD.md](archive/INTERIM_SCORECARD.md) | every committed, source-verified result in one place | ASR AMI dev WER 20.63 % ours (streaming @160ms) vs 14.14 % Parakeet-CTC 0.6B offline (+5.11 [+3.4, +7.1] pts); equal on LibriSpeech (2.27 % vs 1.63 %) | 2026-09-27 |
| [FINAL_REPORT.md](FINAL_REPORT.md) | single-document final report (supersedes INTERIM_SCORECARD): 14-row scorecard vs measured and published systems, per-task tables, product results, recommended stack, negative results, unfinished items, limitations | dev-era, superseded by FINAL_COMPARE.md: better than every streaming VAD and every dedicated turn detector on AMI / ICSI all-ends (hybrid 61.9 % vs Silero timeout 72.7 % missed, n=974, 6 s); worse at meeting ASR on AMI dev (FINAL_REPORT §1; TDT v3 per turn far ahead), speaker, diarization, LID | 2026-09-27 |
| [BRIEF.md](archive/BRIEF.md) | project goal, what exists, scorecard as of the brief (context for collaborators) | - (scorecard superseded by INTERIM_SCORECARD) | 2026-09-26 |
| [PLAN.md](archive/PLAN.md) | the claim to earn, risks, gated execution queue, findings log | - | 2026-09-25 (header); git 2026-09-25 → 2026-09-26 |
| [GAME_CHANGER.md](archive/GAME_CHANGER.md) | is a unified streaming voice-agent front end new? landscape and targets | verdict: gap is real but narrower than the thesis; the unshipped piece is speaker-aware end-of-turn in one model | 2026-09-25 |
| [DATA_PLAN.md](archive/DATA_PLAN.md) | which data would close each task gap, in what order, with gates | - (writing note, no model run) | 2026-09-26 |
| [IMPROVEMENTS.md](archive/IMPROVEMENTS.md) | ranked improvement candidates (user-channel predictive turn head first) | - (results, where they exist, are in IMPROVE_115M.md / HYBRID_ASR.md, drafts) | 2026-09-28 |
| [SINGLE_MODEL.md](SINGLE_MODEL.md) | can one 115M checkpoint run the whole front end for a known user (`--mode single`)? voice-print quality (A2); GPU plan for target-speaker + turn heads (part B) | meetings: best turn detector measured with a stored 5 s print (AMI 34.2 %, ICSI 18.7 % missed); two-party calls live: see A1; live prints lose 16-55 points | 2026-09-29 |
| [MPS_115M.md](MPS_115M.md) | the served 115M streaming engine on the Mac GPU (`--device mps`) vs CPU: same events? compute, streams, WER | same events and identical LS-200 decodes (2.29 %); ASR core 10.2 vs 17.6 ms per 160 ms chunk; full engine 28.7 vs 29.8 ms; 5 vs 4 real-time streams | 2026-09-30 |
| [EOT_LATENCY.md](EOT_LATENCY.md) | end-of-turn latency of single mode, measured the standard way; the shipped `vad_head` rule | see the note's "after the print fix" rows | 2026-09-30 |
| [EOT_ASSISTANT.md](EOT_ASSISTANT.md) | end of turn on assistant-directed speech (smart-turn v3.2-test clips) vs Pipecat smart-turn and LiveKit | ours 1218 / 1581 ms p50 / p95, 100 % false fire on incomplete clips, 41.1 % accuracy; Pipecat 211 ms, 69.7 %; LiveKit 547 ms, 72.7 % | 2026-09-30 |
| [LATENCY_BUDGET.md](LATENCY_BUDGET.md) | every delay on the live path except the turn rule's own wait | see the note | 2026-09-30 |
| [TSWER.md](TSWER.md) | target-speaker WER: how well single mode transcribes only the enrolled user | see the note | 2026-09-29 |
| [VAD_SINGLE.md](VAD_SINGLE.md) | a block-4 single-layer VAD head vs the served all-layer head | see the note | 2026-09-29 |
| [VOICE_GENDER.md](VOICE_GENDER.md) | an optional perceived voice-gender head (female / male voice probabilities, not gender identity) on both cores | LibriSpeech test-clean balanced accuracy 97.8 / 97.7 % at 2 s (115M / 0.6B; 40 speakers, CI ~94-100), FLEURS-17 test 96.3 / 96.7 % | 2026-10-03 |

## ASR and model import

Drafts that appeared during finalisation and are cited by code (FINAL_REPORT Appendix B, item 12):
[HYBRID_ASR.md](archive/HYBRID_ASR.md) (per-turn Parakeet-TDT v3 final ASR and the [70,13] lookahead pass; `runs/hybrid_asr.json`)
and [IMPROVE_115M.md](IMPROVE_115M.md) (TS-VAD head prototype and other 115M improvements; `runs/improve_115m.json`).


| doc | what it answers | key number | date |
|---|---|---|---|
| [ANALYSIS.md](ANALYSIS.md) | what NVIDIA's 149 audio models on Hugging Face have in common | 51 of 54 NVIDIA recognition-type models since 2023-06-01 use the same FastConformer encoder | 2026-09-25 |
| [TEACHERS.md](archive/TEACHERS.md) | which NVIDIA ASR checkpoints run on this Mac as teachers, and how well | parakeet-ctc-1.1b 1.64 % WER vs parakeet-ctc-0.6b 1.73 %, LibriSpeech test-clean first 200 utts | 2026-09-25 |
| [NEMO_IMPORT.md](archive/NEMO_IMPORT.md) | loading NeMo FastConformer / TDT weights without NeMo | 115M streaming hybrid 1.92 % RNNT WER at 1.04 s lookahead (2.29 % at 0.08 s); parakeet-ctc-0.6b 1.68 % vs 1.73 % for the transformers port, LibriSpeech test-clean 200 utts | 2026-09-25 → 2026-09-27 |
| [ENC_0P6B.md](archive/ENC_0P6B.md) | does NVIDIA's 0.6B streaming encoder carry more speaker / VAD information than the 115M? | negative: best untrained block within-meeting EER 32 % AMI / 24 % ICSI vs 33 % / 29 % for the 115M (n=200), at 739 vs 176 ms per 160 ms chunk | 2026-09-26 |
| [ONDEVICE.md](archive/ONDEVICE.md) | can the full pipeline run at RTF ≤ 0.3 on one CPU core? | yes: 111M model, 160 ms chunks, fp32, 1 thread, median RTF 0.16 (quiet) / 0.19 (load 15) | 2026-09-25 |

## Speaker, VAD and diarization

| doc | what it answers | key number | date |
|---|---|---|---|
| [SORTFORMER_IMPORT.md](archive/SORTFORMER_IMPORT.md) | NVIDIA Streaming Sortformer v2 (and v2.1, Nemotron-3-Diarization) in audioforge | Sortformer v2 pooled DER 0.201 offline / 0.252 streaming (1.04 s) vs 0.384 for our stage-1 head, AMI dev 64 × 20 s windows | 2026-09-26 |
| [SPK_HEAD.md](archive/SPK_HEAD.md) | is the weak speaker head an architecture or a supervision problem? | dev-era (n=64): the block-4 relational head cuts within-meeting EER by about 18 points (AMI) and 37 points (ICSI) against the stage-1 all-layer head (32.2 % / 42.3 %) | 2026-09-26 |
| [LAYER_ROUTING.md](archive/LAYER_ROUTING.md) | do single-block taps help the diarization and turn heads? | negative: best diar tap (block 6) −0.019 AMI / −0.023 ICSI DER (seed mean), below the pre-registered 0.03 bar; gap to Sortformer v2 0.16-0.18 | 2026-09-27 |
| [VAD_LAYERS.md](archive/VAD_LAYERS.md) | which encoder block carries VAD, and does a gated encoder save compute? | block-4 probe F1 0.9476 vs served head 0.9485; gating needs 420 frames (34 s) of re-prime per onset, so do not gate | 2026-09-26 |

## End-of-turn

| doc | what it answers | key number | date |
|---|---|---|---|
| [BASELINES.md](archive/BASELINES.md) | our heads vs dedicated open models on the same data and scorer (turn, VAD, diarization, speaker, LID) | hybrid 61.9 % missed turn ends vs 72.7 % Silero timeout (open: 45.9 % vs 26.7 %), AMI dev n=974, 6 s, ≤ 5 % FC | 2026-09-26 → 2026-09-27 |
| [EOT_BENCH_V2.md](archive/EOT_BENCH_V2.md) | leak-free, speaker-aware end-of-turn benchmark; enrollment and primary-speaker binding | enrollment leak (timeout, causal deployable minus oracle label-armed): +45.5 [+41.9, +48.8] pts miss, AMI dev n=974, 6 s | 2026-09-26 |
| [STAGE1.md](archive/STAGE1.md) | first real-speech numbers for the fused front end (AMI dev) | turn head with oracle activity 5.8 % miss, with the streaming Sortformer track 62.6 % vs 38.4 % for a timeout (n=200, ≤ 5 % FC); superseded by EOT_BENCH_V2 | 2026-09-26 |
| [TURN_ABLATION.md](archive/TURN_ABLATION.md) | 4-way ablation (acoustic / speaker / text) of a speaker-aware turn head on synthetic conversations | +speaker head P50 400 ms with oracle activity, 93.9 % miss with its own diarization (1000 synthetic conversations, ≤ 5 % FC) | 2026-09-26 |
| [TURN_ERRORS.md](archive/TURN_ERRORS.md) | why turn head v3 works on oracle activity and fails on the Sortformer track | cleaning only the speaker-kernel input: 62 % → 9.5 % miss (AMI dev n=200, ≤ 5 % FC) | 2026-09-26 |
| [COMPLETENESS.md](archive/COMPLETENESS.md) | can smart-turn fusion or a completeness head fix open-end misses? | completeness head frame AUC 0.995 vs smart-turn v3.2 0.994 (399 held-out human_5 clips); negative for fusion on open ends | 2026-09-26 |
| [YIELD_TOKENS.md](archive/YIELD_TOKENS.md) | turn-taking as `<YIELD>` / `<HOLD>` transducer tokens | negative: never emitted; posterior misses 94.4 % [92.9, 95.9] of turn ends vs hybrid 61.9 %, AMI dev n=974, 6 s | 2026-09-26 |
| [CONTAMINATION.md](archive/CONTAMINATION.md) | does a transient wrong speaker binding persist after correction? | yes: half-life about 1.5-2 s (\|Δp\| 0.12 at 0 ms, 0.06 at 1 s, 0.04 at 2 s after a 1 s zeroed window, n=321 turns); replay costs 208 ms per rebind | 2026-09-26 |
| [DYADIC.md](archive/DYADIC.md) | transfer to two-party audio; TurnBench dev with the official scorer; training on otoSpeech | head OR per-channel Silero 0.835 recall / 0.080 FP / P50 1390 ms vs VAP 0.841 / 0.045 / 463 ms, TurnBench dev | 2026-09-26 → 2026-09-27 |

## Integration and product

| doc | what it answers | key number | date |
|---|---|---|---|
| [INTEGRATION.md](archive/INTEGRATION.md) | streaming server, Pipecat and LiveKit adapters, measured end to end | 1000 ms timeout, 0.32 s diarizer preset: median dead air 1.92 s Pipecat / 1.68 s LiveKit, 5 AMI windows | 2026-09-26 |
| [INTEGRATION_VERIFY.md](archive/INTEGRATION_VERIFY.md) | independent re-run of every integration claim | RTF at 1x, 2 threads confirmed (0.453-0.519); as-fast-as-possible RTF 0.33 / 0.27 not reproduced (0.42-0.53) | 2026-09-26 |
| [E2E_FINAL.md](E2E_FINAL.md) | our front end vs the Pipecat and LiveKit default local stacks, live (7 systems, 37 clips, paired CIs) | product default C 1.1-1.4 vs Pipecat default A 2.1-7.5 cut-ins per minute, fewer missed ends on every clip set; Nemotron-3 recommended as the default diarizer | 2026-09-27 |
| [BULLETPROOF.md](archive/BULLETPROOF.md) | what a real deployment throws at the server, stream client and adapters; each fix and the test that pins it | - | 2026-09-28 |
| [DIARIZATION_FIX.md](archive/DIARIZATION_FIX.md) | "many speakers in the room doesn't work": reproduction, diagnosis and the served fix (`--diar-labels`, `--shed-diar hold`) | - | 2026-09-28 |
| [PERFORMANCE.md](archive/PERFORMANCE.md) | what every RTF means, what the served model costs on CPU, and the fast paths (`--perf`) | - | 2026-09-28 |

## Language ID

| doc | what it answers | key number | date |
|---|---|---|---|
| [LID.md](archive/LID.md) | can the front end also identify the spoken language? | worse: head 75.0 % (2 s) / 89.3 % (full utterance) vs AmberNet 95.1 % / 99.5 %, FLEURS test, 17 languages, n=2550 | 2026-09-27 |

## Verification, reviews and idea rounds

| doc | what it answers | key number | date |
|---|---|---|---|
| [VERIFICATION.md](archive/VERIFICATION.md) | adversarial check of early claims, ablations and edge-case bugs | 28 claims: 4 confirmed, 20 partially true, 4 refuted | 2026-09-25 |
| [IDEAS.md](archive/IDEAS.md) | ideas tournament, round 1 | 1 of 32 ideas survived (architecture-2); prototype inconclusive | 2026-09-26 |
| [IDEAS2.md](archive/IDEAS2.md) | ideas tournament, round 2 (the measured turn-taking failure) | 1 of 21 ideas survived (wildcards-3, which became eot-bench v2) | 2026-09-26 |
| [OUTSIDE.md](archive/OUTSIDE.md) | outside corpora, objectives, embedders and benchmarks (TurnBench, otoSpeech, VAP) | - (survey, no model run) | 2026-09-26 |
| [QUESTIONS.md](archive/QUESTIONS.md) | open questions sent to outside reviewers | - | 2026-09-26 |
| [REVIEW_ACTIONS.md](archive/REVIEW_ACTIONS.md) | what the outside review changed, queued or was disputed | - | 2026-09-26 |
| [RESEARCH_PRIORITY_REVIEW.md](archive/RESEARCH_PRIORITY_REVIEW.md) | research priorities reconciled with the implemented experiments (state-contamination probe design) | - | 2026-09-26 |
| [FRONTIER.md](archive/FRONTIER.md) | new problems the 2025-2026 NVIDIA speech recipe makes tractable for a small team | - (survey, no model run) | 2026-09-28 |

## Data and housekeeping

| doc | what it answers | key number | date |
|---|---|---|---|
| [AMI.md](archive/AMI.md) | AMI subset, splits, word-level labels and turn-taking statistics | median within-turn hesitation (1.1-1.3 s) is longer than the median silence at a true end of turn (0.9-1.2 s); 43-47 % of switches overlap | 2026-09-25 |
| [ICSI.md](archive/ICSI.md) | ICSI feasibility, conversion to AMI labels, statistics | median hesitation 0.56-0.59 s vs median positive switch gap 0.60-0.64 s; 40-42 % of switches overlap | 2026-09-26 |
| [CLEANUP_TODO.md](archive/CLEANUP_TODO.md) | code-cleanup backlog (ruff findings, duplicates, suspected bugs) | - (items addressed in ISSUES_FIXED) | 2026-09-26 |
| [ISSUES_FIXED.md](archive/ISSUES_FIXED.md) | log of fixes for known open issues | - (674 passed, 5 skipped after part A) | 2026-09-28 |

## Other files

- `papers/`: plain-text copies of three arXiv papers (2604.19079, 2605.15442, 2606.25621) read for PLAN.md §3.
- `archive/raw/`: Hugging Face API pulls and model cards for NVIDIA's audio models, the input to ANALYSIS.md.
- `catalog.json`: the classified catalog of NVIDIA audio models (family, encoder, decoder, licence, downloads) behind ANALYSIS.md.
- `features.json`: per-model architecture feature tags extracted from the model cards.
