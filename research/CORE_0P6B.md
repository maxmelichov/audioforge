# A second core: nemotron-speech-streaming-en-0.6b with every head retrained (`--core 0.6b`)

2026-10-01. Script: `scripts/research/core_0p6b_heads.py`. Results: `runs/core_0p6b.json`; the WER rows come from
`runs/hybrid_asr.json` (`core_0p6b`); the 115M rows come from the files named in each table. Mac only: every job ran
through `scripts/dev/gate.sh` on CPU (2 threads) or MPS, one job at a time. Caches are on the SSD
(`scratch/core_0p6b/`, 101 GB, of which the VAD block caches are 29 GB).

## Verdict

**Ship the 0.6B as an opt-in second core for transcripts and target-speaker transcripts. Keep the 115M as the
default.**

The 0.6B transcribes meetings at about half the error:

| | 115M | 0.6B |
|---|---:|---:|
| AMI WER | 24.4 % | 11.2 % |
| ICSI WER | 27.3 % | 14.4 % |
| live two-party sessions, every word | 22.5 % | 13.7 % |
| live sessions, the user's own channel | 17.1 % | 7.7 % |

It also does better on:
- target-speaker WER on ICSI (31.7 vs 37.1 %),
- AMI within-meeting speaker EER (13.6 vs 17.4 %, same recipe),
- spoken-language ID with the same training data,
- ICSI VAD F1 and ICSI speaker tracking.

It is **not better at end of turn**, which the product is built around. With the same rules:
- AMI turn ends are answered 150 ms sooner with fewer misses (28.0 vs 33.5 %).
- The two-party calls miss more turn ends (11.0 vs 7.3 %).
- AMI false interruptions rise (13.0 vs 10.5 %).
- `fast` interrupts far more often (calls 33.9 vs 24.8 %).
- The re-tuned `assistant` preset nearly matches the 115M (91.2 vs 92.2 % on smart-turn's test clips) but answers 80 ms later.

It costs:
- 3× the compute per chunk on CPU (96 vs 30 ms per 160 ms chunk).
- 1 real-time stream on 2 CPU threads instead of 4, and 3 on the Apple GPU instead of 5.
- 5 GB RSS (+3.3 GB GPU) instead of 1.1 GB.
- A different licence: NVIDIA Open Model License instead of CC-BY-4.0.

**Correction.** The 0.6B is real time on 2 CPU threads for one stream: 96 ms per chunk, RTF 0.61, measured on a quiet
machine. The "not real time on CPU" note in the brief (RTF 1.68, IMPROVEMENTS.md item 6) and my first measurement (240
ms, `core_cost_contended`) were both taken while another job held the CPU and GPU.

## What was built

- **Import.** `audioforge/nemo_import.py` gives 618.5 M parameters. The served model sets attention [70,1] (160 ms
  chunks) as its default; the import's default is [70,13].
- **Streaming equality at [70,1]** (`verify`), on 3 LibriSpeech utterances:
  - `stream_step` matches the masked offline forward to max |d| 3.1e-6;
  - `StreamingSession` (raw audio in 160 ms pieces) gives exactly the offline text for 3 of 3.
- **Hub.** Component `asr_0p6b`:
  - repo `nvidia/nemotron-speech-streaming-en-0.6b`, revision `ebe59e5a817142986528bbbee5dba8db7b38ed50`;
  - sha256 `283638054c44…62ac9cd`, which equals the local file and the HF etag;
  - licence: NVIDIA Open Model License Agreement (model card: "ready for commercial/non-commercial use").
- **Heads and assets.**
  - `assets/served_heads_0p6b_v0.1.pt`: 118 tensors, 16.0 MB. `hub.build_served` rebuilds `served_0p6b_v0.1.afm`
    bit-identically (tensor hash `564a419c…`).
  - `assets/tsvad_0p6b.pt`, `assets/lid_0p6b.pt`.
  - `audioforge-download --core 0.6b`, `audioforge-serve --core 0.6b`, `audioforge.load(core="0.6b")`,
    `audioforge.voiceprint(core="0.6b")`. Voice prints belong to one core.
- **Serving changes.**
  - A recurrent VAD head type `frame_gru` with per-session state (equal to the whole-sequence forward).
  - Per-model preset constants (`cfg["turn_presets"]`, merged over `TURN_PRESETS`).
  - A fix: the v5 classifier is now attached on the model's device. A `--device mps|cuda` session with `fast` /
    `assistant` used to pass CPU tensors to the GPU model.
  - **Check:** the served session gives the same turn_end times as the offline twin of each preset on 16 of 16
    sessions per preset (`served_check`).

## Heads on the 0.6B

| head | reads | recipe (and the cheap improvements) | file |
|---|---|---|---|
| VAD | softmax mix of blocks 4/8/12/16/20/24 | causal GRU-64 frame head (90.6 K params). Data: 1200 AMI train windows plus 2 SpecAugment re-encoded views, frame-centre labels, 25 % room-tone negatives, feature dropout / noise / time masks. Early stopping on held-out AMI meetings plus ICSI train windows | `heads/vad.pt` (`vad2 gru64_sa`) |
| speaker | block 5 | crop-level TitaNet-L distillation (`spk_frame.py`: cos + relational + AAM, AMI + ICSI + LibriSpeech-100, 4000 steps) | `spk_frame/head_c0p6b5_fresh.pt` |
| TS-VAD | block 5 + the speaker head's print | `tsvad.py` recipe (24 train meetings, prints from the speaker head) plus 30 simulated LibriSpeech conversations (0.3 of batches) and an overlap-weighted loss (×2.5) | `heads/tsvad.pt` |
| turn classifier v5 | block 12 + VAD / P(user) / P(other) + RNNT text | turn_v5 `c5` recipe on the 0.6B's own turn inputs (26 k clips; smart-turn distillation and LM-completeness targets reused, both audio- or time-aligned) | `turn/s12/model.pt` |
| per-frame turn head | top block + TS-VAD columns + RNNT text | the served TurnHead without the speaker-kernel pass, heads-only on cached inputs, served objective | `heads/turn_f1.pt` |
| LID | softmax mix of blocks 16 / 20 | `lid_fix.py` kd recipe (AmberNet KL, α 0.8) on FLEURS train (250 per language) plus extra English. No `trainx` | `heads/lid.pt` |

- **Pass-2 speaker kernels.** The encoder supports kernel injection, but the 0.6B has no trained kernels. Training them
  needs backprop through all 24 × 1024 blocks, which is not feasible on the Mac. The 0.6B's turn path is therefore
  pass 1: the per-frame head reads pass-1 frames plus the TS-VAD columns for `balanced` / `steady`, and the v5
  classifier serves `fast` / `assistant`. It has no second encoder pass, which is also why its engine costs about the
  same as its bare core.
- **Block choice.** The layer sweeps (`research/LAYER_SWEEP_0P6B.md`, selected on held-out meetings and speakers)
  moved the speaker and TS-VAD heads from block 11 to block 5.
  - AMI within-meeting EER: 13.6 vs 16.2 %. ICSI: 3.6 vs 2.9 %.
  - TS-VAD AMI F1: 0.760 vs 0.748.
  - The turn-set TS-VAD tracks were recomputed after the move (`turn_retrack`).

## Side by side (same harnesses)

Unless stated otherwise:
- CI = 95 % bootstrap;
- 115M = the shipped single-mode model (`stage1_served_v3.afm`, the shipped TS-VAD / LID heads);
- 0.6B = `served_0p6b_v0.1.afm` + `tsvad_0p6b.pt` + `lid_0p6b.pt`.

| category | metric (set) | 115M | 0.6B | 0.6B − 115M [CI] | source |
|---|---|---:|---:|---|---|
| ASR | WER LibriSpeech-200, [70,1] | 2.29 % | 2.20 % | −0.09 [−0.46, +0.28] | hybrid_asr.json core_0p6b |
| | WER AMI-200 | 24.43 % | **11.16 %** | **−13.27 [−15.48, −11.23]** | " |
| | WER ICSI-200 | 27.29 % | **14.35 %** | **−12.94 [−15.11, −10.90]** | " |
| | WER live two-party sessions, every word (32 TurnBench sessions, mono + user channel) | 22.47 % | **13.70 %** | | core_0p6b.json tswer_live |
| | WER the same, user channel only | 17.09 % | **7.69 %** | | " |
| target-speaker WER | AMI eot-bench windows, primary (TS-VAD mask, 5 s print) | 62.10 % | 63.21 % | +1.11 [−3.94, +7.47] | core_0p6b.json tswer |
| | ICSI | 37.14 % | **31.69 %** | **−5.45 [−8.84, −2.60]** | " |
| | live, all 32 sessions | 29.14 % | 25.63 % | (CIs overlap) | tswer_live |
| VAD | F1 / AUC, AMI dev 64 × 20 s | 0.9511 / 0.9719 | 0.9507 / 0.9715 | F1 −0.0004 [−0.005, +0.004]; AUC −0.0004 [−0.003, +0.002] | vad_cmp |
| | F1 / AUC, ICSI dev | 0.8977 / 0.9313 | **0.9055** / 0.9245 | F1 +0.008 [+0.006, +0.010]; AUC −0.007 [−0.011, −0.003] | vad_cmp |
| | miss at FPR 0.075, AMI / ICSI | 10.6 / 19.8 % | 10.7 / 22.4 % | | vad_miss_at_fpr0075 |
| | room tone: p95 / frames > 0.5 | 0.39 / 2.5 % | 0.21 / 2.1 % | | vad_room_115m, vad2 |
| speaker | within-meeting EER AMI dev n = 200 (same crop-level recipe) | 17.4 % (19.8 shipped) | **13.6 %** | **−3.8 [−6.6, −1.7]** | spk_paired |
| | within-meeting EER ICSI dev n = 200 | 4.0 % (7.0 shipped) | 3.6 % | −0.4 [−1.4, +0.7] | " |
| tracking | target F1, AMI / ICSI eot-bench (primary, 5 s print) | 0.743 / 0.882 | 0.760 / 0.894 | | improve_115m.json, tsvad_frame |
| | target-speaker DER, AMI / ICSI | 50.9 / 23.0 % | 51.8 / 21.1 % | | " |
| end of turn | `balanced`: calls p50 / p95 ms, false interruptions, missed | 956 / 1919, 20.2 %, 7.3 % | 920 / 1870, 20.2 %, **11.0 %** | | core_0p6b.json eot |
| | `balanced`: AMI | 1326, 10.5 %, 33.5 % | **1176**, 13.0 %, **28.0 %** | | " |
| | `fast`: calls | 547 / 1862, 24.8 %, 5.5 % | 530 / 1817, **33.9 %**, 11.9 % | | " |
| | `fast`: AMI | 1247, 11.5 %, 34.0 % | 1017, 19.5 %, 24.5 % | | " |
| | `assistant`: smart-turn test (399): accuracy, p50 / p95 ms, false fires | 92.2 %, 292 / 704, 5.4 % | 91.2 %, 374 / 920, 7.1 % | | " (0.6B constants: 320 ms quiet, p > 0.99) |
| LID | FLEURS-17 test, 2 s / full (cached clips), same training data (FLEURS train + extra English) | 84.6 / 94.5 % | **87.4 / 95.3 %** | | lid_fix, lid_fix_115m |
| | EdAcc English called English, 2 s / full | 45.1 / 55.2 % | **71.2 / 83.3 %** | | " |
| | shipped 115M head (trained with FLEURS `trainx`, 47 k more rows) | 91.0 / 97.8 % | | | runs/lid.json |
| compute | full single-mode engine, ms per 160 ms chunk p50 / RTF, CPU 2 threads | 29.8 / 0.19 | 95.7 / 0.61 | | mps_115m.json, cost_engine |
| | same, MPS | 28.7 / 0.18 | 39.3 / 0.25 | | " |
| | bare ASR core (StreamingSession), CPU / MPS ms per chunk | 17.6 / 10.2 | 99.5 / 30.8 | | mps_115m.json, core_cost |
| streams | real-time single-mode sessions, CPU 2 threads / MPS | 4 / 5 | 1 / 3 | | streams_cpu / mps_115m.json, cost_streams |
| memory | peak RSS (+ MPS driver) | 1.1 GB (+0.5) | 5.0 GB (+3.3) | | " |

**How the end-of-turn rows were measured.**
- The 0.6B rows come from the served 0.6B engine, dumped once per session (232 calls / AMI sessions with their 5 s
  print re-made for the 0.6B; 399 assistant clips without a print). They are scored with turn_v5's `run_policy` twin and
  the eot_latency / eot_assistant scorers.
- The 115M rows are the shipped dumps through the same scorer; they reproduce the published numbers exactly.
- Compute inside "total" latency: 115M on CPU (30 ms per chunk), 0.6B on MPS (41 ms).
- The 0.6B `assistant` constants were re-tuned on these sets, as the 115M's were (`presets_0p6b`). With the 115M's
  constants it scores 78.2 % / 230 ms / 23.2 % false fires.
- For `balanced` and `fast`, no setting of the scan met the 115M's bars on calls and AMI (0 of 48 and 0 of 243), so
  they keep the shared constants.

## Why end of turn did not improve

**The calls miss more turn ends** (12 of 109). The TurnBench user channels keep the 0.6B's VAD at 0.4–0.8 for 1–2 s
after the user stops, so neither the head path nor the 640 ms fallback fires before the other party answers. The GRU
VAD's state lengthens this tail.

**A per-frame VAD is not a clean fix.** Swapping in the per-frame mlp VAD on the same dumps (`vad_swap`) gives:
- calls missed 9.2 % instead of 11.0 %,
- AMI false interruptions 17.5 % instead of 13.0 %.

So the VAD alone does not account for the turn result.

**The v5 classifier on the 0.6B matches the 115M's on held-out data:**

| | 0.6B | 115M (c5) |
|---|---:|---:|
| end-vs-pause AUC | 0.815 | 0.816 |
| end vs cut | 0.908 | 0.903 |
| smart-turn clips | 98.7 % | 99.2 % |

But it is calibrated differently. Thresholds tuned for the 115M make it fire early, which is why `fast` interrupts
more.

## Not done / follow-ups

- **Target-speaker WER and the TS-VAD head choice.** The plain-recipe TS-VAD head (no simulated mixtures, no overlap
  weight) gives AMI tWER 52.0 % (against 62.1 for the 115M and 63.2 for the shipped 0.6B head) and ICSI 30.0 %
  (`tswer_head_base`), with lower tracking F1 (0.751 vs 0.760) but better AMI DER (48.0 %). The shipped head was
  selected on held-out target F1, as the user's rule says. Switching to the plain head means retraining both turn heads
  on its tracks (~2 h).
- **The 115M layer sweep** (`LAYER_SWEEP_115M.md`) did not fit after the 0.6B evaluation; it is listed as a follow-up.
- **LID without `trainx`.** The 0.6B LID head was trained without FLEURS `trainx` (not cached for the 0.6B). The same-data
  115M control is the fair comparison.
- **TurnBench precision / recall.** Not measured for the 0.6B.
- **WER on the served model.** The WER rows were measured on the imported 0.6B. The served model's base tensors are
  identical (the heads asset rebuilds them by hash), at the same [70,1].
