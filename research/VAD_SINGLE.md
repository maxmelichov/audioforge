# Single-layer VAD head (block 4) vs the served all-layer head

2026-09-29. Follow-up to research/VAD_LAYERS.md, where a block-4 probe came within 0.001 F1 of the served VAD head.
The served head (`runs/stage1_served.afm` heads.vad, bit-identical to `stage1_heads_pretrained.afm`'s) has 64 hidden
units and reads a learned softmax mix of all 17 FastConformer blocks. This study trains real heads on one block and
checks them against it, both directly and through every consumer of the VAD in the server.

Driver: `scripts/research/vad_single.py`. Recipe: `research/recipes/vad_single.yaml`. Numbers: `runs/vad_single.json`.
Checkpoints: `/Volumes/ExternalSSD/nvidia-audio-models/runs/vad_single/L{3,8}/head.pt`.
Built model: `runs/stage1_served_v2.afm`. `stage1_served.afm` is untouched.

**Pre-registered bar.** A single-tap head ships only if both of these hold:
- its F1 is within 0.005 of the served head's on AMI dev and on ICSI dev;
- no downstream metric (LID, TS-VAD tracking, turn misses, streaming words) is worse outside its CI.

If block 4 meets the bar it is preferred, because it shares the speaker head's tap. Otherwise block 9 is next, and
failing both the all-layer head stays.

**Verdict: block 4 meets the bar; block 9 does not.**
- Block 4 is +0.0026 F1 on AMI and −0.0022 on ICSI. It misses less at FPR 0.075 on both corpora (AMI 10.6 vs 12.1 %,
  ICSI 19.8 vs 24.9 %).
- Every downstream paired CI for block 4 contains 0.
- Block 9 matches on VAD and LID. Its turn-end misses on the VAD-armed TS-VAD track are worse outside the CI (AMI
  head +1.1 / +1.3 points at 2 s / 6 s, ICSI hybrid_dyn +1.0 at 2 s).
- `runs/stage1_served_v2.afm` = `stage1_served.afm` with heads.vad on block 4 (`vad_layer: 3`, zero-based) and
  `layer_mix.vad` dropped. The other 754 tensors are bit-identical, and the LibriSpeech transcripts are identical
  (50 utterances, WER 1.23 % for both).
- **The compute saving is negligible.** The ASR still runs all 17 blocks, so a single tap only removes the 17-way
  weighted sum: 0.037 → 0.013 ms per 160 ms chunk, which is noise in a 72 ms session step. The reason to ship it is
  a simpler head that sits earlier in the pipeline, not speed.

## Training (the served recipe, only `from_layers` changed)

Everything is the recipe of `stage1_heads_pretrained.yaml` for this head:
- Data: the same three sources in the same batch order (`Trainer._batches`, `random.Random(0)`):
  - synthetic conversations, 2500;
  - AMI turn windows, 16 s, 3274;
  - AMI diar windows, 16 s, 2697.
- Labels: vad = any speaker active.
- The frozen encoder runs in train mode: dropout 0.1, SpecAugment, and a random lookahead per step from the
  checkpoint's `[[70,13],[70,6],[70,1],[70,0]]`, as the original run had.
- Optimisation: batch 6, AdamW lr 1e-3 (betas 0.9 / 0.98, wd 1e-3), 100 warmup steps + cosine, 2000 steps, BCE
  weight 0.5, grad clip 1.0.

The one difference is that only blocks 1..k+1 run in the loop, since block k+1's output does not depend on deeper
blocks. The gradient clip applies to the VAD head's gradients alone, where the original clipped the joint gradient of
all heads (Adam is scale-invariant to first order).

Run cost on MPS: block 4 took 2.7 min and block 9 took 3.9 min. Each head has 32 897 parameters (the served head
has 32 897 + 17 mix weights).

## Before / after

VAD sets: AMI dev diar 64 × 20 s (the BASELINES set; the served head reproduces F1 0.9485 and miss 12.07 % exactly)
and ICSI dev diar 64 × 20 s. All heads are scored from one CPU encoder pass. Δ columns are paired: a window bootstrap
for VAD and TS-VAD frames, an utterance bootstrap for LID, and eot-bench v2 pairs for turns.

| metric | served (all 17) | **block 4** | Δ block 4 [95 % CI] | block 9 | Δ block 9 [95 % CI] |
|---|---|---|---|---|---|
| AMI VAD F1 @0.5 | 0.9485 | **0.9511** | +0.0026 [−0.0005, +0.0059] | 0.9503 | +0.0018 [−0.0006, +0.0043] |
| AMI miss @FPR 0.075 / AUC | 12.07 % / 0.970 | **10.62 % / 0.972** | | 12.32 % / 0.970 | |
| ICSI VAD F1 @0.5 | 0.8999 | 0.8977 | −0.0022 [−0.0040, −0.0006] (< 0.005) | 0.9005 | +0.0006 [−0.0010, +0.0020] |
| ICSI miss @FPR 0.075 / AUC | 24.93 % / 0.919 | **19.83 % / 0.931** | | 24.12 % / 0.923 | |
| LID (lid_distill, VAD-gated pooling), FLEURS-17 2 s, n = 2550 | 91.02 % | 90.94 % | −0.08 [−0.24, +0.08] | 91.06 % | +0.04 [−0.16, +0.24] |
| LID, full utterance | 97.80 % | 97.84 % | +0.04 [−0.08, +0.16] | 97.92 % | +0.12 [0.00, +0.27] |
| TS-VAD armed-print track, frame F1, AMI 974 | 0.6345 | 0.6349 | +0.0004 [−0.0006, +0.0013] | 0.6347 | +0.0002 [−0.0006, +0.0011] |
| same, ICSI 1312 | 0.7024 | 0.7033 | +0.0009 [+0.0003, +0.0015] | 0.7024 | 0.0000 [−0.0003, +0.0004] |
| turn miss, head on that track, AMI 2 s / 6 s | 79.5 / 55.4 % | 80.2 / 56.2 % | +0.65 [−0.17, +1.54] / +0.75 [−0.16, +1.80] | 80.7 / 56.7 % | **+1.09 [+0.15, +2.09] / +1.31 [+0.32, +2.52]** |
| turn miss, hybrid_dyn, AMI 2 s / 6 s | 75.2 / 50.5 % | 75.5 / 51.3 % | +0.33 [−0.90, +1.61] / +0.84 [−0.17, +1.96] | 76.0 / 51.3 % | **+0.83 [+0.08, +1.75]** / +0.84 [−0.17, +1.96] |
| turn miss, head, ICSI 2 s / 6 s | 71.4 / 57.1 % | 71.5 / 56.9 % | +0.10 [−0.34, +0.55] / −0.21 [−0.75, +0.33] | 71.7 / 57.0 % | +0.33 [−0.08, +0.75] / −0.08 [−0.49, +0.33] |
| turn miss, hybrid_dyn, ICSI 2 s / 6 s | 69.6 / 55.6 % | 70.0 / 55.2 % | +0.46 [−0.52, +1.52] / −0.35 [−0.94, +0.19] | 70.6 / 55.5 % | **+1.04 [+0.41, +1.77]** / −0.15 [−0.82, +0.55] |
| streaming tokens, `--asr-vad-gate` off (32 AMI dev windows, 640 s) | ref | identical on 32 / 32 | | not run | |
| streaming, gate on (0.5, hangover 15) | ref (910 frames gated) | 31 / 32 windows identical, 0.08 % word diff (929 gated) | | not run | |
| VAD read-out ms per 160 ms chunk (CPU, 2 threads) | 0.037 | 0.013 | −0.024 | | |
| `bench_serve components`, `asr.pass_total` ms per 160 ms (e2e5, nemotron) | 22.96 | 22.75 | −0.2 (noise) | | |
| `session.total` ms per 160 ms / chunk_ms p95 | 72.4 / 125.5 | 72.0 / 125.7 | noise | | |

Turn misses are measured at a cross-fitted ≤ 5 % per-turn false-cutoff rate. On AMI the block-4 systems land at a
slightly lower realised FC (7.7 vs 8.1 %), which accounts for part of their +0.7-point miss estimate.

## Protocols

- **VAD.** `sd.score_vad` at 0.5 on the any-speaker 80 ms label, AUC, and the miss rate at FPR 0.075 (the point
  where BASELINES compares us with Silero). `vad_single.py eval`.
- **LID.** The `lid.py eval_head` VAD-gated rule: frames with VAD ≤ 0.5 are not pooled, and a clip with no speech
  frame pools everything. It uses the served `runs/lid_distill.pt` on the research/LID.md clips (onset − 0.1 s,
  2 s; full utterance), with all three VADs computed from the same encoder pass. The served-VAD column reproduces
  runs/lid.json `lid_distill_vadgated` exactly (91.02 / 97.80); the ungated head gives 91.18 / 97.69.
  `vad_single.py lid`: 2 s on CPU, full on MPS.
- **TS-VAD.** The VAD reaches the track only through `TSVADTrack.arm`: the live print (`--enroll after_agent_arm`,
  part of the single-model default preset) is the first 5 s of frames with VAD > 0.5 after the arm. The track is
  scored on the eot-bench v2 windows (AMI dev 974, ICSI held-out 1312), armed at the last other-speaker speech end
  before the primary's onset (`tsvad_enroll.arm_frame`), with `runs/tsvad_spk.pt` and the plain-VAD mode before the
  print. That gives one track per VAD (`vad_single.py vadwin` + `tsvad`; tracks in tsvad.py's `bind_tsvad_armvad_*`).
  The served turn head scores each track (`tsvad.py scores --bindings tsvad_armvad_served,tsvad_armvad_L3,
  tsvad_armvad_L8`, CPU). `vad_single.py tsvadturn` then runs `tsvad.score_systems`: head alone and hybrid_dyn with
  Silero, 1000 bootstraps, paired against the served-VAD track. With explicit enrollment the VAD does not touch the
  track at all.
- **Streaming.** `ASRStream` (the server's ASR pass) fed in 20 ms blocks with `turn=None`.
  - Gate off: the VAD cannot change the transcript by construction (streams.py decodes every frame), and the tokens
    match on every window.
  - Gate on: 1 of 32 windows differs by a few words at a gate boundary (19 more frames gated, 929 vs 910).
- **Compute.** `vad_single.py compute` times `head(head_input(...)).sigmoid().tolist()` on 125 real streaming chunks
  (interleaved A/B, 50 reps). `bench_serve.py components` ran on both checkpoints (e2e5 clips, nemotron diarizer,
  2 threads). The encoder's per-layer outputs are returned for the speaker / LID / TS-VAD taps either way, so the
  single tap saves only the 17-way weighted sum.

## Commands

```
S=scripts/dev/gate.sh; P=.venv/bin/python; export PYTHONPATH=.
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 $S $P scripts/research/vad_single.py train --layer 3 --budget 1020
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 $S $P scripts/research/vad_single.py train --layer 8 --budget 1020
$S $P scripts/research/vad_single.py eval
$S $P scripts/research/vad_single.py lid --budget 540          # 2 s on CPU; then, for full:
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 $S $P scripts/research/vad_single.py lid --device mps --budget 1000
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 $S $P scripts/research/vad_single.py vadwin --device mps --budget 1000
$S $P scripts/research/vad_single.py tsvad --budget 540
$S $P scripts/research/tsvad.py scores --device cpu --bindings tsvad_armvad_served,tsvad_armvad_L3,tsvad_armvad_L8 --budget 540   # repeat
$S $P scripts/research/vad_single.py tsvadturn
$S $P scripts/research/vad_single.py stream --layers 3 --n-stream 32 --budget 540     # repeat
$S $P scripts/research/make_served_model.py --vad-head /Volumes/ExternalSSD/nvidia-audio-models/runs/vad_single/L3/head.pt \
    --out runs/stage1_served_v2.afm --check-wer 50
$S $P scripts/research/vad_single.py compute --layers 3 --reps 50
$S $P scripts/research/bench_serve.py components --asr runs/stage1_served_v2.afm --out <ssd>/components_v2.json   # and stage1_served.afm
$S $P -m pytest -q tests/test_served_v2.py tests/test_serve_shipped.py tests/test_make_served_model.py
```
