# Distil nemotron-speech-streaming-en-0.6b into the 115M streaming core

research/IMPROVEMENTS.md section 7. The served 115M cache-aware FastConformer transcribes AMI at 24.4 % WER in
streaming ([70,1], 160 ms chunks); the 0.6B streaming model of the same family reaches 11.2 % at the same latency
(section 6) but costs 3x the compute. This job trains the 115M encoder + RNNT towards the 0.6B, keeping the 115M's
cost and every other head of the served model.

**Pre-registered bar:** AMI-200 streaming WER **<= 17 %** at [70,1] and LibriSpeech-200 **<= 2.59 %** (+0.3 of 2.29),
the VAD / turn heads re-checked on the new encoder (they read its features) and not worse beyond their CIs.
**Kill:** the WER gate fires at every checkpoint, or AMI not below the no-teacher control by > 1.5 points.

## What trains

| part | what |
|---|---|
| teacher | the 0.6B at [70,1] (and [70,13] with `--contexts 1,13`): greedy transcript (normalised) and top-layer encoder output (T, 1024) fp16 for every item (~16 GB per context for ~170 h) |
| student | the whole 115M encoder + RNNT prediction net and joint of `runs/stage1_served.afm`; every other head (VAD, speaker, turn, CTC, diar, EOU) frozen and asserted bit-identical at the end |
| losses | AMI / ICSI batches: 0.5 RNNT(reference) + 0.5 RNNT(teacher transcript) + 1.0 (1 - cos(P h_student, h_teacher)) on valid frames (P = Linear(512, 1024)); LibriSpeech batches (25 %): RNNT(reference) + 0.5 KL(student joint || frozen initial student) (the repo's R2 anchor) |
| optimiser | AdamW, lr 2e-4, weight decay 1e-3, 1000 warm-up steps then cosine, layer-wise LR decay 0.9 per block from the top (block 17 at lr, block 1 at 0.9^16 lr, pre-encode 0.9^17), heads at lr; bf16 autocast; grad-norm clip 1.0 |
| schedule | 20 000 steps x batch 32 (~10 s items) ~ 1800 h of audio seen; checkpoint every 500 steps (`<work>/<tag>/ckpt.pt`, resumable: re-run the same command) |
| WER gate | every 500 steps, LibriSpeech test-clean first 100 utterances at [70,1]; WER > step-0 baseline + 0.003 -> the last good checkpoint is restored and the run stops (`"gate_fired"` in `history.json`) |
| control | `--control`: the same run with no teacher term (continued training on the references at the same budget) |

Data: AMI (136 train meetings) + ICSI (70 train meetings) single-speaker 1-15 s segments (the library's asr mode) as
the training corpus, ~100 h of LibriSpeech train-clean-100 as the anchor (~170 h in total). Eval: the HYBRID_ASR.md sets (AMI-200 from the
4 AMI dev meetings, ICSI-200 from the 2 ICSI dev meetings, LibriSpeech test-clean first 200), seeded exactly as on
the laptop.

## Steps (from the repo root on the GPU box)

```bash
# 0. copy the student from the laptop (it carries the trained heads; not downloadable):
scp <laptop>:nvidia-audio-models/runs/stage1_served.afm runs/
# 1. preflight, then environment, teacher import, datasets (~1 h wall, network-bound; no GPU):
bash scripts/research/distill_0p6b_to_115m/preflight.sh
bash scripts/research/distill_0p6b_to_115m/setup.sh
D=scripts/research/distill_0p6b_to_115m/distill.py; W=work/distill; export AUDIOFORGE_0P6B_AFM=runs/nemo_nemotron_speech_streaming_en_0.6b.afm
PY=.venv/bin/python
# 2. manifest and eval sets (CPU, ~15 min)
$PY $D manifest --work $W && $PY $D evalsets --work $W
# 3. teacher cache (GPU): [70,1] everywhere, [70,13] optional
$PY $D teacher --work $W --teacher $AUDIOFORGE_0P6B_AFM --contexts 1 --batch 32
# 4. training: KD, then the control (each resumable; re-run the same line after an interruption)
$PY $D train --work $W --tag kd
$PY $D train --work $W --tag kd --control
# 5. evaluation -> runs/distill_0p6b.json; head check (VAD) on the new encoder
$PY $D eval --work $W --tag kd && $PY $D heads --work $W --tag kd
# 6. turn-head check on the new encoder (eot-bench v2, the served head's protocol):
$PY scripts/research/eval_stage1.py --bench v2 --ckpt $W/kd/student.afm   # compare with runs/stage1_served.afm
```

Bring back `runs/distill_0p6b.json`, `$W/kd*/history.json` and `$W/kd/student.afm` (443 MB).

## Estimated cost (one A100 80 GB; H100 ~1.5-2x faster; the two-5090 plan is below)

| stage | GPU hours | notes |
|---|---:|---|
| setup + downloads (AMI ~11 GB, ICSI 8 GB, LibriSpeech 7 GB, 0.6B 2.5 GB) | 0 | ~1 h wall |
| manifest + eval sets | 0 | CPU, ~15 min |
| teacher [70,1] over ~170 h (0.6B, bf16, batch 32; offline RTF ~0.002) | 0.4 | + 0.4 h for [70,13] (`--contexts 1,13`) |
| KD training, 20 k steps (~0.35 s / step incl. the frozen anchor copy) + 40 gate evaluations | 2.5 | ~10 passes over the meeting data |
| control (no teacher), same steps | 2.0 | |
| eval (3 models x 3 sets x 2 contexts) + VAD head check + eot-bench turn check | 0.5 | |
| **total** | **~6 h** | fits one GPU day with room for a second KD run (e.g. lr or w_enc sweep); disk ~60 GB peak |

## Two RTX 5090s (the user's box: 2 x 32 GB, Blackwell sm_120)

`setup.sh` installs PyTorch from the **cu128** wheels (CUDA >= 12.8 is required for sm_120) and refuses to continue
unless a kernel runs on every GPU and the capability reads (12, 0). Run `preflight.sh` first: it prints the GPU names,
VRAM, driver / torch / CUDA versions, free disk, RAM and whether the download hosts answer.

**Windows box (WSL2).** Install the NVIDIA Windows driver (R570 or newer, the one that supports the 5090) on the
**Windows host only**; do not install a Linux NVIDIA driver inside WSL. In an Ubuntu 22.04 / 24.04 WSL2 distribution
`nvidia-smi` must list both GPUs; clone the repo inside the Linux filesystem (`~/`, not `/mnt/c`, which is ~10x
slower for the data caches) and run everything else there exactly as below.

**Plan: one process per GPU, `CUDA_VISIBLE_DEVICES` pins each.**

```bash
# teacher cache split by shard across both GPUs (concurrently; each process takes every 2nd 256-item shard)
CUDA_VISIBLE_DEVICES=0 $PY $D teacher --work $W --teacher $AUDIOFORGE_0P6B_AFM --batch 24 --part 0/2 > teacher0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $D teacher --work $W --teacher $AUDIOFORGE_0P6B_AFM --batch 24 --part 1/2 > teacher1.log 2>&1 &
wait
# distilled student on GPU 0 and the no-teacher control on GPU 1, concurrently (same data, steps, effective batch)
CUDA_VISIBLE_DEVICES=0 $PY $D train --work $W --tag kd --batch 16 --accum 2 > kd.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $D train --work $W --tag kd --control --batch 16 --accum 2 > control.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY $D eval --work $W --tag kd && CUDA_VISIBLE_DEVICES=0 $PY $D heads --work $W --tag kd
```

- **Memory (32 GB, bf16 autocast):** micro-batch 16 items of <= 15 s with 2-step gradient accumulation keeps the
  effective batch of 32 of the A100 plan. The largest term is the RNNT joint (B x T x U x 1025); the head's
  `fused_batch_size` 2 already chunks it. The KD process also holds the frozen anchor copy (~0.5 GB) and the
  projector; peak ~22-26 GB expected. If a step runs out of memory, use `--batch 8 --accum 4` (same effective
  batch). Teacher: batch 24 at bf16 fits the 0.6B comfortably (~10 GB).
- **Throughput:** a 5090 is roughly 0.7-0.9x an A100 80 GB for this bf16 conv/attention mix (fewer tensor-core FLOPs
  in bf16 dense, similar memory bandwidth); micro-batching adds ~10 % per optimiser step.

| stage (two 5090s) | wall time | notes |
|---|---:|---|
| preflight + setup + downloads | ~1 h | network-bound, no GPU |
| manifest + eval sets | ~15 min | CPU |
| teacher [70,1] over ~170 h, split across both GPUs | **~20 min** | + ~20 min for [70,13] (`--contexts 1,13`) |
| KD (GPU 0) and control (GPU 1) concurrently, 20 k steps each + gates | **~3.5 h** | the two runs overlap fully |
| eval (3 models x 3 sets x 2 contexts) + VAD head check + eot-bench turn check | ~40 min | GPU 0 (GPU 1 free) |
| **total** | **~5.5-6 h wall (~9 GPU-hours)** | one working day with room for a second KD run (e.g. an lr or w_enc sweep on GPU 1 while GPU 0 evaluates) |

## Smoke test (done on the laptop, research/IMPROVEMENTS.md section 7)

~1 h of AMI / ICSI / LibriSpeech (`manifest --n-ami 12 --n-icsi 12 --max-hours 1`), teacher on MPS, a short KD
run, and the gate check with a deliberately high learning rate (`--lr 1e-2`), which must print `GATE FIRED` and
restore the last good checkpoint.

Licences: 0.6B teacher NVIDIA Open Model License; 115M CC-BY-4.0; AMI CC BY 4.0; ICSI (the ICSI release's terms,
research use); LibriSpeech CC BY 4.0.
