# The true single model: target-speaker and turn heads with more data (two RTX 5090s)

research/SINGLE_MODEL.md part B. `audioforge-serve --mode single` already runs everything from the one served 115M
checkpoint for a known user (research/SINGLE_MODEL.md part A). What limits it is the target-speaker (TS-VAD) head,
trained on 24 meetings with one print length, and the turn head, which was trained on diarizer tracks and reads the
TS-VAD track untrained. This package trains both with more speakers and data. It can optionally unfreeze the top
encoder blocks under a WER gate, and optionally distil nemotron-speech-streaming-en-0.6b into the ASR.

Product decision (2026-09-29): the product needs **target** diarization (is the known user talking, is the user's
turn over), not a general "who spoke when" for everyone in the room. That is why there is **no diarization teacher**
here. The Nemotron-3 distillation of the earlier plan is dropped.

**Invariant:** encoder blocks 1-4 and the speaker head (block 4) stay frozen. Every voice print a product has stored
stays valid, and the TS-VAD head's input never moves. With the default `--unfreeze 5` (blocks 13-17), every other
served tap below block 13 is also bit-identical: the LID head (blocks 8-12), the speaker head and the TS-VAD head.
`train` asserts that every tensor outside the trainable set is bit-identical to `runs/stage1_served.afm`.

## Pre-registered bars (written before any GPU run)

| # | what | bar | measured on | today (served single mode) |
|---|---|---|---|---|
| 1 | target-tracking F1 (P(target) vs the primary's labels, 5 s print from elsewhere in the meeting) | **≥ 0.80 AMI dev, ≥ 0.90 ICSI held-out** | eot-bench v2 windows, 974 / 1312 (`eval`) | 0.743 / 0.882 (research/IMPROVE_115M.md A.1) |
| 2 | turn-end misses at matched false cut-offs (≤ 5 % per-turn FC, cross-fitted, 6 s horizon), TS-VAD-fed turn head | **below today's single mode with a CI excluding 0**: hybrid_dyn 34.2 % AMI / 18.7 % ICSI; turn head alone 40.0 / 20.5 % | eot-bench v2 (laptop, step 7) | as left |
| 3 | streaming ASR WER, [70,1] | **LibriSpeech-200 ≤ 2.59 %** (2.29 + 0.3); AMI-200 not worse than 24.4 % + 1.0 | `eval` | 2.29 % / 24.4 % |
| 4 | VAD F1 (only if `--unfreeze` > 0) | ≥ 0.94 | 64 × 20 s AMI dev windows (`distill_0p6b_to_115m/distill.py heads`) | 0.949 |
| 5 | LID 2 s (only if `--unfreeze` > 5, which touches block 12) | ≥ 90 % FLEURS-17 | `scripts/research/lid.py eval_head` on the laptop | 91.2 % |

**Kill criteria:** the WER gate fires twice in a run (automatic: the run stops at the last passing state,
`"killed"` in `history.json`). Stage S1 does not lift AMI tracking F1 by ≥ 0.02 over the served head: then do not
run S2 or S3, because more data is not the lever and the head architecture is.

## What trains

| part | S1 (heads only) | S2 (`--unfreeze 5`) | S3 (`--unfreeze 5 --asr-kd`, optional) |
|---|---|---|---|
| TS-VAD head (0.26 M, init `runs/tsvad_spk.pt`) | yes | yes | yes |
| turn head + its speaker kernels (served, conditioned second pass) | yes | yes | yes |
| encoder blocks 13-17, VAD head, RNNT pred + joint | – | yes | yes |
| teacher | – | – | the 0.6B at [70,1]: transcript (RNNT) + top-layer cosine |

Tasks per step (sampled): **tsvad** is a 16 s AMI / ICSI train crop (60 %) or a simulated LibriSpeech-100 mixture of
2-3 of its 251 speakers at random offsets and gains, so overlap happens (40 %). The target is a speaker in the crop
(80 %) or one who is absent. The voice print is 1.5-10 s (uniform) of that speaker's single-speaker speech from
elsewhere, embedded by the frozen served speaker head. The print is dropped 15 % of the time, and the target then
becomes any speech. **turn** is an AMI / ICSI turn window (the served turn head's windows). The primary's print gives
the current TS-VAD track, which goes into the turn head exactly as `--mode single` serves it: [P(user), P(other), 0, 0],
primary 0, through the head's external-track input. **asr** runs with S2 / S3 only: AMI / ICSI segments with
RNNT(reference) [+ the 0.6B terms], plus a LibriSpeech anchor with KL to the frozen initial model. With unfreezing,
the tsvad crops also train the VAD head. The **WER gate** runs every 1000 steps on the first 200 LibriSpeech
test-clean and AMI-200 utterances at [70,1]. LibriSpeech above baseline + 0.3 or AMI above baseline + 1.0 restores
the last passing state and halves the LR. The second firing kills the run.

Data: AMI (136 train meetings, ~80 h) + ICSI (70 train meetings, ~70 h) + LibriSpeech train-clean-100 (100 h, 251
speakers). Eval: the eot-bench v2 windows (AMI dev 4 meetings, ICSI held-out 5 meetings) and the HYBRID_ASR.md
200-utterance sets. None of these is used for training.

## Steps (repo root on the GPU box; Linux or WSL2 with the Windows R570+ driver)

```bash
# 0. from the laptop (these carry this project's trained heads; not downloadable):
scp <laptop>:nvidia-audio-models/runs/{stage1_served.afm,tsvad_spk.pt} runs/
# 1. preflight (read-only), then environment + data (~1-1.5 h wall, network-bound; the 0.6B only for S3):
bash scripts/research/single_model_distill/preflight.sh
bash scripts/research/single_model_distill/setup.sh            # --no-teacher skips the 0.6B
P=scripts/research/single_model_distill/single_distill.py; W=work/single; PY=.venv/bin/python
# 2. manifest + eval sets (CPU, ~40 min: turn windows of 206 meetings)
$PY $P manifest --work $W && $PY $P evalsets --work $W
# 3. GPU 0: S1 heads only.  GPU 1 meanwhile: the 0.6B teacher cache for S3 (skip without S3), then the S1 control
CUDA_VISIBLE_DEVICES=0 $PY $P train --work $W --tag s1 --steps 20000 --batch 16 > s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 sh -c "$PY $P teacher --work $W --batch 24 && \
  $PY $P train --work $W --tag s1_meet --p-meeting 1.0 --steps 20000 --batch 16" > gpu1.log 2>&1 &
wait
# 4. eval S1 (+ its control: no LibriSpeech mixtures); check the S1 kill criterion before going on
CUDA_VISIBLE_DEVICES=0 $PY $P eval --work $W --tag s1 & CUDA_VISIBLE_DEVICES=1 $PY $P eval --work $W --tag s1_meet & wait
# 5. S2 (GPU 0) and S3 (GPU 1) from S1, concurrently; each resumable: re-run the same line after an interruption
CUDA_VISIBLE_DEVICES=0 $PY $P train --work $W --tag s2 --init $W/s1/good.pt --unfreeze 5 --steps 20000 > s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $P train --work $W --tag s3 --init $W/s1/good.pt --unfreeze 5 --asr-kd --steps 20000 > s3.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY $P eval --work $W --tag s2 & CUDA_VISIBLE_DEVICES=1 $PY $P eval --work $W --tag s3 & wait
# 6. VAD check on an unfrozen student (bar 4)
$PY scripts/research/distill_0p6b_to_115m/distill.py heads --work $W --tag s2 --student runs/stage1_served.afm --out runs/single_distill.json
```

The `train` micro-batch is 16 items of ≤ 20 s, in bf16. If a 32 GB card runs out of memory, use `--batch 8 --accum 2`
(same effective batch). Bring back `runs/single_distill.json`, `$W/*/history.json` and, for each tag,
`$W/<tag>/{student.afm,tsvad.pt}` (443 MB + 1 MB).

**7. Turn misses (bar 2), on the laptop**, where the eot-bench Silero and Sortformer caches live. Use one binding name
per (head, model) pair:

```bash
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/tsvad.py bind --heads s2=work/single/s2/tsvad.pt --bindings tsvad_s2 --budget 540   # repeat
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/tsvad.py scores --bindings tsvad_s2 --model work/single/s2/student.afm --device cpu --budget 540  # repeat
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/tsvad.py report --bindings tsvad_spk,tsvad_s2
```

**Serving the result:** `audioforge-serve --mode single --asr work/single/s2/student.afm --tsvad work/single/s2/tsvad.pt`.
The protocol does not change, and stored voice prints stay valid.

## Estimated cost on two RTX 5090s (estimates, not measured on a 5090)

Per-step times scale from the smoke test's MPS timing (below) and the A100 figures of `distill_0p6b_to_115m`. A 5090
does roughly 0.7-0.9x an A100 80 GB on this bf16 conformer mix.

| stage | where | GPU-hours | wall |
|---|---|---:|---:|
| preflight + setup + downloads (AMI ~11 GB, ICSI ~8 GB, LibriSpeech ~7 GB, 0.6B 2.5 GB) | CPU / network | 0 | ~1-1.5 h |
| manifest + eval sets | CPU | 0 | ~40 min |
| 0.6B teacher cache [70,1], ~170 h (S3 only) | GPU 1 | 0.4 | ~25 min |
| S1 heads only, 20 k steps × 16 (~0.25 s / step, gates cheap: ASR frozen) | GPU 0 | 1.5 | 1.5 h |
| S1 control (`--p-meeting 1.0`: no LibriSpeech mixtures) | GPU 1 | 1.5 | (parallel) |
| eval S1 + control (2 × 2286 windows + 3 × 200 × 2 WER sets) | both | 0.4 | ~12 min |
| S2 `--unfreeze 5`, 20 k steps (~0.4 s / step) + 20 gates | GPU 0 | 2.5 | 2.5 h |
| S3 `--unfreeze 5 --asr-kd`, 20 k steps (~0.45 s / step) + 20 gates | GPU 1 | 2.8 | (parallel) |
| eval S2 + S3 + VAD check | both | 0.5 | ~15 min |
| **total** | | **~9.6 GPU-hours** | **~7-8 h wall** (one working day) |

Disk: ~30 GB of data, a 16 GB teacher cache (S3), ~2 GB of checkpoints per tag. Checkpoints hold only the
trainable tensors and the optimiser, and are read back after writing.

## Smoke test (laptop, MPS, research/SINGLE_MODEL.md part B)

About 1 h of audio (2 AMI + 2 ICSI train meetings, 12 LibriSpeech speakers). Every stage runs once, and the gate is
checked with a deliberately high learning rate, which must print `GATE FIRED` twice and `KILLED` and restore the last
passing state: `bash scripts/research/single_model_distill/smoke.sh` (through `scripts/dev/gate.sh`, only when no
other MPS job runs).

Result on 2026-09-29: **passed in 226 s**. Every stage ran. S1 ran at ~0.5 s per step (batch 4, MPS). S3's first gate
fired at the smoke LR, restored the state, halved the LR, then passed. The forced test fired twice and ended KILLED at
the last passing state. Both exports were verified bit-identical outside the trainable set (730 / 529 tensors).
research/SINGLE_MODEL.md part B has the details.

Licences: 0.6B teacher NVIDIA Open Model License; 115M CC-BY-4.0; AMI CC BY 4.0; ICSI (the ICSI release's terms,
research use); LibriSpeech CC BY 4.0.
