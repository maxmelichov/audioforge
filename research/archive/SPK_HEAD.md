# Speaker head: architecture or supervision?

2026-09-26. Our speaker head (attentive-stats pooling + AAM-softmax on the frozen 115M streaming FastConformer,
`from_layers: all`) scores **15.0 % all-pairs / 32.2 % within-meeting EER** on the AMI dev trials, against
TitaNet-L's 6.6 % / 8.2 % (BASELINES.md). Within a meeting it is barely better than the untrained
mean-pooled features (33.5 %). Two explanations compete:

- **Architecture:** the frozen ASR encoder's features do not carry speaker identity (ASR training makes the top
  layers speaker-invariant), so no head on top of them can do better.
- **Supervision:** the head was trained with AAM cross-entropy (190 output classes, but only the **45 speakers** of
  the 12 AMI train meetings ever appear), at weight 0.3, on 16 s multi-speaker turn windows labelled with the
  primary speaker. That is weak and noisy supervision; better targets on the same features might work.

The experiment separates them: (A) a **layer sweep** keeps the supervision (AAM over AMI ids) and changes only
which encoder layer the head reads; (B) **distillation** keeps the features (`from_layers: all`) and replaces the
supervision by a frozen TitaNet-L teacher (cosine loss on single-speaker segments from AMI, ICSI and LibriSpeech).
If no single layer helps but distillation does, the features were fine and the supervision was the limit; if
neither helps, the features are the limit.

Pre-registered kill lines: the sweep "wins" if any single layer reaches < 20 % within-meeting EER on AMI dev;
distillation "wins" if it reaches < 20 % within-meeting on both AMI dev and ICSI dev.

## Setup

- **Code.** `from_layers: [k]` single-layer taps and `layer_weights()` in `audioforge/model.py` (runtime and
  `FrozenTeacher` follow); `SpeakerHead(distill=..., aam_weight=...)` in `audioforge/heads/audio.py` (bit-identical
  without `distill`, tests/test_spk_head.py); `data.spk_teacher` attach (`data.attach_teacher`, keyed by
  `data.segment_id`) and `init.train_only` (head-only training) in `audioforge/train.py`; teacher cache
  `scripts/cache_titanet.py`; train / eval / report driver `scripts/spk_head.py`. Numbers: `runs/spk_head.json`.
- **Init.** Every variant starts from `runs/stage1_heads_pretrained.afm`, trains only `heads.spk` (+ its layer mix),
  batch 6, lr 1e-3, cosine schedule, MPS. All other heads have weight 0 and the driver asserts that every tensor
  outside `heads.spk` / `layer_mix.spk` is bit-identical to the init checkpoint after training (so the RNNT /
  CTC / encoder are untouched and no WER gate is needed).
- **Sweep (A).** `research/recipes/spk_layer_sweep_L{2,4,8,12,17}.yaml` (+ `_all` as the control): the stage-1 objective
  (AAM-softmax, 190 output classes) on the AMI train single-speaker segments (asr mode: 930 segments, **45
  speakers**, 1.8 h; cleaner than the turn windows), 1000 steps, head warm-started from stage 1. L{n} = output of
  encoder block n of 17 (`from_layers: [n-1]`). `research/recipes/spk_layer_sweep_L4_libri.yaml` is the identity-diversity
  control: the same AAM-only objective on block 4 with 5000 LibriSpeech train-clean-100 utterances (251 more
  speakers, ids 1000+) added -> 296 speakers / 1251 classes, same 1000 steps.
- **Distillation (B).** `research/recipes/spk_distill_titanet.yaml`: `distill: {target: spk_teacher, weight: 1.0}`,
  `aam_weight: 0`, 2000 steps on AMI train segments (weight 3), ICSI train segments (3249, 33 speakers, 3.3 h) and
  5000 LibriSpeech train-clean-100 utterances (≤ 12 s), single-source batches. Teacher: NVIDIA TitaNet-Large
  (CC-BY-4.0, `nemo_import.import_titanet`), one L2-normalised 192-d embedding per segment cached in
  `data/cache/titanet/<set>.npz`. `research/recipes/spk_distill_titanet_bestlayer.yaml` repeats it on the sweep's best layer (block 4).
  `research/recipes/spk_relational_titanet.yaml` ("relational", OUTSIDE.md §3): block 4, loss = AAM (AMI batches, 45
  speakers / 190 classes) + relational MSE between the student's and TitaNet's within-batch cosine matrices (1.0)
  + cosine term (0.5), same data and steps. The head's pooling is attentive statistics in every variant (it always
  was); only the loss and the input layer change, so the head's 0.50 M parameters and the shared encoder pass are
  the same for all rows.
- **Evaluation.** Exactly the BASELINES.md / STAGE1 §4 protocol: AMI dev asr-mode single-speaker segments with the
  recipe's seeded cap (n = 64: 15 speakers, 2016 trials, 487 within-meeting; n = 200: 19 900 trials, 5101 within),
  cosine scores, `metrics.eer`, all pairs and within-meeting pairs; the n = 64 numbers are cross-checked against
  `eval_stage1.eval_spk`. ICSI dev (Bmr021, Bns001; n = 64 / 200 of 975 segments, 12 speakers) is the held-out
  corpus: no variant saw ICSI dev, and the sweep never saw ICSI at all. Teacher-student cosine is the mean cosine
  between the head's and TitaNet's embedding of the same segment.

## Results

All numbers are EER in %, cosine scoring, `runs/spk_head.json`. "within" = within-meeting trials. Every trained row
starts from the stage-1 head, trains only `heads.spk`, and left the other 744 tensors bit-identical (asserted).
Train time on MPS: 4-5 min per 1000-step run, 8 min per 2000-step run.

| variant | input layer(s) | objective | train speakers / classes | AMI dev n=64 all / within | AMI dev n=200 all / within | ICSI dev n=64 all / within | ICSI dev n=200 all / within | teacher cos AMI / ICSI |
|---|---|---|---|---|---|---|---|---|
| untrained mean-pool, stage-1 mix (STAGE1 / BASELINES) | all, near-uniform | none | – | 19.9 / 33.5 | 22.6 / 36.8 | – | – | – |
| untrained mean-pool, best single block | block 4 | none | – | 16 / 30 | 19 / 34 | 23 / 28 | 24 / 29 | – |
| untrained mean-pool, top block | block 17 | none | – | 39 / 46 | 41 / 49 | 47 / 51 | 47 / 49 | – |
| **stage-1 spk head (served today)** | all (learned, near-uniform) | AAM 0.3, turn windows | 45 / 190 | 15.0 / 32.2 | 16.3 / 29.2 | 36.5 / 42.3 | 35.3 / 41.0 | -0.04 / -0.06 |
| sweep L2 | block 2 | AAM | 45 / 190 | 9.9 / 20.5 | 11.8 / 22.2 | 22.5 / 27.6 | 20.5 / 25.1 | -0.01 / -0.05 |
| **sweep L4** | block 4 | AAM | 45 / 190 | **8.5 / 15.7** | 11.0 / 19.1 | 14.4 / 17.4 | 14.8 / 17.6 | -0.02 / -0.05 |
| sweep L8 | block 8 | AAM | 45 / 190 | 11.9 / 27.5 | 14.0 / 27.3 | 17.4 / 19.2 | 17.8 / 19.3 | -0.01 / -0.06 |
| sweep L12 | block 12 | AAM | 45 / 190 | 23.7 / 39.0 | 27.0 / 40.6 | 42.0 / 44.0 | 41.5 / 43.0 | 0.01 / -0.01 |
| sweep L17 | block 17 (top) | AAM | 45 / 190 | 29.0 / 43.0 | 34.6 / 45.1 | 44.9 / 48.0 | 45.6 / 47.7 | -0.01 / -0.01 |
| sweep control "all" | all (learned) | AAM | 45 / 190 | 10.5 / 21.2 | 13.4 / 22.2 | 20.9 / 26.0 | 19.4 / 23.9 | -0.02 / -0.05 |
| control L4 + LibriSpeech ids | block 4 | AAM | 296 / 1251 | 12.4 / 22.7 | 13.9 / 28.8 | 16.7 / 20.9 | 15.9 / 19.1 | -0.02 / -0.08 |
| distill, all-layer mix | all (learned) | 1 - cos(TitaNet) | 329 / no ids | 9.9 / 19.9 | 12.3 / 23.1 | 5.2 / 6.2 | 7.8 / 8.4 | 0.41 / 0.55 |
| distill, block 4 | block 4 | 1 - cos(TitaNet) | 329 / no ids | 8.5 / 15.8 | 11.1 / 21.6 | 5.0 / 5.7 | 6.4 / 7.0 | 0.42 / 0.58 |
| **relational** (AAM + rel. MSE + cos) | block 4 | AAM + relational | 329 (45 with ids) / 190 | 8.5 / **14.4** | 11.3 / **19.8** | **4.7 / 5.2** | **6.4 / 7.0** | 0.31 / 0.51 |
| TitaNet-L (teacher, 22 M) | own encoder | – | VoxCeleb | 6.6 / 8.2 | 10.9 / 12.0 | 1.4 / 1.0 | 2.2 / 2.1 | 1 |

Sanity: the stage-1 row reproduces BASELINES.md / STAGE1 exactly (`eval_stage1.eval_spk` is run as a cross-check
for n = 64: 15.02 / 32.22). n = 64 within-meeting has 487 trials (about ±3 points), n = 200 has 5101.

### Untrained features per encoder block (mean-pooled, within-meeting EER %, AMI n=200 / ICSI n=200)

L1 35/33, L2 33/32, L3 33/30, L4 34/29, L5 35/30, L6 34/31, L7 35/35, L8 36/34, L9 37/36, L10 43/43, L11 46/46,
L12 48/47, L13 49/49, L14 48/48, L15 48/48, L16 50/49, L17 49/49. Speaker identity lives in blocks 1-9 and is
gone from block 10 up; the top of the ASR encoder is at chance (50 % = no information). Even untrained, mean-pooled
block 4 beats the trained stage-1 head on ICSI (29 vs 41 %).

### Learned layer-mix weights (softmax over 17 blocks)

- **Served stage-1 spk head:** 0.062 0.065 0.067 0.071 0.072 0.070 0.064 0.062 0.057 0.053 0.051 0.050 0.050 0.050
  0.051 0.052 0.053 (uniform = 0.059). It never selected the speaker-bearing blocks: 9 of its 17 inputs are at
  chance, and they get 46 % of the mass. The diar and vad mixes are equally flat.
- Sweep control "all" (1000 more AAM steps on clean segments): peak 0.089 at blocks 5-6, 0.042 at blocks 11-16.
- Distill, all-layer mix: 0.136 / 0.149 / 0.147 at blocks 4-6, 0.025 at blocks 11-16 - the teacher signal pushes the
  mix onto blocks 3-8 (64 % of the mass), which is what the sweep found by hand.
- Block-4 taps are one-hot by construction (`from_layers: [3]`).

### Teacher-student cosine (mean over held-out segments)

AAM-trained heads are uncorrelated with TitaNet (cos -0.01 to -0.08): a different embedding space, as expected.
The cosine-distilled heads reach 0.41-0.43 on AMI dev and 0.55-0.59 on ICSI dev (TitaNet's own same-speaker cosine
is 0.59 on AMI); the relational head, whose cosine term has weight 0.5, sits at 0.31 / 0.51. On the same trials
the distilled heads' same/different within-meeting cosine is 0.70 / 0.40 (AMI) and 0.69 / 0.25 (ICSI), against
0.75 / 0.60 and 0.73 / 0.67 for the stage-1 head, whose "different" pairs were almost as close as its "same" pairs.

### Cost

Unchanged: the head is the same 496 448 parameters (attentive-stats pool + linear + BN + 190x192 classifier, of
which the classifier is not used at inference), reading one frame sequence from the shared encoder pass. A
single-layer tap is cheaper than the 17-layer mix (one tensor instead of a 17-term weighted sum). Measured RTF of
encoder + head on the eval sets (CPU, 2 threads, batch 8) was 0.02-0.07 for every variant, dominated by the
shared encoder; the differences are load noise.

## Overlap robustness

Question: does a frame-level lower-layer embedding average to garbage in overlap, follow the louder speaker, or
keep the primary? `scripts/spk_overlap.py` on the 64 AMI dev turn windows (STAGE1 §1 set): frames are labelled per
speaker with the word-level activity, tiled into 0.96 s spans; **single** = ≥ 9 of 12 frames active and no
2-speaker frame (259 spans), **overlap** = ≥ 6 frames with ≥ 2 speakers active (125 spans). Each span is embedded
and compared (cosine) with every meeting speaker's clean enrollment = mean embedding of that speaker's single-speaker
segments elsewhere in the meeting (segments intersecting the window excluded). **"Louder" is a proxy: the speaker
with more labelled active frames in the span** (there is no per-speaker headset audio in the eval set), defined for
72 of the 125 overlap spans (the rest tie). "Primary" = the turn owner (present in 97 spans). Frame vectors:
`block k` = mean of that encoder block's frames; `head` = the trained head's attentive pool + embedding over the
span's frames from its own input layer; TitaNet on the span's audio.

| representation | single: nearest = active | single: cos active / absent | overlap: nearest = louder | nearest = primary | nearest = neither present | overlap: cos primary / other present / absent |
|---|---|---|---|---|---|---|
| block 4, mean-pooled (untrained) | 0.85 | 0.78 / 0.73 | 0.53 | 0.39 | 0.11 | 0.72 / 0.72 / 0.69 |
| block 8, mean-pooled (untrained) | 0.60 | 0.70 / 0.67 | 0.36 | 0.38 | 0.19 | 0.67 / 0.67 / 0.65 |
| sweep L4 head (AAM, block 4) | 0.83 | 0.69 / 0.49 | 0.53 | 0.36 | 0.11 | 0.60 / 0.60 / 0.51 |
| distill, all-layer mix | 0.92 | 0.71 / 0.52 | 0.49 | 0.40 | 0.10 | 0.60 / 0.61 / 0.52 |
| distill, block 4 | 0.92 | 0.72 / 0.51 | 0.49 | 0.38 | 0.10 | 0.59 / 0.61 / 0.51 |
| relational, block 4 | 0.91 | 0.74 / 0.55 | 0.51 | 0.34 | 0.12 | 0.62 / 0.63 / 0.54 |
| TitaNet-L on the audio | 0.97 | 0.51 / 0.13 | 0.67 | 0.40 | 0.01 | 0.27 / 0.27 / 0.12 |

Reading:
- **In single-speaker spans of 1 s** the distilled heads identify the active speaker among the meeting's 4 in 91-92 %
  of spans (TitaNet 97 %, untrained block 4 85 %, AAM head 83 %). One second of block-4 features is enough for
  identity; the servable head is close to the teacher there.
- **In overlap, every representation collapses toward "a present speaker" without preferring the primary.** For all
  of ours the cosine to the primary and to the other present speaker is equal (0.59-0.62 vs 0.60-0.63), i.e. the
  span embedding is an average of the two voices; TitaNet is the same (0.27 / 0.27). Nobody keeps the primary:
  nearest = primary is 0.34-0.40 for every row, at the chance level of "one of the two present" (given that the
  primary is the more-active speaker in only part of the spans).
- **Follow-the-louder is weak in ours (0.49-0.53, near coin-flip) and clearer in TitaNet (0.67)**; block 8 (0.36)
  is worse than block 4 in every column, so its mean is closer to garbage. Our heads land on an absent speaker in
  10-12 % of overlap spans against 1 % for TitaNet: the frame average does drift out of the present set at times.
- Our absent-speaker cosine stays high (0.51-0.54 vs TitaNet's 0.12): the head's space is less discriminative
  overall (its "different" pairs are still 0.4-0.5 apart, TitaNet's 0.1), which is the same fact as the EER gap.
- Implication for the turn stack: a frame-level embedding from block 4 is not a primary-keeping signal in overlap;
  it is a "who is present" average. Speaker binding in overlap has to come from the activity track (which speaker's
  frames to pool), not from the embedding; pooling only the frames the diarizer assigns to the primary (as
  `enrollment.py` does with columns) is the right use, and then the 92 % single-span accuracy applies.

## Verdict

**Both pre-registered kill lines are crossed, and the answer is "supervision, with a large architectural caveat".**

- **Sweep (A) wins: block 4 alone reaches 15.7 % within-meeting on AMI dev (n=64; 19.1 % at n=200)**, half the
  served head's 32 %, with the same objective, the same 45 speakers and 1000 head-only steps. So the frozen ASR
  encoder does carry enough speaker information; the stage-1 head was reading the wrong layers (a near-uniform mix
  in which more than half the inputs are speaker-blind) and training on multi-speaker windows. The "all" control
  on the same clean segments lands at 21-22 %: clean single-speaker data explains part of the gain (32 -> 21 %),
  the layer choice the rest (21 -> 16-19 %).
- **Distillation (B) wins on both corpora:** 19.9 % AMI (n=64) and 6.2-8.4 % ICSI with the all-layer mix; 15.8 % /
  5.7-7.0 % on block 4. The **relational** variant (AAM + relational MSE + cosine on block 4) is the best head
  overall: 14.4 / 19.8 % AMI within, 4.7 / 5.2 % ICSI n=64, 6.4 / 7.0 % ICSI n=200.
- **Objective vs identity diversity.** Adding 251 LibriSpeech identities to AAM-only on block 4 did not help (22.7 %
  vs 15.7 % on AMI, 20.9 vs 17.4 % on ICSI, with a 1251-way classifier that never converged in 1000 steps: loss 4.0
  vs 0.03). Distillation on the same block with the same extra data helps a lot on ICSI (17.6 -> 7.0 %) and is neutral
  on AMI (19.1 -> 21.6 %, within noise). So the ICSI gain is the teacher's objective, not the extra identities;
  cosine-only distillation vs AAM on AMI is a wash (consistent with OUTSIDE.md §3), and the relational term is
  what makes the combination beat both.
- **The architectural caveat is real and bounded.** No variant approaches the teacher on AMI within-meeting
  (best 14-16 % vs 8 %; at n=200 20 % vs 12 %), although on ICSI the distilled heads come within 5 points of TitaNet
  (5-7 % vs 1-2 %). AMI's far harder same-room pairs need spectral detail that the 80 ms, 8x-subsampled, ASR-trained
  block-4 features only partly keep. The frozen encoder is not the reason for 32 %, but it is the reason for the
  remaining 6-8 points.
- **The 0.04 teacher cosine of the served head, its 42 % on ICSI and its flat layer mix** together say the served
  head learned AMI room/channel priors rather than voices. The AMI-only gains of the sweep should therefore be read
  with the ICSI column: the block-4 AAM head generalises (17.6 % ICSI, never saw ICSI), the stage-1 head does not.

**What it implies for the single-model claim.** A speaker embedding from the shared frozen encoder is viable: a
0.5 M-parameter head on block 4 gets within 2x of a dedicated 22 M embedder on unseen meetings (ICSI 7 % vs 2 %,
AMI 20 % vs 12 % at n=200) at zero extra encoder cost, and the layer that carries it is available in the streaming
runtime (`layer_weights` / one-hot tap, `runtime.py`). It is not yet a replacement for TitaNet where within-meeting
precision matters most (AMI-like rooms), and it cannot become one without touching the encoder (the top-layer
features are speaker-blind, and blocks 1-9 are what they are). For enrollment / voice binding in the turn stack the
distilled head is now a usable label-free fallback; BASELINES.md's product recommendation (TitaNet for enrollment
when its 22 M / RTF 0.017 is affordable) still stands.

**Recommendation: transplant the relational head (`runs/stage1_spk_relational.afm`, also saved as
`runs/spk_relational_titanet.afm`) into the served model.** How: copy `heads.spk.*` from that checkpoint into
`runs/stage1_heads_pretrained.afm`, set `heads.spk.from_layers: [3]` in its config and delete `layer_mix.spk`
(the trained .afm itself carries weight-0 configs for the other heads, so do not serve it directly). Nothing else
changes: the encoder, RNNT, CTC, VAD, EOU, diar and turn tensors are bit-identical in every variant. If the
served model must stay AAM-only / teacher-free, use `runs/stage1_spk_L4.afm` (block 4, AAM) the same way.
Next levers, in order: (1) re-run the diar head with a block 3-8 tap instead of its equally flat all-layer mix
(same failure mode, STAGE1 §2); (2) distil with TitaNet at the frame level for the overlap case (below); (3) if the
AMI within-meeting gap matters, unfreeze blocks 1-4 with the WER gate on, which is where the speaker detail is.

### Files

`audioforge/model.py` (`from_layers: [k]`, `layer_weights`), `audioforge/heads/audio.py` (`SpeakerHead` distill /
relational / `aam_weight`), `audioforge/train.py` (`data.spk_teacher`, `init.train_only`, init.from re-sizing the
speaker classifier), `audioforge/data.py` (`segment_id`, `attach_teacher`), `audioforge/datasets/librispeech.py`
(`speaker_offset`), `audioforge/runtime.py` / `losses/consistency.py` (taps), `scripts/cache_titanet.py`,
`scripts/spk_head.py`, `scripts/spk_overlap.py`, `research/recipes/spk_*.yaml`, `tests/test_spk_head.py`,
`runs/spk_head.json`, `runs/stage1_spk_{L4,distill_titanet,distill_L4,relational}.afm`,
`data/cache/titanet/*.npz` (teacher embeddings: AMI train/dev, ICSI train/dev, 5000 LibriSpeech).
