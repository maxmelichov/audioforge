# Layer sweep: which block of the 0.6B each head should read

2026-10-01. Script: `scripts/research/core_0p6b_heads.py` (stages `sweep_vad`, `sweep_spk_feats` / `sweep_spk`,
`sweep_tsvad`, `sweep_seg_feats` / `sweep_seg`, `sweep_lid`, `sweep_report`). Results: `runs/core_0p6b.json`
(`sweep_*`). Context: research/CORE_0P6B.md.

## Summary

Each head's job peaks at a different depth of the 24-block encoder, and the chosen layer now differs per head:

| head | sweep says | shipped on the 0.6B | changed by the sweep? |
|---|---|---|---|
| VAD | blocks 2-4 and 13-15 are the single-block peaks (held-out mean AUC 0.973), the mix of all 24 0.972 | GRU head on the mix of blocks 4/8/12/16/20/24: held-out 0.977 with the full recipe, more than either block 4 (0.974) or blocks 4+12 (0.975) under the same recipe | no |
| speaker | block 4 (AMI 17.3 / ICSI 7.0 % within-meeting EER), 5 equal; block 11 20.6 / 9.4 % | block 5 (was block 11) | **yes**: the full recipe at block 5 gives AMI 13.6 % (block 11: 16.2 %) |
| TS-VAD | block 5 (target F1 0.751) / 6 / 10; the concat 5+6+10 0.758; block 11 0.704 | block 5 (shares the speaker head's tap; was 11) | **yes**: AMI target F1 0.760 vs 0.748 |
| v5 turn classifier | block 3 has the best held-out end-vs-pause AUC (0.790) but answers only 84 % of held-out smart-turn clips right; blocks 12-18 and the mix: AUC 0.76-0.79 at 96-100 % | block 12 (within noise of 13 / 18 on AUC, 100 % on held-out smart-turn clips, already cached) | no |
| LID | blocks 18-20 (FLEURS dev at 2 s 0.79-0.80), the concat 18+19+20 0.818 | softmax mix of blocks 16 / 20 (the frame caches held 8/12/16/20) | partly (follow-up: blocks 18-20) |

**Domain shift.** The VAD prefers different depths per corpus. AMI peaks at blocks 13-14 (held-out AUC 0.995); ICSI
peaks at blocks 2-4 (0.953). The early blocks generalise across rooms, the middle ones fit the training meetings.
Speaker and TS-VAD agree across corpora: blocks 4-6 are best on both. Their best blocks are shallow: identity sits
early and ASR training erases it with depth (block 24: 44 / 43 % EER, chance). LID is the opposite: language is a deep,
ASR-like feature.

**Cost of the choice.** None in encoder compute: the served pass runs all 24 blocks for the transcript anyway, and a
head reads one block or a mix of them from that pass. Head costs are part of the engine's 39 ms (MPS) / 96 ms (CPU) per
160 ms chunk (`cost_engine`). A concat input triples the input width of a head (e.g. TS-VAD 1024 → 3072 inputs:
+393 K params in its projection and pre-net). The served TS-VAD reads the speaker head's tap, so it stays single-block.

## Protocol

Same small probe per head for every configuration: (a) each block 1..24 alone, (b) a learned softmax mix of all
blocks, (c) the concatenation of the top-3 singles. Selection is on held-out data only, never the evaluation sets:

- **VAD:** the served FrameHead probe (Linear-SiLU-Linear 64), 1500 × 2048 frames, trained on the 300 AMI train windows
  minus 2 held-out meetings (TS3011b, ES2015c) + 15 % room tone. Scored on the held-out AMI meetings and 150 ICSI train
  windows, each with room-tone validation negatives (so firing on room tone costs AUC).
- **Speaker:** pooled mean + std of the block → Linear(·, 192), distilled to TitaNet-L (0.5 × cos + relational, the
  recipe's terms) on 200 LibriSpeech-100 speakers (3589 utterances). Scored by within-meeting EER on enrollment clips of
  the 24 AMI / ICSI train meetings (928 clips; never seen by the probe).
- **TS-VAD:** TSVADHead (hidden 128, prenet), 1000 × 32 windows, overlap weight 2.5. Enrollment is a clip from elsewhere
  in the meeting embedded by the 0.6B speaker head. Train = AMI / ICSI train windows minus held-out meetings (AMI
  TS3011b, ES2015c; ICSI Bro026, Bmr022); scored on those.
- **v5 turn classifier:** SegTurn (2 × 256, the c5 architecture), 500 × 256 steps, on 1000 turn clips (50 % oto / AMI /
  ICSI, 30 % smart-turn, 20 % cuts) cached with all 24 blocks.
  - Selection: turn_v5's held-out end-vs-pause AUC plus the held-out smart-turn clip accuracy (re-run for the leading
    blocks, `*_r`).
  - The smart-turn test accuracy (399 clips) is shown for information only.
  - The mix uses the 12 even blocks: 24 blocks of 1000 clips do not fit in RAM next to the model.
- **LID:** lid_fix's probe (standardise + logistic regression, C 0.1) on the 2 s onset-anchored block mean, fitted on
  FLEURS train (250 per language), scored on FLEURS dev.

The held-out sets are small (e.g. 2448 v5 samples; 2 AMI meetings for TS-VAD), so differences under ~0.01 AUC / F1
are noise. The decisions above changed a block only where the full recipe confirmed the probe.

## Tables

### VAD

| config | AMI held-out AUC | ICSI held-out AUC | AMI F1 | ICSI F1 | room-tone p95 | selection (mean AUC) |
|---|---:|---:|---:|---:|---:|---:|
| b1 | 0.9924 | 0.9397 | 0.9575 | 0.8751 | 0.064 | 0.966 |
| b2 | 0.9938 | 0.9529 | 0.9617 | 0.8865 | 0.037 | 0.9733 |
| b3 | 0.9942 | 0.9517 | 0.9625 | 0.8868 | 0.0338 | 0.9729 |
| b4 | 0.9938 | 0.9523 | 0.9614 | 0.8889 | 0.0331 | 0.973 |
| b5 | 0.9946 | 0.9486 | 0.9643 | 0.8851 | 0.0356 | 0.9716 |
| b6 | 0.9941 | 0.9467 | 0.9637 | 0.885 | 0.0363 | 0.9704 |
| b7 | 0.994 | 0.9444 | 0.965 | 0.887 | 0.025 | 0.9692 |
| b8 | 0.9936 | 0.9457 | 0.9614 | 0.8879 | 0.0246 | 0.9697 |
| b9 | 0.9935 | 0.9451 | 0.962 | 0.8894 | 0.0271 | 0.9693 |
| b10 | 0.9938 | 0.9407 | 0.9597 | 0.8869 | 0.0134 | 0.9672 |
| b11 | 0.9945 | 0.9431 | 0.9637 | 0.8878 | 0.022 | 0.9688 |
| b12 | 0.9945 | 0.9477 | 0.9653 | 0.891 | 0.0193 | 0.9711 |
| b13 | 0.995 | 0.9506 | 0.9662 | 0.8882 | 0.0182 | 0.9728 |
| **b14** | 0.995 | 0.952 | 0.965 | 0.8848 | 0.0227 | 0.9735 |
| b15 | 0.9939 | 0.9503 | 0.962 | 0.8893 | 0.0319 | 0.9721 |
| b16 | 0.9936 | 0.9444 | 0.962 | 0.8882 | 0.027 | 0.969 |
| b17 | 0.993 | 0.9385 | 0.9601 | 0.886 | 0.0172 | 0.9657 |
| b18 | 0.9935 | 0.9463 | 0.9616 | 0.8874 | 0.0342 | 0.9699 |
| b19 | 0.9937 | 0.9408 | 0.9623 | 0.8876 | 0.0466 | 0.9672 |
| b20 | 0.9938 | 0.9296 | 0.9637 | 0.886 | 0.0621 | 0.9617 |
| b21 | 0.9932 | 0.9234 | 0.964 | 0.8835 | 0.0747 | 0.9583 |
| b22 | 0.9914 | 0.9176 | 0.9602 | 0.881 | 0.0899 | 0.9545 |
| b23 | 0.9896 | 0.9077 | 0.9578 | 0.8782 | 0.0993 | 0.9486 |
| b24 | 0.9912 | 0.9083 | 0.9617 | 0.8754 | 0.1818 | 0.9497 |
| mix_all | 0.9951 | 0.9488 | 0.9644 | 0.8894 | 0.0254 | 0.9719 |
| concat 2+4+14 | 0.9947 | 0.9496 | 0.966 | 0.8806 | 0.0055 | 0.9722 |

selected by `val_auc`: **b14**

### speaker

| config | AMI within-meeting EER | ICSI within-meeting EER |
|---|---:|---:|
| b1 | 27.5 | 13.7 |
| b2 | 23.4 | 11.4 |
| b3 | 19.0 | 8.1 |
| **b4** | 17.3 | 7.0 |
| b5 | 18.0 | 6.6 |
| b6 | 19.0 | 6.8 |
| b7 | 18.1 | 7.1 |
| b8 | 18.4 | 8.3 |
| b9 | 22.4 | 10.4 |
| b10 | 19.7 | 8.3 |
| b11 | 20.6 | 9.4 |
| b12 | 21.9 | 10.6 |
| b13 | 22.9 | 11.7 |
| b14 | 27.0 | 15.2 |
| b15 | 28.0 | 21.9 |
| b16 | 30.5 | 23.9 |
| b17 | 32.5 | 23.9 |
| b18 | 33.6 | 27.4 |
| b19 | 35.9 | 30.7 |
| b20 | 38.8 | 33.4 |
| b21 | 38.9 | 35.4 |
| b22 | 42.2 | 38.0 |
| b23 | 43.3 | 40.3 |
| b24 | 44.1 | 43.3 |
| mix_all | 18.2 | 7.0 |
| concat 4+5+7 | 18.7 | 8.5 |

selected by `score`: **b4**

### TS-VAD

| config | AMI target F1 | AMI overlap recall | ICSI target F1 | ICSI overlap recall | selection (mean F1) |
|---|---:|---:|---:|---:|---:|
| b1 | 0.614 | 0.7007 | 0.815 | 0.6656 | 0.7145 |
| b2 | 0.6394 | 0.7304 | 0.826 | 0.6667 | 0.7327 |
| b3 | 0.6208 | 0.7314 | 0.8349 | 0.6924 | 0.7278 |
| b4 | 0.6362 | 0.742 | 0.8454 | 0.6506 | 0.7408 |
| b5 | 0.6679 | 0.7068 | 0.8334 | 0.6656 | 0.7507 |
| b6 | 0.6577 | 0.6797 | 0.8416 | 0.6763 | 0.7496 |
| b7 | 0.6323 | 0.6661 | 0.8286 | 0.6592 | 0.7305 |
| b8 | 0.6326 | 0.7165 | 0.8352 | 0.686 | 0.7339 |
| b9 | 0.6416 | 0.7075 | 0.8284 | 0.6549 | 0.735 |
| b10 | 0.645 | 0.8353 | 0.8403 | 0.6752 | 0.7427 |
| b11 | 0.5653 | 0.6535 | 0.8422 | 0.6806 | 0.7037 |
| b12 | 0.5786 | 0.7171 | 0.8339 | 0.686 | 0.7063 |
| b13 | 0.5571 | 0.6829 | 0.8178 | 0.6967 | 0.6875 |
| b14 | 0.5052 | 0.6729 | 0.8007 | 0.6635 | 0.6529 |
| b15 | 0.5409 | 0.64 | 0.7703 | 0.6967 | 0.6556 |
| b16 | 0.5385 | 0.7107 | 0.7801 | 0.716 | 0.6593 |
| b17 | 0.5145 | 0.7197 | 0.7735 | 0.6902 | 0.644 |
| b18 | 0.527 | 0.7081 | 0.7569 | 0.687 | 0.642 |
| b19 | 0.5142 | 0.6742 | 0.7095 | 0.6452 | 0.6119 |
| b20 | 0.5179 | 0.6871 | 0.6313 | 0.6195 | 0.5746 |
| b21 | 0.4516 | 0.6245 | 0.5501 | 0.5531 | 0.5009 |
| b22 | 0.4767 | 0.5967 | 0.5057 | 0.5263 | 0.4912 |
| b23 | 0.4 | 0.4265 | 0.4044 | 0.4126 | 0.4022 |
| b24 | 0.4131 | 0.3985 | 0.3558 | 0.3826 | 0.3845 |
| mix_all | 0.6485 | 0.7649 | 0.8439 | 0.7138 | 0.7462 |
| **concat 5+6+10** | 0.6702 | 0.7549 | 0.8451 | 0.6592 | 0.7576 |

selected by `score`: **concat_top3**

### v5 turn classifier

| config | end-vs-pause AUC | AUC, first quiet frames | held-out smart-turn clip acc % | end vs cut AUC | smart-turn test acc % |
|---|---:|---:|---:|---:|---:|
| b1 | 0.7549 | 0.74 | - | 0.8134 | 89.5 |
| b2 | 0.7785 | 0.7609 | - | 0.8202 | 91.5 |
| **b3** | 0.7895 | 0.7734 | - | 0.8382 | 92.7 |
| b4 | 0.7848 | 0.7612 | - | 0.7717 | 93.0 |
| b5 | 0.7705 | 0.751 | - | 0.7551 | 94.2 |
| b6 | 0.7581 | 0.7498 | - | 0.7548 | 95.2 |
| b7 | 0.7515 | 0.7384 | - | 0.7202 | 96.2 |
| b8 | 0.7586 | 0.7458 | - | 0.7874 | 96.2 |
| b9 | 0.7692 | 0.7584 | - | 0.7898 | 97.7 |
| b10 | 0.7546 | 0.7484 | - | 0.7717 | 99.0 |
| b11 | 0.7492 | 0.7352 | - | 0.826 | 98.7 |
| b12 | 0.7614 | 0.7453 | - | 0.8174 | 98.2 |
| b13 | 0.7736 | 0.7593 | - | 0.8392 | 98.5 |
| b14 | 0.7674 | 0.752 | - | 0.829 | 97.7 |
| b15 | 0.7666 | 0.7499 | - | 0.845 | 98.5 |
| b16 | 0.7735 | 0.7548 | - | 0.8223 | 98.0 |
| b17 | 0.7749 | 0.757 | - | 0.8091 | 98.5 |
| b18 | 0.7858 | 0.7683 | - | 0.824 | 98.2 |
| b19 | 0.78 | 0.7569 | - | 0.7772 | 98.5 |
| b20 | 0.7601 | 0.7436 | - | 0.7853 | 98.2 |
| b21 | 0.7555 | 0.7436 | - | 0.7737 | 98.5 |
| b22 | 0.7272 | 0.711 | - | 0.7764 | 97.0 |
| b23 | 0.7111 | 0.7043 | - | 0.7278 | 95.7 |
| b24 | 0.7497 | 0.738 | - | 0.761 | 95.7 |
| mix_even | 0.7872 | 0.7564 | - | 0.8292 | 98.7 |
| b3_r | 0.7895 | 0.7734 | 83.6 | 0.8382 | 92.7 |
| b12_r | 0.7614 | 0.7453 | 100.0 | 0.8174 | 98.2 |
| b13_r | 0.7736 | 0.7593 | 100.0 | 0.8392 | 98.5 |
| b16_r | 0.7735 | 0.7548 | 96.5 | 0.8223 | 98.0 |
| b17_r | 0.7749 | 0.757 | 96.9 | 0.8091 | 98.5 |
| b18_r | 0.7858 | 0.7683 | 96.2 | 0.824 | 98.2 |
| mix_even_r | 0.7872 | 0.7564 | 98.3 | 0.8292 | 98.7 |

selected by `auc`: **b3**

### LID

| config | FLEURS dev acc @2 s | dev acc, full |
|---|---:|---:|
| b1 | 0.2216 | 0.2569 |
| b2 | 0.2657 | 0.3265 |
| b3 | 0.3137 | 0.3941 |
| b4 | 0.3333 | 0.4245 |
| b5 | 0.4363 | 0.5167 |
| b6 | 0.5412 | 0.6304 |
| b7 | 0.6422 | 0.7471 |
| b8 | 0.7176 | 0.851 |
| b9 | 0.7618 | 0.8882 |
| b10 | 0.6735 | 0.8735 |
| b11 | 0.6284 | 0.8598 |
| b12 | 0.6255 | 0.848 |
| b13 | 0.6157 | 0.8461 |
| b14 | 0.6324 | 0.8382 |
| b15 | 0.7196 | 0.8794 |
| b16 | 0.7725 | 0.9 |
| b17 | 0.7745 | 0.9059 |
| b18 | 0.7882 | 0.9225 |
| b19 | 0.8 | 0.9529 |
| b20 | 0.7931 | 0.9529 |
| b21 | 0.7324 | 0.9588 |
| b22 | 0.6971 | 0.952 |
| b23 | 0.6539 | 0.9431 |
| b24 | 0.5931 | 0.9373 |
| **concat 18+19+20** | 0.8176 | 0.9422 |
| mix_all | 0.698 | 0.8804 |

selected by `dev_acc_2s`: **concat_top3**

## Follow-up

- **The same sweep for the 115M** (17 blocks; `LAYER_SWEEP_115M.md`) did not fit after the 0.6B evaluation. The VAD,
  speaker and LID caches of the 115M (`scratch/core_0p6b/vad/115m`, `cache/spk_frame`, `cache/lid_fix`) would cover
  three of the five heads without new encoding.
- **LID frames for blocks 18-20** (the sweep's best) and a retrain of the LID head on them.
