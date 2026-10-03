# Capacity sweep: voice_gender head (`hidden`), both cores

- Date: 2026-10-03
- Script: `scripts/research/voice_gender.py sweep` (the protocol of scripts/sweep_capacity.py: equal wall-clock budget per candidate, cosine LR over it, one seed, best held-out eval loss, knee = fewest parameters within tol of the best; that script has no registry entry for pooled classifier heads)
- Eval loss: class-balanced cross-entropy of P(male), mean of FLEURS dev and LibriSpeech dev-clean (20 utterances x 40 held-out speakers), after 2 s of pooled speech and on the full clip
- Train: FLEURS train (17 languages) + LibriSpeech train-clean-100 (251 speakers), class-balanced
- Other sizes fixed: att_hidden 64 (placeholder: never swept)

## 115m (block 4; budget 60 s per candidate on mps, batch 128, crop 75 frames, lr 0.002, seed 0, 5 evals)

| hidden | params | steps in budget | best eval loss | at t (s) | FLEURS dev bal. acc 2 s | LS dev bal. acc 2 s | |
|---|---|---|---|---|---|---|---|
| 16 | 11,426 | 4673 | 0.33294 | 12.0 | 96.2 | 95.5 |  |
| 32 | 21,762 | 4664 | 0.32659 | 12.0 | 96.5 | 95.5 | **knee** |
| 64 | 42,434 | 4170 | 0.35483 | 24.0 | 96.8 | 95.4 |  |
| 128 | 83,778 | 2735 | 0.35029 | 24.0 | 97.0 | 95.2 |  |
| 256 | 166,466 | 1536 | 0.35929 | 60.0 | 96.4 | 95.4 |  |

**Pick (115m):** hidden 32 (21,762 parameters); L_min 0.32659, tol 0.01. Shipped as assets/voice_gender_115m.pt (the pick's best-eval-loss state).

## 0p6b (block 5; budget 60 s per candidate on mps, batch 128, crop 75 frames, lr 0.002, seed 0, 5 evals)

| hidden | params | steps in budget | best eval loss | at t (s) | FLEURS dev bal. acc 2 s | LS dev bal. acc 2 s | |
|---|---|---|---|---|---|---|---|
| 16 | 20,642 | 2813 | 0.29300 | 12.0 | 96.4 | 95.0 | **knee** |
| 32 | 39,170 | 3154 | 0.34228 | 12.0 | 96.2 | 95.0 |  |
| 64 | 76,226 | 2848 | 0.34317 | 24.0 | 96.7 | 95.1 |  |
| 128 | 150,338 | 1879 | 0.34074 | 24.0 | 96.4 | 95.5 |  |
| 256 | 298,562 | 1227 | 0.30858 | 12.0 | 96.5 | 95.4 |  |

**Pick (0p6b):** hidden 16 (20,642 parameters); L_min 0.29300, tol 0.01. The pick is the smallest candidate: the knee may lie lower. Shipped as assets/voice_gender_0p6b.pt (the pick's best-eval-loss state).

## Tap (block) probe, before the size sweep

Linear probe (standardised logistic regression, class-balanced) on the LID caches' pooled block means, FLEURS train ->
FLEURS dev balanced accuracy, % (`plans/voice_gender/voice_gender_001.py`, log `runs/voice_gender/blocks.log`). The
speaker taps (115M block 4, 0.6B block 5) sit on the plateau; nothing beats them by more than ~0.5 points (about one
dev standard error), so the head reads the speaker tap (no extra encoder pass in the server).

| block | 115M 2 s | 115M whole | 0.6B 2 s | 0.6B whole |
|---|---|---|---|---|
| 1 | 95.4 | 94.2 | 96.1 | 94.5 |
| 2 | 95.9 | 94.8 | 96.8 | 95.7 |
| 3 | 96.3 | 95.7 | 96.9 | 96.1 |
| 4 (115M tap) | 96.0 | 95.6 | 96.7 | 96.4 |
| 5 (0.6B tap) | 96.1 | 96.2 | 96.5 | 96.9 |
| 6 | 95.5 | 95.7 | 96.5 | 96.9 |
| 7 | 95.4 | 95.6 | 97.0 | 97.3 |
| 8 | 95.2 | 95.8 | 97.0 | 97.3 |
| 9 | 95.3 | 96.2 | 96.9 | 97.4 |
| 10 | 95.4 | 95.6 | 96.2 | 96.3 |
| 11 | 91.2 | 94.1 | 96.2 | 96.2 |
| 12 | 85.2 | 87.7 | 96.1 | 96.2 |
| 13 | 78.1 | 81.5 | 96.2 | 96.3 |
| 14 | 73.4 | 75.7 | 96.0 | 96.1 |
| 15 | 69.6 | 68.4 | 96.2 | 96.5 |
| 16 | 66.5 | 67.7 | 96.1 | 96.5 |
| 17 | 61.6 | 62.7 | 95.9 | 96.5 |
| 18 | – | – | 95.9 | 96.8 |
| 19 | – | – | 96.0 | 96.4 |
| 20 | – | – | 95.0 | 96.1 |
| 21 | – | – | 93.6 | 93.8 |
| 22 | – | – | 91.4 | 90.5 |
| 23 | – | – | 86.2 | 85.5 |
| 24 | – | – | 80.2 | 77.5 |

Note: every size reaches the same held-out balanced accuracy (FLEURS dev 96-97 %, LibriSpeech dev ~95 %, the misses
are a few readers); the eval-loss spread between sizes is mostly how confidently those readers are missed.
