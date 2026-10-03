# Perceived voice gender: an optional head on both cores (2026-10-03)

**What it is.** `--voice-gender head` adds a small head that reports two probabilities, *female voice* and *male
voice*: the two classes the training data annotate (the FLEURS `gender` field, LibriSpeech `SPEAKERS.TXT`). It is a
**perceived vocal characteristic estimated from audio, not a person's gender identity**. It can be wrong for any
individual voice (one held-out LibriSpeech reader below is called the other class on every one of their utterances),
and it must not be used to make decisions about people. It is off by default, never part of a default install, and
ships as separate files: `assets/voice_gender_115m.pt` (21.8 K parameters, 92 KB) and `assets/voice_gender_0p6b.pt`
(20.6 K, 87 KB), pinned in `audioforge/hub.py` as OPTIONAL components (`voice_gender`, `voice_gender_0p6b`).

Code: `audioforge/heads/voice_gender.py` (the head), `audioforge/voice_gender.py` (files, attach, the session
stream), `scripts/research/voice_gender.py` + `voice_gender_train.py` (features, sweep, test, latency),
`tests/test_voice_gender.py`. Raw numbers: `runs/voice_gender/results.json`; sweep table:
[plans/sweeps/voice_gender_2026-10-03.md](../plans/sweeps/voice_gender_2026-10-03.md).

## Architecture (lifted from published models)

| part | from |
|---|---|
| attentive statistics pooling: per-frame, per-channel attention weights, then the weighted mean and standard deviation | the decoder of ECAPA-TDNN (Desplanques et al. 2020), TitaNet (Koluguri et al. 2022) and AmberNet (Jia et al. 2023) |
| one linear layer on [mean, std] | AmberNet's decoder (pooling + linear classifier) |
| causal pooling (weights depend on the frame alone, `exp(5 tanh(a/5))`, so the posterior is a function of running sums and streams with constant state) | this project's LID head (`LanguageHead`, research/archive/LID.md) |
| LayerNorm + linear projection to `hidden` channels before the pooling | the only swept size |

It reads the speaker head's tap: block 4 on the 115M, block 5 on the 0.6B (1-based), so in the server it costs no
extra encoder pass. Only VAD-gated frames (served `heads.vad` > 0.5) are pooled, as for LID.

**Tap check.** A linear probe on the LID caches' pooled block means (FLEURS train -> FLEURS dev, all blocks,
`plans/voice_gender/voice_gender_001.py`, log `runs/voice_gender/blocks.log`): dev balanced accuracy at 2 s is flat
over blocks 2-10 on the 115M (95.2-96.3 %) and 2-9 on the 0.6B (96.5-97.0 %), and falls off at the top (115M block
17: 61.6 %; 0.6B block 24: 80.2 %). The speaker taps sit on the plateau (115M block 4: 96.0 %, 0.6B block 5:
96.5 %); no block beats them by more than ~0.5 points (about one dev-set standard error), so the head keeps the
speaker tap.

## Data and splits

| set | licence | role | rows | speakers |
|---|---|---|---|---|
| FLEURS train, 17 languages | CC BY 4.0 | train | 4250 (33 % male) | no speaker id |
| LibriSpeech train-clean-100 (the speaker head's cached frames, `cache/spk_frame`) | CC BY 4.0 | train | 6954 (49 % male) | 251 |
| FLEURS dev | CC BY 4.0 | selection | 1020 | no speaker id |
| LibriSpeech dev-clean, 20 utterances per speaker | CC BY 4.0 | selection | 800 | 40 |
| FLEURS test (all 17 languages; English alone) | CC BY 4.0 | test, once | 2550; 150 | no speaker id |
| LibriSpeech test-clean | CC BY 4.0 | test, once | 2620 | 40 |

Not used for training, by rule: oto (its card forbids speaker identification), TurnBench, AMI / ICSI speakers.
Training is class-balanced (each class half of every batch, then FLEURS / LibriSpeech with equal odds), random
crops of up to 6 s.

**Disjointness check** (`voice_gender.py check`, printed): LibriSpeech train speakers = the 251 train-clean-100
readers; overlap with dev-clean 0, with test-clean 0, dev-clean vs test-clean 0. FLEURS: no shared row ids and no
shared (language, sentence) between train, dev and test. **FLEURS publishes no speaker id, so speaker disjointness
of its splits cannot be checked here**; for the same reason its test CIs are bootstrapped over rows, which makes them
too narrow (a reader's rows are not independent; the identical female-row accuracy across windows and cores, 716 of
732, suggests the errors sit on a few readers).

## Size (equal wall-clock sweep)

Five `hidden` sizes, 60 s of training each on MPS, best held-out eval loss (class-balanced CE, FLEURS dev +
LibriSpeech dev-clean, 2 s and full), knee = fewest parameters within 1 % of the best:

| core | 16 | 32 | 64 | 128 | 256 | pick |
|---|---|---|---|---|---|---|
| 115M eval loss | 0.333 | **0.327** | 0.355 | 0.350 | 0.359 | hidden 32, 21.8 K params |
| 0.6B eval loss | **0.293** | 0.342 | 0.343 | 0.341 | 0.309 | hidden 16, 20.6 K params |

Balanced accuracy on the selection sets is flat across sizes (FLEURS dev 96.2-97.0 %, LibriSpeech dev 95.0-95.5 %
at 2 s); every size fits in 12-24 s and then over-fits (train CE ~0.002), so the best-eval state is the shipped one.
On the 0.6B the pick is the smallest candidate (the knee may lie lower; at 20 K parameters there is nothing to save).
The LibriSpeech dev errors are speaker-level (115M head): at 2 s, 36 of 800 rows, 20 of them one male-labelled reader (7976)
called female on every utterance, 10 another (2078).

## Test (public test rows, unseen speakers; touched once)

Accuracy / balanced accuracy / per-class accuracy, %, with 95 % bootstrap CIs (1000 resamples; over speakers for
LibriSpeech, over rows for FLEURS, see above). "1 s" / "2 s" = the posterior after that much pooled (VAD-gated)
speech; "full" = the whole clip.

**115M (`voice_gender_115m.pt`)**

| set | window | accuracy | balanced acc. | female | male |
|---|---|---|---|---|---|
| FLEURS-17 test (2550; 732 F / 1818 M) | 1 s | 95.7 [94.9, 96.4] | 96.3 [95.6, 97.0] | 97.8 [96.7, 98.7] | 94.8 [93.8, 95.8] |
| | 2 s | 95.7 [94.9, 96.4] | 96.3 [95.5, 97.0] | 97.8 [96.7, 98.7] | 94.8 [93.8, 95.8] |
| | full | 95.9 [95.1, 96.6] | 96.5 [95.7, 97.1] | 97.8 [96.7, 98.7] | 95.1 [94.2, 96.1] |
| FLEURS English test (150; 93 F / 57 M) | 1 s / 2 s / full | 100.0 | 100.0 | 100.0 | 100.0 |
| LibriSpeech test-clean (2620; 40 speakers) | 1 s | 97.0 [93.2, 99.3] | 97.1 [93.5, 99.4] | 95.7 [88.3, 99.6] | 98.5 [96.4, 99.8] |
| | 2 s | 97.7 [94.0, 99.8] | 97.8 [94.0, 99.8] | 96.3 [88.7, 100.0] | 99.3 [97.7, 100.0] |
| | full | 98.0 [94.3, 100.0] | 98.1 [94.4, 100.0] | 96.6 [89.2, 100.0] | 99.6 [98.6, 100.0] |

**0.6B (`voice_gender_0p6b.pt`)**

| set | window | accuracy | balanced acc. | female | male |
|---|---|---|---|---|---|
| FLEURS-17 test | 1 s | 95.8 [95.1, 96.5] | 96.4 [95.6, 97.1] | 97.8 [96.7, 98.7] | 95.0 [94.0, 96.0] |
| | 2 s | 96.2 [95.5, 96.9] | 96.7 [96.0, 97.4] | 97.8 [96.7, 98.7] | 95.6 [94.6, 96.5] |
| | full | 96.5 [95.8, 97.2] | 96.9 [96.1, 97.6] | 97.8 [96.7, 98.7] | 96.0 [95.1, 96.8] |
| FLEURS English test | 1 s / 2 s / full | 100.0 | 100.0 | 100.0 | 100.0 |
| LibriSpeech test-clean | 1 s | 96.8 [93.3, 99.1] | 96.9 [93.4, 99.1] | 95.1 [88.2, 99.1] | 98.8 [97.3, 99.8] |
| | 2 s | 97.6 [94.1, 99.6] | 97.7 [94.2, 99.6] | 96.3 [89.4, 99.7] | 99.0 [97.7, 99.9] |
| | full | 98.1 [94.8, 99.8] | 98.1 [94.9, 99.8] | 96.8 [90.1, 99.9] | 99.5 [98.7, 100.0] |

FLEURS English is 150 rows from few readers; 100 % there says little. The wide LibriSpeech CIs are the honest
picture of 40 test speakers: one or two misheard voices move the number by several points.

**Baseline.** No open voice-gender classifier is cached on this machine (the cached SpeechBrain models are a
speaker-verification ECAPA and a language-ID ECAPA, neither has a gender output), so none was run.

## Cost (the served engine, MPS)

`voice_gender.py latency`: the `--mode single` engine (ASR, VAD, turn, TS-VAD, LID) on the bundled 16 s call, one
engine with the head attached, sessions with and without the stream alternating (9 rounds each, order flipped per
round), chunk compute per 160 ms chunk on MPS. The machine was shared with other jobs during the run, so the
engine-level p50s move by several ms between rounds.

| core | engine p50, off -> on (best round) | Δ best / Δ median of rounds | the stream's own time per chunk |
|---|---|---|---|
| 115M | 34.18 -> 33.75 ms | −0.43 / −0.53 ms | 0.051 ms |
| 0.6B | 42.60 -> 42.99 ms | +0.39 / −0.09 ms | 0.076 ms |

**Gate (≤ +0.5 ms per chunk): met.** The engine A/B shows no increase beyond its ±0.5 ms noise, and the stream's
own measured time (it includes the batched head run at each final) is 0.05-0.08 ms per chunk. The first build ran
the head eagerly on every chunk and measured +1.64 ms (115M, two engines, best round; 0.49 ms of its own per chunk);
it was rewritten to keep only references to the tap frames per chunk and run the head once per reading (each final,
each stats message), which is what ships. Both runs: `runs/logs/vg_latency*_*.log`.

## Limitations

- Two classes only, because that is what the data annotate; voices outside the training distribution (children,
  many accents, whispering, pathological or synthetic speech, telephone audio) were not tested.
- The labels are dataset metadata about the recording, not ground truth about anyone.
- FLEURS speaker disjointness is unverifiable (no speaker ids); its CIs are row-level.
- Not tested on live calls or meetings (AMI / ICSI speakers were excluded from training and were not scored either).
