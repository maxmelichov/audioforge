# Spoken language identification: one front-end vs dedicated LID models

2026-09-27. Can the single-model front end (the frozen NVIDIA 115M streaming FastConformer we serve,
`runs/stage1_served.afm`, [70, 1] = 160 ms chunks) also tell which language is spoken, the way it serves VAD, speaker
and turn heads? The encoder is an **English** ASR model, so weak LID is a legitimate outcome. The question is measured
the way BASELINES.md measures the other tasks: same clips, same metric code for every system, CPU cost on 2 threads.

**Update 2026-09-29 ([fix pass](#fix-pass-2026-09-29)):** an AmberNet-distilled head on the same frozen encoder (`serve --lid head`) now scores **91.2 % at 2 s / 97.7 % on full utterances and 65.8 % on accented English**, at RTF 0.0013. That misses the pre-registered bar (92 / 98 %) by under a point, so LID stays off by default; the text below is the original study.

**Verdict: worse.** The best head reaches 89.3 % accuracy on full FLEURS utterances (17 languages) and 75.0 % at 2 s.
NVIDIA AmberNet reaches 99.5 % / 95.1 % and SpeechBrain's VoxLingua107 ECAPA 99.3 % / 94.6 %. On accented,
conversational English (EdAcc) the head says "English" for only 19-31 % of segments, against 68 % for AmberNet and
95 % for Whisper-base. The
head costs almost nothing (0.16 ms per 160 ms chunk, about 0.4 M parameters, no extra encoder pass). It is ahead of
Whisper-tiny only at 1 s of speech, and level with Whisper-base there. **Recommendation:** if LID is wanted in the served front end, use AmberNet
(`serve --lid ambernet`, now importable without NeMo). The head file stays an opt-in option. LID stays **off by
default**.

## Data and licences

| set | use | languages | per language | licence (verbatim from the source) |
|---|---|---|---|---|
| **FLEURS** (`google/fleurs`, HF) | train / dev / test | 17 (below) | train 250 utts (38-59 min), dev 60 (9-15 min), test 150 (23-40 min) | card metadata `license: cc-by-4.0`; README: "All datasets are licensed under the [Creative Commons license (CC-BY)](https://creativecommons.org/licenses/)." |
| **EdAcc** (`edinburghcstr/edacc`, HF) | accented-English probe (test only) | en | 257 segments ≥ 2 s, 55 min, 16 speakers, 11 L1s (5 row groups of the test split) | README: "Public Domain, Creative Commons Attribution-ShareAlike International Public License ([CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/deed.en))" |

- **Languages (17):** the 7 required ones plus 10 common ones picked for hard pairs. They are en (en_us), he (he_il), ar
  (ar_eg, Egyptian), ru, es (es_419), fr, de, pt (pt_br), it, nl, pl, uk, tr, fa, hi, zh (cmn_hans_cn), ja. Hard pairs
  in the set: he/ar, ar/fa, es/pt/it, ru/uk/pl, de/nl, hi/fa, zh/ja. Chance is 5.9 %.
- **Splits.** FLEURS train / dev / test are speaker-disjoint (by construction of FLEURS; the sentences differ too).
  Each split is the **first N utterances of its audio tarball**, streamed from the HF repo; the rest of the tarball
  is never downloaded (`scripts/lid_data.py fetch`, 1.6 GB on disk). A tarball prefix holds few speakers. The test
  prefix is **single-gender for 8 of 17 languages** (he, de, pt, nl, uk, tr, fa all male; ar all female; it is 142/8).
  Per-language test accuracy is therefore closer to "1-3 speakers per language" than to a language average. The
  same caveat hits every system equally.
- **Levels.** FLEURS recordings range from -65 to -9 dBFS RMS (peak 0.005 to 1.0), so every clip is peak-normalised to
  -1 dBFS before any system sees it (`lid_data.load_audio`, stored as 16-bit FLAC).
- **Clips.** For every test utterance: the first **1 / 2 / 3 / 5 s** from the speech onset, and the **full**
  utterance. The onset is Silero VAD v5 (the first 3 consecutive 32 ms chunks with p ≥ 0.5) minus 0.1 s. The median
  onset is 1.0 s, so "2 s" means 2 s of audio starting at speech, pauses included. Every system gets exactly these
  clips. Utterances shorter than onset + N s give a shorter clip.
- **Not used.** VoxLingua107: its licence was not checked here, and the dedicated models were trained on it, so it
  cannot be a neutral test set. Common Voice is gated. **No code-switched set was readily available** (no licensed,
  ungated one on HF at a size the Mac can handle), so there is no code-switching result. The VoxPopuli
  `en_accented` shards are 2.5 GB row groups, too large for the disk budget; EdAcc replaces them.
- **Disk.** The Mac had 0.4-3 GB free during this work, shared with other agents. Whisper-base was run through the
  locally cached faster-whisper (CTranslate2) conversion instead of downloading the transformers weights.

## 1. Probe: which encoder block carries language?

Untrained per-block probe of the frozen served encoder: mean-pool one block's output, standardise, multinomial
logistic regression (C = 0.1). It is fitted on the 4250 train utterances and scored on the 2550 test utterances.
One encoder pass runs over the 5 s clip from the onset. "2 s" pools its first 25 frames, which is the streaming
view: the encoder is causal with one 80 ms frame of lookahead. Accuracy in %, `runs/lid.json["probe"]`.

| block | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 s | 19.9 | 23.4 | 27.3 | 33.2 | 38.3 | 49.4 | 59.7 | 66.2 | **68.3** | 67.8 | 63.8 | 59.7 | 55.2 | 52.9 | 49.8 | 47.6 | 44.3 |
| 5 s | 23.4 | 29.2 | 34.6 | 41.7 | 50.1 | 63.4 | 72.9 | 81.5 | 82.9 | 85.5 | **86.6** | 86.1 | 85.0 | 82.4 | 80.2 | 78.0 | 76.5 |

- Language lives in **blocks 8-13**, above the speaker blocks (1-9, SPK_HEAD.md) and below the top blocks, which
  specialise to English tokens. The best block is **above the ~70 % bar at 5 s (86.6 %)** and just under it at 2 s
  (68.3 %). The 0.6B encoder was therefore not probed, as the pre-registered rule said.
- Even the top block (17) keeps 76.5 % at 5 s. An English-only ASR encoder still separates 17 languages, because
  phonotactics and prosody survive into its representation.

## 2. The head

`heads.audio.LanguageHead` (`type: language`) has four parts:
- a per-frame MLP (LayerNorm, 512→256, ReLU, 256→256, ReLU);
- **causal attentive-statistics pooling**: per-frame, per-channel weights `w_t = exp(5 tanh(a(h_t)/5))`, running
  weighted mean and std over *all frames so far*;
- a classifier (512→256→17);
- 0.40 M parameters in total.

The pooling weights depend on frame t only (no utterance-level context as in `AttentiveStatsPool`). The posterior
after any frame is therefore a function of running sums: `running_logits` (offline cumulative sums) equals `step`
(streaming carried sums) to 1e-5 (test). It emits a posterior every 160 ms chunk from the chunk's own per-layer
encoder outputs. The loss is cross-entropy of the running posterior averaged over every frame from 0.48 s on (an
anytime classifier), plus the cross-entropy at the last frame.

- **Input.** A learned softmax mix of blocks 8-12 (`from_layers: [7..11]`). It converged to 0.70 / 0.13 / 0.05 /
  0.06 / 0.07, so mostly block 8.
- **Training.** Head only, MPS, 3500 steps, batch 32, random 1-8 s crops (a quarter start at the speech onset), gain
  -6..0 dB, AdamW 2e-3 one-cycle. The best step is picked by dev accuracy (2 s + full). Two variants:
  - `lid_head`: no input augmentation, head dropout 0.1. Best step 1750, dev 72.6 / 87.1 %, 17.8 min.
  - `lid_aug`: SpecAugment (2×15 mel bins, 4×5 % time) on the frozen encoder's input plus the encoder's own dropout.
    `model.train()` switches only dropout / dither / masking on; no weight changes. Head dropout 0.3. Best step 2500,
    dev 73.4 / 88.4 %, 18.6 min. **This is the dev-selected head.**
- **Frozen check.** After training, all **759** tensors of the served model are asserted bit-identical to
  `runs/stage1_served.afm`. The only new tensors are `heads.lid.*` + `layer_mix.lid`. ASR / VAD / speaker / turn are
  untouched, so no WER gate is needed. The head is saved alone (`audioforge.lid.save_head`, 1.6 MB: `runs/lid_aug.pt`,
  `runs/lid_head.pt`, gitignored like every runs/*.pt) and attached at load (`audioforge.lid.attach_head`).
- **Overfitting.** Train loss falls to 0.02-0.3 while dev accuracy plateaus from about step 1000. With 250
  utterances from a few speakers per language, the head learns speakers and channels as much as languages. The
  EdAcc result below is the symptom.

## 3. Head vs dedicated models (FLEURS test, 17 languages, n = 2550)

Accuracy / macro-F1 in %. Every system's posterior is restricted to the 17 languages and renormalised. Hebrew is `iw`
in the VoxLingua107 label sets and is mapped. 95 % bootstrap CIs over utterances are about ±1.5 points at 75 % and
±0.3 at 99 % (`runs/lid.json["systems"][*][*]["acc_ci95"]`). With few speakers per language the effective n is
smaller.

| system | params | licence | 1 s | 2 s | 3 s | 5 s | full |
|---|---|---|---|---|---|---|---|
| probe, block 9 / 11, logistic regression (untrained) | 8.7 k | ours | – | 68.3 / 67.5 | – | 86.6 / 86.5 | – |
| **head `lid_aug` (ours, dev-selected)** | 0.40 M head, shared encoder | ours | 59.2 / 58.6 | 75.0 / 74.6 | 81.6 / 81.3 | 85.7 / 85.7 | 89.3 / 89.1 |
| head `lid_aug`, VAD-gated (the server's pooling) | same | ours | 59.1 / 58.4 | 75.0 / 74.7 | 81.3 / 81.0 | 85.8 / 85.8 | 88.7 / 88.6 |
| head `lid_head` (no augmentation) | same | ours | 58.2 / 57.8 | 72.6 / 72.5 | 79.7 / 79.8 | 83.8 / 84.1 | 87.7 / 87.9 |
| **NVIDIA langid_ambernet** (ported, `nemo_import.import_ambernet`) | 28.9 M | NGC Terms of Use | **83.9 / 83.8** | **95.1 / 95.1** | **98.2 / 98.2** | **99.2 / 99.2** | **99.5 / 99.5** |
| SpeechBrain lang-id-voxlingua107-ecapa | 21.2 M | Apache-2.0 | 81.1 / 80.9 | 94.6 / 94.5 | 97.3 / 97.2 | 98.5 / 98.5 | 99.3 / 99.3 |
| Whisper-tiny (transformers, language token) | 37.8 M | Apache-2.0 | 54.9 / 52.6 | 78.2 / 77.4 | 86.9 / 86.5 | 93.7 / 93.6 | 97.0 / 97.0 |
| Whisper-base (faster-whisper CT2, `detect_language`) | 74 M | Apache-2.0 (MIT code) | 61.5 / 58.5 | 83.6 / 82.7 | 90.9 / 90.5 | 96.3 / 96.3 | 98.5 / 98.5 |

- **The gap is large at every duration.** Against AmberNet the head is 25 points behind at 1 s, 20 at 2 s, 13 at 5 s
  and 10 on full utterances. The dedicated models are out-of-domain on FLEURS (trained on VoxLingua107 YouTube
  audio). The head is **in-domain** (trained on FLEURS train speakers of the same recording campaign), and it still
  loses.
- **The head is ahead of Whisper-tiny only at 1 s** (59.2 [57.3, 61.1] vs 54.9 [53.1, 57.0] %), level with
  Whisper-base there (61.5 [59.7, 63.5]), and behind both from 2 s on. Whisper always pads to 30 s, so its cost does
  not shrink with the clip.
- The trained head is **no better than a linear probe on one block** at 5 s (85.7 vs 86.6 %). It is better at 2 s
  (75.0 vs 68.3 %), where the multi-block input and the anytime loss help. More head capacity is not the lever; data
  diversity is.
- **Per language (head `lid_aug`, 2 s / full):** en 97 / 100, zh 98 / 100, ar 92 / 99, tr 92 / 98, nl 93 / 99, fr
  90 / 100, hi 88 / 98, pl 75 / 96, ru 73 / 85, it 71 / 90, es 69 / 90, pt 67 / 93, fa 63 / 86, ja 61 / 85, **he 53 /
  72**, **uk 49 / 69**, **de 45 / 57**. The weakest languages are exactly those whose test prefix is one male
  speaker group (de, he, uk). AmberNet's weakest are de 81 / 99 and he 89 / 95.

### Confusion on the hard pairs (fraction of the first language classified as the second, full / 5 s)

| pair | head `lid_aug` | head `lid_head` | AmberNet | SpeechBrain |
|---|---|---|---|---|
| he → ar | 10.0 / 7.3 % | 5.3 / 3.3 % | 5.3 / 5.3 % | 1.3 / 4.7 % |
| uk → ru | **28.0 / 33.3 %** | 20.7 / 42.0 % | < 2 % | < 2 % |
| ru → uk | 12.7 / 18.0 % | 15.3 / 14.7 % | < 2 % | 2.0 / 3.3 % |
| es → it | 10.0 / 14.7 % | 5.3 / 7.3 % | < 2 % | < 2 % |
| it → pt | 2.7 / 5.3 % | 14.7 / 12.0 % | < 2 % | < 2 % |
| fa → hi | 5.3 / 10.7 % | 11.3 / 11.3 % | < 2 % | < 2 % |
| fa → ar | 4.7 / 4.7 % | < 2 % | < 2 % | < 2 % |
| es → pt | < 2 % | 1.3 / 2.0 % | < 2 % | < 2 % |

(< 2 % = below the report's listing threshold. The full 17×17 matrices are in `runs/lid.json`.) The head's errors
are the linguistically close pairs: East Slavic ru/uk above all, then Romance and Indo-Iranian. Hebrew/Arabic, the
pair asked about, is 5-10 % for the head and about 5 % for AmberNet, the one pair AmberNet also confuses. Whisper
shares the head's weak pairs: he → ar 10 / 15 % (tiny) and 5 / 11 % (base), uk → ru 18 / 32 % and 12 / 21 %.

### Accented English (EdAcc, n = 257, fraction classified "en" among the 17)

| system | 1 s | 2 s | 3 s | 5 s | full | top wrong languages (full) |
|---|---|---|---|---|---|---|
| head `lid_aug` | 10.9 | 13.2 | 15.6 | 16.7 | 18.7 | fa 54, ar 42, tr 33, hi 31 |
| head `lid_head` | 17.1 | 21.8 | 23.7 | 24.1 | 31.1 | hi 44, fa 39, ar 20, nl 14 |
| AmberNet | 30.3 | 47.1 | 56.4 | 60.7 | 67.7 | fa 14, pt 11, he 10, it 10 |
| SpeechBrain ECAPA | 30.7 | 43.2 | 45.5 | 58.8 | 70.0 | pt 24, he 13, pl 9, it 8 |
| Whisper-tiny | 87.5 | 93.8 | 94.2 | 92.6 | 93.8 | pt 6, ru 3, ar 2, pl 2 |
| **Whisper-base** | **87.9** | **92.6** | **94.5** | **96.1** | **95.3** | pt 3, it 2, pl 2, ar 1 |

EdAcc is conversational, far-field and accented (L1s include Hebrew, Spanish, Italian, Vietnamese, Nigerian and
Kenyan English), so it is hard for everyone. The head, however, **fails**: 69-81 % of accented English segments are
called another language, mostly Persian / Arabic / Hindi / Turkish. The per-L1 split (head `lid_head`, full) shows
it: mainstream US English 100 %, Mandarin 83 %, Tagalog 100 %, but Hebrew-L1 26 %, Spanish-L1 32 %, Italian-L1 0 %.
The head learned "FLEURS-English", a read-speech recording condition, together with the language. The augmented
head is better in-domain and **worse** out-of-domain (19 vs 31 %). The VoxLingua107 models are much better but not
robust either: AmberNet calls 32 % of accented English segments another language on full segments and 53 % at 2 s
(mostly Persian, Portuguese, Hebrew, Italian: often the speaker's L1). **Whisper is the robust one on accented
English (94-95 % at every length ≥ 2 s)**, because it was trained on far more, and more varied, English. Its
English prior is also why it is weak on short clips of other languages (FLEURS 1 s: 55-62 %).

## 4. Streaming: the server's announcement rule

`serve --lid` announces the top language once its posterior reaches the threshold after ≥ 1 s of pooled speech, and
again on a change (`audioforge.lid.LangDecider`). This was simulated on whole utterances streamed from the file start
(`scripts/lid.py decide`, 20 utterances per language = 340 per split; the time includes the ~1 s of leading silence).
Head: VAD-gated running posterior with a 30 s half-life. AmberNet backend (`AmberNetLIDStream`): re-classifies the
last 8 s of VAD speech at 1, 1.5, 2, 3, 5 and 8 s of pooled speech, then every 4 s.

| backend, threshold | announced | first announcement correct | first announcement at (p50 / p90) | flips per utterance |
|---|---|---|---|---|
| head `lid_aug`, 0.9 (test) | 96.8 % | 78.7 % | 2.40 / 4.58 s | 0.17 |
| head `lid_aug`, 0.95 (test) | 90.6 % | 86.4 % | 2.72 / 5.04 s | 0.08 |
| **AmberNet, 0.9 (test)** | 99.4 % | **96.8 %** | **2.16 / 3.78 s** | **0.03** |
| AmberNet, 0.8 (test) | 100 % | 95.0 % | 2.16 / 3.69 s | 0.05 |

On dev the ordering is the same (head 0.9: 76.2 % correct at 2.56 s; AmberNet 0.9: 95.9 % at 2.20 s). The head is
over-confident: at a 0.9 posterior its first call is wrong one time in five. AmberNet is both earlier and right.

## 5. Cost (CPU, 2 threads, this Mac)

`scripts/lid.py cost`: median of 5 calls after a warm-up, 1-min load 4.5-8.4 (the Mac is shared; numbers under load
are upper bounds. An earlier run at load 6-14 measured about 1.5-2× these).

| system | cost per decision | notes |
|---|---|---|
| **head** | **0.16 ms per 160 ms chunk** (RTF 0.001), 0.40 M params | reads the chunk's per-layer encoder outputs; the encoder (23 ms per chunk) is already paid by ASR |
| AmberNet (fast depthwise path) | 16 / 25 / 36 / 56 / 81 ms per call on 1 / 2 / 3 / 5 / 8 s | 28.9 M params. The served schedule (≈ 6 calls in the first 8 s of speech, then one per 4 s): RTF ≈ 0.02-0.03 |
| SpeechBrain ECAPA | 10 / 15 / 19 / 30 / 43 ms | 21.2 M params |
| Whisper-tiny | ~63 ms at any length | pads to 30 s |
| Whisper-base (CT2) | ~155 ms at any length | |

- **AmberNet port.** `import_ambernet()` fetches the official NGC file (`nvidia/nemo/langid_ambernet` v1.12.0, 116 MB,
  guest download; not on Hugging Face). It loads **all 157 NeMo tensors except `loss.weight` and the two preprocessor
  buffers, `strict=True`** into `baselines.lid.AmberNet`. That is the TitaNet `ConvEncoder` port (SE separable
  blocks, 1024 channels, kernels 3/7/11/15) plus an x-vector decoder (mean + unbiased std, Linear 6144→512, BN
  without affine, ReLU, Linear 512→107). Validation is behavioural: 99.5 % on FLEURS full utterances in 17 languages
  (card: 5.22 % error on the VoxLingua107 eval set of 33 languages). PyTorch runs the 1024-group depthwise convs as
  16 k small convs per forward (~200 ms). `baselines.lid.fast_depthwise` rebinds them to unfold × w: logits within
  4e-6, 2.7× faster.
- The SpeechBrain and Whisper numbers use the packages' own inference (`EncoderClassifier.classify_batch`; the
  language-token logits after `<|startoftranscript|>`; faster-whisper `detect_language`).

## Verdict and recommendation

- **Accuracy: worse than the dedicated models**, at every duration and on both test sets. AmberNet and SpeechBrain
  ECAPA are 20 points better at 2 s and 10 points better on full utterances. The head is also far less robust
  (accented conversational English 19-31 % vs 68-70 % for the VoxLingua107 models and 94-95 % for Whisper).
- **Cost: much cheaper.** The head adds 0.16 ms per chunk to a pass the ASR already runs, against 16-81 ms per
  AmberNet call. At the served schedule AmberNet adds RTF ~0.02-0.03 on top of the front end's ~0.5. That is
  affordable.
- **Why the head loses.** The frozen encoder does carry language (86.6 % at 5 s from one block, untrained). But its
  features are English-ASR features, and 42 min per language from a few FLEURS speakers is too little diverse
  supervision to separate language from speaker and channel. The same lesson as SPK_HEAD.md applies: the features
  are usable, the supervision is the limit. Levers that could close part of the gap without touching the encoder,
  none tested: many more speakers per language (e.g. VoxLingua107, whose licence must be checked first), or
  **distilling AmberNet's posteriors on unlabelled multilingual audio** (the SPK_HEAD relational / distillation
  result). Even then, 2 s accuracy starts from a 68 % linear probe, against 95 % for a dedicated model.
- **Recommendation for the served default: LID stays off by default.** When a product needs it, run
  `serve --lid ambernet` (96.8 % first-call accuracy on FLEURS at a 2.2 s median, 0.03 flips per utterance, RTF
  +0.02-0.03), with `--lid-langs` restricted to the languages the product actually supports. **Caveat for
  accented-English users:** AmberNet calls a third of accented English segments another language (EdAcc). A product
  whose users are mostly non-native English speakers should restrict the label set and raise `--lid-threshold`, or
  use Whisper's detector (95 % on EdAcc, ~155 ms per call for base; not wired into the server). The head
  (`--lid runs/lid_aug.pt`) is for a near-zero-cost hint only, over a small, known language set, where a wrong first
  guess is cheap. In the scorecard: **Language ID: worse** (single model 75 % / 89 % vs AmberNet 95 % / 99.5 % at
  2 s / full), with AmberNet shipped inside the runtime, as Sortformer is for diarization.

## Fix pass, 2026-09-29

Request: "fix the language detector". **Pre-registered bar** for a served LID from the shared encoder pass (no AmberNet
at inference): ≥ 92 % at 2 s and ≥ 98 % on full utterances on FLEURS-17 (n = 2550), ≥ 60 % English recall on EdAcc,
≤ 0.01 added RTF. **Kill criterion:** if the best head is still < 88 % at 2 s, ship the transcript-text fusion
instead. Driver: `scripts/research/lid_fix.py`; numbers: `runs/lid.json["fix_2026_09_29"]` (the official head rows
also in `runs/lid.json["systems"]` / `["edacc"]` under `lid_distill`).

**Result: large gain, bar missed on FLEURS by a hair, met on EdAcc and cost.** The dev-selected head
(`lid_distill`, AmberNet-distilled, 0.92 M parameters on blocks 8-12) scores **91.2 % [90.1, 92.2] at 2 s and
97.7 % [97.1, 98.2] on full utterances** (was 75.0 / 89.3 %), **65.8 % [60.3, 71.6] English on EdAcc** (was 18.7 %;
AmberNet 67.7 %), at 0.21 ms per 160 ms chunk (RTF 0.0013). Both FLEURS numbers are below the bar (92 / 98), with
the bar inside or at the edge of the 95 % CI. The kill criterion is not triggered (≥ 88 %), so the head is wired as
`serve --lid head`, **LID stays off by default**, and the fusion fallback was measured (section F4).

### Before / after (FLEURS-17 test, n = 2550; EdAcc n = 257; the research/archive/LID.md clips, encoder run on each clip)

Accuracy % [95 % bootstrap CI]. EdAcc = fraction of accented-English segments called English (full / 2 s). The
`lid_distill` row is the official path (`scripts/research/lid.py eval_head`: each clip encoded alone, CPU); on
the cached features the same head read 91.1 / 97.7 / 66.1, so the cached numbers below are faithful.

| system | 1 s | 2 s | 3 s | 5 s | full | EdAcc en, full / 2 s | added cost |
|---|---|---|---|---|---|---|---|
| head `lid_aug` (before) | 59.2 | 75.0 [73.4, 76.5] | 81.6 | 85.7 | 89.3 [88.1, 90.6] | 18.7 / 13.2 | 0.16 ms / chunk |
| **head `lid_distill` (after)** | **78.6** [77.0, 80.2] | **91.2** [90.1, 92.2] | **94.2** | **96.2** | **97.7** [97.1, 98.2] | **65.8** [60.3, 71.6] / 51.4 | **0.21 ms / chunk, RTF 0.0013** |
| same, VAD-gated pooling (the server's) | 78.4 | 91.0 | 94.4 | 96.2 | 97.8 | – | same |
| NVIDIA AmberNet (teacher) | 83.9 | 95.1 | 98.2 | 99.2 | 99.5 | 67.7 [62.3, 73.2] / 47.1 | 16-81 ms / call, RTF 0.02-0.03 |
| bar | – | ≥ 92 | – | – | ≥ 98 | ≥ 60 | ≤ 0.01 RTF |

Per language (`lid_distill`, 2 s / full): en 99 / 100, zh 100 / 100, ar 99 / 100, fr 98 / 100, pl 97 / 100, tr 97 /
99, nl 97 / 99, ru 96 / 100, hi 95 / 99, es 95 / 100, ja 94 / 100, it 93 / 99, pt 89 / 99, fa 87 / 99, he 86 / 97,
**de 70 / 87**, **uk 58 / 83**. Half of the remaining 2 s errors are uk → ru (47 of 150 uk clips) and de → he /
ru / pl (the single-male-speaker test prefixes of section "Data"). EdAcc per L1 (full): US English, Mandarin, Filipino,
Tagalog, Spanish 100 %, Vietnamese 90 %, Catalan 70 %, Kenyan / Nigerian English 50 / 49 %, Hebrew 44 %, Italian 33 %;
wrong calls are mostly fa (33) and ar (14).

### F1. Diagnosis

- **Which blocks carry language** (linear probe per block, `probe`; standardise + logistic regression on the
  block's mean over the window, fitted on the 250 train utterances per language, scored on the test clips; it
  reproduces section 1 exactly, block 9 at 2 s = 68.3 %). 2 s: blocks 8-10 (66.2 / 68.3 / 67.8 %); 5 s: blocks 10-13
  (84.7-85.9 %); whole utterance: blocks 11-13 (87.2-89.4 %). Blocks 1-5 stay < 50 % at every length. Language
  therefore sits in the middle of the encoder, the short-window evidence lower (8-10) than the long-window one
  (11-13). The served head's learned mix over blocks 8-12 converged to 0.68 / 0.13 / 0.06 / 0.06 / 0.07, i.e.
  mostly block 8.
- **Start-up transient.** The same probe scores 4-6 points higher when the test clip is encoded with the ~1 s of
  leading silence before the onset (the file-start pass) than when the encoder starts at the onset (block 10:
  2 s 65.2 vs 61.2 %, 5 s 85.1 vs 79.1 %, both fitted on file-start features). Train / test views must match; the
  heads below are trained on both views (the onset pass and the file-start pass of the same audio span).
- **Confusions of the old head** (`confusion_before`): at 2 s, uk → ru 64 / 150, es → it 29, ru → uk 28, ja → es 23,
  fa → hi 22, de → ru 20, he → uk 18, de → he / fr 17; the ten largest pairs are 40 % of the 2 s errors and 64 % of the
  full-utterance ones. AmberNet's largest are he → ar 10 and ru ↔ uk 10 / 8.
- **Longer windows.** Accuracy rises steeply with the window for every head (old: 59 / 75 / 82 / 86 / 89 % at 1 / 2 /
  3 / 5 s / full; new: 79 / 91 / 94 / 96 / 98 %), so a 3 s decision buys +3-7 points. The streaming rule (F3) is where
  that trade is made.
- **How much is data.** FLEURS train has 7-11 h per language; section 2 used the first 250 utterances (38-59 min).
  `fetch_more` streamed the rest of every train tarball (42 913 more utterances, 136 h; speaker-disjoint from dev /
  test by FLEURS's construction). With cross-entropy only, 250 → 1000 → all utterances per language moves the head
  from 70.0 → 71.6 → 73.6 % at 2 s and 85.0 → 84.9 → 87.4 % on full utterances (runs `ce250_6k`, `ce1000`, `ceall`
  below), and the linear probe on block 10 from 67.8 → 68.9 → 71.3 % (2 s) and 85.3 → 90.0 → 91.1 % (whole utterance,
  `probe_scale_b10`). **Data alone is a weak lever**: 13x more labelled audio buys 3-4 points at 2 s, and the CE head
  still over-fits within 500-1250 steps (train loss 0.03-0.2 while dev stalls).

### F2. AmberNet distillation (the SPK_HEAD recipe)

- **Teacher cache** (`teacher`). NVIDIA AmberNet's 107-way logits (float16) on 5 windows per training utterance:
  1 / 2 / 3 / 5 s from the onset and one random 1-5 s window (the first 500 rows also got the whole utterance and two
  more random windows), 267 k windows on FLEURS-17 train (all 47 163 utterances) and 7 179 extra English segments
  (`extra_en`: AMI train 930 and ICSI train 3249 single-speaker segments, many non-native speakers, and 3000
  LibriSpeech train-clean-100 utterances; the SPK_HEAD teacher sets). The whole utterance from the onset (≤ 12 s) is a
  further label-only (CE) window, 315 k windows in all. AmberNet is 78-96 % right on these windows (per 500-row shard). Native grouped convs on MPS
  (320 audio-s / s; the unfold path is 2.6x slower there), batched with padding (logits equal to single-clip calls
  within 3e-5).
- **Feature cache** (`feats`, SSD `cache/lid_fix`, 105 GB). The frozen served encoder once per utterance and view,
  blocks 8-12 (6-13 for the 250-per-language set) as float16 frames: the onset pass (from the Silero onset, or the
  served VAD head's onset for rows without one, - 0.1 s), the file-start pass, and an augmented onset pass (speed
  0.9 / 0.95 / 1 / 1.05 / 1.1, white / pink / brown noise at 5-30 dB SNR, random tilt and gain, SpecAugment 2 x 15 mel bins + 4 x 5 % time on
  the frozen encoder's input) for the 250-set, the extra English and every second trainx shard. Every shard is read
  back after writing.
- **Training** (`train`, MPS, head only; the encoder is never touched). `LanguageHead` on a softmax mix of the
  tapped blocks; loss = (1 - α) CE (anytime running posterior + last frame, as section 2) + α T² KL(AmberNet_T ‖
  head_T) at the window's last frame, T = 2, AmberNet restricted to the 17 languages; class-balanced batches of 128
  windows (English: half FLEURS, half extra English); each window read from the onset pass (40 %), the file-start
  pass (35 %) or the augmented pass (25 %, `--aug`); 10 % frame dropout on the pooling; AdamW 2e-3 one-cycle; dev
  (60 per language) 2 s + full picks the step. Runs of 6000-8000 steps in ≤ 9-minute resumable segments.

| run | blocks | α | data | aug | hidden | steps | dev 2 s / full | test 1 s | test 2 s | test 5 s | test full | EdAcc 2 s / full | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ce250_6k | 8-12 | 0 | 250 / lang | – | 256 | 6000 | 70.7 / 85.7 | 57.0 | 70.0 [68.3, 71.8] | 82.3 | 85.0 [83.5, 86.4] | 11.3 / 9.3 | 12.6 |
| ce1000 | 8-12 | 0 | 1000 / lang | – | 256 | 6000 | 72.5 / 83.5 | 59.4 | 71.6 [69.9, 73.2] | 82.2 | 84.9 [83.5, 86.2] | 31.1 / 40.9 | 13.1 |
| ceall | 8-12 | 0 | all FLEURS train | – | 256 | 6000 | 72.8 / 87.1 | 58.7 | 73.6 [72.0, 75.2] | 83.8 | 87.4 [86.0, 88.7] | 12.1 / 15.2 | 14.5 |
| kdall | 8-12 | 0.5 | all FLEURS train | – | 256 | 6000 | 86.5 / 94.2 | 74.1 | 85.5 [84.2, 86.9] | 93.0 | 94.9 [94.0, 95.7] | 33.9 / 42.8 | 14.5 |
| kdall_en | 8-12 | 0.5 | + extra English | – | 256 | 6000 | 87.1 / 95.1 | 75.3 | 87.5 [86.2, 88.7] | 93.8 | 95.3 [94.5, 96.1] | 48.2 / 61.5 | 14.6 |
| kd_a8 | 8-12 | 0.8 | + extra English | – | 256 | 8000 | 89.2 / 97.2 | 75.5 | 88.8 [87.5, 90.0] | 95.5 | 97.2 [96.5, 97.8] | 51.4 / 67.3 | 19.4 |
| kd_a8_rnn (+ causal GRU 256) | 8-12 | 0.8 | + extra English | – | 256 | 8000 | 86.9 / 95.9 | 74.6 | 87.8 [86.4, 89.1] | 92.9 | 95.5 [94.6, 96.2] | 45.9 / 63.0 | 21.5 |
| kd_b8 (best single tap) | 8 | 0.8 | + extra English | – | 256 | 8000 | 90.1 / 96.3 | 76.2 | 88.8 [87.6, 90.0] | 94.6 | 96.4 [95.6, 97.1] | 49.0 / 62.6 | 18.6 |
| kd_b8_11 (2-block mix) | 8, 11 | 0.8 | + extra English | – | 256 | 8000 | 89.5 / 96.1 | 77.1 | 89.1 [88.0, 90.3] | 94.8 | 96.4 [95.6, 97.1] | 48.2 / 59.1 | 18.9 |
| kd_a8_aug | 8-12 | 0.8 | + extra English | yes | 256 | 8000 | 90.4 / 96.4 | 76.7 | 89.0 [87.8, 90.3] | 95.2 | 96.5 [95.8, 97.2] | 53.3 / **73.9** | 19.6 |
| **kd_h512 = `lid_distill`** | 8-12 | 0.8 | + extra English | – | 512 | 8000 | **91.2 / 97.5** | 78.5 | **91.1** [89.9, 92.1] | 96.2 | **97.7** [97.0, 98.2] | 50.6 / 66.1 | 23.4 |
| kd_h512_aug | 8-12 | 0.8 | + extra English | yes | 512 | 6500 | 90.5 / 97.4 | 78.5 | 90.5 [89.4, 91.6] | 96.2 | 97.2 [96.5, 97.8] | 41.2 / 57.6 | 19.3 |

(Test columns on the cached clip features, n = 2550 / 257; "min" = MPS wall time. `ce250_6k` replicates section 2's
`lid_head` within CI: 70.0 / 85.0 vs 72.6 / 87.7 % without SpecAugment through the encoder.)

- **Distillation is the lever, as for the speaker head.** On the same data and features, KL to AmberNet takes the
  head from 73.6 → 85.5 % at 2 s and 87.4 → 94.9 % on full utterances (`ceall` → `kdall`, +12 / +7.5 points); a
  larger teacher weight (α 0.8) and more steps add +1.3 / +1.9, and a wider head (hidden 512) +2.3 / +0.5. The CE
  heads memorise speakers (train loss → 0.01, dev best after 500-1250 steps); the distilled ones keep improving to the
  end of the schedule (KL ≈ 0.12-0.2 at the end, the student never matches the teacher).
- **Extra English fixes accented English**: 42.8 → 61.5 % on EdAcc full (`kdall` → `kdall_en`) at no FLEURS cost
  (+2.0 / +0.4); the waveform augmentation adds another +6.6 (67.3 → 73.9 %, `kd_a8` → `kd_a8_aug`) with FLEURS
  unchanged, but not on the 512-wide head (57.6 %, where it also cost 0.6 / 0.5 on FLEURS, and the run was 1500 steps
  shorter to fit the time box).
- **Tap.** Block 8 alone (88.8 / 96.4) and blocks 8 + 11 (89.1 / 96.4) equal the 5-block mix at 2 s (88.8) and are
  0.8 points behind on full utterances (97.2); the mix is kept. A causal GRU in the head (sequence / phonotactic
  context, streaming state) did not help (87.8 / 95.5); the option stays in `LanguageHead` (`rnn`, `context`; unset =
  the original head, tested for streaming = offline).
- **Selection** is by dev (60 utterances per language, few speakers): every run's dev peak sits within ±1 point of
  its neighbours (the batch order is seeded identically across runs), so differences under ~1.5 points between rows are
  not resolved. `kd_h512` has the best dev score and is the served head; `kd_a8_aug` is the EdAcc-leaning alternative
  (89.0 / 96.5 / 73.9 %), kept at `runs/lid_kd_a8_aug.pt`.
- **Time box.** Three runs (`kd_a8_rnn` 21.5 min, `kd_h512` 23.4 min, first `ce250` 23 min) overran the 20-minute
  budget per MPS run; they ran as resumable 9-minute segments, one MPS job at a time.

### F3. Streaming decision rule

The server's view: whole FLEURS-17 test utterances (all 2550) and EdAcc segments streamed from the file start, the
head's VAD-gated running posterior (30 s half-life), `audioforge.lid.LangDecider` (min 1 s of pooled speech), now
with an optional timeout `max_ms` (announce the top language after that much pooled speech if nothing was confident;
`rule`). Time = seconds from the file start (the median FLEURS onset is 1.0 s).

| rule (`lid_distill`) | FLEURS announced | FLEURS first call correct | time p50 / p90 | flips / utt | EdAcc announced | EdAcc first call = en |
|---|---|---|---|---|---|---|
| p ≥ 0.9 (previous rule) | 98.1 % | 94.2 % [93.3, 95.2] | 2.32 / 4.24 s | 0.04 | 48.2 % | 78.2 % of announced (37.7 % of all) |
| **p ≥ 0.9 or after 3 s (pre-registered)** | 99.96 % | 91.8 % [90.6, 92.7] | 2.40 / 4.08 s | 0.05 | 86.8 % | 59.2 % (51.4 % of all) |
| p ≥ 0.95 or after 3 s | 99.96 % | 93.4 % [92.3, 94.3] | 2.56 / 4.24 s | 0.03 | 85.6 % | 60.0 % |
| p ≥ 0.8 or after 3 s | 100 % | 89.5 % | 2.24 / 3.84 s | 0.10 | 90.3 % | 56.9 % |
| p ≥ 0.9 or after 2 s | 100 % | 89.5 % | 2.32 / 3.77 s | 0.08 | 98.4 % | 51.0 % |
| old head `lid_aug`, p ≥ 0.9 (section 4, 340 utts) | 96.8 % | 78.7 % | 2.40 / 4.58 s | 0.17 | – | – |
| AmberNet, p ≥ 0.9 (section 4, 340 utts) | 99.4 % | 96.8 % | 2.16 / 3.78 s | 0.03 | – | – |

The pre-registered rule announces for every FLEURS utterance and is right 91.8 % of the time at a 2.4 s median
(from the file start, ~1.4 s of speech); raising the threshold to 0.95 buys 1.6 points for 0.16 s. On EdAcc the
timeout is what makes the head speak at all (48 → 87 % announced), at 59 % English among the calls; the remaining
13 % are segments with < 3 s of speech.

### F4. Fusion fallback: head early, transcript-text LID to confirm or override (`fusion`)

Measured because the bar was missed. The transcript is Parakeet-TDT v3's (the `--final-asr tdt_v3` per-turn final;
the streaming 115M partials are English-only and cannot carry this), classified with langid.py as in
research/archive/HYBRID_ASR.md section 5, on the utterances that have a v3 transcript (30 per language = 510, both the 2 s
clips and the full utterances; the rest of those transcripts were in a wiped scratch). "Guarded" = take the text's
language when it is one of v3's 10 supported FLEURS-17 languages with langid p ≥ 0.9 **and** the head's own top-1 is
also a supported language; otherwise keep the head (v3 writes he / ar / tr / fa / hi / zh / ja as fluent text in the
wrong language, so an unguarded override is harmful).

| on the same 510 utterances | 2 s | full |
|---|---|---|
| head `lid_distill` alone | 90.8 % [88.0, 93.1] | 97.3 % [95.7, 98.6] |
| text LID alone (empty = wrong) | 53.9 % (91.7 % on supported, 0 % on the rest) | 59.0 % (100 % / 0.5 %) |
| override whenever the text is a supported language (p ≥ 0.9) | 72.5 % | 87.8 % |
| **guarded fusion (p ≥ 0.9)** | **93.3 %** [91.2, 95.5] | **99.2 %** [98.4, 99.8] |

Guarded fusion clears the bar on this subset (93.3 / 99.2 % vs 92 / 98), by fixing ru / uk / de / pt / es / it
confusions on the supported side (88.7 → 93.0 % there) while leaving the unsupported languages to the head (93.8 /
99.5 %). Caveats: n = 510 (CIs ±2 points); the text only exists once a v3 final exists, i.e. at the first turn end
plus 0.33 s (2 s clip) / 0.60 s (full utterance) of v3 compute on 2 CPU threads, not "after ~1 s of words"; it needs
`--final-asr tdt_v3` (+2.5 GB); and it only helps within v3's 25 languages. It is **not wired into the server** (the
language message would have to be re-issued from the final path); the head's early call is what `--lid head` serves.

### F5. What is served, and the verdict

- `serve --lid head` loads the shipped head file `lid_distill.pt` (models directory, else `runs/`; 3.7 MB, 924 817
  parameters, blocks 8-12 of the shared encoder pass) and uses the pre-registered rule: announce at `--lid-threshold`
  (0.9) or after `--lid-max-ms` (default 3000 with `head`, `0` = off) of pooled speech. `--lid ambernet` and
  `--lid <file>` are unchanged; the `language` message is unchanged. Enabling LID leaves every other message
  bit-identical (tests/test_lid.py).
- **Bar: missed** on FLEURS (91.2 % at 2 s vs 92; 97.7 % full vs 98), **met** on EdAcc (65.8 % vs 60) and cost
  (RTF 0.0013 vs 0.01). Kill criterion not triggered (91.2 ≥ 88). **LID stays off by default**; the launcher is not
  changed. Against AmberNet the head is now 3.9 points behind at 2 s (was 20), 1.8 on full utterances (was 10), level
  on accented English, at 1 / 100 of its per-call cost and no second model.
- **Scorecard: language ID stays "worse"** than the dedicated model, but narrowly; with the TDT v3 final, guarded
  fusion reaches 93.3 / 99.2 % on the 510-utterance subset.
- **Licence caveat.** The served head is trained on AmberNet's outputs (NGC Terms of Use) and on FLEURS (CC-BY-4.0),
  AMI / ICSI / LibriSpeech (CC-BY-4.0); check the NGC terms before redistributing a teacher-distilled head.
- **Next levers**, in order: (1) more speakers for the weak languages (uk, de, he; FLEURS gives ~10 per language), or
  unlabelled multilingual audio for pure distillation, since the KL term does not need labels; (2) wire guarded
  fusion into the final path when `--final-asr tdt_v3` is on; (3) the frozen encoder is the ceiling: block probes top
  out at 71 % (2 s, block 10) with all the data, and closing the last 4 points at 2 s likely needs a trainable block.

Commands (each call < 10 min, resumable; model-loading calls through `scripts/dev/gate.sh`):

```bash
.venv/bin/python scripts/research/lid_fix.py fetch_more                      # rest of FLEURS-17 train, 136 h
.venv/bin/python scripts/research/lid_fix.py extra_en                        # AMI / ICSI / LibriSpeech English
for sv in train:full train:on dev:full dev:on test:full test:on edacc:full edacc:on; do ...   # (blocks 6-13)
  scripts/dev/gate.sh .venv/bin/python scripts/research/lid_fix.py feats --set ${sv%%:*} --view ${sv##*:}; done
scripts/dev/gate.sh .venv/bin/python scripts/research/lid_fix.py feats --set trainx --view full --blocks 8-12  # also on, aug (--stride 2), extra_en
scripts/dev/gate.sh .venv/bin/python scripts/research/lid_fix.py teacher --set trainx  # also train, extra_en
.venv/bin/python scripts/research/lid_fix.py probe; .venv/bin/python scripts/research/lid_fix.py probe_scale --blocks 10
.venv/bin/python scripts/research/lid_fix.py confusion lid_aug ambernet
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 scripts/dev/gate.sh .venv/bin/python \
  scripts/research/lid_fix.py train --tag kd_h512 --blocks 8-12 --steps 8000 --alpha 0.8 --trainx --extra-en --hidden 512
cp runs/lid_kd_h512.pt runs/lid_distill.pt
scripts/dev/gate.sh .venv/bin/python scripts/research/lid.py eval_head --head runs/lid_distill.pt [--set edacc]
.venv/bin/python scripts/research/lid_fix.py rule --head runs/lid_distill.pt
.venv/bin/python scripts/research/lid_fix.py fusion --tag kd_h512
```

## Commands

```bash
.venv/bin/python scripts/lid_data.py fetch            # FLEURS 17 x (250 / 60 / 150), streamed tar prefixes, ~1 min
.venv/bin/python scripts/lid_data.py edacc            # EdAcc 5 row groups (accented English probe)
.venv/bin/python scripts/lid.py onsets                # Silero onsets
.venv/bin/python scripts/lid.py probe_feats --split test; .venv/bin/python scripts/lid.py probe_feats --split train
.venv/bin/python scripts/lid.py probe
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \
  .venv/bin/python scripts/lid.py train --recipe research/recipes/lid_aug.yaml     # re-run until "[train] finished" (also lid_head.yaml)
.venv/bin/python scripts/lid.py eval_head --head runs/lid_aug.pt [--set edacc]   # re-run until "all conditions done"
for s in ambernet speechbrain whisper_tiny whisper_base; do
  .venv/bin/python scripts/lid.py baseline --system $s; .venv/bin/python scripts/lid.py baseline --system $s --set edacc; done
.venv/bin/python scripts/lid.py decide --head runs/lid_aug.pt --system ambernet --n 20
.venv/bin/python scripts/lid.py cost --head runs/lid_aug.pt
.venv/bin/python scripts/lid.py report
.venv/bin/python -m pytest -q tests/test_lid.py
# serving
python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm --lid ambernet [--lid-langs en,he,ar]
python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm --lid runs/lid_aug.pt
```

Code: `audioforge/heads/audio.py` (`LanguageHead`), `audioforge/lid.py` (head files, `LIDStream`,
`AmberNetLIDStream`, `LangDecider`), `audioforge/baselines/lid.py` (AmberNet port, SpeechBrain, Whisper),
`audioforge/nemo_import.py` (`import_ambernet`), `audioforge/serve.py` (`--lid`), `scripts/lid_data.py`,
`scripts/lid.py`, `research/recipes/lid_{head,aug}.yaml`. Tests: `tests/test_lid.py` (streaming == offline, causality,
padding, loss, registry / tap, head-file round trip, the announcement rule, both server backends, the default
protocol unchanged, the x-vector decoder, the fast depthwise path). Numbers: `runs/lid.json`.
