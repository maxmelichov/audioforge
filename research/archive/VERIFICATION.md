# Verification report

An adversarial verification pass over this project's claims, ablations and edge-case bugs. Date: 2026-09-25. This file was compiled from the JSON returned by the verification workers; no numbers were recomputed while writing it.

> **Input was cut off.** The JSON this report was built from stops partway through the 15th confirmed bug (Trainer.fit hang, `audioforge/train.py:108`). The sections holding the refuted/unreproducible reports, the unverified low-priority findings, the list of applied **fixes** and the **deferred** list never arrived. Where this report needs them, it says so. The "fixed?" column below is therefore **unknown** for every bug. Nothing in the received JSON records a fix as applied.

## 1. Executive summary

- **28 claims checked: 4 confirmed, 20 partially true, 4 refuted, 0 unverifiable.** Confirmed: C01, C02, C15, C19. Refuted: C09 (model sizes, run times, test and recipe counts, "four new designs"), C17 (catalog family row), C22 ("beats every commercial API" end-of-turn target), C26 ("4 separate models" and the "~2× not 4×" compute argument).
- **Ablation, one encoder with five heads vs separate models: mixed.** Inference is 4× cheaper and parameters 3.5× fewer. At the same number of steps, accuracy is **much worse**: TDT WER 51.7% vs 0.26% at 1500 steps, and VAD 86% vs 94%.
- **Ablation, codec-token enhancer: mixed.** 26.5% token accuracy is well above trivial baselines (majority 6.9%) and above per-frame MLPs (23.2%). The decoded mel is **no better than a per-frame MLP**, and it is **beaten by 1-parameter spectral subtraction**. The limit is the codec.
- **Ablation, speaker-attributed ASR in one model vs a cascade: mixed.** The only real gain is ~43% fewer parameters. Compute and latency are the same (still one re-encode per speaker). DER is worse (7.2% vs 5.1%). With 3 speakers, every system fails.
- **Headline corrections:** the EOU head scores **below the always-"no EOU" baseline** (88% vs 92%). Speaker-attributed WER is 3.0% only with oracle activity; it is ~18% end to end. Codec "chance ~0.5%" is wrong (uniform is 0.125%, majority 6–7%). The SALM LM is **not frozen**. VibeVoice-ASR-Streaming refutes "no open speaker-attributed streaming model". JoinIn AI Baton (577 ms) is missing from the end-of-turn comparisons.
- **Confirmed edge-case bugs received: 15**, 4 high and 11 medium. Two of the medium ones are the same StreamingSession layer-mix defect. The ablation adds one more (train.py:164). High: read_manifest ignores offset/duration; a FrameHead with num_classes>1 crashes evaluate before the model is saved; Collate takes its keys from batch[0] only; load_wav fails on 24-bit, float and truncated WAVs.
- **Fixed vs deferred:** unknown, because those sections were cut off. The bugs in model.py, train.py and fastconformer.py can only be applied by the lead, so they are listed as deferred in §6.

## 2. Claims

| id | claim (short) | verdict | evidence (short) | correction |
|---|---|---|---|---|
| C01 | Parakeet TDT+CTC and Nemotron streaming RNNT, 2.3M: WER 0.0%, CTC 1.0% / CTC aux 30% | confirmed | Exact on val. Fresh set of 500 utterances: TDT/RNNT 0/1459 words (95% upper bound ≈0.2%). Streaming at 160 ms matches offline on 564/564 | none |
| C02 | Canary AED 2.8M: WER 2.1% on ASR plus "translation" prompts | confirmed | Exact (4/189 words). Fresh seeds 1.35–1.82%. Swapping the prompt flips the task. 17/64 val transcripts are also in train | optional: add error count and fresh-seed range |
| C03 | Sortformer 2.3M: DER 4.0% on overlapping mixtures | partially | 4.10% on CPU. Fresh sets 4.7–5.7%. Only ~3% of speech overlaps, and each mixture has exactly one A→B change. Midpoint-split baseline 15%, oracle change point 2.4% | describe as a single A→B turn with ~3% overlap, and add fresh-set numbers and baselines |
| C04 | SALM ~3M: WER 9.1%, frozen causal LM | partially | 9.1% = 5/55 words. A CPU re-run gave 7.3% on the same set and 5.7% on 200 fresh utterances. The model has 2.62M params. **The LM is fine-tuned, not frozen** (freeze_llm=False) | 2.6M; state the real training command; ANALYSIS: "fine-tuned together with the encoder" |
| C05 | voice_agent_frontend: one pass WER 0.0%, VAD 91%, EOU 88%, speaker ID 100% | partially | Exact to 4 decimals. **EOU 88% is below the 92% always-0 baseline** (precision 0.40, F1 0.55), and the head fires early in 66% of utterances. Energy VAD scores 99.9%. Speaker ID is closed-set | EOU is "not yet solved"; give the VAD base rate; speaker ID is closed-set |
| C06 | speaker_attributed_asr: DER 8.8%, per-speaker WER 3.0%; its own diarization steers ASR | partially | 3.0% uses **oracle** speaker activity. With its own diarization the WER is **18.8% / 17.5%** (seeds 1 and 7) | give both numbers |
| C07 | codec enhancer: 25% token accuracy, chance ~0.5% | partially | 25.2% reproduced. **Uniform chance is 0.125%.** The majority baseline is 6.7% and re-encoding the noisy audio gives 1.6%. Speech frames score 27% | give the baselines and decoded mel L1 |
| C08 | results are on a held-out validation set | partially | Same 8 speakers, same lexicon and same generator as training; ~1/3 of transcripts also appear in train. Results still hold within ~1.5 points on text-disjoint audio and on unseen speakers. Exceptions: speaker ID falls to 38% on unseen speakers, codec accuracy to 22.2% | reword the table header and add a fresh-set re-check paragraph |
| C09 | 0.3–2M params, about a minute each, 56 tests, 7 recipes, four new designs | **refuted** | Models are 2.28–2.82M (LibriSpeech runs 13–14M). Runs take 51–143 s (newer runs 1159 s and 5794 s). 104 tests are collected. There are 10 recipes. The docs' own table marks only 3 designs as new | see §7 |
| C10 | streaming WER ~96% was caused by raw log-mel and fixed by fixed global normalization | partially | NVIDIA's streaming model uses `normalize: NA` (raw log-mel). The normalization and the schedule were changed together, and no saved run isolates which one mattered. The speaker head reads `from_layers: all`, not the "middle layers" | see §7 |
| C11 | Teachers on LS test-clean, first 200 utterances: CTC-0.6b 1.73, CTC-1.1b **1.64** best | partially | Both reproduced exactly. The subset has only 4 of 40 speakers. The 0.6b vs 1.1b gap is not significant (bootstrap 68%, CI −0.20 to +0.40 pp). TDT and Nemotron were not re-run. RTFx and memory could not be checked | remove the bold; add a subset and significance note |
| C12 | close to the card numbers; batch 1 gives the same WER at RTFx 76 | partially | Card numbers exact. Batch 1 and batch 8 give identical transcripts (0/200 differ). The cards cover the full test-clean set and use the Whisper normalizer. Nemotron's 2.32 is its 1.12 s chunk setting. RTFx 76 is MPS-only | note the full set, the normalizer and the chunk setting |
| C13 | TDT-v3 is worse because of British spellings; writes punctuation and casing, no digits | partially | 2.01% = 93 errors, exact. Spelling variants are **16 of 93 (~17%)**, not most. Mapping them gives 1.66%. Most remaining errors are compound words and rare words. No digits confirmed for TDT-v3 only | quantify it |
| C14 | Nemotron lookahead 13 frames (1.12 s); ~1 h/2 h labelling per minute; table ends at 5000 frames | partially | 13 frames = **1.04 s**; 1.12 s is the 14-frame chunk. Throughput is correct. Parakeet positions in transformers are computed on the fly and not capped at 5000 (a 6000-frame input ran fine); only Nemotron streaming raises | see §7 |
| C15 | 149 audio models, 5.6M downloads | confirmed | Exact. None of the 16 dropped candidates is audio | none |
| C16 | 144 of 149 cards; 5 missing because gated | partially | Only 4 are gated. The flamingo card is public and was simply not fetched. raw/missing.txt is a stale snapshot | see §7 |
| C17 | family row "VAD, EOU, SLU, NEST... 12 / ~6,000, mostly FastConformer" | **refuted** | 12 only if the multimodal embedder counts, and then downloads are ~16,700. EOU belongs to the Parakeet row. Only 4 of the 12 are FastConformer | replace the row |
| C18 | since mid-2023, 51 of 53 recognition models use FastConformer | partially | It is **51 of 54**. The MarbleNet VAD (2025) was dropped without reason. The count is sensitive to the cutoff date | "51 of 54", with three exceptions |
| C19 | 84 of 149 are cc-by-4.0; OpenMDW; non-commercial exceptions | confirmed | Exact | none (the catalog's `other` bucket needs license_name) |
| C20 | one front end (128 mel?), FastConformer layer counts | partially | Every checked config uses 80 mels; 128 is unconfirmed. Normalization and conv norm differ between models. Canary-1B has 24 layers (only v2 has 32). Sortformer v1 has 18 | see §7 |
| C21 | eot-bench numbers; EOU card 160 ms; Pine AI recall 0.17 | partially | Every number matches. **JoinIn AI Baton, 577/350 ms, is a commercial API** and beats Soniox. Recall 0.17 is a figure label from Pine AI, as-shipped | see §7 |
| C22 | target ≤600 ms beats every commercial API on eot-bench | **refuted** | LiveKit v1 cloud (543/295) and JoinIn AI Baton (577/350) are hosted systems that already beat it | see §7 |
| C23 | credibility-bar baselines (EOU-120M, Sortformer-v2, TitaNet, Zipformer) | partially | Every number matches. The Earnings-22 target of 15.8 is **worse** than the 15.76 baseline, which breaks Bar A. The EOU numbers are cited to a CSV that has no row for that model | ≤15.62 / ≤15.76; cite the model card |
| C24 | Nemotron-3-Diarization: 100M, Sept 2026, DIHARD 13.55 / CALLHOME 11.32 | partially | Facts exact. The "(forced-aligned refs)" tag is **wrong** for DIHARD and CALLHOME (both use original references). Comparable Sortformer-v2.1 numbers: 19.85 / 12.67 | fix the tag; add the same-protocol comparison |
| C25 | no open model does speaker-attributed streaming ASR; NVIDIA needs two models | partially | **Refuted in part:** Microsoft VibeVoice-ASR-Streaming (1.5B/7B, MIT, arXiv 2609.02812) does it. The NVIDIA two-model part is confirmed. The sub-claim that nobody ships words+VAD+EOT+speaker together was not refuted | see §7 |
| C26 | NVIDIA ships 4 models that each re-encode; fusion saves ~2× not 4× | **refuted** | NVIDIA's EOU model already includes ASR, and the VAD is a 91.5K CNN, so there are 2 real encoders. Streaming Sortformer re-encodes cache+FIFO+chunk at every step: ~127 passes per frame at 0.32 s. The saving is ~2× only at ≥10 s buffers and 1–2 orders of magnitude at 0.32–1 s | see §7 |
| C27 | 1.3e19 FLOPs ≈ 44 A100-h, $200–400; 18 ms per 80 ms step | partially | FLOPs and A100-h correct. ×3–5 gives **$183–350**. The §5 table's lower bounds imply ~$1.2/h, not $1.39/h. The 18 ms step was **not reproduced** (~26 ms median under load) | see §7 |
| C28 | FastConformer and NanoCodec share a 12.5 Hz clock; codec-token enhancement is new | partially | Only 2 of 3 NanoCodecs run at 12.5 fps (Magpie prefers the 21.5 fps one). The experiment uses audioforge's own MelCodec, not NanoCodec. The VoiceChat card never names NanoCodec. Prior art exists (SELM, MaskSR, Genhancer, LLaSE-G1) | see §7 |

## 3. Ablations

### 3.1 One encoder with five heads vs separate single-purpose models

**Question.** Is the one-encoder, five-head voice-agent front end (research/recipes/voice_agent_frontend.yaml) a real improvement over separate single-purpose models, with the same encoder, data, steps and batch size?

**Setup.** The recipe encoder (d144, 4 layers, causal, multi-lookahead), synthetic multitask data (seed 0, n_train=1500), batch 16, lr 2e-3 with cosine schedule, CPU with 2 threads, seed 0. (A) The joint model with the recipe's 5 heads. (B) Separate asr (TDT+CTC), vad, eou and spk models; the single-head models use weight 1.0 and no grad_scale. Evaluation on 128 utterances from seed 2. All models were run at 500 steps, plus a matched joint vs ASR pair at 1500 steps. Inference was timed with analyze() on a 60 s clip; FLOPs from FlopCounterMode. Script: scratchpad/wf-verify/abl-multihead/driver.py.

| Metric (held-out, 128 utts) | A: joint 5-head | B: separate model | Joint worse? |
|---|---:|---:|---|
| TDT WER @500 steps | 95.8% | 95.6% (asr) | tie, neither converged |
| CTC WER @500 | 100.5% | 100.5% | tie |
| **TDT WER @1500 steps** | **51.7%** | **0.26%** (asr) | **yes, severe** |
| CTC WER @1500 | 84.7% | 2.9% | yes, severe |
| TDT train loss, last 25 steps @1500 | 0.40 | 0.027 | yes |
| VAD acc / recall @500 | 79.5% / 89.3% | 94.4% / 95.8% | yes |
| VAD acc / recall, joint @1500 (single at 500) | 86.4% / 86.4% | 94.4% / 95.8% | yes, even with 3x steps |
| EOU acc / recall @500 | 84.3% / 60.2% | 88.8% / 82.0% | yes |
| EOU acc / recall, joint @1500 (single at 500) | 85.8% / 87.5% | 88.8% / 82.0% | about even |
| Speaker acc @500 | 100% | 98.4% | no (tie) |
| Speaker acc, joint @1500 | 99.2% | n/a | no |
| **Params** | **2.42M** | 2.30 + 2.04 + 2.04 + 2.13 = **8.51M** | joint is 3.5x smaller |
| **Encoder GFLOPs, 60 s** | **3.75** | 4 x 3.75 = **15.0** | joint uses 4x less |
| **analyze() on 60 s, trained, 2 threads** | **0.082 s** | 0.081 + 0.061 + 0.059 + 0.067 = **0.268 s** | joint 3.3x faster |
| Training wall time @500 | 463 s | 459 + 376 + 261 + 379 = 1475 s | joint 3.2x cheaper |

**Verdict: mixed.** Inference really improves: one encoder pass does all four jobs with 4× fewer encoder FLOPs, runs analyze() 3.3× faster, has 3.5× fewer parameters and costs ~3.2× less to train. Speaker ID is no worse. At the same number of steps, though, there is strong interference during training. ASR suffers most (0.26% vs 51.7% WER at 1500 steps, with 15× higher TDT loss even with grad_scale 0.3). VAD is clearly worse (86.4% at 1500 steps vs 94.4% for the single model at 500), and EOU recall lags at 500 steps. The README's 0% WER came from 2000 steps on 3000 utterances, so the joint model probably does converge eventually, but it needs several times more steps. "One pass, ~1/4 the encoder compute" holds for inference. **"No accuracy cost" is not supported at matched training budgets.**

**Caveats.** One seed, 2M parameters, synthetic data. The small VAD and EOU gaps could move with another seed; the WER gap will not. The single VAD, EOU and speaker models were not run at 1500 steps. Dropping grad_scale interacts slightly with gradient clipping. Timings come from a heavily loaded machine; the ratios are consistent across runs. analyze() also includes TDT decoding, so the ratio is 3.3× rather than 4×. Longer joint training and loss reweighting were not tried. CPU only, with 1500 training utterances instead of 3000.

### 3.2 Codec-token enhancer

**Question.** Is the codec-token enhancer (noisy audio → FastConformer → the clean speech's 4×800-way FSQ MelCodec tokens) a real improvement, or can ~26.6% token accuracy be reached by trivial means? And does it actually enhance the audio?

**Setup.** CPU with 2 threads. Test set: synthetic_dataset('enhance', 128, seed=2) with runs/codec.pt (3041 tokens, 51% speech), from a replica generator verified bit-identical. Val seed 1, train seed 0. Compared: (a) the model; (b) re-encoding the noisy audio; (c) majority code, including VAD-gated and per-digit variants; (d) per-frame logistic regression and MLPs with no encoder; (e) mel-domain metrics against the clean codec mel. Extra baselines: spectral subtraction (alpha tuned on val, with a prior floor), a direct mel-regression MLP, and a probe that trains a regression head on the frozen trained encoder. Paired bootstrap with 4000 resamples. Scripts: scratchpad/wf-verify/abl-codec/.

**Token level (test seed=2, 128 utts, 3041 tokens). Accuracy in %.**

| Method | Token acc | speech | silence | cb0 | cb1 | cb2 | cb3 | FSQ-digit acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a) Model argmax** | **26.46** | 29.17 | 23.64 | 32.4 | 33.3 | 20.4 | 19.8 | 68.4 |
| Model, per-digit marginal-mode decoding | 24.57 | 28.38 | 20.63 | 29.0 | 32.5 | 19.4 | 17.4 | 69.9 |
| (b) Identity codec(noisy) | 1.59 | 1.29 | 1.91 | 6.3 | 0.0 | 0.0 | 0.1 | 30.1 |
| (c) Majority per codebook | 6.86 | 1.86 | 12.04 | 9.7 | 7.0 | 6.5 | 4.2 | 43.6 |
| Majority gated by oracle VAD | 7.06 | 2.26 | 12.04 | 9.7 | 7.0 | 6.6 | 4.9 | 44.3 |
| (d) Logistic regression, 1 token | 14.78 | 17.72 | 11.74 | 18.3 | 17.8 | 11.5 | 11.6 | 58.4 |
| (d) MLP, 1 token | 17.87 | 20.06 | 15.61 | 21.7 | 21.2 | 13.9 | 14.7 | 61.9 |
| (d) MLP, ±1 token | 21.79 | 24.95 | 18.52 | 24.9 | 28.6 | 15.9 | 17.7 | 65.4 |
| (d) MLP 2×768, ±2 tokens (5.5M) | 23.15 | 25.31 | 20.91 | 27.8 | 29.6 | 17.2 | 18.1 | 66.5 |
| codec(DSP spectral subtraction with prior floor) | 13.47 | 11.00 | 16.02 | 17.1 | 17.8 | 10.0 | 9.0 | 60.1 |
| codec(frozen-encoder regressed mel) | 22.01 | 25.97 | 17.90 | 21.4 | 32.4 | 14.6 | 19.7 | 68.0 |

Other checks:
- Model on val seed 1: 25.16%. The training log reports 25.28%.
- Model on the first 128 training utterances: 28.95%, so the gap to test is small (mild underfit, not overfit).
- All 4 codebooks right in the same frame: model 0.66%. Top-5 accuracy: 67.4%.
- Silence tokens are high-entropy: 312–444 distinct codes per codebook, 5.4–6.1 bits, and the most common code covers only 9–14% of silence frames.

**Enhancement (decoded mel vs clean mel, valid frames). Lower is better except mel-SNR.**

| Method | L1 all | L1 speech | L1 silence | LSD (dB) | mel-SNR speech (dB) | peak-bin L1 |
|---|---:|---:|---:|---:|---:|---:|
| Noisy mel as-is | 1.372 | 1.267 | 1.485 | 20.25 | 1.61 | 0.510 |
| Constant: mean clean mel | 0.455 | 0.548 | 0.354 | 8.67 | 0.27 | 1.480 |
| Majority tokens, decoded | 0.460 | 0.562 | 0.350 | 8.84 | 0.26 | 1.536 |
| Identity tokens, decoded | 0.677 | 0.702 | 0.649 | 10.64 | 0.64 | 1.012 |
| Logistic-regression tokens, decoded | 0.406 | 0.449 | 0.361 | 7.21 | 0.90 | 0.891 |
| MLP ±2 tokens, decoded | 0.394 | 0.437 | 0.348 | 6.98 | 0.97 | 0.847 |
| **Model argmax tokens, decoded** | **0.392** | **0.435** | 0.346 | 6.92 | 1.04 | 0.817 |
| Model soft decode (expected latent) | 0.396 | 0.437 | 0.352 | 7.02 | 1.03 | 0.874 |
| **codec(clean) decode = ceiling with perfect tokens** | 0.278 | 0.340 | 0.212 | 5.60 | **1.05** | 0.818 |
| DSP spectral subtraction with prior floor (1 parameter) | 0.388 | 0.420 | 0.352 | 6.97 | **11.08** | 0.564 |
| DSP spectral subtraction, plain | 0.433 | 0.450 | 0.416 | 7.45 | 11.04 | 0.524 |
| MLP direct mel regression (no encoder) | 0.358 | 0.363 | 0.353 | 5.93 | 3.16 | 0.444 |
| **Frozen trained encoder + MLP mel-regression head** | **0.340** | **0.337** | 0.344 | **5.64** | **5.31** | **0.365** |
| codec(frozen-encoder regressed mel), decoded | 0.394 | 0.435 | 0.349 | 6.97 | 1.01 | 0.849 |

**Paired bootstrap on speech-frame L1: model minus baseline, 95% CI (negative = model better)**

| Baseline | Difference | 95% CI |
|---|---:|---|
| Logistic-regression tokens | −0.014 | [−0.019, −0.009] |
| MLP ±2 tokens | −0.0007 | [−0.006, +0.005], no difference |
| MLP ±1 tokens | −0.003 | [−0.009, +0.002], no difference |
| DSP spectral subtraction with prior floor | +0.0165 | [+0.002, +0.031], DSP better |
| Direct regression MLP | +0.074 | [+0.061, +0.086] |
| Frozen-encoder regression | +0.100 | [+0.090, +0.110] |

- Frozen-encoder regression minus the perfect-token ceiling: −0.004 [−0.020, +0.012], so the probe matches perfect tokens.
- Codec diagnostic (first 40 test utterances, speech frames): the clean spectral peak averages 3.16 normalized units, but codec(clean) reconstructs only 2.05. That is about 15 dB of peak energy lost. The codec puts the peak in exactly the right mel band only 16% of the time (23% within ±1 band).

**Verdict: mixed.** Token accuracy is not trivial. It beats the majority baseline (6.9%) and re-encoding the noisy audio (1.6%), and the FastConformer adds +3.3 points over the best per-frame MLP. **As enhancement it has no advantage**: the decoded mel is statistically tied with the per-frame MLP's tokens, spectral subtraction beats it on speech L1 and by ~10 dB of mel-SNR, and direct mel regression is far better. In silence frames the model is no better than a constant spectrum. The root cause is the codec: even perfect tokens decode to 1.05 dB mel-SNR, worse than leaving the noisy input alone. The encoder did learn to denoise (the frozen-encoder probe matches the perfect-token L1 and reaches 5.3 dB), but routing its output through the codec throws that gain away. Token accuracy is a poor proxy for quality. Suggested next steps: add a mel or mask head, keeping tokens as an auxiliary output; fix the codec first; report FSQ-digit accuracy and decoded-mel metrics against majority and DSP baselines.

**Caveats.** Synthetic data, 128 test utterances, one seed per baseline. The MLPs were early-stopped on val, which helps them slightly. There is no vocoder, so nothing is measured at waveform level. The two metric families pull in different directions. The spectral-subtraction alpha was tuned on val. The frozen-encoder regression is a probe, not a retrain. The verdict applies to this particular 480 bps codec. During the run the worker killed only its own two scratch jobs (PIDs 23603 and 23605) because swap was full, and deleted its own 1.1 GB scratch file.

### 3.3 Speaker-attributed ASR in one model vs a two-model cascade

**Question.** Is speaker-attributed ASR in one model (a Sortformer head and a speaker-kernel ASR head sharing one encoder) really better than a two-model cascade?

**Setup.** CPU with 2 threads. Test sets: 64 two-speaker mixtures (diar, seed 2) and 64 three-speaker mixtures; every model was trained on 2 speakers. Systems: (A) the joint model steering itself; (B) sortformer_diar's activity driving the joint ASR head; (C) sortformer_diar driving a new ASR-only speaker-kernel model, trained on CPU for 1500 steps (the joint model had 2000); (X) the joint model's diarization driving the separate ASR; (D) oracle activity. Metrics: frame DER, cpWER, speaker-count accuracy, encoder passes and ms per utterance. Scripts: scratchpad/wf-verify/abl-spkasr/.

Params: joint model 2.71M (diar head 0.25M). Cascade: sortformer_diar 2.28M + ASR-only 2.47M = 4.75M. B deploys both whole models, 4.99M.
Single-target oracle WER (seed 2): joint ASR 4.5%, separate ASR (C) 13.4%.

| System | 2spk DER | 2spk cpWER | 2spk count acc | 3spk DER | 3spk cpWER | 3spk count acc | enc passes/utt (2spk / 3spk) | ms/utt (2spk / 3spk) |
|---|---|---|---|---|---|---|---|---|
| A: one model, self-steered | 7.2% | 12.5% | 100% | 27.8% | 55.4% | 0% | 3.0 / 3.0 | 53.4 / 45.3 |
| B: sortformer_diar -> joint ASR | 5.1% | 11.3% | 100% | 26.2% | 51.6% | 0% | 3.0 / 3.0 | 53.4 / 44.5 |
| C: sortformer_diar -> separate ASR (true 2-model) | 5.1% | 22.7% | 100% | 26.2% | 57.3% | 0% | 3.0 / 3.0 | 53.8 / 43.9 |
| X: joint diar -> separate ASR | 7.2% | 22.7% | - | 27.8% | 59.5% | - | 3.0 / 3.0 | - |
| D: oracle -> joint ASR (per-speaker WER) | 0 | 2.3% | - | 0 | 4.1% | - | 2.0 / 3.0 | - |
| D: oracle -> separate ASR (per-speaker WER) | 0 | 10.2% | - | 0 | 8.7% | - | 2.0 / 3.0 | - |

**Verdict: mixed.** The only clear gain is parameters: 2.71M against 4.75M, about 43% fewer, shipped as one artifact. There is **no compute saving**: both designs run one plain pass plus one conditioned re-encode per speaker, and wall time is the same. The shared encoder **hurts diarization** (DER 7.2% vs 5.1%). The best system is B, a separate diarizer driving the joint ASR, at 11.3% vs 12.5% cpWER; that gap is within noise. The joint ASR head does beat the separately trained ASR, but that comparison is confounded by training budget (1500 vs 2000 steps, final loss 0.06 vs 0.003). With 3 speakers, every diarizer fails to turn on a third slot. One model should be described as **smaller, not better**.

**Caveats.** C is budget-confounded. n=64 from one seed; the 1–2 point differences are noise. Soft activity is used for conditioning and binary targets for the oracle. Timings are from a loaded machine. The ablation also found a bug: train.py:164 returns early in evaluate, so a speaker-kernel ASR-only recipe logs `final {}`.

## 4. Confirmed edge-case bugs

The "fixed?" column is unknown for every row, because the fixes and deferred sections of the input were cut off. Rows marked **lead** are in files other agents may not edit (model.py, train.py, fastconformer.py).

| severity | file:line | title | fixed? | one-line fix |
|---|---|---|---|---|
| high | audioforge/data.py:210 | read_manifest ignores NeMo `offset`/`duration`, so every segment line loads the whole file (the teachers pseudo_label output has the same mismatch) | unknown | add offset/duration to load_wav, pass them when an `offset` key is present, shift RTTM times by the offset |
| high | audioforge/train.py:162 | A FrameHead with num_classes>1 crashes in evaluate (broadcast of (B,T,C) against (B,T)) before save_model, so the trained model is lost | unknown (lead) | branch on num_classes and use argmax against long labels; call save_model before evaluate |
| high | audioforge/data.py:248 | Collate takes its key set from batch[0], so mixed manifests either raise KeyError or silently drop prompts, depending on order | unknown | read_manifest gives every line a prompt when any line has one; Collate uses the union of keys and raises on partial keys |
| high | audioforge/data.py:40 | load_wav crashes on 24-bit PCM, float32/float64 and truncated WAVs that teachers.load_audio reads fine | unknown | read everything with soundfile (`sf.read(..., dtype='float32', always_2d=True)`), keep `wave` only as a fallback |
| medium | audioforge/data.py:255 | A FrameHead key other than the fixed list is dropped from the batch, so the head never trains, then evaluate raises KeyError before saving | unknown | collate every frame key a head needs; skip missing keys in evaluate (lead); save before evaluate (lead) |
| medium | audioforge/data.py:69 | rttm_to_frames silently drops speakers after the 4th, corrupting both targets and the DER reference | unknown | keep all speakers by default and warn; crop to num_spks by arrival order inside SortformerHead.loss |
| medium | audioforge/data.py:211 | Canary prompts ignore `taskname`/`pnc`, and task='transcribe' maps to `<|translate|>` | unknown | accept taskname and pnc; map transcribe/asr and ast/s2t_translation/translate explicitly; raise on anything else |
| medium | audioforge/teachers.py:247 | pseudo_label(wav_dir) overwrites WAV copies on name collisions (a_2, case differences on APFS, re-runs) | unknown | loop until the name is unused, compared case-insensitively and against files on disk |
| medium | audioforge/model.py:264 | StreamingSession and transcribe() ignore `from_layers: all` (heads are decoded from the top layer), so reported wer_* is also wrong | unknown (lead) | decode `head_input(head, enc, hidden)`; at minimum assert in StreamingSession as StreamingDiarizer does |
| medium | audioforge/model.py:266 | Same defect for streamed frame heads (VAD/EOU): streaming and offline disagree (duplicate of the row above) | unknown (lead) | same fix as above |
| medium | audioforge/modules/fastconformer.py:253 | att_context_size is silently ignored when att_context_sizes is given, so the saved config and the actual model disagree | unknown (lead) | default att_context_size=None; honour it when given and validate it is in (or append it to) att_context_sizes |
| medium | audioforge/losses/transducer.py:95 | TDT with no 0 duration: an item with no possible alignment gives NLL 1e30 and gradients that grow exponentially, until everything is NaN | unknown | `ll = torch.where(ll > NEG/2, ll, 0)` before `_reduce` (zero_infinity behaviour) |
| medium | audioforge/heads/audio.py:213 | CodecTokenHead with upsample>1 gives identical logits to every sub-frame | unknown | add a learned sub-frame embedding (or a linear upsampler) before the transformer |
| medium | audioforge/heads/audio.py:98 | SortformerHead with num_spks other than 4 crashes on every data source; PIL is silently disabled above 4 speakers | unknown | sort and crop/pad targets to num_spks; PIL over a cost matrix; at least warn |
| medium | audioforge/data.py:65 | rttm_to_frames ignores the file-id field, so a combined RTTM leaks speakers across recordings | unknown | filter lines by uniq_id (default the wav stem); raise ValueError on malformed lines |
| medium | audioforge/train.py:108 | Trainer.fit hangs forever when len(train) < batch_size (the batch range is empty) | unknown (lead) | the fix text was cut off; the obvious intent is to raise or clamp when len(train) < bs, or to allow a final partial batch |
| (ablation) | audioforge/train.py:164 | evaluate returns early (`if not others: return res`) before the condition_on_speaker oracle-WER block, so the log shows `final {}` | unknown (lead) | move the early return after the speaker-conditioned block |

## 5. Refuted or unreproducible reports; unverified findings

**Not received.** The input JSON was cut off before these sections. From the received text, these reported items could not be reproduced or checked:
- C11/C12/C13: the saved teacher outputs (`scratchpad/teachers/bench_fp32_bs8.json`, `hyps/*.jsonl`, `hyps_test`) do not exist. The workers regenerated them instead.
- C04: there is no saved SALM model, so the logged 9.1% model itself could not be re-evaluated (a re-run was done instead).
- C27: the 18 ms per step was not reproduced (~26 ms under load). `scratchpad/landscape/bench_m5.py` is not in the repo.
- All RTFx and peak-memory columns (MPS-only) and the 128-mel claim (C20) are unverified.
- The vendor-docs part of C25 (Speechmatics, AssemblyAI, Deepgram, Soniox) was not checked.

The lead should re-send the missing JSON sections (refuted and unverified findings, fixes, deferred) to complete this section.

## 6. Deferred fixes for the lead (model.py / train.py / fastconformer.py)

1. **train.py, Trainer.evaluate, frame branch (~l.162, 178–184):** `out = head.decode(enc, elen)`. If `head.num_classes > 1`: `p = out.argmax(-1); t = batch[head.key][:, :p.shape[1]].long()`. Otherwise keep `p = out > 0.5; t = ... > 0.5`. Keep the valid mask. For multi-class heads, compute recall per class c>0. Also skip frame heads whose key is not in the batch (to match model.py).
2. **train.py, run_recipe:** call `save_model` **before** the final `evaluate` (or wrap evaluate in try/except), so an evaluation error can never throw away a trained model.
3. **train.py:108, Trainer.fit:** when `len(train) < batch_size` (or train is empty), raise a clear ValueError, or allow a final partial batch, instead of spinning forever in `while step < self.steps`.
4. **train.py:164, Trainer.evaluate:** move `if not others: return res` below the condition_on_speaker oracle-WER block, so a speaker-kernel ASR-only recipe reports a metric.
5. **model.py, SpeechModel.transcribe (~l.151–155):** `enc, elen, hidden = self.encode(x, lens, att_context_size, spk_act, return_hidden=True)` and decode `self.head_input(head, enc, hidden)`.
6. **model.py, StreamingSession (~l.264–266):** minimal fix in `__init__`: `bad = [k for k in [self.head_name, *self.frame_events] if k in model.layer_mix]; assert not bad, f"streaming needs heads on the top encoder layer (no from_layers: all): {bad}"`. Proper fix: see item 8, then use `self.m.head_input(k, enc, hid)` for `_decode` and for every frame-head decode. Add a test that streaming equals analyze() for a head with from_layers: all.
7. **fastconformer.py:253, FastConformerEncoder.__init__:** default `att_context_size=None`. `sizes = [list(c) for c in (att_context_sizes or [att_context_size or (-1, -1)])]`. If att_context_size is given, set `self.att_context_size = list(att_context_size)` and append it to sizes if missing (or raise ValueError). Otherwise use `sizes[0]`, which keeps nemo_import's behaviour. Test: voice_agent_frontend with `encoder.att_context_size=[16,3]` should give `encoder.att_context_size == [16,3]` and `StreamingSession(m).chunk_mel == 32`. Note that scripts/bench_ondevice.py '70,6' currently runs at [70,1].
8. **fastconformer.py, stream_step:** add `return_hidden=False`, which returns the per-layer outputs already computed in the loop. This is needed for the proper fix in item 6.

The bugs in data.py, teachers.py, heads/audio.py and losses/transducer.py are not in lead-only files. Whether other agents fixed them is unknown (see §4).

## 7. Required documentation corrections (old → new)

### README.md
- [ ] l.9: "...and four new solutions..." → "...recipes that reproduce NVIDIA's models and three **new** solutions built from the same parts."
- [ ] l.30: recipes line → "research/recipes/                 YAML recipes (reproductions, new designs and LibriSpeech variants; SALM via CLI)"
- [ ] l.31: "56 tests (incl. batch-invariance...)" → "pytest suite (incl. batch-invariance: batched == single-utterance output)"
- [ ] l.39: "# 56 tests" → drop the comment, or give the current `pytest --collect-only` count (104 at the time of checking)
- [ ] Results table header → "| Recipe | Type | Params | Result on synthetic validation set (seed 1, n=64; same 8 speakers and 24-word lexicon as training, about 1/3 of transcripts also appear in train) |", and add the fresh-set re-check paragraph from C08 below the table (fresh-set numbers within ~1.5 points; speaker ID 100% on training speakers vs 38% on unseen ones; codec 25.1% vs 22.2%; SALM not re-checked)
- [ ] sortformer_diar row → "| `sortformer_diar` | reproduction | 2.3M | DER **4.0%** on 64 held-out 2-speaker mixtures; 4.7–5.7% on fresh 256-mixture sets. Frame-level, 80 ms, no collar, mean per utterance. Each mixture is one speaker turn A→B, and only ~3% of speech overlaps. Baselines: true speech regions split at the midpoint 15%, true change point 2.4% |"
- [ ] l.79 SALM row → "| SALM (`train-salm --steps 800 --encoder-from runs/parakeet_tdt_ctc.afm`) | reproduction | 2.6M | WER **9.1%** on 16 utterances (5/55 words; a CPU re-run gave 5.7% on 200 fresh utterances) (encoder initialized from Parakeet run, LM fine-tuned) |"
- [ ] voice_agent_frontend row → "| **`voice_agent_frontend`** | new | 2.4M | one pass: WER **0.0%**, VAD frame acc 91% (base rate 51%; errors are ±1-frame boundary shifts), speaker ID **100%** (closed set: the same 8 synthetic speakers seen in training). EOU is **not yet solved**: frame acc 88% is below the 92% always-"no EOU" baseline (precision 40%, recall 91%, F1 0.55), and the head fires before speech ends in 66% of utterances |"
- [ ] l.81 speaker_attributed_asr row → "| **`speaker_attributed_asr`** | new | 2.7M | DER 8.8%, per-speaker WER **3.0%** with oracle speaker activity; ~18% when conditioned on its own diarization output |". The ablation (§3.3) also suggests adding "(smaller than a cascade, not more accurate)".
- [ ] l.82 codec row: "(4 x 800-way, chance ~0.5%)" → "| **`codec_token_enhancer`** | new | 2.8M | clean-token accuracy **25%** from noisy input (4 x 800-way; uniform chance 0.125%, majority-code baseline 6%, re-encoding the noisy audio 1.6%; 27% on speech frames only); decoded log-mel L1 0.41 vs 0.70 for codes of the noisy input (clean-code ceiling 0.28) |". Per §3.2, also note that it does not beat spectral subtraction or a per-frame MLP on decoded mel.
- [ ] Optional Canary row: "AED WER **2.1%** on mixed ASR + "translation" prompts (4 errors / 189 words; fresh seeds 1.4–1.8%; swapping the prompt flips the output task)"
- [ ] l.102: "| streaming WER ~96% | raw log-mel (utterance normalization can't stream) + short schedule | fixed global normalization + 2k steps → 0.0% |" → "| streaming WER ~96% | utterance-level (per_feature) normalization can't stream, plus a short schedule (the two were changed together; no run isolates which one mattered) | frame-local normalization (here fixed global stats; NVIDIA's streaming FastConformer uses raw log-mel, `normalize: NA`) + 2k steps → 0.0% |"
- [ ] l.107: "about 0.3–2M-parameter models trained for about a minute each" → "These runs use about 2.3–2.8M-parameter models trained for about 1–2.5 minutes each on an Apple M5 (the later speaker_aware_turn and streaming_sortformer runs take longer), on a synthetic tone language."
- [ ] Quick start: `train-salm --steps 300` does not reproduce the SALM row; point to the command in cli.py:11.

### research/ANALYSIS.md
- [ ] l.1–8 card count → "(144 of 149 cards downloaded; 4 are gated (Nemotron-3-Diarization-preview, personaplex-7b-v1, Audio2Emotion-v2.2, Audio2Emotion-v3.0), and the card for nemotron-labs-audio-visual-flamingo-hf, which is public, was not fetched in this pull.)"
- [ ] Family table row → "| VAD, SLU, NEST, multitalker, animation, multimodal embedding | 12 | ~16,700 | mixed: FastConformer (multitalker, NEST, SLU), MarbleNet (VAD), audio-feature net (animation), LLM (embedding) | frame heads, RNNT, AED, blendshapes, embeddings |" (EOU belongs to the Parakeet/Nemotron row)
- [ ] l.31: "Since mid-2023, 51 of 53..." → "**Since mid-2023 (Hugging Face creation date ≥ 2023-06-01), 51 of 54 NVIDIA recognition-type models use the same FastConformer encoder.** The three exceptions are one small 2023 Conformer (stt_en_conformer_ctc_small), the multilingual frame-VAD MarbleNet v2.0 (a 1-D conv network), and PersonaPlex, which is built on Moshi."
- [ ] Front-end and encoder points → "1. **One front end.** 16 kHz mono audio → log-mel (25 ms window, 10 ms hop; 80 bands in the configs we checked) → SpecAugment. Feature normalisation varies (per_feature in Parakeet-CTC-0.6B, none in the cache-aware streaming models)... 2. **One encoder: FastConformer.** ... 80 ms frames (12.5 Hz) ... 17 layers (EOU 120M, streaming Sortformer v2/v2.1), 18 (NEST-L, Sortformer v1), 24 (Parakeet/Nemotron 0.6B, original Canary-1B), 32 (Canary-1B-v2)." (remove 128 mel; Canary-1B has 24 layers, not 32)
- [ ] l.77: "adds four new designs" → "adds three new designs"
- [ ] l.85: "NVIDIA ships these as 4 separate models that each re-encode the same audio." → "NVIDIA covers this with three models: a 91.5K MarbleNet VAD, a 120M ASR+EOU model (parakeet_realtime_eou), and a 117M Streaming Sortformer that re-encodes its [speaker cache; FIFO; chunk] window at every streaming step. TitaNet (23M) is added only for enrolled speaker ID." Per §3.1, also add: "One pass costs ~1/4 of the encoder compute at inference, but at equal training steps the shared encoder learns ASR and VAD much more slowly."
- [ ] l.86 speaker_attributed → "the Sortformer head and the speaker-kernel ASR share **one** encoder. At inference its own diarization output can steer per-speaker transcription (per-speaker WER ~18% end-to-end on the synthetic task, vs. 3.0% with oracle activity, the gap is diarization error propagating into ASR). NVIDIA needs two models for this (diarizer + one ASR instance per speaker); Microsoft's VibeVoice-ASR-Streaming does it in one 1.5B/7B LLM-based model, not a shared small streaming encoder."
- [ ] l.88: "frozen causal LM (the Canary-Qwen pattern)" → "FastConformer → frame-stacking projector → small causal LM, pre-trained on text and then fine-tuned together with the encoder (stands in for Canary-Qwen's LoRA-adapted LLM)". Also fix the stale train.py:389 docstring (lead).
- [ ] l.91–94 → "Getting there took six fixes, most of which mirror an NVIDIA practice: CTC alignment before AED, pretrained encoders for SALM plus LLM adaptation, frame-local (non-utterance) feature normalization for streaming (NVIDIA uses none; we use fixed global stats), balancing gradients into a shared encoder, reading speaker information from a learned weighted mix of all encoder layers (as in NEST), and normalizing latents before FSQ."
- [ ] l.100: "0.3–2M-parameter models" → "2.3–2.8M-parameter models"

### research/archive/TEACHERS.md
- [ ] Subset description → "LibriSpeech test-clean, first 200 utterances by sorted id (`1089-134686-0000` to `1221-135767-0012`, 30.3 min, 4634 words, only 4 of test-clean's 40 speakers)."
- [ ] Remove the bold from parakeet-ctc-1.1b's 1.64 and add: "The 0.6b-vs-1.1b CTC gap is 80 vs 76 errors and is not significant on this subset: in a paired bootstrap 1.1b wins only 68% of utterance resamples (95% CI of the difference -0.20 to +0.40 pp) and 63% of speaker resamples; 0.6b is better on speaker 1089. The 1.73/1.64 values were reproduced exactly on CPU; the RTFx and memory columns are MPS-only measurements from a single run."
- [ ] Card comparison → "They are close to each card's full test-clean number (2620 utterances): 1.87 / 1.83 / 1.93 / 2.32. Nemotron's 2.32 is its 1.12 s chunk setting ([70,13]); smaller chunks score 2.46 / 2.56 / 2.80. The cards use the Whisper English normalizer (Open ASR Leaderboard), which maps British to American spellings, and our normalize_text does not, so the comparison is approximate. At batch 1, parakeet-ctc-0.6b gives the same WER (1.73%) and identical transcripts on all 200 utterances, at RTFx 76 on MPS."
- [ ] TDT-v3 explanation → "**Why TDT-v3 scores worse here.** It writes British spellings where LibriSpeech uses American ones (colour/color, grey/gray, honour/honor, counseled/counselled, favourite, jeweller's), and it spells the name as daedalus instead of dedalus. These are 16 of its 93 errors (about 17%). Mapping them back gives 1.66% WER, level with parakeet-ctc-1.1b. Most of the remaining errors are compound words split or merged (woodbegirt, horseplay, watermill) and misheard rare words or names. The model is multilingual and writes punctuation and casing. TDT-v3 wrote no digits on this subset."
- [ ] l.38–39 → "The Nemotron model runs in the offline `generate` mode, which uses the default lookahead of 13 frames (1.04 s; with the current frame, a 14-frame chunk of 1.12 s, att_context_size [70,13]). That is the configuration the card's LibriSpeech number (2.32%, 1.12 s chunk) refers to."
- [ ] l.142 → "These models use full attention, so memory grows quadratically with length. (In transformers the parakeet relative positions are computed on the fly and are not capped at max_position_embeddings=5000 frames = 400 s; only the Nemotron streaming model raises past 5000 frames.)"
- [ ] l.136 (optional) → "labels about 2 hours per minute, at +0.09 WER (within noise on 200 utterances)."

### research/archive/GAME_CHANGER.md
- [ ] l.9: keep, but add "(VibeVoice-ASR-Streaming does speaker-attributed streaming ASR in one open model, but has no VAD, end-of-turn or small-encoder deployment; Alibaba UAF fuses VAD+turn+target-speaker ASR but has no released weights.)"
- [ ] l.10: "the best model needs 543 ms of dead air, the best commercial API 647 ms" → "the best system (LiveKit's hosted Turn Detector v1) needs 543 ms of dead air, JoinIn AI Baton 577 ms, the best STT API (Soniox) 647 ms, plain VAD 1,600 ms"
- [ ] l.11 and the thesis at 102–105 → "at ≤540 ms dead air @5% false cutoffs (≤290 ms @10%) on eot-bench, beating every system tested there, including the hosted LiveKit Turn Detector v1 (543/295 ms), JoinIn AI Baton (577/350 ms) and Soniox (647/512 ms)." To keep ≤600: "at ≤600 ms dead air @5% false cutoffs on eot-bench, beating Soniox (647 ms) and every other STT API tested there, though still behind the hosted end-of-turn models LiveKit Turn Detector v1 (543 ms) and JoinIn AI Baton (577 ms)."
- [ ] Credibility bar → "**Credibility bar:** LibriSpeech test-other ≤7% @160 ms (EOU-120M 7.79); AMI-IHM ≤15.62 and Earnings-22 ≤15.76 (no worse than EOU-120M @160 ms, per Bar A); DER ≤ Sortformer-v2 @0.32 s without post-processing (DIHARD3 20.19, CALLHOME-p2 13.57); VoxCeleb1-O EER ≤1.5% (TitaNet-L 0.66); full-pipeline RTF ≤0.3 on one CPU core."
- [ ] Compute paragraph → "**Compute is cheap:** 120M × 10k h × 30 epochs ≈ 1.3e19 FLOPs ≈ 44 A100-h FLOP-bound ($61–70 at RunPod $1.39–1.59/h; ~$180–350/run realistic at ×3–5); 600M × 100k h × 10 epochs ≈ 2.1e20 ≈ 730 A100-h ($1,020–1,160). This M5 trains a 105M encoder at ~4,500 audio-h/day (MPS, no data loading) and streams one 80 ms encoder step in ~18–26 ms on one CPU thread (18 ms measured once; 26 ms median under load on the shared machine)."; §5 table lower bounds: 146 A100-h → $203–233, 219 → $305–350, 731 → $1,016–1,163
- [ ] l.56: "best commercial API on eot-bench" → "best STT API on eot-bench"
- [ ] l.65–66 → "**Speaker-attributed streaming ASR in one open model now exists, but without turn signals.** Microsoft's VibeVoice-ASR-Streaming (1.5B/7B, MIT, released 2026-09, arXiv 2609.02812) streams who-said-what from one LLM-based model; it emits no VAD or end-of-turn. NVIDIA's version needs Sortformer (Nemotron-3-Diarization) plus multitalker-Parakeet or Nemotron 3.5 ASR, run as one ASR instance per speaker."
- [ ] l.74 (optional) → "streaming diarization (Nemotron-3-Diarization DIHARD3 13.55 vs Sortformer-v2.1 19.85 @0.32 s, same references; a large gap for a first multi-task 120M student to close)"
- [ ] l.76–77 → the full C26 paragraph: "**Compute saving depends on the latency operating point, not on counting 4 encoders:** NVIDIA already fuses ASR and end-of-utterance in one 120M cache-aware model (parakeet_realtime_eou_120m-v1), and the VAD is a 91.5K CNN, so the cascade has two real encoders: ASR+EOU (120M) and Streaming Sortformer (117M), plus an optional TitaNet (23M) for enrolled speaker ID. Streaming Sortformer re-runs its full 17-layer FastConformer and 18-layer Transformer over [speaker cache; FIFO; chunk] at every step. At the 0.32 s config (chunk 3, right context 1, FIFO 188, cache 188) that is about 127 encoder passes per frame (about 177 for Nemotron-3-Diarization, cache/FIFO 264), and about 65 at 1.04 s, versus about 1.8 at the 30.4 s offline-style config. The card's RTF (0.180 vs 0.002) shows the same ~90× gap. So a single cache-aware pass saves about 2× encoder compute only at ≥10 s buffers, and one to two orders of magnitude at 0.32-1 s latency. That saving holds only if the fused model matches diarization with its own long-range speaker memory (for example, an arrival-order cache of encoder states the diarization head attends to), which costs extra per-frame attention and is not yet demonstrated."
- [ ] l.81 → "a third-party test (Pine AI, arXiv 2609.04225, Fig. 2, as-shipped) measured Parakeet-EOU at recall 0.17"
- [ ] l.120 table: AMI-IHM / Earnings-22 target → "≤15.62 / ≤15.76"; cite EOU-120M 15.62 / 15.76 to https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1, not the leaderboard CSV
- [ ] l.121: add "JoinIn AI Baton 577 / 350" to the baselines. Change the target "≤600 / ≤350 ms" → "≤540 / ≤290 ms", or keep it and add "beats Soniox; trails LiveKit v1 and Baton"
- [ ] l.124 DER cell → "at 0.32 s: Sortformer-v2 DIHARD3 20.19, CALLHOME-p2 13.57 (https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2); Nemotron-3-Diarization DIHARD3 13.55 / CALLHOME-p2 11.32 (original references, same protocol as Sortformer-v2.1's 19.85 / 12.67 on the same card) / AMI-SDM 12.95 (forced-aligned references) (https://huggingface.co/nvidia/Nemotron-3-Diarization)"
- [ ] Codec link and table row → "**The link between the two sides:** FastConformer runs at 12.5 Hz (8x subsampling of 10 ms frames = 80 ms), and so do two of the three NanoCodecs (the 12.5 fps variants: 22,050/12.5 = 1,764 samples = 80 ms; the 21.5 fps variant, which Magpie-TTS prefers, does not match)..." plus the row text: "Our experiment uses its own 16 kHz FSQ MelCodec (8 stacked 10 ms mel frames), not NanoCodec, so the NanoCodec pairing is still untested. Codec-token speech enhancement has prior art (e.g. SELM, MaskSR, Genhancer, LLaSE-G1); what is new here is only the FastConformer frame-matched, single-pass NAR variant."
- [ ] l.212: "read speaker/diarization from middle layers" → "read speaker/diarization from a learned mix of all layers"

## Note on workflow size

The user asked, relayed with this task, "r u sure we need 254 agents?" Across the reports, the workers agree. Each claim check needed one agent and a few greps, web fetches, or a CPU run of 2–12 minutes. The three ablations were the only heavy jobs (19–50 CPU-minutes each). At the start of one ablation the 15-minute load average was ~37 on 10 cores, and swap filled to 35.8/35.8 GB during another. The work in this report could have been done by **about 10–15 agents**, with claims batched by document, run **sequentially** on this shared machine. 254 agents was far more than needed, and the parallel fan-out is a likely cause of the overload.
