# Head architectures: what the best systems do, and what to build on our frozen encoder

2026-09-30. Research only: web and repo reading. No training, no torch jobs.

**Scope.** For each head on the frozen NVIDIA 115M cache-aware streaming FastConformer, this note covers:
- what the best published and shipping systems do today;
- what of that transfers to a small streaming head on frozen per-layer features;
- a concrete proposal that addresses our measured failures.

**Constraints on every proposal:**
- The encoder has 17 blocks, 160 ms chunks and 80 ms frames. Per-layer outputs are available.
- Heads must stream with small per-frame state.
- Training runs on a Mac from cached features.
- The shipped heads are in `assets/served_heads_v0.2.pt`.

**How numbers are sourced:**
- An external number carries its URL.
- An internal number carries its repo file.
- *[abstract]* means the number was read from an abstract or a summary page, not checked against the paper's table.
- *UNVERIFIED* means we could not confirm it at the source. Those numbers are not used to justify any proposal.
- Expected gains are our estimates. Each one names the evidence it rests on.

## Summary

| head | today (measured) | best-known approach | proposal | expected gain | added cost / 160 ms chunk |
|---|---|---|---|---|---|
| **1. VAD** | block-4 MLP, 33 K params. AUC 0.972 on AMI (Silero 0.956, MarbleNet 0.959). But it reads **0.66 on −50 dBFS room tone** and ~0.55 at session start. Tail after speech ends: 212-320 ms at 0.5 (Silero: about 0). | **Silero v6:** noise-only false alarms cut sharply (ESC-50 accuracy 0.61 → 0.87); recipe undisclosed. **Braun & Tashev:** audibility (VNR) targets plus stationary-noise and level augmentation fix "quiet office" false positives. **MarbleNet v2:** 330 h noise, volume perturbation. | VAD v3, same tap. Three changes: <br>• train on room-tone and noise-floor augmented audio at −70..−35 dBFS; <br>• label a frame by speech at its centre, not "any part of the frame"; <br>• add an audibility (VNR) output and a noise-floor-relative energy input. <br>Tiny causal conv, ~40 K params, 2-frame state. | • Room tone read as silence, so our silence clock starts when the user stops, not at the digital pad (assistant clips: 990 ms answered from clip end, 240 ms median room tone). <br>• Removes the pre-speech fallback turn_ends (137 / 399 sessions). <br>• **No** calls-EOT gain claimed: VAD_TAIL showed a tail-free VAD does not speed EOT at equal FI. | < 0.05 ms, no extra encoder |
| **2a. Speaker embedding** | block-4 relational head, 0.50 M params. Within-meeting EER: AMI 14.4 / 19.8 %, ICSI 4.7 / 5.2 %. TitaNet-L: 8.2 / 12.0 % and 1.4 / 1.0 %. | **ReDimNet B0-B6:** 1.16-0.40 % EER on Vox1-O at 1-15 M params. **MHFA / CA-MHFA** on a frozen SSL backbone: 1.77-1.87 % Vox1-O. Bottom SSL layers carry speaker identity. | MHFA-style pooling (keys and values from separate learned mixes of blocks 1-6). Multi-teacher distillation: TitaNet-L plus one Apache-licensed embedder, with the relational loss kept. ~0.8 M params. | AMI within-meeting EER: modest, 1-3 points (estimate). SPK_HEAD bounds it: the block-4 features cost 6-8 points against TitaNet. Main value is a better print for 2b. | only at enrollment and adaptation |
| **2b. TS-VAD** | block-4 FiLM + cosine pre-net + GRU, 0.26 M params, 5 s print. <br>• Frame F1: ICSI 0.882, AMI 0.743. <br>• Misses **32-38 % of target frames in overlap**. <br>• A 1.5 s print costs 5-7 F1 points. <br>• Target-speaker WER: ICSI 37.1 % (oracle mask 31.3 %), AMI 61.6 % (oracle 47.4 %). | **PVAD 2.0:** FiLM on the pre-net cosine, enrollment dropout. **Online TS-VAD:** a running target buffer filled from confident, non-overlap frames. **Sortformer / Nemotron:** AOSC speaker cache, pretrain on simulation then fine-tune on real. **Mind the Gap:** synthetic → real gives 15.5 vs 17.4 macro DER. | TS-VAD v2: <br>• learned mix of blocks 1-6; <br>• FiLM kept, plus cross-attention to a cache of enrolled and confident target frames (AOSC-like, 32 frames); <br>• overlap-weighted loss; <br>• pretrain on simulated conversations (LibriSpeech-100 + AMI/ICSI single-speaker segments + room tone), then fine-tune on the 24 real meetings. <br>~0.4 M params, GRU 128 state. | • Overlap miss down, toward the Sortformer column's 13-30 %. <br>• Short-print penalty reduced: ASE got +1.6 to +4 F1 at 0.5-1.5 s. <br>• tWER headroom is bounded: ICSI 5.8 points, AMI 14.2 points to the oracle mask. | ~0.1 ms plus a 32×128 cache attention, no extra encoder |
| **3. LID** | blocks 8-12, causal attentive-stats pooling, AmberNet-distilled, 0.92 M params. FLEURS-17: **91.2 % at 2 s** / 97.7 % full utterance (AmberNet 95.1 / 99.5 %). EdAcc 65.8 %. | **Google streaming LID in RNN-T:** running mean+std pooling, ~1 M params, mid-encoder layers. That is our design. **Kukk & Alumäe:** fusing an ASR-text classifier fixes accented English (L2-ARCTIC 74.6 → 90.5 %). **NVIDIA:** an English-only encoder carries much less language information. | Keep the head. Add three things: <br>• RNNT-derived features: blank rate, token confidence, and an English char-n-gram likelihood of the partial; <br>• AmberNet + SpeechBrain-ECAPA teacher ensemble; <br>• VoxLingua107 (CC BY 4.0) windows for speaker diversity; <br>• taps widened to blocks 6-13. | • FLEURS 2 s: +1-2 points (estimate). The gap to AmberNet is unlikely to close on an English-only encoder. <br>• EdAcc: large (35-63 % relative error reduction in Kukk & Alumäe). | ~0.25 ms |
| **4. End of turn** | GRU on a speaker-conditioned second encoder pass. <br>• Assistant-directed accuracy **41 %** (smart-turn classifier 97.0 %). <br>• Calls EOT p50 956 ms. <br>• TurnBench median delay 1316 ms (VAP 463 ms). <br>• End-vs-pause frame AUC 0.59-0.66. | **smart-turn v3.x:** Whisper-tiny encoder + linear head on the last 8 s, 270 k clips in 23 languages, human data added in v3.1 (English 88.3 → 94.7 %). **LiveKit audio turn detector:** audio→LLM branch plus prosody RNN. **VAP:** 256-class future-activity projection (TurnBench test 0.845 / 0.055 / 368 ms). **Deepgram Flux:** fused ASR + EOT with an eager event. | Three parts: <br>• **(a) Segment classifier** fired at the quiet trigger: attention pooling over ≤ 8 s of pass-1 blocks, plus RNNT text features and prosody. Trained on **smart-turn-data v3.2 train, now CC BY 4.0 (changed 2026-09-29)**, human_5_all, oto and synthetic cuts. <br>• **(b) VAP-style multi-horizon bins** as an auxiliary output and a two-party trigger. <br>• **(c) Soft targets from a text-LM teacher.** <br>~0.3-0.6 M params. | • (a) Assistant accuracy from 41 % toward ≥ 90 %. Our completeness head already reached frame AUC 0.995 on smart-turn clips from 3.5 k clips (COMPLETENESS.md). <br>• (b) Two-party EOT p50 −200 to −375 ms at matched FP (DYADIC predictive trigger: 961 vs 1336 ms). This **regressed on meetings**, so it is gated by mode. | (a) ~1 ms once per pause; (b) < 0.1 ms per frame |
| **5. New: barge-in / backchannel intent** | none. Any VAD onset during agent speech is treated as an interruption. | **LiveKit adaptive interruption:** audio encoder + CNN on overlap audio, 86 % precision / 100 % recall at 500 ms of overlap, rejects 51 % of VAD barge-ins. **VAP / TurnBench** interruption task: VAP recall 0.945 at FPR 0.107. | A 3-way classifier (backchannel / real interruption / noise-echo) on the first 240-560 ms of user speech while the agent talks. Inputs: pass-1 blocks, TS-VAD P(user), the first RNNT tokens. Trained on oto (real overlaps, CC BY) and Behavior-SD (CC BY, TTS, labelled backchannels and interruptions). ~0.15 M params. | Fewer false barge-ins. We have no local measurement yet; LiveKit reports 51 % of VAD barge-ins rejected. Needs a new metric: false-barge-in rate on oto and on the TurnBench interruption task (eval only). | < 0.1 ms, only while the agent speaks |
| **5b. Overlap / second-voice flag** | implicit in TS-VAD [P(target), P(other)] | a separate 7-class powerset (pyannote) or multi-sigmoid (Sortformer) | No new head. Expose `overlap = P(target)·P(other)` from TS-VAD v2, and train its overlap frames with the higher loss weight above. | Comes with 2b | 0 |

## Prioritized build list

The order goes by measured-failure size × evidence strength ÷ cost. Every item runs head-only on cached features, one
MPS job at a time behind `scripts/dev/gate.sh`. Each item gets a pre-registered bar before training, as in earlier
notes.

1. **EOT segment classifier on smart-turn v3.2 train (item 4a).**
   - The largest measured gap: 41 % vs 97 % on assistant speech.
   - The strongest transfer evidence: our own completeness head reached 0.995 frame AUC on this task, and smart-turn
     reports +6.4 points in English from adding human data.
   - The training set's licence blocker is gone (see "Data").
   - This extends the in-flight turn v5 work (`scripts/research/turn_v5.py`, segment classifier over blocks 4/8/12/17
     + RNNT text + prosody), so hand it to that agent rather than starting a parallel line.
   - Bar: ≥ 90 % accuracy on the 399 held-out human_5 test clips through the served `vad_head` rule; calls and AMI FI
     and misses no worse than the shipped rule.
2. **VAD v3 with room tone and audibility targets (item 1).**
   - Fixes a measured 0.66-on-room-tone failure and the session-start false turn_ends.
   - It is the input every EOT path reads, and the energy gate in flight (`EnergyGate`) is a rule-level patch for the
     same fault.
   - Cheap: 2.7 min per head on MPS (VAD_SINGLE.md), plus one re-encode of the augmented audio.
   - Bar: median VAD < 0.2 on room tone of the assistant clips; AUC ≥ 0.970 on AMI and ICSI dev; no downstream regression.
3. **VAP-style bins on the turn head, two-party mode only (item 4b).**
   - DYADIC measured 961 vs 1336 ms at matched FP on TurnBench halves.
   - A pass-1 version was killed on meetings (FINAL_REPORT §5.8), so ship it only behind `--mode single` on a user
     channel, and evaluate it on calls, AMI and assistant clips separately.
4. **TS-VAD v2 with simulated-conversation pretraining and a target-frame cache (item 2b).**
   - Addresses overlap misses and the short-print penalty.
   - The ICSI tWER headroom is only 5.8 points, so it sits below the EOT items.
5. **Barge-in intent head (item 5).**
   - Needed by any voice agent with interruptions.
   - Needs a metric first (false-barge-in rate); build the eval before the head.
6. **Text-LM teacher for turn completeness (item 4c).** A training-time enrichment of items 1 and 3. The Qwen2.5-7B
   log-prob stage already exists in turn_v5 (`lm`).
7. **Speaker head MHFA + multi-teacher (item 2a).** Improves the print that 2b consumes. Small expected gain.
8. **LID text fusion + teacher ensemble (item 3).**
   - The voice-agent target is a known English speaker (project memory), so LID is off by default and matters least.
   - Text fusion is nearly free and has the best evidence for the accented-English gap.

---

## Data available locally, and licences

Inventory from research/TURN_V4.md §2 and the `LICENSE.md` files under `/Volumes/ExternalSSD/nvidia-audio-models/data/`.
599 GB is free on the SSD; the internal disk has 84 GB free.

| source | size here | licence | usable for |
|---|---|---|---|
| AMI (12 train meetings local of 136) | 4.1 GB | CC BY 4.0 | VAD, TS-VAD, speaker, turn |
| ICSI (12 train meetings local) | 5.2 GB | per corpus card (see ICSI.md) | VAD, TS-VAD, speaker, turn |
| otoSpeech-141h (two-channel full duplex) | 23 GB | CC BY 4.0; **no speaker-identity training** | turn, VAP bins, barge-in, VAD. Not for speaker or TS-VAD identity supervision |
| LibriSpeech (train-clean-100 + test) | 57 GB | CC BY 4.0 | simulated conversations for TS-VAD and speaker |
| DailyTalkContiguous (stereo dialogue) | 2.6 GB | CC BY-SA 4.0 | turn, VAP bins (TTS-like read dialogue) |
| Behavior-SD (test split local; 2,164 h in full) | 1.4 GB local | CC BY 4.0 | backchannel / interruption labels (synthetic speech) |
| pipecat-ai/human_5_all | local | BSD-2-Clause | turn (3,463 train + 399 test clips) |
| **pipecat-ai/smart-turn-data-v3.2-train / -test** | not local (41.4 GB + 4.8 GB) | **CC BY 4.0 since 2026-09-29** (HF commits "Add CC-BY-4.0 license" at 15:07 UTC and "Fix LICENSE link in README" at 15:16 UTC; the card's `license: cc-by-4.0`). COMPLETENESS.md §1 (2026-09-26) found no licence; that finding is now out of date. | turn: 270,946 clips, 23 languages, `endpoint_bool`, `midfiller`, `endfiller`, `synthetic` columns. The English subset is the target. Holding out ids that overlap human_5 test is required |
| FLEURS-17 (+ all train), EdAcc, extra English | 15 GB | CC BY 4.0; EdAcc CC BY-SA | LID |
| TurnBench dev | 5.5 GB | Mundo AI DPL v1.0: **evaluation only** | eval only (EOT + interruption tasks) |
| NeMo checkpoints: TitaNet-L, AmberNet, MarbleNet v2, Sortformer v2/v2.1, Nemotron-3-Diarization, Parakeet-EOU, TDT v3, 0.6B streaming | 14 GB | NGC ToU / NVIDIA Open Model License | teachers (offline label generation) |

**Not local, licence-clean, worth fetching for items 1 and 2** (licences checked at source by the VAD research pass):

| corpus | licence | source |
|---|---|---|
| MUSAN | CC BY 4.0 | https://www.openslr.org/17/ |
| OpenSLR 28 (real and simulated RIRs + isotropic noise) | Apache 2.0 | https://www.openslr.org/28/ |
| OpenSLR 26 (simulated RIRs) | Apache 2.0 | https://www.openslr.org/26/ |
| BUT ReverbDB (includes per-room background recordings, i.e. real room tone) | CC BY 4.0 | https://speech.fit.vut.cz/software/but-speech-fit-reverb-database |
| DEMAND | CC BY 4.0 on Zenodo | https://zenodo.org/records/1227121 |
| VoxLingua107 | CC BY 4.0 (video copyright stays with the owners) | https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds |

Avoid WHAM! noise: it is CC BY-NC 4.0 (https://ar5iv.labs.arxiv.org/html/2005.11262).

---

## 1. VAD

### What we measured
- **Ranking is good; the calibrated level is not.** AUC is 0.972 on AMI (Silero v5 0.956, MarbleNet v2 0.959) and
  F1 0.951 (METRICS.md rows 11-12).
- **Room tone.** The head reads **0.66 (median) on −50 dBFS room tone** after the last word of the assistant clips.
  It reads ~0.55 on leading silence (EOT_ASSISTANT.md). Consequences:
  - the `vad_head` silence clock starts only at the digital pad;
  - 137 of 399 balanced sessions get a fallback turn_end before the user speaks.
- **Tail.** At 0.5 the tail is 212 ms p50 on calls and 320 ms against its own AMI labels. Of that, ~40 ms is the
  "any part of the 80 ms frame" label grid; the rest is the head (VAD_TAIL.md §1).
- **Removing the tail does not speed calls EOT.**
  - An oracle tail-free VAD with any hangover is worse than the served VAD at equal FI (VAD_TAIL.md §2).
  - The reason: the tail is a uniform hangover that also bridges 67 % of mid-turn pauses.
  - So this head is **not** a calls-latency lever. It is a robustness fix: room tone, session start, noisy rooms.

### What the best systems do

**Silero VAD** (https://github.com/snakers4/silero-vad; https://api.github.com/repos/snakers4/silero-vad/releases):
- About 2 MB, under 1 ms per 30 ms chunk on one thread, MIT licence.
- v5.0 (2024-06): "5-7 % quality increase on clean data".
- v6.0 (2025-08): "16 % less errors on noisy real-life data; 11 % less errors on multi-domain validation".
- v6.2 (2025-11): "VAD training paradigm reworked".
- No training data or noise recipe is published.

Silero's quality wiki (https://raw.githubusercontent.com/wiki/snakers4/silero-vad/Quality-Metrics.md):
- **Noise-only accuracy**, ESC-50 / private noisy calls: v5 0.61 / 0.44, **v6 0.87 / 0.71**, TEN VAD 0.42 / 0.47.
- **Multi-domain ROC-AUC:** v5 0.96, v6 0.97.
- So v6's gain is mostly false-alarm rejection on noise, which is our failure.
- Caveat: the wiki's v5 and v6 AUC rows are identical except the multi-domain cell, which looks like a copy error.

Silero post-processing (`utils_vad.py`): exit threshold = entry − 0.15 (0.35), min_silence 100 ms, speech_pad 30 ms.
The LSTM state is (2, B, 128). The architecture (STFT → 4 conv → LSTM 128 → conv decoder, ~309 K params) is known
only from third-party reverse engineering (https://huggingface.co/aufklarer/Silero-VAD-v5-MLX), not from Silero.

**NVIDIA Frame-VAD MarbleNet v2.0** (https://huggingface.co/nvidia/frame_vad_multilingual_marblenet_v2.0):
- 91.5 K params, 20 ms frames.
- Data: 2600 h real + 1000 h synthetic + **330 h noise** (MUSAN, Freesound, …).
- Augmentation: "white noise and real-word noise perturbations … the volume of audios was also varied".
- ROC-AUC: AMI-test 96.25, VoxConverse-test 96.65, CH109 94.44.

**pyannote segmentation-3.0** (https://huggingface.co/pyannote/segmentation-3.0): a 7-class powerset over 10 s windows.
It publishes no VAD numbers.

**TEN VAD** (https://github.com/TEN-framework/ten-vad):
- 10 / 16 ms hop.
- Claims faster speech-to-non-speech transitions than Silero, shown in a figure only, with no number.
- Scored below Silero v6 on Silero's wiki.

**Braun & Tashev, EUSIPCO 2021, "On training targets for noise-robust VAD"** (https://arxiv.org/pdf/2102.07445). The
single most relevant paper:
- **Data:** 544 h speech, 247 h noise plus **1 h of coloured stationary noise**, 7000 RIRs. Levels drawn from
  N(−28, 10) dBFS.
- **Target:** they replace the presence label with a **segmental voice-to-noise ratio (VNR)** target.
- **AUC, BCE-VAD vs VNR:** KAIST 95.9 → 98.8, HAVIC 86.8 → 88.1, AVA 92.4 vs 92.3.
- **Their Fig. 3 is our failure:** in a quiet office with slight ambient noise, the BCE-VAD output "becomes quite
  large" while VNR stays low.

**Endpointing on ASR-encoder features:**
- Bijwadia et al., SLT 2022 (https://arxiv.org/abs/2211.00786): a joint ASR + endpointer that reads low-level encoder
  latents. Median endpoint latency −120 ms (−30.8 %), P90 −170 ms, WER unchanged.
- Shannon et al. 2017 (https://www.isca-archive.org/interspeech_2017/shannon17_interspeech.html): an end-of-query
  target instead of a VAD target, ~100 ms lower latency at equal accuracy.

Also seen (all 2025-2026, *[abstract]*):
- SincQDR-VAD (https://arxiv.org/html/2508.20885v1): 8 K params, pairwise ranking + BCE loss, white-noise
  augmentation at −90 to −46 dB re peak. AVA AUROC 0.914 vs MarbleNet 0.858.
- S4VAD (https://arxiv.org/abs/2609.11110): argues AUROC ignores timing, and controls the decay of past evidence.
- Next-Turn (https://arxiv.org/abs/2606.18094): regresses time-to-next-onset.

### What transfers to a head on frozen features
- **Data, not capacity.**
  - Both Silero v6's noise jump and MarbleNet's recipe put noise-only audio and level variation into training.
  - Our head saw only AMI, synthetic conversations and the 16 s AMI windows, with no low-level stationary noise
    (VAD_SINGLE.md "Training").
  - Augmentation must go through the frozen encoder, so it needs a re-encode of the augmented audio. That is a cached
    feature job, not a training-loop cost.
- **Target.** Braun & Tashev's audibility target fixes exactly the quiet-room false positives. It can be an extra
  output: keep BCE on presence, and add a VNR regression head trained with MAE.
- **Labels.** Our "any part of the 80 ms frame" rule biases the head late by up to 80 ms at offsets (VAD_TAIL.md §1).
  Label by the frame centre or by majority instead.
- **Tap.** Block 4 is already the best single tap (VAD_SINGLE.md). S4VAD suggests testing block 2-3 for sharper
  offsets. Cheap to probe on the existing `scratch/vad_tail/b4` style cache.
- **Energy.** The in-flight `EnergyGate` uses a per-session noise floor (the 10th percentile of 80 ms log energies
  over 3 s). Feeding the head that same "dB above floor" scalar gives it the level information a normalised
  log-mel encoder removes.

### Proposal: VAD v3
- **Architecture.** Block-4 input (optionally a learned mix of blocks 2-4), LayerNorm, then a causal depthwise conv
  (kernel 3), a 64-unit MLP and two outputs:
  - P(speech), BCE;
  - VNR in dB, MAE, clipped to [−15, 40].
  - One extra scalar input: frame energy relative to the running noise floor (EnergyGate's statistic).
  - ~40 K params. State: 2 frames of conv history plus the floor estimator.
- **Data.** The existing mix (synthetic conversations, AMI turn and diar windows), plus these, re-encoded once through
  the frozen encoder:
  - **room-tone / noise-only segments at −70..−35 dBFS**: BUT ReverbDB room recordings, MUSAN noise, DEMAND,
    OpenSLR 28 isotropic noise, generated coloured noise, and the quiet stretches of oto and AMI;
  - leading and trailing silence padding at session start;
  - speech mixed at SNR ~ N(5, 10) dB and level ~ N(−28, 10) dBFS (Braun & Tashev's distributions);
  - RIRs on 50-80 % of speech.
- **Labels.** Frame centre ± 20 ms. The VNR target comes from clean-speech and noise energies, available for the
  simulated mixtures. On real data, mask the VNR loss.
- **Post-processing.** Silero-style hysteresis (exit = entry − 0.15) in the rule, instead of a learned hangover.
- **Expected gain:**
  - Room tone is read as silence, so on the assistant clips the silence clock starts at the audible end instead of
    the pad (240 ms median of room tone, p90 789 ms; EOT_ASSISTANT.md).
  - The pre-speech fallback turn_ends (137 / 399) go away.
  - Evidence: Silero v6 noise-only accuracy +0.26 (ESC-50); Braun & Tashev Fig. 3.
  - **No calls-EOT gain is claimed** (VAD_TAIL.md §2).
- **Cost.** < 0.05 ms per chunk (the served head is 0.013 ms), no extra encoder. Training: one augmented re-encode
  (hours on MPS, SSD) plus ~3 min per head.
- **Risk.**
  - TS-VAD arming, LID pooling and the turn hint all read this VAD, so VAD_SINGLE.md's downstream battery must be
    rerun.
  - A lower tail could raise calls FI under the existing rule constants. Re-scan them with `eot_latency.py sweep`.
- **Addresses:** room tone 0.66, session start 0.55, and part of the tail (the label-grid share).

---

## 2. Speaker embedding and target-speaker VAD

### What we measured
- **Speaker head** (SPK_HEAD.md): block-4 relational head, 0.50 M params.
  - Within-meeting EER: AMI 14.4 % (n = 64) / 19.8 % (n = 200); ICSI 4.7 / 5.2 %.
  - TitaNet-L: 8.2 / 12.0 % on AMI and 1.4 / 1.0 % on ICSI.
  - The block-4 features account for the remaining 6-8 points on AMI.
- **TS-VAD head** (IMPROVE_115M.md A.1), 5 s print:
  - F1 0.882 on ICSI and 0.743 on AMI.
  - It **misses 32-38 % of target frames in overlap**. The Sortformer / Nemotron column misses 13-30 %.
  - A 1.5 s print costs 5-7 F1 points.
- **Target-speaker WER** (TSWER.md): ICSI 37.1 % (oracle mask 31.3 %), AMI 61.6 % (oracle 47.4 %). The AMI gap is
  mostly leaked crosstalk: insertions 15.1 vs 5.1 %.
- **Training data:** 12 AMI + 12 ICSI meetings.

### What the best systems do

Speaker embedding:
- **ReDimNet** (https://github.com/IDRnD/redimnet/blob/master/EVALUATION.md, *[abstract]*-level check): Vox1-O EER
  1.16 % (B0, 1.0 M params) to 0.40 % (B6, 15 M).
- **WeSpeaker** (https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md): ResNet34 0.797 %
  at 6.6 M params, ResNet293 0.532 %, CAM++ 0.707 % at 7.2 M, ECAPA c1024 0.798 %.
- **ERes2NetV2** (https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.pdf, table checked): 0.61 %
  full-length; at 2 s it is 1.48 %, vs ECAPA 1.95 %.
- **TitaNet-L** (https://huggingface.co/nvidia/speakerverification_en_titanet_large): 0.66 % on VoxCeleb1.

Heads on frozen encoders:
- **MHFA / CA-MHFA** (https://arxiv.org/html/2409.15234, *[abstract]*): with the backbone frozen, Vox1-O EER 1.78 %
  (MHFA) and 1.77 % (CA-MHFA) on WavLM-Large. The back-end is ~2.3 M params. Keys and values come from separate
  layer mixes.
- **WavLM layer analysis** (https://arxiv.org/pdf/2110.13900, Fig. 3): speaker information comes "mostly from the
  bottom layers", consistent with our block-4 finding.
- **Frozen ASR Conformer + adaptor** (Cai & Li, https://arxiv.org/abs/2309.03019, *[abstract]*): about 0.43-0.48 %
  Vox1-O with a ~4.9 M adaptor. The versions of the paper disagree on the exact number.

Personal VAD:
- **PVAD 2.0** (https://arxiv.org/pdf/2204.03793, table checked): 4 causal conformer layers of dim 64, a conformer
  speaker pre-net, and FiLM on the pre-net cosine. Conversational (concat) WER:
  - PVAD 1.0: 41.0;
  - FiLM on the d-vector: 29.5;
  - **FiLM on the pre-net cosine: 27.5**;
  - int8: 1.0 MB.
- **Enrollment-less training** (PVAD 2.0): with p = 0.2, zero the embedding and relabel non-target as target. We
  already do this with p = 0.15 and a learned null vector.
- **Adaptive embedding self-augmentation** (https://arxiv.org/pdf/2601.12769, checked): target F1 at
  1.5 / 1 / 0.5 s enrollment goes 85.22 → 86.87, 83.66 → 86.16 and 80.28 → 84.30. Iterative refresh lets 0.5 s match
  a full enrollment after 5 iterations. λ = 0.05-0.1.

TS-VAD:
- **TS-VAD, CHiME-6** (https://arxiv.org/pdf/2005.07272, checked): single-channel DER 35.80 / 39.80 (dev / eval) vs
  47.29 / 60.10 for x-vector + SC. Refining the profiles in a second iteration helps.
- **Online TS-VAD** (https://www.isca-archive.org/interspeech_2022/wang22j_interspeech.pdf, checked):
  - A target buffer of running-mean frame embeddings, updated only when P > 0.7 and not in overlap.
  - AliMeeting DER 8.14 / 11.42 (eval / test) vs offline clustering 13.79 / 14.54.
  - Pretrained on simulation, then fine-tuned on real.
- **Representation study** (https://arxiv.org/abs/2410.11243, *[abstract]*): speaker-verification quality is
  "somewhat unrelated" to TS-task performance.

NVIDIA diarizers and speaker-conditioned ASR:
- **Streaming Sortformer v2.1** (https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1, checked): 117 M
  params, AOSC speaker cache. AMI IHM DER at 1.04 s: 25.11 (v2) → 16.67 (v2.1), from adding real meeting data.
- **Nemotron-3-Diarization** (https://huggingface.co/nvidia/Nemotron-3-Diarization, checked): ~10 k h real + 82,611 h
  simulated. AMI MHM DER 9.48 at 1.04 s, 10.05 at 0.32 s. AMI is in its training data.
- **AOSC** (https://arxiv.org/html/2507.18446, *[abstract]*): keeps the top-scoring frames per speaker in a
  188-frame cache, with a minimum quota per speaker.
- **Self-speaker adaptation** (https://arxiv.org/html/2506.22646v1, *[abstract]*): a Sortformer activity mask on a
  1.1 M "speaker kernel" at the ASR pre-encode. That is the mechanism our pass-2 speaker kernels already use.

Simulated data:
- **Mind the Gap** (https://arxiv.org/html/2605.15442, local copy `research/papers/2605.15442.txt`): Sortformer macro
  DER over 8 sets is synthetic-only 22.2, real-only 17.4, mixed 16.3, **synthetic → real 15.5**. Boosting overlap
  hurts diarization (26.1 → 27.6). Diverse sources beat an exact domain match.
- **NeMo property-aware simulator** (https://www.chimechallenge.org/workshops/chime2023/papers/CHiME_2023_GENERAL_park.pdf,
  checked): simulated AMI statistics match the real ones (silence 0.1804 vs 0.1814). Gain perturbation and noise are
  "necessary" for VAD.

### What transfers
- **Taps.** Bottom layers carry speaker identity. Use a learned mix of blocks 1-6, not only block 4, with MHFA's
  separate key and value mixes.
- **Conditioning.** Keep FiLM on the pre-net cosine; PVAD 2.0 found it best on conversational speech. Add an adaptive
  target memory, which three independent lines point to:
  - Online TS-VAD's confident-frame buffer;
  - AOSC's top-frame cache;
  - ASE's embedding self-augmentation.
  Our anchored 0.8 / 0.2 print blend (TSWER.md fix) is a crude form of this.
- **Data.** Pretrain on simulation, then fine-tune on real. That is the consistent recipe (Mind the Gap,
  Online TS-VAD, Nemotron). For overlap misses, weight the loss on overlap frames rather than simulating
  overlap-heavy mixtures (Mind the Gap).
- **Do not expect TitaNet parity** from a 0.5 M head. Frozen-backbone heads land at about 3x the EER of fine-tuned
  ones even on WavLM (MHFA).

### Proposal: TS-VAD v2 (and a speaker head v2 to feed it)

**TS-VAD v2:**
- **Input.** Learned mix of blocks 1-6, projected to 128.
- **Conditioning.** FiLM from the print cosine (kept). Add a single-head cross-attention from each frame to a
  **target memory** of 32 × 128 vectors:
  - the enrolled print's own block-mix frames (5 s = 62 frames, subsampled to 16);
  - 16 slots refreshed from frames with P(target) > 0.7 and P(other) < 0.3 (Online TS-VAD's rule);
  - kept anchored to the enrolled half, as the served adaptation already is.
- **Temporal model.** Causal GRU 128 (kept).
- **Outputs.** [P(target), P(other)], plus the overlap flag as their product.
- **Size.** ~0.4 M params. Per-frame state: GRU 128 + memory 32 × 128 (16 KB in fp32).

**Training data:**
- **Stage 1, simulation.** Conversations built with NeMo-simulator-style statistics (CALLHOME / AMI turn-taking,
  silence ~18 %, overlap ≤ 15 %) from three sources:
  - LibriSpeech train-clean-100 (251 speakers, local, CC BY 4.0);
  - AMI / ICSI train single-speaker segments;
  - the extra English set;
  plus the VAD v3 room-tone and RIR augmentation, and prints drawn at 1-8 s.
- **Stage 2, real.** The 24 real meetings, with the overlap-frame loss weight at 2-3.
- **Exclusion.** oto is excluded from identity supervision (card terms).

**Speaker head v2 (item 2a):**
- Same taps. MHFA pooling: 8 heads, keys and values from separate mixes, ~0.8 M params.
- Distilled from TitaNet-L plus one Apache-licensed embedder (WeSpeaker ResNet34 or CAM++; the licence of the chosen
  checkpoint must be checked before caching). The relational loss is kept.

**Expected gain:**
- Overlap-frame miss down from 32-38 % toward the diarizer column's 13-30 %. That comparison is the only in-repo
  evidence that overlap frames are recoverable at 80 ms.
- The 1.5 s print penalty (5-7 F1) roughly halved, per ASE's +1.6 to +4 F1 at 0.5-1.5 s.
- tWER can gain at most 5.8 points on ICSI and 14.2 on AMI (the oracle-mask gap). A realistic share is 2-4 points on
  ICSI (estimate).

**Cost.** ~0.1-0.2 ms per chunk (32-slot attention on 128-d), no extra encoder. Training: a block 1-6 feature cache
of the simulated set (tens of GB on the SSD) and CPU / MPS head training in minutes.

**Risk:**
- The simulation-to-real gap on a frozen ASR encoder that was never trained on mixtures.
- The memory can drift to a second voice. Keep the enrolled anchor, as the served fix does.
- The pass-2 speaker kernels read P(user) from this track, so the turn head's inputs shift. Re-dump
  `eot_latency.py` after any TS-VAD change, as the print fix did.

**Addresses:** TS-VAD F1 0.88 / 0.74, overlap misses, the short print, and tWER 37 vs 31 %.

---

## 3. Spoken language ID

### What we measured
- `lid_distill` (archive/LID.md fix pass): **91.2 % at 2 s** and 97.7 % full utterance on FLEURS-17. AmberNet:
  95.1 / 99.5 %. SpeechBrain ECAPA: 94.6 / 99.3 %.
- Half of the remaining 2 s errors are uk → ru and de → he / ru / pl.
- EdAcc: 65.8 % of accented-English segments called English.
- What we already tried:
  - distillation: +12 points;
  - 13x more FLEURS data: +3-4 points;
  - a causal GRU: no help;
  - hidden 512: +2.3 points.

### What the best systems do

**AmberNet, renamed TitaNet-LID in v2 of the paper** (https://arxiv.org/pdf/2210.15781, checked;
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet):
- ContextNet-style depthwise-separable convolutions with SE, then mean+std pooling.
- 29 M params, trained on VoxLingua107.
- VoxLingua107 error: 7.5 % (0-5 s) and 5.2 % (5-20 s). SpeechBrain ECAPA: 11.9 / 6.7 %.
- Error falls from about 11-12 % at 3 s segments to about 5 % at 9-11 s (read off Fig. 3).
- Its top FLEURS confusions are closely related pairs: Serbian → Bosnian, Urdu → Hindi, Croatian → Serbian.

**MMS-LID** (https://arxiv.org/html/2305.13516, Table 7): 97.2-97.5 % on FLEURS-102. Whisper: 64.5 % zero-shot on
FLEURS-102 (https://arxiv.org/pdf/2212.04356, Table 5).

**Google, per-frame LID inside a cascaded-encoder RNN-T** (https://arxiv.org/pdf/2209.06058, checked):
- Running mean+std pooling, 2 × 512 FC, ~0.7-1 M params. That is our design.
- 88.1 % at frame 0 → 97.6 % at frame 30 → 98.8 % full utterance, over 9 locales.
- Pooling vs none: 92.9 vs 89.0 %.
- Mid-encoder b4 + b5: 91.5 % vs b4 alone 89.8 %.
- It is trained jointly with a multilingual ASR, not on a frozen English encoder.

**NVIDIA "Accidental Learners"** (https://arxiv.org/pdf/2211.05103):
- Language information peaks in layers ~7-11 of 18 and collapses at the top.
- An **English-only** pretrained encoder has 35-60 % error on seen languages, against ~10 % for the multilingual one.

**Frozen multilingual FastConformer** (https://arxiv.org/html/2606.09317, *[abstract]*): 94.2 % on 42-language Indic
FLEURS. The encoder is multilingual.

**Text fusion** (Kukk & Alumäe, https://arxiv.org/pdf/2203.16972, Table 2):
- An XLS-R acoustic LID calls L2-ARCTIC accented English "English" 74.6 % of the time.
- A char-4-gram classifier on **English ASR output**: 79.8 %.
- Equal-weight fusion: **90.5 %** L2-ARCTIC, 88.2 % CSLU-FAE, VoxLingua107 unchanged (95.3 %).

**Short-utterance errors** (Samsung, https://www.isca-archive.org/interspeech_2025/dey25_interspeech.pdf): 36.94 % of
errors on 2 s inputs come from out-of-scope content (non-speech, named entities, fillers, overlap).

### What transfers
- Our head already matches the best streaming design (Google's). The remaining gap is the **encoder**: it is
  English-only (NVIDIA's result), and head capacity will not fix that.
- Three cheap levers remain:
  - **Transcript features from the RNNT we already run.** This is the best-evidenced fix for accented English, the
    one sub-task where we trail on a product-relevant axis.
  - **More speaker diversity** through VoxLingua107 windows with teacher labels. The CE heads memorised FLEURS
    speakers.
  - **Masking non-speech and fillers** in the pooling, per the Samsung finding. The VAD-gated pooling already does
    part of this.

### Proposal
- **Keep `LanguageHead`** (causal attentive-stats pooling), with taps widened to a learned mix of blocks 6-13.
- **Add a 16-d "RNNT evidence" vector per frame, pooled with the rest:**
  - blank posterior;
  - the top-token log-prob EMA;
  - tokens per second;
  - the log-likelihood of the running partial under a small English char-4-gram model, trained on LibriSpeech text.
- **Teachers.** AmberNet + SpeechBrain ECAPA (Apache-2.0) logits averaged, T = 2, α = 0.8.
- **Data.** Add VoxLingua107 windows for the 17 languages (CC BY 4.0) to FLEURS train + extra English.
- **Size.** ~1.0 M params. State: running sums, as today.
- **Expected gain:**
  - FLEURS 2 s: +1-2 points (estimate from the diversity and ensemble trend in the fix pass). Reaching AmberNet's
    95.1 % is not expected on an English-only encoder.
  - EdAcc English: toward 80 %+. Evidence: Kukk & Alumäe's 35-63 % relative error reduction from text fusion.
- **Cost.** ~0.25 ms per chunk. The char-n-gram scoring of the partial is microseconds.
- **Risk.** The RNNT features are English-biased by construction. They help "is it English" more than they help
  uk vs ru. Check per language that non-English pairs do not regress.
- **Addresses:** 91.2 vs 95.1 % at 2 s (partly) and EdAcc 65.8 %.

---

## 4. End of turn / turn-taking

### What we measured
- **Assistant-directed clips** (EOT_ASSISTANT.md): accuracy 41.1 % (fast preset 42.9 %), vs Pipecat smart-turn +
  Silero 69.7 %, LiveKit 72.7 % and the smart-turn classifier alone 97.0 %. 172 / 175 complete and 223 / 224
  incomplete clips end on the 640 ms fallback. The head path almost never fires.
- **Calls:** EOT p50 956 ms at 20.2 % FI and 7.3 % missed. 65 of 101 answered ends go to the fallback.
- **End-vs-pause frame AUC:** 0.59 (served) and 0.66 (turn v4 heads) on calls (TURN_V4.md). 45 % of two-party turns
  contain a ≥ 160 ms pause (EOT_LATENCY.md).
- **TurnBench dev:** F1 0.892 but median delay 1316 ms, vs VAP 463 ms (METRICS.md).
- **Internal evidence already in the repo:**
  - **Completeness head** on the frozen encoder (COMPLETENESS.md §4): frame AUC **0.995** vs smart-turn v3.2's 0.994
    on the 399 held-out clips, with 124 K params and 3.5 k training clips. Accuracy at 0.5 is only 0.87-0.94
    (calibration). It did not help meeting open ends.
  - **Multi-horizon future-activity bins with a predictive trigger** (archive/DYADIC.md §8): on TurnBench held-out
    halves, 0.804 / 0.091 / **961 ms** P50, against its own reactive head at 0.720 / 0.064 / 1336 ms. OR-ed with
    Silero: 0.862 / 0.092 / 1137 ms. It fires after silence starts, never before.
  - **A pass-1 VAP-style head replacing the served one was killed** (FINAL_REPORT.md §5.8): +9.9 / +13.5 miss points
    on AMI / ICSI, because it fires inside long meeting pauses.
  - **In flight:** turn v5 (`scripts/research/turn_v5.py`: a segment classifier over blocks 4/8/12/17 + RNNT text +
    prosody + LM log-probs), the `EnergyGate`, and the `--turn-model smartturn` bridge.

### What the best systems do

**Pipecat smart-turn:**
- **v2** (https://huggingface.co/pipecat-ai/smart-turn-v2/blob/main/README.md;
  https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/): wav2vec2 + linear
  head.
  - How incomplete examples are made: an LLM cuts a sentence "near the end, then add[s] the filler word and ellipses",
    and TTS renders it with non-final intonation.
  - 50-80 % of generated sentences were discarded by LLM cleaning.
- **v3** (https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/;
  https://huggingface.co/pipecat-ai/smart-turn-v3): **Whisper-tiny encoder + linear classifier**, ~8 M params, int8
  QAT, 8 MB ONNX, 12.6 ms on a c7a.2xlarge. Accuracy across 23 languages: 81.27-97.10 %.
- **v3.1** (https://www.daily.co/blog/improved-accuracy-in-smart-turn-v3-1/): added human English and Spanish audio.
  **English 88.3 → 94.7 %** (8 MB model) / 95.6 % (32 MB). The blog says earlier versions were "heavily reliant on
  synthetic data".
- **v3.2** (https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/): cafe and
  office noise added; short utterances misclassified "40 % less often". Up to 8 s of input, left-padded.
- **v3.2 data** (https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train): 270,946 clips in 23
  languages, **CC BY 4.0** (card `license: cc-by-4.0`; licence commit 2026-09-29, checked via the HF API).

**LiveKit:**
- **Text turn detector** (https://huggingface.co/livekit/turn-detector;
  https://docs.livekit.io/agents/logic/turns/turn-detector/):
  - Qwen2.5-0.5B, distilled from a fine-tuned Qwen2.5-7B.
  - Input: the last 6 turns, ≤ 128 tokens.
  - TPR 99.3-99.4 %, TNR 85.1-96.3 % across languages, 50-160 ms per turn.
  - LiveKit Model License. Now deprecated.
- **Audio turn detector v1 / v1-mini** (https://livekit.com/blog/solving-end-of-turn-detection):
  - Two branches, fused: audio encoder → adapter → LLM, with no transcript; and an encoder → recurrent layer for
    timing and prosody.
  - False-cutoff rate at 300 ms latency: 9.9 %, vs Deepgram Flux 12.9 % and ultraVAD 27.7 %.
  - Mean latency: 543 ms at a 5 % false-cutoff target, 295 ms at 10 %.
  - Params not disclosed.

**VAP:**
- **Original** (Ekstedt & Skantze, https://arxiv.org/abs/2205.09812):
  - Frozen CPC + causal transformer (4 layers, 256-d).
  - Target: 2 s future split into 4 bins per speaker (200 / 400 / 600 / 800 ms), 2^8 = 256 classes, cross-entropy.
  - Shift / hold is read zero-shot in mutual silence.
  - Weighted F1 on Switchboard: shift / hold 0.899, shift prediction 0.733, backchannel prediction 0.723.
- **Multilingual VAP** (https://arxiv.org/html/2403.06487v1): shift / hold balanced accuracy 77.16 % English,
  84.60 % Mandarin, 76.54 % Japanese.
- **Real-time VAP** (Inoue et al., https://sap.ist.i.kyoto-u.ac.jp/lab/bib/intl/INO-IWSDS24a.pdf): a 1 s transformer
  context gives 76.16 % balanced accuracy vs 74.20 % at 20 s, and 14.61 ms vs 273.84 ms per frame on CPU. The CPC
  GRU carries the long context.
- **Code:** MIT with a checkpoint (https://github.com/ErikEkstedt/VoiceActivityProjection).
- **Prosody study** (https://aclanthology.org/2022.sigdial-1.51.pdf): removing intensity and phonetic detail hurts
  most; flattening F0 alone hurts least.

**TurnBench** (https://arxiv.org/html/2608.25218v2; https://turnbench.sesame.com/):
- 154 dialogues, triple-labelled. Humans start their reply a median 151 ms before the turn ends.
- Test leaderboard (recall / FPR / latency):

  | system | recall | FPR | latency |
  |---|---|---|---|
  | Vox Maru v1 | 0.960 | 0.072 | 548 ms |
  | VAP | 0.845 | 0.055 | 368 ms |
  | Kyutai semantic VAD | 0.773 | 0.059 | 1007 ms |
  | smart-turn v3 | 0.752 | 0.047 | 1017 ms |
  | Moshi | 0.233 | 0.044 | 702 ms |

- Interruption task: VAP 0.945 recall at 0.107 FPR.

**Fused ASR + EOT:**
- **NVIDIA Parakeet-Realtime-EOU 120M** (https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1):
  - The same 17-layer cache-aware FastConformer at [70, 1] as ours, emitting an `<EOU>` token.
  - EOU delay on TTS audio with silence appended: P50 160 ms, P90 280 ms.
  - The card gives no precision or recall and does not describe how the token was trained.
  - On TurnBench dev we measured it at 0.942 / 0.763 / 0.843 (P / R / F1; METRICS.md).
- **Deepgram Flux** (https://deepgram.com/learn/introducing-flux-conversational-speech-recognition;
  https://developers.deepgram.com/docs/flux/configuration): fused ASR + turn model. EagerEndOfTurn fires 150-250 ms
  earlier, at the cost of 50-70 % more LLM calls.
- **AssemblyAI** (https://www.assemblyai.com/docs/streaming/universal-streaming/turn-detection): a semantic
  confidence threshold (default 0.4) plus min / max silence. Presets: aggressive 160 / 400 ms, balanced
  400 / 1280 ms. No accuracy numbers published.
- **Joint endpointing in RNN-T** (Google, https://arxiv.org/pdf/2211.00786): −120 ms median, WER unchanged.

**Other audio + text turn models:**
- **Krisp v1** (https://krisp.ai/blog/turn-taking-for-voice-ai/): audio-only, 6.1 M params, 100 ms frames, ~2000 h.
  Balanced accuracy 0.82 vs smart-turn v2 0.78.
- **Vogent-Turn-80M** (https://blog.vogent.ai/posts/voturn-80m-state-of-the-art-turn-detection-for-voice-agents):
  Whisper encoder tokens prepended to the transcript and fed to 12 layers of SmolLM2. 94.1 % accuracy.
- **Easy Turn** (https://arxiv.org/html/2509.23938v1): Whisper-medium + Qwen2.5-0.5B, four states (complete /
  incomplete / backchannel / wait), 96-98 % on its own test set.
- **FD-VAD** (https://arxiv.org/html/2609.35791, *[abstract]*): trained on the English part of smart-turn v3.1
  (63,386 clips). Zero-shot on TurnBench dev: recall 0.853 at FP ≤ 0.10, median 1019 ms.
- **Causal-label paper** (https://arxiv.org/abs/2609.04225, *[abstract]*): offline labels leak the future. Appending
  1 s of silence moved EOT recall from 0.10 to 1.00, so labels must be computable from past input only.
- Unverified, not used: TEN Turn Detection accuracies and Krisp v2 numbers.

### What transfers
1. **The segment-level completeness decision** is what separates smart-turn's 97 % from our 41 %.
   - Our encoder already ranks it as well as smart-turn (AUC 0.995).
   - Missing: a decision at the right moment (the quiet trigger, which the VAD v3 / EnergyGate work enables),
     calibration, and training data at smart-turn's scale. v3.1 shows human data is the lever: +6.4 points in
     English.
2. **Text semantics.** LiveKit, Vogent, Easy Turn and UltraVAD all put a language model on the transcript. We have a
   streaming RNNT transcript for free. A text-LM teacher (turn v5's Qwen2.5-7B log P(sentence-final punctuation))
   gives soft targets for every pause without new labels, as LiveKit's 7B → 0.5B distillation did.
3. **Future-activity projection** is the one objective with in-repo evidence of shorter two-party latency
   (961 vs 1336 ms) and evidence of harm on meetings. Use it as an auxiliary output and a mode-gated trigger, not as
   the only EOT signal.
4. **Prosody.** The VAP perturbation study ranks intensity above F0. Our 12-value causal prosody extractor
   (`audioforge/heads/prosody.py`) already covers the level, F0 and final lengthening that turn v5 feeds.
5. **Causal labels.** Label EOT from the offset onward, and never append silence that is absent at inference.

### Proposal: turn head v5+ (a) + (b) + (c)

**(a) Segment classifier at the quiet trigger.** This is the turn v5 main line, with the data change below.
- **Inputs:**
  - pass-1 blocks 8 / 12 / 17, one learned mix, over the last ≤ 8 s (≤ 100 frames);
  - the 12-d prosody frames;
  - the RNNT partial as a text feature: a bag of the last 16 word-piece embeddings plus the LM teacher's
    P(complete | words) as an input at train time only.
- **Model.** 2-head attention pooling + GRU-final state + MLP, ~0.3 M params.
- **Fires** once per silence run at the quiet trigger (VAD v3 / EnergyGate), and again at +160 ms.
- **Training data:**
  - **smart-turn-data-v3.2-train, English subset** (CC BY 4.0), excluding any id in human_5 test;
  - human_5_all train;
  - turn_v5's own energy-dip cuts of oto / AMI / ICSI turns;
  - oto turn ends and ≥ 400 ms pauses as real two-party positives and negatives.
- **Loss.** Clip BCE + the LM teacher KL on pauses (T = 2).
- **Calibration.** Temperature fitted on a held-out split of the training clips, never on the eval sets.

**(b) Multi-horizon future-activity bins.**
- Four disjoint bins for the user, (t, t+240], (+240, +400], (+400, +640], (+640, +1040] ms, as DYADIC (c).
- Plus two bins for "other speaker / agent", so P(other takes the floor) is explicit, as in VAP's joint code.
- An auxiliary output of the per-frame turn GRU.
- The predictive trigger (bins 1-2 below threshold) is enabled only in two-party user-channel mode.

**(c)** Is the LM teacher inside (a).

**Pass 2.** Keep the speaker-conditioned pass 2 for meetings; it is what AMI / ICSI need (FINAL_REPORT.md §5.8). In
assistant mode, (a) reads pass 1 only.

**Size and state.** ~0.3 M (a) + ~10 K (b). State: an 8 s ring buffer of 100 × 256 mixed frames (fp16, 50 KB) plus
the GRU hidden state.

**Expected gain:**
- Assistant-directed accuracy from 41 % toward ≥ 90 %. Evidence:
  - the encoder-level AUC of 0.995 (COMPLETENESS.md);
  - smart-turn's own 97 % on the same clips;
  - the 270 k-clip training set, against the 3.5 k clips that reached 0.995.
- The assistant EOT p50 falls from ~1.2 s toward ~0.3-0.5 s: the trigger's quiet time plus one classifier call,
  once room tone reads as silence (item 1).
- Two-party calls: −200 to −375 ms p50 at matched FP from (b). Evidence: DYADIC 961 vs 1336 ms; the oto-fitted
  point reached 755 ms, but at a higher FP.
- Meetings: none expected. The goal there is no regression.

**Cost:**
- (a) is ~1 ms per call, once or twice per silence run. Smart-turn's own ONNX call is 12.6-65 ms, which we avoid by
  reusing the encoder.
- (b) is < 0.1 ms per frame.
- Training: download 41 GB to the SSD, one pass-1 re-encode of the English subset (the largest cache job in this
  plan; cap it at ~50 k clips first), and head training in minutes.

**Risk:**
- Smart-turn clips are single-utterance, assistant-style and partly synthetic. Calls and AMI are human-human with
  long pauses. The COMPLETENESS.md fusion found no meeting gain, so gate on calls / AMI no-regression.
- The v3.2 test split overlaps human_5 ids. Deduplicate by id before any training.
- The dataset licence was added 2026-09-29. Re-check it at download time and record it in `LICENSE.md`.

**Addresses:** 41 % vs 97 % assistant accuracy, the missing pause-vs-end separation, and the TurnBench 1316 vs 463 ms
delay (partly, via (b)).

---

## 5. Heads we lack that voice agents ship

### 5a. Barge-in / backchannel intent

**What shipping systems do:**
- **LiveKit adaptive interruption handling** (https://livekit.com/blog/adaptive-interruption-handling;
  https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/):
  - Audio encoder + CNN on overlapping speech, "hundreds of hours" of human-human speech with noise augmentation.
  - **86 % precision and 100 % recall at 500 ms of overlap. Rejects 51 % of VAD barge-ins.**
  - Median 216 ms of audio needed; inference ≤ 30 ms. Cloud only.
  - False interruptions (VAD fires but no transcript) resume playback.
- **Pipecat:** `MinWordsInterruptionStrategy` is deprecated in favour of `MinWordsUserTurnStartStrategy`, a
  word-count rule
  (https://reference-server.pipecat.ai/en/latest/api/pipecat.audio.interruptions.min_words_interruption_strategy.html).
- **OpenAI Realtime** `semantic_vad` eagerness (https://developers.openai.com/api/docs/guides/realtime-vad): no
  timings published.
- **Backchannel prediction on VAP** (Inoue et al., NAACL 2025, https://arxiv.org/html/2410.15929): frame F1 42.85 %
  for timing. **Easy Turn** has a "backchannel" state (91 %, its own test set).
- **TurnBench's interruption task** (eval only for us): VAP 0.945 recall / 0.107 FPR; smart-turn v3 0.107 / 0.093.

**Proposal:**
- **Task.** A classifier that runs only while the agent is speaking and the user's TS-VAD P(user) rises. Output
  classes: {backchannel, interruption, noise / echo}, emitted at 240, 400 and 560 ms after onset.
- **Inputs.** Pass-1 block-4 and block-12 mix, TS-VAD P(user), the energy above the floor, and the first RNNT tokens
  as an 8-token bag. "mhm / yeah / right" are strong lexical cues.
- **Model.** Attention pool + MLP, ~0.15 M params. State: the onset buffer of ≤ 7 frames.
- **Data:**
  - oto two-channel calls: overlaps where the other party is speaking. Label short (< 1 s) overlaps that do not take
    the floor as backchannels, and overlaps after which the floor changes as interruptions. These are the VAP-style
    self-supervised labels.
  - Behavior-SD (CC BY 4.0, synthetic, per-speaker backchannel and interruption annotations; only the test split is
    local, 1.4 GB).
  - The VAD v3 noise set for the noise / echo class.
- **Eval:** false-barge-in rate and interruption latency on held-out oto, plus the TurnBench interruption task
  (evaluation only).
- **Expected gain.** Not measurable until the eval exists. The only external reference is LiveKit's 51 % rejection
  of VAD barge-ins at 100 % recall.
- **Cost.** < 0.1 ms per chunk, only during agent speech.
- **Risk.** oto labels come from Silero turns (DYADIC caveat), and Behavior-SD is TTS.

### 5b. Overlap / second-voice

TS-VAD's two sigmoids already represent overlap (Sortformer and TS-VAD use the same per-speaker sigmoid form). No new
head is proposed: expose `P(target)·P(other)` as a frame signal, and improve it through 2b's overlap-weighted loss.

### Out of scope

Emotion (per the brief). Device-directed speech detection has published gains from text + audio fusion (EER 7.45-7.95 %,
https://arxiv.org/html/2403.14438v1), but it needs agent-context data we do not have.

---

## Unverified or not found (not used above)

- A paper that studies −60 to −40 dBFS room-tone augmentation specifically: not found. Braun & Tashev's
  N(−28, 10) dBFS is the only published level distribution we found.
- Silero's architecture: third-party only.
- TEN VAD's transition latency: a figure, no number.
- COIN-AT-PVAD, SVVAD and AS-pVAD numbers: not opened.
- TS-VAD numbers on AMI / ICSI with target-speaker frame F1: none found. Our A.1 table has no external comparator.
- smart-turn v3.2's overall "92.9 %" (search snippet only); Krisp v2; TEN Turn Detection; UltraVAD details; the
  PersonaPlex Full-Duplex-Bench numbers.
- Parakeet-Realtime-EOU's `<EOB>` backchannel token: third-party pages only.
- NSD-MS2S and Seq2Seq-TSVAD numbers: search snippets only.
