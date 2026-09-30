# Outside components for a dyadic, speaker-aware turn model on our substrate

*2026-09-26. Research note only: no model was run and no file other than this one was written. Sources were fetched
on this date; every claim carries a URL. "(est.)" marks our own estimate with its arithmetic. Local anchors:
BRIEF.md, BASELINES.md (turn section), EOT_BENCH_V2.md §1/§7/§8, DATA_PLAN.md, CONTAMINATION.md, SPK_HEAD.md.*

## 0. What changed after this pass (read this first)

1. **A public dyadic benchmark with exactly our protocol exists and we are not on it.** TurnBench (Sesame + Mundo AI +
   oto, arXiv 2608.25218, SLT 2026) scores end-of-turn and interruption events on **dual-channel, hand-labelled human
   conversations** as *recall at a capped false-positive rate* (FPR ≤ 0.10 on dev, ≤ 0.15 on test) with a
   [−0.25 s, +3.0 s] match window and a strict causality rule. That is eot-bench v2's "miss at ≤ 5 % false cutoffs,
   6 s horizon" with different constants. Best published baseline is **VAP: 0.845 EOT recall at 0.055 FPR, 368 ms
   median latency**; smart-turn v3 0.752/0.047 at 1017 ms; Kyutai's STT-side VAD 0.773/0.059 at 1007 ms; Moshi 0.233.
   Leaderboard top: Vox Maru v1 0.960 recall / 0.072 FPR / 548 ms. Humans lead the turn end by a median 151 ms.
   https://turnbench.sesame.com/ , https://arxiv.org/html/2608.25218 , https://github.com/SesameAILabs/turnbench
2. **Real two-track human dyadic audio with a permissive licence exists: otoSpeech-full-duplex-280h + processed-141h,
   CC BY 4.0, 48.8 + 19.4 GB, 48 kHz, ch0 = speaker A, ch1 = speaker B, contact-form gate.** No turn labels (derive
   them per channel). Its sibling, the 104 h hand-labelled TurnBench train set, is non-commercial and 290 GB.
   https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h ,
   https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-turn-104h
3. **The original VAP (2022) is our architecture:** mono mix of both speakers + per-speaker activity features → causal
   transformer → 256-way discrete future-activity code. The stereo two-tower came later. Adopting the 256-way head on
   our GRU is a faithful reproduction, not a variant (§2). https://arxiv.org/abs/2205.09812
4. **Two 2026 papers already claim "speaker-aware, frame-synchronous EOT on a speech encoder":** Meta's hierarchical
   EOT with online primary-speaker tracking (no code, proprietary 350 h; https://arxiv.org/html/2603.13379) and X2-Turn
   (fully fine-tuned Voxtral-Realtime, 80 ms turn-state head; https://arxiv.org/html/2608.10878). Neither is on a frozen
   ASR encoder, neither is evaluated at a matched false-cutoff rate on a public dyadic set, and neither conditions on an
   enrollment. That is the space left for us (§6).
5. **For the speaker head, the literature says the head is the bottleneck, not the teacher.** Systems that reach
   < 1 % VoxCeleb EER on a *frozen* ASR Conformer/Whisper encoder use multi-layer taps + per-layer adaptors + 1–2
   trainable Conformer/TDNN layers + attentive pooling (~5 M params), and cosine-only distillation adds almost nothing on
   top of a classification loss (§3). Our 32 % within-meeting EER with a pooling layer over a frozen layer mix is the
   expected outcome of that design, which sharpens SPK_HEAD.md's experiment.
6. **For target-speaker VAD, FiLM on the encoder output beats input concatenation in every head-to-head; enrollment
   accuracy falls off a cliff below 1.5 s and saturates by 3–5 s** (§4). Our 1.5 s enrollment window (EOT_BENCH_V2 §8)
   sits on the cliff edge.
7. **"Learned repair" has no direct precedent, but a 2026 KV-cache result predicts that bounded replay of cached frozen
   encoder frames wins** (§5). The paper-worthy artefact is the contamination measurement plus a revision-event
   interface, unless learned repair matches replay at O(1) cost.

## 1. Dyadic corpora

Scorecard rows moved: "Turn end, floor-open ends" (a two-party corpus has no bystander, so enrollment errors vanish and
the agent-side signal is exact) and "Product dead air". Nothing here moves diarization DER.

### 1.1 Table

| corpus | hours / convs / spk | channels | labels | licence | obtain (gating, size) | 2024–26 turn use |
|---|---|---|---|---|---|---|
| **otoSpeech-full-duplex-280h** | 280 h, "1K–10K" sessions, remote two-party | stereo FLAC 48 kHz, **ch0 = A, ch1 = B** | none; session ids, speaker profiles, surveys, redaction spans | **CC BY 4.0**; prohibits attempts to identify / de-anonymise speakers | HF contact form, 48.8 GB. https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h | anyreach DualTurn derives 80 ms EOT/HOLD/BOT/BC labels from it (codes only, no audio, https://huggingface.co/datasets/anyreach-ai/dualturn-otospeech-turn-taking , https://arxiv.org/abs/2603.08216) |
| otoSpeech-processed-141h | 141 h | 44.1 kHz WebDataset, two channels | none | CC BY 4.0 | HF form, 19.4 GB. https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-processed-141h | – |
| **otoSpeech-turn-104h** (TurnBench train) | 104.9 h, 420 convs, speaker-disjoint from TurnBench | mono WAV per speaker + stereo, 48 kHz float32 | **17 event classes** (normal turn, 3 backchannel types, 4 interruption types, overlap, filler, laughter, strong floor hold…) as per-speaker SRT with transcripts | **oto Speech License v1.0: non-commercial research**; prohibits voice-identity uses incl. speaker identification | manual gate, **~290 GB** | TurnBench baselines (WavLM, ESPnet) fine-tuned on it |
| **TurnBench dev** | 7.3 h, 38 convs | dual mono 48 kHz 24-bit FLAC | 3 annotator tracks per speaker: EOT, interruption, backchannel, laughter, ms timestamps, transcripts; κ 0.78 | Dataset Public License v1.0: **non-commercial**, attribution, no voice cloning | gated form, 4.22 GB. https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev | it *is* the benchmark |
| TurnBench test | 22.9 h, 116 convs | same | withheld; score by e-mail | same | 13.3 GB | leaderboard |
| **Behavior-SD** | 2,164 h, 108 K dialogues, 52 CosyVoice voices | two-channel, one speaker per channel | per-utterance start/end, backchannels, interruptions, fillers; timing priors published (gap N(0.4, 0.2) s, BC delay N(0.2, 0.02) s, interrupt overlap N(0.45, 0.05) s) | **CC BY 4.0**, code MIT, ungated | HF, 188 GB (val 1.6 GB, test 1.5 GB). https://huggingface.co/datasets/yhytoto12/behavior-sd , https://aclanthology.org/2025.naacl-long.484.pdf | full-duplex SLM training |
| **kyutai/DailyTalkContiguous** | ~20 h, 2,541 dialogues, **2 actors** | stereo, one speaker per channel, word timestamps | text, speaker, turn index, emotion/act | **CC BY-SA 4.0** | HF, ungated, 13.8 GB. https://huggingface.co/datasets/kyutai/DailyTalkContiguous | none; acted, no overlap |
| CANDOR | > 850 h, 1,656 calls, 1,456 spk | stereo MP3, one participant per channel + per-user mp4 | AWS Transcribe JSON; 3 turn segmentations incl. **Backbiter** (backchannel-separated) | terms shown only inside the request form; article CC BY-NC; derivative claims conflict. **Assume non-commercial** | BetterUp request form, human approval, links live 1 week, > 1 TB. https://betterup-data-requests.herokuapp.com/ , https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445/ | de-facto VAP training corpus: https://arxiv.org/abs/2505.21043 , https://arxiv.org/abs/2601.13835 |
| SpokenWOZ | 249 h, 5,700 dialogues, 250 spk, human role-play user/agent | **8 kHz two-track user / agent** | word-level times (ASR, 6.1 % WER), backchannel act, DST | CC BY-NC 4.0 | https://spokenwoz.github.io/ (HF mirror 10.3 GB) | none |
| MultiDialog | 340 h, 8,733 dialogues, 12 actors | one 16 kHz WAV per utterance per speaker; **no inter-turn timing** | transcript, emotion | HF `license: cc` with no variant (unverified) | HF, 32.9 GB, ungated. https://huggingface.co/datasets/IVLLab/MultiDialog | none |
| Switchboard-1 R2 | 260 h, 2,400 calls | 2-ch µ-law 8 kHz | free MS-State word alignments (https://isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz), SwDA backchannel acts (CC BY-NC-SA), Ekstedt backchannel CSV | LDC non-member: non-commercial research; fee shown only after catalog login (unverified ballpark $2.5k) | https://catalog.ldc.upenn.edu/LDC97S62 | VAP, Talking Turns, ESPnet TT-pred |
| Fisher P1+P2 | 1,960 h, 11,699 calls | 2-ch 8 kHz | utterance-level transcripts, no word times | LDC, same terms (ballpark $2k per speech part, unverified) | https://catalog.ldc.upenn.edu/LDC2004S13 | Moshi, X2-Turn (249 h) |
| CallHome AE | 56 h, 120 calls | 2-ch 8 kHz | turn-level transcripts, 10 min per call | LDC | https://catalog.ldc.upenn.edu/LDC97S42 | SEAL/DiarizationLM evals |
| TalkBank / CABank (CallFriend, CallHome, SBCSAE) | small | SBCSAE is a stereo *room* recording, not per speaker; CallFriend layout on TalkBank unverified | CHAT with ms bullets | **CC BY-NC-SA; rules ban use in commercial ML products**; media needs login (verified: URL returns auth page) | https://talkbank.org/0share/rules.html | none |
| Spotify Podcast Dataset | – | – | – | **withdrawn Dec 2023**, site DNS dead | https://trecpodcasts.github.io/ | drop |
| Pipecat smart-turn-data-v3.2 | 271 K train / 31.5 K test clips, 23 languages | **mono, single user utterance** ≤ 32 s | `endpoint_bool`, `midfiller`, `endfiller`, `synthetic`, source | no licence field on the card (code/model BSD-2) | HF, 41.4 GB. https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train | smart-turn v3 |
| LiveKit / Sesame CSM / Moshi / Hume / Ultravox training data | not released (Moshi: Fisher + > 20 k h synthetic) | – | – | – | https://arxiv.org/html/2410.00037 | – |

Verdict from the table: **nothing is simultaneously real, per-speaker, human-labelled and permissive; you get three of
four.** The workable combination is otoSpeech-280h/141h (real, per-speaker, permissive, unlabelled) for training with
derived labels, TurnBench dev (labelled, non-commercial) for evaluation only, Behavior-SD (synthetic, exact labels) for
the agent-track pretraining, DailyTalkContiguous for pipeline tests.

### 1.2 Integration (smallest)

- `audioforge/datasets/oto.py` (~250 LOC): read stereo FLAC, resample 48 → 16 kHz (`torchaudio`), yield per-channel
  waveforms + the mix; per-channel activity from Silero VAD (`baselines/turn.py` already wraps it) or our own VAD head
  on each channel; derive `spk_targets` (2 columns, exact by construction), `eot_targets` via the existing floor rules
  (`AMI.turn_examples` logic generalised to 2 speakers), backchannels as isolated ≤ 1 s speech (DualTurn's rule,
  https://arxiv.org/abs/2603.08216). Word timings from our own CTC teacher on each channel (`teachers.py`).
- CPU cost: Silero 0.08 ms per 32 ms chunk (BASELINES.md) → 280 h ≈ 2 CPU-h; CTC teacher at 63× on MPS ≈ 4.5 h for
  280 h; disk after 16 kHz mono WAV per channel ≈ 2 × 280 × 115 MB ≈ 64 GB (DATA_PLAN.md rate). Start with the 141 h set
  (≈ 32 GB) or a 50 h slice.
- Control: the same head trained on AMI/ICSI only, evaluated on TurnBench dev and on eot-bench v2.
- Kill: after 50 h of otoSpeech, TurnBench-dev EOT recall at FPR ≤ 0.10 does not beat the Silero-timeout baseline we
  can compute locally with the MIT scorer (`python -m turnbench.score`).
- Licence trap: otoSpeech bans *identifying speakers*. Within-session enrollment/following (a slot, not an identity) is
  defensible; training a speaker-verification head on it is not. Keep TitaNet distillation on AMI/ICSI/LibriSpeech.

### 1.3 Synthetic agent + human two-channel data

No public pipeline mixes a TTS agent with real human speech into two-channel EOT data; the nearest recipe is
Behavior-SD's schema and timing priors (https://github.com/yhytoto12/Behavior-SD). Agent voice on a Mac CPU:
**Kokoro-82M, Apache-2.0, 24 kHz, 54 voices, word-level `start_ts/end_ts` for English via `KPipeline.join_timestamps`**
(https://huggingface.co/hexgrad/Kokoro-82M , https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro/pipeline.py).
Piper is now GPL-3 (https://github.com/OHF-Voice/piper1-gpl); Kyutai pocket-tts is MIT, 100 M
(https://github.com/kyutai-labs/pocket-tts). Integration: `conversation.py` already has the four-move generator
(`TurnTakingConfig`); add an `agent_channel` that renders the "other" role with Kokoro into column 1 and keeps the human
side from LibriSpeech/otoSpeech single-speaker stretches (~150 LOC). The agent track is then a *known* input, which is
the product situation (§6.3).

## 2. Turn-taking objectives and codebases

### 2.1 VAP, exactly (from `vap/objective.py`)

Source: https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/main/vap/objective.py (repo MIT).

- Targets: for frame t, take the *strictly future* activity `va[t+1 : t+1+100]` at 50 Hz for both speakers; split into
  bins of `bin_times = [0.2, 0.4, 0.6, 0.8]` s (= [10, 20, 30, 40] frames, horizon 2.0 s); bit = (mean activity in the
  bin ≥ `threshold_ratio` 0.5). 2 speakers × 4 bins = 8 bits → **256 classes**; class index = Σ bit_i 2^i in the order
  [s0b0..s0b3, s1b0..s1b3].
- Loss: `F.cross_entropy` over the 256 classes, plus an auxiliary per-frame per-speaker VAD BCE (`loss_vad`).
- Derived scores: `p_all[b,t,s] = Σ_c probs[c] · (#active bins of s in bins from..to) / Σ_s'(...)`. `p_now` = bins 0–1
  (next 0–600 ms), `p_future` = bins 2–3 (600–2000 ms), `p_tot` = all. **These are relative next-speaker scores
  normalised across the two speakers, not absolute activity probabilities.**
- 2022 paper (https://arxiv.org/abs/2205.09812): input was **one mono mix** + per-speaker VA frame + a 5-bin VA-history
  vector on CPC features; Discrete vs Independent (per-bin BCE) vs Comparative: S/H bAcc .899 / .897 / .893; the
  discrete head wins only on the predictive tasks (S-pred .733 vs .718, BC-pred .723 vs .685). So our cumulative
  "active within h" BCE (`future_act_aux`, `heads/turn.py`) costs ≈ 1.5–4 bAcc on prediction tasks and nothing on S/H.
- Current repo: stereo `GPTStereo` (256-d, 4 layers, cross-attention), mono "padded with a silent channel"; checkpoint
  `VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt` trained on Switchboard/Fisher/CANDOR, **no checkpoint licence**.
  CPC encoder: `60k_epoch4-d0f474de.pt` from https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/ (CPC_audio MIT).
- Streaming VAP (Inoue, https://github.com/inokoj/VAP-Realtime): code MIT, **models "academic use only"**; 14.6 ms/frame
  on a Xeon at 1 s context (https://arxiv.org/abs/2401.04868). Multilingual VAP https://arxiv.org/abs/2403.06487.
  Noise-robust VAP for a robot sets the robot channel to zero (https://arxiv.org/abs/2503.06241). Mono multi-party VAP
  with role-relative projection: MuVAP https://arxiv.org/abs/2606.16731.
- Evaluation (`vap/events.py`, https://raw.githubusercontent.com/ErikEkstedt/VoiceActivityProjection/main/vap/events.py):
  shift/hold at mutual silences ≥ 0.25 s with 1.0 s single-speaker pre/post conditions, shift-prediction region 0.5 s
  before the offset, backchannel ≤ 1.0 s; **negatives sampled to equal counts, metric = balanced accuracy with a
  validation-tuned threshold.** No VAP paper reports miss at a matched false-cutoff rate; there is no per-turn FC budget
  and no latency deadline. eot-bench v2 and TurnBench are stricter on both counts.

**Mapping to our 80 ms grid.** `bin_times_to_frames` truncates: at 12.5 Hz, [2.5, 5, 7.5, 10] → [2, 5, 7, 10] frames
(1.92 s horizon). Use 80 ms-aligned bins summing to 25 frames, e.g. **[2, 5, 8, 10]** (160/400/640/800 ms), or compute
the bin ratios on the 10 ms word grid and subsample. Our h ∈ {6, 12, 25} frames ≈ p_now (7.5 frames) and p_tot; a clean
p_future needs a bins-2–3 head. X2-Turn also runs its turn head at 80 ms, so the grid has precedent.

**Integration.** `heads/turn.py`: replace/extend `self.fut` (currently `Linear(hidden, 2·len(horizons))`) with a
`Linear(hidden, 256)` + CE on the codebook index from `future_act_targets`' inputs (~80 LOC incl. the codebook and
p_now/p_future decode); the score fed to eot-bench is `1 − p_now[primary]` or a learned readout. CPU: +256 logits per
frame, negligible. Control: the same head with the current BCE aux and with no aux (TURN_ABLATION.md grid). Kill: no
paired improvement on eot-bench 6 s all/open with CI excluding 0; VAP's own paper predicts the discrete head only helps
prediction-type tasks, so a null here is the likely outcome and is worth exactly one run.

### 2.2 Other objectives (what decides "speak now")

| system | objective / signal | separate channels? | supervision | metric | open | licence | URL |
|---|---|---|---|---|---|---|---|
| TurnGPT / RC-TurnGPT | P(turn-shift token) in a text LM; RC conditions on the planned response | text | transcript turns | shift bAcc | yes | repo | https://arxiv.org/abs/2010.10874 , https://arxiv.org/abs/2305.02036 |
| Moshi | implicit: multi-stream next-token at 12.5 Hz, inner monologue; no turn labels | 2 token streams | Fisher + 20 k h synthetic | Full-Duplex-Bench | yes | MIT/Apache code, CC-BY-4.0 weights | https://arxiv.org/html/2410.00037 |
| Kyutai STT `stt-1b-en_fr` "semantic VAD" | 4 extra heads: "has there been a pause of ≥ 0.5 / 1.0 / 2.0 / 3.0 s"; Unmute fires on the 2.0 s head at p > 0.5 | mono user | undocumented | TurnBench 0.773 / 0.059 / 1007 ms | weights yes | CC-BY-4.0 | https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/main/scripts/stt_from_mic_rust_server.py |
| Freeze-Omni | 3-state CE on the LLM hidden state at each chunk end (listen / interrupt / end) | mono | multi-round QA data | none quantitative | yes | custom | https://arxiv.org/html/2411.00774 |
| Sesame CSM | none: TTS only | – | – | – | yes | Apache-2.0 | https://github.com/SesameAILabs/csm |
| hertz-dev | implicit duplex audio LM | 1–2 ch | undisclosed | – | yes | Apache-2.0 | https://github.com/Standard-Intelligence/hertz-dev |
| Talking Turns judge | 5-class next event {NA, BC, I, T, C} per 40 ms on a Whisper encoder | dual (Switchboard) | VAD-derived events | ROC-AUC 92 | promised | – | https://arxiv.org/html/2503.01174 |
| Full-Duplex-Bench | pause handling, backchannel, smooth turn, interruption (TOR, latency) | user stimuli | CANDOR + synthetic | – | yes | CC BY-NC | https://github.com/DanielLin94144/Full-Duplex-Bench |
| Parakeet-Realtime-EOU | `<EOU>` token in RNNT | mono | TTS audio + ASR text | p50 160 ms | ckpt | NVIDIA Open Model | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| **X2-Turn** (Aug 2026) | frame-synchronous dual head on Voxtral-Mini-4B-Realtime, **80 ms**, 5 states idle/noidle/incomplete/complete/backchannel, CE λ = 0.1 + ASR CE; LLM word-level labels | mono | EasyTurn 126 h zh + Fisher 249 h | per-class acc, latency vs word end | code | check repo | https://arxiv.org/html/2608.10878 , https://github.com/X-Square-Robot/X2-Turn |
| **Meta hierarchical EOT** (Mar 2026) | enrollment-free online primary-speaker tracking (peeling embeddings, momentum centroid, cos > 0.7) → 1.14 M-param EOT, 5 frame classes per speaker, horizon-weighted CE, 100 Hz | mono | 350 h proprietary | recall 87.7 % vs smart-turn 58.9 %, median 36 ms | no | – | https://arxiv.org/html/2603.13379 |
| Voice-Light (Sep 2026) | causal adapter sharing the ASR encoder; learned EOT reached **12.5 % recall vs 95.6 % for a Silero policy**, shipped as hybrid | mono | – | recall | CC-BY-4.0 | https://arxiv.org/abs/2609.20995 |
| Google joint endpointer / turn-taking RNNT | EOQ token; later pause / EOQ / backchannel events in the transducer | mono | transcripts + heuristics | EP50/90; 97 % R / 85 % P at 100 ms | no | – | https://arxiv.org/abs/2004.11544 , https://arxiv.org/abs/2208.13321 |
| LiveKit turn detector | binary EOT on text, Qwen2.5-0.5B distilled | text | call-center + synthetic | TPR > 99 %, TNR 85–96 % | weights | LiveKit Model License | https://huggingface.co/livekit/turn-detector |
| smart-turn v3 | Whisper-tiny + linear, binary complete/incomplete on ≤ 8 s | mono | v3.2 data | TurnBench 0.752 / 0.047 | yes | BSD-2 | https://github.com/pipecat-ai/smart-turn |
| "Less can be more" (Sep 2026) | light acoustic + prosodic EOT; adding text hurt | mono | – | F1 0.93, 7.8 % FA at 400 ms | – | – | https://arxiv.org/abs/2609.11066 |

Reading: (a) every open production system decides on *one* channel with a VAD-shaped target ("pause reached h",
"complete?", `<EOU>`); (b) the only systems that use the other party are VAP-family and the closed full-duplex LLMs;
(c) the two speaker-aware papers of 2026 both fine-tune the encoder or use a proprietary tracker; (d) Voice-Light is
the honest negative: a learned EOT on a shared ASR encoder lost to a Silero policy, which matches our floor-open result.
Kyutai's "pause ≥ h" heads are the cleanest published analogue of our duration features and worth copying as an
auxiliary target (4 extra sigmoids on the GRU state, ~20 LOC).

### 2.3 TurnBench as our second benchmark

- Scorer (MIT): EOT match window `[t − 0.25, min(t + 3.0, next event)]`; recall = TP/(TP+FN); **FPR = FP/(FP+TN) over
  labelled negative spans (mid-turn pauses, backchannels), one FP max per span**; latency p10/50/90 of t_pred − t;
  commit rule "one event per rising edge above θ with a 2 s refractory"; dev operating point = highest recall at
  FPR ≤ 0.10, frozen for test. https://raw.githubusercontent.com/SesameAILabs/turnbench/main/turnbench/README.md
- Causality: "a timestamp is the time by which all audio the decision depended on has been heard".
  https://raw.githubusercontent.com/SesameAILabs/turnbench/main/docs/SUBMISSION_FORMAT.md
- Differences from eot-bench v2: per-negative-span FPR (ours: per-turn FC, which is stricter for long turns); 3 s
  deadline (ours: 2 s and 6 s); both channels given (ours: mixed mono + diarizer); their negatives are hand-labelled
  pauses and backchannels (ours: word-gap pauses, AMI convention hides < 0.5 s gaps, DATA_PLAN.md cause 1).
- Integration: `scripts/bench_turnbench.py` (~200 LOC): load dev FLAC pairs, downmix or feed per-channel, run
  `serve.py`'s stack or the head directly, emit `predictions.json`, call the scorer. Dual-channel input is a privileged
  control; the like-for-like row is the *mixed mono* with our own tracker, which no leaderboard entry reports.
- Effort: 1 day once the 4.2 GB gate is opened. This is the cheapest external validation available to us.

## 3. Open speaker embedding models (teacher and companion)

Row moved: "Speaker EER within meeting" (32 % → target ≤ TitaNet-L + 3 pts), and indirectly "Turn end, floor-open"
through enrollment agreement (EOT_BENCH_V2 §8: given a correct identity, following ends on the right column in 97 %).

| model | params | dim | licence (code / weights) | download | Vox1-O EER | short-segment EER | CPU (2 s seg, 2 threads, est.) | framework-free |
|---|---|---|---|---|---|---|---|---|
| **TitaNet-L** (ours, ported) | 25 M | 192 | CC-BY-4.0 | https://huggingface.co/nvidia/speakerverification_en_titanet_large | 0.66 | AMI within-meeting 8.2 % (BASELINES.md) | **45 ms measured** | yes (`nemo_import.import_titanet`) |
| **TitaNet-S** | 6.4 M | 192 | NeMo toolkit (Apache-2.0) per NGC; no official HF repo | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small (35.8 MB) | 1.08 | – | ~10–15 ms (4× fewer params, same blocks) | same importer, untested |
| **WeSpeaker CAM++(-LM)** | 7.2 M, 1.15 GFLOPs | 512 | Apache-2.0 code; weights CC-BY-4.0 per docs (HF tags say apache-2.0; docs win) | https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM , ONNX in repo | 0.66 | not published for this ckpt | **~10–12 ms** | ONNX |
| WeSpeaker ECAPA-512-LM | 6.2 M, 1.04 GFLOPs | 192 | same | https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM | 0.78 | ECAPA: 2 s 1.95 %, 1 s ≈ 12.8 % (https://arxiv.org/html/2406.02167 , https://arxiv.org/html/2506.14226) | ~10 ms | ONNX |
| WeSpeaker ResNet34-LM (= pyannote 3.1 embedding) | 6.6 M, 4.55 GFLOPs | 256 | CC-BY-4.0 | https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM | 0.72 | 2 s 2.12 %, 1 s 4.33 % (https://arxiv.org/html/2601.13999) | ~30–45 ms | plain PyTorch / ONNX |
| WeSpeaker ResNet293-LM | 28.6 M, 28 GFLOPs | 256 | CC-BY-4.0 | https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM | **0.43** | – | ~250 ms → teacher only | ONNX |
| WeSpeaker SimAM-ResNet100 (VoxBlink2 → Vox2) | ~? | 256 | no licence tag on HF | https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md | **0.20** | – | offline | ONNX |
| ReDimNet B0–B6 (IDRnD) | 1.0–15 M, 0.43–20 GMAC | 192 | **MIT** (VoxCeleb-trained) | `torch.hub.load('IDRnD/ReDimNet', ...)` https://github.com/IDRnD/ReDimNet | B6 0.40 | pretrained on 2 s crops | B0–B2 ~5–10 ms; 72-d fbank, 15 ms hop (non-standard) | plain torch |
| 3D-Speaker ERes2NetV2 | 17.8 M, 12.6 GFLOPs | 192 | Apache-2.0 code | ModelScope | 0.61 | **2 s 1.48 %, 3 s 0.98 %** (best published) | ~100 ms → teacher | ONNX export |
| SpeechBrain ECAPA | 20.8 M | 192 | **Apache-2.0** (weights too) | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | 0.80 | see ECAPA | ~20–25 ms | needs speechbrain to load; exports |
| pyannote/embedding | ~4 M | 512 | MIT, gated | https://huggingface.co/pyannote/embedding | 2.8 | – | cheap | pyannote.audio |
| pyannote community-1 | pipeline | – | CC-BY-4.0, gated; embedding = WeSpeaker ResNet34-LM; VBx | https://huggingface.co/pyannote/speaker-diarization-community-1 | AMI-IHM DER 17.0 | – | – | pyannote.audio 4 |
| ECAPA2 | 27 M | 192 | **CC-BY-NC-4.0** | https://huggingface.co/Jenthe/ECAPA2 | 0.34 | 0.5–2 s crops 7.9 % | heavy | TorchScript; **exclude (NC)** |
| WavLM-Base-Plus-SV | 95 M | 512 | MIT | https://huggingface.co/microsoft/wavlm-base-plus-sv | 0.84 | 1 s 18.4 %, 2 s 5.2 % (https://arxiv.org/html/2609.25007) | 300–500 ms | transformers; not a companion |

Short-segment ceiling (clean VoxCeleb): 2 s costs 2–3× the full-length EER (≈ 1.5–2 %), 1 s costs 6–15× (≈ 5–13 %).
Meeting audio roughly doubles that, so TitaNet-L's 8 % within-meeting at 1–2 s is about what a good model does; a
distilled head should aim for that band, not < 2 %.

Distillation recipes that worked (and the one that did not):
- **Cosine-only embedding KD adds almost nothing over a classification loss** (CAM++ 0.718 → 0.713); logit-KL 0.633;
  emphasised-non-target KL 0.590. https://arxiv.org/abs/2309.14838
- **Frozen ASR Conformer + per-layer adaptors + K light Conformer layers + attentive pooling, +4.9 M params → 0.57 %**
  (beats ECAPA 0.68 %). https://arxiv.org/abs/2309.03019 . Whisper middle-to-late blocks carry the most speaker
  information (Whisper-PMFA 1.42 %). https://arxiv.org/abs/2408.15585
- **MT2KD: cosine to an ECAPA teacher through a small head on a shared Zipformer encoder on unlabelled audio, then
  multi-task fine-tune: 2.40 → 1.13 %.** Works because the encoder trains through the loss; with ours frozen expect a
  larger gap. https://arxiv.org/html/2409.17010
- **MSA-ASR: frozen Whisper encoder, transformer speaker module cross-attending to encoder states, loss = cosine to
  TitaNet-L + MSE between pairwise similarity matrices.** The relational term is what SPK_HEAD.md's recipe lacks.
  https://arxiv.org/abs/2411.18152
- Long-to-short KD: teacher sees the whole turn, student the 1–2 s window. https://arxiv.org/abs/1810.10884
- Duration-aware Matryoshka (DAME): ResNet34 5 s-vs-1 s 4.33 → 3.29 %. https://arxiv.org/html/2601.13999

Integration: `heads/audio.py SpeakerHead`: add `n_layers: 2` causal FastConformer blocks (d 144–192, reuse
`modules/fastconformer.py`) between the layer mix and `AttentiveStatsPool`, per-layer LayerNorm+Linear adaptors on
taps {6, 9, 12, 15} (~120 LOC); loss = AAM + cosine-to-teacher + relational MSE over the batch similarity matrix (~40 LOC
in `SpeakerHead.distill`). CPU: 2 blocks × d 192 on 12.5 Hz frames ≈ 1/20 of the encoder (est. from d² ratio) ≈ 2–3 ms
per 160 ms chunk. Companion for the server: CAM++-LM or TitaNet-S at ~10 ms per look-back embedding instead of 45 ms
(`enrollment.TitaNetEmbedder` gets an ONNX sibling, ~60 LOC). Control: SPK_HEAD.md arms A (layer sweep) and B (cosine
distillation) as pre-registered. Kill: the 2-block head with relational KD does not reach < 20 % within-meeting on AMI
dev **and** ICSI dev (SPK_HEAD.md's line); then the features are the limit and the companion model is the product answer.

## 4. Target-speaker VAD / personal VAD

Row moved: "Turn end, floor-open ends" (replaces diarizer + enrollment rule with one conditioned head; removes the 840/
240 ms diarizer delay) and "Product dead air".

| method | conditioning | enrollment | data | result | streaming | code | URL |
|---|---|---|---|---|---|---|---|
| Personal VAD (Google 2019) | ET: d-vector concatenated to every frame; ST: frame-level cosine score concatenated; SET both; 2× LSTM-64, 3 classes ns/tss/ntss, weighted pairwise loss | LibriSpeech utterances | concatenated LibriSpeech + MTR | AP(tss): SC .886, ST .956, ET .932 (CE) / .955 (WPL), SET .970; 0.13 M params | yes | unofficial GPL-3 https://github.com/pirxus/personalVAD | https://arxiv.org/abs/1908.04284 |
| **Personal VAD 2.0** (2022) | **FiLM on Conformer output** γ(e)·h + β(e), or "speaker pre-net" cosine → FiLM; paper: input concat is "sub-optimal"; enrollment-less training (e = 0 w.p. 0.2, ntss → tss) | d-vector | concat LibriSpeech + internal | downstream WER (non-concat/concat test): concat 15.3/31.5 → FiLM 11.3/29.5 → pre-net 11.7/27.5; 1.0 MB int8 | yes: causal Conformer, no right context | none | https://arxiv.org/abs/2204.03793 |
| TS-VAD (2020) | i-vector concatenated to MFCC per speaker → BLSTMP → combining BLSTM | clustering i-vectors, iterated | CHiME-6 | DER 32.8 / 36.0; x-vectors "much worse" than i-vectors here | no | https://github.com/dodohow1011/TS-VAD (no licence) | https://arxiv.org/abs/2005.07272 |
| Online TS-VAD (2022) | target-embedding buffer repeated over time + Transformer | none: first 16 s block assumed single speaker | AliMeeting | 2 s latency after 16 s init | block-online | no | https://arxiv.org/abs/2207.05920 |
| Seq2Seq-TS-VAD / S2SND | speaker embeddings as decoder queries (cross-attention) | ≥ 2 s per speaker | VoxConverse, DIHARD-III | DER 4.55 / 10.77 | S2SND online | AED-TSVAD https://github.com/Clovermax/AED-TSVAD | https://arxiv.org/abs/2210.16127 , https://arxiv.org/abs/2411.13849 |
| **Bovbjerg 2025** (causal 2-layer Conformer-64, 310 ms context) | **explicit comparison: FiLM 79.54 mAP, add 79.29, mult 79.00, concat 78.47** | ≥ 5 s, 256-d d-vector | concat LibriSpeech + noise | spread ≈ 1 point; mult best on tss AP | yes | no | https://arxiv.org/abs/2501.03184 |
| PVAD comparison (2024) | DSC / EF (concat) / LF / CLF (FiLM) / DCLF | d-vector | LibriSpeech + MUSAN | CLF best frame EER; 5 % of DSC params; ≥ 40 % lower latency than score combination | yes | no | https://arxiv.org/abs/2406.09443 |
| **Short-enrollment PVAD** (2026) | PVAD-2.0 FiLM Conformer + CAM++ 192-d, self-augments the enrollment from tss frames | **0.5 / 1 / 1.5 s vs 7.4 s** | LibriSpeech 500 h + MUSAN | tss AP 78.98 @0.5 s, **89.41 @1.5 s**; 5 self-updates → 88.39 @0.5 s ≈ full enrollment | yes | anonymous repo | https://arxiv.org/abs/2601.12769 |
| HyWA (2025) | hypernetwork writes weight deltas into a frozen VAD (MarbleNet / FSMN) from the enrollment | ~11 s | LibriSpeech mixes + 1,560 real recordings | mAP 91.6 clean; false interruptions 88.9 → 9.9 % | yes | promised | https://arxiv.org/abs/2510.12947 |
| USEF-TP (2025) | cross-attention to raw enrollment frames, no fixed vector | full utterance | LibriMix | recall .970 vs .874 with an embedding | not causal | https://github.com/ZBang/USEF-TP | https://arxiv.org/abs/2501.03612 |
| FireRedChat pVAD (2025) | ECAPA embedding concatenated after causal convs → GRU | – | 2000 h zh/en | false barge-in 10.2 % vs LiveKit 33.4 % | yes, 1 s chunks | no | https://arxiv.org/abs/2509.06502 |
| EEND-SAA main-speaker VAD (2025) / Mamba-FVAD (2026) | enrollment-less "dominant speaker" by continuity/volume | none | LibriSpeech mixes / VOiCES | main-speaker DER 6.63 → 3.61; FVAD 1–2 ms/frame CPU | yes | no | https://arxiv.org/abs/2509.11957 , https://arxiv.org/abs/2609.19856 |
| DiCoW / Whisper-TS (BUT) | frame-level STNO mask → per-class affine transforms at every encoder layer (FDDT); "a single bias per class before the first block suffices" | none (needs a diarization mask) | AMI, NOTSOFAR | tcpWER AMI 17.6 with oracle diar | no | https://github.com/BUTSpeechFIT/DiCoW (Apache-2.0 code, CC-BY-4.0 weights) | https://arxiv.org/abs/2409.09543 , https://arxiv.org/abs/2501.00114 |
| Sortformer speaker kernels (NVIDIA) | additive sinusoidal kernels on encoder states from diarizer probabilities (what our `condition_on_speaker` copies) | none | LibriSpeechMix, AMI | Sortformer-MS-Canary 4.61 WER 2-mix | streaming Sortformer | NeMo Apache-2.0 | https://arxiv.org/abs/2409.06656 |
| pyannote 3.x/4.x | no target-speaker or enrollment option in the open models; voiceprints/streaming only in the paid API | – | – | – | – | – | https://docs.pyannote.ai/tutorials/identification-with-voiceprints.md |

What won: **FiLM (or any multiplicative modulation) of hidden features after at least one nonlinear layer**, never raw
input concatenation; applied early when the encoder is trainable (SpeakerBeam after block 1, DiCoW's bias before block
1), on the encoder output when it is frozen (PVAD 2.0). Score-based conditioning (frame cosine to the target) is as good
or better but needs a frame-level speaker model at run time. Enrollment: ≥ 1.5 s minimum, 3–5 s comfortable, marginal
past 5–8 s; refresh the vector from confident tss frames. Enrollment-less training keeps plain-VAD behaviour before the
user has spoken, which is our product's first second.

Smallest build (`heads/audio.py`, new `TSVADHead`, ~150 LOC): input = frozen encoder frames (layer mix) + 192-d
enrollment vector; 2-layer MLP → (γ, β) FiLM on the frames; causal GRU-96 (reuse `TurnHead`'s pieces) → 3-way
ns/tss/ntss softmax; weighted pairwise loss (w⟨ns, ntss⟩ = 0.1) or CE + enrollment-less dropout (p 0.2, ntss → tss).
Optional pre-net: cosine between a per-frame learned projection and the enrollment as a second FiLM input. Training
data we already have: AMI/ICSI single-speaker stretches (`asr` mode) with cross-meeting enrollment clips (DATA_PLAN.md
(e)), LibriSpeech concatenations + MUSAN, and the synthetic generator with a Kokoro agent track. Teacher vector:
TitaNet-L (cached by `scripts/cache_titanet.py`), later our own head. CPU: one GRU pass on 12.5 Hz frames, < 1 ms per
chunk; the enrollment embedding is computed once. Control: the Sortformer track + `causal_dominant`, and the Sortformer
track + TitaNet `VoiceFollower` (EOT_BENCH_V2 §8/§9 rows). Kill: enrollment agreement at the turn end < 85 % on
eot-bench v2 with the "first voice after the agent stops" rule, or floor-open miss not below the timeout's 46.1 % with
CI excluding 0 (DATA_PLAN.md gate). Expected from the literature: tss AP 0.90–0.95 clean / 0.85–0.90 noisy for a
100–300 k-param head with ≥ 3 s enrollment; ~10 points lower at 0.5 s.

## 5. State repair / delayed correction

Verdict: **no paper does "learned repair of a streaming model's hidden state after a delayed upstream speaker
correction."** Everything found is either recompute/replay or post-hoc label editing that never touches model state.

| work | state | trigger | replay or learned | effect | URL |
|---|---|---|---|---|---|
| SEAL (speaker error correction with an acoustic-conditioned LLM, ICASSP 2025) | none: post-hoc word speaker labels | second pass over the transcript + diarizer posteriors | learned, offline | 24–43 % rel. speaker-error reduction on Fisher/CallHome/RT03 | https://arxiv.org/abs/2501.08421 |
| DiarizationLM (2024) | post-hoc labels | offline LLM | learned, offline | 55.5 % rel. WDER Fisher | https://arxiv.org/abs/2401.03506 |
| Streaming Sortformer AOSC | speaker cache + FIFO | periodic cache update; frames scored once (log P_i + Σ log(1−P_j), recency bonus, top-K boost, silence padding) | **neither**: eviction only, no re-scoring or re-permutation of past outputs; no ablation | DIHARD3 ≤ 4 spk 13.3–13.7 % at 0.32–10 s latency | https://arxiv.org/html/2507.18446 , https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/modules/sortformer_modules.py |
| Sortformer v2.1 card | same | same; "filtered to retain high-quality cache vectors"; no self-correction | – | AMI-IHM 16.67 % vs v2 25.11 % | https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1 |
| Speaker-tracing buffer online EEND (2021), FLEX-STB, EEND-GLA | buffer of past frames + outputs | every chunk | recompute; emitted labels not revised | DER 12.5 CALLHOME at 1.4 s | https://arxiv.org/abs/2006.02616 , https://arxiv.org/abs/2101.08473 , https://arxiv.org/abs/2206.02432 |
| diart (Coria 2021) | rolling buffer + incremental centroids | Hungarian per buffer | recompute; **past labels never change**; latency 1 → 5 s trades confusion 27.6 → 25.0 DER | – | https://arxiv.org/abs/2109.06483 |
| Google multi-stage clustering (2022) | all past embeddings | re-clusters on every turn | recompute; **explicitly re-labels history** | bounded cost | https://arxiv.org/abs/2210.13690 |
| Asynchronous revision ASR (2020); encoder-state revision (2022) | encoder/decoder states | arrival of right context / fixed delay | recompute | 8–14 % rel. WER; 3.7/9.2 WER causal | https://arxiv.org/abs/2011.01570 , https://arxiv.org/abs/2207.02495 |
| LocalAgreement (Whisper-Streaming), partial rewriting, stability metrics | emitted text | re-decode | policy | – | https://arxiv.org/abs/2307.14743 , https://arxiv.org/abs/2312.09463 , https://arxiv.org/abs/2006.01416 |
| Label-delayed endpointer training (2025) | training targets | target delay at train time | training scheme | 42.7 % rel. cutoff-error reduction at 160 ms | https://arxiv.org/abs/2506.07081 |
| **KVEraser** (2026) | LLM KV span | stale / retracted observation | **learned** steering states replace the span | ≈ recompute quality at +24 % latency vs 17.6× | https://arxiv.org/abs/2606.17034 |
| **Budgeted repair of stale KV caches** (2026) | KV cache | document edit | bounded contiguous recompute **beats learned selectors** | ≥ 0.94 of margin recovered, 13–21× faster | https://arxiv.org/abs/2609.17983 |
| Counterfactual latent carriers (2026) | GRU world-model state | requested edit | learned rank-4 intervention | stays on counterfactual trajectory 12 steps | https://arxiv.org/abs/2608.15156 |
| AssemblyAI streaming | product | `speaker_labels_revision_interval_ms` → `SpeakerRevision` messages | provider re-clustering | only speaker fields change | https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api |
| Speechmatics real-time | product | sentence-boundary relabel inside Finals | heuristic | – | https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization |
| LiveKit / Pipecat / Deepgram | product | no revision event type | – | – | https://docs.livekit.io/agents/models/stt/ |

Implications for CONTAMINATION.md:
- The reviewer's baselines, in order: ignore; hard reset of GRU + counters at the correction; **bounded replay of the
  head over the last W cached encoder frames with corrected columns (`head_replay`, already implemented, O(W) GRU steps
  ≈ microseconds)**; full replay as the oracle; plus a stability/flicker metric. `head_replay` is the 2609.17983
  analogue and is the one to beat.
- It is a paper only if learned repair matches `head_replay` within noise while being O(1) per correction, beats reset
  clearly, holds on real Sortformer column swaps (not synthetic), and generalises across correction delays 0.3–10 s.
- It is a footnote if `head_replay` over a few hundred ms recovers > 90 % of the gap (the KV-repair result predicts
  this). Then the publishable artefacts are the contamination measurement itself and a `SpeakerRevision`-style event
  from `streaming_diar.py` to `serve.py` consumers (AssemblyAI is the only product that ships one).
- Framing that survives review: "delayed-correction robustness for speaker-conditioned streaming heads", unifying
  KVEraser, Google re-clustering and AssemblyAI revision events.

## 6. Combinations that would be new on this substrate

### 6.1 VAP-style discrete future-activity head on a frozen streaming ASR encoder, scored leak-free at matched FC on both eot-bench v2 and TurnBench

- Why nobody has done it: VAP runs on frozen CPC (Libri-light) and its evaluations are balanced-accuracy at sampled
  silences; the ASR-encoder turn heads (Parakeet-EOU, Kyutai, X2-Turn, Google) use one-channel VAD-shaped targets;
  Voice-Light tried a shared-encoder EOT and lost to Silero without matched-FC analysis. No one has put the two-speaker
  projection target on an ASR encoder *and* reported miss at a fixed false-cutoff budget on a public dyadic set.
- What we already have: `future_act_aux` (independent bins), the leak-free scorer, the streaming == offline test.
- Build: §2.1 head (~80 LOC) + `datasets/oto.py` (~250 LOC) + TurnBench runner (~200 LOC). Train on AMI/ICSI + 50–141 h
  otoSpeech with derived two-column activity; the primary is column 0, the other party column 1.
- Paper result: on TurnBench dev (mixed mono input, our tracker, causal), EOT recall at FPR ≤ 0.10 ≥ the published VAP
  row (0.845 at 0.055 on test, which had both channels) with lower median latency, *and* on eot-bench v2 a paired gain
  over the trail6 hybrid on floor-open ends with CI excluding 0. The interesting scientific sentence is "a frozen ASR
  encoder carries enough future-activity information for VAP-level turn prediction at no extra encoder cost".
- Kill: TurnBench-dev recall < the Silero-timeout row we compute locally, or the discrete head is a null vs the BCE aux
  on eot-bench (VAP's own ablation says the discrete edge is small). Then the contribution shrinks to "TurnBench numbers
  for a mixed-mono system", still useful, not a paper.

### 6.2 Enrollment-conditioned TS-VAD driving the turn head with no diarizer

- Why nobody has done it: personal-VAD papers stop at frame AP on concatenated LibriSpeech; turn-taking papers either
  ignore identity or (Meta 2026) use a proprietary enrollment-free tracker; our own measurement (oracle identity →
  misses 62 → 29 %) is the only quantified link between identity and turn misses we know of.
- Build: §4 `TSVADHead` (~150 LOC) + feed its tss posterior as `spk_act` / column 0 to `TurnHead` (existing interface;
  `cols` becomes [tss, ntss]) + `serve.py --enroll` path already planned. Enrollment: "first voice after the agent stops"
  with ≥ 3 s (the 1.5 s used in §8 is below the literature's cliff) and online refresh from tss frames
  (https://arxiv.org/abs/2601.12769). Remove `StreamingDiarizer` from the turn path; keep it for diarization output.
- Paper result: eot-bench v2 floor-open miss below the timeout's 46.1 % with CI excluding 0, enrollment agreement
  ≥ 85 %, and product dead air down by the diarizer delay (240–840 ms) in Pipecat/LiveKit. Title-grade claim: "a 0.2 M
  parameter enrollment-conditioned head replaces a 100 M-parameter diarizer for turn-taking".
- Kill: agreement stays ≤ 75 % (DATA_PLAN.md stop rule), or the TS-VAD head's tss AP on AMI bystander-heavy windows is
  below 0.80 (then AMI is simply the wrong test; report on the dyadic sets and move on).
- Caveat: on two-party otoSpeech/TurnBench audio, identity is trivial (the other party is on the other channel), so
  this idea is only testable on mixed audio with bystanders: AMI/ICSI, plus synthetic bystander injection into dyadic
  audio (our generator, or Behavior-SD voices mixed in as a third source).

### 6.3 Agent-aware training: the agent channel as a known input, on real dyadic audio

- Why nobody has done it: full-duplex LLMs consume both streams but train on Fisher/synthetic and never publish matched-
  FC numbers; cascaded agents (Pipecat, LiveKit) discard the agent's own TTS at the turn model; VAP treats both channels
  symmetrically. Nobody trains a cheap turn head with column 1 = *exact* agent activity (available for free from the
  TTS timeline) and column 0 = the user, and asks how much of VAP's dual-channel advantage (TurnBench: ESPnet dual .826
  vs per-channel .711 recall) survives when only the agent side is exact.
- Build: `conversation.py` agent channel from Kokoro word timestamps (~150 LOC), plus otoSpeech with one channel
  designated "agent" (its activity given as oracle, its audio optionally replaced by Kokoro re-synthesis of the CTC
  transcript to make the agent side synthetic-clean); `TurnHead` already takes `cols`.
- Paper result: the agent-aware head beats the same head without the agent column on TurnBench dev and on eot-bench
  "taken" ends by a paired margin, and the gain persists when the human side is real (otoSpeech) not TTS (Behavior-SD).
  Also a clean negative is publishable: "the agent channel adds nothing once the user's own end is well modelled".
- Kill: no paired gain on TurnBench dev at FPR ≤ 0.10 after 50 h; or the gain exists only with a TTS human (Behavior-SD),
  which would mean it is a synthetic artefact.

### 6.4 (Smaller) Kyutai-style "pause ≥ h" heads as calibrated timeouts

Four sigmoids on the GRU state for h ∈ {0.5, 1, 2, 3} s (Unmute fires at h = 2 s), trained on all corpora; a
matched-FC comparison of "learned pause ≥ h" against the fitted silence timeout isolates whether any semantics survive
on open ends. ~20 LOC. Kill: identical curve to the timeout (likely on AMI given Voice-Light and our floor-open tie).

## 7. Start this week (ranked, Mac only)

| # | item | why first | effort | gate to continue |
|---|---|---|---|---|
| 1 | **Request TurnBench dev (4.2 GB) and otoSpeech-141h/280h (19–49 GB); build `scripts/bench_turnbench.py` and score our current hybrid + Silero timeout + Parakeet-EOU with the MIT scorer on mixed mono** | external, hand-labelled, dyadic, our protocol; tells us in one day whether the AMI floor-open loss is an AMI artefact | 1 day after the gate opens (forms are contact-only) | our hybrid ≥ Silero timeout at FPR ≤ 0.10 on dev; if not, item 2 is the fix path |
| 2 | **`datasets/oto.py` + derived two-column labels; retrain trail6 with column 1 = other party on AMI/ICSI + 50 h otoSpeech; re-score eot-bench v2 and TurnBench dev** | first real dyadic training signal, permissive licence, no diarizer on that data | 2–3 days (label derivation ~1 day; training 2000 steps ≈ 20 min on MPS per DATA_PLAN.md rates) | paired gain on TurnBench dev recall or eot-bench open ends; else stop adding hours (PLAN.md: more turns alone did nothing) |
| 3 | **Speaker head redesign per §3: 2 causal blocks on taps {6, 9, 12, 15} + AAM + cosine + relational MSE to TitaNet-L; plus CAM++-LM / TitaNet-S ONNX as the 10 ms companion in `enrollment.py`** | the measured 37-point loss is identity; literature says the head design, not the teacher, is what we got wrong | 2 days (head 1 day, eval reuses `scripts/spk_head.py`; companion 0.5 day) | < 20 % within-meeting EER on AMI **and** ICSI dev (SPK_HEAD.md line); else ship the companion and stop |
| 4 | **`TSVADHead` (FiLM, ns/tss/ntss, enrollment-less dropout) on AMI/ICSI + LibriSpeech concatenations, enrollment ≥ 3 s from the "first voice after the agent stops" rule; feed `TurnHead`** | removes the diarizer and its 240–840 ms from the turn path; the only route to the floor-open row that attacks the measured bottleneck | 3 days (head + data 1.5, eot-bench rows 0.5, serve path 1) | agreement ≥ 85 %, floor-open < 46.1 % with CI excluding 0; else record the negative and keep Sortformer + timeout |
| 5 | **VAP 256-way head as a drop-in for `future_act_aux` with 80 ms bins [2, 5, 8, 10]; one ablation run** | cheapest possible test of the best TurnBench baseline's objective on our encoder | 0.5 day | paired gain vs BCE aux on eot-bench 6 s; a null is expected and closes the question |

Deferred (not this week): CANDOR (human approval, > 1 TB, licence unclear); LDC purchases (fees hidden behind login,
non-commercial terms); learned state repair (run `head_replay` sweep first; §5 says it probably wins); the 290 GB
labelled otoSpeech-104h (non-commercial; use only if item 2 shows the derived labels are the limit).

Open items to close before any claim: CANDOR licence text; MultiDialog's CC variant; Pipecat v3.x dataset licence;
whether anyreach DualTurn session ids join to the CC BY otoSpeech audio (would give free 80 ms labels); exact LDC fees.
