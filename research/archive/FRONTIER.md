# Frontier: new problems the 2025-2026 NVIDIA speech recipe makes tractable for a small team

2026-09-28. Owner: the frontier-scout agent. Request (verbatim): "launch another agent... new, and more interesting
work with the nvidia 2025-2026 architecture". Scope rule: nothing below repeats the project's existing lines (turn
detection, VAD, speaker head, diarization head, LID head, YIELD tokens, completeness head, encoder gating, the 0.6B
encoder; see `IMPROVEMENTS.md` and `FINAL_REPORT.md` §10). Sources were checked on **2026-09-28**; every external
claim carries a URL. "(est.)" marks our own estimate. Machine: one Apple M5 laptop shared with other agents (CPU
2 threads through `scripts/dev/gate.sh`, no MPS training by this agent), plus rented GPUs for anything marked GPU.

Part 1 ranks eleven problems. Part 2 runs the two cheapest informative pilots on CPU. Part 3 is next month's plan.

## 0. What changed in 2025-2026 (why now)

| release (date) | what it gives a small team | source |
|---|---|---|
| Nemotron-Speech-Streaming-en-0.6B (Jan 2026, new checkpoint 2026-03-13) and **Nemotron-3.5-ASR-Streaming-0.6B** (2026-06-04, OpenMDW-1.1): cache-aware FastConformer-RNNT, 24 layers, chunks 80 / 160 / 320 / 560 / 1120 ms, **40 language-locales** with a language-ID prompt (`target_lang` or `auto`); FLEURS WER en 7.91 %, es 4.11 % at 1.12 s | a multilingual streaming encoder on the same 80 ms clock as our 115M core | https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b , https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/commit/29175b5e2cf5f359740c512f8a243257416ad31e |
| Cache-aware serving numbers: ~560 concurrent streams per H100 at 320 ms chunks; 240 vs 14 streams at 80 ms, 2400 vs 400 at 1.12 s against buffered streaming | fixed-size per-stream state makes batching many sessions predictable | https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents |
| **Nemotron-3-Diarization** 100M (2026-09-23, OpenMDW-1.1): **8 speakers**, 10 ms output resolution, operating points 0.32 / 0.64 / 1.04 / 30.4 s; #1 on Voice Arena Diarization-Bench (14.72 % DER) | a small, commercially licensed streaming diarizer that already runs in our server at RTF 0.21-0.30 (we import 4 of its 8 columns) | https://www.marktechpost.com/2026/09/23/nvidia-releases-nemotron-3-diarization/ , `research/E2E_FINAL.md` |
| Streaming Sortformer v2 / v2.1 (AOSC speaker cache, arrival-order columns, 4 speakers) | per-frame speaker columns with stable identity across a session | https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/blob/main/README.md |
| **multitalker-parakeet-streaming-0.6b-v1** (speaker kernels injected into encoder layers, driven by Sortformer activity; cpWER AMI-IHM 21.26 %, AMI-SDM 37.44 %, CH109 15.81 % at 1.12 s) and NVIDIA's systematic study of streaming multi-speaker ASR (2026-09-24): cascade 45.25 % average cpWER vs **self-speaker adaptation (SSA-v2) 15.38 %**, oracle-diar SSA 13.00 %, all on Nemotron streaming ASR + Sortformer v2.1 at 1.04 s | the target: per-word speaker labels in streaming, with published numbers on AMI to compare against | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 , https://arxiv.org/html/2609.10265 |
| Parakeet-TDT-0.6B-v3 (25 European languages, automatic language detection, word timestamps) | the per-turn offline transcript pass we already run (AMI 9.7 % WER) is a free, stronger "teacher" on every turn | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| Canary-Qwen-2.5B (SALM: FastConformer encoder + frozen Qwen3-1.7B + LoRA; 5.63 % Open ASR Leaderboard; 90k steps on 32 A100) | the recipe for putting an LLM on a FastConformer, with the LLM frozen | https://huggingface.co/nvidia/canary-qwen-2.5b |
| Nemotron 3 VoiceChat / NemotronLabs-VoiceChat-11B (FastConformer → Nemotron Nano v2 9B → TTS codec decoder, full duplex); **streaming user transcription as a light ASR head beside the agent text head** (10.21 % streaming Open-ASR WER inside the duplex model, 2026-09-14) | NVIDIA itself now bolts light heads onto a duplex model's frames: the same "one encoder, many heads" pattern we use | https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B , https://arxiv.org/abs/2609.15759 |
| NanoCodec (Interspeech 2025): 12.5 fps FSQ codec, 22.05 kHz, 0.6-1.78 kbps, 62M parameters, 13 codebooks × 2016 codes at 1.78 kbps | a generation-side token stream on the **same 80 ms clock** as the FastConformer (`ANALYSIS.md` §2) | https://arxiv.org/abs/2508.05835 , https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps |
| VibeVoice-ASR-Streaming (Microsoft, 2026-09-02; 1.5B / 7B LLM, 2.0 s algorithmic delay): AMI-IHM 19.83 WER / **27.48 cpWER**, AMI-SDM 39.01 cpWER | the strongest end-to-end speaker-attributed streaming baseline, at 13-60x our parameter count | https://arxiv.org/html/2609.02812v1 |
| Full-duplex models take the floor "when asked, not when needed": Moshi / PersonaPlex address false claims in 14-15 % and hazards in 4-7 % of opportunities (2026-09-17) | the content-aware "should I intervene" decision is open, even for duplex models | https://arxiv.org/abs/2609.19596 |
| Stateful streaming ASR across turns: 15-21 % relative WER reduction at utterance onsets vs resetting state (2026-08-22) | our server already keeps state; the onset gain is a free product claim to measure | https://arxiv.org/abs/2608.22101 |

## Part 1. Ranked problems

Ranking = (value to a voice product or a paper) × (chance the smallest experiment gives a clean answer) / cost,
discounted for overlap with other agents. Every item reuses the audioforge pieces: pure-PyTorch `SpeechModel`
(frozen NVIDIA encoder, new heads as small `nn.Module`s), `.afm` checkpoints, `nemo_import` / `load_any` for NVIDIA
weights (no NeMo), `StreamingSession` / `StreamingDiarizer` for streaming, `scripts/dev/gate.sh` for every run.

### Summary table

| # | problem | smallest falsifying experiment | metric vs baseline | data here | cost |
|---|---|---|---|---|---|
| 1 | **Speaker-attributed live captions** (per-word speaker labels) from the served stack | cascade on AMI dev: served RNNT words → Nemotron-3 / Sortformer columns → cpWER (**run, P1: Nemotron-3 32.4 % = oracle 32.3 %, WER floor 30.7 %; survives**) | cpWER vs speaker-agnostic WER (the attribution cost) and vs published cascades (NVIDIA 45.25 %, multitalker-parakeet 21.26 %, VibeVoice 27.48 %) | AMI, ICSI | CPU hours; then 1-2 GPU-days |
| 2 | **User-only transcript** (target-speaker ASR) in a shared room, from the TS-VAD track | mask / re-decode the served RNNT and the per-turn TDT v3 with the TS-VAD track on AMI dev | target-speaker WER incl. bystander insertions vs unfiltered transcript and vs a bound diarizer column | AMI, ICSI, TS-VAD checkpoints | ~1 CPU-day |
| 3 | **Partial-caption stability head** (commit / flicker) trained on free labels from the per-turn TDT v3 pass | label which streaming words survive into the TDT v3 final; logistic / tiny GRU on encoder frames + RNNT posteriors | flicker (revisions per word) and commit latency at matched final WER vs "show all partials" and fixed delays | AMI, ICSI, LibriSpeech | CPU hours |
| 4 | **Codec tokens from the streaming ASR encoder** (80 ms-synchronous speech tokens: voice conversion, S2S, backchannel synthesis) | per-frame probes from frozen encoder layers to 12.5 Hz codec codes on LibriSpeech (**run, P2: semantic codebook 44.7 % top-1 from block 9 vs 9.7 % log-mel; acoustic codebook at the control; semantic half survives**) | code top-1 / top-5 vs majority, persistence and log-mel controls; content WER of resynthesis | LibriSpeech; Mimi (cached), NanoCodec (download 0.25 GB est.) | CPU hour; then GPU-days |
| 5 | **Echo-aware barge-in** with the agent's TTS as a known second input | synthetic echo (RIR + nonlinearity + delay) of one DailyTalk / Behavior-SD channel into the other; a head reading encoder frames of mic *and* TTS reference | false barge-ins per agent-minute at ≥ 95 % true-barge-in recall within 300 ms vs Silero on the mic, residual-energy after a linear AEC | DailyTalk, Behavior-SD test, otoSpeech dev | 1-2 CPU-days |
| 6 | **Streaming SALM decoder on the frozen 115M encoder** (early semantics before the turn ends) | projector + LoRA on a small frozen LLM reading 12.5 Hz frames chunk by chunk (Canary-Qwen recipe, chunk-interleaved as in VibeVoice-ASR-Streaming) | time-to-correct-intent and WER vs cascade (streaming RNNT text → same LLM) | LibriSpeech, AMI, DailyTalk transcripts | 2-4 GPU-days |
| 7 | **Multi-tenant serving**: batched cache-aware sessions on one CPU / GPU worker | batch the per-session encoder step across N synthetic sessions; measure p95 chunk time | sessions per worker at p95 < 160 ms vs today's 1 session / 2 threads | any audio | 1-2 days engineering, CPU |
| 8 | **Beyond four speakers**: all 8 Nemotron-3 columns for captions and TS-VAD binding in large rooms | re-import with 8 columns; ICSI meetings with 5-9 speakers | cpWER / target-speaker F1 vs the 4-column import | ICSI | CPU hours |
| 9 | **Echo as free enrollment** (the agent's own TTS echo as labelled, channel-matched audio) | use echo segments as negative enrollment and channel calibration for the TS-VAD head | TS-VAD target F1 under channel mismatch vs no calibration | synthetic from #5 data, AMI | ~1 CPU-day after #5 |
| 10 | **Cross-lingual front end** on Nemotron-3.5-ASR-Streaming (40 locales) | read our English-trained heads' recipe on its frames; FLEURS-synthesised dialogues + EdAcc | head transfer (VAD F1, activity F1) vs English-only encoder; language-prompt ablation | FLEURS, EdAcc | model download ~2.4 GB (> 500 MB, note); CPU-days (0.6B at RTF ~4.6 on CPU) |
| 11 | **Speaker-attributed streaming translation captions** (DiariST-style) | Canary-1B-v2 / TDT v3 per attributed turn + a small MT LLM | speaker-attributed BLEU / COMET and latency | FLEURS (single speaker), AMI (en only) | GPU; no multilingual multi-party test set here |

### 1. Speaker-attributed live captions with per-word speaker labels

*What.* Live captions of a multi-party call where every word carries a speaker label, produced by the pieces we
already serve: the frozen 115M streaming RNNT at 160 ms chunks and an NVIDIA streaming diarizer on the same 80 ms
clock. Words are attributed to the diarizer column with the most activity over the word's estimated interval.

*Why newly possible.* Streaming Sortformer's arrival-order speaker cache gives columns whose identity is stable over a
session, and Nemotron-3-Diarization makes that cheap (RTF 0.21-0.30 on our CPU, 1.5 GB). NVIDIA's own 2026-09-24
study (https://arxiv.org/html/2609.10265) now quantifies the design space on AMI with exactly these components: a
cascade at 45.25 % average cpWER, self-speaker adaptation at 15.38 %; multitalker-parakeet reports AMI-IHM 21.26 %
cpWER at 1.12 s (https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1). Nobody has published the
*lightweight* point: a 115M frozen encoder + 100M diarizer at 0.32 s diarizer latency on a CPU.

*Smallest falsifying experiment.* Part 2 P1. Falsified if the attribution cost (cpWER − speaker-agnostic WER) with
the streaming diarizer is more than 2x the oracle-activity cost, i.e. if the diarizer, not the ASR, dominates.

*Metric / baseline.* cpWER (Hungarian over speakers per session and per minute); agnostic WER floor; oracle-activity
arm; published cascades above (different protocols: theirs are full-meeting, ours 6-minute spans of 4 AMI dev meetings).

*Next step if it survives.* Self-speaker adaptation *without touching the encoder* (the encoder fine-tune tripped the
WER gate every time, `FINAL_REPORT.md` §10): a per-slot conditioned decoder (FiLM of the column's activity into the
RNNT predictor / joint), trained on cached frozen frames of AMI + ICSI train; 1-2 GPU-days. Kill if cpWER does not
drop by ≥ 20 % relative over the cascade.

*audioforge.* `StreamingSession` (words + emission times), `StreamingDiarizer` with `SORTFORMER_PRESETS`,
`nemo_import.load_any` for Nemotron-3; a new `serve` message field `speaker` per word (the protocol already carries
a per-turn speaker).

### 2. User-only transcript (target-speaker ASR) in a shared room

*What.* The agent should transcribe *its user* and ignore the TV, the colleague and the bystander. Today the
transcript is everything the ASR hears; `FINAL_REPORT.md` shows who-the-user-is is the bottleneck for turns, and the
same holds for the transcript an LLM reads.

*Why newly possible.* The TS-VAD head (0.26 M, block 4 of the frozen encoder, `IMPROVE_115M.md` Part A) already
tracks an enrolled user at frame F1 0.743 on AMI and 0.882 on ICSI, beating a voice-print-bound Sortformer column
(0.629 / 0.690). Multitalker-parakeet shows that driving ASR from a speaker-activity stream is the NVIDIA way
(speaker kernels); the lightweight version is to use the activity track to *select* words (cascade) and then to
*condition* only the decoder.

*Smallest falsifying experiment.* On the 974 AMI dev eot-bench windows with a 5 s voice print: (a) served RNNT words
kept iff the TS-VAD track is on at their estimated time; (b) TDT v3 run on TS-VAD segments only. Score target-speaker
WER against the target's reference words (bystander words become insertions). Falsified if (a) is not better than
the unfiltered transcript by ≥ 5 WER points on windows with a second active speaker.

*Metric / baseline.* Target-speaker WER (and insertions from others per minute); baselines: unfiltered transcript,
Sortformer column bound by the same print (`enrollment.VoiceFollower`), oracle activity.

*Data / cost.* AMI, ICSI, cached TS-VAD checkpoints (`runs/tsvad_spk.pt`); ~1 CPU-day. *audioforge:* `heads/tsvad.py`,
`tsvad_stream.py`, `final_asr` worker.

### 3. Partial-caption stability head trained on free TDT v3 labels

*What.* A streaming caption should commit words as soon as they will not change. We run two ASR passes already: the
streaming RNNT partials (24 % AMI WER) and Parakeet-TDT v3 per turn (9.7 %). Every turn therefore labels, for free,
which streaming words survive. A tiny head on the frozen encoder frames and RNNT posteriors predicts "this word is
final" and gates display.

*Why newly possible.* The per-turn TDT v3 pass is cheap enough to run on every turn (RTF 0.065 on CPU) and has word
timestamps; the cache-aware encoder gives per-frame features at no extra cost. Recent work treats flicker and
erasure as first-class streaming metrics (https://arxiv.org/html/2609.26427v1 ,
https://arxiv.org/pdf/2601.20992).

*Smallest falsifying experiment.* Align streaming words to TDT v3 finals on 200 AMI dev turns; fit logistic regression
on (RNNT max-posterior, word age, encoder-frame norm, VAD); falsified if AUC < 0.75 or if, at the operating point
that keeps final WER unchanged, flicker falls by < 30 % vs showing every partial.

*Metric / baseline.* Revisions per displayed word; commit latency (word end → committed); baselines: show all,
fixed 0.5 / 1.0 s delay. *Data / cost:* AMI dev, ICSI, LibriSpeech; CPU hours.

### 4. Codec tokens from the streaming ASR encoder

*What.* Predict the tokens of a 12.5 Hz speech codec (NanoCodec, Mimi) frame-synchronously from the frozen 80 ms
FastConformer frames. If the encoder carries the codec's content stream, a small head gives streaming speech-to-speech
building blocks (voice conversion into the agent's voice, backchannel synthesis in the agent's voice, codec-domain
enhancement) with no second encoder.

*Why newly possible.* NanoCodec (https://arxiv.org/abs/2508.05835) and Mimi run at 12.5 fps, the FastConformer clock
(`ANALYSIS.md` §2); VoiceChat already feeds FastConformer frames to an LLM that emits codec tokens
(https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B). Whether a *streaming ASR* encoder (trained to
discard speaker and prosody) keeps enough for codec targets is untested; our earlier `codec_token_enhancer` used a
toy mel codec (`VERIFICATION.md` §3.2).

*Smallest falsifying experiment.* Part 2 P2: linear / MLP probes, speaker-disjoint LibriSpeech dev-clean. Falsified
for the semantic stream if codebook-0 top-1 is not clearly above a log-mel probe with the same capacity; falsified for
acoustic codebooks if they sit at the persistence / majority floor.

*audioforge.* `encode(..., return_hidden=True)` per-layer taps; a codec-token head is a frame head (`heads/`), `.afm`.

### 5. Echo-aware barge-in with the agent's TTS as a known input

*What.* The agent knows exactly what it is saying. A head that reads the encoder frames of the mic and of the TTS
reference (the same frozen encoder run on the known TTS signal, 80 ms aligned) and outputs "the user is speaking over
me" would separate true barge-ins from residual echo, the dominant false-interrupt source (Silero fires on agent
residual echo at −40 dB in 26-36 of 36 agent lines in one practitioner report,
https://github.com/qvd808/agent-emergency-call/issues/32 ; JarvisBench uses a 0.8 s sustained-speech rule after AEC,
https://arxiv.org/pdf/2608.14870).

*Why newly possible.* Cache-aware encoders make a second stream cheap (fixed state, batchable, #7); two-channel
conversational corpora with one speaker per channel (DailyTalk, Behavior-SD, otoSpeech) give exact ground truth for
synthetic echo mixing. Not in our negative table: `FINAL_REPORT.md` §11 lists "echo of the agent's own TTS" as never
measured.

*Smallest falsifying experiment.* 200 DailyTalk dialogues: channel A = agent, channel B = user; mic = B + echo(A)
(random RIR, 50-250 ms delay, soft clipping, −35 to −15 dB). Head = 2-layer GRU on [enc(mic) ‖ enc(A)] frames,
trained on cached features. Falsified if false barge-ins per agent-minute at ≥ 95 % recall within 300 ms are not
halved vs Silero on the mic after a linear NLMS AEC.

*Cost.* 1-2 CPU-days (feature caching dominates). *audioforge:* a two-input frame head; `serve` already receives the
TTS-end event (`--enroll after_agent_arm`), extend to the TTS audio.

### 6. Streaming SALM decoder on the frozen 115M encoder

*What.* A small frozen LLM (0.5-1.7B) with LoRA and a frame-stacking projector reading the 115M encoder's frames in
160 ms chunks, emitting text / intents while the user is still talking, so the agent's LLM prefill and retrieval start
before the turn ends.

*Why newly possible.* Canary-Qwen shows the frozen-LLM + LoRA recipe on FastConformer; VibeVoice-ASR-Streaming shows
chunk-interleaved streaming generation; NVIDIA added a light streaming ASR head to a duplex LLM
(https://arxiv.org/abs/2609.15759). The negative result that matters here: the completeness head found semantic
completeness no help for *turn timing* on meetings (`FINAL_REPORT.md` §10); this item targets early *content*, not
timing.

*Smallest falsifying experiment.* SALM on 960 h LibriSpeech, then DailyTalk (intents from the transcript by an LLM);
falsified if at equal final WER the streaming SALM gives the correct intent later than "streaming RNNT text → same
LLM" (the cascade is the baseline to beat, and it is strong).

*Cost.* 2-4 A100-days (est.: Canary-Qwen used 32 A100 × 90k steps at 2.5B; a 0.6B LLM on 1k h is ~1/100 of that).
*audioforge:* `train-salm` already exists (SALM bridge, `ANALYSIS.md` §3).

### 7. Multi-tenant serving of cache-aware sessions

*What.* Today one worker sustains one session per 2 threads (`README.md`, `PERFORMANCE.md`). Cache-aware streaming
state is fixed-size per session, so the encoder step can be batched across sessions (the GEMMs grow from 2-4 rows to
2N-4N rows, where BLAS is far more efficient: `perf.py` notes the 2-4-row GEMMs are the slow path).

*Smallest falsifying experiment.* Batch `stream_step` across N ∈ {1, 2, 4, 8} synthetic sessions on 2 threads;
falsified if sessions per worker at p95 chunk time < 160 ms does not reach ≥ 2.

*Cost.* 1-2 days engineering, CPU; GPU numbers need one rented hour. *audioforge:* `StreamState` batching in
`modules/fastconformer.py`, `serve.py` worker pool.

### 8. Beyond four speakers

Nemotron-3-Diarization emits 8 columns (https://www.marktechpost.com/2026/09/23/nvidia-releases-nemotron-3-diarization/);
our import keeps 4 (`serve --diar-spks 4`). ICSI meetings have up to 9 speakers. Smallest test: rerun item 1 and the
TS-VAD column binding with 8 columns on the ICSI held-out meetings; falsified if cpWER / binding F1 do not improve on
meetings with > 4 active speakers. CPU hours; `load_any` + `--diar-spks 8`.

### 9. Echo as free enrollment

Every TTS utterance is known-speaker, channel-matched audio arriving through the user's mic. Use it (a) as a negative
enrollment for the TS-VAD head (the head already has two outputs: target / other) and (b) to estimate a channel shift
for the user's voice print. Smallest test after #5: on synthetic echo data with a channel mismatch between print and
session, falsified if TS-VAD F1 gains < 0.03. The label-free-enrollment negative result (`FINAL_REPORT.md` §10) was
about choosing the user by voice alone; this uses the agent's own known voice, which is new information.

### 10. Cross-lingual front end on Nemotron-3.5-ASR-Streaming

The 40-locale streaming encoder (https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) is the multilingual
successor of our English core. Question: do frozen-encoder heads trained on English data transfer to other languages
on it, and does the language prompt matter for non-ASR heads? Smallest test: VAD and TS-VAD heads on its frames,
English train, FLEURS-concatenated two-speaker test in 5 languages. Low rank: the model is 0.6B (RTF ~4.6 on our CPU
for the English 0.6B, `FINAL_REPORT.md` §10), the download is ~2.4 GB (over the 500 MB note threshold), and we have
no real multilingual conversational test data.

### 11. Speaker-attributed streaming translation captions

DiariST-style captions (https://arxiv.org/pdf/2309.08007) from item 1's attributed turns fed to Canary-1B-v2 or an MT
LLM. Parked: no multilingual multi-party test set here; revisit after items 1 and 10.

## Part 2. Pilots (CPU, through `scripts/dev/gate.sh`, 2 threads, machine shared with other agents' jobs)

### P1. Speaker-attributed live captions from the served stack (problem 1)

**Setup** (`scripts/research/frontier_sa_captions.py`, output `runs/frontier_sa_captions.json`). Four AMI dev meetings
(IS1008b, ES2011b, TS3004b, IB4002; mix-headset audio), spans starting at 300 s: 360 s for the first two, 120 s for the
last two (the 30-minute budget ran out at the slow shared-machine RTFs below). The served streaming RNNT
(`runs/stage1_served.afm`, fed 160 ms blocks) emits words with emission times; each word takes the diarizer column
with the most activity over [t_first − 0.24 − 0.3, t_last − 0.24 + 0.08] s (the 0.24 s lag was fixed a priori from the
chunk + right context, not tuned). Diarizers at the served "ultra low latency" setting (chunk 3 + rc 1 = 0.32 s):
Nemotron-3-Diarization (max pooling, 4 columns, the recommended product diarizer) on all four meetings; Streaming
Sortformer v2 on IS1008b only (budget). Arms: oracle activity (AMI words, isolates ASR + the attribution rule) and
speaker-agnostic WER (time-ordered references vs all words: the floor that attribution cannot beat). Scoring: cpWER
with one Hungarian speaker permutation per span ("session", what a live caption must get right) and per 60 s window;
95 % CIs by bootstrap over the 60 s windows (16 windows, 2099 reference words; windows are clustered in 4 meetings,
so the CIs are optimistic).

**Results.**

| arm | session cpWER | [95 % CI, window bootstrap] | per-minute cpWER | n |
|---|---:|---|---:|---|
| speaker-agnostic WER (floor) | **30.7 %** | [24.6, 41.3] | 32.5 % | 16 windows, 2099 words |
| oracle activity (AMI word labels) | 32.3 % | [23.8, 44.1] | 33.1 % | same |
| **Nemotron-3-Diarization, 0.32 s, streaming** | **32.4 %** | [24.1, 43.6] | 33.1 % | same |
| Streaming Sortformer v2, 0.32 s (IS1008b only) | 45.8 % | [27.0, 87.8] | 43.5 % | 6 windows, 790 words |
| Nemotron-3 on IS1008b only (same 6 windows) | 24.1 % | | | 790 words |

Paired, per-minute permutation: Nemotron-3 − oracle **0.0 [−1.5, +2.0]** points; Nemotron-3 − agnostic +0.6
[−0.9, +2.5]; Sortformer v2 − Nemotron-3 on IS1008b **+18.6 [+3.1, +32.4]** points (6 windows of one meeting). Per
meeting (session cpWER, Nemotron-3 / oracle / agnostic): IS1008b 24.1 / 24.1 / 22.9; ES2011b 24.1 / 22.5 / 22.4;
TS3004b 41.6 / 41.6 / 41.6; IB4002 61.9 / 64.2 / 56.4 (IB4002's ASR alone is at 56 % WER). Compute (model only,
shared machine at load 3.5-4): ASR pass RTF 0.51-0.66, Nemotron-3 0.39-0.63, Sortformer v2 1.07; ~25 min of compute
in total for the pilot.

**Reading.** (1) *Attribution is nearly free with Nemotron-3:* the streaming 0.32 s diarizer attributes words as well
as the oracle labels do (session cpWER 32.4 vs 32.3 %) and only 1.8 points above speaker-agnostic WER. The
falsification test of problem 1 (attribution cost > 2x the oracle's) **fails to falsify**: 1.8 vs 1.6 points.
(2) *Sortformer v2 at 0.32 s loses speaker identity mid-session on IS1008b:* its per-minute cpWER equals the others in
minutes 1, 2 and 6, and jumps in minutes 3-5 (105 vs 35 errors in minute 3): column swaps inside a session, which a
per-window permutation would hide. One meeting; consistent with the product recommendation of Nemotron-3 over
Sortformer v2 (`E2E_FINAL.md`). (3) *The bottleneck is the words, not the speakers:* with a single-stream ASR the
floor is 30.7 % WER on real meeting spans (overlap included), so the remaining gap to multitalker-parakeet's 21.26 %
AMI-IHM cpWER (https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1, 1.12 s latency, full meetings)
and VibeVoice's 27.48 % is ASR on overlapped speech. Our cascade sits well below NVIDIA's own cascade figure (45.25 %
average over CH109 / Mixer6 / AMI, https://arxiv.org/html/2609.10265) but the protocols differ (spans vs full
meetings, 4 speakers, our normaliser, mix-headset only), so this is a pilot, not a leaderboard claim.

**Verdict: keep, and redirect.** Problem 1 is real and cheap: per-word speaker labels cost ~2 cpWER points over
plain WER with the diarizer we already serve. The next experiment is not self-speaker adaptation of the encoder but
better words inside attributed segments: re-decode each attributed turn with Parakeet-TDT v3 (the per-turn pass,
AMI 9.7 % on clean segments) and score cpWER on full meetings; kill if session cpWER does not fall below 25 % on the
AMI dev full meetings.

### P2. Codec tokens from the frozen streaming ASR encoder (problem 4)

**Setup** (`scripts/research/frontier_codec_probe.py`, output `runs/frontier_codec_probe.json`). LibriSpeech
dev-clean, speaker-disjoint: 25 train speakers × 10 utterances (25 187 frames), 5 validation speakers × 10 (used only
to choose the lag), 10 test speakers × 8 (7 158 frames). Features: outputs of blocks 2, 4, 6, 9, 12, 17 of the served
115M encoder (offline forward with its [70, 1] streaming mask, 80 ms frames). Targets: Kyutai Mimi codes (24 kHz,
12.5 Hz, 8 RVQ levels; codebook 0 is WavLM-distilled "semantic", 1-7 acoustic; 2048 codes each), which share the
80 ms clock. Mimi (385 MB) was already in the local HF cache (fetched by the previous, rebooted run of this agent) and
was read offline; NanoCodec, NVIDIA's own 12.5 fps codec, would need a download and a `nemo_import` of its FSQ
decoder, so it is the follow-up, not the pilot. Probes: linear softmax (AdamW, 12 epochs) from encoder frame t + lag to
code t; a 1024-unit MLP on the best layer. Controls: majority class; persistence (code(t) = code(t − 1)); a linear
probe on 5 stacked 80 ms log-mel frames (400 ms of context ending one frame after the source, matching the encoder's
one-frame lookahead). Resynthesis: 24 test utterances decoded by Mimi and transcribed by the served ASR. ~7 min of
compute in total (feature and code extraction 108 s).

**Lag** (validation speakers, block 12, codebook 0 top-1): 0 frames 12.0 %, 2: 32.5 %, **3: 37.8 %**, 4: 35.3 %,
5: 30.3 %, 6: 19.9 %, 8: 7.6 % → lag 3 (the encoder's content for a Mimi frame is best read 240 ms later).

**Per-layer probes, test speakers, lag 3** (top-1 / top-5; binomial 95 % half-width ≈ ±1.2 points at 45 %, frames are
clustered in 80 utterances so treat it as a lower bound):

| input | codebook 0 (semantic) | codebook 1 (first acoustic) |
|---|---|---|
| majority class | 1.1 % | 1.5 % |
| persistence (previous true code) | 6.2 % | – |
| log-mel, 5 × 80 ms stacked (400 ms) | 9.7 % / 27.5 % | **9.8 % / 32.3 %** |
| block 2 | 6.3 / 17.3 | 4.2 / 11.8 |
| block 4 | 12.9 / 31.2 | 4.8 / 13.8 |
| block 6 | 28.8 / 56.0 | 6.0 / 17.2 |
| **block 9** | **44.7 / 77.8** | 4.7 / 16.5 |
| block 12 | 32.0 / 63.8 | 3.4 / 10.7 |
| block 17 (top) | 10.0 / 22.0 | 2.4 / 6.9 |
| block 9, MLP (1024) | **48.9 / 81.8** | – |

**Resynthesis** (Mimi decode → served ASR WER, 24 utterances, 518 words; no CI stored, binomial ± ≈ 2.3 points at
8 %): original audio 2.9 %; Mimi with all 8 true codebooks 6.0 %; **predicted codebook 0 + true codebooks 1-7: 8.3 %**;
true codebook 0 alone 85.7 %, predicted codebook 0 alone 87.1 % (codebook 0 alone is not intelligible through Mimi's
decoder, so this pair says nothing).

**Reading.** (1) The frozen streaming ASR encoder carries the codec's *semantic* stream: block 9 of 17 predicts Mimi
codebook 0 at 44.7 % top-1 linearly (48.9 % with an MLP), 4.6x the 400 ms log-mel control and 40x the majority class,
while the codes change on 94 % of frames. The information peaks mid-network (block 9) and is almost gone at the top
(block 17: 10.0 %), the same shape as our speaker probes (`SPK_HEAD.md`): the RNNT-facing top layer has thrown away
everything that is not the token. (2) It does *not* carry the acoustic stream: codebook 1 from any encoder layer is
at or below the log-mel control (4.7 vs 9.8 %). As expected for an ASR encoder, voice and fine acoustics must come
from somewhere else. (3) Substituting predicted semantic codes costs +2.3 WER points of intelligibility after Mimi
decoding (6.0 → 8.3 %, within noise of n = 518 words), but the true acoustic codebooks also carry content, so this is
an upper bound on usefulness, not a demonstration.

**Verdict: keep the semantic half, kill the acoustic half.** "Frame-synchronous content tokens from the ASR encoder
we already run" survives at 240 ms extra delay: a streaming voice-conversion / speech-to-speech path can take its
content stream from block 9 for free and needs its own generator for the acoustic codebooks (conditioned on a target
voice, e.g. the agent's TTS voice). Kill criterion for the follow-up: with NanoCodec 12.5 fps as the target and a
small causal acoustic generator trained on LibriSpeech (content from block 9, speaker from a TitaNet embedding), kill
if resynthesis WER exceeds 2x the codec's own WER or if the encoder-side lag cannot be brought under 160 ms.

## Part 3. What I would do next month

Four weeks, one person, the laptop for CPU work (through the gate) and ~$300-600 of rented GPU (est. at $2-3 per
A100-hour). Ordered by what the pilots showed; each item has a kill criterion fixed now.

| week | item | concrete steps | success bar | kill criterion | cost |
|---|---|---|---|---|---|
| 1 | **Speaker-attributed captions, full meetings, better words** (problem 1) | run P1 on the 4 AMI dev meetings in full (plus ICSI held-out 5) with Nemotron-3 at 0.32 s; add the per-turn Parakeet-TDT v3 re-decode of each attributed segment; add a `speaker` field per word to `serve` partials | session cpWER below multitalker-parakeet's published 21.26 % AMI-IHM at ≤ 1.12 s caption latency, on CPU | kill the TDT arm if full-meeting cpWER does not fall below 25 %; kill the whole line if Nemotron-3 attribution cost exceeds 2x the oracle's on ICSI | ~6 CPU-h (RTF 0.5 on the shared machine), 0 GPU |
| 1-2 | **User-only transcript** (problem 2) | TS-VAD-masked words and TS-VAD-segmented TDT v3 on the 974 AMI + 1312 ICSI eot-bench windows, 5 s print | target-speaker WER −5 points vs unfiltered on windows with a second speaker | no gain ≥ 5 points on AMI *and* ICSI | ~1 CPU-day |
| 2 | **Caption stability head** (problem 3) | align streaming words to TDT v3 finals on AMI train turns; logistic then a 20 k-parameter GRU; flicker / commit-latency curves on AMI dev + ICSI | flicker −30 % at unchanged final WER | AUC < 0.75 on held-out ICSI | CPU hours |
| 2-3 | **Echo-aware barge-in** (problems 5, 9) | synthetic echo on 500 DailyTalk + Behavior-SD test dialogues; cache encoder frames of mic and TTS reference; two-input GRU head; NLMS-AEC + Silero baseline; then echo-as-negative-enrollment for TS-VAD | false barge-ins per agent-minute halved at ≥ 95 % recall within 300 ms | not halved on held-out otoSpeech-dev-based echo mixes | 1-2 CPU-days |
| 3-4 | **Content tokens → streaming voice path** (problem 4) | download NanoCodec 12.5 fps (1.78 kbps; 62 M parameters, est. ~0.25 GB); `nemo_import` its FSQ decoder; train a causal acoustic generator on LibriSpeech-960 (content = block-9 frames, speaker = TitaNet) on a rented GPU | resynthesis WER ≤ 2x the codec's own at ≤ 400 ms total delay | WER > 2x codec, or the encoder-side lag cannot go below 160 ms | ~1 A100-day (~$60) |
| 4 | **Multi-tenant serving** (problem 7) | batch `stream_step` across sessions (CPU) and measure on one rented L4 / A10 | ≥ 2 sessions per 2-thread worker at p95 chunk < 160 ms; ≥ 100 sessions per L4 | < 2 sessions per worker on CPU | 2 days eng., 2 GPU-hours |
| backlog | streaming SALM decoder (6), 8 speakers (8), multilingual front end (10), translation captions (11) | start 6 only if 1-3 are done and a GPU budget of ~$500 is approved; 8 folds into week 1 if ICSI shows > 4 active speakers matter | – | 6: intent no earlier than the cascade at equal WER | – |

What would change the plan: if week 1 shows Nemotron-3's attribution degrading on ICSI (up to 9 speakers, 4 columns
imported), problem 8 moves up to week 2; if problem 2's masked transcript already reaches the bar, the stability head
and echo work take its time.

## Appendix. Files and reproduction

- `scripts/research/frontier_sa_captions.py` (`run` resumable per unit under a wall budget, `score`); cached units on
  `/Volumes/ExternalSSD/nvidia-audio-models/scratch/frontier/sa_captions/`; result `runs/frontier_sa_captions.json`.
  Calls used: `run --budget 360`, `run --budget 420` (IS1008b, ES2011b at 360 s), `run --meetings TS3004b,IB4002
  --dur 120 --late ""`, `score`.
- `scripts/research/frontier_codec_probe.py` (`extract --budget 400`, `probe`); features and codes on
  `/Volumes/ExternalSSD/nvidia-audio-models/scratch/frontier/codec/` (380 files); result `runs/frontier_codec_probe.json`.
- Both drafts were left untracked by the previous, rebooted run of this agent under `scripts/`; they were moved to
  `scripts/research/`, given SSD scratch paths, resumable budgets, a validation split for the lag, the persistence and
  400 ms log-mel controls, the Nemotron-3 arm, per-window bootstrap CIs and the exact CPU fast paths
  (`audioforge.perf.linear_t` / `pos_cache` / `joint_cache`). The smoke output `runs/frontier_sa_captions_smoke.json`
  named in the hand-over did not exist on disk.
- Every model launch went through `scripts/dev/gate.sh`; no MPS; no model downloaded.
