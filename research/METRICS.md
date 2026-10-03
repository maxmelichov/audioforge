# Metrics: the standard voice-agent scorecard

2026-10-03 (scorecard values updated to the test-split pass; first written 2026-09-29). This is the one place that
defines every number we publish about audioforge. Each metric has one definition, one unit and one direction. Every
value comes from [FINAL_COMPARE.md](FINAL_COMPARE.md) (`runs/final_compare.json`) unless the row says otherwise. Where
we did not measure something, the table says "not measured".

- **audioforge** is the shipped product: `audioforge-serve` in single-model mode (`--mode single`), on either core:
  the frozen NVIDIA 115M streaming FastConformer with heads v0.4 (`stage1_served_v4.afm`), or the English 0.6B
  (nemotron-speech-streaming-en-0.6b) with heads v0.4 (`served_0p6b_v0.4.afm`); LID head v2 on both; the user's
  stored 5 s voice print. Cells give "115M / 0.6B".
- **LiveKit default** is LiveKit Agents 1.8.3 with Silero VAD and the LiveKit turn detector (EnglishModel). LiveKit
  has no default STT (`AgentSession(stt=...)` must be given).
- **Pipecat default** is Pipecat 1.12 with Silero VAD and smart-turn v3.2. Its local `WhisperSTTService` defaults to
  faster-distil-whisper-medium.en.
- For the STT rows, both stacks' columns show **Whisper small at beam 5** (faster-whisper defaults) as a small local
  Whisper; FINAL_COMPARE.md "Words" has Whisper large-v3 / large-v3-turbo too.
- **NVIDIA baseline** is NVIDIA's own model for the same job, where one exists.
- Every row is on a public **test** split, except the two-party calls (row 11), for which no labelled public test
  split exists. Rows flagged by the leakage audit are not quoted here (FINAL_COMPARE.md "Appendix A"); ⚑ marks a
  weaker flag explained there.
- Hardware: an Apple M5 laptop. Each row names the device.

## Scorecard

| # | metric | what it means | unit | better | audioforge 115M / 0.6B | LiveKit default | Pipecat default | NVIDIA baseline |
|---|---|---|---|---|---|---|---|---|
| **STT** | | | | | | | | |
| 1 | WER, LibriSpeech test-clean | share of words wrong on clean read speech (300 random utterances, Whisper normalizer) | % | lower | 2.38 / 2.69 | 3.10 (Whisper small) | 3.10 (Whisper small) | **1.83** (Parakeet-TDT 0.6B v3, offline) |
| 2 | WER, LibriSpeech test-other | the same, on harder read speech (300 random utterances) | % | lower | 6.82 / 5.71 | 7.10 | 7.10 | **3.40** (Parakeet-TDT v3) |
| 3 | WER, meeting speech (AMI test) | the same, on 200 single-speaker segments of the AMI test meetings (headset mix, Whisper normalizer) | % | lower | 16.11 / **7.87** | 10.90 | 10.90 | 8.31 (Parakeet-TDT v3, offline; − 0.6B: +0.44 [−0.89, +1.79]) |
| 4 | partial latency | time from the end of a spoken word to that word first appearing in the live transcript (MPS) | ms, p50 / p95 | lower | **274 / 571**, 291 / 591 | none: an end-of-utterance STT shows no text until the turn ends (row 5) | none (same) | – |
| 5 | final latency | time from the end of the user's turn to the turn's final transcript (MPS unless noted) | ms, p50 / p95 | lower | **25 / 48**, 49 / 71 (`--final-chunk-ms 1120`) | 1804 / 3610 (Whisper small beam 5, CPU 2 threads) | same as LiveKit | 232 / 639 (Parakeet-TDT v3) |
| **Turn detection** | | | | | | | | |
| 6 | end-of-turn accuracy, assistant clips | smart-turn v3.2 test (399 clips): turn ends on complete clips, none within 2.5 s on clips cut mid-utterance | % | higher | 96.0 (held-out pick, candidate heads v0.5, not the default) / **97.7** ⚑ (`assistant`) | 85.2 | 75.9 | 49.4 (Parakeet-Realtime-EOU) |
| 7 | false fires, assistant clips | turn ends on an incomplete clip within 2.5 s of its audible end | % of incomplete clips | lower | 2.7 / **0.9** ⚑ | 22.3 | 29.0 | 89.7 |
| 8 | end-of-turn latency, assistant clips | audible end of speech to the turn end, compute included | ms, p50 / p95 | lower | 381 / 929, 351 / 763 ⚑ | 547 / 3016 | **211 / 242** | 462 / 1210 |
| 9 | end-of-turn latency, meetings (AMI test, 200 turns) | the same, default `balanced` preset with the voice print | ms, p50 / p95 | lower | 1527 / 4439, 1498 / 4075 | **745** / 4640 | 752 / 4419 | 1251 / 4928 |
| 10 | false interruptions / missed, meetings (AMI test) | false interruption = a turn end inside the user's turn; missed = no turn end before the next speaker or within 6 s | % of turns | lower | 15.5 / 36.0, 10.0 / 33.5 (no print: 6.5 / 74.0, 5.5 / 73.5) | 13.0 / 72.0 | 39.5 / 53.0 | **4.5** / 86.0 |
| 11 | end-of-turn latency, false interruptions / missed, two-party calls (**not a test split**) | 109 user turn ends, user channel (16 TurnBench dev + 16 oto conversations), `balanced` | ms p50 / p95; % of turns | lower | 955 / 1918 ms, 20.2 / 7.3 %; 729 / 2034 ms, 19.3 / 10.1 % ⚑ | 567 / 3127 ms, 26.6 / 22.9 % | 237 / 3217 ms, 35.8 / 24.8 % | 1277 / 5362 ms, 7.3 / 30.3 % |
| **VAD** | | | | | | | | |
| 12 | VAD ROC-AUC, AMI test | how well the speech score separates speech from non-speech over every threshold (1.0 = perfect) | 0-1 | higher | 0.966 / **0.967** | 0.961 (Silero v5) | 0.961 (Silero v5) | **0.967** (MarbleNet v2, offline here) |
| 13 | VAD F1, AMI test | balance of missed and false speech frames at the 0.5 threshold | 0-1 | higher | **0.959** / 0.957 | 0.901 | 0.901 | 0.941 |
| **Speaker** | | | | | | | | |
| 14 | target-speaker WER (tWER), AMI test | WER of the user's words only, 5 s voice print | % | lower | 51.5 / **47.1** | no speaker tracking | no speaker tracking | 64.4 (Nemotron-3-Diarization bound to the same print, 0.6B words) |
| 15 | target miss + false-alarm rate, AMI test | (missed user frames + other people's frames given to the user) / user frames, 80 ms, no collar | % | lower | 37.3 / **35.8** | – | – | 62.9 (Nemotron-3 + print) |
| 16 | speaker EER, AMI test | within-meeting equal error rate of the speaker embedding | % | lower | 5.0 / 3.8 | – | – | **1.9** (TitaNet-L) |
| **Language** | | | | | | | | |
| 17 | LID accuracy, FLEURS-17 test, 2 s from speech onset | share of clips whose language is right (17 languages) | % | higher | 92.4 / 92.7 ⚑ | – | – | 95.1 (AmberNet); Whisper large-v3 **95.9** |
| **Efficiency** | | | | | | | | |
| 18 | compute per 160 ms chunk | processing time for each 160 ms of audio, whole single-mode engine (ASR, speech detector, VAD, turn, TS-VAD, LID) | ms, p50 / p95 | lower | MPS 28.6 / 30.7, 42.9 / 49.0; CPU 2 threads 30.4 / 33.0, 97.1 / 105.1 | – (Whisper runs per segment, not per chunk) | – | – |
| 19 | real-time streams per device | how many live sessions one process keeps up with | streams | higher | MPS 5 / 3; CPU 2 threads 4 / 1 | not measured | not measured | – |
| 20 | peak memory | peak RSS of the process | GB | lower | 1.2 / 4.9 (+0.5 / +3.3 GB MPS) | not measured here | not measured here | – |

Rows 6-8: the headline false-fire window is 2.5 s, the largest of 2.0-3.5 s more than 0.25 s from every system's
fallback timer (FINAL_COMPARE.md "Which false-fire window W"). The 115M's shipped `assistant` rule was tuned on these
clips, so its row is in FINAL_COMPARE.md Appendix A and not here; the 115M cell is the held-out re-pick. Row 10: our
engine knows the user's voice print, the baselines do not; the "no print" figures show what the print is worth.

### Where we lose

- **Words on read speech and ICSI meetings:** Parakeet-TDT v3 (offline) is better than both cores on LibriSpeech and
  on ICSI test (7.6 vs 10.3 % for the 0.6B); Whisper large-v3 is best on FLEURS en. The 115M makes about twice the
  0.6B's errors on meetings (row 3).
- **Speed of the meeting turn end:** Pipecat and LiveKit answer about twice as fast on AMI test (row 9), at the
  price of interrupting more (Pipecat) or missing most turns (LiveKit).
- **VAD on AMI:** MarbleNet ties our heads on AUC (row 12) and misses less speech at a fixed 7.5 % false-alarm rate
  (9.6 vs 12.5 %); pyannote's segmentation model (offline, reads 10 s ahead) is best on F1 (0.977).
- **Speaker embeddings:** TitaNet-L and WeSpeaker are better voice matchers (row 16); our tracker still wins on tWER.
- **Language ID:** Whisper large-v3 and AmberNet are ahead (row 17).

## Definitions and how each was measured

Full protocols, CIs and every baseline are in [FINAL_COMPARE.md](FINAL_COMPARE.md); the summary:

**WER (rows 1-3).** Word error rate = (substituted + deleted + inserted words) / reference words, with Whisper's
English text normalizer on both sides, as the Open ASR Leaderboard does. audioforge's transcript is the served
streaming model's (greedy, 160 ms chunks). LibriSpeech: 300 utterances drawn at random (seed 0) from each test split.
AMI: 200 single-speaker segments (1-15 s) of the test meetings IS1009b, ES2004b, TS3003b, EN2002a; CIs resample the
4 meetings. Whisper small runs through faster-whisper at beam 5, its front end's default.

**Partial latency (row 4).** `scripts/research/stt_latency.py`. The served session was fed in 160 ms blocks on MPS,
as a real-time client sends them, on 12 two-minute windows of the four AMI test meetings (headset mix). For each
reference word the final transcript gets right (jiwer alignment), latency = when the word, in its final spelling,
first appears in a `partial` minus the word's end time in AMI's manual word timings. "When it appears" is the audio
clock after the block that produced it plus the measured compute of that block. Network time is not included. Words
that appear before their annotated end (AMI's boundaries are approximate) are kept as negative values.

**Final latency (row 5).** Time from the labelled end of each user turn to the turn's final transcript, on the user
channel of the 16 TurnBench clips of the live set (56 turns). Offline systems get the whole turn's audio at the turn
end, as an end-of-utterance STT does in Pipecat / LiveKit; batch 1, warm models.

**Turn detection (rows 6-11).** One scorer for every system. Latency is from the audible end of speech to the turn
end, plus the measured compute. Accuracy (row 6) = share of clips handled right: a turn end on a complete clip, none
within W = 2.5 s on an incomplete one. False fire (row 7) = a turn end on an incomplete clip within W of its audible
end. False interruption (rows 10-11) = a turn end inside the user's turn, in its speech or in a pause within it;
missed = no turn end before the next speaker or within 6 s. Rates are % of turns (clips for rows 6-7).

**VAD (rows 12-13).** 64 windows of 20 s from the AMI test meetings, 80 ms frames, label = anyone speaking; only
frames inside the audio are scored. Baselines are max-pooled onto the 80 ms grid. Our speech heads were trained on
AMI and ICSI train meetings (in domain for them, not for Silero or MarbleNet). The ICSI column, where the headline
uses heads retrained with the ICSI test speakers unseen in training, is in FINAL_COMPARE.md "Speech detection".

**Speaker (rows 14-16).** Each target is the main speaker of an eot-bench v2 window of the AMI test meetings (941
windows), with a 5 s voice print cut from elsewhere in the same meeting. tWER keeps the streaming words where the
system's target mask is on. The miss + false-alarm rate has no speaker-confusion term, so it is not the standard
DER (the diarizers' standard DER is in FINAL_COMPARE.md: Nemotron-3 25.4 %, pyannote 3.1 17.2 % on AMI test). The
diarizer is bound to the same print by the better of two binders. AMI is in Nemotron-3's training data. EER: 200
single-speaker segments, 6941 within-meeting pairs.

**LID (row 17).** FLEURS-17 test, 2550 clips; every system's posterior restricted to the 17 languages; the first
2 s from speech onset.

**Efficiency (rows 18-20).** The full single-mode engine on the bundled 16 s call, best of 3. Streams = interleaved
sessions on 40 s AMI windows, real time while the p95 of the summed block compute stays under 160 ms.

**Voice-to-voice latency** = end-of-turn latency + LLM time to first token + TTS time to first audio. We do not
measure LLM or TTS time; it depends on the models you choose.

## Why the old "missed within 3 s" number was dropped

History: this section and the next are about the 2026-09-29 live run (69 sessions through Pipecat / LiveKit, earlier
turn rule `hybrid_dyn 2000,960`), not a test split.

Earlier documents and images headlined "37.7 % of user turns not answered within 3 s" for audioforge, against 41.3 %
(LiveKit) and 58.7 % (Pipecat). That number pools 69 sessions, and 32 of them are **mono mixes**: the user and the
recorded other party in one channel. In a mono mix the other party starts answering right after the user stops, in
the same audio. No system hears silence, so none of them can fire in time. That is an artifact of the test, not a
property of the product. A real agent hears the user's channel (its own voice is removed by echo cancellation).

The same metric per condition (`runs/metrics.json` `live`), as % of user turn ends with no response within 3 s:

| condition | sessions / turn ends | audioforge | LiveKit default | Pipecat default | audioforge room mode (Pipecat) |
|---|---|---|---|---|---|
| user's own channel (realistic) | 32 / 109 | 17.4 | 24.8 | 48.6 | 12.8 |
| mono mix (test artifact) | 32 / 109 | 58.7 | 57.8 | 68.8 | 55.0 |
| AMI meeting windows (count of 5, not %) | 5 / 5 | 1 of 5 | 2 of 5 | 3 of 5 | 2 of 5 |
| all 69 pooled (the old headline) | 69 / 223 | 37.7 | 41.3 | 58.7 | 34.1 |

The scorecard therefore reports the user-channel sessions and replaces "missed within 3 s" with end-of-turn latency
and the share of turns missed (rows 8-11 today). On the mono mixes the response rate is 58.7 % (audioforge), 59.6 % (LiveKit)
and 57.8 % (Pipecat): no system copes, and they are equal. On the user's channel audioforge's room mode (Nemotron-3
diarizer, 1 s timeout) is faster (1272 / 1631 ms) and answers slightly more (89.9 %), but interrupts more (26.6 % of
turns).

## Names used in earlier documents

| old name | standard name here | note |
|---|---|---|
| dead air | end-of-turn latency | the same measurement; now reported on the user channel, p50 / p95 |
| cut-ins, "talks over you", interruptions per call | false interruptions | now % of user turns (per minute given too); "per call" was per 20-60 s clip |
| unanswered / missed within 3 s | response rate (+ late responses) | the pooled number mixed in the mono artifact; see above |
| first words / first text (1081 ms) | removed | it counted from the start of speech, so it included the time it takes to say the first word. Partial latency (row 4) is the standard measure |
| finds your voice / tracking F1 / target DER | target-speaker WER, target miss + false-alarm rate | the latter has no speaker-confusion term, so it is not called DER; F1 kept in FINAL_COMPARE.md |
| hears speech / speech missed at 7.5 % FA | VAD ROC-AUC, F1 | the miss rate at a fixed false-alarm rate is kept as a detail |
| missed turn ends at ≤ 5 % false cut-offs (AMI 34 %, ICSI 19 %) | appendix | an offline research metric on meeting windows with a stored print; not a standard voice-agent number |
| RTF 0.33 | compute per chunk, streams per device, CPU per audio second | RTF is wall time; the live run measured the default stacks in CPU seconds per audio second (0.34 audioforge, 0.28 LiveKit, 0.43 Pipecat) |

## Appendix: research metrics kept out of the scorecard

- **Missed turn ends in meetings, earlier offline metric** (eot-bench v2 on AMI **dev** and ICSI, each system at ≤ 5 %
  of turns cut off early, 6 s horizon; a selection-era research number, not a test result, superseded by rows 9-10):
  audioforge with a stored print 34.2 % (AMI dev) / 18.7 % (ICSI). The detectors without a print: Parakeet-EOU
  70.1 %, smart-turn + Silero 72.6 %, LiveKit turn detector + timeout 74.2 % (AMI); Silero timeout 84.4 % (ICSI)
  (`runs/single_model.json`, `runs/baselines_turn*.json`). audioforge had the print and the others did not.
- **Language ID**, full clip, FLEURS-17 test: 98.2 / 98.6 % (115M / 0.6B) against AmberNet's 99.5 % and Whisper
  large-v3's 99.5 % (FINAL_COMPARE.md).
- **Image numbers:** `demo/images/redesign/numbers_final.json`, built from `runs/final_compare.json`.
- **Voice print length**: research/SINGLE_MODEL.md A2.

