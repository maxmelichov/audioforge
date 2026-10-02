# Final comparison: both audioforge cores against the best open models, one pass (2026-10-01)

**Status: complete.** Both cores are the shipped builds: the 115M with served heads v0.3 and the English 0.6B
(nemotron-speech-streaming-en-0.6b) with heads v0.2. The run took 13:05-20:35 on 2026-10-01, about 4.5 h of compute
plus waiting for other jobs. Every number is in `runs/final_compare.json`. The script is
`scripts/research/final_compare.py`. The image numbers are in `demo/images/redesign/numbers_final.json`, built by
`export_final.py`.

## In plain words

- **Words.** The 0.6B core makes about as few mistakes as the best open models. On meetings and live calls it ties
  NVIDIA's offline Parakeet-TDT v3. It beats Whisper large-v3 and large-v3-turbo on AMI meetings and on the user's own
  channel of live calls. It is a little worse than Parakeet-TDT on ICSI meetings. It does this while streaming: a word
  shows up about 0.3 s after it is said. The Whisper models and Parakeet-TDT only transcribe after the speaker stops.
  The small 115M core makes about twice as many mistakes on meetings as all of them.
- **Speech detection.** Both cores beat the VADs that ship in voice agents today (Silero, TEN VAD, NVIDIA MarbleNet) on
  AMI meetings and, since the 2026-10-01 speech head (research/FIXALL.md), on ICSI meetings too (0.947 / 0.951 vs TEN
  VAD 0.935). pyannote's segmentation model is best on AMI. It reads 10 s of audio ahead, so it cannot run live.
- **Turn taking.** On speech aimed at an assistant, our `assistant` preset is right on 93 % of clips with either core.
  It answers in 0.30 s (115M) or 0.38 s (0.6B). Pipecat's smart-turn stack is right on 70 % and LiveKit's on 73 %. NVIDIA's
  Parakeet-EOU is right on 48 %. On human two-party calls, our default preset interrupts on 20 % of turns and misses
  7 %. Pipecat interrupts on 36 % and misses 25 %; LiveKit interrupts on 27 % and misses 23 %. Pipecat answers
  faster (0.24 s, against 0.96 s for the 115M and 0.73 s for the 0.6B).
- **Following your voice.** With a 5 s voice print, our tracker keeps your words and drops other people's much better
  than an open diarizer bound to the same print. On ICSI meetings, your-words-only WER is 32 % (0.6B) and 37 % (115M).
  It is 65 % for NVIDIA Nemotron-3 diarization and 75 % for pyannote 3.1. As plain speaker embedders, though, TitaNet-L
  and pyannote's WeSpeaker beat our speaker heads.
- **Cost.** The 115M runs a whole session in 28 ms per 160 ms of audio on the Mac GPU, 4 streams live, 1.2 GB. The
  0.6B needs 44 ms, 3 streams, 4.8 GB.
- **Language.** Whisper's own language detection is better than our small heads: 96 % vs 91 % (115M) and 88 % (0.6B)
  after 2 s of speech.

## Words: WER %, Whisper English normalizer, [95 % CI]

Same audio and the same references for every system. The bootstrap is over utterances; on the live sessions it is
over clips. Lower is better.

| system | LibriSpeech test-clean (200) | AMI dev (200 segments) | ICSI dev (200 segments) | live calls, every word (32 sessions) | live calls, user channel (16) | mode |
|---|---|---|---|---|---|---|
| **audioforge 0.6B** | 2.07 [1.54, 2.71] | **10.39** [8.61, 12.29] | 13.62 [11.62, 15.49] | 11.55 [8.96, 14.93] | **5.70** [4.39, 7.18] | streaming, 160 ms |
| audioforge 115M | 2.27 [1.75, 2.85] | 20.63 [17.94, 23.48] | 26.21 [23.40, 29.17] | 20.34 [16.67, 25.08] | 15.35 [12.56, 19.18] | streaming, 160 ms |
| NVIDIA Parakeet-TDT 0.6B v3 | 1.88 [1.36, 2.50] | **9.50** [7.88, 11.29] | **10.37** [8.79, 11.97] | **11.04** [8.30, 14.67] | 7.68 [5.23, 11.67] | offline |
| Whisper large-v3 | 1.41 [0.95, 1.97] | 12.36 [10.52, 14.26] | 12.79 [11.11, 14.76] | 12.69 [9.79, 15.93] | 8.45 [6.00, 11.25] | offline |
| Whisper large-v3-turbo | **1.28** [0.89, 1.80] | 12.33 [10.33, 14.20] | 13.86 [12.13, 15.88] | 12.91 [10.02, 16.33] | 8.38 [6.63, 10.34] | offline |
| Whisper small (LiveKit default STT) | 2.29 [1.69, 2.96] | 14.49 [12.36, 16.88] | 18.17 [15.99, 20.69] | 14.67 [11.60, 18.18] | 10.14 [8.25, 12.25] | offline |
| Nemotron 3.5 ASR 0.6B (multilingual), en-US prompt | 2.59 [1.94, 3.30] | 14.78 [12.85, 17.20] | 20.27 [17.80, 23.00] | 16.99 * | 11.17 * | streaming, 160 ms |
| Deepgram Nova-3 | not measured (paid API) | | | | | |

\* Repo normaliser, from `runs/core_3p5.json`. The other cells of that row also come from that file: the same 200-item
sets and sessions, reused.

Paired difference against the 0.6B, percentage points [95 % CI] (positive = the baseline makes more errors):

| baseline − 0.6B | LibriSpeech | AMI | ICSI | live, every word |
|---|---|---|---|---|
| Parakeet-TDT v3 | −0.19 [−0.62, +0.28] | −0.89 [−2.38, +0.40] | **−3.25 [−4.46, −2.05]** | −0.51 [−1.50, +0.69] |
| Whisper large-v3 | **−0.66 [−1.13, −0.25]** | **+1.97 [+0.35, +3.55]** | −0.83 [−2.40, +0.76] | +1.14 [−0.46, +2.89] |
| Whisper large-v3-turbo | **−0.79 [−1.28, −0.32]** | +1.94 [−0.00, +3.74] | +0.24 [−1.38, +2.16] | +1.36 [−0.12, +2.82] |
| Whisper small | +0.22 [−0.29, +0.70] | **+4.10 [+2.33, +5.89]** | **+4.55 [+2.92, +6.38]** | **+3.12 [+1.25, +4.92]** |
| audioforge 115M | +0.20 [−0.25, +0.57] | **+10.24 [+8.29, +12.34]** | **+12.59 [+10.37, +15.01]** | **+8.79 [+7.13, +10.93]** |

With the repo's own normaliser (`teachers.normalize_text`, used in earlier tables), Whisper scores much worse on the
meeting sets: AMI 19.1 / 19.4 / 21.2 % for large-v3 / turbo / small, against 11.2 % (0.6B) and 9.7 % (TDT). This is
because it counts spelling differences that Whisper's normaliser removes. The table above uses Whisper's normaliser,
which is the fair choice when Whisper is in the comparison. Both are in the json.

**Streaming word latency (ours only).** This is the time from the end of a word to the word first appearing in the
live transcript. It was measured through the served engine on MPS, on 12 two-minute AMI test windows
(`stt_latency.py` protocol):
- 115M: p50 273 ms [250, 290], p95 570 ms.
- 0.6B: p50 290 ms [270, 309], p95 590 ms.

The 115M's published 441 ms was measured on CPU before the chunk-trigger fix (d832fca, −70 ms). Its block compute was
32 ms there, against 29 ms on MPS here. The baselines above are offline and have no streaming latency. Whisper small
in the LiveKit default waits for the end of the utterance.

## Speech detection

AMI dev (64 × 20 s) and ICSI dev (64 × 20 s), 80 ms frames, label = anyone speaking. Baselines are max-pooled onto the
80 ms grid. The bootstrap is over windows. "Miss" = speech frames missed at the threshold where 7.5 % of non-speech
frames are false alarms. "Onset lag" = labelled speech onset (after ≥ 400 ms of silence) to the first frame > 0.5, on
the frame grid. It does not include each model's own look-ahead.

| system | AMI F1 @0.5 | AMI ROC-AUC | AMI miss @7.5 % FA | ICSI F1 | ICSI AUC | ICSI miss | onset lag p50 / p90 (AMI) | live-capable |
|---|---|---|---|---|---|---|---|---|
| audioforge 115M speech head (v0.4) | 0.951 [0.939, 0.960] | 0.970 [0.960, 0.978] | 12.1 % [8.4, 14.8] | **0.947** [0.942, 0.952] | 0.964 [0.959, 0.969] | 10.7 % | 0 / 160 ms | yes (80 ms right context) |
| audioforge 0.6B speech head (v0.3) | 0.950 [0.938, 0.959] | 0.969 [0.958, 0.979] | 12.5 % [8.0, 16.9] | **0.951** [0.946, 0.955] | **0.966** [0.962, 0.970] | **10.2 %** | 80 / 160 ms | yes (80 ms right context) |
| *before (FINAL_COMPARE pass): 115M heads v0.3* | *0.951* | *0.972* | *10.6 %* | *0.898* | *0.931* | *19.8 %* | *80 / 160 ms* | |
| *before: 0.6B heads v0.2* | *0.951* | *0.972* | *10.7 %* | *0.906* | *0.925* | *22.4 %* | *80 / 160 ms* | |
| Silero VAD v5 (Pipecat / LiveKit) | 0.915 [0.900, 0.926] | 0.956 [0.941, 0.969] | 13.8 % [8.9, 21.8] | 0.929 [0.922, 0.935] | 0.934 [0.925, 0.942] | 19.9 % | 80 / 160 ms | yes |
| TEN VAD | 0.925 [0.912, 0.935] | 0.948 [0.934, 0.960] | 13.6 % [11.1, 19.2] | 0.935 [0.929, 0.942] | 0.952 [0.945, 0.957] | 14.0 % | 0 / 80 ms | yes |
| NVIDIA MarbleNet v2 frame VAD | 0.937 [0.919, 0.951] | 0.959 [0.946, 0.971] | 12.4 % [8.5, 16.2] | 0.929 [0.923, 0.935] | 0.940 [0.932, 0.948] | 19.8 % | 0 / 0 ms | offline here (whole-window conv) |
| pyannote segmentation-3.0 | **0.963** [0.952, 0.972] | **0.984** [0.978, 0.988] | **4.6 %** [3.2, 7.3] | 0.899 [0.891, 0.907] | 0.904 [0.876, 0.926] | 28.6 % | 0 / 0 ms | no (10 s windows, sees the future) |

Updated 2026-10-01 (research/FIXALL.md step 1): the rows are the new `speech` head of each core (served heads 115M
v0.4 / 0.6B v0.3), a stateless head trained on AMI + 10 h of ICSI train meetings + oto user channels. ICSI dev
meetings (Bmr021, Bns001) were never in training or selection, but ICSI is no longer an unseen corpus for us. Paired
against TEN VAD on ICSI: F1 +0.009 to +0.015 (115M), +0.011 to +0.019 (0.6B). Against the old heads on AMI: F1 −0.0006
[−0.0033, +0.0019] (115M) and −0.0007 [−0.0048, +0.0035] (0.6B), AUC −0.002 / −0.003, miss +1.5 / +1.8 points: within
the CIs, but a small step back. Paired, ours vs Silero on AMI: F1 +0.027 to +0.047, AUC +0.006 to +0.024 (95 % CI,
both cores). The turn rules keep reading the old `vad` head, so the turn rows below are unchanged.

## Turn taking

The data:
- **Assistant clips**: smart-turn v3.2 test, 399 clips (175 complete, 224 cut mid-utterance), each padded with 3 s of
  silence.
- **Calls**: 32 two-party user channels, 109 turn ends.
- **AMI**: 200 meeting turns.

Latency is from the audible end of speech to the turn end, plus the measured compute. "False fire" = a turn end within
3 s of an incomplete clip's cut. "Interrupt" = a turn end inside the user's turn. "Missed" = no turn end before the
next speaker or within 6 s. The bootstrap is over clips / sessions (1000 resamples).

| system | assistant: accuracy | assistant: p50 / p95 ms | assistant: false fire on incomplete | assistant: missed | calls: p50 / p95 ms | calls: interrupt | calls: missed | AMI: p50 ms | AMI: interrupt | AMI: missed |
|---|---|---|---|---|---|---|---|---|---|---|
| **audioforge 115M `assistant`** | **92.7 %** [90.2, 95.0] | 299 / 710 | **5.4 %** | 5.1 % | 3232 / 4801 | 5.5 % | 33.9 % | 1647 | 7.0 % | 52.5 % |
| audioforge 115M `balanced` (default) | 43.9 % | 1226 / 1587 | 100 % | 0.0 % | 955 / 1918 | 20.2 % [13.1, 29.2] | **7.3 %** [3.5, 12.0] | 1326 | 10.5 % | **33.5 %** |
| audioforge 115M `fast` | 41.6 % | 461 / 698 | 100 % | 4.0 % | 547 / 1861 | 24.8 % | 5.5 % | 1248 | 11.5 % | 34.0 % |
| **audioforge 0.6B `assistant`** | **93.0 %** [90.5, 95.5] | 379 / 702 | **4.5 %** | 9.7 % | 3749 / 4755 | 1.8 % | 43.1 % | 1981 | 1.0 % | 64.5 % |
| audioforge 0.6B `balanced` (default) | 42.6 % | 634 / 794 | 100 % | 2.9 % | 730 / 2036 | 19.3 % [12.1, 26.4] | 10.1 % [4.9, 16.0] | 1180 | 11.0 % | 33.0 % |
| audioforge 0.6B `fast` | 40.1 % | 409 / 602 | 100 % | 8.0 % | 494 / 1977 | 26.6 % | 11.0 % | 1102 | 17.0 % | **27.0 %** |
| Pipecat smart-turn v3.2 + Silero (defaults) | 69.7 % [65.2, 74.2] | **211** / 242 | 40.2 % | 13.1 % | **237** / 3217 | 35.8 % [25.7, 46.6] | 24.8 % [16.1, 33.6] | 385 | 27.5 % | 44.5 % |
| LiveKit Agents 1.8 EnglishModel + Silero | 72.7 % [67.9, 76.9] | 547 / 3016 | 44.6 % | 4.6 % | 567 / 3127 | 26.6 % [20.2, 34.0] | 22.9 % [15.1, 30.2] | 1890 | 12.5 % | 67.5 % |
| NVIDIA Parakeet-Realtime-EOU 120M | 48.4 % [43.6, 53.1] | 462 / 1210 | 91.5 % | 0.6 % | 1277 / 5362 | 7.3 % | 30.3 % | 2251 | **1.5 %** | 77.5 % |
| *smart-turn v3.2 classifier alone (upper bound: one call per whole clip, no timing)* | *97.0 %* [95.2, 98.5] | – | *2.7 %* | *3.4 %* | – | – | – | – | – | – |

The presets trade one domain for another:
- `assistant` waits up to 3.4 s for a classifier "complete". That suits assistant speech, but it misses a third of
  human call turns.
- `balanced` (the default) is the calls preset.

No single setting of any system wins every column. The CIs of every cell are in the json.

## Speaker tracking ("your words only")

The windows are the eot-bench v2 windows: AMI dev, 974 windows; ICSI held out, 1312 windows and 1286 target units.
- The target is the main speaker of each window, with a 5 s voice print cut from elsewhere in the same meeting.
- **tWER** = WER of the target's words only. Streaming words are kept where the system's target mask is on, with the
  same keep rule for every arm.
- **Target DER** = (missed target frames + false target frames) / target frames, at 80 ms, with no collar.
- The open diarizers' columns are bound to the same print by `tsvad.vp_follow`. The better of two binders (served
  speaker head, TitaNet-L) is shown.

| system | AMI tWER % [meeting CI] | ICSI tWER % [meeting CI] | AMI target DER % | ICSI target DER % | AMI tracking F1 | ICSI tracking F1 |
|---|---|---|---|---|---|---|
| audioforge 115M (115M words) | **62.1** [47.7, 75.7] | 37.1 [33.7, 41.3] | **50.9** | 23.0 | 0.743 [0.726, 0.759] | 0.882 [0.875, 0.889] |
| audioforge 0.6B (0.6B words) | 63.2 [47.7, 78.1] | **31.7** [28.5, 34.8] | 51.8 | **21.1** | **0.760** | **0.894** |
| Nemotron-3-Diarization + print, 115M words | 74.7 [59.5, 84.8] | 65.1 [56.9, 73.3] | 69.5 | 74.9 | 0.648 | 0.698 |
| Nemotron-3-Diarization + print, 0.6B words | 71.6 [55.5, 81.8] | 60.0 [49.6, 69.5] | (same track) | | | |
| pyannote speaker-diarization-3.1 + print, 115M words | 85.3 [69.6, 98.3] | 74.8 [65.1, 84.1] | 77.8 | 89.2 | 0.690 | 0.670 |
| pyannote speaker-diarization-3.1 + print, 0.6B words | 87.0 [69.8, 97.0] | 70.0 [58.4, 80.4] | (same track) | | | |
| no filter (115M / 0.6B words) | 106.5 / 117.8 | 100.8 / 99.0 | – | – | – | – |
| oracle filter (115M / 0.6B words) | 47.4 / 34.3 | 31.3 / 24.6 | – | – | – | – |

- The audioforge, Nemotron-3-with-115M-words, no-filter and oracle rows are reused unchanged from `runs/tswer.json`,
  `runs/core_0p6b.json` (tswer, tsvad_frame) and `runs/improve_115m.json`. They come from the same windows, words and
  keep rule.
- The pyannote rows and the Nemotron-3-with-0.6B-words row were measured in this pass. Two consistency checks: this
  pass recomputes the Nemotron-3 115M-word arm as exactly 80.89 / 74.66 (AMI) and 66.77 / 65.09 (ICSI), and the 115M
  TS-VAD frame row as 0.7428 / 50.93 %.
- pyannote runs offline on the whole 20 s window, which helps it.
- pyannote's DER is high because its column, even with the oracle column picked, keeps much non-target speech: oracle
  column DER 69 % (AMI) / 83 % (ICSI).
- Nemotron-3 has AMI in its training data.
- The 32 live sessions: our tWER is in `runs/tswer_live.json`. No diarizer was run on them (see "Not measured").

**Within-meeting speaker EER** (`spk_head.py` dev segments, n = 200 per corpus, segment bootstrap):

| embedder | AMI dev EER % | ICSI dev EER % |
|---|---|---|
| audioforge 115M speaker head (block 4) | 19.8 [15.3, 23.9] | 7.0 [4.3, 10.0] |
| audioforge 0.6B speaker head (block 5) | 13.6 [9.8, 17.3] | 3.6 [1.9, 5.4] |
| NVIDIA TitaNet-L (the teacher) | 12.0 [8.1, 16.2] | 2.1 [1.1, 3.0] |
| pyannote WeSpeaker ResNet34 (in pyannote 3.1) | **11.9** [8.4, 15.8] | **1.0** [0.5, 1.6] |

The dedicated embedders are better voice matchers than our heads. Our tracker still wins on tWER and DER, because the
TS-VAD head is trained to track the print frame by frame, not only to embed.

## Language ID: FLEURS-17 test (2550 clips), accuracy % [95 % CI]

| system | 2 s from speech onset | full clip |
|---|---|---|
| Whisper large-v3 | **95.9** [95.2, 96.7] | 99.5 [99.2, 99.7] |
| Whisper large-v3-turbo | 95.2 [94.3, 96.0] | 99.3 [99.0, 99.7] |
| NVIDIA AmberNet (the teacher; reused, `data/lid/preds/ambernet`) | 95.1 [94.3, 95.9] | **99.5** [99.3, 99.8] |
| Whisper small | 90.6 [89.4, 91.7] | 99.1 [98.7, 99.4] |
| audioforge 115M head (reused, `lid_distill_vadgated`) | 91.0 [89.9, 92.0] | 97.8 [97.2, 98.4] |
| audioforge 0.6B head | 87.6 [86.3, 88.8] | 95.5 [94.6, 96.2] |

Every system's posterior is restricted to the 17 test languages (the `lid.py` protocol). Both of our heads are
VAD-gated, as served. The 0.6B head has less training data than the 115M head: FLEURS `trainx` was not cached for it.

## Cost on this Mac (Apple M5)

Full single-mode engine: ASR, VAD, turn, TS-VAD and LID. The bundled 16 s call, best of 3. Streams = interleaved
sessions on 40 s AMI windows, real time while p95 of the summed block compute stays under 160 ms (`mps_115m.py`
protocols).

| core | ms per 160 ms chunk, MPS (p50 / p95) | CPU 2 threads | real-time streams, MPS | real-time streams, CPU | peak RSS |
|---|---|---|---|---|---|
| audioforge 115M | 27.7 / 29.5 | 31.0 / 33.1 | 4 | 4 | 1.2 GB (+0.5 GB MPS) |
| audioforge 0.6B | 43.7 / 49.3 | 99.4 / 107.1 | 3 | 1 | 4.8 GB (+3.3 GB MPS) |

**Model sizes** (parameters counted from the weight files):

| model | params |
|---|---|
| audioforge 115M (whole served model) | 122.1 M |
| audioforge 0.6B (whole served model) | 622.5 M |
| Whisper small / large-v3-turbo / large-v3 | 241.7 M / 808.9 M / 1543.5 M |
| Parakeet-TDT 0.6B v3 | 627.0 M |
| Parakeet-Realtime-EOU | 114.9 M |
| smart-turn v3.2 (ONNX) | 8.0 M |
| LiveKit turn detector (largest ONNX in the cache) | 65.5 M |
| Silero VAD v5 (TorchScript, its 8 kHz and 16 kHz branches together) | 0.46 M |
| MarbleNet v2 frame VAD | 0.09 M |
| pyannote segmentation-3.0 / WeSpeaker ResNet34 | 1.47 M / 6.63 M |
| TEN VAD | parameter count not exposed (compiled library, 0.7 MB) |
| Nemotron-3-Diarization | 99.2 M |
| TitaNet-L / AmberNet | 25.3 M / 28.9 M |

The baselines' cost per stream was not measured. Their RTFx in this pass is in the json, and it mixes devices: Whisper
large on MPS, the rest on CPU.

## How each system was run (versions, licences)

- **audioforge 115M**: `runs/stage1_served_v3.afm` (served heads v0.3), NVIDIA 115M streaming FastConformer, CC-BY-4.0.
  - ASR: masked [70,1] forward on CPU (equal to 160 ms streaming).
  - VAD: the served head.
  - Turn: the served `--mode single` engine on MPS. Each session is dumped once, and each preset is applied with the
    served policy twin (`core_0p6b_turn.served_rules` / `ev_score`). The twin reproduces the served turn_ends.
- **audioforge 0.6B**: `served_0p6b_v0.2.afm` (`assets/served_heads_0p6b_v0.2.pt`; the same tensors as v0.1, with
  re-picked preset constants) on nvidia/nemotron-speech-streaming-en-0.6b, NVIDIA Open Model License. Same harnesses
  as the 115M.
- **Whisper small**: faster-whisper 1.2.1 int8, CPU 2 threads, beam 1, English. MIT.
- **Whisper large-v3 / large-v3-turbo**: transformers 5.17, fp16 on MPS, greedy, English. Clips over 30 s use
  sequential long-form decoding. MIT. For LID, the language-token posterior after `<|startoftranscript|>`.
- **Parakeet-TDT 0.6B v3**: imported into our PyTorch modules (`nemo_import`), offline, greedy TDT, CPU. CC-BY-4.0.
- **Nemotron 3.5 ASR 0.6B**: from `runs/core_3p5.json` (en-US prompt), NVIDIA Open Model License.
- **Silero VAD v5.1.2**: TorchScript, 32 ms chunks. MIT.
- **TEN VAD 1.0.6.8**: pip `ten-vad`, 16 ms hop. Apache-2.0 with additional conditions.
- **MarbleNet v2**: `Frame_VAD_Multilingual_MarbleNet_v2.0`, imported. NVIDIA Open Model License.
- **pyannote segmentation-3.0**: pyannote.audio 4.0.7, speech = 1 − P(no speaker), 10 s windows. MIT (gated, already
  accepted).
- **Pipecat smart-turn v3.2 + Silero**: Pipecat 1.12's own `LocalSmartTurnAnalyzerV3` replayed on a simulated clock
  (defaults: confidence 0.7, 0.2 s stop, 3 s fallback). BSD-2-Clause. The per-session decisions are reused from
  `eot_latency.py` / `eot_assistant.py` (same audio and versions) and rescored here with CIs. Pipecat's STT wait is not
  modelled, which flatters Pipecat.
- **LiveKit Agents 1.8 EnglishModel + Silero**: the plugin's own runner; min / max delay 0.5 / 3.0 s. Its text input is
  our 115M streaming transcript. LiveKit Model License. Reused the same way.
- **Parakeet-Realtime-EOU 120M v1**: imported, streamed in 160 ms blocks (`StreamingSession`, att [70,1]) on MPS. A
  turn end is the block in which the model emits `<EOU>`.
  - After an `<EOU>` the streaming state is restarted. The model never emits a second `<EOU>` in a long session
    otherwise (checked on the calls).
  - NVIDIA Open Model License.
- **pyannote speaker-diarization-3.1**: pyannote.audio 4.0.7 on MPS, its default hyper-parameters, the whole 20 s
  window. MIT (gated, already accepted). The tWER and frame arms use the same code as the Nemotron-3 arm.
- **TitaNet-L / WeSpeaker** (EER rows): the imported TitaNet-L and pyannote's WeSpeaker ResNet34 `Inference(window="whole")`.
  CC-BY-4.0.
- **smart-turn classifier alone**: reused from `runs/smartturn_audit.json` (same 399 clips, same model file).
- **Nemotron-3-Diarization**: cached streaming tracks (`low_latency_032`). Its column is bound by the print with
  `tsvad.vp_follow`; the better of two binders (served speaker head, TitaNet-L) is shown. NVIDIA Open Model License.
- **AmberNet**: reused predictions on the same clips. CC-BY-4.0.
- **Paid APIs** (Deepgram Nova-3, AssemblyAI, Deepgram Flux turn detection, pyannoteAI): not measured.

## Not measured, and why

- **Paid APIs** (Deepgram Nova-3 for words, Deepgram Flux / AssemblyAI turn detection, pyannoteAI): no paid APIs in
  this pass.
- **Diarizer baselines on the 32 live sessions** (tWER): there are no cached Nemotron-3 tracks for these sessions, and
  no TitaNet print for the second binder. Ours: `runs/tswer_live.json`.
- **Baselines' per-stream cost**: not run. Whisper, Parakeet-TDT and the diarizers are offline per-utterance or
  per-window systems, so "ms per 160 ms chunk" does not apply to them. Their decode RTFx is in the json.
- **The baselines' streaming word latency**: they are offline. Whisper small in the LiveKit default shows no text until
  the end of the utterance.
- **Silero / TEN VAD inside the turn rows** are the Pipecat / LiveKit stacks' own. A Silero-only silence-timeout turn
  row was not re-run here; it is in `research/EOT_LATENCY.md`.
- **0.6B tracking F1 CI**: `runs/core_0p6b.json` stores no bootstrap for it.

## Reproduce

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_compare.py asr --system <S> --set <libri|ami|icsi|live> [--device mps]
    ... vad --system <S> | lid --system <S> --device mps | eou --which calls|asst --device mps
    ... eotdump --sys 115m|0p6b --which calls|asst --device mps | sttlat --sys 115m|0p6b --device mps
    ... spkeer --system <S> | pyatracks --device mps | pyatn | pyaframe | pyatwer
    ... cost --sys 115m|0p6b --sub engine|streams --device mps|cpu | params | report
    python3 demo/images/redesign/export_final.py

Scratch: `/Volumes/ExternalSSD/nvidia-audio-models/scratch/final_compare/`.
