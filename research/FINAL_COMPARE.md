# Final comparison: both audioforge cores against the best open models, on public test splits (2026-10-02)

**Status: test-split pass.** Both cores are the shipped builds:
- the 115M with served heads v0.4: v0.3 + the `speech` detector head, and LID head v2;
- the English 0.6B (nemotron-speech-streaming-en-0.6b) with heads v0.4: v0.3 (v0.2 + `speech`) + the `assistant`
  preset's own turn classifier and turn VAD (research/TURN_DATA.md), and LID head v2.

Every headline number below is on a standard public **test** split, with every baseline re-run on exactly the same
audio. Where no labelled public test split exists, the row is labelled and kept out of the headline tables.

The first pass (2026-10-01) used AMI / ICSI **dev** meetings. Its tables are kept in research/FIXALL.md
("Appendix: FINAL_COMPARE first pass on dev splits"); what moved is listed under "What changed with the test splits".
All numbers are in `runs/final_compare.json` (script `scripts/research/final_compare.py`). The image numbers are in
`demo/images/redesign/numbers_final.json` (built by `export_final.py`). The heads work is in research/FIXALL.md.

## In plain words

- **Words.** On AMI test meetings the 0.6B core makes the fewest mistakes of every system here: 7.9 %, a tie with
  NVIDIA's offline Parakeet-TDT v3 (8.3 %), and clearly ahead of Whisper large-v3 (10.7 %). It does this while
  streaming: a word shows up about 0.3 s after it is said.
  - On ICSI test meetings and on LibriSpeech, Parakeet-TDT is better: 7.6 vs 10.3 % on ICSI, 3.4 vs 5.7 % on
    test-other.
  - On the user's channel of live calls the 0.6B is best (5.7 % against 7.7 % for Parakeet-TDT).
  - The 115M makes about twice as many mistakes on meetings. Its optional `--beam 8` removes 1.5 of them per 100 words
    on AMI.
- **Speech detection.** On both test corpora our heads beat the VADs that ship in voice agents today (Silero, TEN VAD,
  NVIDIA MarbleNet).
  - ICSI test: 0.938 / 0.940 F1 against 0.922 for the best of them (Silero).
  - AMI test: 0.959 / 0.957 against 0.941 (MarbleNet).
  - pyannote's segmentation model is better on AMI (0.975), but it reads 10 s ahead, so it cannot run live. It is the
    worst on ICSI.
- **Turn taking.** On speech aimed at an assistant (smart-turn v3.2 test), our `assistant` preset is right on 93 % of
  clips with the 115M and 97 % with the 0.6B. It answers in 0.30 s (115M) or 0.35 s (0.6B). Pipecat's smart-turn stack is right on 70 %,
  LiveKit's on 73 % and NVIDIA's Parakeet-EOU on 48 %.
  - On AMI test meetings our default `balanced` preset interrupts on 16 % (115M) / 10 % (0.6B) of turns and misses
    36 / 34 %.
  - Pipecat interrupts on 40 % and misses 53 %; LiveKit interrupts on 13 % and misses 72 %. Both answer faster
    (0.75 s against 1.5 s for ours).
- **Following your voice.** With a 5 s voice print, our tracker keeps your words and drops other people's much better
  than an open diarizer bound to the same print. On ICSI test meetings, your-words-only WER is 29 % (0.6B) and
  35 % (115M), against 59 % for NVIDIA Nemotron-3 diarization and 69 % for pyannote 3.1.
  - As plain speaker embedders, TitaNet-L and pyannote's WeSpeaker beat our speaker heads.
- **Language.** Whisper large-v3 is still the best language detector: 95.9 % after 2 s of speech. Our new heads reach
  92.4 % (115M) and 92.7 % (0.6B), up from 91.0 and 87.6 %.
- **Cost.** The 115M runs a whole session in 29 ms per 160 ms of audio on the Mac GPU, 5 streams live, 1.2 GB. The
  0.6B needs 43 ms, 3 streams, 4.9 GB.

## Which audio each head saw: train / selection / test

Every headline number in this file is on a public **test** split, or, where none with labels exists, on a set no
head was trained or selected on, labelled as such. Dev-split numbers from the first pass are kept only in
research/FIXALL.md ("first pass, dev splits").

| head (shipped) | trained on | selected on (held-out) | tested on (this file) | clean? |
|---|---|---|---|---|
| words: the frozen NVIDIA encoders + RNNT (115M, 0.6B); not trained by us | NVIDIA's own data (not disclosed per split; may include AMI / ICSI train and LibriSpeech train) | – (greedy); `--beam 8` picked on held-out AMI / ICSI **train** meetings TS3011b, ES2015c, Bro026, Bmr022 | LibriSpeech test-clean / test-other (300 random utterances each), AMI **test** meetings (IS1009b, ES2004b, TS3003b, EN2002a; 200 segments), ICSI **test** meetings (Bmr013, Bmr018, Bro021; 200 segments), 32 live sessions | yes (for every system; NVIDIA's training data is unknown, the same caveat applies to Parakeet-TDT) |
| speech detector `speech` (115M v0.4, 0.6B v0.3) | AMI train (1200 windows), ICSI train (10 meetings), oto train conversations (clean + quiet variants), room tone | AMI TS3011b / ES2015c, ICSI Bro026 / Bmr022, oto held-out conversations, held-out room tone | AMI **test** meetings, ICSI **test** meetings (64 × 20 s each) | yes |
| turn heads (VAD `vad`, per-frame turn head, v5 classifier `turn_seg`; 0.6B v0.4 also `turn_vad` / `turn_seg_a`) | turn_v4 / v5 mixes: oto, AMI / ICSI train, smart-turn v3.2 **train**, cuts; v0.4's two heads also AMI individual-headset **train** meetings and otoSpeech-280h train sessions (research/TURN_DATA.md) | turn_v4 / v5 held-out split (oto conversations, AMI ES2015c, smart-turn train clips held out) | smart-turn v3.2 **test** (399 clips); AMI **test** meetings (200 turns, see Turn taking); calls = TurnBench dev + oto conversations (no labelled public test split: TurnBench's test labels are withheld) | assistant: yes. AMI: yes. Calls: **not a test split**, and the 115M's `fast` / `assistant` constants were tuned on these calls and on the assistant clips (TURN_V5.md); the 0.6B's were picked on held-out data |
| speaker head `spk` | LibriSpeech train-clean-100 (251 speakers), AMI train, ICSI train | first pass: AMI / ICSI **dev** segments (spk_frame `ami_n200`) | within-meeting EER on AMI / ICSI **test** meetings (200 segments each) | yes (now); the dev numbers of the first pass were also the selection set, so they are dropped |
| TS-VAD (`tsvad_spk.pt`, `tsvad_0p6b.pt`) | AMI / ICSI train windows | AMI / ICSI held-out train meetings (layer sweeps) | eot-bench v2 windows of the AMI **test** meetings (941) and the ICSI **test** meetings (847) | yes (now); the first pass used AMI dev, the speaker-tracking development set |
| LID head (`lid_distill.pt`, `lid_0p6b.pt`) | FLEURS train (+ trainx), extra English | FLEURS **dev** (step choice) | FLEURS **test** (2550 clips) | yes |

## Words: WER %, Whisper English normalizer, [95 % CI]

The same audio and references are used for every system. Lower is better.
- LibriSpeech: 300 utterances drawn at random (seed 0) from each test split, so every speaker can be drawn.
- AMI / ICSI: 200 single-speaker segments (1-15 s) of the corpora's test meetings: AMI IS1009b, ES2004b, TS3003b,
  EN2002a; ICSI Bmr013, Bmr018, Bro021.
- Live calls: the 32 two-party sessions. They are not a corpus split, and no system was trained or selected on them.
- The bootstrap is over utterances; for the live sessions it is over clips.

| system | LibriSpeech test-clean (300) | LibriSpeech test-other (300) | AMI test (200) | ICSI test (200) | live calls, every word (32) | live calls, user channel (16) | mode |
|---|---|---|---|---|---|---|---|
| **audioforge 0.6B** | 2.69 [2.05, 3.37] | 5.71 [4.90, 6.64] | **7.87** [6.80, 9.07] | 10.25 [8.56, 12.49] | 11.55 [8.96, 14.93] | **5.70** [4.39, 7.18] | streaming, 160 ms |
| audioforge 115M | 2.38 [1.86, 2.96] | 6.82 [5.81, 7.81] | 16.11 [14.26, 18.02] | 18.44 [16.34, 20.76] | 20.34 [16.67, 25.08] | 15.35 [12.56, 19.18] | streaming, 160 ms |
| audioforge 115M `--beam 8` (option) | 2.23 [1.73, 2.80] | 6.26 [5.30, 7.24] | 14.58 [12.80, 16.49] | 18.03 [15.78, 20.24] | 18.61 [15.16, 23.02] | 12.89 [10.39, 16.05] | streaming, 160 ms; finals from a beam |
| NVIDIA Parakeet-TDT 0.6B v3 | **1.83** [1.39, 2.28] | **3.40** [2.80, 4.06] | 8.31 [7.02, 9.78] | **7.56** [6.33, 9.05] | **11.04** [8.30, 14.67] | 7.68 [5.23, 11.67] | offline |
| Whisper large-v3 | 2.04 [1.37, 2.90] | 3.55 [2.73, 4.48] | 10.69 [9.29, 12.42] | 13.24 [11.56, 15.21] | 12.69 [9.79, 15.93] | 8.45 [6.00, 11.25] | offline |
| Whisper large-v3-turbo | 2.62 [1.63, 3.94] | 3.57 [2.87, 4.26] | 10.59 [9.11, 12.07] | 13.35 [11.75, 15.26] | 12.91 [10.02, 16.33] | 8.38 [6.63, 10.34] | offline |
| Whisper small (LiveKit default STT) | 3.65 [2.80, 4.61] | 7.50 [6.35, 8.69] | 11.91 [10.35, 13.62] | 15.04 [13.16, 17.11] | 14.67 [11.60, 18.18] | 10.14 [8.25, 12.25] | offline |
| Nemotron 3.5 ASR 0.6B (multilingual) | not measured on the test splits (first-pass dev numbers: research/CORE_3P5.md) | | | | | | streaming |
| Deepgram Nova-3 | not measured (paid API) | | | | | | |

Paired difference against the 0.6B, percentage points [95 % CI] (positive = the baseline makes more errors):

| baseline − 0.6B | test-clean | test-other | AMI test | ICSI test | live, every word |
|---|---|---|---|---|---|
| Parakeet-TDT v3 | **−0.86 [−1.30, −0.43]** | **−2.31 [−3.02, −1.63]** | +0.44 [−0.75, +1.78] | **−2.69 [−4.63, −1.16]** | −0.51 [−1.50, +0.69] |
| Whisper large-v3 | −0.65 [−1.24, +0.09] | **−2.16 [−2.97, −1.27]** | **+2.82 [+1.80, +4.01]** | **+2.99 [+0.77, +5.17]** | +1.14 [−0.46, +2.89] |
| Whisper large-v3-turbo | −0.07 [−0.94, +1.17] | **−2.14 [−2.88, −1.40]** | **+2.72 [+1.72, +3.83]** | **+3.10 [+0.87, +5.19]** | +1.36 [−0.12, +2.82] |
| Whisper small | **+0.96 [+0.32, +1.81]** | **+1.79 [+0.77, +2.91]** | **+4.04 [+2.95, +5.37]** | **+4.79 [+2.43, +6.95]** | **+3.12 [+1.25, +4.92]** |
| audioforge 115M | −0.31 [−0.75, +0.15] | **+1.11 [+0.37, +1.91]** | **+8.24 [+6.78, +9.77]** | **+8.19 [+5.90, +10.35]** | **+8.79 [+7.13, +10.93]** |

`--beam 8` against greedy on the 115M: AMI test −1.53 [−2.56, −0.66], ICSI test −0.41 [−1.15, +0.35], live
−1.73 [−2.77, −0.92]. It is an option, not the default, because it adds 1.5 ms (CPU) / 2.2 ms (MPS) per 160 ms chunk
(research/FIXALL.md step 5). The 0.6B gains nothing from a beam.

**Streaming word latency (ours only).** This is the time from the end of a word to the word first appearing in the
live transcript. It was measured through the served engine on MPS, on 12 two-minute windows of the AMI test meetings
(`stt_latency.py` protocol):
- 115M: p50 274 ms [250, 291], p95 571 ms.
- 0.6B: p50 291 ms [272, 311], p95 591 ms.

The baselines are offline and have no streaming latency.

### Final transcript ready after you stop

The time from the moment the user stops talking to the turn's final transcript being ready, for every system. Lower
is better. ms, [95 % CI].

| system | device | p50 ms | p95 ms | p50, turns < 2 s (9) | p50, 2-5 s (17) | p50, > 5 s (30) |
|---|---|---|---|---|---|---|
| **audioforge 115M**, `--final-chunk-ms 1120` | MPS | **25** [22, 28] | **48** [36, 66] | 26 | 27 | 24 |
| audioforge 0.6B, `--final-chunk-ms 1120` | MPS | 49 [46, 51] | 71 [59, 77] | 50 | 49 | 48 |
| NVIDIA Parakeet-TDT 0.6B v3 (offline) | CPU, 2 threads | 341 [305, 375] | 617 [493, 885] | 257 | 300 | 411 |
| Whisper large-v3-turbo | MPS, fp16 | 282 [265, 305] | 498 [396, 1314] | 243 | 263 | 333 |
| Whisper large-v3 | MPS, fp16 | 576 [494, 711] | 1631 [1097, 4762] | 382 | 455 | 825 |
| Whisper small (LiveKit default STT) | CPU, int8, 2 threads | 1424 [1377, 1468] | 2308 [1704, 3584] | 1306 | 1358 | 1526 |

- **Audio and turns.** The user channel of the 16 TurnBench clips of the live set. This is the "user" half of the 32
  live sessions (the other half is the mono mix of the same calls). There are 56 labelled user turns (clips.json
  `user_turns`): 1.1-39.7 s long, p50 5.4 s, p95 21.6 s. As for the live words rows, these sessions are a public test
  set. They are not used for any training or selection of the words path.
- **Clock start.** The clock starts at the labelled end of each user turn. No end-of-turn detector is part of this
  number: every system is told that the turn has ended at the same moment. End-of-turn timing is in "Turn taking".
- **Offline systems (Parakeet-TDT, Whisper).** At the turn end the system gets the whole turn's audio, as an
  end-of-utterance STT does in Pipecat / LiveKit. The clock runs until the text comes back. The settings are the
  ones in the words table, batch 1. Turns > 30 s use Whisper's sequential long-form decoding.
- **audioforge.** The served single-mode engine with `--final-chunk-ms 1120` runs over the whole session in 20 ms
  blocks. The block that holds a turn end is cut at the end sample. The clock covers three steps:
  1. processing that last piece of audio;
  2. the slow pass's flush, a partial chunk up to the turn end (`LookaheadStream.flush_view`, the same code the
     engine runs at its own `turn_end`);
  3. decoding the turn's tokens.

  The flush alone takes 21 / 39 ms p50 / p95 (115M) and 45 / 60 ms (0.6B).
- **Setup.** Every model is warmed up first: load time and first-call compile are left out. One stream, nothing else
  running on the Mac (Apple M5), one process per system through `scripts/dev/gate.sh`. Each system runs on its
  device in this report.
- **CI.** 1000 bootstrap resamples over turns.
- **Why these numbers are not the ones in research/DUAL_RATE.md.** That file reports 9 / 22 ms (115M) and 36 / 48 ms
  (0.6B) for the same engine. Those were timed after the engine's *own* `turn_end` decisions: 87 / 83 decisions on
  all 32 sessions, mono and user. There the slow pass often had its frames decoded already (flush 0 ms). At the
  labelled turn ends a flush is almost always needed, and the time of the last block is included. So the numbers here
  are higher, and they are the ones to compare with the offline systems.
- **Length.** Offline systems cost more on longer turns. Parakeet-TDT's p50 goes from 257 ms (< 2 s) to 411 ms
  (> 5 s), and Whisper large-v3's from 382 to 825 ms. Whisper small pads every input to 30 s, so it costs ~1.3 s even
  on short turns. The streaming cores already hold the turn's encoder state, so their time does not grow with turn
  length.
- **Words during the turn.** audioforge also streams partial words while the user is talking: a word appears
  ~0.29 s after it is said (p50 274 / 291 ms above). The offline systems produce nothing until the turn ends.
- The tails of Whisper large-v3 (max 5.0 s) and turbo (max 1.4 s) come from the long turns: > 30 s turns use the
  long-form path.
- The run took ~13 min wall-clock: 115M 193 s, 0.6B 287 s, Parakeet-TDT 36 s, Whisper large-v3 58 s, turbo 25 s,
  small 95 s, each including the model load. Numbers: `runs/final_compare.json` `final_latency`. Per-turn rows:
  `/Volumes/ExternalSSD/nvidia-audio-models/scratch/finallat/<system>.jsonl`.

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_latency.py run --system \
        <ours_115m_1120|ours_0p6b_1120|parakeet_tdt|whisper_large|whisper_turbo|whisper_small>
    .venv/bin/python scripts/research/final_latency.py report; python3 demo/images/redesign/export_final.py

## Speech detection

AMI test and ICSI test meetings, 64 × 20 s windows each, 80 ms frames, label = anyone speaking.
- Baselines are max-pooled onto the 80 ms grid. The bootstrap is over windows.
- "Miss" = speech frames missed at the threshold where 7.5 % of non-speech frames are false alarms.
- "Onset lag" = from a labelled speech onset (after ≥ 400 ms of silence) to the first frame > 0.5, on the frame grid.
  It does not include each model's own look-ahead.
- Our heads were trained on AMI and ICSI train meetings and selected on held-out train meetings. No test meeting was
  used.

| system | AMI test F1 @0.5 | AMI AUC | AMI miss @7.5 % FA | ICSI test F1 | ICSI AUC | ICSI miss | onset lag p50 / p90 (AMI) | live-capable |
|---|---|---|---|---|---|---|---|---|
| audioforge 115M speech head (v0.4) | 0.959 [0.949, 0.968] | 0.966 | 12.5 % | 0.938 [0.919, 0.950] | 0.951 | 12.8 % | 0 / 160 ms | yes (80 ms right context) |
| audioforge 0.6B speech head (v0.3) | 0.957 [0.948, 0.965] | 0.967 | 12.6 % | **0.940** [0.923, 0.952] | **0.956** | **12.1 %** | 0 / 160 ms | yes (80 ms right context) |
| Silero VAD v5 (Pipecat / LiveKit) | 0.899 [0.882, 0.916] | 0.958 | 13.3 % | 0.922 [0.901, 0.935] | 0.923 | 26.6 % | 80 / 320 ms | yes |
| TEN VAD | 0.928 [0.916, 0.940] | 0.952 | 13.7 % | 0.920 [0.898, 0.936] | 0.929 | 20.7 % | 80 / 80 ms | yes |
| NVIDIA MarbleNet v2 frame VAD | 0.941 [0.928, 0.953] | 0.967 | 9.7 % | 0.914 [0.891, 0.931] | 0.924 | 24.7 % | 0 / 80 ms | offline here (whole-window conv) |
| pyannote segmentation-3.0 | **0.975** [0.966, 0.982] | **0.983** | **4.1 %** | 0.891 [0.869, 0.907] | 0.929 | 21.1 % | 0 / 0 ms | no (10 s windows, sees the future) |

Paired F1 differences [95 % CI]:
- TEN VAD − ours: AMI −0.039 to −0.022 (115M) and −0.037 to −0.021 (0.6B); ICSI −0.022 to −0.013 and −0.026 to −0.014.
- Silero − 115M: AMI −0.074 to −0.046, ICSI −0.021 to −0.011.
- MarbleNet − 0.6B: AMI −0.028 to −0.006, ICSI −0.034 to −0.018.
- pyannote − 0.6B: AMI +0.010 to +0.024, ICSI −0.057 to −0.040.

The previous heads (115M v0.3 / 0.6B v0.2, AMI-only training) score AMI test 0.957 / 0.958 and ICSI test 0.886 / 0.890 F1
on the same windows. The new heads are level on AMI and +0.05 on ICSI.

## Turn taking

The test data:
- **Assistant clips**: smart-turn v3.2 **test**, 399 clips (175 complete, 224 cut mid-utterance), each padded with 3 s
  of silence. smart-turn's own training split is the one our classifier was trained on; the test split was never
  used for training or selection.
- **AMI test**: 200 turns from the AMI test meetings (the eot-bench v2 cut, `EOT_AMI_SPLIT=eval`).

How the rows are scored:
- Latency is from the audible end of speech to the turn end, plus the measured compute.
- "False fire" = a turn end within 3 s of an incomplete clip's cut.
- "Interrupt" = a turn end inside the user's turn.
- "Missed" = no turn end before the next speaker or within 6 s.
- The bootstrap is over clips / sessions (1000 resamples for the assistant clips, 200 for AMI).

| system | assistant: accuracy | assistant: p50 / p95 ms | assistant: false fire | assistant: missed | AMI test: p50 ms | AMI test: interrupt | AMI test: missed |
|---|---|---|---|---|---|---|---|
| **audioforge 115M `assistant`** | **92.7 %** [90.2, 95.0] | 299 / 710 | **5.4 %** | 5.1 % | 1327 | 23.5 % | 40.0 % |
| audioforge 115M `balanced` (default) | 43.9 % | 1226 / 1587 | 100 % | 0.0 % | 1527 [1327, 1806] | 15.5 % [10.5, 20.5] | 36.0 % [30.0, 42.0] |
| audioforge 115M `fast` | 41.6 % | 461 / 698 | 100 % | 4.0 % | 1246 | 20.5 % | 37.5 % |
| **audioforge 0.6B `assistant`** | **96.5 %** [94.5, 98.2] | 351 / 763 | **3.1 %** | 4.0 % | 1497 | 22.5 % | 34.5 % |
| audioforge 0.6B `balanced` (default) | 42.6 % | 639 / 799 | 100 % | 2.9 % | 1498 [1418, 1657] | 10.0 % [6.0, 14.5] | 33.5 % [27.0, 40.0] |
| audioforge 0.6B `fast` | 40.1 % | 414 / 607 | 100 % | 8.0 % | 1338 | 16.0 % | **27.5 %** |
| Pipecat smart-turn v3.2 + Silero (defaults) | 69.7 % [65.2, 74.2] | **211** / 242 | 40.2 % | 13.1 % | 752 [496, 1296] | 39.5 % [33.0, 47.0] | 53.0 % [45.5, 60.0] |
| LiveKit Agents 1.8 EnglishModel + Silero | 72.7 % [67.9, 76.9] | 547 / 3016 | 44.6 % | 4.6 % | **745** [641, 1330] | 13.0 % [8.5, 18.0] | 72.0 % [66.0, 78.0] |
| NVIDIA Parakeet-Realtime-EOU 120M | 48.4 % [43.6, 53.1] | 462 / 1210 | 91.5 % | 0.6 % | 1251 | **4.5 %** | 86.0 % |
| *smart-turn v3.2 classifier alone (upper bound: one call per whole clip, no timing)* | *97.0 %* [95.2, 98.5] | – | *2.7 %* | *3.4 %* | – | – | – |

The presets trade one domain for another:
- `assistant` waits up to 3.4 s for a classifier "complete". That suits assistant speech, but it misses many meeting
  turns.
- `balanced` (the default) is the conversation preset.

On AMI test meetings no system has both few interruptions and few misses:
- Pipecat answers fastest but interrupts on 40 % of turns.
- LiveKit and Parakeet-EOU interrupt rarely but miss 72-86 % of turns.
- Ours sits between them.

Caveat for the 115M's `fast` and `assistant` rows: their constants were tuned in an earlier round on the assistant
clips and the call sessions (research/TURN_V5.md). With constants picked on held-out data only, the 115M's assistant
row would read 95.0 % / 354 ms / 3.1 % (research/FIXALL.md step 2). The 0.6B's constants were picked on held-out
data. The 0.6B's heads v0.4 change only `assistant` (its own classifier, trained with real two-party channels, asked
at 160 ms of quiet): before (v0.3) 93.0 % [90.5, 95.5] / 379 ms / 4.5 %, AMI test 1619 / 4.0 % / 53.0 %. It now
interrupts more on meetings; `balanced` / `fast` make the same decisions as v0.3 (their totals differ by the measured
chunk compute only). research/TURN_DATA.md.

**Two-party calls: no labelled public test split.** TurnBench publishes only its dev split with labels; its test
split's labels are withheld for the submission site. The call rows therefore use the 16 TurnBench **dev** clips plus
16 oto conversations, user channel (109 turn ends). No head was trained on them, but the 115M's `fast` / `assistant`
constants were tuned on them. They are kept out of the headline table above:

| system (calls, not a test split) | p50 / p95 ms | interrupt | missed |
|---|---|---|---|
| audioforge 115M `balanced` | 955 / 1918 | 20.2 % [13.1, 29.2] | 7.3 % [3.5, 12.0] |
| audioforge 115M `fast` | 547 / 1861 | 24.8 % | 5.5 % |
| audioforge 0.6B `balanced` | 729 / 2034 | 19.3 % [12.1, 26.4] | 10.1 % [4.9, 16.0] |
| audioforge 0.6B `fast` | 492 / 1974 | 26.6 % | 11.0 % |
| Pipecat smart-turn v3.2 + Silero | 237 / 3217 | 35.8 % [25.7, 46.6] | 24.8 % [16.1, 33.6] |
| LiveKit Agents 1.8 + Silero | 567 / 3127 | 26.6 % [20.2, 34.0] | 22.9 % [15.1, 30.2] |
| NVIDIA Parakeet-Realtime-EOU | 1277 / 5362 | 7.3 % | 30.3 % |

## Speaker tracking ("your words only")

The windows are the eot-bench v2 windows of the corpora's **test** meetings: ICSI Bmr013 / Bmr018 / Bro021 (847
windows) and AMI IS1009b / ES2004b / TS3003b / EN2002a (941 windows).
- The target is the main speaker of each window, with a 5 s voice print cut from elsewhere in the same meeting.
- **tWER** = WER of the target's words only. Streaming words are kept where the system's target mask is on, with the
  same keep rule for every arm.
- **Target DER** = (missed target frames + false target frames) / target frames, at 80 ms, with no collar.
- The open diarizers' columns are bound to the same print by `tsvad.vp_follow`. The better of two binders (served
  speaker head, TitaNet-L) is shown.
- The bootstrap CI is over meetings (3 / 4 meetings: coarse).

| system | AMI test tWER % | ICSI test tWER % [meeting CI] | AMI test target DER % | ICSI test target DER % | AMI test tracking F1 | ICSI test tracking F1 |
|---|---|---|---|---|---|---|
| audioforge 115M (115M words) | 51.5 [41.1, 58.8] | 34.9 [29.4, 41.0] | 37.3 | 22.4 | 0.811 | 0.884 |
| **audioforge 0.6B (0.6B words)** | **47.1** [32.3, 57.7] | **29.0** [22.8, 36.1] | **35.8** | **20.7** | **0.829** | **0.896** |
| Nemotron-3-Diarization + print, 115M words | 68.6 [55.1, 79.3] | 64.0 [56.6, 72.6] | 62.9 | 75.6 | 0.676 | 0.695 |
| Nemotron-3-Diarization + print, 0.6B words | 64.4 [47.5, 76.9] | 59.4 [49.0, 69.6] | (same track) | | | |
| pyannote speaker-diarization-3.1 + print, 115M words | 77.1 [58.7, 90.3] | 73.6 [62.9, 81.2] | 64.7 | 88.7 | 0.734 | 0.669 |
| pyannote speaker-diarization-3.1 + print, 0.6B words | 77.8 [52.6, 96.2] | 69.0 [55.5, 77.7] | (same track) | | | |
| no filter (115M / 0.6B words) | 94.2 / 102.8 | 104.0 / 101.9 | – | – | – | – |
| oracle filter (115M / 0.6B words) | 43.0 / 31.7 | 29.5 / 23.4 | – | – | – | – |

How these rows were produced:
- ICSI: the first pass scored the dev and test meetings together (1312 windows). The stored per-unit and per-window
  counts are re-pooled here over the test meetings only, so words, prints, tracks and keep rule are those of the first
  pass. The 0.6B tracker's frame metrics were re-run on the test windows. A full-set re-pool reproduces the first-pass
  numbers exactly (115M tWER 37.14; 0.6B tracking F1 0.8944).
- AMI: the whole pipeline was run on the AMI test meetings' windows: words of both cores, prints, Nemotron-3 streaming
  tracks, pyannote 3.1, TS-VAD tracks and frame counts.
- Our TS-VAD heads were trained on AMI / ICSI train meetings. pyannote runs offline on the whole window, which helps
  it. Nemotron-3 has AMI in its training data.

**Within-meeting speaker EER** on the test meetings (200 single-speaker segments per corpus, segment bootstrap):

| embedder | AMI test EER % | ICSI test EER % |
|---|---|---|
| audioforge 115M speaker head (block 4) | 5.0 [3.2, 6.9] | 2.5 [1.3, 3.8] |
| audioforge 0.6B speaker head (block 5) | 3.8 [2.2, 5.4] | 1.9 [1.0, 3.4] |
| NVIDIA TitaNet-L (the teacher) | **1.9** [0.7, 3.7] | 1.2 [0.4, 2.0] |
| pyannote WeSpeaker ResNet34 (in pyannote 3.1) | 2.0 [0.7, 3.7] | **0.9** [0.4, 1.6] |

The test meetings have fewer speakers per meeting than the dev meetings, so every EER is lower than in the first pass.
The order is the same: the dedicated embedders are better voice matchers than our heads. The tracker still wins on tWER
and DER, because the TS-VAD head is trained to track the print frame by frame, not only to embed.

## Language ID: FLEURS-17 test (2550 clips), accuracy % [95 % CI]

| system | 2 s from speech onset | full clip |
|---|---|---|
| Whisper large-v3 | **95.9** [95.2, 96.7] | 99.5 [99.2, 99.7] |
| Whisper large-v3-turbo | 95.2 [94.3, 96.0] | 99.3 [99.0, 99.7] |
| NVIDIA AmberNet (the teacher; reused, `data/lid/preds/ambernet`) | 95.1 [94.3, 95.9] | **99.5** [99.3, 99.8] |
| audioforge 0.6B LID head v2 | 92.7 [91.7, 93.7] | 98.6 [98.1, 99.0] |
| audioforge 115M LID head v2 | 92.4 [91.5, 93.5] | 98.2 [97.7, 98.8] |
| Whisper small | 90.6 [89.4, 91.7] | 99.1 [98.7, 99.4] |
| *previous heads (same clip path): 115M `lid_distill.pt` / 0.6B `lid_0p6b.pt`* | *90.9 / 87.6* | *97.8 / 95.5* |

Every system's posterior is restricted to the 17 test languages (the `lid.py` protocol). Our heads are VAD-gated, as
served. They were selected on FLEURS dev and never saw FLEURS test (research/FIXALL.md step 3).

## Cost on this Mac (Apple M5)

The full single-mode engine: ASR, speech detector, VAD, turn, TS-VAD and LID, on the bundled 16 s call, best of 3.
Streams = interleaved sessions on 40 s AMI windows, real time while p95 of the summed block compute stays under 160 ms
(`mps_115m.py` protocols).

| core | ms per 160 ms chunk, MPS (p50 / p95) | CPU 2 threads | real-time streams, MPS | real-time streams, CPU | peak RSS |
|---|---|---|---|---|---|
| audioforge 115M (heads v0.4) | 28.6 / 30.7 | 30.4 / 33.0 | 5 | 4 | 1.2 GB (+0.5 GB MPS) |
| audioforge 0.6B (heads v0.3) | 42.9 / 49.0 | 97.1 / 105.1 | 3 | 1 | 4.9 GB (+3.3 GB MPS) |

The previous heads cost 28.2 / 30.3 ms (115M, MPS / CPU) and 43.0 / 98.7 ms (0.6B) in the same back-to-back runs
(research/FIXALL.md latency gate). The LID head v2 adds nothing measurable (±0.2 ms). The 0.6B's heads v0.4 add +0.4 ms (MPS) and +0.2-0.6 ms (CPU) per chunk in back-to-back runs against v0.3, with the same 3 real-time MPS streams (research/TURN_DATA.md); the row above is v0.3's.

**Model sizes** (parameters counted from the weight files):

| model | params |
|---|---|
| audioforge 115M (whole served model, heads v0.4) | 122.1 M (+ LID head v2 2.4 M) |
| audioforge 0.6B (whole served model, heads v0.3) | 622.6 M (+ LID head v2 2.9 M) |
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


## What changed with the test splits (first pass, dev → this pass, test)

Each change mixes two things: the split moved (dev → test), and for the speech detector and LID the heads changed
too. The full list (182 image numbers) is the diff of `numbers_final.json`. The headline rows:

| row | first pass (dev) | this pass (test) | why |
|---|---|---|---|
| WER AMI: 0.6B / 115M / Parakeet-TDT / Whisper large-v3 | 10.4 / 20.6 / 9.5 / 12.4 | 7.9 / 16.1 / 8.3 / 10.7 | split (the test meetings are easier for every system); the 0.6B now leads (TDT − 0.6B +0.44, CI includes 0) |
| WER ICSI: 0.6B / 115M / TDT / Whisper large-v3 | 13.6 / 26.2 / 10.4 / 12.8 | 10.3 / 18.4 / 7.6 / 13.2 | split; the 0.6B now beats Whisper large-v3 (it lost on dev), TDT still leads |
| WER LibriSpeech test-clean: 0.6B / TDT / Whisper large-v3 | 2.1 / 1.9 / 1.4 (first 200 utterances, 3 speakers) | 2.7 / 1.8 / 2.0 (300 random utterances) | subset: the old "first 200" covered only the first few speakers |
| WER LibriSpeech test-other | – | 0.6B 5.7, TDT 3.4, Whisper large-v3 3.6 | new |
| speech F1 AMI: 115M / 0.6B / best live baseline | 0.951 / 0.951 / 0.937 (MarbleNet) | 0.959 / 0.957 / 0.941 (MarbleNet) | split + new heads |
| speech F1 ICSI: 115M / 0.6B / best baseline | 0.898 / 0.906 / 0.935 (TEN) | **0.938 / 0.940** / 0.922 (Silero) | new heads (+0.05) and split (TEN 0.935 → 0.920): the row flips from a loss to a win |
| turn, AMI test `balanced` p50 / interrupt / missed: 115M; 0.6B | 1326 / 10.5 / 33.5; 1180 / 11.0 / 33.0 | 1527 / 15.5 / 36.0; 1497 / 10.0 / 33.5 | split; baselines moved too (Pipecat interrupts 27.5 → 39.5 %) |
| turn, assistant clips | unchanged (already smart-turn **test**) | unchanged | – |
| tWER AMI: 0.6B / 115M / Nemotron-3 (0.6B words) / pyannote | 63.2 / 62.1 / 71.6 / 85.3 | 47.1 / 51.5 / 64.4 / 77.1 | split (test windows, new pipeline run) |
| tWER ICSI: 0.6B / 115M / Nemotron-3 / pyannote | 31.7 / 37.1 / 60.0 / 74.8 (dev + test windows) | 29.0 / 34.9 / 59.4 / 73.6 (test windows only) | split |
| speaker EER AMI: ours 115M / 0.6B / TitaNet-L / WeSpeaker | 19.8 / 13.6 / 12.0 / 11.9 | 5.0 / 3.8 / 1.9 / 2.0 | split (fewer speakers per test meeting); same order |
| speaker EER ICSI | 7.0 / 3.6 / 2.1 / 1.0 | 2.5 / 1.9 / 1.2 / 0.9 | split |
| LID 2 s: 115M / 0.6B | 91.0 / 87.6 | 92.4 / 92.7 | new heads (FLEURS test both times) |
| cost per chunk, MPS: 115M / 0.6B | 27.7 / 43.7 ms | 28.6 / 42.9 ms | new heads + run-to-run spread (back to back: +0.4 / −0.1 ms) |
| real-time streams, MPS, 115M | 4 | 5 | re-measured (5 streams: p95 151 ms < 160) |
| call rows (TurnBench dev + oto) | headline | moved out of the headline (no labelled public test split) | policy |


## How each system was run (versions, licences)

- **audioforge 115M**: `runs/stage1_served_v4.afm` (served heads v0.4) + `assets/lid_115m_v2.pt`, NVIDIA 115M
  streaming FastConformer, CC-BY-4.0.
  - ASR: masked [70,1] forward on CPU (equal to 160 ms streaming); `--beam 8`: the same encoder output, RNNT beam search.
  - Speech detection: the served `speech` head (the client's frame probability).
  - Turn: the served `--mode single` engine on MPS. Each session is dumped once, and each preset is applied with the
    served policy twin (`core_0p6b_turn.served_rules` / `ev_score`). The twin reproduces the served turn_ends.
- **audioforge 0.6B**: `served_0p6b_v0.4.afm` (`assets/served_heads_0p6b_v0.4.pt`: v0.3 + `turn_vad` / `turn_seg_a` for
  the `assistant` preset; v0.3 = v0.2 + the `speech` head) +
  `assets/lid_0p6b_v2.pt` on nvidia/nemotron-speech-streaming-en-0.6b, NVIDIA Open Model License. Same harnesses as
  the 115M. The turn rows of both cores read `heads.vad`, unchanged from v0.3 / v0.2 (identical turn ends, research/FIXALL.md);
  the 0.6B's `assistant` preset reads its own classifier (heads v0.4, research/TURN_DATA.md; served == offline 32 / 32).
- **Whisper small**: faster-whisper 1.2.1 int8, CPU 2 threads, beam 1, English. MIT.
- **Whisper large-v3 / large-v3-turbo**: transformers 5.17, fp16 on MPS, greedy, English. Clips over 30 s use
  sequential long-form decoding. MIT. For LID, the language-token posterior after `<|startoftranscript|>`.
- **Parakeet-TDT 0.6B v3**: imported into our PyTorch modules (`nemo_import`), offline, greedy TDT, CPU. CC-BY-4.0.
- **Silero VAD v5.1.2**: TorchScript, 32 ms chunks. MIT.
- **TEN VAD 1.0.6.8**: pip `ten-vad`, 16 ms hop. Apache-2.0 with additional conditions.
- **MarbleNet v2**: `Frame_VAD_Multilingual_MarbleNet_v2.0`, imported. NVIDIA Open Model License.
- **pyannote segmentation-3.0**: pyannote.audio 4.0.7, speech = 1 − P(no speaker), 10 s windows. MIT (gated, already
  accepted).
- **Pipecat smart-turn v3.2 + Silero**: Pipecat 1.12's own `LocalSmartTurnAnalyzerV3` replayed on a simulated clock
  (defaults: confidence 0.7, 0.2 s stop, 3 s fallback). BSD-2-Clause. The per-session decisions come from
  `eot_latency.py baselines` / `eot_assistant.py` (the AMI test turns were run in this pass, same versions) and are
  rescored here with CIs. Pipecat's STT wait is not modelled, which flatters Pipecat.
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
- **Nemotron-3-Diarization**: streaming tracks (`low_latency_032`; ICSI cached, AMI test computed in this pass). Its column is bound by the print with
  `tsvad.vp_follow`; the better of two binders (served speaker head, TitaNet-L) is shown. NVIDIA Open Model License.
- **AmberNet**: reused predictions on the same clips. CC-BY-4.0.
- **Paid APIs** (Deepgram Nova-3, AssemblyAI, Deepgram Flux turn detection, pyannoteAI): not measured.

## Not measured, and why

- **Paid APIs** (Deepgram Nova-3 for words, Deepgram Flux / AssemblyAI turn detection, pyannoteAI): no paid APIs in
  this pass.
- **A labelled public test split for two-party calls**: TurnBench's test labels are withheld (submission site only), so
  the call rows are on TurnBench dev + oto conversations and kept out of the headline table.
- **Nemotron 3.5 ASR 0.6B on the test splits**: only its first-pass dev numbers exist (research/CORE_3P5.md); it is
  not a shipped core.
- **Diarizer baselines on the 32 live sessions** (tWER): no cached tracks; ours: `runs/tswer_live.json`.
- **Baselines' per-stream cost**: offline per-utterance / per-window systems; their decode RTFx is in the json.
- **Full LibriSpeech test sets**: 300 random utterances per split (every system, same audio); the full 2620 / 2939
  would take ~5 h of Whisper / Parakeet decoding on this Mac.

## Reproduce

    P="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_compare.py"
    $P asr --system <S> --set <ls_clean|ls_other|ami_eval|icsi_eval|live> [--device mps]
    $P vad --system <S> | spkeer --system <S> | lid --system <S> --device mps
    EOT_AMI_SPLIT=eval $P eotdump --sys 115m|0p6b --which calls|asst --device mps     # + eot_latency.py prepare_ami / dump / baselines
    EOT_AMI_SPLIT=eval $P eou --which calls --device mps
    $P sttlat --sys 115m|0p6b --device mps | cost --sys 115m|0p6b --sub engine|streams --device mps|cpu | params
    scripts/research/fixall.py twsub --set icsi|ami_eval; frsub; frami                 # target-speaker test rows
    EOT_AMI_SPLIT=eval $P report; python3 demo/images/redesign/export_final.py

Scratch: `/Volumes/ExternalSSD/nvidia-audio-models/scratch/final_compare/` (first-pass cost / word-latency / LID files
of the previous heads kept as `cost_v03`, `sttlat_v03`, and in `scratch/fixall/fc_old`).
