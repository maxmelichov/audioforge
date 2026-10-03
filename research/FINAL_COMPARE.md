# Final comparison: both audioforge cores against the best open models, on public test splits (2026-10-03)

**Status: test-split pass, corrected after the audits of 2026-10-03** (plans/audit/metrics_001.md, fairness_001.md,
leakage_001.md; the fixes are listed in plans/fixwave/eval_fixes.md and under "What the fix wave changed" below).
Both cores are the shipped builds:
- the 115M with served heads v0.4: v0.3 + the `speech` detector head, and LID head v2;
- the English 0.6B (nemotron-speech-streaming-en-0.6b) with heads v0.4: v0.3 (v0.2 + `speech`) + the `assistant`
  preset's own turn classifier and turn VAD (research/TURN_DATA.md), and LID head v2.

Every headline number below is on a standard public **test** split, with every baseline re-run on exactly the same
audio. Rows that the leakage audit found tuned or selected on their own test audio are **not** in the headline tables:
they are in "Appendix A: rows flagged by the leakage audit" and are marked ⚠ wherever they are mentioned. Rows with a
weaker flag stay in place with a ⚑ mark and a note.

All numbers are in `runs/final_compare.json` (script `scripts/research/final_compare.py`, scoring pieces in
`final_scoring.py`); `final_compare.py report` regenerates every number here with no environment variable. The image
numbers are in `demo/images/redesign/numbers_final.json` (built by `export_final.py`). The heads work is in
research/FIXALL.md.

How uncertain the numbers are: the 95 % CIs resample **meetings** for AMI / ICSI (WER, speech detection, speaker
EER, target-speaker WER), **sessions** for the live calls, clips for the assistant clips and utterances for
LibriSpeech / FLEURS. AMI test has 4 meetings and ICSI test 3, so the meeting CIs are coarse: they can be narrow when
the few meetings happen to agree and very wide when one meeting differs.

## In plain words

- **Words.** On AMI test meetings the 0.6B core makes the fewest mistakes of every system here: 7.9 %, level with
  NVIDIA's offline Parakeet-TDT v3 (8.3 %; the difference's CI includes 0) and clearly ahead of Whisper large-v3
  (10.7 %). It does this while streaming: a word shows up about 0.3 s after it is said.
  - On ICSI test meetings and on LibriSpeech, Parakeet-TDT is better: 7.6 vs 10.3 % on ICSI, 3.4 vs 5.7 % on
    test-other.
  - On the user's channel of live calls the 0.6B is best with the verbatim references (5.7 % against 7.7 % for
    Parakeet-TDT and 8.4 % for Whisper turbo). The references keep repetitions and false starts, which Whisper leaves
    out by design; with those collapsed in references and hypotheses alike, the 0.6B and Whisper large-v3-turbo tie
    (5.4 % each), Whisper large-v3 5.8 %, Parakeet-TDT 7.7 %.
  - On FLEURS English test (read sentences) the offline models are better: Whisper large-v3 6.4 %, Parakeet-TDT
    7.0 %, the 0.6B 8.0 % (7.8 % with the 1.12 s final), Whisper small 7.8 %.
  - Whisper small is now run at beam 5, its front end's default: 10.9 % on AMI (11.9 % at beam 1).
  - The 115M makes about twice as many mistakes on meetings. Its optional `--beam 8` removes 1.5 of them per 100 words
    on AMI.
- **Final text after you stop.** With every model on the Mac GPU, the 0.6B's final text is ready 49 ms after the user
  stops, Parakeet-TDT's 232 ms, Whisper turbo's 282 ms. The 115M is fastest (25 ms) but its flushed turn texts have
  the most errors.
- **Speech detection.** On ICSI test, with heads retrained so that no ICSI test speaker was heard in training, ours
  lead on AUC (0.941 / 0.945 against 0.932 for the best of the others) and missed speech (16.0 / 15.5 % against
  20.5 %), and are level with the best live VADs on F1 (0.930 / 0.922 against 0.924 Silero, 0.922 TEN; the
  differences' CIs include 0). The shipped heads score 0.938 / 0.940 there, but they were trained on those speakers
  (Appendix A). On AMI test they beat the live VADs on F1 (0.959 / 0.957 against
  0.941 for MarbleNet, 0.930 TEN, 0.901 Silero), but MarbleNet ties them on AUC (0.967) and misses less speech at a
  fixed false-alarm rate (9.6 % against 12.5 %), and against the 0.6B the F1 gap's CI touches 0. pyannote's
  segmentation model is the best on AMI (0.977) but reads 10 s ahead, so it cannot run live, and it is the worst on
  ICSI.
- **Turn taking.** On speech aimed at an assistant (smart-turn v3.2 test), the 0.6B's `assistant` preset is right on
  97.7 % of clips and answers in 0.35 s. Pipecat's smart-turn stack is right on 75.9 %, LiveKit's on 85.2 % and
  NVIDIA's Parakeet-EOU on 49.4 % (all at the 2.5 s false-fire window, chosen so that no system's fallback timer sits
  on it; at the old 3.0 s window: 96.5 / 69.7 / 72.7 / 48.4 %). The 115M's shipped `assistant` rule was tuned on these
  clips (Appendix A); re-picked on held-out data only it reaches 96.0 % at 381 ms (candidate heads v0.5, not yet the
  default).
  - On AMI test meetings our default `balanced` preset interrupts on 16 % (115M) / 10 % (0.6B) of turns and misses
    36 / 34 %, knowing the user's voice print. Without the print (the same engine, no TS-VAD "others" path) they miss 74 % of turns, like LiveKit (72 %), and interrupt on 6.5 / 5.5 %: knowing the user's voice is what turns the misses into answered turns.
  - Pipecat interrupts on 40 % and misses 53 %; LiveKit interrupts on 13 % and misses 72 %. Both answer faster
    (0.75 s against 1.5 s for ours).
- **Following your voice.** With a 5 s voice print, our tracker keeps your words and drops other people's much better
  than an open diarizer bound to the same print. On AMI test meetings, your-words-only WER is 47 % (0.6B) and 51 %
  (115M), against 64 % for NVIDIA Nemotron-3 diarization and 78 % for pyannote 3.1 (both on the 0.6B's words). Most of
  that margin is in how a diarizer column is bound to the print: with the column chosen by the labels (an upper bound)
  Nemotron-3 tracks the target nearly as well as ours (F1 0.795 against 0.811 / 0.829). The ICSI rows are in
  Appendix A (every ICSI test speaker is also in our training meetings).
  - As plain speaker embedders, TitaNet-L and pyannote's WeSpeaker beat our speaker heads.
- **Language.** Whisper large-v3 is still the best language detector: 95.9 % after 2 s of speech. Our heads reach
  92.4 % (115M) and 92.7 % (0.6B) ⚑.
- **Cost.** The 115M runs a whole session in 29 ms per 160 ms of audio on the Mac GPU, 5 streams live, 1.2 GB. The
  0.6B needs 43 ms, 3 streams, 4.9 GB.

## Which audio each head saw: train / selection / test

| head (shipped) | trained on | selected on (held-out) | tested on (this file) | clean? |
|---|---|---|---|---|
| words: the frozen NVIDIA encoders + RNNT (115M, 0.6B); not trained by us | NVIDIA's own data (not disclosed per split; may include AMI / ICSI train and LibriSpeech train) | – (greedy); `--beam 8` picked on held-out AMI / ICSI **train** meetings TS3011b, ES2015c, Bro026, Bmr022 | LibriSpeech test-clean / test-other (300 random utterances each), AMI **test** meetings (IS1009b, ES2004b, TS3003b, EN2002a; 200 segments), ICSI **test** meetings (Bmr013, Bmr018, Bro021; 200 segments), 32 live sessions, FLEURS en_us **test** (150 of 647 utterances) | yes (for every system; NVIDIA's training data is unknown, the same caveat applies to Parakeet-TDT) |
| speech detector `speech` (115M v0.4, 0.6B v0.3 = v0.4) | AMI train (1200 windows), ICSI train (10 meetings), oto train conversations (clean + quiet variants), room tone | AMI TS3011b / ES2015c, ICSI Bro026 / Bmr022, oto held-out conversations, held-out room tone | AMI **test** meetings, ICSI **test** meetings (64 × 20 s each) | AMI yes. ICSI **⚠** for the shipped heads (every ICSI test speaker is in their ICSI train meetings, leakage item 3): the ICSI headline row uses heads retrained without any ICSI meeting that holds a test speaker (speakers unseen in training; not shipped), the shipped heads' ICSI row is in Appendix A |
| turn heads (VAD `vad`, per-frame turn head, v5 classifier `turn_seg`; 0.6B v0.4 also `turn_vad` / `turn_seg_a`) | turn_v4 / v5 mixes: oto, AMI / ICSI train, smart-turn v3.2 **train**, cuts; v0.4's two heads also AMI individual-headset **train** meetings and otoSpeech-280h train sessions (research/TURN_DATA.md) | turn_v4 / v5 held-out split (oto conversations, AMI ES2015c, smart-turn train clips held out) | smart-turn v3.2 **test** (399 clips); AMI **test** meetings (200 turns); calls = TurnBench dev + oto conversations (no labelled public test split) | **115M ⚠**: its `fast` / `assistant` constants and its v5 classifier (block, recipe) were tuned / selected on the 399 assistant clips and the calls (leakage item 1): assistant and calls rows in Appendix A; its AMI test rows are clean. **0.6B ⚑**: `assistant` v0.4 picked on held-out data only, but it reuses the 115M's classifier recipe and teacher (item 5, weak); `balanced` / `fast` chosen between candidates after both were scored on the evaluation sets (item 4) |
| speaker head `spk` | LibriSpeech train-clean-100 (251 speakers), AMI train, ICSI train | first pass: AMI / ICSI **dev** segments | within-meeting EER on AMI / ICSI **test** meetings (200 segments each) | AMI yes; **ICSI ⚠**: the 13 ICSI test speakers are all in the ICSI train meetings (seen speakers): Appendix A |
| TS-VAD (`tsvad_spk.pt`, `tsvad_0p6b.pt`) | AMI / ICSI train windows | AMI / ICSI held-out train meetings (layer sweeps) | eot-bench v2 windows of the AMI **test** meetings (941) and the ICSI **test** meetings (847) | AMI yes; **ICSI ⚠** (seen speakers; the ICSI test windows were also the non-regression check of a serving rule, item 7): Appendix A |
| LID head (`lid_115m_v2.pt`, `lid_0p6b_v2.pt`) | FLEURS train (+ trainx), extra English | FLEURS **dev** (step choice) | FLEURS **test** (2550 clips) | ⚑ the input blocks (115M 8-12) were chosen with a probe scored on FLEURS test (item 8, probably well under a point) |


## Words: WER %, Whisper English normalizer, [95 % CI]

The same audio and references are used for every system. Lower is better.
- LibriSpeech: 300 utterances drawn at random (seed 0) from each test split, so every speaker can be drawn.
- AMI / ICSI: 200 single-speaker segments (1-15 s) of the corpora's test meetings: AMI IS1009b, ES2004b, TS3003b,
  EN2002a; ICSI Bmr013, Bmr018, Bro021. CI: 1000 resamples of the meetings (4 / 3: coarse).
- Live calls: the 32 two-party sessions (16 TurnBench dev clips × mono mix / user channel). They are not a corpus
  split, and no words path was trained or selected on them. CI: resamples of the 16 sessions. References are verbatim
  (repetitions and false starts kept); see the disfluency column below.
- FLEURS en: 150 of the 647 en_us **test** utterances: the subset cached for the LID test (data/lid/fleurs/manifest.jsonl),
  fixed before any WER was computed. References: the raw `transcription` column of en_us test.tsv. Never used to
  train or select any words path.

| system | LibriSpeech test-clean (300) | LibriSpeech test-other (300) | AMI test (200) | ICSI test (200) | live calls, every word (32) | live calls, user channel (16) | FLEURS en test (150) | mode |
|---|---|---|---|---|---|---|---|---|
| **audioforge 0.6B** | 2.69 [2.05, 3.37] | 5.71 [4.90, 6.64] | **7.87** [7.69, 8.04] | 10.25 [7.75, 13.04] | 11.55 [8.96, 14.93] | **5.70** [4.39, 7.18] | 8.02 [6.32, 9.74] | streaming, 160 ms |
| audioforge 115M | 2.38 [1.86, 2.96] | 6.82 [5.81, 7.81] | 16.11 [12.83, 18.36] | 18.44 [11.91, 25.06] | 20.34 [16.67, 25.08] | 15.35 [12.56, 19.18] | 11.26 [9.27, 13.30] | streaming, 160 ms |
| audioforge 115M `--beam 8` (option) | 2.23 [1.73, 2.80] | 6.26 [5.30, 7.24] | 14.58 [12.03, 16.48] | 18.03 [11.21, 25.40] | 18.61 [15.16, 23.02] | 12.89 [10.39, 16.05] | 10.87 [9.01, 12.86] | streaming, 160 ms; finals from a beam |
| NVIDIA Parakeet-TDT 0.6B v3 | **1.83** [1.39, 2.28] | **3.40** [2.80, 4.06] | 8.31 [6.80, 9.83] | **7.56** [5.39, 10.16] | **11.04** [8.30, 14.67] | 7.68 [5.23, 11.67] | 6.97 [5.43, 8.69] | offline |
| Whisper large-v3 | 2.04 [1.37, 2.90] | 3.55 [2.73, 4.48] | 10.69 [9.36, 11.35] | 13.24 [11.74, 14.90] | 12.69 [9.79, 15.93] | 8.45 [6.00, 11.25] | **6.35** [4.73, 8.10] | offline |
| Whisper large-v3-turbo | 2.62 [1.63, 3.94] | 3.57 [2.87, 4.26] | 10.59 [10.17, 11.43] | 13.35 [10.50, 15.66] | 12.91 [10.02, 16.33] | 8.38 [6.63, 10.34] | 6.48 [4.89, 8.18] | offline |
| Whisper small, beam 5 (faster-whisper defaults) | 3.10 [2.46, 3.85] | 7.10 [6.12, 8.16] | 10.90 [9.82, 11.91] | 15.01 [11.02, 19.56] | 13.81 [11.22, 17.03] | 9.58 [7.18, 12.11] | 7.79 [6.07, 9.59] | offline |
| Whisper small, beam 1 | 3.65 [2.80, 4.61] | 7.50 [6.35, 8.69] | 11.91 [11.20, 12.45] | 15.04 [11.62, 19.05] | 14.67 [11.60, 18.18] | 10.14 [8.25, 12.25] | 7.95 [6.25, 9.66] | offline |
| Nemotron 3.5 ASR 0.6B (multilingual) | not measured on the test splits (first-pass dev numbers: research/CORE_3P5.md) | | | | | | | streaming |
| Deepgram Nova-3 | not measured (paid API) | | | | | | | |

What the voice-agent frameworks actually default to (read from the installed packages): **Pipecat 1.12**'s local
`WhisperSTTService` defaults to `Systran/faster-distil-whisper-medium.en` and calls faster-whisper's `transcribe()`
with its defaults (beam 5), dropping segments with `no_speech_prob` ≥ 0.4. **LiveKit Agents 1.8.3** has no default
STT: `AgentSession(stt=...)` must be given, and its `inference.STT` is LiveKit Cloud's hosted router. Whisper small
here stands for a small local Whisper; the beam-5 row is its front end's default (the earlier "LiveKit default STT"
label had no source and is gone).

Paired difference against the 0.6B, percentage points [95 % CI] (positive = the baseline makes more errors):

| baseline − 0.6B | test-clean | test-other | AMI test | ICSI test | live, every word | FLEURS en test |
|---|---|---|---|---|---|---|
| Parakeet-TDT v3 | **−0.86 [−1.30, −0.43]** | **−2.31 [−3.02, −1.63]** | +0.44 [−0.89, +1.79] | **−2.69 [−3.12, −2.06]** | −0.51 [−1.50, +0.69] | **−1.05 [−1.86, −0.26]** |
| Whisper large-v3 | −0.65 [−1.24, +0.09] | **−2.16 [−2.97, −1.27]** | **+2.82 [+1.53, +3.39]** | **+2.99 [+1.86, +4.00]** | +1.14 [−0.46, +2.89] | **−1.67 [−2.49, −0.82]** |
| Whisper large-v3-turbo | −0.07 [−0.94, +1.17] | **−2.14 [−2.88, −1.40]** | **+2.72 [+2.13, +3.69]** | **+3.10 [+1.99, +4.72]** | +1.36 [−0.12, +2.82] | **−1.54 [−2.32, −0.83]** |
| Whisper small, beam 5 | +0.41 [−0.07, +0.86] | **+1.39 [+0.63, +2.22]** | **+3.03 [+1.99, +3.87]** | **+4.76 [+3.27, +6.52]** | **+2.26 [+0.49, +3.98]** | −0.23 [−1.06, +0.62] |
| audioforge 115M | −0.31 [−0.75, +0.15] | **+1.11 [+0.37, +1.91]** | **+8.24 [+5.09, +10.43]** | **+8.19 [+3.40, +12.02]** | **+8.79 [+7.13, +10.93]** | **+3.24 [+2.14, +4.32]** |

**Live calls with repetitions and false starts collapsed.** The TurnBench references are verbatim ("It's- it's very
scary"); Whisper writes clean text and drops repeats by design, so the verbatim column penalises it. The second
column applies one rule to references and every hypothesis (`final_scoring.disfl_norm`): drop hyphen-final
false-start fragments ("it's-"), Whisper-normalise, then collapse immediate repeats of 1-3-word n-grams ("i i i" →
"i"). **The README quotes the verbatim column (the corpus convention); this column is the sensitivity check, and any
"best on the user channel" claim must hold in both.**

| system | user channel, verbatim | user channel, disfluencies collapsed | every word, verbatim | every word, collapsed |
|---|---|---|---|---|
| audioforge 0.6B | **5.70** [4.39, 7.18] | **5.37** [3.81, 6.86] | 11.55 | 10.56 |
| audioforge 115M | 15.35 | 13.65 | 20.34 | 18.51 |
| Parakeet-TDT v3 | 7.68 | 7.68 | **11.04** | 10.53 |
| Whisper large-v3 | 8.45 | 5.82 | 12.69 | **10.04** |
| Whisper large-v3-turbo | 8.38 | **5.37** [3.63, 7.03] | 12.91 | 10.24 |
| Whisper small, beam 5 | 9.58 | 6.86 | 13.81 | 11.31 |
| Whisper small, beam 1 | 10.14 | 7.46 | 14.67 | 11.82 |

So on the user channel the 0.6B leads Parakeet-TDT either way, and ties Whisper large-v3-turbo once the reference
convention is neutralised. On AMI / ICSI the same rule moves every system by about a point and does not change the
order (not tabled).

**The 1.12 s final on FLEURS en** (`--final-chunk-ms 1120`: masked offline forward at att [70,13], equal to
streaming in 1120 ms chunks, research/DUAL_RATE.md; CPU 2 threads, greedy):
- 0.6B: 7.76 [6.09, 9.51], −0.26 [−0.76, +0.26] against its 160 ms pass (paired; not significant).
- 115M: 10.51 [8.61, 12.40], **−0.75 [−1.35, −0.13]** against its 160 ms pass.
- `--beam 8` on the 115M: −0.39 [−0.77, −0.03] against greedy.

`--beam 8` against greedy on the 115M: AMI test −1.53, ICSI test −0.41, live −1.73 (research/FIXALL.md step 5). It is
an option, not the default, because it adds 1.5 ms (CPU) / 2.2 ms (MPS) per 160 ms chunk. The 0.6B gains nothing
from a beam. The 1.12 s rows on the other test sets are in `runs/dual_rate.json` (their AMI / ICSI CIs are now
meeting resamples too).

**Streaming word latency (ours only).** Time from the end of a word to the word first appearing in the live
transcript, through the served engine on MPS, 12 two-minute windows of the AMI test meetings (`stt_latency.py`):
- 115M: p50 274 ms [250, 291], p95 571 ms.
- 0.6B: p50 291 ms [272, 311], p95 591 ms.

### Final transcript ready after you stop

Time from the moment the user stops talking to the turn's final transcript, for every system, and the WER of the
texts returned at that moment. Lower is better. ms [95 % CI over turns]; WER [CI over sessions].

| system | device | p50 ms | p95 ms | p50, turns < 2 s (9) | p50, 2-5 s (17) | p50, > 5 s (30) | WER of the timed texts % |
|---|---|---|---|---|---|---|---|
| **audioforge 115M**, `--final-chunk-ms 1120` | MPS | **25** [22, 28] | **48** [36, 66] | 26 | 27 | 24 | 14.2 [11.4, 18.0] |
| audioforge 0.6B, `--final-chunk-ms 1120` | MPS | 49 [46, 51] | 71 [59, 77] | 50 | 49 | 48 | 7.8 [5.7, 10.4] |
| NVIDIA Parakeet-TDT 0.6B v3 (offline) | MPS, fp32 | 232 [208, 316] | 639 [478, 997] | 147 | 193 | 367 | **7.2** [5.1, 10.2] |
| NVIDIA Parakeet-TDT 0.6B v3 (offline) | CPU, 2 threads | 341 [305, 375] | 617 [493, 885] | 257 | 300 | 411 | 7.2 [5.1, 10.2] |
| Whisper large-v3-turbo | MPS, fp16 | 282 [265, 305] | 498 [396, 1314] | 243 | 263 | 333 | 10.6 [7.7, 14.3] |
| Whisper large-v3 | MPS, fp16 | 576 [494, 711] | 1631 [1097, 4762] | 382 | 455 | 825 | 11.3 [8.5, 14.8] |
| Whisper small, beam 5 (faster-whisper defaults) | CPU, int8, 2 threads | 1804 [1699, 1955] | 3610 [2698, 5872] | 1532 | 1694 | 2160 | 9.9 [7.7, 13.3] |
| Whisper small, beam 1 | CPU, int8, 2 threads | 1424 [1377, 1468] | 2308 [1704, 3584] | 1306 | 1358 | 1526 | 12.0 [9.6, 15.0] |
| Whisper small, beam 5 (transformers) | MPS, fp16 | 1891 [1426, 2519] | 8056 [5571, 21783] | 564 | 1378 | 2965 | 11.0 [8.5, 14.5] |

On the same GPU, Parakeet-TDT's final text comes 232 ms after the turn end against 49 ms for the 0.6B (on CPU it was
341 ms; the first pass compared that CPU number with our GPU numbers). The 0.6B's timed texts are about as accurate
as Parakeet-TDT's (7.8 vs 7.2 %, CIs overlap); the 115M is the fastest but its flushed texts have twice the errors.
Whisper small through transformers on MPS is no faster than CTranslate2 on CPU at beam 5 (its beam search and
long-form path dominate). The Whisper small beam-5 CPU row was timed while one other agent's job was running on the
Mac (2 threads each); every other row ran alone.

- **Audio and turns.** The user channel of the 16 TurnBench clips of the live set: 56 labelled user turns
  (clips.json `user_turns`), 1.1-39.7 s long, p50 5.4 s, p95 21.6 s. Not used for any training or selection of the
  words path.
- **Clock start.** The labelled end of each user turn; every system is told the turn has ended at the same moment.
  End-of-turn timing is in "Turn taking".
- **Offline systems.** At the turn end the system gets the whole turn's audio, as an end-of-utterance STT does in
  Pipecat / LiveKit; the clock runs until the text comes back; batch 1; turns > 30 s use Whisper's long-form decoding.
- **audioforge.** The served single-mode engine with `--final-chunk-ms 1120` over the whole session in 20 ms blocks;
  the block holding a turn end is cut at the end sample; the clock covers processing that last piece, the slow pass's
  flush (`LookaheadStream.flush_view`) and decoding the turn's tokens.
- **Device.** Every row names its device. Each model was first timed on the device of the words table (Parakeet-TDT
  and Whisper small on CPU); since our cores run on the Mac GPU, Parakeet-TDT (fp32) and Whisper small (transformers
  fp16, beam 5: CTranslate2 has no MPS backend) were re-run on MPS too.
- **WER at the timed point.** The turn texts each system returned in this run, concatenated per session, against the
  user-channel reference (Whisper normalizer, verbatim). The flushed 1.12 s finals at the labelled turn end are about
  2 points worse than the same core's whole-session decode in the words table; the offline models lose little.
- Warm models (load and first-call compile excluded), one stream, one process per system through
  `scripts/dev/gate.sh`. Per-turn rows: `scratch/finallat/<system>.jsonl`.
- research/DUAL_RATE.md reports 9 / 22 ms (115M) and 36 / 48 ms (0.6B) for the same engine, timed after the engine's
  own `turn_end` decisions, where the slow pass often has its frames decoded already; the labelled-end numbers here are
  the ones to compare with the offline systems.


## Speech detection

AMI test and ICSI test meetings, 64 × 20 s windows each, 80 ms frames, label = anyone speaking.
- **Only frames inside the audio are scored, for every system.** Each 20 s window has 251 label frames and the last
  one (20.00-20.08 s) lies past the end of the audio; the first pass scored it, which was a forced miss for Silero, TEN
  VAD and pyannote (they produce nothing past 20.00 s). Dropping it moves those three by about +0.002 F1, +0.0035 AUC
  and −0.3 points of missed speech; MarbleNet and our heads move by at most 0.0003 F1.
- Baselines are max-pooled onto the 80 ms grid. CI: 1000 resamples of meetings.
- "Miss" = speech frames missed at the threshold where 7.5 % of non-speech frames are false alarms.
- "Onset lag" = from a labelled speech onset (after ≥ 400 ms of silence) to the first frame > 0.5, on the frame grid.
  It does not include each model's own look-ahead (ours: 80 ms right context).
- Our heads were trained on AMI and ICSI train meetings and selected on held-out train meetings. Every ICSI test
  speaker also speaks in the ICSI train meetings of the shipped heads, so **the ICSI columns of our rows are the heads
  retrained with those speakers unseen** (all 10 ICSI training meetings holding any of the 13 test speakers removed,
  same recipe, selected on held-out data; plans/fixwave/turn_clean.md section 1, `runs/turn_clean.json` speech; not
  shipped, `vad_speakers_unseen` in the json). The shipped heads' ICSI row is in Appendix A. AMI columns: shipped
  heads (no AMI test speaker is in any training meeting).

| system | AMI test F1 @0.5 | AMI AUC | AMI miss @7.5 % FA | ICSI test F1 (ours: speakers unseen in training) | ICSI AUC | ICSI miss | onset lag p50 / p90 (AMI) | live-capable |
|---|---|---|---|---|---|---|---|---|
| audioforge 115M speech head (v0.4; ICSI: retrained, speakers unseen) | 0.959 [0.942, 0.976] | 0.966 | 12.5 % | **0.930** [0.916, 0.946] | 0.941 | 16.0 % | 0 / 160 ms | yes (80 ms right context) |
| audioforge 0.6B speech head (v0.3 = v0.4; ICSI: retrained, speakers unseen) | 0.957 [0.943, 0.972] | 0.967 | 12.6 % | 0.922 [0.897, 0.941] | **0.945** | **15.5 %** | 0 / 160 ms | yes (80 ms right context) |
| Silero VAD v5 (Pipecat / LiveKit) | 0.901 [0.866, 0.949] | 0.961 | 13.0 % | 0.924 [0.906, 0.947] | 0.926 | 26.4 % | 80 / 320 ms | yes |
| TEN VAD | 0.930 [0.902, 0.959] | 0.955 | 13.4 % | 0.922 [0.902, 0.941] | 0.932 | 20.5 % | 80 / 80 ms | yes |
| NVIDIA MarbleNet v2 frame VAD | 0.941 [0.916, 0.968] | 0.967 | 9.6 % | 0.914 [0.897, 0.934] | 0.925 | 24.6 % | 0 / 80 ms | offline here (whole-window conv) |
| pyannote segmentation-3.0 | **0.977** [0.968, 0.987] | **0.987** | **3.8 %** | 0.893 [0.874, 0.909] | 0.932 | 20.8 % | 0 / 0 ms | no (10 s windows, sees the future) |

Paired differences [95 % CI, meeting resamples], F1 and AUC (negative = the baseline is worse):
- TEN VAD − ours: AMI F1 −0.040 to −0.017 (115M), −0.040 to −0.014 (0.6B); ICSI (speakers unseen) F1 −0.015 to
  −0.001 (115M) and −0.009 to +0.005 (0.6B), AUC −0.036 to −0.003 and −0.055 to −0.001.
- Silero − ours: AMI F1 −0.076 to −0.027 (115M; AUC −0.012 to +0.004); ICSI F1 −0.010 to +0.001 (115M), −0.015 to
  +0.009 (0.6B), AUC −0.061 to −0.004 and −0.080 to −0.002.
- MarbleNet − ours: AMI F1 −0.029 to −0.006 (115M) and **−0.032 to +0.000 (0.6B)**; AMI AUC −0.007 to +0.014 and
  −0.009 to +0.013 (a tie); ICSI F1 −0.021 to +0.000 (0.6B), AUC −0.055 to −0.013.
- pyannote − 0.6B: AMI F1 +0.005 to +0.031; ICSI F1 −0.035 to −0.023.

So on AMI the F1 lead over the live VADs rests on the 0.5 threshold, to which our heads are calibrated by training
on this label convention: on the threshold-free AUC MarbleNet ties our heads, and it misses less speech at 7.5 % false
alarms. On ICSI, with the test speakers unseen, ours lead on AUC and missed speech and tie the best live VADs on F1.

The previous heads (115M v0.3 / 0.6B v0.2, AMI-only training) score AMI test 0.957 / 0.958 and ICSI test 0.886 / 0.890 F1 on the same
windows (same frame rule; `vad_previous_heads` in the json; their ICSI training also held the test speakers).


## Turn taking

The test data:
- **Assistant clips**: smart-turn v3.2 **test**, 399 clips (175 complete, 224 cut mid-utterance), each padded with 3 s
  of silence. Our classifiers train on smart-turn's train split; the test split was never used to train or select the
  0.6B's heads or constants (it was for the 115M's: Appendix A).
- **AMI test**: 200 turns from the AMI test meetings (eot-bench v2 cut).

How the rows are scored (one scorer for every system):
- Latency is from the audible end of speech to the turn end, plus the measured compute.
- "False fire" = a turn end on an incomplete clip within W seconds of its audible end (the speaker has not finished).
- "Interrupt" = a turn end inside the user's turn. "Missed" = no turn end before the next speaker or within 6 s.
- CIs: 1000 resamples of clips / sessions.

**Which false-fire window W.** Every system has a fallback timer that ends the turn when no "complete" decision
comes: Pipecat 1.12 `STOP_SECS` = 3.0 s after its VAD stop, LiveKit 1.8.3 `max_endpointing_delay` = 3.0 s, ours
`assistant` 2.96 s (115M) / 3.44 s (0.6B). The first pass used W = 3.0 s, right on the two frameworks' timers: 62 % of
Pipecat's and 73 % of LiveKit's first turn ends on incomplete clips fall within ±0.25 s of 3.0 s, so whether a
fallback counted as a false fire was decided by tens of milliseconds. **Rule: the headline W is the largest of 2.0 /
2.5 / 3.0 / 3.5 s that is more than 0.25 s from every system's nominal fallback timer → W = 2.5 s** (at most 5.3 % of
any system's first turn ends lie within ±0.25 s of it). The 3.0 s column is kept for continuity, and the full curve
is below.

| system (assistant clips) | accuracy, W = 2.5 s | false fire, W = 2.5 s | accuracy, W = 3.0 s | false fire, W = 3.0 s | p50 / p95 ms | missed |
|---|---|---|---|---|---|---|
| **audioforge 0.6B `assistant`** ⚑ | **97.7 %** [96.2, 99.0] | **0.9 %** [0.0, 2.2] | **96.5 %** [94.5, 98.2] | **3.1 %** | 351 / 763 | 4.0 % |
| audioforge 0.6B `balanced` (default) | 42.6 % | 100 % | 42.6 % | 100 % | 639 / 799 | 2.9 % |
| audioforge 0.6B `fast` | 40.1 % | 100 % | 40.1 % | 100 % | 414 / 607 | 8.0 % |
| audioforge 115M `balanced` (default; re-picked on held-out data = shipped) | 43.9 % | 100 % | 43.9 % | 100 % | 1226 / 1587 | 0.0 % |
| audioforge 115M `fast` (re-picked on held-out data = shipped) | 41.6 % | 100 % | 41.6 % | 100 % | 461 / 698 | 4.0 % |
| audioforge 115M `assistant`, held-out pick (classifier c5s1; candidate heads v0.5, **not shipped**) | 96.0 % [94.0, 97.7] | 2.7 % | 95.7 % | 3.1 % | 381 / 929 | 5.1 % |
| Pipecat smart-turn v3.2 + Silero (defaults) | 75.9 % [71.7, 80.2] | 29.0 % [23.0, 34.7] | 69.7 % | 40.2 % | **211** / 242 | 13.1 % |
| LiveKit Agents 1.8 EnglishModel + Silero | 85.2 % [82.0, 88.5] | 22.3 % [17.1, 27.6] | 72.7 % | 44.6 % | 547 / 3016 | 4.6 % |
| NVIDIA Parakeet-Realtime-EOU 120M | 49.4 % [44.9, 54.6] | 89.7 % | 48.4 % | 91.5 % | 462 / 1210 | 0.6 % |
| *smart-turn v3.2 classifier alone (upper bound: one call per whole clip, no timing)* | – | – | *97.0 %* | *2.7 %* | – | *3.4 %* |
| ⚠ audioforge 115M `assistant` as shipped (classifier c5) | Appendix A | | | | | |

**Held-out re-pick of the turn presets** (plans/fixwave/turn_clean.md, `runs/turn_clean.json`, merged as `turn_clean`
in the json). The turn-clean work re-ran the preset selection with a rule written before scoring, on held-out audio
only (none of the 399 test clips, the AMI test meetings, TurnBench or the evaluation calls), then scored the picks once.
- 115M `balanced` and `fast`: the held-out pick **is** the shipped rule, so their test rows are clean.
- 115M `assistant`: the held-out pick differs. The best servable one (classifier c5s1, k 2, P 0.95) is in the table
  above; it ships only in the candidate heads v0.5 (`assets/served_heads_v0.5_candidate.pt`), not the default. The
  shipped c5 rule stays in Appendix A.
- 0.6B: the held-out picks differ from the shipped v0.4 rules. Scored once on the test rows (not shipped): `balanced`
  (per-frame head) AMI test 1379 ms / 14.5 % interrupt / 26.0 % missed against shipped 1498 / 10.0 / 33.5; `fast`
  assistant-clip p50 335 ms, AMI 1459 / 11.0 / 29.0; `assistant` 97.7 % / 319 ms / 0.9 % at 2.5 s (same accuracy as
  shipped, 32 ms faster), AMI 1298 / 28.5 / 29.0.

Accuracy / false fire against the window (every system; the 115M's shipped `assistant` in Appendix A):

| system | W = 2.0 s | 2.5 s | 3.0 s | 3.5 s |
|---|---|---|---|---|
| audioforge 0.6B `assistant` | 97.7 / 0.9 | 97.7 / 0.9 | 96.5 / 3.1 | 96.5 / 3.1 |
| Pipecat smart-turn v3.2 + Silero | 77.9 / 25.4 | 75.9 / 29.0 | 69.7 / 40.2 | 44.9 / 84.4 |
| LiveKit Agents 1.8 + Silero | 88.2 / 17.0 | 85.2 / 22.3 | 72.7 / 44.6 | 46.4 / 91.5 |
| Parakeet-Realtime-EOU | 51.4 / 86.2 | 49.4 / 89.7 | 48.4 / 91.5 | 47.1 / 93.8 |
| audioforge 115M `assistant`, held-out pick (not shipped) | 96.5 / 1.8 | 96.0 / 2.7 | 95.7 / 3.1 | 92.0 / 9.8 |

The order (0.6B > LiveKit > Pipecat > Parakeet-EOU) holds at every window; the size of the gap does not: at 3.5 s
the frameworks' fallbacks all count as false fires, at 2.0 s almost none do.

`balanced` and `fast` are conversation presets: they end a turn after 0.6-0.7 s of quiet, so on clips that stop
mid-sentence they always fire (100 % false fire). `assistant` waits up to 3.4 s for a classifier "complete".

**AMI test meetings** (200 turns). Our engine is enrolled with the user's 5 s voice print, and every preset has the
"others" path (end the turn when the user is silent while someone else talks); the baselines see the meeting without
knowing who the user is. The "no print" rows run our cores on the same sessions with no print armed (no TS-VAD
"others" path), which measures what knowing the user's voice is worth.

| system (AMI test, 200 turns) | p50 / p95 ms | interrupt | missed |
|---|---|---|---|
| audioforge 115M `balanced` (default), with print | 1527 [1327, 1806] / 4439 | 15.5 % [10.5, 20.5] | 36.0 % [30.0, 42.0] |
| audioforge 115M `balanced`, **no print** | 1508 [1327, 1967] / 4773 | 6.5 % [3.5, 10.0] | 74.0 % [66.5, 80.0] |
| audioforge 115M `fast`, with print / no print | 1246 / 1206 | 20.5 % / 11.5 % | 37.5 % / 64.0 % |
| audioforge 115M `assistant`, with print / no print | 1327 / 1447 | 23.5 % / 24.5 % | 40.0 % / 64.0 % |
| audioforge 0.6B `balanced` (default), with print | 1498 [1418, 1657] / 4075 | 10.0 % [6.0, 14.5] | 33.5 % [27.0, 40.0] |
| audioforge 0.6B `balanced`, **no print** | 1499 [1339, 1817] / 4650 | 5.5 % [2.5, 9.0] | 73.5 % [67.0, 78.5] |
| audioforge 0.6B `fast`, with print / no print | 1338 / 859 | 16.0 % / 12.5 % | **27.5 %** / 65.5 % |
| audioforge 0.6B `assistant` ⚑, with print / no print | 1497 / 1618 | 22.5 % / 23.0 % | 34.5 % / 63.0 % |
| Pipecat smart-turn v3.2 + Silero | 752 [496, 1296] / 4419 | 39.5 % [33.0, 47.0] | 53.0 % [45.5, 60.0] |
| LiveKit Agents 1.8 EnglishModel + Silero | **745** [641, 1330] / 4640 | 13.0 % [8.5, 18.0] | 72.0 % [66.0, 78.0] |
| NVIDIA Parakeet-Realtime-EOU 120M | 1251 / 4928 | **4.5 %** | 86.0 % |

Without the print our cores behave like the Silero-driven baselines: they rarely interrupt but miss about three turns
in four, because other people keep talking after the user stops and nothing tells the engine those voices are not
the user's. With the print the "others" path ends those turns, and the misses drop to 34-36 %. The print costs the
user a 5 s enrolment; the baselines have no such option.

On AMI test meetings no system has both few interruptions and few misses: Pipecat answers fastest but interrupts on
40 % of turns; LiveKit and Parakeet-EOU interrupt rarely but miss 72-86 % of turns; ours sits between them.

The 0.6B's heads v0.4 change only `assistant` (its own classifier, trained with real two-party channels, asked at
160 ms of quiet): before (v0.3) 93.0 % [90.5, 95.5] / 379 ms / 4.5 % at W = 3.0 s, AMI test 1619 / 4.0 % / 53.0 %
(`superseded` in the json). `balanced` / `fast` make the same decisions as v0.3. ⚑ The 0.6B's `assistant` v0.4 was
picked on held-out data only, but it reuses the 115M's classifier recipe and distillation teacher, which were chosen
partly on these clips (leakage item 5, weak); its `balanced` / `fast` were chosen between two candidates after both
had been scored on the evaluation sets (item 4).

**Two-party calls: no labelled public test split.** TurnBench publishes only its dev split with labels. The call rows
(16 TurnBench dev clips + 16 oto conversations, user channel, 109 turn ends) are therefore not a test split; the
0.6B's preset choice also looked at them (item 4). Kept out of the headline:

| system (calls, not a test split) | p50 / p95 ms | interrupt | missed |
|---|---|---|---|
| audioforge 0.6B `balanced` ⚑ | 729 / 2034 | 19.3 % [12.1, 26.4] | 10.1 % [4.9, 16.0] |
| audioforge 0.6B `fast` ⚑ | 492 / 1974 | 26.6 % | 11.0 % |
| Pipecat smart-turn v3.2 + Silero | 237 / 3217 | 35.8 % [25.7, 46.6] | 24.8 % [16.1, 33.6] |
| LiveKit Agents 1.8 + Silero | 567 / 3127 | 26.6 % [20.2, 34.0] | 22.9 % [15.1, 30.2] |
| NVIDIA Parakeet-Realtime-EOU | 1277 / 5362 | 7.3 % | 30.3 % |
| audioforge 115M `balanced` (held-out pick = shipped) | 955 / 1918 | 20.2 % [13.1, 29.2] | 7.3 % [3.5, 12.0] |
| audioforge 115M `fast` (held-out pick = shipped) | 547 / 1861 | 24.8 % | 5.5 % |


## Speaker tracking ("your words only")

The windows are the eot-bench v2 windows of the AMI **test** meetings IS1009b / ES2004b / TS3003b / EN2002a (941
windows). The ICSI test rows are in Appendix A: every ICSI test speaker is also in the ICSI train meetings our heads
learned from.
- The target is the main speaker of each window, with a 5 s voice print cut from elsewhere in the same meeting.
- **tWER** = WER of the target's words only. Streaming words are kept where the system's target mask is on, with the
  same keep rule for every arm. The diarizer arms use the **0.6B's words**, like our 0.6B row (their 115M-words arms
  are in the json: Nemotron-3 68.6, pyannote 77.1).
- **Target miss + false-alarm rate** = (missed target frames + false target frames) / target frames, at 80 ms, no
  collar. It is not the standard DER (no speaker-confusion term, one speaker); the first pass called it "target DER".
- The open diarizers' columns are bound to the same print by `tsvad.vp_follow`; the better of two binders (served
  speaker head, TitaNet-L) is shown. The **oracle-binding** rows pick, per window, the diarizer column that best
  matches the target using the labels: an upper bound of what any binder could get from that diarizer.
- CI: 1000 resamples of the 4 meetings (coarse).

| system | AMI test tWER % [meeting CI] | AMI test target miss + FA % | AMI test tracking F1 |
|---|---|---|---|
| audioforge 115M (115M words) | 51.5 [41.1, 58.8] | 37.3 | 0.811 |
| **audioforge 0.6B (0.6B words)** | **47.1** [32.3, 57.7] | **35.8** | **0.829** |
| Nemotron-3-Diarization + print, 0.6B words | 64.4 [47.5, 76.9] | 62.9 | 0.676 |
| pyannote speaker-diarization-3.1 + print, 0.6B words | 77.8 [52.6, 96.2] | 64.7 | 0.734 |
| *Nemotron-3, oracle binding (upper bound)* | – (not run) | *39.7* | *0.795* |
| *pyannote 3.1, oracle binding (upper bound)* | – | *58.5* | *0.767* |
| no filter (115M / 0.6B words) | 94.2 / 102.8 | – | – |
| oracle filter (115M / 0.6B words) | 43.0 / 31.7 | – | – |

Most of the margin over Nemotron-3 is the binding step: with the column chosen by the labels it reaches F1 0.795,
close to our 0.811 / 0.829; bound by the print through our binder it gets 0.676. Ours still lead in both cases.

**Standard DER of the diarizers** (md-eval style: missed + false alarm + speaker confusion over reference speech, every
speaker of the window, 0.25 s collar on each side of every reference boundary, overlap scored; pyannote.metrics on
the cached tracks, reference = the corpus speaker intervals). It is not defined for our tracker, which outputs one
target track rather than a diarization.

| diarizer (AMI test, 941 windows, every speaker) | DER % [meeting CI] | missed | false alarm | confusion |
|---|---|---|---|---|
| Nemotron-3-Diarization (`low_latency_032`, streaming) | 25.4 [15.6, 31.0] | 22.7 | 0.8 | 1.9 |
| pyannote speaker-diarization-3.1 (offline, whole window) | 17.2 [11.5, 21.3] | 8.4 | 2.1 | 6.7 |

How these rows were produced: the whole pipeline was run on the AMI test meetings' windows (words of both cores,
prints, Nemotron-3 streaming tracks, pyannote 3.1, TS-VAD tracks and frame counts). Our TS-VAD heads were trained on
AMI / ICSI train meetings; pyannote runs offline on the whole window, which helps it; Nemotron-3 has AMI in its
training data.

**Within-meeting speaker EER**, AMI test meetings (200 single-speaker segments, 6941 within-meeting pairs; CI: 1000
resamples of the 4 meetings, each pooling its pairs):

| embedder | AMI test EER % |
|---|---|
| audioforge 115M speaker head (block 4) | 5.0 [3.5, 12.6] |
| audioforge 0.6B speaker head (block 5) | 3.8 [2.4, 11.8] |
| NVIDIA TitaNet-L (the teacher) | **1.9** [0.7, 8.8] |
| pyannote WeSpeaker ResNet34 (in pyannote 3.1) | 2.0 [0.3, 8.1] |

The dedicated embedders are better voice matchers than our heads. The tracker still wins on tWER and tracking F1,
because the TS-VAD head is trained to track the print frame by frame, not only to embed. (The first pass resampled
segments; the meeting CIs are much wider because one meeting dominates the errors.)

## Language ID: FLEURS-17 test (2550 clips), accuracy % [95 % CI]

| system | 2 s from speech onset | full clip |
|---|---|---|
| Whisper large-v3 | **95.9** [95.2, 96.7] | 99.5 [99.2, 99.7] |
| Whisper large-v3-turbo | 95.2 [94.3, 96.0] | 99.3 [99.0, 99.7] |
| NVIDIA AmberNet (the teacher; reused, `data/lid/preds/ambernet`) | 95.1 [94.3, 95.9] | **99.5** [99.3, 99.8] |
| audioforge 0.6B LID head v2 ⚑ | 92.7 [91.7, 93.7] | 98.6 [98.1, 99.0] |
| audioforge 115M LID head v2 ⚑ | 92.4 [91.5, 93.5] | 98.2 [97.7, 98.8] |
| Whisper small | 90.6 [89.4, 91.7] | 99.1 [98.7, 99.4] |
| *previous heads (same clip path): 115M `lid_distill.pt` / 0.6B `lid_0p6b.pt`* | *90.9 / 87.6* | *97.8 / 95.5* |

Every system's posterior is restricted to the 17 test languages (the `lid.py` protocol). Our heads are VAD-gated, as
served. Weights, steps, hidden size and teacher were selected on FLEURS dev only. ⚑ The encoder blocks the heads read
(115M 8-12) were chosen earlier with a linear probe scored on FLEURS test (leakage item 8; a later dev-only sweep
agrees for the 0.6B; probably well under a point).

## Cost on this Mac (Apple M5)

The full single-mode engine: ASR, speech detector, VAD, turn, TS-VAD and LID, on the bundled 16 s call, best of 3.
Streams = interleaved sessions on 40 s AMI windows, real time while p95 of the summed block compute stays under 160 ms
(`mps_115m.py` protocols).

| core | ms per 160 ms chunk, MPS (p50 / p95) | CPU 2 threads | real-time streams, MPS | real-time streams, CPU | peak RSS |
|---|---|---|---|---|---|
| audioforge 115M (`stage1_served_v4.afm` + `lid_115m_v2.pt`) | 28.6 / 30.7 | 30.4 / 33.0 | 5 | 4 | 1.2 GB (+0.5 GB MPS) |
| audioforge 0.6B (measured on `served_0p6b_v0.3.afm` + `lid_0p6b_v2.pt`) | 42.9 / 49.0 | 97.1 / 105.1 | 3 | 1 | 4.9 GB (+3.3 GB MPS) |

The 0.6B's heads v0.4 add +0.4 ms (MPS) and +0.2-0.6 ms (CPU) per chunk in back-to-back runs against v0.3, with the
same 3 real-time MPS streams (research/TURN_DATA.md). The LID head v2 adds nothing measurable (±0.2 ms).

**Model sizes** (parameters counted from the weight files of the shipped builds):

| model | params |
|---|---|
| audioforge 115M (whole served model `stage1_served_v4.afm`) | 122.1 M (+ LID head v2 2.4 M) |
| audioforge 0.6B (whole served model `served_0p6b_v0.4.afm`) | 625.2 M (+ LID head v2 2.9 M) |
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

(The first pass counted the 0.6B on its v0.2 build, 622.6 M; v0.4 adds the two turn heads.)


## What changed with the test splits (first pass on dev → test-split pass, before the fix wave)

Kept as history; the numbers in the right column are the test-split pass before the corrections above (the
speech-detection and speaker-EER cells there predate the frame and CI fixes). ⚠ The first-pass AMI turn row was on the
AMI **dev** meetings, the set the 115M's turn constants were tuned on (leakage item 2), and the ICSI speaker rows are
seen-speaker numbers (item 3). Each change mixes two things: the split moved (dev → test), and for the speech detector and LID the heads changed
too. The full list (182 image numbers) is the diff of `numbers_final.json`. The headline rows:

| row | first pass (dev) | this pass (test) | why |
|---|---|---|---|
| WER AMI: 0.6B / 115M / Parakeet-TDT / Whisper large-v3 | 10.4 / 20.6 / 9.5 / 12.4 | 7.9 / 16.1 / 8.3 / 10.7 | split (the test meetings are easier for every system); the 0.6B now leads (TDT − 0.6B +0.44, CI includes 0) |
| WER ICSI: 0.6B / 115M / TDT / Whisper large-v3 | 13.6 / 26.2 / 10.4 / 12.8 | 10.3 / 18.4 / 7.6 / 13.2 | split; the 0.6B now beats Whisper large-v3 (it lost on dev), TDT still leads |
| WER LibriSpeech test-clean: 0.6B / TDT / Whisper large-v3 | 2.1 / 1.9 / 1.4 (first 200 utterances, 3 speakers) | 2.7 / 1.8 / 2.0 (300 random utterances) | subset: the old "first 200" covered only the first few speakers |
| WER LibriSpeech test-other | – | 0.6B 5.7, TDT 3.4, Whisper large-v3 3.6 | new |
| speech F1 AMI: 115M / 0.6B / best live baseline | 0.951 / 0.951 / 0.937 (MarbleNet) | 0.959 / 0.957 / 0.941 (MarbleNet) | split + new heads |
| speech F1 ICSI ⚠ (shipped heads, seen speakers): 115M / 0.6B / best baseline | 0.898 / 0.906 / 0.935 (TEN) | **0.938 / 0.940** / 0.922 (Silero) | new heads (+0.05) and split (TEN 0.935 → 0.920): the row flips from a loss to a win |
| turn, AMI test `balanced` p50 / interrupt / missed: 115M; 0.6B | ⚠ 1326 / 10.5 / 33.5; 1180 / 11.0 / 33.0 (AMI dev: tuning set) | 1527 / 15.5 / 36.0; 1497 / 10.0 / 33.5 | split; baselines moved too (Pipecat interrupts 27.5 → 39.5 %) |
| turn, assistant clips | unchanged (already smart-turn **test**) | unchanged | – |
| tWER AMI: 0.6B / 115M / Nemotron-3 (0.6B words) / pyannote | 63.2 / 62.1 / 71.6 / 85.3 | 47.1 / 51.5 / 64.4 / 77.1 | split (test windows, new pipeline run) |
| tWER ICSI ⚠: 0.6B / 115M / Nemotron-3 / pyannote | 31.7 / 37.1 / 60.0 / 74.8 (dev + test windows) | 29.0 / 34.9 / 59.4 / 73.6 (test windows only) | split |
| speaker EER AMI: ours 115M / 0.6B / TitaNet-L / WeSpeaker | 19.8 / 13.6 / 12.0 / 11.9 | 5.0 / 3.8 / 1.9 / 2.0 | split (fewer speakers per test meeting); same order |
| speaker EER ICSI ⚠ | 7.0 / 3.6 / 2.1 / 1.0 | 2.5 / 1.9 / 1.2 / 0.9 | split |
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
    served policy twin (`core_0p6b_turn.served_rules` / `ev_score`). The twin reproduces the served turn_ends. The
    calls and assistant-clip dumps were made with `stage1_served_v3.afm`, the AMI test dumps with v4: the turn path
    reads `heads.vad` / `turn` / `turn_seg`, bit-identical in v3 and v4.
- **audioforge 0.6B**: `served_0p6b_v0.4.afm` (`assets/served_heads_0p6b_v0.4.pt`: v0.3 + `turn_vad` / `turn_seg_a` for
  the `assistant` preset; v0.3 = v0.2 + the `speech` head) +
  `assets/lid_0p6b_v2.pt` on nvidia/nemotron-speech-streaming-en-0.6b, NVIDIA Open Model License. Same harnesses as
  the 115M. Words, speech detection, speaker EER, LID and cost were run on `served_0p6b_v0.3.afm`: every v0.3 tensor
  is bit-identical in v0.4, which only adds `turn_vad` / `turn_seg_a` and changes the `assistant` preset (checked on
  the heads files). Its turn rows are scored from dumps of `served_0p6b_v0.4.afm` (`scratch/turndata/fc_v04/eot`).
  The turn rows of both cores read `heads.vad`, unchanged from v0.3 / v0.2 (identical turn ends, research/FIXALL.md);
  the 0.6B's `assistant` preset reads its own classifier (heads v0.4, research/TURN_DATA.md; served == offline 32 / 32).
- **Whisper small**: faster-whisper 1.2.1 int8, CPU 2 threads, English. Headline row: faster-whisper's own
  `transcribe()` defaults (beam 5, best_of 5, temperature fallback, previous-text conditioning), as Pipecat's
  `WhisperSTTService` calls it; second row: beam 1, no timestamps, no previous-text conditioning. On MPS (final-text
  latency only): transformers fp16, beam 5. MIT.
- **Whisper large-v3 / large-v3-turbo**: transformers 5.17, fp16 on MPS, greedy, English. Clips over 30 s use
  sequential long-form decoding. MIT. For LID, the language-token posterior after `<|startoftranscript|>`.
- **Parakeet-TDT 0.6B v3**: imported into our PyTorch modules (`nemo_import`), offline, greedy TDT, CPU 2 threads
  (words table); CPU and fp32 MPS (final-text latency). CC-BY-4.0.
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
- **tWER of the oracle-binding diarizer arms**: only their frame metrics exist.
- **Standard DER of our tracker**: not defined (one target track, not a diarization).
- **Baselines' per-stream cost**: offline per-utterance / per-window systems; their decode RTFx is in the json.
- **Full LibriSpeech test sets**: 300 random utterances per split (every system, same audio).

## Reproduce

No environment variable is needed: the defaults are the shipped builds (115M `runs/stage1_served_v4.afm`, 0.6B
`served_0p6b_v0.4.afm`, LID heads `assets/lid_*_v2.pt`) and the AMI turn rows use the AMI test meetings.

    P="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_compare.py"
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_scoring.py meta     # meetings, window lengths
    $P asr --system <S> --set <ls_clean|ls_other|ami_eval|icsi_eval|live|fleurs_en> [--device mps]
    #   S includes whisper_small_b5 (beam 5) and, on fleurs_en, core_115m_f1120 | core_0p6b_f1120
    $P vad --system <S> | spkeer --system <S> | lid --system <S> --device mps
    $P eotdump --sys 115m|0p6b --which calls|asst --device mps           # + eot_latency.py prepare_ami / dump / baselines
    $P eotdump --sys 115m|0p6b --which calls --noprint --device mps      # AMI test, no voice print
    $P eou --which calls|asst --device mps
    $P sttlat --sys 115m|0p6b --device mps | cost --sys 115m|0p6b --sub engine|streams --device mps|cpu | params
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/final_scoring.py stdder   # diarizer DER
    scripts/research/fixall.py twsub --set icsi|ami_eval; frsub; frami                 # target-speaker test rows
    PYTHONPATH=. .venv/bin/python scripts/research/final_latency.py run --system <S>; ... final_latency.py report
    PYTHONPATH=. .venv/bin/python scripts/research/dual_rate.py score; ... dual_rate.py report
    $P report; python3 demo/images/redesign/export_final.py

Scratch: `/Volumes/ExternalSSD/nvidia-audio-models/scratch/final_compare/` (meta: `meta/`; the previous heads' VAD /
LID outputs: `scratch/fixall/fc_old`; the 0.6B v0.4 turn dumps: `scratch/turndata/fc_v04/eot`).

## What the fix wave changed (2026-10-03)

The audits (plans/audit/metrics_001.md, fairness_001.md, leakage_001.md) re-derived every number from the saved
outputs; they matched, apart from one scoring bug and several design and labelling problems. What was fixed and how
much the headline numbers moved:

| fix (audit finding) | row | before | after |
|---|---|---|---|
| VAD: the past-end frame dropped for every system (metrics 1) | speech F1 AMI test: Silero / TEN / pyannote / MarbleNet | 0.899 / 0.928 / 0.975 / 0.941 | 0.901 / 0.930 / 0.977 / 0.941 |
| | speech F1 ICSI test: Silero / TEN / pyannote | 0.922 / 0.920 / 0.891 | 0.924 / 0.922 / 0.893 |
| | AUC AMI test: Silero / TEN / pyannote | 0.958 / 0.952 / 0.983 | 0.961 / 0.955 / 0.987 |
| | missed speech @7.5 % FA, AMI test: Silero / TEN / MarbleNet / pyannote | 13.3 / 13.7 / 9.7 / 4.1 % | 13.0 / 13.4 / 9.6 / 3.8 % |
| | ours (115M / 0.6B) F1 AMI, ICSI | 0.959 / 0.957, 0.938 / 0.940 | unchanged by the frame fix (≤ 0.0003) |
| Speakers-unseen rule (leakage 3) | ours ICSI speech F1 / AUC / miss, 115M; 0.6B | 0.938 / 0.951 / 12.7 %; 0.940 / 0.956 / 12.0 % (shipped heads, seen speakers) | 0.930 / 0.941 / 16.0 %; 0.922 / 0.945 / 15.5 % (retrained, speakers unseen; shipped rows to Appendix A) |
| | ranking | – | ICSI dev only: Silero now ahead of MarbleNet (0.9305 vs 0.9289); no test ranking changes |
| Turn: headline false-fire window 2.5 s instead of 3.0 s (metrics 2) | assistant accuracy: 0.6B / Pipecat / LiveKit / Parakeet-EOU | 96.5 / 69.7 / 72.7 / 48.4 % | 97.7 / 75.9 / 85.2 / 49.4 % |
| | assistant false fire: same order | 3.1 / 40.2 / 44.6 / 91.5 % | 0.9 / 29.0 / 22.3 / 89.7 % |
| CIs over meetings (metrics 3) | WER AMI test, 0.6B / Parakeet-TDT | [6.80, 9.07] / [7.02, 9.78] | [7.69, 8.04] / [6.80, 9.83] |
| | WER ICSI test, 0.6B | [8.56, 12.49] | [7.75, 13.04] |
| | speaker EER AMI test, 115M / 0.6B / TitaNet-L | [3.2, 6.9] / [2.2, 5.4] / [0.7, 3.7] | [3.5, 12.6] / [2.4, 11.8] / [0.7, 8.8] |
| | speech F1 AMI test, 0.6B; MarbleNet − 0.6B | [0.948, 0.965]; −0.028 to −0.006 | [0.943, 0.972]; −0.032 to +0.000 (now touches 0) |
| Reproducibility (metrics 4, fairness 10) | 0.6B turn row source | pasted from turn_data.json | scored from the v0.4 dumps by `report` (same numbers) |
| | parameters: 115M / 0.6B | 122.1 M (v3) / 622.5 M (v0.2) | 122.1 M (v4) / 625.2 M (v0.4) |
| Final-text latency on MPS (fairness 2) | Parakeet-TDT p50 / p95 | 341 / 617 ms (CPU) | 232 / 639 ms (MPS; CPU row kept) |
| | WER of the timed texts | not shown | 0.6B 7.8, Parakeet-TDT 7.2, 115M 14.2, Whisper turbo 10.6 % |
| Whisper small at beam 5 (fairness 9) | WER AMI / ICSI / live user / test-clean / FLEURS | 11.9 / 15.0 / 10.1 / 3.7 / 8.0 (beam 1) | 10.9 / 15.0 / 9.6 / 3.1 / 7.8 (beam 1 row kept) |
| | final-text latency p50 (CPU) | 1424 ms (beam 1) | 1804 ms (beam 5); 1891 ms on MPS |
| | label | "LiveKit default STT" | removed; Pipecat / LiveKit defaults stated |
| Speaker tracking (fairness 7, metrics 6) | diarizer tWER, 0.6B words: Nemotron-3 / pyannote, AMI | 68.6 / 77.1 (115M words in the image) | 64.4 / 77.8 |
| | ICSI (Appendix A) | 64.0 / 73.6 | 59.4 / 69.0 |
| | oracle-binding rows | in the json only | Nemotron-3 F1 0.795, miss + FA 39.7 % (AMI) |
| | metric name | "target DER" | "target miss + FA rate"; standard DER of the diarizers added (AMI: Nemotron-3 25.4, pyannote 17.2 %) |
| Live-call disfluencies (fairness 4) | user channel, collapsed: 0.6B / Whisper turbo / large-v3 / Parakeet-TDT | – | 5.4 / 5.4 / 5.8 / 7.7 % (verbatim 5.7 / 8.4 / 8.5 / 7.7) |
| AMI turns without the voice print (fairness 3) | `balanced` missed, 115M / 0.6B | not measured | 74.0 / 73.5 % (with print 36.0 / 33.5) |
| Leakage (leakage 1-3, 8) | 115M `assistant` 92.7 % row, ICSI speaker rows | headline | Appendix A; 115M `balanced` / `fast` cleared by the held-out re-pick |

## Appendix A: rows flagged by the leakage audit

These rows are measured correctly, but the audio they are measured on was also used to tune or select what produced
them (⚠ contaminated), or overlaps the training data by speaker. They stay here, out of the headline tables, until
held-out replacements exist (the turn-clean work, plans/fixwave/turn_clean.md, re-picks the 115M's turn constants and
classifier on held-out audio only).

**A.1 ⚠ The 115M's shipped `assistant` rows on the assistant clips and the calls** (leakage item 1: its `fast` /
`assistant` constants were scanned on the 399 test clips and the calls, and its v5 classifier was picked with columns
scored on the 399). The held-out re-pick (plans/fixwave/turn_clean.md) reproduced the shipped `balanced` and `fast`
rules, which cleared those rows (they are back in the headline tables, listed here for reference); it picked a
different `assistant` rule, so the shipped one stays here. Its AMI test rows are clean and are in the headline AMI table.

| audioforge 115M (⚠) | accuracy, W = 2.5 s | false fire, 2.5 s | accuracy, 3.0 s | false fire, 3.0 s | p50 / p95 ms | missed | calls p50 / p95 | calls interrupt | calls missed |
|---|---|---|---|---|---|---|---|---|---|
| `assistant` | 92.7 % [90.2, 95.2] | 5.4 % | 92.7 % | 5.4 % | 299 / 710 | 5.1 % | 3232 / 4801 | 5.5 % | 33.9 % |
| `balanced` (default) | 43.9 % | 100 % | 43.9 % | 100 % | 1226 / 1587 | 0.0 % | 955 / 1918 | 20.2 % [13.1, 29.2] | 7.3 % [3.5, 12.0] |
| `fast` | 41.6 % | 100 % | 41.6 % | 100 % | 461 / 698 | 4.0 % | 547 / 1861 | 24.8 % | 5.5 % |

The held-out replacement for the shipped `assistant` row is the c5s1 pick in the headline table (96.0 % / 381 ms /
2.7 % at W = 2.5 s; candidate heads v0.5, not the default). An earlier check with held-out constants but the shipped
classifier read 95.0 % / 354 ms / 3.1 % at W = 3.0 s (research/FIXALL.md step 2).

**A.2 ⚠ AMI dev turn rows.** The first-pass AMI turn row (115M 1326 ms / 10.5 % / 33.5 %, 0.6B 1180 / 11.0 / 33.0) was
on the AMI dev meetings, the set the 115M's constants were tuned on (the 10.5 / 33.5 % is the tuning goal itself).
Superseded by the AMI test rows.

**A.3 ⚠ ICSI test speaker rows: seen speakers.** All 13 speakers of the ICSI test meetings Bmr013 / Bmr018 / Bro021 also
speak in the ICSI train meetings (65 % of the TitaNet ICSI train cache), on which our speaker head, TS-VAD and speech
heads were trained; the ICSI test windows were also the non-regression check of a TS-VAD serving rule (item 7). The
baselines were not trained on these speakers by us.

| system (ICSI test, 847 windows; ⚠ ours: seen speakers) | tWER % [meeting CI] | target miss + FA % | tracking F1 | speaker EER % [meeting CI] |
|---|---|---|---|---|
| audioforge 115M (115M words) | 34.9 [29.4, 41.0] | 22.4 | 0.884 | 2.5 [0.3, 4.0] |
| audioforge 0.6B (0.6B words) | 29.0 [22.8, 36.1] | 20.7 | 0.896 | 1.9 [0.0, 2.4] |
| Nemotron-3 + print, 0.6B words | 59.4 [49.0, 69.6] | 75.6 | 0.695 | – |
| pyannote 3.1 + print, 0.6B words | 69.0 [55.5, 77.7] | 88.7 | 0.669 | – |
| *Nemotron-3, oracle binding* | – | *39.8* | *0.831* | – |
| *pyannote 3.1, oracle binding* | – | *80.2* | *0.711* | – |
| TitaNet-L / WeSpeaker (EER only) | – | – | – | 1.2 [0.8, 2.1] / 0.9 [0.2, 1.6] |
| no filter / oracle filter (0.6B words) | 101.9 / 23.4 | – | – | – |

Standard DER of the diarizers on the ICSI test windows (every speaker, 0.25 s collar): Nemotron-3 15.5 % [12.1, 22.6]
(missed 0.8, false alarm 13.4, confusion 1.3), pyannote 3.1 18.5 % [15.8, 23.6] (1.0 / 10.6 / 6.9).

The ICSI rows were re-pooled from the first pass's per-unit counts over the test meetings (words, prints, tracks and
keep rule of the first pass); a full-set re-pool reproduces the first-pass numbers exactly.

**A.4 ⚠ The shipped speech heads on ICSI test: seen speakers.** The shipped `speech` heads trained on ICSI meetings
that hold all 13 ICSI test speakers. Retrained without those meetings (same recipe, held-out selection; not shipped,
because held-out was slightly worse), they score lower on ICSI test (paired, F1 −0.008 [−0.011, −0.005] for the 115M
and −0.018 [−0.023, −0.014] for the 0.6B, turn_clean.json) and about the same on AMI test (0.956 / 0.956). The
headline ICSI row uses the retrained heads.

| shipped head (⚠ ICSI test speakers in training) | ICSI test F1 [meeting CI] | ICSI AUC | ICSI miss @7.5 % FA |
|---|---|---|---|
| audioforge 115M speech head v0.4 | 0.938 [0.923, 0.954] | 0.951 | 12.7 % |
| audioforge 0.6B speech head (v0.3 = v0.4) | 0.940 [0.924, 0.955] | 0.956 | 12.0 % |

**A.5 ⚑ Marked in place, not moved** (weaker flags): the 0.6B's turn presets (items 4-5), the LID heads' input blocks
(item 8).
