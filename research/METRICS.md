# Metrics: the standard voice-agent scorecard

2026-09-29. This is the one place that defines every number we publish about audioforge. Each metric has one
definition, one unit and one direction. Every value names the file it comes from. Where we did not measure
something, the table says "not measured".

- **audioforge** is the shipped product: `audioforge-serve` in single-model mode (`--mode single`). That is one
  frozen NVIDIA 115M streaming FastConformer plus our heads, with the user's stored 5 s voice print.
- **Pipecat default** is Pipecat 1.12 with Silero VAD, smart-turn v3 and Whisper small, run locally.
- **LiveKit default** is LiveKit Agents 1.8 with Silero VAD, the LiveKit turn detector (English) and Whisper small,
  run locally.
- **NVIDIA baseline** is NVIDIA's own model for the same job, where one exists.
- Hardware: an Apple M5 laptop CPU with 2 threads per process, unless a row says otherwise. GPU numbers come from
  the RTX 5090 run ([PR #1](https://github.com/maxmelichov/audioforge/pull/1),
  `research/GPU_RUN_2026-09-29.md` on that branch).

## Scorecard

| # | metric | what it means | unit | better | audioforge | LiveKit default | Pipecat default | NVIDIA baseline | source |
|---|---|---|---|---|---|---|---|---|---|
| **STT** | | | | | | | | | |
| 1 | WER, LibriSpeech test-clean | share of words wrong on clean read speech (full set, 2620 utterances, Whisper normalizer) | % | lower | **2.48** | *measuring* (Whisper small) | *measuring* (Whisper small) | not measured | `runs/asr_leaderboard.json` |
| 2 | WER, LibriSpeech test-other | the same, on harder read speech (2939 utterances) | % | lower | **6.13** | *measuring* | *measuring* | not measured | `runs/asr_leaderboard.json` |
| 3 | WER, meeting speech (AMI) | the same, on 200 single-speaker segments of AMI dev meetings (headset mix, Whisper normalizer). The leaderboard's AMI IHM test set is not on disk | % | lower | 20.6 | 14.4 | 14.4 | 10.4 (Nemotron streaming 0.6B, also streaming); 9.5 (Parakeet-TDT 0.6B v3, offline) | `runs/final_asr.json`, `runs/hybrid_asr.json` |
| 4 | RTFx | seconds of audio transcribed per second of compute (ASR model only, offline decode, CPU 2 threads) | x | higher | 59-71 (full LibriSpeech sets) | *measuring* | same as LiveKit | 25 (Parakeet-CTC 0.6B, offline, 200 LibriSpeech utterances) | `runs/asr_leaderboard.json`, `runs/final_asr.json` |
| 5 | partial latency | time from the end of a spoken word to that word first appearing in the live transcript | ms, p50 / p95 | lower | **441 / 732** (CPU); GPU about 429 / 720 (estimate) | none: Whisper small shows no text until the segment ends (see row 6) | none (same) | not measured | `runs/stt_latency.json` |
| 6 | final latency | time from the end of the user's utterance to the final transcript of it | ms, p50 / p95 | lower | 1362 / 2072 | not measured (Whisper small alone takes 787 / 1884 ms per segment, after a 550 ms silence wait) | not measured (Whisper small alone takes 712 / 1257 ms per segment, after a 200 ms silence wait) | – | `runs/metrics.json` |
| **Turn detection** | | | | | | | | | |
| 7 | end-of-turn latency | time from the user going silent at the end of a turn to the agent framework receiving "turn over" (the agent's own reply time not included) | ms, p50 / p95 | lower | 1382 / 3400 | 1350 / 3096 | 1675 / 3197 | – | `runs/metrics.json` |
| 7b | end-of-turn latency, fixed rule | **PLACEHOLDER** (another agent is measuring it: `research/EOT_LATENCY.md`, `runs/eot_latency.json`) | ms, p50 / p95 | lower | pending | – | – | – | pending |
| 8 | false-interruption rate | share of the user's turns in which the agent would start talking before the user finished | % of turns | lower | **17.4** | 23.9 | 30.3 | – | `runs/metrics.json` |
| 9 | response rate | share of the user's turns that get an answer at all before the user speaks again | % of turns | higher | **88.1** | 82.6 | 78.9 | – | `runs/metrics.json` |
| 10 | end-of-turn precision / recall / F1 (TurnBench dev) | on a public two-party call set: of the "turn over" calls, how many were right (precision); of the real turn ends, how many were found (recall); F1 combines both | 0-1 | higher | 0.940 / 0.849 / **0.892** | not measured (the LiveKit turn detector reads text; it was not run on TurnBench) | 0.931 / 0.754 / 0.833 (smart-turn v3) | 0.942 / 0.763 / 0.843 (Parakeet-Realtime-EOU) | `runs/metrics.json` (`turnbench_eot`) |
| **VAD** | | | | | | | | | |
| 11 | VAD ROC-AUC | how well the speech score separates speech from non-speech over every threshold (1.0 = perfect) | 0-1 | higher | **0.972** | 0.956 (Silero v5) | 0.956 (Silero v5) | 0.959 (MarbleNet v2) | `runs/vad_auc.json` |
| 12 | VAD F1 | balance of missed and false speech frames at the 0.5 threshold | 0-1 | higher | **0.951** | 0.915 | 0.915 | 0.937 | `runs/vad_auc.json` |
| **Speaker** | | | | | | | | | |
| 13 | target-speaker DER, AMI / ICSI | share of the user's speech time that is missed, plus other people's speech wrongly given to the user, over the user's speech time (5 s voice print; 80 ms frames, no collar, overlap counted) | % | lower | **50.9 / 23.0** | no speaker tracking | no speaker tracking | 69.5 / 73.4 (Nemotron-3-Diarization with a column bound to the same print) | `runs/metrics.json` (`target_speaker_der`) |
| 13b | target-speaker WER | **PLACEHOLDER** (another agent is measuring it: `research/TSWER.md`, `runs/tswer.json`) | % | lower | pending | – | – | – | pending |
| **Efficiency** | | | | | | | | | |
| 14 | compute per 160 ms chunk | processing time for each 160 ms of audio, whole server (ASR, VAD, turn, voice tracking, language) | ms | lower | 53 (CPU, live p50); 19.9 (RTX 5090, idle GPU) | – (Whisper runs per segment, not per chunk) | – | – | live records (`runs/metrics.json` `compute`), PR #1 |
| 15 | real-time streams per device | how many live sessions one server process keeps up with | streams | higher | CPU M5 2 threads: **3** (43 ms per stream per chunk); 5090 box CPU 2 threads: 1; RTX 5090: 4 (shared GPU) | not measured | not measured | – | `runs/streams_cpu.json`, PR #1 |
| 16 | CPU per audio second | CPU seconds the whole process spends per second of audio | s/s | lower | 0.34 | **0.28** | 0.43 | – | `runs/metrics.json` (`compute`) |

Rows 7-9 are the live test: 32 two-party calls (16 TurnBench, 16 otoSpeech), fed on the **user's own channel**
through each framework at real time, 109 user turn ends. A stub LLM and TTS answer instantly, so rows 7-9 measure
only the listening side. The two default stacks ran in their own framework; audioforge ran through Pipecat.

### Where we lose

- **End-of-turn latency is the same as LiveKit's default** (1382 vs 1350 ms median), and the slow tail is longer
  (p95 3400 vs 3096 ms). audioforge answers more turns (row 9) and interrupts less (row 8), but it does not answer
  faster.
- **Transcripts are worse than Whisper small on meetings** (row 3: 20.6 vs 14.4 %), and NVIDIA's larger models are
  much better there: 10.4 % for the 0.6B streaming model, 9.5 % for Parakeet-TDT v3. On the live calls, the
  concatenated transcript scored 23.2 % WER for audioforge against 19.3 % for LiveKit's default and 23.5 % for
  Pipecat's (`research/SINGLE_MODEL.md`).
- **LiveKit's default uses less CPU** (row 16: 0.28 vs 0.34 CPU seconds per audio second).
- **On TurnBench, a tuned Silero silence timeout is close** (F1 0.878 vs 0.892), and VAP, a research model, is
  better (0.901, with a 463 ms median against our 1316 ms).

### Rows that need the RTX 5090

- Row 5 on GPU: the GPU partial latency is an estimate (CPU latency with the CPU chunk time swapped for 20 ms). It
  needs an end-to-end run on the 5090.
- Row 15 on GPU: "4 streams" was measured while another job shared the GPU. It needs a rerun on an idle GPU.
- Rows 1-4 for the NVIDIA baselines (Nemotron streaming 0.6B, Parakeet-TDT v3 on full LibriSpeech), and GPU RTFx for
  every system. These are pending from the 5090.
- Rows 7-9 through LiveKit for audioforge (it has only run through Pipecat), so the framework is the same in every
  column. This runs on CPU; it needs no GPU.

## Definitions and how each was measured

**WER (rows 1-3).** Word error rate = (substituted + deleted + inserted words) / reference words, on the full test
sets, with Whisper's English text normalizer on both sides, as the Open ASR Leaderboard does
(`scripts/research/asr_leaderboard.py`). audioforge's transcript is the served streaming model's. The offline masked
forward used here gives the same output as chunk-by-chunk streaming at 160 ms (`tests/test_streaming.py`). Whisper
small is faster-whisper int8, greedy, English, the model the default stacks were run with (research/E2E_FINAL.md).
Row 3 uses the 200-segment AMI set of research/FINAL_REPORT.md §1.2. On the same set with our own normalizer
(`normalize_text`), the values are 24.4 % (audioforge), 21.2 % (Whisper small), 11.2 % (0.6B) and 9.7 % (TDT v3).
On overlapping speech from all speakers of four AMI test meetings (the partial-latency run), audioforge's streaming
WER is 35.3 %.

**RTFx (row 4).** Total audio seconds / total decode seconds, model load excluded. Offline batch decode on CPU with
2 threads. In the live server, which runs every head, RTFx is 3.0 (1 / RTF 0.33).

**Partial latency (row 5).** `scripts/research/stt_latency.py`, `runs/stt_latency.json`. The served session in
single mode was fed in 160 ms blocks, as a real-time client sends them. It ran on 12 two-minute windows of four AMI
test meetings (IS1009b, ES2004b, TS3003b, EN2002a; headset mix), 2853 correctly recognised words. For each reference
word the final transcript gets right (jiwer alignment), latency = when the word, in its final spelling, first
appears in a `partial` minus the word's end time in AMI's manual word timings. "When it appears" is the audio clock
after the block that produced it plus the measured compute time of that block. Network time is not included.
- p50 441 ms [412, 459], p90 652 ms, p95 732 ms [703, 771] (95 % CIs by resampling windows).
- The median block compute in this run was 32 ms; in the live sessions it was 53 ms, which would add about 20 ms.
- 7 % of words appear before their annotated end (AMI's word boundaries are approximate); they are kept as
  negative values.
- GPU estimate: the same with 20 ms per chunk (PR #1): p50 429, p95 720 ms. Most of the latency is the model's
  own emission delay, not compute.

**Final latency (row 6).** `scripts/research/metrics_table.py`, from the live Pipecat records of single mode on the
user channel: user turn end (reference) → the framework receives the server's `final` for that turn. It is sent
together with the `turn_end`, so it is the end-of-turn latency seen as text (93 turns). The default stacks' stored
records keep no final times. Their transcription time is known from the component timers in `runs/e2e_final.json`,
so row 6 gives it as a partial fact, not as a latency.

**End-of-turn latency (row 7).** For each user turn end with a response within 6 s: the time from the reference end
(annotated speech end) to the moment Pipecat's LLM stage (or LiveKit's committed user turn) receives the turn. p50
and p95 are over those answered ends. This is what earlier documents called "dead air"; with a zero-delay stub
agent it is the listening side's share of voice-to-voice latency.

**Voice-to-voice latency** = end-of-turn latency + LLM time to first token + TTS time to first audio. We do not
measure LLM or TTS time; it depends on the models you choose.

**False-interruption rate (row 8).** A false interruption is a response moment that falls inside a user turn, in
its speech or in a pause within the turn (`audioforge/e2e_metrics.py` `score_clip`). The rate is the share of user
turns with at least one. Per minute of audio: 0.89 (audioforge), 1.29 (LiveKit default), 2.17 (Pipecat default).

**Response rate (row 9).** Share of scored user turn ends with any response before the user's next turn starts.

**TurnBench precision / recall / F1 (row 10).** TurnBench dev (Sesame; 38 two-party conversations, 1904 turn ends,
1063 within-turn pauses), official scorer (vendored, MIT). A detection counts if it lands in [−0.25 s, +3 s] of an
end; a detection in a pause is a false positive. Precision = TP / (TP + FP), recall = TP / (TP + FN).
- audioforge's row is the fixed turn-head policy with thresholds fitted on the other half of the set and pooled
  (held out; `runs/turnbench_latency.json` `families > fixed > crossfit_pooled_heldout`). It reads the turn head
  with Silero on the user's own channel, which is the two-party setting but not the exact `--mode single` input.
  The thresholds picked on the whole set give 0.949 / 0.835 / 0.889.
- smart-turn v3 is the authors' own dev predictions, rescored. Parakeet-Realtime-EOU and the Silero timeout
  (0.951 / 0.815 / 0.878) had their threshold picked on this set (false-positive rate ≤ 0.10).
- Median detection delay: 1316 ms (audioforge), 1010 ms (smart-turn v3), 1097 ms (Parakeet-EOU), 463 ms (VAP).

**VAD (rows 11-12).** 64 windows of 20 s from AMI dev meetings, 80 ms frames, label = anyone speaking.
`scripts/research/vad_auc.py`. Silero and MarbleNet outputs are max-pooled onto the 80 ms grid. audioforge's head
was trained on AMI's labels (in domain for it, not for Silero or MarbleNet). At the same false-alarm rate (7.5 %),
it misses 10.6 % of speech frames against 12.3 % (MarbleNet) and 13.8 % (Silero).

**Target-speaker DER (row 13).** Each target is the main speaker of an eot-bench v2 window (AMI dev 974, ICSI held
out 1312), with a 5 s voice print cut from elsewhere in the same meeting.
- audioforge = the served TS-VAD head.
- Nemotron-3 = its column bound to the same print, the better of two binders (TitaNet-L on both corpora).
- DER = (missed target frames + false-alarm frames) / target frames, at 80 ms, with no collar and overlap counted.
  It is computed from the frame counts in `runs/improve_115m.json` (`frame > primary`).
- AMI is in Nemotron-3's training data.
- The same numbers as F1 (the earlier documents' "tracking F1"): AMI 0.743 vs 0.648, ICSI 0.882 vs 0.696.

**Efficiency (rows 14-16).**
- Row 14 CPU: the server's own `chunk_ms_p50`, median over the 69 live sessions (`runs/metrics.json` `compute`).
- Row 15 CPU M5: `scripts/research/streams_cpu.py`. K single-mode sessions were fed interleaved on one thread;
  real time means the summed compute for one 160 ms block of every stream stays below 160 ms at p95.
- Row 16: the process's CPU seconds / audio seconds. For audioforge that is the server; for the default stacks it
  is the whole agent process, where their models run.
- Peak memory: 1156 MB (audioforge server), 2802 MB (LiveKit default), 2372 MB (Pipecat default).

## Why the old "missed within 3 s" number was dropped

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
| AMI meeting windows | 5 / 5 | 20.0 | 40.0 | 60.0 | 40.0 |
| all 69 pooled (the old headline) | 69 / 223 | 37.7 | 41.3 | 58.7 | 34.1 |

The scorecard therefore reports the user-channel sessions and replaces "missed within 3 s" with end-of-turn latency
(row 7) and the response rate (row 9). On the mono mixes the response rate is 58.7 % (audioforge), 59.6 % (LiveKit)
and 57.8 % (Pipecat): no system copes, and they are equal. On the user's channel audioforge's room mode (Nemotron-3
diarizer, 1 s timeout) is faster (1272 / 1631 ms) and answers slightly more (89.9 %), but interrupts more (26.6 % of
turns).

## Names used in earlier documents

| old name | standard name here | note |
|---|---|---|
| dead air | end-of-turn latency | the same measurement; now reported on the user channel, p50 / p95 |
| cut-ins, "talks over you", interruptions per call | false interruptions | now % of user turns (per minute given too); "per call" was per 20-60 s clip |
| unanswered / missed within 3 s | response rate (+ late responses) | the pooled number mixed in the mono artifact; see above |
| first words / first text (1081 ms) | removed | it counted from the start of speech, so it included the time it takes to say the first word. Partial latency (row 5) is the standard measure |
| finds your voice / tracking F1 | target-speaker DER | F1 kept as a secondary number |
| hears speech / speech missed at 7.5 % FA | VAD ROC-AUC, F1 | the miss rate at a fixed false-alarm rate is kept as a detail |
| missed turn ends at ≤ 5 % false cut-offs (AMI 34 %, ICSI 19 %) | appendix | an offline research metric on meeting windows with a stored print; not a standard voice-agent number |
| RTF 0.33 | compute per chunk, streams per device, CPU per audio second | RTF is wall time; the default stacks were measured in CPU seconds, so row 16 compares like with like |

## Appendix: research metrics kept out of the scorecard

- **Missed turn ends in meetings** (eot-bench v2, offline, each system at ≤ 5 % of turns cut off early, 6 s
  horizon): audioforge with a stored print 34.2 % (AMI) / 18.7 % (ICSI). The detectors without a print: Parakeet-EOU
  70.1 %, smart-turn + Silero 72.6 %, LiveKit turn detector + timeout 74.2 % (AMI); Silero timeout 84.4 % (ICSI)
  (`runs/single_model.json`, `runs/baselines_turn*.json`). audioforge had the print and the others did not.
- **Language ID**, FLEURS-17: 91.0 % at 2 s / 97.8 % full utterance, against AmberNet's 95.1 / 99.5 %
  (`runs/lid.json`).
- **Voice print length**: research/SINGLE_MODEL.md A2.

## Image text changes needed (not applied here)

`demo/images/redesign/results_v7.html` and its numbers (`numbers_single.json`) belong to the image work. To match this
scorecard, it needs these changes:

1. **"Responds after your turn"**, hero "37.7 % unanswered after 3 s", bars 37.7 / 41.3 / 58.7. This pools 69
   sessions, 32 of them mono mixes (see above). Replace it with the **response rate** on the user's channel:
   "88 % of turns answered" vs LiveKit default 83 %, Pipecat default 79 % (`runs/metrics.json`
   `live > *|user > answered_pct`). Do not use end-of-turn latency as a win: 1382 ms ties LiveKit's 1350 ms.
2. **"Talks over you less"**, hero "0.67 interruptions / call", "40 % fewer than LiveKit". Replace it with **false
   interruptions, % of user turns**: 17 % vs LiveKit 24 %, Pipecat 30 %, "27 % fewer than LiveKit default"
   (`false_int_pct_of_turns`, user channel). "Call" was a 20-60 s clip.
3. **Method lines "live, 69 calls" / "the same 69 calls"** and the footer "69 live calls": use "live, 32 two-party
   calls, user's channel". The 69 sessions included 5 AMI meeting windows and 32 mono mixes.
4. **"Missed turn ends" (34 %, "51 % fewer than Parakeet-EOU")** is an offline research metric in which audioforge
   has a voice print and the others do not. Replace it with **end-of-turn F1 on TurnBench**: 0.89 vs Pipecat
   smart-turn v3 0.83 and NVIDIA Parakeet-Realtime-EOU 0.84 (`runs/metrics.json` `turnbench_eot`). The LiveKit turn
   detector was not run on TurnBench. The caveat line should say that audioforge's thresholds were fitted on held-out
   halves and that its input is the turn head with Silero on the user's channel.
5. **"Finds your voice", F1 0.88 / 0.74 vs Nemotron-3 0.70 / 0.65.** Show **target-speaker DER** (lower is
   better): ICSI 23 % vs 73 %, AMI 51 % vs 70 % (`target_speaker_der`, `tsvad_spk_vp5p0` vs
   `nemotron3_vp_titanet_vp5p0`). Keep "AMI is in Nemotron-3's training data".
6. **"Hears speech", 10.6 % speech missed.** Show **VAD ROC-AUC** 0.972 vs MarbleNet 0.959 and Silero 0.956, or F1
   0.951 / 0.937 / 0.915 (`runs/vad_auc.json`). Add "audioforge's VAD head was trained on AMI labels". On a
   zero-based 0-1 axis these bars look alike, so the hero line must carry the numbers.
7. **Optional new STT card: "Words while you talk"**, partial latency 441 ms median (word end to word shown,
   `runs/stt_latency.json` `cpu.p50_ms`). The default stacks show no partial text (Whisper small waits for the end
   of the utterance), so there is no grey bar. Do not bring back "first words" (1081 / 2812 / 5120 ms): it counted
   from the start of speech.
8. **numbers_single.json:** drop `first/*`, `firstms/*` and `v7/first_x`. Re-source `s69/livekit/*`: the pooled
   LiveKit numbers are not in `runs/single_model.json`; they are in `runs/metrics.json`.
