# Results

How audioforge compares with the open models voice agents use today, in plain words. Public **test** splits only
(2026-10-03): both of our model sizes and the best open baselines, every baseline re-run on the same audio and labels.
"Ours 115M" is audioforge on the default 115M core, "ours 0.6B" on the English 0.6B core (heads v0.4 on both).

The full tables, with 95 % confidence intervals, which audio each head was trained and selected on, how each
baseline was run, and the rows left out because they were tuned on their own test audio, are in
[`research/FINAL_COMPARE.md`](../research/FINAL_COMPARE.md). Metric definitions: [`research/METRICS.md`](../research/METRICS.md).
How the charts are made: [`demo/images/EXPLAINER.md`](../demo/images/EXPLAINER.md).

## Words

![words](../demo/images/compare_asr.png)

**How to read it.** Five charts of word error rate (lower is better), one per test set, and the time until the final
text of a turn is ready (top right). The 0.6B core is the best on meetings (AMI) and on the user's side of live calls
while streaming; offline Parakeet-TDT v3 is better on LibriSpeech and ICSI meetings. The 115M is about twice as wrong
on meetings, but it is the fastest to finish a turn.

WER %, Whisper normaliser; ours: the default 160 ms streaming pass, a word shows up about 0.3 s after it is said; the
others transcribe after the speaker stops.

| system | LibriSpeech test-clean | LibriSpeech test-other | AMI test meetings | ICSI test meetings | live calls, user channel |
|---|---:|---:|---:|---:|---:|
| ours 0.6B | 2.7 | 5.7 | **7.9** | 10.3 | **5.7** |
| ours 115M | 2.4 | 6.8 | 16.1 | 18.4 | 15.4 |
| Parakeet-TDT 0.6B v3 | **1.8** | **3.4** | 8.3 | **7.6** | 7.7 |
| Whisper large-v3 | 2.0 | 3.6 | 10.7 | 13.2 | 8.5 |
| Whisper small, beam 5 | 3.1 | 7.1 | 10.9 | 15.0 | 9.6 |

Whisper small runs at beam 5, faster-whisper's default, which is what Pipecat's local Whisper service calls (its
default model is distil-medium.en; LiveKit Agents ships no default speech-to-text). The live-call references keep
repetitions and false starts, which Whisper leaves out; with those collapsed in references and hypotheses alike, the
0.6B and Whisper large-v3-turbo tie on the user channel (5.4 % each).

**Final text after you stop** (56 labelled user turns of the live calls; clock from the labelled end of the turn;
ours with `--final-chunk-ms 1120`; WER of the turn texts each system returned at that moment)

| system | device | p50 | WER of the timed texts |
|---|---|---:|---:|
| ours 115M | Mac GPU (MPS) | **25 ms** | 14.2 % |
| ours 0.6B | Mac GPU (MPS) | 49 ms | 7.8 % |
| Parakeet-TDT 0.6B v3 (offline) | Mac GPU (MPS) | 232 ms | **7.2 %** |
| Whisper large-v3-turbo | Mac GPU (MPS) | 282 ms | 10.6 % |
| Whisper small, beam 5 | CPU, 2 threads (CTranslate2 has no MPS) | 1804 ms | 9.9 % |

On the same GPU the 0.6B's final text is ready about 4.7× sooner than Parakeet-TDT's, about as accurate (CIs
overlap). The 115M is fastest but its flushed turn texts have twice the errors.

## Turn taking

![turn taking](../demo/images/compare_turn.png)

**How to read it.** On 399 short clips of people talking to an assistant, how often each system is right about
whether the person is done, how long it waits before answering, and how often it cuts off someone who has not
finished. Ours are right more often and almost never cut in; Pipecat's smart-turn answers sooner but cuts off about
three unfinished sentences in ten.

smart-turn v3.2's 399 public test clips, speech to an agent; `--turn-preset assistant`:

| system | right about "done" | answers after you stop, p50 | cuts off unfinished sentences |
|---|---:|---:|---:|
| ours 0.6B | **97.7 %** | 351 ms | **0.9 %** |
| ours 115M, held-out pick (candidate heads v0.5, not the default) | 96.0 % | 381 ms | 2.7 % |
| LiveKit turn detector + Silero | 85.2 % | 547 ms | 22.3 % |
| Pipecat smart-turn v3.2 + Silero | 75.9 % | **211 ms** | 29.0 % |
| NVIDIA Parakeet-Realtime-EOU | 49.4 % | 462 ms | 89.7 % |

A cut-off counts when a turn end lands within 2.5 s of where an unfinished clip stops. 2.5 s keeps clear of every
system's fallback timer (Pipecat and LiveKit end the turn 3 s after silence, ours after 3.0-3.4 s), so a fallback never
decides a row by tens of milliseconds; at the earlier 3 s window the order is the same. The 115M's shipped `assistant`
rule is a faster preset tuned earlier on these test clips, so it is not reported; the row above is the rule re-picked
on held-out audio only.

**Meetings.** On AMI test meetings (200 turns, default `balanced` preset; p50 / interruptions / missed turns), ours
knows the user's 5 s voice print, the baselines do not:
- with the print: ours 115M 1527 ms / 15.5 % interruptions / 36.0 % missed, ours 0.6B 1498 ms / 10.0 % / 33.5 %;
- without it (same engine): ours 115M 1508 ms / 6.5 % / 74.0 %, ours 0.6B 1499 ms / 5.5 % / 73.5 %;
- LiveKit 745 ms / 13.0 % / 72.0 %, Pipecat 752 ms / 39.5 % / 53.0 %, Parakeet-EOU 1251 ms / 4.5 % / 86.0 %.

Knowing the user's voice is what wins on meetings: without the print our cores miss as many turns as LiveKit, because
other people keep talking after the user stops. Two-party call rows are in FINAL_COMPARE.md, outside the headline: no
labelled public test split exists for them.

## Speech detection

![speech detection](../demo/images/compare_vad.png)

**How to read it.** Four charts for AMI and ICSI test meetings, every 80 ms: F1 at the usual 0.5 threshold, and
ROC-AUC, which does not depend on any threshold. On AMI our lead holds only at the 0.5 threshold; on ICSI it holds on
both.

Test meetings, 80 ms frames. On AMI our heads lead the live VADs on F1 at threshold 0.5 (0.959
115M / 0.957 0.6B against MarbleNet v2 0.941, TEN VAD 0.930, Silero v5 0.901), but that lead rests on the threshold:
threshold-free, MarbleNet ties us (AUC 0.967 against 0.966 / 0.967) and misses less speech at 7.5 % false alarms
(9.6 against 12.5 / 12.6 %). pyannote's segmentation model is best on AMI (F1 0.977) but reads 10 s ahead, so it
cannot run live. On ICSI, with heads retrained without the ICSI test speakers (the shipped heads heard them in
training), the F1 lead is small (115M 0.930 against Silero 0.924 and TEN VAD 0.922; the 0.6B's 0.922 ties TEN VAD),
and threshold-free ours lead clearly (AUC 0.941 / 0.945 against at most 0.932; 16.0 / 15.5 % speech missed against
at least 20.5 %).

## Speaker tracking ("your words only")

![speaker tracking](../demo/images/compare_spk.png)

**How to read it.** How many of the user's own words come out wrong when a meeting is filtered down to the user,
given the same 5 s voice print for every system; how well each system tracks when the user speaks; and how well the
voice prints alone tell people apart. Ours give the best user transcript, but dedicated speaker-embedding models
(TitaNet-L, WeSpeaker) match voices better.

Target-speaker WER %, AMI test meetings, the same 5 s voice print for every system; the diarizers on the 0.6B's
words: ours 0.6B 47.1, ours 115M 51.5, Nemotron-3 diarizer 64.4, pyannote 3.1 77.8. Most of
that margin is how a diarizer's column is bound to the print: with the column picked from the labels (an upper bound)
Nemotron-3 tracks the user nearly as well as ours (tracking F1 0.795 against 0.811 / 0.829). ICSI speaker rows are not
reported: every ICSI test speaker is in our training meetings.

## Language ID

![language](../demo/images/compare_lid.png)

**How to read it.** Accuracy over 17 languages after 2 s of speech and on the whole clip. The big offline models
(Whisper large-v3, AmberNet) are better; ours are close behind and cost almost nothing, because they read the same
encoder pass as everything else.

FLEURS 17 languages, test, 2 s of speech: Whisper large-v3 95.9 %, AmberNet 95.1 %, ours 0.6B
92.7 %, ours 115M 92.4 %, Whisper small 90.6 %. (The encoder blocks our heads read were picked with a probe scored on
FLEURS test, probably worth well under a point.)

## Perceived voice gender

Optional head, off by default: balanced accuracy 96.3 % on FLEURS-17 test and 97.1 % on
LibriSpeech test-clean after 1 s of speech (115M). It gives female / male voice probabilities, a perceived vocal
characteristic and not anyone's gender identity; it can be wrong for any one voice (one LibriSpeech test reader is
called the other class on every utterance), must not be used to make decisions about people, and has no baseline here
([`research/VOICE_GENDER.md`](../research/VOICE_GENDER.md)).

## Cost

On an Apple-silicon laptop (whole engine, per 160 ms of audio): 115M 28.6 ms on the GPU, 30.4 ms on 2 CPU
threads, 5 real-time streams on the GPU, 1.2 GB; 0.6B 42.9 ms GPU / 97.1 ms CPU, 3 streams on the GPU, 4.9 GB.

## The two cores side by side

The test-split numbers above, for choosing a core ([`USAGE.md`](USAGE.md#the-two-cores)); heads v0.4 on both:

| | 115M (default) | 0.6B |
|---|---:|---:|
| WER, AMI / ICSI test meetings | 16.1 / 18.4 % | **7.9 / 10.3 %** |
| WER, live two-party calls (all words / user channel) | 20.3 / 15.4 % | **11.6 / 5.7 %** |
| target-speaker WER, AMI test | 51.5 % | **47.1 %** |
| speaker EER within a meeting, AMI test | 5.0 % | **3.8 %** |
| speech detection F1 at 0.5, AMI test | **0.959** | 0.957 |
| turn end, AMI test (`balanced`, with the print): p50, false interruptions, missed | 1527 ms, 15.5 %, 36.0 % | **1498 ms, 10.0 %, 33.5 %** |
| speech to an agent (`assistant`, 2.5 s window): accuracy, p50, cut-offs | 96.0 %, 381 ms, 2.7 % (held-out pick, candidate heads v0.5) | **97.7 %, 351 ms, 0.9 %** |
| final text after you stop (`--final-chunk-ms 1120`, Mac GPU): p50, WER of the timed texts | **25 ms**, 14.2 % | 49 ms, **7.8 %** |
| compute per 160 ms chunk, CPU 2 threads / Apple GPU | **30.4 / 28.6 ms** | 97.1 / 42.9 ms |
| real-time streams, CPU 2 threads / Apple GPU | **4 / 5** | 1 / 3 |
| memory | **1.2 GB** | 4.9 GB (+3.3 GB GPU) |

## Where it loses

- **Turn answers are not the fastest.** Pipecat smart-turn answers assistant speech sooner (211 ms against 351 / 381 ms
  p50), at 29 % cut-offs against our 0.9 / 2.7 %. On meetings Pipecat and LiveKit answer in 0.75 s against our 1.5 s.
  The `assistant` preset misses many meeting turns, so conversations use `balanced`; no single preset wins every
  column.
- **Meetings without the voice print:** our turn ends miss about three meeting turns in four, like LiveKit. The win
  on meetings needs the user's 5 s enrolment.
- **Words:** Parakeet-TDT v3 (offline) beats the 0.6B on ICSI test meetings (7.6 against 10.3 %), on LibriSpeech
  (3.4 against 5.7 % on test-other) and on FLEURS English (7.0 against 8.0 %). Whisper large-v3-turbo ties the 0.6B on
  the live user channel once repetitions are collapsed. The 115M core is behind every offline baseline on meetings,
  and its fast final texts have 14.2 % WER; use `--core 0.6b` for transcript quality.
- **Speech detection on AMI:** threshold-free, MarbleNet v2 ties our heads and misses less speech; pyannote (offline)
  beats every live detector.
- **Language ID:** Whisper large-v3 and AmberNet beat our heads, most at 2 s (95.9 against 92.7 %).
- **Speaker embeddings alone:** TitaNet-L and WeSpeaker match voices better than our speaker heads (EER AMI test
  1.9 % against 3.8-5.0 %); our tracker still gives the best target-speaker WER.
- Not measured: paid APIs (Deepgram Nova-3, AssemblyAI).
