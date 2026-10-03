# audioforge

One frozen NVIDIA streaming speech model with small heads on top: live words, voice activity, "is it the user?" and
"is the turn over?" for a voice agent that talks to one known user, over one WebSocket.

![architecture](demo/images/architecture_v10.png)

A 115M NVIDIA cache-aware FastConformer ([`stt_en_fastconformer_hybrid_large_streaming_multi`](docs/MODELS.md),
frozen, 160 ms chunks) gives the words (RNNT). The heads read its layers ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)):

| head | reads (115M; 0.6B in brackets) | size | job |
|---|---|---|---|
| speech detector | a learned mix of blocks 2-6 [8-16] | 33K [66K] | is anyone speaking? (the client's speech probability, every 80 ms) |
| VAD (turn rules) | block 4 | 33K | the quiet gate the turn rules, TS-VAD and LID read |
| speaker | block 4 [5] | 0.5M | a 192-number voice print |
| TS-VAD | block 4 [5] + the stored print | 0.26M | is it the user or someone else? |
| turn | a second, speaker-conditioned pass (GRU; 115M only) | 0.32M | is the user's turn over? (`balanced`, `steady`) |
| turn classifier v5 | block 8 [12] of the first pass + VAD / TS-VAD + the words so far, at each quiet frame | 2.5M [2.6M for `assistant`] | is the user's turn over? (`fast`, `assistant`) |
| LID v2 (optional) | blocks 8-12 [16-20] | 2.4M [2.9M] | which language? |
| voice gender (optional, off by default) | block 4 [5] | 22K [21K] | perceived female / male voice probabilities, not gender identity ([research/VOICE_GENDER.md](research/VOICE_GENDER.md)) |

## What you get

- **Live transcripts** as the user talks (`partial`), and one `final` per turn.
- **Voice activity** every 80 ms from the model's own speech-detector head. No Silero.
- **Target speaker:** each frame and turn says whether the user or someone else is talking, from the user's stored
  voice print. No diarizer is loaded.
- **Turn ends** (`turn_end`) from the `vad_head` rule: the VAD and turn heads together, or the v5 turn classifier
  (`--turn-preset fast` / `assistant`).
- **One model, plain PyTorch** (no NeMo), on CPU, Mac GPU (`--device mps`) or CUDA. Pipecat and LiveKit adapters.

## Quickstart

Python 3.10+. `audioforge-download` shows each weight's licence and asks before fetching.

```bash
git clone https://github.com/maxmelichov/audioforge && cd audioforge
pip install -e ".[serve]"
audioforge-download          # the 115M model + heads, sha256-checked, into ./models
audioforge-serve             # single mode, ws://127.0.0.1:8765  (--device mps|cuda for a GPU)
```

Turn presets: `--turn-preset balanced` (default) `| fast` (calls, turn head v5) `| steady | assistant` (speech to an agent).

In a second terminal, stream the bundled 16 s two-party call ([`examples/audio/two_party_call_16s.wav`](examples/audio),
otoSpeech, CC BY 4.0) with the user's stored print (`two_party_call_16s.voiceprint.json`):

```bash
python examples/quickstart_client.py        # or: ... my.wav --voiceprint me.json
```

Output on an Apple M5 laptop (CPU, 2 threads; intermediate partials omitted):

```
  0.00s  voiceprint source=explicit seconds=0.0
  8.16s  partial   so um what value guides your life
 11.36s  partial   so um what value guides your life what values do you live by
 11.78s  turn_end  policy=vad_head silence_ms=640
 11.78s  final     speaker=0 'so um what value guides your life what values do you live by'
 15.20s  partial   value guys my life
 15.30s  turn_end  policy=vad_head silence_ms=160
 15.30s  final     speaker=1 'value guys my life'
```

The user's question is one turn despite a half-second pause after "life"; the other party's answer comes out as
`speaker=1`. In Python, without a socket ([`examples/python_api.py`](examples/python_api.py)):

```python
import audioforge
fe = audioforge.load()                      # device="mps" / "cuda" also work
s = fe.session(sample_rate=16000)
s.enroll(voiceprint)                        # 192 numbers, or fe.voiceprint(>= 5 s of the user's clean speech)
events = s.feed(pcm) + s.end()              # the protocol's messages as dicts
```

## Results

Public **test** splits only (2026-10-03): both cores and the best open models in use today, every baseline re-run on
the same audio and labels. Full tables with 95 % confidence intervals, which audio each head was trained and selected
on, and the rows left out because they were tuned on their own test audio:
[`research/FINAL_COMPARE.md`](research/FINAL_COMPARE.md). Ours = audioforge on the 115M core (heads v0.4) and on the
English 0.6B core (heads v0.4).

**Words** (WER %, Whisper normaliser; ours: the default 160 ms streaming pass, a word shows up about 0.3 s after it is
said; the others transcribe after the speaker stops)

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

![words](demo/images/compare_asr.png)

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

**Turn taking** (smart-turn v3.2's 399 public test clips, speech to an agent; `--turn-preset assistant`)

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

On AMI test meetings (200 turns, default `balanced` preset; p50 / interruptions / missed turns), ours knows the user's
5 s voice print, the baselines do not:
- with the print: ours 115M 1527 ms / 15.5 % interruptions / 36.0 % missed, ours 0.6B 1498 ms / 10.0 % / 33.5 %;
- without it (same engine): ours 115M 1508 ms / 6.5 % / 74.0 %, ours 0.6B 1499 ms / 5.5 % / 73.5 %;
- LiveKit 745 ms / 13.0 % / 72.0 %, Pipecat 752 ms / 39.5 % / 53.0 %, Parakeet-EOU 1251 ms / 4.5 % / 86.0 %.

Knowing the user's voice is what wins on meetings: without the print our cores miss as many turns as LiveKit, because
other people keep talking after the user stops. Two-party call rows are in FINAL_COMPARE.md, outside the headline: no
labelled public test split exists for them.

![turn taking](demo/images/compare_turn.png)

**Speech detection** (test meetings, 80 ms frames). On AMI our heads lead the live VADs on F1 at threshold 0.5 (0.959
115M / 0.957 0.6B against MarbleNet v2 0.941, TEN VAD 0.930, Silero v5 0.901), but that lead rests on the threshold:
threshold-free, MarbleNet ties us (AUC 0.967 against 0.966 / 0.967) and misses less speech at 7.5 % false alarms
(9.6 against 12.5 / 12.6 %). pyannote's segmentation model is best on AMI (F1 0.977) but reads 10 s ahead, so it
cannot run live. On ICSI, with heads retrained without the ICSI test speakers (the shipped heads heard them in
training), the F1 lead is small (115M 0.930 against Silero 0.924 and TEN VAD 0.922; the 0.6B's 0.922 ties TEN VAD),
and threshold-free ours lead clearly (AUC 0.941 / 0.945 against at most 0.932; 16.0 / 15.5 % speech missed against
at least 20.5 %).

![speech detection](demo/images/compare_vad.png)

**Your words only** (target-speaker WER %, AMI test meetings, the same 5 s voice print for every system; the
diarizers on the 0.6B's words): ours 0.6B 47.1, ours 115M 51.5, Nemotron-3 diarizer 64.4, pyannote 3.1 77.8. Most of
that margin is how a diarizer's column is bound to the print: with the column picked from the labels (an upper bound)
Nemotron-3 tracks the user nearly as well as ours (tracking F1 0.795 against 0.811 / 0.829). ICSI speaker rows are not
reported: every ICSI test speaker is in our training meetings.

![speaker tracking](demo/images/compare_spk.png)

**Language ID** (FLEURS 17 languages, test, 2 s of speech): Whisper large-v3 95.9 %, AmberNet 95.1 %, ours 0.6B
92.7 %, ours 115M 92.4 %, Whisper small 90.6 %. (The encoder blocks our heads read were picked with a probe scored on
FLEURS test, probably worth well under a point.)

![language](demo/images/compare_lid.png)

**Perceived voice gender** (optional head, off by default): balanced accuracy 96.3 % on FLEURS-17 test and 97.1 % on
LibriSpeech test-clean after 1 s of speech (115M). It gives female / male voice probabilities, a perceived vocal
characteristic and not anyone's gender identity; it can be wrong for any one voice (one LibriSpeech test reader is
called the other class on every utterance), must not be used to make decisions about people, and has no baseline here
([`research/VOICE_GENDER.md`](research/VOICE_GENDER.md)).

**Cost on an Apple-silicon laptop** (whole engine, per 160 ms of audio): 115M 28.6 ms on the GPU, 30.4 ms on 2 CPU
threads, 5 real-time streams on the GPU, 1.2 GB; 0.6B 42.9 ms GPU / 97.1 ms CPU, 3 streams on the GPU, 4.9 GB.

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

## Modes and the voice sample

**Single mode** (`audioforge-serve`, the default) is the product: one model, no Silero, no diarizer. It needs the
user's voice print, taken from **at least 5 s of their clean speech, alone (10 s for meetings)**. Store it per user
and send it right after the session config:

```json
{"type": "enroll", "embedding": [0.0213, -0.0871, "... 192 numbers"]}
```

Make one with `audioforge.voiceprint(audio)`, or start the server with `--enroll explicit` and send `{"type":
"enroll"}` to take the next 5 s of speech. Without a stored print the server grabs one live after `agent_end`,
which tracks the user worse ([`research/SINGLE_MODEL.md`](research/SINGLE_MODEL.md)).

**Room mode** (`audioforge-download --diarizer nemotron3`, then `audioforge-serve --mode room`) is opt-in: NVIDIA
Nemotron-3-Diarization labels everyone in the room, and `--final-asr tdt_v3` rewrites each turn with Parakeet-TDT v3.

**Dual rate** (`audioforge-serve --final-chunk-ms 1120`, opt-in): the heads and turn decisions stay on the 160 ms
pass, unchanged byte for byte, and a second pass of the same encoder at NVIDIA's largest trained chunk (1.12 s) writes
the final transcript. The 160 ms text still arrives at the turn end as `final_fast` (what the Pipecat / LiveKit
adapters forward by default); the slow `final` follows 9-48 ms later on the GPU (~105 ms for the 0.6B on CPU, where it
is not recommended). Test sets: ICSI test −3.5 (115M) / −1.9 (0.6B) WER points, live calls −1.7 / −0.8, AMI test
−0.6 / −0.1 (not significant). Cost: +9-27 ms at p95 on turn-end delivery from the slow chunk's burst (+93 ms for the
0.6B on CPU), and fewer real-time streams when sessions start together (same-run comparison): 115M on the GPU 4 → 3, 0.6B on
the GPU 3 → 1
([`research/DUAL_RATE.md`](research/DUAL_RATE.md)).

## A bigger core: `--core 0.6b`

```bash
audioforge-download --core 0.6b          # NVIDIA nemotron-speech-streaming-en-0.6b + its heads (2.5 GB download)
audioforge-serve --core 0.6b --device mps
```

The same server and protocol on NVIDIA's 618M streaming model
([`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), NVIDIA Open
Model License). Every head is retrained on it (speech detector, VAD, speaker, TS-VAD, turn classifiers, LID). The 115M
stays the default. Voice prints belong to one core, so re-enroll after switching. The test-split numbers from the
Results above, side by side (heads v0.4 on both):

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

What you gain: transcripts at about half the error, a better voice print and tracker, a more accurate agent preset
and fewer interruptions on meetings. What it costs: 3× the CPU per chunk, a quarter of the CPU streams, 4× the
memory, and a less permissive licence.

## Integrations

- **Pipecat:** `pip install -e ".[pipecat]"`, `audioforge.integrations.pipecat` (VAD, STT and turn analyzer);
  demo [`examples/pipecat_local_demo.py`](examples/pipecat_local_demo.py).
- **LiveKit Agents:** `pip install -e ".[livekit]"`, `audioforge.integrations.livekit.AudioforgeFrontend`; worker
  [`examples/livekit_agent_worker.py`](examples/livekit_agent_worker.py).
- **Any language:** the WebSocket protocol ([`docs/PROTOCOL.md`](docs/PROTOCOL.md)); a torch-free Python client is in
  [`packages/audioforge-client`](packages/audioforge-client).

## Docs

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the model and the server on one page |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | every flag and what it costs |
| [`docs/PROTOCOL.md`](docs/PROTOCOL.md) | the WebSocket messages |
| [`docs/MODELS.md`](docs/MODELS.md) | the downloaded weights, pinned revisions, licences |
| [`docs/SERVER_INTERNALS.md`](docs/SERVER_INTERNALS.md) | frame clock, policies, enrollment |
| [`research/METRICS.md`](research/METRICS.md) | metric definitions and the full scorecard |
| [`research/README.md`](research/README.md) | index of every lab note, with its key number |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed |

## Development

Set up, tests and conventions are in [`CONTRIBUTING.md`](CONTRIBUTING.md); the binding rules for contributors and
agents are [`AGENTS.md`](AGENTS.md) and [`docs/PROJECT.md`](docs/PROJECT.md). Tasks run through
[`chore`](https://github.com/getchore/chore) from the root `chorefile`: `chore list`, then e.g. `chore lint`,
`chore test` (background, log under `runs/logs/`), `chore serve`. Scripts are `uv` standalone scripts
(`uv run scripts/stream_client.py ...`); plans and scratch notes live in `plans/`.

## Licence

- Code and the trained heads: Apache-2.0 ([`LICENSE`](LICENSE)).
- The NVIDIA encoder: CC-BY-4.0; `audioforge-download` fetches it from NVIDIA and the merged model keeps that licence
  (attribution to NVIDIA).
- The LID head was trained on NVIDIA AmberNet outputs (NGC Terms of Use). Treat it as under those terms until its
  redistribution is confirmed; it is optional.
- Silero VAD is no longer needed in single mode. Room-mode weights keep their own licences. Full list:
  [`NOTICE`](NOTICE), [`docs/MODELS.md`](docs/MODELS.md).
