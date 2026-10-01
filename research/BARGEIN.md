# Barge-in / backchannel classifier: eval set, training data, baselines, first probe

2026-09-30. This is the head we lack (HEAD_ARCHITECTURES.md §5a). This note covers the data and the measured
baselines, plus a first probe on frozen 115M features. Everything ran on the CPU (2 threads, under `scripts/dev/gate.sh`).
Code: `scripts/research/bargein.py`. Numbers: `runs/bargein.json`. Scratch:
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/bargein/`.

## 1. Task and metrics

**Task.** The agent is speaking and the user makes a sound. The classifier decides between three classes:

| class | what the agent should do |
|---|---|
| **backchannel** ("mm-hm", "yeah", "okay", laughter) | keep talking |
| **interruption** (the user takes the floor) | stop |
| **noise** (cough, sigh, other non-speech) | keep talking |

**Metrics:**
- Precision, recall and F1 for each class, plus the macro F1.
- The stop decision's view of the same results:
  - **false-stop rate**: the share of backchannels and noise that make the agent stop.
  - **VAD barge-ins rejected**: the share of all user onsets that do not stop the agent. This is LiveKit's "51 %"
    number.
- **Decision latency** in ms after the user's onset. The classifier is read at 240, 400 and 560 ms.

The reference point is LiveKit adaptive interruption handling: 86 % precision and 100 % recall at 500 ms of overlap,
rejecting 51 % of VAD barge-ins. That was measured on LiveKit's own data, so it cannot be compared directly with the
numbers here.

## 2. Data

### Sources and licences

| corpus | local | licence (file checked) | used for | audio the classifier hears |
|---|---|---|---|---|
| AMI meetings (Mix-Headset, manual words v1.6.2) | 12 train / 4 dev / 4 test meetings | CC BY 4.0 (`data/ami/annotations/LICENCE.txt`) | train + eval | the headset **mix**: the agent's voice is in it too, which is the no-echo-cancellation case |
| ICSI meetings (headset mix, NXT words) | 12 train / 2 dev / 3 test meetings | CC BY 4.0 (`data/icsi/annotations/LICENCE.txt`) | train + eval | the headset **mix** |
| otoSpeech-141h (two-channel calls) | 204 labelled conversations (188 train, 16 eval) | CC BY 4.0. The card forbids identifying speakers, so there is no speaker-identity supervision (`data/oto141/LICENSE.md`) | train + eval | the **user's own channel**, which is the echo-cancelled case |
| Switchboard / Fisher | **not local** (checked `data/`) | LDC | – | – |
| Behavior-SD test shard | local | CC BY 4.0 | not used yet: synthetic TTS. It is the next-phase candidate for backchannel and interruption labels | – |
| TurnBench dev | local | Mundo AI DPL v1.0, **evaluation only** | not used | – |

Nothing was downloaded. For the 16 eval oto conversations, the 16 kHz channel caches were built from the local shards.

### Speaker roles

- **AMI / ICSI.** Every party is taken in turn as the user. The agent is the single other party whose word activity
  covers the user's onset. The pair must be a clean two-party moment: no third party speaks within ±1 s of the
  user's run.
- **oto.** The user is one channel and the agent is the other channel.

### Splits

The corpus **dev** meetings and the 16 oto eval conversations form our eval set:
- AMI: IS1008b, ES2011b, TS3004b, IB4002.
- ICSI: Bmr021, Bns001.
- oto: the 16 E2E eval conversations (`scratch/e2e_tsvad/clips.json`). The ids are listed in `runs/bargein.json`
  under `data.eval_ids`.

The corpus test meetings are held back and not scored here (AMI IS1009b, ES2004b, TS3003b, EN2002a; ICSI Bmr013,
Bmr018, Bro021).

### Label rules

These rules are implemented in `label_run` / `events_of`. The constants are stored in `runs/bargein.json` under
`data.rules`.

1. **Runs.** A party's speech is the union of its timed items with gaps of 0.3 s or less bridged, like a VAD
   hangover.
   - AMI / ICSI items: the reference words. AMI's timed `<vocalsound>` elements (laugh, cough, ...) also count, on
     the user side only.
   - oto items: the Silero VAD v5 segments of each channel, which are the Dyadic label cache.
2. **Candidate.** A user run is a candidate when all of these hold:
   - The user has been silent for at least 1.0 s before the onset.
   - Exactly one other party (the agent) has a run that covers the onset and started at least 1.0 s before it.
   - (Meetings) No third party speaks in [onset − 1 s, run end + 1 s].
   - (ICSI) The run does not overlap an untimed-word zone. (oto) The run does not overlap a redaction zone.
3. **backchannel.** The run lasts 1.0 s or less, and the agent speaks again within 1.0 s of the run's end (the agent
   held the floor). There is also a content condition:
   - AMI / ICSI: every token is a backchannel token (mm, hmm, mm-hmm, uh-huh, yeah, yes, yep, ok, okay, right,
     alright, sure, oh, ah, wow, true, exactly, cool, nice, great, good, really, "i see", 'kay, ...), or the run is
     laughter only.
   - oto: there is no content check (no transcripts), which is DualTurn's rule. Some oto "backchannels" are therefore
     laughs or noises.
4. **interruption.** The run lasts at least 1.5 s, and the agent **yields**:
   - the agent's run ends at least 0.5 s before the user's run ends, and
   - until the user's run ends, the agent says nothing except runs of 1 s or less (its own backchannels).
5. **noise** (AMI only). The run has no words and consists only of non-laughter vocal sounds: cough, sneeze, sigh,
   yawn, sharp inhale / exhale, humming, whistling, ... It lasts 1.5 s or less. ICSI's vocal and non-vocal sounds
   have no times, and oto has no labels, so neither contributes noise.
6. **ambiguous** (counted, excluded from training and scoring). Short non-backchannel words ("no", "well", "so",
   "um"), short runs after which the agent stops, runs of 1-1.5 s, long runs during which the agent keeps talking,
   and AMI vocal sounds of type "other".

Caveat for the duration baselines: the backchannel and interruption classes are defined partly by run length (1 s
or less vs 1.5 s or more). A duration rule read at **1 s or later** reproduces the label rule and is therefore near
perfect (see the table). Only the 240-560 ms rows are informative.

### Class counts

The eval set (clean) has 665 events: 514 backchannel, 140 interruption, 11 noise. Train (clean) has 6,663 events.

| corpus | split | backchannel | interruption | noise | ambiguous (excluded) |
|---|---|---|---|---|---|
| AMI | train | 696 | 154 | 22 | 564 |
| AMI | **eval** (dev) | **124** | **52** | **11** | 115 |
| AMI | test (held back) | 227 | 29 | 5 | 121 |
| ICSI | train | 629 | 158 | 0 | 308 |
| ICSI | **eval** (dev) | **71** | **29** | 0 | 63 |
| ICSI | test (held back) | 142 | 42 | 0 | 79 |
| oto | train (188 conversations) | 3,882 | 1,122 | 0 | 1,106 |
| oto | **eval** (16 conversations) | **319** | **59** | 0 | 63 |

The most frequent short texts are yeah 588, mm-hmm 447, mm 122, right 116, okay 89, yep 66, hmm 65, ok 62,
laugh 61, uh-huh 28, oh 27, yes 25 (`data.top_short_texts`).

**The noise class is almost empty:** 22 train and 11 eval events, all AMI coughs and sighs. A real noise set is
needed before that class can be trained. MUSAN (CC BY 4.0) and the VAD v3 noise plan are in
HEAD_ARCHITECTURES.md §1. Echo is not represented at all.

## 3. Baselines (eval set, measured)

The 665 clean eval events are scored. "int" = interruption. "Rejected" = VAD barge-ins that do not stop the agent
(the ideal is 79 %, the share of non-interruptions). "Latency" = median decision time of the true stops.

**How the user's speech is timed:**
- The VAD run on oto is Silero on the user's channel, which is what Pipecat and LiveKit run.
- On AMI / ICSI it is the reference word activity. This is an oracle VAD: a real VAD on the mix would also hear the
  agent.

**Min-words baselines:**
- On oto they are measured with our streaming RNNT (the served 115M, greedy) decoding the user's channel. Each word
  is timed at its emission.
- On AMI / ICSI the transcript on the mix would include the agent's words, so reference words stamped at their end
  time are used instead. These rows are an upper bound with no ASR delay.

| system | int P | int R | int F1 | backchannel F1 | macro F1 | false-stop | rejected | latency ms |
|---|---|---|---|---|---|---|---|---|
| (i) VAD-only (every onset stops the agent: Pipecat / LiveKit basic) | 0.210 | 1.000 | 0.348 | 0.000 | 0.116 | 1.000 | 0.0 % | 0 |
| (ii) duration > 240 ms | 0.227 | 1.000 | 0.369 | 0.164 | 0.178 | 0.910 | 7.1 % | 240 |
| (ii) duration > 400 ms | 0.331 | 1.000 | 0.497 | 0.624 | 0.374 | 0.539 | 36.4 % | 400 |
| (ii) duration > 500 ms (= LiveKit's `min_interruption_duration` default) | 0.425 | 1.000 | 0.597 | 0.774 | 0.457 | 0.360 | 50.5 % | 500 |
| (ii) duration > 560 ms | 0.438 | 1.000 | 0.609 | 0.787 | 0.465 | 0.343 | 51.9 % | 560 |
| (ii) duration > 800 ms | 0.778 | 1.000 | 0.875 | 0.953 | 0.609 | 0.076 | 72.9 % | 800 |
| (ii) duration > 1000 ms (reproduces the label rule) | 0.993 | 1.000 | 0.996 | 0.990 | 0.662 | 0.002 | 78.8 % | 1000 |
| (iii) min 1 word within 560 ms | 0.285 | 0.507 | 0.365 | 0.723 | 0.362 | 0.339 | 62.6 % | 190 |
| (iii) min 1 non-backchannel word within 560 ms | 0.936 | 0.421 | 0.581 | 0.914 | 0.498 | 0.008 | 90.5 % | 190 |
| (iii) min 2 words within 1000 ms | 0.731 | 0.543 | 0.623 | 0.904 | 0.509 | 0.053 | 84.4 % | 405 |
| (iii) min 2 non-backchannel words within 1500 ms | 0.803 | 0.700 | 0.748 | 0.927 | 0.558 | 0.046 | 81.7 % | 765 |

**By corpus** (interruption P / R, rejected):

| system | AMI | ICSI | oto |
|---|---|---|---|
| VAD-only | 0.28 / 1.00, 0 % | 0.29 / 1.00, 0 % | 0.16 / 1.00, 0 % |
| duration > 240 ms | 0.33 / 1.00, 16 % | 0.35 / 1.00, 18 % | 0.16 / 1.00, 0 % (Silero's 250 ms minimum segment) |
| duration > 400 ms | 0.46 / 1.00, 40 % | 0.59 / 1.00, 51 % | 0.23 / 1.00, 31 % |
| duration > 560 ms | 0.63 / 1.00, 56 % | 0.78 / 1.00, 63 % | 0.30 / 1.00, 47 % |
| min 1 non-backchannel word, 560 ms | 1.00 / 0.62 (oracle words) | 0.96 / 0.79 (oracle words) | 0.57 / **0.07** (our RNNT) |
| min 2 words, 1000 ms | 0.73 / 0.77 (oracle) | 0.84 / 0.93 (oracle) | 0.53 / 0.15 (our RNNT) |

**Reading the baselines:**
- A plain 500-560 ms duration rule already rejects about 51 % of VAD barge-ins at 100 % recall on this set, the same
  rejection rate LiveKit reports for its model. Its precision is 0.43-0.44, against LiveKit's 0.86. One reason the
  duration rule keeps 100 % recall is that our interruption label requires at least 1.5 s of speech. A short "wait!"
  is labelled ambiguous, not interruption.
- Word-count rules are high-precision and low-recall.
- On oto, our streaming RNNT emits almost nothing within 560 ms of an onset on the user's channel: 1 word by 560 ms
  for 7 % of interruptions. That fits the 441 ms p50 partial latency in METRICS.md row 5, plus the time to say the
  word. So a LiveKit-style min-words gate cannot decide by 560 ms with our transcript.

## 4. First probe head (frozen 115M, blocks 4 / 8)

**Features.** Pass-1 encoder blocks 4 and 8 (512 dims each) of the served 115M (`runs/stage1_served_v2.afm`, masked
[70, 1] forward), with the same recipe as the `turn_v5/blk` cache:
- 2,052 train events fall inside a cached `turn_v5/blk` window (at least 3 s after the window start, same audio
  source), and their frames are reused.
- The other 4,611 train events and all 906 eval events (clean + ambiguous) were encoded on the CPU over
  [onset − 8 s, onset + 0.8 / 2 s]. That took 351 s + 101 s.
- Frames run from 4 before the onset to onset + 560 ms. At decision time d the head reads only frames that need no
  audio after onset + d: the last usable 80 ms frame ends 80 ms before onset + d, because of the encoder's one-frame
  look-ahead.

**Head.** 35 k parameters:
- LayerNorm → Linear(1024 → 32) + a learned frame position → GELU → attention pool ++ mean pool → Linear(64 → 3).
- Class-weighted cross-entropy (noise weight capped at 20), random decision length 160-560 ms, 10 % frame dropout,
  AdamW, 15 epochs.
- Every 10th train conversation is held out as validation. The stop threshold is chosen there: P(interruption) at
  95 % validation recall.
- 3 seeds, reported as mean ± std. Training takes about 1 s per seed.

**Results on the eval set** (interruption P / R, rejected VAD barge-ins):

| decision time | features | int AUC (val / eval) | probe alone, stop at val-recall-95 | probe + duration gate (stop iff the VAD run is still on at d **and** P(int) ≥ θ) | 3-class argmax: macro F1 (int F1 / bc F1 / noise F1) |
|---|---|---|---|---|---|
| 240 ms | b4+b8 | 0.74 / 0.71 | 0.227 / 0.94, 12.8 % | 0.242 / 0.94, 18.2 % | 0.45 (0.46 / 0.81 / 0.10) |
| 240 ms | b4 | 0.73 / 0.68 | 0.248 / 0.93, 21.4 % | 0.266 / 0.93, 26.6 % | 0.47 (0.40 / 0.78 / 0.24) |
| 400 ms | b4+b8 | 0.80 / 0.79 | 0.241 / 0.95, 17.0 % | 0.355 / 0.95, 43.6 % | 0.47 (0.57 / 0.85 / 0.00) |
| 400 ms | b4 | 0.78 / 0.77 | 0.247 / 0.98, 16.8 % | 0.361 / 0.98, 43.1 % | 0.53 (0.51 / 0.82 / 0.26) |
| 560 ms | b4+b8 | 0.82 / 0.81 | 0.257 / 0.96, 21.4 % | 0.463 / 0.96, 56.4 % | 0.47 (0.57 / 0.86 / 0.00) |
| 560 ms | b4 | 0.79 / 0.79 | 0.260 / 0.98, 20.6 % | **0.485 / 0.98, 57.5 %** | 0.53 (0.52 / 0.83 / 0.25) |

Seed std is at most 0.01 on precision and at most 0.04 on rejection (`probe.variants.*.summary_mean_std`). By corpus,
the fused b4 head at 560 ms scores:
- AMI: 0.64 / 0.98, 57 %.
- ICSI: 0.88 / 1.00, 67 %.
- oto: 0.33 / 0.97, 55 %.

(Seed 0, from `probe.variants.b4.seed0_detail`.)

**Verdict on the probe:**
- At 240 ms the probe alone beats the duration rule. It rejects 13-21 % of VAD barge-ins at 93-94 % recall, where
  the duration rule rejects 7 % at 100 % recall. Its 3-class macro F1 is 0.45-0.47 against 0.18.
- At 400-560 ms the probe on its own is **worse** than the duration rule. Fused with the duration rule it adds a
  little: at 560 ms, 0.485 precision and 57.5 % rejected at 97.9 % recall, against 0.438 and 51.9 % at 100 % recall.
- It is far from LiveKit's 0.86 precision at 100 % recall.
- No system here reaches 0.86 precision at 95 % or higher recall before the label rule itself does: the duration rule
  gets there at 800-1000 ms.
- The frozen block-4 / 8 features hold some backchannel-vs-floor information (eval AUC 0.79-0.81), and they
  generalise from validation to eval without a gap. They do not separate the classes well within 560 ms.
- The noise class is untrainable with 22 examples.

**Next phase:**
1. Add the proposed inputs that are not in the probe: TS-VAD P(user) and the energy above the floor per frame (on
   the mix these tell the user apart from the agent), and the first RNNT tokens.
2. Get a real noise / echo set (MUSAN, plus TTS-echo simulation).
3. Train on all 136 AMI and 75 ICSI meetings (only 12 train meetings each are local). Behavior-SD's explicit
   backchannel / interruption labels are a further source.
4. Add short interruptions: relabel the "ambiguous / short non-backchannel words" events ("no", "wait") as a
   short-interruption class.
5. Score on the held-back test meetings once, at the end.

## 5. How it will be served

- **When it runs.** Only while the agent's TTS is playing, and only after the user's VAD / TS-VAD onset fires. At
  other times it costs nothing.
- **What it reads.** Block 4 is already computed in pass 1 for the served VAD head, and block 8 comes from the same
  forward, so both are free. The head keeps ≤ 11 frames (4 before the onset + 7 after), about 22 KB of state per
  session.
- **When it decides.** At 240, 400 and 560 ms after the onset:
  - At 240 ms, a confident backchannel or noise lets the agent keep talking, perhaps with the playback ducked.
  - A confident interruption stops the TTS at once.
  - At 560 ms the final decision is taken with the duration gate: stop iff the user is still speaking **and**
    P(interruption) ≥ θ.
  - A late transcript can still overrule the decision (LiveKit's "false interruption → resume").
- **Cost.** 35 k parameters, one attention pool per decision point, well under 0.1 ms on the CPU.
- **Not wired into `audioforge-serve` in this phase.** No served code was changed.

## Reproduce

```
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py build        # events + counts (~1 min)
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py evalfeats    # 906 eval windows (~2 min)
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py trainextra   # 4,611 train windows (~6 min)
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py trainfeats   # onset-aligned arrays
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py baselines
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/bargein.py probe        # 3 feature sets x 3 seeds (~15 s)
```
