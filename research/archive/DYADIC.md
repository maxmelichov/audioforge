# Dyadic (two-party) pipeline and benchmarks: Behavior-SD, DailyTalk, otoSpeech, TurnBench dev

2026-09-26. Code: `audioforge/datasets/dyadic.py` (loader), `scripts/prepare_dyadic.py` (downloads, label caches),
`scripts/bench_turn_dyadic.py` (eot-bench v2 protocol on a dyadic slice), `scripts/bench_turnbench.py` (TurnBench dev
with the official MIT scorer, vendored unchanged in `integrations/turnbench_scorer/`), `research/recipes/stage1_turn_dyadic.yaml`
(not run). Tests: `tests/test_dyadic.py` (hand-made pair: frame grid, floor turn ends, agent_end causality, roles),
`tests/test_bench_turnbench.py` (event commitment rules, vendored scorer on a toy gold). Numbers: `runs/dyadic_bench.json`
(summary), `runs/dyadic_bench_<corpus>.json` (full), `runs/turnbench_dev.json`. Nothing here retrains anything: the head
is `runs/stage1_turn_v3_trail6.afm` exactly as scored on AMI dev in research/archive/EOT_BENCH_V2.md and BASELINES.md.

Question (research/archive/BRIEF.md): our measured bottleneck on AMI is primary-speaker identity (oracle identity moves misses
62 -> 29 %). In a two-party conversation identity is trivial (the other party is the other channel), so how does the
AMI-trained head transfer, and how do the deployable baselines look when the "user" is a known channel?

## 1. Corpora, licences, what was downloaded

| corpus | nature | licence | slice used | layout / labels | party activity we use |
|---|---|---|---|---|---|
| Behavior-SD (`yhytoto12/behavior-sd`) | SYNTHETIC: CosyVoice TTS dialogues from SODA narratives, 52 voices, 2,164 h | CC BY 4.0 (code MIT), ungated | `test/0001.tar` (727 MB, 425 dialogues); first 253 = 4.99 h | stereo FLAC 22.05 kHz, channel k = speaker k (energy check: 99.1 % of 637 utterances; 94.8 % of 192 labelled backchannels are on the LISTENER's channel); per-utterance start/end, backchannels, interruption types; no word timings | utterance intervals spread over tokens by character length (`word_timing = interpolated`) |
| DailyTalkContiguous (`kyutai/DailyTalkContiguous`) | ACTED: DailyTalk (2 actors reading DailyDialog), 21.7 h | CC BY-SA 4.0, ungated | dialogues 0-350 = 3.00 h (1.7 GB) | stereo WAV 44.1 kHz, one actor per channel; word alignments of ONE actor only (all tagged `SPEAKER_MAIN`; the other actor speaks in the gaps with no timestamps) | aligned channel by energy; other channel from Silero VAD v5 (`aligned+vad`); Silero vs alignment frame agreement on the aligned channel 0.935 (p10 0.883) |
| otoSpeech-full-duplex-processed-141h (`otoearth/...`) | REAL remote two-party conversations, denoised, ~18 min each | CC BY 4.0, gated (access granted 2026-09-26); the card prohibits attempts to identify / de-anonymise speakers | first 23 WebDataset shards = 7.4 GB, 207 conversations, ~62 h; labels/caches built for the first 12 (3.16 h) | stereo FLAC 44.1 kHz, session json (topics, redaction intervals, speaker profiles which we do not read) | Silero VAD v5 per channel (`get_speech_timestamps` defaults), each segment one wordless token (`vad`); redactions = excluded zones (3 in 12 conversations) |
| TurnBench dev (`mundo-ai/turn-benchmark-dev`) | REAL two-party, 38 conversations, 7.3 h, 3 annotator tracks per speaker | Dataset Public License v1.0: NON-COMMERCIAL, attribution, no voice cloning. Evaluation only | all 3 parquet shards, 4.2 GB | 48 kHz 24-bit FLAC per speaker, annotations (Turn / backchannel / interruption / Channel Bleed ...) | official: their scorer's consensus gold; v2 protocol: consensus turn spans + backchannels as tokens (`annotated`) |

Total new downloads 14.0 GB (limit 15 GB); derived caches (16 kHz mixes, labels, oto per-channel VAD) ~2.7 GB. The
TurnBench 16 kHz channel / mix caches (3.1 GB) were deleted after scoring to free disk; `Dyadic.channels16k` /
`_clip` rebuild them on demand (~4 min). Licence consequences: no speaker-identity supervision is derived from oto
(`speaker = -1`, profile ids never read); TurnBench dev is used for evaluation only and its numbers must not feed a
commercial claim; DailyTalk derivatives are share-alike; Behavior-SD is fine for anything with attribution.

Label statistics of the slices (`Dyadic.stats`, same code as AMI's `corpus_stats`):

| slice | h | turns | backchannels | switch gap p10 / p50 / p90 (s) | overlap share of switches | hesitations per turn | timeout 1.0 s: cutoff / too late |
|---|---|---|---|---|---|---|---|
| Behavior-SD 253 dialogues | 4.99 | 2597 | 1195 | -0.47 / 0.34 / 0.64 | 0.22 | 0.04 (pauses inside a TTS utterance are invisible) | 0.4 % / 99.9 % |
| DailyTalk 351 dialogues | 3.00 | 2938 | 193 | 0.16 / 0.47 / 0.89 | 0.05 | 0.26 | 4.1 % / 93.6 % |
| oto 12 conversations | 3.16 | 941 | 753 | -1.10 / 0.40 / 1.80 | 0.35 | 2.29 (VAD gaps >= 0.3 s) | 25.1 % / 71.4 % |
| AMI dev (reference, research/archive/AMI.md) | 3.2 | 974 windows | | ~ -0.2 / ~0.3 / ~1.5 | | ~1.2 s median hesitation | |

Read: on the synthetic and acted corpora almost every switch happens within a second (their timing priors), so a 1 s
timeout is "too late" for 94-100 % of switches but almost never cuts anyone off; on real oto audio a 1 s timeout cuts
off 25 % of turns (within-turn VAD gaps) and is still late for 71 % of switches. oto is the only one of the three that
looks like AMI in this respect.

## 2. Pipeline (`audioforge/datasets/dyadic.py`)

- `Dyadic(ids, corpus)` subclasses `ami.AMI`: the same `turn_examples / diar / stats`, the same label rules
  (`ami.speaker_turns`: runs at 0.5 s, floor definition with max_hold 2 s, backchannel <= 2 words and <= 1 s,
  hesitation >= 0.3 s), the same example keys, so recipes and `eval_stage1.turn_windows` run unchanged. Party names
  are `<id>:<channel>`; the primary of a window is recovered through `tag_primary()` (the AMI builders call `gid`
  once per example).
- (a) mixed mono at 16 kHz = the SUM of the two resampled channels, clipped (one microphone hearing both), cached
  as `data/<corpus>/cache/<id>.mix16k.npy`; TurnBench also keeps the two channels (`channels16k`, float16).
- (b) per-party activity on the 80 ms grid from the timestamps (`ami.frames`: [s, e) -> frames int(s/0.08) ..
  ceil(e/0.08)-1), column 0 = primary, column 1 = the other party.
- (c) turn ends per party with our floor definition. Rule differences come from the tokens, not the code:
  Behavior-SD's interpolated words make hesitations invisible and turn the "<= 2 words" half of the backchannel rule
  into a text-length rule; oto / DailyTalk-channel-1 VAD segments have no words, so a backchannel is an isolated VAD
  segment <= 1 s (DualTurn's rule) and a hesitation a >= 0.3 s VAD gap inside a turn.
- (d) `agent_end` (T,): a CAUSAL event stream, 1 on the frame where the other party's activity goes 1 -> 0 (frame 0 if
  it was active just before the window); `agent_end_frame` = the last such event at or before the human's onset
  (-1 if none: the human's first turn, 42 % of Behavior-SD and 24 % of DailyTalk windows); `agent_turn_end` = the
  other party's floor turn ends (labels; a subset of `agent_end`); `first_voice_after(any_act, agent_end)` = the
  "first voice after the agent stops" arming rule on real events. Tests: prefix property on every window, events ==
  1 -> 0 transitions, turn ends subset of events, anchor <= onset.
- Roles: `agent_rule = second` (the party whose first voice comes later is the agent); `alternate` and fixed channels
  exist. `turn_examples(roles=("human",))` keeps the human's turns.
- `recipe_data` (`data: {dyadic: {corpus, hours, roles, ...}}`) is registered in `audioforge.train._load_source`;
  `split_ids` = every 5th conversation is dev.

## 3. eot-bench v2 on the dyadic slices (no retraining)

Protocol = research/archive/EOT_BENCH_V2.md: the HUMAN party's non-backchannel turns as the library's windows (20 s, lead 4 s;
block A trail 2 s, block C the same starts with a 6 s trail; horizons 25 / 75 post-end emission frames), operating
point at <= 5 % per-turn false cutoffs cross-fitted over conversation halves, 1000 bootstraps, floor strata at 1.04 s.
The head is fed the ORACLE party track (spk_act = the human's labelled activity, cols = [human, agent, 0, 0]): what a
two-channel product has. Streaming Sortformer rows: NOT run on these sets. Window-mode streaming Sortformer v2 costs
39 s of CPU per 20 s window on this machine (measured), i.e. 6.5 h per 600-window set; the rules (CPU only, <= 10 min
per process, machine under load) did not leave room for it. The oracle-track rows are the identity-free reading the
brief asked for; the identity problem is by construction absent on a two-channel input.

Strata note: on Behavior-SD 0 of 652 human turn ends are floor-open (the other party always replies within 1.04 s:
the timing prior N(0.4, 0.2) s), on DailyTalk 40 of 650. The "open" column, our hard case on AMI, barely exists on the
synthetic / acted data; oto has it.

### behavior_sd: 652 human-party turn ends, 137 conversations (post-end frames p50 57, end reasons {'resume': 410, 'trail': 242, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 4.5 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 320 | 100.0 (0) | 0.0 (652) |
| timeout_1040ms_primary_oracle | 0.1 | 0.6 [0.1, 1.2] | 0.6 [0.1, 1.2] | 1040 | 100.0 (0) | 0.6 (652) |
| timeout_any_speaker_oracle | 3.7 | 18.1 [15.1, 21.0] | 25.0 [21.5, 28.2] | 80 | 100.0 (0) | 18.1 (652) |
| silero_timeout | 6.4 | 97.7 [96.5, 98.8] | 99.7 [99.2, 100.0] | inf | 100.0 (0) | 97.7 (652) |
| silero_timeout_1000ms | 8.0 | 96.5 [95.0, 97.8] | 99.3 [98.5, 100.0] | inf | 100.0 (0) | 96.5 (652) |
| head_trail6_oracle_track | 4.3 | 3.0 [1.8, 4.5] | 6.9 [5.1, 9.0] | 880 | 100.0 (0) | 3.0 (652) |
| hybrid_oracle | 5.7 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 240 | 100.0 (0) | 0.0 (652) |
| head_trail6+silero_timeout | 4.6 | 3.0 [1.8, 4.5] | 6.9 [5.1, 9.0] | 880 | 100.0 (0) | 3.0 (652) |
| eou_posterior | 4.8 | 57.5 [53.4, 61.3] | 62.6 [58.9, 66.3] | inf | 100.0 (0) | 57.5 (652) |
| eou_native (<EOU> emitted) | 2.9 | 63.8 [60.0, 67.3] | 68.2 [64.5, 71.6] | inf | 100.0 (0) | 63.8 (652) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +3.0 [+1.8, +4.5] | n/a | +3.0 |
| hybrid_oracle - timeout_primary_oracle | +0.0 [+0.0, +0.0] | n/a | +0.0 |
| hybrid_oracle - head_trail6_oracle_track | -3.0 [-4.6, -1.8] | n/a | -3.0 |
| silero_timeout - timeout_primary_oracle | +97.7 [+96.5, +98.8] | n/a | +97.7 |
| head_trail6_oracle_track - silero_timeout | -94.7 [-96.3, -92.8] | n/a | -94.7 |
| head_trail6+silero_timeout - silero_timeout | -94.7 [-96.3, -92.8] | n/a | -94.7 |
| eou_posterior - silero_timeout | -40.2 [-44.4, -36.1] | n/a | -40.2 |
| eou_posterior - timeout_primary_oracle | +57.5 [+53.4, +61.3] | n/a | +57.5 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +18.1 [+15.1, +21.0] | n/a | +18.1 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 0.0 | 8.3 [6.1, 10.3] | 8.3 [6.1, 10.3] | 1520 | 100.0 | 8.3 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 0.0 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | inf | 100.0 | 100.0 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 0.1 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | inf | 100.0 | 100.0 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 0.9 | 6.8 [5.0, 8.8] | 14.5 [11.9, 17.5] | 1280 | 100.0 | 6.8 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 0.9 | 5.0 [3.2, 6.7] | 5.0 [3.2, 6.7] | 1280 | 100.0 | 5.0 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 0.9 | 7.0 [5.1, 9.1] | 15.3 [12.5, 18.5] | 1280 | 100.0 | 7.0 |

### dailytalk: 650 human-party turn ends, 157 conversations (post-end frames p50 45, end reasons {'resume': 554, 'trail': 96, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 4.6 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 720 | 0.0 (40) | 0.0 (610) |
| timeout_1040ms_primary_oracle | 3.2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1040 | 0.0 (40) | 0.0 (610) |
| timeout_any_speaker_oracle | 5.1 | 50.1 [46.0, 54.1] | 64.5 [60.9, 68.1] | inf | 0.0 (40) | 53.2 (610) |
| silero_timeout | 4.8 | 72.4 [68.6, 76.0] | 80.1 [76.9, 83.2] | inf | 5.3 (40) | 76.8 (610) |
| silero_timeout_1000ms | 2.0 | 89.6 [87.2, 92.1] | 92.8 [90.7, 94.8] | inf | 10.0 (40) | 95.0 (610) |
| head_trail6_oracle_track | 4.9 | 1.3 [0.5, 2.2] | 3.9 [2.6, 5.5] | 880 | 2.7 (40) | 1.2 (610) |
| hybrid_oracle | 5.2 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 720 | 0.0 (40) | 0.0 (610) |
| head_trail6+silero_timeout | 5.4 | 1.3 [0.5, 2.3] | 3.7 [2.4, 5.3] | 960 | 2.7 (40) | 1.2 (610) |
| eou_posterior | 4.9 | 87.9 [85.2, 90.5] | 92.1 [89.9, 94.1] | inf | 76.3 (40) | 88.6 (610) |
| eou_native (<EOU> emitted) | 0.5 | 93.3 [91.5, 95.2] | 96.8 [95.2, 98.0] | inf | 84.6 (40) | 93.9 (610) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +1.3 [+0.5, +2.2] | +2.7 | +1.2 |
| hybrid_oracle - timeout_primary_oracle | +0.0 [+0.0, +0.0] | +0.0 | +0.0 |
| hybrid_oracle - head_trail6_oracle_track | -1.3 [-2.2, -0.5] | -2.7 | -1.2 |
| silero_timeout - timeout_primary_oracle | +72.4 [+68.6, +76.0] | +5.3 | +76.8 |
| head_trail6_oracle_track - silero_timeout | -71.1 [-74.9, -67.2] | -2.6 | -75.6 |
| head_trail6+silero_timeout - silero_timeout | -71.1 [-74.9, -67.2] | -2.6 | -75.5 |
| eou_posterior - silero_timeout | +15.5 [+11.4, +19.5] | +71.0 | +11.9 |
| eou_posterior - timeout_primary_oracle | +87.9 [+85.2, +90.5] | +76.3 | +88.6 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +50.1 [+46.0, +54.1] | +0.0 | +53.2 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 1.2 | 3.1 [1.9, 4.7] | 3.1 [1.9, 4.7] | 1520 | 10.3 | 2.6 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 0.1 | 97.7 [96.5, 98.8] | 98.6 [97.7, 99.4] | inf | 77.5 | 99.0 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 0.0 | 99.7 [99.2, 100.0] | 99.9 [99.5, 100.0] | inf | 97.5 | 99.8 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 1.5 | 8.4 [6.4, 10.7] | 23.0 [20.1, 26.2] | 1600 | 15.0 | 8.0 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 1.5 | 3.1 [1.7, 4.5] | 3.1 [1.7, 4.5] | 1600 | 10.0 | 2.7 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 1.5 | 8.4 [6.4, 10.7] | 23.9 [20.9, 27.1] | 1600 | 15.0 | 8.0 |

### oto: 581 human-party turn ends, 16 conversations (post-end frames p50 52, end reasons {'resume': 380, 'trail': 201, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 5.2 | 11.1 [8.4, 13.8] | 11.1 [8.4, 13.8] | 1600 | 1.8 (177) | 14.9 (404) |
| timeout_1040ms_primary_oracle | 19.3 | 1.3 [0.4, 2.4] | 1.3 [0.4, 2.4] | 1040 | 0.0 (177) | 1.7 (404) |
| timeout_any_speaker_oracle | 5.5 | 75.8 [72.2, 79.2] | 81.4 [78.3, 84.4] | inf | 38.8 (177) | 91.7 (404) |
| silero_timeout | 7.4 | 77.7 [74.1, 81.2] | 83.8 [80.7, 86.7] | inf | 45.1 (177) | 91.8 (404) |
| silero_timeout_1000ms | 27.2 | 54.6 [49.5, 59.4] | 61.9 [56.9, 66.4] | inf | 0.0 (177) | 72.9 (404) |
| head_trail6_oracle_track | 5.5 | 12.9 [10.1, 15.8] | 32.8 [28.9, 36.8] | 1680 | 12.0 (177) | 13.4 (404) |
| hybrid_oracle | 6.7 | 10.2 [7.6, 12.7] | 10.2 [7.6, 12.7] | 1680 | 1.9 (177) | 13.6 (404) |
| head_trail6+silero_timeout | 6.9 | 11.5 [8.7, 14.2] | 29.0 [25.2, 33.1] | 1760 | 4.9 (177) | 14.4 (404) |
| eou_posterior | 2.4 | 94.0 [91.9, 95.9] | 94.2 [92.1, 96.0] | inf | 88.4 (177) | 96.5 (404) |
| eou_native (<EOU> emitted) | 10.3 | 72.7 [68.7, 76.5] | 77.3 [73.8, 80.9] | inf | 46.2 (177) | 84.1 (404) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | +1.9 [-0.5, +4.2] | +10.1 | -1.6 |
| hybrid_oracle - timeout_primary_oracle | -0.9 [-2.2, +0.2] | +0.0 | -1.3 |
| hybrid_oracle - head_trail6_oracle_track | -2.8 [-4.8, -0.8] | -10.1 | +0.3 |
| silero_timeout - timeout_primary_oracle | +66.6 [+62.8, +70.6] | +43.2 | +76.8 |
| head_trail6_oracle_track - silero_timeout | -64.8 [-69.1, -60.6] | -33.1 | -78.4 |
| head_trail6+silero_timeout - silero_timeout | -66.2 [-70.1, -62.3] | -40.2 | -77.4 |
| eou_posterior - silero_timeout | +16.3 [+12.6, +20.2] | +43.4 | +4.7 |
| eou_posterior - timeout_primary_oracle | +82.9 [+79.8, +86.1] | +86.6 | +81.5 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +64.7 [+60.8, +68.5] | +36.9 | +76.8 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 7.1 | 9.8 [7.3, 12.6] | 9.8 [7.3, 12.6] | 1520 | 1.3 | 13.4 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 6.4 | 74.6 [71.2, 78.1] | 80.3 [77.2, 83.4] | inf | 35.2 | 91.4 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 2.8 | 80.2 [76.9, 83.4] | 85.7 [82.8, 88.6] | inf | 49.1 | 93.9 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 16.7 | 4.5 [2.9, 6.5] | 12.0 [9.2, 15.1] | 1200 | 2.8 | 5.3 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 15.8 | 4.1 [2.5, 5.9] | 4.1 [2.5, 5.9] | 1200 | 1.4 | 5.2 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 15.8 | 4.5 [2.9, 6.4] | 11.9 [9.1, 14.8] | 1200 | 2.7 | 5.2 |

### turnbench: 419 both-party turn ends, 10 conversations (post-end frames p50 58, end reasons {'resume': 255, 'trail': 164, 'meeting_end': 0})

Block C (6 s windows), cross-fitted <= 5 % per-turn FC (natives at their fixed point):

| system | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % (n) | taken miss % (n) |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 6.0 | 5.3 [3.1, 7.5] | 5.3 [3.1, 7.5] | 1440 | 2.4 (51) | 5.7 (368) |
| timeout_1040ms_primary_oracle | 6.0 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 1040 | 0.0 (51) | 0.0 (368) |
| timeout_any_speaker_oracle | 5.2 | 80.3 [76.3, 84.2] | 84.4 [80.9, 88.1] | inf | 7.5 (51) | 88.5 (368) |
| silero_timeout | 7.9 | 90.7 [87.6, 93.6] | 93.5 [91.0, 95.9] | inf | 42.9 (51) | 96.5 (368) |
| silero_timeout_1000ms | 15.5 | 82.5 [78.2, 86.4] | 85.9 [82.1, 89.5] | inf | 3.3 (51) | 89.8 (368) |
| head_trail6_oracle_track | 4.3 | 4.2 [2.5, 6.3] | 9.7 [6.8, 12.5] | 1120 | 8.5 (51) | 3.7 (368) |
| hybrid_oracle | 7.6 | 2.3 [0.8, 3.9] | 2.3 [0.8, 3.9] | 800 | 2.4 (51) | 2.3 (368) |
| head_trail6+silero_timeout | 7.4 | 3.6 [1.8, 5.5] | 7.8 [5.0, 10.4] | 1120 | 2.3 (51) | 3.8 (368) |
| eou_posterior | 7.9 | 92.8 [90.2, 95.1] | 95.9 [93.8, 97.7] | inf | 62.8 (51) | 96.5 (368) |
| eou_native (<EOU> emitted) | 1.4 | 96.6 [94.9, 98.1] | 98.6 [97.3, 99.5] | inf | 88.2 (51) | 97.8 (368) |

Paired differences at 6 s (miss points, 95 % CI; open / taken point estimates):

| a - b | all | open | taken |
|---|---|---|---|
| head_trail6_oracle_track - timeout_primary_oracle | -1.1 [-3.7, +1.5] | +6.1 | -2.0 |
| hybrid_oracle - timeout_primary_oracle | -3.0 [-5.1, -1.0] | +0.0 | -3.4 |
| hybrid_oracle - head_trail6_oracle_track | -1.9 [-3.4, -0.7] | -6.1 | -1.4 |
| silero_timeout - timeout_primary_oracle | +85.3 [+81.9, +88.9] | +40.4 | +90.8 |
| head_trail6_oracle_track - silero_timeout | -86.4 [-90.1, -82.9] | -34.4 | -92.8 |
| head_trail6+silero_timeout - silero_timeout | -87.1 [-90.4, -83.7] | -40.5 | -92.7 |
| eou_posterior - silero_timeout | +2.1 [-0.8, +4.9] | +19.9 | -0.0 |
| eou_posterior - timeout_primary_oracle | +87.4 [+84.2, +90.6] | +60.4 | +90.8 |
| timeout_any_speaker_oracle - timeout_primary_oracle | +75.0 [+70.7, +79.3] | +5.1 | +82.8 |

Frozen AMI operating points (fitted on all 974 AMI dev turns, applied unchanged):

| system | AMI point | FC % | miss 6 s % [CI] | miss 2 s % [CI] | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|---|
| timeout_primary_oracle | {"timeout_threshold": 18.0, "k_frames": 19, "k_ms": 1520.0} | 1.9 | 8.3 [5.6, 10.9] | 8.3 [5.6, 10.9] | 1520 | 2.2 | 9.0 |
| timeout_any_speaker_oracle | {"timeout_threshold": 17.0, "k_frames": 18, "k_ms": 1440.0} | 2.1 | 88.5 [85.3, 91.4] | 92.0 [89.2, 94.4] | inf | 23.9 | 96.7 |
| silero_timeout | {"timeout_threshold": 23.200000762939453, "k_frames": 24, "k_ms": 1920.0} | 3.8 | 92.3 [89.8, 94.8] | 94.5 [92.3, 96.8] | inf | 47.8 | 98.0 |
| head_trail6_oracle_track | {"theta": 0.9791455821976657} | 6.4 | 3.8 [2.0, 5.8] | 8.2 [5.4, 10.8] | 1040 | 8.9 | 3.2 |
| hybrid_oracle | {"theta": 0.9798172116279602, "timeout_threshold": 23.0} | 6.2 | 2.8 [1.2, 4.4] | 2.8 [1.2, 4.4] | 1040 | 2.2 | 2.9 |
| head_trail6+silero_timeout | {"theta": 0.9798172116279602, "timeout_threshold": 46.79999923706055, "k_frames": 47} | 6.4 | 3.6 [1.8, 5.6] | 8.4 [5.8, 10.8] | 1040 | 6.7 | 3.2 |

**Reading the v2 tables.**
- With the party's own activity known (a two-channel product), the identity problem is gone and so is most of the
  difficulty: on all four sets the plain primary timeout at <= 5 % FC misses 0-11 % at 6 s, and the AMI-trained trail6
  head on that track is within a few points of it (Behavior-SD +3.0, DailyTalk +1.3, oto +1.9 [-0.5, +4.2], TurnBench
  -1.1 [-3.7, +1.5]); the hybrid (head OR timeout) is never worse than the timeout and is the best row everywhere (oto
  10.2 %, TurnBench 2.3 %, P50 0.8-1.7 s). On AMI the same head with the oracle track sat at 29 % (research/archive/BRIEF.md);
  here it is 1-13 %. The head transfers; what it could not do on AMI was know who the user was.
- The speaker-unaware rows collapse on dyadic audio exactly as on AMI: the Silero timeout on the mixed mono misses
  72-98 % (it waits for BOTH parties to be silent, and the other party answers within a second), the any-speaker oracle
  timeout 18-80 %. Parakeet-EOU on the mix misses 58-94 %. The 1000 ms native timeouts confirm the label statistics of
  section 1: on oto a 1040 ms primary timeout cuts off 19 % of turns (VAD gaps inside turns), on TurnBench 6 %, on the
  synthetic / acted sets 0-3 %.
- Floor-open ends (the AMI hard case): only oto (177) and TurnBench (51) have them in numbers. There the head alone
  loses to the timeout on open ends (+10 / +6 points) and the hybrid recovers it (1.9 / 2.4 %). Same shape as AMI, ten
  times smaller.
- Frozen AMI operating points (no re-fitting at all): the AMI thresholds transfer to DailyTalk / TurnBench with 1-6 %
  FC and 3-8 % misses; on oto the head's AMI theta gives 16-17 % false cutoffs (the VAD-segment "words" produce
  within-turn gaps the AMI label convention hides, DATA_PLAN.md cause 1), so a real deployment on this audio needs the
  threshold re-fitted or the VAD gaps bridged. The AMI Silero timeout (1.92 s) and any-speaker points are useless
  everywhere (100 % miss): on two-party audio the other party always speaks within 2 s.
- Not run: streaming Sortformer tracks with causal_dominant enrollment (cost, section 3 header); the oracle-track rows
  are exactly the "identity solved" condition the brief asked about.


## 4. TurnBench dev with the official scorer

Official protocol (`scripts/bench_turnbench.py`, scorer vendored unchanged at commit
`integrations/turnbench_scorer/VENDORED_COMMIT.txt`): every system runs causally on the 38 conversations (7.3 h, both
speakers = 76 speaker-channels), commits discrete EOT events (timeouts: the end of the 32 ms chunk completing k s of
silence; head / EOU: rising edge above theta with their 2 s refractory, committed at the end of the 160 ms encoder
chunk), and the dev operating point is their rule: highest recall at fp_rate <= 0.10 over the system's own score
quantiles. `runs/turnbench_dev.json` holds the full curves and `<work>/preds/<system>.json` the frozen predictions
(the test submission needs only the test parquet). Inputs: `silero_timeout` / `rms_timeout` / `eou` per speaker
channel; `head_trail6` on the MIXED MONO with spk_act = that speaker's Silero activity and cols = [speaker, other, 0, 0]
(the two-channel product input; the streaming Sortformer track was not run: 39 s CPU per 20 s of audio here);
`silero_mix_timeout` / `head+silero_mix` on the mix alone (speaker-unaware controls).

| system | operating point | recall | fp_rate | latency p10 / p50 / p90 ms |
|---|---|---|---|---|
| rms_timeout | 2.0 | 0.648 | 0.071 | 604 / 1653 / 1905 |
| silero_timeout | 1.5 | 0.815 | 0.075 | 844 / 1399 / 1525 |
| silero_mix_timeout | 1.2 | 0.117 | 0.091 | 901 / 1210 / 2418 |
| head_trail6 | 0.9973489046096802 | 0.752 | 0.099 | 237 / 1058 / 2258 |
| hybrid | [0.9988547563552856, 1.5] | 0.835 | 0.080 | 760 / 1390 / 1537 |
| head+silero_mix | [0.9988547563552856, 1.5] | 0.353 | 0.051 | 914 / 1884 / 2807 |
| eou_posterior | -749.5036926269531 | 0.763 | 0.084 | 401 / 1097 / 2151 |
| eou_native | None | 0.123 | 0.003 | 648 / 1788 / 2733 |

Published dev predictions re-scored with the same vendored scorer (their committed operating points):

| baseline | recall | fp_rate | p50 ms | published test recall / fp / p50 |
|---|---|---|---|---|
| vap | 0.841 | 0.045 | 463 | 0.845 / 0.055 / 368 |
| smart_turn_v3 | 0.754 | 0.100 | 1010 | 0.752 / 0.047 / None |
| kyutai_semantic_vad | 0.803 | 0.100 | 1024 | 0.773 / 0.059 / 1007 |
| rms_vad | 0.595 | 0.547 | -98 | - |
| espnet_turntaking | 0.836 | 0.074 | 895 | - |
| espnet_turntaking_perchannel | 0.640 | 0.100 | 846 | - |
| wavlm_large_causal | 0.472 | 0.100 | 683 | - |
| mimi_endpointer | 0.759 | 0.047 | 742 | - |
| openai_server_vad | 0.933 | 0.564 | 281 | - |
| openai_semantic_vad | 0.310 | 0.037 | 763 | - |
| oracle_annotator | 1.000 | 0.000 | 0 | - |

Consistency check: another agent's independent run of this script on a completeness head (`runs/turnbench_dev_
completeness.json`) reproduces the Silero per-channel row exactly (0.8146 / 0.0753 / 1399 ms).

**Reading.**
- Our AMI-trained head on the mixed mono + Silero track reaches 0.752 recall at fp 0.099 with P50 1058 ms: equal to
  smart-turn v3 (0.754 / 0.100 / 1010 on dev, 0.752 on test) and Kyutai's semantic VAD on latency, below VAP (0.841 /
  0.045 / 463 ms) and espnet dual-channel (0.836), and below the plain per-channel Silero timeout (0.815 / 0.075 /
  1399 ms). The hybrid head-OR-Silero is the best of ours: 0.835 / 0.080 / 1390 ms, level with espnet and 1 point
  under VAP in recall but 0.9 s slower.
- The two speaker-unaware mono rows are the AMI story again: Silero on the mix 0.117, head OR mix timeout 0.353. On
  TurnBench the other speaker replies fast, so anything that waits for the whole room to fall silent misses.
- Parakeet-EOU (per channel) 0.763 / 0.084 / 1097 ms: the same level as the head; its native <EOU> emission fires on
  12 % of turn ends only.
- The head's curve is steep (theta between 0.994 and 0.999): at fp 0.147 it reaches 0.756 with P50 827 ms, at fp
  0.062 it drops to 0.720. Which turn ends it misses relative to the Silero timeout was not analysed (the per-event
  outcomes are in the scorer, not in our JSON); TurnBench's negatives are hand-labelled mid-turn pauses, which AMI's
  word-gap convention never showed the head.
- eot-bench v2 on the same audio (10 conversations, 419 turn ends, table above) ranks the systems the same way with
  the identity given (hybrid 2.3 % miss, head 4.2 %, timeout 5.3 %; Silero mix 90.7 %), so the two protocols agree
  once the input condition is matched. The absolute numbers differ because TurnBench's FP denominator is hand-labelled
  pauses and backchannels (their fp 0.10 budget allows more firing inside long turns than our <= 5 % per-turn FC).


## 5. Honest caveats

- Behavior-SD is TTS: no real hesitations, no breath, timing from Gaussian priors; its backchannels are rendered on
  the listener channel but the listener's channel is otherwise digital silence. DailyTalk is two actors reading a
  script with only one channel aligned; the other channel's activity is Silero's. oto is real but denoised and its
  "words" are VAD segments (so per-pause FC is VAD-gap FC). TurnBench dev is real and hand-labelled but non-commercial.
- Behavior-SD word timings are interpolated inside utterances: a detector firing in a real intra-utterance TTS pause
  is charged as a false cutoff, which is correct, but hesitation strata / per-pause FC are meaningless there.
- Block A head / EOU scores are the first frames of the block C causal stream (no re-run with a 2 s end-of-file
  flush), unlike the AMI tables which re-ran the 2 s windows; this only affects the last ~13 frames of block A.
- All 16 kHz audio is resampled from 22.05 / 44.1 / 48 kHz with torchaudio; Silero runs at 16 kHz (their RMS baseline
  runs at 48 kHz; we apply the same 20 ms / 0.01 rule at 16 kHz).
- Folds are conversation halves, not speaker-disjoint (DailyTalk has 2 actors; oto and TurnBench have repeated
  participants across sessions).

## 6. What changes once more real data is in hand

- TurnBench dev is scored (section 4). The test split (116 conversations, labels withheld) needs an e-mail
  submission with a public write-up; `scripts/bench_turnbench.py` writes the predictions JSONs
  (`<work>/preds/<system>.json`) at the dev operating points, so a test run is the same script on the test parquet
  (`resolve_dataset("mundo-ai/turn-benchmark-test")`) plus an e-mail.
- The streaming Sortformer rows on dyadic audio (our deployable cascade with `causal_dominant` enrollment) are the one
  missing comparison: ~6.5 CPU-hours per 600-window set on this machine, or ~15 min on MPS with
  `eval_stage1.v2_tracks(..., device="mps")`; `bench_turn_dyadic.py --stage tracks / scores --max-tracks N` is wired.
- Training on real dyadic audio (`research/recipes/stage1_turn_dyadic.yaml`, 20 h of oto + AMI + synthetic, agent column exact)
  is the first experiment that can move the "open" stratum on real two-party data; the control is the same head
  scored here without retraining. Full 141 h on a GPU: ~8 CPU-hours of Silero labels (or our VAD head on MPS), 16 GB
  float16 mix cache (or decode on the fly), ~30 k human+agent turn windows, ~50 min per epoch at the DATA_PLAN.md
  rate; 3 epochs + WER gate = one GPU afternoon. Kill rule (OUTSIDE.md 1.2): TurnBench-dev EOT recall at fp <= 0.10
  does not beat the per-channel Silero timeout row above.
- Licence gates that remain closed: otoSpeech-turn-104h (hand-labelled, non-commercial, 290 GB) and TurnBench train;
  neither is needed for the above.


## 7. Latency: policies fitted on dyadic data

2026-09-26 (the "§4 latency" follow-up; numbered 7 because sections 4-6 already existed). Code:
`scripts/bench_turnbench_latency.py` (stages oto_vad / oto_head / sweep / report / v2 / tables); numbers:
`runs/turnbench_latency.json`; tests: `tests/test_bench_turnbench_latency.py` (exponential wait, isotonic calibration
monotone + pointwise-causal + fitting-split-only, silence / velocity / head branches, the scorer-matching replica, oto
gold rules). Nothing is retrained. Question: section 4's best system (head OR per-channel Silero, 0.835 / 0.080 /
P50 1390 ms) is ~0.9 s slower than VAP (0.841 / 0.045 / 463 ms). How much of that gap is the decision policy, which
was fitted on AMI or picked on TurnBench's own dev set, and how much is the model?

**Inputs** exactly as section 4: per speaker channel, Silero v5 + the Pipecat state machine on that channel (silence =
time since the last speech chunk), and the trail6 head on the mixed mono with that channel's Silero track as the party
input (p at time τ = the latest frame emitted by τ, 160 ms chunks). TurnBench streams are the section 4 ones
(`work_official/{vad,head}`); for oto the same two stages were run on the 16 conversations of section 3 (4.3 h,
`<scratch>/dyadic/oto/work_tb`, ~18 min of CPU).

**Policies** (per channel; one silence-branch event per silence run, head-branch events = rising edges above θ; the
two OR-ed with the scorer's 2 s refractory):
(a) `fixed`: p > θ OR silence ≥ k (section 4's hybrid);
(b) `lin`: p > θ OR silence ≥ clamp(T0 − a·p, Tmin, T0) (BASELINES.md), on the raw p; `lin_cal`: the same on phat;
(c) `exp`: p > θ OR silence ≥ Tmin + (Tmax − Tmin)·exp(−λ·phat^γ), phat = isotonic calibration of p fitted on the
fitting split only (labels at the user's silent chunks: 1 in (end, end + 3 s] of a gold EOT, 0 inside a gold pause,
each gold event weighted once); the map is pointwise, so phat is as causal as p;
(d) `vel`: (c) + early exit at Tmin if phat > 0.85 and phat rose by > 0.5 within the last 2 head frames at any time
in the current silence run (latched: fire at max(trigger, run start + Tmin)).
Grids: θ ∈ {0.9985 .. 0.9996, off}, k 0.5-3 s (84 points); lin T0 1-3 s, a 0.25-2.5 s, Tmin 0.2-0.7 s (840); exp
Tmin 0.2-0.7, Tmax 1.2-3, λ 1-8, γ 0.5-4 (3500).

**Scoring.** TurnBench dev: the vendored scorer's own gold and `score_task`, per speaker (identical to
`score_submission`: the section 4 hybrid point reproduces 0.8351 / 0.0800 / 1390 ms through both paths, and every
sweep point asserts that our TP-to-branch attribution matches the scorer's TPs and latencies). **oto** has no human
TurnBench labels (otoSpeech-turn-104h is gated and non-commercial), so its gold is the scorer's own floor construction
(`turnbench.gold.build_conversation_events`) applied to our per-channel Silero segments: runs of ≤ 2 segments lasting
≤ 1 s and separated by ≥ 0.5 s are Backchannel, everything else Turn, Turn segments merged across gaps < 0.2 s (with
that merge oto's pause-span p10 / p50 / p90 = 0.2 / 0.4 / 1.1 s matches TurnBench dev's 0.1 / 0.42 / 1.21 s; oto has
2.5 pauses per EOT against TurnBench's 0.56), redactions excluded. oto: 1252 EOT positives (629 of the human party,
the section 3 role) and 3166 pauses; TurnBench dev: 1904 / 1063. Caveat: the oto gold comes from the same Silero
that drives the silence branch, which flatters silence rules on oto.

**Protocol.** Fit = the lowest P50 with recall ≥ 0.80 and fp ≤ 0.10 on the FITTING split, then scored unchanged on
the evaluation split: (i) fit on oto (both parties), evaluate on all of TurnBench dev; (ii) fit on one half of
TurnBench dev (conversations at even / odd positions of the sorted ids, calibration refitted on that half), evaluate
on the other, both directions, and pool the two held-out halves. The "in-sample" rows below are selected on
TurnBench dev itself and are labelled so; they describe what the grid can reach, not a held-out result.

**Chosen points and held-out scores** (recall / fp / P50 / P90 ms; head share = fraction of TPs whose matched event
came from the head branch):

| family | protocol | chosen point | fit: recall / fp / P50 / P90 | HELD-OUT: recall / fp / P50 / P90 | held-out head share | held-out P50 head / silence ms |
|---|---|---|---|---|---|---|
| fixed | oto->tb | θ 0.9992, k 1.2 s | 0.813 / 0.099 / 1144 / 2076 | 0.851 / 0.129 / 1115 / 1239 | 0.002 | 1039 / 1115 |
| fixed | tbA->tbB | θ 0.9988, k 1.5 s | 0.833 / 0.095 / 1390 / 1531 | 0.841 / 0.070 / 1382 / 1547 | 0.096 | 1148 / 1389 |
| fixed | tbB->tbA | θ 0.9985, k 1.4 s | 0.854 / 0.077 / 1262 / 1455 | 0.855 / 0.129 / 1264 / 1426 | 0.223 | 845 / 1306 |
| fixed | TB halves pooled | (per half) | | 0.849 / 0.098 / 1316 / 1513 | 0.168 | 918 / 1338 |
| lin | oto->tb | θ 0.9985, T0 1.5, a 0.25, Tmin 0.2 | 0.819 / 0.098 / 1200 / 2077 | 0.867 / 0.123 / 1147 / 1317 | 0.173 | 790 / 1179 |
| lin | tbA->tbB | θ 0.9988, T0 2.0, a 0.5, Tmin 0.2 | 0.833 / 0.095 / 1396 / 1541 | 0.841 / 0.069 / 1388 / 1551 | 0.096 | 1148 / 1399 |
| lin | tbB->tbA | θ 0.9985, T0 1.5, a 0.25, Tmin 0.2 | 0.869 / 0.099 / 1153 / 1350 | 0.866 / 0.151 / 1144 / 1303 | 0.185 | 786 / 1184 |
| lin | TB halves pooled | (per half) | | 0.855 / 0.107 / 1228 / 1514 | 0.147 | 845 / 1246 |
| lin_cal | oto->tb | θ 0.9985, T0 2.5, a 2.5, Tmin 0.2 | 0.818 / 0.096 / 1040 / 2056 | 0.852 / 0.125 / 970 / 1860 | 0.072 | 902 / 972 |
| lin_cal | tbA->tbB | θ 0.9985, T0 2.5, a 1.0, Tmin 0.2 | 0.818 / 0.093 / 1484 / 1816 | 0.829 / 0.046 / 1522 / 1919 | 0.233 | 1010 / 1574 |
| lin_cal | tbB->tbA | θ 0.9994, T0 3.0, a 2.5, Tmin 0.7 | 0.840 / 0.090 / 930 / 1762 | 0.829 / 0.176 / 828 / 1701 | 0.001 | 1571 / 827 |
| lin_cal | TB halves pooled | (per half) | | 0.829 / 0.106 / 1106 / 1823 | 0.102 | 1012 / 1123 |
| exp | oto->tb | θ 0.999, Tmin 0.5, Tmax 3.0, λ 3.0, γ 1.0 | 0.825 / 0.097 / 960 / 2006 | 0.857 / 0.126 / 914 / 1850 | 0.033 | 1988 / 903 |
| exp | tbA->tbB | θ 0.9992, Tmin 0.5, Tmax 3.0, λ 2.0, γ 4.0 | 0.818 / 0.099 / 1209 / 2022 | 0.829 / 0.051 / 1276 / 2046 | 0.003 | 1412 / 1276 |
| exp | tbB->tbA | θ 0.9994, Tmin 0.5, Tmax 3.0, λ 5.0, γ 4.0 | 0.825 / 0.095 / 874 / 1935 | 0.814 / 0.214 / 758 / 1918 | 0.001 | 1571 / 758 |
| exp | TB halves pooled | (per half) | | 0.820 / 0.127 / 1046 / 2004 | 0.002 | 1571 / 1044 |
| vel | oto->tb | θ 0.999, Tmin 0.5, Tmax 3.0, λ 3.0, γ 1.0, vel (0.85, 0.5, 2 fr) | 0.825 / 0.097 / 960 / 2006 | 0.857 / 0.126 / 914 / 1850 | 0.033 | 1988 / 903 |
| vel | tbA->tbB | θ 0.9992, Tmin 0.5, Tmax 3.0, λ 2.0, γ 4.0, vel (0.85, 0.5, 2 fr) | 0.818 / 0.099 / 1209 / 2022 | 0.829 / 0.051 / 1276 / 2046 | 0.003 | 1412 / 1276 |
| vel | tbB->tbA | θ 0.9994, Tmin 0.5, Tmax 3.0, λ 5.0, γ 4.0, vel (0.85, 0.5, 2 fr) | 0.825 / 0.095 / 874 / 1935 | 0.814 / 0.214 / 758 / 1918 | 0.001 | 1571 / 758 |
| vel | TB halves pooled | (per half) | | 0.820 / 0.127 / 1046 / 2004 | 0.002 | 1571 / 1044 |

**What the grid can reach on TurnBench dev (IN-SAMPLE, not held-out)**, fp ≤ 0.10 Pareto fronts in (recall, P50):

| family | TB-dev in-sample Pareto front at fp <= 0.10 (recall / fp / P50 / P90; point) |
|---|---|
| fixed | 0.851 / 0.098 / 1294 / 1447 (θ 0.9988, k 1.4 s) |
| lin | 0.842 / 0.087 / 1357 / 1540 (θ 0.9985, T0 2.0, a 0.5, Tmin 0.2) |
| lin_cal | 0.827 / 0.091 / 1116 / 1872 (θ 0.9994, T0 2.5, a 2.0, Tmin 0.2)<br>0.828 / 0.091 / 1117 / 1883 (θ 0.9992, T0 2.5, a 2.0, Tmin 0.2)<br>0.855 / 0.092 / 1120 / 1924 (θ 0.9988, T0 2.5, a 2.0, Tmin 0.2) |
| exp | 0.819 / 0.100 / 1026 / 1993 (θ 0.9996, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0)<br>0.820 / 0.100 / 1027 / 1992 (θ 0.9994, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0)<br>0.823 / 0.100 / 1028 / 2017 (θ 0.9992, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0)<br>0.839 / 0.099 / 1040 / 2084 (θ 0.999, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0)<br>0.861 / 0.097 / 1053 / 1944 (θ 0.9988, Tmin 0.5, Tmax 2.5, λ 3.0, γ 2.0)<br>0.861 / 0.100 / 1080 / 1804 (θ 0.9988, Tmin 0.5, Tmax 2.0, λ 2.0, γ 2.0)<br>0.862 / 0.100 / 1216 / 1547 (θ 0.9988, Tmin 0.7, Tmax 2.0, λ 1.0, γ 0.5) |
| vel | 0.819 / 0.100 / 1026 / 1993 (θ 0.9996, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.820 / 0.100 / 1027 / 1992 (θ 0.9994, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.823 / 0.100 / 1028 / 2017 (θ 0.9992, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.839 / 0.099 / 1040 / 2084 (θ 0.999, Tmin 0.2, Tmax 3.0, λ 3.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.861 / 0.097 / 1053 / 1944 (θ 0.9988, Tmin 0.5, Tmax 2.5, λ 3.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.861 / 0.100 / 1080 / 1804 (θ 0.9988, Tmin 0.5, Tmax 2.0, λ 2.0, γ 2.0, vel (0.85, 0.5, 2 fr))<br>0.862 / 0.100 / 1216 / 1547 (θ 0.9988, Tmin 0.7, Tmax 2.0, λ 1.0, γ 0.5, vel (0.85, 0.5, 2 fr)) |

**The oto fronts and where they land on TurnBench** (every oto point with fp ≤ 0.10 that is not dominated, left =
oto fit, right = TurnBench dev held-out):

| family | oto Pareto front at fp <= 0.10: fit (oto) -> held-out TB dev |
|---|---|
| fixed | 0.813 / 0.099 / 1144 / 2076 -> 0.851 / 0.129 / 1115 / 1239 |
| lin | 0.819 / 0.098 / 1200 / 2077 -> 0.867 / 0.123 / 1147 / 1317 |
| lin_cal | 0.818 / 0.096 / 1040 / 2056 -> 0.852 / 0.125 / 970 / 1860<br>0.819 / 0.092 / 1100 / 1855 -> 0.865 / 0.122 / 1012 / 1684 |
| exp | 0.825 / 0.097 / 960 / 2006 -> 0.857 / 0.126 / 914 / 1850<br>0.827 / 0.097 / 968 / 1848 -> 0.857 / 0.126 / 915 / 1749<br>0.834 / 0.099 / 1040 / 1802 -> 0.858 / 0.125 / 950 / 1478 |
| vel | 0.825 / 0.097 / 960 / 2006 -> 0.857 / 0.126 / 914 / 1850<br>0.827 / 0.097 / 968 / 1848 -> 0.857 / 0.126 / 915 / 1749<br>0.834 / 0.099 / 1040 / 1802 -> 0.858 / 0.125 / 950 / 1478 |

**Can this head reach VAP's region?** (P50 targets at fp ≤ 0.10; in-sample on TurnBench dev unless marked):

| family | TB in-sample min P50 at fp <= 0.10 | best recall P50 <= 700, fp <= 0.10 (in-sample / pooled cross-fit) | best recall P50 <= 500 | lowest fp at P50 <= 700, recall >= 0.80 (in-sample) | lowest fp at P50 <= 500, recall >= 0.80 |
|---|---|---|---|---|---|
| fixed | 0.851 / 0.098 / 1294 / 1447 | none / none | none | 0.847 / 0.287 / 624 / 836 | 0.807 / 0.376 / 441 / 808 |
| lin | 0.842 / 0.087 / 1357 / 1540 | none / none | none | 0.864 / 0.266 / 687 / 894 | 0.821 / 0.331 / 476 / 998 |
| lin_cal | 0.827 / 0.091 / 1116 / 1872 | none / none | none | 0.860 / 0.213 / 694 / 1261 | 0.823 / 0.301 / 494 / 1048 |
| exp | 0.819 / 0.100 / 1026 / 1993 | none / none | none | 0.808 / 0.185 / 699 / 1779 | 0.805 / 0.263 / 496 / 1106 |
| vel | 0.819 / 0.100 / 1026 / 1993 | none / none | none | 0.808 / 0.185 / 699 / 1779 | 0.805 / 0.263 / 496 / 1106 |

Velocity sensitivity (`vel_sensitivity` in the JSON): the task's rule (phat > 0.85, rise > 0.5 in 2 frames) never
fires. Under the oto calibration phat tops out at 0.81; under a TurnBench calibration a > 0.5 rise within
160 ms happens on 0.00 % of silent chunks with phat > 0.85 (0.05 % for > 0.3, 0.9 % for > 0.1). Looser rules on
top of each chosen exp point (rise 0.1-0.5, 2-6 frames), fitted the same way, move the fitting-split P50 by at most
−44 ms (rise > 0.1 over 6 frames, at +3.2 fp points) and are selected once (tbB->tbA: 4 frames, −1 ms). The head's
posterior rises slowly, so a velocity cue has nothing to act on at this frame rate. `vel` = `exp` in every table.

**The same points through eot-bench v2** (section 3 windows: oto 581 human turns, TurnBench-derived 419 both-party
ends; the policies run continuously over the whole conversation and each window reads their committed events;
operating points frozen, no v2 re-fit; FC per turn and per v2 pause; miss / P50 at the 6 s horizon):

| set | system (frozen point) | FC % turn / pause | miss 6 s % [CI] | P50 / P90 ms | open miss % (n) | taken miss % |
|---|---|---|---|---|---|---|
| oto (581) | hybrid_dyadic_md (theta 0.99885, k 1.5 s; TurnBench-dev-selected) | 14.6 / 6.9 | 9.7 [7.2, 12.4] | 1440.0 / 2240.0 | 1.4 (177) | 12.9 |
| oto (581) | fixed (oto-fitted) | 21.7 / 12.1 | 4.6 [2.7, 6.6] | 1120.0 / 1200.0 | 0.0 (177) | 6.3 |
| oto (581) | lin (oto-fitted) | 24.8 / 11.8 | 4.6 [2.7, 6.6] | 1200.0 / 1280.0 | 0.0 (177) | 6.3 |
| oto (581) | lin_cal (oto-fitted) | 26.0 / 13.6 | 4.4 [2.5, 6.4] | 1040.0 / 1920.0 | 0.8 (177) | 5.8 |
| oto (581) | exp (oto-fitted) | 26.9 / 15.0 | 3.5 [1.8, 5.2] | 960.0 / 1840.0 | 0.0 (177) | 4.8 |
| oto (581) | vel (oto-fitted) | 26.9 / 15.0 | 3.5 [1.8, 5.2] | 960.0 / 1840.0 | 0.0 (177) | 4.8 |
| turnbench (419) | hybrid_dyadic_md (theta 0.99885, k 1.5 s; TurnBench-dev-selected) | 18.4 / 14.3 | 6.7 [4.1, 9.4] | 1440.0 / 1680.0 | 2.8 (51) | 7.2 |
| turnbench (419) | fixed (oto-fitted) | 19.3 / 25.5 | 6.8 [4.2, 9.4] | 1120.0 / 1280.0 | 0.0 (51) | 7.6 |
| turnbench (419) | lin (oto-fitted) | 28.2 / 20.4 | 4.0 [1.9, 6.4] | 1120.0 / 1360.0 | 0.0 (51) | 4.4 |
| turnbench (419) | lin_cal (oto-fitted) | 30.1 / 24.5 | 2.7 [1.0, 4.8] | 960.0 / 1920.0 | 3.5 (51) | 2.6 |
| turnbench (419) | exp (oto-fitted) | 25.5 / 25.5 | 4.5 [2.2, 7.0] | 960.0 / 2080.0 | 3.3 (51) | 4.6 |
| turnbench (419) | vel (oto-fitted) | 25.5 / 25.5 | 4.5 [2.2, 7.0] | 960.0 / 2080.0 | 3.3 (51) | 4.6 |

**Reading.**
- The policy is worth ~0.35 s at most, not 0.9 s. On TurnBench dev with fp ≤ 0.10 the lowest P50 any policy in the
  grid reaches is 1026 ms (exp, recall 0.819; in-sample) against 1294 ms for the best fixed hybrid (0.851) and
  1390 ms for section 4's point. Held out, only the fixed family stays inside the budget: the two-halves cross-fit gives
  **0.849 / 0.098 / 1316 ms** (a small, honest gain over section 4's in-sample 1390 ms, from a finer k grid); the dynamic
  families cross-fitted land at P50 1046-1228 ms but at fp 0.106-0.127, and their two directions disagree wildly (exp:
  fp 0.051 on B vs 0.214 on A), i.e. the dynamic wait's FP rate is not stable across 19-conversation halves.
- Fitted on oto and moved to TurnBench, every family keeps recall (0.85-0.87, above the fit's 0.81-0.83) and cuts the
  P50 to 914-1147 ms, but all of them overshoot the TurnBench FP budget (fp 0.123-0.129): oto's Silero-derived pauses
  are easier than TurnBench's hand-labelled ones. With the ranking kept (exp < lin_cal < fixed < lin in P50), oto is a
  usable proxy for the SHAPE of the policy, not for its FP calibration; transferring it needs a TurnBench-side FP
  re-fit (one scalar, e.g. Tmin) or TurnBench-style labels on oto.
- The head branch barely matters at these points: 0.1-22 % of detections come from it, and those are the SLOW ones
  (held-out head-branch P50 0.79-1.99 s vs silence-branch 0.76-1.57 s). The trail6 posterior crosses a high θ
  late in the silence; what the calibrated posterior does buy is the shape of the wait (short when phat is high),
  worth ~0.24 s of P50 in-sample (exp 0.861 / 0.097 / 1053 ms vs fixed 0.851 / 0.098 / 1294 ms).
- VAP's region is out of reach for a reactive head + dynamic wait. No point in any family has P50 ≤ 700 ms with fp ≤
  0.10 (in-sample or cross-fitted). Getting P50 ≤ 700 ms with recall ≥ 0.80 costs fp ≥ 0.185 (exp; 0.21-0.29 for the
  others), P50 ≤ 500 ms costs fp ≥ 0.263; VAP is at 463 ms with fp 0.045. The reason is the negatives: TurnBench's
  mid-turn pauses have median 0.42 s and p90 1.21 s, so any rule that waits for silence must wait ~1 s to stay under
  fp 0.10 unless the model separates ends from pauses AT the silence onset, and the calibrated trail6 posterior does
  not separate them well enough (phat spans 0.12-1.0 on TurnBench, yet no configuration with P50 below
  1026 ms stays under fp 0.10).
- What remains is architectural: VAP's p10 is −34 ms, i.e. it predicts the end before the silence, from both channels'
  future voice activity. Our head is trained on AMI with a reactive target and reads a mono mix plus one party track;
  it cannot anticipate. Closing the gap needs a predictive (voice-activity projection) target, both channels as
  input, and dyadic training data with TurnBench-style pause/end labels (section 6's oto training run is the first
  step; the kill rule there should now also include P50 at fp ≤ 0.10).
- eot-bench v2 ranks the frozen points the same way (exp P50 960 ms < fixed 1120 < section 4's hybrid 1440 ms, on
  both sets; misses 3.5-4.6 % vs 6.7-9.7 %) and shows the price explicitly: these TurnBench-budget points cut off
  15-30 % of turns by v2's per-turn count (section 3's operating points are at ≤ 5 %). TurnBench's fp ≤ 0.10 per
  labelled pause is a much looser budget than v2's ≤ 5 % per turn.


## 8. Training on otoSpeech

2026-09-27. This is the first turn head trained on real two-party conversation with a real agent-end signal, plus two
cheap ablations. Code:
- `audioforge/heads/turn.py`: `energy_input`, `multi_horizon_aux`, `energy_features`, `multi_horizon_targets`,
  `decode_aux`.
- `audioforge/model.py`: conditioning `rebind`.
- `audioforge/datasets/dyadic.py`: `n_conv` / `split_mod` / `dev_res` split, `asr_text`.
- `scripts/prepare_dyadic_asr.py` (per-channel ASR transcripts).
- `scripts/bench_dyadic_heads.py` (stages scores / check / points / report / tb_policy / oto_tb_head / predictive).
- `scripts/bench_turnbench.py --tag/--ckpt/--device/--frozen`.

Recipes: `research/recipes/stage1_turn_dyadic{,_energy,_mh,_energy_mh}.yaml`. Tests: `tests/test_turn_dyadic_feats.py`.
Numbers: `runs/dyadic_train.json`, `runs/turnbench_dev_<tag>.json`. Heads: `runs/stage1_turn_dyadic{,_energy,_mh,_energy_mh}.afm`.
The best is `runs/stage1_turn_dyadic_energy.afm` (variant b).

**Data.**
- **oto split.** The first 160 conversations of the downloaded oto slice (23 shards, 204 conversations, ~40 h).
  They are split by conversation position: i % 8 in {0, 1, 2} is dev (60 conversations, 2175 human-party turn ends,
  never trained on). The other 100 are train (~25 h, 7360 turn windows of both parties).
- **oto labels.** Silero labels and float16 mix caches: `prepare_dyadic.py --labels --n 160 --cache-dtype float16
  --part k/2`, about 8 min.
- **oto text.** oto has no transcripts. Each training window's text is therefore the frozen RNNT's greedy decode of
  that party's own channel (`asr_text`, 20 min on MPS). This follows the AMI convention (the primary's words), with
  the ASR hypothesis standing in for the reference.
- **Roles.** User = the window's party track (column 0). Agent = the other party's exact activity (column 1). The
  causal `agent_end` stream of section 2 is implied by column 1: the head sees the agent's 1 → 0 transitions as
  they happen.
- **Mix.** oto weight 1.0, AMI train_trail6 turn windows (streaming Sortformer tracks) weight 0.5 (1637), the
  synthetic four-move conversations weight 0.3 (750).
- **Conditioning.** trail6's (p_ext 0.9 on AMI items, ext_noise, flip 0.01). On top of it, `rebind` p 0.1:
  - The final track binds the wrong party for 1-4 s: the primary column is swapped with the most active other
    column (CONTAMINATION.md section 9).
  - The swap ends between the user's onset and the turn end. The track is correct again after it.
  - The EOT labels stay the clean ones.
- **Training.** Warm start from trail6. Encoder, RNNT, CTC and the other heads are at lr × 0 (heads.turn only,
  0.32 M parameters). 3000 steps of batch 6 at lr 5e-4, checkpoint every 500. Run times: (a) 33 min, (b) 36 min,
  (c) 37 min, (d) 39 min on MPS.
- **Frozen check.** `bench_dyadic_heads.py --stage check` confirms all 747 non-turn tensors are bit-identical to
  trail6's in all four runs (11 RNNT tensors among them). Only heads.turn moved, plus the new `energy_in` / `mh`.

**Variants.**
- (a) `dyadic`: baseline.
- (b) `energy`: adds a causal standardised log-RMS input.
  - One scalar per 80 ms frame of the raw mixed audio, computed on samples [1280 t, 1280 (t+1)).
  - Standardised by an EMA mean and variance: cumulative for the first 125 frames, then a ~10 s window.
  - Clipped to ±4 and halved.
  - Enters through a zero-initialised projection, which amounts to a column concatenated to the head input.
- (c) `mh`: multi-horizon future-activity targets for the user's party only.
  - Disjoint bins (t, t+240], (t+240, t+400], (t+400, t+640], (t+640, t+1040] ms: the +200/+400/+600/+1000 ms
    horizons rounded up to the 80 ms grid.
  - They replace the auxiliary head: `future_act_aux` is kept at weight 0 so the warm-start tensors load.
  - Relation to `future_act_aux`: that head predicts cumulative windows (t, t+h] at 480 / 960 / 2000 ms, for the user
    and for "any other speaker". This one predicts finer, disjoint bins inside the first second, for the user only.
    The OR of the bins is "the user speaks within 1040 ms".
  - The EOT target is unchanged.
- (d) `energy_mh`: (b) + (c).

### 8.1 oto dev (eot-bench v2, 2175 human turn ends, 60 conversations)

Input: the mixed mono plus the human's Silero track, cols [human, agent, 0, 0] (section 3's oracle-track condition).
All heads use the same windows. The operating point is cross-fitted at ≤ 5 % per-turn false cutoffs (FC) over
conversation halves, with 1000 bootstraps.

| system | FC % | miss 6 s % [CI] | miss 2 s % | P50 6 s ms | open miss % | taken miss % |
|---|---|---|---|---|---|---|
| timeout_primary_oracle | 4.3 | 10.0 [8.8, 11.3] | 10.0 | 1680 | 1.5 | 14.7 |
| head trail6 (AMI) | 5.0 | 11.9 [10.6, 13.3] | 38.3 | 1760 | 10.5 | 12.7 |
| head (a) dyadic | 4.8 | 10.5 [9.3, 11.8] | 34.9 | 1680 | 6.6 | 12.7 |
| head (b) energy | 5.0 | **9.2 [8.1, 10.6]** | **30.4** | **1600** | 5.4 | 11.3 |
| head (c) mh | 5.1 | 10.8 [9.6, 12.2] | 35.6 | 1680 | 5.1 | 14.0 |
| head (d) energy_mh | 5.1 | 9.7 [8.5, 11.0] | 30.2 | 1600 | 4.6 | 12.5 |
| hybrid trail6 (head OR primary timeout) | 5.1 | 8.5 [7.4, 9.7] | 8.5 | 1600 | 1.1 | 12.6 |
| hybrid (a) | 5.1 | 8.5 [7.4, 9.8] | 8.5 | 1600 | 1.0 | 12.7 |
| hybrid (b) | 3.9 | **7.5 [6.5, 8.7]** | 7.5 | 1680 | 0.9 | 11.1 |
| hybrid (c) | 5.1 | 8.6 [7.5, 9.8] | 8.6 | 1600 | 1.0 | 12.7 |
| hybrid (d) | 5.1 | 8.3 [7.2, 9.6] | 8.3 | 1600 | 1.1 | 12.2 |

Paired against trail6 on the same ends. Miss points at 6 s with 95 % CI; P50 is the difference in ms.

| a − b | all | open | taken | P50 |
|---|---|---|---|---|
| head (a) − head trail6 | −1.4 [−2.6, −0.2] | −3.9 [−5.7, −2.1] | +0.0 | −80 |
| head (b) − head trail6 | **−2.7 [−3.8, −1.4]** | −5.0 [−6.9, −3.3] | −1.3 | −160 |
| head (c) − head trail6 | −1.1 [−2.3, +0.2] | −5.3 [−7.2, −3.5] | +1.4 | −80 |
| head (d) − head trail6 | −2.2 [−3.4, −1.0] | −5.8 [−8.0, −4.0] | −0.2 | −160 |
| hybrid (a) − hybrid trail6 | −0.0 [−0.6, +0.6] | −0.1 [−0.7, +0.3] | +0.1 | +0 |
| hybrid (b) − hybrid trail6 | **−1.0 [−1.7, −0.4]** | −0.2 [−0.7, +0.3] | −1.5 | +80 |
| hybrid (c) − hybrid trail6 | +0.0 [−0.5, +0.6] | −0.1 [−0.7, +0.3] | +0.1 | +0 |
| hybrid (d) − hybrid trail6 | −0.2 [−0.8, +0.3] | +0.0 [−0.4, +0.4] | −0.4 | +0 |

Frozen AMI points (no refit). Each head's own point is fitted on all 974 AMI dev turns with the oracle party track
at ≤ 5 % FC, then applied unchanged. The paired column is against trail6 at trail6's own frozen point.

| system | AMI θ (, k) | FC % | miss 6 s % [CI] | miss 2 s % | P50 6 s ms | open / taken % | paired vs trail6, 6 s |
|---|---|---|---|---|---|---|---|
| head trail6 | 0.97915 | 15.4 | 4.5 [3.6, 5.4] | 15.3 | 1200 | 3.7 / 4.9 | ref |
| head (a) | 0.99158 | 6.3 | 7.8 [6.7, 8.9] | 24.7 | 1440 | 5.0 / 9.3 | +3.2 [+2.3, +4.3] |
| head (b) | 0.99087 | 7.1 | 6.8 [5.8, 7.9] | 20.7 | 1360 | 4.0 / 8.4 | +2.3 [+1.3, +3.4] |
| head (c) | 0.99333 | 6.4 | 8.8 [7.6, 10.0] | 28.1 | 1520 | 4.4 / 11.1 | +4.2 [+3.2, +5.3] |
| head (d) | 0.99143 | 7.4 | 6.8 [5.8, 7.9] | 19.5 | 1360 | 3.6 / 8.5 | +2.3 [+1.3, +3.3] |
| hybrid trail6 | 0.97982, 23 | 15.3 | 3.1 [2.4, 3.9] | 3.1 | 1280 | 0.5 / 4.4 | ref |
| hybrid (a) | 0.9918, 22 | 7.0 | 5.0 [4.1, 5.9] | 5.0 | 1520 | 0.8 / 7.3 | +1.9 [+1.2, +2.6] |
| hybrid (b) | 0.99212, 21 | 7.6 | 4.8 [3.9, 5.7] | 4.8 | 1440 | 0.7 / 7.0 | +1.7 [+0.9, +2.5] |
| hybrid (c) | 0.99881, 18 | 7.0 | 8.1 [7.0, 9.3] | 8.1 | 1520 | 0.9 / 12.0 | +5.0 [+4.0, +6.0] |
| hybrid (d) | 0.99136, 24 | 7.5 | 4.8 [4.0, 5.8] | 4.8 | 1360 | 0.7 / 7.1 | +1.7 [+1.0, +2.5] |

How to read the frozen rows: the frozen comparison is **not at matched FC**.
- trail6's AMI θ produces 15 % false cutoffs on oto. This is section 3's finding: VAD-gap "words" produce
  within-turn gaps.
- Every dyadic head's own AMI θ lands at 6-7.6 %, i.e. its calibration transfers across corpora. The extra misses
  are the price of cutting off half as many users.
- The cross-fitted table is the matched-FC comparison.

### 8.2 TurnBench dev (official scorer, section 4's inputs and rules)

Section 4 protocol: θ over the head's score quantiles (their grid), head OR Silero-per-channel at the joint grid of
section 4. "Dev point" = the maximum recall at fp ≤ 0.10. "Frozen" = the head's AMI points: θ for the head, and
(θ, k_frames × 80 ms) of head + Silero timeout.

| head | head alone, dev point: recall / fp / P50 | head OR Silero, dev point: recall / fp / P50 | head alone, frozen: recall / fp / P50 | head OR Silero, frozen: recall / fp / P50 |
|---|---|---|---|---|
| trail6 | **0.752** / 0.099 / **1058** | 0.835 / 0.080 / 1390 | 0.397 / 0.241 / 300 | 0.465 / 0.256 / 456 |
| (a) dyadic | 0.676 / 0.063 / 1302 | 0.831 / 0.074 / 1399 | 0.694 / 0.257 / 538 | 0.807 / 0.251 / 678 |
| (b) energy | 0.705 / 0.061 / 1316 | 0.834 / 0.074 / 1399 | 0.700 / 0.268 / 543 | 0.802 / 0.257 / 654 |
| (c) mh | 0.720 / 0.064 / 1337 | 0.835 / 0.075 / 1396 | 0.718 / 0.222 / 618 | 0.763 / 0.224 / 667 |
| (d) energy_mh | 0.713 / 0.056 / 1392 | 0.824 / 0.074 / 1396 | 0.725 / 0.207 / 597 | 0.737 / 0.203 / 602 |
| references (section 4) | Silero per channel 0.815 / 0.075 / 1399; VAP 0.841 / 0.045 / 463; smart-turn v3 0.754 / 0.100 / 1010 | | | |

Finer grid (section 7's replica with TP-to-branch attribution; `--stage tb_policy`): θ at 40 score quantiles × k 0.5-3 s.

| head | head alone, max recall at fp ≤ 0.10 | head OR Silero, max recall at fp ≤ 0.10 (head share, P50 head / silence) | lowest P50 with recall ≥ 0.80, fp ≤ 0.10 | P50 ≤ 700 ms reachable? |
|---|---|---|---|---|
| trail6 | 0.752 / 0.099 / 1058 | 0.854 / 0.099 / 1274 (0.16; 938 / 1303) | 1151 (recall 0.828) | no |
| (a) | 0.677 / 0.063 / 1302 | 0.850 / 0.090 / 1303 (0.06; 1146 / 1306) | 1263 | no |
| (b) | 0.705 / 0.061 / 1316 | 0.853 / 0.089 / 1303 (0.06; 1154 / 1306) | 1265 | no |
| (c) | 0.720 / 0.064 / 1336 | 0.851 / 0.090 / 1287 (0.12; 994 / 1303) | 1278 | no |
| (d) | 0.713 / 0.056 / 1392 | 0.852 / 0.092 / 1297 (0.08; 1037 / 1306) | 1268 | no |

On TurnBench the dyadic heads alone sit 3-8 recall points below trail6 at the dev point, but at fp 0.056-0.064
instead of 0.099. Their score curves are even steeper at the top than trail6's: the next quantile overshoots fp
0.10, so the budget goes unused. Head OR Silero is level with trail6 everywhere (0.824-0.835 on the coarse grid,
0.850-0.853 on the fine grid) because the Silero branch carries it: head share 6-12 %.

At the AMI-frozen points the dyadic heads keep 0.69-0.73 recall at P50 540-620 ms, against trail6's 0.40. The
price is fp 0.21-0.27, over the budget; trail6's frozen θ also fails the budget, at 0.24. As on oto, the dyadic
heads' AMI calibration transfers much better.

**The predictive trigger** (coordinator's addition; `--stage predictive`). CPU only, on the saved outputs of (c) and
(d); no retraining.
- Rule: fire at the rising edge of P(user active in bin 1, ≈ +200 ms) < t1 AND P(bin 2, ≈ +400 ms) < t2. It is
  committed at the 160 ms chunk end with the scorer's 2 s refractory and uses no silence timer. A second version ORs
  it with the Silero-per-channel timeout k.
- Fit: (t1, t2 [, k]) at the maximum recall at fp ≤ 0.10, fitted (i) on oto and (ii) on one TurnBench half.
  - (i) The oto fit uses the 6 dev conversations among section 7's 16 (1.6 h, section 7's scorer-built gold) and is
    evaluated on all of TurnBench dev.
  - (ii) The TurnBench fit is evaluated on the other half; the table pools the two held-out halves.
- The reactive heads are fitted the same way.
- Columns: TP before end = share of matched fires that land before the gold end, i.e. before the user's last word
  ends. In-speech = share of all fires landing while the user's own channel is in a Silero speech chunk (mean of the
  two held-out halves).

| system | fit | held-out recall / fp / P10 / P50 ms | TP before end | in-speech | branch share |
|---|---|---|---|---|---|
| trail6 reactive head | TB halves | 0.733 / 0.087 / 270 / 1138 | 0.023 | 0.012 | – |
| trail6 head OR Silero | TB halves | 0.836 / 0.118 / 358 / 1279 | 0.016 | 0.008 | head 0.42 |
| (c) reactive head | TB halves | 0.720 / 0.064 / 406 / 1336 | 0.022 | 0.014 | – |
| (c) head OR Silero | TB halves | 0.845 / 0.088 / 673 / 1330 | 0.017 | 0.001 | head 0.10 |
| **(c) predictive** | TB halves | **0.804 / 0.091 / 378 / 961** | 0.021 | 0.006 | – |
| **(c) predictive OR Silero** | TB halves | **0.862 / 0.092 / 400 / 1137** | 0.021 | 0.003 | predictive 0.70 |
| (c) predictive | oto → TB | 0.801 / 0.147 / 230 / 755 | 0.032 | 0.014 | – |
| (c) predictive OR Silero | oto → TB | 0.866 / 0.175 / 238 / 800 | 0.032 | 0.012 | predictive 0.80 |
| (d) predictive | TB halves | 0.790 / 0.088 / 379 / 980 | 0.019 | 0.005 | – |
| (d) predictive OR Silero | TB halves | 0.859 / 0.105 / 365 / 1103 | 0.023 | 0.004 | predictive 0.68 |
| (d) predictive | oto → TB | 0.801 / 0.150 / 207 / 752 | 0.039 | 0.015 | – |

Fitted points:
- (c): (t1, t2) = (0.009, 0.009) and (0.010, 0.010) on the halves, (0.012, 0.012) on oto. With Silero, k = 1.5 s
  and 3.0 s.
- The bins' probabilities are small (TurnBench medians 0.017 / 0.027), so the useful thresholds sit at the bottom of
  their range.

What the trigger shows:
- **It is the first head-driven policy that beats the reactive systems at matched fp.**
  - (c) predictive alone: 0.804 / 0.091 / P50 961 ms, against its own reactive head at 0.720 / 0.064 / 1336 and
    trail6's reactive head at 0.733 / 0.087 / 1138.
  - OR-ed with the Silero timeout: 0.862 / 0.092 / 1137 ms, against section 4's hybrid at 0.835 / 0.080 / 1390 and
    the per-channel Silero timeout at 0.815 / 0.075 / 1399.
  - The model branch takes 70 % of the detections here, against 6-16 % for the reactive hybrids.
- **It does not reach P50 ≤ 700 ms at fp ≤ 0.10.**
  - The oto-fitted point gets P50 755-800 ms, but at fp 0.15-0.18 on TurnBench (oto's gold has 2.5 pauses per EOT,
    so oto-fitted thresholds are too loose for TurnBench).
  - P10 never goes negative: 207-673 ms. The trigger fires after the silence starts, like the reactive head, just
    earlier in the silence.
  - Only 2-4 % of its TPs land before the gold end, and 0.3-1.5 % of its fires fall inside the user's own speech.
- So the multi-horizon bins do separate "about to stop for good" from "pausing" better than the EOT logit does,
  which section 7 found the reactive head cannot. They still do not supply VAP's pre-silence projection.

### 8.3 AMI dev: regression check (deployable cascade, 974 turns)

Setup: cached 1.04 s streaming Sortformer tracks with causal_dominant binding, cross-fitted ≤ 5 % FC,
leave-meetings-out. trail6's rows reproduce research/archive/EOT_BENCH_V2.md section 7 exactly (61.9 / 45.9).

| system | held-out FC % | miss 6 s % [CI] | 2 s (A) % | open % | taken % | paired vs trail6 (6 s, all) | open |
|---|---|---|---|---|---|---|---|
| timeout (causal_dominant) | 5.7 | 74.8 [72.0, 77.5] | 92.5 | 46.1 | 83.6 | | |
| head trail6 | 5.3 | 66.5 [63.4, 69.7] | 79.8 | 57.3 | 69.4 | ref | |
| head (a) | 7.0 | 69.2 [66.0, 72.1] | 85.0 | 57.1 | 73.0 | +2.7 [−0.2, +5.6] | −0.1 |
| head (b) | 7.8 | 68.4 [65.1, 71.4] | 84.4 | 55.6 | 72.4 | +1.9 [−1.1, +4.7] | −1.7 |
| head (c) | 5.4 | 70.4 [67.2, 73.2] | 85.3 | 55.7 | 74.9 | +3.9 [+0.8, +6.9] | −1.6 |
| head (d) | 6.8 | 69.4 [66.1, 72.4] | 84.0 | 57.1 | 73.2 | +2.9 [−0.2, +5.9] | −0.1 |
| **hybrid trail6** | 6.5 | **61.9 [58.7, 65.0]** | 79.1 | **45.9** | 67.0 | ref | |
| hybrid (a) | 8.2 | 64.2 [60.8, 67.1] | 84.5 | 41.6 | 71.3 | +2.3 [−0.5, +4.8] | −4.3 [−10.2, +1.0] |
| hybrid (b) | 8.9 | 63.0 [59.4, 66.1] | 84.1 | 41.5 | 69.8 | +1.1 [−1.6, +3.8] | −4.4 [−10.1, +1.0] |
| hybrid (c) | 6.9 | 65.0 [61.8, 68.1] | 85.2 | 40.2 | 72.7 | +3.1 [+0.4, +5.6] | −5.7 [−11.1, −1.1] |
| hybrid (d) | 6.9 | 65.5 [62.4, 68.4] | 83.1 | 41.5 | 73.0 | +3.6 [+0.9, +6.0] | −4.4 [−9.6, +0.6] |

AMI got slightly worse:
- **Taken ends: +3-6 points.** The CI excludes 0 for the (c) and (d) hybrids and the (c) head.
- **Held-out FC** is 0.4-2.4 points higher.
- **Block A (2 s windows): 5 points worse throughout.**
- **Floor-open ends are 4-6 points better**, which is what two-party training should buy.
- (b) has the smallest hybrid difference: +1.1 [−1.6, +3.8]. (a)'s CI also includes 0.

The regression is plausibly the AMI share of the mix (17 % of the items) plus 3000 steps at lr 5e-4 on oto: the
head moved toward oto's VAD-segmented turns.

### 8.4 Verdict

- **Does two-party training with a real agent-end signal beat the AMI head?**
  - **On oto: yes for the head, marginally for the system.** Cross-fitted at matched FC, the head's misses drop by
    1.4 (a) to 2.7 (b) points, with CIs excluding 0, mostly on floor-open ends (−4 to −6). The head OR timeout
    hybrid only improves with the energy input: (b) −1.0 [−1.7, −0.4], at 3.9 % instead of 5.1 % held-out FC.
    Without the energy input the hybrid is unchanged, because the primary timeout already covers the open ends.
  - The dyadic heads' own AMI thresholds also transfer to oto at 6-7.6 % FC instead of trail6's 15 %.
  - **On TurnBench: no.** At the official dev point the head alone is below trail6: 0.68-0.72 vs 0.752 recall. This
    is partly grid granularity; they stop at fp 0.06. Head OR Silero is unchanged: 0.824-0.835 vs 0.835, and
    0.850-0.853 vs 0.854 on the fine grid. Section 6's kill rule (beat the per-channel Silero 0.815) is met only by
    the hybrids, as before.
- **Do the energy input and the multi-horizon targets help latency at matched FP?**
  - **Energy (b):** a small win on oto only. The head's P50 is −160 ms and misses −2.7. The 2 s-horizon miss drops
    from 38.3 to 30.4 %.
  - Energy moved nothing on TurnBench (hybrid 0.834 vs 0.831 without it; the head's P50 at the dev point 1316 vs
    1302 ms) and nothing on AMI beyond noise.
  - So a pitch + pitch-slope input (variant e) has weak support: frame energy helped only in-domain, where the VAD
    labels come from the same channel energy. **Not recommended as the next GPU run** unless it is evaluated on
    TurnBench first.
  - **Multi-horizon targets (c)** did not help the EOT head: +0.0 hybrid on oto, worse on AMI. The EOT head's
    TurnBench P50 at fp ≤ 0.10 is unchanged (1287 vs 1274 ms; head share 12 % vs 16 %).
  - **But the bins themselves are a better decision signal.** The predictive trigger reaches 0.804 / 0.091 / 961 ms
    alone and 0.862 / 0.092 / 1137 ms OR Silero on held-out TurnBench halves. That is higher recall and about
    250 ms faster than any reactive policy at matched fp, with the model branch at 70 % of detections.
  - It still does not reach P50 ≤ 700 ms at fp ≤ 0.10: P10 stays positive, so there is no pre-silence firing.
    Section 7's conclusion therefore holds for the EOT logit, and is softened for the multi-horizon bins.
- **AMI:** a small regression on the deployable cascade (hybrid +1.1 to +3.6 miss points; (c) and (d) CIs exclude
  0), with better floor-open ends (−4 to −6).
- **Best head:** (b) `runs/stage1_turn_dyadic_energy.afm`. It is best on oto (head and hybrid), has the smallest AMI
  regression (CI includes 0) and matches trail6 on TurnBench. For latency work, (c)'s multi-horizon bins with the
  predictive rule are the lead to follow; `decode_aux` exposes them.
- **Not shipped:** `runs/stage1_served.afm` is unchanged. The serve path (`TurnHead.step`) supports the energy input
  through `audio=`, but serve.py does not pass it; that belongs to the ship agent.

**Caveats.**
- **The oto labels are Silero VAD segments** on the same channel the energy is read from (the mix sums both
  channels, which blurs this but does not remove it). The oto "text" is ASR hypotheses.
- **Greedy forced alignment** places the hypothesis tokens for only 4 % of training items. The rest fall back to
  uniform spreading over the party's speech frames. This is the pre-existing convention: it is anticausal inside
  the turn, since it knows where speech ends, and is a train-inference mismatch for the text branch shared with
  trail6.
- **Splits are by conversation, not by speaker.** The profile ids are not read (licence), and oto participants
  recur across sessions.
- **One seed per variant.** The oto differences between variants (b) and (d) are within CIs.
- **TurnBench mix.** On the full disk the TurnBench mix is computed in memory from the float16 channel cache,
  bit-identical to section 4's mix path.
- **The TurnBench predictive fits select on TurnBench halves** (held out across halves, but the same corpus). The
  oto → TB transfer is the cleaner out-of-domain number, and it misses the fp budget.
