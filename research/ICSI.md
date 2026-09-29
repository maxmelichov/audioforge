# ICSI Meeting Corpus: more real conversation for speaker-aware turn-taking

AMI gives us 12 training meetings (7.5 h, 45 speakers). ICSI adds 75 meetings (71.7 h, 60 speakers) under the
same licence, with per-word times, and the Edinburgh site distributes a headset mix. It is feasible, and the
first subset is prepared. Code: `audioforge/datasets/icsi.py`, `scripts/prepare_icsi.py`, `tests/test_icsi.py`.
The labels are *identical by construction*: `icsi.ICSI` subclasses `ami.AMI`. So the AMI turn / hesitation /
backchannel rules, window builders and `corpus_stats` run unchanged on converted ICSI annotations. The
definitions are in research/AMI.md.

## Feasibility (checked 2026-09-26)

| item | finding |
|---|---|
| licence | **CC BY 4.0**. https://groups.inf.ed.ac.uk/ami/icsi/license.shtml says: "The ICSI corpus and its annotations are released under the Creative Commons Attribution 4.0 license". The same text is `LICENCE.txt` in the annotation zip (copied to `data/icsi/annotations/LICENCE.txt`). The site says "all of the signals and transcription, and some of the annotations" are CC BY 4.0. The core zip we use (words, segments, dialogue acts) carries the CC BY 4.0 LICENCE. Attribution: A. Janin et al., "The ICSI Meeting Corpus", ICASSP 2003. |
| audio, headset mix | `https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/<m>.interaction.wav`: one **already-mixed** WAV per meeting, 16 kHz mono PCM16. The download page calls it "Headset mix", average 120 MB. All 75 files total 8.26 GB (HTTP HEAD, pinned in `icsi.WAV_BYTES`). **No channel mixing was needed.** |
| audio, individual channels | `ICSIsignals/SPH/<m>/chan{0..9,A..F}.sph`, about 350 MB per meeting: the close-talk channels plus distant mics. In the typical assignment the desktop PZMs D1–D4 are chanE, chanF, chan6, chan7 and the PDA is chanC/D (lhotse recipe notes). Not used. |
| annotations | `https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip`: v1.0, 22-Jul-2016, 19.5 MB, md5 `e5e42cf5…65b7`. It holds NXT `Words/<m>.<agent>.words.xml` with **per-word start/end times for all 75 meetings**, `Segments/*.segs.xml` (transcriber segments, speaker tag), dialogue acts and speakers.xml. `ICSI_plus_NXT.zip` (55 MB) adds topics, summaries and hot-spots; not needed. `ICSI_original_transcripts.zip` (3.7 MB) is the MRT format, with segment-level times only and no word times. |
| word-time coverage | 797,295 word tokens. **95.3 %** are timed after the conversion below. The rest are "untimed zones", 6052 stretches (3.7 h). About two thirds of the untimed tokens are the read digit strings; the others are unaligned stretches and "@@" (unintelligible speech). A further 0.7 h lies after the last transcriber segment (Bed003 has 15 min). Segment times: `timing-provenance` = dialogueact for 60 %, segment for 40 %. |
| alignment check | Correlation between the any-speaker word-activity mask and 80 ms frame energy of the mix peaks at **lag 0** (Bmr021 / Bro021 / Bsr001: 0.44 / 0.52 / 0.38). It falls to 55–65 % of the peak at ±0.24 s and to about a third at ±0.5 s. The word times are frame-accurate for this audio. Frame RMS separates labelled speech from silence with AUC 0.79–0.81 (AMI dev, same measure: 0.63). |
| standard split | There is **no official partition** (ICSI was training material for NIST RT). We use the Kaldi `egs/icsi` / lhotse `research/recipes/icsi.py` split (Renals & Swietojanski, "Neural networks for distant speech recognition", HSCMA 2014): **dev = Bmr021 Bns001 (2.3 h), eval = Bmr013 Bmr018 Bro021 (2.8 h), train = the other 70 (65.5 h)**. It is **not speaker-disjoint**. All 12 dev and all 13 eval speakers also occur in our 12-meeting train subset. me013 is in 49 of 75 meetings, me018 in 45, me011 in 39. |

## What was downloaded

| item | size |
|---|---|
| `ICSI_core_NXT.zip` → `data/icsi/raw/ICSI/{Words,Segments,speakers.xml,LICENCE.txt,...}` | 19 MB zip, 117 MB extracted |
| converted annotations, all 75 meetings (`data/icsi/annotations/`) | 58 MB |
| headset mix, 17 meetings (`data/icsi/audio/<m>.Mix-Headset.wav`, size + header + pinned md5 `icsi.WAV_MD5`) | 1.76 GB, **~170 s** at 4 connections |
| float32 `.npy` cache (`data/icsi/cache/`) | 3.3 GB |
| **total `data/icsi/`** | **5.3 GB** |

| split | meetings | hours (timed) | speakers (per meeting) |
|---|---|---|---|
| train (default 12) | Bdb001 Bed010 Bed011 Bmr009 Bmr014 Bmr022 Bmr026 Bro010 Bro016 Bro026 Bns003 Bsr001 | 10.27 (9.72) | 35 (5–8) |
| dev | Bmr021 Bns001 | 2.28 (2.15) | 12 (6–7) |
| eval | Bmr013 Bmr018 Bro021 | 2.77 (2.65) | 13 (7) |

The default train meetings are 45–58 min long and come from every series: Bdb, Bed (EDU), Bmr (Meeting Recorder),
Bro (Robustness), Bns (Network services), Bsr. Each has word annotation up to the end of its recording.
`--n-train N` appends the remaining train meetings in list order. Cost per 10 extra meetings (about 9.5 h):
about 1.1 GB wav + 2.2 GB cache. The full corpus is 8.3 GB wav + 16.5 GB cache, which is over this task's 15 GB cap.

## Conversion (ICSI NXT → AMI-format words; `icsi.convert_words`)

- **Pieces → tokens.** ICSI splits contractions, possessives and hyphenated words ("you" + "'re", "mm" - "hmm").
  Pieces joined by a HYPH element, or starting with an apostrophe, are merged into one token, the way AMI writes
  them ("you're", "mm-hmm" → `mm hmm`). The backchannel rule (≤ 2 words) therefore counts the same way. Most of
  the 5 % "half-timed" pieces are exactly these, with times on the outer piece edges.
- **Chains.** A token with only a start, followed by tokens without times and closed by a token with only an end
  (for example "two hour long" [698.64, 699.35]), has its span split over its tokens in proportion to character
  count. This affects 628 tokens in the whole corpus.
- **Untimed zones.** For an untimed token, the zone is its transcriber segment, narrowed to the timed neighbours
  inside that segment. Everything after the last segment is also a zone (segments > 120 s are ignored as
  bogus, e.g. Bro025). Zones go to `annotations/untimed.json`. Any example that overlaps a zone is dropped
  (`exclude_untimed`, default true), so no window contains unlabelled speech. The statistics use words only,
  as for AMI.
- Dropped as in ami.py: punctuation (CM . QM EXCLM HYPH quotes, SYM "-"), vocal/non-vocal sounds, pauses,
  comments, disfluency markers. Text goes through `ami.normalize`.
- **Speaker ids** = `icsi.SPEAKER_OFFSET` (1000) + index into the 60 sorted ICSI speaker tags (fe004 … xe902).
  So ICSI uses 1000–1059 and AMI uses 0–189, and ids are stable for any subset. A speaker head over both
  corpora needs 1060 classes, or `drop_keys: [speaker]` in a mix.
- **More than 4 speakers.** ICSI has 3–10 speakers per meeting, but the targets have 4 columns. A window with
  more than 4 active speakers is dropped by default (`drop_overfull`): 8 % of diar and 10 % of turn windows.
  `drop_overfull: false` keeps them, with the 5th and later speakers missing from `spk_targets`.

| examples (default filters) | diar (20 s) | turn (mean length) | asr (hours) | dropped: untimed / >4 spk |
|---|---|---|---|---|
| train | 2868 | 3082 (10.2 s) | 3249 (3.33 h) | diar 506 / 304, turn 308 / 360 |
| dev | 676 | 477 (10.4 s) | 975 (1.13 h) | diar 113 / 28, turn 53 / 24 |
| eval | 795 | 871 (10.3 s) | 868 (0.92 h) | diar 138 / 58, turn 74 / 60 |
| *AMI train (for scale)* | *2693* | *3274 (10.5 s)* | *930 (1.81 h)* | |

In 52–53 % of ICSI turn examples the primary hesitates at least once (AMI: 15–17 %). In 11–17 % the turn onset
lies before the window.

## Turn-taking statistics (`floor` definition, the same code as AMI; `data/icsi/stats.json`, `stats_all75.json`)

| | ICSI train-12 | ICSI dev | ICSI eval | ICSI train-70 | AMI train | AMI eval |
|---|---|---|---|---|---|---|
| hours | 10.27 | 2.28 | 2.77 | 65.5 | 7.53 | 2.43 |
| speech / time | 0.69 | 0.70 | 0.69 | 0.69 | 0.84 | 0.85 |
| **overlap / speech** | 8.0 % | 4.9 % | 7.8 % | 7.4 % | 12.5 % | 13.2 % |
| turns / min (per timed min) | 7.1 (7.5) | 4.9 (5.2) | 7.0 (7.3) | 6.8 (7.1) | 7.4 | 6.6 |
| backchannels / min (per timed min) | 5.9 (6.2) | 3.1 (3.2) | 5.3 (5.5) | 5.8 (6.1) | 4.9 | 5.3 |
| turn length p50 / p90 (s) | 3.1 / 18.0 | 3.1 / 28.3 | 3.3 / 18.1 | 3.1 / 19.6 | 3.4 / 16.8 | 3.8 / 19.0 |
| **switch gap** p10 / p50 / p90 (s) | −0.91 / 0.16 / 2.01 | −0.97 / 0.20 / 1.96 | −0.96 / 0.22 / 1.90 | −0.90 / 0.20 / 2.15 | −1.54 / 0.12 / 3.64 | −1.73 / 0.08 / 4.17 |
| switches that overlap | 42.2 % | 39.2 % | 41.2 % | 40.4 % | 45.5 % | 47.1 % |
| switch gap when positive p25 / **p50** / p75 (s) | 0.29 / **0.60** / 1.42 | 0.29 / **0.60** / 1.31 | 0.30 / **0.64** / 1.35 | 0.30 / **0.64** / 1.43 | 0.38 / **0.86** / 2.42 | 0.39 / **1.03** / 3.00 |
| **hesitations** per turn | 2.02 | 3.36 | 2.01 | 2.17 | 0.23 | 0.29 |
| hesitation p10 / **p50** / p90 (s) | 0.35 / **0.59** / 1.17 | 0.35 / **0.56** / 1.08 | 0.34 / **0.56** / 1.14 | 0.34 / **0.57** / 1.20 | 0.77 / **1.29** / 1.77 | 0.93 / **1.32** / 1.83 |
| hesitations longer than the median positive switch gap | 48 % | 44 % | 41 % | 42 % | 87 % | 83 % |
| hold gap p50 (s) | 2.49 | 2.70 | 2.77 | 2.64 | 3.08 | 2.80 |

| silence timeout τ | 0.3 s | 0.5 s | 0.7 s | 1.0 s | 1.5 s |
|---|---|---|---|---|---|
| turns containing a pause ≥ τ → false cutoff (ICSI train / eval) | 51.1 / 52.4 % | 39.4 / 39.5 % | 31.0 / 30.0 % | 19.0 / 17.4 % | 6.4 / 5.9 % |
| (AMI train / eval, from AMI.md) | 14.7 / 15.8 % | 14.4 / 15.8 % | 14.1 / 15.6 % | 13.5 / 14.9 % | 5.8 / 6.6 % |
| switches where the next speaker started < τ after the end (ICSI train / eval) | 57 / 56 % | 67 / 66 % | 74 / 73 % | 81 / 81 % | 86 / 86 % |

**What ICSI adds.** ICSI word times keep the short pauses that AMI hides. Of the gaps between consecutive timed
words of one speaker, 77 % are exactly 0 in ICSI and 91 % in AMI. Of the non-zero gaps, 48 % are shorter than
0.5 s in ICSI, against 0.9 % in AMI (AMI caveat 1). As a result ICSI has about 9× more hesitation labels per turn,
and their median (0.56–0.59 s) is realistic. The main AMI finding holds, and ICSI makes it sharper. The median
positive switch gap (0.60–0.64 s) is about the same as the median hesitation, and 40–42 % of switches overlap.
So a 0.5 s timeout would cut 39 % of ICSI turns mid-turn, and would still be too late for 66–67 % of floor
transfers. ICSI is also less overlapped (7–8 % of speech, AMI 12.5 %) and less dense (69 % speech). With 5–8
speakers, a primary's turn has more listeners, which fits a speaker-aware head better.

**Caveats.**

- (1) The splits share speakers. Train on ICSI, and select models on AMI's speaker-disjoint dev/eval. Report ICSI
  dev/eval separately, as a seen-speaker test.
- (2) Untimed zones are treated as silence inside the statistics: 5.4 % of train time, mostly the digit
  readings. The examples avoid them.
- (3) 8 % of inter-word gaps fall in [0.1, 0.5) s, which is close to forced-alignment resolution. Hesitations
  just above 0.3 s are the least certain labels.
- (4) The dev split is small (2 meetings), and Bns001 is 100 min with long monologues (turn p90 28 s).

## Commands

```bash
.venv/bin/python scripts/prepare_icsi.py                        # annotations + 17 wavs + .npy cache + stats (~3 min cold)
.venv/bin/python scripts/prepare_icsi.py --no-cache --max-minutes 4   # download only, resumable
.venv/bin/python scripts/prepare_icsi.py --n-train 24            # 12 more train meetings (~+3.3 GB with cache)
.venv/bin/python scripts/prepare_icsi.py --stats-only --all --json data/icsi/stats_all75.json   # all 75, no audio
.venv/bin/python -m pytest -q tests/test_icsi.py
```

Recipe: `data: {icsi: {mode: turn|diar|asr, n_meetings: {train: 12, val: 2}, window_sec: 20, exclude_untimed: true,
drop_overfull: true, ...}}` (`audioforge.train.load_data` routes it to `icsi.recipe_data`; val uses dev). To combine
with AMI: `data: {mix: [{ami: {mode: turn}}, {icsi: {mode: turn}}], val: {ami: {mode: turn}}}`.
