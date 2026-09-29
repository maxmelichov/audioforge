# AMI Meeting Corpus: conversational data for diarization and speaker-aware end-of-turn

PLAN.md R3: LibriSpeech is only a regression test. Turn-taking and diarization are trained and
evaluated on real conversation. This note covers the first AMI subset. Code:
`audioforge/datasets/ami.py`, `scripts/prepare_ami.py`, `tests/test_ami.py`.

## What was downloaded (2026-09-25)

| item | source | size |
|---|---|---|
| Manual annotations v1.6.2 (words, segments, corpusResources) | https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip (md5 `b9db0918…d475`) | 22 MB zip |
| Headset-mix audio, 20 meetings (`<m>.Mix-Headset.wav`, 16 kHz mono PCM16) | https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<m>/audio/ | 1.4 GB |
| Split lists (Full-corpus-ASR partition, BUT/pyannote AMI diarization setup) | https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/lists/{train,dev,test}.meetings.txt (fork of https://github.com/BUTSpeechFIT/AMI-diarization-setup) | 136/18/16 meetings |

The audio download took **109 s** (4 parallel connections). The float32 `.npy` cache adds 2.6 GB,
so `data/ami/` holds 4.1 GB in total. The AMI site publishes no checksums, so every wav is checked for
size, header (16 kHz, mono) and md5 against values pinned from this first download
(`ami.WAV_MD5`, also stored in `data/ami/audio/md5.json`).

Default subset: mid-series meetings, one per scenario series or site. It skips the short kick-off "a"
sessions and has 4 speakers everywhere. `--n-train/--n-dev/--n-eval` extends each split in official
list order.

| split (list) | meetings | hours | speakers | speech / time | overlap / speech |
|---|---|---|---|---|---|
| train (`train`) | IS1000b IS1003c IS1006b ES2003b ES2007c ES2012b ES2015c TS3006b TS3009c TS3011b IN1002 EN2003a | 7.53 | 45 | 0.84 | 12.5 % |
| dev (`dev`) | IS1008b ES2011b TS3004b IB4002 | 2.08 | 15 | 0.77 | 13.5 % |
| eval (`test`) | IS1009b ES2004b TS3003b EN2002a | 2.43 | 16 | 0.85 | 13.2 % |

Speaker ids are indices into the 190 global AMI speaker names in `meetings.xml`, sorted. They stay the same
for any subset. The speaker sets of train, dev and eval do not overlap, because the official
partition is speaker-disjoint.

**License:** CC BY 4.0 (`data/ami/annotations/LICENCE.txt`). Attribution: the AMI Meeting Corpus,
J. Carletta et al., 2005/2006, University of Edinburgh / AMI consortium. If you use the split,
also cite Landini et al. 2020 (VBx).

## Label definitions (defaults in `ami.LABEL_DEFAULTS`, all configurable)

All labels come from **word timings only** (`words/*.words.xml`). Punctuation tokens, vocal sounds
(laughs), gaps and disfluency markers are dropped. This follows the BUT "only_words" reference. There is
no energy VAD. Frames are 80 ms. An interval [s, e) sets frames `int(s/0.08) … ceil(e/0.08)-1`, the same
convention as `data.rttm_to_frames`. For every example, `T == ToneLanguage.n_frames(len(audio))`.

- **activity**: the union of a speaker's word intervals. `act_bridge=0`: only touching words merge.
- **run**: a maximal sequence of one speaker's words where every inter-word gap is < `turn_gap` = 0.5 s.
- **backchannel**: a run of ≤ 2 words that lasts ≤ 1.0 s (for example "yeah", "mm-hmm").
- **turn** (`turn_def`):
  - `"gap"`: turn = run. This is the literal "gap < 0.5 s" rule.
  - `"floor"` (**default**): consecutive runs of the same speaker form one turn, unless the pause
    between them is ≥ `max_hold` = 2.0 s or another speaker produces a non-backchannel run inside
    the pause (someone else took the floor).
- **hesitation**: a silence of ≥ `hes_gap` = 0.3 s between consecutive words of one turn.
- **switch gap**: next other-speaker non-backchannel turn start minus turn end. A negative value means overlap.
  This is the *true* end-of-turn gap.
- **hold gap**: silence before the *same* speaker's next turn when nobody took the floor in between.

Why `floor` is the default: AMI word timings are **contiguous inside a transcriber segment**. 91.4 % of
inter-word gaps are exactly 0, and 99 % of the non-zero ones are ≥ 0.5 s (segment boundaries). So under
the `gap` rule there are almost no hesitations: 36 in 7.5 h of train, all 0.3–0.5 s. Also, 27 % of
`gap` turn ends (train: 1056 of 3914) are the same speaker simply resuming after 0.5 s or more of
silence with nobody else talking. Those are exactly the pauses a silence timeout wrongly treats as an
end of turn, so `gap` labels them as the wrong answer.

Example modes (`AMI.examples(mode)`, `recipe_data(cfg, split)` with `data: {ami: {mode: ...}}`):

- `diar`: fixed 20 s windows with a 10 s hop. Returns `audio` and `spk_targets` (T,4), columns in arrival order within the window.
- `turn`: one example per non-backchannel turn.
  - The window starts at the latest of: the speaker's previous turn end, and turn start − 4 s.
  - It ends at the earliest of: the speaker's next turn start, and turn end + 2 s.
  - Its length is capped at 20 s. Turns with < 1 s of trail are skipped.
  - So the primary speaks exactly **one** turn per example, as the turn head's label requires
    (`heads.turn.eot_targets`: EOT is every frame after the last active frame of `spk_act`).
  - Keys follow `conversation.py`: `spk_targets` (column 0 = primary, then arrival order), `spk_act` = `primary_act`,
    `eot`, `turn_end_frame`, `onset_frame`, `text`, `speaker` (global id). There is also `hes` (T,): the primary's
    silent hesitation frames.
  - Note: the plan asked for fixed windows with "the speaker with the most speech" as the primary.
    That gives several primary turns per window, which the EOT label definition cannot express, so
    the windows are built around turns instead.
- `asr`: single-speaker stretches. These are runs with no other speaker's word within 0.2 s, split at word
  boundaries into chunks ≤ 15 s. Keys are `audio`, `text`, `speaker`, `vad`, `eou`, the same as LibriSpeech.
- `text`: lowercased, AMI punctuation tokens removed. `T_V_` becomes `tv`, hyphens become spaces, and only `[a-z0-9' ]` is kept.

| examples | diar (20 s) | turn (mean length) | asr (hours) |
|---|---|---|---|
| train | 2693 | 3274 (10.5 s) | 930 (1.81 h) |
| dev | 742 | 974 (10.1 s) | 354 (0.63 h) |
| eval | 870 | 941 (10.8 s) | 323 (0.71 h) |

In 15–17 % of turn examples the primary hesitates at least once. In 6–11 % the turn started before the
20 s window (`onset_clipped`). No window had more than 4 active speakers.

## Turn-taking statistics (`floor` definition; `data/ami/stats.json`)

| | train | dev | eval |
|---|---|---|---|
| turns / min (non-backchannel) | 7.4 | 8.0 | 6.6 |
| backchannels / min | 4.9 | 3.7 | 5.3 |
| turn length p50 / p90 (s) | 3.4 / 16.8 | 3.2 / 13.7 | 3.8 / 19.0 |
| **switch gap** p10 / p50 / p90 (s) | −1.54 / 0.12 / 3.64 | −1.73 / 0.24 / 5.09 | −1.73 / 0.08 / 4.17 |
| switches that overlap (gap < 0) | 45.5 % | 43.1 % | 47.1 % |
| switch gap when positive p25 / **p50** / p75 (s) | 0.38 / **0.86** / 2.42 | 0.50 / **1.22** / 3.25 | 0.39 / **1.03** / 3.00 |
| **hesitations** per turn | 0.23 | 0.33 | 0.29 |
| hesitation p10 / **p50** / p90 (s) | 0.77 / **1.29** / 1.77 | 0.47 / **1.14** / 1.67 | 0.93 / **1.32** / 1.83 |
| hesitations longer than the median positive switch gap | **87 %** | 40 % | **83 %** |
| hold gap (same speaker resumes, nobody else talked) p50 (s) | 3.08 | 3.15 | 2.80 |

**The key number: the median within-turn hesitation (1.1–1.3 s) is *longer* than the median
silence at a true end of turn (0.9–1.2 s).** Half of all floor transfers are not preceded by any
silence at all: 43–47 % overlap. A silence timeout therefore cannot separate the two, as the sweep
shows:

| silence timeout τ | 0.3 s | 0.5 s | 0.7 s | 1.0 s | 1.5 s |
|---|---|---|---|---|---|
| turns containing a pause ≥ τ (train / eval) → false cutoff | 14.7 / 15.8 % | 14.4 / 15.8 % | 14.1 / 15.6 % | 13.5 / 14.9 % | 5.8 / 6.6 % |
| switches where the next speaker started < τ after the end (train / eval) | 56 / 58 % | 64 / 63 % | 69 / 67 % | 76 / 73 % | 82 / 80 % |

Caveats, all caused by the annotation:

- (1) Pauses shorter than about 0.5 s inside a transcriber segment are absorbed into word durations. They
  are invisible, so the real hesitation count is higher and the real hesitation distribution has more
  short pauses.
- (2) Under `floor`, hesitations are < `max_hold` = 2 s by construction (the cutoff at τ = 2 s is 0).
- (3) "Too late" counts other human participants starting to talk. An agent would be one more
  participant, so this measures how contested the floor is, not agent latency.

To recover short within-segment pauses, the next step is to add the individual headset channels
(`Headset-0..3.wav`, about 4× the audio) and run a per-speaker energy VAD inside words, or to re-align
with a CTC aligner.

## Commands

```bash
.venv/bin/python scripts/prepare_ami.py                    # lists + annotations + 20 wavs (+ .npy cache, stats)
.venv/bin/python scripts/prepare_ami.py --no-cache --max-minutes 4   # download only, resumable
.venv/bin/python scripts/prepare_ami.py --stats-only --turn-def gap  # annotation statistics, other rule
.venv/bin/python -m pytest -q tests/test_ami.py
```

Recipe: `data: {ami: {mode: turn|diar|asr, n_meetings: {train: 12, val: 4}, window_sec: 20, ...}}`
(`audioforge.train.load_data` routes it to `ami.recipe_data`). A speaker-ID head needs ≥ 190 classes,
because ids are global.
