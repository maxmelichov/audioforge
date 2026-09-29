# Multi-speaker diarization in the served system: what the user saw, why, and the fix (2026-09-28)

User report: "Diarization, and many speakers in the room doesn't work." This note reproduces it through the live
server exactly as the README quickstart starts it, diagnoses the six suspects with numbers, and measures a
flag-gated fix. Driver: `scripts/research/diar_fix.py` (clips, live-server runs, offline Session runs, scoring);
everything ran through `scripts/dev/gate.sh` on the shared laptop (CPU, 2 threads, load average 3-6 from other
jobs the whole time, so every RTF here is on a loaded machine). Scratch: `/Volumes/ExternalSSD/.../scratch/diar`.
Tests: `tests/test_diarization_fix.py`.

**Short version.** Three things were true at once. (1) `final.speaker` is not the speaker of the turn: it is the
*primary* column = the column with the most activity over the last 5 s, and a turn only ends when *that* column
has been silent for 1 s; in a room, other people's turns are appended to the primary's segment until the primary
changes, so finals mix speakers (purity 0.69-0.85) and are labelled with whoever dominated the window. (2) The
launcher's `--diarizer nemotron3` adds `--diar-spks 4`, which cuts the 8-column head to its first 4 arrival-order
columns: speakers 5-8 are not merged, they are *invisible* (6-speaker mix -> 4 ids; ICSI 5 speakers -> 4 ids);
Sortformer v2 has 4 columns by construction. (3) On this loaded machine a single 1x session already runs at RTF
0.87-0.98, so load shedding fires within the first minute, and shedding level 1 replaces the diarizer by the served
VAD in column 0: every final while shed is speaker 0 (two concurrent sessions: 8-30 s finals all labelled 0, ids
collapse to 3). The fix (section 3) keeps all 8 Nemotron-3 columns, attaches a voice-keyed stable id to each final
through a per-session speaker registry (the served speaker head, no extra model pass), holds the last stable column
at half diarizer cadence instead of dropping the diarizer when shed, and adds an opt-in `timeout_any` policy that
ends every speaker's turn. Numbers in sections 3-4.

## 1. Reproduction (live server, README quickstart, 1x)

Server: `python -m audioforge.launch --diarizer nemotron3 --port 8792 --threads 2` (= `audioforge-serve
--diarizer nemotron3`; the launcher adds `--diar-spks 4 --diar-pool max --diar-left 1`), then the same with
`--diarizer sortformer` (port 8793). Client: `diar_fix.py run` streams int16 PCM in 20 ms blocks at 1x with the
default `timeout` policy (1000 ms) and logs every message. Clips:

| clip | source | seconds | speakers (>= 1 s of speech) | reference turns | overlap frames |
|---|---|---|---|---|---|
| `icsi_Bmr021_940s` | ICSI dev Bmr021 [940, 1000) s, the 60 s window of that meeting with the most speakers | 60 | 6 (5) | 30 | 121 |
| `mix6_s0` | 6 LibriSpeech test-clean speakers, 30 turns of 2-5 s (every speaker in the first 6, then shuffled round robin), gaps 0.3-1.2 s, 25 % of turns overlap the previous by 0.3-0.8 s, equal loudness | 118.8 | 6 (6) | 30 | 60 |
| `ami_IS1008b_1670s` | AMI dev IS1008b [1670, 1730) s | 60 | 4 (4) | 13 | 117 |

Scoring (per session, `diar_fix.score_events`): a final's span is [previous final, this final); its reference
speaker is the one with most speech in the span and its *purity* is that speaker's share of the span's speech;
**acc** = per-final speaker accuracy after a Hungarian map from emitted ids to reference speakers (generous: it
forgives a permutation but not a merge); **ids** = distinct `final.speaker` values on non-empty finals;
**count err** = ids minus reference speakers. `overloaded` = the number of `error/overloaded` messages;
`shed` = `stats.degraded.shed_diar_frames`.

### 1.1 Nemotron-3-Diarization (the recommended diarizer), one session at 1x

| clip | finals | ids emitted | count err | acc | purity | overloaded (first at) | shed frames | session RTF (load avg) |
|---|---|---|---|---|---|---|---|---|
| ICSI Bmr021, 5 speakers | 13 | 0 1 2 3 | -1 | 0.58 | 0.73 | 0 | 0 | 0.87 (4.3) |
| mix6, 6 speakers | 21 | 0 1 2 3 | -2 | 0.80 | 0.71 | 1 (74 s, backlog 2.4 s, rtf 1.42) | 33 / 1485 | 0.92 (5.2) |
| AMI IS1008b, 4 speakers | 7 | 0 1 2 | -1 | 0.86 | 0.72 | 6 (15 s) | 78 / 750 | 0.96 (5.5) |

Per-final detail, ICSI (speaker = `final.speaker`, gt = reference speaker of the span, n_gt = reference speakers
with >= 1 s in the span):

```
   0.00-  8.96  speaker 0  gt 0  purity 0.55  n_gt 3   "so i didn't really get any responses from the name conventio"
   8.96- 10.88  speaker 1  gt 2  purity 0.69  n_gt 0   ""
  10.88- 13.28  speaker 2  gt 5  purity 0.50  n_gt 2   "i don't know about the naming i"
  13.28- 14.96  speaker 0  gt 5  purity 1.00  n_gt 0   "mean the so"
  14.96- 16.40  speaker 3  gt 5  purity 1.00  n_gt 1   "these names that we've been using so far"
  16.40- 18.08  speaker 3  gt 5  purity 1.00  n_gt 0   "with the"
  18.08- 37.76  speaker 3  gt 5  purity 0.77  n_gt 3   "uh uh i wouldn't just want to change some you know without s"
  37.76- 40.40  speaker 0  gt 2  purity 0.39  n_gt 2   "well"
  40.40- 42.56  speaker 2  gt 3  purity 1.00  n_gt 1   "eventually she'll probably be consistent"
  42.56- 48.80  speaker 1  gt 3  purity 0.84  n_gt 1   "with what you are doing but yeah i kind of agree with andrea"
  48.80- 54.08  speaker 1  gt 3  purity 0.58  n_gt 2   "look fine to me but at the same time it was just like well n"
  54.08- 56.24  speaker 2  gt 3  purity 0.65  n_gt 1   "really affect what you're doing no but i think just to be co"
  56.24- 60.00  speaker 1  gt 3  purity 0.52  n_gt 2   "we should also i mean have the same conventions just so you "
```

Reference speaker 3 (the most talkative, 40-60 s) is reported as 1, 2 and 1 again; speaker 5 as 2, 0, 3, 3, 3;
the 18-38 s final holds three people. Per-final detail, the 6-speaker mix (Nemotron-3, 4 columns): 21 finals for
30 turns; the finals at 13.8-26.7 s (4 reference speakers, purity 0.36), 78.3-91.5 s (4 speakers, 0.33) and
99.4-111.0 s (3 speakers, 0.33) are several turns glued together while the primary stayed on one column; reference
speakers 4 and 5 never get an id of their own (`ids_per_ref_speaker`: 4 -> {2}, 5 -> {0, 1, 3}).

### 1.2 Streaming Sortformer v2 (the download default), one session at 1x

| clip | finals | ids emitted | count err | acc | purity | overloaded (first at) | shed frames | RTF (load) |
|---|---|---|---|---|---|---|---|---|
| ICSI Bmr021, 5 speakers | 9 | 0 1 2 3 | -1 | 0.50 | 0.69 | 3 (44 s) | 84 | 0.96 (4.8) |
| mix6, 6 speakers | 35 | 0 1 2 3 | -2 | **0.26** | 0.80 | 9 (42 s) | 96 | 0.98 (4.5) |
| AMI IS1008b, 4 speakers | 7 | 0 1 2 3 | 0 | 0.57 | 0.85 | 1 (49 s) | 6 | 0.96 (3.9) |

On the mix Sortformer v2 produces 35 finals for 30 turns: its columns swap during utterances ("the influence which
de timaeus has exer" / "cised upon posteria" as two finals with speakers 0 and 3), and reference speaker 0's turns
are labelled 0, 2, 3 and 2. Its 4 columns cannot hold 6 people either.

### 1.3 Two concurrent sessions at 1x (load shedding)

Session 0 streams the ICSI clip, session 1 the mix, both at 1x against the same server (2 threads).

| server | session | finals | ids | count err | acc | purity | overloaded | shed frames / batched / partials dropped | RTF |
|---|---|---|---|---|---|---|---|---|---|
| Nemotron-3 | ICSI | 9 | 0 2 3 | -2 | 0.38 | 0.79 | 2 (level 2 at 24 s, backlog 6.3 s) | 483 / 300 / 47 | 0.53 |
| Nemotron-3 | mix | 23 | 0 1 2 3 | -2 (a) | 0.68 (a) | 0.76 (a) | 7 (level 2 at 28 s, backlog 8.0 s) | 570 / 282 / 41 | 0.75 |
| Sortformer v2 | ICSI | 7 | 0 1 2 | -2 | 0.50 | 0.63 | 4 (level 2 at 19 s) | 435 / 194 / 29 | 0.52 |
| Sortformer v2 | mix | 28 | 0 1 2 3 | -2 | 0.30 | 0.79 | 11 (level 2 at 24 s) | 582 / 168 / 27 | 0.75 |

(a) No stored file holds these three values: `scratch/diar/logs/n3_concurrent.json` has no `clip2` and scored this
session against the ICSI reference (5 speakers, 9 of 23 finals scored: count -1, acc 0.667, purity 0.669). The
Sortformer and post-fix concurrent files score the mix correctly. Noted by the 2026-09-28 verification pass.

While shed, the served VAD stands in as column 0 and the primary is 0, so the finals are speaker 0: ICSI
(Nemotron-3) 18.1-44.0 s (4 reference speakers), 44.0-59.8 s and 59.8-60.0 s are all speaker 0; on the mix the
13.8-35.6 s final (5 reference speakers, purity 0.34), 35.6-43.8, 47.4-56.2, 56.2-63.7, 65.8-69.7, 78.3-81.2,
81.2-83.6, 83.6-86.7 s are speaker 0 although they are reference speakers 4, 5, 4, 3, 5, 5, 5, 4. 483-582 of the
750 / 1485 diarizer frames were shed (level 2 most of the time: `frames` batches, partials dropped).

The `/health` counters after the five Nemotron-3 sessions: `overloaded 16, shed_diar_frames 1164, shed_partials
88, frames_batched 582`.

## 2. Diagnosis

The offline driver (`diar_fix.py offline`) runs the same `Session` the server runs, unpaced, block 160 ms, on the
three clips, and reproduces the live numbers (legacy Nemotron-3: mix acc 0.80, ICSI 0.583, AMI 0.833, identical
ids). Shedding is forced with `--shed 1` (`SHED_RTF` set to 0 so level 1 is entered). Every configuration below is
one call; "DER" is the pooled frame DER of the diarizer rows the session actually used (all columns, Hungarian
map, 0.5 threshold, no collar, the word-level references of section 1), "count" the number of columns active for
>= 1 s minus the reference speaker count, "ids" the distinct `final.speaker` values. Suspect by suspect:

| # | suspect | verdict | evidence |
|---|---|---|---|
| 1 | shedding level 1 replaces the diarizer with the VAD in column 0 | **confirmed, the largest effect once the machine is loaded** | forced level 1, legacy: **one id (0) per clip**, count -4 on average, mix 9 finals all speaker 0, ICSI and AMI one 60 s final each; DER 0.81 (confusion 0.46). Live: a single 1x session already sheds within 15-75 s on this machine (RTF 0.87-0.98), two sessions shed 483-582 of 750 / 1485 frames |
| 2 | `--diar-spks 4 --diar-pool max` merges 8 columns into 4 | **confirmed, but it *drops*, it does not merge**: the cut keeps the first 4 arrival-order columns; speakers 5-8 get all-zero rows | uncut Nemotron-3 on the same clips: DER 0.395 -> **0.247**, speaker count exact on all three clips (6 / 5 / 4; cut: 4 / 4 / 4), mix DER 0.45 -> 0.15 (the 0.45 was pure miss: two speakers had no column). On the ICSI window the 8 columns over-split (FA 0.12 -> 0.21, confusion 0.14 -> 0.21) while miss falls 0.28 -> 0.09 |
| 3 | Sortformer v2 is a 4-speaker model and the 0.32 s setting has DER ~0.28 | **confirmed and worse than that here**: DER 0.67 pooled on the clips, confusion 0.42 (mix 0.80 / 0.50) | its columns swap inside utterances (36 finals for the mix's 30 turns, each reference speaker labelled with 3-4 different ids); the registry recovers per-final accuracy 0.42 -> 0.76 but over-splits (9 ids for 6) because the swaps cut turns into pieces. Our AOSC procedure is not NeMo's for this model (SORTFORMER_IMPORT.md "Streaming caveat"), which is a separate issue; the product recommendation is Nemotron-3 anyway |
| 4 | the final's speaker is the dominant column over the last 5 s and flips between columns | **confirmed as the mechanism, in two parts**: (a) the label is the 5 s primary, not the turn's speaker; (b) a turn only ends when the *primary* falls silent, so other speakers' turns are appended to it | purity (share of a final's speech that belongs to its dominant speaker) 0.71 / 0.73 / 0.72 with `timeout`; mix: 21 finals for 30 turns; three finals hold 3-4 speakers each. With `timeout_any` (section 3.1) purity 0.86 / 0.80 / 0.75 and 30 finals for 30 turns. Column permutation proper (the same voice on different Nemotron-3 columns within a clip) is rare in these 1-2 min clips: with the registry off, the reference speakers of the mix map to one column each except speaker 5 (4 columns cut) |
| 5 | enrollment / primary binding collapsing the room to primary vs other | **not on this path**: the quickstart runs `--enroll dominant` (no binder) and `--turn-input diar`; the collapse comes from suspect 4's rule, not from a voice binding. `--enroll after_agent_arm` / `--turn-input tsvad` do collapse to user-vs-other by design (documented) | code path (`Session.binder is None`, `tsvad is None` in the quickstart) |
| 6 | the diar head's speaker under-count (1.8-2.0 vs 2.62) | **not on this path**: the server never runs our diar head; the served ASR model's `diar` head has weight only in training. Section 5 is where it comes back (distillation) | `Engine.diar_head` is Nemotron-3's / Sortformer's; `stage1_served.afm`'s diar head is not called in `Session.process` |

So what the user saw is (1) + (2) + (4): a room of 5-6 people comes out as at most 4 ids, each final labelled with
whoever dominated the last 5 s, several people's turns glued into one final, and, as soon as the machine is busy
(which on this laptop is always), everything is speaker 0.

## 3. Fix (flag-gated; `audioforge/speaker_registry.py`, `audioforge/serve.py`, `audioforge/launch.py`)

### 3.1 What changed

- **`--diar-spks` default = the model's maximum.** The launcher no longer adds `--diar-spks 4` for `--diarizer
  nemotron3`; `frame.speakers` carries 8 columns for an uncut Nemotron-3 (`validate` accepts 4-8, `primary` < the
  count). `--diar-spks 4` restores the old wire shape.
- **`--diar-labels registry`** (default `column` = old behaviour). Each final's speaker is a stable per-session id
  keyed by voice: the column that dominates the turn's *own span* (`speaker_registry.turn_column`) selects the
  speaker's frames, the served speaker head embeds them from the block-4 frames the ASR pass already produced
  (`ASRStream.spk_feats`, no extra model pass), and `SpeakerRegistry.assign` returns the closest known speaker when
  the cosine is >= the threshold (else a new id; centroids are running means, capped). Too few own frames (< 0.64 s;
  overlap or shedding) -> the served VAD's frames; still too few -> the column's last id, else `null`. Finals get
  `speaker_conf` (the match cosine; for a new speaker how far the closest known one was below the threshold) and
  `diar_shed`; `stats` gets `speakers_seen` (= the speaker-count estimate). `--diar-embed titanet` embeds each turn's
  audio with TitaNet-L instead (one call per final; +91 MB). Thresholds (`--diar-reg-thr`) come from section 3.2.
- **`--shed-diar hold`** (default `vad` = old behaviour). A shed diarizer frame reports the last real row's most
  active column (else the primary, else 0) carrying the served VAD (`speaker_registry.held_row`): speech keeps the
  last stable column, silence stays silence so the timeout keeps firing; level 1 runs the diarizer on every other
  block (half its cost) instead of not at all, level 2 still skips it; `diar_shed: true` marks the affected finals.
- **`turn_policy: "timeout_any"`** (session config, opt-in; `AnySpeakerTimeout`): the turn belongs to whoever spoke
  last; it ends after `timeout_ms` of nobody talking (`turn_end.policy: timeout`) or when another column has been
  active for 240 ms while the previous speaker has been silent for 240 ms (`policy: change`; overlap and short
  backchannels do not cut). After a timeout the floor is open, so the next speaker starts a fresh turn.
- Protocol additions are all optional keys / a wider range (PROTOCOL.md §4.2, §5.2, §5.4, §5.5, §5.8); the shipped
  clients (`scripts/stream_client.py`, the Pipecat / LiveKit adapters, `audioforge-client`) read `speakers` as a
  list of any length and ignore unknown keys.

### 3.2 Registry threshold calibration (`diar_fix.py calib`, the mix's 30 reference turns)

| embedder | same-speaker cosine mean (p10) | different-speaker mean (p90, max) | chosen `--diar-reg-thr` | sequential-registry sweep |
|---|---|---|---|---|
| served speaker head (block-4 frames of the turn) | 0.761 (0.659) | 0.290 (0.452, 0.613) | **0.55** | acc 1.0 / 6 ids at 0.6; 0.55 sits between p10 same and p90 different |
| TitaNet-L (turn audio) | 0.685 (0.560) | 0.015 (0.151, 0.333) | **0.40** | acc 1.0 / 6 ids from 0.30 |

These are clean read speech; on the meeting clips the speaker-head margins are smaller (section 3.3: it splits a
speaker in two on the mix and on ICSI at 0.55; TitaNet does not on the mix).

### 3.3 Before / after on the three clips (Nemotron-3 unless stated; offline `Session`, 2 threads, loaded machine)

acc = per-final speaker accuracy after a Hungarian map; ids = distinct final speakers (reference 6 / 5 / 4);
purity as in section 1; DER = pooled over the three clips; RTF = the session's wall RTF (compute / audio) in
process, mean over the clips (server RTF is measured live in section 4).

| configuration | mix acc / ids / purity | ICSI acc / ids / purity | AMI acc / ids / purity | mean acc | pooled DER (miss / FA / conf) | diarizer speaker count (6/5/4) | RTF |
|---|---|---|---|---|---|---|---|
| **before**: `--diar-spks 4`, `column`, `vad`, `timeout` | 0.80 / 4 / 0.71 | 0.58 / 4 / 0.73 | 0.83 / 3 / 0.72 | 0.74 | 0.395 (.321 / .034 / .040) | 4 / 4 / 4 | 0.53 |
| uncut (8 columns), `registry` (spk), `hold`, `timeout` | 0.87 / 7 / 0.82 | 0.80 / 4 / 0.74 | 0.83 / 3 / 0.72 | 0.83 | **0.247** (.136 / .056 / .055) | **6 / 5 / 4** | 0.53 |
| uncut, `registry` (spk), `hold`, **`timeout_any`** | 0.80 / 7 / **0.86** | 0.82 / 4 / **0.80** | 0.75 / 3 / 0.75 | 0.79 | 0.247 | 6 / 5 / 4 | 0.53 |
| uncut, `registry` (**TitaNet**), `hold`, `timeout_any` | **1.00 / 6** / 0.86 | 0.77 / 5 / 0.80 | 0.88 / 5 / 0.75 | **0.88** | 0.247 | 6 / 5 / 4 | 0.57 |
| `--diar-spks 4`, `registry` (spk), `hold`, `timeout_any` | 0.70 / 4 / 0.72 | 0.93 / 3 / 0.74 | 0.75 / 3 / 0.75 | 0.79 | 0.395 | 4 / 4 / 4 | 0.53 |
| **before, shed level 1 forced** | 0.33 / **1** / 0.64 | 1.00 / **1** / 0.32 (one 60 s final) | 1.00 / **1** / 0.45 (one final) | - | **0.809** (.296 / .049 / .464) | 1 / 1 / 1 | 0.27 |
| uncut, `registry` (spk), `hold`, `timeout_any`, **shed level 1 forced** | 0.91 / 9 / 0.79 | 0.73 / 4 / 0.70 | 0.57 / 5 / 0.75 (1 null) | 0.74 | 0.329 (.215 / .067 / .047) | 6 / 5 / 4 | 0.44 |
| Sortformer v2, before | 0.25 / 4 / 0.81 | 0.43 / 4 / 0.64 | 0.57 / 4 / 0.85 | 0.42 | 0.673 (.183 / .071 / .418) | 4 / 4 / 4 | 0.78 |
| Sortformer v2, `registry` (spk), `hold`, `timeout_any` | 0.69 / 9 / 0.87 | 0.75 / 4 / 0.83 | 0.85 / 4 / 0.86 | 0.76 | 0.673 | 4 / 4 / 4 | 0.75 |

Reading:

- The 8 columns are what fixes the room: DER 0.395 -> 0.247 and the speaker count exact on every clip, at the same
  RTF (the head's last linear layer is the only difference). With the cut in place the registry cannot recover the
  missing people (row 5: their frames have no active column, so no turn ends on them).
- The registry fixes the *labels*: per-final accuracy 0.74 -> 0.83 with the served speaker head (free) and 0.88
  with TitaNet (+0.04 RTF, +91 MB). The speaker head over-splits (7 ids for 6 on the mix: speaker 1 -> {1, 4},
  speaker 2 -> {2, 6}); TitaNet gets the mix exactly right (6 ids, acc 1.0) and over-splits on ICSI / AMI (5 ids for
  5 / 4, where the merged turns of `timeout_any` still mix voices).
- `timeout_any` fixes the *segmentation*: purity 0.71 -> 0.86 (mix) and 0.73 -> 0.80 (ICSI), 30 finals for the
  mix's 30 turns; it costs a little accuracy with the speaker head (shorter turns, fewer frames per embedding),
  none with TitaNet.
- Shedding: legacy collapses to one id per clip and DER 0.81; `hold` keeps the count and the ids (DER 0.33,
  accuracy 0.74, one `null` on AMI) at RTF 0.44 instead of 0.27 (legacy drops the diarizer entirely; `hold` keeps
  half of it) - still well under the unshed 0.53, which is the point of level 1.
- Sortformer v2 gains from the registry (0.42 -> 0.76) but its column swaps make it over-split; it stays the wrong
  diarizer for rooms (FINAL_REPORT §4 already recommends Nemotron-3).

### 3.4 AMI dev, 64 x 20 s windows (the FINAL_REPORT §4 set; 8-column references, pooled frame DER)

Same offline `Session` path, 64 windows = 1280 s, 2 threads, loaded machine (load 3-5); the "after" row is the
full proposed configuration (uncut, `registry` with the speaker head, `hold`, `timeout_any`). "count" = diarizer
columns active >= 1 s minus reference speakers (mean signed / mean absolute); "final ids" = distinct final speakers
minus reference speakers per window; acc = per-final Hungarian accuracy (windows are 20 s, 2.6 speakers on average,
so the column label is already a strong baseline here); RTF = in-process wall RTF.

| configuration | pooled DER (miss / FA / conf) | count err (signed / abs) | final ids err | finals (null) | per-final acc | shed frames of 16 000 | RTF |
|---|---|---|---|---|---|---|---|
| before: `--diar-spks 4`, `column`, `vad`, `timeout` | 0.245 (.195 / .024 / .025) | -0.09 / 0.22 | -0.63 | 167 (0) | 0.926 | 0 | 0.46 |
| after: uncut, `registry` (spk), `hold`, `timeout_any` | 0.245 (.195 / .024 / .025) | -0.09 / 0.22 | -1.03 | 241 (3) | 0.765 | 0 | 0.44 |
| before, shed level 1 forced | 0.490 (.329 / .033 / .127) | -1.28 / 1.31 | -1.28 | 97 (0) | 0.915 (one id per window) | 15 936 (all) | 0.25 |
| after, shed level 1 forced | **0.335** (.258 / .036 / .041) | -0.25 / 0.38 | -1.08 | 207 (10) | 0.767 | 7 872 (half) | 0.30 |

(Corrected 2026-09-28 by the final-report verification pass, `research/VERIFICATION_2026-09-28.md`: this row first
showed the aggregate of the resumable run's first pass, 57 of 64 windows (DER 0.229, acc 0.772, 206 finals, ids
-0.93, count 0.19). The complete 64-window file, `scratch/diar/offline/ami_B_uncut_hold.json` `aggregate`, gives the
values above; the pooled DER and count error equal the "before" row to four decimals.)

Reading: on 20 s windows with at most 4 speakers the 8 columns change nothing measurable (DER 0.245 in both rows,
the FINAL_REPORT §4 offline figure for this model is 0.232), and the voice registry is *worse* than the column label
for per-final accuracy (0.93 -> 0.77): a registry that starts empty every 20 s sees one or two short turns per speaker,
the speaker head's meeting-speech margins are small, and `timeout_any` makes more and shorter finals (241 vs 167) that
fall back to the column's last id or `null` (3) when they hold less than 0.64 s of the speaker's own frames. The
registry earns its place in long multi-party sessions (section 3.3: 6 speakers, 2 minutes) and under shedding, not
on short clips with few speakers, which is why the proposal below keeps the column label as the default and makes
the registry a flag. Under forced level-1 shedding the picture is the same as on the clips: legacy DER 0.49 with
one id per window, `hold` 0.34 with the count within 0.4 of the truth, at RTF 0.30 instead of 0.25 (the diarizer
still runs on every other block).

## 4. Live server with the new flags, and the proposed default

Server: `python -m audioforge.launch --diarizer nemotron3 --port 8794 --threads 2 --diar-labels registry
--shed-diar hold` (the launcher no longer adds `--diar-spks 4`), same client and clips as section 1, 1x. The
machine was quieter than in section 1 (load 3.7 vs 4.3-6.0), so the RTFs are not paired with section 1's; the
controlled shedding comparison is section 3.3 / 3.4 (forced level 1). Both runs stayed under the 0.8 bar.

| clip | policy | finals | ids emitted | count err | acc | purity | overloaded | session RTF |
|---|---|---|---|---|---|---|---|---|
| ICSI Bmr021, 5 speakers | `timeout` | 11 | 0 2 3 4 | -1 | 0.80 | 0.74 | 0 | 0.64 |
| mix6, 6 speakers | `timeout` | 31 | 0-6 (7) | +1 | 0.87 | 0.82 | 0 | 0.66 |
| AMI IS1008b, 4 speakers | `timeout` | 6 | 0 1 2 | -1 | 0.83 | 0.72 | 0 | 0.60 |
| ICSI Bmr021 | `timeout_any` | 17 | 0 1 2 3 | -1 | 0.81 | 0.75 | 0 | 0.55 |
| mix6 | `timeout_any` | 32 | 0-6 (7) | +1 | 0.84 | 0.86 | 0 | 0.62 |
| AMI IS1008b | `timeout_any` | 9 | 0 1 2 | -1 | 0.75 | 0.75 | 0 | 0.57 |
| ICSI + mix, two concurrent sessions | `timeout_any` | 17 / 32 | 4 / 7 | -1 / +1 | 0.81 / 0.84 | 0.75 / 0.86 | 0 / 0 | 0.61 / 0.67 |

Against section 1.1 (same clips, legacy flags): the mix goes from 4 ids / 21 finals / acc 0.80 / purity 0.71 to
7 ids / 31-32 finals / acc 0.84-0.87 / purity 0.82-0.86; ICSI from 4 ids / acc 0.58 to 4 ids / acc 0.80-0.81 (its
fifth speaker has 3.5 s of speech in the window and is still merged); AMI IS1008b is unchanged (3 ids for 4: the
fourth speaker's turns are short and overlapped). The live outputs equal the offline runs of section 3.3 message
for message where the machine did not shed, as in section 2. Peak RSS 1.55 GB (1.24 GB with the legacy flags: the
speaker head's per-frame features and the 8 columns). On the quieter machine the two concurrent sessions did not
reach level 1 (RTF 0.61 / 0.67, no `overloaded`), which is why the shedding comparison is the forced one above.

### Proposed default

1. **Uncut Nemotron-3** (`audioforge-serve --diarizer nemotron3` no longer adds `--diar-spks 4`; done in
   `audioforge/launch.py`): the largest fix for rooms (DER 0.395 -> 0.247 on the clips, speaker count exact; on AMI-64, at most
   4 speakers per 20 s window, DER is unchanged at 0.245) at the same RTF. `frame.speakers` becomes 8 long for this diarizer; clients that hard-code 4 need
   `--diar-spks 4`.
2. **`--shed-diar hold`** as the launcher default for both diarizers (done in `audioforge/launch.py`; `serve.py`'s
   own default stays `vad` so the measured behaviour of a bare `python -m audioforge.serve` is unchanged): it only
   acts while shed, and there it replaces "everyone is speaker 0" (DER 0.49-0.81, one id) by the last stable column
   (DER 0.33-0.34, count within 0.4) for +0.05 RTF while shed. `--shed-diar vad` restores the old rule.
3. **`--diar-labels registry` stays opt-in**, recommended together with `turn_policy: "timeout_any"` for a
   multi-party room, and with `--diar-embed titanet` when TitaNet-L is downloaded (per-final accuracy 0.88 vs 0.79
   with the served speaker head on the 6-speaker mix, +0.04 RTF, +91 MB). For a voice agent following one user
   (the E2E-measured product) the primary column label and `timeout` remain the default: on short windows with few
   speakers the column label is more accurate than the voice registry (AMI-64: 0.93 vs 0.77).
4. Sortformer v2 should not be used for rooms (section 2, suspect 3); it stays the download default only because
   every benchmark number was measured with it.

What is not fixed: turns that hold several overlapping speakers still get one label (purity 0.75-0.86, not 1);
speakers with under 0.64 s of clean speech in a turn fall back to their column's last id or `null`; the speaker
head over-splits a voice now and then (7 ids for 6); Sortformer v2's column swaps in our streaming procedure.


## 5. Distilling Nemotron-3 into our own head (one experiment, killed)

Question (the user: "NVIDIA has a diarization model; I know we can make our model be this model well"): can a head
of at most 2 M parameters on the frozen served encoder reach Nemotron-3-Diarization's quality? If it could, the
server would drop the second model: the diarizer is about half of server compute (a separate 176 MB model with
its own encoder; the improvements agent measured served RTF 0.34 without the diarizer pass vs 0.80 with it,
commit 41008ab), and it would free about 180 MB of RSS.

Setup (`scripts/research/diar_distill.py`, `research/recipes/diar_distill_nemotron3.yaml`, results `runs/diar_distill.json`):

- Teacher: Nemotron-3-Diarization offline, max pooling to 80 ms, all 8 columns, cached once on the AMI (12 meetings,
  2 697 windows of 16 s) and ICSI (12 meetings, 2 977 windows) train diar windows: 144 s each on the CPU,
  `data/cache/diar_teacher/*.npz` (symlink to the SSD, 9-10 MB, read back after writing).
- Student: the stage-1 Sortformer head (1.92 M params, 4 columns), warm-started from `stage1_served.afm`, encoder
  frozen (`init.train_only`; after training, 704 frozen tensors compared bit-identical to the init). Taps: block 6
  (LAYER_ROUTING.md's best single block) and a learned mix of blocks 4 + 6.
- Loss: `(1 - w) * (PIL + sorted BCE vs the labels) + w * (the same vs the teacher's posteriors sorted by arrival and
  cropped to 4 columns)`, w = 0.5 (`SortformerHead(distill=...)`; unset means the old loss, bit-identical, see the
  test). 1 800 steps, batch 6, lr 1e-3, MPS, 8-9 min per run, run only when no other MPS job was up.
  Checkpoints on the SSD (`scratch/diar/ckpt`), loaded back for the evaluation.
- Evaluation: pooled frame DER (Hungarian map, 0.5 threshold, no collar) and speaker-count error (columns active
  >= 1 s minus reference speakers per window) on the 64 AMI dev windows of FINAL_REPORT §4 and 64 ICSI dev windows
  (20 s, seeded, never trained on). Bar: DER within 0.05 of Nemotron-3 on both, count error no worse. Kill rule:
  stop after two runs if the gap is above 0.10.

| model | params (diarizer) | AMI-64 DER (miss / FA / conf) | AMI count err (signed / abs) | ICSI-64 DER (miss / FA / conf) | ICSI count err |
|---|---|---|---|---|---|
| Nemotron-3-Diarization, teacher (8 columns, max pool) | separate model, 176 MB | **0.232** (.191 / .027 / .014) | -0.14 / 0.23 | 0.236 (.005 / .228 / .004) | +0.06 / 0.09 |
| served stage-1 head (before) | 1.92 M on the served encoder | 0.394 (.246 / .059 / .089) | -0.72 / 0.75 | 0.354 (.079 / .209 / .066) | -0.28 / 0.28 |
| run 1: distilled, block 6 | 1.92 M | 0.385 (.304 / .031 / .050) | -0.69 / 0.75 | **0.193** (.076 / .082 / .036) | -0.22 / 0.22 |
| run 2: distilled, blocks 4 + 6 (learned 0.37 / 0.63) | 1.92 M | 0.367 (.270 / .031 / .067) | -0.66 / 0.72 | 0.196 (.075 / .081 / .040) | -0.23 / 0.23 |

Verdict: **killed after two runs; the bar is not met.** On AMI the gap to the teacher is 0.135-0.153 (above the
0.10 kill line) and the student still under-counts speakers (-0.66 vs -0.14): it misses a quarter of the speech of
the third and fourth speakers in a window, which is exactly what Nemotron-3 gets right. On ICSI dev the student is
*better* than the teacher (0.19 vs 0.24, count error -0.22 vs +0.06 in the other direction): the teacher's ICSI error
is almost all false alarm against this project's word-level labels, and the student was trained on those labels.
Caveats that point the same way: (1) Nemotron-3's model card lists AMI train + dev in its training data, so its
AMI-dev DER of 0.23 is partly contaminated and not a fair target; (2) the two runs have no no-teacher control
(the two-run cap), so the ICSI gain cannot be split between the teacher term and simply adding ICSI train windows
(the served head's recipe had none; LAYER_ROUTING.md's block-6 head without ICSI was 0.32-0.34 on ICSI);
(3) 1.9 M parameters on a 16-block encoder tuned for ASR, 4 columns, 22 h of meetings. What it would have saved:
the second encoder pass, about half of server compute (RTF 0.80 -> about 0.4 on 2 threads) and 180 MB. What would
be needed instead, if this is picked up again: more teacher-labelled data (the teacher caches 22 h in under 5
minutes on the CPU, so all 170 h of AMI + ICSI or unlabelled meeting audio is cheap), a partially unfrozen upper
encoder, and 8 output columns. Until then the served diarizer stays Nemotron-3 with the section 3 fixes.
