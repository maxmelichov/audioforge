# Target-speaker WER (tWER) of single-model mode, 2026-09-29

**Question.** In a recording with several speakers, how accurately do we transcribe **only the enrolled user's
words**? Single-model mode (`--mode single`) keeps the streaming words where the TS-VAD head (`assets/tsvad_spk.pt`, 5 s
stored voice print) says "target". This had not been measured before. All earlier WERs (`runs/final_asr.json`,
`runs/hybrid_asr.json`, AMI-200 24.4 %) score single segments or all speakers.

## Definition

- **Units.** The IMPROVE_115M A.1 / A.1b frame-eval units: the eot-bench v2 extended windows (AMI dev 974 in 4
  meetings, ICSI held-out 1312 in 5 meetings, about 20 s each). Each window is scored once per labelled speaker who
  talks in it and has a 5 s print (the cached `tsvad.py vprints`: that speaker's single-speaker speech elsewhere in the
  meeting, embedded by the served block-4 speaker head). **Primary** = column 0, the turn's speaker (A.1's headline
  group). **All speakers** = every such target.
- **Reference.** The target's own words, taken from the corpus word alignments, whose midpoint falls in the window.
- **Hypothesis.** Streaming pass 1 of the served model (`runs/stage1_served.afm` RNNT, att [70,1], greedy, masked
  forward on the window audio, which gives the same output as cache-aware streaming). Each word is timed by the
  encoder frames where its tokens were emitted.
- **Filter.** A word is kept if the arm's target mask is on at the word's frame. The frame is shifted back by the
  median emission lag, 5 frames = 400 ms, measured on correctly recognised words against all speakers' reference
  words, so it uses no target information. The mask is widened by ±2 frames. Every arm uses the same rule, and
  dilation 0 / 4 are in the json. Our track: P(target) > 0.5.
- **Scoring.** `teachers.normalize_text`. tWER = (S + D + I) / N, summed over units. 95 % CIs come from 1000 bootstrap
  resamples, by meeting (only 4 / 5 meetings, so these are coarse) and, in the json, by window. Paired deltas use the
  same resamples.
- **Arms.** *ours* = served words kept by our TS-VAD track. *oracle* = kept by the target's reference activity, which
  isolates ASR error. *no filter* = every word. *Nemotron-3* = the cached A.1b Nemotron-3-Diarization column (served
  settings) bound by the same 5 s print with `vp_follow`, using the served speaker head (TitaNet-L in brackets).
  *0.6B + ours* = nemotron-speech-streaming-en-0.6b words at [70,1] kept by our TS-VAD track. Its decodes of all 2286
  windows took 16 CPU-minutes, with no downloads.
- **TurnBench.** The 16 dev clips of the live single-model study (IMPROVEMENTS.md section 1): 35-53 s mono mixes, with
  the human party as the target and the stored 5 s print taken from the user's own channel. The reference is
  annotator a's transcript of the user's channel, with word times interpolated inside each segment. The oracle mask is
  those segments. There are no Nemotron-3 tracks for these clips. **oto: not scored**, because it has no human
  transcripts.

## Results (tWER %, [meeting CI]; S / D / I in % of reference words)

| set (units, ref words) | ours (TS-VAD) | oracle filter | no filter | Nemotron-3 + print | 0.6B + ours |
|---|---|---|---|---|---|
| AMI dev, primary (974, 14028) | **61.6** [48.3, 74.0]<br>20.4 / 26.1 / 15.1 | 47.4 [36.7, 53.6]<br>15.6 / 26.8 / 5.1 | 106.5 [89.6, 121.3]<br>30.6 / 11.2 / 64.7 | 80.9 [64.6, 92.5] (TitaNet 74.7)<br>29.5 / 22.4 / 29.0 | 57.0 [43.3, 68.8]<br>16.8 / 15.6 / 24.6 |
| AMI dev, all speakers (2947, 35415) | **75.7** [60.8, 88.7]<br>24.2 / 28.9 / 22.6 | 54.8 [46.1, 59.0]<br>18.7 / 30.5 / 5.6 | 148.9<br>35.4 / 12.9 / 100.7 | 112.0 (TitaNet 103.7)<br>33.1 / 25.3 / 53.7 | 72.8<br>20.4 / 17.3 / 35.1 |
| ICSI held-out, primary (1286, 25056) | **37.1** [33.4, 41.6]<br>13.8 / 15.8 / 7.4 | 31.3 [28.6, 34.9]<br>14.7 / 12.7 / 4.0 | 100.8 [84.0, 115.6]<br>20.2 / 6.0 / 74.6 | 66.8 [59.1, 74.7] (TitaNet 65.1)<br>21.9 / 11.1 / 33.8 | 30.6 [27.7, 33.9]<br>9.5 / 13.9 / 7.3 |
| ICSI held-out, all speakers (3596, 54759) | **43.4** [40.3, 47.9]<br>16.2 / 18.9 / 8.3 | 37.7 [35.6, 40.8]<br>17.5 / 15.1 / 5.1 | 156.9<br>24.4 / 6.7 / 125.7 | 106.2 (TitaNet 105.0)<br>26.3 / 12.9 / 67.0 | 36.4<br>11.8 / 16.1 / 8.5 |
| TurnBench dev clips, user (16, 1350) | **45.0** [35.4, 58.0]<br>13.9 / 15.0 / 16.0 | 36.8 [28.2, 50.8]<br>13.5 / 6.3 / 17.0 | 62.8 [48.4, 86.7]<br>14.7 / 5.0 / 43.0 | not cached (not run) | 37.7 [29.2, 50.0]<br>8.3 / 11.3 / 18.2 |

Paired deltas (meeting CI):

- **Ours vs no filter:** AMI -44.9 [-49.3, -40.2], ICSI -63.7 [-80.6, -44.7], TurnBench -17.9 [-36.1, -6.0].
- **Ours vs Nemotron-3 + print (speaker head):** AMI -19.3 [-24.8, -15.8], ICSI -29.7 [-38.1, -19.8]. Against the
  TitaNet binding: -13.1 / -28.0.
- **Ours vs oracle:** AMI +14.2 [+9.1, +21.7], ICSI +5.8 [+4.8, +7.0], TurnBench +8.2 [-2.5, +20.5].
- **0.6B + ours vs ours:** AMI -4.6 [-5.6, -3.6], ICSI -6.5 [-8.2, -5.0], TurnBench -7.3 [-10.7, -4.2].

**The ASR ceiling on these windows.** Scoring every speaker's words against every decoded word gives an all-speaker
WER of 50.9 % on AMI (D 40.3) and 33.8 % on ICSI (D 23.7). For the 0.6B model it is 34.5 / 26.3. This is multi-party
mixed audio with overlap, which is harder than the single-speaker AMI-200 segments (24.4 %). The oracle-filter tWER
tracks this ceiling.

## Verdict

- **Our target track works as a word filter.** It removes most of the other people's words. Insertions fall from 65 %
  to 15 % on AMI and from 75 % to 7 % on ICSI, and tWER roughly halves (AMI 106 -> 62, ICSI 101 -> 37). It is also
  clearly better than today's Nemotron-3 column bound by the same print (-19 / -30 points), mainly because that column
  lets through twice as many crosstalk insertions.
- **What is left is mostly ASR, not attribution.** On ICSI, ours is within 6 points of the oracle filter. The user's
  own words are deleted a little more (D 15.8 vs 12.7) and a little more crosstalk gets through (I 7.4 vs 4.0). On AMI
  the gap is 14 points, nearly all of it leaked crosstalk (I 15.1 vs 5.1) and substitutions. The filter deletes no
  more of the user's words than the oracle does (D 26.1 vs 26.8). But even with perfect attribution the user's words
  are at 47 % (AMI) / 31 % (ICSI) tWER, because the 115M streaming pass drops overlapped and quiet speech.
- **A bigger ASR on top of the same track helps.** Swapping in the 0.6B words gains 5-7 points everywhere. Its oracle
  filter shows about 13 points of headroom on AMI.

## Caveats

- The windows overlap, so a word can be scored in several windows. There are only 4 AMI / 5 ICSI meetings, so the
  meeting CIs are coarse. The window CIs are in the json and are much narrower.
- Nemotron-3's model card lists AMI train + dev in its training data. **Its AMI rows are not held out** and favour
  it. ICSI is held out for every arm.
- The word timing is a model. It uses the emission frame minus a global 400 ms lag, and the P10-P90 lag spread is 190
  to 630 ms. A word near a speaker change can be kept or dropped wrongly by every arm, oracle included. With dilation 0 / 4
  instead of 2 (json `*_d0`, `*_d4`), each arm moves by 0-11 points and the gaps between arms by a few points. The
  ranking never changes: oracle < ours < Nemotron-3 < no filter. At dilation 0, ours trades insertions for deletions
  (AMI primary 60.3: D 34.4, I 9.4).
- *No filter* S is inflated by the alignment. Other people's words aligned against the target's words count as
  substitutions instead of I + D pairs, so read S + I together for that arm.
- The TS-VAD tracks, the ASR and the Nemotron-3 tracks each start cold on the ~20 s window, as in A.1. A live session
  has longer context.
- TurnBench has 16 clips and interpolated word times. The oracle arm still has 17 % insertions there, from the other
  party's speech inside the user's annotated segments and from timing. In 3 clips the track keeps everything, and in
  1 (tb_160) it drops almost all of the user's words, so the per-clip spread is large.

Files: `scripts/research/tswer.py` (stages `asr`, `score`, `turnbench`, `report`; CPU 2 threads, gated, resumable),
`runs/tswer.json` (every number, with its definition and protocol), and intermediates under
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/tswer/`.

## The live two-party suite with the filter on (2026-09-30)

**Question.** The published live WER of single-model mode is 23.2 % (`runs/single_model.json` table.live_69
`single_S`). It counts every word, with no target-speaker filter. What is the WER of the **user's** transcript on the same
sessions when the TS-VAD filter is on?

**Which sessions have a reference.** The live suite has 37 clips and 69 sessions (`runs/e2e_final.json`, prepared under
`scratch/e2e_tsvad/clips`). Only the 16 TurnBench clips have per-speaker transcripts. That gives **32 scored sessions**:
16 mono mixes with both parties in one channel, and 16 with the user's channel alone. **37 sessions are excluded:** the
32 oto sessions (16 clips x mono / user; otoSpeech has no human transcripts), and the 5 AMI mono sessions (the rebuilt
windows in the clip list carry no reference: `has_text` false, `text_user` null). None of the 37 counts toward the
published 23.2 % either, which is 818 / 3525 on exactly these 32 sessions. "All 69" below therefore means these 32.

**Method.** Unchanged from the protocol above (`tswer.py live` / `live_report`). Served streaming RNNT words come from
each session's own wav (masked [70,1] forward), timed by emission frame minus the 400 ms lag, with the mask widened by
±160 ms. The TS-VAD head runs with the clip's stored 5 s print (`prints.json` "5.0", the print the live sessions enrolled
with). The oracle is the user's annotated segments. The user reference is `text_user` and the full reference is
`text_mono`, both passed through `normalize_text`. Totals are checked against `edit_distance` (the `score_record`
scorer). CIs come from 1000 bootstrap resamples by clip, so a clip's two sessions move together.

| row (WER %, [clip CI]; S / D / I words) | all 32 sessions | user channel (16) | mono (16) |
|---|---|---|---|
| ref words: full / user | 3525 / 2704 | 1352 / 1352 | 2173 / 1352 |
| (a) all words, no filter, vs full reference | 22.5 [18.8, 27.2]<br>276 / 445 / 71 | 17.1 [14.5, 20.6]<br>112 / 95 / 24 | 25.8 [20.9, 31.2]<br>164 / 350 / 47 |
| (b) all words vs the user's words | 39.9 [32.0, 53.0]<br>310 / 161 / 608 | 17.1 [14.5, 20.6]<br>112 / 95 / 24 | 62.7 [48.5, 86.6]<br>198 / 66 / 584 |
| (c) **TS-VAD-filtered words vs the user's words** | **34.5** [26.8, 46.3]<br>293 / 398 / 242 | **24.4** [17.0, 36.8]<br>107 / 199 / 24 | **44.6** [35.0, 57.6]<br>186 / 199 / 218 |
| (d) oracle filter vs the user's words | 26.8 [21.4, 34.9]<br>293 / 178 / 253 | 17.1 [14.5, 20.6]<br>112 / 95 / 24 | 36.5 [28.0, 50.2]<br>181 / 83 / 229 |

Paired deltas (clip CI): (c) - (b) is -5.4 [-16.7, +4.8] over all sessions, +7.3 [+0.1, +19.9] on the user channel, and
-18.1 [-36.2, -6.3] on mono. (c) - (d) is +7.7 [-0.5, +19.1] over all sessions.

- **Row (a) vs 23.2 %.** Re-scoring the stored live Pipecat finals of `single_S` with the same references gives exactly
  **23.21 %** (`live_none_full`). The offline re-decode gives 22.5 %, 0.7 points lower. The only difference is the
  decode: the live server finalises segment by segment, while the offline decode is one masked forward over the clip.
  Scoring the live finals against the user's words gives nearly the same numbers as rows (a) and (b): 39.8 / 17.3 / 62.2.
- **One clip dominates the filter's cost.** In tb_160 the track drops 79 of the user's 81 words in both sessions. That
  clip is also named in the caveats above. Without it (15 clips / 30 sessions), (c) is **30.5** [25.7, 37.0] over all
  sessions, 19.7 on the user channel (vs 17.7 unfiltered and oracle), and 41.2 on mono (vs 62.7 unfiltered and 37.6
  oracle). These are in the json as `*_without_tb_160`.
- **Dilation** (json `tsvad_d0` / `d4`). All sessions: 36.1 / 34.5 / 34.9. The ranking is unchanged.
- **Consistency with the TurnBench row above.** The mono sessions are the same 16 clips as that row. Scored with its
  interpolated-word reference (1350 words; `text_user` tokenises "2,000" and "p.m." one word longer), they give none
  62.9 / ours 44.7 / oracle 36.6. The row above has 62.8 / 45.0 / 36.8. S, D and I differ by at most 5 words per arm. The
  cause is the audio: this run reads the prepared 16-bit mono wav that the live session played, while the TurnBench stage
  re-mixes the channels in float.

**Verdict.** On the sessions where the user wants their own transcript out of a two-party mono mix, the filter cuts WER
from 62.7 % to 44.6 % (41.2 % without tb_160). The oracle filter would reach 36.5 %, so most of what remains is ASR error
plus word timing, not attribution. On the user's own channel there is no one to filter out, and the filter only costs
deletions: 17.1 -> 24.4 %, and 17.7 -> 19.7 % without tb_160. Across all 32 sessions, the user's transcript comes out at
34.5 % with the filter on, against 39.9 % without it and 26.8 % with a perfect filter. The published 23.2 % is not
comparable to these numbers, because on the mono sessions its reference also counts the other party's words.

Files: `runs/tswer_live.json` (every arm, S/D/I, clip CIs, excluded sessions, consistency), `scripts/research/tswer.py`
stages `live` / `live_report`, and intermediates in `/Volumes/ExternalSSD/nvidia-audio-models/scratch/tswer_live/`.

## Fix: clean voice prints and anchored adaptation (2026-09-30)

**Bug.** On the user's own channel, with nobody else to filter out, the TS-VAD filter raised the user's WER from 17.1 %
to 24.4 %. Most of that came from tb_160, which lost 79 of the user's 81 words in both sessions.

**Root cause.**
- **tb_160: the stored print mixes two voices.** The 5 s print ([497.0, 502.0] s of the user's channel) was cut
  inside the user's long turn while the other party laughs and talks ("[laughs] Pull you back in", 497.8-503.2 s).
  The served VAD head finds speech on the other party's channel on 62 % of the print's frames. The print is closer to
  the other party (cosine 0.56) than to the user's own speech in the clip (0.50). So the head calls the user "other"
  (P(other) 0.95 on the user's channel). The print picker trusted the Dyadic labels for "the other party is silent",
  and those labels drop some of the other party's speech inside the user's long turns. The same check flags tb_106
  (92 % of frames), tb_107 (52 %), tb_114 (19 %), and tb_20 / tb_157 (they overlap annotated other-party events).
- **tb_138: the print is clean but weak.** It matches only about half of the user's frames (cosine 0.64 to the
  clip's own speech), so the user's words drop in and out.
- **Ruled out:**
  - The threshold: 0.2-0.7 moves user-channel WER by 2 points at most.
  - Channel gain: re-embedding the print at 0.25-4x gain changes nothing.
  - The print's self-similarity: halves 0.78, sub-windows unremarkable, P(target) 0.97 on its own audio. The bad print
    looks normal by every print-only measure.

**Fix.**
1. **Enrolment.** `enrollment.print_is_clean` requires the served VAD to find the user speaking on ≥ 80 % of the
   print's frames and, where a second channel exists, the other party on ≤ 10 %. `e2e_tsvad.py prints --recheck`
   also rejects TurnBench prints that overlap any annotated other-party segment (annotators a and b, bracketed events
   included). It re-draws the prints that fail, with seeded retries. The old print is kept in `prints.json` as
   `"5.0_v1"`. 6 of the 16 TurnBench prints changed; all oto prints passed.
2. **Serving** (`tsvad_stream.TSVADTrack`, on by default). The track now adapts the print, anchored to the enrolled
   one. Every 1 s of speech it accepts as the target alone (P(target) > 0.5 and P(other) < 0.5), it sets the working
   print to 0.8 × the enrolled print + 0.2 × the embedding of the last 5 s of that speech. No event is sent to the
   client. The enrolled print stays the anchor of every update, so the working print cannot drift to another voice.
   A free refresh does drift: +2 to +4 points on mono. A blend weight of 0.6-0.8 gives the same results.
   `--tsvad-refresh-s` > 0 still replaces the adaptation, as before.

The tswer stages now read the served track through `tsvad_stream.track_probs`. The new stage `tsvad_arms` rewrites
only the tsvad arms of the meeting units; the Nemotron-3 arms do not depend on the track.

| WER % | before | after | unfiltered | oracle |
|---|---|---|---|---|
| user channel, 16 sessions (filtered) | 24.41 | **18.05** | 17.09 | 17.09 |
| mono, 16 sessions (filtered) | 44.60 | **40.24** | 62.72 | 36.46 |
| all 32 sessions (filtered) | 34.50 | 29.14 | 39.90 | 26.78 |
| ICSI primary (tWER) | 37.11 | 37.14 | 100.77 | 31.33 |
| AMI primary (tWER) | 61.61 | 62.10 | 106.46 | 47.43 |
| ICSI all speakers | 43.43 | 43.77 | 156.88 | 37.71 |
| AMI all speakers | 75.69 | 76.83 | 148.94 | 54.78 |
| TurnBench row of runs/tswer.json | 44.96 | 40.52 | 62.81 | 36.81 |

- **User channel.** tb_160 goes from 79 deletions to 4 (the unfiltered count), and tb_138 from 28 to 3. The
  remaining +1.0 point over unfiltered is mostly tb_114: its new, clean print still misses half of that user's
  frames (6 → 15 deletions).
- **Meetings.** The meeting prints are unchanged, and the adaptation costs +0.5 on AMI primary. AMI all speakers
  moves +1.1, within its meeting CI [60.9, 90.0].
- **EOT.** The fix changes the TS-VAD columns, so the sessions of `eot_latency.py` were re-dumped into
  `scratch/tswer_fix/eot_dump` and the shipped `vad_head` rule was re-scored. `runs/eot_latency.json` was not
  touched.
  - Calls: p50 881 → 916 ms, p95 1789 → 2020 ms, with false interruptions and misses unchanged (22.9 / 7.3 %).
  - AMI: p50 1330 → 1327 ms, p95 4160 → 4162 ms, false interruptions 10.0 → 10.0 %, misses 33.5 → 34.0 %.
  - Where the calls change: tb_160's first end used to fire early through the others path, as a false interruption,
    because the user was "other". It now waits for the head (2.0 s). tb_157 answers an end it used to miss. tb_20
    misses one. The adaptation moves three ends by 160-480 ms.
  - Split: new prints alone give p50 883 / p95 2024 ms; the adaptation alone gives 921 / 1787 ms.
  - Numbers: `scratch/tswer_fix/eot_shipped_rule.json`.
  - Follow-up: that rule then cut the quickstart clip mid-question over the websocket. The default is now p ≥ 0.99
    after 160 ms, or 640 ms (research/EOT_LATENCY.md, "After the print fix: the quickstart cut").

**Tried and rejected: a "single-voice guard".** The guard kept the user's words while no second voice had been
heard. A print-free second copy of the head took a session print from the first 1.5-3 s of speech, and the guard
latched off once that copy said "another voice".
- It fixes the user channel: 17.3-17.7 %.
- The meeting windows start cold, and the target starts speaking about 4 s in. The session copy hears everyone in the
  room as one voice (P(other) < 0.02 on other speakers), so the guard keeps the first 4 s of other people's words.
  ICSI goes 37.1 → 49-55 %.
- Gating the guard on the speaker-verification cosine (≥ 0.45) and on ≥ 3-6 s of rejected speech still cost ICSI
  +0.9 to +6.7 points and AMI +3 to +6 points. Every setting that spared the meetings missed the user-channel goal.
- On TurnBench, tb_125 (the other party talks alone for 25 s before the user) cannot be told apart from tb_160 by
  any causal signal we measured.

Files: `scratch/tswer_fix/` (diagnostics, the before json files, `prints_v1_backup.json`, the eot dumps).
