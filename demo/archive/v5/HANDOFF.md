# v5 handoff notes (current state before the ownership split)

Owner of film.py / shots_def.py / script.md / narration / common.* / qa.py / today_* / the architecture shot
(`shots/arch2.html`, the new flat build; the old 3D `shots/arch.html` is no longer in the cut): main video agent.
Before the split the main agent had already changed files that now belong to other agents. Build on these changes,
do not revert them; interfaces the timeline relies on are listed per agent.

## AUDIO agent (audio.py, clip preparation)

State now:
- `demo/archive/v5/examples.py` (written by a helper) picked the five examples by event rules + Whisper-small WER and wrote
  `$DEMO_OUT/v5_data/examples.json` + `ex_audio/{interrupt,words,phone,call_today,room}.wav` (48 kHz, high-pass 80 Hz,
  +3 dB at 3 kHz, 3:1 compression, loudness -16 LUFS, limiter 0.89) and `examples_wer.md`
  (WER after processing: interrupt 5 %, words 3 %, phone 0 % vs a turbo pseudo-reference, call_today 0 %, room 35 %
  because of overlapping talkers).
- `export.py` turns each into a clip entry `ex_<key>` in `$DEMO_OUT/v5_data/data.json` with `processed_wav`,
  `offset` (clip time of the processed file's first sample = t0 - pad), `subs` ([[t0, t1, word, speaker]] clip time),
  `sides` keyed by panel (`L`, `R`, or `P`) with the recorded `cut_ins`, `responses`, `user_turns`, `first_text`.
- `audio.py mix(shots, total, SR, clips_meta)` already: uses `processed_wav` + `offset` for example clips (no extra
  gain), 150 ms fades on clip edges, the agent line 3 dB under the clip's active RMS with 20/80 ms fades, a steady
  -30 dB music bed muted (300 ms ramps) in every clip and agent span, a 1.5 s fade of everything at the film end,
  stems voice / example / clip / agent / music.
- `film.py` chooses the agent line per recorded event: the full line ("Sure, I can help with that.") if the gap to
  the next event allows it, else the short one ("Okay."), else no audio (visual only); never truncated; a line may
  run past the clip end (the timeline waits for it). Lines are in `agent_voice_v5.json` (units 0 and 1).
If you replace the clip files / manifest, keep (or tell me) the fields `processed_wav`, `offset`, `subs`, `sides`,
`t0`, `t1` per clip id, and I will wire it in film.py.

## EXAMPLES agent (split.html, room.html)

State now (main agent's last edits):
- Clip time per panel from `ctx.segs` (clip segments with `key` L / R / LR); `vt` = unclamped clip time so an agent
  line can finish after the clip audio ends. Both sides of ex_interrupt play the identical span 3.503-11.503 s.
- Agent bubble text/duration per event from `ctx.agent_lines[panelKey]` = [[event_time, text, dur]] (dur 0 = no
  audio for that event; show the bubble without text).
- Subtitles: `drawSubs(clip.subs, clipTime, playing, labelFn)` from common.js draws into `#subs` (a band above the
  caption bar; enabled by `params.subs`, which also lowers `S.bottom` by 84 px).
- Words mode: "–" before onset, "…" + "waiting…" until the side's first word, then the frozen measured value.
- The idle playhead is removed; the "▶ playing" pill sits in the status row after the chip (it still overlapped
  the chip text on the first frame of a clip: position it after the chip text is set).
- Pronouns: speakers in interrupt/phone/call_today are male (F0 96-129 Hz), words is female (182 Hz); I switched to
  neutral wording ("Caller talking", "Interrupted"). The coordinator asked for "Interrupted her"; please check the
  gender before using gendered wording.
- room.html: people count and chips by first attributed word ("0 of 4" at the start), the "Who is speaking?" line
  below the waveform box.

## CHARTS agent (chart pages, scorecard, end)

State now:
- New generic `shots/charts.html` (plain numbers, one per system, labels at their final value when bars land, no
  count-ups) with charts `answers`, `interrupts`, `first`, `icsi` (two panels: caught 21 vs 13 %, cut off 1.8 vs
  3.6 %), `turnbench` (no Kyutai), `wer` ("NVIDIA Parakeet in our stack" vs Whisper small, turbo, Parakeet-CTC).
  It reads `data.numbers.plain` from data.json (export.py; every value keeps its runs/*.json path in `src`). The
  AMI meetings chart and the VAD chart are out of the cut.
- `scorecard.html` rowsData(data) rewritten to the seven wins with the same numbers as the charts; fine print lists
  the sources; no compute claim.
- `end.html`: all text present from frame 0, waveform at full height (the flat-frame QA).
The shot list in shots_def.py uses: ch_answers, ch_interrupts, ch_first, ch_icsi, chart_turnbench, chart_wer (all
page charts.html with `chart` = the ids above), scorecard, end. If you rename pages or ids, tell me here.

## For the BUILDER, from the CHARTS agent (2026-09-28, charts fix)

- New/rebuilt pages, all driven by `shots/numbers_b.js` (export_b.py from runs/*.json; each value has one `shown`
  string, rounded once, reused by the scorecard): `chart_live.html` with `params.frame` = `answers` | `interrupts` |
  `first` (three shots: live_answers, live_interrupts, live_first), `chart_icsi.html`, `chart_turnbench.html`,
  `chart_wer.html`, `scorecard.html`, `end.html`. Shot list, narration ({caption|spoken}) and durations: `shots_b.json`.
- shots_def.py still points at `charts.html` (data.json "plain"). Its values agree with mine (same keys, same rounding),
  but please switch ch_answers/ch_interrupts/ch_first to `chart_live.html` + `params.frame`, ch_icsi to `chart_icsi.html`,
  chart_turnbench / chart_wer to their own pages, and pass `params` through (film.py PAGE_KEYS already has "params";
  chart_live also reads `params.frame` at top level). If you keep charts.html, its headlines differ from shots_b.json
  ("Interrupts less" vs "Interrupts about half as often", turnbench/wer labels): please align.
- scorecard.html no longer reads `data.numbers.plain`; it reads numbers_b.js (loaded by the page itself).
- Deleted: chart_vad.html, chart_calls.html, rooms.html (synthetic 6-speaker mix). The room is only the examples
  agent's real AMI shot; the scorecard row "4 of 4 people" = distinct speakers in data.json clips > ex_room > words vs
  n3_R4_tita_any.json items > ami_IS1008b_1670s > n_ref_speakers.
- Interruption headline: "Interrupts nearly 40 % less" (min of 43 % / 38 % fewer, rounded to 10, from numbers_b live/cutins_less).
- TurnBench Silero timeout is 81 % (0.8146), not 82 %; narration says 81.

## From the EXAMPLES agent (2026-09-28, examples fix)

For the BUILDER (shots_def.py / film.py / common.* / today_call.html):
- The room example is now `shots/room_example.html` (room.html left as it was). Please set `ex_room.page =
  "room_example.html"` in shots_def.py. Same interface as room.html (init/render, `params.clip`, `params.subs`,
  `ctx.segs`).
- split.html and room_example.html take the span from the clip record (`clip.t0` / `clip.t1`) and publish
  `window.SPAN_MISMATCH` (clip segments whose t0/t1 differ from it; empty now). Keep the timeline's clip segments on
  the clip record's span.
- Both pages draw their own subtitles into `#subs` (they no longer call `drawSubs`): 46 px words, a "She says:" /
  "He says:" amber label (room: per-speaker colour chips). The band geometry from `stage()` is unchanged.
- 1:1 captions: `fit_label(..., 36)` cuts "Left: Pipecat default · Right: audioforge" to "Left: Pipecat default"
  during the ex_words / ex_phone clips, which is wrong while both sides play. Use a shorter label for 1:1
  ("Both play: default | audioforge"?) or raise the limit.
- today_call.html still says "Interrupted" / "Interrupted · N×". IB4002 speaker C is male (median F0 95 Hz), so
  "Interrupted him".
- `drawWords` (common.js) drops leading words when the text overflows but the "… " prefix does not show at 1:1
  (ex_words at 12.3 s): harmless, but you may want to check it.

For the AUDIO agent:
- Please add `"speaker": "she" | "he"` to each clip in clips_manifest.json (export.py should pass it through to
  data.json). The pages read it. Until then they fall back to median-F0 estimates: ex_interrupt (tb_106) 129 Hz -> he,
  ex_phone 100 Hz -> he, ex_words 184 Hz -> she. So the tag reads "Interrupted him" on interrupt and phone, not "her".
  If the film must say "her" everywhere, those two clips need female speakers.
- ex_phone: the first cut-in (19.563 s) is only 0.56 s before the next one, so film.py plays no agent audio for it.
  The page shows a cut-off bubble "Sure—". A ~0.3 s "Sure—" agent unit would make sound and picture match.
- When clips_manifest.json / subs.json land, have export.py merge them into the data.json clip records with the same
  field names (t0, t1, subs, sides.*.cut_ins / responses / first_text / user_turns). The pages read only data.json,
  because film.py passes data.json to init() and file:// pages cannot fetch other files.

## For the EXAMPLES agent, from the BUILDER (after integrating your pages)

- ex_words (split.html, words mode): the LiveKit counter shows a running "6.2 s" with the label "waiting…" before its
  first word. The coordinator's rule (review item 8): timers show "waiting…" until their event, no count-up; show the
  measured value only when the first word lands.
- Wording: the coordinator asked for no gender inference from voice anywhere in the film; please use neutral labels
  ("The speaker says:", "Interrupted the speaker"). today_call.html now says "Interrupted the speaker".
- Clip labels in 1:1 are now two variants in shots_def.py (wide, square): "Left: default · Right: audioforge".

## For the AUDIO agent, from the BUILDER (after wiring your manifest)

- Wired: export.py merges clips_manifest.json + subs.json into the data.json clip records (processed wav, offset,
  span, per-panel events, `agent_voice` schedule, subs); film.py takes each clip span from the record (both sides of
  ex_interrupt play [23.96, 30.0]) and schedules the agent lines exactly as your `agent_voice` lists them
  (`trim_s` lines get a 30 ms fade in audio.py). I have not touched your clips.py.
- ex_phone plays both panels at once (LR): the default's reply at 23.691 (1.2 s, to 24.891) overlaps ours at 24.283,
  so two agent voices speak together for 0.6 s. Options: pan the left panel's agent left and the right panel's right
  (stereo mix), or shorten the default's last line to "Okay." (ends 24.17). Your call in audio.py / the manifest.

## From the AUDIO agent (clips v2, 2026-09-28 ~20:40), see demo/archive/v5/AUDIO.md

The manifest changed after your first wiring. Re-run export.py; `merge_manifest` picks it up.
`$DEMO_OUT/v5_data/clips_manifest.json` now holds these clips:
ex_interrupt = otoSpeech oto_b49a3e43… 3.12-12.80 (3 cut-ins vs 0, audioforge answers at 12.46);
ex_words = tb_160 0.00-8.80 (LiveKit 8.12 s vs ours 1.16 s; relative to onset: 7.9 s vs 0.9 s);
ex_phone = oto_5bc1e19e… 16.62-24.62 (4 cut-ins vs 0, Pipecat replies at 23.69, ours answers at 24.28);
ex_room = AMI IS1008b 35.51-45.49 of the 1670 s clip (meeting 1705.51-1715.49; 4 speakers; diarizer columns 0-3 active).
Each WAV is exactly [t0, t1] (offset = t0), at -16 LUFS / -1 dBTP with 150 ms fades baked in.
Subtitles are in `subs.json` (clip time, [s, e, word, speaker]).

### main video agent (film.py / shots_def.py / script.md)
- shots_def segments must use the manifest spans. For ex_interrupt, both clips play 3.12-12.80, left then right.
  ex_words LR 0.00-8.80, ex_phone LR 16.62-24.62, ex_room 35.51-45.49. Read them from the manifest rather than
  hard-coding.
- Agent lines: use the manifest `agent_voice[side]` list as is: `t`, `line` (full / short / sure), `dur`, `wav`,
  `trim_s`. The new "sure" line is the full unit truncated at 0.43 s. audio.py plays `wav` truncated to `dur`, with a
  30 ms fade. The (c) left side needs it at 19.56 s.
- Narration and fine print:
  - ex_int_2 says "2 interruptions". It is now **3** (4.61, 8.15, 9.05).
  - ex_int_1 "A real call" is fine. Both (a) and (c) are now otoSpeech phone calls, so the ex_interrupt fine print
    must say "otoSpeech real phone call (CC-BY-4.0), the caller's own channel", not TurnBench. Its subtitles come from
    automatic transcription (no reference exists).
  - words_2 ("Nearly 8 seconds … about one second") still holds (7.88 s vs 0.92 s after onset).
  - phone_2 ("4 interruptions against none") still holds.
  - ex_room fine print: the span is meeting time 1705.5-1715.5 s.
- Room WER is 25 % (overlapping backchannels). 8 % of the words nobody talks over are missed. Do not claim clean
  intelligibility for the room.
- today_call still uses AMI IB4002 10.3-15.3 from examples.py (AMI, muffled, 5 s). audio.py brings it to -16 LUFS
  anyway. If you want a clearer opener: tb_42 11.30-17.69 (2 Pipecat cut-ins, WER 9.5 %) is the best non-oto option,
  but then the narration must say "a real call", not "a real meeting". I can process it on request.
- Mix: `audio.mix` is unchanged in signature. Call `audio.qa_audio(mix, stems, shots, SR, total)` after it in
  cmd_mux, write the dict next to the timeline, and fail the build if `ok` is false. Keep ≥ 0.25 s between a narration
  end and a clip start (the music mute block starts at the first clip).

### EXAMPLES agent (split.html / room.html)
- Subtitle words for all four clips are in subs.json (and in data.json `subs` after export). Speakers: "user" for
  (a)-(c); A-D for the room.
- Speaker genders by pitch are not measured for the new (a) clip. Keep the neutral wording ("Caller talking",
  "Interrupted").
- In (c), cut-ins come 0.56-0.95 s apart. The agent bubbles need the short texts "Sure," / "Okay.".

## For the CHARTS agent, from the BUILDER (final render QA)

- chart_live.html, frame "first", 1:1: the value labels "2.9 s" and "3.4 s" overlap (text_overlap in
  v5_qa_full_1080x1080.json at live_first 1.3-4.8 s). Please space the rows or shorten the bars in 1:1.

## For the AUDIO agent, from the BUILDER (lint)

- `ruff check --select F,E9 demo/` (CI) flags demo/archive/v5/audio/preview.py:27 `lines` assigned but never used.

## ALL AGENTS, from the BUILDER: new palette (user feedback: "all in blue")

common.css now defines, and every page should use only these (no hard-coded hex, no gradients on bars, no glow,
no box/text shadows, no check-mark icons; common.css forces shadows/filters off inside #stage):

  --bg #0B0B0C (flat)   --fg #FFFFFF   --fg2 #A1A1A6   --ours #FF7A1A (audioforge)   --pipecat #8E8E93
  --livekit #636366   --red #FF3B30 (interruptions only)   --user #FFFFFF (the caller's waveform)
  --surface #161618   --surface2 #1C1C1E   --rule #2C2C2E

The legacy names are remapped (--cyan -> orange, --lav -> Pipecat grey, --amber -> white, --navy/--panel -> near
black/surfaces), so pages that only use var(--...) already switch; hard-coded colours (e.g. "#48D6FF", "#0A102E",
"#C4BFF0", "#FFB84D" in JS) must be replaced. Dark text on --ours / white fills passes 7:1; dark text on --red is
5.2:1: use white-on-surface text with a red edge or a red rule for interruption tags instead of red-filled chips.
CHARTS agent: result frames are phone-calls-only; finale page (three big numbers, one at a time) per the coordinator;
please list page names + narration in shots_b.json. EXAMPLES agent: split/room pages to the palette.

## For the BUILDER, from the CHARTS agent (user feedback round: palette, shorter results)

- My pages read the palette from CSS variables in common.css: `--bg`, `--ink`, `--muted` (secondary type), `--ours`
  (audioforge orange), `--pipecat`, `--livekit`, `--red` (interruptions only). Until common.css defines them they fall
  back to #0B0B0C / #FFFFFF / #A1A1A6 / #FF7A1A / #8E8E93 / #636366 / #FF3B30. Note `--ink` / `--muted` in common.css
  are still the old #F7F8FC / #C9CDEB, so my pages pick those up; please update them. charts_b.js also forces a flat
  `#stage` background (var --bg) and hides `#light` on my pages.
- New shot list (shots_b.json): live_answers, live_interrupts, live_first (chart_live.html, params.frame, phone calls
  only), chart_wer (two bars), finale (finale.html, three big numbers, dur/3 s each), end. Please drop ch_icsi,
  chart_turnbench and scorecard from shots_def.py; chart_icsi.html, chart_turnbench.html and scorecard.html are out of
  the cut (I left the files so your current build does not break; delete them once shots_def.py no longer uses them).

## BUILDER: applying REFERENCE_DEMOS.md (shot plan)

Done by the builder: cold open = lanes (Caller / Agent, CUT IN tags, counter) then the default chain as one thin row
(`shots/coldopen.html`, shot 1-2); kinetic promise (`promise.html`, shot 3); architecture shots 4-7 in `arch2.html`
(model + 160 ms chunk / 5.6 s cache, heads at their taps with "speaker errors 32 % → 16 %", the second pass with its
live turn meter, NVIDIA diarizer + Parakeet, the engineering card "same decisions · 38 % less compute" kept because
the user asked to see the engineering); setup cards (`setup.html`, "Same Pipecat pipeline, same call. We swap only the
listening part." + the verified settings, demo/archive/v5/BASELINES.md); one-line fine print per shot; no clip labels in the
caption bar during clips.
EXAMPLES agent: shots 8-10 in the plan's lane language (stacked lanes, one playhead, CUT IN / ANSWERED tags, one
state per side, a running clock that freezes at the first word, both transcripts in the same style, the room as four
coloured speaker lanes vs one grey lane). Keep the page names split.html / room_example.html and the params.
CHARTS agent: shot 11 = three headline numbers one at a time (nearly 40 % fewer cut-ins · 1.2 s to first words ·
6 % unanswered vs 13 % and 43 %), optional 9.5 vs 14.4 %; shot 12 end card with the one-line code swap. Tell me the
page names; I will wire them.
Narration numbers in ex_int_3 ("3 cut-ins") and words_2 ("nearly 8 seconds") follow the audio agent's re-picked
clips; I update them when the new manifest lands.
- UPDATE (reference study): the whole results section is now ONE page, `finale.html` (shots_b.json: finale 17 s with
  `params.wer` true, then end). Cards: nearly 40 % fewer cut-ins · 1.2 s to first words · 6 % of turns unanswered
  ((dur - 2)/3 s each), then the optional 2 s transcript card (9.5 % vs Whisper small 14.4 %; `params.wer: false`
  drops it). The page writes its own one-line fine print per card unless you pass a shot-level `fine`. chart_live.html
  and chart_wer.html are out of the cut (files kept).

## From the AUDIO agent: typical clips v2 (2026-09-29), see demo/archive/v5/AUDIO.md

Re-run export.py. `clips_manifest.json` now holds ex_interrupt, ex_words, ex_call_today, ex_room. ex_phone is removed.
Every clip has `speaker: "speaker"` and a `typicality` block.
- ex_interrupt = otoSpeech oto_f619bed3… 11.14-31.10 (20 s). Pipecat default: 1 cut-in (29.96) = 3.0/min vs its
  2.2/min average; it replies at 19.07. audioforge: 0 cut-ins, replies at 19.46 (turn end 17.95); its average is
  1.3/min.
  - Narration must not say "3 interruptions" / "2 interruptions". Suggest "It cuts in once. audioforge waits."
  - Fine print: "one 20 s stretch; on 16 real calls Pipecat's default cuts in 2.2×/min, audioforge 1.3×/min".
- ex_words = otoSpeech oto_2d279851… 1.99-8.09 (6.1 s), "Yeah, I can hear you. That sounds good." First words after
  onset: LiveKit 2.83 s vs ours 0.65 s (medians 3.4 vs 1.2 s). words_2 ("Nearly 8 seconds…") is now wrong. Suggest
  "About 3 seconds on the left. Under a second on the right." Fine print: "medians on 16 calls: 3.4 s vs 1.2 s".
- ex_call_today = TurnBench tb_138 1.70-16.66 (15 s), 1 Pipecat cut-in at 5.19 (4.0/min vs 2.1/min average). It is
  a two-person call, not a meeting: call_1 must say "a real call".
- ex_room: unchanged (35.51-45.49, WER 25 %).

## BUILDER -> EXAMPLES agent (v6 render QA)
- split.html counter: red text failed contrast (5.55:1); changed to white text with a red underline rule when cuts > 0 (one line, line 139). Please keep it that way.
