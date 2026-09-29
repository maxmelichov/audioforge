# v5 audio: example clips, processing, mix rules

Owner: AUDIO agent. Code: `demo/v5/audio/clips.py` (clip choice, processing, WER, manifest, subtitles),
`demo/v5/audio.py` (the film mix + `qa_audio`), `demo/v5/audio/preview.py` (audio-only preview of the four examples).
Outputs on the SSD (`$DEMO_OUT/v5_data/`): `clips/ex_{interrupt,words,phone,room}.wav`, `clips_manifest.json`,
`subs.json`, `clips/intelligibility.md` (all 97 scored candidate spans), `clips/preview_mix.wav` + `preview_qa.json`.

## Chosen clips (v2: typical spans, 2026-09-29)

v1 picked the most dramatic spans: Pipecat cutting in 3 times in 9.7 s, about 18 per minute, when its measured average
on real calls is 2.2 per minute. It read as rigged. v2 picks spans where the default behaves as it does on average.
The processing chain, the WER bar (≤ 15 %, Whisper small through gate.sh, before = raw cut, after = processed clip),
neutral content, and the manifest schema are unchanged. Each clip also carries `speaker: "speaker"` (neutral) and a
`typicality` block. Averages and medians come from `runs/e2e_final.json` › table (16 calls per set, user channel).
Event times come from per_clip. Both sides of a split play the identical span.

| clip | source | span (clip s) | dur | typicality: in-span vs average | recorded events in span | ref words | WER before → after | loudness raw → processed |
|---|---|---|---|---|---|---|---|---|
| ex_interrupt (Pipecat default vs audioforge) | otoSpeech `oto_f619bed3…` caller channel | 11.14–31.10 | 20.0 s | default **1 cut-in = 3.0/min** vs its average **2.2/min** on otoSpeech (2.1 on TurnBench); audioforge 0 in span, average 1.3/min | default replies at 19.07, cuts in at 29.96; audioforge replies at 19.46 (turn ended 17.95) | 44 | 0.0 → 6.8 % | -18.5 → -16.1 LUFS, -1.4 dBTP |
| ex_words (LiveKit default vs audioforge) | otoSpeech `oto_2d279851…` caller channel | 1.99–8.09 | 6.1 s | first words after onset: LiveKit **2.83 s** (median 3.38 s), audioforge **0.65 s** (median 1.18 s); gap **2.2 s** = the median gap 2.2 s | onset 2.75; LiveKit first text 5.58, audioforge 3.40 | 8 | 0.0 → 0.0 % | -18.6 → -16.1 |
| ex_call_today (opening, Pipecat default only) | TurnBench `tb_138` user channel | 1.70–16.66 | 15.0 s | default **1 cut-in = 4.0/min** vs average **2.1/min** on TurnBench | cut-in at 5.19 | 37 | 0.0 → 0.0 % | -33.8 → -16.1 |
| ex_room (unchanged) | AMI IS1008b (meeting 1705.51–1715.49 s) | 35.51–45.49 | 10.0 s | n/a; 4 speakers, 4 diarizer columns lit | – | 48 | 29.2 → **25.0 %** (8 % of words nobody talks over are missed) | -20.2 → -16.0 |

Texts: interrupt "…good talking to you and always a pleasure. And again, I look forward to talking to you again in the
future. So it's always a pleasant conversation. Thank you. I wish you [the same] as well. Thank you, sir. You are,
too. I wish you…" · words "Yeah, I can hear you. That sounds good." · call_today "I've always been a little bit of a
foodie. Yeah, I used to. Not anymore. In the beginning, when we first opened, I did, but at this point, because I'm
the one doing the research and development…" · room as before.

Honest caveats:
- The interrupt span shows audioforge at 0 cut-ins. Its own average on these calls is 1.3/min, against 2.2 for the
  default. The manifest states both averages. A span "within 2× of the default average" still favours us, so the
  fine print should give the averages.
- 1 cut-in in a 15-20 s span is 3-4 per minute, the closest a short span can get to 2.2/min (1 cut-in per 27 s).
- In the words clip, audioforge's 0.65 s is faster than its 1.18 s median, and LiveKit's 2.83 s is faster than its
  3.38 s median. Only the gap matches the median. No span met the WER bar with both values closer: oto_7342
  (3.41 / 1.29 s) and oto_5f91 (2.73 / 1.11 s) scored 33 % and > 15 % WER on sparse speech. tb_22 (3.13 / 0.69 s)
  is about deaths.
- otoSpeech clips have no transcript. Their reference is Whisper large-v3-turbo on the whole raw channel, and their
  subtitles are turbo word timings.
- The phone example (ex_phone) is out of the cut and out of the manifest. Its 4 cut-ins in 8 s were not typical.
- Rejected on content: tb_22 (deaths, abortion politics), tb_128 (drugs), tb_112 ("they just suck"), tb_42 ("stinky
  vanlifers").

v1 choices, superseded: interrupt oto_b49a 3.12-12.80 (3 cut-ins in 9.7 s), words tb_160 (8.1 vs 1.2 s, an outlier),
phone oto_5bc1 16.62-24.62 (4 cut-ins in 8 s). All candidate rows are in `clips/intelligibility.md`.

## Processing chain (per clip, ffmpeg, `clips.py:process`)

The file holds exactly [t0, t1] of the source, so offset = t0. The chain is: `aresample=48000`, high-pass 80 Hz (2
poles), presence +3 dB (one octave around 2.83 kHz, which covers 2-4 kHz), compressor 2:1 above -22 dBFS with a 6 dB
knee (15/200 ms), then two-pass loudnorm to -16 LUFS. For every chosen clip, loudnorm's linear pass 2 refused because
of the true peak and would have switched to its dynamic (AGC) mode. So the pass-1 measurement is applied as a plain
linear gain (iterated to ±0.2 LU) into the limiter. The limiter is a true-peak limiter at -1 dBTP: `alimiter` at 4×
oversampling (192 kHz, limit 0.85). Last come 150 ms fades in and out. Results: -16.0 to -16.2 LUFS, -1.4 dBTP.
Noise floor after gain: (b) -45 dBFS (TurnBench user channel, +19.8 dB of gain); the others are gated or below
-78 dBFS. (d) is -29 dBFS, which is room tone.

## Agent voice (manifest `agent_voice`, per side)

Lines come from `agent_voice_v5.json`: "Sure, I can help with that." (1.20 s), "Okay." (0.48 s), and "Sure," (0.43 s).
"Sure," is the full reply ended at its first word boundary, the comma pause found by energy, so no word is cut.
Each event gets the longest line that ends 0.1 s before the next recorded event, or no line if even "Sure," does not
fit (none of the chosen clips needed that). Lines sit 3 dB below the clip, at -19 LUFS integrated for the whole unit.
On the rendered stems they measure 2.7-6 dB below the clip's speech RMS.
- ex_interrupt left: 19.07 full, 29.96 full (0.06 s past the span end). Right: 19.46 full.
- ex_call_today: 5.19 full.
- ex_words: no agent line (words mode).

## Mix rules (`audio.py`) and QA

`mix(shots, total, SR, clips_meta)` keeps the same interface and stems (voice, clip, agent, example, music).
- Every narration unit is normalised to -16 LUFS integrated (ffmpeg ebur128), with a 5/80 ms raised-cosine fade.
- A clip that lies inside a manifest span is read from the manifest WAV, which is already at -16 LUFS with its fades
  baked in. A fade is added only at an inner edge. Any other clip audio is measured and brought to -16 LUFS, with
  150 ms fades.
- The agent voice is 3 dB below its clip, with 10/30 ms fades.
- Music is one constant bed at -34 LUFS, with an equal-power chord crossfade (the old one swelled every 8 bars) and no
  ducking between lines. It is muted across each whole example block, from the first clip start to the last clip or
  agent end of the shot, including the narration between the left and right clips. The mute uses 0.4 s raised-cosine
  ramps outside the block.
- End: a 1.5 s raised-cosine fade, then 0.1 s of digital silence, so the last frame is silent.
- One lookahead limiter at -1.5 dBFS sample peak runs on the sum. The same gain goes on every stem, so the stems still
  add up to the mix.

`qa_audio(mix, stems, shots, SR, total)` returns `{"ok", "fail": [...], ...}`. It checks:
- integrated loudness and true peak of the mix;
- -16 ± 1.5 LUFS for each narration line and each clip;
- the agent level against its clip;
- no narration inside a clip and no music inside an example block;
- music spread under 3 dB, measured in 3 s windows between blocks;
- a silent last frame;
- a transient (click) detector at every edit boundary, on the stem edited there, and on the mix for transients the sum
  or the limiter created;
- a hard-cut detector (a drop from above -30 dBFS to below -80 dBFS within 10 ms).

| run | result |
|---|---|
| `preview.py` v2 (opening call, interrupt L+R, words, room + agent lines, end card) | ok; mix -16.6 LUFS, -1.4 dBTP; clips -16.0 to -16.4 LUFS; music -33.5 to -32.1 dB; 0 clicks, 0 hard cuts; last frame 0.0 |
| current `film.timeline("full")` (169.2 s, as wired at 20:16) | ok after the manifest span update (before it: the ex_phone sub-span measured -17.6 LUFS); narration -16.0 to -16.9 LUFS; 0 clicks, 0 hard cuts; music spread 1.4 dB |
