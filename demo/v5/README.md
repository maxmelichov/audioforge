# v5 showcase film (LinkedIn)

One command rebuilds everything, from the numbers to both MP4s and their QA reports:

    demo/v5/build.sh

Stills: `film.py --stills [--size 1080x1080]`. Outputs (on the SSD, `$DEMO_OUT` = `/Volumes/ExternalSSD/nvidia-audio-models/demo_out/`): `showcase_v5.mp4` (1920x1080) and
`showcase_v5_square.mp4` (1080x1080, re-laid out by each page, never cropped), their `*.timeline.json`, the mix and its
stems (`v5_full_mix.wav`, `v5_full_stem_{voice,example,music}.wav`), the frames (`v5_frames/full_<W>x<H>/`, with the 2 fps
DOM samples `dom_samples.jsonl`) and the QA reports `v5_qa_full_<W>x<H>.json`.

## Who it is for, and what it may say

`CUSTOMER.md`: engineers and founders on Pipecat / LiveKit, the framework teams, NVIDIA, investors. The film shows only
results measured better than the default stacks, or capabilities they lack; every number is read from `runs/*.json`
(`demo/numbers.json`, `shots/numbers_b.json`) and its source is in the fine print. `ARCH_REFERENCES.md`: the architecture
explainers studied for the build shot and what was taken from them.

## Structure (about 164 s)

1. Today: the default chain (`today_chain`), then a real AMI meeting through Pipecat's default stack on a phone-shaped
   call UI (`today_call`, audible, subtitled, the demo agent's voice at the recorded cut-ins).
2. What we built (`arch`, `shots/arch2.html`): a flat build, one idea per beat: NVIDIA's streaming speech model turns
   voice into words; three small heads answer "Is someone speaking?", "Who is speaking?", "Is the turn over?" with live
   values from a recorded session; NVIDIA's speaker model and Parakeet complete the stack. Sizes in the fine print.
3. Real examples, each narrator line -> the real recording alone (narration and music silent) -> narrator line, with
   subtitles of the speaker's words: interruption (TurnBench call, Pipecat default vs ours, the identical span on both
   sides), words while you speak (TurnBench call, LiveKit default vs ours), a real phone call (otoSpeech, CC-BY-4.0),
   a four-person meeting (AMI IS1008b, live session with the fixed diarization defaults). Clips are chosen by
   `examples.py` for intelligibility (Whisper-small WER after processing: `v5_data/examples_wer.md`).
4. Results in plain numbers (`shots/chart_live.html`, `chart_icsi.html`, `chart_turnbench.html`, `chart_wer.html`): answers when you finish, interrupts less, words while you speak
   (live, the caller's own microphone), unseen meetings, TurnBench detectors, final transcript; the seven-win
   scorecard; the end card.

## How it is built

- `examples.py` (gated, Whisper small): picks the example moments and prepares their audio (EQ, compression, -16 LUFS)
  and subtitles -> `$DEMO_OUT/v5_data/examples.json`, `ex_audio/*.wav`.
- `export.py` (repo venv, no model): recorded E2E sessions, server events, the examples, the room session and the plain
  result numbers (with their runs/*.json paths) -> `$DEMO_OUT/v5_data/data.json`. `shots/export_b.py` ->
  `shots/numbers_b.js` for the remaining B pages.
- `script.md`: the narration, one bullet per line, `{caption|spoken}` for pronunciation; captions show digits.
  `agent_voice.md`: the demo agent's line. Both synthesised by `demo/tts.py` (Qwen3-TTS 1.7B CustomVoice; Ryan with the
  lively instruct for the narrator, Aiden for the agent) and checked by `check_voice.py` (Whisper small, 0 word errors).
- `shots/*.html`: one page per composition; `init(data, params, W, H)` / `render(t, ctx)`, the frame is a pure function of t.
  `shots_def.py` (+ `shots_b.json`): the shot list, segments (`say` / `clip` / `hold`), fine print (short variant for 1:1).
- `film.py` (Playwright venv `/Volumes/afdev/venvs/video`): timeline (shot length follows the narration and the clips,
  snapped to the 100 BPM beat), frames at 30 fps via one headless Chromium, 2 fps DOM samples, mux with ffmpeg -threads 2.
- `audio.py`: narration at -16 dBFS active RMS, example audio at -18 dBFS, the demo agent's voice at the recorded times,
  the numpy music bed (public domain by construction) ducked under the voice and muted inside every example span.
- `qa.py`: picture (text >= 48 px from every edge, no overflow or ellipsis, nothing crossing its panel, no label over the
  3D geometry, <= 60 % flat frame, digits in captions, numbers at their final value, headline holds >= 2 s, type sizes,
  one family and weights 400/700, contrast >= 7:1, motion in every 500 ms) and sound (no narration inside an example span,
  timeline and energy; no music inside example spans; no clicks at boundaries; loudness per line and per clip).
  `shots/stills_b.py`: design checks on single stills of the chart pages.

## Machine rules

Model-loading steps (TTS, Whisper check) go through `scripts/dev/gate.sh`; the TTS on MPS is the only GPU job. One Chromium
instance; each render call stops after `--budget` seconds (exit 3 = run again, frames resume). Frames and renders on the SSD.
