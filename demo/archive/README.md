# Showcase video (LinkedIn)

## Post images

`demo/archive/images/architecture.png` and `results.png` (plus `_square`) are the stills for the LinkedIn post. They are
rendered from `demo/images/redesign/arch.html` (layout A1, flow) and `results.html` (layout B1, grid) by
`demo/images/redesign/render.py`; every results number comes from `demo/v5/shots/numbers_b.json` and is checked on
render. The design, the references it follows and the alternate layouts are described in `demo/images/DESIGN_NOTES.md`.
Engineering details behind the architecture image (no longer printed on it): FastConformer, 109 M encoder,
cache-aware (70 frames), log-mel 80 bins, ×8 subsampling, RNNT decoder · heads 33 K / 0.5 M / 0.32 M (GRU) · turn pass
conditioned at blocks 1 and 3 on the diarizer's (Nemotron-3) speaker activity · column-major weights, cached projections
and RNNT joint.

## v5 (current): HTML/Playwright film, real audible examples, the model built on screen

See `demo/archive/v5/README.md`; rebuild with `demo/archive/v5/build.sh`. The PIL renderers below (v1-v4) are kept for reference.

## v4: the product-spot cut with graphs and the how-it-works flow, 90-120 s

Renderer `demo/archive/render_v4.py`, narration `demo/archive/script_v4.md` (15 lines), 120.0 s in 16:9 (`showcase_v4.mp4`) and 1:1 (`showcase_v4_square.mp4`). Voice: Ryan with the lively instruct at 1.0x. Both Ryan and Uncle_Fu were synthesised on the real script and transcribed with Whisper small (numbers spelled out, `audio forge` joined): Ryan 7 / 247 words wrong (2.8 %), Uncle_Fu 16 / 247 (6.5 %); neither reached zero, so the deeper voice with fewer errors was kept. Rebuild: `demo_out/v4_final.sh` (gated TTS, then `render_v4.py --seg i --nseg 3` and `--mux` per size). Same machine rules and data as v3 below; four graphs read `demo/archive/numbers.json` (WER rows from runs/final_asr.json + runs/hybrid_asr.json, turn rows from runs/baselines_turn*.json + runs/improve_115m.json, product rows from runs/e2e_final.json, compute from runs/e2e_final.json + runs/perf.json).

## v3: the product-spot cut, 60-90 s

Renderer `demo/archive/render_v3.py`, narration `demo/archive/script_v3.md` (one plain sentence per shot, Dylan voice at 1.06x).
The "footage" is a minimal call window replaying the recorded E2E sessions: the framework default stack on the
left, ours on the right, same call audio; the user's bubble shows a live waveform and the words as they arrive, the
agent's reply is a bubble that appears at the recorded response time (a red edge when it pops while the user is still
speaking). Seven short phrases on screen, each with its test basis as fine print. Rebuild:

    PYTHONPATH=. scripts/dev/gate.sh /Volumes/afdev/venvs/tts/bin/python demo/archive/tts.py synth --script demo/archive/script_v3.md --out narration_v3 --speaker Dylan --tempo 1.06
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python demo/archive/render_v3.py --size 1920x1080 --out $DEMO_OUT/showcase_v3.mp4
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python demo/archive/render_v3.py --size 1080x1080 --out $DEMO_OUT/showcase_v3_square.mp4

(`--seg i --nseg n` then `--mux` splits a render into short calls; `--storyboard DIR --dry` writes one still per shot.)
Music: a pad and a soft beat generated in numpy inside `render_v3.py` (`music_bed`), no samples, public domain by
construction; call audio -9 dB, music -22 dB, both ducked under the voice. Data: `runs/e2e_final.json` per-clip
records (cut-in and response times, first-text latency) for the default stacks and ours, `scratch/e2e_tsvad/runs`
for the served voice-print session, `demo/events/*.json` for our live words (streaming partials at word granularity,
split tokens re-joined against the system dictionary and the clip's TDT rewrite), `demo/clips/*.mono.wav` for the audio.
The baseline's words are the reference transcript shown at its measured arrival time (the E2E session texts were not
retained); the end card says so.

`demo/archive/render_v2.py` (light, split-screen, charts) and the v1 pipeline below remain for the longer cut.

## v4 fixes (bug pass over showcase_v3.mp4 at 2 fps, frames in demo_out/bugpass_v3/)

| # | defect in v3 (video time) | fix in v4 (renderer / shot) |
|---|---|---|
| 1 | 0–4.8 s title: the call window is 60 % empty; no wordmark until 0.6 s | title shot: wordmark and subtitle scale in from frame 0, chips ease in, window sized to its content |
| 2 | 4.8–8.8 s "two systems": both panels show only typing dots for 4 s (the clip starts 3.4 s before speech); 10–14.4 s no narration (bar padding) | shot offset 3.4 s so speech starts within 0.5 s; shots are narration-fitted (grow by whole bars only when a sentence needs it) |
| 3 | 19.0–21.6 s: our transcript shows the split token "me ssed up" (the video segments were rendered before the token-mending patch; only the mux was redone) | word-granular partials + dictionary/TDT-vocabulary mending in render_v2.Events; the v4 render is one pass |
| 4 | 21.6–24.4 s dead air: the caption "She stops" plays 2.4 s before she actually stops (offset 10.5 s) | offset 12.0 s: she stops 1.3 s into the shot, "3.2 s" / "1.6 s" labels appear beside the bubbles when the replies land |
| 5 | 38.4–48.0 s room: the reply never appears (turn end at clip 15.0 s = video 53.4 s, after the cut); the phrase fades while the caption is still up | offset 6.0 s, 4 bars: the bound-speaker reply lands inside the shot; phrase/fine print fade only in the last 0.4 s |
| 6 | 48.0–55.2 s "one model": static for 6 s after the four lines draw | replaced by the 20 s "how it works" flow: live chunk conveyor, encoder pulse, four heads with live widgets (VAD lamp, speaker map, turn meter to threshold, words), eight diarizer lanes, TDT rewrite |
| 7 | 55.2–62.4 s voice print: the "with print" side never replies on screen (reply at clip 17.7 s = video 62.9 s) | offset 12.4 s: left cut-in at 1.8 s in, right reply at 5.3 s in, both with their dead-air labels |
| 8 | 62.4–76.8 s honest: 14.4 s on one static window (narration 11.4 s + bar padding) | shorter sentence (8 s) and a 3-bar shot |
| 9 | header timers "4.5 s" in #606C80 on the card: 3.1:1 contrast | timers use MUTED (>= 7:1 on the card) |
| 10 | typing-dot glyphs ●○ fell back to boxes in an early build; user-turn label overlapped the waveform in v2 | dots drawn as circles; labels sit above the band |
| 11 | flat grey palette, no motion between cuts | navy → indigo gradient with a drifting light, amber user / cyan ours / red cut-ins, 300 ms slide with a light seam on every cut, bubbles scale in, bars grow, numbers count up |
| 12 | audio: 0.4 s release on the duck could pump under fast narration | 0.8 s release, 100 ms envelope, music -21 dB |

Live clips: the v3/v4 split screens replay stored E2E records (no live session), so machine load cannot affect them;
the two served TS-VAD recordings (demo/events/*_tsvad.json) were made at server RTF 0.26 on a quiet machine.

## v1 / v2 pipeline

One command rebuilds the v1 set, from the numbers to the final MP4s:

    PYTHONPATH=. .venv/bin/python demo/archive/build.py

Outputs go to `$DEMO_OUT` (default `/Volumes/ExternalSSD/nvidia-audio-models/demo_out/`, gitignored path on the
external SSD; nothing large is written to the repo or the internal disk):

| file | what |
|---|---|
| `showcase.mp4` (+ `_720p`) | main cut, 16:9, 60-90 s, H.264 + AAC, captions burned in (LinkedIn autoplays muted) |
| `showcase_square.mp4` | the same cut in 1:1 for the feed |
| `showcase_full.mp4` (+ `_720p`) | full version, 3-5 min |
| `narration_{short,full}.{wav,json}` | the voice-over and its per-sentence timings |
| `*.timeline.json` | scene boundaries of each render |

Kept in the repo: the scripts here, `numbers.json`, the two scripts, `linkedin_post.md`, and the small poster PNGs in
`demo/archive/stills/`. `demo/clips/` (the audio the server heard) and `demo/events/` (the recorded server events) are
gitignored intermediates that the build recreates.

## What each step does (all idempotent; re-run any one alone)

1. `demo/archive/extract_numbers.py` -> `demo/archive/numbers.json`. Every number shown on screen is read from `runs/*.json` with its
   JSON path; a missing key fails the build, so a stale figure cannot ship. The renderer prints the source file in
   the corner of every numbers card.
2. `demo/archive/tts.py synth` -> the narration. Qwen3-TTS-12Hz-1.7B-CustomVoice (Apache-2.0) on MPS, one wav per script
   bullet, joined with 0.35 s / 0.7 s gaps. Scripts: `script_short.md` (60-90 s) and `script_full.md` (3-5 min); one
   bullet = one synthesized unit = one caption. `demo/archive/tts.py candidates` synthesises the first ~20 s of the script
   with every male voice and writes `voice_candidates/metrics.json` (median F0, pace, clipping) so the voice is
   chosen by measurement; `--speaker` / `DEMO_SPEAKER` picks it (Dylan: the lowest median F0 of the five male
   voices, 109 Hz, and 0 % WER on the sample). The instruct prompt asks for a deep, calm documentary narrator; the
   voice's natural pace (~0.10 s per character) is raised 1.12x with ffmpeg's pitch-preserving `atempo`
   (`--tempo`), and a unit delivered slower than `--max-spc` seconds per character is re-sampled up to `--retries`
   times, keeping the shortest take (the model occasionally drawls a line). Unit wavs are cached by text hash, so
   editing one sentence re-synthesises only that sentence.
3. `audioforge.serve` + `demo/archive/record_events.py` -> `demo/events/*.json`. The real server (`stage1_served.afm` +
   Nemotron-3 diarizer + Silero branch + Parakeet-TDT v3 final pass + AmberNet language ID, 2 threads) is started,
   each clip in `demo/clips/` is streamed at 1x in 20 ms frames, and every message (`frame`, `partial`, `turn_end`,
   `final`, `enrolled`, `language`, `stats`) is stored with its arrival time. Nothing in the live scenes is simulated.
4. `demo/archive/render.py` -> the MP4s. PIL frames piped to ffmpeg (libx264, crf 20, 30 fps, `-threads 2`), audio mixed in
   numpy (narration at -16 dBFS, clip audio ducked under it, a generated ambient pad at -30 dB). Rendered in
   segments (`--seg i --nseg n`, then `--mux`) so every call stays short. `--preview T` writes one frame as PNG;
   `--dry` previews layouts before the narration exists.

## Machine rules baked into the build

- Every model launch (TTS, server) and every render goes through `scripts/dev/gate.sh` (waits for a quiet machine,
  2 threads). The TTS on MPS is the only GPU job and runs alone; `--budget` bounds every call under 10 minutes and
  the per-unit cache makes it resumable (exit 3 = run again, which `build.py` does).
- The TTS lives in its own venv on an APFS volume (`TTS_PY`, default `/Volumes/afdev/venvs/tts/bin/python`; exFAT
  cannot hold a venv): `uv venv --python 3.12 /Volumes/afdev/venvs/tts && uv pip install --python
  /Volumes/afdev/venvs/tts/bin/python qwen-tts soundfile`. The model is read from `~/.cache/huggingface` (offline).
- `build.py` refuses to start the server while a training job is running.

## Style brief

Dark background (`#0B0D12`), Avenir Next for text and Menlo for numbers/paths, one accent (amber) for decisions, blue
for audio and "ours", red for framework defaults, green for the offline final pass. Large type, one idea per card,
captions in a rounded black box at the bottom, no stock footage, no fade-in on the first frame (the poster frame is a
full composition). The live view shows the scrolling waveform, the VAD row, four diarizer speaker rows with the primary
speaker underlined, markers for `agent TTS end`, `user enrolled`, `language` and `turn end`, an amber flash and the
dead-air counter at every `turn_end`, the live partial and the last two finals (the TDT v3 rewrite replaces the
streaming final), and a status line with the audio time, frame lag, session RTF and policy.

## Claims policy

Every sentence of the narration is checked against `research/FINAL_REPORT.md` (scorecard and sections 1-8),
`research/IMPROVE_115M.md` A.2 (the TS-VAD turn rows) and `research/E2E_FINAL.md` (Pipecat / LiveKit). Caveats stay
in the script: "on AMI at matched false cutoffs", "held-out ICSI", "with a 5 s voice print", "offline, not served yet",
"VAP wins on latency", "a silence timeout is as good or better on floor-open ends".
