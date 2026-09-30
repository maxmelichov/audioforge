# results_v9.png: the mock-up version

`results_v9.png` (and `_square`, `_notext`, `_square_notext`) follows the user's mock-up. It has four white cards in a
2 × 2 grid. Each card has a title, one grey plain-English line, and one or two paired vertical bar charts: audioforge in
orange against one competitor in grey. Every chart has a y-axis with ticks and a value on each bar. A green bracket joins
the two bars and shows the relative change ("↓ 26%"). Latency is not shown: we lose on it, and the user chose to show
only the results where we win. The architecture image stays `architecture_v8.png`; there is no `architecture_v9`.

Title: "audioforge: higher quality, fewer mistakes". Subtitle: "Standard evaluation against LiveKit and Pipecat models,
same audio for every system."

**How the numbers get on.** Every bar value is an existing `v8/...` key in `redesign/numbers_single.json`
(`export_single.py`, read from `runs/*.json`). The relative changes are new `v9/...` keys. `export_single.py` computes
them from the same rounded values printed on the bars (the `shown` strings), so a reader who divides the two bar labels
gets the printed %. They are never typed in, and the export stops if any of them is not an improvement. The raw-value
change is kept in each entry's `raw` field.
- Lower is better: `100 × (1 − ours / theirs)`.
- Higher is better (VAD F1): `100 × (ours / theirs − 1)`.

**Checks.** `render.py r9` renders the page, copies the finals to `demo_out/images/` (full size) and `demo/images/`
(half size), and checks:
- every number equals its `numbers_single.json` string;
- every bar has a value label, and its height matches its value on its axis;
- each change equals the formula applied to the printed bar labels, and its arrow points the right way;
- every card has a plain line and a footnote, and one change per chart;
- the tWER card does not name LiveKit or Pipecat, and its grey bar reads "no speaker filter";
- the words "latency", "ms" and "p50" do not appear;
- plus the usual type-size, edge, contrast, overlap and clipping checks.

All of these pass for both sizes (`redesign/checks.json`).

| Card | Chart | audioforge | Grey bar | Change | Source |
|---|---|---|---|---|---|
| 1. Turn-taking on calls | False interruptions, % of your turns | 20 (20.2) | LiveKit 27 (26.6) | ↓ 26% | `runs/eot_latency.json`, `table > two_party_user` |
| | Missed turn ends, % of turn ends | 7 (7.3) | LiveKit 23 (22.9) | ↓ 70% | same |
| 2. Turn-taking in meetings | False interruptions | 11 (10.5) | Pipecat 28 (28.0) | ↓ 61% | `runs/eot_latency.json`, `table > ami` |
| | Missed turn ends | 34 (33.5) | Pipecat 45 (44.5) | ↓ 24% | same |
| 3. Your words when others talk | Target-speaker WER, words wrong per 100 of yours | 40 (40.24) | no speaker filter 63 (62.72) | ↓ 37% | `runs/tswer_live.json` @ fe28a9e, `results > mono` |
| 4. Voice activity detection | F1 at threshold 0.5, y-axis 0.8–1.0 | 0.951 (0.9511) | NVIDIA MarbleNet 0.937 (0.9367) | ↑ 1.5% | `runs/vad_auc.json`; `runs/vad_single.json` and `runs/baselines_sd.json` agree |

- **Card 1: Turn-taking on calls.** Plain line: "audioforge cuts you off less and misses the fewest turn ends."
  - False interruptions: how often the agent answers while you are still mid-turn.
  - Missed turn ends: how often it never notices you finished.
  - audioforge runs the shipped `vad_head` rule (160 ms of quiet, p ≥ 0.99, 640 ms fallback). LiveKit runs
    EnglishModel + Silero at its defaults.
  - Footnote: 109 turn ends, user's own channel, 32 two-party calls. Every system is scored on the same turn ends.
- **Card 2: Turn-taking in meetings.** Same two measures on 200 AMI turns. The comparison is against Pipecat
  smart-turn v3.2 + Silero at its defaults. audioforge also beats LiveKit here (LiveKit: 12.5 % false interruptions,
  67.5 % missed turn ends).
- **Card 3: Your words when others talk.** Plain line: "Lower WER — fewer other people's words transcribed as yours."
  - It shows target-speaker WER on 16 calls with both voices mixed into one channel.
  - **Label correction.** The mock-up labels the grey bar "LiveKit". That is not what was measured. The 63 is our own
    speech-to-text with no speaker filter, which is what LiveKit and Pipecat do: neither has a speaker filter. The bar
    is labelled "no speaker filter", and the checks fail if the card names LiveKit or Pipecat.
- **Card 4: Voice activity detection.** Plain line: "Higher F1 — better balance of missed and false speech."
  - F1 at threshold 0.5 on AMI dev (64 × 20 s windows).
  - The y-axis starts at 0.8, as in the mock-up. That makes the 1.5 % gap look larger than it is; the printed values
    and the "↑ 1.5%" are exact.
  - Our head was trained on AMI labels.
- **Footer.** Pipecat 1.12 / LiveKit Agents 1.8 defaults. Turn detectors are replayed without waiting for their
  speech-to-text, which is in their favour. Word accuracy is that of a 115M streaming model: 2.3 % WER on LibriSpeech,
  23.2 % on live calls (LiveKit default 19.3 %).

**Printed % vs raw-value %.** The printed % are worked out from the bar labels, so the picture is consistent with
itself, and they match the mock-up. Worked out from the raw file values, four of them come out slightly different:

| Chart | Printed (from bar labels) | Raw file values |
|---|---|---|
| Calls, false interruptions | 26 % (1 − 20/27) | 24 % (1 − 20.2/26.6) |
| Calls, missed turn ends | 70 % (1 − 7/23) | 68 % (1 − 7.3/22.9) |
| Meetings, false interruptions | 61 % (1 − 11/28) | 63 % (1 − 10.5/28.0) |
| Meetings, missed turn ends | 24 % (1 − 34/45) | 25 % (1 − 33.5/44.5) |
| tWER | 37 % (1 − 40/63) | 36 % (1 − 40.24/62.72) |
| VAD F1 | 1.5 % (0.951/0.937 − 1) | 1.5 % (0.9511/0.9367 − 1) |

Differences from the mock-up:
- The square version puts a legend row above the turn-taking charts in place of names under the bars, and says
  "lower is better" once per card.
- In the square version's single-chart cards, the change sits on the bracket line.

# What the two audioforge images show (v8)

This covers `architecture_v8.png` and `results_v8.png` in `demo/images/` (half size), each with a `_square` version and
a `_notext` version (the same render with the explanatory lines hidden). Full-size copies are in `demo_out/images/`
on the SSD.

Both images describe **single-model mode** (`audioforge-serve --mode single`, the default): one frozen NVIDIA
streaming speech model with 115M parameters, five small heads of ours, and the user's stored 5-second voice sample.
There is no diarizer and no Silero in this mode.

## How the numbers get onto the image

- `demo/images/redesign/export_single.py` reads the run files and writes `numbers_single.json`. The v8 entries are the
  `v8/...` keys. Each entry holds the value, the string shown, the file and the path inside the file. Nothing is typed
  by hand.
- The pages are `redesign/arch_v8.html` and `redesign/results_v8.html`. `render.py v8 r8` renders them with Playwright.
  It then checks the rendered page:
  - every number on the page equals its `numbers_single.json` string;
  - every bar carries its value;
  - every card has a plain-English line, a metric name and a BETTER cue;
  - a card that shows F1 says what F1 means;
  - none of these words appear: "dead air", "cut-in", "first words", "unanswered", "0.6B", "Parakeet", "per call";
  - no text is smaller than 22 px, closer than 64 px to the edge, below 7:1 contrast, overlapping or clipped;
  - no card runs into the footer.

  The results are in `redesign/checks.json`.
- Style, as before: white page, dark ink, one orange for audioforge, grey for every other system, the value printed on
  every bar, and a "← BETTER" or "BETTER →" cue under each set of bars.

## results_v8.png

Title: "audioforge: one 115M model for voice agents". Subtitle: standard metrics against LiveKit, Pipecat and NVIDIA
models, on the same audio for every system in a card.

Every card has a title, one plain line saying what is measured, the metric's standard name and unit, the bars, and a
line saying which audio it was run on. Where audioforge is not best, the card says so in bold.

### 1. Turn-taking on calls

- **What it means.** How long after you stop talking the agent may answer, how often it cuts you off, and how often it
  never notices you finished.
- **Metrics, lower is better for all three.**
  - End-of-turn latency, p50, in ms.
  - False interruptions, in % of your turns.
  - Missed turn ends, in % of turn ends.
- **Numbers.** audioforge 956 ms, 20 %, 7 % (20.2 %, 7.3 % in the file). LiveKit turn detector 567 ms, 27 %, 23 %.
  Pipecat smart-turn v3 2268 ms, 38 %, 25 %.
- **Where we lose.** LiveKit answers sooner (567 vs 956 ms). The card says so: "LiveKit answers sooner; audioforge
  cuts you off least and misses the fewest."
- **Audio.** The user's own channel of 32 two-party calls (16 TurnBench, 16 otoSpeech), 109 reference turn ends,
  the same for every system.
- **Source.** `runs/eot_latency.json`, `table > two_party_user > <rule>`. For audioforge the rule is the shipped
  `vad_head` rule; its key is taken from `selection > fastest_print_fix_goal_no_clip_cut` (commit 6ad219a), and the
  export stops if that key is not the 160 ms / p ≥ 0.99 / 640 ms rule.

### 2. Turn-taking in meetings

- **What it means.** The same three measures in meetings, where other people keep talking after you stop.
- **Numbers.** audioforge 1326 ms, 11 %, 34 % (10.5 %, 33.5 % in the file; the image rounds half away from zero).
  LiveKit 1890 ms, 13 %, 68 %. Pipecat 384 ms, 28 %, 45 %.
- **Where we lose.** Pipecat answers sooner (384 vs 1326 ms), and the card says so.
- **Audio.** 200 AMI dev turns, with the other speakers in the audio, the same for every system.
- **Caveat on the card.** Only audioforge uses your 5 s voice sample; the other two have none.
- **Source.** `runs/eot_latency.json`, `table > ami > <rule>`.

**How the turn-taking cards were measured** (research/EOT_LATENCY.md):
- All three systems were scored by one offline harness on the same audio.
- **End-of-turn latency** = the time from the annotated end of your speech to the system's "turn over", including
  the decision's compute. The agent's own reply time is not included. p50 is over the turn ends that got a "turn over"
  within 6 s.
- **False interruption** = a "turn over" while you are still in your turn, in your speech or in a pause inside it.
- **Missed turn end** = no "turn over" between the end of your speech and 6 s later (or your next turn).
- **The baselines** are Pipecat 1.12 and LiveKit Agents 1.8 at their default settings, each with Silero VAD.
  - Pipecat: smart-turn v3.2 at each VAD stop, with the 3 s fallback.
  - LiveKit: the English turn-detector model on the transcript at each end of speech, with its 0.5 s minimum and 3 s
    fallback.
  - Neither baseline waits for its speech-to-text final, which it would in a live call. Both are therefore measured in
    their favour; the footer says so.
- **audioforge** is the shipped `vad_head` rule (since 6ad219a): our VAD head below 0.4 for at least 160 ms and the
  turn head at 0.99 or more. Otherwise it fires after 640 ms of that silence, or, when someone else holds the floor,
  after 960 ms of your own silence. No Silero.
- The rule was chosen on both corpora, on the dump made after the TS-VAD print fix (fe28a9e): the fastest calls p50
  that keeps calls false interruptions ≤ 22.9 % and misses ≤ 7.3 %, AMI misses ≤ 34.0 % and false interruptions
  ≤ 10.5 %, and does not cut the bundled quickstart clip under any of six input deliveries. Neither corpus is held out.
- The LiveKit and Pipecat rows did not change: they read Silero and the ASR text, which the print fix does not touch.

### 3. Your words when others talk

- **What it means.** Other people's words that land in your transcript count as errors. Lower means a cleaner
  transcript of just you.
- **Metric.** Target-speaker WER (tWER): words wrong per 100 of your words, lower is better.
- **Calls with both voices mixed into one channel** (16 TurnBench calls):
  - audioforge 40;
  - the same streaming words with no speaker filter 63;
  - perfect speaker labels 36. This is drawn as a dashed outline: it is a limit, not a system.
- **Meetings, ICSI** (held out for every system):
  - audioforge 37;
  - NVIDIA Nemotron-3-Diarization 65, with one of its speaker columns bound to you by the same 5 s sample;
  - no speaker filter 101.
- **Sources.** Both files as committed at HEAD (commit fe28a9e, the TS-VAD print fix; `export_single.py`
  `load_committed` reads them with `git show HEAD:`).
  - Calls: `runs/tswer_live.json`, `results > mono > {tsvad_d2, none, oracle_d2} > wer`.
  - Meetings: `runs/tswer.json`, `results > icsi > primary > arms > {tsvad_d2, none, n3_*_d2} > twer`.
- **How it was measured** (research/TSWER.md). The served streaming words are kept where our target-speaker track
  says "you". Each word is timed by the frame where it was emitted, shifted back by the median lag of 400 ms, and the
  track is widened by ±160 ms. The reference is only your own words. tWER = (substitutions + deletions + insertions) /
  your reference words.
- **Nemotron-3's binding.** It is shown with the better of the two binders: TitaNet-L, 65.1 (the speaker head gives
  66.8). AMI is in Nemotron-3's training data, which is why the card uses ICSI.
- **Not shown.** On the user's own channel, where there is no one to filter out, the filter only costs words
  (17.1 → 18.1 %, after the print fix).

### 4. Voice activity detection

- **What it means.** Detecting when anyone is speaking.
- **Metrics.**
  - F1 at threshold 0.5, higher is better. F1 balances missed and false speech; 1 is perfect.
  - Miss rate, lower is better: the % of speech frames missed when each system is set to the same false-alarm rate of
    7.5 %.
- **Numbers.**
  - F1: audioforge 0.951, Silero VAD 0.915, NVIDIA MarbleNet 0.937.
  - Miss rate: 10.6, 13.8 and 12.3 %.
- **Audio.** 64 windows of 20 s from AMI dev meetings, 80 ms frames.
- **Caveat on the card.** Our VAD head was trained on AMI labels; AMI is in domain for it and not for the other two.
- **Sources.**
  - F1: `runs/vad_auc.json`, `<system> > f1_at_0.5`.
  - Miss rate, ours: `runs/vad_single.json`, `eval > ami_dev > L3 > at_fpr0.075`.
  - Miss rate, the others: `runs/baselines_sd.json`, `vad > <system> > sweep`, at the threshold whose false-alarm rate
    is closest to ours. The export stops if they are more than 0.2 points apart.

### 5. Cost

- **What it means.** Compute time for each 160 ms of audio. Under 160 ms keeps up with live speech.
- **Metric.** Compute per 160 ms of audio, p50, in ms, for the whole single-mode engine (every head and the streaming
  words), lower is better. A dashed line marks 160 ms.
- **Numbers.**
  - 29.8 ms on the Mac CPU with 2 threads and 28.7 ms on the Mac GPU (MPS): `runs/mps_115m.json`,
    `engine > {cpu, mps} > chunk_ms_p50`. One run, same machine (Apple M5), the full `--mode single` engine on the
    bundled 16 s clip with its stored print (research/MPS_115M.md). The MPS engine emits the same events as the CPU
    one. It is only 4 % faster because the turn pass and the heads are tiny batch-1 calls; the ASR core alone is
    10.2 vs 17.6 ms. The square image labels the rows "Mac CPU", "Mac GPU", "RTX 5090".
  - 20 ms on an RTX 5090, from PR #1 (`research/GPU_RUN_2026-09-29.md` on that branch). No GPU run file exists in
    `runs/` locally. The value is read from the citation in `runs/stt_latency.json` (`gpu_estimate > note`), and the
    label says "(PR #1)".
- **Line under the bars.** One 115M model + 5 small heads, no Silero, no diarizer.
- **No comparison bar.** The default stacks run Whisper once per utterance, not per chunk, so there is no like-for-like
  number for them.

### Footer

- **Word accuracy.** "Word accuracy is that of a 115M streaming model: 2.3 % WER on clean speech (LibriSpeech; Whisper
  small 2.4 %), 23.2 % on live calls vs 19.3 % for LiveKit's default." This is where we lose, stated once, in the
  footer. Pipecat's default scored 23.5 % on the same calls.
  - Clean speech: `runs/hybrid_asr.json`, LibriSpeech test-clean, 200 utterances.
  - Live calls: `runs/single_model.json`, `table > live_69 > systems`: the 32 live sessions that have transcripts,
    each system through its own framework.
- **Setup.** The same audio for every system in a card; Pipecat 1.12 and LiveKit Agents 1.8 defaults, each with Silero
  VAD, replayed without their speech-to-text wait.
- **Datasets.** TurnBench and otoSpeech calls, AMI and ICSI meetings, LibriSpeech. AMI is in Nemotron-3's training
  data.

## architecture_v8.png

v8 is architecture_v7 with the shipped turn rule drawn in. The rest was checked against `audioforge/server/streams.py`,
`policies.py`, `constants.py` and `cli.py` (`MODES["single"]`) and is unchanged.

- **The dark block** is NVIDIA's frozen streaming FastConformer, drawn as 17 layers. We never change its weights. Audio
  enters on the left in 160 ms chunks (①, the first run).
- **Streaming words.** Layer 17 feeds NVIDIA's own RNNT decoder, which sends words every 160 ms while you talk.
- **VAD head** (33K parameters, layer 4) answers "Speech?" every 80 ms.
- **Language head** (0.92M, layers 8-12) answers "Language?".
- **"Is it you?"** is the target-speaker head (TS-VAD, 0.26M, layer 4). It is conditioned on your voice print, which
  the speaker head makes once from your 5 s sample (layer 4).
- **Your voice track** (②) goes back into the same model at layers 1 and 3 for a second run.
- **Turn head** (0.32M) reads that second run and gives "Turn over?" every 80 ms.
- **The turn decision box** reads "Turn end: VAD quiet ≥ 160 ms + turn head ≥ 0.99". These are the `VadHeadPolicy`
  defaults in `audioforge/server/constants.py` (`VAD_HEAD_SIL_THR` 0.4, `VAD_HEAD_WAIT_MS` (160, 640),
  `POLICY_THETA["vad_head"]` 0.99). The 640 ms fallback and the path for when someone else has the floor are in
  research/EOT_LATENCY.md; they are not drawn.
- **Footer.** "109M frozen NVIDIA encoder · 5 small heads · no Silero · no diarizer". In v7 it said "+ 2.3 MB Silero for
  silence timing". Single mode no longer loads Silero.

## Mismatches with the brief (the files were used)

- **Silero VAD F1** is 0.915 at the 0.5 threshold (`runs/vad_auc.json`, `runs/baselines_sd.json` sweep 0.5). The 0.938
  in the brief is Silero's F1 at threshold 0.1, its best threshold. Every system is shown at 0.5.
- **Nemotron-3 tWER on ICSI** is shown as 65 (the TitaNet-L binder, the better one for NVIDIA). The brief's 67 is the
  speaker-head binder (66.8).
- **Meeting turn-taking rates** are 10.5 % and 33.5 % in `runs/eot_latency.json`; the image shows whole percents,
  rounded half away from zero, so 11 % and 34 %.
- **Mac CPU compute** is 29.8 ms from `runs/mps_115m.json` (the same run as the MPS row). The previous v8 image showed
  31 ms from `runs/latency_budget.json`, an older run of the whole session; the card now uses one run for both Mac rows.
- **Word accuracy line.** "4 points behind Whisper small on calls" would be wrong: Pipecat's default also runs Whisper
  small and scored 23.5 %, worse than our 23.2 %. The footer gives the two measured numbers against LiveKit's default
  (23.2 vs 19.3 %) instead.
- **Head sizes.** "~1.6M" for the five heads could not be sourced from `runs/`. The heads the files do record already
  add up to 1.45M: VAD 32 897, speaker 496 448, language 924 817. Adding TS-VAD 0.26M and turn 0.32M gives about
  2.0M. The cost card therefore says "5 small heads" with no total.

## Earlier images

`architecture_v3`-`v7` and `results_v3`-`v7`, `architecture_single` and `results_single` are kept in this folder as a
record. Do not quote them: they use metric names and session pools replaced by research/METRICS.md.
