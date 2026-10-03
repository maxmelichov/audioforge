# Audit 001: baseline fairness and published numbers (2026-10-03)

Read-only audit of `research/FINAL_COMPARE.md`, `runs/final_compare.json`, `README.md`, the `compare_*.png`
pipeline and the public docs. Nothing outside `plans/audit/` was edited. No model was run; the only computations
are text re-scoring of the cached hypotheses on the SSD (`scratch/final_compare/asr/*.jsonl`,
`scratch/finallat/*.jsonl`) with a simple normaliser, and the number diff of `plans/audit/fairness_001.py`.

Verdict key: **fair** / **disadvantages baseline** / **disadvantages us**.

## Findings ranked by impact

1. **README "A bigger core: `--core 0.6b`" table is entirely stale** (README.md:179-198): first-pass dev-split
   numbers, mostly heads v0.2. Every row disagrees with the test json (AMI / ICSI WER 24.4 / 27.3 and 11.2 / 14.4
   vs 16.1 / 18.4 and 7.9 / 10.3; tWER, EER, VAD F1, turn rows, cost, memory). The prose under it ("heads v0.2",
   "answers an agent 80 ms later" (now 52 ms), "misses more call ends") is stale too. Public, contradicts the
   Results section 100 lines above.
2. **Final-text latency puts Parakeet-TDT on CPU (2 threads) against ours on the Mac GPU, and compare_asr.png
   shows it with no device label** (341 vs 49 / 25 ms, "−234" gap). Same-CPU throughput of Parakeet-TDT and our
   0.6B is identical (RTFx ~20), so the 7x gap is mostly device. Lower-bound estimate on MPS ≈ 150 ms, likely
   50-150 ms; the 0.6B's lead could shrink to ~1-2x. Parakeet would also be the fastest baseline, not Whisper
   turbo. Re-run Parakeet on MPS (~40 s) before publishing that chart again. The table also omits WER at the timed
   point: the 0.6B's flushed turn texts are about equal to Parakeet's (8.97 vs 8.43, simple normaliser), the 115M's
   are much worse (15.1).
3. **AMI turn rows: ours is enrolled with the target's voice print, the baselines are not, and this is not stated.**
   Our "others" path (user silent while others talk) is what the Silero-based baselines lack, and it drives their
   72-86 % misses. Real capability, but presented as like-for-like; no unenrolled run of ours exists.
4. **"0.6B best on the live user channel" depends on the verbatim reference convention.** TurnBench references keep
   repetitions and false starts; Whisper drops them. Collapsing them in reference and hypothesis moves Whisper turbo
   from 8.6 to 6.2 and large-v3 from 9.0 to 7.0, against the 0.6B's 7.0 to 6.7 (simple normaliser). The claim holds
   against Parakeet-TDT, not against Whisper. AMI / ICSI orderings are unaffected.
5. **The AMI speech-detection "win" rests on F1 at 0.5**, where our heads are trained on the label convention. On
   the threshold-free metrics in the same json, MarbleNet ties ours on AUC (0.967 vs 0.966 / 0.967, CI of the
   difference spans 0) and misses less speech (9.7 vs 12.5 %). compare_vad.png does show the miss chart honestly;
   README's prose does not. ICSI is a clean win on every metric.
6. **compare_asr.png bars show the opt-in 1.12 s final, the README table right above it shows the default 160 ms
   pass** (0.6B ICSI 8.4 vs 10.3; live = all words 10.7 vs user channel 5.7 in the table). The 1.12 s mode costs
   real-time streams (115M MPS 4 → 3, 0.6B MPS 3 → 1, dual_rate.json), which no image or cost line shows.
7. **Speaker tracking: diarizers are scored through our own causal binder.** With the oracle column (already in the
   json) Nemotron-3 on AMI reaches DER 39.7 / F1 0.795 vs our 37.3 / 0.811 (115M), against the published 62.9 /
   0.676. Ours still win, but most of the AMI margin is the binder. compare_spk.png / README also mix word sources:
   pyannote bars use 115M words (ICSI 73.6) while ours and Nemotron use 0.6B words (pyannote on 0.6B words 69.0);
   README uses 69.0 for ICSI but the 115M-words 77.1 for AMI.
8. **LiveKit is fed our 115M transcript** (16 % WER on AMI) instead of a production STT; stated, impact not
   discussed. Pipecat's STT wait is not modelled (flatters Pipecat, stated). Parakeet-EOU's restart after each
   `<EOU>` is a correct fix that helps it.
9. **Smaller handicaps of Whisper**: Whisper small uses beam 1 (its faster-whisper / Pipecat front end defaults to
   beam 5); the live sessions (all > 30 s) go through long-form decoding without temperature fallback; Whisper small
   and Parakeet get 2 CPU threads. Large-v3 / turbo greedy = transformers default and leaderboard practice (fair;
   LS numbers match published ones). "Whisper small (LiveKit default STT)" has no source: LiveKit 1.8.3 ships no
   Whisper default; Pipecat's local default is distil-medium.en.
10. **Head versions and provenance.** README.md:79 says the 0.6B uses heads v0.3, but its turn numbers are v0.4;
    README.md:105 "1497 ms" is the v0.3 value (json 1498). `final_compare.py` defaults to the v0.3 115M afm, the
    v0.2 0.6B afm and the old LID heads; the published rows need three env vars that the Reproduce block omits.
    `params` were counted on stage1_served_v3 / served_0p6b_v0.2. README:13/18 head sizes / layers (VAD "layer 4,
    33K", LID "0.92M") and architecture_v10.png ("VAD · block 4") predate the speech head and LID v2.
11. **Stale public docs** (143 lines flagged by the script): docs/CONFIGURATION.md, docs/MODELS.md ("ICSI dev F1"),
    docs/PROTOCOL.md:460, docs/SERVER_INTERNALS.md:30, research/METRICS.md scorecard (all first pass),
    research/README.md:34; demo/images/EXPLAINER.md documents only results_v8-v10 / architecture_v8-v9 (92.2 %,
    0.951, 956, 1326) and not compare_*.png; arch_vs.html FOCUS rows hard-code first-pass numbers (render the
    untracked focus_*.png, not referenced publicly).
12. **FINAL_COMPARE.md itself is clean**: ~1190 headline cells and CIs match the json. Five trivial mismatches
    (0.6B params 622.6 vs 622.5; two CI bounds rounded from x.xxx5; README 1497 vs 1498; README "~110 ms" vs 105)
    and 10 numbers with no json source (previous-LID-head row 90.9 / 87.6 / 97.8 / 95.5, LID head sizes 2.4 / 2.9 M,
    "95.0 % / 354 ms / 3.1 %" from FIXALL, flush 21 / 39 and 45 / 60 ms). README's "+20-27 ms at p95" dual-rate cost
    holds for 115M CPU and 0.6B MPS only (115M MPS +8.8, 0.6B CPU +92.8). The compare_*.png images are current: no
    image key changed in numbers_final.json since each PNG was committed, and numbers_final.json matches the run
    json for every entry.

## (A) Baseline fairness, system by system

### Words (WER)

Common to every system (`final_compare.py asr_set`, `score_asr`): one shared list of arrays per set, `assert sr ==
16000` on every file, the same reference strings, the same Whisper English normaliser, the same bootstrap groups.
LibriSpeech: seeded sample of 300 (seed 0) over the whole split; AMI / ICSI: one cached `.npz` of 200 segments per
set, built once and read by every system; FLEURS: the 150-row manifest, asserted. **Same audio, sample rate,
segments and references: fair.**

| system | settings in code | documented default | verdict |
|---|---|---|---|
| Whisper small | faster-whisper 1.2.1, int8, CPU 2 threads, `beam_size=1`, `language="en"`, no timestamps, `condition_on_previous_text=False` (`make_asr`) | faster-whisper's default is `beam_size=5`, `condition_on_previous_text=True`; Pipecat's local `WhisperSTTService` uses faster-whisper defaults (beam 5) with `distil-medium.en` | **disadvantages baseline (small)**: greedy instead of the front end's beam 5. Stated ("beam 1"). English forced: fair (the sets are English; it helps Whisper). The label "LiveKit default STT" has no source: LiveKit Agents 1.8.3 (installed) has no Whisper default, and Pipecat's default local Whisper is distil-medium.en. |
| Whisper large-v3 / turbo | transformers 5.17, fp16 MPS, `num_beams=1`, `do_sample=False`, `language="en"`; > 30 s: long-form with `return_timestamps=True`, `condition_on_prev_tokens=False`, no temperature fallback | transformers' own default is greedy (matches the Open ASR Leaderboard); OpenAI's `whisper` CLI default is beam 5 + best-of 5 + temperature fallback | **fair for short segments** (greedy = transformers default; LS numbers 2.04 / 3.55 sit on the public leaderboard values, so the import is not broken). **Disadvantages baseline (small) on the live sessions**: all 32 are > 30 s (mean 45.6 s), so they go through long-form without temperature fallback or compression-ratio checks. No repetition loops were found (hyp / ref length 0.73-1.04), so the cost is small. |
| Parakeet-TDT 0.6B v3 | imported into our modules (`nemo_import`, TDT durations read from the config, `type="tdt"`), offline, greedy TDT, CPU 2 threads | NeMo default decoding for TDT is greedy (`greedy_batch`) | **fair** for WER. LS 1.83 / 3.40 match NVIDIA's card (1.93 / 3.59 on the full sets), so the import does not handicap it. |
| ours 0.6B / 115M | masked [70,1] offline forward = 160 ms streaming, greedy, CPU | n/a | – |

**Reference convention (live calls) — disadvantages baseline, unstated, and it decides one headline claim.** The
TurnBench references are verbatim: they keep repetitions and false starts ("It's- it's very scary", "I'm- I'm a
PR"). Whisper writes clean verbatim (drops repetitions); the Whisper normaliser removes "um / uh" but not repeats.
Re-scoring the cached hypotheses with a simple normaliser, with and without repeats / false starts collapsed in
both reference and hypothesis (`scratchpad/disfl.py`; absolute values differ from the Whisper normaliser, the
shift is the point):

| live calls, user channel (16) | verbatim reference | repeats and false starts collapsed |
|---|---:|---:|
| ours 0.6B | 6.99 | 6.67 |
| Parakeet-TDT v3 | 8.51 | 8.82 |
| Whisper large-v3 | 8.97 | 6.99 |
| Whisper large-v3-turbo | 8.59 | **6.20** |
| Whisper small | 10.11 | 7.94 |

So "on the user's channel of live calls the 0.6B is best (5.7 % against 7.7 %)" holds against Parakeet-TDT but not
against Whisper turbo / large-v3 once the reference convention is neutralised; 16 sessions, CIs overlap anyway.
On AMI test the same collapse moves Whisper by only 0.6-0.7 pp (13.3 → 12.7), on ICSI by 1.4 pp, ours by 0.1-0.8 pp:
the AMI / ICSI orderings do not change.

### Final transcript latency ("Final transcript ready after you stop")

| system | device in the row | verdict |
|---|---|---|
| ours 115M / 0.6B (`--final-chunk-ms 1120`) | MPS | – |
| Parakeet-TDT v3 | **CPU, 2 threads** (`final_latency.py OFFLINE`) | **disadvantages baseline (large), flagged only by the device column** |
| Whisper large-v3 / turbo | MPS fp16 | fair |
| Whisper small | CPU int8, 2 threads (CTranslate2 has no MPS) | disadvantages baseline (moderate): 2 of the M5's cores |

Quantification for Parakeet-TDT. On the same CPU (2 threads) Parakeet-TDT and our 0.6B have the same offline
throughput (RTFx 19.7 vs 19.8 on test-clean, 20.6 vs 20.9 on AMI test: same FastConformer size), so the 341 ms p50
is a CPU number for an encoder that we time on the GPU for ourselves. Our own 0.6B engine runs 2.26x faster on MPS
than on CPU (42.9 vs 97.1 ms per chunk), and a single full-turn offline forward gains at least that much. Applying
only that lower-bound ratio gives Parakeet-TDT ≈ 150 ms p50 on MPS; a larger batch-forward gain (likely) puts it in
the 50-150 ms range. Consequences: Parakeet-TDT would be the fastest baseline (ahead of Whisper turbo's 282 ms), and
the gap to our 0.6B (49 ms) shrinks from 7x to at most ~3x, possibly ~1-2x. The 115M's 25 ms lead is likely to
survive. Fix: run `final_latency.py run --system parakeet_tdt` with `"mps"` in `OFFLINE` (~40 s job).

Other points in this table:
- **Accuracy at the timed operating point is not shown.** Concatenating each system's per-turn texts from the
  timing run and scoring against `text_user` (simple normaliser): ours 0.6B 8.97, Parakeet-TDT 8.43, Whisper turbo
  10.79, large-v3 11.70, small 12.08, ours 115M 15.12. The flushed 1.12 s final at the labelled turn end is about
  2 pp worse than the 0.6B's whole-session decode (6.99), while Parakeet-TDT loses nothing. The 115M, fastest at
  25 ms, delivers the most errors. The table should carry a WER column. (neutral; framing)
- Clock: ours starts before processing the last ≤ 20 ms piece, with `torch.mps.synchronize()` before each stop;
  queued work from earlier blocks would only add to ours. Offline systems get the whole turn at the same instant.
  Load and first-call compile excluded for all. **fair.**
- The streaming cores have already spent compute during the turn; that is the design and it is stated. **fair.**

### Speech detection (VAD)

Common: `vad_layers.load_set` windows; every system's labels are asserted equal (`score_vad`: `np.array_equal(z["y"],
y)`), 16 kHz, same windows. **Same audio, labels, boundaries: fair.**

| system | settings in code | verdict |
|---|---|---|
| Silero v5.1.2 | TorchScript, 512-sample chunks, `reset_states()` per 20 s window, max-pooled to 80 ms | fair (Pipecat / LiveKit run the same model in 512-sample chunks) |
| TEN VAD 1.0.6.8 | 256-sample hop on int16, fresh instance per window | fair |
| MarbleNet v2 | imported, whole-window convolution (non-causal), raw frame probabilities, offset −hop/2 | fair (offline here, helps it; stated) |
| pyannote seg-3.0 | 10 s windows, default step, `1 − P(no speaker)` | fair (offline, stated) |

Two caveats that tilt the headline toward us:
- **Metric choice on AMI — disadvantages baselines, partly stated.** The claim "our heads beat the VADs that ship
  today" rests on F1 at a fixed 0.5 threshold. Our heads were trained on AMI / ICSI train with exactly this label
  convention, so their 0.5 is calibrated to it; the baselines' 0.5 is not. On AMI test the threshold-free metrics in
  the same json do not support the win: AUC MarbleNet 0.967 = ours 0.966 / 0.967 (paired CI of the AUC difference
  −0.011 to +0.013), and miss at 7.5 % false alarms MarbleNet **9.7 %** vs ours 12.5 / 12.6 %. On ICSI test ours win
  on all three metrics (AUC 0.951 / 0.956 vs ≤ 0.929; miss 12.1-12.8 % vs ≥ 20.7 %).
- **Onset lag column — disadvantages causal baselines by ~80 ms, stated in a footnote.** Lag excludes look-ahead;
  our heads have 80 ms right context, Silero / TEN have none, so "0 ms vs 80 ms" is the look-ahead itself.
- Pipecat's `min_volume` gate is not modelled (only noted in `baselines/turn.py`); not used in the VAD table.

### Turn taking

Common: the same 399 smart-turn v3.2 test clips (`eot_assistant.clips()` asserts the split name) + 3 s silence, and
the same 200 AMI test turns / 109 call turns (`eot_latency.sessions()`). The baselines read the Silero v5
confidences computed by the served session on the same audio. Scoring (`score_session`) is shared.

| system | settings vs installed defaults | verdict |
|---|---|---|
| Pipecat smart-turn v3.2 + Silero | `VADParams` confidence 0.7 / start 0.2 / stop 0.2 = Pipecat 1.12 `VAD_*` constants; smart-turn `STOP_SECS=3`, `PRE_SPEECH_MS=500`, `MAX_DURATION_SECONDS=8` = installed defaults; Pipecat's own analyzer classes on a simulated clock; model compute added | fair; STT wait not modelled (flatters Pipecat, stated); `min_volume=0.6` not modelled (unstated, direction unclear) |
| LiveKit Agents 1.8 EnglishModel + Silero | min / max endpointing 0.5 / 3.0 s = `_ENDPOINTING_DEFAULTS` in livekit-agents 1.8.3; Silero plugin activation 0.5, min silence 0.55 s = plugin defaults; EOU compute added | **disadvantages baseline (moderate)**: its text input is our 115M streaming transcript (16.1 % WER on AMI test) instead of a production STT; at an end of speech with no text yet LiveKit gets no EOU call, which turns into misses. Stated as a fact, impact not discussed. |
| Parakeet-Realtime-EOU 120M | streamed in 160 ms blocks (`StreamingSession`, att [70,1]) on MPS; compute added; **restart after each `<EOU>`** | **fair, and the restart helps it**: without the reset the model emits one `<EOU>` per long session (checked by the authors), so every later turn would count as missed. The reset drops left context after each turn; that only affects the next few blocks. |
| smart-turn classifier alone | reused from `runs/smartturn_audit.json`, same 399 clips, same file | fair (labelled as an upper bound) |

- **AMI meetings: ours is told who the user is, the baselines are not — disadvantages baselines, unstated.** On
  the AMI mono-mix turns our engine is enrolled with the target's 5 s voice print (`stage_eotdump`:
  `arm_enrollment(... embedding=emb)`), and every served preset has the "others" path (`others: [12, 8]`: end the
  turn when the user is silent while someone else talks). Silero-driven baselines see a meeting where other people
  keep the VAD on, which is exactly what produces their 72-86 % misses. This is a real product capability, but the
  AMI turn rows and README sentence present it as a like-for-like comparison. No no-print run of ours on AMI exists,
  so the size is unknown. Fix: state it, and add an unenrolled row (`eotdump` without `arm_enrollment`).
- The 115M's `fast` / `assistant` constants were tuned on the assistant test clips (stated; held-out constants give
  a better 95.0 %, so no inflation). The 0.6B's v0.4 `assistant` was selected on held-out scopes only
  (TURN_DATA.md §2.2). Our turn classifier and Pipecat's model both train on smart-turn train. fair.
- False fire = a turn end within 3 s of the cut; every framework's fallback (Pipecat 3 s after the VAD stop, LiveKit
  3 s after speech end + 0.55 s, ours 3.44 s) lands just outside it. Same rule for all: fair.

### Speaker tracking and speaker EER

- tWER / target DER: the same eot-bench v2 test windows, the same 5 s print, the same words (115M or 0.6B) and keep
  rule for every arm. Nemotron-3 runs at its 0.32 s card preset (`low_latency_032`, the lowest-latency NVIDIA preset;
  its 1.04 s preset costs 2 DER points less on AMI dev, EARLY_RESULTS.md) — latency-matched, **fair**. pyannote 3.1
  offline on the whole window with default hyper-parameters — helps it, **fair**.
- **The diarizer arms are measured through our own causal binder (`tsvad.vp_follow`) — disadvantages baselines,
  partly stated.** The json already holds an oracle-column upper bound: on AMI test Nemotron-3 with the oracle
  column reaches F1 0.795 / DER 39.7 against our tracker's 0.811 / 37.3 (115M) and 0.829 / 35.8 (0.6B), versus the
  published 0.676 / 62.9 with the binder. On ICSI the oracle column gives 0.831 / 39.8 against ours 0.884-0.896 /
  20.7-22.4. Most of the AMI margin is the binder, not the diarizer; ours still win in both cases. The oracle rows
  should be shown.
- "Better of two binders" is chosen by F1 (`frame_best`), so the ICSI Nemotron-3 DER cell shows 75.6 (speaker-head
  binder) while the TitaNet binder gives 74.5. 1 point against the baseline; minor.
- Nemotron-3 has AMI in its training data, our TS-VAD heads train on AMI / ICSI train: stated.
- EER: same 200 segments per corpus, same within-meeting pairs and bootstrap; TitaNet-L imported, WeSpeaker
  `Inference(window="whole")`. **fair** (the baselines win).

### Language ID

Same 2550 FLEURS test clips, the same `lid.clip` path (onset, pre-roll) for 2 s and full; every posterior restricted
to the 17 languages; Whisper's language-token posterior after `<|startoftranscript|>` is what `detect_language`
does. AmberNet predictions are reused from the same `lid.py` clip path (labels asserted equal; clip identity is
implied by the shared code, not asserted). **fair.**

### Reproducibility (affects whether the published numbers can be regenerated)

- `final_compare.py` defaults to `runs/stage1_served_v3.afm` (`AFM_115M`), the v0.2 0.6B build (`afm_0p6b`) and the
  old LID heads (`lid_distill.pt`, `lid_0p6b.pt`, `FINAL_LID_HEAD`). The published rows (115M v0.4, 0.6B v0.3 /
  v0.4, LID v2) need `FINAL_115M_AFM`, `FINAL_0P6B_AFM` and `FINAL_LID_HEAD` set, which the "Reproduce" block in
  FINAL_COMPARE.md never mentions. `stage_report` records the env value at report time, not at stage time, and the
  stage npz files do not store which head produced them. The `systems` strings in the json still say
  `stage1_served_v3.afm` for the 115M rows.

## (B) Published numbers vs the run json

Script: `plans/audit/fairness_001.py` (stdlib-only uv script; `uv run plans/audit/fairness_001.py [-v]`, exit 1 when
anything is flagged). It carries an explicit cell → json-path map for the README tables, the FINAL_COMPARE.md
headline tables, a list of prose claims, every `numbers_final.json` entry (re-read through its stored source
path) and the 71 keys `arch_vs.html` reads for `compare_*.png`; it checks image keys against each PNG's last
commit, flags dev-split / old-head numbers in public docs, and cross-checks numbers repeated across files.

Run on 2026-10-03: **OK 1194, MISMATCH 5, NOSRC 10, STALE 143, NOTE 11.** The ranked findings above (1, 6, 7,
10, 11, 12) summarise them; the full list is the script's output.

How compare_*.png get their numbers: `arch_vs.html` `VS` config → `numbers_final.js(on)` (built by
`export_final.py` from `runs/final_compare.json`, `runs/dual_rate.json`, `runs/turn_data.json`). compare_asr:
`final/dualrate/wer/*/1120` for ours, `final/wer/*` for baselines, `final/finallat/p50/*` for the latency chart
(no device label). compare_spk: `twer/*/pyannote31` is the 115M-words arm.

## Suggested fixes (cheap first)

1. Replace or delete the README 0.6B table and its prose; fix README:79 / :105 head versions and numbers.
2. Re-run `final_latency.py` for Parakeet-TDT on MPS; until then add "(CPU)" to the Parakeet bar in compare_asr and
   a WER column to the latency table.
3. State in the AMI turn rows that ours is enrolled; add an unenrolled `eotdump` row.
4. Add a disfluency-neutral sensitivity row for the live sets (or drop "best on the user channel").
5. Report AUC / miss next to F1 in README's VAD sentence; show Nemotron-3's oracle-column row; use 0.6B words for
   the pyannote bars.
6. Put `FINAL_115M_AFM` / `FINAL_0P6B_AFM` / `FINAL_LID_HEAD` into the Reproduce block (or change the defaults);
   rewrite EXPLAINER.md for compare_*.png; sweep the docs listed under 11.


## Docs split (2026-10-03)

The README's results tables and prose moved verbatim to `docs/RESULTS.md`, its dual-rate paragraph to
`docs/USAGE.md`; the claim map follows them (`RES`, the `TXT` file names) and checks the new short README table
(24 claims). Output after the move: OK 1318, MISMATCH 0, NOSRC 0, STALE 33 (the same pre-existing lines in
CONFIGURATION.md, PROTOCOL.md, research/README.md and CHANGELOG.md).
