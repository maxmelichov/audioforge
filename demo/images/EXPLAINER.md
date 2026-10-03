# The published images: architecture_v10.png and compare_*.png

The [README](../../README.md) shows the architecture image and [docs/RESULTS.md](../../docs/RESULTS.md) the five
comparison images. All numbers on them are public **test**-split numbers from
[research/FINAL_COMPARE.md](../../research/FINAL_COMPARE.md) (headline tables only; nothing from its Appendix A of
rows tuned on their own test audio). Updated 2026-10-03 after the evaluation fix wave.

| image | page | what it shows |
|---|---|---|
| `architecture_v10.png` | `redesign/arch_v10.html` | the frozen NVIDIA encoder (115M or 0.6B) and its heads, with sizes |
| `compare_asr.png` | `redesign/arch_vs.html`, `buildVs(…, 'asr')` | words: WER on the RESULTS.md words table's five test sets, final-text latency |
| `compare_turn.png` | same, `'turn'` | turn taking on the 399 smart-turn v3.2 test clips |
| `compare_vad.png` | same, `'vad'` | speech detection on AMI / ICSI test meetings |
| `compare_spk.png` | same, `'spk'` | speaker tracking on AMI test meetings |
| `compare_lid.png` | same, `'lid'` | language ID on FLEURS-17 test |

## How the numbers get onto the images

`arch_vs.html` has no number in it. Each bar names a key of `redesign/numbers_final.json` (loaded as
`numbers_final.js`), which `redesign/export_final.py` writes from `runs/final_compare.json` and `runs/dual_rate.json`.
The printed value is that entry's own `shown` string, so the images round exactly like docs/RESULTS.md. A key that is
missing stops the page with an error; a key whose value is null draws a dashed "still running" bar. The green / red
number over each of our bars is our value minus the best grey (baseline) bar on that chart; "same" means within half a
percent. The 83 keys the pages read are checked by `plans/audit/fairness_001.py` against the run json.

Render:

```bash
python3 demo/images/redesign/export_final.py                                  # numbers_final.js(on) from runs/
PLAYWRIGHT_BROWSERS_PATH=/Volumes/afdev/venvs/video/browsers \
  /Volumes/afdev/venvs/video/bin/python scripts/dev/render_compare.py         # compare_*.png, with the check
chore images                                                                  # both steps
```

**The layout check.** `render_compare.py` checks every page before taking its screenshot and exits 1 on a problem:
no two text runs overlap (measured on the text itself, so a long word overflowing its box counts), x-axis labels keep
6 px apart and other chart labels 3 px, no text spills out of its chart card or the 1920 × 1080 frame, and no chart
card overlaps another block. `--check-only` runs the check without writing PNGs. Inside each chart, `barchart()`
first shrinks the x labels (19 → 13 px) until they fit. `architecture_v10.png` gets the same text-overlap and frame
check when it is rendered from `arch_v10.html` (`build(1920, 1080)` at device scale 2).

## architecture_v10.png

One frozen encoder, drawn once: FastConformer Hybrid Streaming 115M (17 blocks) or Nemotron Speech Streaming 0.6B
(24 blocks); block numbers are written 115M / 0.6B. Around it:
- speech detector on a learned mix of blocks 2-6 / 8-16 ("Speech?", every 80 ms);
- language detector LID v2 on blocks 8-12 / 16-20 (optional);
- RNNT decoder → streaming words, every 160 ms;
- speaker tracker ("Is it you?") on block 4 / 5, fed the voice print (a 5 s sample run once through block 4 / 5 and
  the speaker head);
- the 115M's second, voice-conditioned run into the turn detector (0.32M); the turn-end rule box names both deciders
  (turn detector at ≥ 160 ms of quiet, or the end-of-turn classifier at each quiet frame).

The footer lists every head with its size, 115M / 0.6B: speech detector 33K / 66K, the turn-rule VAD 33K, speaker
head 0.5M, tracker 0.26M, turn detector 0.32M (115M only), end-of-turn classifier 2.5M, and the optional LID v2
(2.4M / 2.9M) and voice-gender (22K) heads.

## compare_asr.png: words

- Five WER charts with the RESULTS.md words table's columns: LibriSpeech test-clean, test-other, AMI test, ICSI test,
  the live calls' user channel. Ours are the **default 160 ms streaming pass** (the subtitle says so), the same values
  as the RESULTS.md table. Bars: ours 0.6B, ours 115M, Parakeet-TDT 0.6B v3, Whisper large-v3. Whisper small (beam 5) is
  in the RESULTS.md table, not on the image (five bars do not fit five charts at this width).
- "Final text after you stop" (top right): median ms from the labelled end of a user turn to its final text, **every
  system on the Mac GPU (MPS)**; ours with the 1.12 s final (`--final-chunk-ms 1120`), Parakeet-TDT and Whisper turbo
  offline on the whole turn. The chart says both. Not shown: the WER of the timed texts (RESULTS.md table) and that the
  1.12 s mode costs real-time streams.

## compare_turn.png: turn taking

Accuracy, median answer latency and cut-offs on the 399 smart-turn v3.2 test clips, `assistant` preset.
- The cut-off window is **2.5 s** (named on both charts that depend on it): a turn end within 2.5 s of where an
  unfinished clip stops counts as a cut-off. 2.5 s keeps clear of every system's fallback timer (3.0 s for Pipecat and
  LiveKit, 3.0 / 3.4 s for ours).
- Latency includes the measured compute: ours and Parakeet-EOU on MPS, smart-turn and LiveKit's ONNX models on CPU
  (the unit line says so).
- "ours 115M" is the rule re-picked on held-out audio only (candidate heads v0.5, not the default), as in
  RESULTS.md; the shipped 115M `assistant` rule was tuned on these clips and is not shown.
- Not shown: AMI meeting turns (RESULTS.md text; knowing the user's voice print is what wins there).

## compare_vad.png: speech detection

AMI test and ICSI test meetings, live-capable detectors only (pyannote, which reads 10 s ahead, is in the RESULTS.md
text). Four charts: F1 **at threshold 0.5** and ROC-AUC (**threshold-free**) for each corpus. On AMI the threshold
matters: our F1 lead over MarbleNet v2 disappears on AUC ("same"). The ICSI bars of ours are the heads retrained
without the ICSI test speakers (the shipped heads heard them in training; their row is in Appendix A), as the
subtitle says.

## compare_spk.png: speaker tracking

AMI test meetings only (every ICSI test speaker is in our training meetings, so those rows are in Appendix A). The
same 5 s voice print for every system.
- Your words only (target-speaker WER): ours on our own words; Nemotron-3 and pyannote 3.1 on the 0.6B's words,
  bound to the print; dashed "perfect filter" = the 0.6B's words with the reference speaker mask.
- Tracking F1: the dashed bar is Nemotron-3 with its column picked from the labels (an upper bound for any binder),
  named in the chart's third header line. Most of our margin over Nemotron-3 is the binding step.
- Voice match error (EER within a meeting): TitaNet-L and WeSpeaker beat our speaker heads; shown on purpose.

## compare_lid.png: language ID

FLEURS-17 test, 2550 clips, after 2 s of speech and on the whole clip. Whisper large-v3 and AmberNet beat our heads
at 2 s. FINAL_COMPARE marks our rows ⚑: the encoder blocks the heads read were chosen with a probe scored on FLEURS
test (probably well under a point).

## Not on any image

- Voice gender: no open baseline was available to compare with (research/VOICE_GENDER.md), so it has no page.
- Two-party calls: no labelled public test split exists for them.

## Earlier images

The v8-v10 result cards and the v8 / v9 architecture drafts (and their `_square` / `_notext` variants) are retired:
they used first-pass dev-split numbers and are not referenced from the README or the docs. Their write-ups are in
this file's git history.
