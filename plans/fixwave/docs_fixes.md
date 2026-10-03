
## Stale docs (docs agent)

2026-10-03. Fixed every line `plans/audit/fairness_001.py` flagged STALE in docs/CONFIGURATION.md, docs/MODELS.md,
docs/PROTOCOL.md, docs/SERVER_INTERNALS.md, research/METRICS.md and research/README.md (list before:
runs/docsfix/fairness_before.txt). Current numbers come from research/FINAL_COMPARE.md headline tables /
runs/final_compare.json; no Appendix A row and no ICSI speech-detection or ICSI speaker row is quoted as a result.

- docs/CONFIGURATION.md: shipped build is `stage1_served_v4.afm`; `--turn-preset` help (audioforge/server/cli.py,
  flag block regenerated) and the Turn presets table rewritten to the 115M test rows (AMI test) and the calls rows
  (labelled "not a test split"); the 115M shipped `assistant` "92 % at 291 ms" claim removed (Appendix A); vad_head
  row points to the selection experiment and the test table; hybrid ASR / single-mode WER and LID moved to AMI test /
  FLEURS-17 test (16.1 %, TDT v3 8.3 %, LibriSpeech 2.38 / 6.82 %, LID 92.4 / 98.2 %); dev-split tables captioned.
- docs/MODELS.md: speech-head and LID lines on test numbers (AMI test F1 0.959 / 0.957; LID +5.1 / +3.1 points over
  `lid_0p6b.pt`); block-4 VAD paragraph labelled as an AMI dev selection experiment; LID file size 9.48 MB; heads
  v0.2 / v0.3 checksum rows labelled "(previous; v0.4 is current)".
- docs/SERVER_INTERNALS.md:30: the 0.32 vs 1.04 s miss-rate pairs replaced by a sentence, labelled as a dev-split
  selection experiment.
- docs/PROTOCOL.md: the turn-hint table caption says it is a dev-split selection experiment for the default H.
- research/METRICS.md: scorecard rewritten (20 rows, both cores, test splits; calls row labelled not a test split;
  definitions updated); the 2026-09-29 live-run sections kept as labelled history; obsolete "Image text changes"
  section removed.
- research/README.md: FINAL_COMPARE.md indexed and named as the numbers to quote; CORE_0P6B / METRICS key numbers on
  the test splits; FINAL_REPORT and SPK_HEAD lines labelled dev-era.

### Kept on purpose

Each is a dev-split selection experiment that documents why a default was chosen, labelled as such in the line or
its table caption; the flagged value is a measured dev number (or a coincidental match), not a test result.

- docs/CONFIGURATION.md:361 — energy gate / `--turn-model smartturn` comparison table, default row (calls 956 ms,
  AMI dev 1326 ms): same run as the smartturn rows it is compared with; column header now "AMI dev" and the caption
  below the table says selection experiment, test rows in FINAL_COMPARE.
- docs/CONFIGURATION.md:382 — §4.1 `timeout` row, eot-bench v2 AMI dev 74.8 % miss: research table whose caption now
  says AMI dev selection experiments, not test results.
- docs/CONFIGURATION.md:448, :450 — binder table (eot-bench v2, AMI dev n = 974; 74.8 % / 87.6 % cells): explains
  the `--enroll` modes; caption now "a dev-split selection experiment, not a test result".
- docs/CONFIGURATION.md:480 — diarizer-preset table, 0.32 vs 1.04 s miss rates (74.8 %): why `low_latency_032` is
  the default; the row says "dev-split selection experiment, not a test result".
- docs/PROTOCOL.md:463 — turn-hint H sweep, AMI dev row (62.1 % is the "confirmed" column, a coincidental match):
  why H = 0.8 is the default; caption labelled dev-split selection experiment.
- research/README.md:88 — archive index line for TURN_ERRORS.md ("62 % → 9.5 % miss, AMI dev n=200"): a historical
  note's key number, labelled AMI dev in the line.

## README, images, audit cell map (README / images agent)

2026-10-03, after the eval commits ca25833 / a5c070c (speakers-unseen ICSI speech rows, turn_clean merged).

- README.md: every number from numbers_final.json / the FINAL_COMPARE headline (test splits, nothing from Appendix A):
  words table with Whisper small at beam 5 (no "LiveKit default" label) and the disfluency tie line; final-text table
  on the same device (MPS; Whisper small CPU, named) with WER of the timed texts; turn table at the 2.5 s window with
  the one-line why, the 115M row = held-out pick 96.0 % / 381 ms / 2.7 % (candidate v0.5, not default) and one line
  that the shipped faster preset was tuned on the test clips; AMI test rows with and without the voice print; speech
  detection worded to the threshold-free numbers (AMI: MarbleNet ties on AUC; ICSI unseen speakers: F1 lead small,
  0.6B ties TEN, AUC / miss clear win); AMI-only speaker line (0.6B words for the diarizers, oracle-binding note);
  voice-gender line with its caveat; head table (speech detector, LID v2 2.4 / 2.9 M, voice gender); dual-rate cost
  +9-27 ms (+93 ms 0.6B CPU) and streams 4 → 3 / 3 → 1; the `--core 0.6b` table rewritten from the same test rows.
- Images: arch_vs.html reads only numbers_final.json keys (83, all present; FOCUS rows emptied), prints each entry's
  `shown` string; words page = README columns at 160 ms + final-text chart labelled MPS / 1.12 s final; VAD page F1
  at threshold 0.5 and threshold-free AUC, ICSI = speakers-unseen heads; turn page labels the 2.5 s window and the
  compute device; speaker page AMI only, 0.6B-words diarizers, oracle-binding upper bound, EER. No voice-gender page
  (no fair baseline). architecture_v10.png: head list and sizes. scripts/dev/render_compare.py now checks every page
  (text overlap on the text extents, label gaps, spill out of chart / frame, chart-vs-block overlap) and exits 1;
  output: runs/docsfix/render_check.txt (all clean).
- docs/ARCHITECTURE.md (not flagged, stale): speech detector, v5 classifier, LID v2, voice gender; docs/CONFIGURATION.md
  LID row now the v2 head.
- EXPLAINER.md rewritten for architecture_v10 / compare_*.png; the v8-v10 write-ups retired (git history).

### plans/audit/fairness_001.py cell-map changes (the script's own stale mapping, not real mismatches)

Before (runs/docsfix/fairness_before.txt): OK 1024, MISMATCH 22, NOSRC 48, STALE 146, NOTE 12.
After (runs/docsfix/fairness_after.txt): OK 1294, MISMATCH 0, NOSRC 0, STALE 33, NOTE 11.

- README tables: turn table maps to `asst_windows > 2.5` and the 115M held-out row to `turn_clean`; new final-text
  table spec; the `--core` table labels of the rewrite added to its `now` dict; README prose claims replaced by the
  new text's claims (every one checked against a json path).
- FINAL_COMPARE: Whisper small beam 5 / beam 1 rows (`whisper_small_b5` / `whisper_small`); final-latency rows keyed
  on label + device (the 9 Parakeet MISMATCHes were the MPS row matched to the CPU path) plus the WER column; VAD rows'
  new labels, ICSI columns of ours from `vad_speakers_unseen`; calls-table labels with ⚑ / "held-out pick"; speaker
  table is AMI-only with 3 columns (the 6 MISMATCHes were ICSI columns hitting it) plus oracle-binding rows; LID rows
  with ⚑ and the previous-heads row (`lid_previous_heads`); two removed prose claims dropped.
- NOTE texts: dual-rate cost line and head versions follow the README; the words-image 1.12 s note is printed only
  while the image reads the 1.12 s keys.

### Remaining flags (explained)

- STALE 7: the docs agent's "Kept on purpose" list above (dev-split selection tables, labelled).
- STALE 26: CHANGELOG.md, dated history entries that quote the numbers of their day; not rewritten (history).
- NOTE 11: image / architecture references (informational), and provenance notes: the 115M MPS stream count is 5 in
  final_compare and 4 in the dual-rate run (two runs; README quotes 5 for cost and the same-run 4 → 3 for the mode
  cost), the dual-rate p95 costs match the README's +9-27 / +93 ms.
