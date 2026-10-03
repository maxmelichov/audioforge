# Evaluation fixes (fix wave, 2026-10-03)

Owner: the evaluation agent. Brief: plans/audit/{metrics,fairness,leakage,tests}_001.md. Files:
`scripts/research/final_compare.py`, `final_scoring.py` (new: the scoring pieces below), `final_latency.py`,
`dual_rate.py` (scoring only), `runs/final_compare.json`, `research/FINAL_COMPARE.md`,
`demo/images/redesign/export_final.py` -> `numbers_final.json` / `numbers_final.js`. Heavy jobs: `plans/fixwave/queue_eval.sh`
and `queue_eval2.sh` (one at a time through `scripts/dev/gate.sh`, logs in `runs/fixwave/`).

## What was changed

1. **VAD past-end frame.** Each 20 s window has 251 label frames; frame 250 (20.00-20.08 s) lies past the audio and
   was scored as a forced miss for Silero / TEN VAD / pyannote. `final_scoring.vad_inside` keeps only frames that start
   inside the audio, for every system (64 frames dropped per set). Window lengths come from `final_scoring.py meta`.
2. **Turn false-fire windows.** `final_scoring.asst_curve` scores accuracy / false fire at 2.0 / 2.5 / 3.0 / 3.5 s for
   every system (eot_assistant.FF_WINDOW set per window; at 3.0 s it reproduces the published rows, asserted).
   Headline-window rule (`final_scoring.headline_window`): the largest candidate more than 0.25 s away from every
   system's nominal fallback timer. Fallbacks: Pipecat 1.12 `STOP_SECS` 3.0 s, LiveKit 1.8.3 `max_endpointing_delay`
   3.0 s, ours `assistant` 37 frames = 2.96 s (115M) / 43 frames = 3.44 s (0.6B), `balanced` / `fast` 0.64-0.72 s.
   Result: **2.5 s**. The 3.0 s column is kept for continuity.
3. **CIs by meeting.** `final_scoring.py meta` rebuilds the AMI / ICSI test item lists, asserts they equal the cached
   sets item by item, and stores each item's meeting. WER (AMI / ICSI test; also the 1.12 s rows of dual_rate), VAD
   (windows grouped by meeting) and speaker EER (within-meeting pairs pooled per drawn meeting) now resample meetings.
   Live calls: sessions (clip; mono and user channel together), as before.
4. **Reproducibility.** Defaults = shipped builds (115M `stage1_served_v4.afm`, 0.6B `served_0p6b_v0.4.afm`, LID v2
   heads), `EOT_AMI_SPLIT=eval` set in code, the 0.6B turn rows scored from the v0.4 dumps
   (`scratch/turndata/fc_v04/eot`), the v0.3 "before" row read from `runs/turn_data.json`, the previous-heads VAD /
   LID rows scored from `scratch/fixall/fc_old`. `files_used` in the json names the build behind each section.
   Checked: 0.6B v0.3 -> v0.4 heads files have all 123 v0.3 tensors bit-identical (v0.4 adds `turn_vad`, `turn_seg_a`
   and changes only the `assistant` preset); 115M v0.3 -> v0.4 all 190 identical (v0.4 adds `speech`).
5. **Final-text latency.** Parakeet-TDT and Whisper small re-run on MPS (Whisper small through transformers fp16,
   beam 5: CTranslate2 has no MPS); CPU rows kept; device on every row; WER of the timed turn texts next to latency.
6. **Whisper small beam 5** (faster-whisper `transcribe()` defaults, what Pipecat 1.12's `WhisperSTTService` calls) is
   the headline row; beam 1 kept. Defaults read from the installed packages: Pipecat 1.12 local `WhisperSTTService`
   = `Systran/faster-distil-whisper-medium.en`, faster-whisper defaults (beam 5), `no_speech_prob` 0.4; LiveKit Agents
   1.8.3 has no default STT (`AgentSession(stt=...)` is NOT_GIVEN -> no STT; `inference.STT` is LiveKit Cloud's hosted
   router). The "LiveKit default STT" label is gone.
7. **Speaker tracking.** Diarizer headline arms on the 0.6B words; Nemotron-3 / pyannote oracle-binding rows (frame
   F1 / miss + FA) added; "target DER" renamed "target-speaker miss + false-alarm rate"; standard DER (md-eval style,
   0.25 s collar, overlap scored, every speaker; pyannote.metrics) of the two diarizers' cached tracks
   (`final_scoring.py stdder`). Not defined for our single-target tracker.
8. **Live-call disfluencies.** `whisper_norm_disfl` = Whisper normalizer after dropping hyphen-final false-start
   fragments, then immediate 1-3-word repeats collapsed, same rule for references and all hypotheses.
9. **AMI turn without the voice print.** `final_compare.py eotdump --noprint`: the AMI test sessions with no print
   armed (no TS-VAD "others" path), both cores, scored with the same presets (`ami_noprint`).
10. **Leakage.** FINAL_COMPARE.md marks every flagged row; contaminated rows moved to an appendix.

## Runs

`plans/fixwave/queue_eval.sh` (meta, Whisper small beam 5 on the six test sets, Parakeet-TDT on MPS) and
`queue_eval3.sh` (the rest; written after the first queue starved behind other agents' back-to-back jobs: CPU jobs at
`GATE_MAX_JOBS=2`, MPS jobs at 1). Logs: `runs/fixwave/*.log`. Then, niced single-thread CPU: `final_latency.py report`,
`dual_rate.py score` + `report` (only the WER CIs changed), `final_compare.py report` with no FINAL_* / EOT_* variable
set (4.5 min), `export_final.py`.

Reproduction check: every numeric leaf of the pre-fix `runs/final_compare.json` (a89cba6) under words, lid, turn,
speaker, speaker_test, stt_latency, cost, final_latency, speaker_eer, pyannote_* (1854 leaves) was compared with the new
report: all identical except the VAD rows (the frame fix), the CIs (meeting resamples) and `params` (now counted on
the v4 / v0.4 builds). The 0.6B turn row now comes from code and equals the pasted v0.4 row.

## Results (what moved)

See research/FINAL_COMPARE.md "What the fix wave changed" for the table. In short:
- VAD: Silero / TEN / pyannote +0.002 F1, +0.003-0.004 AUC, −0.3 points missed; ours unchanged. With meeting CIs the
  AMI F1 gap MarbleNet − 0.6B now touches 0 (−0.032 to +0.000).
- Turn, assistant clips at W = 2.5 s: 0.6B 97.7 % (was 96.5 at 3.0), Pipecat 75.9 (69.7), LiveKit 85.2 (72.7),
  Parakeet-EOU 49.4 (48.4). False fire 0.9 / 29.0 / 22.3 / 89.7 % (3.1 / 40.2 / 44.6 / 91.5). Order unchanged at every
  window; LiveKit's gap to us halves.
- Meeting CIs: AMI / ICSI WER and speaker EER CIs are much wider where one meeting dominates (EER AMI 115M
  [3.2, 6.9] -> [3.5, 12.6]); the 0.6B's AMI WER CI is narrow ([7.69, 8.04]) because its 4 meetings agree.
- Final-text latency: Parakeet-TDT on MPS 232 ms p50 (CPU 341); the 0.6B (49 ms) stays ~4.7x faster on the same GPU.
  WER of the timed texts: 0.6B 7.8, Parakeet-TDT 7.2, 115M 14.2, Whisper turbo 10.6, large-v3 11.3, small b5 9.9 %.
  Whisper small on MPS (transformers, beam 5): 1891 ms p50, no faster than CPU.
- Whisper small beam 5: −0.4 to −1.0 WER points (AMI 11.9 -> 10.9, live user 10.1 -> 9.6); +380 ms final latency.
- Live user channel with disfluencies collapsed: 0.6B 5.37 = Whisper turbo 5.37 < large-v3 5.82 < Parakeet-TDT 7.68
  (verbatim 5.70 / 8.38 / 8.45 / 7.68). "0.6B best on the user channel" holds against Parakeet-TDT only; a tie with
  Whisper turbo once the reference convention is neutralised.
- Speaker: diarizer arms on 0.6B words (Nemotron-3 AMI 68.6 -> 64.4, pyannote 77.1 -> 77.8); Nemotron-3 oracle
  binding F1 0.795 vs ours 0.811 / 0.829 (AMI); standard DER of the diarizers (AMI test): Nemotron-3 25.4 %,
  pyannote 17.2 %; ICSI: 15.5 / 18.5 %.
- AMI turns without the voice print: our cores miss 74.0 % (115M) / 73.5 % (0.6B) with `balanced` (36.0 / 33.5 % with
  the print), interrupt 6.5 / 5.5 % (15.5 / 10.0). Knowing the voice is what separates us from LiveKit (72 % missed)
  on meetings.
- Leakage: the 115M shipped `assistant` rows (assistant clips, calls) and every ICSI speaker row moved to Appendix A;
  the 115M `balanced` / `fast` rows came back after the turn-clean held-out re-pick reproduced them; the 115M's
  held-out `assistant` pick (c5s1, candidate heads v0.5, not default) is shown in the headline table instead: 96.0 % /
  381 ms / 2.7 % at W = 2.5 s.

## Audit scripts after the fixes

`uv run plans/audit/metrics_001.py` (runs/fixwave/metrics_after.txt): asr, longform, dualrate all reproduce (every
WER error count, every 1.12 s row, r1 hyps == final_compare core rows). It stops at `vad` because its uv environment
has no sklearn; the remaining checks were run with `.venv/bin/python plans/audit/metrics_001.py <check>` as the audit
itself does (runs/fixwave/metrics_after_venv.txt):
- `vad` (the audit's all-frames re-score) now differs from the report by exactly the fixed frame, and
  `vad_lastframe` (the audit's corrected scoring) matches the report on all 12 test rows to the 4th decimal.
- `lid`: 6 / 6 match. `turn`: 27 / 27 system x scope rows MATCH. `turn_window`: the same curve as the report.
  `eer`: 16 / 16 match.

`uv run plans/audit/fairness_001.py` (runs/fixwave/fairness_after.txt; before: runs/fixwave/fairness_before.txt):
OK 1024, MISMATCH 22, NOSRC 48, STALE 146, NOTE 12 (before: OK 1194, MISMATCH 5, NOSRC 10, STALE 143, NOTE 11).
The new flags are the audit's fixed cell -> json map meeting the new layout, not wrong numbers:
- FINAL_COMPARE rows it cannot map (new rows: Whisper small beam 5 / beam 1 / MPS, oracle binding, 115M held-out
  rows, labels with ⚑): NOSRC. The LID cells (92.7, 92.4 ...) are flagged only because the row labels now carry ⚑;
  the `lid` check above matches them.
- The final-latency "Parakeet-TDT" MISMATCH is the new MPS row (232 ms) being matched to the CPU path (341 ms), which
  is the next row. The speaker-table MISMATCHes are the map's ICSI columns hitting the new AMI-only table.
- README.md mismatches (VAD 0.899 / 0.928 / 0.975 / 0.922 / 0.920, "1497") are README text that still quotes the
  pre-fix numbers: for the README owner.
The audit scripts were not edited (plans/audit belongs to the auditors).

## Open / handed over

- README / docs / images: the numbers above, the 2.5 s window, the beam-5 Whisper small row, the leakage appendix and
  the "verbatim column" statement for the live calls (README quotes the verbatim column) are for the README agent;
  `numbers_final.json` keeps every old key (paths updated) and adds 100 new ones (`final/asst3/*` = the 3.0 s column,
  `final/ami_noprint/*`, `final/stdder/*`, `final/finallat/wer/*`, `*_heldout`, `*_115mwords`, `*_disfl`, `*_oracle`).
- The speech-head retrain (turn-clean agent, plans/fixwave/turn_clean.md section 1) had no results yet; when it lands,
  re-run `final_compare.py vad --system core_*` on its build (its own WORK dir) and the VAD rows can be swapped.

## Follow-up (coordinator): turn_clean re-merge and speakers-unseen ICSI speech rows

- `runs/turn_clean.json` (rewritten in af80788) is merged whole as `turn_clean` (all §7 keys: heldout, test, build115,
  cost115, served115, speech, splits). Its `test` rows are identical to the ones merged before, so no turn number moved.
- ICSI speech detection, our cores: the headline row is now the speakers-unseen retrained heads
  (`scratch/speech_clean/fc_*_new`, scored by `final_compare.py report` with the same scorer and baselines ->
  `vad_speakers_unseen`): 115M F1 0.930 [0.916, 0.946] / AUC 0.941 / miss 16.0 %; 0.6B 0.922 [0.897, 0.941] / 0.945 /
  15.5 %. They match turn_clean.json's own scoring exactly (0.9295 / 0.9409 / 16.03; 0.9217 / 0.9453 / 15.49). The
  shipped heads' ICSI row (0.938 / 0.940) is in FINAL_COMPARE Appendix A.4. Baselines unchanged. Export:
  `final/vad/icsi/{f1,auc,miss}/ours_*` -> unseen heads, `..._shipped` -> shipped heads, `final/vad/ami/*/ours_*_unseen`.
- Checks: `metrics_001.py vad_lastframe` still matches every ICSI row incl. the shipped heads (runs/fixwave/
  metrics_after_speech.txt); `fairness_001.py`: OK 1024, MISMATCH 22, NOSRC 49 (the +1 NOSRC is the renamed row label
  of our ICSI speech rows, which its fixed map cannot match), runs/fixwave/fairness_after_speech.txt.
