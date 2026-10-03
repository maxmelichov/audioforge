# Metrics audit 001: how the FINAL_COMPARE numbers are computed

Date: 2026-10-03. Read-only audit of `scripts/research/final_compare.py`, `dual_rate.py`, `final_latency.py`,
`turn_data.py` (evalv04), `core_0p6b_turn.py` (`ev_score`, `served_rules`), `eot_latency.py`, `eot_assistant.py`,
`tswer.py`, `vad_auc.py`, `fixall.py` (twsub / frsub) and the scoring helpers they call (`hybrid_asr` normalisers and
edit distance, `audioforge.baselines.sd.pool_probs` / `intervals_to_frames`, `audioforge.baselines.turn` replays,
`audioforge.metrics.eer`).

The numbers were re-derived from the saved per-item outputs (`scratch/final_compare/{asr,vad,lid,eou,eot,spk,pya}`,
`scratch/dualrate/wer`, `scratch/finallat`, `scratch/turndata/fc_v04/eot`, the baseline decision files), using my own
scoring code in `plans/audit/metrics_001.py`: jiwer plus a plain Levenshtein, my own turn scorer and EER sweep, and
sklearn AUC. The run: `.venv/bin/python plans/audit/metrics_001.py asr|longform|dualrate|vad|vad_lastframe|lid|turn|turn_window|eer`
(CPU only, a few minutes in total).

**Bottom line.** Every reported number I re-derived matches the saved outputs exactly or within rounding. That covers
about 400 numbers in total: WER, VAD, LID, turn, EER, tWER, final-text latency and the 1.12 s rows. I found one real
computation bug, in the VAD rows. It is small, it always works against the three causal or offline baselines, and it
flips one ranking. I also found one measurement-design weakness that matters for the turn headline: the 3 s false-fire
window sits exactly on the baselines' default fallback. The remaining findings are about provenance and labelling.

---

## Findings, ranked by impact

### 1. VAD: a label frame past the end of the audio is scored as a forced miss for Silero, TEN VAD and pyannote (bug)

**What the script does.** Each 20 s window has **251** label frames. Frame 250 covers 20.00 to 20.08 s, which is past
the end of the audio. The label grid follows the encoder frame count, and the speaker segments are not clipped to the
window, so that frame is labelled speech in 77 % of AMI-dev windows. `sd.pool_probs` gives a label frame the max over
the fine frames that overlap it, or **0 if none overlap**:
- Silero (32 ms chunks), TEN VAD (16 ms hops) and pyannote have no fine frame past 20.00 s, so they score 0 there:
  exactly one zero per window (`p == 0` on 0.4 % of frames = 1/251).
- MarbleNet (offset −hop/2) and both cores do produce a value there.

So the three baselines take a forced miss on that frame in most windows. Their score on a frame that has no audio
also enters the AUC and the miss-at-FPR numbers.

**What it should do.** Score only label frames that lie inside the audio, for every system (drop frame `T-1` when
`T*0.08 > len(audio)/SR`). The alternative is to give the baselines their last fine value there.

**Recomputed after dropping that frame for every system** (reported → corrected):

| set | system | F1 @0.5 | AUC | miss @FPR 7.5 % |
|---|---|---|---|---|
| ami_dev | silero_v5 | 0.9147 → 0.9166 | 0.9561 → 0.9597 | 13.81 → 13.52 |
| ami_dev | pyannote_seg3 | 0.9628 → 0.9647 | 0.9835 → 0.9874 | 4.63 → 4.27 |
| ami_dev | ten_vad | 0.9253 → 0.9272 | 0.9481 → 0.9517 | 13.59 → 13.28 |
| icsi_dev | silero_v5 | 0.9287 → 0.9305 | 0.9338 → 0.9371 | 19.87 → 19.60 |
| icsi_dev | pyannote_seg3 | 0.8988 → 0.9003 | 0.9041 → 0.9071 | 28.64 → 28.43 |
| icsi_dev | ten_vad | 0.9353 → 0.9370 | 0.9515 → 0.9549 | 14.01 → 13.79 |
| ami_eval | silero_v5 | 0.8989 → 0.9008 | 0.9579 → 0.9613 | 13.33 → 13.02 |
| ami_eval | pyannote_seg3 | 0.9746 → 0.9765 | 0.9832 → 0.9869 | 4.14 → 3.77 |
| ami_eval | ten_vad | 0.9281 → 0.9299 | 0.9517 → 0.9551 | 13.69 → 13.37 |
| icsi_eval | silero_v5 | 0.9217 → 0.9235 | 0.9226 → 0.9260 | 26.57 → 26.38 |
| icsi_eval | pyannote_seg3 | 0.8909 → 0.8926 | 0.9288 → 0.9323 | 21.14 → 20.82 |
| icsi_eval | ten_vad | 0.9199 → 0.9217 | 0.9288 → 0.9323 | 20.73 → 20.49 |

MarbleNet and both cores move by at most 0.0003 F1, 0.0008 AUC and 0.3 pp miss.

**Which reported numbers move.** Every Silero, TEN and pyannote VAD number moves: about +0.002 F1, +0.0035 AUC and
−0.3 pp miss. The core-minus-baseline gaps shrink by that much, and the `* - core_*` delta CIs shift with them. For
example, the ICSI-dev F1 gap from TEN VAD to the 115M core goes from 0.0121 to 0.0107, and the AMI-dev AUC gap from the
115M core to Silero goes from 0.0136 to 0.0099. One ranking flips: on icsi_dev F1, Silero (0.9305) now beats MarbleNet
(0.9289). The cores still lead on ICSI F1 and still trail pyannote on AMI. The onset-lag numbers are not affected.

### 2. Turn, assistant clips: the 3 s "not cut" window sits on the baselines' default 3.0 s fallback (design weakness)

Pipecat's `STOP_SECS = 3` and LiveKit 1.8's `max_delay = 3.0` (I checked both in the installed packages) are the
fallback times of the two replays. `eot_assistant.FF_WINDOW = 3.0` counts an incomplete clip as cut when a turn end
comes before the audible end + 3 s. The audible end is the later of the Silero end and the energy end, while the
baselines count their fallback from their own Silero stop. Whether a fallback fire counts as a false fire therefore
depends on differences of tens of milliseconds.

For incomplete clips, the first fire minus the audible end has a median of 3.1 s for Pipecat and 3.0 s for LiveKit,
and 44 % and 51 % of those fires fall in [3, 4) s. Moving the window shows how unstable the baseline numbers are:

| system | window 2 s | 3 s (reported) | 3.5 s | 4 s |
|---|---|---|---|---|
| Pipecat + smart-turn | 77.9 % acc / 25.4 % FF | 69.7 / 40.2 | 44.9 / 84.4 | 44.9 / 84.4 |
| LiveKit + EOU | 88.2 / 17.0 | 72.7 / 44.6 | 46.4 / 91.5 | 46.1 / 92.0 |
| ours 115M assistant | 93.2 / 4.5 | 92.7 / 5.4 | 89.0 / 12.1 | 87.5 / 14.7 |
| ours 0.6B assistant (v0.4) | 97.7 / 0.9 | 96.5 / 3.1 | 96.5 / 3.1 | 94.2 / 7.1 |

Our "assistant" fallbacks (37 and 43 frames = 2.96 and 3.44 s) also sit near the edge.

The code computes the definition correctly, and the ranking (ours > LiveKit ≈ Pipecat) holds at every window. The
size of the gap does not. Suggestion: report the fired-by-cut+w curve (`score_system` already computes +0.5/1/2/3 s), or
a window that no system's fallback lands on, next to the 3 s headline.

### 3. Bootstrap unit too fine for WER on AMI/ICSI and for speaker EER (CIs too narrow; point values unaffected)

- **WER.** `_boot_rate` resamples items. The AMI, ICSI, ami_eval and icsi_eval sets are 200 segments drawn from only a
  few meetings (for example 4 for ami_eval), so the errors are correlated within a meeting and the item-level CIs and
  paired-delta CIs are too narrow. The meeting id is not saved with the cached sets (`*_refs.json` / npz), so I could
  not recompute meeting-level CIs. The live set is grouped by clip, correctly (mono and user of one clip resample
  together). LibriSpeech and FLEURS item-level CIs are fine.
- **Speaker EER.** `score_spkeer` resamples segments, but the pairs come from 4 / 2 / 4 / 3 meetings (AMI dev, ICSI
  dev, AMI eval, ICSI eval). The same caveat applies.
- **tWER** (`score_pyatwer`, `tswer`, `fixall.twsub`) already uses a meeting bootstrap, which is correct.
- I re-ran the bootstrap with my own code. The CIs agree within Monte-Carlo noise (±0.05 to 0.4 pp); the order of the
  resamples differs because `_boot_rate` sorts groups as strings.

### 4. The final_compare.json turn row for the 0.6B is not what `final_compare.py report` would produce (reproducibility trap)

`turn.ours_0p6b` and `superseded` were pasted in from `runs/turn_data.json evalv04["v0.4"]`. All 9 preset × scope rows
and the rules match exactly, so the numbers are right. But `score_turn()` reads `WORK/eot/0p6b_*`, which holds the
**v0.3** dumps, and resolves rules from `afm_0p6b()` (default v0.3). Re-running `final_compare.py report` without
`--only` would silently replace the published v0.4 row with v0.3 numbers. Fix: point `score_turn` at
`turndata/fc_v04/eot` and the v0.4 afm (or regenerate WORK/eot/0p6b_*), so the published row comes from code.

### 5. Provenance labels (no number moves)

- The `ours_115m` turn row is labelled `stage1_served_v4.afm`. Its 32 call dumps and 399 assistant dumps were made with
  **v3**; only the 200 AMI-test dumps used v4. I checked: v4 = v3 + the `speech` head and its `layer_mix`, with all
  other tensors bit-identical. The presets and `served_rules` are identical, and the turn path reads the `vad` head,
  not `speech`. So the numbers are unaffected, but the label is not accurate.
- The `systems.asr` / `systems.vad` descriptions say the 115M core is `stage1_served_v3.afm`, but the run used v4
  (`afm.115m`).

### 6. "Target-speaker DER" is not the standard DER (label)

`score_pyaframe`, `compile_speaker` and `fixall.frsub` compute **(miss + false alarm) / target frames** for one target
on the 80 ms grid. There is no speaker-confusion term and no collar; the pyannote tracks are rasterised with the same
any-overlap rule as the labels. The code computes it the same way for every system (I re-derived ours, Nemotron-3 and
pyannote from the frame counts). Under the standard-metrics rule it should be named "target-speaker miss + FA rate",
or standard DER (pyannote.metrics, stated collar) should be reported next to it.

### 7. Small things checked and found negligible

- **Our turn compute.** We add each session's median 160 ms chunk compute (MPS) to every decision, not the deciding
  chunk's compute. The recorder also removes the turn-classifier calls from `chunk_ms`. Using the p95 chunk compute
  instead moves the assistant p50 by +4 to +10 ms (115M 1226 → 1231, 461 → 471, 299 → 307; 0.6B 639 → 643, 414 → 419,
  351 → 357). The baselines' compute is measured per decision on CPU (smart-turn p50 16 to 22 ms; LiveKit 1 to 2 ms,
  28 % of calls being cache hits). LiveKit's compute is already inside `t` (`max(t_eos + ms, t_se + delay)`), so the
  scorer correctly adds 0 for it.
- **Session-start exclusion** (`drop_pre`, turn ends before the first Silero chunk > 0.5). It is applied identically
  to all five systems through `EA.score_system`, and in this run it affected **0 clips for every system**, so it has no
  effect.
- **The policy replay for our turn rows.** `turn_v5.run_policy` (the served `VadHeadPolicy`) reproduces the served
  engine's own `turn_end` times on all 399 / 399 assistant clips for both cores (balanced preset, the policy those
  dumps ran).

---

## Numbers re-derived that matched

**WER, Whisper English normaliser, corpus-level (sum of errors / sum of reference words), empty hypotheses counted as
deletions.** All 9 sets × all complete systems match to the error count:
- libri, ami, icsi, live, ami_eval, icsi_eval, ls_clean, ls_other, fleurs_en for whisper_small / turbo / large_v3,
  tdt_v3, core_115m, core_115m_beam8 and core_0p6b. For example: AMI 456/3146, 388, 389, 299, 649, 595, 327; live 541/3687
  … 426/3687; FLEURS 243/3055 … 245/3055.
- The FLEURS 1.12 s rows: core_115m_f1120 10.51 (321/3055) and core_0p6b_f1120 7.76 (237/3055).
- The same normaliser is applied to the references and every system's hypotheses. Empty hypotheses: up to 8 per set
  for the cores and ≤ 2 for Whisper; all are counted.
- **Long-form Whisper is complete.** On the 32 live sessions (all > 30 s), the hypothesis/reference word ratio has a
  median of 0.93 to 0.95 for the three Whisper models (0.97 for core_0p6b and 0.98 for TDT), with no truncated 30 s
  windows. The single > 30 s LibriSpeech item is complete for every system.
- **The 1.12 s final rows** (`runs/dual_rate.json` wer), re-derived with a plain Levenshtein: all 40 values (2 cores ×
  5 sets × r0/r1/r6/r13) match. For example: 115M ls_other 6.82 → 5.39, icsi_eval 18.44 → 14.93; 0.6B live 11.55 → 10.74,
  icsi_eval 10.25 → 8.37. The r1 hypotheses are identical to the final_compare core rows (300/300, 200/200, 32/32).
- The bootstrap CIs reproduce within Monte-Carlo noise (see finding 3).

**VAD** (from `vad/*.npz`). F1 @0.5, AUC and miss @FPR 7.5 % match for all 24 system × set rows before the fix. The
other points I checked:
- Threshold 0.5 is applied to probabilities: every system's scores lie in [0, 1]; the cores apply `sigmoid` once, and
  pyannote's `1 − exp(logp[no speaker])` is in [0, 1].
- The achieved FPR at the chosen threshold is 7.51 to 7.52 % for every system, so ties do not distort it.
- Labels are identical across systems (asserted).
- Labels use an "any overlap" rule rather than the frame centre, and the baselines are max-pooled with the same
  any-overlap rule, so the convention is consistent across systems.

**LID** (from `lid/*.npz` and `data/lid/preds/ambernet`). Accuracy at 2 s and on the full clip matches for all six
systems: whisper_small 90.59 / 99.06, turbo 95.18 / 99.33, large_v3 95.92 / 99.45, core_0p6b 92.67 / 98.55,
core_115m 92.43 / 98.24, AmberNet 95.10 / 99.53. All systems use the same `lid.clip` (Silero onset − 0.1 s) and the same
onsets file. VAD gating differs by design: the cores gate on their own VAD head, while Whisper and AmberNet see the whole
clip.

**Turn**, with my own scorer:
- Calls (32 sessions), AMI test (200) and assistant clips (399). p50 / p95 / false interruption / missed / accuracy /
  false fire match for Pipecat + smart-turn, LiveKit + EOU, Parakeet-Realtime-EOU, and ours 115M and 0.6B (v0.4) ×
  balanced / fast / assistant: 15 system × scope rows for the baselines and 18 for ours, all matching.
- The definitions as implemented:
  - A complete clip is answered when the first turn end falls in [audible end − 80 ms, min(end + 6 s, clip end + 3 s))
    and no turn end comes earlier.
  - An incomplete clip is cut by any turn end before the audible end + 3 s.
  - p50 is measured from the audible end, decision time plus measured compute.
- The same scorer is used for every system, including the replays.

**Final-text latency** (`scratch/finallat`). p50 / p95 match for all six systems: ours 115M 24.7 / 47.9, ours 0.6B
48.7 / 71.2, TDT 341.2 / 616.8, Whisper large 576.1 / 1630.6, turbo 282.4 / 498.4, small 1423.6 / 2308.1 ms. All six
use the same 56 turns in the same order.
- **The clock starts at the labelled turn end for every system.** For ours, it starts when the last piece, cut at the
  end sample, is handed to the engine. For the offline models, it starts when the whole turn is handed over.
- **Warm-up is excluded for all.** Ours runs a 12 s two-turn session; the offline models run 3 s, 8 s and 35 s
  segments.
- Ours does not synchronise MPS before `t0`, which can only add to our time.

**Speaker EER** (`spk/*.npy`, my own EER sweep). All 16 values match: AMI dev 19.78 / 13.64 / 11.97 / 11.88, ICSI dev
7.02 / 3.57 / 2.06 / 1.01, AMI eval 5.03 / 3.80 / 1.92 / 2.02, ICSI eval 2.53 / 1.94 / 1.18 / 0.89 (115M / 0.6B /
TitaNet-L / WeSpeaker). Only within-meeting pairs are used, and every embedder gets the same segment list (5101 / 14524
/ 6941 / 7401 pairs).

**Target-speaker WER**, re-pooled from `pya/*/units.jsonl` and the tswer / core_0p6b unit files:
- All 8 pyannote and Nemotron-3 arms on AMI and ICSI match (for example pya_tn_d2 85.32 / 74.83, n3_tn_d2 74.66 / 65.09).
- The test-meeting re-pool `fixall.twsub` matches for ICSI (836 units) and `twsub_ami_eval` for AMI eval (940 units):
  ours 115M 34.87 / 51.49, ours 0.6B 28.95 / 47.12, n3_tn 64.03 / 68.61, pya_tn 73.60 / 77.05, oracle 0.6B
  23.43 / 31.67.
- The unit counts (974 / 1286 primary units) match tswer.json.
- The keep rule is the same `tswer.keep_mask` (±2 frames, ASR lag per word set) for every arm. Words are counted with
  `normalize_text` on both sides; a reference word counts if its midpoint is in the window.
