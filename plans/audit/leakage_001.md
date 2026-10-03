# Leakage audit 001 (2026-10-03)

Read-only audit: does any reported number (research/FINAL_COMPARE.md, FIXALL.md, TURN_DATA.md, DUAL_RATE.md,
CORE_0P6B_TURN.md, README.md, demo/images/redesign/numbers_final.json) come from audio that a head, preset, threshold,
rule or size was trained, tuned, early-stopped or selected on? Checked by recording / meeting / speaker / clip id and
by audio hash. Where only code could be checked, the code path is cited. Re-run the id checks with
`uv run plans/audit/leakage_001.py` (no torch, no token).

Severity: **contaminated** = the number's own audio was used to tune or select what produced it;
**partial** = indirect (design choices, speaker overlap, a choice between candidates made after seeing the set);
**fine** = no path found.

## Findings, ranked

### 1. Contaminated: 115M turn presets and the v5 classifier were tuned and selected on the 399 smart-turn clips and the calls
- `scripts/research/turn_v5.py` `cmd_scan3` / `cmd_scan4` scan the served-policy rule grid directly on `eot_latency.sessions()`
  (calls = 16 TurnBench dev + 16 oto user channels, plus AMI **dev** windows, the default `EOT_AMI_SPLIT=dev`) and on
  `eot_assistant.clips()` (the 399 test clips). `asst_probs` / `load_sets` feed the 399 into the scan.
- The shipped classifier was also picked on them. research/TURN_V5.md's ablation table picks c5 (block 8) using the columns
  "smart-turn test" (399), "calls fi25" and "assistant >= 90 %" (399). `cmd_stest` scores every tag on the 399.
- **Affected:**
  - 115M `assistant` and `fast` rows on the assistant clips: FINAL_COMPARE 92.7 % / 299 ms / 5.4 % and the fast row; README turn table 92.7 %; README 0.6B table "92.2 %, 292 ms, 5.4 %"; `numbers_final.json` `turn > ours_115m` assistant clips.
  - All 115M calls rows: FINAL_COMPARE calls table and `numbers_final` `no_public_test_split`.
  - README 0.6B table, 115M calls rows "956 ms / 20.2 / 7.3" and "547 / 24.8 / 5.5". These carry **no caveat in the README**.
- FINAL_COMPARE has a caveat, but it names only the constants. It does not say that the classifier (block and recipe) was also chosen on the 399.
- The honest held-out 115M numbers already exist (FIXALL step 2: 95.0 % / 354 ms; fast calls 589 / 30.3 / 8.3).

### 2. Contaminated: the README "turn end, AMI (balanced)" row is AMI dev, the set the 115M constants were tuned on
- README line 188 (115M 1326 ms / 10.5 % / 33.5 %; 0.6B 1177 / 11.0 / 33.0) comes from research/CORE_0P6B.md, which uses
  the default `eot_latency/ami` windows: IS1008b 18, ES2011b 38, TS3004b 62, IB4002 82. Those are the AMI dev meetings.
- TURN_V5's scan used the "AMI goal ≤ 10.5 / 33.5" on these windows. The 115M's 10.5 / 33.5 is that goal value exactly.
- The 0.6B v0.1 constants were tuned on the same sets: `core_0p6b_heads.py stage_preset_scan` docstring says "Tuned on the evaluation sets themselves".
- The AMI **test** turn row (FINAL_COMPARE / README line 104 / numbers_final: `ami_eval`, EN2002a 105, ES2004b 44, IS1009b 31, TS3003b 20 windows) is clean (item 9).

### 3. Partial (strong): every ICSI test speaker is in the ICSI training meetings
- The 13 speakers of Bmr013 / Bmr018 / Bro021 all appear in the 12 ICSI train meetings. They make up 2102 of 3249 segments (65 %) of `data/cache/titanet/icsi_train.npz`; me013 alone has 825.
- Trained on that audio: the 115M speaker head (`spk_relational_titanet.yaml`, `icsi_asr`), the 0.6B speaker head (`spk_frame.py`), the 115M and 0.6B TS-VAD (`icsi_*` feats) and the speech head (`fixall.py icsi600`).
- **Affected:** ICSI test speaker EER (115M 2.53 %, 0.6B 1.94 %) and ICSI test target-speaker tWER / DER / tracking F1 (`final_compare speaker_test.icsi`). The ICSI test VAD rows are affected mildly.
- These are seen-speaker numbers. The baselines (TitaNet, WeSpeaker, pyannote) were not trained on these speakers by us.
- AMI is clean: the 16 test-meeting speakers appear in no train, dev, TURN_DATA train_new / heldout_new or train_shipped meeting.

### 4. Partial: the 0.6B turn presets were chosen between candidates after scoring both on the evaluation sets
- `runs/core_0p6b_turn.json evverify` holds eval-set scores for `Q`, `S_v01heads_heldout_constants` (= v0.2), `dump` (v0.1) and the 115M.
- `Q` was dropped because of its eval TurnBench false interruptions (41 %; CORE_0P6B_TURN §2.4 "Since Q cannot ship"). v0.2 shipped instead.
- v0.2's own constants were picked on held-out audio, but against v0.1 constants that had been tuned on the eval sets (item 2).
- TURN_DATA's design came from eval diagnostics: the quiet-channel levels (`core_0p6b_turn.py:1495`, `Q_SPEECH_DB` / `Q_FLOOR_DB` "TurnBench user channels: median ...") and the v0.4 "classifier clock on the stateless VAD" decision ("the classifier path transfers from the scopes to TurnBench", TURN_DATA §2.1). `turn_data.py stage_test` / `stage_diag` read the cached eval signals.
- **Affected:**
  - 0.6B `balanced` / `fast` rows on calls, AMI test and the assistant clips.
  - README 0.6B table calls rows (725 / 19.3 / 10.1, 487 / 26.6 / 11.0).
  - The calls rows most of all, since TurnBench drove the decisions. The AMI-test and 399 effect is weak.

### 5. Partial (weak): 0.6B v0.4 `assistant` (96.5 %) inherits choices made on the 399
- The v0.4 selection itself is held-out only, as claimed. I verified it:
  - `turn_data.py stage_pick` reads the `hscan_*` files. For the assistant family `all_scores` uses only core_0p6b_turn's `ho` set.
  - The `st` groups in `ho` are 279 smart-turn **train-split** clips, 0 of the 399.
  - Its oto groups (14) are disjoint from the 16 eval conversations.
  - The eval sets were scored once for v0.3 / v0.4 (`stage_evalv04`).
- But the v5 recipe it reuses (c5, block 8) and its distillation teacher (`teach115` = the 115M's c5) were picked partly on the 399 (item 1).
- Affects FINAL_COMPARE / README / numbers_final `ours_0p6b` assistant 96.5 % / 351 ms / 3.1 %. Probably a small effect.

### 6. Partial (weak): README 0.6B-table speaker EER and VAD rows are on the dev meetings used for validation
- README "speaker EER AMI / ICSI 19.8 / 7.0 · 13.6 / 3.6" and "VAD F1 0.951 / 0.898 · 0.951 / 0.906" come from CORE_0P6B.md: AMI dev n = 200, and AMI dev 64 × 20 s.
- Every training recipe validates on `val_split: dev`, the AMI dev meetings IS1008b / ES2011b / TS3004b / IB4002 and ICSI dev Bmr021 / Bns001.
- README target-speaker WER "62.1 / 37.1 · 63.2 / 31.7" is on the eot-bench pools. The ICSI pool includes the test meetings (2414 of 3596 units) and was the non-regression check for TS-VAD serving changes such as fe28a9e (item 7).
- The FINAL_COMPARE test-meeting versions of these rows are the ones to quote (AMI test clean; ICSI see item 3).

### 7. Partial (weak): ICSI test meetings were used as a design check for the TS-VAD serving rule
- `bench_turn_icsi.ICSI_MEETINGS` = ICSI dev + eval. `scratch/tswer/icsi/units.jsonl` holds 2414 test-meeting units.
- It was used as the meeting non-regression check for fe28a9e (TSWER.md 37.11 → 37.14) and to reject the single-voice guard (37.1 → 49-55 %).
- That serving rule was itself tuned on the 32 live sessions (TSWER.md). No reported live **words** number depends on it (item 10).
- Affects ICSI test tWER / DER / tracking F1 (`fixall.py:1360-1405`, the same pool restricted to the test meetings).
- `bench_turn_icsi.py` also cross-fits operating points on dev + eval halves, but no reported number uses that output.

### 8. Partial (minor): LID input blocks chosen with a probe scored on FLEURS test
- `lid.py probe` and `lid_fix.py stage_probe` / `stage_probe_scale` fit on train and score on the 2550 test clips. research/archive/LID.md uses that probe to set "blocks 8-13", which the 115M head reads (8-12). The trainx decision was also read off test.
- 0.6B: `core_0p6b_heads.py stage_lid_probe` calls the same probe. A later dev-only sweep agrees with blocks 16 / 20.
- Weights, steps, hidden size and teacher were picked on dev only.
- Affects the FLEURS-17 LID rows for core_115m / core_0p6b (2 s and full). Probably well under a point.

### 9. Fine, with a note: test-clean is a training WER gate, but the ASR weights are frozen
- `data/librispeech/test-clean-first200.jsonl` (4 speakers) stops and restores runs in 16 recipes (`audioforge/train.py:688-696`). 29 of the 300 `ls_clean` picks are in it.
- Shipped ASR tensors are NVIDIA's and unchanged (`pretrained_lr_mult: 0`). The decoders that this gate controlled were never shipped. ls_clean WER is therefore effectively clean.
- Hygiene: give the gate a dev-clean file.

### 10. Fine: TurnBench and the 32 live sessions were never trained on
- The 32 live sessions are the 16 TurnBench dev clips × {mono, user}: `final_compare.live_sessions` filters `e2e_tsvad/clips.json` to `set == "turnbench"`.
- No recipe, training manifest or `train_shipped` set contains TurnBench.
- But they were **tuned on**:
  - the 115M turn constants (item 1, user channels);
  - the TS-VAD print adaptation (item 7);
  - the 0.6B preset decisions and the quiet-channel design (item 4).
- The live **words** rows (WER all / user, every system, 160 and 1120 ms), the final-transcript latency (56 labelled ends, `final_latency.py` cuts at labelled ends, not at preset decisions) and the live cost / RSS rows do not depend on those tuned parts. They are clean.

### 11. Disjointness check in TURN_DATA: it runs on the real ids, with two gaps
- `turn_data.py stage_splits` builds its sets from the manifests that were actually used and writes `scratch/turndata/splits.json`. I recomputed the intersections. The oto ids share one scheme (11 sessions are in both the 280 h and 141 h slices), so the checks are meaningful. The 16 eval oto conversations are in neither the raw slice nor any new split.
- **Gaps:**
  - (a) `heldout_old ∩ eval` is not in `pairs`. I checked it: 0.
  - (b) `train_shipped ∩ heldout_old` is not in `pairs` either, and it is **1**: `ami:TS3011b`. The shipped turn heads trained on 81 TS3011b clips, yet TS3011b is in the speech-head / FIXALL "AMI held-out" selection set. This makes held-out selection slightly optimistic. It does not touch any test number.
  - (c) `ev_calls` is computed and never used. The TurnBench dev conversations are not in `eval`. That is harmless only because nothing trains on TurnBench.
- AppTek: the split is by locale (dev en-US / GB / CA vs never-touched AU / IE / ZA / IN), and its rows are in the never/heldout sets. No speaker ids exist to check.

## Clean checks (by id or hash)
- **Smart-turn:**
  - The 399 = human_5_all uuids in v3.2-test (399 of 402 matched).
  - v3.2 train ∩ test uuids = 0.
  - No exact or trimmed-fingerprint audio duplicate between the 3463 train and 399 test clips.
  - The 6265 fetched v3.2-train English rows (c4 data) have 0 uuid and 0 fingerprint hits on the 399.
  - Speaker identity cannot be checked (no ids).
- **FLEURS:**
  - 0 test utterance ids in train / trainx / dev.
  - The 350 test sentence ids appear in **no** train, trainx or dev row, pooled over 17 languages or per language.
  - The en_us 150 are the first 150 of the test tarball, fixed at fetch time.
  - Nothing trains ASR or text on FLEURS.
- **LibriSpeech:** none of the 40 test-clean or 33 test-other speakers is in train-clean-100. The speaker caches hold train-clean-100 only.
- **AMI / ICSI meetings:** no test meeting appears in any training cache, recipe, manifest or selection path. Selection uses AMI TS3011b / ES2015c and ICSI Bro026 / Bmr022. `--beam 8` and the dual-rate rules were picked there (FIXALL §5, DUAL_RATE §2).
- `bargein.py` scores corpus dev, and nothing loads its "test". It is not reported.

## Reported numbers that are clean
- **Words (WER), every system:** LibriSpeech test-clean / test-other (300 each), AMI test, ICSI test, FLEURS en_us (150), the 32 live sessions (all / user channel), 115M beam 8, the 160 / 1120 ms dual-rate rows, and the DUAL_RATE deltas.
- **README 0.6B-table WER rows** (AMI-200 / ICSI-200 dev sets, live): clean, because the ASR is frozen.
- **LID, external systems** (AmberNet, Whisper small / turbo / large-v3). Ours are clean in data, with the item 8 caveat.
- **VAD on AMI test:** F1 / AUC / miss at 7.5 % FA, every system. ICSI test VAD is clean of meetings, with the item 3 speaker caveat.
- **AMI test:** within-meeting speaker EER (115M 5.03 %, 0.6B 3.80 %, and the baselines) and target-speaker tWER / DER / tracking F1.
- **AMI test turn rows:** all baselines, and the 115M and 0.6B rows. Item 4 applies weakly to the 0.6B balanced / fast choice.
- **Assistant-clip baselines** (Pipecat, LiveKit, Parakeet-EOU, smart-turn classifier alone): their timing and scoring use our harness, but nothing of theirs was tuned.
- **Latency and cost:** streaming word latency, final-transcript latency (56 labelled ends), ms per chunk, streams, RSS, and parameter counts.
- **TURN_DATA never-touched scopes** (AMI test headsets, oto 280 h never-touched, AppTek AU / IE / ZA / IN): disjoint by id. TURN_DATA §1.3 notes they were used once to re-run the data test after the decision was made on held-out data.
