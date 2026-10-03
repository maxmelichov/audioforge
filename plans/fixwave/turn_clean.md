# Turn heads and speech head: contamination fixes (fix wave, 2026-10-03)

Owner: the turn-clean agent. Brief: plans/audit/leakage_001.md findings 1-6 and 11; plans/audit/metrics_001.md
findings 1-2. Code: `scripts/research/turn_clean.py` (new), small fixes in `turn_data.py`, `fixall.py`. Numbers:
`runs/turn_clean.json` (keys documented at the end). Nothing here edits final_compare.py, FINAL_COMPARE.md,
numbers_final.json, tests/ or docs/.

## 0. Selection rule (written and committed BEFORE any candidate was scored)

### 0.1 Audio each step may read

| use | audio |
|---|---|
| selection (held-out) | core_0p6b_turn's held-out set `ho`: 224 oto user-channel windows (calls-like), their 224 quiet-channel variants, 76 AMI windows (ES2015c), 279 smart-turn **train-split** clips + 3 s silence (st3, assistant-like); the eot-bench AMI **dev** turn windows (IS1008b, ES2011b, TS3004b, IB4002; 200 turns); TURN_DATA held-out scopes `dev` (AMI dev headsets), `devq` (same at TurnBench levels), `odev` (oto-280h held-out), `odevq`, `atdev` (AppTek held-out calls) |
| never read for selection | smart-turn v3.2 test (399), AMI test meetings (IS1009b, ES2004b, TS3003b, EN2002a), TurnBench, the 16 oto evaluation conversations (the "calls" set) |
| scored once, after the pick | smart-turn v3.2 test 399 (false-fire window 2.0 / 2.5 / 3.0 / 3.5 s), AMI test turns (200) |

Conversational scopes (9): `ho:calls`, `ho:quiet`, `ho:meet`, `amidev`, `dev`, `devq`, `odev`, `odevq`, `atdev`.
Assistant scope (1): `ho:st3`, false-fire window 3.0 s (the reported definition). Metrics: eot_latency's pooled
p50 total ms (decision + the core's fixed chunk compute: 115M 30 ms, 0.6B 41 ms, as core_0p6b_turn), false
interruption %, missed %; eot_assistant's accuracy %, complete-clip p50 ms, false fire % on incomplete clips.

### 0.2 Candidates

- **115M**: every classifier of research/TURN_V5.md's ablation table: c1, c2, a_notext, a_nost, a_nocompl, c3
  (prosody), c4, a_pool, a_big, c5 (shipped), c5s1, c23 (mean of c2 and c3); and the shipped per-frame v2 turn head
  (head mode, `balanced` only, as shipped).
  - model-mode grid (balanced and fast): k in {1, 2, 3, 4} frames, classifier VAD line in {0.4, 0.5, 0.6},
    P threshold in {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}, fallback in {8, 9, 12} frames, re-asked, others (12, 8),
    energy gate on: 216 rules per classifier.
  - head-mode grid (balanced): k in {1, 2, 3, 4}, theta in {0.9, 0.95, 0.97, 0.99}, fallback in {6, 8, 10, 12},
    VAD line in {0.4, 0.5, 0.6}: 192 rules.
  - assistant grid: k in {2, 3, 4, 5}, VAD line in {0.4, 0.5}, P in {0.8, 0.85, 0.9, 0.95, 0.97, 0.99}, energy-quiet
    trigger in {6 dB, off}, fallback in {37, 40, 43}: 288 rules per classifier.
  - The shipped rules are in these grids (balanced = head k 2, theta 0.99, fallback 8, VAD 0.4; fast = c5 k 1,
    VAD 0.6, P 0.7, fallback 8; assistant = c5 k 3, 6 dB, VAD 0.4, P 0.9, fallback 37).
  - A preset may pick its own classifier. If fast and assistant pick different classifiers, the served config
    carries two (`turn_seg` + `turn_seg_a`, the mechanism the 0.6B v0.4 already serves).
- **0.6B**: constants only, on the v0.4 heads: balanced / fast on the served GRU VAD + classifier `s12` (model grid)
  or the per-frame head `f1` (head grid, balanced); assistant on classifier `kd1stqr` (turn_seg_a) with the GRU VAD
  (assistant grid). The shipped v0.4 rules are in the grids.

### 0.3 The pick

For each core and preset, with a reference R (below):
- **balanced / fast**: among the candidates whose false interruptions AND missed ends are no worse than R's on
  **every** conversational scope, the lowest mean p50 over the 9 scopes. Ties: lower mean FI, then lower mean missed.
- **assistant**: among the candidates with st3 accuracy >= R's and false fires <= R's, the lowest st3 p50. Ties:
  higher accuracy, then fewer false fires.
- **If no candidate passes** (possible only for the 0.6B): the candidate with the fewest violated constraints, then
  the smallest summed excess over R (percentage points, summed over scopes), then the lowest p50. The outcome is
  reported as "none passes" with that closest rule.

References:
- **115M: its shipped presets replayed on the same held-out audio.** This fixes the operating point (how many
  interruptions / misses / false fires the preset accepts) at what ships, which is the AGENTS latency gate's own
  frame ("no slower turn ends than shipped at equal false fires on held-out"). The shipped rule is itself a
  candidate, so it always passes: a different pick means held-out audio finds a candidate at least as fast with no
  more errors on every scope.
- **0.6B: the 115M's clean picks (above) replayed on the same held-out audio**: a reference that was itself picked
  on held-out audio, not v0.1's evaluation-tuned constants (leakage finding 4).

Outcome rules, fixed now:
- 115M: pick == shipped -> the shipped test numbers are clean (held-out selection reproduces them). Pick != shipped
  -> both are scored once on the test rows, and a served-heads v0.5 config with the clean presets is prepared but
  not made the default. It must pass the latency gate: mean held-out p50 not slower than shipped at no more errors
  (true by construction), assistant st3 p50 not slower, and <= +1 ms per 160 ms chunk.
- 0.6B: the clean picks and the shipped v0.4 rules are both scored once on the test rows.

## 1. Speech head (both cores)

Retrain with FIXALL step 1's chosen recipe without every ICSI training meeting that contains any of the 13 speakers
of the ICSI test meetings Bmr013 / Bmr018 / Bro021; select on FIXALL's held-out `sel`; test once on ICSI test and
AMI test, frames inside the audio only. Ships only if held-out is no worse. (Results in §5.)

## 2. TURN_DATA disjointness check (finding 11)

Fixed in `turn_data.py stage_splits` and re-run (commit 7492424): added `heldout_old & eval` (0) and
`train_shipped_turn & heldout_old_turn` (0) and `train_new & heldout_old_turn` (0); TurnBench dev sessions (16) are
now in `eval`. `train_shipped & heldout_old` = 1 (`ami:TS3011b`) is reported as allowed with the reason: the turn
heads trained on TS3011b clips, and TS3011b is only a speech-head (VAD) selection meeting; no turn head or preset is
selected on it (the turn held-out AMI scope is ES2015c). The check asserts that.


## 3. Held-out picks (2026-10-03)

Run: `turn_clean.py x115` (6 scopes), `p115` (12 classifiers), `scan` (2 784 / 2 592 / 3 456 115M candidates for
balanced / fast / assistant; 408 / 216 / 288 for the 0.6B), `pick`. Extraction check: the new 115M quiet-scope cache
equals core_0p6b_turn's `i115` bit for bit, and the AMI-test extraction reproduces the served dumps' c5 p to 7e-4.

Held-out p50 ms / false interruption % / missed % per conversational scope (assistant: st3 accuracy / p50 / false fires):

| core, preset | config | ho:calls | ho:quiet | ho:meet | amidev | dev | devq | odev | odevq | atdev |
|---|---|---|---|---|---|---|---|---|---|---|
| 115M balanced | shipped = pick (head, VAD < 0.4 >= 160 ms & p >= 0.99, OR 640 ms) | 900/28.5/6.5 | 750/39.3/7.2 | 1230/18.4/27.6 | 1326/10.5/33.5 | 1320/14.3/20.0 | 1240/15.2/18.1 | 740/37.8/5.2 | 760/33.8/5.8 | 1051/20.8/3.6 |
| 115M fast | shipped = pick (c5, 80 ms of VAD < 0.6, p > 0.7, OR 640 ms) | 640/28.9/6.1 | 805/32.2/7.8 | 990/18.4/19.7 | 1246/11.5/34.0 | 890/22.3/14.4 | 810/24.8/13.2 | 680/34.0/5.6 | 790/30.1/6.3 | 570/23.5/2.9 |
| 0.6B balanced | shipped v0.4 (s12, 320 ms, p > 0.6, OR 720 ms) | 721/21.7/8.0 | 961/24.1/8.0 | 1401/10.5/40.8 | 1177/11.0/33.0 | 1341/11.7/25.4 | 1311/11.5/23.0 | 761/30.6/8.4 | 841/24.3/9.0 | 691/19.4/3.2 |
| 0.6B balanced | closest (none passes): head, VAD < 0.5 >= 160 ms & p >= 0.95, OR 800 ms | 871/25.6/6.5 | 866/30.9/5.9 | 1161/15.8/25.0 | 1017/23.5/25.5 | 1131/15.9/16.2 | 1071/19.1/14.2 | 781/35.9/8.0 | 731/33.8/7.4 | 791/20.4/3.4 |
| 0.6B fast | shipped v0.4 (s12, 80 ms, p > 0.5, OR 720 ms) | 466/29.1/5.9 | 891/28.7/7.8 | 1161/13.2/31.6 | 1097/17.0/27.0 | 951/18.1/17.0 | 891/18.1/14.9 | 501/37.7/8.3 | 651/28.8/8.7 | 451/26.5/4.6 |
| 0.6B fast | closest (none passes): s12, 80 ms of VAD < 0.5, p > 0.8, OR 640 ms | 826/24.3/8.1 | 921/27.0/7.8 | 1321/11.8/36.8 | 1097/11.5/31.5 | 1146/15.5/19.1 | 1131/15.5/17.7 | 831/33.3/8.9 | 841/27.2/9.2 | 611/22.8/4.2 |

| core, assistant | config | st3 accuracy / p50 / false fires |
|---|---|---|
| 115M | shipped (c5, 240 ms energy-or-VAD quiet, p > 0.9, OR 2960 ms) | 91.8 % / 350 ms / 3.5 % |
| 115M | **pick (pre-registered rule): c23** (mean of c2 and c3), 160 ms, p > 0.9, OR 3200 ms | 91.8 % / 329 ms / 2.1 % |
| 115M | pick among servable classifiers: **c5s1** (c5's second seed), 160 ms, p > 0.95, OR 2960 ms | 96.1 % / 350 ms / 1.4 % |
| 0.6B | shipped v0.4 (kd1stqr, 160 ms, p > 0.9, OR 3440 ms) | 97.8 % / 361 ms / 1.4 % |
| 0.6B | pick: the same with p > 0.85 (an exact held-out tie with shipped; the tie broke on grid order) | 97.8 % / 361 ms / 1.4 % |

What the picks say:
- **115M balanced and fast: the held-out pick is the shipped config.** Of 2 784 / 2 592 candidates, only the shipped
  rule is no worse than itself on all nine scopes: no classifier of the ablation table and no constant in the grid is
  as good on every held-out scope and faster. Read this for what it is: at the shipped operating point, held-out audio
  finds nothing better, so the shipped balanced / fast test numbers are what a held-out selection gives. It does not
  show that held-out audio would have chosen that operating point (how many interruptions to accept); that was set
  on the calls and AMI dev audio earlier.
- **115M assistant: the pick differs.** The rule picks `c23`, the c2 + c3 average, 21 ms faster on held-out with fewer
  false fires. The server cannot run it: c3 reads 12 prosody features the served engine does not compute, and c23 is
  an average of two classifiers. That servability limit was missing from §0; it is applied after the fact and stated
  as such. Among servable classifiers the same rule picks `c5s1`: the same held-out p50 as shipped, with 96.1 vs 91.8 %
  accuracy and 1.4 vs 3.5 % false fires.
- **0.6B balanced / fast: no rule matches the 115M reference on all nine scopes** (8-10 scopes worse at best). The
  0.6B misses more ends on the oto and AMI-headset scopes than the 115M at any of its own constants that keep its
  false interruptions as low. The closest rules are scored below, but they are not held-out winners. They violate
  5 and 7 constraints, while the shipped rules violate 8 and 10.
- **0.6B assistant: held-out cannot tell the pick from shipped** (identical results). The shipped constants therefore
  stand as a held-out-valid choice.

## 4. Test rows, scored once (smart-turn v3.2 test 399; AMI test turns 200)

Same dumps, scorer and compute as FINAL_COMPARE: the shipped rows reproduce its published numbers exactly. CIs are
1000-resample clip / window bootstraps.

| core, preset | config | AMI test p50 / interrupt / missed | smart-turn test acc / p50 / FF at window 2.0 s | 2.5 s | 3.0 s (reported) | 3.5 s |
|---|---|---|---|---|---|---|
| 115M balanced | shipped = held-out pick | 1527 / 15.5 / 36.0 | 44.1 / 1226 / 99.6 | 43.9 / 1226 / 100 | 43.9 / 1226 / 100 | 43.9 / 1226 / 100 |
| 115M fast | shipped = held-out pick | 1246 / 20.5 / 37.5 | 41.9 / 461 / 99.6 | 41.6 / 461 / 100 | 41.6 / 461 / 100 | 41.6 / 461 / 100 |
| 115M assistant | shipped (c5) | 1327 / 23.5 / 40.0 | 93.2 / 299 / 4.5 | 92.7 / 299 / 5.4 | **92.7 [90.2, 95.2] / 299 / 5.4** | 89.0 / 299 / 12.1 |
| 115M assistant | held-out pick c23 (not servable) | 1487 / 24.0 / 44.0 | 92.7 / 298 / 3.1 | 92.5 / 298 / 3.6 | **92.0 [89.2, 94.5] / 298 / 4.5** | 91.2 / 298 / 5.8 |
| 115M assistant | held-out pick, servable: c5s1 | 1487 / 17.0 / 44.5 | 96.5 / 381 / 1.8 | 96.0 / 381 / 2.7 | **95.7 [93.7, 97.5] / 381 [362, 457] / 3.1** | 92.0 / 381 / 9.8 |
| 0.6B balanced | shipped v0.4 | 1498 / 10.0 / 33.5 | 42.9 / 639 / 99.6 | 42.6 / 639 / 100 | 42.6 / 639 / 100 | 42.6 / 639 / 100 |
| 0.6B balanced | closest (none passed) | 1379 / 14.5 / 26.0 | 43.6 / 607 / 99.6 | 43.4 / 607 / 100 | 43.4 / 607 / 100 | 43.4 / 607 / 100 |
| 0.6B fast | shipped v0.4 | 1338 / 16.0 / 27.5 | 40.4 / 414 / 99.6 | 40.1 / 414 / 100 | 40.1 / 414 / 100 | 40.1 / 414 / 100 |
| 0.6B fast | closest (none passed) | 1459 / 11.0 / 29.0 | 39.8 / 335 / 99.6 | 39.6 / 335 / 100 | 39.6 / 335 / 100 | 39.6 / 335 / 100 |
| 0.6B assistant | shipped v0.4 | 1497 / 22.5 / 34.5 | 97.7 / 351 / 0.9 | 97.7 / 351 / 0.9 | **96.5 [94.5, 98.2] / 351 / 3.1** | 96.5 / 351 / 3.1 |
| 0.6B assistant | held-out tie (p > 0.85) | 1298 / 28.5 / 29.0 | 97.7 / 319 / 0.9 | 97.7 / 319 / 0.9 | 96.5 / 319 / 3.1 | 96.5 / 319 / 3.1 |

What should change, and what should not:
- **115M balanced / fast:** nothing. The shipped constants are the held-out pick, so their test numbers (AMI test
  1527 / 15.5 / 36.0 and 1246 / 20.5 / 37.5; smart-turn test as above) are now clean, with the operating-point caveat
  of §3.
- **115M assistant:** the shipped 92.7 % / 299 ms / 5.4 % comes from a classifier and constants that held-out audio
  does not pick. The clean numbers:
  - the pre-registered pick, c23: **92.0 % / 298 ms / 4.5 %**. It cannot be served today.
  - the servable pick, c5s1: **95.7 % / 381 ms / 3.1 %**.
  The v0.5 candidate (§6) carries c5s1. Its trade-off in plain words: it answers correctly on about 3 more clips in
  100 and falsely cuts in on about 2 fewer in 100. It waits about 80 ms longer before answering on the test clips:
  0.38 s instead of 0.30 s at the median. On held-out audio the two answered equally fast; the 80 ms only shows on
  the test clips.
  - It was not made the default.
  - The test set cannot decide the switch: that would be selecting on test.
  - The held-out rule says it is no slower at fewer errors, so it passes the latency gate as defined.
- **0.6B balanced / fast:** keep v0.4. No held-out rule reaches the held-out-picked reference. The closest rules are
  not winners: each trades interruptions for misses on the AMI test, one each way.
- **0.6B assistant:** keep v0.4. Held-out ties it with the p > 0.85 variant. The variant is 32 ms faster on test, but
  choosing it now would be a choice made on the test clips.
- On the window question (metrics audit finding 2): the 115M assistant rows lose 3-4 accuracy points at 3.5 s; the
  0.6B assistant and c23 rows move by at most 1.3 points between 3.0 and 3.5 s. The 2.0 s window favours every row.

## 5. Speech head without the ICSI test speakers (FIXALL step-1 recipe)

The 13 speakers of Bmr013 / Bmr018 / Bro021: fe008, fe016, fn002, me001, me006, me011, me013, me018, me026, mn007,
mn014, mn017, mn052.
- Dropped from training: Bdb001, Bmr009, Bmr014, Bmr026, Bro010, Bro016, Bns003, Bsr001.
- ICSI training left: Bed010 and Bed011, 127 of 600 windows. The ICSI share stays 0.25.
- Code: `scripts/research/speech_clean.py`, `fixall.py --icsi-exclude-test-speakers` (commit 4460910).

Held-out (FIXALL `sel`; ICSI-held Bro026 / Bmr022 contain test speakers but only select; AMI-held TS3011b, see §2):

| core | head | AMI-held F1 / AUC | ICSI-held F1 / AUC | oto F1 | quiet oto F1 | room p95 | sel |
|---|---|---|---|---|---|---|---|
| 115M | shipped | 0.964 / 0.988 | 0.955 / 0.991 | 0.963 | 0.957 | 0.177 | 0.9904 |
| 115M | without the test speakers' meetings | 0.961 / 0.988 | 0.948 / 0.988 | 0.963 | 0.957 | 0.101 | 0.9896 |
| 0.6B | shipped | 0.963 / 0.989 | 0.959 / 0.992 | 0.966 | 0.957 | 0.123 | 0.9915 |
| 0.6B | without the test speakers' meetings | 0.965 / 0.989 | 0.948 / 0.990 | 0.963 | 0.957 | 0.127 | 0.9908 |

**Not shipped (both cores):** held-out `sel` and ICSI-held F1 are worse. Scored once on the test windows anyway
(frames inside the audio only, 64 frames dropped per set; 95 % window bootstrap):

| core | head | ICSI test F1 / AUC / miss @7.5 % FA | AMI test F1 / AUC / miss |
|---|---|---|---|
| 115M | shipped (trained on all 13 test speakers) | 0.938 [0.919, 0.950] / 0.951 / 12.7 % | 0.959 / 0.966 / 12.5 % |
| 115M | **speakers unseen in training** | **0.930 [0.911, 0.942] / 0.941 / 16.0 %** | 0.956 / 0.965 / 12.7 % |
| 0.6B | shipped | 0.940 [0.923, 0.952] / 0.956 / 12.0 % | 0.957 / 0.967 / 12.6 % |
| 0.6B | **speakers unseen in training** | **0.922 [0.903, 0.936] / 0.945 / 15.5 %** | 0.956 / 0.966 / 11.6 % |

Paired new - shipped on ICSI test: F1 -0.008 [-0.011, -0.005] (115M) and -0.018 [-0.023, -0.014] (0.6B). For
reference, the metrics audit's corrected ICSI test F1 of the baselines: Silero 0.924, TEN VAD 0.922, pyannote 0.893.
With speakers unseen, the 115M head still leads them. The 0.6B head ties TEN VAD and is 0.002 under Silero. The
shipped heads' ICSI test lead (0.938 / 0.940) is partly speaker familiarity.

## 6. 115M served-heads v0.5 candidate (prepared, not the default)

`assets/served_heads_v0.5_candidate.pt` (39.5 MB, sha256 `c04418fb…`, state hash `1d5083a3…`). It is rebuilt
bit-identically from the NVIDIA .nemo + this file. The candidate served model is `scratch/turn_clean/stage1_served_v5_cand.afm`. It is v0.4 plus:
- `turn_seg_a` = c5s1;
- `cfg turn_presets`: balanced and fast unchanged; assistant = c5s1, 160 ms of energy-or-VAD quiet, p > 0.95,
  fallback 2960 ms.

Code: `scripts/research/turn_clean_serve.py build115 | cost115 | served115`.

| latency gate (MPS, full single-mode engine, bundled call, best of 3, interleaved v0.4 / candidate) | v0.4 | candidate | pass? |
|---|---|---|---|
| balanced ms per chunk p50 (p95) | 28.80 (31.41) | 28.84 (31.00) | yes (+0.04) |
| fast | 34.07 (40.53) | 34.94 (40.07) | yes (+0.87; the preset is identical in both, so this is run-to-run spread) |
| assistant | 35.40 (42.07) | 35.29 (43.72) | yes (-0.11) |
| turn_end times on the bundled call, every preset | | | identical |
| held-out turn ends (the AGENTS rule) | assistant st3 p50 350 ms | 350 ms, with fewer false fires | yes (no slower at fewer errors) |

Served vs offline:
- On 16 held-out st3 clips through the raw-audio served session, the candidate and v0.4 match the offline twin
  equally well: most decisions sit on the twin's frame (+16 ms clock offset), a few are one frame early, and 2 per
  build decide later.
- So the offline cache, not the candidate, limits this check. A dump-based served == offline check (as TURN_V5 did on
  the 399) was not repeated.

## 7. `runs/turn_clean.json` keys (for the eval agent's merge)

- `heldout.<core>.<preset>`: `reference`, `n_candidates`, `n_pass`, `strict_pass`, `pick`, `shipped` (each
  `{track, rule, res: {scope: {n_turns, eot_total_ms_p50, false_interruption_pct, missed_pct}} | {ho:st3: {accuracy_pct,
  p50, false_fire_pct}}}`), `pick_equals_shipped`, `shipped_violations_vs_reference`. The 115M assistant also has
  `pick_servable`, `n_pass_servable`, `servable_equals_shipped`, `not_servable_reason`. core is `115m` or `0p6b`.
- `test.<core>.<preset>.<config>` with config = `shipped` | `clean` (the pre-registered pick, when it differs) |
  `clean_servable` (115M assistant): `track`, `rule`, `ami_test` (eot_latency pool + `ci95`),
  `smartturn_test_w2` / `_w2.5` / `_w3` / `_w3.5` (`accuracy_pct, p50, p95, false_fire_pct, missed_pct, ci95`);
  plus `test.<core>.<preset>.clean_equals_shipped`.
- `build115`: the v0.5 candidate's files, hashes and `turn_presets`.
- `cost115.mps.<preset>`: `{v0.4, v0.5_candidate: {chunk_ms_p50, chunk_ms_p95, turn_end_t}, delta_p50_ms, turn_ends_identical}`.
- `served115.<v0.4|v0.5_candidate>`: the served-vs-twin rows.
- `speech.<115m|0p6b>`: a copy of `runs/fixall.json speech_clean`: `test_speakers`, `excluded_meetings`,
  `speakers`, `heldout`, `ships`, `candidate`, `test.{icsi_test, ami_test}`. The ICSI test entry carries the label
  "speakers unseen in training".
- `splits`: the re-run TURN_DATA disjointness check (§2).
