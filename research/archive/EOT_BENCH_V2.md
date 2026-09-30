# eot-bench v2: leak-free, speaker-aware end-of-turn benchmark on AMI dev (2 s and 6 s horizons)

2026-09-26. The benchmark is the one survivor of research/archive/IDEAS2.md (`wildcards-3`). Code: `scripts/eval_stage1.py --bench v2`
(stages `tracks`, `scores`, `report`). Numbers: `runs/archive/turn_v3_eval_leakfree.json` (turn head `runs/stage1_turn_v3.afm`,
1000 bootstraps). This file also records a bug found while building it: the cached streaming Sortformer tracks had an
**end-of-file flush artifact** (§5). All tables here use the fixed tracks.

## 1. Definitions

- **Turns.** All 974 AMI dev turn windows (4 meetings), not a 200-turn draw. Windows are the library's (`AMI.turn_examples`:
  start = max(turn start - 4 s, previous same-speaker word, end + trail - 20 s)).
- **Horizons.**
  - *2 s* (block A): the default windows (trail 2 s). A post-end frame counts if it lies inside the window; for streaming
    rows its decision may be emitted up to (C + R) frames later (the v1 convention).
  - *6 s* (block C): the same windows re-cut with the same start and a 6 s trail, stopping at the primary's next turn
    (`end_reason` resume) or the meeting end. Horizon = post-end *emission* time ≤ 25 frames ("2s") or ≤ 75 frames
    ("6s"). Block C's "2s" is therefore stricter than block A's (emission time, not frame time).
  - Post-end frames available in C: p5 / p50 / p95 = 18 / 64 / 76. End reasons: 537 resume, 437 trail. At 6 s no miss is
    right-censored (every window either reaches 6 s or ends because the primary resumes).
- **Strata** (floor at the end, horizon 13 frames = 1.04 s): open 236 (nobody else speaks: the case where an agent should
  reply), taken 738 (= overlap 526 + switch 212).
- **Enrollment** (which diarizer column is "the user"):
  - `oracle`: the column that overlaps the oracle primary on [onset, end): the **enrollment leak** of the v1 protocol.
  - `causal_dominant` (k = s = 25 frames): bind at the first active frame to the column dominant over the last 2 s;
    re-bind only after > 2 s of silence of the bound column. Label-free and causal.
  - `first_active`: the first column that becomes active in the window.
- **Arming.** `_labelarm` = the silence timeout counts from the label onset (v1). Without the suffix it counts from the
  bound track's first active frame (deployable).
- **Operating point.** A fixed threshold at ≤ 5 % false cutoffs per turn, chosen on one leave-meetings-out half
  (IS1008b + ES2011b | TS3004b + IB4002) and applied to the other half (cross-fitted). Bootstrap 95 % CIs over turns;
  paired bootstraps for differences. A miss means no firing within the horizon.
- **Streaming tracks.** NVIDIA streaming Sortformer v2 (`runs/nemo_sortformer_v2.afm`), `StreamingDiarizer` window mode,
  card low-latency config (chunk 6, right context 7, FIFO 188, cache 188, update 144, encoder left context 188). The
  emission delay after a frame's chunk is (C + R) frames.

## 2. The 2 s table (block A, n = 974, fixed tracks)

Miss at the cross-fitted ≤ 5 % per-turn FC operating point, with 95 % CI. The artifacted values in brackets are the
previous agent's run on the pre-fix `*.stream_v1.npy` tracks.

| system | all | open | taken |
|---|---|---|---|
| timeout, stream, oracle enrollment, label arming | 76.4 [73.6, 79.1] (76.6) | 63.9 [57.1, 70.1] (65.3) | 80.2 |
| timeout, stream, causal enrollment | 92.5 [90.8, 94.1] (92.5) | 81.1 [75.8, 86.0] (81.1) | 96.0 |
| head v3, stream, oracle enrollment | 72.9 [70.4, 75.9] (72.6) | 77.5 [72.3, 82.8] (77.0) | 71.5 |
| head v3, stream, causal enrollment | 85.9 [83.7, 88.2] (86.0) | 88.1 [83.7, 92.1] (88.1) | 85.2 |
| timeout, any-speaker stream | 88.2 | 54.0 | 98.3 |
| timeout, oracle primary (label bound) | 6.2 | 0.9 | 7.9 |

- **Enrollment leak** (timeout, causal minus oracle-label-armed, paired): +16.0 [13.0, 19.2] points (artifacted: +15.9
  [12.9, 19.2]).
- **v1 protocol at n = 974** (oracle enrollment, label arming, in-sample threshold): timeout on stream 75.7 % (was 75.5),
  head v3 on stream 74.5 % (was 74.1).
- At 2 s the head beats the timeout: causal −6.6 [−9.5, −3.8] points, oracle −3.5 [−6.9, +0.4]. On open ends it loses:
  causal +7.0 [+0.1, +13.4], oracle +13.6 [+5.2, +22.4].

## 3. The 6 s table (block C, n = 974 extended windows, fixed streaming tracks)

| system | miss 2 s (emission) | miss 6 s | P50 6 s | open miss 6 s | taken miss 6 s |
|---|---|---|---|---|---|
| timeout, stream, oracle, label arming | 97.3 | **29.2 [26.7, 32.0]** | 3120 ms | 10.6 [6.9, 14.8] | 34.9 |
| timeout, stream, oracle | 97.4 | 32.0 [29.2, 35.0] | 3200 ms | 15.2 [10.6, 20.3] | 37.2 |
| timeout, stream, causal, label arming | 92.6 | 64.7 [61.4, 67.6] | inf | 30.2 [24.2, 36.6] | 75.3 |
| **timeout, stream, causal (deployable)** | 96.7 | **74.8 [72.0, 77.5]** | inf | **46.1 [39.6, 52.6]** | 83.6 |
| timeout, stream, first active | 99.1 | 95.0 | inf | 90.7 | 96.4 |
| timeout, any-speaker stream | 99.4 | 75.5 [72.7, 78.1] | inf | 31.3 [25.4, 37.4] | 88.6 |
| head v3, stream, oracle | 88.6 | 49.7 [46.7, 53.2] | 6000 ms | 52.2 [45.9, 58.8] | 48.9 |
| **head v3, stream, causal** | 93.2 | **79.6 [77.0, 82.1]** | inf | **80.3 [74.3, 85.5]** | 79.4 |
| head v3, stream, first active | 97.8 | 88.1 | inf | 89.1 | 87.8 |
| timeout, oracle primary | 6.2 | 6.2 | 1520 ms | 0.9 | 7.9 |
| timeout, any-speaker oracle | 77.9 | 68.3 | inf | 16.3 | 84.4 |

Per-pause FC gives the same picture (timeout causal 6 s: 67.6 % all, 35.5 % open).

**Paired differences at 6 s** (miss, points, 95 % CI):
- Enrollment leak, timeout (causal deployable minus oracle label-armed): **+45.5 [+41.9, +48.8]**; open +35.4
  [+28.1, +42.2]. Label-armed vs label-armed: +35.5 [+32.2, +38.6].
- Head v3 minus timeout, causal: **+4.9 [+1.3, +8.5]** (open **+34.2 [+25.9, +42.0]**, taken −4.2 [−8.0, −0.6]).
- Head v3 minus timeout, oracle: +20.4 [+17.4, +23.6] (open +41.6).
- Head v3 causal minus head v3 oracle (the leak inside the head): +29.9 [+26.8, +32.9].

**Horizon share of misses.**
- At 6 s: 0 of the misses are right-censored for any system (0 %). The misses that remain split into "the primary
  resumed speaking before the detector fired" and "the track never gave a long enough silence". For the deployable
  cascade: 420 of 687 misses are resumed turns.
- At 2 s: the share of misses that fire by 6 s is 70 % for the oracle label-armed timeout (633 / 905; open 183 / 225)
  and 23 % for the causal timeout (202 / 889; open 99 / 216). For head v3 it is 44 % (oracle) and 15 % (causal).
  So the 2 s rows mostly measure the window, not the detector.

## 4. Adopt / kill (research/archive/IDEAS2.md §2 / §5) and the v3 verdict

Rule: adopt only if (a) the paired 95 % CI of causal minus oracle enrollment (stream cascade miss) excludes 0 and is
≥ 10 points, and (b) the open-stratum cascade miss stays ≥ 50 % with ≤ 20 % of misses caused by the horizon. Kill if
the open-stratum miss falls below 30 % at 6 s.

- (a) **passes**: +45.5 [+41.9, +48.8] (deployable arming) or +35.5 [+32.2, +38.6] (label arming).
- (b) **fails narrowly** on the level: the deployable cascade's open miss is 46.1 % [39.6, 52.6]. The CI contains 50,
  and label arming gives 30.2 %. The horizon part passes: 0 % censored at 6 s.
- Kill: **not triggered**. The only cascade under 30 % on open ends is the leaky oracle-enrolled one (10.6 / 15.2 %).
- **Outcome: neither adopted as specified nor killed.** The enrollment leak is real and large (a). The horizon claim
  is confirmed: the 2 s window hides most timeout firings. But "the open stratum stays hard" holds only at the 50 %
  bar's edge. I recommend adopting the protocol (6 s windows, causal enrollment, deployable arming, cross-fitted
  threshold, strata) as the scorer for all turn rows. The "≥ 50 % open miss" claim should be restated as "46 %
  [40, 53]".

**Turn head v3 verdict under both horizons.**
- 2 s (the horizon it was trained and selected on): v3 beats the timeout cascade overall (−6.6 points causal) but
  not on open ends (+7.0).
- 6 s: **v3 loses** to the plain timeout on the same causal track: +4.9 points overall and +34 points on open ends.
  Its misses barely fall with horizon (93 → 80 % causal), while the timeout's fall 97 → 75 %. The head was trained on
  2 s trails, so it has never seen a post-end frame beyond 2 s. It has not learned "long silence ⇒ end", which is the
  one thing a timeout does.
- The earlier "v3 improves on streaming" conclusion (research/archive/STAGE1.md) is a 2 s-window artifact. It does not survive
  a 6 s horizon.

## 5. The end-of-file flush artifact in the cached streaming tracks

**What.** `scripts/make_sortformer_tracks.py --track-source stream` (and `eval_stage1.external_diar_probs`) fed only the
window audio and called `StreamingDiarizer.feed(final=True)`. At the end the diarizer flushes: the last chunks run with
a right context of 7 → 0 frames and a shorter encoder window. A live stream keeps receiving audio. So the last 13
frames of every cached track (1.04 s) differed from what a deployed streaming diarizer computes on the same frames.
Those are post-turn-end frames: every window's trail is ≥ 1 s. The turn head was also *trained* on them (train windows
have the same 2 s trail).

**Quantified** (`scratchpad/flushfix/quant.py`, 30 windows per set, reference = the same stream with 1.5 s of extra
real meeting audio, cropped). |Δp| of the cached track by frame position from the end (1 = last frame). Frames 14–16
from the end are exactly 0 (the artifact is confined to 13 frames). Frames before them match to ≤ 9e-6 (MPS vs CPU).

| set | primary col, mean \|Δp\| at pos 1 / 2 / 3 / 5 / 9 / 13 | primary col max (pos 1) | any col max | post-end primary frames changed > 0.01 / > 0.05 | 0.5-threshold flips (windows) |
|---|---|---|---|---|---|
| AMI dev (20 s) | 0.098 / 0.052 / 0.034 / 0.016 / 0.006 / 0.001 | 0.95 | 0.95 | 13.8 % / 6.1 % of 719 | 8 frames (3 / 30) |
| AMI train (16 s) | 0.053 / 0.036 / 0.020 / 0.013 / 0.004 / 0.000 | 0.64 | 0.65 | 8.0 % / 2.8 % of 741 | 2 (1 / 30) |
| ICSI train (16 s) | 0.151 / 0.087 / 0.060 / 0.031 / 0.017 / 0.003 | 0.82 | 0.82 | 23.5 % / 9.5 % of 686 | 10 (7 / 30) |

Full caches (v1 vs fixed track, all columns, last 13 frames): a change > 0.05 appears in 53 % of AMI dev, 58 % of AMI
train, 69 % of ICSI dev and 68 % of ICSI train windows. The mean per-window max is 0.16–0.20, the largest 0.99, and
there are about 1 threshold flip per window (956 / 974 on AMI dev). Frames before the last 13 are unchanged
(≤ 1.8e-4, CPU vs MPS).

**Which padding.** On AMI dev, right-padding with *silence* (mean last-16 |Δp| 0.020) or *repeating the last 80 ms*
(0.021) is no closer to the real-audio stream than the flush itself (0.017). Silence is also worse on ICSI (0.053 vs
0.030). Only real audio removes the artifact: with 14 frames (1.12 s) of the following meeting audio the window frames
are **bit-identical** to the long-context stream (0.0 on all 30 dev windows; proof in tests/test_track_flush.py).

**Fix.** `make_sortformer_tracks.py` now feeds the window plus PAD_FRAMES = C + R + 1 = 14 frames of the following
meeting audio and crops to T. The option `--pad-mode {audio,silence,none}` defaults to audio. At a meeting end it feeds
what is left, and the stream flushes where a live one also ends. `--regen-v1` migrates a cache and keeps
`<key>.stream_v1.npy`. `eval_stage1.v2_tracks` (the 6 s tracks) already used the same real-audio padding and is
identical to it.

**Regenerated** (MPS, `--regen-v1`, `--verify-cpu 3` max |MPS − CPU| ≤ 2.3e-6 on every split): AMI train 3274
(1547 s), AMI dev 974 (479 s), ICSI train 3093 (1487 s), ICSI dev 477 (302 s). Total 7818 tracks in 64 min.
Manifests carry `stream.flush_fix` and `stream_v1`.

**Offline tracks: not regenerated.** Offline (one pass over the window) has no flush. Its last frames still lack future
context. Mean primary |Δp| against a pass over window + 1.5 s is 0.08 / 0.06 / 0.17 at the last frame (max 0.94). But
every other frame moves too (mean 0.013 / 0.012 / 0.026), because full attention sees the extra audio. That is the
definition of an offline-over-the-window row (a non-deployable bound), not a procedural bug. Adding audio would change
the definition.

**Effect on results.** Small at the benchmark level: 2 s rows move ≤ 1.4 points (§2), and n = 200 v1 rows move ≤ 1
point (§6). The artifact is real (up to 0.95 on single frames) but sits in the last second, where the 5 % FC threshold
rarely fires. Results that used the artifacted tracks and should be re-scored or re-read against the fixed caches:
- research/archive/STAGE1.md: the external-diarizer tables (n = 64: `runs/archive/stage1_turn_external_diar.json`,
  `runs/archive/stage1_160ms_turn_external_diar.json`, pickle cache with the same flush), the turn-on-Sortformer rows
  (`runs/stage1_turn_on_sortformer_eval_n{64,200}.json`), the n = 200 table (IDEAS2's target table) and the v3 table.
- `runs/stage1_turn_v3_eval_n200.json`, `..._v4_kernels_eval_n200.json`, `..._v5_headonly_eval_n200.json`,
  `..._v3_ami_icsi_eval_n200.json` (re-scored in §6), and the hybrid run in scratchpad/hybrid (it read the dev cache
  before the regeneration).
- research/archive/TURN_ERRORS.md / `runs/turn_v3_error_analysis.json`: track quality per turn, "primary column re-activates
  after the end in 61 %", post-end bins. The post-end statistics are exactly the affected frames.
- research/archive/IDEAS2.md critic numbers measured on the cached dev rows (66.8 vs 38.4 %, open 89.6 %).
- The previous eot-bench v2 run's block A (replaced here).
- **Training:** stage1_turn_v3, v4_kernels, v5_headonly and v3_ami_icsi trained with `ext_tracks: {source: stream}` on
  artifacted post-end frames. Their checkpoints are unchanged by this fix. Measuring the training effect needs a retrain
  on the fixed caches. `eval_stage1.external_diar_probs` without a cache directory (pickle path) still flushes; use
  `--diar-cache data/ami/cache/sortformer/dev`.

## 6. Re-scored v1 protocol (n = 200, fixed dev tracks) vs artifacted

`scripts/eval_stage1.py --tasks turn --n 200 --diar-ckpt runs/nemo_sortformer_v2.afm --diar-cache data/ami/cache/sortformer/dev`
(the same 200 turns as research/archive/STAGE1.md). Miss at ≤ 5 % FC (in-sample, v1). Before is the committed `runs/*_eval_n200.json`
(artifacted), after is the fixed cache. P50 is unchanged everywhere (inf for the heads, 2640 ms for the timeout).

| checkpoint | head + stream: before → after | timeout on stream | duration rule on stream |
|---|---|---|---|
| v3 | 62.6 → 62.1 | 38.4 → 38.4 | 68.4 → 69.5 |
| v4_kernels | 62.1 → 61.6 | 38.4 → 38.4 | 68.4 → 69.5 |
| v5_headonly | 82.6 → 82.6 | 38.4 → 38.4 | 68.4 → 69.5 |
| v3_ami_icsi | 61.1 → 61.1 | 38.4 → 38.4 | 68.4 → 69.5 |

All deltas are ≤ 1 point (2 turns out of 200). No ranking changes, and the v1 conclusions (head ≫ timeout miss on the
stream track) stand. The 6 s horizon (§3–4) is what changes the verdict, not the fix.

## Commands

```
W=<scratch>
.venv/bin/python scripts/make_sortformer_tracks.py --dataset {ami,icsi} --split {train,dev} --track-source stream \
    --device mps --regen-v1 --budget 540          # repeat until no "re-run"; then --verify-cpu 3
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage tracks --v2-work $W/work --v2-budget 530 --threads 2
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work $W/work --v2-budget 530 --threads 2 \
    --v2-bindings oracle,causal_dominant,first_active,oracle@2s,causal_dominant@2s,first_active@2s --ckpt runs/stage1_turn_v3.afm
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $W/work --n-boot 1000 \
    --ckpt runs/stage1_turn_v3.afm --out runs/archive/turn_v3_eval_leakfree.json
```

## 7. Hybrid (head OR timeout) under leak-free scoring, and the 0.32 s diarizer setting (2026-09-26)

Head = `runs/stage1_turn_v3_trail6.afm`. n = 974 AMI dev turns. The operating point is cross-fitted at ≤ 5 % per-turn
FC (2-fold leave-meetings-out). Miss % [95 % bootstrap CI]; paired differences are paired bootstraps over turns.
Runs: `runs/turn_trail6_hybrid_leakfree.json` (1.04 s tracks) and `runs/archive/turn_trail6_ll032_leakfree.json` (0.32 s tracks).

**Hybrid definition.** `hybrid_stream_<binding>` fires at the earlier of two detectors on the same bound track: the
head at θ (head emission rule) and the deployable timeout at k (armed at the bound track's first active frame, stream
emission rule). The combinator is `conversation.or_outcomes` / `eot_outcomes_or`: fc = fc_a | fc_b, lat = min, and a
pause counts if either detector fires on it.
- **Grid.** θ and k are swept jointly over each detector's pre-end and pause maxima at or above their 85 % quantile,
  plus "never". Only those values can change an FC set.
- **Cross-fit.** Same as for a single threshold (`v2_crossfit` / `_select`): the lowest P50, then P90, chosen on one
  half and applied to the other half.
- **One addition.** Remaining ties go to the lower miss rate (`tie_miss`, eot_bench_or's rule), and only for the
  joint grid. At miss > 50 % every candidate has P50 = P90 = inf. On one threshold axis "first candidate" means the
  lowest threshold, but a joint grid's column order has no meaning. Without the tie-break the choice was the smallest
  k with any θ, and the hybrid scored 73.1 % at 6 s (causal).
- **Chosen (θ, k_frames) per fold** at 6 s, turn FC, 1.04 s tracks:
  - causal: (0.99824, 52) / (0.99813, 49). The pure head's θ was 0.99817 / 0.99808 and the timeout's k 39 / 35.
  - oracle: (0.99651, 48) / (0.9955, 52).
  So the hybrid is the head at nearly its own θ plus a long (≈ 4 s) timeout backstop.
- **Caveat.** Held-out FC is higher for the hybrid: 6.5 % causal vs 5.3 % head and 5.7 % timeout. The union of two
  detectors overshoots more when moved to the other fold.

**A. 1.04 s tracks (the card low-latency config, as §3).** Causal binding:

| system | 6 s all | 6 s floor-open | 6 s taken | 2 s (block A) all | 2 s open | 2 s taken |
|---|---|---|---|---|---|---|
| timeout (deployable) | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 83.6 [80.8, 86.4] | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 |
| head trail6 | 66.5 [63.4, 69.7] | 57.3 [50.9, 63.9] | 69.4 [66.4, 72.7] | 79.8 [77.4, 82.5] | 75.0 [69.4, 80.7] | 81.3 |
| **hybrid trail6** | **61.9 [58.7, 65.0]** | **45.9 [39.3, 52.4]** | **67.0 [63.7, 70.2]** | **79.1 [76.6, 81.9]** | **72.9 [67.7, 78.9]** | 81.1 |
| hybrid v3 (reference) | 70.1 [67.3, 73.0] | 47.4 [41.0, 54.1] | 77.1 | | | |

Oracle binding, 6 s: timeout 32.0 (open 15.2), head 30.3 (21.7), hybrid **28.7 [25.8, 31.6]** (open 17.1). The
label-armed timeout is 29.2 (10.7). Block C at the 2 s emission horizon: hybrid causal 92.5 vs timeout 96.7 and head 92.4.

Paired differences (hybrid minus X, miss points), causal:
- **vs timeout, 6 s:** all **−12.8 [−15.9, −9.8]**, floor-open **−0.2 [−6.7, +5.9]**, taken −16.7 [−19.8, −13.2].
  At 2 s (A): all −13.4 [−16.1, −10.5], open −8.2 [−15.2, −1.7].
- **vs head, 6 s:** all **−4.6 [−6.4, −2.9]**, open −11.4 [−16.5, −6.4], taken −2.4 [−4.1, −0.9]. At 2 s (A): all
  −0.7 [−1.6, +0.2], open −2.1 [−5.0, +0.7].
- Oracle binding, 6 s: vs timeout −3.3 [−5.8, −1.1] (open +1.9 [−3.3, +7.0]); vs head −1.6 [−2.5, −0.8].
- v3 hybrid, causal, 6 s: vs timeout −4.6 [−6.8, −2.6] (open +1.3), vs head v3 −9.5 (open −32.9).

Per-pause FC (only 207 pauses, so noisy; the held-out per-turn FC then reaches about 12 %): hybrid causal 6 s 61.5
(open 33.9), timeout 67.6 (35.5), head 68.9 (59.7).

**Reading (A).** The expectation holds as stated:
- **The hybrid is at least as good as the head overall**, and strictly better at 6 s (−4.6, CI excludes 0).
- **It matches the timeout on floor-open ends** (45.9 vs 46.1).
- **But it does not beat the timeout on floor-open ends:** −0.2 [−6.7, +5.9], and the CI contains 0. On open ends
  the hybrid is essentially the timeout. The head adds nothing there, and at the per-turn operating point the head
  part's θ is set by the taken ends.
- Where the head adds something is taken ends: −16.7 vs the timeout.
- The hybrid is now the best deployable row at 6 s: 61.9 % all, about 46 % open.

**B. 0.32 s tracks** (card "ultra low latency" row: chunk 3, right context 1, FIFO 188, update 144, cache 188; the
card gives the same FIFO / update / cache as the 1.04 s row; encoder left context 188 as before).
- **Preset.** `AOSCConfig.preset("low_latency_032")` / `streaming_diar.SORTFORMER_PRESETS`.
- **Tracks.** 974 extended-window tracks in `data/ami/cache/sortformer/dev_ll032/` (same keys as the 1.04 s set).
  Flush-fixed: window + C + R + 1 = 5 frames of the following meeting audio. MPS, 3 resumable runs, 1110 s.
  `max |MPS − CPU|` = 2.1e-6 (3 tracks).
- **Block A.** The 2 s windows use these tracks cropped. On the 1.04 s set, crop vs the dev cache: max 1.8e-4 (MPS vs
  CPU), 0.1 % of windows > 1e-4.

Track quality (6 s windows; DER pooled over all 4 columns, frame level, no collar, best permutation; primary = the
bound column vs the oracle primary):

| | 1.04 s | 0.32 s |
|---|---|---|
| DER (miss / FA / confusion) | 26.3 (16.4 / 5.0 / 4.9) | 28.3 (17.6 / 5.0 / 5.7) |
| primary miss / FA per speech / post-end active, oracle binding | 16.6 / 25.3 / 17.0 | 17.6 / 25.8 / 17.1 |
| same, causal binding | 32.1 / 69.6 / 37.9 | 32.4 / 69.7 / 37.5 |
| nominal emission delay (mean emit(t) − t) | 10.5 frames (840 ms) | 3.0 frames (240 ms) |
| measured offset lag after the label end, oracle (median / mean frames) | 1 / 3.39 | 1 / 3.27 |
| enrollment agreement, causal vs oracle column at the turn end (all / open) | 70.1 / 77.1 % | 69.7 / 75.0 % |
| causal frames on the oracle column, onset to end / rebinds per turn | 69.4 % / 1.59 | 69.7 % / 1.53 |
| first_active vs oracle | 33.0 % | 34.8 % |

Eot-bench v2 on the 0.32 s tracks (6 s horizon; block A 2 s in the last columns):

| system | 6 s all | 6 s open | 6 s taken | held-out FC | P50 | 2 s (A) all | 2 s open |
|---|---|---|---|---|---|---|---|
| timeout, causal | 70.1 [67.2, 72.9] | **36.5 [30.3, 42.9]** | 80.3 | **8.4 %** | inf | 91.0 | 78.8 |
| head trail6, causal | 65.8 [62.6, 68.7] | 55.0 [48.8, 61.3] | 69.1 | 5.5 % | inf | 82.1 | 80.7 |
| hybrid, causal | **63.7 [60.5, 67.0]** | 42.1 [35.6, 48.8] | 70.2 | 7.3 % | inf | 81.3 | 77.9 |
| timeout, oracle | 34.4 [31.4, 37.6] | 17.3 | 39.5 | 5.5 % | 2720 ms (1.04 s: 3200) | 90.3 | 88.3 |
| timeout, oracle, label arming | 28.2 [25.4, 31.0] | 8.6 | 34.2 | 4.5 % | 2320 ms (3120) | 65.5 | 52.3 |
| head trail6, oracle | 28.3 [25.3, 31.2] | 20.4 | 30.8 | 5.8 % | 2320 ms (2960) | 52.8 | 57.9 |
| hybrid, oracle | 26.8 [23.9, 29.7] | 16.7 | 29.8 | 6.0 % | 2320 ms | 51.9 | 56.7 |

Paired, 0.32 s, causal, 6 s: hybrid − timeout −6.4 [−9.1, −3.8] (open **+5.6 [−0.5, +11.5]**); hybrid − head −2.1
[−4.5, +0.2] (open −12.9 [−18.6, −7.6]). Chosen hybrid (θ, k) causal: (0.99827, 52) / (0.99832, 38).

**Reading (B).**
- **Latency.** The 0.32 s setting cuts the nominal diarizer emission delay by 600 ms (840 → 240 ms). Wherever
  anything fires by the median, dead air falls by the same order: P50 at 6 s goes 3200 → 2720 ms (oracle timeout),
  3120 → 2320 ms (label-armed), and 2960 → 2320 ms (head, oracle).
- **Track quality.** It is slightly worse (DER +2.0 points, mostly miss and confusion) but not by 10 points.
  Primary-track statistics and post-end activity are unchanged.
- **Miss.** The miss rates stay within the 1.04 s CIs: head causal 65.8 vs 66.5, head oracle 28.3 vs 30.3, hybrid
  causal 63.7 vs 61.9. The 2 s rows improve because firings arrive earlier (label-armed timeout 76.4 → 65.5).
- **Timeout caveat.** The causal timeout's apparent gain (70.1 vs 74.8, open 36.5 vs 46.1) comes with a held-out FC
  of 8.4 %: the fold-1 threshold (k = 29) overshoots on fold 0. Read it as the FC budget moving, not as a
  like-for-like gain.
- **IDEAS2 item 3 pass rule** (miss within the 1.04 s CI while the delay falls by about 500 ms): **met.** 0.32 s is
  a free latency win, not a miss win.
- **Enrollment hypothesis: rejected.** Lower lag does not make causal binding better: agreement at the turn end is
  69.7 vs 70.1 % (open 75.0 vs 77.1). The causal − oracle gap stays at about 37 points (hybrid). Enrollment, not
  diarizer lag, remains the dominant loss.
- **Not run.** Head v3 on the 0.32 s tracks, and training on 0.32 s tracks (trail6 was trained on 1.04 s tracks, so
  the head sees a train/test lag mismatch here).

Commands (S = scratch dir):
```
# A (report only; reuses the trail6 work dir)
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $S/trail6/work --v2-tag trail6 --v2-hybrid \
    --n-boot 1000 --ckpt runs/stage1_turn_v3_trail6.afm --out runs/turn_trail6_hybrid_leakfree.json
# B tracks (MPS, x3 until "0 left"; --out only names the process for the machine's GPU pgrep check, nothing is written)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 .venv/bin/python scripts/eval_stage1.py --bench v2 \
    --v2-stage tracks --v2-work $S/hyb6/work --v2-diar-config low_latency_032 --v2-tracks-dir data/ami/cache/sortformer/dev_ll032 \
    --device mps --v2-budget 500 --out $S/hyb6/make_sortformer_tracks_ll032.unused.json
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work $S/hyb6/work --v2-tracks-dir data/ami/cache/sortformer/dev_ll032 \
    --v2-tag trail6_ll032 --v2-budget 500 --v2-bindings oracle,causal_dominant,oracle@2s,causal_dominant@2s --ckpt runs/stage1_turn_v3_trail6.afm  # x2
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $S/hyb6/work --v2-tracks-dir data/ami/cache/sortformer/dev_ll032 \
    --v2-diar-config low_latency_032 --v2-tag trail6_ll032 --v2-hybrid --n-boot 1000 --ckpt runs/stage1_turn_v3_trail6.afm \
    --out runs/archive/turn_trail6_ll032_leakfree.json
```
`make_sortformer_tracks.py --diar-config low_latency_032` makes the same preset for the library's windows. It writes
to `<split>_ll032/` and was not run here.

## 8. Voice enrollment: follow the primary by voice (2026-09-26)

Question (§7): enrollment is the dominant loss (causal_dominant agrees with the oracle column at the turn end in 70 %
of turns; hybrid causal − oracle = +33.2 [+29.8, +36.5] points at 6 s on the 1.04 s tracks, ≈ 37 on the 0.32 s ones).
Can a label-free binding that follows the primary's *voice* across diarizer column swaps close part of that gap?
Head = `runs/stage1_turn_v3_trail6.afm`, 1.04 s tracks (the §7 A setting), n = 974, cross-fitted 5 % per-turn FC
operating point, 1000 bootstraps. Run: `runs/turn_trail6_enroll_leakfree.json`. Code: `audioforge/enrollment.py`,
`scripts/eval_stage1.py` stages `embed` / `bind`. Every causal_dominant / oracle row below reproduces §7 exactly.

**Bindings** (`enrollment.enroll_voice`; per-frame primary column, stored per window by the `bind` stage):
- **Embedding.** The model's own speaker head (`heads.spk`, attentive-stats pooling over its layer mix). Its per-frame
  input comes from the plain causal encode (`enrollment.speaker_frames`, att context [70, 1]), computed once per
  extended window (74 s for all 974 on CPU). The 2 s windows use the same features cropped (same start, causal
  encoder).
- **Following** (`follow_voice`). At every frame t, each column with ≥ 8 active frames among [t − 25, t − 1] gets the
  embedding of its active speech in that 2 s look-back. The challenger is the column closest (cosine) to the
  enrollment. It replaces the current column after qualifying on `hold` = 6 consecutive frames:
  - if the current column is speaking: it must beat it by `margin` = 0.1;
  - if the current column is silent: it must reach the current column's own mean similarity since binding, minus
    the margin (a per-window, label-free self level).
  Parameters were fixed a priori; the sensitivity is below.
- **`voice_first`.** Enroll the first column to become active (the "user speaks first" proxy), using its first
  1.5 s (19 frames) of active speech. Until then the binding is that column.
- **`voice_dominant`.** Enroll the column with the most active frames over the first 5 s after the first speech,
  decided at that frame. Before it, the causal_dominant binding. Then as voice_first.
- **`voice_oracle`** (upper bound: oracle identity + voice following). The oracle column until enrolled. The
  enrollment embedding comes from the first 1.5 s of the *labelled* primary speech. Following is label-free.
- **`oracle_reordered`** (control, head rows only). The oracle column fed to the head in the per-frame-binding
  representation (`bound_track`: bound column moved to column 0, prim = 0), which causal_dominant and the voice rows
  use. The static oracle row instead keeps the diarizer's column order with prim = c.
- **Causality.** A decision at frame t reads features of frames ≤ t − 1 (each sees audio ≤ its frame + 1) and track
  probabilities ≤ t, so audio ≤ frame t. The diarizer lag is handled by the emission rule, as for every binding.
  Tests (`tests/test_enrollment.py`):
  - future tracks, features and audio (through a tiny causal SpeechModel) never change an earlier binding;
  - mutated labels leave voice_first / voice_dominant bit-identical, and voice_oracle moves;
  - hysteresis behaves as specified;
  - agreement is correct on a hand-made case.

**Embedding quality (the quick check; labels only define trials).** Within-window trials: enrollment = a speaker's
first 1.5 s of solo speech; test = a later 2 s block with ≥ 0.64 s of speech of one speaker; target iff same speaker.
- **Label masks:** EER **34.4 %** (3639 trials, 1971 target; mean cosine 0.62 target vs 0.51 non-target).
- **Diarizer-column masks** (blocks where one oracle speaker covers ≥ 80 % of the column's active frames): EER
  **37.1 %** (5195 trials).
So the speaker head is weak even in-window (STAGE1: 31 % within meeting on longer segments).

**Agreement with the oracle column at the turn end** (6 s windows; 2 s windows give the same agreement):

| binding | all | floor-open | frames onset→end on oracle column | rebinds / turn (6 s; 2 s) |
|---|---|---|---|---|
| causal_dominant | 70.1 | 77.1 | 69.4 | 1.59; 1.13 |
| first_active (static) | 33.0 | 39.8 | | 0 |
| voice_first | 40.1 | 42.4 | 45.3 | 0.71; 0.59 |
| voice_dominant | 58.5 | 67.4 | 59.6 | 1.42; 1.20 |
| voice_oracle | 95.2 | 97.9 | 95.4 | 0.28; 0.18 |

**Decomposition** (`scratchpad/enroll/decomp.py`):
- **Enrollment identity correct** (c0 = oracle column): voice_first 33.0 %, voice_dominant 46.7 %.
- **Given a correct identity**, following ends on the right column in 96.9 % / 90.1 % of turns. Given a wrong one it
  stays wrong: 12.3 % / 30.8 % agreement.
- causal_dominant's first binding is right in only 33.0 % of turns too. It reaches 70 % because it re-binds to whoever
  dominates after 2 s of silence, and on AMI that is usually the speaker whose turn is ending (57.3 % agreement even
  after a wrong first binding).
- voice_oracle enrolls in 77.0 % of turns (the others have < 1.5 s of primary speech before the window ends).
- Within-turn column swaps: the whole-turn oracle column differs from the column carrying the primary's last 2 s of
  speech in 8.4 % of turns. That is the most that voice following could fix under a known identity.

**Eot-bench v2 tables.** Miss % [95 % CI] at the cross-fitted ≤ 5 % per-turn FC point. "6 s" = block C (extended
windows, 75-frame emission horizon), "2 s" = block A (default windows).

| system | binding | 6 s all | 6 s open | 6 s taken | held-out FC | 2 s all | 2 s open | 2 s taken |
|---|---|---|---|---|---|---|---|---|
| timeout | causal_dominant | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 83.6 | 5.7 | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 |
| timeout | voice_first | 95.6 [94.2, 96.8] | 91.1 [87.1, 94.7] | 97.0 | 4.8 | 98.4 [97.5, 99.1] | 95.5 [92.6, 97.8] | 99.3 |
| timeout | voice_dominant | 88.4 [86.2, 90.2] | 75.3 [69.7, 80.6] | 92.6 | 3.7 | 98.3 [97.5, 99.0] | 97.8 [95.7, 99.6] | 98.4 |
| timeout | voice_oracle | 42.8 [39.6, 46.0] | 21.0 [15.8, 26.6] | 49.6 | 4.8 | 93.3 [91.6, 94.8] | 90.4 [86.3, 94.0] | 94.2 |
| timeout | oracle | 32.0 [29.1, 35.0] | 15.2 [10.6, 20.3] | 37.2 | 4.8 | 87.2 [84.9, 89.1] | 83.4 [78.1, 88.3] | 88.3 |
| head | causal_dominant | 66.5 [63.4, 69.7] | 57.3 [50.9, 63.9] | 69.4 | 5.3 | 79.8 [77.4, 82.5] | 75.0 [69.4, 80.7] | 81.3 |
| head | voice_first | 88.5 [86.2, 90.7] | 83.2 [78.3, 88.0] | 90.2 | 5.2 | 96.3 [95.0, 97.5] | 94.1 [90.8, 97.2] | 97.0 |
| head | voice_dominant | 85.5 [83.1, 87.9] | 77.6 [71.9, 83.3] | 88.0 | 5.2 | 95.8 [94.4, 97.2] | 93.3 [90.0, 96.5] | 96.6 |
| head | voice_oracle | 46.0 [42.8, 49.1] | 38.6 [32.1, 44.8] | 48.4 | 4.7 | 79.6 [77.1, 82.2] | 83.0 [77.9, 88.0] | 78.6 |
| head | oracle_reordered | 30.8 [27.8, 33.6] | 22.7 [17.1, 28.2] | 33.2 | 5.2 | 59.4 [56.4, 62.6] | 66.2 [59.5, 72.5] | 57.3 |
| head | oracle | 30.3 [27.3, 33.1] | 21.7 [16.7, 27.2] | 33.0 | 5.4 | 53.8 [50.7, 56.9] | 59.9 [53.7, 66.7] | 51.8 |
| hybrid | causal_dominant | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 67.0 | 6.5 | 79.1 [76.6, 81.9] | 72.9 [67.7, 78.9] | 81.1 |
| hybrid | voice_first | 88.3 [86.1, 90.4] | 81.8 [76.5, 86.7] | 90.3 | 6.2 | 96.3 [94.9, 97.5] | 93.6 [90.1, 96.8] | 97.1 |
| hybrid | voice_dominant | 84.4 [82.0, 86.7] | 73.2 [67.6, 79.0] | 87.9 | 5.5 | 95.8 [94.4, 97.0] | 93.3 [89.9, 96.4] | 96.6 |
| hybrid | voice_oracle | 41.4 [38.1, 44.7] | 21.4 [16.1, 26.7] | 47.7 | 5.1 | 79.6 [77.0, 82.1] | 82.9 [78.0, 87.9] | 78.6 |
| hybrid | oracle_reordered | 30.6 [27.8, 33.5] | 14.6 [10.1, 19.6] | 35.4 | 5.8 | 59.0 [56.0, 62.3] | 65.6 [59.1, 72.0] | 57.0 |
| hybrid | oracle | 28.7 [25.8, 31.6] | 17.1 [12.6, 22.3] | 32.2 | 5.5 | 53.1 [50.0, 56.4] | 57.9 [51.4, 64.5] | 51.7 |

**Paired differences** (miss points, 95 % CI; all / open):

| X − Y | timeout 6 s | head 6 s | hybrid 6 s | timeout 2 s | head 2 s | hybrid 2 s |
|---|---|---|---|---|---|---|
| voice_first − causal_dominant | +20.8 [+17.8, +23.9] / +45.0 | +22.0 [+18.7, +25.2] / +25.9 | +26.4 [+23.1, +29.6] / +35.9 | +5.9 [+4.1, +7.8] / +14.4 | +16.5 [+13.7, +19.2] / +19.1 | +17.1 [+14.2, +19.8] / +20.7 |
| voice_dominant − causal_dominant | +13.6 [+10.6, +16.5] / +29.2 | +19.0 [+16.0, +22.0] / +20.3 | +22.4 [+19.3, +25.5] / +27.3 | +5.8 [+4.0, +7.6] / +16.7 | +16.0 [+13.5, +18.3] / +18.3 | +16.6 [+13.9, +19.0] / +20.3 |
| voice_oracle − causal_dominant | −31.9 [−35.3, −28.6] / −25.1 | −20.5 [−23.9, −17.0] / −18.7 | −20.5 [−24.1, −17.0] / −24.5 | +0.8 [−1.3, +3.0] / +9.3 | −0.2 [−3.0, +2.7] / +8.0 | +0.5 [−2.5, +3.3] / +10.0 |
| voice_first − oracle | +63.5 [+60.4, +66.7] / +75.9 | +58.2 [+54.8, +61.7] / +61.5 | +59.6 [+56.1, +63.0] / +64.7 | +11.2 [+9.0, +13.7] / +12.1 | +42.6 [+39.2, +45.8] / +34.2 | +43.1 [+39.8, +46.3] / +35.7 |
| voice_dominant − oracle | +56.3 [+53.1, +59.4] / +60.1 | +55.2 [+51.8, +58.4] / +55.9 | +55.6 [+52.3, +58.8] / +56.1 | +11.1 [+8.9, +13.5] / +14.4 | +42.0 [+38.7, +45.1] / +33.4 | +42.6 [+39.4, +45.7] / +35.4 |
| voice_oracle − oracle | +10.8 [+8.8, +12.9] / +5.8 | +15.7 [+13.4, +18.3] / +16.9 | +12.8 [+10.0, +15.3] / +4.2 | +6.2 [+4.5, +7.8] / +7.0 | +25.9 [+22.9, +28.9] / +23.1 | +26.5 [+23.5, +29.4] / +25.0 |
| voice_oracle − oracle_reordered | (= row above) | +15.2 [+12.9, +17.9] / +15.9 | +10.8 [+8.6, +13.0] / +6.7 | | +20.3 [+17.6, +23.2] / +16.8 | +20.6 [+17.9, +23.6] / +17.3 |
| oracle_reordered − oracle | 0 | +0.5 [−0.8, +1.7] / +1.0 | +1.9 [−0.1, +4.0] / −2.5 | 0 | +5.6 [+3.6, +7.5] / +6.3 | +5.9 [+3.9, +7.7] / +7.7 |
| causal_dominant − oracle_reordered | | +35.7 [+32.1, +39.0] | +31.3 [+27.8, +34.7] | | +20.4 [+17.3, +23.7] | +20.1 [+16.8, +23.5] |

(At 2 s the voice_first / voice_dominant − voice_oracle differences are +5 (timeout) and +16 to +17 (head, hybrid);
at 6 s they are +40 to +53. All in the JSON.)

**Sensitivity of the follower** (model-free: deployable timeout at 6 s, same cross-fit; `scratchpad/enroll/sens.py`
reproduces the report's rows exactly). Cells: miss all / open / agreement at end / % of turns whose binding
switches after the turn end.

| follower | voice_first | voice_dominant | voice_oracle |
|---|---|---|---|
| default (margin 0.1, hold 6) | 95.6 / 91.1 / 40.1 / 20.0 | 88.4 / 75.3 / 58.5 / 40.7 | 42.8 / 21.0 / 95.2 / 16.6 |
| hold 12 | 95.6 / 90.7 / 38.2 / 12.8 | 85.9 / 69.1 / 57.1 / 32.4 | 37.3 / 18.4 / 97.6 / 11.6 |
| margin 0.2 | 93.7 / 88.3 / 48.0 / 30.7 | 81.2 / 55.7 / 60.3 / 46.8 | 43.4 / 16.0 / 97.2 / 28.3 |
| margin 0.2, hold 12 | 93.6 / 86.4 / 44.3 / 23.4 | 80.8 / 58.6 / 58.5 / 39.0 | 36.5 / 15.6 / 98.4 / 21.8 |
| no switching while the current column is silent | 95.1 / 90.3 / 34.6 / 5.1 | 94.4 / 89.9 / 49.8 / 15.3 | 39.2 / 22.7 / 95.6 / 3.4 |
| never switch (enrollment column only) | 95.0 / 90.7 / 33.0 / 0 | 94.3 / 90.0 / 48.6 / 10.7 | 32.0 / 15.2 / 100 / 0 |

References: causal_dominant 74.8 / 46.1 / 70.1 / 58.9; oracle 32.0 / 15.2 / 100 / 0. No follower setting brings a
voice binding below causal_dominant (label-free) or below the static oracle column (oracle identity).

**Reading.**
- **Voice enrollment does not close the gap: it widens it.** The paired CIs exclude 0 in the wrong direction for
  every system and both horizons:
  - voice_dominant − causal_dominant at 6 s: hybrid **+22.4 [+19.3, +25.5]**, timeout +13.6, head +19.0;
  - voice_first is worse still (+20.8 to +26.4).
  So the share of the 33-point hybrid gap that voice enrollment closes is negative: about −68 % for
  voice_dominant (−58 to −77 % from the paired CI of the numerator).
- **The loss is "who to enroll", not "how to follow".**
  - The deployable identity proxies pick the benchmark's primary in 33 % (first speaker) and 47 % (dominant over
    the first 5 s) of AMI windows.
  - Following then keeps whatever it was given: ≥ 90 % agreement after a correct enrollment, ≤ 31 % after a wrong one.
  - causal_dominant wins because it never commits to an identity. Its "whoever dominated the last 2 s" re-binding
    tracks the speaker whose turn is ending, and that is how the benchmark defines the primary.
  - No embedding can repair a wrong identity. That is why I did **not** port TitaNet-L: even perfect following caps
    voice_dominant's agreement at about its 47 % identity accuracy plus the windows where a wrong enrollment still ends
    on the right column (16 % of all turns with the current follower, fewer with a better one): ≤ ~63 %, below
    causal_dominant's 70 %.
- **Upper bound (oracle identity + voice following).**
  - voice_oracle closes most of the gap relative to causal_dominant: hybrid 41.4 vs 61.9 at 6 s (−20.5
    [−24.1, −17.0], ≈ 62 % of the 33 points), timeout 42.8 vs 74.8.
  - But it is still *worse* than simply keeping the oracle column: +10.8 [+8.8, +12.9] (timeout), +12.8 (hybrid).
  - Cause: with a 34 %-EER embedding the follower switches after the turn end in 16.6 % of turns (someone else speaks
    and is accepted as the user), so the bound column stays active and the timeout never fires.
  - The within-turn swaps it could fix are only 8.4 % of turns. Every setting that switches at all loses to the
    static oracle column. With oracle identity, the best follower is the one that never switches.
  - A stronger embedding (e.g. TitaNet-L; not ported) could only win back part of this +11-point following cost.
- **Representation control** (a side finding about the §3 / §7 head rows):
  - The trail6 head is *not* invariant to the column order: 65 % of never-switching voice_oracle windows score
    differently from the static oracle row, exactly those whose oracle column ≠ 0. causal_dominant is always fed in
    the reordered form.
  - At 6 s the effect is negligible: oracle_reordered − oracle = +0.5 [−0.8, +1.7] (head).
  - At 2 s it is +5.6 [+3.6, +7.5] (head) / +5.9 (hybrid). So about 6 of the 26 points of the block-A head
    causal − oracle gap are the representation, not the binding. Against the like-for-like control the gap is
    +20.4 [+17.3, +23.7] (head, 2 s) and +35.7 [+32.1, +39.0] (head, 6 s).
- **What this means for the enrollment problem.**
  - On AMI turn windows the "user" is whichever participant's turn is scored. A label-free system has no identity
    signal for that beyond recency and dominance, which causal_dominant already exploits.
  - Voice following only pays off when the identity is known and the embedding is strong. The deployment case is an
    enrolled device user. Even there, on these tracks it must beat the static column (8 % swaps) without false
    post-end switches.
  - A benchmark that models the deployed case would need a voiceprint from outside the window (e.g. the speaker's
    earlier turns) instead of the in-window label.

Commands (E = scratch dir; E/work links `tracks` and the four trail6 score dirs of §7):
```
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage embed --v2-work $E/work --v2-budget 500 --ckpt runs/stage1_turn_v3_trail6.afm
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage bind --v2-work $E/work --v2-budget 520 --ckpt runs/stage1_turn_v3_trail6.afm
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work $E/work --v2-tag trail6 --v2-budget 500 --ckpt runs/stage1_turn_v3_trail6.afm \
    --v2-bindings voice_oracle,voice_dominant,voice_first,voice_oracle@2s,voice_dominant@2s,voice_first@2s,oracle_reordered,oracle_reordered@2s   # x4
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $E/work --v2-tag trail6 --v2-hybrid --n-boot 1000 \
    --ckpt runs/stage1_turn_v3_trail6.afm --out runs/turn_trail6_enroll_leakfree.json
```

## 9. Who is the primary speaker: TitaNet-L following, an agent-end identity rule, and enrollment length (2026-09-26)

§8 left two open questions: how much of the upper bound's "following cost" (voice_oracle − oracle = +11 to +13
points) a real speaker embedding wins back, and whether any label-free identity rule can beat causal_dominant. This
section answers both with the ported NVIDIA TitaNet-Large, adds the one identity signal a voice agent actually has
(it knows when its own TTS ended), and bounds the "say hello" UX. Same protocol as §7/§8: head
`runs/stage1_turn_v3_trail6.afm`, 1.04 s tracks, n = 974, cross-fitted ≤ 5 % per-turn FC, 1000 bootstraps, nothing
refit. Run: `runs/turn_trail6_primary_titanet.json`. Every causal_dominant / oracle / §8 voice row reproduces §7/§8
exactly (the §8 score and binding dirs are linked into the work dir).

**Backend** (`enrollment.TitaNetEmbedder`, `recent_embeddings_audio`).
- TitaNet-L (`nemo_import.import_titanet`, 25.3 M params, CC-BY-4.0; BASELINES.md: 8.2 % within-meeting EER on
  AMI) embeds *audio*: for a column and a frame t, the audio of the column's active frames (p > 0.5) among
  [t − 25, t − 1] (2 s look-back, ≥ 8 active frames, as §8) concatenated and embedded as one utterance; the
  enrollment likewise over the enrollment frames. Causal by construction (frame t reads audio and probabilities of
  frames ≤ t − 1; `tests/test_enrollment.py`).
- **Update grid: every 5 frames (400 ms), fixed a priori from cost before any scoring.** TitaNet costs 51 ms per
  embedding batched (149 ms single) on 2 CPU threads; per-frame per-column updates over 974 windows would take
  ~10 h and are not a product setting either. The embedding and the "enough speech" flag of grid frame g are held
  for frames g .. g + 4; the follower still steps every frame with §8's hysteresis (margin 0.1, hold 6 frames), so
  `hold` now means two consecutive updates. (Batch size 8: batch 16 hits a pathological CPU path, 1.3 s per
  embedding.)
- The 2 s windows reuse the extended windows' embeddings cropped (same start, causal; the cached 2 s tracks differ
  from the cropped extended ones by ≤ 1.8e-4 at a 0.5 threshold). Init / choice / enrollment use each window's own
  track as in §8.

**Embedding quality** (same trials as §8: enrollment = a speaker's first 1.5 s of solo speech, test = a later 2 s
block with ≥ 0.64 s of one speaker's speech, same window):

| embedding | label masks | diarizer-column masks (block ≥ 80 % one speaker) |
|---|---|---|
| own speaker head (§8) | 34.4 % EER (3639 trials) | 37.1 % (5195) |
| **TitaNet-L** | **10.2 %** (3598 trials, 1950 target; mean cos 0.46 target / 0.10 non-target) | **19.7 %** (5141, 2535 target; 0.40 / 0.12) |

TitaNet is 3.4× better on clean masks, but the diarizer's column masks cost it 9.5 points (impure columns,
concatenated fragments): in the operating regime the follower works with a ~20 % EER embedding, not 8 %.

**Bindings** (per-frame column, stored by the `bind` stage; `<rule>_titanet` = the §8 rule with TitaNet following):
- `voice_first_titanet`, `voice_dominant_titanet`, `voice_oracle_titanet`: exactly §8's identity rules (first
  active column / dominant over the first 5 s / the labelled primary's column), 1.5 s enrollment, TitaNet following.
- **`after_prev_end`** (product-realistic; the agent knows when it stopped speaking). The "agent" turn is the
  previous turn by another speaker in the window: `agent_end_frame` = the end of the last non-primary label
  speaker's activity run that starts before the primary's onset. **That time comes from the labels only as a
  stand-in for the agent's known TTS end; nothing else about the choice uses labels.** From it, the first
  diarizer column active for ≥ 3 consecutive frames is the primary (`after_prev_end_choice`), decided on the 3rd
  frame; before that, and when the window has no earlier other-speaker turn (fallback, fraction reported) or no
  column becomes active, the binding is causal_dominant. Two followers: `after_prev_end_titanet` (enroll on the
  column's first 19 active frames after the agent end, then TitaNet following) and `after_prev_end_causal`
  (control: the causal_dominant rule seeded at the chosen column, `causal_dominant_from`).
- **`voice_explicit_titanet`** (the "say hello" UX bound): enroll on the true primary's first utterance in the window
  (its first 19 label frames; the timing stands in for the prompt's answer), column = the diarizer column
  dominating those frames, causal_dominant before, TitaNet following.
- **Enrollment length** (coordinator, OUTSIDE.md §4 / arXiv 2601.12769: target-speaker enrollment falls off below
  1.5 s and saturates at 3-5 s): `_e40` (3.2 s) and `_e60` (4.8 s) variants of `voice_explicit_titanet` and
  `after_prev_end_titanet`, identical otherwise (a window with fewer active frames than that never enrolls and
  stays on the chosen column).

**Agreement with the oracle column at the turn end** (6 s windows; the 2 s windows agree the same way):

| binding | all | floor-open | frames onset→end on oracle column | rebinds / turn (6 s; 2 s) |
|---|---|---|---|---|
| causal_dominant | 70.1 | 77.1 | 69.4 | 1.59; 1.13 |
| voice_first | 40.1 | 42.4 | 45.3 | 0.71; 0.58 |
| voice_first_titanet | 34.7 | 39.8 | 40.4 | 0.50; 0.44 |
| voice_dominant | 58.5 | 67.4 | 59.6 | 1.42; 1.20 |
| voice_dominant_titanet | 54.4 | 66.5 | 55.6 | 1.21; 1.04 |
| after_prev_end_causal | 82.7 | 90.2 | 80.5 | 1.78; 1.27 |
| after_prev_end_titanet | 77.8 | 84.8 | 76.9 | 1.23; 1.06 |
| after_prev_end_titanet_e40 | 77.0 | 83.5 | 76.5 | 1.21; 1.05 |
| after_prev_end_titanet_e60 | 77.2 | 83.5 | 76.5 | 1.19; 1.03 |
| after_prev_end_titanet_clean | 77.0 | 83.5 | 76.8 | 1.18; 1.03 |
| voice_explicit_titanet | 76.7 | 77.5 | 71.1 | 1.25; 1.13 |
| voice_explicit_titanet_e40 | 73.6 | 76.3 | 69.8 | 1.31; 1.09 |
| voice_explicit_titanet_e60 | 70.6 | 77.5 | 69.3 | 1.38; 1.08 |
| voice_explicit_titanet_clean | 76.7 | 78.4 | 72.1 | 1.10; 1.01 |
| voice_oracle | 95.2 | 97.9 | 95.4 | 0.28; 0.18 |
| voice_oracle_titanet | 93.5 | 94.5 | 94.1 | 0.19; 0.16 |

**Decomposition** (`scratchpad/primary/decomp9.py`, 6 s windows):
- **Agent turn.** 83.6 % of windows have an earlier other-speaker turn (16.4 % fall back to causal_dominant: the
  window starts with the primary). In 51.6 % the agent's run ends *after* the primary's onset (the previous speaker is
  still talking when the primary starts: overlap / barge-in; median gap onset − agent end = −6.5 frames), and in
  20.5 % after the primary's turn end (the choice then comes too late to matter and the turn is scored on the
  causal_dominant part). A column becomes active after the agent end in 68.8 % of windows (14.8 % none); the choice
  falls a median of 4 frames after the onset and before the turn end in 61 % of the chosen windows (after_prev_end_causal).
- **Identity.** The chosen column is the oracle column in **75.7 %** of the chosen windows (52.1 % of all), against
  33 % (first active), 48 % (voice_dominant_titanet) and 86 % (voice_explicit: the column dominating the primary's
  first utterance; 77 % of windows have 1.5 s of primary speech).
- **Following, given the identity.** With a correct choice the binding ends on the right column in 97.2 % of turns
  (TitaNet) / 96.4 % (causal); with a wrong one in 19.6 % (TitaNet) / 50.9 % (causal: it re-binds after 2 s of
  silence). Net agreement 77.8 % (TitaNet) vs 82.7 % (causal following) vs 70.1 % (causal_dominant).
- **Enrollment fraction.** 1.5 s: 48.5 % of windows (after_prev_end) / 77.0 % (explicit) enroll before the window
  ends; 3.2 s: 27.1 % / 50.3 %; 4.8 s: 15.3 % / 33.5 %; clean spans (3.2 s admitted): 14.7 % / 24.7 %. The rest stay
  on the chosen column (identity only, no following).
- Own-head §8 rows reproduced (voice_first 40.1, voice_dominant 58.5, voice_oracle 95.2). TitaNet identity rules on
  the §8 proxies: voice_first_titanet 32.9 % correct identity, 97.8 % agreement after a correct one, 3.7 % after a
  wrong one.

**Eot-bench v2 tables.** Miss % [95 % CI] at the cross-fitted ≤ 5 % per-turn FC point. "6 s" = block C, "2 s" =
block A. Rows: §8 references, the TitaNet-followed §8 rules, the §9 identity rules with their enrollment-length and
§9b clean-span variants, the bounds.

| system | binding | 6 s all | 6 s open | 6 s taken | held-out FC | 2 s all | 2 s open | 2 s taken |
|---|---|---|---|---|---|---|---|---|
| timeout | causal_dominant | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 83.6 | 5.7 | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 |
| timeout | voice_first | 95.6 [94.2, 96.8] | 91.1 [87.1, 94.7] | 97.0 | 4.8 | 98.4 [97.5, 99.1] | 95.5 [92.6, 97.8] | 99.3 |
| timeout | voice_first_titanet | 94.9 [93.4, 96.3] | 90.7 [86.5, 94.1] | 96.3 | 4.9 | 97.8 [96.9, 98.7] | 95.1 [92.0, 97.8] | 98.7 |
| timeout | voice_dominant | 88.4 [86.2, 90.2] | 75.3 [69.7, 80.6] | 92.6 | 3.7 | 98.3 [97.5, 99.0] | 97.8 [95.7, 99.6] | 98.4 |
| timeout | voice_dominant_titanet | 91.9 [90.1, 93.7] | 84.0 [79.6, 88.3] | 94.6 | 4.5 | 98.5 [97.7, 99.2] | 97.8 [95.7, 99.6] | 98.7 |
| timeout | after_prev_end_causal | 67.6 [64.5, 70.4] | 35.3 [29.1, 42.1] | 77.1 | 6.6 | 92.2 [90.3, 93.9] | 83.1 [77.7, 87.9] | 94.9 |
| timeout | after_prev_end_titanet | 53.7 [50.5, 56.9] | 32.6 [26.9, 39.0] | 60.2 | 6.2 | 94.8 [93.2, 96.2] | 90.7 [86.5, 94.6] | 96.0 |
| timeout | after_prev_end_titanet_e40 | 59.2 [56.3, 62.1] | 37.7 [31.8, 44.2] | 65.8 | 4.7 | 96.4 [95.2, 97.5] | 90.9 [86.6, 94.5] | 98.2 |
| timeout | after_prev_end_titanet_e60 | 59.2 [56.3, 62.1] | 37.7 [31.8, 44.2] | 65.8 | 4.7 | 96.5 [95.4, 97.6] | 91.4 [87.3, 95.0] | 98.2 |
| timeout | after_prev_end_titanet_clean | 59.2 [56.3, 62.1] | 37.7 [31.8, 44.2] | 65.8 | 4.7 | 96.7 [95.5, 97.7] | 91.4 [87.3, 95.0] | 98.3 |
| timeout | voice_explicit_titanet | 52.9 [49.8, 56.2] | 34.9 [28.7, 41.6] | 58.5 | 6.5 | 93.5 [91.9, 95.0] | 88.4 [83.6, 92.4] | 95.1 |
| timeout | voice_explicit_titanet_e40 | 60.8 [57.7, 63.7] | 35.8 [29.7, 42.1] | 68.7 | 5.1 | 93.0 [91.3, 94.6] | 83.7 [78.4, 87.9] | 95.9 |
| timeout | voice_explicit_titanet_e60 | 64.9 [61.9, 67.9] | 38.9 [32.6, 45.2] | 73.1 | 5.2 | 92.7 [91.1, 94.3] | 84.2 [78.9, 88.5] | 95.4 |
| timeout | voice_explicit_titanet_clean | 54.9 [51.8, 58.1] | 38.4 [32.1, 45.1] | 60.1 | 6.2 | 94.8 [93.3, 96.1] | 90.4 [86.5, 94.1] | 96.1 |
| timeout | voice_oracle | 42.8 [39.6, 46.0] | 21.0 [15.8, 26.6] | 49.6 | 4.8 | 93.3 [91.6, 94.8] | 90.4 [86.3, 94.0] | 94.2 |
| timeout | voice_oracle_titanet | 36.0 [32.8, 39.2] | 18.8 [13.6, 24.0] | 41.2 | 5.9 | 90.3 [88.4, 92.2] | 87.3 [82.6, 91.6] | 91.2 |
| timeout | oracle | 32.0 [29.1, 35.0] | 15.2 [10.6, 20.3] | 37.2 | 4.8 | 87.2 [84.9, 89.1] | 83.4 [78.1, 88.3] | 88.3 |
| head | causal_dominant | 66.5 [63.4, 69.7] | 57.3 [50.9, 63.9] | 69.4 | 5.3 | 79.8 [77.4, 82.5] | 75.0 [69.4, 80.7] | 81.3 |
| head | voice_first | 88.5 [86.2, 90.7] | 83.2 [78.3, 88.0] | 90.2 | 5.2 | 96.3 [95.0, 97.5] | 94.1 [90.8, 97.2] | 97.0 |
| head | voice_first_titanet | 88.8 [86.6, 90.9] | 85.1 [80.4, 89.5] | 90.0 | 5.7 | 96.7 [95.5, 97.9] | 95.5 [92.7, 98.1] | 97.1 |
| head | voice_dominant | 85.5 [83.1, 87.9] | 77.6 [71.9, 83.3] | 88.0 | 5.2 | 95.8 [94.4, 97.2] | 93.3 [90.0, 96.5] | 96.6 |
| head | voice_dominant_titanet | 84.9 [82.5, 87.4] | 78.7 [73.5, 84.1] | 86.9 | 5.0 | 95.7 [94.3, 97.0] | 93.8 [90.5, 96.9] | 96.3 |
| head | after_prev_end_causal | 60.5 [57.2, 63.8] | 52.8 [46.3, 59.6] | 62.8 | 5.2 | 75.9 [73.1, 78.9] | 77.5 [71.6, 82.7] | 75.3 |
| head | after_prev_end_titanet | 58.3 [55.3, 61.4] | 54.6 [48.1, 61.3] | 59.5 | 5.1 | 88.0 [85.9, 90.2] | 89.5 [85.5, 93.4] | 87.5 |
| head | after_prev_end_titanet_e40 | 60.0 [57.0, 63.1] | 56.0 [49.5, 62.8] | 61.3 | 4.9 | 89.8 [87.7, 91.8] | 90.4 [86.1, 94.1] | 89.7 |
| head | after_prev_end_titanet_e60 | 59.9 [56.9, 63.1] | 55.0 [48.4, 62.1] | 61.4 | 4.9 | 89.6 [87.6, 91.6] | 89.9 [85.5, 93.8] | 89.5 |
| head | after_prev_end_titanet_clean | 59.9 [56.8, 63.0] | 55.5 [48.6, 62.4] | 61.3 | 4.9 | 89.6 [87.5, 91.6] | 90.4 [86.1, 94.1] | 89.4 |
| head | voice_explicit_titanet | 52.8 [49.6, 56.0] | 49.5 [43.2, 56.4] | 53.8 | 6.5 | 81.7 [79.3, 84.1] | 79.4 [74.2, 84.3] | 82.4 |
| head | voice_explicit_titanet_e40 | 58.8 [55.5, 61.9] | 51.8 [45.5, 58.8] | 60.9 | 5.7 | 80.6 [78.1, 83.2] | 76.6 [71.1, 82.4] | 81.8 |
| head | voice_explicit_titanet_e60 | 61.7 [58.6, 65.0] | 51.6 [45.2, 58.3] | 64.9 | 5.3 | 80.9 [78.4, 83.5] | 77.2 [71.8, 82.7] | 82.1 |
| head | voice_explicit_titanet_clean | 54.6 [51.4, 58.0] | 51.1 [44.5, 57.9] | 55.8 | 6.3 | 84.8 [82.5, 87.1] | 83.6 [79.1, 88.4] | 85.2 |
| head | voice_oracle | 46.0 [42.8, 49.1] | 38.6 [32.1, 44.8] | 48.4 | 4.7 | 79.6 [77.1, 82.2] | 83.0 [77.9, 88.0] | 78.6 |
| head | voice_oracle_titanet | 38.3 [35.2, 41.5] | 31.9 [26.2, 38.1] | 40.3 | 6.5 | 71.6 [68.8, 74.4] | 77.5 [71.9, 83.0] | 69.8 |
| head | oracle_reordered | 30.8 [27.8, 33.6] | 22.7 [17.1, 28.2] | 33.2 | 5.2 | 59.4 [56.4, 62.6] | 66.2 [59.5, 72.5] | 57.3 |
| head | oracle | 30.3 [27.3, 33.1] | 21.7 [16.7, 27.2] | 33.0 | 5.4 | 53.8 [50.7, 56.9] | 59.9 [53.7, 66.7] | 51.8 |
| hybrid | causal_dominant | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 67.0 | 6.5 | 79.1 [76.6, 81.9] | 72.9 [67.7, 78.9] | 81.1 |
| hybrid | voice_first | 88.3 [86.1, 90.4] | 81.8 [76.5, 86.7] | 90.3 | 6.2 | 96.3 [94.9, 97.5] | 93.6 [90.1, 96.8] | 97.1 |
| hybrid | voice_first_titanet | 88.1 [85.9, 90.2] | 82.8 [77.7, 87.7] | 89.8 | 6.8 | 96.8 [95.5, 97.9] | 94.6 [91.3, 97.3] | 97.5 |
| hybrid | voice_dominant | 84.4 [82.0, 86.7] | 73.2 [67.6, 79.0] | 87.9 | 5.5 | 95.8 [94.4, 97.0] | 93.3 [89.9, 96.4] | 96.6 |
| hybrid | voice_dominant_titanet | 84.3 [81.9, 86.8] | 77.8 [72.5, 83.5] | 86.4 | 5.8 | 95.4 [94.1, 96.7] | 92.9 [89.2, 96.1] | 96.2 |
| hybrid | after_prev_end_causal | 56.0 [52.5, 59.1] | 40.3 [33.2, 47.1] | 60.7 | 6.1 | 75.3 [72.4, 78.1] | 75.2 [68.8, 81.2] | 75.3 |
| hybrid | after_prev_end_titanet | 51.6 [48.5, 54.7] | 34.0 [28.0, 40.6] | 57.0 | 6.2 | 87.6 [85.5, 89.8] | 88.5 [84.2, 92.6] | 87.4 |
| hybrid | after_prev_end_titanet_e40 | 54.7 [51.6, 57.9] | 38.3 [32.2, 45.2] | 59.7 | 5.9 | 89.3 [87.2, 91.3] | 89.9 [85.5, 93.6] | 89.1 |
| hybrid | after_prev_end_titanet_e60 | 54.5 [51.5, 57.6] | 37.4 [31.6, 44.3] | 59.7 | 5.9 | 89.2 [87.1, 91.1] | 89.5 [85.0, 93.4] | 89.1 |
| hybrid | after_prev_end_titanet_clean | 54.6 [51.6, 57.6] | 37.9 [31.9, 45.0] | 59.7 | 5.9 | 89.2 [87.2, 91.2] | 89.9 [85.5, 93.6] | 89.0 |
| hybrid | voice_explicit_titanet | 47.1 [43.8, 50.4] | 33.0 [27.1, 39.9] | 51.4 | 7.6 | 81.0 [78.5, 83.5] | 78.0 [72.6, 83.1] | 81.9 |
| hybrid | voice_explicit_titanet_e40 | 52.5 [49.1, 55.8] | 35.4 [29.0, 42.1] | 57.9 | 6.8 | 79.3 [76.7, 82.1] | 74.4 [69.0, 80.4] | 80.8 |
| hybrid | voice_explicit_titanet_e60 | 58.0 [54.7, 61.2] | 40.1 [33.5, 46.6] | 63.5 | 5.8 | 79.8 [77.4, 82.5] | 76.0 [70.8, 81.7] | 81.0 |
| hybrid | voice_explicit_titanet_clean | 49.3 [45.9, 52.7] | 37.6 [31.6, 44.3] | 53.0 | 6.7 | 83.9 [81.6, 86.2] | 82.2 [77.2, 87.3] | 84.4 |
| hybrid | voice_oracle | 41.4 [38.1, 44.7] | 21.4 [16.1, 26.7] | 47.7 | 5.1 | 79.6 [77.0, 82.1] | 82.9 [78.0, 87.9] | 78.6 |
| hybrid | voice_oracle_titanet | 35.1 [31.9, 38.3] | 19.0 [13.7, 24.3] | 40.0 | 6.7 | 71.4 [68.6, 74.1] | 77.4 [71.7, 82.9] | 69.6 |
| hybrid | oracle_reordered | 30.6 [27.8, 33.5] | 14.6 [10.1, 19.6] | 35.4 | 5.8 | 59.0 [56.0, 62.3] | 65.6 [59.1, 72.0] | 57.0 |
| hybrid | oracle | 28.7 [25.8, 31.6] | 17.1 [12.6, 22.3] | 32.2 | 5.5 | 53.1 [50.0, 56.4] | 57.9 [51.4, 64.5] | 51.7 |

**Paired differences** (miss points, 95 % CI; all / open):

| X − Y | timeout 6 s | head 6 s | hybrid 6 s | timeout 2 s | head 2 s | hybrid 2 s |
|---|---|---|---|---|---|---|
| voice_first_titanet − voice_first | -0.7 [-1.5, +0.2] / -0.4 | +0.3 [-1.1, +1.5] / +1.9 | -0.2 [-1.2, +0.9] / +1.0 | -0.5 [-1.2, +0.1] / -0.4 | +0.4 [-0.1, +1.1] / +1.4 | +0.5 [-0.1, +1.2] / +1.0 |
| voice_dominant_titanet − voice_dominant | +3.6 [+2.0, +5.3] / +8.7 | -0.6 [-2.1, +1.1] / +1.1 | -0.0 [-1.9, +2.2] / +4.6 | +0.2 [-0.4, +0.8] / +0.0 | -0.1 [-1.1, +0.9] / +0.5 | -0.3 [-1.2, +0.6] / -0.4 |
| voice_oracle_titanet − voice_oracle | -6.8 [-8.9, -4.8] / -2.2 | -7.7 [-10.3, -5.2] / -6.6 | -6.4 [-8.4, -4.2] / -2.4 | -3.0 [-4.3, -1.9] / -3.1 | -8.1 [-10.9, -5.5] / -5.5 | -8.2 [-10.9, -5.6] / -5.5 |
| voice_dominant_titanet − causal_dominant | +17.2 [+14.3, +20.1] / +38.0 | +18.4 [+15.3, +21.4] / +21.4 | +22.4 [+19.2, +25.8] / +31.9 | +6.0 [+4.2, +7.8] / +16.7 | +15.8 [+13.3, +18.3] / +18.8 | +16.3 [+13.6, +18.7] / +20.0 |
| voice_first_titanet − causal_dominant | +20.2 [+17.1, +23.2] / +44.6 | +22.3 [+19.2, +25.4] / +27.8 | +26.2 [+22.9, +29.6] / +36.9 | +5.3 [+3.5, +7.4] / +14.0 | +16.9 [+14.2, +19.6] / +20.4 | +17.7 [+14.7, +20.3] / +21.6 |
| after_prev_end_causal − causal_dominant | -7.2 [-9.6, -4.9] / -10.8 | -6.0 [-8.7, -3.6] / -4.5 | -5.9 [-8.6, -3.5] / -5.6 | -0.3 [-2.4, +1.6] / +2.0 | -4.0 [-6.6, -1.7] / +2.5 | -3.9 [-6.6, -1.4] / +2.3 |
| after_prev_end_titanet − causal_dominant | -21.0 [-24.1, -17.9] / -13.5 | -8.2 [-10.8, -5.3] / -2.7 | -10.3 [-13.5, -6.9] / -11.9 | +2.3 [+0.4, +4.0] / +9.6 | +8.2 [+5.8, +10.4] / +14.4 | +8.5 [+6.2, +10.9] / +15.6 |
| after_prev_end_titanet − after_prev_end_causal | -13.9 [-17.1, -10.5] / -2.7 | -2.1 [-4.8, +1.2] / +1.8 | -4.3 [-7.7, -0.8] / -6.3 | +2.5 [+0.7, +4.5] / +7.6 | +12.1 [+9.8, +14.5] / +11.9 | +12.4 [+9.9, +14.7] / +13.3 |
| voice_explicit_titanet − causal_dominant | -21.9 [-24.8, -18.8] / -11.2 | -13.7 [-16.5, -10.9] / -7.7 | -14.8 [-17.6, -11.6] / -12.8 | +1.0 [-0.5, +2.8] / +7.3 | +1.8 [-0.4, +4.0] / +4.4 | +1.8 [-0.5, +4.0] / +5.1 |
| voice_oracle_titanet − causal_dominant | -38.8 [-42.1, -35.3] / -27.3 | -28.2 [-31.5, -24.5] / -25.4 | -26.8 [-30.5, -23.2] / -26.9 | -2.2 [-4.6, +0.1] / +6.2 | -8.3 [-11.1, -5.5] / +2.5 | -7.7 [-10.6, -4.8] / +4.4 |
| voice_dominant_titanet − oracle | +59.9 [+56.4, +63.1] / +68.8 | +54.6 [+51.2, +58.0] / +57.0 | +55.6 [+52.3, +59.0] / +60.7 | +11.3 [+9.2, +13.8] / +14.4 | +41.9 [+38.7, +45.0] / +33.9 | +42.3 [+38.9, +45.4] / +35.0 |
| after_prev_end_titanet − oracle | +21.7 [+18.8, +24.8] / +17.3 | +28.0 [+24.8, +31.5] / +32.9 | +22.9 [+19.7, +26.2] / +16.8 | +7.6 [+5.4, +10.0] / +7.3 | +34.2 [+30.7, +37.3] / +29.5 | +34.5 [+31.0, +37.6] / +30.7 |
| after_prev_end_causal − oracle | +35.5 [+32.2, +38.6] / +20.1 | +30.2 [+26.7, +33.5] / +31.1 | +27.3 [+23.8, +30.6] / +23.2 | +5.0 [+3.1, +7.1] / -0.3 | +22.1 [+18.9, +25.2] / +17.6 | +22.1 [+18.6, +25.3] / +17.4 |
| voice_explicit_titanet − oracle | +20.9 [+18.0, +23.6] / +19.7 | +22.5 [+19.6, +25.4] / +27.9 | +18.4 [+15.5, +21.3] / +15.9 | +6.4 [+4.0, +9.0] / +5.0 | +27.9 [+24.6, +31.3] / +19.4 | +27.8 [+24.5, +31.1] / +20.1 |
| voice_oracle_titanet − oracle | +4.0 [+2.4, +5.6] / +3.6 | +8.0 [+5.9, +10.3] / +10.3 | +6.4 [+3.9, +9.0] / +1.8 | +3.1 [+1.6, +4.9] / +3.9 | +17.8 [+14.9, +20.6] / +17.6 | +18.3 [+15.3, +21.0] / +19.5 |
| voice_oracle_titanet − oracle_reordered | - | +7.5 [+5.5, +9.7] / +9.2 | +4.5 [+2.6, +6.5] / +4.3 | - | +12.2 [+9.8, +14.5] / +11.3 | +12.4 [+10.0, +14.9] / +11.8 |
| voice_dominant_titanet − voice_oracle_titanet | +56.0 [+52.4, +59.3] / +65.3 | +46.6 [+43.0, +50.0] / +46.7 | +49.2 [+46.1, +52.7] / +58.8 | +8.2 [+6.2, +10.3] / +10.5 | +24.1 [+21.1, +27.0] / +16.3 | +24.0 [+21.0, +26.9] / +15.5 |
| after_prev_end_titanet − voice_oracle_titanet | +17.7 [+14.7, +20.9] / +13.8 | +20.0 [+17.0, +23.2] / +22.7 | +16.6 [+13.6, +19.5] / +15.0 | +4.5 [+2.5, +6.5] / +3.4 | +16.4 [+13.8, +19.1] / +12.0 | +16.2 [+13.5, +19.0] / +11.2 |
| voice_explicit_titanet − voice_oracle_titanet | +16.9 [+14.2, +19.6] / +16.1 | +14.5 [+11.8, +17.4] / +17.6 | +12.0 [+9.2, +14.7] / +14.1 | +3.2 [+1.1, +5.3] / +1.1 | +10.1 [+7.6, +12.8] / +1.9 | +9.5 [+6.9, +12.2] / +0.6 |
| after_prev_end_titanet_e40 − after_prev_end_titanet | +5.4 [+3.8, +7.2] / +5.2 | +1.7 [+0.7, +2.9] / +1.4 | +3.1 [+1.3, +5.0] / +4.4 | +1.7 [+0.4, +3.0] / +0.2 | +1.9 [+0.6, +3.1] / +0.9 | +1.7 [+0.4, +2.9] / +1.4 |
| after_prev_end_titanet_e40 − causal_dominant | -15.6 [-18.4, -12.4] / -8.4 | -6.4 [-9.0, -3.5] / -1.3 | -7.2 [-10.2, -4.0] / -7.5 | +4.0 [+2.1, +5.7] / +9.8 | +10.0 [+7.6, +12.4] / +15.4 | +10.2 [+7.6, +12.5] / +17.0 |
| after_prev_end_titanet_e40 − oracle | +27.1 [+23.9, +30.3] / +22.5 | +29.8 [+26.7, +33.1] / +34.3 | +26.1 [+22.8, +29.3] / +21.2 | +9.3 [+7.1, +11.7] / +7.5 | +36.1 [+32.6, +39.2] / +30.5 | +36.1 [+32.7, +39.3] / +32.0 |
| after_prev_end_titanet_e60 − after_prev_end_titanet | +5.4 [+3.7, +7.2] / +5.2 | +1.6 [+0.4, +2.8] / +0.5 | +2.9 [+1.0, +4.8] / +3.4 | +1.8 [+0.6, +3.0] / +0.7 | +1.7 [+0.4, +3.0] / +0.5 | +1.6 [+0.3, +2.9] / +0.9 |
| after_prev_end_titanet_e60 − causal_dominant | -15.6 [-18.5, -12.3] / -8.4 | -6.6 [-9.3, -3.7] / -2.2 | -7.4 [-10.4, -4.2] / -8.5 | +4.1 [+2.2, +5.8] / +10.3 | +9.8 [+7.4, +12.1] / +14.9 | +10.1 [+7.6, +12.4] / +16.5 |
| after_prev_end_titanet_e60 − oracle | +27.1 [+23.9, +30.3] / +22.5 | +29.6 [+26.6, +32.9] / +33.4 | +25.8 [+22.6, +29.0] / +20.2 | +9.4 [+7.2, +11.7] / +8.0 | +35.9 [+32.4, +39.1] / +30.0 | +36.0 [+32.5, +39.3] / +31.6 |
| voice_explicit_titanet_e40 − voice_explicit_titanet | +7.9 [+5.3, +10.5] / +0.9 | +6.0 [+3.4, +8.3] / +2.3 | +5.4 [+2.8, +7.9] / +2.3 | -0.6 [-1.9, +0.6] / -4.7 | -1.1 [-3.2, +0.9] / -2.8 | -1.7 [-4.0, +0.5] / -3.6 |
| voice_explicit_titanet_e40 − causal_dominant | -13.9 [-16.5, -11.4] / -10.3 | -7.7 [-10.1, -5.5] / -5.4 | -9.4 [-11.8, -7.1] / -10.5 | +0.5 [-0.7, +1.8] / +2.6 | +0.7 [-0.7, +2.2] / +1.6 | +0.1 [-1.2, +1.6] / +1.5 |
| voice_explicit_titanet_e40 − oracle | +28.8 [+25.7, +31.9] / +20.5 | +28.5 [+25.3, +31.5] / +30.2 | +23.8 [+20.6, +26.8] / +18.2 | +5.8 [+3.4, +8.6] / +0.3 | +26.8 [+23.7, +30.0] / +16.7 | +26.1 [+23.0, +29.4] / +16.6 |
| voice_explicit_titanet_e60 − voice_explicit_titanet | +12.0 [+9.3, +14.8] / +4.0 | +8.9 [+6.3, +11.4] / +2.1 | +10.8 [+8.0, +13.6] / +7.1 | -0.8 [-2.3, +0.7] / -4.2 | -0.8 [-2.9, +1.4] / -2.2 | -1.1 [-3.3, +1.1] / -1.9 |
| voice_explicit_titanet_e60 − causal_dominant | -9.9 [-12.1, -7.7] / -7.2 | -4.8 [-6.8, -2.9] / -5.7 | -4.0 [-6.2, -1.8] / -5.8 | +0.2 [-0.8, +1.2] / +3.1 | +1.1 [-0.1, +2.3] / +2.2 | +0.7 [-0.5, +1.8] / +3.1 |
| voice_explicit_titanet_e60 − oracle | +32.9 [+29.5, +35.9] / +23.7 | +31.4 [+28.0, +34.4] / +29.9 | +29.3 [+26.0, +32.2] / +23.0 | +5.6 [+3.2, +8.5] / +0.8 | +27.2 [+23.9, +30.1] / +17.3 | +26.7 [+23.2, +29.7] / +18.2 |
| after_prev_end_titanet_clean − after_prev_end_titanet_e40 | +0.0 [-0.4, +0.4] / +0.0 | -0.1 [-0.7, +0.4] / -0.5 | -0.1 [-0.7, +0.3] / -0.5 | +0.2 [+0.0, +0.5] / +0.4 | -0.2 [-0.6, +0.0] / +0.0 | -0.1 [-0.3, +0.0] / +0.0 |
| after_prev_end_titanet_clean − after_prev_end_titanet | +5.4 [+3.7, +7.2] / +5.2 | +1.6 [+0.5, +2.7] / +0.9 | +3.0 [+1.1, +4.9] / +3.9 | +1.9 [+0.7, +3.2] / +0.7 | +1.7 [+0.5, +3.0] / +0.9 | +1.6 [+0.3, +2.8] / +1.4 |
| after_prev_end_titanet_clean − causal_dominant | -15.6 [-18.5, -12.3] / -8.4 | -6.6 [-9.3, -3.7] / -1.8 | -7.3 [-10.3, -4.2] / -8.0 | +4.2 [+2.3, +5.9] / +10.3 | +9.8 [+7.5, +12.1] / +15.4 | +10.1 [+7.5, +12.4] / +17.0 |
| after_prev_end_titanet_clean − oracle | +27.1 [+23.9, +30.3] / +22.5 | +29.6 [+26.6, +33.0] / +33.9 | +25.9 [+22.8, +29.1] / +20.7 | +9.5 [+7.3, +11.8] / +8.0 | +35.9 [+32.4, +39.0] / +30.5 | +36.0 [+32.5, +39.3] / +32.0 |
| voice_explicit_titanet_clean − voice_explicit_titanet_e40 | -5.9 [-8.6, -3.2] / +2.6 | -4.1 [-6.5, -1.7] / -0.7 | -3.2 [-5.8, -0.6] / +2.3 | +1.8 [+0.4, +3.3] / +6.7 | +4.2 [+2.2, +6.3] / +7.0 | +4.6 [+2.4, +6.9] / +7.8 |
| voice_explicit_titanet_clean − voice_explicit_titanet | +2.0 [+0.3, +3.7] / +3.5 | +1.9 [+0.4, +3.4] / +1.6 | +2.2 [+0.4, +4.0] / +4.6 | +1.2 [+0.2, +2.4] / +2.0 | +3.1 [+1.7, +4.5] / +4.2 | +2.9 [+1.5, +4.2] / +4.2 |
| voice_explicit_titanet_clean − causal_dominant | -19.8 [-23.1, -16.6] / -7.7 | -11.8 [-14.6, -8.8] / -6.1 | -12.6 [-15.6, -9.5] / -8.3 | +2.3 [+0.5, +4.0] / +9.3 | +5.0 [+2.7, +7.1] / +8.6 | +4.7 [+2.5, +7.0] / +9.3 |
| voice_explicit_titanet_clean − oracle | +22.9 [+19.9, +25.7] / +23.2 | +24.4 [+21.5, +27.4] / +29.5 | +20.6 [+17.7, +23.5] / +20.5 | +7.6 [+5.2, +10.1] / +7.0 | +31.0 [+27.9, +34.1] / +23.6 | +30.7 [+27.5, +33.9] / +24.3 |

Key strata (all / open / taken, 6 s): after_prev_end_titanet − causal_dominant, hybrid −10.3 [−13.5, −6.9] /
−11.9 [−18.1, −5.8] / −10.0 [−13.7, −6.2]; timeout −21.0 [−24.1, −17.9] / −13.5 [−19.5, −7.8] / −23.4;
after_prev_end_causal − causal_dominant, hybrid −5.9 [−8.6, −3.5] / −5.6 [−10.4, −0.8] / −6.3; at 2 s
after_prev_end_causal − causal_dominant hybrid −3.9 [−6.6, −1.4] / +2.3 [−4.0, +8.8] / −5.8 [−8.6, −3.1], while
after_prev_end_titanet − causal_dominant is +8.5 [+6.2, +10.9] / +15.6 [+10.1, +21.7] / +6.3. Held-out FC of the
after_prev_end rows: 6.2 % (hybrid, timeout) vs 6.5 / 5.7 % for causal_dominant.

**Reading.**
- **A real embedding recovers most of the following cost, not all of it.** voice_oracle_titanet − voice_oracle =
  −6.8 [−8.9, −4.8] (timeout), −7.7 (head), −6.4 (hybrid) at 6 s; the residual over the static oracle column falls
  from +10.8 / +15.7 / +12.8 (§8) to **+4.0 [+2.4, +5.6] / +8.0 / +6.4**, i.e. 55-65 % of the following cost is won
  back, and post-end switches fall (rebinds 0.19 vs 0.28 per turn). At 2 s the head/hybrid residual is still +18
  (from +26): the follower's first update after a swap comes 8 + 5 + 6 frames late by construction.
- **The §8 identity proxies stay broken with TitaNet.** voice_dominant_titanet − causal_dominant = +22.4 [+19.2,
  +25.8] (hybrid, 6 s), the same as with the own head (−0.0 [−1.9, +2.2] between the two backends); voice_first_titanet
  +26.2. Agreement 54 % / 35 %. §8's arithmetic holds: the identity is right in 48 % / 33 % of windows and no follower
  repairs a wrong identity (agreement 19 % / 4 % after a wrong choice).
- **The agent-end rule is the first label-free binding that beats causal_dominant, and it does so at 6 s with the
  CI excluding 0 in every stratum and system**: hybrid **51.6 vs 61.9** (−10.3 [−13.5, −6.9]; open 34.0 vs 45.9,
  −11.9 [−18.1, −5.8]), timeout 53.7 vs 74.8 (−21.0), head 58.3 vs 66.5 (−8.2). Its identity is right in 76 % of
  the chosen windows because "the first voice after the agent stops" is the benchmark's primary far more often than
  "the first / dominant voice in the window".
- **Most of that gain needs no embedding.** after_prev_end_causal (the choice, then the causal_dominant rule; no
  TitaNet, no per-frame cost) is −5.9 [−8.6, −3.5] (hybrid, 6 s), −7.2 (timeout), and it is the only new binding that
  is also better at 2 s (−3.9 [−6.6, −1.4], hybrid; taken −5.8). TitaNet following adds −4.3 [−7.7, −0.8] (hybrid,
  6 s) / −13.9 (timeout) on top, but costs **+12.4 [+9.9, +14.7] at 2 s** (hybrid; +8.5 vs causal_dominant): the
  voice-followed column stays or becomes active after the end more often within the first 2 s (the 5-frame update
  grid plus `hold` delay every switch by ≥ 1 s, and a wrong post-end switch keeps the timeout from firing), so the
  same turns are caught between 2 and 6 s. For dead air the voice follower is a loss; for the 6 s miss rate a gain.
- **Longer enrollment is worse here, for an operational reason, not the literature's.** e40 − e19 = +3.1 [+1.3, +5.0]
  (after_prev_end, hybrid 6 s), +5.4 [+2.8, +7.9] (explicit); e60 +2.9 / +10.8; at 2 s the explicit rows are within
  noise (−1.7 [−4.0, +0.5]). Waiting for 3.2 / 4.8 s of the chosen column's speech means enrolling in 27 / 15 % of the
  windows instead of 48 % (explicit: 50 / 34 % instead of 77 %), and an un-enrolled window keeps a static column
  through every swap. OUTSIDE.md §4's cliff is about embedding quality at a fixed enrollment; on ≤ 20 s turn windows
  the availability cost dominates it. The identity choice itself is unchanged (same c0), so the rows differ only by
  following: e40 vs e60 are within 0.2 points everywhere.
- **Explicit enrollment (the "say hello" UX) bounds what a prompt buys:** hybrid 47.1 [43.8, 50.4] at 6 s (−14.8 vs
  causal_dominant; open 33.0), 2 s 81.0 (+1.8 [−0.5, +4.0], no gain), agreement 76.7 %, still +18.4 [+15.5, +21.3]
  above the oracle column: it enrolls the right speaker (86 %) but only 77 % of windows have the 1.5 s, and the
  follower's post-end behaviour is the same as above.
- **Remaining gap to the oracle** (hybrid, 6 s): causal_dominant +33.2 → after_prev_end_titanet **+22.9 [+19.7,
  +26.2]**, after_prev_end_causal +27.3, voice_explicit_titanet +18.4; voice_oracle_titanet +6.4 marks what perfect
  identity would leave. About a third of the label-free loss is closed by the agent-end signal; the rest is
  windows without a usable agent end (31 %), wrong first-active choices (24 % of the chosen), and post-end following.
- **Caveats.** (i) The agent end is a label stand-in; in the product it is exact, and the user's turn follows it by
  construction, whereas here it ends after the primary's onset in half the windows (overlap) and after the turn end
  in a fifth. (ii) The rule was fixed before scoring (min_run 3, 1.5 s, stride 5); no follower parameter was tuned on
  these results. (iii) The 2 s windows use the extended windows' embeddings cropped. (iv) Held-out FC is 6.2 % for
  the winning rows vs 6.5 % (hybrid) / 5.7 % (timeout) for causal_dominant, so the comparison is not bought with FC.

**Verdict.** Ship the arming, not (yet) the following: `after_agent` arming of the primary at the agent's TTS end is
the one label-free change that lowers misses with the CI excluding 0 (−5.9 to −7.2 points at 6 s with the causal
rule, −10 to −21 with TitaNet following), while keeping the 5 % FC point. TitaNet following should be enabled only
where the 6 s miss rate matters more than dead air, until its post-end switching is fixed (a no-switch-while-silent
setting, §8, is the first thing to test). The server implements both modes with TitaNet following (`--enroll
after_agent | explicit`, §INTEGRATION 7); an arming-only mode is the next item.

### 9b. Clean-span masks for enrollment and following (VAD > 0.9, no overlap, spans ≥ 0.56 s, consistent-mean pooling)

Coordinator request after §9: admit only frames that are column-active AND the model's own VAD (the trail6
checkpoint's `vad` head, causal encode) > 0.9 AND have exactly one active column; use contiguous admitted spans of
≥ 7 frames (0.56 s) only; embed each span separately and pool a set of span embeddings by `consistent_mean` (drop the
span with the lowest mean cosine to the others while it is < 0.3, majority kept) instead of concatenating everything;
enrollment on the chosen column's first admitted spans totalling 40 frames (3.2 s, the e40 row's duration). Column
choice and follower unchanged (`enrollment.clean_mask / spans_of / consistent_mean / recent_embeddings_spans /
enroll_spans`, modes `after_prev_end_titanet_clean`, `voice_explicit_titanet_clean`; stage `--v2-embedder
titanet_clean`). Parameters fixed a priori.

- **EER on the clean masks** (same trial definition, but only trials whose enrollment and test blocks have admitted
  spans survive, so the trial set is about half of §9's and not the same trials): label masks **8.0 %** (1877 trials,
  956 target; cos 0.53 / 0.12) and diarizer-column masks **9.3 %** (1925 trials, 961 target; 0.52 / 0.12), vs 10.2 /
  19.7 % on the plain masks. The column-mask penalty (+9.5 points) disappears almost entirely: the impurity that hurt
  the follower sits in the overlap and low-VAD frames the clean mask removes.
- **Bindings.** after_prev_end_titanet_clean is indistinguishable from e40 (−0.1 [−0.7, +0.3] hybrid 6 s; same 59.2 /
  54.6 / 59.9 rows; enrolled in 14.7 % of windows vs 27.1 %). voice_explicit_titanet_clean beats e40 (−3.2 [−5.8,
  −0.6] hybrid 6 s; 49.3 vs 52.5; agreement 76.7 vs 73.6, rebinds 1.10 vs 1.31) but stays behind the 1.5 s plain row
  (+2.2 [+0.4, +4.0]) and, at 2 s, behind both (+4.6 vs e40, +2.9 vs e19). Agreement after a wrong identity drops to
  0 % (the clean follower never rescues a wrong choice, it only holds a right one: 97.2 %).
- **Reading.** Cleaner masks make the embedding better (EER 19.7 → 9.3 % on column masks) and the follower steadier,
  but at equal admitted duration they enroll even later and more rarely (15-25 % of windows), and that availability
  loss is what the 6 s rows measure. The masks are the right input for the follower once enrolled; the enrollment
  budget should be the plain 1.5 s (or "whatever is there by the first pause"), not 3.2 s of clean speech.

Commands (S = `scratchpad/primary`; S/work links `tracks`, `spkfeat`, the §7 score dirs and the §8 score / bind dirs):
```
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage embed --v2-embedder titanet --v2-work $S/work --v2-budget 480 --ckpt runs/stage1_turn_v3_trail6.afm  # x10 (spkemb_titanet/)
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage bind  --v2-embedder titanet --v2-work $S/work --v2-budget 480 --ckpt runs/stage1_turn_v3_trail6.afm  # x3; --v2-bindings for the _e40/_e60/_clean modes
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage embed --v2-embedder titanet_clean --v2-work $S/work --v2-budget 480 --ckpt runs/stage1_turn_v3_trail6.afm  # §9b (vad/, spkemb_titanet_clean/)
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage scores --v2-work $S/work --v2-tag trail6 --v2-budget 480 --ckpt runs/stage1_turn_v3_trail6.afm \
    --v2-bindings <mode>[,<mode>@2s,...]   # 12 + 8 + 4 bindings, ~250 s each
.venv/bin/python scripts/eval_stage1.py --bench v2 --v2-stage report --v2-work $S/work --v2-tag trail6 --v2-hybrid --n-boot 1000 \
    --ckpt runs/stage1_turn_v3_trail6.afm --out runs/turn_trail6_primary_titanet.json
python $S/eer_titanet.py $S/work 480 [clean]; python $S/summ9.py runs/turn_trail6_primary_titanet.json; python $S/decomp9.py $S/work
```
