# Turn head v3: why it works with oracle activity and fails with the Sortformer track (2026-09-26)

Setup: the same 200 AMI dev turns as research/archive/STAGE1.md (turn head v3 table). Model `runs/stage1_turn_v3.afm`. The streaming
Sortformer tracks come from the cache in `data/ami/cache/sortformer/dev`, with the enrollment column as in `scripts/eval_stage1.py`. Every
system is scored at its own threshold for ≤ 5 % false cutoffs (FC), recomputed per turn exactly as `eot_bench` /
`eot_bench_emit` do it (the replay asserts the same FC rate, miss rate and P50). Inputs derived from the diarizer use
the streaming emission rule. Code: `scripts/turn_error_analysis.py`. Numbers: `runs/turn_v3_error_analysis.json`,
which includes a per-turn table.

Reproduced: head + oracle gives 5.8 % miss, P50 560 ms. Head + stream gives 62.6 % miss (threshold 0.99218). Timeout on the stream track gives 38.4 %, P50 2640 ms.

## 1. Track quality of the streaming primary column (per turn)

| feature | P25 | P50 | P75 | P90 | mean |
|---|---|---|---|---|---|
| miss inside turn | 0 | 5.7 % | 18.4 % | 31.1 % | 11.5 % |
| primary "on" in post-end window | 0 | 7.4 % | 31.7 % | 88 % | 22.5 % |
| longest miss run inside turn (frames of 80 ms) | 0 | 1 | 7 | 15 | 4.7 |
| end lag (frames the column stays on after the true end) | 0 | 1 | 3 | 15 | 2.5 |
| onset lag (frames) | −1 | 0 | 1 | 2 | −1.2 |

- 116 of 200 turns are **clean** after the end: the primary column is off from end + 2 frames onward. The other 84 have post-end FA.
- Post-end FA is the **next speaker landing in the primary's slot**. In 95 % of the FA turns an oracle
  next speaker talks in the window. The median share of FA frames that overlap the next speaker's speech is 100 %. In AMI 84 % of
  windows have another speaker after the end, and 55 % have one talking at the last primary frame, because the "floor" turns overlap.
- Lag is not the problem: the median end lag is 1 frame. The track errors are dropouts inside the turn and leakage after the end.

## 2. Outcomes at the ≤ 5 % FC operating points (fire / miss / FC counts, n = 200)

**Head (stream) × timeout (stream), same turns:**

| head \ timeout | fire | miss | FC |
|---|---|---|---|
| fire (71) | 63 | 6 | 2 |
| miss (119) | **51** | 66 | 2 |
| FC (10) | 3 | 1 | 6 |

**Post-end FA share of the primary column × outcome:**

| post-end FA | n | head: fire / miss / FC | timeout: fire / miss / FC | oracle-fed head: fire / miss / FC |
|---|---|---|---|---|
| 0 | 66 | 37 / 25 / 4 | 58 / 5 / 3 | — |
| (0, .25] | 78 | 33 / 39 / 6 | 56 / 16 / 6 | — |
| (.25, .5] | 18 | 1 / 17 / 0 | 2 / 16 / 0 | — |
| > .5 | 38 | 0 / 38 / 0 | 1 / 36 / 1 | — |
| **clean (off by end + 2)** | 116 | 60 / **49** / 7 | 97 / **12** / 7 | 99 / 9 / 8 |
| **post-end FA** | 84 | 11 / 70 / 3 | 20 / 61 / 3 | 80 / 2 / 2 |

**Longest SF silence inside the turn × outcome (head | timeout):** 0–2 frames: 38/68/0 | 58/48/0. 3–5: 8/11/0 | 14/5/0.
6–10: 7/16/0 | 16/7/0. 11–20: 16/22/4 | 29/13/0. > 20: 2/2/6 | 0/0/**10**.

(a) **With a clean post-end window the head still misses 49 of its 109 turns that are not cut off (45 %). The timeout misses 12
(11 %).** So the head has a problem of its own. Those 49 misses do reach high scores: 41 of them exceed the oracle-fed threshold of 0.947,
and their median max post-end score is 0.983. But the stream-fed threshold is 0.992. The threshold is pushed that high by the pre-end side (§4).

(b) With post-end FA the timeout also misses (61 of 81). It cannot fire until the column goes silent. The timeout's 38 % comes from the
**clean turns**: in 54 turns the timeout fires and the head does not, and 41 of those are clean. The head's median max post-end score in
those turns is 0.984, below its threshold. The timeout, in turn, is almost immune to dropouts inside the turn. Its 20-frame threshold ignores
every SF silence of 20 frames or less. Only 10 turns have a longer one, and those are exactly its 10 FCs, 9 of them true hesitations. The head wins
in only 8 turns.

(c) The oracle-fed head fires in 111 turns where the stream-fed head does not (105 misses, 6 FCs). Feeding the oracle primary with
Sortformer's other columns (hybrid below) recovers 102 of the 111. Inputs in those 111 turns compared with the 68 turns where both fire: the SF primary
column is on for 38 % vs 11 % of the first 13 post-end frames. Another SF column shows the next speaker for 33 % vs 59 % of those frames, so the next speaker sits in the
primary's slot instead of its own. Frames since the SF primary was last on, 6 frames after the end: 4.1 vs 6.5. Coverage in the last 13 turn frames is the same
(0.78 in both).

## 3. Which input's noise hurts: hybrid and decomposed inputs

enc = the track fed to the frozen speaker-kernel encoder. The text state is not speaker-conditioned, so the encoder is the only path. prim / oth =
the head's own primary track and other columns.

| variant (enc / prim / oth) | miss @≤5 % FC | P50 ms | threshold |
|---|---|---|---|
| oracle O/O/O (turn-chunk emission) | 5.8 % | 560 | 0.947 |
| oracle O/O/O, streaming emission delay | 5.8 % | 1280 | 0.948 |
| **O/O/S: oracle primary + SF other columns** | **9.5 %** | 1520 | 0.968 |
| **O/S/S: encoder sees oracle, head's inputs all SF** | **9.5 %** | 1520 | 0.968 |
| S/S/O: SF primary + oracle other columns | 60.5 % | inf | 0.990 |
| S/O/O: encoder sees SF, head's inputs all oracle | 62.1 % | inf | 0.990 |
| stream S/S/S | 62.6 % | inf | 0.992 |
| SF with causal 8-frame hangover on the primary (all inputs) / on the encoder input only | 51.1 % / 51.1 % | inf | 0.975 / 0.971 |
| SF, primary fixed to oracle **before** the end only | 29.5 % | 1840 | 0.968 |
| SF, primary fixed to oracle **after** the end only | 46.8 % | 2800 | 0.992 |
| timeout (same track): fixed before / after the end | 34.2 % / **14.2 %** | 2400 / 2400 | 20 / 20 frames |

**The damage enters almost entirely through the speaker-kernel conditioning of the encoder.** The encoder is frozen since the stage-1
pretraining on clean or flip-noised oracle activity, and v3 trained only the head. With the oracle track in the kernels the head copes with fully noisy
Sortformer head inputs (9.5 %). With the SF track in the kernels, clean head inputs do not help (62 %). The other columns cost only 3.7 points
(oracle 11 misses → 18). The two error types also affect the two systems in opposite ways. The head is hurt most by
**dropouts inside the turn** (fixing them: 63 → 30 %). The timeout is hurt most by **post-end leakage** (fixing it: 38 → 14 %).

## 4. Score distributions and threshold pressure

Max score per turn (counts, n = 200):

```
bin            oracle pre  stream pre | oracle post  stream post
[0,0.5)            125        102     |      1           10
[0.5,0.8)           29         26     |      1           18
[0.8,0.9)           26         19     |      6            6
[0.9,0.95)          10         13     |      4           17
[0.95,0.98)          2         18     |     16           26
[0.98,0.99)          5          9     |     29           28
[0.99,0.995)         3          5     |     65           44
>=0.995              0          8     |     78           51
```

- With the SF track, 40 turns score above 0.95 **before** the end, against 10 with the oracle. 35 of the 40 have an SF miss run of 5 or more frames
  (≥ 400 ms) inside the turn. Only 19 contain a true hesitation. Those spurious silences, fed through the kernels, make the encoder
  report "primary silent", and the head's pre-end score rises. At the same time post-end scores fall: 95 turns reach ≥ 0.99, against 143 with the oracle. The
  threshold is therefore forced up to 0.992, and the clean turns fall under it.
- Miss rate vs allowed FC (stream head): 5 % → 62.6 %, 7.5 % → 49.7 %, 10 % → 45.3 %, 15 % → 32.8 % (P50 1920 ms),
  20 % → 31.3 %. The curve is not driven by a handful of outliers. It degrades smoothly, so dropping a few hesitation turns would not fix it.
- The FC turns are hesitations. Oracle-fed head: 10 of 10 FC turns contain a hesitation, median longest pause 1320 ms. Stream-fed head:
  8 of 10, median 1320 ms (the other 2 are SF dropouts of 17 and 28 frames). Timeout: 9 of 10, median 1480 ms. Among turns without an FC, about 10 %
  contain any hesitation. Per-turn calibration has a trivial upper bound: 94.5 % of turns have their own max post-end score above their max pre-end score. No causal
  signal identifies those turns, so this bound is not a usable lever.

## 5. Simulated diarizer quality ("DER-to-miss" curve)

Each contiguous error run of each column is replaced by the oracle with probability q. The oracle is first mapped into Sortformer's column layout. All
inputs, the kernel included, come from the repaired track. Frame error = (miss + FA) / oracle speech frames at the fixed mapping.

| repaired q | frame err | prim miss | prim post-end on | head miss / P50 | timeout miss / P50 |
|---|---|---|---|---|---|
| 0 (Sortformer stream) | 41.8 % | 15.2 % | 21.9 % | 62.6 % / inf | 38.4 % / 2640 |
| 25 % | 31.0 % | 10.5 % | 17.0 % | 56.8 % / inf | 31.8 % / 2560 |
| 50 % | 21.2 % | 7.0 % | 12.8 % | 41.6 % / 2400 | 29.7 % / 2560 |
| 75 % | 10.3 % | 3.5 % | 5.7 % | 24.7 % / 1840 | 18.9 % / 2400 |
| 100 % (oracle in SF layout, soft values kept) | 0 | 0 | 0 | 10.0 % / 1360 | 6.3 % / 2240 |

The head's miss rate at ≤ 5 % FC is higher than the timeout's at **every** track quality. Its only advantage is latency, and that appears
only at a frame error of about 10 % or less (q ≥ 75 %: 1840 vs 2400 ms). A diarizer about 4 times better than streaming Sortformer on AMI would
still leave the current architecture behind the cascade on misses. The residual 10.0 % vs 5.8 % at q = 100 % comes from the soft, non-binary
input values and the column layout (hybrid O/O/S shows the same 18 vs 11 misses).

## 6. Decision combinations on the stream track (best over timeout k)

- head AND primary silent ≥ k: 38.4 % (k = 21). This reduces to the timeout.
- head OR silent ≥ k: 37.4 % / 2640 ms (k = 21). About 1 point better than the timeout alone. Not a fix.
- On the oracle track, head OR silent ≥ 24 frames gives **1.6 % miss at P50 560 ms**, against 5.8 % for the head alone. That is worth keeping for good tracks.

## 7. Recommendations (ranked by expected effect on stream-track miss @≤5 % FC, now 62.6 % vs timeout 38.4 %)

1. **Train the speaker-kernel path on real streaming tracks.** Unfreeze the encoder's speaker-kernel parameters (layers 0 and 2) or the whole
   conditioning branch, with p_ext on the cached streaming tracks and labels from the oracle. Alternatively, remove kernel conditioning (mode `concat`) so the track
   enters only the head, which §3 shows is tolerant: O/S/S gives 9.5 %. Evidence: 62 % → 9.5 % when only the kernel input is cleaned. That is the upper bound of what a
   robust kernel can recover. The realistic target is to beat the 38 % timeout. **Highest expected effect, one training run.**
2. **Make ext_noise match the real error statistics, aimed at the kernel.** Real in-turn dropouts reach 7 frames at P75 and 15 at P90, against
   the training drop_max of 4 frames. Real post-end leakage covers more than 32 % of the post-end window in 25 % of turns, and it is the next speaker's speech. Use drop_max of about 20 and leakage copied
   from the next speaker's oracle activity. Only useful together with (1), because the head alone already saw these tracks with p_ext 0.9.
3. **Cheap inference-time patch:** feed the kernels an 8-frame causal hangover of the SF primary. It gives 62.6 → 51.1 % with no retraining. It is
   not enough alone, and hangover on the timeout's own track is worse.
4. **Decision rule:** ship head OR timeout (k about 21–24 frames). It is free, gives −1 point on SF and −4 points on the oracle track, and at worst equals the cascade.
   Allowing 10 % FC instead of 5 % cuts misses to 45 %, which still loses.
5. **Better diarizer:** worth it only after (1). With the current kernel, even frame error of 10 % (q = 75 %) leaves the head at 24.7 % vs the timeout's 18.9 %.
   The error type that matters for the head is **in-turn dropouts**. Fixing them gives 63 → 30 %, while fixing post-end leakage gives 63 → 47 %.
   Improving recall inside the turn helps the head. Suppressing next-speaker leakage helps the timeout (38 → 14 %).

Caveats: n = 200 turns from 4 dev meetings, one checkpoint, one seed. Enrollment uses the oracle overlap inside the turn, as in the eval. The
repaired-track curve fixes whole error runs at random, which is not a model of any real diarizer. The hybrid and "fix" variants look at the oracle and are
diagnostics, not systems.

## 8. Hybrid decision rule, joint (θ, k) sweep (2026-09-26)

The rule is: fire at the first frame where the head's p > θ **or** the primary has been silent for ≥ k frames. This is the plain label-free timeout
(`serve.py` `turn_policy: "hybrid"`). Each path keeps its own emission rule. On oracle activity the head uses the turn chunk (2 frames) and the timeout has zero look-ahead. On the stream track both use the streaming rule.
`conversation.eot_bench_or` sweeps θ over eot_bench's own candidates (plus "never") and k over every silence value (plus "never"). It picks the
point with eot_bench's rule: lowest P50, then P90, subject to FC ≤ 5 %. Remaining ties go to the lower miss rate. Setup: same 200 AMI dev turns, `runs/stage1_turn_v3.afm`,
existing cached streaming tracks (not the regenerated ones). Numbers: `runs/turn_v3_hybrid_n200.json`. All stored pure rows reproduce bit for bit, and
the pure rows re-derived from the joint sweep are asserted equal to them.

| input (column binding) | system | θ | k (frames) | P50 / P90 ms | miss | FC turn / pause |
|---|---|---|---|---|---|---|
| oracle activity | head | 0.947 | — | 560 / 1440 | 5.8 % | 5.0 % / 33 % |
| | timeout | — | 18 | 1440 / 1440 | 6.3 % | 5.0 % / 30 % |
| | **hybrid** | 0.947 | 24 | **560 / 1440** | **1.6 %** | 5.0 % / 33 % |
| SF stream (oracle overlap) | head | 0.992 | — | inf / inf | 62.6 % | 5.0 % / 9 % |
| | timeout | — | 21 | 2640 / inf | 38.4 % | 5.0 % / 21 % |
| | **hybrid** | 0.997 | 21 | 2640 / inf | **37.4 %** | 5.0 % / 21 % |
| SF stream (causal_dominant, label-free) | head | 0.996 | — | inf / inf | 91.6 % | 5.0 % / 6 % |
| | timeout | — | 26 | inf / inf | 88.4 % | 5.0 % / 6 % |
| | **hybrid** | 0.998 | 26 | inf / inf | **86.8 %** | 5.0 % / 6 % |

**≤ 5 % per-pause FC.** Only 33 within-turn pauses (≥ 0.3 s) exist in the 200 turns, so this budget allows one pause. Oracle: hybrid 11.6 % miss at 1920 ms
(θ 0.994, k 24) vs timeout 18.6 % / 1920 and head 46.7 % / 2000. SF stream: hybrid 88.9 %, timeout 95.5 %, head 90.9 % (all P50 inf).
causal_dominant: hybrid 91.2 %, timeout 94.3 %, head 95.9 %.

**Grid** (fixed points, not FC-constrained): miss (per-turn FC), oracle | SF stream (oracle overlap). A point is admissible at ≤ 5 % FC only if its FC is ≤ 5 %.

| k \ θ | 0.90 | 0.95 | 0.98 | 0.99 |
|---|---|---|---|---|
| 13 | 0.0 (15.0) \| 18.0 (30.5) | 0.0 (12.5) \| 20.4 (26.5) | 0.0 (12.5) \| 23.2 (22.5) | 0.0 (12.5) \| 25.0 (22.0) |
| 18 | 1.1 (12.0) \| 20.8 (28.0) | 1.1 (8.0) \| 24.4 (22.0) | 1.1 (7.0) \| 27.1 (15.0) | 3.2 (5.5) \| 29.9 (11.5) |
| 21 | 1.1 (11.0) \| 21.1 (26.5) | 1.6 (6.5) \| 26.4 (20.5) | 2.7 (5.5) \| 29.1 (12.5) | 5.7 (3.5) \| 33.9 (8.5) |
| 24 | 1.1 (10.0) \| 21.8 (26.5) | **1.6 (5.0)** \| 27.7 (20.5) | 2.6 (4.0) \| 32.4 (12.0) | 6.1 (1.5) \| 38.6 (8.0) |
| 30 | 4.4 (10.0) \| 23.1 (26.5) | 6.3 (5.0) \| 31.3 (20.0) | 14.1 (4.0) \| 40.7 (11.5) | 28.4 (1.5) \| 53.0 (7.5) |

On the stream track no cell of this grid reaches ≤ 5 % FC. The admissible hybrid needs θ ≈ 0.997, so the head adds almost nothing there.

**Does hybrid ever lose to the timeout?** Per turn, at a fixed k, it never fires later than the timeout alone. But the head can add pre-end
firings, so at a fixed k its FC is ≥ the timeout's (for example k 21 on the stream: 5 % → 8.5 % at θ 0.99). Once θ and k are re-tuned jointly
for the same FC budget, it never loses, by construction: the pure timeout is a point of the sweep. At the chosen points it equals the timeout's P50/P90 or beats it, with lower miss in all
3 conditions and under both FC units. The gain is large only on a clean track: −4.7 points vs the timeout on the oracle, with the head's 560 ms P50 kept.
On streaming Sortformer it is −1.0 point (oracle overlap) and −1.6 points (label-free). This confirms §6 and recommendation 4. The serving default stays
`timeout`, and `hybrid` is available as an opt-in.

Commands (≈ 345 s each, CPU, 2 threads):
`.venv/bin/python scripts/eval_stage1.py --ckpt runs/stage1_turn_v3.afm --n 200 --tasks turn --diar-ckpt runs/nemo_sortformer_v2.afm --diar-cache data/ami/cache/sortformer/dev --hybrid [--enroll causal_dominant] --out <json>`
(the two outputs are merged into `runs/turn_v3_hybrid_n200.json`; the causal run is stored under `turn_enroll_causal_dominant`). Tests: `tests/test_hybrid_policy.py`.
Caveats: n = 200, one checkpoint, and operating points are chosen in-sample, so the joint sweep has more freedom than the single-threshold rows. The re-scoring on the
regenerated Sortformer tracks is pending.
