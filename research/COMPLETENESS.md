# smart-turn x our single model: learned fusion and a completeness head on the shared encoder

2026-09-26. Two ways of using pipecat-ai smart-turn to make OUR model better, both measured on eot-bench v2
(research/EOT_BENCH_V2.md; the same 974 AMI dev turn windows, scorer, meeting folds, <= 5 % per-turn false cutoffs,
6 s / 2 s horizons, floor-open / taken strata and bootstrap CIs as research/BASELINES.md). Code:
`scripts/bench_completeness.py` (+ `bench_completeness_fusion.py`), `audioforge/heads/completeness.py`,
`audioforge/datasets/smartturn.py`, `research/recipes/stage1_completeness.yaml`; tests `tests/test_completeness.py`; numbers
`runs/completeness.json`.

Scorecard rows this can move: "Turn end, floor-open ends" (ours 45.9 % missed vs Silero timeout 26.7 %) and, for Part B,
whether a completeness head on the frozen encoder matches a dedicated model (smart-turn v3.2) at zero extra encoder cost.

## 1. Licence check (Part B, step 1)

Sources fetched 2026-09-26 (Hugging Face API / raw files, GitHub raw files).

| artefact | licence statement | verbatim |
|---|---|---|
| smart-turn code + model (github.com/pipecat-ai/smart-turn `LICENSE`; HF `pipecat-ai/smart-turn-v3` card `license: bsd-2-clause`) | BSD-2-Clause | "BSD 2-Clause License / Copyright (c) 2024–2025, Daily / Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: ..." |
| `pipecat-ai/smart-turn-data-v3.2-train` (270 946 clips, 41.4 GB), `-v3.2-test` (31 527, 4.8 GB), `-v3.1-*`, `-v3-*` dataset cards | **no `license:` field in the YAML front matter, no LICENSE file in the repo** (top level: `data/`, `.gitattributes`, `README.md`) | body of the v3.2-train card, in full apart from the 39 FreeSound links: "# Training dataset for Smart Turn v3.2. / Thank you to the following contributors whose audio samples are included in this dataset: * The Pipecat team * Liva AI * Midcentury * MundoAI / Also, thank you to the following people for the CC-0 background noise sample data which has been used in this dataset: [39 freesound.org links]" |
| smart-turn README (GitHub, main) | code/model only | "This is a truly open model (BSD 2-clause license). Anyone can use, fork, and contribute to this project." and "The training code will download datasets from the pipecat-ai HuggingFace repository." No data licence. |
| `docs/data_generation_contribution_guide.md` | contributors grant rights to Daily; nothing granted downstream | "Smart Turn is a **fully open model**, and we release all datasets publicly: https://huggingface.co/pipecat-ai/datasets" ... "By contributing, you confirm you own the recordings, the speakers consent to public release of their voice, and grant us the rights to redistribute." |
| HF API listing of the org's 14 datasets (`?author=pipecat-ai`) | only ONE dataset carries a licence tag | `pipecat-ai/human_5_all`: `license: bsd-2-clause` (3 862 clips, `audio` + `endpoint_bool`, 312 MB, single parquet). The others (`rime_2`, `human_convcollector_1`, `orpheus_*_1`, `chirp3_1`, `smart-turn-data-v3*`, `stt-benchmark-data`): no licence tag; e.g. `human_convcollector_1` card = YAML dataset_info only (722 clips), no licence. |

Reading. The v3.x training/test sets are published without any licence: "we release all datasets publicly" is a
statement of availability, not a grant of rights, and the contributor clause grants redistribution rights to Daily
only. So the v3.2 audio was NOT downloaded (only its metadata columns `id`, `language`, `endpoint_bool`, `dataset`,
`synthetic` were read over HTTP range requests to align splits, see below). `human_5_all` IS licensed
(BSD-2-Clause on the card), is human speech (its clips appear in v3.2 with `synthetic = False`, `language = eng`) and
is balanced (1 931 complete / 1 931 incomplete), so Part B proceeds on it alone.

Split alignment (metadata only). v3.2-train's `dataset` column contains `human_5` (about 43 of the 3 265 rows in shard 0,
all English) and v3.2-test contains `human_5` too (35 of 3 153 rows in shard 0). Smart-turn v3.2 therefore trained on
most of human_5_all. Our eval split is the human_5 clips that sit in **v3.2-test** (`data/smartturn/v32_human5_ids.json`,
matched by clip basename), so neither model saw them; if the ids cannot be matched the loader falls back to a seeded
stratified 15 % split and the comparison is flagged as contaminated in smart-turn's favour.

## 2. Part A: learned fusion on AMI (CPU)

Inputs are the per-window score streams already computed for BASELINES.md (nothing recomputed; `--stage extract`):
our trail6 head p (Sortformer causal_dominant binding), our Sortformer-primary silence run, the Silero/Pipecat
silence run, smart-turn v3.2 P(complete) held from Silero triggers and from Sortformer-primary triggers (+ trigger
indicators). Features per frame: logit(head), log-clipped silence runs (64 frames), the two held smart-turn values and
their "held" indicators. Model: logistic regression (C = 1) or a gradient-boosted classifier (200 x depth 3), frames
[onset, end + horizon), label = post-end, pre- and post-end halves of each turn weighted equally. **Cross-fitted**: the
model for fold f is fitted on the other meeting fold, so every window's fused score is out-of-fold; the <= 5 % FC
threshold is then cross-fitted on the same folds exactly as for every other row. Emission of a fused decision = that of
its slowest input (Sortformer rule + 160 ms chunk when the head or the primary timeout is in; t + 1 for the
speaker-unaware set). "+silero_timeout" = the fused score OR the Silero timeout on the joint (θ, k) grid, i.e. the
fusion in the place of the head inside the hybrid.

### 2.1 Results, 6 s horizon (block C), miss % at the cross-fitted <= 5 % per-turn FC point [95 % CI]

| system | all | open (236) | taken (738) | FC turn / pause | Δ vs hybrid all / open | Δ vs head-OR-Silero all / open |
|---|---|---|---|---|---|---|
| hybrid trail6 (ours, ref) | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 67.0 | 6.5 / 5.3 | ref | +2.4 [+0.5, +4.2] / +11.6 [+6.1, +17.2] |
| head OR Silero timeout (exploratory ref) | 59.6 [56.3, 62.6] | 34.3 [28.6, 40.4] | 67.3 | 6.6 / 6.3 | −2.4 [−4.2, −0.5] / −11.6 [−17.2, −6.1] | ref |
| Silero VAD timeout | 72.7 [69.8, 75.4] | 26.7 [21.1, 32.4] | 86.8 | 4.8 / 5.3 | | |
| fusion logit, all 5 streams | 65.3 [62.1, 68.3] | 41.1 [34.8, 47.6] | 72.7 | 4.8 / 9.7 | +3.4 [+0.3, +6.3] / −4.8 [−11.4, +1.5] | +5.7 [+2.9, +8.5] / +6.8 [+0.5, +12.7] |
| fusion logit, all + Silero timeout (OR) | 63.0 [59.6, 65.9] | 29.5 [23.1, 35.8] | 72.9 | 6.8 / 10.6 | +1.1 [−2.2, +4.2] / −16.4 [−23.8, −9.4] | +3.4 [+0.6, +6.2] / −4.8 [−10.6, +0.6] |
| fusion GBM, all 5 streams | 63.2 [59.9, 66.2] | 28.6 [22.4, 34.8] | 73.5 | 6.4 / 13.5 | +1.2 [−2.3, +4.6] / −17.3 [−25.4, −9.7] | +3.6 [+0.5, +6.7] / −5.7 [−12.2, +0.4] |
| fusion GBM, all + Silero timeout (OR) | 62.9 [59.5, 65.9] | 26.7 [20.7, 32.6] | 73.7 | 6.6 / 14.0 | +0.9 [−2.6, +4.3] / −19.2 [−26.9, −12.2] | +3.3 [+0.2, +6.3] / **−7.6 [−13.7, −2.2]** |
| fusion GBM, head + Silero run only (no smart-turn) | 64.8 [61.4, 67.8] | 24.8 [18.7, 30.9] | 76.8 | 5.3 / 13.0 | +2.8 [−0.6, +6.2] / −21.1 [−28.4, −14.0] | +5.2 [+2.2, +8.0] / −9.5 [−15.1, −4.2] |
| fusion logit, head + Silero run only | 65.8 [62.5, 68.7] | 39.1 [32.6, 45.9] | 73.7 | 5.9 / 13.0 | +3.9 [+0.8, +6.8] / −6.8 [−13.6, −0.3] | +6.2 [+3.4, +9.1] / +4.8 [−2.0, +11.2] |
| fusion GBM, no head (timeouts + smart-turn) | 68.5 [65.4, 71.5] | 27.8 [21.7, 34.0] | 80.5 | 7.0 / 11.6 | +6.6 [+2.9, +9.8] / −18.1 [−26.1, −10.9] | +9.0 [+5.7, +11.9] / −6.5 [−13.0, +0.1] |
| fusion logit, speaker-unaware (Silero run + smart-turn) | 72.8 [69.8, 75.5] | 26.3 [20.5, 32.2] | 86.8 | 5.1 / 6.3 | +10.9 / −19.6 | +13.3 / −8.0; vs Silero timeout +0.1 [−0.8, +1.0] / −0.4 [−3.0, +2.0] |
| fusion GBM, speaker-unaware | 72.6 [69.6, 75.4] | 29.2 [23.1, 35.1] | 85.5 | 6.3 / 7.7 | +10.7 / −16.7 | +13.1 / −5.1; vs Silero timeout −0.1 [−1.9, +1.6] / +2.5 [−3.1, +7.6] |

2 s horizon (block A, default windows): every fusion is within noise of, or worse than, head-OR-Silero on all ends
(e.g. GBM all: 84.9 vs 83.3 all, 43.7 vs 40.3 open) and the speaker-unaware fusions equal the Silero timeout; full
rows in `runs/completeness.json["fusion"]["blocks"]["A_default_windows_all"]`.

### 2.2 Reading and kill line

- **Kill line ("beat head-OR-Silero on open ends, CI excluding 0, at equal or lower FC").** One row passes it
  literally: GBM-all OR Silero timeout, open 26.7 vs 34.3 (−7.6 [−13.7, −2.2]) at FC 6.6 = 6.6. But it is not an
  improvement: (i) on all ends it is *worse* than head-OR-Silero (+3.3 [+0.2, +6.3]) because it gives back 6 points on
  floor-taken ends (73.7 vs 67.3); (ii) on open ends it exactly equals the plain Silero timeout (26.7 vs 26.7, −0.1
  [−6.1, +6.2]); (iii) the same open-end number is reached *without any smart-turn feature* (GBM head + Silero run:
  24.8, −9.5 [−15.1, −4.2] vs head-OR-Silero). The fused detectors move along the open/taken trade-off that the
  <= 5 % FC budget imposes; none of them extends the frontier. The GBM rows also spend 11-14 % per-pause FC (vs 5-6 %
  for the references): they learn to fire inside hesitations.
- **smart-turn adds nothing that a silence run does not already carry.** Speaker-unaware fusion of the Silero run +
  smart-turn = the Silero timeout to within 0.1 point (both models), reproducing BASELINES.md's "smart-turn + timeout
  = timeout". With the head present, adding the smart-turn streams changes open-end misses by −3.8 (GBM) / +2.0
  (logit) points with CIs far wider than that.
- **The logistic fits say why** (out-of-fold coefficients on unit-scaled features, both folds,
  `runs/completeness.json ... fits`). All 5 streams: Silero silence run +1.62 / +1.25, head logit +0.55 / +0.54,
  primary-silence run −0.49 / −0.29 (redundant with the head, which already counts it), Silero-triggered smart-turn
  value +0.22 / +0.17 and its trigger indicator +0.20 / +0.39, Sortformer-triggered value **−0.57 / −0.44** and
  indicator −0.24 / −0.50. Speaker-unaware set: Silero run +2.19 / +1.91, the bare "a Pipecat VAD stop happened"
  indicator +0.42 / +0.63, and the smart-turn probability itself **−0.08 / −0.06**. A silence trigger is informative;
  the value smart-turn attaches to it is not, and on Sortformer-primary silences it is anti-informative (it scores the
  primary's pauses as complete more often than the primary's ends).

### 2.3 Error complementarity on floor-open ends (6 s horizon, 236 turns)

"smart-turn says complete" = its own decision (P >= 0.5) at a trigger whose decision time falls within 1 s after the
true end. Systems at their cross-fitted <= 5 % FC operating points.

| | Silero-triggered smart-turn | Sortformer-triggered smart-turn |
|---|---|---|
| open ends where smart-turn says complete within 1 s (base rate) | 110 / 236 = 46.6 % | 40.2 % (either trigger: 53.8 %) |
| ... but at that P >= 0.5 it also said complete at a pre-end trigger (a cutoff) | 48.7 % of open turns are cut; only 24.6 % are "complete within 1 s AND no cutoff" | 47.0 % cut; 24.6 % clean |
| open ends our hybrid misses (100): smart-turn says complete within 1 s | 55 / 100 = 55 % (86 had a trigger) | 55 % (either: 61 %) |
| open ends head-OR-Silero misses (73): smart-turn says complete within 1 s | 40 / 73 = 54.8 % (65 had a trigger) | 50.7 % (either: 61.6 %) |
| open ends the head alone misses (126) | 66 / 126 = 52.4 % | 49.2 % |
| converse: open ends where smart-turn does NOT say complete within 1 s (126 / 141): hybrid catches within 1 s / within 6 s | 0 % / 53.2 % | 0 % / 57.4 % |
| the same for head-OR-Silero | 0 % / 60.3 % | 0 % / 61.7 % |
| the same for the head alone | 0 % / 42.9 % | 0 % / 44.7 % |

Reading: smart-turn's hit rate on the turns we miss (55 %) is its base rate on all open ends (47 %) plus a little: the
two systems' errors are close to independent (no anti-correlation that a gate could exploit), so on paper a perfect
gate would recover about half of our open-end misses (55 of the hybrid's 100, 40 of head-OR-Silero's 73). The catch is the first row's cost: at the P >= 0.5 that produces those hits smart-turn also cuts off 49 % of
open turns before their end (48.7 %), which is why the <= 5 % FC budget pushes its threshold to ~0.99 and its
standalone miss rate to 99 %. There is no threshold at which its post-end "complete" decisions are separable from its
pre-end ones on this audio (AUC 0.58, BASELINES.md); the fusion rows above are the measured version of that: the
learned combination cannot spend smart-turn's hits without also buying its cutoffs. The converse cell (we catch 0 %
of anything within 1 s) is the diarizer's 840 ms emission delay plus the timeout backstop, the known latency floor of
our track (EOT_BENCH_V2 §7).

### 2.4 Dynamic timeout instead of OR (`--stage dyntimeout`)

Policy: fire when the silence run >= clamp(T0 − Σ a_i s_i, Tmin, Tmax) frames; s = the trail6 head posterior
("head"), smart-turn's held P(complete) ("st"), or both; silence = the Silero any-speaker run or our Sortformer-primary
run. Since s >= 0 the upper clamp only matters below T0, so Tmax = T0 and the grid is T0 ∈ {0.7 .. 6 s, 10 values} x
Tmin ∈ {2, 4, 7, 10} x a ∈ {0 .. 55} frames (both: a_head x a_st ∈ {0, 10, 20, 35, 50}²). Selection = the scorer's own
operating-point rule (lowest P50, then P90, then miss, subject to <= 5 % per-turn FC) over the whole grid on the OTHER
meeting fold, applied to this fold; emission = the slowest input's (Sortformer rule + 160 ms chunk whenever the head
is read; t + 1 for Silero + smart-turn only). Tmin = 2 was chosen in every fold and horizon: the floor never binds.

| policy (silence | s) | 6 s all | 6 s open | 6 s taken | FC turn / pause | P50 open | 2 s (block A) all | 2 s open | fitted (fold 0 / fold 1: T0 frames, a) |
|---|---|---|---|---|---|---|---|---|
| Silero timeout (ref) | 72.7 | 26.7 [21.1, 32.4] | 86.8 | 4.8 / 5.3 | 2000 ms | 83.5 | 39.6 [32.7, 45.7] | k ≈ 23–24 |
| head OR Silero (ref) | 59.6 | 34.3 [28.6, 40.4] | 67.3 | 6.6 / 6.3 | 2800 ms | 78.3 | 58.1 [51.2, 64.7] | |
| **Silero | head** | 71.5 [68.3, 74.3] | **23.0 [17.5, 28.7]** | 86.2 | 4.2 / 7.7 | 2640 ms | 80.5 [78.0, 83.0] | **30.9 [24.4, 37.1]** | 75, a 55 / 37, a 15 |
| Silero | st | 73.9 | 27.9 [22.0, 33.8] | 87.7 | 4.4 / 3.9 | 2000 ms | 83.9 | 41.5 [35.1, 47.6] | 30, a 10 / 24, a 0 (= the plain timeout) |
| Silero | head + st | 69.6 [66.4, 72.4] | **20.5 [15.1, 25.7]** | 84.2 | 5.8 / 13.0 | 2400 ms | 79.3 | **27.5 [21.3, 33.7]** | 55, a_head 35, a_st 0 / 75, 50, 10 |
| primary | head | 74.0 | 45.3 | 83.0 | 4.6 / 3.9 | 3840 ms | 91.9 | 79.3 | 65, a 30 / 55, a 20 |
| primary | st | 75.7 | 48.1 | 84.2 | 6.9 / 3.9 | 4400 ms | 93.2 | 84.2 | 45, a 5 / 45, a 15 |
| primary | head + st | 70.8 | 40.0 | 80.3 | 6.2 / 5.3 | 3680 ms | 90.6 | 77.2 | |

Paired differences (points, [95 % CI], same resamples):

| policy | vs Silero timeout 6 s all / open | vs head-OR-Silero 6 s all / open | vs Silero timeout 2 s all / open | vs head-OR-Silero 2 s all / open |
|---|---|---|---|---|
| Silero | head | −1.2 [−2.3, −0.1] / **−3.7 [−7.2, −0.3]** (FC 4.2 vs 4.8) | +11.9 [+8.5, +14.7] / **−11.2 [−17.3, −5.0]** (FC 4.2 vs 6.6) | −3.0 [−4.3, −1.7] / **−8.8 [−13.5, −4.4]** | +2.1 [−0.9, +5.1] / **−27.2 [−34.7, −20.2]** |
| Silero | st | +1.2 [+0.4, +2.0] / +1.2 [−1.4, +3.7] | +14.3 / −6.4 [−12.7, +0.3] | +0.4 [−0.1, +0.8] / +1.8 [+0.4, +3.8] | +5.5 / −16.6 |
| Silero | head + st | −3.1 [−4.7, −1.6] / **−6.2 [−11.0, −1.9]** (FC 5.8 vs 4.8) | +10.1 [+6.6, +13.2] / **−13.8 [−21.1, −6.3]** | −4.2 [−6.1, −2.4] / **−12.1 [−18.1, −6.5]** | +0.9 [−2.3, +4.2] / **−30.6 [−38.4, −22.7]** |
| primary | head | +1.2 / +18.6 | +14.4 / +11.0 | +8.4 / +39.6 | +13.6 / +21.2 |

Reading.
- **A dynamic Silero timeout shortened by the head posterior is the first thing in this note that beats the plain
  Silero timeout on open ends at lower FC**: 23.0 vs 26.7 % (−3.7 [−7.2, −0.3]) at 6 s, 30.9 vs 39.6 % (−8.8 [−13.5,
  −4.4]) at 2 s, per-turn FC 4.2 vs 4.8 %. Against head-OR-Silero it is −11.2 / −27.2 points on open ends at lower FC,
  but it is a *timeout*: it loses the floor-taken ends the OR wins through the head (86.2 vs 67.3 %), so on all ends it
  is +11.9 points worse at 6 s and equal at 2 s (+2.1 [−0.9, +5.1]). The fitted shape is "wait up to 3–6 s, minus up
  to 15–55 frames x p(head)": at p = 0.9 fold 0 waits 75 − 50 = 25 frames (2 s), fold 1 37 − 13 = 24 frames; below
  p ≈ 0.5 it waits longer than the fixed timeout. It is the head's posterior used as a *confidence*, not as a trigger.
- **smart-turn adds nothing here either**: as the only modulator the fitted a is 0 or 10 frames and the policy equals
  the plain timeout (27.9 vs 26.7, +1.2 [+0.4, +2.0]: slightly worse); next to the head it gets a_st = 0 in fold 0 and
  10 in fold 1, and the head + st row's extra 2.5 open points over the head-only row come with FC 5.8 vs 4.2 and
  13 % per-pause FC (it fires inside hesitations), so it is not a like-for-like gain.
- **Modulating our primary-silence run does not help open ends** (45.3 vs 46.1 for the fixed primary timeout): the
  bottleneck there is the diarizer's binding and 840 ms latency, not the wait length (EOT_BENCH_V2 §7).
- **The 2 s horizon exposes the head's input latency**: under the Sortformer emission rule any policy that reads the
  head cannot fire within 25 frames of the end unless T0 <= 10, so in block C at 2 s the head-modulated policies miss
  95–98 % (block A, the default windows, is the 2 s number that matters and is reported above).
- Dead air on open ends: 2.6 s (Silero | head) vs 2.0 s for the fixed timeout, i.e. the extra recall is bought by
  waiting longer on low-confidence ends and shorter on high-confidence ones; P50 of head-OR-Silero is 2.8 s.
- **head OR dynamic timeout** (the dynamic policy in the timeout's place inside the OR, joint (θ, offset) grid
  cross-fitted as for every hybrid): head OR dyn(Silero | head) = 58.2 [54.9, 61.5] all, 31.5 [25.6, 37.5] open,
  66.5 taken, FC 6.1 / 6.3, vs head-OR-Silero −1.3 [−2.4, −0.3] all / −2.8 [−6.1, +0.5] open; at 2 s (block A) 73.2 all,
  29.7 open, vs head-OR-Silero −5.1 [−7.2, −3.0] / −28.4 [−35.0, −22.1] at FC 6.8 vs 6.6. Head OR dyn(Silero | head + st)
  = 55.9 [52.7, 59.1] all, 23.6 [18.1, 29.5] open, 65.9 taken, FC 6.2 / 6.8: vs head-OR-Silero **−3.6 [−5.0, −2.3] all /
  −10.7 [−15.4, −6.5] open**, vs our hybrid −6.0 / −22.3; at 2 s 73.7 all / 48.4 open (−4.7 / −9.7 vs head-OR-Silero).
  This is the best all-ends row in the note (55.9 vs the hybrid's 61.9 and head-OR-Silero's 59.6), and the one place
  smart-turn contributes: a_st = 10 frames in fold 1 (0 in fold 0), so the gain over head OR dyn(Silero | head) (−2.3
  all / −7.9 open) is within one fold's fit and comes with per-pause FC 6.8 vs 6.3. Both rows are selected after
  seeing the table (they need an ICSI confirmation exactly like head-OR-Silero, BASELINES.md).

**Continuous run over the whole windows** (lead 4 s + turn + 6 s trail, 974 windows, 6 s-fitted parameters of each
window's fold; event = rising edge of the fire condition; late = a post-end event after another speaker has started
since the end):

| system | events / window | lead events / window | windows with a cutoff | windows with a post-end event | duplicate post-end events / window (windows with one) | first post-end event is late, all / open / taken |
|---|---|---|---|---|---|---|
| Silero timeout | 0.38 | 0.062 | 3.3 % | 27.5 % | 0.009 (0.9 %) | 12.2 % / 8.5 % / 13.4 % |
| head OR Silero | 0.65 | 0.028 | 5.8 % | 43.5 % | 0.106 (7.9 %) | 29.9 % / 10.2 % / 36.2 % |
| hybrid (head OR primary timeout) | 0.61 | 0.001 | 6.5 % | 41.3 % | 0.108 (8.6 %) | 30.0 % / 10.6 % / 36.2 % |
| dyn Silero | head | 0.37 | 0.032 | 3.0 % | 29.1 % | 0.014 (1.4 %) | 12.6 % / 7.2 % / 14.4 % |
| dyn Silero | head + st | 0.40 | 0.025 | 4.9 % | 30.6 % | 0.021 (2.1 %) | 13.6 % / 7.2 % / 15.6 % |
| dyn primary | head | 0.31 | 0.002 | 4.5 % | 25.9 % | 0.001 (0.1 %) | 14.9 % / 7.6 % / 17.2 % |

Reading: the dynamic Silero policies fire about as often as the fixed timeout (0.37 vs 0.38 events per window) and
far less than the OR rules (0.61–0.65), with 1–2 % of windows re-firing after a detection against 8–9 % for the ORs;
the ORs' extra events are largely head firings on floor-taken ends after the next speaker has started (36 % of their
first post-end events on taken ends, which the OR counts as detections while a product would count them as talking
over the new speaker; on open ends all systems fire late in 7–11 % of windows). Fires in the lead (another speaker's
silence before the primary starts) are 2.5–6 % per window for every Silero-based policy and ~0 for the primary-track
ones: the price of a speaker-unaware silence signal.

## 3. Part B: completeness head on our encoder (from smart-turn's data)

### 3.1 Data and head

- Data: `pipecat-ai/human_5_all` (BSD-2-Clause, §1), 3 862 clips, 4.8 h, p50 4.3 s, p90 6.6 s, max 14 s; decoded once
  to a 1.0 GB float32 cache (`data/smartturn/cache/`). Eval = the 399 human_5 clips found in smart-turn-data-v3.2-test
  (399 of its 402 human_5 ids matched by uuid; 175 complete / 224 incomplete), train = the other 3 463. Utterance end
  = last frame of the energy VAD (LibriSpeech's rule) + 1; the clips carry a median 5 frames (0.4 s) of silence after it.
- Head (`heads/completeness.py`, 124 K parameters): Linear-SiLU -> GRU(96) over the frozen encoder's frames; a frame
  logit "the speech so far is complete" (strictly causal, streams with `step`; test: offline == streaming, frames after
  t do not change logit t) and an utterance logit from the mean of the projected frames over the last 25 frames (2 s)
  before the end plus the GRU state at the end. Loss = utterance BCE + 0.5 x frame BCE (post-end frames carry the
  label, pre-end frames are 0 at weight 0.1).
- Recipe `research/recipes/stage1_completeness.yaml`: init from `runs/stage1_heads_pretrained.afm`, every pretrained tensor
  frozen (lr x0), all other heads weight 0, batch 8, 2000 steps, checkpoints every 250. MPS: 7.5 min (0.22 s / step).
  Trainer log: loss 1.03 -> 0.09–0.87 (noisy at batch 8).

### 3.2 Held-out human_5 clips: our head vs smart-turn v3.2's own ONNX (`--stage human5`, 399 clips)

| system | input at decision | accuracy | ROC AUC | cost |
|---|---|---|---|---|
| smart-turn v3.2 (`smart-turn-v3.2-cpu.onnx`, their inference: last <= 8 s, left zero-pad, Whisper features) | whole clip | **0.970** | **0.994** | 28–36 ms per call, 2 CPU threads |
| ours, utterance logit at the VAD end (no trailing silence seen) | up to the last speech frame | 0.877 | 0.953 | encoder + head 73 ms per clip on MPS |
| ours, utterance logit 3 / 6 / 12 frames after the VAD end | + 0.24 / 0.48 / 0.96 s of silence | | 0.989 / 0.989 / 0.990 | |
| ours, utterance logit at the clip end | whole clip | 0.937 | 0.988 | |
| ours, frame logit P(complete so far) at the VAD end / +2 / +4 / +6 / +9 frames | | 0.61 / 0.70 / 0.79 / 0.84 / 0.86 | 0.874 / 0.952 / 0.988 / 0.994 / 0.996 | **head alone: 0.10 ms per 160 ms chunk on CPU** (the encoder already runs for ASR / VAD / turn) |
| ours, frame logit at the clip end | whole clip | 0.872 | **0.995** | |
| paired bootstrap, AUC(ours utterance @ VAD end) − AUC(smart-turn) | | | −0.042 [−0.064, −0.024] | |

Reading.
- **Given the same input (the clip including its ~0.4 s of trailing silence) the frame-level head matches smart-turn:
  AUC 0.995 vs 0.994, 6 frames (0.48 s) after the last speech frame it is at 0.994.** Its accuracy at the fixed 0.5
  threshold is lower (0.87 vs 0.97: the balance point of a head trained with a soft "not complete yet" prior on pre-end
  frames sits below 0.5 on complete clips), which is a calibration, not a ranking, difference; every downstream use here
  cross-fits the threshold.
- At the VAD end itself, before any silence has been heard, the ranking is weaker (frame AUC 0.87, utterance 0.95):
  about half a second of post-speech context is worth 0.12 AUC to the causal head. Smart-turn never makes that
  decision (it is triggered after 200 ms of silence and sees the whole 8 s buffer).
- Cost: 124 K parameters, 0.10 ms per 160 ms chunk on 2 CPU threads (MPS: 1.2 ms) on top of an encoder that runs
  anyway, against 28–36 ms per smart-turn call (8 M parameters, Whisper features), i.e. the same decision at roughly
  1/300 of the marginal compute, streamed per frame instead of per trigger.
- Caveat: 3.5 k training clips, one English human dataset; smart-turn trained on 271 k clips in 23 languages and was
  measured here on its own test distribution. This is a "same encoder, tiny head, same task" existence result, not a
  multilingual or robustness claim.

### 3.3 The completeness stream on eot-bench v2 (`--stage streams`, then `fusion` / `dyntimeout` again)

The head's per-frame P(complete so far) on the 974 extended AMI windows (frozen encoder, [70, 1] mask, 0.32 s of
following meeting audio, 974 windows in 105 s on MPS), emitted at the 160 ms chunk end; block A = the same track
cropped. Same scorer, folds and CIs as §2.

| system (6 s horizon, block C) | all | open | taken | FC turn / pause | Δ vs head-OR-Silero all / open | Δ vs Silero timeout all / open |
|---|---|---|---|---|---|---|
| cmp alone | 93.9 [92.4, 95.4] | 91.8 [88.2, 95.0] | 94.6 | 3.7 / 5.8 | +34.4 / +57.6 | +21.2 / +65.1 |
| cmp OR Silero timeout | 72.3 [69.2, 75.1] | 24.9 [19.1, 30.6] | 86.2 | 5.5 / 5.3 | +12.7 [+9.5, +15.5] / **−9.4 [−16.0, −2.9]** | −0.4 [−1.5, +0.6] / −1.8 [−4.8, +0.9] |
| head OR cmp | 66.1 [63.1, 69.3] | 56.5 [50.0, 62.8] | 69.0 | 6.2 / 7.7 | +6.5 / +22.2 | |
| fusion logit head + Silero run + cmp | 65.5 | 35.5 [29.0, 42.2] | 74.6 | 5.7 / 14.5 | +5.9 / +1.2 | ; vs the same fusion without cmp −0.2 [−2.2, +1.6] / −3.5 [−7.6, +0.3] |
| fusion GBM head + Silero run + cmp | 61.0 [57.6, 63.9] | 21.1 [15.6, 26.9] | 73.0 | 5.2 / 12.6 | +1.4 [−1.6, +4.4] / **−13.2 [−19.4, −7.4]** | −11.7 / −5.6 [−11.3, +0.1]; vs without cmp **−3.8 [−5.4, −2.2] / −3.6 [−6.6, −0.8]** |
| fusion GBM all 5 + cmp, OR Silero | 63.8 | 23.9 [18.0, 29.9] | 75.6 | 6.5 / 11.1 | +4.2 / −10.3 [−16.4, −4.9] | vs without cmp +0.0 / +0.3 |
| dyn timeout Silero | cmp | 74.8 | 31.1 [25.0, 37.5] | 87.6 | 4.9 / 9.2 | +15.3 / −3.2 | +2.1 [+0.9, +3.4] / +4.4 [+0.3, +8.5] (fold 1 fits a = 0) |
| dyn timeout Silero | head + cmp | 70.9 | 22.5 [17.4, 28.1] | 85.8 | 3.4 / 6.8 | +11.3 / −11.8 | −1.8 [−3.1, −0.5] / −4.2 [−9.0, −0.2] (a_cmp = 0 / 10) |
| smart-turn (Silero trigger) OR Silero timeout, BASELINES.md | 72.6 | 26.7 | 86.7 | 5.2 / 5.8 | | |

2 s horizon (block A): cmp alone 96.8 / 94.5 open; cmp OR Silero 84.3 all / 43.7 open (Silero timeout 83.5 / 39.6);
fusion GBM head + Silero + cmp 91.3 / 76.5 (worse than head-OR-Silero's 78.3 / 58.1: the fusions read the head under
the Sortformer emission rule); dyn Silero | head + cmp 82.7 / 36.9 open with a_cmp = 0 in both folds.

Reading.
- **On AMI the completeness head behaves exactly like smart-turn**: alone it misses 92–94 % at <= 5 % FC (smart-turn
  99 %), OR-ed with the Silero timeout it equals the Silero timeout (72.3 vs 72.7 all, 24.9 vs 26.7 open, −1.8
  [−4.8, +0.9]), and the same diagnostic that gave smart-turn AUC 0.58 (BASELINES.md) gives, on this run's pause
  maxima vs the first 1.04 s after the end, **0.47 for our head and 0.39 for smart-turn's held Silero-trigger track**:
  neither ranks a turn end above a mid-turn hesitation on meeting audio. The logistic coefficients agree (cmp logit
  +0.07 / +0.05 next to the head's +0.49 and the Silero run's +1.4).
- The one row where cmp moves a number is GBM head + Silero run + cmp: −3.8 [−5.4, −2.2] all / −3.6 [−6.6, −0.8]
  open over the same GBM without it, giving the best open-end miss of any fused row (21.1 %) at FC 5.2 — but with
  12.6 % per-pause FC (it fires inside hesitations, which the per-turn budget does not see) and it is still worse
  than head-OR-Silero on all ends (+1.4 [−1.6, +4.4]) and than the dynamic timeout of §2.4 on open ends at lower FC.
- So a completeness signal, whether smart-turn's or ours, does not address the open-end problem on AMI: the problem
  is which speaker the silence belongs to and how fast the track says so (EOT_BENCH_V2 §7), and "was that a complete
  sentence" is orthogonal to it on multi-party audio where 47 % of open ends already look complete to smart-turn
  before they end (§2.3).

### 3.4 TurnBench dev with the official scorer (`scripts/bench_turnbench_cmp.py`)

38 two-channel conversations, per-speaker events on channel k, their rule (highest EOT recall at fp_rate <= 0.10,
swept over the system's own score quantiles; 2 s refractory; commit at the 160 ms chunk end). Silero rows are
recomputed in this work dir with the same `bench_turnbench.py` code so all rows share one VAD stage.

| system | EOT recall | FP rate | latency p10 / p50 / p90 (ms) | operating point |
|---|---|---|---|---|
| Silero VAD timeout (Pipecat state machine, channel k) | 0.815 | 0.075 | 844 / 1399 / 1525 | k = 1.5 s |
| **completeness head, channel k** | 0.280 | 0.089 | 113 / 597 / 1631 | θ = 0.943 |
| completeness head OR Silero timeout | 0.810 | 0.078 | 745 / 1390 / 1523 | θ = 0.984, k = 1.5 s |
| published smart-turn v3 (dev file re-scored here) | 0.754 | 0.100 | 684 / 1010 / 1204 | theirs |
| published VAP | 0.841 | 0.045 | −34 / 463 / 1657 | theirs |
| published Kyutai semantic VAD | 0.803 | 0.100 | 224 / 1024 / 1752 | theirs |

Reading: on clean two-channel dyadic audio our 124 K head alone reaches 28 % recall within the FP budget (the
published smart-turn v3 row, an 8 M model on 271 k clips, reaches 75 % on the same split with its own VAD trigger
and buffering), and OR-ed with the Silero timeout it adds nothing to the timeout (0.810 vs 0.815). Its p10 latency
(113 ms) shows it does fire at the end of complete sentences; its FP rate at that recall shows it also fires inside
turns, as on AMI. Compute for the stream: 59 350 s of audio in 172 s on MPS (345x real time) for encoder + head.

## 4. Verdict

1. **Does a completeness head on our shared encoder match smart-turn?** On smart-turn's own task and a held-out slice
   of its own test data (399 human_5 clips, English), yes at the ranking level: frame-level AUC 0.995 vs smart-turn
   v3.2's 0.994 given the same input (clip + its trailing silence), 0.994 within 0.5 s of the last speech frame,
   utterance-level 0.988–0.990, for 124 K parameters and 0.10 ms per 160 ms chunk on top of an encoder that already
   runs, against 8 M parameters and 28–36 ms per call. Accuracy at a fixed 0.5 is lower (0.87–0.94 vs 0.97:
   calibration) and the claim is English-only, 3.5 k training clips, one dataset. As a standalone end-of-turn
   detector on TurnBench it does not match smart-turn (recall 0.28 vs 0.75 at FP <= 0.10).
2. **Does it help our open-end problem?** No, and neither does smart-turn, in any of the three ways tried:
   - learned fusion (logistic / GBM, cross-fitted): every row that reaches Silero-timeout-level open-end misses
     (24–29 %) does so by becoming a timeout (taken-end misses 73–88 %), and the smart-turn / completeness features
     carry coefficients of 0.05–0.2 against 1.2–2.5 for the Silero silence run; fusing Silero + smart-turn (or + cmp)
     without the head equals the Silero timeout to within 0.1–2 points;
   - error complementarity: smart-turn says "complete" within 1 s on 55 % of the open ends our hybrid misses, which is
     its base rate (47 %) — the errors are independent, not complementary — and at that threshold it also cuts off
     49 % of open turns before their end;
   - the completeness stream on AMI: pause-vs-end AUC 0.47 (smart-turn 0.39); cmp OR Silero = Silero.
   The kill line ("beat head-OR-Silero on open ends, CI excluding 0, at equal or lower FC") is passed literally by
   two fusion rows (GBM all OR Silero: −7.6 [−13.7, −2.2]; GBM head + Silero + cmp: −13.2 [−19.4, −7.4]), but both are
   worse or equal on all ends and equal the plain Silero timeout on open ends, so nothing here should ship as a fusion.
3. **What did move.** The coordinator's dynamic-timeout family, with the *trail6 head posterior* shortening a Silero
   any-speaker timeout (required silence = T0 − a·p): open-end misses 23.0 vs 26.7 % for the fixed Silero timeout at
   lower FC (4.2 vs 4.8 %; −3.7 [−7.2, −0.3] at 6 s, −8.8 [−13.5, −4.4] at 2 s), and head OR that dynamic timeout =
   55.9 % all ends / 23.6 % open (vs the hybrid's 61.9 / 45.9 and head-OR-Silero's 59.6 / 34.3; −3.6 [−5.0, −2.3] /
   −10.7 [−15.4, −6.5] vs head-OR-Silero). smart-turn's contribution to that row is one fold's a_st = 10 frames; the
   head-only version is within 2 points on all ends. The continuous whole-window run shows the dynamic timeouts fire
   as rarely as the fixed one (0.37 events per window, 1.4 % of windows re-fire) where the OR rules fire 0.6 times
   per window and re-fire in 8–9 %. Like head-OR-Silero this was chosen on dev and needs the ICSI held-out check
   before it replaces the serve.py hybrid.
4. Data note for the record: the smart-turn v3.x datasets carry no licence; `human_5_all` (BSD-2-Clause) is the only
   licensed piece, so it is the only smart-turn audio on this machine; the v3.2 audio was never downloaded.

Files: `runs/completeness.json` (fusion, dyntimeout, human5 with per-clip scores), `runs/turnbench_dev_completeness.json`,
`runs/stage1_completeness.afm` (+ step checkpoints), `data/smartturn/` (parquet, 1 GB decode cache, v3.2-test id list).
