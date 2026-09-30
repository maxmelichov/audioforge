# State contamination after a transient wrong speaker binding (2026-09-26)

**Question.** A speaker-conditioned streaming model receives a WRONG speaker assignment for a short window, then the
assignment is corrected. Does the damage stop at the window (an instantaneous error), or does it persist in the
model's outputs after the correction (state contamination)? If it persists: how long, in which state (encoder caches
vs recurrent head / decoder state), which reset removes it, what does a bounded replay cost, and how much history a
replay needs to match the correct-history outputs.

Code: `scripts/contamination_probe.py` (`turn`, `turn-extra`, `saasr`, `report`); numbers: `runs/contamination.json`;
test: `tests/test_contamination.py`. CPU only, 2 threads. No library code was changed: the segmented runs use the
existing `TurnHead.v3_features / _fuse / rnn` pieces and `greedy_decode_frames`' loop with a state hook, and both
encoders are causal, so an offline forward over a corrupted-then-restored track *is* the stream (offline == `stream_step`,
tests/test_streaming.py). Every state-substitution protocol below is therefore an exact stream continuation.

## 1. Subjects and setup

**A. Turn head** `runs/stage1_turn_v3_trail6.afm`: speaker kernels at encoder layers 0 and 2 (17 layers, d 512, causal
chunked attention [70, 1] = 5.6 s left context, 160 ms chunks), TurnHead v3 (GRU, hidden 96; inputs: the conditioned
encoding, the diarizer's 4 columns + primary one-hot, duration counters, the RNNT text state; the RNNT is *not*
speaker-conditioned, so the text state never changes). Data: the eot-bench v2 extended AMI dev turn windows (6 s trail,
research/archive/EOT_BENCH_V2.md) with their cached streaming Sortformer tracks (trail6 work dir) under the **causal_dominant**
binding (bound column first, prim 0). Frozen hybrid point θ = 0.998164, k = 53 frames.

Turn selection: t0 = turn end − 2.0 s (25 frames); the largest window (2 s) must start ≥ 0.48 s after the onset and
the 4 s persistence window must lie inside the track → **321 of 974 turns** (turn length ≥ 4.5 s), 4 meetings. "Several
seconds before the end" was not possible on AMI (median turn 3.3 s); 2 s leaves 2 s of turn plus the whole 6 s trail
after the correction, so the decision-relevant region is covered.

Corruptions of the bound track on [t0 − W, t0), W ∈ {0.48, 0.96, 2.0 s} (6 / 12 / 25 frames), audio identical:
- `zero`: the bound column reads silent (act = 0, column 0 = 0);
- `swap`: the bound column shows the most active *other* column of the window (columns 0 and j swapped, act = j's);
- `rand`: swap with a random other column (control);
- `noop`: identical re-run (determinism check); `eps`: ±1e-4 on the window (numerical-noise bound).
Because t0 is mid-turn (the primary talks, the others mostly do not), `swap` and `rand` coincide with `zero` on most
turns; the turns where the partner column actually spoke on the window are reported separately.

Measurement: |Δp| = |p_corrupted − p_baseline| of the per-frame end-of-turn probability, from t0 in 160 ms bins to 4 s,
mean over turns with 95 % percentile-bootstrap CIs (1000 resamples over turns); also |Δ logit|, frame-level decision
flips at θ, the fraction of turns with |Δp| > 0.01 per bin, and the turn-level outcome (fire / miss / false cutoff,
latency) of the head at θ and of the hybrid (head OR deployable timeout at k, stream emission rule, 6 s horizon).

Localization protocols (applied identically to the baseline; the difference is measured against the baseline under the
same protocol, and the protocol's own cost = baseline-under-protocol vs plain baseline is reported):
- `enc_sub`: encoder frames ≥ t0 from the baseline run (= a stream whose KV / conv caches were swapped for correctly
  conditioned ones at t0), corrupted head state (GRU + counters) kept → what the head state alone carries;
- `state_sub`: corrupted encoder frames, baseline head state substituted at t0 → what the encoder caches alone carry;
  `gru_sub` / `dur_sub` split the head state into the GRU hidden and the duration counters;
- `head_reset` (GRU + counters to zero at t0; `gru_reset`, `dur_reset` separately), `both` (reset + baseline frames);
- `reprime`: the encoder re-run from t0 − 70 frames (its attention left context) with the corrected track, empty
  caches, head state kept; `reprime2x`: from t0 − 140 frames;
- `head_replay` (deployable, no encoder recompute): restore the head state snapshot from the window start, re-run the
  head over the window with the corrected columns / counters on the encoder frames as they are;
- `reprime2x_replay` (deployable with recompute): encoder re-run from window start − 140 frames with the corrected
  track, head replayed from the window start on those frames.

**B. Speaker-attributed ASR** (`research/recipes/speaker_attributed_asr.yaml`: 4-layer d 144 FastConformer with speaker kernels
at layers 0 / 2, Sortformer head, RNNT head conditioned on the speaker's activity, char tokens). Two checkpoints:
- `runs/speaker_attributed_asr.afm` as shipped: **full-context (non-causal) encoder**. Not a streaming model; kept as a
  contrast (the corruption reaches frames before the window through bidirectional attention).
- a **causal** variant of the same recipe (`causal: true`, `att_context_size: [70, 1]`) trained here on CPU in two
  chained 500-step processes (< 10 min each, init.from the first; held-out WER with oracle activity 2.3 %, the shipped
  model's 3.0 %). Scratch: `<scratchpad>/contam/saasr_causal_s2.afm` (not committed to runs/).
Data: 256 held-out synthetic mixtures (`synthetic_dataset('speaker_attributed', seed=1)`; the first 64 are the recipe's
val set), both speakers as the target → 194 cases (target span ≥ window + margin). The utterances are short (median
target speech 0.7 s, mixture 2 s), so the windows are W ∈ {160, 320, 480 ms}, t0 = 60 % through the target's active
span, and the bins are 160 ms. Corruption: the target's oracle activity zeroed / swapped for the other speaker's
activity (mostly silent at t0: swap ≈ zero again). Per-bin metric: normalized edit distance between the tokens emitted
in the bin by the corrupted and the baseline decode (plus the fraction of bins that differ at all); per utterance:
WER against the reference, attribution flips (words of the other speaker's text that are not in the target's).
Protocols: `dec_reset` (prediction net restarted at t0), `enc_sub` (baseline encoder frames ≥ t0, corrupted decoder
prefix), `both`, and bounded replay with buffers {0.5, 1, 2 s}.

## 2. Checks

- Frames before the chunk-aligned window start: max |Δp| = 0 over all turns and conditions (causality holds; the
  chunk partner frame lo − 1 legitimately sees frame lo through the 1-frame right context).
- `noop`: |Δp| = 0 exactly (deterministic); `eps` (±1e-4 on the track): |Δp| ≤ 2e-6. Numerical noise is at the 1e-6
  level; everything below is 3–5 orders of magnitude above it.
- `both` (baseline frames + fresh state) gives Δ = 0 in every subject, as it must.
- Early check after 60 turns (the stop-early rule): noise 2e-6 vs effect CI lower bound ≥ 0.2 in bins ≥ 1: the effect
  is not noise, the run continued.

## 3. Subject A: persistence curve of the turn head (n = 321 turns)

Mean |Δp| after the correction [95 % CI], `full` protocol (nothing reset; the stream simply continues). The window
ends at t0; the true turn end is at t0 + 2 s; the 6 s trail follows. "aff" = fraction of turns with |Δp| > 0.01 in
the bin. The corruption changed the input on 80–87 % of the windows (`n_eff`, mean |Δinput| 0.62–0.66; the bound
column is not always active mid-turn on these tracks).

| window | 0–160 ms | 160–320 | 320–480 | 0.96 s | 1.92 s | 2.88 s | 3.84 s | aff at 0 / 1 / 2 / 3 / 4 s |
|---|---|---|---|---|---|---|---|---|
| zero 0.5 s | 0.071 [0.058, 0.084] | 0.058 [0.048, 0.070] | 0.049 [0.041, 0.059] | 0.030 [0.024, 0.037] | 0.019 [0.014, 0.024] | 0.004 [0.003, 0.005] | 0.001 | 0.56 / 0.45 / 0.33 / 0.09 / 0.02 |
| zero 1 s | 0.122 [0.103, 0.142] | 0.104 [0.088, 0.121] | 0.091 [0.077, 0.106] | 0.058 [0.048, 0.069] | 0.039 [0.030, 0.048] | 0.007 [0.005, 0.010] | 0.003 | 0.66 / 0.55 / 0.44 / 0.15 / 0.05 |
| zero 2 s | 0.189 [0.160, 0.217] | 0.170 [0.145, 0.196] | 0.155 [0.132, 0.178] | 0.112 [0.093, 0.132] | 0.074 [0.058, 0.089] | 0.013 [0.010, 0.017] | 0.005 | 0.76 / 0.68 / 0.56 / 0.23 / 0.11 |
| swap 1 s | 0.162 [0.135, 0.188] | 0.147 [0.122, 0.171] | 0.134 [0.112, 0.156] | 0.103 [0.084, 0.123] | 0.069 [0.054, 0.085] | 0.013 [0.009, 0.017] | 0.006 | 0.69 / 0.57 / 0.48 / 0.22 / 0.08 |
| rand 1 s (control) | 0.171 [0.144, 0.198] | 0.156 [0.131, 0.182] | 0.142 [0.119, 0.166] | 0.109 [0.090, 0.129] | 0.074 [0.058, 0.090] | 0.014 [0.009, 0.018] | 0.006 | 0.70 / 0.59 / 0.50 / 0.22 / 0.09 |
| noop / eps | 0 / ≤ 2e-6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- In logit units the 1 s zero corruption is 1.1 at 0 ms, 0.6 at 1 s, 0.3 at 3 s, 0.2 at 4 s. The 90th percentile of
  the per-turn max |Δp| over the 4 s is 0.24 / 0.41 / 0.68 (zero 0.5 / 1 / 2 s); the largest deviation beyond 4 s is
  0.10 / 0.29 / 0.36.
- The decay is slow: from the first bin to the 1 s bin |Δp| falls by about half; it is still a third at 2 s (the true
  turn end) and collapses only in the trail (3–4 s), where both runs saturate near p = 1. Half of the affected turns
  remain affected at 2 s.
- `swap` / `rand` are ≥ `zero` although on 40–50 % of the windows the partner column is silent and the swap *is* the
  zero corruption (`n_swap_differs_from_zero` 155–188 of 321). On the turns where the partner spoke, swap 1 s gives
  0.078 at 0 ms → 0.052 at 1 s → 0.028 at 2 s (the input changed less on those windows), so the curve shape is the same.
- Frame-level decision flips at θ are rare before the end (0.3–6 %) but reach 3–13 % of frames in the 3–4 s bins
  (the trail, where θ = 0.998 is decided).

**Turn-level outcomes at the frozen point** (head at θ; hybrid = head OR timeout at k; class = fire / miss / FC over
the 6 s horizon). Baseline: head 134 fire / 150 miss / 37 FC; hybrid 141 / 141 / 39.

| corruption | head: class changed | head corrupted (fire / miss / FC) | hybrid: class changed | hybrid corrupted | main transitions (hybrid) |
|---|---|---|---|---|---|
| zero 0.5 s | 4.4 % | 134 / 142 / 45 | 4.7 % | 141 / 132 / 48 | fire→FC 6, miss→fire 6 |
| zero 1 s | 11.8 % | 140 / 124 / 57 | 11.8 % | 146 / 115 / 60 | miss→fire 17, fire→FC 12, miss→FC 9 |
| zero 2 s | 33.3 % | 128 / 88 / 105 | 34.9 % | 133 / 74 / 114 | fire→FC 45, miss→fire 37, miss→FC 30 |
| swap 2 s | 35.5 % | 139 / 95 / 87 | 35.5 % | 142 / 88 / 91 | fire→FC 37, miss→fire 35, miss→FC 26 |
| rand 2 s | 41.4 % | 132 / 74 / 115 | 42.1 % | 135 / 65 / 121 | fire→FC 49, miss→fire 41, miss→FC 38 |
| noop / eps | 0 | = baseline | 0 | = baseline | — |

A silent bound track for 1–2 s mid-turn makes the head fire during the turn: false cutoffs triple at 2 s. How much of
that is the window itself and how much the aftermath is answered by the `both` protocol (everything reset at t0, so
the composite differs from its baseline only on the window): window-only class changes are 17.4 % (head) / 20.6 %
(hybrid) for zero 2 s, against 33.3 % / 34.9 % for the free-running stream. **About 40 % of the decision changes come
from the frames after the correction.** Including latency changes among fired turns, 25 / 41 / 66 % of the turns end
differently for the 0.5 / 1 / 2 s windows.

## 4. Where the persistence lives (localization)

|Δp| vs the baseline under the same protocol, zero 1 s (2 s in brackets), bins 0 ms / 0.96 s / 1.92 s / 2.88 s, and
the protocol's own cost on the baseline (bins 0 / 0.96 / 1.92 s).

| protocol | what stays corrupted after t0 | 0 ms | 0.96 s | 1.92 s | 2.88 s | protocol cost |
|---|---|---|---|---|---|---|
| `full` | encoder caches + GRU + counters | 0.122 (0.189) | 0.058 (0.112) | 0.039 (0.074) | 0.007 (0.013) | 0 |
| `enc_sub` | GRU + counters (baseline encoder frames) | 0.123 (0.189) | 0.058 (0.110) | 0.037 (0.071) | 0.006 (0.011) | 0 |
| `state_sub` | encoder caches (baseline GRU + counters) | 0.016 (0.021) | 0.013 (0.023) | 0.013 (0.019) | 0.003 (0.005) | 0 |
| `gru_sub` | encoder caches + counters (baseline GRU) | 0.017 (0.022) | 0.013 (0.023) | 0.013 (0.019) | 0.003 (0.005) | 0 |
| `dur_sub` | encoder caches + GRU (baseline counters) | 0.123 (0.190) | 0.058 (0.113) | 0.039 (0.074) | 0.007 (0.013) | 0 |
| `head_reset` (GRU + counters → 0 at t0) | encoder caches | 0.060 (0.073) | 0.041 (0.061) | 0.034 (0.046) | 0.006 (0.010) | **0.24 / 0.21 / 0.15** |
| `gru_reset` | encoder caches + counters | 0.060 (0.074) | 0.040 (0.061) | 0.033 (0.045) | 0.006 (0.010) | 0.24 / 0.20 / 0.14 |
| `dur_reset` | encoder caches + GRU | 0.122 (0.188) | 0.058 (0.112) | 0.039 (0.074) | 0.007 (0.013) | 0.002 / 0.004 / 0.003 |
| `reprime` (encoder from t0 − 5.6 s, state kept) | GRU (+ re-prime error) | 0.131 (0.211) | 0.057 (0.116) | 0.040 (0.085) | 0.010 (0.019) | 0.051 / 0.047 / 0.052 |
| `reprime2x` (from t0 − 11.2 s, state kept) | GRU | 0.123 (0.190) | 0.058 (0.113) | 0.038 (0.075) | 0.006 (0.012) | 0.002 / 0.005 / 0.006 |
| `head_replay` (state snapshot + corrected head inputs, stale frames) | GRU (rebuilt from stale frames) | 0.124 (0.190) | 0.059 (0.114) | 0.040 (0.075) | 0.007 (0.013) | 0 |
| **`reprime2x_replay`** (encoder from window − 11.2 s + head replay) | nothing | **0.005 (0.005)** | 0.003 (0.003) | 0.002 (0.002) | 0.001 (0.001) | 0.002 / 0.005 / 0.005 |
| `both` | nothing (check) | 0 | 0 | 0 | 0 | 0.24 / 0.21 / 0.15 |

Reading:
1. **The persistence is in the GRU hidden state of the head, not in the encoder caches.** Substituting the baseline
   encoder frames after t0 while keeping the corrupted GRU (`enc_sub`) reproduces the full curve (0.123 vs 0.122);
   substituting the baseline GRU while keeping the corrupted encoder caches (`state_sub`) leaves 0.016 → 0.013,
   about one eighth, flat rather than decaying (the encoder's deep receptive field: 17 layers × 70 frames). The
   duration counters carry nothing (`dur_sub` = `full`, `dur_reset` costs 0.002).
2. **The GRU state is contaminated by the encoder frames of the window, not by the head's own inputs.** `head_replay`
   (rebuild the GRU state over the window with the corrected columns / counters but the stale, wrongly conditioned
   encoder frames) does not help at all for `zero` (0.124) and only partly for `swap` (0.105 vs 0.162, where the
   corrected columns matter). Speaker kernels at layers 0 / 2 change the encoding a lot (mean |Δenc| on the window
   0.10–0.14 on an activation scale of 0.13), and the GRU integrates those frames.
3. **Resets are not a repair.** Zeroing the GRU at t0 halves the difference but costs 0.24 on the baseline for 2 s
   (larger than the contamination) and moves the baseline's own outcomes (fire 141 → 130, FC 39 → 34). Re-priming the
   encoder from its 5.6 s attention context alone costs 0.05, persistently, because a frame at layer 17 depends on
   17 × 70 frames of history; from 11.2 s the cost is 0.002–0.006 (see §6 for the buffer sweep).
4. **The repair that works is deterministic replay with retained evidence:** re-run the encoder from ≈ 11 s before the
   window with the corrected track and replay the head from its state snapshot at the window start
   (`reprime2x_replay`): |Δp| 0.005 at 0 ms, 0.002 at 2 s, 10 % of turns above 0.01 in any bin, outcome class changes
   16.5 % (head) = the window-only 17.4 % within noise: all after-correction damage removed, at the cost of one encoder
   pass over ≈ 13 s of audio (§6).
5. Outcome view (hybrid, zero 2 s, class changed): full 34.9 %, `enc_sub` 30.5 %, `state_sub` 22.4 %, `gru_sub`
   22.4 %, `reprime2x_replay` 19.3 %, window-only (`both`) 20.6 %.

## 5. Subject B: speaker-attributed ASR

Per-bin token change vs the baseline decode after t0 (mean normalized edit distance [95 % CI]; the fraction of cases
whose tokens differ in the bin in brackets), n = 194 cases. Bins beyond 640 ms are exactly 0 wherever audio remains
(184 / 107 cases in bins 4 / 5). The noop control is 0 everywhere.

**Causal model** (the streaming answer):

| condition / protocol | 0–160 ms | 160–320 | 320–480 | 480–640 | 640–800 | utterance WER (baseline 4.0 %) | flips / utt (baseline 0.021) |
|---|---|---|---|---|---|---|---|
| zero 160 ms | 0.68 [0.62, 0.74] (72 %) | 0.40 [0.33, 0.47] (43 %) | 0.20 [0.16, 0.25] (28 %) | 0.11 [0.07, 0.15] (12 %) | 0 | 30.6 % | 0.067 |
| zero 320 ms | 0.67 [0.61, 0.73] (72 %) | 0.43 [0.37, 0.50] (47 %) | 0.32 [0.25, 0.38] (40 %) | 0.07 [0.04, 0.11] (9 %) | 0 | 39.7 % | 0.052 |
| zero 480 ms | 0.78 [0.73, 0.84] (79 %) | 0.54 [0.47, 0.61] (56 %) | 0.36 [0.30, 0.43] (42 %) | 0.09 [0.06, 0.13] (11 %) | 0 | 60.0 % | 0.103 |
| swap / rand 320 ms | 0.67 | 0.43 | 0.32 | 0.07 | 0 | 39.4 % | 0.052 |
| `enc_sub`, zero 320 ms | 0.62 [0.56, 0.69] | **0.05 [0.02, 0.08]** | 0.02 | 0.02 | 0 | 34.0 % | 0.036 |
| `dec_reset`, zero 320 ms | 0.72 | 0.63 | 0.19 | 0.08 | 0 | 88.7 % (baseline under reset 27.1 %) | 0.015 |

Reading: the decode differs from the baseline for about 500–640 ms after the correction (bins 0–3), then is identical.
Bin 0 is largely catch-up (the window's characters are emitted late once the conditioning returns: `enc_sub`, which
keeps only the decoder prefix, still shows 0.62 in bin 0 but 0.05 afterwards). Bins 1–3 are the **encoder cache**:
swapping in correctly conditioned frames at t0 removes them; restarting the prediction net does not (and costs 27
points of WER on the baseline). The utterance WER is dominated by the window itself (a 320 ms window is half of a 0.7 s
utterance); attribution flips rise from 0.02 to 0.05–0.10 words per utterance. The 'swap' condition equals 'zero'
because the other speaker is silent at t0 in these mixtures (190 / 194 cases).

**Non-causal shipped model** (contrast): changes are confined to bins 0–1 (0.37 / 0.13, then exactly 0), but the
tokens emitted *before* the window change in 73–97 % of cases (bidirectional attention); there is no "persistence" to
speak of because there is no state: the whole utterance is re-read. Utterance WER 39–62 %.

**Bounded replay, causal model** (encoder re-run from t0 − L with empty caches and the corrected activity, decoder
prefix kept; identical for every corruption because the buffer covers the window, so the number is the replay's own
deviation from the full correct history): L = 0.5 s: bin 0–3 change 0.08 / 0.04 / 0.04 / 0.04, WER 23 % (vs 4 %);
L = 1 s: 0.008 / 0.005 / 0.008 / 0.005, WER 3.6 %; L = 2 s (the whole utterance): 0. With a 5.6 s attention context a
1 s cold start is already within noise of the full history for this 4-layer model; 0.5 s is not.

Replay wall-clock (2 threads, idle machine, causal model, encoder + greedy decode over the buffer): 1 s 9 ms, 2 s
10 ms, 4 s 12 ms, 6 s 16 ms, 10 s 21 ms, 15 s 27 ms; one normal 160 ms step (stream_step + greedy) 7.1 ms. A 6 s
replay costs 2.2 chunk steps = 0.10 real-time chunks; 15 s costs 3.7 chunk steps = 0.17 real-time chunks.

## 6. Replay cost baseline (turn head; n = 150 turns, zero / swap 1 s window ending at t0)

Bounded replay at t0 with buffer L: the encoder is re-run from t0 − L with empty caches and the corrected track, the
head is replayed from its state snapshot at the window start (clean for every L ≥ W) on those frames, and the
deviation from the full correct history after t0 is measured. Because the buffer covers the window the result is the
same for every corruption, so the numbers are the replay's own error. "enc err" = mean |Δ encoder output| relative to
the activation scale, on the window and on the 2 s after t0. "clipped" = the buffer reaches the window start (then the
replay is the full history). Outcome change = hybrid outcome (class or latency) differs from the baseline's.

| buffer L | 0 ms | 0.48 s | 0.96 s | 1.92 s | 2.88 s | aff > 0.01 at 0 / 1 / 2 s | turns with max |Δp| < 0.01 | enc err window / post | outcome changed | clipped |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 s | 0.70 [0.65, 0.76] | 0.71 | 0.72 | 0.68 | 0.49 | 1.00 / 0.99 / 1.00 | 0 % | 1.16 / 0.95 | 49 % | 0 % |
| 2 s | 0.50 [0.45, 0.54] | 0.53 | 0.53 | 0.49 | 0.24 | 0.99 / 0.99 / 0.99 | 0 % | 0.94 / 0.89 | 49 % | 0 % |
| 4 s | 0.30 [0.26, 0.33] | 0.33 | 0.35 | 0.26 | 0.10 | 0.91 / 0.91 / 0.90 | 1 % | 0.79 / 0.79 | 49 % | 1 % |
| 6 s | 0.12 [0.10, 0.14] | 0.11 | 0.11 | 0.10 | 0.04 | 0.77 / 0.68 / 0.66 | 14 % | 0.68 / 0.52 | 35 % | 7 % |
| 10 s | 0.018 [0.012, 0.024] | 0.018 | 0.020 | 0.024 | 0.006 | 0.28 / 0.29 / 0.29 | 53 % | 0.23 / 0.20 | 23 % | 45 % |
| 15 s | 0.002 [0.001, 0.003] | 0.002 | 0.003 | 0.002 | 0.001 | 0.08 / 0.09 / 0.09 | 85 % | 0.034 / 0.024 | 4 % | 61 % |
| (no repair, zero 1 s, §3) | 0.122 | 0.091 | 0.058 | 0.039 | 0.007 | 0.66 / 0.55 / 0.44 | | | 41 % | |

- A cold-started FastConformer with 17 layers of 70-frame attention needs far more than its 5.6 s attention span to
  reproduce its warm outputs: the encoder error on the window is 68 % of the activation scale with a 6 s buffer and
  still 23 % with 10 s; only ≈ 15 s (which for 61 % of these turns is the whole window, i.e. the full history) is
  within noise. Replaying with 1–4 s of buffer is **worse than not repairing** (0.30–0.70 vs 0.12) and changes half
  of the outcomes; 6 s is a wash (0.12 vs 0.12); 10 s recovers most turns (53 % within 0.01; 23 % of outcomes still
  moved, most of them latency changes); 15 s recovers 85 % of turns within 0.01 and 96 % of outcomes.
- The 11.2 s re-prime of §4 (`reprime2x`, cost 0.002–0.006) is not in contradiction: there the head state at t0 came
  from the warm run and only the frames after t0 were cold-started; here the head must be replayed over the window,
  which sits in the coldest part of the buffer. Any repair that rebuilds the head state needs warm encoder frames on
  the window, so the buffer must extend about 10 s *before the window*.

**Wall-clock** (2 threads, idle machine, median of 7; `turn_timing`): encoder + head over the buffer, vs one normal
160 ms step (`encoder.stream_step` on a 16-mel chunk with primed caches + `TurnHead.step` on its 2 frames).

| | 1 s | 2 s | 4 s | 6 s | 10 s | 15 s | one 160 ms step |
|---|---|---|---|---|---|---|---|
| turn head (17 × 512, 120 M params) | 152 ms | 165 | 121 | 125 | 189 | 208 | 97 ms |
| = chunk steps | 1.6 | 1.7 | 1.2 | 1.3 | 2.0 | 2.1 | 1 |
| = real-time chunks (160 ms) | 0.95 | 1.0 | 0.75 | 0.78 | 1.2 | 1.3 | 0.6 |
| SA-ASR causal (4 × 144, 2.7 M) | 9 ms | 10 | 12 | 16 | 21 | 27 | 7.1 ms |

The batched offline pass is dominated by fixed overhead (the 4 s replay is not slower than the 1 s one), so a full
15 s replay of the large model costs about two streaming steps, 1.3 real-time chunks; for the tiny model 3.7 steps but
27 ms. (Under the machine's peak load during the main run the same numbers were 217–434 ms vs a 316 ms step; the
ratios hold.)

## 7. Correction-delay sweep (turn head; n = 150, window 1 s ending 2 s before the turn end)

The diarizer error is transient (the track is correct again after the window) but the system learns about it only
d ∈ {0, 0.5, 1, 2, 4} s after the window ends, at t_c = t0 + d, and then repairs: (i) nothing; (ii) `head_replay`
(state snapshot + corrected head inputs on the stale encoder frames); (iii) `enc_head_replay` (encoder re-run from
11.2 s before the window with the corrected track, head replayed from the snapshot); (iv) the same repair but pointing
the window at a WRONG third column (`wrong_repair`). |Δp| vs the baseline after t_c; outcomes on the composite output
(uncorrected until t_c, repaired after). "unc" = mean |Δp| over the uncorrected stretch [t0, t_c). Baseline hybrid on
these 150 turns: 59 fire / 72 miss / 19 FC.

| d | unc | no repair: 0 ms / 0.48 / 0.96 / 1.92 s | `head_replay` | `enc_head_replay` | `wrong_repair` | hybrid class changed: none / head / enc+head / wrong |
|---|---|---|---|---|---|---|
| zero, 0 s | – | 0.125 / 0.083 / 0.063 / 0.040 | 0.126 / 0.084 / 0.063 / 0.041 | 0.010 / 0.011 / 0.011 / 0.011 | 0.196 / 0.151 / 0.121 / 0.074 | 10.7 / 10.7 / **1.3** / 14.0 % |
| zero, 0.5 s | 0.108 | 0.083 / 0.063 / 0.055 / 0.013 | 0.084 / 0.063 / 0.055 / 0.013 | 0.011 / 0.011 / 0.012 / 0.004 | 0.151 / 0.121 / 0.106 / 0.025 | 10.7 / 10.7 / 2.0 / 14.0 |
| zero, 1 s | 0.091 | 0.063 / 0.055 / 0.040 / 0.008 | 0.063 / 0.055 / 0.041 / 0.008 | 0.011 / 0.012 / 0.011 / 0.002 | 0.121 / 0.106 / 0.074 / 0.015 | 10.7 / 10.7 / 3.3 / 14.0 |
| zero, 2 s | 0.073 | 0.030 / 0.012 / 0.008 / 0.003 | 0.031 / 0.012 / 0.008 / 0.004 | 0.009 / 0.004 / 0.002 / 0.001 | 0.056 / 0.023 / 0.014 / 0.005 | 10.7 / 10.7 / 6.0 / 14.7 |
| zero, 4 s (n 142) | 0.039 | 0.004 / 0.007 / 0.008 / 0.005 | 0.004 / 0.007 / 0.008 / 0.006 | 0.001 / 0.002 / 0.002 / 0.002 | 0.006 / 0.010 / 0.010 / 0.008 | 10.6 / 10.6 / 9.2 / 13.4 |
| swap, 0 s | – | 0.205 / 0.164 / 0.133 / 0.085 | 0.125 / 0.084 / 0.063 / 0.043 | 0.010 / 0.011 / 0.011 / 0.011 | 0.196 / 0.151 / 0.121 / 0.074 | 15.3 / 9.3 / 2.0 / 14.7 |
| swap, 1 s | 0.170 | 0.133 / 0.118 / 0.085 / 0.016 | 0.063 / 0.056 / 0.043 / 0.008 | 0.011 / 0.012 / 0.011 / 0.002 | 0.121 / 0.106 / 0.074 / 0.015 | 15.3 / 10.0 / 3.3 / 14.7 |
| swap, 2 s | 0.144 | 0.065 / 0.025 / 0.015 / 0.005 | 0.032 / 0.012 / 0.008 / 0.003 | 0.009 / 0.004 / 0.002 / 0.001 | 0.056 / 0.023 / 0.014 / 0.005 | 15.3 / 12.0 / 6.7 / 15.3 |
| swap, 4 s (n 142) | 0.078 | 0.006 / 0.011 / 0.010 / 0.009 | 0.004 / 0.007 / 0.008 / 0.006 | 0.001 / 0.002 / 0.002 / 0.002 | 0.006 / 0.010 / 0.010 / 0.008 | 14.1 / 14.1 / 12.0 / 14.8 |

- The uncorrected stream keeps a mean |Δp| of 0.11 / 0.09 / 0.07 / 0.04 over the first 0.5 / 1 / 2 / 4 s after the
  window (the §3 curve seen from the other end); what a repair at t_c can still remove is what is left at t_c
  (0.125 at d = 0 → 0.004 at d = 4 s).
- A full repair (`enc_head_replay`) reduces the after-correction difference to 0.01 at any delay, but the **decision
  damage is committed while the correction is pending**: the hybrid's class changes go 1.3 % (d = 0) → 2.0 (0.5 s) →
  3.3 (1 s) → 6.0 (2 s) → 9.2 % (4 s), against 10.7 % without repair. Half of the repair's value is gone after 2 s
  (here: at the true turn end), essentially all of it after 4 s (in the trail, where the decision is taken).
- The head-only replay is worthless for `zero` and recovers a third for `swap` (the corrected columns matter there).
- A **wrong correction is worse than none**: pointing the window at a third column gives 0.196 vs 0.125 at d = 0 and
  14–15 % class changes vs 10.7 % (it replaces one wrong binding by another, and rebuilds the state from it).

## 8. Real delayed corrections: causal_dominant rebinds (n = 508 events on 974 windows)

The follower binding of eot-bench v2 (`enroll_causal_dominant`, k = s = 25) produces real (old column → new column,
time) events. Selected: the first re-bind that lands on the **oracle column from a wrong one** (a genuine delayed
correction), with ≥ 3 wrong frames before and ≥ 1 s of track after; 508 of the 974 windows have one. The wrong
stretch is long: median 55 frames = 4.4 s (p10 2.4 s, p90 7.1 s); the rebind comes a median 2.1 s after the primary's
onset (p10 0.08 s, p90 4.7 s), in 21 % of the events after the turn end and in 4 % before the onset. On the wrong
stretch the follower's column is active 43 % of the time vs 37 % for the oracle column (mean |Δ input| 0.58): the
follower is mostly on a column that *speaks*, so the real corruption is a `swap`, not a `zero` (the swap partner is
active on 97 % of these windows).

Baseline = the oracle column bound throughout (`bound_track` representation, prim 0). Real = the same track with the
follower's wrong column bound on [t_a, t_r). Synthetic controls on the same window: `zero`, `swap` (most active other
column). |Δp| after the rebind:

| condition | 0 ms | 0.48 s | 0.96 s | 1.92 s | 2.88 s | aff > 0.01 at 0 / 1 / 2 / 3 s | mean max |Δp| 4 s |
|---|---|---|---|---|---|---|---|
| **real rebind** | 0.488 [0.461, 0.515] | 0.394 [0.363, 0.423] | 0.352 [0.324, 0.383] | 0.284 [0.256, 0.313] | 0.236 [0.208, 0.266] | 0.95 / 0.89 / 0.73 / 0.60 | 0.60 |
| swap, same window | 0.485 | 0.392 | 0.350 | 0.282 | 0.233 | 0.95 / 0.89 / 0.73 / 0.60 | 0.59 |
| zero, same window | 0.191 | 0.176 | 0.155 | 0.104 | 0.081 | 0.67 / 0.66 / 0.55 / 0.46 | 0.32 |
| real, `state_sub` (encoder caches only) | 0.023 | 0.035 | 0.043 | 0.037 | 0.031 | 0.40 / 0.49 / 0.49 / 0.43 | 0.13 |
| real, `enc_sub` (GRU + counters only) | 0.493 | 0.378 | 0.332 | 0.264 | 0.213 | 0.95 / 0.88 / 0.72 / 0.60 | 0.59 |
| real, rebind ≥ 1 s after onset and before the end (n = 294) | 0.586 | 0.509 | 0.459 | 0.364 | 0.289 | | |

- After a real rebind the head runs for the rest of the turn on a state built from the wrong speaker's frames: |Δp|
  is still 0.24–0.29 three to four seconds later and 60 % of the events are above 0.01 at 3 s. This is 2–3 × the
  synthetic 2 s `zero` window of §3 (0.074 at 2 s), for two reasons the synthetic design under-represents: the wrong
  stretch is 2–3 × longer, and it is a swap (another speaker's speech read as the primary's), which the head finds
  far more confusing than silence (0.49 vs 0.19 at 0 ms on the same windows).
- The synthetic `swap` on the same windows reproduces the real curve exactly (0.485 vs 0.488): the synthetic
  corruption is a faithful model of the real event once the window length and the partner's activity match.
- The localization is the same as in §4: the GRU carries it (`enc_sub` 0.49), the encoder caches 0.02–0.04.

## 9. Verdict and implications

**Persistent.** Not negligible and not short-lived: after the binding is corrected, the turn head's end-of-turn
probability keeps differing from the correct-history run for seconds, with a half-life of about 1.5–2 s (zero 1 s
window: |Δp| 0.12 at 0 ms, 0.06 at 1 s, 0.04 at 2 s, still > 0.01 on 44 % of turns at 2 s), scaling with the length of
the wrong stretch (0.19 at 0 ms for 2 s) and with its kind (swap ≫ zero). After the real enrollment rebinds of the
causal follower (median 4.4 s wrong stretch, a swap) it is 0.49 at 0 ms and 0.24 at 3 s. It changes decisions: 12 %
(1 s) to 35 % (2 s) of the turns change class at the frozen hybrid point, and about 40 % of those changes are caused by
the frames after the correction, not by the window itself. For the small speaker-attributed ASR the persistence is
short: the decode differs for ≤ 640 ms after the correction (bins 0–3), then is identical.

(a) **Where it lives.** In the turn head it is the **GRU hidden state**, filled by the wrongly conditioned encoder
frames of the window: substituting the baseline encoder frames after t0 changes nothing (0.123 vs 0.122), substituting
the baseline GRU state removes 7/8 of it (0.016); the duration counters carry nothing; the encoder caches carry a small,
non-decaying remainder (0.013–0.02, the 17 × 70-frame receptive field). Replaying the head with corrected inputs on the
stale frames does not help (the state is made of the frames, not of the head's own inputs). In the causal SA-ASR it is
the **encoder cache** (bins 1–3 vanish when correctly conditioned frames are substituted; restarting the prediction
net does not remove them and costs 27 WER points); the RNNT prefix carries only the catch-up of the window's tokens.

(b) **Bounded replay is cheap on CPU.** Replaying encoder + head over 15 s of retained audio with the corrected track
costs 208 ms on 2 threads for the 120 M-parameter turn model = 2.1 normal 160 ms steps = 1.3 real-time chunks (the
offline pass is overhead-dominated: 1 s costs 152 ms); for the 2.7 M SA-ASR 27 ms. A learned repair module (a network
that maps the contaminated state to a corrected one) therefore cannot justify itself on compute: two chunk steps once
per rebind (1.6 rebinds per turn) is noise. **It would have to win on context preservation** (no need to retain 10–15 s
of audio and track history, or a recomputation-free path when the history is gone) or on latency jitter (a 200 ms
stall at the moment of the rebind, which a replay spread over the next chunks or run in a side thread also avoids).
The resets are not an alternative: zeroing the GRU costs 0.24 on the baseline for 2 s, more than the contamination.

(c) **How much buffer replay needs.** For the turn head, a lot: the cold-started encoder reproduces its warm outputs on
the window only after ≈ 10 s of warm-up (window encoder error 116 / 94 / 79 / 68 / 23 / 3 % of the activation scale for
1 / 2 / 4 / 6 / 10 / 15 s). Replay with 1–4 s is worse than no repair (|Δp| 0.30–0.70 vs 0.12, 49 % of outcomes moved),
6 s breaks even (0.12), **10 s** brings the median turn within 0.01 (mean 0.018, 23 % of outcomes still moved), and
only ≈ 15 s (the full history for most turns) is within noise (0.002, 4 % of outcomes moved). So the buffer must cover
the wrong stretch (median 4.4 s for real rebinds) plus ≈ 10 s of warm-up before it: **about 15 s of audio + track**,
or a periodically saved warm encoder state to replay from. The small 4-layer SA-ASR needs 1 s (0.5 s costs 19 WER
points, 1 s is within noise). The repair's value also decays with the correction delay: at the frozen operating point
a full repair removes 88 % of the decision damage when applied at the window end, 70 % after 1 s, 44 % after 2 s and
14 % after 4 s, and a wrong correction (third column) is worse than none.

**Implications.**
- *Enrollment rebinding* (research/archive/EOT_BENCH_V2.md §7–8): a follower rebind is not a clean switch. After it, the head
  runs on a state built from the wrong speaker for the rest of the turn; this is a plausible part of why the causal
  binding loses ≈ 35 points of miss to the oracle binding while its agreement at the turn end is 70 %. The deployable
  fix is deterministic: keep ≈ 15 s of audio and track (or a warm encoder state snapshot every few seconds plus the
  head state per chunk), and on a rebind re-run the encoder over the retained history with the new binding and replay
  the head from its snapshot (≈ 200 ms once per rebind). Rebinds after the turn end (21 %) do not need it.
- *Training with corrupted conditioning*: the head was trained with corrupted windows (`ext_noise` swap ≤ 12 frames,
  drop) but the labels never asked it to *recover* after a correction, and the evidence says it does not: the GRU
  keeps the wrong-speaker evidence for seconds. A training condition "wrong binding for W then corrected, loss on the
  frames after the correction against the clean-history targets" (and, for the encoder, the same for the speaker
  kernels) is the learning-based counterpart of the replay, and it is exactly what this probe measures; success = the
  §3 curve falling to the `state_sub` level (≈ 0.015) without a replay. It would also make the swap case (the real
  one) tractable, where head-input correction alone recovers only a third.
- *Turn-end decisions*: the damage that matters is decided in the trail; a correction that arrives more than ≈ 2 s
  after the wrong stretch cannot undo it. Rebind latency, not only rebind accuracy, is on the critical path.
