# VAD per encoder block, and what a gated encoder would save

2026-09-26. Three questions about the served VAD head (`stage1_heads_pretrained.afm`: a 64-hidden frame head on a
learned, near-uniform mix of all 17 FastConformer blocks; BASELINES.md VAD: F1 0.949 at 160 ms on the 64 × 20 s AMI dev
diar windows):

1. Is a probe on an early block (2–4) as good as the served head? If so, a "gated encoder" could run blocks 1..k
   always and blocks k+1..17 only while someone speaks.
2. What does the encoder truncated after block k cost per 160 ms chunk, and what does it cost to re-prime blocks
   k+1..17 when speech resumes?
3. How much of the audio is non-speech, and what would the gated design save?

Code: `scripts/vad_layers.py` (stages `features`, `probes`, `compute`, `silence`, `report`), helpers
`truncated_encoder`, `DeepRunner`, `GatedEncoder` (tests/test_vad_layers.py). Numbers: `runs/vad_layers.json`.
Serving code is untouched. CPU only, 2 threads, other agents on the machine (read wall times as ±20 %).

**Verdict in one paragraph.** A probe on block 4 is as good as the served head on the BASELINES protocol
(F1 0.9476 vs 0.9485, AUC 0.968 vs 0.970; blocks 2 and 3 are not: −0.011 / −0.0055 F1), and every block from 4 to 14
matches within 0.005; the best single block is 9 (0.9496). AUC peaks at blocks 6–7 and falls from block 12 on (top
block 0.936), so the mix's slight preference for early blocks is real. But the gated encoder does not pay: blocks
k+1..17 need **420 frames (34 s) of recomputed history to match the ungated outputs within 1e-3** (280 frames for
1e-2; 70 frames, the attention left context, leaves errors of order 1), because each of the 13–15 deep blocks adds
70 frames of receptive field. That re-prime is 210 chunk-equivalents of deep-block work per speech onset, against
about 2 chunk-equivalents saved per gated chunk, and AMI has 6–8 onsets per minute: in FLOPs the design loses by an
order of magnitude at any k, and it only looks like a 10–17 % saving in the Python streaming path because that path
is per-call overhead. Recommendation: do not gate the encoder. If a cheap early VAD is wanted, take it from block 4
(a 33 K head, available at 25 % of the encoder's per-chunk cost and 80 ms earlier in the pipeline) and use it to
gate stateless or cheap-to-restart consumers (the RNNT decoder loop, the turn head, the diarizer preset), not the
encoder itself.

## 1. Accuracy per block

**Protocol.** Features are the per-block outputs of the [70, 1] chunked-limited forward, which is what cache-aware
streaming computes (asserted on real windows in §2: max |Δ| 3.4e-6 / 1.8e-5). Probes: logistic regression (sklearn,
C = 1) and a 64-hidden MLP in the served head's shape (Dropout 0.2 – Linear(512, 64) – SiLU – Linear(64, 1), Adam
1e-3 cosine, 1500 steps × 2048 frames, wd 1e-3), per-block standardisation, fitted on **1000 AMI train diar windows**
(12 meetings, 20 s / hop 10 s, seeded; 251 000 frames, 85.0 % speech). Scored with exactly the BASELINES VAD code
(`sd.score_vad`: pooled acc / recall / precision / F1 at 0.5 on the any-speaker 80 ms label) on the 64 AMI dev diar
windows (16 064 frames, 76.9 % speech), plus AUC and the miss rate at FPR 0.075 (the point at which BASELINES compares
us with Silero: served head 12.1 %, Silero 13.8 %), and on 64 ICSI dev diar windows (16 064 frames, 79.0 % speech) as
the held-out corpus. The served head is scored from the same pass and reproduces BASELINES exactly (acc 0.9206,
F1 0.9485, FPR 0.179, miss@FPR 0.075 = 0.1207).

| input | logreg AMI dev F1 / AUC / miss@FPR .075 | **mlp64 AMI dev F1 / AUC / miss@FPR .075** | mlp64 acc / rec / prec | mlp64 ICSI dev F1 / AUC / miss@FPR .075 | Δ F1 vs served (mlp64) |
|---|---|---|---|---|---|
| block 1 | 0.9257 / 0.9506 / 0.189 | 0.9321 / 0.9580 / 0.155 | 0.898 / 0.911 / 0.954 | 0.9044 / 0.8933 / 0.312 | −0.0164 worse |
| block 2 | 0.9367 / 0.9582 / 0.160 | 0.9379 / 0.9620 / 0.144 | 0.906 / 0.925 / 0.951 | 0.9099 / 0.9095 / 0.260 | −0.0106 worse |
| block 3 | 0.9418 / 0.9639 / 0.141 | 0.9430 / 0.9672 / 0.128 | 0.913 / 0.934 / 0.952 | 0.9111 / 0.9214 / 0.241 | −0.0055 worse (just) |
| **block 4** | 0.9456 / 0.9651 / 0.142 | **0.9476 / 0.9680 / 0.133** | 0.920 / 0.946 / 0.949 | 0.9095 / 0.9150 / 0.278 | **−0.0009 match** |
| block 5 | 0.9433 / 0.9658 / 0.135 | 0.9464 / 0.9677 / 0.130 | 0.918 / 0.943 / 0.950 | 0.9087 / 0.9112 / 0.297 | −0.0021 match |
| block 6 | 0.9463 / 0.9678 / 0.132 | 0.9458 / 0.9687 / 0.125 | 0.917 / 0.941 / 0.951 | 0.9077 / 0.9231 / 0.232 | −0.0027 match |
| block 7 | 0.9467 / 0.9684 / 0.133 | 0.9481 / **0.9692** / **0.124** | 0.920 / 0.945 / 0.951 | 0.9043 / 0.9260 / 0.222 | −0.0004 match |
| block 8 | 0.9475 / 0.9672 / 0.133 | 0.9483 / 0.9684 / 0.127 | 0.920 / 0.949 / 0.948 | 0.9012 / 0.9203 / 0.255 | −0.0002 match |
| **block 9** | 0.9481 / 0.9672 / 0.134 | **0.9496** / 0.9685 / 0.128 | 0.922 / 0.952 / 0.947 | 0.9032 / 0.9108 / 0.294 | +0.0011 match (best) |
| block 10 | 0.9472 / 0.9659 / 0.134 | 0.9488 / 0.9680 / 0.128 | 0.921 / 0.955 / 0.943 | 0.9028 / 0.9071 / 0.304 | +0.0003 match |
| block 11 | 0.9480 / 0.9654 / 0.135 | 0.9470 / 0.9657 / 0.138 | 0.918 / 0.958 / 0.937 | 0.9005 / 0.9052 / 0.310 | −0.0015 match |
| block 12 | 0.9472 / 0.9608 / 0.153 | 0.9461 / 0.9626 / 0.147 | 0.916 / 0.960 / 0.933 | 0.8998 / 0.8930 / 0.355 | −0.0024 match |
| block 13 | 0.9463 / 0.9573 / 0.169 | 0.9476 / 0.9594 / 0.155 | 0.918 / 0.964 / 0.931 | 0.8976 / 0.8761 / 0.413 | −0.0009 match |
| block 14 | 0.9443 / 0.9516 / 0.195 | 0.9459 / 0.9538 / 0.182 | 0.915 / 0.967 / 0.926 | 0.8963 / 0.8577 / 0.455 | −0.0026 match |
| block 15 | 0.9422 / 0.9456 / 0.235 | 0.9430 / 0.9469 / 0.227 | 0.910 / 0.968 / 0.919 | 0.8950 / 0.8268 / 0.525 | −0.0055 worse |
| block 16 | 0.9408 / 0.9386 / 0.261 | 0.9421 / 0.9391 / 0.266 | 0.908 / 0.971 / 0.915 | 0.8940 / 0.8049 / 0.577 | −0.0064 worse |
| block 17 (top) | 0.9376 / 0.9336 / 0.273 | 0.9390 / 0.9355 / 0.279 | 0.903 / 0.970 / 0.910 | 0.8924 / 0.7847 / 0.645 | −0.0095 worse |
| served head's mix (probe refit) | 0.9445 / 0.9675 / 0.142 | 0.9450 / 0.9683 / 0.131 | 0.916 / 0.940 / 0.950 | 0.9102 / 0.9137 / 0.277 | −0.0035 match |
| **served head** (as shipped) | – | **0.9485 / 0.9699 / 0.121** | 0.921 / 0.951 / 0.946 | 0.8999 / 0.9192 / 0.249 | ref |

"match" = within 0.005 F1 of the served head on AMI dev. The served head's layer-mix weights are 0.051–0.072 per
block (heaviest on blocks 1–4: 0.066 / 0.066 / 0.072 / 0.068; lightest on 10–14: 0.051).

**Reading.**
- **Blocks 2–4 vs the served head:** block 4 matches (−0.0009 F1, −0.002 AUC; 1.2 points more miss at the matched
  FPR: 13.3 vs 12.1 %); block 3 misses the 0.005 line by 0.0005 and block 2 by 0.006. Logistic regression on block 4
  is 0.9456: a linear read-out is enough from block 4 on.
- **Best block: 9** (F1 0.9496, +0.001 over the served head); blocks 4–14 are all within 0.005. AUC peaks at
  blocks 6–7 (0.969) and degrades steadily from block 12 to the top (0.936): the top blocks trade frame-accurate
  speech activity for token alignment (recall rises to 0.97 while precision falls to 0.91). The head's slight
  preference for early blocks in its learned mix is consistent with this.
- **The mix probe (0.9450) is 0.0035 below the shipped head**, so this probe recipe is a little weaker than the
  head's real training (2000 steps over three data sources with the frozen encoder in train mode). The comparison
  that holds the recipe fixed is block 4 (0.9476) vs mix (0.9450): block 4 is not worse than the mix.
- **Data matters:** with 300 train windows the same probes were 0.010–0.025 lower (block 4: 0.925, mix: 0.923;
  `rows_train300` in the JSON). The table above uses 1000 windows; a further gain from more data is likely but
  cannot change the ordering.
- **ICSI dev (held-out):** the probes on blocks 2–6 and the mix are *better* than the served head on F1
  (0.908–0.911 vs 0.900) but not on AUC (0.909–0.923 vs 0.919). Everything is worse on ICSI than on AMI (the served
  head fires on 77 % of ICSI non-speech frames at 0.5: ICSI's untimed / dropped-speaker labelling makes "non-speech"
  a noisier class there), and the ordering over blocks is the same: early-to-middle blocks best, top blocks worst.

## 2. Compute: truncated encoder, front-end, re-prime

**Setup.** AMI dev diar window 47 (IB4002 @ 60 s; 77.3 % speech, the set's mean is 76.9 %) as a 20 s window
(125 chunks of 160 ms, 5 speech onsets) and the 60 s from the same start (375 chunks, 11 onsets). Cache-aware
streaming (`FastConformerEncoder.stream_step`, the NeMo-aligned path the server uses, [70, 1]); ungated streaming
equals the offline masked forward (max |Δ| 3.4e-6 / 1.8e-5, the tests/test_streaming.py pattern). Per-chunk wall
time on 2 threads, p50 over the chunks. The truncated encoder is a weight-sharing view with `layers[:k]`
(`truncated_encoder`), asserted equal to block k's output in the full pass (max |Δ| 0).

| streaming cost per 160 ms chunk (p50 ms) | 20 s window | 60 s window | fraction of the full encoder |
|---|---|---|---|
| log-mel of the chunk (`StreamingSession._mel`) | 0.06 | 0.04 | 0.0002 |
| + subsampling (k = 0; log-mel + 8× conv front-end) | 11.4 | 14.0 | 0.05 |
| blocks 1..2 | 36.5 | 29.8 | 0.12–0.14 |
| blocks 1..3 | 48.3 | 44.6 | 0.18–0.19 |
| blocks 1..4 | 65.2 | 75.9 | 0.25–0.30 |
| blocks 1..8 | 131 | 123 | 0.48–0.51 |
| full encoder (17) | 258 | 255 | 1 |
| one block (per-layer hooks, p50 over 17 blocks) | 10.7–11.2 | 9.9–10.3 | 0.04 each |

Every block costs the same (~10–11 ms per 2-frame chunk), the front-end 5 %; so a gated encoder that skips blocks
k+1..17 during silence saves (17 − k) / 17 of the block cost while gated: **86–88 % of the chunk at k = 2,
81 % at k = 3, 70–75 % at k = 4, 49–52 % at k = 8**. Note that this Python path is ~4 % efficient: the offline
forward of the same 20 s takes ~0.7 s for 250 frames (0.16 ms per frame-block) vs 5.5 ms per frame-block here, so
these are overhead costs, not FLOPs; the fractions still hold because the overhead is per block.

**Re-prime.** `DeepRunner` keeps blocks k+1..17 with their own caches, skips them while gated (oracle any-speaker
label, a chunk is gated iff both of its 80 ms frames are non-speech), and on resume recomputes them over the last
P frames of block-k output (chunk-aligned; 35 → 36) with fresh caches, then runs the new chunk. Two
implementations, numerically identical (test): chunk by chunk (P/2 streaming steps) and one masked forward over the
P frames (`chunked_attention_mask`, as the offline pass). Errors are max |Δ| of the top-block output (LayerNorm
scale, ~1) over every un-gated chunk after the first resume, and of the VAD probability.

| k | P frames (s) | 20 s: max \|Δ\| top / VAD prob | 60 s: max \|Δ\| top / VAD prob | re-prime wall, batched (ms) | re-prime wall, chunk by chunk (ms) |
|---|---|---|---|---|---|
| 2 | 8 (0.6) | 1.74 / 0.26 | – | 209 | 654 |
| 2 | 16 (1.3) | 1.42 / 0.25 | – | 148 | 2215 |
| 2 | 36 (2.9) | 1.34 / 0.17 | – | 165 | 4395 |
| 2 | 70 (5.6) | 1.27 / 0.11 | 1.51 / 0.19 | 129–240 | 6918 |
| 2 | 140 (11) | 0.79 / 0.08 | 0.92 / 0.08 | 164–268 | 14 239 |
| 2 | 280 (22) | 1e-6\* | 5.1e-2 / 3.5e-3 | 291–358 | – |
| 2 | 420 (34) | – | **6.1e-4** / 3e-5 | 485 | – |
| 2 | 560 (45) | – | 1.4e-5 / 0 | 474 | – |
| 3 | 70 | 1.16 / 0.08 | 1.37 / 0.16 | 166–230 | 5889 |
| 3 | 140 | 0.79 / 0.08 | 0.85 / 0.08 | 228–348 | 12 206 |
| 3 | 280 | 2e-6\* | 2.6e-2 / 1.9e-3 | 246–438 | – |
| 3 | 420 | – | **2.3e-4** / 9e-6 | 405 | – |
| 4 | 8 / 16 / 36 | 1.59 / 1.32 / 1.47 | – | 194 / 194 / 321 | 660 / 1092 / 2094 |
| 4 | 70 | 0.76 / 0.09 | 1.28 / 0.09 | 206–220 | 5696 |
| 4 | 140 | 0.55 / 0.05 | 0.87 / 0.05 | 162–275 | 12 305 |
| 4 | 280 | 2e-6\* | 1.5e-2 / 9.8e-4 | 274–324 | – |
| 4 | 420 | – | **1.7e-4** / 3e-6 | 363 | – |
| 4 | 560 | – | 7e-6 / 0 | 538 | – |
| 8 | 8 / 16 / 36 | 1.44 / 1.19 / 0.24 | – | 85 / 84 / 165 | 459 / 862 / 1912 |
| 8 | 70 | 0.11 / 0.010 | 0.31 / 0.023 | 132–146 | 5005 |
| 8 | 140 | 0.010 / 5e-4 | 0.16 / 1.7e-3 | 131–193 | 6144 |
| 8 | 280 | 1e-6\* | **4.8e-4** / 5e-6 | 134–198 | – |

\* On the 20 s window P ≥ 250 is the whole history since the window start, so it is trivially exact; the 60 s
window is the informative one.

**Reading.**
- **The deep blocks need 280–420 frames of history**, not 70. The attention left context is 70 frames per block, so
  after 13–15 stacked blocks the output depends on up to ~1000 frames, and the error decays slowly: with the full
  left context of one block (P = 70) the top output is off by 0.8–1.5 and the VAD probability by 0.08–0.19; at
  P = 140 still 0.5–0.9; 1e-2 needs P = 280 (22 s) and **1e-3 needs P = 420 (34 s)** for k ≤ 4, P = 280 for k = 8.
  The first chunk after resume is as wrong as the later ones (the caches built during a short re-prime stay in the
  70-frame window for 5.6 s).
- **Re-prime wall time** in the batched form is 0.32–0.49 s per onset at P = 420 (1.3–1.9 full-encoder chunks), or
  0.13–0.20 s at k = 8 / P = 280. Chunk by chunk it is 5–14 s and useless. But the batched number is cheap only
  because the streaming path wastes 95 % of its time in per-call overhead: **in FLOPs, a P-frame re-prime is P/2
  chunk-equivalents of deep-block work** (210 at P = 420, 140 at P = 280), while every gated chunk saves one
  chunk-equivalent. Break-even is one onset per 210 gated chunks (34 s of silence per onset); AMI has one onset per
  ~2–3 s of silence (§3).

## 3. Silence fraction and expected saving

Labels on the 80 ms grid; "anyone" = any speaker active (the VAD label), "primary" = the turn windows' primary
speaker (`spk_act`). Gate = chunk (2 frames) fully non-speech, oracle; "hang 5" keeps the deep blocks running for
5 chunks (800 ms) after the last speech chunk. No Pipecat event log is checked into the repo (the demos' `ev.jsonl`
live in another agent's scratch), so the Pipecat row uses the AMI labels of the demos' own 5 windows
(`ami_{IS1008b_003, IS1008b_014, ES2011b_027, TS3004b_064, IB4002_119}`, INTEGRATION.md §4).

| set | definition | non-speech frames | gated chunks (hang 0 / 5) | onsets per min (hang 0 / 5) |
|---|---|---|---|---|
| AMI dev diar 64 (the VAD set, 1280 s) | anyone | **23.1 %** | 22.3 / 14.8 % | 6.1 / 3.3 |
| AMI dev turn 64 (1046 s) | anyone | 20.2 % | 19.1 / 9.5 % | 7.6 / 3.0 |
| AMI dev turn 64 | primary | **59.2 %** | 58.3 / 40.1 % | 7.2 / 7.0 |
| Pipecat demo windows (5, 98 s) | anyone | 12.7 % | 11.8 / 4.5 % | 4.8 / 1.4 |
| Pipecat demo windows | primary | 22.6 % | 21.7 / 11.0 % | 5.4 / 3.4 |

**Expected saving, wall-time model of this Python path** (cost per chunk = c_k while gated, c_17 otherwise, from
the 60 s window; plus one batched re-prime per onset at the cheapest P with top error < 1e-3: 420 frames for
k = 2–4, 280 for k = 8):

| set / definition | k = 2 | k = 3 | k = 4 | k = 8 |
|---|---|---|---|---|
| AMI dev diar, anyone, hang 0 | 16.8 % | 15.9 % | 13.6 % | 10.1 % |
| AMI dev diar, anyone, hang 5 | 11.5 % | 10.9 % | 9.3 % | 6.9 % |
| AMI dev turn, anyone, hang 0 / 5 | 13.2 / 7.0 % | 12.7 / 6.6 % | 10.9 / 5.7 % | 8.1 / 4.2 % |
| AMI dev turn, primary, hang 0 / 5 | 48.1 / 32.1 % | 45.2 / 30.2 % | 38.6 / 25.8 % | 28.4 / 19.0 % |
| Pipecat windows, anyone, hang 0 / 5 | 8.1 / 3.4 % | 7.8 / 3.2 % | 6.7 / 2.7 % | 5.0 / 2.0 % |
| Pipecat windows, primary, hang 0 / 5 | 16.6 / 8.1 % | 15.7 / 7.7 % | 13.5 / 6.6 % | 9.9 / 4.9 % |

**Expected saving in FLOPs** (what a compiled runtime such as `audioforge.runtime` would pay; unit = one block on
one frame): per minute of AMI dev diar audio at hang 0 the gate covers 84 chunks = 167 frame-blocks × (17 − k) saved,
and 6.1 onsets × 420 frames = 2580 frame-blocks × (17 − k) re-primed: **a 15× net loss**, independent of k. With
hang 5: 111 saved vs 1400 re-primed (13×). At P = 70 (outputs wrong by O(1)) it is still 430 vs 167. The
primary-speaker definition (59 % non-speech) does not rescue it (7 onsets per minute), and it would also blind the
diarizer, the turn head and the ASR to the other speakers while gated, which is the opposite of what the
speaker-aware stack is for. On a single-user agent microphone "primary" ≈ "anyone", i.e. the 13–23 % rows.

## 4. Verdict and recommendation

- **Q1: yes for block 4, no for blocks 2–3.** A 33 K probe on block 4 matches the served head within 0.005 F1
  (0.9476 vs 0.9485; AUC 0.968 vs 0.970; +1.2 points miss at FPR 0.075). Blocks 4–14 all match; block 9 is the best
  single block (0.9496); the top block is the worst input (0.939). The served head's near-uniform mix over 17 blocks
  buys nothing measurable over block 4 (mix probe with the same recipe: 0.9450).
- **Q2: a gated encoder costs more than it saves.** Skipping blocks k+1..17 saves 70–88 % of a chunk while gated,
  but restoring their state on resume needs 280–420 frames (22–34 s) of recomputation for 1e-2–1e-3 agreement, i.e.
  140–210 chunk-equivalents of deep work per speech onset, against ~10 gated chunks per onset in AMI (6–8 onsets per
  minute). In FLOPs it loses by an order of magnitude; the 10–17 % wall-time "saving" exists only in the
  overhead-bound Python path and disappears in any runtime worth deploying. Inexact re-priming (P ≤ 70) leaves the
  encoder output wrong by O(1) for 5.6 s after every onset, which the ASR and the heads would see.
- **Q3:** 23 % of AMI dev frames (13 % of the demo windows) are non-speech for anyone; 59 % (23 %) for the primary
  speaker. The any-speaker definition bounds any encoder-level saving at 12–23 % of the block cost before re-prime.
- **Recommendation.** Keep the encoder ungated. Two cheap things are worth doing instead:
  1. If a VAD is wanted before the full encoder finishes (e.g. to start the decoder loop or a diarizer step early,
     or on a device that streams blocks over time), read it from **block 4**: same accuracy, 25–30 % of the chunk
     cost, no state problem. A `from_layers: [3]` frame head is a one-line recipe change (audioforge/model.py
     single-tap support) and needs a short head-only training, not this probe.
  2. If compute must drop during silence, gate the *consumers* that carry no long state or can be re-run cheaply
     (RNNT/TDT decoding steps, the turn head, the speaker-conditioned second encoder pass that INTEGRATION.md prices
     at RTF +0.1–0.15, the Sortformer 0.32 s preset), not the encoder. A runtime port of the encoder
     (ONDEVICE.md: ~30 ms per chunk) is the actual lever on encoder cost.

## Commands

```
W=<scratch>/vad_layers
for s in ami_dev icsi_dev; do .venv/bin/python scripts/vad_layers.py features --set $s --work $W; done
.venv/bin/python scripts/vad_layers.py features --set train --n-train 1000 --work $W    # ~6 min, resumable
.venv/bin/python scripts/vad_layers.py probes --n-train 1000 --work $W                   # ~30 s per block, resumable
.venv/bin/python scripts/vad_layers.py compute --window 20 --k 2,3 --work $W             # each < 10 min
.venv/bin/python scripts/vad_layers.py compute --window 20 --k 4,8 --work $W
.venv/bin/python scripts/vad_layers.py compute --window 60 --k 17 --primes "" --work $W
.venv/bin/python scripts/vad_layers.py compute --window 60 --k 2,3 --primes 70,140,280 --work $W
.venv/bin/python scripts/vad_layers.py compute --window 60 --k 4,8 --primes 70,140,280 --work $W
.venv/bin/python scripts/vad_layers.py compute --window 60 --k 2,3,4 --primes 420,560 --work $W
.venv/bin/python scripts/vad_layers.py silence && .venv/bin/python scripts/vad_layers.py report
.venv/bin/python -m pytest -q tests/test_vad_layers.py
```
