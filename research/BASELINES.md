# Baselines: our heads vs dedicated open models

## Turn detection

2026-09-26. Our turn detection vs the dedicated open turn-detection models on **the same data and scorer**: eot-bench
v2 (research/EOT_BENCH_V2.md), all 974 AMI dev turn windows. The 6 s horizon uses block C (6 s-extended windows,
emission horizon 75 frames). The 2 s horizon uses block A (the default 2 s-trail windows). Where a speaker track is
needed, the enrollment is label-free (`causal_dominant`) on the 1.04 s streaming Sortformer v2 tracks. Every system
gets a fixed threshold at ≤ 5 % false cutoffs per turn, chosen on one leave-meetings-out half and applied to the
other. Strata are floor-open (236) and taken (738). CIs are 95 % bootstraps over turns (1000 resamples), and
differences are paired bootstraps against our hybrid.

- **Code.** Baselines: `audioforge/baselines/turn.py`. Stages and scorer: `scripts/bench_turn_baselines.py`.
  EOU importer: `nemo_import.import_eou`. Tests: `tests/test_baselines_turn.py`.
- **Numbers.** `runs/baselines_turn.json`. Our rows are recomputed in the same run from the stored trail6 head
  scores and tracks, and asserted equal to `runs/turn_trail6_hybrid_leakfree.json` (miss, FC and CI match exactly).
  The scorer is the same code (`eval_stage1._emit_transform / v2_crossfit / v2_hybrid_grid / _paired`,
  `conversation.eot_outcomes[_or]`), so nothing needed recomputing for consistency.

### How each baseline becomes a per-frame score

Each baseline gives one score per 80 ms frame. A score on frame t uses input that ends inside frame t. The emission
rule adds only input buffering: t + 1 for frame-level decisions, the Sortformer rule for Sortformer triggers, and
160 ms chunks for the EOU model. Compute time is reported separately, as it is for our rows.

| system | what it uses (speaker awareness) | score track | license / size |
|---|---|---|---|
| **hybrid trail6 (ours)** | audio + Sortformer primary track with causal enrollment (**audio, speaker-aware**) | head p OR primary-silence timeout, joint (θ, k) | ours |
| Silero VAD timeout | Silero VAD v5 (**none**) | silence since the last speech chunk, from Pipecat's VAD state machine (conf 0.7, start/stop 0.2 s) | MIT, 2.3 MB ONNX |
| Pipecat smart-turn v3.2 (`smart-turn-v3.2-cpu.onnx`) | audio, Silero trigger (**none**) or Sortformer trigger (**speaker-aware trigger**) | P(complete) of the last ≤ 8 s, run at each Pipecat-VAD stop (0.2 s silence) or each ≥ 200 ms (3-frame) silence of the bound Sortformer primary; held until speech resumes. "+ timeout" = OR with the Silero timeout (Pipecat's `stop_secs` fallback) or with our Sortformer timeout, joint (θ, k) | BSD-2-Clause, 8M params, 8.7 MB int8 |
| LiveKit text turn detector (`EnglishModel` v1.2.2-en, `MultilingualModel` v0.4.1-intl) | **text**: (a) our streaming ASR partials (stage1_heads_pretrained, 160 ms, all speakers mixed), (b) oracle primary words (speaker-aware oracle text) | P(EOU) of a one-message chat context, at every frame after a LiveKit-Silero end-of-speech (activation 0.5, min silence 0.55 s) until speech restarts. "+ timeout" = OR with silence since the speech end (LiveKit `max_delay`) | LiveKit Model License (use only inside LiveKit Agents); 66 MB / 396 MB q8 ONNX |
| LiveKit audio `turn-detector-v1-mini` (`livekit.local_inference.EOT`) | audio, last 1.2 s (**none**) | P(turn ended), predicted at 0.2 s of raw silence and applied from end-of-speech (0.55 s), as in `AudioRecognition`; held until speech restarts | Apache-2.0 + LiveKit Model License; ~108 MB resident |
| NVIDIA Parakeet-Realtime-EOU 120m-v1 | audio, all speakers (**none**) | per-frame log P(`<EOU>`) (max over the frame's RNNT symbol steps); native rule = `<EOU>` emitted | **NVIDIA Open Model License** (the card; not CC-BY-4.0); 120M (115.0M loaded), 460 MB .nemo |

Implementation notes:
- **LiveKit.** Run through the plugin's own runners (`_EUORunnerEn`/`_EUORunnerMultilingual`: chat template,
  128-token left truncation, ONNX q8), with the session rebuilt at 2 threads. No LiveKit runtime was needed.
- **Pipecat.** The Pipecat library is not installed. Its `VADAnalyzer` state machine and
  `LocalSmartTurnAnalyzerV3` inference (left zero-pad to 8 s, Whisper features) are re-implemented from source.
  The `min_volume` gate is not modelled.
- **EOU import.** It works. The only gap in the generic importer is `encoder.use_bias: false`: 187 encoder bias
  tensors are filled with zeros, which is exact. `import_nemo` itself is unchanged and still rejects the file.
  `<EOU>` = 1024, `<EOB>` = 1025, blank = 1026.
- **EOU decoding.** The encoder runs once per window with its [70, 1] chunked_limited mask, which is what
  cache-aware streaming computes (max |Δ| 1e-5 vs `model.StreamingSession`; identical tokens; test). It gets 0.32 s
  of the following meeting audio so the last frame has no end-of-file artifact. The same applies to our ASR partials.
- **EOU posterior.** Posteriors are kept in the log domain because float32 underflows far from an emission. The
  cross-fitted thresholds are log P > −62.5 / −89.2, so at ≤ 5 % FC the model is used as a ranking well below its
  own `<EOU>` decision.

### Results (miss % at the cross-fitted ≤ 5 % per-turn FC point, [95 % CI]; Δ = system − our hybrid, paired, points)

| system | 6 s miss, all | 6 s open | 6 s taken | 6 s FC turn / pause | 6 s P50 open | 2 s miss, all | 2 s open | 2 s taken | Δ vs hybrid 6 s: all / open | Δ vs hybrid 2 s: all / open |
|---|---|---|---|---|---|---|---|---|---|---|
| **hybrid trail6 (ours)** | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 67.0 | 6.5 / 5.3 | 4880 ms | 79.1 [76.6, 81.9] | 72.9 [67.7, 78.9] | 81.1 | ref / ref | ref / ref |
| head trail6 (ours) | 66.5 [63.4, 69.7] | 57.3 [50.9, 63.9] | 69.4 | 5.3 / 5.3 | inf | 79.8 [77.4, 82.5] | 75.0 [69.4, 80.7] | 81.3 |  /  |  /  |
| timeout, Sortformer primary (ours) | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 83.6 | 5.7 / 3.9 | 4080 ms | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 |  /  |  /  |
| timeout, Sortformer any-speaker | 75.5 [72.7, 78.1] | 31.3 [25.4, 37.4] | 88.6 | 5.2 / 3.4 | 2880 ms | 88.2 [86.1, 90.2] | 54.0 [47.0, 60.6] | 98.3 |  /  |  /  |
| Silero VAD timeout | 72.7 [69.8, 75.4] | 26.7 [21.1, 32.4] | 86.8 | 4.8 / 5.3 | 2000 ms | 83.5 [81.1, 85.8] | 39.6 [32.7, 45.7] | 96.9 | +10.8 [+7.0, +14.2] / -19.1 [-27.0, -11.1] | +4.3 [+0.8, +7.9] / -33.3 [-41.7, -24.8] |
| Parakeet-Realtime-EOU, P(<EOU>) | 70.1 [67.0, 73.0] | 42.9 [36.4, 49.5] | 78.4 | 5.3 / 8.7 | 3040 ms | 81.6 [78.8, 84.0] | 59.5 [52.8, 66.1] | 88.4 | +8.2 [+4.2, +11.8] / -3.0 [-11.9, +5.7] | +2.4 [-1.6, +5.9] / -13.5 [-22.5, -4.5] |
| smart-turn v3.2, Silero trigger | 98.9 [98.3, 99.5] | 99.6 [98.7, 100.0] | 98.7 | 4.6 / 3.4 | inf | 99.0 [98.4, 99.6] | 99.6 [98.7, 100.0] | 98.9 | +37.0 [+33.8, +40.3] / +53.7 [+46.7, +60.7] | +19.9 [+17.0, +22.5] / +26.6 [+20.3, +33.0] |
| smart-turn + Silero timeout (Pipecat) | 72.6 [69.7, 75.3] | 26.7 [21.1, 32.4] | 86.7 | 5.2 / 5.8 | 2000 ms | 83.4 [81.0, 85.7] | 39.6 [32.7, 45.7] | 96.9 | +10.7 [+6.9, +14.1] / -19.1 [-27.0, -11.1] | +4.3 [+0.7, +7.9] / -33.3 [-41.7, -24.8] |
| smart-turn v3.2, Sortformer trigger | 98.4 [97.5, 99.1] | 98.2 [96.4, 99.6] | 98.4 | 4.7 / 4.8 | inf | 98.5 [97.7, 99.2] | 98.2 [96.4, 99.6] | 98.6 | +36.5 [+33.3, +39.6] / +52.3 [+45.3, +59.4] | +19.4 [+16.4, +22.0] / +25.3 [+18.9, +31.6] |
| smart-turn (Sortformer trig.) + our timeout | 74.7 [71.9, 77.5] | 46.1 [39.6, 52.6] | 83.5 | 5.9 / 3.9 | 4080 ms | 92.5 [90.8, 94.1] | 81.1 [75.8, 86.0] | 96.0 | +12.8 [+9.6, +15.9] / +0.2 [-5.9, +6.7] | +13.3 [+10.5, +16.1] / +8.2 [+1.7, +15.2] |
| LiveKit audio v1-mini | 95.5 [94.1, 96.7] | 89.6 [85.1, 93.3] | 97.3 | 4.3 / 8.7 | inf | 96.8 [95.6, 97.8] | 90.5 [86.3, 94.2] | 98.7 | +33.6 [+30.2, +36.9] / +43.8 [+35.7, +51.7] | +17.6 [+14.5, +20.5] / +17.6 [+9.8, +25.1] |
| LiveKit audio v1-mini + 3 s-style timeout | 74.5 [71.5, 77.1] | 28.8 [23.0, 34.8] | 87.7 | 5.1 / 2.9 | 2080 ms | 89.8 [87.9, 91.8] | 60.1 [53.6, 67.0] | 98.5 | +12.6 [+8.9, +16.0] / -17.0 [-24.9, -9.1] | +10.7 [+7.2, +13.8] / -12.8 [-21.5, -3.7] |
| LiveKit text EN, our ASR | 91.9 [90.1, 93.6] | 88.9 [84.7, 93.1] | 92.8 | 4.7 / 8.7 | inf | 95.7 [94.3, 97.0] | 92.2 [88.5, 95.5] | 96.8 | +30.0 [+26.3, +33.7] / +43.1 [+34.7, +51.8] | +16.6 [+13.3, +19.5] / +19.2 [+12.1, +27.1] |
| LiveKit text EN, oracle words | 94.2 [92.7, 95.6] | 92.6 [89.1, 95.8] | 94.7 | 4.6 / 10.1 | inf | 95.6 [94.2, 96.9] | 92.6 [89.1, 95.8] | 96.5 | +32.3 [+28.7, +35.6] / +46.8 [+38.5, +54.5] | +16.4 [+13.5, +19.3] / +19.7 [+12.2, +26.7] |
| LiveKit text multilingual, our ASR | 90.4 [88.4, 92.3] | 85.8 [80.9, 90.2] | 91.9 | 5.5 / 8.7 | inf | 93.2 [91.5, 94.7] | 88.6 [83.9, 92.6] | 94.6 | +28.5 [+24.6, +32.1] / +40.0 [+31.6, +48.5] | +14.0 [+10.7, +17.1] / +15.7 [+8.4, +23.8] |
| LiveKit text multilingual, oracle words | 90.7 [88.8, 92.5] | 85.8 [80.7, 90.0] | 92.2 | 4.0 / 6.8 | inf | 92.7 [91.0, 94.5] | 85.8 [80.7, 90.0] | 94.8 | +28.8 [+25.0, +32.1] / +39.9 [+31.3, +48.1] | +13.6 [+10.4, +16.7] / +12.8 [+4.1, +20.6] |
| LiveKit text EN, ASR + timeout | 74.2 [71.2, 77.0] | 30.9 [24.8, 37.2] | 86.8 | 5.7 / 2.9 | 2320 ms | 89.7 [87.8, 91.8] | 61.9 [54.5, 68.9] | 97.6 | +12.3 [+8.4, +15.7] / -14.9 [-22.7, -6.8] | +10.5 [+7.1, +13.9] / -11.1 [-19.9, -1.6] |
| LiveKit text EN, oracle + timeout | 73.7 [70.9, 76.3] | 28.7 [22.9, 34.6] | 86.9 | 5.4 / 1.9 | 2080 ms | 90.1 [88.1, 92.0] | 63.7 [56.9, 70.2] | 97.6 | +11.8 [+8.2, +15.2] / -17.2 [-25.1, -9.6] | +11.0 [+7.6, +14.3] / -9.2 [-17.6, -0.2] |
| LiveKit text multi, ASR + timeout | 73.9 [71.0, 76.6] | 28.2 [22.2, 34.1] | 87.3 | 5.1 / 1.0 | 2080 ms | 90.5 [88.6, 92.3] | 66.0 [59.1, 72.4] | 97.6 | +12.0 [+8.5, +15.5] / -17.6 [-25.4, -9.8] | +11.4 [+8.0, +14.6] / -6.9 [-15.7, +2.2] |
| LiveKit text multi, oracle + timeout | 74.0 [70.9, 76.6] | 30.5 [23.9, 37.0] | 86.4 | 6.2 / 4.3 | 2400 ms | 88.8 [86.6, 90.7] | 63.3 [56.4, 69.4] | 95.9 | +12.0 [+8.5, +15.4] / -15.3 [-23.1, -6.9] | +9.6 [+6.0, +13.1] / -9.6 [-18.3, -0.5] |

| out-of-the-box rule (no fitting) | 6 s miss, all | 6 s open | FC per turn / per pause | P50 all | 2 s miss, all (FC) |
|---|---|---|---|---|---|
| pipecat_native (smart-turn P>0.5 OR 3 s silence) | 48.3 [44.0, 52.7] | 20.2 | 49.8 / 42.5 | 5360 ms | 63.8 (49.8) |
| livekit_audio_native (P>=0.36 at EOS OR 3 s) | 48.6 [44.4, 52.5] | 10.4 | 35.5 / 69.1 | 5120 ms | 64.0 (35.5) |
| eou_native (<EOU> emitted) | 79.1 [76.4, 81.5] | 60.2 | 3.0 / 3.9 | inf | 89.3 (3.0) |
| livekit_text_en_asr_native (P>=0.0289 at EOS OR 3 s) | 59.4 [56.2, 62.8] | 26.3 | 20.0 / 33.3 | inf | 78.4 (20.0) |
| livekit_text_en_oracle_native (P>=0.0289 at EOS OR 3 s) | 59.9 [56.8, 63.3] | 28.2 | 17.8 / 38.6 | inf | 75.7 (17.8) |
| livekit_text_multilingual_asr_native (P>=0.011 at EOS OR 3 s) | 55.0 [51.3, 58.7] | 21.4 | 27.4 / 48.8 | inf | 72.6 (27.4) |
| livekit_text_multilingual_oracle_native (P>=0.011 at EOS OR 3 s) | 50.2 [46.7, 53.8] | 20.6 | 24.1 / 55.6 | inf | 66.2 (24.1) |

Exploratory (chosen after seeing the table, same dev data): **our head OR the Silero timeout**, replacing the
Sortformer-primary timeout in the hybrid:
- 6 s: 59.6 [56.3, 62.6] all, open 34.3 [28.6, 40.4], taken 67.3; held-out FC 6.6 % per turn, 6.3 % per pause.
  Δ vs our hybrid: all −2.4 [−4.2, −0.5], open −11.6 [−17.2, −6.1], taken +0.3 [−1.4, +2.1].
- 2 s (A): 78.4 [75.7, 81.0], open 58.1 [51.2, 64.7]. Δ: all −0.8 [−3.0, +1.4], open −14.8 [−21.2, −8.6].
- Our hybrid's rows vs our own components are in EOT_BENCH_V2.md §7 (e.g. hybrid − head at 6 s: −4.6 [−6.4, −2.9]).

**Latency budget** (this CPU, 2 threads; model compute is not in the emission rule):

| system | trigger / buffering before a decision can exist | compute per decision |
|---|---|---|
| hybrid (ours) | Sortformer 1.04 s config: C + R = 13-frame buffer, mean emission delay 840 ms (the 0.32 s preset: 240 ms, EOT_BENCH_V2 §7B); head chunk 160 ms; plus the fitted timeout k ≈ 4 s backstop | full serve.py stack (ASR + heads + Sortformer) RTF 0.45–0.52, chunk p50 38 ms (INTEGRATION.md) |
| Silero timeout | 32 ms chunks; the fitted timeout is k ≈ 1.8–1.9 s | 0.08 ms / 32 ms chunk |
| smart-turn | Pipecat VAD stop = 6 chunks (192 ms) of silence | 22 ms / call (v3.2-cpu int8 + Whisper features) |
| LiveKit text | end-of-speech after 0.55 s of silence; ASR partial on 160 ms chunks | 1.8 ms (EN) / 13.5 ms (multilingual) per call, + the ASR |
| LiveKit audio v1-mini | predicted at 0.2 s of silence, used at end-of-speech (0.55 s) | 6.2 ms / call |
| Parakeet-Realtime-EOU | 160 ms chunks (att context [70, 1]) | 176 ms per 160 ms chunk in `model.StreamingSession` (Python path: **not real time** on 2 threads). A same-size 17×512 encoder runs at about 30 ms/chunk in `audioforge.runtime` (ONDEVICE.md; not measured for this checkpoint). Offline masked pass: 0.17 s per window |

### Reading

- **Overall, we are better than every dedicated model at 6 s, and tied or better at 2 s.**
  - At 6 s, every model is worse than our hybrid (61.9 %): NVIDIA's EOU +8.2 [+4.2, +11.8], Silero timeout
    +10.8, Pipecat (smart-turn + timeout) +10.7, LiveKit text + timeout +11.8 to +12.3, LiveKit audio + timeout
    +12.6.
  - At 2 s, only the EOU model is statistically tied (+2.4 [−1.6, +5.9]).
- **The whole advantage comes from floor-taken ends (67.0 % vs 78–88 %).** At those ends another person speaks
  within 1 s or overlaps. A speaker-unaware detector never sees silence there, and only a speaker-aware system can
  resolve them.
- **On floor-open ends (the ones an agent must answer) we are worse.**
  - A plain speaker-unaware Silero VAD timeout misses 26.7 % vs our 45.9 % at 6 s (Δ −19.1 [−27.0, −11.1]), and
    39.6 % vs 72.9 % at 2 s (Δ −33.3).
  - The EOU model ties us on open ends at 6 s (−3.0, CI contains 0) and beats us at 2 s (−13.5).
  - Why: our deployable track loses the user on open ends. Causal enrollment agrees with the oracle column at only
    70 % of turn ends, the causal primary is "active" on 38 % of post-end frames (EOT_BENCH_V2 §7), and the
    diarizer adds 840 ms. At a floor-open end, silence from anyone is exactly the right signal. Silero is a
    cleaner and faster any-speaker VAD than the Sortformer column max (26.7 vs 31.3 % open).
  - The exploratory row confirms this: our head OR the Silero timeout is better than our hybrid on open ends
    (−11.6 at 6 s) and overall (−2.4). Adding an any-speaker VAD branch is the cheap fix to try in serve.py. It
    must be confirmed on held-out data (e.g. ICSI dev), since it was picked after seeing these numbers.
    **Confirmed** on 1312 held-out ICSI turns: see "Turn detection: ICSI held-out confirmation" below.
- **The dedicated turn models add nothing over their own VAD timeout on meeting audio at ≤ 5 % FC.**
  - Smart-turn + timeout = Silero timeout (72.6 vs 72.7), and LiveKit + timeout ≈ 74.
  - Alone they miss 90–99 %, even with a speaker-aware trigger (smart-turn on Sortformer silences: 98.4 %).
  - Their P(complete) barely separates within-turn pauses from true ends in AMI. Smart-turn's AUC is 0.58 between
    triggers inside the turn and the first trigger after the end, against 0.70 on LibriSpeech complete vs cut
    utterances. So the per-turn FC budget pushes θ to about 0.99.
  - Out of the box they cut users off in 18–50 % of turns (Pipecat 49.8 %, LiveKit audio 35.5 %, LiveKit text
    18–27 %).
  - For LiveKit text, oracle words do not help (EN 94.2 % oracle vs 91.9 % ASR), so ASR errors are not the limit.
    AMI's lowercase, unpunctuated conversational text is off-distribution. The punctuation-normalizing
    multilingual model does slightly better.
- **Dead air.** P50 is infinite for every fitted system at 6 s over all turns (miss > 50 %). On open ends:
  Silero timeout 2000 ms, EOU 3040 ms, our hybrid 4880 ms.
- **Caveats.**
  - Held-out per-turn FC above 5 % for the joint-grid rows: our hybrid 6.5 %, LiveKit + timeout up to 6.2 %.
  - Pipecat's segment start (speech start − 0.7 s) is simplified to the last 8 s.
  - LiveKit's `DynamicEndpointing` adaptation is not modelled, and the chat context holds only the current text.

**Not run:**
- LiveKit's cloud `turn-detector-v1` (needs LiveKit Cloud credentials).
- smart-turn v3.2-gpu (fp32 variant of the same model).
- The real `livekit-plugins-silero` and Pipecat packages (not installed; their VAD logic is re-implemented as above).

### Turn detection: ICSI held-out confirmation

2026-09-26. The exploratory rule above (**our trail6 head OR the Silero timeout**) was picked after seeing the AMI dev
table. This re-runs eot-bench v2 on ICSI meetings that no fitting ever touched and scores every rule two ways.
Code: `scripts/bench_turn_icsi.py` (stages tracks / vad / scores / report); numbers: `runs/baselines_turn_icsi.json`;
test: `tests/test_bench_turn_icsi.py`.

- **Set.** ICSI dev (Bmr021, Bns001) alone gives only 465 extended windows (85 floor-open) after the untimed-zone and
  > 4-speaker filter, so the set is **all 5 held-out ICSI meetings** (Kaldi dev + eval: Bmr021 Bns001 Bmr013 Bmr018
  Bro021; the other 12 ICSI meetings with audio are the train subset, and the trail6 head was trained on AMI only):
  **n = 1312 turns, floor-open 216, taken 1096, 1991 within-turn pauses** (AMI dev: 974 / 236 / 738 / 207). Same
  rules as AMI: block C = the library's 20 s windows re-cut with a 6 s trail from the same start (stopping at the
  primary's next turn; 747 resume / 565 trail), horizons 25 / 75 post-end emission frames; block A = the default
  2 s-trail windows; causal_dominant enrollment on new 1.04 s streaming Sortformer tracks (`eval_stage1.v2_tracks`,
  real-audio right padding, the same code and flags as the AMI trail6 dev tracks; the existing
  `data/icsi/cache/sortformer/dev` tracks are 2 s-trail windows and have no post-end frames beyond 2 s). Extended
  windows that reach an untimed zone are dropped too, so every scored frame is labelled. Dev-only and eval-only
  subsets are reported as well.
- **Frozen reading.** Every operating point is fitted on all 974 AMI dev turns (the in-sample ≤ 5 % per-turn FC point
  of the same grids; the all-AMI hybrid fit reproduces the committed in-sample point θ 0.99816 / k 53 exactly) and
  applied to ICSI unchanged. 6 s points: hybrid (θ 0.998164, k 53 = 4.24 s of Sortformer-primary silence);
  **head OR Silero (θ 0.998285, Silero silence > 32.8 frames, i.e. k 33 = 2.64 s)**; Silero alone k 24 (1.92 s);
  Sortformer-primary timeout k 37 (2.96 s); head θ 0.998168. The AMI table's two per-fold points are scored too
  (a sensitivity range; the head-OR-Silero folds are (0.998285, k 36) / (0.998162, k 29)).
- **Cross-fit reading.** The AMI protocol re-run on ICSI: the point chosen on one leave-meetings-out half
  ((Bmr021, Bmr018) = 619 turns / (Bns001, Bmr013, Bro021) = 693) and applied to the other.

**Frozen AMI points on ICSI** (miss % [95 % CI]; FC per turn / per pause; Δ = system − hybrid, paired bootstrap):

| system | 6 s miss, all | 6 s open | 6 s taken | FC turn / pause | 2 s (A) all | 2 s open | Δ vs hybrid 6 s: all / open / taken | Δ 2 s (A): all / open |
|---|---|---|---|---|---|---|---|---|
| **hybrid trail6 (ours)** | 82.1 [79.8, 84.4] | 71.5 [65.7, 77.3] | 84.2 | 1.4 / 0.2 | 91.8 [90.3, 93.3] | 89.2 | ref | ref |
| head trail6 | 82.9 [80.7, 85.1] | 75.2 [69.6, 80.8] | 84.4 | 1.1 / 0.2 | 91.8 | 89.2 | +0.8 [+0.3, +1.4] / +3.7 [+1.4, +6.5] / +0.2 | 0.0 / 0.0 |
| timeout, Sortformer primary | 91.6 [90.2, 93.2] | 77.8 [72.2, 83.2] | 94.4 | 1.4 / 0.1 | 96.8 | 94.8 | +9.6 [+7.6, +11.7] / +6.3 [+0.8, +12.4] / +10.2 | +5.0 / +5.6 |
| timeout, Sortformer any-speaker | 93.4 [92.0, 94.7] | 68.6 [62.5, 74.3] | 98.2 | 0.8 / 0.1 | 99.4 | 96.7 | +11.3 [+9.0, +13.8] / −2.9 [−10.8, +4.9] / +14.0 | +7.6 / +7.4 |
| Silero VAD timeout | 87.4 [85.5, 89.1] | 45.5 [38.4, 52.5] | 95.2 | 3.6 / 1.1 | 92.2 | 58.5 | +5.3 [+2.8, +7.8] / −26.0 [−34.3, −17.5] / +11.0 | +0.4 / −30.8 |
| **head trail6 OR Silero timeout** | **81.7 [79.5, 83.9]** | **62.2 [56.1, 68.7]** | 85.5 | **2.4 / 0.6** | 91.5 | 88.7 | **−0.4 [−1.8, +1.0] / −9.3 [−14.7, −4.1] / +1.4 [+0.1, +2.8]** | −0.3 [−0.7, −0.1] / −0.6 [−1.7, 0.0] |
| oracle: timeout on primary labels | 9.8 [8.2, 11.6] | 3.0 [1.0, 5.6] | 11.1 | 4.4 / 3.2 | 9.8 | 3.0 | −72.3 / −68.5 / −73.1 | −82.0 / −86.2 |
| oracle: timeout on any-speaker labels | 82.3 [80.3, 84.5] | 27.6 [21.4, 33.7] | 92.4 | 4.0 / 3.0 | 87.6 | 31.1 | +0.2 [−2.4, +2.9] / −43.9 [−51.9, −35.4] / +8.2 | −4.2 / −58.1 |

- The AMI-fold points bracket the frozen row: head OR Silero at 6 s misses 82.7 / 77.7 % overall and 65.5 / 52.4 %
  on open ends (FC 2.2 / 2.9 %) at the fold-0 (k 36) / fold-1 (k 29) points; the hybrid 83.4 / 80.8 % (open
  72.0 / 69.0, FC 1.1 / 1.8 %). The open-end gain holds at every AMI point (−6.5 to −16.6).
- Subsets, 6 s, frozen all-AMI point: **dev** (465 turns, 85 open) hybrid 85.9 % / open 71.1 / FC 2.1 vs head OR
  Silero 85.0 / **65.1** / 2.6; **eval** (847 turns, 131 open) hybrid 80.0 / 71.8 / 0.9 vs 79.9 / **60.3** / 2.4.
- Block C at the 2 s emission horizon (not in the table): the all-AMI head-OR-Silero point for that horizon is
  effectively Silero alone (θ 0.99936, k 24): 92.8 % all / 62.0 open vs hybrid 98.0 / 96.7 (Δ −5.2 / −34.7),
  FC 3.6 %.

**Cross-fitted on ICSI** (the AMI protocol; miss % [CI]; held-out FC per turn / per pause; Δ vs hybrid, paired):

| system | 6 s miss, all | 6 s open | 6 s taken | FC turn / pause | chosen (θ, k) per fold, 6 s | 2 s (A) all | 2 s open | 2 s FC | Δ vs hybrid 6 s: all / open / taken | Δ 2 s (A): all / open |
|---|---|---|---|---|---|---|---|---|---|---|
| **hybrid trail6 (ours)** | 68.5 [65.8, 71.2] | 55.2 [48.5, 61.8] | 71.1 | 5.9 / 1.4 | (0.99648, 52) / (0.99773, 29) | 79.6 [77.3, 81.8] | 80.2 | 5.0 | ref | ref |
| head trail6 | 69.1 [66.6, 71.6] | 65.2 [58.9, 71.8] | 69.8 | 4.9 / 1.1 | θ 0.99649 / 0.99733 | 79.9 | 80.2 | 4.9 | +0.6 [−0.9, +2.1] / +10.0 [+4.5, +15.4] / −1.2 | +0.3 / 0.0 |
| timeout, Sortformer primary | 85.3 [83.4, 87.3] | 60.9 [54.6, 66.7] | 90.1 | 3.4 / 1.2 | k 26 / 26 | 94.6 | 89.9 | 3.4 | +16.8 / +5.7 [+0.1, +11.6] / +19.0 | +15.0 / +9.7 |
| timeout, Sortformer any-speaker | 84.6 [82.5, 86.6] | 35.2 [28.3, 42.5] | 93.2 | 7.2 / 3.6 | k 18 / 12 | 88.7 | 40.1 | 7.2 | +16.1 / −20.1 [−28.8, −11.9] / +22.2 | +9.1 / −40.1 |
| Silero VAD timeout | 84.4 [82.3, 86.4] | 37.4 [30.1, 44.4] | 92.7 | 8.4 / 3.7 | k 24 / 17 | 89.7 | 46.9 | 8.4 | +15.9 / −17.8 [−26.6, −9.9] / +21.6 | +10.1 / −33.3 |
| **head trail6 OR Silero timeout** | **65.2 [62.4, 68.0]** | **36.3 [29.5, 42.9]** | 70.6 | **7.7 / 2.4** | (0.99650, 58) / (0.99774, 22) | **76.5 [73.8, 79.0]** | **49.5** | 7.7 | **−3.3 [−4.8, −1.7] / −18.9 [−25.1, −12.4] / −0.5 [−1.8, +0.9]** | −3.1 [−4.9, −1.5] / −30.7 [−37.8, −23.9] |
| oracle: timeout on primary labels | 10.0 [8.3, 11.7] | 2.1 | 11.5 | 5.1 / 3.8 | k 20 / 17 | 10.0 | 2.1 | 5.1 | −58.5 / −53.2 / −59.6 | −69.6 / −78.1 |
| oracle: timeout on any-speaker labels | 80.9 [78.8, 83.1] | 18.7 | 91.7 | 6.8 / 5.0 | k 19 / 15 | 86.0 | 20.9 | 6.8 | +12.4 / −36.5 / +20.7 | +6.4 / −59.3 |

**Reading.**

- **The open-end gain of head OR Silero is confirmed on held-out data.** At the frozen AMI point it misses 62.2 %
  of floor-open ends vs the hybrid's 71.5 % (Δ **−9.3 [−14.7, −4.1]**, CI excludes 0), on both the dev (−6.0) and
  the eval (−11.5) meetings and at every AMI fold point; overall it is tied (−0.4 [−1.8, +1.0]) and taken ends are
  1.4 points worse [+0.1, +2.8]. Re-fitted on ICSI the same picture is larger: open −18.9 [−25.1, −12.4], overall
  −3.3 [−4.8, −1.7], taken tied, and at 2 s (A) −30.7 on open ends. It also beats the head alone on open ends
  (−13.0 frozen, −28.9 cross-fit) and Silero alone on taken ends (−9.7 / −22.1), so the OR does what it should.
- **The cutoff rate is acceptable at the frozen point.** 2.4 % of turns and 0.6 % of the 1991 within-turn pauses
  on ICSI (the hybrid: 1.4 / 0.2; Silero alone 3.6 / 1.1). Everything fitted on AMI is *conservative* on ICSI
  (FC 0.8–4.4 %): ICSI turns pause more (52 % of turns hesitate vs 15–17 % on AMI, research/ICSI.md) and the AMI
  thresholds sit above those pauses. The cross-fitted head OR Silero overshoots the budget instead (7.7 % held-out
  per turn, as on AMI: 6.6 %; the union of two detectors moves worse across folds than either alone, and its ICSI
  fold points differ a lot: k 58 vs 22). Ship the AMI point, not an ICSI refit.
- **Everything misses far more on ICSI than on AMI.** Frozen hybrid 82.1 % vs 61.9 % on AMI; even cross-fitted it
  is 68.5 %. The label-only any-speaker timeout is 82.3 % (AMI 68.3 %) and the oracle-primary timeout 9.8 %
  (AMI 6.2 %): ICSI ends are followed by another speaker more often (taken 84 % of turns vs 76 %), so the absolute
  numbers are not comparable across corpora, only the paired deltas are. The AMI-fitted Sortformer-primary timeout
  (k 37) is nearly useless on ICSI (91.6 %, open 77.8 %); the causal track is the weak part, as on AMI.
- **Silero remains the best open-end detector alone** (45.5 % frozen, 37.4 % cross-fit; the Sortformer any-speaker
  timeout 68.6 / 35.2 at a worse FC), and the head is what saves the taken ends. Head OR Silero is the only rule
  on both corpora that is never worse than the hybrid overall and better on open ends.

**Verdict: confirmed.** Ship **head OR Silero** in place of the head OR Sortformer-primary-timeout hybrid, at the
AMI-fitted point: fire at the earlier of (a) turn head trail6 (`runs/stage1_turn_v3_trail6.afm`, causal_dominant
binding, 160 ms chunk) **p > 0.99828**, or (b) Silero VAD v5 through Pipecat's VAD state machine (confidence 0.7,
start / stop 0.2 s, 32 ms chunks) **silence > 2.62 s** since the end of the last speech chunk (the first 80 ms
frame whose end exceeds it: 2.64 s), re-armed by speech as the hybrid is. Expected on ICSI: 81.7 % miss at 6 s
(open 62.2 %) at 2.4 % FC; on AMI dev 59.6 % (open 34.3 %) at 6.6 % held-out FC. In `audioforge/serve.py` this
is the `hybrid` policy (`Server._hybrid`, the timeout + head candidate merge) with the timeout candidate's silence
source changed from `TimeoutPolicy` (Sortformer primary column, `timeout_ms`) to an any-speaker Silero silence
(`timeout_ms` 2640, `eot_threshold` 0.99828); the head path is unchanged. Not edited here. Remaining caveats: the
per-turn FC of the union overshoots 5 % when re-fitted on either corpus (6.6 / 7.7 %), the Silero branch is
speaker-unaware (a floor-open end and a pause where nobody else speaks look the same to it: hence the taken-end
cost of +1.4), and ICSI is not speaker-disjoint from ICSI train, which no rule here was fitted on.

Commands (S = scratch; each stage one process ≤ 10 min, CPU, resumable):
```
W=$S/icsi_turn/work
TMPDIR=$S/icsi_turn .venv/bin/python scripts/bench_turn_icsi.py --stage tracks --work $W --budget 510   # x14 until "'todo': 0"
.venv/bin/python scripts/bench_turn_icsi.py --stage vad --work $W
TMPDIR=$S/icsi_turn .venv/bin/python scripts/bench_turn_icsi.py --stage scores --work $W               # x2 until 0 left
.venv/bin/python scripts/bench_turn_icsi.py --stage report --work $W --out runs/baselines_turn_icsi.json
.venv/bin/python -m pytest -q tests/test_bench_turn_icsi.py
```

### Dynamic timeout: ICSI held-out confirmation

2026-09-26. COMPLETENESS.md §2.4 picked, after seeing the AMI dev table, the **dynamic Silero timeout modulated by the
trail6 head posterior** (fire when the Silero any-speaker silence run >= clamp(T0 − a·p_head, Tmin, Tmax = T0) frames)
and **head OR dyn(Silero | head)** (that policy in the timeout's place inside the OR). Both are scored here on the same
1312 held-out ICSI turns, with the same two readings, as head-OR-Silero above. Code: `scripts/bench_turn_icsi_dyn.py`
(stages extract / report, on top of bench_turn_icsi's tracks, Silero probabilities and head scores); numbers:
`runs/baselines_turn_icsi_dyn.json`; tests: `tests/test_bench_turn_icsi_dyn.py`.

- **Frozen reading.** The policy grid of COMPLETENESS.md (T0 ∈ {9 .. 75} frames, Tmin ∈ {2, 4, 7, 10}, a ∈ {0 .. 55})
  fitted in-sample on all 974 AMI dev turns with the scorer's rule (≤ 5 % per-turn FC, lowest P50, P90, miss):
  **dyn Silero | head = (T0 75, a 55, Tmin 2)** at both horizons and both blocks (it is also the AMI fold-0 point; the
  fold-1 point is (37, 15, 2)); **head OR dyn = θ 0.998283 on the head, offset 4.957 frames on the policy track**
  (silence − required > 4.957, i.e. wait >= clamp(80 − 55·p, 7, 80) frames: 6.4 s at p = 0, 4.24 s at p = 0.5,
  2.48 s at p = 0.9, 2.0 s at p = 1; the block-A 2 s fit is θ 0.998507 / offset 2.08). The AMI per-fold points of
  both rules are re-derived and asserted equal to `runs/completeness.json`; the reference rows (Silero timeout k 24,
  hybrid, head OR Silero (θ 0.998285, k 33), head) are the frozen points of the section above and are asserted equal to
  `runs/baselines_turn_icsi.json`. Emission = the head's (Sortformer rule + 160 ms chunk) for every rule that reads
  the head.
- **Cross-fit reading.** The same grid selected on one ICSI leave-meetings-out half, applied to the other (the
  head-OR-dyn joint (θ, offset) grid with each half's own policy inside, as on AMI).
- **Skipped: Silero | head + smart-turn** and head OR dyn(Silero | head + st). Smart-turn v3.2 was never run on the
  ICSI windows (no `<work>/smartturn`), so the two rows COMPLETENESS.md flagged as needing confirmation are not
  confirmed here; on AMI their gain over the head-only variants was within one fold's fit (a_st = 0 in fold 0).
- Continuous whole-window statistics = `bench_completeness_dyn.continuous_summary` on the ICSI windows (lead 4 s +
  turn + 6 s trail; event = rising edge of the fire condition; late = a post-end event after another speaker has
  started since the end), at the frozen and at the cross-fitted 6 s points.

**Frozen AMI points on ICSI, 6 s horizon** (miss % [95 % CI]; FC per turn / per pause; P50 = dead air over the
detected floor-open ends, the P50 over all open ends is inf for every rule with open miss > 50 %; Δ = paired bootstrap
on the same resamples):

| system (all-AMI point) | 6 s all | 6 s open | 6 s taken | FC turn / pause | P50 open, detected | 2 s (A) all | 2 s (A) open | Δ vs head OR Silero 6 s: all / open / taken | Δ vs Silero timeout 6 s: all / open / taken |
|---|---|---|---|---|---|---|---|---|---|
| Silero VAD timeout (k 24) | 87.4 [85.5, 89.1] | 45.5 [38.4, 52.5] | 95.2 | 3.6 / 1.1 | 1920 ms | 92.2 | 58.5 | +5.6 [+3.5, +7.8] / −16.7 [−24.3, −9.1] / +9.7 | ref |
| hybrid trail6 (ours) | 82.1 [79.8, 84.4] | 71.5 [65.7, 77.3] | 84.2 | 1.4 / 0.2 | 3200 ms | 91.8 | 89.2 | +0.4 [−1.0, +1.8] / +9.3 [+4.1, +14.7] / −1.4 | −5.3 / +26.0 / −11.0 |
| head trail6 OR Silero timeout (θ 0.998285, k 33) | 81.7 [79.5, 83.9] | 62.2 [56.1, 68.7] | 85.5 | 2.4 / 0.6 | 2720 ms | 91.5 | 88.7 | ref | −5.6 [−7.8, −3.5] / +16.7 [+9.1, +24.3] / −9.7 [−12.0, −7.6] |
| **dyn Silero \| head** (75, a 55, Tmin 2) | 86.1 [84.2, 88.0] | **41.8 [35.3, 48.5]** | 94.5 | **2.1 / 0.9** | 2560 ms | 90.4 | **51.0** | +4.3 [+2.2, +6.7] / **−20.4 [−28.0, −13.2]** / +9.0 [+6.8, +11.2] | −1.3 [−2.4, −0.2] / **−3.8 [−8.8, +0.8]** / −0.7 [−1.6, +0.2] |
| **head OR dyn(Silero \| head)** (θ 0.998283, offset 4.96) | **79.5 [77.2, 81.7]** | **50.5 [44.5, 56.7]** | 85.2 | **1.8 / 0.5** | 2800 ms | **90.2** | 61.7 | **−2.2 [−3.2, −1.3] / −11.7 [−17.0, −7.0] / −0.4 [−1.0, +0.2]** | −7.8 [−10.0, −5.8] / +5.0 [−1.4, +11.5] / −10.1 [−12.3, −8.0] |

More paired deltas (6 s, frozen): head OR dyn vs hybrid **−2.6 [−4.0, −1.1] all / −21.0 [−26.9, −15.1] open / +1.0
[−0.3, +2.4] taken**; vs the head alone −3.4 / −24.8 / +0.8; vs dyn Silero | head −6.6 / +8.7 [+2.7, +15.0] / −9.4
(the OR pays 8.7 open points for the head's taken ends, as head-OR-Silero pays 16.7 against Silero alone). Dyn
Silero | head vs hybrid +4.0 / −29.8 / +10.4. Block A at 2 s: head OR dyn vs head OR Silero −1.3 [−3.1, +0.5] all /
**−27.0 [−34.1, −19.8]** open / +3.7 taken at FC 1.8 vs 1.5; vs Silero timeout −2.0 [−3.2, −0.8] / +3.2 [−1.7, +8.1] /
−2.9; dyn Silero | head vs Silero timeout −1.8 [−3.0, −0.7] / **−7.5 [−13.4, −2.4]** / −0.6. Block C at the 2 s
emission horizon every rule that reads the head misses 98–99 % (the head's input latency, as on AMI; Silero alone
92.8 / 62.0 open).

- **AMI fold points** (a sensitivity range, 6 s): dyn Silero | head 86.1 / open 41.8 / FC 2.1 (fold 0 = the all-AMI
  point) and 86.6 / 41.9 / 2.7 (fold 1: T0 37, a 15); head OR dyn 80.7 / **59.9** / 1.5 (fold 0: θ 0.998243, offset
  10.40, dyn 75 / 55) and **77.1 / 48.1** / 3.0 (fold 1: θ 0.998161, offset 3.76, dyn 37 / 15). Head OR Silero at its
  fold points: 82.7 / 65.5 and 77.7 / 52.4. The head-OR-dyn rule is better than head-OR-Silero on all ends and on open
  ends at every AMI point (−2.0 to −0.6 all, −5.6 to −4.3 open, fold-matched).
- **Subsets** (6 s, all-AMI point; all / open / FC): **dev** (465 turns, 85 open) dyn 84.0 / 37.0 / 1.9, head OR dyn
  81.6 / **48.8** / 1.7 vs head OR Silero 85.0 / 65.1 / 2.6 and Silero 85.7 / 43.0 / 4.1; **eval** (847 / 131) dyn
  87.2 / 44.8 / 2.1, head OR dyn 78.3 / **51.6** / 1.9 vs 79.9 / 60.3 / 2.4 and 88.3 / 47.1 / 3.3. The direction holds
  on both.

**Cross-fitted on ICSI** (6 s; held-out FC; chosen per fold):

| system | 6 s all | 6 s open | 6 s taken | FC turn / pause | chosen per fold | Δ vs head OR Silero: all / open | Δ vs Silero timeout: all / open |
|---|---|---|---|---|---|---|---|
| Silero VAD timeout | 84.4 [82.3, 86.4] | 37.4 [30.1, 44.4] | 92.7 | 8.4 / 3.7 | k 24 / 17 | +19.2 / +1.1 | ref |
| head trail6 OR Silero timeout | 65.2 [62.4, 68.0] | 36.3 [29.5, 42.9] | 70.6 | 7.7 / 2.4 | (0.99650, k 58) / (0.99774, k 22) | ref | −19.2 [−21.9, −16.5] / −1.1 [−7.5, +5.9] |
| dyn Silero \| head | 83.0 [80.8, 85.2] | **30.7 [24.2, 37.1]** | 92.5 | 6.6 / 3.2 | (75, a 55) / (30, a 15), Tmin 2 | +17.7 / −5.6 [−12.2, +0.3] | **−1.5 [−2.7, −0.4] / −6.7 [−11.5, −2.2]** |
| head OR dyn(Silero \| head) | 62.0 [59.1, 64.7] | 26.8 [19.9, 33.5] | 67.7 | **14.8 / 8.2** | (θ 0.99706, offset +12.0) / (θ 0.99773, offset −6.2) | −3.2 [−5.4, −1.3] / −9.6 [−16.0, −3.3] | −22.5 / −10.7 |

Block A at 2 s, cross-fitted: dyn Silero | head 87.6 / open 36.7 / FC 6.6 vs Silero 89.7 / 46.9 / 8.4 (−2.1 [−3.2,
−1.1] / −10.2 [−15.2, −5.3]); head OR dyn 72.6 / 34.3 at FC 11.8 vs head OR Silero 76.5 / 49.5 at 7.7.

**Continuous run over the whole ICSI windows** (6 s points; per window):

| system | point | events / window | lead events / window | windows with a cutoff | windows with a post-end event | duplicate post-end events / window (windows with one) | first post-end event is late, all / open / taken |
|---|---|---|---|---|---|---|---|
| Silero timeout | frozen | 0.19 | 0.030 | 2.4 % | 13.4 % | 0.002 (0.1 %) | 5.3 % / 8.8 % / 4.6 % |
| head OR Silero | frozen | 0.28 | 0.009 | 2.0 % | 19.6 % | 0.050 (4.0 %) | 15.3 % / 14.3 % / 15.5 % |
| hybrid | frozen | 0.28 | 0.000 | 1.4 % | 19.4 % | 0.069 (5.6 %) | 16.4 % / 13.9 % / 16.9 % |
| dyn Silero \| head | frozen | 0.18 | 0.007 | 1.8 % | 14.6 % | 0.004 (0.4 %) | 6.4 % / 11.1 % / 5.5 % |
| head OR dyn(Silero \| head) | frozen | 0.29 | 0.003 | 1.8 % | 22.0 % | 0.050 (4.0 %) | 15.6 % / 13.9 % / 16.0 % |
| Silero timeout | cross-fit | 0.32 | 0.064 | 6.5 % | 17.7 % | 0.007 (0.7 %) | 7.5 % / 7.4 % / 7.5 % |
| head OR Silero | cross-fit | 0.61 | 0.027 | 6.9 % | 38.2 % | 0.122 (9.8 %) | 29.1 % / 18.1 % / 31.3 % |
| dyn Silero \| head | cross-fit | 0.30 | 0.027 | 5.7 % | 19.1 % | 0.014 (1.4 %) | 8.2 % / 8.3 % / 8.1 % |
| head OR dyn(Silero \| head) | cross-fit | 0.84 | 0.041 | 13.7 % | 44.9 % | 0.161 (12.7 %) | 31.0 % / 5.1 % / 36.1 % |

**Reading.**

- **The open-end gain of head OR dyn(Silero | head) over head OR Silero is confirmed, and it is larger on ICSI than
  on AMI.** At the frozen AMI point it misses 50.5 % of floor-open ends vs 62.2 % (Δ **−11.7 [−17.0, −7.0]**) and
  79.5 vs 81.7 % of all ends (**−2.2 [−3.2, −1.3]**), taken ends tied (−0.4 [−1.0, +0.2]), at a *lower* cutoff rate
  (1.8 vs 2.4 % of turns, 0.5 vs 0.6 % of the 1991 within-turn pauses), on both the dev (−3.4 all / −16.3 open) and the
  eval meetings (−1.6 / −8.7) and at both AMI fold points. On AMI dev the same comparison was −1.3 all / −2.8 [−6.1,
  +0.5] open (not significant); here the CI excludes 0 on both. Against our hybrid it is −2.6 all / −21.0 open, and
  it is the best all-ends row on ICSI at any frozen point (79.5 %; the next is head OR Silero at 81.7).
- **The dynamic timeout alone beats the fixed Silero timeout on open ends in direction, at a lower FC, but the frozen
  CI touches 0**: 41.8 vs 45.5 % (−3.8 [−8.8, +0.8]) at 6 s, 51.0 vs 58.5 % (−7.5 [−13.4, −2.4]) at 2 s (block A), FC
  2.1 vs 3.6 % per turn; cross-fitted on ICSI it is −6.7 [−11.5, −2.2] at 6 s and −10.2 at 2 s. As on AMI it loses
  the taken ends the OR wins through the head (94.5 % missed), so it is +4.3 points worse than head OR Silero overall
  and is not a shipping candidate by itself; its value is inside the OR.
- **What the rule does.** The all-AMI fit sits at the corner of the grid (T0 = 75 and a = 55 are the grid maxima, Tmin
  never binds): the branch waits 6.4 s of Silero silence when the head is at 0, 4.2 s at 0.5, 2.5 s at 0.9 and
  2.0 s at 1, so within the 6 s horizon it only ever fires when the head posterior is above ≈ 0.1. It is the head's
  posterior used as a confidence on an any-speaker silence, not as a trigger: the dyn branch alone at the shipped
  offset detects 87 open ends against the head's 47 at θ, and 34 taken ends against the head's 142. The continuous
  run shows the price: head OR dyn fires 0.29 events per window (head OR Silero 0.28, the hybrid 0.28, Silero alone
  0.19), with the same 4 % of windows re-firing after a detection and the same 16 % of first post-end events landing
  after the next speaker has started (all ORs: those are the head's taken-end firings); its fires in the lead
  (another speaker's silence before the primary starts) are the lowest of the Silero-based rules (0.003 per window vs
  0.009 head OR Silero, 0.030 Silero timeout), because a low head posterior lengthens the wait there.
- **Dead air.** Over the detected floor-open ends the P50 is 2.80 s (head OR dyn) vs 2.72 s (head OR Silero), 2.56 s
  (dyn alone), 1.92 s (Silero timeout) and 3.20 s (hybrid); P90 3.6 / 3.9 / 3.0 / 2.2 / 5.4 s. The extra recall is
  bought on high-confidence ends (waits of 2.0–2.5 s) and paid on low-confidence ones (up to 6.4 s), the same trade
  as on AMI.
- **The ICSI cross-fit is not a shipping candidate.** Re-fitted on ICSI, head OR dyn reaches 62.0 % all / 26.8 % open
  but at **14.8 % held-out per-turn FC** (8.2 % per pause, 13.7 % of windows with a cutoff in the continuous run):
  the joint (θ, offset) grid picks a negative offset on one fold (fire before the policy's own wait) and a low θ on the
  other, i.e. a two-branch rule whose second branch already depends on the head is even less stable across meeting
  folds than head OR Silero (7.7 % there). The dyn policy alone cross-fits at 6.6 % FC (Silero 8.4 %) with the same
  fold-0 point as AMI (75, a 55) and (30, a 15) on fold 1. Everything fitted on AMI is again conservative on ICSI
  (FC 1.8–2.1 %), for the reason given above (ICSI turns pause more).
- **Caveats.** Smart-turn variants unconfirmed (skipped, above). The AMI fit is at the grid's T0 / a maxima, so a
  wider grid might move the point; it was not extended here because the frozen point is what was tested. The head-OR-
  dyn point was chosen on AMI after seeing the AMI table (COMPLETENESS.md), which is exactly why this held-out run
  exists; ICSI is one corpus, not speaker-disjoint from ICSI train (no rule here was fitted on it).

**Verdict: confirmed.** The open-end gain holds at an acceptable (lower) cutoff rate. Ship **head OR dyn(Silero | head)**
in place of head OR Silero, at the AMI-fitted point: fire at the earlier of (a) turn head trail6 (causal_dominant
binding, 160 ms chunk) **p > 0.998283**, or (b) Silero VAD v5 through Pipecat's VAD state machine (confidence 0.7,
start / stop 0.2 s, 32 ms chunks) any-speaker silence, in 80 ms frames since the end of the last speech chunk, with
**silence − clamp(75 − 55·p, 2, 75) > 4.957 frames, i.e. silence >= clamp(80 − 55·p, 7, 80) frames, rounded up (T0 80, a 55,
Tmin 7 in the wait; p = the same head posterior at that frame)**: 6.4 s at p = 0, 4.24 s at p = 0.5, 2.48 s at
p = 0.9, 2.0 s at p = 1; re-armed by speech as the timeout is. Expected on ICSI: 79.5 % miss at 6 s (open 50.5 %) at
1.8 % FC; on AMI dev 58.2 % (open 31.5 %) at 6.1 % held-out FC (COMPLETENESS.md §2.4, cross-fitted; the in-sample
all-AMI point is what ships). In `audioforge/serve.py` this is the head-OR-Silero change described above with the
Silero candidate's fixed `timeout_ms` replaced by the per-frame wait `80 ms x clamp(80 − 55 p, 7, 80)` read from the
head's latest posterior; not edited here.

Commands (S = scratch; after bench_turn_icsi's tracks / vad / scores stages; CPU, < 1 min each):
```
W=$S/icsi_turn/work
.venv/bin/python scripts/bench_turn_icsi_dyn.py --stage extract --work $W
.venv/bin/python scripts/bench_turn_icsi_dyn.py --stage report --work $W --out runs/baselines_turn_icsi_dyn.json
.venv/bin/python -m pytest -q tests/test_bench_turn_icsi_dyn.py
```

### Planned additions: VAP and TurnBench (not run)

These close comparison gaps in the existing evaluation; neither is evidence for a new turn-head contribution.

| addition | role | integration and comparison contract |
|---|---|---|
| **Voice Activity Projection (VAP)** | pretrained turn-taking model baseline | Pin an upstream checkpoint and code revision. Convert its future-activity output to a primary-speaker turn score using a mapping fixed on development data. Feed the same eot-bench v2 scorer, meeting folds, 2 s / 6 s emission horizons, and floor-open/taken strata. Report actual held-out FC as well as the calibration target; do not describe overshooting rows as exactly matched at 5 %. |
| **TurnBench** | external dataset and evaluation protocol, not another model | Run our head/hybrid and selected baselines under the official end-of-turn/interruption definitions and scorer. Keep its results separate from AMI eot-bench v2: its labels, input channels, event matching, and false-positive denominator are not automatically equivalent. Use its available development split for calibration and official held-out evaluation where available. |

**Input parity matters for VAP.** The upstream stereo model expects separate speaker channels; its documented
mono fallback adds a silent second channel and assumes the first contains one speaker. A mixed four-speaker AMI
recording is not that input. Use an explicitly documented mono-capable configuration for a like-input comparison,
or report a separate channel-assisted dyadic comparison. If constructing channels from estimated diarization,
label that adaptation and include its errors and delay; multiplying a mixture by a speaker activity mask does not
separate simultaneous voices. Oracle-separated channels are a privileged-input control. Align native VAP scores
to our 80 ms grid using only scores available by the decision time; do not interpolate from future samples.

VAP code/checkpoint source: [VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection);
objective: [VAP paper](https://arxiv.org/abs/2205.09812). External protocol and data links:
[TurnBench](https://turnbench.sesame.com/). Sources checked September 26, 2026. No checkpoint was installed or
evaluated for these additions. The existing `future_act_aux` is related supervision, not a substitute for an
independently trained VAP baseline. A learned additional-wait policy remains an incremental ablation above the
existing cross-fitted threshold/hybrid curves.

### Commands

```
W=<scratch>/baselines_turn   # models: silero_vad_v5.onnx (github snakers4/silero-vad v5.1.2), smart-turn-v3/smart-turn-v3.2-cpu.onnx
uv pip install --python .venv/bin/python livekit-plugins-turn-detector==1.8.3     # the only install
.venv/bin/python -c "from audioforge.nemo_import import resolve; resolve('nvidia/parakeet_realtime_eou_120m-v1')"
for s in vad smartturn lkaudio eou asr lktext; do
  .venv/bin/python scripts/bench_turn_baselines.py --stage $s --work $W/work   # each < 10 min, resumable
done
.venv/bin/python scripts/bench_turn_baselines.py --stage report --work $W/work --out runs/baselines_turn.json
.venv/bin/python -m pytest -q tests/test_baselines_turn.py
```

## VAD

2026-09-26. Same data and label as `scripts/eval_stage1.py --tasks diar` (STAGE1.md §3). The data is the 64 × 20 s AMI dev diar
windows (4 meetings, 1280 s, 76.9 % speech frames). The label is "any speaker active" (`train.derive_labels`) on the 80 ms grid, and
AMI words are rasterised with `int(s/.08) .. ceil(e/.08)-1`. Finer outputs are **max-pooled** onto that grid (`sd.pool_probs`: a label
frame takes the max over every model frame that overlaps it), which is the rule the labels are built with. Metrics are pooled
over all 16 064 frames. FPR is the false-alarm rate on non-speech frames, and miss = 1 − recall. CPU is the wall time for the 64 windows
on 2 threads (RTF = sec / 1280 s). Other agents shared the machine during the runs, so read CPU as an order of magnitude.

| system | what it is | params | license | acc / recall / prec / F1 @0.5 | FPR | latency (frame + lookahead) | CPU RTF |
|---|---|---|---|---|---|---|---|
| **our VAD head** (stage1_heads_pretrained.afm, [70,1]) | frame head on the shared FastConformer | 33 K head (encoder 111 M shared with ASR) | ours | **0.921 / 0.951 / 0.946 / 0.949** | 0.179 | 80 ms + 80 ms (160 ms chunk) | 0.0061 for the whole model pass (free if the encoder runs anyway) |
| Silero VAD v5.1.2 (jit) | LSTM on STFT, 32 ms chunks | 0.24 M (16 kHz branch) | MIT | 0.876 / 0.862 / 0.975 / 0.915 | 0.075 | 32 ms + 0 | 0.0023 |
| Silero VAD, model shipped in pip 6.2.3 | same family | 0.24 M | MIT | 0.882 / 0.869 / 0.974 / 0.919 | 0.076 | 32 ms + 0 | 0.0021 |
| WebRTC VAD mode 0 / 1 / 2 / 3 | GMM, 30 ms frames | — | BSD | 0.877 / 0.877 / 0.851 / 0.656 acc; F1 0.919 / 0.919 / 0.898 / 0.725 | 0.25 / 0.23 / 0.16 / 0.12 | 30 ms + 0 | 0.0001 |
| NVIDIA Frame-VAD MarbleNet v2.0 (ported, `sd.FrameVAD`) | 1-D separable conv, 20 ms frames | 91 378 | NVIDIA Open Model License | 0.903 / 0.935 / 0.939 / 0.937 | 0.204 | 20 ms + ≈1.46 s (symmetric conv receptive field; NeMo streams it with buffered windows) | 0.0003 |
| pyannote segmentation-3.0 (any speaker of the diarization pipelines, 3.1 = community-1) | PyanNet, 10 s windows | 1.5 M | MIT | **0.947 / 0.981 / 0.951 / 0.966** | 0.169 | offline (10 s windows + whole-file aggregation) | ≈0.35 (whole pipeline) |
| NVIDIA Sortformer v2, any column (offline) | diarizer | 118 M | CC-BY-4.0 | 0.901 / 0.898 / 0.972 / 0.933 | 0.086 | offline 20 s (streaming 1.04 s) | 0.0087 |
| always speech | — | — | — | 0.769 / 1 / 0.769 / 0.869 | 1 | — | — |

**DET (miss / FPR at thresholds 0.1 → 0.9)**, pooled 80 ms frames:

| thr | ours | Silero v5.1.2 | MarbleNet v2.0 |
|---|---|---|---|
| 0.1 | 0.005 / 0.484 | 0.084 / 0.124 | 0.022 / 0.443 |
| 0.3 | 0.025 / 0.277 | 0.117 / 0.089 | 0.049 / 0.292 |
| 0.5 | 0.049 / 0.179 | 0.138 / 0.075 | 0.065 / 0.204 |
| 0.7 | 0.090 / 0.108 | 0.164 / 0.058 | 0.085 / 0.135 |
| 0.8 | 0.121 / 0.074 | 0.185 / 0.052 | 0.099 / 0.103 |
| 0.9 | 0.173 / 0.040 | 0.219 / 0.042 | 0.123 / 0.076 |

Best acc over thresholds: ours 0.922 (0.4), MarbleNet 0.904 (0.7), Silero v5 0.907 (0.1). Best F1: ours 0.950, MarbleNet 0.937, Silero 0.938.
The full sweeps are in `runs/baselines_sd.json` → `vad.*.sweep` and `ours.vad_sweep`.

**Reading.**
- Our head is **better than every dedicated streaming VAD** on this label, at every operating point we can match.
  - At FPR ≈ 0.075 it misses 12.1 % of speech. Silero misses 13.8 %, and MarbleNet 12.3 % with 1.46 s of lookahead.
  - At FPR ≈ 0.18 it misses 4.9 %. MarbleNet misses about 7 % there, and Silero cannot reach that FPR (0.12 at threshold 0.1).
  - WebRTC is clearly worse.
- **Caveat: this is a label-convention advantage.** Our head was trained on AMI train with exactly this word-level, any-overlap label.
  The dedicated VADs detect acoustic speech, so laughter, breaths and cross-talk score as "false alarms". Silero's low recall is the other side of the same mismatch:
  it is conservative on short, soft AMI words.
- The only system that beats our head is **pyannote's segmentation model** (acc 0.947, F1 0.966). It is offline, with 10 s windows, and ~50× the CPU.
  The product has no reason to switch for real-time use. If an offline pass exists anyway (meeting post-processing), its speech/non-speech is the best here.
- **Product:** keep our VAD head. It costs nothing on top of the ASR encoder and has 160 ms latency. Silero is the right fallback only where the FastConformer
  does not run (a pre-gate on the device), and it should run at threshold ≈ 0.1–0.2 on meeting audio.

**MarbleNet port.** `sd.load_nemo_conv("nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0")` loads **84 NeMo tensors as 82, `strict=True`**, with
91 378 params (the card says 91.5K). The two preprocessor buffers are recomputed by our `LogMel` with `normalize=None` (NeMo's `normalize: "None"`
is a no-op). Checks: mel fb max |Δ| 1.9e-9, window max |Δ| 6e-8. The module names mirror NeMo's `ConvASREncoder`/`JasperBlock`
(masked separable convs, BN eps 1e-3, residual 1×1 conv + BN), so the mapping is the identity. There is no NeMo in the venv to compare
logits against. The check is behavioural: on a TitaNet card clip (an255-fash-b.wav) the probabilities go from 0.01 on silence to 1.00 on speech, and F1 is 0.937 on AMI.

## Diarization

Same 64 windows and **the same pooled frame DER code** as STAGE1 §2 and SORTFORMER_IMPORT: `eval_stage1.der_parts` summed over windows,
with 80 ms frames, threshold 0.5, no collar, overlap scored, and a word-level 4-column reference. pyannote's segments are rasterised with the label rule
(`sd.segments_to_matrix`), keeping the 4 most active speakers, columns in arrival order. No window had more than 4. The pyannote.metrics
`DiarizationErrorRate` is computed against the same frame reference turned into segments, pooled over windows, with uem [0, 20.08 s).
**pyannote's `collar` is the total width**, so `collar=0.5` is the NIST "±0.25 s" convention.

| system | what it is | params | license | frame DER (miss / FA / conf) | pyannote.metrics DER: collar 0 / 0.25 (±0.125 s) / 0.5 (±0.25 s) | latency | CPU RTF |
|---|---|---|---|---|---|---|---|
| our diar head (Sortformer head on the frozen ASR encoder) | ours | 1.9 M head + shared encoder | ours | 0.394 (.246 / .059 / .089) | 0.394 / 0.346 / 0.312 | 160 ms chunk (offline pass here) | 0.0061 (whole model) |
| **NVIDIA Sortformer v2** (ported; product) | FastConformer (NEST) + Sortformer | 118 M | CC-BY-4.0 | **0.201** (.140 / .045 / .016) | **0.201** / 0.167 / 0.159 | offline 20 s; streaming 1.04 s gives 0.252 | 0.0087 |
| NVIDIA Nemotron-3-Diarization (SORTFORMER_IMPORT.md) | 31 RoPE blocks, 8 spk | 99 M | OpenMDW-1.1 | 0.255 (.222 / .021 / .013) | — | offline; streaming 0.263 | ~2× faster than v2 |
| pyannote speaker-diarization-3.1 | segmentation-3.0 + WeSpeaker ResNet34 + AHC | 1.5 M + 6.6 M | MIT (gated) | 0.220 (.070 / .063 / .087) | 0.219 / 0.167 / **0.132** (continuous segments) | offline (10 s windows, clustering over the file) | ≈0.35 |
| pyannote speaker-diarization-community-1 | segmentation-3.0 + WeSpeaker + VBx/PLDA | 1.5 M + 6.6 M | CC-BY-4.0 (gated) | 0.264 (.070 / .060 / .133) | 0.264 / 0.214 / 0.181 | offline | ≈0.40 |
| one speaker = oracle VAD (STAGE1) | trivial | — | — | 0.312 (.144 / 0 / .168) | — | — | — |

- **Mapping frame DER ↔ standard DER.** With collar 0, pyannote.metrics on the rasterised hypotheses reproduces our pooled frame DER **exactly**
  (0.2014 = 0.2014, 0.3937 = 0.3937, 0.2204 = 0.2204; this is also a unit test). pyannote's own continuous segments differ by ≤ 0.001.
  The standard ±0.25 s collar lowers the DER by about 0.04 for Sortformer (0.20 → 0.16), 0.08 for our head, and 0.09 for pyannote 3.1.
- pyannote 3.1 and community-1 found 1.83 and 1.64 speakers per window, against 2.62 in the reference. They **under-cluster** on 20 s windows,
  so their error is confusion (0.09 and 0.13). Sortformer's error is miss (0.14), mostly short words and backchannels, as SORTFORMER_IMPORT describes.
  At collar 0, Sortformer v2 is the best system here. With the ±0.25 s collar, pyannote 3.1 edges ahead (0.132 vs 0.159), because its misses sit at
  boundaries and Sortformer's do not. On full meetings pyannote would cluster over more speech; these are 20 s windows only.
- **Reading:** our diar head is **clearly worse than every dedicated diarizer** (0.39 vs 0.20–0.26). It is also worse than the speaker-blind oracle-VAD
  baseline. **The product should keep NVIDIA Sortformer v2 for diarization.** It is the best at collar 0, streams, runs 40× cheaper than pyannote, and is CC-BY.
  pyannote is an offline alternative for meeting post-processing when a ±0.25 s collar is the metric. It is not usable for real-time turn-taking.
- Could not run: nothing blocked. Both pyannote pipelines are gated, but this machine's HF token (`notmax123`) already had access. pyannote.audio
  4.0.7 ran both the 3.1 and community-1 pipelines on in-memory waveforms. A process takes ~8–10 s per 20 s window on 2 threads, so the stage is resumable
  (per-window cache, `--budget`) and each process stayed under 10 min.

## Speaker verification

Same trials as `eval_stage1 --tasks spk` (STAGE1 §4): AMI dev asr-mode single-speaker segments, cosine scores, `metrics.eer`, all pairs
and within-meeting pairs (the honest subset, since room and channel give cross-meeting pairs away). n = 64 is 15 speakers, 2016 trials
(152 target, 487 within-meeting), mean 6.2 s. n = 200 is 15 speakers, 19 900 trials (1752 target, 5101 within), mean 5.9 s.

| system | what it is | params | license | EER n=64 all / within | EER n=200 all / within | mean cos same / diff (n=64) | CPU RTF |
|---|---|---|---|---|---|---|---|
| our speaker head (AAM, 190 train speakers) | attentive pooling on the shared encoder | 0.5 M head | ours | 15.0 % / 32.2 % | 16.3 % / 29.2 % | — | shared pass |
| **NVIDIA TitaNet-Large** (ported, `nemo_import.import_titanet`) | SE separable 1-D conv + attentive stats pooling, 192-d | 22.1 M (+3.2 M AAM classifier) | CC-BY-4.0 | **6.6 % / 8.2 %** | 10.9 % / 12.0 % | 0.59 / 0.08 | 0.017 |
| SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) | ECAPA, 192-d | 20.8 M | Apache-2.0 | 9.9 % / 9.6 % | 12.6 % / 13.0 % | 0.54 / 0.06 | 0.007 |
| WeSpeaker ResNet34-LM (pyannote/wespeaker-voxceleb-resnet34-LM) | ResNet34, 256-d (pyannote's embedder) | 6.6 M | CC-BY-4.0 | 9.9 % / 9.6 % | **10.6 % / 11.9 %** | 0.58 / 0.08 | 0.010 |
| untrained mean-pooled layer mix (STAGE1) | baseline | — | — | 19.9 % / 33.5 % | 22.6 % / 36.8 % | — | — |

- **TitaNet port.** `import_titanet()` downloads the 102 MB `.nemo` and loads **124 NeMo tensors as 122, `strict=True`**, 25.3 M params. The two preprocessor
  buffers are recomputed and checked (fb 1.9e-9, window 6e-8). The layout mirrors NeMo: `ConvASREncoder` blocks with masked depthwise + pointwise convs,
  BN eps 1e-3, global SE (reduction 8), a residual 1×1 conv + BN, ReLU. Then `SpeakerDecoder`: an attentive pool (TDNN conv → ReLU → BN, tanh, conv, masked
  softmax, weighted mean/std), then BN + 1×1 conv → 192. VoxCeleb1-O (card: 0.66 % EER) is not measurable here, so the validation is behavioural:
  - the card's example pair (an255-fash-b / cen7-fash-b, same speaker) scores cos **0.81**, above NeMo's `verify_speakers` threshold 0.7;
  - on AMI, same-speaker cos is 0.59 against 0.08 for different speakers (0.62 vs 0.13 within a meeting);
  - batched and unbatched embeddings match (max |Δ| 3e-7).
- **Reading:** our speaker head is **far worse than every dedicated embedder**. All pairs it is 2.3× TitaNet's EER. Within a meeting it is 32 % vs 8–10 %, which is
  near chance where it matters (telling apart people in the same room). The dedicated models barely degrade within a meeting. Ours degrades by 2×, so it is
  still keying on channel and room. TitaNet is best at n=64, and WeSpeaker ties it at n=200 (10.6–10.9 %), with a CI of about ±1 point at 1752 target trials.
- **Product:** for enrollment and speaker identity (`enrollment.py`, voice binding in the turn stack), **use TitaNet-Large** (CC-BY-4.0, NVIDIA, 22 M params, RTF 0.017,
  now importable without NeMo). WeSpeaker ResNet34 is the small alternative (6.6 M) at equal accuracy. Our head is not a substitute. Distilling TitaNet
  into the head is the obvious next lever, if a single-pass embedding is wanted.

### Commands (VAD, diarization, speaker verification)

```bash
S=<scratch>; export TMPDIR=$S BASELINES_SCRATCH=$S
uv pip install --python .venv/bin/python silero-vad webrtcvad-wheels pyannote.audio speechbrain   # 6.2.3, 2.0.14, 4.0.7, 1.1.1 (additions only)
.venv/bin/python scripts/bench_sd_baselines.py --stage vad          # Silero v5.1.2 + pip v6, WebRTC 0-3, MarbleNet (7 s)
.venv/bin/python scripts/bench_sd_baselines.py --stage ours         # our VAD sweep, diar head via pyannote.metrics, spk EER n=64/200 (40 s)
.venv/bin/python scripts/bench_sd_baselines.py --stage sortformer   # Sortformer v2 offline: frame DER + pyannote.metrics (15 s)
.venv/bin/python scripts/bench_sd_baselines.py --stage pyannote --pipeline pyannote/speaker-diarization-3.1 --budget 420        # re-run until done
.venv/bin/python scripts/bench_sd_baselines.py --stage pyannote --pipeline pyannote/speaker-diarization-community-1 --budget 420
.venv/bin/python scripts/bench_sd_baselines.py --stage spk          # TitaNet-L, SpeechBrain ECAPA, WeSpeaker ResNet34 (60 s)
.venv/bin/python -m pytest -q tests/test_baselines_sd.py
```
Code: `audioforge/baselines/sd.py` (grid conversions, VAD scoring, NeMo ConvASREncoder / TitaNet / FrameVAD port), `nemo_import.import_titanet`,
`scripts/bench_sd_baselines.py`. Numbers: `runs/baselines_sd.json`.

## Language ID

2026-09-27. Full write-up: `research/LID.md`. FLEURS test, 17 languages (en he ar ru es fr de pt it nl pl uk tr fa hi zh
ja), 150 utterances each (n = 2550, speaker-disjoint from train). Every system gets the same clips: the first 1 / 2 / 3 / 5 s
from the Silero speech onset, and the full utterance. Posteriors are restricted to the 17 languages. Accuracy in %.

| system | params | licence | 1 s | 2 s | 5 s | full | EdAcc accented English, full (called "en") | CPU per decision (2 threads) |
|---|---|---|---|---|---|---|---|---|
| **our LID head** (`lid_aug`, blocks 8-12, causal attentive-stats) | 0.40 M head | ours | 59.2 | 75.0 | 85.7 | 89.3 | 18.7 | 0.16 ms per 160 ms chunk (shared encoder pass) |
| **NVIDIA langid_ambernet** (ported, `nemo_import.import_ambernet`) | 28.9 M | NGC Terms of Use | **83.9** | **95.1** | **99.2** | **99.5** | 67.7 | 16-81 ms per call (1-8 s) |
| SpeechBrain lang-id-voxlingua107-ecapa | 21.2 M | Apache-2.0 | 81.1 | 94.6 | 98.5 | 99.3 | 70.0 | 10-43 ms per call |
| Whisper-tiny (language token) | 37.8 M | Apache-2.0 | 54.9 | 78.2 | 93.7 | 97.0 | 93.8 | ~63 ms per call |
| Whisper-base (faster-whisper) | 74 M | Apache-2.0 | 61.5 | 83.6 | 96.3 | 98.5 | **95.3** | ~155 ms per call |

**Reading: worse.** The head is 20 points behind AmberNet at 2 s and 10 points behind on full utterances. It fails on accented
conversational English, where Whisper is the robust system (95 %) and the VoxLingua107 models call a third of it another
language. The head is only cheaper; it is ahead of Whisper-tiny only at 1 s. Product: `serve --lid ambernet` (AmberNet re-run
on a sparse schedule, RTF +0.02-0.03, 96.8 % correct first announcement at a 2.2 s median); LID is off by default.
Commands: `research/LID.md`. Numbers: `runs/lid.json`.
