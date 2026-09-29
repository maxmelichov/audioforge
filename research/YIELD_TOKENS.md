# Turn-taking as ASR vocabulary: `<YIELD>` / `<HOLD>` action tokens in the transducer

2026-09-26. Test of an outside reviewer's claim: fold turn-taking into the ASR vocabulary as conversational action
tokens, `<YIELD>` (pause after which the floor is open or taken: a true end of turn) and `<HOLD>` (an intra-speaker
pause, a hesitation), trained with the transducer loss on the frozen NVIDIA encoder, so that the joint of acoustics and
text state predicts yield vs hold with no diarizer in the loop, and "syntax solves hesitations". Prior evidence against
it (research/BASELINES.md): NVIDIA's parakeet_realtime_eou (an `<EOU>` token in an RNNT) misses 70.1 % of turn ends at
6 s / 42.9 % on floor-open ends on eot-bench v2, vs our hybrid 61.9 / 45.9 and a Silero VAD timeout 72.7 / 26.7; the
LiveKit text model with oracle words does not improve on AMI.

- **Code.** Vocabulary extension: `audioforge/vocab.py`; special-aware SentencePiece encoding: `audioforge/tokenizer.py`;
  transcripts: `audioforge/datasets/ami.py` (`action_transcript`, `window_action_transcript`, `AMI.asr(action_tokens=)`,
  recipe key `ami: {action_tokens: ...}`); scorer: `scripts/bench_yield_tokens.py` (stages decode, wer, segments,
  report; reuses `scripts/bench_turn_baselines.py`'s cached tracks and scorer); tests: `tests/test_yield_tokens.py`.
- **Recipes / runs.** Run 1 `research/recipes/stage1_rnnt_yieldhold_run1_primary_only.yaml` (log
  `runs/stage1_rnnt_yieldhold_run1_primary_only.log`); run 2 `research/recipes/stage1_rnnt_yieldhold.yaml` ->
  `runs/stage1_rnnt_yieldhold.afm` (+ `.log`); run 1a `research/recipes/stage1_rnnt_yieldhold_1a.yaml` ->
  `runs/stage1_rnnt_yieldhold_1a.afm` (+ `.log`). Numbers: `runs/yield_tokens.json` (run 2, eot-bench v2, 1000
  bootstraps) and `runs/yield_tokens_1a.json` (run 1a).
- GPU used: run 1 5 min, run 2 41 min, run 1a 21 min, run 1b 13 min (one MPS process at a time, watermarks 0.6 / 0.5).
- **Note on scope.** Run 1a was run although run 2's gate did NOT trip (the coordinator's fallback condition), because
  run 2's surviving model never emits a token and so does not test the claim; run 1b (§4, diagnosis check) was
  trained but its scoring was stopped on the coordinator's wrap-up instruction: `runs/stage1_rnnt_yieldhold_1b.afm`
  exists, `runs/yield_tokens_1b.json` does not.

## 1. Setup

**Vocabulary.** The pretrained hybrid RNNT/CTC of `runs/stage1_heads_pretrained.afm` (NVIDIA
stt_en_fastconformer_hybrid_large_streaming_multi, 160 ms context) uses a 1024-piece SentencePiece BPE model; the
transducer blank is id 1024 (= vocab_size). SentencePiece models cannot take new ids in the middle, so the two tokens
are **appended** to the model proto as user-defined pieces: `<HOLD>` = 1024, `<YIELD>` = 1025, blank -> 1026. Every
existing piece keeps its id and its rows; the joint's output rows, the PredictionNet embedding rows and the CTC
projection rows are re-indexed (old blank/SOS row moved to 1026; the two new rows start at the mean token row with the
lowest token bias, so nothing changes at initialisation: identical transcripts, old logits bit-equal; test).
`runs/stage1_rnnt_yieldhold_init.afm` is that model. SentencePiece inserts a lone `▁` piece before an inline
user symbol ("world <HOLD>" -> `▁world ▁ <HOLD>`), so `SentencePieceTokenizer.encode` splits the text at
the specials and encodes the plain segments separately (plain text ids unchanged; test).

**Labels (the benchmark's own).** All from the AMI word timings and `datasets/ami.py`'s floor definition of a turn
(runs of one speaker with gaps < 0.5 s, merged unless the pause is >= 2 s or someone else takes the floor):
- `<HOLD>`: between two consecutive words of one turn whose silence (next start - running max end) is >= 0.3 s = the
  `hes` intervals eot-bench uses as within-turn pauses (false-cutoff sites). Placed at the pause start.
- `<YIELD>`: once at the end of a turn (its last word's end = `turn_end_frame`).

**Training (all runs).** Encoder frozen (PLAN.md §6: the WER-gate history forbids touching it). Trainable: RNNT
PredictionNet + joint and the CTC projection (5.86 M parameters), 160 ms context only, batch 6, fused joint. LibriSpeech
WER gate on the RNNT (`test-clean-first200`, first 100 utterances, [70, 1] context, baseline 2.26 %), stop and restore
if it rises > 0.5 points.

| run | targets (audio) | anchor | lr / steps / gate every | outcome |
|---|---|---|---|---|
| 1 | primary speaker's words only, on the mixed 16 s turn windows (3274; 3274 `<YIELD>`, 454 `<HOLD>`) | none | 2e-4 / 2000 / 250 | **gate tripped at step 250: 2.26 -> 85.1 % WER.** The decoder learnt to delete speech (the other speakers' words have no transcript). Not a test of the hypothesis. |
| 2 | **speaker-unaware**: all speakers' words of the window in start-time order (overlap interleaved word by word), `<HOLD>` / `<YIELD>` for every speaker at their times (10,413 `<YIELD>`, 643 `<HOLD>`; U p50 / p90 / max 48 / 83 / 135 tokens) | LibriSpeech train-clean-100 1:1 (3300 utterances, single-source batches) | 5e-5 / 2000 / 50 | gate held on all 40 checks (max 2.47 %, final 2.30 %). **Never emits either token** (§2). |
| 1a (run although 2's gate held) | single-speaker AMI train segments (`AMI.asr`: no other speaker within 0.2 s, <= 15 s; 930 segments): `<HOLD>` at >= 0.3 s pauses inside, and an end token from the label rule: `<YIELD>` if the last word ends the speaker's floor turn (410), `<HOLD>` if the run ends but the turn continues (317), none if cut by max_sec (203). The audio stops 0.1 s after the last word: the end token has to come from the words / prosody, not the following silence | LibriSpeech 1:1 (2800) | 1e-4 / 1200 / 100 | gate held (max 2.43 %); **never emits either token**; yield-vs-hold AUC at segment ends 0.445 (§4) |
| 1b (diagnosis check, not scored) | as 1a | as 1a | 1e-4 / 800 / 100, new rows at 3e-3 (`trainer.row_lr`) | gate held (2.30-2.39 %); AMI-batch loss 1.2-1.4 vs 1a's 0.43 (the fast rows do change the fit); scoring not run |

Run 2's transcript rule (`ami.window_action_transcript`): words with start < window end and end > window start,
sorted by (start, end, speaker); `<HOLD>` only when both words are inside the window; `<YIELD>` only when the turn's
end is inside the window; tokens sort before words at equal times. Example: `okay what's the agenda <YIELD> for this
meeting <YIELD> the i will uh <HOLD> present here agenda with with with with slides okay <HOLD> <YIELD> to you um <HOLD>`.

**Scoring** (`bench_yield_tokens.py`, identical to how the EOU baseline was scored in BASELINES.md): the 974 AMI dev
turn windows, raw audio + 0.32 s of the following meeting audio, one pass with the [70, 1] chunked-limited mask (=
160 ms cache-aware streaming, checked against `StreamingSession` in tests/test_baselines_turn.py), greedy RNNT decoding
frame by frame; the per-frame score is the max over the frame's symbol steps of log P(`<YIELD>`) at the joint
(`yield_posterior`), or log P(`<YIELD>`) - log P(`<HOLD>`) (`yield_vs_hold`), or "`<YIELD>` emitted" (native rule);
emission rule = 160 ms chunks. Cross-fitted fixed threshold at <= 5 % false cutoffs per turn (leave-meetings-out
halves), 6 s horizon = block C, 2 s = block A, strata floor-open 236 / taken 738, 1000 bootstraps, paired bootstraps.
No diarizer anywhere in these rows.

## 2. Run 2 on eot-bench v2 (n = 974; miss % at the cross-fitted <= 5 % per-turn FC point, [95 % CI])

| system | 6 s all | 6 s open | 6 s taken | held-out FC turn / pause | 6 s P50 open | 2 s (A) all | 2 s open | 2 s taken |
|---|---|---|---|---|---|---|---|---|
| hybrid trail6 (ours, ref) | 61.9 [58.7, 65.0] | 45.9 [39.3, 52.4] | 67.0 | 6.5 / 5.3 | 4880 ms | 79.1 [76.6, 81.9] | 72.9 | 81.1 |
| timeout, Sortformer primary (ours) | 74.8 [72.0, 77.5] | 46.1 [39.6, 52.6] | 83.6 | 5.7 / 3.9 | 4080 ms | 92.5 | 81.1 | 96.0 |
| Silero VAD timeout | 72.7 [69.8, 75.4] | 26.7 [21.1, 32.4] | 86.8 | 4.8 / 5.3 | 2000 ms | 83.5 [81.1, 85.8] | 39.6 | 96.9 |
| Parakeet-Realtime-EOU, P(`<EOU>`) | 70.1 [67.0, 73.0] | 42.9 [36.4, 49.5] | 78.4 | 5.3 / 8.7 | 3040 ms | 81.6 [78.8, 84.0] | 59.5 | 88.4 |
| EOU OR Silero timeout | 69.1 [65.8, 72.0] | 29.5 [23.1, 35.7] | 80.9 | 6.4 / 6.8 | 2080 ms | 82.8 | 54.8 | 91.2 |
| head trail6 OR Silero timeout (exploratory, BASELINES) | 59.6 [56.3, 62.6] | 34.3 [28.6, 40.4] | 67.3 | 6.6 / 6.3 | 2800 ms | 78.3 | 58.1 | 84.4 |
| **`<YIELD>` posterior (run 2)** | **94.4 [92.9, 95.9]** | **92.6 [89.0, 95.8]** | 94.9 | 7.0 / 0.5 | inf | 96.7 [95.5, 97.8] | 95.9 | 97.0 |
| `<YIELD>` - `<HOLD>` log-odds (run 2) | 100.0 | 100.0 | 100.0 | 3.4 / 0.5 | inf | 100.0 | 100.0 | 100.0 |
| `<YIELD>` posterior OR Silero timeout | 71.9 [68.7, 74.8] | 26.7 [20.5, 32.6] | 85.7 | 7.5 / 5.3 | 2080 ms | 85.0 [82.7, 87.3] | 49.3 | 95.9 |
| `<YIELD>` posterior OR our Sortformer timeout | 74.6 [71.8, 77.3] | 45.4 [38.8, 51.6] | 83.6 | 5.9 / 3.9 | 4000 ms | 92.2 | 80.6 | 95.9 |

Native rules (no threshold fitting), 6 s: `<YIELD>` emitted: **100 % miss, 0 emissions on all 974 windows** (FC 0);
`<YIELD>` emitted OR 3 s Silero silence: 83.9 % (open 57.5, FC 0.7 %); `<EOU>` emitted: 79.1 % (open 60.2, FC 3.0 %).
The cross-fitted `<YIELD>` thresholds are log P = -7.78 / -8.02 (P ~ 4e-4): at <= 5 % FC the posterior is a ranking
far below any decision the model would take, as for the EOU model (-62 / -89), but here even the ranking is useless.

**Paired differences, 6 s** (system - reference, miss points, [95 % CI]; open stratum second):
- `<YIELD>` posterior - hybrid: **+32.5 [+29.0, +35.8]**, open +46.8 [+39.6, +54.1]; - EOU posterior: **+24.3 [+21.0,
  +27.8]**, open +49.8 [+41.5, +57.9]; - Silero timeout: +21.7 [+18.8, +24.8], open +65.9 [+58.8, +72.9]; - Sortformer
  timeout: +19.6 [+16.6, +22.5], open +46.5.
- `<YIELD>` OR Silero timeout - Silero timeout: **-0.8 [-1.8, +0.2]**, open -0.1 [-2.6, +2.4], taken -1.1 [-2.2, -0.0]:
  the token adds nothing to the VAD timeout it is OR-ed with. - hybrid: +10.0 [+6.2, +13.4] (open -19.2 [-27.0, -11.1],
  which is the Silero timeout's known open-end advantage, BASELINES.md). - EOU posterior: +1.9 [-1.2, +4.8]; - EOU OR
  Silero: +2.8 [+0.3, +5.3]. At 2 s (A) it is worse than the Silero timeout alone: +1.5 [+0.1, +2.9], open +9.7.
- `<YIELD>` OR our timeout - our timeout: -0.2 [-0.6, +0.1], open -0.7 [-1.8, 0.0]: nothing.

**Tokens at the labelled sites (dev, greedy decoding).** Run 2 emits **no `<YIELD>` and no `<HOLD>` anywhere**: 0 of
974 turn ends within 6 s, 0 of 207 within-turn pauses, 0 in-turn. For reference the EOU model emits `<EOU>` after
197 / 974 ends (20.2 %) within 6 s, inside 8 / 207 pauses (3.9 %), in-turn in 29 turns (3.0 %); 82 % of its emissions
are post-end. The per-window maximum of log P(`<YIELD>`) on dev is -8.8 (p50) / -7.9 (p90) / -6.9 (max), of
log P(`<HOLD>`) -10.4 / -9.4 / -8.5. On 8 AMI **train** windows (with 39 `<YIELD>` in their targets) the model also
emits 0 tokens (max log P -7.7 to -9.3). The token was never learnt, not merely calibrated low.

**Why.** The RNNT loss on the AMI batches stayed at 2.0-3.3 nats / token for the whole run (3.5 at step 1), while
the LibriSpeech batches sat at 0.25-0.4: with a frozen encoder the decoder cannot align interleaved transcripts of
overlapped 4-speaker audio, so the lattice probability of the whole AMI target is tiny and the two rare tokens (6 %
and 0.4 % of AMI target tokens, 3 % and 0.2 % of all tokens) get no usable gradient. A low learning rate and the 1:1
LibriSpeech anchor were what kept the gate (2.26 -> 2.30 %); run 1 shows that letting the decoder move faster on this
data destroys the ASR before it learns the tokens. The gate-preserving regime and the token-learning regime do not
overlap for this target design.

**WER (before -> after run 2).** LibriSpeech gate subset (n = 100): 2.26 -> 2.30 %. AMI dev single-speaker segments
(n = 200, `AMI.asr`): 24.4 -> **22.8 %** (the anchor plus AMI text made the decoder slightly better on meeting speech).
AMI dev turn windows, all 974, hypothesis = every token decoded on the window vs the primary speaker's words: 104.6 ->
107.6 % (insertion-dominated for both models: they transcribe every speaker, the reference is one).

## 3. Emission timing at the turn end (block C; lag = emission time - last word's end from the word timings)

Firings considered: the first one at frame >= turn_end - 2 (the last 160 ms of the last word or later); earlier
firings are the scorer's in-turn false cutoffs and are counted separately. Greedy = the token actually emitted
(emission time = end of its 160 ms chunk); threshold = first crossing of the cross-fitted 6 s threshold on the scorer's
emission grid. Histogram bins are 160 ms from -0.32 s.

| detector | turns fired at the end (of 974) | in-turn firings (turns) | lag P10 / P50 / P90 | mean | share <= 160 ms after the word end | share before the word end | median in-turn pause on those turns (n pauses) |
|---|---|---|---|---|---|---|---|
| `<YIELD>` greedy (run 2) | 0 | 0 | - | - | - | - | - |
| `<YIELD>` posterior >= theta (6 s cross-fit) | 76 | 63 | 0.06 / 1.16 / 4.22 s | 1.76 s | 14.5 % | 5.3 % | 1.28 s (12) |
| `<EOU>` greedy (Parakeet-EOU) | 197 | 29 | 0.69 / 2.16 / 4.64 s | 2.40 s | **0.0 %** | 0.0 % | 0.96 s (36) |
| `<EOU>` posterior >= theta (6 s cross-fit) | 287 | 51 | 0.56 / 1.78 / 4.02 s | 2.03 s | 1.1 % | 0.7 % | 1.04 s (57) |

Histogram of the `<EOU>` greedy lag (counts per 160 ms bin from 0.16 s): 3, 7, 5, 9, 5, 11, 10, 5, 17, 6, 9, 10, 6,
11, 5, 13, 7, 5, 2, 4, 2, 4, 4, 2, 7, 1, 4, 3, 3, 3, 3, 3, 3, 1, 1, 2, 1 (nothing below 0.16 s; full arrays in
`runs/yield_tokens.json` `emission_timing`). The `<YIELD>` threshold crossings: 4, 7, 4, 6, 2, 6, 5, 3, 4, 1, 2, ...
from -0.16 s, i.e. a flat spread over 0-5 s with 63 of 139 crossings inside the turn.

**Reading.** NVIDIA's `<EOU>`, the one action token that does get emitted, is never emitted from the phonemes: 0 of
197 greedy emissions fall within 160 ms of the last word's end; the median lag is 2.2 s and the spread (P10-P90
0.7-4.6 s) is the spread of a silence detector, not of a word-end detector. The token has learnt to defer to
silence, which is also why its fitted operating point behaves like a timeout (BASELINES.md: EOU + Silero ~ Silero).
Run 2's `<YIELD>` never fires; its sub-threshold posterior crossings are spread over 0-5 s and fire inside the turn
almost as often as after it (63 vs 76), i.e. noise.

## 4. Run 1a: single-speaker segments (the clean-audio fallback)

**Training.** 930 single-speaker AMI train segments (x3) + 2800 LibriSpeech utterances, lr 1e-4, 1200 steps, 12
gates all held (max 2.43 %, final 2.26 % = baseline). AMI-batch RNNT loss 1.7 -> 0.43 nats / token, LibriSpeech
batches 0.25-0.4. WER after: LibriSpeech 2.26 -> 2.26 %, AMI dev segments 24.4 -> 23.4 %, dev windows 104.6 -> 103.2 %.

**End-token test on the 354 clean dev segments** (`--stage segments`; 155 `<YIELD>` / 140 `<HOLD>` labelled ends,
59 cut by max_sec): greedy decoding emits **no end token on any segment** (0 / 295; the word transcripts are fine,
21.6 % WER vs the segments' plain text). The sub-threshold contrast log P(`<YIELD>`) - log P(`<HOLD>`) over the last
5 frames separates yield from hold ends with **AUC 0.445**, i.e. chance. Max log P(`<YIELD>`) per segment: p50 -10.4,
p90 -8.7.

**eot-bench v2 (n = 974, 6 s / A 2 s).** `<YIELD>` posterior 95.8 [94.4, 97.0] all, 93.0 [89.3, 95.9] open, 96.6
taken (FC 5.9 / 1.5; thresholds log P -8.78 / -9.02); vs hybrid +33.8 [+30.6, +37.0], vs EOU +25.7 [+22.6, +28.8]
(open +50.2). `<YIELD>` OR Silero timeout: 72.7 [69.8, 75.3], open 26.4, taken 86.8: minus the Silero timeout **-0.0
[-0.5, +0.4]**. Greedy emissions on the 974 windows: 0. Threshold crossings at the 6 s point: 53 turns at the end,
52 in-turn; lag P50 1.16 s, 15 % within 160 ms of the word end. Same picture as run 2 on clean, alignable data.

**Diagnosis (why neither run 2 nor 1a can learn the token, with the ASR intact).** The residual AMI loss of run 1a
(~0.43 nats / token at U ~ 25 tokens ~ 9-10 nats per segment) is the cost of the one end token the model never
emits (log P about -9 to -10). Adam moves every parameter element by about lr per step; at the gate-safe learning
rates (5e-5 / 1e-4) the two new rows of the joint output (initialised at the mean token weights with the lowest
token bias, log P ~ -9) can move by at most 0.1-0.2 in 1000-2000 steps, so they can never grow into a token that
wins the argmax, while the existing rows only need small corrections. The gate-safe regime for the pretrained rows
and the growth regime for new rows do not overlap when they share one learning rate. Run 1b tests that directly:
same data, the two new rows at their own lr (3e-3, `trainer.row_lr`), the other rows of those tensors frozen.

**Run 1b (trained, not scored).** Same data and gate schedule as 1a, 800 steps, rows 1024-1025 of the joint output,
CTC projection and PredictionNet embedding at lr 3e-3 with the other rows of those tensors frozen. The gate held
(2.30-2.39 %). The AMI-batch RNNT loss on the log is 1.2-1.4 nats / token at steps 300-800 (1a: 0.43), so the fast rows
did change the fit, in which direction is unknown: its decode / segments scoring was stopped on the wrap-up
instruction. It is the next thing to score if the question is reopened (`scripts/bench_yield_tokens.py --ckpt
runs/stage1_rnnt_yieldhold_1b.afm --stage segments`, then decode / wer / report).

## 5. Verdict

**Does putting turn tokens in the transducer beat EOU / hybrid / Silero on open ends? No, and on this evidence it
cannot even be made to fire.** Under the repo's rules (frozen NVIDIA encoder, LibriSpeech WER gate at 0.5 points,
head-only RNNT training):
- The only design that trains at all without destroying the ASR (run 2, speaker-unaware all-speaker transcripts,
  lr 5e-5, LibriSpeech anchor) yields a model that **never emits `<YIELD>` or `<HOLD>`** on 974 dev windows or on
  its own training windows. Its `<YIELD>` posterior, used as a ranking exactly as the EOU model's was, misses **94.4 %
  [92.9, 95.9]** of turn ends at 6 s (open **92.6 %**) at <= 5 % FC per turn, vs hybrid 61.9 / 45.9, Parakeet-EOU
  70.1 / 42.9, Silero timeout 72.7 / 26.7 (paired: +32.5 [+29.0, +35.8] vs hybrid, +24.3 [+21.0, +27.8] vs EOU). OR-ed
  with the Silero timeout it is the Silero timeout (-0.8 [-1.8, +0.2]; open -0.1). Run 1a (clean single-speaker
  segments, run although 2's gate held) is the same: 0 emissions, 95.8 / 93.0, yield-vs-hold AUC 0.445 at segment
  ends. Run 1 (primary-only targets) tripped the gate at step 250 (85 % WER).
- WER stayed intact in the runs that ran to the end: LibriSpeech 2.26 -> 2.30 % (run 2) / 2.26 % (1a); AMI dev
  single-speaker segments 24.4 -> 22.8 % (run 2) / 23.4 % (1a): the AMI text and the anchor slightly helped the
  decoder on meeting speech, so the failure is not forgetting.
- The token-learning problem is structural for this setup (§4): new output rows starting at log P ~ -9 cannot grow
  under Adam at the gate-safe learning rates (0.1-0.2 total movement), and (run 2) the decoder cannot align
  interleaved overlapped transcripts on a frozen encoder (AMI loss 2-3 nats / token). Run 1b's per-row learning rate is
  the untested fix; even if it makes the token fire, the reference point for what a fired transducer token does is
  NVIDIA's own `<EOU>`, trained on far more data with an unfrozen model.

**What the emission timing says about "syntax solves hesitations".** The one transducer action token that does fire,
Parakeet-EOU's `<EOU>`, is emitted with a median lag of **2.16 s** after the last word (P10 / P90 0.69 / 4.64 s) and
**0.0 % of its 197 end emissions fall within 160 ms of the word end**; the median in-turn pause on those turns is
0.96 s. Its sub-decision posterior crossings at the 6 s operating point behave the same (median 1.78 s, 1.1 % within
160 ms). A token that decided from the phonemes and the text state would fire at the word end; this one fires after a
silence about as long as a hesitation, i.e. the transducer learnt to defer to silence, which is exactly why EOU + a
VAD timeout equals the timeout and why it ties or loses to the hybrid on open ends. Run 2's `<YIELD>` posterior
crossings (76 at the end vs 63 inside the turn, median lag 1.16 s, 14.5 % within 160 ms, 5.3 % before the word end)
are noise around the same silence-driven pattern, not a syntactic end-of-turn signal. On the clean segments, where the
audio stops 0.1 s after the last word and only the words can tell a yield from a hold, the yield-vs-hold contrast is
at chance (AUC 0.445). Nothing here supports the claim that the joint's text state resolves hesitations; the measured
behaviour of action tokens in a transducer is that of a silence detector with a language-model prior on when the
silence starts.

**Standing.** The prior verdict holds: on AMI at <= 5 % false cutoffs, speaker-unaware tokens (EOU, and now
`<YIELD>` / `<HOLD>`) do not beat a VAD timeout on floor-open ends and lose to the speaker-aware hybrid on taken ends
(BASELINES.md). If the idea is pursued, it needs (i) per-row learning rates or a proper initialisation for new tokens
(run 1b), (ii) targets a frozen-encoder decoder can align (single-speaker or separated audio), and (iii) far more
labelled turn ends than AMI's 3352 train turns / 765 hesitations.
