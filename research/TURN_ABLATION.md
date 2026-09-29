# Speaker-aware end-of-turn: 4-way ablation and diagnosis (2026-09-26)

Recipe `research/recipes/speaker_aware_turn.yaml` (2.8M params, causal FastConformer, `att_context_size [16,1]`,
1000 steps, batch 16). Runs: `runs/turn_{acoustic,speaker,text,speaker_text}.{afm,log}`. Benchmark:
`python -m audioforge.conversation ... --n 1000` gives `runs/turn_ablation.json`. Conversations come from
`conversation_dataset(seed=7)`. The metric is eot-bench: sweep the decision threshold and report dead-air
latency at ≤ 5 % false cutoffs (FC).

## Results (1000 conversations, operating point ≤ 5 % FC)

| detector | primary activity | P50 | P90 | miss (never fires) |
|---|---|---|---|---|
| silence timeout | oracle primary | 2560 ms | 2560 ms | 0 % |
| silence timeout | oracle any-speaker VAD | 1840 ms | inf | 10.9 % |
| turn head, acoustic | none | 1440 ms | 2080 ms | 0.7 % |
| turn head, +speaker | **oracle** | **400 ms** | 1760 ms | 4.3 % |
| turn head, +speaker | own diarization | inf | inf | 93.9 % |
| turn head, +text | none | 1520 ms | inf | 23.8 % |
| turn head, +speaker+text | oracle | 1200 ms | inf | 29.3 % |
| turn head, +speaker+text | own diarization | inf | inf | 94.1 % |

Two findings need explaining. (A) The diarization path does not work: `der_diar` is 0.48–0.56 in every
joint run, against 0.04 for the standalone Sortformer, and the diar-conditioned turn head misses 94 % of
turn ends. (B) Adding text made the head worse.

I ran all the diagnostics below on CPU with ≤ 64 validation conversations (`conversation_dataset(64, seed=1)`,
the training val split) and the saved models. No model was retrained. The scripts are in the session
scratchpad (`diag/a1_der.py`, `a1b_stream.py`, `a1c_prefix.py`, `a1d_eot.py`, `b_text.py`, `b2_shift.py`).

## A. Diarization

### A1. The DER is real. It is not an artefact of the metric or of column order.

On `turn_speaker.afm` I recomputed the DER independently (best permutation, with a miss / FA / confusion
breakdown) on 64 conversations:

| model / data | DER mean | median | miss | FA | confusion |
|---|---|---|---|---|---|
| joint (turn_speaker), conversations, [16,1] | **0.494** (= logged 0.4935) | 0.483 | 0.214 | 0.178 | 0.102 |
| joint, conversations, [16,3] look-ahead | 0.478 | 0.483 | 0.218 | 0.159 | 0.101 |
| joint, conversations, full context (never trained) | 0.764 | – | 0.656 | – | – |
| joint, 2-speaker ToneLanguage mixtures (`synthetic_dataset('diar',64,seed=2)`) | 0.661 | 0.645 | 0.149 | 0.247 | 0.265 |
| standalone `sortformer_diar.afm`, mixtures | 0.051 | 0.000 | 0.007 | 0.015 | 0.030 |
| standalone, conversations (zero-shot) | 0.342 | 0.349 | 0.027 | 0.100 | 0.215 |

- 80 % of conversations have DER > 0.3. The error is spread across items and is not caused by a few outliers.
  By number of other speakers (0/1/2), the joint DER is 0.28 / 0.50 / 0.59. So even single-speaker
  conversations have DER 0.28, which is pure activity (VAD) error.
- Column order is not the cause. The DER *without* permutation against arrival-sorted targets is 0.513, against
  0.494 with the best permutation. The head follows arrival order, and the permutation-free metric hides
  only 2 points.
- The printed tracks show a smeared, under-confident head: probabilities of 0.3–0.6, merged segments, and
  the third speaker often dropped entirely.

Which of (i)–(iv) causes it:

| hypothesis | evidence | verdict |
|---|---|---|
| (i) head under-trained or under-weighted | Joint `loss_diar` is 0.26 at step 150 and plateaus at 0.16–0.20 from step 400 to step 1000. Standalone reaches 0.08 at step 150 and 0.03 at step 1000. The standalone model *zero-shot* on conversations (0.34) beats the joint head trained on them (0.49). | **Main cause of the DER.** It is not only a matter of steps: the head plateaus at weight 0.2, sharing the encoder with TDT and CTC (weights 0.5 and 0.15), on harder data (3 speakers, gains 0.5–1.0, backchannels, SpecAugment on the labels). |
| (ii) causal encoder / [16,1] | Extra look-ahead [16,3] gains only 1.6 points. Speaker identity is still in the encoder: nearest-centroid speaker accuracy per layer is joint 0.55/0.61/0.73/0.70 against standalone 0.60/0.60/0.72/0.76. The top layer is slightly eroded by ASR, and layer 2 is as good as standalone. | Minor, a few points. It cannot be isolated without retraining (see the reruns below). |
| (iii) primary column vs arrival order | `primary_column` (arrival rank of column 0) and `sort_by_arrival` use the same stable argsort on the first active frame. Ties, silent columns, batch padding and the loss's label crop were all checked (new tests). The arrival-rank column is the best-matching offline column in 80 % of conversations; the other 20 % are head swaps and are already counted in the confusion term. | **No mismatch.** |
| (iv) a real bug | **Yes, in the evaluation path.** See below. | **Main cause of the 94 % miss.** |

**The bug: `streaming_diar_act` feeds the turn head a dead activity track.** To keep the diar source causal,
it re-runs the offline Sortformer head on every growing prefix and keeps each prefix's last chunk. The
head was only ever trained on whole episodes, and every episode ends in ≥ 2.6 s of trailing silence. On
a prefix, it outputs almost zero. For conversation #1, the maximum probability is 0.0 up to a 30-frame prefix
and 0.6 at 40 frames, while the full-length pass gives 0.5–0.9. Measured on 64 conversations, the causal
primary track **misses 91 % of the primary's speech frames**. The offline pass on the same column misses 22 %.
With the primary "never speaking", the kernel-conditioned turn head has no turn to end.

Turn-head latency with each primary-activity source (turn_speaker, 64 conversations, ≤ 5 % FC):

| activity fed to the turn head | P50 | miss |
|---|---|---|
| oracle | 240 ms | 6.6 % |
| offline Sortformer, arrival-rank column (non-causal upper bound) | 2240 ms | 47.5 % |
| offline Sortformer, binarised | inf | 52.5 % |
| streaming prefix re-run (what `eot_bench ... diar` uses) | inf | 86.9 % |

Even the non-causal upper bound is poor, which has two causes. The DER is 0.49, and the turn head was
trained only on clean oracle activity, so it never learned to handle a noisy track.

### A2. Train/inference consistency of the primary column

There is no mismatch. `tests/test_diag_turn.py` now pins it:
- `test_primary_column_matches_sort_loss_order_random`: 200 random batches with ties, silent speakers,
  and labels cropped or padded to the encoder length.
- `test_primary_column_on_collated_conversations`: real conversations. It includes lead-other episodes,
  where the primary is not column 0.

## B. Text branch

### B1. Training labels vs inference timing

- The logged `turn_align_greedy_frac` is **0.0023 for turn_text** and 0.49 for turn_speaker_text (cumulative
  over training). On the final models and 64 conversations, greedy forced alignment succeeds for 0 % of
  items (turn_text) and 78 % (speaker_text).
- **Bug: turn_text aligned on the wrong encoding.** The TDT head is `condition_on_speaker: true` in all four
  arms. `TurnHead.loss` ran `greedy_align` on *its own* input, which in the turn_text arm is the plain
  encoding. On that encoding the TDT joint gives 95 % WER, so greedy alignment always failed and **the arm
  trained on even-spread (uniform) labels 99.8 % of the time**. On the conditioned encoding, which the ASR
  head reads and which the inference path `decoded_text_state` uses, greedy succeeds for 58 %.
- Timing. With uniform spreading, the last token sits exactly on the last speech frame. The greedy-aligned
  last token is at median 0 (mean −0.4 frames). Greedy-*decoded* tokens at inference arrive at median **+1
  frame** (mean +1.1). In 77–81 % of turns the last decoded token comes after the last speech frame, and
  in 27–33 % it is ≥ 2 frames late.
- The hypothesis "text complete ⇒ end" is confirmed. With no tokens the head never fires (miss 100 % /
  98 %). The table feeds the reference text at controlled delays, with oracle activity and 64 conversations:

| turn head fed | turn_text P50 / miss | speaker_text P50 / miss |
|---|---|---|
| reference, uniform timing (turn_text's training distribution) | 880 ms / 9.8 % | 240 ms / 16.4 % |
| reference, greedy-aligned timing (speaker_text's training mix) | 1040 ms / 13.1 % | **160 ms / 3.3 %** (P90 1200) |
| reference, uniform +2 frames | 1280 ms / 18.0 % | 1040 ms / 18.0 % |
| reference, uniform +4 frames | 1440 ms / 18.0 % | inf / 59.0 % |
| reference, uniform +8 frames | 1520 ms / 23.0 % | inf / 93.4 % |
| **own greedy decode (inference)** | **1520 ms / 25.8 %** | **320 ms / 19.7 %** |
| own decode, emissions moved 1 frame earlier (non-causal probe) | 1280 ms / 14.5 % | 240 ms / 18.0 % |

  The text state carries the signal. Given text timed the way it was trained, speaker+text reaches
  160 ms / 3.3 %, better than speaker-only on the same 64 conversations (240 ms / 6.6 %). At inference,
  **turn_text** loses about half its extra misses to token lateness (26 % drops to 15 % when the lateness
  is removed). **speaker_text** loses almost nothing to timing (20 % to 18 %). Its misses come from
  *wrong* tokens: the ASR has 24 % WER, and when the last word is garbled (for example "righ" for "six"),
  the transcript never looks complete to the head.

### B2. Does the inference decode use the right encoding?

Yes. `decoded_text_state` decodes from the speaker-conditioned encoding whenever the ASR head is
conditioned. Inference token WER / CER with oracle activity, on 64 conversations:

- turn_text: **0.236 / 0.144**
- turn_speaker_text: **0.242 / 0.147**

These match the logged `wer_tdt_oracle_act`. Decoding from the wrong (plain) encoding would give
0.95–0.96 WER. That silent failure happened only on the *training* side, in the turn_text alignment
described in B1.

## Bug vs budget

| effect | kind | size |
|---|---|---|
| streaming diar track is dead (offline head re-run on prefixes) | **bug** (design of `streaming_diar_act` vs how the head is trained) | 91 % primary-frame miss, which becomes a 94 % turn miss |
| joint diar head plateaus at DER 0.49 | budget / weighting / data (not ordering, not a metric artefact) | 0.49 against standalone zero-shot 0.34 on the same data |
| causal [16,1] encoder | architecture, minor | [16,3] gains 1.6 DER points |
| turn head conditioned only on oracle activity | train/inference mismatch | offline diar gives 2240 ms / 48 % miss, against oracle 240 ms / 7 % |
| turn_text aligned on the plain encoding (99.8 % uniform labels) | **bug** | fixed |
| text labels 1 frame earlier than inference tokens, and error-free | train/inference mismatch | turn_text: 26 % miss vs 10 %; speaker_text: 20 % vs 3 % |

## Changes made (test-first, `tests/test_diag_turn.py`, 9 tests)

- `heads/turn.py`
  - `TurnHead.asr_view`: forced alignment and decoded text now use the encoding the ASR head reads. When the
    head and the ASR head differ in `condition_on_speaker`, the head re-encodes without gradient and without
    SpecAugment, or uses `batch["_asr_enc"]` if the model provides it.
    Test: `test_text_alignment_uses_the_asr_heads_encoding` (failed before the fix).
  - New train-time text options, all off by default so the existing runs are unchanged:
    - `text_delay: k` shifts each item's token frames by d ~ U{0..k}, clamped and monotonic.
    - `text_noise: p` substitutes each token with probability p.
    - `decoded_prob: p` trains on the ASR head's own greedy decode (tokens and emission frames, exactly the
      inference path) with probability p per step.
    - Diagnostics: `turn_decoded_frac`; `turn_align_greedy_frac` now counts only reference-aligned items.
  - Labels come from `batch["spk_act_oracle"]` when it is present, so the conditioning `spk_act` can be
    diarization output or noise-augmented without corrupting the EOT label.
  - `evaluate_model` reports `eot_turn_diar_act_miss`, the primary-frame miss of the activity the diar path
    feeds the turn head. It would have flagged this failure at 0.91.
- `heads/audio.py` `SortformerHead`: `prefix_prob` / `prefix_min`. With that probability per step, the head
  also takes the loss on a random prefix, with the cropped targets re-sorted (arrival order is causal). This
  makes the offline head valid for `streaming_diar_act`.
- The 5-step smoke run (8 items; arms text and speaker+text, all new options on) trains, and the metrics
  report the new diagnostics.

## Recommended recipe changes

```yaml
heads:
  diar: {type: sortformer, num_spks: 4, d_hidden: 96, n_layers: 2, weight: 0.5, from_layers: all,
         prefix_prob: 0.5, prefix_min: 8}
  turn: {type: turn, mode: kernel, condition_on_speaker: true, use_text: true, k_tokens: 4, text_dim: 64,
         align: greedy, text_delay: 2, text_noise: 0.1, decoded_prob: 0.3,
         hidden: 96, history: 8, pos_weight: 2.0, weight: 0.3}
```

`from_layers: all` lets the diar head read the middle layers, where speaker identity is best preserved.

Changes needed outside my files (patch intent only; not edited):

1. **`model.py` `SpeechModel.forward`: train the conditioning on imperfect activity.** With probability
   `p_diar_cond` (0.5), replace `batch["spk_act"]` for the conditioned encoder pass with the diar head's
   detached, prefix-causal primary column (`primary_column` + `streaming_diar_act`, or an offline pass
   for speed). Otherwise use an augmented oracle: boundary jitter of ±2 frames, dropped segments, and false
   alarms from other speakers. Keep the clean track in `batch["spk_act_oracle"]`; TurnHead already reads its
   labels from it. The same pass also makes the TDT head robust.
2. **`model.py`**: put the speaker-conditioned encoding in `batch["_asr_enc"]` (for example
   `batch["_asr_enc"] = enc_spk.detach()`) after it is computed. `TurnHead` then skips its extra no-grad
   encoder pass in the +text/no-speaker arm.
3. **`train.py`** (optional): log the diar loss per head weight. For the real-time path, the long-term fix is
   the streaming Sortformer with AOSC (`streaming_diar.py`, `pos_emb: true`, streaming-simulation loss)
   instead of the prefix re-run.

## Rerun commands

To separate (i) from (ii) for the DER, run a standalone Sortformer on conversations with a causal [16,1]
encoder against the non-causal version:

```bash
.venv/bin/python -m audioforge.cli train research/recipes/sortformer_diar.yaml data.synthetic.kind=conversation \
  encoder.causal=true 'encoder.att_context_size=[16,1]' -o runs/sortformer_conv_causal.afm > runs/sortformer_conv_causal.log
.venv/bin/python -m audioforge.cli train research/recipes/sortformer_diar.yaml data.synthetic.kind=conversation \
  -o runs/sortformer_conv_full.afm > runs/sortformer_conv_full.log
```

The ablation with the fixes. This is `scripts/train_turn_ablation.sh` with the new heads options appended to
every `run` line. Do it after the `model.py` patch (1) for the diar numbers:

```bash
FIX="heads.diar.weight=0.5 heads.diar.from_layers=all heads.diar.prefix_prob=0.5 \
     heads.turn.text_delay=2 heads.turn.text_noise=0.1 heads.turn.decoded_prob=0.3"
.venv/bin/python -m audioforge.cli train research/recipes/speaker_aware_turn.yaml $FIX heads.turn.mode=kernel \
  heads.turn.condition_on_speaker=true heads.turn.use_text=false -o runs/turn_speaker_v3.afm > runs/turn_speaker_v3.log
.venv/bin/python -m audioforge.cli train research/recipes/speaker_aware_turn.yaml $FIX heads.turn.mode=none \
  heads.turn.condition_on_speaker=false heads.turn.use_text=true -o runs/turn_text_v3.afm > runs/turn_text_v3.log
.venv/bin/python -m audioforge.cli train research/recipes/speaker_aware_turn.yaml $FIX \
  -o runs/turn_speaker_text_v3.afm > runs/turn_speaker_text_v3.log
PYTHONPATH=. .venv/bin/python -m audioforge.conversation runs/turn_acoustic.afm runs/turn_speaker_v3.afm \
  runs/turn_text_v3.afm runs/turn_speaker_text_v3.afm --n 1000 --out runs/turn_ablation_v3.json
```

What to check:
- `eot_turn_diar_act_miss` should be far below 0.91.
- `turn_align_greedy_frac` for the text arm should be ≫ 0.002.
- The +speaker+text arm should reach a miss rate at or below +speaker (4 %). With text arriving as in
  training, the diagnosis shows 160 ms / 3 %.
