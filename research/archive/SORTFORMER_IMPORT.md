# NVIDIA Streaming Sortformer → audioforge (no NeMo)

**Checkpoint used: `nvidia/diar_streaming_sortformer_4spk-v2` (CC-BY-4.0), not v2.1.**
The v2.1 card (`research/archive/raw/cards/diar_streaming_sortformer_4spk-v2.1.md`, and `research/catalog.json`:
`"license": "other"`) is under the **NVIDIA Open Model License**, not CC-BY-4.0. The fallback named in the task,
`diar_sortformer_4spk-v1`, is **CC-BY-NC-4.0** (non-commercial), so it was not used either. v2 has the
same architecture as v2.1 and its card and catalog entry both say CC-BY-4.0. `nemo_import.license_of()` records
the card license in `cfg.nemo_source.license`, so a v2.1 import would be labelled correctly.

Output: `runs/nemo_sortformer_v2.afm`, a `SpeechModel` with no tokenizer and a single `diar` head
(type `sortformer`, top encoder layer, no `from_layers`). `scripts/eval_stage1.py --ckpt` evaluates it directly.

## What the .nemo contains (from the archive itself)

`model_config.yaml`, target `SortformerEncLabelModel`, nemo 2.2.0rc0, `streaming_mode: true`. There are 990 tensors:

| part | config | tensors |
|---|---|---|
| preprocessor | 128 mels, n_fft 512, 25/10 ms, **normalize NA**, dither 1e-5, preemph default | window, fb (recomputed by our LogMel, max abs diff 1.2e-7 / 5.6e-9) |
| encoder (NEST) | ConformerEncoder 17 × d512, 8 heads, rel_pos, untie_biases, xscaling, dw_striding 8× (256 ch), batch_norm conv, kernel 9, full context, non-causal | same names as the ASR models → existing `_LAYER`/`_TOP` mapping |
| `sortformer_modules` | fc 512 → tf 192, 4 spk | encoder_proj, first_hidden_to_hidden, single_hidden_to_spks, hidden_to_spks (unused) |
| `transformer_encoder` | 18 layers, hidden 192, inner 768, 8 heads, relu, **pre_ln false** | per layer: q/k/v/out nets, layer_norm_1/2, dense_in/out |

Forward pass, read from the NeMo sources (`sortformer_diar_models.py`, `sortformer_modules.py`,
`transformer_encoders.py`, `transformer_modules.py`, NeMo main):
`emb = encoder_proj(ConformerEncoder(mel))`, then `TransformerEncoder(emb, mask)` with **no positional
encoding**. The blocks are post-LN: `LN1(x + MHA(x))`, `LN2(y + FF(y))`, and there is no final LN because pre_ln is false. MHA
scales both q and k by head_dim^-0.25 and masks padded keys with -10000. The output is
`logits = single_hidden_to_spks(relu(first_hidden_to_hidden(relu(h))))` and `sigmoid` gives the probabilities.
`hidden_to_spks` (Linear 384→4, `requires_grad_(False)`) is never called. `process_signal` peak-normalizes the waveform
only when `streaming_mode` is false. It is true for this model, so no waveform normalization is applied.

## Mapping (audioforge/nemo_import.py)

- `SortformerHead` gets two opt-in knobs whose defaults leave the head unchanged (bit-identical, tested):
  `norm_first=False` (post-LN `nn.TransformerEncoderLayer`) and `out_pre_relu=True` (output MLP becomes
  `[ReLU, Linear, ReLU, Linear]`). `pos_emb=False`. The FastConformer encoder needed no change.
- `encoder_proj` → `heads.diar.proj`; `first_hidden_to_hidden` → `out.1`; `single_hidden_to_spks` → `out.3`.
- Per layer: `cat(query_net, key_net, value_net)` → `self_attn.in_proj_{weight,bias}`; `out_projection` →
  `self_attn.out_proj`; `layer_norm_1/2` → `norm1/2`; `dense_in/out` → `linear1/2`.
- Strict load: **990 NeMo tensors → 914 audioforge tensors, `load_state_dict(strict=True)`, with no missing,
  unexpected or wrong-shape tensors.** The difference is 108 q/k/v tensors fused into 36 in_proj tensors (−72), 2 frontend buffers recomputed,
  and 2 dead `hidden_to_spks` tensors dropped. Total 117.7M params (the card says 117M).
- The NeMo training-time streaming settings are copied to `cfg.streaming` (chunk 188, rc 1, fifo 0, cache 188,
  update 188). Non-streaming Sortformers raise `NotImplementedError` because the waveform peak-normalization step is not implemented.
- `tests/test_sortformer_import.py` builds a tiny synthetic Sortformer .nemo and checks that our head equals an
  independent re-implementation of NeMo's forward (atol 1e-5, with padding). It also checks that the default head is unchanged
  and loads the real checkpoint when it is present (990/914 counts; silence gives no speaker).

## Results

**Synthetic ToneLanguage mixtures (8, `synthetic_dataset("diar", 8)`):** the model outputs almost no activity. The mean
frame DER is 0.95, with 7 of 8 windows all-silent and 1 window with speaker 0 roughly placed. These mixtures are pure-tone "speech"
about 2 s long. A speech-trained diarizer treating them as non-speech is expected, so this set cannot check
the port. The numeric test above and AMI below are the real checks.

**AMI dev, 64 × 20 s diar windows** (`audioforge.datasets.ami`, mode diar, split dev, 4 meetings, seed 0).
The numbers come from the same code path as research/archive/STAGE1.md (`scripts/eval_stage1.py --tasks diar`, `metrics.frame_der`, pooled
miss/FA/confusion at the best permutation, threshold 0.5). 14428 reference speaker-frames, 1838 overlap frames.

| system | pooled DER | miss | FA | confusion | mean of window DERs |
|---|---|---|---|---|---|
| **NeMo Sortformer v2, offline** (one head pass over the 20 s window) | **0.201** | 0.140 | 0.045 | 0.016 | 1.45 |
| NeMo Sortformer v2, our streaming, card "high latency" (chunk 124, rc 1, fifo 124, update 124, cache 188) | 0.212 | 0.142 | 0.046 | 0.024 | 1.46 |
| NeMo Sortformer v2, our streaming, card "low latency" (chunk 6, rc 7, fifo 188, update 144, cache 188) | 0.252 | 0.146 | 0.045 | 0.061 | 1.63 |
| our stage-1 diar head (STAGE1.md, final) | 0.384 | 0.24 | 0.06 | 0.09 | — |
| trivial: one speaker + oracle VAD | 0.312 | 0.144 | 0 | 0.168 | 0.29 |
| trivial: one speaker always on | 0.569 | 0.144 | 0.257 | 0.168 | 9.83 |

The mean of per-window DERs is dominated by near-silent windows, as noted in STAGE1.md. Pooled DER is the number to compare.
The imported model roughly halves our head's DER and beats the speaker-blind oracle-VAD baseline by 0.11.
Most of the remaining error is miss (14%). Confusion is small (1.6% offline).
Card numbers (about 10–20% on AMI-like sets) use full sessions, collar and scoring conventions that are not
comparable to these 20 s frame-level windows with no collar.

### Streaming caveat

`audioforge/streaming_diar.py`'s `StreamingDiarizer` (window mode, because the encoder is non-causal) does not run the NeMo procedure. We
re-encode audio for [enc_left_context ‖ chunk ‖ rc] and cache the **head's projected embeddings**
(`encoder_proj` outputs). Only the 18-layer transformer runs over [cache ‖ fifo ‖ chunk ‖ rc].
NeMo caches **pre-encode (subsampling) frames** and re-runs the full FastConformer and transformer over the
concatenation. Its cache-compression rule also differs in detail: we use our `select_cache_frames`/AOSC
defaults (sil_frames 4, strong_boost 0.75), while NeMo uses spkcache_sil_frames_per_spk 3 plus its own boosting and silence profile.
The runs used `enc_left_context` = 124 (high) and 188 (low), so the encoder saw about the FIFO span of past audio.
A 20 s window is 250 frames, so the "high latency" config takes 2 steps and cannot exercise the cache much. The
"very high latency" config (chunk 340) would equal the offline pass here and was not run. These streaming numbers therefore measure the
imported weights under our streaming procedure. They are not a reproduction of NeMo streaming.

## Commands

```bash
# import (downloads the .nemo, 471 MB, into data/nemo/) and save
.venv/bin/python -m audioforge.nemo_import data/nemo/diar_streaming_sortformer_4spk-v2.nemo runs/nemo_sortformer_v2.afm
# offline AMI DER (15 s on 2 CPU threads)
TMPDIR=<scratch> .venv/bin/python scripts/eval_stage1.py --ckpt runs/nemo_sortformer_v2.afm --n 64 --tasks diar --out <scratch>/eval.json
# tests
.venv/bin/python -m pytest -q tests/test_sortformer_import.py tests/test_nemo_import.py
```

The streaming numbers came from a scratch script (not in the repo). For each of the same 64 windows
(`eval_stage1.ami_dev("diar", 64)`) it ran `StreamingDiarizer(model, mode="window", enc_left_context=L, **card_cfg).feed(audio, final=True)`
and thresholded `all_probs` at 0.5. It then pooled `eval_stage1.der_parts` over the windows. The low-latency run took
about 7.5 min of CPU time, split across 3 processes.

## Newer NVIDIA diarizers: Sortformer v2.1 and Nemotron-3-Diarization (2026-09-26)

### Licenses (checked in the cards and the license texts)
- **`diar_streaming_sortformer_4spk-v2.1`: NVIDIA Open Model License** (card front matter `license: other`, `nvidia-open-model-license`, and the
  agreement dated 2025-10-24). You may use it commercially. NVIDIA claims no ownership of outputs, so the tracks and labels it produces are
  not derivative models. Obligations:
  (a) any redistribution of the model or a derivative must include the agreement and the notice
  **"Licensed by NVIDIA Corporation under the NVIDIA Open Model License"**;
  (b) **guardrail clause**: your rights terminate automatically if you "bypass, disable, reduce the efficacy of, or circumvent any technical
  limitation, safety guardrail … without a substantially similar Guardrail appropriate for your use case";
  (c) your rights also terminate if you file patent or copyright litigation over the model;
  (d) use must follow NVIDIA's Trustworthy AI terms.
  `license_of()` labels the import accordingly.
- **`Nemotron-3-Diarization`: OpenMDW-1.1** (card front matter `license: openmdw-1.1`, section "License/Terms of Use", and the HF API).
  The license is permissive and imposes no restrictions on outputs. A redistribution must keep a copy of the agreement and all copyright
  and origin notices. Your rights terminate if you sue claiming the model infringes a patent or copyright. It was released 2026-09-23.

### What loaded
- **v2.1** has the same architecture as v2. Its config differs only in the streaming-cache score fields and the nemo version (2.6.0rc0).
  The unchanged importer loads **990 NeMo tensors into 914, `strict=True`**, 117.7M params → `runs/nemo_sortformer_v2_1.afm`.
- **Nemotron-3** is a different network. It has no FastConformer and no Sortformer transformer:
  - 10 ms mel (128), stacked 8 frames at a time, then `Linear(1024→512, no bias)`;
  - LayerNorm, then **31 pre-LN blocks** with RoPE (rotate-half, θ 1e4, positions restart every chunk), fused bias-free `w_qkv`, and exact-GELU FFN 2048;
  - a final LN, then `encoder_proj` 512→192;
  - a **Conv1d(192→8·192, k3) sub-pixel upsample to 10 ms**;
  - `relu→Linear→relu→Linear`, giving **8 speakers**.

  The streaming cache stores the stacked projections, so NeMo re-runs the whole network over [cache‖fifo‖chunk‖rc].
  `SpeechModel` always builds a FastConformer (model.py, not mine). The port is therefore a separate class in `audioforge/nemo_import.py`,
  `Nemotron3Diarizer` (LogMel + `FeatureStackingEncoder` + `Nemotron3DiarHead`). It is duck-typed to the SpeechModel surface that eval_stage1
  and `StreamingDiarizer` use. **363 NeMo tensors (bf16) load as 355 in fp32, `strict=True`**, 99.2M params (the card says 100M). Dropped:
  2 frontend buffers (fb differs from ours by 1.2e-4, which is bf16 rounding), 2 `hidden_to_spks` (never called), and 4 `activity_head` (auxiliary training loss only).
  `learnable_sil_emb` is kept as `heads.diar.sil_emb`, although only NeMo's cache rule uses it.
  The file is `runs/nemo_nemotron3_diar.afm`: the save_model layout plus `cfg.arch: nemotron3_diar`. Load it with `nemo_import.load_any`.
  `SortformerHead` needed no new knob, and heads/audio.py is unchanged.
- **Numeric check against HF transformers' reference** (`modeling_nemotron3_diarization.py` from transformers main, weights passed through its
  own converter, strict load, on 4 real AMI windows): 10 ms logits max |Δ| ≤ 7.6e-5, and 80 ms probabilities ≤ 1.7e-6.
- **Streaming:** `StreamingDiarizer(mode="window")` with the card low-latency profile (chunk 9, rc 4, fifo 264, cache 264, update 222) matches HF's
  chunked forward on the same profile within **max |Δp| 1.7e-3, with identical hard decisions**. For comparison, streaming differs from offline by up to 0.9.
  For this model our procedure is NeMo's procedure: the head re-runs the full network over the cached encoder-input frames, with no deviation.
  Caveat: in ≤ 21 s windows the FIFO (264 frames) never overflows, so the cache-compression rule is never exercised. Ours also differs from NeMo's there:
  its silence slot uses the learned embedding.
- **80 ms outputs.** The 10 ms probabilities are pooled per 80 ms frame. `pool="mean"` (the default) is the pooling NeMo uses for cache scoring. `pool="max"` marks a frame
  active if any 10 ms sub-frame is active. That is the "any overlap" convention of our AMI labels (`int(s/.08)..ceil(e/.08)-1`).
- **8→4 columns for DER.** `decode_top_k=4` keeps each window's 4 most active columns (by frames > 0.5, ties broken by summed probability).
  Scoring all 8 columns, with the best injective ref→hyp map, gives the same pooled DER at n=64, n=200 and in streaming, so the top-4 choice does not change any number here.

### Comparison (AMI dev, headset mix, word-level labels, 80 ms frames, threshold 0.5, no collar)
DER comes from `scripts/eval_stage1.py --tasks diar` (unmodified; only `load_model` was swapped for `load_any` in a scratch runner). The primary-track stats use the
200 dev TURN windows, offline tracks, and eval_stage1's `enroll_column` + `act_stats`: the column with the most overlap with the oracle over [onset, turn_end).

| diarizer | DER n=64 (miss/FA/conf) | DER n=200 | stream DER n=64 (1.04 s) | primary miss | FA/speech | post-end active | offline CPU (200 turn) |
|---|---|---|---|---|---|---|---|
| **v2** (reproduced) | **0.201** (.140/.045/.016) | **0.210** | 0.252 (ours, not NeMo's procedure) | **0.129** | 0.205 | 0.185 | 47 s |
| v2.1 | 0.242 (.205/.023/.014) | 0.248 | — | 0.194 | 0.162 | **0.141** | 71 s |
| Nemotron-3, mean pool | 0.255 (.222/.021/.013) | 0.255 | 0.263 | 0.220 | 0.162 | 0.148 | **22 s** |
| Nemotron-3, max pool | 0.232 (.191/.027/.014) | 0.229 | **0.241** | 0.192 | 0.179 | 0.162 | 22 s |

Nemotron-3 streaming primary track (max pool, 200 turn windows): miss 0.207, FA 0.181, post-end 0.163. The v2 streaming track (cached, research/archive/STAGE1.md) has
miss 0.152, FA 0.231, post-end 0.219.

**Why the newer models score worse here, although the cards say 16 → 9 DER on AMI test.** The error is almost all **missed speech at the VAD
level**: frames where a reference speaker talks and the model marks nobody. At threshold 0.5 that is 10.3% for v2, 15.6% for v2.1 and 19.0% / 16.4% for Nemotron (mean / max),
against a VAD false-alarm rate of only 1.3–2.6% (scratch `diag` run). Confusion is 1–1.6% for all models, and overlap detection is similar.
v2.1 and Nemotron were trained on (and are scored by NVIDIA against) the NTT forced-alignment RTTMs, which mark less speech. Our references mark
every 80 ms frame touched by an AMI word. Lowering the threshold does not rescue them: the best DER over thresholds 0.3/0.4/0.5 is v2 0.201, v2.1 0.232,
and Nemotron-max 0.231. The comparison is therefore about label conventions as well as model quality, and the card itself warns that DER depends on the reference labels.
Contamination: the Nemotron card lists **AMI train + dev** among its training data, so our dev set is not held out for it. v2/v2.1 list AMI without splits.
Even so, it scores worse on our labels.

**Recommendation.**
- **Turn-head training tracks:** keep **v2**. It has the lowest primary miss (0.129 vs ≥ 0.19) and the best offline DER on our labels. The existing caches
  (data/ami/cache/sortformer/dev) stay valid. v2.1 and Nemotron trade about 6–9 points of miss for 3–4 points less FA and post-end activity, which does not help a silence-timeout or turn head
  that must see the primary speaking.
- **Product (streaming):** **Nemotron-3 with max pooling** is the best measured low-latency diarizer: 0.241 vs v2's 0.252 at the same 1.04 s buffer.
  Its streaming loses only 0.009 DER against offline, versus 0.051 for v2. It also has the cleanest license (OpenMDW-1.1, no notice text or guardrail clause), runs 2× faster, handles 8 speakers,
  and our StreamingDiarizer reproduces NVIDIA's procedure for it. Its miss bias needs a label-convention check, or a small fine-tune on our word-level labels, before it replaces v2 for turn tracks.
  Do not use v2.1: it is worse than v2 here and carries the NVIDIA OML notice and guardrail obligations.

### Patch intent (files owned by other agents)
- `audioforge/train.py::load_model` (or eval_stage1): dispatch on `cfg.get("arch") == "nemotron3_diar"` to `nemo_import.Nemotron3Diarizer`,
  i.e. call `nemo_import.load_any`. Then `eval_stage1.py --ckpt/--diar-ckpt runs/nemo_nemotron3_diar.afm` works as is. The head reports
  `type: sortformer`, so `_diar_name` finds it. For DER with 8 columns, set `heads.diar.decode_top_k = 4`: eval_stage1's `der_parts`
  enumerates 8! permutations, which takes about 4 min at n=200. `make_sortformer_tracks.py` / `ext_tracks.py`: use `enc_left_context=1` for this model. Its encoder is
  frame-local, so a larger left context only costs time.

### Commands
```bash
.venv/bin/python -m audioforge.nemo_import data/nemo/diar_streaming_sortformer_4spk-v2.1.nemo runs/nemo_sortformer_v2_1.afm
.venv/bin/python -m audioforge.nemo_import data/nemo/Nemotron-3-Diarization.nemo runs/nemo_nemotron3_diar.afm
TMPDIR=<scratch> .venv/bin/python scripts/eval_stage1.py --ckpt runs/nemo_sortformer_v2_1.afm --n 64 --tasks diar --out <scratch>/v21.json
# Nemotron / primary-track / streaming numbers: scratch runner <scratch>/diar2/run_eval.py
#   {der <afm> <n> 4 | turn <afm> 200 | stream <afm> 64 <budget> | turnstream <afm> 200 <budget> | diag <afm,...> 64}, POOL=max env
.venv/bin/python -m pytest -q tests/test_sortformer_import.py tests/test_nemo_import.py
```
