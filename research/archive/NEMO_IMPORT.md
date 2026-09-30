# Importing NVIDIA's pretrained NeMo FastConformer weights (no NeMo install)

`audioforge/nemo_import.py` loads a `.nemo` checkpoint (CC-BY-4.0 models on Hugging Face) into our
from-scratch `SpeechModel`, weight for weight, so fine-tuning can start from a 100M-600M pretrained
encoder instead of from scratch.

```bash
.venv/bin/python -m audioforge.nemo_import nvidia/stt_en_fastconformer_hybrid_large_streaming_multi \
    runs/nemo_hybrid_streaming_multi.afm            # [--att-context-size 70,0] to change the default
.venv/bin/python -m audioforge.nemo_import data/nemo/parakeet-ctc-0.6b.nemo runs/nemo_parakeet_ctc_0.6b.afm
.venv/bin/python scripts/benchmark_teachers.py --teachers none --student runs/nemo_hybrid_streaming_multi.afm
```

In Python: `from audioforge.nemo_import import import_nemo; m = import_nemo("<hf id or .nemo path>")`.
HF ids are downloaded into `data/nemo/`. A `.nemo` is a tar of `model_config.yaml`,
`model_weights.ckpt` (torch state_dict) and the SentencePiece `*_tokenizer.model`; it is read in
place with `tarfile` (no extraction), the config is parsed with PyYAML.

## What had to be added to our encoder (all opt-in; defaults unchanged)

| NeMo | ours before | option added (`audioforge/modules/fastconformer.py`) |
|---|---|---|
| Transformer-XL rel-pos MHSA: `linear_q/k/v/out`, `linear_pos` (no bias), per-layer `pos_bias_u/v`, `rel_shift`, scores `(ac+bd)/sqrt(d_k)`, masked with -10000 | RoPE, fused qkv | `pos_emb: rel_pos` -> `modules/relpos.py` (same parameter names as NeMo) |
| `xscaling: true`: x * sqrt(d_model) after subsampling | none | `xscaling: true` |
| dw_striding subsampling with ReLU | SiLU | `subsampling_activation: relu` |
| `CausalConv2D`: pads (left 2, right 1) in time **and** frequency; length floor(l/2)+1 per stage; 80 mels -> 11 bins (Linear 2816->512) | causal pads time (2,0), freq (1,1); length ceil(l/2) | `subsampling_padding: nemo` (non-causal NeMo = our old non-causal layout) |
| encoder frame v ends at mel frame 8v; NeMo streams 8R+1 mels first, then 8(R+1) | frame v ends at 8v+7 | `stream_step` path `_stream_step_aligned` (new `final=` kwarg) |

Already identical: macaron block order and 0.5 FF residuals, pre-LN placement, Swish FF, conv module
(pw1 -> GLU -> pad-mask -> depthwise k=9 (causal: left pad 8) -> LayerNorm/BatchNorm -> Swish -> pw2),
`chunked_limited` mask (chunk = R+1, left = L // (R+1) chunks), log-mel front end (Slaney mel fb
matches the checkpoint's stored `fb` to 1.9e-9, Hann window to 6e-8; preemph 0.97, log(x+2^-24),
`normalize: NA` = none, or `per_feature` with (n-1) std + 1e-5), RNNT prediction net (LSTM; the
blank/SOS embedding row is all zeros in the checkpoint, equal to NeMo's zero SOS input), joint
(enc+pred -> ReLU -> Linear), output order `[vocab..., blank]` with blank = 1024 for both heads.

## Key mapping (every checkpoint tensor mapped; missing/unexpected/shape mismatch raise)

`pre_encode.conv.{0,2,3,5,6}` -> `pre_encode.convs.{0,1.0,1.1,2.0,2.1}`; `layers.N.norm_feed_forward{1,2}`
-> `ff{1,2}.0`, `feed_forward{i}.linear{1,2}` -> `ff{i}.{1,4}`; `norm_self_att` -> `norm_att`;
`self_attn.*` -> `att.*`; `conv.{pointwise_conv1,depthwise_conv,batch_norm,pointwise_conv2}` ->
`conv.{pw1,dw,norm,pw2}` (`batch_norm` is a LayerNorm when `conv_norm_type: layer_norm`);
`decoder.prediction.{embed,dec_rnn.lstm}` -> `heads.rnnt.pred.{embed,lstm}`; `joint.{enc,pred,joint_net.2}`
-> `heads.rnnt.joint.{enc,pred,out.2}`; `ctc_decoder.decoder_layers.0` (hybrid) / `decoder.decoder_layers.0`
(CTC model) -> `heads.ctc.proj` (Conv1d k=1 squeezed to Linear). `preprocessor.featurizer.{window,fb}`
are recomputed and compared, not loaded.

## Results (LibriSpeech test-clean, first 200 utterances, greedy, MPS fp32)

stt_en_fastconformer_hybrid_large_streaming_multi (114.8M params, load 1.5 s from .nemo, 3.6 s from .afm):

| att_context_size | lookahead | RNNT WER % | CTC WER % |
|---|---|---:|---:|
| [70,13] (default) | 1.04 s | **1.92** | 2.05 |
| [70,6] | 0.48 s | 1.94 | 2.27 |
| [70,1] | 0.08 s | 2.29 | 2.63 |
| [70,0] | 0 | 2.48 | 2.89 |

The model card only reports full LibriSpeech **test-other** (RNNT 5.4/5.7/6.4/7.0, CTC 6.2/6.7/7.8/8.4
for the four contexts); the test-clean-200 numbers show the same monotone ordering and RNNT < CTC.

parakeet-ctc-0.6b (608.8M params, 24 x d1024, non-causal, BatchNorm conv, per_feature norm):
**1.68 %** vs 1.73 % for the transformers port of the same checkpoint on the same 200 utterances
(same benchmark run, batch 8). The hypotheses are identical on 199/200 utterances; the one difference
is 121-127105-0014, "you are acute" (ours, correct) vs "you are ac cute" (transformers) - a 1.2 s
utterance, consistent with the batch-padding difference noted below. RTFx 75.8 (ours) vs 5.5.

## Streaming

On the ported streaming model, `encoder.stream_step` fed 8(R+1) mel frames per call reproduces the
offline chunked-limited forward: max |diff| 1e-6..1.4e-5 (activations ~1.6) over 3 real utterances at
[70,13] and [70,0], identical frame counts, and `StreamingSession` (raw audio in 100 ms pieces) gives
exactly the offline transcript. Covered by `tests/test_nemo_import.py` (synthetic tiny .nemo, 5 lengths
x 3 contexts, plus the real checkpoint when present).

## Differences from NeMo that remain

* NeMo's streaming caches the normalized layer input and re-projects (and keeps a fixed 70-frame,
  zero-padded cache with masking); we cache projected k/v of exactly L frames. Same outputs.
* We re-zero padded frames between subsampling stages (NeMo >= 2.x does too; the 1.16 code that
  trained this model did not). For the causal model this changes nothing (no right context); for the
  non-causal parakeet it only affects the last valid frame of padded batch items (ours is batch-invariant).
* `StreamingSession.feed(final=True)` does not pass `final` to `stream_step`; `final` is inferred from a
  short last chunk, so if the audio ends exactly on a chunk boundary the one trailing frame that the
  offline forward computes over zero padding is not emitted (could change the last token at most).
* Not supported (raise `NotImplementedError`): other subsampling types, abs_pos/rope/local attention,
  `regular` style with limited context, AED (Canary) decoders. (Stacked LSTM prediction nets were added with the
  0.6B import, research/archive/ENC_0P6B.md; TDT with the Parakeet-TDT import below.)

## Parakeet-TDT (token-and-duration transducer), 2026-09-27

`nvidia/parakeet-tdt-0.6b-v3` (CC-BY-4.0; research/archive/HYBRID_ASR.md) goes through `import_nemo` unchanged except for the
head: `tdt_durations(nc)` reads the duration set from `model_defaults.tdt_durations`, `loss.tdt_kwargs.durations` and
`decoding.durations` (all must agree) and checks it against `joint.num_extra_outputs`; the head is built as
`type: tdt` (key still `rnnt`, so the weight mapping is the RNNT one) with the joint's output layer
`[vocab..., blank, durations...]` = 8192 + 1 + 5 rows, exactly NeMo's `RNNTJoint` layout. Greedy decoding
(`RNNTHead._greedy`, and `GreedyTransducerStream` for frames arriving in pieces) takes the argmax token and the argmax
duration; blank with duration 0 advances one frame; `max_symbols` 10 per frame. Real import: 725 tensors, 723
loaded, 264 zero biases (`use_bias: false`), 627.5M parameters, mel filterbank 3.7e-9, window 6e-8, 6 s from the
memory-mapped .nemo. Tests (`tests/test_hybrid_asr.py`): a synthetic TDT archive with the v3 config shape (strict
count `n_loaded == archive - 2`, bit-identical tensors and transcripts), loud failures for an unmapped tensor, a
missing tensor, `num_extra_outputs` != durations and inconsistent durations; the incremental greedy decoder equals
the offline one for 4 piece patterns; the real checkpoint (gated on `AUDIOFORGE_BIG_TESTS=1`) gives the counts and
the same tokens decoded offline and in 3-frame pieces.
