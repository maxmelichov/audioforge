# NVIDIA audio models on Hugging Face: what they have in common

*Data pulled 2026-09-25 from `huggingface.co/api/models?author=nvidia` plus every model card
(144 of 149 cards downloaded; 4 are gated (Nemotron-3-Diarization-preview, personaplex-7b-v1,
Audio2Emotion-v2.2, Audio2Emotion-v3.0), and the card for nemotron-labs-audio-visual-flamingo-hf, which is
public, was not fetched in this pull.) The raw data is in `raw/`, the classified catalog in
`catalog.json`. Refresh with `python -m audioforge.cli catalog fetch`. Download counts are HF's
rolling 30-day numbers.*

## 1. The landscape

There are **149 audio models** out of 943 NVIDIA repos. Together they had **5.6M downloads in 30 days**.

| Family | # | 30-day downloads | Encoder | Head / decoder |
|---|---:|---:|---|---|
| Parakeet / Nemotron ASR | 16 | 2,544,279 | FastConformer | CTC, RNNT, TDT, TDT+CTC |
| Vocoders (BigVGAN, HiFi-GAN) | 10 | 2,177,324 | – (mel input) | GAN upsampler |
| Sortformer diarization | 5 | 273,519 | FastConformer (NEST) | Transformer + sigmoid per speaker, sort loss |
| Full-duplex speech LLM (VoiceChat, PersonaPlex) | 2 | 185,375 | FastConformer / Mimi | LLM + TTS codec decoder |
| TitaNet speaker embedding | 1 | 168,878 | 1-D conv + SE | attentive-stats pooling + AAM-softmax |
| Audio/Music Flamingo | 16 | 110,161 | AF-Whisper | decoder-only LLM (Qwen2.5) |
| Canary (ASR + translation) | 4 | 64,191 | FastConformer | Transformer AED with task prompts |
| Canary-Qwen (SALM) | 1 | 35,174 | FastConformer | Qwen3-1.7B + LoRA |
| NeMo audio codecs (NanoCodec, LFR) | 8 | 21,464 | conv + FSQ | HiFi-GAN decoder |
| Legacy `stt_*` Conformer / FastConformer / Citrinet | 65 | 28,934 | Conformer → FastConformer | CTC, RNNT, hybrid |
| Enhancement / restoration | 7 | 11,882 | Mamba, U-Net, flow-matching Transformer | generative |
| TTS (Magpie, FastPitch) | 2 | 6,956 | causal Transformer | codec tokens / mel |
| VAD, SLU, NEST, multitalker, animation, multimodal embedding | 12 | ~16,700 | mixed: FastConformer (multitalker, NEST, SLU), MarbleNet (VAD), audio-feature net (animation), LLM (embedding) | frame heads, RNNT, AED, blendshapes, embeddings |

The EOU model (parakeet_realtime_eou) is counted in the Parakeet / Nemotron ASR row. (Corrected
2026-09-25: this row used to read "VAD, EOU, SLU, NEST... 12 / ~6,000, mostly FastConformer"; only 4 of
the 12 are FastConformer, see `VERIFICATION.md` C17.)

## 2. The common denominator: the NVIDIA speech recipe

Across Parakeet, Canary, Nemotron-ASR, Sortformer, NEST, multitalker, EOU, SLU and the VoiceChat
encoder, the same design repeats. **Since mid-2023 (Hugging Face creation date ≥ 2023-06-01), 51 of
54 NVIDIA recognition-type models use the same FastConformer encoder.** The three exceptions are one
small 2023 Conformer (stt_en_conformer_ctc_small), the multilingual frame-VAD MarbleNet v2.0 (a 1-D
conv network), and PersonaPlex, which is built on Moshi.

1. **One front end.** 16 kHz mono audio → log-mel (25 ms window, 10 ms hop; 80 bands in the configs
   we checked) → SpecAugment. Feature normalisation varies (per_feature in Parakeet-CTC-0.6B, none in
   the cache-aware streaming models). Every recognition card lists "Input: 16kHz Audio, 1D".
2. **One encoder: FastConformer.** An 8× depthwise-separable conv subsampling stage gives
   **80 ms frames (12.5 Hz)**, followed by macaron Conformer blocks. Only the size changes:
   17 layers (EOU 120M, streaming Sortformer v2/v2.1), 18 (NEST-L, Sortformer v1), 24 (Parakeet/Nemotron
   0.6B, original Canary-1B), 32 (Canary-1B-v2).
3. **Swappable heads on that encoder.** The task is set only by the head:
   CTC (fastest), RNNT (streaming), **TDT** (RNNT plus a duration output that skips frames and gives
   2–3× faster decoding), Transformer AED (multitask via prompts), sigmoid-per-speaker (Sortformer),
   or an LLM through a projector (SALM). Hybrid heads share one encoder and add their losses
   (`parakeet-tdt_ctc-*`, `*_hybrid_*`).
4. **Streaming is a mask, not a separate model.** Cache-aware streaming uses
   `att_context_size=[left, right]` in 80 ms frames, plus causal convolutions and caches for
   attention and conv state. Training on several context sizes gives one checkpoint that covers
   80 ms to 1.1 s of latency and offline use (`*_streaming_multi`, `parakeet-unified`, Nemotron-ASR).
5. **Pretrain once, fine-tune everywhere.** The encoder is initialized from SSL (NEST) or a large
   CTC model trained on Granary pseudo-labels (660k–1.7M hours), then fine-tuned in stage 2 on
   about 7.5–10k hours of human-labeled data. Sortformer, multitalker and Canary-Qwen all start from
   an existing FastConformer.
6. **Tokens are the interface.** Each model has one SentencePiece BPE vocabulary (1k–16k), and
   tasks are expressed as tokens: Canary's `<|src|><|task|><|tgt|><|pnc|>`, EOU's `<EOU>`, and
   Nemotron-3.5's language-ID prompt. The model structure stays fixed while tasks grow.
7. **Conditioning is added inside the encoder, not bolted on after it.** Multitalker ASR adds
   "speaker kernels" to encoder layers. Nemotron-3.5 fuses a language-ID encoding before the decoder.
   Streaming Sortformer caches speaker embeddings (AOSC) taken from the pre-encode layer.
8. **Same packaging and deployment.** Every model is a `.nemo` archive (YAML config, weights and
   tokenizer), trained from JSONL manifests with Hydra configs and `from_pretrained`, exported to
   ONNX/TensorRT for Riva or NIM, and most are released under CC-BY-4.0 (84 of 149).

The **generation side** repeats a second pattern: **FSQ codec tokens + HiFi-GAN-style decoder
+ Transformer that predicts tokens.** NanoCodec and the Low-Frame-Rate codec use FSQ with 4-dim
codes and GAN training with an SLM discriminator. Magpie-TTS is a causal Transformer
encoder-decoder that predicts NanoCodec tokens. VoiceChat's speech output uses the same codec.
BigVGAN is the same GAN recipe, but conditioned on mel.

**The link between the two sides:** FastConformer runs at 12.5 Hz (8x subsampling of 10 ms frames =
80 ms), and so do two of the three NanoCodecs (the 12.5 fps variants: 22,050/12.5 = 1,764 samples =
80 ms; the 21.5 fps variant, which Magpie-TTS prefers, does not match). The recognition encoder and
those generation tokenizers share a clock.

## 3. Can we build new solutions this way? Yes, and here is how

The recipe can be summed up as *a shared front end and FastConformer, many small heads, streaming
through masks, tasks through tokens*. That makes new models a matter of **combining heads** rather
than designing new architectures. `audioforge/` rebuilds the recipe from scratch in plain PyTorch
(no NeMo dependency) and adds three new designs:

| Recipe | Status | Idea |
|---|---|---|
| `parakeet_tdt_ctc` | reproduction | offline FastConformer + TDT + CTC hybrid |
| `nemotron_streaming_rnnt` | reproduction | cache-aware causal encoder, multi-lookahead training |
| `canary_aed` | reproduction | AED with task prompts (ASR vs. "translation") |
| `sortformer_diar` | reproduction | sort loss + PIL diarization |
| **`voice_agent_frontend`** | new | **one** streaming encoder pass → ASR (TDT) + VAD + end-of-utterance + speaker ID. NVIDIA covers this with three models: a 91.5K MarbleNet VAD, a 120M ASR+EOU model (parakeet_realtime_eou), and a 117M Streaming Sortformer that re-encodes its [speaker cache; FIFO; chunk] window at every streaming step. TitaNet (23M) is added only for enrolled speaker ID. One pass costs ~1/4 of the encoder compute at inference, but at equal training steps the shared encoder learns ASR and VAD much more slowly. |
| **`speaker_attributed_asr`** | new | the Sortformer head and the speaker-kernel ASR share **one** encoder. At inference its own diarization output can steer per-speaker transcription (per-speaker WER ~18% end-to-end on the synthetic task, vs. 3.0% with oracle activity, the gap is diarization error propagating into ASR). NVIDIA needs two models for this (diarizer + one ASR instance per speaker); Microsoft's VibeVoice-ASR-Streaming does it in one 1.5B/7B LLM-based model, not a shared small streaming encoder. |
| **`codec_token_enhancer`** | new | noisy audio → *clean* FSQ codec tokens in one non-autoregressive pass, using the 12.5 Hz frame match with the codec. The same approach applies to voice conversion and bandwidth extension. Our experiment uses its own 16 kHz FSQ MelCodec (8 stacked 10 ms mel frames), not NanoCodec, so the NanoCodec pairing is still untested. Codec-token speech enhancement has prior art (e.g. SELM, MaskSR, Genhancer, LLaSE-G1); what is new here is only the FastConformer frame-matched, single-pass NAR variant. Its decoded mel is no better than a per-frame MLP and loses to spectral subtraction (`VERIFICATION.md` §3.2). |
| **SALM bridge** (`train-salm`) | reproduction+ | FastConformer → frame-stacking projector → small causal LM, pre-trained on text and then fine-tuned together with the encoder (stands in for Canary-Qwen's LoRA-adapted LLM) |

Results are in the README (they come from laptop-scale synthetic runs, not benchmarks). All 8
designs learn their tasks, except that the voice-agent EOU head is still below its always-"no EOU"
baseline (88% vs 92%). Getting there took six fixes, most of which mirror an NVIDIA practice:
CTC alignment before AED, pretrained encoders for SALM plus LLM adaptation, frame-local
(non-utterance) feature normalization for streaming (NVIDIA uses none; we use fixed global stats),
balancing gradients into a shared encoder, reading speaker information from a learned weighted mix
of all encoder layers (as in NEST), and normalizing latents before FSQ. The recipe only works if you keep all of it, including the training practices.

## 4. What is *not* replicated

- **Scale and data.** NVIDIA trains 0.1–1B-parameter encoders on up to 1.7M hours. Our recipes are
  2.3–2.8M-parameter models trained on a synthetic "tone language" so the whole pipeline can be
  checked in minutes. The production sizes are noted in each recipe.
- **Positional encoding.** We use rotary embeddings, where NeMo uses Transformer-XL relative
  positions. We also use LayerNorm in the conv module by default (BatchNorm is available).
- **Deployment and special training features.** Streaming Sortformer's AOSC speaker cache,
  TensorRT and Riva deployment, LoRA inside the LLM, the SLM discriminator for codecs, and
  CUDA-graph TDT decoding are not included.
- **Model families outside the shared recipe.** Audio Flamingo (Whisper encoder), BigVGAN, the
  Mamba enhancers and Audio2Face are not part of the FastConformer recipe. Only the codec/GAN
  pattern is reproduced.
