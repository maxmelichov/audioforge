# Is a unified streaming voice-agent front-end a game changer? Landscape check (2026-09-25)

*A stress test of the thesis in `ANALYSIS.md` §3. Every claim about a product or model has a URL. "(unverified)" marks anything
we could not confirm from a primary source. The M5 numbers in §5 were measured on this machine with `audioforge`.*

## Executive summary
1. **Verdict: the gap is real but narrower than the thesis says. "Step change" cannot be defended as written.**
2. **Already shipped:** ASR+end-of-turn in one open model (NVIDIA `parakeet_realtime_eou_120m`, Kyutai `stt-1b`, X2-Turn-4B); commercial fused end-of-turn (Deepgram Flux, Soniox); 0.32 s streaming diarization (Nemotron-3-Diarization, 100M, released 2026-09-23); streaming ASR+language ID (Nemotron 3.5); NVIDIA's own on-device runtime (NeMo-Speech.cpp: ggml, Metal/CPU).
3. **Not shipped by anyone:** one streaming model emitting words+VAD+end-of-turn+speaker+language, or end-of-turn that knows who is speaking. Flux has no diarization; Speechmatics' end-of-utterance carries no speaker label; AssemblyAI forces 640–768 ms turn silence once diarization is on. (VibeVoice-ASR-Streaming does speaker-attributed streaming ASR in one open model, but has no VAD, end-of-turn or small-encoder deployment; Alibaba UAF fuses VAD+turn+target-speaker ASR but has no released weights.)
4. **Weakest claim: "80–160 ms instead of 500 ms+".** NVIDIA's 160 ms P50 was measured on TTS audio with 3 s of trailing silence and reports no false-cutoff rate. On real turns (LiveKit eot-bench) at 5% false cutoffs, the best system (LiveKit's hosted Turn Detector v1) needs 543 ms of dead air, JoinIn AI Baton 577 ms, the best STT API (Soniox) 647 ms, plain VAD 1,600 ms. The 80 ms clock is a floor; the false-cutoff trade-off sets the latency.
5. **Sharpest defensible claim:** an open (CC-BY/Apache) ≤120M cache-aware model, one encoder pass on one CPU core, parakeet_realtime_eou-level WER, plus *speaker-aware* end-of-turn/barge-in (primary user, ignore bystanders/backchannels) that matches or beats the best end-of-turn detector on eot-bench (LiveKit Turn Detector v1, ~543 ms dead air @5% false cutoffs, 295 ms @10%) *while also being speaker-aware*, i.e. ≤540 ms dead air @5% false cutoffs (≤290 ms @10%), which also beats JoinIn AI Baton (577/350 ms) and Soniox (647/512 ms). (Corrected 2026-09-25: this used to say "≤600 ms … beating every commercial API tested there"; LiveKit v1 and JoinIn AI Baton, both hosted, already beat 600 ms, see `VERIFICATION.md` C22.)
6. **Credibility bar:** LibriSpeech test-other ≤7% @160 ms (EOU-120M 7.79); AMI-IHM ≤15.62 and Earnings-22 ≤15.76 (no worse than EOU-120M @160 ms, per Bar A); DER ≤ Sortformer-v2 @0.32 s without post-processing (DIHARD3 20.19, CALLHOME-p2 13.57); VoxCeleb1-O EER ≤1.5% (TitaNet-L 0.66); full-pipeline RTF ≤0.3 on one CPU core.
7. **Legal data:** LibriSpeech, MLS, People's Speech (non-SA), Common Voice, VoxPopuli, Granary manifests (YODAS part CC-BY-3.0), AMI/ICSI/NOTSOFAR, own MUSAN/RIR mixtures. Paid: Fisher/CALLHOME/DIHARD (LDC). Avoid (NC/risky): VoxCeleb, Emilia, WHAM!, GigaSpeech, SPGISpeech.
8. **Teachers:** CC-BY-4.0 Parakeet v2/v3, Canary-1B-v2, Sortformer-v2, TitaNet-L; OpenMDW Nemotron 3.5 and Nemotron-3-Diarization; NVIDIA Open Model License outputs are "not a Derivative Model". Avoid NC Sortformer-v1/Canary-1B and LiveKit (no training on outputs).
9. **Compute is cheap:** 120M × 10k h × 30 epochs ≈ 1.3e19 FLOPs ≈ 44 A100-h FLOP-bound ($61–70 at RunPod $1.39–1.59/h; ~$180–350/run realistic at ×3–5); 600M × 100k h × 10 epochs ≈ 2.1e20 ≈ 730 A100-h ($1,020–1,160). This M5 trains a 105M encoder at ~4,500 audio-h/day (MPS, no data loading) and streams one 80 ms encoder step in ~18–26 ms on one CPU thread (18 ms measured once; 26 ms median under load on the shared machine).
10. **Biggest risk: NVIDIA ships it.** It has the EOU model, language ID, an 8-speaker streaming diarizer and a local runtime on the same 80 ms clock. Move fast; compete on speaker-aware turn-taking, license, and honest evaluation.

## 1. Competitors and prior art

"Fused" means one network produces both outputs. "Cascade" means separate models are chained.

| System | Tasks | Streaming latency claim | Size | On-device | License | Source |
|---|---|---|---|---|---|---|
| Silero VAD v6.2.3 | VAD | <1 ms per 30 ms+ chunk on 1 CPU thread | ~2 MB | yes (ONNX/CPU) | MIT | https://github.com/snakers4/silero-vad |
| TEN VAD | VAD | RTF 0.016 on M1 | 306–731 KB library | yes (incl. iOS/Android/Web) | Apache-2.0 + Agora non-compete | https://github.com/TEN-framework/ten-vad |
| NVIDIA Frame VAD MarbleNet v2.0 | VAD (20 ms frames) | none published | 91.5K | ONNX on CPU | NVIDIA Open Model License | https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0 |
| pyannote 3.1 / community-1 | diarization + embeddings | offline only | n/a | CPU/MPS; Core ML port in FluidAudio | MIT / CC-BY-4.0 (gated) | https://huggingface.co/pyannote/speaker-diarization-community-1 |
| pyannoteAI Precision-3 / Live-1 | diarization; STT orchestration (batch only) | Live-1 "under 300ms"; no streaming DER published | closed | on-prem | commercial (€0.198/h for Live-1) | https://www.pyannote.ai/streaming |
| LiveKit turn detector: text v0.4.1; audio v1 / v1-mini (2026-06) | end-of-turn (text, or audio+text) | text 50–160 ms per turn; audio v1 fires after 543 ms of silence at 5% false cutoffs (eot-bench) | text model 396 MB (q8); v1-mini size not disclosed | v1-mini on CPU | LiveKit Model License: only with LiveKit Agents; no training on outputs | https://livekit.com/blog/solving-end-of-turn-detection |
| Pipecat Smart Turn v3.2 | end-of-turn (audio only) | 12.6 ms on CPU; triggered after a 200 ms VAD silence | 8M params (8.7 MB int8) | yes (CPU) | BSD-2-Clause | https://huggingface.co/pipecat-ai/smart-turn-v3 |
| Krisp VIVA turn v3 | end-of-turn (audio) | "69% of true turn-shifts … within 200 ms of silence" | ~9M | CPU SDK | closed | https://krisp.ai/blog/viva-2-0-ai-infrastructure-for-voice-ai-agents/ |
| Vogent-Turn-80M / UltraVAD / Easy Turn | end-of-turn (audio + text context) | ~7 ms on T4 / 65–110 ms on A6000 / 263 ms on RTX 4090 | 80M / 0.7B–8B (card conflicts; unverified) / 850 MB | GPU | modified Apache / none stated (unverified) / Apache-2.0 | https://huggingface.co/vogent/Vogent-Turn-80M , https://huggingface.co/fixie-ai/ultraVAD , https://github.com/ASLP-lab/Easy-Turn |
| **NVIDIA parakeet_realtime_eou_120m-v1** | **ASR + end-of-turn, fused** (English, no punctuation) | 80–160 ms; end-of-turn P50/P90/P95 = 160/280/320 ms, measured on TTS audio with 3 s of silence appended | 120M | GPU per card; Core ML port reaches RTFx 19 at 320 ms chunks on M2 | NVIDIA Open Model License | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| **Kyutai stt-1b-en_fr** | **ASR + "semantic VAD" end-of-turn, fused** (English/French) | 80 ms steps, 0.5 s text delay | ~1B | yes (MLX on Mac/iPhone, Rust) | CC-BY-4.0 | https://kyutai.org/stt/ , https://github.com/kyutai-labs/delayed-streams-modeling |
| Kyutai stt-2.6b / Moshi / Unmute | ASR / full-duplex speech LLM / cascade built on stt-1b (Unmute details unverified) | 2.5 s delay / 160 ms theoretical | 2.6B / 7B | MLX | CC-BY-4.0 | https://github.com/kyutai-labs/moshi |
| **X2-Turn-4B** (Aug 2026) | **ASR + 6-state turn head, fused** (Chinese/English) | 80 ms frame-synchronous | ~4B; needs ≥24 GB VRAM | no | Apache-2.0 | https://github.com/X-Square-Robot/X2-Turn |
| Pine AI turn-aware ASR | ASR + end-of-turn token, fused (LoRA on Qwen3-ASR-0.6B) | 0.39 s median; recall 0.97 | 0.9B | – | code only, **no weights** | https://arxiv.org/pdf/2609.04225 |
| whisper.cpp / faster-whisper / WhisperLive | ASR (+ Silero VAD; tinydiarize emits a speaker-turn token only) | 0.5 s step in whisper-stream; faster-whisper does not stream natively | 39M–1.55B | yes (Metal / Core ML) | MIT | https://github.com/ggml-org/whisper.cpp , https://github.com/SYSTRAN/faster-whisper , https://github.com/collabora/WhisperLive |
| WhisperKit + SpeakerKit (Argmax) | ASR + separate diarizer (cascade) | 0.46 s at 2.2% WER on M3 Max | – | Apple | MIT | https://arxiv.org/abs/2507.10860 , https://github.com/argmaxinc/argmax-oss-swift |
| Moonshine v2 streaming | ASR only (separate VAD segmenter) | 18 / 38 / 59 ms after phrase end on MacBook Pro | 34M / 123M / 245M | yes (incl. Raspberry Pi, phones) | MIT | https://arxiv.org/abs/2602.12241 , https://github.com/moonshine-ai/moonshine |
| k2 / sherpa-onnx streaming Zipformer | ASR; VAD, speaker-ID and offline diarization are separate models | 320 ms chunks: LibriSpeech test-other 7.79 (66M, trained on LibriSpeech only) | 20–66M | yes (Raspberry Pi, Jetson, mobile) | Apache-2.0 | https://github.com/k2-fsa/sherpa-onnx , https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md |
| NVIDIA Nemotron ASR streaming (English 0.6B; 3.5 multilingual) | ASR (+ language tag with `target_lang=auto` in 3.5) | 80–1120 ms chunks; 240 streams per H100 at 80 ms | 600M | NeMo-Speech.cpp (GGUF on CPU/Metal) | NVIDIA Open Model License / OpenMDW-1.1 | https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b |
| NVIDIA Sortformer v2 / v2.1 / Nemotron-3-Diarization | streaming diarization (4 / 4 / 8 speakers) | 0.32 s recommended minimum (80 ms possible) | 117M / 117M / 100M | NeMo-Speech.cpp | CC-BY-4.0 / NVIDIA Open Model License / OpenMDW-1.1 | https://huggingface.co/nvidia/Nemotron-3-Diarization |
| NVIDIA multitalker-parakeet-streaming 0.6B | speaker-attributed ASR; needs Sortformer (2 models, one ASR instance per speaker) | 1.12 s | 600M + 117M | GPU | NVIDIA Open Model License | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 |
| NVIDIA Riva / Speech NIM / NeMo-Speech.cpp | cascade: ASR + Silero VAD + silence endpointing (`endpointing_ms`) + Sortformer | speaker labels "can take up to 10 s" to settle (Sortformer V2) | – | Jetson Thor; Metal/CPU (NeMo-Speech.cpp) | Riva commercial; NeMo-Speech.cpp Apache-2.0 | https://github.com/NVIDIA/NeMo-Speech.cpp , https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html |
| NVIDIA TitaNet-L | speaker embedding | – | 23M | yes | CC-BY-4.0 | https://huggingface.co/nvidia/speakerverification_en_titanet_large |
| Ultravox | speech LLM; no turn detection (UltraVAD is a separate model) | – | 1B–70B LLM backbone | GPU | MIT | https://github.com/fixie-ai/ultravox |
| Mistral Voxtral Realtime | ASR only | 80–2400 ms delay | 4.4B | ≥16 GB GPU | Apache-2.0 | https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 |
| Qwen3-ASR 0.6B / 1.7B | ASR + language ID (30 languages) | streaming in 2 s chunks (vLLM backend only) | 0.9B / 1.7B | GPU | Apache-2.0 | https://github.com/QwenLM/Qwen3-ASR |
| SenseVoice-Small / VibeVoice-ASR | ASR + language ID + emotion + events / ASR + diarization + timestamps | **not streaming** (either) | 234M / ~9B | – | FunASR license / MIT | https://huggingface.co/microsoft/VibeVoice-ASR |
| **Microsoft VibeVoice-ASR-Streaming** (2026-09) | **speaker-attributed streaming ASR, one model** (LLM-based); no VAD or end-of-turn | streaming (latency not checked here) | 1.5B / 7B | – | MIT | https://arxiv.org/abs/2609.02812 |
| JoinIn AI Baton | end-of-turn (commercial API) | 577 ms at 5% false cutoffs, 350 ms at 10% (eot-bench) | – | no (hosted) | commercial | https://github.com/livekit/eot-bench |
| Alibaba UAF (research) | VAD + turn-taking + target-speaker ASR in one model | 600 ms chunks | 30B-A3B | – | weights not released (unverified) | https://arxiv.org/html/2604.19221v2 |
| Deepgram Flux (English + multilingual) | **ASR + end-of-turn, fused** (vendor: "single model"); **no diarization** | ~260 ms median end-of-turn; 1,151 ms at 5% false cutoffs (eot-bench) | – | self-host containers | $0.0065–0.0078/min | https://deepgram.com/learn/introducing-flux-conversational-speech-recognition |
| Deepgram Nova-3 | ASR + streaming diarization; VAD endpointing | – | – | self-host | $0.0048–0.0092/min + $0.002/min for diarization | https://developers.deepgram.com/docs/diarization |
| AssemblyAI Universal-Streaming / U3.5 Pro RT | ASR + turn detection (silence gate, then the ASR judges the transcript complete) + streaming diarization | "around 300ms"; with diarization, turn silence is forced to 640–768 ms | – | self-host (24 GB GPU) | $0.15–0.45/h + $0.12/h for diarization | https://www.assemblyai.com/docs/streaming/universal-3-pro/turn-detection-and-partials |
| Speechmatics RT / Linden-1 | ASR + real-time diarization + speaker ID; end-of-turn from silence, or Smart Turn v3 in the SDK | ~350 ms finalization | on-device build ~800 MB | Mac / Windows | $0.16–0.43/h | https://docs.speechmatics.com/speech-to-text/realtime/turn-detection , https://www.speechmatics.com/speech-to-text/on-device |
| Soniox stt-rt-v5 | ASR + semantic `<end>` token + diarization + per-token language ID | 647 ms at 5% false cutoffs (best STT API on eot-bench) | – | cloud (on-prem unverified) | $0.12/h, everything included | https://soniox.com/docs/stt/rt/endpoint-detection |
| OpenAI Realtime | `server_vad` (500 ms default) or a separate `semantic_vad`; no streaming speaker labels | 1,143 ms at 5% false cutoffs (GPT Realtime 2) | – | no | per audio token | https://developers.openai.com/api/reference/resources/realtime/server-events |
| Google Gemini Live | VAD-style activity detection (~800 ms default); no diarization documented | – | – | no | – | https://ai.google.dev/gemini-api/docs/live-guide |
| ElevenLabs Scribe v2 Realtime / Cartesia Ink-2 / Gradium | ASR (no real-time diarization) / ASR + turn events "signaled directly by the model" / ASR + multi-horizon VAD | ~150 ms / 1,056 ms at 5% false cutoffs / 913 ms at 5% false cutoffs | – | no | $0.28/h / $0.39/h / closed | https://elevenlabs.io/realtime-speech-to-text , https://www.cartesia.ai/blog/ink-2 , https://github.com/livekit/eot-bench |

**End-of-turn fused with ASR in one model already ships:**
- Open: NVIDIA parakeet_realtime_eou (120M), Kyutai stt-1b (1B), X2-Turn (4B).
- Commercial: Flux, Soniox and Ink-2, according to their vendors.

**Speaker-attributed streaming ASR in one open model now exists, but without turn signals.** Microsoft's VibeVoice-ASR-Streaming (1.5B/7B, MIT, released 2026-09, arXiv 2609.02812) streams who-said-what from one LLM-based model; it emits no VAD or end-of-turn. (Corrected 2026-09-25: this heading used to say such a model "does not exist", see `VERIFICATION.md` C25.)
- NVIDIA's version needs Sortformer (Nemotron-3-Diarization) plus multitalker-Parakeet or Nemotron 3.5 ASR, run as one ASR instance per speaker.
  https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md
- Commercially it exists as a feature (Speechmatics, AssemblyAI, Nova-3, Soniox, AWS, Azure). None of these tie diarization to their end-of-turn.
- **No open model emits all five signals from one streaming encoder.**

## 2. Verdict

**Already solved, so not our differentiator:** (1) ASR+end-of-turn in one network (NVIDIA, 120M, cache-aware, Oct 2025); (2) tiny VAD
(Silero 2 MB, <1 ms/chunk); (3) streaming diarization (Nemotron-3-Diarization DIHARD3 13.55 vs Sortformer-v2.1 19.85 @0.32 s, same references; a large
gap for a first multi-task 120M student to close); (4) streaming language ID (Nemotron 3.5 tag); (5) on-device runtimes (NeMo-Speech.cpp, FluidAudio, sherpa-onnx,
Moonshine). **Compute saving depends on the latency operating point, not on counting 4 encoders:** NVIDIA already fuses ASR and
end-of-utterance in one 120M cache-aware model (parakeet_realtime_eou_120m-v1), and the VAD is a 91.5K CNN, so the cascade has two real
encoders: ASR+EOU (120M) and Streaming Sortformer (117M), plus an optional TitaNet (23M) for enrolled speaker ID. Streaming Sortformer re-runs
its full 17-layer FastConformer and 18-layer Transformer over [speaker cache; FIFO; chunk] at every step. At the 0.32 s config (chunk 3, right
context 1, FIFO 188, cache 188) that is about 127 encoder passes per frame (about 177 for Nemotron-3-Diarization, cache/FIFO 264), and about
65 at 1.04 s, versus about 1.8 at the 30.4 s offline-style config. The card's RTF (0.180 vs 0.002) shows the same ~90× gap. So a single
cache-aware pass saves about 2× encoder compute only at ≥10 s buffers, and one to two orders of magnitude at 0.32-1 s latency. That saving
holds only if the fused model matches diarization with its own long-range speaker memory (for example, an arrival-order cache of encoder
states the diarization head attends to), which costs extra per-frame attention and is not yet demonstrated. (Corrected 2026-09-25: this
used to say merging saves "about 2× on the encoder, not 4×", see `VERIFICATION.md` C26.)

**Where the thesis overclaims.** "End-of-turn in 80–160 ms instead of 500 ms+" confuses the frame clock with the operating point. Nobody has
shown 80–160 ms dead air at an acceptable false-cutoff rate on real speech: NVIDIA's 160 ms P50 is on TTS audio with 3 s of trailing silence
and no false-cutoff rate, and a third-party test (Pine AI, arXiv 2609.04225, Fig. 2, as-shipped) measured Parakeet-EOU at recall 0.17
(https://arxiv.org/pdf/2609.04225). On eot-bench (real human-to-agent turns; latency = dead air after the true end) the best system (LiveKit v1)
needs 543 ms at 5% false cutoffs (295 ms at 10%), JoinIn AI Baton 577 ms (350 ms); Flux needs 1,151 ms and Smart Turn v3.2 1,051 ms (https://github.com/livekit/eot-bench). Humans start replying a median 151 ms *before*
the turn ends (TurnBench, https://arxiv.org/abs/2608.25218, abstract only), so ~150 ms needs semantic prediction, not a faster clock. The
"500 ms+ timeout" premise is true for silence-only defaults (OpenAI `server_vad` 500 ms, Gemini ~800 ms, Azure 500 ms) but out of date for
state-of-the-art stacks: LiveKit's turn model with a 0.3 s minimum delay (https://docs.livekit.io/reference/agents/turn-handling-options/),
Pipecat's 0.2 s VAD stop plus Smart Turn (https://docs.pipecat.ai/pipecat/learn/speech-input).

**Where the gap is real** (most defensible first):
1. **Speaker-aware turn-taking.** No product fuses end-of-turn with speaker identity: Speechmatics' end-of-utterance "will not contain speaker
   information" (https://docs.speechmatics.com/speech-to-text/realtime/turn-detection); Flux has no diarization
   (https://developers.deepgram.com/docs/flux/flux-nova-3-comparison); AssemblyAI slows turns to 640/768 ms with diarization. Rejecting
   bystanders/TV/backchannels is bolted on (AssemblyAI Voice Focus, https://www.assemblyai.com/docs/streaming/voice-focus; Krisp background-voice
   cancellation, https://krisp.ai/blog/improving-turn-taking-of-ai-voice-agents-with-background-voice-cancellation/). A shared encoder with
   speaker+VAD+end-of-turn heads gives "the enrolled user finished / barged in" on one clock. That output is novel.
2. **Open license + edge + fusion.** The fused open models are OML-licensed and GPU-only per card (parakeet_realtime_eou, absent from
   NeMo-Speech.cpp), 1B with 0.5 s delay (Kyutai), or 4B (X2-Turn). LiveKit, TEN and Vogent weights carry use restrictions.
3. **One aligned event stream**: words, VAD, end-of-turn and speaker events share one timebase with no cascade skew; NVIDIA's local runtime
   needs up to 10 s to confirm a speaker label.

**Sharpest defensible thesis:**
> *"An open-weights (CC-BY/Apache), ≤120M, cache-aware model that in one encoder pass on one CPU core gives English streaming ASR at
> parakeet_realtime_eou-level WER **plus speaker-aware turn events** (primary-user end-of-turn, user barge-in vs bystander/backchannel),
> matching or beating the best end-of-turn detector on eot-bench (LiveKit Turn Detector v1, ~543 ms) at ≤540 ms dead air @5% false
> cutoffs (≤290 ms @10%) *while also being speaker-aware* — ahead of the hosted LiveKit Turn Detector v1 (543/295 ms), JoinIn AI Baton
> (577/350 ms) and Soniox (647/512 ms) — while also emitting VAD, speaker-change and language on the same 80 ms clock."*

(Corrected 2026-09-25: the thesis used to say "≤600 ms … better than every commercial STT API measured there". ≤600 ms beats Soniox
(647 ms) and every other STT API tested, but still trails the hosted end-of-turn models LiveKit Turn Detector v1 (543 ms) and JoinIn AI
Baton (577 ms); see `VERIFICATION.md` C22.)

That is incremental on latency, novel on speaker-aware turn-taking, openness and edge deployment, and not a step change in ASR.

## 3. Target metrics for a credible result

Evaluate with pinned revisions. The Open ASR Leaderboard changed normalizer, GPU and test splits in June–Aug 2026, so numbers from before and after are not comparable (https://huggingface.co/spaces/hf-audio/open_asr_leaderboard/blob/main/init.py).

**Gates:**
- **Bar A:** ASR not worse than parakeet_realtime_eou at the same chunk size.
- **Bar B:** end-of-turn beats Smart Turn v3.2 + Silero on eot-bench, and speaker-aware end-of-turn beats a cascade of Sortformer-v2 plus our own end-of-turn.

| Metric | Public baselines (source) | Our target (120M, 160 ms) |
|---|---|---|
| LibriSpeech test-other WER | EOU-120M 7.79 at 160 ms (https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1); streaming_multi 114M 6.4 at [70,1] (https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi); Nemotron-0.6B 5.57 at 0.16 s (https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b); parakeet-tdt_ctc-110m 5.22 offline; tdt-0.6b-v3 3.59 offline (https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/7c92e785bb0d/english_shortform_results.csv); whisper-small 7.6 (https://cdn.openai.com/papers/whisper.pdf) | ≤7.0 |
| In-the-wild WER: AMI-IHM / Earnings-22 | EOU-120M 15.62 / 15.76 (https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1); tdt_ctc-110m 15.89 / 12.37; tdt-0.6b-v3 11.39 / 11.19 (leaderboard CSV above); whisper-small AMI-IHM 19.0, AMI-SDM 39.6, CHiME-6 29.3, Earnings-22 14.3 (Whisper paper) | ≤15.62 / ≤15.76 (no worse than EOU-120M, per Bar A); also report AMI-SDM and CHiME-6 |
| End-of-turn on eot-bench (English): dead air at 5% / 10% false cutoffs | LiveKit v1 543 / 295 ms; JoinIn AI Baton 577 / 350; Soniox 647 / 512; Smart Turn v3.2 1,051 / 739; Flux 1,151 / 548; VAD baseline 1,600 / 1,000 (https://github.com/livekit/eot-bench). Latency = dead air after the true end of turn; the false-cutoff rate is the share of mid-turn pauses of ≥100 ms where the model fires. | ≤540 / ≤290 ms (match or beat LiveKit v1 while speaker-aware; ≤600 / ≤350 would beat Soniox but trail LiveKit v1 and Baton) |
| End-of-turn, secondary sets | Smart Turn v3.2: 92.63% accuracy, 4.73% false positives on a 31,527-clip, 23-language test set (https://huggingface.co/pipecat-ai/smart-turn-v3/blob/main/benchmarks/smart-turn-v3.2-cpu.md); LiveKit text model: TNR 87.0% at a fixed 99.3% TPR (https://huggingface.co/livekit/turn-detector); Pine AI: recall vs false fires per speech-minute (https://arxiv.org/pdf/2609.04225) | ≥ Smart Turn on its own test set; report false fires per minute |
| Speaker-aware end-of-turn (new) | no public baseline. Build one: eot-bench turns with an overlaid bystander voice or TV from permissively licensed audio, scoring false cutoffs and false barge-ins | publish the set and the harness |
| DER (overlap scored; collar 0 s, or 0.25 s for CALLHOME) | at 0.32 s: Sortformer-v2 DIHARD3 20.19, CALLHOME-p2 13.57 (https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2); Nemotron-3-Diarization DIHARD3 13.55 / CALLHOME-p2 11.32 (original references, same protocol as Sortformer-v2.1's 19.85 / 12.67 on the same card) / AMI-SDM 12.95 (forced-aligned references) (https://huggingface.co/nvidia/Nemotron-3-Diarization); offline pyannote community-1 20.2 / 26.7 / AMI-SDM 19.9 (https://huggingface.co/pyannote/speaker-diarization-community-1) | ≤ Sortformer-v2 at 0.32 s |
| Speaker-attributed cpWER | multitalker-parakeet + Sortformer-v2 at 1.12 s: AMI-IHM 21.26, CH109 15.81 (https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) | ≤25 on AMI-IHM with one model |
| Speaker verification EER, VoxCeleb1-O (cleaned) | TitaNet-L 0.66% (HF card); WeSpeaker ResNet34 0.867%; ECAPA 0.80% (https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md , https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | ≤1.5% from the shared encoder |
| Language ID | Qwen3-ASR-0.6B 96.8% average (https://github.com/QwenLM/Qwen3-ASR); Nemotron 3.5 English FLEURS WER 9.72 with auto language vs 9.43 with the language given, at 80 ms | ≥95% on FLEURS for our languages |
| RTF / latency on device | EOU-120M on M2 via Core ML: RTFx 19.25 at 320 ms, 5.78 at 160 ms; Nemotron-0.6B on M5 Pro: RTFx 40.7 at 560 ms (https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md); parakeet-tdt-v3 on x86 CPU: RTFx 35.4, on Cortex-A53 int8: 1.0 (https://istupakov.github.io/onnx-asr/benchmarks/); Jetson Thor Nemotron+Sortformer: p50 15.14 ms (https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-performance.html); no public Parakeet-on-Orin row | full pipeline (mel + encoder + all heads): RTF ≤0.3 on 1 M-series CPU core, ≤0.5 on Jetson Orin Nano |

## 4. Data plan

Every source below has its license checked at the source. **NC** = non-commercial.

| Use | Source | License / terms | Link |
|---|---|---|---|
| ASR core | LibriSpeech 960 h; MLS | CC-BY-4.0 | https://www.openslr.org/12/ , https://www.openslr.org/94/ |
| ASR core | LibriHeavy (50k h, labeled Libri-Light) | Apache-2.0 | https://github.com/k2-fsa/libriheavy |
| ASR core | People's Speech (30k h+) | `clean` / `dirty` subsets are CC-BY; **skip `_sa` (CC-BY-SA)** | https://huggingface.co/datasets/MLCommons/peoples_speech |
| ASR + language ID | Common Voice | CC0; now distributed via Mozilla Data Collective (its extra terms unverified) | https://datacollective.mozillafoundation.org |
| ASR + language ID | VoxPopuli | CC0 | https://huggingface.co/datasets/facebook/voxpopuli |
| ASR, large scale | NVIDIA Granary (~643k h of ASR manifests, 25 languages; manifests only, no audio) | card says CC-BY-4.0, but **YODAS-Granary is CC-BY-3.0**; audio follows its source licenses | https://huggingface.co/datasets/nvidia/Granary |
| ASR, large scale | YODAS / YODAS2; YouTube-Commons; MOSEL | CC-BY-3.0 (creators keep copyright); CC-BY-4.0; CC-BY-4.0 | https://huggingface.co/datasets/espnet/yodas2 , https://huggingface.co/datasets/FBK-MT/mosel |
| ASR / TTS source | **Emilia-YODAS is CC-BY-4.0; Emilia is NC (avoid)** | as stated in the gated terms | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| Diarization, meetings | AMI; ICSI; NOTSOFAR-1 | CC-BY-4.0 (all three) | https://groups.inf.ed.ac.uk/ami/corpus/license.shtml , https://github.com/microsoft/NOTSOFAR1-Challenge |
| Diarization, telephone | CALLHOME = 2000 NIST SRE (LDC2001S97); Fisher (LDC2004S13 / LDC2005S13); Switchboard (LDC97S62); DIHARD III (LDC2022S12 / S14) | **paid LDC user agreement** (fees unverified) | https://catalog.ldc.upenn.edu/LDC2001S97 , https://catalog.ldc.upenn.edu/LDC2004S13 |
| Far-field eval | CHiME-6 | CC-BY-SA-4.0 (use for eval only) | https://www.openslr.org/150/ |
| Overlap mixtures | our own mixtures from LibriSpeech/MLS + MUSAN (CC-BY-4.0) + RIRS_NOISES (Apache-2.0), using FastMSS-style simulation. **Do not use LibriMix noisy: WHAM! noise is NC.** | as listed | https://www.openslr.org/17/ , https://www.openslr.org/28/ , http://wham.whisper.ai/ |
| Speaker ID | CN-Celeb (CC-BY-SA-4.0); speaker labels from LibriSpeech/MLS/Common Voice. **VoxCeleb:** metadata is CC-BY-SA and the audio is no longer distributed (YouTube rights), so use it for eval only, legally risky. **VoxBlink2 is NC.** | as listed | https://www.openslr.org/82/ , https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html , https://voxblink2.github.io/ |
| End-of-turn | Smart Turn training data (weights and code BSD-2; the dataset's own license unverified); DialogStudio commercial subset re-voiced with TTS (NVIDIA's EOU recipe); force-aligned pauses in Fisher/AMI as hold / end-of-turn labels | check each | https://github.com/pipecat-ai/smart-turn , https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| End-of-turn eval | eot-bench (harness Apache-2.0; the data's license unverified) | eval only | https://huggingface.co/datasets/livekit/eot-bench-data |
| Avoid for commercial use | GigaSpeech (non-commercial research only), SPGISpeech (academic / internal use), Earnings-22 (CC-BY-SA; eval only) | – | https://huggingface.co/datasets/speechcolab/gigaspeech , https://huggingface.co/datasets/kensho/spgispeech |

**Teacher distillation (pseudo-labels).** From the card license fields (https://huggingface.co/nvidia/<model>):
- **CC-BY-4.0, free with attribution:** parakeet-tdt-0.6b-v2/v3, parakeet-tdt_ctc-110m, canary-1b-v2, stt_en_fastconformer_hybrid_large_streaming_multi, diar_streaming_sortformer_4spk-**v2**, TitaNet-L.
- **OpenMDW-1.1, which puts no restriction on outputs:** nemotron-3.5-asr-streaming-0.6b, Nemotron-3-Diarization (https://openmdw.ai/license/1-1/).
- **NVIDIA Open Model License:** nemotron-speech-streaming-en, parakeet_realtime_eou, Sortformer **v2.1**, multitalker, MarbleNet.
  - The license says "An output is not a Derivative Model" (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  - So labels are fine. Initializing from those weights does create a Derivative Model, which keeps the license's obligations. (Our reading, not legal advice.)
- **Avoid:** diar_sortformer_4spk-**v1** and canary-1b (both CC-BY-NC-4.0), and LiveKit (no training on outputs).
- **Labeling cost:**
  - parakeet-tdt-0.6b-v3 runs at RTFx 3,333 on an A100 (leaderboard CSV), so 100k h ≈ 30 A100-hours.
  - Nemotron-3-Diarization offline runs at RTFx 1,340 at batch 1 on an RTX PRO 5000 (its card), so 100k h ≈ 75 GPU-hours.

## 5. Compute estimate

**Arithmetic.**
- 16 kHz audio → 100 Hz log-mel → 8× subsampling = **12.5 Hz = 45,000 encoder frames ("tokens") per audio hour**, i.e. 45M / 450M / 4.5B frames per epoch for 1k / 10k / 100k h.
- Forward ≈ **2N FLOPs per frame**. Limited-context attention (d=512, 70-frame left context, 17 layers) adds ~4·80·512·17 ≈ 2.8 MFLOP/frame (~1% at 120M); subsampling convs ~0.1 GFLOP per audio-second vs 3 GFLOP for the encoder.
- Training ≈ **6N per frame × 1.3** (TDT joint T×U×V logits, CTC, aux heads). Per audio-hour per epoch: **30M 1.05e13; 120M 4.2e13; 600M 2.1e14 FLOPs**. Inference 2N×12.5 = **0.75 / 3.0 / 15 GFLOP per audio-second**, so one CPU core is enough at 120M.
- Epochs: icefall Zipformer on LibriSpeech 960 h uses 30–50 (174 for the large model); NVIDIA streaming_multi "over several hundred epochs"; Canary-1B saw 150k steps × 360 s/GPU × 128 A100 = 1.92M audio-h on 85k h ≈ 23 epochs (https://huggingface.co/nvidia/canary-1b). **Plan: 1k h × 100, 10k h × 30, 100k h × 10.**
- Throughput: A100 312 dense BF16 TFLOPS (https://www.nvidia.com/en-us/data-center/a100/), assume **80 TFLOP/s effective (~25% MFU)**; H100 SXM 1,979 with sparsity ≈ 989 dense (https://www.nvidia.com/en-us/data-center/h100/), assume **200 TFLOP/s**. RunPod on-demand: A100 SXM $1.39–1.59/h, H100 SXM $2.69–3.49/h (https://www.runpod.io/pricing).

**Ideal cost: one run, FLOP-bound.** Cells: total FLOPs → A100-hours (cost); H100-hours for 100k h.

| Student | 1k h × 100 ep (100k h seen) | 10k h × 30 ep (300k h seen) | 100k h × 10 ep (1M h seen) |
|---|---|---|---|
| 30M | 1.1e18 → 4 A100-h ($5) | 3.2e18 → 11 A100-h ($15) | 1.1e19 → 37 A100-h ($50) / 15 H100-h |
| 120M | 4.2e18 → 15 A100-h ($20) | 1.3e19 → 44 A100-h ($61–70) | 4.2e19 → 146 A100-h ($203–233) / 58 H100-h |
| 600M | 2.1e19 → 73 A100-h ($90–115) | 6.3e19 → 219 A100-h ($305–350) | 2.1e20 → 731 A100-h ($1,016–1,163) / 292 H100-h ($790–1,020) |

**Realistic budget:** ×3–5 for data loading, padding, eval, restarts and low MFU at 30M (a 30M model on an A100 must decode ~7 audio-hours
per second, so it is dataloader-bound unless mel runs on the GPU), then ×5–10 runs for ablations (head weights, `grad_scale`, context sets).
That gives **~$1–3k for a 120M/10k h programme and ~$3–6k per flagship 600M/100k h run**, plus <$300 of pseudo-labeling (§4). Storage: 16-bit
16 kHz WAV is 115 MB/h → 115 GB / 1.15 TB / 11.5 TB for 1k / 10k / 100k h (FLAC ≈ half).

**Measured on this M5 (24 GB)** with `audioforge` FastConformer, fp32, PyTorch 2.14 MPS, shared machine (lower bounds; script
`scratchpad/landscape/bench_m5.py`, not in the repo):
- Training (fwd+bwd+AdamW, CTC head V=1024, random 10 s inputs, no data loading): 28.4M 423 audio-s/s (10,151 h/day); 104.8M 188 (4,520 h/day);
  583.6M 36.5 (875 h/day). That is ~0.9–1.6 effective TFLOP/s; expect about half with a TDT joint and real data loading.
- Streaming, encoder only, att_context [70,0]/[70,1]: 104.8M on 1 CPU thread 17.9 ms per 80 ms step (RTF 0.22; not reproduced by the 2026-09-25 verification pass, which measured ~26 ms median under load, see `VERIFICATION.md` C27), 24.5 ms per 160 ms (RTF 0.15);
  on MPS ~9 ms per step (launch-bound); 28.4M on 1 thread 9.8 ms per 80 ms. Four threads were *slower* (contention).

**First real-data milestone that fits on the M5.**
- **(A, recommended)** Start from NVIDIA's CC-BY-4.0 cache-aware `stt_en_fastconformer_hybrid_large_streaming_multi` (114M, 6.4 test-other
  @[70,1]). Add VAD/end-of-turn/speaker/language heads reading a learned all-layer mix. Fine-tune on ~2k h (LibriSpeech + AMI + NOTSOFAR + TTS
  turns + mixtures) × 5–10 epochs = 10–20k audio-h ≈ 4–9 days at 2–2.5k h/day; ~230 GB disk. **Prerequisite:** port NeMo relative-position
  attention into `audioforge` (it uses RoPE), or fine-tune in NeMo.
- **(B)** 30M from scratch on LibriSpeech 960 h × 50 epochs (48k h seen) ≈ 5–10 days; reference: icefall's ~20M streaming model reaches 9.79%
  test-other @320 ms after 30 epochs.
- Anything ≥10k h or 600M goes to rented GPUs (120M × 10k h ≈ 100 M5-days).

## 6. Risks
- **NVIDIA ships the same thing (high, near-term).** In 12 months it released fused end-of-turn ASR (Oct 2025), a streaming language tag
  (Jun 2026), an on-device runtime (NeMo-Speech.cpp, Jul/Aug 2026) and an 8-speaker streaming diarizer (2026-09-23), all at 80 ms frames.
  Merging heads is a small step for them. *Mitigation:* ship within months; differentiate on speaker-aware turn events, a permissive license
  (their newest are OML/OpenMDW), CPU-first release and a public eval harness.
- **Kyutai / Gradium / Deepgram / LiveKit move first.** Kyutai already fuses semantic VAD into ASR and Gradium sells it; LiveKit owns the only
  cross-vendor end-of-turn benchmark and ranks first; Flux could add diarization. *Mitigation:* the gap is **open + small + speaker-aware**,
  which none of them has.
- **Multi-task interference.** Our toy runs already needed per-head `grad_scale` and an all-layer mix for speaker (see `README.md`). Speaker
  invariance helps ASR but fights speaker ID; diarization wants ~90 s context, ASR short chunks. Expect a 0.5–1 point WER regression vs
  ASR-only (our estimate, unverified). *Mitigation:* freeze/low-LR the ASR encoder first, read speaker/diarization from a learned mix of all layers, give
  diarization a longer context.
- **Streaming vs accuracy.** Nemotron-0.6B goes 6.93 → 8.43 average WER from 1.12 s to 0.08 s chunks; Sortformer-v2 19.68 → 20.19 DER on
  DIHARD3 from 30.4 s to 0.32 s. Fast end-of-turn trades directly against false cutoffs, so report operating curves, never one median.
- **Evaluation credibility.** Vendors use their own end-of-turn test sets; eot-bench is vendor-authored; NVIDIA's end-of-turn latency is from
  TTS audio; the Open ASR Leaderboard changed in 2026. *Mitigation:* pinned leaderboard revisions; eot-bench + TurnBench + our own
  speaker-overlap set; release code, checkpoints and labels; re-run Smart Turn + Silero and NVIDIA's EOU model as baselines.
- **Licensing.** VoxCeleb, WHAM!, Emilia, GigaSpeech and SPGISpeech are unusable or risky commercially; LDC telephone data is paid; starting from
  OML weights inherits OML terms; YODAS audio is CC-BY-3.0 but creators keep copyright; LiveKit forbids training on outputs. *Mitigation:*
  start from CC-BY-4.0 checkpoints, label only with CC-BY / OpenMDW / OML teachers, keep a per-source license manifest in the pipeline.
