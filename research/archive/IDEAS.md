# IDEAS: tournament of new ideas for the unified, speaker-aware streaming front end

Generated 2026-09-26 by a workflow run (generate, then novelty critic, feasibility critic, judge, prototype). Context: research/archive/GAME_CHANGER.md §2 (speaker-aware turn-taking, fused single-pass front end), §3 (target metrics). Everything below comes from the critics' and prototyper's reports. Nothing was re-run for this file. Where the reports are truncated or unverified, this file says so.

## 1. Executive summary

- **32 ideas** were generated across 8 lanes: architecture, objectives/distillation, data, evaluation, systems, product, generation-side and cross-field.
- **1 survived both critics:** `architecture-2`, in-pass self-conditioning. It scored 14.5 with the judge and passed novelty and feasibility at 3/5 each, both narrowly. **31 were rejected.** 30 of those failed novelty, 7 failed feasibility, and 6 failed both.
- **Prototype, architecture-2: inconclusive.**
  - The machine-safety rule forbids training, so none of the quality criteria were run (EOT P50@5fc, WER with diarized activity, diar F1, long-form swap rate).
  - On the untrained 2.4M model, the mechanism works. Gradients reach every part. The causal diarizer is exactly streamable (difference 1.8e-7), and streaming matches offline (2.1e-6).
  - The whole pipeline takes **0.56x** the time of the two-pass design, which passes the ≤0.7x criterion.
  - The real risk is still untested. runs/archive/turn_ablation.json shows the two-pass model scoring P50@5fc = 400 ms with oracle activity, but it misses 94% of turn ends (P50 infinite) with diarized activity.
- **What to build next:**
  1. **architecture-2 trained arms A', B, C and B0.** This is the only surviving idea, and teacher forcing directly targets the 94%-miss failure caused by exposure to the model's own diar output.
  2. **A KV-cached causal diar head.** The streamed causal diar currently re-runs the whole prefix every chunk, so its cost is O(T²). Idea 1 needs this fixed before it can be deployed.
  3. **Slot batching (systems-4), as engineering only.** It is not novel, because NeMo already ships it. The critic measured B=4 at 80 ms against B=1 at 49 ms per 160 ms step: a cheap win for multi-speaker RTF.

## 2. Prototyped ideas

### architecture-2: In-pass self-conditioning. A mid-layer causal diarizer steers the upper layers' speaker kernels in the same forward pass.

**One-liner.** Today, speaker-aware ASR and EOT need two encoder passes. `model.forward` builds `enc_spk` with a second encoder call, and `transcribe_speakers` re-encodes once per speaker. The idea attaches a causal diarization head at layer k and feeds its primary-speaker activity straight into the SpeakerKernels above k. That takes one pass, costs about half the encoder compute, and removes the cascade skew.

**Mechanism.**
- `FastConformerEncoder.forward` and `_run_layers`/`stream_step` get a `self_cond(hidden_k) -> act` hook after layer k. That is k=1 of 4 in the tiny model, or about 6 of 17 at 120M. (The claim that speaker information peaks near layer 6 is not in the README; see the feasibility critique.)
- `SortformerHead.forward_emb` runs on hidden_k with a causal attention mask. Its primary column (`heads/turn.primary_column`) becomes `spk_act` for the kernels at layers above k.
- Training uses scheduled teacher forcing: oracle activity with probability p, annealed from 1 to 0.3, otherwise the detached prediction. The diar loss is applied at layer k.
- In streaming, the diar output for chunk c is ready before layers k+1..N run on chunk c, so latency does not change.

**The test.**
- **Planned falsification test** (from the feasibility critic):
  - Arms, each trained for 700 steps on 1500 conversations:
    - A': two passes, kernels at [2,3], causal [-1,1] diar called once on the final plain layer.
    - B: in-pass, diar at layer 1, kernels at [2,3], teacher forcing on.
    - C: unconditioned.
    - B0: B with p fixed at 1.
  - Confirm, all must hold: B P50@5fc ≤ A' + 80 ms; B beats C by at least 20%; B's WER with diarized activity ≤ A' + 1.0; B's pipeline time ≤ 0.7x A'.
  - Kill, any one: B no better than C; B > A' + 160 ms; B's diar primary F1 more than 8 points below A'; long-form swap rate above 10%.
- **What actually ran:**
  - Inference-only checks on an untrained 4x144 model with the speaker_aware_turn recipe (TDT dropped, use_text=false, kernels [2,3], [16,1]).
  - CPU, 1 thread, 8 validation conversations, about 3 minutes.
  - No training, because the safety rule forbids it. The logs also show 50 steps take 505 to 1159 s on this shared machine, so the planned test is several hours.

**Measured (verbatim from the prototype):**

Untrained 4x144 model, 2.4M params, CPU, 1 thread. Times are ms per 10 s of audio, median of 20 runs.

| Arm | total | features | encoder | diar | CTC + turn |
|---|---|---|---|---|---|
| A' two-pass (causal diar on final plain layer) | 19.20 | 0.86 | 16.80 | 0.68 | 0.84 |
| B in-pass (diar at layer 1, kernels at 2,3) | 10.83 | 0.84 | 8.53 | 0.66 | 0.82 |
| C unconditioned | 9.94 | 0.83 | 8.28 | 0 | 0.81 |

| Check | Value | Criterion |
|---|---|---|
| Pipeline B / A' | 0.56x | ≤ 0.7x: pass |
| Encoder B / A' (arithmetic, about 2x expected) | 0.51x (1.97x) | reported only |
| Causal [-1,1] diar, prefix vs full, max difference | 1.8e-7 | streamable |
| Unmasked diar, prefix vs full, max difference | 0.10 | not streamable |
| Streamed B vs offline B, max difference | 2.1e-6 | exact, latency unchanged |
| In-pass vs external spk_act, max difference | 0.0 | hook is exact |
| Gradient norms (layer 0 / diar proj / kernel 2) | 0.20 / 0.15 / 0.69 | trains end to end |
| P50@5fc, B vs A' and B vs C | not run (training forbidden) | – |
| WER with diar vs oracle activity | not run | – |
| Diar primary F1 / DER | not run | – |
| Long-form swap rate, [-1,1] vs [16,1] | not run | – |
| B0 vs B (teacher-forcing effect) | not run | – |
| Earlier result: two-pass turn_speaker P50@5fc, oracle / diar activity | 400 ms / infinite (94% miss) | the exposure risk B is meant to fix |

**Caveats from the prototype.**
- The speed criterion passes on architecture alone. The feasibility critic pointed out that "one pass is faster than two" is close to true by construction, so this result shows the design is plumbed correctly, not that the idea works.
- `nn.TransformerEncoder` has no KV cache, so the streamed causal diar re-runs the whole prefix every chunk. Its cost grows as O(T²) unless a cached causal transformer is written.

**Verdict: inconclusive.** The mechanism and the speed criterion are confirmed. The accuracy criteria, which decide the idea, were not measured.

**Next step.**
- Start from `/private/tmp/claude-501/-Users-maxm/54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad/wf-ideas/proto-architecture-2/proto.py`. Its hooks are already checked for gradients and streaming.
- Add a training loop with p annealed from 1 to 0.3, and run A', B, C and B0 when the MPS GPU is free.
- Report P50@5fc and WER with diarized activity.
- Write a KV-cached causal diar.
- Results so far are in `.../proto-architecture-2/result.json`.

## 3. Ranked survivors

| rank | id | title | judge score | novelty score | feasibility score | target metric | one-line why |
|---|---|---|---|---|---|---|---|
| 1 | architecture-2 | In-pass self-conditioning: mid-layer causal diarizer steers upper-layer speaker kernels in one pass | 14.5 | 3 (narrow pass) | 3 (pass, test rewritten) | Full-pipeline RTF (≤0.3 on one CPU core) at unchanged speaker-aware EOT dead air @5% false cutoffs and conditioned WER | Removes the second encoder pass (measured 0.56x pipeline time). Teacher forcing may close the oracle vs diarized-activity gap (400 ms vs 94% miss), but that is untested. |

No other idea passed both critics.

## 4. Survivor in detail: architecture-2

**Mechanism.** See §2. The diar head runs at layer k with a causal mask. Its primary column drives the SpeakerKernels at layers above k in the same pass. Training uses scheduled teacher forcing (oracle with probability p, annealed 1 → 0.3, otherwise the detached prediction). The diar loss is applied at layer k. The second encoder pass in `model.py` for `condition_on_speaker` heads is removed.

**Prior art, and why this is different.**
- **Self-conditioning on intermediate predictions already exists.**
  - SC-EEND / intermediate attractors (https://arxiv.org/pdf/2303.06806) condition later layers on intermediate speaker labels, but only to improve diarization, and offline.
  - Self-conditioned CTC (Nozaki & Komatsu 2021, arXiv:2104.02724) does the same with text posteriors.
- **Speaker kernels driven by diarization activity are NVIDIA's shipped mechanism.**
  - Examples: SSA (https://arxiv.org/pdf/2506.22646) and https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1.
  - In all four strategies of the streaming multi-speaker study (https://arxiv.org/html/2609.10265), the diarization comes from a separate Streaming Sortformer. The feasibility critic saw only that paper's title, so this rests on the novelty critic's check.
- **Layer-wise diarization conditioning in Whisper** (DiCoW / SE-DiCoW FDDT: https://arxiv.org/html/2510.03723v2, https://arxiv.org/html/2601.19194; Dixtral https://arxiv.org/html/2606.18134v1) uses an external diarizer, offline.
- **Joint heads on a shared encoder** use parallel losses, not feedback: https://arxiv.org/html/2601.17640, UME https://arxiv.org/abs/2508.20474, Honda TS-ASR-AD https://www.isca-archive.org/interspeech_2025/maeda25_interspeech.pdf.
- **Closest structural precedent: Sidecar** (https://arxiv.org/pdf/2302.09908, https://arxiv.org/pdf/2305.16263), which the idea's author missed. A mid-layer speaker module steers the upper layers, the lower layers are shared, and there is a diarization branch. It is offline and uses separation masks rather than activity-driven kernels.
- **What is new:** the diarizer that produces the kernel input is computed inside the same encoder at layer k, fed back through SpeakerKernels with scheduled teacher forcing, cache-aware, and serves the conditioned CTC/TDT and TurnHead in one streaming pass.
- **How to frame it:** the novelty critic suggests "the first single-encoder, self-conditioned, speaker-aware streaming ASR+EOT model", not a new mechanism.

**Overclaims the critics flagged.**
- "Half the encoder compute" holds only for the primary speaker. Each extra speaker still reruns layers k+1..N. It also only holds for recipes where every non-diar head is conditioned. `voice_agent_frontend` has unconditioned heads, which puts it at about 1.5-1.65x passes.
- The O(T²) cost belongs to the evaluation cascade `heads/turn.streaming_diar_act`, which the critic timed at 28.7 ms against 12-14 ms per encoder pass. A causal mask fixes that for the two-pass design too.
- The kernels move above layer k, which confounds the comparison unless A' also uses kernels at [2,3].
- The diar mask must be [-1,1] or a spkcache, not [16,1]. Otherwise arrival-order columns swap beyond about 5 s.
- A causal diar at layer k will be weaker than the full Streaming Sortformer.
- "Speaker info peaks at ~6 of 17" is not in the README (README L128 only reports a layer-mix finding).
- The 0.16 → 0.10 RTF figure at 111M cannot be tested here.

**Expected gain.**
- Compute: about 2x less encoder compute on the speaker-conditioned path. The prototype measured 0.51x encoder and 0.56x pipeline time.
- Accuracy, per the proposer: parity or small losses (≤1 frame of dead air, ≤1 absolute WER).
- Possible upside: a win on diarizer-driven eval, because the kernels see realistic noisy activity during training. That upside is speculative.

**Test.** The refined test in §2. Add these comparisons from the novelty critic, all at matched parameter count: an external-diarizer stand-in feeding the kernels (SSA/WL-SOT style), and a Sidecar-style shared lower split with no feedback. Also run k ∈ {1,2} and teacher-forcing floors of 0 / 0.3 / 1.

## 5. Rejected ideas

N = novelty critic, F = feasibility critic. Critic reports were truncated in the source data. Reasons below are condensed from the visible text.

| id | title | failed by | one-line reason + citation |
|---|---|---|---|
| architecture-1 | Pinned enrollment KV: user's voice as permanent slots in every layer's cache-aware attention | N (F pass) | Raw enrollment passed through the same encoder's self-attention with no speaker model is already published: Meng et al. 2024 https://arxiv.org/abs/2407.09817, WhisperTSE https://arxiv.org/html/2501.14477, USEF-TSE https://arxiv.org/html/2409.02615v3 |
| architecture-3 | Speaker-conditioned `<eot>` token inside the target-speaker transducer | N (F pass) | Google's streaming RNN-T already has `<end-primary>`/`<end-others>` tokens: https://arxiv.org/abs/2312.11123. Also Microsoft EOS in multi-talker ASR: https://arxiv.org/abs/2201.09979 |
| architecture-4 | Fork the clock: short-context ASR plus long-context half-rate speaker tower on a shared stem | N + F | SURT speaker attribution already forks a long-context speaker branch off the streaming encoder: https://arxiv.org/html/2401.15676. F: the premised bottleneck does not exist, because SortformerHead has its own transformer (audioforge/heads/audio.py:50-97) |
| objectives-distillation-1 | Clean-channel teacher, mixture student: speaker-aware EOT without multi-party labels | N + F | UAF already does this: https://arxiv.org/abs/2604.19221. F: the existing recipe already builds exact EOT labels from separate tracks (conversation.py:39-47), so the claim is empty |
| objectives-distillation-2 | Parakeet-TDT punctuation/timestamps distilled into a text-free streaming EOT head | N (F pass, test confounded) | Semantic VAD uses frame-level ending vs non-ending punctuation targets: https://arxiv.org/abs/2305.12450 |
| objectives-distillation-3 | Prefix-anticipation speaker distillation replacing the enrollment stand-in | N (F pass) | Long-to-short cosine teacher-student is established: https://arxiv.org/abs/1810.10884. Online causal frame-level speaker embeddings: https://arxiv.org/abs/2207.05920 |
| objectives-distillation-4 | Label-disjoint fused training with cross-head logical constraints on unlabeled audio | N (F pass) | Partially annotated MTL and semantic loss exist: https://arxiv.org/abs/2111.14893, https://arxiv.org/abs/1711.11157. Diarization conditioned on VAD/overlap via the chain rule: https://arxiv.org/abs/2106.04078 |
| data-1 | Punctuation-teacher hindsight labels ("utterance end is not turn end") | N + F | Semantic VAD already labels pauses by the preceding punctuation: https://arxiv.org/abs/2305.12450. F: the pilot's "half of ends aren't sentence ends" figure comes from a word-lookup bug |
| data-2 | Leak-audited turn data: trivial-probe gate plus unedited real-hold anchor | N (F pass) | Trivial-feature shortcut probes are standard: https://arxiv.org/abs/2106.12914, https://arxiv.org/abs/1805.01042. EOT label leakage: https://arxiv.org/abs/2609.04225 |
| data-3 | TRP-anchored interference: bystander ends placed inside the primary's real holds | N + F | Timing tied to the other speaker is established: https://arxiv.org/abs/2608.16053. F: on real LibriSpeech the premise is backwards for silence-driven detectors |
| data-4 | Voice-neighbour bystanders mined by speaker-embedding similarity | N (F pass) | Similarity-based interferer curriculum in TSE: https://arxiv.org/pdf/2406.07845 |
| evaluation-1 | Honest eot-bench: turn-level bootstrap, cross-fitted operating points, matched scorer | N (F pass) | Mostly already implemented: https://github.com/txya900619/turn-detection-benchmark (pinned scorer, cluster bootstrap, held-out calibration) |
| evaluation-2 | Voice-swap twins: Speaker-Awareness Index and clean-tuned policy transfer | N (F pass, headline numbers unverifiable) | Same-voice resynthesis control in TPI-Bench Janus-Test: https://arxiv.org/html/2604.17358.pdf. Paired foreground/background mixes: https://arxiv.org/html/2609.19856 |
| evaluation-3 | Deadline-scored speaker attribution: cpWER(L) plus onset-attribution latency | N (F pass; prototype files not found on disk) | Incremental-ASR first-occurrence and final-decision latency, including for diarization: https://aclanthology.org/2020.coling-main.312/ |
| evaluation-4 | Barge-in operating curve by cause (echo/bystander/backchannel) | N (F pass, barely; evidence files missing) | FireRedChat already reports barge-in latency vs false barge-ins with bystanders: https://arxiv.org/abs/2509.06502. Operating curve from https://github.com/livekit/eot-bench |
| systems-1 | Fork-at-depth-K speaker kernels: one shared trunk for diar and per-speaker ASR | N (F pass, mechanism exact) | Streaming shared mixture encoder with per-speaker encoders: https://arxiv.org/pdf/2011.11671. Sidecar: https://arxiv.org/pdf/2302.09908 |
| systems-2 | Silence-elided cache-aware streaming: non-speech frames skip the upper encoder | N (F pass) | Google on-device frame filtering: https://arxiv.org/abs/2211.00786. Skipformer: https://arxiv.org/pdf/2403.08258 |
| systems-3 | Speaker-sparse branches: re-encode each slot only where its speaker is active | N (F pass, smaller saving than claimed) | NeMo already skips inactive speaker instances per chunk and keeps their caches: https://raw.githubusercontent.com/NVIDIA/NeMo/main/nemo/collections/asr/parts/utils/multispk_transcribe_utils.py |
| systems-4 | Batch all speaker slots into one encoder call | N (F pass: B=4 80 ms vs B=1 49 ms) | NeMo defaults to `parallel_speaker_strategy=True`, stacking active speakers into one step: same NeMo file, plus https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py |
| product-1 | Commit events from CTC+RNNT head agreement | N + F | Hybrid-model confidence from both heads: https://arxiv.org/abs/2309.14922, https://arxiv.org/abs/2109.07750. F: the kill test fired, because agreement does not beat plain RNNT entropy confidence at [70,0] |
| product-2 | Load-adaptive lookahead: switch the chunk size mid-stream | N (F pass; pilot tested nothing) | Whisper-Streaming adaptive policy: https://arxiv.org/abs/2307.14743. Variable-context training: https://arxiv.org/pdf/2312.17279 |
| product-3 | TTS-reference-conditioned barge-in: the agent's own voice as a known speaker | N (F pass) | Implicit AEC feeding the playback reference to the detector: https://arxiv.org/abs/2111.10639, https://arxiv.org/abs/2111.09935 |
| product-4 | Network-aware turn-taking: packet-loss/concealment flags as model input | N (F pass; test leaks oracle activity) | Loss-aware models and per-frame error flags are known: https://arxiv.org/abs/2406.18928, ETSI Aurora DSR ES 202 050 |
| generation-side-1 | Own-voice FSQ codec tokens as echo reference, no AEC | N (F pass) | Textual Echo Cancellation uses a symbolic TTS-side reference: https://arxiv.org/abs/2008.06006 |
| generation-side-2 | Codec self-reconstruction auxiliary to keep speaker/prosody in the top layer | N + F | Supervised autoencoders / joint ASR+reconstruction: https://papers.nips.cc/paper/7296-supervised-autoencoders-improving-generalization-performance-with-unsupervised-regularizers, https://arxiv.org/pdf/2512.23808. F: the targeted failure did not reproduce in the critic's run |
| generation-side-3 | Diarization-steered codec-token extraction as auxiliary | N (F pass; test can't isolate the target) | Selective HuBERT predicts the target speaker's clean units as an auxiliary: https://arxiv.org/abs/2311.04526 |
| generation-side-4 | Label-free future codec-token pretraining for EOT | N (F pass, test rewritten) | NEST-RQ next-k-token causal SSL: https://arxiv.org/abs/2409.08680. DualTurn: https://arxiv.org/pdf/2603.08216. VAP: https://ar5iv.labs.arxiv.org/html/2205.09812 |
| cross-field-1 | Pinned enrollment prefix as an attention sink | N (F pass; hidden leak path) | Enrollment inside the model's own context: https://arxiv.org/abs/2505.05114. Speaker Prompt Cache: https://arxiv.org/html/2511.16046v1. Streaming Sortformer cache: https://arxiv.org/html/2507.18446. F: an oracle arrival-rank leak exists at heads/turn.py 382-383, 441-442 |
| cross-field-2 | Classifier-free guidance for speaker-conditioned heads | F (N narrow pass) | For a discriminative head, w>1 amplifies the speaker log-likelihood ratio in the wrong direction for bystander rejection, and the repo's logs show it. Nearest prior art: https://arxiv.org/abs/2603.06193 |
| cross-field-3 | Shared-trunk, speaker-branched encoder (LLM prefix sharing) | N (F pass, sharing exact) | Shared trunk with per-speaker upper layers: https://arxiv.org/abs/1811.02062. Sidecar split-depth ablation: https://arxiv.org/pdf/2302.09908 |
| cross-field-4 | Recurrent memory-token speaker cache replacing AOSC | N (F pass) | Online attractors with retention: https://arxiv.org/abs/2410.06670. Block-level recurrent EEND: https://arxiv.org/abs/2011.02678 |

Note: several ideas that failed novelty still passed feasibility and could be useful engineering for this repo even though they are not publishable. These include systems-1/cross-field-3 (exact prefix sharing), systems-4 (slot batching), data-2 (leak audit), and evaluation-1 (whose reference implementation exists).

## 6. Recommended next builds (ordered)

1. **architecture-2 trained arms A', B, C and B0.** Use the §2 refined test, 128 validation conversations, when the MPS GPU is free or the safety budget allows training.
   - Deciding metrics: B P50@5fc ≤ A' + 80 ms; B beats C by at least 20%; B's WER with diarized activity ≤ A' + 1.0.
   - Kill if B's diar primary F1 is more than 8 points below A' or long-form swap rate exceeds 10%.
   - Key secondary: whether B with diarized activity escapes the 94%-miss / P50 infinite collapse of the existing two-pass model (runs/archive/turn_ablation.json). B0 vs B isolates teacher forcing.
2. **A KV-cached causal diar head**, which idea 1 needs for streaming.
   - Deciding metric: streamed vs offline logits within 1e-5, with per-chunk diar cost flat in stream length (no O(T²) growth) over a 24 s concatenated conversation.
3. **Slot batching in `streaming_diar.py`** (systems-4 mechanism; engineering only, since NeMo ships it).
   - Deciding metric: wall time per 160 ms step for 4 slots ≤ 1.7x one slot, matching the critic's measurement of 80 vs 49 ms, with identical outputs. The feasibility critic flagged that "one call covering diarizer plus all slots" does not work as written. Batch the slots only.
4. **Leak-audit gate on turn data** (data-2 hygiene, not novel). Adopt it before trusting any new synthetic recipe.
   - Deciding metric: a trivial-feature probe AUC near 0.5 on every recipe. The existing pilot gave 0.630 (gen-data/p2.log).
