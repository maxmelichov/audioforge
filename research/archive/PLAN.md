# Plan: one open streaming speech front-end, one clock

*Last updated 2026-09-25. This is the working plan; results live in the README and the other
research/ files. Every item here has a gate — a number that decides whether we continue.*

## 0. Where we are (verified)

| Piece | Status | Evidence |
|---|---|---|
| NVIDIA's recipe reproduced in plain PyTorch (FastConformer + heads, cache-aware streaming, RNNT/TDT, Sortformer, FSQ codec) | done, 100+ tests | `tests/`, README |
| NVIDIA's own models running locally as teachers/benchmarks | done | `research/archive/TEACHERS.md`: parakeet-ctc-1.1b 1.64% WER (not significantly below 0.6b's 1.73% on this 200-utt subset), 63× realtime on M5 |
| NVIDIA's pretrained weights loaded into our code (no NeMo) | done | `research/archive/NEMO_IMPORT.md`: 115M streaming model 1.92% WER at 1 s lookahead, 2.48% at 0 |
| Pretrained encoder + our new heads (VAD, EOU, speaker, turn) | done, frozen only | `init.from`: 117.5M model, identical transcripts before training |
| On-device budget | done | `research/archive/ONDEVICE.md`: 110M at 160 ms chunks = RTF 0.16 on one CPU core |
| Real-speech training from scratch | tried, negative | 14M model on 16 h: VAD/speaker learn, ASR does not in 2k steps → start from pretrained weights |
| Speaker-aware turn-taking, streaming diarization with speaker cache | code + tests done; models untrained (runs killed by machine overload) | `d248dd7`, `1a2361f` |
| Verification pass (28 claims, 3 ablations, edge-case bugs) | done | [`research/archive/VERIFICATION.md`](VERIFICATION.md): 4 confirmed, 20 partially true, 4 refuted; all 3 ablations mixed; EOU head below its always-0 baseline; doc corrections applied (README "Corrections log") |
| Landscape | done | `research/archive/GAME_CHANGER.md`: the gap is *speaker-aware* end-of-turn in one model; real bar ≈ 540 ms dead air at 5% false cutoffs (eot-bench), not 80 ms |

## 1. The claim we are trying to earn

> An open (CC-BY-4.0 encoder), ~120M-parameter, cache-aware streaming model that emits on one
> 80 ms clock: words, is-speech, **speaker-aware end-of-turn**, speaker activity/identity — at
> NVIDIA-level WER, in real time on one CPU core — and that cuts users off less often than a
> silence timeout *at the same dead-air latency* on real conversations.

Not claimed: SOTA WER, sub-100 ms turn detection, a moat. NVIDIA has every piece (EOU model,
language-ID prompt, 8-speaker streaming diarizer, NeMo-Speech.cpp) and could fuse them any month.
Our edge is being first, open, small, and honestly evaluated.

## 2. Three risks the critique raised, and the design answer to each

### R1. Acoustics alone cannot detect end-of-turn
A breath mid-sentence is silence to any acoustic head; 5% false cutoffs in a live call is worse UX
than a 600 ms wait.
- **Design:** the turn head reads three things per frame: encoder features (acoustics), the
  **transducer prediction-network state + last k token embeddings** (a running summary of the
  text: is the sentence syntactically open?), and the **primary speaker's activity** (who is
  talking). The ASR state acts as a veto on acoustic cues. Output is a soft probability; the
  agent picks its operating point on the false-cutoff/latency curve.
- **Why this is new:** LiveKit/Pipecat detectors are text-only after ASR; NVIDIA's EOU token is
  acoustic-only inside the ASR; Deepgram Flux fuses ASR+EOT but has no speaker. Nobody fuses all three.
- **Gate:** match or beat LiveKit ~543 ms at ≤5% false cutoffs while speaker-aware (eot-bench
  dead air; LiveKit Turn Detector v1 543 ms, JoinIn AI Baton 577 ms, see `VERIFICATION.md` C22).
  On held-out conversations, at ≤5% false cutoffs, dead-air P50 must also beat (a) silence timeout
  and (b) the acoustic-only head by ≥150 ms; and at fixed 540 ms dead air, false cutoffs must be
  lower than both. (Corrected 2026-09-25: the fixed point was 600 ms, which hosted detectors already beat.)

### R2. Multi-head training will damage the pretrained ASR (catastrophic forgetting / gradient conflict)
Diarization and turn heads want speaker- and prosody-sensitive features; CTC/RNNT wants invariance.
- **Design, staged:**
  1. **Stage 1 — heads only.** `init.pretrained_lr_mult: 0` (encoder frozen). Speaker heads read a
     learned mix of all layers (`from_layers: all`, NEST-style) so they get speaker information
     without moving the encoder. Cheap, safe, establishes head baselines.
  2. **Stage 2 — gentle unfreeze.** `pretrained_lr_mult: 0.05–0.1`, plus a **frozen-teacher KL
     anchor**: keep a copy of the pretrained RNNT and add KL(student ‖ teacher) on the joint
     outputs (the same mechanism as NVIDIA's 2026 mode-consistency loss, used here as a
     forgetting guard). Per-head `grad_scale` already limits how hard auxiliary heads push on the
     shared encoder.
  3. GradNorm / dynamic loss weighting only if the gate below fails.
- **Gate:** WER on the fixed 200 LibriSpeech test-clean utterances may not rise more than **+0.2
  absolute** from 1.92% (checked every 500 steps; the run stops if it does). Streaming == offline
  must still hold exactly.

### R3. LibriSpeech proves nothing about conversation
Read speech: clean, single speaker, no overlap, no turn dynamics.
- **Design:** LibriSpeech is demoted to a **regression test** for R2. Turn-taking and diarization
  are trained and evaluated on conversational data:
  - **AMI Meeting Corpus** (CC-BY-4.0, ~100 h, real meetings, overlap, backchannels, word timings →
    true turn ends and diarization labels). Headset-mix first, distant mics later.
  - **Synthetic conversations** from diverse single-speaker sources (LibriSpeech, Common Voice,
    People's Speech) using the four-move turn model from NVIDIA's FastMSS paper (hold / switch /
    interruption / backchannel, exponential pauses), with different overlap mixes for the ASR and
    diarization heads (the paper: overlap helps ASR, hurts diarization).
  - Fisher / CALLHOME / DIHARD only if budget allows (LDC, paid). VoxCeleb avoided (license).
- **Gate:** all turn-taking numbers reported on AMI held-out meetings, never on synthetic data alone.

## 3. What the 2026 NVIDIA papers change

| Paper | Take-away | Action |
|---|---|---|
| Unified ASR / MCR-RNNT (2604.19079, `parakeet-unified-en-0.6b`) | one model offline+streaming; mode-consistency KL between modes cut 0.32 s-latency WER from 15.9% to 8.3% | add symmetric KL across attention-context modes to the trainer (~30 lines); reuse the same term as the R2 forgetting guard; consider porting their newer checkpoint |
| FastMSS synthetic conversations (2605.15442, Nemotron-3-Diarization) | four-move turn model; overlap helps ASR, hurts diarization; source diversity beats domain match; synthetic+real > real | rebuild `conversation.py`'s generator on the four-move model; separate overlap settings per head; mix in AMI |
| One-for-all latency USE (2606.25621, Real-time_RE-USE) | one model for all latency budgets: per-lookahead conv branches + early exit | early exit for the on-device runtime when the CPU is busy |
| Nemotron 3.5 ASR (language-ID prompt fused before the decoder) | NVIDIA keeps merging tasks into one encoder | supports the thesis; add a language-ID frame head later |
| FDB-v3 (VoiceChat paper) | even GPT-Realtime interrupts users 13.5% of the time on real disfluent speech | the problem is real and measured; use FDB-v3's disfluency categories in our synthetic generator |

## 4. Execution queue (strictly one job at a time on this Mac)

| # | Job | Compute | Gate / output |
|---|---|---|---|
| 1 | Verification workflow finishes (capped: top-25 findings, one round) | done 2026-09-25 | `research/archive/VERIFICATION.md`; apply required doc corrections; commit fixes |
| 2 | Ideas tournament, resumed with prototypes capped to 3 | CPU | `research/archive/IDEAS.md`; reprioritize this queue |
| 3 | Turn head v2: + prediction-net state + token context; generator v2: four-move model + FDB-v3 disfluency types | code only | tests green |
| 4 | Trainer: frozen-teacher KL anchor + WER gate + mode-consistency option | code only | unit test: KL=0 when student==teacher; gate stops a run |
| 5 | AMI download + manifest (turn ends, RTTM, speakers) | disk | `audioforge/datasets/ami.py`; stats doc |
| 6 | Train synthetic turn-taking pair (speaker-aware vs acoustic-only) — the run that was killed | MPS, ~15 min each, sequential | eot-bench table incl. silence baseline |
| 7 | Stage 1 on pretrained 115M: heads only (vad, eou, spk, diar, speaker-aware turn), synthetic conversations + AMI turn + AMI diar (`research/recipes/stage1_heads_pretrained.yaml`): `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 .venv/bin/python -m audioforge.cli train research/recipes/stage1_heads_pretrained.yaml -o runs/stage1_heads_pretrained.afm 2>&1 \| tee runs/stage1_heads_pretrained.log` | MPS, ~30 min, ~3-5 GB | R1 gate on AMI dev; WER unchanged by construction (frozen; wer_gate tripwire) |
| 8 | Stage 2: gentle unfreeze (backbone + ASR at lr x0.05) with KL anchor + mode consistency, from row 7's output (`research/recipes/stage2_unfreeze_pretrained.yaml`): `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 .venv/bin/python -m audioforge.cli train research/recipes/stage2_unfreeze_pretrained.yaml -o runs/stage2_unfreeze_pretrained.afm 2>&1 \| tee runs/stage2_unfreeze_pretrained.log` | MPS, ~1 h, ~8-12 GB | R2 gate (+0.2 WER max = wer_gate max_delta 0.002, every 500 steps, n 100) and R1 gate |
| 9 | Streaming diarization model retrain + eval (the other killed run) | MPS, ~15 min | offline vs streaming DER, slot-flip rate with/without cache |
| 10 | On-device: export the stage-2 model to ONNX int8, re-measure on an idle machine | CPU | RTF and WER delta from int8 on a real model |

Rules: no workflow while a training runs; one training at a time; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`;
checkpoints every 500 steps; every eval on a fixed, named held-out set.

## 5. What would make us stop

- R2 gate fails even with the KL anchor and frozen encoder → the fused model costs ASR accuracy;
  fall back to "frozen encoder + probes" as the product (still useful, weaker claim).
- Speaker-aware turn head does not beat the acoustic-only head on AMI → the speaker signal is
  not the missing piece; the product becomes "open fused front-end", not "better turn-taking".
- NVIDIA ships the fused model first → pivot to evaluation + on-device tooling around theirs.

## 6. Findings log

- **2026-09-26 — turn ablation v1 (synthetic):** with oracle speaker activity the speaker-aware head
  reaches 400 ms P50 dead air at ≤5% false cutoffs vs 1440 ms acoustic-only and 2560 ms silence
  timeout. Text-only and speaker+text arms were worse (training bug: alignment on the wrong
  encoding). End-to-end failed (94% miss): eval re-ran an offline diarizer on prefixes.
  → `research/archive/TURN_ABLATION.md`.
- **2026-09-26 — turn ablation v3:** end-to-end path fires (6% miss) but with a ~40% DER diarizer the
  speaker-aware head is no better than acoustic-only, and noisy-activity conditioning also dragged
  the oracle case to 1680 ms. **Diarizer quality is the bottleneck for the whole tier-1 claim.**
- **2026-09-26 — stage 1, step 500 (AMI dev, unseen meetings/speakers):** VAD acc 0.93; frozen RNNT
  intact (LibriSpeech gate 1.13%) and 23.2% WER on AMI segments; diar head collapsed into a VAD
  (DER 0.43); speaker EER 20% = untrained features; turn head below the oracle-primary silence
  timeout, which on AMI is nearly the label itself (word timings hide pauses < 0.5 s). Two config
  errors found: heads ran at the ported model's 1.12 s lookahead (must be [70,1] = 160 ms), and the
  recipe lacked the diar fixes (from_layers all, prefix loss). → `research/archive/STAGE1.md`.
- **Consequence for evaluation:** on AMI the fair baseline is the any-speaker (VAD) timeout (61%
  misses because of overlap), not the oracle-primary timeout; report both, and report the
  encoder's chunk size next to every latency number.
- **Ideas tournament:** 32 ideas, 1 survivor (in-pass self-conditioning), 31 with prior art
  (`research/archive/IDEAS.md`). Differentiation is the combination + honest real-meeting evaluation, not a
  new mechanism.
- **2026-09-26 — stage 1 v2 (2000 steps, 160 ms context):** turn head with oracle activity misses 6.6%
  (was 47.5%), speaker EER 20.4 → 14.5%, VAD 0.93; diarization DER 0.43 → 0.38, still below the trivial
  one-speaker+VAD baseline (0.31) → the frozen ASR encoder + small Sortformer head does not learn
  diarization. → `research/archive/STAGE1.md`.
- **2026-09-26 — stage 2, first attempts:** full unfreeze OOMs on this Mac (10.7 GB at batch 4); with
  the top 6 of 17 blocks trainable at lr×0.05 and the KL anchor, the **WER gate tripped at step 500
  (2.13% → 8.18%)** — R2 (forgetting) confirmed. Cause: the RNNT was training on the synthetic
  tone-language "transcripts" (its loss fell 7.6 → 4.1). Fix: drop synthetic text for ASR heads,
  anchor 1.0, checkpoint/gate every 250. Lesson: the gate + checkpoint-before-gate is mandatory;
  never let a proxy-language source reach a pretrained text head.
- **2026-09-26 — stage 2, final:** with only the top 6 blocks at lr×0.05, KL anchor 1.0 and no synthetic
  text, the WER gate held at step 250 (2.05%) and tripped at step 500 (4.31%); the run restored the
  step-250 checkpoint. Diarization did not improve (DER 0.45). **Decision: keep the NVIDIA ASR encoder
  frozen (stage 1) and stop trying to teach it diarization.**
- **2026-09-26 — NVIDIA Streaming Sortformer v2 ported (CC-BY-4.0):** DER 0.201 on the same AMI dev
  windows (ours 0.384, trivial 0.312). **Decision: the diarization branch is NVIDIA's Sortformer, run
  in our code; our contribution is the turn/VAD/speaker heads on the frozen ASR encoder and the fusion.
  "One encoder" becomes a research item (in-pass self-conditioning, IDEAS.md), not the product.**
- **2026-09-26 — stage 1 at the TRUE 160 ms context:** the v2 recipe had a duplicated `encoder:` key
  (YAML keeps the last), so v2 ran at 1.12 s. At 160 ms: oracle-activity misses 23% (not 6.6%), with
  NVIDIA's Sortformer 61–82% vs **26–33% for Sortformer + silence timeout**. **On real meetings at low
  latency the turn head currently loses to the cascade.** Last untried lever: train on real Sortformer
  tracks. If that fails, the tier-1 claim is unsupported on AMI and the product is the fused on-device
  front-end (frozen NVIDIA ASR + ported Sortformer + our VAD/speaker heads), not better turn-taking.
- **2026-09-26 — turn head trained on real Sortformer tracks (final of this session):** with oracle
  activity it beats the silence timeout on AMI at 160 ms for the first time (880 vs 1200 ms P50 at
  ≤5% FC; 6.3% vs 9.4% FC at 400 ms). With NVIDIA's streaming diarizer it loses to the diarizer +
  timeout cascade (49% vs 33% misses). **Tier-1 status: mechanism confirmed on real meetings with
  clean speaker information; end-to-end claim not supported.** The wall is diarizer-error robustness,
  not the head or the encoder.
- **2026-09-26 — n=200 correction:** the n=64 oracle-activity win (880 vs 1200 ms) does not hold at
  n=200 (29.5% vs 6.3% misses at ≤5% FC; only 11% vs 14% FC at the 400 ms point). **Tier-1 claim: not
  supported on AMI at this scale.** Product stands: fused on-device front-end (frozen NVIDIA ASR +
  ported Sortformer + VAD/speaker heads). Research stays open, with a fixed rule: n ≥ 200 per report.
- **2026-09-26 — turn head v3 (4 columns + duration counters + future-activity aux, streaming tracks), n=200:**
  with oracle activity **560 ms P50 / 5.8% misses vs timeout 1440 ms / 6.3%** — the mechanism now
  clearly beats the timeout on AMI at 160 ms. With Sortformer streaming tracks 63% vs 38% misses:
  still loses end to end. Next: error analysis (fail where the track is wrong?), a better diarizer
  (Nemotron-3-Diarization / v2.1), and the AMI+ICSI run.
- **2026-09-26 — v4 / v5 negative results (n=200):** v4 (retrain the speaker kernels on noisy streaming
  tracks): no end-to-end gain (62% vs 63%), oracle path worse (880 ms / 10.5%). v5 (track reaches only
  the head, no encoder conditioning): oracle 47% misses, streaming 83%. **The speaker kernels trained on
  clean activity are essential; v3 stays the best configuration.** Remaining levers: fixed streaming
  tracks (end-of-file flush artifact in the post-end frames of every cached streaming track), ICSI data,
  and the 6 s scoring horizon.
- **2026-09-26 — live integration (research/archive/INTEGRATION.md, verified):** server + Pipecat + LiveKit run
  at 1x on 2 CPU threads (RTF ~0.5, 3.4 GB); turn-taking is timeout-grade (median 1.7–1.9 s dead air at the 0.32 s default; hybrid rule gives no product-level gain because the served head never reaches θ=0.998); the
  shipped default is now the plain label-free timeout (verifier D1).
- **2026-09-26 — AMI+ICSI turn training (v3 head, 7117 turns, 3000 steps), AMI dev n=200:** end to end
  61% misses (v3 63%), oracle 720 ms / 9.5% (v3 560 / 5.8%). More conversational data did not flip the
  result on AMI dev (ICSI's label conventions differ: FA on laughter/breaths). Last lever in this
  session: retrain v3 on flush-free streaming tracks and score at the 6 s horizon.
- **2026-09-26 — session verdict on tier 1:** across v3/v4/v5, +ICSI, flush-free tracks and a hybrid rule,
  the head beats the timeout only with clean speaker activity (560 ms / 1.6% hybrid vs 1440 / 6.3%);
  end to end the best is 60% vs 38% misses. Not supported end to end. See STAGE1 "Consolidated".
- **2026-09-26 — trail-6 turn head (trained on 6 s post-turn trails), eot-bench v2, n=974, causal
  enrollment:** **66.5% vs 74.8% misses for the deployable timeout at 6 s (CI excludes 0) — first
  leak-free end-to-end win**; still loses on floor-open ends (57.3 vs 46.1); with oracle enrollment 30.3%.
  The remaining loss is label-free speaker enrollment (+36 pts), not the head. Next: hybrid rule at 6 s,
  0.32 s diarizer setting, better causal binding.
- **2026-09-26 — hybrid at 6 s + 0.32 s diarizer (eot-bench v2, n=974, causal enrollment):** hybrid
  (trail-6 head OR ~4 s timeout) 61.9% misses vs timeout 74.8% overall (CI excludes 0), and ties the
  timeout on floor-open ends (45.9 vs 46.1). The 0.32 s diarizer setting cuts latency by 0.5–0.8 s
  wherever systems fire but does not change misses or causal-binding agreement (70%): **label-free
  speaker enrollment (~37 pts causal vs oracle) is the dominant remaining loss.**
