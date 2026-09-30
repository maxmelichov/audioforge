# Open questions: how do we improve further? (prompt for outside reviewers, 2026-09-26)

Context first: read research/archive/BRIEF.md (goal, scorecard, how we work). Short version: one frozen NVIDIA
FastConformer streaming encoder (115M, 160 ms chunks, laptop CPU) with our heads for VAD, speaker, diarization
and speaker-aware end-of-turn. ASR equal to NVIDIA's own, VAD better than every streaming VAD, turn-end better
than every dedicated turn model overall but 19 points worse than a plain Silero timeout on floor-open ends,
diarization and speaker verification worse than dedicated models. All numbers leak-free, AMI dev, matched
<=5% false cutoffs, CIs (research/archive/BASELINES.md, research/archive/EOT_BENCH_V2.md).

New facts since the brief (today):
- Speaker head: a head-only retrain reading a single early encoder block (block 4 of 17) halves within-meeting
  EER (AMI 32 -> 16-19%, held-out ICSI 42 -> 17%); the old all-layer softmax mix was near-uniform although
  speaker information lives in blocks 1-8 and is at chance from block 10 up. TitaNet-L is still 8-12% (AMI) and
  1% (ICSI). Distillation and a relational loss + attentive pooling variant are running.
- Semantic end-of-turn via YIELD/HOLD tokens in the transducer: negative. Head-only on the frozen encoder the
  token never emits; its posterior fires on silence (median lag 1.16 s vs 1.28 s median pause). NVIDIA's own
  end-to-end Parakeet-EOU is the upper bound and does not beat a VAD timeout on open ends.
- Product: serving the bench-winning head in Pipecat/LiveKit gives no dead-air or cut-in gain over a 1000 ms
  timeout; every head crossing lands 0.7-2.4 s after the timeout's decision. The bench win is a 6 s-horizon
  miss-rate gain on ends a timeout never catches.
- TitaNet within-window EER: 10.2% on clean masks, 19.7% on diarizer-column masks: column impurity costs ~10
  points before any following logic runs. Our voice enrollment used 1.5 s of speech; literature says the cliff
  is at 1.5 s and saturation at 3-5 s (variants running).
- Outside survey (research/archive/OUTSIDE.md): TurnBench (Sesame) is our benchmark with different constants; otoSpeech
  is the only real, permissive, per-speaker two-party corpus; two 2026 papers claim speaker-aware frame-level EOT
  but not on a frozen encoder, enrollment-conditioned, or matched-FC on a public set; Voice-Light is an
  independent negative for learned EOT on a shared ASR encoder.

Answer as many of the questions below as you can. For every proposal: which scorecard row it moves, the control,
the kill criterion, and whether it runs on a 24 GB Mac or needs a GPU. Prefer things we can measure in a day.

## A. Speaker identity (the measured bottleneck)
1. Block 4 gives 16-19% within-meeting EER with 190 AMI speakers and a 0.5M head. What closes the rest of the
   gap to TitaNet (8-12% AMI, 1% ICSI): more speakers (VoxCeleb-scale) alone, distillation, a 1-2 block
   trainable adapter, attentive pooling, or longer segments? What is the expected ceiling for a frozen ASR
   encoder, given the literature (2309.03019 reached 0.57% Vox1-O with adapters)?
2. Should the served model use block-4 features for enrollment and following, keep TitaNet-L as a companion
   (+200 MB, 51 ms per batched embedding), or distill TitaNet into a block-4 head? What decides it?
3. Column impurity: the diarizer's columns cost a real embedding ~10 EER points. Is the fix better columns
   (a different diarizer, a target-speaker VAD instead of clustering), or embedding from raw audio masked by
   our own VAD rather than by the diarizer?
4. Enrollment identity is wrong 30-50% of the time label-free. In a two-party product it is trivial. Is there
   any principled label-free rule for multi-party audio that beats "first voice after the agent stops", or
   should we stop trying and require the agent-end signal?
5. Overlap: does a frame-level lower-layer embedding track the louder speaker, the primary, or garbage during
   overlapping speech? (Measurement running.) If garbage, what conditioning survives overlap?

## B. Turn-taking on floor-open ends (the product problem)
6. The head OR an any-speaker Silero timeout scores 59.6 / 34.3% (open) on AMI dev vs hybrid 61.9 / 45.9.
   ICSI held-out confirmation is running. If it holds, is there a better combination than OR: a learned
   fusion of head, silence durations and a completeness score? What would make it fail in a product?
7. Our margin is entirely on floor-taken ends. On open ends the silence timeout wins. Is there any acoustic or
   contextual cue on a floor-open end that a speaker-unaware VAD does not already use, on real meeting audio?
   Evidence so far says no (LiveKit text with oracle words, smart-turn, EOU, tokens all fail). Prove us wrong.
8. Voice Activity Projection: the original VAP used exactly our input layout (mono mix + per-speaker activity).
   Is a VAP-style 256-way future-activity head on our frozen encoder likely to beat independent BCE on our
   6 s / 2 s horizons, given VAP's own ablation says the discrete head only helps on prediction tasks?
9. Dead air in Pipecat/LiveKit is timeout-grade (1.7-1.9 s median). The head's operating point (theta 0.998 for
   <=5% cutoffs at 6 s) fires after the timeout. Is there an operating-point or policy design that gets earlier
   decisions on open ends without more cut-ins, or is the 1 s timeout already near the floor for this audio?
10. Two-party data: we have TurnBench dev and otoSpeech access now. What is the right first experiment: score
    our AMI-trained head as-is, retrain the head on otoSpeech with a real agent-end signal, or both? What result
    on TurnBench would be publishable next to VAP 0.845 / smart-turn 0.752?

## C. Completeness and semantics
11. smart-turn's completeness classifier adds nothing over its own timeout on meetings but reaches 0.752 on
    dyadic TurnBench. A completeness head trained on smart-turn's data on our shared encoder is being built.
    Where should completeness enter the decision: as a validator after silence, as a fusion feature, or as a
    prior on the timeout length?
12. Given the YIELD/HOLD failure, is there any training design under which turn tokens in a transducer learn on
    ~3k windows with a frozen encoder, or does this need end-to-end training on much more data (as EOU had)?

## D. State, latency and compute
13. State contamination: if wrong speaker conditioning leaves persistent damage in the turn head (probe
    running), and bounded replay of 1-4 s is cheap on CPU, is a learned repair module ever worth it, or is the
    publishable artefact the measurement plus a speaker-revision event in the protocol?
14. Gated encoder: a block-3 VAD probe could skip blocks 5-17 during silence. The left-context cache of the deep
    layers then has holes when speech resumes. What is the cheapest correct re-prime, and is the saving worth
    it at RTF 0.16 per core?
15. Would a 0.6B cache-aware streaming encoder (Nemotron-Speech-Streaming, multitalker Parakeet) carry more
    speaker information in its lower blocks, and what does it cost on a CPU? (Import running.) With cached
    features, head training cost is independent of encoder size; is that the right way to use a bigger model
    without a GPU?
16. Encoder fine-tuning always tripped the WER gate on this machine (batch 3, partial unfreeze). Is that a
    batch-size/anchor artefact that a GPU fixes, or is a frozen encoder simply correct for a multi-head
    front-end? What recipe (LoRA, adapters, KL anchor, replay mix) would you try first on one 24 GB GPU?

## E. Evaluation and claims
17. eot-bench v2 vs TurnBench: which constants differ (window, FPR cap, causality, matching), and which of our
    conclusions could flip under TurnBench's protocol?
18. Our VAD win has a label caveat (trained on the same word-level AMI label as the test). What evaluation
    removes it? Is pyannote's offline VAD the right reference, and at what CPU cost?
19. What is the honest headline claim we can make today, and what single experiment would most strengthen or
    kill it? Candidates: "one frozen streaming encoder serves ASR, VAD, speaker and speaker-aware turn ending on
    a laptop CPU"; "on overlapping multi-party audio, semantic end-of-turn adds nothing over a VAD timeout";
    "the identity bottleneck is routing/supervision, not the encoder".

## F. Data and training plan
20. research/archive/DATA_PLAN.md ranks a label-consistency pass and enrollment supervision above more hours. With
    otoSpeech (141 h, two-party, CC BY 4.0, no speaker-ID training allowed) now available, does that ranking
    change? What is the minimal GPU budget that would settle questions 1, 10 and 16?
