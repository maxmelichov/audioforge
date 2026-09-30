# Outside review, 2026-09-26: actions taken, queued, and disagreements

Two reviewers answered research/archive/QUESTIONS.md. Verified against the repo before acting.

## Verified facts that change the reading
- The speaker layer sweep trained on **45 speakers** (930 AMI single-speaker segments, 1.8 h), not 190; 190 is
  the output-class count. Block 4's 16-19% within-meeting EER was reached with 45 identities. Identity diversity
  is therefore a prime lever, and the distillation rows need a data-only control (queued).
- The speaker head already uses attentive-statistics pooling; "add attentive pooling" is moot.
- The overlap probe defines "louder" as more labelled active frames (activity dominance), not level. A
  controlled-mixture test (fixed words/speakers, swept relative level) is the right follow-up (backlog).
- TurnBench credits detections from -0.25 s to +3 s after an end, per-pause false positives, unscored regions
  ignored. Our 6 s-horizon miss advantage may vanish under it; the official scorer run is in progress.

## Actions taken (agents updated)
- Speaker head: report 45 speakers; add AAM + LibriSpeech-speakers control to separate objective gain from
  diversity gain; relational-MSE variant kept.
- Completeness/fusion: add a dynamic-timeout policy family (required silence shrinks with head posterior and/or
  completeness) fitted jointly under the <=5% FC constraint, evaluated continuously (duplicates, late fires).
  Both reviewers converged on "modulate the timeout" over OR.
- Enrollment: queued a VAD-masked, contiguous-span, majority-consistent enrollment variant (§9b) after chain2.

## Backlog (not started; machine is saturated)
- Q7 test: at pause ages 200/400/600/800 ms, does the last voiced second (pitch trajectory, final-syllable
  duration, energy decay, block-4 features) add anything over duration/activity features, with shuffled-context
  control? Cached features, CPU.
- Overlap: controlled two-speaker mixtures with swept level; does block-4 identity follow level, primary, or
  centroid?
- VAD label caveat: independent acoustic labels on an untouched AMI/ICSI subset, or an external SAD set
  (DIHARD is LDC-licensed; AVA-Speech labels are free but audio is YouTube-bound). Decide after VAD_LAYERS lands.
- Latency decomposition of the 1.7-1.9 s product dead air (offset detection, policy, scheduling, generation,
  first audible output) before any more model work aimed at latency.
- Turn tokens: one last diagnostic only if cheap: can a constrained action branch overfit a tiny clean subset?
  Otherwise closed.
- TurnBench: score identical continuous predictions under both protocols, changing one rule at a time.

## Disagreements recorded
- "1.0 s is the physical floor": unproven; it is our best operating point on AMI, not a bound.
- "0.6B encoders will have worse early-block EER": a guess; being measured (ENC_0P6B).
- "Bounded replay costs < 5 ms": a guess; being measured (CONTAMINATION replay cost).
- "Transducer tokens fail because of monotonic alignment": our diagnosis is optimizer/gate (new rows at
  log P ~ -9 cannot grow at gate-safe lr); the interleaved-transcript alignment problem is real but secondary.
- "Block-4 ceiling is 10-12%": no evidence either way yet.
- Headline wording: keep the narrow version. "A frozen streaming ASR encoder supports a multitask front-end on
  a laptop CPU; layer routing substantially improves speaker verification; speaker-end detection gains do not
  by themselves improve response timing." Do not yet claim "semantics adds nothing generally" or "identity is
  limited only by routing and supervision".
- Prior art to cite: layer-wise speaker information in ASR encoders (arXiv 2402.19443); DualTurn endpointing on
  a frozen Mimi encoder with a small causal head (anyreach-ai/dualturn-endpointing). "Frozen encoder + turn
  head" alone is not the novelty; the shared-computation, enrollment-conditioned, matched-FC comparison is.
