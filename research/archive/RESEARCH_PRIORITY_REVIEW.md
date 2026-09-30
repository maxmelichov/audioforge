# Research priorities reconciled with the implemented experiments

September 26, 2026. This assessment supersedes the proposed priorities in the separate
`nvidia-audio-lab/research/tier1-research-directions.md` catalog workspace. It does not report a new experiment.

## Active: persistence after speaker-conditioning errors

The missing question is **how long damage persists after the conditioning becomes correct**, and which state
carries that damage. Conditioning sensitivity and noise-robustness training already exist here.

Existing evidence and controls:

- [ANALYSIS.md](../ANALYSIS.md) reports approximately 3 % oracle-conditioned versus 18 % own-diarization WER for the
  synthetic speaker-attributed recipe. These are synthetic-task results, not AMI or NVIDIA-checkpoint WER.
- [TURN_ERRORS.md §3](TURN_ERRORS.md) already isolates the speaker-kernel encoder as the main noise entry path
  in its historical v3 diagnostic: oracle encoder conditioning largely restores decisions even with noisy head
  inputs. That is evidence about the entry path, not a post-correction recovery measurement. Its historical
  operating points should not be substituted for the later leak-free trail6 evaluation.
- [The trail6 recipe](../recipes/stage1_turn_v3_trail6.yaml) includes external-track corruption and future-activity
  auxiliary targets. [The conditioning implementation](../../audioforge/model.py) supports flips, jitter, drops, and
  track noise. Existing corrupted-conditioning training is a required baseline; generic "recovery training" is
  not a new contribution by itself.
- [EOT_BENCH_V2.md §7B](EOT_BENCH_V2.md) reports 1.59 rebinds/turn at the 1.04 s preset and 1.53 at 0.32 s.
  These are binding events, not established counts of erroneous state transitions. Lower buffering did not
  improve enrollment agreement, motivating an identity/state intervention separately from latency tuning.

The user is launching the CPU-only probe on cached AMI tracks and the synthetic recipe. The first answer can be
obtained with audioforge and existing checkpoints; importing Multitalker Parakeet is not a prerequisite.

## Controls that make the probe interpretable

1. **Hold the target speaker fixed initially.** Perturb a chunk-aligned conditioning window, restore the exact
   baseline track afterward, and keep audio, weights, initial states, thresholds, and reference labels fixed.
   Verify output equality before the first affected chunk. Repeat on clean oracle and cached estimated tracks.
2. **Separate input errors from binding changes.** After the fixed-target experiment, allow enrollment to run
   normally and stratify rebinds into true identity changes, correct relabelings, and wrong bindings. A legitimate
   target change is not evidence of corruption. Do not let a changed enrollment rule silently alter all suffix
   inputs in a test intended to restore the original track.
3. **Separate the state paths.** The turn head has GRU memory, activity history, and duration counters; the model
   also has conditioned encoder state. Treat these separately. Compare restoring encoder state, turn GRU state,
   explicit counters/history, and all affected states from an identical baseline run. Such oracle state swaps
   identify causes; they are not deployable fixes. Include the RNNT predictor for the synthetic ASR arm. The
   documented v3 turn path uses unconditioned text state, so verify actual routing rather than assuming a
   speaker-mask intervention necessarily changes that text state.
4. **Measure the suffix, not cumulative mistakes.** Record excess word errors outside the corruption interval,
   time-aligned turn-score differences, and changed decisions under the original fixed thresholds. Whole-window
   WER includes irreversible errors emitted during corruption and cannot alone establish persistence. Define
   recovery as a predeclared tolerance sustained over several chunks; report non-recovery as censored. An
   encoder ablation on a full masked forward can show contextual propagation; cache-repair claims additionally
   require stateful streaming parity checks.
5. **Keep both decision and representation measurements.** A long-lived hidden-state difference need not change
   any word or turn decision. Conversely, different duration counters can produce a long score tail without a
   new neural-memory pathology. Include activity perturbation duration, counter clipping/saturation, and actual
   streaming delay in plots. Then use existing eot-bench v2 folds, floor strata, and measured held-out FC for the
   system-level comparison. Do not recalibrate thresholds independently for each corruption window.
6. **Respect the unit of evidence.** Cached-track AMI tests establish behavior of the local front-end; synthetic
   ASR tests establish the mechanism on that task. Neither alone establishes the behavior of NVIDIA's full
   multitalker checkpoint. Pair interventions on the same recording, cluster uncertainty by meeting where
   feasible, and acknowledge the small number of dev meetings. Keep ICSI or an untouched split for confirmation.

The publishable result, if the probe supports it, would be persistent, consequential, recoverable state damage
beyond the corrupted interval, with an efficient remedy that beats existing corruption training, reset policies,
and bounded replay at comparable compute and commitment delay. If fixing the current input removes the effect
quickly, the actionable problem remains enrollment/track quality rather than a new recovery method.

## Parked: codec contrast preservation

The frame-versus-symbol calculation is correct but established accounting. FSQ codecs and the enhancer recipe
could support an ASR probe, but a persuasive acoustic-contrast study needs independent perceptual evidence or a
carefully verified corpus. It is outside the present voice-agent front-end priority. No codec experiment is
scheduled by this review.

## Incremental: value of waiting

Future-speaker activity is already an auxiliary objective, and eot-bench v2 already compares cross-fitted
operating points and delay. A learned wait policy is an ablation on that foundation, not a stand-alone major
contribution without stronger evidence. [BASELINES.md](BASELINES.md) now records VAP as a pending model baseline
and TurnBench as a pending external benchmark, including the input-parity and protocol differences.
