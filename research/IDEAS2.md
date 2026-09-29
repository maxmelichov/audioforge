# IDEAS2: round 2 of the ideas tournament (the measured turn-taking failure)

Generated 2026-09-26 by a workflow run: generate (7 lanes), then a novelty critic, a feasibility critic and a judge. Round 1 is research/IDEAS.md; none of its 31 rejected ideas were re-proposed. Everything below comes from the critics' and judge's reports. Nothing was re-run for this file. Several rejected-idea critiques reached the synthesizer truncated; where a reason is cut short, this file gives only the part that arrived.

**The failure this round targets** (research/STAGE1.md, n=200 table, which is the current truth):

| system (200 AMI dev turns, 160 ms chunks) | P50 @≤5% FC | miss @≤5% FC | FC @P50≤400 ms |
|---|---|---|---|
| turn head + oracle activity | 1440 ms | 29.5% | 11.0% |
| timeout on oracle primary | 1440 ms | 6.3% | 14.0% |
| turn head + Sortformer offline / streaming | inf / inf | 64.2% / 69.0% | 33.5% / — |
| timeout on Sortformer offline / streaming | 1760 / 2640 ms | 32.1% / 38.4% | 40.5% / — |
| timeout on any-speaker activity | inf | 77.0% | — |

In AMI the median mid-turn hesitation (1.2 s) is longer than the median silence at a true speaker change (about 1 s), and 45% of changes overlap (research/AMI.md).

## 1. Executive summary

- **21 ideas** were generated across 7 lanes: turn-under-noise, VAP/projection, semantics, data and labels, product semantics, architecture fusion and wildcards.
- **1 survived both critics. 20 were rejected.** All 20 failed novelty. 9 of them also failed feasibility. No idea failed feasibility alone.
- **There is no top 3, only one survivor:** `wildcards-3`, a leak-free, speaker-aware end-of-turn benchmark on CC-BY meetings (AMI + ICSI). Judge score 16.5. Novelty passed at 3/5 and feasibility at 3/5, both narrowly.
  - **Why it survived:** it is the only idea that changes which tier-1 conclusions are valid. It makes no detector faster; all three judge comments cap its impact for that reason.
  - **Decisive numbers the critics measured on the Mac** (cached dev rows, CPU, no model loaded):
    - Causal enrollment vs oracle slot binding, Sortformer-stream cascade: 66.8% vs 38.4% miss (n=200).
    - On floor-open ends (n=210 of 974), the stream cascade misses 89.6% at 4.3% FC (bootstrap 79-96%).
  - **The critics also found a flaw in the idea's own metric.** Most of the "subset optimism" comes from the 2 s window horizon: random 200-turn draws give 38-92% miss, and bootstrapping all 974 gives 44-85%. That claim is dropped. The benchmark is adopted only if the 6 s-horizon check in §5 passes.
- **Run FIRST after turn-v3 (each ≤ 30 min):**
  1. The wildcards-3 horizon and stability check (§5 item 1). CPU, labels only, under 3 min, so it can run before turn-v3 finishes.
  2. Re-score the turn-v3 checkpoint under the corrected protocol (§5 item 2), so that the v3 result is not judged against oracle-bound rows.
  3. Streaming tracks at NVIDIA's 0.32 s ultra-low-latency Sortformer setting (§5 item 3). Three rejected ideas (turn-under-noise-3, data-and-labels-2, architecture-fusion-3) were refuted partly by this setting, and it has never been run here.

## 2. Ranked survivors

| rank | id | title | judge score | novelty | feasibility | target row of the STAGE1 n=200 table | decisive test |
|---|---|---|---|---|---|---|---|
| 1 | wildcards-3 | Leak-free public speaker-aware eot-bench on CC-BY meetings (AMI + ICSI): per-pause FC, floor strata, deployable enrollment, full dev sets | 16.5 | pass, 3/5 | pass, 3/5 | Every row, re-issued at n=974 AMI dev (+ ICSI dev 477) with per-pause FC, floor strata and causal enrollment. Main target: "timeout on Sortformer streaming" (2640 ms / 38.4%) and the oracle rows, which get relabelled as multi-stream bounds. | 6 s-horizon rebuild plus a fixed operating point chosen on held-out meetings. Adopt only if (a) the paired bootstrap 95% CI of causal:25 minus oracle enrollment (stream-cascade miss) excludes 0 and is ≥ 10 points, and (b) open-stratum cascade miss stays ≥ 50% with ≤ 20% of misses caused by the horizon. Kill if open-stratum miss falls below 30%. |

## 3. Survivor in detail

### wildcards-3: A leak-free public speaker-aware eot-bench on CC-BY meetings (AMI + ICSI)

**One-liner.** GAME_CHANGER §3 lists "speaker-aware end-of-turn: no public baseline, build one". The generator found five independent ways the current AMI protocol misstates the tier-1 question. The fix is to correct them in the scorer and release turn windows, labels, cached Sortformer baselines and the scorer as a benchmark.

**Mechanism: the five protocol effects.** All were measured by the generator on cached AMI dev data (scratch scripts gen-wildcards/a1-a17).

1. **Floor leak.**
   - With oracle multi-speaker labels, one rule reaches P50 80 ms and 0% miss at 5% FC: "another speaker's slot has been on for ≥ 240 ms while the primary has been silent for ≥ 80 ms".
   - The "floor" turn label is defined by someone else taking the floor. Rows that use oracle multi-speaker activity are therefore multi-stream upper bounds, not single-channel speaker-awareness bounds.
   - The feasibility critic confirmed this in code: the turn head reads the other speakers' columns (heads/turn.py:35-37, 491-509).
   - Strata at n=974: 558 overlap endings, 206 fast switches, 210 floor-open endings. Floor-open ends are the only ones where a speaking agent should reply, and there the Sortformer cascade misses about 90% at 5% FC.
2. **Enrollment leak.** Binding the user's slot with oracle whole-turn labels, instead of causally, makes the cascade look much better: 38.4% vs 66.8% miss (n=200). Today `enroll_column` (scripts/eval_stage1.py:216, datasets/ext_tracks.py:58) uses the oracle primary over [onset, turn_end), and research/recipes/stage1_turn_v3.yaml trains with the same binding.
3. **Subset optimism (withdrawn).** Over all 974 dev turns: stream timeout inf / 75.5%, offline timeout 2000 ms / 44.1%. On n=200 the same rows are 2640 ms / 38.4% and 1760 ms / 32.1%. **The feasibility critic showed this is mostly a horizon effect, not a subset effect:**
   - Windows keep only 2 s after the end (ami.py:450-483, trail_sec=2.0). Post-end length is 25 frames for 5-95% of windows.
   - The n=974 stream-cascade curve goes (FC 11.9%, 2400 ms) → (6.6%, 2880 ms) → (3.4%, inf). The 5% FC threshold sits at the horizon, where miss jumps from about 40% to about 90%.
   - Across 30 random 200-turn draws: miss 38% to 92% (median 61%). Only 13% of draws give a finite P50. The official draw sits at the 7th percentile.
   - Bootstrap of all 974: 44-85%.
   - A genuine failure sits under this: the stream primary column comes back on after the end in 61% of turns.
4. **FC unit.**
   - LiveKit eot-bench and the 540 ms R1 gate count false cutoffs per mid-turn pause. `conversation.eot_bench` counts them per turn, and picks the threshold on the evaluated set (conversation.py:528-580).
   - At n=974 the stream cascade is inf / 75.5% per turn but 2560 ms / 30.5% per pause.
   - Minor inconsistency: script a11 counts pauses of ≥ 2 frames (160 ms), but the idea text says 100 ms.
5. **Hidden pauses.**
   - At timeouts ≤ 800 ms, about 50-60% of the stream timeout's cutoffs happen in runs that the word timings call speech but that are acoustically silent. Their median level is -13 dB relative to the speaker's speech; labeled hesitations sit at -10 dB.
   - An energy-corrected oracle timeout has 21-25% FC at 400 ms, not 14%. research/ICSI.md confirms the artifact from the label side.
   - Also measured: labeled turn ends are acoustically accurate (median offset 0 ms, p90 80 ms). Wall-clock scoring gives 2400 ms / 32.6%, against 2640 ms / 38.4% under the data-frame convention.

**Code sketch (proposed, not yet written):**
- `conversation.eot_bench`: `fc_unit='turn'|'pause'`, `strata=('overlap','fast','open')` with a talk-over rate on overlap and fast ends, and `emission='wall'|'data'`.
- `datasets/ami.py`: `act_source='words'|'energy'`, which gates primary activity by energy inside word spans.
- `scripts/eval_stage1.py`: `--n all` and `--enroll causal:k`.
- New `scripts/export_meeting_eot_bench.py`: AMI and ICSI dev/eval windows, labels, cached Sortformer offline and stream tracks, and a pinned scorer.

**Prior art and delta.**
- Prior art:
  - LiveKit eot-bench (https://github.com/livekit/eot-bench): 1:1 human-agent, per-pause FC, no speakers, data license unverified.
  - TurnBench (arXiv:2608.25218): dyadic, 30 h, gated.
  - MP-Bench (arXiv:2609.13076): synthetic ElevenLabs audio with 0% overlap; scores the agent's answer/stay-silent choice.
  - Speak or Stay Silent (arXiv:2603.11409): AMI, but text only, with pauses taken from transcripts, no latency or FC metric, CC BY-NC-ND.
  - arXiv:2606.17542: offline LLM turn-change prediction on AMI.
  - arXiv:2603.13379: the closest speaker-aware streaming detector. Two speakers, F1 and latency only, no benchmark released.
  - txya900619/turn-detection-benchmark: the reason round 1 rejected evaluation-1. No AMI or ICSI, no diarization conditioning, no per-pause split.
  - ModeratorLM (arXiv:2606.13544) and triadic VAP (arXiv:2507.07518): no benchmark released.
  - An arXiv API query for ICSI + turn-taking returned 0 results.
- **Delta:** nothing found scores streaming, acoustic, speaker-track-level dead air against per-pause FC on redistributable meeting audio, stratifies by floor state, or reports the floor leak and the slot-binding leak.
- **Why novelty is only 3/5** (novelty critic):
  - Per-pause FC is eot-bench's own definition.
  - Overlap/gap/switch strata are standard (Heldner & Edlund; VAP shift/hold).
  - The floor leak is close to true by definition.
  - Acoustically silent "speech" in word alignments is a known forced-alignment artifact.
  - The headline result is negative.
  - The web-search budget was exhausted, so a vendor-internal benchmark could have been missed.
  - ICSI's redistribution terms (believed CC-BY 4.0) must be checked before release.

**Expected gain.**
- No detector gets faster. The gain is decision validity.
- The deployable Sortformer-stream cascade goes from "2640 ms / 38%" to inf / 67-95% miss per turn, or inf / 89% per pause (n=974).
- The oracle rows become explicit multi-stream bounds: the 80 ms floor rule, and a pause-honest timeout of 1520-1600 ms at 5% per-pause FC, which matches eot-bench's plain-VAD level of 1600 ms.
- The target moves to floor-open ends, where every current system fails.
- Judge: "Recommend doing it as internal scorer hygiene this quarter. Release it publicly only once there is a model worth putting on the leaderboard." Another judge comment says multi-party floor changes are a weak stand-in for one primary user talking to an agent with bystanders present.

**Test.**
- **Mac test (feasibility critic's rewrite; replaces the generator's, which was "already met by construction").** One CPU process, `torch.set_num_threads(1)`, under 3 min. Inputs: gen-wildcards/dev_rows.pkl plus AMI labels. No model load.
  - Horizon check:
    - Rebuild the dev windows with trail_sec=6.0, keeping the rule that stops a window when the same speaker resumes.
    - Compute oracle-primary, any-speaker and energy-corrected timeouts, per turn and per pause, for all ends and for the open stratum.
    - Report the share of right-censored latencies at 2 s and at 6 s.
    - Cascade rows have only 2 s tracks. Split their misses into "primary column re-activates" and "horizon reached".
  - Stability check:
    - A 1000-resample bootstrap 5-95% range for every cell.
    - A fixed operating point: the 5% per-pause-FC timeout chosen on train turns (or on one leave-meetings-out half of dev), then applied unchanged to the other half.
  - Adopt only if both (a) and (b) in §2 hold. Kill if open-stratum miss falls below 30%.
- **Release gate (novelty critic).**
  - Score at least four systems under both the naive and the corrected protocol, on AMI dev+eval and ICSI dev+eval: a 1440 ms timeout on Sortformer stream tracks, Sortformer + our head with causal k-second enrollment, Smart Turn v3.2 + Silero on the primary track, and NVIDIA EOU-120M.
  - Pass only if, on BOTH corpora, the corrected protocol flips at least one pairwise ranking or moves a headline operating point by ≥ 300 ms or ≥ 10 miss points. Use cluster-bootstrap CIs by meeting.
  - Report the floor-leak row separately, labelled as an upper bound.
  - Ship checksums and a license manifest.

## 4. Rejected ideas (all 20)

"Novelty" means the novelty critic failed the idea, "feasibility" the feasibility critic, "both" both of them.

| id | title | failing critic | one-line reason + citation |
|---|---|---|---|
| turn-under-noise-1 | Clock-in-the-loop: differentiable CUSUM stopping head initialised at the silence timeout, trained on eot-bench's objective over streaming Sortformer tracks | novelty | InDiD already trains recurrent detectors end to end with a differentiable first-passage delay loss that equals the proposal's E[delay]; arXiv:2106.02602. Feasibility passed, but the critic expects a smaller gain than claimed and called the test too weak. |
| turn-under-noise-2 | Un-capture the user's slot: turn-local voice audit in the streaming diarizer's own embedding space | novelty | Streaming, training-free remapping of local EEND speakers against embedding centroids is diart; Streaming Sortformer's AOSC already caches speaker embeddings; arXiv:2109.06483, 2507.18446. The diagnosis (next speaker captured in the user's slot) is worth logging. Feasibility passed at 3/5, with about 20% odds of working. |
| turn-under-noise-3 | Identity at VAD latency: 160 ms voice-continuity head trained on AMI hesitation-resume vs speaker-switch pairs | both | The 720 ms diarizer lag it would bridge can be removed with NVIDIA's shipped 0.32 s config (chunk 3, rc 1, DER about the same); the claimed gain does not follow from the mechanism; https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2, runs/stage1_turn_on_sortformer_eval_n200.json. |
| vap-and-projection-1 | Voice-anchored user projection: check the diarizer's user slot at speaker changes with its own embeddings | novelty | Each part is established: EEND-VC resolves slots with internal embeddings, diart matches centroids, and IRAF gates user evidence with target embeddings for full-duplex turn-taking; arXiv:2010.13366, 2109.06483, 2606.06559. Feasibility passed only with a rewritten test. |
| vap-and-projection-2 | First-passage projection readout: train the EOT hazard on eot-bench's stopping cost, not per-frame BCE | novelty | ELECTS already uses a d_t·Π(1−d_s) stopping distribution with a differentiable expected-cost loss and no RL; arXiv:1901.10681 (method section checked). Feasibility confirmed the math is sound. |
| vap-and-projection-3 | Clock-relative projection: nowcast the streaming diarizer's pending frames on the ASR's 160 ms clock | both | This is offline-to-streaming self-distillation (dual-mode ASR); the new part is too small to measure, and the idea's own kill rule already fires on the emission-timing arithmetic (emit 7-12 frames, mean 760 ms); arXiv:2010.06030, scripts/eval_stage1.py `_stream_emit`. |
| semantics-1 | Speaker-routed transcript state: "self" and "other" text streams from one mixture ASR decode, split by the diarizer | novelty | NVIDIA NeMo's SpeakerTaggedASR already slices streaming Sortformer frames at word emission frames; https://github.com/NVIDIA/NeMo (multispk_transcribe_utils.py), arXiv:2312.11123. Feasibility passed, but its test confounds routing with decoded_prob, and the recipe never sets decoded_prob. |
| semantics-2 | Floor-claim semantics as a competing risk: a claimant's first words predict that the primary will yield | both | Classifying cooperative vs competitive overlap on AMI is old (Truong 2013), and under turn_def="floor" the label and the signal are the same thing (ami.py:319-343); Interspeech 2013 truong13. |
| semantics-3 | Completion semantics only where the floor is open: a hold/release probe on the frozen PredictionNet, trained text-only on 171 AMI meetings | novelty | The negative arm is already mostly known: LLMs fall below humans on AMI turn-change and multi-party speak/stay-silent; arXiv:2606.17542, 2603.11409. Feasibility passed, but the ceiling is low and it is not a tier-1 gain. |
| data-and-labels-1 | Close the text channel that leaks the label: train the text state on the frozen RNNT's own streaming decode | novelty | The fix already exists in this repo (decoded_prob, text_delay, text_noise in turn.py:518-530; recommended in TURN_ABLATION.md:150-220). The leak is real: turn_align_greedy_frac is 0.0001, so tokens are timed from spk_act. The closest precedent says the best outcome is matching a head with no text. |
| data-and-labels-2 | Availability-true teacher tracks: train and score on the Sortformer track as it exists at each 160 ms decision | novelty | NVIDIA ships 0.32 s and 80 ms streaming configs, so the 1040 ms buffer is a choice; delayed-observation state augmentation is established; HF diar_streaming_sortformer_4spk-v2, Nemotron-3-Diarization, arXiv:2005.05440. Feasibility passed: the mechanism is causal. |
| data-and-labels-3 | Voice-diverse, bleed-matched switch data with real-voice speaker changes | both | UAF already builds target-anchored 2-4 speaker data from single-speaker corpora, and SE-DiCoW flips conditioning masks; the critic re-measured and found most of the failure is not bleed; arXiv:2604.19221, 2601.19194. |
| product-semantics-1 | Hold-back verification: use the diarizer's latency window to veto end-of-turn firings | both | LiveKit already cancels a pending EOT on VAD speech start during the endpointing delay, and Speechmatics resets it on speaker start; the claimed gain did not hold beyond the proposer's 32 turns (duration rule inf / 68.4%); livekit/agents audio_recognition.py, speechmatics-python-sdk. |
| product-semantics-2 | Speaker-attributed speculative drafting, scored as end-to-first-audio | both | Speaker-gated speculation already ships (Deepgram Flux EagerEndOfTurn, Speechmatics SpeakerFocus, LiveKit preemptive generation); at n=200 the payoff comes from how the comparison is paired, not from speaker awareness; developers.deepgram.com/docs/flux/voice-agent-eager-eot. |
| product-semantics-3 | Stale-track turn head: train on streaming tracks exactly as known at each wall-clock frame | novelty | Delay-aware conditioning on a stale input plus its age is V2X-ViT's delay-aware positional encoding, and delayed-MDP state augmentation is older; arXiv:2203.10638, Katsikopoulos & Engelbrecht 2003. Feasibility passed only with a refined test: the current scorer gives P50 = inf for every streaming arm. |
| architecture-fusion-1 | Speaker-attributed token events: tag each frozen-RNNT token with the diarizer slot active at emission | novelty | NeMo SpeakerTaggedASR maps token frames onto streaming Sortformer output, and FireRedChat gates ASR with pVAD before semantic EOT; NeMo multispk_transcribe_utils.py, arXiv:2509.06502. Feasibility passed narrowly (3/5) as an engineering fix. |
| architecture-fusion-2 | Slot-purity verifier: cross-attention from the current frame into the Sortformer speaker cache | both | Frame-level identity checks against self-built profiles exist (online TS-VAD, arXiv:2603.13379), and the verifier's memory is built from the same mixed slot it is meant to check; the test hid this with an oracle-onset memory; arXiv:2207.05920. |
| architecture-fusion-3 | Stitched Sortformer: NVIDIA's 18-layer Sortformer driven by our frozen 160 ms ASR encoder through a small adapter | both | Model stitching and SimKD are established; the latency gain belongs to NVIDIA's own ultra-low-latency setting, not to stitching; arXiv:2203.14001, 2106.07682; HF Sortformer v2 card. It also needs checkpoints over 50 MB. |
| wildcards-1 | Deployable slot binding: re-run head-vs-cascade with causal enrollment, and train v3 on causally bound tracks | novelty | Causal first-heard user binding is published (arXiv:2603.13379). Feasibility passed: the repo needs this fix (the oracle [onset, turn_end) binding in ext_tracks.py:58 and eval_stage1.py:216/332), and wildcards-3 absorbs it as its enrollment leak. |
| wildcards-2 | Near-field-first speaker awareness from AMI's synchronized headset and array channels | both | Krisp already evaluates AMI headset recordings + Silero VAD for turn-taking; at g=1 the test is circular, because AMI labels were aligned on each speaker's headset; krisp.ai BVC blog post, Wrigley et al. 2005. |

## 5. Prototype queue (ordered, ≤ 5)

These are the synthesizer's sketches built from the critics' findings. Items 2-5 need small code changes that do not exist yet, so none has been run. None is a new tier-1 method. Item 1 decides whether the others are scored under the old or the new protocol.

1. **wildcards-3 horizon and stability check (Mac, CPU, < 3 min, no model).**
   - Recipe: one script in `scratchpad/wf-ideas2/synthesize/`. It loads `gen-wildcards/dev_rows.pkl` and AMI labels and calls `AMI.turn_examples(window_sec=20, lead_sec=4.0, trail_sec=6.0)` from labels only. The same-speaker-resume stop stays.
   - It computes timeouts (oracle primary, any speaker, energy-gated primary) per turn and per pause for all ends and for the open stratum. It splits the cascade's 2 s-track misses into re-activation vs horizon.
   - Fixed operating point: the 5% per-pause-FC timeout from one leave-meetings-out half of dev, applied to the other half. 1000-sample bootstrap.
   - **Pass:** (a) causal:25 minus oracle enrollment, stream-cascade miss: 95% CI excludes 0 and is ≥ 10 points; (b) open-stratum miss ≥ 50% with ≤ 20% of misses from the horizon.
   - **Kill the benchmark as specified:** open-stratum miss < 30% at 6 s.
2. **Re-score turn-v3 under the corrected protocol (GPU/MPS eval, ≤ 30 min, after `runs/stage1_turn_v3.afm` exists).**
   - Code: add `--enroll causal:25` to `scripts/eval_stage1.py`, replacing `enroll_column`'s oracle [onset, turn_end) with argmax over [onset, onset+25) as in a17.py. Add `--n all` (974), and add `fc_unit='pause'` plus strata to `conversation.eot_bench`.
   - Run: `.venv/bin/python scripts/eval_stage1.py --ckpt runs/stage1_turn_v3.afm --tasks turn --n 974 --enroll causal:25 --fc-unit pause --out runs/stage1_turn_v3_eval_n974_causal.json`, with the cached stream tracks through `--diar-cache`. Report the naive protocol alongside.
   - **Pass (tier-1 signal):** v3 + streaming Sortformer beats the timeout on the same stream track at the fixed operating point from item 1, on all ends and on the open stratum, with a meeting-cluster bootstrap CI excluding 0. Reference at n=200 naive: head 69.0% vs cascade 38.4%.
   - **Fail:** v3 is no better than the cascade under the corrected protocol, even though v3 trained with oracle binding (so causal enrollment is a train/test mismatch for it). If so, go to item 4.
3. **Ultra-low-latency streaming tracks, 0.32 s (GPU/MPS, dev only, about 5-10 min).**
   - Code: add a `SORTFORMER_ULTRA_LOW` config (chunk_len 3, chunk_right_context 1) next to `SORTFORMER_LOW_LATENCY` in `scripts/make_sortformer_tracks.py` and `scripts/eval_stage1.py`. The two copies are checked for equality in tests, so change both. Add a track source `stream_ul` in `datasets/ext_tracks.py`.
   - Run: `make_sortformer_tracks.py --split dev --track-source stream_ul --device mps`, then score the cascade and v3 under item 2's protocol.
   - By `_stream_emit`, emission drops from 7-12 frames (mean 760 ms) to 2-4 frames (160-320 ms).
   - **Pass:** stream-cascade miss at the fixed operating point is within the bootstrap CI of the 1040 ms tracks while emission delay falls by about 500 ms. That would make the 0.32 s tracks the default for training data.
   - **Fail:** miss rises by ≥ 10 points; then keep 1040 ms.
4. **Retrain v3 on causally bound streaming tracks (wildcards-1's arm; GPU, only if item 2 fails on the enrollment mismatch).**
   - Recipe: `research/recipes/stage1_turn_v3.yaml` with `spk_prim_ext` computed by causal:25 binding, not oracle overlap on [onset, turn_end). Everything else unchanged (2000 steps, batch 6, head only).
   - The v3 ablations are 2000-step single-GPU jobs. Whether one fits in 30 min is not established here; check against the turn-v3 wall-clock time.
   - **Pass:** item 2's pass condition met by the retrained head.
   - **Fail:** head miss ≥ cascade miss on the open stratum. At that point the speaker-aware EOT claim on AMI stays unsupported (STAGE1.md verdict).
5. **Text-channel leak arm (data-and-labels-1 / semantics-1; engineering, not tier 1).**
   - Recipe: v3 plus `heads.turn.align=greedy text_delay=2 text_noise=0.1 decoded_prob=0.3` (as in research/recipes/speaker_aware_turn.yaml), plus a no-text arm. Two 2000-step head-only runs.
   - **Pass:** the decoded-text arm is at least as good as the no-text arm under item 2's protocol. The critic's precedent says matching is the best case.
   - **Fail:** worse than no text; then drop the text channel from the deployable head.
