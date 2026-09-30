# Turn head v5: a segment end-of-turn classifier on the frozen encoder (2026-09-30)

**Verdict.** The balanced ship bar was missed. The `fast` bar was met, so turn head v5 ships as the new
`--turn-preset fast`. It also ships as a new `--turn-preset assistant` for speech directed at the agent. `balanced`
stays the default.

- **The balanced bar was not met.** It asks for ≥ 150 ms faster than 956 ms at ≤ 20.2 % FI / ≤ 7.3 % missed, with
  AMI ≤ 10.5 / 33.5 %.
  - No rule with any v5 model meets it without cutting the bundled demo clip.
  - Allowing the cut, the best under the calls FI cap still misses AMI (e.g. 746 ms, 22.0 / 7.3, AMI 10.5 / 33.5,
    cuts 4 of 6 deliveries, c4).
- **The fast bar was met** (≥ 300 ms faster at ≤ 25 % FI). The v5 `fast` preset gives calls 547 ms p50 (−409 ms)
  at 24.8 % FI and 5.5 % missed, with AMI 1247 ms at 11.5 / 34.0 %. It does not cut the demo clip under any of the
  six deliveries.
  - Session bootstrap of the calls p50 difference: −385 ms, 90 % interval −480 to −190.
  - Audible-end labels: 629 ms, 23.9 / 4.6 %, −320 ms.
  - A second training seed with the same rule: 586 ms, 26.6 / 5.5 %. Its own best rule at ≤ 25 % FI: 580 ms,
    24.8 / 7.3.
- **The assistant preset** was measured on smart-turn's 399 assistant-directed test clips (research/EOT_ASSISTANT.md
  protocol):
  - 92.2 % accuracy, 292 ms p50, 5.4 % false fires on incomplete clips.
  - The served dump gives 93.0 % and 317 ms; it includes chunk compute measured under load.
  - Seed 2: 92.0 %, 323 ms.
  - For comparison: the smart-turn bridge 95.7 % / 770 ms, Pipecat 69.7 % / 211 ms, LiveKit 72.7 % / 547 ms.
- **The old `fast` rule stays** as `--turn-preset steady`: 886 / 1434 ms, 24.8 / 3.7 %, the best p95 and misses.
- **What moved early confidence.** v5 is a segment classifier: bidirectional attention inside the last 8 s of a
  frozen encoder block, read when the VAD goes quiet. It was trained on ends vs in-turn pauses vs smart-turn style
  cuts.
  - On the quiet frames where the rule actually decides, it is the first of our turn models that ranks a turn end
    above an in-turn pause.
  - Calls frame AUC on quiet frames: 0.37 for the shipped v2 head, 0.30 for v4's h2, 0.73 for v5 (c5).
  - AMI: 0.46 / 0.54 / 0.87.
  - The capacity, text, prosody and distillation ingredients each moved it by ≤ 0.04 (ablation table). The encoder
    block (8 over 4 on calls) and the segment formulation are what count.
- **It is still not confident on human calls.** Median p is 0.53 at call ends vs 0.19 in pauses, so it ranks well
  but reaches p ≥ 0.9 on only 1-6 % of call ends. The `fast` rule therefore fires at p > 0.7.
  - That is also why the balanced bar stays out of reach. A threshold that keeps FI ≤ 20 % leaves too many ends to
    the 640 ms fallback.

Code: `scripts/research/turn_v5.py`, `audioforge/heads/turn_seg.py`, `audioforge/heads/prosody.py`. Numbers:
`runs/turn_v5.json`. Served: `stage1_served_v3.afm` / `assets/served_heads_v0.3.pt`, `TURN_PRESETS["fast" |
"assistant" | "steady"]`, `ASRStream.attach_seg`, `VadHeadPolicy(model_reask=...)`. Tests: `tests/test_turn_seg.py`,
`tests/test_turn_preset.py`. Scratch (caches, checkpoints, dumps, scans): `scratch/turn_v5/`.

## Final table

EOT total p50 / p95 ms, FI % / missed %. Calls = the 32 two-party user-channel sessions (109 ends). AMI = the 200
eot-bench v2 turns. Assistant = smart-turn's 399 v3.2-test clips + 3 s of silence (accuracy = complete answered
without an early fire + incomplete not fired within 3 s). All our rows run the served `VadHeadPolicy` with the energy
gate (default since 1b39952), replayed offline on the dumped frames. The served engine equals the replay on the
bundled clip (every preset, six deliveries) and on the 399 assistant clips.

| system | calls | AMI | assistant acc / p50 / FF | demo clip |
|---|---|---|---|---|
| balanced (default, unchanged) | 956 / 1919, 20.2 / 7.3 | 1326 / 3758, 10.5 / 33.5 | 43.9 % / 1218 / 100 % | no cut |
| **fast = v5** (c5): classifier after 80 ms of VAD < 0.6, re-asked at every quiet frame, P > 0.7; OR 640 ms; others 960,640 | **547 / 1862, 24.8 / 5.5** | 1247 / 4047, 11.5 / 34.0 | 41.6 % / 453 / 100 % | **no cut (6 / 6)** |
| fast rule, second seed (c5s1) | 586 / 1835, 26.6 / 5.5 | 1326, 12.0 / 33.5 | 41.9 % / 451 / 100 % | no cut |
| fast, audible-end labels | 629 / 1862, 23.9 / 4.6 | 1247, 11.5 / 34.0 | – | no cut |
| steady (the pre-v5 fast) | 886 / 1434, 24.8 / 3.7 | 1086 / 3886, 16.0 / 30.5 | 43.9 % / 1156 / 100 % | no cut |
| **assistant = v5**: after 240 ms of energy-or-VAD quiet, P > 0.9; OR 2960 ms | 3232 / 4802, 5.5 / 33.9 | 1647, 7.0 / 52.5 | **92.2 % / 292 / 5.4 %** (served 93.0 % / 317) | no cut |
| assistant rule, second seed | – | – | 92.0 % / 323 / 5.4 % | – |
| `--turn-model smartturn` bridge (VAD trigger, 3 s), research/EOT_ASSISTANT.md | 629, 19.3 / 23.9 | 1087, 13.0 / 36.5 | 95.7 % / 770 / 5.8 % | no cut |
| Pipecat smart-turn v3.2 + Silero (defaults) | 237 / 3217, 35.8 / 24.8 | 385, 27.5 / 44.5 | 69.7 % / 211 / 40.2 % | – |
| LiveKit EnglishModel + Silero (defaults) | 567 / 3127, 26.6 / 22.9 | 1890, 12.5 / 67.5 | 72.7 % / 547 / 44.6 % | – |

- The assistant column needs a ≥ 3 s timer: any shorter fallback fires on nearly every incomplete clip
  (research/EOT_ASSISTANT.md). That is why `fast` and `balanced` score ~42 % there. One model serves both domains,
  at two operating points.
- **Compute:** the classifier reads the block-8 frames the ASR pass already computed.
  - 1.0-1.7 ms per call on the Mac CPU (2 threads).
  - Interleaved runs on the demo clip: 114 calls in 16 s, chunk p50 49.2 → 50.5 ms (balanced → fast, same load).
  - The old turn head still runs, since the hint and balanced read it.
  - The model is 9.8 MB of the 29.4 MB heads file.

**Head reach** (p ≥ θ on a decision-ready frame within 200 / 300 / 500 ms of the true end; original labels) and
calibration:

| model | calls p ≥ 0.5 | calls p ≥ 0.9 | calls p ≥ 0.95 | AMI p ≥ 0.9 | AMI p ≥ 0.95 | calls quiet-frame AUC | AMI quiet-frame AUC | calls median p: quiet in-turn / pause / end (first 320 ms) |
|---|---|---|---|---|---|---|---|---|
| shipped v2 head | 76.1 / 77.1 / 77.1 | 56.0 / 57.8 / 59.6 | 48.6 / 50.5 / 52.3 | 25.0 / 26.5 / 35.5 | 19.5 / 22.5 / 25.0 | 0.37 | 0.46 | 0.98 / 0.98 / 0.95 |
| v4 h2 (per-frame, research/TURN_V4.md) | 22.9 / 30.3 / 37.6 | 3.7 / 5.5 / 9.2 | 1.8 / 1.8 / 2.8 | 9.5 / 11.5 / 19.5 | 5.0 / 5.5 / 7.5 | 0.30 | 0.54 | 0.67 / 0.74 / 0.22 |
| v5 c2 (block 4) | 43.1 / 48.6 / 57.8 | 0.0 / 0.9 / 2.8 | 0 / 0 / 0 | 30.0 / 33.5 / 44.0 | 18.0 / 18.5 / 30.5 | 0.68 | 0.88 | 0.18 / 0.14 / 0.40 |
| **v5 c5 (block 8, shipped)** | 62.4 / 64.2 / 73.4 | 0.9 / 2.8 / 5.5 | 0.9 / 0.9 / 0.9 | 36.0 / 38.0 / 45.0 | 20.5 / 23.0 / 32.5 | **0.73** | 0.87 | 0.23 / 0.19 / 0.53 |
| v5 c5s1 (seed 2) | 64.2 / 66.1 / 75.2 | 1.8 / 1.8 / 5.5 | 0 / 0 / 0.9 | 30.5 / 35.0 / 44.0 | 16.0 / 20.5 / 31.5 | 0.69 | 0.86 | 0.29 / 0.22 / 0.55 |

"Quiet-frame AUC": frames of the first 320 ms after each scored end against frames inside the turn where the served
VAD is < 0.4. These are the only frames on which the rule can act.
- The shipped head is high on every quiet frame, and higher in pauses than at ends (AUC < 0.5). Its 50 % reach at
  0.95 is the same p it gives pauses.
- v5 is calibrated by temperature on the held-out split, where the turn ends are separable. On calls it is sure
  only on the easy ends, so its p ≥ 0.9 reach is low, but it keeps pauses low (0.19).

**Smart-turn's 399 test clips** (turn_v4 protocol: p on the frame 160 ms after the clip's last VAD speech frame; the
clips were never trained on):

| model | accuracy at 0.5 | complete / incomplete recall | AUC |
|---|---|---|---|
| smart-turn v3.2 (ONNX, runs/smartturn_audit.json) | 97.0 % | 96.6 / 97.3 % | – |
| shipped v2 head | 63.2 % | – | – |
| v4 h1 (best v4) | 84.0 % (θ from train) | – | – |
| v5 c2 | 98.5 % | 97.7 / 99.1 % | 0.999 |
| **v5 c5 (shipped)** | **99.0 %** | 99.4 / 98.7 % | – |
| v5 c5s1 | 99.2 % | 99.4 / 99.1 % | – |
| v5 without text (a_notext) | 95.2 % | 96.0 / 94.6 % | – |

## Ablations

Every v5 model is a segment classifier: 100-frame window, attention pooling, smart-turn's MLP. The base recipe is
block 4, 2 bidirectional Transformer layers of d 256, text on, smart-turn aux 0.5, LM-completeness aux 0.3, lr 1e-4,
dropout 0.2, 3 000 steps, batch 256, the smart-turn v3.2 English data off. Val = the turn_v4 held-out 8 %
(conversations / meetings / clips), never the eval sets. Calls / AMI columns: the operating points of each model's own
scan (1 958 rules of the served policy).
- "fi25 no cut": the fastest calls p50 at ≤ 25 % FI and ≤ 7.3 % missed without cutting the clip.
- "fi25 + AMI goal": the same with AMI ≤ 10.5 / 33.5 but ignoring the clip.
- "assistant ≥ 90 %": the fastest assistant p50 at ≥ 90 % accuracy.

| model | change | params | val AUC (all / first 320 ms) | calls / AMI quiet AUC | smart-turn test | calls fi25 no cut | calls fi25 + AMI goal (clip ignored) | assistant ≥ 90 % |
|---|---|---|---|---|---|---|---|---|
| probe_b17 / b12 / b8 / b4 / pass-2 | acoustics only, 2 500 clips, 1 000 steps | 1.9M | 0.72 / 0.77 / 0.79 / 0.81 / 0.73 (peak) | – | – | – | – | – |
| c1 | block 4, lr 3e-4 (overfits after 500 steps), no LM aux | 2.46M | 0.804 / 0.782 | 0.68 / 0.90 | 97.7 % | 844, 23.9 / 7.3 | – | 92.0 % / 308 ms |
| c2 | the base recipe | 2.46M | 0.812 / 0.791 | 0.68 / 0.88 | 98.5 % | 944, 16.5 / 7.3 | 618, 23.9 / 7.3 (cut 6) | 91.5 % / 292 ms |
| a_notext | − text | 1.9M | 0.815 / 0.794 | 0.70 / 0.88 | **95.2 %** | 876, 22.9 / 7.3 | 651, 24.8 / 7.3 (cut 5) | 92.5 % / **387 ms** |
| a_nost | − smart-turn aux | 2.46M | 0.813 / 0.793 | 0.68 / 0.88 | 98.0 % | 911, 17.4 / 7.3 | 736, 22.0 / 7.3 (cut 5) | 90.5 % / 322 ms |
| a_nocompl | − LM-completeness aux | 2.46M | 0.811 / 0.790 | 0.69 / 0.88 | 98.5 % | 946, 16.5 / 7.3 | 618, 24.8 / 7.3 (cut 6) | 91.2 % / 292 ms |
| c3 | + prosody (12 features / 80 ms) | 2.46M | 0.815 / 0.789 | **0.65** / 0.86 | 98.2 % | 896, 24.8 / 5.5 | 606, 23.9 / 7.3 (cut 6) | 92.7 % / 291 ms |
| c4 | + smart-turn v3.2 English train (6 256 clips) | 2.46M | 0.809 / 0.786 | 0.68 / 0.86 | 97.0 % | 926, 16.5 / 7.3 | 746, 22.0 / 7.3 (cut 4) | 90.0 % / 292 ms |
| a_pool | − Transformer (attention pooling only) | 0.9M | 0.785 / 0.770 | 0.65 / 0.85 | 97.7 % | 881, 24.8 / 4.6 | 716, 22.9 / 7.3 (cut 4) | 90.2 % / 291 ms |
| a_big | 4 layers | 4.0M | 0.815 / 0.791 | 0.70 / 0.90 | 98.2 % | 916, 17.4 / 7.3 | 629, 24.8 / 7.3 (cut 6) | 91.7 % / 322 ms |
| **c5** | **block 8** | 2.46M | 0.817 / 0.793 | **0.73** / 0.87 | **99.0 %** | **547, 24.8 / 5.5** | (547, 24.8 / 5.5, AMI 11.5 / 34.0) | 92.2 % / 291 ms |
| c5s1 | block 8, seed 2 | 2.46M | 0.809 / 0.787 | 0.69 / 0.86 | 99.2 % | 580, 24.8 / 7.3 | – | 92.0 % / 322 ms |
| c23 | mean of c2 and c3 | – | – | – | – | 944, 20.2 / 7.3 | 593, 22.9 / 7.3 (cut 6) | 90.0 % / 290 ms |

What each ingredient did:
- **Segment formulation + cuts: the step change.**
  - Per-frame heads (shipped v2, v4 h1-h5) learned "silence means end": their quiet-frame AUC is 0.30-0.37 on calls.
  - Classifying the window at a quiet frame, with smart-turn style cut negatives, gives 0.65-0.73.
  - The ends-vs-cuts AUC is 0.88-0.91.
- **Encoder block.** In the acoustic probe block 4 ranks best on val (0.81; block 17 0.72, the speaker-conditioned
  pass-2 top 0.73).
  - With text, block 8 is the best on calls (quiet AUC 0.73 vs 0.68). It is also the only one whose fast operating
    point does not cut the demo clip.
  - Its second seed is at 0.69, so the block-8 margin is partly seed noise. The calls p50 gain holds on both seeds
    (547 / 586 ms).
- **Capacity.** Dropping the Transformer (a_pool, 0.9M) costs 0.03 val AUC and 0.03 calls AUC. 4 layers (a_big,
  4.0M) gains 0.02 calls / AMI quiet AUC and nothing on val or the operating points. The 2-layer 2.5M model was
  kept.
- **Text** (RNNT tokens + a token Transformer + time since the last token):
  - No gain on conversational val/calls (a_notext 0.815 / 0.70 vs 0.812 / 0.68).
  - +3.3 points on smart-turn's clips (95.2 → 98.5 %).
  - Makes the assistant preset 95 ms faster at ≥ 90 % (387 → 292 ms).
- **Smart-turn v3.2 distillation.** 168 547 calls (0.94 CPU-h, Pipecat's own features and ONNX, 7 offsets around
  every end, half the pauses, every cut).
  - Its own end-vs-pause AUC on our training conversations is 0.61 (oto), 0.52 (AMI), 0.62 (ICSI), against 0.998 on
    its own clips. So as a teacher for conversation it carries little.
  - Removing it (a_nost) changes calls/val by ≤ 0.004 and the assistant preset by −1 point / +30 ms.
- **LM completeness** (Qwen2.5-7B-Instruct 4-bit via mlx, Apache-2.0, local; its log P(sentence-final punctuation |
  words so far) under a transcript prompt, one causal pass per clip, 8 454 clips in 10 min, calibrated to P(end) by a
  logistic fit on the training ends vs pauses):
  - no measurable effect (a_nocompl within 0.005 everywhere);
  - kept at weight 0.3 because it costs nothing at serve time (training-only target).
- **Prosody** (log-energy contour, YIN F0 against a running speaker mean, F0 slope, last voiced run vs its running
  mean; `audioforge/heads/prosody.py`, 0.23 ms per 80 ms frame on the CPU, streaming equal to offline):
  - worse on calls (quiet AUC 0.65 vs 0.68);
  - equal elsewhere;
  - not shipped.
- **Smart-turn v3.2 English train data.** It was confirmed CC BY 4.0 on the HF card, modified 2026-09-29. 8 of 83
  shards were streamed and filtered to English: 6 256 clips; the rest was discarded, and the full set is 41 GB.
  - No gain (c4). Our human_5_all train clips already bring smart-turn's own test accuracy to 98-99 %.
- **Warm start.** The v5 architecture shares nothing with the 0.32M GRU head, so every model was trained fresh.
  The shipped GRU cannot initialise it; v4 showed that fine-tuning it with this objective loses its calibration.

## Method

**Data** (all local except the smart-turn v3.2 shards, 3.9 GB streamed; licences as in research/TURN_V4.md §2):
- oto 2 974 windows of 40 s (188 conversations; the 16 eval conversations excluded by id).
- AMI train 900 and ICSI train 885 turn windows (12 + 12 meetings).
- smart-turn human_5_all train: 3 463 clips, BSD-2, twice.
  - Once with 1.2 s of −70 dBFS noise and the clip as the voice print (the turn_v4 cache).
  - Once as the assistant benchmark feeds clips: + 3 s digital silence, no print (`st3`).
- 7 626 smart-turn style cuts (`cutplan` / `extract --set cuts`).
  - Each is a training utterance cut inside running speech at a 10 ms energy minimum ≥ 10 dB below the turn's
    median, with served VAD speech on both sides (a word-boundary proxy).
  - It is padded with 640 ms of the clip's own quietest audio, then run through the full served-input extractor, so
    the classifier sees a real mid-sentence stop followed by room tone.
- smart-turn-data-v3.2-train English: 6 256 clips (c4 only).
- TurnBench dev: evaluation only.
- Its 399 v3.2-test clips (human_5_all) are held out everywhere.

**Cache** (scratch/turn_v5, on the SSD):
- Pass-1 blocks 4 / 8 / 12 / 17 of every clip (14 GB fp16; `blocks`, 10 min on MPS). The turn_v4 cache supplies
  the served VAD, TS-VAD P(user) / P(other) and the RNNT text.
- Smart-turn targets (`st/`), transcripts (`texts.jsonl`), LM scores (`lm/`), prosody (`pros/`).
- Frame energies at each frame's decision-ready time, as `Session._energy` reads them (`energy/`).

**Samples and objective:**
- A sample is a (clip, frame) pair:
  - positives: the first 12 frames after an audible end, and every frame after a complete smart-turn clip;
  - negatives: every frame of an in-turn pause ≥ 160 ms, quiet dips inside turns, the first 8 frames after a cut,
    and every frame after an incomplete smart-turn clip.
- Batches are drawn per kind (ends 1, pauses 1.5, dips 0.5, cuts 0.7, smart-turn 0.5).
- The loss is BCE with smart-turn's per-batch pos_weight, plus smart-turn's soft target at its call frames (0.5) and
  the LM completeness from the text branch (0.3).
- P(user) / P(other) are zeroed on 25 % of samples, so an unenrolled session (the assistant benchmark, no print)
  reads zeros.
- Checkpoint: best val AUC. Calibration: temperature + bias on the val split.

**Evaluation:**
- `evaldump` computes p on every frame of the 232 eval sessions from the cached served inputs. The same windows,
  tokens and features equal the served `SegTurnStream` to ≤ 3.5e-4 (`served_v5`: the demo clip × 6 deliveries,
  tb_114, an AMI session).
- `scan4` replays the served `VadHeadPolicy` + `EnergyGate` (the classes the server runs; without the gate it
  reproduces the shipped 956 ms / 20.2 / 7.3 exactly).
  - Families: the head family (v5 p as the head's p), and the classifier family (turn_model: trigger 80-240 ms on
    VAD < 0.4-0.6, optionally energy-or-VAD, once per run or re-asked, P > 0.3-0.97, fallback 640-2960 ms, others
    path 640-960 ms).
  - It scores calls, AMI, the 399 assistant clips and the demo clip's six deliveries.
- `boot`: session bootstrap. `stest`: smart-turn's test clips. `served_presets`: the real engine on the demo clip
  for every preset (served == offline in all 24 runs). `asst_served`: the served assistant preset on the 399 clips
  (served == offline 399 / 399).

## Not done

- **VAD-head retrain with room-tone / session-start negatives.** The energy gate (1b39952) removed the pre-speech
  fires and the room-tone problem for the rule. The v5 classifier reads the VAD only as a trigger and an input
  channel, and the assistant preset's energy-or-VAD trigger already opens at room tone. A new VAD head would change
  every downstream consumer (TS-VAD track, ASR gate, LID, the turn heads' inputs) and need the VAD_SINGLE checks
  again. Not needed for the shipped presets.
- **Pipecat's pre-speech window.** The classifier always reads the last 8 s (masked before the session start). It
  does not read Pipecat's "turn start − 0.5 s" window, which depends on the previous decision.
- **Skipping the v2 turn head's pass-2** under fast / assistant (it only feeds balanced and the hint's fallback)
  would save most of a speaker-conditioned encoder pass per chunk. Left for a follow-up.
- **Per-frame variants as full candidates.** The per-frame line is covered by v4 (research/TURN_V4.md) and the
  shipped head, both in the reach table.

## Reproduce

```bash
P=scripts/research/turn_v5.py; G="PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python"
$G $P blocks --which eval; $G $P blocks --which train                     # repeat until 0 left
$G $P cutplan; $G $P extract --set cuts; $G $P extract --set st3; $G $P extract --set asst
$G $P smartturn; $G $P texts; $G $P prosody; $G $P energy_ready
(cd $W && scripts/dev/gate.sh venv_mlx/bin/python $P lm)                   # Qwen2.5-7B-Instruct-4bit, mlx
$G $P train --tag c5 --block 8 --steps 3000 --lr 1e-4 --dropout 0.2 --wd 0.05 --eval-every 250 --w-compl 0.3 --no-st32
$G $P evaldump --tag c5; $G $P report --tag c5; $G $P scan4 --tag c5 --keep-all
$G $P scan4 --tag c5 --keep-all --labels runs/turn_v4_labels_eval.json; $G $P boot --tag c5; $G $P stest --tags c5
$G $P build_v3 --tag c5; $G $P served_presets; $G $P served_v5 --tag c5 --keys tb_114.user
EOT_ASSISTANT_DUMP=$W/asst_dump_assistant EOT_ASSISTANT_ENGINE='{"afm": "runs/stage1_served_v3.afm", "turn_preset": "assistant"}' \
  $G scripts/research/eot_assistant.py dump --budget 540; $G $P asst_served
```

`W=/Volumes/ExternalSSD/nvidia-audio-models/scratch/turn_v5`.
