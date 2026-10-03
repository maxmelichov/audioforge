# Changelog

All notable changes to audioforge. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Every measured number cited here has its source in
`runs/*.json` (see `research/FINAL_REPORT.md`, Appendix A).

## [Unreleased]

### Added (2026-10-03): optional perceived voice-gender head, both cores (research/VOICE_GENDER.md)
- `serve --voice-gender head|PATH` (off by default, never in a default install): probabilities for the two classes
  the training data annotate, female voice / male voice, on each streaming `final` (`voice_gender`: that segment's
  speech) and in `stats` (the session's). A perceived vocal characteristic estimated from audio, not a person's
  gender identity; it can be wrong for any individual.
- Causal attentive statistics pooling (the ECAPA / TitaNet / AmberNet decoder) + a linear classifier on the speaker
  head's tap (115M block 4, 0.6B block 5; a block probe shows them on the plateau). Sized by an equal wall-clock
  sweep (hidden 16-256): `assets/voice_gender_115m.pt` (hidden 32, 21.8 K parameters) and
  `assets/voice_gender_0p6b.pt` (hidden 16, 20.6 K), hub OPTIONAL components `voice_gender` / `voice_gender_0p6b`.
- Trained on FLEURS train (17 languages) + LibriSpeech train-clean-100 (CC BY 4.0), selected on FLEURS dev +
  LibriSpeech dev-clean; LibriSpeech splits speaker-disjoint (checked), FLEURS has no speaker ids. Test, balanced
  accuracy at 2 s of speech (115M / 0.6B): LibriSpeech test-clean 97.8 / 97.7 % (95 % CI over 40 speakers about
  94-100), FLEURS-17 test 96.3 / 96.7 %. Engine cost on MPS within noise (Δ p50 −0.4 / +0.4 ms
  best-of-9, 115M / 0.6B); the stream's own time 0.05-0.08 ms per 160 ms chunk (it runs the head once per final,
  not per chunk).

### Added (2026-10-02): dual rate, `serve --final-chunk-ms {160,560,1120}` (research/DUAL_RATE.md)
- Heads, partials and turn decisions keep the 160 ms pass (turn ends and fast finals byte-identical, 32 / 32 live
  sessions, both cores); a text-only pass of the same encoder at [70,6] / [70,13] writes the `final` (`source: slow`),
  the 160 ms text is sent at the turn end as `final_fast`; adapters `final_text="fast"` (default) | `"slow"`. 1120 ms vs
  160 ms on the test sets: ICSI −3.5 / −1.9, AMI −0.6 / −0.1 (n.s.), live −1.7 / −0.8 WER points (115M / 0.6B);
  slow final 9-48 ms after the turn end; +20-27 ms p95 turn-end delivery (0.6B on CPU: ~110 ms / +93 ms, not recommended). Off by default.

### Changed (2026-10-02): 0.6B heads v0.4, real two-party channels for the turn heads (research/TURN_DATA.md)
- New data, all licensed for commercial training, split once by recording into train / held-out / never touched
  (disjointness checked by id, `turn_data.py splits`): AMI individual headsets (CC BY 4.0; 45 / 18 / 16 meetings),
  otoSpeech-full-duplex-280h raw channels (CC BY 4.0; 64 / 32 / 32 sessions); AppTek call-center dialogues
  (CC BY-SA 4.0, evaluation only per its card; 48 held-out / 48 never-touched calls). Level-varied and G.711 / Opus
  8 kHz copies for training.
- The test of the data failed: no source reproduces the TurnBench behaviour that blocked the `Q` heads (on every
  never-touched scope `Q` interrupts 6-15 points more than v0.2 on `balanced`, TurnBench 34, and less on `fast`,
  TurnBench 9 more). `balanced` / `fast` therefore stay as they are (identical decisions; calls missed 10.1 %, the
  ≤ 7.5 % goal is not met).
- `assets/served_heads_0p6b_v0.4.pt` (hub `HEADS_0P6B` 0.4, the default; `served_0p6b_v0.4.afm`): v0.3 +
  `turn_seg_a` (the `assistant` preset's own v5 classifier, trained with the real channels) + `turn_vad` (the
  stateless block-12 VAD it reads). Smart-turn v3.2 test, `assistant`: 93.0 → 96.5 % accuracy, 4.5 → 3.1 % false fires,
  379 → 351 ms p50. It interrupts AMI test meetings more (4.0 → 22.5 %). +0.4 ms per chunk (MPS).
- Serving: a preset's `turn_model.head` may name its own classifier (`ASRStream.attach_seg2`); a turn_seg head may read
  `turn_vad` as its VAD channel (cfg `vad_input`); `VadHeadPolicy(vad_m=…)` / a preset's `model_clock: "turn_vad"`
  runs the classifier's clock on the stateless VAD (no shipped preset uses it).
- `scripts/research/turn_data.py`; `core_0p6b_turn.py` options `--ihm*`, `--base`, `--turn-vad`, `--seg-a`;
  `tests/test_turn_data.py`.


### Changed (2026-10-02): LID heads v2, `--beam`, test-split comparison (research/FIXALL.md, research/FINAL_COMPARE.md)
- LID heads v2 on both cores: hidden 1024 (2.37 M / 2.89 M parameters), AmberNet distillation, the 0.6B now also trained on
  FLEURS `trainx`. FLEURS-17 test at 2 s / full clip: 115M 90.9 / 97.8 → 92.4 / 98.2 %, 0.6B 87.6 / 95.5 → 92.7 / 98.6 %.
  No measurable cost (±0.2 ms per chunk). Files `assets/lid_115m_v2.pt`, `assets/lid_0p6b_v2.pt` (hub `lid`, `lid_0p6b`).
  A Whisper large-v3 teacher was tried and hurt on dev.
- `serve --beam K`: finals from an RNNT beam search (≤ 3 tokens per frame) run on a CPU copy of the transducer next to
  the greedy decoder; partials and turn taking keep the greedy tokens. 115M at K = 8: AMI test WER 16.1 → 14.6 %, live
  calls 20.3 → 18.6 %; +1.5 ms (CPU) / +2.2 ms (MPS) per chunk, so it is off by default. The 0.6B does not gain.
- `VadHeadPolicy(reset_thr=…)` / a preset's `reset_thr`: a two-threshold silence clock (off by default; no shipped preset
  uses it).
- `SpeakerHead(hidden=…)`: optional two-layer projection (off by default).
- research/FINAL_COMPARE.md now reports public test splits only: LibriSpeech test-clean / test-other, AMI / ICSI test
  meetings, smart-turn v3.2 test, AMI test turns, FLEURS test. Every baseline was re-run on the same audio. The dev-split
  first pass moved to research/FIXALL.md.


### Added (2026-10-01): speech-detector head on both cores (research/FIXALL.md step 1)
- `heads.speech`: a stateless speech detector (served FrameHead shape on a learned block mix: 115M blocks 2-6, 0.6B
  blocks 8-16), trained on AMI + ICSI train meetings + oto user channels (clean and quiet-channel variants) with
  SpecAugment views and a held-out-calibrated threshold. It is the client's per-frame speech probability (the frame
  message's `vad` field). Turn rules, TS-VAD, LID gating and the v5 classifier keep reading `heads.vad`, so turn taking
  is unchanged (served check: identical turn ends on the bundled call, every preset, both cores).
- ICSI dev F1 0.898 → 0.947 (115M) and 0.906 → 0.951 (0.6B), above TEN VAD's 0.935. AMI dev F1 0.951 → 0.951 / 0.950
  (paired difference within its CI; AUC −0.002 / −0.003). Teacher soft targets (TEN VAD, pyannote segmentation-3.0)
  were tried and did not help on held-out data.
- `assets/served_heads_v0.4.pt` (hub `HEADS` 0.4, the default; `stage1_served_v4.afm`) and
  `assets/served_heads_0p6b_v0.3.pt` (hub `HEADS_0P6B` 0.3, the default; `served_0p6b_v0.3.afm`).
- `RNNTHead.beam_search` / `BeamTransducerStream` (research use, step 5).
- `scripts/research/fixall.py` (stages vman / vteach / vfeat / vtrain / vserved / vbuild / servedeq / wprep / wdec /
  wscore / lteach); `tests/test_fixall.py`.

### Changed (2026-10-01): 0.6B core heads v0.2, end of turn diagnosed (research/CORE_0P6B_TURN.md)
- Why the 0.6B was worse at end of turn: two heads, shown by swapping the 115M's per-frame signals into the 0.6B
  session.
  - The recurrent VAD head, trained on AMI only, stays up 1-2 s after the user stops and flickers on quiet phone
    channels. With the 115M's VAD, fast calls missed 11.9 → 7.3 %.
  - The v5 classifier fires inside utterances under the assistant trigger. With the 115M's v5, the assistant row goes
    78.2 → 90.2 % on shared constants.
- `assets/served_heads_0p6b_v0.2.pt` (hub `HEADS_0P6B` 0.2, the default): v0.1's tensors (same state hash) with every
  preset's constants re-picked on a held-out set. `balanced` now decides with the v5 classifier. On the evaluation sets
  (`runs/core_0p6b_turn.json`):
  - assistant 91.2 → 93.0 % accuracy and 7.1 → 4.5 % false fires at 374 ms;
  - fast calls 530 / 33.9 / 11.9 → 487 ms / 26.6 % / 11.0 %;
  - balanced calls 920 / 20.2 / 11.0 → 725 ms / 19.3 % / 10.1 %;
  - served == offline on 32 / 32 sessions per preset;
  - MPS cost 43-45 ms per chunk.
- The 115M's end-of-turn bars are still not all met: the calls misses, fast false interruptions, and the assistant
  p50. A full head fix (stateless VAD on block 12 + retrained v5 + block-12 per-frame turn head) was built and
  measured. It halves the calls misses (4.6 %) and gives assistant 95.2 % / 342 ms, but it is not shipped: it raises
  `balanced` false interruptions on the TurnBench calls.
- Serving: a model's `cfg["turn_presets"]` can give a preset a `theta` of its own and give `balanced` the v5 decider
  (`V5_TURN_MODEL`). Models without them are unchanged.
- `scripts/research/core_0p6b_turn.py`: evaluation cache, cross-core swap, held-out end-of-turn set with a
  quiet-channel scope, VAD / v5 / per-frame head candidates, rule scans, served check, cost.

### Added (2026-10-01): a second core, `--core 0.6b` (nemotron-speech-streaming-en-0.6b), every head retrained
- `audioforge-download --core 0.6b` / `audioforge-serve --core 0.6b` / `audioforge.load(core="0.6b")` /
  `audioforge.voiceprint(core="0.6b")`.
  - Hub component `asr_0p6b`: `nvidia/nemotron-speech-streaming-en-0.6b` at revision `ebe59e5a`, sha256-pinned,
    NVIDIA Open Model License.
  - Heads assets `assets/served_heads_0p6b_v0.1.pt` (rebuilt bit-identically by `hub.build_served`),
    `assets/tsvad_0p6b.pt`, `assets/lid_0p6b.pt`.
  - Clear errors when a 0.6B file is missing; the launcher states the CPU cost (one real-time stream per process).
  - The 115M stays the default.
- Heads on the 0.6B (`scripts/research/core_0p6b_heads.py`, research/CORE_0P6B.md):
  - VAD: causal GRU frame head on a block mix, SpecAugment views, room-tone negatives, frame-centre labels.
  - Speaker and TS-VAD at block 5: TitaNet distillation; simulated conversations and an overlap-weighted loss.
  - Turn classifier v5 (block 12), and a pass-1 per-frame turn head: the 0.6B has no speaker-kernel pass.
  - LID distilled from AmberNet.
  - Results (all in `runs/core_0p6b.json`; the WER rows are in `runs/hybrid_asr.json`):
    - meeting WER halved (AMI 24.4 → 11.2 %);
    - live-call WER 22.5 → 13.7 %;
    - AMI within-meeting speaker EER 17.4 → 13.6 % (same recipe);
    - end of turn not better (calls miss 11.0 vs 7.3 %);
    - 96 vs 30 ms per chunk on CPU, 1 vs 4 real-time CPU streams, 5 GB RSS.
- Layer sweeps per head on the 0.6B (research/LAYER_SWEEP_0P6B.md): every block, the mix and the top-3 concat,
  selected on held-out meetings / speakers. They moved the speaker and TS-VAD heads to block 5.
- Serving:
  - `frame_gru` head type (recurrent frame head with per-session state).
  - Per-model turn-preset constants (`cfg["turn_presets"]`; the 0.6B's `assistant` waits 320 ms and needs p > 0.99).
- Fixed: the turn head v5 classifier is attached on the model's device. A `--device mps|cuda` session with
  `--turn-preset fast` / `assistant` passed CPU tensors to the GPU model.

### Changed (2026-09-30): public images for turn head v5
- `demo/images/results_v10.png` (+ `_square`, `_notext`): the v9 style with a speech-to-an-agent card
  (`--turn-preset assistant` vs Pipecat smart-turn on smart-turn's 399 test clips: 92 vs 70 % accuracy, 5 vs 40 %
  false fires) and the calls card on `--turn-preset fast` vs LiveKit (547 vs 567 ms p50, 25 vs 27 % false
  interruptions, 5.5 vs 23 % missed). Numbers from `runs/turn_v5.json`, `runs/eot_assistant.json`,
  `runs/eot_latency.json` via `demo/images/redesign/export_single.py` (`v10/*` keys); `render.py r10` checks.
- `demo/images/architecture_v9.png`: the turn-end box names both deciders (GRU turn head under `balanced`, the v5
  classifier under `fast` / `assistant`); six heads. README points at both.

### Added (2026-09-30): turn head v5, served heads v0.3, `--turn-preset fast` (v5) / `steady` / `assistant`
- **Turn head v5** (`audioforge/heads/turn_seg.py`, research/TURN_V5.md): a smart-turn style segment classifier on
  the frozen encoder.
  - Attention pooling over the last 8 s of encoder block 8 (already computed by the ASR pass), the served VAD /
    TS-VAD tracks and the RNNT tokens so far; smart-turn's MLP head; 2.46M parameters.
  - Trained offline on oto / AMI / ICSI turn ends vs in-turn pauses, 7 626 utterances cut at word boundaries, and
    smart-turn's human_5_all train clips, with smart-turn v3.2 soft targets and an LM-completeness target
    (Qwen2.5-7B-Instruct) as auxiliaries.
  - Smart-turn's 399 test clips: 99.0 % (smart-turn v3.2: 97.0 %; the v2 turn head: 63 %).
  - Served: `ASRStream.attach_seg` / `seg_prob`, equal to the offline classifier to 3.5e-4; 1.0-1.7 ms per call,
    about +1.5 ms per 160 ms chunk.
- **`--turn-preset fast` is now v5.** The classifier is asked after 80 ms of VAD < 0.6 and again at every quiet
  frame (`VadHeadPolicy(model_reask=True)`); it ends the turn at P(complete) > 0.7, with the 640 ms fallback.
  - Two-party calls: 547 ms p50 (balanced 956), 24.8 % FI, 5.5 % missed. AMI: 1247 ms, 11.5 % / 34.0 %.
  - No cut of the bundled clip under any of the six deliveries.
  - A second training seed: 586 ms, 26.6 % / 5.5 %.
- **`--turn-preset steady`** = the pre-v5 `fast` rule, unchanged: 886 / 1434 ms, 24.8 / 3.7 %. It keeps the best
  calls p95 and misses.
- **`--turn-preset assistant`** is for speech directed at the agent. v5 is asked after 240 ms of energy-or-VAD
  quiet, ends the turn at P > 0.9, and a 2.96 s timer backs it up.
  - Smart-turn's 399 test clips: 92.2 % accuracy, 292 ms p50, 5.4 % false fires. Served: 93.0 % / 317 ms.
  - The smart-turn bridge gives 95.7 % / 770 ms, Pipecat 69.7 % / 211 ms.
  - Not for human conversation: the timer misses 34 % of call ends.
- `turn_end_hint` under a v5 preset reads the classifier's p at quiet frames (threshold 0.5, `V5_HINT_P`).
- **Served heads v0.3** (`assets/served_heads_v0.3.pt`, 29.4 MB, `hub.HEADS["0.3"]`, `HEADS_VERSION` 0.3) builds
  `stage1_served_v3.afm` = `stage1_served_v2.afm` + `heads.turn_seg`; every v0.2 tensor is unchanged, so
  `balanced` / `steady` are bit-identical. `--heads-version 0.2` still builds v2 (the v5 presets need v0.3).
- `balanced` stays the default. The ship bar for replacing it (>= 150 ms faster at <= 20.2 % FI / <= 7.3 % missed
  and AMI <= 10.5 / 33.5 %) was not met.

### Added (2026-09-30): `vad_head` energy gate (default) and `--turn-model smartturn` (opt-in bridge)
- **Energy gate** (`--energy-gate on|off`, default on; `audioforge.server.policies.EnergyGate`,
  `constants.ENERGY_GATE`).
  - A per-session noise floor: the 10th percentile of the 80 ms frame log energies over 3 s.
  - A user turn is armed only by an onset (VAD > 0.5 AND energy > floor + 6 dB).
  - No `turn_end` / `turn_end_hint` before 160 ms of onsets.
  - Served on smart-turn's 399 assistant test clips: turn_ends before the user spoke 137 -> 0, early fires on complete
    clips 6.3 -> 0 %, accuracy 41.1 -> 43.9 %.
  - Two-party calls, AMI and the bundled clip (six deliveries) are unchanged: 956 ms / 20.2 % / 7.3 % and 1326 ms /
    10.5 % / 33.5 %.
- **`--energy-quiet-db X`** (off by default): frames below floor + X dB count as silence. Assistant EOT 1218 -> 611 ms
  at X = 6, but calls false interruptions 20.2 -> 48.6 %. No X in 3..12 dB (fallback 640-1280 ms) holds calls / AMI:
  mid-turn pauses are room tone too.
- **`--turn-model smartturn`** (+ `--smartturn-onnx`, `--smartturn-trigger vad|energy`; audioforge/server/smartturn.py).
  - Pipecat's smart-turn v3.2 ONNX (BSD-2-Clause) runs inside `vad_head` at our quiet trigger, with Pipecat's exact
    input preparation.
  - It is a second model and opt-in: a bridge until turn head v5.
  - Assistant clips (served): 770 / 1004 ms, 5.8 % false fires on incomplete clips, 95.7 % accuracy (Pipecat + Silero:
    211 ms, 40.2 %, 69.7 %); ~20 ms per call.
  - Calls: 629 ms / 19.3 % FI / 23.9 % missed. AMI: 1087 ms / 13.0 % / 36.5 %.
- **Protocol:** `turn_end.path` (`head` | `fallback` | `others` | `model`) under `vad_head`; `turn_end.model_ms` and
  `stats.turn_model` with smartturn.
- Measured by `scripts/research/eot_energy_gate.py` -> `runs/eot_energy_gate.json`; research/EOT_ASSISTANT.md "Energy
  gate and the smart-turn bridge".

### Added (2026-09-30): `--turn-preset balanced | fast`
- **Server:** `--turn-preset` (and `config.turn_preset`, per session, before the first audio) names `vad_head`'s
  constants (`TURN_PRESETS` in audioforge/server/constants.py). `balanced` is the current rule and the default
  (`MODES["single"]`). `fast` = VAD < 0.6 for ≥ 480 ms AND p ≥ 0.99, OR 720 ms, others path 640,640.
  `--vad-wait-ms` / `--others-wait-ms` still override a preset.
- **Measured** (scripts/research/vad_tail.py `presets`, `runs/vad_tail.json`). Calls, `fast` vs `balanced`:
  - EOT p50 886 vs 956 ms (only ~70 ms faster);
  - p95 1434 vs 1919 ms (~490 ms faster);
  - missed 3.7 vs 7.3 %;
  - false interruptions 24.8 vs 20.2 % (LiveKit defaults: 26.6 %).

  AMI, `fast` vs `balanced`: 1086 vs 1326 ms, 16.0 vs 10.5 % FI, 30.5 vs 33.5 % missed. Neither preset cuts the
  bundled clip under the six deliveries (`served_check --preset fast`). Tables: docs/CONFIGURATION.md §4 "Turn
  presets", research/EOT_LATENCY.md "Turn presets".

### Research (2026-09-30): the VAD tail is a hangover, and removing it does not help (research/VAD_TAIL.md)
- The served VAD head (block 4) stays above 0.5 for 212 ms after call-turn ends (p50) and above 0.4 for 320 ms, where
  Silero has already stopped. About 40 ms of that is the 80 ms frame grid. The rest is the head: it shows the same tail
  against its own AMI training labels, and block 4 has no lookahead delay.
- The tail is the same at mid-turn pauses (368 vs 416 ms after Silero's end) and it bridges 67 % of pauses ≥ 160 ms.
  Today's 20 % false-interruption rate depends on it.
- Tail-free alternatives were scored with the harness: a backdated silence clock; an oracle tail-free VAD (Silero on
  the frame grid) with 0-320 ms of hangover; a tail-free VAD on the head path only. None meets the goal. The oracle
  needs 1176-1216 ms of calls p50 for ≤ 20.2 % FI. No VAD head was retrained and served_heads_v0.2 is unchanged.

### Added (2026-09-30): early end-of-turn hints (`turn_end_hint`)
- **Server:** new events `turn_end_hint` `{t, p, kind: "hint", text}` (once per user turn, on 80 ms of served-VAD
  silence with turn-head p >= `--turn-hint-p`, default 0.8) and `turn_end_hint_cancel` `{t}` (the user resumed:
  VAD > 0.5 on 2 frames); the next `turn_end` confirms the hint in its new `hinted_at` field; `stats.turn_hints`
  counts sent / confirmed / cancelled and each outcome is logged. `turn_end` is unchanged; `--turn-hint-off`
  restores the previous protocol exactly (docs/PROTOCOL.md §5.11, audioforge/server/turn_hint.py).
- **Pipecat:** `AudioforgeSTTService(turn_hints=True)` + `AudioforgeEagerTurnStopStrategy`: the hint drives Pipecat
  1.12's eager end of turn (speculative inference held by the LLM service's `SpeculationGate`, released at the
  `turn_end` if the final matches the hint's text, discarded on cancel). The STT now also records `language` events
  instead of warning about them.
- **LiveKit:** `AudioforgeFrontend(turn_hints=True)`: the hint becomes a `PREFLIGHT_TRANSCRIPT`, which LiveKit 1.8's
  preemptive generation answers; the worker enables `preemptive_generation` only with hints on
  (`AUDIOFORGE_TURN_HINTS=1`). Both adapters are opt-in; defaults unchanged.
- **Measured** (scripts/research/turn_hint.py, `runs/turn_hint.json`, stored dumps): at H 0.8 the reply could start
  250 / 364 ms earlier at the p50 on calls with 300 / 600 ms of LLM + TTS prep (1226 -> 976, 1526 -> 1162 ms after
  the true end), 80 ms on AMI; recall 64 % (calls) / 44 % (AMI), 0.42 / 0.31 discarded preps per turn, false
  interruptions unchanged. Live on the bundled clip: hint 160 ms before the `turn_end`; Pipecat demo with a mock LLM:
  reply 1342 -> 1173 ms (300 ms prep), 1642 -> 1477 ms (600 ms prep) after the true end.

### Research (2026-09-30): turn head v4, not shipped
- Audible-end reference labels for the EOT sessions (`runs/turn_v4_labels_eval.json`). Under them the shipped
  `vad_head` rule gives 951 ms, 19.3 % false interruptions and 6.4 % missed on calls; AMI is unchanged. No faster rule
  meets the goal, so the default is unchanged.
- The turn head was retrained heads-only for early confidence, on cached served pass-2 inputs (otoSpeech, 12 AMI and
  12 ICSI train meetings, smart-turn human_5_all train). No head plus rule beat today's calls p50 by 150 ms at equal
  or better FI and misses, so nothing ships. The best new head reaches calls 989 ms at 15.6 / 7.3 %, and AMI
  1167 ms at 10.5 / 33.5 %.
- The limit is the served VAD head's 240-320 ms tail after call ends. Details: research/TURN_V4.md,
  `runs/turn_v4.json`, `scripts/research/turn_v4.py`.

### Results (2026-09-30): where single mode stands after today's changes
- **Turn rule:** single mode's default is `vad_head` (VAD head < 0.4 for ≥ 160 ms and turn head p ≥ 0.99, or 640 ms
  of silence). Calls: EOT p50 956 ms, 20 % false interruptions, 7 % missed; AMI: 1326 ms, 11 %, 34 %
  (research/EOT_LATENCY.md, `runs/eot_latency.json`).
- **Silero dropped** from single mode: not downloaded by default, not loaded; `hybrid_dyn` still works with `--silero`.
- **70 ms chunk-trigger fix** (d832fca): each ASR chunk runs once its mels are complete, so every frame, token and
  head output is ready 70 ms earlier, with identical outputs (research/LATENCY_BUDGET.md).
- **Voice-print check + anchored adaptation** (fe28a9e): clean prints and a working print kept at 0.8 × the enrolled
  one. Target-speaker WER on mixed calls 44.6 → 40.2 %, user channel 24.4 → 18.1 % (research/TSWER.md).
- **`--device mps`** (2bdf3e9): same events and decodes as CPU; 28.7 vs 29.8 ms per 160 ms chunk, 5 vs 4 real-time
  streams per process (research/MPS_115M.md).
- **README rewritten** around single mode: one results table with every number linked to its source.

### Changed (2026-09-30): `vad_head` waits for a surer head; the quickstart clip is no longer cut (research/EOT_LATENCY.md)
- **Why:** after the TS-VAD print fix (fe28a9e), the rule below cut the bundled quickstart clip mid-question at
  9.06 s when streamed with `examples/quickstart_client.py`. In the user's half-second pause the turn head reads
  0.94-0.98 and the served VAD dips in and out of 0.4, so the client's int16 truncation decided the cut.
- **The rule's defaults now:** VAD below 0.4 for ≥ 160 ms and turn head p ≥ 0.99, or 640 ms of that silence
  (`--vad-wait-ms 160,640`, `POLICY_THETA["vad_head"]` 0.99). The others path is unchanged. No code path changed.
- **Measured offline** on the post-fix dump, against the fe28a9e rule:
  - Two-party calls: EOT p50 / p95 956 / 1919 vs 916 / 2020 ms, false interruptions 20.2 vs 22.9 %, missed 7.3 vs
    7.3 %.
  - AMI: 1326 / 3758 vs 1327 / 4162 ms, false interruptions 10.5 vs 10.0 %, missed 33.5 vs 34.0 %.
  - The quickstart clip is not cut under any of six deliveries (float, int16 truncated / rounded, -6 / +6 dB,
    dither), in process and over the websocket (3 runs at 1x). VAD hysteresis, the head path at 400 ms and a
    minimum-speech guard were tried and each still cut it or missed more AMI ends.
  - Source: `runs/eot_latency.json` (`selection.fastest_print_fix_goal_no_clip_cut`, `demo_clip`, `served_check`).

### Changed (2026-09-30): single mode's turn rule is `vad_head`, no Silero (research/EOT_LATENCY.md)
- **`--mode single` adds `--turn-policy vad_head`.** This is the new server default for a session whose `config` names
  no `turn_policy`; a client's `config` still wins. Room mode keeps `timeout`.
- **The rule:** the served VAD head below 0.4 for ≥ 320 ms and turn head p ≥ 0.95, or 800 ms of that silence. With an
  enrolled TS-VAD track there is also the others path: the user's own silence reaching 960 ms while P(other) ≥ 0.9
  has held for 640 ms.
  - New flag `--others-wait-ms USER_SIL,HOLD`.
  - `--vad-wait-ms` now defaults to `320,800`.
- **Measured offline** on the served clock after the chunk-trigger fix, against `hybrid_dyn 2000,960`:
  - Two-party calls: EOT p50 / p95 881 / 1789 vs 1287 / 1923 ms, false interruptions 22.9 vs 22.9 %, missed 7.3 vs
    7.3 %.
  - AMI: 1330 / 4160 vs 1794 / 4285 ms, false interruptions 10.0 vs 8.5 %, missed 33.5 vs 37.5 %.
  - Source: `runs/eot_latency.json`.
- **Silero is no longer part of single mode.**
  - `audioforge-download`'s default set is `asr + tsvad` (+ `lid`).
  - `audioforge-serve --mode single` no longer passes `--silero`.
  - `hybrid_dyn` still works with `--silero` or `--with silero`.
- **`audioforge.load()` sessions** default to the engine's policy (`Frontend.session(turn_policy=None)`).
- **`scripts/research/eot_latency.py`:**
  - Re-dumped on `stage1_served_v2.afm` after d832fca: every frame is ready 70 ms earlier.
  - "Decision" latency is now total − compute; it no longer subtracts a 240 ms constant.
  - LiveKit's transcript lookup follows the new ASR clock.
  - New room-aware family `sim_room`.

### Changed (2026-09-29): one standard scorecard (research/METRICS.md)
- **New `research/METRICS.md`:** every published number in standard voice-agent metrics, one definition each. STT:
  WER on the full LibriSpeech test-clean / test-other (Whisper normalizer) and AMI, RTFx, partial latency (word end
  to word shown) and final latency. Turn detection: end-of-turn latency p50 / p95, false-interruption rate, response
  rate, TurnBench precision / recall / F1. VAD: ROC-AUC and F1. Speaker: target-speaker DER. Efficiency: compute per
  160 ms chunk, real-time streams per process, CPU seconds per audio second. The README's Results table and
  `research/SINGLE_MODEL.md` now use it.
- **Measured for it:**
  - Partial latency: 441 / 732 ms p50 / p95 on CPU (`scripts/research/stt_latency.py`, `runs/stt_latency.json`).
  - Full-LibriSpeech WER: 2.48 / 6.13 % (`scripts/research/asr_leaderboard.py`, `runs/asr_leaderboard.json`).
  - VAD AUC: 0.972 vs Silero 0.956 and MarbleNet 0.959 (`scripts/research/vad_auc.py`, `runs/vad_auc.json`).
  - 3 real-time sessions per process on 2 CPU threads (`scripts/research/streams_cpu.py`, `runs/streams_cpu.json`).
  - The live records re-scored per condition, with the target-speaker DER and TurnBench F1 derived from stored
    counts (`scripts/research/metrics_table.py`, `runs/metrics.json`).
- **Removed:**
  - The "first words" metric (1081 ms): it counted from the start of speech.
  - The pooled "missed within 3 s" headline: 32 of its 69 sessions were mono mixes, where the recorded partner
    answers in the same audio and no system can respond in time.
  - The name "dead air" (now end-of-turn latency) and "cut-ins" (now false interruptions).
- `research/FINAL_REPORT.md` is marked historical (room-mode era).


### Changed (2026-09-29): the shipped model's VAD head reads block 4
- `audioforge-download` now builds **`stage1_served_v2.afm`** from `assets/served_heads_v0.2.pt` (104 tensors,
  19,629,563 bytes, sha256 `cb5aa069…`; pinned in `hub.HEADS`). Its VAD head reads encoder block 4 only (the speaker
  and TS-VAD heads' tap): AMI VAD F1 0.951 vs 0.949, no downstream regression (research/VAD_SINGLE.md,
  runs/vad_single.json). Every other tensor is bit-identical to `stage1_served.afm`.
- `audioforge-serve`, `audioforge.load()` and `audioforge-bench` load that build (`hub.SERVED`). An existing
  `models/stage1_served.afm` is no longer picked up: re-run `audioforge-download`.
- `runs/stage1_served.afm` stays the 2026-09-27 measured checkpoint that the research documents cite.
  `audioforge-download --heads-version 0.1` rebuilds it from `assets/served_heads_v0.1.pt`, which is kept.
- The heads file now has its size and sha256 checked, not only the merged model's tensor hash.

### Changed (2026-09-29): single-model mode is the default
- **`audioforge-serve` runs single-model mode by default.** One 115M model follows the known user by a stored voice
  print (target diarization) and loads no diarizer. The previous two-model stack is **`--mode room`**: general
  diarization with NVIDIA Nemotron-3-Diarization (the room default when downloaded, else Sortformer v2), optionally
  with `--final-asr tdt_v3`. Without `--mode`, `--diarizer`, `--diar` or `--final-asr` select room mode, so older
  command lines keep working. `--mode default` is gone (renamed `room`). `audioforge.load()` and `audioforge-bench`
  follow the same rule.
- **Voice sample requirement.** Send the user's stored print, `{"type": "enroll", "embedding": [...]}`, made from
  ≥ 5 s of clean speech (10 s for meetings). Live grabs after `agent_end` cost 16-55 points of missed turn ends in
  meetings.
  - New: `audioforge.voiceprint(audio)` and `Frontend.voiceprint`.
  - `voiceprint` messages now carry the `embedding`.
  - Both adapters send a stored print (`AudioforgeSTTService.enroll(embedding=...)`,
    `AudioforgeFrontend.enroll(embedding=...)`) under any enroll mode.
  - `examples/quickstart_client.py` streams a 16 s otoSpeech two-party call (`examples/audio/two_party_call_16s.wav`,
    CC BY 4.0) and sends its user's stored print. The LibriSpeech clip `two_speakers_10s.wav` stays as room mode's
    example.
- **`audioforge-download` defaults to `asr + tsvad + lid + silero`.** The TS-VAD head (`tsvad_spk.pt`, 1.0 MB) and
  the LID head (`lid_distill.pt`, 3.7 MB) are sha256-pinned components, shipped in `assets/` and attached to the
  v0.1.0 release. `--diarizer nemotron3|sortformer` adds room mode's diarizer. A failed head download stops with one
  line naming the file.
- In single-model mode `final.speaker` is 0 for the user and 1 for someone else (the dominant TS-VAD column of the
  turn).

### Added (2026-09-29, single-model mode, research/SINGLE_MODEL.md)
- **`audioforge-serve --mode single`** (config key `mode: single`, `audioforge.load(mode="single")`,
  `audioforge-bench --mode single`): everything from the one 115M checkpoint for a known user. It adds `--turn-input
  tsvad --diar-off --lid head --enroll after_agent_arm --dyn-wait-ms 2000,960`, loads no diarizer and no final-ASR
  worker, and refuses options that would load a second model. The default mode is unchanged.
- `--diar-off` without `--diar` no longer loads a diarizer at all; `ready.diar_config` is `"off"` then.
- An `enroll` message with a stored `embedding` is accepted under every `--enroll` mode when `--turn-input tsvad` is on.
- `--dyn-wait-ms CAP,FLOOR` sets `hybrid_dyn`'s silence wait.
- `scripts/research/single_model_distill/`: the GPU package for target-speaker and turn heads trained on more data
  (prepared, smoke-tested, not run).

### Changed (2026-09-29 cleanup for the public release)
- **Layout.** Research drivers and drafts left the top level: `scripts/bench_serve.py`, `uc_turn.py`,
  `make_served_model.py`, `run_all.sh`, `audioforge/uc_stream.py` and `audioforge/heads/uc_turn.py` (now
  `scripts/research/uc_turn_head.py`) moved to `scripts/research/` (indexed in its new README); `recipes/` moved to
  `research/recipes/` (with an index; `runs/*.json` provenance still cites the old paths). `scripts/bench.py` and
  `scripts/download_models.py` were deleted (duplicates of `audioforge-bench` / `audioforge-download`).
- **Server.** `audioforge/serve.py` (2755 lines) was split: the engine, sessions and connection loop stay in
  `serve.py`; the protocol, turn policies, speaker binding, ASR streams, flags and helpers are in `audioforge/server/`.
  `audioforge.serve.X` and `python -m audioforge.serve` are unchanged. The design docstring moved to
  `docs/SERVER_INTERNALS.md`.
- **Config.** One flag table (`audioforge/server/cli.py`) produces `--help` (everyday flags, one screen),
  `--help-advanced` (all; `--help-serve` still works), `--config FILE` / `$AUDIOFORGE_CONFIG` (YAML or JSON, flags
  win) and the flag reference in `docs/CONFIGURATION.md`, generated by `scripts/dev/gen_config_doc.py` and checked by
  `tests/test_config_doc.py` and CI. The Silero default path follows `$AUDIOFORGE_DATA`.
- **Adapters.** The Pipecat and LiveKit adapters ship in the package as `audioforge.integrations.pipecat` /
  `.livekit` (`pip install "audioforge[pipecat]"` is enough); `integrations.pipecat_audioforge` /
  `integrations.livekit_audioforge` remain as aliases.
- **Logging.** Library code logs under the `audioforge` logger (plain messages to stdout at INFO by default, so the
  console output is unchanged); `audioforge.use_app_logging()` hands records to the application's handlers.
- `audioforge-bench --diarizer nemotron3` now applies the launcher's settings (all 8 columns, hold on shed): the
  product diarizer settings live once in `hub.diarizer_defaults`.
- README cut to what a user needs; its synthetic-results, early live-server and corrections-log sections moved to
  `research/EARLY_RESULTS.md`.

### Added (2026-09-29)
- Python API: `audioforge.load()` -> `Frontend.session()` -> `Session.feed(pcm)` / `end()` returning the protocol's
  messages; `examples/python_api.py`; `tests/test_api.py`.
- `docs/ARCHITECTURE.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.pre-commit-config.yaml`.
- Lint: ruff now also enforces import order (I) and bugbear (B); mypy runs on the public surface in CI; CI checks the
  generated config doc and smoke-tests the CLIs.

### Earlier unreleased changes

### Added
- `audioforge/perf.py` + `audioforge-serve --perf` (default on): exact CPU inference fast paths (column-major
  `nn.Linear` weights for Accelerate's NN GEMM path, cached relative-position projection, one shared subsampling for
  the ASR and turn passes, cached RNNT joint prediction projection). Server compute 0.62x with identical protocol
  decisions (0 differences in 1313 messages, probabilities within 1e-5; A/B on the 5 AMI windows; `runs/perf.json`,
  `research/PERFORMANCE.md`); `scripts/research/bench_serve.py`
  (`run` / `components` / `ab` / `compare` / `table`) and `tests/test_perf.py`.
- Many-speaker diarization (2026-09-28, `research/DIARIZATION_FIX.md`, `tests/test_diarization_fix.py`):
  `serve --diar-labels registry` (stable voice-keyed speaker ids per session from the served speaker head, or
  TitaNet-L with `--diar-embed titanet`; `final.speaker_conf`, `final.diar_shed`, `stats.speakers_seen`),
  `--shed-diar hold` (no speaker-0 collapse under load shedding) and session config `turn_policy: "timeout_any"`.
- TS-VAD turn path (2026-09-28, behind flags; `research/IMPROVEMENTS.md` §1): `audioforge/tsvad_stream.py`,
  `serve --turn-input tsvad [--diar-off] [--tsvad-print-s S] [--tsvad-refresh-s S]`, the `enroll` message with a stored
  `embedding`, the `voiceprint` message; Pipecat adapter `enroll(embedding=...)`; `tests/test_tsvad_stream.py`.
- Server hardening (2026-09-28, `research/BULLETPROOF.md`): structured `error` messages, `GET /health`, `--log-json`,
  RTF-gated load shedding, final-ASR watchdog, idle / session limits; `tests/test_bulletproof.py` (48 tests).
- `scripts/research/distill_0p6b_to_115m/`: the GPU package that distils the 0.6B streaming model into the 115M core
  (setup, preflight, KD with a WER gate and a no-teacher control; smoke-tested on ~1 h; ~6 A100 hours estimated).
- `research/VERIFICATION_2026-09-28.md`: the final report's numbers read back from their JSON (551 claims).
- `research/PERFORMANCE.md`: every RTF figure in the repository labelled by what it measures (model-only offline
  batch decode 0.015 / 0.022 vs live server 0.64 / 0.80 vs the on-device runtime study 0.16), per-component cost of
  the served path per 160 ms, concurrency, accepted / rejected optimisations.
- `pyproject.toml`: pip-installable `audioforge` with a minimal core (torch, numpy, pyyaml, sentencepiece,
  soundfile, huggingface_hub) and extras `serve`, `pipecat`, `livekit`, `nemo-import`, `dev`, `research`; console
  entry points `audioforge`, `audioforge-serve`, `audioforge-download`, `audioforge-bench`.
- `audioforge-download` (`audioforge/hub.py`): fetches the pinned NVIDIA / Silero checkpoints by Hugging Face
  revision or fixed URL, verifies size and sha256, shows each model licence and asks for acceptance, converts
  `.nemo` to `.afm`, and rebuilds the served model bit-identically from NVIDIA's weights plus
  `assets/served_heads_v0.1.pt` (tensor-hash asserted). Reuses files already under `$AUDIOFORGE_DATA/nemo` and
  `--from-local` directories instead of downloading.
- `audioforge-serve` (`audioforge/launch.py`): starts `audioforge.serve` with the downloaded models;
  `--diarizer nemotron3` applies the measured Nemotron-3 configuration (`--diar-pool max --diar-left 1`).
- `audioforge-bench`: in-process RTF / per-block latency / events on a WAV file, no server.
- `AUDIOFORGE_DATA` (datasets, `.nemo`, Silero; default `<repo>/data`) and `AUDIOFORGE_HOME` (downloaded models;
  default `<repo>/models` in a checkout, else `~/.cache/audioforge`), so data can live on external storage.
- `examples/quickstart_client.py` and the bundled 10 s two-speaker clip `examples/audio/two_speakers_10s.wav`
  (LibriSpeech, CC-BY-4.0).
- `docs/CONFIGURATION.md` (every server flag with its measured cost), `docs/PROTOCOL.md` (the WebSocket wire
  protocol), `research/README.md` (index of the research notes), `docs/RELEASE_CHECKLIST.md`.
- GitHub Actions CI (`.github/workflows/ci.yml`): ruff plus the fast CPU tests (`tests/fast_ci.txt`) on Python
  3.10 and 3.12, and the `audioforge-client` package tests; no checkpoints are downloaded.
- `packages/audioforge-client`: a torch-free WebSocket client and protocol framing for `audioforge.serve`.
- `LICENSE` (Apache-2.0) and `NOTICE` (model-weight and vendored-code licences). The licence choice is still to be
  confirmed by the maintainer (see `docs/RELEASE_CHECKLIST.md`).

### Changed
- Repository layout: research drivers moved from `scripts/` to `scripts/research/`; `scripts/` keeps the user-facing
  `bench.py`, `download_models.py`, `stream_client.py`, `make_served_model.py`, `run_all.sh`; machine helpers in
  `scripts/dev/`. `tests/conftest.py` puts both script directories on `sys.path`. Research notes under `research/`
  still cite the old `scripts/<name>.py` paths.
- README: the results section is now the final scorecard from `research/FINAL_REPORT.md` (14 rows, each traced to
  its `runs/*.json`); the 2026-09-26 "one model vs the dedicated models" table it replaces showed the original
  speaker head (32.2 % EER) and a product verdict that were both superseded.
- `.gitignore`: `data/`, `models/`, `runs/*.afm|pt|onnx`, `*.log`, demo renders and scratch venvs are ignored.
- `audioforge-serve --diarizer nemotron3` no longer adds `--diar-spks 4` (it hid speakers 5-8): `frame.speakers` now
  carries 8 columns for Nemotron-3 (`--diar-spks 4` restores 4); the launcher adds `--shed-diar hold` for either
  diarizer unless given (2026-09-28). `python -m audioforge.serve` defaults are unchanged.
- `research/FINAL_REPORT.md` / `.html` refreshed on 2026-09-28 (scorecard rows 15-26, §4.1, §5.6-5.8, §7.1-7.3, §8.1);
  README scorecard, live-server and demo sections follow it. Doc corrections from the verification pass:
  `DIARIZATION_FIX.md` §3.4 (AMI-64 DER after the fix is 0.245, not 0.229: the old row was a 57-of-64-window partial),
  `BULLETPROOF.md` (48 tests, not 45), `IMPROVEMENTS.md` §3 (TitaNet row columns).

### Measured (2026-09-27, `research/FINAL_REPORT.md`)
- One frozen 115M NVIDIA streaming encoder with small trained heads, plus NVIDIA's diarizer, at 160 ms chunks on a
  2-thread laptop CPU: best streaming VAD measured on AMI (F1 0.949), fewer missed turn ends than every dedicated
  turn detector at matched false cutoffs on AMI (hybrid 61.9 % vs Silero timeout 72.7 %) and held-out ICSI, and
  1.6-5.5x fewer cut-ins than Pipecat's default stack end to end; worse than the dedicated models at meeting ASR
  (24.4 % WER, fixed by a per-turn Parakeet-TDT v3 pass: 9.7 %), speaker verification, diarization and language ID.

### Measured (2026-09-28, `research/FINAL_REPORT.md` rows 15-26)
- Many-speaker rooms: DER 0.395 -> 0.247 and the speaker count exact on three 4-6-speaker clips, forced-shedding DER
  0.809 -> 0.329, AMI 20 s windows unchanged (0.245). Server compute 0.62x with the fast paths.
- TS-VAD turn path: missed turn ends 61.9 -> 39.3 % (AMI) and 68.5 -> 20.4 % (ICSI) offline with a 5 s voice print;
  live on 69 mostly two-party sessions +10.8 [+5.6, +15.8] points of misses at 3 s vs the default, equal to the
  no-cut-in rule at 0.43x compute (behind flags).
- 0.6B streaming core not adopted (AMI `hybrid_dyn` misses +2.8 [+0.1, +5.3], five categories equal; meeting WER -13
  points). Decoder-only adaptation AMI 24.43 -> 22.50 %; `--asr-lookahead 13` -1.41 / -2.50 / -0.37 WER points;
  crop-level speaker head 19.8 -> 17.1-17.4 % EER; transcript LID 100 % / 91.7 %.
- Negative: TS-VAD-trained predictive turn head, `hybrid_fast`, diarization distillation into a 1.9 M head.

### Open before release (2026-09-28; details in `docs/RELEASE_CHECKLIST.md`)
- Licence confirmation (Apache-2.0 is a placeholder), author e-mail in the commit metadata, Hugging Face token
  rotation, the upstream PRs of `docs/UPSTREAM_PRS.md`, the GPU distillation run, a human listen of the demo video.

### Tests (2026-09-28)
- After the final-report refresh (one gated run each): `ruff check audioforge tests integrations examples scripts`
  reports 4 findings (F401 in `audioforge/heads/uc_turn.py` and `scripts/research/uc_turn.py`, F841 in
  `tests/test_diarization_fix.py` and `tests/test_uc_turn.py`), none in files this refresh touched, so CI's ruff step
  fails until they are fixed (`ruff check --fix` handles the two F401); the fast CI subset (`tests/fast_ci.txt`):
  524 tests, 523 passed, 1 skipped, 0 failed.
- Full suite on the development machine (`scripts/dev/gate.sh .venv/bin/python -m pytest -q tests`, 10 min 47 s):
  718 passed, 5 skipped, 7 failed; all 7 failures are in the untracked, in-progress `tests/test_bulletproof.py`
  against an uncommitted `audioforge/serve.py` (another workstream), none in committed code.
- CI subset (`tests/fast_ci.txt`) in the fresh quickstart venv: 536 passed, 7 skipped, 84 s for the superset it was
  cut from; the 11 failures in that superset were research-only imports and those four files are excluded.

## [0.1.0] - unreleased
Initial public version; the entries above describe it.
