# End of turn on assistant-directed speech (2026-09-30)

**Verdict: on speech aimed at a voice assistant, our end of turn is the slowest of the three and cannot tell a
finished request from a mid-sentence stop.** The shipped rule answers 1.22 s after the audible end at p50
(1.58 s at p95). `--turn-preset fast` answers in 1.16 / 1.42 s. Pipecat smart-turn answers in 0.21 / 0.24 s and
LiveKit in 0.55 / 3.02 s. Our turn head almost never reaches p ≥ 0.99 here: 172 of 175 complete clips and 223 of 224
incomplete clips end on the 640 ms silence fallback. So every incomplete clip gets a turn_end within 2 s. Accuracy is
41-43 % for us, 70-73 % for the baselines, and 97.0 % for the smart-turn classifier alone.

**Update, same day: energy gate + smart-turn bridge** ([below](#energy-gate-and-the-smart-turn-bridge)). The energy
gate now ships as the default (`--energy-gate on`): it removes the turn_ends fired before the user spoke (137 -> 0
served sessions) and the early fires on complete clips (6.3 -> 0 %), accuracy 41.1 -> 43.9 %, with calls, AMI and the
bundled clip unchanged. It does not make the rule faster: an energy-quiet silence would (1218 -> 611 ms) but breaks
calls / AMI at every setting. The opt-in `--turn-model smartturn` (a second model, a bridge until turn head v5)
reaches 95.7 % accuracy at 770 ms p50 with 5.8 % false fires on incomplete clips, and misses more turn ends on calls.

Our published EOT numbers (research/EOT_LATENCY.md: calls 956 ms p50, AMI 1326 ms) come from human-to-human speech
with long pauses inside turns. This note measures the voice-agent case on smart-turn's own data.

## Result

| system | EOT p50 (ms) | EOT p95 (ms) | false fire on incomplete (≤ 3 s; ≤ 2 s) | missed on complete | accuracy |
|---|---|---|---|---|---|
| ours, balanced (shipped `vad_head`) | 1218 | 1581 | 100 %; 99.6 % | 0.0 % | 41.1 % |
| ours, `--turn-preset fast` | 1156 | 1420 | 100 %; 100 % | 0.0 % | 42.9 % |
| Pipecat smart-turn v3.2 + Silero (defaults) | 211 | 242 | 40.2 %; 25.4 % | 13.1 % | 69.7 % |
| LiveKit EnglishModel + Silero (0.5 / 3.0 s) | 547 | 3016 | 44.6 %; 17.0 % | 4.6 % | 72.7 % |
| *smart-turn v3.2 classifier alone (one call per clip, no timing; runs/smartturn_audit.json)* | – | – | *2.7 %* | *3.4 %* | *97.0 %* |

`turn_end_hint` (p ≥ 0.8 after 80 ms of VAD silence, balanced session):

| hint | value |
|---|---|
| latency after the audible end, p50 / p95 | 803 / 1238 ms |
| precision (hints at a complete clip's end / all hints; 67 sent) | 53.7 % |
| recall on complete clips | 20.0 % |
| incomplete clips that got a hint | 12.5 % |

EOT = decision audio time + measured compute (ours: + the session's chunk p50, median 36.7 ms; Pipecat: + the
smart-turn call). Source: `runs/eot_assistant.json`.

## Why

- **The served VAD head does not hear the clips' room tone as silence.** After the last word, these clips keep
  240 ms of room tone (median; p90 789 ms) at about −50 dBFS. The VAD head reads 0.66 there (median), above the rule's
  0.4, so our silence starts only when the digital pad starts. Measured from the clip end, ours answers in 990 ms
  (fast 971 ms). The same tail is in research/VAD_TAIL.md.
- **The head path does not fire.** On this domain the turn head rarely reaches p ≥ 0.99 (research/TURN_V4.md: 63 %
  accuracy at p ≥ 0.5 on these clips). The rule falls back to its silence timeout, which fires on incomplete clips as
  often as on complete ones. The fast preset changes the timeout, not that.
- **The baselines' misses are timing effects.** Pipecat fires at mid-utterance pauses: 17.1 % of complete clips and
  21.0 % of incomplete clips fire before the end. When its model says "incomplete" at a true end, the 3 s `stop_secs`
  fallback lands outside the window, and that clip counts as missed (13.1 %). LiveKit's 3.0 s max delay and Pipecat's
  3 s fallback sit at the window's 3 s edge, so read the ≤ 2 s false-fire column as well.

**Session start (left out of the table).** Each clip is a fresh session. For its first few frames the VAD head reads
about 0.55 on leading silence, so the fallback fires a turn_end before the user speaks. This happened in 137 of 399
balanced sessions and 93 fast ones. Those turn_ends carry an empty final, so the table drops turn_ends decided before
the clip's speech onset (the first Silero chunk > 0.5). Kept in, balanced accuracy is 26.8 % and fast 33.1 %
(`systems_raw_incl_pre_speech_turn_ends`). In a live session this happens once, at the start.

## Energy gate and the smart-turn bridge

Source: `runs/eot_energy_gate.json` (scripts/research/eot_energy_gate.py). Every row runs the SERVED objects
(`VadHeadPolicy`, `EnergyGate`, `SmartTurnModel` / `SmartTurnTrigger`). They are replayed on the stored per-frame
dumps of all three benchmarks, with frame energies recomputed from the same audio (frame v's samples known at its
decision-ready time, as `Session._energy` reads them: an ASR frame can be ready before its 80 ms end). Replay with the gate off = the
published rows exactly (1218 / 956 / 1326 ms).
- **served** rows: fresh eot_assistant dumps of the changed server.
  - `dump_gate` and `dump_smartturn_energy`: equal the replay on 399 of 399 clips.
  - `dump_smartturn`: equals it on 399 of 399 too.
  - Bundled clip: `served_check` (six deliveries) equals the gated replay in all six.

The EOT total adds the session's measured chunk p50 and, for smart-turn, the deciding call's compute. The served
dump_gate ran next to the v5 agent's training, so its chunk p50 was ~51 ms instead of 36.7 ms (all served dumps: 47-51 ms). Its decision p50 /
p95 are identical to the shipped row's: 1184 / 1546 ms.

| system | assistant EOT p50 / p95 (ms) | FF incomplete (≤ 3 s) | missed / early (complete) | accuracy | pre-speech turn_ends | calls p50 / FI / missed | AMI p50 / FI / missed | bundled clip |
|---|---|---|---|---|---|---|---|---|
| shipped `vad_head` balanced, VAD only (`--energy-gate off`) | 1218 / 1581 | 100 % | 0.0 / 6.3 % | 41.1 % | 137 / 399 | 956 / 20.2 / 7.3 | 1326 / 10.5 / 33.5 | no cut, +976 ms |
| **shipped + energy gate (default)**, balanced, served | 1240 / 1628 (decision 1184 / 1546) | 100 % | 0.0 / 0.0 % | **43.9 %** | **0** | 956 / 20.2 / 7.3 | 1326 / 10.5 / 33.5 | no cut, +976 ms |
| shipped + energy gate, `--turn-preset fast` | 1156 / 1420 | 100 % | 0.0 / 0.0 % | 43.9 % | 0 (93 without) | 886 / 24.8 / 3.7 | 1086 / 16.0 / 30.5 | no cut, +816 ms |
| *not shipped:* + energy-quiet silence X 6 dB (`--energy-quiet-db 6`) | 611 / 833 | 100 % | 3.4 / 5.1 % | 41.6 % | 0 | 582 / **48.6** / 5.5 | 1006 / **30.5** / 18.0 | **cuts** |
| *not shipped:* X 3 dB, fallback re-tuned to 960 / 1280 ms | 994 / 1453; 1314 / 1773 | 100 % | 0 / 0.6 %; 0 / 0 % | 43.6; 43.9 % | 0 | 930 / 31.2 / 6.4; 1196 / 24.8 / 9.2 | 1326 / 17.5 / 23.0; 1566 / 12.0 / 28.0 | no cut |
| *not shipped:* digital silence (< −85 dBFS) counts as quiet | 835 / 1319 | 100 % | 0 / 0 % | 43.9 % | 0 | 789 / 22.9 / 7.3 | 1326 / 11.0 / 33.5 | **cuts** |
| **`--turn-model smartturn`** (VAD trigger 160 ms, 3 s fallback), balanced, served | **770 / 1004** (decision 691 / 910) | **5.8 %** | 1.7 / 1.1 % | **95.7 %** | 4 | 629 / 19.3 / **23.9** | 1087 / 13.0 / **36.5** | no cut, +496 ms |
| `--turn-model smartturn`, `--turn-preset fast` (VAD < 0.6) | 554 / 795 | 13.4 % | 3.4 / 2.9 % | 90.5 % | 8 | 514 / 30.3 / 25.7 | 846 / 25.0 / 29.5 | no cut; +336 ms, or +4336 ms (fallback) on 3 of 6 |
| `--turn-model smartturn --smartturn-trigger energy` (two clocks: X 6 dB 240 ms, p > 0.97; VAD fallback 640), served | 290 / 1267 | 100 % | 8.0 / 14.3 % | 37.6 % | 0 | 961 / 24.8 / 8.3 | 1247 / 18.0 / 33.5 | no cut, +976 ms |
| two clocks, p > 0.5 (Pipecat's threshold), 240 ms | 250 / 540 | 100 % | 10.9 / 20.6 % | 34.8 % | 4 | 438 / **49.5** / 9.2 | 739 / **47.5** / 21.0 | no cut |
| Pipecat smart-turn v3.2 + Silero (defaults; 3 s fallback) | 211 / 242 | 40.2 % | 13.1 / 17.1 % | 69.7 % | 0 | 237 / 35.8 / 24.8 | 385 / 27.5 / 44.5 | – |
| LiveKit EnglishModel + Silero (0.5 / 3.0 s) | 547 / 3016 | 44.6 % | 4.6 / 2.9 % | 72.7 % | 0 | 567 / 26.6 / 22.9 | 1890 / 12.5 / 67.5 | – |

The FF column counts any turn_end within 3 s of an incomplete clip's cut. So a silence fallback shorter than about
2.5 s fires on nearly every incomplete clip, whatever the classifier says. Our 640 / 720 ms timers score 100 % there;
Pipecat's and our smartturn rows' 3 s fallbacks mostly fall outside the window. A 3 s fallback also costs misses on
calls and AMI: a human-to-human turn end the classifier calls "incomplete" is answered 3 s late, often after the next
talker has started. The timer is a different trade in each design, so both are shown.

**Energy gate (shipped).** Per session, a noise floor: the 10th percentile of the 80 ms frame log energies of the last
3 s. A turn is armed only by an onset frame (VAD > 0.5 AND energy > floor + 6 dB), and nothing fires before 160 ms of
onset frames (warm-up guard).
- Onset Y 3-8 dB and warm-up 0-4 frames were swept. Y 6 / warm-up 2 is the setting that removes the pre-speech
  turn_ends and keeps calls, AMI and the clip exactly unchanged.
- A 300 ms warm-up costs one AMI turn end (missed 33.5 -> 34.0 %), so the guard is 160 ms.
- No `agent_end` / `enroll` guard was added: none of the three benchmarks has a spurious turn_end after those
  messages. The assistant clips have no enrollment; calls, AMI and the clip enroll at t = 0.
- The hint tracker sees the same gate: no `turn_end_hint` before the warm-up.

**Why the energy-quiet silence is not shipped.** The frame is quiet if VAD < thr OR energy < floor + X (X swept over
3-12 dB, onset X + 3, fallback 640-1280 ms, θ 0.8-0.99, k 160-320 ms). On the assistant clips it removes the VAD
head's tail, and EOT falls to 578-1000 ms at the 640 ms fallback. On calls and AMI, mid-turn pauses are the same room tone with the VAD at
0.6-0.9 (`scripts/research/eot_energy_gate.py`, the diagnosis in the log). The tail is the hangover that keeps those
turns whole (research/VAD_TAIL.md).
- No quiet-gate row of either preset holds calls FI ≤ 20.2 % and AMI FI ≤ 10.5 % (`sweep`: 0 of 18 per preset), and
  neither does any row of the wider θ / k / fallback scan.
- Treating digital silence as quiet fails too. The calls user channels and the bundled clip carry digital zeros inside
  turns. It stays available as `--energy-quiet-db X` (off by default).

**`--turn-model smartturn` (opt-in bridge).** Pipecat's smart-turn v3.2 ONNX runs inside our server
(audioforge/server/smartturn.py; BSD-2-Clause, the copy bundled with `pipecat-ai`). The rule asks it once per silence
run, at 160 ms of VAD silence. Its input is the turn from its first onset − 0.5 s (never before the last turn_end), the
last ≤ 8 s, int16-quantised, with Pipecat's own log-mel. P(complete) equals Pipecat's `LocalSmartTurnAnalyzerV3` to
1e-6 (tests/test_energy_gate.py). Complete -> `turn_end` (`path: "model"`); incomplete -> it waits for the next silence
run, or for 3 s of silence.
- **Compute:** 1.10 calls per assistant clip, 17.4 ms p50 / 30.6 ms p95 per call on 2 CPU threads (served; 21.9 /
  25.7 ms over the 11 411 offline calls). 38 ms on 1 thread.
- **It is a second model.** The native answer is turn head v5's classifier.
- **The accuracy comes from the late trigger.** The VAD's tail bridges the clips' mid-utterance pauses, so smart-turn
  sees the whole utterance, near its 97 % classifier-alone accuracy.
- **Triggered early** (energy quiet at 160-240 ms, like Pipecat's Silero stop), it answers in 150-280 ms. It then
  fires early, at an internal pause, on 15-29 % of complete clips (Pipecat: 17 %).
- **On human-to-human calls, triggered early at Pipecat's 0.5 threshold, it interrupts about half of the turns**
  (49.5 % FI): it calls mid-turn pauses complete.
  - p > 0.97 (`--smartturn-trigger energy`) brings calls FI to 24.8 % (today 20.2 %) and keeps 290 ms p50 on complete
    assistant clips (served). Every incomplete clip still ends at the 640 ms timer (FF 100 %).
  - p > 0.99 restores calls (13.8 % FI, 7.3 % missed) but loses the speed (1193 ms).
- **The ≤ 300 ms / FF ≤ 10 % target is not reachable with one rule here.** ≤ 300 ms needs the early trigger, FF ≤ 10 %
  needs a ≥ 3 s timer, and the two together (one clock, energy X 6, 3 s) give 152 ms at 64 % FF with 29 % early fires.

**Design note for the native classifier (turn head v5), from these runs:**
1. Keep two clocks. The classifier is asked early, at energy-or-VAD quiet 160-240 ms. The fallback timer stays on
   the VAD's silence, 640 / 720 ms: today's calls FI with a timer; with a 3 s timer, accuracy on assistant speech.
2. The classifier must be right at short internal pauses of human-to-human speech (smart-turn interrupts ~50 % of
   calls turns there), not only at clip ends.
3. Its decision threshold is a per-domain operating point (0.5 assistant, ~0.97 calls).
4. The timer length decides the FF column more than the classifier does.

## Protocol

- **Data.** pipecat-ai/human_5_all (BSD-2-Clause, `data/smartturn`), the 399 clips with ids in
  smart-turn-data-v3.2-test: 175 complete, 224 incomplete, median 4.32 s. Each clip is padded with 3 s of digital
  silence. No other assistant-directed set with turn labels is local: `data/` holds calls, meetings and TTS dialogue.
- **Reference end e.** The later of the last Silero chunk > 0.5 and the energy-VAD utterance end
  (`audioforge.datasets.smartturn.utterance_end_frame`), clipped to the clip end. It deliberately does not use
  TURN_V4's `audible_end`, which also reads our VAD head. That head's tail would put e at the clip end on 90 % of the
  clips, and Pipecat's correct stops would then count as early.
- **Complete clips.** EOT latency, missed and early are counted exactly as `eot_latency.score_session` counts them,
  with the horizon at the padded end (clip end + 3 s). A clip is correct when it is answered and nothing fired early.
- **Incomplete clips.** A false fire is any turn_end before e + 3 s. The JSON also splits them by when they fired:
  before the cut, and by +0.5, +1, +2 and +3 s. Accuracy = (complete correct + incomplete not fired) / 399.
- **Ours.** The served --mode single engine (`Session`, stage1_served_v2.afm, `vad_head` defaults), fed each clip
  alone as the user channel in 20 ms blocks. There is **no voice print**: nobody else talks, so the TS-VAD others path
  cannot fire, and the served policy only reads P(user) / P(other) after enrollment. The balanced rows use the served
  turn_end messages. The fast rows run `eot_latency.sim_room` with the fast rule (VAD < 0.6 for ≥ 480 ms AND
  p ≥ 0.99, OR 720 ms) over the same session's per-frame signals. The simulator on the balanced rule reproduces the
  served turn_ends on 399 of 399 clips. `--turn-preset fast` is not committed yet, so the rule is used as specified.
  The hint comes from the served `turn_end_hint` messages.
- **Baselines.** `eot_latency.cmd_baselines` runs unchanged on the padded clips: Pipecat 1.12's own
  `LocalSmartTurnAnalyzerV3` replay, and LiveKit EnglishModel on our streaming ASR's text. Both use Silero v5 on the
  same audio. Pipecat's STT-transcript wait is not modelled, which flatters Pipecat.

To reproduce:

    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py dump --budget 540       # x2
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py baselines --budget 540
    PYTHONPATH=. .venv/bin/python scripts/research/eot_assistant.py score

Energy gate / smart-turn bridge (runs/eot_energy_gate.json):

    PYTHONPATH=. .venv/bin/python scripts/research/eot_energy_gate.py sweep
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_energy_gate.py smartturn
    EOT_ASSISTANT_DUMP=$WORK/dump_gate PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py dump --budget 540
    EOT_ASSISTANT_DUMP=$WORK/dump_smartturn EOT_ASSISTANT_ENGINE='{"turn_model": "smartturn"}' PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py dump --budget 540
    EOT_ASSISTANT_DUMP=$WORK/dump_smartturn_energy EOT_ASSISTANT_ENGINE='{"turn_model": "smartturn", "smartturn_trigger": "energy"}' PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_assistant.py dump --budget 540
    PYTHONPATH=. .venv/bin/python scripts/research/eot_energy_gate.py served
    PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py served_check [--preset fast] [--turn-model smartturn]

Scratch data is in `/Volumes/ExternalSSD/nvidia-audio-models/scratch/eot_assistant/`.
