# End-of-turn latency of single mode, measured the standard way (2026-09-30)

**Update (after fe28a9e, the TS-VAD print fix):** the rule below cut the bundled quickstart clip mid-question. The
defaults are now VAD < 0.4 for ≥ 160 ms AND p ≥ 0.99, OR 640 ms, OR the others path (`--vad-wait-ms 160,640`, θ 0.99).
See "After the print fix: the quickstart cut" and the "after the print fix" rows below.

Code: `scripts/research/eot_latency.py`. Numbers: `runs/eot_latency.json`. Served rule: `turn_policy vad_head`
(`audioforge/server/policies.py` `VadHeadPolicy`), **`--mode single`'s default since this note** (`--turn-policy
vad_head`, flags `--vad-wait-ms`, `--others-wait-ms`). Tests: `tests/test_vad_head_policy.py`, `tests/test_single_mode.py`.

## Result

All systems were scored by the same offline harness on the same audio. Signals were dumped after the 70 ms
chunk-trigger fix (commit d832fca, research/LATENCY_BUDGET.md) on the shipped `stage1_served_v2.afm`. EOT total
(the decision's input complete + its compute) is the standard number.

Two-party user-channel sessions (the user's audio alone; 32 sessions, 109 reference turn ends):

| rule | EOT p50 | EOT p95 | false-interruption % | missed % |
|---|---|---|---|---|
| today: `hybrid_dyn --dyn-wait-ms 2000,960` (Silero silence OR head p ≥ 0.99748) | 1287 ms | 1923 ms | 22.9 | 7.3 |
| shipped until fe28a9e: `vad_head` (VAD < 0.4 for ≥ 320 ms AND p ≥ 0.95, OR 800 ms, OR others path) | 881 ms | 1789 ms | 22.9 | 7.3 |
| previous `vad_head` option (VAD < 0.3 for ≥ 80 ms AND p ≥ 0.95, OR 960 ms) | 774 ms | 1963 ms | 27.5 | 9.2 |
| Pipecat smart-turn v3.2 + Silero (Pipecat defaults; p50 is bimodal, see the audit) | 237 ms | 3217 ms | 35.8 | 24.8 |
| LiveKit turn detector (EnglishModel) + Silero (LiveKit defaults) | 567 ms | 3127 ms | 26.6 | 22.9 |
| *after the print fix (fe28a9e dump):* | | | | |
| today: `hybrid_dyn --dyn-wait-ms 2000,960` | 1290 ms | 1921 ms | 17.4 | 6.4 |
| the fe28a9e rule (320 ms at p ≥ 0.95, 800 ms) | 916 ms | 2020 ms | 22.9 | 7.3 |
| **shipped, after the print fix: `vad_head` (VAD < 0.4 for ≥ 160 ms AND p ≥ 0.99, OR 640 ms, OR others path)** | **956 ms** | **1919 ms** | **20.2** | **7.3** |

AMI (200 dev eot-bench v2 turns, meeting audio with the other speakers, one scored target turn each):

| rule | EOT p50 | EOT p95 | false-interruption % | missed % |
|---|---|---|---|---|
| today | 1794 ms | 4285 ms | 8.5 | 37.5 |
| shipped until fe28a9e | 1330 ms | 4160 ms | 10.0 | 33.5 |
| previous `vad_head` option | 1488 ms | 3992 ms | 10.5 | 51.0 |
| Pipecat smart-turn v3.2 + Silero | 385 ms | 3992 ms | 27.5 | 44.5 |
| LiveKit EnglishModel + Silero | 1890 ms | 4295 ms | 12.5 | 67.5 |
| *after the print fix (fe28a9e dump):* | | | | |
| today | 1807 ms | 4286 ms | 8.0 | 39.0 |
| the fe28a9e rule | 1327 ms | 4162 ms | 10.0 | 34.0 |
| **shipped, after the print fix** | **1326 ms** | **3758 ms** | **10.5** | **33.5** |

The LiveKit rows need no re-run: they read the Silero confidences and the ASR text, which the print fix does not
touch, on the same clips. The Pipecat rows were re-run after the smart-turn audit (next section, `baselines4`):
calls p50 2268 → 237 ms, p95 3226 → 3217 ms, false interruptions 37.6 → 35.8 %; AMI p50 384 → 385 ms, false
interruptions 28.0 → 27.5 %; misses unchanged.

### Audit of the Pipecat row (smart-turn v3.2)

Question: is the Pipecat row's 2.3 s calls p50 a harness bug? Smart-turn's own compute is 16 ms per call, so a 2.3 s
median can only come from "incomplete" answers at true ends, which leave the turn to the 3 s `stop_secs` fallback.
Code: `scripts/research/smartturn_audit.py`. Numbers: `runs/smartturn_audit.json`.

**Verdict: the slow Pipecat answers are real, but the published p50 was fragile.** The harness had small fidelity
errors, now fixed. They did not cause the slow answers. They flipped a few decisions that sit near p = 0.5, and the
median moved from one mode of the latency distribution to the other.

- **The harness reproduces smart-turn's published accuracy.** `audioforge.baselines.turn.SmartTurn` on the 399
  human_5_all clips that are in smart-turn-data-v3.2-test (held out from its training) scores 97.0 % (complete
  recall 96.6 %, incomplete recall 97.3 %). On 600 train clips it scores 99.5 %.
- **The inputs are right.**
  - Audio: 16 kHz, mono, float32 in [-1, 1].
  - Level: our calls' speech is at -23 dBFS (p50), smart-turn's test clips at -19 dBFS. The smart-turn features are
    normalised to zero mean and unit variance anyway.
  - Bandwidth: both are wideband.
  - Channel: the user's own channel, not the mix.
  - Silero track: equal to Silero v5 recomputed on the same audio (max difference 0.0).
  - Model: the bundled `smart-turn-v3.2-cpu.onnx`, the same md5 as in Pipecat 1.12.
  - Settings: `SmartTurnParams` at their 1.12 defaults (stop_secs 3, pre_speech_ms 500, max_duration_secs 8) and VAD
    stop_secs 0.2.
- **What the harness got wrong, and the fix.** The old row re-implemented Pipecat's analyzer (`baselines3`). The Pipecat
  row now drives Pipecat's own `LocalSmartTurnAnalyzerV3` / `BaseSmartTurn` chunk by chunk, on a simulated clock and
  in `LLMUserAggregator`'s frame order (`audioforge.baselines.turn.pipecat_smartturn_replay`, into `baselines4`).
  Three differences with the old re-implementation:
  - **Pre-speech window.** Pipecat clears its buffer at every turn end. The next turn's 0.7 s pre-speech therefore
    never reaches back before that end. The old harness took 0.7 s regardless, including the previous turn's tail.
  - **Speech start.** Pipecat sets the speech start at the chunk after the VAD start frame, one 32 ms chunk later
    than the old harness.
  - **Features.** Pipecat 1.12 computes the log-mel with its own numpy code; the old harness used transformers'
    `WhisperFeatureExtractor`. The two differ by at most 2e-5, but the int8 model turns that into up to 0.13 of p.
  - 50 of 232 sessions differ between the two harnesses. With transformers features inside the replay, 42 still
    differ, so the pre-speech window is the main cause.
- **Why the p50 is fragile.** Pipecat's answered calls ends split into 44 model answers at about 0.2 s and 38
  fallback answers at about 3.1 s. The median sits at the boundary between the two groups.
  - A session bootstrap puts the p50's 90 % interval at 196-2784 ms. Only 52 % of answered ends come within 0.5 s.
  - The p95 (3217 ms), the false interruptions and the misses are the stable numbers.
- **Smart-turn does say "incomplete" at about half of the true ends** (Pipecat's own classes, defaults):
  - 93 of the 109 calls ends get a smart-turn call. The first call after the end says "complete" on 45 of 109 (41 %;
    45 of 93 = 48 %). The p of those first calls is bimodal: 37 below 0.2, 30 above 0.8.
  - The last call inside each turn's span (the call at its audible end) says "complete" on 52 of 106.
  - Where the other party then takes the floor for more than 1 s (a clear turn end), it says "complete" on 30 of 68.
  - The 16 ends without a call: the VAD stop fell more than 80 ms before the labelled end. The labelled end is later
    than the VAD's speech end by more than 120 ms at 27.5 % of ends. That call then counts as a false interruption
    and the end as missed; this is the same for every system.
  - On its own test clips smart-turn is 97 % right, so our clips are the difference. They are human-to-human
    conversation (TurnBench, otoSpeech), where turns often trail off; smart-turn was trained on speech addressed to
    an assistant.
  - The ~25 ms seen in production is the per-call compute (16 ms here). The fastest possible answer after the end of
    speech is the 0.2 s VAD stop plus that call; the model path here answers at 0.16-0.24 s after the labelled end.

**A tuned Pipecat: `stop_secs` swept** (Pipecat's own classes; everything else at its defaults):

| stop_secs | calls p50 (90 % CI) | calls p95 | calls FI % | calls missed % | first call "complete" | AMI p50 | AMI p95 | AMI FI % | AMI missed % |
|---|---|---|---|---|---|---|---|---|---|
| 3.0 (default) | 236 ms (196-2784) | 3217 ms | 35.8 | 24.8 | 45 / 109 | 384 ms | 3992 ms | 27.5 | 44.5 |
| 2.0 | 1302 ms (226-2042) | 2230 ms | 41.3 | 16.5 | 44 / 109 | 448 ms | 3613 ms | 28.0 | 41.5 |
| 1.5 | 550 ms (214-1504) | 1751 ms | 45.9 | 11.9 | 48 / 109 | 448 ms | 3600 ms | 30.0 | 39.5 |
| 1.0 | 232 ms (195-486) | 1237 ms | 52.3 | 11.0 | 54 / 109 | 592 ms | 3594 ms | 33.5 | 35.5 |
| 0.8 | 227 ms (192-576) | 1013 ms | 55.0 | 11.0 | 53 / 109 | 704 ms | 3403 ms | 36.0 | 32.5 |
| 0.5 | 194 ms (182-232) | 724 ms | 63.3 | 10.1 | 59 / 109 | 656 ms | 3664 ms | 37.5 | 23.5 |

A shorter fallback cuts the p95 and the misses. It adds false interruptions, because it fires in the pauses where
smart-turn said "incomplete". With `stop_secs` 0.8 s Pipecat misses 11.0 % of calls ends (shipped `vad_head`:
7.3 %), with 55 % false interruptions (shipped: 20.2 %).

**LiveKit, checked the same way.** LiveKit Agents 1.8.3's defaults are min_delay 0.5 s and max_delay 3.0 s
(`livekit.agents.voice.turn`; 6 s was the old default). The harness uses these. Its first answer at the calls ends is
the 0.5 s path (P ≥ threshold) on 56 of 109, the 3 s path on 28, and missing on 25. Sweeping max_delay, calls
(p50 / p95 / FI % / missed %):
- 6.0 s: 544 / 5939 ms / 24.8 / 33.9.
- 3.0 s: 564 / 3127 ms / 26.6 / 22.9.
- 1.5 s: 594 / 1639 ms / 25.7 / 12.8.
- 1.0 s: 594 / 1098 ms / 33.9 / 12.8.
- 0.8 s: 567 / 940 ms / 37.6 / 12.8.

No harness problem there. Its p50 interval (541-674 ms) is narrow, because its model path dominates.

The next two bullets compare the rule shipped until fe28a9e with today's on the pre-fix dump.

- **Shipped vs today, calls:** 406 ms faster at p50 and 134 ms at p95, with the same false interruptions (25 of 109)
  and the same misses (8 of 109).
- **Shipped vs today, AMI:** 464 ms faster at p50, 4 points fewer missed ends (67 vs 75 of 200), 1.5 points more
  false interruptions (20 vs 17 of 200).
- **The rule** (the defaults since the print fix; until fe28a9e the head path was 4 frames at p ≥ 0.95 and the
  fallback 10 frames):
  - Head path: the served VAD head below 0.4 for ≥ 2 frames (160 ms) and the turn head's p ≥ 0.99.
  - Fallback: 8 frames (640 ms) of that silence.
  - Others path (needs the enrolled TS-VAD track): the user's own silence (P(user) < 0.5) reaches 960 ms while
    P(other) ≥ 0.9 has held for 640 ms. Someone else has the floor, so the rule does not wait for the room to go quiet.
  - One `turn_end` per user turn. No Silero.
- **Decision latency** (= total − compute, the audio time at which the deciding input was complete) is 31 ms below
  the total for ours: calls 850 ms, AMI 1296 ms p50 (after the print fix, shipped rule: 926 / 1296 ms). For the baselines it is the total minus the model call.
- **Compute per decision:**
  - Ours: 31 ms per 160 ms chunk (p50; p95 34 ms) on the Mac CPU with 2 threads. The VAD, TS-VAD and turn heads run
    on every frame anyway.
  - Silero: 0.087 ms per 32 ms chunk.
  - smart-turn v3.2: 15.9 ms per call (p50; Pipecat's features + model), added to its EOT.
  - LiveKit EnglishModel: 1.1 ms per call, hidden under its 0.5 s minimum delay.

**What changed in the default.** `MODES["single"]` adds `--turn-policy vad_head`, the server default for a session
whose `config` names no `turn_policy`. A client's `config` still wins. `audioforge.load()` sessions,
`examples/quickstart_client.py` and `examples/python_api.py` use it too. `vad_head`'s defaults became that rule:
`--vad-wait-ms 320,800`, VAD < 0.4, θ 0.95, `--others-wait-ms 960,640` (since the print fix: `160,640`, θ 0.99). Single mode no longer downloads Silero
(`audioforge-download` default set: asr + tsvad [+ lid]). `audioforge-serve` in single mode no longer passes or
preloads `--silero`. A client that asks for `hybrid_dyn` gets it when Silero is at serve's default path or given with
`--silero`, else `hybrid` with a `silero_unavailable` notice. Room mode is unchanged: `--turn-policy` defaults to
`timeout`, and without an enrolled TS-VAD track `vad_head` has no others path.

**Caveats** (of the rule shipped until fe28a9e, on the pre-fix dump).
- The rule was picked on both corpora. 16 of 13 142 rules meet the goal, so neither corpus is held out any more.
- The margins are one turn or so. Calls false interruptions equal today's (22.9 %). AMI false interruptions are 3
  turns above today's.
- A neighbour with more margin on calls: the head path at 400 ms instead of 320 ms. It gives 948 ms p50, 17.4 %
  false interruptions and 7.3 % missed on calls, and 1407 ms, 9.5 % and 34.5 % on AMI.
- The goal thresholds (calls ≤ 23.9 % / 8.3 %, AMI ≤ 38.5 % missed at ≤ ~10 % false interruptions) are today's
  numbers from the pre-fix dump. On the re-dump today scores 22.9 / 7.3 and 37.5 / 8.5. The shipped rule meets the
  re-dump's calls numbers and AMI misses. It does not meet the AMI false-interruption number (10.0 vs 8.5).

## After the print fix: the quickstart cut

The print fix (fe28a9e, research/TSWER.md) changed the TS-VAD columns and, through them, the turn head's p. The sessions
were re-dumped into `scratch/tswer_fix/eot_dump`, which is now the script's `DUMP` (dump3 = the pre-fix dump). On it
the fe28a9e rule scores 916 / 2020 ms, 22.9 / 7.3 % on calls, and 1327 / 4162 ms, 10.0 / 34.0 % on AMI.

**The cut.** Streamed with `examples/quickstart_client.py` at 1x, the bundled clip's question was cut at 9.06 s.
- The user pauses after "life", 8.6-9.1 s.
- There the client's input (float → int16 by truncation) gives the served VAD 0.32, 0.40, 0.21, 0.37, 0.20, 0.12 on
  six frames, and the turn head 0.94-0.98.
- The 0.40 frame is the last one ≥ 0.4, so the silence reaches 4 frames (320 ms) on the frame decided at 9.056 s,
  with p 0.983 ≥ 0.95.
- The float input gives 0.45 and 0.41 on those two frames, so there the silence stops at 2 frames and there is no
  cut. That is why the in-process runs did not show it.
- The VAD does not hover at the line: it reaches 0.12-0.21 inside the pause. It is a real half-second pause with a
  confident head.

**How candidates were checked on the clip.** `served_check` now records the clip's frames under six deliveries
(`CLIP_VARIANTS`):
- float;
- int16 truncated, as the quickstart client sends it;
- int16 rounded;
- −6 dB and +6 dB, truncated;
- −70 dBFS dither, rounded.

The sweep counts a rule as cutting the clip if it puts a `turn_end` inside the user's turn (3.9 s to 10.8 − 0.08 s)
under any of them. The fe28a9e rule cuts it under four of the six.

**Candidates** (the sweep's (g) family, each a change of the fe28a9e rule; calls = the 32 two-party sessions):

| rule | calls EOT p50 / p95 | calls false int. / missed % | AMI EOT p50 / p95 | AMI false int. / missed % | deliveries cut |
|---|---|---|---|---|---|
| fe28a9e rule (320 ms at p ≥ 0.95, 800 ms) | 916 / 2020 | 22.9 / 7.3 | 1327 / 4162 | 10.0 / 34.0 | 4 of 6 |
| hysteresis: silence from VAD < 0.35, speech again at ≥ 0.5 (320 ms) | 906 / 1591 | 22.9 / 7.3 | 1326 / 4127 | 11.0 / 34.0 | 6 |
| hysteresis: < 0.3 / ≥ 0.5 (320 ms) | 946 / 1778 | 22.0 / 7.3 | 1327 / 4170 | 11.0 / 35.0 | 6 |
| hysteresis: < 0.3 / ≥ 0.5 (400 ms) | 1056 / 1778 | 20.2 / 7.3 | 1326 / 4178 | 9.5 / 36.0 | 2 |
| head path at 400 ms (the neighbour in the caveats) | 960 / 2011 | 17.4 / 6.4 | 1327 / 4174 | 9.5 / 35.5 | 2 |
| 400 ms + others path at 800 ms user silence | 960 / 2011 | 17.4 / 6.4 | 1326 / 4206 | 10.5 / 34.0 | 2 |
| ≥ 640 ms of VAD speech before an end | 916 / 2020 | 18.3 / 7.3 | 1327 / 4242 | 10.0 / 34.0 | 4 |
| p ≥ 0.98, 400 ms, 640 ms fallback | 953 / 1916 | 17.4 / 6.4 | 1326 / 4046 | 10.5 / 35.5 | 1 |
| **p ≥ 0.99, 160 ms, 640 ms fallback (shipped)** | **956 / 1919** | **20.2 / 7.3** | **1326 / 3758** | **10.5 / 33.5** | **0** |
| VAD < 0.5 for 480 ms, p ≥ 0.99, 640 ms fallback | 899 / 1599 | 22.9 / 7.3 | 1247 / 4046 | 13.5 / 34.0 | 0 |

- **Hysteresis cannot fix this pause.** The VAD falls to 0.2 inside it, so the silence starts early and a high reset
  line (0.5) only makes it longer.
- **The minimum-speech guard does not apply.** The user has spoken for 4 s before the pause.
- **The 400 ms neighbour** misses 1.5 points more AMI ends (35.5 %). Under the −6 dB and dither deliveries the silence
  reaches 5-6 frames, so it still cuts the clip.
- **What does fix it: a surer head.** The head reads at most 0.988 in the pause under any delivery. With θ 0.99 the
  head path cannot fire there, and the pause (≤ 6 frames of silence) is shorter than the 640 ms fallback.
  - A surer head allows a shorter silence wait (160 ms).
  - The shorter fallback (640 ms) answers the ends where the head stays unsure. It keeps AMI misses at 33.5 %.
- **Selection** (`selection.fastest_print_fix_goal_no_clip_cut` in `runs/eot_latency.json`): the fastest calls p50
  with
  - calls false interruptions ≤ 22.9 % and misses ≤ 7.3 %, AMI misses ≤ 34.0 % (the fe28a9e rule's numbers);
  - AMI false interruptions ≤ 10.5 % (its 10.0 % + one turn);
  - no cut under any delivery.

  The last row is faster (899 ms) and was found in a wider scratch grid. It costs 3.5 points of AMI false
  interruptions (27 vs 20 of 200), so the AMI cap excludes it.
  - On the post-fix dump, the pre-fix selection (`fastest_meeting_goal_both_corpora`: the 23.9 / 8.3 / 38.5 / 10.0
    goal, no clip check) names the hysteresis rule < 0.35 / ≥ 0.6 at 400 ms: 913 ms on calls, 35.0 % AMI missed. It
    cuts the clip under five of the six deliveries.
- **Against the fe28a9e rule:**
  - Calls: 40 ms slower at p50, 101 ms faster at p95, 3 fewer false interruptions (22 vs 25 of 109), the same misses.
  - AMI: the same p50, 404 ms faster at p95, one more false interruption, one fewer miss.
  - Against today's rule on the same dump: 334 ms faster on calls and 481 ms on AMI. Today's rule itself improved
    with the print fix, to 17.4 / 6.4 % on calls. The shipped rule has 3 more false interruptions and one more miss
    there. The goal was set against the fe28a9e rule's numbers.
- **The margin is the head's.** Under −6 dB the head reads 0.988 in the pause; under the quickstart client's
  delivery, 0.983. A head that reads higher on a pause would cut again. The VAD no longer decides it.
- **The bundled clip, served** (`served_check`, the float and five int16 deliveries): no cut under any of them. The
  question ends at 11.776 s, 976 ms after its reference end; the fe28a9e rule answered at 11.936 s. Served equals
  offline in all six. Over the websocket (`audioforge-serve` on port 8797, `quickstart_client.py`, 3 runs at 1x),
  all three runs gave the same events: one `turn_end` at 11.78 s (640 ms fallback) and the full question in one
  final.
  - The +6 dB and dither deliveries also end a "turn" at 1.9-2.0 s, before the user speaks: a short VAD blip in the
    lead-in, then the 640 ms fallback. The fe28a9e rule fires there too, at 2.0-2.2 s. It is outside the user's
    turn, and the quickstart delivery does not show it.
- **Mechanism.** No code changed in the served path: `VAD_HEAD_WAIT_MS` = (160, 640), `POLICY_THETA["vad_head"]` =
  0.99, and the `VadHeadPolicy` defaults to match. `check` confirms that the policy with its defaults equals
  `sim_room(CHOSEN)` on all 555 `turn_end`s of the 232 dumped sessions. `tests/test_vad_head_policy.py` replays the
  clip's recorded frames (`tests/fixtures/two_party_call_16s_frames.json`, all six deliveries) through the defaults.
- `sim_room` gained two options for the candidates: `vad_hi` (hysteresis) and `min_sp` (minimum speech frames).

## The re-dump and the -70 ms

The first version of this note was dumped before d832fca. It ran `stage1_served.afm` (v1 VAD head); the re-dump runs
the shipped `stage1_served_v2.afm` (block-4 VAD head).

- **Frame ready times:** 98.99 % of the 56 331 frames in all 232 sessions are ready exactly 70 ms earlier. The rest
  are the last frames of a session, capped at the end of the audio. The turn head's p is identical on every frame.
- **Today's rule:** 1317 → 1287 ms p50 on calls. Only the head-bound firings move, since a Silero chunk often binds
  (LATENCY_BUDGET.md). Its false interruptions and misses each drop by one turn: 23.9 → 22.9 %, 8.3 → 7.3 %.
  - On the new frames with the old clock (ready times + 70 ms), it reproduces the pre-fix numbers exactly: 1317 /
    1950 ms, 23.9 %, 8.3 %; AMI 1798 ms, 9.5 %, 38.5 %. So the v2 VAD head does not change today's rule, and the
    whole change is the clock.
- **VAD-bound rules move by the full 70 ms:**
  - The previous `vad_head` option: 844 → 774 ms on calls, 1558 → 1488 ms on AMI.
  - The shipped rule: 951 → 881 ms on calls, 1399 → 1330 ms on AMI.
- **The v2 VAD head versus the v1 one:** it made the previous `vad_head` option worse on calls. At the old clock, v1
  gave 743 ms, 25.7 % false interruptions and 7.3 % missed. v2 gives 844 ms, 27.5 % and 9.2 %.
- **Two definitions were fixed with the re-dump:**
  - "Decision" latency used to be total − 240 ms. That counted the buffering twice: `turn_end.t` already holds the
    chunk and lookahead wait on the served clock. It is now total − compute.
  - LiveKit's transcript lookup now follows the new ASR clock: the first chunk after 9 mel frames, then every 16.
    The LiveKit rows moved slightly. Before the fix they were 570 / 3235 ms, 25.7 %, 22.9 % on calls and 2137 /
    4830 ms, 12.5 %, 68.5 % on AMI. Pipecat's rows are unchanged, since they read no ASR text.

## Room-aware silence (step 2 of the brief)

Sweep, all in numpy on the dumped frames (`sim_room`). 13 142 rules were scored in 2 min.

- **(e) Head path and fallback on each silence source.** Four sources were tried:
  - our VAD < 0.3 / 0.4 / 0.5;
  - the target's own silence, TS-VAD P(user) < 0.5;
  - `both` (user AND room quiet);
  - `either` (user OR room quiet).

  The grid was k = 1-5 frames, θ from 0.8 to 0.98, and a fallback of 800-1520 ms or none. Results:
  - Only our VAD meets the calls goal (50 rules). The fastest is 856 ms, but all of them miss 54-59 % of AMI ends.
  - The target's own silence cuts AMI misses to 18-20 %, as the first note said. On the user channel it misses
    ≥ 17.4 % of ends: the TS-VAD track lags and misses the user's short turns. At ≤ 23.9 % false interruptions its
    best is 18.3 % missed at 1207 ms, so no rule on it meets the calls goal.
  - `both` misses ≥ 11 % of call ends and ≥ 42 % of AMI ends. `either` misses ≥ 11.9 % of call ends.
- **(f) The others path on the 20 fastest (e) rules that meet the calls goal.** Parameters tried:
  - P(other) ≥ 0.5 / 0.7 / 0.9, held for 1 / 4 / 8 frames;
  - user silence 320-2000 ms, checked once at that frame or at any frame after;
  - optionally a head-gated variant (user silence ≥ 160 / 320 ms AND p ≥ 0.95 / 0.9).

  This path attacks the reason AMI misses ends: after the target stops, the others keep the any-speaker VAD up.
  - **The shipped base alone:** 55.0 % AMI missed. **With the others path:** 33.5 %. On calls it adds one false
    interruption (22.0 → 22.9 %). P(other) is mostly low on a user channel: ≥ 0.9 on 0.2-2.8 % of frames, against
    26.6 % on AMI. So the path fires once in all 32 calls (TurnBench tb_160 at 2.0 s), where the TS-VAD track takes
    the user's own opening words for "other".
  - **Why the path needs a hold and a single check:** a lower P(other) threshold, a short hold, or a check at any
    frame after the silence lets backchannels and the TS-VAD's onset lag end the target's turn mid-sentence. Those
    variants reach 14-40 % AMI false interruptions.
  - **Why no head gate:** gating the path on the head (p ≥ 0.9 / 0.95) did not help. On AMI the head confirms only
    20-25 % of true ends within 300 ms (table below).
- **Selection:** the fastest calls EOT p50 with calls false interruptions ≤ 23.9 %, calls missed ≤ 8.3 %, AMI missed
  ≤ 38.5 % and AMI false interruptions ≤ 10.0 %. Ties go to the faster AMI EOT, then the fewer calls false
  interruptions. 16 rules qualify; the fastest is the shipped one. One rule serves both corpora, so no per-corpus
  flag is needed.

## Metrics (defined once)

For each reference user turn [s, e] (e = the reference end of the user's speech):
- **EOT latency** = (the first `turn_end` at or after e − 80 ms, before min(e + 6 s, the user's next turn onset)) − e.
  Reported as p50 / p95 over the answered ends.
  - **total** = the audio time at which the decision's input was complete, plus the measured compute of that
    decision. This is the standard number.
  - **decision** = total − compute. For ours, the input time is the frame's decision-ready time on the served ASR
    clock (`Session._asr_ready_t`), which already holds the chunk and lookahead buffering. For the baselines it is
    the end of the deciding Silero chunk.
  - The floor of any rule on our frames is firing on the first frame wholly after e (k = 1). It is 137 ms total and
    106 ms decision at p50 on calls, and 130 / 96 ms on AMI.
- **false-interruption rate** = % of reference turns with a `turn_end` in [s, e − 80 ms): the system ended the turn
  while the user was still in it and went on to speak again.
- **missed turn ends** = % of reference turn ends with no `turn_end` in [e − 80 ms, min(e + 6 s, next user onset)).

## Method

1. **Signals, dumped once** (`dump`, through `scripts/dev/gate.sh`, one torch job, runs of under 10 min, resumable;
   into `dump3`).
   - The exact `--mode single` server session (`audioforge.serve.Session` on `stage1_served_v2.afm`, TS-VAD track,
     stored 5 s voice print, no diarizer) ran over every session.
   - Stored per 80 ms frame: decision-ready time, turn-head p, served VAD head, TS-VAD P(user) / P(other). Also the
     Silero v5 confidence per 32 ms chunk, the Silero silence clock, the streaming ASR tokens per frame, and the
     per-chunk compute.
   - `check` confirms two things:
     - The numpy simulator reproduces all 471 server `turn_end` times of today's rule in all 232 sessions.
     - The served `VadHeadPolicy` with its defaults, fed the same frames, equals `sim_room` with the shipped rule on
       all 508 `turn_end`s.
2. **Data.**
   - Two-party: E2E_FINAL's 16 TurnBench + 16 otoSpeech user channels with a 6 s pad; 109 scored ends.
   - AMI: 200 seeded turns out of the 974 AMI-dev eot-bench v2 windows. The audio runs from the window start to 6.5 s
     after the turn end and includes other speakers. The print comes from 5 s of the same speaker elsewhere in the
     meeting. A turn end is not counted as answered after the target's own next turn starts.
3. **Rules, simulated in numpy.** Frames are 80 ms. "Our VAD" is the served VAD head (block 4). Families:
   - (a) VAD-head silence ≥ k frames AND p ≥ θ, with or without a fallback.
   - (b) Head only, with hysteresis.
   - (c) Today's Silero rule with FLOOR ∈ {0, 100, 200, 300, 400} ms.
   - (d) Smart-turn style: decide once at VAD silence 160 / 240 ms, else fall back.
   - (a') / (d') the same on the target's silence; (a'') on "VAD OR target quiet".
   - (e) / (f) the room-aware family (`sim_room`, previous section).

   Families (a), (b), (d), (e) and (f) use no Silero.
4. **Baselines, run offline on the same audio** (`baselines`; onnx; into `baselines4`; `baselines3` holds the Pipecat
   re-implementation used before the smart-turn audit). They read the Silero v5
   confidences the served session computed, which come from the same ONNX model and the same 512-sample chunks.
   - **Pipecat 1.12 defaults, through Pipecat's own `LocalSmartTurnAnalyzerV3`.** VAD: confidence 0.7, start / stop
     0.2 s. At each VAD stop, smart-turn v3.2 runs on the turn audio from speech start − 0.7 s (≤ 8 s). The window
     never reaches back before the previous turn end, where Pipecat clears its buffer. "Complete" (p > 0.5) ends the
     turn. Otherwise it waits for the 3 s `stop_secs` fallback.
   - **LiveKit agents 1.8 defaults.** Silero plugin: activation 0.5, min silence 0.55 s. At each END_OF_SPEECH, the
     EnglishModel runs on the current turn's transcript. The turn ends at max(EOS + compute, speech end + 0.5 s) when
     P ≥ its threshold (0.0289), otherwise at speech end + 3.0 s. A new START_OF_SPEECH cancels it.
   - **Transcript used for LiveKit:** our served streaming ASR text ready at that moment, on the served clock.
   - **What neither baseline waits for:** the STT final that Pipecat and LiveKit also wait for in a live call. Both
     baselines are therefore measured optimistically.

## What the head lacks

The rule is not held back by compute (31 ms per chunk). It is held back by the turn head, which does not separate
mid-turn pauses from turn ends in the first few hundred ms of silence.

- **Reference turns contain pauses.** 45 % of the two-party reference turns have an internal pause of at least 160 ms
  (33 % at least 400 ms, 18 % at least 960 ms). A rule that fires 80-160 ms into any silence therefore interrupts
  about 45 % of turns unless the head vetoes the pauses. That is why the shipped head path waits 320 ms.
- **The head cannot confirm most true ends fast.** Share of true ends where the posterior reaches θ within
  200 / 300 / 500 ms of the end (decision-ready frames, new clock):

  | θ | two-party (n = 109) | AMI (n = 200) |
  |---|---|---|
  | 0.5 | 77 / 78 / 81 % | 55 / 58 / 68 % |
  | 0.8 | 63 / 66 / 70 % | 36 / 42 / 53 % |
  | 0.9 | 58 / 58 / 62 % | 25 / 28 / 38 % |
  | 0.95 | 51 / 53 / 54 % | 20 / 24 / 26 % |
  | 0.98 | 28 / 32 / 36 % | 12 / 13 / 17 % |
  | 0.99748 (today) | 6.4 / 7.3 / 7.3 % | 3.0 / 3.5 / 5.5 % |

  At θ 0.95, only half of the true call ends get a fast "yes". The rest wait for the 800 ms fallback, which is why
  the p95 stays near 1.8 s.
- **What would move the numbers further:**
  - A turn head that is confident within 1-2 frames after true ends.
  - A target-speaker silence that drops as fast as the VAD head does. With it, the target's own silence could drive
    the head path on calls too, not just the others path.

## Served policy `vad_head`

- **What it does:** see "The rule" above. It is decided with the frame's 160 ms chunk. The others path reads
  `asr.tsvad_p[v]` of the same frame, and only while a voice print is enrolled.
- **How to select it:**
  - Default in `--mode single` (`--turn-policy vad_head`), or `config.turn_policy = "vad_head"`.
  - Flags: `--vad-wait-ms K,FALLBACK` (default `160,640`), `--others-wait-ms USER_SIL,HOLD` (default `960,640`, 0 =
    off).
  - θ defaults to 0.99 (`POLICY_THETA`), and `eot_threshold` overrides it.
  - The thresholds (VAD 0.4, P(user) 0.5, P(other) 0.9) are `VAD_HEAD_*` in `audioforge/server/constants.py`.
- **No Silero.** The session never builds the Silero silence clock and never loads the model.
- **Checked against the offline numbers:**
  - `tests/test_vad_head_policy.py` asserts that `VadHeadPolicy` equals `sim_room` (with the others path) and
    `sim_vadhead` (without it) on random tracks, and that its defaults equal `CHOSEN`.
  - `check` runs the policy class over all 232 dumped sessions (above).
  - `served_check` runs the real single-mode engine on the bundled clip (`examples/audio/two_party_call_16s.wav` + its
    voice print) under the six `CLIP_VARIANTS` deliveries. It writes their frames to `WORK/clip_frames.json` for the
    sweep's clip check.
    - Float input: the served `turn_end`s at 11.776 / 15.296 s equal the offline rule on the same frames
      (`served_equals_offline: true`), and so do the other five deliveries.
    - On the clip's one reference end (10.8 s) the head reaches p 0.88 < 0.99, so the 640 ms fallback answers at
      11.776 s. Today's rule answers at 12.096 s.
    - Until fe28a9e the served rule answered at 11.936 s (float), and at 9.056 s mid-question as the quickstart client
      delivers the clip.

## Reproduce

```bash
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py prepare_ami
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py dump --budget 480   # until done
PYTHONPATH=. .venv/bin/python scripts/research/eot_latency.py check
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py baselines --budget 480
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/eot_latency.py served_check   # clip frames first
PYTHONPATH=. .venv/bin/python scripts/research/eot_latency.py sweep
PYTHONPATH=. scripts/dev/gate.sh .venv/bin/python scripts/research/smartturn_audit.py inputs    # + replay, livekit, eval
```

`dump` writes to `DUMP`, which is now the post-print-fix dump `scratch/tswer_fix/eot_dump`.
