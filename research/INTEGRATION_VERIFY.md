# Integration verification: live server, Pipecat, LiveKit (2026-09-26)

An independent pass re-ran every claim from the three integration reports against the repo as it stands. CPU only,
`--threads 2`, and at most one server plus one client loaded at a time. Load average during the runs was 3.5–5.7,
with a GPU job and other agents on the machine. The AMI windows were regenerated with the server agent's
`prep_ami.py` and are byte-identical to the ones all three agents used (`cmp`).

Instead of the report's utterance 1089-134686-0000, the LibriSpeech run used a different one: test-clean
1188-133604-0003, 12.6 s.

Scratch (logs, summaries, probe scripts):
`/private/tmp/claude-501/-Users-maxm/54361310-ccc6-4257-a73f-3341f209b7ca/scratchpad/wf-integ/verify/`

## Commands run

```
# (1) tests
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve.py tests/test_pipecat_integration.py tests/test_livekit_integration.py
RUN_REAL=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve.py -k real
RUN_REAL=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_pipecat_integration.py
RUN_REAL=1 LIVEKIT_DEMO_WAV=verify/lk/ami_IS1008b_003.wav LIVEKIT_DEMO_PORT=8802 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_livekit_integration.py
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_runtime.py tests/test_streaming.py tests/test_streaming_diar.py
# (2) server + client at 1x (plain protocol, then --debug-fields for the lag check)
PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_heads_pretrained.afm --diar runs/nemo_sortformer_v2.afm --port 8801 --threads 2 [--debug-fields]
PYTHONPATH=. .venv/bin/python scripts/stream_client.py verify/audio/<win> --url ws://127.0.0.1:8801 --speed 1 --policy timeout|both --log .. --summary .. --ref ..
# independent checks of the logs: verify/check_log.py (exact key sets, NaN, monotonic t, frame spacing, turn_end->final pairing), verify/check_lag.py
# adversarial probes: verify/probe.py (digital silence, clipped noise, empty stream, odd byte frames + junk text, 8 kHz, bad config, mid-stream disconnect)
# fast-conv equivalence: verify/fastconv.py (Engine fast=True vs False on IS1008b_003)
# (3) demos on IS1008b_003 + ES2011b_027
PYTHONPATH=. .venv/bin/python examples/livekit_offline_demo.py verify/lk/ami_IS1008b_003.wav verify/lk/ami_ES2011b_027.wav --url ws://127.0.0.1:8801 --agent-session both --out verify/runs/lk_demo.json   (server with --debug-fields, as in the LiveKit agent's run_demo.sh)
PYTHONPATH=. .venv/bin/python examples/pipecat_local_demo.py run verify/pc/ami_IS1008b_003.wav verify/pc/ami_ES2011b_027.wav --url ws://127.0.0.1:8801 --policy timeout head --out verify/runs/pc_res.json   (plain server)
```

## Claim by claim

| # | Claimed | Observed | Verdict |
|---|---|---|---|
| T1 | `test_serve.py`: 37 pass, plus "2 real-model tests" with RUN_REAL=1 | 37 passed, 1 skipped. The file has only **one** RUN_REAL test. With `-k real` and RUN_REAL=1, 2 tests pass (a paced-client test also matches "real"). | True, but the count is overstated (1 real-model test, not 2) |
| T2 | Pipecat tests: 21 pass + 1 skip; 22 with RUN_REAL=1 | 21 passed, 1 skipped; RUN_REAL=1: 22 passed (35 s, real server + 1 AMI window) | Confirmed |
| T3 | LiveKit tests: 15 pass + 1 skip | 15 passed, 1 skipped; RUN_REAL=1 + LIVEKIT_DEMO_WAV: 16 passed | Confirmed |
| T4 | Existing streaming / streaming-diarizer / runtime tests still pass | 39 passed | Confirmed |
| T5 | Package versions; pipecat downgraded onnxruntime/protobuf/soundfile | pipecat-ai 1.12.0, livekit-agents 1.8.3, livekit 1.1.18, websockets 17.1, onnxruntime 1.24.4, protobuf 6.33.6, soundfile 0.13.1 | Confirmed (the downgrades are in place in the shared venv) |
| S1 | Every protocol message type appears and is well-formed | All 6 types (`ready`, `frame`, `partial`, `turn_end`, `final`, `stats`) appear. Without `--debug-fields`, every message has exactly the protocol's keys. No NaN/Inf in any run or probe, `frame.t` is strictly +0.08 s, `speakers` always has length 4, and every `turn_end` is immediately followed by a `final` with the same `t`. | Confirmed (see defects D4–D7) |
| S2 | Default policy = 1000 ms timeout on the diarizer primary; head only as `eot` + opt-in | Code: `SessionConfig.turn_policy="timeout"`. Finals are cut by the timeout unless the policy is `head`. The Pipecat and LiveKit adapters default to `"timeout"`. `serve.py` docstring, README and research/INTEGRATION.md all say the head loses on AMI (STAGE1 n=200). | Confirmed for code and docs, but the cited evidence does not match the implemented rule (**D1**) |
| S3 | Transcripts sensible; LibriSpeech WER 3.6 % on 1089-134686-0000 | Different utterance (1188-133604-0003): WER **0.0 %**. AMI WER against all speakers: 28.3 % (IS1008b_003), 25.8 % (ES2011b_027), identical to the claimed figures. | Confirmed |
| S4 | First partial: IS1008b 350 ms / ES2011b 81 ms after the first word end; ~50 ms after that audio was sent | IS1008b 357 ms, ES2011b 89 ms after the first word end. 56 / 59 / 54 ms after that audio was sent. LibriSpeech: 454 ms after the energy onset. | Confirmed |
| S5 | RTF at 1x, 2 threads: 0.43–0.52 | 0.479 (IS1008b), 0.519 (ES2011b), 0.453 (Libri); 0.488 / 0.533 in the `both` runs; 0.38–0.55 inside the demos | Confirmed |
| S6 | Keeps up at 1x; frames arrive 124–132 ms after their audio (p50); max backlog ≤ 200 ms | Frame lag p50 129–136 ms, p95 167–220 ms, max 361–525 ms, slope 2.9–4.2 ms per s (no growth). `backlog_ms_max` 180 / 200 ms (debug stats). | Confirmed |
| S7 | Chunk processing p50/p95 33–45 / 170–260 ms | 37.8–38.8 / 195–250 ms. p95 exceeds the 160 ms chunk period (the diarizer step lands every 480 ms), which is absorbed by the backlog. | Confirmed |
| S8 | As fast as possible, 2 threads: RTF 0.33 (AMI 20 s) / 0.27 (Libri) | 0.527 and 0.507 (AMI ES2011b, two runs), 0.424 (Libri 1188) at load 4.2–4.6. Speed 0 is no faster than 1x here. | **Not reproduced** (1.5x slower than claimed) |
| S9 | fast-conv: identical tokens, diarizer diff ≤ 3e-7, ~3x ASR / 2x diar; without it RTF 1.2–1.3 | IS1008b offline, 20 ms blocks: fast RTF 0.49, original conv RTF 1.13 / 1.21 (two runs). Finals are identical, and the maximum `speakers` difference is 0.0 at the protocol's 4 decimals. Chunk p50 39.6 vs 115.6 ms. | Confirmed: overall 2.5x, and without it the server cannot keep up |
| S10 | Diarizer lag: column final 560–960 ms (mean 760) after the frame ends, + compute; speaker columns in `frame` are 480–960 ms old (p50 720) | `frame.t − spk_t`: p50 720 ms, max 960 ms. In wall-clock time, a speaker column is p50 859–860 ms old when it arrives (max 1030–1034 ms). The debug stat `diar_lag_ms_mean_measured` is computed from sample counts, so it always equals 760 at 1x; it is not a wall-clock measurement. | Confirmed (the "measured" label is misleading, D8) |
| S11 | README: "ASR frames arrive within about 90 ms" | Observed frame arrival p50 129–136 ms, p95 up to 220 ms (the server report itself says 124–132 ms) | **Overstated** (90 ms is the structural 6–86 ms availability without compute) |
| S12 | Peak memory 3.0–3.4 GB | Sampled server RSS peak 3.01–3.45 GB. Clients: `stream_client` 0.20–0.21 GB, Pipecat demo 0.24 GB, LiveKit demo 0.30 GB. Total server + client ≤ 3.75 GB. | Confirmed |
| S13 | Timeout turn ends: IS1008b +1.8 s late; ES2011b cut ~0.9 s early | IS1008b: `turn_end.t` 14.96 s vs reference end 13.28 s (decision +1.68 s, arrival +1.86 s). ES2011b: turn_end at 16.88 s during the speaker's 1.56 s "um" (15.86–17.42 s); reference end 18.0 s, arrival 0.93 s early. The second turn_end is at 20.0 s, from the end-of-stream flush. | Confirmed |
| S14 | Head (default `session` input) never reaches 0.98 on 2 of 3 files | `eot` max 0.959 (IS1008b), 0.940 (ES2011b), 0.836 (Libri): no head turn_end on the raw windows | Confirmed |
| S15 | Robustness: mid-stream disconnect, 8 kHz input, sequential connections | Probes: disconnect → "session dropped", and the next session works. 8 kHz config gives the same transcript. Odd-byte frames and non-JSON text work (logged, ignored). Digital silence and clipped noise give no NaN and no spurious turn_end. | Confirmed |
| P1 | Pipecat timeout dead-air IS1008b 1903 ms (server 1680), ES2011b 2503 ms (server 2240), 1 cut-in on ES2011b | IS1008b 1862 ms (server 1680, Pipecat +1 ms), 0 cut-ins. ES2011b 2441 ms (server 2240), 1 interruption. | Confirmed |
| P2 | Pipecat head dead-air IS1008b 2463, ES2011b 4163 ms, 0 cut-ins; head fires mostly in the added silence | IS1008b 2461 ms (server +2406 ms after the true end → t = 15.69 s, after the 15.21 s WAV ends). ES2011b 4142 ms (fires at 22.08 s, inside the 3 s pad). Both firings are in the padding. | Confirmed (and the caveat is right: without the pad the head does not fire) |
| P3 | Zero Pipecat warnings/errors, zero ordering violations, clean EndFrame | `violations: []`, `n_warn 0`, `n_pywarn 0`, `stopped_by TurnAnalyzerUserTurnStopStrategy`. No WARNING/ERROR lines in the demo log. | Confirmed |
| P4 | Server bug: a final can split a word ("every" \| "one") | Code: `_turn_end` cuts at `asr.tok_at[f-1]`, a raw token index with no word-boundary check. The existing raw_v2 logs show `"effort for every"` followed by a final starting with "one to consider…". | Confirmed (by code and logs; not hit on my 2 windows) |
| L1 | LiveKit decisions IS1008b +1680, ES2011b +2240 ms; first interim 355 / 93 ms; WER 28 % / 26 % | Decisions +1680 / +2240 ms (identical). First interim 346 / 76 ms. WER 28.3 % / 25.8 %. | Confirmed |
| L2 | LiveKit STT dead air 2902 / 2536 ms; `stt` commit 2413 / 2550 ms | 1806 / 2448 ms; `stt` commit 1814 / 2592 ms. Mine is lower on IS1008b because the server ran at RTF 0.38 vs their 0.79. | Consistent (load-dependent, as they state) |
| L3 | "stt" mode commits 2–15 ms after our turn_end arrives | `commit_ms_after_decision_arrival` = 2 ms (both windows, both modes) | Confirmed |
| L4 | Detector mode: **no early commits** on the 5 windows (ES2011b 0) | ES2011b detector mode **committed early at 17.58 s** (true end 18.0 s; `early_commits: 1`), with prediction "turn_end_before_request" at 17.18 s. This happens when the any-speaker VAD goes quiet during the "um". | **Not reproduced**: timing-dependent, 1 of 2 windows cut |
| L5 | LiveKit's own warnings: "stt end of speech received while vad is still in a speech segment", "resume_false_interruption…", "metrics_collected is deprecated" | All three seen; no tracebacks | Confirmed |

## Defects

1. **D1 (evidence mismatch, most important).** The docs justify the default with STAGE1 n=200 "timeout on Sortformer
   streaming 38.4 % misses vs head 69 %". That row does not match the product in two ways:
   - **Oracle enrollment.** `scripts/eval_stage1.py` picks the timeout's column by overlap with the oracle primary
     activity (`enroll_mode="oracle"`, `enroll_column`), which leaks the labels. The server's primary is label-free
     (most activity over the last 5 s).
   - **A different rule.** The server fires only when the primary is silent AND no other column is active (as the
     protocol specifies). That is STAGE1's "primary silent & nobody else" rule (`duration_rule_*`), which at n=200
     has **66–68 % misses**, about the same as the served head (69 %). STAGE1's own v3 head reached 62.6 %.

   The label-free default policy that ships has never been scored at n=200. "The head loses to this cascade" is
   therefore not established for the implemented rule. The next step is to run the n=200 bench with the server's
   exact rule and primary (label-free, 5 s window).
2. **D2: finals split words** (P4). `_turn_end` cuts at a token index, not at a word start (a SentencePiece piece
   without `▁`). The result is text like "every" | "one" in the downstream LLM context. The fix belongs in `serve.py`.
3. **D3: LiveKit detector mode can commit early.** The LiveKit report says "no early commits"; this run saw one on
   ES2011b. It depends on when the any-speaker VAD goes quiet relative to the timeout decision, so the "0 early"
   figure in research/INTEGRATION.md is not robust.
4. **D4: placeholder zeros.** Until the first diarizer column exists (the first ~0.9–1.0 s), `frame.speakers` is sent as
   `[0,0,0,0]` (with `primary: null`). That looks the same as "nobody is talking", and the protocol has no field for
   "not yet known".
5. **D5: every `speakers` value is stale.** It is 480–960 ms older than the frame's `t` (wall clock p50 ~860 ms at
   arrival). The column's own time (`spk_t`) is sent only with `--debug-fields`, so a client cannot align speakers
   to audio without debug mode.
6. **D6: `stats` edge cases.**
   - An empty session (`end` with no audio) returns `rtf: 65291.9989`.
   - `peak_rss_mb` is the process lifetime maximum (`ru_maxrss`), not the session's. The LibriSpeech session
     reported 3413 MB while its sampled RSS was 3010 MB.
   - `first_partial_ms` measures only the server's own lag from audio arrival to the first partial (~30 ms), not the
     latency from speech.
7. **D7: config problems are silent to the client.** A bad `turn_policy` is ignored and `timeout_ms: -5` is clamped
   to 80 ms. Both are only logged on the server; the client gets no message and is not told the effective values.
   Empty `final` messages (`text: ""`) are emitted at end of stream after a turn_end final.
8. **D8: misleading stat name.** `diar_lag_ms_mean_measured` is derived from sample counts (always 760 at 1x), not
   from wall-clock time. The real lag including compute and transport is ~100 ms higher.
9. **D9: overstated performance claims.**
   - The as-fast-as-possible RTF (0.33 / 0.27) did not reproduce (0.42–0.53).
   - The README's "ASR frames arrive within about 90 ms" is wrong for the live system (p50 ~130 ms).
   - test_serve has 1 real-model test, not 2.
10. **Carried over from the reports and confirmed:**
    - The minimum dead air with the default policy is about 1.0 s timeout + 0.56–0.96 s diarizer + ~0.15 s compute
      and transport. That is ~1.7 s at best; 1.8–2.6 s was observed.
    - The timeout cuts turns during long filled pauses (ES2011b "um").
    - The head without the pad does not fire at 0.98.

No NaN, no backwards timestamps, no swallowed server exceptions (server logs contain only config / end / dropped-session
lines), and no stuck sessions were found. The server keeps up at 1x on 2 threads at load ~5, but only because of the
fast-conv path.

## Verdict

The protocol, the server and both adapters work end to end on real models at 1x on this CPU. They are ready for a
live integration test with a real LiveKit room or Pipecat transport; neither was tested here. Two things block using
them as the product's turn-taker:

- The default policy's accuracy has not been measured without the oracle (D1).
- Its dead air is 1.8–2.6 s, with early cuts on filled pauses.

D2 (split words) should be fixed before any LLM sits downstream.
