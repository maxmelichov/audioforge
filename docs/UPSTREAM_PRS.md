# Upstream PRs: Pipecat service + LiveKit Agents plugin for the audioforge server

Prepared 2026-09-28. Both integrations are committed and pushed to **the user's forks only**; **no pull request
has been opened** on `pipecat-ai/pipecat` or `livekit/agents`, and nothing was pushed anywhere else. This file
holds, per project: the branch and fork URL, the complete PR title and body ready to paste, the files on the
branch, the upstream checklist and its status, what the maintainers will likely ask, and the end-to-end check
that was run once against a local server with a 10 s clip (§7).

| piece | where | branch / commit | status |
|---|---|---|---|
| `audioforge-client` 0.1.0 (WebSocket client + protocol framing, no torch; Apache-2.0) | this repo, `packages/audioforge-client/` | `main` (fix committed with this file, §6) | wheel + sdist rebuilt in `packages/audioforge-client/dist/` (wheel sha256 `3bda6095…`); 12 tests pass; **not on PyPI** |
| Pipecat `AudioforgeSTTService` + `AudioforgeTurnAnalyzer` + `AudioforgeVADAnalyzer` | https://github.com/maxmelichov/pipecat | `feat/audioforge-stt-turn` @ `940d163d7` (on `pipecat-ai/pipecat` `main` = fork `main` `3763bd7d9`, 10 commits after `v1.12.0`) | pushed; 16 tests pass; ruff + pyright clean; E2E run (§7.1) |
| LiveKit `livekit-plugins-audioforge` (STT + VAD + TurnDetector) | https://github.com/maxmelichov/agents | `feat/audioforge-stt-turn` @ `494585479` (on `livekit/agents` `main` = fork `main` `57b3227`, agents 1.8.3) | pushed; 14 tests pass; ruff + `mypy --strict` clean; E2E run (§7.2) |

Order of operations when the user decides to open the PRs:

1. Publish `audioforge-client` 0.1.0 to PyPI (`cd packages/audioforge-client && uv build && uv publish`, needs a
   token). Until it is on PyPI neither upstream CI can resolve the new dependency and `uv lock` fails in both forks;
   that is why `uv.lock` is untouched on both branches.
2. In each clone run `uv lock` (Pipecat: `uv lock && uv sync --group dev --all-extras --no-extra gstreamer
   --no-extra local`; LiveKit: `uv lock && make install`), commit the lock on the branch, push.
3. Open the two PRs (§4), then rename the Pipecat changelog fragment to the PR number (§2.4) and push.
4. Answer the maintainer checklist (§5).

The clones live on the external SSD: `/Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/{pipecat,agents}`
(remotes: `origin` = the fork, `upstream` = the upstream repo). Their virtualenvs are in an APFS sparse bundle on the
same SSD (`/Volumes/afdev/venvs/{pipecat,livekit}`, mounted with `hdiutil attach -nobrowse
/Volumes/ExternalSSD/nvidia-audio-models/scratch/afdev.sparsebundle`; exFAT cannot hold a venv).

---

## 1. The measured numbers used in both PR bodies (sources and caveats)

All numbers come from this repository; each table names its file. CPU only, one Mac (2 threads per model process),
shared with other jobs; n is small. The ASR is English-only. "C" is the configuration both plugins ship as their
defaults: `audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm` (diarizer preset
`low_latency_032`), client `turn_policy="timeout"`, `timeout_ms=1000`, no enrollment, streaming finals. The E2E
drivers used the reference adapters `audioforge/integrations/pipecat.py` and `audioforge/integrations/livekit.py`; the
upstream code is a port of them onto `audioforge-client` with the same event mapping, so the numbers describe the
same behaviour but were **not re-measured with the upstream branches** (only the 10 s check of §7 was run).

### 1.1 End-to-end through the adapters (`research/E2E_FINAL.md` §4-5, `runs/e2e_final.json`)

Response moment = Pipecat's `LLMContextFrame` reaching the LLM / LiveKit's committed user turn. Dead air = first
response after a labelled turn end (median / P90 over ends answered within 6 s); missed@h = no response within h
seconds; a cut-in = a response inside a user turn. AMI: 5 windows / 5 ends, mono. TurnBench dev: 16 clips / 56 ends
(non-commercial licence, evaluation only). otoSpeech dev: 16 clips / 53 ends.

| clips | stack | dead air med / P90 (ms) | missed @3 s / @6 s | cut-ins (per min) | WER | first text (ms) |
|---|---|---|---|---|---|---|
| AMI mono | Pipecat A: Silero VAD + LocalSmartTurnAnalyzerV3 + faster-whisper small | 2548 / 4605 | 60 % / 20 % | 11 (7.46) | 44 % | 3689 |
| AMI mono | **Pipecat C (defaults)** | **1683 / 3042** | 20 % / 0 % | **2 (1.36)** | 39 % | 828 |
| AMI mono | Pipecat D: `hybrid_dyn` + `after_agent_arm` | 2561 / 2990 | 20 % / 0 % | 0 | 39 % | 827 |
| AMI mono | LiveKit B: Silero + turn-detector EnglishModel + faster-whisper small | 2660 / 4280 | 40 % / 0 % | 5 (3.39) | 43 % | 9190 |
| AMI mono | **LiveKit C (defaults, `turn_detection="stt"`)** | **1660 / 2224** | 0 % / 0 % | 5 (3.39) | 39 % | 810 |
| AMI mono | LiveKit D | 2420 / 2956 | 20 % / 0 % | 0 | 39 % | 810 |
| TurnBench mono | Pipecat A | 1054 / 4576 | 75 % / 62 % | 36 (2.96) | 26 % | 2598 |
| TurnBench mono | Pipecat C | 1346 / 3355 | 68 % / 62 % | 14 (1.15) | 27 % | 1005 |
| TurnBench mono | LiveKit B | 1714 / 2993 | 70 % / 66 % | 13 (1.07) | 23 % | 11288 |
| TurnBench mono | LiveKit C | 1312 / 1933 | 34 % / 30 % | 55 (4.52) | 27 % | 999 |
| TurnBench user channel | Pipecat A | 2099 / 3186 | 54 % / 32 % | 26 (2.14) | 16 % | 2644 |
| TurnBench user channel | Pipecat C | 1353 / 1617 | 23 % / 20 % | 16 (1.32) | 17 % | 942 |
| TurnBench user channel | LiveKit B | 1350 / 3080 | 36 % / 25 % | 12 (0.99) | 10 % | 5041 |
| TurnBench user channel | LiveKit C | 1330 / 1500 | 23 % / 23 % | 27 (2.22) | 17 % | 936 |
| otoSpeech mono | Pipecat A → C | 2072 / 4311 → 1372 / 3553 | 62 % / 34 % → 42 % / 32 % | 36 → 16 | – | 2828 → 1077 |
| otoSpeech mono | LiveKit B → C | 1310 / 3286 → 1350 / 3414 | 45 % / 36 % → 32 % / 21 % | 27 → 49 | – | 4610 → 1075 |

Paired bootstrap over clips (bold = CI excludes 0): AMI Pipecat C−A cut-ins/clip **−1.8 [−3.2, −0.4]**, dead-air
median −865 [−3522, +807] (not significant); TurnBench mono LiveKit C−B missed@6s **−36 pts [−53, −20]** but
cut-ins/clip **+2.63 [+1.81, +3.44]**; TurnBench user channel Pipecat C−A missed@3s **−30 pts [−51, −9]**; AMI
LiveKit D−C dead air **+760 ms [+700, +980]**, cut-ins/clip **−1.0 [−1.6, −0.4]**. Server RTF median 0.70–0.81
(2 threads), peak server RSS 3.56–3.59 GB; driver RSS 91–125 MB (Pipecat) / 159–202 MB (LiveKit) vs 1.6–2.8 GB for the
local baselines. `E2E_FINAL.md` recommends Nemotron-3-Diarization as the diarizer (RTF 0.63–0.65, 1.5 GB RSS, lower
dead air); it is a server flag, nothing changes in the plugins.

### 1.2 Turn policy options (`research/INTEGRATION.md` §8, 5 AMI windows, `runs/integration_deadair_shipped.json`)

| policy | Pipecat dead air med / cut-ins | LiveKit dead air med / cut-ins | decision median |
|---|---|---|---|
| `timeout` 1000 ms (default) | 1702 / 3 | 1676 / 5 | 1520 |
| `hybrid_dyn` (head OR any-speaker Silero silence, 2.0–6.4 s by head probability) | 2662 / 0 | 2425 / 0 | 2240 |
| `hybrid_silero` (head OR 2.64 s Silero silence) | 2941 / 0 | 2921 / 0 | 2736 |

ICSI held-out (n = 1312 turns, `research/BASELINES.md`): `hybrid_dyn` misses 79.5 % (open-floor 50.5 %) at 1.8 %
false cutoffs per turn vs `hybrid_silero` 81.7 / 62.2 at 2.4 %; paired −2.2 [−3.2, −1.3] / −11.7 [−17.0, −7.0].

### 1.3 Enrollment (`research/EOT_BENCH_V2.md` §9, AMI dev n = 974, ≤ 5 % FC cross-fitted)

| primary binding | misses @6 s all / open | Δ vs default |
|---|---|---|
| `dominant` (server default) | 61.9 / 45.9 | – |
| `after_agent_arm` (client sends `agent_end` after its TTS) | 56.0 / 40.3 | −5.9 [−8.6, −3.5] |
| `after_agent` (TitaNet voice following) | 51.6 / 34.0 at 6 s, but **+8.5 [+6.2, +10.9] at 2 s** | not recommended |
| oracle | 28.7 / 17.1 | – |

At the product level (5 windows) `after_agent_arm` changed nothing measurable, and in the E2E runs the `agent_end`
time was a **label-derived stand-in** for the TTS-end event, not a real TTS event.

### 1.4 Offline final ASR (`runs/hybrid_asr.json`; server `--final-asr tdt_v3`, plugin `final_source="offline"`)

| model | AMI dev (200 segs) WER | LibriSpeech first-200 WER | CPU RTF (batch 1, 2 threads) |
|---|---|---|---|
| streaming (served) | 0.244 | 0.023 | – |
| Parakeet-TDT 0.6B v3 | 0.097 [0.082, 0.115] | 0.020 | 0.065 |

Paired AMI −0.147 [−0.170, −0.127]; LibriSpeech not significant. **The serving latency of the offline final
(`final.latency_ms`) has not been measured.**

### 1.5 TurnBench with its official scorer (`research/DYADIC.md` §4, `runs/turnbench_dev.json`; max recall at fp ≤ 0.10)

| system | recall / fp / P50 ms |
|---|---|
| ours: head OR per-channel Silero | 0.835 / 0.080 / 1390 |
| Silero timeout per channel | 0.815 / 0.075 / 1399 |
| smart-turn v3 | 0.754 / 0.100 / 1010 |
| VAP | 0.841 / 0.045 / 463 |
| espnet turn-taking | 0.836 / 0.074 / 895 |

Caveats: TurnBench is licensed for evaluation only; these inputs were per-speaker channels + Silero tracks, not the
streaming diarizer the server uses; no LiveKit turn detector was scored by this scorer.

### 1.6 Speed (`research/INTEGRATION.md` §1)

RTF 0.45–0.52 at 1× on 2 threads with the pretrained heads (0.68–0.81 with the served model and the diarizer);
frame lag p50 129–136 ms, p95 ≤ 220 ms; first partial 54–59 ms after the audio is sent; peak RSS 3.0–3.6 GB. It keeps
up only with the fast-conv CPU path (RTF 1.13–1.21 without it). On a machine that is already loaded the server sheds
load (§7 shows it: RTF 1.3, diarizer skipped, `overloaded` notices).

### 1.7 Honest limitations (stated in every PR body)

- Turn-taking is "timeout-grade": on label-free speaker tracks the learned turn head is not better than the silence
  timeout (`research/INTEGRATION.md` §6); the STAGE1 38.4 % figure used an oracle speaker column and does not describe
  the product. Label-free enrollment loses the user: 61.9 % misses vs 28.7 % with an oracle column.
- English-only streaming ASR; meeting WER ≈ 0.39 on AMI (whisper-small 0.43 in the same runs; 0.10 with `tdt_v3`);
  17 % vs 10 % on the clean TurnBench user channel.
- CPU-only measurements on one machine, n = 5 clips for AMI; no real WebRTC room, no real microphone / echo path,
  no concurrent sessions were tested.
- The LiveKit turn detector implements the private `_StreamingTurnDetector` protocol; `turn_detection="stt"` is
  the recommended mode. Pipecat's `UserTurnController` never ends a turn while its VAD hears speech ("the user" is
  whoever is audible), which the turn analyzer's `wait_for_silence` gate respects.
- Known open defects in `research/INTEGRATION.md` §5 (D2: streaming finals can split words at the cut; use
  `final_source="offline"` with `--final-asr tdt_v3` if the LLM sees the finals).

### 1.8 Models and licences

| file | how to get it | licence |
|---|---|---|
| `runs/stage1_served.afm` (NVIDIA `stt_en_fastconformer_hybrid_large_streaming_multi` encoder + our VAD / turn / speaker heads) | `audioforge-download` (pinned revisions + sha256, rebuilt bit-identical from NVIDIA weights + `assets/served_heads_v0.1.pt`) or `research/NEMO_IMPORT.md` | encoder CC-BY-4.0; heads trained on AMI (CC BY 4.0) |
| `runs/nemo_sortformer_v2.afm` (NVIDIA Streaming Sortformer 4spk v2) | `audioforge-download` / `research/SORTFORMER_IMPORT.md` | CC-BY-4.0 (v2.1 / v1 are not used: other licences) |
| `runs/nemo_nemotron3_diar.afm` (Nemotron-3-Diarization 100M, recommended diarizer) | `audioforge-download` | NVIDIA Open Model License (check before redistribution) |
| `data/silero/silero_vad_v5.onnx` (`hybrid_dyn` / `hybrid_silero`) | snakers4/silero-vad v5.1.2 | MIT |
| `data/nemo/parakeet-tdt-0.6b-v3.nemo` (`--final-asr tdt_v3`) | Hugging Face `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 |
| `data/nemo/speakerverification_en_titanet_large.nemo` (`--enroll after_agent` only) | `audioforge.nemo_import.import_titanet()` | CC-BY-4.0 |
| `data/nemo/langid_ambernet.nemo` (`--lid`) | NGC `nvidia/nemo/langid_ambernet` 1.12.0 | NGC Terms of Use (not redistributed) |
| `examples/audio/two_speakers_10s.wav` (the E2E clip) | this repo | LibriSpeech CC-BY-4.0 |
| `audioforge-client` | PyPI (after step 1) / `packages/audioforge-client` | Apache-2.0 |
| plugin code | the PRs | Pipecat: BSD-2-Clause (repo licence); LiveKit: Apache-2.0 (repo licence, CLA) |

---

## 2. Pipecat PR

**Fork:** https://github.com/maxmelichov/pipecat, branch `feat/audioforge-stt-turn` (commit `940d163d7`, base
`pipecat-ai/pipecat` `main`). Compare view: https://github.com/maxmelichov/pipecat/compare/main...feat/audioforge-stt-turn

### 2.1 Title

`Add audioforge STT service, turn analyzer and VAD analyzer (self-hosted streaming ASR + speaker-aware end-of-turn)`

### 2.2 Body (paste as is)

```markdown
## What

A self-hosted speech front end for Pipecat, wrapping the **audioforge** WebSocket server
(https://github.com/maxmelichov/audioforge, `audioforge/serve.py`): NVIDIA's streaming FastConformer
ASR with learned VAD / turn heads and NVIDIA Streaming Sortformer v2 diarization on one 80 ms clock, CPU-only.
Three pieces share one WebSocket session per pipeline through an `AudioforgeHub`:

- `AudioforgeSTTService` (`WebsocketSTTService`): `partial` -> `InterimTranscriptionFrame`, `final` ->
  `TranscriptionFrame(finalized=True)` with `metadata["audioforge"] = {t, speaker, kind, source}`; `EndFrame`
  sends `end` and waits for the server's last final + stats; runtime `Settings` (`turn_policy`, `timeout_ms`,
  `eot_threshold`) reconnect on change like the other websocket STTs; `supports_ttfs` is False because each final
  arrives together with its turn decision.
- `AudioforgeTurnAnalyzer` (`BaseTurnAnalyzer`, for `TurnAnalyzerUserTurnStopStrategy`): end of turn from the
  server's `turn_end` events on the diarizer's primary-speaker track (other speakers do not delay it). A decision
  latches until Pipecat ends the turn, is reported only while the VAD hears no speech (`wait_for_silence`), and is
  dropped once the server's VAD reports `resume_ms` of speech after the decision.
- `AudioforgeVADAnalyzer` (`VADAnalyzer`): the server's per-frame VAD probability under Pipecat's own hysteresis;
  no local model.

Optional, off by default: `turn_policy="hybrid_dyn"` / `"hybrid_silero"` (server-side head OR any-speaker Silero
silence), `enroll="after_agent_arm"` (sends `agent_end` at `BotStoppedSpeakingFrame` so the server binds the primary
speaker to whoever answers the bot; `enroll="explicit"` + `await stt.enroll()`), `final_source="offline"` (the
server's per-turn offline Parakeet-TDT transcript, `--final-asr tdt_v3`), and the server's `--lid` language
identification (`frame.language` on the transcripts + an `on_language_detected` event).

Dependency: `audioforge-client` (PyPI, Apache-2.0, depends on `websockets` only), new extra `pipecat-ai[audioforge]`.
The three modules only import public Pipecat APIs, so they lift out unchanged into a community package if you
prefer that route (COMMUNITY_INTEGRATIONS.md).

## Why

Every local Pipecat stack today runs a VAD, a turn model and an STT as separate models on the mixed input.
audioforge makes one server decide turn boundaries on a *speaker-aware* track and cut the transcript at exactly
those decisions, so the LLM never sees a final that a later `turn_end` contradicts, and the bot does not answer a
bystander. Measured end to end through the same event mapping (reference adapter in the server repo; AMI meeting
audio, mixed mono, 5 windows; response moment = `LLMContextFrame` reaching the LLM; `runs/e2e_final.json` /
`research/E2E_FINAL.md` there):

| stack (Pipecat 1.12, CPU) | dead air median / P90 | missed @3 s | cut-ins (/min) | WER | first text |
|---|---|---|---|---|---|
| Silero VAD + LocalSmartTurnAnalyzerV3 + faster-whisper small | 2548 / 4605 ms | 60 % | 11 (7.46) | 44 % | 3689 ms |
| **this PR, defaults (`timeout` 1000 ms)** | **1683 / 3042 ms** | 20 % | **2 (1.36)** | 39 % | 828 ms |
| this PR, `hybrid_dyn` + `after_agent_arm` | 2561 / 2990 ms | 20 % | 0 | 39 % | 827 ms |

Paired over clips: cut-ins/clip −1.8 [−3.2, −0.4]; dead-air median −865 ms [−3522, +807] (not significant at n = 5).
TurnBench dev (16 two-party clips, evaluation-only licence), user channel: missed@3s 23 % vs 54 % for the local
stack (paired −30 pts [−51, −9]), dead air 1353 / 1617 vs 2099 / 3186 ms; mono mix: cut-ins/min 1.15 vs 2.96.
Driver RSS 91 MB vs 1.6 GB (the models live in the server: RTF ≈ 0.7–0.8 on 2 CPU threads, 3.6 GB RSS).

## Options and their measured cost (5 AMI windows, server repo `research/INTEGRATION.md` §8)

| option | effect | cost |
|---|---|---|
| `turn_policy="hybrid_dyn"` | 0 cut-ins vs 3 | median dead air +0.96 s (2662 vs 1702 ms) |
| `turn_policy="hybrid_silero"` | 0 cut-ins | +1.2 s |
| `enroll="after_agent_arm"` | −5.9 pts misses at 6 s in the offline benchmark (AMI dev n = 974, [−8.6, −3.5]) | nothing measurable at the product level yet |
| `final_source="offline"` (server `--final-asr tdt_v3`) | AMI WER 0.097 vs 0.244 (paired −0.147) | a second model in the server; its serving latency is not measured yet |

## Limitations (honest)

- Turn-taking is timeout-grade: on label-free speaker tracks the learned turn head does not beat the silence
  timeout, so the default is the plain 1000 ms timeout on the primary speaker. Dead air ≈ 1.7 s median.
- English-only streaming ASR; meeting WER ≈ 0.39 (faster-whisper small 0.43 in the same runs; 0.10 with the
  offline pass); on clean speech it is 7 points behind Whisper small.
- All measurements are CPU-only on one Mac, n = 5 AMI clips / 16 TurnBench clips; no WebRTC transport, echo path or
  concurrent sessions were tested. The server needs ~3.6 GB RSS and keeps up at 1× on 2 threads only with its
  fast-conv path. Self-hosted only, no hosted endpoint; I am the server's author.
- Pipecat's `UserTurnController` never ends a turn while the VAD hears speech, and "the user" is whoever is
  audible; the analyzer's `wait_for_silence` respects that, so on multi-party audio a bystander can hold the turn.
- The turn analyzer route was measured; an `ExternalUserTurnStrategies` / `ProposedUser*SpeakingFrame` mode (the
  STT proposing the turn boundaries itself) is a natural follow-up and not part of this PR.

## How to run

```bash
pip install "pipecat-ai[audioforge]"
# server (once): NVIDIA FastConformer streaming encoder + Streaming Sortformer v2 (both CC-BY-4.0, ~1 GB)
pip install "audioforge[serve]" && audioforge-download && audioforge-serve --port 8765
#   (or from source: python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm)
python examples/transcription/transcription-audioforge.py -t webrtc
python examples/turn-management/turn-management-audioforge.py -t webrtc      # AUDIOFORGE_TURN_POLICY=hybrid_dyn
```

## Tests

`tests/test_audioforge_stt.py` (6) and `tests/test_audioforge_turn.py` (8): hermetic, the socket tests run against
`audioforge_client.testing.FakeAudioforgeServer` (a scripted server speaking the same protocol: frames, partials,
turn_end / final at a configurable timeout, `--final-asr` twins, `--enroll` and `--lid` events, dropped sockets).
`tests/test_service_init.py` covers the new `Settings` class automatically (2 more). `ruff format` / `ruff check`
(including F401) and `pyright` are clean on the new files.

## Checklist

- [x] changelog fragment (`changelog/0000.added.md`, to be renamed to this PR's number)
- [x] `ruff format` / `ruff check` clean, Google docstrings, pyright clean
- [x] examples under `examples/transcription/` and `examples/turn-management/`
- [x] unit tests in `tests/`
- [ ] `uv.lock`: regenerated once `audioforge-client` is on PyPI (step 1 of this PR); until then CI cannot resolve
  the extra. Happy to move the integration to a community package `pipecat-audioforge` instead, per
  COMMUNITY_INTEGRATIONS.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01JakvDTG2e6oYJAWNJ9nG3j
```

### 2.3 Files on the branch (`git diff --stat upstream/main..HEAD`, 10 files, +1558)

| file | lines | what |
|---|---|---|
| `src/pipecat/services/audioforge/stt.py` | 645 | `AudioforgeSTTSettings`, `AudioforgeHub`, `ReceivedTurnEnd`, `AudioforgeSTTService` |
| `src/pipecat/services/audioforge/__init__.py` | 21 | exports |
| `src/pipecat/audio/turn/audioforge_turn.py` | 197 | `AudioforgeTurnParams`, `AudioforgeTurnAnalyzer` |
| `src/pipecat/audio/vad/audioforge_vad.py` | 84 | `AudioforgeVADAnalyzer` |
| `tests/test_audioforge_stt.py` | 188 | 6 tests (defaults/validation, hub policy, fake-server round trip with `after_agent_arm` + `--lid`, `final_source="offline"`, `frames` batches + structured errors via a fake websocket, event units) |
| `tests/test_audioforge_turn.py` | 144 | 8 tests (latch / silence gate / clear, no-gate, stale after resumed speech incl. same-batch frames, policy filter + `both`, offline-final gating, session reset, VAD analyzer) |
| `examples/transcription/transcription-audioforge.py` | 109 | transcription + `VADProcessor` on the server VAD + language events |
| `examples/turn-management/turn-management-audioforge.py` | 168 | full bot: VAD + turn analyzer from the STT's hub, `enroll="after_agent_arm"`, `AUDIOFORGE_TURN_POLICY` |
| `pyproject.toml` | +1 | extra `audioforge = [ "audioforge-client>=0.1.0,<0.2" ]` |
| `changelog/0000.added.md` | 1 | fragment (§2.4) |

Public surface: `AudioforgeSTTService(url, hub, sample_rate, enroll, final_source, end_timeout, settings)`,
`.Settings(model, language, turn_policy, timeout_ms, eot_threshold)`, `.create_vad_analyzer()`,
`.create_turn_analyzer()`, `await .enroll()`, `await .agent_end()`, `.handle_server_event()`, event
`on_language_detected(service, language, confidence)`; `AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(
policy, wait_for_silence, resume_ms))`; `AudioforgeVADAnalyzer(hub, params=VADParams(...))` (`min_volume` 0 unless set).

### 2.4 Changelog fragment (`changelog/0000.added.md`; rename to `<PR>.added.md` after opening the PR)

> - Added `AudioforgeSTTService`, `AudioforgeTurnAnalyzer` and `AudioforgeVADAnalyzer` for the self-hosted
>   [audioforge](https://github.com/maxmelichov/audioforge) speech server: NVIDIA's streaming FastConformer
>   ASR with learned VAD and end-of-turn heads plus Streaming Sortformer diarization on one 80 ms clock, CPU-only. One
>   server session per pipeline feeds streaming transcription cut at speaker-aware turn decisions, a VAD analyzer with
>   no local model, and a turn analyzer that ends the user turn where the server cut the transcript. Optional
>   server-side turn policies (`timeout`, `hybrid_dyn`, `hybrid_silero`), primary-speaker enrollment after the bot's
>   speech (`enroll="after_agent_arm"`), an offline per-turn final transcript (`final_source="offline"`) and language
>   identification events. Install with `pip install "pipecat-ai[audioforge]"`.

### 2.5 Pipecat checklist (CONTRIBUTING.md / AGENTS.md) and status

| item | status |
|---|---|
| fork, branch, meaningful commit message | done (`940d163d7`) |
| changelog fragment `<PR>.<type>.md`, bullet, one logical change | done as `0000.added.md`; rename after the PR number exists |
| `uv run ruff check` / `uv run ruff format --check` | clean on the new files (repo-wide run not done: it needs the full dev sync) |
| `uv run pyright` (basic mode, `src/pipecat`) | clean on the three new modules |
| Google docstrings, `Args:` on `__init__`, `Parameters:` on dataclasses | done |
| `Settings` pattern (store vs delta, `NOT_GIVEN` defaults, `_update_settings` → reconnect) | done; `test_service_init.py` passes for the new class |
| `self.create_task` / `cancel_task`, no slow work in `process_frame` | done (receive task via `_receive_task_handler`) |
| tests under `tests/` | 14 new tests + 2 auto-discovered |
| examples | 2 |
| `uv lock` + `uv sync` after editing `pyproject.toml`, commit both | **blocked** on the PyPI upload of `audioforge-client` |
| pre-commit hooks | not run (no dev sync); ruff format applied by hand |

---

## 3. LiveKit Agents PR

**Fork:** https://github.com/maxmelichov/agents, branch `feat/audioforge-stt-turn` (commit `494585479`, base
`livekit/agents` `main`, agents version 1.8.3). Compare view:
https://github.com/maxmelichov/agents/compare/main...feat/audioforge-stt-turn

### 3.1 Title

`feat(audioforge): new plugin - self-hosted streaming STT, VAD and speaker-aware turn detection`

### 3.2 Body (paste as is)

```markdown
## What

`livekit-plugins-audioforge`: STT, VAD and a streaming turn detector for the **audioforge** WebSocket server
(https://github.com/maxmelichov/audioforge, `audioforge/serve.py`): NVIDIA's streaming FastConformer ASR
with learned VAD / turn heads plus NVIDIA Streaming Sortformer v2 diarization on one 80 ms clock, CPU-only,
self-hosted. One `AudioforgeFrontend` opens **one server session per AgentSession** and hands it to:

- `STT` (streaming, interim results, diarization): `partial` -> INTERIM_TRANSCRIPT, `final` -> FINAL_TRANSCRIPT with
  `speaker_id="S<k>"`, `is_primary_speaker` and `metadata["audioforge"]`, the server's `turn_end` -> END_OF_SPEECH
  right after its final, so `turn_detection="stt"` commits the user turn exactly where the server cut the
  transcript; `stats` -> RECOGNITION_USAGE; `recognize()` for batch. With the server's `--lid`, language events
  update the `language` of later transcripts.
- `VAD`: VADEvents from the server's per-frame VAD probability (any-speaker), for barge-in
  (`interruption: {"mode": "vad"}`); no local model.
- `TurnDetector` (implements `voice.turn._StreamingTurnDetector`): resolves p = 1.0 when the server's `turn_end`
  arrives for the current pause and 0.0 on resumed speech or after `prediction_timeout`, for users who want
  LiveKit's endpointing min/max delay logic; policy `"head"` resolves the turn head's probability instead.
- `frontend.attach(session)`: sends `agent_end` on `agent_state_changed` speaking -> other, for the server's
  `--enroll after_agent_arm` (bind the primary speaker to whoever answers the agent). `frontend.enroll()` for
  `--enroll explicit`.

Options on the frontend: `turn_policy` (`timeout` default, `hybrid_dyn`, `hybrid_silero`, …), `timeout_ms`,
`eot_threshold`, `final_source` (`"offline"` = the server's per-turn offline Parakeet-TDT transcript with
`--final-asr tdt_v3`), `language`. Dependencies: `livekit-agents>=1.8.3`, `audioforge-client` (PyPI, Apache-2.0,
`websockets` only), `numpy`.

## Why

Measured end to end in a room-less `AgentSession` with the same event mapping (reference adapter in the server
repo; response moment = the committed user turn; AMI meeting audio, mixed mono, 5 windows; `runs/e2e_final.json` /
`research/E2E_FINAL.md` there):

| stack (LiveKit Agents 1.8, CPU) | dead air median / P90 | missed @3 s | cut-ins (/min) | WER | first text |
|---|---|---|---|---|---|
| Silero VAD + turn-detector EnglishModel + faster-whisper small (StreamAdapter) | 2660 / 4280 ms | 40 % | 5 (3.39) | 43 % | 9190 ms |
| **this plugin, defaults (`timeout` 1000 ms, `turn_detection="stt"`)** | **1660 / 2224 ms** | **0 %** | 5 (3.39) | 39 % | 810 ms |
| this plugin, `hybrid_dyn` + `after_agent_arm` | 2420 / 2956 ms | 20 % | 0 | 39 % | 810 ms |

TurnBench dev (16 two-party clips, evaluation-only licence), mixed mono: missed@3s 34 % vs 70 % for the local stack
(paired missed@6s −36 pts [−53, −20]) at the price of more cut-ins (55 vs 13; paired +2.63/clip [+1.81, +3.44]);
user channel only: 1330 / 1500 ms and missed@3s 23 % vs 1350 / 3080 ms and 36 %. Driver RSS 159 MB vs 2.2–2.8 GB (the
models live in the server: RTF ≈ 0.7–0.8 on 2 CPU threads, 3.6 GB RSS). The local stack's first partial arrives
3.4–11.3 s after speech onset (StreamAdapter + Whisper); ours 0.8–1.2 s.

## Options and their measured cost (5 AMI windows, server repo `research/INTEGRATION.md` §8)

| option | effect | cost |
|---|---|---|
| `turn_policy="hybrid_dyn"` | 0 cut-ins vs 5 | median dead air +0.75 s (2425 vs 1676 ms) |
| `turn_policy="hybrid_silero"` | 0 cut-ins | +1.2 s |
| `after_agent_arm` + `frontend.attach(session)` | −5.9 pts misses at 6 s in the offline benchmark (AMI dev n = 974, [−8.6, −3.5]) | nothing measurable at the product level yet |
| `final_source="offline"` (server `--final-asr tdt_v3`) | AMI WER 0.097 vs 0.244 (paired −0.147) | a second model in the server; its serving latency is not measured yet |

## Limitations (honest)

- Turn-taking is timeout-grade: on label-free speaker tracks the learned turn head does not beat the silence
  timeout, so the default is the plain 1000 ms timeout on the diarizer's primary speaker.
- English-only streaming ASR; meeting WER ≈ 0.39 (0.10 with the offline pass); 17 % vs Whisper small's 10 % on the
  clean TurnBench user channel.
- `TurnDetector` implements a **private** protocol (`_StreamingTurnDetector`) and may need updating between
  releases; `turn_detection="stt"` is the recommended and measured mode. I can drop the turn detector from this PR
  if you would rather not carry that.
- CPU-only measurements on one Mac, n = 5 AMI clips / 16 TurnBench clips, room-less sessions; no real WebRTC room,
  Opus path, echo or concurrent sessions were tested. Self-hosted only, no hosted endpoint; I am the server's author.
- With `turn_detection="stt"` LiveKit logs `stt end of speech received while vad is still in a speech segment`
  whenever our turn end fires inside a pause the VAD has not closed yet (expected; it does not change the committed
  turn).

## How to run

```bash
pip install livekit-plugins-audioforge
# server (once): NVIDIA FastConformer streaming encoder + Streaming Sortformer v2 (both CC-BY-4.0, ~1 GB)
pip install "audioforge[serve]" && audioforge-download && audioforge-serve --port 8765
#   (or from source: python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm)
```

```python
from livekit.agents import AgentSession
from livekit.plugins import audioforge

fe = audioforge.AudioforgeFrontend("ws://127.0.0.1:8765")
session = AgentSession(
    stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
    turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.0},
                   "interruption": {"mode": "vad"}},
)
fe.attach(session)  # agent_end after every agent utterance (server --enroll after_agent_arm)
```

## Tests

`tests/test_plugin_audioforge.py` (14, `uv run pytest --plugin audioforge`), hermetic against
`audioforge_client.testing.FakeAudioforgeServer`: shared session for STT + VAD + turn detector with `agent_end`
ordering, lost-session error, `final_source="offline"` + `--lid`, batch `recognize`, the two mappers, the turn
detector's resolve / deadline / stale / head paths, PCM conversion. `ruff` and `mypy --strict`
(`livekit.plugins.audioforge`) are clean.

## Notes for the reviewers

- Version is `1.8.3` like the other plugins (release PRs bump it); tell me if a new plugin should start elsewhere.
- Registered in `[tool.uv.sources]` and as the `livekit-agents[audioforge]` extra; `uv.lock` is untouched because
  `audioforge-client` is being published to PyPI as the first step of this PR, then I will run `uv lock`.
- Docs: I can provide the integration page text for docs.livekit.io (README has the material).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01JakvDTG2e6oYJAWNJ9nG3j
```

### 3.3 Files on the branch (`git diff --stat upstream/main..HEAD`, 13 files, +1871)

| file | lines | what |
|---|---|---|
| `livekit-plugins/livekit-plugins-audioforge/livekit/plugins/audioforge/frontend.py` | 515 | `AudioforgeOptions`, `_PcmConverter` (any rate/channels → 16 kHz mono via `rtc.AudioResampler`), `_Link` (one socket: single feeder, writer queue so `agent_end` lands at the exact audio position, reader fan-out, `wall_of(t)`), `AudioforgeFrontend` (factories, `agent_end()`, `enroll()`, `attach(session)`, session sharing) |
| `.../stt.py` | 326 | `_SpeechMapper` (pure state machine), `STT`, `SpeechStream` |
| `.../vad.py` | 248 | `_VadMapper`, `VAD`, `VADStream` (reconnects up to 3× on a lost session) |
| `.../turn_detector.py` | 261 | `TurnDetector`, `TurnDetectorStream` (`_StreamingTurnDetectorStream`) |
| `.../__init__.py`, `log.py`, `version.py` (`1.8.3`), `py.typed` | 58 / 3 / 15 / 0 | template files (`Plugin.register_plugin`, `__pdoc__`) |
| `livekit-plugins/livekit-plugins-audioforge/pyproject.toml`, `README.md` | 45 / 52 | package metadata (hatchling, dynamic version), usage |
| `tests/test_plugin_audioforge.py` | 346 | 14 tests, `pytestmark = pytest.mark.plugin("audioforge")` |
| `pyproject.toml` | +1 | `livekit-plugins-audioforge = { workspace = true }` |
| `livekit-agents/pyproject.toml` | +1 | extra `audioforge = ["livekit-plugins-audioforge>=1.8.3"]` |

Not touched (CI matrix): `.github/workflows/tests.yml` runs a fixed list of cloud plugins with keys; the new tests
need no credentials and run under `--plugin audioforge`. Ask the maintainers whether they want it in the unit gate.

### 3.4 LiveKit checklist (CONTRIBUTING.md / AGENTS.md) and status

| item | status |
|---|---|
| plugin layout copied from `livekit-plugins-minimal` (`livekit/plugins/<name>/{__init__,log,version}.py`, hatchling, `[tool.uv]` exclude-newer) | done |
| `ruff check --fix` + `ruff format` | clean |
| `make type-check` (`scripts/check_types.py`, mypy strict, auto-discovers `livekit.plugins.audioforge`) | `mypy -p livekit.plugins.audioforge` clean; the repo-wide script not run (needs `make install`) |
| pdoc documentation of every public class / method | docstrings on all public surface; `__pdoc__` hides internals |
| test module with a category marker | `pytestmark = pytest.mark.plugin("audioforge")` |
| no `version.py` bumps, no `CHANGELOG.md` edits (bot) | respected: new plugin at the current agents version `1.8.3` |
| CLA Assistant on the first PR | to sign when the PR is opened |
| `uv lock` after adding the workspace member | **blocked** on the PyPI upload of `audioforge-client` |

---

## 4. Commands (run only when the user decides to open the PRs)

```bash
# 0. publish the client (needs a PyPI token); wheel + sdist are already built
cd /Users/maxm/nvidia-audio-models/packages/audioforge-client && uv build && uv publish

# 1. lock files, once audioforge-client resolves from PyPI
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat && uv lock && git add uv.lock && git commit -m "Lock audioforge-client" && git push
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/agents  && uv lock && git add uv.lock && git commit -m "chore: lock audioforge-client" && git push

# 2. PR bodies: extract the fenced markdown blocks of §2.2 and §3.2 from this file
python3 - <<'EOF'
import re, pathlib
doc = pathlib.Path("/Users/maxm/nvidia-audio-models/docs/UPSTREAM_PRS.md").read_text()
bodies = re.findall(r"### \d\.2 Body \(paste as is\)\n\n```markdown\n(.*?)\n```\n", doc, re.S)
pathlib.Path("/tmp/pr_pipecat.md").write_text(bodies[0]); pathlib.Path("/tmp/pr_livekit.md").write_text(bodies[1])
EOF

# 3. open the PRs (upstream repos; the user's decision)
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat && \
  gh pr create --repo pipecat-ai/pipecat --base main --head maxmelichov:feat/audioforge-stt-turn \
    --title "Add audioforge STT service, turn analyzer and VAD analyzer (self-hosted streaming ASR + speaker-aware end-of-turn)" \
    --body-file /tmp/pr_pipecat.md
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/agents && \
  gh pr create --repo livekit/agents --base main --head maxmelichov:feat/audioforge-stt-turn \
    --title "feat(audioforge): new plugin - self-hosted streaming STT, VAD and speaker-aware turn detection" \
    --body-file /tmp/pr_livekit.md

# 4. Pipecat: rename the changelog fragment to the PR number
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat && git mv changelog/0000.added.md changelog/<PR>.added.md && \
  git commit -m "Rename changelog fragment to PR number" && git push
```

Running the tests locally (venvs on the SSD sparse bundle, see the top of this file):

```bash
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat && /Volumes/afdev/venvs/pipecat/bin/python -m pytest -q tests/test_audioforge_stt.py tests/test_audioforge_turn.py tests/test_service_init.py -k "audioforge or Audioforge"
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/agents  && /Volumes/afdev/venvs/livekit/bin/python -m pytest -q tests/test_plugin_audioforge.py --plugin audioforge && /Volumes/afdev/venvs/livekit/bin/python -m mypy -p livekit.plugins.audioforge
```

## 5. What the maintainers will likely ask

**Both**
- [ ] `audioforge-client` on PyPI **before** CI can pass, then regenerated `uv.lock` files (§4 step 1).
- [ ] "Is there a hosted endpoint?" No: self-hosted only. Hardware: server RTF ≈ 0.7–0.8 on 2 CPU threads, 3.6 GB
  RSS (1.5 GB with Nemotron-3); GPU untested.
- [ ] "Who maintains it?" The user, as the server author; no vendor. Pipecat's community guide asks for company
  attribution when relevant: state that there is none.
- [ ] Licences of the models (CC-BY-4.0 for the FastConformer encoder, Sortformer v2 and Parakeet-TDT; NVIDIA Open
  Model License for Nemotron-3; Silero MIT; AmberNet NGC terms, not redistributed) and of the eval sets (TurnBench
  non-commercial: keep it out of any marketing claim; AMI CC BY 4.0).
- [ ] "Why did the measured numbers use a different code path?" They were measured with the reference adapters in
  the server repo (`integrations/`), of which the upstream code is a port with the same mapping; only the 10 s
  check of §7 was run with the upstream branches. Offer to re-run `scripts/e2e_final.py` against the branches.
- [ ] Demo video (Pipecat's community guide asks for 30–60 s showing an interruption): a screen recording of
  `examples/turn-management/turn-management-audioforge.py -t webrtc` is the cheapest way; not recorded yet.
- [ ] Reconnect semantics: a new server session restarts the diarizer, so speaker ids (`S<k>`) are not stable across
  reconnects; both plugins handle it (Pipecat resets the hub and the analyzer's latch; LiveKit's VAD stream
  reconnects up to 3×, the STT stream relies on LiveKit's own retry).

**Pipecat**
- [ ] They may redirect this to a **community integration** (COMMUNITY_INTEGRATIONS.md: separate repo
  `pipecat-audioforge`, a row on the Supported Services page with a `Community` badge, a docs PR against
  `pipecat-ai/docs`). The code lifts out unchanged: `pipecat/services/audioforge/`,
  `pipecat/audio/turn/audioforge_turn.py`, `pipecat/audio/vad/audioforge_vad.py` import only public Pipecat APIs.
- [ ] Why a turn analyzer rather than `ExternalUserTurnStrategies` + `ProposedUserStarted/StoppedSpeakingFrame`
  (the pattern Soniox / Deepgram Flux use for provider-side endpointing)? Because the analyzer route is what was
  measured and it keeps Pipecat's VAD gate; the proposed-frames mode is a small follow-up (the hub already has
  everything: start on the first partial / VAD onset, stop at the cutting final).
- [ ] The extra name `audioforge`, the `Settings` reconnect semantics, `can_generate_metrics()` returning False
  (no TTFB in Pipecat's sense), `supports_ttfs` False.
- [ ] pre-commit (`uv run pre-commit install`) and a repo-wide `ruff` / `pyright` run after the dev sync.

**LiveKit**
- [ ] Private protocol use (`_StreamingTurnDetector`): they may ask to keep only STT + VAD, or to implement a
  `_StreamingTurnDetectionTransport` for `inference.TurnDetector` instead (the newer extension point in
  `livekit/agents/inference/eot/`); both are fine, the transport variant would replace `turn_detector.py`.
- [ ] `is_primary_speaker` semantics: set from the diarizer's current primary column at the final; LiveKit's
  multi-speaker adapter may want it only when diarization is on (it is always on here).
- [ ] Whether the `--plugin audioforge` tests should join the CI matrix (no credentials needed) and the Renovate
  config for the new package.
- [ ] Docs: LiveKit's docs live in a separate repo (docs.livekit.io); offer the README text.

## 6. Repo-side changes in this commit (this repo, not pushed)

- `packages/audioforge-client/src/audioforge_client/protocol.py`: `ServerError` now carries the server's structured
  `code`, `detail` and `fatal` fields (the hardened reference adapters read them; the client used to flatten the
  message and drop `fatal`). Client-side parse failures use `code="client_parse"`. `tests/test_protocol.py` +1 test
  (12 pass). Wheel + sdist rebuilt in `dist/`. Both upstream plugins depend on this (they log non-fatal errors as
  warnings and treat `fatal` as a session failure).
- `docs/UPSTREAM_PRS.md` (this file).
- `audioforge/integrations/pipecat.py` and `audioforge/integrations/livekit.py` were being hardened by another agent
  during this work (structured `error` handling); their latest state was used as the source of the ports and they are
  **not** part of this commit.

## 7. End-to-end check of the upstream branches (one run each, local server, 10 s clip)

Server: `scripts/dev/gate.sh .venv/bin/python -m audioforge.serve --asr runs/stage1_served.afm --diar
runs/nemo_sortformer_v2.afm --port 8799 --lid data/nemo/langid_ambernet.nemo` (default `--enroll dominant`; loaded in
7.1 s, warmup 1.8 s, `ready` = `{'model': 'stage1_served+nemo_sortformer_v2', 'chunk_ms': 160, 'frame_ms': 80,
'diar_config': 'low_latency_032', 'column_lag_ms': 160.0}`). Client `turn_policy="timeout"`, `timeout_ms=1000`. Clip:
`examples/audio/two_speakers_10s.wav` (16 kHz mono, 10.2 s; speaker A 0.20–4.48 s "a cold lucid indifference reigned
in his soul", speaker B 5.88–9.90 s "heaven a good place to be raised to"; LibriSpeech, CC-BY-4.0), pushed at 1× in
20 ms frames from the plugin venvs (Pipecat `1.12.1.dev10` = fork main, livekit-agents 1.8.3 = fork main), followed by
3 s of silence. Drivers: `e2e_pipecat.py` and `e2e_livekit.py` in the session scratchpad (they build the pipeline /
streams the way the examples and README do). Times are wall-clock ms since the first pushed audio frame.

**Caveat on the timings.** The machine was saturated by other agents' jobs during the run (swap 13–16 GB of 17 GB,
three busy Python processes): the server ran at **RTF 1.27–1.39 instead of 0.7–0.8**, announced `overloaded` (load
shedding level 1: backlog 1.6–3.4 s, diarizer skipped, "VAD-only speaker activity"), and its stats show `peak_rss_mb`
1581 (the diarizer's state never grew). Consequences visible below: the finals arrive 1.5–3 s later than the audio
clock would allow, and the speaker columns are `S0` for both speakers in the Pipecat run (LiveKit's run saw the
diarizer switch to `S1` before shedding). Both plugins behaved correctly under this: no errors, no reconnects, both
turns detected once each, transcripts exact. Under the load of §1 the same events arrive ~1 s after the 1000 ms
timeout elapses. The two `overloaded` notices are the non-fatal structured `error` messages of §6, surfaced as
warnings by both plugins.

### 7.1 Pipecat (`AudioforgeSTTService` + `AudioforgeVADAnalyzer` + `AudioforgeTurnAnalyzer` in `LLMUserAggregator`)

Pipeline `WavSource -> AudioforgeSTTService -> Tap -> LLMUserAggregator(vad_analyzer, TurnAnalyzerUserTurnStopStrategy)
-> Sink`; `interim` / `FINAL` are the frames the STT pushed, the `User*` frames and `LLMContextFrame` are the
aggregator's decisions (the LLM would run at each `LLMContextFrame`).

```
WARNING  AudioforgeSTTService#0: audioforge server error overloaded: backlog 1760 ms at rtf 1.39: level 1, diarizer skipped (VAD-only speaker activity)
   2139 ms  VADUserStartedSpeakingFrame
   2139 ms  UserStartedSpeakingFrame
   2540 ms  interim      'a co' {'t': 1.6, 'speaker': 0, 'kind': 'partial'}
   3057 ms  language     en (0.98)
   3057 ms  interim      'a cold lu' {'t': 2.24, 'speaker': 0, 'kind': 'partial'}
   3544 ms  interim      'a cold lucid' {'t': 2.72, 'speaker': 0, 'kind': 'partial'}
   4301 ms  interim      'a cold lucid indifference' {'t': 3.2, 'speaker': 0, 'kind': 'partial'}
   5086 ms  interim      'a cold lucid indifference reign' {'t': 3.68, 'speaker': 0, 'kind': 'partial'}
   5709 ms  interim      'a cold lucid indifference reigned in his soul' {'t': 4.16, 'speaker': 0, 'kind': 'partial'}
   7226 ms  VADUserStoppedSpeakingFrame
   7587 ms  FINAL        'a cold lucid indifference reigned in his soul' lang=en {'t': 5.36, 'speaker': 0, 'kind': 'final', 'source': None}
   7591 ms  UserStoppedSpeakingFrame
   7591 ms  LLMContextFrame (LLM would run now) last user message='a cold lucid indifference reigned in his soul'
   9125 ms  VADUserStartedSpeakingFrame
   9125 ms  UserStartedSpeakingFrame
   9370 ms  interim      'hea' {'t': 7.2, 'speaker': 0, 'kind': 'partial'}
   9466 ms  VADUserStoppedSpeakingFrame
  10578 ms  interim      'heaven' {'t': 7.68, 'speaker': 0, 'kind': 'partial'}
  10684 ms  VADUserStartedSpeakingFrame
  11416 ms  interim      'heaven a good place' {'t': 8.64, 'speaker': 0, 'kind': 'partial'}
  11763 ms  interim      'heaven a good place to be' {'t': 9.12, 'speaker': 0, 'kind': 'partial'}
  12105 ms  interim      'heaven a good place to be raised to' {'t': 9.6, 'speaker': 0, 'kind': 'partial'}
  12194 ms  VADUserStoppedSpeakingFrame
  13293 ms  FINAL        'heaven a good place to be raised to' lang=en {'t': 10.64, 'speaker': 0, 'kind': 'final', 'source': None}
  13294 ms  UserStoppedSpeakingFrame
  13294 ms  LLMContextFrame (LLM would run now) last user message='heaven a good place to be raised to'

hub: frames=166 partials=11 turn_ends=[(5.36, 'timeout', 1040), (10.64, 'timeout', 1040)]
     finals=[(5.36, 0, 'a cold lucid indifference reigned in his soul'), (10.64, 0, 'heaven a good place to be raised to'), (13.195, 0, '')]
     errors=['overloaded: backlog 1760 ms at rtf 1.39: level 1, diarizer skipped (VAD-only speaker activity)']
turn analyzer: received=2 discarded=0
stats: rtf=1.287 first_partial_ms=662.0 peak_rss_mb=1581.2 lang=en
```

Reading: both `turn_end`s are the timeout policy at 1040 ms of primary silence (decision at audio time 5.36 s for a
turn that ended at 4.48 s; 10.64 s for one that ended at 9.90 s); each `FINAL` carries the turn's exact text and is
followed within 4 ms by `UserStoppedSpeakingFrame` and `LLMContextFrame` whose last user message is that text; the
language event (`--lid`, AmberNet) fired at 3.06 s with `en` 0.98 and tagged the finals (`lang=en`); the VAD start /
stop frames come from the server's VAD head under Pipecat's hysteresis (the short stop / start pair at 9.47–10.68 s is
speaker B's pause after "heaven"); `EndFrame` flushed the session (third, empty final at 13.195 s, then `stats`). A
first run without the `Tap` (the aggregator consumes the transcription frames) gave the same two turns at 5.84 / 10.64 s
under a heavier backlog (2.7–3.4 s).

### 7.2 LiveKit (`AudioforgeFrontend`: `STT`, `VAD` and `TurnDetector` streams on one session)

The three streams are fed the same frames; `predict()` is issued at each VAD `end_of_speech`, the way `AgentSession`
does at a pause. "audio clock" = `speech_start_time` / `speech_end_time` relative to the first pushed frame's wall time.

```
WARNING:livekit.plugins.audioforge:audioforge session eae7361c5af6: server overloaded: backlog 1560 ms at rtf 1.28: level 1, diarizer skipped (VAD-only speaker activity)
   1017 ms  STT start_of_speech    235 ms of audio clock
   1017 ms  VAD start_of_speech t=0.40 p=0.00
   2836 ms  STT interim_transcript 'a cold' speaker=S0 primary=True lang=en t=None turn_end=None
   3352 ms  STT interim_transcript 'a cold lu' speaker=S0 primary=True lang=en t=None turn_end=None
   3945 ms  STT interim_transcript 'a cold lucid' speaker=S0 primary=True lang=en t=None turn_end=None
   4381 ms  STT interim_transcript 'a cold lucid indifference' speaker=S0 primary=True lang=en t=None turn_end=None
   4920 ms  STT interim_transcript 'a cold lucid indifference reign' speaker=S0 primary=True lang=en t=None turn_end=None
   5415 ms  STT interim_transcript 'a cold lucid indifference reigned in his soul' speaker=S0 primary=True lang=en t=None turn_end=None
   6418 ms  VAD end_of_speech   t=5.28 speech=4.48s -> predict()
   7013 ms  TURN prediction p=1.00 (turn_end) delay=1.2654247283935547
   7013 ms  STT final_transcript   'a cold lucid indifference reigned in his soul' speaker=S0 primary=True lang=en t=5.36 turn_end=timeout
   7013 ms  STT end_of_speech      4450 ms of audio clock
   8465 ms  STT start_of_speech    6946 ms of audio clock
   8465 ms  VAD start_of_speech t=6.64 p=0.00
   9122 ms  STT interim_transcript 'hea' speaker=S0 primary=True lang=en t=None turn_end=None
   9797 ms  STT interim_transcript 'heaven' speaker=S0 primary=True lang=en t=None turn_end=None
  10884 ms  STT interim_transcript 'heaven a good place' speaker=S1 primary=True lang=en t=None turn_end=None
  11900 ms  STT interim_transcript 'heaven a good place to be rais' speaker=S1 primary=True lang=en t=None turn_end=None
  12126 ms  STT interim_transcript 'heaven a good place to be raised to' speaker=S1 primary=True lang=en t=None turn_end=None
  12306 ms  VAD end_of_speech   t=10.08 speech=3.04s -> predict()
  13117 ms  TURN prediction p=1.00 (turn_end) delay=2.003739833831787
  13117 ms  STT final_transcript   'heaven a good place to be raised to' speaker=S1 primary=True lang=en t=10.4 turn_end=timeout
  13117 ms  STT end_of_speech      9833 ms of audio clock
  16897 ms  STT recognition_usage audio_duration=13.20

sessions opened: 1; predictions: [(1.0, 'turn_end'), (1.0, 'turn_end')]
stats: rtf=1.266 first_partial_ms=911.0 peak_rss_mb=1581.2 lang=en
```

Reading: one server session for the three streams (`sessions opened: 1`); `end_of_speech` is placed on the audio
clock at 4450 ms and 9833 ms, i.e. the last speech of each turn (labels 4475 / 9895 ms) reconstructed from
`turn_end.t − column_lag − silence_ms`, so LiveKit's `speech_end_time` is right even though the event itself arrives
2.5 s (turn 1) and 3.2 s (turn 2) later under this overload; the turn detector resolved p = 1.0 on the server's
`turn_end` for both pauses (`detection_delay` 1.27 / 2.00 s = the server's backlog), which under `endpointing` would
commit at `min_delay`; the diarizer moved speaker B to `S1` before load shedding switched it off (`primary=True` for
the current primary column); the `--lid` language reached the transcripts as `lang=en`; the VAD `start_of_speech`
events carry `p=0.00` because the start event reports the probability field of the aggregate, not of the triggering
frame (cosmetic, `INFERENCE_DONE` events carry the per-frame probability). The interims have `t=None` because only
finals carry `metadata["audioforge"]` (the interims' `end_time` holds the server time).

Both runs, plus the hermetic suites (16 Pipecat tests, 14 LiveKit tests), are the whole verification of the upstream
branches so far; no WebRTC room, microphone or real TTS was involved, and `enroll="after_agent_arm"`,
`turn_policy="hybrid_dyn"` and `final_source="offline"` were exercised only by the fake-server tests.
