# Upstream PRs: Pipecat service + LiveKit Agents plugin for the audioforge server

Updated 2026-10-03 (first prepared 2026-09-28). **No pull request is open** on `pipecat-ai/pipecat` or
`livekit/agents`, and the updated branches are **committed locally only, not pushed** (the user put pushing and
opening on hold). There is no PyPI step: the plugins no longer depend on `audioforge-client`. This file holds,
per project: the branch, the PR title and body ready to paste, the files on the branch, the upstream rules and how
they were met, and the steps left for the user.

| piece | where | state |
|---|---|---|
| `audioforge-client` 0.2.0 (WebSocket client, protocol types, fake server for tests; Apache-2.0, `websockets` only) | this repo, `packages/audioforge-client/` | committed on `main`; direct users install it from GitHub (`pip install "audioforge-client @ git+https://github.com/maxmelichov/audioforge#subdirectory=packages/audioforge-client"`); **not published to PyPI, and not needed there** |
| Pipecat `AudioforgeSTTService`, `AudioforgeVADAnalyzer`, `AudioforgeTurnAnalyzer`, `AudioforgeEagerTurnStopStrategy` | clone `/Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat`, branch `feat/audioforge-stt-turn` | one local commit `0b3e88fe4` on `pipecat-ai/pipecat` `main` `cae1f51a4`, `uv.lock` included; fork still holds the 2026-09-28 commit `940d163d7` |
| LiveKit `livekit-plugins-audioforge` (STT, VAD, TurnDetector) | clone `/Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/agents`, branch `feat/audioforge-stt-turn` | one local commit `afcc89e` on `livekit/agents` `main` `96341b0` (agents 1.8.4), `uv.lock` included; fork still holds `4945854` |

**No external dependency.** Neither upstream can take a git or URL dependency (both publish to PyPI, which refuses
direct references in metadata, and both CIs install from `uv.lock`), and the client is not going to PyPI. So the
protocol framing is vendored into each plugin as a private module, trimmed to what the plugin uses and depending on
the standard library only: Pipecat `src/pipecat/services/audioforge/_client.py` (522 lines), LiveKit
`livekit/plugins/audioforge/_client.py` (526 lines), each with the Apache-2.0 header and a line saying it comes
from `audioforge-client` 0.2.0. The sockets use what each repo already depends on: Pipecat's core `websockets`;
LiveKit's plugin lists `websockets>=14.0,<16.0` (already in `uv.lock` through other plugins; `livekit-agents` itself
does not depend on it). The fake server for the tests is vendored the same way (`tests/audioforge_fake_server.py`,
`tests/fake_audioforge_server.py`, 415 lines each). `uv lock` adds no package in either repo.

Forks: https://github.com/maxmelichov/pipecat and https://github.com/maxmelichov/agents (remote `origin`;
`upstream` = the upstream repo). The 2026-09-28 commits are kept locally as branch `backup/audioforge-0928` in each
clone. Virtualenvs: `/Volumes/afdev/venvs/{pipecat,livekit}` in the sparse bundle
`/Volumes/ExternalSSD/nvidia-audio-models/scratch/afdev.sparsebundle` (`hdiutil attach -nobrowse <bundle>`).

## 1. What is left for the user

1. Push both branches to the forks (the branches were rebuilt on the current upstream `main`, so the push replaces
   the 2026-09-28 commits): `git push --force-with-lease origin feat/audioforge-stt-turn` in each clone.
2. Open both PRs ready for review (§4). No PyPI step: the plugins carry their own protocol module and `uv.lock` is
   already in each commit.
3. Pipecat only: rename `changelog/0000.added.md` to `changelog/<PR number>.added.md`, amend, push (§4).
4. LiveKit only: sign the CLA when the CLA Assistant bot asks on the PR.

## 2. What changed since 2026-09-28

- **Server.** The branches targeted the old room stack (FastConformer + Streaming Sortformer v2, `timeout` policy,
  old E2E numbers). They now target the current server: single-model mode by default (the 115M core; `--core 0.6b`
  for the English Nemotron streaming 0.6B), and the protocol of `docs/PROTOCOL.md`.
- **Defaults.** `turn_policy="vad_head"` (the server's single-mode default, and the rule behind the test-split
  numbers) instead of `timeout`; `turn_preset` (None = the server's `balanced`; `fast`, `steady`, `assistant`). The
  reference adapters in `audioforge/integrations/` still default to `timeout`; the upstream plugins send `vad_head`
  explicitly so the measured rule runs on both server modes.
- **Protocol ported from `audioforge/integrations/{pipecat,livekit}.py`:** `turn_end_hint` / `turn_end_hint_cancel`
  (Pipecat: `EagerTranscriptionFrame` / `EagerEndOfTurnCancelFrame` + `AudioforgeEagerTurnStopStrategy`; LiveKit:
  `PREFLIGHT_TRANSCRIPT`; both opt-in with `turn_hints=True`), `turn_end.hinted_at` / `path`, `final_fast` + the
  `source: "slow"` final of `--final-chunk-ms` (`final_text="fast" | "slow"`, the turn analyzer / mapper holds the
  turn for the slow text), `enroll` with a stored voice print and the `voiceprint` event (Pipecat
  `enroll(embedding)` + `on_voiceprint`; LiveKit `fe.enroll(embedding)` + `fe.voiceprint`), optional
  `final.voice_gender` (copied into the transcript metadata), `language`.
- **LiveKit TurnDetector.** Policy `"turn_end"` (default) resolves on the server's cutting `turn_end` of whatever
  `turn_policy` the frontend uses (it listened only for `timeout` before, which never fires under `vad_head`);
  `prediction_timeout` covers the `assistant` preset's ~3 s fallback.
- **`audioforge-client` 0.2.0:** all of the above as typed messages (`TurnHint`, `TurnHintCancel`, `Voiceprint`,
  `Final.fast` / `stream_source` / `voice_gender`, `Ready.final_chunk_ms`), `config_message(turn_policy=None,
  turn_preset=...)` (omitted fields = the server's default), `control_message("enroll", embedding)`,
  `use_final(msg, final_source, final_text)`; the fake server gained `default_policy="vad_head"`, `hints`,
  `final_chunk_ms`, `voice_gender` and voice-print replies.
- **Client vendored (2026-10-03).** `audioforge-client` is no longer a dependency of either plugin (no PyPI
  upload planned); the plugins carry a trimmed private copy of its protocol module and the tests a trimmed copy of
  its fake server. Pipecat's `audioforge` extra stays as an empty extra, like `assemblyai = []`.
- **Rebased** on the current upstream `main` of both projects (Pipecat 165 commits newer; LiveKit 23 newer, now
  1.8.4: the plugin version, its `livekit-agents` floor and the extra follow 1.8.4).
- **PR bodies** rewritten from `research/FINAL_COMPARE.md` headline rows (public test splits only, nothing from its
  Appendix A), with the limitations stated.

## 3. Checks run (2026-10-03, local, after vendoring; `audioforge-client` uninstalled from both venvs)

| repo | command (its CI's) | result |
|---|---|---|
| audioforge-client | `pytest tests`, `ruff format/check src tests`, `mypy --strict src/audioforge_client` | 16 passed; clean; no issues |
| audioforge-client | in throwaway venvs: `pip install ./packages/audioforge-client`, and `pip install "audioforge-client @ git+https://github.com/maxmelichov/audioforge#subdirectory=packages/audioforge-client"` (the public repo's current `main`), then `import audioforge_client` | both install and import 0.2.0 (requires only `websockets`) |
| Pipecat | `pytest tests/test_audioforge_stt.py tests/test_audioforge_turn.py` | 21 passed |
| Pipecat | `ruff format --check`, `ruff check` on the audioforge files | clean |
| Pipecat | `pyright` on the new modules (`services/audioforge`, the VAD and turn analyzers, the fake server) | 0 errors |
| Pipecat | `scripts/cli/check_registry.py` | all services have configs and imports |
| Pipecat | `uv lock --check` (uv 0.11.0; uv 0.12 rewrites the whole lock to revision 5) | in sync; the commit's lock change is the one line adding the empty `audioforge` extra |
| LiveKit | `pytest --plugin audioforge` | 18 passed |
| LiveKit | `ruff check`, `ruff format --check` on the plugin and its tests | clean |
| LiveKit | `scripts/check_types.py` (strict mypy, `--platform linux`, every package) | no issues in 699 source files |
| LiveKit | `uv lock --check` (uv 0.11.0) | in sync; the lock change is the new workspace member only (+22 / -1, no new package) |

Earlier full-suite runs, before vendoring (2026-10-03): Pipecat `pytest` 6616 passed, 154 skipped; LiveKit
`pytest --unit --audio_eot` 3902 passed, 5 skipped, 9 errors in `tests/test_room.py` (needs a local
`livekit-server` binary, unrelated).

All tests are hermetic: the socket tests run against the vendored fake server. The branches were **not** re-run
against a live server or in a WebRTC room for this update (the 2026-09-28 10 s live check covered the old room
stack only).

## 4. Commands to push and open the PRs (only when the user decides)

```bash
# 0. no PR from these branches yet
gh pr list --repo pipecat-ai/pipecat --author maxmelichov --state all
gh pr list --repo livekit/agents --author maxmelichov --state all

# 1. PR bodies: the ~~~markdown blocks of §5.2 and §6.2
python3 - <<'EOF'
import re, pathlib
doc = pathlib.Path("/Users/maxm/nvidia-audio-models/docs/UPSTREAM_PRS.md").read_text()
bodies = re.findall(r"### \d\.2 Body\n\n~~~markdown\n(.*?)\n~~~\n", doc, re.S)
pathlib.Path("/tmp/pr_pipecat.md").write_text(bodies[0]); pathlib.Path("/tmp/pr_livekit.md").write_text(bodies[1])
EOF

# 2. Pipecat: push, open, then name the changelog fragment after the PR number
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/pipecat
git push --force-with-lease origin feat/audioforge-stt-turn
gh pr create --repo pipecat-ai/pipecat --base main --head maxmelichov:feat/audioforge-stt-turn \
  --title "Add audioforge STT service, VAD analyzer and turn analyzer" --body-file /tmp/pr_pipecat.md
N=$(gh pr view --repo pipecat-ai/pipecat maxmelichov:feat/audioforge-stt-turn --json number -q .number)
git mv changelog/0000.added.md changelog/$N.added.md && git commit --amend --no-edit
git push --force-with-lease origin feat/audioforge-stt-turn

# 3. LiveKit: push, open (sign the CLA when the bot asks)
cd /Volumes/ExternalSSD/nvidia-audio-models/scratch/upstream/agents
git push --force-with-lease origin feat/audioforge-stt-turn
gh pr create --repo livekit/agents --base main --head maxmelichov:feat/audioforge-stt-turn \
  --title "feat(audioforge): new plugin - self-hosted streaming STT, VAD and end-of-turn detection" \
  --body-file /tmp/pr_livekit.md
```

Both open ready for review (no `--draft`): nothing waits on an upload.

## 5. Pipecat PR

### 5.1 Rules followed (CONTRIBUTING.md, AGENTS.md, .claude/skills/pr-description)

| rule | how |
|---|---|
| fork, branch, one meaningful commit | `feat/audioforge-stt-turn`, one commit on `main` `cae1f51a4` |
| changelog fragment `<PR>.<type>.md`, one bullet, no wrapping | `changelog/0000.added.md`; rename after the PR exists (§1 step 3, commands in §4) |
| `uv lock` + `uv sync` after editing `pyproject.toml`, commit both | done: `uv.lock` is in the commit (one line, the empty `audioforge` extra) |
| Ruff (line length 100), Google docstrings (`Args:` on `__init__`, `Parameters:` on dataclasses / params) | done; `ruff check` / `ruff format` clean |
| `self.create_task` / `task_manager`, nothing slow in `process_frame` | receive loop via `_receive_task_handler`; the strategy's timer via `task_manager` |
| errors via `push_error(..., fatal=False)` | fatal server errors pushed, degradations logged as warnings |
| tests under `tests/` (`run_test()` for pipeline behaviour) | `tests/test_audioforge_stt.py` (11), `tests/test_audioforge_turn.py` (10, two through `LLMUserAggregator` with `run_test`) |
| PR description: opening paragraph + labeled bullets, no Testing section or test counts, "writing for future readers" | body §5.2 follows it; it is longer than the skill's 200-word aim because the user asked for the measured numbers, limitations, how to run and licences |
| AI-assisted contributions | no rule for PRs (only for bug reports); the body ends with the Claude Code line |
| community-integration route (COMMUNITY_INTEGRATIONS.md) | possible: the modules import only public Pipecat APIs and lift out into a `pipecat-audioforge` package unchanged if the maintainers prefer that |

### 5.2 Body

~~~markdown
audioforge is a self-hosted speech server I maintain (https://github.com/maxmelichov/audioforge). It runs one streaming model per session, NVIDIA's 115M streaming FastConformer (or NVIDIA's English Nemotron streaming 0.6B with `--core 0.6b`), with learned speech, end-of-turn and voice-print heads, so the transcript, the VAD and the end-of-turn decision all come from one WebSocket session. This PR adds a Pipecat STT service and two analyzers on that session; no model runs in the Pipecat process.

- **`AudioforgeSTTService`** (`WebsocketSTTService`). Partials become `InterimTranscriptionFrame` and finals `TranscriptionFrame(finalized=True)`, cut exactly where the server ends the turn. `Settings` carry the server's turn policy (`vad_head`, the server default) and preset (`balanced`, `fast`, `assistant`); changing them reconnects. `enroll(embedding)` sends the user's stored voice print, and `on_voiceprint` reports the print the server uses so the app can store it.
- **`AudioforgeVADAnalyzer`.** Pipecat's own start/stop hysteresis on the server's per-frame speech probability.
- **`AudioforgeTurnAnalyzer`.** Reports the server's `turn_end` to `TurnAnalyzerUserTurnStopStrategy`. The decision is held until Pipecat ends the turn, applied only while the VAD hears silence, and dropped if the server hears speech resume after it.
- **Early hints.** With `turn_hints=True` the server's end-of-turn hint becomes an `EagerTranscriptionFrame` and its withdrawal an `EagerEndOfTurnCancelFrame`. `AudioforgeEagerTurnStopStrategy` speculates on the hint like `EagerUserTurnStopStrategy`, but the turn still ends on the server's `turn_end`, so nothing reaches the user earlier.
- **Second-pass finals.** `final_text="slow"` (server `--final-chunk-ms`) or `final_source="offline"` (server `--final-asr`) push the better per-turn text and hold the turn until it arrives.
- **No new dependency.** The protocol framing is a private, stdlib-only module (`services/audioforge/_client.py`, Apache-2.0, from the server's `audioforge-client`), and the socket uses Pipecat's own `websockets`. `pipecat-ai[audioforge]` is an empty extra.

Measured on public test splits, every baseline re-run on the same audio, one Apple M5 Mac (server repo: `research/FINAL_COMPARE.md`):

| | audioforge 115M (default) | audioforge 0.6B | others |
|---|---|---|---|
| WER, AMI test meetings (200 segments) | 16.1 % | 7.9 % | Parakeet-TDT 0.6B v3 (offline) 8.3 %, Whisper large-v3 10.7 %, Whisper small 10.9 % |
| WER, LibriSpeech test-other (300 utterances) | 6.8 % | 5.7 % | Parakeet-TDT 3.4 %, Whisper small 7.1 % |
| word shown after it is said (streaming, p50, Mac GPU) | 274 ms | 291 ms | offline models: after the turn |
| speech detection F1, AMI test | 0.959 | 0.957 | Silero 0.901, TEN 0.930, MarbleNet 0.941 (MarbleNet ties ours on AUC) |
| AMI test turns (200), user's voice print known: end-of-turn latency p50 / false interruptions / missed | 1527 ms / 15.5 % / 36.0 % | 1498 ms / 10.0 % / 33.5 % | smart-turn v3.2 + Silero (Pipecat defaults): 752 ms / 39.5 % / 53.0 % |
| compute per 160 ms of audio on 2 CPU threads / real-time streams / memory | 30.4 ms / 4 / 1.2 GB | 97.1 ms / 1 / 4.9 GB | |

On speech directed at an assistant (smart-turn v3.2 test, 399 clips, a turn end within 2.5 s of an unfinished utterance counts as wrong), the 0.6B's `assistant` preset is right on 97.7 % of clips with a 351 ms p50, against 75.9 % at 211 ms for smart-turn v3.2 + Silero. That preset reuses a classifier recipe partly chosen on these clips (flagged in the results file). The 115M's default `balanced` preset is a conversation preset and ends the turn on every clip that stops mid-sentence.

Limitations:

- Pipecat's smart-turn stack answers faster: 211 vs 351 ms on assistant clips, and 752 ms vs about 1.5 s on meetings, where it interrupts on 39.5 % of turns.
- The default 115M's transcripts trail the offline models (AMI 16.1 % against 8.3 % for Parakeet-TDT and 10.7 % for Whisper large-v3). The 0.6B ties Parakeet-TDT on AMI (the difference's CI includes 0), trails it on ICSI meetings (10.3 vs 7.6 %) and LibriSpeech, and needs about 3x the compute.
- Turn detection on meetings relies on the user's voice print: without it both cores miss about 74 % of turn ends.
- English-only ASR. Self-hosted only, measured on one machine. The integration's tests run against a scripted fake server that speaks the protocol (`tests/audioforge_fake_server.py`), not in a live WebRTC room.

How to run:

```bash
pip install "pipecat-ai[audioforge]"
git clone https://github.com/maxmelichov/audioforge && cd audioforge
pip install -e ".[serve]" && audioforge-download && audioforge-serve   # ws://127.0.0.1:8765; --core 0.6b, --device cuda|mps
python examples/turn-management/turn-management-audioforge.py -t webrtc   # AUDIOFORGE_TURN_HINTS=1, AUDIOFORGE_VOICEPRINT=me.json
```

Licences: the audioforge server and the vendored protocol module and test fake server (from `audioforge-client`) are Apache-2.0. NVIDIA's 115M streaming FastConformer and Parakeet-TDT are CC-BY-4.0; the 0.6B (`nvidia/nemotron-speech-streaming-en-0.6b`) is under the NVIDIA Open Model License. The audioforge heads come from the server repo's GitHub release; `audioforge-download` shows each licence and asks before fetching.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
~~~

### 5.3 Files on the branch (`git diff --stat upstream/main..HEAD`: 13 files, +2944 / -1)

| file | lines | what |
|---|---|---|
| `src/pipecat/services/audioforge/stt.py` | 745 | `POLICIES`, `ReceivedTurnEnd`, `AudioforgeHub`, `AudioforgeSTTSettings`, `AudioforgeSTTService` |
| `src/pipecat/services/audioforge/__init__.py` | 21 | exports |
| `src/pipecat/services/audioforge/_client.py` | 522 | private protocol framing (typed events, `parse_message`, `config_message`, `control_message`, `use_final`), trimmed from `audioforge-client` 0.2.0, Apache-2.0, stdlib only |
| `src/pipecat/audio/turn/audioforge_turn.py` | 343 | `AudioforgeTurnParams`, `AudioforgeTurnAnalyzer`, `AudioforgeEagerTurnStopStrategy` |
| `src/pipecat/audio/vad/audioforge_vad.py` | 77 | `AudioforgeVADAnalyzer` |
| `tests/test_audioforge_stt.py` | 303 | defaults / validation, fake-server round trip (timeout policy, `agent_end`, language), offline finals, batches + errors, hints -> eager frames, hints off, `final_text` fast / slow + preset + voice_gender, stored voice print |
| `tests/audioforge_fake_server.py` | 415 | scripted fake server for the socket tests, trimmed from `audioforge_client.testing` (Apache-2.0) |
| `tests/test_audioforge_turn.py` | 230 | latch / silence gate / clear, stale after resumed speech, policy filter, second-final gating (offline, slow), reset, VAD analyzer, eager strategy kept / re-answered through `LLMUserAggregator` |
| `examples/transcription/transcription-audioforge.py` | 109 | transcription + `VADProcessor` on the server's speech probability |
| `examples/turn-management/turn-management-audioforge.py` | 176 | full bot; `AUDIOFORGE_TURN_PRESET`, `AUDIOFORGE_VOICEPRINT`, `AUDIOFORGE_TURN_HINTS` |
| `pyproject.toml`, `uv.lock` | +1 / 1 line | empty extra `audioforge = []`; the lock's `provides-extras` |
| `changelog/0000.added.md` | 1 | fragment |

## 6. LiveKit Agents PR

### 6.1 Rules followed (CONTRIBUTING.md, AGENTS.md, REVIEW.md)

| rule | how |
|---|---|
| plugin layout as the other plugins (`livekit/plugins/<name>/{__init__,log,version}.py`, hatchling, `py.typed`) | done |
| `ruff check --fix` + `ruff format` (`make fix`) | clean repo-wide |
| `make type-check` (strict mypy over every package) | no issues |
| pdoc docs on every public class / method | docstrings on the public surface; `__pdoc__` hides the rest |
| test module with exactly one category marker | `pytestmark = pytest.mark.plugin("audioforge")`; runs with `--plugin audioforge`, hermetic |
| no `version.py` bumps / no `CHANGELOG.md` edits outside release PRs | the new plugin starts at the current agents version 1.8.4; no changelog edit |
| REVIEW.md: no transcripts or other PII in logs | the plugin logs session ids, server error codes / details and probabilities only |
| CLA Assistant on the first PR | to sign when the PR is opened |
| `uv lock` after adding the workspace member; CI runs `uv sync --locked` | done: `uv.lock` is in the commit (workspace member only, no new package) |

### 6.2 Body

~~~markdown
`livekit-plugins-audioforge` connects LiveKit Agents to audioforge, a self-hosted speech server I maintain (https://github.com/maxmelichov/audioforge). The server runs one streaming model per session, NVIDIA's 115M streaming FastConformer (or NVIDIA's English Nemotron streaming 0.6B with `--core 0.6b`), with learned speech, end-of-turn and voice-print heads. One `AudioforgeFrontend` opens one server session per `AgentSession` and hands it to an STT, a VAD and a turn detector, so the user turn is committed exactly where the server cut the transcript.

- **`STT`** (streaming, interim results). `partial` -> `INTERIM_TRANSCRIPT`, `final` -> `FINAL_TRANSCRIPT` with `metadata["audioforge"]`, and the server's `turn_end` -> `END_OF_SPEECH` right after its final, for `turn_detection="stt"`. With `turn_hints=True` the server's early end-of-turn hint becomes a `PREFLIGHT_TRANSCRIPT`, so preemptive generation prepares the reply and LiveKit plays it only if the committed transcript matches.
- **`VAD`.** `VADEvent`s from the server's per-frame speech probability, for barge-in with `interruption: {"mode": "vad"}`.
- **`TurnDetector`.** For LiveKit's endpointing min/max delay logic: resolves p = 1.0 when the server's `turn_end` arrives for the current pause. It implements the private `voice.turn._StreamingTurnDetector` protocol; `turn_detection="stt"` is the recommended mode, and I can drop the detector if you'd rather not carry a private interface.
- **Dependencies.** `livekit-agents`, `numpy` and `websockets`, all already in the workspace lock. The protocol framing is a private, stdlib-only module (`livekit/plugins/audioforge/_client.py`, Apache-2.0, from the server's `audioforge-client`).
- **Options.** `turn_policy` (`vad_head`, the server default), `turn_preset` (`balanced`, `fast`, `assistant`), `final_text="slow"` (server `--final-chunk-ms`) or `final_source="offline"` (server `--final-asr`) for the better second-pass text of each turn. `fe.enroll(embedding)` sends the user's stored voice print; `fe.voiceprint` holds the one the server uses.

Measured on public test splits, every baseline re-run on the same audio, one Apple M5 Mac (server repo: `research/FINAL_COMPARE.md`):

| | audioforge 115M (default) | audioforge 0.6B | others |
|---|---|---|---|
| WER, AMI test meetings (200 segments) | 16.1 % | 7.9 % | Parakeet-TDT 0.6B v3 (offline) 8.3 %, Whisper large-v3 10.7 %, Whisper small 10.9 % |
| WER, LibriSpeech test-other (300 utterances) | 6.8 % | 5.7 % | Parakeet-TDT 3.4 %, Whisper small 7.1 % |
| word shown after it is said (streaming, p50, Mac GPU) | 274 ms | 291 ms | offline models: after the turn |
| speech detection F1, AMI test | 0.959 | 0.957 | Silero 0.901, TEN 0.930, MarbleNet 0.941 (MarbleNet ties ours on AUC) |
| AMI test turns (200), user's voice print known: end-of-turn latency p50 / false interruptions / missed | 1527 ms / 15.5 % / 36.0 % | 1498 ms / 10.0 % / 33.5 % | LiveKit turn detector (EnglishModel) + Silero: 745 ms / 13.0 % / 72.0 % |
| compute per 160 ms of audio on 2 CPU threads / real-time streams / memory | 30.4 ms / 4 / 1.2 GB | 97.1 ms / 1 / 4.9 GB | |

On speech directed at an assistant (smart-turn v3.2 test, 399 clips, a turn end within 2.5 s of an unfinished utterance counts as wrong), the 0.6B's `assistant` preset is right on 97.7 % of clips with a 351 ms p50, against 85.2 % at 547 ms for LiveKit's EnglishModel + Silero (min / max delay 0.5 / 3.0 s). That preset reuses a classifier recipe partly chosen on these clips (flagged in the results file). The 115M's default `balanced` preset is a conversation preset and ends the turn on every clip that stops mid-sentence.

Limitations:

- On meetings LiveKit's stack answers faster (745 ms vs about 1.5 s p50), though it misses 72 % of turn ends; on assistant clips Pipecat's smart-turn answers faster still (211 ms).
- The default 115M's transcripts trail the offline models (AMI 16.1 % against 8.3 % for Parakeet-TDT and 10.7 % for Whisper large-v3). The 0.6B ties Parakeet-TDT on AMI (the difference's CI includes 0), trails it on ICSI meetings (10.3 vs 7.6 %) and LibriSpeech, and needs about 3x the compute.
- Turn detection on meetings relies on the user's voice print: without it both cores miss about 74 % of turn ends.
- English-only ASR. Self-hosted only, measured on one machine. The plugin's tests run against a scripted fake server that speaks the protocol (`tests/fake_audioforge_server.py`), not in a live room.

How to run:

```bash
pip install livekit-plugins-audioforge
git clone https://github.com/maxmelichov/audioforge && cd audioforge
pip install -e ".[serve]" && audioforge-download && audioforge-serve   # ws://127.0.0.1:8765; --core 0.6b, --device cuda|mps
```

```python
from livekit.plugins import audioforge

fe = audioforge.AudioforgeFrontend("ws://127.0.0.1:8765")
session = AgentSession(
    stt=fe.stt(), vad=fe.vad(), llm=..., tts=...,
    turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.0},
                   "interruption": {"mode": "vad"}},
)
```

Licences: the plugin, its vendored protocol module and the test fake server are Apache-2.0, as is the audioforge server. NVIDIA's 115M streaming FastConformer and Parakeet-TDT are CC-BY-4.0; the 0.6B (`nvidia/nemotron-speech-streaming-en-0.6b`) is under the NVIDIA Open Model License. The audioforge heads come from the server repo's GitHub release; `audioforge-download` shows each licence and asks before fetching.

Notes for review: the plugin starts at the current agents version (1.8.4); it is registered in `[tool.uv.sources]` and as the `livekit-agents[audioforge]` extra; its tests need no credentials (`uv run pytest --plugin audioforge`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
~~~

### 6.3 Files on the branch (`git diff --stat upstream/main..HEAD`: 16 files, +2993 / -1)

| file | lines | what |
|---|---|---|
| `livekit-plugins/livekit-plugins-audioforge/livekit/plugins/audioforge/frontend.py` | 573 | `POLICIES`, `AudioforgeOptions`, `_PcmConverter`, `_Link` (one socket: single feeder, ordered writer queue, reader fan-out, `wall_of(t)`, voice print), `AudioforgeFrontend` (factories, `agent_end`, `enroll(embedding)`, `voiceprint`, `attach`, session sharing) |
| `.../_client.py` | 526 | private protocol framing (typed events, `parse_message`, `config_message`, `control_message`, `cut_policy`, `use_final`), trimmed from `audioforge-client` 0.2.0, Apache-2.0, stdlib only |
| `.../stt.py` | 343 | `_SpeechMapper` (pure state machine: hints, finals, second-pass finals, END_OF_SPEECH timing), `STT`, `SpeechStream` |
| `.../vad.py` | 247 | `_VadMapper`, `VAD`, `VADStream` (reconnects up to 3x on a lost session) |
| `.../turn_detector.py` | 262 | `TurnDetector`, `TurnDetectorStream` (`_StreamingTurnDetectorStream`) |
| `.../__init__.py`, `log.py`, `version.py` (1.8.4), `py.typed` | 58 / 3 / 15 / 0 | template files |
| `livekit-plugins/livekit-plugins-audioforge/pyproject.toml`, `README.md` | 45 / 67 | metadata (`livekit-agents`, `numpy`, `websockets>=14.0,<16.0`), usage |
| `tests/fake_audioforge_server.py` | 415 | scripted fake server, trimmed from `audioforge_client.testing` (Apache-2.0) |
| `tests/test_plugin_audioforge.py` | 415 | 18 tests: options, shared session for STT + VAD + turn detector with `agent_end` ordering, lost session, offline finals + language, batch `recognize`, hints -> preflight, `final_text` fast / slow, stored voice print, mappers, turn detector paths |
| `pyproject.toml`, `livekit-agents/pyproject.toml`, `uv.lock` | +1 / +1 / +22 -1 | workspace member; extra `audioforge = ["livekit-plugins-audioforge>=1.8.4"]`; the lock entries for both |

`.github/workflows/tests.yml` is not touched: its plugin matrix lists cloud plugins with keys. Ask the maintainers
whether the credential-free `--plugin audioforge` tests should join it.

## 7. What the maintainers will likely ask

**Both**
- "Why vendor the protocol module instead of depending on a package?" It keeps the PR free of new dependencies; it
  is ~520 lines of stdlib-only message types and parsing, Apache-2.0, and changes only with the server protocol.
- "Is there a hosted endpoint?" No, self-hosted only. Cost on the Mac: 115M 30 ms per 160 ms chunk on 2 CPU threads
  (4 real-time streams), 1.2 GB; GPU numbers in the results file. Not measured on a Linux server or under many
  concurrent sessions.
- "Who maintains it?" The user, as the server's author; no company behind it.
- Licences of the models and of the evaluation data (AMI CC BY 4.0; LibriSpeech CC BY 4.0; smart-turn v3.2 test
  set as published by Pipecat).
- Reconnect semantics: a new server session starts the user tracking over; send the stored voice print again
  (both plugins queue `enroll` / `agent_end` sent before the session and replay them after the config).
- A demo video (Pipecat's community guide asks for 30-60 s with an interruption): not recorded.

**Pipecat**
- They may ask for a community integration instead (separate `pipecat-audioforge` repo, a row on the Supported
  Services page, a docs PR). The modules use only public Pipecat APIs and move out unchanged.
- Why a turn analyzer and not `ExternalUserTurnStrategies` + `ProposedUser*SpeakingFrame` (the pattern of
  provider-side endpointing)? The analyzer keeps Pipecat's VAD gate; the external-strategy variant is a small
  follow-up (the hub already has every event).

**LiveKit**
- The private `_StreamingTurnDetector`: they may want STT + VAD only, or a transport for `inference.TurnDetector`.
- `is_primary_speaker` / `speaker_id` come from the server's speaker column (single mode: 0 = the user, 1 = someone
  else).
- Whether the plugin's tests join the CI matrix; Renovate config for the new package; docs live in a separate repo.
