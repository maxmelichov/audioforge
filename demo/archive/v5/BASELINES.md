# The default stacks in the live runs: verified settings

Verified from `scripts/research/e2e_final.py` (E) and the installed packages in the repo `.venv` (SP = site-packages):
pipecat-ai 1.12.0, livekit-agents 1.8.3, livekit-plugins-silero 1.8.3, livekit-plugins-turn-detector 1.8.3,
faster-whisper 1.2.1. The film says "Pipecat 1.12.0" and "LiveKit Agents 1.8.3".

## A: Pipecat 1.12.0 default local stack (E:476-481)

| item | run | framework default | same? |
|---|---|---|---|
| Silero VAD confidence / start / stop / min volume | `SileroVADAnalyzer()` | 0.7 / 0.2 s / 0.2 s / 0.6 (SP/pipecat/audio/vad/vad_analyzer.py:25-28) | yes |
| turn stop strategy | default `UserTurnStrategies()` = smart-turn v3.2 (bundled ONNX) | turns/user_turn_strategies.py:53 | yes |
| smart-turn stop_secs / pre_speech_ms / max_duration | 3 s / 500 ms / 8 s | base_smart_turn.py:27-29 | yes |
| aggregator user_turn_stop_timeout | 5 s | llm_response_universal.py:182 | yes |
| **Whisper model** | **small** (E:54) | **distil-medium.en** (services/whisper/stt.py:281) | **no: disclosed on screen** (same Whisper small in A and B) |
| Whisper language / no_speech_prob / beam | en / 0.4 / 5 | stt.py:282-285, faster-whisper defaults | yes |
| interruptions | enabled | default | yes |

Stock 1.12.0 decodes Whisper on the event loop (fixed upstream after the release, PR #5931); that is the release's
own behaviour.

## B: LiveKit Agents 1.8.3 (E:577-665)

| item | run | framework default | same? |
|---|---|---|---|
| Silero VAD | `silero.VAD.load()`: 0.05 / 0.55 / 0.5 / 0.5 | vad.py:63-73 | yes |
| turn detector | `EnglishModel` (v1.2.2-en, q8) | no model chosen when unset | our pick of LiveKit's local English plugin |
| min / max endpointing delay | 0.5 / 3.0 s | turn.py:135-140 | yes |
| interruptions | VAD mode, 0.5 s, resume false interruptions | turn.py:190-198 | yes (self-hosted default) |
| preemptive generation | off | on | no (does not move the scored user-turn moment) |
| EOU runner | in process, 2 threads | job executor, 4 threads | latency only |
| STT | harness faster-whisper small (LiveKit ships no local STT) | none | not a LiveKit default: disclosed on screen |

## Harness (both)
WAV at 1x in 20 ms frames; no LLM or TTS audio (Pipecat: mock LLM/TTS; LiveKit: echo LLM, tone TTS into a null
output); a cut-in = a response moment inside a labelled user-turn span (audioforge/e2e_metrics.py:58).
