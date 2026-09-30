"""LiveKit agent worker: the audioforge front end (STT + VAD + turn end) in a real LiveKit room.

    # 1. a LiveKit server (dev keys devkey / secret):   livekit-server --dev     (brew install livekit)
    #    or: docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
    # 2. the audioforge server:
    PYTHONPATH=. .venv/bin/python -m audioforge.serve --asr runs/stage1_heads_pretrained.afm \
        --diar runs/nemo_sortformer_v2.afm --port 8791 --threads 2
    # 3. this worker (registers with the LiveKit server, joins rooms on dispatch):
    LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \
    AUDIOFORGE_URL=ws://127.0.0.1:8791 PYTHONPATH=. .venv/bin/python examples/livekit_agent_worker.py dev
    # 4. a caller: examples/livekit_publish_wav.py (publishes a WAV as a microphone track) or the Agents Playground

Environment: AUDIOFORGE_URL (default ws://127.0.0.1:8765), AUDIOFORGE_TURN = "stt" (default: LiveKit commits the
turn on our END_OF_SPEECH = the server's silence timeout on the diarizer's primary track) or "detector"
(AudioforgeTurnDetector through LiveKit's audio turn-detector protocol), AUDIOFORGE_TIMEOUT_MS (1000),
AUDIOFORGE_TURN_HINTS=1 (LiveKit preemptive generation on the server's turn_end_hint; default off),
AUDIOFORGE_LOG (JSONL of every session event with wall-clock times), AUDIOFORGE_IDLE_PROCS (prewarmed job
processes, default 1).

LLM and TTS are local stubs (examples/livekit_offline_demo.py: an LLM that answers "You said: <turn>" and a tone TTS),
so the room test needs no API keys; swap in real plugins (e.g. livekit-plugins-openai) for a conversation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli  # noqa: E402
from livekit_offline_demo import StubLLM, StubTTS  # noqa: E402

from audioforge.integrations.livekit import AudioforgeFrontend  # noqa: E402

logger = logging.getLogger("audioforge-worker")


def build_session(fe: AudioforgeFrontend, turn: str = "stt") -> AgentSession:
    """The AgentSession configuration used in a room (and, room-less, by examples/livekit_offline_demo.py)."""
    th = {"endpointing": {"min_delay": 0.0, "max_delay": 3.0},  # the server already waited timeout_ms
          "interruption": {"mode": "vad"},                     # barge-in from our VAD frames (no cloud model)
          # preemptive generation on the server's turn_end_hint (PREFLIGHT_TRANSCRIPT) only with turn hints on
          "preemptive_generation": {"enabled": fe.opts.turn_hints},
          "turn_detection": "stt" if turn == "stt" else fe.turn_detector()}
    return AgentSession(stt=fe.stt(), vad=fe.vad(), llm=StubLLM(), tts=StubTTS(), turn_handling=th,
                        user_away_timeout=None)


def attach_logger(session: AgentSession, path: str | None):
    f = open(path, "a") if path else None

    def rec(kind, **kw):
        line = {"wall": time.time(), "kind": kind, **kw}
        logger.info("%s %s", kind, json.dumps(kw))
        if f:
            f.write(json.dumps(line) + "\n")
            f.flush()

    session.on("user_input_transcribed", lambda ev: rec("user_transcribed", text=ev.transcript, final=ev.is_final,
                                                        speaker=ev.speaker_id))
    session.on("conversation_item_added", lambda ev: rec("item", role=getattr(ev.item, "role", None),
                                                         text=getattr(ev.item, "text_content", None)))
    session.on("user_state_changed", lambda ev: rec("user_state", state=ev.new_state))
    session.on("agent_state_changed", lambda ev: rec("agent_state", state=ev.new_state))
    return f


# one prewarmed job process (the default spawns several; each imports livekit + this plugin)
server = AgentServer(num_idle_processes=int(os.environ.get("AUDIOFORGE_IDLE_PROCS", "1")))


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    fe = AudioforgeFrontend(os.environ.get("AUDIOFORGE_URL", "ws://127.0.0.1:8765"),
                            timeout_ms=int(os.environ.get("AUDIOFORGE_TIMEOUT_MS", "1000")),
                            turn_hints=os.environ.get("AUDIOFORGE_TURN_HINTS", "0") == "1")
    session = build_session(fe, os.environ.get("AUDIOFORGE_TURN", "stt"))
    f = attach_logger(session, os.environ.get("AUDIOFORGE_LOG"))

    async def _shutdown():
        await fe.aclose()
        if f:
            f.close()
    ctx.add_shutdown_callback(_shutdown)
    await session.start(agent=Agent(instructions="Repeat what the user said (audioforge room test)."),
                        room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
