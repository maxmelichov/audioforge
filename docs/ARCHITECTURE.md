# Architecture

One page: the model that runs, the server around it, and where each piece lives in the code. The measurements
behind each choice are in [`research/FINAL_REPORT.md`](../research/FINAL_REPORT.md) (§8 is the recommended stack).
The server's design notes, such as the frame clock and the exact policy rules, are in
[`SERVER_INTERNALS.md`](SERVER_INTERNALS.md).

## The model

```
 16 kHz PCM ──► 160 ms chunk ──► 80-bin log-mel ──► 17 x FastConformer block (115M, frozen, NVIDIA)
                                                     cache-aware streaming, 80 ms lookahead
                                                        │
        ┌───────────────────────┬───────────────────────┼───────────────────────┬─────────────────────────┐
        ▼                       ▼                       ▼                       ▼                         ▼
   RNNT decoder            VAD head               speaker head           turn head (GRU)           TS-VAD head
   "what was said"         "is anyone             "who is this?"         "is the turn over?"       "is it the user?"
   partial transcripts     speaking?"             block 4, 0.50 M        a second pass of the      block 4, 0.26 M,
                           block 4, 33 K          distilled from         same encoder, told who    conditioned on a
                                                  TitaNet-L              is speaking               voice print (flags)

 also on the same encoder:  LID head (blocks 8-12, 0.92 M, distilled from AmberNet)   "which language?"

 room mode only (--mode room):
              NVIDIA Nemotron-3-Diarization (or Streaming Sortformer v2)   "who is in the room": 4-8 activity columns
              NVIDIA Parakeet-TDT 0.6B v3 on each finished turn (optional)   rewrites the turn's transcript
              TitaNet-L (voice enrollment), AmberNet (language ID) (optional)
```

**Two modes.** In **single-model mode**, the default since 2026-09-29 (docs/CONFIGURATION.md §13), only the model
above runs. The user's stored voice print (192 numbers from the speaker head; ≥ 5 s of clean speech, 10 s for
meetings) conditions the TS-VAD head. Its track [P(user), P(someone else), 0, 0] is `frame.speakers` and is what the
turn head reads. No diarizer is loaded, and server RTF is 0.33 on 2 threads (research/SINGLE_MODEL.md). **Room mode**
(`--mode room`) adds NVIDIA's diarizer to label everyone in a room, and optionally Parakeet-TDT v3 for the final
transcript. The rest of this page notes where the two differ.

- **The encoder** is NVIDIA's `stt_en_fastconformer_hybrid_large_streaming_multi`: 17 FastConformer blocks,
  cache-aware, run at attention context [70, 1], which gives 160 ms chunks and 80 ms lookahead. It is imported without
  NeMo (`audioforge/nemo_import.py`) and **never fine-tuned**: every fine-tuning attempt raised LibriSpeech WER (2.05 to
  4.31 % in 500 steps), so everything this project learned lives in the heads.
- **The heads** read the encoder's frames at no extra encoder cost. The VAD head reads block 4
  (`stage1_served_v2.afm`, research/VAD_SINGLE.md). The measured 2026-09-27 checkpoint used a learned mix of all 17
  blocks. The speaker head reads block 4 too, where speaker information lives; the top blocks are speaker-blind. The turn head
  runs on a second, speaker-conditioned pass of the same encoder, fed the primary speaker's diarizer column. The
  TS-VAD head replaces that column with the enrolled user's activity (single-model mode; `--turn-input tsvad`).
- **The diarizer** (room mode only) is NVIDIA's, because our own general diarization head is not good enough (0.394
  vs 0.232 DER). It runs on its own front end on the same 80 ms clock. Its columns arrive 80-240 ms after their audio with the default 0.32 s
  setting.
- **End of turn** is a policy over these signals, chosen per session: a silence timeout on the primary speaker's
  column (`timeout`, the default), the head's probability OR a Silero silence with a dynamic wait (`hybrid_dyn`),
  any speaker's silence (`timeout_any`), and others ([`CONFIGURATION.md`](CONFIGURATION.md) §4).
- **Speed** comes from exact CPU fast paths (`audioforge/perf.py`, on by default): column-major weights, cached
  position projections, one subsampling shared by both encoder passes, and a cached RNNT joint. Decisions are
  unchanged and compute drops to 0.62x. Conformer convolutions run as unfold + linear (`fast_conv`).

## The server

```
 client (Pipecat / LiveKit adapter, audioforge-client, any WebSocket)
   │  config (JSON) · PCM (binary) · agent_end / enroll · end
   ▼
 connection loop            audioforge/serve.py: handle(), serve(), GET /health
   │  decode + resample, backpressure, idle / session limits, final-ASR delivery
   ▼
 Session (one per connection)   audioforge/serve.py: Session.process() -> messages
   │  ASRStream (encoder pass, RNNT, VAD, turn, speaker)   audioforge/server/streams.py
   │  speaker columns: TS-VAD track (single) or diarizer step + binding (room)   tsvad_stream.py, server/binding.py
   │  turn policies                                        audioforge/server/policies.py
   │  load shedding, NaN guards, segment cap               (research/BULLETPROOF.md)
   ▼
 Engine (shared)            audioforge/serve.py: models, the compute thread, Silero, the final-ASR worker process
   │
   ▼
 ready · frame (80 ms) · partial · turn_end · final · stats · error    audioforge/server/protocol.py (validate)
```

- **One compute thread** serves every session, so sessions never race on torch state. The asyncio loop only moves
  bytes. One session per 2 CPU threads is what one worker sustains in real time.
- **The final-ASR worker** (Parakeet-TDT v3) runs in a child process with its own torch threads, so it never blocks
  the streaming thread. It is restarted if it hangs or dies, and the streaming final stands in meanwhile.
- **Under load** the server sheds work in levels instead of falling behind: first the diarizer runs at half cadence
  and holds the last speaker, then partials are dropped. Every degradation is reported as a non-fatal `error`
  message and counted in `stats` and `/health`.
- **Configuration** is one flag table (`audioforge/server/cli.py`). The same table produces `--help`, the config-file
  keys and the reference in [`CONFIGURATION.md`](CONFIGURATION.md).
- **In-process use** (`audioforge.load()`, `audioforge-bench`) drives the same `Engine` / `Session` without the
  socket.

## Where things live

| concern | code |
|---|---|
| in-process API | `audioforge/api.py` |
| server core (engine, session, connection loop) | `audioforge/serve.py` |
| protocol, policies, binding, streams, flags | `audioforge/server/` |
| launcher and model download | `audioforge/launch.py`, `audioforge/hub.py` |
| framework adapters | `audioforge/integrations/pipecat.py`, `audioforge/integrations/livekit.py` |
| model, encoder, heads, losses | `audioforge/model.py`, `audioforge/modules/`, `audioforge/heads/`, `audioforge/losses/` |
| NVIDIA checkpoint import, diarizer | `audioforge/nemo_import.py`, `audioforge/streaming_diar.py` |
| voice print, speaker registry, enrollment | `audioforge/tsvad_stream.py`, `audioforge/speaker_registry.py`, `audioforge/enrollment.py` |
| training and research library | `audioforge/train.py`, `audioforge/datasets/`, `audioforge/baselines/`, `research/recipes/`, `scripts/research/` |
