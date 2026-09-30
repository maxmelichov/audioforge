"""The parts ``audioforge.serve`` (the WebSocket server) is built from.

* ``constants``: the frame clock, policy constants, diarizer presets, robustness limits
* ``protocol``: message schema and ``validate``, the ``error`` message, ``SessionConfig``, PCM decoding
* ``policies``: end-of-turn policies (silence timeouts, head threshold, Silero silence)
* ``turn_hint``: the early ``turn_end_hint`` / ``turn_end_hint_cancel`` (tracker, ledger, offline replay)
* ``binding``: primary-speaker binding for ``--enroll``
* ``streams``: the ASR model's streaming passes, the CPU convolution fast path, the resampler
* ``cli``: the server's command-line flags
* ``util``: percentiles, memory, a bounded per-frame ring

The engine, the sessions and the connection loop are in ``audioforge/serve.py``, which re-exports all of these.
docs/SERVER_INTERNALS.md has the design notes.
"""
