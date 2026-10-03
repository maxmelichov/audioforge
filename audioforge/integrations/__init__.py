"""Framework adapters for the audioforge server (import the one you use; each needs its framework installed).

* ``audioforge.integrations.pipecat`` (``uv run --extra pipecat``): ``AudioforgeHub``,
  ``AudioforgeSTTService``, ``AudioforgeVADAnalyzer``, ``AudioforgeTurnAnalyzer``
* ``audioforge.integrations.livekit`` (``uv run --extra livekit``): ``AudioforgeFrontend`` (``.stt()``,
  ``.vad()``, ``.turn_detector()``), ``AudioforgeSTT``, ``AudioforgeVAD``, ``AudioforgeTurnDetector``

Both talk to a running ``audioforge-serve`` over its WebSocket protocol (docs/PROTOCOL.md); they log through their
framework's logger (loguru for Pipecat, ``livekit.plugins.audioforge`` for LiveKit).
"""
