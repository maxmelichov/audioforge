"""audioforge: a streaming speech front end for voice agents (ASR, VAD, speaker activity and end-of-turn from one
frozen NVIDIA FastConformer encoder with small heads), in plain PyTorch.

    import audioforge
    fe = audioforge.load()                      # models from audioforge-download
    s = fe.session(turn_policy="timeout")
    events = s.feed(pcm_16k) + s.end()          # partial / turn_end / final / stats dicts (docs/PROTOCOL.md)

Entry points: ``audioforge-download`` (models), ``audioforge-serve`` (WebSocket server), ``audioforge-bench``
(in-process speed), ``audioforge`` (training / catalog CLI).

Logging: every module logs under the ``audioforge`` logger, which prints plain messages to stdout at INFO by default
(the server's console log). ``logging.getLogger("audioforge").setLevel(logging.WARNING)`` quiets it; to route it
through your own handlers instead, call ``audioforge.use_app_logging()``.
"""
import logging
import sys

from .api import Frontend, Session, load, voiceprint

__version__ = "0.1.0"
__all__ = ["Frontend", "Session", "__version__", "load", "use_app_logging", "voiceprint"]


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stdout`` is at emit time (so redirection and test capture see the messages)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


_logger = logging.getLogger(__name__)
if not any(isinstance(h, _StdoutHandler) for h in _logger.handlers):
    _h = _StdoutHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def use_app_logging() -> None:
    """Drop audioforge's own stdout handler and let its records propagate to the application's logging setup."""
    for h in list(_logger.handlers):
        if isinstance(h, _StdoutHandler):
            _logger.removeHandler(h)
    _logger.propagate = True
    _logger.setLevel(logging.NOTSET)
