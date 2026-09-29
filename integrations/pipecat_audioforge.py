"""Moved to ``audioforge.integrations.pipecat`` (2026-09-29); this name stays importable for existing code."""
import sys

from audioforge.integrations import pipecat as _module

sys.modules[__name__] = _module
