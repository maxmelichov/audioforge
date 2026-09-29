"""Where audioforge looks for datasets and downloaded checkpoints.

``AUDIOFORGE_DATA`` (default ``<repo>/data``) is the data root: datasets (``data/ami``, ``data/librispeech``, ...),
NVIDIA ``.nemo`` files (``data/nemo``) and Silero (``data/silero``). Point it at external storage to keep the
checkout small; nothing is copied or resolved through symlinks. ``AUDIOFORGE_HOME`` is where ``audioforge-download``
writes the served models (see ``audioforge.hub.models_dir``).
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["DATA_ROOT", "ROOT"]

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("AUDIOFORGE_DATA") or ROOT / "data").expanduser()
