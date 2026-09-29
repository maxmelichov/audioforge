"""Real-speech datasets in the list-of-dict format of ``audioforge.data``."""
from .librispeech import LibriSpeech, energy_vad, eou_targets, recipe_data

__all__ = ["LibriSpeech", "energy_vad", "eou_targets", "recipe_data"]
