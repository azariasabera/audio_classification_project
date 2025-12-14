# src/utils.py

from omegaconf import DictConfig
from typing import Any

def get_audio_param(config: DictConfig, key: str, default: Any) -> Any:
    """
    Get a parameter from config.audio with fallback default.

    Args:
        config: Hydra DictConfig object.
        key: Parameter name in config.audio.
        default: Fallback value if key is missing.

    Returns:
        The value from config.audio[key] or default.
    """
    return getattr(getattr(config, "audio", {}), key, default)