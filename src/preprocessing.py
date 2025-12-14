# src/preprocessing.py

import numpy as np
from omegaconf import DictConfig
from src.utils import get_audio_param
from enum import Enum
from typing import List

def pad_or_trim(signal: np.ndarray, config: DictConfig) -> np.ndarray:
    """
    Pad or trim an audio signal to a fixed duration using parameters from config.audio.

    Args:
        signal: 1D numpy array containing the audio waveform.
        config: Hydra DictConfig object containing audio parameters (`sr`, `duration`).

    Returns:
        1D numpy array of length `sr * duration`, either padded with zeros or trimmed.
    """
    sr = get_audio_param(config, "sr", 16000)
    duration = get_audio_param(config, "duration", 5.0)
    target = int(sr * duration)

    if len(signal) > target:
        return signal[:target]
    elif len(signal) < target:
        return np.pad(signal, (0, target - len(signal)))
    else:
        return signal