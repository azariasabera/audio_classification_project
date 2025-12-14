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


class SqueezeType(Enum):
    """
    Enum representing different ways to flatten or squeeze time-frequency features
    (e.g., spectrograms, MFCCs) into 2D arrays for ML models.

    Options:
        FLAT: Flatten frequency x time matrix into a 1D vector.
        AVG: Average over the time axis.
        MAX: Take the maximum over the time axis.
        MEAN_STD: Concatenate mean and standard deviation over the time axis.
    """
    FLAT = "flat"
    AVG = "avg"
    MAX = "max"
    MEAN_STD = "mean_std"


class SqueezeAxis(Enum):
    """
    Enum representing which axis to squeeze over.
    """
    TIME = 1        # axis 1 of each sample (columns)
    FREQUENCY = 0   # axis 0 of each sample (rows)

def flatten_data(
    X: List[np.ndarray], 
    squeeze_type: SqueezeType = SqueezeType.FLAT, 
    axis: SqueezeAxis = SqueezeAxis.TIME
) -> np.ndarray:
    """
    Flatten or squeeze a list of 2D time-frequency feature arrays into 2D arrays
    (samples x features) based on the selected method and axis.

    Args:
        X: List of feature arrays, each with shape (frequency_bins, time_frames).
        squeeze_type: Method to apply for flattening/squeezing. One of `SqueezeType`.
        axis: Axis to apply squeeze over. One of `SqueezeAxis`.

    Returns:
        2D numpy array of shape (num_samples, num_features), ready for ML models.

    Raises:
        ValueError: If an unknown squeeze_type is provided.
    """
    # Flattening is independent of SqueezeAxis
    if squeeze_type == SqueezeType.FLAT:
        return np.array([x.reshape(-1) for x in X])

    elif squeeze_type == SqueezeType.AVG:
        return np.array([np.mean(x, axis=axis.value) for x in X])

    elif squeeze_type == SqueezeType.MAX:
        return np.array([np.max(x, axis=axis.value) for x in X])

    elif squeeze_type == SqueezeType.MEAN_STD:
        return np.array([np.concatenate([np.mean(x, axis=axis.value), np.std(x, axis=axis.value)]) for x in X])

    else:
        raise ValueError(f"Unknown squeeze_type: {squeeze_type}. Choose from {list(SqueezeType)}")