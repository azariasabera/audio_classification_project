# src/features.py

import numpy as np
import librosa
from typing import List, Dict
from omegaconf import DictConfig
from src.utils import get_audio_param
from tqdm import tqdm

def extract_mfcc(signals: List[np.ndarray], config: DictConfig) -> np.ndarray:
    """
    Extract MFCC features from audio waveforms.

    Args:
        signals: List of audio waveforms.
        config: Hydra config object (sr, n_mfcc).

    Returns:
        3D numpy array of MFCC features (num_samples, n_mfcc, time_frames).
    """
    sr = get_audio_param(config, "sr", 16000)
    n_mfcc = get_audio_param(config, "n_mfcc", 20)

    features = []
    for sig in tqdm(signals, desc="MFCC"): #for sig in signals:
        mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)  # shape: (n_mfcc, time_frames)
        features.append(mfcc)

    return np.array(features, dtype=np.float32)
