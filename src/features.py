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


def extract_stft(signals: List[np.ndarray], config: DictConfig) -> Dict[str, np.ndarray]:
    """
    Extract STFT spectrograms from audio waveforms.

    Args:
        signals: List of audio waveforms.
        config: Hydra config object (sr, win_length, hop_length).

    Returns:
        Dictionary with keys 'magnitude' and 'db', each a 3D numpy array (num_samples, freq_bins, time_frames).
    """
    sr = get_audio_param(config, "sr", 16000)
    win_length = get_audio_param(config, "win_length", 1024)
    hop_length = get_audio_param(config, "hop_length", 256)

    mag_specs, db_specs = [], []
    for sig in tqdm(signals, desc="STFT"):
        stft_mag = np.abs(librosa.stft(sig, n_fft=win_length, hop_length=hop_length))
        stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
        mag_specs.append(stft_mag)
        db_specs.append(stft_db)
    return {
        "magnitude": np.array(mag_specs, dtype=np.float32),
        "db": np.array(db_specs, dtype=np.float32)
    }