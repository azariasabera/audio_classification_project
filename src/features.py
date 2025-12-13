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


def extract_mel(signals: List[np.ndarray], config: DictConfig) -> np.ndarray:
    """
    Extract Mel spectrograms from audio waveforms.

    Args:
        signals: List of audio waveforms.
        config: Hydra config object (sr, n_mels, win_length, hop_length).

    Returns:
        3D numpy array of Mel spectrograms (num_samples, n_mels, time_frames).
    """
    sr = get_audio_param(config, "sr", 16000)
    n_mels = get_audio_param(config, "n_mels", 128)
    win_length = get_audio_param(config, "win_length", 1024)
    hop_length = get_audio_param(config, "hop_length", 256)

    mel_specs = []
    for sig in tqdm(signals, desc="MEL"):
        mel_spec = librosa.feature.melspectrogram(
            y=sig, sr=sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length
        )
        mel_specs.append(mel_spec)
    return np.array(mel_specs, dtype=np.float32)


def extract_cqt(signals: List[np.ndarray], config: DictConfig) -> np.ndarray:
    """
    Extract Constant-Q Transform (CQT) spectrograms from audio waveforms.

    Args:
        signals: List of audio waveforms.
        config: Hydra config object (sr, n_bins, hop_length).

    Returns:
        3D numpy array of CQT spectrograms (num_samples, freq_bins, time_frames).
    """
    sr = get_audio_param(config, "sr", 16000)
    n_bins = get_audio_param(config, "n_bins", 84)
    hop_length = get_audio_param(config, "hop_length", 256)

    cqt_specs = []
    for sig in tqdm(signals, desc="CQT"):
        cqt_spec = np.abs(librosa.cqt(sig, sr=sr, n_bins=n_bins, hop_length=hop_length))
        cqt_specs.append(cqt_spec)
    return np.array(cqt_specs, dtype=np.float32)


def extract_all_features(signals: List[np.ndarray], config: DictConfig) -> Dict[str, np.ndarray]:
    """
    Extract MFCC, STFT, Mel, and CQT features from audio waveforms.

    Args:
        signals: List of audio waveforms.
        config: Hydra config object.

    Returns:
        Dictionary with keys 'mfcc', 'stft_mag', 'stft_db', 'mel', 'cqt', each containing extracted features.
    """
    stft_feats = extract_stft(signals, config)
    return {
        "mfcc": extract_mfcc(signals, config),
        "stft_mag": stft_feats["magnitude"],
        "stft_db": stft_feats["db"],
        "mel": extract_mel(signals, config),
        "cqt": extract_cqt(signals, config)
    }