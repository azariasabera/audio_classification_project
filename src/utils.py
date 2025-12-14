# src/utils.py

from omegaconf import DictConfig
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split

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

def split_data(X: pd.DataFrame, y: pd.Series, config: DictConfig) -> tuple:
    """
    Split the data into training and test sets based on the `test_proportion` in the config.

    Args:
        X: Feature matrix.
        y: Labels.
        config: Hydra configuration object containing `test_proportion`.

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    test_proportion = getattr(config, "test_proportion", 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, stratify=y, random_state=42)
    
    return X_train, X_test, y_train, y_test