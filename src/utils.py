# src/utils.py

import joblib
from pathlib import Path
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

def save_model(model: Any, config: DictConfig, model_name: str) -> Path:
    """
    Save the trained model to a specified directory with the given extension.

    Args:
        model: The trained model to save.
        config: Hydra config object (for getting output directory).
        model_name: Name to use for the saved model file (without extension).

    Returns:
        Path: The full path to the saved model file.
    """
    # if 'output/dir' doesn't exist default with /output in cwd
    output_dir = Path(getattr(config, "output", {}).get("model_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # I don't use extension = getattr(config.output, "model_extension", ".joblib") to avoid potential error in config.ouput
    extension = getattr(getattr(config, "output", {}), "model_extension", ".joblib") # defaults to .joblib

    model_path = output_dir / (model_name + extension)

    joblib.dump(model, model_path)
    #print(f"Model saved to {model_path}")

    return model_path

def load_model(model_path: Path) -> Any:
    """
    Load a model from a specified file.

    Args:
        model_path: Path to the saved model file.

    Returns:
        The loaded model.
    """
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model