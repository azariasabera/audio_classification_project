# src/train.py

import pandas as pd
from typing import Any

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a given model on the provided training data.

    Args:
        model: An instantiated machine learning model
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The trained model.
    """
    model.fit(X_train, y_train)
    
    return model
