# src/evaluate.py

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Any, Dict, Union
from omegaconf import DictConfig


def evaluate_model(model: Any, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate a model on test data using common classification metrics.

    Args:
        model: Any fitted model implementing `predict`.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dict[str, float]: Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def save_results(df: Union[pd.DataFrame, Dict[str, float]], config: DictConfig, filename: str = "metrics.csv"):
    """
    Save evaluation results to a CSV file in the directory specified by `config.output_dir`.
f
    Args:
        df: Either a DataFrame or a dictionary of metrics.
        config: Hydra configuration object.
        filename: Name of the file to save metrics to (default is 'metrics.csv').

    """
    # if 'output/dir' doesn't exist default with /output in cwd
    output_dir = Path(getattr(config, "output", {}).get("dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    # Convert dict to DataFrame if given as dict
    if isinstance(df, dict):
        df = pd.DataFrame([df])

    df.to_csv(file_path, index=False)
    #print(f"Results saved to {file_path}")