# src/models.py

from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Dict

def instantiate_models(config: DictConfig) -> Dict[str, object]:
    """
    Instantiate models based on configurations in the `models` section of the config.

    Args:
        config: The Hydra DictConfig object containing the model configurations.

    Returns:
        dict: A dictionary where keys are model names (e.g., 'svc', 'rf') and 
            values are the instantiated model objects.
    """
    models = {}

    model_cfgs = config.models

    for model_name, model_cfg in model_cfgs.items():
        try:
            # Instantiate the model using the target specified in the config
            model = instantiate(model_cfg)
            models[model_name] = model
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model {model_name}") from e

    return models
