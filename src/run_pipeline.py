# src/run_pipeline.py

from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig

from src.data_loader import load_dataset
from src.preprocessing import pad_or_trim, flatten_data, SqueezeType, SqueezeAxis
from src.features import extract_all_features
from src.models import instantiate_models
from src.train import train_model
from src.evaluate import evaluate_model, save_results
from src.utils import split_data, save_model

def run_pipeline(config: DictConfig):
    """
    Run the full audio classification pipeline:
        1. Load dataset
        2. Preprocess (pad/trim)
        3. Extract features (MFCC, STFT, Mel, CQT)
        4. Flatten features (all SqueezeType x SqueezeAxis combinations)
        5. Instantiate models
        6. Train/test split
        7. Train models
        8. Evaluate models
        9. Optionally save metrics and models
    
    Args:
        config: Hydra DictConfig object
    """
    print("=== Loading dataset ===")
    X, y, sr = load_dataset(config)
    print(f"Loaded {len(X)} samples with {len(set(y))} classes, sr={sr}")

    print("=== Padding/trimming signals ===")
    X_proc = [pad_or_trim(s, config) for s in tqdm(X, desc="Pad/Trim")]

    print("=== Extracting features ===")
    features_dict = extract_all_features(X_proc, config)
    
    # Flatten features combinations
    flattened_features = {}
    print("=== Flattening features ===")
    for feat_name, feat_array in tqdm(features_dict.items(), desc="Flatten"):
        # feat_array is (n_samples, freq_bins, time_frames)
        for squeeze_type in SqueezeType:
            if squeeze_type == SqueezeType.FLAT:
                # FLAT ignores axis
                X_flat = flatten_data(feat_array, squeeze_type=squeeze_type)
                key = f"{feat_name}_{squeeze_type.value}"
                flattened_features[key] = X_flat
            else:
                for axis in SqueezeAxis:
                    X_flat = flatten_data(feat_array, squeeze_type=squeeze_type, axis=axis)
                    key = f"{feat_name}_{squeeze_type.value}_{axis.name.lower()}"
                    flattened_features[key] = X_flat
    print(f"Generated {len(flattened_features)} flattened feature sets.")

    print("=== Instantiating models ===")
    models_dict = instantiate_models(config.models)
    print(f"Instantiated {len(models_dict)} models: {list(models_dict.keys())}")

    # Train/test split, training, evaluation
    results = []

    print("=== Training and evaluating models ===")
    for feat_name, X_flat in tqdm(flattened_features.items(), desc="Train/Evaluate"):
        # Convert y to pandas Series for stratify compatibility
        X_df = pd.DataFrame(X_flat)
        y_series = pd.Series(y)
        
        X_train, X_test, y_train, y_test = split_data(X_df, y_series, config)
        
        for model_name, model in models_dict.items():
            full_model_name = f"{model_name}_{feat_name}"
            # Train
            trained_model = train_model(model, X_train, y_train)
            # Evaluate
            metrics = evaluate_model(trained_model, X_test, y_test)
            metrics["model"] = full_model_name
            metrics["feature_set"] = feat_name
            results.append(metrics)
            
            # Save model if config says so
            if getattr(config, "save_model", False):
                save_model(trained_model, config, full_model_name)

    # Save metrics if config says so
    if getattr(config, "save_metrics", True):
        results_df = pd.DataFrame(results)
        save_results(results_df, config, filename="metrics.csv")

    print("=== Pipeline finished ===")
