# src/data_loader.py

from typing import Tuple, List, Iterator
import librosa
from pathlib import Path
from omegaconf import DictConfig
from src.utils import get_audio_param
from tqdm import tqdm

def get_classes(input_dir: Path) -> List[str]:
    """
    Discover class labels from subdirectories.

    Args:
        input_dir (Path): Root dataset directory where each subfolder is a class.

    Returns:
        List[str]: Sorted list of class names.
    """
    classes = [p.name for p in input_dir.iterdir() if p.is_dir()] # list comprehension 
    if not classes:
        raise ValueError(f"No class subfolders found in {input_dir}")
    return sorted(classes)


def iter_audio_files(
    class_dir: Path,
    extensions: Tuple[str, ...],
) -> Iterator[Path]:
    """
    Yield audio file paths for a single class directory.

    Args:
        class_dir (Path): Directory corresponding to a single class.
        extensions (Tuple[str, ...]): Allowed audio file extensions (e.g., (".wav", ".mp3")).

    Yields:
        Path: Path to an audio file matching the given extensions.
    """
    for ext in extensions:
        yield from class_dir.rglob(f"*{ext}")  # e.g. pattern:  rglob("*.wav")


def load_dataset(config: DictConfig) -> Tuple[List, List, int]:
    """
    Load audio files from a dataset directory structured by class subfolders.

    Args:
        config (DictConfig): Hydra config containing `input_dir`, `sr`, `extensions`...

    Returns:
        Tuple:
            - X (List[np.ndarray]): List of loaded audio signals.
            - y (List[str]): Corresponding class labels.
            - sr (int): Sampling rate used for loading.
    """
    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    sr = get_audio_param(config, "sr", 16000)
    extensions = get_audio_param(config, "extensions", (".wav",))
    extensions = tuple(ext.lower() for ext in extensions)

    X, y = [], []

    classes = get_classes(input_dir)

    for clss in tqdm(classes, desc="Load"):
        class_dir = input_dir / clss

        for path in iter_audio_files(class_dir, extensions):
            try:
                y_raw, _ = librosa.load(path, sr=sr)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
                continue

            X.append(y_raw)
            y.append(clss)

    return X, y, sr