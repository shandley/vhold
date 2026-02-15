"""Classifier model path management for vhold.

Handles locating and validating the trained MLP classifier
checkpoint used for functional category prediction.
"""

from pathlib import Path

from vhold.utils.constants import CLASSIFIER_MODEL_FILE, get_model_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def get_classifier_model_path(model_dir: Path | None = None) -> Path:
    """Get path to the classifier model checkpoint.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        Path to the .pt classifier checkpoint file
    """
    if model_dir is None:
        model_dir = get_model_dir()
    return Path(model_dir) / "classifier" / CLASSIFIER_MODEL_FILE


def check_classifier_model(model_dir: Path | None = None) -> bool:
    """Check if the classifier model is installed.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        True if the classifier checkpoint file exists
    """
    return get_classifier_model_path(model_dir).exists()
