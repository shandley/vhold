"""Classifier model path management for vhold.

Handles locating and validating the trained MLP classifier
checkpoint used for functional category prediction.
"""

from pathlib import Path

from vhold.utils.constants import CLASSIFIER_MODEL_FILE, get_model_dir, get_vhold_zenodo_url
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# URL for the trained classifier model (from Zenodo)
CLASSIFIER_MODEL_URL = get_vhold_zenodo_url("classifier")


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


def install_classifier_model(model_dir: Path, force: bool = False) -> None:
    """Download and install the MLP classifier model.

    Args:
        model_dir: Model directory
        force: Force re-download even if file exists
    """
    from vhold.databases.install import download_file

    classifier_dir = model_dir / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    dest = classifier_dir / CLASSIFIER_MODEL_FILE

    if dest.exists() and not force:
        logger.info("Classifier model already installed")
        return

    if CLASSIFIER_MODEL_URL is None:
        logger.warning(
            "Classifier model URL not yet configured. "
            "Use scripts/train_classifier.py to train a local model, "
            "then place it at: %s",
            dest,
        )
        return

    logger.info(f"Downloading classifier model from {CLASSIFIER_MODEL_URL}")
    download_file(CLASSIFIER_MODEL_URL, dest, "Classifier model")

    # Validate
    import torch

    try:
        checkpoint = torch.load(dest, map_location="cpu", weights_only=True)
        n_classes = checkpoint.get("n_classes", "unknown")
        logger.info(f"Classifier model installed: {n_classes} classes")
    except Exception as e:
        logger.error(f"Invalid classifier model file: {e}")
        if dest.exists():
            dest.unlink()
        raise
