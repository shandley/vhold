"""Disorder classifier model path management for vhold.

Handles locating and validating the trained disorder-aware MLP classifier
checkpoint used for functional category prediction of disordered proteins.
"""

from pathlib import Path

from vhold.utils.constants import DISORDER_CLASSIFIER_FILE, get_model_dir, get_vhold_zenodo_url
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# URL for the trained disorder classifier model (from Zenodo)
DISORDER_CLASSIFIER_URL = get_vhold_zenodo_url("disorder_classifier")


def get_disorder_classifier_path(model_dir: Path | None = None) -> Path:
    """Get path to the disorder classifier model checkpoint.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        Path to the .pt disorder classifier checkpoint file
    """
    if model_dir is None:
        model_dir = get_model_dir()
    return Path(model_dir) / "disorder_classifier" / DISORDER_CLASSIFIER_FILE


def check_disorder_classifier(model_dir: Path | None = None) -> bool:
    """Check if the disorder classifier model is installed.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        True if the disorder classifier checkpoint file exists
    """
    return get_disorder_classifier_path(model_dir).exists()


def install_disorder_classifier(model_dir: Path, force: bool = False) -> None:
    """Download and install the disorder-aware classifier model.

    Args:
        model_dir: Model directory
        force: Force re-download even if file exists
    """
    from vhold.databases.install import download_file

    classifier_dir = model_dir / "disorder_classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    dest = classifier_dir / DISORDER_CLASSIFIER_FILE

    if dest.exists() and not force:
        logger.info("Disorder classifier model already installed")
        return

    if DISORDER_CLASSIFIER_URL is None:
        logger.warning(
            "Disorder classifier model URL not yet configured. "
            "Use scripts/train_disorder_classifier.py to train a local model, "
            "then place it at: %s",
            dest,
        )
        return

    logger.info(f"Downloading disorder classifier model from {DISORDER_CLASSIFIER_URL}")
    download_file(DISORDER_CLASSIFIER_URL, dest, "Disorder classifier model")

    # Validate
    import torch

    try:
        checkpoint = torch.load(dest, map_location="cpu", weights_only=False)
        n_classes = checkpoint.get("num_classes", "unknown")
        input_dim = checkpoint.get("input_dim", "unknown")
        logger.info(
            f"Disorder classifier installed: {n_classes} classes, "
            f"{input_dim} input dim"
        )
    except Exception as e:
        logger.error(f"Invalid disorder classifier model file: {e}")
        if dest.exists():
            dest.unlink()
        raise
