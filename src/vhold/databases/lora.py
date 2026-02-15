"""LoRA adapter path management for vhold.

Handles locating and validating the contrastive LoRA adapter
for the ProstT5 encoder.
"""

from pathlib import Path

from vhold.utils.constants import CONTRASTIVE_LORA_DIR, get_model_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def get_lora_adapter_path(model_dir: Path | None = None) -> Path:
    """Get path to the LoRA adapter directory.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        Path to the adapter directory containing adapter_config.json
        and adapter_model.safetensors
    """
    if model_dir is None:
        model_dir = get_model_dir()
    return Path(model_dir) / CONTRASTIVE_LORA_DIR


def check_lora_adapter(model_dir: Path | None = None) -> bool:
    """Check if the LoRA adapter is installed.

    Args:
        model_dir: Model directory (default: ~/.vhold/models)

    Returns:
        True if the adapter config and weights exist
    """
    adapter_dir = get_lora_adapter_path(model_dir)
    return (adapter_dir / "adapter_config.json").exists()
