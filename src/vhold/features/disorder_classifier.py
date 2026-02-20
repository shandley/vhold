"""Disorder-aware MLP classifier for viral protein functional categories.

Uses STARLING ensemble-aware embeddings (512-dim) or concatenated
ProstT5+STARLING embeddings (1536-dim) to classify disordered viral
proteins that structural homology methods cannot annotate.

Architecture: input_dim -> 256 -> 128 -> num_classes
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vhold.databases.disorder_classifier import get_disorder_classifier_path
from vhold.results.categories import ALL_CATEGORIES
from vhold.utils.constants import STARLING_EMBEDDING_DIM
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class DisorderClassifier(nn.Module):
    """MLP classifier for disordered viral proteins.

    Smaller architecture than FunctionalClassifier since STARLING
    embeddings are 512-dim (vs ProstT5 1024-dim) and the training
    set for disordered proteins is smaller.
    """

    def __init__(
        self,
        input_dim: int = STARLING_EMBEDDING_DIM,
        num_classes: int = 11,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)


def load_disorder_classifier(
    model_path: Path | None = None,
    model_dir: Path | None = None,
) -> DisorderClassifier | None:
    """Load a trained disorder classifier from checkpoint.

    Args:
        model_path: Direct path to .pt file. If None, uses default location.
        model_dir: Model directory (used if model_path is None)

    Returns:
        Loaded DisorderClassifier, or None if checkpoint not found
    """
    if model_path is None:
        model_path = get_disorder_classifier_path(model_dir)

    if not model_path.exists():
        logger.debug(f"Disorder classifier checkpoint not found at {model_path}")
        return None

    logger.info(f"Loading disorder classifier from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = DisorderClassifier(
        input_dim=checkpoint.get("input_dim", STARLING_EMBEDDING_DIM),
        num_classes=checkpoint.get("num_classes", 11),
        hidden_dims=tuple(checkpoint.get("hidden_dims", [256, 128])),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        f"Disorder classifier loaded: {checkpoint.get('input_dim')} input dim, "
        f"{checkpoint.get('num_classes', 11)} classes"
    )
    return model


def classify_disordered_proteins(
    embeddings: np.ndarray,
    protein_ids: list[str],
    classifier: DisorderClassifier,
    confidence_threshold: float = 0.5,
    category_names: list[str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Classify disordered proteins using STARLING embeddings.

    Args:
        embeddings: Array of shape (N, 512) or (N, 1536) L2-normalized
        protein_ids: List of N protein IDs
        classifier: Loaded DisorderClassifier
        confidence_threshold: Minimum softmax probability to accept
        category_names: Ordered category names (default: ALL_CATEGORIES)

    Returns:
        Dict mapping protein_id to (predicted_category, confidence).
        Only includes proteins above the confidence threshold whose
        predicted category is not "unknown".
    """
    if category_names is None:
        category_names = ALL_CATEGORIES

    classifier.eval()
    with torch.no_grad():
        x = torch.from_numpy(embeddings.astype(np.float32))
        logits = classifier(x)
        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)

    results = {}
    for i, protein_id in enumerate(protein_ids):
        pred_idx = predictions[i].item()
        conf = confidences[i].item()

        if pred_idx >= len(category_names):
            continue

        category = category_names[pred_idx]

        if conf >= confidence_threshold and category != "unknown":
            results[protein_id] = (category, conf)

    return results
