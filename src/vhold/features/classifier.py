"""MLP classifier for viral protein functional categories.

Trained on frozen ProstT5 encoder embeddings (1024-dim) to predict
functional categories. Supplements keyword-based classification by
learning structural embedding patterns that correspond to function.

The classifier only overrides "unknown" proteins -- it never overrides
Pfam/SUPERFAMILY/GO evidence from the hierarchical classifier.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vhold.databases.classifier import get_classifier_model_path, check_classifier_model
from vhold.results.categories import ALL_CATEGORIES
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class FunctionalClassifier(nn.Module):
    """MLP classifier for viral protein functional categories.

    Architecture: 1024 -> 512 -> 256 -> num_classes
    with LayerNorm, ReLU, and Dropout between layers.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 11,
        hidden_dims: tuple[int, ...] = (512, 256),
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


def load_classifier(
    model_path: Path | None = None,
    model_dir: Path | None = None,
) -> FunctionalClassifier | None:
    """Load a trained classifier from checkpoint.

    Args:
        model_path: Direct path to .pt file. If None, uses default location.
        model_dir: Model directory (used if model_path is None)

    Returns:
        Loaded FunctionalClassifier, or None if checkpoint not found
    """
    if model_path is None:
        model_path = get_classifier_model_path(model_dir)

    if not model_path.exists():
        logger.debug(f"Classifier checkpoint not found at {model_path}")
        return None

    logger.info(f"Loading classifier from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    model = FunctionalClassifier(
        input_dim=checkpoint.get("input_dim", 1024),
        num_classes=checkpoint.get("num_classes", 11),
        hidden_dims=tuple(checkpoint.get("hidden_dims", [512, 256])),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        f"Classifier loaded: {checkpoint.get('model_type', 'mlp')}, "
        f"{checkpoint.get('num_classes', 11)} classes"
    )
    return model


def classify_with_model(
    embeddings: np.ndarray,
    protein_ids: list[str],
    classifier: FunctionalClassifier,
    confidence_threshold: float = 0.5,
    category_names: list[str] | None = None,
) -> dict[str, tuple[str, float]]:
    """Classify proteins using the trained MLP.

    Args:
        embeddings: Array of shape (N, 1024), L2-normalized
        protein_ids: List of N protein IDs
        classifier: Loaded FunctionalClassifier
        confidence_threshold: Minimum softmax probability to accept prediction
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
