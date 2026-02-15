"""Tests for MLP functional classifier."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vhold.features.classifier import (
    FunctionalClassifier,
    classify_with_model,
    load_classifier,
)
from vhold.results.categories import ALL_CATEGORIES


# ============================================================================
# Helpers
# ============================================================================


def _make_normalized(shape, rng=None):
    """Create L2-normalized random vectors."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = rng.standard_normal(shape).astype(np.float32)
    if arr.ndim == 1:
        return arr / np.linalg.norm(arr)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return arr / norms


def _save_checkpoint(path, model, **extra):
    """Save a classifier checkpoint in the expected format."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "mlp",
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout_rate,
        "category_names": ALL_CATEGORIES,
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


# ============================================================================
# TestFunctionalClassifier
# ============================================================================


class TestFunctionalClassifier:
    """Tests for the MLP model architecture."""

    def test_forward_shape(self):
        """(B, 1024) input -> (B, 11) output."""
        model = FunctionalClassifier(input_dim=1024, num_classes=11)
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 11)

    def test_forward_shape_custom_dims(self):
        """Custom hidden dims produce correct output."""
        model = FunctionalClassifier(
            input_dim=512, num_classes=5, hidden_dims=(256, 128)
        )
        x = torch.randn(4, 512)
        out = model(x)
        assert out.shape == (4, 5)

    def test_forward_single_sample(self):
        """Single sample (1, 1024) works."""
        model = FunctionalClassifier()
        x = torch.randn(1, 1024)
        out = model(x)
        assert out.shape == (1, 11)

    def test_softmax_sums_to_one(self):
        """Softmax of logits sums to ~1.0 for each sample."""
        model = FunctionalClassifier()
        model.eval()
        x = torch.randn(5, 1024)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_eval_mode_deterministic(self):
        """In eval mode, same input gives same output."""
        model = FunctionalClassifier()
        model.eval()
        x = torch.randn(3, 1024)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ============================================================================
# TestClassifyWithModel
# ============================================================================


class TestClassifyWithModel:
    """Tests for batch inference function."""

    def test_returns_valid_categories(self):
        """All predicted categories are in ALL_CATEGORIES."""
        model = FunctionalClassifier()
        model.eval()
        embeddings = _make_normalized((10, 1024))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for pid, (category, conf) in results.items():
            assert category in ALL_CATEGORIES, f"Invalid category: {category}"

    def test_confidence_range(self):
        """All confidence values are in [0, 1]."""
        model = FunctionalClassifier()
        model.eval()
        embeddings = _make_normalized((20, 1024))
        protein_ids = [f"prot_{i}" for i in range(20)]

        results = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for pid, (category, conf) in results.items():
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_threshold_filtering(self):
        """High threshold filters out low-confidence predictions."""
        model = FunctionalClassifier()
        model.eval()
        embeddings = _make_normalized((10, 1024))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results_low = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )
        results_high = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.99
        )

        assert len(results_high) <= len(results_low)

    def test_unknown_excluded(self):
        """Predictions of 'unknown' category are excluded from results."""
        model = FunctionalClassifier()
        model.eval()
        embeddings = _make_normalized((10, 1024))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for pid, (category, conf) in results.items():
            assert category != "unknown", "unknown should be excluded from results"

    def test_empty_input(self):
        """Empty input returns empty dict."""
        model = FunctionalClassifier()
        model.eval()
        embeddings = np.zeros((0, 1024), dtype=np.float32)
        protein_ids = []

        results = classify_with_model(
            embeddings, protein_ids, model, confidence_threshold=0.5
        )

        assert results == {}

    def test_custom_category_names(self):
        """Custom category names are used for output."""
        model = FunctionalClassifier(num_classes=3)
        model.eval()
        embeddings = _make_normalized((5, 1024))
        protein_ids = [f"prot_{i}" for i in range(5)]
        categories = ["alpha", "beta", "gamma"]

        results = classify_with_model(
            embeddings, protein_ids, model,
            confidence_threshold=0.0,
            category_names=categories,
        )

        for pid, (category, conf) in results.items():
            assert category in categories


# ============================================================================
# TestLoadSaveRoundtrip
# ============================================================================


class TestLoadSaveRoundtrip:
    """Tests for checkpoint save/load."""

    def test_save_and_reload(self, tmp_path):
        """Saved checkpoint can be reloaded with same architecture."""
        model = FunctionalClassifier(
            input_dim=1024, num_classes=11,
            hidden_dims=(512, 256), dropout=0.3,
        )
        model.eval()

        checkpoint_path = tmp_path / "test_classifier.pt"
        _save_checkpoint(checkpoint_path, model)

        loaded = load_classifier(model_path=checkpoint_path)
        assert loaded is not None
        loaded.eval()

        # Same input should produce same output
        x = torch.randn(3, 1024)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded(x)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_load_preserves_architecture(self, tmp_path):
        """Loaded model has correct architecture params."""
        model = FunctionalClassifier(
            input_dim=1024, num_classes=7,
            hidden_dims=(256, 128), dropout=0.2,
        )
        checkpoint_path = tmp_path / "test_classifier.pt"
        _save_checkpoint(checkpoint_path, model)

        loaded = load_classifier(model_path=checkpoint_path)
        assert loaded.input_dim == 1024
        assert loaded.num_classes == 7
        assert loaded.hidden_dims == (256, 128)
        assert loaded.dropout_rate == 0.2

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading from non-existent path returns None."""
        result = load_classifier(model_path=tmp_path / "nonexistent.pt")
        assert result is None

    def test_load_from_model_dir(self, tmp_path):
        """load_classifier with model_dir looks in correct subdirectory."""
        model = FunctionalClassifier()
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "vhold_classifier.pt"
        _save_checkpoint(checkpoint_path, model)

        loaded = load_classifier(model_dir=tmp_path)
        assert loaded is not None


# ============================================================================
# TestCheckClassifierModel
# ============================================================================


class TestCheckClassifierModel:
    """Tests for classifier path management."""

    def test_check_returns_false_when_missing(self, tmp_path):
        """check_classifier_model returns False when no checkpoint exists."""
        from vhold.databases.classifier import check_classifier_model
        assert check_classifier_model(model_dir=tmp_path) is False

    def test_check_returns_true_when_present(self, tmp_path):
        """check_classifier_model returns True when checkpoint exists."""
        from vhold.databases.classifier import check_classifier_model

        model = FunctionalClassifier()
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        _save_checkpoint(checkpoint_dir / "vhold_classifier.pt", model)

        assert check_classifier_model(model_dir=tmp_path) is True

    def test_get_classifier_model_path(self, tmp_path):
        """Path includes 'classifier' subdirectory and correct filename."""
        from vhold.databases.classifier import get_classifier_model_path

        path = get_classifier_model_path(model_dir=tmp_path)
        assert path.parent.name == "classifier"
        assert path.name == "vhold_classifier.pt"
