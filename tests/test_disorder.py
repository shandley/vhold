"""Tests for intrinsic disorder prediction and STARLING embeddings."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vhold.features.disorder import (
    DisorderResult,
    _extract_regions,
    _prepare_sequences_for_starling,
    analyze_disorder,
    check_metapredict_available,
    check_starling_available,
    classify_disorder,
    predict_disorder_batch,
    predict_disorder_single,
)
from vhold.features.disorder_classifier import (
    DisorderClassifier,
    classify_disordered_proteins,
    load_disorder_classifier,
)
from vhold.results.categories import ALL_CATEGORIES
from vhold.utils.constants import STARLING_EMBEDDING_DIM, STARLING_MAX_SEQUENCE_LENGTH

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


def _make_disorder_result(
    protein_id: str = "test",
    length: int = 100,
    fraction: float = 0.5,
    regions: list | None = None,
) -> DisorderResult:
    """Create a DisorderResult for testing."""
    scores = np.random.default_rng(42).random(length).astype(np.float32)
    return DisorderResult(
        protein_id=protein_id,
        disorder_scores=scores,
        disorder_fraction=fraction,
        disorder_regions=regions or [(10, 60)],
        disorder_class=classify_disorder(fraction),
        sequence_length=length,
    )


def _save_disorder_checkpoint(path, model, **extra):
    """Save a disorder classifier checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout_rate,
    }
    checkpoint.update(extra)
    torch.save(checkpoint, path)


# ============================================================================
# TestDisorderClassification
# ============================================================================


class TestDisorderClassification:
    """Tests for disorder fraction classification."""

    def test_highly_disordered(self):
        """Fraction >= 0.4 is highly disordered."""
        assert classify_disorder(0.4) == "highly_disordered"
        assert classify_disorder(0.8) == "highly_disordered"
        assert classify_disorder(1.0) == "highly_disordered"

    def test_partially_disordered(self):
        """Fraction >= 0.15 and < 0.4 is partially disordered."""
        assert classify_disorder(0.15) == "partially_disordered"
        assert classify_disorder(0.25) == "partially_disordered"
        assert classify_disorder(0.39) == "partially_disordered"

    def test_ordered(self):
        """Fraction < 0.15 is ordered."""
        assert classify_disorder(0.0) == "ordered"
        assert classify_disorder(0.1) == "ordered"
        assert classify_disorder(0.14) == "ordered"


# ============================================================================
# TestExtractRegions
# ============================================================================


class TestExtractRegions:
    """Tests for disordered region extraction."""

    def test_single_region(self):
        """Single contiguous disordered region."""
        scores = np.array([0.1, 0.1, 0.8, 0.9, 0.7, 0.8, 0.6, 0.1, 0.1, 0.1])
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert regions == [(2, 7)]

    def test_multiple_regions(self):
        """Two separate disordered regions."""
        scores = np.array(
            [0.8, 0.9, 0.7, 0.8, 0.6, 0.1, 0.1, 0.8, 0.9, 0.7, 0.8, 0.6]
        )
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert len(regions) == 2
        assert regions[0] == (0, 5)
        assert regions[1] == (7, 12)

    def test_min_length_filtering(self):
        """Short regions below min_length are excluded."""
        scores = np.array([0.1, 0.8, 0.9, 0.1, 0.1])
        regions = _extract_regions(scores, threshold=0.5, min_length=5)
        assert regions == []

    def test_region_at_end(self):
        """Region extending to end of sequence."""
        scores = np.array([0.1, 0.1, 0.1, 0.8, 0.9, 0.7, 0.8, 0.6])
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert regions == [(3, 8)]

    def test_no_disorder(self):
        """Fully ordered protein has no regions."""
        scores = np.array([0.1, 0.2, 0.1, 0.3, 0.1])
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert regions == []

    def test_fully_disordered(self):
        """Fully disordered protein is one region."""
        scores = np.array([0.8, 0.9, 0.7, 0.8, 0.9])
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert regions == [(0, 5)]

    def test_empty_scores(self):
        """Empty scores array returns no regions."""
        scores = np.array([])
        regions = _extract_regions(scores, threshold=0.5, min_length=3)
        assert regions == []


# ============================================================================
# TestPredictDisorderSingle
# ============================================================================


def _mock_metapredict():
    """Create a mock metapredict module."""
    mock = MagicMock()
    return mock


class TestPredictDisorderSingle:
    """Tests for single-protein disorder prediction."""

    def test_basic_prediction(self):
        """Basic single-protein prediction returns valid DisorderResult."""
        mock_meta = _mock_metapredict()
        mock_meta.predict_disorder.return_value = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])

        with patch.dict("sys.modules", {"metapredict": mock_meta}):
            result = predict_disorder_single("ACDEFG", protein_id="test1")

        assert isinstance(result, DisorderResult)
        assert result.protein_id == "test1"
        assert result.sequence_length == 6
        assert 0.0 <= result.disorder_fraction <= 1.0
        assert result.disorder_class in ("ordered", "partially_disordered", "highly_disordered")

    def test_fraction_calculation(self):
        """Disorder fraction is correctly computed from scores."""
        mock_meta = _mock_metapredict()
        # 3 out of 6 residues above 0.5 threshold
        mock_meta.predict_disorder.return_value = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])

        with patch.dict("sys.modules", {"metapredict": mock_meta}):
            result = predict_disorder_single("ACDEFG", threshold=0.5)

        assert result.disorder_fraction == pytest.approx(0.5, abs=1e-6)

    def test_custom_threshold(self):
        """Custom threshold changes classification."""
        mock_meta = _mock_metapredict()
        # With threshold 0.85, only 1/6 above
        mock_meta.predict_disorder.return_value = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])

        with patch.dict("sys.modules", {"metapredict": mock_meta}):
            result = predict_disorder_single("ACDEFG", threshold=0.85)

        # 1/6 ~ 0.167 → partially_disordered
        assert result.disorder_fraction == pytest.approx(1 / 6, abs=1e-6)


# ============================================================================
# TestPredictDisorderBatch
# ============================================================================


class TestPredictDisorderBatch:
    """Tests for batch disorder prediction."""

    def test_batch_prediction(self):
        """Batch prediction returns results for all proteins."""
        mock_meta = _mock_metapredict()
        sequences = {"prot1": "ACDEFG", "prot2": "MKLVWXY"}
        mock_meta.predict_disorder.return_value = {
            "prot1": ["ACDEFG", np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])],
            "prot2": ["MKLVWXY", np.array([0.8, 0.9, 0.7, 0.8, 0.6, 0.4, 0.3])],
        }

        with patch.dict("sys.modules", {"metapredict": mock_meta}):
            results = predict_disorder_batch(sequences)

        assert len(results) == 2
        assert "prot1" in results
        assert "prot2" in results
        assert isinstance(results["prot1"], DisorderResult)
        assert isinstance(results["prot2"], DisorderResult)

    def test_batch_empty_input(self):
        """Empty input returns empty dict without calling metapredict."""
        results = predict_disorder_batch({})
        assert results == {}

    def test_batch_extracts_correct_scores(self):
        """Batch correctly extracts scores from metapredict dict format."""
        mock_meta = _mock_metapredict()
        sequences = {"prot1": "ACDEFG"}
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3])
        mock_meta.predict_disorder.return_value = {
            "prot1": ["ACDEFG", scores],
        }

        with patch.dict("sys.modules", {"metapredict": mock_meta}):
            results = predict_disorder_batch(sequences)

        np.testing.assert_array_almost_equal(
            results["prot1"].disorder_scores, scores
        )


# ============================================================================
# TestAnalyzeDisorder
# ============================================================================


class TestAnalyzeDisorder:
    """Tests for disorder report generation."""

    def test_empty_results(self):
        """Empty results produce empty report."""
        report = analyze_disorder({})
        assert report.total_proteins == 0
        assert report.mean_disorder_fraction == 0.0

    def test_report_counts(self):
        """Report correctly counts disorder classes."""
        results = {
            "p1": _make_disorder_result("p1", fraction=0.5),   # highly
            "p2": _make_disorder_result("p2", fraction=0.25),   # partially
            "p3": _make_disorder_result("p3", fraction=0.05),   # ordered
            "p4": _make_disorder_result("p4", fraction=0.45),   # highly
        }
        report = analyze_disorder(results)

        assert report.total_proteins == 4
        assert report.highly_disordered == 2
        assert report.partially_disordered == 1
        assert report.ordered == 1

    def test_mean_fraction(self):
        """Mean disorder fraction is computed correctly."""
        results = {
            "p1": _make_disorder_result("p1", fraction=0.2),
            "p2": _make_disorder_result("p2", fraction=0.8),
        }
        report = analyze_disorder(results)
        assert report.mean_disorder_fraction == pytest.approx(0.5, abs=1e-6)


# ============================================================================
# TestPrepareSequencesForStarling
# ============================================================================


class TestPrepareSequencesForStarling:
    """Tests for STARLING sequence preparation with 380-residue limit."""

    def test_short_sequences_unchanged(self):
        """Sequences under 380 residues are not modified."""
        sequences = {"p1": "A" * 100, "p2": "M" * 380}
        prepared = _prepare_sequences_for_starling(sequences)
        assert prepared["p1"] == "A" * 100
        assert prepared["p2"] == "M" * 380

    def test_long_sequence_truncated(self):
        """Sequences over 380 residues are truncated."""
        sequences = {"p1": "A" * 500}
        prepared = _prepare_sequences_for_starling(sequences)
        assert len(prepared["p1"]) == STARLING_MAX_SEQUENCE_LENGTH

    def test_long_sequence_uses_disorder_region(self):
        """Long proteins use longest disordered region if available."""
        sequences = {"p1": "A" * 200 + "M" * 100 + "G" * 200}
        disorder = {
            "p1": _make_disorder_result(
                "p1", length=500, regions=[(200, 300)]  # 100-aa disordered region
            ),
        }
        prepared = _prepare_sequences_for_starling(sequences, disorder)
        # Should use the disordered region (positions 200-300)
        assert prepared["p1"] == "M" * 100

    def test_long_region_truncated_to_max(self):
        """Disordered regions exceeding max length are truncated."""
        sequences = {"p1": "A" * 600}
        disorder = {
            "p1": _make_disorder_result(
                "p1", length=600, regions=[(0, 500)]  # 500-aa region > 380
            ),
        }
        prepared = _prepare_sequences_for_starling(sequences, disorder)
        assert len(prepared["p1"]) == STARLING_MAX_SEQUENCE_LENGTH

    def test_fallback_without_disorder_data(self):
        """Without disorder data, truncates to first 380 residues."""
        sequences = {"p1": "A" * 500}
        prepared = _prepare_sequences_for_starling(sequences, disorder_results=None)
        assert len(prepared["p1"]) == STARLING_MAX_SEQUENCE_LENGTH
        assert prepared["p1"] == "A" * STARLING_MAX_SEQUENCE_LENGTH


# ============================================================================
# TestExtractStarlingEmbeddings
# ============================================================================


class TestExtractStarlingEmbeddings:
    """Tests for STARLING embedding extraction."""

    def _mock_starling(self):
        """Create a mock starling module with sequence_encoder."""
        mock_module = MagicMock()
        return mock_module

    @patch.dict("sys.modules", {"starling": MagicMock()})
    def test_embedding_shape(self):
        """Embeddings have shape (N, 512) and are L2-normalized."""
        import sys
        mock_encoder = sys.modules["starling"].sequence_encoder
        mock_encoder.return_value = {
            "p1": torch.randn(STARLING_EMBEDDING_DIM),
            "p2": torch.randn(STARLING_EMBEDDING_DIM),
        }

        from vhold.features.disorder import extract_starling_embeddings
        sequences = {"p1": "ACDEFGH", "p2": "MKLVWXY"}

        ids, embeddings = extract_starling_embeddings(sequences)

        assert len(ids) == 2
        assert embeddings.shape == (2, STARLING_EMBEDDING_DIM)

    @patch.dict("sys.modules", {"starling": MagicMock()})
    def test_l2_normalized(self):
        """Output embeddings are L2-normalized."""
        import sys
        mock_encoder = sys.modules["starling"].sequence_encoder
        mock_encoder.return_value = {
            "p1": torch.randn(STARLING_EMBEDDING_DIM) * 5,
        }

        from vhold.features.disorder import extract_starling_embeddings
        sequences = {"p1": "ACDEFGH"}

        ids, embeddings = extract_starling_embeddings(sequences)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    @patch.dict("sys.modules", {"starling": MagicMock()})
    def test_passes_kwargs_to_encoder(self):
        """Correct kwargs are passed to STARLING encoder."""
        import sys
        mock_encoder = sys.modules["starling"].sequence_encoder
        mock_encoder.return_value = {
            "p1": torch.randn(STARLING_EMBEDDING_DIM),
        }

        from vhold.features.disorder import extract_starling_embeddings
        sequences = {"p1": "ACDEFGH"}

        extract_starling_embeddings(
            sequences, ionic_strength=300, batch_size=16, device="cpu"
        )

        call_kwargs = mock_encoder.call_args[1]
        assert call_kwargs["ionic_strength"] == 300
        assert call_kwargs["batch_size"] == 16
        assert call_kwargs["device"] == "cpu"


# ============================================================================
# TestCheckAvailability
# ============================================================================


class TestCheckAvailability:
    """Tests for package availability checks."""

    @patch.dict("sys.modules", {"metapredict": MagicMock()})
    def test_metapredict_available(self):
        """Returns True when metapredict is importable."""
        assert check_metapredict_available() is True

    @patch.dict("sys.modules", {"metapredict": None})
    def test_metapredict_unavailable(self):
        """Returns False when metapredict is not importable."""
        assert check_metapredict_available() is False

    @patch.dict("sys.modules", {"starling": MagicMock()})
    def test_starling_available(self):
        """Returns True when starling is importable."""
        assert check_starling_available() is True

    @patch.dict("sys.modules", {"starling": None})
    def test_starling_unavailable(self):
        """Returns False when starling is not importable."""
        assert check_starling_available() is False


# ============================================================================
# TestDisorderClassifierModel
# ============================================================================


class TestDisorderClassifierModel:
    """Tests for the DisorderClassifier MLP architecture."""

    def test_forward_shape(self):
        """(B, 512) input -> (B, 11) output."""
        model = DisorderClassifier(input_dim=512, num_classes=11)
        x = torch.randn(8, 512)
        out = model(x)
        assert out.shape == (8, 11)

    def test_forward_shape_custom_dims(self):
        """Custom hidden dims produce correct output."""
        model = DisorderClassifier(
            input_dim=256, num_classes=5, hidden_dims=(128, 64)
        )
        x = torch.randn(4, 256)
        out = model(x)
        assert out.shape == (4, 5)

    def test_forward_single_sample(self):
        """Single sample (1, 512) works."""
        model = DisorderClassifier()
        x = torch.randn(1, STARLING_EMBEDDING_DIM)
        out = model(x)
        assert out.shape == (1, 11)

    def test_softmax_sums_to_one(self):
        """Softmax of logits sums to ~1.0."""
        model = DisorderClassifier()
        model.eval()
        x = torch.randn(5, STARLING_EMBEDDING_DIM)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_eval_mode_deterministic(self):
        """In eval mode, same input gives same output."""
        model = DisorderClassifier()
        model.eval()
        x = torch.randn(3, STARLING_EMBEDDING_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_1536_input_dim(self):
        """Supports 1536-dim concatenated ProstT5+STARLING embeddings."""
        model = DisorderClassifier(input_dim=1536, num_classes=11)
        x = torch.randn(4, 1536)
        out = model(x)
        assert out.shape == (4, 11)


# ============================================================================
# TestClassifyDisorderedProteins
# ============================================================================


class TestClassifyDisorderedProteins:
    """Tests for batch disorder classifier inference."""

    def test_returns_valid_categories(self):
        """All predictions are valid categories."""
        model = DisorderClassifier()
        model.eval()
        embeddings = _make_normalized((10, STARLING_EMBEDDING_DIM))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results = classify_disordered_proteins(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for _pid, (category, _conf) in results.items():
            assert category in ALL_CATEGORIES

    def test_confidence_range(self):
        """Confidence values are in [0, 1]."""
        model = DisorderClassifier()
        model.eval()
        embeddings = _make_normalized((20, STARLING_EMBEDDING_DIM))
        protein_ids = [f"prot_{i}" for i in range(20)]

        results = classify_disordered_proteins(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for _pid, (_category, conf) in results.items():
            assert 0.0 <= conf <= 1.0

    def test_threshold_filtering(self):
        """High threshold filters out low-confidence predictions."""
        model = DisorderClassifier()
        model.eval()
        embeddings = _make_normalized((10, STARLING_EMBEDDING_DIM))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results_low = classify_disordered_proteins(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )
        results_high = classify_disordered_proteins(
            embeddings, protein_ids, model, confidence_threshold=0.99
        )

        assert len(results_high) <= len(results_low)

    def test_unknown_excluded(self):
        """Predictions of 'unknown' are excluded."""
        model = DisorderClassifier()
        model.eval()
        embeddings = _make_normalized((10, STARLING_EMBEDDING_DIM))
        protein_ids = [f"prot_{i}" for i in range(10)]

        results = classify_disordered_proteins(
            embeddings, protein_ids, model, confidence_threshold=0.0
        )

        for _pid, (category, _conf) in results.items():
            assert category != "unknown"

    def test_empty_input(self):
        """Empty input returns empty dict."""
        model = DisorderClassifier()
        model.eval()
        embeddings = np.zeros((0, STARLING_EMBEDDING_DIM), dtype=np.float32)

        results = classify_disordered_proteins(
            embeddings, [], model, confidence_threshold=0.5
        )

        assert results == {}


# ============================================================================
# TestDisorderClassifierLoadSave
# ============================================================================


class TestDisorderClassifierLoadSave:
    """Tests for checkpoint save/load."""

    def test_save_and_reload(self, tmp_path):
        """Saved checkpoint can be reloaded."""
        model = DisorderClassifier(
            input_dim=512, num_classes=11,
            hidden_dims=(256, 128), dropout=0.3,
        )
        model.eval()

        checkpoint_path = tmp_path / "test_dc.pt"
        _save_disorder_checkpoint(checkpoint_path, model)

        loaded = load_disorder_classifier(model_path=checkpoint_path)
        assert loaded is not None
        loaded.eval()

        x = torch.randn(3, 512)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded(x)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_load_preserves_architecture(self, tmp_path):
        """Loaded model has correct architecture params."""
        model = DisorderClassifier(
            input_dim=512, num_classes=7,
            hidden_dims=(128, 64), dropout=0.2,
        )
        checkpoint_path = tmp_path / "test_dc.pt"
        _save_disorder_checkpoint(checkpoint_path, model)

        loaded = load_disorder_classifier(model_path=checkpoint_path)
        assert loaded.input_dim == 512
        assert loaded.num_classes == 7
        assert loaded.hidden_dims == (128, 64)
        assert loaded.dropout_rate == 0.2

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading from non-existent path returns None."""
        result = load_disorder_classifier(model_path=tmp_path / "nonexistent.pt")
        assert result is None


# ============================================================================
# TestDisorderClassifierPathManagement
# ============================================================================


class TestDisorderClassifierPathManagement:
    """Tests for disorder classifier path management."""

    def test_check_returns_false_when_missing(self, tmp_path):
        """check_disorder_classifier returns False when no checkpoint."""
        from vhold.databases.disorder_classifier import check_disorder_classifier
        assert check_disorder_classifier(model_dir=tmp_path) is False

    def test_check_returns_true_when_present(self, tmp_path):
        """check_disorder_classifier returns True when checkpoint exists."""
        from vhold.databases.disorder_classifier import check_disorder_classifier

        model = DisorderClassifier()
        checkpoint_dir = tmp_path / "disorder_classifier"
        checkpoint_dir.mkdir()
        _save_disorder_checkpoint(
            checkpoint_dir / "vhold_disorder_classifier.pt", model
        )

        assert check_disorder_classifier(model_dir=tmp_path) is True

    def test_get_path_structure(self, tmp_path):
        """Path includes 'disorder_classifier' subdirectory."""
        from vhold.databases.disorder_classifier import get_disorder_classifier_path

        path = get_disorder_classifier_path(model_dir=tmp_path)
        assert path.parent.name == "disorder_classifier"
        assert path.name == "vhold_disorder_classifier.pt"


# ============================================================================
# TestConsensusResultDisorderFields
# ============================================================================


class TestConsensusResultDisorderFields:
    """Tests for disorder fields on ConsensusResult."""

    def test_default_none(self):
        """Disorder fields default to None."""
        from vhold.results.consensus import ConsensusResult

        result = ConsensusResult(query_id="test", query_length=100)
        assert result.disorder_fraction is None
        assert result.disorder_regions is None
        assert result.disorder_class is None

    def test_to_dict_without_disorder(self):
        """to_dict omits disorder fields when None."""
        from vhold.results.consensus import ConsensusResult

        result = ConsensusResult(query_id="test", query_length=100)
        d = result.to_dict()
        assert "disorder_fraction" not in d
        assert "disorder_class" not in d
        assert "disorder_regions" not in d

    def test_to_dict_with_disorder(self):
        """to_dict includes disorder fields when set."""
        from vhold.results.consensus import ConsensusResult

        result = ConsensusResult(
            query_id="test",
            query_length=100,
            disorder_fraction=0.35,
            disorder_class="partially_disordered",
            disorder_regions=[(10, 25), (50, 70)],
        )
        d = result.to_dict()
        assert d["disorder_fraction"] == 0.35
        assert d["disorder_class"] == "partially_disordered"
        assert d["disorder_regions"] == "10-25;50-70"
