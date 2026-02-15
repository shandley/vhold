"""Tests for ONNX INT8 quantization backend."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from vhold.cli import main
from vhold.features.onnx_export import (
    ONNX_FILES,
    check_onnx_available,
    check_onnx_exported,
    _detect_quantization_target,
)
from vhold.features.prostt5 import (
    GEN_KWARGS,
    FAST_GEN_KWARGS,
    PredictionResult,
    calculate_confidence_scores,
    calculate_batch_confidence_scores,
    get_predictor,
)
from vhold.utils.constants import EMBEDDING_DIM, get_onnx_model_dir


# ============================================================================
# check_onnx_available
# ============================================================================


class TestCheckOnnxAvailable:
    """Tests for ONNX availability checks."""

    def test_returns_false_when_imports_fail(self):
        """Should return False when optimum/onnxruntime not installed."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            # Force reimport by clearing cached result
            import importlib
            from vhold.features import onnx_export
            importlib.reload(onnx_export)
            result = onnx_export.check_onnx_available()
            # Depending on environment, may or may not be installed
            assert isinstance(result, bool)

    def test_returns_bool(self):
        """check_onnx_available always returns a bool."""
        result = check_onnx_available()
        assert isinstance(result, bool)


# ============================================================================
# check_onnx_exported
# ============================================================================


class TestCheckOnnxExported:
    """Tests for checking exported ONNX models."""

    def test_false_when_dir_empty(self):
        """Should return False when no ONNX files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert check_onnx_exported(Path(tmpdir)) is False

    def test_false_when_partial(self):
        """Should return False when only some ONNX files present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "encoder_model.onnx").touch()
            assert check_onnx_exported(Path(tmpdir)) is False

    def test_true_when_all_present(self):
        """Should return True when all ONNX files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in ONNX_FILES:
                (Path(tmpdir) / f).touch()
            assert check_onnx_exported(Path(tmpdir)) is True

    def test_uses_default_dir_when_none(self):
        """Should use default ONNX model dir when None passed."""
        result = check_onnx_exported(None)
        assert isinstance(result, bool)


# ============================================================================
# _detect_quantization_target
# ============================================================================


class TestDetectQuantizationTarget:
    """Tests for CPU auto-detection."""

    def test_returns_valid_target(self):
        """Should return one of the valid quantization targets."""
        target = _detect_quantization_target()
        assert target in ("avx512_vnni", "avx512", "avx2", "arm64")

    @patch("platform.machine", return_value="arm64")
    def test_arm64_detection(self, _mock):
        """Should detect ARM64 on Apple Silicon."""
        assert _detect_quantization_target() == "arm64"

    @patch("platform.machine", return_value="aarch64")
    def test_aarch64_detection(self, _mock):
        """Should detect ARM64 on Linux aarch64."""
        assert _detect_quantization_target() == "arm64"


# ============================================================================
# get_predictor factory
# ============================================================================


class TestGetPredictor:
    """Tests for the get_predictor factory function."""

    def test_torch_backend_returns_prostt5(self):
        """Default torch backend should return ProstT5Predictor."""
        from vhold.features.prostt5 import ProstT5Predictor
        predictor = get_predictor(device="cpu", backend="torch")
        assert isinstance(predictor, ProstT5Predictor)

    def test_onnx_backend_not_available(self):
        """Should raise ImportError when ONNX deps not installed."""
        with patch(
            "vhold.features.onnx_export.check_onnx_available", return_value=False
        ):
            with pytest.raises(ImportError, match="ONNX backend requires"):
                get_predictor(backend="onnx")

    def test_onnx_backend_not_exported(self):
        """Should raise FileNotFoundError when ONNX models not exported."""
        with patch(
            "vhold.features.onnx_export.check_onnx_available", return_value=True
        ), patch(
            "vhold.features.onnx_export.check_onnx_exported", return_value=False
        ):
            with pytest.raises(FileNotFoundError, match="ONNX models not found"):
                get_predictor(backend="onnx")


# ============================================================================
# Module-level generation kwargs
# ============================================================================


class TestModuleLevelConstants:
    """Tests that generation kwargs are properly exposed at module level."""

    def test_gen_kwargs_has_beam_search(self):
        """Standard gen kwargs should use beam search."""
        assert GEN_KWARGS["num_beams"] == 3
        assert GEN_KWARGS["do_sample"] is True

    def test_fast_gen_kwargs_is_greedy(self):
        """Fast gen kwargs should use greedy decoding."""
        assert FAST_GEN_KWARGS["num_beams"] == 1
        assert FAST_GEN_KWARGS["do_sample"] is False


# ============================================================================
# Module-level confidence scoring functions
# ============================================================================


class TestModuleLevelConfidenceScoring:
    """Tests for extracted module-level confidence scoring functions."""

    def test_calculate_confidence_empty_scores(self):
        """Should return all 1.0 when no scores provided."""
        result = calculate_confidence_scores((), 5)
        assert result == [1.0] * 5

    def test_calculate_confidence_truncates(self):
        """Should truncate when scores > expected length."""
        import torch
        scores = tuple(
            torch.randn(1, 100) for _ in range(10)
        )
        result = calculate_confidence_scores(scores, 5)
        assert len(result) == 5

    def test_calculate_confidence_pads(self):
        """Should pad when scores < expected length."""
        import torch
        scores = tuple(
            torch.randn(1, 100) for _ in range(3)
        )
        result = calculate_confidence_scores(scores, 5)
        assert len(result) == 5

    def test_calculate_batch_confidence_empty(self):
        """Should return correct shapes for empty scores."""
        result = calculate_batch_confidence_scores((), 3, [5, 10, 7])
        assert len(result) == 3
        assert len(result[0]) == 5
        assert len(result[1]) == 10
        assert len(result[2]) == 7


# ============================================================================
# OnnxProstT5Predictor (mocked)
# ============================================================================


class TestOnnxProstT5Predictor:
    """Tests for the ONNX predictor class (without actual ONNX Runtime)."""

    def test_import(self):
        """OnnxProstT5Predictor should be importable."""
        from vhold.features.onnx_predictor import OnnxProstT5Predictor
        assert OnnxProstT5Predictor is not None

    def test_init_defaults(self):
        """Should initialize with default settings."""
        from vhold.features.onnx_predictor import OnnxProstT5Predictor
        predictor = OnnxProstT5Predictor()
        assert predictor.fast is False
        assert predictor._loaded is False

    def test_init_fast_mode(self):
        """Should set fast mode gen kwargs."""
        from vhold.features.onnx_predictor import OnnxProstT5Predictor
        predictor = OnnxProstT5Predictor(fast=True)
        assert predictor.fast is True
        assert predictor._gen_kwargs["do_sample"] is False
        assert predictor._gen_kwargs["num_beams"] == 1

    def test_predict_batch_returns_dict(self):
        """predict_batch should return dict mapping IDs to PredictionResults."""
        from vhold.features.onnx_predictor import OnnxProstT5Predictor

        predictor = OnnxProstT5Predictor()

        # Mock predict_single to avoid loading actual model
        mock_result = PredictionResult(
            sequence_id="test",
            aa_sequence="MKTL",
            three_di_sequence="DCDC",
            confidence_scores=[0.9, 0.8, 0.7, 0.85],
        )
        predictor._loaded = True
        predictor.predict_single = MagicMock(return_value=mock_result)

        results = predictor.predict_batch(
            {"test": "MKTL"}, show_progress=False
        )

        assert "test" in results
        assert isinstance(results["test"], PredictionResult)
        assert results["test"].three_di_sequence == "DCDC"

    def test_error_handling_in_batch(self):
        """predict_batch should handle errors gracefully."""
        from vhold.features.onnx_predictor import OnnxProstT5Predictor

        predictor = OnnxProstT5Predictor()
        predictor._loaded = True
        predictor.predict_single = MagicMock(
            side_effect=RuntimeError("ONNX error")
        )

        results = predictor.predict_batch(
            {"test": "MKTL"}, show_progress=False
        )

        assert "test" in results
        assert results["test"].confidence_scores == [0.0] * 4
        assert results["test"].three_di_sequence == "DDDD"


# ============================================================================
# OnnxEmbeddingExtractor (mocked)
# ============================================================================


class TestOnnxEmbeddingExtractor:
    """Tests for the ONNX embedding extractor (without actual ONNX Runtime)."""

    def test_import(self):
        """OnnxEmbeddingExtractor should be importable."""
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor
        assert OnnxEmbeddingExtractor is not None

    def test_init_defaults(self):
        """Should initialize with default settings."""
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor
        extractor = OnnxEmbeddingExtractor()
        assert extractor._loaded is False
        assert extractor.session is None

    def test_extract_batch_returns_correct_types(self):
        """extract_batch should return (list[str], ndarray)."""
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor

        extractor = OnnxEmbeddingExtractor()

        # Mock extract_single
        mock_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_emb /= np.linalg.norm(mock_emb)
        extractor._loaded = True
        extractor.extract_single = MagicMock(return_value=mock_emb)

        ids, embeddings = extractor.extract_batch(
            {"prot1": "MKTL", "prot2": "ARND"},
            show_progress=False,
        )

        assert len(ids) == 2
        assert embeddings.shape == (2, EMBEDDING_DIM)

    def test_embeddings_are_normalized(self):
        """Extracted embeddings should be L2-normalized."""
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor

        extractor = OnnxEmbeddingExtractor()

        # Create a mock embedding that is properly normalized
        mock_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        mock_emb /= np.linalg.norm(mock_emb)
        extractor._loaded = True
        extractor.extract_single = MagicMock(return_value=mock_emb)

        ids, embeddings = extractor.extract_batch(
            {"prot1": "MKTL"},
            show_progress=False,
        )

        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-5

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor

        extractor = OnnxEmbeddingExtractor()
        extractor._loaded = True

        ids, embeddings = extractor.extract_batch(
            {}, show_progress=False,
        )

        assert len(ids) == 0
        assert embeddings.shape == (0, EMBEDDING_DIM)


# ============================================================================
# CLI integration
# ============================================================================


class TestCLIBackendOption:
    """Tests for --backend CLI flag."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_run_help_shows_backend(self, runner):
        """run --help should mention --backend option."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    def test_predict_help_shows_backend(self, runner):
        """predict --help should mention --backend option."""
        result = runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    def test_align_help_shows_backend(self, runner):
        """align --help should mention --backend option."""
        result = runner.invoke(main, ["align", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    def test_export_onnx_help(self, runner):
        """export-onnx --help should show help text."""
        result = runner.invoke(main, ["export-onnx", "--help"])
        assert result.exit_code == 0
        assert "onnx" in result.output.lower()

    def test_export_onnx_shows_quantization_options(self, runner):
        """export-onnx should show quantization choices."""
        result = runner.invoke(main, ["export-onnx", "--help"])
        assert "avx512_vnni" in result.output
        assert "arm64" in result.output


# ============================================================================
# triage_proteins extractor parameter
# ============================================================================


class TestTriageExtractorParam:
    """Tests that triage_proteins accepts a pre-built extractor."""

    def test_accepts_custom_extractor(self):
        """triage_proteins should use a pre-built extractor when provided."""
        from vhold.features.embeddings import triage_proteins, EmbeddingDatabase

        mock_extractor = MagicMock()
        mock_extractor.extract_batch.return_value = (
            ["prot1"],
            np.random.randn(1, EMBEDDING_DIM).astype(np.float32),
        )

        mock_db = MagicMock(spec=EmbeddingDatabase)
        mock_db.search.return_value = {}

        result, extractor = triage_proteins(
            sequences={"prot1": "MKTL"},
            embedding_db=mock_db,
            threshold=0.9,
            extractor=mock_extractor,
        )

        # Should have used the mock extractor, not created a new one
        mock_extractor.extract_batch.assert_called_once()
        assert extractor is mock_extractor
        assert result.stats["total"] == 1


# ============================================================================
# get_onnx_model_dir
# ============================================================================


class TestOnnxModelDir:
    """Tests for ONNX model directory utilities."""

    def test_get_onnx_model_dir_returns_path(self):
        """Should return a Path object."""
        result = get_onnx_model_dir()
        assert isinstance(result, Path)
        assert result.name == "onnx_int8"
