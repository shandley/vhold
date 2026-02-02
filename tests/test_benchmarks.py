"""Tests for benchmarking and evaluation tools."""

import pytest
import tempfile
from pathlib import Path

from vhold.benchmarks.metrics import (
    ConfusionMatrix,
    BenchmarkMetrics,
    CalibrationResult,
    calculate_confusion_matrix,
    calculate_metrics,
    calculate_calibration,
    calculate_dark_matter_reduction,
)
from vhold.benchmarks.datasets import (
    GoldStandardEntry,
    BenchmarkDataset,
    load_gold_standard,
    create_holdout_set,
    stratify_by_identity,
)
from vhold.benchmarks.evaluation import (
    EvaluationResult,
    evaluate_annotations,
    compare_to_baseline,
    generate_benchmark_report,
)


class TestConfusionMatrix:
    """Tests for ConfusionMatrix dataclass."""

    def test_basic_metrics(self):
        """Test basic metric calculations."""
        cm = ConfusionMatrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )
        assert cm.total == 200
        assert cm.sensitivity == 0.8  # 80 / (80 + 20)
        assert cm.specificity == 0.9  # 90 / (90 + 10)
        assert cm.precision == pytest.approx(0.889, rel=0.01)  # 80 / (80 + 10)
        assert cm.accuracy == 0.85  # (80 + 90) / 200

    def test_f1_score(self):
        """Test F1 score calculation."""
        cm = ConfusionMatrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )
        # F1 = 2 * (precision * recall) / (precision + recall)
        expected_f1 = 2 * (0.889 * 0.8) / (0.889 + 0.8)
        assert cm.f1_score == pytest.approx(expected_f1, rel=0.01)

    def test_zero_division(self):
        """Test handling of zero division cases."""
        cm = ConfusionMatrix()
        assert cm.sensitivity == 0.0
        assert cm.specificity == 0.0
        assert cm.precision == 0.0
        assert cm.f1_score == 0.0
        assert cm.accuracy == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cm = ConfusionMatrix(
            true_positives=80,
            true_negatives=90,
            false_positives=10,
            false_negatives=20,
        )
        d = cm.to_dict()
        assert "sensitivity" in d
        assert "f1_score" in d
        assert d["true_positives"] == 80


class TestBenchmarkMetrics:
    """Tests for BenchmarkMetrics dataclass."""

    def test_annotation_rate(self):
        """Test annotation rate calculation."""
        metrics = BenchmarkMetrics(
            total_proteins=100,
            annotated_proteins=80,
        )
        assert metrics.annotation_rate == 0.8

    def test_annotation_accuracy(self):
        """Test annotation accuracy calculation."""
        metrics = BenchmarkMetrics(
            total_proteins=100,
            annotated_proteins=80,
            correct_annotations=60,
        )
        assert metrics.annotation_accuracy == 0.75

    def test_dark_matter_reduction(self):
        """Test dark matter reduction calculation."""
        metrics = BenchmarkMetrics(
            dark_matter_baseline=50,
            dark_matter_vhold=20,
        )
        assert metrics.dark_matter_reduction == 0.6  # (50-20)/50

    def test_zero_proteins(self):
        """Test with zero proteins."""
        metrics = BenchmarkMetrics(total_proteins=0)
        assert metrics.annotation_rate == 0.0
        assert metrics.annotation_accuracy == 0.0


class TestCalculateConfusionMatrix:
    """Tests for calculate_confusion_matrix function."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        predictions = ["annotated", "annotated", "unknown", "unknown"]
        ground_truth = ["annotated", "annotated", "unknown", "unknown"]
        cm = calculate_confusion_matrix(predictions, ground_truth)
        assert cm.true_positives == 2
        assert cm.true_negatives == 2
        assert cm.false_positives == 0
        assert cm.false_negatives == 0

    def test_all_wrong(self):
        """Test with all wrong predictions."""
        predictions = ["annotated", "annotated", "unknown", "unknown"]
        ground_truth = ["unknown", "unknown", "annotated", "annotated"]
        cm = calculate_confusion_matrix(predictions, ground_truth)
        assert cm.false_positives == 2
        assert cm.false_negatives == 2

    def test_length_mismatch(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError):
            calculate_confusion_matrix(["a", "b"], ["a"])


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_basic_calculation(self):
        """Test basic metrics calculation."""
        predicted = ["structural", "replication", "structural", "unknown"]
        true = ["structural", "replication", "replication", "unknown"]

        metrics = calculate_metrics(predicted, true)

        assert metrics.total_proteins == 4
        assert metrics.annotated_proteins == 3  # 3 non-unknown predictions
        assert metrics.correct_annotations == 2  # 2 exact matches

    def test_with_confidence_scores(self):
        """Test with confidence scores."""
        predicted = ["structural", "structural"]
        true = ["structural", "replication"]
        confidence = [0.9, 0.3]

        metrics = calculate_metrics(predicted, true, confidence)

        assert metrics.calibration_error >= 0
        assert metrics.brier_score >= 0


class TestCalculateCalibration:
    """Tests for calculate_calibration function."""

    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        # All predictions at 0.5 confidence, 50% correct
        confidence = [0.5] * 10
        correct = [True, False] * 5

        result = calculate_calibration(confidence, correct, n_bins=5)

        # Should be well calibrated
        assert result.expected_calibration_error < 0.2

    def test_overconfident(self):
        """Test overconfident predictions."""
        # High confidence but all wrong
        confidence = [0.9] * 10
        correct = [False] * 10

        result = calculate_calibration(confidence, correct, n_bins=5)

        # Should have high calibration error
        assert result.expected_calibration_error > 0.5

    def test_brier_score(self):
        """Test Brier score calculation."""
        confidence = [1.0, 0.0]
        correct = [True, True]  # Second is wrong (confidence 0 but correct)

        result = calculate_calibration(confidence, correct)

        # Brier = (1-1)^2 + (0-1)^2 / 2 = 0.5
        assert result.brier_score == 0.5


class TestCalculateDarkMatterReduction:
    """Tests for dark matter reduction calculation."""

    def test_basic_reduction(self):
        """Test basic reduction calculation."""
        result = calculate_dark_matter_reduction(
            baseline_unannotated=100,
            vhold_unannotated=40,
            total_proteins=200,
        )
        assert result["reduction_rate"] == 0.6
        assert result["proteins_illuminated"] == 60

    def test_no_baseline_dark_matter(self):
        """Test when baseline has no dark matter."""
        result = calculate_dark_matter_reduction(
            baseline_unannotated=0,
            vhold_unannotated=10,
            total_proteins=100,
        )
        assert result["reduction_rate"] == 0.0


class TestGoldStandardEntry:
    """Tests for GoldStandardEntry dataclass."""

    def test_creation(self):
        """Test creating an entry."""
        entry = GoldStandardEntry(
            protein_id="test_001",
            sequence="MVLSPADKTN",
            true_function="Major capsid protein",
            true_category="structural",
            evidence_code="ECO:0000269",
        )
        assert entry.protein_id == "test_001"
        assert entry.true_category == "structural"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        entry = GoldStandardEntry(
            protein_id="test_001",
            sequence="MVLSPADKTN",
            true_function="Capsid",
            true_category="structural",
            evidence_code="ECO:0000269",
            viral_family="Caudovirales",
        )
        d = entry.to_dict()
        assert d["protein_id"] == "test_001"
        assert d["viral_family"] == "Caudovirales"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "protein_id": "test_001",
            "sequence": "MVLSPADKTN",
            "true_function": "Capsid",
            "true_category": "structural",
            "evidence_code": "ECO:0000269",
        }
        entry = GoldStandardEntry.from_dict(data)
        assert entry.protein_id == "test_001"


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset dataclass."""

    def test_creation(self):
        """Test creating a dataset."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            ),
            GoldStandardEntry(
                protein_id="p2",
                sequence="MGKL",
                true_function="Polymerase",
                true_category="replication",
                evidence_code="ECO:0000269",
            ),
        ]
        dataset = BenchmarkDataset(
            name="test_dataset",
            description="Test",
            entries=entries,
        )
        assert dataset.size == 2
        assert dataset.categories == {"structural": 1, "replication": 1}

    def test_to_fasta(self):
        """Test writing to FASTA."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            ),
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
            dataset.to_fasta(Path(f.name))

            # Read back and verify
            with open(f.name) as fasta:
                content = fasta.read()
                assert ">p1" in content
                assert "MVLS" in content

    def test_filter_by_identity(self):
        """Test filtering by sequence identity."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
                max_identity_bfvd=0.15,
            ),
            GoldStandardEntry(
                protein_id="p2",
                sequence="MGKL",
                true_function="Polymerase",
                true_category="replication",
                evidence_code="ECO:0000269",
                max_identity_bfvd=0.45,
            ),
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        filtered = dataset.filter_by_identity(0.3, database="bfvd")
        assert filtered.size == 1
        assert filtered.entries[0].protein_id == "p1"

    def test_json_roundtrip(self):
        """Test JSON save and load."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            ),
        ]
        dataset = BenchmarkDataset(
            name="test",
            description="Test dataset",
            entries=entries,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dataset.to_json(Path(f.name))
            loaded = BenchmarkDataset.from_json(Path(f.name))

            assert loaded.name == "test"
            assert loaded.size == 1
            assert loaded.entries[0].protein_id == "p1"


class TestCreateHoldoutSet:
    """Tests for create_holdout_set function."""

    def test_basic_split(self):
        """Test basic dataset split."""
        entries = [
            GoldStandardEntry(
                protein_id=f"p{i}",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            )
            for i in range(10)
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        train, holdout = create_holdout_set(dataset, holdout_fraction=0.2)

        assert train.size + holdout.size == 10
        assert holdout.size >= 1

    def test_stratified_split(self):
        """Test stratified split by category."""
        entries = [
            GoldStandardEntry(
                protein_id=f"p{i}",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural" if i < 5 else "replication",
                evidence_code="ECO:0000269",
            )
            for i in range(10)
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        train, holdout = create_holdout_set(
            dataset, holdout_fraction=0.2, stratify_by="category"
        )

        # Each category should have at least 1 in holdout
        holdout_cats = [e.true_category for e in holdout.entries]
        assert "structural" in holdout_cats or "replication" in holdout_cats


class TestStratifyByIdentity:
    """Tests for stratify_by_identity function."""

    def test_basic_stratification(self):
        """Test basic identity stratification."""
        entries = [
            GoldStandardEntry(
                protein_id=f"p{i}",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
                max_identity_bfvd=i * 0.1,  # 0.0, 0.1, 0.2, ..., 0.9
            )
            for i in range(10)
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        stratified = stratify_by_identity(dataset)

        assert len(stratified) > 0
        # Check that entries are in correct bins
        for bin_name, bin_dataset in stratified.items():
            assert bin_dataset.size >= 0


class TestEvaluateAnnotations:
    """Tests for evaluate_annotations function."""

    def test_basic_evaluation(self):
        """Test basic evaluation."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            ),
            GoldStandardEntry(
                protein_id="p2",
                sequence="MGKL",
                true_function="Polymerase",
                true_category="replication",
                evidence_code="ECO:0000269",
            ),
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        vhold_results = {
            "p1": {"functional_category": "structural", "consensus_score": 0.8},
            "p2": {"functional_category": "structural", "consensus_score": 0.6},
        }

        result = evaluate_annotations(dataset, vhold_results)

        assert result.dataset_size == 2
        assert result.metrics.annotated_proteins == 2
        assert result.metrics.correct_annotations == 1  # Only p1 is correct


class TestCompareToBaseline:
    """Tests for compare_to_baseline function."""

    def test_basic_comparison(self):
        """Test basic baseline comparison."""
        entries = [
            GoldStandardEntry(
                protein_id="p1",
                sequence="MVLS",
                true_function="Capsid",
                true_category="structural",
                evidence_code="ECO:0000269",
            ),
            GoldStandardEntry(
                protein_id="p2",
                sequence="MGKL",
                true_function="Polymerase",
                true_category="replication",
                evidence_code="ECO:0000269",
            ),
        ]
        dataset = BenchmarkDataset(name="test", description="", entries=entries)

        vhold_results = {
            "p1": {"functional_category": "structural"},
            "p2": {"functional_category": "replication"},
        }
        baseline_results = {
            "p1": {"functional_category": "unknown"},
            "p2": {"functional_category": "replication"},
        }

        comparison = compare_to_baseline(dataset, vhold_results, baseline_results)

        assert comparison["head_to_head"]["vhold_only_correct"] == 1
        assert comparison["head_to_head"]["both_correct"] == 1


class TestGenerateBenchmarkReport:
    """Tests for generate_benchmark_report function."""

    def test_report_generation(self):
        """Test report generation."""
        result = EvaluationResult(
            dataset_name="test",
            dataset_size=100,
            metrics=BenchmarkMetrics(
                total_proteins=100,
                annotated_proteins=80,
                correct_annotations=60,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = generate_benchmark_report([result], Path(tmp_dir))

            assert report_path.exists()

            import json
            with open(report_path) as f:
                report = json.load(f)

            assert report["summary"]["total_datasets"] == 1
            assert report["summary"]["total_proteins"] == 100
