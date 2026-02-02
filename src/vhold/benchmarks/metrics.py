"""Benchmark metrics for vHold evaluation.

This module provides functions for calculating standard metrics
used in evaluating annotation performance.
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfusionMatrix:
    """Confusion matrix for binary classification."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def total(self) -> int:
        """Total number of samples."""
        return (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )

    @property
    def sensitivity(self) -> float:
        """Sensitivity (recall, true positive rate)."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """Alias for sensitivity."""
        return self.sensitivity

    @property
    def specificity(self) -> float:
        """Specificity (true negative rate)."""
        denom = self.true_negatives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_negatives / denom

    @property
    def precision(self) -> float:
        """Precision (positive predictive value)."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f1_score(self) -> float:
        """F1 score (harmonic mean of precision and recall)."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def matthews_correlation(self) -> float:
        """Matthews correlation coefficient."""
        tp, tn = self.true_positives, self.true_negatives
        fp, fn = self.false_positives, self.false_negatives

        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        if denom == 0:
            return 0.0
        return (tp * tn - fp * fn) / denom

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "mcc": round(self.matthews_correlation, 4),
        }


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""

    # Basic counts
    total_proteins: int = 0
    annotated_proteins: int = 0
    correct_annotations: int = 0

    # Confusion matrix (for category-level evaluation)
    confusion_matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    # Per-category metrics
    category_metrics: dict[str, ConfusionMatrix] = field(default_factory=dict)

    # Confidence calibration
    calibration_error: float = 0.0
    brier_score: float = 0.0

    # Dark matter metrics
    dark_matter_baseline: int = 0
    dark_matter_vhold: int = 0

    # Stratified metrics
    metrics_by_identity: dict[str, "BenchmarkMetrics"] = field(default_factory=dict)
    metrics_by_family: dict[str, "BenchmarkMetrics"] = field(default_factory=dict)

    @property
    def annotation_rate(self) -> float:
        """Fraction of proteins that received annotation."""
        if self.total_proteins == 0:
            return 0.0
        return self.annotated_proteins / self.total_proteins

    @property
    def annotation_accuracy(self) -> float:
        """Fraction of annotated proteins with correct annotation."""
        if self.annotated_proteins == 0:
            return 0.0
        return self.correct_annotations / self.annotated_proteins

    @property
    def dark_matter_reduction(self) -> float:
        """Reduction in dark matter compared to baseline."""
        if self.dark_matter_baseline == 0:
            return 0.0
        return (
            (self.dark_matter_baseline - self.dark_matter_vhold)
            / self.dark_matter_baseline
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_proteins": self.total_proteins,
                "annotated_proteins": self.annotated_proteins,
                "correct_annotations": self.correct_annotations,
                "annotation_rate": round(self.annotation_rate, 4),
                "annotation_accuracy": round(self.annotation_accuracy, 4),
            },
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "calibration": {
                "expected_calibration_error": round(self.calibration_error, 4),
                "brier_score": round(self.brier_score, 4),
            },
            "dark_matter": {
                "baseline": self.dark_matter_baseline,
                "vhold": self.dark_matter_vhold,
                "reduction": round(self.dark_matter_reduction, 4),
            },
            "category_metrics": {
                cat: cm.to_dict() for cat, cm in self.category_metrics.items()
            },
        }


@dataclass
class CalibrationResult:
    """Results from confidence calibration analysis."""

    # Per-bin statistics
    bin_edges: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_confidences: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    # Summary metrics
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    brier_score: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "bins": {
                "edges": self.bin_edges,
                "accuracies": [round(a, 4) for a in self.bin_accuracies],
                "confidences": [round(c, 4) for c in self.bin_confidences],
                "counts": self.bin_counts,
            },
            "metrics": {
                "expected_calibration_error": round(self.expected_calibration_error, 4),
                "maximum_calibration_error": round(self.maximum_calibration_error, 4),
                "brier_score": round(self.brier_score, 4),
            },
        }


def calculate_confusion_matrix(
    predictions: list[str],
    ground_truth: list[str],
    positive_class: str = "annotated",
) -> ConfusionMatrix:
    """Calculate confusion matrix from predictions and ground truth.

    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        positive_class: Label to consider as positive

    Returns:
        ConfusionMatrix object
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    cm = ConfusionMatrix()

    for pred, truth in zip(predictions, ground_truth):
        pred_positive = pred == positive_class
        truth_positive = truth == positive_class

        if pred_positive and truth_positive:
            cm.true_positives += 1
        elif not pred_positive and not truth_positive:
            cm.true_negatives += 1
        elif pred_positive and not truth_positive:
            cm.false_positives += 1
        else:
            cm.false_negatives += 1

    return cm


def calculate_metrics(
    predicted_categories: list[str],
    true_categories: list[str],
    confidence_scores: list[float] | None = None,
) -> BenchmarkMetrics:
    """Calculate comprehensive benchmark metrics.

    Args:
        predicted_categories: List of predicted functional categories
        true_categories: List of true functional categories
        confidence_scores: Optional list of confidence scores

    Returns:
        BenchmarkMetrics object
    """
    if len(predicted_categories) != len(true_categories):
        raise ValueError("Predictions and ground truth must have same length")

    metrics = BenchmarkMetrics(total_proteins=len(predicted_categories))

    # Count annotations and correct predictions
    for pred, truth in zip(predicted_categories, true_categories):
        if pred and pred != "unknown":
            metrics.annotated_proteins += 1
            if pred == truth:
                metrics.correct_annotations += 1

    # Overall confusion matrix (annotated vs not)
    pred_annotated = [
        "annotated" if (p and p != "unknown") else "unknown"
        for p in predicted_categories
    ]
    truth_annotated = [
        "annotated" if (t and t != "unknown") else "unknown" for t in true_categories
    ]
    metrics.confusion_matrix = calculate_confusion_matrix(
        pred_annotated, truth_annotated, "annotated"
    )

    # Per-category metrics
    all_categories = set(predicted_categories) | set(true_categories)
    all_categories.discard("unknown")
    all_categories.discard("")

    for category in all_categories:
        pred_cat = [category if p == category else "other" for p in predicted_categories]
        truth_cat = [category if t == category else "other" for t in true_categories]
        metrics.category_metrics[category] = calculate_confusion_matrix(
            pred_cat, truth_cat, category
        )

    # Calibration metrics (if confidence scores provided)
    if confidence_scores:
        calibration = calculate_calibration(
            confidence_scores,
            [p == t for p, t in zip(predicted_categories, true_categories)],
        )
        metrics.calibration_error = calibration.expected_calibration_error
        metrics.brier_score = calibration.brier_score

    return metrics


def calculate_calibration(
    confidence_scores: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """Calculate confidence calibration metrics.

    Args:
        confidence_scores: Predicted confidence scores (0-1)
        correct: Whether each prediction was correct
        n_bins: Number of bins for calibration

    Returns:
        CalibrationResult object
    """
    if len(confidence_scores) != len(correct):
        raise ValueError("Scores and correctness must have same length")

    result = CalibrationResult()

    # Create bins
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    result.bin_edges = bin_edges

    # Assign predictions to bins
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]

        # Find predictions in this bin
        in_bin = [
            (conf, corr)
            for conf, corr in zip(confidence_scores, correct)
            if low <= conf < high or (i == n_bins - 1 and conf == high)
        ]

        if in_bin:
            bin_confs = [conf for conf, _ in in_bin]
            bin_corrs = [corr for _, corr in in_bin]

            accuracy = sum(bin_corrs) / len(bin_corrs)
            avg_confidence = sum(bin_confs) / len(bin_confs)

            bin_accuracies.append(accuracy)
            bin_confidences.append(avg_confidence)
            bin_counts.append(len(in_bin))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((low + high) / 2)
            bin_counts.append(0)

    result.bin_accuracies = bin_accuracies
    result.bin_confidences = bin_confidences
    result.bin_counts = bin_counts

    # Calculate ECE (Expected Calibration Error)
    total_samples = sum(bin_counts)
    if total_samples > 0:
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        )
        result.expected_calibration_error = ece

        # Maximum calibration error
        mce = max(
            abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        )
        result.maximum_calibration_error = mce

    # Brier score
    if confidence_scores:
        brier = sum(
            (conf - (1.0 if corr else 0.0)) ** 2
            for conf, corr in zip(confidence_scores, correct)
        ) / len(confidence_scores)
        result.brier_score = brier

    return result


def calculate_dark_matter_reduction(
    baseline_unannotated: int,
    vhold_unannotated: int,
    total_proteins: int,
) -> dict:
    """Calculate dark matter reduction metrics.

    Args:
        baseline_unannotated: Proteins unannotated by baseline (BLAST)
        vhold_unannotated: Proteins unannotated by vHold
        total_proteins: Total number of proteins

    Returns:
        Dict with dark matter metrics
    """
    if baseline_unannotated == 0:
        reduction_rate = 0.0
    else:
        reduction_rate = (baseline_unannotated - vhold_unannotated) / baseline_unannotated

    return {
        "baseline_dark_matter": baseline_unannotated,
        "baseline_dark_matter_rate": baseline_unannotated / total_proteins if total_proteins > 0 else 0,
        "vhold_dark_matter": vhold_unannotated,
        "vhold_dark_matter_rate": vhold_unannotated / total_proteins if total_proteins > 0 else 0,
        "reduction_count": baseline_unannotated - vhold_unannotated,
        "reduction_rate": reduction_rate,
        "proteins_illuminated": baseline_unannotated - vhold_unannotated,
    }
