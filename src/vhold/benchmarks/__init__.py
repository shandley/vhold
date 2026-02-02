"""Benchmarking and evaluation tools for vHold."""

from vhold.benchmarks.metrics import (
    BenchmarkMetrics,
    ConfusionMatrix,
    CalibrationResult,
    calculate_metrics,
    calculate_confusion_matrix,
    calculate_calibration,
    calculate_dark_matter_reduction,
)
from vhold.benchmarks.datasets import (
    BenchmarkDataset,
    GoldStandardEntry,
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

__all__ = [
    # Metrics
    "BenchmarkMetrics",
    "ConfusionMatrix",
    "CalibrationResult",
    "calculate_metrics",
    "calculate_confusion_matrix",
    "calculate_calibration",
    "calculate_dark_matter_reduction",
    # Datasets
    "BenchmarkDataset",
    "GoldStandardEntry",
    "load_gold_standard",
    "create_holdout_set",
    "stratify_by_identity",
    # Evaluation
    "EvaluationResult",
    "evaluate_annotations",
    "compare_to_baseline",
    "generate_benchmark_report",
]
