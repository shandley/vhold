"""Benchmark evaluation for vHold.

This module provides functions for evaluating vHold predictions
against gold standard annotations and comparing to baseline methods.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

from vhold.benchmarks.metrics import (
    BenchmarkMetrics,
    calculate_metrics,
    calculate_dark_matter_reduction,
)
from vhold.benchmarks.datasets import BenchmarkDataset, GoldStandardEntry
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating vHold on a benchmark dataset."""

    dataset_name: str
    dataset_size: int

    # Overall metrics
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    # Per-protein results
    predictions: dict[str, dict] = field(default_factory=dict)

    # Comparison to baselines
    baseline_comparisons: dict[str, BenchmarkMetrics] = field(default_factory=dict)

    # Stratified results
    metrics_by_identity: dict[str, BenchmarkMetrics] = field(default_factory=dict)
    metrics_by_category: dict[str, BenchmarkMetrics] = field(default_factory=dict)
    metrics_by_family: dict[str, BenchmarkMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "dataset": {
                "name": self.dataset_name,
                "size": self.dataset_size,
            },
            "metrics": self.metrics.to_dict(),
            "baseline_comparisons": {
                name: m.to_dict() for name, m in self.baseline_comparisons.items()
            },
            "stratified": {
                "by_identity": {
                    bin_name: m.to_dict()
                    for bin_name, m in self.metrics_by_identity.items()
                },
                "by_category": {
                    cat: m.to_dict() for cat, m in self.metrics_by_category.items()
                },
                "by_family": {
                    fam: m.to_dict() for fam, m in self.metrics_by_family.items()
                },
            },
        }

    def to_json(self, output_path: Path) -> None:
        """Save results to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")


def evaluate_annotations(
    dataset: BenchmarkDataset,
    vhold_results: dict[str, dict],
    category_mapping: dict[str, str] | None = None,
) -> EvaluationResult:
    """Evaluate vHold annotations against gold standard.

    Args:
        dataset: Gold standard benchmark dataset
        vhold_results: Dict mapping protein_id to vHold result dict
        category_mapping: Optional mapping from vHold categories to gold standard categories

    Returns:
        EvaluationResult with evaluation metrics
    """
    result = EvaluationResult(
        dataset_name=dataset.name,
        dataset_size=dataset.size,
    )

    # Collect predictions and ground truth
    predicted_categories = []
    true_categories = []
    confidence_scores = []

    for entry in dataset.entries:
        vhold_result = vhold_results.get(entry.protein_id, {})

        pred_category = vhold_result.get("functional_category", "unknown")
        if category_mapping and pred_category in category_mapping:
            pred_category = category_mapping[pred_category]

        predicted_categories.append(pred_category)
        true_categories.append(entry.true_category)

        # Get confidence score
        score = vhold_result.get("consensus_score", 0.0)
        confidence_scores.append(score)

        # Store per-protein prediction
        result.predictions[entry.protein_id] = {
            "predicted": pred_category,
            "true": entry.true_category,
            "correct": pred_category == entry.true_category,
            "confidence": score,
        }

    # Calculate overall metrics
    result.metrics = calculate_metrics(
        predicted_categories,
        true_categories,
        confidence_scores,
    )

    # Stratify by sequence identity
    identity_bins = {
        "0-20%": (0.0, 0.2),
        "20-30%": (0.2, 0.3),
        "30-40%": (0.3, 0.4),
        "40-50%": (0.4, 0.5),
        "50-100%": (0.5, 1.0),
    }

    for bin_name, (low, high) in identity_bins.items():
        bin_entries = [
            e for e in dataset.entries
            if low <= e.max_identity_bfvd < high
        ]
        if bin_entries:
            bin_pred = [
                vhold_results.get(e.protein_id, {}).get("functional_category", "unknown")
                for e in bin_entries
            ]
            bin_true = [e.true_category for e in bin_entries]
            bin_conf = [
                vhold_results.get(e.protein_id, {}).get("consensus_score", 0.0)
                for e in bin_entries
            ]

            result.metrics_by_identity[bin_name] = calculate_metrics(
                bin_pred, bin_true, bin_conf
            )

    # Stratify by category
    for category in dataset.categories.keys():
        cat_entries = [e for e in dataset.entries if e.true_category == category]
        if cat_entries:
            cat_pred = [
                vhold_results.get(e.protein_id, {}).get("functional_category", "unknown")
                for e in cat_entries
            ]
            cat_true = [e.true_category for e in cat_entries]
            cat_conf = [
                vhold_results.get(e.protein_id, {}).get("consensus_score", 0.0)
                for e in cat_entries
            ]

            result.metrics_by_category[category] = calculate_metrics(
                cat_pred, cat_true, cat_conf
            )

    # Stratify by viral family
    for family in dataset.viral_families.keys():
        fam_entries = [e for e in dataset.entries if e.viral_family == family]
        if fam_entries:
            fam_pred = [
                vhold_results.get(e.protein_id, {}).get("functional_category", "unknown")
                for e in fam_entries
            ]
            fam_true = [e.true_category for e in fam_entries]
            fam_conf = [
                vhold_results.get(e.protein_id, {}).get("consensus_score", 0.0)
                for e in fam_entries
            ]

            result.metrics_by_family[family] = calculate_metrics(
                fam_pred, fam_true, fam_conf
            )

    logger.info(
        f"Evaluation: {result.metrics.annotation_rate:.1%} annotation rate, "
        f"{result.metrics.annotation_accuracy:.1%} accuracy"
    )

    return result


def compare_to_baseline(
    dataset: BenchmarkDataset,
    vhold_results: dict[str, dict],
    baseline_results: dict[str, dict],
    baseline_name: str = "BLAST",
) -> dict:
    """Compare vHold to a baseline method.

    Args:
        dataset: Gold standard benchmark dataset
        vhold_results: Dict mapping protein_id to vHold result
        baseline_results: Dict mapping protein_id to baseline result
        baseline_name: Name of baseline method

    Returns:
        Dict with comparison metrics
    """
    # Calculate metrics for both methods
    vhold_pred = []
    baseline_pred = []
    true_cats = []

    for entry in dataset.entries:
        vhold_cat = vhold_results.get(entry.protein_id, {}).get(
            "functional_category", "unknown"
        )
        baseline_cat = baseline_results.get(entry.protein_id, {}).get(
            "functional_category", "unknown"
        )

        vhold_pred.append(vhold_cat)
        baseline_pred.append(baseline_cat)
        true_cats.append(entry.true_category)

    vhold_metrics = calculate_metrics(vhold_pred, true_cats)
    baseline_metrics = calculate_metrics(baseline_pred, true_cats)

    # Dark matter comparison
    vhold_unannotated = sum(1 for p in vhold_pred if p == "unknown")
    baseline_unannotated = sum(1 for p in baseline_pred if p == "unknown")

    dm_reduction = calculate_dark_matter_reduction(
        baseline_unannotated,
        vhold_unannotated,
        len(dataset.entries),
    )

    # Head-to-head comparison
    vhold_better = 0
    baseline_better = 0
    both_correct = 0
    both_wrong = 0

    for vhold, baseline, true in zip(vhold_pred, baseline_pred, true_cats):
        vhold_correct = vhold == true
        baseline_correct = baseline == true

        if vhold_correct and baseline_correct:
            both_correct += 1
        elif vhold_correct and not baseline_correct:
            vhold_better += 1
        elif not vhold_correct and baseline_correct:
            baseline_better += 1
        else:
            both_wrong += 1

    comparison = {
        "baseline_name": baseline_name,
        "vhold_metrics": vhold_metrics.to_dict(),
        "baseline_metrics": baseline_metrics.to_dict(),
        "dark_matter_reduction": dm_reduction,
        "head_to_head": {
            "both_correct": both_correct,
            "vhold_only_correct": vhold_better,
            "baseline_only_correct": baseline_better,
            "both_wrong": both_wrong,
        },
        "improvement": {
            "annotation_rate": (
                vhold_metrics.annotation_rate - baseline_metrics.annotation_rate
            ),
            "accuracy": (
                vhold_metrics.annotation_accuracy - baseline_metrics.annotation_accuracy
            ),
            "f1_score": (
                vhold_metrics.confusion_matrix.f1_score
                - baseline_metrics.confusion_matrix.f1_score
            ),
        },
    }

    logger.info(
        f"Comparison vs {baseline_name}: "
        f"vHold better on {vhold_better} proteins, "
        f"{baseline_name} better on {baseline_better}"
    )

    return comparison


def generate_benchmark_report(
    evaluation_results: list[EvaluationResult],
    output_dir: Path,
) -> Path:
    """Generate comprehensive benchmark report.

    Args:
        evaluation_results: List of evaluation results
        output_dir: Output directory for report

    Returns:
        Path to generated report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {
            "total_datasets": len(evaluation_results),
            "total_proteins": sum(r.dataset_size for r in evaluation_results),
        },
        "datasets": [],
    }

    # Aggregate metrics
    all_annotation_rates = []
    all_accuracies = []
    all_f1_scores = []

    for result in evaluation_results:
        report["datasets"].append({
            "name": result.dataset_name,
            "size": result.dataset_size,
            "metrics": result.metrics.to_dict(),
        })

        all_annotation_rates.append(result.metrics.annotation_rate)
        all_accuracies.append(result.metrics.annotation_accuracy)
        all_f1_scores.append(result.metrics.confusion_matrix.f1_score)

    # Summary statistics
    if all_annotation_rates:
        import statistics

        report["summary"]["mean_annotation_rate"] = round(
            statistics.mean(all_annotation_rates), 4
        )
        report["summary"]["mean_accuracy"] = round(
            statistics.mean(all_accuracies), 4
        )
        report["summary"]["mean_f1_score"] = round(
            statistics.mean(all_f1_scores), 4
        )

    # Write report
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Generated benchmark report: {report_path}")

    return report_path
