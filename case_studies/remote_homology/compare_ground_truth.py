#!/usr/bin/env python3
"""
Compare vHold annotations against ground truth.

Evaluates annotation accuracy by comparing:
1. Functional category (structural, replication, etc.)
2. Protein description quality
3. Confidence levels
"""

import json
import csv
import sys
from pathlib import Path


def load_ground_truth(case_study_dir: Path) -> dict:
    """Load ground truth annotations."""
    gt_path = case_study_dir / "ground_truth.json"
    with open(gt_path) as f:
        return json.load(f)


def load_vhold_results(results_dir: Path) -> dict:
    """Load vHold annotation results."""
    results = {}
    tsv_path = results_dir / "vhold_results.tsv"

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_id = row["query_id"]
            results[query_id] = row

    return results


def compare_annotations(results_dir: Path, case_study_dir: Path):
    """Compare vHold annotations against ground truth."""

    ground_truth = load_ground_truth(case_study_dir)
    vhold_results = load_vhold_results(results_dir)

    print("=" * 80)
    print("GROUND TRUTH COMPARISON")
    print("=" * 80)
    print()

    # Track metrics
    category_correct = 0
    total = 0
    results_table = []

    for gt_id, gt_info in ground_truth.get("proteins", {}).items():
        # Find matching vHold result
        short_id = gt_id.split("|")[1] if "|" in gt_id else gt_id

        vhold_result = None
        for result_id, result in vhold_results.items():
            if short_id in result_id or result_id in gt_id:
                vhold_result = result
                break

        if not vhold_result:
            print(f"WARNING: No vHold result for {gt_id}")
            continue

        total += 1

        # Compare categories
        gt_category = gt_info.get("category", "unknown")
        vhold_category = vhold_result.get("functional_category", "unknown")
        category_match = gt_category == vhold_category

        if category_match:
            category_correct += 1

        # Get other fields
        gt_name = gt_info.get("name", "Unknown")
        vhold_desc = vhold_result.get("description", "Unknown")
        confidence = vhold_result.get("confidence_level", "unknown")
        agreement = vhold_result.get("agreement", "none")

        results_table.append({
            "protein": gt_name,
            "gt_category": gt_category,
            "vhold_category": vhold_category,
            "match": "YES" if category_match else "NO",
            "confidence": confidence,
            "agreement": agreement,
            "vhold_description": vhold_desc
        })

    # Print comparison table
    print(f"{'Protein':<32} {'Expected':<12} {'vHold':<12} {'Match':<6} {'Confidence':<10}")
    print("-" * 80)

    for r in results_table:
        match_str = "✓" if r["match"] == "YES" else "✗"
        print(f"{r['protein'][:32]:<32} {r['gt_category']:<12} {r['vhold_category']:<12} "
              f"{match_str:<6} {r['confidence']:<10}")

    print("-" * 80)
    print()

    # Summary statistics
    accuracy = category_correct / total * 100 if total > 0 else 0
    print(f"CATEGORY ACCURACY: {category_correct}/{total} ({accuracy:.1f}%)")
    print()

    # Detailed descriptions
    print("=" * 80)
    print("ANNOTATION DETAILS")
    print("=" * 80)
    print()

    for r in results_table:
        match_icon = "✓" if r["match"] == "YES" else "✗"
        print(f"[{match_icon}] {r['protein']}")
        print(f"    Expected category: {r['gt_category']}")
        print(f"    vHold category: {r['vhold_category']}")
        print(f"    vHold description: {r['vhold_description']}")
        print(f"    Confidence: {r['confidence']}, Agreement: {r['agreement']}")
        print()

    # Category breakdown
    print("=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    print()

    category_stats = {}
    for r in results_table:
        cat = r["gt_category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if r["match"] == "YES":
            category_stats[cat]["correct"] += 1

    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        print(f"{cat:<12}: {stats['correct']}/{stats['total']} correct ({acc:.0f}%)")

    # Save comparison
    output_path = results_dir / "ground_truth_comparison.json"
    comparison = {
        "total_proteins": total,
        "category_accuracy": accuracy,
        "category_correct": category_correct,
        "per_protein": results_table,
        "per_category": category_stats
    }
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print()
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_ground_truth.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    case_study_dir = results_dir.parent

    compare_annotations(results_dir, case_study_dir)
