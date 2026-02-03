#!/usr/bin/env python3
"""
Analyze vHold results for Case Study 4: crAssphage ORFans.

Compares vHold annotations against ground truth from literature.
"""

import json
import sys
from pathlib import Path


def load_results(results_dir: Path) -> tuple[dict, list[dict]]:
    """Load summary JSON and parse TSV results."""
    summary_path = results_dir / "vhold_summary.json"
    tsv_path = results_dir / "vhold_results.tsv"

    with open(summary_path) as f:
        summary = json.load(f)

    results = []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if line.strip():
                values = line.strip().split("\t")
                results.append(dict(zip(header, values)))

    return summary, results


def load_ground_truth(path: Path) -> dict:
    """Load ground truth JSON."""
    with open(path) as f:
        return json.load(f)


def analyze(results_dir: Path, ground_truth_path: Path):
    """Analyze results against ground truth."""
    summary, results = load_results(results_dir)
    ground_truth = load_ground_truth(ground_truth_path)

    print("=" * 70)
    print("Case Study 4: crAssphage ORFans - Analysis")
    print("=" * 70)
    print()

    # Basic stats
    total = summary["statistics"]["total_proteins"]
    annotated = summary["statistics"]["annotated"]
    annotation_rate = summary["statistics"]["annotation_rate"]

    print(f"Total proteins: {total}")
    print(f"Annotated: {annotated} ({annotation_rate:.1%})")
    print()

    # Category distribution
    print("Functional Category Distribution:")
    category_dist = summary["statistics"]["category_distribution"]
    for cat, count in sorted(category_dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/total:.1%})")
    print()

    # Novelty distribution
    print("Novelty Distribution:")
    novelty_dist = summary["statistics"]["novelty_distribution"]
    for level, count in sorted(novelty_dist.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {level}: {count} ({count/annotated:.1%} of annotated)")
    print()

    # Compare to ground truth
    print("=" * 70)
    print("Ground Truth Comparison")
    print("=" * 70)
    print()

    gt_proteins = ground_truth["proteins"]
    correct_category = 0
    correct_function = 0
    annotated_with_gt = 0

    for r in results:
        query_id = r.get("query_id", "")
        # Extract gp number from query_id (e.g., "gp74|116aa|..." -> "gp74")
        gp_id = query_id.split("|")[0] if "|" in query_id else query_id

        if gp_id in gt_proteins:
            gt = gt_proteins[gp_id]
            vhold_cat = r.get("functional_category", "")
            vhold_desc = r.get("description", "").lower()

            if r.get("primary_source"):  # Has annotation
                annotated_with_gt += 1

                # Check category match
                if gt["expected_category"] == vhold_cat:
                    correct_category += 1

                # Check function keywords
                expected_func = gt["expected_function"].lower()
                keywords = expected_func.split()
                if any(kw in vhold_desc for kw in keywords if len(kw) > 3):
                    correct_function += 1

    if annotated_with_gt > 0:
        print(f"Proteins with ground truth that were annotated: {annotated_with_gt}")
        print(f"Correct category: {correct_category}/{annotated_with_gt} ({correct_category/annotated_with_gt:.1%})")
        print(f"Function keyword match: {correct_function}/{annotated_with_gt} ({correct_function/annotated_with_gt:.1%})")
    print()

    # Key protein analysis
    print("=" * 70)
    print("Key Protein Results")
    print("=" * 70)
    print()

    key_proteins = ["gp74", "gp50", "gp61", "gp64", "gp58", "gp14", "gp12"]

    for r in results:
        query_id = r.get("query_id", "")
        gp_id = query_id.split("|")[0] if "|" in query_id else query_id

        if gp_id in key_proteins:
            desc = r.get("description", "")[:60]
            category = r.get("functional_category", "")
            novelty = r.get("novelty", "")
            identity = r.get("primary_identity", "")
            source = r.get("primary_source", "no hit")

            gt_func = gt_proteins.get(gp_id, {}).get("expected_function", "?")

            print(f"{gp_id} (expected: {gt_func})")
            print(f"  vHold: {desc}")
            print(f"  Category: {category}, Novelty: {novelty}, Identity: {identity}")
            print(f"  Source: {source}")
            print()

    # ORFan discoveries
    print("=" * 70)
    print("ORFan Discoveries")
    print("=" * 70)
    print()

    orfan_list = ground_truth.get("orfans", {}).get("proteins", [])
    discovered = 0

    for r in results:
        query_id = r.get("query_id", "")
        gp_id = query_id.split("|")[0] if "|" in query_id else query_id

        if gp_id in orfan_list and r.get("primary_source"):
            discovered += 1
            desc = r.get("description", "")[:50]
            category = r.get("functional_category", "")
            print(f"  {gp_id}: {desc} [{category}]")

    print()
    print(f"ORFans with new annotations: {discovered}/{len(orfan_list)}")
    print()

    # Success criteria
    print("=" * 70)
    print("Success Criteria Evaluation")
    print("=" * 70)
    print()

    # Calculate structural protein accuracy
    structural_proteins = ground_truth["expected_annotations"]["structural_proteins"]
    structural_correct = 0
    structural_annotated = 0

    for r in results:
        query_id = r.get("query_id", "")
        gp_id = query_id.split("|")[0] if "|" in query_id else query_id

        if gp_id in structural_proteins:
            if r.get("primary_source"):
                structural_annotated += 1
                if r.get("functional_category") == "structural":
                    structural_correct += 1

    structural_accuracy = structural_correct / structural_annotated if structural_annotated > 0 else 0

    criteria = [
        ("Annotation rate >=60%", annotation_rate >= 0.60, f"{annotation_rate:.1%}"),
        ("Category accuracy >=50%", correct_category / annotated_with_gt >= 0.50 if annotated_with_gt > 0 else False,
         f"{correct_category/annotated_with_gt:.1%}" if annotated_with_gt > 0 else "N/A"),
        ("Structural proteins >=70%", structural_accuracy >= 0.70, f"{structural_accuracy:.1%}"),
        ("ORFan discoveries", discovered > 0, f"{discovered} found"),
    ]

    all_pass = True
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status} ({value})")

    print()
    print(f"Overall: {'ALL CRITERIA MET' if all_pass else 'SOME CRITERIA NOT MET'}")
    print()


def main():
    """Main entry point."""
    results_dir = Path(__file__).parent / "results"
    ground_truth_path = Path(__file__).parent / "ground_truth.json"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run vHold first: vhold run -i crass_orfans.fasta -o results/")
        sys.exit(1)

    if not (results_dir / "vhold_summary.json").exists():
        print("Error: vHold results not found. Pipeline may still be running.")
        print("Check progress: tail -f /private/tmp/claude/.../tasks/bf52c2a.output")
        sys.exit(1)

    analyze(results_dir, ground_truth_path)


if __name__ == "__main__":
    main()
