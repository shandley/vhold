#!/usr/bin/env python3
"""SARS-CoV-2 Proteome Annotation Case Study.

This script runs vHold on the complete SARS-CoV-2 proteome and
compares results against gold standard annotations.

Usage:
    python run_case_study.py --output results/
    python run_case_study.py --output results/ --device cuda
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Case study paths
CASE_STUDY_DIR = Path(__file__).parent
PROTEOME_FASTA = CASE_STUDY_DIR / "sars_cov_2_proteome.fasta"
GROUND_TRUTH = CASE_STUDY_DIR / "ground_truth.json"


def load_ground_truth() -> dict:
    """Load ground truth annotations."""
    with open(GROUND_TRUTH) as f:
        return json.load(f)


def run_vhold(output_dir: Path, device: str = "cpu", threads: int = 4) -> bool:
    """Run vHold on the SARS-CoV-2 proteome.

    Args:
        output_dir: Directory for vHold output
        device: Computing device (cpu, cuda, mps)
        threads: Number of CPU threads

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vhold", "run",
        "-i", str(PROTEOME_FASTA),
        "-o", str(output_dir),
        "--device", device,
        "-t", str(threads),
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running vHold:\n{result.stderr}")
            return False
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("Error: vhold command not found. Install with: pip install -e .")
        return False


def load_vhold_results(output_dir: Path) -> dict:
    """Load vHold results from TSV file.

    Args:
        output_dir: Directory containing vHold output

    Returns:
        Dict mapping protein_id to result dict
    """
    results_file = output_dir / "vhold_results.tsv"

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return {}

    results = {}
    with open(results_file) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            # Pad fields with empty strings if shorter than header
            while len(fields) < len(header):
                fields.append("")

            row = dict(zip(header, fields))
            protein_id = row.get("query_id", "")
            if protein_id:
                results[protein_id] = row

    return results


def evaluate_results(
    vhold_results: dict,
    ground_truth: dict,
) -> dict:
    """Evaluate vHold results against ground truth.

    Args:
        vhold_results: Dict mapping protein_id to vHold result
        ground_truth: Ground truth data with entries

    Returns:
        Evaluation metrics dict
    """
    gt_entries = {e["protein_id"]: e for e in ground_truth["entries"]}

    evaluation = {
        "total_proteins": len(gt_entries),
        "annotated": 0,
        "correct_category": 0,
        "incorrect_category": 0,
        "no_annotation": 0,
        "per_protein": [],
        "category_accuracy": {},
        "confusion_matrix": {},
    }

    # Category tracking
    categories = set()
    for entry in ground_truth["entries"]:
        categories.add(entry["true_category"])

    # Initialize confusion matrix
    for cat in categories:
        evaluation["confusion_matrix"][cat] = {c: 0 for c in categories}
        evaluation["confusion_matrix"][cat]["unknown"] = 0
    evaluation["confusion_matrix"]["unknown"] = {c: 0 for c in categories}
    evaluation["confusion_matrix"]["unknown"]["unknown"] = 0

    # Evaluate each protein
    for protein_id, gt_entry in gt_entries.items():
        vhold_result = vhold_results.get(protein_id, {})

        pred_category = vhold_result.get("functional_category", "unknown")
        true_category = gt_entry["true_category"]

        protein_eval = {
            "protein_id": protein_id,
            "gene": gt_entry.get("gene", ""),
            "true_category": true_category,
            "predicted_category": pred_category,
            "correct": pred_category == true_category,
            "description": vhold_result.get("description", "No annotation"),
            "consensus_score": float(vhold_result.get("consensus_score", 0)),
            "confidence_level": vhold_result.get("confidence_level", "none"),
        }

        # Update counts
        if pred_category != "unknown" and pred_category:
            evaluation["annotated"] += 1
            if pred_category == true_category:
                evaluation["correct_category"] += 1
            else:
                evaluation["incorrect_category"] += 1
        else:
            evaluation["no_annotation"] += 1

        # Update confusion matrix
        if true_category in evaluation["confusion_matrix"]:
            if pred_category in evaluation["confusion_matrix"][true_category]:
                evaluation["confusion_matrix"][true_category][pred_category] += 1
            else:
                evaluation["confusion_matrix"][true_category]["unknown"] += 1

        evaluation["per_protein"].append(protein_eval)

    # Calculate per-category accuracy
    for cat in categories:
        cat_proteins = [e for e in gt_entries.values() if e["true_category"] == cat]
        cat_correct = sum(
            1 for p in evaluation["per_protein"]
            if p["true_category"] == cat and p["correct"]
        )
        evaluation["category_accuracy"][cat] = {
            "total": len(cat_proteins),
            "correct": cat_correct,
            "accuracy": cat_correct / len(cat_proteins) if cat_proteins else 0,
        }

    # Overall metrics
    evaluation["annotation_rate"] = evaluation["annotated"] / evaluation["total_proteins"]
    evaluation["overall_accuracy"] = (
        evaluation["correct_category"] / evaluation["annotated"]
        if evaluation["annotated"] > 0 else 0
    )

    return evaluation


def generate_report(
    evaluation: dict,
    ground_truth: dict,
    output_dir: Path,
) -> Path:
    """Generate case study report.

    Args:
        evaluation: Evaluation metrics
        ground_truth: Ground truth data
        output_dir: Output directory

    Returns:
        Path to report file
    """
    report = {
        "case_study": "SARS-CoV-2 Proteome Annotation",
        "date": datetime.now().isoformat(),
        "ground_truth_source": ground_truth.get("source", ""),
        "reference_genome": ground_truth.get("reference_genome", ""),
        "summary": {
            "total_proteins": evaluation["total_proteins"],
            "annotated": evaluation["annotated"],
            "annotation_rate": f"{evaluation['annotation_rate']:.1%}",
            "correct_category": evaluation["correct_category"],
            "overall_accuracy": f"{evaluation['overall_accuracy']:.1%}",
        },
        "category_performance": evaluation["category_accuracy"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "per_protein_results": evaluation["per_protein"],
    }

    report_path = output_dir / "case_study_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate markdown summary
    md_report = generate_markdown_report(evaluation, ground_truth)
    md_path = output_dir / "case_study_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    return report_path


def generate_markdown_report(evaluation: dict, ground_truth: dict) -> str:
    """Generate markdown report.

    Args:
        evaluation: Evaluation metrics
        ground_truth: Ground truth data

    Returns:
        Markdown report string
    """
    md = f"""# SARS-CoV-2 Proteome Annotation Case Study

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Reference Genome**: {ground_truth.get("reference_genome", "NC_045512.2")}
**Ground Truth Source**: {ground_truth.get("source", "UniProt")}

## Summary

| Metric | Value |
|--------|-------|
| Total Proteins | {evaluation["total_proteins"]} |
| Annotated | {evaluation["annotated"]} ({evaluation["annotation_rate"]:.1%}) |
| Correct Category | {evaluation["correct_category"]} |
| Overall Accuracy | {evaluation["overall_accuracy"]:.1%} |

## Category Performance

| Category | Total | Correct | Accuracy |
|----------|-------|---------|----------|
"""

    for cat, metrics in sorted(evaluation["category_accuracy"].items()):
        md += f"| {cat} | {metrics['total']} | {metrics['correct']} | {metrics['accuracy']:.1%} |\n"

    md += """
## Per-Protein Results

| Protein | Gene | True Category | Predicted | Correct | Score |
|---------|------|---------------|-----------|---------|-------|
"""

    for result in evaluation["per_protein"]:
        correct_mark = "Yes" if result["correct"] else "No"
        md += f"| {result['protein_id'].split('|')[1]} | {result['gene']} | {result['true_category']} | {result['predicted_category']} | {correct_mark} | {result['consensus_score']:.2f} |\n"

    md += """
## Key Findings

### Structural Proteins
- Spike (S), Nucleocapsid (N), Membrane (M), Envelope (E)
- Expected: All classified as "structural"

### Replication Machinery
- RdRp (NSP12), Helicase (NSP13)
- Expected: Classified as "replication"

### Proteases
- Main protease (NSP5/Mpro/3CLpro), Papain-like protease (NSP3/PLpro)
- Expected: Classified as "protease"

### Accessory Proteins
- ORF3a, ORF6, ORF7a, ORF8, ORF9b
- Expected: Classified as "host_interaction" or "unknown"

## Methodology

1. **Input**: Complete SARS-CoV-2 proteome (18 proteins)
2. **Annotation**: vHold structural homology pipeline
3. **Databases**: BFVD and Viro3D
4. **Evaluation**: Comparison to UniProt reviewed annotations

## Conclusions

[To be filled based on actual results]
"""

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Run SARS-CoV-2 case study"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=CASE_STUDY_DIR / "results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Computing device"
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Number of CPU threads"
    )
    parser.add_argument(
        "--skip-vhold",
        action="store_true",
        help="Skip running vHold (use existing results)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SARS-CoV-2 Proteome Annotation Case Study")
    print("=" * 60)

    # Load ground truth
    print("\nLoading ground truth annotations...")
    ground_truth = load_ground_truth()
    print(f"  - {len(ground_truth['entries'])} proteins with annotations")
    print(f"  - Categories: {ground_truth['category_summary']}")

    # Run vHold
    if not args.skip_vhold:
        print("\nRunning vHold annotation pipeline...")
        if not run_vhold(args.output, args.device, args.threads):
            print("vHold failed. Use --skip-vhold to evaluate existing results.")
            sys.exit(1)
    else:
        print("\nSkipping vHold (using existing results)...")

    # Load results
    print("\nLoading vHold results...")
    vhold_results = load_vhold_results(args.output)
    if not vhold_results:
        print("No results to evaluate.")
        sys.exit(1)
    print(f"  - {len(vhold_results)} proteins processed")

    # Evaluate
    print("\nEvaluating results against ground truth...")
    evaluation = evaluate_results(vhold_results, ground_truth)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total proteins:     {evaluation['total_proteins']}")
    print(f"Annotated:          {evaluation['annotated']} ({evaluation['annotation_rate']:.1%})")
    print(f"Correct category:   {evaluation['correct_category']}")
    print(f"Overall accuracy:   {evaluation['overall_accuracy']:.1%}")

    print("\nPer-category accuracy:")
    for cat, metrics in sorted(evaluation['category_accuracy'].items()):
        print(f"  {cat:20s}: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")

    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(evaluation, ground_truth, args.output)
    print(f"  - JSON report: {report_path}")
    print(f"  - Markdown report: {args.output / 'case_study_report.md'}")

    print("\n" + "=" * 60)
    print("Case study complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
