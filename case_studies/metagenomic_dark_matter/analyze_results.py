#!/usr/bin/env python3
"""
Analyze vHold results for Case Study 3: Metagenomic Dark Matter.

Compares vHold annotations against ground truth (all sequences are RdRps).
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

    print("=" * 60)
    print("Case Study 3: Metagenomic Dark Matter - Analysis")
    print("=" * 60)
    print()

    # Basic stats
    total = summary["statistics"]["total_proteins"]
    annotated = summary["statistics"]["annotated"]
    annotation_rate = summary["statistics"]["annotation_rate"]

    print(f"Total proteins: {total}")
    print(f"Annotated: {annotated} ({annotation_rate:.1%})")
    print()

    # Category accuracy
    category_dist = summary["statistics"]["category_distribution"]
    replication_count = category_dist.get("replication", 0)
    expected_category = ground_truth["expected_annotations"]["functional_category"]

    print(f"Expected category: {expected_category}")
    print(f"Classified as '{expected_category}': {replication_count}/{total} ({replication_count/total:.1%})")
    print()

    # Novelty distribution
    print("Novelty Distribution:")
    novelty_dist = summary["statistics"]["novelty_distribution"]
    for level, count in sorted(novelty_dist.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {level}: {count} ({count/annotated:.1%} of annotated)")
    print()

    # RdRp keyword detection
    keywords = ground_truth["expected_annotations"]["description_keywords"]
    keyword_hits = 0
    for r in results:
        desc = r.get("description", "").lower()
        if any(kw.lower() in desc for kw in keywords):
            keyword_hits += 1

    print(f"RdRp keywords in description: {keyword_hits}/{annotated} ({keyword_hits/annotated:.1%} of annotated)")
    print()

    # Identity distribution for annotated proteins
    print("Identity Distribution of Hits:")
    identities = []
    for r in results:
        if r.get("primary_identity") and r["primary_identity"] != "":
            try:
                identities.append(float(r["primary_identity"]))
            except ValueError:
                pass

    if identities:
        print(f"  Min: {min(identities):.1%}")
        print(f"  Max: {max(identities):.1%}")
        print(f"  Mean: {sum(identities)/len(identities):.1%}")
        print(f"  Median: {sorted(identities)[len(identities)//2]:.1%}")
    print()

    # Dark matter analysis
    dark_matter = summary.get("dark_matter", {})
    if dark_matter:
        print("Dark Matter Analysis:")
        print(f"  Total dark matter: {dark_matter['total_dark_matter']} ({dark_matter['dark_matter_rate']:.1%})")
        print(f"  No hits: {dark_matter['by_category'].get('no_hits', 0)}")
        print(f"  Unknown function: {dark_matter['by_category'].get('unknown_function', 0)}")
    print()

    # Success criteria evaluation
    print("=" * 60)
    print("Success Criteria Evaluation")
    print("=" * 60)
    print()

    criteria = [
        ("Annotation rate >=80%", annotation_rate >= 0.80, f"{annotation_rate:.1%}"),
        ("Category accuracy >=70%", replication_count / total >= 0.70, f"{replication_count/total:.1%}"),
        ("RdRp keywords >=50%", keyword_hits / annotated >= 0.50, f"{keyword_hits/annotated:.1%}"),
        (
            "Novelty at remote/twilight",
            (novelty_dist.get("remote_homolog", 0) + novelty_dist.get("twilight_zone", 0)) / annotated > 0.50,
            f"{(novelty_dist.get('remote_homolog', 0) + novelty_dist.get('twilight_zone', 0))/annotated:.1%}",
        ),
    ]

    all_pass = True
    for name, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status} ({value})")

    print()
    print(f"Overall: {'ALL CRITERIA MET' if all_pass else 'SOME CRITERIA FAILED'}")
    print()

    # Sample annotated proteins
    print("=" * 60)
    print("Sample Annotations")
    print("=" * 60)
    print()

    for r in results[:5]:
        query_id = r.get("query_id", "")
        desc = r.get("description", "")[:50]
        category = r.get("functional_category", "")
        novelty = r.get("novelty", "")
        identity = r.get("primary_identity", "")
        print(f"{query_id}: {desc}")
        print(f"  Category: {category}, Novelty: {novelty}, Identity: {identity}")
        print()


def main():
    """Main entry point."""
    results_dir = Path(__file__).parent / "results"
    ground_truth_path = Path(__file__).parent / "ground_truth.json"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run vHold first: vhold run -i test_palmprints.fasta -o results/")
        sys.exit(1)

    analyze(results_dir, ground_truth_path)


if __name__ == "__main__":
    main()
