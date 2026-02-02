#!/usr/bin/env python3
"""Analyze vHold results stratified by sequence identity.

This script demonstrates vHold's ability to annotate proteins
at low sequence identity where BLAST would fail.

Usage:
    python analyze_identity.py /path/to/vhold/results/
    python analyze_identity.py ../sars_cov_2/results/
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd


# Foldseek output columns
FOLDSEEK_COLUMNS = [
    "query", "target", "fident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
    "qlen", "tlen", "qcov", "tcov"
]

# Identity bins for stratification
IDENTITY_BINS = {
    "twilight": (0.0, 0.2),    # <20% - BLAST completely fails
    "remote": (0.2, 0.3),      # 20-30% - BLAST unreliable
    "moderate": (0.3, 0.5),    # 30-50% - BLAST marginal
    "easy": (0.5, 1.01),       # >50% - BLAST works
}


def load_foldseek_hits(results_dir: Path) -> pd.DataFrame:
    """Load and combine foldseek hits from all databases."""
    hits = []

    for db in ["bfvd", "viro3d"]:
        hits_file = results_dir / "foldseek" / f"{db}_hits.tsv"
        if hits_file.exists():
            df = pd.read_csv(hits_file, sep="\t", header=None, names=FOLDSEEK_COLUMNS)
            df["database"] = db
            hits.append(df)

    if not hits:
        return pd.DataFrame()

    return pd.concat(hits, ignore_index=True)


def get_identity_bin(fident: float) -> str:
    """Classify sequence identity into bins."""
    for bin_name, (low, high) in IDENTITY_BINS.items():
        if low <= fident < high:
            return bin_name
    return "unknown"


def analyze_identity_distribution(hits: pd.DataFrame) -> dict:
    """Analyze hit distribution by sequence identity."""
    if hits.empty:
        return {}

    # Add identity bin
    hits = hits.copy()
    hits["identity_bin"] = hits["fident"].apply(get_identity_bin)

    # Get best hit per query (lowest E-value)
    best_hits = hits.loc[hits.groupby("query")["evalue"].idxmin()]

    results = {
        "total_queries": len(best_hits),
        "total_hits": len(hits),
        "bins": {},
        "examples": [],
    }

    # Analyze each bin
    for bin_name in IDENTITY_BINS.keys():
        bin_hits = best_hits[best_hits["identity_bin"] == bin_name]
        all_bin_hits = hits[hits["identity_bin"] == bin_name]

        results["bins"][bin_name] = {
            "queries_with_best_hit": len(bin_hits),
            "total_hits": len(all_bin_hits),
            "mean_identity": float(bin_hits["fident"].mean()) if len(bin_hits) > 0 else 0,
            "mean_evalue": float(bin_hits["evalue"].mean()) if len(bin_hits) > 0 else 0,
            "identity_range": IDENTITY_BINS[bin_name],
        }

        # Collect examples of remote homology
        if bin_name in ("twilight", "remote") and len(bin_hits) > 0:
            for _, row in bin_hits.head(3).iterrows():
                results["examples"].append({
                    "query": row["query"],
                    "target": row["target"],
                    "database": row["database"],
                    "identity": f"{row['fident']:.1%}",
                    "evalue": f"{row['evalue']:.2e}",
                    "bin": bin_name,
                })

    return results


def load_vhold_results(results_dir: Path) -> pd.DataFrame:
    """Load vHold annotation results."""
    results_file = results_dir / "vhold_results.tsv"
    if not results_file.exists():
        return pd.DataFrame()

    return pd.read_csv(results_file, sep="\t")


def print_report(analysis: dict, vhold_results: pd.DataFrame):
    """Print formatted analysis report."""
    print("=" * 70)
    print("SEQUENCE IDENTITY STRATIFICATION ANALYSIS")
    print("=" * 70)
    print()
    print(f"Total queries with hits: {analysis['total_queries']}")
    print(f"Total hits across all databases: {analysis['total_hits']}")
    print()

    print("IDENTITY BIN DISTRIBUTION")
    print("-" * 70)
    print(f"{'Bin':<12} {'Identity Range':<15} {'Queries':<10} {'Mean ID':<10} {'Mean E-value':<12}")
    print("-" * 70)

    for bin_name in ["easy", "moderate", "remote", "twilight"]:
        if bin_name not in analysis["bins"]:
            continue
        bin_data = analysis["bins"][bin_name]
        low, high = bin_data["identity_range"]
        print(f"{bin_name:<12} {low:.0%}-{high:.0%}{'':>7} {bin_data['queries_with_best_hit']:<10} "
              f"{bin_data['mean_identity']:.1%}{'':>5} {bin_data['mean_evalue']:.2e}")

    print()

    # Calculate remote homology statistics
    remote_twilight = 0
    for bin_name in ["remote", "twilight"]:
        if bin_name in analysis["bins"]:
            remote_twilight += analysis["bins"][bin_name]["queries_with_best_hit"]

    if analysis["total_queries"] > 0:
        remote_pct = remote_twilight / analysis["total_queries"] * 100
        print(f"REMOTE HOMOLOGY DETECTION: {remote_twilight}/{analysis['total_queries']} "
              f"({remote_pct:.1f}%) best hits at <30% identity")
    print()

    # Show examples of remote homology
    if analysis["examples"]:
        print("EXAMPLES OF REMOTE HOMOLOGY (<30% identity)")
        print("-" * 70)
        for ex in analysis["examples"]:
            print(f"  {ex['query']}")
            print(f"    → {ex['target']} ({ex['database']})")
            print(f"    → Identity: {ex['identity']}, E-value: {ex['evalue']}")
            print()

    # Cross-reference with annotations
    if not vhold_results.empty:
        print("ANNOTATION SUCCESS BY IDENTITY")
        print("-" * 70)
        annotated = vhold_results[vhold_results["description"] != "hypothetical protein"]
        print(f"Total annotated: {len(annotated)}/{len(vhold_results)}")

        if "primary_identity" in vhold_results.columns:
            for _, row in annotated.iterrows():
                try:
                    identity = float(row["primary_identity"])
                    bin_name = get_identity_bin(identity)
                    if bin_name in ("remote", "twilight"):
                        print(f"  ✓ {row['query_id']}: {row['description'][:40]}...")
                        print(f"    Identity: {identity:.1%} ({bin_name})")
                except (ValueError, TypeError):
                    pass

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Proteins annotated at <30% sequence identity demonstrate vHold's ability
to detect remote homology that BLAST would miss. Structure is 3-10× more
conserved than sequence during evolution, enabling functional annotation
of divergent viral proteins.

Key metrics:
- "remote" + "twilight" bins: Proteins where BLAST fails, structure succeeds
- High E-value significance at low identity: Structural similarity is real
- Functional annotations transferred: Structure-based annotation works
""")


def save_report(analysis: dict, output_path: Path):
    """Save analysis to JSON."""
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vHold results by sequence identity"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to vHold results directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file for analysis"
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Load data
    print(f"Loading results from: {args.results_dir}")
    hits = load_foldseek_hits(args.results_dir)

    if hits.empty:
        print("Error: No foldseek hits found")
        sys.exit(1)

    vhold_results = load_vhold_results(args.results_dir)

    # Analyze
    analysis = analyze_identity_distribution(hits)

    # Report
    print_report(analysis, vhold_results)

    # Save
    if args.output:
        save_report(analysis, args.output)
    else:
        default_output = args.results_dir / "identity_analysis.json"
        save_report(analysis, default_output)


if __name__ == "__main__":
    main()
