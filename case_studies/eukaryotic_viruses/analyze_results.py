#!/usr/bin/env python3
"""
Analyze vHold results for eukaryotic virus proteins.

Evaluates:
1. Annotation rate across virus families
2. Functional category accuracy
3. Identity distribution (novelty assessment)
4. Special cases: accessory proteins and ORFans
"""

import json
import sys
from pathlib import Path


def load_ground_truth(path: Path) -> dict:
    """Load ground truth annotations."""
    with open(path) as f:
        return json.load(f)


def load_vhold_results(results_dir: Path) -> dict:
    """Load vHold consensus results."""
    tsv_path = results_dir / "vhold_results.tsv"
    if not tsv_path.exists():
        print(f"ERROR: Results file not found: {tsv_path}")
        sys.exit(1)

    results = {}
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            row = dict(zip(header, fields))
            # Extract protein ID from query_id (format: PROTEIN_ID|virus|gene|length)
            query_id = row.get("query_id", "")
            protein_id = query_id.split("|")[0] if "|" in query_id else query_id
            results[protein_id] = row
    return results


def classify_category(description: str, keywords: dict = None) -> str:
    """Classify protein based on description keywords."""
    if not description or description == "NA":
        return "unknown"

    desc_lower = description.lower()

    # Structural proteins
    structural_kw = [
        "capsid", "nucleocapsid", "coat", "envelope", "spike", "matrix",
        "glycoprotein", "membrane", "virion", "fusion", "attachment",
        "nucleoprotein", "structural", "shell", "tegument"
    ]

    # Replication proteins
    replication_kw = [
        "polymerase", "replicase", "helicase", "rdrp", "transcriptase",
        "primase", "phosphoprotein", "cofactor"
    ]

    # Regulatory proteins
    regulatory_kw = [
        "transcription", "activator", "repressor", "regulator",
        "interferon", "antagonist", "accessory"
    ]

    for kw in structural_kw:
        if kw in desc_lower:
            return "structural"

    for kw in replication_kw:
        if kw in desc_lower:
            return "replication"

    for kw in regulatory_kw:
        if kw in desc_lower:
            return "regulatory"

    return "unknown"


def analyze_by_virus_family(ground_truth: dict, results: dict) -> dict:
    """Analyze results grouped by virus family."""
    families = ground_truth.get("virus_families", {})
    proteins = ground_truth.get("proteins", {})

    family_stats = {}
    for family, viruses in families.items():
        family_proteins = [
            pid for pid, info in proteins.items()
            if info.get("virus") in viruses
        ]

        annotated = 0
        correct_category = 0

        for pid in family_proteins:
            result = results.get(pid, {})
            expected = proteins.get(pid, {})

            description = result.get("description", "")
            if description and description != "NA":
                annotated += 1

                # Check category
                expected_cat = expected.get("category", "unknown")
                predicted_cat = classify_category(description)
                if predicted_cat == expected_cat:
                    correct_category += 1

        family_stats[family] = {
            "total": len(family_proteins),
            "annotated": annotated,
            "annotation_rate": annotated / len(family_proteins) if family_proteins else 0,
            "correct_category": correct_category,
            "category_accuracy": correct_category / annotated if annotated else 0
        }

    return family_stats


def analyze_identity_distribution(results: dict) -> dict:
    """Analyze identity distribution of hits."""
    identity_bins = {
        "database_match": 0,  # >95%
        "close_homolog": 0,   # 70-95%
        "remote_homolog": 0,  # 30-70%
        "twilight_zone": 0    # <30%
    }

    identities = []
    for pid, row in results.items():
        try:
            identity = float(row.get("pident", 0))
            if identity > 0:
                identities.append(identity)

                if identity > 95:
                    identity_bins["database_match"] += 1
                elif identity > 70:
                    identity_bins["close_homolog"] += 1
                elif identity > 30:
                    identity_bins["remote_homolog"] += 1
                else:
                    identity_bins["twilight_zone"] += 1
        except (ValueError, TypeError):
            pass

    total = sum(identity_bins.values())
    return {
        "bins": identity_bins,
        "percentages": {k: v/total*100 if total else 0 for k, v in identity_bins.items()},
        "mean_identity": sum(identities) / len(identities) if identities else 0,
        "total_with_identity": total
    }


def analyze_special_cases(ground_truth: dict, results: dict) -> dict:
    """Analyze accessory proteins and ORFans."""
    proteins = ground_truth.get("proteins", {})

    # Accessory/regulatory proteins (V, C, VP24, VP30, VP35)
    accessory = {}
    accessory_ids = [
        pid for pid, info in proteins.items()
        if info.get("category") == "regulatory"
    ]

    for pid in accessory_ids:
        result = results.get(pid, {})
        info = proteins[pid]
        accessory[pid] = {
            "virus": info.get("virus"),
            "gene": info.get("gene"),
            "expected": info.get("function"),
            "vhold_description": result.get("description", "NA"),
            "annotated": result.get("description", "") not in ["", "NA"]
        }

    # Hantavirus ORFans (labeled hypothetical in NCBI)
    orfans = {}
    orfan_ids = ["NP_941977.1", "NP_941978.1"]

    for pid in orfan_ids:
        result = results.get(pid, {})
        info = proteins.get(pid, {})
        orfans[pid] = {
            "virus": info.get("virus"),
            "expected": info.get("function"),
            "ncbi_annotation": "hypothetical protein",
            "vhold_description": result.get("description", "NA"),
            "annotated": result.get("description", "") not in ["", "NA"],
            "correct_structure": any(
                kw in result.get("description", "").lower()
                for kw in ["nucleocapsid", "nucleoprotein", "glycoprotein", "capsid"]
            )
        }

    return {
        "accessory_proteins": accessory,
        "accessory_annotation_rate": sum(1 for v in accessory.values() if v["annotated"]) / len(accessory) if accessory else 0,
        "orfans": orfans,
        "orfan_annotation_rate": sum(1 for v in orfans.values() if v["annotated"]) / len(orfans) if orfans else 0,
        "orfan_correct_rate": sum(1 for v in orfans.values() if v["correct_structure"]) / len(orfans) if orfans else 0
    }


def main():
    """Run the analysis."""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    ground_truth_path = script_dir / "ground_truth.json"

    # Check for results
    if not results_dir.exists() or not (results_dir / "vhold_results.tsv").exists():
        print("ERROR: vHold results not found. Run vHold first:")
        print("  vhold run -i test_proteins.fasta -o results/ -t 4")
        sys.exit(1)

    # Load data
    print("Loading data...")
    ground_truth = load_ground_truth(ground_truth_path)
    results = load_vhold_results(results_dir)

    print(f"Ground truth: {len(ground_truth['proteins'])} proteins")
    print(f"vHold results: {len(results)} proteins")

    # Overall annotation rate
    proteins = ground_truth["proteins"]
    annotated = sum(
        1 for pid in proteins
        if results.get(pid, {}).get("description", "") not in ["", "NA"]
    )

    print("\n" + "="*60)
    print("CASE STUDY 5: EUKARYOTIC VIRUS PROTEINS")
    print("="*60)

    print(f"\n## Overall Annotation Rate")
    print(f"  Annotated: {annotated}/{len(proteins)} ({100*annotated/len(proteins):.1f}%)")

    # By virus family
    print(f"\n## Results by Virus Family")
    family_stats = analyze_by_virus_family(ground_truth, results)
    print(f"\n{'Family':<20} {'Annotated':<15} {'Category Accuracy':<20}")
    print("-"*55)
    for family, stats in sorted(family_stats.items()):
        ann = f"{stats['annotated']}/{stats['total']} ({100*stats['annotation_rate']:.0f}%)"
        cat = f"{stats['correct_category']}/{stats['annotated']} ({100*stats['category_accuracy']:.0f}%)" if stats['annotated'] else "N/A"
        print(f"{family:<20} {ann:<15} {cat:<20}")

    # Identity distribution
    print(f"\n## Identity Distribution (Novelty)")
    identity_stats = analyze_identity_distribution(results)
    print(f"  Mean identity: {identity_stats['mean_identity']:.1f}%")
    print(f"  Distribution:")
    for bin_name, count in identity_stats['bins'].items():
        pct = identity_stats['percentages'][bin_name]
        print(f"    {bin_name:<15}: {count:3d} ({pct:5.1f}%)")

    # Special cases
    print(f"\n## Special Cases")
    special = analyze_special_cases(ground_truth, results)

    print(f"\n### Accessory/Regulatory Proteins")
    print(f"  Annotation rate: {100*special['accessory_annotation_rate']:.0f}%")
    for pid, info in special['accessory_proteins'].items():
        status = "+" if info['annotated'] else "-"
        desc = info['vhold_description'][:50] + "..." if len(info['vhold_description']) > 50 else info['vhold_description']
        print(f"  [{status}] {info['gene']:<5} ({info['virus']:<15}): {desc}")

    print(f"\n### Hantavirus ORFans (NCBI: 'hypothetical')")
    print(f"  Annotation rate: {100*special['orfan_annotation_rate']:.0f}%")
    print(f"  Correct structure: {100*special['orfan_correct_rate']:.0f}%")
    for pid, info in special['orfans'].items():
        status = "+" if info['correct_structure'] else ("~" if info['annotated'] else "-")
        desc = info['vhold_description'][:50] + "..." if len(info['vhold_description']) > 50 else info['vhold_description']
        print(f"  [{status}] {info['expected']}: {desc}")

    # Detailed protein-by-protein results
    print(f"\n## Detailed Results")
    print(f"\n{'Protein':<15} {'Virus':<15} {'Gene':<6} {'Expected':<20} {'vHold':<30}")
    print("-"*90)

    for pid, info in proteins.items():
        result = results.get(pid, {})
        desc = result.get("description", "NA")
        desc_short = desc[:27] + "..." if len(desc) > 30 else desc
        expected = info.get("function", "?")[:17] + "..." if len(info.get("function", "")) > 20 else info.get("function", "?")

        status = "+" if desc and desc != "NA" else "-"
        print(f"[{status}] {pid:<12} {info['virus']:<15} {info['gene']:<6} {expected:<20} {desc_short:<30}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Calculate category accuracy
    correct_categories = 0
    total_annotated = 0
    for pid, info in proteins.items():
        result = results.get(pid, {})
        desc = result.get("description", "")
        if desc and desc != "NA":
            total_annotated += 1
            predicted = classify_category(desc)
            if predicted == info.get("category"):
                correct_categories += 1

    print(f"""
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Annotation rate | {100*annotated/len(proteins):.0f}% | >70% | {'PASS' if annotated/len(proteins) > 0.7 else 'FAIL'} |
| Category accuracy | {100*correct_categories/total_annotated:.0f}% | >60% | {'PASS' if correct_categories/total_annotated > 0.6 else 'FAIL'} |
| Accessory proteins | {100*special['accessory_annotation_rate']:.0f}% | >50% | {'PASS' if special['accessory_annotation_rate'] > 0.5 else 'FAIL'} |
| Hantavirus ORFans | {sum(1 for v in special['orfans'].values() if v['annotated'])}/2 | 2/2 | {'PASS' if all(v['annotated'] for v in special['orfans'].values()) else 'FAIL'} |
| Novel hits (remote+twilight) | {identity_stats['percentages']['remote_homolog'] + identity_stats['percentages']['twilight_zone']:.0f}% | - | {'HIGH VALUE' if identity_stats['percentages']['remote_homolog'] + identity_stats['percentages']['twilight_zone'] > 50 else 'MODERATE'} |
""")

    # Save JSON summary
    summary = {
        "case_study": "Eukaryotic Virus Proteins",
        "total_proteins": len(proteins),
        "annotated": annotated,
        "annotation_rate": annotated / len(proteins),
        "category_accuracy": correct_categories / total_annotated if total_annotated else 0,
        "identity_distribution": identity_stats,
        "family_stats": family_stats,
        "special_cases": {
            "accessory_annotation_rate": special['accessory_annotation_rate'],
            "orfan_annotation_rate": special['orfan_annotation_rate'],
            "orfan_correct_rate": special['orfan_correct_rate']
        }
    }

    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
