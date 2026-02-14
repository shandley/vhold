#!/usr/bin/env python3
"""Calibrate the embedding triage threshold for vHold.

Loads the pre-computed embedding database, extracts query embeddings for
proteins across 4 case studies, sweeps cosine similarity thresholds, and
compares annotation transfer against Foldseek pipeline results and ground
truth.

Usage:
    uv run python scripts/calibrate_triage_threshold.py --device cpu
    uv run python scripts/calibrate_triage_threshold.py --device cpu --output calibration_results.tsv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vhold.databases.embeddings import get_embedding_db_path
from vhold.features.embeddings import (
    EmbeddingDatabase,
    EmbeddingExtractor,
    _lookup_annotation,
    _metadata_state,
)
from vhold.io.fasta import read_fasta
from vhold.results.categories import AnnotationEvidence, classify_protein


# ============================================================================
# Case study definitions
# ============================================================================

CASE_STUDIES = [
    {
        "name": "CS1: SARS-CoV-2",
        "fasta": "case_studies/sars_cov_2/sars_cov_2_proteome.fasta",
        "results_tsv": "case_studies/sars_cov_2/results/vhold_results.tsv",
        "ground_truth": "case_studies/sars_cov_2/ground_truth.json",
        "gt_format": "entries",  # entries list with protein_id, true_category
    },
    {
        "name": "CS2: Remote Homology",
        "fasta": "case_studies/remote_homology/divergent_proteins.fasta",
        "results_tsv": "case_studies/remote_homology/results/vhold_results.tsv",
        "ground_truth": "case_studies/remote_homology/ground_truth.json",
        "gt_format": "proteins_category",  # proteins dict with category
    },
    {
        "name": "CS3: Dark Matter",
        "fasta": "case_studies/metagenomic_dark_matter/test_palmprints.fasta",
        "results_tsv": "case_studies/metagenomic_dark_matter/results/vhold_results.tsv",
        "ground_truth": "case_studies/metagenomic_dark_matter/ground_truth.json",
        "gt_format": "proteins_expected",  # proteins dict with expected_category
    },
    {
        "name": "CS5: Eukaryotic Viruses",
        "fasta": "case_studies/eukaryotic_viruses/test_proteins_small.fasta",
        "results_tsv": "case_studies/eukaryotic_viruses/results/vhold_results.tsv",
        "ground_truth": "case_studies/eukaryotic_viruses/ground_truth.json",
        "gt_format": "proteins_category",  # proteins dict with category
    },
]

THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]


# ============================================================================
# Ground truth loading
# ============================================================================


def _normalize_id(query_id: str, gt_keys: set[str]) -> str | None:
    """Try to match a FASTA query ID to a ground truth key.

    Handles format mismatches like:
    - FASTA: "NP_112021.1|Nipah" → GT: "NP_112021.1"
    - FASTA: "sp|P0DTC2|SPIKE_SARS2" → GT: "sp|P0DTC2|SPIKE_SARS2"
    - FASTA: "u1000001" → GT: "u1000001"
    """
    # Direct match
    if query_id in gt_keys:
        return query_id

    # Try first pipe-delimited field (e.g., "NP_112021.1|Nipah" → "NP_112021.1")
    first_field = query_id.split("|")[0]
    if first_field in gt_keys:
        return first_field

    # Try prefix match for versioned accessions (e.g., "NP_112021.1")
    for gt_key in gt_keys:
        if query_id.startswith(gt_key) or gt_key.startswith(query_id):
            return gt_key

    return None


def load_ground_truth(gt_path: Path, gt_format: str) -> dict[str, str]:
    """Load ground truth categories from JSON.

    Returns:
        Dict mapping protein_id to expected functional category.
    """
    with open(gt_path) as f:
        data = json.load(f)

    gt = {}
    if gt_format == "entries":
        for entry in data["entries"]:
            gt[entry["protein_id"]] = entry["true_category"]
    elif gt_format == "proteins_category":
        for pid, info in data["proteins"].items():
            gt[pid] = info["category"]
    elif gt_format == "proteins_expected":
        for pid, info in data["proteins"].items():
            gt[pid] = info["expected_category"]
    else:
        raise ValueError(f"Unknown ground truth format: {gt_format}")

    return gt


def load_foldseek_results(tsv_path: Path) -> dict[str, dict]:
    """Load Foldseek pipeline results from TSV.

    Returns:
        Dict mapping query_id to {description, functional_category}.
    """
    results = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results[row["query_id"]] = {
                "description": row.get("description", ""),
                "functional_category": row.get("functional_category", "unknown"),
            }
    return results


# ============================================================================
# Calibration logic
# ============================================================================


def get_triage_annotation(
    target_id: str, source_db: str, db_dir: Path | None
) -> tuple[str, str, dict]:
    """Get annotation and category for a triage match.

    Returns:
        Tuple of (description, functional_category, full_annotation_dict).
    """
    annotation = _lookup_annotation(target_id, source_db, db_dir, _metadata_cache={})
    description = annotation.get("description", "hypothetical protein")
    gene = annotation.get("gene")

    # Build evidence from enriched annotation fields
    evidence = AnnotationEvidence(
        pfam=annotation.get("pfam"),
        go_bp=annotation.get("go_bp"),
        go_mf=annotation.get("go_mf"),
        superfamily=annotation.get("superfamily"),
    )
    category = classify_protein(description, gene, evidence=evidence)
    return description, category, annotation


def evaluate_threshold(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    embedding_db: EmbeddingDatabase,
    threshold: float,
    ground_truth: dict[str, str],
    foldseek_results: dict[str, dict],
    db_dir: Path | None,
    llm_client=None,
    llm_model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Evaluate a single threshold against ground truth and Foldseek.

    Returns dict with:
        matched: number of proteins with embedding match
        correct_vs_gt: matches where triage category == ground truth category
        wrong_vs_gt: matches where triage category != ground truth category
        correct_vs_foldseek: matches where triage category == Foldseek category
        wrong_vs_foldseek: matches where triage != Foldseek
        false_positives: list of (query_id, triage_cat, gt_cat, similarity, description)
        llm_reclassified: number of proteins reclassified by LLM
    """
    matches = embedding_db.search(
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        threshold=threshold,
        top_k=1,
    )

    correct_gt = 0
    wrong_gt = 0
    correct_fs = 0
    wrong_fs = 0
    false_positives = []
    llm_reclassified = 0

    gt_keys = set(ground_truth.keys())
    fs_keys = set(foldseek_results.keys())

    for qid, match_list in matches.items():
        if not match_list:
            continue
        best = match_list[0]
        description, triage_cat, annotation = get_triage_annotation(
            best.target_id, best.source_db, db_dir
        )

        # LLM reclassification for unknowns
        if triage_cat == "unknown" and llm_client is not None:
            from vhold.results.llm_classify import classify_single
            organism = annotation.get("organism")
            if not organism and "|" in qid:
                parts = qid.split("|")
                if len(parts) >= 2:
                    organism = parts[1]
            gene = annotation.get("gene")
            try:
                llm_cat, reasoning = classify_single(
                    client=llm_client,
                    description=description,
                    gene=gene,
                    organism=organism,
                    model=llm_model,
                )
                if llm_cat != "unknown":
                    triage_cat = llm_cat
                    llm_reclassified += 1
            except Exception as e:
                pass  # Keep keyword classification on LLM error

        # Compare vs ground truth (with ID normalization)
        gt_key = _normalize_id(qid, gt_keys)
        gt_cat = ground_truth.get(gt_key) if gt_key else None
        if gt_cat:
            if triage_cat == gt_cat:
                correct_gt += 1
            else:
                wrong_gt += 1
                false_positives.append(
                    (qid, triage_cat, gt_cat, best.similarity, description)
                )

        # Compare vs Foldseek (with ID normalization)
        fs_key = _normalize_id(qid, fs_keys)
        fs = foldseek_results.get(fs_key) if fs_key else None
        if fs:
            if triage_cat == fs["functional_category"]:
                correct_fs += 1
            else:
                wrong_fs += 1

    return {
        "matched": len(matches),
        "correct_vs_gt": correct_gt,
        "wrong_vs_gt": wrong_gt,
        "correct_vs_foldseek": correct_fs,
        "wrong_vs_foldseek": wrong_fs,
        "false_positives": false_positives,
        "llm_reclassified": llm_reclassified,
    }


# ============================================================================
# Similarity distribution
# ============================================================================


def get_max_similarities(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    embedding_db: EmbeddingDatabase,
) -> dict[str, float]:
    """Get maximum similarity for each query protein (no threshold)."""
    similarities = query_embeddings @ embedding_db.embeddings.T
    result = {}
    for i, qid in enumerate(query_ids):
        result[qid] = float(np.max(similarities[i]))
    return result


def print_similarity_histogram(all_sims: dict[str, float]) -> None:
    """Print ASCII histogram of max similarity distribution."""
    values = list(all_sims.values())
    bins = [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99, 1.01]
    counts, _ = np.histogram(values, bins=bins)

    print("\n--- Similarity Distribution (max similarity per query) ---")
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        lo = bins[i]
        hi = bins[i + 1]
        bar = "#" * int(40 * counts[i] / max_count)
        label = f"[{lo:.2f}-{hi:.2f})"
        print(f"  {label:14s} | {bar:40s} {counts[i]:3d}")

    print(f"\n  Total proteins: {len(values)}")
    print(f"  Min: {min(values):.4f}  Median: {np.median(values):.4f}  Max: {max(values):.4f}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate embedding triage threshold for vHold"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for ProstT5 encoder inference",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="ProstT5 model cache directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV file for detailed results (optional)",
    )
    parser.add_argument(
        "--llm-classify",
        action="store_true",
        help="Use Claude to reclassify 'unknown' triage results (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--llm-model",
        default="claude-haiku-4-5-20251001",
        help="Claude model for LLM reclassification",
    )
    args = parser.parse_args()

    # Load embedding database
    db_path = get_embedding_db_path()
    if not db_path.exists():
        print(f"ERROR: Embedding database not found at {db_path}")
        print("Run scripts/build_embedding_db.py first.")
        sys.exit(1)

    print(f"Loading embedding database from {db_path}")
    embedding_db = EmbeddingDatabase(db_path)
    embedding_db.load()
    print(f"  {embedding_db.size:,} reference proteins loaded")

    # Initialize ProstT5 encoder
    print(f"\nInitializing ProstT5 encoder on {args.device}")
    extractor = EmbeddingExtractor(
        device=args.device,
        model_dir=args.model_dir,
        encoder_only=True,
    )
    extractor.load_model()
    print("  Encoder ready")

    # Initialize LLM client if requested
    llm_client = None
    if args.llm_classify:
        import os
        try:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if key:
                llm_client = anthropic.Anthropic(api_key=key)
                print(f"\n  LLM reclassification enabled (model: {args.llm_model})")
            else:
                print("\n  WARNING: --llm-classify requested but ANTHROPIC_API_KEY not set")
        except ImportError:
            print("\n  WARNING: --llm-classify requested but anthropic package not installed")

    # Process each case study
    all_per_threshold = defaultdict(lambda: defaultdict(int))
    all_false_positives = defaultdict(list)
    all_max_sims = {}
    per_cs_results = {}

    total_proteins = 0

    for cs in CASE_STUDIES:
        fasta_path = PROJECT_ROOT / cs["fasta"]
        results_path = PROJECT_ROOT / cs["results_tsv"]
        gt_path = PROJECT_ROOT / cs["ground_truth"]

        if not fasta_path.exists():
            print(f"\n  SKIP {cs['name']}: FASTA not found at {fasta_path}")
            continue
        if not results_path.exists():
            print(f"\n  SKIP {cs['name']}: Results not found at {results_path}")
            continue
        if not gt_path.exists():
            print(f"\n  SKIP {cs['name']}: Ground truth not found at {gt_path}")
            continue

        print(f"\n--- {cs['name']} ---")

        # Load data
        records = read_fasta(fasta_path)
        sequences = {rid: rec.sequence for rid, rec in records.items()}
        ground_truth = load_ground_truth(gt_path, cs["gt_format"])
        foldseek_results = load_foldseek_results(results_path)

        print(f"  {len(sequences)} proteins, {len(ground_truth)} ground truth, {len(foldseek_results)} Foldseek results")
        total_proteins += len(sequences)

        # Extract embeddings
        ids, embeddings = extractor.extract_batch(
            sequences, batch_size=8, show_progress=True
        )

        # Get max similarities for distribution
        sims = get_max_similarities(ids, embeddings, embedding_db)
        all_max_sims.update(sims)

        # Evaluate each threshold
        cs_results = {}
        for thresh in THRESHOLDS:
            result = evaluate_threshold(
                ids, embeddings, embedding_db, thresh,
                ground_truth, foldseek_results, db_dir=None,
                llm_client=llm_client,
                llm_model=args.llm_model,
            )
            cs_results[thresh] = result

            # Accumulate into global totals
            for key in ["matched", "correct_vs_gt", "wrong_vs_gt", "correct_vs_foldseek", "wrong_vs_foldseek"]:
                all_per_threshold[thresh][key] += result[key]
            all_per_threshold[thresh]["total"] += len(ids)
            all_false_positives[thresh].extend(result["false_positives"])

        per_cs_results[cs["name"]] = {
            "n_proteins": len(ids),
            "results": cs_results,
        }

    # ========================================================================
    # Report: Overall results
    # ========================================================================

    print("\n" + "=" * 80)
    print("TRIAGE THRESHOLD CALIBRATION RESULTS")
    print("=" * 80)
    print(f"Total proteins: {total_proteins} across {len(per_cs_results)} case studies")
    print(f"Embedding DB: {embedding_db.size:,} references")

    print("\n--- Overall Results (vs Ground Truth) ---")
    print(f"{'Threshold':>10s} | {'Matched':>10s} | {'Correct':>8s} | {'Wrong':>6s} | {'Precision':>10s} | {'Recall':>8s} | {'F1':>6s}")
    print("-" * 75)

    best_f1 = 0
    best_thresh = None

    for thresh in THRESHOLDS:
        t = all_per_threshold[thresh]
        matched = t["matched"]
        total = t["total"]
        correct = t["correct_vs_gt"]
        wrong = t["wrong_vs_gt"]

        precision = correct / (correct + wrong) if (correct + wrong) > 0 else 1.0
        recall = matched / total if total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

        print(f"{thresh:>10.2f} | {matched:>5d}/{total:<4d} | {correct:>8d} | {wrong:>6d} | {precision:>10.3f} | {recall:>8.3f} | {f1:>6.3f}")

    # Also show vs Foldseek
    print("\n--- Overall Results (vs Foldseek Pipeline) ---")
    print(f"{'Threshold':>10s} | {'Matched':>10s} | {'Agree':>8s} | {'Disagree':>8s} | {'Agreement':>10s}")
    print("-" * 60)

    for thresh in THRESHOLDS:
        t = all_per_threshold[thresh]
        matched = t["matched"]
        total = t["total"]
        agree = t["correct_vs_foldseek"]
        disagree = t["wrong_vs_foldseek"]
        agreement = agree / (agree + disagree) if (agree + disagree) > 0 else 1.0
        print(f"{thresh:>10.2f} | {matched:>5d}/{total:<4d} | {agree:>8d} | {disagree:>8d} | {agreement:>10.3f}")

    # ========================================================================
    # Report: Per case study
    # ========================================================================

    print("\n--- Per Case Study Breakdown (vs Ground Truth) ---")
    for cs_name, cs_data in per_cs_results.items():
        n = cs_data["n_proteins"]
        print(f"\n  {cs_name} ({n} proteins):")
        print(f"  {'Threshold':>10s} | {'Matched':>10s} | {'Correct':>8s} | {'Wrong':>6s} | {'Precision':>10s}")
        print(f"  {'-' * 55}")
        for thresh in THRESHOLDS:
            r = cs_data["results"][thresh]
            m = r["matched"]
            c = r["correct_vs_gt"]
            w = r["wrong_vs_gt"]
            p = c / (c + w) if (c + w) > 0 else 1.0
            print(f"  {thresh:>10.2f} | {m:>5d}/{n:<4d} | {c:>8d} | {w:>6d} | {p:>10.3f}")

    # ========================================================================
    # Report: False positives
    # ========================================================================

    print("\n--- False Positive Details (wrong annotations) ---")
    for thresh in THRESHOLDS:
        fps = all_false_positives[thresh]
        if fps:
            print(f"\n  Threshold {thresh:.2f} ({len(fps)} false positives):")
            for qid, triage_cat, gt_cat, sim, desc in fps:
                print(f"    {qid}")
                print(f"      Triage: {triage_cat} ({desc[:60]})")
                print(f"      Truth:  {gt_cat}  |  Similarity: {sim:.4f}")

    # ========================================================================
    # Report: Similarity distribution
    # ========================================================================

    print_similarity_histogram(all_max_sims)

    # ========================================================================
    # Recommendation
    # ========================================================================

    # Find threshold with precision >= 0.95 and maximum recall
    recommended = None
    best_recall_at_high_precision = 0
    for thresh in sorted(THRESHOLDS):
        t = all_per_threshold[thresh]
        correct = t["correct_vs_gt"]
        wrong = t["wrong_vs_gt"]
        matched = t["matched"]
        total = t["total"]
        precision = correct / (correct + wrong) if (correct + wrong) > 0 else 1.0
        recall = matched / total if total > 0 else 0.0
        if precision >= 0.95 and recall > best_recall_at_high_precision:
            best_recall_at_high_precision = recall
            recommended = thresh

    print(f"\n{'=' * 80}")
    if recommended is not None:
        print(f"RECOMMENDED THRESHOLD: {recommended:.2f}")
        print(f"  (highest recall with precision >= 0.95)")
    else:
        print(f"BEST F1 THRESHOLD: {best_thresh:.2f} (F1={best_f1:.3f})")
        print(f"  (no threshold achieved precision >= 0.95)")
    print(f"{'=' * 80}")

    # Optional TSV output
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "threshold", "total", "matched", "correct_vs_gt", "wrong_vs_gt",
                "precision_gt", "recall", "f1",
                "agree_foldseek", "disagree_foldseek", "agreement_foldseek",
            ])
            for thresh in THRESHOLDS:
                t = all_per_threshold[thresh]
                total = t["total"]
                matched = t["matched"]
                correct = t["correct_vs_gt"]
                wrong = t["wrong_vs_gt"]
                agree = t["correct_vs_foldseek"]
                disagree = t["wrong_vs_foldseek"]
                precision = correct / (correct + wrong) if (correct + wrong) > 0 else 1.0
                recall = matched / total if total > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                agreement = agree / (agree + disagree) if (agree + disagree) > 0 else 1.0
                writer.writerow([
                    f"{thresh:.2f}", total, matched, correct, wrong,
                    f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
                    agree, disagree, f"{agreement:.4f}",
                ])
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
