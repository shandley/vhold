"""Output generation for vhold results."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from vhold import __version__
from vhold.results.annotations import AnnotatedProtein
from vhold.results.consensus import ConsensusResult
from vhold.results.dark_matter import (
    analyze_dark_matter,
    get_dark_matter_summary,
    DarkMatterReport,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# Output TSV columns
TSV_COLUMNS = [
    "query_id",
    "query_length",
    "description",
    "confidence_level",
    "source_db",
    "target_id",
    "evalue",
    "identity",
    "coverage",
    "bits",
    "organism",
    "gene",
    "uniprot_id",
    "pdb_id",
    "num_hits",
]

# Consensus output columns (includes multi-database info)
CONSENSUS_TSV_COLUMNS = [
    "query_id",
    "query_length",
    "description",
    "confidence_level",
    "consensus_score",
    "agreement",
    "functional_category",
    "classification_source",
    "novelty",
    "structure_quality_score",
    "structure_quality_source",
    "primary_source",
    "primary_target",
    "primary_evalue",
    "primary_identity",
    "primary_coverage",
    "secondary_source",
    "secondary_target",
    "secondary_evalue",
    "secondary_identity",
    "organism",
    "gene",
    "uniprot_id",
    "pfam",
    "go_bp",
    "go_bp_ids",
    "go_mf",
    "go_mf_ids",
    "superfamily",
    "bfvd_hits",
    "viro3d_hits",
]


def write_tsv_output(
    annotations: dict[str, AnnotatedProtein],
    output_path: Path | str,
    include_unannotated: bool = True,
) -> None:
    """Write annotations to TSV file.

    Args:
        annotations: Dict of AnnotatedProtein objects
        output_path: Output file path
        include_unannotated: Include proteins without hits
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for query_id, ann in annotations.items():
        if not include_unannotated and not ann.is_annotated:
            continue

        row = ann.to_dict()
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns (keeping any extra annotation columns at end)
    ordered_cols = [c for c in TSV_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in TSV_COLUMNS]
    df = df[ordered_cols + extra_cols]

    # Write TSV
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df)} annotations to {output_path}")


def write_summary_json(
    annotations: dict[str, AnnotatedProtein],
    output_path: Path | str,
    input_file: str = "",
    databases_searched: list[str] | None = None,
    parameters: dict | None = None,
) -> None:
    """Write summary statistics to JSON file.

    Args:
        annotations: Dict of AnnotatedProtein objects
        output_path: Output file path
        input_file: Input file name
        databases_searched: List of databases searched
        parameters: Run parameters
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total = len(annotations)
    annotated = sum(1 for a in annotations.values() if a.is_annotated)
    unannotated = total - annotated

    # Confidence distribution
    confidence_dist = {"high": 0, "medium": 0, "low": 0, "very_low": 0, "none": 0}
    for ann in annotations.values():
        confidence_dist[ann.confidence_level] = confidence_dist.get(ann.confidence_level, 0) + 1

    # Source database distribution
    source_dist = {}
    for ann in annotations.values():
        if ann.source_db:
            source_dist[ann.source_db] = source_dist.get(ann.source_db, 0) + 1

    # E-value distribution
    evalues = [ann.evalue for ann in annotations.values() if ann.evalue is not None]
    evalue_stats = {}
    if evalues:
        evalue_stats = {
            "min": min(evalues),
            "max": max(evalues),
            "median": sorted(evalues)[len(evalues) // 2],
        }

    # Build summary
    summary = {
        "vhold_version": __version__,
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "databases_searched": databases_searched or [],
        "parameters": parameters or {},
        "statistics": {
            "total_proteins": total,
            "annotated": annotated,
            "unannotated": unannotated,
            "annotation_rate": annotated / total if total > 0 else 0,
            "confidence_distribution": confidence_dist,
            "source_distribution": source_dist,
            "evalue_stats": evalue_stats,
        },
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote summary to {output_path}")


def generate_report(
    annotations: dict[str, AnnotatedProtein],
    output_dir: Path,
    prefix: str = "vhold",
    input_file: str = "",
    databases_searched: list[str] | None = None,
    parameters: dict | None = None,
    include_unannotated: bool = True,
) -> dict[str, Path]:
    """Generate all output files.

    Args:
        annotations: Dict of AnnotatedProtein objects
        output_dir: Output directory
        prefix: Output file prefix
        input_file: Input file name
        databases_searched: List of databases searched
        parameters: Run parameters
        include_unannotated: Include unannotated proteins in TSV

    Returns:
        Dict of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Write main TSV
    tsv_path = output_dir / f"{prefix}_results.tsv"
    write_tsv_output(annotations, tsv_path, include_unannotated=include_unannotated)
    output_files["tsv"] = tsv_path

    # Write summary JSON
    json_path = output_dir / f"{prefix}_summary.json"
    write_summary_json(
        annotations,
        json_path,
        input_file=input_file,
        databases_searched=databases_searched,
        parameters=parameters,
    )
    output_files["json"] = json_path

    return output_files


# ============================================================
# Consensus output functions
# ============================================================


def write_consensus_tsv(
    annotations: dict[str, ConsensusResult],
    output_path: Path | str,
    include_unannotated: bool = True,
) -> None:
    """Write consensus annotations to TSV file.

    Args:
        annotations: Dict of ConsensusResult objects
        output_path: Output file path
        include_unannotated: Include proteins without hits
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for query_id, ann in annotations.items():
        if not include_unannotated and not ann.is_annotated:
            continue

        row = ann.to_dict()
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    ordered_cols = [c for c in CONSENSUS_TSV_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CONSENSUS_TSV_COLUMNS]
    df = df[ordered_cols + extra_cols]

    # Write TSV
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df)} consensus annotations to {output_path}")


def write_consensus_summary_json(
    annotations: dict[str, ConsensusResult],
    output_path: Path | str,
    input_file: str = "",
    databases_searched: list[str] | None = None,
    parameters: dict | None = None,
) -> None:
    """Write consensus summary statistics to JSON file.

    Args:
        annotations: Dict of ConsensusResult objects
        output_path: Output file path
        input_file: Input file name
        databases_searched: List of databases searched
        parameters: Run parameters
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total = len(annotations)
    annotated = sum(1 for a in annotations.values() if a.is_annotated)
    unannotated = total - annotated
    with_consensus = sum(1 for a in annotations.values() if a.has_consensus)

    # Confidence distribution
    confidence_dist = {"high": 0, "medium": 0, "low": 0, "very_low": 0, "none": 0}
    for ann in annotations.values():
        confidence_dist[ann.confidence_level] = confidence_dist.get(ann.confidence_level, 0) + 1

    # Agreement distribution
    agreement_dist = {"agree": 0, "partial": 0, "disagree": 0, "single": 0, "none": 0}
    for ann in annotations.values():
        agreement_dist[ann.agreement] = agreement_dist.get(ann.agreement, 0) + 1

    # Functional category distribution
    category_dist: dict[str, int] = {}
    for ann in annotations.values():
        cat = ann.functional_category
        category_dist[cat] = category_dist.get(cat, 0) + 1

    # Primary source distribution
    source_dist = {}
    for ann in annotations.values():
        if ann.primary_source:
            source_dist[ann.primary_source] = source_dist.get(ann.primary_source, 0) + 1

    # Consensus score stats
    scores = [ann.consensus_score for ann in annotations.values() if ann.is_annotated]
    score_stats = {}
    if scores:
        score_stats = {
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "mean": round(sum(scores) / len(scores), 3),
            "median": round(sorted(scores)[len(scores) // 2], 3),
        }

    # E-value stats from primary hits
    evalues = [
        ann.primary_hit.evalue
        for ann in annotations.values()
        if ann.primary_hit is not None
    ]
    evalue_stats = {}
    if evalues:
        evalue_stats = {
            "min": min(evalues),
            "max": max(evalues),
            "median": sorted(evalues)[len(evalues) // 2],
        }

    # Structure quality statistics
    structure_quality_scores = [
        ann.structure_quality_score
        for ann in annotations.values()
        if ann.is_annotated and ann.structure_quality_source != "none"
    ]
    structure_quality_stats = {}
    if structure_quality_scores:
        structure_quality_stats = {
            "min": round(min(structure_quality_scores), 3),
            "max": round(max(structure_quality_scores), 3),
            "mean": round(sum(structure_quality_scores) / len(structure_quality_scores), 3),
            "median": round(sorted(structure_quality_scores)[len(structure_quality_scores) // 2], 3),
        }

    # Structure quality source distribution
    structure_source_dist: dict[str, int] = {}
    for ann in annotations.values():
        if ann.is_annotated:
            src = ann.structure_quality_source
            structure_source_dist[src] = structure_source_dist.get(src, 0) + 1

    # Novelty distribution (how much value does structural search provide?)
    novelty_dist: dict[str, int] = {}
    for ann in annotations.values():
        if ann.is_annotated:
            nov = ann.novelty
            novelty_dist[nov] = novelty_dist.get(nov, 0) + 1

    # Analyze dark matter proteins
    dark_matter_report = analyze_dark_matter(annotations)
    dark_matter_summary = get_dark_matter_summary(dark_matter_report)

    # Build summary
    summary = {
        "vhold_version": __version__,
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "databases_searched": databases_searched or [],
        "parameters": parameters or {},
        "statistics": {
            "total_proteins": total,
            "annotated": annotated,
            "unannotated": unannotated,
            "annotation_rate": annotated / total if total > 0 else 0,
            "with_multi_db_consensus": with_consensus,
            "consensus_rate": with_consensus / annotated if annotated > 0 else 0,
            "confidence_distribution": confidence_dist,
            "agreement_distribution": agreement_dist,
            "category_distribution": category_dist,
            "novelty_distribution": novelty_dist,
            "primary_source_distribution": source_dist,
            "consensus_score_stats": score_stats,
            "evalue_stats": evalue_stats,
            "structure_quality_stats": structure_quality_stats,
            "structure_quality_source_distribution": structure_source_dist,
        },
        "dark_matter": dark_matter_summary,
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote consensus summary to {output_path}")


def write_dark_matter_tsv(
    dark_matter_report: DarkMatterReport,
    output_path: Path | str,
) -> None:
    """Write dark matter proteins to a separate TSV file.

    Args:
        dark_matter_report: DarkMatterReport from analysis
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dark_matter_report.proteins:
        logger.info("No dark matter proteins to write")
        return

    rows = []
    for protein in dark_matter_report.proteins:
        row = {
            "query_id": protein.query_id,
            "query_length": protein.query_length,
            "category": protein.category,
            "reason": protein.reason,
            "best_evalue": protein.best_evalue if protein.best_evalue else "",
            "best_identity": protein.best_identity if protein.best_identity else "",
            "confidence_level": protein.confidence_level,
            "consensus_score": protein.consensus_score if protein.consensus_score > 0 else "",
            "description": protein.description,
            "databases_with_hits": ",".join(protein.databases_with_hits),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df)} dark matter proteins to {output_path}")


def generate_report_consensus(
    annotations: dict[str, ConsensusResult],
    output_dir: Path,
    prefix: str = "vhold",
    input_file: str = "",
    databases_searched: list[str] | None = None,
    parameters: dict | None = None,
    include_unannotated: bool = True,
) -> dict[str, Path]:
    """Generate all output files for consensus results.

    Args:
        annotations: Dict of ConsensusResult objects
        output_dir: Output directory
        prefix: Output file prefix
        input_file: Input file name
        databases_searched: List of databases searched
        parameters: Run parameters
        include_unannotated: Include unannotated proteins in TSV

    Returns:
        Dict of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # Write main TSV
    tsv_path = output_dir / f"{prefix}_results.tsv"
    write_consensus_tsv(annotations, tsv_path, include_unannotated=include_unannotated)
    output_files["tsv"] = tsv_path

    # Write summary JSON
    json_path = output_dir / f"{prefix}_summary.json"
    write_consensus_summary_json(
        annotations,
        json_path,
        input_file=input_file,
        databases_searched=databases_searched,
        parameters=parameters,
    )
    output_files["json"] = json_path

    # Write dark matter TSV
    dark_matter_report = analyze_dark_matter(annotations)
    if dark_matter_report.total_dark_matter > 0:
        dm_path = output_dir / f"{prefix}_dark_matter.tsv"
        write_dark_matter_tsv(dark_matter_report, dm_path)
        output_files["dark_matter"] = dm_path

    return output_files
