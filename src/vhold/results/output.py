"""Output generation for vhold results."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from vhold import __version__
from vhold.results.annotations import AnnotatedProtein
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
    "prob",
    "organism",
    "gene",
    "uniprot_id",
    "pdb_id",
    "num_hits",
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
