"""Viro3D database handling for vhold."""

from pathlib import Path

import pandas as pd

from vhold.utils.constants import get_db_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def load_viro3d_metadata(db_dir: Path | None = None) -> pd.DataFrame:
    """Load Viro3D metadata into a DataFrame.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        DataFrame with Viro3D metadata
    """
    if db_dir is None:
        db_dir = get_db_dir()

    # Try multiple possible locations for extracted metadata
    possible_paths = [
        Path(db_dir) / "viro3d" / "viro3d_metadata.tsv",
        Path(db_dir) / "viro3d" / "viro3d_metadata" / "viro3d_metadata.tsv",
    ]

    metadata_path = None
    for path in possible_paths:
        if path.exists():
            metadata_path = path
            break

    if metadata_path is None:
        raise FileNotFoundError(
            f"Viro3D metadata not found. Searched: {possible_paths}. "
            "Run 'vhold install' to download the databases."
        )

    logger.info(f"Loading Viro3D metadata from {metadata_path}")
    df = pd.read_csv(metadata_path, sep="\t")
    logger.info(f"Loaded {len(df)} Viro3D entries")

    return df


def load_viro3d_annotations(db_dir: Path | None = None) -> pd.DataFrame:
    """Load Viro3D expanded annotations.

    Args:
        db_dir: Database directory

    Returns:
        DataFrame with expanded annotation information
    """
    if db_dir is None:
        db_dir = get_db_dir()

    # Try multiple possible locations
    possible_paths = [
        Path(db_dir) / "viro3d" / "viro3d_annotation_expansion.tsv",
        Path(db_dir) / "viro3d" / "viro3d_annotation_expansion" / "viro3d_annotation_expansion.tsv",
    ]

    annotations_path = None
    for path in possible_paths:
        if path.exists():
            annotations_path = path
            break

    if annotations_path is None:
        raise FileNotFoundError(
            f"Viro3D annotations not found. Searched: {possible_paths}. "
            "Run 'vhold install' to download the databases."
        )

    logger.info(f"Loading Viro3D annotations from {annotations_path}")
    df = pd.read_csv(annotations_path, sep="\t")
    logger.info(f"Loaded {len(df)} Viro3D annotation entries")

    return df


def get_viro3d_annotation(
    target_id: str,
    metadata_df: pd.DataFrame,
    annotations_df: pd.DataFrame | None = None,
) -> dict | None:
    """Get annotation for a Viro3D target.

    Args:
        target_id: Target protein ID from Foldseek hit
        metadata_df: Viro3D metadata DataFrame
        annotations_df: Optional expanded annotations DataFrame

    Returns:
        Dict with annotation info or None if not found
    """
    # Viro3D IDs may have format variations
    matches = metadata_df[metadata_df.iloc[:, 0] == target_id]

    if len(matches) == 0:
        # Try partial match
        matches = metadata_df[metadata_df.iloc[:, 0].str.contains(target_id, na=False)]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]

    annotation = {
        "target_id": target_id,
        "source": "viro3d",
    }

    # Map common column names (adjust based on actual Viro3D metadata structure)
    column_mapping = {
        "protein_id": ["protein_id", "id", "accession", "uniprot_id"],
        "description": ["description", "protein_name", "name", "function", "product"],
        "organism": ["organism", "species", "source_organism", "host"],
        "virus_name": ["virus_name", "virus", "species_name"],
        "gene": ["gene", "gene_name"],
        "uniprot_id": ["uniprot_id", "uniprot", "uniprot_accession"],
        "pdb_id": ["pdb_id", "pdb", "structure_id"],
        "go_terms": ["go_terms", "go", "go_annotation"],
        "pfam": ["pfam", "pfam_id", "domains"],
    }

    for field, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                annotation[field] = str(row[col])
                break

    # Merge with expanded annotations if available
    if annotations_df is not None:
        ann_matches = annotations_df[annotations_df.iloc[:, 0] == target_id]
        if len(ann_matches) > 0:
            ann_row = ann_matches.iloc[0]
            for col in ann_row.index:
                if col not in annotation and pd.notna(ann_row[col]):
                    annotation[col] = str(ann_row[col])

    return annotation
