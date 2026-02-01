"""BFVD database handling for vhold."""

from pathlib import Path

import pandas as pd

from vhold.utils.constants import get_db_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def load_bfvd_metadata(db_dir: Path | None = None) -> pd.DataFrame:
    """Load BFVD metadata into a DataFrame.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        DataFrame with BFVD metadata
    """
    if db_dir is None:
        db_dir = get_db_dir()

    metadata_path = Path(db_dir) / "bfvd" / "bfvd_metadata.tsv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"BFVD metadata not found at {metadata_path}. "
            "Run 'vhold install' to download the databases."
        )

    logger.info(f"Loading BFVD metadata from {metadata_path}")
    df = pd.read_csv(metadata_path, sep="\t")
    logger.info(f"Loaded {len(df)} BFVD entries")

    return df


def load_bfvd_taxonomy(db_dir: Path | None = None) -> pd.DataFrame:
    """Load BFVD taxonomy information.

    Args:
        db_dir: Database directory

    Returns:
        DataFrame with taxonomy information
    """
    if db_dir is None:
        db_dir = get_db_dir()

    taxid_path = Path(db_dir) / "bfvd" / "bfvd_taxid.tsv"

    if not taxid_path.exists():
        raise FileNotFoundError(
            f"BFVD taxonomy not found at {taxid_path}. "
            "Run 'vhold install' to download the databases."
        )

    logger.info(f"Loading BFVD taxonomy from {taxid_path}")
    df = pd.read_csv(taxid_path, sep="\t")
    logger.info(f"Loaded {len(df)} BFVD taxonomy entries")

    return df


def get_bfvd_annotation(
    target_id: str,
    metadata_df: pd.DataFrame,
) -> dict | None:
    """Get annotation for a BFVD target.

    Args:
        target_id: Target protein ID from Foldseek hit
        metadata_df: BFVD metadata DataFrame

    Returns:
        Dict with annotation info or None if not found
    """
    # BFVD IDs may have format variations - try exact match first
    matches = metadata_df[metadata_df.iloc[:, 0] == target_id]

    if len(matches) == 0:
        # Try partial match (some IDs have prefixes/suffixes)
        matches = metadata_df[metadata_df.iloc[:, 0].str.contains(target_id, na=False)]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]

    # Build annotation dict from available columns
    annotation = {
        "target_id": target_id,
        "source": "bfvd",
    }

    # Map common column names (adjust based on actual BFVD metadata structure)
    column_mapping = {
        "protein_id": ["protein_id", "id", "accession"],
        "description": ["description", "protein_name", "name", "function"],
        "organism": ["organism", "species", "source_organism"],
        "gene": ["gene", "gene_name"],
        "uniprot_id": ["uniprot_id", "uniprot", "uniprot_accession"],
        "pdb_id": ["pdb_id", "pdb"],
    }

    for field, possible_cols in column_mapping.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                annotation[field] = str(row[col])
                break

    return annotation
