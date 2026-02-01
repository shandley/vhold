"""BFVD database handling for vhold."""

from pathlib import Path

import pandas as pd

from vhold.utils.constants import get_db_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# BFVD metadata columns (file has no header)
BFVD_METADATA_COLUMNS = [
    "uniprot_id",
    "structure_id",
    "plddt",
    "ptm",
    "flag",
    "source",
]


def load_bfvd_metadata(db_dir: Path | None = None) -> pd.DataFrame:
    """Load BFVD metadata into a DataFrame.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        DataFrame with BFVD metadata indexed by uniprot_id
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
    df = pd.read_csv(
        metadata_path,
        sep="\t",
        header=None,
        names=BFVD_METADATA_COLUMNS,
    )
    logger.info(f"Loaded {len(df)} BFVD entries")

    return df


def load_bfvd_taxonomy(db_dir: Path | None = None) -> pd.DataFrame:
    """Load BFVD taxonomy information.

    Args:
        db_dir: Database directory

    Returns:
        DataFrame with taxonomy information indexed by structure file name
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
    df = pd.read_csv(
        taxid_path,
        sep="\t",
        header=None,
        names=["structure_file", "taxid"],
    )
    logger.info(f"Loaded {len(df)} BFVD taxonomy entries")

    return df


def get_bfvd_annotation(
    target_id: str,
    metadata_df: pd.DataFrame,
) -> dict | None:
    """Get annotation for a BFVD target.

    BFVD target IDs are UniProt accessions (e.g., D3TVS4).
    The metadata contains structural quality metrics but not functional
    descriptions. For full annotations, UniProt API would be needed.

    Args:
        target_id: Target protein ID from Foldseek hit (UniProt accession)
        metadata_df: BFVD metadata DataFrame

    Returns:
        Dict with annotation info or None if not found
    """
    # BFVD target IDs are UniProt accessions
    matches = metadata_df[metadata_df["uniprot_id"] == target_id]

    if len(matches) == 0:
        # Try partial match
        matches = metadata_df[
            metadata_df["uniprot_id"].str.contains(target_id, na=False)
        ]

    if len(matches) == 0:
        return None

    row = matches.iloc[0]

    # Build annotation dict
    # Note: BFVD metadata doesn't contain protein descriptions
    # Use UniProt ID as the primary identifier
    annotation = {
        "target_id": target_id,
        "source": "bfvd",
        "uniprot_id": row["uniprot_id"],
        "structure_id": row["structure_id"],
        "plddt": float(row["plddt"]),
        "ptm": float(row["ptm"]),
        # Description uses UniProt ID since BFVD lacks functional annotation
        "description": f"UniProt:{row['uniprot_id']} (BFVD structural homolog)",
    }

    return annotation
