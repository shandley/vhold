"""BFVD database handling for vhold."""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from vhold.utils.constants import get_db_dir
from vhold.utils.logging import get_logger

if TYPE_CHECKING:
    from vhold.databases.uniprot import UniProtAnnotationCache

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
    uniprot_cache: "UniProtAnnotationCache | None" = None,
) -> dict | None:
    """Get annotation for a BFVD target.

    BFVD target IDs are UniProt accessions (e.g., D3TVS4).
    The metadata contains structural quality metrics. Functional descriptions
    are fetched from UniProt if a cache is provided.

    Args:
        target_id: Target protein ID from Foldseek hit (UniProt accession)
        metadata_df: BFVD metadata DataFrame
        uniprot_cache: Optional UniProt annotation cache for functional descriptions

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
    uniprot_id = row["uniprot_id"]

    # Build base annotation dict from BFVD metadata
    annotation = {
        "target_id": target_id,
        "source": "bfvd",
        "uniprot_id": uniprot_id,
        "structure_id": row["structure_id"],
        "plddt": float(row["plddt"]),
        "ptm": float(row["ptm"]),
        # Default description if UniProt fetch fails
        "description": f"UniProt:{uniprot_id}",
    }

    # Try to get functional annotation from UniProt
    if uniprot_cache is not None:
        uniprot_data = uniprot_cache.get(uniprot_id)
        if uniprot_data:
            # Use UniProt description if available
            if uniprot_data.get("description"):
                annotation["description"] = uniprot_data["description"]
            if uniprot_data.get("gene"):
                annotation["gene"] = uniprot_data["gene"]
            if uniprot_data.get("organism"):
                annotation["organism"] = uniprot_data["organism"]

    return annotation


def get_bfvd_uniprot_ids(metadata_df: pd.DataFrame) -> list[str]:
    """Extract all UniProt IDs from BFVD metadata.

    Args:
        metadata_df: BFVD metadata DataFrame

    Returns:
        List of unique UniProt accessions
    """
    return metadata_df["uniprot_id"].dropna().unique().tolist()
