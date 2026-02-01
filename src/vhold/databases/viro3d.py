"""Viro3D database handling for vhold."""

import re
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

    # The actual file is Viro3D_proteins_list.csv
    possible_paths = [
        Path(db_dir) / "viro3d" / "viro3d_metadata" / "Viro3D_proteins_list.csv",
        Path(db_dir) / "viro3d" / "Viro3D_proteins_list.csv",
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
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} Viro3D entries")

    return df


def load_viro3d_annotations(db_dir: Path | None = None) -> pd.DataFrame:
    """Load Viro3D expanded annotations (Pfam).

    Args:
        db_dir: Database directory

    Returns:
        DataFrame with Pfam annotation information
    """
    if db_dir is None:
        db_dir = get_db_dir()

    # Try to load Pfam annotations (most useful for functional annotation)
    possible_paths = [
        Path(db_dir) / "viro3d" / "viro3d_annotation_expansion" / "Viro3D_proteins_with_Pfam_annotation_expansion.csv",
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
    df = pd.read_csv(annotations_path)
    logger.info(f"Loaded {len(df)} Viro3D annotation entries")

    return df


def normalize_viro3d_target_id(target_id: str) -> str:
    """Normalize a Foldseek target ID to match Viro3D ID format.

    Foldseek returns IDs like: CF-QHD43423.2_10195_relaxed
    Viro3D metadata has IDs like: QHD43423.2_10195

    Args:
        target_id: Target ID from Foldseek hit

    Returns:
        Normalized ID for matching against Viro3D metadata
    """
    # Remove prefix (CF- for ColabFold, EF- for ESMFold)
    normalized = re.sub(r"^[CE]F-", "", target_id)
    # Remove suffix (_relaxed or _unrelaxed)
    normalized = re.sub(r"_(relaxed|unrelaxed)$", "", normalized)
    return normalized


def get_viro3d_annotation(
    target_id: str,
    metadata_df: pd.DataFrame,
    annotations_df: pd.DataFrame | None = None,
) -> dict | None:
    """Get annotation for a Viro3D target.

    Args:
        target_id: Target protein ID from Foldseek hit
        metadata_df: Viro3D metadata DataFrame
        annotations_df: Optional Pfam annotations DataFrame

    Returns:
        Dict with annotation info or None if not found
    """
    # Normalize the target ID for matching
    normalized_id = normalize_viro3d_target_id(target_id)

    # Match against "Viro3D ID" column
    matches = metadata_df[metadata_df["Viro3D ID"] == normalized_id]

    if len(matches) == 0:
        # Try partial match on GenBank Protein ID (the part before the underscore)
        genbank_id = normalized_id.split("_")[0] if "_" in normalized_id else normalized_id
        matches = metadata_df[
            metadata_df["GenBank Protein ID"].str.contains(genbank_id, na=False)
        ]

    if len(matches) == 0:
        logger.debug(f"No Viro3D match for {target_id} (normalized: {normalized_id})")
        return None

    row = matches.iloc[0]

    # Extract description from "Viro3D Name" field
    # Format is typically: "Gene: N; Product: nucleocapsid phosphoprotein"
    viro3d_name = row.get("Viro3D Name", "")
    description = viro3d_name

    # Parse out the Product if available
    if "Product:" in str(viro3d_name):
        product_match = re.search(r"Product:\s*(.+?)(?:;|$)", str(viro3d_name))
        if product_match:
            description = product_match.group(1).strip()

    # Parse out the Gene if available
    gene = ""
    if "Gene:" in str(viro3d_name):
        gene_match = re.search(r"Gene:\s*(.+?)(?:;|$)", str(viro3d_name))
        if gene_match:
            gene = gene_match.group(1).strip()

    # Build annotation dict
    annotation = {
        "target_id": target_id,
        "source": "viro3d",
        "viro3d_id": row.get("Viro3D ID", ""),
        "description": description if description else "viral protein",
        "organism": row.get("ICTV Species", ""),
        "gene": gene,
        "genbank_id": row.get("GenBank Protein ID", ""),
        "uniprot_id": row.get("UniProt ID", ""),
        "protein_length": row.get("Protein Length", ""),
    }

    # Add Pfam annotation if available
    if annotations_df is not None and len(annotations_df) > 0:
        ann_matches = annotations_df[annotations_df["Viro3D ID"] == normalized_id]
        if len(ann_matches) > 0:
            ann_row = ann_matches.iloc[0]
            pfam = ann_row.get("Pfam Annotation", "")
            if pd.notna(pfam) and pfam:
                annotation["pfam"] = pfam
                # If we don't have a good description, use Pfam
                if annotation["description"] in ("viral protein", ""):
                    annotation["description"] = f"Pfam: {pfam}"

    return annotation
