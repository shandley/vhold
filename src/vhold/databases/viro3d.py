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


class Viro3DAnnotationExpansion:
    """Loader for all Viro3D annotation expansion data.

    Loads and indexes:
    - Pfam domain annotations
    - GO Biological Process annotations
    - GO Molecular Function annotations
    - GO Cellular Component annotations
    - SUPERFAMILY annotations
    - Gene3D annotations
    """

    def __init__(self, db_dir: Path | None = None):
        """Initialize the annotation expansion loader.

        Args:
            db_dir: Database directory (default: ~/.vhold/databases)
        """
        if db_dir is None:
            db_dir = get_db_dir()

        self.db_dir = Path(db_dir)
        self.expansion_dir = self.db_dir / "viro3d" / "viro3d_annotation_expansion"

        # Annotation DataFrames (loaded lazily)
        self._pfam_df: pd.DataFrame | None = None
        self._go_bp_df: pd.DataFrame | None = None
        self._go_mf_df: pd.DataFrame | None = None
        self._go_cc_df: pd.DataFrame | None = None
        self._superfamily_df: pd.DataFrame | None = None
        self._gene3d_df: pd.DataFrame | None = None

        # Indexed lookups (built on first access)
        self._pfam_index: dict[str, dict] = {}
        self._go_bp_index: dict[str, dict] = {}
        self._go_mf_index: dict[str, dict] = {}
        self._go_cc_index: dict[str, dict] = {}
        self._superfamily_index: dict[str, dict] = {}
        self._gene3d_index: dict[str, dict] = {}

        self._loaded = False

    def _load_annotation_file(self, filename: str) -> pd.DataFrame | None:
        """Load an annotation expansion CSV file.

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame or None if file doesn't exist
        """
        filepath = self.expansion_dir / filename
        if not filepath.exists():
            logger.debug(f"Annotation file not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            logger.debug(f"Loaded {len(df)} entries from {filename}")
            return df
        except Exception as e:
            logger.warning(f"Error loading {filename}: {e}")
            return None

    def _build_index(self, df: pd.DataFrame, annotation_col: str, confidence_col: str = "Annotation Expansion Ratio") -> dict[str, dict]:
        """Build a Viro3D ID -> annotation index.

        Args:
            df: DataFrame with annotations
            annotation_col: Column containing the annotation value
            confidence_col: Column containing confidence/ratio

        Returns:
            Dict mapping Viro3D ID to annotation dict
        """
        index = {}
        if df is None:
            return index

        for _, row in df.iterrows():
            viro3d_id = row.get("Viro3D ID", "")
            if not viro3d_id:
                continue

            annotation = row.get(annotation_col, "")
            confidence = row.get(confidence_col, 1.0)

            if pd.notna(annotation) and annotation:
                # If multiple annotations exist, keep the highest confidence one
                if viro3d_id not in index or confidence > index[viro3d_id].get("confidence", 0):
                    index[viro3d_id] = {
                        "annotation": str(annotation),
                        "confidence": float(confidence) if pd.notna(confidence) else 1.0,
                    }

        return index

    def load_all(self) -> None:
        """Load all annotation expansion files and build indices."""
        if self._loaded:
            return

        if not self.expansion_dir.exists():
            logger.warning(f"Viro3D annotation expansion directory not found: {self.expansion_dir}")
            self._loaded = True
            return

        logger.info("Loading Viro3D annotation expansion data...")

        # Load Pfam annotations
        self._pfam_df = self._load_annotation_file(
            "Viro3D_proteins_with_Pfam_annotation_expansion.csv"
        )
        if self._pfam_df is not None:
            self._pfam_index = self._build_index(self._pfam_df, "Pfam Annotation")
            logger.info(f"Indexed {len(self._pfam_index)} Pfam annotations")

        # Load GO Biological Process
        self._go_bp_df = self._load_annotation_file(
            "Viro3D_proteins_with_GO_biological_process_annotation_expansion.csv"
        )
        if self._go_bp_df is not None:
            self._go_bp_index = self._build_index(self._go_bp_df, "Biological Process Annotation")
            logger.info(f"Indexed {len(self._go_bp_index)} GO BP annotations")

        # Load GO Molecular Function
        self._go_mf_df = self._load_annotation_file(
            "Viro3D_proteins_with_GO_molecular_function_annotation_expansion.csv"
        )
        if self._go_mf_df is not None:
            self._go_mf_index = self._build_index(self._go_mf_df, "Molecular Function Annotation")
            logger.info(f"Indexed {len(self._go_mf_index)} GO MF annotations")

        # Load GO Cellular Component
        self._go_cc_df = self._load_annotation_file(
            "Viro3D_proteins_with_GO_cellular_component_annotation_expansion.csv"
        )
        if self._go_cc_df is not None:
            self._go_cc_index = self._build_index(self._go_cc_df, "Cellular Component Annotation")
            logger.info(f"Indexed {len(self._go_cc_index)} GO CC annotations")

        # Load SUPERFAMILY
        self._superfamily_df = self._load_annotation_file(
            "Viro3D_proteins_with_SUPERFAMILY_annotation_expansion.csv"
        )
        if self._superfamily_df is not None:
            self._superfamily_index = self._build_index(self._superfamily_df, "SUPERFAMILY Annotation")
            logger.info(f"Indexed {len(self._superfamily_index)} SUPERFAMILY annotations")

        # Load Gene3D
        self._gene3d_df = self._load_annotation_file(
            "Viro3D_proteins_with_Gene3D_annotation_expansion.csv"
        )
        if self._gene3d_df is not None:
            self._gene3d_index = self._build_index(self._gene3d_df, "Gene3D Annotation")
            logger.info(f"Indexed {len(self._gene3d_index)} Gene3D annotations")

        self._loaded = True
        logger.info("Viro3D annotation expansion loading complete")

    def get_expanded_annotations(self, viro3d_id: str) -> dict:
        """Get all expanded annotations for a Viro3D protein.

        Args:
            viro3d_id: Normalized Viro3D ID (e.g., "QHD43423.2_10195")

        Returns:
            Dict with available annotations:
            {
                "pfam": {"annotation": str, "confidence": float},
                "go_bp": {"annotation": str, "confidence": float},
                "go_mf": {"annotation": str, "confidence": float},
                "superfamily": {"annotation": str, "confidence": float},
                "gene3d": {"annotation": str, "confidence": float},
            }
        """
        if not self._loaded:
            self.load_all()

        result = {}

        if viro3d_id in self._pfam_index:
            result["pfam"] = self._pfam_index[viro3d_id]

        if viro3d_id in self._go_bp_index:
            result["go_bp"] = self._go_bp_index[viro3d_id]

        if viro3d_id in self._go_mf_index:
            result["go_mf"] = self._go_mf_index[viro3d_id]

        if viro3d_id in self._go_cc_index:
            result["go_cc"] = self._go_cc_index[viro3d_id]

        if viro3d_id in self._superfamily_index:
            result["superfamily"] = self._superfamily_index[viro3d_id]

        if viro3d_id in self._gene3d_index:
            result["gene3d"] = self._gene3d_index[viro3d_id]

        return result

    def has_annotations(self) -> bool:
        """Check if any annotation data was loaded."""
        if not self._loaded:
            self.load_all()
        return bool(
            self._pfam_index
            or self._go_bp_index
            or self._go_mf_index
            or self._go_cc_index
            or self._superfamily_index
            or self._gene3d_index
        )

    @property
    def stats(self) -> dict:
        """Get statistics about loaded annotations."""
        if not self._loaded:
            self.load_all()
        return {
            "pfam_count": len(self._pfam_index),
            "go_bp_count": len(self._go_bp_index),
            "go_mf_count": len(self._go_mf_index),
            "go_cc_count": len(self._go_cc_index),
            "superfamily_count": len(self._superfamily_index),
            "gene3d_count": len(self._gene3d_index),
        }


# Module-level singleton for caching
_annotation_expansion: Viro3DAnnotationExpansion | None = None


def get_annotation_expansion(db_dir: Path | None = None) -> Viro3DAnnotationExpansion:
    """Get or create the annotation expansion loader singleton.

    Args:
        db_dir: Database directory (only used on first call)

    Returns:
        Viro3DAnnotationExpansion instance
    """
    global _annotation_expansion
    if _annotation_expansion is None:
        _annotation_expansion = Viro3DAnnotationExpansion(db_dir)
    return _annotation_expansion


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
    annotation_expansion: "Viro3DAnnotationExpansion | None" = None,
) -> dict | None:
    """Get annotation for a Viro3D target.

    Args:
        target_id: Target protein ID from Foldseek hit
        metadata_df: Viro3D metadata DataFrame
        annotations_df: Optional Pfam annotations DataFrame (legacy)
        annotation_expansion: Optional Viro3DAnnotationExpansion for full annotations

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
        # Structure quality metrics
        "esmfold_plddt": row.get("ESMFold pLDDT", None),
        "esmfold_ptm": row.get("ESMFold pTM", None),
        "colabfold_plddt": row.get("ColabFold pLDDT", None),
        "colabfold_ptm": row.get("ColabFold pTM", None),
        "colabfold_msa_depth": row.get("ColabFold median MSA depth", None),
    }

    # Get expanded annotations if available
    if annotation_expansion is not None:
        expanded = annotation_expansion.get_expanded_annotations(normalized_id)

        if "pfam" in expanded:
            annotation["pfam"] = expanded["pfam"]["annotation"]
            annotation["pfam_confidence"] = expanded["pfam"]["confidence"]

        if "go_bp" in expanded:
            annotation["go_bp"] = expanded["go_bp"]["annotation"]
            annotation["go_bp_confidence"] = expanded["go_bp"]["confidence"]

        if "go_mf" in expanded:
            annotation["go_mf"] = expanded["go_mf"]["annotation"]
            annotation["go_mf_confidence"] = expanded["go_mf"]["confidence"]

        if "go_cc" in expanded:
            annotation["go_cc"] = expanded["go_cc"]["annotation"]
            annotation["go_cc_confidence"] = expanded["go_cc"]["confidence"]

        if "superfamily" in expanded:
            annotation["superfamily"] = expanded["superfamily"]["annotation"]
            annotation["superfamily_confidence"] = expanded["superfamily"]["confidence"]

        if "gene3d" in expanded:
            annotation["gene3d"] = expanded["gene3d"]["annotation"]
            annotation["gene3d_confidence"] = expanded["gene3d"]["confidence"]

        # Enhance description with domain info if original is poor
        if annotation["description"] in ("viral protein", "", "hypothetical protein"):
            if "pfam" in expanded:
                annotation["description"] = f"{expanded['pfam']['annotation']}"
            elif "superfamily" in expanded:
                annotation["description"] = f"{expanded['superfamily']['annotation']}"

    # Legacy: Add Pfam annotation from DataFrame if expansion not available
    elif annotations_df is not None and len(annotations_df) > 0:
        ann_matches = annotations_df[annotations_df["Viro3D ID"] == normalized_id]
        if len(ann_matches) > 0:
            ann_row = ann_matches.iloc[0]
            pfam = ann_row.get("Pfam Annotation", "")
            if pd.notna(pfam) and pfam:
                annotation["pfam"] = pfam
                # If we don't have a good description, use Pfam
                if annotation["description"] in ("viral protein", ""):
                    annotation["description"] = f"Pfam: {pfam}"

    from vhold.results.go_terms import enrich_annotation_with_go_ids
    return enrich_annotation_with_go_ids(annotation)
