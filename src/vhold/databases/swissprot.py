"""Swiss-Prot reference database management for GO term transfer.

Handles path management, installation status checking, and metadata loading
for the Swiss-Prot embedding database used in FANTASIA-style GO term transfer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vhold.utils.constants import (
    SWISSPROT_ALL_EMBEDDINGS,
    SWISSPROT_ALL_METADATA,
    SWISSPROT_VIRAL_EMBEDDINGS,
    SWISSPROT_VIRAL_METADATA,
    get_db_dir,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Subdirectory within database dir
SWISSPROT_DB_SUBDIR = "swissprot"


@dataclass
class GOAnnotation:
    """A single GO annotation with experimental evidence."""

    go_id: str  # GO:0039694
    go_term: str  # "viral process"
    aspect: str  # BP, MF, CC
    evidence_code: str  # EXP, IDA, etc.


@dataclass
class SwissProtEntry:
    """A Swiss-Prot reference protein with experimental GO annotations."""

    accession: str  # P0DTC2
    name: str  # SPIKE_SARS2
    organism: str  # Severe acute respiratory syndrome coronavirus 2
    lineage: str  # Viruses; Riboviria; ...
    go_annotations: list[GOAnnotation] = field(default_factory=list)
    sequence: str = ""
    length: int = 0

    @property
    def is_viral(self) -> bool:
        """Check if this protein is from a virus."""
        return self.lineage.startswith("Viruses")

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        d = asdict(self)
        # Remove sequence to save space in metadata (stored in embeddings .npz)
        d.pop("sequence", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SwissProtEntry:
        """Create from serialized dict."""
        go_annotations = [GOAnnotation(**go) for go in d.get("go_annotations", [])]
        return cls(
            accession=d["accession"],
            name=d.get("name", ""),
            organism=d.get("organism", ""),
            lineage=d.get("lineage", ""),
            go_annotations=go_annotations,
            sequence=d.get("sequence", ""),
            length=d.get("length", 0),
        )


def _get_scope_files(scope: str) -> tuple[str, str]:
    """Get embedding and metadata filenames for a scope.

    Args:
        scope: "viral" or "all"

    Returns:
        Tuple of (embeddings_filename, metadata_filename)

    Raises:
        ValueError: If scope is not valid
    """
    if scope == "viral":
        return SWISSPROT_VIRAL_EMBEDDINGS, SWISSPROT_VIRAL_METADATA
    elif scope == "all":
        return SWISSPROT_ALL_EMBEDDINGS, SWISSPROT_ALL_METADATA
    else:
        msg = f"Invalid scope '{scope}'. Must be 'viral' or 'all'."
        raise ValueError(msg)


def get_swissprot_db_dir(db_dir: Path | None = None) -> Path:
    """Get directory for Swiss-Prot reference database files.

    Args:
        db_dir: Base database directory (default: ~/.vhold/databases)

    Returns:
        Path to Swiss-Prot subdirectory
    """
    if db_dir is None:
        db_dir = get_db_dir()
    return Path(db_dir) / SWISSPROT_DB_SUBDIR


def get_swissprot_embeddings_path(
    scope: str = "viral", db_dir: Path | None = None
) -> Path:
    """Get path to Swiss-Prot embeddings file.

    Args:
        scope: "viral" (17K) or "all" (570K)
        db_dir: Base database directory

    Returns:
        Path to the .npz embeddings file
    """
    emb_file, _ = _get_scope_files(scope)
    return get_swissprot_db_dir(db_dir) / emb_file


def get_swissprot_metadata_path(
    scope: str = "viral", db_dir: Path | None = None
) -> Path:
    """Get path to Swiss-Prot metadata file.

    Args:
        scope: "viral" (17K) or "all" (570K)
        db_dir: Base database directory

    Returns:
        Path to the metadata JSON file
    """
    _, meta_file = _get_scope_files(scope)
    return get_swissprot_db_dir(db_dir) / meta_file


def check_swissprot_db(scope: str = "viral", db_dir: Path | None = None) -> bool:
    """Check if Swiss-Prot reference database is installed.

    Both the embeddings (.npz) and metadata (.json) files must be present.

    Args:
        scope: "viral" or "all"
        db_dir: Base database directory

    Returns:
        True if both files exist
    """
    emb_path = get_swissprot_embeddings_path(scope, db_dir)
    meta_path = get_swissprot_metadata_path(scope, db_dir)
    return emb_path.exists() and meta_path.exists()


def get_available_scope(db_dir: Path | None = None) -> str | None:
    """Get the best available Swiss-Prot scope.

    Prefers "all" over "viral" since it enables cross-domain discovery.

    Args:
        db_dir: Base database directory

    Returns:
        "all", "viral", or None if neither is installed
    """
    if check_swissprot_db("all", db_dir):
        return "all"
    if check_swissprot_db("viral", db_dir):
        return "viral"
    return None


def load_swissprot_metadata(
    scope: str = "viral", db_dir: Path | None = None
) -> dict[str, SwissProtEntry]:
    """Load Swiss-Prot metadata from JSON file.

    Args:
        scope: "viral" or "all"
        db_dir: Base database directory

    Returns:
        Dict mapping accession to SwissProtEntry

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    meta_path = get_swissprot_metadata_path(scope, db_dir)
    if not meta_path.exists():
        msg = (
            f"Swiss-Prot metadata not found at {meta_path}. "
            f"Run 'scripts/build_swissprot_reference.py --scope {scope}' to build it."
        )
        raise FileNotFoundError(msg)

    logger.info(f"Loading Swiss-Prot metadata from {meta_path}")
    with open(meta_path) as f:
        raw = json.load(f)

    metadata = {}
    for accession, entry_data in raw.items():
        metadata[accession] = SwissProtEntry.from_dict(entry_data)

    logger.info(f"Loaded {len(metadata)} Swiss-Prot entries (scope={scope})")
    return metadata


def install_swissprot_db(
    db_dir: Path, scope: str = "viral", force: bool = False
) -> None:
    """Download and install Swiss-Prot reference database from Zenodo.

    Args:
        db_dir: Base database directory
        scope: "viral" or "all"
        force: Force re-download even if files exist
    """
    from vhold.databases.install import download_file
    from vhold.utils.constants import get_vhold_zenodo_url

    sp_dir = get_swissprot_db_dir(db_dir)
    sp_dir.mkdir(parents=True, exist_ok=True)

    emb_file, meta_file = _get_scope_files(scope)

    for filename in [emb_file, meta_file]:
        dest = sp_dir / filename
        if dest.exists() and not force:
            logger.info(f"Swiss-Prot {filename} already installed")
            continue

        url = get_vhold_zenodo_url(f"swissprot_{scope}_{filename.split('_')[-1].split('.')[0]}")
        if url is None:
            logger.warning(
                "Swiss-Prot database URL not yet configured for '%s'. "
                "Run 'scripts/build_swissprot_reference.py --scope %s' to build locally.",
                filename,
                scope,
            )
            return

        logger.info(f"Downloading {filename} from {url}")
        download_file(url, dest, f"Swiss-Prot {scope} {filename}")

    # Validate embeddings
    import numpy as np

    emb_path = sp_dir / emb_file
    if emb_path.exists():
        try:
            data = np.load(emb_path, allow_pickle=True)
            n = len(data["protein_ids"])
            logger.info(f"Swiss-Prot embeddings validated: {n} proteins (scope={scope})")
        except Exception as e:
            logger.error(f"Invalid Swiss-Prot embeddings file: {e}")
            if emb_path.exists():
                emb_path.unlink()
            raise
