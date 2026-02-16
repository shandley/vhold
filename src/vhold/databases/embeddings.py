"""Embedding database installation for vhold.

Handles downloading and validating the pre-computed reference
embedding database used for embedding-based triage.
"""

from pathlib import Path

from vhold.utils.constants import EMBEDDING_DB_FILE, get_db_dir, get_vhold_zenodo_url
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# URL for the pre-computed embedding database (from Zenodo)
EMBEDDING_DB_URL = get_vhold_zenodo_url("embeddings")


def get_embedding_db_path(db_dir: Path | None = None) -> Path:
    """Get path to the embedding database file.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        Path to the .npz embedding database file
    """
    if db_dir is None:
        db_dir = get_db_dir()
    return Path(db_dir) / "embeddings" / EMBEDDING_DB_FILE


def check_embedding_db(db_dir: Path | None = None) -> bool:
    """Check if the embedding database is installed.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        True if the embedding database file exists
    """
    return get_embedding_db_path(db_dir).exists()


def install_embedding_db(db_dir: Path, force: bool = False) -> None:
    """Download and install the embedding database.

    Args:
        db_dir: Database directory
        force: Force re-download even if file exists
    """
    from vhold.databases.install import download_file

    emb_dir = db_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    dest = emb_dir / EMBEDDING_DB_FILE

    if dest.exists() and not force:
        logger.info("Embedding database already installed")
        return

    if EMBEDDING_DB_URL is None:
        logger.warning(
            "Embedding database URL not yet configured. "
            "Use scripts/build_embedding_db.py to generate a local database, "
            "then place it at: %s",
            dest,
        )
        return

    logger.info(f"Downloading embedding database from {EMBEDDING_DB_URL}")
    download_file(EMBEDDING_DB_URL, dest, "Embedding database")

    # Validate
    import numpy as np

    try:
        data = np.load(dest, allow_pickle=True)
        n = len(data["protein_ids"])
        logger.info(f"Embedding database installed: {n} reference proteins")
    except Exception as e:
        logger.error(f"Invalid embedding database file: {e}")
        if dest.exists():
            dest.unlink()
        raise
