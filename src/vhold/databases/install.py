"""Database download and installation for vhold."""

import tarfile
from pathlib import Path

import requests
from alive_progress import alive_bar

from vhold.utils.constants import (
    BFVD_BASE_URL,
    BFVD_FILES,
    VIRO3D_BASE_URL,
    VIRO3D_FILES,
    get_db_dir,
)
from vhold.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def download_file(
    url: str,
    dest: Path,
    desc: str | None = None,
    chunk_size: int = 8192,
) -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        dest: Destination path
        desc: Description for progress bar
        chunk_size: Download chunk size
    """
    desc = desc or dest.name

    # Get file size
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress
    response = requests.get(url, stream=True)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        if total_size > 0:
            with alive_bar(
                total_size,
                title=desc,
                unit="B",
                scale=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar(len(chunk))
        else:
            # No content-length header, download without progress
            logger.info(f"Downloading {desc} (size unknown)")
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def extract_tarball(
    tar_path: Path,
    dest_dir: Path,
    desc: str | None = None,
) -> None:
    """Extract a tarball to a directory.

    Args:
        tar_path: Path to tarball
        dest_dir: Destination directory
        desc: Description for logging
    """
    desc = desc or tar_path.name
    logger.info(f"Extracting {desc}...")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        # Use fully_trusted filter for database files from known sources
        # Some databases (like Viro3D) contain symlinks that need this
        try:
            tar.extractall(dest_dir, filter='fully_trusted')
        except TypeError:
            # Python < 3.12 doesn't have filter parameter
            tar.extractall(dest_dir)


def install_bfvd(db_dir: Path, force: bool = False) -> None:
    """Install BFVD database and metadata.

    Args:
        db_dir: Database directory
        force: Force re-download even if files exist
    """
    bfvd_dir = db_dir / "bfvd"
    bfvd_dir.mkdir(parents=True, exist_ok=True)

    # Download Foldseek database
    foldseek_tar = bfvd_dir / BFVD_FILES["foldseekdb"]
    foldseek_marker = bfvd_dir / "bfvd_foldseekdb" / "_db.index"

    if not foldseek_marker.exists() or force:
        if not foldseek_tar.exists() or force:
            url = f"{BFVD_BASE_URL}/{BFVD_FILES['foldseekdb']}"
            logger.info(f"Downloading BFVD Foldseek database from {url}")
            download_file(url, foldseek_tar, "BFVD Foldseek DB")

        extract_tarball(foldseek_tar, bfvd_dir, "BFVD Foldseek DB")
        # Clean up tarball after extraction
        if foldseek_tar.exists():
            foldseek_tar.unlink()
    else:
        logger.info("BFVD Foldseek database already installed")

    # Download metadata
    metadata_file = bfvd_dir / BFVD_FILES["metadata"]
    if not metadata_file.exists() or force:
        url = f"{BFVD_BASE_URL}/{BFVD_FILES['metadata']}"
        logger.info(f"Downloading BFVD metadata from {url}")
        download_file(url, metadata_file, "BFVD metadata")
    else:
        logger.info("BFVD metadata already installed")

    # Download taxonomy
    taxid_file = bfvd_dir / BFVD_FILES["taxid"]
    if not taxid_file.exists() or force:
        url = f"{BFVD_BASE_URL}/{BFVD_FILES['taxid']}"
        logger.info(f"Downloading BFVD taxonomy from {url}")
        download_file(url, taxid_file, "BFVD taxonomy")
    else:
        logger.info("BFVD taxonomy already installed")

    logger.info("BFVD installation complete")


def install_viro3d(db_dir: Path, force: bool = False) -> None:
    """Install Viro3D database and metadata.

    Args:
        db_dir: Database directory
        force: Force re-download even if files exist
    """
    viro3d_dir = db_dir / "viro3d"
    viro3d_dir.mkdir(parents=True, exist_ok=True)

    # Download Foldseek database
    foldseek_tar = viro3d_dir / VIRO3D_FILES["foldseekdb"]
    # Check for extracted database - look for typical foldseek db files
    foldseek_db_dir = viro3d_dir / "foldseekViro3D"

    if not foldseek_db_dir.exists() or force:
        if not foldseek_tar.exists() or force:
            url = f"{VIRO3D_BASE_URL}/{VIRO3D_FILES['foldseekdb']}"
            logger.info(f"Downloading Viro3D Foldseek database from {url}")
            download_file(url, foldseek_tar, "Viro3D Foldseek DB")

        extract_tarball(foldseek_tar, viro3d_dir, "Viro3D Foldseek DB")
        # Clean up tarball after extraction
        if foldseek_tar.exists():
            foldseek_tar.unlink()
    else:
        logger.info("Viro3D Foldseek database already installed")

    # Download metadata
    metadata_tar = viro3d_dir / VIRO3D_FILES["metadata"]
    metadata_marker = viro3d_dir / "viro3d_metadata.tsv"

    if not metadata_marker.exists() or force:
        if not metadata_tar.exists() or force:
            url = f"{VIRO3D_BASE_URL}/{VIRO3D_FILES['metadata']}"
            logger.info(f"Downloading Viro3D metadata from {url}")
            download_file(url, metadata_tar, "Viro3D metadata")

        extract_tarball(metadata_tar, viro3d_dir, "Viro3D metadata")
        # Clean up tarball
        if metadata_tar.exists():
            metadata_tar.unlink()
    else:
        logger.info("Viro3D metadata already installed")

    # Download annotations
    annotations_tar = viro3d_dir / VIRO3D_FILES["annotations"]
    annotations_marker = viro3d_dir / "viro3d_annotation_expansion.tsv"

    if not annotations_marker.exists() or force:
        if not annotations_tar.exists() or force:
            url = f"{VIRO3D_BASE_URL}/{VIRO3D_FILES['annotations']}"
            logger.info(f"Downloading Viro3D annotations from {url}")
            download_file(url, annotations_tar, "Viro3D annotations")

        extract_tarball(annotations_tar, viro3d_dir, "Viro3D annotations")
        # Clean up tarball
        if annotations_tar.exists():
            annotations_tar.unlink()
    else:
        logger.info("Viro3D annotations already installed")

    logger.info("Viro3D installation complete")


def check_databases(db_dir: Path | None = None) -> dict[str, bool]:
    """Check which databases are installed.

    Args:
        db_dir: Database directory (default: ~/.vhold/databases)

    Returns:
        Dict with database availability status
    """
    if db_dir is None:
        db_dir = get_db_dir()

    db_dir = Path(db_dir)

    from vhold.utils.constants import EMBEDDING_DB_FILE

    return {
        "bfvd": (db_dir / "bfvd" / "bfvd_metadata.tsv").exists(),
        "viro3d": (db_dir / "viro3d" / "viro3d_metadata.tsv").exists(),
        "embeddings": (db_dir / "embeddings" / EMBEDDING_DB_FILE).exists(),
    }


def get_bfvd_db_path(db_dir: Path | None = None) -> Path:
    """Get the path to the BFVD Foldseek database.

    Args:
        db_dir: Database directory

    Returns:
        Path to BFVD Foldseek database
    """
    if db_dir is None:
        db_dir = get_db_dir()
    # Database files are extracted directly: bfvd/bfvd, bfvd/bfvd_ss, etc.
    return Path(db_dir) / "bfvd" / "bfvd"


def get_viro3d_db_path(db_dir: Path | None = None) -> Path:
    """Get the path to the Viro3D Foldseek database.

    Args:
        db_dir: Database directory

    Returns:
        Path to Viro3D Foldseek database
    """
    if db_dir is None:
        db_dir = get_db_dir()
    # Database files are in foldseekViro3D/viro3d, viro3d_ss, etc.
    return Path(db_dir) / "viro3d" / "foldseekViro3D" / "viro3d"


def install_databases(
    database_dir: str | Path | None = None,
    install_bfvd: bool = True,
    install_viro3d: bool = True,
    install_embeddings: bool = False,
    force: bool = False,
) -> None:
    """Install databases for vhold.

    Args:
        database_dir: Directory to install databases (default: ~/.vhold/databases)
        install_bfvd: Whether to install BFVD
        install_viro3d: Whether to install Viro3D
        install_embeddings: Whether to install embedding database for triage
        force: Force re-download even if files exist
    """
    setup_logging()

    if database_dir is None:
        db_dir = get_db_dir()
    else:
        db_dir = Path(database_dir)

    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Installing databases to {db_dir}")

    if not install_bfvd and not install_viro3d and not install_embeddings:
        logger.warning("No databases selected for installation")
        return

    if install_bfvd:
        logger.info("Installing BFVD database...")
        from vhold.databases.install import install_bfvd as _install_bfvd
        _install_bfvd(db_dir, force=force)

    if install_viro3d:
        logger.info("Installing Viro3D database...")
        from vhold.databases.install import install_viro3d as _install_viro3d
        _install_viro3d(db_dir, force=force)

    if install_embeddings:
        logger.info("Installing embedding database...")
        from vhold.databases.embeddings import install_embedding_db
        install_embedding_db(db_dir, force=force)

    logger.info("Database installation complete!")

    # Show summary
    status = check_databases(db_dir)
    logger.info("Installed databases:")
    for db_name, installed in status.items():
        status_str = "✓ installed" if installed else "✗ not installed"
        logger.info(f"  {db_name}: {status_str}")
