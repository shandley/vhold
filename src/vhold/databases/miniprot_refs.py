"""Miniprot reference protein database management for vhold.

The reference protein FASTA is bundled with the package in src/vhold/data/
and contains ~300-500 representative proteins from splicing-prone virus
families (Adenoviridae, Papillomaviridae, Polyomaviridae, Retroviridae,
Herpesviridae, Orthomyxoviridae, Bornaviridae, Circoviridae).
"""

from pathlib import Path

from vhold.utils.constants import MINIPROT_REF_DB, get_db_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def get_miniprot_ref_path() -> Path:
    """Get path to miniprot reference protein FASTA.

    Checks two locations:
    1. Bundled with the package (src/vhold/data/)
    2. User-installed location (~/.vhold/databases/)

    Returns:
        Path to reference FASTA file

    Raises:
        FileNotFoundError: If reference DB not found in either location
    """
    # Check bundled location first
    import importlib.resources

    try:
        ref = importlib.resources.files("vhold.data") / MINIPROT_REF_DB
        if ref.is_file():
            return Path(str(ref))
    except (TypeError, FileNotFoundError, AttributeError):
        pass

    # Fall back to user-installed location
    user_path = get_db_dir() / "miniprot" / MINIPROT_REF_DB
    if user_path.exists():
        return user_path

    raise FileNotFoundError(
        f"Miniprot reference DB '{MINIPROT_REF_DB}' not found. "
        "This file should be bundled with vhold. Try reinstalling."
    )


def check_miniprot_refs() -> bool:
    """Check if miniprot reference DB is available.

    Returns:
        True if reference FASTA exists
    """
    try:
        get_miniprot_ref_path()
        return True
    except FileNotFoundError:
        return False
