"""Shared IO utilities for vhold: gzip support, sequence validation, duplicate ID handling."""

import gzip
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Standard amino acid alphabet + common ambiguity/special codes
# B=Asx(D/N), Z=Glx(E/Q), X=unknown, U=selenocysteine, O=pyrrolysine, J=Leu/Ile
_VALID_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZUOJ")

# Gzip magic bytes
_GZIP_MAGIC = b"\x1f\x8b"


@contextmanager
def open_maybe_gzip(path: Path | str, mode: str = "rt") -> Iterator[IO]:
    """Open a file, transparently decompressing gzip if detected.

    Detection uses gzip magic bytes (0x1f 0x8b) rather than file extension,
    so it works for files with any extension.

    Args:
        path: Path to file
        mode: Open mode ('rt' for text, 'rb' for binary)

    Yields:
        File handle
    """
    path = Path(path)

    with open(path, "rb") as f:
        magic = f.read(2)
    is_gzip = magic == _GZIP_MAGIC

    if is_gzip:
        with gzip.open(path, mode) as f:
            yield f
    else:
        with open(path, mode) as f:
            yield f


def validate_protein_sequence(
    sequence: str,
    protein_id: str,
    strip_stop: bool = True,
) -> tuple[str, list[str]]:
    """Validate and clean a protein sequence.

    Operations:
    1. Strip trailing '*' (stop codon) if strip_stop=True
    2. Check all characters are valid amino acid codes
    3. Warn on internal '*' characters
    4. Replace invalid characters with 'X'

    Args:
        sequence: Raw protein sequence (should already be uppercased)
        protein_id: ID for warning messages
        strip_stop: Whether to strip trailing stop codons

    Returns:
        Tuple of (cleaned_sequence, list_of_warnings)
    """
    warnings: list[str] = []
    seq = sequence.upper()

    # Strip trailing stop codons
    if strip_stop:
        seq = seq.rstrip("*")

    if not seq:
        warnings.append(f"{protein_id}: sequence empty after cleaning")
        return seq, warnings

    # Check for internal stop codons
    if "*" in seq:
        count = seq.count("*")
        warnings.append(
            f"{protein_id}: {count} internal stop codon(s) '*' found"
        )
        seq = seq.replace("*", "X")

    # Check for invalid characters
    invalid_chars = set(seq) - _VALID_AA_CHARS - {"X"}
    if invalid_chars:
        sorted_chars = sorted(invalid_chars)
        warnings.append(
            f"{protein_id}: invalid characters replaced with X: "
            f"{', '.join(repr(c) for c in sorted_chars)}"
        )
        seq = "".join(c if c in _VALID_AA_CHARS else "X" for c in seq)

    return seq, warnings


def check_duplicate_id(
    protein_id: str,
    existing_records: dict,
    contig_id: str = "",
) -> str:
    """Check for duplicate protein ID and generate unique suffix if needed.

    If protein_id already exists in existing_records, appends _dup1, _dup2, etc.
    until a unique ID is found.

    Args:
        protein_id: Proposed protein ID
        existing_records: Dict of already-parsed records
        contig_id: Contig for context in warning message

    Returns:
        Unique protein ID (original or suffixed)
    """
    if protein_id not in existing_records:
        return protein_id

    suffix = 1
    while f"{protein_id}_dup{suffix}" in existing_records:
        suffix += 1

    new_id = f"{protein_id}_dup{suffix}"
    context = f" (contig: {contig_id})" if contig_id else ""
    logger.warning(
        f"Duplicate protein ID '{protein_id}'{context}, renamed to '{new_id}'"
    )
    return new_id
