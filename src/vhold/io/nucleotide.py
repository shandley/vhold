"""Nucleotide FASTA detection and reading for vhold.

Provides utilities to detect whether a FASTA file contains nucleotide
(not protein) sequences, and to read nucleotide contigs for gene calling.
"""

from pathlib import Path

from Bio import SeqIO

from vhold.io.utils import open_maybe_gzip
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Characters that indicate nucleotide sequence
_NUCLEOTIDE_CHARS = set("ATGCNURYMKSWBDHV")

# Maximum characters to sample for detection
_DETECT_MAX_CHARS = 5000
_DETECT_MAX_SEQS = 10

# Fraction of residues that must be nucleotide chars to classify as nucleotide
_NUCLEOTIDE_THRESHOLD = 0.80


def is_nucleotide_fasta(path: str | Path) -> bool:
    """Detect if a FASTA file contains nucleotide (not protein) sequences.

    Reads first 10 sequences or 5000 residues and checks if >80% of
    characters are in the nucleotide alphabet {A, T, G, C, N, U, R, Y, M, K, ...}.

    Args:
        path: Path to FASTA file (plain or gzip)

    Returns:
        True if file appears to contain nucleotide sequences
    """
    path = Path(path)
    if not path.exists():
        return False

    total_chars = 0
    nuc_chars = 0
    seqs_read = 0

    try:
        with open_maybe_gzip(path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq).upper()
                for ch in seq:
                    if ch in _NUCLEOTIDE_CHARS:
                        nuc_chars += 1
                    total_chars += 1
                    if total_chars >= _DETECT_MAX_CHARS:
                        break
                seqs_read += 1
                if seqs_read >= _DETECT_MAX_SEQS or total_chars >= _DETECT_MAX_CHARS:
                    break
    except Exception:
        return False

    if total_chars == 0:
        return False

    return (nuc_chars / total_chars) >= _NUCLEOTIDE_THRESHOLD


def read_nucleotide_fasta(path: str | Path) -> dict[str, str]:
    """Read nucleotide FASTA into a dict of contig_id -> sequence.

    Supports plain text and gzip-compressed files.

    Args:
        path: Path to nucleotide FASTA file

    Returns:
        Dict mapping contig IDs to uppercase nucleotide sequences

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nucleotide FASTA not found: {path}")

    logger.info(f"Reading nucleotide contigs from {path}")

    contigs: dict[str, str] = {}
    with open_maybe_gzip(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            contig_id = record.id
            if contig_id in contigs:
                logger.warning(f"Duplicate contig ID '{contig_id}', skipping")
                continue
            contigs[contig_id] = str(record.seq).upper()

    logger.info(f"Read {len(contigs)} contigs")
    return contigs
