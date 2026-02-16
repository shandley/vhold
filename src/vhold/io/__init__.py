"""Input/output handling for vhold."""

from pathlib import Path

from vhold.io.fasta import read_fasta, write_fasta, ProteinRecord
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "read_fasta",
    "read_input",
    "write_fasta",
    "ProteinRecord",
]

# Extension → format mapping
_FORMAT_MAP = {
    ".fasta": "fasta",
    ".fa": "fasta",
    ".faa": "fasta",
    ".fna": "fasta",
    ".gb": "genbank",
    ".gbk": "genbank",
    ".gbf": "genbank",
    ".genbank": "genbank",
    ".gff": "gff",
    ".gff3": "gff",
}


def read_input(
    path: str | Path,
    input_format: str | None = None,
    fasta_path: str | Path | None = None,
    max_length: int | None = None,
    min_length: int = 1,
) -> dict[str, ProteinRecord]:
    """Auto-detect input format and read protein sequences.

    Supports FASTA, GenBank, and GFF3 formats. Format is detected
    from the file extension unless explicitly specified.

    Args:
        path: Input file path
        input_format: Override format ('fasta', 'genbank', 'gff', or None for auto)
        fasta_path: Genome FASTA for GFF input (if not embedded)
        max_length: Maximum protein sequence length
        min_length: Minimum protein sequence length

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    path = Path(path)

    if input_format is None or input_format == "auto":
        fmt = _FORMAT_MAP.get(path.suffix.lower(), "fasta")
    else:
        fmt = input_format

    if fmt == "genbank":
        from vhold.io.genbank import read_genbank
        logger.info(f"Detected GenBank format for {path.name}")
        return read_genbank(path, max_length=max_length, min_length=min_length)

    elif fmt == "gff":
        from vhold.io.gff import read_gff
        logger.info(f"Detected GFF3 format for {path.name}")
        return read_gff(
            path, fasta_path=fasta_path,
            max_length=max_length, min_length=min_length,
        )

    else:
        return read_fasta(path, max_length=max_length, min_length=min_length)
