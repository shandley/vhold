"""Input/output handling for vhold."""

from pathlib import Path

from vhold.io.fasta import ProteinRecord, read_fasta, write_fasta
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
    ".gbff": "genbank",
    ".genbank": "genbank",
    ".gff": "gff",
    ".gff3": "gff",
}


def _detect_format(path: Path) -> str:
    """Detect input format from file extension, stripping .gz if present.

    Args:
        path: Input file path

    Returns:
        Format string: 'fasta', 'genbank', or 'gff'
    """
    suffix = path.suffix.lower()
    if suffix == ".gz":
        # Strip .gz and use the next suffix (e.g., .gb.gz → .gb)
        suffix = Path(path.stem).suffix.lower()
    fmt = _FORMAT_MAP.get(suffix)
    if fmt is None:
        if suffix:
            logger.warning(
                f"Unknown file extension '{suffix}', defaulting to FASTA format. "
                f"Use --input-format to override."
            )
        return "fasta"
    return fmt


def read_input(
    path: str | Path,
    input_format: str | None = None,
    fasta_path: str | Path | None = None,
    max_length: int | None = None,
    min_length: int = 1,
    include_mat_peptide: bool = True,
) -> dict[str, ProteinRecord]:
    """Auto-detect input format and read protein sequences.

    Supports FASTA, GenBank, and GFF3 formats. Format is detected
    from the file extension unless explicitly specified. Gzip-compressed
    files (.gz) are handled transparently.

    Args:
        path: Input file path (plain or .gz)
        input_format: Override format ('fasta', 'genbank', 'gff', or None for auto)
        fasta_path: Genome FASTA for GFF input (if not embedded)
        max_length: Maximum protein sequence length
        min_length: Minimum protein sequence length
        include_mat_peptide: Extract mat_peptide features from GenBank (default: True)

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    path = Path(path)

    if input_format is None or input_format == "auto":
        fmt = _detect_format(path)
    else:
        fmt = input_format

    if fmt == "genbank":
        from vhold.io.genbank import read_genbank
        logger.info(f"Detected GenBank format for {path.name}")
        return read_genbank(
            path, max_length=max_length, min_length=min_length,
            include_mat_peptide=include_mat_peptide,
        )

    elif fmt == "gff":
        from vhold.io.gff import read_gff
        logger.info(f"Detected GFF3 format for {path.name}")
        return read_gff(
            path, fasta_path=fasta_path,
            max_length=max_length, min_length=min_length,
        )

    else:
        return read_fasta(path, max_length=max_length, min_length=min_length)
