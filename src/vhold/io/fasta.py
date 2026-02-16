"""FASTA file parsing and writing for vhold."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProteinRecord:
    """A protein sequence record."""

    id: str
    sequence: str
    description: str = ""
    # Genomic position (populated from GenBank/GFF, None for FASTA)
    contig: str | None = None
    start: int | None = None
    end: int | None = None
    strand: int | None = None  # +1 or -1
    source_annotations: dict | None = None

    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.sequence)

    def to_fasta(self) -> str:
        """Convert to FASTA format string."""
        header = f">{self.id}"
        if self.description and self.description != self.id:
            header = f">{self.id} {self.description}"
        return f"{header}\n{self.sequence}\n"


def read_fasta(
    path: str | Path,
    max_length: int | None = None,
    min_length: int = 1,
) -> dict[str, ProteinRecord]:
    """Read protein sequences from a FASTA file.

    Args:
        path: Path to FASTA file
        max_length: Maximum sequence length (None for no limit)
        min_length: Minimum sequence length

    Returns:
        Dict mapping sequence IDs to ProteinRecord objects
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    logger.info(f"Reading sequences from {path}")

    records = {}
    skipped_short = 0
    skipped_long = 0

    for record in SeqIO.parse(path, "fasta"):
        seq = str(record.seq).upper()

        # Filter by length
        if len(seq) < min_length:
            skipped_short += 1
            continue

        if max_length and len(seq) > max_length:
            skipped_long += 1
            continue

        records[record.id] = ProteinRecord(
            id=record.id,
            sequence=seq,
            description=record.description,
        )

    logger.info(f"Read {len(records)} sequences from {path}")

    if skipped_short > 0:
        logger.warning(f"Skipped {skipped_short} sequences shorter than {min_length} aa")

    if skipped_long > 0:
        logger.warning(f"Skipped {skipped_long} sequences longer than {max_length} aa")

    return records


def iterate_fasta(path: str | Path) -> Iterator[ProteinRecord]:
    """Iterate over sequences in a FASTA file without loading all into memory.

    Args:
        path: Path to FASTA file

    Yields:
        ProteinRecord for each sequence
    """
    path = Path(path)

    for record in SeqIO.parse(path, "fasta"):
        yield ProteinRecord(
            id=record.id,
            sequence=str(record.seq).upper(),
            description=record.description,
        )


def write_fasta(
    records: dict[str, ProteinRecord] | dict[str, str] | list[ProteinRecord],
    path: str | Path,
) -> None:
    """Write protein sequences to a FASTA file.

    Args:
        records: Dict or list of ProteinRecord objects, or dict mapping id -> sequence
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bio_records = []

    if isinstance(records, dict):
        for key, value in records.items():
            if isinstance(value, ProteinRecord):
                rec = value
                bio_rec = SeqRecord(
                    Seq(rec.sequence),
                    id=rec.id,
                    description=rec.description if rec.description != rec.id else "",
                )
            else:
                # value is a string (sequence)
                bio_rec = SeqRecord(
                    Seq(value),
                    id=key,
                    description="",
                )
            bio_records.append(bio_rec)
    else:
        # list of ProteinRecord
        for rec in records:
            bio_rec = SeqRecord(
                Seq(rec.sequence),
                id=rec.id,
                description=rec.description if rec.description != rec.id else "",
            )
            bio_records.append(bio_rec)

    SeqIO.write(bio_records, path, "fasta")
    logger.info(f"Wrote {len(bio_records)} sequences to {path}")


def write_3di_fasta(
    sequences: dict[str, str],
    path: str | Path,
    descriptions: dict[str, str] | None = None,
) -> None:
    """Write 3Di sequences to a FASTA file.

    Args:
        sequences: Dict mapping IDs to 3Di sequences
        path: Output path
        descriptions: Optional descriptions for each sequence
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptions = descriptions or {}

    with open(path, "w") as f:
        for seq_id, seq in sequences.items():
            desc = descriptions.get(seq_id, "")
            if desc:
                f.write(f">{seq_id} {desc}\n")
            else:
                f.write(f">{seq_id}\n")
            f.write(f"{seq}\n")

    logger.info(f"Wrote {len(sequences)} 3Di sequences to {path}")
