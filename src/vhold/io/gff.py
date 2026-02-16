"""GFF3 file parsing for vhold.

Parses GFF3 format files with optional embedded ##FASTA section
(common in Prodigal output). Extracts CDS features and translates
nucleotide sequences to protein if no protein FASTA is provided.

Custom parser — no external GFF library dependency needed.
"""

import urllib.parse
from pathlib import Path

from Bio.Seq import Seq

from vhold.io.fasta import ProteinRecord
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def read_gff(
    gff_path: str | Path,
    fasta_path: str | Path | None = None,
    max_length: int | None = None,
    min_length: int = 1,
) -> dict[str, ProteinRecord]:
    """Read protein sequences from a GFF3 file.

    Handles two common scenarios:
    1. GFF3 with embedded ##FASTA section (Prodigal output)
    2. GFF3 with separate genome FASTA file

    CDS features are translated from nucleotide to protein unless
    a protein FASTA is explicitly provided via fasta_path.

    Args:
        gff_path: Path to GFF3 file
        fasta_path: Optional path to genome FASTA (if not embedded)
        max_length: Maximum protein sequence length
        min_length: Minimum protein sequence length

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    gff_path = Path(gff_path)
    if not gff_path.exists():
        raise FileNotFoundError(f"GFF3 file not found: {gff_path}")

    logger.info(f"Reading proteins from GFF3 file: {gff_path}")

    # Parse GFF3 and extract features + embedded FASTA
    features, embedded_fasta = _parse_gff3(gff_path)

    # Load genome sequences
    genome_seqs = {}
    if fasta_path:
        genome_seqs = _read_fasta_dict(Path(fasta_path))
    elif embedded_fasta:
        genome_seqs = _parse_fasta_text(embedded_fasta)

    if not genome_seqs:
        logger.warning("No genome sequences found. Need FASTA to translate CDS features.")
        return {}

    # Extract CDS features and translate
    records = {}
    skipped_short = 0
    skipped_long = 0
    skipped_no_seq = 0

    for feat in features:
        if feat["type"] != "CDS":
            continue

        contig = feat["seqid"]
        genome_seq = genome_seqs.get(contig)
        if genome_seq is None:
            skipped_no_seq += 1
            continue

        start = feat["start"]
        end = feat["end"]
        strand = feat["strand"]

        # Extract nucleotide sequence (GFF3 is 1-based, inclusive)
        nt_seq = genome_seq[start - 1:end]
        if strand == -1:
            nt_seq = str(Seq(nt_seq).reverse_complement())

        # Translate to protein
        try:
            protein_seq = str(Seq(nt_seq).translate(to_stop=True)).upper()
        except Exception:
            continue

        if len(protein_seq) < min_length:
            skipped_short += 1
            continue
        if max_length and len(protein_seq) > max_length:
            skipped_long += 1
            continue

        # Get protein ID from attributes
        protein_id = _get_gff_protein_id(feat, contig, len(records))
        product = feat["attributes"].get("product", "")
        gene = feat["attributes"].get("gene", "")
        name = feat["attributes"].get("Name", "")

        source_annotations = {}
        if product:
            source_annotations["product"] = product
        if gene:
            source_annotations["gene"] = gene

        description = product or name or gene or "hypothetical protein"

        records[protein_id] = ProteinRecord(
            id=protein_id,
            sequence=protein_seq,
            description=description,
            contig=contig,
            start=start,
            end=end,
            strand=strand,
            source_annotations=source_annotations,
        )

    logger.info(f"Read {len(records)} proteins from {gff_path}")

    if skipped_no_seq > 0:
        logger.warning(f"Skipped {skipped_no_seq} CDS features (contig not in FASTA)")
    if skipped_short > 0:
        logger.warning(f"Skipped {skipped_short} proteins shorter than {min_length} aa")
    if skipped_long > 0:
        logger.warning(f"Skipped {skipped_long} proteins longer than {max_length} aa")

    return records


def _parse_gff3(path: Path) -> tuple[list[dict], str]:
    """Parse a GFF3 file, returning features and any embedded FASTA.

    Args:
        path: Path to GFF3 file

    Returns:
        Tuple of (features list, embedded_fasta_text)
    """
    features = []
    fasta_lines = []
    in_fasta = False

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "##FASTA":
                in_fasta = True
                continue

            if in_fasta:
                fasta_lines.append(line)
                continue

            if line.startswith("#") or not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            seqid, source, feat_type, start, end, score, strand, phase, attrs = parts

            features.append({
                "seqid": seqid,
                "source": source,
                "type": feat_type,
                "start": int(start),
                "end": int(end),
                "score": score,
                "strand": 1 if strand == "+" else (-1 if strand == "-" else 0),
                "phase": phase,
                "attributes": _parse_gff_attributes(attrs),
            })

    fasta_text = "\n".join(fasta_lines) if fasta_lines else ""
    return features, fasta_text


def _parse_gff_attributes(attrs_str: str) -> dict[str, str]:
    """Parse GFF3 attribute column (key=value;key=value).

    Args:
        attrs_str: GFF3 column 9 string

    Returns:
        Dict of attribute key→value pairs
    """
    attrs = {}
    for item in attrs_str.split(";"):
        item = item.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key] = urllib.parse.unquote(value)
    return attrs


def _get_gff_protein_id(feat: dict, contig: str, index: int) -> str:
    """Extract protein ID from GFF3 attributes.

    Priority: ID > locus_tag > Name > contig_CDS_N

    Args:
        feat: Parsed GFF3 feature dict
        contig: Contig/seqid
        index: Sequential fallback index

    Returns:
        Protein identifier string
    """
    attrs = feat["attributes"]

    gff_id = attrs.get("ID", "")
    if gff_id:
        return gff_id

    locus_tag = attrs.get("locus_tag", "")
    if locus_tag:
        return locus_tag

    name = attrs.get("Name", "")
    if name:
        return name

    return f"{contig}_CDS_{index}"


def _read_fasta_dict(path: Path) -> dict[str, str]:
    """Read a FASTA file into a dict of id→sequence."""
    from Bio import SeqIO

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(path, "fasta")}


def _parse_fasta_text(text: str) -> dict[str, str]:
    """Parse FASTA-formatted text into a dict of id→sequence."""
    sequences = {}
    current_id = None
    current_seq = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq).upper()
            current_id = line[1:].split()[0]
            current_seq = []
        elif current_id and line:
            current_seq.append(line)

    if current_id:
        sequences[current_id] = "".join(current_seq).upper()

    return sequences
