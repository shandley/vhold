"""GenBank file parsing for vhold.

Extracts protein sequences from GenBank CDS features with /translation
qualifiers. Preserves genomic coordinates (contig, start, end, strand)
and source annotations (product, gene, db_xref).
"""

from pathlib import Path

from Bio import SeqIO

from vhold.io.fasta import ProteinRecord
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def read_genbank(
    path: str | Path,
    max_length: int | None = None,
    min_length: int = 1,
) -> dict[str, ProteinRecord]:
    """Read protein sequences from a GenBank file.

    Extracts CDS features with /translation qualifiers. Each protein
    gets a ProteinRecord with genomic coordinates and source annotations.

    Args:
        path: Path to GenBank file (.gb, .gbk, .gbf, .genbank)
        max_length: Maximum protein sequence length (None for no limit)
        min_length: Minimum protein sequence length

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GenBank file not found: {path}")

    logger.info(f"Reading proteins from GenBank file: {path}")

    records = {}
    skipped_short = 0
    skipped_long = 0
    skipped_no_translation = 0

    for gb_record in SeqIO.parse(path, "genbank"):
        contig_id = gb_record.id

        for feature in gb_record.features:
            if feature.type != "CDS":
                continue

            # Get translation (protein sequence)
            translation = feature.qualifiers.get("translation")
            if not translation:
                skipped_no_translation += 1
                continue

            seq = translation[0].upper()

            # Filter by length
            if len(seq) < min_length:
                skipped_short += 1
                continue
            if max_length and len(seq) > max_length:
                skipped_long += 1
                continue

            # Determine protein ID
            protein_id = _get_protein_id(feature, contig_id, len(records))

            # Get source annotations
            product = feature.qualifiers.get("product", [""])[0]
            gene = feature.qualifiers.get("gene", [""])[0]
            db_xref = feature.qualifiers.get("db_xref", [])
            note = feature.qualifiers.get("note", [""])[0]
            locus_tag = feature.qualifiers.get("locus_tag", [""])[0]

            source_annotations = {}
            if product:
                source_annotations["product"] = product
            if gene:
                source_annotations["gene"] = gene
            if db_xref:
                source_annotations["db_xref"] = db_xref
            if note:
                source_annotations["note"] = note
            if locus_tag:
                source_annotations["locus_tag"] = locus_tag

            # Get coordinates
            start = int(feature.location.start)
            end = int(feature.location.end)
            strand = feature.location.strand  # +1 or -1

            # Build description
            description = product or gene or "hypothetical protein"

            records[protein_id] = ProteinRecord(
                id=protein_id,
                sequence=seq,
                description=description,
                contig=contig_id,
                start=start,
                end=end,
                strand=strand,
                source_annotations=source_annotations,
            )

    logger.info(f"Read {len(records)} proteins from {path}")

    if skipped_no_translation > 0:
        logger.debug(f"Skipped {skipped_no_translation} CDS features without /translation")
    if skipped_short > 0:
        logger.warning(f"Skipped {skipped_short} proteins shorter than {min_length} aa")
    if skipped_long > 0:
        logger.warning(f"Skipped {skipped_long} proteins longer than {max_length} aa")

    return records


def _get_protein_id(feature, contig_id: str, index: int) -> str:
    """Extract the best protein identifier from a CDS feature.

    Priority: protein_id > locus_tag > gene > contig_CDS_N

    Args:
        feature: BioPython SeqFeature
        contig_id: Parent contig/record ID
        index: Sequential index as fallback

    Returns:
        Protein identifier string
    """
    # Try protein_id first
    protein_id = feature.qualifiers.get("protein_id")
    if protein_id:
        return protein_id[0]

    # Try locus_tag
    locus_tag = feature.qualifiers.get("locus_tag")
    if locus_tag:
        return locus_tag[0]

    # Try gene name
    gene = feature.qualifiers.get("gene")
    if gene:
        return f"{contig_id}_{gene[0]}"

    # Fallback to contig + index
    return f"{contig_id}_CDS_{index}"
