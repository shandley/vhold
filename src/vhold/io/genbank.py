"""GenBank file parsing for vhold.

Extracts protein sequences from GenBank CDS and mat_peptide features
with /translation qualifiers. Preserves genomic coordinates (contig,
start, end, strand) and source annotations (product, gene, db_xref).

Handles:
- Standard CDS features
- mat_peptide features (Flavivirus polyprotein cleavage products)
- Compound locations / join() (spliced genes like HIV tat/rev)
- Gzip-compressed input files
- Duplicate protein IDs (multi-segment genomes)
- Sequence validation
"""

from pathlib import Path

from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation

from vhold.io.fasta import ProteinRecord
from vhold.io.utils import check_duplicate_id, open_maybe_gzip, validate_protein_sequence
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def read_genbank(
    path: str | Path,
    max_length: int | None = None,
    min_length: int = 1,
    include_mat_peptide: bool = True,
) -> dict[str, ProteinRecord]:
    """Read protein sequences from a GenBank file.

    Extracts CDS features with /translation qualifiers. When
    include_mat_peptide is True (default), also extracts mat_peptide
    features and excludes parent polyprotein CDS that have been
    cleaved into mature peptides.

    Supports plain text and gzip-compressed files.

    Args:
        path: Path to GenBank file (.gb, .gbk, .gbf, .genbank, or .gz)
        max_length: Maximum protein sequence length (None for no limit)
        min_length: Minimum protein sequence length
        include_mat_peptide: Extract mat_peptide features and exclude
            parent polyprotein CDS (default: True)

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GenBank file not found: {path}")

    logger.info(f"Reading proteins from GenBank file: {path}")

    records: dict[str, ProteinRecord] = {}
    skipped_short = 0
    skipped_long = 0
    skipped_no_translation = 0

    with open_maybe_gzip(path) as handle:
        for gb_record in SeqIO.parse(handle, "genbank"):
            contig_id = gb_record.id

            # Phase 1: Extract CDS features
            cds_records: dict[str, ProteinRecord] = {}
            cds_ranges: list[tuple[int, int, str]] = []  # (start, end, protein_id)

            for feature in gb_record.features:
                if feature.type != "CDS":
                    continue

                translation = feature.qualifiers.get("translation")
                if not translation:
                    skipped_no_translation += 1
                    continue

                seq = translation[0].upper()
                seq, warnings = validate_protein_sequence(seq, "CDS")
                for w in warnings:
                    logger.warning(w)
                if not seq:
                    continue

                if len(seq) < min_length:
                    skipped_short += 1
                    continue
                if max_length and len(seq) > max_length:
                    skipped_long += 1
                    continue

                protein_id = _get_protein_id(feature, contig_id, len(records) + len(cds_records))
                protein_id = check_duplicate_id(protein_id, records, contig_id)
                protein_id = check_duplicate_id(protein_id, cds_records, contig_id)

                source_annotations = _extract_source_annotations(feature)
                start, end, strand, exon_parts = _extract_coordinates(feature)

                if len(exon_parts) > 1:
                    source_annotations["exon_parts"] = exon_parts
                    source_annotations["is_spliced"] = True

                description = (
                    source_annotations.get("product")
                    or source_annotations.get("gene")
                    or "hypothetical protein"
                )

                rec = ProteinRecord(
                    id=protein_id,
                    sequence=seq,
                    description=description,
                    contig=contig_id,
                    start=start,
                    end=end,
                    strand=strand,
                    source_annotations=source_annotations,
                    feature_type="CDS",
                )
                cds_records[protein_id] = rec
                cds_ranges.append((start, end, protein_id))

            # Phase 2: Extract mat_peptide features
            mat_records: dict[str, ProteinRecord] = {}
            parent_cds_ids: set[str] = set()  # CDS IDs to exclude

            if include_mat_peptide:
                mat_count_by_parent: dict[str, int] = {}

                for feature in gb_record.features:
                    if feature.type != "mat_peptide":
                        continue

                    translation = feature.qualifiers.get("translation")
                    if not translation:
                        skipped_no_translation += 1
                        continue

                    seq = translation[0].upper()
                    seq, warnings = validate_protein_sequence(seq, "mat_peptide")
                    for w in warnings:
                        logger.warning(w)
                    if not seq:
                        continue

                    if len(seq) < min_length:
                        skipped_short += 1
                        continue
                    if max_length and len(seq) > max_length:
                        skipped_long += 1
                        continue

                    protein_id = _get_protein_id(
                        feature, contig_id,
                        len(records) + len(cds_records) + len(mat_records),
                    )
                    all_so_far = {**records, **cds_records, **mat_records}
                    protein_id = check_duplicate_id(protein_id, all_so_far, contig_id)

                    source_annotations = _extract_source_annotations(feature)
                    source_annotations["feature_type"] = "mat_peptide"
                    start, end, strand, exon_parts = _extract_coordinates(feature)

                    # Find parent CDS by coordinate containment
                    parent_id = _find_parent_cds(start, end, cds_ranges)
                    if parent_id:
                        mat_count_by_parent.setdefault(parent_id, 0)
                        mat_count_by_parent[parent_id] += 1
                        source_annotations["parent_cds"] = parent_id

                    description = (
                        source_annotations.get("product")
                        or source_annotations.get("gene")
                        or "hypothetical protein"
                    )

                    mat_records[protein_id] = ProteinRecord(
                        id=protein_id,
                        sequence=seq,
                        description=description,
                        contig=contig_id,
                        start=start,
                        end=end,
                        strand=strand,
                        source_annotations=source_annotations,
                        feature_type="mat_peptide",
                    )

                # Exclude parent CDS polyproteins that have >=2 mat_peptide children
                for parent_id, count in mat_count_by_parent.items():
                    if count >= 2:
                        parent_cds_ids.add(parent_id)
                    elif count == 1:
                        logger.warning(
                            f"CDS '{parent_id}' has only 1 mat_peptide child — "
                            f"keeping both (may indicate partial annotation)"
                        )

                if parent_cds_ids:
                    logger.info(
                        f"Excluding {len(parent_cds_ids)} polyprotein CDS in favor of "
                        f"{len(mat_records)} mature peptides"
                    )

            # Merge: CDS (minus excluded polyproteins) + mat_peptides
            for pid, rec in cds_records.items():
                if pid not in parent_cds_ids:
                    records[pid] = rec

            for pid, rec in mat_records.items():
                records[pid] = rec

    logger.info(f"Read {len(records)} proteins from {path}")

    if skipped_no_translation > 0:
        logger.debug(
            f"Skipped {skipped_no_translation} features without /translation"
        )
    if skipped_short > 0:
        logger.warning(f"Skipped {skipped_short} proteins shorter than {min_length} aa")
    if skipped_long > 0:
        logger.warning(f"Skipped {skipped_long} proteins longer than {max_length} aa")

    return records


def _extract_coordinates(feature) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Extract coordinates from a BioPython feature, handling compound locations.

    Converts from BioPython 0-based half-open to 1-based inclusive coordinates,
    matching GFF3 convention. All internal coordinates in vHold are 1-based.

    Args:
        feature: BioPython SeqFeature

    Returns:
        Tuple of (start, end, strand, exon_parts) where:
        - start: leftmost coordinate (1-based inclusive)
        - end: rightmost coordinate (1-based inclusive)
        - strand: +1 or -1
        - exon_parts: list of (start, end) for each exon/part (1-based)
    """
    if isinstance(feature.location, CompoundLocation):
        # BioPython 0-based half-open → 1-based inclusive: start + 1, end unchanged
        parts = [(int(part.start) + 1, int(part.end)) for part in feature.location.parts]
        start = min(p[0] for p in parts)
        end = max(p[1] for p in parts)
        strand = feature.location.strand
        return start, end, strand, parts
    else:
        start = int(feature.location.start) + 1  # 0-based → 1-based
        end = int(feature.location.end)
        strand = feature.location.strand
        return start, end, strand, [(start, end)]


def _extract_source_annotations(feature) -> dict:
    """Extract source annotation qualifiers from a GenBank feature.

    Args:
        feature: BioPython SeqFeature

    Returns:
        Dict of annotation key-value pairs
    """
    product = feature.qualifiers.get("product", [""])[0]
    gene = feature.qualifiers.get("gene", [""])[0]
    db_xref = feature.qualifiers.get("db_xref", [])
    note = feature.qualifiers.get("note", [""])[0]
    locus_tag = feature.qualifiers.get("locus_tag", [""])[0]

    source_annotations: dict = {}
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

    return source_annotations


def _find_parent_cds(
    mp_start: int, mp_end: int, cds_ranges: list[tuple[int, int, str]]
) -> str | None:
    """Find the parent CDS whose range contains a mat_peptide.

    Args:
        mp_start: mat_peptide start coordinate
        mp_end: mat_peptide end coordinate
        cds_ranges: List of (start, end, protein_id) for CDS features

    Returns:
        Parent CDS protein_id or None
    """
    for cds_start, cds_end, cds_id in cds_ranges:
        if mp_start >= cds_start and mp_end <= cds_end:
            return cds_id
    return None


def _get_protein_id(feature, contig_id: str, index: int) -> str:
    """Extract the best protein identifier from a GenBank feature.

    Priority: protein_id > locus_tag > gene > contig_TYPE_N

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

    # Fallback to contig + feature type + index
    feat_type = feature.type if feature.type != "CDS" else "CDS"
    return f"{contig_id}_{feat_type}_{index}"
