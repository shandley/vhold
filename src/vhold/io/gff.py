"""GFF3 file parsing for vhold.

Parses GFF3 format files with optional embedded ##FASTA section
(common in Prodigal output). Extracts CDS features and translates
nucleotide sequences to protein if no protein FASTA is provided.

Handles:
- Embedded ##FASTA section (Prodigal, geNomad)
- Separate genome FASTA file
- Pre-translated protein FASTA (.faa)
- Multi-row CDS features sharing same ID or Parent (spliced genes in NCBI GFF3)
- Translation table detection from attributes and Prodigal headers
- Gzip-compressed input files
- Sequence validation and duplicate ID detection

Custom parser — no external GFF library dependency needed.
"""

import re
import urllib.parse
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

from vhold.io.fasta import ProteinRecord
from vhold.io.utils import check_duplicate_id, open_maybe_gzip, validate_protein_sequence
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def read_gff(
    gff_path: str | Path,
    fasta_path: str | Path | None = None,
    max_length: int | None = None,
    min_length: int = 1,
) -> dict[str, ProteinRecord]:
    """Read protein sequences from a GFF3 file.

    Handles three common scenarios:
    1. GFF3 with embedded ##FASTA section (Prodigal output)
    2. GFF3 with separate genome FASTA file (nucleotide translation)
    3. GFF3 with separate protein FASTA file (.faa, auto-detected)

    Multi-row CDS features sharing the same ID (NCBI GFF3 for spliced genes)
    are automatically merged before translation.

    Supports plain text and gzip-compressed files.

    Args:
        gff_path: Path to GFF3 file (plain or .gz)
        fasta_path: Optional path to genome or protein FASTA (if not embedded)
        max_length: Maximum protein sequence length
        min_length: Minimum protein sequence length

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    gff_path = Path(gff_path)
    if not gff_path.exists():
        raise FileNotFoundError(f"GFF3 file not found: {gff_path}")

    logger.info(f"Reading proteins from GFF3 file: {gff_path}")

    # Parse GFF3 and extract features + embedded FASTA + header comments
    features, embedded_fasta, header_comments = _parse_gff3(gff_path)

    # Merge multi-row CDS features (e.g., NCBI spliced genes)
    features = _merge_multipart_cds(features)

    # Load sequences from external FASTA or embedded section
    fasta_seqs: dict[str, str] = {}
    if fasta_path:
        fasta_seqs = _read_fasta_dict(Path(fasta_path))
    elif embedded_fasta:
        fasta_seqs = _parse_fasta_text(embedded_fasta)

    if not fasta_seqs:
        logger.warning("No sequences found. Need FASTA to process CDS features.")
        return {}

    # Detect whether FASTA contains protein or nucleotide sequences
    is_protein = _is_protein_fasta(fasta_seqs)
    if is_protein:
        logger.info("Detected protein FASTA — using sequences directly (skipping translation)")
        return _build_records_from_protein_fasta(
            features, fasta_seqs, min_length, max_length, gff_path,
        )

    # Detect translation table from features or Prodigal header
    transl_table = _detect_translation_table(features, header_comments)
    if transl_table != 1:
        logger.info(f"Using translation table {transl_table}")

    # Nucleotide FASTA — translate CDS features
    records: dict[str, ProteinRecord] = {}
    skipped_short = 0
    skipped_long = 0
    skipped_no_seq = 0

    for feat in features:
        if feat["type"] != "CDS":
            continue

        contig = feat["seqid"]
        genome_seq = fasta_seqs.get(contig)
        if genome_seq is None:
            skipped_no_seq += 1
            continue

        strand = feat["strand"]

        # Extract nucleotide sequence from intervals (handles multi-part CDS)
        intervals = feat.get("intervals", [(feat["start"], feat["end"])])
        nt_parts = []
        for iv_start, iv_end in intervals:
            # GFF3 is 1-based inclusive
            nt_parts.append(genome_seq[iv_start - 1:iv_end])

        if strand == -1:
            # Reverse complement: reverse the parts order then RC the joined sequence
            nt_seq = str(Seq("".join(reversed(nt_parts))).reverse_complement())
        else:
            nt_seq = "".join(nt_parts)

        # Translate to protein
        try:
            protein_seq = str(
                Seq(nt_seq).translate(table=transl_table, to_stop=True)
            ).upper()
        except Exception as e:
            logger.debug(f"Translation failed for {feat.get('attributes', {}).get('ID', '?')}: {e}")
            continue

        # Validate sequence
        protein_id_hint = feat["attributes"].get("ID", contig)
        protein_seq, warnings = validate_protein_sequence(protein_seq, protein_id_hint)
        for w in warnings:
            logger.warning(w)
        if not protein_seq:
            continue

        if len(protein_seq) < min_length:
            skipped_short += 1
            continue
        if max_length and len(protein_seq) > max_length:
            skipped_long += 1
            continue

        # Get protein ID from attributes
        protein_id = _get_gff_protein_id(feat, contig, len(records))
        protein_id = check_duplicate_id(protein_id, records)

        product = feat["attributes"].get("product", "")
        gene = feat["attributes"].get("gene", "")
        name = feat["attributes"].get("Name", "")

        source_annotations: dict = {}
        if product:
            source_annotations["product"] = product
        if gene:
            source_annotations["gene"] = gene
        if len(intervals) > 1:
            source_annotations["intervals"] = intervals
            source_annotations["is_spliced"] = True

        description = product or name or gene or "hypothetical protein"

        records[protein_id] = ProteinRecord(
            id=protein_id,
            sequence=protein_seq,
            description=description,
            contig=contig,
            start=feat["start"],
            end=feat["end"],
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


def _parse_gff3(path: Path) -> tuple[list[dict], str, list[str]]:
    """Parse a GFF3 file, returning features, embedded FASTA, and header comments.

    Args:
        path: Path to GFF3 file

    Returns:
        Tuple of (features_list, embedded_fasta_text, header_comments)
    """
    features = []
    fasta_lines: list[str] = []
    header_comments: list[str] = []
    in_fasta = False

    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "##FASTA":
                in_fasta = True
                continue

            if in_fasta:
                fasta_lines.append(line)
                continue

            if line.startswith("#"):
                header_comments.append(line)
                continue

            if not line.strip():
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
    return features, fasta_text, header_comments


def _merge_multipart_cds(features: list[dict]) -> list[dict]:
    """Merge multi-row CDS features sharing the same ID or Parent.

    NCBI GFF3 convention: spliced CDS features share the same ID
    attribute. The nucleotide intervals must be concatenated (in
    strand-aware order) before translation.

    When CDS rows lack an ID attribute but share a Parent attribute
    (NCBI 3-tier hierarchy: gene → mRNA → CDS), they are grouped
    by (seqid, Parent) instead.

    Args:
        features: Parsed GFF3 feature dicts from _parse_gff3()

    Returns:
        Features with multi-part CDS merged. Each merged CDS has
        an 'intervals' key: [(start, end), ...] sorted by start.
    """
    cds_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    non_cds: list[dict] = []

    for feat in features:
        if feat["type"] == "CDS":
            feat_id = feat["attributes"].get("ID", "")
            seqid = feat["seqid"]
            if feat_id:
                cds_by_key[(seqid, feat_id)].append(feat)
            elif feat["attributes"].get("Parent"):
                # No ID but has Parent — group by Parent (NCBI 3-tier hierarchy)
                parent = feat["attributes"]["Parent"]
                cds_by_key[(seqid, parent)].append(feat)
            else:
                # No ID and no Parent — treat as independent
                feat["intervals"] = [(feat["start"], feat["end"])]
                non_cds.append(feat)
        else:
            non_cds.append(feat)

    merged: list[dict] = []
    for (_seqid, _feat_id), parts in cds_by_key.items():
        if len(parts) == 1:
            f = parts[0]
            f["intervals"] = [(f["start"], f["end"])]
            merged.append(f)
        else:
            # Multi-part CDS: sort by start position
            parts.sort(key=lambda x: x["start"])

            # Use first part as base, merge attributes
            base = parts[0].copy()
            intervals = [(p["start"], p["end"]) for p in parts]
            base["start"] = min(p["start"] for p in parts)
            base["end"] = max(p["end"] for p in parts)
            base["intervals"] = intervals

            logger.debug(
                f"Merged {len(parts)} CDS rows for {_feat_id} "
                f"({len(intervals)} exons)"
            )
            merged.append(base)

    return non_cds + merged


def _detect_translation_table(
    features: list[dict], header_comments: list[str]
) -> int:
    """Detect the translation table from GFF features or header comments.

    Priority:
    1. Feature attribute: transl_table=N
    2. Prodigal header comment: transl_table=N
    3. Default: 1 (standard genetic code)

    Args:
        features: Parsed GFF3 features
        header_comments: Lines starting with # from GFF header

    Returns:
        Translation table number (e.g., 1, 4, 11, 15)
    """
    # Check feature attributes
    for feat in features:
        tt = feat["attributes"].get("transl_table", "")
        if tt and tt.isdigit():
            return int(tt)

    # Check header comments (Prodigal format: "transl_table=11" in Model Data line)
    for comment in header_comments:
        match = re.search(r"transl_table=(\d+)", comment)
        if match:
            return int(match.group(1))

    return 1  # Default standard genetic code


def _parse_gff_attributes(attrs_str: str) -> dict[str, str]:
    """Parse GFF3 attribute column (key=value;key=value).

    Args:
        attrs_str: GFF3 column 9 string

    Returns:
        Dict of attribute key->value pairs
    """
    attrs: dict[str, str] = {}
    for item in attrs_str.split(";"):
        item = item.strip()
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        attrs[key] = urllib.parse.unquote(value)
    return attrs


def _get_gff_protein_id(feat: dict, contig: str, index: int) -> str:
    """Extract protein ID from GFF3 attributes.

    Priority: ID > protein_id > locus_tag > Name > contig_CDS_N

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

    protein_id = attrs.get("protein_id", "")
    if protein_id:
        return protein_id

    locus_tag = attrs.get("locus_tag", "")
    if locus_tag:
        return locus_tag

    name = attrs.get("Name", "")
    if name:
        return name

    return f"{contig}_CDS_{index}"


def _read_fasta_dict(path: Path) -> dict[str, str]:
    """Read a FASTA file into a dict of id->sequence.

    Supports gzip-compressed files.
    """
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    with open_maybe_gzip(path) as handle:
        return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(handle, "fasta")}


def _parse_fasta_text(text: str) -> dict[str, str]:
    """Parse FASTA-formatted text into a dict of id->sequence."""
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

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


# Characters that only appear in amino acid sequences, never in nucleotide
_PROTEIN_ONLY_CHARS = set("EFILPQZ")


def _is_protein_fasta(seqs: dict[str, str]) -> bool:
    """Detect whether sequences are protein (amino acid) or nucleotide.

    Checks for amino acid-only characters (E, F, I, L, P, Q) that
    cannot appear in nucleotide sequences (even with IUPAC ambiguity).

    Args:
        seqs: Dict of id->sequence

    Returns:
        True if sequences appear to be protein
    """
    for seq in seqs.values():
        upper = seq.upper()
        if _PROTEIN_ONLY_CHARS & set(upper):
            return True
    return False


def _build_records_from_protein_fasta(
    features: list[dict],
    protein_seqs: dict[str, str],
    min_length: int,
    max_length: int | None,
    gff_path: Path,
) -> dict[str, ProteinRecord]:
    """Build ProteinRecords by matching GFF features to protein sequences.

    Prodigal protein FASTA IDs typically match the GFF feature IDs.
    For each CDS feature, look up the protein sequence by ID and
    attach genomic coordinates from the GFF.

    Args:
        features: Parsed GFF3 features
        protein_seqs: Dict of protein_id->sequence from .faa file
        min_length: Minimum protein length filter
        max_length: Maximum protein length filter
        gff_path: Path to GFF3 file (for logging)

    Returns:
        Dict mapping protein IDs to ProteinRecord objects
    """
    records: dict[str, ProteinRecord] = {}
    skipped_short = 0
    skipped_long = 0

    for feat in features:
        if feat["type"] != "CDS":
            continue

        contig = feat["seqid"]
        protein_id = _get_gff_protein_id(feat, contig, len(records))

        # Look up protein sequence by feature ID
        protein_seq = protein_seqs.get(protein_id)
        if protein_seq is None:
            continue

        # Validate and clean sequence (also strips trailing *)
        protein_seq, warnings = validate_protein_sequence(protein_seq, protein_id)
        for w in warnings:
            logger.warning(w)
        if not protein_seq:
            continue

        if len(protein_seq) < min_length:
            skipped_short += 1
            continue
        if max_length and len(protein_seq) > max_length:
            skipped_long += 1
            continue

        # Handle duplicate IDs
        safe_id = check_duplicate_id(protein_id, records)

        product = feat["attributes"].get("product", "")
        gene = feat["attributes"].get("gene", "")
        name = feat["attributes"].get("Name", "")

        source_annotations: dict = {}
        if product:
            source_annotations["product"] = product
        if gene:
            source_annotations["gene"] = gene

        description = product or name or gene or "hypothetical protein"

        records[safe_id] = ProteinRecord(
            id=safe_id,
            sequence=protein_seq,
            description=description,
            contig=contig,
            start=feat["start"],
            end=feat["end"],
            strand=feat["strand"],
            source_annotations=source_annotations,
        )

    logger.info(f"Read {len(records)} proteins from {gff_path} (protein FASTA mode)")

    if skipped_short > 0:
        logger.warning(f"Skipped {skipped_short} proteins shorter than {min_length} aa")
    if skipped_long > 0:
        logger.warning(f"Skipped {skipped_long} proteins longer than {max_length} aa")

    return records
