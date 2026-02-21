"""Gene calling for nucleotide sequences.

Provides unified gene calling using Pyrodigal-gv (unspliced ORFs) and
optionally miniprot (splice-aware protein-to-genome alignment). Results
are merged and converted to ProteinRecord objects for the vHold pipeline.

Pyrodigal-gv handles >97% of viral genes. miniprot catches spliced genes
in eukaryotic viruses (adenovirus, polyomavirus, retrovirus, etc.).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from vhold.io.fasta import ProteinRecord
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalledGene:
    """A gene called by Pyrodigal or miniprot."""

    contig: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    strand: int  # +1 or -1
    protein_sequence: str
    gene_id: str  # e.g., "contig_1" or "contig_mp_1"
    caller: str  # "pyrodigal" or "miniprot"
    partial_5prime: bool = False
    partial_3prime: bool = False
    is_spliced: bool = False
    exon_parts: list[tuple[int, int]] = field(default_factory=list)


def _check_pyrodigal_available() -> bool:
    """Check if pyrodigal is importable."""
    return importlib.util.find_spec("pyrodigal") is not None


class PyrodigalCaller:
    """Wrapper for Pyrodigal-gv gene calling."""

    def __init__(
        self,
        translation_table: int = 11,
        closed: bool = False,
        min_gene: int = 90,
        meta: bool = True,
    ):
        """Initialize Pyrodigal gene finder.

        Args:
            translation_table: Genetic code (11=bacterial/viral, 1=eukaryotic)
            closed: Only call complete genes (ATG...stop)
            min_gene: Minimum gene length in nucleotides
            meta: Use metagenomic mode (default, includes Prodigal-gv viral models)
        """
        if not _check_pyrodigal_available():
            raise ImportError(
                "pyrodigal is required for gene calling. "
                "Install with: pip install 'vhold[genecalling]'"
            )

        self.translation_table = translation_table
        self.closed = closed
        self.min_gene = min_gene
        self.meta = meta

    def call_genes(self, contigs: dict[str, str]) -> list[CalledGene]:
        """Call genes on all contigs using Pyrodigal metagenomic mode.

        Args:
            contigs: Dict of contig_id -> nucleotide sequence

        Returns:
            List of CalledGene objects
        """
        import pyrodigal

        if self.meta:
            gene_finder = pyrodigal.GeneFinder(meta=True)
        else:
            gene_finder = pyrodigal.GeneFinder(meta=False)
            # For single-mode, train on concatenated sequences
            training_seq = "".join(contigs.values())
            gene_finder.train(training_seq, translation_table=self.translation_table)

        genes: list[CalledGene] = []
        total_genes = 0

        for contig_id, sequence in contigs.items():
            if len(sequence) < self.min_gene:
                continue

            predicted = gene_finder.find_genes(sequence)

            gene_num = 0
            for pred in predicted:
                # Filter by minimum gene length
                gene_length = abs(pred.end - pred.begin)
                if gene_length < self.min_gene:
                    continue

                # Filter closed genes if requested
                if self.closed and (pred.partial_begin or pred.partial_end):
                    continue

                # Translate to protein
                protein_seq = pred.translate(translation_table=self.translation_table)
                if not protein_seq:
                    continue

                # Strip trailing stop codon
                protein_str = str(protein_seq).rstrip("*")
                if not protein_str:
                    continue

                gene_num += 1
                strand = 1 if pred.strand == 1 else -1

                # Pyrodigal uses 1-based coordinates
                genes.append(
                    CalledGene(
                        contig=contig_id,
                        start=min(pred.begin, pred.end),
                        end=max(pred.begin, pred.end),
                        strand=strand,
                        protein_sequence=protein_str,
                        gene_id=f"{contig_id}_{gene_num}",
                        caller="pyrodigal",
                        partial_5prime=pred.partial_begin,
                        partial_3prime=pred.partial_end,
                    )
                )

            total_genes += gene_num

        logger.info(
            f"Pyrodigal called {total_genes} genes from {len(contigs)} contigs"
        )
        return genes


class MiniprotCaller:
    """Wrapper for miniprot splice-aware protein-to-genome alignment."""

    def __init__(self, ref_db: Path, threads: int = 4):
        """Initialize miniprot caller.

        Args:
            ref_db: Path to reference protein FASTA
            threads: Number of threads
        """
        from vhold.utils.external import get_miniprot

        self.tool = get_miniprot()
        self.ref_db = ref_db
        self.threads = threads

    def call_genes(
        self, contigs_fasta: Path, output_dir: Path
    ) -> list[CalledGene]:
        """Run miniprot and parse GFF output.

        Args:
            contigs_fasta: Path to nucleotide FASTA of contigs
            output_dir: Directory for intermediate files

        Returns:
            List of CalledGene objects (potentially spliced)
        """
        gff_out = output_dir / "miniprot_alignments.gff"

        args = [
            "--gff",
            "-t", str(self.threads),
            str(contigs_fasta),
            str(self.ref_db),
        ]

        result = self.tool.run(args, check=False)
        if not result.success:
            logger.warning(f"miniprot failed: {result.stderr}")
            return []

        # Write GFF output
        gff_out.write_text(result.stdout)

        return self._parse_gff(result.stdout)

    def _parse_gff(self, gff_text: str) -> list[CalledGene]:
        """Parse miniprot GFF3 output into CalledGene objects.

        miniprot outputs mRNA and CDS features. Protein sequence is in
        the Target attribute or can be derived from the alignment.
        """
        from collections import defaultdict

        genes: list[CalledGene] = []
        # Group CDS by parent mRNA
        mrna_cds: dict[str, list[dict]] = defaultdict(list)
        mrna_info: dict[str, dict] = {}
        gene_counter: dict[str, int] = defaultdict(int)

        for line in gff_text.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) != 9:
                continue

            seqid, source, feat_type, start, end, score, strand, phase, attrs_str = parts

            # Parse attributes
            attrs: dict[str, str] = {}
            for attr in attrs_str.split(";"):
                if "=" in attr:
                    key, val = attr.split("=", 1)
                    attrs[key] = val

            if feat_type == "mRNA":
                mrna_id = attrs.get("ID", "")
                mrna_info[mrna_id] = {
                    "seqid": seqid,
                    "strand": 1 if strand == "+" else -1,
                    "start": int(start),
                    "end": int(end),
                    "identity": attrs.get("Identity", ""),
                    "target": attrs.get("Target", ""),
                }
            elif feat_type == "CDS":
                parent = attrs.get("Parent", "")
                mrna_cds[parent].append({
                    "start": int(start),
                    "end": int(end),
                    "phase": int(phase) if phase.isdigit() else 0,
                })

        # Build genes from mRNA + CDS
        for mrna_id, info in mrna_info.items():
            cds_parts = mrna_cds.get(mrna_id, [])
            if not cds_parts:
                continue

            # Sort CDS parts by position
            cds_parts.sort(key=lambda x: x["start"])

            # Reconstruct protein sequence from CDS
            # miniprot provides protein in the Rank/Identity attrs;
            # we need to extract from the alignment or use the translated CDS
            # For now, we'll need the sequence from the genome + translation
            exon_parts = [(cds["start"], cds["end"]) for cds in cds_parts]
            is_spliced = len(exon_parts) > 1

            contig = info["seqid"]
            gene_counter[contig] += 1
            gene_id = f"{contig}_mp_{gene_counter[contig]}"

            # Get overall span
            overall_start = min(cds["start"] for cds in cds_parts)
            overall_end = max(cds["end"] for cds in cds_parts)

            genes.append(
                CalledGene(
                    contig=contig,
                    start=overall_start,
                    end=overall_end,
                    strand=info["strand"],
                    protein_sequence="",  # Filled by _translate_miniprot_genes
                    gene_id=gene_id,
                    caller="miniprot",
                    is_spliced=is_spliced,
                    exon_parts=exon_parts,
                )
            )

        logger.info(f"miniprot found {len(genes)} alignments")
        return genes


def _translate_miniprot_genes(
    genes: list[CalledGene],
    contigs: dict[str, str],
    translation_table: int = 1,
) -> list[CalledGene]:
    """Translate miniprot gene calls using genome sequence.

    Extracts and concatenates exon nucleotides, then translates.
    Uses standard eukaryotic code (table 1) by default since miniprot
    targets eukaryotic viruses.

    Args:
        genes: CalledGene objects from miniprot (protein_sequence empty)
        contigs: Dict of contig_id -> nucleotide sequence
        translation_table: Genetic code table number

    Returns:
        Filtered list with protein_sequence filled in
    """
    from Bio.Data.CodonTable import TranslationError
    from Bio.Seq import Seq

    translated = []
    for gene in genes:
        seq = contigs.get(gene.contig)
        if seq is None:
            logger.warning(f"Contig {gene.contig} not found for {gene.gene_id}")
            continue

        # Extract and concatenate exon sequences
        parts = gene.exon_parts if gene.exon_parts else [(gene.start, gene.end)]
        nuc_parts = []
        for start, end in parts:
            # Convert 1-based to 0-based for slicing
            nuc_parts.append(seq[start - 1 : end])

        nuc_seq = "".join(nuc_parts)

        # Reverse complement if minus strand
        if gene.strand == -1:
            nuc_seq = str(Seq(nuc_seq).reverse_complement())

        # Translate
        try:
            protein = str(Seq(nuc_seq).translate(table=translation_table))
            protein = protein.rstrip("*")
            if protein:
                gene.protein_sequence = protein
                translated.append(gene)
        except TranslationError as e:
            logger.debug(f"Translation failed for {gene.gene_id}: {e}")

    return translated


def _compute_overlap(
    gene_a: CalledGene, gene_b: CalledGene
) -> float:
    """Compute reciprocal overlap fraction between two genes on the same contig/strand.

    Returns the maximum of (overlap / len_a, overlap / len_b).
    """
    if gene_a.contig != gene_b.contig:
        return 0.0
    if gene_a.strand != gene_b.strand:
        return 0.0

    overlap_start = max(gene_a.start, gene_b.start)
    overlap_end = min(gene_a.end, gene_b.end)
    overlap = max(0, overlap_end - overlap_start + 1)

    if overlap == 0:
        return 0.0

    len_a = gene_a.end - gene_a.start + 1
    len_b = gene_b.end - gene_b.start + 1

    return max(overlap / len_a, overlap / len_b)


def _merge_gene_calls(
    pyrodigal_genes: list[CalledGene],
    miniprot_genes: list[CalledGene],
    overlap_threshold: float = 0.5,
) -> list[CalledGene]:
    """Merge Pyrodigal and miniprot gene calls.

    Strategy: Keep all Pyrodigal genes. Add miniprot genes only when they
    don't overlap >50% with any Pyrodigal gene on the same contig/strand.

    Args:
        pyrodigal_genes: Primary gene calls from Pyrodigal
        miniprot_genes: Supplementary calls from miniprot
        overlap_threshold: Maximum overlap fraction to add miniprot gene

    Returns:
        Merged list of CalledGene objects
    """
    merged = list(pyrodigal_genes)
    added = 0

    for mp_gene in miniprot_genes:
        if not mp_gene.protein_sequence:
            continue

        has_overlap = False
        for pg_gene in pyrodigal_genes:
            if _compute_overlap(mp_gene, pg_gene) >= overlap_threshold:
                has_overlap = True
                break

        if not has_overlap:
            merged.append(mp_gene)
            added += 1

    if added > 0:
        logger.info(
            f"Merged {added} miniprot genes (no Pyrodigal overlap) "
            f"with {len(pyrodigal_genes)} Pyrodigal genes"
        )

    return merged


def _genes_to_protein_records(genes: list[CalledGene]) -> dict[str, ProteinRecord]:
    """Convert CalledGene objects to ProteinRecord dict.

    Args:
        genes: List of called genes with protein sequences

    Returns:
        Dict mapping gene_id -> ProteinRecord
    """
    records: dict[str, ProteinRecord] = {}
    for gene in genes:
        if not gene.protein_sequence:
            continue
        records[gene.gene_id] = ProteinRecord(
            id=gene.gene_id,
            sequence=gene.protein_sequence,
            description=f"caller={gene.caller}",
            contig=gene.contig,
            start=gene.start,
            end=gene.end,
            strand=gene.strand,
        )
    return records


def _write_called_genes_gff3(genes: list[CalledGene], output_path: Path) -> None:
    """Write called genes as GFF3 for user inspection.

    Args:
        genes: List of called genes
        output_path: Output GFF3 file path
    """
    lines = ["##gff-version 3"]
    for gene in genes:
        strand_char = "+" if gene.strand == 1 else "-"
        attrs = f"ID={gene.gene_id};caller={gene.caller}"
        if gene.partial_5prime:
            attrs += ";partial_5prime=true"
        if gene.partial_3prime:
            attrs += ";partial_3prime=true"
        if gene.is_spliced:
            attrs += ";is_spliced=true"

        if gene.is_spliced and gene.exon_parts:
            # Write gene feature spanning full region
            lines.append(
                f"{gene.contig}\t{gene.caller}\tgene\t{gene.start}\t{gene.end}\t"
                f".\t{strand_char}\t.\t{attrs}"
            )
            # Write individual CDS exons
            for i, (start, end) in enumerate(gene.exon_parts, 1):
                lines.append(
                    f"{gene.contig}\t{gene.caller}\tCDS\t{start}\t{end}\t"
                    f".\t{strand_char}\t0\tID={gene.gene_id}_exon{i};Parent={gene.gene_id}"
                )
        else:
            lines.append(
                f"{gene.contig}\t{gene.caller}\tCDS\t{gene.start}\t{gene.end}\t"
                f".\t{strand_char}\t0\t{attrs}"
            )

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {len(genes)} called genes to {output_path}")


def _write_called_proteins_fasta(genes: list[CalledGene], output_path: Path) -> None:
    """Write called protein sequences as FASTA.

    Args:
        genes: List of called genes with protein sequences
        output_path: Output FASTA file path
    """
    lines: list[str] = []
    for gene in genes:
        if gene.protein_sequence:
            lines.append(f">{gene.gene_id} caller={gene.caller}")
            lines.append(gene.protein_sequence)

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {len(lines) // 2} protein sequences to {output_path}")


def call_genes(
    contigs: dict[str, str],
    contigs_fasta: Path,
    output_dir: Path,
    gene_caller: str = "auto",
    translation_table: int = 11,
    closed: bool = False,
    min_gene: int = 90,
    threads: int = 4,
) -> dict[str, ProteinRecord]:
    """Unified gene calling: Pyrodigal-gv + optional miniprot.

    Strategy:
    1. Run Pyrodigal-gv on all contigs -> primary gene set
    2. If miniprot available and ref DB installed:
       a. Run miniprot with curated reference proteins
       b. Merge: keep miniprot genes that don't overlap (>50%)
          with any Pyrodigal gene
    3. Convert CalledGene -> ProteinRecord with genomic coordinates
    4. Write called genes GFF3 and protein FASTA to output_dir

    Args:
        contigs: Dict of contig_id -> nucleotide sequence
        contigs_fasta: Path to nucleotide FASTA file (for miniprot)
        output_dir: Output directory for gene calling results
        gene_caller: "auto" (Pyrodigal + miniprot if available),
                     "pyrodigal" (Pyrodigal only)
        translation_table: Genetic code table number
        closed: Only call complete genes
        min_gene: Minimum gene length in nucleotides
        threads: Number of threads

    Returns:
        Dict mapping gene_id -> ProteinRecord
    """
    if not contigs:
        logger.warning("No contigs provided for gene calling")
        return {}

    # Step 1: Pyrodigal gene calling
    pyrodigal = PyrodigalCaller(
        translation_table=translation_table,
        closed=closed,
        min_gene=min_gene,
    )
    pyrodigal_genes = pyrodigal.call_genes(contigs)

    # Step 2: Optional miniprot
    all_genes = pyrodigal_genes
    if gene_caller == "auto":
        miniprot_genes = _try_miniprot(contigs, contigs_fasta, output_dir, threads)
        if miniprot_genes:
            # Translate miniprot genes (they come without protein sequence)
            miniprot_genes = _translate_miniprot_genes(
                miniprot_genes, contigs, translation_table=1
            )
            all_genes = _merge_gene_calls(pyrodigal_genes, miniprot_genes)

    # Step 3: Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    gff_path = output_dir / "called_genes.gff3"
    fasta_path = output_dir / "called_proteins.faa"
    _write_called_genes_gff3(all_genes, gff_path)
    _write_called_proteins_fasta(all_genes, fasta_path)

    # Step 4: Convert to ProteinRecord
    records = _genes_to_protein_records(all_genes)
    logger.info(f"Gene calling complete: {len(records)} proteins from {len(contigs)} contigs")

    return records


def _try_miniprot(
    contigs: dict[str, str],
    contigs_fasta: Path,
    output_dir: Path,
    threads: int,
) -> list[CalledGene]:
    """Try running miniprot if available. Returns empty list on failure.

    Args:
        contigs: Dict of contig_id -> nucleotide sequence
        contigs_fasta: Path to nucleotide FASTA
        output_dir: Output directory
        threads: Number of threads

    Returns:
        List of CalledGene from miniprot, or empty list
    """
    from vhold.utils.external import check_miniprot

    available, version = check_miniprot()
    if not available:
        logger.debug("miniprot not found, skipping splice-aware gene calling")
        return []

    # Check for reference DB
    from vhold.databases.miniprot_refs import check_miniprot_refs, get_miniprot_ref_path

    if not check_miniprot_refs():
        logger.debug("miniprot reference DB not found, skipping")
        return []

    ref_path = get_miniprot_ref_path()
    logger.info(f"Running miniprot v{version} for splice-aware gene calling")

    try:
        caller = MiniprotCaller(ref_db=ref_path, threads=threads)
        return caller.call_genes(contigs_fasta, output_dir)
    except Exception as e:
        logger.warning(f"miniprot failed, continuing without: {e}")
        return []
