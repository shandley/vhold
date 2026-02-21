"""Integration tests for gene calling with real Pyrodigal execution.

These tests require pyrodigal to be installed. They exercise the full
gene calling pipeline on real viral genome fragments (T4 phage) to
verify correct end-to-end behavior.

Run with: uv run pytest tests/test_genecall_integration.py -v
Skip if pyrodigal not installed: tests auto-skip via importlib check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if not importlib.util.find_spec("pyrodigal"):
    pytest.skip("pyrodigal not installed", allow_module_level=True)

from vhold.features.genecall import (  # noqa: E402
    CalledGene,
    PyrodigalCaller,
    _genes_to_protein_records,
    _merge_gene_calls,
    _write_called_genes_gff3,
    _write_called_proteins_fasta,
    call_genes,
)
from vhold.io.nucleotide import read_nucleotide_fasta  # noqa: E402

TEST_DATA = Path(__file__).parent / "test_data"
INTEGRATION_CONTIGS = TEST_DATA / "integration_contigs.fna"


# ============================================================
# PyrodigalCaller integration tests
# ============================================================


class TestPyrodigalCallerIntegration:
    """Tests that actually run Pyrodigal on nucleotide sequences."""

    def test_single_contig_gene_calling(self):
        """Pyrodigal should find genes in a real T4 phage fragment."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()

        # Call only on fragment 1
        single = {"T4_fragment_1": contigs["T4_fragment_1"]}
        genes = caller.call_genes(single)

        assert len(genes) >= 3, f"Expected >=3 genes, got {len(genes)}"
        for gene in genes:
            assert gene.contig == "T4_fragment_1"
            assert gene.caller == "pyrodigal"
            assert gene.protein_sequence
            assert len(gene.protein_sequence) >= 10
            assert gene.start < gene.end
            assert gene.strand in (1, -1)

    def test_multi_contig_gene_calling(self):
        """Pyrodigal should call genes across multiple contigs."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)

        assert len(genes) >= 6, f"Expected >=6 genes across 2 contigs, got {len(genes)}"

        # Check both contigs have genes
        contig_ids = {g.contig for g in genes}
        assert "T4_fragment_1" in contig_ids
        assert "T4_fragment_2" in contig_ids

    def test_gene_ids_unique(self):
        """All gene IDs should be unique."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)

        gene_ids = [g.gene_id for g in genes]
        assert len(gene_ids) == len(set(gene_ids)), "Duplicate gene IDs found"

    def test_gene_ids_follow_convention(self):
        """Gene IDs should follow {contig}_{number} convention."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)

        for gene in genes:
            parts = gene.gene_id.rsplit("_", 1)
            assert len(parts) == 2, f"Bad gene ID format: {gene.gene_id}"
            assert parts[0] == gene.contig
            assert parts[1].isdigit()

    def test_protein_sequences_valid(self):
        """Protein sequences should contain only valid amino acids."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)

        for gene in genes:
            invalid = set(gene.protein_sequence.upper()) - valid_aa
            assert not invalid, (
                f"Invalid AA in {gene.gene_id}: {invalid}"
            )

    def test_coordinates_within_contig(self):
        """Gene coordinates should fall within contig boundaries."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)

        for gene in genes:
            contig_len = len(contigs[gene.contig])
            assert gene.start >= 1, f"{gene.gene_id} start < 1"
            assert gene.end <= contig_len, (
                f"{gene.gene_id} end {gene.end} > contig length {contig_len}"
            )

    def test_min_gene_filter(self):
        """Increasing min_gene should reduce gene count."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)

        caller_default = PyrodigalCaller(min_gene=90)
        genes_default = caller_default.call_genes(contigs)

        caller_strict = PyrodigalCaller(min_gene=300)
        genes_strict = caller_strict.call_genes(contigs)

        assert len(genes_strict) <= len(genes_default), (
            "Stricter min_gene should not produce more genes"
        )

    def test_closed_mode(self):
        """Closed mode should only return complete genes (no partials)."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller(closed=True)
        genes = caller.call_genes(contigs)

        for gene in genes:
            assert not gene.partial_5prime, f"{gene.gene_id} has partial 5'"
            assert not gene.partial_3prime, f"{gene.gene_id} has partial 3'"

    def test_translation_table_11(self):
        """Bacterial/viral translation table 11 should work."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller(translation_table=11)
        genes = caller.call_genes(contigs)
        assert len(genes) >= 3

    def test_short_contig_skipped(self):
        """Contigs shorter than min_gene should be skipped."""
        caller = PyrodigalCaller(min_gene=90)
        genes = caller.call_genes({"tiny": "ATGCATGC"})
        assert len(genes) == 0


# ============================================================
# Full call_genes() orchestrator integration tests
# ============================================================


class TestCallGenesIntegration:
    """Tests for the full call_genes() orchestrator with real Pyrodigal."""

    def test_pyrodigal_only_mode(self, tmp_path):
        """gene_caller='pyrodigal' should run Pyrodigal without miniprot."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        records = call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )

        assert len(records) >= 6
        for _rec_id, rec in records.items():
            assert rec.sequence
            assert rec.contig is not None
            assert rec.start is not None
            assert rec.end is not None
            assert rec.strand in (1, -1)

    def test_output_files_created(self, tmp_path):
        """call_genes should create GFF3 and FASTA output files."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )

        gff_path = tmp_path / "called_genes.gff3"
        fasta_path = tmp_path / "called_proteins.faa"

        assert gff_path.exists(), "GFF3 not created"
        assert fasta_path.exists(), "FASTA not created"

        # GFF3 should have header and gene entries
        gff_content = gff_path.read_text()
        assert "##gff-version 3" in gff_content
        assert "pyrodigal" in gff_content
        assert "CDS" in gff_content

        # FASTA should have protein entries
        fasta_content = fasta_path.read_text()
        assert fasta_content.startswith(">")
        assert "caller=pyrodigal" in fasta_content

    def test_gff3_coordinates_match_records(self, tmp_path):
        """GFF3 coordinates should match the returned ProteinRecords."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        records = call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )

        gff_content = (tmp_path / "called_genes.gff3").read_text()
        for rec_id, _rec in records.items():
            # Each gene should appear in GFF3
            assert f"ID={rec_id}" in gff_content

    def test_fasta_sequences_match_records(self, tmp_path):
        """FASTA sequences should match the returned ProteinRecords."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        records = call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )

        fasta_content = (tmp_path / "called_proteins.faa").read_text()

        # Parse FASTA output
        fasta_seqs = {}
        current_id = None
        for line in fasta_content.strip().split("\n"):
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                fasta_seqs[current_id] = ""
            elif current_id:
                fasta_seqs[current_id] += line.strip()

        for rec_id, rec in records.items():
            assert rec_id in fasta_seqs, f"{rec_id} missing from FASTA output"
            assert fasta_seqs[rec_id] == rec.sequence, (
                f"Sequence mismatch for {rec_id}"
            )

    def test_empty_contigs(self, tmp_path):
        """Empty contigs dict should return empty results."""
        records = call_genes(
            contigs={},
            contigs_fasta=tmp_path / "empty.fna",
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )
        assert len(records) == 0

    def test_auto_mode_without_miniprot(self, tmp_path):
        """auto mode should work even if miniprot is not on PATH."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        # auto mode should gracefully skip miniprot if not available
        records = call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="auto",
        )

        # Should still get Pyrodigal results
        assert len(records) >= 6


# ============================================================
# Nucleotide IO integration tests
# ============================================================


class TestNucleotideIOIntegration:
    """Tests for nucleotide FASTA reading integrated with gene calling."""

    def test_read_and_call(self):
        """read_nucleotide_fasta output should feed directly into gene calling."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)

        assert len(contigs) == 2
        assert "T4_fragment_1" in contigs
        assert "T4_fragment_2" in contigs

        # All sequences should be pure nucleotide
        nuc_chars = set("ATGCN")
        for cid, seq in contigs.items():
            invalid = set(seq.upper()) - nuc_chars
            assert not invalid, f"Non-nucleotide chars in {cid}: {invalid}"

    def test_roundtrip_contig_ids(self, tmp_path):
        """Contig IDs should be preserved through read -> call -> records."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        contigs_fasta = tmp_path / "contigs.fna"
        contigs_fasta.write_text(INTEGRATION_CONTIGS.read_text())

        records = call_genes(
            contigs=contigs,
            contigs_fasta=contigs_fasta,
            output_dir=tmp_path,
            gene_caller="pyrodigal",
        )

        for rec_id, rec in records.items():
            assert rec.contig in contigs, (
                f"Record {rec_id} has unknown contig {rec.contig}"
            )


# ============================================================
# Merge with real Pyrodigal output
# ============================================================


class TestMergeWithRealData:
    """Tests for merge logic using real Pyrodigal output."""

    def test_merge_with_synthetic_miniprot(self):
        """Merge real Pyrodigal genes with synthetic miniprot genes."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        pyrodigal_genes = caller.call_genes(contigs)

        # Create synthetic miniprot gene in a region without Pyrodigal genes
        # Find a gap in Pyrodigal coverage on fragment_1
        f1_genes = [g for g in pyrodigal_genes if g.contig == "T4_fragment_1"]
        # Place miniprot gene well outside any Pyrodigal gene
        max_end = max(g.end for g in f1_genes)
        contig_len = len(contigs["T4_fragment_1"])

        if max_end + 100 < contig_len:
            mp_gene = CalledGene(
                contig="T4_fragment_1",
                start=max_end + 50,
                end=min(max_end + 200, contig_len),
                strand=1,
                protein_sequence="MFAKEGENEPROTEIN",
                gene_id="T4_fragment_1_mp_1",
                caller="miniprot",
            )

            merged = _merge_gene_calls(pyrodigal_genes, [mp_gene])
            assert len(merged) == len(pyrodigal_genes) + 1
            mp_in_merged = [g for g in merged if g.caller == "miniprot"]
            assert len(mp_in_merged) == 1

    def test_merge_overlapping_rejected(self):
        """Synthetic miniprot gene overlapping real Pyrodigal gene is rejected."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        pyrodigal_genes = caller.call_genes(contigs)

        # Place miniprot gene exactly on top of first Pyrodigal gene
        first_pg = pyrodigal_genes[0]
        mp_gene = CalledGene(
            contig=first_pg.contig,
            start=first_pg.start,
            end=first_pg.end,
            strand=first_pg.strand,
            protein_sequence="MOVERLAPPING",
            gene_id=f"{first_pg.contig}_mp_1",
            caller="miniprot",
        )

        merged = _merge_gene_calls(pyrodigal_genes, [mp_gene])
        assert len(merged) == len(pyrodigal_genes)
        assert all(g.caller == "pyrodigal" for g in merged)


# ============================================================
# ProteinRecord conversion with real data
# ============================================================


class TestProteinRecordConversion:
    """Tests for converting real Pyrodigal output to ProteinRecords."""

    def test_all_fields_populated(self):
        """Every ProteinRecord should have all genomic fields."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)
        records = _genes_to_protein_records(genes)

        for rec_id, rec in records.items():
            assert rec.id == rec_id
            assert rec.sequence, f"{rec_id} missing sequence"
            assert rec.contig, f"{rec_id} missing contig"
            assert rec.start is not None, f"{rec_id} missing start"
            assert rec.end is not None, f"{rec_id} missing end"
            assert rec.strand in (1, -1), f"{rec_id} invalid strand"
            assert "caller=" in rec.description


# ============================================================
# Output file format validation
# ============================================================


class TestOutputFormats:
    """Tests for GFF3 and FASTA output from real gene calling."""

    def test_gff3_parseable(self, tmp_path):
        """GFF3 output should be well-formed."""
        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)
        gff_path = tmp_path / "genes.gff3"
        _write_called_genes_gff3(genes, gff_path)

        lines = gff_path.read_text().strip().split("\n")
        assert lines[0] == "##gff-version 3"

        for line in lines[1:]:
            parts = line.split("\t")
            assert len(parts) == 9, f"Bad GFF3 line: {line}"
            # Verify numeric fields
            assert parts[3].isdigit(), f"Start not numeric: {parts[3]}"
            assert parts[4].isdigit(), f"End not numeric: {parts[4]}"
            assert parts[6] in ("+", "-"), f"Bad strand: {parts[6]}"

    def test_fasta_parseable(self, tmp_path):
        """FASTA output should be parseable by BioPython."""
        from Bio import SeqIO

        contigs = read_nucleotide_fasta(INTEGRATION_CONTIGS)
        caller = PyrodigalCaller()
        genes = caller.call_genes(contigs)
        fasta_path = tmp_path / "proteins.faa"
        _write_called_proteins_fasta(genes, fasta_path)

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        assert len(records) == len(genes)
        for rec in records:
            assert len(rec.seq) > 0
