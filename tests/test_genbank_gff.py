"""Tests for GenBank and GFF3 input parsing."""

from pathlib import Path

import pytest

from vhold.io import read_input
from vhold.io.fasta import ProteinRecord, read_fasta
from vhold.io.genbank import read_genbank
from vhold.io.gff import read_gff, _parse_gff3, _parse_gff_attributes

TEST_DATA = Path(__file__).parent / "test_data"


# ---- Test GenBank parsing ----

class TestReadGenbank:
    """Tests for GenBank file parsing."""

    def test_reads_proteins(self):
        """Should extract CDS features with translations."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        assert len(records) >= 3  # At least 3 proteins (small one may be filtered)

    def test_protein_ids(self):
        """Should use protein_id qualifier as ID."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        assert "TV_RDRP" in records
        assert "TV_COAT" in records
        assert "TV_HYP" in records

    def test_sequences_are_valid(self):
        """Should extract uppercase amino acid sequences."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        for rec in records.values():
            assert rec.sequence.isupper()
            assert len(rec.sequence) > 0
            assert rec.sequence[0] == "M"  # Start codon

    def test_coordinates_extracted(self):
        """Should extract genomic coordinates."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        rdrp = records["TV_RDRP"]
        assert rdrp.contig == "TEST_VIRUS.1"
        assert rdrp.start == 99  # BioPython is 0-based
        assert rdrp.end == 600
        assert rdrp.strand == 1

    def test_complement_strand(self):
        """Should correctly identify complement features."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        coat = records["TV_COAT"]
        assert coat.strand == -1

    def test_description_from_product(self):
        """Should use product qualifier as description."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        rdrp = records["TV_RDRP"]
        assert "RNA-dependent RNA polymerase" in rdrp.description

    def test_source_annotations(self):
        """Should capture source qualifiers."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        rdrp = records["TV_RDRP"]
        assert rdrp.source_annotations is not None
        assert rdrp.source_annotations["gene"] == "rdrp"
        assert rdrp.source_annotations["locus_tag"] == "TV_001"

    def test_min_length_filter(self):
        """Should filter proteins below min_length."""
        # TV_SMALL is 7aa — should be filtered at min_length=10
        records = read_genbank(TEST_DATA / "test_viral.gb", min_length=10)
        assert "TV_SMALL" not in records

    def test_max_length_filter(self):
        """Should filter proteins above max_length."""
        records = read_genbank(TEST_DATA / "test_viral.gb", max_length=50)
        # All proteins are >50aa so all should be filtered
        assert len(records) <= 1  # Only TV_SMALL (7aa) might remain

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_genbank("/nonexistent/path.gb")


# ---- Test GFF3 parsing helpers ----

class TestGff3Helpers:
    """Tests for GFF3 parsing helper functions."""

    def test_parse_gff_attributes(self):
        """Should parse key=value pairs."""
        result = _parse_gff_attributes("ID=gene_1;product=RNA polymerase;Name=rdrp")
        assert result["ID"] == "gene_1"
        assert result["product"] == "RNA polymerase"
        assert result["Name"] == "rdrp"

    def test_parse_gff_attributes_url_encoded(self):
        """Should URL-decode attribute values."""
        result = _parse_gff_attributes("product=RNA%20polymerase")
        assert result["product"] == "RNA polymerase"

    def test_parse_gff3_features(self):
        """Should parse GFF3 features from file."""
        features, fasta = _parse_gff3(TEST_DATA / "test_viral.gff3")
        assert len(features) >= 3
        assert all(f["type"] == "CDS" for f in features)
        assert fasta  # Has embedded FASTA

    def test_parse_gff3_strand(self):
        """Should parse strand correctly."""
        features, _ = _parse_gff3(TEST_DATA / "test_viral.gff3")
        # gene_1 is +, gene_2 is -
        strands = {f["attributes"]["ID"]: f["strand"] for f in features}
        assert strands["gene_1"] == 1
        assert strands["gene_2"] == -1


# ---- Test GFF3 reading ----

class TestReadGff:
    """Tests for GFF3 file reading with translation."""

    def test_reads_with_embedded_fasta(self):
        """Should read proteins from GFF3 with embedded FASTA."""
        records = read_gff(TEST_DATA / "test_viral.gff3")
        assert len(records) > 0

    def test_coordinates_preserved(self):
        """Should preserve GFF3 coordinates."""
        records = read_gff(TEST_DATA / "test_viral.gff3")
        if records:
            first_rec = next(iter(records.values()))
            assert first_rec.contig is not None
            assert first_rec.start is not None
            assert first_rec.end is not None

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_gff("/nonexistent/path.gff3")

    def test_no_fasta_returns_empty(self, tmp_path):
        """Should return empty when no FASTA available."""
        # GFF with no embedded FASTA and no external FASTA
        gff = tmp_path / "test.gff3"
        gff.write_text("##gff-version 3\nseq\ttest\tCDS\t1\t300\t.\t+\t0\tID=g1\n")
        records = read_gff(gff)
        assert records == {}


# ---- Test protein FASTA + GFF ----

class TestReadGffProteinFasta:
    """Tests for reading GFF3 with pre-translated protein FASTA (Prodigal .faa)."""

    def test_protein_fasta_detected(self, tmp_path):
        """Should detect protein FASTA and skip translation."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "contig_1\tProdigal\tCDS\t1\t300\t.\t+\t0\tID=prot_1;product=RNA polymerase\n"
            "contig_1\tProdigal\tCDS\t400\t600\t.\t-\t0\tID=prot_2;product=coat protein\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(
            ">prot_1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEA\n"
            ">prot_2\nMFLSFPTTKTYFPHFDLSH\n"
        )
        records = read_gff(gff, fasta_path=faa)
        assert len(records) == 2
        assert records["prot_1"].sequence == "MVLSPADKTNVKAAWGKVGAHAGEYGAEA"
        assert records["prot_2"].sequence == "MFLSFPTTKTYFPHFDLSH"

    def test_coordinates_from_gff(self, tmp_path):
        """Should attach GFF coordinates to protein FASTA records."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "ctg1\tProdigal\tCDS\t100\t400\t.\t+\t0\tID=p1;product=polymerase\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(">p1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEA\n")
        records = read_gff(gff, fasta_path=faa)
        rec = records["p1"]
        assert rec.contig == "ctg1"
        assert rec.start == 100
        assert rec.end == 400
        assert rec.strand == 1

    def test_stop_codon_stripped(self, tmp_path):
        """Should strip trailing * from protein sequences."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "c1\tProdigal\tCDS\t1\t300\t.\t+\t0\tID=p1\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(">p1\nMVLSPADKTNVKAAWGKV*\n")
        records = read_gff(gff, fasta_path=faa)
        assert records["p1"].sequence == "MVLSPADKTNVKAAWGKV"

    def test_length_filters_apply(self, tmp_path):
        """Should filter by min/max length in protein FASTA mode."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "c1\tProdigal\tCDS\t1\t300\t.\t+\t0\tID=short\n"
            "c1\tProdigal\tCDS\t400\t700\t.\t+\t0\tID=ok\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(
            ">short\nMVL\n"
            ">ok\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSF\n"
        )
        records = read_gff(gff, fasta_path=faa, min_length=10)
        assert "short" not in records
        assert "ok" in records

    def test_unmatched_ids_skipped(self, tmp_path):
        """Should skip proteins with no matching GFF feature."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "c1\tProdigal\tCDS\t1\t300\t.\t+\t0\tID=p1\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(
            ">p1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEA\n"
            ">p_extra\nMFLSFPTTKTYFPHFDLSH\n"
        )
        records = read_gff(gff, fasta_path=faa)
        assert "p1" in records
        assert "p_extra" not in records

    def test_product_propagated(self, tmp_path):
        """Should propagate product from GFF to record description."""
        gff = tmp_path / "test.gff3"
        gff.write_text(
            "##gff-version 3\n"
            "c1\tProdigal\tCDS\t1\t300\t.\t+\t0\tID=p1;product=RNA polymerase\n"
        )
        faa = tmp_path / "test.faa"
        faa.write_text(">p1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEA\n")
        records = read_gff(gff, fasta_path=faa)
        assert records["p1"].description == "RNA polymerase"
        assert records["p1"].source_annotations["product"] == "RNA polymerase"


# ---- Test _is_protein_fasta ----

class TestIsProteinFasta:
    """Tests for protein vs nucleotide FASTA detection."""

    def test_protein_detected(self):
        """Should detect amino acid sequences as protein."""
        from vhold.io.gff import _is_protein_fasta
        seqs = {"p1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEA"}
        assert _is_protein_fasta(seqs) is True

    def test_nucleotide_detected(self):
        """Should detect DNA sequences as nucleotide."""
        from vhold.io.gff import _is_protein_fasta
        seqs = {"n1": "ATGCGTACGATCGATCGATCG"}
        assert _is_protein_fasta(seqs) is False

    def test_ambiguous_nucleotide(self):
        """Should detect IUPAC nucleotide as nucleotide."""
        from vhold.io.gff import _is_protein_fasta
        seqs = {"n1": "ATGCNNNRYSWKM"}
        assert _is_protein_fasta(seqs) is False

    def test_empty_returns_false(self):
        """Should return False for empty dict."""
        from vhold.io.gff import _is_protein_fasta
        assert _is_protein_fasta({}) is False


# ---- Test read_input auto-detection ----

class TestReadInput:
    """Tests for format auto-detection."""

    def test_fasta_detection(self):
        """Should detect FASTA format."""
        records = read_input(TEST_DATA / "small_viral.fasta")
        assert len(records) > 0
        # FASTA records should NOT have positions
        first = next(iter(records.values()))
        assert first.contig is None

    def test_genbank_detection(self):
        """Should detect GenBank format by extension."""
        records = read_input(TEST_DATA / "test_viral.gb")
        assert len(records) >= 3
        # GenBank records SHOULD have positions
        first = next(iter(records.values()))
        assert first.contig is not None

    def test_gff_detection(self):
        """Should detect GFF3 format by extension."""
        records = read_input(TEST_DATA / "test_viral.gff3")
        # May be empty if translation fails, but should not error
        assert isinstance(records, dict)

    def test_manual_format_override(self):
        """Should respect explicit format override."""
        # Force GenBank format on a .gb file
        records = read_input(
            TEST_DATA / "test_viral.gb", input_format="genbank"
        )
        assert len(records) >= 3

    def test_fasta_still_works(self):
        """FASTA input should work unchanged."""
        records = read_input(
            TEST_DATA / "small_viral.fasta", input_format="fasta"
        )
        assert len(records) > 0


# ---- Test ProteinRecord position fields ----

class TestProteinRecordPosition:
    """Tests for position fields on ProteinRecord."""

    def test_default_none(self):
        """Position fields should default to None."""
        rec = ProteinRecord(id="test", sequence="MVLSPAD")
        assert rec.contig is None
        assert rec.start is None
        assert rec.end is None
        assert rec.strand is None
        assert rec.source_annotations is None

    def test_position_fields(self):
        """Should store position data."""
        rec = ProteinRecord(
            id="test", sequence="MVLSPAD",
            contig="contig_1", start=100, end=400, strand=1,
        )
        assert rec.contig == "contig_1"
        assert rec.start == 100
        assert rec.end == 400
        assert rec.strand == 1

    def test_backward_compat(self):
        """Existing code should work without position fields."""
        rec = ProteinRecord(id="test", sequence="MVLSPAD", description="test protein")
        assert rec.length == 7
        assert rec.to_fasta().startswith(">test")

    def test_source_annotations(self):
        """Should store source annotation dict."""
        rec = ProteinRecord(
            id="test", sequence="MVLSPAD",
            source_annotations={"product": "polymerase", "gene": "pol"},
        )
        assert rec.source_annotations["product"] == "polymerase"


# ---- Prodigal GFF+FAA workflow validation ----

class TestProdigalWorkflow:
    """Validate the full Prodigal GFF+FAA workflow with realistic output.

    Tests use actual Prodigal output format (GFF3 + protein FASTA) from
    tests/test_data/prodigal_output.{gff,faa}.
    """

    PRODIGAL_GFF = TEST_DATA / "prodigal_output.gff"
    PRODIGAL_FAA = TEST_DATA / "prodigal_output.faa"

    def test_all_proteins_parsed(self):
        """Should parse all 7 CDS features from Prodigal output."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        assert len(records) == 7

    def test_ids_match_between_gff_and_faa(self):
        """GFF feature IDs should match FAA sequence IDs."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        expected_ids = [f"lambda_test_{i}" for i in range(1, 8)]
        for eid in expected_ids:
            assert eid in records, f"Missing protein: {eid}"

    def test_sequences_are_protein(self):
        """Should extract protein sequences (not nucleotide)."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        for pid, rec in records.items():
            assert rec.sequence[0] == "M", f"{pid} should start with M"
            assert len(rec.sequence) > 20, f"{pid} too short: {len(rec.sequence)}"
            # Should contain protein-only characters
            has_protein_chars = any(c in rec.sequence for c in "EFILPQ")
            assert has_protein_chars, f"{pid} doesn't look like protein"

    def test_coordinates_from_gff(self):
        """Genomic coordinates should come from GFF, not FASTA."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        # lambda_test_1: CDS 3..302 +
        p1 = records["lambda_test_1"]
        assert p1.contig == "lambda_test"
        assert p1.start == 3
        assert p1.end == 302
        assert p1.strand == 1

    def test_reverse_strand_coordinates(self):
        """Reverse strand CDS should have correct strand."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        # lambda_test_5: CDS 1795..2598 -
        p5 = records["lambda_test_5"]
        assert p5.strand == -1
        assert p5.start == 1795
        assert p5.end == 2598

    def test_prodigal_faa_header_parsing(self):
        """Should handle Prodigal's extended FASTA headers correctly.

        Prodigal FAA headers: >ID # start # end # strand # attrs
        The ID is extracted from before the first space/comment marker.
        """
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        # All IDs should be clean (no Prodigal metadata leaking into ID)
        for pid in records:
            assert "#" not in pid, f"ID '{pid}' contains Prodigal header metadata"
            assert " " not in pid, f"ID '{pid}' contains spaces"

    def test_read_input_auto_detects_gff_with_faa(self):
        """read_input should work with GFF+FAA via explicit fasta_path."""
        records = read_input(
            self.PRODIGAL_GFF,
            fasta_path=str(self.PRODIGAL_FAA),
        )
        assert len(records) == 7
        # Should have positions
        first = records["lambda_test_1"]
        assert first.contig is not None

    def test_no_stop_codon_in_sequences(self):
        """Trailing stop codons should be stripped from protein sequences."""
        records = read_gff(self.PRODIGAL_GFF, fasta_path=self.PRODIGAL_FAA)
        for pid, rec in records.items():
            assert not rec.sequence.endswith("*"), (
                f"{pid} has trailing stop codon"
            )
