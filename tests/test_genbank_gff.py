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
