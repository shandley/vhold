"""Comprehensive tests for IO hardening: gzip, validation, duplicate IDs, mat_peptide, etc.

Tests cover:
- open_maybe_gzip: transparent gzip handling
- validate_protein_sequence: cleaning, validation, warning generation
- check_duplicate_id: duplicate ID detection and renaming
- GenBank mat_peptide extraction and parent CDS exclusion
- GenBank CompoundLocation (spliced CDS) handling
- GenBank duplicate protein IDs across segments
- GFF3 multi-row CDS merging
- GFF3 translation table detection
- Gzip-compressed input files (FASTA, GenBank, GFF3)
- Format detection with .gz suffix
- ProteinRecord feature_type field
"""

import gzip
from pathlib import Path

import pytest

from vhold.io import read_input
from vhold.io.fasta import ProteinRecord, read_fasta
from vhold.io.genbank import read_genbank
from vhold.io.gff import (
    _detect_translation_table,
    _merge_multipart_cds,
    _parse_gff3,
    read_gff,
)
from vhold.io.utils import check_duplicate_id, open_maybe_gzip, validate_protein_sequence

TEST_DATA = Path(__file__).parent / "test_data"


# ---- open_maybe_gzip ----

class TestOpenMaybeGzip:
    """Tests for transparent gzip file handling."""

    def test_reads_plain_text(self, tmp_path):
        """Should read plain text files normally."""
        f = tmp_path / "test.txt"
        f.write_text("hello world\n")
        with open_maybe_gzip(f) as handle:
            content = handle.read()
        assert content == "hello world\n"

    def test_reads_gzip(self, tmp_path):
        """Should transparently decompress gzip files."""
        f = tmp_path / "test.txt.gz"
        with gzip.open(f, "wt") as gz:
            gz.write("compressed content\n")
        with open_maybe_gzip(f) as handle:
            content = handle.read()
        assert content == "compressed content\n"

    def test_detects_by_magic_bytes_not_extension(self, tmp_path):
        """Should detect gzip by magic bytes, not file extension."""
        # Gzip file with .txt extension
        f = tmp_path / "sneaky.txt"
        with gzip.open(f, "wt") as gz:
            gz.write("I am actually gzipped\n")
        with open_maybe_gzip(f) as handle:
            content = handle.read()
        assert content == "I am actually gzipped\n"

    def test_binary_mode(self, tmp_path):
        """Should support binary mode."""
        f = tmp_path / "test.bin"
        f.write_bytes(b"binary data")
        with open_maybe_gzip(f, mode="rb") as handle:
            content = handle.read()
        assert content == b"binary data"


# ---- validate_protein_sequence ----

class TestValidateProteinSequence:
    """Tests for protein sequence validation and cleaning."""

    def test_valid_sequence_unchanged(self):
        """Valid sequences should pass through unchanged."""
        seq, warnings = validate_protein_sequence("MVLSPADKTNVKAAWGKV", "test")
        assert seq == "MVLSPADKTNVKAAWGKV"
        assert warnings == []

    def test_trailing_stop_stripped(self):
        """Trailing * should be removed."""
        seq, warnings = validate_protein_sequence("MVLSPAD*", "test")
        assert seq == "MVLSPAD"
        assert warnings == []

    def test_multiple_trailing_stops_stripped(self):
        """Multiple trailing * should all be removed."""
        seq, warnings = validate_protein_sequence("MVLSPAD***", "test")
        assert seq == "MVLSPAD"
        assert warnings == []

    def test_internal_stop_warned_and_replaced(self):
        """Internal * should trigger warning and be replaced with X."""
        seq, warnings = validate_protein_sequence("MVL*PAD", "test")
        assert seq == "MVLXPAD"
        assert len(warnings) == 1
        assert "internal stop codon" in warnings[0]

    def test_invalid_chars_replaced_with_x(self):
        """Non-amino-acid characters should be replaced with X."""
        seq, warnings = validate_protein_sequence("MVL1PAD", "test")
        assert seq == "MVLXPAD"
        assert len(warnings) == 1
        assert "invalid characters" in warnings[0]

    def test_empty_after_cleaning(self):
        """Sequence that becomes empty should return empty with warning."""
        seq, warnings = validate_protein_sequence("***", "test")
        assert seq == ""
        assert len(warnings) == 1
        assert "empty" in warnings[0]

    def test_strip_stop_disabled(self):
        """strip_stop=False should leave trailing * alone."""
        seq, warnings = validate_protein_sequence("MVLSPAD*", "test", strip_stop=False)
        # Trailing * stays but gets converted to X as internal
        assert "X" in seq or "*" in seq

    def test_lowercase_uppercased(self):
        """Lowercase input should be uppercased."""
        seq, warnings = validate_protein_sequence("mvlspad", "test")
        assert seq == "MVLSPAD"
        assert warnings == []

    @pytest.mark.parametrize("char", ["E", "F", "I", "L", "P", "Q"])
    def test_protein_only_chars_valid(self, char):
        """Characters that distinguish protein from nucleotide should be valid."""
        seq, warnings = validate_protein_sequence(f"M{char}VLSPAD", "test")
        assert seq == f"M{char}VLSPAD"
        assert warnings == []

    @pytest.mark.parametrize("char", ["B", "Z", "U", "O", "J", "X"])
    def test_ambiguity_codes_valid(self, char):
        """Ambiguity codes and special AAs should be accepted."""
        seq, warnings = validate_protein_sequence(f"M{char}VLSPAD", "test")
        assert seq == f"M{char}VLSPAD"
        assert warnings == []


# ---- check_duplicate_id ----

class TestCheckDuplicateId:
    """Tests for duplicate protein ID detection."""

    def test_unique_id_unchanged(self):
        """Unique IDs should pass through unchanged."""
        existing = {"prot_1": "seq1"}
        result = check_duplicate_id("prot_2", existing)
        assert result == "prot_2"

    def test_duplicate_gets_dup1(self):
        """First duplicate should get _dup1 suffix."""
        existing = {"prot_1": "seq1"}
        result = check_duplicate_id("prot_1", existing)
        assert result == "prot_1_dup1"

    def test_second_duplicate_gets_dup2(self):
        """Second duplicate should get _dup2 suffix."""
        existing = {"prot_1": "seq1", "prot_1_dup1": "seq2"}
        result = check_duplicate_id("prot_1", existing)
        assert result == "prot_1_dup2"

    def test_empty_dict(self):
        """Empty existing records should return ID unchanged."""
        result = check_duplicate_id("prot_1", {})
        assert result == "prot_1"


# ---- GenBank mat_peptide support ----

class TestMatPeptide:
    """Tests for GenBank mat_peptide feature extraction."""

    def test_mat_peptide_extracted(self):
        """Should extract mat_peptide features with translations."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        # Should have capsid, prM, E, and NS5
        ids = list(records.keys())
        assert "DTV_C" in ids
        assert "DTV_PRM" in ids
        assert "DTV_E" in ids
        assert "DTV_NS5" in ids

    def test_polyprotein_excluded(self):
        """Parent polyprotein CDS should be excluded when >=2 mat_peptides."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        # DTV_POLY has 3 mat_peptides -> should be excluded
        assert "DTV_POLY" not in records

    def test_non_polyprotein_cds_kept(self):
        """CDS without mat_peptide children should be kept."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        # NS5 is a standalone CDS, not a polyprotein
        assert "DTV_NS5" in records

    def test_mat_peptide_feature_type(self):
        """mat_peptide records should have feature_type='mat_peptide'."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        capsid = records["DTV_C"]
        assert capsid.feature_type == "mat_peptide"

    def test_cds_feature_type(self):
        """Regular CDS should have feature_type='CDS'."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        ns5 = records["DTV_NS5"]
        assert ns5.feature_type == "CDS"

    def test_mat_peptide_coordinates(self):
        """mat_peptide coordinates should come from feature location."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        capsid = records["DTV_C"]
        assert capsid.contig == "DENGUE_TEST.1"
        assert capsid.start is not None
        assert capsid.end is not None

    def test_mat_peptide_annotations(self):
        """mat_peptide should have product and parent_cds in annotations."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        capsid = records["DTV_C"]
        assert capsid.source_annotations is not None
        assert capsid.source_annotations.get("product") == "capsid protein C"

    def test_disable_mat_peptide(self):
        """include_mat_peptide=False should skip mat_peptide features."""
        records = read_genbank(
            TEST_DATA / "dengue_mat_peptide.gb",
            include_mat_peptide=False,
        )
        # Should only have CDS features
        assert "DTV_POLY" in records
        assert "DTV_NS5" in records
        # mat_peptide IDs should be absent
        assert "DTV_C" not in records
        assert "DTV_PRM" not in records
        assert "DTV_E" not in records

    def test_mat_peptide_count(self):
        """Should have correct number of proteins."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        # 3 mat_peptides + 1 standalone CDS (NS5) = 4
        # Polyprotein excluded (3 mat_peptide children)
        assert len(records) == 4


# ---- GenBank CompoundLocation (spliced CDS) ----

class TestCompoundLocation:
    """Tests for spliced CDS handling (join() locations)."""

    def test_spliced_cds_extracted(self):
        """Should extract protein from join() CDS."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        assert "HTV_TAT" in records

    def test_spliced_cds_coordinates(self):
        """Bounding box coordinates for spliced CDS."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        tat = records["HTV_TAT"]
        # join(100..300,500..700) -> bounding box 100..700 (1-based inclusive)
        assert tat.start == 100
        assert tat.end == 700
        assert tat.strand == 1

    def test_spliced_cds_exon_parts(self):
        """Exon parts should be in source_annotations."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        tat = records["HTV_TAT"]
        assert tat.source_annotations is not None
        assert tat.source_annotations.get("is_spliced") is True
        exon_parts = tat.source_annotations.get("exon_parts")
        assert exon_parts is not None
        assert len(exon_parts) == 2

    def test_simple_cds_not_spliced(self):
        """Simple CDS should NOT have is_spliced annotation."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        gag = records["HTV_GAG"]
        if gag.source_annotations:
            assert gag.source_annotations.get("is_spliced") is not True

    def test_all_proteins_extracted(self):
        """Should extract all proteins including spliced ones."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        assert "HTV_TAT" in records
        assert "HTV_GAG" in records
        assert len(records) == 2


# ---- GenBank duplicate protein IDs ----

class TestDuplicateProteinIds:
    """Tests for duplicate protein_id handling in multi-segment GenBank files."""

    def test_duplicate_ids_renamed(self):
        """Duplicate protein_ids should get _dup suffixes."""
        records = read_genbank(TEST_DATA / "influenza_segments.gb")
        # Both segments have protein_id="SHARED_PROTEIN_ID"
        assert "SHARED_PROTEIN_ID" in records
        assert "SHARED_PROTEIN_ID_dup1" in records

    def test_all_proteins_kept(self):
        """All proteins should be kept despite duplicate IDs."""
        records = read_genbank(TEST_DATA / "influenza_segments.gb")
        # 2 proteins total (one from each segment)
        assert len(records) == 2


# ---- GFF3 multi-row CDS merging ----

class TestGffMultiRowCds:
    """Tests for merging multi-row CDS features in GFF3."""

    def test_merge_multipart_features(self):
        """Multi-row CDS with same ID should be merged."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"ID": "cds-1"}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {"ID": "cds-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 1
        assert cds_features[0]["intervals"] == [(100, 300), (500, 700)]
        assert cds_features[0]["start"] == 100
        assert cds_features[0]["end"] == 700

    def test_single_row_cds_unaffected(self):
        """Single-row CDS should get intervals added but stay independent."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"ID": "cds-1"}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {"ID": "cds-2"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2
        for f in cds_features:
            assert len(f["intervals"]) == 1

    def test_no_id_stays_independent(self):
        """CDS without ID attribute should not be merged."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2

    def test_different_contigs_not_merged(self):
        """CDS with same ID on different contigs should not be merged."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"ID": "cds-1"}},
            {"seqid": "ctg2", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"ID": "cds-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2

    def test_ncbi_spliced_gff_file(self):
        """Integration test: read NCBI-style spliced GFF3."""
        features, fasta_text, comments = _parse_gff3(
            TEST_DATA / "ncbi_spliced.gff3"
        )
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        # cds-1 has 2 rows (spliced), cds-2 and cds-3 are single
        spliced = [f for f in cds_features if len(f.get("intervals", [])) > 1]
        assert len(spliced) == 1
        assert spliced[0]["attributes"]["ID"] == "cds-1"
        assert len(spliced[0]["intervals"]) == 2


# ---- GFF3 translation table detection ----

class TestTranslationTable:
    """Tests for translation table detection from GFF3."""

    def test_default_table_1(self):
        """Should default to table 1 (standard)."""
        features = [{"attributes": {}}]
        table = _detect_translation_table(features, [])
        assert table == 1

    def test_from_feature_attribute(self):
        """Should detect transl_table from feature attributes."""
        features = [{"attributes": {"transl_table": "4"}}]
        table = _detect_translation_table(features, [])
        assert table == 4

    def test_from_prodigal_header(self):
        """Should detect transl_table from Prodigal header comment."""
        features = [{"attributes": {}}]
        comments = [
            "# Sequence Data: seqnum=1;seqlen=3000",
            "# Model Data: version=Prodigal.v2.6.3;transl_table=11;uses_sd=1",
        ]
        table = _detect_translation_table(features, comments)
        assert table == 11

    def test_attribute_priority_over_header(self):
        """Feature attribute should take priority over header comment."""
        features = [{"attributes": {"transl_table": "4"}}]
        comments = ["# Model Data: transl_table=11"]
        table = _detect_translation_table(features, comments)
        assert table == 4

    @pytest.mark.parametrize("table_num", [1, 4, 11, 15])
    def test_various_tables(self, table_num):
        """Should detect various translation table numbers."""
        features = [{"attributes": {"transl_table": str(table_num)}}]
        table = _detect_translation_table(features, [])
        assert table == table_num

    def test_phage_gff_translation_table(self):
        """Integration test: detect table 11 from Prodigal phage GFF."""
        features, _fasta, comments = _parse_gff3(
            TEST_DATA / "test_phage_table11.gff3"
        )
        table = _detect_translation_table(features, comments)
        assert table == 11


# ---- Gzip-compressed input files ----

class TestGzipFormats:
    """Tests for gzip-compressed input file support."""

    def test_fasta_gzip(self, tmp_path):
        """Should read gzipped FASTA files."""
        gz_path = tmp_path / "test.fasta.gz"
        content = ">prot1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEA\n>prot2\nMFLSFPTTKTYFPHFDLSH\n"
        with gzip.open(gz_path, "wt") as f:
            f.write(content)
        records = read_fasta(gz_path)
        assert len(records) == 2
        assert "prot1" in records
        assert "prot2" in records

    def test_genbank_gzip(self, tmp_path):
        """Should read gzipped GenBank files."""
        gb_path = TEST_DATA / "test_viral.gb"
        gz_path = tmp_path / "test_viral.gb.gz"
        with open(gb_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        records = read_genbank(gz_path)
        assert len(records) >= 3

    def test_gff_gzip(self, tmp_path):
        """Should read gzipped GFF3 files with embedded FASTA."""
        gff_path = TEST_DATA / "test_viral.gff3"
        gz_path = tmp_path / "test_viral.gff3.gz"
        with open(gff_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        records = read_gff(gz_path)
        assert isinstance(records, dict)

    def test_format_detection_with_gz_suffix(self, tmp_path):
        """read_input should detect format through .gz extension."""
        from vhold.io import _detect_format

        assert _detect_format(Path("genome.gb.gz")) == "genbank"
        assert _detect_format(Path("genes.gff3.gz")) == "gff"
        assert _detect_format(Path("proteins.fasta.gz")) == "fasta"
        assert _detect_format(Path("proteins.faa.gz")) == "fasta"

    def test_read_input_genbank_gz(self, tmp_path):
        """read_input should handle .gb.gz files."""
        gb_path = TEST_DATA / "test_viral.gb"
        gz_path = tmp_path / "test_viral.gb.gz"
        with open(gb_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        records = read_input(gz_path)
        assert len(records) >= 3


# ---- ProteinRecord feature_type ----

class TestFeatureType:
    """Tests for the feature_type field on ProteinRecord."""

    def test_default_cds(self):
        """Default feature_type should be 'CDS'."""
        rec = ProteinRecord(id="test", sequence="MVLSPAD")
        assert rec.feature_type == "CDS"

    def test_mat_peptide(self):
        """Should accept feature_type='mat_peptide'."""
        rec = ProteinRecord(
            id="test", sequence="MVLSPAD", feature_type="mat_peptide"
        )
        assert rec.feature_type == "mat_peptide"

    def test_backward_compat(self):
        """Old code creating ProteinRecord without feature_type should work."""
        rec = ProteinRecord(id="test", sequence="MVLSPAD", description="desc")
        assert rec.feature_type == "CDS"
        assert rec.length == 7
        assert rec.to_fasta().startswith(">test")


# ---- Integration: read_input format detection ----

class TestReadInputGzDetection:
    """Tests for read_input auto-detection with .gz files."""

    def test_fasta_unchanged(self):
        """Existing FASTA input should work unchanged."""
        records = read_input(TEST_DATA / "small_viral.fasta")
        assert len(records) > 0

    def test_genbank_unchanged(self):
        """Existing GenBank input should work unchanged."""
        records = read_input(TEST_DATA / "test_viral.gb")
        assert len(records) >= 3

    def test_gff_unchanged(self):
        """Existing GFF3 input should work unchanged."""
        records = read_input(TEST_DATA / "test_viral.gff3")
        assert isinstance(records, dict)

    def test_include_mat_peptide_passed(self):
        """include_mat_peptide should be passed to read_genbank."""
        # With mat_peptide enabled (default)
        records = read_input(TEST_DATA / "dengue_mat_peptide.gb")
        assert "DTV_C" in records
        assert "DTV_POLY" not in records

        # With mat_peptide disabled
        records = read_input(
            TEST_DATA / "dengue_mat_peptide.gb",
            include_mat_peptide=False,
        )
        assert "DTV_POLY" in records
        assert "DTV_C" not in records


# ---- Real-world-like patterns ----

class TestRealWorldPatterns:
    """Tests using synthetic but realistic viral genome patterns."""

    def test_flavivirus_mat_peptide_workflow(self):
        """Flavivirus: polyprotein excluded, mature peptides extracted."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        # Should have individual proteins, not polyprotein
        assert "DTV_POLY" not in records
        # Each mat_peptide should have a sequence
        for pid in ["DTV_C", "DTV_PRM", "DTV_E"]:
            assert pid in records
            assert len(records[pid].sequence) > 10

    def test_hiv_like_spliced_genes(self):
        """HIV-like: spliced gene products extracted with exon info."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        tat = records["HTV_TAT"]
        assert tat.source_annotations.get("is_spliced") is True
        # Translation should be from the /translation qualifier
        assert len(tat.sequence) > 10

    def test_segmented_virus_duplicate_ids(self):
        """Segmented virus: duplicate protein_ids resolved."""
        records = read_genbank(TEST_DATA / "influenza_segments.gb")
        assert len(records) == 2
        # Both proteins should be present
        ids = list(records.keys())
        assert any("SHARED_PROTEIN_ID" in i for i in ids)

    def test_prodigal_phage_table11_gff(self):
        """Prodigal output: translation table 11 detected from header."""
        features, fasta_text, comments = _parse_gff3(
            TEST_DATA / "test_phage_table11.gff3"
        )
        table = _detect_translation_table(features, comments)
        assert table == 11
        assert len(features) == 3  # 3 CDS features
        assert fasta_text  # Has embedded FASTA

    def test_ncbi_gff_spliced_cds(self):
        """NCBI GFF3: multi-row CDS merged into single feature."""
        features, fasta_text, comments = _parse_gff3(
            TEST_DATA / "ncbi_spliced.gff3"
        )
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        # Original has 4 CDS rows (gene rows are not CDS type)
        # After merge: cds-1 merged (2 rows -> 1), cds-2, cds-3 = 3 total
        assert len(cds_features) == 3

        # The spliced one should have 2 intervals
        spliced = [f for f in cds_features if len(f["intervals"]) > 1]
        assert len(spliced) == 1
        intervals = spliced[0]["intervals"]
        assert intervals == [(100, 300), (500, 700)]


# ---- FASTA validation integration ----

class TestFastaValidationIntegration:
    """Tests for validation wired into FASTA reader."""

    def test_trailing_stops_stripped(self, tmp_path):
        """read_fasta should strip trailing stop codons."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMVLSPADKTNVKAAWGKV*\n")
        records = read_fasta(fasta)
        assert records["prot1"].sequence == "MVLSPADKTNVKAAWGKV"

    def test_duplicate_ids_handled(self, tmp_path):
        """read_fasta should rename duplicate IDs."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMVLSPADKTNVKAAWGKV\n>prot1\nMFLSFPTTKTYFPHFDLSH\n")
        records = read_fasta(fasta)
        assert "prot1" in records
        assert "prot1_dup1" in records
        assert len(records) == 2

    def test_invalid_chars_replaced(self, tmp_path):
        """read_fasta should replace invalid characters with X."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMVL1PAD2KTN\n")
        records = read_fasta(fasta)
        assert "1" not in records["prot1"].sequence
        assert "2" not in records["prot1"].sequence

    def test_empty_sequence_skipped(self, tmp_path):
        """Sequences that become empty after cleaning should be skipped."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">empty\n***\n>valid\nMVLSPADKTNVKAAWGKV\n")
        records = read_fasta(fasta)
        assert "empty" not in records
        assert "valid" in records


# ---- _parse_gff3 returns header comments ----

class TestParseGff3HeaderComments:
    """Test that _parse_gff3 returns header comments as third value."""

    def test_returns_three_values(self):
        """_parse_gff3 should return (features, fasta_text, header_comments)."""
        result = _parse_gff3(TEST_DATA / "test_phage_table11.gff3")
        assert len(result) == 3
        features, fasta_text, comments = result
        assert isinstance(features, list)
        assert isinstance(fasta_text, str)
        assert isinstance(comments, list)

    def test_prodigal_comments_captured(self):
        """Should capture Prodigal header comments."""
        _, _, comments = _parse_gff3(TEST_DATA / "test_phage_table11.gff3")
        assert len(comments) >= 2
        # Should have Model Data comment with transl_table
        assert any("transl_table" in c for c in comments)


# ---- Coordinate normalization ----

class TestCoordinateNormalization:
    """Tests for consistent 1-based coordinate convention."""

    def test_genbank_coords_are_1based(self):
        """GenBank CDS 100..600 should produce start=100 (1-based), not 99."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        rdrp = records["TV_RDRP"]
        # GenBank: CDS 100..600 → 1-based start=100, end=600
        assert rdrp.start == 100
        assert rdrp.end == 600

    def test_genbank_complement_coords(self):
        """Complement strand coordinates should also be 1-based."""
        records = read_genbank(TEST_DATA / "test_viral.gb")
        coat = records["TV_COAT"]
        # GenBank: complement(700..1200) → 1-based start=700, end=1200
        assert coat.start == 700
        assert coat.end == 1200
        assert coat.strand == -1

    def test_genbank_compound_location_1based(self):
        """Compound location (join) parts should be 1-based."""
        records = read_genbank(TEST_DATA / "hiv_spliced.gb")
        tat = records["HTV_TAT"]
        # join(100..300,500..700) → parts (100,300) and (500,700), bounding 100..700
        assert tat.start == 100
        assert tat.end == 700
        exon_parts = tat.source_annotations["exon_parts"]
        assert exon_parts[0] == (100, 300)
        assert exon_parts[1] == (500, 700)

    def test_gff_coords_already_1based(self):
        """GFF3 coordinates should remain 1-based (unchanged)."""
        records = read_gff(
            TEST_DATA / "test_viral.gff3",
        )
        if records:
            first = next(iter(records.values()))
            # GFF3 coords are already 1-based: CDS 100 399
            assert first.start == 100
            assert first.end == 399


# ---- Format detection enhancements ----

class TestFormatDetectionEnhancements:
    """Tests for .gbff extension and unknown extension warnings."""

    def test_gbff_maps_to_genbank(self):
        """The .gbff extension should map to genbank format."""
        from vhold.io import _detect_format
        assert _detect_format(Path("genome.gbff")) == "genbank"
        assert _detect_format(Path("genome.gbff.gz")) == "genbank"

    def test_unknown_extension_falls_back_to_fasta(self):
        """Unknown extensions should fall back to FASTA."""
        from vhold.io import _detect_format
        assert _detect_format(Path("data.dat")) == "fasta"
        assert _detect_format(Path("data.xyz")) == "fasta"

    def test_unknown_extension_logs_warning(self, capsys):
        """Unknown extensions should log a warning to stderr."""
        from loguru import logger

        from vhold.io import _detect_format
        # Add a temporary loguru sink to capture warnings
        output = []
        sink_id = logger.add(lambda msg: output.append(str(msg)), level="WARNING")
        try:
            _detect_format(Path("data.dat"))
        finally:
            logger.remove(sink_id)
        assert any("Unknown file extension" in msg for msg in output)


# ---- feature_type propagation ----

class TestFeatureTypePropagation:
    """Tests for feature_type flowing through ConsensusResult to output."""

    def test_consensus_result_has_feature_type(self):
        """ConsensusResult should have feature_type field."""
        from vhold.results.consensus import ConsensusResult
        cr = ConsensusResult(query_id="test", query_length=100)
        assert cr.feature_type == "CDS"  # default

    def test_consensus_result_feature_type_in_dict(self):
        """feature_type should appear in to_dict() when contig is set."""
        from vhold.results.consensus import ConsensusResult
        cr = ConsensusResult(
            query_id="test", query_length=100,
            contig="ctg1", start=100, end=400, strand=1,
            feature_type="mat_peptide",
        )
        d = cr.to_dict()
        assert d["feature_type"] == "mat_peptide"

    def test_consensus_result_feature_type_not_in_dict_without_contig(self):
        """feature_type should NOT appear in to_dict() when contig is None."""
        from vhold.results.consensus import ConsensusResult
        cr = ConsensusResult(query_id="test", query_length=100)
        d = cr.to_dict()
        assert "feature_type" not in d

    def test_mat_peptide_feature_type_from_genbank(self):
        """mat_peptide records from GenBank should have correct feature_type."""
        records = read_genbank(TEST_DATA / "dengue_mat_peptide.gb")
        capsid = records["DTV_C"]
        assert capsid.feature_type == "mat_peptide"
        ns5 = records["DTV_NS5"]
        assert ns5.feature_type == "CDS"


# ---- Single mat_peptide child warning ----

class TestSingleMatPeptideWarning:
    """Tests for warning when CDS has exactly 1 mat_peptide child."""

    def test_single_mat_peptide_keeps_both(self, tmp_path):
        """CDS with 1 mat_peptide child should keep both."""
        # Create a GenBank file with 1 polyprotein + 1 mat_peptide
        gb_content = '''LOCUS       TEST_SINGLE             2000 bp    ss-RNA  linear   VRL 01-JAN-2024
DEFINITION  Test single mat_peptide.
ACCESSION   TEST_SINGLE
VERSION     TEST_SINGLE.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..2000
                     /organism="Test virus"
                     /mol_type="genomic RNA"
     CDS             100..600
                     /product="polyprotein"
                     /protein_id="POLY_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYV
                     QERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
     mat_peptide     100..300
                     /product="capsid protein"
                     /protein_id="CAP_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFK"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "single_mp.gb"
        gb_file.write_text(gb_content)
        records = read_genbank(gb_file)
        # Both polyprotein AND mat_peptide should be present (threshold is >=2)
        assert "POLY_1" in records
        assert "CAP_1" in records

    def test_single_mat_peptide_logs_warning(self, tmp_path):
        """CDS with 1 mat_peptide child should log a warning."""
        from loguru import logger
        gb_content = '''LOCUS       TEST_SINGLE             2000 bp    ss-RNA  linear   VRL 01-JAN-2024
DEFINITION  Test single mat_peptide.
ACCESSION   TEST_SINGLE
VERSION     TEST_SINGLE.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..2000
                     /organism="Test virus"
                     /mol_type="genomic RNA"
     CDS             100..600
                     /product="polyprotein"
                     /protein_id="POLY_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYV
                     QERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
     mat_peptide     100..300
                     /product="capsid protein"
                     /protein_id="CAP_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFK"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "single_mp.gb"
        gb_file.write_text(gb_content)
        # Add a temporary loguru sink to capture warnings
        output = []
        sink_id = logger.add(lambda msg: output.append(str(msg)), level="WARNING")
        try:
            read_genbank(gb_file)
        finally:
            logger.remove(sink_id)
        assert any("only 1 mat_peptide child" in msg for msg in output)


# ---- CLI mat_peptide flag ----

class TestCliMatPeptideFlag:
    """Tests for --mat-peptide/--no-mat-peptide CLI option."""

    def test_mat_peptide_flag_in_help(self):
        """--mat-peptide should appear in run help."""
        from click.testing import CliRunner

        from vhold.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "--mat-peptide" in result.output
        assert "--no-mat-peptide" in result.output


# ---- GFF3 Parent-based CDS merging ----

class TestGffParentFallbackMerging:
    """Tests for merging CDS rows by Parent when ID is absent.

    NCBI GFF3 files sometimes use a 3-tier hierarchy (gene → mRNA → CDS)
    where CDS rows lack an ID attribute but share a Parent attribute.
    These should be merged like ID-based rows.
    """

    def test_parent_fallback_merges_cds(self):
        """CDS rows with no ID but same Parent should be merged."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"Parent": "rna-tat-1"}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {"Parent": "rna-tat-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 1
        assert cds_features[0]["intervals"] == [(100, 300), (500, 700)]
        assert cds_features[0]["start"] == 100
        assert cds_features[0]["end"] == 700

    def test_different_parents_not_merged(self):
        """CDS rows with different Parent values should stay separate."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"Parent": "rna-tat-1"}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {"Parent": "rna-gag-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2

    def test_no_id_no_parent_stays_independent(self):
        """CDS with neither ID nor Parent should stay independent."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2

    def test_id_takes_priority_over_parent(self):
        """When both ID and Parent exist, ID should be the grouping key."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"ID": "cds-1", "Parent": "rna-1"}},
            {"seqid": "ctg1", "type": "CDS", "start": 500, "end": 700,
             "strand": 1, "attributes": {"ID": "cds-1", "Parent": "rna-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 1
        assert cds_features[0]["intervals"] == [(100, 300), (500, 700)]

    def test_parent_different_contigs_not_merged(self):
        """CDS with same Parent on different contigs should not merge."""
        features = [
            {"seqid": "ctg1", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"Parent": "rna-1"}},
            {"seqid": "ctg2", "type": "CDS", "start": 100, "end": 300,
             "strand": 1, "attributes": {"Parent": "rna-1"}},
        ]
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        assert len(cds_features) == 2

    def test_ncbi_3tier_gff3_integration(self, tmp_path):
        """Integration: NCBI 3-tier GFF3 with CDS→mRNA→gene hierarchy."""
        gff_content = """##gff-version 3
ctg1\tNCBI\tgene\t1\t1500\t.\t+\t.\tID=gene-tat;Name=tat
ctg1\tNCBI\tmRNA\t1\t1500\t.\t+\t.\tID=rna-tat;Parent=gene-tat
ctg1\tNCBI\tCDS\t100\t300\t.\t+\t0\tParent=rna-tat;product=transactivating protein
ctg1\tNCBI\tCDS\t500\t700\t.\t+\t0\tParent=rna-tat;product=transactivating protein
ctg1\tNCBI\tgene\t800\t1200\t.\t+\t.\tID=gene-gag;Name=gag
ctg1\tNCBI\tmRNA\t800\t1200\t.\t+\t.\tID=rna-gag;Parent=gene-gag
ctg1\tNCBI\tCDS\t800\t1200\t.\t+\t0\tParent=rna-gag;product=Gag polyprotein
"""
        gff_file = tmp_path / "ncbi_3tier.gff3"
        gff_file.write_text(gff_content)
        features, _, _ = _parse_gff3(gff_file)
        merged = _merge_multipart_cds(features)
        cds_features = [f for f in merged if f["type"] == "CDS"]
        # The 2 CDS rows for tat (no ID, Parent=rna-tat) → merged into 1
        # The 1 CDS row for gag (no ID, Parent=rna-gag) → stays as 1
        assert len(cds_features) == 2
        spliced = [f for f in cds_features if len(f["intervals"]) > 1]
        assert len(spliced) == 1
        assert spliced[0]["intervals"] == [(100, 300), (500, 700)]


# ---- Known limitation: circular genome wrap-around ----

class TestCircularGenomeWrapAround:
    """Document known limitation: circular genome CDS spanning the origin.

    When a CDS wraps around the origin of a circular genome
    (e.g., join(4500..5000,1..300)), the bounding-box start/end
    coordinates are wrong (start=1, end=5000 = entire genome).
    The protein sequence is still correctly extracted from the
    /translation qualifier. This limitation is documented here.
    """

    def test_wraparound_sequence_correct(self, tmp_path):
        """Protein sequence should be extracted correctly from /translation."""
        gb_content = '''LOCUS       CIRCULAR                5000 bp    ds-DNA  circular VRL 01-JAN-2024
DEFINITION  Circular test virus.
ACCESSION   CIRCULAR
VERSION     CIRCULAR.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..5000
                     /organism="Test virus"
                     /mol_type="genomic DNA"
     CDS             join(4500..5000,1..300)
                     /product="replication protein"
                     /protein_id="REP_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFK"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "circular.gb"
        gb_file.write_text(gb_content)
        records = read_genbank(gb_file)
        assert "REP_1" in records
        # Sequence from /translation is correct regardless of coord wrapping
        assert records["REP_1"].sequence.startswith("MSKGEELFTG")

    def test_wraparound_bounding_box_spans_genome(self, tmp_path):
        """Known limitation: bounding box covers entire genome for wrap-around CDS.

        For join(4500..5000,1..300), the bounding box is start=1, end=5000.
        This is a known limitation — the exon_parts correctly store the
        individual segments but the start/end bounding box is misleading.
        """
        gb_content = '''LOCUS       CIRCULAR                5000 bp    ds-DNA  circular VRL 01-JAN-2024
DEFINITION  Circular test virus.
ACCESSION   CIRCULAR
VERSION     CIRCULAR.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..5000
                     /organism="Test virus"
                     /mol_type="genomic DNA"
     CDS             join(4500..5000,1..300)
                     /product="replication protein"
                     /protein_id="REP_1"
                     /translation="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYG
                     KLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFK"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "circular.gb"
        gb_file.write_text(gb_content)
        records = read_genbank(gb_file)
        rep = records["REP_1"]
        # Known limitation: bounding box is the min/max of all parts
        # For wrap-around, this produces a misleadingly large range
        assert rep.start == 1  # min(4500, 1) = 1
        assert rep.end == 5000  # max(5000, 300) = 5000
        # But exon_parts correctly store the individual segments
        assert rep.source_annotations["is_spliced"] is True
        exon_parts = rep.source_annotations["exon_parts"]
        assert len(exon_parts) == 2


# ---- Known limitation: _is_protein_fasta false negative ----

class TestIsProteinFastaLimitation:
    """Document known limitation: protein sequences without EFILPQZ chars.

    _is_protein_fasta detects proteins by checking for amino acid-only
    characters (E, F, I, L, P, Q, Z). A protein composed entirely of
    characters shared with IUPAC nucleotide codes (A, C, D, G, H, K,
    M, N, R, S, T, V, W, Y, B, X) would be misclassified as nucleotide.
    This is statistically near-impossible for real viral proteins.
    """

    def test_normal_protein_detected(self):
        """Normal protein sequences should be detected correctly."""
        from vhold.io.gff import _is_protein_fasta
        seqs = {"p1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEA"}
        assert _is_protein_fasta(seqs) is True

    def test_ambiguous_short_peptide_not_detected(self):
        """Known limitation: short peptide with only shared chars misclassified.

        A very short peptide like MAAATTTCCC uses only characters valid
        in both amino acid and nucleotide alphabets. _is_protein_fasta
        returns False (nucleotide), which is incorrect but statistically
        improbable for real viral proteomes.
        """
        from vhold.io.gff import _is_protein_fasta
        # This peptide uses only chars shared between AA and nt alphabets
        seqs = {"p1": "MAAATTTCCC"}
        # Known false negative: returns False even though this is protein
        assert _is_protein_fasta(seqs) is False

    def test_one_detectable_sequence_saves_batch(self):
        """One protein with EFILPQZ chars makes the whole batch detectable."""
        from vhold.io.gff import _is_protein_fasta
        seqs = {
            "ambiguous": "MAAATTTCCC",
            "detectable": "MVLSPADKTNVKAAWGKVGAHAGEYGAEA",  # has L, P, E
        }
        assert _is_protein_fasta(seqs) is True


# ---- Empty GenBank file ----

class TestEmptyGenBankFile:
    """Tests for GenBank files with no extractable proteins."""

    def test_empty_genbank_returns_empty_dict(self, tmp_path):
        """GenBank file with no CDS features should return empty dict."""
        gb_content = '''LOCUS       EMPTY                   1000 bp    ss-RNA  linear   VRL 01-JAN-2024
DEFINITION  Empty test virus.
ACCESSION   EMPTY
VERSION     EMPTY.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..1000
                     /organism="Test virus"
                     /mol_type="genomic RNA"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "empty.gb"
        gb_file.write_text(gb_content)
        records = read_genbank(gb_file)
        assert records == {}

    def test_genbank_no_translations_returns_empty(self, tmp_path):
        """GenBank with CDS but no /translation should return empty dict."""
        gb_content = '''LOCUS       NO_TRANS                1000 bp    ss-RNA  linear   VRL 01-JAN-2024
DEFINITION  No translation test.
ACCESSION   NO_TRANS
VERSION     NO_TRANS.1
KEYWORDS    .
SOURCE      Test virus
  ORGANISM  Test virus
            Viruses.
FEATURES             Location/Qualifiers
     source          1..1000
                     /organism="Test virus"
                     /mol_type="genomic RNA"
     CDS             100..600
                     /product="hypothetical protein"
                     /gene="hyp"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
'''
        gb_file = tmp_path / "no_trans.gb"
        gb_file.write_text(gb_content)
        records = read_genbank(gb_file)
        assert records == {}
