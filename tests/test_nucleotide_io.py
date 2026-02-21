"""Tests for nucleotide FASTA detection and reading."""

import gzip

import pytest

from vhold.io.nucleotide import is_nucleotide_fasta, read_nucleotide_fasta


class TestIsNucleotideFasta:
    """Tests for is_nucleotide_fasta detection."""

    def test_nucleotide_fasta_detected(self, tmp_path):
        """Pure nucleotide FASTA should be detected."""
        fasta = tmp_path / "contigs.fna"
        fasta.write_text(
            ">contig_1\n"
            "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC\n"
            ">contig_2\n"
            "GGCCTTAATTCCGGAATTCCGGAATTCCGGAATTCCGGAA\n"
        )
        assert is_nucleotide_fasta(fasta) is True

    def test_protein_fasta_not_detected(self, tmp_path):
        """Protein FASTA should NOT be detected as nucleotide."""
        fasta = tmp_path / "proteins.faa"
        fasta.write_text(
            ">prot_1\n"
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH\n"
            ">prot_2\n"
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKD\n"
        )
        assert is_nucleotide_fasta(fasta) is False

    def test_nucleotide_with_n_ambiguity(self, tmp_path):
        """Nucleotide with N ambiguity codes still detected."""
        fasta = tmp_path / "contigs.fasta"
        fasta.write_text(
            ">contig_1\n"
            "ATGNNNNNNATGCATGCATGCNNNNATGCATGCATGCATGC\n"
        )
        assert is_nucleotide_fasta(fasta) is True

    def test_empty_file_not_detected(self, tmp_path):
        """Empty file should return False."""
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        assert is_nucleotide_fasta(fasta) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Non-existent file should return False."""
        assert is_nucleotide_fasta(tmp_path / "missing.fasta") is False

    def test_gzip_nucleotide_detected(self, tmp_path):
        """Gzipped nucleotide FASTA should be detected."""
        fasta = tmp_path / "contigs.fna.gz"
        content = b">contig_1\nATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC\n"
        with gzip.open(fasta, "wb") as f:
            f.write(content)
        assert is_nucleotide_fasta(fasta) is True

    def test_short_ambiguous_sequence(self, tmp_path):
        """Very short sequence with mixed content handled."""
        fasta = tmp_path / "short.fasta"
        # A/C/G overlap between protein and nucleotide
        fasta.write_text(">seq1\nACGACGACG\n")
        # This is 100% nucleotide chars, so should be detected
        assert is_nucleotide_fasta(fasta) is True

    def test_rna_sequence_detected(self, tmp_path):
        """RNA sequences (with U instead of T) should be detected."""
        fasta = tmp_path / "rna.fasta"
        fasta.write_text(
            ">rna_1\n"
            "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC\n"
        )
        assert is_nucleotide_fasta(fasta) is True


class TestReadNucleotideFasta:
    """Tests for read_nucleotide_fasta."""

    def test_basic_read(self, tmp_path):
        """Read simple nucleotide FASTA."""
        fasta = tmp_path / "contigs.fna"
        fasta.write_text(
            ">contig_1\n"
            "ATGCATGCATGC\n"
            ">contig_2\n"
            "GGCCTTAATTCC\n"
        )
        contigs = read_nucleotide_fasta(fasta)
        assert len(contigs) == 2
        assert contigs["contig_1"] == "ATGCATGCATGC"
        assert contigs["contig_2"] == "GGCCTTAATTCC"

    def test_uppercase_conversion(self, tmp_path):
        """Sequences should be uppercased."""
        fasta = tmp_path / "contigs.fna"
        fasta.write_text(">contig_1\natgcatgc\n")
        contigs = read_nucleotide_fasta(fasta)
        assert contigs["contig_1"] == "ATGCATGC"

    def test_multiline_sequence(self, tmp_path):
        """Multi-line sequences should be concatenated."""
        fasta = tmp_path / "contigs.fna"
        fasta.write_text(
            ">contig_1\n"
            "ATGCATGC\n"
            "GGCCTTAA\n"
        )
        contigs = read_nucleotide_fasta(fasta)
        assert contigs["contig_1"] == "ATGCATGCGGCCTTAA"

    def test_duplicate_contig_skipped(self, tmp_path):
        """Duplicate contig IDs should be skipped with warning."""
        fasta = tmp_path / "contigs.fna"
        fasta.write_text(
            ">contig_1\n"
            "ATGCATGC\n"
            ">contig_1\n"
            "GGCCTTAA\n"
        )
        contigs = read_nucleotide_fasta(fasta)
        assert len(contigs) == 1
        assert contigs["contig_1"] == "ATGCATGC"

    def test_file_not_found(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_nucleotide_fasta(tmp_path / "missing.fna")

    def test_gzip_read(self, tmp_path):
        """Gzipped FASTA should be readable."""
        fasta = tmp_path / "contigs.fna.gz"
        content = b">contig_1\nATGCATGC\n>contig_2\nGGCCTTAA\n"
        with gzip.open(fasta, "wb") as f:
            f.write(content)
        contigs = read_nucleotide_fasta(fasta)
        assert len(contigs) == 2

    def test_empty_file(self, tmp_path):
        """Empty file should return empty dict."""
        fasta = tmp_path / "empty.fna"
        fasta.write_text("")
        contigs = read_nucleotide_fasta(fasta)
        assert contigs == {}


class TestNucleotideFormatMapping:
    """Tests for .fna/.ffn extension mapping in IO __init__."""

    def test_fna_maps_to_nucleotide(self):
        """The .fna extension should map to 'nucleotide' format."""
        from vhold.io import _FORMAT_MAP
        assert _FORMAT_MAP[".fna"] == "nucleotide"

    def test_ffn_maps_to_nucleotide(self):
        """The .ffn extension should map to 'nucleotide' format."""
        from vhold.io import _FORMAT_MAP
        assert _FORMAT_MAP[".ffn"] == "nucleotide"

    def test_faa_still_maps_to_fasta(self):
        """The .faa extension should still map to 'fasta' (protein)."""
        from vhold.io import _FORMAT_MAP
        assert _FORMAT_MAP[".faa"] == "fasta"


class TestNucleotideInputError:
    """Tests for NucleotideInputError when reading nucleotide without gene caller."""

    def test_fna_without_gene_caller_raises(self, tmp_path):
        """Reading .fna without gene caller should raise NucleotideInputError."""
        from vhold.io import NucleotideInputError, read_input

        fasta = tmp_path / "contigs.fna"
        fasta.write_text(">contig_1\nATGCATGCATGC\n")
        with pytest.raises(NucleotideInputError, match="nucleotide"):
            read_input(fasta)

    def test_fna_with_gene_caller_returns_empty(self, tmp_path):
        """Reading .fna with gene caller should return empty dict."""
        from vhold.io import read_input

        fasta = tmp_path / "contigs.fna"
        fasta.write_text(">contig_1\nATGCATGCATGC\n")
        result = read_input(fasta, gene_caller="auto")
        assert result == {}

    def test_nucleotide_fasta_content_detected(self, tmp_path):
        """Nucleotide content in .fasta file should raise without gene caller."""
        from vhold.io import NucleotideInputError, read_input

        fasta = tmp_path / "contigs.fasta"
        fasta.write_text(
            ">contig_1\n"
            "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC\n"
            ">contig_2\n"
            "GGCCTTAATTCCGGAATTCCGGAATTCCGGAATTCCGGAA\n"
        )
        with pytest.raises(NucleotideInputError, match="nucleotide"):
            read_input(fasta)

    def test_protein_fasta_content_passes(self, tmp_path):
        """Protein content in .fasta file should pass through normally."""
        from vhold.io import read_input

        fasta = tmp_path / "proteins.fasta"
        fasta.write_text(
            ">prot_1\n"
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH\n"
        )
        result = read_input(fasta)
        assert "prot_1" in result
