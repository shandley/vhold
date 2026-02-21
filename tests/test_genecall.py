"""Tests for gene calling module (Pyrodigal-gv + miniprot)."""

from unittest.mock import MagicMock, patch

import pytest

from vhold.features.genecall import (
    CalledGene,
    MiniprotCaller,
    PyrodigalCaller,
    _compute_overlap,
    _genes_to_protein_records,
    _merge_gene_calls,
    _write_called_genes_gff3,
    _write_called_proteins_fasta,
)

# ============================================================
# CalledGene dataclass tests
# ============================================================

class TestCalledGene:
    """Tests for CalledGene dataclass."""

    def test_basic_creation(self):
        gene = CalledGene(
            contig="contig_1",
            start=100,
            end=400,
            strand=1,
            protein_sequence="MVLSPAD",
            gene_id="contig_1_1",
            caller="pyrodigal",
        )
        assert gene.contig == "contig_1"
        assert gene.start == 100
        assert gene.end == 400
        assert gene.strand == 1
        assert gene.caller == "pyrodigal"
        assert gene.is_spliced is False
        assert gene.exon_parts == []

    def test_spliced_gene(self):
        gene = CalledGene(
            contig="contig_1",
            start=100,
            end=700,
            strand=1,
            protein_sequence="MVLSPAD",
            gene_id="contig_1_mp_1",
            caller="miniprot",
            is_spliced=True,
            exon_parts=[(100, 300), (500, 700)],
        )
        assert gene.is_spliced is True
        assert len(gene.exon_parts) == 2


# ============================================================
# PyrodigalCaller tests
# ============================================================

class TestPyrodigalCaller:
    """Tests for PyrodigalCaller wrapper."""

    def test_import_error_without_pyrodigal(self):
        """Should raise ImportError when pyrodigal not installed."""
        with patch("vhold.features.genecall._check_pyrodigal_available", return_value=False):
            with pytest.raises(ImportError, match="pyrodigal"):
                PyrodigalCaller()

    def test_init_with_defaults(self):
        """Should initialize with default parameters."""
        with patch("vhold.features.genecall._check_pyrodigal_available", return_value=True):
            caller = PyrodigalCaller()
            assert caller.translation_table == 11
            assert caller.closed is False
            assert caller.min_gene == 90
            assert caller.meta is True

    def test_init_custom_params(self):
        """Should accept custom parameters."""
        with patch("vhold.features.genecall._check_pyrodigal_available", return_value=True):
            caller = PyrodigalCaller(
                translation_table=1,
                closed=True,
                min_gene=150,
                meta=False,
            )
            assert caller.translation_table == 1
            assert caller.closed is True
            assert caller.min_gene == 150
            assert caller.meta is False

    def test_empty_contigs(self):
        """Empty contigs dict should return empty list."""
        with patch("vhold.features.genecall._check_pyrodigal_available", return_value=True):
            caller = PyrodigalCaller()
            # Mock pyrodigal import
            with patch.dict("sys.modules", {"pyrodigal": MagicMock()}):
                result = caller.call_genes({})
                assert result == []


# ============================================================
# MiniprotCaller tests
# ============================================================

class TestMiniprotCaller:
    """Tests for MiniprotCaller wrapper."""

    def test_init_requires_miniprot(self):
        """Should fail if miniprot not available."""
        with patch("vhold.utils.external.ExternalTool.check_available",
                    side_effect=FileNotFoundError("miniprot not found")):
            with pytest.raises(FileNotFoundError):
                MiniprotCaller(ref_db="/fake/refs.faa")

    def test_parse_gff_empty(self):
        """Empty GFF output should return empty list."""
        with patch("vhold.utils.external.ExternalTool.check_available"):
            caller = MiniprotCaller.__new__(MiniprotCaller)
            caller.tool = MagicMock()
            caller.ref_db = "/fake/refs.faa"
            caller.threads = 4
            result = caller._parse_gff("")
            assert result == []

    def test_parse_gff_comments_only(self):
        """GFF with only comments should return empty list."""
        with patch("vhold.utils.external.ExternalTool.check_available"):
            caller = MiniprotCaller.__new__(MiniprotCaller)
            caller.tool = MagicMock()
            caller.ref_db = "/fake/refs.faa"
            caller.threads = 4
            result = caller._parse_gff("##gff-version 3\n# comment\n")
            assert result == []

    def test_parse_gff_spliced_gene(self):
        """Should parse spliced gene from miniprot GFF output."""
        gff = (
            "##gff-version 3\n"
            "contig_1\tminiprot\tmRNA\t100\t700\t500\t+\t.\t"
            "ID=MP000001.mrna1;Target=TAT_PROT 1 100;Identity=0.95\n"
            "contig_1\tminiprot\tCDS\t100\t300\t500\t+\t0\t"
            "Parent=MP000001.mrna1\n"
            "contig_1\tminiprot\tCDS\t500\t700\t500\t+\t0\t"
            "Parent=MP000001.mrna1\n"
        )
        caller = MiniprotCaller.__new__(MiniprotCaller)
        caller.tool = MagicMock()
        caller.ref_db = "/fake/refs.faa"
        caller.threads = 4
        result = caller._parse_gff(gff)

        assert len(result) == 1
        gene = result[0]
        assert gene.contig == "contig_1"
        assert gene.is_spliced is True
        assert gene.exon_parts == [(100, 300), (500, 700)]
        assert gene.strand == 1
        assert gene.caller == "miniprot"

    def test_parse_gff_unspliced_gene(self):
        """Should parse single-exon gene from miniprot GFF output."""
        gff = (
            "contig_1\tminiprot\tmRNA\t100\t400\t300\t-\t.\t"
            "ID=MP000001.mrna1;Target=GAG_PROT 1 100\n"
            "contig_1\tminiprot\tCDS\t100\t400\t300\t-\t0\t"
            "Parent=MP000001.mrna1\n"
        )
        caller = MiniprotCaller.__new__(MiniprotCaller)
        caller.tool = MagicMock()
        caller.ref_db = "/fake/refs.faa"
        caller.threads = 4
        result = caller._parse_gff(gff)

        assert len(result) == 1
        gene = result[0]
        assert gene.is_spliced is False
        assert gene.strand == -1


# ============================================================
# Overlap computation tests
# ============================================================

class TestComputeOverlap:
    """Tests for _compute_overlap function."""

    def test_no_overlap(self):
        a = CalledGene("c1", 100, 200, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 300, 400, 1, "BBB", "g2", "miniprot")
        assert _compute_overlap(a, b) == 0.0

    def test_full_overlap(self):
        a = CalledGene("c1", 100, 200, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 100, 200, 1, "BBB", "g2", "miniprot")
        assert _compute_overlap(a, b) == 1.0

    def test_partial_overlap(self):
        a = CalledGene("c1", 100, 200, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 150, 250, 1, "BBB", "g2", "miniprot")
        # Overlap: 150-200 = 51 nt
        # len_a = 101, len_b = 101
        # max(51/101, 51/101) ≈ 0.505
        overlap = _compute_overlap(a, b)
        assert 0.49 < overlap < 0.52

    def test_different_contigs_no_overlap(self):
        a = CalledGene("c1", 100, 200, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c2", 100, 200, 1, "BBB", "g2", "miniprot")
        assert _compute_overlap(a, b) == 0.0

    def test_different_strands_no_overlap(self):
        a = CalledGene("c1", 100, 200, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 100, 200, -1, "BBB", "g2", "miniprot")
        assert _compute_overlap(a, b) == 0.0

    def test_contained_gene(self):
        """Smaller gene fully inside larger gene."""
        a = CalledGene("c1", 100, 500, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 200, 300, 1, "BBB", "g2", "miniprot")
        # b is fully inside a → overlap = 101 / 101 = 1.0
        assert _compute_overlap(a, b) == 1.0

    def test_adjacent_genes_no_overlap(self):
        """Adjacent genes (touching but not overlapping)."""
        a = CalledGene("c1", 100, 199, 1, "AAA", "g1", "pyrodigal")
        b = CalledGene("c1", 200, 300, 1, "BBB", "g2", "miniprot")
        assert _compute_overlap(a, b) == 0.0


# ============================================================
# Merge logic tests
# ============================================================

class TestMergeGeneCalls:
    """Tests for _merge_gene_calls function."""

    def test_no_miniprot_genes(self):
        """Should keep all Pyrodigal genes when no miniprot."""
        pg = [CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal")]
        result = _merge_gene_calls(pg, [])
        assert len(result) == 1
        assert result[0].caller == "pyrodigal"

    def test_non_overlapping_kept(self):
        """Non-overlapping miniprot genes should be added."""
        pg = [CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal")]
        mp = [CalledGene("c1", 500, 800, 1, "AAAA", "g2", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 2
        callers = {g.caller for g in result}
        assert "pyrodigal" in callers
        assert "miniprot" in callers

    def test_overlapping_miniprot_excluded(self):
        """Miniprot gene overlapping >50% with Pyrodigal should be excluded."""
        pg = [CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal")]
        mp = [CalledGene("c1", 100, 400, 1, "AAAA", "g2", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 1
        assert result[0].caller == "pyrodigal"

    def test_empty_protein_miniprot_excluded(self):
        """Miniprot genes without protein sequence should be excluded."""
        pg = [CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal")]
        mp = [CalledGene("c1", 500, 800, 1, "", "g2", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 1

    def test_different_contigs_no_overlap(self):
        """Genes on different contigs should never overlap."""
        pg = [CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal")]
        mp = [CalledGene("c2", 100, 400, 1, "AAAA", "g2", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 2

    def test_multiple_pyrodigal_one_miniprot_overlap(self):
        """Miniprot gene significantly overlapping a Pyrodigal gene should be excluded."""
        pg = [
            CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal"),
            CalledGene("c1", 500, 800, 1, "BBBB", "g2", "pyrodigal"),
        ]
        # This miniprot gene overlaps >50% with g1 (200-400 = 201bp, gene is 201bp)
        mp = [CalledGene("c1", 200, 400, 1, "CCCC", "g3", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 2
        assert all(g.caller == "pyrodigal" for g in result)

    def test_partial_overlap_below_threshold_kept(self):
        """Miniprot gene with <50% overlap should be kept."""
        pg = [
            CalledGene("c1", 100, 400, 1, "MVLS", "g1", "pyrodigal"),
        ]
        # Overlap: 300-400 = 101bp, miniprot gene = 301bp → 33.5% overlap
        mp = [CalledGene("c1", 300, 600, 1, "CCCC", "g3", "miniprot")]
        result = _merge_gene_calls(pg, mp)
        assert len(result) == 2


# ============================================================
# Gene-to-ProteinRecord conversion tests
# ============================================================

class TestGenesToProteinRecords:
    """Tests for _genes_to_protein_records function."""

    def test_basic_conversion(self):
        genes = [
            CalledGene("c1", 100, 400, 1, "MVLSPAD", "c1_1", "pyrodigal"),
            CalledGene("c1", 500, 800, -1, "AAABBB", "c1_2", "pyrodigal"),
        ]
        records = _genes_to_protein_records(genes)
        assert len(records) == 2
        assert "c1_1" in records
        assert "c1_2" in records
        assert records["c1_1"].sequence == "MVLSPAD"
        assert records["c1_1"].contig == "c1"
        assert records["c1_1"].start == 100
        assert records["c1_1"].end == 400
        assert records["c1_1"].strand == 1

    def test_empty_protein_skipped(self):
        """Genes without protein sequence should be skipped."""
        genes = [
            CalledGene("c1", 100, 400, 1, "", "c1_1", "pyrodigal"),
            CalledGene("c1", 500, 800, 1, "MVLS", "c1_2", "pyrodigal"),
        ]
        records = _genes_to_protein_records(genes)
        assert len(records) == 1
        assert "c1_2" in records

    def test_description_includes_caller(self):
        """Description should mention the caller."""
        genes = [CalledGene("c1", 100, 400, 1, "MVLS", "c1_1", "miniprot")]
        records = _genes_to_protein_records(genes)
        assert "miniprot" in records["c1_1"].description


# ============================================================
# GFF3 output tests
# ============================================================

class TestWriteCalledGenesGff3:
    """Tests for _write_called_genes_gff3 function."""

    def test_basic_gff3_output(self, tmp_path):
        genes = [
            CalledGene("c1", 100, 400, 1, "MVLS", "c1_1", "pyrodigal"),
        ]
        output = tmp_path / "genes.gff3"
        _write_called_genes_gff3(genes, output)

        content = output.read_text()
        assert "##gff-version 3" in content
        assert "c1\tpyrodigal\tCDS\t100\t400" in content
        assert "ID=c1_1" in content

    def test_spliced_gene_gff3_output(self, tmp_path):
        genes = [
            CalledGene(
                "c1", 100, 700, 1, "MVLS", "c1_mp_1", "miniprot",
                is_spliced=True, exon_parts=[(100, 300), (500, 700)],
            ),
        ]
        output = tmp_path / "genes.gff3"
        _write_called_genes_gff3(genes, output)

        content = output.read_text()
        assert "gene\t100\t700" in content
        assert "is_spliced=true" in content
        assert "CDS\t100\t300" in content
        assert "CDS\t500\t700" in content

    def test_minus_strand(self, tmp_path):
        genes = [
            CalledGene("c1", 100, 400, -1, "MVLS", "c1_1", "pyrodigal"),
        ]
        output = tmp_path / "genes.gff3"
        _write_called_genes_gff3(genes, output)

        content = output.read_text()
        assert "\t-\t" in content


# ============================================================
# Protein FASTA output tests
# ============================================================

class TestWriteCalledProteinsFasta:
    """Tests for _write_called_proteins_fasta function."""

    def test_basic_fasta_output(self, tmp_path):
        genes = [
            CalledGene("c1", 100, 400, 1, "MVLSPAD", "c1_1", "pyrodigal"),
            CalledGene("c1", 500, 800, 1, "AAABBB", "c1_2", "miniprot"),
        ]
        output = tmp_path / "proteins.faa"
        _write_called_proteins_fasta(genes, output)

        content = output.read_text()
        assert ">c1_1 caller=pyrodigal" in content
        assert "MVLSPAD" in content
        assert ">c1_2 caller=miniprot" in content
        assert "AAABBB" in content

    def test_empty_protein_skipped(self, tmp_path):
        genes = [
            CalledGene("c1", 100, 400, 1, "", "c1_1", "pyrodigal"),
            CalledGene("c1", 500, 800, 1, "MVLS", "c1_2", "pyrodigal"),
        ]
        output = tmp_path / "proteins.faa"
        _write_called_proteins_fasta(genes, output)

        content = output.read_text()
        assert "c1_1" not in content
        assert "c1_2" in content


# ============================================================
# CLI integration tests
# ============================================================

class TestGeneCaIlerCli:
    """Tests for gene caller CLI options."""

    def test_gene_caller_option_exists(self):
        """The --gene-caller option should be defined on the run command."""
        from click.testing import CliRunner

        from vhold.cli import run

        runner = CliRunner()
        result = runner.invoke(run, ["--help"])
        assert "--gene-caller" in result.output

    def test_translation_table_option_exists(self):
        """The --translation-table option should be defined."""
        from click.testing import CliRunner

        from vhold.cli import run

        runner = CliRunner()
        result = runner.invoke(run, ["--help"])
        assert "--translation-table" in result.output

    def test_closed_option_exists(self):
        """The --closed option should be defined."""
        from click.testing import CliRunner

        from vhold.cli import run

        runner = CliRunner()
        result = runner.invoke(run, ["--help"])
        assert "--closed" in result.output

    def test_min_gene_option_exists(self):
        """The --min-gene option should be defined."""
        from click.testing import CliRunner

        from vhold.cli import run

        runner = CliRunner()
        result = runner.invoke(run, ["--help"])
        assert "--min-gene" in result.output

    def test_nucleotide_fna_without_gene_caller_aborts(self, tmp_path):
        """Running with .fna without --gene-caller should abort."""
        from click.testing import CliRunner

        from vhold.cli import run

        fasta = tmp_path / "contigs.fna"
        fasta.write_text(">c1\nATGCATGCATGC\n")

        runner = CliRunner()
        result = runner.invoke(run, [
            "-i", str(fasta),
            "-o", str(tmp_path / "output"),
        ])
        assert result.exit_code != 0
        assert "nucleotide" in result.output.lower() or "gene-caller" in result.output.lower()
