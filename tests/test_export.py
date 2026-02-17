"""Tests for metagenomic export formats."""

import pandas as pd
import pytest
from pathlib import Path

from vhold.results.consensus import ConsensusResult
from vhold.results.export import (
    EXPORT_FORMATS,
    contig_id_from_protein_id,
    export_anvio,
    export_vcontact2,
    export_vcontact3,
    export_dramv,
    export_gff3,
    export_all_formats,
    load_results_tsv,
)
from vhold.results.output import write_consensus_tsv


# ============================================================================
# Fixtures
# ============================================================================


def _make_consensus(
    query_id, length, description, category,
    contig=None, start=None, end=None, strand=None,
    pfam=None, go_bp=None, go_bp_ids=None,
    go_mf=None, go_mf_ids=None,
    evalue=1e-20, target="BFVD:AF-Q9YMB0",
):
    """Create a ConsensusResult with realistic data."""
    from vhold.results.parser import FoldseekHit

    hit = FoldseekHit(
        query=query_id,
        target=target,
        fident=0.45,
        alnlen=length,
        mismatch=0,
        gapopen=0,
        qstart=1,
        qend=length,
        tstart=1,
        tend=length,
        evalue=evalue,
        bits=120.0,
        qlen=length,
        tlen=length,
        qcov=0.95,
        tcov=0.90,
        source_db="bfvd",
    )

    annotation = {
        "description": description,
        "organism": "Test virus",
        "gene": query_id.split("_")[0],
    }
    if pfam:
        annotation["pfam"] = pfam
    if go_bp:
        annotation["go_bp"] = go_bp
    if go_bp_ids:
        annotation["go_bp_ids"] = go_bp_ids
    if go_mf:
        annotation["go_mf"] = go_mf
    if go_mf_ids:
        annotation["go_mf_ids"] = go_mf_ids

    return ConsensusResult(
        query_id=query_id,
        query_length=length,
        primary_hit=hit,
        primary_annotation=annotation,
        primary_source="bfvd",
        consensus_score=0.85,
        confidence_level="high",
        agreement="single",
        functional_category=category,
        classification_source="keywords",
        contig=contig,
        start=start,
        end=end,
        strand=strand,
        hits_by_db={"bfvd": [hit]},
    )


@pytest.fixture
def sample_results():
    """4 proteins across 2 contigs with realistic annotations."""
    return {
        "contig1_1": _make_consensus(
            "contig1_1", 300, "RNA-dependent RNA polymerase", "replication",
            contig="contig1", start=100, end=1000, strand=1,
            pfam="PF00680", go_bp="viral genome replication",
            go_bp_ids="GO:0019079",
        ),
        "contig1_2": _make_consensus(
            "contig1_2", 250, "major capsid protein", "structural",
            contig="contig1", start=1100, end=1850, strand=1,
        ),
        "contig1_3": _make_consensus(
            "contig1_3", 100, "hypothetical protein", "unknown",
            contig="contig1", start=1900, end=2200, strand=-1,
            target="", evalue=1.0,
        ),
        "contig2_1": _make_consensus(
            "contig2_1", 400, "main protease", "protease",
            contig="contig2", start=50, end=1250, strand=1,
            pfam="PF05409",
            go_mf="cysteine-type peptidase activity",
            go_mf_ids="GO:0008234",
        ),
    }


@pytest.fixture
def sample_results_no_positions():
    """Results from FASTA input (no genomic coordinates)."""
    return {
        "protein_A": _make_consensus(
            "protein_A", 300, "RNA-dependent RNA polymerase", "replication",
        ),
        "protein_B": _make_consensus(
            "protein_B", 250, "capsid protein", "structural",
        ),
    }


# ============================================================================
# contig_id_from_protein_id tests
# ============================================================================


class TestContigIdExtraction:
    def test_explicit_contig(self):
        assert contig_id_from_protein_id("prot_1", contig="contig1") == "contig1"

    def test_prodigal_convention(self):
        assert contig_id_from_protein_id("contig_1_42") == "contig_1"

    def test_prodigal_simple(self):
        assert contig_id_from_protein_id("scaffold1_3") == "scaffold1"

    def test_genomad_provirus(self):
        assert contig_id_from_protein_id(
            "seq1|provirus_100_5000_3"
        ) == "seq1|provirus_100_5000"

    def test_no_numeric_suffix(self):
        """Protein IDs without trailing _N should return as-is."""
        assert contig_id_from_protein_id("proteinA") == "proteinA"

    def test_explicit_overrides_suffix(self):
        assert contig_id_from_protein_id("contig1_5", contig="real_contig") == "real_contig"

    def test_complex_prodigal_id(self):
        # k141_12345_67 -> k141_12345
        assert contig_id_from_protein_id("k141_12345_67") == "k141_12345"


# ============================================================================
# anvi'o export tests
# ============================================================================


class TestAnvioExport:
    def test_basic_export(self, sample_results, tmp_path):
        output = tmp_path / "functions.txt"
        export_anvio(sample_results, output)

        assert output.exists()
        df = pd.read_csv(output, sep="\t")
        assert set(df.columns) == {
            "gene_callers_id", "source", "accession", "function", "e_value",
        }

    def test_multiple_rows_per_gene(self, sample_results, tmp_path):
        output = tmp_path / "functions.txt"
        export_anvio(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        # Gene 0 (RdRp): structural + Pfam + GO_BP + category = 4 rows
        gene0_rows = df[df["gene_callers_id"] == 0]
        sources = set(gene0_rows["source"])
        assert "vHold_structural" in sources
        assert "vHold_Pfam" in sources
        assert "vHold_GO_BP" in sources
        assert "vHold_category" in sources
        assert len(gene0_rows) == 4

    def test_source_prefix(self, sample_results, tmp_path):
        output = tmp_path / "functions.txt"
        export_anvio(sample_results, output, source_prefix="myTool")

        df = pd.read_csv(output, sep="\t")
        assert all(src.startswith("myTool_") for src in df["source"])

    def test_unknown_protein_gets_category_only(self, sample_results, tmp_path):
        output = tmp_path / "functions.txt"
        export_anvio(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        # Gene 2 (hypothetical) should only have category row
        gene2_rows = df[df["gene_callers_id"] == 2]
        assert len(gene2_rows) == 1
        assert gene2_rows.iloc[0]["source"] == "vHold_category"

    def test_protease_with_go_mf(self, sample_results, tmp_path):
        output = tmp_path / "functions.txt"
        export_anvio(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        # Gene 3 (protease): structural + Pfam + GO_MF + category = 4
        gene3_rows = df[df["gene_callers_id"] == 3]
        sources = set(gene3_rows["source"])
        assert "vHold_GO_MF" in sources
        assert "vHold_Pfam" in sources


# ============================================================================
# vConTACT2 export tests
# ============================================================================


class TestVcontact2Export:
    def test_basic_export(self, sample_results, tmp_path):
        output = tmp_path / "gene2genome.csv"
        export_vcontact2(sample_results, output)

        assert output.exists()
        df = pd.read_csv(output)
        assert set(df.columns) == {"protein_id", "contig_id", "keywords"}
        assert len(df) == 4

    def test_contig_derivation_from_explicit(self, sample_results, tmp_path):
        output = tmp_path / "gene2genome.csv"
        export_vcontact2(sample_results, output)

        df = pd.read_csv(output)
        # All 4 proteins have explicit contig
        assert "contig1" in df["contig_id"].values
        assert "contig2" in df["contig_id"].values

    def test_fasta_input_contig_derivation(self, sample_results_no_positions, tmp_path):
        output = tmp_path / "gene2genome.csv"
        export_vcontact2(sample_results_no_positions, output)

        df = pd.read_csv(output)
        # protein_A doesn't end in _<number>, so contig = protein_A as-is
        assert "protein_A" in df["contig_id"].values

    def test_keywords_include_category(self, sample_results, tmp_path):
        output = tmp_path / "gene2genome.csv"
        export_vcontact2(sample_results, output)

        df = pd.read_csv(output)
        rdrp_row = df[df["protein_id"] == "contig1_1"].iloc[0]
        assert "replication" in rdrp_row["keywords"]
        assert "RNA-dependent" in rdrp_row["keywords"]


# ============================================================================
# vConTACT3 export tests
# ============================================================================


class TestVcontact3Export:
    def test_produces_two_files(self, sample_results, tmp_path):
        result = export_vcontact3(sample_results, tmp_path / "vc3")

        assert "gene2genome" in result
        assert "genome_lengths" in result
        assert result["gene2genome"].exists()
        assert result["genome_lengths"].exists()

    def test_gene2genome_format(self, sample_results, tmp_path):
        result = export_vcontact3(sample_results, tmp_path / "vc3")

        df = pd.read_csv(result["gene2genome"], sep="\t")
        assert set(df.columns) == {"protein_id", "genome_id"}
        assert len(df) == 4

    def test_genome_lengths_format(self, sample_results, tmp_path):
        result = export_vcontact3(sample_results, tmp_path / "vc3")

        df = pd.read_csv(result["genome_lengths"], sep="\t")
        assert set(df.columns) == {"genome_id", "length"}
        assert len(df) == 2  # contig1 and contig2

    def test_genome_lengths_sum_proteins(self, sample_results, tmp_path):
        result = export_vcontact3(sample_results, tmp_path / "vc3")

        df = pd.read_csv(result["genome_lengths"], sep="\t")
        c1 = df[df["genome_id"] == "contig1"].iloc[0]["length"]
        # contig1: 300 + 250 + 100 = 650
        assert c1 == 650


# ============================================================================
# DRAM-v export tests
# ============================================================================


class TestDramvExport:
    def test_basic_export(self, sample_results, tmp_path):
        output = tmp_path / "dramv.tsv"
        export_dramv(sample_results, output)

        assert output.exists()
        df = pd.read_csv(output, sep="\t")
        expected_cols = {
            "scaffold", "gene_position", "start_position", "end_position",
            "strandedness", "vhold_description", "vhold_category",
            "vhold_confidence", "vhold_pfam", "vhold_go_bp", "vhold_go_mf",
        }
        assert expected_cols == set(df.columns)

    def test_scaffold_derivation(self, sample_results, tmp_path):
        output = tmp_path / "dramv.tsv"
        export_dramv(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        assert "contig1" in df["scaffold"].values
        assert "contig2" in df["scaffold"].values

    def test_gene_position_incrementing(self, sample_results, tmp_path):
        output = tmp_path / "dramv.tsv"
        export_dramv(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        contig1_rows = df[df["scaffold"] == "contig1"]
        positions = contig1_rows["gene_position"].tolist()
        assert positions == [1, 2, 3]

    def test_strandedness(self, sample_results, tmp_path):
        output = tmp_path / "dramv.tsv"
        export_dramv(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        # contig1_3 has strand=-1 -> "-"
        row3 = df[(df["scaffold"] == "contig1") & (df["gene_position"] == 3)].iloc[0]
        assert row3["strandedness"] == "-"
        # contig1_1 has strand=1 -> "+"
        row1 = df[(df["scaffold"] == "contig1") & (df["gene_position"] == 1)].iloc[0]
        assert row1["strandedness"] == "+"

    def test_annotations_present(self, sample_results, tmp_path):
        output = tmp_path / "dramv.tsv"
        export_dramv(sample_results, output)

        df = pd.read_csv(output, sep="\t")
        rdrp_row = df[df["vhold_description"] == "RNA-dependent RNA polymerase"].iloc[0]
        assert rdrp_row["vhold_category"] == "replication"
        assert rdrp_row["vhold_confidence"] == "high"
        assert rdrp_row["vhold_pfam"] == "PF00680"


# ============================================================================
# GFF3 export tests
# ============================================================================


class TestGff3Export:
    def test_basic_export(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        assert output.exists()
        content = output.read_text()
        assert content.startswith("##gff-version 3")

    def test_only_positioned_proteins(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        lines = [l for l in output.read_text().splitlines() if not l.startswith("#")]
        assert len(lines) == 4  # All 4 have positions

    def test_skips_unpositioned(self, sample_results_no_positions, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results_no_positions, output)

        lines = [l for l in output.read_text().splitlines() if not l.startswith("#")]
        assert len(lines) == 0

    def test_attributes_format(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        content = output.read_text()
        assert "vhold_category=" in content
        assert "vhold_confidence=" in content
        assert "ID=" in content

    def test_go_terms_in_gff3(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        content = output.read_text()
        assert "Ontology_term=" in content
        assert "GO:0019079" in content

    def test_pfam_dbxref(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        content = output.read_text()
        assert "Dbxref=Pfam:" in content

    def test_strand_encoding(self, sample_results, tmp_path):
        output = tmp_path / "annotations.gff3"
        export_gff3(sample_results, output)

        lines = [l for l in output.read_text().splitlines() if not l.startswith("#")]
        # contig1_3 is on - strand
        negative_lines = [l for l in lines if "\t-\t" in l]
        assert len(negative_lines) == 1
        assert "contig1_3" in negative_lines[0]


# ============================================================================
# TSV round-trip tests
# ============================================================================


class TestTsvRoundTrip:
    def test_export_from_dataframe(self, sample_results, tmp_path):
        """Export should work from both ConsensusResult dict and DataFrame."""
        # Write TSV from ConsensusResult
        tsv_path = tmp_path / "results.tsv"
        write_consensus_tsv(sample_results, tsv_path)

        # Read back as DataFrame
        df = load_results_tsv(tsv_path)
        assert len(df) == 4

        # Export from DataFrame
        output = tmp_path / "from_df.csv"
        export_vcontact2(df, output)

        assert output.exists()
        result_df = pd.read_csv(output)
        assert len(result_df) == 4

    def test_anvio_from_tsv_roundtrip(self, sample_results, tmp_path):
        """anvi'o export works from TSV round-trip."""
        tsv_path = tmp_path / "results.tsv"
        write_consensus_tsv(sample_results, tsv_path)

        df = load_results_tsv(tsv_path)
        output = tmp_path / "functions.txt"
        export_anvio(df, output)

        assert output.exists()
        result_df = pd.read_csv(output, sep="\t")
        assert len(result_df) > 0
        assert "vHold_category" in result_df["source"].values


# ============================================================================
# export_all_formats tests
# ============================================================================


class TestExportAll:
    def test_all_formats(self, sample_results, tmp_path):
        result = export_all_formats(sample_results, tmp_path, formats=["all"])
        assert "anvio" in result
        assert "vcontact2" in result
        assert "vcontact3" in result
        assert "dramv" in result
        assert "gff3" in result

    def test_specific_formats(self, sample_results, tmp_path):
        result = export_all_formats(
            sample_results, tmp_path, formats=["anvio", "dramv"],
        )
        assert "anvio" in result
        assert "dramv" in result
        assert "vcontact2" not in result

    def test_none_means_all(self, sample_results, tmp_path):
        result = export_all_formats(sample_results, tmp_path, formats=None)
        assert len(result) == 5

    def test_output_file_naming(self, sample_results, tmp_path):
        result = export_all_formats(
            sample_results, tmp_path, formats=["all"], prefix="test",
        )
        assert result["anvio"].name == "test_anvio_functions.txt"
        assert result["vcontact2"].name == "test_vcontact2_gene2genome.csv"
        assert result["dramv"].name == "test_dramv_annotations.tsv"
        assert result["gff3"].name == "test_annotations.gff3"
