"""End-to-end integration tests with mocked heavy dependencies.

Tests the full pipeline flow without requiring:
- ProstT5 model loading (~2GB download, GPU)
- Foldseek databases (~5GB download)
- Embedding database (~822MB)
- LLM API access

Instead, mocks return realistic synthetic data so the annotation,
classification, consensus, and output logic run for real.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from vhold.results.consensus import ConsensusResult


# ============================================================================
# Test fixtures
# ============================================================================


MOCK_PROTEINS = {
    "polymerase_1": "MDYKDDDDKGGGGMFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGV",
    "capsid_1": "MSTLNIDQALAGNRKSTLNIDQALAGNRKSTLNIDQALAGNRK",
    "protease_1": "MFRQVLPAEFALLRLNAAPKLQVLPAEFALLRLNAAPK",
    "unknown_1": "MKKLLGGFDETGVKVKKLLGGFDETGVKV",
    "unknown_2": "MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEK",
}

MOCK_FASTA_CONTENT = "\n".join(
    f">{pid}\n{seq}" for pid, seq in MOCK_PROTEINS.items()
)


def _make_mock_consensus(query_id, length, description, category,
                         source="keywords", score=0.95, novelty="close_homolog"):
    """Create a ConsensusResult with realistic annotation data."""
    return ConsensusResult(
        query_id=query_id,
        query_length=length,
        consensus_score=score,
        confidence_level="high" if score >= 0.8 else "medium",
        agreement="single",
        functional_category=category,
        classification_source=source,
        novelty=novelty,
        primary_source="bfvd",
        primary_annotation={
            "description": description,
            "organism": "Test virus",
            "gene": query_id.split("_")[0],
        },
    )


def _mock_predictor(protein_ids_to_lengths):
    """Create a mock ProstT5 predictor returning fake 3Di results."""
    predictor = MagicMock()

    @dataclass
    class FakePrediction:
        sequence_id: str
        aa_sequence: str
        three_di_sequence: str
        confidence_scores: list

        @property
        def mean_confidence(self):
            return sum(self.confidence_scores) / len(self.confidence_scores)

    predictions = {
        pid: FakePrediction(
            sequence_id=pid,
            aa_sequence="A" * length,
            three_di_sequence="D" * length,
            confidence_scores=[0.85] * length,
        )
        for pid, length in protein_ids_to_lengths.items()
    }
    predictor.predict_batch.return_value = predictions
    return predictor


def _pipeline_mocks(predictor, annotations):
    """Return a context manager stacking all pipeline mocks."""
    from contextlib import ExitStack

    stack = ExitStack()
    patches = [
        patch("vhold.subcommands.run.get_predictor", return_value=predictor),
        patch("vhold.subcommands.run.search_databases", return_value=[]),
        patch("vhold.subcommands.run.merge_results", return_value=[]),
        patch("vhold.subcommands.run.parse_dataframe_results", return_value={}),
        patch("vhold.subcommands.run.transfer_annotations_consensus",
              return_value=annotations),
    ]
    for p in patches:
        stack.enter_context(p)
    return stack


# ============================================================================
# Integration tests
# ============================================================================


class TestPipelineEndToEnd:
    """Test full pipeline flow with mocked model/database dependencies."""

    def test_fasta_pipeline_produces_output(self, tmp_path):
        """Full FASTA pipeline produces TSV, JSON, and dark matter report."""
        input_fasta = tmp_path / "test.fasta"
        input_fasta.write_text(MOCK_FASTA_CONTENT)
        output_dir = tmp_path / "results"

        mock_annotations = {
            "polymerase_1": _make_mock_consensus(
                "polymerase_1", 49, "RNA-dependent RNA polymerase", "replication",
            ),
            "capsid_1": _make_mock_consensus(
                "capsid_1", 43, "major capsid protein", "structural",
            ),
            "protease_1": _make_mock_consensus(
                "protease_1", 38, "main protease", "protease",
            ),
        }

        predictor = _mock_predictor({
            pid: len(seq) for pid, seq in MOCK_PROTEINS.items()
        })

        with _pipeline_mocks(predictor, mock_annotations):
            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(input_fasta),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
            )

        # Verify output files exist
        assert (output_dir / "vhold_results.tsv").exists()
        assert (output_dir / "vhold_summary.json").exists()

        # Verify TSV content
        tsv = (output_dir / "vhold_results.tsv").read_text()
        lines = tsv.strip().split("\n")
        assert len(lines) == 6  # header + 5 proteins
        header = lines[0].split("\t")
        assert "functional_category" in header
        assert "consensus_score" in header

        # Verify categories in TSV
        assert "replication" in tsv
        assert "structural" in tsv
        assert "protease" in tsv
        assert "unknown" in tsv

        # Verify JSON summary
        with open(output_dir / "vhold_summary.json") as f:
            summary = json.load(f)
        assert summary["statistics"]["total_proteins"] == 5
        assert summary["statistics"]["annotated"] == 3
        assert summary["statistics"]["annotation_rate"] == 3 / 5

    def test_genbank_pipeline_with_positions(self, tmp_path):
        """GenBank input propagates genomic positions to output."""
        gb_content = """\
LOCUS       TEST_VIRUS               5000 bp    DNA     linear   VRL 01-JAN-2026
DEFINITION  Test virus, complete genome.
ACCESSION   TEST001
VERSION     TEST001.1
FEATURES             Location/Qualifiers
     CDS             100..400
                     /protein_id="prot_capsid"
                     /product="major capsid protein"
                     /translation="MSTLNIDQALAGNRKSTLNIDQALAGNRKSTLNIDQ"
     CDS             complement(500..800)
                     /protein_id="prot_poly"
                     /product="RNA polymerase"
                     /translation="MDYKDDDDKGGGGMFVFLVLLPLVSSQCVNLTTRTQ"
     CDS             900..1200
                     /protein_id="prot_hyp"
                     /product="hypothetical protein"
                     /translation="MKKLLGGFDETGVKVKKLLGGFDETGVKVK"
ORIGIN
        1 atgcatgcat
//
"""
        gb_file = tmp_path / "test.gb"
        gb_file.write_text(gb_content)
        output_dir = tmp_path / "results"

        predictor = _mock_predictor({
            "prot_capsid": 37, "prot_poly": 37, "prot_hyp": 30,
        })
        mock_annotations = {
            "prot_capsid": _make_mock_consensus(
                "prot_capsid", 37, "major capsid protein", "structural",
            ),
            "prot_poly": _make_mock_consensus(
                "prot_poly", 37, "RNA polymerase", "replication",
            ),
        }

        with _pipeline_mocks(predictor, mock_annotations):
            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(gb_file),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
            )

        tsv = (output_dir / "vhold_results.tsv").read_text()
        assert "contig" in tsv
        assert "TEST001.1" in tsv

    def test_triage_skips_decoder_for_matched(self, tmp_path):
        """Embedding triage matched proteins skip ProstT5 decoder."""
        input_fasta = tmp_path / "test.fasta"
        input_fasta.write_text(
            ">capsid_1\nMSTLNIDQALAGNRK\n>unknown_1\nMKKLLGGFDETGVKV\n"
        )
        output_dir = tmp_path / "results"

        @dataclass
        class MockTriageResult:
            matched: dict = field(default_factory=dict)
            unmatched: list = field(default_factory=list)
            stats: dict = field(default_factory=dict)

        triage_result = MockTriageResult(
            matched={"capsid_1": [MagicMock()]},
            unmatched=["unknown_1"],
            stats={"matched": 1, "unmatched": 1},
        )

        triage_consensus = {
            "capsid_1": _make_mock_consensus(
                "capsid_1", 15, "major capsid protein", "structural",
                source="embedding", novelty="embedding_match",
            ),
        }

        predictor = _mock_predictor({"unknown_1": 15})
        mock_emb_db = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor.model = None
        mock_extractor.tokenizer = None

        with _pipeline_mocks(predictor, {}), \
             patch("vhold.databases.embeddings.check_embedding_db", return_value=True), \
             patch("vhold.databases.embeddings.get_embedding_db_path", return_value="/fake"), \
             patch("vhold.features.embeddings.EmbeddingDatabase", return_value=mock_emb_db), \
             patch("vhold.features.embeddings.triage_proteins",
                   return_value=(triage_result, mock_extractor)), \
             patch("vhold.features.embeddings.build_embedding_consensus_results",
                   return_value=triage_consensus):

            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(input_fasta),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
                triage=True,
            )

        # Decoder should only have been called with unmatched proteins
        call_args = predictor.predict_batch.call_args
        sequences_passed = call_args[0][0]
        assert "unknown_1" in sequences_passed
        assert "capsid_1" not in sequences_passed

        # Both proteins should be in output
        tsv = (output_dir / "vhold_results.tsv").read_text()
        assert "capsid_1" in tsv
        assert "unknown_1" in tsv

    def test_neighborhood_voting_activates_with_positions(self, tmp_path):
        """Neighborhood voting runs when positional data is available."""
        gb_content = """\
LOCUS       TEST_VIRUS               5000 bp    DNA     linear   VRL 01-JAN-2026
DEFINITION  Test virus, complete genome.
ACCESSION   TEST002
VERSION     TEST002.1
FEATURES             Location/Qualifiers
     CDS             100..400
                     /protein_id="prot_1"
                     /product="major capsid protein"
                     /translation="MSTLNIDQALAGNRKSTLNIDQALAGNRKSTLNIDQ"
     CDS             450..750
                     /protein_id="prot_2"
                     /product="tail fiber protein"
                     /translation="MDYKDDDDKGGGGMFVFLVLLPLVSSQCVNLTTRTQ"
     CDS             800..1100
                     /protein_id="prot_3"
                     /product="hypothetical protein"
                     /translation="MKKLLGGFDETGVKVKKLLGGFDETGVKVK"
     CDS             1150..1450
                     /protein_id="prot_4"
                     /product="coat protein"
                     /translation="MAAHKGAEHHHKAAEHHEQAAKHHHAAAEHHEK"
ORIGIN
        1 atgcatgcat
//
"""
        gb_file = tmp_path / "test.gb"
        gb_file.write_text(gb_content)
        output_dir = tmp_path / "results"

        predictor = _mock_predictor({
            "prot_1": 37, "prot_2": 37, "prot_3": 30, "prot_4": 33,
        })
        mock_annotations = {
            "prot_1": _make_mock_consensus(
                "prot_1", 37, "major capsid protein", "structural",
            ),
            "prot_2": _make_mock_consensus(
                "prot_2", 37, "tail fiber protein", "structural",
            ),
            # prot_3 not in annotations → unknown → should be reclassified
            "prot_4": _make_mock_consensus(
                "prot_4", 33, "coat protein", "structural",
            ),
        }

        with _pipeline_mocks(predictor, mock_annotations):
            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(gb_file),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
            )

        # prot_3 should be reclassified by neighborhood voting
        tsv = (output_dir / "vhold_results.tsv").read_text()
        lines = tsv.strip().split("\n")
        header = lines[0].split("\t")
        cat_idx = header.index("functional_category")
        src_idx = header.index("classification_source")

        prot3_row = [l for l in lines if l.startswith("prot_3")][0].split("\t")
        assert prot3_row[cat_idx] == "structural"
        assert prot3_row[src_idx] == "neighborhood_vote"

    def test_llm_reclassify_called_when_enabled(self, tmp_path):
        """LLM classification is invoked when enabled."""
        input_fasta = tmp_path / "test.fasta"
        input_fasta.write_text(">prot1\nMSTLNIDQALAGNRK\n")
        output_dir = tmp_path / "results"

        predictor = _mock_predictor({"prot1": 15})
        mock_annotations = {
            "prot1": _make_mock_consensus(
                "prot1", 15, "hypothetical protein", "unknown",
            ),
        }

        with _pipeline_mocks(predictor, mock_annotations), \
             patch("vhold.results.llm_classify.llm_reclassify",
                   return_value=mock_annotations) as mock_llm:

            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(input_fasta),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
                llm_classify=True,
                classify=False,  # Skip MLP classifier (avoids ProstT5 load)
            )

        mock_llm.assert_called_once()

    def test_output_json_summary_statistics(self, tmp_path):
        """Summary JSON includes correct statistics."""
        input_fasta = tmp_path / "test.fasta"
        input_fasta.write_text(
            ">rdrp\nMDYKDDDDK\n>capsid\nMSTLNIDQA\n>hyp\nMKKLLGGFD\n"
        )
        output_dir = tmp_path / "results"

        predictor = _mock_predictor({"rdrp": 9, "capsid": 9, "hyp": 9})
        mock_annotations = {
            "rdrp": _make_mock_consensus(
                "rdrp", 9, "RNA-dependent RNA polymerase", "replication",
            ),
            "capsid": _make_mock_consensus(
                "capsid", 9, "major capsid protein", "structural",
            ),
        }

        with _pipeline_mocks(predictor, mock_annotations):
            from vhold.subcommands.run import run_pipeline
            run_pipeline(
                input_file=str(input_fasta),
                output_dir=str(output_dir),
                threads=1,
                device="cpu",
                fast=True,
            )

        with open(output_dir / "vhold_summary.json") as f:
            summary = json.load(f)

        stats = summary["statistics"]
        assert stats["total_proteins"] == 3
        assert stats["annotated"] == 2
        assert stats["unannotated"] == 1
        cats = stats["category_distribution"]
        assert cats["replication"] == 1
        assert cats["structural"] == 1
        assert cats["unknown"] == 1
