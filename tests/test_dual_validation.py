"""Tests for dual validation (GO transfer + Foldseek)."""

import pytest

from vhold.features.go_transfer import GOTransferResult, TransferredGOTerm
from vhold.results.consensus import ConsensusResult
from vhold.results.dual_validation import (
    DualValidationResult,
    merge_go_into_consensus,
    validate_dual_evidence,
)


# ============================================================================
# Fixtures
# ============================================================================


def _make_go_result(
    query_id: str = "q1",
    go_terms: list[TransferredGOTerm] | None = None,
    best_similarity: float = 0.8,
    best_accession: str = "P12345",
    is_cross_domain: bool = False,
) -> GOTransferResult:
    """Helper to create GOTransferResult."""
    return GOTransferResult(
        query_id=query_id,
        top_k_matches=[(best_accession, best_similarity)],
        go_terms=go_terms or [],
        best_similarity=best_similarity,
        best_accession=best_accession,
        is_cross_domain=is_cross_domain,
    )


def _make_go_term(
    go_id: str = "GO:0039694",
    go_term: str = "viral genome replication",
    aspect: str = "BP",
    evidence_code: str = "IDA",
    ri: float = 0.85,
    is_cross_domain: bool = False,
    donor_organism: str = "Test virus",
    donor_lineage: str = "Viruses; test",
) -> TransferredGOTerm:
    """Helper to create a TransferredGOTerm."""
    return TransferredGOTerm(
        go_id=go_id,
        go_term=go_term,
        aspect=aspect,
        evidence_code=evidence_code,
        reliability_index=ri,
        donor_accession="P12345",
        donor_similarity=0.85,
        is_cross_domain=is_cross_domain,
        donor_organism=donor_organism,
        donor_lineage=donor_lineage,
    )


# ============================================================================
# Tests: validate_dual_evidence
# ============================================================================


class TestValidateDualEvidence:
    """Tests for the dual validation function."""

    def test_dual_agree(self):
        """Both GO and Foldseek assign the same category."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="replication",
            consensus_score=0.7,
        )
        go_result = _make_go_result(
            go_terms=[_make_go_term(go_term="viral genome replication")]
        )

        result = validate_dual_evidence(consensus, go_result)

        assert result.agreement == "dual_agree"
        assert result.go_category == "replication"
        assert result.foldseek_category == "replication"
        assert result.final_category == "replication"
        assert result.confidence_boost == 1.3

    def test_dual_disagree(self):
        """GO and Foldseek assign different categories, GO preferred."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="structural",
            consensus_score=0.7,
        )
        go_result = _make_go_result(
            go_terms=[_make_go_term(go_term="proteolysis", go_id="GO:0006508")]
        )

        result = validate_dual_evidence(consensus, go_result)

        assert result.agreement == "dual_disagree"
        assert result.go_category == "protease"
        assert result.foldseek_category == "structural"
        assert result.final_category == "protease"  # GO preferred
        assert result.confidence_boost == 1.0

    def test_go_only(self):
        """GO transfer succeeds, Foldseek returned unknown."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="unknown",
            consensus_score=0.4,
        )
        go_result = _make_go_result(
            go_terms=[_make_go_term(go_term="viral genome replication")]
        )

        result = validate_dual_evidence(consensus, go_result)

        assert result.agreement == "go_only"
        assert result.final_category == "replication"

    def test_foldseek_only_no_go_result(self):
        """No GO transfer result at all."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="structural",
            consensus_score=0.8,
        )

        result = validate_dual_evidence(consensus, None)

        assert result.agreement == "foldseek_only"
        assert result.final_category == "structural"
        assert result.confidence_boost == 1.0

    def test_foldseek_only_empty_go_terms(self):
        """GO result exists but has no terms."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="structural",
            consensus_score=0.8,
        )
        go_result = _make_go_result(go_terms=[])

        result = validate_dual_evidence(consensus, go_result)

        assert result.agreement == "foldseek_only"

    def test_cross_domain_detection(self):
        """Cross-domain flag and details are set correctly."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="unknown",
            consensus_score=0.4,
        )
        go_result = _make_go_result(
            is_cross_domain=True,
            go_terms=[
                _make_go_term(
                    go_term="proteolysis",
                    go_id="GO:0006508",
                    is_cross_domain=True,
                    donor_organism="E. coli",
                    donor_lineage="Bacteria",
                )
            ],
        )

        result = validate_dual_evidence(consensus, go_result)

        assert result.is_cross_domain
        assert result.cross_domain_details is not None
        assert "E. coli" in result.cross_domain_details

    def test_go_terms_no_category_mapping(self):
        """GO terms that don't map to any vHold category."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="structural",
            consensus_score=0.7,
        )
        # GO term that doesn't match any category mapping
        go_result = _make_go_result(
            go_terms=[
                _make_go_term(
                    go_term="some obscure process",
                    go_id="GO:9999999",
                )
            ]
        )

        result = validate_dual_evidence(consensus, go_result)

        # GO terms exist but don't map to any category, so foldseek_only
        assert result.agreement == "foldseek_only"
        assert result.final_category == "structural"


# ============================================================================
# Tests: merge_go_into_consensus
# ============================================================================


class TestMergeGoIntoConsensus:
    """Tests for merging GO transfer data into ConsensusResult."""

    def test_merge_dual_agree(self):
        """Dual agree boosts confidence and sets classification source."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="replication",
            consensus_score=0.7,
            confidence_level="medium",
        )
        go_result = _make_go_result(
            go_terms=[_make_go_term(go_term="viral genome replication")]
        )
        validation = DualValidationResult(
            agreement="dual_agree",
            go_category="replication",
            foldseek_category="replication",
            final_category="replication",
            confidence_boost=1.3,
            is_cross_domain=False,
            cross_domain_details=None,
        )

        merge_go_into_consensus(consensus, go_result, validation)

        assert consensus.classification_source == "go_transfer+foldseek"
        assert consensus.consensus_score == pytest.approx(0.91, abs=0.01)
        assert consensus.go_transfer_ri > 0
        assert consensus.dual_validation == "dual_agree"

    def test_merge_go_only(self):
        """GO-only updates category and classification source."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="unknown",
            consensus_score=0.4,
        )
        go_result = _make_go_result(
            go_terms=[_make_go_term(go_term="viral genome replication")]
        )
        validation = DualValidationResult(
            agreement="go_only",
            go_category="replication",
            foldseek_category="unknown",
            final_category="replication",
            confidence_boost=1.0,
            is_cross_domain=False,
            cross_domain_details=None,
        )

        merge_go_into_consensus(consensus, go_result, validation)

        assert consensus.functional_category == "replication"
        assert consensus.classification_source == "go_transfer"

    def test_merge_foldseek_only(self):
        """Foldseek-only doesn't change category."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="structural",
            classification_source="keywords",
            consensus_score=0.8,
        )
        validation = DualValidationResult(
            agreement="foldseek_only",
            go_category=None,
            foldseek_category="structural",
            final_category="structural",
            confidence_boost=1.0,
            is_cross_domain=False,
            cross_domain_details=None,
        )

        merge_go_into_consensus(consensus, None, validation)

        assert consensus.functional_category == "structural"
        assert consensus.classification_source == "keywords"  # Unchanged
        assert consensus.dual_validation == "foldseek_only"

    def test_merge_cross_domain(self):
        """Cross-domain details are propagated to ConsensusResult."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="unknown",
            consensus_score=0.4,
        )
        go_result = _make_go_result(
            is_cross_domain=True,
            go_terms=[
                _make_go_term(
                    go_term="proteolysis",
                    go_id="GO:0006508",
                    is_cross_domain=True,
                    donor_organism="E. coli",
                )
            ],
        )
        validation = DualValidationResult(
            agreement="go_only",
            go_category="protease",
            foldseek_category="unknown",
            final_category="protease",
            confidence_boost=1.0,
            is_cross_domain=True,
            cross_domain_details="Matched E. coli (P12345, sim=0.85)",
        )

        merge_go_into_consensus(consensus, go_result, validation)

        assert consensus.is_cross_domain
        assert "E. coli" in consensus.cross_domain_organism

    def test_merge_evidence_codes(self):
        """Evidence codes are collected from GO terms."""
        consensus = ConsensusResult(
            query_id="q1",
            query_length=200,
            functional_category="unknown",
        )
        go_result = _make_go_result(
            go_terms=[
                _make_go_term(evidence_code="IDA"),
                _make_go_term(
                    go_id="GO:0005198",
                    go_term="structural molecule activity",
                    aspect="MF",
                    evidence_code="EXP",
                ),
            ]
        )
        validation = DualValidationResult(
            agreement="go_only",
            go_category="replication",
            foldseek_category="unknown",
            final_category="replication",
            confidence_boost=1.0,
            is_cross_domain=False,
            cross_domain_details=None,
        )

        merge_go_into_consensus(consensus, go_result, validation)

        # Evidence codes should be pipe-separated and sorted
        assert "EXP" in consensus.go_evidence_codes
        assert "IDA" in consensus.go_evidence_codes
        assert "|" in consensus.go_evidence_codes
