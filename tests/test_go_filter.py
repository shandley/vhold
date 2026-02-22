"""Tests for two-layer GO term plausibility filtering."""

import json
from unittest.mock import MagicMock, patch

import pytest

from vhold.databases.swissprot import GOAnnotation, SwissProtEntry
from vhold.features.go_filter import (
    build_viral_go_set,
    filter_cross_domain_go_terms,
)
from vhold.features.go_transfer import GOTransferResult, TransferredGOTerm


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def viral_entry():
    """A viral Swiss-Prot entry with GO annotations."""
    return SwissProtEntry(
        accession="P0DTC2",
        name="SPIKE_SARS2",
        organism="SARS-CoV-2",
        lineage="Viruses; Riboviria; Orthornavirae",
        go_annotations=[
            GOAnnotation(
                go_id="GO:0039694",
                go_term="viral process",
                aspect="BP",
                evidence_code="IDA",
            ),
            GOAnnotation(
                go_id="GO:0046718",
                go_term="viral entry into host cell",
                aspect="BP",
                evidence_code="IDA",
            ),
        ],
        length=1273,
    )


@pytest.fixture
def nonviral_entry():
    """A non-viral Swiss-Prot entry."""
    return SwissProtEntry(
        accession="P00720",
        name="LYS_ECOLI",
        organism="Escherichia coli",
        lineage="Bacteria; Proteobacteria",
        go_annotations=[
            GOAnnotation(
                go_id="GO:0003796",
                go_term="lysozyme activity",
                aspect="MF",
                evidence_code="EXP",
            ),
            GOAnnotation(
                go_id="GO:0009507",
                go_term="chloroplast",
                aspect="CC",
                evidence_code="IDA",
            ),
        ],
        length=164,
    )


@pytest.fixture
def viral_go_ids():
    """A set of GO IDs from viral Swiss-Prot entries."""
    return {"GO:0039694", "GO:0046718", "GO:0003796", "GO:0005198"}


def _make_term(
    go_id: str,
    go_term: str,
    aspect: str = "BP",
    is_cross_domain: bool = False,
    ri: float = 0.9,
    donor_organism: str = "Test virus",
    donor_lineage: str = "Viruses; test",
) -> TransferredGOTerm:
    """Helper to create a TransferredGOTerm."""
    return TransferredGOTerm(
        go_id=go_id,
        go_term=go_term,
        aspect=aspect,
        evidence_code="IDA",
        reliability_index=ri,
        donor_accession="P12345",
        donor_similarity=ri,
        is_cross_domain=is_cross_domain,
        donor_organism=donor_organism,
        donor_lineage=donor_lineage,
    )


# ============================================================================
# Tests: build_viral_go_set
# ============================================================================


class TestBuildViralGoSet:
    """Tests for build_viral_go_set."""

    def test_extracts_viral_go_ids(self, viral_entry, nonviral_entry):
        """Only GO IDs from viral entries are included."""
        metadata = {
            viral_entry.accession: viral_entry,
            nonviral_entry.accession: nonviral_entry,
        }
        result = build_viral_go_set(metadata)
        assert "GO:0039694" in result
        assert "GO:0046718" in result
        # Non-viral GO IDs should NOT be in the set
        assert "GO:0009507" not in result

    def test_empty_metadata(self):
        """Empty metadata returns empty set."""
        assert build_viral_go_set({}) == set()

    def test_no_viral_entries(self, nonviral_entry):
        """Metadata with no viral entries returns empty set."""
        metadata = {nonviral_entry.accession: nonviral_entry}
        assert build_viral_go_set(metadata) == set()

    def test_deduplicates(self):
        """Same GO ID from multiple viral entries is deduplicated."""
        entry1 = SwissProtEntry(
            accession="V1",
            name="V1",
            organism="Virus A",
            lineage="Viruses; A",
            go_annotations=[
                GOAnnotation(go_id="GO:0001", go_term="t", aspect="BP", evidence_code="IDA"),
            ],
            length=100,
        )
        entry2 = SwissProtEntry(
            accession="V2",
            name="V2",
            organism="Virus B",
            lineage="Viruses; B",
            go_annotations=[
                GOAnnotation(go_id="GO:0001", go_term="t", aspect="BP", evidence_code="IDA"),
            ],
            length=100,
        )
        result = build_viral_go_set({"V1": entry1, "V2": entry2})
        assert result == {"GO:0001"}


# ============================================================================
# Tests: filter_cross_domain_go_terms (Layer 1 - hard filter)
# ============================================================================


class TestFilterCrossDomainGoTerms:
    """Tests for the hard filter layer."""

    def test_same_domain_terms_not_filtered(self, viral_go_ids):
        """Same-domain (viral) GO terms are never filtered."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term("GO:0039694", "viral process", is_cross_domain=False),
                ],
            ),
        }
        filtered = filter_cross_domain_go_terms(results, viral_go_ids)
        assert len(filtered["q1"].go_terms) == 1

    def test_cross_domain_known_viral_kept(self, viral_go_ids):
        """Cross-domain term IN the viral GO set is kept."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:0003796", "lysozyme activity", aspect="MF",
                        is_cross_domain=True, donor_organism="E. coli",
                        donor_lineage="Bacteria",
                    ),
                ],
            ),
        }
        filtered = filter_cross_domain_go_terms(results, viral_go_ids)
        assert len(filtered["q1"].go_terms) == 1

    def test_cross_domain_unknown_removed(self, viral_go_ids):
        """Cross-domain term NOT in the viral GO set is removed."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:0009507", "chloroplast", aspect="CC",
                        is_cross_domain=True, donor_organism="Arabidopsis",
                        donor_lineage="Eukaryota; Viridiplantae",
                    ),
                ],
            ),
        }
        filtered = filter_cross_domain_go_terms(results, viral_go_ids)
        assert len(filtered["q1"].go_terms) == 0

    def test_mixed_terms_partial_filter(self, viral_go_ids):
        """Mix of same-domain, known cross-domain, and unknown cross-domain."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    # Same-domain — kept
                    _make_term("GO:0039694", "viral process", ri=0.95),
                    # Cross-domain, in viral set — kept
                    _make_term(
                        "GO:0003796", "lysozyme activity", aspect="MF",
                        is_cross_domain=True, ri=0.90,
                    ),
                    # Cross-domain, NOT in viral set — removed
                    _make_term(
                        "GO:0009507", "chloroplast", aspect="CC",
                        is_cross_domain=True, ri=0.85,
                    ),
                ],
            ),
        }
        filtered = filter_cross_domain_go_terms(results, viral_go_ids)
        assert len(filtered["q1"].go_terms) == 2
        ids = {t.go_id for t in filtered["q1"].go_terms}
        assert "GO:0039694" in ids
        assert "GO:0003796" in ids
        assert "GO:0009507" not in ids

    def test_no_viral_go_ids_skips_filter(self):
        """Empty viral GO set means no filtering (graceful skip)."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:0009507", "chloroplast", aspect="CC",
                        is_cross_domain=True,
                    ),
                ],
            ),
        }
        filtered = filter_cross_domain_go_terms(results, set())
        # No filtering — term is preserved
        assert len(filtered["q1"].go_terms) == 1

    def test_empty_results(self, viral_go_ids):
        """Empty results dict passes through."""
        filtered = filter_cross_domain_go_terms({}, viral_go_ids)
        assert filtered == {}

    def test_protein_with_no_terms(self, viral_go_ids):
        """Protein with no GO terms is unchanged."""
        results = {
            "q1": GOTransferResult(query_id="q1", go_terms=[]),
        }
        filtered = filter_cross_domain_go_terms(results, viral_go_ids)
        assert len(filtered["q1"].go_terms) == 0


# ============================================================================
# Tests: LLM validation (Layer 2)
# ============================================================================


class TestLLMValidation:
    """Tests for the LLM validation layer."""

    def test_llm_rescues_plausible_term(self, viral_go_ids):
        """LLM can rescue a flagged term by returning 'keep'."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:9999999", "novel kinase activity", aspect="MF",
                        is_cross_domain=True, donor_organism="E. coli",
                        donor_lineage="Bacteria",
                    ),
                ],
            ),
        }

        # Mock the anthropic module
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps([
            {"go_id": "GO:9999999", "verdict": "keep", "reason": "Viral kinases exist"}
        ])
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                filtered = filter_cross_domain_go_terms(
                    results, viral_go_ids, llm_validate=True
                )

        assert len(filtered["q1"].go_terms) == 1
        assert filtered["q1"].go_terms[0].go_id == "GO:9999999"

    def test_llm_confirms_rejection(self, viral_go_ids):
        """LLM confirms rejection of implausible term."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:9999999", "photosynthesis", aspect="BP",
                        is_cross_domain=True, donor_organism="Arabidopsis",
                        donor_lineage="Eukaryota; Viridiplantae",
                    ),
                ],
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps([
            {"go_id": "GO:9999999", "verdict": "reject", "reason": "Impossible for virus"}
        ])
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                filtered = filter_cross_domain_go_terms(
                    results, viral_go_ids, llm_validate=True
                )

        assert len(filtered["q1"].go_terms) == 0

    def test_llm_graceful_no_api_key(self, viral_go_ids):
        """Without API key, LLM is skipped and hard filter result stands."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:9999999", "novel function", aspect="MF",
                        is_cross_domain=True,
                    ),
                ],
            ),
        }

        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY from env
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)

            filtered = filter_cross_domain_go_terms(
                results, viral_go_ids, llm_validate=True
            )

        # Term was removed by hard filter, LLM couldn't rescue (no key)
        assert len(filtered["q1"].go_terms) == 0

    def test_llm_graceful_no_package(self, viral_go_ids):
        """Without anthropic package, LLM is skipped."""
        results = {
            "q1": GOTransferResult(
                query_id="q1",
                go_terms=[
                    _make_term(
                        "GO:9999999", "novel function", aspect="MF",
                        is_cross_domain=True,
                    ),
                ],
            ),
        }

        with patch.dict("sys.modules", {"anthropic": None}):
            filtered = filter_cross_domain_go_terms(
                results, viral_go_ids, llm_validate=True
            )

        # Hard filter result stands
        assert len(filtered["q1"].go_terms) == 0
