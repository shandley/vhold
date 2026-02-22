"""Tests for FANTASIA-style GO term transfer."""

import numpy as np
import pytest

from vhold.databases.swissprot import GOAnnotation, SwissProtEntry
from vhold.features.go_transfer import (
    GOTransferResult,
    SwissProtReferenceDB,
    TransferredGOTerm,
    transfer_go_terms,
)


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
            GOAnnotation(
                go_id="GO:0005198",
                go_term="structural molecule activity",
                aspect="MF",
                evidence_code="EXP",
            ),
        ],
        length=1273,
    )


@pytest.fixture
def bacterial_entry():
    """A non-viral Swiss-Prot entry for cross-domain testing."""
    return SwissProtEntry(
        accession="P00720",
        name="LYS_BPT4",
        organism="Escherichia coli phage T4",
        lineage="Bacteria; Proteobacteria; Gammaproteobacteria",
        go_annotations=[
            GOAnnotation(
                go_id="GO:0003796",
                go_term="lysozyme activity",
                aspect="MF",
                evidence_code="EXP",
            ),
            GOAnnotation(
                go_id="GO:0009253",
                go_term="peptidoglycan catabolic process",
                aspect="BP",
                evidence_code="IDA",
            ),
        ],
        length=164,
    )


@pytest.fixture
def empty_go_entry():
    """A Swiss-Prot entry with no GO annotations."""
    return SwissProtEntry(
        accession="Q9999",
        name="HYPO_VIRUS",
        organism="Unknown virus",
        lineage="Viruses; unclassified",
        go_annotations=[],
        length=200,
    )


@pytest.fixture
def metadata(viral_entry, bacterial_entry, empty_go_entry):
    """Mock metadata dict."""
    return {
        viral_entry.accession: viral_entry,
        bacterial_entry.accession: bacterial_entry,
        empty_go_entry.accession: empty_go_entry,
    }


# ============================================================================
# Tests: transfer_go_terms
# ============================================================================


class TestTransferGoTerms:
    """Tests for the core GO transfer function."""

    def test_transfer_single_donor(self, metadata):
        """Single match transfers all GO terms from the donor."""
        matches = [("P0DTC2", 0.85)]
        result = transfer_go_terms(
            "query1", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert result.query_id == "query1"
        assert result.best_similarity == 0.85
        assert result.best_accession == "P0DTC2"
        assert result.has_terms
        assert len(result.go_terms) == 3  # 2 BP + 1 MF
        assert not result.is_cross_domain

    def test_transfer_multi_donor_combined_ri(self, metadata, viral_entry):
        """Combined RI = 1 - prod(1 - RI_i) across donors."""
        # Add a second viral entry with overlapping GO term
        second_entry = SwissProtEntry(
            accession="P99999",
            name="TEST_VIRAL",
            organism="Test virus",
            lineage="Viruses; test",
            go_annotations=[
                GOAnnotation(
                    go_id="GO:0039694",
                    go_term="viral process",
                    aspect="BP",
                    evidence_code="IDA",
                ),
            ],
            length=500,
        )
        meta = {**metadata, "P99999": second_entry}

        matches = [("P0DTC2", 0.85), ("P99999", 0.70)]
        result = transfer_go_terms(
            "query", matches, meta, threshold=0.5, reliability_threshold=0.3
        )

        # Find the "viral process" term
        viral_process = [t for t in result.go_terms if t.go_id == "GO:0039694"]
        assert len(viral_process) == 1

        # Combined RI = 1 - (1-0.85) * (1-0.70) = 1 - 0.045 = 0.955
        expected_ri = 1 - (1 - 0.85) * (1 - 0.70)
        assert abs(viral_process[0].reliability_index - expected_ri) < 0.001

    def test_reliability_threshold_filtering(self, metadata):
        """GO terms below reliability threshold are filtered out."""
        matches = [("P0DTC2", 0.2)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.1, reliability_threshold=0.3
        )

        # RI = 0.2 < 0.3 threshold, so no terms should pass
        assert not result.has_terms
        assert len(result.go_terms) == 0

    def test_similarity_threshold_filtering(self, metadata):
        """Matches below similarity threshold are excluded."""
        matches = [("P0DTC2", 0.3)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        # Below similarity threshold
        assert not result.has_terms

    def test_cross_domain_detection(self, metadata):
        """Non-viral donors are flagged as cross-domain."""
        matches = [("P00720", 0.75)]  # Bacterial entry
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert result.is_cross_domain
        assert result.has_terms
        for term in result.go_terms:
            assert term.is_cross_domain
            assert term.donor_organism == "Escherichia coli phage T4"

    def test_cross_domain_mixed(self, metadata):
        """Cross-domain flag set when any donor is non-viral."""
        matches = [("P0DTC2", 0.85), ("P00720", 0.65)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert result.is_cross_domain

    def test_empty_go_annotations(self, metadata):
        """Donor with no GO terms produces no transferred terms."""
        matches = [("Q9999", 0.9)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert not result.has_terms

    def test_go_aspect_separation(self, metadata):
        """GO terms correctly separated by aspect (BP, MF, CC)."""
        matches = [("P0DTC2", 0.85)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert len(result.go_bp) == 2  # viral process + viral entry
        assert len(result.go_mf) == 1  # structural molecule activity
        assert len(result.go_cc) == 0  # no CC terms

    def test_go_ids_by_aspect(self, metadata):
        """go_ids_by_aspect returns IDs sorted by RI."""
        matches = [("P0DTC2", 0.85)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        bp_ids = result.go_ids_by_aspect("BP")
        assert len(bp_ids) == 2
        assert all(gid.startswith("GO:") for gid in bp_ids)

    def test_empty_matches(self, metadata):
        """Empty match list returns empty result."""
        result = transfer_go_terms(
            "query", [], metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert not result.has_terms
        assert result.best_similarity == 0.0

    def test_unknown_accession(self, metadata):
        """Unknown accession in matches is handled gracefully."""
        matches = [("UNKNOWN_ACC", 0.9)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        # Match exists but no metadata — no terms transferred
        assert not result.has_terms

    def test_max_ri_property(self, metadata):
        """max_ri returns the highest RI across all terms."""
        matches = [("P0DTC2", 0.85)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        assert result.max_ri == 0.85  # Single donor, RI = similarity

    def test_terms_sorted_by_ri(self, metadata):
        """Transferred terms are sorted by RI descending."""
        matches = [("P0DTC2", 0.85), ("P00720", 0.65)]
        result = transfer_go_terms(
            "query", matches, metadata, threshold=0.5, reliability_threshold=0.3
        )

        ris = [t.reliability_index for t in result.go_terms]
        assert ris == sorted(ris, reverse=True)


# ============================================================================
# Tests: GOTransferResult
# ============================================================================


class TestGOTransferResult:
    """Tests for GOTransferResult dataclass."""

    def test_empty_result(self):
        """Empty result has no terms and zero similarity."""
        result = GOTransferResult(query_id="test")
        assert not result.has_terms
        assert result.max_ri == 0.0
        assert result.go_bp == []
        assert result.go_mf == []
        assert result.go_cc == []

    def test_has_terms(self):
        """has_terms returns True when GO terms exist."""
        result = GOTransferResult(
            query_id="test",
            go_terms=[
                TransferredGOTerm(
                    go_id="GO:0000001",
                    go_term="test",
                    aspect="BP",
                    evidence_code="IDA",
                    reliability_index=0.8,
                    donor_accession="P12345",
                    donor_similarity=0.8,
                    is_cross_domain=False,
                    donor_organism="test",
                    donor_lineage="Viruses",
                )
            ],
        )
        assert result.has_terms


# ============================================================================
# Tests: SwissProtReferenceDB
# ============================================================================


class TestSwissProtReferenceDB:
    """Tests for the Swiss-Prot reference DB wrapper."""

    def test_search_basic(self, tmp_path):
        """Basic search returns correct matches."""
        # Create a small mock DB
        n_ref = 5
        dim = 1024
        embeddings = np.random.randn(n_ref, dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        protein_ids = np.array([f"P{i:05d}" for i in range(n_ref)])

        # Save to .npz
        db_dir = tmp_path / "swissprot"
        db_dir.mkdir()
        np.savez(
            db_dir / "swissprot_viral_embeddings.npz",
            embeddings=embeddings,
            protein_ids=protein_ids,
        )

        # Create minimal metadata
        import json

        metadata = {}
        for pid in protein_ids:
            metadata[pid] = {
                "accession": pid,
                "name": f"TEST_{pid}",
                "organism": "Test virus",
                "lineage": "Viruses; test",
                "go_annotations": [],
                "length": 100,
            }
        with open(db_dir / "swissprot_viral_metadata.json", "w") as f:
            json.dump(metadata, f)

        # Search with a query that is identical to first reference
        query = embeddings[0:1]  # (1, 1024)

        # Load and search
        from vhold.databases.swissprot import get_swissprot_embeddings_path

        db = SwissProtReferenceDB.__new__(SwissProtReferenceDB)
        db.scope = "viral"
        db.db_dir = tmp_path
        db.embeddings = embeddings
        db.protein_ids = list(protein_ids)
        db.metadata = {}
        db._loaded = True

        results = db.search(query, k=3, threshold=0.0)
        assert len(results) == 1  # One query
        assert len(results[0]) <= 3  # Up to k matches
        assert results[0][0][0] == "P00000"  # Best match is self
        assert abs(results[0][0][1] - 1.0) < 0.01  # Similarity ~1.0

    def test_search_threshold_filtering(self):
        """Matches below threshold are excluded."""
        db = SwissProtReferenceDB.__new__(SwissProtReferenceDB)
        db.scope = "viral"
        db.db_dir = None
        db._loaded = True

        # Create orthogonal vectors (low similarity)
        dim = 1024
        e1 = np.zeros((1, dim), dtype=np.float32)
        e1[0, 0] = 1.0
        e2 = np.zeros((1, dim), dtype=np.float32)
        e2[0, 1] = 1.0

        db.embeddings = e2
        db.protein_ids = ["P00001"]
        db.metadata = {}

        results = db.search(e1, k=1, threshold=0.5)
        assert len(results) == 1
        assert len(results[0]) == 0  # Below threshold
