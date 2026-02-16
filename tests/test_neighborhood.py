"""Tests for neighborhood voting functional classification."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vhold.results.neighborhood import (
    build_gene_index,
    get_neighbors,
    vote_neighborhood,
)


# ---- Mock ConsensusResult-like objects ----

@dataclass
class MockAnnotation:
    """Minimal mock for ConsensusResult."""
    functional_category: str = "unknown"
    confidence_level: str = "medium"
    classification_source: str = "keywords"


# ---- Test build_gene_index ----

class TestBuildGeneIndex:
    """Tests for grouping proteins by contig."""

    def test_single_contig(self):
        """Should group all genes on one contig sorted by start."""
        positions = {
            "g3": {"contig": "c1", "start": 900, "end": 1200, "strand": 1},
            "g1": {"contig": "c1", "start": 100, "end": 400, "strand": 1},
            "g2": {"contig": "c1", "start": 500, "end": 800, "strand": -1},
        }
        index = build_gene_index(positions)
        assert "c1" in index
        assert len(index["c1"]) == 3
        # Should be sorted by start
        ids = [g[0] for g in index["c1"]]
        assert ids == ["g1", "g2", "g3"]

    def test_multiple_contigs(self):
        """Should separate genes by contig."""
        positions = {
            "g1": {"contig": "c1", "start": 100, "end": 400, "strand": 1},
            "g2": {"contig": "c2", "start": 100, "end": 400, "strand": 1},
            "g3": {"contig": "c1", "start": 500, "end": 800, "strand": 1},
        }
        index = build_gene_index(positions)
        assert len(index) == 2
        assert len(index["c1"]) == 2
        assert len(index["c2"]) == 1

    def test_empty_positions(self):
        """Should return empty dict for no positions."""
        assert build_gene_index({}) == {}

    def test_missing_contig(self):
        """Should skip proteins without contig."""
        positions = {
            "g1": {"contig": None, "start": 100, "end": 400, "strand": 1},
        }
        assert build_gene_index(positions) == {}


# ---- Test get_neighbors ----

class TestGetNeighbors:
    """Tests for finding neighboring genes."""

    def _setup_linear_contig(self, n=10):
        """Create n genes on a single contig."""
        positions = {}
        for i in range(n):
            pid = f"g{i}"
            positions[pid] = {
                "contig": "c1",
                "start": i * 300 + 100,
                "end": i * 300 + 400,
                "strand": 1,
            }
        index = build_gene_index(positions)
        return positions, index

    def test_middle_gene(self):
        """Gene in the middle should have neighbors on both sides."""
        positions, index = self._setup_linear_contig(10)
        neighbors = get_neighbors("g5", index, positions, window_size=3)
        neighbor_ids = {nid for nid, _ in neighbors}
        assert "g2" in neighbor_ids
        assert "g3" in neighbor_ids
        assert "g4" in neighbor_ids
        assert "g6" in neighbor_ids
        assert "g7" in neighbor_ids
        assert "g8" in neighbor_ids
        assert len(neighbors) == 6

    def test_first_gene(self):
        """First gene should only have neighbors on one side."""
        positions, index = self._setup_linear_contig(10)
        neighbors = get_neighbors("g0", index, positions, window_size=3)
        neighbor_ids = {nid for nid, _ in neighbors}
        assert "g1" in neighbor_ids
        assert "g2" in neighbor_ids
        assert "g3" in neighbor_ids
        assert len(neighbors) == 3

    def test_last_gene(self):
        """Last gene should only have neighbors on one side."""
        positions, index = self._setup_linear_contig(10)
        neighbors = get_neighbors("g9", index, positions, window_size=3)
        neighbor_ids = {nid for nid, _ in neighbors}
        assert "g6" in neighbor_ids
        assert "g7" in neighbor_ids
        assert "g8" in neighbor_ids
        assert len(neighbors) == 3

    def test_rank_distance(self):
        """Rank distances should be correct."""
        positions, index = self._setup_linear_contig(10)
        neighbors = get_neighbors("g5", index, positions, window_size=2)
        distances = {nid: dist for nid, dist in neighbors}
        assert distances["g4"] == 1
        assert distances["g3"] == 2
        assert distances["g6"] == 1
        assert distances["g7"] == 2

    def test_unknown_protein(self):
        """Should return empty for unknown protein."""
        positions, index = self._setup_linear_contig(5)
        assert get_neighbors("unknown", index, positions) == []

    def test_no_position_data(self):
        """Should return empty when protein has no position."""
        positions = {"g1": {"contig": None}}
        index = build_gene_index(positions)
        assert get_neighbors("g1", index, positions) == []


# ---- Test vote_neighborhood ----

class TestVoteNeighborhood:
    """Tests for the voting algorithm."""

    def _make_annotations(self, categories: dict[str, str]) -> dict[str, MockAnnotation]:
        """Create mock annotations from category dict."""
        return {pid: MockAnnotation(functional_category=cat) for pid, cat in categories.items()}

    def _make_positions(self, n: int, contig: str = "c1") -> dict[str, dict]:
        """Create n genes in a line on one contig."""
        return {
            f"g{i}": {
                "contig": contig,
                "start": i * 300 + 100,
                "end": i * 300 + 400,
                "strand": 1,
            }
            for i in range(n)
        }

    def test_clear_majority_reclassifies(self):
        """Unknown surrounded by structural proteins → structural."""
        annotations = self._make_annotations({
            "g0": "structural",
            "g1": "structural",
            "g2": "unknown",  # target
            "g3": "structural",
            "g4": "replication",
        })
        positions = self._make_positions(5)

        result = vote_neighborhood(
            ["g2"], annotations, positions, window_size=5,
        )
        assert "g2" in result
        assert result["g2"][0] == "structural"

    def test_no_majority_stays_unknown(self):
        """Equal mix of categories → stays unknown."""
        annotations = self._make_annotations({
            "g0": "structural",
            "g1": "replication",
            "g2": "unknown",  # target
            "g3": "lysis",
            "g4": "packaging",
        })
        positions = self._make_positions(5)

        result = vote_neighborhood(
            ["g2"], annotations, positions,
            min_vote_fraction=0.5,
        )
        # No clear majority → should not reclassify
        assert "g2" not in result

    def test_min_votes_threshold(self):
        """Too few annotated neighbors → stays unknown."""
        annotations = self._make_annotations({
            "g0": "unknown",
            "g1": "structural",  # only 1 vote
            "g2": "unknown",  # target
            "g3": "unknown",
            "g4": "unknown",
        })
        positions = self._make_positions(5)

        result = vote_neighborhood(
            ["g2"], annotations, positions, min_votes=2,
        )
        assert "g2" not in result

    def test_cross_contig_isolation(self):
        """Genes on different contigs should not vote for each other."""
        annotations = self._make_annotations({
            "g0": "structural",
            "g1": "structural",
            "g2": "unknown",  # target, on different contig
        })
        positions = {
            "g0": {"contig": "c1", "start": 100, "end": 400, "strand": 1},
            "g1": {"contig": "c1", "start": 500, "end": 800, "strand": 1},
            "g2": {"contig": "c2", "start": 100, "end": 400, "strand": 1},
        }

        result = vote_neighborhood(["g2"], annotations, positions)
        assert "g2" not in result

    def test_distance_weighting(self):
        """Closer neighbors should have more weight."""
        # structural at distance 1 on both sides
        # replication at distance 2 and 3
        annotations = self._make_annotations({
            "g0": "replication",  # distance 3
            "g1": "replication",  # distance 2
            "g2": "structural",  # distance 1
            "g3": "unknown",     # target
            "g4": "structural",  # distance 1
        })
        positions = self._make_positions(5)

        result = vote_neighborhood(
            ["g3"], annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.3,
        )
        # structural: 2 votes at distance 1 → weight = 2 * 0.7 * (1/2) = 0.7
        # replication: 2 votes at distance 2,3 → weight = 0.7*(1/3) + 0.7*(1/4) = 0.408
        # structural should win due to proximity
        assert "g3" in result
        assert result["g3"][0] == "structural"

    def test_empty_unknown_ids(self):
        """Should return empty for no unknowns."""
        result = vote_neighborhood([], {}, {})
        assert result == {}

    def test_empty_positions(self):
        """Should return empty when no positions available."""
        annotations = self._make_annotations({"g0": "unknown"})
        result = vote_neighborhood(["g0"], annotations, {})
        assert result == {}

    def test_multiple_unknowns(self):
        """Should process multiple unknown proteins."""
        annotations = self._make_annotations({
            "g0": "structural",
            "g1": "structural",
            "g2": "unknown",
            "g3": "structural",
            "g4": "unknown",
            "g5": "structural",
            "g6": "structural",
        })
        positions = self._make_positions(7)

        result = vote_neighborhood(
            ["g2", "g4"], annotations, positions,
        )
        assert "g2" in result
        assert "g4" in result
        assert result["g2"][0] == "structural"
        assert result["g4"][0] == "structural"

    def test_vote_confidence(self):
        """Vote confidence should be between 0 and 1."""
        annotations = self._make_annotations({
            "g0": "structural",
            "g1": "structural",
            "g2": "unknown",
            "g3": "structural",
        })
        positions = self._make_positions(4)

        result = vote_neighborhood(["g2"], annotations, positions)
        assert "g2" in result
        _, confidence = result["g2"]
        assert 0.0 < confidence <= 1.0
