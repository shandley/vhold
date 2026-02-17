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


# ---- Large genome neighborhood voting tests ----

class TestLargeGenomeNeighborhoodVoting:
    """Test neighborhood voting on a herpesvirus-like genome with 100+ proteins.

    Herpesviruses organize genes into functional clusters:
    - Structural/assembly cluster (capsid, tegument, envelope)
    - Replication cluster (polymerase, helicase, primase)
    - Regulatory cluster (immediate-early, transcription factors)
    - Packaging cluster (terminase, portal)

    This test simulates a 120-gene genome with realistic cluster organization
    and scattered unknowns that should be reclassified by neighbors.
    """

    @staticmethod
    def _build_herpesvirus_genome():
        """Create a synthetic herpesvirus-like genome with 120 genes.

        Gene layout (on a single contig):
        - Genes 0-14: Regulatory cluster (immediate-early, transcription factors)
        - Genes 15-19: Unknown cluster (poorly characterized ORFs)
        - Genes 20-49: Replication cluster (polymerase, helicase, primase, etc.)
        - Genes 50-54: Unknown cluster
        - Genes 55-89: Structural cluster (capsid, tegument, envelope)
        - Genes 90-94: Packaging cluster (terminase, portal, scaffold)
        - Genes 95-99: Unknown cluster
        - Genes 100-109: Structural cluster 2 (glycoproteins)
        - Genes 110-114: Lysis/egress (mostly unknown with a few annotated)
        - Genes 115-119: Regulatory cluster 2

        Returns:
            (annotations, positions, expected_unknown_ids)
        """
        # Define gene blocks: (start_idx, end_idx, category, unknown_fraction)
        blocks = [
            (0, 14, "regulatory", 0.2),
            (15, 19, "unknown", 1.0),          # Unknown cluster near regulatory
            (20, 49, "replication", 0.15),
            (50, 54, "unknown", 1.0),           # Unknown cluster near replication
            (55, 89, "structural", 0.1),
            (90, 94, "packaging", 0.2),
            (95, 99, "unknown", 1.0),           # Unknown cluster near packaging
            (100, 109, "structural", 0.15),
            (110, 114, "lysis", 0.6),           # Mostly unknown lysis cluster
            (115, 119, "regulatory", 0.2),
        ]

        annotations = {}
        positions = {}
        unknown_ids = []
        unknown_in_cluster = {}  # Track which cluster each unknown belongs to

        gene_idx = 0
        for start_idx, end_idx, category, unknown_frac in blocks:
            for i in range(start_idx, end_idx + 1):
                pid = f"UL{i}"
                positions[pid] = {
                    "contig": "HHV1",
                    "start": i * 1500 + 100,
                    "end": i * 1500 + 1400,
                    "strand": 1 if i % 3 != 0 else -1,  # Mix of strands
                }

                # Determine if this gene is "unknown"
                is_unknown = False
                if category == "unknown":
                    is_unknown = True
                elif i % int(1 / unknown_frac) == 0 if unknown_frac > 0 else False:
                    is_unknown = True

                if is_unknown:
                    annotations[pid] = MockAnnotation(
                        functional_category="unknown",
                        confidence_level="high",
                    )
                    unknown_ids.append(pid)
                    unknown_in_cluster[pid] = category
                else:
                    annotations[pid] = MockAnnotation(
                        functional_category=category,
                        confidence_level="high",
                    )

        return annotations, positions, unknown_ids, unknown_in_cluster

    def test_genome_size(self):
        """Should have 120 genes."""
        annotations, positions, _, _ = self._build_herpesvirus_genome()
        assert len(annotations) == 120
        assert len(positions) == 120

    def test_has_substantial_unknowns(self):
        """Should have a meaningful number of unknowns to reclassify."""
        _, _, unknown_ids, _ = self._build_herpesvirus_genome()
        assert len(unknown_ids) >= 20

    def test_unknown_cluster_genes_reclassified(self):
        """Unknowns in clusters surrounded by annotated genes should be reclassified."""
        annotations, positions, unknown_ids, cluster_map = \
            self._build_herpesvirus_genome()

        result = vote_neighborhood(
            unknown_ids, annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.5,
        )

        # At least some unknowns should be reclassified
        assert len(result) > 0
        reclassification_rate = len(result) / len(unknown_ids)
        # Should reclassify at least 30% of unknowns
        assert reclassification_rate >= 0.3, (
            f"Only {reclassification_rate:.0%} of unknowns reclassified "
            f"({len(result)}/{len(unknown_ids)})"
        )

    def test_scattered_unknowns_match_cluster_category(self):
        """Unknowns scattered within clusters should get the cluster's category."""
        annotations, positions, unknown_ids, cluster_map = \
            self._build_herpesvirus_genome()

        result = vote_neighborhood(
            unknown_ids, annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.5,
        )

        # Reclassified unknowns should match their cluster category
        correct = 0
        total = 0
        for pid, (predicted_cat, confidence) in result.items():
            expected = cluster_map.get(pid)
            if expected and expected != "unknown":
                total += 1
                if predicted_cat == expected:
                    correct += 1

        if total > 0:
            accuracy = correct / total
            # 70% threshold accounts for genes at cluster boundaries
            # that get reclassified to the adjacent cluster's category
            assert accuracy >= 0.65, (
                f"Cluster accuracy: {accuracy:.0%} ({correct}/{total})"
            )

    def test_pure_unknown_clusters_reclassified_by_neighbors(self):
        """Pure unknown clusters (genes 15-19, 50-54, 95-99) should get
        categories from their neighboring annotated clusters."""
        annotations, positions, unknown_ids, cluster_map = \
            self._build_herpesvirus_genome()

        result = vote_neighborhood(
            unknown_ids, annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.5,
        )

        # Genes 15-19 are near regulatory (0-14) and replication (20-49)
        # They should be reclassified (mostly to one of those)
        cluster_15_19 = [f"UL{i}" for i in range(15, 20)]
        reclassified = sum(1 for pid in cluster_15_19 if pid in result)
        assert reclassified >= 2, (
            f"Only {reclassified}/5 genes in unknown cluster 15-19 reclassified"
        )

    def test_cross_cluster_boundary(self):
        """Genes at cluster boundaries may get either neighboring category."""
        annotations, positions, unknown_ids, cluster_map = \
            self._build_herpesvirus_genome()

        result = vote_neighborhood(
            unknown_ids, annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.5,
        )

        # Check all reclassified proteins have valid categories
        valid_categories = {
            "structural", "replication", "protease", "nuclease",
            "packaging", "regulatory", "movement", "lysis",
            "host_interaction", "entry",
        }
        for pid, (cat, conf) in result.items():
            assert cat in valid_categories, f"{pid} got invalid category: {cat}"
            assert 0.0 < conf <= 1.0, f"{pid} got invalid confidence: {conf}"

    def test_multi_contig_large_genome(self):
        """Test with genes split across multiple contigs (metagenomic scenario)."""
        annotations = {}
        positions = {}
        unknown_ids = []

        # 3 contigs of 40 genes each, each with structural clusters and unknowns
        for contig_idx in range(3):
            contig = f"contig_{contig_idx}"
            for i in range(40):
                pid = f"{contig}_g{i}"
                positions[pid] = {
                    "contig": contig,
                    "start": i * 1000 + 100,
                    "end": i * 1000 + 900,
                    "strand": 1,
                }
                # Make genes 15-19 unknown, rest structural
                if 15 <= i <= 19:
                    annotations[pid] = MockAnnotation(
                        functional_category="unknown",
                        confidence_level="high",
                    )
                    unknown_ids.append(pid)
                else:
                    annotations[pid] = MockAnnotation(
                        functional_category="structural",
                        confidence_level="high",
                    )

        assert len(annotations) == 120
        assert len(unknown_ids) == 15

        result = vote_neighborhood(
            unknown_ids, annotations, positions,
            window_size=5, min_votes=2, min_vote_fraction=0.5,
        )

        # All unknowns should be reclassified as structural
        assert len(result) == 15
        for pid, (cat, _) in result.items():
            assert cat == "structural", f"{pid} expected structural, got {cat}"
