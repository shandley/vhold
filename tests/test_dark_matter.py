"""Tests for the dark matter analysis module."""

import pytest

from vhold.results.dark_matter import (
    DarkMatterProtein,
    DarkMatterReport,
    classify_dark_matter,
    analyze_dark_matter,
    get_dark_matter_summary,
    WEAK_HIT_EVALUE_THRESHOLD,
    WEAK_HIT_IDENTITY_THRESHOLD,
)
from vhold.results.consensus import ConsensusResult
from vhold.results.parser import FoldseekHit


def make_hit(
    query: str = "test_query",
    target: str = "test_target",
    evalue: float = 1e-20,
    fident: float = 0.8,
    qcov: float = 0.9,
    source_db: str = "viro3d",
) -> FoldseekHit:
    """Create a FoldseekHit for testing."""
    return FoldseekHit(
        query=query,
        target=target,
        fident=fident,
        alnlen=100,
        mismatch=10,
        gapopen=2,
        qstart=1,
        qend=100,
        tstart=1,
        tend=100,
        evalue=evalue,
        bits=200.0,
        qlen=100,
        tlen=100,
        qcov=qcov,
        tcov=0.9,
        source_db=source_db,
    )


def make_consensus(
    query_id: str = "test_query",
    query_length: int = 100,
    has_hit: bool = True,
    functional_category: str = "structural",
    evalue: float = 1e-20,
    fident: float = 0.8,
    confidence_level: str = "high",
    consensus_score: float = 0.9,
    source_db: str = "viro3d",
) -> ConsensusResult:
    """Create a ConsensusResult for testing."""
    result = ConsensusResult(
        query_id=query_id,
        query_length=query_length,
        functional_category=functional_category,
        confidence_level=confidence_level,
        consensus_score=consensus_score,
    )

    if has_hit:
        hit = make_hit(query=query_id, evalue=evalue, fident=fident, source_db=source_db)
        result.primary_hit = hit
        result.primary_source = source_db
        result.primary_annotation = {"description": "Test protein"}
        result.hits_by_db = {source_db: [hit]}

    return result


class TestClassifyDarkMatter:
    """Tests for classify_dark_matter function."""

    def test_no_hits_is_dark_matter(self):
        """Proteins with no hits are classified as dark matter."""
        result = make_consensus(has_hit=False)
        dm = classify_dark_matter(result)

        assert dm is not None
        assert dm.category == "no_hits"
        assert dm.confidence_level == "none"
        assert "No structural homologs" in dm.reason

    def test_unknown_function_is_dark_matter(self):
        """Proteins with unknown function are dark matter."""
        result = make_consensus(functional_category="unknown")
        dm = classify_dark_matter(result)

        assert dm is not None
        assert dm.category == "unknown_function"
        assert "uncharacterized" in dm.reason.lower()

    def test_weak_evalue_is_dark_matter(self):
        """Proteins with weak e-values are dark matter."""
        result = make_consensus(
            functional_category="unknown",
            evalue=1e-3,  # Worse than threshold
            fident=0.8,
        )
        dm = classify_dark_matter(result)

        assert dm is not None
        assert dm.category == "weak_hits"
        assert "weak" in dm.reason.lower()

    def test_weak_identity_is_dark_matter(self):
        """Proteins with weak identity are dark matter."""
        result = make_consensus(
            functional_category="unknown",
            evalue=1e-30,
            fident=0.2,  # Below threshold
        )
        dm = classify_dark_matter(result)

        assert dm is not None
        assert dm.category == "weak_hits"

    def test_low_confidence_is_dark_matter(self):
        """Proteins with very low confidence are dark matter."""
        result = make_consensus(
            functional_category="structural",
            consensus_score=0.1,  # Very low
        )
        dm = classify_dark_matter(result)

        assert dm is not None
        assert dm.category == "weak_hits"

    def test_good_annotation_not_dark_matter(self):
        """Well-annotated proteins are not dark matter."""
        result = make_consensus(
            functional_category="structural",
            evalue=1e-50,
            fident=0.95,
            confidence_level="high",
            consensus_score=0.95,
        )
        dm = classify_dark_matter(result)

        assert dm is None

    def test_medium_confidence_known_function_not_dark_matter(self):
        """Medium confidence with known function is not dark matter."""
        result = make_consensus(
            functional_category="protease",
            consensus_score=0.5,
            confidence_level="medium",
        )
        dm = classify_dark_matter(result)

        assert dm is None


class TestAnalyzeDarkMatter:
    """Tests for analyze_dark_matter function."""

    def test_empty_annotations(self):
        """Empty annotations returns empty report."""
        report = analyze_dark_matter({})

        assert report.total_proteins == 0
        assert report.total_dark_matter == 0
        assert report.dark_matter_rate == 0.0
        assert len(report.proteins) == 0

    def test_all_dark_matter(self):
        """All proteins are dark matter."""
        annotations = {
            "prot1": make_consensus(query_id="prot1", has_hit=False),
            "prot2": make_consensus(query_id="prot2", functional_category="unknown"),
        }
        report = analyze_dark_matter(annotations)

        assert report.total_proteins == 2
        assert report.total_dark_matter == 2
        assert report.dark_matter_rate == 1.0
        assert report.no_hits == 1
        assert report.unknown_function == 1

    def test_no_dark_matter(self):
        """No proteins are dark matter."""
        annotations = {
            "prot1": make_consensus(query_id="prot1", functional_category="structural"),
            "prot2": make_consensus(query_id="prot2", functional_category="protease"),
        }
        report = analyze_dark_matter(annotations)

        assert report.total_proteins == 2
        assert report.total_dark_matter == 0
        assert report.dark_matter_rate == 0.0

    def test_mixed_annotations(self):
        """Mix of dark matter and annotated proteins."""
        annotations = {
            "known1": make_consensus(query_id="known1", functional_category="structural"),
            "unknown1": make_consensus(query_id="unknown1", has_hit=False),
            "unknown2": make_consensus(query_id="unknown2", functional_category="unknown"),
            "weak1": make_consensus(
                query_id="weak1",
                functional_category="unknown",
                evalue=1e-3,
            ),
        }
        report = analyze_dark_matter(annotations)

        assert report.total_proteins == 4
        assert report.total_dark_matter == 3
        assert report.dark_matter_rate == 0.75
        assert report.no_hits == 1
        assert report.unknown_function == 1
        assert report.weak_hits == 1

    def test_length_stats(self):
        """Length statistics are calculated correctly."""
        annotations = {
            "prot1": make_consensus(query_id="prot1", query_length=100, has_hit=False),
            "prot2": make_consensus(query_id="prot2", query_length=200, has_hit=False),
            "prot3": make_consensus(query_id="prot3", query_length=300, has_hit=False),
        }
        report = analyze_dark_matter(annotations)

        assert report.length_stats["min"] == 100
        assert report.length_stats["max"] == 300
        assert report.length_stats["mean"] == 200.0
        assert report.length_stats["median"] == 200

    def test_confidence_distribution(self):
        """Confidence levels are tracked."""
        annotations = {
            "prot1": make_consensus(
                query_id="prot1",
                functional_category="unknown",
                confidence_level="high",
            ),
            "prot2": make_consensus(
                query_id="prot2",
                functional_category="unknown",
                confidence_level="low",
            ),
            "prot3": make_consensus(query_id="prot3", has_hit=False),
        }
        report = analyze_dark_matter(annotations)

        assert report.by_confidence.get("high", 0) == 1
        assert report.by_confidence.get("low", 0) == 1
        assert report.by_confidence.get("none", 0) == 1


class TestDarkMatterReport:
    """Tests for DarkMatterReport class."""

    def test_to_dict(self):
        """Report converts to dict correctly."""
        report = DarkMatterReport(
            total_proteins=10,
            total_dark_matter=3,
            no_hits=1,
            unknown_function=1,
            weak_hits=1,
            dark_matter_rate=0.3,
            by_confidence={"none": 1, "low": 2},
            length_stats={"min": 100, "max": 500, "mean": 250.0, "median": 200},
        )
        report.proteins = [
            DarkMatterProtein(
                query_id="prot1",
                query_length=100,
                category="no_hits",
                reason="No hits",
            )
        ]

        d = report.to_dict()

        assert d["summary"]["total_proteins"] == 10
        assert d["summary"]["total_dark_matter"] == 3
        assert d["summary"]["dark_matter_rate"] == 0.3
        assert d["summary"]["by_category"]["no_hits"] == 1
        assert d["summary"]["by_confidence"]["none"] == 1
        assert len(d["proteins"]) == 1
        assert d["proteins"][0]["query_id"] == "prot1"


class TestGetDarkMatterSummary:
    """Tests for get_dark_matter_summary function."""

    def test_summary_structure(self):
        """Summary has expected structure."""
        report = DarkMatterReport(
            total_proteins=10,
            total_dark_matter=3,
            no_hits=1,
            unknown_function=1,
            weak_hits=1,
            dark_matter_rate=0.3,
        )

        summary = get_dark_matter_summary(report)

        assert "total_dark_matter" in summary
        assert "dark_matter_rate" in summary
        assert "by_category" in summary
        assert "no_hits" in summary["by_category"]
        assert summary["total_dark_matter"] == 3
        assert summary["dark_matter_rate"] == 0.3


class TestDarkMatterProtein:
    """Tests for DarkMatterProtein dataclass."""

    def test_defaults(self):
        """Default values are set correctly."""
        dm = DarkMatterProtein(
            query_id="test",
            query_length=100,
            category="no_hits",
            reason="Test reason",
        )

        assert dm.best_evalue is None
        assert dm.best_identity is None
        assert dm.confidence_level == "none"
        assert dm.consensus_score == 0.0
        assert dm.description == "hypothetical protein"
        assert dm.databases_with_hits == []

    def test_with_values(self):
        """Values are stored correctly."""
        dm = DarkMatterProtein(
            query_id="test",
            query_length=150,
            category="weak_hits",
            reason="Weak hit",
            best_evalue=1e-5,
            best_identity=0.3,
            confidence_level="low",
            consensus_score=0.25,
            description="putative protein",
            databases_with_hits=["viro3d", "bfvd"],
        )

        assert dm.query_length == 150
        assert dm.category == "weak_hits"
        assert dm.best_evalue == 1e-5
        assert dm.best_identity == 0.3
        assert dm.confidence_level == "low"
        assert "viro3d" in dm.databases_with_hits
