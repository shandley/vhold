"""Tests for cross-database validation framework."""

import pytest

from vhold.validation.cross_database import (
    ValidationStatus,
    ConflictSeverity,
    AnnotationConflict,
    ValidationResult,
    DatabaseConsistency,
    ValidationReport,
    validate_annotation,
    validate_annotations_batch,
    detect_conflicts,
    calculate_database_consistency,
    generate_validation_report,
    _extract_terms,
    _calculate_term_similarity,
    _infer_category,
    _determine_conflict_severity,
)


class TestExtractTerms:
    """Tests for term extraction from descriptions."""

    def test_basic_extraction(self):
        """Test basic term extraction."""
        terms = _extract_terms("Major capsid protein")
        assert "major" in terms
        assert "capsid" in terms
        # "protein" is a stop word
        assert "protein" not in terms

    def test_empty_description(self):
        """Test empty description."""
        terms = _extract_terms("")
        assert terms == set()

    def test_none_description(self):
        """Test None description."""
        terms = _extract_terms(None)
        assert terms == set()

    def test_stop_words_removed(self):
        """Test that stop words are removed."""
        terms = _extract_terms("hypothetical viral protein domain")
        assert "hypothetical" not in terms
        assert "viral" not in terms
        assert "protein" not in terms
        assert "domain" not in terms

    def test_short_words_removed(self):
        """Test that short words (<= 3 chars) are removed."""
        terms = _extract_terms("RdRp RNA polymerase")
        assert "rna" not in terms  # 3 chars
        assert "rdrp" in terms  # 4 chars
        assert "polymerase" in terms

    def test_hyphen_handling(self):
        """Test hyphen handling."""
        terms = _extract_terms("RNA-dependent polymerase")
        assert "dependent" in terms
        assert "polymerase" in terms


class TestCalculateTermSimilarity:
    """Tests for term similarity calculation."""

    def test_identical_terms(self):
        """Test identical term sets."""
        terms = {"capsid", "major", "head"}
        sim = _calculate_term_similarity(terms, terms)
        assert sim == 1.0

    def test_no_overlap(self):
        """Test no overlapping terms."""
        terms1 = {"capsid", "major"}
        terms2 = {"polymerase", "helicase"}
        sim = _calculate_term_similarity(terms1, terms2)
        assert sim == 0.0

    def test_partial_overlap(self):
        """Test partial overlap."""
        terms1 = {"capsid", "major", "head"}
        terms2 = {"capsid", "minor", "shell"}
        sim = _calculate_term_similarity(terms1, terms2)
        # Jaccard: 1 / 5 = 0.2
        assert 0.15 <= sim <= 0.25

    def test_empty_sets(self):
        """Test empty term sets."""
        assert _calculate_term_similarity(set(), set()) == 0.0
        assert _calculate_term_similarity({"capsid"}, set()) == 0.0


class TestInferCategory:
    """Tests for category inference."""

    def test_structural_category(self):
        """Test structural category inference."""
        assert _infer_category("Major capsid protein") == "structural"
        assert _infer_category("Envelope glycoprotein") == "structural"
        assert _infer_category("Tail fiber protein") == "structural"

    def test_replication_category(self):
        """Test replication category inference."""
        assert _infer_category("RNA-dependent RNA polymerase") == "replication"
        assert _infer_category("DNA helicase") == "replication"
        assert _infer_category("Replicase complex") == "replication"

    def test_protease_category(self):
        """Test protease category inference."""
        assert _infer_category("3CL protease") == "protease"
        assert _infer_category("Viral peptidase") == "protease"

    def test_nuclease_category(self):
        """Test nuclease category inference."""
        assert _infer_category("Endonuclease VII") == "nuclease"
        assert _infer_category("Integrase protein") == "nuclease"

    def test_packaging_category(self):
        """Test packaging category inference."""
        assert _infer_category("Terminase large subunit") == "packaging"
        assert _infer_category("Portal protein") == "packaging"

    def test_lysis_category(self):
        """Test lysis category inference."""
        assert _infer_category("Endolysin") == "lysis"
        assert _infer_category("Holin protein") == "lysis"

    def test_unknown_category(self):
        """Test unknown category inference."""
        assert _infer_category("hypothetical protein") == "unknown"
        assert _infer_category("") == "unknown"


class TestDetermineConflictSeverity:
    """Tests for conflict severity determination."""

    def test_same_category_high_similarity(self):
        """Test same category with high similarity."""
        severity = _determine_conflict_severity("structural", "structural", 0.6)
        assert severity == ConflictSeverity.NEGLIGIBLE

    def test_same_category_low_similarity(self):
        """Test same category with low similarity."""
        severity = _determine_conflict_severity("structural", "structural", 0.3)
        assert severity == ConflictSeverity.MINOR

    def test_different_related_categories(self):
        """Test different but related categories."""
        severity = _determine_conflict_severity("structural", "packaging", 0.1)
        assert severity == ConflictSeverity.MAJOR

    def test_different_unrelated_categories(self):
        """Test completely different categories."""
        severity = _determine_conflict_severity("structural", "protease", 0.1)
        assert severity == ConflictSeverity.CRITICAL

    def test_unknown_category(self):
        """Test with unknown category."""
        severity = _determine_conflict_severity("structural", "unknown", 0.4)
        assert severity == ConflictSeverity.MINOR


class TestAnnotationConflict:
    """Tests for AnnotationConflict dataclass."""

    def test_creation(self):
        """Test creating a conflict."""
        conflict = AnnotationConflict(
            query_id="protein_1",
            database1="bfvd",
            database2="viro3d",
            annotation1="Major capsid protein",
            annotation2="DNA polymerase",
            category1="structural",
            category2="replication",
            severity=ConflictSeverity.CRITICAL,
            similarity_score=0.1,
        )
        assert conflict.query_id == "protein_1"
        assert conflict.severity == ConflictSeverity.CRITICAL

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conflict = AnnotationConflict(
            query_id="protein_1",
            database1="bfvd",
            database2="viro3d",
            annotation1="Capsid",
            annotation2="Polymerase",
            category1="structural",
            category2="replication",
            severity=ConflictSeverity.CRITICAL,
            similarity_score=0.15,
            resolution="Prefer bfvd",
        )
        d = conflict.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["severity"] == "critical"
        assert d["resolution"] == "Prefer bfvd"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_creation(self):
        """Test default creation."""
        result = ValidationResult(
            query_id="protein_1",
            status=ValidationStatus.UNVALIDATED,
        )
        assert result.query_id == "protein_1"
        assert result.databases_with_hits == []
        assert result.has_conflicts is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ValidationResult(
            query_id="protein_1",
            status=ValidationStatus.VALIDATED,
            databases_with_hits=["bfvd", "viro3d"],
            reliability_score=0.85,
        )
        d = result.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["status"] == "validated"
        assert d["reliability_score"] == 0.85


class TestDatabaseConsistency:
    """Tests for DatabaseConsistency dataclass."""

    def test_creation(self):
        """Test creating consistency metrics."""
        consistency = DatabaseConsistency(
            database1="bfvd",
            database2="viro3d",
            total_queries=100,
            queries_in_both=50,
            annotations_agree=40,
            annotations_partial=5,
            annotations_disagree=5,
            categories_agree=45,
            categories_disagree=5,
        )
        assert consistency.overlap_rate == 0.5
        assert consistency.agreement_rate == 0.8
        assert consistency.category_agreement_rate == 0.9

    def test_zero_queries(self):
        """Test with zero queries."""
        consistency = DatabaseConsistency(
            database1="bfvd",
            database2="viro3d",
            total_queries=0,
        )
        assert consistency.overlap_rate == 0.0
        assert consistency.agreement_rate == 0.0
        assert consistency.consistency_score == 0.0

    def test_consistency_score(self):
        """Test consistency score calculation."""
        consistency = DatabaseConsistency(
            database1="bfvd",
            database2="viro3d",
            total_queries=100,
            queries_in_both=50,
            annotations_agree=40,  # 80% agreement
            categories_agree=45,  # 90% category agreement
        )
        # Score = 0.6 * 0.8 + 0.3 * 0.9 + 0.1 * min(1.0, 0.5*2)
        # = 0.48 + 0.27 + 0.1 = 0.85
        assert 0.8 <= consistency.consistency_score <= 0.9


class TestValidateAnnotation:
    """Tests for validate_annotation function."""

    def test_no_annotations(self):
        """Test with no annotations."""
        result = validate_annotation(
            query_id="protein_1",
            annotations_by_db={},
            categories_by_db={},
        )
        assert result.status == ValidationStatus.UNVALIDATED

    def test_single_database(self):
        """Test with single database."""
        result = validate_annotation(
            query_id="protein_1",
            annotations_by_db={
                "bfvd": {"description": "Major capsid protein"}
            },
            categories_by_db={"bfvd": "structural"},
        )
        assert result.status == ValidationStatus.SINGLE_SOURCE
        assert result.reliability_score == 0.5
        assert result.consensus_annotation == "Major capsid protein"

    def test_agreeing_databases(self):
        """Test with agreeing databases."""
        result = validate_annotation(
            query_id="protein_1",
            annotations_by_db={
                "bfvd": {"description": "Major capsid protein"},
                "viro3d": {"description": "Major capsid head protein"},
            },
            categories_by_db={
                "bfvd": "structural",
                "viro3d": "structural",
            },
        )
        assert result.status in (ValidationStatus.VALIDATED, ValidationStatus.PARTIAL)
        assert result.reliability_score > 0.5

    def test_conflicting_databases(self):
        """Test with conflicting databases."""
        result = validate_annotation(
            query_id="protein_1",
            annotations_by_db={
                "bfvd": {"description": "Major capsid protein"},
                "viro3d": {"description": "DNA polymerase"},
            },
            categories_by_db={
                "bfvd": "structural",
                "viro3d": "replication",
            },
        )
        assert result.has_conflicts is True
        assert len(result.conflicts) > 0
        assert result.conflicts[0].severity in (
            ConflictSeverity.MAJOR,
            ConflictSeverity.CRITICAL,
        )

    def test_evalue_consistency(self):
        """Test e-value consistency calculation."""
        result = validate_annotation(
            query_id="protein_1",
            annotations_by_db={
                "bfvd": {"description": "Major capsid protein"},
                "viro3d": {"description": "Major capsid protein"},
            },
            categories_by_db={
                "bfvd": "structural",
                "viro3d": "structural",
            },
            evalues_by_db={
                "bfvd": 1e-30,
                "viro3d": 1e-28,
            },
        )
        assert result.evalue_consistency > 0


class TestValidateAnnotationsBatch:
    """Tests for batch validation."""

    def test_empty_batch(self):
        """Test with empty batch."""
        results = validate_annotations_batch({}, {})
        assert results == {}

    def test_multiple_proteins(self):
        """Test with multiple proteins."""
        annotations = {
            "protein_1": {
                "bfvd": {"description": "Capsid protein"},
                "viro3d": {"description": "Capsid protein"},
            },
            "protein_2": {
                "bfvd": {"description": "Polymerase"},
            },
        }
        categories = {
            "protein_1": {"bfvd": "structural", "viro3d": "structural"},
            "protein_2": {"bfvd": "replication"},
        }

        results = validate_annotations_batch(annotations, categories)

        assert len(results) == 2
        assert results["protein_1"].status != ValidationStatus.SINGLE_SOURCE
        assert results["protein_2"].status == ValidationStatus.SINGLE_SOURCE


class TestDetectConflicts:
    """Tests for conflict detection."""

    def test_no_conflicts(self):
        """Test with no conflicts."""
        results = {
            "protein_1": ValidationResult(
                query_id="protein_1",
                status=ValidationStatus.VALIDATED,
                has_conflicts=False,
            )
        }
        conflicts = detect_conflicts(results)
        assert conflicts == []

    def test_filter_by_severity(self):
        """Test filtering by severity."""
        minor_conflict = AnnotationConflict(
            query_id="protein_1",
            database1="bfvd",
            database2="viro3d",
            annotation1="Capsid A",
            annotation2="Capsid B",
            category1="structural",
            category2="structural",
            severity=ConflictSeverity.MINOR,
            similarity_score=0.4,
        )
        critical_conflict = AnnotationConflict(
            query_id="protein_2",
            database1="bfvd",
            database2="viro3d",
            annotation1="Capsid",
            annotation2="Polymerase",
            category1="structural",
            category2="replication",
            severity=ConflictSeverity.CRITICAL,
            similarity_score=0.1,
        )

        results = {
            "protein_1": ValidationResult(
                query_id="protein_1",
                status=ValidationStatus.PARTIAL,
                has_conflicts=True,
                conflicts=[minor_conflict],
            ),
            "protein_2": ValidationResult(
                query_id="protein_2",
                status=ValidationStatus.CONFLICTING,
                has_conflicts=True,
                conflicts=[critical_conflict],
            ),
        }

        # Get all conflicts
        all_conflicts = detect_conflicts(results, min_severity=ConflictSeverity.MINOR)
        assert len(all_conflicts) == 2

        # Get only major+ conflicts
        major_conflicts = detect_conflicts(results, min_severity=ConflictSeverity.MAJOR)
        assert len(major_conflicts) == 1
        assert major_conflicts[0].severity == ConflictSeverity.CRITICAL


class TestCalculateDatabaseConsistency:
    """Tests for database consistency calculation."""

    def test_two_databases(self):
        """Test consistency between two databases."""
        annotations = {
            "protein_1": {
                "bfvd": {"description": "Capsid protein"},
                "viro3d": {"description": "Capsid protein"},
            },
            "protein_2": {
                "bfvd": {"description": "Polymerase"},
            },
            "protein_3": {
                "viro3d": {"description": "Helicase"},
            },
        }
        categories = {
            "protein_1": {"bfvd": "structural", "viro3d": "structural"},
            "protein_2": {"bfvd": "replication"},
            "protein_3": {"viro3d": "replication"},
        }

        consistency = calculate_database_consistency(annotations, categories)

        assert len(consistency) == 1  # One pair
        assert consistency[0].total_queries == 3
        assert consistency[0].queries_in_both == 1
        assert consistency[0].queries_in_db1_only == 1  # protein_2
        assert consistency[0].queries_in_db2_only == 1  # protein_3

    def test_empty_annotations(self):
        """Test with empty annotations."""
        consistency = calculate_database_consistency({}, {})
        assert consistency == []


class TestGenerateValidationReport:
    """Tests for validation report generation."""

    def test_basic_report(self):
        """Test basic report generation."""
        annotations = {
            "protein_1": {
                "bfvd": {"description": "Capsid protein"},
                "viro3d": {"description": "Capsid protein"},
            },
            "protein_2": {
                "bfvd": {"description": "Polymerase"},
            },
        }
        categories = {
            "protein_1": {"bfvd": "structural", "viro3d": "structural"},
            "protein_2": {"bfvd": "replication"},
        }

        validation_results = validate_annotations_batch(annotations, categories)
        report = generate_validation_report(
            validation_results, annotations, categories
        )

        assert report.total_queries == 2
        assert report.queries_single_source == 1
        assert report.mean_reliability_score > 0

    def test_report_to_dict(self):
        """Test report conversion to dictionary."""
        report = ValidationReport(
            total_queries=10,
            queries_validated=5,
            queries_with_conflicts=2,
            mean_reliability_score=0.75,
        )

        d = report.to_dict()
        assert "summary" in d
        assert d["summary"]["total_queries"] == 10
        assert d["summary"]["validation_rate"] == 0.5

    def test_report_with_conflicts(self):
        """Test report with conflicts."""
        conflict = AnnotationConflict(
            query_id="protein_1",
            database1="bfvd",
            database2="viro3d",
            annotation1="Capsid",
            annotation2="Polymerase",
            category1="structural",
            category2="replication",
            severity=ConflictSeverity.CRITICAL,
            similarity_score=0.1,
        )

        result = ValidationResult(
            query_id="protein_1",
            status=ValidationStatus.CONFLICTING,
            has_conflicts=True,
            conflicts=[conflict],
        )

        annotations = {
            "protein_1": {
                "bfvd": {"description": "Capsid"},
                "viro3d": {"description": "Polymerase"},
            }
        }
        categories = {
            "protein_1": {"bfvd": "structural", "viro3d": "replication"}
        }

        report = generate_validation_report(
            {"protein_1": result}, annotations, categories
        )

        assert report.total_conflicts == 1
        assert report.conflicts_by_severity["critical"] == 1
        assert len(report.critical_conflicts) == 1


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ValidationStatus.VALIDATED.value == "validated"
        assert ValidationStatus.PARTIAL.value == "partial"
        assert ValidationStatus.CONFLICTING.value == "conflicting"
        assert ValidationStatus.SINGLE_SOURCE.value == "single_source"
        assert ValidationStatus.UNVALIDATED.value == "unvalidated"


class TestConflictSeverity:
    """Tests for ConflictSeverity enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ConflictSeverity.CRITICAL.value == "critical"
        assert ConflictSeverity.MAJOR.value == "major"
        assert ConflictSeverity.MINOR.value == "minor"
        assert ConflictSeverity.NEGLIGIBLE.value == "negligible"


class TestValidationReportProperties:
    """Tests for ValidationReport properties."""

    def test_validation_rate(self):
        """Test validation rate calculation."""
        report = ValidationReport(
            total_queries=100,
            queries_validated=75,
        )
        assert report.validation_rate == 0.75

    def test_conflict_rate(self):
        """Test conflict rate calculation."""
        report = ValidationReport(
            total_queries=100,
            queries_with_conflicts=20,
        )
        assert report.conflict_rate == 0.20

    def test_zero_queries(self):
        """Test with zero queries."""
        report = ValidationReport(total_queries=0)
        assert report.validation_rate == 0.0
        assert report.conflict_rate == 0.0
