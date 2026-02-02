"""Cross-database validation framework for vhold annotations.

This module provides tools for validating annotations by comparing results
across multiple databases (BFVD, Viro3D, ESM Atlas) to assess consistency,
detect conflicts, and score reliability.

Key features:
1. Annotation validation - Assess quality based on cross-database support
2. Conflict detection - Identify disagreements between databases
3. Consistency metrics - Measure annotation agreement across databases
4. Reliability scoring - Score confidence based on multiple evidence sources
5. Validation reports - Generate comprehensive quality reports
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import statistics

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationStatus(Enum):
    """Validation status for an annotation."""

    VALIDATED = "validated"  # Consistent across multiple databases
    PARTIAL = "partial"  # Some agreement, minor differences
    CONFLICTING = "conflicting"  # Databases disagree
    SINGLE_SOURCE = "single_source"  # Only one database has annotation
    UNVALIDATED = "unvalidated"  # No validation possible


class ConflictSeverity(Enum):
    """Severity level of annotation conflicts."""

    CRITICAL = "critical"  # Completely different functions predicted
    MAJOR = "major"  # Different functional categories
    MINOR = "minor"  # Same category but different details
    NEGLIGIBLE = "negligible"  # Terminology differences only


@dataclass
class AnnotationConflict:
    """Represents a conflict between database annotations."""

    query_id: str
    database1: str
    database2: str
    annotation1: str
    annotation2: str
    category1: str
    category2: str
    severity: ConflictSeverity
    similarity_score: float  # 0-1, how similar the annotations are
    resolution: Optional[str] = None  # Suggested resolution
    details: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "database1": self.database1,
            "database2": self.database2,
            "annotation1": self.annotation1,
            "annotation2": self.annotation2,
            "category1": self.category1,
            "category2": self.category2,
            "severity": self.severity.value,
            "similarity_score": round(self.similarity_score, 3),
            "resolution": self.resolution,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Validation result for a single protein annotation."""

    query_id: str
    status: ValidationStatus = ValidationStatus.UNVALIDATED

    # Database coverage
    databases_with_hits: list[str] = field(default_factory=list)
    databases_agreeing: list[str] = field(default_factory=list)

    # Validation metrics
    cross_database_score: float = 0.0  # 0-1, based on agreement
    reliability_score: float = 0.0  # 0-1, overall reliability
    evidence_count: int = 0  # Number of supporting pieces of evidence

    # Conflict information
    has_conflicts: bool = False
    conflicts: list[AnnotationConflict] = field(default_factory=list)

    # Annotation details
    consensus_annotation: str = ""
    consensus_category: str = "unknown"
    annotation_by_db: dict[str, str] = field(default_factory=dict)
    category_by_db: dict[str, str] = field(default_factory=dict)

    # E-value agreement
    evalue_consistency: float = 1.0  # 0-1, how consistent e-values are

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "status": self.status.value,
            "databases_with_hits": self.databases_with_hits,
            "databases_agreeing": self.databases_agreeing,
            "cross_database_score": round(self.cross_database_score, 3),
            "reliability_score": round(self.reliability_score, 3),
            "evidence_count": self.evidence_count,
            "has_conflicts": self.has_conflicts,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "consensus_annotation": self.consensus_annotation,
            "consensus_category": self.consensus_category,
            "evalue_consistency": round(self.evalue_consistency, 3),
        }


@dataclass
class DatabaseConsistency:
    """Consistency metrics for a pair of databases."""

    database1: str
    database2: str

    # Overlap statistics
    total_queries: int = 0
    queries_in_both: int = 0  # Queries with hits in both
    queries_in_db1_only: int = 0
    queries_in_db2_only: int = 0

    # Agreement statistics
    annotations_agree: int = 0
    annotations_partial: int = 0
    annotations_disagree: int = 0

    # Category agreement
    categories_agree: int = 0
    categories_disagree: int = 0

    # Computed metrics
    @property
    def overlap_rate(self) -> float:
        """Fraction of queries with hits in both databases."""
        if self.total_queries == 0:
            return 0.0
        return self.queries_in_both / self.total_queries

    @property
    def agreement_rate(self) -> float:
        """Fraction of overlapping queries that agree."""
        if self.queries_in_both == 0:
            return 0.0
        return self.annotations_agree / self.queries_in_both

    @property
    def category_agreement_rate(self) -> float:
        """Fraction of overlapping queries with same category."""
        if self.queries_in_both == 0:
            return 0.0
        return self.categories_agree / self.queries_in_both

    @property
    def consistency_score(self) -> float:
        """Overall consistency score (0-1)."""
        if self.queries_in_both == 0:
            return 0.0
        # Weight: 60% agreement, 30% category match, 10% overlap
        return (
            0.6 * self.agreement_rate
            + 0.3 * self.category_agreement_rate
            + 0.1 * min(1.0, self.overlap_rate * 2)  # Cap overlap contribution
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "database1": self.database1,
            "database2": self.database2,
            "total_queries": self.total_queries,
            "queries_in_both": self.queries_in_both,
            "queries_in_db1_only": self.queries_in_db1_only,
            "queries_in_db2_only": self.queries_in_db2_only,
            "annotations_agree": self.annotations_agree,
            "annotations_partial": self.annotations_partial,
            "annotations_disagree": self.annotations_disagree,
            "categories_agree": self.categories_agree,
            "categories_disagree": self.categories_disagree,
            "overlap_rate": round(self.overlap_rate, 3),
            "agreement_rate": round(self.agreement_rate, 3),
            "category_agreement_rate": round(self.category_agreement_rate, 3),
            "consistency_score": round(self.consistency_score, 3),
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""

    # Summary statistics
    total_queries: int = 0
    queries_validated: int = 0
    queries_with_conflicts: int = 0
    queries_single_source: int = 0

    # Validation status distribution
    status_distribution: dict[str, int] = field(default_factory=dict)

    # Reliability scores
    mean_reliability_score: float = 0.0
    median_reliability_score: float = 0.0
    reliability_distribution: dict[str, int] = field(default_factory=dict)

    # Database consistency
    database_consistency: list[DatabaseConsistency] = field(default_factory=list)

    # Conflicts
    total_conflicts: int = 0
    conflicts_by_severity: dict[str, int] = field(default_factory=dict)
    critical_conflicts: list[AnnotationConflict] = field(default_factory=list)

    # Per-query results (optional, may be large)
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)

    @property
    def validation_rate(self) -> float:
        """Fraction of queries that could be validated."""
        if self.total_queries == 0:
            return 0.0
        return self.queries_validated / self.total_queries

    @property
    def conflict_rate(self) -> float:
        """Fraction of queries with conflicts."""
        if self.total_queries == 0:
            return 0.0
        return self.queries_with_conflicts / self.total_queries

    def to_dict(self, include_per_query: bool = False) -> dict:
        """Convert to dictionary.

        Args:
            include_per_query: Whether to include per-query results
        """
        result = {
            "summary": {
                "total_queries": self.total_queries,
                "queries_validated": self.queries_validated,
                "queries_with_conflicts": self.queries_with_conflicts,
                "queries_single_source": self.queries_single_source,
                "validation_rate": round(self.validation_rate, 3),
                "conflict_rate": round(self.conflict_rate, 3),
            },
            "status_distribution": self.status_distribution,
            "reliability_scores": {
                "mean": round(self.mean_reliability_score, 3),
                "median": round(self.median_reliability_score, 3),
                "distribution": self.reliability_distribution,
            },
            "database_consistency": [dc.to_dict() for dc in self.database_consistency],
            "conflicts": {
                "total": self.total_conflicts,
                "by_severity": self.conflicts_by_severity,
                "critical": [c.to_dict() for c in self.critical_conflicts[:10]],
            },
        }

        if include_per_query:
            result["per_query"] = {
                qid: vr.to_dict() for qid, vr in self.validation_results.items()
            }

        return result


# Stop words for annotation comparison
ANNOTATION_STOP_WORDS = {
    "protein",
    "viral",
    "virus",
    "hypothetical",
    "putative",
    "like",
    "domain",
    "containing",
    "family",
    "related",
    "probable",
    "possible",
    "uncharacterized",
    "unknown",
}

# Functional category keywords for conflict detection
CATEGORY_KEYWORDS = {
    "structural": {"capsid", "coat", "envelope", "spike", "matrix", "tail", "fiber"},
    "replication": {"polymerase", "replicase", "helicase", "primase", "rdrp"},
    "protease": {"protease", "peptidase", "maturase", "cleavage"},
    "nuclease": {"nuclease", "endonuclease", "exonuclease", "integrase"},
    "packaging": {"terminase", "packaging", "scaffold", "portal"},
    "regulatory": {"transcription", "repressor", "activator", "regulator"},
    "lysis": {"lysin", "holin", "endolysin", "spanin"},
}


def _extract_terms(description: str) -> set[str]:
    """Extract significant terms from a description.

    Args:
        description: Annotation description

    Returns:
        Set of significant terms (lowercase)
    """
    if not description:
        return set()

    # Normalize and split
    text = description.lower().replace("-", " ").replace("_", " ")
    words = text.split()

    # Filter stop words and short words
    return {w for w in words if len(w) > 3 and w not in ANNOTATION_STOP_WORDS}


def _calculate_term_similarity(terms1: set[str], terms2: set[str]) -> float:
    """Calculate Jaccard similarity between term sets.

    Args:
        terms1: First set of terms
        terms2: Second set of terms

    Returns:
        Similarity score (0-1)
    """
    if not terms1 or not terms2:
        return 0.0

    intersection = len(terms1 & terms2)
    union = len(terms1 | terms2)

    if union == 0:
        return 0.0

    return intersection / union


def _infer_category(description: str) -> str:
    """Infer functional category from description.

    Args:
        description: Annotation description

    Returns:
        Functional category string
    """
    if not description:
        return "unknown"

    text = description.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category

    return "unknown"


def _determine_conflict_severity(
    category1: str,
    category2: str,
    similarity: float,
) -> ConflictSeverity:
    """Determine the severity of an annotation conflict.

    Args:
        category1: First functional category
        category2: Second functional category
        similarity: Term similarity score

    Returns:
        Conflict severity level
    """
    # Same category is at most minor
    if category1 == category2:
        if similarity >= 0.5:
            return ConflictSeverity.NEGLIGIBLE
        return ConflictSeverity.MINOR

    # Unknown category means we can't assess properly
    if category1 == "unknown" or category2 == "unknown":
        if similarity >= 0.3:
            return ConflictSeverity.MINOR
        return ConflictSeverity.MAJOR

    # Different categories - check for related categories
    related_categories = [
        {"structural", "packaging"},  # Related to virion assembly
        {"replication", "nuclease"},  # Related to genome processing
        {"protease", "regulatory"},  # Processing functions
    ]

    for related_set in related_categories:
        if category1 in related_set and category2 in related_set:
            return ConflictSeverity.MAJOR

    # Completely different functions
    return ConflictSeverity.CRITICAL


def validate_annotation(
    query_id: str,
    annotations_by_db: dict[str, dict],
    categories_by_db: dict[str, str],
    evalues_by_db: dict[str, float] | None = None,
) -> ValidationResult:
    """Validate a single protein annotation across databases.

    Args:
        query_id: Protein identifier
        annotations_by_db: Dict mapping database name to annotation dict
        categories_by_db: Dict mapping database name to functional category
        evalues_by_db: Dict mapping database name to e-value (optional)

    Returns:
        ValidationResult with validation metrics
    """
    result = ValidationResult(query_id=query_id)

    # Filter to databases with actual annotations
    valid_dbs = {
        db: ann
        for db, ann in annotations_by_db.items()
        if ann and ann.get("description")
    }

    result.databases_with_hits = list(valid_dbs.keys())

    # Store annotations
    for db, ann in valid_dbs.items():
        result.annotation_by_db[db] = ann.get("description", "")
        result.category_by_db[db] = categories_by_db.get(db, "unknown")

    # Handle different cases
    if len(valid_dbs) == 0:
        result.status = ValidationStatus.UNVALIDATED
        return result

    if len(valid_dbs) == 1:
        result.status = ValidationStatus.SINGLE_SOURCE
        db = list(valid_dbs.keys())[0]
        result.consensus_annotation = valid_dbs[db].get("description", "")
        result.consensus_category = categories_by_db.get(db, "unknown")
        result.reliability_score = 0.5  # Single source = moderate reliability
        result.evidence_count = 1
        return result

    # Multiple databases - check for agreement
    databases = list(valid_dbs.keys())
    descriptions = [valid_dbs[db].get("description", "") for db in databases]
    term_sets = [_extract_terms(desc) for desc in descriptions]
    categories = [categories_by_db.get(db, "unknown") for db in databases]

    # Calculate pairwise similarities
    similarities = []
    conflicts = []

    for i in range(len(databases)):
        for j in range(i + 1, len(databases)):
            sim = _calculate_term_similarity(term_sets[i], term_sets[j])
            similarities.append(sim)

            # Check for conflicts
            if sim < 0.2 or categories[i] != categories[j]:
                severity = _determine_conflict_severity(
                    categories[i], categories[j], sim
                )

                conflict = AnnotationConflict(
                    query_id=query_id,
                    database1=databases[i],
                    database2=databases[j],
                    annotation1=descriptions[i],
                    annotation2=descriptions[j],
                    category1=categories[i],
                    category2=categories[j],
                    severity=severity,
                    similarity_score=sim,
                )

                # Suggest resolution
                if evalues_by_db:
                    ev1 = evalues_by_db.get(databases[i], 1)
                    ev2 = evalues_by_db.get(databases[j], 1)
                    if ev1 < ev2 / 10:
                        conflict.resolution = f"Prefer {databases[i]} (better e-value)"
                    elif ev2 < ev1 / 10:
                        conflict.resolution = f"Prefer {databases[j]} (better e-value)"
                    else:
                        conflict.resolution = "Manual review recommended"

                conflicts.append(conflict)

    # Determine validation status
    avg_similarity = statistics.mean(similarities) if similarities else 0

    if avg_similarity >= 0.5:
        result.status = ValidationStatus.VALIDATED
        result.databases_agreeing = databases
    elif avg_similarity >= 0.2:
        result.status = ValidationStatus.PARTIAL
        # Find which databases agree
        for i, db in enumerate(databases):
            agrees_with_any = any(
                _calculate_term_similarity(term_sets[i], term_sets[j]) >= 0.3
                for j in range(len(databases))
                if i != j
            )
            if agrees_with_any:
                result.databases_agreeing.append(db)
    else:
        result.status = ValidationStatus.CONFLICTING

    # Set conflicts
    result.has_conflicts = len(conflicts) > 0
    result.conflicts = conflicts

    # Calculate cross-database score
    result.cross_database_score = avg_similarity

    # Calculate e-value consistency
    if evalues_by_db and len(evalues_by_db) > 1:
        log_evalues = [
            -1 * (10 ** -50 if ev == 0 else ev)
            for ev in evalues_by_db.values()
            if ev is not None
        ]
        if len(log_evalues) > 1:
            # Normalize by range
            ev_range = max(log_evalues) - min(log_evalues)
            if ev_range > 0:
                result.evalue_consistency = max(0, 1 - (ev_range / 50))
            else:
                result.evalue_consistency = 1.0

    # Calculate reliability score
    # Factors: number of databases, agreement, category consistency
    db_count_factor = min(1.0, len(valid_dbs) / 3)  # Cap at 3 databases
    agreement_factor = avg_similarity
    category_factor = 1.0 if len(set(categories)) == 1 else 0.5

    result.reliability_score = (
        0.3 * db_count_factor + 0.5 * agreement_factor + 0.2 * category_factor
    )

    result.evidence_count = len(valid_dbs)

    # Set consensus annotation (from database with best support)
    # Prefer database with most agreement
    if result.databases_agreeing:
        best_db = result.databases_agreeing[0]
    else:
        best_db = databases[0]

    result.consensus_annotation = valid_dbs[best_db].get("description", "")
    result.consensus_category = categories_by_db.get(best_db, "unknown")

    return result


def validate_annotations_batch(
    annotations: dict[str, dict[str, dict]],
    categories: dict[str, dict[str, str]],
    evalues: dict[str, dict[str, float]] | None = None,
) -> dict[str, ValidationResult]:
    """Validate annotations for multiple proteins.

    Args:
        annotations: Dict mapping query_id -> {db -> annotation_dict}
        categories: Dict mapping query_id -> {db -> category}
        evalues: Dict mapping query_id -> {db -> evalue} (optional)

    Returns:
        Dict mapping query_id to ValidationResult
    """
    results = {}

    for query_id in annotations:
        ann_by_db = annotations.get(query_id, {})
        cat_by_db = categories.get(query_id, {})
        ev_by_db = evalues.get(query_id, {}) if evalues else None

        results[query_id] = validate_annotation(
            query_id=query_id,
            annotations_by_db=ann_by_db,
            categories_by_db=cat_by_db,
            evalues_by_db=ev_by_db,
        )

    logger.info(f"Validated {len(results)} annotations")
    return results


def detect_conflicts(
    validation_results: dict[str, ValidationResult],
    min_severity: ConflictSeverity = ConflictSeverity.MINOR,
) -> list[AnnotationConflict]:
    """Extract conflicts from validation results.

    Args:
        validation_results: Dict of ValidationResult objects
        min_severity: Minimum severity to include

    Returns:
        List of AnnotationConflict objects
    """
    severity_order = [
        ConflictSeverity.NEGLIGIBLE,
        ConflictSeverity.MINOR,
        ConflictSeverity.MAJOR,
        ConflictSeverity.CRITICAL,
    ]
    min_idx = severity_order.index(min_severity)

    conflicts = []
    for result in validation_results.values():
        for conflict in result.conflicts:
            if severity_order.index(conflict.severity) >= min_idx:
                conflicts.append(conflict)

    # Sort by severity (critical first)
    conflicts.sort(key=lambda c: -severity_order.index(c.severity))

    return conflicts


def calculate_database_consistency(
    annotations: dict[str, dict[str, dict]],
    categories: dict[str, dict[str, str]],
) -> list[DatabaseConsistency]:
    """Calculate pairwise consistency metrics between databases.

    Args:
        annotations: Dict mapping query_id -> {db -> annotation_dict}
        categories: Dict mapping query_id -> {db -> category}

    Returns:
        List of DatabaseConsistency objects for each database pair
    """
    # Find all databases
    all_dbs = set()
    for ann_by_db in annotations.values():
        all_dbs.update(ann_by_db.keys())

    databases = sorted(all_dbs)
    consistency_results = []

    # Calculate pairwise consistency
    for i, db1 in enumerate(databases):
        for db2 in databases[i + 1 :]:
            consistency = DatabaseConsistency(
                database1=db1,
                database2=db2,
                total_queries=len(annotations),
            )

            for query_id, ann_by_db in annotations.items():
                has_db1 = db1 in ann_by_db and ann_by_db[db1].get("description")
                has_db2 = db2 in ann_by_db and ann_by_db[db2].get("description")

                if has_db1 and has_db2:
                    consistency.queries_in_both += 1

                    # Check annotation agreement
                    desc1 = ann_by_db[db1].get("description", "")
                    desc2 = ann_by_db[db2].get("description", "")
                    terms1 = _extract_terms(desc1)
                    terms2 = _extract_terms(desc2)
                    sim = _calculate_term_similarity(terms1, terms2)

                    if sim >= 0.5:
                        consistency.annotations_agree += 1
                    elif sim >= 0.2:
                        consistency.annotations_partial += 1
                    else:
                        consistency.annotations_disagree += 1

                    # Check category agreement
                    cat_by_db = categories.get(query_id, {})
                    cat1 = cat_by_db.get(db1, "unknown")
                    cat2 = cat_by_db.get(db2, "unknown")

                    if cat1 == cat2:
                        consistency.categories_agree += 1
                    else:
                        consistency.categories_disagree += 1

                elif has_db1:
                    consistency.queries_in_db1_only += 1
                elif has_db2:
                    consistency.queries_in_db2_only += 1

            consistency_results.append(consistency)

    return consistency_results


def generate_validation_report(
    validation_results: dict[str, ValidationResult],
    annotations: dict[str, dict[str, dict]],
    categories: dict[str, dict[str, str]],
) -> ValidationReport:
    """Generate a comprehensive validation report.

    Args:
        validation_results: Dict of ValidationResult objects
        annotations: Dict mapping query_id -> {db -> annotation_dict}
        categories: Dict mapping query_id -> {db -> category}

    Returns:
        ValidationReport with summary statistics
    """
    report = ValidationReport(
        total_queries=len(validation_results),
        validation_results=validation_results,
    )

    # Calculate statistics
    reliability_scores = []

    for result in validation_results.values():
        # Status distribution
        status_key = result.status.value
        report.status_distribution[status_key] = (
            report.status_distribution.get(status_key, 0) + 1
        )

        # Count validated/conflicts
        if result.status == ValidationStatus.VALIDATED:
            report.queries_validated += 1
        elif result.status == ValidationStatus.PARTIAL:
            report.queries_validated += 1
        elif result.status == ValidationStatus.SINGLE_SOURCE:
            report.queries_single_source += 1

        if result.has_conflicts:
            report.queries_with_conflicts += 1

        reliability_scores.append(result.reliability_score)

        # Conflict statistics
        for conflict in result.conflicts:
            report.total_conflicts += 1
            sev_key = conflict.severity.value
            report.conflicts_by_severity[sev_key] = (
                report.conflicts_by_severity.get(sev_key, 0) + 1
            )

            if conflict.severity == ConflictSeverity.CRITICAL:
                report.critical_conflicts.append(conflict)

    # Reliability score statistics
    if reliability_scores:
        report.mean_reliability_score = statistics.mean(reliability_scores)
        report.median_reliability_score = statistics.median(reliability_scores)

        # Distribution (binned)
        for score in reliability_scores:
            if score >= 0.8:
                bin_key = "high (>=0.8)"
            elif score >= 0.5:
                bin_key = "medium (0.5-0.8)"
            elif score >= 0.3:
                bin_key = "low (0.3-0.5)"
            else:
                bin_key = "very_low (<0.3)"
            report.reliability_distribution[bin_key] = (
                report.reliability_distribution.get(bin_key, 0) + 1
            )

    # Database consistency
    report.database_consistency = calculate_database_consistency(
        annotations, categories
    )

    logger.info(
        f"Validation report: {report.queries_validated}/{report.total_queries} validated, "
        f"{report.queries_with_conflicts} with conflicts"
    )

    return report


# Export all classes and functions
__all__ = [
    "ValidationStatus",
    "ConflictSeverity",
    "AnnotationConflict",
    "ValidationResult",
    "DatabaseConsistency",
    "ValidationReport",
    "validate_annotation",
    "validate_annotations_batch",
    "detect_conflicts",
    "calculate_database_consistency",
    "generate_validation_report",
]
