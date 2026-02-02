"""Cross-database validation framework for vhold."""

from vhold.validation.cross_database import (
    ValidationResult,
    AnnotationConflict,
    DatabaseConsistency,
    ValidationReport,
    validate_annotation,
    validate_annotations_batch,
    detect_conflicts,
    calculate_database_consistency,
    generate_validation_report,
)

__all__ = [
    "ValidationResult",
    "AnnotationConflict",
    "DatabaseConsistency",
    "ValidationReport",
    "validate_annotation",
    "validate_annotations_batch",
    "detect_conflicts",
    "calculate_database_consistency",
    "generate_validation_report",
]
