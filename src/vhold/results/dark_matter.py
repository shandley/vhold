"""Dark matter analysis for viral proteins with unknown or weak annotations.

Dark matter proteins are viral proteins that:
1. Have no detectable homologs in structural databases
2. Have hits but only to uncharacterized/hypothetical proteins
3. Have only weak hits (low confidence scores)

This module provides analysis and reporting for these proteins, which are
particularly interesting for eukaryotic virus research where many proteins
remain functionally uncharacterized.
"""

from dataclasses import dataclass, field
from typing import Optional

from vhold.results.consensus import ConsensusResult, CONFIDENCE_THRESHOLDS
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# Thresholds for dark matter classification
WEAK_HIT_EVALUE_THRESHOLD = 1e-5  # E-values worse than this are "weak"
WEAK_HIT_IDENTITY_THRESHOLD = 0.3  # Identity below this is "weak"
WEAK_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLDS["low"]  # Below "low" confidence


@dataclass
class DarkMatterProtein:
    """A protein classified as dark matter."""

    query_id: str
    query_length: int
    category: str  # "no_hits", "unknown_function", "weak_hits"
    reason: str  # Human-readable explanation

    # Hit info if available
    best_evalue: Optional[float] = None
    best_identity: Optional[float] = None
    confidence_level: str = "none"
    consensus_score: float = 0.0

    # Description if any
    description: str = "hypothetical protein"

    # Source databases that had hits
    databases_with_hits: list[str] = field(default_factory=list)

    # Disorder prediction (propagated from ConsensusResult)
    disorder_class: str | None = None
    disorder_fraction: float | None = None


@dataclass
class DarkMatterReport:
    """Summary report of dark matter proteins."""

    # Counts by category
    total_proteins: int = 0
    total_dark_matter: int = 0
    no_hits: int = 0
    unknown_function: int = 0
    weak_hits: int = 0

    # Rates
    dark_matter_rate: float = 0.0

    # By confidence level (for proteins with hits)
    by_confidence: dict[str, int] = field(default_factory=dict)

    # Length distribution of dark matter proteins
    length_stats: dict = field(default_factory=dict)

    # Individual proteins
    proteins: list[DarkMatterProtein] = field(default_factory=list)

    # Metagenomic characterizations (optional, added by enhance_dark_matter_with_metagenomics)
    metagenomic_characterizations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        result = {
            "summary": {
                "total_proteins": self.total_proteins,
                "total_dark_matter": self.total_dark_matter,
                "dark_matter_rate": round(self.dark_matter_rate, 4),
                "by_category": {
                    "no_hits": self.no_hits,
                    "unknown_function": self.unknown_function,
                    "weak_hits": self.weak_hits,
                },
                "by_confidence": self.by_confidence,
                "length_stats": self.length_stats,
                "by_disorder_class": {
                    "highly_disordered": sum(
                        1 for p in self.proteins if p.disorder_class == "highly_disordered"
                    ),
                    "partially_disordered": sum(
                        1 for p in self.proteins if p.disorder_class == "partially_disordered"
                    ),
                    "ordered": sum(
                        1 for p in self.proteins if p.disorder_class == "ordered"
                    ),
                    "unknown": sum(
                        1 for p in self.proteins if p.disorder_class is None
                    ),
                },
            },
            "proteins": [
                {
                    "query_id": p.query_id,
                    "query_length": p.query_length,
                    "category": p.category,
                    "reason": p.reason,
                    "best_evalue": p.best_evalue,
                    "best_identity": p.best_identity,
                    "confidence_level": p.confidence_level,
                    "description": p.description,
                    "databases_with_hits": p.databases_with_hits,
                    "disorder_class": p.disorder_class,
                    "disorder_fraction": p.disorder_fraction,
                }
                for p in self.proteins
            ],
        }

        # Add metagenomic characterizations if available
        if self.metagenomic_characterizations:
            result["metagenomic_characterizations"] = [
                c.to_dict() for c in self.metagenomic_characterizations
            ]
            # Add summary stats
            high_priority = sum(
                1 for c in self.metagenomic_characterizations if c.priority_score >= 0.5
            )
            conserved = sum(
                1 for c in self.metagenomic_characterizations
                if c.metagenomic_context and c.metagenomic_context.is_conserved_in_metagenomes
            )
            result["summary"]["metagenomic_enhancement"] = {
                "high_priority_count": high_priority,
                "conserved_in_metagenomes": conserved,
            }

        return result


def classify_dark_matter(result: ConsensusResult) -> Optional[DarkMatterProtein]:
    """Classify a protein as dark matter if applicable.

    Args:
        result: ConsensusResult from annotation pipeline

    Returns:
        DarkMatterProtein if classified as dark matter, None otherwise
    """
    # Category 1: No hits at all
    if not result.is_annotated:
        return DarkMatterProtein(
            query_id=result.query_id,
            query_length=result.query_length,
            category="no_hits",
            reason="No structural homologs detected in any database",
            confidence_level="none",
            disorder_class=result.disorder_class,
            disorder_fraction=result.disorder_fraction,
        )

    # Category 2: Unknown functional category
    if result.functional_category == "unknown":
        # Check if it's a weak hit or just unknown function
        if result.primary_hit:
            evalue = result.primary_hit.evalue
            identity = result.primary_hit.fident

            # Is it also a weak hit?
            is_weak = (
                evalue > WEAK_HIT_EVALUE_THRESHOLD
                or identity < WEAK_HIT_IDENTITY_THRESHOLD
            )

            databases_with_hits = list(result.hits_by_db.keys())

            if is_weak:
                return DarkMatterProtein(
                    query_id=result.query_id,
                    query_length=result.query_length,
                    category="weak_hits",
                    reason=f"Only weak structural homologs (e-value: {evalue:.2e}, identity: {identity:.1%})",
                    best_evalue=evalue,
                    best_identity=identity,
                    confidence_level=result.confidence_level,
                    consensus_score=result.consensus_score,
                    description=result.description,
                    databases_with_hits=databases_with_hits,
                    disorder_class=result.disorder_class,
                    disorder_fraction=result.disorder_fraction,
                )
            else:
                return DarkMatterProtein(
                    query_id=result.query_id,
                    query_length=result.query_length,
                    category="unknown_function",
                    reason="Structural homologs found but function is uncharacterized",
                    best_evalue=evalue,
                    best_identity=identity,
                    confidence_level=result.confidence_level,
                    consensus_score=result.consensus_score,
                    description=result.description,
                    databases_with_hits=databases_with_hits,
                    disorder_class=result.disorder_class,
                    disorder_fraction=result.disorder_fraction,
                )

    # Category 3: Has hits but very weak confidence
    if result.consensus_score < WEAK_CONFIDENCE_THRESHOLD:
        if result.primary_hit:
            return DarkMatterProtein(
                query_id=result.query_id,
                query_length=result.query_length,
                category="weak_hits",
                reason=f"Low confidence annotation (score: {result.consensus_score:.3f})",
                best_evalue=result.primary_hit.evalue,
                best_identity=result.primary_hit.fident,
                confidence_level=result.confidence_level,
                consensus_score=result.consensus_score,
                description=result.description,
                databases_with_hits=list(result.hits_by_db.keys()),
                disorder_class=result.disorder_class,
                disorder_fraction=result.disorder_fraction,
            )

    # Not dark matter
    return None


def analyze_dark_matter(
    annotations: dict[str, ConsensusResult],
) -> DarkMatterReport:
    """Analyze annotations for dark matter proteins.

    Args:
        annotations: Dict mapping query ID to ConsensusResult

    Returns:
        DarkMatterReport with analysis results
    """
    report = DarkMatterReport(total_proteins=len(annotations))

    dark_proteins: list[DarkMatterProtein] = []
    lengths: list[int] = []

    for query_id, result in annotations.items():
        dm = classify_dark_matter(result)
        if dm:
            dark_proteins.append(dm)
            lengths.append(dm.query_length)

            # Count by category
            if dm.category == "no_hits":
                report.no_hits += 1
            elif dm.category == "unknown_function":
                report.unknown_function += 1
            elif dm.category == "weak_hits":
                report.weak_hits += 1

            # Count by confidence
            conf = dm.confidence_level
            report.by_confidence[conf] = report.by_confidence.get(conf, 0) + 1

    report.total_dark_matter = len(dark_proteins)
    report.proteins = dark_proteins

    if report.total_proteins > 0:
        report.dark_matter_rate = report.total_dark_matter / report.total_proteins

    # Length statistics for dark matter proteins
    if lengths:
        sorted_lengths = sorted(lengths)
        report.length_stats = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "median": sorted_lengths[len(sorted_lengths) // 2],
        }

    logger.info(
        f"Dark matter analysis: {report.total_dark_matter}/{report.total_proteins} proteins "
        f"({report.dark_matter_rate:.1%}) classified as dark matter"
    )
    logger.info(
        f"  - No hits: {report.no_hits}, Unknown function: {report.unknown_function}, "
        f"Weak hits: {report.weak_hits}"
    )

    return report


def get_dark_matter_summary(report: DarkMatterReport) -> dict:
    """Get a summary dict suitable for including in the main summary JSON.

    Args:
        report: DarkMatterReport from analysis

    Returns:
        Dict with summary statistics
    """
    summary = {
        "total_dark_matter": report.total_dark_matter,
        "dark_matter_rate": round(report.dark_matter_rate, 4),
        "by_category": {
            "no_hits": report.no_hits,
            "unknown_function": report.unknown_function,
            "weak_hits": report.weak_hits,
        },
        "by_confidence": report.by_confidence,
        "length_stats": report.length_stats,
    }

    # Add disorder breakdown if any protein has disorder data
    if any(p.disorder_class is not None for p in report.proteins):
        summary["by_disorder_class"] = {
            "highly_disordered": sum(
                1 for p in report.proteins if p.disorder_class == "highly_disordered"
            ),
            "partially_disordered": sum(
                1 for p in report.proteins if p.disorder_class == "partially_disordered"
            ),
            "ordered": sum(
                1 for p in report.proteins if p.disorder_class == "ordered"
            ),
            "unknown": sum(
                1 for p in report.proteins if p.disorder_class is None
            ),
        }

    return summary


def enhance_dark_matter_with_metagenomics(
    report: DarkMatterReport,
    sequences: dict[str, str] | None = None,
    search_serratus: bool = True,
) -> "DarkMatterReport":
    """Enhance dark matter report with metagenomic context.

    This function searches metagenomic databases (Serratus, Logan) to
    provide additional context for dark matter proteins.

    Args:
        report: DarkMatterReport from analyze_dark_matter
        sequences: Optional dict mapping query_id to amino acid sequence
        search_serratus: Whether to search Serratus (requires network)

    Returns:
        Enhanced DarkMatterReport with metagenomic context
    """
    if not report.proteins:
        return report

    try:
        from vhold.databases.metagenomics import characterize_dark_matter_batch

        characterizations = characterize_dark_matter_batch(
            report.proteins,
            sequences=sequences,
            search_serratus=search_serratus,
            search_logan=False,  # Logan requires external setup
        )

        # Store characterizations in report
        report.metagenomic_characterizations = characterizations

        # Count high-priority proteins
        high_priority = sum(1 for c in characterizations if c.priority_score >= 0.5)
        conserved_count = sum(
            1 for c in characterizations
            if c.metagenomic_context and c.metagenomic_context.is_conserved_in_metagenomes
        )

        logger.info(
            f"Metagenomic enhancement: {high_priority} high-priority proteins, "
            f"{conserved_count} conserved in metagenomes"
        )

    except ImportError as e:
        logger.warning(f"Metagenomic enhancement not available: {e}")
    except Exception as e:
        logger.warning(f"Metagenomic enhancement failed: {e}")

    return report
