"""Multi-database consensus scoring for vhold."""

import math
from dataclasses import dataclass, field
from typing import Optional

from vhold.results.categories import classify_protein
from vhold.results.parser import FoldseekHit
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# Database weights (Viro3D has better annotations)
DATABASE_WEIGHTS = {
    "viro3d": 1.2,
    "bfvd": 1.0,
}

# Thresholds for confidence levels
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}


@dataclass
class HitScore:
    """Scored hit from a single database."""

    hit: FoldseekHit
    quality_score: float
    weighted_score: float
    annotation: dict = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Consensus annotation from multiple databases."""

    query_id: str
    query_length: int

    # Primary annotation (best overall)
    primary_hit: Optional[FoldseekHit] = None
    primary_annotation: dict = field(default_factory=dict)
    primary_source: str = ""

    # Secondary annotation (from other database if available)
    secondary_hit: Optional[FoldseekHit] = None
    secondary_annotation: dict = field(default_factory=dict)
    secondary_source: str = ""

    # Consensus metrics
    consensus_score: float = 0.0
    confidence_level: str = "none"
    agreement: str = "none"  # "agree", "partial", "disagree", "single", "none"

    # Functional classification
    functional_category: str = "unknown"

    # All hits by database
    hits_by_db: dict = field(default_factory=dict)

    @property
    def is_annotated(self) -> bool:
        """Check if any annotation exists."""
        return self.primary_hit is not None

    @property
    def description(self) -> str:
        """Get best description."""
        if self.primary_annotation:
            return self.primary_annotation.get("description", "hypothetical protein")
        return "hypothetical protein"

    @property
    def has_consensus(self) -> bool:
        """Check if we have agreement from multiple databases."""
        return self.agreement in ("agree", "partial")

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        result = {
            "query_id": self.query_id,
            "query_length": self.query_length,
            "description": self.description,
            "confidence_level": self.confidence_level,
            "consensus_score": round(self.consensus_score, 3),
            "agreement": self.agreement,
            "functional_category": self.functional_category,
            "primary_source": self.primary_source,
            "primary_target": self.primary_hit.target if self.primary_hit else "",
            "primary_evalue": self.primary_hit.evalue if self.primary_hit else "",
            "primary_identity": self.primary_hit.fident if self.primary_hit else "",
            "primary_coverage": self.primary_hit.qcov if self.primary_hit else "",
        }

        # Add secondary hit info if available
        if self.secondary_hit:
            result["secondary_source"] = self.secondary_source
            result["secondary_target"] = self.secondary_hit.target
            result["secondary_evalue"] = self.secondary_hit.evalue
            result["secondary_identity"] = self.secondary_hit.fident

        # Add annotation fields
        for key, value in self.primary_annotation.items():
            if key not in result and key != "source":
                result[key] = value

        # Add hit counts per database
        for db, hits in self.hits_by_db.items():
            result[f"{db}_hits"] = len(hits)

        return result


def calculate_hit_quality(hit: FoldseekHit) -> float:
    """Calculate quality score for a single hit.

    Combines e-value, identity, and coverage into a 0-1 score.

    Args:
        hit: FoldseekHit object

    Returns:
        Quality score between 0 and 1
    """
    # E-value contribution: -log10(evalue) normalized
    # e-value of 1e-50 -> score of 1.0
    # e-value of 1e-10 -> score of 0.2
    # e-value of 1 -> score of 0
    if hit.evalue > 0:
        evalue_score = min(1.0, -math.log10(hit.evalue) / 50)
    else:
        evalue_score = 1.0

    # Identity is already 0-1 (as fraction)
    identity_score = hit.fident

    # Coverage is already 0-1
    coverage_score = hit.qcov

    # Weighted combination
    # E-value is most important, then identity, then coverage
    quality = (0.5 * evalue_score + 0.3 * identity_score + 0.2 * coverage_score)

    return quality


def score_hit(hit: FoldseekHit, annotation: dict) -> HitScore:
    """Score a hit with its annotation.

    Args:
        hit: FoldseekHit object
        annotation: Annotation dict from database

    Returns:
        HitScore object
    """
    quality = calculate_hit_quality(hit)
    db_weight = DATABASE_WEIGHTS.get(hit.source_db, 1.0)

    # Bonus for having a real description (not just UniProt ID)
    desc = annotation.get("description", "")
    if desc and "UniProt:" not in desc and "hypothetical" not in desc.lower():
        db_weight *= 1.1  # 10% bonus for real functional annotation

    weighted = quality * db_weight

    return HitScore(
        hit=hit,
        quality_score=quality,
        weighted_score=weighted,
        annotation=annotation,
    )


def check_annotation_agreement(ann1: dict, ann2: dict) -> str:
    """Check if two annotations agree.

    Args:
        ann1: First annotation dict
        ann2: Second annotation dict

    Returns:
        Agreement level: "agree", "partial", "disagree"
    """
    desc1 = ann1.get("description", "").lower()
    desc2 = ann2.get("description", "").lower()

    # Skip comparison if either is just a UniProt ID
    if "uniprot:" in desc1 or "uniprot:" in desc2:
        return "partial"

    # Check for key term overlap
    # Extract significant words (>3 chars, not common words)
    stop_words = {"protein", "viral", "virus", "hypothetical", "putative", "like", "domain"}

    def extract_terms(desc: str) -> set:
        words = desc.replace("-", " ").replace("_", " ").split()
        return {w for w in words if len(w) > 3 and w not in stop_words}

    terms1 = extract_terms(desc1)
    terms2 = extract_terms(desc2)

    if not terms1 or not terms2:
        return "partial"

    # Calculate overlap
    overlap = len(terms1 & terms2)
    union = len(terms1 | terms2)

    if union == 0:
        return "partial"

    similarity = overlap / union

    if similarity >= 0.5:
        return "agree"
    elif similarity >= 0.2:
        return "partial"
    else:
        return "disagree"


def build_consensus(
    query_id: str,
    query_length: int,
    scored_hits: dict[str, list[HitScore]],
) -> ConsensusResult:
    """Build consensus annotation from scored hits.

    Args:
        query_id: Query protein ID
        query_length: Query protein length
        scored_hits: Dict mapping database name to list of HitScore

    Returns:
        ConsensusResult object
    """
    result = ConsensusResult(
        query_id=query_id,
        query_length=query_length,
        hits_by_db={db: [s.hit for s in scores] for db, scores in scored_hits.items()},
    )

    # Get best hit from each database
    best_by_db: dict[str, HitScore] = {}
    for db, scores in scored_hits.items():
        if scores:
            best_by_db[db] = max(scores, key=lambda s: s.weighted_score)

    if not best_by_db:
        # No hits at all
        result.confidence_level = "none"
        result.agreement = "none"
        return result

    # Sort databases by best hit score
    sorted_dbs = sorted(best_by_db.keys(), key=lambda db: best_by_db[db].weighted_score, reverse=True)

    # Primary is the best overall
    primary_db = sorted_dbs[0]
    primary_scored = best_by_db[primary_db]

    result.primary_hit = primary_scored.hit
    result.primary_annotation = primary_scored.annotation
    result.primary_source = primary_db

    # Classify protein into functional category
    result.functional_category = classify_protein(
        result.description,
        result.primary_annotation.get("gene"),
    )

    # Check for secondary database
    if len(sorted_dbs) > 1:
        secondary_db = sorted_dbs[1]
        secondary_scored = best_by_db[secondary_db]

        result.secondary_hit = secondary_scored.hit
        result.secondary_annotation = secondary_scored.annotation
        result.secondary_source = secondary_db

        # Check agreement
        result.agreement = check_annotation_agreement(
            primary_scored.annotation,
            secondary_scored.annotation,
        )

        # Calculate consensus score with agreement bonus
        base_score = primary_scored.weighted_score
        if result.agreement == "agree":
            # Strong agreement bonus
            result.consensus_score = min(1.0, base_score * 1.3)
        elif result.agreement == "partial":
            # Moderate bonus
            result.consensus_score = min(1.0, base_score * 1.15)
        else:
            # No bonus for disagreement
            result.consensus_score = base_score
    else:
        # Single database
        result.agreement = "single"
        result.consensus_score = primary_scored.weighted_score

    # Determine confidence level
    if result.consensus_score >= CONFIDENCE_THRESHOLDS["high"]:
        result.confidence_level = "high"
    elif result.consensus_score >= CONFIDENCE_THRESHOLDS["medium"]:
        result.confidence_level = "medium"
    elif result.consensus_score >= CONFIDENCE_THRESHOLDS["low"]:
        result.confidence_level = "low"
    else:
        result.confidence_level = "very_low"

    # Boost confidence if we have agreement
    if result.agreement == "agree" and result.confidence_level in ("medium", "low"):
        # Upgrade one level
        levels = ["very_low", "low", "medium", "high"]
        current_idx = levels.index(result.confidence_level)
        if current_idx < len(levels) - 1:
            result.confidence_level = levels[current_idx + 1]

    return result


def calculate_consensus(
    hits_by_query: dict[str, list[FoldseekHit]],
    annotations_by_hit: dict[str, dict],
    query_lengths: dict[str, int],
) -> dict[str, ConsensusResult]:
    """Calculate consensus annotations for all queries.

    Args:
        hits_by_query: Dict mapping query ID to list of hits
        annotations_by_hit: Dict mapping (target_id, source_db) to annotation
        query_lengths: Dict mapping query ID to sequence length

    Returns:
        Dict mapping query ID to ConsensusResult
    """
    results = {}

    # Process each query
    for query_id, length in query_lengths.items():
        hits = hits_by_query.get(query_id, [])

        # Group hits by database and score them
        scored_by_db: dict[str, list[HitScore]] = {}
        for hit in hits:
            db = hit.source_db
            if db not in scored_by_db:
                scored_by_db[db] = []

            # Get annotation for this hit
            ann_key = (hit.target, db)
            annotation = annotations_by_hit.get(ann_key, {})

            scored = score_hit(hit, annotation)
            scored_by_db[db].append(scored)

        # Build consensus
        results[query_id] = build_consensus(query_id, length, scored_by_db)

    # Log summary
    annotated = sum(1 for r in results.values() if r.is_annotated)
    with_consensus = sum(1 for r in results.values() if r.has_consensus)
    logger.info(f"Consensus: {annotated}/{len(results)} annotated, {with_consensus} with multi-DB agreement")

    return results
