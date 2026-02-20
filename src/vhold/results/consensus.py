"""Multi-database consensus scoring for vhold."""

import math
from dataclasses import dataclass, field
from typing import Optional

from vhold.results.categories import (
    classify_protein,
    AnnotationEvidence,
    get_classification_source,
)
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

# Structure quality thresholds (AlphaFold2/ESMFold/ColabFold)
# pLDDT: predicted Local Distance Difference Test (0-100)
# pTM: predicted Template Modeling score (0-1)
STRUCTURE_QUALITY_THRESHOLDS = {
    "high_plddt": 70.0,     # High confidence structure prediction
    "medium_plddt": 50.0,   # Medium confidence
    "low_plddt": 30.0,      # Low confidence (mostly disorder)
    "high_ptm": 0.5,        # High global structure confidence
    "medium_ptm": 0.3,      # Medium global confidence
    "good_msa_depth": 100,  # MSA depth for reliable ColabFold predictions
}

# Structure quality weighting factor (0-1)
# This determines how much structure quality affects the final score
STRUCTURE_QUALITY_WEIGHT = 0.15  # 15% of total score influenced by structure quality

# Novelty classification based on sequence identity
# Higher novelty = more value from structural search (BLAST would miss these)
NOVELTY_THRESHOLDS = {
    "database_match": 0.95,  # Same protein, different DB entry
    "close_homolog": 0.70,   # Related strain/variant, BLAST would find
    "remote_homolog": 0.30,  # Structure-based transfer, BLAST marginal
    # Below 0.30 = "twilight_zone" - BLAST fails, structure succeeds
}


def classify_hit_novelty(identity: float) -> str:
    """Classify hit by novelty based on sequence identity.

    Higher novelty = more value from structural search.

    Args:
        identity: Sequence identity (0-1)

    Returns:
        Novelty classification string
    """
    if identity >= NOVELTY_THRESHOLDS["database_match"]:
        return "database_match"  # Likely same protein in DB
    elif identity >= NOVELTY_THRESHOLDS["close_homolog"]:
        return "close_homolog"   # BLAST would find this
    elif identity >= NOVELTY_THRESHOLDS["remote_homolog"]:
        return "remote_homolog"  # Structure-based annotation
    else:
        return "twilight_zone"   # Novel structural similarity


@dataclass
class HitScore:
    """Scored hit from a single database."""

    hit: FoldseekHit
    quality_score: float  # Combined alignment + structure quality
    weighted_score: float  # quality_score * database_weight * bonuses
    annotation: dict = field(default_factory=dict)
    structure_quality: float = 1.0  # Structure prediction quality (0-1)
    structure_quality_source: str = "none"  # Source of structure quality


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
    classification_source: str = "keywords"  # What evidence was used for classification

    # Structure quality metrics
    structure_quality_score: float = 1.0  # Structure prediction quality (0-1)
    structure_quality_source: str = "none"  # Source: colabfold, esmfold, bfvd_af2, none

    # Novelty classification (indicates value of structural search)
    novelty: str = "none"  # database_match, close_homolog, remote_homolog, twilight_zone

    # Genomic position (from GenBank/GFF input)
    contig: str | None = None
    start: int | None = None
    end: int | None = None
    strand: int | None = None

    # Disorder prediction (from metapredict, populated in Step 4a.2)
    disorder_fraction: float | None = None
    disorder_regions: list | None = None
    disorder_class: str | None = None  # "ordered", "partially_disordered", "highly_disordered"

    # All hits by database
    hits_by_db: dict = field(default_factory=dict)

    @property
    def is_annotated(self) -> bool:
        """Check if any annotation exists (Foldseek hit or embedding match)."""
        return self.primary_hit is not None or bool(self.primary_annotation)

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
            "classification_source": self.classification_source,
            "novelty": self.novelty,
            "structure_quality_score": round(self.structure_quality_score, 3),
            "structure_quality_source": self.structure_quality_source,
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

        # Add genomic position if available
        if self.contig is not None:
            result["contig"] = self.contig
            result["start"] = self.start
            result["end"] = self.end
            result["strand"] = self.strand

        # Add disorder prediction if available
        if self.disorder_fraction is not None:
            result["disorder_fraction"] = round(self.disorder_fraction, 3)
            result["disorder_class"] = self.disorder_class
            if self.disorder_regions:
                result["disorder_regions"] = ";".join(
                    f"{s}-{e}" for s, e in self.disorder_regions
                )

        # Add hit counts per database
        for db, hits in self.hits_by_db.items():
            result[f"{db}_hits"] = len(hits)

        return result


def calculate_structure_quality(annotation: dict) -> tuple[float, str]:
    """Calculate structure quality score from pLDDT/pTM metrics.

    Uses available structure quality metrics from ESMFold or ColabFold
    predictions. Higher quality structures provide more reliable homology
    detection.

    Args:
        annotation: Annotation dict containing structure quality fields

    Returns:
        Tuple of (quality_score, quality_source) where:
        - quality_score: 0-1 score (1 = highest quality)
        - quality_source: Which prediction method was used
    """
    if not annotation:
        return 1.0, "none"  # No penalty if no quality data available

    # Extract quality metrics
    esmfold_plddt = annotation.get("esmfold_plddt")
    esmfold_ptm = annotation.get("esmfold_ptm")
    colabfold_plddt = annotation.get("colabfold_plddt")
    colabfold_ptm = annotation.get("colabfold_ptm")
    colabfold_msa_depth = annotation.get("colabfold_msa_depth")
    # BFVD uses different field names
    bfvd_plddt = annotation.get("plddt")
    bfvd_ptm = annotation.get("ptm")

    # Determine which source to use
    # Prefer ColabFold if MSA depth is good, otherwise use ESMFold or BFVD
    use_colabfold = (
        colabfold_plddt is not None
        and colabfold_msa_depth is not None
        and colabfold_msa_depth >= STRUCTURE_QUALITY_THRESHOLDS["good_msa_depth"]
    )

    if use_colabfold:
        plddt = colabfold_plddt
        ptm = colabfold_ptm
        source = "colabfold"
    elif esmfold_plddt is not None:
        plddt = esmfold_plddt
        ptm = esmfold_ptm
        source = "esmfold"
    elif bfvd_plddt is not None:
        plddt = bfvd_plddt
        ptm = bfvd_ptm
        source = "bfvd_af2"
    elif colabfold_plddt is not None:
        # Use ColabFold even with low MSA depth if nothing else available
        plddt = colabfold_plddt
        ptm = colabfold_ptm
        source = "colabfold_low_msa"
    else:
        # No quality metrics available - no penalty
        return 1.0, "none"

    # Convert pLDDT to 0-1 score
    # pLDDT is 0-100, we normalize and apply sigmoid-like scaling
    # High confidence (>70) -> score near 1.0
    # Medium confidence (50-70) -> score 0.7-0.9
    # Low confidence (<50) -> score 0.5-0.7
    # Very low (<30) -> score 0.3-0.5
    if plddt is None:
        plddt_score = 0.8  # Default if missing
    elif plddt >= STRUCTURE_QUALITY_THRESHOLDS["high_plddt"]:
        # High confidence: 70-100 -> 0.9-1.0
        plddt_score = 0.9 + 0.1 * ((plddt - 70) / 30)
    elif plddt >= STRUCTURE_QUALITY_THRESHOLDS["medium_plddt"]:
        # Medium confidence: 50-70 -> 0.7-0.9
        plddt_score = 0.7 + 0.2 * ((plddt - 50) / 20)
    elif plddt >= STRUCTURE_QUALITY_THRESHOLDS["low_plddt"]:
        # Low confidence: 30-50 -> 0.5-0.7
        plddt_score = 0.5 + 0.2 * ((plddt - 30) / 20)
    else:
        # Very low: 0-30 -> 0.3-0.5
        plddt_score = 0.3 + 0.2 * (plddt / 30)

    # Convert pTM to 0-1 score
    # pTM is already 0-1, but we apply similar scaling
    if ptm is None:
        ptm_score = 0.8  # Default if missing
    elif ptm >= STRUCTURE_QUALITY_THRESHOLDS["high_ptm"]:
        # High confidence: 0.5-1.0 -> 0.85-1.0
        ptm_score = 0.85 + 0.15 * ((ptm - 0.5) / 0.5)
    elif ptm >= STRUCTURE_QUALITY_THRESHOLDS["medium_ptm"]:
        # Medium confidence: 0.3-0.5 -> 0.7-0.85
        ptm_score = 0.7 + 0.15 * ((ptm - 0.3) / 0.2)
    else:
        # Low confidence: 0-0.3 -> 0.5-0.7
        ptm_score = 0.5 + 0.2 * (ptm / 0.3)

    # Combine pLDDT and pTM (pLDDT is more important for local accuracy)
    # pLDDT weight 0.7, pTM weight 0.3
    combined_score = 0.7 * plddt_score + 0.3 * ptm_score

    # Apply MSA depth bonus for ColabFold (high MSA = more reliable)
    if source == "colabfold" and colabfold_msa_depth:
        if colabfold_msa_depth >= 1000:
            combined_score = min(1.0, combined_score * 1.05)  # 5% bonus
        elif colabfold_msa_depth >= 500:
            combined_score = min(1.0, combined_score * 1.02)  # 2% bonus

    return combined_score, source


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

    Combines hit quality (e-value, identity, coverage), database weight,
    annotation quality bonus, and structure quality into a final weighted score.

    Args:
        hit: FoldseekHit object
        annotation: Annotation dict from database (may include structure quality)

    Returns:
        HitScore object
    """
    # Base quality from alignment metrics
    alignment_quality = calculate_hit_quality(hit)

    # Structure quality from pLDDT/pTM (if available)
    structure_quality, structure_source = calculate_structure_quality(annotation)

    # Combine alignment and structure quality
    # Structure quality affects the score proportionally to STRUCTURE_QUALITY_WEIGHT
    # Formula: (1 - weight) * alignment + weight * (alignment * structure)
    # This means structure quality can reduce score by up to STRUCTURE_QUALITY_WEIGHT
    combined_quality = alignment_quality * (
        (1 - STRUCTURE_QUALITY_WEIGHT) + STRUCTURE_QUALITY_WEIGHT * structure_quality
    )

    # Database weight
    db_weight = DATABASE_WEIGHTS.get(hit.source_db, 1.0)

    # Bonus for having a real description (not just UniProt ID)
    desc = annotation.get("description", "")
    if desc and "UniProt:" not in desc and "hypothetical" not in desc.lower():
        db_weight *= 1.1  # 10% bonus for real functional annotation

    weighted = combined_quality * db_weight

    return HitScore(
        hit=hit,
        quality_score=combined_quality,
        weighted_score=weighted,
        annotation=annotation,
        structure_quality=structure_quality,
        structure_quality_source=structure_source,
    )


def _build_annotation_evidence(annotation: dict) -> AnnotationEvidence | None:
    """Build AnnotationEvidence from an annotation dictionary.

    Args:
        annotation: Annotation dict that may contain pfam, go_bp, go_mf, superfamily

    Returns:
        AnnotationEvidence object or None if no evidence available
    """
    if not annotation:
        return None

    # Check if any annotation evidence exists
    has_evidence = any(
        key in annotation
        for key in ["pfam", "go_bp", "go_mf", "superfamily", "gene3d"]
    )

    if not has_evidence:
        return None

    return AnnotationEvidence(
        pfam=annotation.get("pfam"),
        pfam_confidence=annotation.get("pfam_confidence", 0.0),
        go_bp=annotation.get("go_bp"),
        go_bp_confidence=annotation.get("go_bp_confidence", 0.0),
        go_mf=annotation.get("go_mf"),
        go_mf_confidence=annotation.get("go_mf_confidence", 0.0),
        superfamily=annotation.get("superfamily"),
        superfamily_confidence=annotation.get("superfamily_confidence", 0.0),
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

    # Structure quality from primary hit
    result.structure_quality_score = primary_scored.structure_quality
    result.structure_quality_source = primary_scored.structure_quality_source

    # Novelty classification based on sequence identity
    # This indicates how much value the structural search provides
    result.novelty = classify_hit_novelty(primary_scored.hit.fident)

    # Build AnnotationEvidence from annotation dict for enhanced classification
    evidence = _build_annotation_evidence(primary_scored.annotation)

    # Classify protein into functional category using enhanced classification
    result.functional_category, result.classification_source = get_classification_source(
        result.description,
        result.primary_annotation.get("gene"),
        evidence,
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
            result.consensus_score = min(1.0, base_score)
    else:
        # Single database
        result.agreement = "single"
        result.consensus_score = min(1.0, primary_scored.weighted_score)

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
