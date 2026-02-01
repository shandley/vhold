"""Annotation transfer for vhold."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vhold.databases.bfvd import load_bfvd_metadata, get_bfvd_annotation
from vhold.databases.viro3d import (
    load_viro3d_metadata,
    load_viro3d_annotations,
    get_viro3d_annotation,
)
from vhold.results.parser import FoldseekHit
from vhold.utils.constants import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnnotatedProtein:
    """An annotated protein with transferred annotations."""

    query_id: str
    query_length: int
    best_hit: FoldseekHit | None = None
    annotation: dict = field(default_factory=dict)
    confidence_level: str = "none"
    all_hits: list[FoldseekHit] = field(default_factory=list)

    @property
    def is_annotated(self) -> bool:
        """Check if protein has annotation."""
        return self.best_hit is not None

    @property
    def description(self) -> str:
        """Get annotation description."""
        return self.annotation.get("description", "hypothetical protein")

    @property
    def source_db(self) -> str:
        """Get source database."""
        if self.best_hit:
            return self.best_hit.source_db
        return ""

    @property
    def evalue(self) -> float | None:
        """Get best hit e-value."""
        if self.best_hit:
            return self.best_hit.evalue
        return None

    @property
    def identity(self) -> float | None:
        """Get best hit identity."""
        if self.best_hit:
            return self.best_hit.fident
        return None

    @property
    def coverage(self) -> float | None:
        """Get best hit coverage."""
        if self.best_hit:
            return self.best_hit.coverage
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        result = {
            "query_id": self.query_id,
            "query_length": self.query_length,
            "description": self.description,
            "confidence_level": self.confidence_level,
            "source_db": self.source_db,
            "target_id": self.best_hit.target if self.best_hit else "",
            "evalue": self.evalue if self.evalue is not None else "",
            "identity": self.identity if self.identity is not None else "",
            "coverage": self.coverage if self.coverage is not None else "",
            "bits": self.best_hit.bits if self.best_hit else "",
            "num_hits": len(self.all_hits),
        }

        # Add extra annotation fields
        for key, value in self.annotation.items():
            if key not in result:
                result[key] = value

        return result


def calculate_confidence_level(hit: FoldseekHit) -> str:
    """Calculate confidence level based on hit statistics.

    Args:
        hit: FoldseekHit object

    Returns:
        Confidence level string ('high', 'medium', 'low')
    """
    # Use combination of e-value, identity, and coverage
    score = 0.0

    # E-value contribution
    if hit.evalue < 1e-50:
        score += 0.4
    elif hit.evalue < 1e-20:
        score += 0.3
    elif hit.evalue < 1e-10:
        score += 0.2
    elif hit.evalue < 1e-5:
        score += 0.1

    # Identity contribution
    if hit.fident >= 0.7:
        score += 0.3
    elif hit.fident >= 0.5:
        score += 0.2
    elif hit.fident >= 0.3:
        score += 0.1

    # Coverage contribution
    if hit.coverage >= 0.8:
        score += 0.3
    elif hit.coverage >= 0.6:
        score += 0.2
    elif hit.coverage >= 0.4:
        score += 0.1

    # Determine level
    if score >= CONFIDENCE_HIGH:
        return "high"
    elif score >= CONFIDENCE_MEDIUM:
        return "medium"
    elif score >= CONFIDENCE_LOW:
        return "low"
    else:
        return "very_low"


def transfer_annotations(
    hits: list[FoldseekHit],
    query_lengths: dict[str, int],
    db_dir: Path | None = None,
) -> dict[str, AnnotatedProtein]:
    """Transfer annotations from database hits to query proteins.

    Args:
        hits: List of Foldseek hits
        query_lengths: Dict mapping query IDs to lengths
        db_dir: Database directory

    Returns:
        Dict mapping query IDs to AnnotatedProtein objects
    """
    # Group hits by query
    hits_by_query: dict[str, list[FoldseekHit]] = {}
    for hit in hits:
        if hit.query not in hits_by_query:
            hits_by_query[hit.query] = []
        hits_by_query[hit.query].append(hit)

    # Load metadata lazily
    bfvd_metadata = None
    viro3d_metadata = None
    viro3d_annotations = None

    # Transfer annotations
    results = {}

    # First, create entries for all queries (even those without hits)
    for query_id, length in query_lengths.items():
        results[query_id] = AnnotatedProtein(
            query_id=query_id,
            query_length=length,
        )

    # Process hits
    for query_id, query_hits in hits_by_query.items():
        # Sort by e-value
        query_hits.sort(key=lambda h: h.evalue)
        best_hit = query_hits[0]

        # Get annotation from appropriate database
        annotation = {}
        if best_hit.source_db == "bfvd":
            if bfvd_metadata is None:
                try:
                    bfvd_metadata = load_bfvd_metadata(db_dir)
                except FileNotFoundError:
                    logger.warning("BFVD metadata not found, skipping annotation")
                    bfvd_metadata = pd.DataFrame()

            if len(bfvd_metadata) > 0:
                ann = get_bfvd_annotation(best_hit.target, bfvd_metadata)
                if ann:
                    annotation = ann

        elif best_hit.source_db == "viro3d":
            if viro3d_metadata is None:
                try:
                    viro3d_metadata = load_viro3d_metadata(db_dir)
                except FileNotFoundError:
                    logger.warning("Viro3D metadata not found, skipping annotation")
                    viro3d_metadata = pd.DataFrame()

            if viro3d_annotations is None:
                try:
                    viro3d_annotations = load_viro3d_annotations(db_dir)
                except FileNotFoundError:
                    viro3d_annotations = pd.DataFrame()

            if len(viro3d_metadata) > 0:
                ann = get_viro3d_annotation(
                    best_hit.target,
                    viro3d_metadata,
                    viro3d_annotations if len(viro3d_annotations) > 0 else None,
                )
                if ann:
                    annotation = ann

        # Calculate confidence
        confidence = calculate_confidence_level(best_hit)

        # Create annotated protein
        length = query_lengths.get(query_id, best_hit.qlen)
        results[query_id] = AnnotatedProtein(
            query_id=query_id,
            query_length=length,
            best_hit=best_hit,
            annotation=annotation,
            confidence_level=confidence,
            all_hits=query_hits,
        )

    logger.info(
        f"Annotated {sum(1 for r in results.values() if r.is_annotated)}/{len(results)} proteins"
    )

    return results


def transfer_annotations_consensus(
    hits: list[FoldseekHit],
    query_lengths: dict[str, int],
    db_dir: Path | None = None,
):
    """Transfer annotations using multi-database consensus scoring.

    This function considers hits from all databases and builds a consensus
    annotation that weighs hit quality and database agreement.

    Args:
        hits: List of Foldseek hits from all databases
        query_lengths: Dict mapping query IDs to lengths
        db_dir: Database directory

    Returns:
        Dict mapping query IDs to ConsensusResult objects
    """
    from vhold.results.consensus import calculate_consensus

    # Group hits by query
    hits_by_query: dict[str, list[FoldseekHit]] = {}
    for hit in hits:
        if hit.query not in hits_by_query:
            hits_by_query[hit.query] = []
        hits_by_query[hit.query].append(hit)

    # Load metadata
    bfvd_metadata = None
    viro3d_metadata = None
    viro3d_annotations = None

    try:
        bfvd_metadata = load_bfvd_metadata(db_dir)
    except FileNotFoundError:
        logger.warning("BFVD metadata not found")
        bfvd_metadata = pd.DataFrame()

    try:
        viro3d_metadata = load_viro3d_metadata(db_dir)
    except FileNotFoundError:
        logger.warning("Viro3D metadata not found")
        viro3d_metadata = pd.DataFrame()

    try:
        viro3d_annotations = load_viro3d_annotations(db_dir)
    except FileNotFoundError:
        viro3d_annotations = pd.DataFrame()

    # Get annotations for all hits
    annotations_by_hit: dict[tuple[str, str], dict] = {}

    for query_hits in hits_by_query.values():
        for hit in query_hits:
            key = (hit.target, hit.source_db)
            if key in annotations_by_hit:
                continue

            annotation = {}
            if hit.source_db == "bfvd" and len(bfvd_metadata) > 0:
                ann = get_bfvd_annotation(hit.target, bfvd_metadata)
                if ann:
                    annotation = ann

            elif hit.source_db == "viro3d" and len(viro3d_metadata) > 0:
                ann = get_viro3d_annotation(
                    hit.target,
                    viro3d_metadata,
                    viro3d_annotations if len(viro3d_annotations) > 0 else None,
                )
                if ann:
                    annotation = ann

            annotations_by_hit[key] = annotation

    # Calculate consensus
    results = calculate_consensus(
        hits_by_query=hits_by_query,
        annotations_by_hit=annotations_by_hit,
        query_lengths=query_lengths,
    )

    return results
