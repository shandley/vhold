"""Intrinsic disorder prediction and ensemble-aware embeddings for viral proteins.

Layer 1: metapredict for fast per-residue disorder scoring (CPU/GPU).
Layer 2: STARLING for ensemble-aware embeddings of disordered regions.

Requires: pip install vhold[disorder]  (installs metapredict)
Optional: pip install vhold[starling]  (installs idptools-starling)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vhold.utils.constants import (
    DEFAULT_DISORDER_THRESHOLD,
    HIGH_DISORDER_FRACTION,
    MODERATE_DISORDER_FRACTION,
    STARLING_MAX_SEQUENCE_LENGTH,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DisorderResult:
    """Disorder prediction for a single protein."""

    protein_id: str
    disorder_scores: np.ndarray  # Per-residue scores (0-1), shape (L,)
    disorder_fraction: float  # Fraction of residues above threshold
    disorder_regions: list[tuple[int, int]]  # (start, end) 0-indexed disordered regions
    disorder_class: str  # "ordered", "partially_disordered", "highly_disordered"
    sequence_length: int


@dataclass
class DisorderReport:
    """Summary of disorder predictions for a protein set."""

    total_proteins: int = 0
    highly_disordered: int = 0
    partially_disordered: int = 0
    ordered: int = 0
    mean_disorder_fraction: float = 0.0
    results: dict[str, DisorderResult] = field(default_factory=dict)


def check_metapredict_available() -> bool:
    """Check if metapredict is installed."""
    try:
        import metapredict  # noqa: F401

        return True
    except ImportError:
        return False


def check_starling_available() -> bool:
    """Check if STARLING is installed."""
    try:
        import starling  # noqa: F401

        return True
    except ImportError:
        return False


def classify_disorder(fraction: float) -> str:
    """Classify protein disorder level.

    Args:
        fraction: Fraction of disordered residues (0-1)

    Returns:
        "ordered", "partially_disordered", or "highly_disordered"
    """
    if fraction >= HIGH_DISORDER_FRACTION:
        return "highly_disordered"
    elif fraction >= MODERATE_DISORDER_FRACTION:
        return "partially_disordered"
    else:
        return "ordered"


def _extract_regions(
    scores: np.ndarray,
    threshold: float,
    min_length: int = 5,
) -> list[tuple[int, int]]:
    """Extract contiguous disordered regions.

    Args:
        scores: Per-residue disorder scores
        threshold: Score threshold for disorder
        min_length: Minimum region length to report

    Returns:
        List of (start, end) tuples (0-indexed, end exclusive)
    """
    regions = []
    in_region = False
    start = 0

    for i, score in enumerate(scores):
        if score >= threshold and not in_region:
            start = i
            in_region = True
        elif score < threshold and in_region:
            if i - start >= min_length:
                regions.append((start, i))
            in_region = False

    # Handle region extending to end
    if in_region and len(scores) - start >= min_length:
        regions.append((start, len(scores)))

    return regions


def predict_disorder_single(
    sequence: str,
    protein_id: str = "",
    threshold: float = DEFAULT_DISORDER_THRESHOLD,
) -> DisorderResult:
    """Predict disorder for a single protein.

    Args:
        sequence: Amino acid sequence
        protein_id: Protein identifier
        threshold: Per-residue disorder threshold

    Returns:
        DisorderResult with per-residue scores and summary
    """
    import metapredict as meta

    scores = meta.predict_disorder(sequence, return_numpy=True)
    scores = np.asarray(scores, dtype=np.float32)

    n_disordered = int((scores >= threshold).sum())
    fraction = n_disordered / len(scores) if len(scores) > 0 else 0.0

    regions = _extract_regions(scores, threshold)

    return DisorderResult(
        protein_id=protein_id,
        disorder_scores=scores,
        disorder_fraction=fraction,
        disorder_regions=regions,
        disorder_class=classify_disorder(fraction),
        sequence_length=len(sequence),
    )


def predict_disorder_batch(
    sequences: dict[str, str],
    threshold: float = DEFAULT_DISORDER_THRESHOLD,
    show_progress: bool = True,
) -> dict[str, DisorderResult]:
    """Predict disorder for multiple proteins.

    metapredict V3 auto-detects GPU for batch prediction.

    Args:
        sequences: Dict mapping protein_id to AA sequence
        threshold: Per-residue disorder threshold
        show_progress: Show progress bar

    Returns:
        Dict mapping protein_id to DisorderResult
    """
    if not sequences:
        return {}

    import metapredict as meta

    # metapredict V3 accepts dict input and auto-batches
    # Return format: {key: [sequence, scores_array]}
    raw_results = meta.predict_disorder(
        sequences,
        return_numpy=True,
        show_progress_bar=show_progress,
    )

    results = {}
    for protein_id, value in raw_results.items():
        # metapredict dict return: [sequence_str, scores_ndarray]
        scores = np.asarray(value[1], dtype=np.float32)

        n_disordered = int((scores >= threshold).sum())
        fraction = n_disordered / len(scores) if len(scores) > 0 else 0.0
        regions = _extract_regions(scores, threshold)

        results[protein_id] = DisorderResult(
            protein_id=protein_id,
            disorder_scores=scores,
            disorder_fraction=fraction,
            disorder_regions=regions,
            disorder_class=classify_disorder(fraction),
            sequence_length=len(scores),
        )

    return results


def analyze_disorder(results: dict[str, DisorderResult]) -> DisorderReport:
    """Summarize disorder predictions.

    Args:
        results: Dict of DisorderResult objects

    Returns:
        DisorderReport with summary statistics
    """
    report = DisorderReport(total_proteins=len(results), results=results)

    fractions = []
    for r in results.values():
        fractions.append(r.disorder_fraction)
        if r.disorder_class == "highly_disordered":
            report.highly_disordered += 1
        elif r.disorder_class == "partially_disordered":
            report.partially_disordered += 1
        else:
            report.ordered += 1

    if fractions:
        report.mean_disorder_fraction = float(np.mean(fractions))

    return report


def _prepare_sequences_for_starling(
    sequences: dict[str, str],
    disorder_results: dict[str, DisorderResult] | None = None,
) -> dict[str, str]:
    """Prepare sequences for STARLING, handling the 380-residue max length.

    For proteins exceeding STARLING's max sequence length (380 residues),
    extracts the longest disordered region if disorder data is available,
    otherwise truncates to the first 380 residues.

    Args:
        sequences: Dict mapping protein_id to AA sequence
        disorder_results: Optional disorder predictions for region extraction

    Returns:
        Dict mapping protein_id to (possibly truncated) sequences
    """
    prepared = {}
    n_truncated = 0

    for pid, seq in sequences.items():
        if len(seq) <= STARLING_MAX_SEQUENCE_LENGTH:
            prepared[pid] = seq
            continue

        # Try to extract longest disordered region
        if disorder_results and pid in disorder_results:
            regions = disorder_results[pid].disorder_regions
            if regions:
                # Find longest region
                longest = max(regions, key=lambda r: r[1] - r[0])
                region_seq = seq[longest[0] : longest[1]]
                if len(region_seq) > STARLING_MAX_SEQUENCE_LENGTH:
                    region_seq = region_seq[:STARLING_MAX_SEQUENCE_LENGTH]
                prepared[pid] = region_seq
                n_truncated += 1
                continue

        # Fallback: truncate to first 380 residues
        prepared[pid] = seq[:STARLING_MAX_SEQUENCE_LENGTH]
        n_truncated += 1

    if n_truncated > 0:
        logger.info(
            f"STARLING: {n_truncated}/{len(sequences)} proteins truncated "
            f"to {STARLING_MAX_SEQUENCE_LENGTH} residues"
        )

    return prepared


def extract_starling_embeddings(
    sequences: dict[str, str],
    disorder_results: dict[str, DisorderResult] | None = None,
    aggregate: bool = True,
    ionic_strength: int = 150,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Extract STARLING ensemble-aware embeddings.

    STARLING produces 512-dim embeddings optimized for conformational
    ensemble similarity. Disordered proteins with similar conformational
    properties cluster together.

    Args:
        sequences: Dict mapping protein_id to AA sequence
        disorder_results: Optional disorder predictions for truncation strategy
        aggregate: If True, mean-pool to single vector per protein
        ionic_strength: Salt concentration in mM (20, 150, 300)
        batch_size: Batch size for encoding
        device: Device ('cuda', 'cpu', None for auto)

    Returns:
        Tuple of (protein_ids, embeddings_matrix) where
        embeddings_matrix has shape (N, 512) when aggregate=True
    """
    from starling import sequence_encoder

    # Handle sequences exceeding STARLING's max length
    prepared = _prepare_sequences_for_starling(sequences, disorder_results)

    # STARLING sequence_encoder accepts dict input
    kwargs: dict = {
        "ionic_strength": ionic_strength,
        "aggregate": aggregate,
        "batch_size": batch_size,
    }
    if device is not None:
        kwargs["device"] = device

    embedding_dict = sequence_encoder(prepared, **kwargs)

    ids = list(prepared.keys())
    embeddings = []
    for pid in ids:
        emb = embedding_dict[pid]
        if hasattr(emb, "numpy"):
            emb = emb.detach().cpu().numpy()
        embeddings.append(np.asarray(emb, dtype=np.float32))

    embeddings_matrix = np.stack(embeddings)  # (N, 512)

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings_matrix = embeddings_matrix / norms

    return ids, embeddings_matrix.astype(np.float32)
