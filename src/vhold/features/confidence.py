"""Confidence-based masking for 3Di sequences."""

from vhold.features.prostt5 import PredictionResult
from vhold.utils.constants import DEFAULT_CONFIDENCE_THRESHOLD
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def apply_confidence_mask(
    result: PredictionResult,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    mask_char: str = "X",
) -> str:
    """Apply confidence masking to a 3Di sequence.

    Low-confidence residues are replaced with a mask character.
    This can improve Foldseek search specificity by ignoring
    unreliable predictions.

    Args:
        result: PredictionResult with 3Di sequence and confidence scores
        threshold: Minimum confidence to keep (0-1)
        mask_char: Character to use for masked positions

    Returns:
        Masked 3Di sequence
    """
    if not result.confidence_scores:
        return result.three_di_sequence

    masked_seq = ""
    for residue, conf in zip(result.three_di_sequence, result.confidence_scores):
        if conf >= threshold:
            masked_seq += residue
        else:
            masked_seq += mask_char

    return masked_seq


def filter_low_confidence(
    results: dict[str, PredictionResult],
    min_confidence: float = 0.5,
) -> dict[str, PredictionResult]:
    """Filter out predictions with low overall confidence.

    Args:
        results: Dict of prediction results
        min_confidence: Minimum mean confidence to keep

    Returns:
        Filtered dict of prediction results
    """
    filtered = {}

    for seq_id, result in results.items():
        if result.mean_confidence >= min_confidence:
            filtered[seq_id] = result
        else:
            logger.warning(
                f"Filtered {seq_id}: mean confidence {result.mean_confidence:.3f} "
                f"< threshold {min_confidence}"
            )

    logger.info(
        f"Kept {len(filtered)}/{len(results)} predictions "
        f"(min confidence: {min_confidence})"
    )

    return filtered


def get_confidence_stats(results: dict[str, PredictionResult]) -> dict:
    """Calculate confidence statistics for a set of predictions.

    Args:
        results: Dict of prediction results

    Returns:
        Dict with confidence statistics
    """
    if not results:
        return {
            "count": 0,
            "mean_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
        }

    confidences = [r.mean_confidence for r in results.values()]

    return {
        "count": len(results),
        "mean_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
    }
