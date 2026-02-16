"""Neighborhood voting for functional classification.

Uses genomic context (gene order) to reclassify unknown proteins
based on the functional categories of their neighboring genes.
Activates automatically when positional data is available
(GenBank/GFF input).
"""

from __future__ import annotations

from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Confidence weights for neighboring annotations
_CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
    "very_low": 0.2,
    "none": 0.1,
}


def build_gene_index(
    protein_positions: dict[str, dict],
) -> dict[str, list[tuple[str, int, int, int]]]:
    """Group proteins by contig and sort by start position.

    Args:
        protein_positions: Dict of protein_id → {contig, start, end, strand}

    Returns:
        Dict of contig → [(protein_id, start, end, strand), ...] sorted by start
    """
    index: dict[str, list[tuple[str, int, int, int]]] = {}

    for pid, pos in protein_positions.items():
        contig = pos.get("contig")
        if contig is None:
            continue
        if contig not in index:
            index[contig] = []
        index[contig].append((
            pid,
            pos.get("start", 0),
            pos.get("end", 0),
            pos.get("strand", 1),
        ))

    # Sort each contig by start position
    for contig in index:
        index[contig].sort(key=lambda x: x[1])

    return index


def get_neighbors(
    protein_id: str,
    gene_index: dict[str, list[tuple[str, int, int, int]]],
    protein_positions: dict[str, dict],
    window_size: int = 5,
) -> list[tuple[str, int]]:
    """Get neighboring genes on the same contig.

    Args:
        protein_id: Query protein ID
        gene_index: Gene index from build_gene_index()
        protein_positions: Protein position data
        window_size: Number of neighbors on each side

    Returns:
        List of (neighbor_id, rank_distance) pairs
    """
    pos = protein_positions.get(protein_id)
    if pos is None or pos.get("contig") is None:
        return []

    contig = pos["contig"]
    genes = gene_index.get(contig)
    if genes is None:
        return []

    # Find this protein's position in the sorted gene list
    target_idx = None
    for i, (pid, *_) in enumerate(genes):
        if pid == protein_id:
            target_idx = i
            break

    if target_idx is None:
        return []

    neighbors = []
    for i in range(max(0, target_idx - window_size), min(len(genes), target_idx + window_size + 1)):
        if i == target_idx:
            continue
        neighbor_id = genes[i][0]
        rank_distance = abs(i - target_idx)
        neighbors.append((neighbor_id, rank_distance))

    return neighbors


def vote_neighborhood(
    unknown_ids: list[str],
    annotations: dict,
    protein_positions: dict[str, dict],
    window_size: int = 5,
    min_votes: int = 2,
    min_vote_fraction: float = 0.5,
) -> dict[str, tuple[str, float]]:
    """Majority voting with distance and confidence weighting.

    For each unknown protein:
    1. Find ±window_size neighbors on same contig
    2. Collect non-unknown categories with weights:
       weight = confidence_weight * (1 / (1 + rank_distance))
    3. Winner needs ≥min_votes and ≥min_vote_fraction of total weight

    Args:
        unknown_ids: Protein IDs to reclassify
        annotations: Dict of protein_id → ConsensusResult
        protein_positions: Dict of protein_id → position data
        window_size: Number of neighbors on each side
        min_votes: Minimum number of non-unknown neighbors voting
        min_vote_fraction: Minimum fraction of total weight for winner

    Returns:
        Dict of protein_id → (predicted_category, vote_confidence)
    """
    if not unknown_ids or not protein_positions:
        return {}

    gene_index = build_gene_index(protein_positions)
    predictions: dict[str, tuple[str, float]] = {}

    for pid in unknown_ids:
        neighbors = get_neighbors(pid, gene_index, protein_positions, window_size)
        if not neighbors:
            continue

        # Collect votes from annotated neighbors
        category_weights: dict[str, float] = {}
        vote_count = 0

        for neighbor_id, rank_distance in neighbors:
            ann = annotations.get(neighbor_id)
            if ann is None:
                continue

            category = ann.functional_category
            if category == "unknown":
                continue

            conf_level = getattr(ann, "confidence_level", "none")
            conf_weight = _CONFIDENCE_WEIGHTS.get(conf_level, 0.1)
            distance_weight = 1.0 / (1.0 + rank_distance)
            weight = conf_weight * distance_weight

            category_weights[category] = category_weights.get(category, 0.0) + weight
            vote_count += 1

        if vote_count < min_votes:
            continue

        if not category_weights:
            continue

        # Find winner
        total_weight = sum(category_weights.values())
        winner = max(category_weights, key=category_weights.get)
        winner_weight = category_weights[winner]
        vote_fraction = winner_weight / total_weight if total_weight > 0 else 0

        if vote_fraction >= min_vote_fraction:
            predictions[pid] = (winner, round(vote_fraction, 3))

    return predictions
