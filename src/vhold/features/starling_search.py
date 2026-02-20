"""STARLING conformational ensemble similarity search for disordered proteins.

Searches STARLING's pre-built FAISS index of ~35M IDR sequences from UniRef50
to find proteins with similar conformational ensembles — essentially
"Foldseek for disordered proteins."

UniRef50 hits are resolved to functional annotations via the UniProt REST API.

Requires: pip install vhold[starling]  (installs idptools-starling)
The STARLING search database (~18.6GB) auto-downloads on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vhold.features.disorder import (
    DisorderResult,
    _prepare_sequences_for_starling,
)
from vhold.utils.constants import (
    DEFAULT_STARLING_SIMILARITY_THRESHOLD,
    STARLING_SEARCH_K,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level cache for SearchEngine (avoids reloading ~18.6GB index).
# Not thread-safe; acceptable for single-threaded pipeline.
_cached_engine: Any = None


@dataclass
class StarlingMatch:
    """A match from STARLING similarity search."""

    query_id: str
    target_id: str  # UniRef50 ID (e.g., "UniRef50_Q9BYF1")
    distance: float  # L2 distance from FAISS
    similarity: float  # Cosine similarity (converted from L2 distance)
    sequence_length: int  # Target sequence length


@dataclass
class StarlingSearchResult:
    """Results from STARLING search for a batch of proteins."""

    matched: dict[str, list[StarlingMatch]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    total_queries: int = 0


def check_starling_search_available() -> bool:
    """Check if STARLING SearchEngine is available.

    Returns True only if both the starling package and the search
    engine module are importable.
    """
    try:
        from starling.search.search_engine import SearchEngine  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _distance_to_similarity(distance: float) -> float:
    """Convert L2 distance to cosine similarity.

    For L2-normalized vectors: L2_distance^2 = 2 - 2*cosine_similarity
    Therefore: cosine_similarity = 1 - distance^2 / 2

    Args:
        distance: L2 distance from FAISS

    Returns:
        Cosine similarity in [0, 1]
    """
    similarity = 1.0 - (distance ** 2) / 2.0
    return max(0.0, min(1.0, similarity))


def _get_search_engine() -> Any:
    """Get or create the cached STARLING SearchEngine.

    Caches the engine at module level to avoid reloading the ~18.6GB
    FAISS index on repeated calls.

    Returns:
        SearchEngine instance
    """
    global _cached_engine
    if _cached_engine is None:
        from starling.search.search_engine import SearchEngine

        logger.info("Loading STARLING search engine...")
        _cached_engine = SearchEngine.load("default")
    return _cached_engine


def search_starling_database(
    sequences: dict[str, str],
    disorder_results: dict[str, DisorderResult] | None = None,
    k: int = STARLING_SEARCH_K,
    similarity_threshold: float = DEFAULT_STARLING_SIMILARITY_THRESHOLD,
    device: str | None = None,
) -> StarlingSearchResult:
    """Search STARLING's UniRef50 IDR database for similar proteins.

    Args:
        sequences: Dict mapping protein_id to AA sequence
        disorder_results: Optional disorder predictions for truncation strategy
        k: Number of top hits to retrieve per query
        similarity_threshold: Minimum cosine similarity for hits
        device: Device for encoding ('cuda', 'cpu', None for auto)

    Returns:
        StarlingSearchResult with matched and unmatched protein IDs
    """
    if not sequences:
        return StarlingSearchResult()

    from starling import sequence_encoder

    result = StarlingSearchResult(total_queries=len(sequences))

    # Prepare sequences (handle >380 residue truncation)
    prepared = _prepare_sequences_for_starling(sequences, disorder_results)

    # Encode query sequences
    logger.info(f"Encoding {len(prepared)} sequences with STARLING...")
    kwargs: dict = {
        "aggregate": True,
        "ionic_strength": 150,
    }
    if device is not None:
        kwargs["device"] = device

    embedding_dict = sequence_encoder(prepared, **kwargs)

    # Stack into tensor for search
    ids = list(prepared.keys())
    import torch

    embeddings = []
    for pid in ids:
        emb = embedding_dict[pid]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        if emb.dim() > 1:
            emb = emb.squeeze()
        embeddings.append(emb)
    query_tensor = torch.stack(embeddings)  # (N, 512)

    # L2 normalize for consistent cosine similarity conversion
    norms = torch.norm(query_tensor, dim=1, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    query_tensor = query_tensor / norms

    # Load search engine (cached at module level)
    engine = _get_search_engine()

    # Search
    logger.info(f"Searching {len(ids)} queries against ~35M IDR sequences (k={k})...")
    raw_results = engine.search(queries=query_tensor, k=k)

    # Process results
    for pid, hits in zip(ids, raw_results):
        matches = []
        for hit in hits:
            # hit format: (distance, global_idx, uniref50_id, sequence_length)
            distance, _global_idx, uniref50_id, seq_length = hit
            similarity = _distance_to_similarity(float(distance))

            if similarity >= similarity_threshold:
                matches.append(
                    StarlingMatch(
                        query_id=pid,
                        target_id=str(uniref50_id),
                        distance=float(distance),
                        similarity=similarity,
                        sequence_length=int(seq_length),
                    )
                )

        if matches:
            # Sort by similarity descending
            matches.sort(key=lambda m: m.similarity, reverse=True)
            result.matched[pid] = matches
        else:
            result.unmatched.append(pid)

    logger.info(
        f"STARLING search: {len(result.matched)} proteins matched, "
        f"{len(result.unmatched)} unmatched"
    )

    return result


def build_starling_consensus_results(
    matches: dict[str, list[StarlingMatch]],
    query_lengths: dict[str, int],
    uniprot_cache: dict[str, dict] | None = None,
) -> tuple[dict, list[str]]:
    """Build ConsensusResult objects for STARLING-matched proteins.

    Resolves UniRef50 hits to UniProt annotations and builds ConsensusResult
    objects following the same pattern as build_embedding_consensus_results().

    Args:
        matches: Dict mapping query_id to list of StarlingMatch
        query_lengths: Dict mapping query IDs to sequence lengths
        uniprot_cache: Optional UniProt annotation cache

    Returns:
        Tuple of (results_dict, skipped_ids) where results_dict maps
        query_id to ConsensusResult
    """
    from vhold.features.uniprot_lookup import (
        extract_uniprot_accession,
        lookup_uniprot_batch,
    )
    from vhold.results.categories import AnnotationEvidence, get_classification_source
    from vhold.results.consensus import ConsensusResult

    if uniprot_cache is None:
        uniprot_cache = {}

    # Collect all unique UniProt accessions to look up
    accession_map: dict[str, str] = {}  # uniref50_id -> uniprot_accession
    all_accessions: list[str] = []

    for match_list in matches.values():
        for match in match_list:
            if match.target_id not in accession_map:
                acc = extract_uniprot_accession(match.target_id)
                if acc:
                    accession_map[match.target_id] = acc
                    if acc not in uniprot_cache:
                        all_accessions.append(acc)

    # Batch lookup UniProt annotations
    if all_accessions:
        unique_accessions = list(set(all_accessions))
        logger.info(f"Looking up {len(unique_accessions)} UniProt annotations...")
        lookup_uniprot_batch(unique_accessions, cache=uniprot_cache)

    # Build ConsensusResult for each matched protein
    results: dict[str, ConsensusResult] = {}
    skipped_ids: list[str] = []

    for query_id, match_list in matches.items():
        if not match_list:
            continue

        best_match = match_list[0]  # Already sorted by similarity
        length = query_lengths.get(query_id, 0)

        # Resolve annotation
        acc = accession_map.get(best_match.target_id)
        annotation: dict = {}
        if acc and acc in uniprot_cache:
            annotation = uniprot_cache[acc]
        else:
            # No annotation available — skip
            skipped_ids.append(query_id)
            continue

        # Check for minimal annotation
        description = annotation.get("description", "hypothetical protein")
        if description in ("hypothetical protein", ""):
            # Still useful if we have other evidence (GO, Pfam)
            has_evidence = any(
                annotation.get(k) for k in ("go_bp", "go_mf", "pfam", "keywords")
            )
            if not has_evidence:
                skipped_ids.append(query_id)
                continue

        # Classify using all available evidence, tracking source
        gene = annotation.get("gene")
        evidence = AnnotationEvidence(
            pfam=annotation.get("pfam"),
            go_bp=annotation.get("go_bp"),
            go_mf=annotation.get("go_mf"),
        )
        category, evidence_source = get_classification_source(
            description, gene, evidence=evidence,
        )

        # Map similarity to consensus score
        consensus_score = min(1.0, best_match.similarity * 0.95 + 0.05)

        # Determine confidence level
        if consensus_score >= 0.8:
            confidence = "high"
        elif consensus_score >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        result = ConsensusResult(
            query_id=query_id,
            query_length=length,
            primary_hit=None,  # No FoldseekHit for STARLING matches
            primary_annotation=annotation,
            primary_source=f"starling_{annotation.get('source', 'uniprot')}",
            consensus_score=consensus_score,
            confidence_level=confidence,
            agreement="single",
            functional_category=category,
            classification_source=f"starling_search:{evidence_source}",
            novelty="ensemble_match",
            structure_quality_score=1.0,
            structure_quality_source="none",
        )

        results[query_id] = result

    if skipped_ids:
        logger.info(
            f"Skipped {len(skipped_ids)} STARLING hits with no useful annotation"
        )
    logger.info(f"Built {len(results)} STARLING-matched annotations")

    return results, skipped_ids
