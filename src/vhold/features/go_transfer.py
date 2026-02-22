"""FANTASIA-style GO term transfer via embedding similarity.

Transfers Gene Ontology annotations from well-characterized Swiss-Prot
reference proteins to query viral proteins using ProstT5 embedding
cosine similarity. Only experimental GO evidence codes are used
(EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC).

The Reliability Index (RI) follows FANTASIA (Bucio-Dominguez et al.,
Communications Biology 2025):
  - Per-donor RI = cosine similarity to donor
  - Combined RI = 1 - prod(1 - RI_i) across all k donors with that term

Cross-domain discovery: when the best donor is non-viral (lineage does
not start with "Viruses"), the match is flagged as a cross-domain hit,
indicating potential functional homology between viral and non-viral
proteins at the structural level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vhold.features.embeddings import EmbeddingExtractor

from vhold.databases.swissprot import (
    GOAnnotation,
    SwissProtEntry,
    get_available_scope,
    get_swissprot_embeddings_path,
    load_swissprot_metadata,
)
from vhold.utils.constants import (
    DEFAULT_CROSS_DOMAIN_THRESHOLD,
    DEFAULT_GO_TRANSFER_K,
    DEFAULT_GO_TRANSFER_THRESHOLD,
    DEFAULT_RELIABILITY_THRESHOLD,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class TransferredGOTerm:
    """A single GO term transferred from a Swiss-Prot donor."""

    go_id: str  # GO:0039694
    go_term: str  # "viral process"
    aspect: str  # BP, MF, CC
    evidence_code: str  # from donor (EXP, IDA, etc.)
    reliability_index: float  # combined RI across donors
    donor_accession: str  # best donor accession
    donor_similarity: float  # cosine similarity to best donor
    is_cross_domain: bool  # donor is non-viral
    donor_organism: str  # donor organism name
    donor_lineage: str  # donor taxonomic lineage


@dataclass
class GOTransferResult:
    """Complete GO transfer result for a single query protein."""

    query_id: str
    top_k_matches: list[tuple[str, float]] = field(default_factory=list)
    go_terms: list[TransferredGOTerm] = field(default_factory=list)
    best_similarity: float = 0.0
    best_accession: str = ""
    is_cross_domain: bool = False  # any donor is non-viral

    @property
    def go_bp(self) -> list[TransferredGOTerm]:
        """GO Biological Process terms."""
        return [t for t in self.go_terms if t.aspect == "BP"]

    @property
    def go_mf(self) -> list[TransferredGOTerm]:
        """GO Molecular Function terms."""
        return [t for t in self.go_terms if t.aspect == "MF"]

    @property
    def go_cc(self) -> list[TransferredGOTerm]:
        """GO Cellular Component terms."""
        return [t for t in self.go_terms if t.aspect == "CC"]

    @property
    def has_terms(self) -> bool:
        """Whether any GO terms were transferred."""
        return len(self.go_terms) > 0

    @property
    def max_ri(self) -> float:
        """Maximum Reliability Index across all transferred terms."""
        if not self.go_terms:
            return 0.0
        return max(t.reliability_index for t in self.go_terms)

    def go_ids_by_aspect(self, aspect: str) -> list[str]:
        """Get GO IDs for a specific aspect.

        Args:
            aspect: "BP", "MF", or "CC"

        Returns:
            List of GO IDs sorted by reliability index (descending)
        """
        terms = [t for t in self.go_terms if t.aspect == aspect]
        terms.sort(key=lambda t: t.reliability_index, reverse=True)
        return [t.go_id for t in terms]


# ============================================================================
# Swiss-Prot Reference Database
# ============================================================================


class SwissProtReferenceDB:
    """Swiss-Prot reference database for GO term transfer.

    Wraps pre-computed ProstT5 embeddings and Swiss-Prot metadata.
    Search uses brute-force cosine similarity (dot product on
    L2-normalized vectors), same as the triage EmbeddingDatabase.
    """

    def __init__(
        self,
        scope: str | None = None,
        db_dir: Path | None = None,
    ):
        """Initialize Swiss-Prot reference DB.

        Args:
            scope: "viral", "all", or None (auto-detect best available)
            db_dir: Base database directory
        """
        if scope is None:
            scope = get_available_scope(db_dir)
            if scope is None:
                msg = (
                    "No Swiss-Prot reference database installed. "
                    "Run 'scripts/build_swissprot_reference.py' to build one, "
                    "or use 'vhold install --swissprot' to download."
                )
                raise FileNotFoundError(msg)

        self.scope = scope
        self.db_dir = db_dir
        self.embeddings: np.ndarray | None = None  # (N, 1024) float32
        self.protein_ids: list[str] = []
        self.metadata: dict[str, SwissProtEntry] = {}
        self._loaded = False
        self._viral_go_ids: set[str] | None = None

    def load(self) -> None:
        """Load embeddings and metadata into memory."""
        if self._loaded:
            return

        # Load embeddings
        emb_path = get_swissprot_embeddings_path(self.scope, self.db_dir)
        logger.info(f"Loading Swiss-Prot embeddings from {emb_path}")
        data = np.load(emb_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.protein_ids = list(data["protein_ids"])
        logger.info(
            f"Loaded {len(self.protein_ids)} Swiss-Prot embeddings (scope={self.scope})"
        )

        # Load metadata
        self.metadata = load_swissprot_metadata(self.scope, self.db_dir)

        # Compute viral GO set for plausibility filtering
        from vhold.features.go_filter import build_viral_go_set

        viral_go_set = build_viral_go_set(self.metadata)
        self._viral_go_ids = viral_go_set
        logger.info(f"Viral GO set: {len(viral_go_set)} unique GO IDs")

        self._loaded = True

    @property
    def viral_go_ids(self) -> set[str]:
        """GO IDs from viral Swiss-Prot entries (for plausibility filtering)."""
        if self._viral_go_ids is None:
            return set()
        return self._viral_go_ids

    @property
    def size(self) -> int:
        """Number of reference proteins."""
        return len(self.protein_ids)

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = DEFAULT_GO_TRANSFER_K,
        threshold: float = DEFAULT_GO_TRANSFER_THRESHOLD,
    ) -> list[list[tuple[str, float]]]:
        """Batch cosine similarity search.

        Args:
            query_embeddings: (N, 1024) L2-normalized query matrix
            k: Number of nearest neighbors per query
            threshold: Minimum cosine similarity

        Returns:
            List of N lists, each containing up to k (accession, similarity)
            tuples sorted by similarity descending. Empty list for queries
            with no matches above threshold.
        """
        if not self._loaded or self.embeddings is None:
            raise RuntimeError("Swiss-Prot DB not loaded. Call load() first.")

        # Similarity matrix: (N_query, N_ref) via dot product
        similarities = query_embeddings @ self.embeddings.T  # (N_q, N_ref)

        results: list[list[tuple[str, float]]] = []

        for i in range(similarities.shape[0]):
            row = similarities[i]

            # Get top-k indices
            if k >= len(row):
                top_indices = np.argsort(-row)
            else:
                # Use argpartition for efficiency on large arrays
                top_indices = np.argpartition(-row, k)[:k]
                top_indices = top_indices[np.argsort(-row[top_indices])]

            matches = []
            for idx in top_indices:
                sim = float(row[idx])
                if sim < threshold:
                    break  # sorted descending, so all remaining are below
                matches.append((self.protein_ids[idx], sim))

            results.append(matches)

        return results


# ============================================================================
# GO Term Transfer
# ============================================================================


def transfer_go_terms(
    query_id: str,
    matches: list[tuple[str, float]],
    metadata: dict[str, SwissProtEntry],
    threshold: float = DEFAULT_GO_TRANSFER_THRESHOLD,
    reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
    cross_domain_threshold: float = DEFAULT_CROSS_DOMAIN_THRESHOLD,
    cross_domain: bool = True,
) -> GOTransferResult:
    """Transfer GO terms from k nearest Swiss-Prot neighbors.

    For each GO term appearing in any of the k donors:
    - Per-donor RI = cosine_similarity (following FANTASIA)
    - Combined RI = 1 - prod(1 - RI_i) across all donors with that term
    - Filter by reliability_threshold

    Domain-aware filtering: non-viral donors must exceed
    ``cross_domain_threshold`` (default 0.95) to contribute GO terms.
    When ``cross_domain=False``, all non-viral donors are excluded.

    Cross-domain flag is set when any donor's lineage does not start
    with "Viruses".

    Args:
        query_id: Query protein ID
        matches: List of (accession, similarity) tuples from search
        metadata: Swiss-Prot metadata dict
        threshold: Minimum similarity (matches below are excluded)
        reliability_threshold: Minimum combined RI to keep a GO term
        cross_domain_threshold: Minimum similarity for non-viral donors
        cross_domain: Whether to include non-viral donors at all

    Returns:
        GOTransferResult with transferred GO terms
    """
    if not matches:
        return GOTransferResult(query_id=query_id)

    # Filter matches above threshold
    valid_matches = [(acc, sim) for acc, sim in matches if sim >= threshold]
    if not valid_matches:
        return GOTransferResult(query_id=query_id)

    best_acc, best_sim = valid_matches[0]

    # Check cross-domain status
    any_cross_domain = False
    for acc, _ in valid_matches:
        entry = metadata.get(acc)
        if entry and not entry.is_viral:
            any_cross_domain = True
            break

    # Collect GO terms from all donors with per-donor RI
    # Key: (go_id, aspect) → list of (similarity, donor_accession, go_annotation)
    go_term_donors: dict[
        tuple[str, str], list[tuple[float, str, GOAnnotation, SwissProtEntry]]
    ] = {}

    for acc, sim in valid_matches:
        entry = metadata.get(acc)
        if entry is None:
            continue

        # Domain-aware filtering: non-viral donors need higher similarity
        if not entry.is_viral:
            if not cross_domain:
                continue  # Skip all non-viral donors
            if sim < cross_domain_threshold:
                continue  # Below cross-domain threshold

        for go_ann in entry.go_annotations:
            key = (go_ann.go_id, go_ann.aspect)
            if key not in go_term_donors:
                go_term_donors[key] = []
            go_term_donors[key].append((sim, acc, go_ann, entry))

    # Calculate combined RI for each GO term
    transferred_terms: list[TransferredGOTerm] = []

    for (go_id, aspect), donors in go_term_donors.items():
        # Combined RI = 1 - prod(1 - RI_i)
        product = 1.0
        for sim, _, _, _ in donors:
            product *= 1.0 - sim
        combined_ri = 1.0 - product

        if combined_ri < reliability_threshold:
            continue

        # Use the best donor for this term as representative
        best_donor = max(donors, key=lambda d: d[0])
        donor_sim, donor_acc, go_ann, donor_entry = best_donor

        transferred_terms.append(
            TransferredGOTerm(
                go_id=go_id,
                go_term=go_ann.go_term,
                aspect=aspect,
                evidence_code=go_ann.evidence_code,
                reliability_index=combined_ri,
                donor_accession=donor_acc,
                donor_similarity=donor_sim,
                is_cross_domain=not donor_entry.is_viral,
                donor_organism=donor_entry.organism,
                donor_lineage=donor_entry.lineage,
            )
        )

    # Sort by RI descending
    transferred_terms.sort(key=lambda t: t.reliability_index, reverse=True)

    return GOTransferResult(
        query_id=query_id,
        top_k_matches=valid_matches,
        go_terms=transferred_terms,
        best_similarity=best_sim,
        best_accession=best_acc,
        is_cross_domain=any_cross_domain,
    )


# ============================================================================
# Pipeline entry point
# ============================================================================


def run_go_transfer(
    sequences: dict[str, str],
    query_embeddings: np.ndarray | None = None,
    query_ids: list[str] | None = None,
    swissprot_db: SwissProtReferenceDB | None = None,
    embedding_extractor: "EmbeddingExtractor | None" = None,
    k: int = DEFAULT_GO_TRANSFER_K,
    threshold: float = DEFAULT_GO_TRANSFER_THRESHOLD,
    reliability_threshold: float = DEFAULT_RELIABILITY_THRESHOLD,
    cross_domain_threshold: float = DEFAULT_CROSS_DOMAIN_THRESHOLD,
    cross_domain: bool = True,
    device: str = "auto",
    model_dir: Path | None = None,
    batch_size: int = 32,
    db_dir: Path | None = None,
    scope: str | None = None,
) -> dict[str, GOTransferResult]:
    """Run GO term transfer pipeline.

    If query_embeddings are provided (e.g., from triage step), they are
    reused to avoid redundant ProstT5 encoder calls.

    Args:
        sequences: Dict mapping protein IDs to AA sequences
        query_embeddings: Pre-computed (N, 1024) L2-normalized embeddings.
            If None, will be extracted from sequences.
        query_ids: Protein IDs corresponding to rows of query_embeddings.
            Required if query_embeddings is provided.
        swissprot_db: Pre-loaded Swiss-Prot reference DB. If None, will
            be loaded automatically.
        embedding_extractor: Pre-loaded EmbeddingExtractor. Only used if
            query_embeddings is None.
        k: Number of nearest neighbors
        threshold: Minimum cosine similarity
        reliability_threshold: Minimum RI to keep a GO term
        cross_domain_threshold: Minimum similarity for non-viral donors
        cross_domain: Whether to include non-viral donors
        device: Device for embedding extraction
        model_dir: Model cache directory
        batch_size: Batch size for embedding extraction
        db_dir: Database directory for Swiss-Prot DB
        scope: Swiss-Prot scope ("viral", "all", or None for auto)

    Returns:
        Dict mapping protein ID to GOTransferResult
    """
    # Load Swiss-Prot reference DB
    if swissprot_db is None:
        try:
            swissprot_db = SwissProtReferenceDB(scope=scope, db_dir=db_dir)
            swissprot_db.load()
        except FileNotFoundError:
            logger.info("Swiss-Prot reference DB not installed, skipping GO transfer")
            return {}

    if not swissprot_db._loaded:
        swissprot_db.load()

    # Get or extract query embeddings
    if query_embeddings is None or query_ids is None:
        if embedding_extractor is None:
            from vhold.features.embeddings import EmbeddingExtractor as _ExtractorCls

            embedding_extractor = _ExtractorCls(
                device=device,
                model_dir=model_dir,
                encoder_only=True,
            )

        logger.info("Extracting embeddings for GO transfer")
        query_ids, query_embeddings = embedding_extractor.extract_batch(
            sequences, batch_size=batch_size
        )

    # At this point query_embeddings and query_ids are guaranteed non-None
    assert query_embeddings is not None
    assert query_ids is not None

    # Search Swiss-Prot reference DB
    logger.info(
        f"Searching {swissprot_db.size} Swiss-Prot references "
        f"(k={k}, threshold={threshold})"
    )
    all_matches = swissprot_db.search(
        query_embeddings=query_embeddings,
        k=k,
        threshold=threshold,
    )

    # Transfer GO terms for each query
    results: dict[str, GOTransferResult] = {}
    n_with_terms = 0
    n_cross_domain = 0

    for i, qid in enumerate(query_ids):
        matches = all_matches[i]
        result = transfer_go_terms(
            query_id=qid,
            matches=matches,
            metadata=swissprot_db.metadata,
            threshold=threshold,
            reliability_threshold=reliability_threshold,
            cross_domain_threshold=cross_domain_threshold,
            cross_domain=cross_domain,
        )
        results[qid] = result

        if result.has_terms:
            n_with_terms += 1
        if result.is_cross_domain:
            n_cross_domain += 1

    logger.info(
        f"GO transfer: {n_with_terms}/{len(results)} proteins received GO terms, "
        f"{n_cross_domain} cross-domain matches"
    )

    return results
