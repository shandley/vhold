"""Embedding-based triage for vhold.

Uses ProstT5's encoder to extract per-protein embeddings and compare
them against pre-computed reference embeddings. Proteins with close
matches skip the expensive decoder step and get annotation transferred
directly from the reference database.

The encoder produces a 1024-dimensional vector per protein in ~0.1s,
compared to minutes for the full decoder. Cosine similarity against
~436K reference embeddings takes ~100ms with numpy.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoModelForSeq2SeqLM

from vhold.features.prostt5 import prepare_prostt5_sequence, get_device
from vhold.utils.constants import (
    PROSTT5_MODEL_NAME,
    EMBEDDING_DIM,
    DEFAULT_TRIAGE_THRESHOLD,
    get_model_dir,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class EmbeddingMatch:
    """A match from the embedding database."""

    query_id: str
    target_id: str
    source_db: str  # "bfvd" or "viro3d"
    similarity: float  # cosine similarity (0-1)


@dataclass
class TriageResult:
    """Result of embedding-based triage."""

    matched: dict[str, list[EmbeddingMatch]] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ============================================================================
# Embedding extraction
# ============================================================================


class EmbeddingExtractor:
    """Extract per-protein embeddings from ProstT5 encoder.

    Uses only the encoder half of the T5 model -- a single forward pass
    producing a (seq_len, 1024) representation that is mean-pooled to
    a single 1024-dim vector per protein.

    Two loading modes:
    - encoder_only=True (default): Loads T5EncoderModel (~4.8 GB fp32),
      saving ~6.5 GB vs the full model. Best for standalone embedding
      extraction (e.g., building the reference database).
    - encoder_only=False: Loads the full AutoModelForSeq2SeqLM and uses
      its .encoder sub-module. This allows the loaded model to be reused
      by ProstT5Predictor for decoding, avoiding a double load.
    """

    def __init__(
        self,
        device: str = "auto",
        model_dir: Path | None = None,
        model: object | None = None,
        tokenizer: object | None = None,
        encoder_only: bool = True,
    ):
        """Initialize embedding extractor.

        Args:
            device: Device for inference ('auto', 'cuda', 'mps', 'cpu')
            model_dir: Directory to cache models
            model: Pre-loaded ProstT5 model (avoids double loading)
            tokenizer: Pre-loaded tokenizer
            encoder_only: If True, load only the encoder half (saves ~6.5GB).
                If False, load full model so it can be reused for decoding.
        """
        self.device = get_device(device)
        self.model_dir = model_dir or get_model_dir()
        self.model = model
        self.tokenizer = tokenizer
        self._encoder_only = encoder_only
        self._loaded = model is not None and tokenizer is not None

    def _run_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run the encoder forward pass.

        Handles both T5EncoderModel (direct call) and full
        AutoModelForSeq2SeqLM (.encoder sub-module).
        """
        if isinstance(self.model, T5EncoderModel):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def load_model(self) -> None:
        """Load the ProstT5 model and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading ProstT5 model for embedding extraction on {self.device}")

        model_dir = Path(self.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(
                PROSTT5_MODEL_NAME,
                cache_dir=model_dir,
                do_lower_case=False,
            )

        if self.model is None:
            if self._encoder_only:
                logger.info("Loading encoder-only model (T5EncoderModel, ~4.8 GB)")
                self.model = T5EncoderModel.from_pretrained(
                    PROSTT5_MODEL_NAME,
                    cache_dir=model_dir,
                )
            else:
                logger.info("Loading full model (AutoModelForSeq2SeqLM, ~11.3 GB)")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    PROSTT5_MODEL_NAME,
                    cache_dir=model_dir,
                )
            self.model = self.model.to(self.device)
            self.model = self.model.float()
            self.model.eval()

        self._loaded = True
        logger.info("Model loaded for embedding extraction")

    def extract_single(self, sequence: str) -> np.ndarray:
        """Extract a 1024-dim embedding for a single protein sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            L2-normalized numpy array of shape (1024,)
        """
        self.load_model()

        prepared = prepare_prostt5_sequence(sequence)
        inputs = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_output = self._run_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Mean pool over sequence length, respecting attention mask
        hidden = encoder_output.last_hidden_state  # (1, seq_len, 1024)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, 1024)

        embedding = pooled[0].cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_batch(
        self,
        sequences: dict[str, str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        """Extract embeddings for multiple protein sequences.

        Args:
            sequences: Dict mapping sequence IDs to AA sequences
            batch_size: Number of sequences per batch
            show_progress: Show progress bar

        Returns:
            Tuple of (protein_ids, embeddings_matrix) where
            embeddings_matrix has shape (N, 1024), L2-normalized rows
        """
        self.load_model()

        ids = list(sequences.keys())
        seqs = [sequences[sid] for sid in ids]
        all_embeddings = []

        iterator = range(0, len(seqs), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings", unit="batch")

        for i in iterator:
            batch_seqs = seqs[i : i + batch_size]
            prepared = [prepare_prostt5_sequence(s) for s in batch_seqs]

            inputs = self.tokenizer(
                prepared,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                encoder_output = self._run_encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            hidden = encoder_output.last_hidden_state  # (B, seq_len, 1024)
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, seq_len, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, 1024)

            batch_np = pooled.cpu().numpy()

            # L2 normalize each row
            norms = np.linalg.norm(batch_np, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            batch_np = batch_np / norms

            all_embeddings.append(batch_np)

        embeddings = np.vstack(all_embeddings)  # (N, 1024)
        return ids, embeddings


# ============================================================================
# Embedding database
# ============================================================================


class EmbeddingDatabase:
    """Pre-computed reference embeddings for triage.

    Stores L2-normalized embeddings for all BFVD + Viro3D reference
    proteins. Search is a simple matrix multiply (dot product = cosine
    similarity for normalized vectors).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.embeddings: np.ndarray | None = None  # (N, 1024) float32
        self.protein_ids: np.ndarray | None = None  # (N,) strings
        self.source_dbs: np.ndarray | None = None  # (N,) strings
        self._loaded = False

    def load(self) -> None:
        """Load .npz file into memory."""
        logger.info(f"Loading embedding database from {self.db_path}")
        data = np.load(self.db_path, allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.protein_ids = data["protein_ids"]
        self.source_dbs = data["source_dbs"]
        self._loaded = True
        logger.info(f"Loaded {len(self.protein_ids)} reference embeddings")

    @property
    def size(self) -> int:
        """Number of reference proteins."""
        return len(self.protein_ids) if self.protein_ids is not None else 0

    def search(
        self,
        query_embeddings: np.ndarray,
        query_ids: list[str],
        threshold: float = DEFAULT_TRIAGE_THRESHOLD,
        top_k: int = 1,
    ) -> dict[str, list[EmbeddingMatch]]:
        """Search query embeddings against the reference database.

        Both query and reference embeddings must be L2-normalized so
        that dot product equals cosine similarity.

        Args:
            query_embeddings: (N, 1024) L2-normalized query matrix
            query_ids: list of N query protein IDs
            threshold: Minimum cosine similarity for a match
            top_k: Maximum matches per query

        Returns:
            Dict mapping query_id to list of EmbeddingMatch (only for
            queries with at least one match above threshold)
        """
        if not self._loaded:
            raise RuntimeError("Embedding database not loaded. Call load() first.")

        # Similarity matrix: (N_query, N_ref) via dot product
        similarities = query_embeddings @ self.embeddings.T  # (N, M)

        results: dict[str, list[EmbeddingMatch]] = {}

        for i, query_id in enumerate(query_ids):
            row = similarities[i]  # (M,)

            # Find indices above threshold
            above_mask = row >= threshold
            if not above_mask.any():
                continue

            above_indices = np.where(above_mask)[0]
            above_scores = row[above_indices]

            # Sort by similarity descending, take top_k
            sorted_order = np.argsort(-above_scores)[:top_k]
            top_indices = above_indices[sorted_order]
            top_scores = above_scores[sorted_order]

            matches = []
            for idx, score in zip(top_indices, top_scores):
                matches.append(
                    EmbeddingMatch(
                        query_id=query_id,
                        target_id=str(self.protein_ids[idx]),
                        source_db=str(self.source_dbs[idx]),
                        similarity=float(score),
                    )
                )

            results[query_id] = matches

        return results


# ============================================================================
# Triage orchestration
# ============================================================================


def triage_proteins(
    sequences: dict[str, str],
    embedding_db: EmbeddingDatabase,
    device: str = "auto",
    model_dir: Path | None = None,
    threshold: float = DEFAULT_TRIAGE_THRESHOLD,
    batch_size: int = 32,
    model: object | None = None,
    tokenizer: object | None = None,
    encoder_only: bool = False,
) -> tuple[TriageResult, EmbeddingExtractor]:
    """Run embedding-based triage on input sequences.

    Extracts embeddings for all query proteins using the ProstT5 encoder,
    searches against the reference database, and partitions proteins into
    matched (skip decoder) and unmatched (need full pipeline).

    Args:
        sequences: Dict mapping sequence IDs to AA sequences
        embedding_db: Loaded embedding database
        device: Device for inference
        model_dir: Model cache directory
        threshold: Cosine similarity threshold
        batch_size: Batch size for embedding extraction
        model: Pre-loaded ProstT5 model
        tokenizer: Pre-loaded tokenizer
        encoder_only: If True, load only encoder (saves memory but model
            can't be reused for decoding). Default False for pipeline use.

    Returns:
        Tuple of (TriageResult, EmbeddingExtractor) -- the extractor is
        returned so its loaded model can be reused by ProstT5Predictor
    """
    extractor = EmbeddingExtractor(
        device=device,
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        encoder_only=encoder_only,
    )

    # Extract embeddings (encoder only, fast)
    ids, embeddings = extractor.extract_batch(
        sequences, batch_size=batch_size, show_progress=True
    )

    # Search against reference database
    matches = embedding_db.search(
        query_embeddings=embeddings,
        query_ids=ids,
        threshold=threshold,
    )

    # Partition
    matched_ids = set(matches.keys())
    unmatched = [sid for sid in ids if sid not in matched_ids]

    stats = {
        "total": len(ids),
        "matched": len(matched_ids),
        "unmatched": len(unmatched),
        "threshold": threshold,
        "match_rate": len(matched_ids) / len(ids) if ids else 0,
    }

    result = TriageResult(
        matched=matches,
        unmatched=unmatched,
        stats=stats,
    )

    return result, extractor


# ============================================================================
# Annotation transfer for embedding matches
# ============================================================================


def build_embedding_consensus_results(
    matches: dict[str, list[EmbeddingMatch]],
    query_lengths: dict[str, int],
    db_dir: Path | None = None,
) -> dict:
    """Build ConsensusResult objects for embedding-matched proteins.

    Looks up the matched reference protein in BFVD/Viro3D metadata and
    transfers annotation, building a ConsensusResult with embedding-specific
    provenance fields.

    Args:
        matches: Dict mapping query_id to list of EmbeddingMatch
        query_lengths: Dict mapping query IDs to sequence lengths
        db_dir: Database directory for metadata lookup

    Returns:
        Dict mapping query_id to ConsensusResult
    """
    from vhold.results.consensus import ConsensusResult
    from vhold.results.categories import classify_protein

    # Lazy load metadata
    bfvd_metadata = None
    viro3d_metadata = None
    viro3d_annotations_df = None
    viro3d_expansion = None

    results = {}

    for query_id, match_list in matches.items():
        if not match_list:
            continue

        best_match = match_list[0]  # Already sorted by similarity
        length = query_lengths.get(query_id, 0)

        # Look up annotation from reference database
        annotation = _lookup_annotation(
            best_match.target_id,
            best_match.source_db,
            db_dir,
            _metadata_cache={},
        )

        # Classify using all available evidence
        description = annotation.get("description", "hypothetical protein")
        gene = annotation.get("gene")

        # Build evidence from enriched annotation fields
        from vhold.results.categories import AnnotationEvidence
        evidence = AnnotationEvidence(
            pfam=annotation.get("pfam"),
            go_bp=annotation.get("go_bp"),
            go_mf=annotation.get("go_mf"),
            superfamily=annotation.get("superfamily"),
        )
        category = classify_protein(description, gene, evidence=evidence)

        # Map similarity to consensus score (0.95-1.0 → 0.85-1.0)
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
            primary_hit=None,  # No FoldseekHit for embedding matches
            primary_annotation=annotation,
            primary_source=f"embedding_{best_match.source_db}",
            consensus_score=consensus_score,
            confidence_level=confidence,
            agreement="single",
            functional_category=category,
            classification_source="embedding",
            novelty="embedding_match",
            structure_quality_score=1.0,
            structure_quality_source="none",
        )

        results[query_id] = result

    logger.info(f"Built {len(results)} embedding-matched annotations")
    return results


# Metadata cache shared across calls within a single pipeline run
_metadata_state: dict = {}


def _lookup_annotation(
    target_id: str,
    source_db: str,
    db_dir: Path | None,
    _metadata_cache: dict,
) -> dict:
    """Look up annotation for a reference protein.

    Args:
        target_id: Reference protein ID
        source_db: Database name ("bfvd" or "viro3d")
        db_dir: Database directory
        _metadata_cache: Shared cache dict to avoid reloading metadata

    Returns:
        Annotation dict with description, gene, organism, etc.
    """
    import pandas as pd

    if source_db == "bfvd":
        if "bfvd_metadata" not in _metadata_state:
            try:
                from vhold.databases.bfvd import load_bfvd_metadata
                _metadata_state["bfvd_metadata"] = load_bfvd_metadata(db_dir)
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Could not load BFVD metadata: {e}")
                _metadata_state["bfvd_metadata"] = pd.DataFrame()

        if "bfvd_enriched" not in _metadata_state:
            try:
                from vhold.databases.bfvd import load_bfvd_enriched_metadata
                _metadata_state["bfvd_enriched"] = load_bfvd_enriched_metadata(db_dir)
            except Exception as e:
                logger.debug(f"Enriched BFVD metadata not available: {e}")
                _metadata_state["bfvd_enriched"] = {}

        metadata = _metadata_state["bfvd_metadata"]
        enriched = _metadata_state["bfvd_enriched"]
        if len(metadata) > 0:
            from vhold.databases.bfvd import get_bfvd_annotation
            ann = get_bfvd_annotation(target_id, metadata, enriched_metadata=enriched)
            if ann:
                return ann

    elif source_db == "viro3d":
        if "viro3d_metadata" not in _metadata_state:
            try:
                from vhold.databases.viro3d import load_viro3d_metadata
                _metadata_state["viro3d_metadata"] = load_viro3d_metadata(db_dir)
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Could not load Viro3D metadata: {e}")
                _metadata_state["viro3d_metadata"] = pd.DataFrame()

        if "viro3d_annotations" not in _metadata_state:
            try:
                from vhold.databases.viro3d import load_viro3d_annotations
                _metadata_state["viro3d_annotations"] = load_viro3d_annotations(db_dir)
            except (FileNotFoundError, Exception):
                _metadata_state["viro3d_annotations"] = pd.DataFrame()

        metadata = _metadata_state["viro3d_metadata"]
        annotations_df = _metadata_state["viro3d_annotations"]
        if len(metadata) > 0:
            from vhold.databases.viro3d import get_viro3d_annotation
            ann = get_viro3d_annotation(
                target_id,
                metadata,
                annotations_df if len(annotations_df) > 0 else None,
            )
            if ann:
                return ann

    return {"description": "hypothetical protein"}
