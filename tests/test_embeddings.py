"""Tests for embedding-based triage."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vhold.features.embeddings import (
    EmbeddingDatabase,
    EmbeddingExtractor,
    EmbeddingMatch,
    TriageResult,
    build_embedding_consensus_results,
    triage_proteins,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_normalized(shape, rng=None):
    """Create L2-normalized random vectors."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = rng.standard_normal(shape).astype(np.float32)
    if arr.ndim == 1:
        return arr / np.linalg.norm(arr)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return arr / norms


def _make_test_npz(tmp_path, n_proteins=100, dim=1024, seed=42):
    """Create a test .npz embedding database file."""
    rng = np.random.default_rng(seed)
    embeddings = _make_normalized((n_proteins, dim), rng)
    protein_ids = np.array([f"PROT_{i:04d}" for i in range(n_proteins)])
    source_dbs = np.array(
        ["bfvd" if i < n_proteins // 2 else "viro3d" for i in range(n_proteins)]
    )
    path = tmp_path / "test_embeddings.npz"
    np.savez_compressed(path, embeddings=embeddings, protein_ids=protein_ids, source_dbs=source_dbs)
    return path, embeddings, protein_ids, source_dbs


def _make_mock_encoder_output(batch_size, seq_len, dim=1024):
    """Create a mock encoder output with last_hidden_state."""
    import torch

    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(batch_size, seq_len, dim)
    return mock_output


# ============================================================================
# TestEmbeddingExtractor
# ============================================================================


class TestEmbeddingExtractor:
    """Tests for ProstT5 encoder embedding extraction."""

    def test_extract_single_returns_correct_shape(self):
        """Mock model, verify output is (1024,) numpy array."""
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return proper tensors
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        # Mock encoder output
        mock_model.encoder.return_value = _make_mock_encoder_output(1, 10)

        extractor = EmbeddingExtractor(
            device="cpu", model=mock_model, tokenizer=mock_tokenizer
        )
        result = extractor.extract_single("ACDEFGHIK")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_extract_single_is_normalized(self):
        """Output vector should have L2 norm ~= 1.0."""
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_model.encoder.return_value = _make_mock_encoder_output(1, 10)

        extractor = EmbeddingExtractor(
            device="cpu", model=mock_model, tokenizer=mock_tokenizer
        )
        result = extractor.extract_single("ACDEFGHIK")

        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ~1.0, got {norm}"

    def test_extract_batch_returns_correct_shapes(self):
        """N sequences -> (N, 1024) matrix."""
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        n_seqs = 3
        seq_len = 10
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(n_seqs, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n_seqs, seq_len, dtype=torch.long),
        }
        mock_model.encoder.return_value = _make_mock_encoder_output(n_seqs, seq_len)

        extractor = EmbeddingExtractor(
            device="cpu", model=mock_model, tokenizer=mock_tokenizer
        )
        sequences = {"prot1": "ACDEF", "prot2": "GHIKL", "prot3": "MNPQR"}
        ids, embeddings = extractor.extract_batch(
            sequences, batch_size=10, show_progress=False
        )

        assert len(ids) == 3
        assert embeddings.shape == (3, 1024)

    def test_extract_batch_all_normalized(self):
        """All rows in batch output should be L2-normalized."""
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        n_seqs = 5
        seq_len = 8
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(n_seqs, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n_seqs, seq_len, dtype=torch.long),
        }
        mock_model.encoder.return_value = _make_mock_encoder_output(n_seqs, seq_len)

        extractor = EmbeddingExtractor(
            device="cpu", model=mock_model, tokenizer=mock_tokenizer
        )
        sequences = {f"prot{i}": "ACDEF" for i in range(n_seqs)}
        _, embeddings = extractor.extract_batch(
            sequences, batch_size=10, show_progress=False
        )

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ============================================================================
# TestEmbeddingDatabase
# ============================================================================


class TestEmbeddingDatabase:
    """Tests for reference embedding database."""

    def test_load_npz(self, tmp_path):
        """Load a small test .npz file."""
        path, _, _, _ = _make_test_npz(tmp_path, n_proteins=50)
        db = EmbeddingDatabase(path)
        db.load()

        assert db.size == 50
        assert db.embeddings.shape == (50, 1024)
        assert len(db.protein_ids) == 50
        assert len(db.source_dbs) == 50

    def test_search_finds_exact_match(self, tmp_path):
        """Query = reference should give similarity ~1.0."""
        path, embeddings, protein_ids, source_dbs = _make_test_npz(tmp_path, n_proteins=10)
        db = EmbeddingDatabase(path)
        db.load()

        # Use the first reference embedding as query
        query = embeddings[0:1]  # (1, 1024)
        results = db.search(query, query_ids=["test_query"], threshold=0.99)

        assert "test_query" in results
        assert len(results["test_query"]) >= 1
        best = results["test_query"][0]
        assert best.target_id == str(protein_ids[0])
        assert best.similarity > 0.99

    def test_search_threshold_filtering(self, tmp_path):
        """Below threshold -> not returned."""
        path, embeddings, _, _ = _make_test_npz(tmp_path, n_proteins=20)
        db = EmbeddingDatabase(path)
        db.load()

        # Create a random query unlikely to match any reference at 0.99
        rng = np.random.default_rng(999)
        query = _make_normalized((1, 1024), rng)
        results = db.search(query, query_ids=["rand_query"], threshold=0.99)

        # Very unlikely to have a match at 0.99 with random vectors
        assert "rand_query" not in results

    def test_search_top_k(self, tmp_path):
        """Returns at most k results per query."""
        path, embeddings, _, _ = _make_test_npz(tmp_path, n_proteins=50)
        db = EmbeddingDatabase(path)
        db.load()

        # Low threshold so we get many matches
        query = embeddings[0:1]
        results = db.search(query, query_ids=["q1"], threshold=0.0, top_k=3)

        assert "q1" in results
        assert len(results["q1"]) <= 3

    def test_search_empty_no_matches(self, tmp_path):
        """No matches above threshold returns empty dict."""
        path, _, _, _ = _make_test_npz(tmp_path, n_proteins=5)
        db = EmbeddingDatabase(path)
        db.load()

        # Threshold of 2.0 is impossible for cosine similarity
        query = _make_normalized((1, 1024))
        results = db.search(query, query_ids=["q1"], threshold=2.0)

        assert len(results) == 0

    def test_search_multiple_queries(self, tmp_path):
        """Multiple queries searched simultaneously."""
        path, embeddings, protein_ids, _ = _make_test_npz(tmp_path, n_proteins=20)
        db = EmbeddingDatabase(path)
        db.load()

        # Use first two reference embeddings as queries
        queries = embeddings[0:2]
        results = db.search(queries, query_ids=["q0", "q1"], threshold=0.99)

        assert "q0" in results
        assert "q1" in results
        assert results["q0"][0].target_id == str(protein_ids[0])
        assert results["q1"][0].target_id == str(protein_ids[1])

    def test_search_not_loaded_raises(self, tmp_path):
        """Searching before load() raises RuntimeError."""
        path, _, _, _ = _make_test_npz(tmp_path)
        db = EmbeddingDatabase(path)

        query = _make_normalized((1, 1024))
        with pytest.raises(RuntimeError, match="not loaded"):
            db.search(query, query_ids=["q1"])


# ============================================================================
# TestTriageProteins
# ============================================================================


class TestTriageProteins:
    """Tests for the full triage pipeline."""

    def _make_mocked_triage(self, sequences, n_ref=50, threshold=0.5, tmp_path=None):
        """Run triage with mocked extractor."""
        import torch

        # Create a temporary embedding DB
        if tmp_path is None:
            tmp_path = Path(tempfile.mkdtemp())
        path, ref_embeddings, _, _ = _make_test_npz(tmp_path, n_proteins=n_ref)
        db = EmbeddingDatabase(path)
        db.load()

        # Mock the model to return embeddings that partially match references
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        n_seqs = len(sequences)
        seq_len = 10

        # Make some queries match references, some not
        rng = np.random.default_rng(42)
        query_embeddings = []
        for i, sid in enumerate(sequences):
            if i < n_seqs // 2:
                # Match: use a reference embedding
                emb = ref_embeddings[i % n_ref]
            else:
                # No match: random embedding
                emb = _make_normalized((1024,), rng)
            query_embeddings.append(emb)

        query_matrix = np.vstack(query_embeddings)

        # Mock tokenizer and model to produce these specific embeddings
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(n_seqs, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n_seqs, seq_len, dtype=torch.long),
        }

        # The encoder needs to return embeddings that after mean-pooling
        # and normalization produce our target query_matrix.
        # Simplest: make hidden_state = desired embedding repeated across seq_len
        hidden = torch.from_numpy(query_matrix).unsqueeze(1).expand(-1, seq_len, -1).float()
        mock_output = MagicMock()
        mock_output.last_hidden_state = hidden
        mock_model.encoder.return_value = mock_output

        result, extractor = triage_proteins(
            sequences=sequences,
            embedding_db=db,
            device="cpu",
            threshold=threshold,
            batch_size=n_seqs,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        return result

    def test_all_proteins_partitioned(self, tmp_path):
        """len(matched) + len(unmatched) == len(input)."""
        sequences = {f"prot_{i}": "ACDEFGHIK" for i in range(6)}
        result = self._make_mocked_triage(sequences, threshold=0.99, tmp_path=tmp_path)

        total = len(result.matched) + len(result.unmatched)
        assert total == len(sequences)

    def test_stats_populated(self, tmp_path):
        """Stats dict has expected keys."""
        sequences = {f"prot_{i}": "ACDEFGHIK" for i in range(4)}
        result = self._make_mocked_triage(sequences, threshold=0.5, tmp_path=tmp_path)

        assert "total" in result.stats
        assert "matched" in result.stats
        assert "unmatched" in result.stats
        assert "threshold" in result.stats
        assert "match_rate" in result.stats
        assert result.stats["total"] == 4

    def test_returns_extractor(self, tmp_path):
        """triage_proteins returns the extractor for model reuse."""
        import torch

        path, _, _, _ = _make_test_npz(tmp_path, n_proteins=5)
        db = EmbeddingDatabase(path)
        db.load()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_model.encoder.return_value = _make_mock_encoder_output(1, 5)

        result, extractor = triage_proteins(
            sequences={"p1": "ACDEF"},
            embedding_db=db,
            device="cpu",
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert extractor is not None
        assert extractor.model is mock_model


# ============================================================================
# TestBuildEmbeddingConsensusResults
# ============================================================================


class TestBuildEmbeddingConsensusResults:
    """Tests for annotation transfer from embedding matches."""

    def test_consensus_result_fields(self):
        """primary_source, novelty, classification_source set correctly."""
        matches = {
            "query_1": [
                EmbeddingMatch(
                    query_id="query_1",
                    target_id="PROT_001",
                    source_db="viro3d",
                    similarity=0.98,
                )
            ]
        }

        # Mock the metadata lookup
        with patch(
            "vhold.features.embeddings._lookup_annotation",
            return_value={"description": "matrix protein", "gene": "M"},
        ):
            results = build_embedding_consensus_results(
                matches=matches,
                query_lengths={"query_1": 300},
            )

        assert "query_1" in results
        r = results["query_1"]
        assert r.primary_source == "embedding_viro3d"
        assert r.novelty == "embedding_match"
        assert r.classification_source == "embedding"
        assert r.primary_hit is None
        assert r.agreement == "single"

    def test_consensus_result_no_foldseek_hit(self):
        """primary_hit is None, output dict has empty Foldseek columns."""
        matches = {
            "q1": [
                EmbeddingMatch(
                    query_id="q1",
                    target_id="T1",
                    source_db="bfvd",
                    similarity=0.97,
                )
            ]
        }

        with patch(
            "vhold.features.embeddings._lookup_annotation",
            return_value={"description": "capsid protein"},
        ):
            results = build_embedding_consensus_results(
                matches=matches,
                query_lengths={"q1": 250},
            )

        r = results["q1"]
        d = r.to_dict()
        assert d["primary_target"] == ""
        assert d["primary_evalue"] == ""
        assert d["primary_identity"] == ""

    def test_similarity_maps_to_confidence(self):
        """High similarity should produce high confidence."""
        matches = {
            "q1": [
                EmbeddingMatch(
                    query_id="q1", target_id="T1",
                    source_db="bfvd", similarity=0.99,
                )
            ]
        }

        with patch(
            "vhold.features.embeddings._lookup_annotation",
            return_value={"description": "spike protein"},
        ):
            results = build_embedding_consensus_results(
                matches=matches,
                query_lengths={"q1": 1000},
            )

        assert results["q1"].confidence_level == "high"
        assert results["q1"].consensus_score > 0.8


# ============================================================================
# TestGracefulDegradation
# ============================================================================


class TestGracefulDegradation:
    """Tests for graceful error handling."""

    def test_embedding_database_file_not_found(self):
        """Non-existent path raises FileNotFoundError on load."""
        db = EmbeddingDatabase(Path("/nonexistent/path.npz"))
        with pytest.raises(FileNotFoundError):
            db.load()

    def test_triage_result_empty_when_no_matches(self, tmp_path):
        """When threshold is impossibly high, all proteins are unmatched."""
        import torch

        path, _, _, _ = _make_test_npz(tmp_path, n_proteins=5)
        db = EmbeddingDatabase(path)
        db.load()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        }
        mock_model.encoder.return_value = _make_mock_encoder_output(2, 5)

        result, _ = triage_proteins(
            sequences={"p1": "ACDEF", "p2": "GHIKL"},
            embedding_db=db,
            device="cpu",
            threshold=2.0,  # Impossible threshold
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert len(result.matched) == 0
        assert len(result.unmatched) == 2
