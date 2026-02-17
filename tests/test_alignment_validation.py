"""Tests for alignment validation of embedding triage matches."""

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vhold.features.alignment_validation import (
    FoldseekSequenceIndex,
    align_pair,
    validate_triage_matches,
)
from vhold.features.embeddings import EmbeddingMatch


# ---- Helpers ----

def _make_mmseqs_db(tmp_path: Path, db_name: str, sequences: dict[str, str]) -> Path:
    """Create a minimal MMseqs2-style database for testing.

    Creates the data file, data index, header file, and header index
    with the given protein sequences.

    Args:
        tmp_path: Temporary directory
        db_name: Database name (e.g., "bfvd")
        sequences: Dict mapping protein_id → amino acid sequence

    Returns:
        Path to the database (without extension)
    """
    db_path = tmp_path / db_name

    # Build data file and index
    data_entries = []
    header_entries = []
    data_offset = 0
    header_offset = 0

    data_bytes = b""
    header_bytes = b""

    for i, (protein_id, sequence) in enumerate(sequences.items()):
        # Data: sequence + null terminator
        seq_data = (sequence + "\n").encode("utf-8") + b"\x00"
        data_entries.append((i, data_offset, len(seq_data)))
        data_bytes += seq_data
        data_offset += len(seq_data)

        # Header: protein name + null terminator
        hdr_data = (protein_id + "\n").encode("utf-8") + b"\x00"
        header_entries.append((i, header_offset, len(hdr_data)))
        header_bytes += hdr_data
        header_offset += len(hdr_data)

    # Write data file
    with open(db_path, "wb") as f:
        f.write(data_bytes)

    # Write data index
    with open(db_path.with_suffix(".index"), "w") as f:
        for entry_id, offset, length in data_entries:
            f.write(f"{entry_id}\t{offset}\t{length}\n")

    # Write header file
    header_path = Path(str(db_path) + "_h")
    with open(header_path, "wb") as f:
        f.write(header_bytes)

    # Write header index
    header_index_path = Path(str(db_path) + "_h.index")
    with open(header_index_path, "w") as f:
        for entry_id, offset, length in header_entries:
            f.write(f"{entry_id}\t{offset}\t{length}\n")

    return db_path


def _make_embedding_match(
    query_id: str,
    target_id: str,
    source_db: str = "bfvd",
    similarity: float = 0.95,
) -> EmbeddingMatch:
    """Create an EmbeddingMatch for testing."""
    return EmbeddingMatch(
        query_id=query_id,
        target_id=target_id,
        source_db=source_db,
        similarity=similarity,
    )


# ---- Test FoldseekSequenceIndex ----

class TestFoldseekSequenceIndex:
    """Tests for reading sequences from MMseqs2 database format."""

    def test_load_and_get_sequence(self, tmp_path):
        """Should load index and retrieve correct sequences."""
        sequences = {
            "PROT_A": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            "PROT_B": "MKLFFLFCLLLCFSFACQASRKYEPMVSFGNELVASGFIEEKPQNAFYEALIAS",
            "PROT_C": "ACDEFGHIKLMNPQRSTVWY",
        }
        db_path = _make_mmseqs_db(tmp_path, "testdb", sequences)

        idx = FoldseekSequenceIndex(db_path)
        idx.load()

        assert idx.get_sequence("PROT_A") == sequences["PROT_A"]
        assert idx.get_sequence("PROT_B") == sequences["PROT_B"]
        assert idx.get_sequence("PROT_C") == sequences["PROT_C"]

    def test_missing_protein_returns_none(self, tmp_path):
        """Should return None for unknown protein IDs."""
        db_path = _make_mmseqs_db(tmp_path, "testdb", {"PROT_A": "MVLSPAD"})
        idx = FoldseekSequenceIndex(db_path)
        idx.load()

        assert idx.get_sequence("NONEXISTENT") is None

    def test_not_loaded_raises(self, tmp_path):
        """Should raise RuntimeError if load() not called."""
        db_path = _make_mmseqs_db(tmp_path, "testdb", {"PROT_A": "MVLSPAD"})
        idx = FoldseekSequenceIndex(db_path)

        with pytest.raises(RuntimeError, match="not loaded"):
            idx.get_sequence("PROT_A")

    def test_missing_db_raises(self, tmp_path):
        """Should raise FileNotFoundError for nonexistent DB path."""
        idx = FoldseekSequenceIndex(tmp_path / "nonexistent")

        with pytest.raises(FileNotFoundError):
            idx.load()

    def test_empty_database(self, tmp_path):
        """Should handle empty database gracefully."""
        db_path = _make_mmseqs_db(tmp_path, "testdb", {})
        idx = FoldseekSequenceIndex(db_path)
        idx.load()

        assert idx.get_sequence("anything") is None


# ---- Test align_pair ----

class TestAlignPair:
    """Tests for pairwise sequence alignment."""

    def test_identical_sequences(self):
        """Identical sequences should have identity ~1.0."""
        seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        identity, score = align_pair(seq, seq)
        assert identity == pytest.approx(1.0, abs=0.01)
        assert score > 0

    def test_very_similar_sequences(self):
        """Similar sequences should have high identity."""
        seq1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        seq2 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSK"  # Last residue changed
        identity, score = align_pair(seq1, seq2)
        assert identity > 0.9

    def test_unrelated_sequences(self):
        """Unrelated sequences should have low identity."""
        seq1 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        seq2 = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        identity, score = align_pair(seq1, seq2)
        assert identity < 0.3

    def test_different_lengths(self):
        """Should handle sequences of different lengths."""
        seq1 = "MVLSPADKTNVKAAWGKVGA"  # 20aa
        seq2 = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"  # 50aa
        identity, score = align_pair(seq1, seq2)
        # Identity normalized by aligned positions (not longer sequence)
        # The first 20 residues match, so identity reflects that
        assert 0 < identity <= 1.0

    def test_empty_sequences(self):
        """Should return 0 for empty sequences."""
        identity, score = align_pair("", "MVLSPAD")
        assert identity == 0.0
        assert score == 0.0

    def test_short_sequences(self):
        """Should work on very short sequences."""
        identity, score = align_pair("MVL", "MVL")
        assert identity == pytest.approx(1.0, abs=0.01)


# ---- Test validate_triage_matches ----

class TestValidateTriageMatches:
    """Tests for the full validation pipeline."""

    def test_high_identity_passes(self, tmp_path):
        """Matches with high sequence identity should pass validation."""
        # Create mock Foldseek DB
        ref_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        bfvd_dir = tmp_path / "bfvd"
        bfvd_dir.mkdir()
        _make_mmseqs_db(bfvd_dir, "bfvd", {"TARGET_1": ref_seq})

        matches = {
            "QUERY_1": [_make_embedding_match("QUERY_1", "TARGET_1", "bfvd")],
        }
        query_seqs = {
            "QUERY_1": ref_seq,  # Same sequence → identity ~1.0
        }

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=bfvd_dir / "bfvd"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "QUERY_1" in validated
        assert len(rejected) == 0

    def test_low_identity_rejected(self, tmp_path):
        """Matches with very low sequence identity should be rejected."""
        bfvd_dir = tmp_path / "bfvd"
        bfvd_dir.mkdir()
        _make_mmseqs_db(bfvd_dir, "bfvd", {
            "TARGET_1": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
        })

        matches = {
            "QUERY_1": [_make_embedding_match("QUERY_1", "TARGET_1", "bfvd")],
        }
        query_seqs = {
            # Completely unrelated sequence
            "QUERY_1": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        }

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=bfvd_dir / "bfvd"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "QUERY_1" not in validated
        assert "QUERY_1" in rejected

    def test_mixed_results(self, tmp_path):
        """Should validate some and reject others."""
        ref_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        bfvd_dir = tmp_path / "bfvd"
        bfvd_dir.mkdir()
        _make_mmseqs_db(bfvd_dir, "bfvd", {"TARGET_1": ref_seq, "TARGET_2": ref_seq})

        matches = {
            "GOOD": [_make_embedding_match("GOOD", "TARGET_1", "bfvd")],
            "BAD": [_make_embedding_match("BAD", "TARGET_2", "bfvd")],
        }
        query_seqs = {
            "GOOD": ref_seq,
            "BAD": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
        }

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=bfvd_dir / "bfvd"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "GOOD" in validated
        assert "BAD" in rejected
        assert len(validated) == 1
        assert len(rejected) == 1

    def test_missing_query_sequence_passes(self, tmp_path):
        """Matches for queries without sequences should pass through."""
        bfvd_dir = tmp_path / "bfvd"
        bfvd_dir.mkdir()
        _make_mmseqs_db(bfvd_dir, "bfvd", {"TARGET_1": "MVLSPAD"})

        matches = {
            "QUERY_1": [_make_embedding_match("QUERY_1", "TARGET_1", "bfvd")],
        }
        query_seqs = {}  # No sequences

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=bfvd_dir / "bfvd"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "QUERY_1" in validated
        assert len(rejected) == 0

    def test_missing_reference_passes(self, tmp_path):
        """Matches for unknown reference proteins should pass through."""
        bfvd_dir = tmp_path / "bfvd"
        bfvd_dir.mkdir()
        _make_mmseqs_db(bfvd_dir, "bfvd", {"OTHER": "MVLSPAD"})

        matches = {
            "QUERY_1": [_make_embedding_match("QUERY_1", "TARGET_UNKNOWN", "bfvd")],
        }
        query_seqs = {"QUERY_1": "MVLSPAD"}

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=bfvd_dir / "bfvd"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "QUERY_1" in validated
        assert len(rejected) == 0


class TestGracefulDegradation:
    """Tests for graceful handling of missing resources."""

    def test_no_db_dir_passes_all(self):
        """Should pass all matches when db_dir is None."""
        matches = {
            "Q1": [_make_embedding_match("Q1", "T1", "bfvd")],
        }
        validated, rejected = validate_triage_matches(
            matches, {"Q1": "MVLSPAD"}, db_dir=None,
        )
        assert "Q1" in validated
        assert len(rejected) == 0

    def test_empty_matches_returns_empty(self):
        """Should handle empty matches dict."""
        validated, rejected = validate_triage_matches({}, {}, db_dir=None)
        assert validated == {}
        assert rejected == []

    def test_no_foldseek_dbs_passes_all(self, tmp_path):
        """Should pass all matches when no Foldseek DBs exist."""
        matches = {
            "Q1": [_make_embedding_match("Q1", "T1", "bfvd")],
        }
        with patch("vhold.databases.install.get_bfvd_db_path", return_value=tmp_path / "nonexistent"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=tmp_path / "nonexistent2"):
                validated, rejected = validate_triage_matches(
                    matches, {"Q1": "MVLSPAD"}, tmp_path, min_identity=0.15,
                )

        assert "Q1" in validated
        assert len(rejected) == 0

    def test_viro3d_source_uses_viro3d_db(self, tmp_path):
        """Should use Viro3D DB for viro3d-sourced matches."""
        ref_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
        viro3d_dir = tmp_path / "viro3d" / "foldseekViro3D"
        viro3d_dir.mkdir(parents=True)
        _make_mmseqs_db(viro3d_dir, "viro3d", {"V3D_TARGET": ref_seq})

        matches = {
            "Q1": [_make_embedding_match("Q1", "V3D_TARGET", "viro3d", 0.95)],
        }
        query_seqs = {"Q1": ref_seq}

        with patch("vhold.databases.install.get_bfvd_db_path", return_value=tmp_path / "nonexistent"):
            with patch("vhold.databases.install.get_viro3d_db_path", return_value=viro3d_dir / "viro3d"):
                validated, rejected = validate_triage_matches(
                    matches, query_seqs, tmp_path, min_identity=0.15,
                )

        assert "Q1" in validated
        assert len(rejected) == 0
