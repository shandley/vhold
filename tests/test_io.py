"""Tests for vhold IO modules."""

import tempfile
from pathlib import Path

import pytest

from vhold.io.fasta import read_fasta, write_fasta, ProteinRecord


def test_read_fasta(small_fasta):
    """Test reading FASTA file."""
    records = read_fasta(small_fasta)

    assert len(records) == 3
    assert "test_protein_1" in records
    assert "test_protein_2" in records
    assert "test_protein_3" in records

    # Check first record
    rec1 = records["test_protein_1"]
    assert rec1.id == "test_protein_1"
    assert rec1.sequence.startswith("MVLSPAD")
    assert rec1.length > 0


def test_write_fasta():
    """Test writing FASTA file."""
    records = {
        "seq1": ProteinRecord(id="seq1", sequence="MVLSPAD", description="Test 1"),
        "seq2": ProteinRecord(id="seq2", sequence="MKFLIL", description="Test 2"),
    }

    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
        output_path = Path(f.name)

    try:
        write_fasta(records, output_path)

        # Read back and verify
        read_back = read_fasta(output_path)
        assert len(read_back) == 2
        assert read_back["seq1"].sequence == "MVLSPAD"
        assert read_back["seq2"].sequence == "MKFLIL"
    finally:
        output_path.unlink()


def test_protein_record():
    """Test ProteinRecord dataclass."""
    rec = ProteinRecord(
        id="test",
        sequence="MVLSPADKTNVK",
        description="Test protein"
    )

    assert rec.id == "test"
    assert rec.length == 12
    assert rec.to_fasta().startswith(">test")
