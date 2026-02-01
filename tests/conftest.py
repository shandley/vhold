"""Pytest configuration and fixtures for vhold tests."""

import pytest
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def test_data_dir():
    """Return the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def small_fasta(test_data_dir):
    """Return path to small test FASTA file."""
    return test_data_dir / "small_viral.fasta"
