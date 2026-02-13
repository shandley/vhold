"""Tests for the vhold align CLI command."""

from click.testing import CliRunner
import pytest

from vhold.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_align_help(runner):
    """Test that align --help shows all options."""
    result = runner.invoke(main, ["align", "--help"])
    assert result.exit_code == 0
    assert "Multiple structural alignment" in result.output
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--threads" in result.output
    assert "--device" in result.output
    assert "--fast" in result.output
    assert "--confidence-threshold" in result.output
    assert "--refine-iters" in result.output
    assert "--predictions-dir" in result.output
    assert "FoldMason" in result.output


def test_align_missing_input(runner):
    """Test that align fails without required -i option."""
    result = runner.invoke(main, ["align", "-o", "/tmp/test"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_align_missing_output(runner):
    """Test that align fails without required -o option."""
    result = runner.invoke(main, ["align", "-i", "test.fasta"])
    assert result.exit_code != 0


def test_align_listed_in_main_help(runner):
    """Test that align appears in main help."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "align" in result.output
