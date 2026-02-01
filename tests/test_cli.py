"""Tests for vhold CLI."""

import pytest
from click.testing import CliRunner

from vhold import __version__
from vhold.cli import main


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


def test_version(runner):
    """Test --version flag."""
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help(runner):
    """Test --help flag."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "vhold" in result.output
    assert "structural homology" in result.output.lower()


def test_install_help(runner):
    """Test install --help."""
    result = runner.invoke(main, ["install", "--help"])
    assert result.exit_code == 0
    assert "download" in result.output.lower() or "install" in result.output.lower()


def test_predict_help(runner):
    """Test predict --help."""
    result = runner.invoke(main, ["predict", "--help"])
    assert result.exit_code == 0
    assert "prostt5" in result.output.lower() or "3di" in result.output.lower()


def test_compare_help(runner):
    """Test compare --help."""
    result = runner.invoke(main, ["compare", "--help"])
    assert result.exit_code == 0
    assert "foldseek" in result.output.lower() or "search" in result.output.lower()


def test_run_help(runner):
    """Test run --help."""
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "pipeline" in result.output.lower()
