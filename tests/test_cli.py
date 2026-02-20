"""Tests for vhold CLI."""

from unittest.mock import patch

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


def test_run_help_shows_llm_option(runner):
    """--llm/--no-llm should appear in run help."""
    result = runner.invoke(main, ["run", "--help"])
    assert "--llm" in result.output
    assert "--no-llm" in result.output


class TestLlmAutoDetection:
    """Tests for --llm/--no-llm auto-detection based on ANTHROPIC_API_KEY."""

    def test_llm_auto_enabled_when_api_key_set(self, runner, tmp_path):
        """LLM should auto-enable when ANTHROPIC_API_KEY is present."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            result = runner.invoke(
                main,
                ["run", "-i", str(fasta), "-o", str(out)],
                env={"ANTHROPIC_API_KEY": "sk-test-key"},
            )

        assert "LLM classification auto-enabled" in result.output
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_classify"] is True

    def test_llm_not_auto_enabled_without_api_key(self, runner, tmp_path):
        """LLM should not auto-enable when ANTHROPIC_API_KEY is absent."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            result = runner.invoke(
                main,
                ["run", "-i", str(fasta), "-o", str(out)],
                env={"ANTHROPIC_API_KEY": ""},
            )

        assert "LLM classification auto-enabled" not in result.output
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_classify"] is False

    def test_no_llm_flag_overrides_api_key(self, runner, tmp_path):
        """--no-llm should disable LLM even when API key is set."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            result = runner.invoke(
                main,
                ["run", "-i", str(fasta), "-o", str(out), "--no-llm"],
                env={"ANTHROPIC_API_KEY": "sk-test-key"},
            )

        assert "LLM classification auto-enabled" not in result.output
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_classify"] is False

    def test_llm_flag_enables_without_api_key(self, runner, tmp_path):
        """--llm should explicitly enable LLM classification."""
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            result = runner.invoke(
                main,
                ["run", "-i", str(fasta), "-o", str(out), "--llm"],
                env={"ANTHROPIC_API_KEY": ""},
            )

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["llm_classify"] is True
