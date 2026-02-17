"""Tests for vhold export CLI command."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vhold.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestExportCli:
    def test_export_help(self, runner):
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "anvio" in result.output
        assert "vcontact2" in result.output
        assert "dramv" in result.output
        assert "gff3" in result.output

    def test_export_in_main_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "export" in result.output

    def test_export_missing_input(self, runner):
        result = runner.invoke(main, ["export", "-o", "/tmp/test"])
        assert result.exit_code != 0

    def test_export_missing_output(self, runner, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("query_id\tquery_length\n")
        result = runner.invoke(main, ["export", "-i", str(tsv)])
        assert result.exit_code != 0

    def test_export_calls_run_export(self, runner, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("query_id\tquery_length\tdescription\n")
        out = tmp_path / "exports"

        with patch("vhold.subcommands.export.run_export") as mock:
            runner.invoke(
                main, ["export", "-i", str(tsv), "-o", str(out)],
            )

        mock.assert_called_once()

    def test_export_specific_formats(self, runner, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("query_id\tquery_length\tdescription\n")
        out = tmp_path / "exports"

        with patch("vhold.subcommands.export.run_export") as mock:
            runner.invoke(
                main,
                ["export", "-i", str(tsv), "-o", str(out),
                 "-f", "anvio", "-f", "dramv"],
            )

        call_kwargs = mock.call_args
        assert "anvio" in call_kwargs.kwargs["formats"]
        assert "dramv" in call_kwargs.kwargs["formats"]


class TestRunWithExportFormat:
    def test_export_format_in_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--export-format" in result.output

    def test_export_format_passed_to_pipeline(self, runner, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            runner.invoke(
                main,
                ["run", "-i", str(fasta), "-o", str(out),
                 "--export-format", "anvio", "--export-format", "dramv"],
            )

        call_kwargs = mock_run.call_args.kwargs
        assert "export_formats" in call_kwargs
        assert "anvio" in call_kwargs["export_formats"]
        assert "dramv" in call_kwargs["export_formats"]

    def test_no_export_format_passes_none(self, runner, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">prot1\nMKKLLGG\n")
        out = tmp_path / "out"

        with patch("vhold.subcommands.run.run_pipeline") as mock_run:
            runner.invoke(
                main, ["run", "-i", str(fasta), "-o", str(out)],
            )

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["export_formats"] is None
