"""Export subcommand — convert vHold results to metagenomic tool formats."""

from pathlib import Path

from vhold.results.export import export_all_formats, load_results_tsv
from vhold.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def run_export(
    input_tsv: str | Path,
    output_dir: str | Path,
    formats: list[str] | None = None,
    prefix: str = "vhold",
) -> None:
    """Export vHold results to metagenomic tool formats.

    Args:
        input_tsv: Path to vHold results TSV
        output_dir: Output directory for exported files
        formats: List of format names (default: all)
        prefix: Output file prefix
    """
    setup_logging()

    input_path = Path(input_tsv)
    output_path = Path(output_dir)

    logger.info("=" * 60)
    logger.info("vHold export — Metagenomic format conversion")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Formats: {formats or 'all'}")
    logger.info("")

    df = load_results_tsv(input_path)
    logger.info(f"Loaded {len(df)} proteins from {input_path}")

    output_files = export_all_formats(
        results=df,
        output_dir=output_path,
        formats=formats,
        prefix=prefix,
    )

    logger.info("")
    logger.info("Export complete!")
    for fmt, path in output_files.items():
        if isinstance(path, dict):
            for sub_fmt, sub_path in path.items():
                logger.info(f"  {fmt}/{sub_fmt}: {sub_path}")
        else:
            logger.info(f"  {fmt}: {path}")
