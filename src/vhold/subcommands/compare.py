"""Compare subcommand for vhold - Foldseek database search."""

from pathlib import Path

from vhold.features.foldseek import search_databases, merge_results
from vhold.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_compare(
    predictions_dir: str | Path,
    output_dir: str | Path,
    database_dir: str | Path | None = None,
    databases: str = "all",
    threads: int = 4,
    evalue: float = 1e-3,
    sensitivity: float = 9.5,
    max_seqs: int = 1000,
) -> None:
    """Run Foldseek search against reference databases.

    Args:
        predictions_dir: Directory containing 3Di predictions
        output_dir: Output directory for results
        database_dir: Database directory
        databases: Which databases to search ('all', 'bfvd', 'viro3d')
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Foldseek sensitivity
        max_seqs: Maximum sequences to report per query
    """
    setup_logging()

    pred_path = Path(predictions_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_dir = Path(database_dir) if database_dir else None

    logger.info(f"Predictions: {pred_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Databases: {databases}")

    # Find 3Di FASTA file
    # Prefer masked version if available
    query_fasta = pred_path / "3di_sequences_masked.fasta"
    if not query_fasta.exists():
        query_fasta = pred_path / "3di_sequences.fasta"

    if not query_fasta.exists():
        # Try to find any fasta file
        fasta_files = list(pred_path.glob("*.fasta")) + list(pred_path.glob("*.fa"))
        if fasta_files:
            query_fasta = fasta_files[0]
        else:
            raise FileNotFoundError(
                f"No 3Di FASTA file found in {pred_path}. "
                "Run 'vhold predict' first to generate 3Di sequences."
            )

    logger.info(f"Query file: {query_fasta}")

    # Run searches
    results = search_databases(
        query_fasta=query_fasta,
        output_dir=output_path,
        databases=databases,
        db_dir=db_dir,
        threads=threads,
        evalue=evalue,
        sensitivity=sensitivity,
        max_seqs=max_seqs,
    )

    # Merge and save combined results
    merged = merge_results(results, keep_best=False)
    merged_output = output_path / "all_hits.tsv"
    merged.to_csv(merged_output, sep="\t", index=False)
    logger.info(f"Wrote all hits to {merged_output}")

    # Also save best hits per query
    best = merge_results(results, keep_best=True)
    best_output = output_path / "best_hits.tsv"
    best.to_csv(best_output, sep="\t", index=False)
    logger.info(f"Wrote best hits to {best_output}")

    # Summary
    total_queries = len(best)
    for db_name, df in results.items():
        logger.info(f"{db_name}: {len(df)} hits")

    logger.info(f"Search complete: {total_queries} queries with hits")
