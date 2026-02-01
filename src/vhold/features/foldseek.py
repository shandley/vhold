"""Foldseek wrapper for structural similarity search."""

import tempfile
from pathlib import Path

import pandas as pd

from vhold.databases.install import get_bfvd_db_path, get_viro3d_db_path
from vhold.utils.constants import (
    FOLDSEEK_OUTPUT_FORMAT,
    FOLDSEEK_OUTPUT_COLUMNS,
    DEFAULT_EVALUE,
    DEFAULT_SENSITIVITY,
    DEFAULT_MAX_SEQS,
)
from vhold.utils.external import ExternalTool, check_foldseek
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


def run_foldseek_search(
    query_fasta: Path,
    output_path: Path,
    database_path: Path,
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
    tmp_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Foldseek easy-search against a database.

    Args:
        query_fasta: Path to query FASTA with 3Di sequences
        output_path: Path for output file
        database_path: Path to Foldseek database
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity (default 9.5)
        max_seqs: Maximum target sequences
        tmp_dir: Temporary directory for Foldseek

    Returns:
        DataFrame with search results
    """
    # Check Foldseek availability
    available, version = check_foldseek()
    if not available:
        raise FileNotFoundError(
            "Foldseek not found in PATH. Please install Foldseek: "
            "https://github.com/steineggerlab/foldseek"
        )
    logger.info(f"Using Foldseek {version}")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory if not provided
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="foldseek_"))
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    foldseek = ExternalTool("foldseek")

    args = [
        "easy-search",
        str(query_fasta),
        str(database_path),
        str(output_path),
        str(tmp_dir),
        "--threads", str(threads),
        "-e", str(evalue),
        "-s", str(sensitivity),
        "--max-seqs", str(max_seqs),
        "--format-output", FOLDSEEK_OUTPUT_FORMAT,
        "--exhaustive-search", "1",  # More thorough search
    ]

    logger.info(f"Running Foldseek search against {database_path.name}")
    logger.debug(f"Query: {query_fasta}, Output: {output_path}")

    try:
        result = foldseek.run(args, check=True)
        logger.debug(f"Foldseek stdout: {result.stdout}")
    except Exception as e:
        logger.error(f"Foldseek search failed: {e}")
        raise

    # Parse results
    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(
            output_path,
            sep="\t",
            header=None,
            names=FOLDSEEK_OUTPUT_COLUMNS,
        )
        logger.info(f"Found {len(df)} hits")
        return df
    else:
        logger.warning("No hits found")
        return pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)


def search_databases(
    query_fasta: Path,
    output_dir: Path,
    databases: str = "all",
    db_dir: Path | None = None,
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
) -> dict[str, pd.DataFrame]:
    """Search against BFVD and/or Viro3D databases.

    Args:
        query_fasta: Path to query FASTA with 3Di sequences
        output_dir: Output directory for results
        databases: Which databases to search ('all', 'bfvd', 'viro3d')
        db_dir: Database directory
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity
        max_seqs: Maximum sequences per query

    Returns:
        Dict mapping database name to results DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Search BFVD
    if databases in ("all", "bfvd"):
        bfvd_db = get_bfvd_db_path(db_dir)
        if bfvd_db.parent.exists():
            logger.info("Searching BFVD database...")
            bfvd_output = output_dir / "bfvd_hits.tsv"
            try:
                results["bfvd"] = run_foldseek_search(
                    query_fasta=query_fasta,
                    output_path=bfvd_output,
                    database_path=bfvd_db,
                    threads=threads,
                    evalue=evalue,
                    sensitivity=sensitivity,
                    max_seqs=max_seqs,
                    tmp_dir=output_dir / "tmp_bfvd",
                )
            except Exception as e:
                logger.error(f"BFVD search failed: {e}")
                results["bfvd"] = pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)
        else:
            logger.warning(f"BFVD database not found at {bfvd_db}")
            results["bfvd"] = pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)

    # Search Viro3D
    if databases in ("all", "viro3d"):
        viro3d_db = get_viro3d_db_path(db_dir)
        if viro3d_db.parent.exists():
            logger.info("Searching Viro3D database...")
            viro3d_output = output_dir / "viro3d_hits.tsv"
            try:
                results["viro3d"] = run_foldseek_search(
                    query_fasta=query_fasta,
                    output_path=viro3d_output,
                    database_path=viro3d_db,
                    threads=threads,
                    evalue=evalue,
                    sensitivity=sensitivity,
                    max_seqs=max_seqs,
                    tmp_dir=output_dir / "tmp_viro3d",
                )
            except Exception as e:
                logger.error(f"Viro3D search failed: {e}")
                results["viro3d"] = pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)
        else:
            logger.warning(f"Viro3D database not found at {viro3d_db}")
            results["viro3d"] = pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)

    return results


def merge_results(
    results: dict[str, pd.DataFrame],
    keep_best: bool = True,
) -> pd.DataFrame:
    """Merge results from multiple databases.

    Args:
        results: Dict mapping database name to results DataFrame
        keep_best: Keep only best hit per query (by e-value)

    Returns:
        Merged DataFrame with source column
    """
    dfs = []

    for db_name, df in results.items():
        if len(df) > 0:
            df = df.copy()
            df["source_db"] = db_name
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS + ["source_db"])

    merged = pd.concat(dfs, ignore_index=True)

    if keep_best and len(merged) > 0:
        # Sort by query, then e-value (ascending)
        merged = merged.sort_values(["query", "evalue"])
        # Keep best hit per query
        merged = merged.drop_duplicates(subset=["query"], keep="first")

    return merged
