"""Foldseek wrapper for structural similarity search."""

import struct
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
from vhold.io.fasta import read_fasta

logger = get_logger(__name__)


def create_query_db(
    aa_fasta: Path,
    three_di_fasta: Path,
    output_db: Path,
    threads: int = 4,
) -> Path:
    """Create a Foldseek query database with pre-computed 3Di sequences.

    This creates a proper Foldseek database by:
    1. Creating a base database from AA sequences
    2. Replacing the _ss (3Di) sequences with our pre-computed ones

    Args:
        aa_fasta: Path to amino acid FASTA file
        three_di_fasta: Path to 3Di FASTA file (from ProstT5)
        output_db: Path for output database (without extension)
        threads: Number of threads

    Returns:
        Path to the created database
    """
    foldseek = ExternalTool("foldseek")
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Create base database from AA sequences
    # Use createdb with a fake PDB to create the database structure
    # Then we'll overwrite the _ss file with our 3Di sequences

    # First, read sequences to get the order
    aa_seqs = read_fasta(aa_fasta)
    three_di_seqs = read_fasta(three_di_fasta)

    # Write the database files manually
    # Foldseek database format:
    # - DB: concatenated sequences, null-terminated
    # - DB.index: id\toffset\tlength\n for each sequence
    # - DB.dbtype: single byte indicating type
    # - DB_h: concatenated headers, null-terminated
    # - DB_h.index: same format as above
    # - DB_h.dbtype: header type
    # - DB_ss: 3Di sequences, null-terminated
    # - DB_ss.index: same format
    # - DB_ss.dbtype: 3Di type

    # Write sequence database
    _write_foldseek_db(
        sequences={seq_id: rec.sequence for seq_id, rec in aa_seqs.items()},
        db_path=output_db,
        dbtype=0,  # Amino acid sequences
    )

    # Write header database
    _write_foldseek_db(
        sequences={seq_id: seq_id for seq_id in aa_seqs.keys()},
        db_path=Path(str(output_db) + "_h"),
        dbtype=12,  # Headers
    )

    # Write 3Di database
    _write_foldseek_db(
        sequences={seq_id: rec.sequence for seq_id, rec in three_di_seqs.items()},
        db_path=Path(str(output_db) + "_ss"),
        dbtype=0,  # Sequence type
    )

    # Write lookup file
    lookup_path = Path(str(output_db) + ".lookup")
    with open(lookup_path, "w") as f:
        for i, seq_id in enumerate(aa_seqs.keys()):
            f.write(f"{i}\t{seq_id}\t0\n")

    # Write source file (empty)
    source_path = Path(str(output_db) + ".source")
    source_path.touch()

    logger.debug(f"Created query database at {output_db}")
    return output_db


def _write_foldseek_db(
    sequences: dict[str, str],
    db_path: Path,
    dbtype: int,
) -> None:
    """Write a Foldseek database file.

    Args:
        sequences: Dict of id -> sequence
        db_path: Path for database (without extensions)
        dbtype: Database type (0=seq, 12=header)
    """
    data_path = db_path
    index_path = Path(str(db_path) + ".index")
    dbtype_path = Path(str(db_path) + ".dbtype")

    # Write data file and build index
    offset = 0
    index_lines = []

    with open(data_path, "wb") as f:
        for i, (seq_id, seq) in enumerate(sequences.items()):
            # Write sequence with null terminator
            seq_bytes = (seq + "\n").encode("utf-8") + b"\0"
            f.write(seq_bytes)

            # Record index entry
            length = len(seq_bytes)
            index_lines.append(f"{i}\t{offset}\t{length}\n")
            offset += length

    # Write index file
    with open(index_path, "w") as f:
        f.writelines(index_lines)

    # Write dbtype file
    with open(dbtype_path, "wb") as f:
        f.write(struct.pack("I", dbtype))


def run_foldseek_search(
    aa_fasta: Path,
    three_di_fasta: Path,
    output_path: Path,
    database_path: Path,
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
    tmp_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Foldseek search against a database using pre-computed 3Di.

    Args:
        aa_fasta: Path to amino acid FASTA file
        three_di_fasta: Path to 3Di FASTA file
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

    foldseek = ExternalTool("foldseek")

    # Create query database with pre-computed 3Di
    query_db = tmp_dir / "queryDB"
    create_query_db(aa_fasta, three_di_fasta, query_db, threads)

    # Run search
    result_db = tmp_dir / "resultDB"

    search_args = [
        "search",
        str(query_db),
        str(database_path),
        str(result_db),
        str(tmp_dir / "search_tmp"),
        "--threads", str(threads),
        "-e", str(evalue),
        "-s", str(sensitivity),
        "--max-seqs", str(max_seqs),
        "--exhaustive-search", "1",
    ]

    logger.info(f"Running Foldseek search against {database_path.name}")
    logger.debug(f"Query DB: {query_db}")

    try:
        result = foldseek.run(search_args, check=True)
        logger.debug(f"Foldseek search stdout: {result.stdout}")
    except Exception as e:
        logger.error(f"Foldseek search failed: {e}")
        raise

    # Convert results to tabular format
    convert_args = [
        "convertalis",
        str(query_db),
        str(database_path),
        str(result_db),
        str(output_path),
        "--threads", str(threads),
        "--format-output", FOLDSEEK_OUTPUT_FORMAT,
    ]

    try:
        result = foldseek.run(convert_args, check=True)
        logger.debug(f"Foldseek convertalis stdout: {result.stdout}")
    except Exception as e:
        logger.error(f"Foldseek convertalis failed: {e}")
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
    aa_fasta: Path,
    three_di_fasta: Path,
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
        aa_fasta: Path to amino acid FASTA file
        three_di_fasta: Path to 3Di FASTA file (from ProstT5)
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
                    aa_fasta=aa_fasta,
                    three_di_fasta=three_di_fasta,
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
                    aa_fasta=aa_fasta,
                    three_di_fasta=three_di_fasta,
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
