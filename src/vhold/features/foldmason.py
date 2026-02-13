"""FoldMason wrapper for multiple structural alignment."""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from vhold.utils.external import ExternalTool, check_foldmason
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldMasonResult:
    """Result from a FoldMason multiple structural alignment."""

    aa_msa: Path
    three_di_msa: Path
    guide_tree: Path
    num_sequences: int
    alignment_length: int


def run_foldmason_msa(
    query_db: Path,
    output_prefix: Path,
    threads: int = 4,
    tmp_dir: Path | None = None,
) -> FoldMasonResult:
    """Run FoldMason structuremsa on a Foldseek-format database.

    When the database lacks C-alpha coordinates (_ca.dbtype),
    FoldMason automatically uses fastMode (pure 3Di+AA alignment).

    Args:
        query_db: Path to Foldseek-format database (from create_query_db).
        output_prefix: Output prefix for result files.
        threads: Number of CPU threads.
        tmp_dir: Temporary directory for FoldMason intermediates.

    Returns:
        FoldMasonResult with paths to output files.
    """
    available, version = check_foldmason()
    if not available:
        raise FileNotFoundError(
            "foldmason not found in PATH. Install with: conda install -c bioconda foldmason"
        )
    logger.info(f"Using FoldMason {version}")

    foldmason = ExternalTool("foldmason")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cleanup_tmp = False
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="foldmason_"))
        cleanup_tmp = True

    try:
        args = [
            "structuremsa",
            str(query_db),
            str(output_prefix),
            str(tmp_dir),
            "--threads", str(threads),
        ]

        logger.info(f"Running FoldMason structuremsa with {threads} threads")
        result = foldmason.run(args)

        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                if line.strip():
                    logger.debug(f"foldmason: {line.strip()}")

        # Parse output files
        aa_msa_path = Path(str(output_prefix) + "_aa.fa")
        three_di_msa_path = Path(str(output_prefix) + "_3di.fa")
        tree_path = Path(str(output_prefix) + ".nw")

        if not aa_msa_path.exists():
            raise RuntimeError(
                f"FoldMason did not produce expected output: {aa_msa_path}"
            )

        # Count sequences and alignment length from AA MSA
        num_sequences = 0
        alignment_length = 0
        with open(aa_msa_path) as f:
            for line in f:
                if line.startswith(">"):
                    num_sequences += 1
                elif num_sequences == 1:
                    alignment_length += len(line.strip())

        logger.info(
            f"Alignment complete: {num_sequences} sequences, "
            f"{alignment_length} columns"
        )

        return FoldMasonResult(
            aa_msa=aa_msa_path,
            three_di_msa=three_di_msa_path,
            guide_tree=tree_path,
            num_sequences=num_sequences,
            alignment_length=alignment_length,
        )
    finally:
        if cleanup_tmp and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_foldmason_refine(
    query_db: Path,
    msa_input: Path,
    msa_output: Path,
    refine_iters: int = 100,
) -> Path:
    """Refine a FoldMason MSA iteratively.

    Randomly splits the MSA, removes gap-only columns, realigns,
    and keeps the result only if LDDT improves. Repeats for
    the specified number of iterations.

    Args:
        query_db: Path to Foldseek-format database.
        msa_input: Path to input MSA (AA FASTA).
        msa_output: Path for refined MSA output.
        refine_iters: Number of refinement iterations.

    Returns:
        Path to the refined MSA.
    """
    foldmason = ExternalTool("foldmason")
    foldmason.check_available()

    args = [
        "refinemsa",
        str(query_db),
        str(msa_input),
        str(msa_output),
        "--refine-iters", str(refine_iters),
    ]

    logger.info(f"Refining MSA with {refine_iters} iterations")
    foldmason.run(args)

    if not msa_output.exists():
        raise RuntimeError(f"FoldMason refinement did not produce output: {msa_output}")

    logger.info(f"Refined MSA written to {msa_output}")
    return msa_output


def write_alignment_summary(
    result: FoldMasonResult,
    output_path: Path,
    refined: bool = False,
) -> Path:
    """Write alignment summary as JSON.

    Args:
        result: FoldMasonResult from alignment.
        output_path: Output directory.
        refined: Whether the alignment was refined.

    Returns:
        Path to summary JSON file.
    """
    summary = {
        "num_sequences": result.num_sequences,
        "alignment_length": result.alignment_length,
        "aa_msa": str(result.aa_msa),
        "three_di_msa": str(result.three_di_msa),
        "guide_tree": str(result.guide_tree),
        "refined": refined,
        "mode": "fastMode (no coordinates, 3Di+AA only)",
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path
