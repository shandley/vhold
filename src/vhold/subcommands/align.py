"""Align subcommand for vhold - Multiple structural alignment via FoldMason."""

import shutil
import tempfile
from pathlib import Path

from vhold.features.confidence import apply_confidence_mask
from vhold.features.foldseek import create_query_db
from vhold.features.foldmason import (
    run_foldmason_msa,
    write_alignment_summary,
)
from vhold.features.prostt5 import ProstT5Predictor
from vhold.io.fasta import read_fasta, write_fasta, write_3di_fasta
from vhold.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_align(
    input_file: str | Path,
    output_dir: str | Path,
    threads: int = 4,
    device: str = "auto",
    fast: bool = False,
    confidence_threshold: float = 0.7,
    model_dir: str | Path | None = None,
    refine_iters: int = 0,
    predictions_dir: str | Path | None = None,
) -> None:
    """Run multiple structural alignment on input protein sequences.

    Pipeline:
    1. Read protein sequences
    2. Predict 3Di with ProstT5 (or load from predictions_dir)
    3. Create Foldseek-format database (AA + 3Di, no coordinates)
    4. Run FoldMason structuremsa (fastMode — 3Di+AA alignment)
    5. Output AA MSA, 3Di MSA, and guide tree

    Args:
        input_file: Input FASTA file with protein sequences
        output_dir: Output directory for alignment results
        threads: Number of threads for FoldMason
        device: Device for ProstT5 inference
        fast: Use greedy decoding for ProstT5
        confidence_threshold: Threshold for 3Di confidence masking
        model_dir: ProstT5 model cache directory
        refine_iters: Ignored. FoldMason refinement requires C-alpha
            coordinates not available in ProstT5 coordinate-free mode.
        predictions_dir: Pre-computed 3Di predictions directory (skip ProstT5)
    """
    setup_logging()

    if refine_iters > 0:
        logger.warning(
            f"--refine-iters {refine_iters} ignored. FoldMason refinement requires "
            "C-alpha coordinates, which are not available in ProstT5 coordinate-free mode."
        )

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("vHold align — Multiple Structural Alignment")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # ========================================
    # Step 1: Read input sequences
    # ========================================
    logger.info("Step 1: Reading input sequences...")
    sequences = read_fasta(input_path)
    logger.info(f"Read {len(sequences)} sequences")

    if len(sequences) < 2:
        logger.error("Multiple structural alignment requires at least 2 sequences")
        raise ValueError("At least 2 sequences are required for alignment")

    seq_dict = {rec.id: rec.sequence for rec in sequences.values()}
    logger.info("")

    # ========================================
    # Step 2: Get 3Di predictions
    # ========================================
    if predictions_dir:
        # Load pre-computed predictions
        pred_path = Path(predictions_dir)
        logger.info(f"Step 2: Loading pre-computed 3Di from {pred_path}")

        aa_fasta = pred_path / "aa_sequences.fasta"
        three_di_fasta = _find_3di_fasta(pred_path)

        if not aa_fasta.exists():
            # Write AA sequences from input (predictions-dir may only have 3Di)
            aa_fasta = output_path / "predictions" / "aa_sequences.fasta"
            aa_fasta.parent.mkdir(parents=True, exist_ok=True)
            write_fasta(seq_dict, aa_fasta)

        if three_di_fasta is None:
            raise FileNotFoundError(
                f"No 3Di FASTA found in {pred_path}. "
                "Expected 3di_sequences_masked.fasta or 3di_sequences.fasta"
            )

        logger.info(f"AA sequences: {aa_fasta}")
        logger.info(f"3Di sequences: {three_di_fasta}")
    else:
        # Run ProstT5 prediction
        logger.info(f"Step 2: Predicting 3Di sequences with ProstT5...")
        logger.info(f"Device: {device}, Fast mode: {fast}")

        pred_output = output_path / "predictions"
        pred_output.mkdir(parents=True, exist_ok=True)

        predictor = ProstT5Predictor(
            device=device,
            model_dir=Path(model_dir) if model_dir else None,
            fast=fast,
        )

        results = predictor.predict_batch(
            seq_dict,
            batch_size=1,
            show_progress=True,
        )

        # Write AA sequences
        aa_fasta = pred_output / "aa_sequences.fasta"
        write_fasta(seq_dict, aa_fasta)

        # Write masked 3Di sequences
        masked_3di = {}
        for seq_id, result in results.items():
            masked = apply_confidence_mask(result, threshold=confidence_threshold)
            masked_3di[seq_id] = masked

        three_di_fasta = pred_output / "3di_sequences_masked.fasta"
        write_3di_fasta(masked_3di, three_di_fasta)

        # Write confidence scores
        confidence_output = pred_output / "confidence_scores.tsv"
        with open(confidence_output, "w") as f:
            f.write("sequence_id\tmean_confidence\tlength\n")
            for seq_id, result in results.items():
                f.write(f"{seq_id}\t{result.mean_confidence:.4f}\t{result.length}\n")

        mean_conf = sum(r.mean_confidence for r in results.values()) / len(results)
        logger.info(f"Mean prediction confidence: {mean_conf:.4f}")

    logger.info("")

    # ========================================
    # Step 3: Create Foldseek database and run FoldMason
    # ========================================
    logger.info("Step 3: Running FoldMason multiple structural alignment...")

    with tempfile.TemporaryDirectory(prefix="vhold_align_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create Foldseek-format database (no coordinates → fastMode)
        query_db = tmp_path / "queryDB"
        create_query_db(aa_fasta, three_di_fasta, query_db, threads)
        logger.info("Created Foldseek database (3Di+AA, no coordinates → fastMode)")

        # Run FoldMason structuremsa
        msa_prefix = tmp_path / "alignment"
        msa_result = run_foldmason_msa(
            query_db=query_db,
            output_prefix=msa_prefix,
            threads=threads,
        )

        # Copy results to output directory
        final_aa = output_path / "alignment_aa.fa"
        final_3di = output_path / "alignment_3di.fa"
        final_tree = output_path / "alignment.nw"

        shutil.copy2(msa_result.aa_msa, final_aa)
        if msa_result.three_di_msa.exists():
            shutil.copy2(msa_result.three_di_msa, final_3di)
        if msa_result.guide_tree.exists():
            shutil.copy2(msa_result.guide_tree, final_tree)

        # Update result paths to final locations
        msa_result.aa_msa = final_aa
        msa_result.three_di_msa = final_3di
        msa_result.guide_tree = final_tree

    # Write summary
    summary_path = write_alignment_summary(msa_result, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Alignment complete!")
    logger.info("=" * 60)
    logger.info(f"Sequences aligned: {msa_result.num_sequences}")
    logger.info(f"Alignment length: {msa_result.alignment_length} columns")
    logger.info(f"AA MSA: {final_aa}")
    logger.info(f"3Di MSA: {final_3di}")
    logger.info(f"Guide tree: {final_tree}")
    logger.info(f"Summary: {summary_path}")


def _find_3di_fasta(predictions_dir: Path) -> Path | None:
    """Find the 3Di FASTA file in a predictions directory.

    Prefers masked version over raw.
    """
    masked = predictions_dir / "3di_sequences_masked.fasta"
    if masked.exists():
        return masked
    raw = predictions_dir / "3di_sequences.fasta"
    if raw.exists():
        return raw
    return None
