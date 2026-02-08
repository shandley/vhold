"""Predict subcommand for vhold - ProstT5 3Di prediction."""

from pathlib import Path

from vhold.features.prostt5 import ProstT5Predictor
from vhold.features.confidence import apply_confidence_mask
from vhold.io.fasta import read_fasta, write_3di_fasta
from vhold.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_predict(
    input_file: str | Path,
    output_dir: str | Path,
    threads: int = 1,
    batch_size: int = 1,
    device: str = "auto",
    confidence_threshold: float = 0.7,
    model_dir: str | Path | None = None,
    fast: bool = False,
) -> None:
    """Run ProstT5 3Di prediction on input sequences.

    Args:
        input_file: Input FASTA file
        output_dir: Output directory for predictions
        threads: Number of threads (currently unused, for future parallelism)
        batch_size: Batch size for inference
        device: Device for inference ('auto', 'cuda', 'mps', 'cpu')
        confidence_threshold: Threshold for confidence masking
        model_dir: Model cache directory
        fast: Use greedy decoding (~3x faster, may reduce sensitivity)
    """
    setup_logging()

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Read input sequences
    logger.info("Reading input sequences...")
    sequences = read_fasta(input_path)
    logger.info(f"Read {len(sequences)} sequences")

    if not sequences:
        logger.warning("No sequences found in input file")
        return

    # Convert to dict of {id: sequence}
    seq_dict = {rec.id: rec.sequence for rec in sequences.values()}

    # Initialize predictor
    predictor = ProstT5Predictor(
        device=device,
        model_dir=Path(model_dir) if model_dir else None,
        fast=fast,
    )

    # Run predictions
    logger.info("Predicting 3Di sequences...")
    results = predictor.predict_batch(
        seq_dict,
        batch_size=batch_size,
        show_progress=True,
    )

    # Write outputs
    # 1. Raw 3Di sequences
    raw_3di = {seq_id: result.three_di_sequence for seq_id, result in results.items()}
    raw_output = output_path / "3di_sequences.fasta"
    write_3di_fasta(raw_3di, raw_output)
    logger.info(f"Wrote raw 3Di sequences to {raw_output}")

    # 2. Masked 3Di sequences (for Foldseek search)
    masked_3di = {}
    for seq_id, result in results.items():
        masked = apply_confidence_mask(result, threshold=confidence_threshold)
        masked_3di[seq_id] = masked

    masked_output = output_path / "3di_sequences_masked.fasta"
    write_3di_fasta(masked_3di, masked_output)
    logger.info(f"Wrote masked 3Di sequences to {masked_output}")

    # 3. Confidence scores (as simple TSV)
    confidence_output = output_path / "confidence_scores.tsv"
    with open(confidence_output, "w") as f:
        f.write("sequence_id\tmean_confidence\tlength\n")
        for seq_id, result in results.items():
            f.write(f"{seq_id}\t{result.mean_confidence:.4f}\t{result.length}\n")
    logger.info(f"Wrote confidence scores to {confidence_output}")

    # Summary
    mean_conf = sum(r.mean_confidence for r in results.values()) / len(results)
    logger.info(f"Prediction complete: {len(results)} sequences")
    logger.info(f"Mean confidence: {mean_conf:.4f}")
