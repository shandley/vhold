"""Run subcommand for vhold - full annotation pipeline."""

from pathlib import Path

from vhold.features.prostt5 import ProstT5Predictor
from vhold.features.confidence import apply_confidence_mask
from vhold.features.foldseek import search_databases, merge_results
from vhold.io.fasta import read_fasta, write_fasta, write_3di_fasta
from vhold.results.parser import parse_dataframe_results
from vhold.results.annotations import transfer_annotations_consensus
from vhold.results.output import generate_report_consensus
from vhold.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def run_pipeline(
    input_file: str | Path,
    output_dir: str | Path,
    database_dir: str | Path | None = None,
    databases: str = "all",
    threads: int = 4,
    batch_size: int = 1,
    device: str = "auto",
    evalue: float = 1e-3,
    sensitivity: float = 9.5,
    confidence_threshold: float = 0.7,
    model_dir: str | Path | None = None,
    prefix: str = "vhold",
    fast: bool = False,
    llm_classify: bool = False,
    llm_model: str = "claude-haiku-4-5-20251001",
) -> None:
    """Run the full vhold annotation pipeline.

    This orchestrates:
    1. ProstT5 3Di prediction
    2. Foldseek database search
    3. Annotation transfer
    4. Output generation

    Args:
        input_file: Input FASTA file
        output_dir: Output directory
        database_dir: Database directory
        databases: Which databases to search ('all', 'bfvd', 'viro3d')
        threads: Number of threads
        batch_size: Batch size for ProstT5
        device: Device for ProstT5
        evalue: E-value threshold for Foldseek
        sensitivity: Foldseek sensitivity
        confidence_threshold: Confidence threshold for masking
        model_dir: Model cache directory
        prefix: Output file prefix
        fast: Use greedy decoding for ProstT5 (~3x faster)
    """
    setup_logging()

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_dir = Path(database_dir) if database_dir else None

    logger.info("=" * 60)
    logger.info("vhold - Viral protein annotation using structural homology")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Databases: {databases}")
    logger.info("")

    # Parameters dict for output
    parameters = {
        "threads": threads,
        "batch_size": batch_size,
        "device": device,
        "evalue": evalue,
        "sensitivity": sensitivity,
        "confidence_threshold": confidence_threshold,
    }

    # ========================================
    # Step 1: Read input sequences
    # ========================================
    logger.info("Step 1: Reading input sequences...")
    sequences = read_fasta(input_path)
    logger.info(f"Read {len(sequences)} sequences")

    if not sequences:
        logger.error("No sequences found in input file")
        return

    # Get sequence dict and lengths
    seq_dict = {rec.id: rec.sequence for rec in sequences.values()}
    seq_lengths = {rec.id: rec.length for rec in sequences.values()}
    logger.info("")

    # ========================================
    # Step 2: Predict 3Di sequences
    # ========================================
    logger.info("Step 2: Predicting 3Di sequences with ProstT5...")
    predictions_dir = output_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictor = ProstT5Predictor(
        device=device,
        model_dir=Path(model_dir) if model_dir else None,
        fast=fast,
    )

    results = predictor.predict_batch(
        seq_dict,
        batch_size=batch_size,
        show_progress=True,
    )

    # Save AA sequences for Foldseek
    aa_fasta = predictions_dir / "aa_sequences.fasta"
    write_fasta(seq_dict, aa_fasta)

    # Save 3Di sequences
    raw_3di = {seq_id: result.three_di_sequence for seq_id, result in results.items()}
    write_3di_fasta(raw_3di, predictions_dir / "3di_sequences.fasta")

    # Save masked 3Di sequences
    masked_3di = {}
    for seq_id, result in results.items():
        masked = apply_confidence_mask(result, threshold=confidence_threshold)
        masked_3di[seq_id] = masked

    three_di_fasta = predictions_dir / "3di_sequences_masked.fasta"
    write_3di_fasta(masked_3di, three_di_fasta)

    mean_conf = sum(r.mean_confidence for r in results.values()) / len(results)
    logger.info(f"Mean prediction confidence: {mean_conf:.4f}")
    logger.info("")

    # ========================================
    # Step 3: Search against databases
    # ========================================
    logger.info("Step 3: Searching against reference databases...")
    search_dir = output_path / "foldseek"
    search_dir.mkdir(parents=True, exist_ok=True)

    search_results = search_databases(
        aa_fasta=aa_fasta,
        three_di_fasta=three_di_fasta,
        output_dir=search_dir,
        databases=databases,
        db_dir=db_dir,
        threads=threads,
        evalue=evalue,
        sensitivity=sensitivity,
    )

    # Merge results
    merged_results = merge_results(search_results, keep_best=False)
    logger.info(f"Total hits: {len(merged_results)}")

    # Get best hits
    best_results = merge_results(search_results, keep_best=True)
    logger.info(f"Proteins with hits: {len(best_results)}")
    logger.info("")

    # ========================================
    # Step 4: Transfer annotations (with consensus)
    # ========================================
    logger.info("Step 4: Transferring annotations with multi-database consensus...")

    # Parse hits
    hits = parse_dataframe_results(merged_results)

    # Transfer annotations using consensus scoring
    annotations = transfer_annotations_consensus(
        hits=hits,
        query_lengths=seq_lengths,
        db_dir=db_dir,
    )

    # Count results
    annotated = sum(1 for a in annotations.values() if a.is_annotated)
    with_consensus = sum(1 for a in annotations.values() if a.has_consensus)
    logger.info(f"Annotated: {annotated}/{len(annotations)} proteins")
    logger.info(f"Multi-database agreement: {with_consensus}/{annotated} proteins")
    logger.info("")

    # ========================================
    # Step 4b: LLM reclassification (optional)
    # ========================================
    if llm_classify:
        logger.info("Step 4b: LLM reclassification of unknown proteins...")
        from vhold.results.llm_classify import llm_reclassify
        annotations = llm_reclassify(annotations, model=llm_model)
        logger.info("")

    # ========================================
    # Step 5: Generate output files
    # ========================================
    logger.info("Step 5: Generating output files...")

    databases_searched = []
    if databases in ("all", "bfvd"):
        databases_searched.append("bfvd")
    if databases in ("all", "viro3d"):
        databases_searched.append("viro3d")

    output_files = generate_report_consensus(
        annotations=annotations,
        output_dir=output_path,
        prefix=prefix,
        input_file=str(input_path),
        databases_searched=databases_searched,
        parameters=parameters,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Output files:")
    for file_type, file_path in output_files.items():
        logger.info(f"  {file_type}: {file_path}")
    logger.info("")

    # Print summary statistics
    annotation_rate = annotated / len(annotations) * 100 if annotations else 0
    consensus_rate = with_consensus / annotated * 100 if annotated else 0
    logger.info("Summary:")
    logger.info(f"  Total proteins: {len(annotations)}")
    logger.info(f"  Annotated: {annotated} ({annotation_rate:.1f}%)")
    logger.info(f"  Multi-DB consensus: {with_consensus} ({consensus_rate:.1f}% of annotated)")
    logger.info(f"  Unannotated: {len(annotations) - annotated}")
