"""Run subcommand for vhold - full annotation pipeline."""

from pathlib import Path

from vhold.features.prostt5 import get_predictor
from vhold.features.confidence import apply_confidence_mask
from vhold.features.foldseek import search_databases, merge_results
from vhold.io import read_input
from vhold.io.fasta import write_fasta, write_3di_fasta
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
    triage: bool = False,
    triage_threshold: float = 0.90,
    classify: bool = True,
    classifier_confidence: float = 0.5,
    backend: str = "torch",
    use_lora: bool = True,
    input_format: str | None = None,
    genome_fasta: str | Path | None = None,
    export_formats: list[str] | None = None,
) -> None:
    """Run the full vhold annotation pipeline.

    This orchestrates:
    1. ProstT5 3Di prediction
    2. Foldseek database search
    3. Annotation transfer
    4. Output generation

    Args:
        input_file: Input file (FASTA, GenBank, or GFF3)
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
        "triage": triage,
        "triage_threshold": triage_threshold,
    }

    # ========================================
    # Step 1: Read input sequences
    # ========================================
    logger.info("Step 1: Reading input sequences...")
    sequences = read_input(
        input_path,
        input_format=input_format,
        fasta_path=genome_fasta,
    )
    logger.info(f"Read {len(sequences)} sequences")

    if not sequences:
        logger.error("No sequences found in input file")
        return

    # Get sequence dict and lengths
    seq_dict = {rec.id: rec.sequence for rec in sequences.values()}
    seq_lengths = {rec.id: rec.length for rec in sequences.values()}

    # Extract positional metadata (from GenBank/GFF input)
    protein_positions = {
        rec.id: {
            "contig": rec.contig, "start": rec.start,
            "end": rec.end, "strand": rec.strand,
        }
        for rec in sequences.values()
        if rec.contig is not None
    }
    if protein_positions:
        logger.info(f"Genomic positions: {len(protein_positions)} proteins")
    logger.info("")

    # ========================================
    # Step 1b: Embedding-based triage (optional)
    # ========================================
    triage_annotations = {}
    sequences_for_decoder = seq_dict
    shared_model = None
    shared_tokenizer = None

    if triage:
        from vhold.databases.embeddings import check_embedding_db, get_embedding_db_path
        from vhold.features.embeddings import (
            EmbeddingDatabase,
            build_embedding_consensus_results,
            triage_proteins,
        )

        if check_embedding_db(db_dir):
            logger.info("Step 1b: Embedding-based triage...")
            emb_db_path = get_embedding_db_path(db_dir)
            emb_db = EmbeddingDatabase(emb_db_path)
            emb_db.load()

            if backend == "onnx":
                from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor
                onnx_extractor = OnnxEmbeddingExtractor(
                    model_dir=Path(model_dir) if model_dir else None,
                )
                triage_result, extractor = triage_proteins(
                    sequences=seq_dict,
                    embedding_db=emb_db,
                    threshold=triage_threshold,
                    extractor=onnx_extractor,
                )
            else:
                triage_result, extractor = triage_proteins(
                    sequences=seq_dict,
                    embedding_db=emb_db,
                    device=device,
                    model_dir=Path(model_dir) if model_dir else None,
                    threshold=triage_threshold,
                    use_lora=use_lora,
                )

                # Save model/tokenizer for reuse by ProstT5Predictor
                shared_model = extractor.model
                shared_tokenizer = extractor.tokenizer

            logger.info(
                f"Triage: {triage_result.stats['matched']} matched, "
                f"{triage_result.stats['unmatched']} need full pipeline"
            )

            # Validate triage matches with sequence alignment (if Foldseek DBs installed)
            if triage_result.matched and db_dir:
                from vhold.features.alignment_validation import validate_triage_matches

                validated, rejected = validate_triage_matches(
                    matches=triage_result.matched,
                    query_sequences=seq_dict,
                    db_dir=db_dir,
                )
                if rejected:
                    logger.info(
                        f"Alignment validation: rejected {len(rejected)} "
                        f"low-identity matches"
                    )
                    triage_result.matched = validated
                    triage_result.unmatched.extend(rejected)

            # Build ConsensusResults for matched proteins
            if triage_result.matched:
                triage_annotations = build_embedding_consensus_results(
                    matches=triage_result.matched,
                    query_lengths=seq_lengths,
                    db_dir=db_dir,
                )

            # Only run decoder on unmatched proteins
            sequences_for_decoder = {
                sid: seq for sid, seq in seq_dict.items()
                if sid in set(triage_result.unmatched)
            }
            logger.info("")
        else:
            logger.warning(
                "Embedding database not installed. "
                "Run 'vhold install --embeddings' to enable triage. "
                "Continuing with full pipeline."
            )
            logger.info("")

    # ========================================
    # Step 2: Predict 3Di sequences
    # ========================================
    predictions_dir = output_path / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not sequences_for_decoder:
        logger.info("Step 2: Skipped (all proteins matched by embedding triage)")
        results = {}
    else:
        logger.info(
            f"Step 2: Predicting 3Di sequences for "
            f"{len(sequences_for_decoder)} proteins with ProstT5..."
        )

        predictor = get_predictor(
            device=device,
            backend=backend,
            model_dir=Path(model_dir) if model_dir else None,
            fast=fast,
        )

        # Reuse model from triage if available (PyTorch only)
        if shared_model is not None and backend == "torch":
            predictor.model = shared_model
            predictor.tokenizer = shared_tokenizer
            predictor._loaded = True
            logger.info("Reusing model from embedding triage")

        results = predictor.predict_batch(
            sequences_for_decoder,
            batch_size=batch_size,
            show_progress=True,
        )

    # Save AA sequences for Foldseek (only those that went through decoder)
    aa_fasta = predictions_dir / "aa_sequences.fasta"
    write_fasta(sequences_for_decoder, aa_fasta)

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

    if results:
        mean_conf = sum(r.mean_confidence for r in results.values()) / len(results)
        logger.info(f"Mean prediction confidence: {mean_conf:.4f}")
    logger.info("")

    # ========================================
    # Step 3: Search against databases
    # ========================================
    annotations = {}

    if sequences_for_decoder:
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
        # Use only the lengths of proteins that went through the decoder
        decoder_lengths = {
            sid: length for sid, length in seq_lengths.items()
            if sid in sequences_for_decoder
        }
        annotations = transfer_annotations_consensus(
            hits=hits,
            query_lengths=decoder_lengths,
            db_dir=db_dir,
        )

        # Count Foldseek results
        annotated_foldseek = sum(1 for a in annotations.values() if a.is_annotated)
        with_consensus = sum(1 for a in annotations.values() if a.has_consensus)
        logger.info(f"Annotated (Foldseek): {annotated_foldseek}/{len(annotations)} proteins")
        logger.info(f"Multi-database agreement: {with_consensus}/{annotated_foldseek} proteins")
        logger.info("")
    else:
        logger.info("Step 3-4: Skipped (all proteins matched by embedding triage)")
        logger.info("")

    # ========================================
    # Step 4a: Merge embedding triage results
    # ========================================
    if triage_annotations:
        for query_id, consensus in triage_annotations.items():
            annotations[query_id] = consensus
        logger.info(f"Merged {len(triage_annotations)} embedding-matched annotations")
        logger.info("")

    # Ensure all query proteins have an entry (even if unannotated)
    from vhold.results.consensus import ConsensusResult
    for query_id, length in seq_lengths.items():
        if query_id not in annotations:
            annotations[query_id] = ConsensusResult(
                query_id=query_id, query_length=length
            )

    # Propagate genomic positions to annotations
    if protein_positions:
        for qid, pos in protein_positions.items():
            if qid in annotations:
                annotations[qid].contig = pos["contig"]
                annotations[qid].start = pos["start"]
                annotations[qid].end = pos["end"]
                annotations[qid].strand = pos["strand"]

    # Count total results
    annotated = sum(1 for a in annotations.values() if a.is_annotated)
    logger.info(f"Total annotated: {annotated}/{len(annotations)} proteins")
    logger.info("")

    # ========================================
    # Step 4a.5: MLP classification (when model installed)
    # ========================================
    if classify:
        from vhold.databases.classifier import check_classifier_model

        if check_classifier_model(Path(model_dir) if model_dir else None):
            # Find unknown proteins that could benefit from MLP classification
            unknown_ids = [
                qid for qid, result in annotations.items()
                if result.is_annotated
                and result.classification_source in ("keywords", "embedding")
                and result.functional_category == "unknown"
            ]

            if unknown_ids:
                logger.info(
                    f"Step 4a.5: MLP classification of {len(unknown_ids)} unknown proteins..."
                )

                from vhold.features.classifier import load_classifier, classify_with_model

                classifier = load_classifier(
                    model_dir=Path(model_dir) if model_dir else None
                )

                if classifier is not None:
                    # Extract embeddings for unknown proteins
                    unknown_seqs = {
                        sid: seq_dict[sid] for sid in unknown_ids if sid in seq_dict
                    }

                    if backend == "onnx":
                        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor
                        extractor = OnnxEmbeddingExtractor(
                            model_dir=Path(model_dir) if model_dir else None,
                        )
                    else:
                        from vhold.features.embeddings import EmbeddingExtractor
                        extractor = EmbeddingExtractor(
                            device=device,
                            model_dir=Path(model_dir) if model_dir else None,
                            model=shared_model,
                            tokenizer=shared_tokenizer,
                            encoder_only=shared_model is None,
                            use_lora=use_lora,
                        )
                    ids, embeddings = extractor.extract_batch(
                        unknown_seqs, show_progress=False,
                    )

                    predictions = classify_with_model(
                        embeddings, ids, classifier,
                        confidence_threshold=classifier_confidence,
                    )

                    reclassified = 0
                    for pid, (category, confidence) in predictions.items():
                        if pid in annotations:
                            annotations[pid].functional_category = category
                            annotations[pid].classification_source = "mlp_classifier"
                            reclassified += 1
                            logger.info(
                                f"  {pid}: unknown -> {category} "
                                f"(confidence: {confidence:.3f})"
                            )

                    logger.info(
                        f"MLP classifier: {reclassified}/{len(unknown_ids)} "
                        f"proteins reclassified"
                    )
                logger.info("")
        else:
            logger.info(
                "MLP classifier not installed, skipping Step 4a.5. "
                "Run 'vhold install --classifier' to enable."
            )

    # ========================================
    # Step 4b: LLM reclassification (optional)
    # ========================================
    if llm_classify:
        logger.info("Step 4b: LLM reclassification of unknown proteins...")
        from vhold.results.llm_classify import llm_reclassify
        annotations = llm_reclassify(annotations, model=llm_model)
        logger.info("")

    # ========================================
    # Step 4c: Neighborhood voting (auto-activates with GenBank/GFF input)
    # ========================================
    if protein_positions:
        unknown_ids = [
            qid for qid, r in annotations.items()
            if r.functional_category == "unknown" and qid in protein_positions
        ]
        if unknown_ids:
            logger.info(
                f"Step 4c: Neighborhood voting for {len(unknown_ids)} "
                f"unknown proteins..."
            )
            from vhold.results.neighborhood import vote_neighborhood
            votes = vote_neighborhood(
                unknown_ids, annotations, protein_positions,
            )
            for pid, (category, conf) in votes.items():
                annotations[pid].functional_category = category
                annotations[pid].classification_source = "neighborhood_vote"
                logger.info(
                    f"  {pid}: unknown -> {category} "
                    f"(vote confidence: {conf:.3f})"
                )
            logger.info(
                f"Neighborhood voting: {len(votes)}/{len(unknown_ids)} "
                f"proteins reclassified"
            )
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
    with_consensus = sum(1 for a in annotations.values() if a.has_consensus)
    consensus_rate = with_consensus / annotated * 100 if annotated else 0
    embedding_matched = sum(
        1 for a in annotations.values() if a.novelty == "embedding_match"
    )
    logger.info("Summary:")
    logger.info(f"  Total proteins: {len(annotations)}")
    logger.info(f"  Annotated: {annotated} ({annotation_rate:.1f}%)")
    if embedding_matched:
        logger.info(f"  Embedding triage: {embedding_matched}")
        logger.info(f"  Structural search: {annotated - embedding_matched}")
    logger.info(f"  Multi-DB consensus: {with_consensus} ({consensus_rate:.1f}% of annotated)")
    logger.info(f"  Unannotated: {len(annotations) - annotated}")

    # ========================================
    # Step 6: Export to metagenomic formats (optional)
    # ========================================
    if export_formats:
        logger.info("")
        logger.info(f"Step 6: Exporting to metagenomic formats: {export_formats}")
        from vhold.results.export import export_all_formats

        export_dir = output_path / "exports"
        export_files = export_all_formats(
            results=annotations,
            output_dir=export_dir,
            formats=export_formats,
            prefix=prefix,
        )

        for fmt, path in export_files.items():
            if isinstance(path, dict):
                for sub_fmt, sub_path in path.items():
                    output_files[f"export_{fmt}_{sub_fmt}"] = sub_path
            else:
                output_files[f"export_{fmt}"] = path
