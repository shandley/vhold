#!/usr/bin/env python3
"""Build embedding database from BFVD + Viro3D reference proteins.

This is a one-time operation that pre-computes ProstT5 encoder embeddings
for all reference proteins. The resulting .npz file is used by vhold's
embedding triage mode (--triage) to quickly identify known proteins
without running the expensive decoder.

Uses T5EncoderModel (encoder-only, ~4.8 GB) instead of the full model
(~11.3 GB) to minimize memory usage. Supports checkpoint/resume for
long-running jobs.

Estimated times on Apple M4 (24GB):
  MPS: ~1.8 proteins/sec -> ~68 hours for 436K proteins
  CPU: ~1.3 proteins/sec -> ~94 hours for 436K proteins

Usage:
    # Basic (uses installed databases, MPS on Apple Silicon):
    caffeinate -i python scripts/build_embedding_db.py \
        --output ~/.vhold/databases/embeddings/vhold_embeddings.npz \
        --device mps

    # With explicit FASTA files:
    python scripts/build_embedding_db.py \
        --bfvd-fasta /path/to/bfvd_sequences.fasta \
        --viro3d-fasta /path/to/viro3d_sequences.fasta \
        --output vhold_embeddings.npz \
        --device cuda --batch-size 64

    # Resume after interruption:
    python scripts/build_embedding_db.py \
        --output ~/.vhold/databases/embeddings/vhold_embeddings.npz \
        --device mps --resume
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def extract_sequences_from_foldseek_db(db_path: Path) -> dict[str, str]:
    """Extract protein sequences from a Foldseek database.

    Uses foldseek convert2fasta to dump sequences.

    Args:
        db_path: Path to the Foldseek database prefix

    Returns:
        Dict mapping protein IDs to AA sequences
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = ["foldseek", "convert2fasta", str(db_path), tmp_path]
    subprocess.run(cmd, check=True, capture_output=True)

    from vhold.io.fasta import read_fasta

    sequences_raw = read_fasta(Path(tmp_path))
    sequences = {rec.id: rec.sequence for rec in sequences_raw.values()}

    Path(tmp_path).unlink()
    return sequences


def get_checkpoint_dir(output_path: Path) -> Path:
    """Get checkpoint directory for a given output path."""
    return output_path.parent / f".{output_path.stem}_checkpoints"


def save_checkpoint(
    checkpoint_dir: Path,
    ids: list[str],
    embeddings: np.ndarray,
    source_dbs: list[str],
    processed_count: int,
) -> None:
    """Save a checkpoint of processed embeddings.

    Args:
        checkpoint_dir: Directory for checkpoint files
        ids: Protein IDs processed so far
        embeddings: Embedding matrix (N, 1024) float32
        source_dbs: Source database for each protein
        processed_count: Total number processed (for naming)
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "latest.npz"
    # np.savez auto-appends .npz, so use stem without extension
    tmp_stem = checkpoint_dir / "latest_tmp"

    np.savez(
        tmp_stem,
        ids=np.array(ids),
        embeddings=embeddings.astype(np.float32),
        source_dbs=np.array(source_dbs),
        processed_count=np.array(processed_count),
    )

    # np.savez wrote to latest_tmp.npz — rename atomically
    tmp_npz = checkpoint_dir / "latest_tmp.npz"
    tmp_npz.rename(checkpoint_path)


def load_checkpoint(checkpoint_dir: Path) -> dict | None:
    """Load the latest checkpoint if it exists.

    Returns:
        Dict with 'ids', 'embeddings', 'source_dbs', 'processed_count'
        or None if no checkpoint exists.
    """
    checkpoint_path = checkpoint_dir / "latest.npz"
    if not checkpoint_path.exists():
        return None

    data = np.load(checkpoint_path, allow_pickle=True)
    return {
        "ids": list(data["ids"]),
        "embeddings": data["embeddings"].astype(np.float32),
        "source_dbs": list(data["source_dbs"]),
        "processed_count": int(data["processed_count"]),
    }


def format_eta(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def main():
    parser = argparse.ArgumentParser(
        description="Build vhold embedding database from reference proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Generate on Apple Silicon (run overnight):
  caffeinate -i python scripts/build_embedding_db.py \\
      --output ~/.vhold/databases/embeddings/vhold_embeddings.npz \\
      --device mps

  # Resume after interruption:
  python scripts/build_embedding_db.py \\
      --output ~/.vhold/databases/embeddings/vhold_embeddings.npz \\
      --device mps --resume

  # CUDA GPU (faster):
  python scripts/build_embedding_db.py \\
      --output vhold_embeddings.npz --device cuda --batch-size 64
""",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="vhold database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument(
        "--bfvd-fasta",
        type=str,
        default=None,
        help="BFVD protein sequences FASTA (alternative to --db-dir)",
    )
    parser.add_argument(
        "--viro3d-fasta",
        type=str,
        default=None,
        help="Viro3D protein sequences FASTA (alternative to --db-dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vhold_embeddings.npz",
        help="Output .npz file path (default: vhold_embeddings.npz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for embedding extraction (default: 1, optimal for MPS)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length to process (default: 2048)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N proteins (default: 10000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    # Collect sequences
    all_sequences: dict[str, str] = {}
    source_map: dict[str, str] = {}  # protein_id -> source_db

    if args.bfvd_fasta or args.viro3d_fasta:
        # Load from explicit FASTA files
        from vhold.io.fasta import read_fasta

        if args.bfvd_fasta:
            print(f"Loading BFVD sequences from {args.bfvd_fasta}...")
            records = read_fasta(Path(args.bfvd_fasta))
            for rec in records.values():
                all_sequences[rec.id] = rec.sequence
                source_map[rec.id] = "bfvd"
            print(f"  Loaded {len(records)} BFVD sequences")

        if args.viro3d_fasta:
            print(f"Loading Viro3D sequences from {args.viro3d_fasta}...")
            records = read_fasta(Path(args.viro3d_fasta))
            for rec in records.values():
                all_sequences[rec.id] = rec.sequence
                source_map[rec.id] = "viro3d"
            print(f"  Loaded {len(records)} Viro3D sequences")

    elif args.db_dir:
        # Extract from Foldseek databases
        from vhold.databases.install import get_bfvd_db_path, get_viro3d_db_path

        db_dir = Path(args.db_dir)

        bfvd_path = get_bfvd_db_path(db_dir)
        if bfvd_path.parent.exists():
            print(f"Extracting BFVD sequences from {bfvd_path}...")
            bfvd_seqs = extract_sequences_from_foldseek_db(bfvd_path)
            for sid, seq in bfvd_seqs.items():
                all_sequences[sid] = seq
                source_map[sid] = "bfvd"
            print(f"  Extracted {len(bfvd_seqs)} BFVD sequences")

        viro3d_path = get_viro3d_db_path(db_dir)
        if viro3d_path.parent.exists():
            print(f"Extracting Viro3D sequences from {viro3d_path}...")
            viro3d_seqs = extract_sequences_from_foldseek_db(viro3d_path)
            for sid, seq in viro3d_seqs.items():
                all_sequences[sid] = seq
                source_map[sid] = "viro3d"
            print(f"  Extracted {len(viro3d_seqs)} Viro3D sequences")
    else:
        # Use default database directory
        from vhold.utils.constants import get_db_dir
        from vhold.databases.install import get_bfvd_db_path, get_viro3d_db_path

        db_dir = get_db_dir()
        print(f"Using default database directory: {db_dir}")

        bfvd_path = get_bfvd_db_path(db_dir)
        if bfvd_path.parent.exists():
            print(f"Extracting BFVD sequences from {bfvd_path}...")
            bfvd_seqs = extract_sequences_from_foldseek_db(bfvd_path)
            for sid, seq in bfvd_seqs.items():
                all_sequences[sid] = seq
                source_map[sid] = "bfvd"
            print(f"  Extracted {len(bfvd_seqs)} BFVD sequences")

        viro3d_path = get_viro3d_db_path(db_dir)
        if viro3d_path.parent.exists():
            print(f"Extracting Viro3D sequences from {viro3d_path}...")
            viro3d_seqs = extract_sequences_from_foldseek_db(viro3d_path)
            for sid, seq in viro3d_seqs.items():
                all_sequences[sid] = seq
                source_map[sid] = "viro3d"
            print(f"  Extracted {len(viro3d_seqs)} Viro3D sequences")

    if not all_sequences:
        print("ERROR: No sequences found. Provide --db-dir or FASTA files.")
        sys.exit(1)

    # Filter by max length
    original_count = len(all_sequences)
    all_sequences = {
        sid: seq for sid, seq in all_sequences.items()
        if len(seq) <= args.max_length
    }
    if len(all_sequences) < original_count:
        print(
            f"Filtered {original_count - len(all_sequences)} sequences "
            f"longer than {args.max_length}aa"
        )

    # Sort by sequence length for efficient processing (less padding waste)
    sorted_ids = sorted(all_sequences.keys(), key=lambda sid: len(all_sequences[sid]))
    sorted_sequences = {sid: all_sequences[sid] for sid in sorted_ids}

    print(f"\nTotal sequences to process: {len(sorted_sequences)}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print()

    # Check for checkpoint/resume
    output_path = Path(args.output)
    checkpoint_dir = get_checkpoint_dir(output_path)
    completed_ids: set[str] = set()
    collected_ids: list[str] = []
    collected_embeddings: list[np.ndarray] = []
    collected_sources: list[str] = []
    start_offset = 0

    if args.resume:
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            completed_ids = set(checkpoint["ids"])
            collected_ids = checkpoint["ids"]
            collected_embeddings = [checkpoint["embeddings"]]
            collected_sources = checkpoint["source_dbs"]
            start_offset = checkpoint["processed_count"]
            print(
                f"Resuming from checkpoint: {start_offset} proteins already processed"
            )
            print()
        else:
            print("No checkpoint found, starting from scratch")
            print()

    # Filter out already-processed sequences
    remaining = {
        sid: seq for sid, seq in sorted_sequences.items()
        if sid not in completed_ids
    }

    if not remaining:
        print("All sequences already processed!")
        if collected_embeddings:
            print("Saving final output...")
            # Fall through to save logic
        else:
            sys.exit(0)

    # Load model (encoder-only to save ~6.5 GB memory)
    import re
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    from vhold.features.prostt5 import prepare_prostt5_sequence, get_device

    device = get_device(args.device)
    print(f"Loading T5EncoderModel (encoder-only, ~4.8 GB)...")
    load_start = time.time()

    from vhold.utils.constants import PROSTT5_MODEL_NAME, get_model_dir

    model_dir = get_model_dir()
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(
        PROSTT5_MODEL_NAME, cache_dir=model_dir, do_lower_case=False
    )
    model = T5EncoderModel.from_pretrained(PROSTT5_MODEL_NAME, cache_dir=model_dir)
    model = model.to(device).float()
    model.eval()

    print(f"Model loaded in {time.time() - load_start:.1f}s")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.1f}M ({total_params * 4 / 1e9:.2f} GB)")
    print()

    # Process sequences
    remaining_ids = list(remaining.keys())
    remaining_seqs = [remaining[sid] for sid in remaining_ids]
    total_remaining = len(remaining_ids)
    batch_size = args.batch_size
    processed_in_session = 0
    session_start = time.time()

    print(f"Processing {total_remaining} remaining sequences...")
    print()

    batch_embeddings: list[np.ndarray] = []
    batch_ids: list[str] = []
    batch_sources: list[str] = []

    try:
        for i in range(0, total_remaining, batch_size):
            batch_end = min(i + batch_size, total_remaining)
            batch_seq_ids = remaining_ids[i:batch_end]
            batch_seqs = remaining_seqs[i:batch_end]

            # Prepare and tokenize
            prepared = [prepare_prostt5_sequence(s) for s in batch_seqs]
            inputs = tokenizer(
                prepared,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                output = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

            # Mean pool and normalize
            hidden = output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            emb_np = pooled.cpu().numpy()

            norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            emb_np = emb_np / norms

            # Collect
            batch_embeddings.append(emb_np)
            batch_ids.extend(batch_seq_ids)
            batch_sources.extend(source_map[sid] for sid in batch_seq_ids)
            processed_in_session += len(batch_seq_ids)

            # Progress
            total_processed = start_offset + processed_in_session
            elapsed = time.time() - session_start
            rate = processed_in_session / elapsed if elapsed > 0 else 0
            eta = (total_remaining - processed_in_session) / rate if rate > 0 else 0

            if processed_in_session % 100 < batch_size or i + batch_size >= total_remaining:
                print(
                    f"  [{total_processed}/{start_offset + total_remaining}] "
                    f"{rate:.1f} prot/s, "
                    f"ETA: {format_eta(eta)}, "
                    f"seq_len: {max(len(s) for s in batch_seqs)}aa"
                )

            # Checkpoint
            if processed_in_session % args.checkpoint_interval < batch_size:
                print(f"  Saving checkpoint ({total_processed} proteins)...")
                # Merge all embeddings so far
                all_embs = collected_embeddings + batch_embeddings
                all_ids_list = collected_ids + batch_ids
                all_sources_list = collected_sources + batch_sources
                merged_emb = np.vstack(all_embs) if all_embs else np.empty((0, 1024))

                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    ids=all_ids_list,
                    embeddings=merged_emb,
                    source_dbs=all_sources_list,
                    processed_count=total_processed,
                )

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        all_embs = collected_embeddings + batch_embeddings
        all_ids_list = collected_ids + batch_ids
        all_sources_list = collected_sources + batch_sources
        total_processed = start_offset + processed_in_session

        if all_embs:
            merged_emb = np.vstack(all_embs)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                ids=all_ids_list,
                embeddings=merged_emb,
                source_dbs=all_sources_list,
                processed_count=total_processed,
            )
            print(f"Checkpoint saved: {total_processed} proteins")
            print(f"Resume with: python {sys.argv[0]} --resume --output {args.output}")
        sys.exit(1)

    # Merge all results
    all_embs = collected_embeddings + batch_embeddings
    all_ids_list = collected_ids + batch_ids
    all_sources_list = collected_sources + batch_sources

    if not all_embs:
        print("ERROR: No embeddings produced.")
        sys.exit(1)

    embeddings = np.vstack(all_embs)
    total_elapsed = time.time() - session_start

    print(f"\nExtraction complete in {format_eta(total_elapsed)}")
    print(f"  Session: {processed_in_session} proteins in {format_eta(total_elapsed)}")
    if total_elapsed > 0:
        print(f"  Rate: {processed_in_session / total_elapsed:.1f} proteins/second")
    print(f"  Total embeddings: {embeddings.shape}")

    # Save as compressed .npz with float16 embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_fp16 = embeddings.astype(np.float16)
    protein_ids = np.array(all_ids_list)
    source_dbs_arr = np.array(all_sources_list)

    np.savez_compressed(
        output_path,
        embeddings=embeddings_fp16,
        protein_ids=protein_ids,
        source_dbs=source_dbs_arr,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Proteins: {len(protein_ids)}")
    print(f"  BFVD: {sum(1 for s in all_sources_list if s == 'bfvd')}")
    print(f"  Viro3D: {sum(1 for s in all_sources_list if s == 'viro3d')}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Storage: float16")

    # Clean up checkpoints
    if checkpoint_dir.exists():
        for f in checkpoint_dir.iterdir():
            f.unlink()
        checkpoint_dir.rmdir()
        print("\nCheckpoint directory cleaned up")


if __name__ == "__main__":
    main()
