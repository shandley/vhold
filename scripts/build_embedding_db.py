#!/usr/bin/env python3
"""Build embedding database from BFVD + Viro3D reference proteins.

This is a one-time operation that pre-computes ProstT5 encoder embeddings
for all reference proteins. The resulting .npz file is used by vhold's
embedding triage mode (--triage) to quickly identify known proteins
without running the expensive decoder.

Requires a GPU for practical runtimes (~hours for 436K proteins).

Usage:
    python scripts/build_embedding_db.py \
        --db-dir ~/.vhold/databases \
        --output vhold_embeddings.npz \
        --device cuda \
        --batch-size 64

    # Or with explicit FASTA files:
    python scripts/build_embedding_db.py \
        --bfvd-fasta /path/to/bfvd_sequences.fasta \
        --viro3d-fasta /path/to/viro3d_sequences.fasta \
        --output vhold_embeddings.npz \
        --device cuda
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


def main():
    parser = argparse.ArgumentParser(
        description="Build vhold embedding database from reference proteins"
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
        default=64,
        help="Batch size for embedding extraction (default: 64)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length to process (default: 2048)",
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

    print(f"\nTotal sequences to process: {len(all_sequences)}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Extract embeddings
    from vhold.features.embeddings import EmbeddingExtractor

    extractor = EmbeddingExtractor(device=args.device)
    start_time = time.time()

    ids, embeddings = extractor.extract_batch(
        all_sequences,
        batch_size=args.batch_size,
        show_progress=True,
    )

    elapsed = time.time() - start_time
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Rate: {len(ids) / elapsed:.1f} proteins/second")
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build source_dbs array
    source_dbs = np.array([source_map[sid] for sid in ids])

    # Save as compressed .npz with float16 embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_fp16 = embeddings.astype(np.float16)
    protein_ids = np.array(ids)

    np.savez_compressed(
        output_path,
        embeddings=embeddings_fp16,
        protein_ids=protein_ids,
        source_dbs=source_dbs,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Proteins: {len(protein_ids)}")
    print(f"  BFVD: {sum(1 for s in source_dbs if s == 'bfvd')}")
    print(f"  Viro3D: {sum(1 for s in source_dbs if s == 'viro3d')}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Storage: float16")


if __name__ == "__main__":
    main()
