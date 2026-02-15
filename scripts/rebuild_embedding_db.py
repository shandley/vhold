#!/usr/bin/env python3
"""Rebuild embedding database with fine-tuned contrastive LoRA encoder.

Re-extracts all 436K reference embeddings using the LoRA-enhanced ProstT5
encoder, then saves to the same .npz format. Uses merge_and_unload() so
the adapter is permanently fused into the encoder weights.

Supports checkpoint/resume for the multi-day extraction process.

Usage:
    # Rebuild with installed adapter:
    python scripts/rebuild_embedding_db.py \
        --output ~/.vhold/databases/embeddings/vhold_embeddings.npz

    # With explicit adapter path:
    python scripts/rebuild_embedding_db.py \
        --adapter results/contrastive/contrastive_lora/ \
        --output results/embeddings/vhold_embeddings_lora.npz

    # Resume interrupted run:
    python scripts/rebuild_embedding_db.py \
        --output results/embeddings/vhold_embeddings_lora.npz --resume
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.features.contrastive import mean_pool_and_normalize
from vhold.utils.constants import PROSTT5_MODEL_NAME, get_db_dir, get_model_dir


def load_model_with_lora(
    adapter_path: Path | None = None,
    model_dir: Path | None = None,
    device: torch.device = torch.device("cpu"),
):
    """Load ProstT5 with LoRA adapter merged and unloaded.

    Returns:
        (model, tokenizer) where model has LoRA permanently fused
    """
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = PROSTT5_MODEL_NAME
    if model_dir and (model_dir / "config.json").exists():
        model_name = str(model_dir)

    print(f"Loading base model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Find adapter
    if adapter_path is None:
        from vhold.databases.lora import check_lora_adapter, get_lora_adapter_path

        if check_lora_adapter(model_dir):
            adapter_path = get_lora_adapter_path(model_dir)
        else:
            print("WARNING: No LoRA adapter found, using base model")
            model = model.to(device)
            model.eval()
            return model, tokenizer

    print(f"Loading LoRA adapter: {adapter_path}")
    try:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA merged and unloaded (zero runtime overhead)")
    except ImportError:
        print("ERROR: peft not installed. Run: pip install 'vhold[contrastive]'")
        sys.exit(1)

    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_all_embeddings(
    model,
    tokenizer,
    protein_ids: list[str],
    sequences: dict[str, str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 1024,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 1000,
) -> np.ndarray:
    """Extract embeddings for all proteins with checkpoint/resume support.

    Returns:
        (N, 1024) float16 array of L2-normalized embeddings
    """
    from vhold.features.prostt5 import prepare_prostt5_sequence

    n_total = len(protein_ids)
    dim = 1024

    # Check for existing checkpoint
    start_idx = 0
    checkpoint_path = checkpoint_dir / "rebuild_checkpoint.npz" if checkpoint_dir else None

    if checkpoint_path and checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        existing_emb = ckpt["embeddings"]
        start_idx = ckpt["next_idx"].item()
        embeddings = np.zeros((n_total, dim), dtype=np.float16)
        embeddings[:start_idx] = existing_emb[:start_idx]
        print(f"Resuming from checkpoint: {start_idx}/{n_total} done")
    else:
        embeddings = np.zeros((n_total, dim), dtype=np.float16)

    # Sort remaining by sequence length for efficient batching
    remaining = [(i, protein_ids[i]) for i in range(start_idx, n_total)]
    remaining.sort(key=lambda x: len(sequences.get(x[1], "")))

    model.eval()
    processed = 0
    batch_indices = []
    batch_seqs = []
    start_time = time.time()

    for orig_idx, pid in remaining:
        if pid not in sequences:
            continue

        batch_indices.append(orig_idx)
        batch_seqs.append(sequences[pid])

        if len(batch_seqs) >= batch_size:
            # Process batch
            prepared = [prepare_prostt5_sequence(s) for s in batch_seqs]
            tokens = tokenizer(
                prepared,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                padding=True,
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
                emb = mean_pool_and_normalize(outputs.last_hidden_state, attention_mask)

            emb_np = emb.cpu().numpy().astype(np.float16)
            for j, idx in enumerate(batch_indices):
                embeddings[idx] = emb_np[j]

            processed += len(batch_indices)
            batch_indices = []
            batch_seqs = []

            # Progress
            total_done = start_idx + processed
            if total_done % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1)
                remaining_count = n_total - total_done
                eta = remaining_count / max(rate, 1e-6)
                print(
                    f"  {total_done}/{n_total} "
                    f"({100 * total_done / n_total:.1f}%) "
                    f"{rate:.1f} prot/s, ETA {eta / 3600:.1f}h"
                )

            # Checkpoint
            if checkpoint_path and processed % checkpoint_interval == 0:
                np.savez_compressed(
                    checkpoint_path,
                    embeddings=embeddings,
                    next_idx=np.array(start_idx + processed),
                )

    # Process final batch
    if batch_seqs:
        prepared = [prepare_prostt5_sequence(s) for s in batch_seqs]
        tokens = tokenizer(
            prepared,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            emb = mean_pool_and_normalize(outputs.last_hidden_state, attention_mask)

        emb_np = emb.cpu().numpy().astype(np.float16)
        for j, idx in enumerate(batch_indices):
            embeddings[idx] = emb_np[j]
        processed += len(batch_indices)

    total_done = start_idx + processed
    elapsed = time.time() - start_time
    print(f"\nExtracted {total_done} embeddings in {elapsed / 3600:.1f}h")

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild embedding database with contrastive LoRA encoder"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output .npz path for rebuilt embeddings",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Path to LoRA adapter directory (default: auto-detect from ~/.vhold/models/)",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory (default: ~/.vhold/models)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")

    args = parser.parse_args()

    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()
    model_dir = Path(args.model_dir) if args.model_dir else get_model_dir()
    output_path = Path(args.output)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load existing embedding DB for protein IDs and source mappings
    from vhold.databases.embeddings import get_embedding_db_path

    emb_path = get_embedding_db_path(db_dir)
    if not emb_path.exists():
        print(f"ERROR: Existing embedding database not found at {emb_path}")
        sys.exit(1)

    print(f"Loading metadata from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    protein_ids = list(data["protein_ids"])
    source_dbs = list(data["source_dbs"])
    print(f"  {len(protein_ids)} proteins to re-embed")

    # Extract sequences from Foldseek DBs
    print("\nExtracting sequences from Foldseek databases...")
    import subprocess
    import tempfile

    sequences = {}
    for db_name in ["bfvd", "viro3d"]:
        db_path = db_dir / db_name / db_name
        if not (db_path.exists() or Path(str(db_path) + ".dbtype").exists()):
            print(f"  WARNING: {db_name} database not found, skipping")
            continue
        with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["foldseek", "convert2fasta", str(db_path), tmp_path],
                check=True, capture_output=True,
            )
            from vhold.io.fasta import read_fasta

            records = read_fasta(Path(tmp_path))
            for rec in records.values():
                sequences[rec.id] = rec.sequence
            print(f"  {db_name}: {len(records)} sequences")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print(f"Total sequences: {len(sequences)}")

    # Load model with LoRA
    adapter_path = Path(args.adapter) if args.adapter else None
    model, tokenizer = load_model_with_lora(adapter_path, model_dir, device)

    # Extract embeddings
    print("\n" + "=" * 60)
    print("Extracting embeddings")
    print("=" * 60)

    checkpoint_dir = output_path.parent if args.resume else None

    embeddings = extract_all_embeddings(
        model, tokenizer, protein_ids, sequences, device,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        protein_ids=np.array(protein_ids, dtype=object),
        source_dbs=np.array(source_dbs, dtype=object),
    )
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({file_size:.1f} MB)")

    # Clean up checkpoint
    if checkpoint_dir:
        ckpt_path = checkpoint_dir / "rebuild_checkpoint.npz"
        if ckpt_path.exists():
            ckpt_path.unlink()
            print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
