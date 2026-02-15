#!/usr/bin/env python3
"""Train contrastive LoRA adapter for ProstT5 encoder.

Fine-tunes ProstT5 encoder with LoRA + supervised contrastive loss so that
functionally similar viral proteins produce closer embeddings. Uses multi-
granularity loss: category-level (11 classes) + Pfam-level contrastive losses.

Based on CLEAN (Yu et al., Science 2023) for contrastive protein learning.

Usage:
    # Train with default settings:
    python scripts/train_contrastive.py --output results/contrastive/

    # Quick test run:
    python scripts/train_contrastive.py --output /tmp/test \
        --epochs 2 --max-proteins 1000 --device cpu

    # Full training with installation:
    python scripts/train_contrastive.py --output results/contrastive/ \
        --epochs 10 --device mps --install
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.databases.bfvd import load_bfvd_enriched_metadata
from vhold.databases.viro3d import (
    get_annotation_expansion,
    get_viro3d_annotation,
    load_viro3d_metadata,
)
from vhold.features.contrastive import (
    CategoryBalancedSampler,
    ContrastiveDataset,
    MultiGranularityLoss,
    collate_contrastive,
    mean_pool_and_normalize,
)
from vhold.results.categories import (
    ALL_CATEGORIES,
    AnnotationEvidence,
    get_classification_source,
)
from vhold.utils.constants import get_db_dir, get_model_dir


# ============================================================================
# Sequence extraction
# ============================================================================


def extract_sequences_from_foldseek_db(db_path: Path) -> dict[str, str]:
    """Extract protein sequences from a Foldseek database.

    Uses foldseek convert2fasta to dump amino acid sequences.
    """
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = ["foldseek", "convert2fasta", str(db_path), tmp_path]
        subprocess.run(cmd, check=True, capture_output=True)
        from vhold.io.fasta import read_fasta

        records = read_fasta(Path(tmp_path))
        sequences = {rec.id: rec.sequence for rec in records.values()}
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return sequences


def load_sequences(
    db_dir: Path,
    bfvd_fasta: Path | None = None,
    viro3d_fasta: Path | None = None,
) -> dict[str, str]:
    """Load sequences from FASTA files or extract from Foldseek DBs.

    Returns:
        Dict mapping protein_id to amino acid sequence
    """
    all_sequences = {}

    # BFVD sequences
    if bfvd_fasta and bfvd_fasta.exists():
        print(f"Loading BFVD sequences from {bfvd_fasta}")
        from vhold.io.fasta import read_fasta

        records = read_fasta(bfvd_fasta)
        for rec in records.values():
            all_sequences[rec.id] = rec.sequence
        print(f"  Loaded {len(records)} BFVD sequences")
    else:
        bfvd_db = db_dir / "bfvd" / "bfvd"
        if bfvd_db.exists() or Path(str(bfvd_db) + ".dbtype").exists():
            print("Extracting BFVD sequences from Foldseek DB...")
            seqs = extract_sequences_from_foldseek_db(bfvd_db)
            all_sequences.update(seqs)
            print(f"  Extracted {len(seqs)} BFVD sequences")
        else:
            print("WARNING: BFVD database not found, skipping")

    # Viro3D sequences
    if viro3d_fasta and viro3d_fasta.exists():
        print(f"Loading Viro3D sequences from {viro3d_fasta}")
        from vhold.io.fasta import read_fasta

        records = read_fasta(viro3d_fasta)
        for rec in records.values():
            all_sequences[rec.id] = rec.sequence
        print(f"  Loaded {len(records)} Viro3D sequences")
    else:
        viro3d_db = db_dir / "viro3d" / "viro3d"
        if viro3d_db.exists() or Path(str(viro3d_db) + ".dbtype").exists():
            print("Extracting Viro3D sequences from Foldseek DB...")
            seqs = extract_sequences_from_foldseek_db(viro3d_db)
            all_sequences.update(seqs)
            print(f"  Extracted {len(seqs)} Viro3D sequences")
        else:
            print("WARNING: Viro3D database not found, skipping")

    print(f"Total sequences loaded: {len(all_sequences)}")
    return all_sequences


# ============================================================================
# Label generation (reuses train_classifier.py patterns)
# ============================================================================


def generate_labels(
    protein_ids: list[str],
    source_db_map: dict[str, str],
    db_dir: Path,
) -> dict[str, tuple[str, str, str | None]]:
    """Generate category and Pfam labels for proteins.

    Returns:
        Dict mapping protein_id to (category, source, pfam_domain_or_None)
    """
    # Load metadata sources
    enriched = load_bfvd_enriched_metadata(db_dir)
    if not enriched:
        print("WARNING: Enriched BFVD metadata not found")

    try:
        viro3d_meta = load_viro3d_metadata(db_dir)
    except FileNotFoundError:
        viro3d_meta = None
        print("WARNING: Viro3D metadata not found")

    try:
        expansion = get_annotation_expansion(db_dir)
        expansion.load_all()
    except Exception:
        expansion = None

    labels = {}
    for pid in protein_ids:
        source = source_db_map.get(pid, "")

        if source == "bfvd" and enriched and pid in enriched:
            row = enriched[pid]
            description = row.get("protein_name", "")
            gene = row.get("gene_name", "")
            evidence = AnnotationEvidence(
                pfam=row.get("pfam_domains") or None,
                go_bp=row.get("go_bp") or None,
                go_mf=row.get("go_mf") or None,
            )
            category, src = get_classification_source(description, gene, evidence)
            pfam = row.get("pfam_domains") or None
            labels[pid] = (category, src, pfam)

        elif source == "viro3d" and viro3d_meta is not None:
            ann = get_viro3d_annotation(
                pid,
                viro3d_meta,
                annotations_df=None,
                annotation_expansion=expansion,
            )
            if ann:
                description = ann.get("description", "")
                gene = ann.get("gene", "")
                evidence = AnnotationEvidence(
                    pfam=ann.get("pfam"),
                    go_bp=ann.get("go_bp"),
                    go_mf=ann.get("go_mf"),
                    superfamily=ann.get("superfamily"),
                )
                category, src = get_classification_source(description, gene, evidence)
                pfam = ann.get("pfam")
                labels[pid] = (category, src, pfam)

    return labels


def build_pfam_label_space(
    labels: dict[str, tuple[str, str, str | None]],
    min_members: int = 10,
) -> dict[str, int]:
    """Build Pfam label space, keeping only families with enough members.

    Args:
        labels: Dict from generate_labels()
        min_members: Minimum family size to include

    Returns:
        Dict mapping Pfam domain string to integer index
    """
    pfam_counts: Counter = Counter()
    for _pid, (_cat, _src, pfam) in labels.items():
        if pfam:
            # Take first Pfam domain if multiple (semicolon-separated)
            primary_pfam = pfam.split(";")[0].strip()
            if primary_pfam:
                pfam_counts[primary_pfam] += 1

    # Keep families with enough members
    pfam_to_idx = {}
    idx = 0
    for pfam_name, count in pfam_counts.most_common():
        if count >= min_members:
            pfam_to_idx[pfam_name] = idx
            idx += 1

    print(f"Pfam label space: {len(pfam_to_idx)} families (>= {min_members} members)")
    if pfam_to_idx:
        top5 = list(pfam_to_idx.keys())[:5]
        print(f"  Top families: {', '.join(top5)}")

    return pfam_to_idx


# ============================================================================
# LoRA setup
# ============================================================================


def setup_lora(model, rank: int = 16, target_layers: int = 6):
    """Apply LoRA adapters to the top encoder layers.

    Targets query and value projections in the self-attention of the
    last `target_layers` layers.

    Args:
        model: ProstT5 model (T5ForConditionalGeneration or T5EncoderModel)
        rank: LoRA rank (default: 16)
        target_layers: Number of top encoder layers to adapt (default: 6)

    Returns:
        PeftModel with LoRA adapters
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        print("ERROR: peft not installed. Run: pip install 'vhold[contrastive]'")
        sys.exit(1)

    # Determine encoder layer count
    n_layers = len(model.encoder.block)
    start_layer = max(0, n_layers - target_layers)

    # Build target module patterns for top layers
    target_modules = []
    for layer_idx in range(start_layer, n_layers):
        target_modules.append(f"encoder.block.{layer_idx}.layer.0.SelfAttention.q")
        target_modules.append(f"encoder.block.{layer_idx}.layer.0.SelfAttention.v")

    print(f"\nLoRA configuration:")
    print(f"  Rank: {rank}")
    print(f"  Target layers: {start_layer}-{n_layers - 1} ({target_layers} layers)")
    print(f"  Target modules: {len(target_modules)} (q + v per layer)")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,  # alpha = rank → scaling = 1.0
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return peft_model


# ============================================================================
# Training data construction
# ============================================================================


def build_training_data(
    db_dir: Path,
    bfvd_fasta: Path | None = None,
    viro3d_fasta: Path | None = None,
    max_proteins: int | None = None,
    val_fraction: float = 0.15,
    min_pfam_members: int = 10,
    seed: int = 42,
    case_studies_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Build training and validation datasets.

    Returns:
        (train_data, val_data) where each dict has keys:
        protein_ids, sequences, category_labels, pfam_labels, stats
    """
    # Load embedding DB for protein IDs and source mappings
    from vhold.databases.embeddings import get_embedding_db_path

    emb_path = get_embedding_db_path(db_dir)
    if not emb_path.exists():
        print(f"ERROR: Embedding database not found at {emb_path}")
        sys.exit(1)

    print(f"Loading embedding database metadata from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    all_protein_ids = list(data["protein_ids"])
    source_dbs = list(data["source_dbs"])
    source_db_map = dict(zip(all_protein_ids, source_dbs))
    print(f"  {len(all_protein_ids)} proteins in embedding DB")

    # Load case study IDs to exclude
    exclude_ids = set()
    if case_studies_dir and case_studies_dir.exists():
        # Reuse from train_classifier
        gt_files = list(case_studies_dir.glob("**/ground_truth.json"))
        for gt_file in gt_files:
            with open(gt_file) as f:
                gt_data = json.load(f)
            if "entries" in gt_data:
                for entry in gt_data["entries"]:
                    exclude_ids.add(entry.get("protein_id", ""))
                    if entry.get("uniprot_id"):
                        exclude_ids.add(entry["uniprot_id"])
            if "proteins" in gt_data:
                exclude_ids.update(gt_data["proteins"].keys())
        print(f"Excluding {len(exclude_ids)} case study proteins")

    # Load sequences
    print("\nLoading sequences...")
    sequences = load_sequences(db_dir, bfvd_fasta, viro3d_fasta)

    # Generate labels
    print("\nGenerating labels...")
    labels = generate_labels(all_protein_ids, source_db_map, db_dir)
    print(f"  {len(labels)} proteins labeled")

    # Build Pfam label space
    pfam_to_idx = build_pfam_label_space(labels, min_members=min_pfam_members)

    # Category label space
    category_to_idx = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

    # Filter to proteins with: sequence + non-unknown category + not in case study
    protein_ids = []
    seqs = []
    cat_labels = []
    pfam_labels = []
    excluded_unknown = 0
    excluded_no_seq = 0
    excluded_no_label = 0
    excluded_case_study = 0

    for pid in all_protein_ids:
        if pid in exclude_ids:
            excluded_case_study += 1
            continue
        if pid not in labels:
            excluded_no_label += 1
            continue
        category, _src, pfam = labels[pid]
        if category == "unknown":
            excluded_unknown += 1
            continue
        if pid not in sequences:
            excluded_no_seq += 1
            continue

        protein_ids.append(pid)
        seqs.append(sequences[pid])
        cat_labels.append(category_to_idx[category])

        # Pfam label
        if pfam:
            primary_pfam = pfam.split(";")[0].strip()
            pfam_labels.append(pfam_to_idx.get(primary_pfam, -1))
        else:
            pfam_labels.append(-1)

    print(f"\nDataset construction:")
    print(f"  Kept: {len(protein_ids)}")
    print(f"  Excluded (unknown): {excluded_unknown}")
    print(f"  Excluded (no sequence): {excluded_no_seq}")
    print(f"  Excluded (no label): {excluded_no_label}")
    print(f"  Excluded (case study): {excluded_case_study}")

    # Optionally limit dataset size
    if max_proteins and len(protein_ids) > max_proteins:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(protein_ids), max_proteins, replace=False)
        indices.sort()
        protein_ids = [protein_ids[i] for i in indices]
        seqs = [seqs[i] for i in indices]
        cat_labels = [cat_labels[i] for i in indices]
        pfam_labels = [pfam_labels[i] for i in indices]
        print(f"  Limited to {max_proteins} proteins")

    cat_labels = np.array(cat_labels, dtype=np.int64)
    pfam_labels = np.array(pfam_labels, dtype=np.int64)

    # Category distribution
    cat_counts = Counter(cat_labels.tolist())
    print(f"\nCategory distribution:")
    for idx in sorted(cat_counts.keys()):
        cat_name = ALL_CATEGORIES[idx]
        count = cat_counts[idx]
        pct = count / len(cat_labels) * 100
        print(f"  {cat_name}: {count} ({pct:.1f}%)")

    # Stratified split
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(protein_ids))
    rng.shuffle(all_indices)

    train_indices = []
    val_indices = []
    for class_idx in sorted(cat_counts.keys()):
        class_mask = cat_labels == class_idx
        class_indices = all_indices[class_mask[all_indices]]
        n_val = max(1, int(len(class_indices) * val_fraction))
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    def subset(indices):
        return {
            "protein_ids": [protein_ids[i] for i in indices],
            "sequences": [seqs[i] for i in indices],
            "category_labels": cat_labels[indices],
            "pfam_labels": pfam_labels[indices],
        }

    train_data = subset(train_indices)
    val_data = subset(val_indices)

    stats = {
        "total_proteins": len(all_protein_ids),
        "kept": len(protein_ids),
        "excluded_unknown": excluded_unknown,
        "excluded_no_seq": excluded_no_seq,
        "excluded_no_label": excluded_no_label,
        "excluded_case_study": excluded_case_study,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_categories": len(cat_counts),
        "n_pfam_families": len(pfam_to_idx),
        "category_counts": {ALL_CATEGORIES[k]: v for k, v in cat_counts.items()},
    }

    print(f"\nTrain: {len(train_indices)}, Val: {len(val_indices)}")
    return train_data, val_data, stats


# ============================================================================
# Training loop
# ============================================================================


def train_epoch(
    model,
    dataloader: DataLoader,
    loss_fn: MultiGranularityLoss,
    optimizer,
    scheduler,
    device: torch.device,
    gradient_accumulation_steps: int = 4,
) -> dict:
    """Train one epoch.

    Returns:
        Dict with avg_loss, category_loss, pfam_loss, n_batches
    """
    model.train()
    total_loss = 0.0
    total_cat_loss = 0.0
    total_pfam_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        category_labels = batch["category_labels"].to(device)
        pfam_labels = batch["pfam_labels"].to(device)

        # Forward through encoder
        outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Mean pool + L2 normalize
        embeddings = mean_pool_and_normalize(hidden_states, attention_mask)

        # Contrastive loss
        loss, details = loss_fn(embeddings, category_labels, pfam_labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += details["total"]
        total_cat_loss += details["category_loss"]
        total_pfam_loss += details["pfam_loss"]
        n_batches += 1

    # Handle remaining gradients
    if n_batches % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "avg_loss": total_loss / max(n_batches, 1),
        "category_loss": total_cat_loss / max(n_batches, 1),
        "pfam_loss": total_pfam_loss / max(n_batches, 1),
        "n_batches": n_batches,
    }


@torch.no_grad()
def validate(
    model,
    dataloader: DataLoader,
    loss_fn: MultiGranularityLoss,
    device: torch.device,
) -> dict:
    """Validate and compute retrieval metrics.

    Returns:
        Dict with avg_loss, map_at_1, intra_inter_ratio
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_embeddings = []
    all_cat_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        category_labels = batch["category_labels"].to(device)
        pfam_labels = batch["pfam_labels"].to(device)

        outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        embeddings = mean_pool_and_normalize(hidden_states, attention_mask)

        loss, details = loss_fn(embeddings, category_labels, pfam_labels)
        total_loss += details["total"]
        n_batches += 1

        all_embeddings.append(embeddings.cpu())
        all_cat_labels.append(category_labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # Compute MAP@1 on collected embeddings
    all_emb = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_cat_labels, dim=0)

    map_at_1 = 0.0
    intra_sim = 0.0
    inter_sim = 0.0
    n_intra = 0
    n_inter = 0

    if len(all_emb) > 1:
        sim_matrix = torch.mm(all_emb, all_emb.t())
        sim_matrix.fill_diagonal_(-1e9)  # Exclude self

        # MAP@1: is the nearest neighbor the same class?
        top1_idx = sim_matrix.argmax(dim=1)
        top1_labels = all_labels[top1_idx]
        map_at_1 = (top1_labels == all_labels).float().mean().item()

        # Intra/inter-class cosine similarity
        for i in range(len(all_labels)):
            for j in range(i + 1, min(len(all_labels), i + 100)):
                s = torch.dot(all_emb[i], all_emb[j]).item()
                if all_labels[i] == all_labels[j]:
                    intra_sim += s
                    n_intra += 1
                else:
                    inter_sim += s
                    n_inter += 1

    intra_avg = intra_sim / max(n_intra, 1)
    inter_avg = inter_sim / max(n_inter, 1)
    ratio = intra_avg / max(abs(inter_avg), 1e-8) if inter_avg != 0 else float("inf")

    return {
        "avg_loss": avg_loss,
        "map_at_1": map_at_1,
        "intra_cos": round(intra_avg, 4),
        "inter_cos": round(inter_avg, 4),
        "intra_inter_ratio": round(ratio, 2),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train contrastive LoRA adapter for ProstT5 encoder"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for adapter and report",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory for ProstT5 (default: auto-detect)",
    )
    parser.add_argument(
        "--bfvd-fasta", type=str, default=None,
        help="BFVD sequences FASTA file (default: extract from Foldseek DB)",
    )
    parser.add_argument(
        "--viro3d-fasta", type=str, default=None,
        help="Viro3D sequences FASTA file (default: extract from Foldseek DB)",
    )
    parser.add_argument(
        "--case-studies-dir", type=str, default=None,
        help="Case studies directory for test set exclusion",
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, mps, cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per forward pass (default: 64)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch-size * this)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-layers", type=int, default=6,
                        help="Number of top encoder layers to adapt")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--category-weight", type=float, default=0.7)
    parser.add_argument("--pfam-weight", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience on MAP@1")
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit dataset size for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--install", action="store_true",
        help="Copy adapter to ~/.vhold/models/contrastive_lora/ after training",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Case studies directory
    if args.case_studies_dir:
        cs_dir = Path(args.case_studies_dir)
    else:
        cs_dir = Path(__file__).resolve().parent.parent / "case_studies"

    # ========================================================================
    # Build training data
    # ========================================================================
    print("\n" + "=" * 60)
    print("Building training data")
    print("=" * 60)
    start = time.time()

    bfvd_fasta = Path(args.bfvd_fasta) if args.bfvd_fasta else None
    viro3d_fasta = Path(args.viro3d_fasta) if args.viro3d_fasta else None

    train_data, val_data, dataset_stats = build_training_data(
        db_dir=db_dir,
        bfvd_fasta=bfvd_fasta,
        viro3d_fasta=viro3d_fasta,
        max_proteins=args.max_proteins,
        seed=args.seed,
        case_studies_dir=cs_dir if cs_dir.exists() else None,
    )
    print(f"Data preparation: {time.time() - start:.1f}s")

    if dataset_stats["n_train"] == 0:
        print("ERROR: No training samples. Check metadata and sequence files.")
        sys.exit(1)

    # ========================================================================
    # Load model and apply LoRA
    # ========================================================================
    print("\n" + "=" * 60)
    print("Loading ProstT5 and applying LoRA")
    print("=" * 60)

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    from vhold.utils.constants import PROSTT5_MODEL_NAME

    model_name = PROSTT5_MODEL_NAME
    if args.model_dir:
        model_path = Path(args.model_dir)
        if (model_path / "config.json").exists():
            model_name = str(model_path)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to encoder
    model = setup_lora(model, rank=args.lora_rank, target_layers=args.lora_layers)
    model = model.to(device)

    # ========================================================================
    # Create datasets and dataloaders
    # ========================================================================
    print("\n" + "=" * 60)
    print("Creating dataloaders")
    print("=" * 60)

    train_dataset = ContrastiveDataset(
        protein_ids=train_data["protein_ids"],
        sequences=train_data["sequences"],
        category_labels=train_data["category_labels"],
        pfam_labels=train_data["pfam_labels"],
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    val_dataset = ContrastiveDataset(
        protein_ids=val_data["protein_ids"],
        sequences=val_data["sequences"],
        category_labels=val_data["category_labels"],
        pfam_labels=val_data["pfam_labels"],
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    train_sampler = CategoryBalancedSampler(
        labels=train_data["category_labels"],
        batch_size=args.batch_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_contrastive,
        num_workers=0,
        pin_memory=device.type != "cpu",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_contrastive,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")

    # ========================================================================
    # Loss, optimizer, scheduler
    # ========================================================================
    loss_fn = MultiGranularityLoss(
        category_weight=args.category_weight,
        pfam_weight=args.pfam_weight,
        category_temperature=args.temperature,
        pfam_temperature=args.temperature * 0.7,  # Slightly lower for finer granularity
        hard_negative_weight=args.hard_negative_weight,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = min(args.warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTotal training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # ========================================================================
    # Training
    # ========================================================================
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_map1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    header = f"{'Epoch':>5} {'Loss':>8} {'CatLoss':>8} {'PfamLoss':>9} {'ValLoss':>8} {'MAP@1':>7} {'IntraCos':>9} {'InterCos':>9} {'LR':>10}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            device, args.gradient_accumulation,
        )

        # Validate
        val_stats = validate(model, val_loader, loss_fn, device)

        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>5} "
            f"{train_stats['avg_loss']:>8.4f} "
            f"{train_stats['category_loss']:>8.4f} "
            f"{train_stats['pfam_loss']:>9.4f} "
            f"{val_stats['avg_loss']:>8.4f} "
            f"{val_stats['map_at_1']:>7.4f} "
            f"{val_stats['intra_cos']:>9.4f} "
            f"{val_stats['inter_cos']:>9.4f} "
            f"{current_lr:>10.2e} "
            f"({epoch_time:.0f}s)"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_stats["avg_loss"], 4),
            "category_loss": round(train_stats["category_loss"], 4),
            "pfam_loss": round(train_stats["pfam_loss"], 4),
            "val_loss": round(val_stats["avg_loss"], 4),
            "map_at_1": round(val_stats["map_at_1"], 4),
            "intra_cos": val_stats["intra_cos"],
            "inter_cos": val_stats["inter_cos"],
            "epoch_time_s": round(epoch_time, 1),
        })

        # Early stopping on MAP@1
        if val_stats["map_at_1"] > best_map1:
            best_map1 = val_stats["map_at_1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save best adapter
            adapter_dir = output_dir / "best_adapter"
            model.save_pretrained(adapter_dir)
            print(f"  -> Saved best adapter (MAP@1={best_map1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ========================================================================
    # Save final outputs
    # ========================================================================
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)

    # Copy best adapter to output root
    best_adapter_dir = output_dir / "best_adapter"
    if best_adapter_dir.exists():
        import shutil

        final_adapter_dir = output_dir / "contrastive_lora"
        if final_adapter_dir.exists():
            shutil.rmtree(final_adapter_dir)
        shutil.copytree(best_adapter_dir, final_adapter_dir)
        print(f"Best adapter (epoch {best_epoch}) saved to {final_adapter_dir}")

        # Report adapter size
        adapter_size = sum(f.stat().st_size for f in final_adapter_dir.rglob("*") if f.is_file())
        print(f"Adapter size: {adapter_size / 1024:.1f} KB")
    else:
        final_adapter_dir = None
        print("WARNING: No adapter saved (training may have failed)")

    # Training report
    report = {
        "dataset": dataset_stats,
        "hyperparameters": {
            "epochs_trained": len(history),
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": warmup_steps,
            "lora_rank": args.lora_rank,
            "lora_layers": args.lora_layers,
            "max_seq_length": args.max_seq_length,
            "category_weight": args.category_weight,
            "pfam_weight": args.pfam_weight,
            "temperature": args.temperature,
            "hard_negative_weight": args.hard_negative_weight,
        },
        "best_epoch": best_epoch,
        "best_map_at_1": round(best_map1, 4),
        "history": history,
        "device": str(device),
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved to {report_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Best MAP@1: {best_map1:.4f} (epoch {best_epoch})")
    print(f"Training proteins: {dataset_stats['n_train']}")
    print(f"Validation proteins: {dataset_stats['n_val']}")
    if history:
        print(f"Final train loss: {history[-1]['train_loss']:.4f}")
        print(f"Final val loss: {history[-1]['val_loss']:.4f}")

    # Optionally install adapter
    if args.install and final_adapter_dir:
        from vhold.databases.lora import get_lora_adapter_path

        install_path = get_lora_adapter_path()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if install_path.exists():
            shutil.rmtree(install_path)
        shutil.copytree(final_adapter_dir, install_path)
        print(f"\nInstalled adapter to {install_path}")


if __name__ == "__main__":
    main()
