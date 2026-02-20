#!/usr/bin/env python3
"""Train disorder-aware MLP classifier on STARLING embeddings.

Filters disordered proteins (disorder_fraction >= 0.15) from the embedding DB,
extracts STARLING 512-dim ensemble-aware embeddings, then trains a small MLP
to classify functional categories.

Requires: pip install vhold[disorder,starling]

Usage:
    # Train with default settings:
    python scripts/train_disorder_classifier.py --output results/disorder_classifier/

    # With custom hyperparameters:
    python scripts/train_disorder_classifier.py \
        --output results/disorder_classifier/ \
        --epochs 50 --lr 0.0005

    # Install after training:
    python scripts/train_disorder_classifier.py \
        --output results/disorder_classifier/ --install
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.features.disorder import (
    check_metapredict_available,
    check_starling_available,
    classify_disorder,
    predict_disorder_batch,
    extract_starling_embeddings,
)
from vhold.features.disorder_classifier import DisorderClassifier
from vhold.results.categories import ALL_CATEGORIES
from vhold.utils.constants import (
    MODERATE_DISORDER_FRACTION,
    STARLING_EMBEDDING_DIM,
    get_db_dir,
)


def load_sequences_from_foldseek_db(db_dir: Path) -> dict[str, str]:
    """Load protein sequences from installed Foldseek databases.

    Falls back to embedding DB protein IDs if Foldseek DBs unavailable.
    """
    from vhold.features.alignment_validation import FoldseekSequenceIndex

    sequences = {}

    # Try BFVD
    bfvd_db = db_dir / "bfvd" / "bfvd"
    if bfvd_db.with_suffix(".dbtype").exists():
        try:
            idx = FoldseekSequenceIndex(bfvd_db)
            for pid in idx.keys():
                seq = idx.get_sequence(pid)
                if seq:
                    sequences[pid] = seq
            print(f"Loaded {len(sequences)} BFVD sequences")
        except Exception as e:
            print(f"Warning: Could not load BFVD sequences: {e}")

    # Try Viro3D
    n_before = len(sequences)
    viro3d_db = db_dir / "viro3d" / "foldseekViro3D" / "viro3d"
    if viro3d_db.with_suffix(".dbtype").exists():
        try:
            idx = FoldseekSequenceIndex(viro3d_db)
            for pid in idx.keys():
                if pid not in sequences:
                    seq = idx.get_sequence(pid)
                    if seq:
                        sequences[pid] = seq
            print(f"Loaded {len(sequences) - n_before} Viro3D sequences")
        except Exception as e:
            print(f"Warning: Could not load Viro3D sequences: {e}")

    return sequences


def generate_labels(
    protein_ids: np.ndarray,
    source_dbs: np.ndarray,
    db_dir: Path,
) -> dict[str, str]:
    """Generate functional category labels from metadata.

    Reuses the same label generation as train_classifier.py.
    """
    from vhold.databases.bfvd import load_bfvd_enriched_metadata
    from vhold.databases.viro3d import (
        load_viro3d_metadata,
        get_annotation_expansion,
        get_viro3d_annotation,
    )
    from vhold.results.categories import (
        AnnotationEvidence,
        get_classification_source,
    )

    labels = {}

    # BFVD labels
    enriched = load_bfvd_enriched_metadata(db_dir)
    if enriched:
        for pid, src in zip(protein_ids, source_dbs):
            if src != "bfvd":
                continue
            pid_str = str(pid)
            if pid_str not in enriched:
                continue
            row = enriched[pid_str]
            evidence = AnnotationEvidence(
                pfam=row.get("pfam_domains") or None,
                go_bp=row.get("go_bp") or None,
                go_mf=row.get("go_mf") or None,
            )
            category, _ = get_classification_source(
                row.get("protein_name", ""), row.get("gene_name", ""), evidence
            )
            labels[pid_str] = category

    # Viro3D labels
    try:
        metadata = load_viro3d_metadata(db_dir)
        try:
            expansion = get_annotation_expansion(db_dir)
            expansion.load_all()
        except Exception:
            expansion = None

        for pid, src in zip(protein_ids, source_dbs):
            if src != "viro3d":
                continue
            pid_str = str(pid)
            ann = get_viro3d_annotation(
                pid_str, metadata, annotations_df=None,
                annotation_expansion=expansion,
            )
            if ann is None:
                continue
            evidence = AnnotationEvidence(
                pfam=ann.get("pfam"),
                go_bp=ann.get("go_bp"),
                go_mf=ann.get("go_mf"),
                superfamily=ann.get("superfamily"),
            )
            category, _ = get_classification_source(
                ann.get("description", ""), ann.get("gene", ""), evidence
            )
            labels[pid_str] = category
    except FileNotFoundError:
        print("Warning: Viro3D metadata not found")

    print(f"Generated {len(labels)} total labels")
    return labels


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.where(counts > 0, counts, 1.0)
    weights = len(y) / (num_classes * counts)
    return torch.from_numpy(weights).float()


def compute_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    """Compute per-class and macro F1 scores."""
    results = {}
    f1_scores = []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = (targets == c).sum()
        if support > 0:
            f1_scores.append(f1)

        results[ALL_CATEGORIES[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }

    results["macro_f1"] = round(np.mean(f1_scores), 4) if f1_scores else 0.0
    return results


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int = STARLING_EMBEDDING_DIM,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    num_classes: int = 11,
) -> tuple[DisorderClassifier, dict]:
    """Train the disorder classifier."""
    model = DisorderClassifier(
        input_dim=input_dim, num_classes=num_classes,
        hidden_dims=(256, 128), dropout=0.3,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: DisorderClassifier, parameters: {param_count:,}")
    print(f"Architecture: {input_dim} -> 256 -> 128 -> {num_classes}")

    class_weights = compute_class_weights(y_train, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val Acc':>8} {'Val F1':>7} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        n_val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                n_val_batches += 1
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.numpy())
                all_targets.append(y_batch.numpy())

        val_loss /= n_val_batches
        all_preds_arr = np.concatenate(all_preds)
        all_targets_arr = np.concatenate(all_targets)
        val_acc = (all_preds_arr == all_targets_arr).mean()
        metrics = compute_f1(all_preds_arr, all_targets_arr, num_classes)
        val_f1 = metrics["macro_f1"]

        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>5} {train_loss:>11.4f} {val_loss:>9.4f} {val_acc:>8.4f} {val_f1:>7.4f} {current_lr:>10.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch).argmax(dim=-1)
            all_preds.append(preds.numpy())
            all_targets.append(y_batch.numpy())

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)
    final_metrics = compute_f1(all_preds_arr, all_targets_arr, num_classes)

    stats = {
        "model_type": "disorder_mlp",
        "parameters": param_count,
        "best_val_f1": round(best_val_f1, 4),
        "final_metrics": final_metrics,
        "history": history,
        "hyperparameters": {
            "epochs": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "input_dim": input_dim,
        },
    }

    return model, stats


def main():
    parser = argparse.ArgumentParser(
        description="Train disorder-aware classifier on STARLING embeddings"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for checkpoint and report",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument(
        "--embedding-db", type=str, default=None,
        help="Path to embedding database .npz (default: auto-detect)",
    )
    parser.add_argument(
        "--disorder-threshold", type=float, default=0.5,
        help="Per-residue disorder threshold (default: 0.5)",
    )
    parser.add_argument(
        "--min-disorder-fraction", type=float, default=MODERATE_DISORDER_FRACTION,
        help=f"Minimum disorder fraction to include (default: {MODERATE_DISORDER_FRACTION})",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--starling-batch-size", type=int, default=32,
        help="Batch size for STARLING embedding extraction",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for STARLING (auto-detect if not specified)",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Copy checkpoint to ~/.vhold/models/disorder_classifier/",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_metapredict_available():
        print("ERROR: metapredict not installed. Run: pip install vhold[disorder]")
        sys.exit(1)
    if not check_starling_available():
        print("ERROR: STARLING not installed. Run: pip install vhold[starling]")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()

    # Find embedding database (for protein IDs and labels)
    if args.embedding_db:
        emb_path = Path(args.embedding_db)
    else:
        from vhold.databases.embeddings import get_embedding_db_path
        emb_path = get_embedding_db_path(db_dir)

    if not emb_path.exists():
        print(f"ERROR: Embedding database not found at {emb_path}")
        sys.exit(1)

    # Load embedding DB metadata
    print(f"Loading embedding database from {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    protein_ids = data["protein_ids"]
    source_dbs = data["source_dbs"]
    print(f"Loaded {len(protein_ids)} protein entries")

    # Load sequences from Foldseek databases
    print("\nLoading sequences from Foldseek databases...")
    sequences = load_sequences_from_foldseek_db(db_dir)
    if not sequences:
        print("ERROR: No sequences found. Foldseek databases must be installed.")
        sys.exit(1)

    # Filter to proteins in embedding DB that have sequences
    available = {str(pid) for pid in protein_ids} & set(sequences.keys())
    print(f"Proteins with both embeddings and sequences: {len(available)}")

    # Step 1: Run metapredict disorder prediction
    print("\n" + "=" * 60)
    print("Step 1: Disorder prediction (metapredict)")
    print("=" * 60)
    seq_subset = {pid: sequences[pid] for pid in available}
    start = time.time()
    disorder_results = predict_disorder_batch(
        seq_subset, threshold=args.disorder_threshold
    )
    print(f"Disorder prediction completed in {time.time() - start:.1f}s")

    # Filter to disordered proteins
    disordered_ids = [
        pid for pid, dr in disorder_results.items()
        if dr.disorder_fraction >= args.min_disorder_fraction
    ]
    print(f"\nDisordered proteins (fraction >= {args.min_disorder_fraction}): {len(disordered_ids)}")

    # Generate labels
    print("\nGenerating functional labels...")
    labels = generate_labels(protein_ids, source_dbs, db_dir)

    # Filter: disordered + labeled + not unknown
    labeled_disordered = [
        pid for pid in disordered_ids
        if pid in labels and labels[pid] != "unknown"
    ]
    print(f"Disordered proteins with non-unknown labels: {len(labeled_disordered)}")

    if len(labeled_disordered) < 100:
        print("ERROR: Too few training samples. Need at least 100 labeled disordered proteins.")
        sys.exit(1)

    # Step 2: Extract STARLING embeddings for disordered proteins
    print("\n" + "=" * 60)
    print("Step 2: STARLING embedding extraction")
    print("=" * 60)
    disordered_seqs = {pid: sequences[pid] for pid in labeled_disordered}
    disordered_disorder = {pid: disorder_results[pid] for pid in labeled_disordered}

    start = time.time()
    starling_ids, starling_embeddings = extract_starling_embeddings(
        disordered_seqs,
        disorder_results=disordered_disorder,
        batch_size=args.starling_batch_size,
        device=args.device,
    )
    print(f"STARLING extraction completed in {time.time() - start:.1f}s")
    print(f"Embeddings shape: {starling_embeddings.shape}")

    # Build training arrays
    category_to_idx = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}
    X_list = []
    y_list = []

    for i, pid in enumerate(starling_ids):
        if pid in labels:
            category = labels[pid]
            if category != "unknown" and category in category_to_idx:
                X_list.append(starling_embeddings[i])
                y_list.append(category_to_idx[category])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Category distribution
    cat_counts = Counter(y.tolist())
    print(f"\nCategory distribution ({len(X)} samples):")
    for idx in sorted(cat_counts.keys()):
        cat_name = ALL_CATEGORIES[idx]
        count = cat_counts[idx]
        pct = count / len(y) * 100
        print(f"  {cat_name}: {count} ({pct:.1f}%)")

    # Stratified split
    rng = np.random.RandomState(args.seed)
    val_fraction = 0.15
    train_indices = []
    val_indices = []

    indices = np.arange(len(X))
    rng.shuffle(indices)

    for class_idx in sorted(cat_counts.keys()):
        class_mask = y == class_idx
        class_indices = indices[class_mask[indices]]
        n_val = max(1, int(len(class_indices) * val_fraction))
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # Step 3: Train
    print("\n" + "=" * 60)
    print("Step 3: Training disorder classifier")
    print("=" * 60)
    start = time.time()
    model, training_stats = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=STARLING_EMBEDDING_DIM,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Training completed in {time.time() - start:.1f}s")

    # Save checkpoint
    checkpoint_path = output_dir / "vhold_disorder_classifier.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout_rate,
        "category_names": ALL_CATEGORIES,
        "training_stats": training_stats,
        "dataset_stats": {
            "total_disordered": len(disordered_ids),
            "labeled_disordered": len(labeled_disordered),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "disorder_threshold": args.disorder_threshold,
            "min_disorder_fraction": args.min_disorder_fraction,
            "category_counts": {ALL_CATEGORIES[k]: v for k, v in cat_counts.items()},
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / 1024:.1f} KB")

    # Save report
    report = {
        "dataset": checkpoint["dataset_stats"],
        "training": training_stats,
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved to {report_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    metrics = training_stats["final_metrics"]
    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")
    print(f"\nPer-class results:")
    print(f"{'Category':<20} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
    print("-" * 55)
    for cat in ALL_CATEGORIES:
        if cat in metrics and metrics[cat]["support"] > 0:
            m = metrics[cat]
            print(f"{cat:<20} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>6.4f} {m['support']:>8}")

    # Install
    if args.install:
        from vhold.databases.disorder_classifier import get_disorder_classifier_path
        install_path = get_disorder_classifier_path()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(checkpoint_path, install_path)
        print(f"\nInstalled disorder classifier to {install_path}")


if __name__ == "__main__":
    main()
