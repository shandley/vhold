#!/usr/bin/env python3
"""Train MLP classifier on frozen ProstT5 embeddings.

Generates training labels from BFVD/Viro3D metadata using the existing
classify_protein() hierarchy, then trains a lightweight MLP to predict
functional categories from 1024-dim ProstT5 encoder embeddings.

Usage:
    # Train with default settings:
    python scripts/train_classifier.py --output results/classifier/

    # With custom hyperparameters:
    python scripts/train_classifier.py \
        --output results/classifier/ \
        --epochs 50 --lr 0.0005 --batch-size 4096

    # Linear probe only (baseline):
    python scripts/train_classifier.py \
        --output results/classifier/ --model-type linear
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

from vhold.databases.bfvd import load_bfvd_enriched_metadata
from vhold.databases.viro3d import (
    load_viro3d_metadata,
    get_annotation_expansion,
    get_viro3d_annotation,
)
from vhold.features.classifier import FunctionalClassifier
from vhold.results.categories import (
    ALL_CATEGORIES,
    AnnotationEvidence,
    get_classification_source,
    classify_by_keywords,
)
from vhold.utils.constants import get_db_dir


def load_case_study_ids(case_studies_dir: Path) -> set[str]:
    """Load all protein IDs from case study ground truth files.

    These must be excluded from training to prevent test set leakage.
    """
    ids = set()
    gt_files = list(case_studies_dir.glob("**/ground_truth.json"))

    for gt_file in gt_files:
        with open(gt_file) as f:
            data = json.load(f)

        # Format 1: "entries" list (SARS-CoV-2 style)
        if "entries" in data:
            for entry in data["entries"]:
                pid = entry.get("protein_id", "")
                ids.add(pid)
                # Also add the UniProt accession
                uniprot = entry.get("uniprot_id", "")
                if uniprot:
                    ids.add(uniprot)

        # Format 2: "proteins" dict (eukaryotic viruses style)
        if "proteins" in data:
            ids.update(data["proteins"].keys())

    print(f"Loaded {len(ids)} case study protein IDs to exclude from training")
    return ids


def generate_bfvd_labels(
    protein_ids: np.ndarray,
    source_dbs: np.ndarray,
    db_dir: Path,
) -> dict[str, tuple[str, str]]:
    """Generate functional category labels for BFVD proteins.

    Returns:
        Dict mapping protein_id to (category, evidence_source)
    """
    enriched = load_bfvd_enriched_metadata(db_dir)
    if not enriched:
        print("WARNING: Enriched BFVD metadata not found, skipping BFVD labels")
        return {}

    labels = {}
    bfvd_ids = [pid for pid, src in zip(protein_ids, source_dbs) if src == "bfvd"]

    for pid in bfvd_ids:
        pid_str = str(pid)
        if pid_str not in enriched:
            continue

        row = enriched[pid_str]
        description = row.get("protein_name", "")
        gene = row.get("gene_name", "")

        evidence = AnnotationEvidence(
            pfam=row.get("pfam_domains") or None,
            go_bp=row.get("go_bp") or None,
            go_mf=row.get("go_mf") or None,
        )

        category, source = get_classification_source(description, gene, evidence)
        labels[pid_str] = (category, source)

    print(f"Generated {len(labels)} BFVD labels")
    return labels


def generate_viro3d_labels(
    protein_ids: np.ndarray,
    source_dbs: np.ndarray,
    db_dir: Path,
) -> dict[str, tuple[str, str]]:
    """Generate functional category labels for Viro3D proteins.

    Returns:
        Dict mapping protein_id to (category, evidence_source)
    """
    try:
        metadata = load_viro3d_metadata(db_dir)
    except FileNotFoundError:
        print("WARNING: Viro3D metadata not found, skipping Viro3D labels")
        return {}

    try:
        expansion = get_annotation_expansion(db_dir)
        expansion.load_all()
    except Exception:
        expansion = None

    labels = {}
    viro3d_ids = [pid for pid, src in zip(protein_ids, source_dbs) if src == "viro3d"]

    for pid in viro3d_ids:
        pid_str = str(pid)
        ann = get_viro3d_annotation(
            pid_str, metadata, annotations_df=None,
            annotation_expansion=expansion,
        )
        if ann is None:
            continue

        description = ann.get("description", "")
        gene = ann.get("gene", "")

        evidence = AnnotationEvidence(
            pfam=ann.get("pfam"),
            go_bp=ann.get("go_bp"),
            go_mf=ann.get("go_mf"),
            superfamily=ann.get("superfamily"),
        )

        category, source = get_classification_source(description, gene, evidence)
        labels[pid_str] = (category, source)

    print(f"Generated {len(labels)} Viro3D labels")
    return labels


def build_dataset(
    embedding_db_path: Path,
    db_dir: Path,
    exclude_ids: set[str],
    extra_labels_path: Path | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Build train/val arrays from embedding DB + generated labels.

    Returns:
        (X_train, y_train, X_val, y_val, category_names, stats)
    """
    print(f"Loading embedding database from {embedding_db_path}")
    data = np.load(embedding_db_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    protein_ids = data["protein_ids"]
    source_dbs = data["source_dbs"]
    print(f"Loaded {len(protein_ids)} embeddings")

    # Generate labels from both databases
    print("\nGenerating labels from BFVD enriched metadata...")
    bfvd_labels = generate_bfvd_labels(protein_ids, source_dbs, db_dir)
    print("Generating labels from Viro3D metadata...")
    viro3d_labels = generate_viro3d_labels(protein_ids, source_dbs, db_dir)

    all_labels = {**bfvd_labels, **viro3d_labels}
    print(f"\nTotal labeled proteins: {len(all_labels)}")

    # Load extra labels (e.g., from LLM classification)
    extra_applied = 0
    if extra_labels_path and extra_labels_path.exists():
        import json
        with open(extra_labels_path) as f:
            extra_data = json.load(f)
        protein_labels = extra_data.get("protein_labels", {})
        print(f"Loading extra labels from {extra_labels_path}: {len(protein_labels)} proteins")
        for pid, category in protein_labels.items():
            if pid in all_labels:
                old_cat, old_src = all_labels[pid]
                if old_cat == "unknown" and category != "unknown":
                    all_labels[pid] = (category, "llm")
                    extra_applied += 1
            else:
                if category != "unknown":
                    all_labels[pid] = (category, "llm")
                    extra_applied += 1
        print(f"Extra labels applied: {extra_applied} (overriding unknowns or adding new)")

    # Build category index
    category_to_idx = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

    # Filter: exclude unknowns, case study IDs, and proteins without labels
    X_list = []
    y_list = []
    kept = 0
    excluded_unknown = 0
    excluded_case_study = 0
    excluded_no_label = 0

    for i, pid in enumerate(protein_ids):
        pid_str = str(pid)

        # Exclude case study proteins
        if pid_str in exclude_ids:
            excluded_case_study += 1
            continue

        # Must have a label
        if pid_str not in all_labels:
            excluded_no_label += 1
            continue

        category, _source = all_labels[pid_str]

        # Exclude unknowns from training
        if category == "unknown":
            excluded_unknown += 1
            continue

        X_list.append(embeddings[i])
        y_list.append(category_to_idx[category])
        kept += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"\nDataset construction:")
    print(f"  Kept: {kept}")
    print(f"  Excluded (unknown): {excluded_unknown}")
    print(f"  Excluded (case study): {excluded_case_study}")
    print(f"  Excluded (no label): {excluded_no_label}")

    # Category distribution
    cat_counts = Counter(y.tolist())
    print(f"\nCategory distribution:")
    for idx in sorted(cat_counts.keys()):
        cat_name = ALL_CATEGORIES[idx]
        count = cat_counts[idx]
        pct = count / len(y) * 100
        print(f"  {cat_name}: {count} ({pct:.1f}%)")

    # Stratified split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    # Simple stratified split: for each class, split proportionally
    train_indices = []
    val_indices = []

    for class_idx in sorted(cat_counts.keys()):
        class_mask = y == class_idx
        class_indices = indices[class_mask[indices]]
        n_val = max(1, int(len(class_indices) * val_fraction))
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Shuffle
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    stats = {
        "total_embeddings": len(protein_ids),
        "total_labeled": len(all_labels),
        "kept": kept,
        "excluded_unknown": excluded_unknown,
        "excluded_case_study": excluded_case_study,
        "excluded_no_label": excluded_no_label,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "category_counts": {ALL_CATEGORIES[k]: v for k, v in cat_counts.items()},
    }

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    return X_train, y_train, X_val, y_val, ALL_CATEGORIES, stats


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    # Avoid division by zero for classes not in training set
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
    model_type: str = "mlp",
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    num_classes: int = 11,
) -> tuple[FunctionalClassifier, dict]:
    """Train the classifier.

    Returns:
        (trained_model, training_stats)
    """
    # Build model
    if model_type == "linear":
        model = FunctionalClassifier(
            input_dim=1024, num_classes=num_classes,
            hidden_dims=(), dropout=0.0,
        )
    else:
        model = FunctionalClassifier(
            input_dim=1024, num_classes=num_classes,
            hidden_dims=(512, 256), dropout=0.3,
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_type}, parameters: {param_count:,}")

    # Class weights
    class_weights = compute_class_weights(y_train, num_classes)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training loop
    best_val_f1 = 0.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val Acc':>8} {'Val F1':>7} {'LR':>10}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # Train
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

        # Validate
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
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_acc = (all_preds == all_targets).mean()
        metrics = compute_f1(all_preds, all_targets, num_classes)
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

        # Early stopping
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

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch).argmax(dim=-1)
            all_preds.append(preds.numpy())
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    final_metrics = compute_f1(all_preds, all_targets, num_classes)

    stats = {
        "model_type": model_type,
        "parameters": param_count,
        "best_val_f1": round(best_val_f1, 4),
        "final_metrics": final_metrics,
        "history": history,
        "hyperparameters": {
            "epochs": epoch,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    }

    return model, stats


def save_checkpoint(
    model: FunctionalClassifier,
    output_path: Path,
    training_stats: dict,
    dataset_stats: dict,
) -> None:
    """Save model checkpoint with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "mlp" if model.hidden_dims else "linear",
        "input_dim": model.input_dim,
        "num_classes": model.num_classes,
        "hidden_dims": list(model.hidden_dims),
        "dropout": model.dropout_rate,
        "category_names": ALL_CATEGORIES,
        "training_stats": training_stats,
        "dataset_stats": dataset_stats,
    }

    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Train MLP classifier on frozen ProstT5 embeddings"
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
        "--case-studies-dir", type=str, default=None,
        help="Case studies directory for test set exclusion",
    )
    parser.add_argument(
        "--model-type", type=str, choices=["mlp", "linear"], default="mlp",
        help="Model type: mlp (default) or linear (baseline)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-labels", type=str, default=None,
        help="JSON file with extra labels (from generate_llm_labels.py)",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Copy checkpoint to ~/.vhold/models/classifier/ after training",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()

    # Find embedding database
    if args.embedding_db:
        emb_path = Path(args.embedding_db)
    else:
        from vhold.databases.embeddings import get_embedding_db_path
        emb_path = get_embedding_db_path(db_dir)

    if not emb_path.exists():
        print(f"ERROR: Embedding database not found at {emb_path}")
        print("Run scripts/build_embedding_db.py first or pass --embedding-db")
        sys.exit(1)

    # Find case studies
    if args.case_studies_dir:
        cs_dir = Path(args.case_studies_dir)
    else:
        cs_dir = Path(__file__).resolve().parent.parent / "case_studies"

    exclude_ids = load_case_study_ids(cs_dir) if cs_dir.exists() else set()

    # Build dataset
    print("\n" + "=" * 60)
    print("Building dataset")
    print("=" * 60)
    start = time.time()
    extra_labels = Path(args.extra_labels) if args.extra_labels else None
    X_train, y_train, X_val, y_val, categories, dataset_stats = build_dataset(
        emb_path, db_dir, exclude_ids,
        extra_labels_path=extra_labels,
        seed=args.seed,
    )
    print(f"Dataset built in {time.time() - start:.1f}s")

    if len(X_train) == 0:
        print("ERROR: No training samples. Check metadata files.")
        sys.exit(1)

    # Train
    print("\n" + "=" * 60)
    print(f"Training {args.model_type} classifier")
    print("=" * 60)
    start = time.time()
    model, training_stats = train_model(
        X_train, y_train, X_val, y_val,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Training completed in {time.time() - start:.1f}s")

    # Save checkpoint
    checkpoint_path = output_dir / "vhold_classifier.pt"
    save_checkpoint(model, checkpoint_path, training_stats, dataset_stats)

    # Save training report
    report = {
        "dataset": dataset_stats,
        "training": training_stats,
    }
    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Training report saved to {report_path}")

    # Print final summary
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

    # Optionally install to default location
    if args.install:
        from vhold.databases.classifier import get_classifier_model_path
        install_path = get_classifier_model_path()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(checkpoint_path, install_path)
        print(f"\nInstalled classifier to {install_path}")


if __name__ == "__main__":
    main()
