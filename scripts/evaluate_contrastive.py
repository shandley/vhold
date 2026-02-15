#!/usr/bin/env python3
"""Evaluate contrastive LoRA adapter vs baseline ProstT5 encoder.

Compares embedding quality before and after contrastive fine-tuning:
- MAP@1/5/10 retrieval accuracy
- NDCG@10
- Intra/inter-class cosine similarity
- Silhouette score
- Triage precision/recall at various thresholds

Usage:
    python scripts/evaluate_contrastive.py \
        --adapter results/contrastive/contrastive_lora/ \
        --output results/contrastive/evaluation/

    # Quick eval on subset:
    python scripts/evaluate_contrastive.py \
        --adapter results/contrastive/contrastive_lora/ \
        --output /tmp/eval --max-proteins 500 --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vhold.features.contrastive import mean_pool_and_normalize
from vhold.utils.constants import PROSTT5_MODEL_NAME, get_db_dir


def extract_embeddings(
    model,
    tokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 1024,
) -> np.ndarray:
    """Extract L2-normalized embeddings from encoder.

    Returns:
        (N, 1024) float32 array of L2-normalized embeddings
    """
    from vhold.features.prostt5 import prepare_prostt5_sequence

    model.eval()
    all_embeddings = []

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start : start + batch_size]
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
            embeddings = mean_pool_and_normalize(outputs.last_hidden_state, attention_mask)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_map_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 1,
) -> float:
    """Compute Mean Average Precision at K."""
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1e9)

    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    n_correct = 0
    total = 0

    for i in range(len(labels)):
        top_k_labels = labels[top_k_indices[i]]
        hits = (top_k_labels == labels[i]).astype(float)
        # Average precision for this query
        if k == 1:
            n_correct += hits[0]
        else:
            precisions = np.cumsum(hits) / np.arange(1, k + 1)
            n_correct += (precisions * hits).sum() / min(k, (labels == labels[i]).sum() - 1 + 1e-8)
        total += 1

    return n_correct / max(total, 1)


def compute_ndcg_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1e9)

    top_k_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    ndcg_sum = 0.0

    for i in range(len(labels)):
        top_k_labels = labels[top_k_indices[i]]
        relevance = (top_k_labels == labels[i]).astype(float)

        # DCG
        discounts = np.log2(np.arange(2, k + 2))
        dcg = (relevance / discounts).sum()

        # Ideal DCG
        n_relevant = min(k, (labels == labels[i]).sum() - 1)
        ideal_relevance = np.zeros(k)
        ideal_relevance[: int(n_relevant)] = 1.0
        idcg = (ideal_relevance / discounts).sum()

        ndcg_sum += dcg / max(idcg, 1e-8)

    return ndcg_sum / len(labels)


def compute_silhouette_sample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 2000,
) -> float:
    """Compute silhouette score on a sample of embeddings."""
    if len(embeddings) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    try:
        from sklearn.metrics import silhouette_score

        return silhouette_score(embeddings, labels, metric="cosine")
    except ImportError:
        # Manual simplified silhouette
        n = len(embeddings)
        sim_matrix = embeddings @ embeddings.T
        scores = []

        for i in range(n):
            same_mask = labels == labels[i]
            same_mask[i] = False
            diff_mask = ~same_mask
            diff_mask[i] = False

            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue

            a = 1.0 - sim_matrix[i][same_mask].mean()  # avg distance to same class
            b = 1.0 - sim_matrix[i][diff_mask].mean()  # avg distance to other classes
            scores.append((b - a) / max(a, b))

        return np.mean(scores) if scores else 0.0


def compute_class_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    category_names: list[str],
) -> dict:
    """Compute per-class intra/inter cosine similarity."""
    unique_labels = np.unique(labels)
    results = {}

    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() < 2:
            continue

        class_emb = embeddings[mask]
        other_emb = embeddings[~mask]

        # Intra-class
        intra_sim = class_emb @ class_emb.T
        np.fill_diagonal(intra_sim, 0)
        n_pairs = mask.sum() * (mask.sum() - 1)
        avg_intra = intra_sim.sum() / max(n_pairs, 1)

        # Inter-class (sample for efficiency)
        if len(other_emb) > 500:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(other_emb), 500, replace=False)
            other_sample = other_emb[sample_idx]
        else:
            other_sample = other_emb

        inter_sim = class_emb @ other_sample.T
        avg_inter = inter_sim.mean()

        name = category_names[lab] if lab < len(category_names) else f"class_{lab}"
        results[name] = {
            "intra_cos": round(float(avg_intra), 4),
            "inter_cos": round(float(avg_inter), 4),
            "n_samples": int(mask.sum()),
        }

    return results


def simulate_triage(
    embeddings: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Simulate triage precision/recall at various thresholds."""
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]

    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1e9)

    results = []
    for threshold in thresholds:
        top1_sim = sim_matrix.max(axis=1)
        top1_idx = sim_matrix.argmax(axis=1)

        matched = top1_sim >= threshold
        if matched.sum() == 0:
            results.append({
                "threshold": threshold,
                "matched": 0,
                "precision": 0.0,
                "recall": 0.0,
            })
            continue

        correct = labels[top1_idx[matched]] == labels[matched]
        precision = correct.mean()
        recall = matched.mean()

        results.append({
            "threshold": threshold,
            "matched": int(matched.sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive LoRA adapter vs baseline"
    )
    parser.add_argument(
        "--adapter", "-a", type=str, required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--db-dir", type=str, default=None,
        help="Database directory (default: ~/.vhold/databases)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-proteins", type=int, default=None,
                        help="Limit evaluation set size")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.db_dir) if args.db_dir else get_db_dir()

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

    # ========================================================================
    # Load evaluation data
    # ========================================================================
    print("\n" + "=" * 60)
    print("Loading evaluation data")
    print("=" * 60)

    # Load embedding DB for protein IDs
    from vhold.databases.embeddings import get_embedding_db_path

    emb_path = get_embedding_db_path(db_dir)
    data = np.load(emb_path, allow_pickle=True)
    protein_ids = list(data["protein_ids"])
    source_dbs = list(data["source_dbs"])
    source_db_map = dict(zip(protein_ids, source_dbs))

    # Generate labels
    from vhold.results.categories import (
        ALL_CATEGORIES,
        AnnotationEvidence,
        get_classification_source,
    )
    from vhold.databases.bfvd import load_bfvd_enriched_metadata
    from vhold.databases.viro3d import (
        get_annotation_expansion,
        get_viro3d_annotation,
        load_viro3d_metadata,
    )

    enriched = load_bfvd_enriched_metadata(db_dir) or {}
    try:
        viro3d_meta = load_viro3d_metadata(db_dir)
    except FileNotFoundError:
        viro3d_meta = None
    try:
        expansion = get_annotation_expansion(db_dir)
        expansion.load_all()
    except Exception:
        expansion = None

    category_to_idx = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

    # Extract sequences from Foldseek DBs
    print("Extracting sequences...")
    import subprocess
    import tempfile

    sequences = {}
    for db_name in ["bfvd", "viro3d"]:
        db_path = db_dir / db_name / db_name
        if not (db_path.exists() or Path(str(db_path) + ".dbtype").exists()):
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
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # Build eval set: proteins with known category + sequence
    eval_ids = []
    eval_seqs = []
    eval_labels = []

    for pid in protein_ids:
        if pid not in sequences:
            continue

        source = source_db_map.get(pid, "")
        category = None

        if source == "bfvd" and pid in enriched:
            row = enriched[pid]
            evidence = AnnotationEvidence(
                pfam=row.get("pfam_domains") or None,
                go_bp=row.get("go_bp") or None,
                go_mf=row.get("go_mf") or None,
            )
            cat, _ = get_classification_source(
                row.get("protein_name", ""), row.get("gene_name", ""), evidence
            )
            if cat != "unknown":
                category = cat

        elif source == "viro3d" and viro3d_meta is not None:
            ann = get_viro3d_annotation(pid, viro3d_meta, None, expansion)
            if ann:
                evidence = AnnotationEvidence(
                    pfam=ann.get("pfam"),
                    go_bp=ann.get("go_bp"),
                    go_mf=ann.get("go_mf"),
                    superfamily=ann.get("superfamily"),
                )
                cat, _ = get_classification_source(
                    ann.get("description", ""), ann.get("gene", ""), evidence
                )
                if cat != "unknown":
                    category = cat

        if category:
            eval_ids.append(pid)
            eval_seqs.append(sequences[pid])
            eval_labels.append(category_to_idx[category])

    # Subsample if needed
    if args.max_proteins and len(eval_ids) > args.max_proteins:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(eval_ids), args.max_proteins, replace=False)
        eval_ids = [eval_ids[i] for i in indices]
        eval_seqs = [eval_seqs[i] for i in indices]
        eval_labels = [eval_labels[i] for i in indices]

    eval_labels = np.array(eval_labels, dtype=np.int64)
    print(f"Evaluation set: {len(eval_ids)} proteins")

    # ========================================================================
    # Extract baseline embeddings
    # ========================================================================
    print("\n" + "=" * 60)
    print("Extracting baseline embeddings")
    print("=" * 60)

    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained(PROSTT5_MODEL_NAME, do_lower_case=False)
    base_model = T5ForConditionalGeneration.from_pretrained(PROSTT5_MODEL_NAME)
    base_model = base_model.to(device)
    base_model.eval()

    start = time.time()
    baseline_emb = extract_embeddings(
        base_model, tokenizer, eval_seqs, device, args.batch_size,
    )
    baseline_time = time.time() - start
    print(f"Baseline: {len(baseline_emb)} embeddings in {baseline_time:.1f}s")

    # ========================================================================
    # Extract LoRA embeddings
    # ========================================================================
    print("\n" + "=" * 60)
    print("Extracting LoRA embeddings")
    print("=" * 60)

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        sys.exit(1)

    try:
        from peft import PeftModel

        lora_model = PeftModel.from_pretrained(base_model, adapter_path)
        lora_model = lora_model.to(device)
        lora_model.eval()
    except ImportError:
        print("ERROR: peft not installed. Run: pip install 'vhold[contrastive]'")
        sys.exit(1)

    start = time.time()
    lora_emb = extract_embeddings(
        lora_model, tokenizer, eval_seqs, device, args.batch_size,
    )
    lora_time = time.time() - start
    print(f"LoRA: {len(lora_emb)} embeddings in {lora_time:.1f}s")

    # ========================================================================
    # Compare metrics
    # ========================================================================
    print("\n" + "=" * 60)
    print("Computing metrics")
    print("=" * 60)

    results = {"baseline": {}, "lora": {}, "comparison": {}}

    for name, emb in [("baseline", baseline_emb), ("lora", lora_emb)]:
        print(f"\n--- {name.upper()} ---")

        map1 = compute_map_at_k(emb, eval_labels, k=1)
        map5 = compute_map_at_k(emb, eval_labels, k=5)
        map10 = compute_map_at_k(emb, eval_labels, k=10)
        ndcg10 = compute_ndcg_at_k(emb, eval_labels, k=10)
        silhouette = compute_silhouette_sample(emb, eval_labels)
        class_metrics = compute_class_metrics(emb, eval_labels, ALL_CATEGORIES)
        triage = simulate_triage(emb, eval_labels)

        print(f"MAP@1:  {map1:.4f}")
        print(f"MAP@5:  {map5:.4f}")
        print(f"MAP@10: {map10:.4f}")
        print(f"NDCG@10: {ndcg10:.4f}")
        print(f"Silhouette: {silhouette:.4f}")

        results[name] = {
            "map_at_1": round(map1, 4),
            "map_at_5": round(map5, 4),
            "map_at_10": round(map10, 4),
            "ndcg_at_10": round(ndcg10, 4),
            "silhouette": round(silhouette, 4),
            "per_class": class_metrics,
            "triage_simulation": triage,
        }

    # Comparison
    results["comparison"] = {
        "map1_delta": round(results["lora"]["map_at_1"] - results["baseline"]["map_at_1"], 4),
        "map5_delta": round(results["lora"]["map_at_5"] - results["baseline"]["map_at_5"], 4),
        "ndcg10_delta": round(results["lora"]["ndcg_at_10"] - results["baseline"]["ndcg_at_10"], 4),
        "silhouette_delta": round(results["lora"]["silhouette"] - results["baseline"]["silhouette"], 4),
        "embedding_cosine_similarity": round(
            float((baseline_emb * lora_emb).sum(axis=1).mean()), 4
        ),
    }

    print(f"\n--- COMPARISON ---")
    print(f"MAP@1 delta:  {results['comparison']['map1_delta']:+.4f}")
    print(f"MAP@5 delta:  {results['comparison']['map5_delta']:+.4f}")
    print(f"NDCG@10 delta: {results['comparison']['ndcg10_delta']:+.4f}")
    print(f"Silhouette delta: {results['comparison']['silhouette_delta']:+.4f}")
    print(f"Baseline-LoRA cosine: {results['comparison']['embedding_cosine_similarity']:.4f}")

    # ========================================================================
    # Save results
    # ========================================================================
    results["metadata"] = {
        "n_proteins": len(eval_ids),
        "adapter_path": str(adapter_path),
        "device": str(device),
        "baseline_time_s": round(baseline_time, 1),
        "lora_time_s": round(lora_time, 1),
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # TSV summary
    tsv_path = output_dir / "evaluation_summary.tsv"
    with open(tsv_path, "w") as f:
        f.write("metric\tbaseline\tlora\tdelta\n")
        for metric in ["map_at_1", "map_at_5", "map_at_10", "ndcg_at_10", "silhouette"]:
            base_val = results["baseline"][metric]
            lora_val = results["lora"][metric]
            delta = lora_val - base_val
            f.write(f"{metric}\t{base_val:.4f}\t{lora_val:.4f}\t{delta:+.4f}\n")
    print(f"Summary saved to {tsv_path}")


if __name__ == "__main__":
    main()
