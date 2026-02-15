#!/usr/bin/env python3
"""Validate ONNX INT8 quantization against PyTorch reference.

Compares 3Di predictions and embedding quality between PyTorch and ONNX
backends on the SARS-CoV-2 case study (18 proteins).

Usage:
    uv run python scripts/validate_onnx_quantization.py
    uv run python scripts/validate_onnx_quantization.py --input case_studies/sars_cov_2/test_proteins.fasta
    uv run python scripts/validate_onnx_quantization.py --fast  # greedy decoding
"""

import argparse
import time
from pathlib import Path

import numpy as np


def sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate character-level identity between two sequences."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    if not seq1:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX quantization against PyTorch"
    )
    parser.add_argument(
        "--input", "-i",
        default="case_studies/sars_cov_2/test_proteins.fasta",
        help="Input FASTA file (default: SARS-CoV-2 case study)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use greedy decoding (faster, less accurate)",
    )
    parser.add_argument(
        "--max-proteins", "-n",
        type=int,
        default=None,
        help="Limit number of proteins to test",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Read sequences
    from vhold.io.fasta import read_fasta
    sequences = read_fasta(input_path)
    seq_dict = {rec.id: rec.sequence for rec in sequences.values()}

    if args.max_proteins:
        seq_dict = dict(list(seq_dict.items())[:args.max_proteins])

    print(f"Validating on {len(seq_dict)} proteins from {input_path}")
    print(f"Mode: {'fast (greedy)' if args.fast else 'standard (beam search)'}")
    print("=" * 70)

    # ========================================
    # Check ONNX availability
    # ========================================
    from vhold.features.onnx_export import check_onnx_available, check_onnx_exported

    if not check_onnx_available():
        print("Error: ONNX Runtime not installed. Run: pip install vhold[onnx]")
        return 1

    if not check_onnx_exported():
        print("Error: ONNX models not exported. Run: vhold export-onnx")
        return 1

    # ========================================
    # PyTorch predictions
    # ========================================
    print("\n--- PyTorch Backend ---")
    from vhold.features.prostt5 import ProstT5Predictor

    torch_predictor = ProstT5Predictor(device="cpu", fast=args.fast)

    t0 = time.time()
    torch_results = torch_predictor.predict_batch(
        seq_dict, batch_size=1, show_progress=True
    )
    torch_time = time.time() - t0
    print(f"PyTorch time: {torch_time:.1f}s ({torch_time / len(seq_dict):.1f}s/protein)")

    # ========================================
    # ONNX predictions
    # ========================================
    print("\n--- ONNX INT8 Backend ---")
    from vhold.features.onnx_predictor import OnnxProstT5Predictor

    onnx_predictor = OnnxProstT5Predictor(fast=args.fast)

    t0 = time.time()
    onnx_results = onnx_predictor.predict_batch(
        seq_dict, batch_size=1, show_progress=True
    )
    onnx_time = time.time() - t0
    print(f"ONNX time: {onnx_time:.1f}s ({onnx_time / len(seq_dict):.1f}s/protein)")

    # ========================================
    # Compare 3Di predictions
    # ========================================
    print("\n--- 3Di Sequence Comparison ---")
    print(f"{'Protein':<30} {'Length':>6} {'Identity':>10} {'PyTorch conf':>13} {'ONNX conf':>10}")
    print("-" * 75)

    identities = []
    torch_confs = []
    onnx_confs = []

    for seq_id in seq_dict:
        torch_r = torch_results[seq_id]
        onnx_r = onnx_results[seq_id]

        ident = sequence_identity(
            torch_r.three_di_sequence,
            onnx_r.three_di_sequence,
        )
        identities.append(ident)
        torch_confs.append(torch_r.mean_confidence)
        onnx_confs.append(onnx_r.mean_confidence)

        print(
            f"{seq_id:<30} {torch_r.length:>6} "
            f"{ident:>9.1%} "
            f"{torch_r.mean_confidence:>12.4f} "
            f"{onnx_r.mean_confidence:>10.4f}"
        )

    mean_identity = np.mean(identities)
    min_identity = np.min(identities)

    print("-" * 75)
    print(f"{'Mean':>30} {'':>6} {mean_identity:>9.1%}")
    print(f"{'Min':>30} {'':>6} {min_identity:>9.1%}")

    # ========================================
    # Confidence correlation
    # ========================================
    if len(torch_confs) > 1:
        corr = np.corrcoef(torch_confs, onnx_confs)[0, 1]
        print(f"\nConfidence correlation (Pearson): {corr:.4f}")

    # ========================================
    # Embedding comparison
    # ========================================
    print("\n--- Embedding Comparison ---")
    try:
        from vhold.features.embeddings import EmbeddingExtractor
        from vhold.features.onnx_embeddings import OnnxEmbeddingExtractor

        # PyTorch embeddings
        torch_ext = EmbeddingExtractor(device="cpu", encoder_only=True)
        t0 = time.time()
        torch_ids, torch_embs = torch_ext.extract_batch(
            seq_dict, show_progress=False
        )
        torch_emb_time = time.time() - t0

        # ONNX embeddings
        onnx_ext = OnnxEmbeddingExtractor()
        t0 = time.time()
        onnx_ids, onnx_embs = onnx_ext.extract_batch(
            seq_dict, show_progress=False
        )
        onnx_emb_time = time.time() - t0

        # Cosine similarity between corresponding embeddings
        cosine_sims = []
        for i in range(len(torch_ids)):
            sim = np.dot(torch_embs[i], onnx_embs[i])
            cosine_sims.append(sim)

        mean_sim = np.mean(cosine_sims)
        min_sim = np.min(cosine_sims)

        print(f"PyTorch embedding time: {torch_emb_time:.1f}s")
        print(f"ONNX embedding time: {onnx_emb_time:.1f}s")
        print(f"Mean cosine similarity: {mean_sim:.4f}")
        print(f"Min cosine similarity: {min_sim:.4f}")
    except Exception as e:
        print(f"Embedding comparison skipped: {e}")

    # ========================================
    # Summary
    # ========================================
    speedup = torch_time / onnx_time if onnx_time > 0 else float("inf")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"3Di identity (mean): {mean_identity:.1%} {'PASS' if mean_identity >= 0.95 else 'FAIL'} (threshold: >= 95%)")
    print(f"3Di identity (min):  {min_identity:.1%} {'PASS' if min_identity >= 0.85 else 'WARN'}")
    print(f"Speedup:             {speedup:.1f}x {'PASS' if speedup >= 1.5 else 'FAIL'} (threshold: >= 1.5x)")
    print(f"PyTorch time:        {torch_time:.1f}s")
    print(f"ONNX time:           {onnx_time:.1f}s")

    all_pass = mean_identity >= 0.95 and speedup >= 1.5
    print(f"\nOverall: {'PASS' if all_pass else 'NEEDS REVIEW'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
