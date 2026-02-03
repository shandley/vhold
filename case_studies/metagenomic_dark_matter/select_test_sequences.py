#!/usr/bin/env python3
"""
Select diverse palmdb sequences for Case Study 3.

Strategy:
1. Sample sequences across the ID range (to get diversity)
2. Filter for reasonable length (80-150 aa palmprints)
3. Select 30 sequences for testing
"""

import random
from pathlib import Path

def parse_fasta(fasta_path: Path) -> dict:
    """Parse FASTA file into dict of id -> sequence."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences


def select_diverse_sequences(sequences: dict, n: int = 30, seed: int = 42) -> dict:
    """Select diverse sequences by sampling across ID range."""
    random.seed(seed)

    # Filter by length (palmprints are ~100 aa)
    filtered = {k: v for k, v in sequences.items() if 80 <= len(v) <= 150}
    print(f"Filtered to {len(filtered)} sequences with length 80-150 aa")

    # Sort by ID to ensure reproducibility
    sorted_ids = sorted(filtered.keys())

    # Sample evenly across the range
    step = len(sorted_ids) // n
    selected_ids = [sorted_ids[i * step] for i in range(n)]

    # Add some random samples for diversity
    remaining = [x for x in sorted_ids if x not in selected_ids]
    random_samples = random.sample(remaining, min(10, len(remaining)))
    selected_ids.extend(random_samples[:n - len(selected_ids)] if len(selected_ids) < n else [])

    return {k: filtered[k] for k in selected_ids[:n]}


def write_fasta(sequences: dict, output_path: Path):
    """Write sequences to FASTA file."""
    with open(output_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")


def main():
    input_path = Path("sotus.palmprint.faa")
    output_path = Path("test_palmprints.fasta")

    print(f"Reading {input_path}...")
    sequences = parse_fasta(input_path)
    print(f"Total sequences: {len(sequences)}")

    print(f"\nSelecting diverse test sequences...")
    selected = select_diverse_sequences(sequences, n=30)

    print(f"\nSelected {len(selected)} sequences:")
    for seq_id, seq in list(selected.items())[:5]:
        print(f"  {seq_id}: {len(seq)} aa")
    print(f"  ...")

    write_fasta(selected, output_path)
    print(f"\nWrote test sequences to {output_path}")

    # Summary stats
    lengths = [len(s) for s in selected.values()]
    print(f"\nLength statistics:")
    print(f"  Min: {min(lengths)} aa")
    print(f"  Max: {max(lengths)} aa")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} aa")


if __name__ == "__main__":
    main()
