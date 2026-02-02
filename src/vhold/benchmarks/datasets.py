"""Benchmark dataset management for vHold evaluation.

This module provides tools for loading and managing benchmark datasets
including gold standard proteins and holdout test sets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GoldStandardEntry:
    """A single entry in the gold standard dataset."""

    protein_id: str
    sequence: str
    true_function: str
    true_category: str
    evidence_code: str  # ECO code for evidence type

    # Metadata
    organism: str = ""
    viral_family: str = ""
    uniprot_id: str = ""
    pdb_id: str = ""

    # Sequence identity to database entries (for stratification)
    max_identity_bfvd: float = 0.0
    max_identity_viro3d: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "true_function": self.true_function,
            "true_category": self.true_category,
            "evidence_code": self.evidence_code,
            "organism": self.organism,
            "viral_family": self.viral_family,
            "uniprot_id": self.uniprot_id,
            "pdb_id": self.pdb_id,
            "max_identity_bfvd": self.max_identity_bfvd,
            "max_identity_viro3d": self.max_identity_viro3d,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldStandardEntry":
        """Create from dictionary."""
        return cls(
            protein_id=data["protein_id"],
            sequence=data["sequence"],
            true_function=data["true_function"],
            true_category=data["true_category"],
            evidence_code=data.get("evidence_code", ""),
            organism=data.get("organism", ""),
            viral_family=data.get("viral_family", ""),
            uniprot_id=data.get("uniprot_id", ""),
            pdb_id=data.get("pdb_id", ""),
            max_identity_bfvd=data.get("max_identity_bfvd", 0.0),
            max_identity_viro3d=data.get("max_identity_viro3d", 0.0),
        )


@dataclass
class BenchmarkDataset:
    """A benchmark dataset for evaluation."""

    name: str
    description: str
    entries: list[GoldStandardEntry] = field(default_factory=list)

    # Dataset statistics
    @property
    def size(self) -> int:
        """Number of entries."""
        return len(self.entries)

    @property
    def categories(self) -> dict[str, int]:
        """Distribution of functional categories."""
        dist = {}
        for entry in self.entries:
            cat = entry.true_category
            dist[cat] = dist.get(cat, 0) + 1
        return dist

    @property
    def viral_families(self) -> dict[str, int]:
        """Distribution of viral families."""
        dist = {}
        for entry in self.entries:
            fam = entry.viral_family or "unknown"
            dist[fam] = dist.get(fam, 0) + 1
        return dist

    def to_fasta(self, output_path: Path) -> None:
        """Write dataset to FASTA file.

        Args:
            output_path: Path to output FASTA file
        """
        with open(output_path, "w") as f:
            for entry in self.entries:
                f.write(f">{entry.protein_id}\n")
                # Wrap sequence at 80 characters
                seq = entry.sequence
                for i in range(0, len(seq), 80):
                    f.write(f"{seq[i:i+80]}\n")

        logger.info(f"Wrote {self.size} sequences to {output_path}")

    def to_json(self, output_path: Path) -> None:
        """Write dataset to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = {
            "name": self.name,
            "description": self.description,
            "size": self.size,
            "categories": self.categories,
            "viral_families": self.viral_families,
            "entries": [e.to_dict() for e in self.entries],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Wrote dataset to {output_path}")

    @classmethod
    def from_json(cls, input_path: Path) -> "BenchmarkDataset":
        """Load dataset from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            BenchmarkDataset object
        """
        with open(input_path) as f:
            data = json.load(f)

        entries = [GoldStandardEntry.from_dict(e) for e in data.get("entries", [])]

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            entries=entries,
        )

    def filter_by_identity(
        self,
        max_identity: float,
        database: str = "bfvd",
    ) -> "BenchmarkDataset":
        """Filter to entries below a maximum sequence identity.

        Args:
            max_identity: Maximum sequence identity (0-1)
            database: Which database identity to use ("bfvd" or "viro3d")

        Returns:
            New filtered BenchmarkDataset
        """
        if database == "bfvd":
            filtered = [e for e in self.entries if e.max_identity_bfvd <= max_identity]
        elif database == "viro3d":
            filtered = [e for e in self.entries if e.max_identity_viro3d <= max_identity]
        else:
            raise ValueError(f"Unknown database: {database}")

        return BenchmarkDataset(
            name=f"{self.name}_identity<={max_identity}",
            description=f"{self.description} (filtered to identity <= {max_identity})",
            entries=filtered,
        )

    def filter_by_category(self, categories: list[str]) -> "BenchmarkDataset":
        """Filter to entries in specified categories.

        Args:
            categories: List of categories to include

        Returns:
            New filtered BenchmarkDataset
        """
        filtered = [e for e in self.entries if e.true_category in categories]

        return BenchmarkDataset(
            name=f"{self.name}_categories",
            description=f"{self.description} (filtered to {categories})",
            entries=filtered,
        )

    def filter_by_family(self, families: list[str]) -> "BenchmarkDataset":
        """Filter to entries in specified viral families.

        Args:
            families: List of viral families to include

        Returns:
            New filtered BenchmarkDataset
        """
        filtered = [e for e in self.entries if e.viral_family in families]

        return BenchmarkDataset(
            name=f"{self.name}_families",
            description=f"{self.description} (filtered to {families})",
            entries=filtered,
        )


def load_gold_standard(
    input_path: Path,
    format: str = "json",
) -> BenchmarkDataset:
    """Load gold standard dataset.

    Args:
        input_path: Path to dataset file
        format: File format ("json" or "tsv")

    Returns:
        BenchmarkDataset object
    """
    if format == "json":
        return BenchmarkDataset.from_json(input_path)

    elif format == "tsv":
        entries = []
        with open(input_path) as f:
            header = f.readline().strip().split("\t")
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < len(header):
                    continue

                row = dict(zip(header, fields))
                entry = GoldStandardEntry(
                    protein_id=row.get("protein_id", ""),
                    sequence=row.get("sequence", ""),
                    true_function=row.get("true_function", ""),
                    true_category=row.get("true_category", ""),
                    evidence_code=row.get("evidence_code", ""),
                    organism=row.get("organism", ""),
                    viral_family=row.get("viral_family", ""),
                    uniprot_id=row.get("uniprot_id", ""),
                )
                entries.append(entry)

        return BenchmarkDataset(
            name=input_path.stem,
            description=f"Loaded from {input_path}",
            entries=entries,
        )

    else:
        raise ValueError(f"Unknown format: {format}")


def create_holdout_set(
    full_dataset: BenchmarkDataset,
    holdout_fraction: float = 0.2,
    stratify_by: str = "category",
    seed: int = 42,
) -> tuple[BenchmarkDataset, BenchmarkDataset]:
    """Split dataset into training and holdout sets.

    Args:
        full_dataset: Full dataset to split
        holdout_fraction: Fraction to hold out
        stratify_by: Stratification variable ("category" or "family")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (training_set, holdout_set)
    """
    import random

    random.seed(seed)

    # Group by stratification variable
    groups: dict[str, list[GoldStandardEntry]] = {}
    for entry in full_dataset.entries:
        if stratify_by == "category":
            key = entry.true_category
        elif stratify_by == "family":
            key = entry.viral_family
        else:
            key = "all"

        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # Split each group
    train_entries = []
    holdout_entries = []

    for key, entries in groups.items():
        random.shuffle(entries)
        n_holdout = max(1, int(len(entries) * holdout_fraction))

        holdout_entries.extend(entries[:n_holdout])
        train_entries.extend(entries[n_holdout:])

    train_set = BenchmarkDataset(
        name=f"{full_dataset.name}_train",
        description=f"Training set ({1-holdout_fraction:.0%})",
        entries=train_entries,
    )

    holdout_set = BenchmarkDataset(
        name=f"{full_dataset.name}_holdout",
        description=f"Holdout set ({holdout_fraction:.0%})",
        entries=holdout_entries,
    )

    logger.info(
        f"Split {full_dataset.size} entries into "
        f"{train_set.size} training + {holdout_set.size} holdout"
    )

    return train_set, holdout_set


def stratify_by_identity(
    dataset: BenchmarkDataset,
    identity_bins: list[float] | None = None,
    database: str = "bfvd",
) -> dict[str, BenchmarkDataset]:
    """Stratify dataset by sequence identity to database.

    Args:
        dataset: Dataset to stratify
        identity_bins: Bin edges (default: [0.0, 0.2, 0.3, 0.4, 0.5, 1.0])
        database: Database to use for identity ("bfvd" or "viro3d")

    Returns:
        Dict mapping bin label to BenchmarkDataset
    """
    if identity_bins is None:
        identity_bins = [0.0, 0.2, 0.3, 0.4, 0.5, 1.0]

    stratified = {}

    for i in range(len(identity_bins) - 1):
        low, high = identity_bins[i], identity_bins[i + 1]
        label = f"{int(low*100)}-{int(high*100)}%"

        if database == "bfvd":
            filtered = [
                e for e in dataset.entries
                if low <= e.max_identity_bfvd < high
            ]
        else:
            filtered = [
                e for e in dataset.entries
                if low <= e.max_identity_viro3d < high
            ]

        stratified[label] = BenchmarkDataset(
            name=f"{dataset.name}_{label}",
            description=f"Identity bin {label}",
            entries=filtered,
        )

    logger.info(
        f"Stratified {dataset.size} entries into {len(stratified)} identity bins"
    )

    return stratified
