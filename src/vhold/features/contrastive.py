"""Contrastive learning components for ProstT5 encoder fine-tuning.

Implements supervised contrastive loss with hard negative mining (SupCon-Hard)
for improving viral protein embedding quality. Based on CLEAN (Yu et al.,
Science 2023).

Components:
- SupConHardLoss: Supervised contrastive loss with hard negative mining
- MultiGranularityLoss: Combined category + Pfam contrastive losses
- CategoryBalancedSampler: Equal category representation per batch
- ContrastiveDataset: Tokenized sequences with category/Pfam labels
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class SupConHardLoss(nn.Module):
    """Supervised contrastive loss with hard negative mining.

    For each anchor, positives are all samples with the same label and
    negatives are all samples with different labels. The hardest negative
    (closest in embedding space) receives extra weight to focus learning
    on difficult cases.

    Based on CLEAN (Yu et al., Science 2023).

    Args:
        temperature: Scaling temperature for similarity (default: 0.07)
        hard_negative_weight: Extra weight for hardest negative (default: 1.0)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SupCon-Hard loss.

        Args:
            embeddings: L2-normalized embeddings (B, D)
            labels: Integer class labels (B,)

        Returns:
            Scalar loss tensor
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarity (embeddings already L2-normed)
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Mask: positive pairs (same label, excluding self)
        labels_col = labels.unsqueeze(1)
        positive_mask = (labels_col == labels_col.t()).float()
        positive_mask.fill_diagonal_(0.0)

        # Negative mask
        negative_mask = 1.0 - positive_mask
        negative_mask.fill_diagonal_(0.0)

        # Check if any anchor has positives
        pos_counts = positive_mask.sum(dim=1)
        valid_anchors = pos_counts > 0
        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Hard negative weighting: boost the hardest negative per anchor
        if self.hard_negative_weight > 0:
            neg_sims = sim_matrix * negative_mask
            # Set non-negatives to -inf so they don't get selected
            neg_sims = neg_sims + (1.0 - negative_mask) * (-1e9)
            hardest_neg_idx = neg_sims.argmax(dim=1)

            hard_weight_matrix = torch.ones_like(negative_mask)
            hard_weight_matrix.scatter_(
                1, hardest_neg_idx.unsqueeze(1), 1.0 + self.hard_negative_weight
            )
            weighted_neg_mask = negative_mask * hard_weight_matrix
        else:
            weighted_neg_mask = negative_mask

        # For numerical stability, subtract max
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over weighted negatives + positives
        exp_logits = torch.exp(logits)
        neg_sum = (exp_logits * weighted_neg_mask).sum(dim=1, keepdim=True)
        pos_exp = exp_logits * positive_mask

        # Log probability of each positive pair
        log_prob = logits - torch.log(neg_sum + pos_exp.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs per anchor
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (pos_counts + 1e-8)

        # Average over valid anchors
        loss = -(mean_log_prob[valid_anchors]).mean()

        return loss


class MultiGranularityLoss(nn.Module):
    """Combined category-level and Pfam-level contrastive losses.

    Uses coarse functional categories (11 classes) as the primary signal
    and finer-grained Pfam families as a secondary signal. Proteins
    without Pfam annotations (label=-1) are excluded from the Pfam
    component but still contribute to the category component.

    Args:
        category_weight: Weight for category-level loss (default: 0.7)
        pfam_weight: Weight for Pfam-level loss (default: 0.3)
        category_temperature: Temperature for category loss (default: 0.07)
        pfam_temperature: Temperature for Pfam loss (default: 0.05)
        hard_negative_weight: Weight for hardest negative (default: 1.0)
    """

    def __init__(
        self,
        category_weight: float = 0.7,
        pfam_weight: float = 0.3,
        category_temperature: float = 0.07,
        pfam_temperature: float = 0.05,
        hard_negative_weight: float = 1.0,
    ):
        super().__init__()
        self.category_weight = category_weight
        self.pfam_weight = pfam_weight
        self.category_loss = SupConHardLoss(
            temperature=category_temperature,
            hard_negative_weight=hard_negative_weight,
        )
        self.pfam_loss = SupConHardLoss(
            temperature=pfam_temperature,
            hard_negative_weight=hard_negative_weight,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        category_labels: torch.Tensor,
        pfam_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined multi-granularity contrastive loss.

        Args:
            embeddings: L2-normalized embeddings (B, D)
            category_labels: Functional category labels (B,)
            pfam_labels: Pfam family labels (B,), -1 for unknown.
                If None, only category loss is computed.

        Returns:
            (total_loss, {"category_loss": ..., "pfam_loss": ..., "total": ...})
        """
        cat_loss = self.category_loss(embeddings, category_labels)
        details = {"category_loss": cat_loss.item()}

        if pfam_labels is not None and self.pfam_weight > 0:
            # Filter to proteins with known Pfam labels
            pfam_mask = pfam_labels >= 0
            if pfam_mask.sum() > 1:
                pfam_emb = embeddings[pfam_mask]
                pfam_lab = pfam_labels[pfam_mask]
                p_loss = self.pfam_loss(pfam_emb, pfam_lab)
                details["pfam_loss"] = p_loss.item()
                total = self.category_weight * cat_loss + self.pfam_weight * p_loss
            else:
                details["pfam_loss"] = 0.0
                total = cat_loss
        else:
            details["pfam_loss"] = 0.0
            total = cat_loss

        details["total"] = total.item()
        return total, details


class CategoryBalancedSampler(Sampler):
    """Sampler that ensures balanced category representation per batch.

    Each batch contains approximately equal numbers of samples from each
    category. Categories with fewer samples are oversampled within each
    epoch to ensure representation.

    Args:
        labels: Array of integer category labels for all samples
        batch_size: Desired batch size (rounded to nearest multiple of
            number of unique categories)
        drop_last: Drop the last incomplete batch
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 256,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.labels = np.asarray(labels)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

        # Group indices by category
        unique_labels = np.unique(self.labels)
        self.n_categories = len(unique_labels)
        self.per_category = max(1, batch_size // self.n_categories)
        self.batch_size = self.per_category * self.n_categories

        self.category_indices = {}
        for label in unique_labels:
            self.category_indices[label] = np.where(self.labels == label)[0]

        # Total batches per epoch based on largest category
        max_category_size = max(len(idx) for idx in self.category_indices.values())
        self.n_batches = max_category_size // self.per_category
        if not self.drop_last and max_category_size % self.per_category > 0:
            self.n_batches += 1

    def __iter__(self):
        # Shuffle indices within each category
        shuffled = {}
        for label, indices in self.category_indices.items():
            perm = self.rng.permutation(len(indices))
            shuffled[label] = indices[perm]

        for batch_idx in range(self.n_batches):
            batch = []
            for label, indices in shuffled.items():
                start = (batch_idx * self.per_category) % len(indices)
                selected = []
                for j in range(self.per_category):
                    idx = (start + j) % len(indices)
                    selected.append(indices[idx])
                batch.extend(selected)

            # Shuffle within batch
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return self.n_batches * self.batch_size


class ContrastiveDataset(Dataset):
    """Dataset for contrastive training from raw sequences + labels.

    Stores protein sequences with their category and Pfam labels.
    Tokenization happens in __getitem__ to support variable-length
    sequences with per-batch padding via collate_contrastive().

    Args:
        protein_ids: List of protein identifiers
        sequences: List of amino acid sequences (same order as protein_ids)
        category_labels: Array of category indices (same order)
        pfam_labels: Array of Pfam family indices, -1 for unknown (same order)
        tokenizer: ProstT5 tokenizer for sequence preparation
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        protein_ids: list[str],
        sequences: list[str],
        category_labels: np.ndarray,
        pfam_labels: np.ndarray,
        tokenizer,
        max_length: int = 1024,
    ):
        assert len(protein_ids) == len(sequences) == len(category_labels) == len(pfam_labels)
        self.protein_ids = protein_ids
        self.sequences = sequences
        self.category_labels = category_labels.astype(np.int64)
        self.pfam_labels = pfam_labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int) -> dict:
        from vhold.features.prostt5 import prepare_prostt5_sequence

        seq = self.sequences[idx]
        prepared = prepare_prostt5_sequence(seq)

        tokens = self.tokenizer(
            prepared,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "category_label": self.category_labels[idx],
            "pfam_label": self.pfam_labels[idx],
        }


def collate_contrastive(batch: list[dict]) -> dict:
    """Collate function that pads sequences to max length in batch.

    Args:
        batch: List of dicts from ContrastiveDataset.__getitem__

    Returns:
        Dict with padded tensors: input_ids, attention_mask,
        category_labels, pfam_labels
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    category_labels = torch.tensor([item["category_label"] for item in batch])
    pfam_labels = torch.tensor([item["pfam_label"] for item in batch])

    # Pad to max length in this batch
    max_len = max(ids.shape[0] for ids in input_ids)

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        length = ids.shape[0]
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "category_labels": category_labels,
        "pfam_labels": pfam_labels,
    }


def mean_pool_and_normalize(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean pool encoder hidden states and L2 normalize.

    Args:
        hidden_states: Encoder output (B, seq_len, D)
        attention_mask: Attention mask (B, seq_len)

    Returns:
        L2-normalized embeddings (B, D)
    """
    mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return F.normalize(pooled, p=2, dim=1)
