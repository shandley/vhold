"""Tests for contrastive learning components."""

import numpy as np
import pytest
import torch

from vhold.features.contrastive import (
    CategoryBalancedSampler,
    ContrastiveDataset,
    MultiGranularityLoss,
    SupConHardLoss,
    collate_contrastive,
    mean_pool_and_normalize,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_normalized_embeddings(batch_size, dim=1024, seed=42):
    """Create L2-normalized random embeddings."""
    rng = torch.Generator().manual_seed(seed)
    emb = torch.randn(batch_size, dim, generator=rng)
    return torch.nn.functional.normalize(emb, p=2, dim=1)


def _make_clustered_embeddings(labels, dim=128, separation=5.0, seed=42):
    """Create embeddings clustered by label for testing loss properties."""
    torch.manual_seed(seed)
    unique = labels.unique()
    # Create well-separated cluster centers
    centers = {}
    for i, lab in enumerate(unique):
        center = torch.zeros(dim)
        center[i * 10 : (i + 1) * 10] = separation
        centers[lab.item()] = center

    emb = torch.stack([centers[lab.item()] + torch.randn(dim) * 0.1 for lab in labels])
    return torch.nn.functional.normalize(emb, p=2, dim=1)


# ============================================================================
# TestSupConHardLoss
# ============================================================================


class TestSupConHardLoss:
    """Tests for supervised contrastive loss with hard negative mining."""

    def test_output_is_scalar(self):
        """Loss output is a scalar tensor."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(16, dim=128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb, labels)
        assert loss.dim() == 0

    def test_loss_is_non_negative(self):
        """Loss value is non-negative."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(16, dim=128)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb, labels)
        assert loss.item() >= 0.0

    def test_gradients_flow(self):
        """Gradients flow through the loss for backprop."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(8, dim=64)
        emb.requires_grad_(True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb, labels)
        loss.backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0

    def test_perfect_clusters_low_loss(self):
        """Well-separated clusters produce lower loss than random embeddings."""
        loss_fn = SupConHardLoss()
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

        clustered = _make_clustered_embeddings(labels, dim=64, separation=10.0)
        random_emb = _make_normalized_embeddings(12, dim=64)

        loss_clustered = loss_fn(clustered, labels)
        loss_random = loss_fn(random_emb, labels)

        assert loss_clustered.item() < loss_random.item()

    def test_single_sample_returns_zero(self):
        """Single sample batch returns zero loss."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(1, dim=64)
        labels = torch.tensor([0])
        loss = loss_fn(emb, labels)
        assert loss.item() == 0.0

    def test_all_same_label_valid(self):
        """All samples with same label: no negatives, returns zero."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(8, dim=64)
        labels = torch.zeros(8, dtype=torch.long)
        loss = loss_fn(emb, labels)
        # No negatives → should still produce a valid (possibly zero) loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_no_positives_returns_zero(self):
        """All unique labels (no positives) returns zero loss."""
        loss_fn = SupConHardLoss()
        emb = _make_normalized_embeddings(4, dim=64)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(emb, labels)
        assert loss.item() == 0.0

    def test_hard_negative_weight_zero(self):
        """hard_negative_weight=0 disables hard negative weighting."""
        loss_fn = SupConHardLoss(hard_negative_weight=0.0)
        emb = _make_normalized_embeddings(8, dim=64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(emb, labels)
        assert loss.item() >= 0.0

    def test_temperature_affects_loss(self):
        """Different temperatures produce different loss values."""
        emb = _make_normalized_embeddings(8, dim=64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss_low_temp = SupConHardLoss(temperature=0.01)(emb, labels)
        loss_high_temp = SupConHardLoss(temperature=1.0)(emb, labels)

        assert loss_low_temp.item() != loss_high_temp.item()


# ============================================================================
# TestMultiGranularityLoss
# ============================================================================


class TestMultiGranularityLoss:
    """Tests for combined category + Pfam contrastive loss."""

    def test_category_only_when_pfam_none(self):
        """With pfam_labels=None, only category loss is computed."""
        loss_fn = MultiGranularityLoss()
        emb = _make_normalized_embeddings(8, dim=128)
        cat_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        total, details = loss_fn(emb, cat_labels, pfam_labels=None)

        assert "category_loss" in details
        assert details["pfam_loss"] == 0.0
        assert total.item() == details["category_loss"]

    def test_pfam_minus_one_excluded(self):
        """Proteins with pfam_label=-1 are excluded from Pfam component."""
        loss_fn = MultiGranularityLoss()
        emb = _make_normalized_embeddings(8, dim=128)
        cat_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        pfam_labels = torch.tensor([-1, -1, 0, 0, 1, 1, -1, -1])

        total, details = loss_fn(emb, cat_labels, pfam_labels)

        assert details["pfam_loss"] >= 0.0
        assert total.item() >= 0.0

    def test_all_pfam_unknown(self):
        """All pfam=-1 results in pfam_loss=0."""
        loss_fn = MultiGranularityLoss()
        emb = _make_normalized_embeddings(8, dim=128)
        cat_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        pfam_labels = torch.full((8,), -1, dtype=torch.long)

        total, details = loss_fn(emb, cat_labels, pfam_labels)

        assert details["pfam_loss"] == 0.0

    def test_details_keys(self):
        """Output details dict has expected keys."""
        loss_fn = MultiGranularityLoss()
        emb = _make_normalized_embeddings(8, dim=128)
        cat_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        pfam_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        _, details = loss_fn(emb, cat_labels, pfam_labels)

        assert "category_loss" in details
        assert "pfam_loss" in details
        assert "total" in details

    def test_weights_applied(self):
        """Category and Pfam weights are applied to total."""
        loss_fn = MultiGranularityLoss(category_weight=0.7, pfam_weight=0.3)
        emb = _make_normalized_embeddings(12, dim=128)
        cat_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        pfam_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

        total, details = loss_fn(emb, cat_labels, pfam_labels)

        # Total should be weighted combination (allowing small float tolerance)
        expected = 0.7 * details["category_loss"] + 0.3 * details["pfam_loss"]
        assert abs(total.item() - expected) < 1e-4


# ============================================================================
# TestCategoryBalancedSampler
# ============================================================================


class TestCategoryBalancedSampler:
    """Tests for balanced category sampling."""

    def test_all_categories_present(self):
        """Each batch should contain samples from all categories."""
        labels = np.array([0] * 100 + [1] * 50 + [2] * 30 + [3] * 20)
        sampler = CategoryBalancedSampler(labels, batch_size=40, seed=42)

        indices = list(sampler)
        # First batch
        batch = indices[: sampler.batch_size]
        batch_labels = set(labels[batch])
        assert batch_labels == {0, 1, 2, 3}

    def test_correct_length(self):
        """Sampler length matches n_batches * batch_size."""
        labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        sampler = CategoryBalancedSampler(labels, batch_size=30, seed=42)
        assert len(sampler) == sampler.n_batches * sampler.batch_size

    def test_batch_size_adjustment(self):
        """Batch size is rounded to multiple of n_categories."""
        labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
        sampler = CategoryBalancedSampler(labels, batch_size=32, seed=42)
        # 3 categories, 32 // 3 = 10 per category, actual batch = 30
        assert sampler.batch_size == 30
        assert sampler.per_category == 10

    def test_minority_oversampling(self):
        """Minority categories are oversampled to fill batches."""
        labels = np.array([0] * 100 + [1] * 5)
        sampler = CategoryBalancedSampler(labels, batch_size=20, seed=42)

        indices = list(sampler)
        # Category 1 has only 5 samples but should appear repeatedly
        cat1_count = sum(1 for i in indices if labels[i] == 1)
        assert cat1_count > 5  # Oversampled

    def test_reproducible_with_seed(self):
        """Same seed produces same order."""
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        sampler1 = CategoryBalancedSampler(labels, batch_size=9, seed=123)
        sampler2 = CategoryBalancedSampler(labels, batch_size=9, seed=123)
        assert list(sampler1) == list(sampler2)


# ============================================================================
# TestContrastiveDataset
# ============================================================================


class TestContrastiveDataset:
    """Tests for the contrastive training dataset."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer that returns fixed-size tensors."""

        class MockTokenizer:
            def __call__(self, text, **kwargs):
                seq_len = min(len(text), kwargs.get("max_length", 1024))
                return {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                }

        return MockTokenizer()

    def test_length(self, mock_tokenizer):
        """Dataset length matches input."""
        ds = ContrastiveDataset(
            protein_ids=["p1", "p2", "p3"],
            sequences=["ACDE", "FGHI", "KLMN"],
            category_labels=np.array([0, 1, 2]),
            pfam_labels=np.array([-1, 0, 1]),
            tokenizer=mock_tokenizer,
        )
        assert len(ds) == 3

    def test_getitem_keys(self, mock_tokenizer):
        """Each item has expected keys."""
        ds = ContrastiveDataset(
            protein_ids=["p1"],
            sequences=["ACDEFGHIK"],
            category_labels=np.array([0]),
            pfam_labels=np.array([-1]),
            tokenizer=mock_tokenizer,
        )
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "category_label" in item
        assert "pfam_label" in item

    def test_labels_correct_type(self, mock_tokenizer):
        """Labels are int64."""
        ds = ContrastiveDataset(
            protein_ids=["p1", "p2"],
            sequences=["ACDE", "FGHI"],
            category_labels=np.array([3, 5]),
            pfam_labels=np.array([-1, 10]),
            tokenizer=mock_tokenizer,
        )
        item = ds[0]
        assert isinstance(item["category_label"], (int, np.int64))
        assert isinstance(item["pfam_label"], (int, np.int64))


# ============================================================================
# TestCollateContrastive
# ============================================================================


class TestCollateContrastive:
    """Tests for the collate function."""

    def test_padding_to_max_length(self):
        """Sequences are padded to max length in batch."""
        batch = [
            {
                "input_ids": torch.ones(5, dtype=torch.long),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "category_label": 0,
                "pfam_label": -1,
            },
            {
                "input_ids": torch.ones(10, dtype=torch.long),
                "attention_mask": torch.ones(10, dtype=torch.long),
                "category_label": 1,
                "pfam_label": 0,
            },
        ]
        collated = collate_contrastive(batch)

        assert collated["input_ids"].shape == (2, 10)
        assert collated["attention_mask"].shape == (2, 10)
        # First sample padded with zeros
        assert collated["attention_mask"][0, 5:].sum() == 0
        assert collated["attention_mask"][1, :].sum() == 10

    def test_labels_stacked(self):
        """Labels are stacked into tensors."""
        batch = [
            {
                "input_ids": torch.ones(3, dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "category_label": 2,
                "pfam_label": 5,
            },
            {
                "input_ids": torch.ones(3, dtype=torch.long),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "category_label": 7,
                "pfam_label": -1,
            },
        ]
        collated = collate_contrastive(batch)

        assert collated["category_labels"].tolist() == [2, 7]
        assert collated["pfam_labels"].tolist() == [5, -1]


# ============================================================================
# TestMeanPoolAndNormalize
# ============================================================================


class TestMeanPoolAndNormalize:
    """Tests for mean pooling and L2 normalization."""

    def test_output_shape(self):
        """Output shape is (B, D)."""
        hidden = torch.randn(4, 10, 64)
        mask = torch.ones(4, 10)
        out = mean_pool_and_normalize(hidden, mask)
        assert out.shape == (4, 64)

    def test_l2_normalized(self):
        """Output vectors have unit L2 norm."""
        hidden = torch.randn(8, 20, 128)
        mask = torch.ones(8, 20)
        out = mean_pool_and_normalize(hidden, mask)
        norms = torch.norm(out, p=2, dim=1)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_mask_excludes_padding(self):
        """Masked positions are excluded from mean pooling."""
        hidden = torch.zeros(1, 5, 4)
        hidden[0, 0, :] = 1.0  # Only first position has signal
        hidden[0, 3, :] = 100.0  # This should be masked out

        mask = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float)
        out = mean_pool_and_normalize(hidden, mask)

        # Result should be the normalized first position vector
        expected = torch.nn.functional.normalize(hidden[0, 0:1, :], p=2, dim=1)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestLoRAAdapterPath
# ============================================================================


class TestLoRAAdapterPath:
    """Tests for LoRA adapter path management."""

    def test_path_includes_lora_dir(self, tmp_path):
        """Path ends with contrastive_lora directory."""
        from vhold.databases.lora import get_lora_adapter_path

        path = get_lora_adapter_path(model_dir=tmp_path)
        assert path.name == "contrastive_lora"
        assert path.parent == tmp_path

    def test_check_returns_false_when_missing(self, tmp_path):
        """check_lora_adapter returns False when no adapter exists."""
        from vhold.databases.lora import check_lora_adapter

        assert check_lora_adapter(model_dir=tmp_path) is False

    def test_check_returns_true_when_present(self, tmp_path):
        """check_lora_adapter returns True when adapter_config.json exists."""
        from vhold.databases.lora import check_lora_adapter

        adapter_dir = tmp_path / "contrastive_lora"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")

        assert check_lora_adapter(model_dir=tmp_path) is True
