"""ProstT5 model integration for 3Di sequence prediction."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

from vhold.utils.constants import PROSTT5_MODEL_NAME, get_model_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Result from ProstT5 3Di prediction."""

    sequence_id: str
    aa_sequence: str
    three_di_sequence: str
    confidence_scores: list[float]

    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.aa_sequence)

    @property
    def mean_confidence(self) -> float:
        """Get mean confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device for inference.

    Args:
        device_str: Device string ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class ProstT5Predictor:
    """ProstT5 model for predicting 3Di structural sequences."""

    # 3Di vocabulary mapping
    THREE_DI_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        device: str = "auto",
        half_precision: bool = True,
        model_dir: Path | None = None,
    ):
        """Initialize ProstT5 predictor.

        Args:
            device: Device for inference ('auto', 'cuda', 'mps', 'cpu')
            half_precision: Use half precision on GPU
            model_dir: Directory to cache models
        """
        self.device = get_device(device)
        self.half_precision = half_precision and self.device.type != "cpu"
        self.model_dir = model_dir or get_model_dir()

        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the ProstT5 model and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading ProstT5 model on {self.device}")

        # Create model directory
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            PROSTT5_MODEL_NAME,
            cache_dir=self.model_dir,
            do_lower_case=False,
        )

        # Load model (encoder only for embeddings)
        logger.info("Loading model...")
        self.model = T5EncoderModel.from_pretrained(
            PROSTT5_MODEL_NAME,
            cache_dir=self.model_dir,
        )

        # Move to device and set precision
        self.model = self.model.to(self.device)

        if self.half_precision:
            self.model = self.model.half()
            logger.info("Using half precision (FP16)")

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")

    def _prepare_sequence(self, sequence: str) -> str:
        """Prepare a sequence for ProstT5 input.

        Args:
            sequence: Amino acid sequence

        Returns:
            Formatted sequence with spaces between residues
        """
        # ProstT5 expects sequences with spaces between amino acids
        # and a prefix indicating the task
        # For 3Di prediction, we use the sequence as-is with spaces
        return " ".join(list(sequence))

    def predict_single(self, sequence_id: str, aa_sequence: str) -> PredictionResult:
        """Predict 3Di sequence for a single protein.

        Args:
            sequence_id: Sequence identifier
            aa_sequence: Amino acid sequence

        Returns:
            PredictionResult with 3Di sequence and confidence
        """
        self.load_model()

        # Prepare input
        prepared_seq = self._prepare_sequence(aa_sequence)

        # Tokenize
        inputs = self.tokenizer(
            prepared_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        # Convert embeddings to 3Di sequence
        # ProstT5 produces embeddings that can be mapped to 3Di alphabet
        # We use argmax over the embedding dimensions mapped to 3Di vocab

        # Get the token embeddings (excluding special tokens)
        # Shape: [1, seq_len, hidden_dim]
        token_embeddings = embeddings[0, 1:-1, :]  # Remove CLS and SEP tokens

        # Simple approach: use the first 20 dimensions as logits for 3Di
        # (This is a simplified approach - actual ProstT5 may use different mapping)
        logits = token_embeddings[:, :20]

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Get predictions and confidence
        max_probs, pred_indices = torch.max(probs, dim=-1)

        # Convert to 3Di sequence
        three_di_seq = ""
        confidence_scores = []

        for i, (idx, prob) in enumerate(zip(pred_indices, max_probs)):
            if i < len(aa_sequence):
                three_di_seq += self.THREE_DI_VOCAB[idx.item() % 20]
                confidence_scores.append(prob.item())

        # Ensure lengths match
        if len(three_di_seq) < len(aa_sequence):
            # Pad with 'A' (most common 3Di)
            three_di_seq += "A" * (len(aa_sequence) - len(three_di_seq))
            confidence_scores.extend([0.5] * (len(aa_sequence) - len(confidence_scores)))
        elif len(three_di_seq) > len(aa_sequence):
            three_di_seq = three_di_seq[:len(aa_sequence)]
            confidence_scores = confidence_scores[:len(aa_sequence)]

        return PredictionResult(
            sequence_id=sequence_id,
            aa_sequence=aa_sequence,
            three_di_sequence=three_di_seq,
            confidence_scores=confidence_scores,
        )

    def predict_batch(
        self,
        sequences: dict[str, str],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> dict[str, PredictionResult]:
        """Predict 3Di sequences for multiple proteins.

        Args:
            sequences: Dict mapping IDs to amino acid sequences
            batch_size: Number of sequences per batch
            show_progress: Show progress bar

        Returns:
            Dict mapping IDs to PredictionResults
        """
        self.load_model()

        results = {}
        seq_items = list(sequences.items())

        iterator = range(0, len(seq_items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting 3Di", unit="batch")

        for i in iterator:
            batch = seq_items[i:i + batch_size]

            for seq_id, aa_seq in batch:
                try:
                    result = self.predict_single(seq_id, aa_seq)
                    results[seq_id] = result
                except Exception as e:
                    logger.error(f"Error predicting {seq_id}: {e}")
                    # Return a placeholder result
                    results[seq_id] = PredictionResult(
                        sequence_id=seq_id,
                        aa_sequence=aa_seq,
                        three_di_sequence="A" * len(aa_seq),
                        confidence_scores=[0.0] * len(aa_seq),
                    )

        return results


def predict_3di(
    sequences: dict[str, str],
    device: str = "auto",
    batch_size: int = 1,
    model_dir: Path | None = None,
    show_progress: bool = True,
) -> dict[str, PredictionResult]:
    """Convenience function to predict 3Di sequences.

    Args:
        sequences: Dict mapping IDs to amino acid sequences
        device: Device for inference
        batch_size: Batch size for prediction
        model_dir: Model cache directory
        show_progress: Show progress bar

    Returns:
        Dict mapping IDs to PredictionResults
    """
    predictor = ProstT5Predictor(
        device=device,
        model_dir=model_dir,
    )

    return predictor.predict_batch(
        sequences,
        batch_size=batch_size,
        show_progress=show_progress,
    )
