"""ProstT5 model integration for 3Di sequence prediction."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
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


def _enable_mps_fallback() -> None:
    """Enable MPS fallback to CPU for unsupported operations.

    Some PyTorch operations (e.g., certain attention variants) are not yet
    implemented on the MPS backend. This env var makes them fall back to CPU
    rather than crashing.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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
            _enable_mps_fallback()
            logger.info("Using MPS (Apple Silicon GPU) with CPU fallback for unsupported ops")
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if device_str == "mps":
        _enable_mps_fallback()
        logger.info("Using MPS (Apple Silicon GPU) with CPU fallback for unsupported ops")

    return torch.device(device_str)


def prepare_prostt5_sequence(sequence: str) -> str:
    """Prepare an amino acid sequence for ProstT5 input.

    Replaces rare/ambiguous amino acids with X, adds spaces between
    residues, and prepends the AA2fold prefix for translation.

    Args:
        sequence: Amino acid sequence

    Returns:
        Formatted sequence with AA2fold prefix and spaces
    """
    sequence = re.sub(r"[UZOB]", "X", sequence.upper())
    spaced = " ".join(list(sequence))
    return f"<AA2fold> {spaced}"


# Generation parameters for AA to 3Di translation
# These are tuned for structure prediction as per ProstT5 documentation
GEN_KWARGS = {
    "do_sample": True,
    "num_beams": 3,
    "top_p": 0.95,
    "temperature": 1.2,
    "top_k": 6,
    "repetition_penalty": 1.2,
    "early_stopping": True,
    "num_return_sequences": 1,
}

# Fast generation: greedy decoding (1 beam, no sampling)
# ~3x faster than beam search, suitable for well-characterized proteins
# where high-identity hits are expected. Not recommended for remote
# homology / twilight zone searches where 3Di quality matters more.
FAST_GEN_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
}


def calculate_confidence_scores(
    scores: tuple,
    expected_len: int,
) -> list[float]:
    """Calculate per-residue confidence scores from generation logits.

    The confidence is calculated as the softmax probability of the
    selected token at each position. Higher probability means the
    model is more confident in that prediction.

    Args:
        scores: Tuple of score tensors from model.generate(), one per position
        expected_len: Expected sequence length

    Returns:
        List of confidence scores (0-1) for each residue
    """
    if not scores:
        return [1.0] * expected_len

    confidence_scores = []

    for score_tensor in scores:
        # score_tensor shape: (batch_size, vocab_size)
        # Apply softmax to get probabilities
        probs = F.softmax(score_tensor[0], dim=-1)
        # Get the maximum probability (confidence in the chosen token)
        max_prob = probs.max().item()
        confidence_scores.append(max_prob)

    # Adjust length if needed
    if len(confidence_scores) > expected_len:
        confidence_scores = confidence_scores[:expected_len]
    elif len(confidence_scores) < expected_len:
        mean_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
        confidence_scores.extend([mean_conf] * (expected_len - len(confidence_scores)))

    return confidence_scores


def calculate_batch_confidence_scores(
    scores: tuple,
    batch_size: int,
    seq_lens: list[int],
) -> list[list[float]]:
    """Calculate per-residue confidence for a batch of sequences.

    Args:
        scores: Tuple of score tensors, one per generated position
        batch_size: Number of sequences in the batch
        seq_lens: List of expected sequence lengths

    Returns:
        List of confidence score lists, one per sequence in batch
    """
    if not scores:
        return [[1.0] * length for length in seq_lens]

    batch_confidences: list[list[float]] = [[] for _ in range(batch_size)]

    for score_tensor in scores:
        probs = F.softmax(score_tensor, dim=-1)
        max_probs = probs.max(dim=-1).values

        for batch_idx in range(batch_size):
            batch_confidences[batch_idx].append(max_probs[batch_idx].item())

    for idx, expected_len in enumerate(seq_lens):
        conf = batch_confidences[idx]
        if len(conf) > expected_len:
            batch_confidences[idx] = conf[:expected_len]
        elif len(conf) < expected_len:
            mean_conf = sum(conf) / len(conf) if conf else 1.0
            batch_confidences[idx].extend(
                [mean_conf] * (expected_len - len(conf))
            )

    return batch_confidences


class ProstT5Predictor:
    """ProstT5 model for predicting 3Di structural sequences.

    Uses the sequence-to-sequence translation capability of ProstT5
    to convert amino acid sequences to 3Di structural alphabet.
    """

    GEN_KWARGS = GEN_KWARGS
    FAST_GEN_KWARGS = FAST_GEN_KWARGS

    def __init__(
        self,
        device: str = "auto",
        half_precision: bool = True,
        model_dir: Path | None = None,
        fast: bool = False,
    ):
        """Initialize ProstT5 predictor.

        Args:
            device: Device for inference ('auto', 'cuda', 'mps', 'cpu')
            half_precision: Use half precision on GPU (not supported on CPU)
            model_dir: Directory to cache models
            fast: Use greedy decoding instead of beam search (~3x faster,
                  suitable for well-characterized proteins)
        """
        self.device = get_device(device)
        # Half precision only works on CUDA, not MPS or CPU
        self.half_precision = half_precision and self.device.type == "cuda"
        self.model_dir = model_dir or get_model_dir()
        self.fast = fast
        self._gen_kwargs = self.FAST_GEN_KWARGS if fast else self.GEN_KWARGS

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

        # Load full seq2seq model for translation
        logger.info("Loading model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            PROSTT5_MODEL_NAME,
            cache_dir=self.model_dir,
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Set precision
        if self.half_precision:
            self.model = self.model.half()
            logger.info("Using half precision (FP16)")
        else:
            self.model = self.model.float()
            if self.device.type == "cpu":
                logger.info("Using full precision (FP32) on CPU")

        self.model.eval()
        self._loaded = True
        mode = "fast (greedy decoding)" if self.fast else "standard (beam search)"
        logger.info(f"Model loaded successfully - {mode}")

    def _prepare_sequence(self, sequence: str) -> str:
        """Prepare an amino acid sequence for ProstT5 input.

        Args:
            sequence: Amino acid sequence

        Returns:
            Formatted sequence with AA2fold prefix and spaces
        """
        return prepare_prostt5_sequence(sequence)

    def _calculate_confidence_scores(
        self,
        scores: tuple[torch.Tensor, ...],
        expected_len: int,
    ) -> list[float]:
        """Calculate per-residue confidence scores from generation logits."""
        return calculate_confidence_scores(scores, expected_len)

    def predict_single(self, sequence_id: str, aa_sequence: str) -> PredictionResult:
        """Predict 3Di sequence for a single protein.

        Args:
            sequence_id: Sequence identifier
            aa_sequence: Amino acid sequence

        Returns:
            PredictionResult with 3Di sequence
        """
        self.load_model()

        # Prepare input with AA2fold prefix
        prepared_seq = self._prepare_sequence(aa_sequence)
        seq_len = len(aa_sequence)

        # Tokenize
        inputs = self.tokenizer(
            prepared_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # ProstT5 max length
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate 3Di sequence with scores for confidence calculation
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=seq_len + 1,  # +1 for potential EOS token
                min_length=seq_len,
                return_dict_in_generate=True,
                output_scores=True,
                **self._gen_kwargs,
            )

        # Decode the generated sequence
        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Remove spaces from the 3Di sequence
        three_di_seq = "".join(decoded.split())

        # Ensure length matches (truncate or pad if needed)
        if len(three_di_seq) > seq_len:
            three_di_seq = three_di_seq[:seq_len]
        elif len(three_di_seq) < seq_len:
            # Pad with 'A' if too short (shouldn't happen normally)
            three_di_seq += "A" * (seq_len - len(three_di_seq))

        # Calculate per-residue confidence from generation scores
        confidence_scores = self._calculate_confidence_scores(
            outputs.scores, seq_len
        )

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
            batch_size: Number of sequences per batch (for batched generation)
            show_progress: Show progress bar

        Returns:
            Dict mapping IDs to PredictionResults
        """
        self.load_model()

        results = {}
        seq_items = list(sequences.items())

        # Process in batches
        iterator = range(0, len(seq_items), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting 3Di", unit="batch")

        for i in iterator:
            batch = seq_items[i:i + batch_size]

            if batch_size == 1:
                # Single sequence processing
                for seq_id, aa_seq in batch:
                    try:
                        result = self.predict_single(seq_id, aa_seq)
                        results[seq_id] = result
                    except Exception as e:
                        logger.error(f"Error predicting {seq_id}: {e}")
                        results[seq_id] = PredictionResult(
                            sequence_id=seq_id,
                            aa_sequence=aa_seq,
                            three_di_sequence="D" * len(aa_seq),  # D is common 3Di
                            confidence_scores=[0.0] * len(aa_seq),
                        )
            else:
                # Batch processing
                try:
                    batch_results = self._predict_batch_internal(batch)
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"Error in batch prediction: {e}")
                    # Fall back to single sequence processing
                    for seq_id, aa_seq in batch:
                        try:
                            result = self.predict_single(seq_id, aa_seq)
                            results[seq_id] = result
                        except Exception as e2:
                            logger.error(f"Error predicting {seq_id}: {e2}")
                            results[seq_id] = PredictionResult(
                                sequence_id=seq_id,
                                aa_sequence=aa_seq,
                                three_di_sequence="D" * len(aa_seq),
                                confidence_scores=[0.0] * len(aa_seq),
                            )

        return results

    def _predict_batch_internal(
        self,
        batch: list[tuple[str, str]],
    ) -> dict[str, PredictionResult]:
        """Internal method for batch prediction.

        Args:
            batch: List of (sequence_id, aa_sequence) tuples

        Returns:
            Dict mapping IDs to PredictionResults
        """
        # Prepare all sequences
        seq_ids = [item[0] for item in batch]
        aa_seqs = [item[1] for item in batch]
        prepared_seqs = [self._prepare_sequence(seq) for seq in aa_seqs]
        seq_lens = [len(seq) for seq in aa_seqs]

        # Tokenize batch
        inputs = self.tokenizer(
            prepared_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with scores for confidence calculation
        max_len = max(seq_lens)
        min_len = min(seq_lens)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_len + 1,
                min_length=min_len,
                return_dict_in_generate=True,
                output_scores=True,
                **self._gen_kwargs,
            )

        # Decode all outputs
        decoded_list = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )

        # Calculate per-batch-item confidence scores
        # For batched generation, scores are shape (num_positions, batch_size, vocab_size)
        batch_confidences = self._calculate_batch_confidence_scores(
            outputs.scores, len(batch), seq_lens
        )

        # Build results
        results = {}
        for idx, (seq_id, aa_seq, decoded, expected_len) in enumerate(zip(
            seq_ids, aa_seqs, decoded_list, seq_lens
        )):
            three_di_seq = "".join(decoded.split())

            # Adjust length
            if len(three_di_seq) > expected_len:
                three_di_seq = three_di_seq[:expected_len]
            elif len(three_di_seq) < expected_len:
                three_di_seq += "A" * (expected_len - len(three_di_seq))

            results[seq_id] = PredictionResult(
                sequence_id=seq_id,
                aa_sequence=aa_seq,
                three_di_sequence=three_di_seq,
                confidence_scores=batch_confidences[idx],
            )

        return results

    def _calculate_batch_confidence_scores(
        self,
        scores: tuple[torch.Tensor, ...],
        batch_size: int,
        seq_lens: list[int],
    ) -> list[list[float]]:
        """Calculate per-residue confidence for a batch of sequences."""
        return calculate_batch_confidence_scores(scores, batch_size, seq_lens)


def predict_3di(
    sequences: dict[str, str],
    device: str = "auto",
    batch_size: int = 1,
    model_dir: Path | None = None,
    show_progress: bool = True,
    fast: bool = False,
) -> dict[str, PredictionResult]:
    """Convenience function to predict 3Di sequences.

    Args:
        sequences: Dict mapping IDs to amino acid sequences
        device: Device for inference
        batch_size: Batch size for prediction
        model_dir: Model cache directory
        show_progress: Show progress bar
        fast: Use greedy decoding (~3x faster, lower quality)

    Returns:
        Dict mapping IDs to PredictionResults
    """
    predictor = ProstT5Predictor(
        device=device,
        model_dir=model_dir,
        fast=fast,
    )

    return predictor.predict_batch(
        sequences,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def get_predictor(
    device: str = "auto",
    backend: str = "torch",
    model_dir: Path | None = None,
    fast: bool = False,
):
    """Factory to get the appropriate predictor backend.

    Args:
        device: Device for PyTorch backend ('auto', 'cuda', 'mps', 'cpu')
        backend: 'torch' or 'onnx'
        model_dir: Model cache directory
        fast: Use greedy decoding

    Returns:
        ProstT5Predictor or OnnxProstT5Predictor
    """
    if backend == "onnx":
        from vhold.features.onnx_export import check_onnx_available, check_onnx_exported
        from vhold.utils.constants import ONNX_MODEL_DIR_NAME

        if not check_onnx_available():
            raise ImportError(
                "ONNX backend requires optimum and onnxruntime. "
                "Install with: pip install vhold[onnx]"
            )

        if model_dir:
            onnx_dir = Path(model_dir) / ONNX_MODEL_DIR_NAME
        else:
            from vhold.utils.constants import get_onnx_model_dir
            onnx_dir = get_onnx_model_dir()

        if not check_onnx_exported(onnx_dir):
            raise FileNotFoundError(
                f"ONNX models not found at {onnx_dir}. "
                "Export first with: vhold export-onnx"
            )

        from vhold.features.onnx_predictor import OnnxProstT5Predictor
        return OnnxProstT5Predictor(model_dir=onnx_dir, fast=fast)

    return ProstT5Predictor(device=device, model_dir=model_dir, fast=fast)
