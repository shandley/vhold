"""ONNX Runtime predictor for ProstT5 3Di sequence prediction.

Drop-in replacement for ProstT5Predictor using quantized ONNX models
for faster CPU inference. Requires pre-exported ONNX models from
`vhold export-onnx`.
"""

from pathlib import Path

from tqdm import tqdm

from vhold.features.prostt5 import (
    GEN_KWARGS,
    FAST_GEN_KWARGS,
    PredictionResult,
    calculate_confidence_scores,
    prepare_prostt5_sequence,
)
from vhold.utils.constants import get_onnx_model_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class OnnxProstT5Predictor:
    """ProstT5 predictor using ONNX Runtime INT8 backend.

    Drop-in replacement for ProstT5Predictor with identical
    predict_single() and predict_batch() interfaces.
    CPU-only. Requires pre-exported ONNX models from `vhold export-onnx`.
    """

    GEN_KWARGS = GEN_KWARGS
    FAST_GEN_KWARGS = FAST_GEN_KWARGS

    def __init__(
        self,
        model_dir: Path | None = None,
        fast: bool = False,
    ):
        """Initialize ONNX predictor.

        Args:
            model_dir: Directory containing quantized ONNX models
            fast: Use greedy decoding instead of beam search
        """
        self.model_dir = model_dir or get_onnx_model_dir()
        self.fast = fast
        self._gen_kwargs = self.FAST_GEN_KWARGS if fast else self.GEN_KWARGS

        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        if self._loaded:
            return

        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import T5Tokenizer

        logger.info(f"Loading ONNX INT8 ProstT5 from {self.model_dir}")

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_dir,
            do_lower_case=False,
        )

        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            use_cache=True,
        )

        self._loaded = True
        mode = "fast (greedy decoding)" if self.fast else "standard (beam search)"
        logger.info(f"ONNX model loaded successfully - {mode}")

    def predict_single(self, sequence_id: str, aa_sequence: str) -> PredictionResult:
        """Predict 3Di sequence for a single protein.

        Args:
            sequence_id: Sequence identifier
            aa_sequence: Amino acid sequence

        Returns:
            PredictionResult with 3Di sequence and confidence scores
        """
        self.load_model()

        prepared_seq = prepare_prostt5_sequence(aa_sequence)
        seq_len = len(aa_sequence)

        inputs = self.tokenizer(
            prepared_seq,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=seq_len + 1,
            min_length=seq_len,
            return_dict_in_generate=True,
            output_scores=True,
            **self._gen_kwargs,
        )

        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        three_di_seq = "".join(decoded.split())

        if len(three_di_seq) > seq_len:
            three_di_seq = three_di_seq[:seq_len]
        elif len(three_di_seq) < seq_len:
            three_di_seq += "A" * (seq_len - len(three_di_seq))

        confidence_scores = calculate_confidence_scores(outputs.scores, seq_len)

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
            batch_size: Ignored (ONNX processes one at a time)
            show_progress: Show progress bar

        Returns:
            Dict mapping IDs to PredictionResults
        """
        self.load_model()

        results = {}
        seq_items = list(sequences.items())

        iterator = range(len(seq_items))
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting 3Di (ONNX)", unit="seq")

        for i in iterator:
            seq_id, aa_seq = seq_items[i]
            try:
                result = self.predict_single(seq_id, aa_seq)
                results[seq_id] = result
            except Exception as e:
                logger.error(f"Error predicting {seq_id}: {e}")
                results[seq_id] = PredictionResult(
                    sequence_id=seq_id,
                    aa_sequence=aa_seq,
                    three_di_sequence="D" * len(aa_seq),
                    confidence_scores=[0.0] * len(aa_seq),
                )

        return results
