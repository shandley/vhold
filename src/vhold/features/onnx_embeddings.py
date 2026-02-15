"""ONNX Runtime embedding extractor for triage.

Uses the encoder component of the exported ONNX model to extract
per-protein embeddings for embedding-based triage. This avoids
loading the full PyTorch model when using ONNX backend.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from vhold.features.prostt5 import prepare_prostt5_sequence
from vhold.utils.constants import EMBEDDING_DIM, get_onnx_model_dir
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class OnnxEmbeddingExtractor:
    """Extract per-protein embeddings using ONNX Runtime encoder.

    Same interface as EmbeddingExtractor but backed by ONNX Runtime.
    CPU-only. Requires pre-exported ONNX models.
    """

    def __init__(self, model_dir: Path | None = None):
        """Initialize ONNX embedding extractor.

        Args:
            model_dir: Directory containing quantized ONNX models
        """
        self.model_dir = Path(model_dir) if model_dir else get_onnx_model_dir()
        self.session = None
        self.tokenizer = None
        self._loaded = False
        # Expose model/tokenizer attributes for compatibility with triage code
        self.model = None

    def load_model(self) -> None:
        """Load the ONNX encoder session and tokenizer."""
        if self._loaded:
            return

        import onnxruntime as ort
        from transformers import T5Tokenizer

        encoder_path = self.model_dir / "encoder_model.onnx"
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"ONNX encoder not found at {encoder_path}. "
                "Export first with: vhold export-onnx"
            )

        logger.info(f"Loading ONNX encoder from {encoder_path}")

        self.session = ort.InferenceSession(
            str(encoder_path),
            providers=["CPUExecutionProvider"],
        )

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_dir,
            do_lower_case=False,
        )

        self._loaded = True
        logger.info("ONNX encoder loaded successfully")

    def extract_single(self, sequence: str) -> np.ndarray:
        """Extract embedding for a single protein sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            L2-normalized 1024-dim embedding vector
        """
        self.load_model()

        prepared = prepare_prostt5_sequence(sequence)
        inputs = self.tokenizer(
            prepared,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Run encoder
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        outputs = self.session.run(None, ort_inputs)

        # outputs[0] is last_hidden_state: (1, seq_len, 1024)
        hidden_states = outputs[0]
        attention_mask = inputs["attention_mask"]

        # Mean pool over sequence length respecting attention mask
        mask = attention_mask[..., np.newaxis].astype(np.float32)
        embedding = (hidden_states * mask).sum(axis=1) / mask.sum(axis=1)
        embedding = embedding[0]  # Remove batch dim

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def extract_batch(
        self,
        sequences: dict[str, str],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        """Extract embeddings for multiple proteins.

        Args:
            sequences: Dict mapping IDs to amino acid sequences
            batch_size: Ignored (processes one at a time)
            show_progress: Show progress bar

        Returns:
            Tuple of (protein_ids, embeddings_matrix) where embeddings
            is (N, 1024) L2-normalized
        """
        self.load_model()

        ids = []
        embeddings = []

        items = list(sequences.items())
        iterator = range(len(items))
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting embeddings (ONNX)", unit="seq")

        for i in iterator:
            seq_id, seq = items[i]
            try:
                emb = self.extract_single(seq)
                ids.append(seq_id)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error extracting embedding for {seq_id}: {e}")

        if embeddings:
            return ids, np.stack(embeddings)
        return ids, np.empty((0, EMBEDDING_DIM), dtype=np.float32)
