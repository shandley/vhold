"""Alignment validation for embedding triage matches.

Validates embedding triage matches by performing lightweight pairwise
sequence alignment between query and reference proteins. Rejects matches
where embedding similarity is high but sequence identity is very low,
reducing false positives from convergent structural similarity.

Reference sequences are read from the installed Foldseek BFVD/Viro3D
databases. If databases are not installed, validation is skipped.
"""

from pathlib import Path

from Bio.Align import PairwiseAligner, substitution_matrices

from vhold.features.embeddings import EmbeddingMatch
from vhold.utils.constants import DEFAULT_TRIAGE_MIN_IDENTITY
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


class FoldseekSequenceIndex:
    """Read protein sequences from a Foldseek/MMseqs2 database.

    MMseqs2 databases store sequences in a data file with a text index
    mapping numeric IDs to byte offsets and lengths. Headers are stored
    separately in a _h file with corresponding _h.index.
    """

    def __init__(self, db_path: Path):
        """Initialize with path to the Foldseek DB (without extension).

        Args:
            db_path: Path like ~/.vhold/databases/bfvd/bfvd
        """
        self.db_path = Path(db_path)
        self._header_to_entry: dict[str, tuple[int, int]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load the index, building a header→(offset,length) mapping."""
        data_index_path = self.db_path.with_suffix(".index")
        header_file = Path(str(self.db_path) + "_h")
        header_index = Path(str(self.db_path) + "_h.index")

        if not all(p.exists() for p in [self.db_path, data_index_path, header_file, header_index]):
            raise FileNotFoundError(f"Foldseek DB not found at {self.db_path}")

        # Parse the data index: each line is "id\toffset\tlength\n"
        data_entries: dict[int, tuple[int, int]] = {}
        with open(data_index_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    entry_id = int(parts[0])
                    offset = int(parts[1])
                    length = int(parts[2])
                    data_entries[entry_id] = (offset, length)

        # Parse the header index and header file to map protein names to entry IDs
        header_entries: dict[int, tuple[int, int]] = {}
        with open(header_index) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    entry_id = int(parts[0])
                    offset = int(parts[1])
                    length = int(parts[2])
                    header_entries[entry_id] = (offset, length)

        # Read headers and build name→data_entry mapping
        with open(header_file, "rb") as f:
            for entry_id, (offset, length) in header_entries.items():
                f.seek(offset)
                header = f.read(length).decode("utf-8", errors="replace").strip().rstrip("\x00")
                # Header is the protein name (e.g., "A0A023GZ41")
                protein_name = header.split()[0] if header else ""
                if protein_name and entry_id in data_entries:
                    self._header_to_entry[protein_name] = data_entries[entry_id]

        self._loaded = True
        logger.debug(f"Loaded {len(self._header_to_entry)} sequences from {self.db_path}")

    def get_sequence(self, protein_id: str) -> str | None:
        """Look up a protein sequence by ID.

        Args:
            protein_id: Protein identifier (e.g., UniProt accession)

        Returns:
            Amino acid sequence string, or None if not found
        """
        if not self._loaded:
            raise RuntimeError("Index not loaded. Call load() first.")

        entry = self._header_to_entry.get(protein_id)
        if entry is None:
            return None

        offset, length = entry
        with open(self.db_path, "rb") as f:
            f.seek(offset)
            data = f.read(length).decode("utf-8", errors="replace").strip().rstrip("\x00")
            # Sequence data may contain newlines
            return data.replace("\n", "")


def align_pair(seq1: str, seq2: str) -> tuple[float, float]:
    """Compute global sequence identity between two proteins.

    Uses BioPython's PairwiseAligner with BLOSUM62 scoring for
    efficient global alignment.

    Args:
        seq1: First amino acid sequence
        seq2: Second amino acid sequence

    Returns:
        Tuple of (identity, alignment_score) where identity is 0-1
    """
    if not seq1 or not seq2:
        return 0.0, 0.0

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return 0.0, 0.0

    best = alignments[0]
    score = best.score

    # Count matches from the alignment
    matches = 0
    aligned_length = 0
    for (q_start, q_end), (t_start, t_end) in zip(
        best.aligned[0], best.aligned[1]
    ):
        block_len = q_end - q_start
        for i in range(block_len):
            aligned_length += 1
            if seq1[q_start + i] == seq2[t_start + i]:
                matches += 1

    # Identity as fraction of the longer sequence
    total = max(len(seq1), len(seq2))
    identity = matches / total if total > 0 else 0.0

    return identity, score


def validate_triage_matches(
    matches: dict[str, list[EmbeddingMatch]],
    query_sequences: dict[str, str],
    db_dir: Path | None,
    min_identity: float = DEFAULT_TRIAGE_MIN_IDENTITY,
) -> tuple[dict[str, list[EmbeddingMatch]], list[str]]:
    """Validate embedding triage matches with sequence alignment.

    For each match, aligns the query against the reference protein
    and rejects matches where global sequence identity is below
    the threshold. Rejected queries are moved to unmatched for
    full pipeline processing.

    Args:
        matches: Triage matches (query_id → list of EmbeddingMatch)
        query_sequences: Query protein sequences (id → AA sequence)
        db_dir: Database directory (for Foldseek DB access)
        min_identity: Minimum global sequence identity (0-1)

    Returns:
        Tuple of (validated_matches, rejected_query_ids)
    """
    if not matches or db_dir is None:
        return matches, []

    from vhold.databases.install import get_bfvd_db_path, get_viro3d_db_path

    # Load Foldseek sequence indices
    indices: dict[str, FoldseekSequenceIndex] = {}

    bfvd_path = get_bfvd_db_path(db_dir)
    if bfvd_path.exists():
        try:
            idx = FoldseekSequenceIndex(bfvd_path)
            idx.load()
            indices["bfvd"] = idx
        except Exception as e:
            logger.debug(f"Could not load BFVD sequence index: {e}")

    viro3d_path = get_viro3d_db_path(db_dir)
    if viro3d_path.exists():
        try:
            idx = FoldseekSequenceIndex(viro3d_path)
            idx.load()
            indices["viro3d"] = idx
        except Exception as e:
            logger.debug(f"Could not load Viro3D sequence index: {e}")

    if not indices:
        logger.debug("No Foldseek databases available for alignment validation")
        return matches, []

    validated: dict[str, list[EmbeddingMatch]] = {}
    rejected: list[str] = []

    for query_id, match_list in matches.items():
        query_seq = query_sequences.get(query_id)
        if not query_seq or not match_list:
            validated[query_id] = match_list
            continue

        best_match = match_list[0]
        source_db = best_match.source_db
        target_id = best_match.target_id

        # Look up reference sequence
        index = indices.get(source_db)
        if index is None:
            # DB not available for this source, pass through
            validated[query_id] = match_list
            continue

        ref_seq = index.get_sequence(target_id)
        if ref_seq is None:
            # Can't find reference sequence, pass through
            validated[query_id] = match_list
            continue

        # Align and check identity
        identity, _ = align_pair(query_seq, ref_seq)

        if identity >= min_identity:
            validated[query_id] = match_list
        else:
            rejected.append(query_id)
            logger.debug(
                f"Rejected triage match {query_id} → {target_id}: "
                f"identity={identity:.3f} < {min_identity}"
            )

    return validated, rejected
