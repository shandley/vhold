"""Foldseek results parsing for vhold."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vhold.utils.constants import FOLDSEEK_OUTPUT_COLUMNS
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldseekHit:
    """A single Foldseek hit."""

    query: str
    target: str
    fident: float
    alnlen: int
    mismatch: int
    gapopen: int
    qstart: int
    qend: int
    tstart: int
    tend: int
    evalue: float
    bits: float
    qlen: int
    tlen: int
    qaln: str
    taln: str
    prob: float
    source_db: str = ""

    @property
    def coverage(self) -> float:
        """Calculate query coverage."""
        if self.qlen == 0:
            return 0.0
        return self.alnlen / self.qlen

    @property
    def target_coverage(self) -> float:
        """Calculate target coverage."""
        if self.tlen == 0:
            return 0.0
        return self.alnlen / self.tlen

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "target": self.target,
            "fident": self.fident,
            "alnlen": self.alnlen,
            "mismatch": self.mismatch,
            "gapopen": self.gapopen,
            "qstart": self.qstart,
            "qend": self.qend,
            "tstart": self.tstart,
            "tend": self.tend,
            "evalue": self.evalue,
            "bits": self.bits,
            "qlen": self.qlen,
            "tlen": self.tlen,
            "qaln": self.qaln,
            "taln": self.taln,
            "prob": self.prob,
            "source_db": self.source_db,
            "coverage": self.coverage,
            "target_coverage": self.target_coverage,
        }


def parse_foldseek_results(
    results_path: Path | str,
    source_db: str = "",
) -> list[FoldseekHit]:
    """Parse Foldseek tabular output into FoldseekHit objects.

    Args:
        results_path: Path to Foldseek output file
        source_db: Source database name to tag hits with

    Returns:
        List of FoldseekHit objects
    """
    results_path = Path(results_path)

    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return []

    if results_path.stat().st_size == 0:
        logger.warning(f"Results file is empty: {results_path}")
        return []

    df = pd.read_csv(
        results_path,
        sep="\t",
        header=None,
        names=FOLDSEEK_OUTPUT_COLUMNS,
    )

    hits = []
    for _, row in df.iterrows():
        hit = FoldseekHit(
            query=str(row["query"]),
            target=str(row["target"]),
            fident=float(row["fident"]),
            alnlen=int(row["alnlen"]),
            mismatch=int(row["mismatch"]),
            gapopen=int(row["gapopen"]),
            qstart=int(row["qstart"]),
            qend=int(row["qend"]),
            tstart=int(row["tstart"]),
            tend=int(row["tend"]),
            evalue=float(row["evalue"]),
            bits=float(row["bits"]),
            qlen=int(row["qlen"]),
            tlen=int(row["tlen"]),
            qaln=str(row["qaln"]) if pd.notna(row["qaln"]) else "",
            taln=str(row["taln"]) if pd.notna(row["taln"]) else "",
            prob=float(row["prob"]) if pd.notna(row["prob"]) else 0.0,
            source_db=source_db,
        )
        hits.append(hit)

    logger.info(f"Parsed {len(hits)} hits from {results_path}")
    return hits


def parse_dataframe_results(
    df: pd.DataFrame,
    source_db: str = "",
) -> list[FoldseekHit]:
    """Parse a DataFrame of Foldseek results into FoldseekHit objects.

    Args:
        df: DataFrame with Foldseek output columns
        source_db: Source database name

    Returns:
        List of FoldseekHit objects
    """
    hits = []
    for _, row in df.iterrows():
        # Handle source_db column if present
        db = row.get("source_db", source_db) if "source_db" in df.columns else source_db

        hit = FoldseekHit(
            query=str(row["query"]),
            target=str(row["target"]),
            fident=float(row["fident"]),
            alnlen=int(row["alnlen"]),
            mismatch=int(row["mismatch"]),
            gapopen=int(row["gapopen"]),
            qstart=int(row["qstart"]),
            qend=int(row["qend"]),
            tstart=int(row["tstart"]),
            tend=int(row["tend"]),
            evalue=float(row["evalue"]),
            bits=float(row["bits"]),
            qlen=int(row["qlen"]),
            tlen=int(row["tlen"]),
            qaln=str(row["qaln"]) if pd.notna(row.get("qaln")) else "",
            taln=str(row["taln"]) if pd.notna(row.get("taln")) else "",
            prob=float(row["prob"]) if pd.notna(row.get("prob")) else 0.0,
            source_db=db,
        )
        hits.append(hit)

    return hits


def get_best_hits(
    hits: list[FoldseekHit],
    by: str = "evalue",
) -> dict[str, FoldseekHit]:
    """Get best hit per query.

    Args:
        hits: List of FoldseekHit objects
        by: Sort criterion ('evalue', 'bits', 'fident')

    Returns:
        Dict mapping query ID to best hit
    """
    if by == "evalue":
        # Lower is better
        reverse = False
    else:
        # Higher is better
        reverse = True

    # Sort hits
    sorted_hits = sorted(
        hits,
        key=lambda h: getattr(h, by),
        reverse=reverse,
    )

    # Keep best per query
    best = {}
    for hit in sorted_hits:
        if hit.query not in best:
            best[hit.query] = hit

    return best
