"""ESM Metagenomic Atlas integration for dark matter analysis.

This module provides integration with the ESM Metagenomic Atlas, a database
of 617+ million protein structures predicted from metagenomic sequences.

The ESM Atlas enables:
1. Finding structural analogs for dark matter proteins in metagenomic data
2. Discovering novel protein folds not present in cultured organisms
3. Identifying conserved domains across the metagenome

Key resources:
- ESM Metagenomic Atlas: https://esmatlas.com/
- ESMAtlas30 Foldseek database: Pre-clustered high-quality structures
- AFESM: Combined AlphaFold + ESM Atlas (821M structures)

References:
- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein
  structure with a language model. Science.
- van Kempen et al. (2023). Fast and accurate protein structure search
  with Foldseek. Nature Biotechnology.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vhold.utils.constants import VHOLD_DB_DIR
from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# ESM Atlas database configuration
ESMATLAS_DB_NAME = "ESMAtlas30"
ESMATLAS_DB_PATH = VHOLD_DB_DIR / "esmatlas30"

# AFESM configuration (when available)
AFESM_DB_NAME = "AFESM"
AFESM_SEARCH_URL = "https://afesm.foldseek.com"

# Quality thresholds for ESM Atlas hits
ESMATLAS_MIN_PLDDT = 0.7  # Minimum predicted LDDT
ESMATLAS_MIN_PTM = 0.7  # Minimum predicted TM score


@dataclass
class ESMAtlasHit:
    """A hit from ESM Atlas search."""

    query_id: str
    target_id: str  # MGnify or ESM Atlas identifier
    evalue: float
    bits: float
    fident: float  # Fraction identity
    alnlen: int  # Alignment length
    qcov: float  # Query coverage
    tcov: float  # Target coverage

    # ESM Atlas metadata (if available)
    mgnify_id: Optional[str] = None
    predicted_plddt: Optional[float] = None
    predicted_ptm: Optional[float] = None
    cluster_id: Optional[str] = None

    # Structural classification
    is_novel_fold: bool = False  # True if not in AFDB
    structural_cluster: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "target_id": self.target_id,
            "evalue": self.evalue,
            "bits": self.bits,
            "fident": self.fident,
            "alnlen": self.alnlen,
            "qcov": self.qcov,
            "tcov": self.tcov,
            "mgnify_id": self.mgnify_id,
            "predicted_plddt": self.predicted_plddt,
            "predicted_ptm": self.predicted_ptm,
            "cluster_id": self.cluster_id,
            "is_novel_fold": self.is_novel_fold,
            "structural_cluster": self.structural_cluster,
        }


@dataclass
class ESMAtlasSearchResult:
    """Results from ESM Atlas search for a single query."""

    query_id: str
    query_length: int

    # Search results
    hits: list[ESMAtlasHit] = field(default_factory=list)
    total_hits: int = 0
    best_evalue: Optional[float] = None
    best_identity: Optional[float] = None

    # Analysis
    has_metagenomic_homolog: bool = False
    has_novel_fold_match: bool = False
    unique_clusters: int = 0

    # Environmental context (if MGnify metadata available)
    environmental_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query_length": self.query_length,
            "total_hits": self.total_hits,
            "best_evalue": self.best_evalue,
            "best_identity": self.best_identity,
            "has_metagenomic_homolog": self.has_metagenomic_homolog,
            "has_novel_fold_match": self.has_novel_fold_match,
            "unique_clusters": self.unique_clusters,
            "environmental_sources": self.environmental_sources,
            "hits": [h.to_dict() for h in self.hits[:10]],  # Top 10 hits
        }


def check_esmatlas_database() -> bool:
    """Check if ESMAtlas30 database is installed.

    Returns:
        True if database is available
    """
    db_path = ESMATLAS_DB_PATH
    # Foldseek databases consist of multiple files
    required_files = [
        db_path.with_suffix(".dbtype"),
        db_path.with_suffix(".index"),
    ]
    return all(f.exists() for f in required_files)


def install_esmatlas_database(
    output_dir: Path | None = None,
    force: bool = False,
) -> bool:
    """Install the ESMAtlas30 Foldseek database.

    Args:
        output_dir: Directory for database (default: VHOLD_DB_DIR)
        force: Force reinstall if exists

    Returns:
        True if installation successful
    """
    if output_dir is None:
        output_dir = VHOLD_DB_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "esmatlas30"

    if check_esmatlas_database() and not force:
        logger.info("ESMAtlas30 database already installed")
        return True

    logger.info("Downloading ESMAtlas30 database (~50 GB compressed)...")
    logger.info("This may take a while depending on your connection speed.")

    try:
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    "foldseek",
                    "databases",
                    ESMATLAS_DB_NAME,
                    str(db_path),
                    str(tmp_dir),
                ],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout for large download
            )

            if result.returncode != 0:
                logger.error(f"Database download failed: {result.stderr}")
                return False

        logger.info(f"ESMAtlas30 database installed at {db_path}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Database download timed out")
        return False
    except FileNotFoundError:
        logger.error("Foldseek not found. Please install foldseek first.")
        return False
    except Exception as e:
        logger.error(f"Database installation failed: {e}")
        return False


def search_esmatlas(
    query_db: Path,
    output_dir: Path,
    evalue: float = 1e-3,
    threads: int = 4,
    max_hits: int = 100,
) -> Path:
    """Search queries against ESMAtlas30 using Foldseek.

    Args:
        query_db: Path to query Foldseek database
        output_dir: Output directory for results
        evalue: E-value threshold
        threads: Number of CPU threads
        max_hits: Maximum hits per query

    Returns:
        Path to results file

    Raises:
        RuntimeError: If search fails or database not installed
    """
    if not check_esmatlas_database():
        raise RuntimeError(
            "ESMAtlas30 database not installed. "
            "Run: vhold install --esmatlas"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "esmatlas_results.tsv"
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    logger.info(f"Searching ESMAtlas30 (e-value: {evalue})...")

    try:
        result = subprocess.run(
            [
                "foldseek",
                "easy-search",
                str(query_db),
                str(ESMATLAS_DB_PATH),
                str(results_path),
                str(tmp_dir),
                "-e", str(evalue),
                "--threads", str(threads),
                "--max-seqs", str(max_hits),
                "--format-output",
                "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,"
                "tstart,tend,evalue,bits,qlen,tlen,qcov,tcov",
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Foldseek search failed: {result.stderr}")
            raise RuntimeError(f"Foldseek search failed: {result.stderr}")

        logger.info(f"ESMAtlas search complete: {results_path}")
        return results_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("ESMAtlas search timed out")
    except FileNotFoundError:
        raise RuntimeError("Foldseek not found")


def parse_esmatlas_results(
    results_path: Path,
    min_identity: float = 0.3,
    min_coverage: float = 0.5,
) -> dict[str, ESMAtlasSearchResult]:
    """Parse ESMAtlas Foldseek results.

    Args:
        results_path: Path to results TSV
        min_identity: Minimum sequence identity filter
        min_coverage: Minimum query coverage filter

    Returns:
        Dict mapping query ID to ESMAtlasSearchResult
    """
    results: dict[str, ESMAtlasSearchResult] = {}

    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return results

    with open(results_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 16:
                continue

            query_id = parts[0]
            target_id = parts[1]
            fident = float(parts[2])
            alnlen = int(parts[3])
            evalue = float(parts[10])
            bits = float(parts[11])
            qlen = int(parts[12])
            qcov = float(parts[14])
            tcov = float(parts[15])

            # Apply filters
            if fident < min_identity or qcov < min_coverage:
                continue

            # Create hit
            hit = ESMAtlasHit(
                query_id=query_id,
                target_id=target_id,
                evalue=evalue,
                bits=bits,
                fident=fident,
                alnlen=alnlen,
                qcov=qcov,
                tcov=tcov,
            )

            # Extract MGnify ID if present in target
            if "MGYP" in target_id or "mgnify" in target_id.lower():
                hit.mgnify_id = target_id

            # Check if this is likely a novel fold (heuristic)
            # Novel folds are typically only in ESM Atlas, not AFDB
            if "ESM" in target_id or "MGY" in target_id:
                hit.is_novel_fold = True

            # Add to results
            if query_id not in results:
                results[query_id] = ESMAtlasSearchResult(
                    query_id=query_id,
                    query_length=qlen,
                )

            results[query_id].hits.append(hit)
            results[query_id].total_hits += 1

            # Update best scores
            if (
                results[query_id].best_evalue is None
                or evalue < results[query_id].best_evalue
            ):
                results[query_id].best_evalue = evalue
            if (
                results[query_id].best_identity is None
                or fident > results[query_id].best_identity
            ):
                results[query_id].best_identity = fident

    # Post-process results
    for result in results.values():
        result.has_metagenomic_homolog = result.total_hits > 0
        result.has_novel_fold_match = any(h.is_novel_fold for h in result.hits)

        # Count unique clusters (by target prefix)
        clusters = set()
        for hit in result.hits:
            # Extract cluster from target ID (heuristic)
            parts = hit.target_id.split("_")
            if len(parts) >= 2:
                clusters.add(parts[0])
        result.unique_clusters = len(clusters)

    logger.info(
        f"Parsed {len(results)} queries with ESMAtlas hits "
        f"({sum(r.total_hits for r in results.values())} total hits)"
    )

    return results


@dataclass
class ESMAtlasAnalysis:
    """Analysis results from ESM Atlas search for dark matter proteins."""

    # Counts
    total_queries: int = 0
    queries_with_hits: int = 0
    queries_with_novel_folds: int = 0

    # Hit statistics
    total_hits: int = 0
    unique_targets: int = 0

    # Individual results
    results: dict[str, ESMAtlasSearchResult] = field(default_factory=dict)

    # Recommendations
    novel_fold_candidates: list[str] = field(default_factory=list)
    high_priority_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "summary": {
                "total_queries": self.total_queries,
                "queries_with_hits": self.queries_with_hits,
                "queries_with_novel_folds": self.queries_with_novel_folds,
                "hit_rate": (
                    self.queries_with_hits / self.total_queries
                    if self.total_queries > 0
                    else 0
                ),
                "total_hits": self.total_hits,
                "unique_targets": self.unique_targets,
            },
            "novel_fold_candidates": self.novel_fold_candidates,
            "high_priority_targets": self.high_priority_targets,
            "results": {
                qid: r.to_dict() for qid, r in list(self.results.items())[:100]
            },
        }


def analyze_dark_matter_with_esmatlas(
    dark_matter_proteins: list,  # List of DarkMatterProtein
    predictions_dir: Path,
    output_dir: Path,
    evalue: float = 1e-3,
    threads: int = 4,
) -> ESMAtlasAnalysis:
    """Analyze dark matter proteins against ESM Metagenomic Atlas.

    This function:
    1. Creates a Foldseek database from dark matter 3Di predictions
    2. Searches against ESMAtlas30
    3. Identifies novel fold candidates and metagenomic homologs
    4. Generates prioritized recommendations

    Args:
        dark_matter_proteins: List of DarkMatterProtein objects
        predictions_dir: Directory containing 3Di predictions
        output_dir: Output directory
        evalue: E-value threshold
        threads: Number of CPU threads

    Returns:
        ESMAtlasAnalysis with search results and recommendations
    """
    analysis = ESMAtlasAnalysis(total_queries=len(dark_matter_proteins))

    if not dark_matter_proteins:
        logger.info("No dark matter proteins to analyze")
        return analysis

    # Check if database is available
    if not check_esmatlas_database():
        logger.warning(
            "ESMAtlas30 database not installed. "
            "Skipping ESM Atlas analysis. "
            "Install with: vhold install --esmatlas"
        )
        return analysis

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find 3Di predictions for dark matter proteins
    dm_ids = {p.query_id for p in dark_matter_proteins}
    tdi_files = list(predictions_dir.glob("*.3di"))

    if not tdi_files:
        logger.warning(f"No 3Di files found in {predictions_dir}")
        return analysis

    # Create temporary query database with dark matter proteins only
    query_fasta = output_dir / "dark_matter_queries.fasta"
    with open(query_fasta, "w") as f:
        for tdi_file in tdi_files:
            # Check if this protein is dark matter
            # Assuming filename format: query_id.3di
            protein_id = tdi_file.stem
            if protein_id in dm_ids:
                with open(tdi_file) as tdi:
                    seq = tdi.read().strip()
                    f.write(f">{protein_id}\n{seq}\n")

    # Create Foldseek database
    query_db = output_dir / "dark_matter_db"
    try:
        subprocess.run(
            ["foldseek", "createdb", str(query_fasta), str(query_db)],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create query database: {e}")
        return analysis

    # Search ESMAtlas
    try:
        results_path = search_esmatlas(
            query_db=query_db,
            output_dir=output_dir,
            evalue=evalue,
            threads=threads,
        )
    except RuntimeError as e:
        logger.error(f"ESMAtlas search failed: {e}")
        return analysis

    # Parse results
    analysis.results = parse_esmatlas_results(results_path)

    # Compute statistics
    all_targets = set()
    for result in analysis.results.values():
        if result.has_metagenomic_homolog:
            analysis.queries_with_hits += 1
        if result.has_novel_fold_match:
            analysis.queries_with_novel_folds += 1
            analysis.novel_fold_candidates.append(result.query_id)
        analysis.total_hits += result.total_hits
        for hit in result.hits:
            all_targets.add(hit.target_id)

    analysis.unique_targets = len(all_targets)

    # Identify high priority targets
    # Priority: no BFVD/Viro3D hits but has ESMAtlas hits
    for dm_protein in dark_matter_proteins:
        if dm_protein.category == "no_hits":
            if dm_protein.query_id in analysis.results:
                # Has ESMAtlas hits but no viral DB hits - interesting!
                analysis.high_priority_targets.append(dm_protein.query_id)

    logger.info(
        f"ESMAtlas analysis: {analysis.queries_with_hits}/{analysis.total_queries} "
        f"have metagenomic homologs, {analysis.queries_with_novel_folds} novel fold candidates"
    )

    return analysis


# Export main classes and functions
__all__ = [
    "ESMAtlasHit",
    "ESMAtlasSearchResult",
    "ESMAtlasAnalysis",
    "check_esmatlas_database",
    "install_esmatlas_database",
    "search_esmatlas",
    "parse_esmatlas_results",
    "analyze_dark_matter_with_esmatlas",
]
