"""Metagenomic database integrations for dark matter characterization.

This module provides integration with large-scale metagenomic resources
to help characterize viral dark matter proteins:

1. **Serratus** (serratus.io): Ultra-deep search for novel viruses
   - 5.7 million SRA datasets processed
   - 132,000+ novel RNA virus species discovered
   - API for searching viral families and sequences

2. **Logan** (logan-search.org): Planetary-scale genome assembly
   - 27.3 million SRA datasets (96% coverage)
   - 44.1 petabases -> 0.9 petabases assembled contigs
   - k-mer search for sequence distribution

3. **MGnify**: 2.4 billion metagenomic protein sequences
   - Taxonomic and functional annotations
   - Environmental metadata

References:
- Edgar et al. (2022). Petabase-scale sequence alignment catalyses viral discovery. Nature.
- Roux et al. (2024). Logan: Planetary-Scale Genome Assembly. bioRxiv.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import requests

from vhold.utils.logging import get_logger

logger = get_logger(__name__)


# Serratus API configuration
SERRATUS_API_BASE = "https://api.serratus.io"
SERRATUS_DATA_API = "https://api.serratus.io/data"
SERRATUS_AUTH = ("serratus", "serratus")  # Public credentials
SERRATUS_TIMEOUT = 30  # seconds

# Logan search configuration
LOGAN_SEARCH_URL = "https://logan-search.org"
LOGAN_TIMEOUT = 60  # seconds

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests


@dataclass
class SerratusMatch:
    """A match from Serratus viral search."""

    run_id: str  # SRA run accession (e.g., SRR1234567)
    family_name: str  # Viral family (e.g., Coronaviridae)
    percent_identity: float  # Alignment identity
    score: float  # Alignment score
    coverage: float  # Query coverage
    n_reads: int  # Number of supporting reads
    aligned_length: int  # Alignment length

    # Optional metadata
    biosample: str = ""
    bioproject: str = ""
    organism: str = ""
    geo_location: str = ""


@dataclass
class MetagenomicContext:
    """Metagenomic context for a dark matter protein."""

    query_id: str
    query_length: int

    # Serratus results
    serratus_searched: bool = False
    serratus_family_hits: list[str] = field(default_factory=list)
    serratus_total_runs: int = 0
    serratus_best_identity: float = 0.0
    serratus_matches: list[SerratusMatch] = field(default_factory=list)

    # Logan results
    logan_searched: bool = False
    logan_k_mer_hits: int = 0
    logan_environmental_sources: list[str] = field(default_factory=list)

    # Environmental distribution
    environments: list[str] = field(default_factory=list)
    geographic_regions: list[str] = field(default_factory=list)

    # Sequence conservation
    is_conserved_in_metagenomes: bool = False
    conservation_score: float = 0.0  # 0-1, based on metagenomic prevalence

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        return {
            "query_id": self.query_id,
            "query_length": self.query_length,
            "serratus": {
                "searched": self.serratus_searched,
                "family_hits": self.serratus_family_hits,
                "total_runs": self.serratus_total_runs,
                "best_identity": round(self.serratus_best_identity, 3),
            },
            "logan": {
                "searched": self.logan_searched,
                "k_mer_hits": self.logan_k_mer_hits,
                "environmental_sources": self.logan_environmental_sources,
            },
            "distribution": {
                "environments": self.environments,
                "geographic_regions": self.geographic_regions,
            },
            "conservation": {
                "is_conserved": self.is_conserved_in_metagenomes,
                "score": round(self.conservation_score, 3),
            },
            "recommendations": self.recommendations,
        }


class SerratusClient:
    """Client for querying the Serratus API."""

    def __init__(
        self,
        base_url: str = SERRATUS_API_BASE,
        timeout: int = SERRATUS_TIMEOUT,
    ):
        """Initialize Serratus client.

        Args:
            base_url: Base URL for Serratus API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()
        self._session.auth = SERRATUS_AUTH
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def search_by_family(
        self,
        family: str,
        identity_min: float = 50.0,
        score_min: float = 50.0,
        limit: int = 100,
    ) -> list[SerratusMatch]:
        """Search Serratus for runs matching a viral family.

        Args:
            family: Viral family name (e.g., "Coronaviridae")
            identity_min: Minimum percent identity
            score_min: Minimum alignment score
            limit: Maximum results to return

        Returns:
            List of SerratusMatch objects
        """
        self._rate_limit()

        url = urljoin(self.base_url, f"/matches/nucleotide/paged")
        params = {
            "family": family,
            "identityMin": identity_min,
            "scoreMin": score_min,
            "perPage": min(limit, 100),
            "page": 1,
        }

        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            matches = []
            for item in data.get("results", []):
                match = SerratusMatch(
                    run_id=item.get("run_id", ""),
                    family_name=family,
                    percent_identity=float(item.get("percent_identity", 0)),
                    score=float(item.get("score", 0)),
                    coverage=float(item.get("coverage", 0)),
                    n_reads=int(item.get("n_reads", 0)),
                    aligned_length=int(item.get("aligned_length", 0)),
                )
                matches.append(match)

            return matches

        except requests.RequestException as e:
            logger.warning(f"Serratus API error for family {family}: {e}")
            return []

    def get_family_counts(self, family: str) -> dict:
        """Get summary counts for a viral family.

        Args:
            family: Viral family name

        Returns:
            Dict with count statistics
        """
        self._rate_limit()

        url = urljoin(self.base_url, f"/counts/nucleotide")
        params = {"family": family}

        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.warning(f"Serratus counts error for family {family}: {e}")
            return {}

    def list_families(self) -> list[str]:
        """Get list of all viral families in Serratus.

        Returns:
            List of family names
        """
        self._rate_limit()

        url = urljoin(self.base_url, "/list/nucleotide/family")

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("families", [])

        except requests.RequestException as e:
            logger.warning(f"Serratus list families error: {e}")
            return []

    def search_run(self, run_id: str) -> dict:
        """Get detailed information for a specific SRA run.

        Args:
            run_id: SRA run accession (e.g., "SRR1234567")

        Returns:
            Dict with run information
        """
        self._rate_limit()

        url = urljoin(self.base_url, f"/summary/nucleotide/run={run_id}")

        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.warning(f"Serratus run query error for {run_id}: {e}")
            return {}


# Mapping of protein functional keywords to viral families for Serratus search
FUNCTION_TO_VIRAL_FAMILIES = {
    # RNA virus families
    "rdrp": ["Coronaviridae", "Picornaviridae", "Flaviviridae", "Togaviridae"],
    "polymerase": ["Coronaviridae", "Rhabdoviridae", "Paramyxoviridae", "Orthomyxoviridae"],
    "capsid": ["Picornaviridae", "Caliciviridae", "Astroviridae", "Nodaviridae"],
    "nucleocapsid": ["Coronaviridae", "Rhabdoviridae", "Paramyxoviridae", "Bunyavirales"],
    "spike": ["Coronaviridae", "Paramyxoviridae"],
    "protease": ["Coronaviridae", "Picornaviridae", "Flaviviridae"],
    "helicase": ["Coronaviridae", "Flaviviridae", "Picornaviridae"],

    # DNA virus families
    "terminase": ["Caudovirales", "Herpesviridae"],
    "portal": ["Caudovirales", "Herpesviridae"],
    "tail": ["Caudovirales", "Myoviridae", "Siphoviridae", "Podoviridae"],
    "lysin": ["Caudovirales"],
    "integrase": ["Retroviridae", "Caudovirales"],
}


def infer_viral_families_from_description(description: str) -> list[str]:
    """Infer potential viral families from protein description.

    Args:
        description: Protein functional description

    Returns:
        List of potentially related viral families
    """
    families = set()
    desc_lower = description.lower()

    for keyword, family_list in FUNCTION_TO_VIRAL_FAMILIES.items():
        if keyword in desc_lower:
            families.update(family_list)

    return list(families)


def generate_dark_matter_recommendations(
    context: MetagenomicContext,
    has_structural_hits: bool = False,
    functional_category: str = "unknown",
) -> list[str]:
    """Generate recommendations for further characterization of dark matter.

    Args:
        context: MetagenomicContext with search results
        has_structural_hits: Whether structural homologs were found
        functional_category: Current functional classification

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Based on metagenomic prevalence
    if context.is_conserved_in_metagenomes:
        recommendations.append(
            "Conserved in metagenomes - likely functionally important. "
            "Consider experimental characterization."
        )

    # Based on Serratus results
    if context.serratus_total_runs > 100:
        recommendations.append(
            f"Found in {context.serratus_total_runs} SRA datasets via Serratus. "
            "Well-distributed in environmental samples."
        )
    elif context.serratus_total_runs > 0:
        recommendations.append(
            f"Found in {context.serratus_total_runs} SRA datasets. "
            "Consider targeted metagenomic mining."
        )

    # Based on structural search results
    if not has_structural_hits:
        recommendations.append(
            "No structural homologs detected. Consider AlphaFold prediction "
            "followed by Foldseek search against ESM Metagenomic Atlas."
        )
        recommendations.append(
            "Novel fold candidate - priority target for structural biology."
        )

    # Based on length
    if context.query_length < 100:
        recommendations.append(
            "Short protein (<100 aa) - may be a regulatory peptide or "
            "part of a larger polyprotein."
        )
    elif context.query_length > 1000:
        recommendations.append(
            "Large protein (>1000 aa) - may contain multiple domains. "
            "Consider domain parsing with InterProScan."
        )

    # Environmental context
    if context.environments:
        env_str = ", ".join(context.environments[:5])
        recommendations.append(
            f"Associated environments: {env_str}. "
            "May provide clues about host range or ecology."
        )

    # Functional category specific
    if functional_category == "unknown":
        recommendations.append(
            "Unknown function with structural evidence. "
            "Consider remote homology search with HHpred or DALI."
        )

    return recommendations


@dataclass
class DarkMatterCharacterization:
    """Enhanced dark matter characterization with metagenomic context."""

    query_id: str
    query_length: int
    dark_matter_category: str  # no_hits, unknown_function, weak_hits
    original_reason: str

    # Current annotation info
    description: str = "hypothetical protein"
    functional_category: str = "unknown"
    confidence_level: str = "none"

    # Metagenomic context
    metagenomic_context: Optional[MetagenomicContext] = None

    # Additional characterization
    sequence_features: dict = field(default_factory=dict)
    predicted_properties: dict = field(default_factory=dict)

    # Priority for follow-up
    priority_score: float = 0.0  # 0-1, higher = more interesting
    priority_reasons: list[str] = field(default_factory=list)

    def calculate_priority(self) -> None:
        """Calculate priority score for follow-up analysis."""
        score = 0.0
        reasons = []

        # Novel proteins (no hits) are highest priority
        if self.dark_matter_category == "no_hits":
            score += 0.4
            reasons.append("No structural homologs detected")

        # Conservation in metagenomes increases priority
        if self.metagenomic_context and self.metagenomic_context.is_conserved_in_metagenomes:
            score += 0.3
            reasons.append("Conserved across metagenomes")

        # Medium-length proteins are more likely to be functional
        if 100 <= self.query_length <= 500:
            score += 0.1
            reasons.append("Optimal length for single-domain protein")

        # Unknown function with structural hits is interesting
        if self.dark_matter_category == "unknown_function" and self.confidence_level in (
            "high",
            "medium",
        ):
            score += 0.2
            reasons.append("Strong structural match but unknown function")

        self.priority_score = min(1.0, score)
        self.priority_reasons = reasons

    def to_dict(self) -> dict:
        """Convert to dictionary for output."""
        result = {
            "query_id": self.query_id,
            "query_length": self.query_length,
            "dark_matter_category": self.dark_matter_category,
            "original_reason": self.original_reason,
            "description": self.description,
            "functional_category": self.functional_category,
            "confidence_level": self.confidence_level,
            "priority": {
                "score": round(self.priority_score, 3),
                "reasons": self.priority_reasons,
            },
            "sequence_features": self.sequence_features,
            "predicted_properties": self.predicted_properties,
        }

        if self.metagenomic_context:
            result["metagenomic_context"] = self.metagenomic_context.to_dict()

        return result


def characterize_dark_matter_batch(
    dark_matter_proteins: list,  # List of DarkMatterProtein
    sequences: dict[str, str] | None = None,
    search_serratus: bool = True,
    search_logan: bool = False,  # Disabled by default (requires manual setup)
) -> list[DarkMatterCharacterization]:
    """Characterize a batch of dark matter proteins with metagenomic context.

    Args:
        dark_matter_proteins: List of DarkMatterProtein from dark_matter.py
        sequences: Optional dict mapping query_id to amino acid sequence
        search_serratus: Whether to search Serratus for viral context
        search_logan: Whether to search Logan (requires external setup)

    Returns:
        List of DarkMatterCharacterization objects
    """
    results = []

    # Initialize Serratus client if needed
    serratus_client = SerratusClient() if search_serratus else None

    for dm_protein in dark_matter_proteins:
        # Create base characterization
        char = DarkMatterCharacterization(
            query_id=dm_protein.query_id,
            query_length=dm_protein.query_length,
            dark_matter_category=dm_protein.category,
            original_reason=dm_protein.reason,
            description=dm_protein.description,
            confidence_level=dm_protein.confidence_level,
        )

        # Create metagenomic context
        context = MetagenomicContext(
            query_id=dm_protein.query_id,
            query_length=dm_protein.query_length,
        )

        # Search Serratus if enabled
        if serratus_client and dm_protein.description:
            families = infer_viral_families_from_description(dm_protein.description)
            if families:
                for family in families[:3]:  # Limit to top 3 families
                    matches = serratus_client.search_by_family(family, limit=10)
                    if matches:
                        context.serratus_searched = True
                        context.serratus_family_hits.append(family)
                        context.serratus_total_runs += len(matches)
                        context.serratus_matches.extend(matches)

                        # Track best identity
                        for match in matches:
                            if match.percent_identity > context.serratus_best_identity:
                                context.serratus_best_identity = match.percent_identity

        # Determine if conserved in metagenomes
        if context.serratus_total_runs >= 10:
            context.is_conserved_in_metagenomes = True
            context.conservation_score = min(1.0, context.serratus_total_runs / 100)

        # Generate recommendations
        context.recommendations = generate_dark_matter_recommendations(
            context,
            has_structural_hits=(dm_protein.category != "no_hits"),
            functional_category=char.functional_category,
        )

        char.metagenomic_context = context
        char.calculate_priority()

        results.append(char)

    logger.info(
        f"Characterized {len(results)} dark matter proteins "
        f"({sum(1 for r in results if r.priority_score >= 0.5)} high priority)"
    )

    return results


# Export main classes and functions
__all__ = [
    "SerratusClient",
    "SerratusMatch",
    "MetagenomicContext",
    "DarkMatterCharacterization",
    "characterize_dark_matter_batch",
    "infer_viral_families_from_description",
    "generate_dark_matter_recommendations",
]
