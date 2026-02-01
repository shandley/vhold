"""UniProt API integration for fetching protein annotations."""

import time
from functools import lru_cache

import requests

from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# UniProt REST API base URL
UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"

# Fields to fetch from UniProt
UNIPROT_FIELDS = [
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "length",
]

# Batch size for UniProt API requests (max 500 per request)
BATCH_SIZE = 100

# Rate limiting: UniProt allows ~10 requests/second
REQUEST_DELAY = 0.1


def fetch_uniprot_entry(accession: str) -> dict | None:
    """Fetch a single UniProt entry by accession.

    Args:
        accession: UniProt accession (e.g., P12345)

    Returns:
        Dict with protein info or None if not found
    """
    url = f"{UNIPROT_API_BASE}/{accession}"
    params = {
        "format": "json",
        "fields": ",".join(UNIPROT_FIELDS),
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return _parse_uniprot_entry(data)
        elif response.status_code == 404:
            logger.debug(f"UniProt entry not found: {accession}")
            return None
        else:
            logger.warning(f"UniProt API error {response.status_code} for {accession}")
            return None
    except requests.RequestException as e:
        logger.warning(f"UniProt API request failed for {accession}: {e}")
        return None


def fetch_uniprot_batch(accessions: list[str]) -> dict[str, dict]:
    """Fetch multiple UniProt entries in batches.

    Uses the UniProt search API to fetch multiple entries efficiently.

    Args:
        accessions: List of UniProt accessions

    Returns:
        Dict mapping accession to protein info
    """
    if not accessions:
        return {}

    results = {}
    unique_accessions = list(set(accessions))

    logger.info(f"Fetching {len(unique_accessions)} UniProt annotations...")

    # Process in batches
    for i in range(0, len(unique_accessions), BATCH_SIZE):
        batch = unique_accessions[i:i + BATCH_SIZE]
        batch_results = _fetch_batch(batch)
        results.update(batch_results)

        # Rate limiting
        if i + BATCH_SIZE < len(unique_accessions):
            time.sleep(REQUEST_DELAY)

    logger.info(f"Retrieved {len(results)}/{len(unique_accessions)} UniProt annotations")
    return results


def _fetch_batch(accessions: list[str]) -> dict[str, dict]:
    """Fetch a batch of UniProt entries.

    Args:
        accessions: List of accessions (max BATCH_SIZE)

    Returns:
        Dict mapping accession to protein info
    """
    # Build query string: accession:P12345 OR accession:P67890 ...
    query = " OR ".join(f"accession:{acc}" for acc in accessions)

    url = f"{UNIPROT_API_BASE}/search"
    params = {
        "query": query,
        "format": "json",
        "fields": ",".join(UNIPROT_FIELDS),
        "size": len(accessions),
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        if response.status_code != 200:
            logger.warning(f"UniProt batch API error: {response.status_code}")
            return {}

        data = response.json()
        results = {}

        for entry in data.get("results", []):
            parsed = _parse_uniprot_entry(entry)
            if parsed:
                results[parsed["accession"]] = parsed

        return results

    except requests.RequestException as e:
        logger.warning(f"UniProt batch API request failed: {e}")
        return {}


def _parse_uniprot_entry(entry: dict) -> dict | None:
    """Parse a UniProt API response entry.

    Args:
        entry: Raw UniProt JSON entry

    Returns:
        Parsed dict with standardized fields
    """
    try:
        accession = entry.get("primaryAccession", "")
        if not accession:
            return None

        # Extract protein name
        protein_name = "hypothetical protein"
        protein_desc = entry.get("proteinDescription", {})

        # Try recommended name first
        rec_name = protein_desc.get("recommendedName", {})
        if rec_name:
            full_name = rec_name.get("fullName", {})
            protein_name = full_name.get("value", protein_name)
        else:
            # Fall back to submitted name
            sub_names = protein_desc.get("submissionNames", [])
            if sub_names:
                full_name = sub_names[0].get("fullName", {})
                protein_name = full_name.get("value", protein_name)

        # Extract gene name
        gene_name = ""
        genes = entry.get("genes", [])
        if genes:
            gene_data = genes[0]
            gene_name = gene_data.get("geneName", {}).get("value", "")
            if not gene_name:
                # Try ordered locus name
                olns = gene_data.get("orderedLocusNames", [])
                if olns:
                    gene_name = olns[0].get("value", "")

        # Extract organism
        organism = ""
        org_data = entry.get("organism", {})
        if org_data:
            organism = org_data.get("scientificName", "")

        # Extract length
        length = entry.get("sequence", {}).get("length", 0)

        return {
            "accession": accession,
            "entry_name": entry.get("uniProtkbId", ""),
            "description": protein_name,
            "gene": gene_name,
            "organism": organism,
            "length": length,
        }

    except Exception as e:
        logger.debug(f"Error parsing UniProt entry: {e}")
        return None


@lru_cache(maxsize=10000)
def get_cached_uniprot_annotation(accession: str) -> dict | None:
    """Get UniProt annotation with caching.

    Args:
        accession: UniProt accession

    Returns:
        Protein info dict or None
    """
    return fetch_uniprot_entry(accession)


class UniProtAnnotationCache:
    """Cache for UniProt annotations with batch prefetching."""

    def __init__(self):
        """Initialize the cache."""
        self._cache: dict[str, dict | None] = {}
        self._fetched = False

    def prefetch(self, accessions: list[str]) -> None:
        """Prefetch annotations for a list of accessions.

        Args:
            accessions: List of UniProt accessions to prefetch
        """
        # Filter out already cached accessions
        missing = [acc for acc in accessions if acc not in self._cache]

        if missing:
            results = fetch_uniprot_batch(missing)
            for acc in missing:
                self._cache[acc] = results.get(acc)

        self._fetched = True

    def get(self, accession: str) -> dict | None:
        """Get annotation for an accession.

        Args:
            accession: UniProt accession

        Returns:
            Protein info dict or None
        """
        if accession in self._cache:
            return self._cache[accession]

        # Fetch single entry if not prefetched
        result = fetch_uniprot_entry(accession)
        self._cache[accession] = result
        return result

    def __contains__(self, accession: str) -> bool:
        """Check if accession is in cache."""
        return accession in self._cache

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
