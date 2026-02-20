"""UniProt REST API client for resolving UniRef50 hits to functional annotations.

Used by STARLING similarity search to annotate disordered proteins matched
against the 35M UniRef50 IDR database. Queries the UniProt REST API for
protein names, gene names, GO terms, Pfam domains, and organism info.

Requires network access for live lookups. Results are cached in-memory
within a pipeline run and optionally persisted to disk across runs.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from vhold.utils.constants import UNIPROT_BATCH_SIZE
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# UniProt REST API endpoint
UNIPROT_API_BASE = "https://rest.uniprot.org"

# User-Agent for API requests (UniProt requests identification)
USER_AGENT = "vHold/1.0 (https://github.com/shandley/vhold)"

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4

# Fields to request from UniProt
UNIPROT_FIELDS = [
    "accession",
    "protein_name",
    "gene_names",
    "organism_name",
    "go_biological_process",
    "go_molecular_function",
    "xref_pfam",
    "keyword",
]


def extract_uniprot_accession(uniref50_id: str) -> str | None:
    """Extract UniProt accession from a UniRef50 cluster ID.

    Args:
        uniref50_id: UniRef50 ID (e.g., "UniRef50_Q9BYF1")

    Returns:
        UniProt accession (e.g., "Q9BYF1") or None if unparseable
    """
    # UniRef50 format: "UniRef50_<accession>"
    match = re.match(r"UniRef50_(\w+)", uniref50_id)
    if match:
        return match.group(1)

    # Fallback: try stripping common prefixes
    if uniref50_id.startswith("UniRef50_"):
        suffix = uniref50_id[9:]
        return suffix if suffix else None

    return None


def lookup_uniprot_batch(
    accessions: list[str],
    cache: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Look up annotations for UniProt accessions via REST API.

    Args:
        accessions: List of UniProt accessions to look up
        cache: Optional cache dict to check/populate

    Returns:
        Dict mapping accession to annotation dict
    """

    if cache is None:
        cache = {}

    # Filter to uncached accessions
    uncached = [acc for acc in accessions if acc not in cache]

    if not uncached:
        return {acc: cache[acc] for acc in accessions if acc in cache}

    # Process in batches
    for i in range(0, len(uncached), UNIPROT_BATCH_SIZE):
        batch = uncached[i : i + UNIPROT_BATCH_SIZE]
        batch_results = _fetch_uniprot_batch(batch)

        for acc, annotation in batch_results.items():
            cache[acc] = annotation

        # Rate limiting: 1 request per second
        if i + UNIPROT_BATCH_SIZE < len(uncached):
            time.sleep(1.0)

    return {acc: cache[acc] for acc in accessions if acc in cache}


def _parse_retry_after(header_value: str | None, attempt: int) -> float:
    """Parse Retry-After header to a wait time in seconds.

    Handles both delay-seconds ("120") and HTTP-date formats per RFC 9110.
    Falls back to exponential backoff if unparseable.

    Args:
        header_value: Raw Retry-After header value (or None)
        attempt: Current retry attempt (0-indexed) for backoff calculation

    Returns:
        Wait time in seconds
    """
    if header_value is not None:
        try:
            return float(header_value)
        except (ValueError, TypeError):
            pass
    return RETRY_BACKOFF_BASE * (2 ** attempt)


def _fetch_uniprot_batch(accessions: list[str]) -> dict[str, dict]:
    """Fetch a single batch of UniProt annotations via GET.

    Retries with exponential backoff on transient errors (429, 503).

    Args:
        accessions: List of accessions (max UNIPROT_BATCH_SIZE)

    Returns:
        Dict mapping accession to parsed annotation
    """
    fields_param = ",".join(UNIPROT_FIELDS)
    query = " OR ".join(f"accession:{acc}" for acc in accessions)
    url = (
        f"{UNIPROT_API_BASE}/uniprotkb/search"
        f"?query={urllib.parse.quote(query)}"
        f"&fields={fields_param}"
        f"&format=json"
        f"&size={len(accessions)}"
    )

    results: dict[str, dict] = {}

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", USER_AGENT)

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                if acc:
                    results[acc] = _parse_uniprot_entry(entry)

            return results

        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES - 1:
                wait = _parse_retry_after(
                    e.headers.get("Retry-After") if e.code == 429 else None,
                    attempt,
                )
                logger.warning(
                    f"UniProt API error ({e.code}), retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                logger.warning(f"UniProt API request failed: {e}")
                return results
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"UniProt request failed, retrying in {wait:.1f}s: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"UniProt API request failed after {MAX_RETRIES} attempts: {e}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"UniProt API response parsing failed: {e}")
            return results

    return results


def _parse_uniprot_entry(entry: dict) -> dict:
    """Parse a UniProt JSON entry into vHold annotation format.

    Normalizes UniProt's nested JSON structure to match the flat dict
    format used by BFVD/Viro3D annotation lookups.

    Args:
        entry: Raw UniProt JSON entry

    Returns:
        Annotation dict compatible with classify_protein()
    """
    annotation: dict = {
        "source": "uniprot",
        "target_id": entry.get("primaryAccession", ""),
    }

    # Protein name (recommended name > submitted name)
    protein_name = entry.get("proteinDescription", {})
    rec_name = protein_name.get("recommendedName", {})
    sub_names = protein_name.get("submissionNames", [])

    if rec_name:
        full_name = rec_name.get("fullName", {})
        annotation["description"] = full_name.get("value", "hypothetical protein")
    elif sub_names:
        full_name = sub_names[0].get("fullName", {})
        annotation["description"] = full_name.get("value", "hypothetical protein")
    else:
        annotation["description"] = "hypothetical protein"

    # Gene names
    genes = entry.get("genes", [])
    if genes:
        gene_names = []
        for gene in genes:
            if "geneName" in gene:
                gene_names.append(gene["geneName"].get("value", ""))
        annotation["gene"] = "; ".join(filter(None, gene_names))

    # Organism
    organism = entry.get("organism", {})
    annotation["organism"] = organism.get("scientificName", "")

    # GO terms and Pfam domains (single pass over cross-references)
    go_bp = []
    go_mf = []
    pfam_domains = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        db = xref.get("database")
        if db == "GO":
            go_id = xref.get("id", "")
            props = {p.get("key"): p.get("value") for p in xref.get("properties", [])}
            term = props.get("GoTerm", "")
            if term.startswith("P:"):
                go_bp.append(f"{term[2:]} [{go_id}]")
            elif term.startswith("F:"):
                go_mf.append(f"{term[2:]} [{go_id}]")
        elif db == "Pfam":
            pfam_id = xref.get("id", "")
            props = {p.get("key"): p.get("value") for p in xref.get("properties", [])}
            name = props.get("EntryName", pfam_id)
            pfam_domains.append(f"{pfam_id}:{name}")

    if go_bp:
        annotation["go_bp"] = "; ".join(go_bp)
    if go_mf:
        annotation["go_mf"] = "; ".join(go_mf)
    if pfam_domains:
        annotation["pfam"] = "; ".join(pfam_domains)

    # Keywords
    keywords = entry.get("keywords", [])
    if keywords:
        annotation["keywords"] = "; ".join(kw.get("name", "") for kw in keywords)

    return annotation


def save_lookup_cache(
    cache: dict[str, dict],
    output_path: Path,
) -> None:
    """Save UniProt lookup cache to JSON file.

    Uses atomic write (temp file + rename) to prevent corruption
    if the process is interrupted mid-write.

    Args:
        cache: Cache dict mapping accession to annotation
        output_path: Path to write JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=output_path.parent, suffix=".json.tmp",
    )
    try:
        with open(fd, "w") as f:
            json.dump(cache, f, indent=2)
        Path(tmp_path).replace(output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    logger.info(f"Saved UniProt cache ({len(cache)} entries) to {output_path}")


def load_lookup_cache(cache_path: Path) -> dict[str, dict]:
    """Load UniProt lookup cache from JSON file.

    Handles corrupt/truncated cache files gracefully by starting fresh.

    Args:
        cache_path: Path to JSON cache file

    Returns:
        Cache dict mapping accession to annotation
    """
    if not cache_path.exists():
        return {}

    try:
        with open(cache_path) as f:
            cache = json.load(f)
        logger.info(f"Loaded UniProt cache ({len(cache)} entries) from {cache_path}")
        return cache
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Corrupt UniProt cache at {cache_path}, starting fresh: {e}")
        return {}
