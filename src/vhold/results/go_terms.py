"""GO term ID resolution for vhold annotations.

Resolves GO IDs (GO:0039694) from GO term names using a bundled
name→ID mapping generated from the Gene Ontology OBO file.

Two input formats are supported:
  1. BFVD enriched: "term_name [GO:ID]; term_name [GO:ID]" → extract IDs directly
  2. Viro3D/plain:  "term_name; term_name" → look up IDs from bundled map
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Pattern to extract GO IDs already present in the string: [GO:0039694]
_INLINE_GO_RE = re.compile(r"\[GO:\d{7}\]")
_GO_ID_RE = re.compile(r"GO:\d{7}")

GO_TERM_MAP_PATH = Path(__file__).parent.parent / "data" / "go_term_map.json"


@lru_cache(maxsize=1)
def load_go_term_map() -> dict[str, str]:
    """Load GO term name → GO ID mapping from bundled JSON.

    Returns:
        Dict mapping lowercase term names to GO IDs.
        Empty dict if the map file is not found.
    """
    if not GO_TERM_MAP_PATH.exists():
        logger.debug("GO term map not found at %s", GO_TERM_MAP_PATH)
        return {}

    with open(GO_TERM_MAP_PATH) as f:
        return json.load(f)


def resolve_go_ids(go_terms_str: str) -> str:
    """Convert GO term string to semicolon-separated GO IDs.

    Handles two formats:
      - "viral genome replication [GO:0019079]; RNA binding [GO:0003723]"
        → "GO:0019079; GO:0003723" (extract inline IDs)
      - "viral genome replication; RNA binding"
        → "GO:0019079; GO:0003723" (look up from map)

    Args:
        go_terms_str: Semicolon-separated GO term names, optionally
            with inline GO IDs in [brackets].

    Returns:
        Semicolon-separated GO IDs. Empty string if no terms resolve.
    """
    if not go_terms_str or not go_terms_str.strip():
        return ""

    # Check if inline GO IDs are present
    inline_ids = _GO_ID_RE.findall(go_terms_str)
    if inline_ids:
        return "; ".join(inline_ids)

    # Look up each term in the map
    term_map = load_go_term_map()
    if not term_map:
        return ""

    go_ids = []
    for term in go_terms_str.split(";"):
        term = term.strip()
        if not term:
            continue
        go_id = term_map.get(term.lower())
        if go_id:
            go_ids.append(go_id)

    return "; ".join(go_ids)


def enrich_annotation_with_go_ids(annotation: dict) -> dict:
    """Add go_bp_ids and go_mf_ids fields from existing go_bp/go_mf.

    Modifies the annotation dict in-place and returns it.

    Args:
        annotation: Annotation dict with optional go_bp and go_mf keys.

    Returns:
        The same dict with go_bp_ids and go_mf_ids added.
    """
    go_bp = annotation.get("go_bp", "")
    if go_bp:
        ids = resolve_go_ids(go_bp)
        if ids:
            annotation["go_bp_ids"] = ids

    go_mf = annotation.get("go_mf", "")
    if go_mf:
        ids = resolve_go_ids(go_mf)
        if ids:
            annotation["go_mf_ids"] = ids

    return annotation
