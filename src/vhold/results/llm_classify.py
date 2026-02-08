"""LLM-based functional classification for viral proteins.

Uses Claude to classify proteins where keyword matching returns "unknown".
The LLM's world knowledge about virology resolves cases where generic
descriptions (e.g., "V protein", "Gene: VP35") cannot be mapped by keywords.

Requires: pip install vhold[llm]  (installs anthropic package)
Requires: ANTHROPIC_API_KEY environment variable
"""

import json
import os
from typing import Optional

from vhold.results.categories import ALL_CATEGORIES
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a virology expert classifying viral proteins into functional categories.

Given a protein description, gene name, and organism, classify the protein into exactly one of these categories:

- structural: Virion structural proteins (capsid, envelope, matrix, nucleocapsid, nucleoprotein, glycoprotein, fusion protein, attachment protein)
- replication: Genome replication machinery (polymerase, helicase, primase, RdRp, phosphoprotein/P protein as polymerase cofactor)
- protease: Proteolytic enzymes
- nuclease: Nucleic acid processing enzymes (nuclease, integrase, ligase)
- packaging: Genome packaging (terminase, scaffold)
- regulatory: Gene regulation (transcription factors, kinases, transcription activators)
- movement: Cell-to-cell movement proteins (plant viruses)
- lysis: Host cell lysis proteins (bacteriophages)
- host_interaction: Host interaction and immune evasion (interferon antagonists, immune modulators, V/C proteins of paramyxoviruses)
- entry: Host cell entry and membrane fusion
- unknown: Cannot determine function from available information

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{"category": "<category>", "reasoning": "<one sentence explanation>"}"""

USER_PROMPT_TEMPLATE = """Classify this viral protein:
Description: {description}
Gene: {gene}
Organism: {organism}"""


def classify_single(
    client,
    description: str,
    gene: str | None = None,
    organism: str | None = None,
    model: str = DEFAULT_MODEL,
    valid_categories: list[str] | None = None,
) -> tuple[str, str]:
    """Classify a single protein using Claude.

    Args:
        client: Anthropic client instance
        description: Protein description from structural search
        gene: Gene name if available
        organism: Organism/virus name if available
        model: Claude model to use
        valid_categories: List of valid category strings

    Returns:
        Tuple of (category, reasoning)
    """
    if valid_categories is None:
        valid_categories = [c for c in ALL_CATEGORIES if c != "unknown"]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        description=description or "hypothetical protein",
        gene=gene or "unknown",
        organism=organism or "unknown",
    )

    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Parse response
    text = response.content[0].text.strip()

    # Handle markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        category = result.get("category", "unknown")
        reasoning = result.get("reasoning", "")

        # Validate category
        if category not in ALL_CATEGORIES:
            logger.warning(
                f"LLM returned invalid category '{category}' for '{description}', "
                f"falling back to unknown"
            )
            return "unknown", f"invalid category: {category}"

        return category, reasoning

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response for '{description}': {e}")
        return "unknown", f"parse error: {e}"


def llm_reclassify(
    annotations: dict,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    reclassify_all_keywords: bool = False,
) -> dict:
    """Reclassify proteins with unknown function using LLM.

    Only targets proteins where:
    - classification_source == "keywords" AND functional_category == "unknown"
    - Or if reclassify_all_keywords=True, all keyword-classified proteins

    Modifies ConsensusResult objects in-place.

    Args:
        annotations: Dict mapping query_id to ConsensusResult
        model: Claude model to use
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        reclassify_all_keywords: If True, reclassify all keyword-classified proteins

    Returns:
        The same annotations dict (modified in-place)
    """
    # Layer 1: Check for anthropic package
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "anthropic package not installed. "
            "Install with: pip install vhold[llm]"
        )
        return annotations

    # Layer 2: Check for API key
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning(
            "ANTHROPIC_API_KEY not set. "
            "Set the environment variable or pass api_key parameter."
        )
        return annotations

    # Find candidates for reclassification
    candidates = {}
    for query_id, result in annotations.items():
        if not result.is_annotated:
            continue
        if result.classification_source != "keywords":
            continue
        if not reclassify_all_keywords and result.functional_category != "unknown":
            continue
        candidates[query_id] = result

    if not candidates:
        logger.info("No proteins need LLM reclassification")
        return annotations

    logger.info(f"LLM reclassification: {len(candidates)} candidate proteins")

    # Create client
    client = anthropic.Anthropic(api_key=key)

    reclassified = 0
    for query_id, result in candidates.items():
        # Extract organism from query_id if possible (e.g., "NP_112023.1|Nipah")
        organism = None
        if "|" in query_id:
            parts = query_id.split("|")
            if len(parts) >= 2:
                organism = parts[1]

        gene = result.primary_annotation.get("gene")

        # Layer 3: Per-protein error handling
        try:
            category, reasoning = classify_single(
                client=client,
                description=result.description,
                gene=gene,
                organism=organism,
                model=model,
            )

            if category != "unknown" and category != result.functional_category:
                old_cat = result.functional_category
                result.functional_category = category
                result.classification_source = f"llm:{model}"
                reclassified += 1
                logger.info(
                    f"  {query_id}: {old_cat} -> {category} ({reasoning})"
                )
            else:
                logger.debug(
                    f"  {query_id}: stays {result.functional_category} ({reasoning})"
                )

        except Exception as e:
            logger.warning(f"  {query_id}: LLM error, skipping: {e}")
            continue

    logger.info(f"LLM reclassification complete: {reclassified}/{len(candidates)} proteins updated")
    return annotations
