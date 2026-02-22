"""Two-layer GO term plausibility filtering for cross-domain transfers.

Layer 1 (hard filter): Data-driven viral GO whitelist. Any GO term
annotated to at least one viral Swiss-Prot entry is considered plausible
for viruses. Cross-domain terms NOT in this set are removed.

Layer 2 (LLM filter): Contextual validation of removed terms. The LLM
can rescue genuinely plausible novel functions that happen to not yet
be annotated in any virus (e.g., a viral kinase with a novel substrate).

Same-domain (viral→viral) GO terms are never filtered.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vhold.databases.swissprot import SwissProtEntry

from vhold.features.go_transfer import GOTransferResult, TransferredGOTerm
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"

GO_FILTER_SYSTEM_PROMPT = """You are a viral biology expert validating GO term annotations.

GO terms were transferred from non-viral organisms to viral proteins based on structural embedding similarity. For each term, determine if the function is biologically plausible for a virus.

Viruses CAN:
- Encode enzymes (polymerases, helicases, nucleases, proteases, kinases, ligases)
- Have structural components (capsid, tail, envelope, membrane)
- Interact with host cell machinery (membranes, nucleus, cytoplasm, ER, mitochondria)
- Modulate host processes (cell cycle, immune response, apoptosis, transcription)
- Bind nucleic acids, lipids, and metals

Viruses CANNOT:
- Perform photosynthesis or have chloroplasts
- Undergo embryonic/organ/tissue development
- Have their own kinetochores, centrosomes, or spindle apparatus
- Grow pollen tubes, develop flowers, seeds, or leaves
- Perform multicellular organism-specific developmental processes
- Have their own ribosomes, peroxisomes, or vacuoles

For each GO term, respond with ONLY a JSON array (no markdown, no extra text):
[{"go_id": "GO:...", "verdict": "keep", "reason": "..."}, ...]

Use "keep" for plausible viral functions, "reject" for impossible ones."""

GO_FILTER_USER_TEMPLATE = """Protein: {protein_id}
Matched to: {donor_organism} ({donor_accession}, similarity={similarity:.3f})

Validate these GO terms:
{term_list}"""


def build_viral_go_set(metadata: dict[str, SwissProtEntry]) -> set[str]:
    """Extract all GO IDs annotated to viral entries in the metadata.

    Args:
        metadata: Swiss-Prot metadata dict (may include viral and non-viral)

    Returns:
        Set of GO IDs (e.g., {"GO:0039694", "GO:0003899", ...})
    """
    viral_go_ids: set[str] = set()
    for entry in metadata.values():
        if entry.is_viral:
            for go in entry.go_annotations:
                viral_go_ids.add(go.go_id)
    return viral_go_ids


def filter_cross_domain_go_terms(
    results: dict[str, GOTransferResult],
    viral_go_ids: set[str],
    llm_validate: bool = False,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
) -> dict[str, GOTransferResult]:
    """Two-layer cross-domain GO term filtering.

    Layer 1 (hard filter): Remove cross-domain GO terms whose GO IDs
    are not in the viral GO whitelist.

    Layer 2 (LLM filter, optional): Validate removed terms with LLM
    and rescue plausible ones.

    Same-domain (viral) GO terms are never filtered.

    Args:
        results: GO transfer results keyed by protein ID
        viral_go_ids: Set of GO IDs from viral Swiss-Prot entries
        llm_validate: Whether to use LLM for uncertain terms
        llm_model: Claude model for LLM validation
        api_key: Anthropic API key (defaults to env var)

    Returns:
        The same results dict with filtered GO terms
    """
    if not viral_go_ids:
        logger.info("No viral GO set available, skipping GO plausibility filter")
        return results

    # Layer 1: Hard filter
    flagged: list[tuple[str, TransferredGOTerm, GOTransferResult]] = []
    n_kept = 0
    n_removed = 0

    for qid, result in results.items():
        if not result.go_terms:
            continue

        kept_terms: list[TransferredGOTerm] = []
        for term in result.go_terms:
            if not term.is_cross_domain:
                # Same-domain terms are never filtered
                kept_terms.append(term)
                n_kept += 1
            elif term.go_id in viral_go_ids:
                # Cross-domain but known in viral context
                kept_terms.append(term)
                n_kept += 1
            else:
                # Cross-domain and not in viral GO set — flag
                flagged.append((qid, term, result))
                n_removed += 1

        result.go_terms = kept_terms

    logger.info(
        f"GO plausibility filter: {n_kept} terms kept, "
        f"{n_removed} cross-domain terms flagged"
    )

    if not flagged:
        return results

    # Layer 2: LLM validation of flagged terms
    if llm_validate:
        rescued = _validate_go_terms_llm(
            flagged_terms=flagged,
            model=llm_model,
            api_key=api_key,
        )

        if rescued:
            # Add rescued terms back to their results
            for qid, term, result in flagged:
                if (qid, term.go_id) in rescued:
                    result.go_terms.append(term)
                    n_removed -= 1
                    n_kept += 1

            # Re-sort terms by RI
            for result in results.values():
                result.go_terms.sort(
                    key=lambda t: t.reliability_index, reverse=True
                )

            logger.info(f"LLM rescued {len(rescued)} terms")
    else:
        logger.info(
            f"Removed {n_removed} implausible cross-domain terms "
            f"(use --llm to validate with LLM)"
        )

    return results


def _validate_go_terms_llm(
    flagged_terms: list[tuple[str, TransferredGOTerm, GOTransferResult]],
    model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
) -> set[tuple[str, str]]:
    """Validate flagged cross-domain GO terms using LLM.

    Groups terms by protein for efficient batching (one API call per protein).

    Args:
        flagged_terms: List of (query_id, term, result) tuples
        model: Claude model to use
        api_key: Anthropic API key

    Returns:
        Set of (query_id, go_id) tuples that should be rescued (kept)
    """
    # Layer 1: Check for anthropic package
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "anthropic package not installed, skipping LLM GO validation. "
            "Install with: pip install vhold[llm]"
        )
        return set()

    # Layer 2: Check for API key
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning(
            "ANTHROPIC_API_KEY not set, skipping LLM GO validation"
        )
        return set()

    # Group flagged terms by protein
    by_protein: dict[str, list[tuple[TransferredGOTerm, GOTransferResult]]] = (
        defaultdict(list)
    )
    for qid, term, result in flagged_terms:
        by_protein[qid].append((term, result))

    logger.info(
        f"LLM GO validation: {len(flagged_terms)} terms across "
        f"{len(by_protein)} proteins"
    )

    client = anthropic.Anthropic(api_key=key)
    rescued: set[tuple[str, str]] = set()

    for qid, terms_and_results in by_protein.items():
        # Build term list for prompt
        term_lines = []
        for i, (term, _) in enumerate(terms_and_results, 1):
            term_lines.append(
                f"{i}. {term.go_id} - {term.go_term} ({term.aspect})"
            )

        # Use first term's result for context
        result = terms_and_results[0][1]
        first_term = terms_and_results[0][0]

        user_prompt = GO_FILTER_USER_TEMPLATE.format(
            protein_id=qid,
            donor_organism=first_term.donor_organism,
            donor_accession=first_term.donor_accession,
            similarity=first_term.donor_similarity,
            term_list="\n".join(term_lines),
        )

        # Layer 3: Per-protein error handling
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=GO_FILTER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text = response.content[0].text.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines).strip()

            verdicts = json.loads(text)

            for verdict in verdicts:
                go_id = verdict.get("go_id", "")
                decision = verdict.get("verdict", "reject")
                reason = verdict.get("reason", "")

                if decision == "keep":
                    rescued.add((qid, go_id))
                    logger.info(
                        f"  LLM rescued {qid} {go_id}: {reason}"
                    )
                else:
                    logger.debug(
                        f"  LLM rejected {qid} {go_id}: {reason}"
                    )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"  {qid}: Failed to parse LLM response: {e}")
        except Exception as e:
            logger.warning(f"  {qid}: LLM error, skipping: {e}")

    return rescued
