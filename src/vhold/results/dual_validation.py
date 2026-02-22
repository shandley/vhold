"""Dual validation: combine GO transfer and Foldseek structural evidence.

When both GO term transfer (functional annotation via Swiss-Prot embedding
similarity) and Foldseek (structural search via BFVD/Viro3D) provide
annotations for the same protein, their agreement is assessed:

- dual_agree: Both methods assign the same functional category.
  Confidence boosted 1.3x — two orthogonal signals agree.
- dual_disagree: Methods disagree. GO transfer is preferred as it
  originates from experimentally validated GO annotations.
- go_only: GO transfer succeeded but Foldseek had no informative hit.
- foldseek_only: No GO transfer, Foldseek annotated (current behavior).

Cross-domain discovery is flagged when the best GO donor is non-viral,
indicating that a viral protein shares structural/functional similarity
with a well-characterized protein from another domain of life.
"""

from __future__ import annotations

from dataclasses import dataclass

from vhold.features.go_transfer import GOTransferResult, TransferredGOTerm
from vhold.results.categories import GO_BP_TO_CATEGORY, GO_MF_TO_CATEGORY
from vhold.results.consensus import ConsensusResult
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

# Confidence boost when both GO transfer and Foldseek agree
DUAL_AGREE_BOOST = 1.3


@dataclass
class DualValidationResult:
    """Result of comparing GO transfer and Foldseek evidence."""

    agreement: str  # "dual_agree", "dual_disagree", "go_only", "foldseek_only"
    go_category: str | None  # functional category derived from GO terms
    foldseek_category: str  # category from Foldseek/keywords
    final_category: str  # resolved category
    confidence_boost: float  # multiplier (1.0 = no boost, 1.3 = dual agreement)
    is_cross_domain: bool  # non-viral donor in GO transfer
    cross_domain_details: str | None  # "Matched E. coli lysozyme (P00720, sim=0.82)"


def _go_to_functional_category(
    go_terms: list[TransferredGOTerm],
) -> str | None:
    """Map transferred GO terms to vHold functional categories.

    Uses existing GO_BP_TO_CATEGORY and GO_MF_TO_CATEGORY mappings
    from categories.py. Returns the category with the highest
    Reliability Index among mapped terms.

    Args:
        go_terms: List of transferred GO terms

    Returns:
        Functional category string, or None if no mapping found
    """
    if not go_terms:
        return None

    # Track (category, max_ri) pairs
    category_ri: dict[str, float] = {}

    for term in go_terms:
        term_lower = term.go_term.lower()
        matched_category = None

        if term.aspect == "BP":
            for category, keywords in GO_BP_TO_CATEGORY.items():
                for keyword in keywords:
                    if keyword.lower() in term_lower:
                        matched_category = category
                        break
                if matched_category:
                    break

        elif term.aspect == "MF":
            for category, keywords in GO_MF_TO_CATEGORY.items():
                for keyword in keywords:
                    if keyword.lower() in term_lower:
                        matched_category = category
                        break
                if matched_category:
                    break

        if matched_category:
            current_ri = category_ri.get(matched_category, 0.0)
            category_ri[matched_category] = max(current_ri, term.reliability_index)

    if not category_ri:
        return None

    # Return category with highest RI
    return max(category_ri, key=lambda c: category_ri[c])


def validate_dual_evidence(
    consensus: ConsensusResult,
    go_result: GOTransferResult | None,
) -> DualValidationResult:
    """Compare GO-derived category with Foldseek-derived category.

    Args:
        consensus: ConsensusResult from Foldseek pipeline
        go_result: GOTransferResult from GO transfer, or None

    Returns:
        DualValidationResult with agreement status and resolved category
    """
    foldseek_category = consensus.functional_category

    # No GO transfer result
    if go_result is None or not go_result.has_terms:
        return DualValidationResult(
            agreement="foldseek_only",
            go_category=None,
            foldseek_category=foldseek_category,
            final_category=foldseek_category,
            confidence_boost=1.0,
            is_cross_domain=False,
            cross_domain_details=None,
        )

    # Map GO terms to functional category
    go_category = _go_to_functional_category(go_result.go_terms)

    # Cross-domain details
    is_cross_domain = go_result.is_cross_domain
    cross_domain_details = None
    if is_cross_domain:
        # Find the best cross-domain donor
        for term in go_result.go_terms:
            if term.is_cross_domain:
                cross_domain_details = (
                    f"Matched {term.donor_organism} "
                    f"({term.donor_accession}, sim={term.donor_similarity:.2f})"
                )
                break

    # No category could be derived from GO terms
    if go_category is None:
        # GO terms exist but don't map to any category — use Foldseek
        return DualValidationResult(
            agreement="foldseek_only",
            go_category=None,
            foldseek_category=foldseek_category,
            final_category=foldseek_category,
            confidence_boost=1.0,
            is_cross_domain=is_cross_domain,
            cross_domain_details=cross_domain_details,
        )

    # Foldseek has no meaningful annotation
    if foldseek_category == "unknown":
        return DualValidationResult(
            agreement="go_only",
            go_category=go_category,
            foldseek_category=foldseek_category,
            final_category=go_category,
            confidence_boost=1.0,
            is_cross_domain=is_cross_domain,
            cross_domain_details=cross_domain_details,
        )

    # Both have categories — compare
    if go_category == foldseek_category:
        return DualValidationResult(
            agreement="dual_agree",
            go_category=go_category,
            foldseek_category=foldseek_category,
            final_category=go_category,
            confidence_boost=DUAL_AGREE_BOOST,
            is_cross_domain=is_cross_domain,
            cross_domain_details=cross_domain_details,
        )
    else:
        # Disagree — prefer GO transfer (experimentally validated)
        return DualValidationResult(
            agreement="dual_disagree",
            go_category=go_category,
            foldseek_category=foldseek_category,
            final_category=go_category,
            confidence_boost=1.0,
            is_cross_domain=is_cross_domain,
            cross_domain_details=cross_domain_details,
        )


def merge_go_into_consensus(
    consensus: ConsensusResult,
    go_result: GOTransferResult | None,
    validation: DualValidationResult,
) -> ConsensusResult:
    """Enrich ConsensusResult with GO transfer data.

    Updates the consensus result in place with GO transfer fields,
    adjusts functional category and confidence based on dual validation.

    Args:
        consensus: ConsensusResult to update
        go_result: GOTransferResult (may be None)
        validation: DualValidationResult from validate_dual_evidence

    Returns:
        The updated ConsensusResult (same object, modified in place)
    """
    # Set dual validation status
    consensus.dual_validation = validation.agreement

    # Cross-domain flags
    consensus.is_cross_domain = validation.is_cross_domain
    consensus.cross_domain_organism = ""

    if go_result is not None and go_result.has_terms:
        # GO transfer metadata
        consensus.go_transfer_ri = go_result.max_ri
        consensus.go_transfer_k = len(go_result.top_k_matches)
        consensus.go_transfer_best_accession = go_result.best_accession
        consensus.go_transfer_best_similarity = go_result.best_similarity

        # Collect evidence codes
        evidence_codes = sorted(
            {t.evidence_code for t in go_result.go_terms}
        )
        consensus.go_evidence_codes = "|".join(evidence_codes)

        # Source scope
        consensus.go_transfer_source = "swissprot"

        # Cross-domain organism
        if validation.is_cross_domain and validation.cross_domain_details:
            consensus.cross_domain_organism = validation.cross_domain_details

    # Update functional category based on validation
    if validation.agreement in ("dual_agree", "go_only", "dual_disagree"):
        consensus.functional_category = validation.final_category
        # Update classification source
        if validation.agreement == "dual_agree":
            consensus.classification_source = "go_transfer+foldseek"
        else:
            consensus.classification_source = "go_transfer"

    # Apply confidence boost for dual agreement
    if validation.confidence_boost > 1.0:
        consensus.consensus_score = min(
            1.0, consensus.consensus_score * validation.confidence_boost
        )
        # Potentially upgrade confidence level
        if consensus.consensus_score >= 0.8:
            consensus.confidence_level = "high"
        elif consensus.consensus_score >= 0.5:
            consensus.confidence_level = "medium"

    return consensus
