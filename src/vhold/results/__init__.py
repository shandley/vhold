"""Results processing and output for vhold."""

from vhold.results.categories import classify_protein, FUNCTIONAL_CATEGORIES
from vhold.results.parser import parse_foldseek_results, FoldseekHit
from vhold.results.annotations import transfer_annotations, AnnotatedProtein
from vhold.results.output import write_tsv_output, write_summary_json
from vhold.results.dark_matter import (
    analyze_dark_matter,
    classify_dark_matter,
    DarkMatterProtein,
    DarkMatterReport,
)

__all__ = [
    "classify_protein",
    "FUNCTIONAL_CATEGORIES",
    "parse_foldseek_results",
    "FoldseekHit",
    "transfer_annotations",
    "AnnotatedProtein",
    "write_tsv_output",
    "write_summary_json",
    "analyze_dark_matter",
    "classify_dark_matter",
    "DarkMatterProtein",
    "DarkMatterReport",
]
