"""Tests for the functional category classification module."""

import pytest

from vhold.results.categories import (
    classify_protein,
    FUNCTIONAL_CATEGORIES,
    UNKNOWN_TERMS,
)


class TestFunctionalCategories:
    """Tests for FUNCTIONAL_CATEGORIES constant."""

    def test_categories_dict_exists(self):
        """FUNCTIONAL_CATEGORIES should be a non-empty dict."""
        assert isinstance(FUNCTIONAL_CATEGORIES, dict)
        assert len(FUNCTIONAL_CATEGORIES) > 0

    def test_expected_categories_present(self):
        """All expected categories should be defined."""
        expected = [
            "structural",
            "replication",
            "protease",
            "nuclease",
            "packaging",
            "regulatory",
            "movement",
            "lysis",
        ]
        for cat in expected:
            assert cat in FUNCTIONAL_CATEGORIES, f"Missing category: {cat}"

    def test_each_category_has_keywords(self):
        """Each category should have a non-empty list of keywords."""
        for cat, keywords in FUNCTIONAL_CATEGORIES.items():
            assert isinstance(keywords, list), f"{cat} keywords should be a list"
            assert len(keywords) > 0, f"{cat} should have at least one keyword"

    def test_keywords_are_lowercase(self):
        """All keywords should be lowercase for case-insensitive matching."""
        for cat, keywords in FUNCTIONAL_CATEGORIES.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {cat} should be lowercase"


class TestUnknownTerms:
    """Tests for UNKNOWN_TERMS constant."""

    def test_unknown_terms_exists(self):
        """UNKNOWN_TERMS should be a non-empty list."""
        assert isinstance(UNKNOWN_TERMS, list)
        assert len(UNKNOWN_TERMS) > 0

    def test_expected_unknown_terms(self):
        """Common unknown indicators should be present."""
        expected = ["hypothetical", "uncharacterized", "unknown"]
        for term in expected:
            assert term in UNKNOWN_TERMS, f"Missing unknown term: {term}"


class TestClassifyProteinStructural:
    """Tests for structural protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("major capsid protein", "structural"),
        ("Capsid protein VP1", "structural"),
        ("coat protein", "structural"),
        ("envelope glycoprotein", "structural"),
        ("spike protein S", "structural"),
        ("matrix protein M", "structural"),
        ("tail fiber protein", "structural"),
        ("portal protein", "structural"),
        ("shell protein", "structural"),
        ("head protein", "structural"),
        ("virion protein", "structural"),
        ("glycoprotein gp120", "structural"),
        ("membrane protein", "structural"),
        ("nucleocapsid protein N", "structural"),
        ("tegument protein", "structural"),
        ("baseplate protein", "structural"),
    ])
    def test_structural_keywords(self, description, expected):
        """Structural keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinReplication:
    """Tests for replication protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("RNA-dependent RNA polymerase", "replication"),
        ("DNA polymerase", "replication"),
        ("replicase", "replication"),
        ("helicase nsp13", "replication"),
        ("primase", "replication"),
        ("RdRp", "replication"),
        ("reverse transcriptase", "replication"),
        ("rep protein", "replication"),
        ("nsp12", "replication"),
        ("nsp13 helicase", "replication"),
    ])
    def test_replication_keywords(self, description, expected):
        """Replication keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinProtease:
    """Tests for protease protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("main protease", "protease"),
        ("3C-like protease", "protease"),
        ("peptidase", "protease"),
        ("maturase", "protease"),
        ("3CL protease", "protease"),
        ("Mpro", "protease"),
        ("nsp5 protease", "protease"),
    ])
    def test_protease_keywords(self, description, expected):
        """Protease keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinNuclease:
    """Tests for nuclease protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("endonuclease", "nuclease"),
        ("exonuclease", "nuclease"),
        ("nuclease", "nuclease"),
        ("integrase", "nuclease"),
        ("recombinase", "nuclease"),
        ("DNA ligase", "nuclease"),
        ("RNase", "nuclease"),
        ("DNase", "nuclease"),
    ])
    def test_nuclease_keywords(self, description, expected):
        """Nuclease keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinPackaging:
    """Tests for packaging protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("terminase large subunit", "packaging"),
        ("packaging protein", "packaging"),
        ("scaffold protein", "packaging"),
        ("portal protein", "structural"),  # portal is in structural first
    ])
    def test_packaging_keywords(self, description, expected):
        """Packaging keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinRegulatory:
    """Tests for regulatory protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("transcription factor", "regulatory"),
        ("repressor protein", "regulatory"),
        ("transcriptional activator", "regulatory"),
        ("gene regulator", "regulatory"),
        ("anti-repressor", "regulatory"),
        ("antirepressor protein", "regulatory"),
    ])
    def test_regulatory_keywords(self, description, expected):
        """Regulatory keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinMovement:
    """Tests for movement protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("movement protein", "movement"),
        ("cell-to-cell movement", "movement"),
        ("transport protein", "movement"),
    ])
    def test_movement_keywords(self, description, expected):
        """Movement keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinLysis:
    """Tests for lysis protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("holin", "lysis"),
        ("endolysin", "lysis"),
        ("lysin", "lysis"),
        ("spanin", "lysis"),
        ("lysis protein", "lysis"),
    ])
    def test_lysis_keywords(self, description, expected):
        """Lysis keywords should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinUnknown:
    """Tests for unknown protein classification."""

    @pytest.mark.parametrize("description,expected", [
        ("hypothetical protein", "unknown"),
        ("uncharacterized protein", "unknown"),
        ("unknown function", "unknown"),
        ("DUF1234 domain", "unknown"),
        ("UniProt:A0A123", "unknown"),
        ("", "unknown"),
        ("protein", "unknown"),
    ])
    def test_unknown_keywords(self, description, expected):
        """Unknown indicators should classify correctly."""
        assert classify_protein(description) == expected


class TestClassifyProteinCaseInsensitive:
    """Tests for case-insensitive matching."""

    def test_uppercase_description(self):
        """Classification should work with uppercase."""
        assert classify_protein("MAJOR CAPSID PROTEIN") == "structural"

    def test_mixed_case_description(self):
        """Classification should work with mixed case."""
        assert classify_protein("Major Capsid Protein") == "structural"

    def test_lowercase_description(self):
        """Classification should work with lowercase."""
        assert classify_protein("major capsid protein") == "structural"


class TestClassifyProteinWithGene:
    """Tests for classification using gene name."""

    def test_gene_contributes_to_classification(self):
        """Gene name should contribute to classification."""
        # Description alone is vague, but gene name helps
        result = classify_protein("viral protein", gene="capsid")
        assert result == "structural"

    def test_gene_none_handled(self):
        """None gene should be handled gracefully."""
        result = classify_protein("major capsid protein", gene=None)
        assert result == "structural"

    def test_gene_empty_string(self):
        """Empty gene string should be handled."""
        result = classify_protein("major capsid protein", gene="")
        assert result == "structural"


class TestClassifyProteinPriority:
    """Tests for category priority when multiple keywords match."""

    def test_first_match_wins(self):
        """First matching category in iteration order wins."""
        # "portal" appears in both structural and packaging
        # structural comes first in dict, so should win
        result = classify_protein("portal protein")
        assert result == "structural"

    def test_structural_before_packaging_for_portal(self):
        """Portal should be classified as structural (appears first)."""
        assert classify_protein("portal vertex protein") == "structural"


class TestClassifyProteinEdgeCases:
    """Tests for edge cases."""

    def test_empty_description(self):
        """Empty description should return unknown."""
        assert classify_protein("") == "unknown"

    def test_whitespace_only(self):
        """Whitespace-only description should return unknown."""
        assert classify_protein("   ") == "unknown"

    def test_special_characters(self):
        """Description with special chars should work."""
        assert classify_protein("capsid (major)") == "structural"

    def test_numbers_in_description(self):
        """Description with numbers should work."""
        assert classify_protein("capsid protein VP1") == "structural"

    def test_keyword_as_substring(self):
        """Keywords should match as substrings."""
        # "polymerase" is in "RNA-dependent RNA polymerase"
        assert classify_protein("RNA-dependent RNA polymerase") == "replication"

    def test_very_long_description(self):
        """Very long descriptions should work."""
        long_desc = "putative " * 100 + "capsid protein"
        assert classify_protein(long_desc) == "structural"
