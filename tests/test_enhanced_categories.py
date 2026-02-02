"""Tests for enhanced functional category classification."""

import pytest
from vhold.results.categories import (
    classify_protein,
    classify_by_pfam,
    classify_by_go_bp,
    classify_by_go_mf,
    classify_by_superfamily,
    classify_by_keywords,
    get_classification_source,
    AnnotationEvidence,
    ALL_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
)


class TestClassifyByPfam:
    """Tests for Pfam-based classification."""

    def test_structural_proteins(self):
        """Test classification of structural Pfam domains."""
        assert classify_by_pfam("Major capsid protein") == "structural"
        assert classify_by_pfam("Picornavirus capsid protein") == "structural"
        assert classify_by_pfam("Viral glycoprotein") == "structural"
        assert classify_by_pfam("Nucleocapsid") == "structural"
        assert classify_by_pfam("Herpesvirus VP23 like capsid protein") == "structural"

    def test_replication_proteins(self):
        """Test classification of replication Pfam domains."""
        assert classify_by_pfam("RNA-dependent RNA polymerase") == "replication"
        assert classify_by_pfam("DNA helicase") == "replication"
        assert classify_by_pfam("DNA primase") == "replication"
        assert classify_by_pfam("Thymidine kinase from herpesvirus") == "replication"
        assert classify_by_pfam("Parvovirus non-structural protein NS1") == "replication"

    def test_protease_proteins(self):
        """Test classification of protease Pfam domains."""
        assert classify_by_pfam("Assemblin (Peptidase family S21)") == "protease"
        assert classify_by_pfam("Cysteine proteinases") == "protease"
        assert classify_by_pfam("OTU-like cysteine protease") == "protease"

    def test_nuclease_proteins(self):
        """Test classification of nuclease Pfam domains."""
        assert classify_by_pfam("DNA/RNA non-specific endonuclease") == "nuclease"
        assert classify_by_pfam("PD-(D/E)XK nuclease superfamily") == "nuclease"
        assert classify_by_pfam("GIY-YIG catalytic domain") == "nuclease"

    def test_no_match(self):
        """Test that unknown Pfam domains return None."""
        assert classify_by_pfam("Unknown protein family") is None
        assert classify_by_pfam("Hypothetical") is None
        assert classify_by_pfam("") is None


class TestClassifyByGOBP:
    """Tests for GO Biological Process classification."""

    def test_structural_processes(self):
        """Test classification of structural GO BP terms."""
        assert classify_by_go_bp("viral capsid assembly") == "structural"
        assert classify_by_go_bp("virion assembly") == "structural"

    def test_replication_processes(self):
        """Test classification of replication GO BP terms."""
        assert classify_by_go_bp("DNA replication") == "replication"
        assert classify_by_go_bp("viral genome replication") == "replication"
        assert classify_by_go_bp("TMP biosynthetic process") == "replication"

    def test_protease_processes(self):
        """Test classification of protease GO BP terms."""
        assert classify_by_go_bp("proteolysis") == "protease"
        assert classify_by_go_bp("viral protein processing") == "protease"

    def test_packaging_processes(self):
        """Test classification of packaging GO BP terms."""
        assert classify_by_go_bp("viral DNA genome packaging") == "packaging"
        assert classify_by_go_bp("chromosome organization") == "packaging"

    def test_no_match(self):
        """Test that unknown GO BP terms return None."""
        assert classify_by_go_bp("unknown process") is None
        assert classify_by_go_bp("") is None


class TestClassifyByGOMF:
    """Tests for GO Molecular Function classification."""

    def test_structural_functions(self):
        """Test classification of structural GO MF terms."""
        assert classify_by_go_mf("structural molecule activity") == "structural"
        assert classify_by_go_mf("structural constituent of virion") == "structural"

    def test_replication_functions(self):
        """Test classification of replication GO MF terms."""
        assert classify_by_go_mf("DNA helicase activity") == "replication"
        assert classify_by_go_mf("RNA-dependent RNA polymerase activity") == "replication"

    def test_protease_functions(self):
        """Test classification of protease GO MF terms."""
        assert classify_by_go_mf("serine-type endopeptidase activity") == "protease"
        assert classify_by_go_mf("cysteine-type peptidase activity") == "protease"


class TestClassifyBySuperfamily:
    """Tests for SUPERFAMILY-based classification."""

    def test_structural_superfamilies(self):
        """Test classification of structural SUPERFAMILY annotations."""
        assert classify_by_superfamily("Viral glycoprotein") == "structural"
        assert classify_by_superfamily("VSV matrix protein") == "structural"

    def test_replication_superfamilies(self):
        """Test classification of replication SUPERFAMILY annotations."""
        assert classify_by_superfamily("DNA/RNA polymerases") == "replication"
        assert classify_by_superfamily("P-loop containing nucleoside triphosphate hydrolases") == "replication"


class TestAnnotationEvidence:
    """Tests for AnnotationEvidence dataclass."""

    def test_has_evidence_with_pfam(self):
        """Test has_evidence with Pfam annotation."""
        evidence = AnnotationEvidence(pfam="Major capsid protein")
        assert evidence.has_evidence() is True

    def test_has_evidence_with_go(self):
        """Test has_evidence with GO annotation."""
        evidence = AnnotationEvidence(go_bp="viral genome replication")
        assert evidence.has_evidence() is True

    def test_has_evidence_empty(self):
        """Test has_evidence with no annotations."""
        evidence = AnnotationEvidence()
        assert evidence.has_evidence() is False


class TestClassifyProtein:
    """Tests for the main classify_protein function."""

    def test_with_pfam_evidence(self):
        """Test classification with Pfam evidence."""
        evidence = AnnotationEvidence(pfam="Major capsid protein")
        category = classify_protein("viral protein", evidence=evidence)
        assert category == "structural"

    def test_with_go_bp_evidence(self):
        """Test classification with GO BP evidence."""
        evidence = AnnotationEvidence(go_bp="viral genome replication")
        category = classify_protein("viral protein", evidence=evidence)
        assert category == "replication"

    def test_with_superfamily_evidence(self):
        """Test classification with SUPERFAMILY evidence."""
        evidence = AnnotationEvidence(superfamily="Cysteine proteinases")
        category = classify_protein("viral protein", evidence=evidence)
        assert category == "protease"

    def test_pfam_priority_over_go(self):
        """Test that Pfam takes priority over GO terms."""
        # Pfam says structural, GO says replication
        evidence = AnnotationEvidence(
            pfam="capsid protein",
            go_bp="DNA replication",
        )
        category = classify_protein("viral protein", evidence=evidence)
        assert category == "structural"  # Pfam wins

    def test_keyword_fallback(self):
        """Test keyword-based fallback when no evidence."""
        category = classify_protein("major capsid protein", evidence=None)
        assert category == "structural"

    def test_keyword_with_gene(self):
        """Test keyword matching with gene name."""
        category = classify_protein("hypothetical protein", gene="gp23")
        assert category == "unknown"  # gp23 doesn't match any keywords


class TestGetClassificationSource:
    """Tests for get_classification_source function."""

    def test_returns_source_with_pfam(self):
        """Test that source is returned correctly for Pfam."""
        evidence = AnnotationEvidence(pfam="Major capsid protein")
        category, source = get_classification_source("viral protein", evidence=evidence)
        assert category == "structural"
        assert "pfam:" in source

    def test_returns_keywords_source(self):
        """Test that keywords source is returned when no evidence."""
        category, source = get_classification_source("capsid protein")
        assert category == "structural"
        assert source == "keywords"


class TestCategoryConstants:
    """Tests for category constants."""

    def test_all_categories_defined(self):
        """Test that all expected categories are defined."""
        expected = [
            "structural", "replication", "protease", "nuclease",
            "packaging", "regulatory", "movement", "lysis",
            "host_interaction", "entry", "unknown"
        ]
        for cat in expected:
            assert cat in ALL_CATEGORIES

    def test_all_categories_have_descriptions(self):
        """Test that all categories have descriptions."""
        for cat in ALL_CATEGORIES:
            assert cat in CATEGORY_DESCRIPTIONS
            assert len(CATEGORY_DESCRIPTIONS[cat]) > 0
