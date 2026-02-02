"""Tests for metagenomic integration for dark matter characterization."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from vhold.databases.metagenomics import (
    SerratusClient,
    SerratusMatch,
    MetagenomicContext,
    DarkMatterCharacterization,
    characterize_dark_matter_batch,
    infer_viral_families_from_description,
    generate_dark_matter_recommendations,
    FUNCTION_TO_VIRAL_FAMILIES,
)
from vhold.results.dark_matter import DarkMatterProtein


class TestInferViralFamilies:
    """Tests for inferring viral families from protein descriptions."""

    def test_rdrp_keywords(self):
        """Test RdRp-related descriptions."""
        families = infer_viral_families_from_description("RdRp domain protein")
        assert "Coronaviridae" in families
        assert "Flaviviridae" in families

    def test_polymerase_keywords(self):
        """Test polymerase-related descriptions."""
        families = infer_viral_families_from_description("RNA-dependent RNA polymerase")
        assert "Coronaviridae" in families
        assert "Orthomyxoviridae" in families

    def test_capsid_keywords(self):
        """Test capsid-related descriptions."""
        families = infer_viral_families_from_description("Major capsid protein")
        assert "Picornaviridae" in families
        assert "Caliciviridae" in families

    def test_protease_keywords(self):
        """Test protease-related descriptions."""
        families = infer_viral_families_from_description("3C-like protease")
        assert "Coronaviridae" in families

    def test_terminase_keywords(self):
        """Test terminase (phage) descriptions."""
        families = infer_viral_families_from_description("Terminase large subunit")
        assert "Caudovirales" in families

    def test_no_match(self):
        """Test description with no matching keywords."""
        families = infer_viral_families_from_description("hypothetical protein")
        assert len(families) == 0

    def test_multiple_keywords(self):
        """Test description matching multiple keywords."""
        families = infer_viral_families_from_description(
            "Polymerase and helicase domain protein"
        )
        # Should match both polymerase and helicase families
        assert len(families) > 0


class TestSerratusMatch:
    """Tests for SerratusMatch dataclass."""

    def test_creation(self):
        """Test creating a SerratusMatch."""
        match = SerratusMatch(
            run_id="SRR1234567",
            family_name="Coronaviridae",
            percent_identity=95.5,
            score=120.0,
            coverage=0.98,
            n_reads=1000,
            aligned_length=500,
        )
        assert match.run_id == "SRR1234567"
        assert match.family_name == "Coronaviridae"
        assert match.percent_identity == 95.5

    def test_optional_fields(self):
        """Test optional metadata fields."""
        match = SerratusMatch(
            run_id="SRR1234567",
            family_name="Coronaviridae",
            percent_identity=95.5,
            score=120.0,
            coverage=0.98,
            n_reads=1000,
            aligned_length=500,
            biosample="SAMN12345",
            geo_location="USA",
        )
        assert match.biosample == "SAMN12345"
        assert match.geo_location == "USA"


class TestMetagenomicContext:
    """Tests for MetagenomicContext dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
        )
        assert context.serratus_searched is False
        assert context.serratus_total_runs == 0
        assert context.is_conserved_in_metagenomes is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
            serratus_searched=True,
            serratus_total_runs=50,
            is_conserved_in_metagenomes=True,
            conservation_score=0.5,
        )
        d = context.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["serratus"]["searched"] is True
        assert d["serratus"]["total_runs"] == 50
        assert d["conservation"]["is_conserved"] is True


class TestDarkMatterCharacterization:
    """Tests for DarkMatterCharacterization dataclass."""

    def test_creation(self):
        """Test creating a characterization."""
        char = DarkMatterCharacterization(
            query_id="protein_1",
            query_length=300,
            dark_matter_category="no_hits",
            original_reason="No structural homologs detected",
        )
        assert char.query_id == "protein_1"
        assert char.dark_matter_category == "no_hits"
        assert char.priority_score == 0.0

    def test_calculate_priority_no_hits(self):
        """Test priority calculation for no_hits category."""
        char = DarkMatterCharacterization(
            query_id="protein_1",
            query_length=300,
            dark_matter_category="no_hits",
            original_reason="No structural homologs detected",
        )
        char.calculate_priority()
        assert char.priority_score >= 0.4  # no_hits gets 0.4 base score

    def test_calculate_priority_conserved(self):
        """Test priority boost for conserved proteins."""
        char = DarkMatterCharacterization(
            query_id="protein_1",
            query_length=300,
            dark_matter_category="unknown_function",
            original_reason="Unknown function",
        )
        char.metagenomic_context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
            is_conserved_in_metagenomes=True,
        )
        char.calculate_priority()
        assert char.priority_score >= 0.3  # conserved gets 0.3 boost
        assert "Conserved across metagenomes" in char.priority_reasons

    def test_calculate_priority_optimal_length(self):
        """Test priority boost for optimal-length proteins."""
        char = DarkMatterCharacterization(
            query_id="protein_1",
            query_length=300,  # Optimal length (100-500)
            dark_matter_category="weak_hits",
            original_reason="Weak hits",
        )
        char.calculate_priority()
        assert char.priority_score >= 0.1  # length gets 0.1 boost

    def test_to_dict(self):
        """Test conversion to dictionary."""
        char = DarkMatterCharacterization(
            query_id="protein_1",
            query_length=300,
            dark_matter_category="no_hits",
            original_reason="No homologs",
        )
        char.calculate_priority()
        d = char.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["dark_matter_category"] == "no_hits"
        assert "priority" in d


class TestGenerateRecommendations:
    """Tests for generating recommendations."""

    def test_conserved_recommendation(self):
        """Test recommendation for conserved proteins."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
            is_conserved_in_metagenomes=True,
        )
        recs = generate_dark_matter_recommendations(context)
        assert any("Conserved in metagenomes" in r for r in recs)

    def test_serratus_hits_recommendation(self):
        """Test recommendation when Serratus has many hits."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
            serratus_total_runs=150,
        )
        recs = generate_dark_matter_recommendations(context)
        assert any("150 SRA datasets" in r for r in recs)

    def test_no_structural_hits_recommendation(self):
        """Test recommendation for proteins without structural homologs."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=300,
        )
        recs = generate_dark_matter_recommendations(
            context, has_structural_hits=False
        )
        assert any("AlphaFold prediction" in r for r in recs)
        assert any("Novel fold" in r for r in recs)

    def test_short_protein_recommendation(self):
        """Test recommendation for short proteins."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=50,  # Short protein
        )
        recs = generate_dark_matter_recommendations(context)
        assert any("Short protein" in r for r in recs)

    def test_large_protein_recommendation(self):
        """Test recommendation for large proteins."""
        context = MetagenomicContext(
            query_id="protein_1",
            query_length=1500,  # Large protein
        )
        recs = generate_dark_matter_recommendations(context)
        assert any("Large protein" in r for r in recs)
        assert any("InterProScan" in r for r in recs)


class TestSerratusClient:
    """Tests for SerratusClient."""

    def test_initialization(self):
        """Test client initialization."""
        client = SerratusClient()
        assert client.base_url == "https://api.serratus.io"
        assert client.timeout == 30

    @patch("vhold.databases.metagenomics.requests.Session")
    def test_search_by_family_success(self, mock_session_class):
        """Test successful family search."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "run_id": "SRR123",
                    "percent_identity": 90.0,
                    "score": 100.0,
                    "coverage": 0.95,
                    "n_reads": 500,
                    "aligned_length": 300,
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        client = SerratusClient()
        client._session = mock_session
        matches = client.search_by_family("Coronaviridae")

        assert len(matches) == 1
        assert matches[0].run_id == "SRR123"
        assert matches[0].percent_identity == 90.0

    @patch("vhold.databases.metagenomics.requests.Session")
    def test_search_by_family_error(self, mock_session_class):
        """Test handling of API errors."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        import requests
        mock_session.get.side_effect = requests.RequestException("API error")

        client = SerratusClient()
        client._session = mock_session
        matches = client.search_by_family("Coronaviridae")

        assert matches == []


class TestCharacterizeDarkMatterBatch:
    """Tests for batch characterization."""

    def test_empty_batch(self):
        """Test with empty protein list."""
        results = characterize_dark_matter_batch(
            [], search_serratus=False
        )
        assert results == []

    def test_single_protein_no_search(self):
        """Test characterizing single protein without API calls."""
        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",
                reason="No structural homologs detected",
            )
        ]
        results = characterize_dark_matter_batch(
            proteins, search_serratus=False
        )
        assert len(results) == 1
        assert results[0].query_id == "protein_1"
        assert results[0].dark_matter_category == "no_hits"
        assert results[0].priority_score > 0  # Should have some priority

    def test_multiple_proteins(self):
        """Test characterizing multiple proteins."""
        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",
                reason="No homologs",
            ),
            DarkMatterProtein(
                query_id="protein_2",
                query_length=150,
                category="unknown_function",
                reason="Unknown function",
                description="Major capsid protein",
            ),
        ]
        results = characterize_dark_matter_batch(
            proteins, search_serratus=False
        )
        assert len(results) == 2


class TestFunctionToViralFamiliesMapping:
    """Tests for the keyword to viral family mapping."""

    def test_mapping_coverage(self):
        """Test that important keywords are mapped."""
        assert "rdrp" in FUNCTION_TO_VIRAL_FAMILIES
        assert "capsid" in FUNCTION_TO_VIRAL_FAMILIES
        assert "polymerase" in FUNCTION_TO_VIRAL_FAMILIES
        assert "terminase" in FUNCTION_TO_VIRAL_FAMILIES

    def test_viral_families_are_lists(self):
        """Test that all values are lists."""
        for keyword, families in FUNCTION_TO_VIRAL_FAMILIES.items():
            assert isinstance(families, list), f"{keyword} should map to a list"
            assert len(families) > 0, f"{keyword} should have at least one family"
