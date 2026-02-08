"""Tests for LLM-based functional classification."""

import json
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from vhold.results.llm_classify import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    classify_single,
    llm_reclassify,
)


# ============================================================================
# Test fixtures - minimal ConsensusResult-like objects for testing
# ============================================================================


@dataclass
class MockConsensusResult:
    """Minimal mock of ConsensusResult for testing."""

    query_id: str
    query_length: int = 300
    functional_category: str = "unknown"
    classification_source: str = "keywords"
    primary_annotation: dict = field(default_factory=dict)
    _is_annotated: bool = True

    @property
    def is_annotated(self) -> bool:
        return self._is_annotated

    @property
    def description(self) -> str:
        return self.primary_annotation.get("description", "hypothetical protein")


def _make_mock_response(category: str, reasoning: str) -> MagicMock:
    """Create a mock Anthropic API response."""
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps({"category": category, "reasoning": reasoning})
    mock_response.content = [mock_content]
    return mock_response


def _make_mock_client(responses: list[tuple[str, str]] | None = None) -> MagicMock:
    """Create a mock Anthropic client.

    Args:
        responses: List of (category, reasoning) tuples. If None, returns structural.
    """
    client = MagicMock()
    if responses is None:
        responses = [("structural", "test reasoning")]

    side_effects = [_make_mock_response(cat, reason) for cat, reason in responses]
    client.messages.create.side_effect = side_effects
    return client


# ============================================================================
# Tests for classify_single
# ============================================================================


class TestClassifySingle:
    """Tests for the classify_single function."""

    def test_classify_fusion_protein_as_structural(self):
        """Fusion protein should be classified as structural."""
        client = _make_mock_client([("structural", "Fusion proteins are structural")])
        category, reasoning = classify_single(
            client=client,
            description="fusion protein",
            gene="F",
            organism="Nipah",
        )
        assert category == "structural"
        assert "structural" in reasoning.lower()

    def test_classify_v_protein_as_host_interaction(self):
        """V protein should be classified as host_interaction."""
        client = _make_mock_client([("host_interaction", "V protein is interferon antagonist")])
        category, reasoning = classify_single(
            client=client,
            description="V protein",
            gene="V",
            organism="Nipah",
        )
        assert category == "host_interaction"

    def test_classify_with_defaults(self):
        """Should work with minimal arguments."""
        client = _make_mock_client([("replication", "polymerase cofactor")])
        category, reasoning = classify_single(
            client=client,
            description="phosphoprotein",
        )
        assert category == "replication"

    def test_invalid_category_returns_unknown(self):
        """LLM returning invalid category should fall back to unknown."""
        client = _make_mock_client([("not_a_real_category", "bad response")])
        category, reasoning = classify_single(
            client=client,
            description="some protein",
        )
        assert category == "unknown"
        assert "invalid" in reasoning.lower()

    def test_malformed_json_returns_unknown(self):
        """Non-JSON response should fall back to unknown."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "This is not JSON at all"
        mock_response.content = [mock_content]
        client.messages.create.return_value = mock_response

        category, reasoning = classify_single(
            client=client,
            description="some protein",
        )
        assert category == "unknown"
        assert "parse error" in reasoning.lower()

    def test_markdown_code_block_stripped(self):
        """JSON wrapped in markdown code blocks should be parsed correctly."""
        client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '```json\n{"category": "structural", "reasoning": "test"}\n```'
        mock_response.content = [mock_content]
        client.messages.create.return_value = mock_response

        category, reasoning = classify_single(
            client=client,
            description="capsid protein",
        )
        assert category == "structural"

    def test_api_called_with_correct_params(self):
        """Verify the API is called with correct model and messages."""
        client = _make_mock_client([("structural", "test")])
        classify_single(
            client=client,
            description="test protein",
            gene="X",
            organism="TestVirus",
            model="claude-haiku-4-5-20251001",
        )

        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs.kwargs["max_tokens"] == 256
        assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT
        assert len(call_kwargs.kwargs["messages"]) == 1
        assert "test protein" in call_kwargs.kwargs["messages"][0]["content"]
        assert "TestVirus" in call_kwargs.kwargs["messages"][0]["content"]


# ============================================================================
# Tests for llm_reclassify
# ============================================================================


class TestLlmReclassify:
    """Tests for the llm_reclassify function."""

    def test_targets_unknown_keywords_only(self):
        """Should only reclassify proteins with unknown + keywords classification."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "V protein"},
            ),
            "prot2": MockConsensusResult(
                query_id="prot2",
                functional_category="structural",
                classification_source="keywords",
                primary_annotation={"description": "capsid protein"},
            ),
            "prot3": MockConsensusResult(
                query_id="prot3",
                functional_category="unknown",
                classification_source="pfam:some_domain",
                primary_annotation={"description": "hypothetical"},
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = _make_mock_client([("host_interaction", "interferon antagonist")])
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = llm_reclassify(annotations, api_key="test-key")

        # Only prot1 should be reclassified (unknown + keywords)
        assert result["prot1"].functional_category == "host_interaction"
        assert result["prot1"].classification_source == f"llm:{DEFAULT_MODEL}"
        # prot2 stays (structural, not unknown)
        assert result["prot2"].functional_category == "structural"
        assert result["prot2"].classification_source == "keywords"
        # prot3 stays (pfam classified, not keywords)
        assert result["prot3"].functional_category == "unknown"
        assert result["prot3"].classification_source == "pfam:some_domain"

    def test_skips_unannotated_proteins(self):
        """Should skip proteins without any annotation."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="unknown",
                classification_source="keywords",
                _is_annotated=False,
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = _make_mock_client()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = llm_reclassify(annotations, api_key="test-key")

        # Should not call API at all
        mock_client.messages.create.assert_not_called()
        assert result["prot1"].functional_category == "unknown"

    def test_reclassify_all_keywords_mode(self):
        """With reclassify_all_keywords=True, should target all keyword-classified proteins."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="structural",
                classification_source="keywords",
                primary_annotation={"description": "capsid protein"},
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = _make_mock_client([("structural", "confirmed structural")])
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = llm_reclassify(
                annotations, api_key="test-key", reclassify_all_keywords=True
            )

        # Should have called the API even though it was already structural
        mock_client.messages.create.assert_called_once()

    def test_extracts_organism_from_query_id(self):
        """Should extract organism from pipe-delimited query IDs."""
        annotations = {
            "NP_112023.1|Nipah": MockConsensusResult(
                query_id="NP_112023.1|Nipah",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "V protein"},
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = _make_mock_client([("host_interaction", "test")])
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            llm_reclassify(annotations, api_key="test-key")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        assert "Nipah" in user_msg

    def test_no_candidates_skips_api(self):
        """If no proteins need reclassification, should not create client."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="structural",
                classification_source="pfam:capsid",
            ),
        }

        mock_anthropic = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = llm_reclassify(annotations, api_key="test-key")

        # Should not even create a client
        mock_anthropic.Anthropic.assert_not_called()


# ============================================================================
# Tests for graceful degradation
# ============================================================================


class TestGracefulDegradation:
    """Tests for graceful error handling."""

    def test_no_anthropic_package(self):
        """Should return unchanged annotations if anthropic is not installed."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "V protein"},
            ),
        }

        with patch.dict("sys.modules", {"anthropic": None}):
            result = llm_reclassify(annotations)

        assert result["prot1"].functional_category == "unknown"

    def test_no_api_key(self):
        """Should return unchanged annotations if no API key."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "V protein"},
            ),
        }

        mock_anthropic = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with patch.dict("os.environ", {}, clear=True):
                result = llm_reclassify(annotations, api_key=None)

        assert result["prot1"].functional_category == "unknown"

    def test_api_error_skips_protein(self):
        """API error on one protein should skip it and continue."""
        annotations = {
            "prot1": MockConsensusResult(
                query_id="prot1",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "V protein"},
            ),
            "prot2": MockConsensusResult(
                query_id="prot2",
                functional_category="unknown",
                classification_source="keywords",
                primary_annotation={"description": "fusion protein"},
            ),
        }

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        # First call raises, second succeeds
        error_response = Exception("API rate limit exceeded")
        success_response = _make_mock_response("structural", "fusion is structural")
        mock_client.messages.create.side_effect = [error_response, success_response]
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = llm_reclassify(annotations, api_key="test-key")

        # prot1 should be unchanged (error), prot2 should be reclassified
        assert result["prot1"].functional_category == "unknown"
        assert result["prot2"].functional_category == "structural"
