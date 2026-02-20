"""Tests for STARLING conformational ensemble similarity search and UniProt lookup."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vhold.features.disorder import DisorderResult, classify_disorder
from vhold.features.starling_search import (
    StarlingMatch,
    StarlingSearchResult,
    _distance_to_similarity,
    build_starling_consensus_results,
    check_starling_search_available,
)
from vhold.features.uniprot_lookup import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    USER_AGENT,
    _parse_uniprot_entry,
    extract_uniprot_accession,
)
from vhold.utils.constants import (
    DEFAULT_STARLING_SIMILARITY_THRESHOLD,
    STARLING_SEARCH_K,
    UNIPROT_BATCH_SIZE,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_disorder_result(
    protein_id: str = "test",
    length: int = 100,
    fraction: float = 0.5,
    regions: list | None = None,
) -> DisorderResult:
    """Create a DisorderResult for testing."""
    scores = np.random.default_rng(42).random(length).astype(np.float32)
    return DisorderResult(
        protein_id=protein_id,
        disorder_scores=scores,
        disorder_fraction=fraction,
        disorder_regions=regions or [(10, 60)],
        disorder_class=classify_disorder(fraction),
        sequence_length=length,
    )


def _make_uniprot_entry(
    accession: str = "Q9BYF1",
    name: str = "Protein kinase C",
    gene: str = "PRKCA",
    organism: str = "Homo sapiens",
) -> dict:
    """Create a mock UniProt JSON entry."""
    return {
        "primaryAccession": accession,
        "proteinDescription": {
            "recommendedName": {
                "fullName": {"value": name},
            },
        },
        "genes": [{"geneName": {"value": gene}}],
        "organism": {"scientificName": organism},
        "uniProtKBCrossReferences": [
            {
                "database": "GO",
                "id": "GO:0006468",
                "properties": [
                    {"key": "GoTerm", "value": "P:protein phosphorylation"},
                ],
            },
            {
                "database": "GO",
                "id": "GO:0004672",
                "properties": [
                    {"key": "GoTerm", "value": "F:protein kinase activity"},
                ],
            },
            {
                "database": "Pfam",
                "id": "PF00069",
                "properties": [
                    {"key": "EntryName", "value": "Pkinase"},
                ],
            },
        ],
        "keywords": [
            {"name": "Kinase"},
            {"name": "Phosphorylation"},
        ],
    }


# ============================================================================
# Constants tests
# ============================================================================


class TestConstants:
    """Test STARLING search constants are defined."""

    def test_similarity_threshold(self):
        assert 0 < DEFAULT_STARLING_SIMILARITY_THRESHOLD <= 1.0

    def test_search_k(self):
        assert STARLING_SEARCH_K > 0

    def test_uniprot_batch_size(self):
        assert UNIPROT_BATCH_SIZE > 0
        assert UNIPROT_BATCH_SIZE <= 500  # UniProt API limit


# ============================================================================
# StarlingMatch tests
# ============================================================================


class TestStarlingMatch:
    """Test StarlingMatch dataclass."""

    def test_create(self):
        m = StarlingMatch(
            query_id="prot1",
            target_id="UniRef50_Q9BYF1",
            distance=0.5,
            similarity=0.875,
            sequence_length=200,
        )
        assert m.query_id == "prot1"
        assert m.target_id == "UniRef50_Q9BYF1"
        assert m.distance == 0.5
        assert m.similarity == 0.875
        assert m.sequence_length == 200

    def test_similarity_is_float(self):
        m = StarlingMatch("q", "t", 0.3, 0.95, 100)
        assert isinstance(m.similarity, float)


class TestStarlingSearchResult:
    """Test StarlingSearchResult dataclass."""

    def test_empty(self):
        r = StarlingSearchResult()
        assert r.matched == {}
        assert r.unmatched == []
        assert r.total_queries == 0

    def test_with_data(self):
        match = StarlingMatch("q1", "t1", 0.3, 0.95, 100)
        r = StarlingSearchResult(
            matched={"q1": [match]},
            unmatched=["q2"],
            total_queries=2,
        )
        assert len(r.matched) == 1
        assert len(r.unmatched) == 1
        assert r.total_queries == 2


# ============================================================================
# Distance to similarity conversion
# ============================================================================


class TestDistanceToSimilarity:
    """Test L2 distance to cosine similarity conversion."""

    def test_zero_distance(self):
        """Zero distance = identical = similarity 1.0."""
        assert _distance_to_similarity(0.0) == 1.0

    def test_max_distance(self):
        """L2 distance of 2.0 for normalized vectors = opposite = similarity 0.0."""
        assert _distance_to_similarity(2.0) == pytest.approx(0.0, abs=1e-6)

    def test_intermediate(self):
        """Intermediate distance gives intermediate similarity."""
        sim = _distance_to_similarity(1.0)
        assert 0.0 < sim < 1.0
        # For d=1.0: sim = 1 - 1/2 = 0.5
        assert sim == pytest.approx(0.5, abs=1e-6)

    def test_clamped_negative(self):
        """Very large distance should clamp to 0."""
        assert _distance_to_similarity(3.0) == 0.0

    def test_small_negative_distance(self):
        """Small negative distance from floating point error stays below 1."""
        sim = _distance_to_similarity(-0.1)
        assert 0.0 <= sim <= 1.0


# ============================================================================
# Availability check
# ============================================================================


class TestCheckAvailability:
    """Test STARLING SearchEngine availability check."""

    def test_available(self):
        mock_module = MagicMock()
        mock_search = MagicMock()
        mock_engine = MagicMock()
        mock_module.search = mock_search
        mock_search.search_engine = mock_engine

        with patch.dict(
            "sys.modules",
            {
                "starling": mock_module,
                "starling.search": mock_search,
                "starling.search.search_engine": mock_engine,
            },
        ):
            mock_engine.SearchEngine = MagicMock()
            assert check_starling_search_available() is True

    def test_not_available(self):
        # Without mocking, starling is not installed
        with patch.dict("sys.modules", {"starling": None}):
            assert check_starling_search_available() is False


# ============================================================================
# UniProt accession extraction
# ============================================================================


class TestExtractUniProtAccession:
    """Test UniRef50 ID to UniProt accession extraction."""

    def test_standard_format(self):
        assert extract_uniprot_accession("UniRef50_Q9BYF1") == "Q9BYF1"

    def test_with_numbers(self):
        assert extract_uniprot_accession("UniRef50_P12345") == "P12345"

    def test_longer_accession(self):
        assert extract_uniprot_accession("UniRef50_A0A1B2C3D4") == "A0A1B2C3D4"

    def test_invalid_format(self):
        assert extract_uniprot_accession("not_a_uniref_id") is None

    def test_empty_string(self):
        assert extract_uniprot_accession("") is None

    def test_prefix_only(self):
        result = extract_uniprot_accession("UniRef50_")
        assert result is None  # Empty accession returns None


# ============================================================================
# UniProt entry parsing
# ============================================================================


class TestParseUniProtEntry:
    """Test parsing of UniProt JSON entries."""

    def test_full_entry(self):
        entry = _make_uniprot_entry()
        result = _parse_uniprot_entry(entry)

        assert result["target_id"] == "Q9BYF1"
        assert result["description"] == "Protein kinase C"
        assert result["gene"] == "PRKCA"
        assert result["organism"] == "Homo sapiens"
        assert "GO:0006468" in result["go_bp"]
        assert "GO:0004672" in result["go_mf"]
        assert "PF00069" in result["pfam"]
        assert "Kinase" in result["keywords"]

    def test_minimal_entry(self):
        entry = {"primaryAccession": "X12345"}
        result = _parse_uniprot_entry(entry)
        assert result["target_id"] == "X12345"
        assert result["description"] == "hypothetical protein"

    def test_submission_name_fallback(self):
        entry = {
            "primaryAccession": "A0A000",
            "proteinDescription": {
                "submissionNames": [
                    {"fullName": {"value": "Uncharacterized protein"}},
                ],
            },
        }
        result = _parse_uniprot_entry(entry)
        assert result["description"] == "Uncharacterized protein"

    def test_go_term_parsing(self):
        entry = {
            "primaryAccession": "TEST",
            "uniProtKBCrossReferences": [
                {
                    "database": "GO",
                    "id": "GO:0006412",
                    "properties": [
                        {"key": "GoTerm", "value": "P:translation"},
                    ],
                },
                {
                    "database": "GO",
                    "id": "GO:0003723",
                    "properties": [
                        {"key": "GoTerm", "value": "F:RNA binding"},
                    ],
                },
            ],
        }
        result = _parse_uniprot_entry(entry)
        assert "translation [GO:0006412]" in result["go_bp"]
        assert "RNA binding [GO:0003723]" in result["go_mf"]

    def test_pfam_parsing(self):
        entry = {
            "primaryAccession": "TEST",
            "uniProtKBCrossReferences": [
                {
                    "database": "Pfam",
                    "id": "PF00680",
                    "properties": [
                        {"key": "EntryName", "value": "RdRP_1"},
                    ],
                },
            ],
        }
        result = _parse_uniprot_entry(entry)
        assert "PF00680:RdRP_1" in result["pfam"]


# ============================================================================
# UniProt batch lookup (mocked HTTP)
# ============================================================================


class TestLookupUniProtBatch:
    """Test batch UniProt lookups with mocked HTTP."""

    def test_cached_results_returned(self):
        from vhold.features.uniprot_lookup import lookup_uniprot_batch

        cache = {"Q9BYF1": {"description": "cached protein", "source": "uniprot"}}
        result = lookup_uniprot_batch(["Q9BYF1"], cache=cache)
        assert "Q9BYF1" in result
        assert result["Q9BYF1"]["description"] == "cached protein"

    def test_all_cached_no_http(self):
        from vhold.features.uniprot_lookup import lookup_uniprot_batch

        cache = {
            "A": {"description": "prot A"},
            "B": {"description": "prot B"},
        }
        # Should not make any HTTP calls since everything is cached
        result = lookup_uniprot_batch(["A", "B"], cache=cache)
        assert len(result) == 2

    @patch("vhold.features.uniprot_lookup._fetch_uniprot_batch")
    def test_uncached_triggers_fetch(self, mock_fetch):
        from vhold.features.uniprot_lookup import lookup_uniprot_batch

        mock_fetch.return_value = {
            "NEW123": {"description": "new protein", "source": "uniprot"},
        }
        cache: dict = {}
        result = lookup_uniprot_batch(["NEW123"], cache=cache)
        mock_fetch.assert_called_once()
        assert "NEW123" in result
        assert "NEW123" in cache  # Should be cached now


# ============================================================================
# STARLING search (mocked)
# ============================================================================


class TestSearchStarlingDatabase:
    """Test STARLING database search with mocked SearchEngine."""

    def test_empty_sequences(self):
        from vhold.features.starling_search import search_starling_database

        result = search_starling_database({})
        assert result.total_queries == 0
        assert result.matched == {}
        assert result.unmatched == []

    @patch.dict(
        "sys.modules",
        {
            "starling": MagicMock(),
            "starling.search": MagicMock(),
            "starling.search.search_engine": MagicMock(),
        },
    )
    def test_search_with_matches(self):
        from vhold.features.starling_search import search_starling_database

        # Mock sequence_encoder
        mock_encoder = MagicMock()
        mock_embeddings = {
            "prot1": torch.randn(512),
            "prot2": torch.randn(512),
        }
        mock_encoder.return_value = mock_embeddings
        sys.modules["starling"].sequence_encoder = mock_encoder

        # Mock SearchEngine
        mock_engine = MagicMock()
        # Return format: List[List[Tuple[distance, global_idx, uniref50_id, seq_length]]]
        mock_engine.search.return_value = [
            [(0.3, 100, "UniRef50_Q9BYF1", 200)],  # prot1 matches
            [(2.0, 200, "UniRef50_P12345", 150)],  # prot2 too distant
        ]
        mock_engine_cls = MagicMock()
        mock_engine_cls.load.return_value = mock_engine
        sys.modules["starling.search.search_engine"].SearchEngine = mock_engine_cls

        sequences = {"prot1": "MQDRVK" * 20, "prot2": "ALSKDJ" * 20}
        result = search_starling_database(
            sequences,
            similarity_threshold=0.7,
        )

        assert result.total_queries == 2
        assert "prot1" in result.matched
        assert "prot2" in result.unmatched
        assert result.matched["prot1"][0].target_id == "UniRef50_Q9BYF1"
        # distance=0.3 -> similarity = 1 - 0.09/2 = 0.955
        assert result.matched["prot1"][0].similarity > 0.9


# ============================================================================
# Consensus building
# ============================================================================


class TestBuildStarlingConsensusResults:
    """Test building ConsensusResult from STARLING matches."""

    def test_with_annotation(self):
        matches = {
            "prot1": [
                StarlingMatch(
                    query_id="prot1",
                    target_id="UniRef50_Q9BYF1",
                    distance=0.3,
                    similarity=0.95,
                    sequence_length=200,
                ),
            ],
        }
        query_lengths = {"prot1": 150}

        # Pre-populate cache with annotation
        cache = {
            "Q9BYF1": {
                "source": "uniprot",
                "target_id": "Q9BYF1",
                "description": "RNA-dependent RNA polymerase",
                "gene": "RdRp",
                "pfam": "PF00680:RdRP_1",
            },
        }

        results, skipped = build_starling_consensus_results(
            matches, query_lengths, uniprot_cache=cache,
        )

        assert "prot1" in results
        r = results["prot1"]
        assert r.classification_source.startswith("starling_search:")
        assert r.novelty == "ensemble_match"
        assert r.primary_source.startswith("starling_")
        assert r.consensus_score > 0.8
        assert len(skipped) == 0

    def test_skip_hypothetical(self):
        """Proteins with only hypothetical annotations and no evidence should be skipped."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_X00001", 0.3, 0.95, 100),
            ],
        }
        cache = {
            "X00001": {
                "source": "uniprot",
                "target_id": "X00001",
                "description": "hypothetical protein",
            },
        }

        results, skipped = build_starling_consensus_results(
            matches, {"prot1": 100}, uniprot_cache=cache,
        )

        assert "prot1" not in results
        assert "prot1" in skipped

    def test_hypothetical_with_pfam_kept(self):
        """Hypothetical proteins with Pfam evidence should be kept."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_X00001", 0.3, 0.95, 100),
            ],
        }
        cache = {
            "X00001": {
                "source": "uniprot",
                "target_id": "X00001",
                "description": "hypothetical protein",
                "pfam": "PF00069:Pkinase",
            },
        }

        results, skipped = build_starling_consensus_results(
            matches, {"prot1": 100}, uniprot_cache=cache,
        )

        assert "prot1" in results
        assert len(skipped) == 0

    def test_no_uniprot_annotation(self):
        """Matches with no UniProt resolution should be skipped."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "invalid_id", 0.3, 0.95, 100),
            ],
        }

        results, skipped = build_starling_consensus_results(
            matches, {"prot1": 100}, uniprot_cache={},
        )

        assert "prot1" not in results
        assert "prot1" in skipped

    def test_confidence_mapping(self):
        """Test consensus score to confidence level mapping."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_HIGH", 0.1, 0.99, 100),
            ],
        }
        cache = {
            "HIGH": {
                "source": "uniprot",
                "target_id": "HIGH",
                "description": "DNA polymerase",
                "gene": "pol",
            },
        }

        results, _ = build_starling_consensus_results(
            matches, {"prot1": 100}, uniprot_cache=cache,
        )

        assert results["prot1"].confidence_level == "high"

    def test_multiple_proteins(self):
        """Test building results for multiple proteins."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_A", 0.3, 0.95, 100),
            ],
            "prot2": [
                StarlingMatch("prot2", "UniRef50_B", 0.5, 0.875, 200),
            ],
        }
        cache = {
            "A": {"source": "uniprot", "target_id": "A", "description": "protease", "gene": "pro"},
            "B": {"source": "uniprot", "target_id": "B", "description": "capsid protein", "gene": "cap"},
        }

        results, skipped = build_starling_consensus_results(
            matches, {"prot1": 100, "prot2": 200}, uniprot_cache=cache,
        )

        assert len(results) == 2
        assert len(skipped) == 0

    def test_empty_matches(self):
        """Empty matches dict should return empty results."""
        results, skipped = build_starling_consensus_results(
            {}, {}, uniprot_cache={},
        )
        assert len(results) == 0
        assert len(skipped) == 0


# ============================================================================
# UniProt cache persistence
# ============================================================================


class TestUniProtCache:
    """Test UniProt cache save/load."""

    def test_save_and_load(self, tmp_path):
        from vhold.features.uniprot_lookup import load_lookup_cache, save_lookup_cache

        cache = {
            "Q9BYF1": {"description": "test protein", "source": "uniprot"},
            "P12345": {"description": "another protein", "source": "uniprot"},
        }

        cache_path = tmp_path / "uniprot_cache.json"
        save_lookup_cache(cache, cache_path)

        loaded = load_lookup_cache(cache_path)
        assert len(loaded) == 2
        assert loaded["Q9BYF1"]["description"] == "test protein"

    def test_load_nonexistent(self, tmp_path):
        from vhold.features.uniprot_lookup import load_lookup_cache

        loaded = load_lookup_cache(tmp_path / "nonexistent.json")
        assert loaded == {}


# ============================================================================
# UniProt API improvements (retry, GET, User-Agent)
# ============================================================================


class TestUniProtAPIConfig:
    """Test UniProt API configuration constants."""

    def test_user_agent_set(self):
        assert "vHold" in USER_AGENT

    def test_retry_config(self):
        assert MAX_RETRIES == 3
        assert RETRY_BACKOFF_BASE > 0

    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_get_request_used(self, mock_urlopen):
        """Verify _fetch_uniprot_batch uses GET (UniProt search is GET-only)."""
        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _fetch_uniprot_batch(["Q9BYF1"])

        # Verify the request is GET (no data = GET)
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.data is None
        assert "query=" in req.full_url
        assert "accession%3AQ9BYF1" in req.full_url

    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_user_agent_header(self, mock_urlopen):
        """Verify User-Agent header is set on requests."""
        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _fetch_uniprot_batch(["Q9BYF1"])

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("User-agent") == USER_AGENT

    @patch("vhold.features.uniprot_lookup.time.sleep")
    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen, mock_sleep):
        """Test retry with backoff on HTTP 429."""
        import urllib.error

        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        # First call: 429, second call: success
        error_429 = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {"Retry-After": "2"}, None,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_429, mock_response]

        _fetch_uniprot_batch(["Q9BYF1"])

        # Should have retried after sleeping
        mock_sleep.assert_called_once_with(2.0)
        assert mock_urlopen.call_count == 2

    @patch("vhold.features.uniprot_lookup.time.sleep")
    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_retry_on_503(self, mock_urlopen, mock_sleep):
        """Test retry with backoff on HTTP 503."""
        import urllib.error

        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        error_503 = urllib.error.HTTPError(
            "url", 503, "Service Unavailable", {}, None,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_503, mock_response]

        _fetch_uniprot_batch(["Q9BYF1"])

        mock_sleep.assert_called_once()
        assert mock_urlopen.call_count == 2

    @patch("vhold.features.uniprot_lookup.time.sleep")
    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_429_exhausted_returns_empty(self, mock_urlopen, mock_sleep):
        """All retries returning 429 should return empty results, not raise."""
        import urllib.error

        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        error_429 = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, None,
        )
        mock_urlopen.side_effect = [error_429] * MAX_RETRIES

        result = _fetch_uniprot_batch(["Q9BYF1"])

        assert result == {}
        assert mock_urlopen.call_count == MAX_RETRIES

    @patch("vhold.features.uniprot_lookup.time.sleep")
    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_429_no_retry_after_uses_backoff(self, mock_urlopen, mock_sleep):
        """429 without Retry-After header should use exponential backoff."""
        import urllib.error

        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        error_429 = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, None,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": []}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [error_429, mock_response]

        _fetch_uniprot_batch(["Q9BYF1"])

        # Without Retry-After, should use RETRY_BACKOFF_BASE * 2^0 = 1.0
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE)

    @patch("vhold.features.uniprot_lookup.time.sleep")
    @patch("vhold.features.uniprot_lookup.urllib.request.urlopen")
    def test_timeout_then_success(self, mock_urlopen, mock_sleep):
        """Timeout followed by success should return results."""
        from vhold.features.uniprot_lookup import _fetch_uniprot_batch

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"results": [{"primaryAccession": "Q9BYF1"}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [TimeoutError("timed out"), mock_response]

        result = _fetch_uniprot_batch(["Q9BYF1"])

        assert "Q9BYF1" in result
        assert mock_urlopen.call_count == 2


# ============================================================================
# Retry-After header parsing
# ============================================================================


class TestParseRetryAfter:
    """Test Retry-After header parsing."""

    def test_numeric_seconds(self):
        from vhold.features.uniprot_lookup import _parse_retry_after
        assert _parse_retry_after("120", attempt=0) == 120.0

    def test_http_date_falls_back(self):
        """HTTP-date format should fall back to exponential backoff."""
        from vhold.features.uniprot_lookup import _parse_retry_after
        result = _parse_retry_after("Fri, 31 Dec 1999 23:59:59 GMT", attempt=1)
        assert result == RETRY_BACKOFF_BASE * 2  # 2^1

    def test_none_falls_back(self):
        from vhold.features.uniprot_lookup import _parse_retry_after
        result = _parse_retry_after(None, attempt=2)
        assert result == RETRY_BACKOFF_BASE * 4  # 2^2


# ============================================================================
# Corrupt cache handling
# ============================================================================


class TestCorruptCache:
    """Test cache resilience to corruption."""

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt JSON should return empty dict, not crash."""
        from vhold.features.uniprot_lookup import load_lookup_cache

        cache_path = tmp_path / "corrupt.json"
        cache_path.write_text('{"truncated": ')
        loaded = load_lookup_cache(cache_path)
        assert loaded == {}

    def test_atomic_save(self, tmp_path):
        """save_lookup_cache should produce valid JSON."""
        from vhold.features.uniprot_lookup import load_lookup_cache, save_lookup_cache

        cache = {"ACC1": {"description": "test"}}
        cache_path = tmp_path / "cache.json"
        save_lookup_cache(cache, cache_path)

        loaded = load_lookup_cache(cache_path)
        assert loaded == cache


# ============================================================================
# SearchEngine module-level cache
# ============================================================================


class TestSearchEngineCache:
    """Test module-level SearchEngine caching."""

    def test_engine_cached(self):
        """Engine should be cached after first call."""
        import vhold.features.starling_search as ss

        # Reset cache
        original = ss._cached_engine
        ss._cached_engine = None

        mock_engine = MagicMock()
        mock_se_cls = MagicMock()
        mock_se_cls.load.return_value = mock_engine

        try:
            with patch.dict(
                "sys.modules",
                {
                    "starling": MagicMock(),
                    "starling.search": MagicMock(),
                    "starling.search.search_engine": MagicMock(),
                },
            ):
                sys.modules["starling.search.search_engine"].SearchEngine = mock_se_cls

                engine1 = ss._get_search_engine()
                engine2 = ss._get_search_engine()

                # Should only call load once
                mock_se_cls.load.assert_called_once()
                assert engine1 is engine2
        finally:
            # Always clean up
            ss._cached_engine = original


# ============================================================================
# Classification source tracking
# ============================================================================


class TestClassificationSourceTracking:
    """Test that classification_source preserves evidence source."""

    def test_pfam_source(self):
        """Pfam-based classification should include pfam in source."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_Q9BYF1", 0.3, 0.95, 200),
            ],
        }
        cache = {
            "Q9BYF1": {
                "source": "uniprot",
                "target_id": "Q9BYF1",
                "description": "hypothetical protein",
                "pfam": "PF00680:RdRP_1",
            },
        }

        results, _ = build_starling_consensus_results(
            matches, {"prot1": 150}, uniprot_cache=cache,
        )

        assert "prot1" in results
        assert "starling_search:pfam:" in results["prot1"].classification_source

    def test_keywords_source(self):
        """Keyword-based classification should include keywords in source."""
        matches = {
            "prot1": [
                StarlingMatch("prot1", "UniRef50_ABC", 0.3, 0.95, 200),
            ],
        }
        cache = {
            "ABC": {
                "source": "uniprot",
                "target_id": "ABC",
                "description": "DNA polymerase",
                "gene": "pol",
            },
        }

        results, _ = build_starling_consensus_results(
            matches, {"prot1": 150}, uniprot_cache=cache,
        )

        assert "prot1" in results
        assert results["prot1"].classification_source == "starling_search:keywords"
