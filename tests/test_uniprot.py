"""Tests for the UniProt API integration module."""

import pytest
from unittest.mock import patch, MagicMock

from vhold.databases.uniprot import (
    UniProtAnnotationCache,
    fetch_uniprot_entry,
    fetch_uniprot_batch,
    _parse_uniprot_entry,
)


class TestParseUniprotEntry:
    """Tests for _parse_uniprot_entry function."""

    def test_parse_complete_entry(self):
        """Parse a complete UniProt entry."""
        entry = {
            "primaryAccession": "P12345",
            "uniProtkbId": "TEST_ECOLI",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Test protein"}
                }
            },
            "genes": [
                {"geneName": {"value": "testA"}}
            ],
            "organism": {"scientificName": "Escherichia coli"},
            "sequence": {"length": 100},
        }

        result = _parse_uniprot_entry(entry)

        assert result is not None
        assert result["accession"] == "P12345"
        assert result["entry_name"] == "TEST_ECOLI"
        assert result["description"] == "Test protein"
        assert result["gene"] == "testA"
        assert result["organism"] == "Escherichia coli"
        assert result["length"] == 100

    def test_parse_entry_with_submission_name(self):
        """Parse entry with submission name instead of recommended name."""
        entry = {
            "primaryAccession": "A0A001",
            "proteinDescription": {
                "submissionNames": [
                    {"fullName": {"value": "Putative capsid protein"}}
                ]
            },
        }

        result = _parse_uniprot_entry(entry)

        assert result is not None
        assert result["description"] == "Putative capsid protein"

    def test_parse_entry_without_gene(self):
        """Parse entry without gene information."""
        entry = {
            "primaryAccession": "P00001",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {"value": "Some protein"}
                }
            },
        }

        result = _parse_uniprot_entry(entry)

        assert result is not None
        assert result["gene"] == ""

    def test_parse_entry_with_ordered_locus_name(self):
        """Parse entry with ordered locus name fallback."""
        entry = {
            "primaryAccession": "P00002",
            "genes": [
                {"orderedLocusNames": [{"value": "b0001"}]}
            ],
        }

        result = _parse_uniprot_entry(entry)

        assert result is not None
        assert result["gene"] == "b0001"

    def test_parse_empty_entry(self):
        """Parse empty entry should return None."""
        result = _parse_uniprot_entry({})
        assert result is None

    def test_parse_entry_missing_accession(self):
        """Parse entry without accession returns None."""
        entry = {
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Test"}}
            }
        }
        result = _parse_uniprot_entry(entry)
        assert result is None


class TestUniProtAnnotationCache:
    """Tests for UniProtAnnotationCache class."""

    def test_cache_initialization(self):
        """Cache initializes empty."""
        cache = UniProtAnnotationCache()
        assert len(cache) == 0

    def test_get_returns_none_for_missing(self):
        """Get returns None for missing accession when fetch fails."""
        cache = UniProtAnnotationCache()

        with patch("vhold.databases.uniprot.fetch_uniprot_entry") as mock_fetch:
            mock_fetch.return_value = None
            result = cache.get("NONEXISTENT")

        assert result is None
        assert "NONEXISTENT" in cache  # Should be cached as None

    def test_get_caches_result(self):
        """Subsequent gets return cached value without API call."""
        cache = UniProtAnnotationCache()

        mock_data = {"accession": "P12345", "description": "Test"}

        with patch("vhold.databases.uniprot.fetch_uniprot_entry") as mock_fetch:
            mock_fetch.return_value = mock_data

            # First call should fetch
            result1 = cache.get("P12345")
            assert result1 == mock_data
            assert mock_fetch.call_count == 1

            # Second call should use cache
            result2 = cache.get("P12345")
            assert result2 == mock_data
            assert mock_fetch.call_count == 1  # No additional call

    def test_prefetch_batches_accessions(self):
        """Prefetch fetches multiple accessions in batch."""
        cache = UniProtAnnotationCache()

        mock_results = {
            "P12345": {"accession": "P12345", "description": "Protein 1"},
            "P67890": {"accession": "P67890", "description": "Protein 2"},
        }

        with patch("vhold.databases.uniprot.fetch_uniprot_batch") as mock_batch:
            mock_batch.return_value = mock_results

            cache.prefetch(["P12345", "P67890", "MISSING"])

            # All accessions should be cached
            assert "P12345" in cache
            assert "P67890" in cache
            assert "MISSING" in cache  # Cached as None

            # Should be able to get cached values
            assert cache.get("P12345")["description"] == "Protein 1"
            assert cache.get("MISSING") is None

    def test_prefetch_skips_already_cached(self):
        """Prefetch doesn't re-fetch already cached accessions."""
        cache = UniProtAnnotationCache()
        cache._cache["P12345"] = {"accession": "P12345", "description": "Cached"}

        with patch("vhold.databases.uniprot.fetch_uniprot_batch") as mock_batch:
            mock_batch.return_value = {}

            cache.prefetch(["P12345", "P67890"])

            # Should only fetch P67890
            mock_batch.assert_called_once()
            fetched_ids = mock_batch.call_args[0][0]
            assert "P12345" not in fetched_ids
            assert "P67890" in fetched_ids

    def test_contains(self):
        """Test __contains__ method."""
        cache = UniProtAnnotationCache()
        cache._cache["P12345"] = {"accession": "P12345"}

        assert "P12345" in cache
        assert "MISSING" not in cache

    def test_len(self):
        """Test __len__ method."""
        cache = UniProtAnnotationCache()
        assert len(cache) == 0

        cache._cache["P12345"] = {"accession": "P12345"}
        cache._cache["P67890"] = None

        assert len(cache) == 2


class TestFetchUniprotEntry:
    """Tests for fetch_uniprot_entry function."""

    def test_successful_fetch(self):
        """Successful fetch returns parsed entry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "primaryAccession": "P12345",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Test protein"}}
            },
        }

        with patch("vhold.databases.uniprot.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = fetch_uniprot_entry("P12345")

            assert result is not None
            assert result["accession"] == "P12345"
            assert result["description"] == "Test protein"

    def test_not_found_returns_none(self):
        """404 response returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("vhold.databases.uniprot.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = fetch_uniprot_entry("NONEXISTENT")

            assert result is None

    def test_request_error_returns_none(self):
        """Request exception returns None."""
        import requests as req

        with patch("vhold.databases.uniprot.requests.get") as mock_get:
            mock_get.side_effect = req.RequestException("Network error")

            result = fetch_uniprot_entry("P12345")

            assert result is None


class TestFetchUniprotBatch:
    """Tests for fetch_uniprot_batch function."""

    def test_empty_list_returns_empty_dict(self):
        """Empty accession list returns empty dict."""
        result = fetch_uniprot_batch([])
        assert result == {}

    def test_deduplicates_accessions(self):
        """Duplicate accessions are deduplicated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "primaryAccession": "P12345",
                    "proteinDescription": {
                        "recommendedName": {"fullName": {"value": "Test"}}
                    },
                }
            ]
        }

        with patch("vhold.databases.uniprot.requests.get") as mock_get:
            mock_get.return_value = mock_response

            result = fetch_uniprot_batch(["P12345", "P12345", "P12345"])

            assert len(result) == 1
            assert "P12345" in result
