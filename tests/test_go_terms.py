"""Tests for GO term ID resolution."""

from unittest.mock import patch

import pytest

from vhold.results.go_terms import (
    enrich_annotation_with_go_ids,
    load_go_term_map,
    resolve_go_ids,
)


class TestLoadGoTermMap:
    """Tests for loading the bundled GO term map."""

    def test_returns_nonempty_dict(self):
        """Should return a non-empty mapping."""
        # Clear cache from previous test runs
        load_go_term_map.cache_clear()
        term_map = load_go_term_map()
        assert isinstance(term_map, dict)
        assert len(term_map) > 10000

    def test_known_term_present(self):
        """Should contain well-known GO terms."""
        term_map = load_go_term_map()
        assert "viral process" in term_map
        assert term_map["viral process"].startswith("GO:")

    def test_case_insensitive_keys(self):
        """Keys should be lowercase for case-insensitive lookup."""
        term_map = load_go_term_map()
        # All keys should be lowercase
        for key in list(term_map.keys())[:100]:
            assert key == key.lower()

    def test_missing_file_returns_empty(self, tmp_path):
        """Should return empty dict if map file not found."""
        load_go_term_map.cache_clear()
        with patch("vhold.results.go_terms.GO_TERM_MAP_PATH", tmp_path / "nonexistent.json"):
            result = load_go_term_map()
            assert result == {}
        load_go_term_map.cache_clear()


class TestResolveGoIds:
    """Tests for resolving GO IDs from term strings."""

    def test_inline_ids_extracted(self):
        """Should extract GO IDs already present in BFVD format."""
        result = resolve_go_ids(
            "DNA-templated transcription [GO:0006351]"
        )
        assert result == "GO:0006351"

    def test_multiple_inline_ids(self):
        """Should extract multiple inline GO IDs."""
        result = resolve_go_ids(
            "RNA binding [GO:0003723]; "
            "RNA-directed RNA polymerase activity [GO:0003968]"
        )
        assert result == "GO:0003723; GO:0003968"

    def test_plain_term_looked_up(self):
        """Should look up GO ID for plain term names (Viro3D format)."""
        result = resolve_go_ids("viral process")
        assert result == "GO:0016032"

    def test_multiple_plain_terms(self):
        """Should look up multiple semicolon-separated terms."""
        result = resolve_go_ids("viral process; RNA binding")
        assert "GO:0016032" in result
        assert "GO:0003723" in result

    def test_empty_string(self):
        """Should return empty string for empty input."""
        assert resolve_go_ids("") == ""
        assert resolve_go_ids("  ") == ""

    def test_unknown_term(self):
        """Should return empty string for unknown terms."""
        result = resolve_go_ids("completely fake term xyz123")
        assert result == ""

    def test_mixed_known_unknown(self):
        """Should only include IDs for terms that resolve."""
        result = resolve_go_ids("viral process; fake_term_xyz")
        assert "GO:0016032" in result
        assert "fake" not in result

    def test_none_input(self):
        """Should handle None gracefully."""
        assert resolve_go_ids(None) == ""


class TestEnrichAnnotation:
    """Tests for enriching annotation dicts with GO IDs."""

    def test_adds_go_bp_ids(self):
        """Should add go_bp_ids from BFVD-format go_bp."""
        annotation = {
            "go_bp": "DNA-templated transcription [GO:0006351]",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_bp_ids"] == "GO:0006351"
        # Original go_bp unchanged
        assert "DNA-templated transcription" in result["go_bp"]

    def test_adds_go_mf_ids(self):
        """Should add go_mf_ids from go_mf."""
        annotation = {
            "go_mf": "RNA binding [GO:0003723]",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_mf_ids"] == "GO:0003723"

    def test_adds_both(self):
        """Should add both go_bp_ids and go_mf_ids."""
        annotation = {
            "go_bp": "viral process [GO:0016032]",
            "go_mf": "RNA binding [GO:0003723]",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_bp_ids"] == "GO:0016032"
        assert result["go_mf_ids"] == "GO:0003723"

    def test_no_go_fields(self):
        """Should not add IDs when no GO fields present."""
        annotation = {"description": "some protein"}
        result = enrich_annotation_with_go_ids(annotation)
        assert "go_bp_ids" not in result
        assert "go_mf_ids" not in result

    def test_empty_go_fields(self):
        """Should not add IDs for empty GO strings."""
        annotation = {"go_bp": "", "go_mf": ""}
        result = enrich_annotation_with_go_ids(annotation)
        assert "go_bp_ids" not in result
        assert "go_mf_ids" not in result

    def test_modifies_in_place(self):
        """Should modify the annotation dict in place."""
        annotation = {"go_bp": "viral process [GO:0016032]"}
        result = enrich_annotation_with_go_ids(annotation)
        assert result is annotation

    def test_viro3d_plain_terms(self):
        """Should resolve plain term names (Viro3D format)."""
        annotation = {
            "go_bp": "viral process",
            "go_mf": "RNA binding",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_bp_ids"] == "GO:0016032"
        assert result["go_mf_ids"] == "GO:0003723"

    def test_adds_go_cc_ids(self):
        """Should add go_cc_ids from go_cc."""
        annotation = {
            "go_cc": "host cell membrane [GO:0033644]",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_cc_ids"] == "GO:0033644"

    def test_adds_all_three_ontologies(self):
        """Should add IDs for BP, MF, and CC simultaneously."""
        annotation = {
            "go_bp": "viral process [GO:0016032]",
            "go_mf": "RNA binding [GO:0003723]",
            "go_cc": "virion [GO:0019012]",
        }
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_bp_ids"] == "GO:0016032"
        assert result["go_mf_ids"] == "GO:0003723"
        assert result["go_cc_ids"] == "GO:0019012"

    def test_go_cc_plain_term(self):
        """Should resolve plain GO CC term names."""
        annotation = {"go_cc": "viral capsid"}
        result = enrich_annotation_with_go_ids(annotation)
        assert result["go_cc_ids"] == "GO:0019028"
