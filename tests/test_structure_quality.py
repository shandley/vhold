"""Tests for structure quality-based confidence weighting."""

import pytest
from vhold.results.consensus import (
    calculate_structure_quality,
    calculate_hit_quality,
    score_hit,
    HitScore,
    STRUCTURE_QUALITY_THRESHOLDS,
    STRUCTURE_QUALITY_WEIGHT,
)
from vhold.results.parser import FoldseekHit


class TestCalculateStructureQuality:
    """Tests for calculate_structure_quality function."""

    def test_no_annotation(self):
        """Test with no annotation data."""
        score, source = calculate_structure_quality({})
        assert score == 1.0
        assert source == "none"

    def test_none_annotation(self):
        """Test with None annotation."""
        score, source = calculate_structure_quality(None)
        assert score == 1.0
        assert source == "none"

    def test_high_quality_colabfold(self):
        """Test with high quality ColabFold prediction."""
        annotation = {
            "colabfold_plddt": 85.0,
            "colabfold_ptm": 0.8,
            "colabfold_msa_depth": 500,
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "colabfold"
        assert score > 0.9  # High quality should be >0.9

    def test_low_quality_colabfold(self):
        """Test with low quality ColabFold prediction."""
        annotation = {
            "colabfold_plddt": 40.0,
            "colabfold_ptm": 0.25,
            "colabfold_msa_depth": 50,  # Low MSA depth
        }
        score, source = calculate_structure_quality(annotation)
        # Low MSA depth should still use colabfold but without high MSA bonus
        assert source == "colabfold_low_msa"
        assert score < 0.8

    def test_esmfold_prediction(self):
        """Test with ESMFold prediction."""
        annotation = {
            "esmfold_plddt": 70.0,
            "esmfold_ptm": 0.6,
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "esmfold"
        assert 0.85 <= score <= 1.0

    def test_bfvd_af2_prediction(self):
        """Test with BFVD AlphaFold2 prediction."""
        annotation = {
            "plddt": 80.0,
            "ptm": 0.7,
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "bfvd_af2"
        assert score > 0.9

    def test_colabfold_preferred_with_high_msa(self):
        """Test that ColabFold is preferred when MSA depth is high."""
        annotation = {
            "esmfold_plddt": 90.0,
            "esmfold_ptm": 0.9,
            "colabfold_plddt": 75.0,
            "colabfold_ptm": 0.6,
            "colabfold_msa_depth": 200,  # Above threshold
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "colabfold"

    def test_esmfold_preferred_with_low_msa(self):
        """Test that ESMFold is preferred when ColabFold MSA depth is low."""
        annotation = {
            "esmfold_plddt": 70.0,
            "esmfold_ptm": 0.5,
            "colabfold_plddt": 80.0,
            "colabfold_ptm": 0.7,
            "colabfold_msa_depth": 50,  # Below threshold
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "esmfold"

    def test_very_low_plddt(self):
        """Test with very low pLDDT (disordered region)."""
        annotation = {
            "esmfold_plddt": 20.0,
            "esmfold_ptm": 0.1,
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "esmfold"
        assert 0.3 <= score <= 0.5  # Should have low score

    def test_missing_ptm(self):
        """Test with missing pTM value."""
        annotation = {
            "esmfold_plddt": 70.0,
            # No pTM
        }
        score, source = calculate_structure_quality(annotation)
        assert source == "esmfold"
        assert score > 0.8  # Should use default pTM score

    def test_msa_depth_bonus(self):
        """Test MSA depth bonus for ColabFold."""
        annotation_low_msa = {
            "colabfold_plddt": 80.0,
            "colabfold_ptm": 0.7,
            "colabfold_msa_depth": 100,
        }
        annotation_high_msa = {
            "colabfold_plddt": 80.0,
            "colabfold_ptm": 0.7,
            "colabfold_msa_depth": 1000,
        }
        score_low, _ = calculate_structure_quality(annotation_low_msa)
        score_high, _ = calculate_structure_quality(annotation_high_msa)
        # High MSA should get bonus
        assert score_high > score_low


class TestStructureQualityIntegration:
    """Tests for structure quality integration in scoring."""

    def _create_hit(
        self, evalue=1e-30, fident=0.9, qcov=0.95, source_db="viro3d"
    ) -> FoldseekHit:
        """Create a test hit."""
        return FoldseekHit(
            query="test_query",
            target="test_target",
            fident=fident,
            alnlen=100,
            mismatch=10,
            gapopen=2,
            qstart=1,
            qend=100,
            tstart=1,
            tend=100,
            evalue=evalue,
            bits=200.0,
            qlen=100,
            tlen=100,
            qcov=qcov,
            tcov=0.9,
            source_db=source_db,
        )

    def test_high_quality_structure_minimal_penalty(self):
        """Test that high quality structures have minimal score reduction."""
        hit = self._create_hit()
        annotation_high = {
            "esmfold_plddt": 90.0,
            "esmfold_ptm": 0.9,
        }
        annotation_none = {}

        scored_high = score_hit(hit, annotation_high)
        scored_none = score_hit(hit, annotation_none)

        # High quality should be close to no-quality score
        assert scored_high.quality_score >= scored_none.quality_score * 0.95

    def test_low_quality_structure_penalty(self):
        """Test that low quality structures reduce the score."""
        hit = self._create_hit()
        annotation_high = {
            "esmfold_plddt": 90.0,
            "esmfold_ptm": 0.9,
        }
        annotation_low = {
            "esmfold_plddt": 30.0,
            "esmfold_ptm": 0.2,
        }

        scored_high = score_hit(hit, annotation_high)
        scored_low = score_hit(hit, annotation_low)

        # Low quality should have lower score
        assert scored_low.quality_score < scored_high.quality_score

    def test_structure_quality_stored_in_hitscore(self):
        """Test that structure quality is stored in HitScore."""
        hit = self._create_hit()
        annotation = {
            "esmfold_plddt": 75.0,
            "esmfold_ptm": 0.6,
        }

        scored = score_hit(hit, annotation)

        assert scored.structure_quality > 0
        assert scored.structure_quality <= 1.0
        assert scored.structure_quality_source == "esmfold"

    def test_structure_quality_weight_effect(self):
        """Test that STRUCTURE_QUALITY_WEIGHT controls penalty magnitude."""
        hit = self._create_hit()
        base_quality = calculate_hit_quality(hit)

        # Very low quality structure
        annotation = {
            "esmfold_plddt": 10.0,
            "esmfold_ptm": 0.05,
        }

        scored = score_hit(hit, annotation)

        # Maximum penalty should be limited by STRUCTURE_QUALITY_WEIGHT
        min_expected = base_quality * (1 - STRUCTURE_QUALITY_WEIGHT)
        assert scored.quality_score >= min_expected * 0.95  # Allow small tolerance


class TestStructureQualityThresholds:
    """Tests for structure quality thresholds."""

    def test_threshold_values(self):
        """Test that thresholds are sensible."""
        assert STRUCTURE_QUALITY_THRESHOLDS["high_plddt"] == 70.0
        assert STRUCTURE_QUALITY_THRESHOLDS["medium_plddt"] == 50.0
        assert STRUCTURE_QUALITY_THRESHOLDS["low_plddt"] == 30.0
        assert STRUCTURE_QUALITY_THRESHOLDS["high_ptm"] == 0.5
        assert STRUCTURE_QUALITY_THRESHOLDS["medium_ptm"] == 0.3
        assert STRUCTURE_QUALITY_THRESHOLDS["good_msa_depth"] == 100

    def test_weight_value(self):
        """Test structure quality weight is reasonable."""
        assert 0 < STRUCTURE_QUALITY_WEIGHT < 0.5  # Should be moderate influence
        assert STRUCTURE_QUALITY_WEIGHT == 0.15


class TestEdgeCases:
    """Edge case tests for structure quality scoring."""

    def test_extreme_plddt_values(self):
        """Test with extreme pLDDT values."""
        # Maximum pLDDT
        annotation_max = {"esmfold_plddt": 100.0, "esmfold_ptm": 1.0}
        score_max, _ = calculate_structure_quality(annotation_max)
        assert score_max == 1.0

        # Minimum pLDDT
        annotation_min = {"esmfold_plddt": 0.0, "esmfold_ptm": 0.0}
        score_min, _ = calculate_structure_quality(annotation_min)
        assert 0.3 <= score_min <= 0.5

    def test_boundary_plddt_values(self):
        """Test at threshold boundaries."""
        # Just above high threshold (with high pTM for combined score)
        annotation = {"esmfold_plddt": 70.0, "esmfold_ptm": 0.9}
        score, _ = calculate_structure_quality(annotation)
        # pLDDT 70 = 0.9, pTM 0.9 = ~0.97, combined: 0.7*0.9 + 0.3*0.97 = 0.921
        assert score >= 0.9

        # Just below high threshold
        annotation = {"esmfold_plddt": 65.0, "esmfold_ptm": 0.9}
        score_medium, _ = calculate_structure_quality(annotation)
        # pLDDT in medium range should score lower than high threshold
        assert score_medium < score  # Medium pLDDT should score lower

    def test_partial_quality_data(self):
        """Test with partial quality data available."""
        # Only pLDDT
        annotation = {"colabfold_plddt": 80.0, "colabfold_msa_depth": 200}
        score, source = calculate_structure_quality(annotation)
        assert source == "colabfold"
        assert score > 0.8
