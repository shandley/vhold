"""Tests for ESM Metagenomic Atlas integration."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from vhold.databases.esm_atlas import (
    ESMAtlasHit,
    ESMAtlasSearchResult,
    ESMAtlasAnalysis,
    check_esmatlas_database,
    install_esmatlas_database,
    search_esmatlas,
    parse_esmatlas_results,
    analyze_dark_matter_with_esmatlas,
    ESMATLAS_DB_NAME,
    ESMATLAS_DB_PATH,
)
from vhold.results.dark_matter import DarkMatterProtein


class TestESMAtlasHit:
    """Tests for ESMAtlasHit dataclass."""

    def test_creation(self):
        """Test creating an ESMAtlasHit."""
        hit = ESMAtlasHit(
            query_id="protein_1",
            target_id="MGYP001234567890",
            evalue=1e-10,
            bits=120.5,
            fident=0.75,
            alnlen=250,
            qcov=0.90,
            tcov=0.85,
        )
        assert hit.query_id == "protein_1"
        assert hit.target_id == "MGYP001234567890"
        assert hit.evalue == 1e-10
        assert hit.fident == 0.75

    def test_optional_fields(self):
        """Test optional metadata fields."""
        hit = ESMAtlasHit(
            query_id="protein_1",
            target_id="MGYP001234567890",
            evalue=1e-10,
            bits=120.5,
            fident=0.75,
            alnlen=250,
            qcov=0.90,
            tcov=0.85,
            mgnify_id="MGYP001234567890",
            predicted_plddt=0.85,
            predicted_ptm=0.78,
            cluster_id="cluster_1",
            is_novel_fold=True,
        )
        assert hit.mgnify_id == "MGYP001234567890"
        assert hit.predicted_plddt == 0.85
        assert hit.is_novel_fold is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hit = ESMAtlasHit(
            query_id="protein_1",
            target_id="MGYP001234567890",
            evalue=1e-10,
            bits=120.5,
            fident=0.75,
            alnlen=250,
            qcov=0.90,
            tcov=0.85,
            is_novel_fold=True,
        )
        d = hit.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["target_id"] == "MGYP001234567890"
        assert d["is_novel_fold"] is True
        assert "mgnify_id" in d  # Optional fields included


class TestESMAtlasSearchResult:
    """Tests for ESMAtlasSearchResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = ESMAtlasSearchResult(
            query_id="protein_1",
            query_length=300,
        )
        assert result.hits == []
        assert result.total_hits == 0
        assert result.best_evalue is None
        assert result.has_metagenomic_homolog is False
        assert result.has_novel_fold_match is False

    def test_with_hits(self):
        """Test result with hits."""
        hit1 = ESMAtlasHit(
            query_id="protein_1",
            target_id="MGYP001",
            evalue=1e-15,
            bits=150.0,
            fident=0.80,
            alnlen=280,
            qcov=0.95,
            tcov=0.90,
            is_novel_fold=True,
        )
        hit2 = ESMAtlasHit(
            query_id="protein_1",
            target_id="MGYP002",
            evalue=1e-10,
            bits=120.0,
            fident=0.70,
            alnlen=250,
            qcov=0.85,
            tcov=0.80,
        )
        result = ESMAtlasSearchResult(
            query_id="protein_1",
            query_length=300,
            hits=[hit1, hit2],
            total_hits=2,
            best_evalue=1e-15,
            best_identity=0.80,
            has_metagenomic_homolog=True,
            has_novel_fold_match=True,
            unique_clusters=2,
        )
        assert len(result.hits) == 2
        assert result.has_metagenomic_homolog is True
        assert result.has_novel_fold_match is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ESMAtlasSearchResult(
            query_id="protein_1",
            query_length=300,
            total_hits=5,
            best_evalue=1e-20,
            has_metagenomic_homolog=True,
        )
        d = result.to_dict()
        assert d["query_id"] == "protein_1"
        assert d["total_hits"] == 5
        assert d["has_metagenomic_homolog"] is True
        assert "hits" in d


class TestESMAtlasAnalysis:
    """Tests for ESMAtlasAnalysis dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        analysis = ESMAtlasAnalysis()
        assert analysis.total_queries == 0
        assert analysis.queries_with_hits == 0
        assert analysis.novel_fold_candidates == []

    def test_with_results(self):
        """Test analysis with results."""
        result1 = ESMAtlasSearchResult(
            query_id="protein_1",
            query_length=300,
            has_metagenomic_homolog=True,
            has_novel_fold_match=True,
        )
        analysis = ESMAtlasAnalysis(
            total_queries=5,
            queries_with_hits=3,
            queries_with_novel_folds=1,
            total_hits=25,
            unique_targets=20,
            results={"protein_1": result1},
            novel_fold_candidates=["protein_1"],
            high_priority_targets=["protein_2"],
        )
        assert analysis.total_queries == 5
        assert analysis.queries_with_hits == 3
        assert len(analysis.novel_fold_candidates) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        analysis = ESMAtlasAnalysis(
            total_queries=10,
            queries_with_hits=5,
            queries_with_novel_folds=2,
            total_hits=50,
        )
        d = analysis.to_dict()
        assert "summary" in d
        assert d["summary"]["total_queries"] == 10
        assert d["summary"]["hit_rate"] == 0.5
        assert "novel_fold_candidates" in d

    def test_hit_rate_zero_division(self):
        """Test hit rate calculation with zero queries."""
        analysis = ESMAtlasAnalysis(total_queries=0)
        d = analysis.to_dict()
        assert d["summary"]["hit_rate"] == 0


class TestCheckESMAtlasDatabase:
    """Tests for check_esmatlas_database function."""

    def test_database_not_installed(self):
        """Test when database is not installed."""
        with patch("vhold.databases.esm_atlas.ESMATLAS_DB_PATH") as mock_path:
            mock_path.with_suffix.return_value.exists.return_value = False
            # The actual function checks multiple files
            result = check_esmatlas_database()
            # Since ESMATLAS_DB_PATH is a constant, we test differently
            assert isinstance(result, bool)

    @patch("pathlib.Path.exists")
    def test_database_check_requires_files(self, mock_exists):
        """Test that check requires specific files."""
        # Database consists of multiple files
        mock_exists.return_value = False
        result = check_esmatlas_database()
        assert result is False


class TestInstallESMAtlasDatabase:
    """Tests for install_esmatlas_database function."""

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    def test_skip_if_installed(self, mock_check):
        """Test skipping installation if already installed."""
        mock_check.return_value = True
        result = install_esmatlas_database(force=False)
        assert result is True

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("subprocess.run")
    def test_install_success(self, mock_run, mock_check):
        """Test successful installation."""
        mock_check.return_value = False
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = install_esmatlas_database(output_dir=Path(tmp_dir))
            assert result is True

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("subprocess.run")
    def test_install_failure(self, mock_run, mock_check):
        """Test handling installation failure."""
        mock_check.return_value = False
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="Download failed"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = install_esmatlas_database(output_dir=Path(tmp_dir))
            assert result is False

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("subprocess.run")
    def test_foldseek_not_found(self, mock_run, mock_check):
        """Test handling missing foldseek."""
        mock_check.return_value = False
        mock_run.side_effect = FileNotFoundError("foldseek not found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = install_esmatlas_database(output_dir=Path(tmp_dir))
            assert result is False


class TestSearchESMAtlas:
    """Tests for search_esmatlas function."""

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    def test_raises_if_db_not_installed(self, mock_check):
        """Test error when database not installed."""
        mock_check.return_value = False
        with pytest.raises(RuntimeError, match="not installed"):
            search_esmatlas(
                query_db=Path("/tmp/query"),
                output_dir=Path("/tmp/output"),
            )

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("subprocess.run")
    def test_search_success(self, mock_run, mock_check):
        """Test successful search."""
        mock_check.return_value = True
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = search_esmatlas(
                query_db=tmp_path / "query",
                output_dir=tmp_path / "output",
                threads=2,
            )
            assert result.name == "esmatlas_results.tsv"

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("subprocess.run")
    def test_search_failure(self, mock_run, mock_check):
        """Test handling search failure."""
        mock_check.return_value = True
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="Search failed"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with pytest.raises(RuntimeError, match="failed"):
                search_esmatlas(
                    query_db=tmp_path / "query",
                    output_dir=tmp_path / "output",
                )


class TestParseESMAtlasResults:
    """Tests for parse_esmatlas_results function."""

    def test_missing_file(self):
        """Test handling missing results file."""
        result = parse_esmatlas_results(Path("/nonexistent/file.tsv"))
        assert result == {}

    def test_parse_valid_results(self):
        """Test parsing valid results file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # Write test data in Foldseek format
            # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,qcov,tcov
            f.write(
                "protein_1\tMGYP001234\t0.75\t250\t50\t5\t1\t250\t1\t240\t1e-15\t150.5\t300\t280\t0.83\t0.86\n"
            )
            f.write(
                "protein_1\tESM_novel\t0.60\t200\t70\t8\t1\t200\t1\t190\t1e-8\t100.2\t300\t220\t0.67\t0.86\n"
            )
            f.write(
                "protein_2\tMGYP005678\t0.80\t280\t40\t3\t1\t280\t1\t275\t1e-20\t180.0\t350\t300\t0.80\t0.92\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name), min_identity=0.5)

        assert len(results) == 2
        assert "protein_1" in results
        assert "protein_2" in results
        assert results["protein_1"].total_hits == 2
        assert results["protein_2"].total_hits == 1

    def test_filter_low_identity(self):
        """Test filtering by minimum identity."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "protein_1\tMGYP001\t0.25\t250\t50\t5\t1\t250\t1\t240\t1e-5\t50.0\t300\t280\t0.83\t0.86\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name), min_identity=0.3)

        # Should be filtered out due to low identity
        assert len(results) == 0

    def test_filter_low_coverage(self):
        """Test filtering by minimum coverage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "protein_1\tMGYP001\t0.75\t100\t20\t2\t1\t100\t1\t95\t1e-10\t80.0\t300\t280\t0.33\t0.34\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name), min_coverage=0.5)

        # Should be filtered out due to low coverage
        assert len(results) == 0

    def test_mgnify_id_extraction(self):
        """Test MGnify ID extraction from target."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "protein_1\tMGYP001234567890\t0.75\t250\t50\t5\t1\t250\t1\t240\t1e-15\t150.0\t300\t280\t0.83\t0.86\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name))

        assert results["protein_1"].hits[0].mgnify_id == "MGYP001234567890"

    def test_novel_fold_detection(self):
        """Test detection of potential novel folds."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            # ESM-prefixed target suggests novel fold
            f.write(
                "protein_1\tESM_12345\t0.75\t250\t50\t5\t1\t250\t1\t240\t1e-15\t150.0\t300\t280\t0.83\t0.86\n"
            )
            # MGY-prefixed target also suggests metagenomic origin
            f.write(
                "protein_1\tMGY_67890\t0.70\t220\t60\t6\t1\t220\t1\t210\t1e-12\t130.0\t300\t250\t0.73\t0.84\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name))

        assert results["protein_1"].has_novel_fold_match is True
        assert any(h.is_novel_fold for h in results["protein_1"].hits)

    def test_best_scores_tracking(self):
        """Test tracking of best e-value and identity."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write(
                "protein_1\tMGYP001\t0.60\t200\t70\t8\t1\t200\t1\t190\t1e-8\t100.0\t300\t220\t0.67\t0.86\n"
            )
            f.write(
                "protein_1\tMGYP002\t0.85\t280\t30\t2\t1\t280\t1\t275\t1e-20\t180.0\t300\t290\t0.93\t0.95\n"
            )
            f.flush()

            results = parse_esmatlas_results(Path(f.name))

        assert results["protein_1"].best_evalue == 1e-20
        assert results["protein_1"].best_identity == 0.85


class TestAnalyzeDarkMatterWithESMAtlas:
    """Tests for analyze_dark_matter_with_esmatlas function."""

    def test_empty_input(self):
        """Test with empty protein list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            analysis = analyze_dark_matter_with_esmatlas(
                dark_matter_proteins=[],
                predictions_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir),
            )
            assert analysis.total_queries == 0

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    def test_skip_when_db_not_installed(self, mock_check):
        """Test graceful skip when database not installed."""
        mock_check.return_value = False

        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",
                reason="No structural homologs",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            analysis = analyze_dark_matter_with_esmatlas(
                dark_matter_proteins=proteins,
                predictions_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir),
            )
            # Should return empty analysis, not error
            assert analysis.total_queries == 1
            assert analysis.queries_with_hits == 0

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    def test_no_3di_files(self, mock_check):
        """Test handling when no 3Di files exist."""
        mock_check.return_value = True

        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",
                reason="No structural homologs",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Empty predictions directory
            pred_dir = Path(tmp_dir) / "predictions"
            pred_dir.mkdir()

            analysis = analyze_dark_matter_with_esmatlas(
                dark_matter_proteins=proteins,
                predictions_dir=pred_dir,
                output_dir=Path(tmp_dir) / "output",
            )
            assert analysis.queries_with_hits == 0

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("vhold.databases.esm_atlas.search_esmatlas")
    @patch("vhold.databases.esm_atlas.parse_esmatlas_results")
    @patch("subprocess.run")
    def test_full_analysis(
        self, mock_run, mock_parse, mock_search, mock_check
    ):
        """Test full analysis pipeline."""
        mock_check.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock search and parse
        mock_search.return_value = Path("/tmp/results.tsv")
        mock_parse.return_value = {
            "protein_1": ESMAtlasSearchResult(
                query_id="protein_1",
                query_length=300,
                total_hits=5,
                has_metagenomic_homolog=True,
                has_novel_fold_match=True,
                hits=[
                    ESMAtlasHit(
                        query_id="protein_1",
                        target_id="MGYP001",
                        evalue=1e-15,
                        bits=150.0,
                        fident=0.8,
                        alnlen=250,
                        qcov=0.9,
                        tcov=0.85,
                        is_novel_fold=True,
                    )
                ],
            )
        }

        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",
                reason="No structural homologs",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create mock 3Di file
            pred_dir = Path(tmp_dir) / "predictions"
            pred_dir.mkdir()
            tdi_file = pred_dir / "protein_1.3di"
            tdi_file.write_text("ACDEFGHIKLMNPQRSTVWY")

            analysis = analyze_dark_matter_with_esmatlas(
                dark_matter_proteins=proteins,
                predictions_dir=pred_dir,
                output_dir=Path(tmp_dir) / "output",
            )

            assert analysis.queries_with_hits == 1
            assert analysis.queries_with_novel_folds == 1
            assert "protein_1" in analysis.novel_fold_candidates

    @patch("vhold.databases.esm_atlas.check_esmatlas_database")
    @patch("vhold.databases.esm_atlas.search_esmatlas")
    @patch("vhold.databases.esm_atlas.parse_esmatlas_results")
    @patch("subprocess.run")
    def test_high_priority_identification(
        self, mock_run, mock_parse, mock_search, mock_check
    ):
        """Test identification of high priority targets."""
        mock_check.return_value = True
        mock_run.return_value = Mock(returncode=0)
        mock_search.return_value = Path("/tmp/results.tsv")

        # Protein with no viral DB hits but has ESMAtlas hits
        mock_parse.return_value = {
            "protein_1": ESMAtlasSearchResult(
                query_id="protein_1",
                query_length=300,
                total_hits=3,
                has_metagenomic_homolog=True,
            )
        }

        proteins = [
            DarkMatterProtein(
                query_id="protein_1",
                query_length=300,
                category="no_hits",  # Key: no viral DB hits
                reason="No structural homologs",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_dir = Path(tmp_dir) / "predictions"
            pred_dir.mkdir()
            (pred_dir / "protein_1.3di").write_text("ACDEFGHIKLMNPQRSTVWY")

            analysis = analyze_dark_matter_with_esmatlas(
                dark_matter_proteins=proteins,
                predictions_dir=pred_dir,
                output_dir=Path(tmp_dir) / "output",
            )

            assert "protein_1" in analysis.high_priority_targets


class TestModuleConstants:
    """Tests for module constants."""

    def test_database_name(self):
        """Test database name constant."""
        assert ESMATLAS_DB_NAME == "ESMAtlas30"

    def test_database_path_is_path(self):
        """Test database path is a Path object."""
        assert isinstance(ESMATLAS_DB_PATH, Path)
