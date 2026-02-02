"""Database management for vhold."""

from vhold.databases.install import install_databases, check_databases
from vhold.databases.bfvd import load_bfvd_metadata
from vhold.databases.uniprot import UniProtAnnotationCache, fetch_uniprot_batch
from vhold.databases.viro3d import (
    load_viro3d_metadata,
    Viro3DAnnotationExpansion,
    get_annotation_expansion,
)
from vhold.databases.metagenomics import (
    SerratusClient,
    SerratusMatch,
    MetagenomicContext,
    DarkMatterCharacterization,
    characterize_dark_matter_batch,
)
from vhold.databases.esm_atlas import (
    ESMAtlasHit,
    ESMAtlasSearchResult,
    ESMAtlasAnalysis,
    check_esmatlas_database,
    install_esmatlas_database,
    search_esmatlas,
    parse_esmatlas_results,
    analyze_dark_matter_with_esmatlas,
)

__all__ = [
    "install_databases",
    "check_databases",
    "load_bfvd_metadata",
    "load_viro3d_metadata",
    "Viro3DAnnotationExpansion",
    "get_annotation_expansion",
    "UniProtAnnotationCache",
    "fetch_uniprot_batch",
    # Metagenomic integration
    "SerratusClient",
    "SerratusMatch",
    "MetagenomicContext",
    "DarkMatterCharacterization",
    "characterize_dark_matter_batch",
    # ESM Atlas integration
    "ESMAtlasHit",
    "ESMAtlasSearchResult",
    "ESMAtlasAnalysis",
    "check_esmatlas_database",
    "install_esmatlas_database",
    "search_esmatlas",
    "parse_esmatlas_results",
    "analyze_dark_matter_with_esmatlas",
]
