"""Constants and configuration for vhold."""

import os
from pathlib import Path

# Environment variable overrides
ENV_VHOLD_DATA = "VHOLD_DATA_DIR"
ENV_VHOLD_DB = "VHOLD_DB_DIR"
ENV_VHOLD_MODEL = "VHOLD_MODEL_DIR"

# Default directories
DEFAULT_DATA_DIR = Path.home() / ".vhold"
DEFAULT_DB_DIR = DEFAULT_DATA_DIR / "databases"
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models"


def get_data_dir() -> Path:
    """Get the vhold data directory, respecting environment variable."""
    env_path = os.environ.get(ENV_VHOLD_DATA)
    if env_path:
        return Path(env_path)
    return DEFAULT_DATA_DIR


def get_db_dir() -> Path:
    """Get the database directory, respecting environment variable."""
    env_path = os.environ.get(ENV_VHOLD_DB)
    if env_path:
        return Path(env_path)
    return get_data_dir() / "databases"


def get_model_dir() -> Path:
    """Get the model directory, respecting environment variable."""
    env_path = os.environ.get(ENV_VHOLD_MODEL)
    if env_path:
        return Path(env_path)
    return get_data_dir() / "models"


# Convenient constants (evaluated at import time)
VHOLD_DATA_DIR = get_data_dir()
VHOLD_DB_DIR = get_db_dir()
VHOLD_MODEL_DIR = get_model_dir()

# Database URLs and files
BFVD_BASE_URL = "https://bfvd.steineggerlab.workers.dev/latest"
BFVD_FILES = {
    "foldseekdb": "bfvd_foldseekdb.tar.gz",  # 533.8 MB
    "metadata": "bfvd_metadata.tsv",  # 14.6 MB
    "taxid": "bfvd_taxid.tsv",  # 24.5 MB
}

VIRO3D_BASE_URL = "https://zenodo.org/records/15622906/files"
VIRO3D_FILES = {
    "foldseekdb": "foldseekViro3D.tar.gz",  # 568.7 MB
    "metadata": "viro3d_metadata.tar.gz",  # 18.5 MB
    "annotations": "viro3d_annotation_expansion.tar.gz",  # 12.3 MB
}

# ProstT5 model
PROSTT5_MODEL_NAME = "Rostlab/ProstT5"

# Foldseek output format columns
# Note: prob, lddt, qtmscore, etc. require C-alpha coordinates which we don't have
# when using pre-computed 3Di sequences from ProstT5
FOLDSEEK_OUTPUT_COLUMNS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
    "qlen",
    "tlen",
    "qcov",
    "tcov",
]

FOLDSEEK_OUTPUT_FORMAT = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,qcov,tcov"

# 3Di alphabet
THREE_DI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Confidence thresholds for annotation quality
CONFIDENCE_HIGH = 0.9
CONFIDENCE_MEDIUM = 0.7
CONFIDENCE_LOW = 0.5

# Default parameters
DEFAULT_EVALUE = 1e-3
DEFAULT_SENSITIVITY = 9.5
DEFAULT_MAX_SEQS = 1000
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Embedding triage
# Calibrated via scripts/calibrate_triage_threshold.py across 85 proteins
# in 4 case studies. Precision is ~78% at any threshold (limited by annotation
# quality, not match quality). Threshold 0.90 gives 100% recall with min
# similarity of 0.904 across all test proteins.
DEFAULT_TRIAGE_THRESHOLD = 0.90
EMBEDDING_DIM = 1024
EMBEDDING_DB_FILE = "vhold_embeddings.npz"
