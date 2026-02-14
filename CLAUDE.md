# vHold Project State - Claude Code Context

Last updated: 2026-02-07

## Project Overview

**vHold** is a viral protein annotation tool using structural homology (ProstT5 + Foldseek). It extends pHold's phage-focused capabilities to include eukaryotic viruses.

### Key Differentiator
- **pHold**: Targets bacteriophages
- **vHold**: Targets ALL viruses including mammalian/eukaryotic viruses

## Architecture

```
vHold Pipeline:
1. Read FASTA sequences
1b. (Optional) Embedding triage: ProstT5 encoder cosine similarity against 436K references
    - Matched proteins: skip decoder, transfer annotation directly
    - Unmatched proteins: continue to step 2
2. Predict 3Di with ProstT5 decoder (--fast for greedy, default beam search)
3. Search BFVD + Viro3D with Foldseek
4. Transfer annotations with multi-database consensus scoring
4a. Merge triage results with structural search results
4b. (Optional) LLM reclassification of unknown proteins (--llm-classify)
5. Generate output: TSV, summary JSON, dark matter report
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/vhold/features/prostt5.py` | 3Di prediction (ProstT5 model, MPS/CUDA/CPU) |
| `src/vhold/features/embeddings.py` | Embedding triage (EmbeddingExtractor, EmbeddingDatabase, triage_proteins) |
| `src/vhold/features/foldseek.py` | Structural search against BFVD/Viro3D + `create_query_db()` |
| `src/vhold/features/foldmason.py` | FoldMason MSA wrapper (`run_foldmason_msa()`) |
| `src/vhold/subcommands/run.py` | Main pipeline orchestration |
| `src/vhold/subcommands/align.py` | Multiple structural alignment pipeline |
| `src/vhold/databases/bfvd.py` | BFVD metadata loading + enriched annotation lookup |
| `src/vhold/databases/embeddings.py` | Embedding DB path/install (URL not yet configured) |
| `src/vhold/results/annotations.py` | Consensus annotation transfer |
| `src/vhold/results/categories.py` | Keyword-based functional classification |
| `src/vhold/results/llm_classify.py` | LLM-based functional classification (keywords + embedding unknowns) |
| `src/vhold/utils/constants.py` | Thresholds, DB URLs, embedding config |
| `src/vhold/cli.py` | Click CLI (run, predict, compare, install, align) |
| `scripts/calibrate_triage_threshold.py` | Triage threshold calibration across 4 case studies |
| `tests/` | 360 tests (pytest) |

## Implemented Features

### Core Pipeline
- ProstT5 3Di prediction with per-residue confidence scoring
- Foldseek structural search against BFVD (351K structures) and Viro3D (85K structures)
- Multi-database consensus scoring with Viro3D 1.2x quality bonus
- Functional classification via Pfam, GO, SUPERFAMILY, and keyword matching
- Dark matter analysis for unannotated proteins
- Novelty classification (database_match, close_homolog, remote_homolog, twilight_zone)

### Embedding-Based Triage (`--triage`) -- COMPLETE

Fast pre-screening using ProstT5 encoder-only embeddings (~0.1s/protein) to skip the expensive decoder for known proteins. End-to-end wired in CLI and pipeline.

**Status**: Code complete, DB built, threshold calibrated. Only missing piece: hosting URL for `vhold install --embeddings`.

| Metric | Value |
|--------|-------|
| Reference proteins | 436,237 (351K BFVD + 85K Viro3D) |
| Embedding dimensions | 1,024 (float16, L2 normalized) |
| File size | 822 MB |
| Search latency | 59ms for 436K proteins (brute-force cosine) |
| Default threshold | 0.90 (calibrated) |
| Location | `~/.vhold/databases/embeddings/vhold_embeddings.npz` |

**Triage calibration results** (85 proteins across 4 case studies):

| Configuration | Precision | F1 | Notes |
|--------------|-----------|------|-------|
| Bare embeddings | 29.4% | - | Before enriched metadata |
| + Enriched BFVD metadata | 67.1% | 0.803 | 345K UniProt annotations |
| + Keyword fixes | 77.6% | 0.874 | Structural, protease, regulatory terms |
| + LLM reclassification | **83.5%** | **0.910** | Claude Haiku resolves ORF/VP unknowns |

At threshold 0.90: 100% recall (all 85 test proteins matched, min similarity 0.904). Remaining 14 false positives are annotation quality issues (deleted UniProt entries, UniParc-only IDs, ground truth category debates).

**Enriched BFVD metadata**: `~/.vhold/databases/bfvd/bfvd_metadata_enriched.tsv` (79 MB, 345,730 entries). Contains protein names, gene names, GO BP/MF, Pfam domains, keywords, function_cc, lineage from UniProt REST API. Generated in separate `bfvd-annotations` repo.

### LLM Classification (`--llm-classify`)
- Post-processing step using Claude to reclassify proteins where keywords return "unknown"
- Targets `classification_source in ("keywords", "embedding")` AND `functional_category == "unknown"`
- Resolves: SARS-CoV-2 accessory proteins (ORF3a/6/7a/8), paramyxovirus V/C proteins, Ebola VP35, Rabies M2
- 3-layer graceful degradation: no anthropic package, no API key, per-protein API errors
- Default model: `claude-haiku-4-5-20251001` (configurable via `--llm-model`)
- Requires: `pip install vhold[llm]` + `ANTHROPIC_API_KEY` environment variable

### Fast Mode (`--fast`)
- Greedy decoding (`num_beams=1, do_sample=False`) vs default beam search (`num_beams=3, do_sample=True`)
- ~2x faster on MPS, ~3x faster theoretical maximum
- 94.8% 3Di identity vs beam search (tested on 173aa protein)

### Apple Silicon (MPS) GPU Acceleration
- MPS auto-detected on Apple Silicon via `--device auto` (default)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically for unsupported ops
- Previous instability fixed in transformers v4.43+
- Current stack: PyTorch 2.10.0 + transformers 5.0.0

**Benchmarks (Apple M4)**:
| Protein Size | CPU | MPS | Speedup |
|--------------|-----|-----|---------|
| 173aa | 88s | 40s | 2.2x |
| 435aa | 170s | 95s | 1.8x |
| 1135aa | ~132 min | 72 min | ~1.8x |

### `vhold align` -- Multiple Structural Alignment via FoldMason -- COMPLETE

Performs multiple structural alignment of viral protein families by bridging ProstT5 3Di predictions with FoldMason's alignment engine.

```bash
vhold align -i proteins.fasta -o alignment/ --device cpu --fast -t 8
```

**Pipeline**: Input FASTA -> ProstT5 3Di -> confidence masking -> Foldseek DB -> FoldMason `structuremsa` (fastMode) -> AA MSA + 3Di MSA + guide tree

## Completed Case Studies

| # | Name | Proteins | Results |
|---|------|----------|---------|
| 1 | SARS-CoV-2 | 18 | 55.6% annotated, 100% structural accuracy |
| 2 | Remote Homology | 10 | 100% annotated, 91.7% twilight zone hits |
| 3 | Metagenomic Dark Matter | 30 | 83.3% annotated, 70% RdRp classification |
| 4 | crAssphage ORFans | 37 | Setup only (deprioritized - phage focus) |
| 5 | Eukaryotic Viruses | 27 | 100% annotated, 81.5% consensus, 88.9% category accuracy (with LLM) |

## Known Issues

- **Triage threshold default mismatch**: Fixed in latest commit. CLI and run.py now both default to 0.90 (matching calibrated constant).
- **Embedding DB not downloadable**: `EMBEDDING_DB_URL = None` in `databases/embeddings.py`. Users must generate locally via `scripts/build_embedding_db.py` or place manually at `~/.vhold/databases/embeddings/vhold_embeddings.npz`.
- **Enriched BFVD metadata not downloadable**: Must be copied manually from `bfvd-annotations` repo to `~/.vhold/databases/bfvd/bfvd_metadata_enriched.tsv`.

## Remaining Work

### Immediate (before release)
- **Host embedding DB**: Upload 822 MB .npz to Zenodo/S3, set `EMBEDDING_DB_URL` in `databases/embeddings.py`
- **Host enriched BFVD metadata**: Bundle with embedding DB or host separately, wire into `vhold install`
- **Tests for triage + LLM integration**: End-to-end test with `--triage --llm-classify`

### Medium-term
- **Metagenomic pipeline integration**: Accept VirSorter2/VIBRANT/geNomad output, produce DRAM-v/anvi'o compatible annotations
- **ONNX INT8 quantization**: ~3x CPU speedup for ProstT5 decoder (no retraining needed)
- **Pre-computed 3Di cache**: Common viral reference proteins pre-computed for instant search
- **Batch processing improvements**: Batch same-length sequences for ProstT5

### Research directions
- **Structural distance-based viral taxonomy**: Pairwise structural distances for phylogenetics below twilight zone
- **BitNet 1.58-bit quantization-aware retraining**: Ternary ProstT5 for 3-6x CPU speedup
- **Metagenomic viral protein detection**: Repurpose embedding DB as fast structural-similarity-based viral protein detector

## Commands Reference

```bash
# Run vHold (auto-selects MPS on Apple Silicon, CUDA on Linux)
uv run vhold run -i input.fasta -o output/ -t 8

# With embedding triage (skip decoder for known proteins)
uv run vhold run -i input.fasta -o output/ --triage

# With LLM classification (requires anthropic package + API key)
uv run vhold run -i input.fasta -o output/ --triage --llm-classify

# Fast mode (greedy decoding, ~2x faster)
uv run vhold run -i input.fasta -o output/ --fast

# Two-step workflow (GPU predict, CPU search)
uv run vhold predict -i input.fasta -o predictions/ --device cuda
uv run vhold compare -p predictions/ -o results/ -t 32

# Multiple structural alignment
uv run vhold align -i proteins.fasta -o alignment/ --fast --device cpu -t 8

# Database setup
uv run vhold install

# Run tests
uv run pytest tests/ -v

# Calibrate triage threshold
uv run python scripts/calibrate_triage_threshold.py --device cpu --llm-classify --output results.tsv
```

## Contact

This project documentation maintained for Claude Code session continuity.
