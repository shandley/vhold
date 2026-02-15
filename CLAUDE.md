# vHold Project State - Claude Code Context

Last updated: 2026-02-14

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
4a.5. (Auto) MLP classifier: reclassify "unknown" proteins using trained embedding classifier
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
| `src/vhold/features/classifier.py` | MLP functional classifier (model definition + batch inference) |
| `src/vhold/databases/classifier.py` | Classifier checkpoint path management |
| `src/vhold/utils/constants.py` | Thresholds, DB URLs, embedding config |
| `src/vhold/cli.py` | Click CLI (run, predict, compare, install, align) |
| `scripts/calibrate_triage_threshold.py` | Triage threshold calibration across 4 case studies |
| `scripts/train_classifier.py` | MLP classifier training with stratified split + class weights |
| `scripts/generate_llm_labels.py` | Batch LLM label generation for training data expansion |
| `tests/` | 378 tests (pytest) |

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

### MLP Functional Classifier (`--classify`) -- COMPLETE

Lightweight MLP trained on frozen ProstT5 encoder embeddings to classify viral proteins into functional categories. Runs automatically in the pipeline (Step 4a.5) for any protein classified as "unknown" by keyword/Pfam/GO matching.

**Status**: Code complete, model trained, integrated into pipeline. Classifier auto-runs when model installed; disable with `--no-classify`.

| Metric | Value |
|--------|-------|
| Architecture | MLP (1024 -> 512 -> 256 -> 11) with LayerNorm, ReLU, Dropout(0.3) |
| Parameters | 660,491 |
| Training samples | 84,250 (from 436K total, excluding 343K "unknown") |
| Validation macro F1 | 0.692 |
| Validation accuracy | 78.8% |
| Training time | 62s on CPU |
| Checkpoint size | 2.6 MB |
| Location | `~/.vhold/models/classifier/vhold_classifier.pt` |

**Training label sources**: Keyword/Pfam/GO annotations (69K proteins) + agreement-filtered LLM labels (15K additional proteins where both text-based LLM classification and structural model predictions agreed).

**Per-category performance** (validation F1):

| Category | F1 | Support | Notes |
|----------|:--:|:--:|-------|
| replication | 0.863 | 3,681 | Best performer |
| structural | 0.823 | 5,262 | High precision (93.1%) |
| nuclease | 0.809 | 854 | |
| protease | 0.783 | 666 | |
| lysis | 0.722 | 185 | |
| regulatory | 0.701 | 1,089 | |
| host_interaction | 0.669 | 231 | |
| packaging | 0.568 | 206 | |
| entry | 0.511 | 393 | Low precision (36.8%) |
| movement | 0.477 | 65 | Rarest category |

**Training iterations and key findings**:
- **v1** (keyword/Pfam/GO labels only, 69K samples): Macro F1 0.635. Baseline model.
- **v2** (+ raw LLM labels, 118K samples): Macro F1 0.531 -- **worse**. LLM labels are text-based but MLP learns from structural embeddings. Generic domain descriptions (ankyrin, F-box) don't reliably predict structural features.
- **v3** (+ agreement-filtered LLM labels, 84K samples): Macro F1 **0.692** -- best. Only LLM labels where v1 structural model independently agreed were kept (33.5% of raw LLM labels). Every category improved over v1.

**Agreement filtering insight**: The key finding is that text-based labels and structure-based predictions are complementary signals. Labels where both agree are high-quality; labels where they disagree are noisy. This principle could be applied iteratively (use v3 as the filter for another round).

**LLM label generation** (`scripts/generate_llm_labels.py`):
- Identifies "unknown" proteins with informative descriptions (not "hypothetical", "uncharacterized", etc.)
- De-duplicates by description (32,312 unique from 90,747 proteins)
- Batch-classifies 40 descriptions per Claude Haiku API call
- 808 API calls, $2.52 cost, 38 minutes, zero errors
- 14,793 non-unknown descriptions (45.8% hit rate) -> 55,941 proteins labeled

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

- **Embedding DB not downloadable**: `EMBEDDING_DB_URL = None` in `databases/embeddings.py`. Users must generate locally via `scripts/build_embedding_db.py` or place manually at `~/.vhold/databases/embeddings/vhold_embeddings.npz`.
- **Enriched BFVD metadata not downloadable**: Must be copied manually from `bfvd-annotations` repo to `~/.vhold/databases/bfvd/bfvd_metadata_enriched.tsv`.
- **Classifier model not downloadable**: Must be trained locally via `scripts/train_classifier.py` or placed manually at `~/.vhold/models/classifier/vhold_classifier.pt`.

## Remaining Work

### Immediate (before release)
- **Host embedding DB**: Upload 822 MB .npz to Zenodo/S3, set `EMBEDDING_DB_URL` in `databases/embeddings.py`
- **Host enriched BFVD metadata**: Bundle with embedding DB or host separately, wire into `vhold install`
- **Host classifier model**: Bundle 2.6 MB checkpoint with embedding DB or host separately
- **Tests for triage + LLM + classifier integration**: End-to-end test with `--triage --classify --llm-classify`

### Medium-term
- **Metagenomic pipeline integration**: Accept VirSorter2/VIBRANT/geNomad output, produce DRAM-v/anvi'o compatible annotations
- **ONNX INT8 quantization**: ~3x CPU speedup for ProstT5 decoder (no retraining needed)
- **Iterative label refinement**: Use v3 classifier as filter for another round of LLM label agreement filtering
- **Batch processing improvements**: Batch same-length sequences for ProstT5

### Research directions
- **Contrastive fine-tuning of ProstT5 encoder**: Learn viral-specific embeddings (would improve both triage and classifier)
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

# Disable MLP classifier (enabled by default when model installed)
uv run vhold run -i input.fasta -o output/ --no-classify

# Adjust classifier confidence threshold (default: 0.5)
uv run vhold run -i input.fasta -o output/ --classifier-confidence 0.8

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

# Train MLP classifier
uv run python scripts/train_classifier.py --output /tmp/classifier_output --install

# Train with LLM-expanded labels
uv run python scripts/train_classifier.py --output /tmp/classifier_output --extra-labels results/llm_labels_filtered.json --install

# Generate LLM training labels (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... uv run python scripts/generate_llm_labels.py --output results/llm_labels.json
```

## Contact

This project documentation maintained for Claude Code session continuity.
