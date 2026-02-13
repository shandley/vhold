# vHold Project State - Claude Code Context

Last updated: 2026-02-13

## Project Overview

**vHold** is a viral protein annotation tool using structural homology (ProstT5 + Foldseek). It extends pHold's phage-focused capabilities to include eukaryotic viruses.

### Key Differentiator
- **pHold**: Targets bacteriophages
- **vHold**: Targets ALL viruses including mammalian/eukaryotic viruses

## Architecture

```
vHold Pipeline:
1. Read FASTA sequences
2. Predict 3Di with ProstT5 (--fast for greedy decoding, default beam search)
3. Search BFVD + Viro3D with Foldseek
4. Transfer annotations with multi-database consensus scoring
4b. (Optional) LLM reclassification of unknown proteins (--llm-classify)
5. Generate output: TSV, summary JSON, dark matter report
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/vhold/features/prostt5.py` | 3Di prediction (ProstT5 model, MPS/CUDA/CPU) |
| `src/vhold/features/foldseek.py` | Structural search against BFVD/Viro3D + `create_query_db()` |
| `src/vhold/features/foldmason.py` | FoldMason MSA wrapper (`run_foldmason_msa()`) |
| `src/vhold/subcommands/run.py` | Main pipeline orchestration |
| `src/vhold/subcommands/align.py` | Multiple structural alignment pipeline |
| `src/vhold/results/annotations.py` | Consensus annotation transfer |
| `src/vhold/results/categories.py` | Keyword-based functional classification |
| `src/vhold/results/llm_classify.py` | LLM-based functional classification |
| `src/vhold/utils/external.py` | External tool wrappers (Foldseek, FoldMason) |
| `src/vhold/cli.py` | Click CLI (run, predict, compare, install, align) |
| `tests/` | 360 tests (pytest) |

## Implemented Features

### Core Pipeline
- ProstT5 3Di prediction with per-residue confidence scoring
- Foldseek structural search against BFVD (351K structures) and Viro3D (85K structures)
- Multi-database consensus scoring with Viro3D 1.2x quality bonus
- Functional classification via Pfam, GO, SUPERFAMILY, and keyword matching
- Dark matter analysis for unannotated proteins
- Novelty classification (database_match, close_homolog, remote_homolog, twilight_zone)

### Fast Mode (`--fast`)
- Greedy decoding (`num_beams=1, do_sample=False`) vs default beam search (`num_beams=3, do_sample=True`)
- ~2x faster on MPS, ~3x faster theoretical maximum
- 94.8% 3Di identity vs beam search (tested on 173aa protein)
- Recommended for well-characterized proteins; not recommended for remote homology searches

**Benchmarks (Apple M4, MPS)**:
| Protein Size | Standard | Fast | Speedup |
|--------------|----------|------|---------|
| 173aa | 40s | 19s | 2.1x |

### LLM Classification (`--llm-classify`)
- Post-processing step using Claude to reclassify proteins where keywords return "unknown"
- Resolves: paramyxovirus V/C proteins, Ebola VP35, MERS ORF4b, and other generic descriptions
- 3-layer graceful degradation: no anthropic package, no API key, per-protein API errors
- Only targets `classification_source == "keywords"` AND `functional_category == "unknown"`
- Default model: `claude-haiku-4-5-20251001` (configurable via `--llm-model`)
- Requires: `pip install vhold[llm]` + `ANTHROPIC_API_KEY` environment variable

### Apple Silicon (MPS) GPU Acceleration
- MPS auto-detected on Apple Silicon via `--device auto` (default)
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically for unsupported ops
- Previous instability (system lockups) was caused by a T5 MPS bug fixed in transformers v4.43+
- Current stack: PyTorch 2.10.0 + transformers 5.0.0

**Benchmarks (Apple M4, beam search)**:
| Protein Size | CPU | MPS | Speedup |
|--------------|-----|-----|---------|
| 173aa | 88s | 40s | 2.2x |
| 435aa | 170s | 95s | 1.8x |
| 1135aa | ~132 min | 72 min | ~1.8x |

### CPU Performance Characteristics

ProstT5 has **O(n^2) scaling** with sequence length due to autoregressive generation:

| Protein Size | Time (CPU, Apple M4) |
|--------------|---------------------|
| <300aa | 1-4 min |
| 300-600aa | 3-10 min |
| 600-800aa | 10-20 min |
| 1135aa | ~2h 12min |
| 1248aa | ~4-5 hours |

GPU acceleration (MPS or CUDA) is strongly recommended for proteins >500aa.

### Embedding-Based Triage (`--triage`)

Fast pre-screening using ProstT5 encoder-only embeddings (~0.1s/protein) to skip the slow decoder for known proteins. Compares query embeddings via cosine similarity against pre-computed reference DB.

### Embedding Database — COMPLETE

**Status**: Generation complete (2026-02-13)

| Metric | Value |
|--------|-------|
| Total proteins | 436,237 |
| BFVD proteins | 351,242 |
| Viro3D proteins | 84,995 |
| Embedding dimensions | 1,024 |
| Storage format | float16, L2 normalized |
| File size | 822 MB |
| Location | `~/.vhold/databases/embeddings/vhold_embeddings.npz` |
| Generation time | 16.2 hours (Apple M4 CPU) |

**Validation results**:
- Zero NaN/Inf values, zero duplicate IDs, zero near-zero vectors
- All norms within 0.9998-1.0002 (properly L2 normalized)
- 2/1024 zero-variance dimensions (negligible)
- Brute-force cosine similarity search: **59ms** over all 436K proteins
- Nearest neighbor results are biologically sensible (related proteins cluster)

### `vhold align` — Multiple Structural Alignment via FoldMason — COMPLETE

**Status**: Implemented and tested (2026-02-13)

Performs multiple structural alignment of viral protein families by bridging ProstT5 3Di predictions with FoldMason's alignment engine. This implements what the FoldMason authors listed as a future feature (direct ProstT5 input).

```bash
# Full pipeline (ProstT5 prediction + FoldMason alignment)
vhold align -i proteins.fasta -o alignment/ --device cpu --fast -t 8

# Two-step workflow (reuse existing predictions)
vhold predict -i proteins.fasta -o predictions/
vhold align -i proteins.fasta --predictions-dir predictions/ -o alignment/
```

**Pipeline**: Input FASTA → ProstT5 3Di prediction → confidence masking → Foldseek DB (`create_query_db()`) → FoldMason `structuremsa` (fastMode) → AA MSA + 3Di MSA + guide tree

**Key discovery**: FoldMason auto-enables fastMode when `_ca.dbtype` (C-alpha coordinates) is absent. vHold's `create_query_db()` already creates exactly this database format (AA + 3Di + headers, no coordinates). Zero modifications to existing code were needed.

**Output files**:
- `alignment_aa.fa` — Amino acid MSA (FASTA)
- `alignment_3di.fa` — 3Di structural alphabet MSA (FASTA)
- `alignment.nw` — Guide tree (Newick format)
- `summary.json` — Alignment metadata
- `predictions/` — ProstT5 intermediates (AA, masked 3Di, confidence scores)

**Tested on**: 4 viral nucleoproteins (Nipah 166aa, Hendra 166aa, Rabies 202aa, MERS-CoV N 224aa). Guide tree correctly groups Nipah+Hendra as sister paramyxoviruses. ~4 min ProstT5 + <1s FoldMason.

**Refinement limitation**: FoldMason's refinement (`--refine-iters`, `refinemsa`) requires C-alpha coordinates for LDDT scoring. Not compatible with ProstT5 coordinate-free mode — causes segfault. The `--refine-iters` CLI flag is accepted but logs a warning and is ignored. If refinement is ever needed, options are: (1) use AlphaFold/ESMFold predicted structures with FoldMason's `easy-msa`, or (2) use Foldseek's `--prostt5-model` flag to generate approximate coordinates alongside 3Di.

**New files**: `features/foldmason.py`, `subcommands/align.py`, `tests/test_align_cli.py`
**Modified files**: `cli.py`, `utils/external.py`
**Reused (unchanged)**: `features/prostt5.py`, `features/foldseek.py`, `features/confidence.py`, `io/fasta.py`, `utils/external.py` (ExternalTool class)

## Completed Case Studies

| # | Name | Proteins | Results |
|---|------|----------|---------|
| 1 | SARS-CoV-2 | 18 | 55.6% annotated, 100% structural accuracy |
| 2 | Remote Homology | 10 | 100% annotated, 91.7% twilight zone hits |
| 3 | Metagenomic Dark Matter | 30 | 83.3% annotated, 70% RdRp classification |
| 4 | crAssphage ORFans | 37 | Setup only (deprioritized - phage focus) |
| 5 | Eukaryotic Viruses | 27 | 100% annotated, 81.5% consensus, 40.7% keyword category accuracy (improved with --llm-classify) |

### Case Study 5 Notes
- 27/27 proteins annotated (100% annotation rate)
- 25/27 high confidence, 81.5% multi-DB consensus
- Keyword classification: 40.7% category accuracy (generic descriptions fail for V/C/VP35 etc.)
- With `--llm-classify`: all 7 previously-unknown proteins correctly reclassified
- 2 very large proteins skipped: MERS spike (7078aa), Dengue polyprotein (3478aa)
- ~10.5 hours wall time on Apple M4 CPU

## Remaining Work

### Immediate
- **Embedding triage threshold calibration**: The embedding DB is built; need to determine optimal cosine similarity threshold for annotation transfer vs full structural search
- **Wire `--triage` end-to-end**: Connect embedding search results to annotation transfer pipeline
- **Dengue polyprotein**: Sequence data corrupted (API rate limit error in FASTA). Needs re-fetch.
- **MERS-CoV spike**: 7078aa, requires GPU or many hours CPU to process.

### Medium-term (from ROADMAP.md)
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

# Fast mode (greedy decoding, ~2x faster)
uv run vhold run -i input.fasta -o output/ --fast

# With LLM classification (requires anthropic package + API key)
uv run vhold run -i input.fasta -o output/ --llm-classify

# Two-step workflow (GPU predict, CPU search)
uv run vhold predict -i input.fasta -o predictions/ --device cuda
uv run vhold compare -p predictions/ -o results/ -t 32

# Multiple structural alignment
uv run vhold align -i proteins.fasta -o alignment/ --fast --device cpu -t 8

# Align from pre-computed predictions
uv run vhold align -i proteins.fasta --predictions-dir predictions/ -o alignment/

# Database setup
uv run vhold install

# Run tests
uv run pytest tests/ -v
```

## Contact

This project documentation maintained for Claude Code session continuity.
