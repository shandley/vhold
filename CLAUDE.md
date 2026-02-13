# vHold Project State - Claude Code Context

Last updated: 2026-02-08

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
| `src/vhold/features/foldseek.py` | Structural search against BFVD/Viro3D |
| `src/vhold/subcommands/run.py` | Main pipeline orchestration |
| `src/vhold/results/annotations.py` | Consensus annotation transfer |
| `src/vhold/results/categories.py` | Keyword-based functional classification |
| `src/vhold/results/llm_classify.py` | LLM-based functional classification |
| `src/vhold/cli.py` | Click CLI (run, predict, compare, install) |
| `tests/` | 334 tests (pytest) |

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

## Embedding-Based Triage (`--triage`)

Fast pre-screening using ProstT5 encoder-only embeddings (~0.1s/protein) to skip the slow decoder for known proteins. Compares query embeddings via cosine similarity against pre-computed reference DB of all BFVD+Viro3D proteins (~436K).

### Embedding Database Generation
- **Status**: 412,190 / 436,237 proteins (94.5%) — remaining proteins are longest (800-2048aa)
- **Rate**: ~0.3-0.5 prot/s for long proteins on CPU (vs ~10 prot/s for short)
- **Resume command**:
  ```bash
  caffeinate -i uv run python scripts/build_embedding_db.py \
    --output ~/.vhold/databases/embeddings/vhold_embeddings.npz \
    --device cpu --resume --checkpoint-interval 1000
  ```

### Future Application: Metagenomic Viral Protein Detection
The same embedding DB could be repurposed as a fast structural-similarity-based viral protein detector for metagenomics:
- Encode ORFs from metagenomic contigs (~0.1s/protein)
- Cosine similarity vs 436K viral reference embeddings (~100ms)
- High similarity → likely viral protein
- Advantages: catches remote homologs that BLAST/HMMs miss (structural dark matter), extremely fast
- Caveats: shared folds (jelly rolls, TIM barrels) could cause false positives; needs negative control calibration with bacterial/archaeal proteomes; best as pre-filter alongside contig-level tools (VirSorter, VIBRANT, geNomad)
- Potential standalone tool or vHold module; benchmark on mock metagenome to measure precision/recall at different thresholds

## FoldMason Integration (Planned)

**FoldMason** (Steinegger lab, *Science* 2026) does multiple protein structure alignment at scale — 700x faster than MUSTANG. Same lab that makes Foldseek and ProstT5.

### Key Discovery: Already Compatible

FoldMason's `structuremsa` command accepts Foldseek-format databases. When C-alpha coordinates (`_ca.dbtype`) are absent, it automatically enables `fastMode` — pure 3Di+AA string alignment without coordinates.

**vHold already creates exactly this database format** in `create_query_db()` (AA + 3Di + headers, no coordinates). So the bridge between ProstT5 and FoldMason essentially already exists.

### What We Can Build

**`vhold align`** — Multiple structural alignment of viral protein families:
1. Take a set of viral proteins (from FASTA or from vHold search results)
2. Predict 3Di with ProstT5 (already implemented)
3. Create a Foldseek database (already implemented in `create_query_db()`)
4. Run `foldmason structuremsa` on the database (will use fastMode)
5. Output: AA alignment FASTA, 3Di alignment FASTA, guide tree (Newick)

**Use cases**:
- Align a viral protein family to find conserved structural motifs
- Structure-based phylogenetics below the twilight zone (demonstrated in FoldMason paper on Flaviviridae)
- Compare novel viral proteins against their closest structural relatives
- Identify variable vs conserved regions in viral protein families

### Trade-offs of fastMode (no coordinates)

- No TM-score validation of alignments
- No LDDT-based alignment quality scoring
- Alignment driven purely by 3Di + AA similarity
- Still produces valid multiple alignments — just without coordinate-based refinement

### Alternative: Full Mode via Foldseek's ProstT5 Integration

Foldseek's `createdb --prostt5-model` can generate predicted C-alpha coordinates alongside 3Di, enabling FoldMason's full mode with LDDT scoring. Trade-off: uses Foldseek's internal ProstT5 (greedy only, no confidence masking) vs vHold's enhanced ProstT5 (beam search, confidence scoring).

### Installation

```bash
# FoldMason is distributed as a standalone binary (like Foldseek)
# Install via conda or from GitHub releases
conda install -c bioconda foldmason
```

## Remaining Work

- **Dengue polyprotein**: Sequence data corrupted (API rate limit error in FASTA). Needs re-fetch.
- **MERS-CoV spike**: 7078aa, requires GPU or many hours CPU to process.
- **Sequence chunking**: For very long proteins (>2000aa), chunking could make processing feasible on CPU.
- **Pre-computed 3Di cache**: Common viral reference proteins could be pre-computed.
- **Batch processing improvements**: Current batch_size=1 default; batching same-length sequences could help.
- **FoldMason `vhold align` subcommand**: Multiple structural alignment via ProstT5 → FoldMason bridge.

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

# Database setup
uv run vhold install

# Run tests
uv run pytest tests/ -v
```

## Contact

This project documentation maintained for Claude Code session continuity.
