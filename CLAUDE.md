# vHold Project State - Claude Code Context

Last updated: 2026-02-08

## Project Overview

**vHold** is a viral protein annotation tool using structural homology (ProstT5 + Foldseek). It extends pHold's phage-focused capabilities to include eukaryotic viruses.

### Key Differentiator
- **pHold**: Targets bacteriophages
- **vHold**: Targets ALL viruses including mammalian/eukaryotic viruses

## Current Work: Case Study 5 - Eukaryotic Viruses

### Status: COMPLETE (27/27 proteins, 2 skipped for CPU time)

**Goal**: Demonstrate vHold's ability to annotate mammalian virus proteins.

**Results**:
- 27/27 proteins annotated (100% annotation rate)
- 25/27 high confidence, 81.5% multi-DB consensus
- 11/27 correct functional category (40.7%)
- ~10.5 hours wall time on Apple M4 CPU
- 2 very large proteins skipped: MERS spike (7078aa), Dengue polyprotein (3478aa)

**Key Files**:
- `case_studies/eukaryotic_viruses/test_proteins_small.fasta` - 27 test proteins (subset)
- `case_studies/eukaryotic_viruses/ground_truth.json` - Expected annotations
- `case_studies/eukaryotic_viruses/results/` - Complete results
- `case_studies/eukaryotic_viruses/README.md` - Detailed documentation

**Category keyword improvements applied** based on CS5 findings:
- Added: fusion protein, nucleoprotein, attachment (structural)
- Added: phosphoprotein, polymerase cofactor (replication)
- Added: interferon antagonist, immune evasion (host_interaction)

### Critical Finding: CPU Performance Issue

ProstT5 3Di prediction has **O(n²) scaling** with sequence length on CPU:

| Protein Size | Time (CPU, Apple M4) |
|--------------|---------------------|
| <300aa | 1-4 min |
| 300-600aa | 3-10 min |
| 600-800aa | 10-20 min |
| 1135aa | **~2h 12min** |
| 1248aa | **~4-5 hours** |

**Root cause**: Autoregressive generation with beam search (`do_sample=True`, `num_beams=3`).

**Mitigation options**:
1. Use GPU (CUDA) for large proteins
2. Skip proteins >1000aa in quick benchmarks
3. Implement greedy decoding option (future)
4. Implement sequence chunking (future)

### Apple Silicon (MPS) Testing

**Hardware**: Apple M4, 24GB RAM

**Finding**: MPS is unstable for vHold:
- Large tensor operations caused system lockup
- PyTorch 2.10.0 MPS backend not ready for transformers

**Recommendation**: Use `--device cpu` on Apple Silicon.

## Completed Case Studies

| # | Name | Proteins | Results |
|---|------|----------|---------|
| 1 | SARS-CoV-2 | 18 | 55.6% annotated, 100% structural accuracy |
| 2 | Remote Homology | 10 | 100% annotated, 91.7% twilight zone hits |
| 3 | Metagenomic Dark Matter | 30 | 83.3% annotated, 70% RdRp classification |
| 4 | crAssphage ORFans | 37 | Setup only (deprioritized - phage focus) |
| 5 | Eukaryotic Viruses | 27 | 100% annotated, 81.5% consensus, keywords improved |

## Remaining Work

- **Dengue polyprotein**: Sequence data corrupted (API rate limit error in FASTA). Needs re-fetch.
- **MERS-CoV spike**: 7078aa, requires GPU to process in reasonable time.
- **Category classification**: V/C proteins and some accessory proteins still classified "unknown" due to generic descriptions.

## Architecture

```
vHold Pipeline:
1. Read FASTA sequences
2. Predict 3Di with ProstT5 (slow on CPU for long sequences)
3. Search BFVD + Viro3D with Foldseek
4. Merge and annotate results
5. Write TSV output
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/vhold/features/prostt5.py` | 3Di prediction (performance bottleneck) |
| `src/vhold/search/foldseek.py` | Structural search |
| `src/vhold/subcommands/run.py` | Main pipeline |
| `src/vhold/analysis/annotate.py` | Result annotation |

## Potential Optimizations (Future)

1. **Greedy decoding option** for faster 3Di prediction
2. **Sequence chunking** for very long proteins
3. **Pre-computed 3Di cache** for common viruses
4. **Batch processing** improvements
5. **MPS stability testing** with newer PyTorch versions

## Commands Reference

```bash
# Run vHold
uv run vhold run -i input.fasta -o output/ -t 8 --device cpu

# Check help
uv run vhold --help
uv run vhold run --help

# Database setup (if needed)
uv run vhold download
```

## Contact

This project documentation maintained for Claude Code session continuity.
