# vHold Case Studies

This directory contains worked examples demonstrating vHold's phage annotation capabilities, with focus on FANTASIA-style GO term transfer from well-characterized to poorly-characterized phages.

Previous results (run at different development stages with different feature sets) are archived in `_archive_2026-02-21/`.

## Case Study Overview

| # | Name | Purpose | Proteins | Priority | Status |
|---|------|---------|----------|:--------:|--------|
| 1 | [SARS-CoV-2](sars_cov_2/) | Pipeline validation baseline | 18 | Low | Has results |
| 2 | [Remote Homology](remote_homology/) | Annotation at low sequence identity | 10 | Medium | Needs rebuild (phage proteins) |
| 3 | [Metagenomic Dark Matter](metagenomic_dark_matter/) | Annotate unknown phage proteins | 30 | Medium | Needs rebuild (phage contigs) |
| 4 | [crAssphage ORFans](crass_phage_orfans/) | Unknown gut phage proteins — GO transfer test | 37 | **High** | Pending re-run |
| 5 | [Eukaryotic Viruses](eukaryotic_viruses/) | Eukaryotic virus annotation | 27 | Low | Deprioritized |
| 6 | [T7 Phage](t7_phage/) | Gold-standard phage proteome | 60 | **High** | Has results |

## Novelty Classification

Not all hits are equal. vHold classifies hits by "novelty" - how much value the structural search provides:

| Identity | Classification | Interpretation | BLAST Status |
|----------|----------------|----------------|--------------|
| >95% | `database_match` | Same protein, different DB entry | Would find |
| 70-95% | `close_homolog` | Related strain/variant | Would find |
| 30-70% | `remote_homolog` | Structure-based functional transfer | Marginal |
| <30% | `twilight_zone` | Novel structural similarity | **Fails** |

**Real value**: `remote_homolog` and `twilight_zone` hits represent annotations that BLAST cannot provide.

## Running Case Studies

```bash
# Full pipeline with all features
vhold run -i input.fasta -o results/ --triage --fast -t 8

# With gene calling from nucleotide contigs
vhold run -i contigs.fna -o results/ --gene-caller auto --triage --fast -t 8
```

## Creating New Case Studies

Each case study should include:

1. **Input FASTA** - Protein sequences to annotate
2. **Ground truth** - Known functions for evaluation (JSON format)
3. **Results** - Output from `vhold run`
4. **Documentation** - README with methods, results, and interpretation
