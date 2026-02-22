# vHold Case Studies

This directory contains worked examples demonstrating vHold's annotation capabilities.

Previous results (run at different development stages with different feature sets) are archived in `_archive_2026-02-21/`. All case studies are being re-run with the full integrated pipeline.

## Case Study Overview

| # | Name | Purpose | Proteins | Status |
|---|------|---------|----------|--------|
| 1 | [SARS-CoV-2](sars_cov_2/) | Pipeline validation with gold-standard proteome | 18 | Pending re-run |
| 2 | [Remote Homology](remote_homology/) | Annotation at low sequence identity | 10 | Pending re-run |
| 3 | [Metagenomic Dark Matter](metagenomic_dark_matter/) | Annotate truly unknown proteins | 30 | Pending re-run |
| 4 | [crAssphage ORFans](crass_phage_orfans/) | Unknown gut phage proteins | 37 | Deprioritized |
| 5 | [Eukaryotic Viruses](eukaryotic_viruses/) | Mammalian virus annotation (vHold focus) | 27 | Pending re-run |
| 6 | [T7 Phage](t7_phage/) | Well-characterized phage proteome | 60 | Pending re-run |

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
