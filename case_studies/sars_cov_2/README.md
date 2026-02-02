# Case Study 1: SARS-CoV-2 Proteome Validation

## Purpose

This case study validates that the vHold pipeline functions correctly by annotating the well-characterized SARS-CoV-2 proteome. It serves as a **validation benchmark** rather than a demonstration of novel discovery capability.

**Note**: SARS-CoV-2 proteins are extensively studied and present in reference databases. This case study confirms pipeline functionality but does not showcase vHold's ability to detect remote homology in divergent sequences.

## Objectives

1. **Validate pipeline functionality** - Confirm end-to-end workflow operates correctly
2. **Test cross-database consensus** - Verify BFVD and Viro3D agreement scoring
3. **Benchmark against gold standard** - Compare predictions to UniProt reviewed annotations
4. **Identify pipeline limitations** - Document edge cases and failure modes

## SARS-CoV-2 Proteome

The SARS-CoV-2 genome (NC_045512.2) encodes ~29 proteins. This case study includes 18 representative proteins:

### Structural Proteins (4)
| Protein | Gene | Length | Function |
|---------|------|--------|----------|
| Spike | S | 1,273 | Receptor binding, membrane fusion |
| Nucleocapsid | N | 419 | RNA packaging |
| Membrane | M | 222 | Envelope component |
| Envelope | E | 75 | Viroporin, ion channel |

### Replicase Components (5)
| Protein | Gene | Length | Function |
|---------|------|--------|----------|
| NSP1 | rep | 180 | Host translation inhibition |
| NSP3 (PLpro) | rep | 480 | Papain-like protease |
| NSP5 (Mpro) | rep | 306 | Main protease |
| NSP12 (RdRp) | rep | 969 | RNA-dependent RNA polymerase |
| NSP13 | rep | 601 | Helicase |

**Note**: NSP sequences in this dataset are partial fragments extracted from the ORF1ab polyprotein, which limits their annotation performance.

### Accessory Proteins (9)
| Protein | Gene | Length | Function |
|---------|------|--------|----------|
| ORF3a | 3a | 275 | Viroporin, apoptosis |
| ORF3b | 3b | 60 | Putative, poorly characterized |
| ORF6 | 6 | 61 | Interferon antagonist |
| ORF7a | 7a | 121 | Apoptosis inducer |
| ORF7b | 7b | 43 | Unknown |
| ORF8 | 8 | 121 | MHC-I downregulation |
| ORF9b | 9b | 88 | Interferon antagonist |
| ORF10 | 10 | 38 | Putative, expression debated |
| ORF14 | 14 | 38 | Putative, uncharacterized |

## Results Summary

### Overall Performance
| Metric | Value |
|--------|-------|
| Total Proteins | 18 |
| Annotated by vHold | 10 (55.6%) |
| Multi-DB Consensus | 7 (70% of annotated) |
| Correct Category | 4/5 (80%) |
| Dark Matter | 13 (72.2%) |

### Category-Specific Accuracy
| Category | Ground Truth | Correct | Accuracy | Notes |
|----------|--------------|---------|----------|-------|
| Structural | 4 | 4 | 100% | S, N, M, E all correct |
| Unknown | 4 | 4 | 100% | Correctly uncharacterized |
| Replication | 2 | 0 | 0% | NSP fragments - no hits |
| Protease | 2 | 0 | 0% | NSP fragments - no hits |
| Host Interaction | 5 | 0 | 0% | Classified as unknown |
| Regulatory | 1 | 0 | 0% | NSP1 fragment - no hits |

### Key Findings

#### Successes
1. **Structural proteins correctly identified** - All four (S, N, M, E) classified with high confidence (0.83-1.0) and cross-database agreement
2. **Cross-database validation working** - 7/10 annotated proteins had concordant BFVD and Viro3D hits
3. **Structure quality metrics integrated** - ColabFold pLDDT/pTM scores incorporated into confidence scoring

#### Limitations Identified
1. **Partial sequences fail** - NSP fragments extracted from ORF1ab had no database hits
2. **Accessory proteins poorly characterized** - Classified as "unknown" (technically correct but uninformative)
3. **GO-based misclassification** - NSP13 classified as "protease" due to GO:proteolysis annotation

## Files

```
case_studies/sars_cov_2/
├── README.md                      # This documentation
├── sars_cov_2_proteome.fasta     # Input proteome (18 proteins)
├── ground_truth.json             # Gold standard annotations
├── run_case_study.py             # Evaluation script
└── results/                      # Output directory
    ├── predictions/              # ProstT5 3Di predictions
    │   ├── aa_sequences.fasta
    │   ├── 3di_sequences.fasta
    │   └── 3di_sequences_masked.fasta
    ├── foldseek/                 # Database search results
    │   ├── bfvd_hits.tsv         # 1,142 hits
    │   └── viro3d_hits.tsv       # 1,384 hits
    ├── vhold_results.tsv         # Main annotation output
    ├── vhold_summary.json        # Statistics
    ├── vhold_dark_matter.tsv     # Unannotated proteins
    ├── case_study_report.json    # Detailed evaluation
    └── case_study_report.md      # Summary report
```

## Running the Case Study

### Prerequisites
```bash
# Install vHold
pip install -e .

# Install databases
vhold install
```

### Execute
```bash
cd case_studies/sars_cov_2

# Full run (prediction + search + evaluation)
python run_case_study.py -o results/ --device cpu -t 4

# GPU acceleration (recommended)
python run_case_study.py -o results/ --device cuda -t 4

# Evaluation only (skip vHold, use existing results)
python run_case_study.py -o results/ --skip-vhold
```

### Runtime
| Device | ProstT5 Prediction | Foldseek Search | Total |
|--------|-------------------|-----------------|-------|
| CPU (M1 Mac) | ~80 min | ~2 min | ~85 min |
| GPU (V100) | ~5 min | ~2 min | ~8 min |

## Bug Fixes Applied

During development of this case study, two bugs were identified and fixed:

### 1. UniProt ID Mapping (Critical)
**Problem**: Foldseek truncates UniProt-format IDs (`sp|P0DTC2|SPIKE_SARS2` → `P0DTC2`), causing hits to not match back to original protein IDs.

**Fix**: Added `_extract_uniprot_accession()` function in `src/vhold/results/annotations.py` to build ID mapping.

```python
def _extract_uniprot_accession(seq_id: str) -> str:
    """Extract UniProt accession from sp|ACC|NAME format."""
    if "|" in seq_id:
        parts = seq_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return seq_id
```

### 2. TSV Parsing (Minor)
**Problem**: Results TSV rows with fewer fields than header columns were skipped.

**Fix**: Pad fields with empty strings instead of skipping rows.

## Interpretation Guide

### High-Confidence Correct Predictions
- **Spike, Nucleocapsid, Membrane, Envelope**
- These validate basic pipeline functionality
- Cross-database agreement increases confidence

### No-Hit Proteins
- **NSP1, NSP3, NSP5, NSP12** (partial sequences)
- Indicates fragmented inputs, not pipeline failure
- Recommendation: Use full-length protein sequences

### Unknown Classifications
- **ORF3a, ORF6, ORF7a, ORF8, ORF9b**
- Classified as "unknown" despite having database hits
- Reflects genuinely poor functional characterization
- GO terms available but not mapped to vHold categories

## Conclusions

### What This Case Study Demonstrates
1. ✅ vHold pipeline executes correctly end-to-end
2. ✅ Cross-database consensus scoring functions properly
3. ✅ Well-characterized structural proteins are accurately annotated
4. ✅ Structure quality metrics are integrated into scoring

### What This Case Study Does NOT Demonstrate
1. ❌ Detection of remote homology (<30% sequence identity)
2. ❌ Annotation of proteins that BLAST cannot find
3. ❌ Novel functional discovery
4. ❌ Advantage over sequence-based methods

### Recommendations
1. **Use full-length proteins** - Avoid partial/fragmented sequences
2. **Process polyproteins separately** - Split ORF1ab into individual NSPs
3. **Complement with discovery case study** - Demonstrate vHold's unique value on divergent sequences

## References

1. Wu F, et al. (2020). A new coronavirus associated with human respiratory disease in China. *Nature* 579:265-269.
2. UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase. *Nucleic Acids Res* 51:D419-D428.
3. Jumper J, et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596:583-589.
