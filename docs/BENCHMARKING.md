# vHold Benchmarking and Validation Plan

This document outlines the benchmarking strategy and validation studies for evaluating vHold's performance in viral protein annotation.

## Overview

### Goals

1. **Quantify annotation accuracy** - Measure sensitivity and specificity of functional annotations
2. **Compare to existing methods** - Benchmark against sequence-based and structure-based alternatives
3. **Evaluate dark matter reduction** - Measure improvement in annotating previously unannotated proteins
4. **Assess cross-database consistency** - Validate the multi-database consensus approach
5. **Characterize performance boundaries** - Identify where vHold excels and where it struggles

### Key Questions

- How much does structural homology improve annotation vs. sequence-only methods?
- What is the false positive rate for transferred annotations?
- How effective is the consensus scoring at improving annotation quality?
- What fraction of viral "dark matter" can vHold illuminate?

---

## Benchmark Datasets

### 1. Gold Standard: Experimentally Characterized Viral Proteins

**Source**: UniProt reviewed viral proteins with experimental evidence codes (ECO:0000269)

**Selection criteria**:
- Function confirmed by direct assay (not inferred)
- Diverse viral families (dsDNA, ssDNA, dsRNA, ssRNA+, ssRNA-, retroviruses)
- Range of functional categories (structural, replication, protease, etc.)
- Sequence identity <30% to database entries (to test remote homology detection)

**Target size**: 500-1,000 proteins

**Construction**:
```bash
# Query UniProt for reviewed viral proteins with experimental evidence
# Filter to remove proteins already in BFVD/Viro3D training sets
# Stratify by viral family and functional category
```

### 2. Holdout Test Set: Recent Viral Discoveries

**Source**: Proteins from viruses characterized after database creation dates

**Rationale**: Tests generalization to truly novel sequences not seen during database construction

**Target sources**:
- SARS-CoV-2 variants (post-2020)
- Recently discovered giant viruses
- Novel phages from metagenomic studies (post-2022)

**Target size**: 200-500 proteins

### 3. Synthetic Benchmark: Controlled Sequence Divergence

**Purpose**: Test annotation transfer across defined sequence identity ranges

**Construction**:
1. Select well-characterized viral proteins
2. Generate artificial homologs at 10%, 15%, 20%, 25%, 30%, 40%, 50% sequence identity
3. Use structure-aware sequence evolution (maintain fold)

**Tool**: Rose (realistic sequence evolution) or custom mutation with structure constraints

**Target**: 50 protein families × 7 identity levels = 350 test cases

### 4. Metagenomic Dark Matter Set

**Source**: Viral proteins from metagenomic assemblies with no BLAST hits

**Sources**:
- IMG/VR uncultivated viral genomes
- Serratus/Logan dark matter sequences
- RVDB uncharacterized entries

**Purpose**: Evaluate real-world dark matter annotation performance

**Target size**: 1,000-5,000 proteins

### 5. Cross-Validation Set: Known Phage Genomes

**Source**: Well-annotated bacteriophage genomes

**Examples**:
- T4, T7, Lambda (classic phages with comprehensive annotation)
- Newly characterized phages from PhagesDB

**Purpose**: Measure annotation completeness on whole genomes

**Target**: 20-50 complete phage genomes

---

## Evaluation Metrics

### Annotation Accuracy

| Metric | Definition | Target |
|--------|------------|--------|
| **Sensitivity (Recall)** | TP / (TP + FN) - fraction of true functions detected | >80% |
| **Specificity** | TN / (TN + FP) - fraction of negatives correctly identified | >95% |
| **Precision** | TP / (TP + FP) - fraction of predictions that are correct | >85% |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | >0.82 |

### Annotation Granularity

| Level | Description | Evaluation |
|-------|-------------|------------|
| **Functional category** | Correct high-level class (structural, replication, etc.) | Primary metric |
| **Specific function** | Correct detailed annotation (e.g., "major capsid protein") | Secondary metric |
| **Pfam domain** | Correct domain assignment | Tertiary metric |

### Dark Matter Reduction

```
Dark matter reduction = (DM_blast - DM_vhold) / DM_blast × 100%
```

Where:
- `DM_blast` = proteins with no BLAST annotation
- `DM_vhold` = proteins with no vHold annotation

**Target**: >50% reduction in dark matter

### Confidence Calibration

Measure whether confidence scores accurately predict annotation correctness:

| Confidence Level | Expected Accuracy |
|-----------------|-------------------|
| High (≥0.8) | >90% correct |
| Medium (0.5-0.8) | 70-90% correct |
| Low (0.3-0.5) | 50-70% correct |
| Very Low (<0.3) | <50% correct |

**Evaluation**: Brier score, calibration plots, reliability diagrams

### Cross-Database Consistency

| Metric | Definition |
|--------|------------|
| **Agreement rate** | Fraction of proteins where BFVD and Viro3D agree |
| **Conflict rate** | Fraction with critical/major conflicts |
| **Consensus improvement** | Accuracy gain from multi-database consensus vs. single database |

---

## Comparison Methods

### Sequence-Based Methods

| Tool | Type | Purpose |
|------|------|---------|
| **BLAST/DIAMOND** | Sequence alignment | Baseline sequence homology |
| **HHblits/HHpred** | Profile-profile | Sensitive sequence search |
| **HMMER/Pfam** | HMM search | Domain annotation |
| **InterProScan** | Multi-tool | Comprehensive sequence annotation |

### Structure-Based Methods

| Tool | Type | Purpose |
|------|------|---------|
| **Foldseek (AFDB)** | Structure search | Structure homology baseline |
| **Phold** | Phage annotation | Phage-specific comparison |
| **DALI** | Structure alignment | Classical structure comparison |
| **TM-align** | Structure superposition | Structure similarity metric |

### Comparison Framework

```
For each test protein:
1. Run BLAST with E < 1e-3
2. Run HHpred with E < 1e-3
3. Run vHold
4. Run Phold (if phage)
5. Compare:
   - Annotation rate (what fraction get annotation?)
   - Annotation accuracy (what fraction are correct?)
   - Consistency (do methods agree?)
```

---

## Validation Studies

### Study 1: Annotation Accuracy on Gold Standard

**Objective**: Measure precision and recall on experimentally verified proteins

**Protocol**:
1. Collect gold standard dataset (n=500-1000)
2. Remove proteins with >90% identity to database entries
3. Run vHold pipeline
4. Compare predicted function to experimental annotation
5. Calculate sensitivity, specificity, precision, F1

**Analysis**:
- Stratify by functional category
- Stratify by sequence identity to best database hit
- Stratify by viral family

### Study 2: Remote Homology Detection

**Objective**: Quantify improvement over sequence methods at low identity

**Protocol**:
1. Use synthetic benchmark with controlled identity levels
2. Run vHold, BLAST, HHpred on each identity bin
3. Plot annotation rate vs. sequence identity
4. Identify crossover point where vHold outperforms sequence methods

**Expected result**: vHold maintains >70% sensitivity at 20% identity where BLAST fails

### Study 3: Dark Matter Illumination

**Objective**: Measure reduction in unannotated proteins

**Protocol**:
1. Collect metagenomic dark matter set (n=1000-5000)
2. Confirm no BLAST hits (E > 0.01)
3. Run vHold pipeline
4. Categorize results:
   - Confident annotation (consensus score >0.5)
   - Weak annotation (score 0.3-0.5)
   - Still dark (no hits or score <0.3)
5. Manually validate subset (n=100) of confident annotations

**Metrics**:
- Dark matter reduction rate
- False positive rate among confident predictions
- Functional category distribution of new annotations

### Study 4: Whole Genome Annotation

**Objective**: Evaluate annotation completeness on complete viral genomes

**Protocol**:
1. Select well-annotated phage genomes (n=20)
2. Mask all annotations, re-annotate with vHold
3. Compare to reference annotations
4. Calculate per-genome metrics:
   - Annotation completeness
   - Accuracy of annotations
   - Novel annotations not in reference

**Comparison**: Run same protocol with BLAST, Phold

### Study 5: Consensus Scoring Validation

**Objective**: Validate that multi-database consensus improves accuracy

**Protocol**:
1. Use gold standard dataset
2. Compare annotation accuracy:
   - BFVD only
   - Viro3D only
   - Consensus (vHold default)
3. Analyze agreement cases:
   - Both agree and correct
   - Both agree but wrong
   - Disagree - which is correct?

**Expected result**: Consensus accuracy > single database accuracy by >5%

### Study 6: Confidence Calibration

**Objective**: Verify confidence scores are well-calibrated

**Protocol**:
1. Bin predictions by confidence level
2. Calculate actual accuracy in each bin
3. Generate reliability diagram
4. Calculate Expected Calibration Error (ECE)

**Target**: ECE < 0.1

### Study 7: ESM Atlas Dark Matter Analysis

**Objective**: Evaluate value of ESM Atlas for dark matter characterization

**Protocol**:
1. Take dark matter proteins with no BFVD/Viro3D hits
2. Search ESM Atlas
3. Categorize:
   - Novel folds (ESM Atlas only)
   - Metagenomic conservation (many ESM Atlas hits)
   - Still unknown
4. Assess functional implications of ESM Atlas matches

---

## Implementation Plan

### Phase 1: Dataset Construction (2 weeks)

| Task | Description | Status |
|------|-------------|--------|
| Gold standard collection | Curate experimentally verified proteins | Pending |
| Holdout set collection | Gather recent viral proteins | Pending |
| Synthetic benchmark generation | Create identity-controlled test sets | Pending |
| Dark matter collection | Assemble metagenomic unknowns | Pending |

### Phase 2: Baseline Comparisons (1 week)

| Task | Description | Status |
|------|-------------|--------|
| BLAST baseline | Run BLAST on all datasets | Pending |
| HHpred baseline | Run HHpred on selected subsets | Pending |
| Phold comparison | Run Phold on phage datasets | Pending |
| Foldseek (AFDB) baseline | Run Foldseek against AlphaFold DB | Pending |

### Phase 3: vHold Evaluation (2 weeks)

| Task | Description | Status |
|------|-------------|--------|
| Run vHold on all datasets | Full pipeline execution | Pending |
| Accuracy analysis | Calculate all metrics | Pending |
| Stratified analysis | Break down by category/family/identity | Pending |
| Confidence calibration | Assess score reliability | Pending |

### Phase 4: Analysis and Reporting (1 week)

| Task | Description | Status |
|------|-------------|--------|
| Statistical analysis | Significance tests, confidence intervals | Pending |
| Visualization | Figures for publication | Pending |
| Manuscript draft | Write results section | Pending |

---

## Scripts and Tools

### Benchmark Runner Script

```python
# benchmarks/run_benchmark.py
"""
Run complete benchmark suite for vHold evaluation.

Usage:
    python run_benchmark.py --dataset gold_standard --output results/
"""
```

### Evaluation Script

```python
# benchmarks/evaluate.py
"""
Calculate benchmark metrics from vHold output.

Metrics: sensitivity, specificity, precision, F1, dark matter reduction
"""
```

### Comparison Script

```python
# benchmarks/compare_methods.py
"""
Compare vHold to baseline methods (BLAST, HHpred, Phold).
Generate comparison tables and figures.
"""
```

---

## Expected Outcomes

### Primary Hypotheses

1. **H1**: vHold achieves >80% sensitivity on gold standard at any sequence identity
2. **H2**: vHold reduces dark matter by >50% compared to BLAST
3. **H3**: Consensus scoring improves accuracy by >5% over single database
4. **H4**: Confidence scores are well-calibrated (ECE < 0.1)

### Publication Figures

1. **Figure 1**: Annotation rate vs. sequence identity (vHold vs. BLAST vs. HHpred)
2. **Figure 2**: Dark matter reduction across viral families
3. **Figure 3**: Confidence calibration reliability diagram
4. **Figure 4**: Cross-database agreement analysis
5. **Figure 5**: Whole genome annotation completeness comparison

### Supplementary Tables

1. **Table S1**: Gold standard dataset composition
2. **Table S2**: Per-category accuracy metrics
3. **Table S3**: Per-viral-family performance
4. **Table S4**: Comparison to Phold on phage genomes
5. **Table S5**: ESM Atlas analysis of dark matter

---

## Resources Required

### Computational

| Resource | Requirement | Purpose |
|----------|-------------|---------|
| GPU | V100 or better | ProstT5 predictions |
| CPU | 32+ cores | Foldseek searches, parallel processing |
| Memory | 64+ GB | Large dataset processing |
| Storage | 500+ GB | Databases, intermediate files, results |

### Databases

| Database | Size | Purpose |
|----------|------|---------|
| BFVD | ~500 MB | Primary viral database |
| Viro3D | ~600 MB | Secondary viral database |
| ESMAtlas30 | ~50 GB | Metagenomic comparison |
| UniProt | ~100 GB | Gold standard construction |
| AlphaFold DB | ~2 TB | Baseline comparison (optional) |

### Time Estimate

| Phase | Duration |
|-------|----------|
| Dataset construction | 2 weeks |
| Baseline comparisons | 1 week |
| vHold evaluation | 2 weeks |
| Analysis and reporting | 1 week |
| **Total** | **6 weeks** |

---

## References

1. van Kempen M, et al. (2024). Fast and accurate protein structure search with Foldseek. Nature Biotechnology.
2. Bouras G, et al. (2023). Phold: Phage annotation using protein structural homology. bioRxiv.
3. Steinegger M, Söding J. (2017). MMseqs2 enables sensitive protein sequence searching. Nature Biotechnology.
4. Edgar RC, et al. (2022). Petabase-scale sequence alignment catalyses viral discovery. Nature.
5. Lin Z, et al. (2023). Evolutionary-scale prediction of atomic-level protein structure. Science.
