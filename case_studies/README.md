# vHold Case Studies

This directory contains worked examples demonstrating vHold's annotation capabilities.

## Case Study Overview

| # | Name | Purpose | Proteins | Status |
|---|------|---------|----------|--------|
| 1 | [SARS-CoV-2](sars_cov_2/) | Pipeline validation | 18 | ✅ Complete |
| 2 | [Remote Homology](remote_homology/) | Demonstrate annotation at <30% identity | TBD | 🔨 In Progress |

## Case Study 1: SARS-CoV-2 Proteome Validation

**Purpose**: Validate that the vHold pipeline functions correctly using well-characterized proteins.

**Key Results**:
- 10/18 proteins annotated (55.6%)
- 100% accuracy on structural proteins (S, N, M, E)
- 7/10 proteins with cross-database consensus

**Limitations**: SARS-CoV-2 is too well-studied to demonstrate vHold's remote homology detection capabilities. This serves as a validation benchmark, not a discovery demonstration.

[Full documentation →](sars_cov_2/README.md)

## Case Study 2: Remote Homology Discovery (In Progress)

**Purpose**: Demonstrate vHold's ability to annotate divergent viral proteins where sequence-based methods (BLAST/DIAMOND) fail.

**Key Insight**: Foldseek already reports sequence identity for every hit. We can stratify results by identity bins without running BLAST:

| Identity Bin | Range | BLAST Status |
|--------------|-------|--------------|
| easy | >50% | Works fine |
| moderate | 30-50% | Marginal |
| remote | 20-30% | Fails |
| twilight | <20% | Structure only |

**Approach**:
- Analyze any vHold run by identity stratification
- Count successful annotations in each bin
- Proteins annotated at <30% identity demonstrate vHold's unique value

**Analysis Script**: `remote_homology/analyze_identity.py`

[Full documentation →](remote_homology/README.md)

## Running Case Studies

```bash
# Case Study 1: SARS-CoV-2
cd case_studies/sars_cov_2
python run_case_study.py -o results/ --device cuda

# View results
cat results/case_study_report.md
```

## Creating New Case Studies

Each case study should include:

1. **Input FASTA** - Protein sequences to annotate
2. **Ground truth** - Known functions for evaluation (JSON format)
3. **Run script** - Automated execution and evaluation
4. **Documentation** - README with methods, results, and interpretation

Template structure:
```
case_studies/new_study/
├── README.md
├── input.fasta
├── ground_truth.json
├── run_case_study.py
└── results/
```
