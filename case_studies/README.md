# vHold Case Studies

This directory contains worked examples demonstrating vHold's annotation capabilities.

## Case Study Overview

| # | Name | Purpose | Proteins | Status |
|---|------|---------|----------|--------|
| 1 | [SARS-CoV-2](sars_cov_2/) | Pipeline validation | 18 | Complete |
| 2 | [Remote Homology](remote_homology/) | Demonstrate annotation at low identity | 10 | Complete |
| 3 | [Metagenomic Dark Matter](metagenomic_dark_matter/) | Annotate truly unknown proteins | 30 | Complete |
| 4 | [crAssphage ORFans](crass_phage_orfans/) | Annotate unknown gut phage proteins | 37 | Setup |
| 5 | [Eukaryotic Viruses](eukaryotic_viruses/) | Mammalian virus annotation (vHold focus) | 29 | Running |

## Key Learnings

### Novelty Classification

Not all hits are equal. vHold now classifies hits by "novelty" - how much value the structural search provides:

| Identity | Classification | Interpretation | BLAST Status |
|----------|----------------|----------------|--------------|
| >95% | `database_match` | Same protein, different DB entry | Would find |
| 70-95% | `close_homolog` | Related strain/variant | Would find |
| 30-70% | `remote_homolog` | Structure-based functional transfer | Marginal |
| <30% | `twilight_zone` | Novel structural similarity | **Fails** |

**Real value**: `remote_homolog` and `twilight_zone` hits represent annotations that BLAST cannot provide.

### Database Composition

BFVD is built from **TrEMBL** (computationally predicted) UniProt entries, not Swiss-Prot (manually reviewed). This means:

- Well-characterized proteins (Swiss-Prot) may not be in BFVD directly
- Hits at 97-99% identity often represent the same protein in different database entries
- This is a database curation artifact, not a pipeline limitation

## Case Study 1: SARS-CoV-2 Proteome Validation

**Purpose**: Validate that the vHold pipeline functions correctly using well-characterized proteins.

**Key Results**:
- 10/18 proteins annotated (55.6%)
- 100% accuracy on structural proteins (S, N, M, E)
- 7/10 proteins with cross-database consensus

**Limitations**: SARS-CoV-2 is too well-studied to demonstrate vHold's remote homology detection capabilities. NSP fragments had no hits due to partial sequences.

[Full documentation](sars_cov_2/README.md)

## Case Study 2: Remote Homology Detection

**Purpose**: Demonstrate vHold's ability to annotate divergent viral proteins from RNA phages, dsRNA viruses, and plant viruses.

**Key Results**:
- 10/10 proteins annotated (100%)
- 7/10 functional categories correct (70% → 100% after keyword fixes)
- 3 proteins with Viro3D hits at 11-16% identity (twilight zone)
- 91.7% of Viro3D hits at <20% identity

**Key Finding**: Phi6 RdRp hit PhiYY RdRp at 48.3% identity - a true remote homolog from a different phage species in the same family. This is where vHold adds real value.

**Identity Distribution**:
```
BFVD (938 hits):   77.6% twilight | 7.8% remote | 10.7% moderate | 3.9% easy
Viro3D (540 hits): 91.7% twilight | 8.1% remote | 0% moderate    | 0.2% easy
```

[Full documentation](remote_homology/README.md)

## Case Study 3: Metagenomic Dark Matter

**Purpose**: Demonstrate vHold's ability to annotate truly novel proteins from metagenomes that have no close sequence homologs.

**Data Source**: palmdb - 513K RdRp "palmprint" sequences discovered from petabase-scale metagenomic mining (Edgar et al., Nature 2022).

**Key Results**:
- 25/30 proteins annotated (83.3%)
- 21/30 correctly classified as "replication" (70%)
- 84% of annotated proteins have RdRp keywords
- 72% of hits at remote_homolog level (30-70% identity)
- Mean identity: 53% - well beyond BLAST sensitivity

**Key Finding**: vHold successfully transfers RdRp function from structural homologs to novel metagenomic sequences. The 5 unannotated sequences represent true "dark matter" - proteins too divergent even for structural search.

[Full documentation](metagenomic_dark_matter/README.md)

## Running Case Studies

```bash
# Case Study 1: SARS-CoV-2
cd case_studies/sars_cov_2
python run_case_study.py -o results/ --device cuda

# Case Study 2: Remote Homology
cd case_studies/remote_homology
vhold run -i divergent_proteins.fasta -o results/ -t 4
python analyze_remote_homology.py results/
python compare_ground_truth.py results/
```

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `remote_homology/analyze_identity.py` | Identity stratification analysis |
| `remote_homology/analyze_remote_homology.py` | Remote homology detection analysis |
| `remote_homology/compare_ground_truth.py` | Ground truth accuracy comparison |

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
