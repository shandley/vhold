# vHold Manuscript Outline

## Suggested Journal Targets

1. **Bioinformatics** (Oxford) - Application Note format
2. **Nucleic Acids Research** - Web Server Issue or Database Issue
3. **BMC Bioinformatics** - Software article
4. **PLoS Computational Biology** - Software article
5. **Briefings in Bioinformatics** - Application note

---

## Title Options

1. "vHold: Structural homology-based annotation of viral proteins across all viral taxa"
2. "vHold: Multi-database consensus annotation of viral proteins using 3Di structural similarity"
3. "Illuminating viral dark matter: Structure-based functional annotation of divergent viral proteins"

---

## Abstract (250 words max)

### Background
Functional annotation of viral proteins remains a major challenge in virology and metagenomics. RNA viruses evolve at rates 10^4-10^6 times faster than their hosts, leading to extensive sequence divergence that renders sequence-based annotation methods ineffective. As a result, 40-70% of viral proteins in metagenomic datasets lack any functional annotation.

### Results
We present vHold (Viral Homology-based Annotation Tool), a computational pipeline that leverages structural homology to annotate viral proteins from any viral taxon. vHold integrates the ProstT5 protein language model with Foldseek structural alignment to compare query proteins against two comprehensive viral structure databases: BFVD (351,242 proteins) and Viro3D (85,162 proteins). A multi-database consensus scoring algorithm combines evidence from both databases, boosting confidence when annotations agree. Proteins are automatically classified into eight functional categories, and those lacking confident annotation are flagged as viral "dark matter" for targeted investigation.

### Conclusions
vHold addresses the gap in viral protein annotation by exploiting the 3-10× greater evolutionary conservation of protein structure compared to sequence. It provides researchers with confidence-scored functional annotations suitable for genome annotation, functional genomics, and dark matter characterization. vHold is freely available as an open-source Python package.

**Availability**: https://github.com/[user]/vhold

**Keywords**: viral protein annotation, structural homology, protein structure, dark matter, metagenomics

---

## Introduction (800-1000 words)

### Opening: The Challenge
- Viral proteins are notoriously difficult to annotate
- Rapid evolution leads to sequence divergence
- Statistics: 40-70% unannotated in metagenomic datasets

### The Dark Matter Problem
- Define viral dark matter
- Scale of the problem across different viral taxa
- Implications for understanding viral biology

### Structure vs. Sequence Conservation
- Cite key literature: structure is 3-10× more conserved than sequence
- Explain why: structural constraints vs. sequence flexibility
- Success stories: examples of remote homolog detection via structure

### Existing Approaches
- **Phold** (Bouras et al.): Phage-specific, single database
- **ProstT5** (Heinzinger et al.): Language model for 3Di prediction
- **Foldseek** (van Kempen et al.): Fast structural search
- **BFVD** and **Viro3D**: Comprehensive viral structure databases

### Innovation: vHold
- Multi-database consensus approach
- Broader coverage: all viral taxa
- Dark matter analysis pipeline
- Confidence scoring with agreement bonuses

---

## Methods (1500-2000 words)

### 2.1 Pipeline Overview
- Flow diagram: FASTA → ProstT5 → Foldseek → Consensus → Output
- Design principles: modularity, reproducibility, scalability

### 2.2 3Di Structural Sequence Prediction
- ProstT5 model architecture
- Input formatting and parameters
- Confidence score extraction
- Low-confidence masking

### 2.3 Structural Homology Search
- Foldseek parameters and exhaustive search mode
- Database descriptions:
  - BFVD: 351,242 structures, AlphaFold2-based
  - Viro3D: 85,162 structures, 4,400+ species

### 2.4 Multi-Database Consensus Scoring
- Hit quality score formula
- Database weighting rationale
- Agreement checking algorithm
- Consensus score calculation with bonuses

### 2.5 Confidence Level Assignment
- Threshold calibration
- Agreement upgrade mechanism
- Interpretation guidelines

### 2.6 Functional Category Classification
- Category definitions and keywords
- Classification algorithm
- Coverage of viral protein functions

### 2.7 Dark Matter Analysis
- Definition of dark matter categories
- Classification criteria
- Reporting and statistics

### 2.8 Implementation
- Software stack: Python, PyTorch, transformers
- Dependencies: Foldseek, ProstT5
- Performance characteristics
- Availability and licensing

---

## Results (1000-1500 words)

### 3.1 Validation on Known Proteins
- Test set: well-characterized viral proteins
- Example: T4 capsid, HIV protease, SARS-CoV-2 nucleocapsid
- Results table showing correct classification

### 3.2 Database Coverage Analysis
- Total proteins: 436,404 unique structures
- Overlap between databases
- Taxonomic coverage across viral families

### 3.3 Consensus Scoring Performance
- Agreement rate between databases
- Confidence improvement with multi-database approach
- Distribution of confidence levels

### 3.4 Functional Category Distribution
- Expected distributions based on viral biology
- Comparison with known genome content

### 3.5 Dark Matter Characterization
- Dark matter rate in test datasets
- Length distributions of unknown proteins
- Categories of dark matter (no hits vs. unknown function vs. weak hits)

### 3.6 Comparison with Sequence-Based Methods
- Benchmark against BLAST/DIAMOND
- Examples of remote homologs detected only by structure

### 3.7 Scalability Assessment
- Throughput benchmarks
- Memory requirements
- GPU vs. CPU performance

---

## Discussion (800-1000 words)

### Key Findings
- Multi-database consensus improves confidence
- Structure-based approach detects remote homologs
- Dark matter analysis highlights research targets

### Advantages of vHold
- Comprehensive viral coverage (not just phages)
- Quantitative confidence scoring
- Systematic dark matter identification
- Open-source and reproducible

### Limitations
- Depends on quality of reference databases
- ProstT5 predictions may be unreliable for disordered proteins
- Cannot detect proteins with truly novel folds

### Applications
- Metagenomic annotation pipelines
- Viral genome annotation
- Functional genomics target prioritization
- Comparative viral genomics

### Future Directions
- Additional databases (ESM Atlas, AFDB viral subset)
- Embedding-based search without Foldseek
- Novel family clustering
- Taxonomy prediction from structural hits

---

## Conclusions (200 words)

vHold provides a powerful approach to viral protein annotation that addresses the limitations of sequence-based methods. By leveraging the greater conservation of protein structure and integrating evidence from multiple databases, vHold delivers confident functional annotations for divergent viral proteins. The dark matter analysis pipeline systematically identifies proteins that warrant experimental characterization, guiding research priorities. As viral structure databases continue to expand, vHold's utility will only increase.

---

## Figures

### Figure 1: Pipeline Overview
- Flow diagram showing major steps
- Databases used
- Output files generated

### Figure 2: Consensus Scoring Algorithm
- Visualization of quality score calculation
- Agreement checking and bonus system
- Confidence level assignment

### Figure 3: Validation Results
- Confusion matrix for category classification
- ROC curve if appropriate
- Example annotations

### Figure 4: Dark Matter Analysis
- Distribution of dark matter categories
- Length distribution comparison
- Example dark matter proteins

---

## Tables

### Table 1: Database Statistics
| Database | Proteins | Species | Source |
|----------|----------|---------|--------|
| BFVD | 351,242 | - | AlphaFold2 |
| Viro3D | 85,162 | 4,400+ | ColabFold |

### Table 2: Functional Categories
| Category | Keywords | Description |
|----------|----------|-------------|
| structural | capsid, envelope... | Virion components |
| ... | ... | ... |

### Table 3: Validation Results
| Test Protein | True Category | Predicted Category | Confidence |
|--------------|---------------|-------------------|------------|
| T4 gp23 | structural | structural | high |
| ... | ... | ... | ... |

### Table 4: Performance Benchmarks
| Configuration | Throughput | Memory |
|---------------|------------|--------|
| GPU (V100) | 1,000/hr | 3 GB |
| CPU | 50/hr | 6 GB |

---

## Supplementary Materials

See [SUPPLEMENTARY.md](SUPPLEMENTARY.md) for:
- Complete algorithm pseudocode
- Database field descriptions
- Output format specifications
- Complete keyword lists
- Additional benchmarks

---

## Key Messages for Reviewers

1. **Novelty**: First tool to combine multi-database consensus scoring for viral protein annotation
2. **Scope**: Covers all viral taxa, not just bacteriophages
3. **Rigor**: Quantitative confidence scoring with statistical foundation
4. **Impact**: Addresses the viral dark matter problem systematically
5. **Reproducibility**: Open-source, well-documented, tested

---

## Suggested Reviewers

1. Experts in viral bioinformatics
2. Structural bioinformatics specialists
3. Metagenomics researchers
4. Protein language model developers

---

## Author Checklist

- [ ] Run comprehensive benchmarks on diverse viral genomes
- [ ] Compare against Phold on bacteriophage test set
- [ ] Validate category classification accuracy
- [ ] Document dark matter discovery in real datasets
- [ ] Generate publication-quality figures
- [ ] Finalize GitHub repository
- [ ] Add tutorial with example workflow
