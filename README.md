# vHold: Viral Protein Annotation Using Structural Homology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**vHold** is a computational pipeline for functional annotation of viral proteins using structural homology. It addresses the challenge of annotating divergent viral proteins where sequence-based methods fail by leveraging the greater evolutionary conservation of protein tertiary structure.

## Key Features

- **Structure-based annotation**: Detects remote homology relationships missed by sequence-based methods
- **Multi-database consensus**: Integrates results from BFVD (351K proteins) and Viro3D (85K proteins)
- **Confidence scoring**: Quantitative assessment of annotation reliability with agreement bonuses
- **Functional classification**: Automatic categorization into 8 viral protein functional classes
- **Dark matter analysis**: Systematic identification and characterization of unknown proteins
- **GPU acceleration**: Fast ProstT5 3Di prediction on CUDA/MPS devices

## Background

### The Problem

Viral proteins evolve rapidly, with RNA viruses showing mutation rates 10^4-10^6 times higher than their hosts. This leads to extensive sequence divergence that renders traditional annotation methods ineffective:

- **40-70%** of viral proteins in metagenomic datasets lack functional annotation
- Sequence identity often falls below **20%** even for functionally related proteins
- BLAST/DIAMOND miss these relationships entirely

### The Solution

Protein structure is **3-10× more conserved** than sequence during evolution. vHold exploits this by:

1. Predicting 3Di structural sequences from amino acid sequences using ProstT5
2. Searching structural databases with Foldseek
3. Building consensus annotations from multiple databases
4. Classifying proteins into functional categories
5. Identifying "dark matter" proteins for further investigation

## Installation

### Requirements

- Python 3.10+
- Foldseek
- ~4 GB disk space for databases
- GPU recommended for optimal ProstT5 performance

### Install vHold

```bash
git clone https://github.com/[user]/vhold.git
cd vhold
pip install -e .
```

### Install Foldseek

```bash
conda install -c conda-forge -c bioconda foldseek
```

### Download Databases

```bash
vhold install                    # Install all databases (~1.1 GB)
vhold install --no-viro3d        # Install BFVD only
vhold install -d /custom/path    # Custom location
```

## Quick Start

```bash
# Run the complete annotation pipeline
vhold run -i proteins.fasta -o results/ -t 4

# View results
cat results/vhold_results.tsv
cat results/vhold_summary.json
```

## Pipeline Overview

```
Input FASTA → ProstT5 Prediction → Confidence Masking → Foldseek Search →
    → Multi-Database Consensus → Functional Classification → Dark Matter Analysis → Output
```

### 1. 3Di Prediction

ProstT5 translates amino acid sequences to Foldseek's 3Di structural alphabet:

```bash
vhold predict -i proteins.fasta -o predictions/
```

### 2. Structural Search

Foldseek searches against BFVD and Viro3D databases:

```bash
vhold compare -p predictions/ -o search_results/
```

### 3. Full Pipeline

Combined prediction and search with consensus scoring:

```bash
vhold run -i proteins.fasta -o results/ -t 4
```

## Output Files

| File | Description |
|------|-------------|
| `vhold_results.tsv` | Main annotation results with consensus scores |
| `vhold_summary.json` | Summary statistics and distributions |
| `vhold_dark_matter.tsv` | Proteins with unknown/weak annotations |
| `predictions/` | 3Di sequence predictions |
| `foldseek/` | Raw Foldseek search results |

### Results Table Columns

| Column | Description |
|--------|-------------|
| query_id | Input protein identifier |
| description | Transferred functional annotation |
| confidence_level | high / medium / low / very_low / none |
| consensus_score | Quality score (0-1) |
| agreement | agree / partial / disagree / single / none |
| functional_category | Assigned functional class |
| primary_source | Database providing primary hit |
| primary_evalue | E-value of best hit |
| bfvd_hits / viro3d_hits | Hit counts per database |

## Consensus Scoring

vHold integrates hits from multiple databases using a weighted consensus algorithm:

### Hit Quality Score

```
quality = 0.5 × evalue_score + 0.3 × identity + 0.2 × coverage
```

### Database Weights

| Database | Weight | Rationale |
|----------|--------|-----------|
| Viro3D | 1.2 | Curated functional annotations |
| BFVD | 1.0 | Broader coverage |

### Agreement Bonuses

| Agreement | Bonus | Condition |
|-----------|-------|-----------|
| Agree | 1.3× | ≥50% keyword overlap |
| Partial | 1.15× | ≥20% keyword overlap |
| Disagree | 1.0× | <20% overlap |

## Functional Categories

Proteins are automatically classified:

| Category | Examples |
|----------|----------|
| **structural** | capsid, envelope, spike, nucleocapsid |
| **replication** | polymerase, helicase, RdRp |
| **protease** | 3CL protease, MPro |
| **nuclease** | endonuclease, integrase |
| **packaging** | terminase, scaffold |
| **regulatory** | repressor, activator |
| **movement** | movement protein |
| **lysis** | holin, endolysin |
| **unknown** | hypothetical, uncharacterized |

## Dark Matter Analysis

Proteins lacking confident functional annotation are flagged as viral "dark matter":

| Category | Criteria | Significance |
|----------|----------|--------------|
| **No hits** | No structural homologs detected | Truly novel proteins |
| **Unknown function** | Hits but uncharacterized | Conserved structure, unknown role |
| **Weak hits** | E-value >1e-5 or identity <30% | Ambiguous homology |

Dark matter proteins are reported separately for targeted investigation.

## Databases

### BFVD (Big Fantastic Virus Database)

- **351,242** viral protein structures
- AlphaFold2-predicted with quality metrics
- UniProt-linked annotations

### Viro3D

- **85,162** high-confidence structures
- **4,400+** virus species
- Curated Pfam annotations

## Advanced Usage

### Custom Parameters

```bash
vhold run -i proteins.fasta -o results/ \
    --evalue 1e-5 \           # Stricter E-value threshold
    --sensitivity 9.5 \       # Foldseek sensitivity
    --confidence 0.8 \        # ProstT5 confidence mask threshold
    --threads 8 \             # CPU threads
    --device cuda             # GPU device
```

### Two-Step Workflow

For cluster environments with separate GPU/CPU nodes:

```bash
# Step 1: GPU node
vhold predict -i proteins.fasta -o predictions/ --device cuda

# Step 2: CPU node
vhold compare -p predictions/ -o results/ -t 32
```

## Performance

| Component | Hardware | Throughput |
|-----------|----------|------------|
| ProstT5 prediction | GPU (V100) | ~1,000 proteins/hour |
| ProstT5 prediction | CPU | ~50 proteins/hour |
| Foldseek search | CPU (8 cores) | ~10,000 proteins/hour |

**Memory**: ~3 GB GPU (FP16) or ~6 GB CPU (FP32) for ProstT5

## Comparison with Related Tools

| Feature | vHold | Phold |
|---------|-------|-------|
| Target viruses | All viruses | Bacteriophages |
| Databases | BFVD + Viro3D | Phold DB |
| Multi-DB consensus | Yes | No |
| Dark matter analysis | Yes | No |
| Functional categories | 8 general | PHROGs-specific |

## Citation

If you use vHold in your research, please cite:

```
[Citation pending publication]
```

## Documentation

- [Methods and Algorithms](docs/METHODS.md) - Detailed methodology
- [Supplementary Materials](docs/SUPPLEMENTARY.md) - Technical specifications

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

vHold builds upon:

- [ProstT5](https://github.com/mheinzinger/ProstT5) - Protein language model
- [Foldseek](https://github.com/steineggerlab/foldseek) - Structural search
- [Phold](https://github.com/gbouras13/phold) - Phage annotation (inspiration)
- [BFVD](https://bfvd.foldseek.com/) - Viral structure database
- [Viro3D](https://viro3d.cvr.gla.ac.uk/) - Viral structure database
