# vHold

**Viral protein annotation using structural homology**

vHold annotates viral proteins by searching structural databases, enabling functional annotation of divergent sequences where BLAST fails. Protein structure is 3-10x more conserved than sequence during evolution, making structure-based search essential for annotating rapidly-evolving viral proteins.

## Why vHold?

Traditional sequence-based tools (BLAST, DIAMOND) fail when sequence identity drops below ~30%. This affects 40-70% of viral proteins in metagenomic datasets. vHold solves this by:

1. Predicting protein structure from sequence using ProstT5
2. Searching viral structure databases with Foldseek
3. Transferring functional annotations from structural homologs

## Installation

### Requirements

- Python 3.10+
- 4 GB disk space for databases
- GPU recommended (CPU works but is slower)

### Install

```bash
git clone https://github.com/shandley/vhold.git
cd vhold
pip install -e .
```

### Install Foldseek

```bash
# conda
conda install -c conda-forge -c bioconda foldseek

# or download binary
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xzf foldseek-linux-avx2.tar.gz
export PATH="$(pwd)/foldseek/bin:$PATH"
```

### Download Databases

```bash
vhold install                    # Download all databases (~1.1 GB)
vhold install --no-viro3d        # BFVD only (smaller)
vhold install -d /custom/path    # Custom location
```

## Quick Start

```bash
# Annotate viral proteins
vhold run -i proteins.fasta -o results/ -t 4

# View results
cat results/vhold_results.tsv
```

### Output Files

| File | Description |
|------|-------------|
| `vhold_results.tsv` | Main annotation table |
| `vhold_summary.json` | Statistics and distributions |
| `vhold_dark_matter.tsv` | Unannotated proteins for follow-up |

## Example

Input (`proteins.fasta`):
```
>protein_1
MKTIIALSYIFCLVFADYKDDDDK...
>protein_2
MSDKIIHLTDDSFDTDVLKADGAI...
```

Output (`vhold_results.tsv`):
```
query_id    description              confidence    category      primary_evalue
protein_1   Major capsid protein     high          structural    1.2e-45
protein_2   RNA-dependent RNA pol    medium        replication   3.4e-12
```

## Databases

vHold searches two curated viral structure databases:

| Database | Structures | Description |
|----------|------------|-------------|
| BFVD | 351,242 | AlphaFold2 predictions of viral proteins |
| Viro3D | 85,162 | Curated structures from 4,400+ virus species |

## Pipeline Overview

```
Input FASTA
    |
    v
ProstT5 (sequence -> 3Di structure alphabet)
    |
    v
Foldseek (structural search against BFVD + Viro3D)
    |
    v
Consensus scoring (multi-database agreement)
    |
    v
Functional classification
    |
    v
Output: annotations + dark matter proteins
```

## Advanced Usage

### Two-Step Workflow

For cluster environments with separate GPU and CPU nodes:

```bash
# Step 1: GPU node - predict structures
vhold predict -i proteins.fasta -o predictions/ --device cuda

# Step 2: CPU node - search databases
vhold compare -p predictions/ -o results/ -t 32
```

### Custom Parameters

```bash
vhold run -i proteins.fasta -o results/ \
    --evalue 1e-5 \           # Stricter threshold
    --sensitivity 9.5 \       # Foldseek sensitivity (1-9.5)
    --threads 8 \             # CPU threads
    --device cuda             # GPU for ProstT5
```

## Performance

| Component | Hardware | Speed |
|-----------|----------|-------|
| ProstT5 | GPU (V100) | ~1,000 proteins/hour |
| ProstT5 | CPU | ~50 proteins/hour |
| Foldseek | 8 CPU cores | ~10,000 proteins/hour |

Memory: ~3 GB GPU or ~6 GB CPU for ProstT5

## Confidence Levels

vHold assigns confidence based on E-value, sequence identity, and database agreement:

| Level | Criteria |
|-------|----------|
| high | E-value < 1e-10, multi-database agreement |
| medium | E-value < 1e-5, single database or partial agreement |
| low | E-value < 1e-3, weak evidence |
| very_low | E-value > 1e-3, use with caution |

## Functional Categories

Proteins are classified into categories based on transferred annotations:

- **structural** - capsid, envelope, spike, tail
- **replication** - polymerase, helicase, primase
- **protease** - proteases, peptidases
- **nuclease** - endonuclease, integrase
- **packaging** - terminase, portal
- **regulatory** - repressor, activator
- **lysis** - holin, endolysin
- **movement** - plant virus movement proteins
- **unknown** - no functional annotation

## Dark Matter Analysis

Proteins without confident annotations are flagged as "dark matter" for follow-up:

| Category | Meaning |
|----------|---------|
| no_hits | No structural homologs found - potentially novel |
| unknown_function | Hits exist but function unknown |
| weak_hits | Low-confidence matches |

## Case Studies

See `case_studies/` for worked examples:

- **SARS-CoV-2** - Pipeline validation with well-characterized proteome
- **Remote Homology** - Annotation of divergent proteins at <30% sequence identity

## Citation

If you use vHold in your research, please cite:

```
[Citation pending publication]
```

## License

MIT License

## Acknowledgments

vHold builds on:

- [ProstT5](https://github.com/mheinzinger/ProstT5) - Protein language model for structure prediction
- [Foldseek](https://github.com/steineggerlab/foldseek) - Fast structural search
- [BFVD](https://bfvd.foldseek.com/) - Big Fantastic Virus Database
- [Viro3D](https://viro3d.cvr.gla.ac.uk/) - Curated viral structures
