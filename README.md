# vHold

**Viral protein annotation using structural homology**

vHold annotates viral proteins by comparing predicted 3D structures against reference databases, enabling functional annotation of divergent sequences where BLAST fails. Protein structure is 3-10x more conserved than sequence during evolution, making structure-based search essential for annotating rapidly-evolving viral proteins.

Unlike [pHold](https://github.com/gbouras13/phold) which targets bacteriophages, vHold annotates proteins from **all viruses** including mammalian and eukaryotic viruses -- paramyxoviruses, filoviruses, coronaviruses, flaviviruses, and more.

## How It Works

```
Input FASTA
    |
    v
ProstT5 (sequence -> 3Di structural alphabet)
    |
    v
Foldseek (structural search against BFVD + Viro3D)
    |
    v
Multi-database consensus scoring
    |
    v
Functional classification (keywords + MLP classifier + optional LLM)
    |
    v
Output: annotations, confidence scores, dark matter report
```

1. **Structure prediction**: ProstT5 translates amino acid sequences into 3Di structural alphabet representations
2. **Structural search**: Foldseek searches predicted structures against two viral databases (BFVD and Viro3D)
3. **Consensus scoring**: Hits from both databases are weighted, scored, and compared for agreement
4. **Classification**: Proteins are assigned functional categories via Pfam/GO/keyword matching, an MLP classifier trained on structural embeddings, and optional LLM reclassification
5. **Dark matter analysis**: Proteins without confident annotations are flagged for follow-up

## Installation

### Requirements

- Python 3.10+
- [Foldseek](https://github.com/steineggerlab/foldseek)
- 4 GB disk space for databases
- GPU recommended for large datasets (CPU and Apple Silicon GPU supported)

### Install vHold

```bash
git clone https://github.com/shandley/vhold.git
cd vhold
pip install -e .
```

### Optional: LLM-based classification

For improved functional classification of ambiguous proteins using Claude:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=your-key
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
| `vhold_results.tsv` | Main annotation table with descriptions, confidence, and categories |
| `vhold_summary.json` | Run statistics and distributions |
| `vhold_dark_matter.tsv` | Unannotated proteins for follow-up investigation |

### Example Output

```
query_id    description              confidence    category          primary_evalue
protein_1   Major capsid protein     high          structural        1.2e-45
protein_2   RNA-dependent RNA pol    medium        replication       3.4e-12
protein_3   V protein                high          host_interaction  9.2e-59
```

## Usage

### Basic

```bash
vhold run -i proteins.fasta -o results/ -t 4
```

### Fast Mode

Use `--fast` for greedy decoding, which is ~2-3x faster than the default beam search. Recommended for well-characterized proteins where high-identity hits are expected. Not recommended for remote homology searches where 3Di prediction quality matters.

```bash
vhold run -i proteins.fasta -o results/ --fast
```

### LLM Classification

Use `--llm-classify` to improve functional classification of ambiguous proteins. This uses Claude to reclassify proteins where keyword matching returns "unknown" -- resolving cases like paramyxovirus V/C proteins (interferon antagonists), Ebola VP35 (immune evasion), and other proteins with generic descriptions.

Requires the `anthropic` package and an `ANTHROPIC_API_KEY` environment variable.

```bash
vhold run -i proteins.fasta -o results/ --llm-classify
```

### Two-Step Workflow

For cluster environments with separate GPU and CPU nodes:

```bash
# Step 1: GPU node - predict structures
vhold predict -i proteins.fasta -o predictions/ --device cuda

# Step 2: CPU node - search databases
vhold compare -p predictions/ -o results/ -t 32
```

### MLP Classifier

An MLP classifier trained on ProstT5 structural embeddings automatically reclassifies "unknown" proteins when a trained model is installed. This runs between keyword matching and LLM reclassification, providing a fast, offline alternative.

```bash
# Runs automatically when model is installed (default: --classify)
vhold run -i proteins.fasta -o results/

# Disable classifier
vhold run -i proteins.fasta -o results/ --no-classify

# Adjust confidence threshold (default: 0.5)
vhold run -i proteins.fasta -o results/ --classifier-confidence 0.8
```

The classifier model is trained via `scripts/train_classifier.py` and installed at `~/.vhold/models/classifier/vhold_classifier.pt`.

### All Options

```bash
vhold run -i proteins.fasta -o results/ \
    -t 8 \                        # CPU threads for Foldseek
    --device auto \               # auto, cuda, mps, or cpu
    --evalue 1e-5 \               # Stricter E-value threshold
    --sensitivity 9.5 \           # Foldseek sensitivity (1-9.5)
    --fast \                      # Greedy decoding (~2-3x faster)
    --no-classify \               # Disable MLP classifier
    --classifier-confidence 0.5 \ # MLP confidence threshold
    --llm-classify \              # LLM functional classification
    --llm-model claude-haiku-4-5-20251001  # LLM model choice
```

## Performance

### Device Selection

vHold automatically selects the best available device:

| Device | Selection | Notes |
|--------|-----------|-------|
| CUDA | Auto on Linux/Windows with NVIDIA GPU | Fastest option |
| MPS | Auto on Apple Silicon Macs | ~2x faster than CPU |
| CPU | Fallback when no GPU available | Works everywhere |

On Apple Silicon, MPS is selected automatically. A previous T5 compatibility issue was resolved in transformers v4.43+.

### Speed Benchmarks (Apple M4)

Standard mode (beam search, default):

| Protein Size | CPU | MPS | Speedup |
|--------------|-----|-----|---------|
| ~170aa | 88s | 40s | 2.2x |
| ~435aa | 170s | 95s | 1.8x |
| ~1135aa | ~132 min | ~72 min | 1.8x |

Fast mode (greedy decoding, `--fast`):

| Protein Size | MPS Standard | MPS Fast | Speedup |
|--------------|-------------|----------|---------|
| ~170aa | 40s | 19s | 2.1x |

ProstT5 has O(n^2) scaling with sequence length due to autoregressive generation. For proteins longer than ~1000aa, GPU acceleration is strongly recommended.

### Memory

- GPU (CUDA): ~3 GB VRAM
- CPU: ~6 GB RAM
- Apple Silicon (MPS): Uses unified memory

## Databases

vHold searches two viral structure databases:

| Database | Structures | Source | Description |
|----------|------------|--------|-------------|
| [BFVD](https://bfvd.foldseek.com/) | 351,242 | AlphaFold2 predictions | Comprehensive viral protein structures from UniProt TrEMBL |
| [Viro3D](https://viro3d.cvr.gla.ac.uk/) | 85,162 | Experimental + ColabFold | Curated structures from 4,400+ virus species |

Viro3D receives a 1.2x weighting bonus in consensus scoring due to its higher annotation quality.

## Confidence Levels

vHold assigns confidence based on E-value, identity, coverage, structure quality, and database agreement:

| Level | Description |
|-------|-------------|
| high | Strong E-value with multi-database agreement |
| medium | Moderate E-value or single database |
| low | Weak statistical support |
| very_low | Marginal hits, use with caution |

## Functional Categories

Proteins are classified into functional categories based on transferred annotations, Pfam domains, GO terms, SUPERFAMILY classifications, and keyword matching:

| Category | Examples |
|----------|----------|
| structural | Capsid, envelope, spike, matrix, nucleocapsid, fusion protein |
| replication | Polymerase, helicase, primase, phosphoprotein |
| protease | Proteases, peptidases |
| nuclease | Endonuclease, integrase, ligase |
| packaging | Terminase, portal, scaffold |
| regulatory | Repressor, activator, transcription factor |
| host_interaction | Interferon antagonist, immune evasion |
| entry | Membrane fusion, receptor binding |
| lysis | Holin, endolysin (bacteriophages) |
| movement | Cell-to-cell movement (plant viruses) |
| unknown | No functional annotation determined |

When a trained MLP classifier model is installed, proteins that fall through keyword matching are automatically reclassified using structural embeddings (macro F1: 0.692). With `--llm-classify`, remaining unknowns are further reclassified using an LLM with virology domain knowledge. The combination of MLP classifier + LLM provides the best accuracy for eukaryotic virus proteins with generic descriptions.

## Novelty Classification

Each hit is classified by how much value the structural search provides over sequence-based methods:

| Identity | Classification | Interpretation |
|----------|----------------|----------------|
| >95% | database_match | Same protein, different database entry |
| 70-95% | close_homolog | Related strain/variant; BLAST would find |
| 30-70% | remote_homolog | Structure-based functional transfer; BLAST marginal |
| <30% | twilight_zone | Novel structural similarity; BLAST fails |

The `remote_homolog` and `twilight_zone` categories represent annotations that sequence-based tools cannot provide.

## Dark Matter Analysis

Proteins without confident annotations are reported in the dark matter output for follow-up:

| Category | Meaning |
|----------|---------|
| no_hits | No structural homologs found -- potentially novel fold |
| unknown_function | Structural homologs exist but function is uncharacterized |
| weak_hits | Low-confidence matches only |

## Case Studies

The `case_studies/` directory contains worked examples demonstrating vHold across different use cases:

| # | Name | Proteins | Key Result |
|---|------|----------|------------|
| 1 | SARS-CoV-2 | 18 | Pipeline validation: 55.6% annotated, 100% structural protein accuracy |
| 2 | Remote Homology | 10 | 100% annotated, 91.7% of Viro3D hits in twilight zone (<20% identity) |
| 3 | Metagenomic Dark Matter | 30 | 83.3% annotated, 72% at remote homolog level (30-70% identity) |
| 4 | crAssphage ORFans | 37 | Gut phage annotation (setup phase) |
| 5 | Eukaryotic Viruses | 27 | 100% annotated across 7 mammalian virus families |

See [case_studies/README.md](case_studies/README.md) for detailed results and methodology.

## License

MIT License

## Acknowledgments

vHold builds on:

- [ProstT5](https://github.com/mheinzinger/ProstT5) - Protein language model for structure prediction
- [Foldseek](https://github.com/steineggerlab/foldseek) - Fast structural search
- [BFVD](https://bfvd.foldseek.com/) - Big Fantastic Virus Database
- [Viro3D](https://viro3d.cvr.gla.ac.uk/) - Curated viral structures
- [Anthropic Claude](https://www.anthropic.com/) - LLM-based functional classification
