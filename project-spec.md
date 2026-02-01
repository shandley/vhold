# vhold: Viral Protein Annotation Using Structural Homology

## Project Overview

vhold is a command-line tool for sensitive annotation of viral proteins from any viral taxon (not just bacteriophages) using protein structural homology. It combines the ProstT5 protein language model with Foldseek structural alignment to annotate viral proteins against comprehensive viral structure databases.

**Core insight**: Protein structure is 3-10x more conserved than sequence. For divergent viruses (especially RNA viruses and viral "dark matter"), structural homology detects relationships that sequence-based tools like BLAST/DIAMOND miss entirely.

## Architecture

The pipeline is simple:

```
protein sequences → ProstT5 → 3Di tokens → Foldseek → BFVD/Viro3D → annotations
```

### Key Dependencies

- **ProstT5**: Protein language model that translates amino acid sequences to Foldseek's 3Di structural alphabet without requiring actual structure prediction
- **Foldseek**: Ultra-fast structural alignment tool (4-5 orders of magnitude faster than traditional structural aligners)
- **BFVD**: Big Fantastic Virus Database - 351,242 viral protein structures linked to UniProt annotations
- **Viro3D**: 85,162 high-confidence viral protein structures for 4,400+ viruses with expanded functional annotations

## Prior Art to Study

**Phold** (https://github.com/gbouras13/phold) is the direct template for this project. It does exactly this workflow for bacteriophages. Key design decisions to adopt:

1. Two-step workflow: `predict` (GPU-accelerated ProstT5) and `compare` (CPU Foldseek search)
2. Confidence masking: mask 3Di residues below ProstT5 confidence threshold
3. Batch processing for large datasets
4. GenBank input/output compatibility
5. Embedding export options

Study Phold's codebase carefully before implementing.

## Functional Specification

### Command Structure

```bash
# Full pipeline (GPU recommended)
vhold run -i input.fasta -o output/ -t 8

# Two-step workflow for cluster environments
vhold predict -i input.fasta -o predictions/  # GPU step
vhold compare -i input.fasta --predictions_dir predictions/ -o output/ -t 8  # CPU step

# Database management
vhold install  # Download BFVD, Viro3D, ProstT5 models
vhold install --db bfvd  # Specific database only
```

### Input Formats

1. **FASTA** (protein sequences) - primary input
2. **GenBank** - if CDS features present, extract protein sequences
3. **GFF3 + FASTA** - extract CDS and translate

### Output Files

```
output/
├── vhold_results.tsv          # Main results table
├── vhold_annotations.gff3     # Annotations in GFF3 format
├── vhold_per_protein.tsv      # Detailed per-protein results
├── vhold_summary.json         # Summary statistics
├── embeddings/                # Optional: saved embeddings
│   ├── per_residue.h5
│   └── per_protein.h5
└── logs/
    └── vhold.log
```

### Results Table Columns

```
query_id                 # Input protein identifier
length                   # Protein length (aa)
target_id                # Best hit identifier in database
target_db                # Which database (bfvd, viro3d)
evalue                   # Foldseek E-value
bitscore                 # Foldseek bit score
query_tmscore            # TM-score normalized by query length
target_tmscore           # TM-score normalized by target length
lddt                     # Local distance difference test score
query_cov                # Query coverage
target_cov               # Target coverage
seq_identity             # Sequence identity in aligned region
prostt5_confidence       # Mean ProstT5 confidence for query
annotation               # Transferred functional annotation
annotation_source        # Source of annotation (uniprot, pfam, interpro)
taxonomy                 # Viral taxonomy of best hit
confidence_level         # high/medium/low based on thresholds
```

### Confidence Levels

Calibrate thresholds based on Phold's approach:

- **High**: E-value < 1e-10, TM-score > 0.5, coverage > 70%
- **Medium**: E-value < 1e-3, TM-score > 0.4, coverage > 50%
- **Low**: E-value < 0.01, any other passing hit

### Command-Line Options

```
Common options:
  -i, --input PATH           Input file (FASTA, GenBank, or GFF3+FASTA)
  -o, --output PATH          Output directory
  -t, --threads INT          Number of threads [default: 1]
  -p, --prefix TEXT          Prefix for output files [default: vhold]
  -d, --database PATH        Path to vhold databases
  -f, --force                Overwrite output directory

Prediction options:
  --batch_size INT           ProstT5 batch size [default: 1]
  --cpu                      Force CPU-only mode
  --gpu                      Enable GPU for Foldseek (if available)
  --mask_threshold FLOAT     Mask 3Di residues below this ProstT5 confidence [default: 25]

Search options:
  --evalue FLOAT             E-value threshold [default: 0.001]
  --max_seqs INT             Max target sequences per query [default: 10000]
  --sensitivity FLOAT        Foldseek sensitivity [default: 9.5]
  --ultra_sensitive          Disable prefiltering for maximum sensitivity (slow)

Database options:
  --db_bfvd                  Search BFVD only
  --db_viro3d                Search Viro3D only
  --db_all                   Search all databases [default]
  --custom_db PATH           Additional custom Foldseek database

Output options:
  --save_embeddings          Save ProstT5 embeddings
  --save_3di                 Save predicted 3Di sequences
  --min_confidence TEXT      Minimum confidence level to report [default: low]
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Project setup**
   - Python package structure with pyproject.toml
   - CLI using Click or Typer
   - Logging configuration
   - Config file support (YAML)

2. **Database management**
   - Download scripts for BFVD, Viro3D
   - Database versioning and integrity checks
   - Annotation mapping files (UniProt ID → function, taxonomy)

3. **Input parsing**
   - FASTA parser (BioPython or custom)
   - GenBank parser with CDS extraction
   - GFF3 + FASTA handling
   - Input validation

### Phase 2: Prediction Module

1. **ProstT5 integration**
   - Load model from HuggingFace (Rostlab/ProstT5)
   - Batch processing with configurable batch size
   - GPU/CPU detection and handling
   - Half-precision (fp16) for GPU efficiency
   - Confidence score extraction per residue
   - 3Di sequence generation

2. **Confidence masking**
   - Mask low-confidence 3Di positions
   - Configurable threshold

3. **Embedding export**
   - Per-residue embeddings to HDF5
   - Per-protein (mean-pooled) embeddings

### Phase 3: Comparison Module

1. **Foldseek integration**
   - Subprocess wrapper for Foldseek
   - Database selection logic
   - Parameter tuning based on sensitivity requirements
   - GPU support detection

2. **Result parsing**
   - Parse Foldseek tabular output
   - Handle multiple databases
   - Merge and deduplicate results

3. **Annotation transfer**
   - Map target IDs to UniProt annotations
   - Retrieve functional annotations (protein names, GO terms, Pfam)
   - Retrieve taxonomy information
   - Confidence level assignment

### Phase 4: Output Generation

1. **Results formatting**
   - TSV output with all columns
   - GFF3 output for genome browsers
   - JSON summary statistics

2. **Reporting**
   - Summary statistics (% annotated, confidence distribution)
   - Per-database hit statistics
   - Taxonomy distribution of hits

### Phase 5: Testing and Documentation

1. **Test suite**
   - Unit tests for parsers
   - Integration tests with small test datasets
   - Regression tests against known annotations

2. **Documentation**
   - README with quick start
   - Full documentation (MkDocs or similar)
   - Example workflows

## Technical Details

### ProstT5 Usage

```python
from transformers import T5Tokenizer, T5EncoderModel
import torch

# Load model
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
model = T5EncoderModel.from_pretrained('Rostlab/ProstT5')

# For GPU with half precision
if torch.cuda.is_available():
    model = model.half().cuda()

# Prepare sequence (add prefix for AA→3Di translation)
sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
sequence_spaced = " ".join(list(sequence))
input_seq = "<AA2fold>" + " " + sequence_spaced

# Tokenize and get embeddings
inputs = tokenizer(input_seq, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# For 3Di prediction, use the full encoder-decoder or CNN head
# See ProstT5 documentation for details
```

### Foldseek Command

```bash
foldseek easy-search \
    query_3di.fasta \
    /path/to/bfvd \
    results.tsv \
    tmp/ \
    --format-output "query,target,evalue,bits,qcov,tcov,qtmscore,ttmscore,lddt,fident" \
    -e 0.001 \
    --max-seqs 10000 \
    -s 9.5
```

### Database Locations

```
~/.vhold/
├── databases/
│   ├── bfvd/
│   │   ├── bfvd_foldseekdb/
│   │   └── bfvd_metadata.tsv
│   ├── viro3d/
│   │   ├── viro3d_foldseekdb/
│   │   └── viro3d_annotations.tsv
│   └── version.json
├── models/
│   └── prostt5/  # Cached HuggingFace model
└── config.yaml
```

## Database Annotation Mapping

### BFVD → UniProt

BFVD entries are UniRef30 representatives with UniProt accessions. Annotation retrieval:

1. Parse BFVD metadata TSV for UniProt accessions
2. Batch query UniProt API or use pre-downloaded mapping file
3. Extract: protein name, gene name, organism, taxonomy, GO terms, Pfam domains, InterPro

### Viro3D

Viro3D provides direct annotations including:
- Pfam domains
- Gene Ontology terms
- Viral taxonomy
- Host information

## Differentiation from Phold

| Feature | Phold | vhold |
|---------|-------|-------|
| Target viruses | Bacteriophages | All viruses |
| Primary database | Phold DB (1.36M phage proteins) | BFVD + Viro3D |
| Annotation source | PHROGs, VFDB, CARD, etc. | UniProt, Pfam, InterPro |
| Input requirement | Pharokka GenBank | Any FASTA/GenBank |
| Functional categories | Phage-specific (PHROGs) | General viral functions |

## Future Enhancements

1. **Novel family discovery**: Cluster unannotated proteins by 3Di similarity
2. **Taxonomy prediction**: Infer viral taxonomy from structural hits
3. **Multi-database search**: Add ESM Atlas, AFDB viral subset
4. **Embedding-based search**: Direct embedding similarity without Foldseek
5. **Batch mode**: Process many genomes efficiently
6. **Snakemake/Nextflow integration**: Workflow manager support

## Resources

- Phold: https://github.com/gbouras13/phold
- ProstT5: https://github.com/mheinzinger/ProstT5
- Foldseek: https://github.com/steineggerlab/foldseek
- BFVD: https://bfvd.foldseek.com/
- Viro3D: https://viro3d.cvr.gla.ac.uk/

## Success Criteria

1. Annotates >50% of proteins on average viral genome (matching Phold's phage performance)
2. Processes a typical viral genome (<100 proteins) in <5 minutes on GPU
3. Handles metagenomic datasets (>100k proteins) efficiently
4. Clear confidence levels guide user interpretation
5. Output compatible with downstream analysis tools

## Getting Started

```bash
# Clone and install
git clone https://github.com/[user]/vhold.git
cd vhold
pip install -e .

# Install databases
vhold install

# Run on test data
vhold run -i test/test_viral_proteins.fasta -o test_output/ -t 4

# Check results
cat test_output/vhold_results.tsv
```
